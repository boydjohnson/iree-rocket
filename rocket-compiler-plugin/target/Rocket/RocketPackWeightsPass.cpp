// Copyright 2026
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Packs each Rocket convolution's and matmul's constant filter into the
// CNA's blocked coefficient stream at compile time, so the driver binds it
// directly and never runs its packer (COMPILER_ROADMAP.md 6.3).
//
// By the flow phase a model's filter is a `util.global.load immutable` of an
// initialized global -- IREE's const-eval has already folded the channels-
// last transpose, the f16 demote and the shim's own reshapes into it -- and
// the dispatch reads it through a `flow.tensor.reshape` to the shim's
// dynamic operand type. This pass follows that chain back to the global,
// packs its bytes through `rocket_pack_conv_weights` (the *same*
// `rocket_core::weights::WeightPlan` the driver packs with at dispatch time,
// so the bytes are identical by construction), stores them in a new i8
// global, and points the dispatch at it. The dispatch is then retargeted at
// a clone of its executable whose target config carries
// `weights_packed = true`, which RocketTarget serializes as
// `Conv2DDef.weights_packed` / `MatmulDef.weights_packed`: the flag is an
// executable property, so no push constant, pipeline layout or shim changes,
// and an executable without the flag packs at runtime exactly as before. A
// dispatch whose filter is not a constant (a function argument, another
// dispatch's result), whose dimensions are not all constants, or whose
// bytes the packer refuses is left alone and counted in the remark.
//
// Runs at the flow phase after rocket-assign-layout; rocket-compiler
// drives it by name the same way.

#include <optional>
#include <string>
#include <vector>

#include "RocketDispatchQuery.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "rocket_plan.h"

namespace mlir::iree_compiler::IREE::HAL {
namespace {

using namespace rocket_query;

constexpr StringLiteral kWeightsPackedKey = "weights_packed";

size_t elementBytes(uint32_t precision) { return inputElementBytes(precision); }

// Everything the packer needs about one dispatch, read from the target
// config and the constant push constants. Either a conv or a matmul
// descriptor is filled, never both.
struct PackRequest {
  bool matmul = false;
  rocket_plan_conv_desc_t conv{};
  rocket_plan_matmul_desc_t matmul_desc{};
  size_t elementBytes = 0;
  // Argument index of the weights binding.
  unsigned weightsIndex = 0;
  // A memoization key: the same global packed for the same geometry is the
  // same bytes.
  std::string key;
};

std::optional<PackRequest> readRequest(DictionaryAttr config,
                                       IREE::Flow::DispatchOp dispatchOp) {
  auto kernel = dyn_cast_or_null<StringAttr>(config.get("kernel"));
  if (!kernel) {
    return std::nullopt;
  }
  std::optional<uint32_t> precision = precisionCode(config);
  if (!precision) {
    return std::nullopt;
  }
  size_t constants = constantCount(config);
  OperandRange arguments = dispatchOp.getArguments();
  if (arguments.size() < constants + 2) {
    return std::nullopt;
  }
  PackRequest request;
  request.elementBytes = elementBytes(*precision);
  request.weightsIndex = static_cast<unsigned>(constants + 1);
  rocket_plan_quantization_t quantization{};
  quantization.input_scale = 1.0f;
  quantization.weights_scale = 1.0f;
  quantization.output_scale = 1.0f;
  if (auto zeroPoint = dyn_cast_or_null<IntegerAttr>(config.get("weights_zero_point"))) {
    quantization.weight_zero_point = static_cast<int32_t>(zeroPoint.getInt());
  }
  llvm::raw_string_ostream key(request.key);
  if (kernel.getValue() == "matmul") {
    auto m = dimension(config, arguments, "m");
    auto k = dimension(config, arguments, "k");
    auto n = dimension(config, arguments, "n");
    if (!m || !k || !n) {
      return std::nullopt;
    }
    request.matmul = true;
    request.matmul_desc.struct_size = sizeof(request.matmul_desc);
    request.matmul_desc.precision = *precision;
    request.matmul_desc.m = static_cast<uint64_t>(*m);
    request.matmul_desc.k = static_cast<uint64_t>(*k);
    request.matmul_desc.n = static_cast<uint64_t>(*n);
    request.matmul_desc.activation = ROCKET_PLAN_ACTIVATION_NONE;
    request.matmul_desc.quantization = quantization;
    key << "matmul|" << *precision << "|" << *m << "x" << *k << "x" << *n;
    return request;
  }
  if (kernel.getValue() != "conv2d") {
    return std::nullopt;
  }
  auto width = dimension(config, arguments, "input_width");
  auto height = dimension(config, arguments, "input_height");
  auto cin = dimension(config, arguments, "input_channels");
  auto cout = dimension(config, arguments, "output_channels");
  auto kw = dimension(config, arguments, "weights_width");
  auto kh = dimension(config, arguments, "weights_height");
  auto stride = dimension(config, arguments, "stride");
  if (!width || !height || !cin || !cout || !kw || !kh || !stride) {
    return std::nullopt;
  }
  bool depthwise = false;
  if (auto attr = dyn_cast_or_null<BoolAttr>(config.get("depthwise"))) {
    depthwise = attr.getValue();
  }
  request.conv.struct_size = sizeof(request.conv);
  request.conv.precision = *precision;
  request.conv.width = static_cast<uint64_t>(*width);
  request.conv.height = static_cast<uint64_t>(*height);
  request.conv.in_channels = static_cast<uint64_t>(*cin);
  request.conv.out_channels = static_cast<uint64_t>(*cout);
  request.conv.stride = static_cast<uint64_t>(*stride);
  request.conv.kernel_height = static_cast<uint64_t>(*kh);
  request.conv.kernel_width = static_cast<uint64_t>(*kw);
  // Padding does not enter the coefficient layout; the planner's default
  // keeps the descriptor valid for any kernel.
  request.conv.pad_top = -1;
  request.conv.pad_left = -1;
  request.conv.activation = ROCKET_PLAN_ACTIVATION_NONE;
  request.conv.depthwise = depthwise ? 1 : 0;
  request.conv.quantization = quantization;
  key << "conv2d|" << *precision << "|" << (depthwise ? "dw|" : "") << *width
      << "x" << *height << "x" << *cin << "->" << *cout << "|k" << *kh << "x"
      << *kw << "|s" << *stride << "|zp" << quantization.weight_zero_point;
  return request;
}

// The raw bytes of an initialized global, splats expanded.
bool rawBytes(Attribute initialValue, std::vector<char> &storage,
              ArrayRef<char> &bytes) {
  if (auto dense = dyn_cast<DenseElementsAttr>(initialValue)) {
    if (dense.isSplat()) {
      ArrayRef<char> one = dense.getRawData();
      int64_t count = dense.getNumElements();
      if (one.empty() || count <= 0) {
        return false;
      }
      storage.reserve(one.size() * static_cast<size_t>(count));
      for (int64_t i = 0; i < count; ++i) {
        storage.insert(storage.end(), one.begin(), one.end());
      }
      bytes = storage;
      return true;
    }
    bytes = dense.getRawData();
    return true;
  }
  if (auto resource = dyn_cast<DenseResourceElementsAttr>(initialValue)) {
    AsmResourceBlob *blob = resource.getRawHandle().getBlob();
    if (!blob) {
      return false;
    }
    bytes = blob->getData();
    return true;
  }
  return false;
}

struct RocketPackWeightsPass
    : public PassWrapper<RocketPackWeightsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RocketPackWeightsPass)

  StringRef getArgument() const final { return "rocket-pack-weights"; }
  StringRef getDescription() const final {
    return "Packs each Rocket convolution's and matmul's constant filter "
           "into the CNA's coefficient layout at compile time and marks the "
           "dispatch's executable weights_packed.";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, IREE::Util::UtilDialect>();
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    if (rocket_plan_abi_version() != ROCKET_PLAN_ABI_VERSION) {
      module.emitError() << "rocket-pack-weights: rocket-plan-ffi reports ABI "
                            "version "
                         << rocket_plan_abi_version()
                         << " but this plugin was built against "
                         << ROCKET_PLAN_ABI_VERSION;
      return signalPassFailure();
    }
    SymbolTable symbolTable(module);
    // One packed clone per source executable, and one packed global per
    // (global, geometry): a filter shared by two dispatches is packed once.
    llvm::DenseMap<Operation *, IREE::HAL::ExecutableOp> packedExecutables;
    llvm::StringMap<IREE::Util::GlobalOp> packedGlobals;
    unsigned packed = 0;
    unsigned nonConstant = 0;
    unsigned refused = 0;
    uint64_t denseBytes = 0;
    uint64_t packedBytes = 0;

    SmallVector<IREE::Flow::DispatchOp> dispatches;
    module.walk([&](IREE::Flow::DispatchOp dispatchOp) { dispatches.push_back(dispatchOp); });
    for (IREE::Flow::DispatchOp dispatchOp : dispatches) {
      std::optional<DispatchTarget> target = rocketTarget(dispatchOp);
      if (!target) {
        continue;
      }
      DictionaryAttr config = target->target.getConfiguration();
      if (!config) {
        continue;
      }
      if (auto already = dyn_cast_or_null<BoolAttr>(config.get(kWeightsPackedKey));
          already && already.getValue()) {
        continue;
      }
      std::optional<PackRequest> request = readRequest(config, dispatchOp);
      if (!request) {
        // Not a conv/matmul, or a dimension is not a constant: the driver
        // packs at dispatch time as before.
        auto kernel = dyn_cast_or_null<StringAttr>(config.get("kernel"));
        if (kernel && (kernel.getValue() == "conv2d" || kernel.getValue() == "matmul")) {
          ++nonConstant;
        }
        continue;
      }

      // The weights binding, back through the shim's reshapes to its global.
      OperandRange arguments = dispatchOp.getArguments();
      Value weights = arguments[request->weightsIndex];
      SmallVector<IREE::Flow::TensorReshapeOp> reshapes;
      Value source = weights;
      while (auto reshape = source.getDefiningOp<IREE::Flow::TensorReshapeOp>()) {
        reshapes.push_back(reshape);
        source = reshape.getSource();
      }
      auto loadOp = source.getDefiningOp<IREE::Util::GlobalLoadOp>();
      if (!loadOp) {
        ++nonConstant;
        continue;
      }
      auto globalOp = symbolTable.lookup<IREE::Util::GlobalOp>(loadOp.getGlobal());
      if (!globalOp || globalOp.getIsMutable() || !globalOp.getInitialValue()) {
        ++nonConstant;
        continue;
      }
      auto sourceType = dyn_cast<RankedTensorType>(source.getType());
      if (!sourceType || !sourceType.hasStaticShape() ||
          !sourceType.getElementType().isIntOrFloat() ||
          sourceType.getElementType().getIntOrFloatBitWidth() !=
              request->elementBytes * 8) {
        ++refused;
        continue;
      }

      std::string globalKey = (globalOp.getSymName() + "|" + request->key).str();
      IREE::Util::GlobalOp packedGlobal = packedGlobals.lookup(globalKey);
      if (!packedGlobal) {
        std::vector<char> storage;
        ArrayRef<char> bytes;
        if (!rawBytes(*globalOp.getInitialValue(), storage, bytes)) {
          ++refused;
          continue;
        }
        char message[256] = {0};
        size_t packedLength = 0;
        std::vector<uint8_t> output;
        uint32_t status;
        auto pack = [&](uint8_t *out, size_t capacity) {
          const auto *dense = reinterpret_cast<const uint8_t *>(bytes.data());
          return request->matmul
                     ? rocket_pack_matmul_weights(&request->matmul_desc, dense,
                                                  bytes.size(), out, capacity,
                                                  &packedLength, message,
                                                  sizeof(message))
                     : rocket_pack_conv_weights(&request->conv, dense, bytes.size(),
                                                out, capacity, &packedLength,
                                                message, sizeof(message));
        };
        status = pack(nullptr, 0);
        if (status == ROCKET_PLAN_OK) {
          output.resize(packedLength);
          status = pack(output.data(), output.size());
        }
        if (status != ROCKET_PLAN_OK) {
          dispatchOp.emitWarning()
              << "rocket-pack-weights: leaving @" << globalOp.getSymName()
              << " to the runtime packer: " << rocket_plan_status_name(status)
              << ": " << message;
          ++refused;
          continue;
        }
        OpBuilder builder(globalOp);
        builder.setInsertionPointAfter(globalOp);
        auto packedType = RankedTensorType::get(
            {static_cast<int64_t>(output.size())}, builder.getI8Type());
        auto packedAttr = DenseElementsAttr::getFromRawBuffer(
            packedType, ArrayRef<char>(reinterpret_cast<const char *>(output.data()),
                                       output.size()));
        packedGlobal = IREE::Util::GlobalOp::create(
            builder, globalOp.getLoc(),
            (globalOp.getSymName() + "_rocket_packed").str(),
            /*isMutable=*/false, packedType, TypedAttr(packedAttr));
        packedGlobal.setPrivate();
        for (StringRef inherited : {"inlining_policy", "stream.affinity.default"}) {
          if (Attribute attr = globalOp->getAttr(inherited)) {
            packedGlobal->setAttr(inherited, attr);
          }
        }
        // Insert into the symbol table so a name clash is renamed rather than
        // duplicated.
        packedGlobal->remove();
        symbolTable.insert(packedGlobal, Block::iterator(globalOp->getNextNode()));
        packedGlobals[globalKey] = packedGlobal;
        denseBytes += bytes.size();
        packedBytes += output.size();
      }

      // Swap the operand: a static i8 tensor has no dynamic dims, so the
      // old operand's dims leave the argument_dims segment.
      OpBuilder builder(dispatchOp);
      Value packedLoad = IREE::Util::GlobalLoadOp::create(
          builder, dispatchOp.getLoc(), packedGlobal.getType(),
          FlatSymbolRefAttr::get(builder.getContext(), packedGlobal.getSymName()),
          builder.getUnitAttr());
      unsigned dimsBefore = 0;
      for (unsigned i = 0; i < request->weightsIndex; ++i) {
        if (auto shaped = dyn_cast<ShapedType>(arguments[i].getType())) {
          dimsBefore += shaped.getNumDynamicDims();
        }
      }
      unsigned dimsOfWeights = 0;
      if (auto shaped = dyn_cast<ShapedType>(weights.getType())) {
        dimsOfWeights = shaped.getNumDynamicDims();
      }
      dispatchOp->setOperand(
          arguments.getBeginOperandIndex() + request->weightsIndex, packedLoad);
      if (dimsOfWeights > 0) {
        dispatchOp.getArgumentDimsMutable().erase(dimsBefore, dimsOfWeights);
      }
      for (IREE::Flow::TensorReshapeOp reshape : reshapes) {
        if (reshape->use_empty()) {
          reshape.erase();
        }
      }
      if (loadOp->use_empty()) {
        loadOp.erase();
      }
      if (symbolTable.symbolKnownUseEmpty(globalOp, module)) {
        symbolTable.erase(globalOp);
      }

      // Retarget at the packed clone of the executable.
      IREE::HAL::ExecutableOp packedExecutable =
          packedExecutables.lookup(target->executableOp);
      if (!packedExecutable) {
        Operation *cloned = target->executableOp->clone();
        packedExecutable = cast<IREE::HAL::ExecutableOp>(cloned);
        SymbolTable::setSymbolName(
            packedExecutable, (target->executableOp.getSymName() + "_packed").str());
        symbolTable.insert(packedExecutable,
                           Block::iterator(target->executableOp->getNextNode()));
        for (auto variantOp : packedExecutable.getOps<IREE::HAL::ExecutableVariantOp>()) {
          IREE::HAL::ExecutableTargetAttr variantTarget = variantOp.getTarget();
          if (!variantTarget || variantTarget.getBackend().getValue() != kRocketBackend) {
            continue;
          }
          SmallVector<NamedAttribute> entries;
          for (NamedAttribute entry : variantTarget.getConfiguration()) {
            if (entry.getName().getValue() != kWeightsPackedKey) {
              entries.push_back(entry);
            }
          }
          entries.emplace_back(builder.getStringAttr(kWeightsPackedKey),
                               builder.getBoolAttr(true));
          variantOp.setTargetAttr(IREE::HAL::ExecutableTargetAttr::get(
              builder.getContext(), variantTarget.getBackend(),
              variantTarget.getFormat(), builder.getDictionaryAttr(entries)));
        }
        packedExecutables[target->executableOp] = packedExecutable;
      }
      dispatchOp.setEntryPointsAttr(builder.getArrayAttr({SymbolRefAttr::get(
          builder.getContext(), packedExecutable.getSymName(),
          target->entryPoint.getNestedReferences())}));
      ++packed;
    }
    if (packed > 0 || nonConstant > 0 || refused > 0) {
      module.emitRemark() << "rocket-pack-weights: " << packed
                          << " dispatch(es) packed (" << denseBytes << " -> "
                          << packedBytes << " bytes), " << nonConstant
                          << " with non-constant weights or dimensions, "
                          << refused << " refused by the packer";
    }
  }
};

static PassRegistration<RocketPackWeightsPass> reg;

} // namespace
} // namespace mlir::iree_compiler::IREE::HAL
