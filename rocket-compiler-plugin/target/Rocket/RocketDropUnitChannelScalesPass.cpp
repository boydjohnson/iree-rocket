// Copyright 2026
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Gives a requantized int8 dispatch whose per-channel factors are all
// exactly 1 an executable without `per_channel_scale`, and drops the
// factors' binding.
//
// The requantized executables carry `per_channel_scale = true`
// (Conv2DDef.per_channel_scale): a binding after the bias of one f32 factor
// per output channel, which the runtime writes into the BS plane's
// multiplier lanes. A per-tensor convolution -- every QOperator model and
// every per-tensor QDQ one -- passes a factor of 1 on every channel, which is
// a splat constant. IREE does not store a splat in the module; it
// materializes it at run time as a fill, on every call, and that fill sits
// between the Rocket dispatches it feeds. Measured on ResNet50 `ours_skip`:
// 4 -> 5 NPU submissions and 4 -> 6 output compactions per call, 17.8 ->
// 18.6 ms. A unit factor is what the BS plane's multiplier lane holds
// without the flag, so the dispatch loses nothing by not passing it: this
// pass gives such a dispatch back the four-binding executable it had before
// per-channel scales existed -- the same executable, name included: the
// spec's executable is rewritten in place for its unit-factor dispatches,
// and the dispatches with real factors move to a `_per_channel` clone that
// keeps the flag. A per-tensor model therefore compiles to the module it
// compiled to before this flag existed.
//
// Runs at the end of preprocessing, after the shims are inlined, where the
// factors operand is a `tensor.cast` of the `arith.constant` the producer
// wrote.

#include <optional>
#include <string>

#include "RocketDispatchQuery.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/ADT/MapVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::HAL {
namespace {

using namespace rocket_query;

constexpr StringLiteral kPerChannelScaleKey = "per_channel_scale";

/// Whether `value` is, through casts and reshapes, a constant whose every
/// element is 1.0.
static bool isUnitSplat(Value value) {
  while (Operation *op = value.getDefiningOp()) {
    if (auto cast = dyn_cast<tensor::CastOp>(op)) {
      value = cast.getSource();
      continue;
    }
    if (auto reshape = dyn_cast<IREE::Flow::TensorReshapeOp>(op)) {
      value = reshape.getSource();
      continue;
    }
    auto constant = dyn_cast<arith::ConstantOp>(op);
    if (!constant) {
      return false;
    }
    auto dense = dyn_cast<DenseFPElementsAttr>(constant.getValue());
    return dense && dense.isSplat() &&
           dense.getSplatValue<APFloat>().isExactlyValue(1.0);
  }
  return false;
}

/// Drops `per_channel_scale` from each Rocket variant's target of
/// `executable`, and binding `scalesBinding` from each export's layout.
static void dropScales(IREE::HAL::ExecutableOp executable, unsigned scalesBinding) {
  OpBuilder builder(executable);
  for (auto variantOp : executable.getOps<IREE::HAL::ExecutableVariantOp>()) {
    IREE::HAL::ExecutableTargetAttr target = variantOp.getTarget();
    if (target && target.getBackend().getValue() == kRocketBackend) {
      SmallVector<NamedAttribute> entries;
      for (NamedAttribute entry : target.getConfiguration()) {
        if (entry.getName().getValue() != kPerChannelScaleKey) {
          entries.push_back(entry);
        }
      }
      variantOp.setTargetAttr(IREE::HAL::ExecutableTargetAttr::get(
          builder.getContext(), target.getBackend(), target.getFormat(),
          builder.getDictionaryAttr(entries)));
    }
    for (auto exportOp : variantOp.getOps<IREE::HAL::ExecutableExportOp>()) {
      IREE::HAL::PipelineLayoutAttr layout = exportOp.getLayout();
      SmallVector<IREE::HAL::PipelineBindingAttr> bindings(layout.getBindings());
      if (scalesBinding < bindings.size()) {
        bindings.erase(bindings.begin() + scalesBinding);
      }
      exportOp.setLayoutAttr(IREE::HAL::PipelineLayoutAttr::get(
          builder.getContext(), bindings, layout.getConstants(), layout.getFlags()));
    }
  }
}

/// Points `dispatchOp` at `executable`, keeping its export.
static void retarget(IREE::Flow::DispatchOp dispatchOp, const DispatchTarget &target,
                     IREE::HAL::ExecutableOp executable) {
  dispatchOp.setEntryPointsAttr(ArrayAttr::get(
      dispatchOp.getContext(),
      {SymbolRefAttr::get(dispatchOp.getContext(), executable.getSymName(),
                          target.entryPoint.getNestedReferences())}));
}

struct RocketDropUnitChannelScalesPass
    : public PassWrapper<RocketDropUnitChannelScalesPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RocketDropUnitChannelScalesPass)

  StringRef getArgument() const final { return "rocket-drop-unit-channel-scales"; }
  StringRef getDescription() const final {
    return "Retargets a requantized int8 dispatch whose per-channel factors "
           "are all 1 at a clone of its executable without "
           "per_channel_scale, dropping the factors' binding.";
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    SymbolTable symbolTable(module);
    // Input, weights, bias, scales after the push constants.
    constexpr unsigned kScalesBinding = 3;

    // Per executable, its unit-factor dispatches and the rest, in order.
    struct Uses {
      SmallVector<IREE::Flow::DispatchOp> unit;
      SmallVector<IREE::Flow::DispatchOp> scaled;
    };
    llvm::MapVector<Operation *, Uses> uses;
    module.walk([&](IREE::Flow::DispatchOp dispatchOp) {
      std::optional<DispatchTarget> target = rocketTarget(dispatchOp);
      if (!target || !boolFlag(target->target.getConfiguration(), kPerChannelScaleKey)) {
        return;
      }
      size_t scalesIndex = constantCount(target->target.getConfiguration()) + kScalesBinding;
      OperandRange arguments = dispatchOp.getArguments();
      // A result tied to an argument names it by index; the requantized
      // shims tie none, and anything else keeps its factors.
      ArrayAttr tied = dispatchOp.getTiedOperandsAttr();
      bool untied = !tied || llvm::all_of(tied, [](Attribute attr) {
        auto index = dyn_cast<IntegerAttr>(attr);
        return index && index.getInt() < 0;
      });
      Uses &entry = uses[target->executableOp];
      if (untied && scalesIndex < arguments.size() && isUnitSplat(arguments[scalesIndex])) {
        entry.unit.push_back(dispatchOp);
      } else {
        entry.scaled.push_back(dispatchOp);
      }
    });

    for (auto &[op, entry] : uses) {
      auto executable = cast<IREE::HAL::ExecutableOp>(op);
      if (entry.unit.empty()) {
        continue;
      }
      // The dispatches with real factors keep them, on a clone made before
      // the original loses the flag.
      if (!entry.scaled.empty()) {
        auto clone = cast<IREE::HAL::ExecutableOp>(executable->clone());
        SymbolTable::setSymbolName(clone, (executable.getSymName() + "_per_channel").str());
        symbolTable.insert(clone, Block::iterator(executable->getNextNode()));
        for (IREE::Flow::DispatchOp dispatchOp : entry.scaled) {
          retarget(dispatchOp, *rocketTarget(dispatchOp), clone);
        }
      }
      dropScales(executable, kScalesBinding);

      for (IREE::Flow::DispatchOp dispatchOp : entry.unit) {
        size_t scalesIndex = constantCount(rocketTarget(dispatchOp)->target.getConfiguration()) +
                             kScalesBinding;
        // `constantCount` reads the config, which no longer lists the flag
        // but has the same push constants.
        OperandRange arguments = dispatchOp.getArguments();
        // The argument's dynamic dims are the run of `argument_dims` after
        // every earlier shaped argument's.
        unsigned dimsBefore = 0;
        for (size_t i = 0; i < scalesIndex; ++i) {
          if (auto shaped = dyn_cast<ShapedType>(arguments[i].getType())) {
            dimsBefore += shaped.getNumDynamicDims();
          }
        }
        unsigned scaleDims =
            cast<ShapedType>(arguments[scalesIndex].getType()).getNumDynamicDims();
        SmallVector<Value> newArguments(arguments.begin(), arguments.end());
        newArguments.erase(newArguments.begin() + scalesIndex);
        SmallVector<Value> newDims(dispatchOp.getArgumentDims().begin(),
                                   dispatchOp.getArgumentDims().end());
        newDims.erase(newDims.begin() + dimsBefore, newDims.begin() + dimsBefore + scaleDims);

        OpBuilder builder(dispatchOp);
        auto replacement = IREE::Flow::DispatchOp::create(
            builder, dispatchOp.getLoc(), cast<SymbolRefAttr>(dispatchOp.getEntryPointsAttr()[0]),
            dispatchOp.getWorkload(), dispatchOp.getResultTypes(), dispatchOp.getResultDims(),
            newArguments, newDims, dispatchOp.getTiedOperandsAttr(),
            llvm::to_vector(dispatchOp->getDiscardableAttrs()));
        dispatchOp.replaceAllUsesWith(replacement.getResults());
        dispatchOp.erase();
      }
    }
  }
};

static PassRegistration<RocketDropUnitChannelScalesPass> reg;

} // namespace
} // namespace mlir::iree_compiler::IREE::HAL
