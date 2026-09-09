// Copyright 2026
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Tripwire for the class of bug that made a stride-2 convolution reach the
// NPU as a stride-1 one: a named convolution or matmul rebuilt by an earlier
// pass as a *different* op.
//
// linalg does not verify this. A conv op's iteration space is driven by its
// output, so shrinking the output (or erasing `strides`, which silently
// resets it to 1) produces an op that still verifies and still lowers -- it
// just computes a different convolution, reading a corner of its input. That
// is what `iree-global-opt-demote-contraction-inputs` used to do here; see
// RocketDemoteConvInputsPass.cpp for the full account. A matmul rebuilt
// without its `indexing_maps` is the same defect with a different name: a
// transposed matmul becomes an untransposed one.
//
// Two passes, one file, because they share the record:
//
//   rocket-record-conv-attrs   stamps every named convolution and contraction
//                              with `rocket.recorded_attrs`, a dictionary of
//                              the attributes that define which op it is --
//                              `strides`, `dilations`, `indexing_maps`,
//                              `cast` -- and marks the enclosing op with
//                              `rocket.conv_attrs_recorded`.
//   rocket-verify-conv-shapes  errors on any op whose attributes no longer
//                              agree with its record, or that has lost its
//                              record entirely while the mark is present,
//                              then strips both.
//
// The record is a discardable attribute, which is the point: every rebuild
// that goes through `linalg::getPrunedAttributeList` (the upstream demote,
// the plugin's own) carries discardable attributes over while eliding the
// inherent ones, so the record survives precisely the rewrites that lose
// `strides`. It is compared on *effective* values -- an absent `strides` is
// all ones, an absent `cast` is `cast_signed` -- so a pass that spells a
// default explicitly is not a change.
//
// This is the check DYNAMIC_SHAPES.md DS4 asked for. The pass used to verify
// only an arithmetic consequence of the attributes -- that the output spatial
// extent is the one the input, filter, stride and dilation imply -- and that
// needs every extent to be static, so it went blind on exactly the symbolic
// models the runtime was built for. The attribute comparison is shape
// independent and names the actual defect. The arithmetic check is kept as
// a second opinion where the extents are static: it also catches an op whose
// attributes are intact but whose output type was rewritten.
//
// Both run in the transform spec around the demotion, immediately before the
// match/rewrite loop, while padding is still explicit (tensor.pad) and
// nothing has tiled or sliced anything, so every convolution in the program
// consumes its whole input and the arithmetic equality is exact. The record
// must be gone before the loop: the DAG matchers compare whole attribute
// dictionaries, and a stray `rocket.recorded_attrs` would make every
// convolution decline. Verify strips it; nothing else should.
//
// Verify reports an error rather than merely holding the op back from the
// NPU. An inconsistent op this early means an earlier pass rewrote it into
// something that is no longer the imported model, so the CPU fallback would
// be just as wrong as the NPU dispatch -- silently returning wrong numbers is
// the outcome most worth preventing.
//
// Run on its own, without the record pass first, verify degrades to the
// arithmetic check alone (no mark, so a missing record is not an error);
// that keeps the pass usable on hand-written IR in tests.

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <string>

namespace mlir::iree_compiler::IREE::HAL {
namespace {

// Per-op record: a DictionaryAttr of the attributes that define the op.
constexpr StringLiteral kRecordAttrName = "rocket.recorded_attrs";
// On the op the record pass ran on (a function in the spec). Its presence is
// what makes a missing per-op record an error.
constexpr StringLiteral kRecordedMarkAttrName = "rocket.conv_attrs_recorded";

constexpr StringLiteral kStridesAttrName = "strides";
constexpr StringLiteral kDilationsAttrName = "dilations";
constexpr StringLiteral kIndexingMapsAttrName = "indexing_maps";
constexpr StringLiteral kCastAttrName = "cast";

// The ops both passes track: every *named* convolution or contraction. This
// is the same predicate the upstream demote uses to pick its candidates, so
// anything it could rebuild is something we record. linalg.generic
// implements neither interface.
bool isTrackedOp(linalg::LinalgOp linalgOp) {
  return isa<linalg::ConvolutionOpInterface, linalg::ContractionOpInterface>(
      linalgOp.getOperation());
}

// A `strides`/`dilations` style attribute as a vector, empty when absent.
// Reads both the op's own DenseIntElementsAttr spelling and the record's
// DenseI64ArrayAttr one.
SmallVector<int64_t> readSteps(Attribute attr) {
  SmallVector<int64_t> steps;
  if (auto dense = dyn_cast_or_null<DenseIntElementsAttr>(attr)) {
    for (APInt value : dense.getValues<APInt>()) {
      steps.push_back(value.getSExtValue());
    }
  } else if (auto array = dyn_cast_or_null<DenseI64ArrayAttr>(attr)) {
    steps.assign(array.asArrayRef().begin(), array.asArrayRef().end());
  }
  return steps;
}

// Equality on effective values: an absent attribute is all ones, which is
// the default the op itself applies.
bool stepsAgree(ArrayRef<int64_t> a, ArrayRef<int64_t> b) {
  if (a.size() == b.size()) {
    return a == b;
  }
  ArrayRef<int64_t> present = a.empty() ? b : a;
  ArrayRef<int64_t> absent = a.empty() ? a : b;
  return absent.empty() && llvm::all_of(present, [](int64_t s) { return s == 1; });
}

std::string stepsToString(ArrayRef<int64_t> steps) {
  if (steps.empty()) {
    return "absent (all 1)";
  }
  std::string out = "[";
  llvm::raw_string_ostream os(out);
  llvm::interleaveComma(steps, os);
  os << "]";
  return out;
}

linalg::TypeFnAttr effectiveCast(Operation *op, Attribute attr) {
  if (auto cast = dyn_cast_or_null<linalg::TypeFnAttr>(attr)) {
    return cast;
  }
  return linalg::TypeFnAttr::get(op->getContext(), linalg::TypeFn::cast_signed);
}

// Reads a `strides`/`dilations` style attribute, defaulting to 1 when absent
// -- the same default the op itself applies.
int64_t getStepAttr(Operation *op, StringRef name, unsigned index) {
  auto attr = op->getAttrOfType<DenseIntElementsAttr>(name);
  if (!attr || index >= attr.getNumElements()) {
    return 1;
  }
  return attr.getValues<APInt>()[index].getSExtValue();
}

struct RocketRecordConvAttrsPass
    : public PassWrapper<RocketRecordConvAttrsPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RocketRecordConvAttrsPass)

  StringRef getArgument() const final { return "rocket-record-conv-attrs"; }
  StringRef getDescription() const final {
    return "Records the shape-defining attributes of every named convolution "
           "and contraction for rocket-verify-conv-shapes to check against.";
  }

  void runOnOperation() final {
    Operation *root = getOperation();
    MLIRContext *context = root->getContext();
    Builder builder(context);
    root->walk([&](linalg::LinalgOp linalgOp) {
      if (!isTrackedOp(linalgOp)) {
        return;
      }
      Operation *op = linalgOp.getOperation();
      SmallVector<NamedAttribute> record;
      for (StringRef name : {kStridesAttrName, kDilationsAttrName}) {
        SmallVector<int64_t> steps = readSteps(op->getAttr(name));
        if (!steps.empty()) {
          record.emplace_back(builder.getStringAttr(name),
                              builder.getDenseI64ArrayAttr(steps));
        }
      }
      // Effective maps, not the attribute: a named op derives them from its
      // strides and dilations (or from its defaults), so this is one value
      // that changes whenever any of them does.
      record.emplace_back(builder.getStringAttr(kIndexingMapsAttrName),
                          linalgOp.getIndexingMaps());
      if (Attribute cast = op->getAttr(kCastAttrName)) {
        record.emplace_back(builder.getStringAttr(kCastAttrName), cast);
      }
      op->setAttr(kRecordAttrName, builder.getDictionaryAttr(record));
    });
    root->setAttr(kRecordedMarkAttrName, builder.getUnitAttr());
  }
};

// Which dimensions carry the two spatial extents, per named-op layout. The
// Rocket matchers only ever claim these four, and spelling the layouts out
// beats inferring them: the whole point of this check is to not trust a
// derived answer.
struct ConvSpatialDims {
  std::array<unsigned, 2> input;
  std::array<unsigned, 2> filter;
  std::array<unsigned, 2> output;
};

std::optional<ConvSpatialDims> getSpatialDims(Operation *op) {
  return llvm::TypeSwitch<Operation *, std::optional<ConvSpatialDims>>(op)
      // input NCHW, filter FCHW, output NFHW
      .Case<linalg::Conv2DNchwFchwOp>(
          [](auto) { return ConvSpatialDims{{2, 3}, {2, 3}, {2, 3}}; })
      // input NHWC, filter HWCF, output NHWC
      .Case<linalg::Conv2DNhwcHwcfOp>(
          [](auto) { return ConvSpatialDims{{1, 2}, {0, 1}, {1, 2}}; })
      // input NHWC, filter HWC, output NHWC
      .Case<linalg::DepthwiseConv2DNhwcHwcOp>(
          [](auto) { return ConvSpatialDims{{1, 2}, {0, 1}, {1, 2}}; })
      // input NCHW, filter CHW, output NCHW
      .Case<linalg::DepthwiseConv2DNchwChwOp>(
          [](auto) { return ConvSpatialDims{{2, 3}, {1, 2}, {2, 3}}; })
      .Default([](auto) { return std::nullopt; });
}

struct RocketVerifyConvShapesPass
    : public PassWrapper<RocketVerifyConvShapesPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RocketVerifyConvShapesPass)

  StringRef getArgument() const final { return "rocket-verify-conv-shapes"; }
  StringRef getDescription() const final {
    return "Errors on a named convolution or contraction whose attributes "
           "disagree with the record rocket-record-conv-attrs left, or whose "
           "output spatial extent disagrees with its input, filter, stride "
           "and dilation.";
  }

  // Compares the op's effective attributes against its record. Reports the
  // first disagreement and returns failure.
  LogicalResult verifyRecord(linalg::LinalgOp linalgOp, DictionaryAttr record) {
    Operation *op = linalgOp.getOperation();
    auto changed = [&](StringRef name, auto &&recorded, auto &&current) {
      op->emitError()
          << "rocket-verify-conv-shapes: " << op->getName() << " '" << name
          << "' changed since rocket-record-conv-attrs: recorded " << recorded
          << ", now " << current
          << ". An earlier pass has rebuilt this op without carrying '" << name
          << "' over (linalg::getPrunedAttributeList elides it)";
      return failure();
    };
    for (StringRef name : {kStridesAttrName, kDilationsAttrName}) {
      Attribute recordedAttr = record.get(name);
      Attribute currentAttr = op->getAttr(name);
      if (!recordedAttr && !currentAttr) {
        continue;
      }
      SmallVector<int64_t> recorded = readSteps(recordedAttr);
      SmallVector<int64_t> current = readSteps(currentAttr);
      if (!stepsAgree(recorded, current)) {
        return changed(name, stepsToString(recorded), stepsToString(current));
      }
    }
    if (auto recorded = record.getAs<ArrayAttr>(kIndexingMapsAttrName)) {
      ArrayAttr current = linalgOp.getIndexingMaps();
      if (recorded != current) {
        return changed(kIndexingMapsAttrName, recorded, current);
      }
    }
    if (Attribute recordedAttr = record.get(kCastAttrName)) {
      linalg::TypeFnAttr recorded = effectiveCast(op, recordedAttr);
      linalg::TypeFnAttr current = effectiveCast(op, op->getAttr(kCastAttrName));
      if (recorded != current) {
        return changed(kCastAttrName, recorded, current);
      }
    }
    return success();
  }

  // The arithmetic check: only where every extent on the axis is static.
  LogicalResult verifyExtents(linalg::LinalgOp linalgOp) {
    Operation *op = linalgOp.getOperation();
    std::optional<ConvSpatialDims> dims = getSpatialDims(op);
    if (!dims) {
      return success();
    }
    auto inputType =
        dyn_cast<RankedTensorType>(linalgOp.getDpsInputs()[0].getType());
    auto filterType =
        dyn_cast<RankedTensorType>(linalgOp.getDpsInputs()[1].getType());
    auto outputType =
        dyn_cast<RankedTensorType>(linalgOp.getDpsInits()[0].getType());
    if (!inputType || !filterType || !outputType) {
      return success();
    }

    for (unsigned axis = 0; axis < 2; ++axis) {
      int64_t input = inputType.getDimSize(dims->input[axis]);
      int64_t filter = filterType.getDimSize(dims->filter[axis]);
      int64_t output = outputType.getDimSize(dims->output[axis]);
      if (ShapedType::isDynamic(input) || ShapedType::isDynamic(filter) ||
          ShapedType::isDynamic(output)) {
        continue;
      }
      int64_t stride = getStepAttr(op, kStridesAttrName, axis);
      int64_t dilation = getStepAttr(op, kDilationsAttrName, axis);
      if (stride <= 0 || dilation <= 0) {
        continue;
      }
      int64_t expected = (input - dilation * (filter - 1) - 1) / stride + 1;
      if (output == expected) {
        continue;
      }
      op->emitError()
          << "rocket-verify-conv-shapes: " << op->getName()
          << " output extent " << output << " on spatial axis " << axis
          << " disagrees with its operands: input " << input << ", filter "
          << filter << ", stride " << stride << ", dilation " << dilation
          << " imply " << expected
          << ". An earlier pass has rewritten this convolution into a "
             "different one -- check whether it dropped 'strides' or "
             "'dilations'";
      return failure();
    }
    return success();
  }

  void runOnOperation() final {
    Operation *root = getOperation();
    bool recorded = root->hasAttr(kRecordedMarkAttrName);
    WalkResult result = root->walk([&](linalg::LinalgOp linalgOp) {
      Operation *op = linalgOp.getOperation();
      auto record = op->getAttrOfType<DictionaryAttr>(kRecordAttrName);
      op->removeAttr(kRecordAttrName);
      if (record) {
        if (failed(verifyRecord(linalgOp, record))) {
          return WalkResult::interrupt();
        }
      } else if (recorded && isTrackedOp(linalgOp)) {
        op->emitError()
            << "rocket-verify-conv-shapes: " << op->getName()
            << " carries no '" << kRecordAttrName
            << "'. A pass between rocket-record-conv-attrs and this one "
               "rebuilt it without carrying its discardable attributes "
               "over, so its shape-defining attributes cannot be checked";
        return WalkResult::interrupt();
      }
      if (failed(verifyExtents(linalgOp))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    root->removeAttr(kRecordedMarkAttrName);
    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

static PassRegistration<RocketRecordConvAttrsPass> recordReg;
static PassRegistration<RocketVerifyConvShapesPass> verifyReg;

} // namespace
} // namespace mlir::iree_compiler::IREE::HAL
