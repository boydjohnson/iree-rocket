// Copyright 2026
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Tripwire for the class of bug that made a stride-2 convolution reach the
// NPU as a stride-1 one: a named convolution whose output spatial extent is
// not the extent its own input, filter, stride and dilation imply.
//
// linalg does not verify this. A conv op's iteration space is driven by its
// output, so shrinking the output (or erasing `strides`, which silently
// resets it to 1) produces an op that still verifies and still lowers -- it
// just computes a different convolution, reading a corner of its input. That
// is what `iree-global-opt-demote-contraction-inputs` used to do here; see
// RocketDemoteConvInputsPass.cpp for the full account.
//
// This runs in the transform spec immediately before the match/rewrite loop,
// while padding is still explicit (tensor.pad) and nothing has tiled or
// sliced anything, so at this point every convolution in the program should
// consume its whole input and equality is exact.
//
// It reports an error rather than merely holding the op back from the NPU.
// An inconsistent extent this early means an earlier pass rewrote the op
// into something that is no longer the imported model, so the CPU fallback
// would be just as wrong as the NPU dispatch -- silently returning wrong
// numbers is the outcome most worth preventing. Ops with dynamic extents are
// skipped: there is nothing to check against.

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include <array>

namespace mlir::iree_compiler::IREE::HAL {
namespace {

// Which dimensions carry the two spatial extents, per named-op layout. The
// Rocket matchers only ever claim these four, and spelling the layouts out
// beats inferring them: the whole point of this pass is to not trust a
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

// Reads a `strides`/`dilations` style attribute, defaulting to 1 when absent
// -- the same default the op itself applies.
int64_t getStepAttr(Operation *op, StringRef name, unsigned index) {
  auto attr = op->getAttrOfType<DenseIntElementsAttr>(name);
  if (!attr || index >= attr.getNumElements()) {
    return 1;
  }
  return attr.getValues<APInt>()[index].getSExtValue();
}

struct RocketVerifyConvShapesPass
    : public PassWrapper<RocketVerifyConvShapesPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RocketVerifyConvShapesPass)

  StringRef getArgument() const final { return "rocket-verify-conv-shapes"; }
  StringRef getDescription() const final {
    return "Errors on a named convolution whose output spatial extent "
           "disagrees with its input, filter, stride and dilation.";
  }

  void runOnOperation() final {
    getOperation()->walk(
        [&](linalg::LinalgOp linalgOp) {
          Operation *op = linalgOp.getOperation();
          std::optional<ConvSpatialDims> dims = getSpatialDims(op);
          if (!dims) {
            return;
          }
          auto inputType =
              dyn_cast<RankedTensorType>(linalgOp.getDpsInputs()[0].getType());
          auto filterType =
              dyn_cast<RankedTensorType>(linalgOp.getDpsInputs()[1].getType());
          auto outputType =
              dyn_cast<RankedTensorType>(linalgOp.getDpsInits()[0].getType());
          if (!inputType || !filterType || !outputType) {
            return;
          }

          for (unsigned axis = 0; axis < 2; ++axis) {
            int64_t input = inputType.getDimSize(dims->input[axis]);
            int64_t filter = filterType.getDimSize(dims->filter[axis]);
            int64_t output = outputType.getDimSize(dims->output[axis]);
            if (ShapedType::isDynamic(input) || ShapedType::isDynamic(filter) ||
                ShapedType::isDynamic(output)) {
              continue;
            }
            int64_t stride = getStepAttr(op, "strides", axis);
            int64_t dilation = getStepAttr(op, "dilations", axis);
            if (stride <= 0 || dilation <= 0) {
              continue;
            }
            int64_t expected =
                (input - dilation * (filter - 1) - 1) / stride + 1;
            if (output == expected) {
              continue;
            }
            op->emitError()
                << "rocket-verify-conv-shapes: " << op->getName()
                << " output extent " << output << " on spatial axis " << axis
                << " disagrees with its operands: input " << input
                << ", filter " << filter << ", stride " << stride
                << ", dilation " << dilation << " imply " << expected
                << ". An earlier pass has rewritten this convolution into a "
                   "different one -- check whether it dropped 'strides' or "
                   "'dilations'";
            return signalPassFailure();
          }
        });
  }
};

static PassRegistration<RocketVerifyConvShapesPass> reg;

} // namespace
} // namespace mlir::iree_compiler::IREE::HAL
