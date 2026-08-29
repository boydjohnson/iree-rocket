// Copyright 2026
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Tags every conv-family linalg op with rocket.origin/rocket.origin_kind
// before rocket_conv2d_transform_spec.mlir's match/rewrite loop runs and
// erases the ones it claims. This is a structural classification only (op
// name + static kernel size + stride) -- it deliberately does not
// re-implement the dynamic-dim/channel-cap eligibility checks the transform
// spec's transform.iree.match.* predicates do, so the two can't drift out of
// sync. Ops that match a Rocket dispatch are erased along with this
// annotation; ops that don't keep it, so a --compile-to=preprocessing dump
// shows exactly which conv-shaped ops fell through to CPU.
//
// Wired in via one `transform.apply_registered_pass
// "rocket-annotate-original-placement"` line in
// rocket_conv2d_transform_spec.mlir's @__transform_main, right before the
// foreach_match loop -- not via a new iree-compile flag.

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir::iree_compiler::IREE::HAL {
namespace {

// Kernel height/width as seen in the filter operand's static shape, or -1 if
// either extent is dynamic. Each named linalg conv op lays its filter out
// differently, so the (height-dim, width-dim) pair is op-specific.
std::pair<int64_t, int64_t> getKernelHW(ShapedType filterType,
                                        unsigned heightDim,
                                        unsigned widthDim) {
  if (!filterType.hasRank()) {
    return {-1, -1};
  }
  int64_t kh = filterType.getDimSize(heightDim);
  int64_t kw = filterType.getDimSize(widthDim);
  return {ShapedType::isDynamic(kh) ? -1 : kh,
          ShapedType::isDynamic(kw) ? -1 : kw};
}

std::string formatExtent(int64_t v) {
  return v < 0 ? "?" : std::to_string(v);
}

std::string formatStride(DenseIntElementsAttr strides) {
  auto values = llvm::to_vector(strides.getValues<int64_t>());
  if (values.size() != 2) {
    return "s?";
  }
  if (values[0] == values[1]) {
    return "s" + std::to_string(values[0]);
  }
  return "s" + std::to_string(values[0]) + "x" + std::to_string(values[1]);
}

// Best-effort structural label mirroring the @match_* sequence names in
// rocket_conv2d_transform_spec.mlir, e.g. "dense_conv2d_3x3_s2" or
// "depthwise_conv2d_nhwc_1x1_s1". Returns std::nullopt for ops this pass
// doesn't recognize as conv-family.
std::optional<std::string> classifyConvOp(Operation *op) {
  return llvm::TypeSwitch<Operation *, std::optional<std::string>>(op)
      .Case<linalg::Conv2DNhwcHwcfOp>([&](auto convOp) {
        auto filterType = cast<ShapedType>(convOp.getInputs()[1].getType());
        auto [kh, kw] = getKernelHW(filterType, /*heightDim=*/0,
                                     /*widthDim=*/1);
        return "dense_conv2d_" + formatExtent(kh) + "x" + formatExtent(kw) +
               "_" + formatStride(convOp.getStrides());
      })
      .Case<linalg::Conv2DNchwFchwOp>([&](auto convOp) {
        auto filterType = cast<ShapedType>(convOp.getInputs()[1].getType());
        auto [kh, kw] = getKernelHW(filterType, /*heightDim=*/2,
                                     /*widthDim=*/3);
        return "dense_conv2d_" + formatExtent(kh) + "x" + formatExtent(kw) +
               "_" + formatStride(convOp.getStrides());
      })
      .Case<linalg::DepthwiseConv2DNhwcHwcOp>([&](auto convOp) {
        auto filterType = cast<ShapedType>(convOp.getInputs()[1].getType());
        auto [kh, kw] = getKernelHW(filterType, /*heightDim=*/0,
                                     /*widthDim=*/1);
        return "depthwise_conv2d_nhwc_" + formatExtent(kh) + "x" +
               formatExtent(kw) + "_" + formatStride(convOp.getStrides());
      })
      .Case<linalg::DepthwiseConv2DNchwChwOp>([&](auto convOp) {
        auto filterType = cast<ShapedType>(convOp.getInputs()[1].getType());
        auto [kh, kw] = getKernelHW(filterType, /*heightDim=*/1,
                                     /*widthDim=*/2);
        return "depthwise_conv2d_nchw_" + formatExtent(kh) + "x" +
               formatExtent(kw) + "_" + formatStride(convOp.getStrides());
      })
      .Case<linalg::DepthwiseConv2DNhwcHwcmOp>([&](auto convOp) {
        auto filterType = cast<ShapedType>(convOp.getInputs()[1].getType());
        auto [kh, kw] = getKernelHW(filterType, /*heightDim=*/0,
                                     /*widthDim=*/1);
        return "depthwise_conv2d_channel_multiplier_" + formatExtent(kh) +
               "x" + formatExtent(kw) + "_" + formatStride(convOp.getStrides());
      })
      .Default([](Operation *) { return std::nullopt; });
}

struct RocketAnnotateOriginalPlacementPass
    : public PassWrapper<RocketAnnotateOriginalPlacementPass,
                         OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      RocketAnnotateOriginalPlacementPass)

  StringRef getArgument() const final {
    return "rocket-annotate-original-placement";
  }
  StringRef getDescription() const final {
    return "Tags conv-family linalg ops with their structural shape/kind "
           "before the Rocket transform spec's match/rewrite step runs.";
  }

  void runOnOperation() final {
    // Applied via transform.apply_registered_pass directly to individual
    // util.func ops (not the enclosing module), so this must tolerate
    // running on any operation, not just ModuleOp.
    Operation *root = getOperation();
    Builder builder(root->getContext());
    unsigned taggedCount = 0;
    root->walk([&](Operation *op) {
      std::optional<std::string> kind = classifyConvOp(op);
      if (!kind) {
        return;
      }
      op->setAttr("rocket.origin", builder.getStringAttr("candidate"));
      op->setAttr("rocket.origin_kind", builder.getStringAttr(*kind));
      ++taggedCount;
    });
    if (taggedCount > 0) {
      root->emitRemark() << "rocket-annotate-original-placement: tagged "
                         << taggedCount << " conv-family op(s)";
    }
  }
};

static PassRegistration<RocketAnnotateOriginalPlacementPass> reg;

} // namespace
} // namespace mlir::iree_compiler::IREE::HAL
