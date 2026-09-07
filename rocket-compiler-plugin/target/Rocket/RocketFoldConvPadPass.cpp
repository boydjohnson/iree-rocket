// Copyright 2026
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Marks a convolution whose input is a symmetric `tensor.pad` so the matcher
// can claim the pair and let the CNA do the padding, instead of IREE
// materializing it as a full-tensor copy first.
//
// A model's "same" padding arrives as an explicit `tensor.pad` in front of
// the convolution, and IREE forms it as its own dispatch -- `slow_memcpy` in
// an `audit` listing. On ResNet50 fp16 that is **16 dispatch sites feeding
// offloaded convolutions**, every one of them symmetric, moving roughly
// 12 MB per inference to copy a tensor into a slightly larger one. The CNA
// pads for free: `CNA_PAD_CON0.pad_top`/`pad_left` cost no cycles and no DMA.
//
// # What this can and cannot fold
//
// **Symmetric only, and that is a hardware limit rather than a wire-format
// one.** `CNA_PAD_CON0` has exactly two fields, `pad_top` and `pad_left`, and
// the hardware applies each to *both* sides -- `Shape::output_width` is
// `(w + 2 * pad_left - kw) / stride + 1`, matched against all 150 strided
// programs in the vendor corpus. There is no trailing-pad register, so a
// `low[0] high[1]` pad -- what ONNX emits for `auto_pad = SAME_UPPER` at
// stride 2 -- cannot be expressed at all. Folding it as a symmetric pad
// would shift every output window by one and silently compute a different
// convolution. Those stay materialized, and MobileNetV2's five stride-2 pads
// are exactly that case.
//
// **Spatial only.** A pad on the batch or channel axis is not padding in the
// convolution's sense and is declined.
//
// **The pad value must be zero.** The CNA fills with
// `CNA_PAD_CON1.pad_value`, which the fp16 targets leave at zero; a nonzero
// fill would need that field on the wire, and no measured model asks for one.
//
// # Why an attribute rather than a rewrite
//
// The convolution keeps reading the *padded* tensor. This pass only records
// on the convolution what the pad was, and the matcher checks the attribute
// and rewires the dispatch to the pad's *source*. Doing it the other way --
// rewriting the convolution to read the unpadded input here -- would change
// the op's own result extent, which `rocket-verify-conv-shapes` then flags,
// and would leave the graph miscompiled for any convolution the matcher
// later declines on a channel bound. Recording and letting the matcher
// decide keeps the decline path correct by construction.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::HAL {
namespace {

/// Names the pad this convolution's input carries, in the order
/// `Conv2DDef` spells it.
constexpr StringLiteral kPadTopAttrName = "rocket.pad_top";
constexpr StringLiteral kPadLeftAttrName = "rocket.pad_left";

/// The largest pad `CNA_PAD_CON0`'s 4-bit fields hold.
constexpr int64_t kMaxPad = 15;

/// Reads a `tensor.pad` whose region yields a zero constant and whose low and
/// high amounts are equal, static, and nonzero on exactly the two spatial
/// axes of `layout`.
///
/// `spatial` names the two dimensions that are height and width in this
/// convolution's own layout -- {1, 2} for NHWC and {2, 3} for NCHW -- because
/// which axes may be padded is a property of the operand, not of the pad.
static bool matchSymmetricSpatialPad(tensor::PadOp pad,
                                     ArrayRef<int64_t> spatial,
                                     int64_t &padHeight, int64_t &padWidth) {
  ArrayRef<int64_t> low = pad.getStaticLow();
  ArrayRef<int64_t> high = pad.getStaticHigh();
  if (low.size() != high.size()) {
    return false;
  }
  // Every dynamic amount makes this unfoldable: the wire carries a constant.
  if (!pad.getLow().empty() || !pad.getHigh().empty()) {
    return false;
  }

  for (int64_t dim = 0, rank = low.size(); dim < rank; ++dim) {
    bool isSpatial = llvm::is_contained(spatial, dim);
    if (!isSpatial) {
      if (low[dim] != 0 || high[dim] != 0) {
        return false;
      }
      continue;
    }
    // Asymmetric is not a missing feature -- see the file comment. There is
    // no trailing-pad register, so folding this would compute a different
    // convolution.
    if (low[dim] != high[dim] || low[dim] < 0 || low[dim] > kMaxPad) {
      return false;
    }
  }
  padHeight = low[spatial[0]];
  padWidth = low[spatial[1]];
  if (padHeight == 0 && padWidth == 0) {
    return false;
  }

  // The fill has to be zero: the CNA's pad value is left at zero by every
  // fp16 target and is not on the wire.
  Region &region = pad.getRegion();
  if (!region.hasOneBlock()) {
    return false;
  }
  auto yield = dyn_cast<tensor::YieldOp>(region.front().getTerminator());
  if (!yield) {
    return false;
  }
  APFloat fill(0.0f);
  if (!matchPattern(yield.getValue(), m_ConstantFloat(&fill)) ||
      !fill.isZero()) {
    return false;
  }
  return true;
}

template <typename ConvOp, int64_t SpatialH, int64_t SpatialW>
struct MarkPaddedConv : public OpRewritePattern<ConvOp> {
  using OpRewritePattern<ConvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvOp conv,
                                PatternRewriter &rewriter) const override {
    if (conv->hasAttr(kPadTopAttrName)) {
      return failure();
    }
    if (conv.getDpsInputs().empty()) {
      return failure();
    }
    // The pad is not the immediate producer. The channels-last conversion
    // pads in the rank-5 `1x1xHxWxC` form and collapses back to rank 4 for
    // the convolution, so the chain is `conv <- collapse_shape <- pad`.
    // Walking the reshapes is safe because they do not move the spatial
    // extents relative to each other -- the pad's own rank is what
    // `spatial` is checked against below, which is why the caller passes the
    // *padded operand's* axes rather than the convolution's.
    Value source = conv.getDpsInputs()[0];
    int64_t reshapes = 0;
    while (reshapes++ < 4) {
      if (auto collapse = source.template getDefiningOp<tensor::CollapseShapeOp>()) {
        source = collapse.getSrc();
        continue;
      }
      if (auto expand = source.template getDefiningOp<tensor::ExpandShapeOp>()) {
        source = expand.getSrc();
        continue;
      }
      break;
    }
    auto pad = source.template getDefiningOp<tensor::PadOp>();
    if (!pad) {
      return failure();
    }
    // The pad's own rank decides which axes are spatial: rank 5 is the
    // channels-last `1x1xHxWxC` form the conversion pads in, rank 4 the
    // plain one. Both put height and width adjacent, one axis later in the
    // rank-5 case.
    auto padType = cast<RankedTensorType>(pad.getResult().getType());
    int64_t shift = padType.getRank() - 4;
    if (shift < 0 || shift > 1) {
      return failure();
    }
    const int64_t spatial[2] = {SpatialH + shift, SpatialW + shift};
    int64_t padHeight = 0;
    int64_t padWidth = 0;
    if (!matchSymmetricSpatialPad(pad, spatial, padHeight, padWidth)) {
      return failure();
    }
    rewriter.modifyOpInPlace(conv, [&] {
      conv->setAttr(kPadTopAttrName, rewriter.getI64IntegerAttr(padHeight));
      conv->setAttr(kPadLeftAttrName, rewriter.getI64IntegerAttr(padWidth));
    });
    return success();
  }
};

struct RocketFoldConvPadPass
    : public PassWrapper<RocketFoldConvPadPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RocketFoldConvPadPass)

  StringRef getArgument() const final { return "rocket-fold-conv-pad"; }
  StringRef getDescription() const final {
    return "Marks a convolution whose input is a symmetric, zero-filled, "
           "spatial-only tensor.pad with rocket.pad_top/rocket.pad_left, so "
           "the matcher can hand the padding to the CNA instead of leaving it "
           "as a materialized copy.";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() final {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    // NHWC puts height and width at 1 and 2; the NCHW depthwise form at 2
    // and 3.
    patterns.add<MarkPaddedConv<linalg::Conv2DNhwcHwcfOp, 1, 2>,
                 MarkPaddedConv<linalg::DepthwiseConv2DNhwcHwcOp, 1, 2>,
                 MarkPaddedConv<linalg::DepthwiseConv2DNchwChwOp, 2, 3>>(
        context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

static PassRegistration<RocketFoldConvPadPass> reg;

} // namespace
} // namespace mlir::iree_compiler::IREE::HAL
