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
// convolution. Those stay materialized. (An older ONNX import of MobileNetV2
// had five such stride-2 pads; the survey's torch export pads every one of
// its convolutions symmetrically, and all seventeen depthwise pads fold.)
//
// **Spatial only.** A pad on the batch or channel axis is not padding in the
// convolution's sense and is declined.
//
// **The pad value must be zero -- or, for int8, the one zero point a target
// bakes.** The CNA fills with `CNA_PAD_CON1.pad_value`, which the fp16
// targets leave at zero. An int8 QLinearConv pads with its *input zero
// point*, because the zero-point correction folded into the bias,
// `-x_zp * sum_k(w)`, counts every tap -- so an out-of-image tap has to
// contribute `x_zp * w` for the border to come out right. The requantized
// pad1 targets carry that fill as a static `input_zero_point`, which the
// driver programs into `pad_value`, and they bake exactly one value: -128,
// the signed encoding of a `ui8` zero point of 0. That is every activation a
// ReLU produces once the quantizer folds the ReLU into the conv (min 0 means
// zero point 0), which is every 3x3 input in a QDQ ResNet. Any other fill is
// left unmarked and its pad stays materialized, rather than being marked
// with nothing to claim it -- a mark changes the convolution's attribute
// dictionary, which would make the unpadded requantized matchers decline it
// too.
//
// # The region is rewritten too, and that is what pins the fill
//
// `transform.iree.match.cast_compatible_dag_from_root` compares regions
// structurally, not just attribute dictionaries -- and a region that yields a
// value defined *outside* it can never match, because structural equivalence
// has nothing to map that value to. Which is exactly how the pad arrives: the
// importer hoists the fill constant above the `tensor.pad` and yields it.
//
// So this pass also sinks the fill into the region, as
// `arith.constant 0.0 : <element type>` immediately before the
// `tensor.yield`. That makes the region self-contained, which makes it
// matchable at all -- and it means the template's own inline constant pins
// the fill to zero by attribute comparison rather than leaving it on this
// pass's word.
//
// The sinking happens *after* the greedy driver, not inside the pattern.
// `applyPatternsGreedily` runs a constant folder that hoists a constant out
// of any region that is not isolated-from-above -- which `tensor.pad`'s is
// not -- so a sink performed inside the pattern is undone before the pass
// returns. Nothing re-hoists it afterwards: no canonicalisation runs between
// this pass and `foreach_match`. The same is true of the scalar bounds
// `rocket-fuse-conv-relu6` introduces, for the same reason.
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

#include <optional>
#include <type_traits>

namespace mlir::iree_compiler::IREE::HAL {
namespace {

/// Names the pad this convolution's input carries, in the order
/// `Conv2DDef` spells it.
constexpr StringLiteral kPadTopAttrName = "rocket.pad_top";
constexpr StringLiteral kPadLeftAttrName = "rocket.pad_left";

/// The largest pad `CNA_PAD_CON0`'s 4-bit fields hold.
constexpr int64_t kMaxPad = 15;

/// The largest pad `PPU_POOLING_PADDING_CFG`'s 3-bit fields hold.
constexpr int64_t kMaxPoolPad = 7;

/// The int8 fill the requantized pad1 targets bake as `input_zero_point`:
/// a `ui8` zero point of 0 in the signed encoding the convolution reads.
constexpr int64_t kInt8PadFill = -128;

/// Reads a `tensor.pad` whose region yields a zero constant and whose low and
/// high amounts are equal, static, and nonzero on exactly the two spatial
/// axes of `layout`.
///
/// `spatial` names the two dimensions that are height and width in this
/// convolution's own layout -- {1, 2} for NHWC and {2, 3} for NCHW -- because
/// which axes may be padded is a property of the operand, not of the pad.
static bool matchSymmetricSpatialPad(tensor::PadOp pad,
                                     ArrayRef<int64_t> spatial,
                                     int64_t &padHeight, int64_t &padWidth,
                                     bool &int8Fill) {
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
  if (matchPattern(yield.getValue(), m_ConstantFloat(&fill))) {
    int8Fill = false;
    return fill.isZero();
  }
  // The int8 requantized pad1 targets' one baked zero point; see the file
  // comment.
  APInt integerFill;
  auto elementType = cast<RankedTensorType>(pad.getResult().getType())
                         .getElementType();
  if (elementType.isInteger(8) &&
      matchPattern(yield.getValue(), m_ConstantInt(&integerFill)) &&
      integerFill.getSExtValue() == kInt8PadFill) {
    int8Fill = true;
    return true;
  }
  return false;
}

/// The single user of `value`, or null when it has any other count.
static Operation *soleUser(Value value) {
  if (!value.hasOneUse()) {
    return nullptr;
  }
  return *value.getUsers().begin();
}

/// Whether `accumulator` is read only by the canonical requantization
/// epilogue: one `linalg.generic` taking the i32 accumulator, the per-channel
/// bias, and the scale, zero point and two clamp bounds as scalars, and
/// writing i8. That is the shape `rocket-fuse-int8-requant-epilogue` emits
/// and the only one the requantized pad1 matchers claim.
static bool feedsRequantEpilogue(Value accumulator) {
  auto epilogue = dyn_cast_or_null<linalg::GenericOp>(soleUser(accumulator));
  if (!epilogue || epilogue.getNumDpsInputs() != 6 ||
      epilogue.getNumDpsInits() != 1) {
    return false;
  }
  auto accumulatorType = dyn_cast<RankedTensorType>(accumulator.getType());
  auto resultType =
      dyn_cast<RankedTensorType>(epilogue.getResult(0).getType());
  return accumulatorType && resultType &&
         accumulatorType.getElementType().isInteger(32) &&
         resultType.getElementType().isInteger(8);
}

/// Whether `conv`'s sole user is the bias + clamp epilogue
/// `rocket-fuse-conv-relu6` builds: accumulator, bias and two bounds in,
/// `addf` then `maximumf` first in the body.
static bool feedsFusedClamp(Operation *conv) {
  auto activation = dyn_cast_or_null<linalg::GenericOp>(soleUser(conv->getResult(0)));
  if (!activation || activation.getNumDpsInputs() != 4) {
    return false;
  }
  Block &body = activation.getRegion().front();
  auto it = body.begin();
  return it != body.end() && isa<arith::AddFOp>(*it) && std::next(it) != body.end() &&
         isa<arith::MaximumFOp>(*std::next(it));
}

/// The padded depthwise convolutions a template claims together with their
/// ReLU6 (`@match_dynamic_depthwise_conv2d_nchw_pad1_relu6*`): 3x3, pad 1 on
/// both axes, stride 1 or 2, no dilation. MobileNetV2's seventeen depthwise
/// convolutions are all of this form, and `depthwise_pad1_hw` in
/// iree-rocket-hal runs each of its ten distinct shapes bit-exactly with the
/// CNA padding.
static bool isClaimablePaddedDepthwise(linalg::DepthwiseConv2DNchwChwOp conv,
                                       int64_t padHeight, int64_t padWidth) {
  auto filterType = dyn_cast<RankedTensorType>(conv.getDpsInputs()[1].getType());
  if (padHeight != 1 || padWidth != 1 || !filterType || filterType.getRank() != 3 ||
      filterType.getDimSize(1) != 3 || filterType.getDimSize(2) != 3) {
    return false;
  }
  SmallVector<int64_t> strides(conv.getStrides().getValues<int64_t>());
  SmallVector<int64_t> dilations(conv.getDilations().getValues<int64_t>());
  return (strides == SmallVector<int64_t>{1, 1} || strides == SmallVector<int64_t>{2, 2}) &&
         dilations == SmallVector<int64_t>{1, 1};
}

/// The padded dense convolutions a template claims together with their
/// ReLU6 (`@match_dynamic_conv2d_3x3_pad1_relu6*`): 3x3, pad 1 on both axes,
/// stride 1 or 2, no dilation. `stem_3x3_pad1_matches_oracle` in
/// iree-rocket-hal runs MobileNetV2's 224x224 Cin 3 stem this way, exact.
static bool isClaimablePaddedDense(linalg::Conv2DNhwcHwcfOp conv, int64_t padHeight,
                                   int64_t padWidth) {
  auto filterType = dyn_cast<RankedTensorType>(conv.getDpsInputs()[1].getType());
  if (padHeight != 1 || padWidth != 1 || !filterType || filterType.getRank() != 4 ||
      filterType.getDimSize(0) != 3 || filterType.getDimSize(1) != 3) {
    return false;
  }
  SmallVector<int64_t> strides(conv.getStrides().getValues<int64_t>());
  SmallVector<int64_t> dilations(conv.getDilations().getValues<int64_t>());
  return (strides == SmallVector<int64_t>{1, 1} || strides == SmallVector<int64_t>{2, 2}) &&
         dilations == SmallVector<int64_t>{1, 1};
}

/// Moves a depthwise convolution's pad below the NCHW transpose in front of
/// it, so the pad is the convolution's immediate producer.
///
/// The channels-last conversion leaves a depthwise convolution in NCHW and
/// pads its input while it is still NHWC, so the chain arrives as
/// `transpose[0, 3, 1, 2](collapse(pad(x)))` -- with the pad two ops from
/// the convolution, where neither `MarkPaddedConv` nor a DAG template can see
/// it. A zero pad commutes with the layout change exactly, so this rebuilds
/// it as `pad(transpose(collapse(x)))` with the spatial amounts moved to the
/// NCHW axes. That is the whole of what MobileNetV2's seventeen CPU pad
/// copies needed: each then folds into the CNA, and the transpose the shim
/// applies to `x` cancels against the model's own.
///
/// Only for the form a template claims (`isClaimablePaddedDepthwise`, ReLU6
/// after it) and a single-use chain, so no other convolution's pad moves.
/// The demote pass delivers the f32 import in this same form, its truncf on
/// `x` (see `demoteInput`), so both imports take this one path.
struct SinkDepthwisePadBelowTranspose
    : public OpRewritePattern<linalg::DepthwiseConv2DNchwChwOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::DepthwiseConv2DNchwChwOp conv,
                                PatternRewriter &rewriter) const override {
    if (conv->hasAttr(kPadTopAttrName) || !feedsFusedClamp(conv)) {
      return failure();
    }
    auto transpose = conv.getDpsInputs()[0].getDefiningOp<linalg::TransposeOp>();
    if (!transpose || transpose.getPermutation() != ArrayRef<int64_t>{0, 3, 1, 2} ||
        !transpose->getResult(0).hasOneUse()) {
      return failure();
    }
    Value inner = transpose.getInput();
    auto collapse = inner.getDefiningOp<tensor::CollapseShapeOp>();
    if (collapse && !inner.hasOneUse()) {
      return failure();
    }
    Value padded = collapse ? collapse.getSrc() : inner;
    auto pad = padded.getDefiningOp<tensor::PadOp>();
    if (!pad || !padded.hasOneUse()) {
      return failure();
    }
    auto padType = cast<RankedTensorType>(pad.getResult().getType());
    int64_t shift = collapse ? 1 : 0;
    if (padType.getRank() != 4 + shift || !padType.hasStaticShape()) {
      return failure();
    }
    const int64_t spatial[2] = {1 + shift, 2 + shift};
    int64_t padHeight = 0;
    int64_t padWidth = 0;
    bool int8Fill = false;
    if (!matchSymmetricSpatialPad(pad, spatial, padHeight, padWidth, int8Fill) || int8Fill ||
        !isClaimablePaddedDepthwise(conv, padHeight, padWidth)) {
      return failure();
    }
    Value fill = pad.getConstantPaddingValue();
    if (!fill) {
      return failure();
    }

    Location loc = conv.getLoc();
    Value source = pad.getSource();
    if (collapse) {
      source = tensor::CollapseShapeOp::create(rewriter, loc, source,
                                               collapse.getReassociationIndices());
    }
    auto sourceType = cast<RankedTensorType>(source.getType());
    SmallVector<int64_t> nchw = {sourceType.getDimSize(0), sourceType.getDimSize(3),
                                 sourceType.getDimSize(1), sourceType.getDimSize(2)};
    Value init = tensor::EmptyOp::create(rewriter, loc, nchw, sourceType.getElementType());
    Value transposed = linalg::TransposeOp::create(rewriter, loc, source, init,
                                                   transpose.getPermutation())
                           ->getResult(0);
    SmallVector<OpFoldResult> amounts = {rewriter.getIndexAttr(0), rewriter.getIndexAttr(0),
                                         rewriter.getIndexAttr(padHeight),
                                         rewriter.getIndexAttr(padWidth)};
    Value newPad = tensor::PadOp::create(rewriter, loc, transpose->getResult(0).getType(),
                                         transposed, amounts, amounts, fill);
    rewriter.modifyOpInPlace(conv, [&] { conv->setOperand(0, newPad); });
    return success();
  }
};

template <typename ConvOp, int64_t SpatialH, int64_t SpatialW>
struct MarkPaddedConv : public OpRewritePattern<ConvOp> {
  MarkPaddedConv(MLIRContext *context, llvm::SetVector<Operation *> *claimed)
      : OpRewritePattern<ConvOp>(context), claimed(claimed) {}

  llvm::SetVector<Operation *> *claimed;

  LogicalResult matchAndRewrite(ConvOp conv,
                                PatternRewriter &rewriter) const override {
    if (conv->hasAttr(kPadTopAttrName)) {
      return failure();
    }
    // A convolution `rocket-fuse-conv-relu6` has already claimed is left
    // alone. This pass runs after it (the sink below has to be the last thing
    // before the match loop), and marking such a convolution would change the
    // attribute dictionary the ReLU6 template compares -- costing the
    // activation fusion and gaining nothing, because no template claims a pad
    // and a clamp together. So a convolution with both folds the ReLU6 and
    // keeps its pad materialized.
    //
    // **That tie-break used to be free and is not any more.** It read "no
    // measured model has both: MobileNetV2's padded convolutions are its
    // depthwise ones and ResNet50's activation is an unbounded ReLU". Since
    // ReLU6 fusion started firing on an already-f16 import, MobileNetV2's
    // seventeen depthwise convolutions have *both* -- they fuse the clamp
    // here and keep their pad on the CPU, which is seventeen `slow_memcpy`
    // dispatches plus seventeen whole-buffer zero fills, about 16 MB of
    // traffic per inference and the largest single item left on the CPU for
    // that model.
    //
    // Nothing regressed -- those pads never folded, because every `pad1`
    // executable in the transform spec is a *dense* one
    // (`rocket_dynamic_conv2d_v1`) and no depthwise padded target or matcher
    // exists at all. But it does set the shape of the fix: a depthwise padded
    // path has to be a **pad + ReLU6** template, not a bare pad one, or this
    // branch will decline every site it was built for and the work will
    // silently buy nothing.
    //
    // **The depthwise half of that is now built** (2026-11-24):
    // `@match_dynamic_depthwise_conv2d_nchw_pad1_relu6*` claim the pad and
    // the clamp together, so a ReLU6 depthwise convolution is marked after
    // all -- but only in the form those claim (`isClaimablePaddedDepthwise`),
    // checked below once the pad is known.
    //
    // **And the dense half** (2026-11-24): `@match_dynamic_conv2d_3x3_pad1_relu6*`
    // do the same for a dense 3x3 at stride 1 or 2 (`isClaimablePaddedDense`) --
    // MobileNetV2's stem, whose pad and clamp both ran on the CPU. Only the
    // NHWC depthwise form, which no padded ReLU6 template covers, still
    // keeps its pad.
    const bool fusedClamp = feedsFusedClamp(conv);
    constexpr bool kNchwDepthwise =
        std::is_same_v<ConvOp, linalg::DepthwiseConv2DNchwChwOp>;
    constexpr bool kNhwcDense = std::is_same_v<ConvOp, linalg::Conv2DNhwcHwcfOp>;
    if (fusedClamp && !kNchwDepthwise && !kNhwcDense) {
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
    bool int8Fill = false;
    if (!matchSymmetricSpatialPad(pad, spatial, padHeight, padWidth,
                                  int8Fill)) {
      return failure();
    }
    // An int8 fill is marked only where a target claims it: a dense 3x3 at
    // pad 1 whose result feeds the requantization epilogue
    // `rocket-fuse-int8-requant-epilogue` produced -- the one form the pad1
    // matchers claim. A mark is an attribute every DAG matcher compares, so
    // marking a convolution no pad1 matcher can take only risks costing it
    // the unpadded matcher that would have claimed it. (The int8_accumulator
    // matchers are not DAG-based and ignore the mark; a -128-padded 3x3
    // whose bias did not reach the requantized form stays theirs either
    // way.)
    if (int8Fill) {
      if (!std::is_same_v<ConvOp, linalg::Conv2DNhwcHwcfOp> ||
          padHeight != 1 || padWidth != 1) {
        return failure();
      }
      auto filterType =
          cast<RankedTensorType>(conv.getDpsInputs()[1].getType());
      if (filterType.getRank() != 4 || filterType.getDimSize(0) != 3 ||
          filterType.getDimSize(1) != 3) {
        return failure();
      }
      if (!feedsRequantEpilogue(conv->getResult(0))) {
        return failure();
      }
    }
    // A ReLU6 depthwise convolution is marked only in the one form a padded
    // template claims; anything else keeps its plain ReLU6 matcher and its
    // materialized pad, as every ReLU6 convolution did before.
    if constexpr (kNchwDepthwise) {
      if (fusedClamp && !isClaimablePaddedDepthwise(conv, padHeight, padWidth)) {
        return failure();
      }
    }
    // The dense pad-1 ReLU6 templates spell the channels-last form, a rank-5
    // pad and its collapse (`shift == 1`). A rank-4 NHWC pad would be marked
    // and then claimed by nothing -- and the mark costs the plain ReLU6
    // matcher that claims it today, sending the convolution to the CPU.
    if constexpr (kNhwcDense) {
      if (fusedClamp &&
          (int8Fill || shift != 1 || !isClaimablePaddedDense(conv, padHeight, padWidth))) {
        return failure();
      }
    }
    // The region is made self-contained after the greedy driver finishes --
    // see the file comment. Record the pad for that second step.
    claimed->insert(pad);

    rewriter.modifyOpInPlace(conv, [&] {
      conv->setAttr(kPadTopAttrName, rewriter.getI64IntegerAttr(padHeight));
      conv->setAttr(kPadLeftAttrName, rewriter.getI64IntegerAttr(padWidth));
    });
    return success();
  }
};

/// The int8 max pool `rocket-fuse-int8-max-pool` produces, reading a
/// symmetric spatial pad filled with -128: marked the same way a convolution
/// is, so `@match_pooling_nhwc_max_s2_i8_pad1` can hand the padding to the
/// PPU.
///
/// -128 is the int8 max's identity, so the padded pool computes exactly the
/// clamp-to-image answer -- which is what the PPU's int8 max produces for
/// every fill (`int8_max_pad_fill_probe` in iree-rocket-hal: the fill is
/// inert). That equivalence needs every window to hold an image pixel, i.e.
/// pad < kernel on each axis, and the PPU hangs otherwise, so a pad as wide
/// as the window is left alone. The PPU's pad fields are 3 bits.
struct MarkPaddedInt8MaxPool
    : public OpRewritePattern<linalg::PoolingNhwcMaxOp> {
  MarkPaddedInt8MaxPool(MLIRContext *context,
                        llvm::SetVector<Operation *> *claimed)
      : OpRewritePattern<linalg::PoolingNhwcMaxOp>(context),
        claimed(claimed) {}

  llvm::SetVector<Operation *> *claimed;

  LogicalResult matchAndRewrite(linalg::PoolingNhwcMaxOp pool,
                                PatternRewriter &rewriter) const override {
    if (pool->hasAttr(kPadTopAttrName)) {
      return failure();
    }
    auto pad = pool.getDpsInputs()[0].getDefiningOp<tensor::PadOp>();
    if (!pad) {
      return failure();
    }
    auto padType = cast<RankedTensorType>(pad.getResult().getType());
    auto windowType =
        dyn_cast<RankedTensorType>(pool.getDpsInputs()[1].getType());
    if (padType.getRank() != 4 || !padType.getElementType().isInteger(8) ||
        !windowType || windowType.getRank() != 2 ||
        windowType.isDynamicDim(0) || windowType.isDynamicDim(1)) {
      return failure();
    }
    int64_t padHeight = 0;
    int64_t padWidth = 0;
    bool int8Fill = false;
    const int64_t spatial[2] = {1, 2};
    if (!matchSymmetricSpatialPad(pad, spatial, padHeight, padWidth,
                                  int8Fill) ||
        !int8Fill) {
      return failure();
    }
    if (padHeight > kMaxPoolPad || padWidth > kMaxPoolPad ||
        padHeight >= windowType.getDimSize(0) ||
        padWidth >= windowType.getDimSize(1)) {
      return failure();
    }
    claimed->insert(pad);
    rewriter.modifyOpInPlace(pool, [&] {
      pool->setAttr(kPadTopAttrName, rewriter.getI64IntegerAttr(padHeight));
      pool->setAttr(kPadLeftAttrName, rewriter.getI64IntegerAttr(padWidth));
    });
    return success();
  }
};

/// A symmetric spatial -inf pad a float max pool can hand to the PPU: the
/// NHWC tensor the pool reads, traced through the rank-5 `1x1xHxWxC`
/// channels-last form's `collapse_shape` when present.
struct FloatPoolPad {
  tensor::PadOp pad;
  int64_t height;
  int64_t width;
};

/// Matches `source` (an NHWC pool input) against a -inf pad below `window`
/// and within the PPU's 3-bit pad fields.
static std::optional<FloatPoolPad> matchFloatPoolPad(Value source,
                                                     RankedTensorType window) {
  if (!window || window.getRank() != 2 || window.isDynamicDim(0) ||
      window.isDynamicDim(1)) {
    return std::nullopt;
  }
  int64_t shift = 0;
  if (auto collapse = source.getDefiningOp<tensor::CollapseShapeOp>()) {
    source = collapse.getSrc();
    shift = 1;
  }
  auto pad = source.getDefiningOp<tensor::PadOp>();
  if (!pad) {
    return std::nullopt;
  }
  auto padType = cast<RankedTensorType>(pad.getResult().getType());
  if (padType.getRank() != 4 + shift ||
      !isa<FloatType>(padType.getElementType())) {
    return std::nullopt;
  }
  ArrayRef<int64_t> low = pad.getStaticLow();
  ArrayRef<int64_t> high = pad.getStaticHigh();
  if (!pad.getLow().empty() || !pad.getHigh().empty() ||
      low.size() != high.size()) {
    return std::nullopt;
  }
  const int64_t spatial[2] = {1 + shift, 2 + shift};
  for (int64_t dim = 0, rank = low.size(); dim < rank; ++dim) {
    bool isSpatial = llvm::is_contained(spatial, dim);
    if (low[dim] != high[dim] || (!isSpatial && low[dim] != 0)) {
      return std::nullopt;
    }
  }
  int64_t height = low[spatial[0]];
  int64_t width = low[spatial[1]];
  if ((height == 0 && width == 0) || height > kMaxPoolPad ||
      width > kMaxPoolPad || height >= window.getDimSize(0) ||
      width >= window.getDimSize(1)) {
    return std::nullopt;
  }
  auto yield = dyn_cast<tensor::YieldOp>(pad.getRegion().front().getTerminator());
  APFloat fill(0.0f);
  if (!yield || !matchPattern(yield.getValue(), m_ConstantFloat(&fill)) ||
      !fill.isInfinity() || !fill.isNegative()) {
    return std::nullopt;
  }
  return FloatPoolPad{pad, height, width};
}

/// The fp16 path's max pool: an f32 `pooling_nhwc_max` reading a symmetric
/// spatial pad filled with -inf, either directly or -- as a ResNet stem
/// arrives after the channels-last conversion -- through the rank-5
/// `1x1xHxWxC` pad and a `collapse_shape`. Marked the same way, so
/// @match_pooling_nhwc_max_s2_pad1 can hand the padding to the PPU.
///
/// The PPU's max ignores its pad fill at fp16 and clamps every window to the
/// image (`fp16_max_pad_fill_probe` in iree-rocket-hal), and a -inf pad never
/// wins a max either, so the two agree exactly wherever every window holds
/// an image pixel -- pad < kernel, checked here. The PPU's pad fields are 3
/// bits.
struct MarkPaddedFloatMaxPool
    : public OpRewritePattern<linalg::PoolingNhwcMaxOp> {
  MarkPaddedFloatMaxPool(MLIRContext *context,
                         llvm::SetVector<Operation *> *claimed)
      : OpRewritePattern<linalg::PoolingNhwcMaxOp>(context),
        claimed(claimed) {}

  llvm::SetVector<Operation *> *claimed;

  LogicalResult matchAndRewrite(linalg::PoolingNhwcMaxOp pool,
                                PatternRewriter &rewriter) const override {
    if (pool->hasAttr(kPadTopAttrName)) {
      return failure();
    }
    auto inputType =
        dyn_cast<RankedTensorType>(pool.getDpsInputs()[0].getType());
    if (!inputType || !isa<FloatType>(inputType.getElementType())) {
      return failure();
    }
    std::optional<FloatPoolPad> found = matchFloatPoolPad(
        pool.getDpsInputs()[0],
        dyn_cast<RankedTensorType>(pool.getDpsInputs()[1].getType()));
    if (!found) {
      return failure();
    }
    claimed->insert(found->pad);
    rewriter.modifyOpInPlace(pool, [&] {
      pool->setAttr(kPadTopAttrName, rewriter.getI64IntegerAttr(found->height));
      pool->setAttr(kPadLeftAttrName, rewriter.getI64IntegerAttr(found->width));
    });
    return success();
  }
};

/// The same pool as the ONNX importer leaves it: `pooling_nchw_max` over a
/// `[0, 3, 1, 2]` transpose of that padded NHWC tensor (the channels-last
/// conversion moves convolutions, not pools). Rewritten to the NHWC pool on
/// the NHWC tensor -- marked like the one above -- and a transpose of its
/// result back to NCHW, which cancels against the transpose to NHWC the
/// next convolution starts with. Only pools whose pad the PPU can take are
/// touched; every other NCHW pool keeps its own matcher.
struct MarkPaddedFloatNchwMaxPool
    : public OpRewritePattern<linalg::PoolingNchwMaxOp> {
  MarkPaddedFloatNchwMaxPool(MLIRContext *context,
                             llvm::SetVector<Operation *> *claimed)
      : OpRewritePattern<linalg::PoolingNchwMaxOp>(context),
        claimed(claimed) {}

  llvm::SetVector<Operation *> *claimed;

  LogicalResult matchAndRewrite(linalg::PoolingNchwMaxOp pool,
                                PatternRewriter &rewriter) const override {
    auto transpose = pool.getDpsInputs()[0].getDefiningOp<linalg::TransposeOp>();
    if (!transpose ||
        transpose.getPermutation() != ArrayRef<int64_t>{0, 3, 1, 2}) {
      return failure();
    }
    Value nhwc = transpose.getInput();
    auto nhwcType = dyn_cast<RankedTensorType>(nhwc.getType());
    auto windowType =
        dyn_cast<RankedTensorType>(pool.getDpsInputs()[1].getType());
    auto initType = dyn_cast<RankedTensorType>(pool.getDpsInits()[0].getType());
    if (!nhwcType || !isa<FloatType>(nhwcType.getElementType()) ||
        !initType || !initType.hasStaticShape()) {
      return failure();
    }
    std::optional<FloatPoolPad> found = matchFloatPoolPad(nhwc, windowType);
    if (!found) {
      return failure();
    }
    // The init is a -inf fill (the reduction's identity); rebuild it NHWC.
    auto fill = pool.getDpsInits()[0].getDefiningOp<linalg::FillOp>();
    APFloat initValue(0.0f);
    if (!fill || !matchPattern(fill.getInputs()[0], m_ConstantFloat(&initValue)) ||
        !initValue.isInfinity() || !initValue.isNegative()) {
      return failure();
    }
    Location loc = pool.getLoc();
    SmallVector<int64_t> nhwcOut = {initType.getDimSize(0),
                                    initType.getDimSize(2),
                                    initType.getDimSize(3),
                                    initType.getDimSize(1)};
    Value empty = tensor::EmptyOp::create(rewriter, loc, nhwcOut,
                                          initType.getElementType());
    Value init = linalg::FillOp::create(rewriter, loc,
                                        ValueRange{fill.getInputs()[0]},
                                        ValueRange{empty})
                     .getResult(0);
    auto nhwcPool = linalg::PoolingNhwcMaxOp::create(
        rewriter, loc, TypeRange{init.getType()},
        ValueRange{nhwc, pool.getDpsInputs()[1]}, ValueRange{init},
        pool.getStridesAttr(), pool.getDilationsAttr());
    nhwcPool->setAttr(kPadTopAttrName, rewriter.getI64IntegerAttr(found->height));
    nhwcPool->setAttr(kPadLeftAttrName, rewriter.getI64IntegerAttr(found->width));
    claimed->insert(found->pad);
    Value back = tensor::EmptyOp::create(rewriter, loc, initType.getShape(),
                                         initType.getElementType());
    rewriter.replaceOpWithNewOp<linalg::TransposeOp>(
        pool, nhwcPool.getResult(0), back, ArrayRef<int64_t>{0, 3, 1, 2});
    return success();
  }
};

struct RocketFoldConvPadPass
    : public PassWrapper<RocketFoldConvPadPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RocketFoldConvPadPass)

  StringRef getArgument() const final { return "rocket-fold-conv-pad"; }
  StringRef getDescription() const final {
    return "Marks a convolution whose input is a symmetric, zero-filled "
           "(or, for an int8 3x3, -128-filled), spatial-only tensor.pad "
           "with rocket.pad_top/rocket.pad_left, so the matcher can hand the "
           "padding to the CNA instead of leaving it as a materialized copy. "
           "Also marks an int8 max pool reading a -128-filled pad, for the "
           "PPU.";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() final {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    llvm::SetVector<Operation *> claimed;
    // NHWC puts height and width at 1 and 2; the NCHW depthwise form at 2
    // and 3.
    patterns.add<SinkDepthwisePadBelowTranspose>(context);
    patterns.add<MarkPaddedConv<linalg::Conv2DNhwcHwcfOp, 1, 2>,
                 MarkPaddedConv<linalg::DepthwiseConv2DNhwcHwcOp, 1, 2>,
                 MarkPaddedConv<linalg::DepthwiseConv2DNchwChwOp, 2, 3>,
                 MarkPaddedInt8MaxPool, MarkPaddedFloatMaxPool,
                 MarkPaddedFloatNchwMaxPool>(context, &claimed);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }

    // Make each claimed pad's region self-contained, now that the driver's
    // constant folder is done and cannot hoist it back out.
    for (Operation *op : claimed) {
      auto pad = cast<tensor::PadOp>(op);
      auto yield =
          cast<tensor::YieldOp>(pad.getRegion().front().getTerminator());
      Operation *fill = yield.getValue().getDefiningOp();
      if (fill && fill->getBlock() == &pad.getRegion().front()) {
        continue;
      }
      // A max pool's -inf (or any constant fill a pattern above verified)
      // is sunk as itself.
      if (auto constant = dyn_cast_or_null<arith::ConstantOp>(fill)) {
        OpBuilder builder(yield);
        Value sunk = arith::ConstantOp::create(builder, pad.getLoc(),
                                               constant.getValue());
        yield.getValueMutable().assign(sunk);
        continue;
      }
      Type elementType =
          cast<RankedTensorType>(pad.getResult().getType()).getElementType();
      OpBuilder builder(yield);
      TypedAttr fillAttr;
      if (auto floatType = dyn_cast<FloatType>(elementType)) {
        fillAttr = builder.getFloatAttr(floatType, 0.0);
      } else if (elementType.isInteger(8)) {
        fillAttr = builder.getIntegerAttr(elementType, kInt8PadFill);
      } else {
        continue;
      }
      Value sunk =
          arith::ConstantOp::create(builder, pad.getLoc(), fillAttr);
      yield.getValueMutable().assign(sunk);
    }
  }
};

static PassRegistration<RocketFoldConvPadPass> reg;

} // namespace
} // namespace mlir::iree_compiler::IREE::HAL
