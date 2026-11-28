// Copyright 2026
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Transposes `linalg.conv_2d_nchw_fchw_q` into `linalg.conv_2d_nhwc_hwcf_q`,
// the one layout conversion nothing else in the pipeline can do.
//
// Two upstream passes each cover half of what an ONNX int8 model needs and
// neither covers this op:
//
//   * iree-preprocessing-convert-conv-to-channels-last has a named
//     NCHW->NHWC pattern for the *unquantized* linalg.conv_2d_nchw_fchw
//     only. On the quantized op it falls through to its generic
//     convolution-interface path, which generalizes to linalg.generic --
//     and a quantized conv's zero points are scalar i32 operands, which
//     linalg.generic cannot take ("operand #2 must be variadic of shaped of
//     any non-token type values, but got 'i32'"). That is a hard verifier
//     error, so before this pass existed an ONNX ConvInteger model did not
//     merely fail to offload, it failed to compile at all.
//
//   * iree-global-opt-quantized-conv-to-conv folds the zero points away
//     (conv_q(x, w, xz, 0) == conv(x, w) - xz * sum(w), exact in i32), which
//     is what turns these into ops the Rocket matchers can claim -- but its
//     patterns are Conv2DNhwcHwcfQOp and DepthwiseConv2DNhwcHwcQOp. The
//     depthwise ops iree-import-onnx produces are already NHWC and go
//     straight through; the dense ones arrive NCHW and are skipped.
//
// So this runs first, purely as a layout change, and the two upstream passes
// then do their jobs in order. It is deliberately a named-op-to-named-op
// rewrite: generalizing to linalg.generic is exactly the thing that does not
// work here.
//
// **A pad in front of the convolution moves to the NHWC side.** An ONNX
// QLinearConv's "same" padding arrives as an NCHW `tensor.pad` (filled with
// the input zero point) directly in front of the quantized conv. Left there,
// the transpose this pass adds lands *between* the pad and the conv, so
// `rocket-fold-conv-pad` cannot see the pad and IREE materializes it as
// three CPU dispatches per 3x3 conv -- transpose to NCHW, pad copy,
// transpose back. Transposing the pad's source instead and padding in NHWC
// is the same tensor (a constant pad commutes with a permutation), puts the
// pad next to the conv where the fold can hand it to the CNA, and leaves the
// transposes adjacent to the producer's own, where they cancel.
//
// **The same move for a quantized depthwise convolution.** The importer
// emits that op NHWC already, so there is no convolution to transpose, but
// its pad is still an NCHW one in front of the importer's NCHW->NHWC
// transpose. `SinkPadBelowQuantizedDepthwiseTranspose` swaps the two, which
// is what lets an int8 MobileNetV2's depthwise pads reach the CNA.
//
// **And a quantized matmul becomes a 1x1 convolution.** An ONNX
// `QLinearMatMul` (what `tools/quantize_onnx_int8.py --quantize-gemm` turns a
// classifier `Gemm` into) imports as `linalg.quantized_matmul` with the same
// scalar zero points and the same five-op requantization epilogue as a
// QLinearConv, and nothing downstream claims it: `quantized-conv-to-conv` has
// no matmul pattern and the int8 matchers are all convolutions. The `[M, K]`
// codes as the NHWC image `[1, 1, M, K]` and the `[K, N]` weight as the HWCF
// filter `[1, 1, K, N]` is exactly the convolution `fc.rs` lowers an FC to,
// so `QuantizedMatmulToConv` writes that, with a collapse back to `[M, N]`
// that `rocket-fuse-int8-requant-epilogue` looks through.

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::iree_compiler::IREE::HAL {
namespace {

// linalg.transpose's permutation is read as
// outputShape[i] == inputShape[permutation[i]].
Value transposeTo(PatternRewriter &rewriter, Location loc, Value source,
                  ArrayRef<int64_t> permutation) {
  auto sourceType = cast<RankedTensorType>(source.getType());
  SmallVector<OpFoldResult> sourceSizes =
      tensor::getMixedSizes(rewriter, loc, source);
  SmallVector<OpFoldResult> resultSizes;
  SmallVector<int64_t> resultShape;
  for (int64_t dim : permutation) {
    resultSizes.push_back(sourceSizes[dim]);
    resultShape.push_back(sourceType.getDimSize(dim));
  }
  Value empty = tensor::EmptyOp::create(rewriter, loc, resultSizes,
                                        sourceType.getElementType());
  return linalg::TransposeOp::create(rewriter, loc, source, empty, permutation)
      .getResult()[0];
}

/// `transposeTo(pad(x))` rebuilt as `pad(transposeTo(x))`, or null when the
/// pad is not one this can move: it must be static, filled with a single
/// constant, and read only by the convolution being rewritten (a second
/// reader would keep the NCHW pad alive next to the new one).
Value transposePadTo(PatternRewriter &rewriter, Location loc, Value input,
                     ArrayRef<int64_t> permutation) {
  auto pad = input.getDefiningOp<tensor::PadOp>();
  if (!pad || !pad->hasOneUse() || !pad.getLow().empty() ||
      !pad.getHigh().empty()) {
    return nullptr;
  }
  Value fill = pad.getConstantPaddingValue();
  if (!fill) {
    return nullptr;
  }
  ArrayRef<int64_t> low = pad.getStaticLow();
  ArrayRef<int64_t> high = pad.getStaticHigh();
  SmallVector<OpFoldResult> permutedLow, permutedHigh;
  SmallVector<int64_t> permutedShape;
  auto paddedType = cast<RankedTensorType>(pad.getResult().getType());
  for (int64_t dim : permutation) {
    permutedLow.push_back(rewriter.getIndexAttr(low[dim]));
    permutedHigh.push_back(rewriter.getIndexAttr(high[dim]));
    permutedShape.push_back(paddedType.getDimSize(dim));
  }
  Value source = transposeTo(rewriter, loc, pad.getSource(), permutation);
  auto resultType =
      RankedTensorType::get(permutedShape, paddedType.getElementType());
  return tensor::PadOp::create(rewriter, loc, resultType, source, permutedLow,
                               permutedHigh, fill, pad.getNofold())
      .getResult();
}

struct TransposeQuantizedConvToNhwc
    : OpRewritePattern<linalg::Conv2DNchwFchwQOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::Conv2DNchwFchwQOp convOp,
                                PatternRewriter &rewriter) const override {
    if (!convOp.hasPureTensorSemantics()) {
      return rewriter.notifyMatchFailure(convOp, "expected tensor semantics");
    }
    Location loc = convOp.getLoc();
    Value input = convOp.getDpsInputs()[0];
    Value filter = convOp.getDpsInputs()[1];
    Value inputZeroPoint = convOp.getDpsInputs()[2];
    Value filterZeroPoint = convOp.getDpsInputs()[3];
    Value init = convOp.getDpsInits()[0];
    if (!isa<RankedTensorType>(input.getType()) ||
        !isa<RankedTensorType>(filter.getType()) ||
        !isa<RankedTensorType>(init.getType())) {
      return rewriter.notifyMatchFailure(convOp, "expected ranked tensors");
    }

    // NCHW -> NHWC for the image operands, FCHW -> HWCF for the filter.
    Value nhwcInput = transposePadTo(rewriter, loc, input, {0, 2, 3, 1});
    if (!nhwcInput) {
      nhwcInput = transposeTo(rewriter, loc, input, {0, 2, 3, 1});
    }
    Value hwcfFilter = transposeTo(rewriter, loc, filter, {2, 3, 1, 0});
    Value nhwcInit = transposeTo(rewriter, loc, init, {0, 2, 3, 1});

    auto nhwcConv = linalg::Conv2DNhwcHwcfQOp::create(
        rewriter, loc, nhwcInit.getType(),
        ValueRange{nhwcInput, hwcfFilter, inputZeroPoint, filterZeroPoint},
        ValueRange{nhwcInit}, convOp.getStrides(), convOp.getDilations());

    // ...and back, so this pass is a pure layout change that leaves every
    // consumer untouched. The round-trip transposes are what the rest of the
    // pipeline (dispatch formation, and the Rocket transform spec's own
    // NC1HWC2 repacking) folds or absorbs.
    Value nchwResult =
        transposeTo(rewriter, loc, nhwcConv.getResult(0), {0, 3, 1, 2});
    rewriter.replaceOp(convOp, nchwResult);
    return success();
  }
};

/// The depthwise half of the pad move above. A quantized depthwise
/// convolution is already NHWC when the importer emits it, so no conv is
/// rewritten here -- but its "same" padding is still done in NCHW, in front
/// of the importer's own NCHW->NHWC transpose:
///
///   depthwise_conv_2d_nhwc_hwc_q(transpose[0, 2, 3, 1](pad_nchw(x)))
///
/// That leaves the transpose between the pad and the convolution, where
/// `rocket-fold-conv-pad` cannot see the pad, and IREE materializes it as a
/// CPU NCHW pad copy plus a transpose -- in front of every 3x3 depthwise in
/// an int8 MobileNetV2, so no expand -> depthwise edge could chain. This
/// rebuilds the chain as `pad_nhwc(transpose[0, 2, 3, 1](x))` (the same
/// tensor: a constant pad commutes with a permutation), so the pad is the
/// convolution's immediate producer and the transpose sits next to the
/// producer's own, where the pair cancels.
struct SinkPadBelowQuantizedDepthwiseTranspose
    : OpRewritePattern<linalg::DepthwiseConv2DNhwcHwcQOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::DepthwiseConv2DNhwcHwcQOp convOp,
                                PatternRewriter &rewriter) const override {
    if (!convOp.hasPureTensorSemantics()) {
      return rewriter.notifyMatchFailure(convOp, "expected tensor semantics");
    }
    Value input = convOp.getDpsInputs()[0];
    auto transpose = input.getDefiningOp<linalg::TransposeOp>();
    if (!transpose || !input.hasOneUse() ||
        transpose.getPermutation() != ArrayRef<int64_t>{0, 2, 3, 1}) {
      return rewriter.notifyMatchFailure(convOp,
                                         "input is not an NCHW->NHWC transpose");
    }
    Value nhwcInput = transposePadTo(rewriter, convOp.getLoc(),
                                     transpose.getInput(), {0, 2, 3, 1});
    if (!nhwcInput) {
      return rewriter.notifyMatchFailure(convOp, "no movable pad in front");
    }
    rewriter.modifyOpInPlace(convOp,
                             [&] { convOp->setOperand(0, nhwcInput); });
    return success();
  }
};

/// `quantized_matmul(a[M, K], b[K, N], za, zb)` as
/// `collapse(conv_2d_nhwc_hwcf_q(a[1, 1, M, K], b[1, 1, K, N], za, zb))`.
///
/// Pure metadata on every operand -- each expand adds two leading unit
/// dimensions -- and the same i32 contraction, so the zero points pass
/// through untouched and `quantized-conv-to-conv` folds them as it does a
/// convolution's. Static shapes only: every Rocket matcher wants them, and a
/// dynamic `M` would need an `output_shape` for nothing.
struct QuantizedMatmulToConv : OpRewritePattern<linalg::QuantizedMatmulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::QuantizedMatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    if (!matmulOp.hasPureTensorSemantics()) {
      return rewriter.notifyMatchFailure(matmulOp, "expected tensor semantics");
    }
    Value lhs = matmulOp.getDpsInputs()[0];
    Value rhs = matmulOp.getDpsInputs()[1];
    Value lhsZeroPoint = matmulOp.getDpsInputs()[2];
    Value rhsZeroPoint = matmulOp.getDpsInputs()[3];
    Value init = matmulOp.getDpsInits()[0];
    auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
    auto initType = dyn_cast<RankedTensorType>(init.getType());
    if (!lhsType || !rhsType || !initType || !lhsType.hasStaticShape() ||
        !rhsType.hasStaticShape() || !initType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(matmulOp, "expected static shapes");
    }
    if (!lhsType.getElementType().isSignlessInteger(8) ||
        !rhsType.getElementType().isSignlessInteger(8) ||
        !initType.getElementType().isSignlessInteger(32)) {
      return rewriter.notifyMatchFailure(matmulOp, "not i8 x i8 -> i32");
    }

    Location loc = matmulOp.getLoc();
    SmallVector<ReassociationIndices> leadingUnits{{0, 1, 2}, {3}};
    auto expand = [&](Value value, RankedTensorType type) -> Value {
      auto expanded = RankedTensorType::get(
          {1, 1, type.getDimSize(0), type.getDimSize(1)},
          type.getElementType());
      return tensor::ExpandShapeOp::create(rewriter, loc, expanded, value,
                                           leadingUnits)
          .getResult();
    };
    Value image = expand(lhs, lhsType);
    Value filter = expand(rhs, rhsType);
    Value accInit = expand(init, initType);

    auto unit = RankedTensorType::get({2}, rewriter.getIntegerType(64));
    auto ones = DenseIntElementsAttr::get(unit, ArrayRef<int64_t>{1, 1});
    auto conv = linalg::Conv2DNhwcHwcfQOp::create(
        rewriter, loc, accInit.getType(),
        ValueRange{image, filter, lhsZeroPoint, rhsZeroPoint},
        ValueRange{accInit}, ones, ones);
    rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
        matmulOp, initType, conv.getResult(0), leadingUnits);
    return success();
  }
};

struct RocketTransposeQuantizedConvPass
    : public PassWrapper<RocketTransposeQuantizedConvPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RocketTransposeQuantizedConvPass)

  StringRef getArgument() const final {
    return "rocket-transpose-quantized-conv-to-nhwc";
  }
  StringRef getDescription() const final {
    return "Rewrites linalg.conv_2d_nchw_fchw_q, and linalg.quantized_matmul "
           "as a 1x1 convolution, into linalg.conv_2d_nhwc_hwcf_q so the "
           "upstream quantized-conv-to-conv fold can reach them.";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect>();
  }

  void runOnOperation() final {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<TransposeQuantizedConvToNhwc,
                 SinkPadBelowQuantizedDepthwiseTranspose,
                 QuantizedMatmulToConv>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

static PassRegistration<RocketTransposeQuantizedConvPass> reg;

} // namespace
} // namespace mlir::iree_compiler::IREE::HAL
