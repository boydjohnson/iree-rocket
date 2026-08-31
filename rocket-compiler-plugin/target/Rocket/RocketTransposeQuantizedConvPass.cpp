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
    Value nhwcInput = transposeTo(rewriter, loc, input, {0, 2, 3, 1});
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

struct RocketTransposeQuantizedConvPass
    : public PassWrapper<RocketTransposeQuantizedConvPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RocketTransposeQuantizedConvPass)

  StringRef getArgument() const final {
    return "rocket-transpose-quantized-conv-to-nhwc";
  }
  StringRef getDescription() const final {
    return "Rewrites linalg.conv_2d_nchw_fchw_q as linalg.conv_2d_nhwc_hwcf_q "
           "so the upstream quantized-conv-to-conv fold can reach it.";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect>();
  }

  void runOnOperation() final {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<TransposeQuantizedConvToNhwc>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

static PassRegistration<RocketTransposeQuantizedConvPass> reg;

} // namespace
} // namespace mlir::iree_compiler::IREE::HAL
