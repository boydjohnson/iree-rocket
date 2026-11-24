// Copyright 2026
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Rewrites a "patchify" convolution -- stride equal to the kernel, so the
// windows tile the image without overlapping -- into the matmul it is.
//
// A vision transformer's patch embedding is a 16x16 stride-16 convolution
// (ViT-B/16, BLIP; CLIP's is 32x32), and ConvNeXt's stem (4x4 stride 4) and
// its three downsamples (2x2 stride 2) are the same thing at smaller sizes.
// The planner refuses every one of them: kernels above 11 have no vendor
// reference data, and even kernels have capture backing only at stride 1
// (model_survey, 2026-11-24). None of that matters, because nothing about
// these convolutions needs a convolution engine. Each output pixel reads its
// own disjoint `kh x kw x C` patch, so
//
//   out[n, oh, ow, f] = sum_{i, j, c} in[n, oh*kh + i, ow*kw + j, c]
//                                     * w[i, j, c, f]
//
// is a `[N*OH*OW, kh*kw*C] x [kh*kw*C, F]` matmul once the input is
// rearranged patch-major (space-to-depth): expand `H` to `OH x kh` and `W` to
// `OW x kw`, move `kh` next to `kw`, collapse. The filter needs no movement
// at all -- HWCF is already `(kh, kw, C)`-major over `F`. The matmul then goes
// through the same demotion, planning and `@match_rocket_matmul` as any
// other, with the row-major output and input the driver already has.
//
// The rearrangement is one CPU transpose over the input, which for a patch
// embedding is the image itself (BLIP's 384x384x3 fp16 is 0.9 MB) and which
// IREE fuses with the channels-last transpose that precedes it.
//
// Rows or columns a floor-mode convolution never reads (H not a multiple of
// the stride) are cropped first. Only static shapes, unit dilation and a
// kernel larger than 1x1 are rewritten; a 1x1 convolution is already the
// matmul the planner handles.

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::HAL {
namespace {

struct PatchifyConv : OpRewritePattern<linalg::Conv2DNhwcHwcfOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::Conv2DNhwcHwcfOp conv,
                                PatternRewriter &rewriter) const override {
    if (!conv.hasPureTensorSemantics()) {
      return rewriter.notifyMatchFailure(conv, "not on tensors");
    }
    Value input = conv.getDpsInputs()[0];
    Value filter = conv.getDpsInputs()[1];
    Value init = conv.getDpsInits()[0];
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto filterType = dyn_cast<RankedTensorType>(filter.getType());
    auto initType = dyn_cast<RankedTensorType>(init.getType());
    if (!inputType || !filterType || !initType ||
        !inputType.hasStaticShape() || !filterType.hasStaticShape() ||
        !initType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(conv, "dynamic shape");
    }

    int64_t kh = filterType.getDimSize(0);
    int64_t kw = filterType.getDimSize(1);
    int64_t channels = filterType.getDimSize(2);
    int64_t features = filterType.getDimSize(3);
    if (kh * kw <= 1) {
      return rewriter.notifyMatchFailure(conv, "1x1 is already a matmul");
    }
    SmallVector<int64_t> strides(conv.getStrides().getValues<int64_t>());
    SmallVector<int64_t> dilations(conv.getDilations().getValues<int64_t>());
    if (strides.size() != 2 || strides[0] != kh || strides[1] != kw) {
      return rewriter.notifyMatchFailure(conv, "stride is not the kernel");
    }
    if (dilations.size() != 2 || dilations[0] != 1 || dilations[1] != 1) {
      return rewriter.notifyMatchFailure(conv, "dilated");
    }

    int64_t batch = inputType.getDimSize(0);
    int64_t height = inputType.getDimSize(1);
    int64_t width = inputType.getDimSize(2);
    int64_t outHeight = initType.getDimSize(1);
    int64_t outWidth = initType.getDimSize(2);
    if (inputType.getDimSize(3) != channels ||
        initType.getDimSize(0) != batch || initType.getDimSize(3) != features ||
        outHeight != height / kh || outWidth != width / kw || outHeight == 0 ||
        outWidth == 0) {
      return rewriter.notifyMatchFailure(conv, "inconsistent extents");
    }

    Location loc = conv.getLoc();
    Type inputElement = inputType.getElementType();

    // Crop what a floor-mode convolution never reads.
    if (height != outHeight * kh || width != outWidth * kw) {
      SmallVector<OpFoldResult> offsets(4, rewriter.getIndexAttr(0));
      SmallVector<OpFoldResult> sizes = {
          rewriter.getIndexAttr(batch), rewriter.getIndexAttr(outHeight * kh),
          rewriter.getIndexAttr(outWidth * kw),
          rewriter.getIndexAttr(channels)};
      SmallVector<OpFoldResult> unit(4, rewriter.getIndexAttr(1));
      input = tensor::ExtractSliceOp::create(rewriter, loc, input, offsets,
                                             sizes, unit);
    }

    // [N, OH*kh, OW*kw, C] -> [N, OH, kh, OW, kw, C]
    auto patchedType = RankedTensorType::get(
        {batch, outHeight, kh, outWidth, kw, channels}, inputElement);
    Value patched = tensor::ExpandShapeOp::create(
        rewriter, loc, patchedType, input,
        SmallVector<ReassociationIndices>{{0}, {1, 2}, {3, 4}, {5}});

    // -> [N, OH, OW, kh, kw, C]
    SmallVector<int64_t> patchMajorShape = {batch, outHeight, outWidth,
                                            kh,    kw,        channels};
    Value empty = tensor::EmptyOp::create(rewriter, loc, patchMajorShape,
                                          inputElement);
    Value patchMajor =
        linalg::TransposeOp::create(rewriter, loc, patched, empty,
                                    ArrayRef<int64_t>{0, 1, 3, 2, 4, 5})
            .getResult()[0];

    // -> [N*OH*OW, kh*kw*C]
    int64_t rows = batch * outHeight * outWidth;
    int64_t depth = kh * kw * channels;
    Value lhs = tensor::CollapseShapeOp::create(
        rewriter, loc, RankedTensorType::get({rows, depth}, inputElement),
        patchMajor, SmallVector<ReassociationIndices>{{0, 1, 2}, {3, 4, 5}});

    // HWCF is already (kh, kw, C)-major: [kh*kw*C, F] with no movement.
    Value rhs = tensor::CollapseShapeOp::create(
        rewriter, loc,
        RankedTensorType::get({depth, features}, filterType.getElementType()),
        filter, SmallVector<ReassociationIndices>{{0, 1, 2}, {3}});
    Value acc = tensor::CollapseShapeOp::create(
        rewriter, loc,
        RankedTensorType::get({rows, features}, initType.getElementType()),
        init, SmallVector<ReassociationIndices>{{0, 1, 2}, {3}});

    Value product =
        linalg::MatmulOp::create(rewriter, loc, TypeRange{acc.getType()},
                                 ValueRange{lhs, rhs}, ValueRange{acc})
            .getResult(0);
    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
        conv, initType, product,
        SmallVector<ReassociationIndices>{{0, 1, 2}, {3}});
    return success();
  }
};

struct RocketPatchifyConvPass
    : public PassWrapper<RocketPatchifyConvPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RocketPatchifyConvPass)

  StringRef getArgument() const final {
    return "rocket-patchify-conv-to-matmul";
  }
  StringRef getDescription() const final {
    return "Rewrites an NHWC convolution whose stride equals its kernel (a "
           "patch embedding or a non-overlapping downsample) into a "
           "space-to-depth transpose and a linalg.matmul, which the Rocket "
           "matmul path claims.";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect>();
  }

  void runOnOperation() final {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<PatchifyConv>(context);
    // The rewrite is all this pass does; folding and canonicalization are
    // left to the pipeline's own passes later on.
    GreedyRewriteConfig config;
    config.enableFolding(false);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      return signalPassFailure();
    }
  }
};

static PassRegistration<RocketPatchifyConvPass> reg;

} // namespace
} // namespace mlir::iree_compiler::IREE::HAL
