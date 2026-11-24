// Copyright 2026
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Rewrites a floating-point 7x7 stride-2 image stem into a 4x4 stride-1
// convolution over a space-to-depth-folded input, the shape the NPU runs well.
//
// ResNet's stem convolves a 3-channel image with a 7x7 kernel at stride 2.
// On the NPU that is correct but slow: at `Cin` 3 the CNA reads the image
// through its dense (ARGB) path and most of the MAC array idles -- ~4 ms per
// dispatch at 600 MHz against ~2 ms for the CPU, so the fp16 stem stays on
// the CPU (ISSUES.md "fp16 7x7 stride-2 stem").
//
// Stride 2 over an image is stride 1 over the image folded 2x2 into
// channels. With `s[y, x, (dy, dx, c)] = in[2y + dy, 2x + dx, c]` and the
// kernel zero-padded from 7x7 to 8x8 and regrouped the same way,
// `w'[a, b, (dy, dx, c), f] = w[2a + dy, 2b + dx, c, f]`,
//
//   out[oh, ow, f] = sum_{ky, kx < 8, c} in[2 oh + ky, 2 ow + kx, c] w[ky, kx, c, f]
//                  = sum_{a, b < 4, dy, dx, c} s[oh + a, ow + b, (dy, dx, c)]
//                                              * w'[a, b, (dy, dx, c), f]
//
// -- a 4x4 stride-1 convolution over `4 * Cin` channels with the same output.
// The padded kernel row and column are zero, so the extra input row and
// column they would read contribute nothing. At `Cin` 3 the folded input has
// 12 channels, which the CNA reads as feature surfaces: measured 1.5 ms per
// dispatch against 4.0 ms for the direct stem, exact
// (`stem_space_to_depth_4x4_matches_oracle`).
//
// The fold is a CPU transpose of the image (230x230x3, a few hundred KB) that
// IREE fuses with the padding and channels-last transposes in front of it;
// the kernel regrouping operates on a constant and is folded at compile time.
//
// Narrow on purpose: floating point (int8 stems have their own requantized
// path), a 7x7 kernel, stride 2, unit dilation, `Cin` <= 4 (the dense image
// path this exists to avoid) and static shapes. Rows and columns a
// floor-mode convolution never reads are cropped; an input one short of the
// padded kernel's reach is zero-padded.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::HAL {
namespace {

constexpr int64_t kKernel = 7;
constexpr int64_t kStride = 2;
constexpr int64_t kMaxChannels = 4;

/// Zero-pads `value` at the high end of every dimension by `high`.
static Value padHighWithZero(PatternRewriter &rewriter, Location loc,
                             Value value, ArrayRef<int64_t> high) {
  auto type = cast<RankedTensorType>(value.getType());
  SmallVector<int64_t> shape(type.getShape());
  SmallVector<OpFoldResult> low, highFold;
  for (auto [index, extra] : llvm::enumerate(high)) {
    shape[index] += extra;
    low.push_back(rewriter.getIndexAttr(0));
    highFold.push_back(rewriter.getIndexAttr(extra));
  }
  Value zero = arith::ConstantOp::create(
      rewriter, loc, rewriter.getZeroAttr(type.getElementType()));
  return tensor::PadOp::create(
             rewriter, loc,
             RankedTensorType::get(shape, type.getElementType()), value, low,
             highFold, zero)
      .getResult();
}

struct SpaceToDepthStem : OpRewritePattern<linalg::Conv2DNhwcHwcfOp> {
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
    if (!isa<FloatType>(inputType.getElementType()) ||
        !isa<FloatType>(filterType.getElementType())) {
      return rewriter.notifyMatchFailure(conv, "not floating point");
    }
    int64_t channels = filterType.getDimSize(2);
    int64_t features = filterType.getDimSize(3);
    if (filterType.getDimSize(0) != kKernel ||
        filterType.getDimSize(1) != kKernel || channels > kMaxChannels) {
      return rewriter.notifyMatchFailure(conv, "not a narrow 7x7 stem");
    }
    SmallVector<int64_t> strides(conv.getStrides().getValues<int64_t>());
    SmallVector<int64_t> dilations(conv.getDilations().getValues<int64_t>());
    if (strides != SmallVector<int64_t>{kStride, kStride} ||
        dilations != SmallVector<int64_t>{1, 1}) {
      return rewriter.notifyMatchFailure(conv, "not stride 2, dilation 1");
    }

    int64_t batch = inputType.getDimSize(0);
    int64_t outHeight = initType.getDimSize(1);
    int64_t outWidth = initType.getDimSize(2);
    // The padded 8x8 kernel reaches `2 * (out - 1) + 8` rows; the folded
    // input has half that.
    constexpr int64_t kPadded = kKernel + 1;
    constexpr int64_t kFolded = kPadded / kStride;
    int64_t needHeight = kStride * (outHeight - 1) + kPadded;
    int64_t needWidth = kStride * (outWidth - 1) + kPadded;
    int64_t height = inputType.getDimSize(1);
    int64_t width = inputType.getDimSize(2);
    // The real kernel only reaches one row short of that, so the input must
    // cover at least `needHeight - 1`; anything less is not this
    // convolution's floor-mode extent.
    if (height < needHeight - 1 || width < needWidth - 1 ||
        inputType.getDimSize(3) != channels || initType.getDimSize(0) != batch ||
        initType.getDimSize(3) != features) {
      return rewriter.notifyMatchFailure(conv, "inconsistent extents");
    }

    Location loc = conv.getLoc();
    Type inputElement = inputType.getElementType();
    Type filterElement = filterType.getElementType();

    // Crop to the padded kernel's reach, or zero-pad the one row/column the
    // padded kernel reads with a zero weight.
    Value cropped = input;
    if (height > needHeight || width > needWidth) {
      SmallVector<OpFoldResult> offsets(4, rewriter.getIndexAttr(0));
      SmallVector<OpFoldResult> sizes = {
          rewriter.getIndexAttr(batch),
          rewriter.getIndexAttr(std::min(height, needHeight)),
          rewriter.getIndexAttr(std::min(width, needWidth)),
          rewriter.getIndexAttr(channels)};
      SmallVector<OpFoldResult> unit(4, rewriter.getIndexAttr(1));
      cropped = tensor::ExtractSliceOp::create(rewriter, loc, cropped, offsets,
                                               sizes, unit);
    }
    int64_t shortHeight = needHeight - std::min(height, needHeight);
    int64_t shortWidth = needWidth - std::min(width, needWidth);
    if (shortHeight || shortWidth) {
      cropped = padHighWithZero(rewriter, loc, cropped,
                                {0, shortHeight, shortWidth, 0});
    }

    // [N, 2 * FH, 2 * FW, C] -> [N, FH, 2, FW, 2, C] -> [N, FH, FW, 2, 2, C]
    //                       -> [N, FH, FW, 4C]
    int64_t foldedHeight = needHeight / kStride;
    int64_t foldedWidth = needWidth / kStride;
    Value split = tensor::ExpandShapeOp::create(
        rewriter, loc,
        RankedTensorType::get(
            {batch, foldedHeight, kStride, foldedWidth, kStride, channels},
            inputElement),
        cropped, SmallVector<ReassociationIndices>{{0}, {1, 2}, {3, 4}, {5}});
    Value foldedEmpty = tensor::EmptyOp::create(
        rewriter, loc,
        SmallVector<int64_t>{batch, foldedHeight, foldedWidth, kStride,
                             kStride, channels},
        inputElement);
    Value grouped =
        linalg::TransposeOp::create(rewriter, loc, split, foldedEmpty,
                                    ArrayRef<int64_t>{0, 1, 3, 2, 4, 5})
            .getResult()[0];
    int64_t foldedChannels = kStride * kStride * channels;
    Value folded = tensor::CollapseShapeOp::create(
        rewriter, loc,
        RankedTensorType::get(
            {batch, foldedHeight, foldedWidth, foldedChannels}, inputElement),
        grouped, SmallVector<ReassociationIndices>{{0}, {1}, {2}, {3, 4, 5}});

    // [7, 7, C, F] -> zero-pad to [8, 8, C, F] -> [4, 2, 4, 2, C, F]
    //              -> [4, 4, 2, 2, C, F] -> [4, 4, 4C, F]
    Value paddedFilter = padHighWithZero(rewriter, loc, filter, {1, 1, 0, 0});
    Value filterSplit = tensor::ExpandShapeOp::create(
        rewriter, loc,
        RankedTensorType::get(
            {kFolded, kStride, kFolded, kStride, channels, features},
            filterElement),
        paddedFilter,
        SmallVector<ReassociationIndices>{{0, 1}, {2, 3}, {4}, {5}});
    Value filterEmpty = tensor::EmptyOp::create(
        rewriter, loc,
        SmallVector<int64_t>{kFolded, kFolded, kStride, kStride, channels,
                             features},
        filterElement);
    Value filterGrouped =
        linalg::TransposeOp::create(rewriter, loc, filterSplit, filterEmpty,
                                    ArrayRef<int64_t>{0, 2, 1, 3, 4, 5})
            .getResult()[0];
    Value foldedFilter = tensor::CollapseShapeOp::create(
        rewriter, loc,
        RankedTensorType::get({kFolded, kFolded, foldedChannels, features},
                              filterElement),
        filterGrouped,
        SmallVector<ReassociationIndices>{{0}, {1}, {2, 3, 4}, {5}});

    // Same container type as the original's attributes (a `vector<2xi64>`
    // after the channels-last conversion): the DAG matchers compare
    // attribute dictionaries exactly.
    auto unitAttr = DenseIntElementsAttr::get(conv.getStrides().getType(),
                                              ArrayRef<int64_t>{1, 1});
    rewriter.replaceOpWithNewOp<linalg::Conv2DNhwcHwcfOp>(
        conv, TypeRange{initType}, ValueRange{folded, foldedFilter},
        ValueRange{init}, unitAttr, unitAttr);
    return success();
  }
};

struct RocketSpaceToDepthStemPass
    : public PassWrapper<RocketSpaceToDepthStemPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RocketSpaceToDepthStemPass)

  StringRef getArgument() const final { return "rocket-space-to-depth-stem"; }
  StringRef getDescription() const final {
    return "Rewrites a floating-point 7x7 stride-2 convolution over at most 4 "
           "channels (an image stem) into a 4x4 stride-1 convolution over the "
           "input folded 2x2 into channels, which the NPU runs on its feature "
           "path instead of the slow dense image path.";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() final {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<SpaceToDepthStem>(context);
    GreedyRewriteConfig config;
    config.enableFolding(false);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      return signalPassFailure();
    }
  }
};

static PassRegistration<RocketSpaceToDepthStemPass> reg;

} // namespace
} // namespace mlir::iree_compiler::IREE::HAL
