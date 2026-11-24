// Copyright 2026
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Rewrites a grouped convolution into the dense convolution with a
// block-diagonal filter it equals, so the dense NPU path claims it.
//
// ResNeXt's 3x3 convolutions have 32 groups (4 to 32 channels each). The
// ONNX importer emits `linalg.conv_2d_ngchw_gfchw`, which nothing claims --
// no channels-last conversion, no matcher -- so all sixteen ran on the CPU
// in f32, ~125 ms of ResNeXt50's ~190 ms, and broke every NPU chain around
// them (3 NPU->NPU edges against ResNet50's 35).
//
// A grouped convolution is a dense one whose filter is zero wherever an
// output channel's group differs from an input channel's:
//
//   W[g*Fg + f, g'*Cg + c, y, x] = (g == g') ? w[g, f, c, y, x] : 0
//
// so this builds that filter as a constant -- folded here from the constant
// grouped filter, through the reshapes and `extf` an fp16 import puts in
// front of it -- and replaces the op with
// `conv_2d_nchw_fchw` over the input's collapsed `[N, G*Cg, H, W]` view. The
// importer's expand/collapse around the grouped op then cancel against this
// pass's, so the pad in front and the bias broadcast in the init reach the
// dense op exactly as a dense model's would: the channels-last conversion,
// demotion, pad fold and bias/ReLU fusion that follow all apply unchanged.
//
// It costs G x the multiply-adds (the zero blocks are computed) and G x the
// filter bytes. On ResNeXt50 that is the trade being measured: ~7.4 G MACs on
// the NPU against ~125 ms of CPU.
//
// Narrow on purpose, because an unclaimed block-diagonal convolution would
// run on the CPU at G x the cost: float only, static shapes, more than one
// group, a constant filter, a 1x1 or 3x3 kernel at stride 1 or 2 with unit
// dilation, and the
// dense channel counts inside the fp16 admission envelope (`Cin` 1152 at
// 3x3). And it runs only in an offloading build -- the `offload` option is
// false in `rocket-compiler --no-offload`'s spec, so the CPU baseline keeps
// the true grouped convolution.

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

/// Whether `generic` is a one-input elementwise `extf` or `truncf` with
/// identity maps -- the widening an fp16 import puts between a constant
/// filter and its convolution.
static bool isFloatCast(linalg::GenericOp generic) {
  if (generic.getNumDpsInputs() != 1 || generic.getNumDpsInits() != 1 ||
      !llvm::all_of(generic.getIndexingMapsArray(),
                    [](AffineMap map) { return map.isIdentity(); })) {
    return false;
  }
  Block &body = generic.getRegion().front();
  if (body.getOperations().size() != 2) {
    return false;
  }
  Operation &cast = body.front();
  return isa<arith::ExtFOp, arith::TruncFOp>(cast) &&
         cast.getOperand(0) == body.getArgument(0) &&
         body.getTerminator()->getOperand(0) == cast.getResult(0);
}

/// The grouped filter's values in row-major order, if it is a constant seen
/// through reshapes and float casts (which preserve row-major order and are
/// applied here with `arith`'s own rounding). `std::nullopt` otherwise.
static std::optional<SmallVector<APFloat>> foldConstantFilter(Value filter) {
  Value value = filter;
  while (true) {
    if (auto expand = value.getDefiningOp<tensor::ExpandShapeOp>()) {
      value = expand.getSrc();
      continue;
    }
    if (auto collapse = value.getDefiningOp<tensor::CollapseShapeOp>()) {
      value = collapse.getSrc();
      continue;
    }
    if (auto generic = value.getDefiningOp<linalg::GenericOp>()) {
      if (!isFloatCast(generic)) {
        return std::nullopt;
      }
      value = generic.getDpsInputs()[0];
      continue;
    }
    break;
  }
  DenseElementsAttr dense;
  if (!matchPattern(value, m_Constant(&dense)) ||
      !isa<FloatType>(dense.getElementType())) {
    return std::nullopt;
  }
  auto target = cast<FloatType>(
      cast<RankedTensorType>(filter.getType()).getElementType());
  SmallVector<APFloat> values;
  values.reserve(dense.getNumElements());
  for (APFloat element : dense.getValues<APFloat>()) {
    bool losesInfo = false;
    element.convert(target.getFloatSemantics(), APFloat::rmNearestTiesToEven,
                    &losesInfo);
    values.push_back(element);
  }
  return values;
}

/// The block-diagonal `[G*Fg, G*Cg, kh, kw]` filter as a constant, built from
/// the grouped filter's row-major values without materializing an APFloat per
/// dense element.
static DenseElementsAttr blockDiagonalConstant(ArrayRef<APFloat> grouped,
                                               RankedTensorType type,
                                               int64_t groups, int64_t fg,
                                               int64_t cg, int64_t taps) {
  auto element = cast<FloatType>(type.getElementType());
  int64_t bytes = element.getWidth() / 8;
  int64_t outChannels = groups * fg;
  int64_t inChannels = groups * cg;
  std::vector<char> raw(outChannels * inChannels * taps * bytes, 0);
  for (int64_t o = 0; o < outChannels; ++o) {
    int64_t group = o / fg;
    for (int64_t c = 0; c < cg; ++c) {
      int64_t i = group * cg + c;
      for (int64_t tap = 0; tap < taps; ++tap) {
        APInt bits = grouped[(o * cg + c) * taps + tap].bitcastToAPInt();
        int64_t at = ((o * inChannels + i) * taps + tap) * bytes;
        for (int64_t b = 0; b < bytes; ++b) {
          raw[at + b] = static_cast<char>(
              bits.extractBitsAsZExtValue(8, static_cast<unsigned>(b * 8)));
        }
      }
    }
  }
  return DenseElementsAttr::getFromRawBuffer(type, raw);
}

/// `Cin` and `Cout` the fp16 dense admission table covers at 3x3 and 1x1
/// (rocket-core `admission.rs`); a larger grouped convolution is left alone.
constexpr int64_t kMaxInChannels3x3 = 1152;
constexpr int64_t kMaxInChannels1x1 = 4096;
constexpr int64_t kMaxOutChannels = 4096;

struct BlockDiagonalGroupedConv
    : OpRewritePattern<linalg::Conv2DNgchwGfchwOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::Conv2DNgchwGfchwOp conv,
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
        !isa<FloatType>(filterType.getElementType()) ||
        !isa<FloatType>(initType.getElementType())) {
      return rewriter.notifyMatchFailure(conv, "not floating point");
    }

    // input [N, G, Cg, H, W]; filter [G, Fg, Cg, kh, kw]; out [N, G, Fg, OH, OW]
    int64_t groups = filterType.getDimSize(0);
    int64_t fg = filterType.getDimSize(1);
    int64_t cg = filterType.getDimSize(2);
    int64_t kh = filterType.getDimSize(3);
    int64_t kw = filterType.getDimSize(4);
    if (groups <= 1) {
      return rewriter.notifyMatchFailure(conv, "one group is dense already");
    }
    bool pointwise = kh == 1 && kw == 1;
    if (!(pointwise || (kh == 3 && kw == 3))) {
      return rewriter.notifyMatchFailure(conv, "not 1x1 or 3x3");
    }
    SmallVector<int64_t> strides(conv.getStrides().getValues<int64_t>());
    SmallVector<int64_t> dilations(conv.getDilations().getValues<int64_t>());
    if (strides.size() != 2 || strides[0] != strides[1] || strides[0] > 2 ||
        dilations != SmallVector<int64_t>{1, 1}) {
      return rewriter.notifyMatchFailure(conv, "not stride 1/2, dilation 1");
    }
    int64_t inChannels = groups * cg;
    int64_t outChannels = groups * fg;
    if (inChannels > (pointwise ? kMaxInChannels1x1 : kMaxInChannels3x3) ||
        outChannels > kMaxOutChannels) {
      return rewriter.notifyMatchFailure(conv, "outside the dense envelope");
    }
    if (inputType.getDimSize(1) != groups || inputType.getDimSize(2) != cg ||
        initType.getDimSize(1) != groups || initType.getDimSize(2) != fg) {
      return rewriter.notifyMatchFailure(conv, "inconsistent extents");
    }

    // The block-diagonal filter as a compile-time constant. A filter that is
    // not a constant is declined rather than expanded at run time: that
    // expansion would write G x the filter on the CPU every inference (and
    // IREE's const-expr hoisting will not lift it, since it grows the value
    // G-fold).
    std::optional<SmallVector<APFloat>> grouped = foldConstantFilter(filter);
    if (!grouped ||
        static_cast<int64_t>(grouped->size()) != filterType.getNumElements()) {
      return rewriter.notifyMatchFailure(conv, "filter is not a constant");
    }

    Location loc = conv.getLoc();
    int64_t batch = inputType.getDimSize(0);

    // [N, G, Cg, H, W] -> [N, G*Cg, H, W]
    Value denseInput = tensor::CollapseShapeOp::create(
        rewriter, loc,
        RankedTensorType::get({batch, inChannels, inputType.getDimSize(3),
                               inputType.getDimSize(4)},
                              inputType.getElementType()),
        input, SmallVector<ReassociationIndices>{{0}, {1, 2}, {3}, {4}});
    Value denseInit = tensor::CollapseShapeOp::create(
        rewriter, loc,
        RankedTensorType::get({batch, outChannels, initType.getDimSize(3),
                               initType.getDimSize(4)},
                              initType.getElementType()),
        init, SmallVector<ReassociationIndices>{{0}, {1, 2}, {3}, {4}});

    auto denseFilterType = RankedTensorType::get(
        {outChannels, inChannels, kh, kw}, filterType.getElementType());
    Value denseFilter = arith::ConstantOp::create(
        rewriter, loc,
        blockDiagonalConstant(*grouped, denseFilterType, groups, fg, cg,
                              kh * kw));

    auto dense = linalg::Conv2DNchwFchwOp::create(
        rewriter, loc, TypeRange{denseInit.getType()},
        ValueRange{denseInput, denseFilter}, ValueRange{denseInit},
        conv.getStridesAttr(), conv.getDilationsAttr());
    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
        conv, initType, dense.getResult(0),
        SmallVector<ReassociationIndices>{{0}, {1, 2}, {3}, {4}});
    return success();
  }
};

struct RocketBlockDiagonalGroupedConvPass
    : public PassWrapper<RocketBlockDiagonalGroupedConvPass,
                         OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      RocketBlockDiagonalGroupedConvPass)

  RocketBlockDiagonalGroupedConvPass() = default;
  RocketBlockDiagonalGroupedConvPass(
      const RocketBlockDiagonalGroupedConvPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const final {
    return "rocket-block-diagonal-grouped-conv";
  }
  StringRef getDescription() const final {
    return "Rewrites a float grouped convolution (conv_2d_ngchw_gfchw) inside "
           "the fp16 dense envelope into conv_2d_nchw_fchw with a "
           "block-diagonal filter, so the dense Rocket path claims it.";
  }

  Option<bool> offload{
      *this, "offload",
      llvm::cl::desc("Rewrite at all. False in a --no-offload build, whose "
                     "CPU baseline must keep the true grouped convolution."),
      llvm::cl::init(true)};

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() final {
    if (!offload) {
      return;
    }
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<BlockDiagonalGroupedConv>(context);
    GreedyRewriteConfig config;
    config.enableFolding(false);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      return signalPassFailure();
    }
  }
};

static PassRegistration<RocketBlockDiagonalGroupedConvPass> reg;

} // namespace
} // namespace mlir::iree_compiler::IREE::HAL
