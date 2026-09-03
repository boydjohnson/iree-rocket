// Copyright 2026
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Fuses an ONNX `QLinearConv` -- as torch-mlir lowers it -- into the
// canonical requantized form the Rocket transform spec's
// `@match_dynamic_conv2d_int8_requant` claims, so the convolution and its
// requantization leave as one NPU dispatch that returns i8.
//
// Why this exists at all: the other int8 path, `int8_accumulator`, writes the
// raw i32 accumulator and leaves requantization on the CPU, and it is capped
// at 384 coefficient bytes per output channel by a DPU limitation with no
// programmable workaround -- which is what holds its dense matchers at Cin 352
// (1x1) and 32 (3x3). Requantized output has no such cap. The model already
// carries everything the DPU's own requantization stages need; before this
// pass, the compiler threw it away and did the work again on the CPU.
//
// What torch-mlir hands us, for `QLinearConv(x, x_scale, x_zp, w, w_scale,
// w_zp, y_scale, y_zp, B)` with ui8 activations and symmetric weights:
//
//     %rebased   = linalg.generic ins(%x)         { addi %in, -128 }
//     %bcast     = linalg.broadcast ins(%bias : tensor<Coutxi32>) ...
//     %acc       = linalg.conv_2d_nchw_fchw_q
//                      ins(%rebased, %w, %x_zp_i32, %w_zp_i32) outs(%bcast)
//     %real      = linalg.generic ins(%acc)       { sitofp; mulf x_scale*w_scale }
//     %quantized = linalg.generic ins(%real)      { divf y_scale; roundeven;
//                                                   addf y_zp; clamp; fptoui }
//
// and what this pass produces in its place:
//
//     %acc = linalg.conv_2d_nhwc_hwcf ins(nhwc(%rebased), hwcf(%w)) outs(zero)
//     %q   = linalg.generic ins(%acc, %bias', %multiplier, %zp, %lo, %hi)
//                { addi bias; sitofp; mulf; roundeven; addf zp; clamp; fptosi }
//     %out = nchw(%q) with the ui8 rebase re-applied
//
// Three transformations are folded into that, each exact:
//
//   * **The input zero point moves into the bias.** The hardware uses it only
//     as `CNA_PAD_CON1.pad_value`, and every Rocket dispatch is unpadded, so
//     what has to survive is the arithmetic identity
//     `conv_q(x, w, xz, 0) == conv(x, w) - xz * sum_k(w)`. The correction is
//     per output channel and the weights are constant, so it folds into the
//     bias here rather than becoming the separate CPU pass
//     `iree-global-opt-quantized-conv-to-conv` would emit.
//   * **The two scales collapse into one multiplier**, `x_scale * w_scale /
//     y_scale`, which is what `DPU_OUT_CVT_SCALE`/`SHIFT` encode. This is the
//     one step that is not bit-exact against the original: multiplying by a
//     single f32 differs from multiply-then-divide in the last ulp, so a
//     product sitting exactly on a rounding tie can land on the other side.
//     The device requantizes in fixed point regardless -- see the atol=1 note
//     on the requantized cases in tools/e2e_conv_regression.py.
//   * **ui8 activations stay in ui8-bit-pattern storage.** iree-rocket-hal
//     has only signed int8, and torch-mlir has already rebased the input by
//     -128 (reporting `x_zp` in the signed domain: 126 becomes -2). So the
//     dispatch consumes that rebased tensor, emits signed i8 with
//     `y_zp - 128`, and one `addi -128` puts the result back in the ui8 bit
//     pattern its consumers expect. Between two fused convolutions that
//     fixup and the next rebase are the same op twice, which folds away.
//
// Everything here is a semantics-preserving rewrite: the result is a correct
// convolution whether or not the transform spec goes on to claim it. The
// spec's matchers decide what the NPU actually runs, and the shape bounds
// live there, not here.

#include <cmath>
#include <limits>
#include <optional>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::iree_compiler::IREE::HAL {
namespace {

// linalg.transpose's permutation is read as
// outputShape[i] == inputShape[permutation[i]], same as
// RocketTransposeQuantizedConvPass.
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

std::optional<double> constantFloat(Value value) {
  FloatAttr attr;
  if (!matchPattern(value, m_Constant(&attr))) {
    return std::nullopt;
  }
  return attr.getValueAsDouble();
}

std::optional<int64_t> constantInt(Value value) {
  IntegerAttr attr;
  if (!matchPattern(value, m_Constant(&attr))) {
    return std::nullopt;
  }
  return attr.getInt();
}

// The trailing `divf; roundeven; addf; clamp; narrow` generic.
struct QuantizeEpilogue {
  Value source;
  double outputScale = 0.0;
  double outputZeroPoint = 0.0;
  bool storesUnsigned = false;
};

// Every op in these bodies is matched positionally rather than by a general
// expression walk. That is deliberate: this is compiler output with one
// spelling, and a loose match here would silently claim some *other*
// elementwise chain and change its meaning.
std::optional<QuantizeEpilogue> matchQuantize(linalg::GenericOp op) {
  if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1 ||
      !op.hasPureTensorSemantics()) {
    return std::nullopt;
  }
  if (!llvm::all_of(op.getIndexingMapsArray(),
                    [](AffineMap map) { return map.isIdentity(); })) {
    return std::nullopt;
  }
  Block &body = op.getRegion().front();
  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yield || yield.getNumOperands() != 1) {
    return std::nullopt;
  }

  Operation *narrow = yield.getOperand(0).getDefiningOp();
  bool storesUnsigned = isa_and_nonnull<arith::FPToUIOp>(narrow);
  if (!storesUnsigned && !isa_and_nonnull<arith::FPToSIOp>(narrow)) {
    return std::nullopt;
  }
  auto high = narrow->getOperand(0).getDefiningOp<arith::MinimumFOp>();
  if (!high) {
    return std::nullopt;
  }
  auto low = high.getLhs().getDefiningOp<arith::MaximumFOp>();
  if (!low) {
    return std::nullopt;
  }
  auto offset = low.getLhs().getDefiningOp<arith::AddFOp>();
  if (!offset) {
    return std::nullopt;
  }
  auto rounded = offset.getLhs().getDefiningOp<math::RoundEvenOp>();
  if (!rounded) {
    return std::nullopt;
  }
  auto scaled = rounded.getOperand().getDefiningOp<arith::DivFOp>();
  if (!scaled || scaled.getLhs() != body.getArgument(0)) {
    return std::nullopt;
  }

  std::optional<double> outputScale = constantFloat(scaled.getRhs());
  std::optional<double> zeroPoint = constantFloat(offset.getRhs());
  std::optional<double> clampLow = constantFloat(low.getRhs());
  std::optional<double> clampHigh = constantFloat(high.getRhs());
  if (!outputScale || !zeroPoint || !clampLow || !clampHigh) {
    return std::nullopt;
  }
  // The clamp bounds are what say which storage domain this is, and they have
  // to agree with the narrowing op: ui8 values live in [0, 255] and are
  // stored through fptoui, signed ones in [-128, 127] through fptosi.
  const double expectedLow = storesUnsigned ? 0.0 : -128.0;
  const double expectedHigh = storesUnsigned ? 255.0 : 127.0;
  if (*clampLow != expectedLow || *clampHigh != expectedHigh) {
    return std::nullopt;
  }
  if (*outputScale <= 0.0 || !std::isfinite(*outputScale)) {
    return std::nullopt;
  }
  if (*zeroPoint != std::round(*zeroPoint)) {
    return std::nullopt;
  }

  return QuantizeEpilogue{op.getDpsInputs()[0], *outputScale, *zeroPoint,
                          storesUnsigned};
}

// The `sitofp; mulf` generic between the convolution and the quantize.
// Returns the i32 source and `x_scale * w_scale`.
std::optional<std::pair<Value, double>> matchDequantize(linalg::GenericOp op) {
  if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1 ||
      !op.hasPureTensorSemantics()) {
    return std::nullopt;
  }
  if (!llvm::all_of(op.getIndexingMapsArray(),
                    [](AffineMap map) { return map.isIdentity(); })) {
    return std::nullopt;
  }
  Block &body = op.getRegion().front();
  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yield || yield.getNumOperands() != 1) {
    return std::nullopt;
  }
  auto scaled = yield.getOperand(0).getDefiningOp<arith::MulFOp>();
  if (!scaled) {
    return std::nullopt;
  }
  auto extended = scaled.getLhs().getDefiningOp<arith::SIToFPOp>();
  if (!extended || extended.getIn() != body.getArgument(0)) {
    return std::nullopt;
  }
  std::optional<double> scale = constantFloat(scaled.getRhs());
  if (!scale || *scale <= 0.0 || !std::isfinite(*scale)) {
    return std::nullopt;
  }
  return std::make_pair(op.getDpsInputs()[0], *scale);
}

// The convolution's init carries the bias: torch-mlir broadcasts a
// `tensor<Coutxi32>` constant across the output, or leaves an all-zero
// constant when the ONNX node had no bias.
std::optional<SmallVector<int64_t>> matchBias(Value init, int64_t outChannels) {
  SmallVector<int64_t> bias(outChannels, 0);
  if (auto broadcast = init.getDefiningOp<linalg::BroadcastOp>()) {
    DenseElementsAttr attr;
    if (!matchPattern(broadcast.getInput(), m_Constant(&attr))) {
      return std::nullopt;
    }
    auto type = dyn_cast<RankedTensorType>(attr.getType());
    if (!type || type.getRank() != 1 || type.getDimSize(0) != outChannels) {
      return std::nullopt;
    }
    // dimensions = [0, 2, 3] is the NCHW spelling: the one dimension the
    // bias is *not* broadcast over is the output channel.
    ArrayRef<int64_t> dims = broadcast.getDimensions();
    if (dims != ArrayRef<int64_t>{0, 2, 3}) {
      return std::nullopt;
    }
    int64_t index = 0;
    for (APInt value : attr.getValues<APInt>()) {
      bias[index++] = value.getSExtValue();
    }
    return bias;
  }

  DenseElementsAttr attr;
  if (matchPattern(init, m_Constant(&attr)) && attr.isSplat() &&
      attr.getSplatValue<APInt>().isZero()) {
    return bias;
  }
  if (auto fill = init.getDefiningOp<linalg::FillOp>()) {
    std::optional<int64_t> value = constantInt(fill.getInputs()[0]);
    if (value && *value == 0) {
      return bias;
    }
  }
  return std::nullopt;
}

struct FuseQLinearConv : OpRewritePattern<linalg::Conv2DNchwFchwQOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::Conv2DNchwFchwQOp convOp,
                                PatternRewriter &rewriter) const override {
    if (!convOp.hasPureTensorSemantics()) {
      return rewriter.notifyMatchFailure(convOp, "expected tensor semantics");
    }
    Value input = convOp.getDpsInputs()[0];
    Value filter = convOp.getDpsInputs()[1];
    Value init = convOp.getDpsInits()[0];
    auto filterType = dyn_cast<RankedTensorType>(filter.getType());
    auto initType = dyn_cast<RankedTensorType>(init.getType());
    if (!filterType || !initType || !filterType.hasStaticShape() ||
        initType.getRank() != 4) {
      return rewriter.notifyMatchFailure(convOp, "expected a static filter");
    }

    // Only a symmetric weight quantization reaches the hardware -- the BS
    // plane carries `-weight_zero_point` per channel and only the zero case
    // is validated -- and ORT's quantizer produces exactly that.
    std::optional<int64_t> inputZeroPoint =
        constantInt(convOp.getDpsInputs()[2]);
    std::optional<int64_t> filterZeroPoint =
        constantInt(convOp.getDpsInputs()[3]);
    if (!inputZeroPoint || !filterZeroPoint || *filterZeroPoint != 0) {
      return rewriter.notifyMatchFailure(convOp, "expected a zero weight zp");
    }

    // The chain has to be exactly conv -> dequantize -> quantize, each with a
    // single use. Anything else reading the raw accumulator means the
    // requantization is not the whole consumer and fusing it would change
    // what that other reader sees.
    if (!convOp->hasOneUse()) {
      return rewriter.notifyMatchFailure(convOp, "accumulator has other uses");
    }
    auto dequantizeOp =
        dyn_cast<linalg::GenericOp>(*convOp->getUsers().begin());
    if (!dequantizeOp || !dequantizeOp->hasOneUse()) {
      return rewriter.notifyMatchFailure(convOp, "no single dequantize user");
    }
    std::optional<std::pair<Value, double>> dequantize =
        matchDequantize(dequantizeOp);
    if (!dequantize || dequantize->first != convOp.getResult(0)) {
      return rewriter.notifyMatchFailure(convOp, "unrecognized dequantize");
    }
    auto quantizeOp =
        dyn_cast<linalg::GenericOp>(*dequantizeOp->getUsers().begin());
    if (!quantizeOp) {
      return rewriter.notifyMatchFailure(convOp, "no quantize consumer");
    }
    std::optional<QuantizeEpilogue> quantize = matchQuantize(quantizeOp);
    if (!quantize || quantize->source != dequantizeOp.getResult(0)) {
      return rewriter.notifyMatchFailure(convOp, "unrecognized quantize");
    }

    const int64_t outChannels = filterType.getDimSize(0);
    std::optional<SmallVector<int64_t>> bias = matchBias(init, outChannels);
    if (!bias) {
      return rewriter.notifyMatchFailure(convOp, "unrecognized bias init");
    }

    // Fold `- x_zp * sum_k(w)` into the bias. Needs the weights themselves,
    // which a quantized model always has as a constant; without them the
    // correction would have to stay a separate CPU pass and there would be
    // nothing to gain here.
    if (*inputZeroPoint != 0) {
      DenseElementsAttr filterAttr;
      if (!matchPattern(filter, m_Constant(&filterAttr))) {
        return rewriter.notifyMatchFailure(convOp, "non-constant weights");
      }
      const int64_t perChannel = filterType.getNumElements() / outChannels;
      SmallVector<int64_t> sums(outChannels, 0);
      int64_t index = 0;
      for (APInt value : filterAttr.getValues<APInt>()) {
        sums[index / perChannel] += value.getSExtValue();
        ++index;
      }
      for (int64_t channel = 0; channel < outChannels; ++channel) {
        // i64 throughout: Cin * kh * kw * 127 * |zp| stays far inside it, and
        // the result is checked back into i32 below.
        (*bias)[channel] -= *inputZeroPoint * sums[channel];
      }
    }
    for (int64_t value : *bias) {
      if (value < std::numeric_limits<int32_t>::min() ||
          value > std::numeric_limits<int32_t>::max()) {
        return rewriter.notifyMatchFailure(convOp, "bias fold overflows i32");
      }
    }

    Location loc = convOp.getLoc();
    MLIRContext *context = rewriter.getContext();
    Type accumulatorType = initType.getElementType();
    auto quantizedType =
        cast<RankedTensorType>(quantizeOp.getResult(0).getType());
    Type storageType = quantizedType.getElementType();

    // NCHW -> NHWC for the image, FCHW -> HWCF for the filter, exactly as
    // RocketTransposeQuantizedConvPass does. The convolution has to be NHWC
    // here rather than after the channels-last pass, because that pass
    // transposes the *result* back and would put a transpose between the
    // convolution and its epilogue -- and the two being adjacent is what the
    // spec's DAG matcher looks for.
    Value nhwcInput = transposeTo(rewriter, loc, input, {0, 2, 3, 1});
    Value hwcfFilter = transposeTo(rewriter, loc, filter, {2, 3, 1, 0});
    SmallVector<int64_t> nhwcShape{initType.getDimSize(0),
                                   initType.getDimSize(2),
                                   initType.getDimSize(3),
                                   initType.getDimSize(1)};
    SmallVector<OpFoldResult> initSizes =
        tensor::getMixedSizes(rewriter, loc, init);
    SmallVector<OpFoldResult> nhwcSizes{initSizes[0], initSizes[2],
                                        initSizes[3], initSizes[1]};

    Value accumulatorEmpty =
        tensor::EmptyOp::create(rewriter, loc, nhwcSizes, accumulatorType);
    Value zero = arith::ConstantOp::create(
        rewriter, loc, rewriter.getIntegerAttr(accumulatorType, 0));
    Value accumulatorInit =
        linalg::FillOp::create(rewriter, loc, ValueRange{zero},
                               ValueRange{accumulatorEmpty})
            .getResult(0);
    // Respell `strides`/`dilations` as tensor-typed attributes. torch-mlir
    // hands them over as `dense<1> : vector<2xi64>` and both spellings are
    // legal here, but the spec's DAG matcher compares whole attribute
    // dictionaries, so the convolution this emits has to agree exactly with
    // the template in @match_dynamic_conv2d_int8_requant -- and with the
    // hand-written canonical-form fixtures, which spell it `tensor<2xi64>`.
    auto respell = [&](DenseIntElementsAttr attr) {
      SmallVector<int64_t> values(attr.getValues<int64_t>());
      return DenseIntElementsAttr::get(
          RankedTensorType::get({static_cast<int64_t>(values.size())},
                                rewriter.getI64Type()),
          values);
    };
    auto nhwcConv = linalg::Conv2DNhwcHwcfOp::create(
        rewriter, loc, accumulatorInit.getType(),
        ValueRange{nhwcInput, hwcfFilter}, ValueRange{accumulatorInit},
        respell(convOp.getStrides()), respell(convOp.getDilations()));

    // The requantization, as one elementwise generic whose scale, zero point
    // and clamp bounds are *operands*. They cannot be constants in the body:
    // the spec's DAG matcher compares regions under a mapping built only from
    // the ops it walked, and the canonicalizer hoists an in-body constant out
    // before that matcher ever runs.
    SmallVector<APInt> biasValues;
    const unsigned accumulatorBits = accumulatorType.getIntOrFloatBitWidth();
    for (int64_t value : *bias) {
      biasValues.push_back(APInt(accumulatorBits, value, /*isSigned=*/true));
    }
    auto biasType =
        RankedTensorType::get(ArrayRef<int64_t>{outChannels}, accumulatorType);
    Value biasTensor = arith::ConstantOp::create(
        rewriter, loc, DenseElementsAttr::get(biasType, biasValues));

    const double multiplier = dequantize->second / quantize->outputScale;
    Value multiplierValue = arith::ConstantOp::create(
        rewriter, loc, rewriter.getF32FloatAttr(multiplier));
    // A ui8 model's zero point is in the unsigned domain; the dispatch emits
    // signed int8, so it moves down by 128 and the storage fixup below moves
    // the result back.
    const int64_t zeroPoint =
        static_cast<int64_t>(quantize->outputZeroPoint) -
        (quantize->storesUnsigned ? 128 : 0);
    Value zeroPointValue = arith::ConstantOp::create(
        rewriter, loc, rewriter.getI32IntegerAttr(zeroPoint));
    Value clampLow = arith::ConstantOp::create(
        rewriter, loc, rewriter.getF32FloatAttr(-128.0));
    Value clampHigh = arith::ConstantOp::create(
        rewriter, loc, rewriter.getF32FloatAttr(127.0));

    Value quantizedEmpty =
        tensor::EmptyOp::create(rewriter, loc, nhwcSizes, storageType);
    AffineMap identity = AffineMap::getMultiDimIdentityMap(4, context);
    AffineMap channel =
        AffineMap::get(4, 0, {rewriter.getAffineDimExpr(3)}, context);
    AffineMap scalar = AffineMap::get(4, 0, {}, context);
    SmallVector<AffineMap> maps{identity, channel, scalar,
                                scalar,   scalar,  scalar, identity};
    SmallVector<utils::IteratorType> iterators(4,
                                               utils::IteratorType::parallel);

    auto requantized = linalg::GenericOp::create(
        rewriter, loc, TypeRange{quantizedEmpty.getType()},
        ValueRange{nhwcConv.getResult(0), biasTensor, multiplierValue,
                   zeroPointValue, clampLow, clampHigh},
        ValueRange{quantizedEmpty}, maps, iterators,
        [&](OpBuilder &builder, Location nested, ValueRange args) {
          Value biased = arith::AddIOp::create(builder, nested, args[0], args[1]);
          Value real = arith::SIToFPOp::create(builder, nested,
                                               builder.getF32Type(), biased);
          Value scaled = arith::MulFOp::create(builder, nested, real, args[2]);
          Value rounded = math::RoundEvenOp::create(builder, nested, scaled);
          Value zeroPointF32 = arith::SIToFPOp::create(
              builder, nested, builder.getF32Type(), args[3]);
          Value offset =
              arith::AddFOp::create(builder, nested, rounded, zeroPointF32);
          Value low =
              arith::MaximumFOp::create(builder, nested, offset, args[4]);
          Value clamped =
              arith::MinimumFOp::create(builder, nested, low, args[5]);
          Value narrowed = arith::FPToSIOp::create(builder, nested,
                                                   storageType, clamped);
          linalg::YieldOp::create(builder, nested, narrowed);
        });

    Value result = transposeTo(rewriter, loc, requantized.getResult(0),
                               {0, 3, 1, 2});

    // Back to the ui8 bit pattern the rest of the graph stores. This is the
    // same `addi -128` torch-mlir puts on every QLinearConv *input*, so
    // between two fused convolutions the pair cancels.
    if (quantize->storesUnsigned) {
      Value rebaseEmpty = tensor::EmptyOp::create(
          rewriter, loc, tensor::getMixedSizes(rewriter, loc, result),
          storageType);
      Value shift = arith::ConstantOp::create(
          rewriter, loc, rewriter.getIntegerAttr(storageType, -128));
      result = linalg::GenericOp::create(
                   rewriter, loc, TypeRange{rebaseEmpty.getType()},
                   ValueRange{result}, ValueRange{rebaseEmpty},
                   SmallVector<AffineMap>{identity, identity}, iterators,
                   [&](OpBuilder &builder, Location nested, ValueRange args) {
                     Value rebased =
                         arith::AddIOp::create(builder, nested, args[0], shift);
                     linalg::YieldOp::create(builder, nested, rebased);
                   })
                   .getResult(0);
    }

    rewriter.replaceOp(quantizeOp, result);
    return success();
  }
};

struct RocketFuseQLinearConvPass
    : public PassWrapper<RocketFuseQLinearConvPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RocketFuseQLinearConvPass)

  StringRef getArgument() const final { return "rocket-fuse-qlinear-conv"; }
  StringRef getDescription() const final {
    return "Fuses an ONNX QLinearConv and its requantization into the "
           "canonical form the Rocket requantized int8 matchers claim.";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    math::MathDialect, tensor::TensorDialect>();
  }

  void runOnOperation() final {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<FuseQLinearConv>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

static PassRegistration<RocketFuseQLinearConvPass> reg;

} // namespace
} // namespace mlir::iree_compiler::IREE::HAL
