// Copyright 2026
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Collapses the requantization epilogue of an ONNX QLinearConv into the one
// elementwise generic `@match_dynamic_conv2d_int8_requant` is written
// against, so a real quantized model can reach the requantized int8 path
// instead of the int32 accumulator one.
//
// The accumulator path returns `i32` and leaves five unfused CPU passes over
// that tensor behind it. ISSUES.md P8 measured the cost of exactly that --
// "stop materializing i32 activations and stop leaving their epilogues
// unfused ... this is the whole 7.4 ms" -- and named the requantized path as
// the structural fix. This pass is what connects the two: the path has been
// board-validated since 2026-09-03 against a hand-written canonical form,
// and no model produced that form.
//
// What arrives, per convolution, after `iree-global-opt-quantized-conv-to-conv`
// and the channels-last conversion (measured on mobilenetv2.static-int8.onnx,
// 34 dense convolutions, all identical in shape):
//
//   %sumw = generic reduce(%filter)            // extsi, addi   -> tensor<Coutxi32>
//   %acc  = conv_2d_nhwc_hwcf(%in, %filter)
//             outs(broadcast(%bias))           // the bias IS the init
//   %corr = generic(%acc, broadcast(%sumw))    // muli x_zp, subi
//   %nchw = transpose %corr                    // NHWC -> NCHW, on i32
//   %real = generic(%nchw)                     // sitofp, mulf  (x_scale*w_scale)
//   %u8   = generic(%real)                     // divf y_scale, roundeven,
//                                              // addf y_zp, clamp 0..255, fptoui
//
// and what leaves:
//
//   %bias2 = generic(%bias, %sumw)             // rank 1: bias - x_zp*sumw
//   %acc   = conv_2d_nhwc_hwcf(%in, %filter) outs(fill 0)
//   %s8    = generic(%acc, %bias2, %scale, %zp, %lo, %hi)   // the canonical form
//   %u8    = generic(%s8)                      // addi -128, the s8->u8 shift
//   %nchw  = transpose %u8                     // now on i8, not i32
//
// Four full-tensor `i32` passes become one `i8` pass plus a `i8` transpose,
// and the convolution is left in the shape the requantized matcher claims.
//
// Three things worth knowing about why it is shaped this way.
//
// **The bias moves out of the init.** The canonical form is a convolution
// over a *zero* init followed by a generic that adds the per-channel bias,
// because that is what the hardware does: the DPU's BS plane adds the bias
// on the way out. `quantized-conv-to-conv` instead seeds the accumulator
// with a broadcast of the bias, which is the same arithmetic and the wrong
// shape, so the bias is lifted back out here.
//
// **The zero-point correction folds into that bias.** `-x_zp * sum_k(w)` is
// a per-output-channel constant whenever the filter is constant, which it is
// in every quantized model this targets. Emitting it as a rank-1 generic
// over `%bias` and `%sumw` leaves an expression IREE's constant evaluation
// folds to a literal, rather than the full-size broadcast-and-subtract over
// the activation tensor that arrives.
//
// **The output is signed, and the shift that follows is not new work.** The
// model quantizes activations as `ui8`: clamp to `[0, 255]` and `fptoui`.
// The DPU's out-convert clamps to `[-128, 127]` and writes signed, so the
// fused generic emits the signed form with `zero_point - 128` and a
// compensating `addi -128` restores the encoding the rest of the graph
// expects. That shift is not a pass this adds -- the model already has one
// in front of every convolution (`quantized-conv-to-conv` needs its operand
// signed too), so this is the same rewrite moved one op earlier.
//
// **Numerics.** `round(acc * s1 / s2)` becomes `round(acc * (s1/s2))`, one
// rounding of the scale ratio rather than two operations. The two differ
// only where a product lands exactly on a tie, which is the same class of
// difference the requantized path's e2e gate already runs at `atol=1` for
// (see `requantized-int8-conv-path`: the device rounds half away from zero
// where the f32 reference rounds half to even).

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::iree_compiler::IREE::HAL {
namespace {

/// The `ui8` quantization range the ONNX importer emits, as the `f32` clamp
/// bounds it materializes them as.
constexpr double kUnsignedLow = 0.0;
constexpr double kUnsignedHigh = 255.0;
/// The signed range the DPU's out-convert stage clamps to, and the shift
/// between the two encodings.
constexpr double kSignedLow = -128.0;
constexpr double kSignedHigh = 127.0;
constexpr int64_t kEncodingShift = -128;

/// The single `linalg.generic` consuming `value`, or null when it has any
/// other number of uses or a different consumer.
///
/// Single-use is the safety property this whole pass rests on: an
/// intermediate with a second reader cannot be absorbed, because the rewrite
/// deletes it.
static linalg::GenericOp soleGenericConsumer(Value value) {
  if (!value.hasOneUse()) {
    return nullptr;
  }
  return dyn_cast<linalg::GenericOp>(*value.getUsers().begin());
}

/// Whether `op` is elementwise over `rank` parallel dimensions with identity
/// maps for every operand except the ones named in `channelOperands`, which
/// must project onto the last (channel) dimension.
static bool isElementwiseWithIdentityMaps(linalg::GenericOp op) {
  if (op.getNumResults() != 1 || !op.isAllParallelLoops()) {
    return false;
  }
  return llvm::all_of(op.getIndexingMapsArray(), [](AffineMap map) {
    return map.isIdentity() || map.getNumResults() == 0;
  });
}

/// Matches a body of exactly the ops named by `opNames`, in order, ending in
/// the `linalg.yield`. Returns the ops on success so the caller can read
/// their operands, or an empty vector on any mismatch.
///
/// Spelled as a flat sequence rather than a pattern-matching DSL because
/// these bodies come out of one upstream pass in one shape; a mismatch means
/// that pass changed, and a caller that silently accepted a *similar* body
/// would miscompile rather than decline.
static SmallVector<Operation *> matchBody(linalg::GenericOp op,
                                          ArrayRef<StringRef> opNames) {
  SmallVector<Operation *> matched;
  Block &body = op.getRegion().front();
  auto it = body.begin();
  for (StringRef name : opNames) {
    if (it == body.end() || it->getName().getStringRef() != name) {
      return {};
    }
    matched.push_back(&*it);
    ++it;
  }
  if (it == body.end() || !isa<linalg::YieldOp>(*it) ||
      std::next(it) != body.end()) {
    return {};
  }
  return matched;
}

/// The constant `f32` an op operand carries, following it out to the
/// generic's own `ins` when the value is a block argument.
///
/// The canonicaliser hoists a scalar constant out of a `linalg.generic` body
/// and passes it in as a zero-rank operand, so both spellings occur and both
/// have to be read. This is the same hazard `transform-dag-matcher-traps`
/// records for the matcher side.
static std::optional<APFloat> constantFloatOperand(linalg::GenericOp op,
                                                   Value value) {
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    unsigned index = blockArg.getArgNumber();
    if (index >= op.getNumDpsInputs()) {
      return std::nullopt;
    }
    APFloat result(0.0f);
    if (!matchPattern(op.getDpsInputs()[index], m_ConstantFloat(&result))) {
      return std::nullopt;
    }
    return result;
  }
  APFloat result(0.0f);
  if (!matchPattern(value, m_ConstantFloat(&result))) {
    return std::nullopt;
  }
  return result;
}

/// The constant `i32` an op operand carries, with the same block-argument
/// indirection [`constantFloatOperand`] handles.
static std::optional<APInt> constantIntOperand(linalg::GenericOp op,
                                               Value value) {
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    unsigned index = blockArg.getArgNumber();
    if (index >= op.getNumDpsInputs()) {
      return std::nullopt;
    }
    APInt result;
    if (!matchPattern(op.getDpsInputs()[index], m_ConstantInt(&result))) {
      return std::nullopt;
    }
    return result;
  }
  APInt result;
  if (!matchPattern(value, m_ConstantInt(&result))) {
    return std::nullopt;
  }
  return result;
}

static bool isFloatValue(std::optional<APFloat> value, double expected) {
  return value && value->convertToDouble() == expected;
}

/// Everything the rewrite needs, recovered from one convolution's epilogue.
struct RequantEpilogue {
  /// The rank-1 `i32` per-channel bias the convolution's init broadcasts, or
  /// null when the init is a zero fill.
  Value bias;
  /// The rank-1 `i32` reduction of the filter, `sum_k(w)`.
  Value weightSum;
  /// The input zero point, in the signed encoding the correction uses.
  int64_t inputZeroPoint = 0;
  /// `x_scale * w_scale`, then `y_scale`. The fused form carries their ratio.
  APFloat accumulatorScale = APFloat(0.0f);
  APFloat outputScale = APFloat(0.0f);
  /// The output zero point, as the `f32` the narrowing body adds.
  APFloat outputZeroPoint = APFloat(0.0f);
  /// The NHWC -> NCHW transpose between the correction and the scaling, when
  /// the model has one. Re-emitted after the fused generic, on `i8`.
  linalg::TransposeOp transpose;
  /// The op whose result the fused chain replaces.
  linalg::GenericOp narrow;
};

/// Reads the per-channel `i32` bias out of a convolution's accumulator init,
/// leaving `bias` null for a zero fill.
///
/// Three spellings, because the init's shape depends on where in the
/// pipeline this runs. `quantized-conv-to-conv` seeds the accumulator with a
/// broadcast of the bias in the layout the model arrived in, and
/// `iree-preprocessing-convert-conv-to-channels-last` then puts a transpose
/// in front of it rather than rewriting the broadcast -- so at this point in
/// the pipeline the init is `transpose(broadcast(bias))`, and it is only a
/// bare `broadcast` after a later canonicalisation folds the pair. Matching
/// only the folded form is why the first version of this pass fired on the
/// end-of-preprocessing IR and on nothing at all in the real pipeline.
static LogicalResult matchBiasInit(Value init, Value &bias) {
  if (auto fill = init.getDefiningOp<linalg::FillOp>()) {
    APInt zero;
    if (!matchPattern(fill.getInputs()[0], m_ConstantInt(&zero)) ||
        !zero.isZero()) {
      return failure();
    }
    bias = nullptr;
    return success();
  }

  auto broadcast = init.getDefiningOp<linalg::BroadcastOp>();
  if (auto transpose = init.getDefiningOp<linalg::TransposeOp>()) {
    broadcast = transpose.getInput().getDefiningOp<linalg::BroadcastOp>();
    if (!broadcast) {
      return failure();
    }
    // The broadcast adds every dimension but one; that surviving dimension
    // is the channel the bias is indexed by. After the transpose it has to
    // be last, because that is where the fused generic's channel map reads
    // it -- a permutation that put it anywhere else would be a different
    // tensor, not a relabelled one.
    auto broadcastType =
        cast<RankedTensorType>(broadcast.getResult()[0].getType());
    llvm::SmallDenseSet<int64_t> added(broadcast.getDimensions().begin(),
                                       broadcast.getDimensions().end());
    std::optional<int64_t> retained;
    for (int64_t dim = 0; dim < broadcastType.getRank(); ++dim) {
      if (!added.contains(dim)) {
        if (retained) {
          return failure();
        }
        retained = dim;
      }
    }
    ArrayRef<int64_t> permutation = transpose.getPermutation();
    if (!retained || permutation.empty() ||
        permutation.back() != *retained) {
      return failure();
    }
  }
  if (!broadcast) {
    return failure();
  }

  bias = broadcast.getInput();
  auto biasType = dyn_cast<RankedTensorType>(bias.getType());
  if (!biasType || biasType.getRank() != 1 ||
      !biasType.getElementType().isSignlessInteger(32)) {
    return failure();
  }
  return success();
}

/// Reads the reduction that produces `sum_k(w)`.
///
/// Requires it to reduce *this convolution's own filter*, which is what
/// makes folding it into the bias sound: a reduction over some other tensor
/// would be a different correction term.
static Value matchWeightSum(Value candidate, Value convFilter) {
  auto reduce = candidate.getDefiningOp<linalg::GenericOp>();
  if (!reduce || reduce.getNumDpsInputs() != 1 ||
      reduce.getNumResults() != 1) {
    return nullptr;
  }
  if (reduce.getDpsInputs()[0] != convFilter) {
    return nullptr;
  }
  auto resultType = dyn_cast<RankedTensorType>(reduce.getResult(0).getType());
  if (!resultType || resultType.getRank() != 1 ||
      !resultType.getElementType().isSignlessInteger(32)) {
    return nullptr;
  }
  if (matchBody(reduce, {"arith.extsi", "arith.addi"}).empty()) {
    return nullptr;
  }
  return reduce.getResult(0);
}

/// Walks a convolution's consumers and recovers the epilogue, or returns
/// nullopt at the first thing that does not fit.
template <typename ConvOp>
static std::optional<RequantEpilogue> matchEpilogue(ConvOp convOp) {
  RequantEpilogue found;

  // The init carries the bias, or is a plain zero fill.
  if (failed(matchBiasInit(convOp.getDpsInits()[0], found.bias))) {
    return std::nullopt;
  }

  // The zero-point correction: acc - sum_k(w) * x_zp.
  linalg::GenericOp correction = soleGenericConsumer(convOp.getResult(0));
  if (!correction || correction.getNumDpsInputs() != 2 ||
      !isElementwiseWithIdentityMaps(correction)) {
    return std::nullopt;
  }
  SmallVector<Operation *> correctionBody =
      matchBody(correction, {"arith.muli", "arith.subi"});
  if (correctionBody.empty()) {
    return std::nullopt;
  }
  auto *muli = correctionBody[0];
  auto *subi = correctionBody[1];
  if (subi->getOperand(1) != muli->getResult(0)) {
    return std::nullopt;
  }
  std::optional<APInt> zeroPoint =
      constantIntOperand(correction, muli->getOperand(1));
  if (!zeroPoint) {
    return std::nullopt;
  }
  found.inputZeroPoint = zeroPoint->getSExtValue();
  // Which of the two inputs is the broadcast weight sum: the one the muli
  // reads, not the one the subi does.
  auto sumArg = dyn_cast<BlockArgument>(muli->getOperand(0));
  auto accArg = dyn_cast<BlockArgument>(subi->getOperand(0));
  if (!sumArg || !accArg || sumArg.getArgNumber() >= correction.getNumDpsInputs() ||
      accArg.getArgNumber() >= correction.getNumDpsInputs()) {
    return std::nullopt;
  }
  if (correction.getDpsInputs()[accArg.getArgNumber()] != convOp.getResult(0)) {
    return std::nullopt;
  }
  auto sumBroadcast = correction.getDpsInputs()[sumArg.getArgNumber()]
                          .getDefiningOp<linalg::BroadcastOp>();
  if (!sumBroadcast) {
    return std::nullopt;
  }
  found.weightSum =
      matchWeightSum(sumBroadcast.getInput(), convOp.getDpsInputs()[1]);
  if (!found.weightSum) {
    return std::nullopt;
  }

  // An optional layout transpose, which the rewrite re-emits on i8.
  Value scaled = correction.getResult(0);
  if (scaled.hasOneUse()) {
    if (auto transpose =
            dyn_cast<linalg::TransposeOp>(*scaled.getUsers().begin())) {
      found.transpose = transpose;
      scaled = transpose.getResult()[0];
    }
  }

  // sitofp, then multiply by x_scale * w_scale.
  linalg::GenericOp toReal = soleGenericConsumer(scaled);
  if (!toReal || toReal.getNumDpsInputs() != 1 ||
      !isElementwiseWithIdentityMaps(toReal)) {
    return std::nullopt;
  }
  SmallVector<Operation *> realBody =
      matchBody(toReal, {"arith.sitofp", "arith.mulf"});
  if (realBody.empty()) {
    return std::nullopt;
  }
  std::optional<APFloat> accumulatorScale =
      constantFloatOperand(toReal, realBody[1]->getOperand(1));
  if (!accumulatorScale) {
    return std::nullopt;
  }
  found.accumulatorScale = *accumulatorScale;

  // Divide by y_scale, round, offset, clamp to the ui8 range, narrow.
  linalg::GenericOp narrow = soleGenericConsumer(toReal.getResult(0));
  if (!narrow || narrow.getNumDpsInputs() != 1 ||
      !isElementwiseWithIdentityMaps(narrow)) {
    return std::nullopt;
  }
  SmallVector<Operation *> narrowBody =
      matchBody(narrow, {"arith.divf", "math.roundeven", "arith.addf",
                         "arith.maximumf", "arith.minimumf", "arith.fptoui"});
  if (narrowBody.empty()) {
    return std::nullopt;
  }
  auto narrowType = dyn_cast<RankedTensorType>(narrow.getResult(0).getType());
  if (!narrowType || !narrowType.getElementType().isSignlessInteger(8)) {
    return std::nullopt;
  }
  std::optional<APFloat> outputScale =
      constantFloatOperand(narrow, narrowBody[0]->getOperand(1));
  std::optional<APFloat> outputZeroPoint =
      constantFloatOperand(narrow, narrowBody[2]->getOperand(1));
  // The clamp bounds say which encoding this is. Only the unsigned one is
  // handled: a signed epilogue would already be in the hardware's own range
  // and would not need the shift below, but no measured model emits one, so
  // it is declined rather than guessed at.
  if (!outputScale || !outputZeroPoint ||
      !isFloatValue(constantFloatOperand(narrow, narrowBody[3]->getOperand(1)),
                    kUnsignedLow) ||
      !isFloatValue(constantFloatOperand(narrow, narrowBody[4]->getOperand(1)),
                    kUnsignedHigh)) {
    return std::nullopt;
  }
  found.outputScale = *outputScale;
  found.outputZeroPoint = *outputZeroPoint;
  found.narrow = narrow;
  return found;
}

/// Emits `bias - x_zp * sum_k(w)` as a rank-1 generic. Both operands are
/// constants in a real model, so this folds away entirely.
static Value buildFoldedBias(PatternRewriter &rewriter, Location loc,
                             Value bias, Value weightSum,
                             int64_t inputZeroPoint) {
  auto sumType = cast<RankedTensorType>(weightSum.getType());
  Value empty = tensor::EmptyOp::create(rewriter, loc, sumType.getShape(),
                                        sumType.getElementType());
  SmallVector<Value> inputs;
  if (bias) {
    inputs.push_back(bias);
  }
  inputs.push_back(weightSum);
  AffineMap identity = rewriter.getMultiDimIdentityMap(1);
  SmallVector<AffineMap> maps(inputs.size() + 1, identity);
  SmallVector<utils::IteratorType> iterators{utils::IteratorType::parallel};
  auto generic = linalg::GenericOp::create(
      rewriter, loc, TypeRange{empty.getType()}, inputs, ValueRange{empty},
      maps, iterators,
      [&](OpBuilder &builder, Location nested, ValueRange args) {
        Value sum = bias ? args[1] : args[0];
        Value zeroPoint = arith::ConstantOp::create(
            builder, nested, builder.getI32IntegerAttr(inputZeroPoint));
        Value scaled = arith::MulIOp::create(builder, nested, sum, zeroPoint);
        Value result =
            bias ? arith::SubIOp::create(builder, nested, args[0], scaled)
                       .getResult()
                 : arith::SubIOp::create(
                       builder, nested,
                       arith::ConstantOp::create(
                           builder, nested, builder.getI32IntegerAttr(0)),
                       scaled)
                       .getResult();
        linalg::YieldOp::create(builder, nested, result);
      });
  return generic.getResult(0);
}

/// Rewrites one convolution and its epilogue into the canonical form.
/// Rewrites one convolution and its epilogue into the canonical form.
///
/// Templated over the convolution op because dense and depthwise differ in
/// nothing this rewrite touches: both take (input, filter) with the filter
/// second, both accumulate `i32` into a single init, both produce NHWC, and
/// `quantized-conv-to-conv` gives both the identical five-op epilogue. The
/// depthwise filter reduction is over `[kh, kw]` rather than `[kh, kw, cin]`,
/// which changes the reduction's indexing maps and not its body, and
/// `matchWeightSum` checks the body and the operand identity rather than the
/// maps.
template <typename ConvOp>
struct FuseInt8RequantEpilogue : public OpRewritePattern<ConvOp> {
  using OpRewritePattern<ConvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvOp convOp,
                                PatternRewriter &rewriter) const override {
    auto accType = dyn_cast<RankedTensorType>(convOp.getResult(0).getType());
    if (!accType || !accType.getElementType().isSignlessInteger(32)) {
      return rewriter.notifyMatchFailure(convOp, "not an i32 convolution");
    }
    for (Value input : convOp.getDpsInputs()) {
      auto type = dyn_cast<RankedTensorType>(input.getType());
      if (!type || !type.getElementType().isSignlessInteger(8)) {
        return rewriter.notifyMatchFailure(convOp, "operands are not i8");
      }
    }
    std::optional<RequantEpilogue> epilogue = matchEpilogue(convOp);
    if (!epilogue) {
      return rewriter.notifyMatchFailure(convOp, "no requantization epilogue");
    }

    Location loc = convOp.getLoc();
    // Build at the *end* of the chain, not at the convolution. The folded
    // bias reads the filter reduction, and `quantized-conv-to-conv` emits
    // that reduction after the convolution it corrects -- inserting at the
    // convolution puts the new bias above its own operand and the rewrite
    // fails to verify. Everything the replacement reads is defined before
    // the op being replaced, so that is the one position that always works.
    rewriter.setInsertionPoint(epilogue->narrow);

    // The convolution keeps its operands and loses its bias: a zero init, so
    // the accumulator it produces is the plain contraction the requantized
    // matcher expects to see.
    Value zero = arith::ConstantOp::create(
        rewriter, loc, rewriter.getI32IntegerAttr(0));
    Value accEmpty = tensor::EmptyOp::create(
        rewriter, loc, accType.getShape(), accType.getElementType());
    Value accInit =
        linalg::FillOp::create(rewriter, loc, ValueRange{zero},
                               ValueRange{accEmpty})
            .getResult(0);
    // Strides and dilations are re-typed as `tensor<2xi64>`, and this is not
    // cosmetic. `transform.iree.match.cast_compatible_dag_from_root` compares
    // whole attribute dictionaries, and the DAG template in
    // `@match_dynamic_conv2d_int8_requant` spells them `dense<1> :
    // tensor<2xi64>` -- what hand-written MLIR produces, and what the
    // canonical fixtures carry. A convolution that came through
    // `iree-preprocessing-convert-conv-to-channels-last` carries `dense<1> :
    // vector<2xi64>` instead: the same values, a different attribute type,
    // and the matcher silently declines it. Measured against the fixture,
    // which reaches the matcher with `tensor` and matches, while the model
    // reached it with `vector` and did not.
    auto attrType = RankedTensorType::get({2}, rewriter.getIntegerType(64));
    SmallVector<int64_t> strideValues(convOp.getStrides().template getValues<int64_t>());
    SmallVector<int64_t> dilationValues(
        convOp.getDilations().template getValues<int64_t>());
    auto fusedConv = ConvOp::create(
        rewriter, loc, TypeRange{accType}, convOp.getDpsInputs(),
        ValueRange{accInit},
        DenseIntElementsAttr::get(attrType, strideValues),
        DenseIntElementsAttr::get(attrType, dilationValues));

    Value foldedBias = buildFoldedBias(rewriter, loc, epilogue->bias,
                                       epilogue->weightSum,
                                       epilogue->inputZeroPoint);

    // One rounding of the ratio rather than a multiply and a divide; see the
    // numerics note in this file's header.
    APFloat scale = epilogue->accumulatorScale;
    scale.divide(epilogue->outputScale, APFloat::rmNearestTiesToEven);
    // The DPU writes signed, the model reads unsigned: shift the zero point
    // by the same 128 the compensating op below puts back.
    APFloat signedZeroPoint = epilogue->outputZeroPoint;
    signedZeroPoint.add(APFloat(static_cast<float>(kEncodingShift)),
                        APFloat::rmNearestTiesToEven);
    int64_t zeroPointInt =
        static_cast<int64_t>(signedZeroPoint.convertToDouble());

    Value scaleValue = arith::ConstantOp::create(
        rewriter, loc, rewriter.getF32FloatAttr(scale.convertToDouble()));
    Value zeroPointValue = arith::ConstantOp::create(
        rewriter, loc, rewriter.getI32IntegerAttr(zeroPointInt));
    Value lowValue = arith::ConstantOp::create(
        rewriter, loc, rewriter.getF32FloatAttr(kSignedLow));
    Value highValue = arith::ConstantOp::create(
        rewriter, loc, rewriter.getF32FloatAttr(kSignedHigh));

    auto int8Type =
        RankedTensorType::get(accType.getShape(), rewriter.getI8Type());
    Value outEmpty = tensor::EmptyOp::create(
        rewriter, loc, int8Type.getShape(), int8Type.getElementType());

    unsigned rank = accType.getRank();
    AffineMap identity = rewriter.getMultiDimIdentityMap(rank);
    AffineMap channel = AffineMap::get(
        rank, 0, {rewriter.getAffineDimExpr(rank - 1)}, rewriter.getContext());
    AffineMap scalar = AffineMap::get(rank, 0, {}, rewriter.getContext());
    SmallVector<AffineMap> maps{identity, channel, scalar,
                                scalar,   scalar,  scalar, identity};
    SmallVector<utils::IteratorType> iterators(rank,
                                               utils::IteratorType::parallel);

    auto requant = linalg::GenericOp::create(
        rewriter, loc, TypeRange{int8Type},
        ValueRange{fusedConv.getResult(0), foldedBias, scaleValue,
                   zeroPointValue, lowValue, highValue},
        ValueRange{outEmpty}, maps, iterators,
        [&](OpBuilder &builder, Location nested, ValueRange args) {
          Value biased = arith::AddIOp::create(builder, nested, args[0], args[1]);
          Value real = arith::SIToFPOp::create(builder, nested,
                                               builder.getF32Type(), biased);
          Value scaled = arith::MulFOp::create(builder, nested, real, args[2]);
          Value rounded = math::RoundEvenOp::create(builder, nested, scaled);
          Value zeroPointReal = arith::SIToFPOp::create(
              builder, nested, builder.getF32Type(), args[3]);
          Value offset =
              arith::AddFOp::create(builder, nested, rounded, zeroPointReal);
          Value low = arith::MaximumFOp::create(builder, nested, offset, args[4]);
          Value clamped =
              arith::MinimumFOp::create(builder, nested, low, args[5]);
          Value narrowed = arith::FPToSIOp::create(builder, nested,
                                                   builder.getI8Type(), clamped);
          linalg::YieldOp::create(builder, nested, narrowed);
        });

    // Back to the unsigned encoding the rest of the graph reads.
    Value shiftEmpty = tensor::EmptyOp::create(
        rewriter, loc, int8Type.getShape(), int8Type.getElementType());
    SmallVector<AffineMap> shiftMaps{identity, identity};
    auto shifted = linalg::GenericOp::create(
        rewriter, loc, TypeRange{int8Type}, ValueRange{requant.getResult(0)},
        ValueRange{shiftEmpty}, shiftMaps, iterators,
        [&](OpBuilder &builder, Location nested, ValueRange args) {
          Value shift = arith::ConstantOp::create(
              builder, nested,
              builder.getIntegerAttr(builder.getI8Type(), kEncodingShift));
          Value result = arith::AddIOp::create(builder, nested, args[0], shift);
          linalg::YieldOp::create(builder, nested, result);
        });

    Value result = shifted.getResult(0);
    if (epilogue->transpose) {
      // The layout change now moves i8, not i32.
      auto transposedType = cast<RankedTensorType>(
          epilogue->transpose.getResult()[0].getType());
      Value transposeEmpty = tensor::EmptyOp::create(
          rewriter, loc, transposedType.getShape(), rewriter.getI8Type());
      result = linalg::TransposeOp::create(
                   rewriter, loc, result, transposeEmpty,
                   epilogue->transpose.getPermutation())
                   .getResult()[0];
    }

    rewriter.replaceOp(epilogue->narrow, result);
    return success();
  }
};

struct RocketFuseInt8RequantEpiloguePass
    : public PassWrapper<RocketFuseInt8RequantEpiloguePass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RocketFuseInt8RequantEpiloguePass)

  StringRef getArgument() const final {
    return "rocket-fuse-int8-requant-epilogue";
  }
  StringRef getDescription() const final {
    return "Collapses an ONNX QLinearConv's requantization epilogue into the "
           "single elementwise generic the requantized int8 conv matcher is "
           "written against, folding the zero-point correction into the bias "
           "and moving the layout transpose off the i32 tensor.";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    math::MathDialect, tensor::TensorDialect>();
  }

  void runOnOperation() final {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<FuseInt8RequantEpilogue<linalg::Conv2DNhwcHwcfOp>,
                 FuseInt8RequantEpilogue<linalg::DepthwiseConv2DNhwcHwcOp>>(
        context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

static PassRegistration<RocketFuseInt8RequantEpiloguePass> reg;

} // namespace
} // namespace mlir::iree_compiler::IREE::HAL
