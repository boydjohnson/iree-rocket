// Copyright 2026
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Rewrites a convolution and the ReLU6 that follows it into the two-op form
// the fused-activation matcher is written against, so the clamp can travel
// into the convolution's own BN stage instead of costing a CPU dispatch.
//
// On MobileNetV2 fp16, 18 of the model's 35 ReLU6 sites follow an offloaded
// convolution, and every one of them is a *standalone* dispatch -- IREE
// cannot fuse across a Rocket dispatch boundary, which is ISSUES.md P8's
// central measurement. Together they read and write 43.8 MB of f32 per
// inference to do four instructions per element. The other 17 follow a CPU
// depthwise convolution and IREE fuses them completely, which is what the
// contrast looks like.
//
// What arrives, after `rocket-demote-conv-inputs-to-f16` and the
// channels-last conversion:
//
//   %conv  = conv_2d_nhwc_hwcf(%in, %w) outs(broadcast(%bias))  // f16, f32 out
//   %exp   = tensor.expand_shape %conv                          // 1xHxWxC -> 1x1xHxWxC
//   %clamp = generic(%exp, %lo, %hi)                            // cmpf/select x2
//
// and what leaves:
//
//   %conv = conv_2d_nhwc_hwcf(%in, %w) outs(fill 0.0)
//   %act  = generic(%conv, %bias: tensor<?xf32>, %lo: f32, %hi: f32)
//                                                  // addf, maximumf, minimumf
//   %exp  = tensor.expand_shape %act
//
// The clamp moves *in front of* the reshape rather than the reshape being
// absorbed. Both are elementwise-vs-pure-reshape so they commute, and it
// leaves the pad, transpose and whatever else follows completely untouched
// -- the rewrite is local to three ops. The reshape has to be got out of the
// way because `transform.iree.match.cast_compatible_dag_from_root` compares
// whole attribute dictionaries, and a `tensor.expand_shape`'s output shape
// is a static attribute that differs at every site, so no DAG template
// spanning one can match more than a single convolution.
//
// Three things worth knowing about why it is shaped this way.
//
// **The bounds become scalar operands.** They arrive as full-size splat
// constant tensors with identity maps -- 2.4 MB of `dense<6.0>` for the
// first site. The canonical form takes them as zero-rank operands instead,
// which is both what the requantized matcher already spells and the only
// form that survives to the match loop: a constant written inside a generic
// body is hoisted out by the canonicaliser, and a value captured from
// outside the region can never match, because the DAG matcher builds its
// value mapping only from the ops it walked. `transform-dag-matcher-traps`
// records that hazard from the other side.
//
// **The body becomes `maximumf`/`minimumf`.** The `cmpf ult`/`select` pair
// the ONNX importer emits is the same function on non-NaN input and a
// longer DAG to match. Matching the arriving form and re-emitting the short
// one keeps the matcher's template readable and means a body that is
// *nearly* this shape declines rather than being quietly accepted.
//
// **Only a ceiling of exactly 6.0 is rewritten, and that is load-bearing.**
// The wire's `activation_cmp` is a static attribute on the executable
// target (`#rocket_dynamic_relu6_target` spells the f32 bit pattern of 6.0),
// so the canonical form encodes exactly one ceiling. The matcher cannot
// check a constant's value -- it matches structure -- so what guarantees the
// target's attribute is right is that this pass never produces the canonical
// form for any other ceiling. WIDENING THE CEILING SET HERE WITHOUT ALSO
// MAKING `activation_cmp` A RUNTIME PUSH CONSTANT WOULD SILENTLY COMPILE
// EVERY OTHER CEILING AS 6.0. `Conv2DDef.runtime_quantization` is the
// mechanism to reach for if that is ever wanted; it already carries
// `output_scale` and `output_zero_point` the same way.
//
// **The bias has to move, and that is the reason this is not a one-line
// change.** The hardware order is accumulate -> BS (bias) -> BN (activation)
// -> OUT_CVT, so a clamp in BN sees the *biased* value -- which is what the
// model asks for, but only if the bias is on the BS plane. Every fp16
// convolution this project compiles today hands the NPU a zero bias and adds
// the real one back in the CPU epilogue shim, so turning BN on without
// moving the bias would clamp the unbiased accumulator: a different
// function, silently. So the canonical form lifts the per-channel bias out
// of the convolution's init and into the epilogue generic, exactly as
// `RocketFuseInt8RequantEpiloguePass` does and for exactly the same reason
// -- that is the shape the hardware computes.
//
// That the bias reaches the BS plane correctly, and that both activations
// see it, is measured rather than assumed: `conv_fp16_bias_activation_hw`
// drives a nonzero per-channel bias with `acc + bias` crossing both ends of
// the ReLU6 range and is exact for bias alone, bias + ReLU and bias + ReLU6.
// A convolution whose init is not a per-channel broadcast is declined rather
// than guessed at.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "rocket-fuse-conv-relu6"

namespace mlir::iree_compiler::IREE::HAL {
namespace {

/// The only ceiling this pass rewrites. See the file comment: the wire
/// carries it as a static attribute, so the guarantee that it is 6.0 lives
/// here and nowhere else.
constexpr double kRelu6Ceiling = 6.0;

/// The `f32` a value carries when it is a constant scalar, a constant splat
/// tensor, or a broadcast/fill of either.
///
/// Three spellings occur, and which one is present depends entirely on where
/// in the pipeline this runs. At the point this pass sits the bounds are
/// `linalg.broadcast` of a rank-0 `dense<0.0>` / `dense<6.0>`; a later
/// canonicalisation folds those into full-size splat constants, and a later
/// one still into scalars. Matching only the folded forms is the same
/// mistake as matching only the folded bias init -- it reads correctly on an
/// end-of-preprocessing dump and fires on nothing in the real pipeline.
static std::optional<double> constantSplatFloat(Value value) {
  // Peel a broadcast or a fill: both are "this whole tensor is one value",
  // which is all this needs to know.
  if (auto broadcast = value.getDefiningOp<linalg::BroadcastOp>()) {
    value = broadcast.getInput();
  } else if (auto fill = value.getDefiningOp<linalg::FillOp>()) {
    value = fill.getInputs()[0];
  }
  Attribute attr;
  if (!matchPattern(value, m_Constant(&attr))) {
    return std::nullopt;
  }
  if (auto floatAttr = dyn_cast<FloatAttr>(attr)) {
    return floatAttr.getValueAsDouble();
  }
  auto elements = dyn_cast<DenseFPElementsAttr>(attr);
  if (!elements || !elements.isSplat()) {
    return std::nullopt;
  }
  return elements.getSplatValue<APFloat>().convertToDouble();
}

/// The clamp's two bounds, in the order the body applies them.
struct ClampBounds {
  Value low;
  Value high;
};

/// Recognises the ReLU6 body the ONNX importer emits:
///
///   %c0 = arith.cmpf ult, %value, %low
///   %s0 = arith.select %c0, %low, %value
///   %c1 = arith.cmpf ugt, %s0, %high
///   %s1 = arith.select %c1, %high, %s0
///   linalg.yield %s1
///
/// Returns which block arguments carry the bounds, or nullopt on any
/// deviation. Spelled as an exact sequence rather than a pattern DSL for the
/// same reason `RocketFuseInt8RequantEpiloguePass` spells its bodies that
/// way: a body that is only *similar* to this one is a different function,
/// and accepting it would miscompile rather than decline.
static std::optional<ClampBounds> matchClampBody(linalg::GenericOp op) {
  Block &body = op.getRegion().front();
  auto it = body.begin();
  auto next = [&](StringRef name) -> Operation * {
    if (it == body.end() || it->getName().getStringRef() != name) {
      return nullptr;
    }
    return &*it++;
  };

  auto *lowCmp = next("arith.cmpf");
  auto *lowSel = next("arith.select");
  auto *highCmp = next("arith.cmpf");
  auto *highSel = next("arith.select");
  if (!lowCmp || !lowSel || !highCmp || !highSel) {
    return std::nullopt;
  }
  if (it == body.end() || !isa<linalg::YieldOp>(*it) ||
      std::next(it) != body.end()) {
    return std::nullopt;
  }

  auto lowPredicate = cast<arith::CmpFOp>(lowCmp).getPredicate();
  auto highPredicate = cast<arith::CmpFOp>(highCmp).getPredicate();
  if (lowPredicate != arith::CmpFPredicate::ULT ||
      highPredicate != arith::CmpFPredicate::UGT) {
    return std::nullopt;
  }

  Value value = body.getArgument(0);
  Value low = lowCmp->getOperand(1);
  Value high = highCmp->getOperand(1);

  // The comparisons and the selects have to agree, and the two stages have
  // to be chained value -> low-clamped -> high-clamped.
  if (lowCmp->getOperand(0) != value || lowSel->getOperand(0) != lowCmp->getResult(0) ||
      lowSel->getOperand(1) != low || lowSel->getOperand(2) != value) {
    return std::nullopt;
  }
  if (highCmp->getOperand(0) != lowSel->getResult(0) ||
      highSel->getOperand(0) != highCmp->getResult(0) ||
      highSel->getOperand(1) != high ||
      highSel->getOperand(2) != lowSel->getResult(0)) {
    return std::nullopt;
  }
  if (cast<linalg::YieldOp>(*it).getOperand(0) != highSel->getResult(0)) {
    return std::nullopt;
  }
  return ClampBounds{low, high};
}

/// The `ins` operand a body block argument reads, or null when the argument
/// is not one of them.
static Value insOperandFor(linalg::GenericOp op, Value blockArg) {
  auto arg = dyn_cast<BlockArgument>(blockArg);
  if (!arg || arg.getOwner() != &op.getRegion().front()) {
    return nullptr;
  }
  unsigned index = arg.getArgNumber();
  if (index >= op.getNumDpsInputs()) {
    return nullptr;
  }
  return op.getDpsInputs()[index];
}

/// Whether every indexing map is an identity or a scalar (zero-result) map
/// and every loop is parallel -- the shape a clamp arrives in.
static bool isElementwiseGeneric(linalg::GenericOp op) {
  if (op.getNumResults() != 1 || !op.isAllParallelLoops()) {
    return false;
  }
  return llvm::all_of(op.getIndexingMapsArray(), [](AffineMap map) {
    return map.isIdentity() || map.getNumResults() == 0;
  });
}

/// The single consumer of `value`, or null when it has any other count.
static Operation *soleConsumer(Value value) {
  if (!value.hasOneUse()) {
    return nullptr;
  }
  return *value.getUsers().begin();
}

/// The rank-1 `f32` per-channel bias a convolution's init broadcasts.
///
/// The channel is the last dimension and every other one is added by the
/// broadcast, which is what "per output channel" means in NHWC. Anything
/// else -- a zero fill, a broadcast over a different axis set, a bias of the
/// wrong element type -- is declined: the fused form puts this vector on the
/// BS plane, and a value that is not a per-channel constant does not belong
/// there.
/// Two spellings, and matching only the folded one is the documented way to
/// write a pass that fires on an end-of-preprocessing dump and on nothing at
/// all in the real pipeline. `iree-preprocessing-convert-conv-to-channels-last`
/// puts a transpose in front of the bias broadcast rather than rewriting the
/// broadcast, so at this point the init is `transpose(broadcast(bias))`; it
/// is a bare broadcast only after a later canonicalisation folds the pair.
/// `RocketFuseInt8RequantEpiloguePass::matchBiasInit` records the same
/// hazard, and this pass hit it too -- the first version matched only the
/// bare broadcast and fused zero of MobileNetV2's 18 sites.
static Value matchPerChannelBias(Value init, int64_t &channelDim) {
  auto transpose = init.getDefiningOp<linalg::TransposeOp>();
  auto broadcast = transpose
                       ? transpose.getInput().getDefiningOp<linalg::BroadcastOp>()
                       : init.getDefiningOp<linalg::BroadcastOp>();
  if (!broadcast) {
    return nullptr;
  }

  // The broadcast adds every dimension but one; that surviving dimension is
  // the channel. Which *position* it occupies in the convolution's own result
  // depends on the layout -- last for NHWC, dim 1 for the NCHW depthwise
  // form -- so it is reported rather than assumed.
  auto broadcastType = cast<RankedTensorType>(broadcast.getResult()[0].getType());
  llvm::SmallDenseSet<int64_t> added(broadcast.getDimensions().begin(),
                                     broadcast.getDimensions().end());
  std::optional<int64_t> retained;
  for (int64_t dim = 0; dim < broadcastType.getRank(); ++dim) {
    if (!added.contains(dim)) {
      if (retained) {
        return nullptr;
      }
      retained = dim;
    }
  }
  if (!retained) {
    return nullptr;
  }
  channelDim = *retained;
  if (transpose) {
    // The transpose relabels it: find where the retained dimension lands.
    ArrayRef<int64_t> permutation = transpose.getPermutation();
    const auto *position = llvm::find(permutation, *retained);
    if (position == permutation.end()) {
      return nullptr;
    }
    channelDim = std::distance(permutation.begin(), position);
  }
  Value bias = broadcast.getInput();
  auto biasType = dyn_cast<RankedTensorType>(bias.getType());
  if (!biasType || biasType.getRank() != 1 || !biasType.getElementType().isF32()) {
    return nullptr;
  }
  return bias;
}

template <typename ConvOp>
struct FuseConvRelu6 : public OpRewritePattern<ConvOp> {
  using OpRewritePattern<ConvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvOp conv,
                                PatternRewriter &rewriter) const override {
    if (conv.getNumResults() != 1) {
      LLVM_DEBUG(llvm::dbgs() << "declined: multiple results\n");
      return failure();
    }
    Value convResult = conv->getResult(0);
    auto convType = dyn_cast<RankedTensorType>(convResult.getType());
    if (!convType || !convType.getElementType().isF32()) {
      LLVM_DEBUG(llvm::dbgs() << "declined: non-f32 result\n");
      return failure();
    }
    // Only the fp16 path fuses. The int8 paths clamp in a different domain
    // (the BN stage sits before OUT_CVT, so an int8 ceiling is in
    // post-BS accumulator units -- `Activation::clamped_int8`) and reach the
    // hardware through their own matchers.
    for (Value input : conv.getDpsInputs()) {
      auto inputType = dyn_cast<RankedTensorType>(input.getType());
      if (!inputType || !inputType.getElementType().isF16()) {
        LLVM_DEBUG(llvm::dbgs() << "declined: non-f16 operand\n");
        return failure();
      }
    }

    // conv -> [transpose] -> [expand_shape] -> clamp, each with a single use,
    // because the rewrite deletes what it walks through.
    //
    // Both relayouts are optional and both are re-emitted below. The NHWC
    // dense form has only the reshape; the NCHW depthwise form has a
    // NCHW -> NHWC transpose in front of it, because the channels-last
    // conversion leaves the depthwise convolution itself in NCHW and puts the
    // layout change on its result instead.
    Operation *consumer = soleConsumer(convResult);
    if (!consumer) {
      LLVM_DEBUG(llvm::dbgs() << "declined: convolution result has no sole consumer\n");
      return failure();
    }
    auto transpose = dyn_cast<linalg::TransposeOp>(consumer);
    if (transpose) {
      consumer = soleConsumer(transpose.getResult()[0]);
      if (!consumer) {
        LLVM_DEBUG(llvm::dbgs() << "declined: transpose has no sole consumer\n");
        return failure();
      }
    }
    auto expand = dyn_cast<tensor::ExpandShapeOp>(consumer);
    if (expand) {
      consumer = soleConsumer(expand.getResult());
      if (!consumer) {
        LLVM_DEBUG(llvm::dbgs() << "declined: reshape has no sole consumer\n");
        return failure();
      }
    }
    auto clamp = dyn_cast<linalg::GenericOp>(consumer);
    if (!clamp || !isElementwiseGeneric(clamp) || clamp.getNumDpsInputs() != 3) {
      LLVM_DEBUG(llvm::dbgs() << "declined: consumer is not an elementwise 3-input generic\n");
      return failure();
    }
    // The convolution has to be the clamped value, not one of the bounds.
    Value clamped = convResult;
    if (transpose) {
      clamped = transpose.getResult()[0];
    }
    if (expand) {
      clamped = expand.getResult();
    }
    if (clamp.getDpsInputs()[0] != clamped) {
      LLVM_DEBUG(llvm::dbgs() << "declined: convolution is not the clamped operand\n");
      return failure();
    }

    std::optional<ClampBounds> bounds = matchClampBody(clamp);
    if (!bounds) {
      LLVM_DEBUG(llvm::dbgs() << "declined: body is not the ReLU6 sequence\n");
      return failure();
    }
    Value lowOperand = insOperandFor(clamp, bounds->low);
    Value highOperand = insOperandFor(clamp, bounds->high);
    if (!lowOperand || !highOperand) {
      LLVM_DEBUG(llvm::dbgs() << "declined: clamp bounds are not ins operands\n");
      return failure();
    }
    std::optional<double> low = constantSplatFloat(lowOperand);
    std::optional<double> high = constantSplatFloat(highOperand);
    if (!low || !high || *low != 0.0 || *high != kRelu6Ceiling) {
      LLVM_DEBUG(llvm::dbgs() << "declined: bounds are not the constants 0.0 and 6.0\n");
      return failure();
    }

    int64_t channelDim = 0;
    Value bias = matchPerChannelBias(conv.getDpsInits()[0], channelDim);
    if (!bias) {
      LLVM_DEBUG(llvm::dbgs() << "declined: init is not a per-channel bias broadcast\n");
      return failure();
    }

    Location loc = conv.getLoc();
    // Everything is built at the clamp, not after the convolution. The
    // relayouts this re-emits are clones, and they carry their original `outs`
    // operands -- `tensor.empty` ops that sit *between* the convolution and
    // the clamp. Building after the convolution puts a clone in front of its
    // own operand and fails the dominance check.
    rewriter.setInsertionPoint(clamp);

    int64_t rank = convType.getRank();
    MLIRContext *context = rewriter.getContext();
    // The bias is indexed by the channel alone -- whichever dimension that is
    // in this convolution's own layout, which is the last one for NHWC and
    // dimension 1 for the NCHW depthwise form -- and the bounds are scalars.
    AffineMap identity = rewriter.getMultiDimIdentityMap(rank);
    AffineMap channel =
        AffineMap::get(rank, 0, {rewriter.getAffineDimExpr(channelDim)}, context);
    AffineMap scalar = AffineMap::get(rank, 0, context);
    SmallVector<AffineMap> maps{identity, channel, scalar, scalar, identity};
    SmallVector<utils::IteratorType> iterators(rank,
                                               utils::IteratorType::parallel);

    // The convolution accumulates over a zero init now: the bias it used to
    // be seeded with is applied by the epilogue below, which is the shape
    // the DPU's BS plane computes.
    SmallVector<OpFoldResult> sizes =
        tensor::getMixedSizes(rewriter, loc, convResult);
    Value zero = arith::ConstantOp::create(
        rewriter, loc, rewriter.getF32FloatAttr(0.0f));
    Value accEmpty = tensor::EmptyOp::create(rewriter, loc, sizes,
                                             convType.getElementType());
    Value accInit =
        linalg::FillOp::create(rewriter, loc, ValueRange{zero},
                               ValueRange{accEmpty})
            .getResult(0);
    auto rawConv = cast<ConvOp>(rewriter.clone(*conv.getOperation()));
    rawConv.getDpsInitsMutable().assign(accInit);

    Value lowScalar = arith::ConstantOp::create(
        rewriter, loc, rewriter.getF32FloatAttr(static_cast<float>(*low)));
    Value highScalar = arith::ConstantOp::create(
        rewriter, loc, rewriter.getF32FloatAttr(static_cast<float>(*high)));
    Value empty =
        tensor::EmptyOp::create(rewriter, loc, sizes, convType.getElementType());

    auto activated = linalg::GenericOp::create(
        rewriter, loc, TypeRange{convType},
        ValueRange{rawConv->getResult(0), bias, lowScalar, highScalar},
        ValueRange{empty}, maps, iterators,
        [&](OpBuilder &builder, Location nested, ValueRange args) {
          Value biased =
              arith::AddFOp::create(builder, nested, args[0], args[1]);
          Value lowClamped =
              arith::MaximumFOp::create(builder, nested, biased, args[2]);
          Value result =
              arith::MinimumFOp::create(builder, nested, lowClamped, args[3]);
          linalg::YieldOp::create(builder, nested, result);
        });

    // The relayouts, if there were any, are re-emitted on the clamped value in
    // the same order, so everything downstream sees the shape it expects.
    Value result = activated.getResult(0);
    if (transpose) {
      IRMapping mapping;
      mapping.map(transpose.getInput(), result);
      result = rewriter.clone(*transpose.getOperation(), mapping)->getResult(0);
    }
    if (expand) {
      IRMapping mapping;
      mapping.map(expand.getSrc(), result);
      result = rewriter.clone(*expand.getOperation(), mapping)->getResult(0);
    }
    rewriter.replaceOp(clamp, result);
    return success();
  }
};

/// The ReLU body the ONNX importer emits, in f16 or f32:
///
///   %c = arith.cmpf ugt, %x, %zero
///   %s = arith.select %c, %x, %zero
///   linalg.yield %s
///
/// `zero` is either an `ins` operand or a constant captured from outside the
/// region -- the f16-import pipeline produces the captured spelling, with a
/// scalar `arith.constant 0.0 : f16` shared by every ReLU in the function.
/// Returns the zero value, or nullopt on any deviation.
static std::optional<Value> matchReluBody(linalg::GenericOp op) {
  if (op.getNumDpsInputs() != 1) {
    return std::nullopt;
  }
  Block &body = op.getRegion().front();
  auto it = body.begin();
  auto cmp = (it != body.end()) ? dyn_cast<arith::CmpFOp>(&*it) : nullptr;
  if (!cmp) {
    return std::nullopt;
  }
  ++it;
  auto select = (it != body.end()) ? dyn_cast<arith::SelectOp>(&*it) : nullptr;
  if (!select) {
    return std::nullopt;
  }
  ++it;
  if (it == body.end() || !isa<linalg::YieldOp>(*it) ||
      std::next(it) != body.end()) {
    return std::nullopt;
  }
  Value value = body.getArgument(0);
  Value zero = cmp.getRhs();
  if (cmp.getPredicate() != arith::CmpFPredicate::UGT || cmp.getLhs() != value ||
      select.getCondition() != cmp.getResult() ||
      select.getTrueValue() != value || select.getFalseValue() != zero ||
      cast<linalg::YieldOp>(*it).getOperand(0) != select.getResult()) {
    return std::nullopt;
  }
  return zero;
}

/// Whether `zero`, as used by `op`'s body, is the constant 0.0 in either
/// spelling `matchReluBody` accepts.
static bool isZeroFor(linalg::GenericOp op, Value zero) {
  if (Value operand = insOperandFor(op, zero)) {
    std::optional<double> value = constantSplatFloat(operand);
    return value && *value == 0.0;
  }
  std::optional<double> value = constantSplatFloat(zero);
  return value && *value == 0.0;
}

/// A one-input elementwise generic whose whole body is one `arith.truncf`
/// -- the narrowing an f16 import puts behind every convolution, since the
/// convolution accumulates in f32 and the model carries f16.
static bool isTruncfGeneric(linalg::GenericOp op) {
  if (!isElementwiseGeneric(op) || op.getNumDpsInputs() != 1) {
    return false;
  }
  Block &body = op.getRegion().front();
  auto it = body.begin();
  if (it == body.end() || !isa<arith::TruncFOp>(*it)) {
    return false;
  }
  auto truncf = cast<arith::TruncFOp>(*it);
  ++it;
  return it != body.end() && isa<linalg::YieldOp>(*it) &&
         std::next(it) == body.end() &&
         truncf.getOperand() == body.getArgument(0) &&
         cast<linalg::YieldOp>(*it).getOperand(0) == truncf.getResult();
}

/// Whether this is a dense convolution the fused-epilogue targets exist for:
/// a 1x1 or 3x3 filter at stride 1 or 2, f16 in, f32 out. The depthwise
/// forms keep their own matchers and are left to the ReLU6 pattern only.
static bool isFusableDenseConv(linalg::Conv2DNhwcHwcfOp conv) {
  auto convType = dyn_cast<RankedTensorType>(conv->getResult(0).getType());
  if (!convType || !convType.getElementType().isF32()) {
    return false;
  }
  for (Value input : conv.getDpsInputs()) {
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType || !inputType.getElementType().isF16()) {
      return false;
    }
  }
  auto filterType = cast<RankedTensorType>(conv.getDpsInputs()[1].getType());
  int64_t kh = filterType.getDimSize(0);
  int64_t kw = filterType.getDimSize(1);
  if (!((kh == 1 && kw == 1) || (kh == 3 && kw == 3))) {
    return false;
  }
  SmallVector<int64_t> strides(conv.getStrides().getValues<int64_t>());
  return strides.size() == 2 && strides[0] == strides[1] &&
         (strides[0] == 1 || strides[0] == 2);
}

/// Builds the canonical fused epilogue for `conv`: a clone of the
/// convolution over a zero init, then one generic adding the per-channel
/// `bias` and, when `relu` is set, clamping at zero. Returns the generic's
/// f32 result. The insertion point is the caller's.
static Value buildFusedEpilogue(PatternRewriter &rewriter, Location loc,
                                linalg::Conv2DNhwcHwcfOp conv, Value bias,
                                int64_t channelDim, bool relu) {
  Value convResult = conv->getResult(0);
  auto convType = cast<RankedTensorType>(convResult.getType());
  int64_t rank = convType.getRank();
  MLIRContext *context = rewriter.getContext();
  AffineMap identity = rewriter.getMultiDimIdentityMap(rank);
  AffineMap channel =
      AffineMap::get(rank, 0, {rewriter.getAffineDimExpr(channelDim)}, context);
  AffineMap scalar = AffineMap::get(rank, 0, context);
  SmallVector<utils::IteratorType> iterators(rank,
                                             utils::IteratorType::parallel);
  SmallVector<OpFoldResult> sizes =
      tensor::getMixedSizes(rewriter, loc, convResult);
  Value zero = arith::ConstantOp::create(
      rewriter, loc, rewriter.getF32FloatAttr(0.0f));
  Value accEmpty = tensor::EmptyOp::create(rewriter, loc, sizes,
                                           convType.getElementType());
  Value accInit =
      linalg::FillOp::create(rewriter, loc, ValueRange{zero},
                             ValueRange{accEmpty})
          .getResult(0);
  auto rawConv =
      cast<linalg::Conv2DNhwcHwcfOp>(rewriter.clone(*conv.getOperation()));
  rawConv.getDpsInitsMutable().assign(accInit);
  Value empty =
      tensor::EmptyOp::create(rewriter, loc, sizes, convType.getElementType());
  SmallVector<Value> inputs{rawConv->getResult(0), bias};
  SmallVector<AffineMap> maps{identity, channel};
  if (relu) {
    inputs.push_back(zero);
    maps.push_back(scalar);
  }
  maps.push_back(identity);
  auto epilogue = linalg::GenericOp::create(
      rewriter, loc, TypeRange{convType}, inputs, ValueRange{empty}, maps,
      iterators, [&](OpBuilder &builder, Location nested, ValueRange args) {
        Value result = arith::AddFOp::create(builder, nested, args[0], args[1]);
        if (relu) {
          result = arith::MaximumFOp::create(builder, nested, result, args[2]);
        }
        linalg::YieldOp::create(builder, nested, result);
      });
  return epilogue.getResult(0);
}

/// conv -> [transpose] -> [expand_shape] -> truncf -> relu, or
/// conv -> [transpose] -> [expand_shape] -> relu, into
/// conv(fill 0) -> generic(addf bias, maximumf 0) -> [relayouts] -> [truncf].
///
/// The f16-import chain is the interesting one: ResNet50's export narrows
/// every convolution's f32 accumulator to f16 and clamps in f16. The clamp
/// moves onto the NPU's BN stage and the narrowing stays, now reading the
/// fused generic; the matched shim widens the NPU's f16 result back to f32
/// with a `linalg.generic`, and that widen and this narrow cancel once the
/// wrapper is inlined -- which is what leaves two Rocket dispatches touching
/// and lets the driver chain them (ISSUES.md P2).
struct FuseConvRelu : public OpRewritePattern<linalg::Conv2DNhwcHwcfOp> {
  FuseConvRelu(MLIRContext *context)
      : OpRewritePattern<linalg::Conv2DNhwcHwcfOp>(context, /*benefit=*/2) {}
  LogicalResult matchAndRewrite(linalg::Conv2DNhwcHwcfOp conv,
                                PatternRewriter &rewriter) const override {
    if (conv.getNumResults() != 1 || !isFusableDenseConv(conv)) {
      return failure();
    }
    Value convResult = conv->getResult(0);
    Operation *consumer = soleConsumer(convResult);
    if (!consumer) {
      return failure();
    }
    auto transpose = dyn_cast<linalg::TransposeOp>(consumer);
    if (transpose) {
      consumer = soleConsumer(transpose.getResult()[0]);
      if (!consumer) {
        return failure();
      }
    }
    auto expand = dyn_cast<tensor::ExpandShapeOp>(consumer);
    if (expand) {
      consumer = soleConsumer(expand.getResult());
      if (!consumer) {
        return failure();
      }
    }
    linalg::GenericOp narrow;
    auto generic = dyn_cast<linalg::GenericOp>(consumer);
    if (generic && isTruncfGeneric(generic)) {
      narrow = generic;
      consumer = soleConsumer(narrow.getResult(0));
      generic = consumer ? dyn_cast<linalg::GenericOp>(consumer) : nullptr;
    }
    if (!generic || !isElementwiseGeneric(generic)) {
      return failure();
    }
    std::optional<Value> zero = matchReluBody(generic);
    if (!zero || !isZeroFor(generic, *zero)) {
      return failure();
    }
    int64_t channelDim = 0;
    Value bias = matchPerChannelBias(conv.getDpsInits()[0], channelDim);
    if (!bias) {
      return failure();
    }
    Location loc = conv.getLoc();
    rewriter.setInsertionPoint(generic);
    Value result =
        buildFusedEpilogue(rewriter, loc, conv, bias, channelDim, /*relu=*/true);
    if (transpose) {
      IRMapping mapping;
      mapping.map(transpose.getInput(), result);
      result = rewriter.clone(*transpose.getOperation(), mapping)->getResult(0);
    }
    if (expand) {
      IRMapping mapping;
      mapping.map(expand.getSrc(), result);
      result = rewriter.clone(*expand.getOperation(), mapping)->getResult(0);
    }
    if (narrow) {
      IRMapping mapping;
      mapping.map(narrow.getDpsInputs()[0], result);
      result = rewriter.clone(*narrow.getOperation(), mapping)->getResult(0);
    }
    rewriter.replaceOp(generic, result);
    return success();
  }
};

/// The residual block's tail on the f16-import chain:
///
///   conv(bias init) -> [transpose] -> [expand_shape] -> truncf
///     -> generic(narrowed, skip) { addf; cmpf ugt %sum, %zero; select }
///
/// into conv(fill 0) -> generic(acc, bias, skip', 0.0) { addf bias; extf
/// skip; addf; maximumf } -> [relayouts] -> truncf, where `skip'` is the
/// skip tensor brought to the convolution's own rank with the inverse of
/// the reshape (which the canonicaliser folds against the producer's own
/// expand). The matched target runs the add and the ReLU in the DPU's EW
/// core after the convolution's tiles (`Conv2DDef.epilogue_add`), so the
/// whole block output stays on the NPU. ResNet50's sixteen residual adds
/// are this exact shape.
///
/// The skip may be either operand of the add. `Cout` must be a whole
/// number of 16-byte atoms, which is all the driver's EW cube handles.
struct FuseConvResidualRelu : public OpRewritePattern<linalg::Conv2DNhwcHwcfOp> {
  FuseConvResidualRelu(MLIRContext *context)
      : OpRewritePattern<linalg::Conv2DNhwcHwcfOp>(context, /*benefit=*/2) {}
  LogicalResult matchAndRewrite(linalg::Conv2DNhwcHwcfOp conv,
                                PatternRewriter &rewriter) const override {
    if (conv.getNumResults() != 1 || !isFusableDenseConv(conv)) {
      return failure();
    }
    auto convType = cast<RankedTensorType>(conv->getResult(0).getType());
    int64_t cout = convType.getDimSize(convType.getRank() - 1);
    if (cout == ShapedType::kDynamic || cout % 16 != 0) {
      return failure();
    }
    Value convResult = conv->getResult(0);
    Operation *consumer = soleConsumer(convResult);
    if (!consumer) {
      return failure();
    }
    auto transpose = dyn_cast<linalg::TransposeOp>(consumer);
    if (transpose) {
      consumer = soleConsumer(transpose.getResult()[0]);
      if (!consumer) {
        return failure();
      }
    }
    auto expand = dyn_cast<tensor::ExpandShapeOp>(consumer);
    if (expand) {
      consumer = soleConsumer(expand.getResult());
      if (!consumer) {
        return failure();
      }
    }
    auto narrow = dyn_cast_or_null<linalg::GenericOp>(consumer);
    if (!narrow || !isTruncfGeneric(narrow)) {
      return failure();
    }
    // Two spellings of "add the skip, then ReLU": the pipeline's, a named
    // `linalg.add` whose sole consumer is the ReLU generic (specialisation
    // runs before this pass, so an ONNX Add arrives named), and the fused
    // one a later elementwise fusion would produce, one generic with addf /
    // cmpf ugt / select. `replaced` is the op whose result the rewrite
    // takes over; `skip` is the add's other operand.
    Value narrowed = narrow.getResult(0);
    Operation *replaced = nullptr;
    Value skip;
    if (auto named = dyn_cast_or_null<linalg::AddOp>(soleConsumer(narrowed))) {
      if (named.getDpsInputs()[0] == narrowed) {
        skip = named.getDpsInputs()[1];
      } else if (named.getDpsInputs()[1] == narrowed) {
        skip = named.getDpsInputs()[0];
      } else {
        return failure();
      }
      auto relu = dyn_cast_or_null<linalg::GenericOp>(soleConsumer(named.getResult(0)));
      if (!relu || !isElementwiseGeneric(relu)) {
        return failure();
      }
      std::optional<Value> zero = matchReluBody(relu);
      if (!zero || !isZeroFor(relu, *zero)) {
        return failure();
      }
      replaced = relu;
    } else if (auto add = dyn_cast_or_null<linalg::GenericOp>(soleConsumer(narrowed))) {
      if (!isElementwiseGeneric(add) || add.getNumDpsInputs() != 2) {
        return failure();
      }
      Block &body = add.getRegion().front();
      auto it = body.begin();
      auto addf = (it != body.end()) ? dyn_cast<arith::AddFOp>(&*it) : nullptr;
      if (!addf) {
        return failure();
      }
      ++it;
      auto cmp = (it != body.end()) ? dyn_cast<arith::CmpFOp>(&*it) : nullptr;
      if (!cmp) {
        return failure();
      }
      ++it;
      auto select = (it != body.end()) ? dyn_cast<arith::SelectOp>(&*it) : nullptr;
      if (!select) {
        return failure();
      }
      ++it;
      if (it == body.end() || !isa<linalg::YieldOp>(*it) ||
          std::next(it) != body.end()) {
        return failure();
      }
      Value a = body.getArgument(0);
      Value b = body.getArgument(1);
      if (!((addf.getLhs() == a && addf.getRhs() == b) ||
            (addf.getLhs() == b && addf.getRhs() == a))) {
        return failure();
      }
      Value zero = cmp.getRhs();
      if (cmp.getPredicate() != arith::CmpFPredicate::UGT ||
          cmp.getLhs() != addf.getResult() ||
          select.getCondition() != cmp.getResult() ||
          select.getTrueValue() != addf.getResult() ||
          select.getFalseValue() != zero ||
          cast<linalg::YieldOp>(*it).getOperand(0) != select.getResult() ||
          !isZeroFor(add, zero)) {
        return failure();
      }
      if (add.getDpsInputs()[0] == narrowed) {
        skip = add.getDpsInputs()[1];
      } else if (add.getDpsInputs()[1] == narrowed) {
        skip = add.getDpsInputs()[0];
      } else {
        return failure();
      }
      replaced = add;
    } else {
      return failure();
    }
    auto skipType = dyn_cast<RankedTensorType>(skip.getType());
    if (!skipType || !skipType.getElementType().isF16()) {
      return failure();
    }
    int64_t channelDim = 0;
    Value bias = matchPerChannelBias(conv.getDpsInits()[0], channelDim);
    if (!bias) {
      return failure();
    }
    // The skip in the convolution's own layout and rank.
    Location loc = conv.getLoc();
    rewriter.setInsertionPoint(replaced);
    Value skipNhwc = skip;
    if (expand) {
      skipNhwc = tensor::CollapseShapeOp::create(
          rewriter, loc, skip, expand.getReassociationIndices());
    }
    if (transpose) {
      // The conv's result was transposed before the reshape; undo that on
      // the skip with the inverse permutation.
      ArrayRef<int64_t> permutation = transpose.getPermutation();
      SmallVector<int64_t> inverse(permutation.size());
      for (auto [index, target] : llvm::enumerate(permutation)) {
        inverse[target] = index;
      }
      auto inType = cast<RankedTensorType>(skipNhwc.getType());
      SmallVector<int64_t> outShape(inType.getRank());
      for (auto [index, target] : llvm::enumerate(inverse)) {
        outShape[index] = inType.getDimSize(target);
      }
      Value empty = tensor::EmptyOp::create(rewriter, loc, outShape,
                                            inType.getElementType());
      skipNhwc = linalg::TransposeOp::create(rewriter, loc, skipNhwc, empty, inverse)
                     .getResult()[0];
    }
    if (cast<RankedTensorType>(skipNhwc.getType()).getShape() != convType.getShape()) {
      return failure();
    }
    // conv over a zero init, then the fused epilogue.
    int64_t rank = convType.getRank();
    MLIRContext *context = rewriter.getContext();
    AffineMap identity = rewriter.getMultiDimIdentityMap(rank);
    AffineMap channel =
        AffineMap::get(rank, 0, {rewriter.getAffineDimExpr(channelDim)}, context);
    AffineMap scalar = AffineMap::get(rank, 0, context);
    SmallVector<utils::IteratorType> iterators(rank, utils::IteratorType::parallel);
    SmallVector<OpFoldResult> sizes = tensor::getMixedSizes(rewriter, loc, convResult);
    Value zeroF32 = arith::ConstantOp::create(rewriter, loc, rewriter.getF32FloatAttr(0.0f));
    Value accEmpty =
        tensor::EmptyOp::create(rewriter, loc, sizes, convType.getElementType());
    Value accInit = linalg::FillOp::create(rewriter, loc, ValueRange{zeroF32},
                                           ValueRange{accEmpty})
                        .getResult(0);
    auto rawConv = cast<linalg::Conv2DNhwcHwcfOp>(rewriter.clone(*conv.getOperation()));
    rawConv.getDpsInitsMutable().assign(accInit);
    Value empty = tensor::EmptyOp::create(rewriter, loc, sizes, convType.getElementType());
    SmallVector<AffineMap> maps{identity, channel, identity, scalar, identity};
    auto epilogue = linalg::GenericOp::create(
        rewriter, loc, TypeRange{convType},
        ValueRange{rawConv->getResult(0), bias, skipNhwc, zeroF32}, ValueRange{empty},
        maps, iterators, [&](OpBuilder &builder, Location nested, ValueRange args) {
          Value biased = arith::AddFOp::create(builder, nested, args[0], args[1]);
          Value skipWide = arith::ExtFOp::create(builder, nested, builder.getF32Type(), args[2]);
          Value summed = arith::AddFOp::create(builder, nested, biased, skipWide);
          Value result = arith::MaximumFOp::create(builder, nested, summed, args[3]);
          linalg::YieldOp::create(builder, nested, result);
        });
    Value result = epilogue.getResult(0);
    if (transpose) {
      IRMapping mapping;
      mapping.map(transpose.getInput(), result);
      result = rewriter.clone(*transpose.getOperation(), mapping)->getResult(0);
    }
    if (expand) {
      IRMapping mapping;
      mapping.map(expand.getSrc(), result);
      result = rewriter.clone(*expand.getOperation(), mapping)->getResult(0);
    }
    {
      IRMapping mapping;
      mapping.map(narrow.getDpsInputs()[0], result);
      result = rewriter.clone(*narrow.getOperation(), mapping)->getResult(0);
    }
    rewriter.replaceOp(replaced, result);
    return success();
  }
};

/// Every other dense fp16 convolution seeded with a per-channel bias:
/// conv(bias init) -> conv(fill 0) -> generic(addf bias). The consumers are
/// untouched. This is the bias moving onto the BS plane for the convolutions
/// that have no activation to fuse, so their shim no longer adds it on the
/// CPU against a full-size hoisted bias tensor. Lowest benefit, so the
/// activation patterns get first refusal on the same convolution.
struct FuseConvBias : public OpRewritePattern<linalg::Conv2DNhwcHwcfOp> {
  FuseConvBias(MLIRContext *context)
      : OpRewritePattern<linalg::Conv2DNhwcHwcfOp>(context, /*benefit=*/1) {}
  LogicalResult matchAndRewrite(linalg::Conv2DNhwcHwcfOp conv,
                                PatternRewriter &rewriter) const override {
    if (conv.getNumResults() != 1 || !isFusableDenseConv(conv)) {
      return failure();
    }
    int64_t channelDim = 0;
    Value bias = matchPerChannelBias(conv.getDpsInits()[0], channelDim);
    if (!bias) {
      return failure();
    }
    rewriter.setInsertionPoint(conv);
    Value result = buildFusedEpilogue(rewriter, conv.getLoc(), conv, bias,
                                      channelDim, /*relu=*/false);
    rewriter.replaceOp(conv, result);
    return success();
  }
};

struct RocketFuseConvRelu6Pass
    : public PassWrapper<RocketFuseConvRelu6Pass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RocketFuseConvRelu6Pass)

  StringRef getArgument() const final { return "rocket-fuse-conv-relu6"; }
  StringRef getDescription() const final {
    return "Rewrites an fp16 convolution and its epilogue -- a ReLU6, a ReLU "
           "behind an f16 narrowing, or a bare per-channel bias -- into the "
           "two-op canonical forms the fused-epilogue matchers claim, with "
           "the bias lifted onto the BS plane and the clamp bounds as scalar "
           "operands.";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() final {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<FuseConvRelu6<linalg::Conv2DNhwcHwcfOp>,
                 FuseConvRelu6<linalg::DepthwiseConv2DNhwcHwcOp>,
                 FuseConvRelu6<linalg::DepthwiseConv2DNchwChwOp>>(
        context, /*benefit=*/3);
    patterns.add<FuseConvRelu, FuseConvResidualRelu, FuseConvBias>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

static PassRegistration<RocketFuseConvRelu6Pass> reg;

} // namespace
} // namespace mlir::iree_compiler::IREE::HAL
