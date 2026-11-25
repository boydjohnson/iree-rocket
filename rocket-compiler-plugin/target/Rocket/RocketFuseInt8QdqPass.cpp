// Copyright 2026
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Two passes that put a quantized op the ONNX importer leaves in float, between
// a DequantizeLinear and a QuantizeLinear, into the int8 form a Rocket matcher
// claims: `rocket-fuse-int8-residual-add` (the residual tail, below) and
// `rocket-fuse-int8-max-pool` (see `FuseInt8MaxPool`).
//
// The residual add: puts a quantized residual block's tail --
// `Q(relu(DQ(a) + DQ(b)))` with both operands int8 -- into the one canonical
// form the int8 element-wise add matcher claims.
//
// An ONNX QOperator model whose residual Add reads both operands through
// DequantizeLinear (tools/quantize_onnx_int8.py --quantize-residual-skip)
// arrives here as five ops on NCHW `ui8` tensors:
//
//   %fa  = generic(%a)        // extui, subi za, sitofp, mulf sa
//   %fb  = generic(%b)        // the same, or uitofp, mulf sb when zb is 0
//   %sum = linalg.add(%fa, %fb)
//   %r   = generic(%sum)      // cmpf ugt 0, select: ReLU
//   %q   = generic(%r)        // divf so, roundeven, addf zo, clamp [0, 255],
//                             // fptoui
//
// The NPU's int8 EW add computes exactly that, apart from rounding ties
// (iree-rocket-hal tests/ew_int8_add_hw.rs), on signed int8 NHWC cubes. So
// this pass emits, for each match:
//
//   %as  = transpose(addi(%a, -128)) to NHWC      // and %bs
//   %out = generic(%as, %bs, sa, za-128, sb, zb-128, so, zo-128, 0, -128, 127)
//   %q'  = addi(transpose(%out) to NCHW, -128)    // replaces %q
//
// The canonical generic is the same arithmetic in the signed encoding --
// `u8 - zp == s8 - (zp - 128)`, and `clamp(round(y) + zo, 0, 255) - 128 ==
// clamp(round(y) + zo - 128, -128, 127)` -- so a build whose matcher declines
// it (or `--no-offload`) is still exact. Its scalars are operands rather
// than body constants for the reason `rocket-fuse-int8-requant-epilogue`
// gives: `cast_compatible_dag_from_root` cannot match a captured value, and
// the canonicaliser hoists body constants out anyway.
//
// The shifts and transposes are not new work on a chained path. The conv
// producing `%a` ends in `transpose(addi(nhwc_s8, -128))` and every conv
// reading `%q` starts with `transpose(addi(nchw_u8, -128))`; with this
// rewrite each pair meets its inverse, and dispatch formation folds them.
//
// The trailing ReLU is required: a residual Add with no activation (a
// MobileNetV2 linear bottleneck's) has no target yet.

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

namespace mlir::iree_compiler::IREE::HAL {
namespace {

constexpr int64_t kEncodingShift = -128;

static bool isIdentityElementwise(linalg::GenericOp op) {
  return op.getNumResults() == 1 && op.isAllParallelLoops() &&
         llvm::all_of(op.getIndexingMapsArray(),
                      [](AffineMap map) { return map.isIdentity(); });
}

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

static std::optional<double> constantFloat(linalg::GenericOp op, Value value) {
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    if (arg.getArgNumber() >= op.getNumDpsInputs()) {
      return std::nullopt;
    }
    value = op.getDpsInputs()[arg.getArgNumber()];
  }
  APFloat result(0.0f);
  if (!matchPattern(value, m_ConstantFloat(&result))) {
    return std::nullopt;
  }
  return result.convertToDouble();
}

static std::optional<int64_t> constantInt(linalg::GenericOp op, Value value) {
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    if (arg.getArgNumber() >= op.getNumDpsInputs()) {
      return std::nullopt;
    }
    value = op.getDpsInputs()[arg.getArgNumber()];
  }
  APInt result;
  if (!matchPattern(value, m_ConstantInt(&result))) {
    return std::nullopt;
  }
  return result.getSExtValue();
}

/// The block argument `value` is, as the generic's input index, when it is
/// one of the generic's own (non-init) inputs.
static std::optional<unsigned> inputIndex(linalg::GenericOp op, Value value) {
  auto arg = dyn_cast<BlockArgument>(value);
  if (!arg || arg.getOwner() != &op.getRegion().front() ||
      arg.getArgNumber() >= op.getNumDpsInputs()) {
    return std::nullopt;
  }
  return arg.getArgNumber();
}

/// One `ui8` dequantization: the tensor it reads, its scale, its zero point.
struct Dequant {
  Value source;
  double scale = 0.0;
  int64_t zeroPoint = 0;
};

static std::optional<Dequant> matchDequant(Value value) {
  auto op = value.getDefiningOp<linalg::GenericOp>();
  if (!op || !value.hasOneUse() || !isIdentityElementwise(op)) {
    return std::nullopt;
  }
  auto sourceType = dyn_cast<RankedTensorType>(op.getDpsInputs()[0].getType());
  auto resultType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
  if (!sourceType || !resultType || !sourceType.getElementType().isInteger(8) ||
      !resultType.getElementType().isF32()) {
    return std::nullopt;
  }
  Dequant found;
  found.source = op.getDpsInputs()[0];
  // Zero point 0: the importer spells it uitofp, mulf.
  if (auto ops = matchBody(op, {"arith.uitofp", "arith.mulf"}); !ops.empty()) {
    if (inputIndex(op, ops[0]->getOperand(0)) != 0u ||
        ops[1]->getOperand(0) != ops[0]->getResult(0)) {
      return std::nullopt;
    }
    std::optional<double> scale = constantFloat(op, ops[1]->getOperand(1));
    if (!scale) {
      return std::nullopt;
    }
    found.scale = *scale;
    found.zeroPoint = 0;
    return found;
  }
  auto ops = matchBody(op, {"arith.extui", "arith.subi", "arith.sitofp",
                            "arith.mulf"});
  if (ops.empty() || inputIndex(op, ops[0]->getOperand(0)) != 0u ||
      ops[1]->getOperand(0) != ops[0]->getResult(0) ||
      ops[2]->getOperand(0) != ops[1]->getResult(0) ||
      ops[3]->getOperand(0) != ops[2]->getResult(0)) {
    return std::nullopt;
  }
  std::optional<int64_t> zeroPoint = constantInt(op, ops[1]->getOperand(1));
  std::optional<double> scale = constantFloat(op, ops[3]->getOperand(1));
  if (!zeroPoint || !scale || *zeroPoint < 0 || *zeroPoint > 255) {
    return std::nullopt;
  }
  found.scale = *scale;
  found.zeroPoint = *zeroPoint;
  return found;
}

/// `cmpf ugt %x, 0; select %c, %x, 0` -- the importer's ReLU.
/// An elementwise add whose maps are all identities: what the named
/// `linalg.add` was until LLVM removed the named elementwise ops (IREE 3.14),
/// after which `linalg-specialize-generic-ops` produces
/// `linalg.elementwise kind=#linalg.elementwise_kind<add>` instead. The
/// identity check keeps the old op's semantics: the category op may also
/// carry broadcasting maps, which `linalg.add` could not.
static linalg::ElementwiseOp asIdentityElementwiseAdd(Operation *op) {
  auto add = dyn_cast_or_null<linalg::ElementwiseOp>(op);
  if (!add || add.getKind() != linalg::ElementwiseKind::add ||
      add.getNumDpsInputs() != 2 ||
      !llvm::all_of(add.getIndexingMapsArray(),
                    [](AffineMap map) { return map.isIdentity(); })) {
    return {};
  }
  return add;
}

static bool isRelu(linalg::GenericOp op) {
  if (!isIdentityElementwise(op) || op.getNumDpsInputs() < 1) {
    return false;
  }
  auto ops = matchBody(op, {"arith.cmpf", "arith.select"});
  if (ops.empty()) {
    return false;
  }
  auto cmp = cast<arith::CmpFOp>(ops[0]);
  auto select = cast<arith::SelectOp>(ops[1]);
  std::optional<double> floor = constantFloat(op, cmp.getRhs());
  std::optional<double> otherwise = constantFloat(op, select.getFalseValue());
  return cmp.getPredicate() == arith::CmpFPredicate::UGT &&
         inputIndex(op, cmp.getLhs()) == 0u && floor && *floor == 0.0 &&
         select.getCondition() == cmp.getResult() &&
         select.getTrueValue() == cmp.getLhs() && otherwise &&
         *otherwise == 0.0;
}

/// `divf so; roundeven; addf zo; maximumf 0; minimumf 255; fptoui` -- the
/// importer's `ui8` QuantizeLinear.
static bool matchQuant(linalg::GenericOp op, double &scale, int64_t &zero) {
  if (!isIdentityElementwise(op)) {
    return false;
  }
  auto resultType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
  if (!resultType || !resultType.getElementType().isInteger(8)) {
    return false;
  }
  auto ops = matchBody(op, {"arith.divf", "math.roundeven", "arith.addf",
                            "arith.maximumf", "arith.minimumf",
                            "arith.fptoui"});
  if (ops.empty() || inputIndex(op, ops[0]->getOperand(0)) != 0u) {
    return false;
  }
  for (size_t i = 1; i < ops.size(); ++i) {
    if (ops[i]->getOperand(0) != ops[i - 1]->getResult(0)) {
      return false;
    }
  }
  std::optional<double> divisor = constantFloat(op, ops[0]->getOperand(1));
  std::optional<double> zeroPoint = constantFloat(op, ops[2]->getOperand(1));
  std::optional<double> low = constantFloat(op, ops[3]->getOperand(1));
  std::optional<double> high = constantFloat(op, ops[4]->getOperand(1));
  if (!divisor || !zeroPoint || !low || !high || *low != 0.0 ||
      *high != 255.0 || *zeroPoint != std::floor(*zeroPoint) ||
      *zeroPoint < 0.0 || *zeroPoint > 255.0) {
    return false;
  }
  scale = *divisor;
  zero = static_cast<int64_t>(*zeroPoint);
  return true;
}

static Value emitShift(PatternRewriter &rewriter, Location loc, Value source) {
  auto type = cast<RankedTensorType>(source.getType());
  Value empty = tensor::EmptyOp::create(rewriter, loc, type.getShape(),
                                        type.getElementType());
  Value shift = arith::ConstantOp::create(
      rewriter, loc, rewriter.getIntegerAttr(type.getElementType(),
                                             kEncodingShift));
  SmallVector<AffineMap> maps(2,
                              rewriter.getMultiDimIdentityMap(type.getRank()));
  SmallVector<utils::IteratorType> iterators(type.getRank(),
                                             utils::IteratorType::parallel);
  return linalg::GenericOp::create(
             rewriter, loc, TypeRange{type}, ValueRange{source},
             ValueRange{empty}, maps, iterators,
             [&](OpBuilder &b, Location nested, ValueRange args) {
               Value shifted =
                   arith::AddIOp::create(b, nested, args[0], shift);
               linalg::YieldOp::create(b, nested, shifted);
             })
      .getResult(0);
}

static Value emitTranspose(PatternRewriter &rewriter, Location loc,
                           Value source, ArrayRef<int64_t> permutation) {
  auto type = cast<RankedTensorType>(source.getType());
  SmallVector<int64_t> shape;
  for (int64_t dim : permutation) {
    shape.push_back(type.getDimSize(dim));
  }
  Value empty =
      tensor::EmptyOp::create(rewriter, loc, shape, type.getElementType());
  return linalg::TransposeOp::create(rewriter, loc, source, empty, permutation)
      .getResult()[0];
}

struct FuseInt8ResidualAdd : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp quant,
                                PatternRewriter &rewriter) const override {
    double outputScale = 0.0;
    int64_t outputZeroPoint = 0;
    if (!matchQuant(quant, outputScale, outputZeroPoint)) {
      return failure();
    }
    auto relu = quant.getDpsInputs()[0].getDefiningOp<linalg::GenericOp>();
    if (!relu || !relu->hasOneUse() || !isRelu(relu)) {
      return failure();
    }
    auto add = asIdentityElementwiseAdd(relu.getDpsInputs()[0].getDefiningOp());
    if (!add || !add->hasOneUse()) {
      return failure();
    }
    std::optional<Dequant> a = matchDequant(add.getDpsInputs()[0]);
    std::optional<Dequant> b = matchDequant(add.getDpsInputs()[1]);
    if (!a || !b) {
      return failure();
    }
    // NCHW in, as every residual from the importer is; the transposes below
    // assume it.
    auto type = cast<RankedTensorType>(quant.getResult(0).getType());
    if (type.getRank() != 4 || !type.hasStaticShape() ||
        type.getDimSize(0) != 1 || a->source.getType() != type ||
        b->source.getType() != type) {
      return failure();
    }
    for (double scale : {a->scale, b->scale, outputScale}) {
      if (!(scale > 0.0) || !std::isfinite(scale)) {
        return failure();
      }
    }

    Location loc = quant.getLoc();
    const int64_t toNhwc[] = {0, 2, 3, 1};
    const int64_t toNchw[] = {0, 3, 1, 2};
    Value aNhwc = emitTranspose(rewriter, loc,
                                emitShift(rewriter, loc, a->source), toNhwc);
    Value bNhwc = emitTranspose(rewriter, loc,
                                emitShift(rewriter, loc, b->source), toNhwc);
    auto nhwcType = cast<RankedTensorType>(aNhwc.getType());

    auto f32 = [&](double value) -> Value {
      return arith::ConstantOp::create(rewriter, loc,
                                       rewriter.getF32FloatAttr(value));
    };
    auto i32 = [&](int64_t value) -> Value {
      return arith::ConstantOp::create(rewriter, loc,
                                       rewriter.getI32IntegerAttr(value));
    };
    SmallVector<Value> inputs = {
        aNhwc,
        bNhwc,
        f32(a->scale),
        i32(a->zeroPoint + kEncodingShift),
        f32(b->scale),
        i32(b->zeroPoint + kEncodingShift),
        f32(outputScale),
        i32(outputZeroPoint + kEncodingShift),
        f32(0.0),
        f32(-128.0),
        f32(127.0),
    };
    AffineMap identity = rewriter.getMultiDimIdentityMap(4);
    AffineMap scalar = AffineMap::get(4, 0, rewriter.getContext());
    SmallVector<AffineMap> maps = {identity, identity};
    maps.append(9, scalar);
    maps.push_back(identity);
    SmallVector<utils::IteratorType> iterators(4,
                                               utils::IteratorType::parallel);
    Value init = tensor::EmptyOp::create(rewriter, loc, nhwcType.getShape(),
                                         nhwcType.getElementType());
    Value fused =
        linalg::GenericOp::create(
            rewriter, loc, TypeRange{nhwcType}, inputs, ValueRange{init}, maps,
            iterators,
            [&](OpBuilder &bld, Location nested, ValueRange args) {
              Type i32Type = bld.getI32Type();
              Type f32Type = bld.getF32Type();
              auto dequant = [&](Value x, Value zero, Value scale) {
                Value wide = arith::ExtSIOp::create(bld, nested, i32Type, x);
                Value centred = arith::SubIOp::create(bld, nested, wide, zero);
                Value real =
                    arith::SIToFPOp::create(bld, nested, f32Type, centred);
                return arith::MulFOp::create(bld, nested, real, scale)
                    .getResult();
              };
              Value fa = dequant(args[0], args[3], args[2]);
              Value fb = dequant(args[1], args[5], args[4]);
              Value sum = arith::AddFOp::create(bld, nested, fa, fb);
              Value positive = arith::CmpFOp::create(
                  bld, nested, arith::CmpFPredicate::UGT, sum, args[8]);
              Value relu =
                  arith::SelectOp::create(bld, nested, positive, sum, args[8]);
              Value scaled = arith::DivFOp::create(bld, nested, relu, args[6]);
              Value rounded = math::RoundEvenOp::create(bld, nested, scaled);
              Value zero =
                  arith::SIToFPOp::create(bld, nested, f32Type, args[7]);
              Value offset = arith::AddFOp::create(bld, nested, rounded, zero);
              Value low = arith::MaximumFOp::create(bld, nested, offset, args[9]);
              Value clamped =
                  arith::MinimumFOp::create(bld, nested, low, args[10]);
              Value narrowed = arith::FPToSIOp::create(
                  bld, nested, bld.getI8Type(), clamped);
              linalg::YieldOp::create(bld, nested, narrowed);
            })
            .getResult(0);
    Value back = emitShift(rewriter, loc,
                           emitTranspose(rewriter, loc, fused, toNchw));
    rewriter.replaceOp(quant, back);
    return success();
  }
};


/// The max pool: `Q(pooling_nchw_max(pad(DQ(x), -inf)))` where the
/// DequantizeLinear and QuantizeLinear share one scale and zero point
/// (tools/quantize_onnx_int8.py makes them), rewritten as an int8 max pool on
/// the codes themselves:
///
///   %p = pooling_nhwc_max(pad(transpose(addi(%x, -128)), -128), fill -128)
///   %q' = addi(transpose(%p) to NCHW, -128)
///
/// Exact, with no rounding anywhere: max commutes with a monotone
/// dequantization, and with equal parameters `Q(DQ(c)) == c` for every code
/// (the f32 round trip lands within an ulp of `c` and `roundeven` takes it
/// back). The `-inf` padding never wins because every window holds at least
/// one real element (the pad is narrower than the kernel), so padding with
/// the smallest code, -128 in the signed encoding, is the same result. And
/// signed max over `u8 - 128` orders exactly like unsigned max over `u8`.
struct FuseInt8MaxPool : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp quant,
                                PatternRewriter &rewriter) const override {
    double outputScale = 0.0;
    int64_t outputZeroPoint = 0;
    if (!matchQuant(quant, outputScale, outputZeroPoint)) {
      return failure();
    }
    auto pool =
        quant.getDpsInputs()[0].getDefiningOp<linalg::PoolingNchwMaxOp>();
    if (!pool || !pool->hasOneUse()) {
      return failure();
    }
    // The init has to be -inf: anything else is a different reduction.
    auto initFill = pool.getDpsInits()[0].getDefiningOp<linalg::FillOp>();
    APFloat initValue(0.0f);
    if (!initFill ||
        !matchPattern(initFill.getInputs()[0], m_ConstantFloat(&initValue)) ||
        !initValue.isInfinity() || !initValue.isNegative()) {
      return failure();
    }
    auto pad = pool.getDpsInputs()[0].getDefiningOp<tensor::PadOp>();
    if (!pad || !pad->hasOneUse() || !pad.getLow().empty() ||
        !pad.getHigh().empty()) {
      return failure();
    }
    Value fill = pad.getConstantPaddingValue();
    APFloat padValue(0.0f);
    if (!fill || !matchPattern(fill, m_ConstantFloat(&padValue)) ||
        !padValue.isInfinity() || !padValue.isNegative()) {
      return failure();
    }
    std::optional<Dequant> dequant = matchDequant(pad.getSource());
    if (!dequant || dequant->scale != outputScale ||
        dequant->zeroPoint != outputZeroPoint) {
      return failure();
    }
    auto sourceType = cast<RankedTensorType>(dequant->source.getType());
    auto resultType = cast<RankedTensorType>(quant.getResult(0).getType());
    if (sourceType.getRank() != 4 || !sourceType.hasStaticShape() ||
        sourceType.getDimSize(0) != 1 || !resultType.hasStaticShape()) {
      return failure();
    }
    ArrayRef<int64_t> low = pad.getStaticLow();
    ArrayRef<int64_t> high = pad.getStaticHigh();
    if (low[0] != 0 || low[1] != 0 || high[0] != 0 || high[1] != 0) {
      return failure();
    }
    auto windowType = cast<RankedTensorType>(pool.getDpsInputs()[1].getType());
    if (windowType.getRank() != 2 || !windowType.hasStaticShape()) {
      return failure();
    }
    // Every window must hold a real element, or a -inf window would come out
    // as the smallest code rather than -inf (and then saturate).
    for (int64_t axis = 0; axis < 2; ++axis) {
      int64_t kernel = windowType.getDimSize(axis);
      if (low[2 + axis] >= kernel || high[2 + axis] >= kernel) {
        return failure();
      }
    }

    Location loc = quant.getLoc();
    const int64_t toNhwc[] = {0, 2, 3, 1};
    const int64_t toNchw[] = {0, 3, 1, 2};
    Value nhwc = emitTranspose(rewriter, loc,
                               emitShift(rewriter, loc, dequant->source),
                               toNhwc);
    Type i8 = rewriter.getI8Type();
    Value smallest = arith::ConstantOp::create(
        rewriter, loc, rewriter.getIntegerAttr(i8, -128));
    SmallVector<OpFoldResult> padLow = {
        rewriter.getIndexAttr(0), rewriter.getIndexAttr(low[2]),
        rewriter.getIndexAttr(low[3]), rewriter.getIndexAttr(0)};
    SmallVector<OpFoldResult> padHigh = {
        rewriter.getIndexAttr(0), rewriter.getIndexAttr(high[2]),
        rewriter.getIndexAttr(high[3]), rewriter.getIndexAttr(0)};
    auto nhwcType = cast<RankedTensorType>(nhwc.getType());
    auto paddedType = RankedTensorType::get(
        {1, nhwcType.getDimSize(1) + low[2] + high[2],
         nhwcType.getDimSize(2) + low[3] + high[3], nhwcType.getDimSize(3)},
        i8);
    Value padded = tensor::PadOp::create(rewriter, loc, paddedType, nhwc,
                                         padLow, padHigh, smallest)
                       .getResult();
    SmallVector<int64_t> outShape = {1, resultType.getDimSize(2),
                                     resultType.getDimSize(3),
                                     resultType.getDimSize(1)};
    Value outEmpty = tensor::EmptyOp::create(rewriter, loc, outShape, i8);
    Value outInit =
        linalg::FillOp::create(rewriter, loc, smallest, outEmpty).getResult(0);
    Value window = tensor::EmptyOp::create(rewriter, loc,
                                           windowType.getShape(), i8);
    Value pooled =
        linalg::PoolingNhwcMaxOp::create(
            rewriter, loc, TypeRange{outInit.getType()},
            ValueRange{padded, window}, ValueRange{outInit},
            pool.getStridesAttr(), pool.getDilationsAttr())
            .getResult(0);
    Value back = emitShift(rewriter, loc,
                           emitTranspose(rewriter, loc, pooled, toNchw));
    rewriter.replaceOp(quant, back);
    return success();
  }
};

struct RocketFuseInt8ResidualAddPass
    : public PassWrapper<RocketFuseInt8ResidualAddPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RocketFuseInt8ResidualAddPass)

  StringRef getArgument() const final {
    return "rocket-fuse-int8-residual-add";
  }
  StringRef getDescription() const final {
    return "Rewrites a quantized residual tail Q(relu(DQ(a) + DQ(b))) on int8 "
           "operands into the canonical signed NHWC form the Rocket int8 "
           "element-wise add matcher claims.";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    math::MathDialect, tensor::TensorDialect>();
  }

  void runOnOperation() final {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<FuseInt8ResidualAdd>(context);
    // No folding: the driver's constant folder would hoist the canonical
    // generic's scalars and fold the transposes before the matcher sees
    // them. The rewrite itself is all this pass is for.
    GreedyRewriteConfig config;
    config.enableFolding(false);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      return signalPassFailure();
    }
  }
};

static PassRegistration<RocketFuseInt8ResidualAddPass> reg;

struct RocketFuseInt8MaxPoolPass
    : public PassWrapper<RocketFuseInt8MaxPoolPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RocketFuseInt8MaxPoolPass)

  StringRef getArgument() const final { return "rocket-fuse-int8-max-pool"; }
  StringRef getDescription() const final {
    return "Rewrites Q(max_pool(pad(DQ(x), -inf))) with shared quantization "
           "into an int8 NHWC max pool on the codes, which the Rocket int8 "
           "max-pool matcher claims.";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() final {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<FuseInt8MaxPool>(context);
    GreedyRewriteConfig config;
    config.enableFolding(false);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      return signalPassFailure();
    }
  }
};

static PassRegistration<RocketFuseInt8MaxPoolPass> maxPoolReg;

} // namespace
} // namespace mlir::iree_compiler::IREE::HAL
