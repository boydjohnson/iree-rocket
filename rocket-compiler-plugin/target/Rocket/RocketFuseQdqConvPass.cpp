// Copyright 2026
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Rewrites an ONNX QDQ-format convolution, `Q(conv(DQ(x), DQ(w)) + DQ(b))`,
// into the canonical requantized int8 form the `rocket_dynamic_int8_requant*`
// (dense) and `rocket_dynamic_depthwise_int8_requant*` matchers claim: an
// i8 x i8 -> i32 `linalg.conv_2d_nhwc_hwcf` or
// `linalg.depthwise_conv_2d_nhwc_hwc` over a zero init, then one
// elementwise generic that adds the per-channel bias, scales, rounds,
// offsets, clamps and narrows. That is the form
// `rocket-fuse-int8-requant-epilogue` produces from a QOperator model; this
// pass is its counterpart for QDQ, the format ONNX Runtime's
// `quantize_static` writes by default.
//
// Without it a QDQ model's convolutions run on the NPU in fp16 on weights
// dequantized at compile time, with a CPU quantize -> dequantize pair on
// every activation edge (agent-docs/quantization-byo-measured.md: ResNet50
// 245 ms against 65 ms for the same network quantized as QOperator).
//
// # What arrives
//
// After `iree-preprocessing-convert-conv-to-channels-last` and
// `linalg-specialize-generic-ops`, per convolution (IREE 3.14, measured on
// stock `quantize_static` ResNet50 and MobileNetV2):
//
//   %x  = dequantize_affine(%codes : NCHW i8, sx, zx)   // activation
//   %w  = dequantize_affine(%wq : FCHW i8 const, sw)    // no zero point
//   %b  = dequantize_affine(%bq : i32 const, sb)        // sb == sx * sw
//   %xt = transpose(%x) to NHWC                         // and, when padded,
//                                                       // expand/pad/collapse
//   %wt = transpose(%w) to HWCF
//   %acc = conv_2d_nhwc_hwcf(%xt, %wt) outs(transpose(broadcast(%b)))
//   %q  = quantize_affine(transpose(%acc) to NCHW, sy, zy)
//
// # What leaves
//
//   %xs   = (addi -128)? (%codes)                 // only for ui8 activations
//   %xt   = the same transpose/expand/pad/collapse chain, on i8, padding
//           with the input zero point
//   %wt   = transpose(%wq) to HWCF
//   %acc  = conv_2d_nhwc_hwcf(%xt, %wt) outs(fill 0) : i32
//   %sumw = reduce(%wt) over (kh, kw, cin)        // rank 1
//   %bias = %bq - zx * %sumw                      // rank 1, folds
//   %s8   = generic(%acc, %bias, scale, zp, -128, 127)   // canonical
//   %q'   = transpose((addi -128)? (%s8)) to NCHW  // replaces %q
//
// with `scale = sx * sw / sy` rounded once to f32, and `zp` the output zero
// point in the signed encoding.
//
// A depthwise convolution is the same rewrite from a different start: the
// channels-last conversion leaves it as the importer's NCHW
// `depthwise_conv_2d_nchw_chw`, so `FuseQdqDepthwiseConv` emits the NHWC
// layout itself (see there).
//
// # Why it is exact (up to ties)
//
// `DQ(x) = (x - zx) * sx` and `DQ(w) = w * sw`, so the float convolution is
// `sx * sw * (sum_k(x * w) - zx * sum_k(w)) + bq * sb`. With `sb == sx * sw`
// -- which `quantize_static` guarantees, since it computes the bias scale as
// exactly that product, and which this pass checks -- that is
// `sx * sw * (acc + bq - zx * sum_k(w))`, an integer times one scale. The
// zero-point term is a per-channel constant because the filter is constant,
// so it folds into the bias, as `rocket-fuse-int8-requant-epilogue` folds the
// same term from a QLinearConv. A padded input pads its codes with `zx`,
// which dequantizes to the 0.0 the float pad wrote, and contributes exactly
// the `zx * w` per tap that the folded term subtracts.
//
// The float graph computes the same sum in f32, so the integer form is the
// more accurate of the two; the two can differ by one code where a product
// lands near a rounding tie, which is the difference the requantized path's
// e2e gate already runs at `atol=1` for.
//
// Measured exact: on a QDQ depthwise layer at stride 1 and 2 with random
// weights, the rewritten form run on the CPU matches a float64 evaluation of
// the original Q/DQ arithmetic on every code (agent-docs/
// quantization-byo-measured.md).
//
// # Signedness
//
// ORT's default is S8S8 (signed activations): the codes are used as they
// are. U8S8 (`ui8` activations, `input_unsigned` / `storage_unsigned`) is
// shifted into the signed encoding the DPU computes in, `u - 128`, and back
// after the requantization, the same convention the QOperator passes use, so
// a chain of rewritten convolutions meets each shift with its inverse.
//
// # Scope
//
// Per-tensor weight scales only. A per-channel weight scale makes the
// requantization ratio per channel, which the BS plane can carry
// (iree-rocket-hal `pack_int8_bias_to_bs`) but the executable ABI cannot yet:
// its `output_scale` push constant is one value. Those convolutions are
// declined and keep the fp16 path. So are a nonzero weight zero point (its
// correction term depends on the activation, not just the filter), a bias
// whose scale is not `sx * sw`, and any clamp range narrower than the full
// 8-bit one.

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <cmath>

namespace mlir::iree_compiler::IREE::HAL {
namespace {

using IREE::LinalgExt::DequantizeAffineOp;
using IREE::LinalgExt::QuantizeAffineOp;

constexpr int64_t kEncodingShift = -128;
constexpr double kSignedLow = -128.0;
constexpr double kSignedHigh = 127.0;
/// How far the bias scale may sit from `sx * sw`, relative. `quantize_static`
/// computes it as that f32 product, so a real model is within an ulp; this
/// only absorbs the f32-vs-f64 rounding of the product itself.
constexpr double kBiasScaleTolerance = 1.0 / (1 << 20);
/// The longest chain of layout ops between a dequantization and the
/// convolution operand it feeds. The padded activation is the longest, at
/// four (transpose, expand_shape, pad, collapse_shape).
constexpr int kMaxChain = 6;

static std::optional<double> constantFloat(Value value) {
  APFloat result(0.0f);
  if (!value || !matchPattern(value, m_ConstantFloat(&result))) {
    return std::nullopt;
  }
  return result.convertToDouble();
}

static bool isPositiveFinite(double value) {
  return value > 0.0 && std::isfinite(value);
}

/// An affine quantization op's zero point, read as the storage encoding
/// says: absent is 0, and `zp_unsigned` zero-extends a narrow integer.
template <typename AffineOp>
static std::optional<int64_t> zeroPointOf(AffineOp op) {
  Value zeroPoint = op.getZeroPoint();
  if (!zeroPoint) {
    return 0;
  }
  if (op.getZeroPointMap().getNumResults() != 0) {
    return std::nullopt;
  }
  APInt result;
  if (!matchPattern(zeroPoint, m_ConstantInt(&result))) {
    return std::nullopt;
  }
  return op.getZpUnsigned() ? static_cast<int64_t>(result.getZExtValue())
                            : result.getSExtValue();
}

/// Per tensor, identity-mapped, a constant scalar f32 scale: the only
/// granularity the executable ABI carries.
template <typename AffineOp>
static bool isPerTensor(AffineOp op) {
  return op->getNumResults() == 1 && op.getInputMap().isIdentity() &&
         op.getOutputMap().isIdentity() &&
         op.getScaleMap().getNumResults() == 0 &&
         op.getScaleElementType().isF32();
}

/// One 8-bit dequantization or quantization, in the signed encoding.
struct Affine8 {
  /// The integer tensor: the codes a dequantization reads.
  Value codes;
  double scale = 0.0;
  /// The zero point in the signed encoding (`zp - 128` for `ui8`).
  int64_t zeroPoint = 0;
  /// Whether the codes are `ui8` and need the encoding shift.
  bool isUnsigned = false;
};

/// An activation dequantization: `i8` or `ui8` codes, per tensor, into f32.
static std::optional<Affine8> matchActivationDequant(DequantizeAffineOp op) {
  if (!op || !isPerTensor(op) ||
      !op.getInputType().getElementType().isInteger(8) ||
      !op.getOutputType().getElementType().isF32()) {
    return std::nullopt;
  }
  std::optional<double> scale = constantFloat(op.getScale());
  std::optional<int64_t> zeroPoint = zeroPointOf(op);
  if (!scale || !isPositiveFinite(*scale) || !zeroPoint) {
    return std::nullopt;
  }
  Affine8 found;
  found.codes = op.getInput();
  found.scale = *scale;
  found.isUnsigned = op.getInputUnsigned();
  int64_t low = found.isUnsigned ? 0 : -128;
  if (*zeroPoint < low || *zeroPoint > low + 255) {
    return std::nullopt;
  }
  found.zeroPoint = found.isUnsigned ? *zeroPoint + kEncodingShift : *zeroPoint;
  return found;
}

/// A weight dequantization: constant signed `i8`, per tensor, symmetric.
static std::optional<Affine8> matchWeightDequant(DequantizeAffineOp op) {
  if (!op || !isPerTensor(op) || op.getInputUnsigned() ||
      !op.getInputType().getElementType().isInteger(8) ||
      !op.getOutputType().getElementType().isF32() ||
      !matchPattern(op.getInput(), m_Constant())) {
    return std::nullopt;
  }
  std::optional<double> scale = constantFloat(op.getScale());
  std::optional<int64_t> zeroPoint = zeroPointOf(op);
  if (!scale || !isPositiveFinite(*scale) || zeroPoint != 0) {
    return std::nullopt;
  }
  Affine8 found;
  found.codes = op.getInput();
  found.scale = *scale;
  return found;
}

/// The output quantization: f32 to `i8` or `ui8`, per tensor, over the
/// encoding's full range.
static std::optional<Affine8> matchOutputQuant(QuantizeAffineOp op) {
  if (!isPerTensor(op) || !op.getInputType().getElementType().isF32() ||
      !op.getOutputType().getElementType().isInteger(8) ||
      !isa<RankedTensorType>(op->getResult(0).getType())) {
    return std::nullopt;
  }
  bool isUnsigned = op.getStorageUnsigned();
  int64_t low = isUnsigned ? 0 : -128;
  if (op.getQuantMin() != low || op.getQuantMax() != low + 255) {
    return std::nullopt;
  }
  std::optional<double> scale = constantFloat(op.getScale());
  std::optional<int64_t> zeroPoint = zeroPointOf(op);
  if (!scale || !isPositiveFinite(*scale) || !zeroPoint ||
      *zeroPoint < low || *zeroPoint > low + 255) {
    return std::nullopt;
  }
  Affine8 found;
  found.codes = op.getInput();
  found.scale = *scale;
  found.isUnsigned = isUnsigned;
  found.zeroPoint = isUnsigned ? *zeroPoint + kEncodingShift : *zeroPoint;
  return found;
}

/// The layout ops between a dequantization and the convolution operand it
/// feeds, in program order, plus the dequantization itself.
struct LayoutChain {
  DequantizeAffineOp dequant;
  SmallVector<Operation *> ops;
};

/// Walks from a convolution operand back to the dequantization it reads,
/// through the transposes, reshapes and zero-filled pads the channels-last
/// conversion leaves in between. Anything else declines: a chain this pass
/// rebuilds on integers has to mean the same thing on integers, and these
/// four do, given the fill rewrite below.
static FailureOr<LayoutChain> walkToDequant(Value value) {
  LayoutChain chain;
  for (int depth = 0; depth <= kMaxChain; ++depth) {
    Operation *def = value.getDefiningOp();
    if (!def) {
      return failure();
    }
    if (auto dequant = dyn_cast<DequantizeAffineOp>(def)) {
      chain.dequant = dequant;
      std::reverse(chain.ops.begin(), chain.ops.end());
      return chain;
    }
    if (auto transpose = dyn_cast<linalg::TransposeOp>(def)) {
      chain.ops.push_back(def);
      value = transpose.getInput();
      continue;
    }
    if (auto expand = dyn_cast<tensor::ExpandShapeOp>(def)) {
      chain.ops.push_back(def);
      value = expand.getSrc();
      continue;
    }
    if (auto collapse = dyn_cast<tensor::CollapseShapeOp>(def)) {
      chain.ops.push_back(def);
      value = collapse.getSrc();
      continue;
    }
    if (auto pad = dyn_cast<tensor::PadOp>(def)) {
      // Zero is what the integer fill replaces: DQ(zx) == 0.0.
      std::optional<double> fill = constantFloat(pad.getConstantPaddingValue());
      if (!fill || *fill != 0.0 || pad.getNofold()) {
        return failure();
      }
      chain.ops.push_back(def);
      value = pad.getSource();
      continue;
    }
    return failure();
  }
  return failure();
}

static RankedTensorType withElementType(Type type, Type elementType) {
  auto ranked = cast<RankedTensorType>(type);
  return RankedTensorType::get(ranked.getShape(), elementType);
}

/// Whether `expand`, `pad`, `collapse` is a pad wrapped in a unit-dimension
/// round trip: the collapse undoes the expand's reassociation, and within
/// each group only one dimension is padded while the rest stay unit. Then it
/// is one pad of the unexpanded tensor, whose amounts are returned.
///
/// This is how `iree-preprocessing-convert-conv-to-channels-last` leaves a
/// padded NCHW input (a rank-5 pad between an `expand_shape` and a
/// `collapse_shape`). The int8 requantized pad1 templates are written
/// against a rank-4 NHWC `tensor.pad` read directly by the convolution --
/// the form a QOperator model reaches them in -- so the rebuilt chain uses
/// that one.
static bool foldUnitPadRoundTrip(tensor::ExpandShapeOp expand,
                                 tensor::PadOp pad,
                                 tensor::CollapseShapeOp collapse,
                                 SmallVector<int64_t> &low,
                                 SmallVector<int64_t> &high) {
  if (pad.getSource() != expand.getResult() ||
      collapse.getSrc() != pad.getResult() ||
      !pad.getLow().empty() || !pad.getHigh().empty() ||
      expand.getReassociationIndices() != collapse.getReassociationIndices()) {
    return false;
  }
  auto expanded = cast<RankedTensorType>(expand.getResult().getType());
  ArrayRef<int64_t> padLow = pad.getStaticLow();
  ArrayRef<int64_t> padHigh = pad.getStaticHigh();
  low.clear();
  high.clear();
  for (const ReassociationIndices &group : expand.getReassociationIndices()) {
    int64_t groupLow = 0;
    int64_t groupHigh = 0;
    int padded = 0;
    for (int64_t dim : group) {
      if (padLow[dim] != 0 || padHigh[dim] != 0) {
        ++padded;
        groupLow = padLow[dim];
        groupHigh = padHigh[dim];
      }
    }
    if (padded > 1) {
      return false;
    }
    // A group that pads one dimension must have every other one unit, or
    // collapsing the padded tensor interleaves fill with data.
    if (padded == 1) {
      for (int64_t dim : group) {
        if ((padLow[dim] == 0 && padHigh[dim] == 0) &&
            expanded.getDimSize(dim) != 1) {
          return false;
        }
      }
    }
    low.push_back(groupLow);
    high.push_back(groupHigh);
  }
  return true;
}

/// Re-emits `chain.ops` over `source` on its integer element type. A pad
/// fills with `fill`.
static FailureOr<Value> rebuildChain(PatternRewriter &rewriter, Location loc,
                                     ArrayRef<Operation *> ops, Value source,
                                     std::optional<int64_t> fill) {
  Type elementType = cast<RankedTensorType>(source.getType()).getElementType();
  Value current = source;
  for (size_t index = 0; index < ops.size(); ++index) {
    Operation *op = ops[index];
    if (index + 2 < ops.size() && fill) {
      auto expand = dyn_cast<tensor::ExpandShapeOp>(ops[index]);
      auto pad = dyn_cast<tensor::PadOp>(ops[index + 1]);
      auto collapse = dyn_cast<tensor::CollapseShapeOp>(ops[index + 2]);
      SmallVector<int64_t> low, high;
      if (expand && pad && collapse &&
          foldUnitPadRoundTrip(expand, pad, collapse, low, high)) {
        Type resultType =
            withElementType(collapse.getResult().getType(), elementType);
        Value fillValue = arith::ConstantOp::create(
            rewriter, loc, rewriter.getIntegerAttr(elementType, *fill));
        current = tensor::PadOp::create(
                      rewriter, loc, resultType, current,
                      getAsIndexOpFoldResult(rewriter.getContext(), low),
                      getAsIndexOpFoldResult(rewriter.getContext(), high),
                      fillValue)
                      .getResult();
        index += 2;
        continue;
      }
    }
    Type resultType = withElementType(op->getResult(0).getType(), elementType);
    if (auto transpose = dyn_cast<linalg::TransposeOp>(op)) {
      auto type = cast<RankedTensorType>(resultType);
      Value empty = tensor::EmptyOp::create(rewriter, loc, type.getShape(),
                                            elementType);
      current = linalg::TransposeOp::create(rewriter, loc, current, empty,
                                            transpose.getPermutation())
                    .getResult()[0];
    } else if (auto expand = dyn_cast<tensor::ExpandShapeOp>(op)) {
      current = tensor::ExpandShapeOp::create(
                    rewriter, loc, resultType, current,
                    expand.getReassociationIndices(),
                    expand.getMixedOutputShape())
                    .getResult();
    } else if (auto collapse = dyn_cast<tensor::CollapseShapeOp>(op)) {
      current = tensor::CollapseShapeOp::create(
                    rewriter, loc, resultType, current,
                    collapse.getReassociationIndices())
                    .getResult();
    } else if (auto pad = dyn_cast<tensor::PadOp>(op)) {
      if (!fill) {
        return failure();
      }
      Value fillValue = arith::ConstantOp::create(
          rewriter, loc, rewriter.getIntegerAttr(elementType, *fill));
      current = tensor::PadOp::create(rewriter, loc, resultType, current,
                                      pad.getMixedLowPad(),
                                      pad.getMixedHighPad(), fillValue)
                    .getResult();
    } else {
      return failure();
    }
  }
  return current;
}

/// `addi -128` over an i8 tensor: between the `ui8` encoding and the signed
/// one, in either direction.
static Value emitShift(PatternRewriter &rewriter, Location loc, Value source) {
  auto type = cast<RankedTensorType>(source.getType());
  Value empty = tensor::EmptyOp::create(rewriter, loc, type.getShape(),
                                        type.getElementType());
  SmallVector<AffineMap> maps(2,
                              rewriter.getMultiDimIdentityMap(type.getRank()));
  SmallVector<utils::IteratorType> iterators(type.getRank(),
                                             utils::IteratorType::parallel);
  return linalg::GenericOp::create(
             rewriter, loc, TypeRange{type}, ValueRange{source},
             ValueRange{empty}, maps, iterators,
             [&](OpBuilder &b, Location nested, ValueRange args) {
               Value shift = arith::ConstantOp::create(
                   b, nested,
                   b.getIntegerAttr(type.getElementType(), kEncodingShift));
               Value shifted = arith::AddIOp::create(b, nested, args[0], shift);
               linalg::YieldOp::create(b, nested, shifted);
             })
      .getResult(0);
}

/// The f32 bias dequantization a convolution's init broadcasts, or a zero
/// fill for a convolution with no bias (`bias` stays null).
///
/// The init arrives as `transpose(broadcast(DQ(b)))`: the importer
/// broadcasts the NCHW bias and the channels-last conversion transposes it
/// rather than rewriting the broadcast (the same shape
/// `rocket-fuse-int8-requant-epilogue`'s `matchBiasInit` documents, on f32).
/// The surviving broadcast dimension has to land on the accumulator's
/// channel: last for NHWC, `channel` when given (1 for the NCHW depthwise
/// convolution, which the channels-last conversion leaves alone).
static LogicalResult matchBiasInit(Value init, DequantizeAffineOp &bias,
                                   int64_t channel = -1) {
  bias = nullptr;
  if (auto fill = init.getDefiningOp<linalg::FillOp>()) {
    std::optional<double> value = constantFloat(fill.getInputs()[0]);
    return success(value && *value == 0.0);
  }
  auto broadcast = init.getDefiningOp<linalg::BroadcastOp>();
  int64_t channelPosition = -1;
  if (auto transpose = init.getDefiningOp<linalg::TransposeOp>()) {
    broadcast = transpose.getInput().getDefiningOp<linalg::BroadcastOp>();
    if (!broadcast) {
      return failure();
    }
    ArrayRef<int64_t> permutation = transpose.getPermutation();
    if (permutation.empty()) {
      return failure();
    }
    channelPosition = permutation.back();
  }
  if (!broadcast) {
    return failure();
  }
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
  if (channelPosition < 0) {
    channelPosition = channel >= 0 ? channel : broadcastType.getRank() - 1;
  }
  if (!retained || *retained != channelPosition) {
    return failure();
  }
  bias = broadcast.getInput().getDefiningOp<DequantizeAffineOp>();
  if (!bias || !isPerTensor(bias) || zeroPointOf(bias) != 0 ||
      bias.getInputType().getRank() != 1 ||
      !bias.getInputType().getElementType().isSignlessInteger(32) ||
      !bias.getOutputType().getElementType().isF32()) {
    return failure();
  }
  return success();
}

/// `sum_k(w)` per output channel, as i32, over a channels-last i8 filter:
/// HWCF for a dense convolution, HWC for a depthwise one.
static Value emitWeightSum(PatternRewriter &rewriter, Location loc,
                           Value filter) {
  auto type = cast<RankedTensorType>(filter.getType());
  int64_t rank = type.getRank();
  int64_t channels = type.getDimSize(rank - 1);
  Type i32 = rewriter.getI32Type();
  Value zero =
      arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(0));
  Value empty = tensor::EmptyOp::create(rewriter, loc, {channels}, i32);
  Value init = linalg::FillOp::create(rewriter, loc, ValueRange{zero},
                                      ValueRange{empty})
                   .getResult(0);
  MLIRContext *context = rewriter.getContext();
  AffineMap input = AffineMap::getMultiDimIdentityMap(rank, context);
  AffineMap output =
      AffineMap::get(rank, 0, {rewriter.getAffineDimExpr(rank - 1)}, context);
  SmallVector<utils::IteratorType> iterators(rank - 1,
                                             utils::IteratorType::reduction);
  iterators.push_back(utils::IteratorType::parallel);
  return linalg::GenericOp::create(
             rewriter, loc, TypeRange{init.getType()}, ValueRange{filter},
             ValueRange{init}, SmallVector<AffineMap>{input, output},
             iterators,
             [&](OpBuilder &b, Location nested, ValueRange args) {
               Value wide = arith::ExtSIOp::create(b, nested, i32, args[0]);
               Value sum = arith::AddIOp::create(b, nested, wide, args[1]);
               linalg::YieldOp::create(b, nested, sum);
             })
      .getResult(0);
}

/// `bias - zx * sumw`, or `-zx * sumw` with no bias, or the bias (or zeros)
/// alone when `zx` is 0. Rank 1 over constants, so constant evaluation folds
/// it to a literal.
static Value emitFoldedBias(PatternRewriter &rewriter, Location loc,
                            Value bias, Value filter, int64_t inputZeroPoint,
                            int64_t channels) {
  Type i32 = rewriter.getI32Type();
  if (inputZeroPoint == 0) {
    if (bias) {
      return bias;
    }
    Value zero =
        arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(0));
    Value empty = tensor::EmptyOp::create(rewriter, loc, {channels}, i32);
    return linalg::FillOp::create(rewriter, loc, ValueRange{zero},
                                  ValueRange{empty})
        .getResult(0);
  }
  Value weightSum = emitWeightSum(rewriter, loc, filter);
  Value empty = tensor::EmptyOp::create(rewriter, loc, {channels}, i32);
  SmallVector<Value> inputs;
  if (bias) {
    inputs.push_back(bias);
  }
  inputs.push_back(weightSum);
  AffineMap identity = rewriter.getMultiDimIdentityMap(1);
  SmallVector<AffineMap> maps(inputs.size() + 1, identity);
  return linalg::GenericOp::create(
             rewriter, loc, TypeRange{empty.getType()}, inputs,
             ValueRange{empty}, maps,
             SmallVector<utils::IteratorType>{utils::IteratorType::parallel},
             [&](OpBuilder &b, Location nested, ValueRange args) {
               Value sum = bias ? args[1] : args[0];
               Value zeroPoint = arith::ConstantOp::create(
                   b, nested, b.getI32IntegerAttr(inputZeroPoint));
               Value correction =
                   arith::MulIOp::create(b, nested, sum, zeroPoint);
               Value base = bias ? args[0]
                                 : arith::ConstantOp::create(
                                       b, nested, b.getI32IntegerAttr(0))
                                       .getResult();
               Value result = arith::SubIOp::create(b, nested, base, correction);
               linalg::YieldOp::create(b, nested, result);
             })
      .getResult(0);
}

/// The canonical requantization of an i32 channels-last accumulator, then
/// the shift back to `ui8` when the output is unsigned, then the layout
/// transpose the model's quantization read through (none when
/// `permutation` is empty).
static Value emitRequantTail(PatternRewriter &rewriter, Location loc,
                             Value integerAcc, Value bias, double ratio,
                             const Affine8 &output,
                             ArrayRef<int64_t> permutation) {
  ArrayRef<int64_t> accShape =
      cast<RankedTensorType>(integerAcc.getType()).getShape();
  Type i8 = rewriter.getI8Type();
  // The canonical requantization; operands, not body constants, for the
  // reason `rocket_int8_requant_match.mlir` pins.
  Value scaleValue = arith::ConstantOp::create(
      rewriter, loc, rewriter.getF32FloatAttr(static_cast<float>(ratio)));
  Value zeroPointValue = arith::ConstantOp::create(
      rewriter, loc, rewriter.getI32IntegerAttr(output.zeroPoint));
  Value lowValue = arith::ConstantOp::create(
      rewriter, loc, rewriter.getF32FloatAttr(kSignedLow));
  Value highValue = arith::ConstantOp::create(
      rewriter, loc, rewriter.getF32FloatAttr(kSignedHigh));
  auto int8Type = RankedTensorType::get(accShape, i8);
  Value outEmpty = tensor::EmptyOp::create(rewriter, loc,
                                           int8Type.getShape(), i8);
  unsigned rank = static_cast<unsigned>(accShape.size());
  AffineMap identity = rewriter.getMultiDimIdentityMap(rank);
  AffineMap channel = AffineMap::get(
      rank, 0, {rewriter.getAffineDimExpr(rank - 1)}, rewriter.getContext());
  AffineMap scalar = AffineMap::get(rank, 0, {}, rewriter.getContext());
  SmallVector<AffineMap> maps{identity, channel, scalar,
                              scalar,   scalar,  scalar, identity};
  SmallVector<utils::IteratorType> iterators(rank,
                                             utils::IteratorType::parallel);
  Value result =
      linalg::GenericOp::create(
          rewriter, loc, TypeRange{int8Type},
          ValueRange{integerAcc, bias, scaleValue, zeroPointValue, lowValue,
                     highValue},
          ValueRange{outEmpty}, maps, iterators,
          [&](OpBuilder &b, Location nested, ValueRange args) {
            Value biased = arith::AddIOp::create(b, nested, args[0], args[1]);
            Value real =
                arith::SIToFPOp::create(b, nested, b.getF32Type(), biased);
            Value scaled = arith::MulFOp::create(b, nested, real, args[2]);
            Value rounded = math::RoundEvenOp::create(b, nested, scaled);
            Value zeroPointReal =
                arith::SIToFPOp::create(b, nested, b.getF32Type(), args[3]);
            Value offset =
                arith::AddFOp::create(b, nested, rounded, zeroPointReal);
            Value low = arith::MaximumFOp::create(b, nested, offset, args[4]);
            Value clamped = arith::MinimumFOp::create(b, nested, low, args[5]);
            Value narrowed =
                arith::FPToSIOp::create(b, nested, b.getI8Type(), clamped);
            linalg::YieldOp::create(b, nested, narrowed);
          })
          .getResult(0);

  if (output.isUnsigned) {
    result = emitShift(rewriter, loc, result);
  }
  if (!permutation.empty()) {
    SmallVector<int64_t> shape;
    for (int64_t dim : permutation) {
      shape.push_back(accShape[dim]);
    }
    Value empty = tensor::EmptyOp::create(rewriter, loc, shape, i8);
    result = linalg::TransposeOp::create(rewriter, loc, result, empty,
                                         permutation)
                 .getResult()[0];
  }
  return result;
}

/// Anchored on the QuantizeLinear: the rewrite replaces the quantization's
/// result, and every operand of the new chain is defined before it.
struct FuseQdqConv : public OpRewritePattern<QuantizeAffineOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(QuantizeAffineOp quant,
                                PatternRewriter &rewriter) const override {
    std::optional<Affine8> output = matchOutputQuant(quant);
    if (!output) {
      return rewriter.notifyMatchFailure(quant, "not a per-tensor 8-bit Q");
    }

    // The quantization reads the convolution, or its NHWC -> NCHW
    // transpose. Each link must have this one reader: the rewrite replaces
    // the whole chain, and a second reader of the f32 accumulator would
    // need the float result it no longer computes.
    Value accumulator = output->codes;
    linalg::TransposeOp outputTranspose =
        accumulator.getDefiningOp<linalg::TransposeOp>();
    if (outputTranspose) {
      if (!accumulator.hasOneUse()) {
        return failure();
      }
      accumulator = outputTranspose.getInput();
    }
    auto conv = accumulator.getDefiningOp<linalg::Conv2DNhwcHwcfOp>();
    if (!conv || !accumulator.hasOneUse()) {
      return rewriter.notifyMatchFailure(quant, "not fed by a NHWC conv");
    }
    auto accType = cast<RankedTensorType>(conv.getResult(0).getType());
    if (!accType.getElementType().isF32() || !accType.hasStaticShape()) {
      return failure();
    }

    FailureOr<LayoutChain> inputChain = walkToDequant(conv.getDpsInputs()[0]);
    FailureOr<LayoutChain> filterChain = walkToDequant(conv.getDpsInputs()[1]);
    if (failed(inputChain) || failed(filterChain)) {
      return rewriter.notifyMatchFailure(quant, "operands are not DQ'd");
    }
    std::optional<Affine8> input = matchActivationDequant(inputChain->dequant);
    std::optional<Affine8> weights = matchWeightDequant(filterChain->dequant);
    if (!input || !weights) {
      return rewriter.notifyMatchFailure(
          quant, "activation or weight DQ is not per-tensor 8-bit");
    }
    // The filter chain is layout only; a pad there is not a convolution.
    for (Operation *op : filterChain->ops) {
      if (isa<tensor::PadOp>(op)) {
        return failure();
      }
    }

    DequantizeAffineOp biasDequant;
    if (failed(matchBiasInit(conv.getDpsInits()[0], biasDequant))) {
      return rewriter.notifyMatchFailure(quant, "init is not a DQ'd bias");
    }
    double accumulatorScale = input->scale * weights->scale;
    if (biasDequant &&
        biasDequant.getInputType().getDimSize(0) != accType.getShape().back()) {
      return failure();
    }
    if (biasDequant) {
      std::optional<double> biasScale = constantFloat(biasDequant.getScale());
      if (!biasScale ||
          std::abs(*biasScale - accumulatorScale) >
              kBiasScaleTolerance * accumulatorScale) {
        return rewriter.notifyMatchFailure(quant, "bias scale is not sx * sw");
      }
    }
    double ratio = accumulatorScale / output->scale;
    if (!std::isfinite(ratio) || !(ratio > 0.0) ||
        !std::isfinite(static_cast<float>(ratio))) {
      return failure();
    }

    Location loc = conv.getLoc();
    rewriter.setInsertionPoint(quant);

    // The activation, in the signed encoding, through the same layout chain.
    Value codes = input->codes;
    if (input->isUnsigned) {
      codes = emitShift(rewriter, loc, codes);
    }
    FailureOr<Value> convInput = rebuildChain(rewriter, loc, inputChain->ops,
                                              codes, input->zeroPoint);
    FailureOr<Value> convFilter = rebuildChain(
        rewriter, loc, filterChain->ops, weights->codes, std::nullopt);
    if (failed(convInput) || failed(convFilter)) {
      return failure();
    }
    if (cast<RankedTensorType>(convFilter->getType()).getRank() != 4) {
      return failure();
    }

    Type i32 = rewriter.getI32Type();
    auto accI32 = RankedTensorType::get(accType.getShape(), i32);
    Value zero =
        arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(0));
    Value accEmpty = tensor::EmptyOp::create(rewriter, loc, accI32.getShape(), i32);
    Value accInit = linalg::FillOp::create(rewriter, loc, ValueRange{zero},
                                           ValueRange{accEmpty})
                        .getResult(0);
    // `tensor<2xi64>`, not the `vector<2xi64>` the channels-last conversion
    // writes: the requantized DAG templates spell it so, and
    // `cast_compatible_dag_from_root` compares attribute types. See
    // `rocket-fuse-int8-requant-epilogue`.
    auto attrType = RankedTensorType::get({2}, rewriter.getIntegerType(64));
    SmallVector<int64_t> strides(conv.getStrides().getValues<int64_t>());
    SmallVector<int64_t> dilations(conv.getDilations().getValues<int64_t>());
    Value integerAcc =
        linalg::Conv2DNhwcHwcfOp::create(
            rewriter, loc, TypeRange{accI32}, ValueRange{*convInput, *convFilter},
            ValueRange{accInit}, DenseIntElementsAttr::get(attrType, strides),
            DenseIntElementsAttr::get(attrType, dilations))
            .getResult(0);

    int64_t channels = accType.getShape().back();
    Value bias = emitFoldedBias(rewriter, loc,
                                biasDequant ? biasDequant.getInput() : Value(),
                                *convFilter, input->zeroPoint, channels);

    Value result = emitRequantTail(
        rewriter, loc, integerAcc, bias, ratio, *output,
        outputTranspose ? outputTranspose.getPermutation() : ArrayRef<int64_t>{});
    if (result.getType() != quant->getResult(0).getType()) {
      return failure();
    }
    rewriter.replaceOp(quant, result);
    return success();
  }
};

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

/// The depthwise twin of `FuseQdqConv`. The channels-last conversion leaves
/// a depthwise convolution alone, so it arrives in the importer's own NCHW
/// form,
///
///   %acc = depthwise_conv_2d_nchw_chw(pad_nchw?(DQ(x)), collapse(DQ(w)))
///            outs(broadcast(DQ(b)))
///   %q   = quantize_affine(%acc)
///
/// and this emits the NHWC form the requantized depthwise targets claim:
/// `transpose(codes)`, then a rank-4 NHWC pad filled with the input zero
/// point (the order `rocket-transpose-quantized-conv-to-nhwc` gives a
/// QOperator depthwise, so `rocket-fold-conv-pad` can mark a -128 pad1 for
/// `@match_dynamic_depthwise_conv2d_3x3_pad1_int8_requant{,_s2}`), an HWC
/// filter, the same folded bias and requantization as the dense case, and a
/// transpose back to NCHW. The input transpose meets the producing
/// convolution's output transpose, and the pair cancels.
struct FuseQdqDepthwiseConv : public OpRewritePattern<QuantizeAffineOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(QuantizeAffineOp quant,
                                PatternRewriter &rewriter) const override {
    std::optional<Affine8> output = matchOutputQuant(quant);
    if (!output) {
      return failure();
    }
    Value accumulator = output->codes;
    auto conv = accumulator.getDefiningOp<linalg::DepthwiseConv2DNchwChwOp>();
    if (!conv || !accumulator.hasOneUse()) {
      return rewriter.notifyMatchFailure(quant, "not fed by a NCHW depthwise");
    }
    auto accType = cast<RankedTensorType>(conv.getResult(0).getType());
    if (!accType.getElementType().isF32() || !accType.hasStaticShape() ||
        accType.getRank() != 4) {
      return failure();
    }

    // The activation: DQ, optionally through a zero-filled spatial pad.
    Value input = conv.getDpsInputs()[0];
    tensor::PadOp pad = input.getDefiningOp<tensor::PadOp>();
    if (pad) {
      std::optional<double> fill = constantFloat(pad.getConstantPaddingValue());
      ArrayRef<int64_t> low = pad.getStaticLow();
      ArrayRef<int64_t> high = pad.getStaticHigh();
      if (!fill || *fill != 0.0 || pad.getNofold() || !pad.getLow().empty() ||
          !pad.getHigh().empty() || low.size() != 4 || low[0] != 0 ||
          low[1] != 0 || high[0] != 0 || high[1] != 0) {
        return failure();
      }
      input = pad.getSource();
    }
    std::optional<Affine8> activation =
        matchActivationDequant(input.getDefiningOp<DequantizeAffineOp>());
    FailureOr<LayoutChain> filterChain = walkToDequant(conv.getDpsInputs()[1]);
    if (!activation || failed(filterChain)) {
      return rewriter.notifyMatchFailure(quant, "operands are not DQ'd");
    }
    std::optional<Affine8> weights = matchWeightDequant(filterChain->dequant);
    if (!weights) {
      return rewriter.notifyMatchFailure(quant, "weight DQ is not per-tensor");
    }
    for (Operation *op : filterChain->ops) {
      if (isa<tensor::PadOp>(op)) {
        return failure();
      }
    }

    DequantizeAffineOp biasDequant;
    if (failed(matchBiasInit(conv.getDpsInits()[0], biasDequant,
                             /*channel=*/1))) {
      return rewriter.notifyMatchFailure(quant, "init is not a DQ'd bias");
    }
    int64_t channels = accType.getDimSize(1);
    double accumulatorScale = activation->scale * weights->scale;
    if (biasDequant) {
      std::optional<double> biasScale = constantFloat(biasDequant.getScale());
      if (biasDequant.getInputType().getDimSize(0) != channels || !biasScale ||
          std::abs(*biasScale - accumulatorScale) >
              kBiasScaleTolerance * accumulatorScale) {
        return rewriter.notifyMatchFailure(quant, "bias scale is not sx * sw");
      }
    }
    double ratio = accumulatorScale / output->scale;
    if (!std::isfinite(ratio) || !(ratio > 0.0) ||
        !std::isfinite(static_cast<float>(ratio))) {
      return failure();
    }

    Location loc = conv.getLoc();
    rewriter.setInsertionPoint(quant);
    Type i8 = rewriter.getI8Type();
    Type i32 = rewriter.getI32Type();
    const int64_t toNhwc[] = {0, 2, 3, 1};
    const int64_t toNchw[] = {0, 3, 1, 2};

    Value codes = activation->codes;
    if (activation->isUnsigned) {
      codes = emitShift(rewriter, loc, codes);
    }
    Value convInput = emitTranspose(rewriter, loc, codes, toNhwc);
    if (pad) {
      ArrayRef<int64_t> low = pad.getStaticLow();
      ArrayRef<int64_t> high = pad.getStaticHigh();
      SmallVector<int64_t> padLow = {0, low[2], low[3], 0};
      SmallVector<int64_t> padHigh = {0, high[2], high[3], 0};
      auto unpadded = cast<RankedTensorType>(convInput.getType());
      auto paddedType = RankedTensorType::get(
          {unpadded.getDimSize(0), unpadded.getDimSize(1) + low[2] + high[2],
           unpadded.getDimSize(2) + low[3] + high[3], unpadded.getDimSize(3)},
          i8);
      Value fillValue = arith::ConstantOp::create(
          rewriter, loc, rewriter.getIntegerAttr(i8, activation->zeroPoint));
      convInput = tensor::PadOp::create(
                      rewriter, loc, paddedType, convInput,
                      getAsIndexOpFoldResult(rewriter.getContext(), padLow),
                      getAsIndexOpFoldResult(rewriter.getContext(), padHigh),
                      fillValue)
                      .getResult();
    }

    FailureOr<Value> filterChw = rebuildChain(rewriter, loc, filterChain->ops,
                                              weights->codes, std::nullopt);
    if (failed(filterChw) ||
        cast<RankedTensorType>(filterChw->getType()).getRank() != 3) {
      return failure();
    }
    const int64_t toHwc[] = {1, 2, 0};
    Value filter = emitTranspose(rewriter, loc, *filterChw, toHwc);

    SmallVector<int64_t> accShape = {accType.getDimSize(0),
                                     accType.getDimSize(2),
                                     accType.getDimSize(3), channels};
    Value zero =
        arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(0));
    Value accEmpty = tensor::EmptyOp::create(rewriter, loc, accShape, i32);
    Value accInit = linalg::FillOp::create(rewriter, loc, ValueRange{zero},
                                           ValueRange{accEmpty})
                        .getResult(0);
    auto attrType = RankedTensorType::get({2}, rewriter.getIntegerType(64));
    SmallVector<int64_t> strides(conv.getStrides().getValues<int64_t>());
    SmallVector<int64_t> dilations(conv.getDilations().getValues<int64_t>());
    Value integerAcc =
        linalg::DepthwiseConv2DNhwcHwcOp::create(
            rewriter, loc, TypeRange{accInit.getType()},
            ValueRange{convInput, filter}, ValueRange{accInit},
            DenseIntElementsAttr::get(attrType, strides),
            DenseIntElementsAttr::get(attrType, dilations))
            .getResult(0);

    Value bias = emitFoldedBias(rewriter, loc,
                                biasDequant ? biasDequant.getInput() : Value(),
                                filter, activation->zeroPoint, channels);
    Value result = emitRequantTail(rewriter, loc, integerAcc, bias, ratio,
                                   *output, toNchw);
    if (result.getType() != quant->getResult(0).getType()) {
      return failure();
    }
    rewriter.replaceOp(quant, result);
    return success();
  }
};

struct RocketFuseQdqConvPass
    : public PassWrapper<RocketFuseQdqConvPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RocketFuseQdqConvPass)

  StringRef getArgument() const final { return "rocket-fuse-qdq-conv"; }
  StringRef getDescription() const final {
    return "Rewrites an ONNX QDQ convolution, Q(conv(DQ(x), DQ(w)) + DQ(b)) "
           "with per-tensor scales, into the canonical requantized int8 form "
           "the Rocket int8_requant matchers claim, folding the input zero "
           "point into the bias.";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    math::MathDialect, tensor::TensorDialect>();
  }

  void runOnOperation() final {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<FuseQdqConv, FuseQdqDepthwiseConv>(context);
    // No folding, as in the other int8 fusions: the folder would hoist the
    // canonical generic's scalar operands and fold its transposes before
    // the matchers see them.
    GreedyRewriteConfig config;
    config.enableFolding(false);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      return signalPassFailure();
    }
  }
};

static PassRegistration<RocketFuseQdqConvPass> reg;

} // namespace
} // namespace mlir::iree_compiler::IREE::HAL
