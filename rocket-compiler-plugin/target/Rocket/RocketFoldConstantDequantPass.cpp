// Copyright 2026
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Folds an `iree_linalg_ext.dequantize_affine` of a constant into the
// constant it computes, when its result is only ever a matmul's or a
// convolution's weight operand.
//
// # Why
//
// An ONNX QDQ model stores every weight as `DequantizeLinear(w_i8)`. The
// convolutions' weights are consumed as codes by `rocket-fuse-qdq-conv`,
// which runs just before this pass. What is left are the weights that pass
// declines -- a Gemm/MatMul's, and a per-channel-scaled convolution's -- and
// those reach the contraction as `transpose(dequantize_affine(constant))`.
// IREE's const-expr hoisting would precompute that, except that widening
// `i8` to `f32` (and then `f16`) grows it past the default 1 MiB
// `--iree-opt-const-expr-max-size-increase-threshold`, so it declines. The
// weight is then dequantized on the CPU on every call, and because the
// Rocket matmul sees a non-constant weight, `rocket-pack-weights` cannot
// pack it at compile time and the driver re-packs it on every call too.
// Stock ORT `quantize_static` ResNet50: a CPU
// `elementwise_transpose_2048x1000_i8xf16` dispatch plus 5.3 ms of
// `pack.weights` per call, about 10 of its 27 ms
// (agent-docs/quantization-byo-measured.md).
//
// Raising the threshold for the whole module fixes that too, but it applies
// to every const-expr in every model. This pass is the narrow version: it
// only touches a dequantization whose every use is a weight. After it the
// weight is an ordinary f32 constant, and the transpose and the f16 demote
// downstream shrink it, so hoisting and const-eval take them as they take a
// float model's weights.
//
// # What counts as a weight
//
// Operand 1 -- `B` of a contraction (`linalg.matmul`, `batch_matmul`, ...),
// the filter of a convolution -- reached through at most `kMaxChain` layout
// ops: `linalg.transpose`, `tensor.expand_shape`, `tensor.collapse_shape`,
// `tensor.cast`. A dequantization with any other use (an activation, an
// embedding table, a bias in an init) is left alone, and so is one whose
// result would exceed `kMaxFoldedBytes`.
//
// # Only when offloading
//
// The fold pays because the NPU packs a constant weight once and the CPU no
// longer dequantizes it per call. A weight the CPU keeps is a different
// trade: its convolution or matmul reads the i8 codes with the
// dequantization fused, and the folded f32 constant is four times the
// bytes. Measured in the `--no-offload` arm, folding made per-channel QDQ
// ResNet50 1512 -> 2034 ms and MobileNetV2 50.2 -> 57.9 ms. So the spec
// runs this pass with `offload = true`, and `rocket-compiler --no-offload`
// flips it to false, as it does for `rocket-block-diagonal-grouped-conv`.
// A weight whose operation the planner later leaves on the CPU in an
// offloading build is still folded; this pass runs before placement.
//
// # Exactness
//
// The fold evaluates the op's own decomposition
// (`DequantizeAffineOp::decomposeOperation`) element by element:
//
//   real = to_compute(input)                         // symmetric
//   real = sitofp(ext(input) - ext(zero_point))      // otherwise, in
//                                                    // getZeroPointDifferenceType
//   out  = to_output(real * scale)                   // in the scale's type
//
// with every conversion and the product rounded to nearest-even, as the
// `arith` ops it lowers to are. So the constant is bit-identical to what the
// CPU dispatch computed at run time.

#include <cstring>
#include <optional>
#include <vector>

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::HAL {
namespace {

using IREE::LinalgExt::DequantizeAffineOp;

/// The longest chain of layout ops between the dequantization and the
/// weight operand. A Gemm's is one transpose; a convolution filter's after
/// the channels-last conversion is one transpose too. Generous on purpose.
constexpr int kMaxChain = 6;

/// The largest constant this pass will materialize. ResNet50's FC weight is
/// 8 MiB in f32; this is the threshold the measurement passed to IREE.
constexpr int64_t kMaxFoldedBytes = int64_t{64} << 20;

/// Whether every use of `value` is, through layout ops only, operand 1 of a
/// contraction or a convolution.
static bool feedsOnlyWeights(Value value, int depth = 0) {
  if (value.use_empty() || depth > kMaxChain) {
    return false;
  }
  for (OpOperand &use : value.getUses()) {
    Operation *owner = use.getOwner();
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(owner)) {
      if (auto transpose = dyn_cast<linalg::TransposeOp>(owner)) {
        if (&use != &transpose.getInputMutable() ||
            !feedsOnlyWeights(transpose->getResult(0), depth + 1)) {
          return false;
        }
        continue;
      }
      bool isWeight =
          use.getOperandNumber() == 1 && linalgOp.isDpsInput(&use) &&
          (linalg::isaContractionOpInterface(linalgOp) ||
           linalg::isaConvolutionOpInterface(linalgOp));
      if (!isWeight) {
        return false;
      }
      continue;
    }
    if (isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp, tensor::CastOp>(
            owner)) {
      if (!feedsOnlyWeights(owner->getResult(0), depth + 1)) {
        return false;
      }
      continue;
    }
    return false;
  }
  return true;
}

/// One constant operand of the op, read element by element through its
/// indexing map.
class ConstantOperand {
public:
  /// Reads `value`, a constant scalar or a constant statically shaped
  /// tensor whose element width is a whole number of bytes, indexed by
  /// `map` (a projected permutation of the op's loops).
  static std::optional<ConstantOperand> read(Value value, AffineMap map) {
    ConstantOperand result;
    Type type = value.getType();
    Type elementType = getElementTypeOrSelf(type);
    if (!elementType.isIntOrFloat()) {
      return std::nullopt;
    }
    unsigned bits = elementType.getIntOrFloatBitWidth();
    if (bits == 0 || bits % 8 != 0 || bits > 64) {
      return std::nullopt;
    }
    result.bitWidth = bits;
    result.elementBytes = bits / 8;
    if (!isa<ShapedType>(type)) {
      if (map.getNumResults() != 0) {
        return std::nullopt;
      }
      APInt pattern;
      APFloat floatValue(0.0f);
      if (matchPattern(value, m_ConstantFloat(&floatValue))) {
        pattern = floatValue.bitcastToAPInt();
      } else if (!matchPattern(value, m_ConstantInt(&pattern))) {
        return std::nullopt;
      }
      if (pattern.getBitWidth() != bits) {
        return std::nullopt;
      }
      result.scalar = pattern;
      result.splat = true;
      return result;
    }
    auto tensorType = dyn_cast<RankedTensorType>(type);
    if (!tensorType || !tensorType.hasStaticShape() ||
        !map.isProjectedPermutation() ||
        map.getNumResults() != static_cast<unsigned>(tensorType.getRank())) {
      return std::nullopt;
    }
    Attribute attr;
    if (!matchPattern(value, m_Constant(&attr))) {
      return std::nullopt;
    }
    if (auto dense = dyn_cast<DenseElementsAttr>(attr)) {
      result.bytes = dense.getRawData();
      result.splat = dense.isSplat();
    } else if (auto resource = dyn_cast<DenseResourceElementsAttr>(attr)) {
      AsmResourceBlob *blob = resource.getRawHandle().getBlob();
      if (!blob) {
        return std::nullopt;
      }
      result.bytes = blob->getData();
    } else {
      return std::nullopt;
    }
    int64_t count = tensorType.getNumElements();
    size_t expected = result.splat ? result.elementBytes
                                   : static_cast<size_t>(count) *
                                         result.elementBytes;
    if (result.bytes.size() != expected) {
      return std::nullopt;
    }
    int64_t stride = 1;
    result.loops.resize(tensorType.getRank());
    result.strides.resize(tensorType.getRank());
    for (int64_t dim = tensorType.getRank() - 1; dim >= 0; --dim) {
      result.loops[dim] = map.getDimPosition(dim);
      result.strides[dim] = stride;
      stride *= tensorType.getDimSize(dim);
    }
    return result;
  }

  /// The element at loop index `index`, as its bit pattern.
  APInt load(ArrayRef<int64_t> index) const {
    if (scalar) {
      return *scalar;
    }
    int64_t offset = 0;
    if (!splat) {
      for (size_t dim = 0; dim < loops.size(); ++dim) {
        offset += index[loops[dim]] * strides[dim];
      }
    }
    uint64_t raw = 0;
    std::memcpy(&raw, bytes.data() + offset * elementBytes, elementBytes);
    return APInt(bitWidth, raw);
  }

  bool isScalar() const { return splat; }

private:
  /// A scalar operand's value; a tensor's elements are in `bytes`, which
  /// points into the attribute's storage.
  std::optional<APInt> scalar;
  ArrayRef<char> bytes;
  unsigned bitWidth = 0;
  unsigned elementBytes = 0;
  bool splat = false;
  SmallVector<int64_t> loops;
  SmallVector<int64_t> strides;
};

/// The folded result of `op`, or null when the op is not foldable here.
static DenseElementsAttr foldDequant(DequantizeAffineOp op) {
  if (op->getNumResults() != 1) {
    return {};
  }
  auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  auto inputType = dyn_cast<RankedTensorType>(op.getInput().getType());
  if (!resultType || !inputType || !resultType.hasStaticShape() ||
      resultType.getShape() != inputType.getShape() ||
      !op.getOutputMap().isIdentity()) {
    return {};
  }
  auto inputElement = dyn_cast<IntegerType>(inputType.getElementType());
  auto computeType = dyn_cast<FloatType>(op.getScaleElementType());
  auto realType = dyn_cast<FloatType>(resultType.getElementType());
  if (!inputElement || inputElement.getWidth() > 32 || !computeType ||
      !realType || realType.getWidth() % 8 != 0) {
    return {};
  }
  int64_t count = resultType.getNumElements();
  int64_t realBytes = realType.getWidth() / 8;
  if (count <= 0 || count > kMaxFoldedBytes / realBytes) {
    return {};
  }

  std::optional<ConstantOperand> input =
      ConstantOperand::read(op.getInput(), op.getInputMap());
  std::optional<ConstantOperand> scale =
      ConstantOperand::read(op.getScale(), op.getScaleMap());
  if (!input || !scale) {
    return {};
  }
  bool symmetric = op.isSymmetric();
  std::optional<ConstantOperand> zeroPoint;
  unsigned differenceWidth = 0;
  if (!symmetric) {
    if (!isa<IntegerType>(getElementTypeOrSelf(op.getZeroPoint().getType()))) {
      return {};
    }
    zeroPoint = ConstantOperand::read(op.getZeroPoint(), op.getZeroPointMap());
    std::optional<IntegerType> differenceType = op.getZeroPointDifferenceType();
    if (!zeroPoint || !differenceType) {
      return {};
    }
    differenceWidth = differenceType->getWidth();
  }

  const llvm::fltSemantics &computeSemantics = computeType.getFloatSemantics();
  const llvm::fltSemantics &realSemantics = realType.getFloatSemantics();
  bool convertResult = &computeSemantics != &realSemantics;
  bool inputUnsigned = op.getInputUnsigned();
  bool zpUnsigned = op.getZpUnsigned();
  constexpr auto kRound = llvm::RoundingMode::NearestTiesToEven;

  // f32 x f32 -> f32 with a per-tensor scale is every real model's weight;
  // native float arithmetic is IEEE round-to-nearest-even and a few hundred
  // times faster than APFloat on a multi-megabyte constant.
  bool nativeF32 = computeType.isF32() && realType.isF32() && scale->isScalar();
  float nativeScale = 0.0f;
  if (nativeF32) {
    uint32_t scaleBits =
        static_cast<uint32_t>(scale->load({}).getZExtValue());
    std::memcpy(&nativeScale, &scaleBits, sizeof(nativeScale));
  }

  std::vector<char> buffer(static_cast<size_t>(count * realBytes));
  int64_t rank = resultType.getRank();
  SmallVector<int64_t> index(rank, 0);
  for (int64_t linear = 0; linear < count; ++linear) {
    APInt code = input->load(index);
    // The integer the conversion to the compute type sees.
    APInt integer = code;
    bool integerSigned = !inputUnsigned;
    if (!symmetric) {
      APInt wide = inputUnsigned ? code.zext(differenceWidth)
                                 : code.sext(differenceWidth);
      APInt zp = zeroPoint->load(index);
      APInt zpWide = zp.getBitWidth() > differenceWidth
                         ? zp.trunc(differenceWidth)
                         : (zpUnsigned ? zp.zext(differenceWidth)
                                       : zp.sext(differenceWidth));
      integer = wide - zpWide;
      integerSigned = true;
    }
    if (nativeF32) {
      // Below 2^24 in magnitude for every input up to 16 bits, so exact;
      // wider ones round to nearest-even as sitofp does.
      int64_t whole = integerSigned ? integer.getSExtValue()
                                    : static_cast<int64_t>(integer.getZExtValue());
      float product = static_cast<float>(whole) * nativeScale;
      std::memcpy(buffer.data() + linear * realBytes, &product, sizeof(product));
    } else {
      APFloat real(computeSemantics);
      real.convertFromAPInt(integer, integerSigned, kRound);
      APFloat factor(computeSemantics, scale->load(index));
      real.multiply(factor, kRound);
      if (convertResult) {
        bool losesInfo = false;
        real.convert(realSemantics, kRound, &losesInfo);
      }
      uint64_t raw = real.bitcastToAPInt().getZExtValue();
      std::memcpy(buffer.data() + linear * realBytes, &raw, realBytes);
    }
    for (int64_t dim = rank - 1; dim >= 0; --dim) {
      if (++index[dim] < resultType.getDimSize(dim)) {
        break;
      }
      index[dim] = 0;
    }
  }
  return DenseElementsAttr::getFromRawBuffer(resultType, buffer);
}

struct RocketFoldConstantDequantPass
    : public PassWrapper<RocketFoldConstantDequantPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RocketFoldConstantDequantPass)

  RocketFoldConstantDequantPass() = default;
  RocketFoldConstantDequantPass(const RocketFoldConstantDequantPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const final {
    return "rocket-fold-constant-dequant";
  }
  StringRef getDescription() const final {
    return "Folds a dequantize_affine of a constant whose every use is a "
           "matmul or convolution weight into the f32 constant it computes, "
           "so const-expr hoisting and rocket-pack-weights see a constant "
           "weight.";
  }

  Option<bool> offload{
      *this, "offload",
      llvm::cl::desc("Fold at all. False in a --no-offload build: there every "
                     "weight stays on the CPU, whose convolutions and "
                     "matmuls read the i8 codes with the dequantization "
                     "fused, faster than they read the f32 constant."),
      llvm::cl::init(true)};

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect>();
  }

  void runOnOperation() final {
    if (!offload) {
      return;
    }
    SmallVector<DequantizeAffineOp> candidates;
    getOperation()->walk([&](DequantizeAffineOp op) {
      if (op->getNumResults() == 1 && feedsOnlyWeights(op->getResult(0))) {
        candidates.push_back(op);
      }
    });
    for (DequantizeAffineOp op : candidates) {
      DenseElementsAttr folded = foldDequant(op);
      if (!folded) {
        continue;
      }
      OpBuilder builder(op);
      Value constant =
          arith::ConstantOp::create(builder, op.getLoc(), folded).getResult();
      op->getResult(0).replaceAllUsesWith(constant);
      op->erase();
    }
  }
};

static PassRegistration<RocketFoldConstantDequantPass> reg;

} // namespace
} // namespace mlir::iree_compiler::IREE::HAL
