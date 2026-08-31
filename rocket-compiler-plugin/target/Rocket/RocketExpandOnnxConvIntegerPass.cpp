// Copyright 2026
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Expands `torch.operator "onnx.ConvInteger"` into the quantized torch ops
// that torch-mlir already lowers, because torch-mlir has no OnnxToTorch
// pattern for ConvInteger at any opset -- it is the single op that stops
// `iree-compile --iree-input-type=onnx` on an ONNX Runtime
// `quantize_dynamic` model (DynamicQuantizeLinear + ConvInteger +
// MatMulInteger; the other two already convert).
//
// This runs as an input-conversion *preprocessing* pass, i.e. before the
// torch plugin's onnx->torch pipeline, which is the only place it can run:
// --iree-preprocessing-pass-pipeline is downstream of input conversion, on
// IR where the offending torch.operator has already failed to legalize.
//
// The expansion mirrors what torch-mlir's own "QLinearConv" pattern does,
// with the scales pinned to 1.0 -- which is exactly ConvInteger's semantics,
// an i8 x i8 -> i32 convolution with per-tensor zero-point offsets and no
// output requantization:
//
//   x_zp   = aten.item(x_zero_point)          // 0 if operand is absent
//   w_zp   = aten.item(w_zero_point)          // 0 if operand is absent
//   xpad   = aten.constant_pad_nd(x, .., x_zp)   // only if pads are asymmetric
//   qx     = aten._make_per_tensor_quantized_tensor(xpad, 1.0, x_zp)
//   qw     = aten._make_per_tensor_quantized_tensor(w,    1.0, w_zp)
//   qy     = aten.convolution(qx, qw, none, strides, pads, dilations, ..)
//   y      = aten.int_repr(qy)
//
// Torch->Linalg turns that into linalg.conv_2d_nchw_fchw_q (dense) or
// linalg.depthwise_conv_2d_nhwc_hwc_q (group == channels) with i8 operands,
// i32 zero points and an i32 accumulator -- the shape RocketTarget.cpp's
// "int8_accumulator" precision serializes.
//
// Two deliberate details:
//
//   * Symmetric padding is handed to aten.convolution rather than
//     materialized here. Torch->Linalg pads a *quantized* convolution with
//     the input zero point already (the tensor.pad it emits yields the same
//     value it feeds to the conv's zero-point operand), which is what ONNX
//     requires of ConvInteger, so folding the pads in is both correct and
//     one less op.
//
//   * Asymmetric padding is materialized here, on the raw integer tensor,
//     with aten.constant_pad_nd. We cannot route it through torch-mlir's
//     onnx.Conv pattern the way QLinearConv does: that path builds an
//     aten.pad whose pad value is left null for quantized dtypes (crashing
//     the compiler) and is an integer for integer dtypes, which aten.pad's
//     own verifier rejects ("operand #3 must be Optional torch float type").
//     Padding before the quantized cast keeps the uniform ui8->i8 shift
//     Torch->Linalg applies covering the padded region too.

#include <memory>
#include <optional>

#include "RocketPasses.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"

namespace mlir::iree_compiler::IREE::HAL {
namespace {

namespace Torch = mlir::torch::Torch;

// Reads an `torch.onnx.<name>` si64 array attribute, or {} when absent.
SmallVector<int64_t> getIntArrayAttr(Operation *op, StringRef name) {
  SmallVector<int64_t> values;
  auto arrayAttr = op->getAttrOfType<ArrayAttr>(name);
  if (!arrayAttr) {
    return values;
  }
  for (Attribute attr : arrayAttr) {
    auto intAttr = dyn_cast<IntegerAttr>(attr);
    if (!intAttr) {
      return {};
    }
    values.push_back(intAttr.getValue().getSExtValue());
  }
  return values;
}

int64_t getIntAttrOr(Operation *op, StringRef name, int64_t fallback) {
  if (auto intAttr = op->getAttrOfType<IntegerAttr>(name)) {
    return intAttr.getValue().getSExtValue();
  }
  return fallback;
}

// The !torch.q* dtype ConvInteger's i8-family operands quantize to. ONNX
// restricts x and w to (u)int8, so anything else means we misread the IR.
Type getQuantizedDtype(Type dtype) {
  MLIRContext *context = dtype.getContext();
  if (dtype.isUnsignedInteger(8)) {
    return Torch::QUInt8Type::get(context);
  }
  if (dtype.isSignedInteger(8)) {
    return Torch::QInt8Type::get(context);
  }
  return nullptr;
}

// Explicit ONNX `pads` for the op, honoring `auto_pad`. Returns the
// [begin_0..begin_n-1, end_0..end_n-1] form ONNX uses, or nullopt when the
// requested padding cannot be resolved statically.
std::optional<SmallVector<int64_t>>
resolvePads(Operation *op, ArrayRef<int64_t> inputSizes,
            ArrayRef<int64_t> kernel, ArrayRef<int64_t> strides,
            ArrayRef<int64_t> dilations) {
  int64_t spatialRank = kernel.size();
  StringRef autoPad = "NOTSET";
  if (auto autoPadAttr = op->getAttrOfType<StringAttr>("torch.onnx.auto_pad")) {
    autoPad = autoPadAttr.getValue();
  }

  if (autoPad == "VALID") {
    return SmallVector<int64_t>(2 * spatialRank, 0);
  }
  if (autoPad == "NOTSET") {
    SmallVector<int64_t> pads = getIntArrayAttr(op, "torch.onnx.pads");
    if (pads.empty()) {
      return SmallVector<int64_t>(2 * spatialRank, 0);
    }
    if (static_cast<int64_t>(pads.size()) != 2 * spatialRank) {
      return std::nullopt;
    }
    return pads;
  }
  if (autoPad != "SAME_UPPER" && autoPad != "SAME_LOWER") {
    return std::nullopt;
  }

  // SAME_*: the pad totals depend on the input's spatial extents, so they can
  // only be resolved when those are static.
  SmallVector<int64_t> pads(2 * spatialRank, 0);
  for (int64_t i = 0; i < spatialRank; ++i) {
    int64_t inputSize = inputSizes[2 + i];
    if (inputSize == Torch::kUnknownSize) {
      return std::nullopt;
    }
    int64_t effectiveKernel = (kernel[i] - 1) * dilations[i] + 1;
    int64_t outputSize = (inputSize + strides[i] - 1) / strides[i];
    int64_t needed = std::max<int64_t>(
        0, (outputSize - 1) * strides[i] + effectiveKernel - inputSize);
    int64_t half = needed / 2;
    if (autoPad == "SAME_UPPER") {
      pads[i] = half;
      pads[spatialRank + i] = needed - half;
    } else {
      pads[spatialRank + i] = half;
      pads[i] = needed - half;
    }
  }
  return pads;
}

LogicalResult expandConvInteger(Torch::OperatorOp op, IRRewriter &rewriter) {
  Location loc = op.getLoc();
  MLIRContext *context = op.getContext();
  rewriter.setInsertionPoint(op);

  if (op.getNumOperands() < 2 || op->getNumResults() != 1) {
    return op.emitWarning("onnx.ConvInteger: expected 2-4 operands and one "
                          "result; left for the ONNX importer to reject");
  }
  auto resultType = dyn_cast<Torch::ValueTensorType>(op->getResult(0).getType());
  auto inputType = dyn_cast<Torch::ValueTensorType>(op.getOperand(0).getType());
  auto weightType = dyn_cast<Torch::ValueTensorType>(op.getOperand(1).getType());
  if (!resultType || !inputType || !weightType || !inputType.hasSizes() ||
      !weightType.hasSizes() || !resultType.hasSizes()) {
    return op.emitWarning(
        "onnx.ConvInteger: expected ranked !torch.vtensor operands and result");
  }

  Type quantizedInputDtype = getQuantizedDtype(inputType.getDtype());
  Type quantizedWeightDtype = getQuantizedDtype(weightType.getDtype());
  if (!quantizedInputDtype || !quantizedWeightDtype) {
    return op.emitWarning("onnx.ConvInteger: expected 8-bit integer input and "
                          "weight element types");
  }

  ArrayRef<int64_t> inputSizes = inputType.getSizes();
  ArrayRef<int64_t> weightSizes = weightType.getSizes();
  int64_t spatialRank = static_cast<int64_t>(inputSizes.size()) - 2;
  if (spatialRank < 1 ||
      static_cast<int64_t>(weightSizes.size()) != spatialRank + 2) {
    return op.emitWarning("onnx.ConvInteger: input and weight ranks disagree");
  }

  // kernel_shape is redundant with the weight's own shape; prefer the shape
  // and only fall back to the attribute when the weight is dynamic there.
  SmallVector<int64_t> kernel(weightSizes.begin() + 2, weightSizes.end());
  SmallVector<int64_t> kernelAttr = getIntArrayAttr(op, "torch.onnx.kernel_shape");
  for (int64_t i = 0; i < spatialRank; ++i) {
    if (kernel[i] == Torch::kUnknownSize) {
      if (static_cast<int64_t>(kernelAttr.size()) != spatialRank) {
        return op.emitWarning("onnx.ConvInteger: dynamic kernel extent with no "
                              "usable kernel_shape attribute");
      }
      kernel[i] = kernelAttr[i];
    }
  }

  SmallVector<int64_t> strides = getIntArrayAttr(op, "torch.onnx.strides");
  if (strides.empty()) {
    strides.assign(spatialRank, 1);
  }
  SmallVector<int64_t> dilations = getIntArrayAttr(op, "torch.onnx.dilations");
  if (dilations.empty()) {
    dilations.assign(spatialRank, 1);
  }
  if (static_cast<int64_t>(strides.size()) != spatialRank ||
      static_cast<int64_t>(dilations.size()) != spatialRank) {
    return op.emitWarning(
        "onnx.ConvInteger: strides/dilations do not match the spatial rank");
  }
  int64_t group = getIntAttrOr(op, "torch.onnx.group", 1);

  std::optional<SmallVector<int64_t>> pads =
      resolvePads(op, inputSizes, kernel, strides, dilations);
  if (!pads) {
    return op.emitWarning("onnx.ConvInteger: unsupported padding (auto_pad "
                          "must be NOTSET/VALID, or SAME_* with static "
                          "spatial input extents)");
  }

  // ONNX makes both zero points optional and defaulting to 0. A 1-D
  // w_zero_point with more than one element is ONNX's per-output-channel
  // form, which the quantized linalg convs cannot express.
  Value constantZero;
  auto extractZeroPoint = [&](unsigned index) -> std::optional<Value> {
    if (op.getNumOperands() <= index) {
      if (!constantZero) {
        constantZero = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(0));
      }
      return constantZero;
    }
    Value zeroPoint = op.getOperand(index);
    auto zeroPointType = dyn_cast<Torch::ValueTensorType>(zeroPoint.getType());
    if (!zeroPointType || !zeroPointType.hasSizes()) {
      return std::nullopt;
    }
    for (int64_t size : zeroPointType.getSizes()) {
      if (size != 1) {
        return std::nullopt;
      }
    }
    return Torch::AtenItemOp::create(rewriter, loc, Torch::IntType::get(context),
                                     zeroPoint)
        .getResult();
  };
  std::optional<Value> inputZeroPoint = extractZeroPoint(2);
  std::optional<Value> weightZeroPoint = extractZeroPoint(3);
  if (!inputZeroPoint || !weightZeroPoint) {
    return op.emitWarning("onnx.ConvInteger: only per-tensor (scalar) zero "
                          "points are supported");
  }

  // Asymmetric ONNX pads have no aten.convolution spelling, so materialize
  // them here, on the pre-quantization integer tensor, filled with the input
  // zero point. aten.constant_pad_nd's list runs from the innermost dimension
  // outwards in (begin, end) pairs, while ONNX lists all begins then all ends.
  Value input = op.getOperand(0);
  SmallVector<int64_t> convPadding(spatialRank, 0);
  bool symmetric = true;
  for (int64_t i = 0; i < spatialRank; ++i) {
    if ((*pads)[i] != (*pads)[spatialRank + i]) {
      symmetric = false;
    }
  }
  if (symmetric) {
    for (int64_t i = 0; i < spatialRank; ++i) {
      convPadding[i] = (*pads)[i];
    }
  } else {
    SmallVector<Value> padValues;
    SmallVector<int64_t> paddedSizes(inputSizes);
    for (int64_t i = spatialRank - 1; i >= 0; --i) {
      padValues.push_back(Torch::ConstantIntOp::create(
          rewriter, loc, rewriter.getI64IntegerAttr((*pads)[i])));
      padValues.push_back(Torch::ConstantIntOp::create(
          rewriter, loc, rewriter.getI64IntegerAttr((*pads)[spatialRank + i])));
      if (paddedSizes[2 + i] != Torch::kUnknownSize) {
        paddedSizes[2 + i] += (*pads)[i] + (*pads)[spatialRank + i];
      }
    }
    Value padList = Torch::PrimListConstructOp::create(
        rewriter, loc, Torch::ListType::get(Torch::IntType::get(context)),
        padValues);
    auto paddedType = rewriter.getType<Torch::ValueTensorType>(
        paddedSizes, inputType.getDtype());
    input = Torch::AtenConstantPadNdOp::create(rewriter, loc, paddedType, input,
                                               padList, *inputZeroPoint);
    inputSizes = cast<Torch::ValueTensorType>(input.getType()).getSizes();
  }

  // ConvInteger is an unscaled integer convolution, so both operands enter
  // the quantized domain with scale 1.0 and leave it via int_repr.
  Value one = Torch::ConstantFloatOp::create(rewriter, loc,
                                             rewriter.getF64FloatAttr(1.0));
  Value quantizedInput = Torch::Aten_MakePerTensorQuantizedTensorOp::create(
      rewriter, loc,
      rewriter.getType<Torch::ValueTensorType>(inputSizes, quantizedInputDtype),
      input, one, *inputZeroPoint);
  Value quantizedWeight = Torch::Aten_MakePerTensorQuantizedTensorOp::create(
      rewriter, loc,
      rewriter.getType<Torch::ValueTensorType>(weightSizes,
                                               quantizedWeightDtype),
      op.getOperand(1), one, *weightZeroPoint);

  auto makeIntList = [&](ArrayRef<int64_t> values) {
    SmallVector<Value> elements;
    for (int64_t value : values) {
      elements.push_back(Torch::ConstantIntOp::create(
          rewriter, loc, rewriter.getI64IntegerAttr(value)));
    }
    return Torch::PrimListConstructOp::create(
               rewriter, loc, Torch::ListType::get(Torch::IntType::get(context)),
               elements)
        .getResult();
  };

  Value convolution = Torch::AtenConvolutionOp::create(
      rewriter, loc,
      rewriter.getType<Torch::ValueTensorType>(resultType.getOptionalSizes(),
                                               Torch::QInt32Type::get(context)),
      quantizedInput, quantizedWeight,
      /*bias=*/Torch::ConstantNoneOp::create(rewriter, loc),
      /*stride=*/makeIntList(strides),
      /*padding=*/makeIntList(convPadding),
      /*dilation=*/makeIntList(dilations),
      /*transposed=*/Torch::ConstantBoolOp::create(rewriter, loc, false),
      /*output_padding=*/makeIntList(SmallVector<int64_t>(spatialRank, 0)),
      /*groups=*/
      Torch::ConstantIntOp::create(rewriter, loc,
                                   rewriter.getI64IntegerAttr(group)));

  rewriter.replaceOpWithNewOp<Torch::AtenIntReprOp>(op, resultType,
                                                    convolution);
  return success();
}

struct RocketExpandOnnxConvIntegerPass
    : public PassWrapper<RocketExpandOnnxConvIntegerPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RocketExpandOnnxConvIntegerPass)

  StringRef getArgument() const final {
    return "rocket-expand-onnx-conv-integer";
  }
  StringRef getDescription() const final {
    return "Expands onnx.ConvInteger into quantized torch ops that lower to "
           "linalg's quantized convolutions.";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<Torch::TorchDialect>();
  }

  void runOnOperation() final {
    Operation *root = getOperation();
    SmallVector<Torch::OperatorOp> targets;
    root->walk([&](Torch::OperatorOp op) {
      if (op.getName() == "onnx.ConvInteger") {
        targets.push_back(op);
      }
    });
    if (targets.empty()) {
      return;
    }
    IRRewriter rewriter(root->getContext());
    for (Torch::OperatorOp op : targets) {
      // A failure here has already emitted a warning explaining which
      // ConvInteger was left alone; the op survives and the ONNX importer
      // reports it as unlegalizable, so there is nothing to abort for.
      (void)expandConvInteger(op, rewriter);
    }
  }
};

static PassRegistration<RocketExpandOnnxConvIntegerPass> reg;

} // namespace

std::unique_ptr<Pass> createRocketExpandOnnxConvIntegerPass() {
  return std::make_unique<RocketExpandOnnxConvIntegerPass>();
}

} // namespace mlir::iree_compiler::IREE::HAL
