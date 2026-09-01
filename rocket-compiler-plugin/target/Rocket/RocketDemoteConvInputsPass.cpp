// Copyright 2026
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Demotes the input operands of an all-f32 named 2-D convolution to f16,
// leaving the accumulator at f32 -- Rocket's ABI is f16-in/f32-accumulate
// (see call_rocket_dynamic_conv2d in the transform spec), while models
// commonly arrive as plain f32 from ONNX/torch import.
//
// This replaces `iree-global-opt-demote-contraction-inputs
// {type=f16 operation=conv}`, which this spec used to call and which is
// silently wrong for any strided or dilated convolution. That pass rebuilds
// the named op with `linalg::getPrunedAttributeList(namedOp)`
// (DemoteContractionInputs.cpp), and that helper elides
// `op.getAttributeNames()` -- which for a named convolution *includes*
// `strides` and `dilations`. The rebuilt op carries neither, so both silently
// fall back to 1.
//
// On MobileNetV2 that turned the stride-2 stem conv (1x225x225x3 ->
// 1x112x112x48) into a nominal stride-1 conv reading only a 114x114 corner of
// its input. It is a correctness bug on its own -- the IR is wrong before any
// Rocket matcher runs, so the op is miscomputed even when it stays on the CPU
// -- and it also let @match_dynamic_conv2d_3x3 (which requires strides == 1)
// claim a convolution it was never meant to, dispatching it to the NPU with
// the wrong stride. Upstream's test only covers `strides = dense<1>`, where
// the loss is invisible.
//
// The set of ops handled here mirrors the replaced pass's `operation=conv`
// list exactly, so this is a behaviour-preserving swap apart from the
// attributes: matmuls are not demoted (they stay on the CPU untouched), and
// neither are the depthwise convs.
//
// Depthwise was tried and reverted 2026-09-01. Demoting it does let three of
// MobileNetV2's stride-2 depthwise convolutions match (18 -> 21 offloaded
// dispatch sites), but the resulting model is *wrong*: max|err| 3.5 on the
// logits with top-1 incorrect on every input measured, against 0.36 for the
// same f16 demotion run entirely on the CPU. It is not the convolutions --
// each of the three is exact to f16 epsilon in isolation, with and without a
// tensor.pad producer, and two of them sharing one dynamic executable is
// exact too. Bisecting by channel bound, offloading the 144-channel one
// alone is fine and adding the 192-channel one breaks it, so it is an
// interaction inside the full model that none of those isolations reproduce.
// See also the HAL's own depthwise gaps, which predate this.
//
// Anything left alone is safe: an op that stays f32 fails the matchers' f16
// typing and goes to the CPU, and RocketPromoteUnclaimedConvInputsPass gives
// f32 back to anything demoted that the match loop then declines.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

#include <array>

namespace mlir::iree_compiler::IREE::HAL {
namespace {

// The named-op attributes `getPrunedAttributeList` drops on the floor. Both
// are optional on the op, so a missing one means "already the default" and is
// simply not carried over.
constexpr std::array<StringRef, 2> kShapeDefiningAttrNames = {"strides",
                                                              "dilations"};

// Marks what this pass rewrote, so RocketPromoteUnclaimedConvInputsPass can
// undo exactly its own work on the convolutions the matchers then decline --
// and nothing else. A convolution a model authored in f16 itself, or any
// truncf a model wrote by hand, carries no tag and is left alone.
constexpr StringLiteral kDemotedAttrName = "rocket.f16_demoted";

// Elementwise truncf of a whole tensor, as a linalg.generic -- the same shape
// of rewrite the upstream pass emits, so dispatch formation folds it into the
// producer exactly as before.
Value truncateToF16(PatternRewriter &rewriter, Location loc, Value input) {
  auto inputType = cast<RankedTensorType>(input.getType());
  Type f16 = rewriter.getF16Type();
  auto resultType =
      RankedTensorType::get(inputType.getShape(), f16, inputType.getEncoding());
  SmallVector<AffineMap> maps(
      2, rewriter.getMultiDimIdentityMap(inputType.getRank()));
  SmallVector<utils::IteratorType> iteratorTypes(inputType.getRank(),
                                                 utils::IteratorType::parallel);
  Value empty = tensor::EmptyOp::create(
      rewriter, loc, tensor::getMixedSizes(rewriter, loc, input), f16);
  return linalg::GenericOp::create(
             rewriter, loc, TypeRange{resultType}, ValueRange{input},
             ValueRange{empty}, maps, iteratorTypes,
             [&](OpBuilder &b, Location loc, ValueRange args) {
               Value truncated = arith::TruncFOp::create(b, loc, f16, args[0]);
               linalg::YieldOp::create(b, loc, truncated);
             })
      ->getResult(0);
}

// Tags an op this pass created or rewrote. See kDemotedAttrName.
void markDemoted(Operation *op, PatternRewriter &rewriter) {
  op->setAttr(kDemotedAttrName, rewriter.getUnitAttr());
}

template <typename ConvOpTy>
struct DemoteConvInputsToF16 : OpRewritePattern<ConvOpTy> {
  using OpRewritePattern<ConvOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvOpTy convOp,
                                PatternRewriter &rewriter) const override {
    // Only all-f32 operand sets, matching the pass this replaces: a conv
    // already authored in f16 (or any mixed-precision one) is left alone.
    if (convOp->hasAttr(kDemotedAttrName)) {
      return failure();
    }
    Type f32 = rewriter.getF32Type();
    if (!llvm::all_of(convOp->getOperands(), [&](Value operand) {
          auto type = dyn_cast<RankedTensorType>(operand.getType());
          return type && type.getElementType() == f32;
        })) {
      return failure();
    }

    // Read the shape-defining attributes off the original op before it is
    // replaced. Named linalg ops keep these as inherent attributes, so this
    // reaches them whether or not they are stored as properties.
    SmallVector<NamedAttribute> attributes =
        linalg::getPrunedAttributeList(convOp);
    for (StringRef name : kShapeDefiningAttrNames) {
      if (Attribute attr = convOp->getAttr(name)) {
        attributes.emplace_back(rewriter.getStringAttr(name), attr);
      }
    }

    Location loc = convOp.getLoc();
    SmallVector<Value> demotedInputs;
    for (OpOperand *inputOperand : convOp.getDpsInputOperands()) {
      Value demoted = truncateToF16(rewriter, loc, inputOperand->get());
      markDemoted(demoted.getDefiningOp(), rewriter);
      demotedInputs.push_back(demoted);
    }
    auto demotedOp = rewriter.replaceOpWithNewOp<ConvOpTy>(
        convOp, demotedInputs, convOp.getDpsInits(), attributes);
    markDemoted(demotedOp, rewriter);
    return success();
  }
};

struct RocketDemoteConvInputsPass
    : public PassWrapper<RocketDemoteConvInputsPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RocketDemoteConvInputsPass)

  StringRef getArgument() const final {
    return "rocket-demote-conv-inputs-to-f16";
  }
  StringRef getDescription() const final {
    return "Demotes all-f32 named 2-D convolution inputs to f16, keeping the "
           "f32 accumulator and preserving strides/dilations.";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() final {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<DemoteConvInputsToF16<linalg::Conv2DOp>,
                 DemoteConvInputsToF16<linalg::Conv2DNchwFchwOp>,
                 DemoteConvInputsToF16<linalg::Conv2DNhwcHwcfOp>,
                 DemoteConvInputsToF16<linalg::Conv2DNhwcFhwcOp>,
                 DemoteConvInputsToF16<linalg::Conv2DNgchwFgchwOp>,
                 DemoteConvInputsToF16<linalg::Conv2DNgchwGfchwOp>>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

static PassRegistration<RocketDemoteConvInputsPass> reg;

} // namespace
} // namespace mlir::iree_compiler::IREE::HAL
