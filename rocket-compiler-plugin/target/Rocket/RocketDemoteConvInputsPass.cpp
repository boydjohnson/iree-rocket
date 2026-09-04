// Copyright 2026
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Demotes the input operands of an all-f32 named 2-D convolution or matmul to
// f16, leaving the accumulator at f32 -- Rocket's ABI is f16-in/f32-accumulate
// (see call_rocket_dynamic_conv2d in the transform spec), while models
// commonly arrive as plain f32 from ONNX/torch import.
//
// The registered name still says "conv" because it is what the spec, the
// tests and the docs all reference; the matmul case was added later and the
// mechanism is identical.
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
// The convolutions handled here mirror the replaced pass's `operation=conv`
// list exactly, so that part is a behaviour-preserving swap apart from the
// attributes. Depthwise convs are still not demoted (see below).
//
// `linalg.matmul` was added 2026-09-04, and not for matching -- the spec's
// matcher already claimed f32 matmuls, because @call_rocket_matmul narrowed
// both operands to f16 itself. That narrowing is the problem: it lives inside
// a `util.func` that is never inlined (every dispatch it forms is named
// `call_rocket_matmul_dispatch_N`), so the constant weights are invisible to
// const-expr hoisting and the truncf runs as a CPU dispatch on **every
// inference** -- 1.79M elements for MobileNetV2's classifier, into a fresh
// transient buffer, which then defeats the runtime's packed-coefficient cache
// as well (all misses, all `miss (new)`; see ISSUES.md P6). Demoting here
// instead puts the truncf in the caller, next to the constant, where
// hoist-into-globals and const-eval fold it into an initializer exactly as
// they already do for every convolution's weights.
//
// `indexing_maps` and `cast` join `strides`/`dilations` in the carried-over
// list for the same reason those two are there: `getPrunedAttributeList`
// elides every inherent attribute name, and for `linalg.matmul` the indexing
// maps are precisely what distinguishes a plain matmul from a transposed or
// broadcasting one. Dropping them would rebuild a transposed matmul as an
// untransposed one -- the same silent miscompile the strides bug was.
//
// Depthwise was tried and reverted 2026-09-01, and the reason has since been
// narrowed twice. Demoting it does let three of MobileNetV2 **static-int8**'s
// stride-2 depthwise convolutions match (18 -> 21 offloaded dispatch sites),
// and that model is then *wrong*: max|err| 3.5 on the logits with top-1
// incorrect on every input measured, against 0.36 for the same f16 demotion
// run entirely on the CPU. It is not the convolutions -- each of the three is
// exact to f16 epsilon in isolation, with and without a tensor.pad producer,
// and two of them sharing one dynamic executable is exact too. Bisecting by
// channel bound, offloading the 144-channel one alone is fine and adding the
// 192-channel one breaks it. That is a command buffer mixing fp16 depthwise
// with int8 dispatches, which is ISSUES.md C8, not a property of depthwise.
//
// On the plain fp16 model it is correct: 44 sites against 37, max|err| 0.0500
// vs 0.0192 on a CPU f32 reference, top-1 and top-5 stable, byte-identical
// over five consecutive runs (measured 2026-09-04). It is simply *slower* --
// 186 ms against 148 -- because a depthwise convolution is the cheapest op in
// the model per byte moved and loses to the per-dispatch layout round trip.
// ISSUES.md P7 has the full accounting and what would change it. So this is
// still the right default, but for a performance reason on one model and a
// correctness reason on the other; do not read it as "depthwise is broken".
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

// The named-op attributes `getPrunedAttributeList` drops on the floor. Each is
// optional on the op, so a missing one means "already the default" and is
// simply not carried over -- which is also why naming an attribute an op does
// not have (`strides` on a matmul, `indexing_maps` on a convolution) costs
// nothing.
constexpr std::array<StringRef, 4> kShapeDefiningAttrNames = {
    "strides", "dilations", "indexing_maps", "cast"};

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

template <typename ContractionOpTy>
struct DemoteInputsToF16 : OpRewritePattern<ContractionOpTy> {
  using OpRewritePattern<ContractionOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(ContractionOpTy convOp,
                                PatternRewriter &rewriter) const override {
    // Only all-f32 operand sets, matching the pass this replaces: an op
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
    auto demotedOp = rewriter.replaceOpWithNewOp<ContractionOpTy>(
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
    return "Demotes all-f32 named 2-D convolution and matmul inputs to f16, "
           "keeping the f32 accumulator and preserving the shape-defining "
           "attributes getPrunedAttributeList elides.";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() final {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<DemoteInputsToF16<linalg::Conv2DOp>,
                 DemoteInputsToF16<linalg::Conv2DNchwFchwOp>,
                 DemoteInputsToF16<linalg::Conv2DNhwcHwcfOp>,
                 DemoteInputsToF16<linalg::Conv2DNhwcFhwcOp>,
                 DemoteInputsToF16<linalg::Conv2DNgchwFgchwOp>,
                 DemoteInputsToF16<linalg::Conv2DNgchwGfchwOp>,
                 DemoteInputsToF16<linalg::MatmulOp>>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

static PassRegistration<RocketDemoteConvInputsPass> reg;

} // namespace
} // namespace mlir::iree_compiler::IREE::HAL
