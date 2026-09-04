// Copyright 2026
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Puts back the f32 inputs that RocketDemoteConvInputsPass took away from
// convolutions and matmuls the match loop then declined to claim.
//
// The demotion has to run *before* matching, because the matchers require
// f16/f16/f32 typing -- but it cannot know which convolutions will be
// claimed, and deciding that up front would mean re-implementing the
// matchers' eligibility predicates in C++ and keeping the two in sync
// forever. So the spec demotes every all-f32 named convolution and matmul,
// matches, and then this pass undoes the demotion wherever the NPU did not
// take the op. Anything still holding a `linalg.conv_2d_*` or `linalg.matmul`
// after `foreach_match` is by definition unclaimed: a claimed one was erased
// along with the region `transform.iree.cast_and_call` replaced.
//
// Without this, an unclaimed convolution runs on the CPU in f16 when f32 was
// available and free. On MobileNetV2 that is the stride-2 stem: measured at
// 0.349 max|err| on the final logits against a plain f32 CPU build, enough to
// move top-1 on a near-tie. It is pure loss -- the demotion buys nothing for
// an op that never reaches the NPU.
//
// Only this project's own demotion is reverted, never a model's. The demote
// pass tags both the convolution and the truncf it inserts with
// `rocket.f16_demoted`, and an input is restored only when it traces to one
// of those tags -- so a convolution a model authored in f16, or a truncf it
// wrote by hand, is left exactly as it was.
//
// Runs in the transform spec immediately after the match/rewrite loop, while
// the truncf still sits next to the convolution and everything is linalg on
// tensors. It cannot run later: by the `flow` phase the truncf and the
// convolution are separate dispatches with an f16 tensor crossing between
// them and f16 weights already hoisted to constants, so undoing it there
// would mean rewriting two executables and re-folding a constant.

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

#include <array>

namespace mlir::iree_compiler::IREE::HAL {
namespace {

constexpr StringLiteral kDemotedAttrName = "rocket.f16_demoted";

// Kept identical to RocketDemoteConvInputsPass's list: these are the named-op
// attributes `getPrunedAttributeList` drops, and losing them here would
// silently change the operation exactly the way the upstream demotion pass
// used to -- `strides`/`dilations` for a convolution, `indexing_maps`/`cast`
// for a matmul.
constexpr std::array<StringRef, 4> kShapeDefiningAttrNames = {"strides",
                                                              "dilations",
                                                              "indexing_maps",
                                                              "cast"};

// The f32 value `demoted` was truncated from, when it is one of this
// project's own inserted truncf generics. Nullptr otherwise.
Value originalF32Source(Value demoted) {
  auto genericOp = demoted.getDefiningOp<linalg::GenericOp>();
  if (!genericOp || !genericOp->hasAttr(kDemotedAttrName)) {
    return {};
  }
  if (genericOp.getDpsInputs().size() != 1) {
    return {};
  }
  Value source = genericOp.getDpsInputs()[0];
  auto sourceType = dyn_cast<RankedTensorType>(source.getType());
  if (!sourceType || !sourceType.getElementType().isF32()) {
    return {};
  }
  return source;
}

template <typename ContractionOpTy>
struct PromoteInputsToF32 : OpRewritePattern<ContractionOpTy> {
  using OpRewritePattern<ContractionOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(ContractionOpTy convOp,
                                PatternRewriter &rewriter) const override {
    if (!convOp->hasAttr(kDemotedAttrName)) {
      return failure();
    }

    SmallVector<Value> promotedInputs;
    for (OpOperand *inputOperand : convOp.getDpsInputOperands()) {
      Value source = originalF32Source(inputOperand->get());
      if (!source) {
        // A demoted input whose truncf is gone or was rewritten. Leave the
        // whole operation alone rather than promote it halfway.
        return failure();
      }
      promotedInputs.push_back(source);
    }

    SmallVector<NamedAttribute> attributes =
        linalg::getPrunedAttributeList(convOp);
    for (StringRef name : kShapeDefiningAttrNames) {
      if (Attribute attr = convOp->getAttr(name)) {
        attributes.emplace_back(rewriter.getStringAttr(name), attr);
      }
    }

    auto promotedOp = rewriter.replaceOpWithNewOp<ContractionOpTy>(
        convOp, promotedInputs, convOp.getDpsInits(), attributes);
    // The tag is this pass's own work list; a promoted op is done.
    promotedOp->removeAttr(kDemotedAttrName);
    return success();
  }
};

struct RocketPromoteUnclaimedConvInputsPass
    : public PassWrapper<RocketPromoteUnclaimedConvInputsPass,
                         OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      RocketPromoteUnclaimedConvInputsPass)

  StringRef getArgument() const final {
    return "rocket-promote-unclaimed-conv-inputs";
  }
  StringRef getDescription() const final {
    return "Restores the f32 inputs of convolutions and matmuls that were "
           "demoted to f16 for matching but not claimed by the Rocket match "
           "loop.";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() final {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<PromoteInputsToF32<linalg::Conv2DOp>,
                 PromoteInputsToF32<linalg::Conv2DNchwFchwOp>,
                 PromoteInputsToF32<linalg::Conv2DNhwcHwcfOp>,
                 PromoteInputsToF32<linalg::Conv2DNhwcFhwcOp>,
                 PromoteInputsToF32<linalg::Conv2DNgchwFgchwOp>,
                 PromoteInputsToF32<linalg::Conv2DNgchwGfchwOp>,
                 PromoteInputsToF32<linalg::MatmulOp>>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

static PassRegistration<RocketPromoteUnclaimedConvInputsPass> reg;

} // namespace
} // namespace mlir::iree_compiler::IREE::HAL
