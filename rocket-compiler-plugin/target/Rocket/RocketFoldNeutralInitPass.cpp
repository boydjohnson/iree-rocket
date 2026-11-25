// Copyright 2026
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Drops the accumulator-initialiser term from a Rocket shim's widen when the
// initialiser is the operation's neutral element, so the widen is a bare
// `extf` that IREE cancels against the consumer's `truncf`.
//
// linalg defines a convolution, matmul or pool as an update of its `outs`
// operand -- `O += conv(I, W)`, `O = max(O, I)` -- so a shim that returns the
// hardware's f16 result has to fold that initialiser in on the host to be
// exact for *any* init. It does so with a `linalg.generic` whose body is
// `OP(extf(raw), init)`. Every real model seeds the init with the neutral
// element (`linalg.fill 0.0` under a convolution or matmul, `fill -inf`
// under a max pool), and then the term is an identity -- but not one IREE
// can see: the fill lowers to a `flow.tensor.splat` operand of a dispatch,
// never to a scalar inside the generic, so `x + 0.0` is never simplified and
// the widen stays a CPU dispatch. That dispatch is what stands between two
// Rocket dispatches on every plain-conv, pool and matmul edge, and why those
// edges could neither chain nor elide their compaction (ISSUES.md P2).
//
// This pass proves the init neutral at the linalg level, where the fill and
// its constant are still visible, and rewrites the generic to `extf(raw)`
// alone. After that the consumer's narrow fuses with it and `truncf(extf(x))`
// folds to `x`, leaving the two Rocket dispatches adjacent -- the form the
// fused (bias/ReLU) shims already had. Anything not provably neutral is left
// exactly as it was.
//
// Runs after the match loop, over the whole module, next to
// rocket-promote-unclaimed-conv-inputs.
//
// It also does that last cancellation itself. Until IREE 3.12 arith folded
// `truncf(extf(x))` to `x` for f16 -> f32 -> f16; the LLVM in IREE 3.14 keeps
// the pair unless the ops carry `nnan` (APFloat's isLosslesslyConvertibleTo
// now refuses any IEEE -> IEEE conversion because converting a signaling NaN
// quiets it). Without the fold every widen/narrow pair became its own CPU
// dispatch between two Rocket dispatches -- MobileNetV2 fp16 went from 5 to
// 55 CPU dispatch sites and 9.6 to 33.7 ms. CancelWidenNarrow restores the
// old result for exactly this shape: a narrowing generic over a widening
// generic, optionally through one reshape, back to the source type. The
// only value it changes is a signaling NaN, which keeps its payload instead
// of coming back quiet; that is what every IREE before 3.14 did too.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::HAL {
namespace {

// The constant a `linalg.fill` seeds `value` with, looking through the
// metadata ops a shim puts between them (a cast, a reshape, the NCHW pool's
// transpose of its init), or null.
// The fill value, float or integer: the int8 max pool's shim combines with
// a fill of the type's minimum under `maxsi` the way the fp16 one does with
// -inf under `maximumf`.
TypedAttr fillConstant(Value value) {
  while (Operation *def = value.getDefiningOp()) {
    if (auto fill = dyn_cast<linalg::FillOp>(def)) {
      auto constant = fill.getInputs()[0].getDefiningOp<arith::ConstantOp>();
      if (!constant) {
        return nullptr;
      }
      Attribute attr = constant.getValue();
      if (isa<FloatAttr, IntegerAttr>(attr)) {
        return cast<TypedAttr>(attr);
      }
      return nullptr;
    }
    if (auto transpose = dyn_cast<linalg::TransposeOp>(def)) {
      value = transpose.getInput();
    } else if (auto cast = dyn_cast<tensor::CastOp>(def)) {
      value = cast.getSource();
    } else if (auto expand = dyn_cast<tensor::ExpandShapeOp>(def)) {
      value = expand.getSrc();
    } else if (auto collapse = dyn_cast<tensor::CollapseShapeOp>(def)) {
      value = collapse.getSrc();
    } else {
      return nullptr;
    }
  }
  return nullptr;
}

// Whether `init` is the neutral element of `op`.
bool isNeutral(Operation *op, TypedAttr attr) {
  if (auto floatAttr = dyn_cast<FloatAttr>(attr)) {
    const APFloat &init = floatAttr.getValue();
    if (isa<arith::AddFOp>(op)) {
      return init.isZero();
    }
    if (isa<arith::MaximumFOp>(op)) {
      return init.isInfinity() && init.isNegative();
    }
    if (isa<arith::MinimumFOp>(op)) {
      return init.isInfinity() && !init.isNegative();
    }
    return false;
  }
  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    const APInt &init = intAttr.getValue();
    if (isa<arith::AddIOp>(op)) {
      return init.isZero();
    }
    if (isa<arith::MaxSIOp>(op)) {
      return init.isMinSignedValue();
    }
    if (isa<arith::MinSIOp>(op)) {
      return init.isMaxSignedValue();
    }
  }
  return false;
}

struct FoldNeutralInit : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp generic,
                                PatternRewriter &rewriter) const override {
    if (generic.getNumDpsInputs() != 2 || generic.getNumDpsInits() != 1 ||
        generic.getNumResults() != 1) {
      return failure();
    }
    if (!llvm::all_of(generic.getIndexingMapsArray(),
                      [](AffineMap map) { return map.isIdentity(); }) ||
        !llvm::all_of(generic.getIteratorTypesArray(), [](utils::IteratorType type) {
          return type == utils::IteratorType::parallel;
        })) {
      return failure();
    }
    Block &body = generic.getRegion().front();
    auto yield = cast<linalg::YieldOp>(body.getTerminator());
    Operation *combine = yield.getOperand(0).getDefiningOp();
    if (!combine || combine->getBlock() != &body ||
        combine->getNumOperands() != 2) {
      return failure();
    }
    // Which input plays the initialiser: the block argument the combining
    // op consumes directly and nothing else in the body touches.
    BlockArgument initArg;
    Value kept;
    for (unsigned index : {0u, 1u}) {
      auto arg = dyn_cast<BlockArgument>(combine->getOperand(index));
      if (arg && arg.getOwner() == &body && arg.getArgNumber() < 2 &&
          arg.hasOneUse()) {
        initArg = arg;
        kept = combine->getOperand(1 - index);
        break;
      }
    }
    if (!initArg) {
      return failure();
    }
    TypedAttr constant =
        fillConstant(generic.getDpsInputOperand(initArg.getArgNumber())->get());
    if (!constant || !isNeutral(combine, constant)) {
      return failure();
    }
    // The other input, and its map, survive; the body is cloned without the
    // combining op, yielding what it combined with the init.
    unsigned keptIndex = 1 - initArg.getArgNumber();
    Value keptInput = generic.getDpsInputOperand(keptIndex)->get();
    SmallVector<AffineMap> maps = {generic.getIndexingMapsArray()[keptIndex],
                                   generic.getIndexingMapsArray()[2]};
    auto replacement = linalg::GenericOp::create(
        rewriter, generic.getLoc(), generic.getResultTypes(),
        ValueRange{keptInput}, generic.getDpsInits(), maps,
        generic.getIteratorTypesArray(),
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          IRMapping mapping;
          mapping.map(body.getArgument(keptIndex), args[0]);
          mapping.map(body.getArgument(2), args[1]);
          for (Operation &op : body.without_terminator()) {
            if (&op == combine) {
              continue;
            }
            builder.clone(op, mapping);
          }
          linalg::YieldOp::create(builder, loc, mapping.lookupOrDefault(kept));
        });
    rewriter.replaceOp(generic, replacement.getResults());
    return success();
  }
};

// The one-input, one-init, identity-map, all-parallel generic whose body is a
// single `OpTy` of the block argument, or null.
template <typename OpTy>
OpTy singleCastBody(linalg::GenericOp generic) {
  if (!generic || generic.getNumDpsInputs() != 1 ||
      generic.getNumDpsInits() != 1 || generic.getNumResults() != 1) {
    return nullptr;
  }
  if (!llvm::all_of(generic.getIndexingMapsArray(),
                    [](AffineMap map) { return map.isIdentity(); }) ||
      !llvm::all_of(generic.getIteratorTypesArray(), [](utils::IteratorType type) {
        return type == utils::IteratorType::parallel;
      })) {
    return nullptr;
  }
  Block &body = generic.getRegion().front();
  if (body.getOperations().size() != 2) {
    return nullptr;
  }
  auto cast = dyn_cast<OpTy>(&body.front());
  auto yield = cast_or_null<linalg::YieldOp>(body.getTerminator());
  if (!cast || cast->getOperand(0) != body.getArgument(0) ||
      yield.getOperand(0) != cast->getResult(0)) {
    return nullptr;
  }
  return cast;
}

// `narrow(reshape?(widen(x)))` -> `reshape?(x)` when the narrow lands back on
// x's element type. See the file comment for why this is no longer left to
// arith's folder.
struct CancelWidenNarrow : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp narrow,
                                PatternRewriter &rewriter) const override {
    auto truncf = singleCastBody<arith::TruncFOp>(narrow);
    if (!truncf) {
      return failure();
    }
    Value input = narrow.getDpsInputOperand(0)->get();
    Operation *reshape = input.getDefiningOp();
    if (isa_and_nonnull<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(reshape)) {
      input = reshape->getOperand(0);
    } else {
      reshape = nullptr;
    }
    auto widen = input.getDefiningOp<linalg::GenericOp>();
    auto extf = singleCastBody<arith::ExtFOp>(widen);
    if (!extf) {
      return failure();
    }
    Value source = widen.getDpsInputOperand(0)->get();
    auto resultType = cast<RankedTensorType>(narrow.getResult(0).getType());
    auto sourceType = dyn_cast<RankedTensorType>(source.getType());
    if (!sourceType ||
        sourceType.getElementType() != resultType.getElementType()) {
      return failure();
    }
    Value replacement;
    if (!reshape) {
      if (sourceType != resultType) {
        return failure();
      }
      replacement = source;
    } else if (auto expand = dyn_cast<tensor::ExpandShapeOp>(reshape)) {
      replacement = tensor::ExpandShapeOp::create(
          rewriter, expand.getLoc(), resultType, source,
          expand.getReassociationIndices(), expand.getMixedOutputShape());
    } else {
      auto collapse = cast<tensor::CollapseShapeOp>(reshape);
      replacement = tensor::CollapseShapeOp::create(
          rewriter, collapse.getLoc(), resultType, source,
          collapse.getReassociationIndices());
    }
    rewriter.replaceOp(narrow, replacement);
    return success();
  }
};

struct RocketFoldNeutralInitPass
    : public PassWrapper<RocketFoldNeutralInitPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RocketFoldNeutralInitPass)

  StringRef getArgument() const final { return "rocket-fold-neutral-init"; }
  StringRef getDescription() const final {
    return "Drops a neutral accumulator initialiser (fill 0 under add, -inf "
           "under max, +inf under min, and the integer minimum/maximum under "
           "signed max/min) from a widening generic, then cancels the bare "
           "extf against the consumer's truncf.";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldNeutralInit, CancelWidenNarrow>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

static PassRegistration<RocketFoldNeutralInitPass> reg;

} // namespace
} // namespace mlir::iree_compiler::IREE::HAL
