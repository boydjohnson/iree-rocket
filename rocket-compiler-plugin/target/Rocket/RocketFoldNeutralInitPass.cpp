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
FloatAttr fillConstant(Value value) {
  while (Operation *def = value.getDefiningOp()) {
    if (auto fill = dyn_cast<linalg::FillOp>(def)) {
      auto constant = fill.getInputs()[0].getDefiningOp<arith::ConstantOp>();
      return constant ? dyn_cast<FloatAttr>(constant.getValue()) : nullptr;
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
bool isNeutral(Operation *op, const APFloat &init) {
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
    FloatAttr constant =
        fillConstant(generic.getDpsInputOperand(initArg.getArgNumber())->get());
    if (!constant || !isNeutral(combine, constant.getValue())) {
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

struct RocketFoldNeutralInitPass
    : public PassWrapper<RocketFoldNeutralInitPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RocketFoldNeutralInitPass)

  StringRef getArgument() const final { return "rocket-fold-neutral-init"; }
  StringRef getDescription() const final {
    return "Drops a neutral accumulator initialiser (fill 0 under add, -inf "
           "under max, +inf under min) from a widening generic, leaving the "
           "bare extf the consumer's truncf can cancel.";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldNeutralInit>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

static PassRegistration<RocketFoldNeutralInitPass> reg;

} // namespace
} // namespace mlir::iree_compiler::IREE::HAL
