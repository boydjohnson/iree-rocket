// Copyright 2026
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Raises `linalg.matvec` and `linalg.vecmat` into `linalg.matmul` by giving
// their vector operand the unit dimension it is missing.
//
//   matvec  A[m,k] * y[k]   -> z[m]      becomes  A[m,k] * y[k,1] -> z[m,1]
//   vecmat  y[k]   * A[k,n] -> z[n]      becomes  y[1,k] * A[k,n] -> z[1,n]
//
// with a `tensor.expand_shape` on the vector operand and the accumulator, and
// a `tensor.collapse_shape` putting the result back to rank one. Both are
// pure metadata on a contiguous tensor, so this costs nothing at runtime and
// the reshapes fold away against their neighbours.
//
// Why raise rather than match. A GEMV is a matmul with one extent pinned to
// 1, and everything downstream of this point already handles a matmul with a
// unit extent: `rocket-demote-conv-inputs-to-f16` demotes it,
// `@match_rocket_matmul` claims it (its `dim_bounds` start at `umin = 1`),
// `@call_rocket_matmul` lowers it, and `#rocket_matmul_target` serializes it.
// Writing three more matchers and three more shims would duplicate all of
// that to express something the existing path already says. It is also the
// trick the transform spec already plays on a unit-batch `linalg.batch_matmul`
// -- fold the degenerate dimension away and let the matmul matcher see what
// is really there -- run in the opposite direction.
//
// `linalg.dot` is deliberately NOT handled. It reduces two vectors to a
// scalar, so offloading it would spend a dispatch, a weight pack and an
// output compaction to produce one number that a CPU computes in a few
// hundred multiply-adds. That is the same reasoning that keeps a 1x1
// stride-1 pool off the NPU: programmable, and not worth programming.
//
// Runs before `rocket-demote-conv-inputs-to-f16`, because the matmul this
// produces is all-f32 and the demotion is what makes it matchable. Running it
// after would leave the raised op in f32 and the matcher would decline.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::iree_compiler::IREE::HAL {
namespace {

/// Expands a rank-1 tensor to rank 2, putting the existing extent at
/// `vectorDim` and a unit extent at the other position.
///
/// The reassociation is `[[0, 1]]` in both directions: one group covering
/// both result dimensions, because exactly one of them is the real extent and
/// the other is the unit one being introduced.
static Value expandToRank2(PatternRewriter &rewriter, Location loc, Value value,
                           unsigned vectorDim) {
  auto sourceType = cast<RankedTensorType>(value.getType());
  assert(sourceType.getRank() == 1 && "expected a rank-1 operand");
  SmallVector<int64_t, 2> shape(2, 1);
  shape[vectorDim] = sourceType.getDimSize(0);
  auto resultType =
      RankedTensorType::get(shape, sourceType.getElementType());

  SmallVector<ReassociationIndices, 1> reassociation = {{0, 1}};
  SmallVector<OpFoldResult, 2> outputShape(
      2, rewriter.getIndexAttr(1));
  outputShape[vectorDim] =
      sourceType.isDynamicDim(0)
          ? OpFoldResult(
                tensor::DimOp::create(rewriter, loc, value, 0).getResult())
          : OpFoldResult(rewriter.getIndexAttr(sourceType.getDimSize(0)));
  return tensor::ExpandShapeOp::create(rewriter, loc, resultType, value,
                                      reassociation, outputShape);
}

/// `matvec` and `vecmat` differ only in which operand carries the vector and
/// which dimension of the rank-2 form that vector occupies, so one pattern
/// parameterised by those two facts covers both.
template <typename GemvOpTy, unsigned kVectorOperand, unsigned kVectorDim>
struct ExpandGemvToMatmul : OpRewritePattern<GemvOpTy> {
  using OpRewritePattern<GemvOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(GemvOpTy gemvOp,
                                PatternRewriter &rewriter) const override {
    if (!gemvOp.hasPureTensorSemantics()) {
      return rewriter.notifyMatchFailure(gemvOp, "not on tensors");
    }
    SmallVector<Value> inputs = gemvOp.getDpsInputs();
    SmallVector<Value> inits = gemvOp.getDpsInits();
    if (inputs.size() != 2 || inits.size() != 1) {
      return rewriter.notifyMatchFailure(gemvOp, "unexpected operand count");
    }
    auto initType = dyn_cast<RankedTensorType>(inits[0].getType());
    if (!initType || initType.getRank() != 1) {
      return rewriter.notifyMatchFailure(gemvOp, "accumulator is not rank 1");
    }
    for (Value input : inputs) {
      if (!isa<RankedTensorType>(input.getType())) {
        return rewriter.notifyMatchFailure(gemvOp, "operand is not a tensor");
      }
    }

    Location loc = gemvOp.getLoc();
    // The vector input takes the unit dimension on the side that keeps the
    // contraction dimension adjacent: for matvec the vector is the rhs and
    // becomes [k, 1]; for vecmat it is the lhs and becomes [1, k].
    inputs[kVectorOperand] =
        expandToRank2(rewriter, loc, inputs[kVectorOperand], kVectorDim);
    // The accumulator takes its unit extent on the *same* side as the
    // vector operand did, not the opposite one. matvec pins N: the rhs is
    // [k, 1] and the result [m, 1], both real extent at dim 0. vecmat pins
    // M: the lhs is [1, k] and the result [1, n], both real extent at dim 1.
    Value expandedInit = expandToRank2(rewriter, loc, inits[0], kVectorDim);

    auto matmulOp = linalg::MatmulOp::create(
        rewriter, loc, TypeRange{expandedInit.getType()}, inputs,
        ValueRange{expandedInit});

    // Back to rank one. The result type is the original accumulator's, so a
    // consumer sees exactly the tensor the GEMV would have produced.
    SmallVector<ReassociationIndices, 1> reassociation = {{0, 1}};
    rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
        gemvOp, initType, matmulOp.getResult(0), reassociation);
    return success();
  }
};

struct RocketExpandGemvToMatmulPass
    : public PassWrapper<RocketExpandGemvToMatmulPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RocketExpandGemvToMatmulPass)

  StringRef getArgument() const final {
    return "rocket-expand-gemv-to-matmul";
  }
  StringRef getDescription() const final {
    return "Raises linalg.matvec and linalg.vecmat to linalg.matmul with a "
           "unit extent, so the existing matmul matcher and lowering claim "
           "them. linalg.dot is left alone deliberately.";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() final {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    // matvec: the vector is input 1 and becomes [k, 1]; its accumulator
    // becomes [m, 1].
    patterns.add<ExpandGemvToMatmul<linalg::MatvecOp, 1, 0>>(context);
    // vecmat: the vector is input 0 and becomes [1, k]; its accumulator
    // becomes [1, n].
    patterns.add<ExpandGemvToMatmul<linalg::VecmatOp, 0, 1>>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

static PassRegistration<RocketExpandGemvToMatmulPass> reg;

} // namespace
} // namespace mlir::iree_compiler::IREE::HAL
