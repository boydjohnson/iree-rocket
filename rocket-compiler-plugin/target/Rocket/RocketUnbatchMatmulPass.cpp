// Copyright 2026
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Splits a `linalg.batch_matmul` with a static batch into one
// `linalg.matmul` per batch element, so the existing matmul matcher and
// lowering claim them:
//
//   [B,M,K] x [B,K,N] -> [B,M,N]   becomes  B x ([M,K] x [K,N] -> [M,N])
//
// with a `tensor.extract_slice` per operand and a `tensor.insert_slice` per
// result. ISSUES.md C15: this is ViT's attention core, `Q K^T` and `attn V`,
// twenty-four dispatch sites on a twelve-layer model that never reach the
// planner at all -- `readRocketCandidate` reads row-major `linalg.matmul`,
// and a batched contraction is not one.
//
// Why unbatch rather than teach the descriptor a batch. Each batch element
// has its *own* right-hand operand, and a convolution shares one coefficient
// set across every pixel it programs -- so there is no mapping of a real
// batch onto the CNA short of a block-diagonal weight matrix, which would
// waste B-fold of the MAC array and materialise a B-times-larger operand
// every inference. B independent matmuls is what the hardware can actually
// run. The spec already folds a *unit* batch away for the same reason, and
// `rocket-expand-gemv-to-matmul` plays the identical trick in the other
// direction.
//
// **This is off by default, and the measurement is why.** It multiplies
// dispatch count by the batch: ViT-B/16's twenty-four sites become
// two hundred and eighty-eight. ISSUES.md P8 measured the offload's cost as a
// flat per-dispatch tax, and on ViT the `record` phase alone is 1.6 ms per
// dispatch -- so this trades a slice of the CPU's attention time against
// something like 460 ms of new host work. The transform spec carries the
// pass line commented out behind `//@ROCKET_BATCH_MATMUL@`, and
// `rocket-compiler --batch-matmul` uncomments it, exactly as `--elementwise`
// gates the element-wise matchers. Turn it on, measure against
// `--no-offload`, and let the number decide.
//
// Both operands here are activations, unlike every matmul this backend
// offloads today: attention multiplies two things the model just computed.
// The weight-cache miss that implies is per inference rather than per
// process, which is part of what makes the trade above unattractive and is
// worth remembering before reading a benchmark of it.
//
// Runs before `rocket-demote-conv-inputs-to-f16`, for the reason the GEMV
// pass documents: the matmuls this produces are all-f32, and the demotion is
// what makes them matchable.

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::iree_compiler::IREE::HAL {
namespace {

/// Batches above this are left alone. A large batch would turn one operation
/// into hundreds of dispatches and hundreds of slice pairs, and the IR growth
/// is paid at compile time whether or not the result is ever faster. ViT-B/16
/// has twelve heads and ViT-L/16 sixteen; a model wanting more than this
/// should be measured before it is admitted, not discovered in a compile that
/// stopped terminating.
constexpr int64_t kMaxUnbatchedBatch = 64;

/// Drops the leading batch dimension of a rank-3 tensor at `index`.
static Value sliceBatch(PatternRewriter &rewriter, Location loc, Value value,
                        int64_t index) {
  auto sourceType = cast<RankedTensorType>(value.getType());
  ArrayRef<int64_t> shape = sourceType.getShape();
  SmallVector<OpFoldResult, 3> offsets = {rewriter.getIndexAttr(index),
                                          rewriter.getIndexAttr(0),
                                          rewriter.getIndexAttr(0)};
  SmallVector<OpFoldResult, 3> sizes = {rewriter.getIndexAttr(1),
                                        rewriter.getIndexAttr(shape[1]),
                                        rewriter.getIndexAttr(shape[2])};
  SmallVector<OpFoldResult, 3> strides(3, rewriter.getIndexAttr(1));
  // The rank-reducing result type: [1, M, K] read back as [M, K].
  auto resultType = RankedTensorType::get({shape[1], shape[2]},
                                          sourceType.getElementType());
  return tensor::ExtractSliceOp::create(rewriter, loc, resultType, value,
                                        offsets, sizes, strides);
}

struct UnbatchMatmul : OpRewritePattern<linalg::BatchMatmulOp> {
  using OpRewritePattern<linalg::BatchMatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::BatchMatmulOp batchOp,
                                PatternRewriter &rewriter) const override {
    if (!batchOp.hasPureTensorSemantics()) {
      return rewriter.notifyMatchFailure(batchOp, "not on tensors");
    }
    SmallVector<Value> inputs = batchOp.getDpsInputs();
    SmallVector<Value> inits = batchOp.getDpsInits();
    if (inputs.size() != 2 || inits.size() != 1) {
      return rewriter.notifyMatchFailure(batchOp, "unexpected operand count");
    }
    // A user-defined indexing map can spell a transposed or otherwise
    // non-standard contraction, which the slices below would silently
    // reinterpret. Only the default maps are unbatched.
    if (batchOp.hasUserDefinedMaps()) {
      return rewriter.notifyMatchFailure(batchOp, "user-defined indexing maps");
    }

    auto initType = dyn_cast<RankedTensorType>(inits[0].getType());
    if (!initType || initType.getRank() != 3) {
      return rewriter.notifyMatchFailure(batchOp, "accumulator is not rank 3");
    }
    for (Value input : inputs) {
      auto type = dyn_cast<RankedTensorType>(input.getType());
      if (!type || type.getRank() != 3) {
        return rewriter.notifyMatchFailure(batchOp, "operand is not rank 3");
      }
      // Every extent has to be static: the slices are built from constant
      // offsets and sizes, and the batch decides how many ops to emit.
      if (!type.hasStaticShape()) {
        return rewriter.notifyMatchFailure(batchOp, "dynamic operand extents");
      }
    }
    if (!initType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(batchOp, "dynamic accumulator extents");
    }

    int64_t batch = initType.getDimSize(0);
    if (batch != cast<RankedTensorType>(inputs[0].getType()).getDimSize(0) ||
        batch != cast<RankedTensorType>(inputs[1].getType()).getDimSize(0)) {
      return rewriter.notifyMatchFailure(batchOp, "operand batches disagree");
    }
    // A unit batch is the transform spec's own job -- it collapses one into a
    // plain matmul -- and doing it here as a slice pair would be strictly
    // worse than the reshape it uses.
    if (batch <= 1) {
      return rewriter.notifyMatchFailure(batchOp, "unit or empty batch");
    }
    if (batch > kMaxUnbatchedBatch) {
      return rewriter.notifyMatchFailure(batchOp, "batch above the bound");
    }

    Location loc = batchOp.getLoc();
    Value result = inits[0];
    SmallVector<OpFoldResult, 3> strides(3, rewriter.getIndexAttr(1));
    for (int64_t index = 0; index < batch; ++index) {
      Value lhs = sliceBatch(rewriter, loc, inputs[0], index);
      Value rhs = sliceBatch(rewriter, loc, inputs[1], index);
      Value init = sliceBatch(rewriter, loc, inits[0], index);
      auto matmulOp = linalg::MatmulOp::create(
          rewriter, loc, TypeRange{init.getType()}, ValueRange{lhs, rhs},
          ValueRange{init});
      // Written back into the running result, not into the original
      // accumulator: each insert consumes the previous one, so the B writes
      // chain into a single value with no aliasing between them.
      SmallVector<OpFoldResult, 3> offsets = {rewriter.getIndexAttr(index),
                                              rewriter.getIndexAttr(0),
                                              rewriter.getIndexAttr(0)};
      SmallVector<OpFoldResult, 3> sizes = {
          rewriter.getIndexAttr(1), rewriter.getIndexAttr(initType.getDimSize(1)),
          rewriter.getIndexAttr(initType.getDimSize(2))};
      result = tensor::InsertSliceOp::create(rewriter, loc,
                                             matmulOp.getResult(0), result,
                                             offsets, sizes, strides);
    }
    rewriter.replaceOp(batchOp, result);
    return success();
  }
};

struct RocketUnbatchMatmulPass
    : public PassWrapper<RocketUnbatchMatmulPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RocketUnbatchMatmulPass)

  StringRef getArgument() const final { return "rocket-unbatch-matmul"; }
  StringRef getDescription() const final {
    return "Splits a static-batch linalg.batch_matmul into one linalg.matmul "
           "per batch element, so the existing matmul path claims them "
           "(ISSUES.md C15). Off by default; see the file comment.";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect>();
  }

  void runOnOperation() final {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<UnbatchMatmul>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

static PassRegistration<RocketUnbatchMatmulPass> reg;

} // namespace
} // namespace mlir::iree_compiler::IREE::HAL
