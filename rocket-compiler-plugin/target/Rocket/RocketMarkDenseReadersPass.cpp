// Copyright 2026
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Tells each Rocket convolution dispatch how many Rocket dispatches read its
// result, so the runtime can skip writing the dense output buffer when every
// one of them reads the output cube in place instead.
//
// The driver already chains a consumer to its producer's NC1HWC2 output cube
// (ISSUES.md P2 step 2), but it still compacts every result into the dense
// IREE buffer, because nothing on a command buffer can prove that buffer has
// no other reader -- a CPU dispatch, a copy, a later command buffer, the
// module's own output. That fact lives here: once dispatch regions are formed
// and outlined, every reader of a dispatch result is an SSA use, and the set
// is final.
//
// The signal is a count, not a flag, so it stays correct however IREE later
// partitions the program into command buffers. The driver elides the dense
// write only when the number of consumers that chained on the *same* command
// buffer equals the count: a reader on another command buffer leaves the
// tally short, and a reader that could not chain (geometry, a kind that
// records no cube) marks the dense bytes as read. Any reader that is not a
// Rocket dispatch, or any use this pass does not understand, sets the count
// to zero, which is "always write" -- the value every shim passes as a
// literal, so a dispatch this pass never reaches behaves as before.
//
// Runs at the flow phase, after rocket-pin-unclaimed-dispatches; rocket-
// compiler drives it by name the same way. The push constant it rewrites is
// the trailing one every convolution target in the transform spec declares
// with `runtime_dense_readers = true`, positioned after the target's
// runtime_dimensions and runtime_quantization entries -- the order the
// driver consumes them in.

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::HAL {
namespace {

constexpr StringLiteral kRocketBackend = "rocket";
constexpr StringLiteral kDenseReadersKey = "runtime_dense_readers";
constexpr StringLiteral kRuntimeDimensionsKey = "runtime_dimensions";
constexpr StringLiteral kRuntimeQuantizationKey = "runtime_quantization";

// The Rocket executable target a dispatch runs on, or null when the dispatch
// is not a Rocket one (an IREE-formed CPU dispatch targets a flow.executable
// or a non-Rocket HAL variant).
IREE::HAL::ExecutableTargetAttr rocketTarget(IREE::Flow::DispatchOp dispatchOp) {
  IREE::HAL::ExecutableTargetAttr found;
  dispatchOp.forEachEntryPointAttr([&](SymbolRefAttr entryPoint) {
    if (found) {
      return;
    }
    Operation *symbol = SymbolTable::lookupNearestSymbolFrom(dispatchOp, entryPoint);
    auto exportOp = dyn_cast_or_null<IREE::HAL::ExecutableExportOp>(symbol);
    if (!exportOp) {
      return;
    }
    auto variantOp = exportOp->getParentOfType<IREE::HAL::ExecutableVariantOp>();
    if (!variantOp) {
      return;
    }
    IREE::HAL::ExecutableTargetAttr target = variantOp.getTarget();
    if (target && target.getBackend().getValue() == kRocketBackend) {
      found = target;
    }
  });
  return found;
}

size_t arrayLength(DictionaryAttr config, StringRef key) {
  auto array = dyn_cast_or_null<ArrayAttr>(config.get(key));
  return array ? array.size() : 0;
}

// Counts the Rocket dispatches that read `value`, looking through the
// metadata-only ops (reshape, bitcast) that dispatch formation leaves
// between a producer and its consumer. Returns false on any other reader.
bool countRocketReaders(Value value, unsigned &count) {
  for (OpOperand &use : value.getUses()) {
    Operation *owner = use.getOwner();
    if (auto reader = dyn_cast<IREE::Flow::DispatchOp>(owner)) {
      if (!rocketTarget(reader)) {
        return false;
      }
      unsigned index = use.getOperandNumber();
      unsigned first = reader.getWorkload().size();
      unsigned last = first + reader.getArguments().size();
      // A tensor can only be an argument, and a tied argument is one the
      // reader writes in place -- not a read this count may describe.
      if (index < first || index >= last || reader.isOperandTied(index)) {
        return false;
      }
      ++count;
      continue;
    }
    if (isa<IREE::Flow::TensorReshapeOp, IREE::Flow::TensorBitCastOp>(owner)) {
      // Operand 0 is the source; the rest are dynamic dims, which a tensor
      // cannot be.
      if (use.getOperandNumber() != 0 || owner->getNumResults() != 1 ||
          !countRocketReaders(owner->getResult(0), count)) {
        return false;
      }
      continue;
    }
    return false;
  }
  return true;
}

struct RocketMarkDenseReadersPass
    : public PassWrapper<RocketMarkDenseReadersPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RocketMarkDenseReadersPass)

  StringRef getArgument() const final { return "rocket-mark-dense-readers"; }
  StringRef getDescription() const final {
    return "Sets each Rocket convolution dispatch's trailing dense-reader "
           "push constant to the number of Rocket dispatches that read its "
           "result, or zero when any other reader exists.";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect>();
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    unsigned marked = 0;
    unsigned elidable = 0;
    unsigned readersTotal = 0;
    WalkResult walk = module.walk([&](IREE::Flow::DispatchOp dispatchOp) {
      IREE::HAL::ExecutableTargetAttr target = rocketTarget(dispatchOp);
      if (!target) {
        return WalkResult::advance();
      }
      DictionaryAttr config = target.getConfiguration();
      auto declared = dyn_cast_or_null<BoolAttr>(config.get(kDenseReadersKey));
      if (!declared || !declared.getValue()) {
        return WalkResult::advance();
      }
      size_t position = arrayLength(config, kRuntimeDimensionsKey) +
                        arrayLength(config, kRuntimeQuantizationKey);
      OperandRange arguments = dispatchOp.getArguments();
      if (position >= arguments.size() ||
          !arguments[position].getType().isInteger(32)) {
        // The target promised a count the shim did not pass. Refuse rather
        // than rewrite some other operand: the driver would read a shape
        // field as the count and a count as a shape field.
        dispatchOp.emitOpError()
            << "rocket-mark-dense-readers: target declares "
            << kDenseReadersKey << " but argument " << position
            << " is not an i32 push constant";
        return WalkResult::interrupt();
      }
      if (dispatchOp.getNumResults() != 1) {
        return WalkResult::advance();
      }

      unsigned readers = 0;
      if (!countRocketReaders(dispatchOp.getResult(0), readers)) {
        readers = 0;
      }
      OpBuilder builder(dispatchOp);
      Value count = arith::ConstantOp::create(
          builder, dispatchOp.getLoc(),
          builder.getI32IntegerAttr(static_cast<int32_t>(readers)));
      dispatchOp->setOperand(arguments.getBeginOperandIndex() + position, count);
      ++marked;
      if (readers > 0) {
        ++elidable;
        readersTotal += readers;
      }
      return WalkResult::advance();
    });
    if (walk.wasInterrupted()) {
      return signalPassFailure();
    }
    if (marked > 0) {
      module.emitRemark() << "rocket-mark-dense-readers: " << marked
                          << " Rocket dispatch(es), " << elidable
                          << " read only by Rocket dispatches (" << readersTotal
                          << " reader(s))";
    }
  }
};

static PassRegistration<RocketMarkDenseReadersPass> reg;

} // namespace
} // namespace mlir::iree_compiler::IREE::HAL
