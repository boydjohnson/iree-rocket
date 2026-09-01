// Copyright 2026
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Enforces the invariant the rest of this backend depends on: Rocket runs
// only what rocket_conv2d_transform_spec.mlir explicitly put there.
//
// RocketTargetBackend has an empty translation pipeline -- there is no
// codegen for "rocket", only serializeExecutable reading a config dict the
// transform spec stamped onto a hand-authored hal.executable. Every dispatch
// the spec creates therefore carries an explicit
// `stream.affinity = #hal.device.affinity<@rocket_device>`; every dispatch
// IREE forms on its own carries no affinity at all and is placed by
// Stream's AffinityAnalysis.
//
// That analysis propagates through consumers, so an IREE-formed dispatch
// whose result is used *only* by a Rocket dispatch gets pulled onto
// @rocket_device -- and then MaterializeInterfaces asks this backend to
// serialize something that is not a conv at all. The observed case is the
// explicit padding for an int8 depthwise (a 112x112x48 -> 114x114x48 copy
// dispatch, named `..._slow_memcpy`), which fails with "executable target
// config is missing required key 'input_width'". Nothing pulls it back
// toward the CPU: its destination is a fresh flow.tensor.splat, so the
// Rocket consumer is the only affinity constraint the analysis can see.
//
// stream.affinity.default (set on the module from --iree-hal-default-device)
// only applies when the analysis finds nothing, so it does not help here.
// This pass makes the placement explicit instead: any flow.dispatch without
// an affinity of its own is stamped with that default before the analysis
// ever runs.
//
// Must run between the `flow` and `stream` phases -- after dispatch region
// formation and outlining (so every dispatch exists and is a flow.dispatch),
// and before ConvertToStream consumes the affinities. There is no plugin
// hook that late in the pipeline, so rocket-compiler drives it by name over
// a --compile-to=flow module and then resumes with --compile-from=flow.

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::HAL {
namespace {

constexpr StringLiteral kAffinityAttrName = "stream.affinity";
constexpr StringLiteral kDefaultAffinityAttrName = "stream.affinity.default";

struct RocketPinUnclaimedDispatchesPass
    : public PassWrapper<RocketPinUnclaimedDispatchesPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RocketPinUnclaimedDispatchesPass)

  StringRef getArgument() const final {
    return "rocket-pin-unclaimed-dispatches";
  }
  StringRef getDescription() const final {
    return "Pins every flow.dispatch without an explicit stream.affinity to "
           "the module's stream.affinity.default, so only dispatches the "
           "Rocket transform spec claimed can be placed on the NPU.";
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    auto defaultAffinityAttr =
        module->getAttrOfType<IREE::Stream::AffinityAttr>(
            kDefaultAffinityAttrName);
    if (!defaultAffinityAttr) {
      // No default to fall back on means there is no device this pass can
      // honestly name. Failing is deliberate: silently doing nothing would
      // let an IREE-formed dispatch drift onto Rocket again and resurface as
      // an unexplained serialization error much later in the pipeline.
      module.emitError()
          << "rocket-pin-unclaimed-dispatches: module has no "
          << kDefaultAffinityAttrName
          << "; pass --iree-hal-default-device=<cpu device name> so there is "
             "a device to pin unclaimed dispatches to";
      return signalPassFailure();
    }

    unsigned pinnedCount = 0;
    module.walk([&](IREE::Flow::DispatchOp dispatchOp) {
      if (dispatchOp->hasAttr(kAffinityAttrName)) {
        return;
      }
      dispatchOp->setAttr(kAffinityAttrName, defaultAffinityAttr);
      ++pinnedCount;
    });

    // Quiet on the common case (every dispatch already placed), loud enough
    // to notice when a model does hit the pattern this pass exists for.
    if (pinnedCount > 0) {
      module.emitRemark() << "rocket-pin-unclaimed-dispatches: pinned "
                          << pinnedCount << " dispatch(es) to "
                          << defaultAffinityAttr;
    }
  }
};

static PassRegistration<RocketPinUnclaimedDispatchesPass> reg;

} // namespace
} // namespace mlir::iree_compiler::IREE::HAL
