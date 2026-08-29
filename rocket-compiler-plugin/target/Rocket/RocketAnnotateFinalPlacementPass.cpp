// Copyright 2026
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Tags every hal.executable.variant (and its exports) with
// rocket.final = "rocket" | "cpu", read straight off the variant's
// #hal.executable.target backend name -- this is the authoritative answer
// to "where did this dispatch actually end up", independent of whichever
// path (Rocket transform-spec splice vs normal codegen) produced it.
//
// Unlike RocketAnnotateOriginalPlacementPass, there is no preprocessing-time
// hook this late in the pipeline: buildTranslationPassPipeline only ever
// sees one variant's already-finalized inner module, not the whole-program
// executable table. Run this as a second, separate iree-opt step over
// `iree-compile ... --compile-to=executable-targets -o -` output -- one
// phase earlier than --compile-to=hal, which already serializes each
// variant into an opaque hal.executable.binary and discards
// #hal.executable.target along with it:
//
//   iree-compile ... --compile-to=executable-targets -o - | \
//     iree-opt --pass-pipeline="builtin.module(rocket-annotate-final-placement)"

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Twine.h"

namespace mlir::iree_compiler::IREE::HAL {
namespace {

struct RocketAnnotateFinalPlacementPass
    : public PassWrapper<RocketAnnotateFinalPlacementPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RocketAnnotateFinalPlacementPass)

  StringRef getArgument() const final {
    return "rocket-annotate-final-placement";
  }
  StringRef getDescription() const final {
    return "Tags hal.executable.variant ops (and their call sites) with "
           "rocket.final = rocket|cpu, read from the resolved "
           "#hal.executable.target backend.";
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    Builder builder(module.getContext());
    unsigned rocketVariantCount = 0;
    unsigned cpuVariantCount = 0;
    unsigned mixedExecutableCount = 0;

    for (auto executableOp : module.getOps<IREE::HAL::ExecutableOp>()) {
      llvm::SmallSet<StringRef, 2> backends;
      for (auto variantOp :
           executableOp.getOps<IREE::HAL::ExecutableVariantOp>()) {
        StringRef backend = variantOp.getTarget().getBackend().getValue();
        backends.insert(backend);
        bool isRocket = backend == "rocket";
        isRocket ? ++rocketVariantCount : ++cpuVariantCount;
        StringAttr tag =
            builder.getStringAttr(isRocket ? "rocket" : "cpu");
        variantOp->setAttr("rocket.final", tag);
        for (auto exportOp :
             variantOp.getOps<IREE::HAL::ExecutableExportOp>()) {
          exportOp->setAttr("rocket.final", tag);
        }
      }
      if (backends.empty()) {
        continue;
      }
      if (backends.size() > 1) {
        // Multiple variants targeting different backends for the same
        // executable -- the runtime picks one at load time, so there is no
        // single static answer. Rare in this pipeline (Rocket dispatches
        // and CPU dispatches live in separate executables), but don't
        // silently mislabel it as one or the other.
        executableOp->setAttr("rocket.final", builder.getStringAttr("mixed"));
        ++mixedExecutableCount;
        continue;
      }
      bool isRocket = *backends.begin() == "rocket";
      StringAttr tag = builder.getStringAttr(isRocket ? "rocket" : "cpu");
      executableOp->setAttr("rocket.final", tag);

      // Best-effort: also tag whatever in the module references this
      // executable by symbol (e.g. a dispatch call site), so the placement
      // is visible without cross-referencing the executable table by hand.
      // Not all IREE::HAL lowering configurations reference executables by
      // symbol at this phase (e.g. resolved command-buffer dispatch ops
      // reference a runtime handle instead) -- when that's the case this
      // simply finds no uses and the variant/executable-level tags above
      // remain the authoritative record.
      if (auto uses = SymbolTable::getSymbolUses(executableOp, module)) {
        for (const SymbolTable::SymbolUse &use : *uses) {
          use.getUser()->setAttr("rocket.final", tag);
        }
      }
    }

    module.emitRemark() << "rocket-annotate-final-placement: "
                        << rocketVariantCount << " variant(s) -> rocket, "
                        << cpuVariantCount << " variant(s) -> cpu"
                        << (mixedExecutableCount > 0
                                ? (", " + Twine(mixedExecutableCount) +
                                  " executable(s) with mixed variants")
                                      .str()
                                : std::string());
  }
};

static PassRegistration<RocketAnnotateFinalPlacementPass> reg;

} // namespace
} // namespace mlir::iree_compiler::IREE::HAL
