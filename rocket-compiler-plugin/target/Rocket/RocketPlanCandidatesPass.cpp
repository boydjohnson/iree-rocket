// Copyright 2026
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Asks the shared planner (rocket-core, through rocket-plan-ffi's C ABI)
// what it would do with every convolution and matmul in a function, and
// records the answer. COMPILER_ROADMAP.md section 2, first slice: the
// decision is *recorded*, not yet *acted on* -- the transform spec's
// matchers still decide what is offloaded, and this pass exists so the two
// can be compared before the matchers' shape predicates are replaced by
// planner queries.
//
// The decision is also *acted on*, in the one direction that is safe: a
// refused op is tagged `rocket.plan_refused = "<status>"`, and every
// shape-admitting matcher in the spec checks that tag through
// `transform.rocket.match.admitted` (the DAG matchers decline a tagged op
// by attribute-dictionary inequality). Admission is therefore the
// matchers' own predicates AND the planner's acceptance -- a strict
// narrowing, never a broadening -- so a shape the matchers' bounds admit
// but the planner would refuse at dispatch (a dense-layout row too wide
// for one CBUF bank, say) now falls back to the CPU instead of compiling
// to a dispatch the runtime rejects with a bare INVALID_ARGUMENT.
//
// Where it runs, and why the decisions are not (otherwise) op attributes. This has to
// see every candidate, so it runs right after `rocket-verify-conv-shapes`,
// before the first claiming loop (`rocket-fuse-conv-relu6` and the
// requantized/residual `foreach_match` claim most of a model's convolutions
// before `rocket-annotate-original-placement` ever runs -- on ResNet50 that
// pass tags one leftover op). The DAG matchers those loops use compare
// whole attribute dictionaries, so anything left on the ops here would
// make every one of them decline. The decisions therefore go on the
// *function*, as `rocket.plan_decisions`: an array of dictionaries, one per
// candidate, each with its kind and layout, a shape summary, the decision,
// the planner's status name, a detail string and the op's location. That is
// the dedicated report sink section 3 asks for, and it survives the
// candidates being erased. Refusals and deferrals are also emitted as
// remarks on the op, so `rocket-compiler audit` shows them next to the
// placement report.
//
// Decisions:
//   direct    one hardware job.
//   tiled     several standalone jobs; detail carries the tile and column
//             counts and the CBUF split.
//   cpu       the planner refused, or the op's form is outside what the
//             lowering expresses (a batch above one, anisotropic strides,
//             dilations, f32 operands the demotion left alone, a matmul
//             with user-defined indexing maps); detail says which.
//   deferred  a dynamic extent; the runtime plans it at dispatch.
//
// What is planned is what the CNA would be programmed with at this point
// in the pipeline: padding is still an explicit `tensor.pad` in front of
// the op, so the descriptor carries zero padding and the padded input. The
// planner's own output extents are compared with the op's result type; a
// disagreement is reported as an invalid shape, the same defect
// `rocket-verify-conv-shapes` looks for through a second route.
//
// The policy passed is "no overrides" (NULL), which is what a runtime with
// a clean environment plans under; the two environment overrides that
// widen the planner are research tools and not the compiler's business.

#include <array>
#include <cstring>
#include <string>

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

#include "RocketPlanQuery.h"
#include "RocketTransformExtension.h"
#include "rocket_plan.h"

namespace mlir::iree_compiler::IREE::HAL {
namespace {

constexpr StringLiteral kDecisionsAttrName = "rocket.plan_decisions";

struct Decision {
  StringRef decision; // direct | tiled | cpu | deferred
  StringRef status;   // planner status name, or "form" / "dynamic"
  std::string detail;
};

Decision decide(const RocketCandidate &c) {
  if (c.dynamic) {
    return {"deferred", "dynamic",
            "dynamic extents; the runtime plans this at dispatch"};
  }
  if (!c.formProblem.empty()) {
    return {"cpu", "form", c.formProblem};
  }
  if (c.batch != 1) {
    return {"cpu", "form", "batch " + std::to_string(c.batch) +
                               "; the Rocket ABI fixes batch at one"};
  }
  if (c.strideH != c.strideW) {
    return {"cpu", "form",
            "anisotropic stride " + std::to_string(c.strideH) + "x" +
                std::to_string(c.strideW) + "; the CNA programs one stride"};
  }
  if (c.dilationH != 1 || c.dilationW != 1) {
    return {"cpu", "form", "dilation; not expressed by the lowering"};
  }
  std::string why;
  std::optional<uint32_t> precision = rocketPrecisionFor(c, why);
  if (!precision) {
    return {"cpu", "form", why};
  }

  std::array<char, 512> message{};
  rocket_plan_conv_plan_t plan{};
  plan.struct_size = sizeof(plan);
  rocket_plan_quantization_t quantization{};
  quantization.input_scale = 1.0f;
  quantization.weights_scale = 1.0f;
  quantization.output_scale = 1.0f;
  uint32_t status;
  if (c.matmul) {
    rocket_plan_matmul_desc_t desc{};
    desc.struct_size = sizeof(desc);
    desc.precision = *precision;
    desc.m = static_cast<uint64_t>(c.width);
    desc.k = static_cast<uint64_t>(c.inChannels);
    desc.n = static_cast<uint64_t>(c.outChannels);
    desc.activation = ROCKET_PLAN_ACTIVATION_NONE;
    desc.quantization = quantization;
    status = rocket_plan_matmul(&desc, /*policy=*/nullptr, &plan,
                                message.data(), message.size());
  } else {
    rocket_plan_conv_desc_t desc{};
    desc.struct_size = sizeof(desc);
    desc.precision = *precision;
    desc.width = static_cast<uint64_t>(c.width);
    desc.height = static_cast<uint64_t>(c.height);
    desc.in_channels = static_cast<uint64_t>(c.inChannels);
    desc.out_channels = static_cast<uint64_t>(c.outChannels);
    desc.stride = static_cast<uint64_t>(c.strideH);
    desc.kernel_height = static_cast<uint64_t>(c.kernelHeight);
    desc.kernel_width = static_cast<uint64_t>(c.kernelWidth);
    // Padding is still an explicit tensor.pad in front of the op here.
    desc.pad_top = 0;
    desc.pad_left = 0;
    desc.activation = ROCKET_PLAN_ACTIVATION_NONE;
    desc.depthwise = c.depthwise ? 1 : 0;
    desc.quantization = quantization;
    status = rocket_plan_conv(&desc, /*policy=*/nullptr, &plan,
                              message.data(), message.size());
  }
  StringRef statusName(rocket_plan_status_name(status));
  if (status != ROCKET_PLAN_OK) {
    return {"cpu", statusName, std::string(message.data())};
  }
  if (plan.output_width != c.outWidth || plan.output_height != c.outHeight) {
    return {"cpu", "invalid_shape",
            "output extent " + std::to_string(c.outWidth) + "x" +
                std::to_string(c.outHeight) + " disagrees with the " +
                std::to_string(plan.output_width) + "x" +
                std::to_string(plan.output_height) +
                " the planner derives from the input, kernel and stride"};
  }
  std::string detail = "cbuf " + std::to_string(plan.data_banks) + "/" +
                       std::to_string(plan.weight_banks);
  if (plan.tile_count == 1) {
    return {"direct", statusName, detail};
  }
  detail += ", tiles " + std::to_string(plan.tile_count) + ", columns " +
            std::to_string(plan.column_count);
  return {"tiled", statusName, detail};
}

struct RocketPlanCandidatesPass
    : public PassWrapper<RocketPlanCandidatesPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RocketPlanCandidatesPass)

  StringRef getArgument() const final { return "rocket-plan-candidates"; }
  StringRef getDescription() const final {
    return "Records, on the function, the shared planner's decision for "
           "every convolution and matmul candidate.";
  }

  void runOnOperation() final {
    Operation *root = getOperation();
    MLIRContext *context = root->getContext();
    Builder builder(context);

    if (rocket_plan_abi_version() != ROCKET_PLAN_ABI_VERSION) {
      root->emitError() << "rocket-plan-candidates: rocket-plan-ffi reports "
                           "ABI version "
                        << rocket_plan_abi_version()
                        << " but this plugin was built against "
                        << ROCKET_PLAN_ABI_VERSION;
      return signalPassFailure();
    }

    SmallVector<Attribute> decisions;
    unsigned direct = 0, tiled = 0, cpu = 0, deferred = 0;
    root->walk([&](Operation *op) {
      std::optional<RocketCandidate> candidate = readRocketCandidate(op);
      if (!candidate) {
        return;
      }
      Decision decision = decide(*candidate);
      if (decision.decision == "cpu") {
        // The one attribute this pass leaves on an op, and only on refused
        // ones: `transform.rocket.match.admitted` declines it, and the DAG
        // matchers decline it by attribute-dictionary inequality. Accepted
        // and deferred ops stay untouched.
        op->setAttr(kRocketPlanRefusedAttrName,
                    builder.getStringAttr(decision.status));
      }
      if (decision.decision == "direct") {
        ++direct;
      } else if (decision.decision == "tiled") {
        ++tiled;
      } else if (decision.decision == "cpu") {
        ++cpu;
      } else {
        ++deferred;
      }
      std::string shape = rocketShapeSummary(*candidate);
      if (decision.decision == "cpu" || decision.decision == "deferred") {
        // Location-only, deliberately: `op->emitRemark()` would append a
        // "see current operation" note printing the op, which the
        // match-boundary tests' CHECK-NOTs would then find.
        emitRemark(op->getLoc())
            << "rocket-plan: " << decision.decision << " ["
                         << decision.status << "] " << candidate->kind
                         << " " << shape << ": " << decision.detail;
      }
      decisions.push_back(builder.getDictionaryAttr({
          builder.getNamedAttr("kind", builder.getStringAttr(candidate->kind)),
          builder.getNamedAttr("layout", builder.getStringAttr(candidate->layout)),
          builder.getNamedAttr("shape", builder.getStringAttr(shape)),
          builder.getNamedAttr("decision", builder.getStringAttr(decision.decision)),
          builder.getNamedAttr("status", builder.getStringAttr(decision.status)),
          builder.getNamedAttr("detail", builder.getStringAttr(decision.detail)),
          builder.getNamedAttr("loc", op->getLoc()),
      }));
    });
    if (decisions.empty()) {
      return;
    }
    root->setAttr(kDecisionsAttrName, builder.getArrayAttr(decisions));
    // Location-only for the same reason: attached to the function, the note
    // would print the whole body.
    emitRemark(root->getLoc()) << "rocket-plan-candidates: " << decisions.size()
                       << " candidate(s): " << direct << " direct, " << tiled
                       << " tiled, " << cpu << " cpu, " << deferred
                       << " deferred";
  }
};

static PassRegistration<RocketPlanCandidatesPass> reg;

} // namespace
} // namespace mlir::iree_compiler::IREE::HAL
