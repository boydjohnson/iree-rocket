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
// Section 3 made the record answer "why is this op not on the NPU?"
// reliably, which meant asking rocket-core *both* of its questions here,
// not just the planner's. `transform.rocket.match.admitted` asks the
// admission envelope too, so a candidate the envelope declines stays on the
// CPU however well it plans -- and this pass used to record it as "direct".
// That is exactly the gap section 2 above describes on MobileNetV2 fp16
// before the depthwise ceiling moved: 53 candidates, none refused by the
// planner, 47 dispatch sites. The six the envelope declined read as
// offloaded in a record whose whole job is to say where work went.
//
// The record now also carries the precision rung the decision was reached
// on, a `limit` class saying whether the refusal is the shape, the
// semantics, a hardware bound or an unmeasured configuration, and the
// hardware-job and column counts of an accepted plan.
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
//   cpu       the planner refused, the admission envelope has no evidence
//             for this shape class, or the op's form is outside what the
//             lowering expresses (a batch above one, anisotropic strides,
//             dilations, f32 operands the demotion left alone, a matmul
//             with user-defined indexing maps); detail says which.
//   deferred  a dynamic extent; the runtime plans it at dispatch.
//
// Only the first two of those tag the op `rocket.plan_refused`: an
// admission refusal is reported and not tagged, because the envelope is
// indexed by the rung the *matcher* claims for and an `i8 x i8 -> i32`
// operation is on either int8 rung depending on which matcher takes it.
// This pass tries both before calling the class unmeasured, so the record
// is right, but it leaves the decision to the matcher.
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

/// Why an operation is not on the NPU, in the three classes
/// COMPILER_ROADMAP.md section 3 asks a report to distinguish. A reader has
/// to be able to tell "the hardware cannot do this" from "nobody has
/// measured it" from "it would be slower", because only the first is
/// permanent.
constexpr StringLiteral kLimitNone = "none";
/// The operation is malformed on its own terms.
constexpr StringLiteral kLimitShape = "shape";
/// Well-formed, but the Rocket lowering does not express its meaning.
constexpr StringLiteral kLimitSemantics = "semantics";
/// A fixed hardware bound, or no legal decomposition under one.
constexpr StringLiteral kLimitHardware = "hardware";
/// Register-representable, past the corpus that backs the planner's rules.
constexpr StringLiteral kLimitValidation = "validation";
/// The planner failed internally, or the boundary rejected the call. A bug
/// report, not a property of the shape.
constexpr StringLiteral kLimitInternal = "internal";

/// Which class a planner or admission status falls in. `form` and `dynamic`
/// are this pass's own statuses, not the ABI's.
StringRef limitClassFor(StringRef status) {
  if (status == "ok" || status == "dynamic") {
    return kLimitNone;
  }
  if (status == "invalid_shape") {
    return kLimitShape;
  }
  if (status == "unsupported_semantics" || status == "form") {
    return kLimitSemantics;
  }
  if (status == "hardware_limit" || status == "capacity_exceeded") {
    return kLimitHardware;
  }
  if (status == "unvalidated_configuration") {
    return kLimitValidation;
  }
  return kLimitInternal;
}

struct Decision {
  StringRef decision; // direct | tiled | cpu | deferred
  StringRef status;   // planner status name, or "form" / "dynamic"
  std::string detail;
  /// The rung the decision was reached on, in `rocket_plan_precision_name`
  /// spelling, or empty when the op's types never named one.
  StringRef precision;
  /// Standalone hardware jobs and column tiles, zero when not planned. The
  /// one-to-many mapping section 3 asks to track: an accepted candidate is
  /// one dispatch site running `tiles` jobs.
  int64_t tiles = 0;
  int64_t columns = 0;
  /// Whether to tag the op `rocket.plan_refused`, which makes every matcher
  /// decline it. True for exactly the refusals that set the tag before this
  /// pass reported admission as well: a form problem or a *planner*
  /// refusal. An admission refusal is reported but never tagged -- the
  /// matcher asks the envelope itself, and with the rung it is claiming
  /// for, which this pass can only guess at for an int8 operation.
  bool tagRefused = false;

  StringRef limitClass() const { return limitClassFor(status); }
};

Decision refuse(StringRef status, std::string detail, bool tag,
                StringRef precision = StringRef()) {
  Decision decision;
  decision.decision = "cpu";
  decision.status = status;
  decision.detail = std::move(detail);
  decision.precision = precision;
  decision.tagRefused = tag;
  return decision;
}

/// Fills `plan` with what the planner would program for `c` at `precision`,
/// returning the status and writing any refusal into `message`.
uint32_t planAt(const RocketCandidate &c, uint32_t precision,
                rocket_plan_conv_plan_t &plan, char *message, size_t capacity) {
  plan = {};
  plan.struct_size = sizeof(plan);
  rocket_plan_quantization_t quantization{};
  quantization.input_scale = 1.0f;
  quantization.weights_scale = 1.0f;
  quantization.output_scale = 1.0f;
  if (c.matmul) {
    rocket_plan_matmul_desc_t desc{};
    desc.struct_size = sizeof(desc);
    desc.precision = precision;
    desc.m = static_cast<uint64_t>(c.width);
    desc.k = static_cast<uint64_t>(c.inChannels);
    desc.n = static_cast<uint64_t>(c.outChannels);
    desc.activation = ROCKET_PLAN_ACTIVATION_NONE;
    desc.quantization = quantization;
    return rocket_plan_matmul(&desc, /*policy=*/nullptr, &plan, message,
                              capacity);
  }
  rocket_plan_conv_desc_t desc{};
  desc.struct_size = sizeof(desc);
  desc.precision = precision;
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
  return rocket_plan_conv(&desc, /*policy=*/nullptr, &plan, message, capacity);
}

/// Whether the admission envelope has evidence for `c`'s shape class at
/// `precision`.
uint32_t admitAt(const RocketCandidate &c, uint32_t precision, char *message,
                 size_t capacity) {
  if (c.matmul) {
    rocket_plan_matmul_admission_desc_t desc{};
    desc.struct_size = sizeof(desc);
    desc.precision = precision;
    desc.m = static_cast<uint64_t>(c.width);
    desc.k = static_cast<uint64_t>(c.inChannels);
    desc.n = static_cast<uint64_t>(c.outChannels);
    return rocket_admit_matmul(&desc, message, capacity);
  }
  rocket_plan_admission_desc_t desc{};
  desc.struct_size = sizeof(desc);
  desc.precision = precision;
  desc.kernel_height = static_cast<uint64_t>(c.kernelHeight);
  desc.kernel_width = static_cast<uint64_t>(c.kernelWidth);
  desc.stride = static_cast<uint64_t>(c.strideH);
  desc.in_channels = static_cast<uint64_t>(c.inChannels);
  desc.out_channels = static_cast<uint64_t>(c.outChannels);
  desc.depthwise = c.depthwise ? 1 : 0;
  return rocket_admit_conv(&desc, message, capacity);
}

Decision decide(const RocketCandidate &c) {
  if (c.dynamic) {
    Decision decision;
    decision.decision = "deferred";
    decision.status = "dynamic";
    decision.detail = "dynamic extents; the runtime plans this at dispatch";
    return decision;
  }
  if (!c.formProblem.empty()) {
    return refuse("form", c.formProblem, /*tag=*/true);
  }
  if (c.batch != 1) {
    return refuse("form",
                  "batch " + std::to_string(c.batch) +
                      "; the Rocket ABI fixes batch at one",
                  /*tag=*/true);
  }
  if (c.strideH != c.strideW) {
    return refuse("form",
                  "anisotropic stride " + std::to_string(c.strideH) + "x" +
                      std::to_string(c.strideW) +
                      "; the CNA programs one stride",
                  /*tag=*/true);
  }
  if (c.dilationH != 1 || c.dilationW != 1) {
    return refuse("form", "dilation; not expressed by the lowering",
                  /*tag=*/true);
  }
  std::string why;
  std::optional<uint32_t> derived = rocketPrecisionFor(c, why);
  if (!derived) {
    return refuse("form", why, /*tag=*/true);
  }
  StringRef derivedName(rocket_plan_precision_name(*derived));

  std::array<char, 512> message{};
  rocket_plan_conv_plan_t plan{};
  uint32_t status = planAt(c, *derived, plan, message.data(), message.size());
  StringRef statusName(rocket_plan_status_name(status));
  if (status != ROCKET_PLAN_OK) {
    // The only refusal that tags: the extents cannot be programmed on the
    // rung the operand types name, and no matcher can change that.
    return refuse(statusName, std::string(message.data()), /*tag=*/true,
                  derivedName);
  }
  if (plan.output_width != c.outWidth || plan.output_height != c.outHeight) {
    return refuse("invalid_shape",
                  "output extent " + std::to_string(c.outWidth) + "x" +
                      std::to_string(c.outHeight) + " disagrees with the " +
                      std::to_string(plan.output_width) + "x" +
                      std::to_string(plan.output_height) +
                      " the planner derives from the input, kernel and stride",
                  /*tag=*/true, derivedName);
  }

  // The second question, and the one that decides most CPU placements on a
  // model whose shapes are all programmable: has this *class* been measured
  // (rocket-core's `admission`)? `transform.rocket.match.admitted` asks it
  // of every matcher, so a candidate the envelope declines will be left on
  // the CPU however well it plans -- and a report that said "direct" here
  // would be describing a dispatch that never happens.
  //
  // An `i8 x i8 -> i32` operation is on whichever int8 rung the matcher
  // that claims it says, not one the types imply, so both are tried before
  // the class is called unmeasured. The rung that admits is the one
  // recorded, and the plan is retaken on it: the two rungs write different
  // output element widths, so their tile counts need not agree.
  uint32_t rung = *derived;
  std::array<char, 512> admitMessage{};
  uint32_t admitStatus =
      admitAt(c, rung, admitMessage.data(), admitMessage.size());
  if (admitStatus != ROCKET_PLAN_OK &&
      *derived == ROCKET_PLAN_PRECISION_INT8_ACCUMULATOR) {
    std::array<char, 512> requantMessage{};
    uint32_t requantStatus = admitAt(c, ROCKET_PLAN_PRECISION_INT8,
                                     requantMessage.data(), requantMessage.size());
    if (requantStatus == ROCKET_PLAN_OK) {
      rung = ROCKET_PLAN_PRECISION_INT8;
      admitStatus = requantStatus;
      status = planAt(c, rung, plan, message.data(), message.size());
      if (status != ROCKET_PLAN_OK) {
        return refuse(rocket_plan_status_name(status),
                      std::string(message.data()), /*tag=*/false,
                      rocket_plan_precision_name(rung));
      }
    }
  }
  StringRef rungName(rocket_plan_precision_name(rung));
  if (admitStatus != ROCKET_PLAN_OK) {
    // Deliberately untagged; see Decision::tagRefused.
    return refuse(rocket_plan_status_name(admitStatus),
                  std::string(admitMessage.data()), /*tag=*/false, rungName);
  }

  Decision decision;
  decision.status = "ok";
  decision.precision = rungName;
  decision.tiles = plan.tile_count;
  decision.columns = plan.column_count;
  decision.detail = "cbuf " + std::to_string(plan.data_banks) + "/" +
                    std::to_string(plan.weight_banks);
  if (plan.tile_count == 1) {
    decision.decision = "direct";
    return decision;
  }
  decision.decision = "tiled";
  decision.detail += ", tiles " + std::to_string(plan.tile_count) +
                     ", columns " + std::to_string(plan.column_count);
  return decision;
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
    // Hardware jobs, not dispatch sites: an accepted candidate is one
    // dispatch that runs `tiles` standalone jobs. Section 3 asks for the
    // two counts kept apart, and this is the only place the second one is
    // known.
    int64_t jobs = 0;
    root->walk([&](Operation *op) {
      std::optional<RocketCandidate> candidate = readRocketCandidate(op);
      if (!candidate) {
        return;
      }
      Decision decision = decide(*candidate);
      if (decision.tagRefused) {
        // The one attribute this pass leaves on an op, and only on refused
        // ones: `transform.rocket.match.admitted` declines it, and the DAG
        // matchers decline it by attribute-dictionary inequality. Accepted
        // and deferred ops stay untouched, and so does one the *admission
        // envelope* declined -- see Decision::tagRefused.
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
      jobs += decision.tiles;
      std::string shape = rocketShapeSummary(*candidate);
      if (decision.decision == "cpu" || decision.decision == "deferred") {
        // Location-only, deliberately: `op->emitRemark()` would append a
        // "see current operation" note printing the op, which the
        // match-boundary tests' CHECK-NOTs would then find.
        emitRemark(op->getLoc())
            << "rocket-plan: " << decision.decision << " [" << decision.status
            << "] " << candidate->kind << " " << shape << ": "
            << decision.detail;
      }
      decisions.push_back(builder.getDictionaryAttr({
          builder.getNamedAttr("kind", builder.getStringAttr(candidate->kind)),
          builder.getNamedAttr("layout", builder.getStringAttr(candidate->layout)),
          builder.getNamedAttr("shape", builder.getStringAttr(shape)),
          builder.getNamedAttr("precision",
                               builder.getStringAttr(decision.precision)),
          builder.getNamedAttr("decision", builder.getStringAttr(decision.decision)),
          builder.getNamedAttr("status", builder.getStringAttr(decision.status)),
          builder.getNamedAttr("limit",
                               builder.getStringAttr(decision.limitClass())),
          builder.getNamedAttr("detail", builder.getStringAttr(decision.detail)),
          builder.getNamedAttr("jobs", builder.getI64IntegerAttr(decision.tiles)),
          builder.getNamedAttr("columns",
                               builder.getI64IntegerAttr(decision.columns)),
          builder.getNamedAttr("loc", op->getLoc()),
      }));
    });
    if (decisions.empty()) {
      return;
    }
    root->setAttr(kDecisionsAttrName, builder.getArrayAttr(decisions));
    // Location-only for the same reason: attached to the function, the note
    // would print the whole body.
    emitRemark(root->getLoc())
        << "rocket-plan-candidates: " << decisions.size()
        << " candidate(s): " << direct << " direct, " << tiled << " tiled, "
        << cpu << " cpu, " << deferred << " deferred; " << jobs
        << " hardware job(s) if every accepted candidate is claimed";
  }
};

static PassRegistration<RocketPlanCandidatesPass> reg;

} // namespace
} // namespace mlir::iree_compiler::IREE::HAL
