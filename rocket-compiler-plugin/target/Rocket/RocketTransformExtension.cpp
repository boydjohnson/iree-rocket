// Copyright 2026
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// `transform.rocket.match.admitted`: the one line every convolution and
// matmul matcher in rocket_conv2d_transform_spec.mlir carries, and the only
// shape predicate left in that file. It asks rocket-core two questions --
// did the planner refuse these extents (recorded ahead of the loop by
// rocket-plan-candidates as a `rocket.plan_refused` tag), and is this shape
// class inside the measured envelope (`rocket_admit_conv`, the ceilings
// that used to be `transform.iree.match.dim_bounds` lines here). See
// COMPILER_ROADMAP.md section 2 and rocket-core's `admission` module.

#include "RocketTransformExtension.h"

#include <array>
#include <string>

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectRegistry.h"

#include "RocketPlanQuery.h"
#include "rocket_plan.h"

using namespace mlir;

namespace mlir::iree_compiler::IREE::HAL {

rocket_transform::RocketTransformExtension::RocketTransformExtension() {
  registerTransformOps<
#define GET_OP_LIST
#include "RocketTransformOps.cpp.inc"
      >();
}

void registerRocketTransformExtension(DialectRegistry &registry) {
  registry.addExtensions<rocket_transform::RocketTransformExtension>();
}

namespace {

/// The `rocket_plan_precision_e` a `precision` attribute spells, or nullopt
/// for an unrecognized spelling.
std::optional<uint32_t> precisionFromAttr(StringRef spelling) {
  if (spelling == "fp16") {
    return ROCKET_PLAN_PRECISION_FP16;
  }
  if (spelling == "int8_accumulator") {
    return ROCKET_PLAN_PRECISION_INT8_ACCUMULATOR;
  }
  if (spelling == "int8_requant") {
    return ROCKET_PLAN_PRECISION_INT8;
  }
  return std::nullopt;
}

} // namespace

DiagnosedSilenceableFailure rocket_transform::MatchAdmittedOp::matchOperation(
    Operation *current, transform::TransformResults &results,
    transform::TransformState &state) {
  if (getNoOffload()) {
    // `rocket-compiler --no-offload`, which rewrites this attribute in.
    // Every matcher declines and the like-for-like CPU baseline keeps the
    // whole pipeline around the loop.
    return emitSilenceableError() << "offload is disabled for this build";
  }
  if (auto refused =
          current->getAttrOfType<StringAttr>(kRocketPlanRefusedAttrName)) {
    return emitSilenceableError()
           << "the shared planner refused this op: " << refused.getValue();
  }

  std::optional<RocketCandidate> candidate = readRocketCandidate(current);
  if (!candidate) {
    return emitSilenceableError()
           << "not a convolution or matmul the Rocket planner reads";
  }
  // The admission envelope is indexed by channels, kernel and stride only,
  // so a dynamic *spatial* extent is fine here -- but a dynamic channel
  // count or kernel is not, and nothing downstream would catch it: the
  // planner defers a dynamic op to the runtime, which would then have to
  // fail a dispatch the compiler had already committed to.
  if (candidate->channelsDynamic) {
    return emitSilenceableError()
           << "dynamic channel counts or kernel extents: the admission "
              "envelope cannot be checked, so the op stays on the CPU";
  }

  uint32_t precision;
  if (StringAttr spelling = getPrecisionAttr()) {
    std::optional<uint32_t> named = precisionFromAttr(spelling.getValue());
    if (!named) {
      // A typo in the spec would otherwise silently pick a rung.
      return emitDefiniteFailure()
             << "unknown precision '" << spelling.getValue()
             << "'; expected fp16, int8_accumulator or int8_requant";
    }
    precision = *named;
  } else {
    std::string why;
    std::optional<uint32_t> derived = rocketPrecisionFor(*candidate, why);
    if (!derived) {
      return emitSilenceableError() << why;
    }
    precision = *derived;
  }

  std::array<char, 512> message{};
  uint32_t status;
  if (candidate->matmul) {
    rocket_plan_matmul_admission_desc_t desc{};
    desc.struct_size = sizeof(desc);
    desc.precision = precision;
    desc.m = static_cast<uint64_t>(candidate->width);
    desc.k = static_cast<uint64_t>(candidate->inChannels);
    desc.n = static_cast<uint64_t>(candidate->outChannels);
    status = rocket_admit_matmul(&desc, message.data(), message.size());
  } else {
    rocket_plan_admission_desc_t desc{};
    desc.struct_size = sizeof(desc);
    desc.precision = precision;
    desc.kernel_height = static_cast<uint64_t>(candidate->kernelHeight);
    desc.kernel_width = static_cast<uint64_t>(candidate->kernelWidth);
    desc.stride = static_cast<uint64_t>(candidate->strideH);
    desc.in_channels = static_cast<uint64_t>(candidate->inChannels);
    desc.out_channels = static_cast<uint64_t>(candidate->outChannels);
    desc.depthwise = candidate->depthwise ? 1 : 0;
    status = rocket_admit_conv(&desc, message.data(), message.size());
  }
  if (status != ROCKET_PLAN_OK) {
    return emitSilenceableError()
           << "outside the Rocket admission envelope ["
           << rocket_plan_status_name(status) << "]: " << message.data();
  }
  return DiagnosedSilenceableFailure::success();
}

void rocket_transform::MatchAdmittedOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getOperandHandleMutable(), effects);
  transform::onlyReadsPayload(effects);
}

} // namespace mlir::iree_compiler::IREE::HAL

#define GET_OP_CLASSES
#include "RocketTransformOps.cpp.inc"
