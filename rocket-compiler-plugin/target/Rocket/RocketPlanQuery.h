// Copyright 2026
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Reading a convolution or matmul candidate out of linalg, in the terms the
// shared planner's C ABI takes. Shared by the two callers that ask
// rocket-core a question about a payload op:
//
//   RocketPlanCandidatesPass.cpp        "would the planner plan this?"
//   RocketTransformExtension.cpp        "is this shape class characterized?"
//
// Keeping one reader means the two cannot disagree about which operand is
// the filter or where the channel axis is -- a disagreement that would show
// up as a matcher admitting a shape the pass refused, or the reverse.

#ifndef ROCKET_COMPILER_PLUGIN_TARGET_ROCKET_ROCKETPLANQUERY_H_
#define ROCKET_COMPILER_PLUGIN_TARGET_ROCKET_ROCKETPLANQUERY_H_

#include <optional>
#include <string>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::iree_compiler::IREE::HAL {

/// Everything the planner's descriptors need, read off one candidate op.
/// `dynamic` means some extent the *planner* needs is not static; the
/// channel counts and the kernel are tracked separately by
/// `channelsDynamic`, because the admission envelope needs only those and
/// can answer while the spatial extents are still symbolic.
struct RocketCandidate {
  StringRef kind;
  /// "nhwc" / "nchw" for convolutions, "row_major" for a matmul. The linalg
  /// op name is deliberately *not* recorded: the record lives on the
  /// function through the whole pipeline, and a test proving a convolution
  /// was claimed does so by CHECK-NOT-ing its op name.
  StringRef layout;
  bool dynamic = false;
  bool channelsDynamic = false;
  bool depthwise = false;
  bool matmul = false;
  int64_t batch = 1;
  int64_t width = 0, height = 0, inChannels = 0, outChannels = 0;
  int64_t kernelHeight = 1, kernelWidth = 1;
  int64_t outWidth = 0, outHeight = 0;
  int64_t strideH = 1, strideW = 1;
  int64_t dilationH = 1, dilationW = 1;
  Type inputElement, filterElement, outputElement;
  /// Set when the op's *form* rules it out before the planner is asked.
  std::string formProblem;
};

/// Reads `op` if it is one of the five candidate forms (dense NHWC/NCHW
/// convolution, depthwise NHWC/NCHW convolution, row-major matmul), or
/// returns nullopt.
std::optional<RocketCandidate> readRocketCandidate(Operation *op);

/// The `rocket_plan_precision_e` the lowering would program for the
/// candidate's element types, or nullopt with a reason in `why`.
///
/// int8 always comes back as the *accumulator* rung here: the requantized
/// lowering has the same `i8 x i8 -> i32` operand types in the IR, and
/// which of the two a given convolution gets is decided by the matcher that
/// claims it, not by the types. A caller that knows it is asking on behalf
/// of the requantized path overrides this.
std::optional<uint32_t> rocketPrecisionFor(const RocketCandidate &candidate,
                                           std::string &why);

/// `240x14 Cin 88 Cout 528 k1x1 s1`, for remarks and decision records.
std::string rocketShapeSummary(const RocketCandidate &candidate);

} // namespace mlir::iree_compiler::IREE::HAL

#endif // ROCKET_COMPILER_PLUGIN_TARGET_ROCKET_ROCKETPLANQUERY_H_
