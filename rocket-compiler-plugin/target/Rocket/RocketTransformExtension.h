// Copyright 2026
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROCKET_COMPILER_PLUGIN_TARGET_ROCKET_ROCKETTRANSFORMEXTENSION_H_
#define ROCKET_COMPILER_PLUGIN_TARGET_ROCKET_ROCKETTRANSFORMEXTENSION_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/MatchInterfaces.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"

namespace mlir {
class DialectRegistry;
} // namespace mlir

#define GET_OP_CLASSES
#include "RocketTransformOps.h.inc"

namespace mlir::iree_compiler::IREE::HAL {

/// The name of the discardable attribute `rocket-plan-candidates` leaves on
/// an op the shared planner refused; `transform.rocket.match.admitted`
/// declines any op that carries it.
inline constexpr llvm::StringLiteral kRocketPlanRefusedAttrName =
    "rocket.plan_refused";

/// Registers `transform.rocket.match.*` with the transform dialect.
void registerRocketTransformExtension(DialectRegistry &registry);

namespace rocket_transform {
class RocketTransformExtension
    : public transform::TransformDialectExtension<RocketTransformExtension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RocketTransformExtension)
  RocketTransformExtension();
};
} // namespace rocket_transform

} // namespace mlir::iree_compiler::IREE::HAL

#endif // ROCKET_COMPILER_PLUGIN_TARGET_ROCKET_ROCKETTRANSFORMEXTENSION_H_
