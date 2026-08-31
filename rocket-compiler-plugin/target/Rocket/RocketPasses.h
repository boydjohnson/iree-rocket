// Copyright 2026
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Factories for the Rocket plugin passes that RocketTarget.cpp adds to an
// IREE pass pipeline itself. The two rocket-annotate-*-placement passes are
// deliberately absent: those are only ever reached by name, through
// rocket_conv2d_transform_spec.mlir's transform.apply_registered_pass, so
// their PassRegistration is the whole interface.

#ifndef ROCKET_COMPILER_PLUGIN_TARGET_ROCKET_ROCKETPASSES_H_
#define ROCKET_COMPILER_PLUGIN_TARGET_ROCKET_ROCKETPASSES_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::HAL {

// Rewrites `torch.operator "onnx.ConvInteger"` into quantized torch ops.
// Only built when IREE_INPUT_TORCH is enabled -- see target/Rocket's
// CMakeLists.txt and the ROCKET_ENABLE_ONNX_INPUT define it sets.
std::unique_ptr<Pass> createRocketExpandOnnxConvIntegerPass();

} // namespace mlir::iree_compiler::IREE::HAL

#endif // ROCKET_COMPILER_PLUGIN_TARGET_ROCKET_ROCKETPASSES_H_
