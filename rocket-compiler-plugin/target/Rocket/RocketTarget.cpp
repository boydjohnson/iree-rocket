// Copyright 2026
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// IREE compiler plugin for the "rocket" NPU HAL driver
// (rocket-hal-driver/iree-rocket-hal, separate sibling repos, already
// hardware-validated). Two things this file registers, mirroring
// compiler/plugins/target/CUDA/CUDATarget.cpp's structure (NOT VMVX's --
// VMVX only shows the TargetBackend half, since it reuses IREE's own
// "local" device; "rocket" is a genuinely separate device the way CUDA is):
//
//   RocketTargetDevice : TargetDevice
//     embeds #hal.device.target<"rocket",...>
//   RocketTargetBackend : TargetBackend
//     embeds #hal.executable.target<"rocket",...>
//
// v1 scope: Conv2d and FullyConnected, and deliberately does NOT rely on IREE's
// generic Flow/DispatchCreation to auto-form dispatch regions targeting
// "rocket" -- buildTranslationPassPipeline is empty. The real "codegen" for
// this backend is a hand-authored Transform Dialect script (modeled on
// samples/custom_dispatch/cpu/embedded/example_transform_spec.mlir) that
// matches linalg.conv_2d_nhwc_hwcf and splices in a flow.dispatch to a
// hand-authored hal.executable already targeting "rocket", with the
// matched op's static shape/dtype facts stamped directly onto that
// executable's #hal.executable.target config dict (see the key list in
// buildRocketConv2dConfigFromTarget below) -- there is nothing left for
// this backend's own pass pipeline to derive. serializeExecutable reads
// that config dict back out and emits the RKT1 FlatBuffer defined by the
// sibling rocket-schema repository. Generated FlatCC bindings keep this
// producer in sync with the Rust consumers.
//
// Real, honest limitation, not glossed over: nothing in this whole project
// wires up real calibrated quantization. Every config-dict-authored shape
// is expected to hardcode scale=1.0/zero_point=0/activation=none/
// precision=int8 unless a real calibration pipeline exists -- exactly the
// same placeholders rocket-hal-driver's own tag=0 hardcoded shape already
// uses.

#include <array>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#ifdef ROCKET_ENABLE_ONNX_INPUT
#include "RocketPasses.h"
#endif // ROCKET_ENABLE_ONNX_INPUT
#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/PluginAPI/Client.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "rocket_executable_def_builder.h"

namespace mlir::iree_compiler::IREE::HAL {
namespace {

struct RocketOptions {
  // No real flags needed for v1 -- kept as an empty options struct rather
  // than omitted, since PluginSession's template contract expects one
  // (matches samples/compiler_plugins/example/src/PluginRegistration.cpp's
  // MyOptions, the minimal real example of this shape in this checkout).
  void bindOptions(OptionsBinder &binder) {}
};

// The 20 ConvShape-mirroring config-dict keys a hand-authored
// #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {...}> is expected
// to carry, and the one place both the transform-script author and this
// backend need to agree on exact key spelling. Deliberately snake_case,
// matching iree-rocket-hal::rocket::regcmd::ConvShape's own field names
// 1:1 for direct visual correspondence between the .mlir config dict and
// the Rust struct it's standing in for.
constexpr std::array<const char *, 20> kRequiredConfigKeys = {
    "input_width",        "input_height",     "input_channels",
    "output_width",       "output_height",    "output_channels",
    "weights_width",      "weights_height",   "stride",
    "depthwise",          "input_zero_point", "output_zero_point",
    "weights_zero_point", "input_scale",      "weights_scale",
    "output_scale",       "truncate_bits",    "activation",
    "activation_cmp",     "precision",
};

constexpr std::array<const char *, 13> kRequiredFcConfigKeys = {
    "m",
    "k",
    "n",
    "input_zero_point",
    "output_zero_point",
    "weights_zero_point",
    "input_scale",
    "weights_scale",
    "output_scale",
    "truncate_bits",
    "activation",
    "activation_cmp",
    "precision",
};

// Pooling carries no weights, no bias and no requantization, so its key
// set is geometry plus the two things that are not geometry: which
// reduction, and what the elements are. The pad *fill* is deliberately
// absent -- it follows from those two (see PoolingMethod::pad_fill_value in
// iree-rocket-hal) and the runtime derives it.
constexpr std::array<const char *, 15> kRequiredPoolingConfigKeys = {
    "input_width", "input_height",  "channels",      "output_width",
    "output_height", "kernel_width", "kernel_height", "stride_x",
    "stride_y",    "pad_left",      "pad_top",       "pad_right",
    "pad_bottom",  "method",        "precision",
};

// Identical to the fully-connected set, because the operation is the same
// one under a name that exists in the input dialect.
constexpr std::array<const char *, 13> kRequiredMatmulConfigKeys = {
    "m",
    "k",
    "n",
    "input_zero_point",
    "output_zero_point",
    "weights_zero_point",
    "input_scale",
    "weights_scale",
    "output_scale",
    "truncate_bits",
    "activation",
    "activation_cmp",
    "precision",
};

struct RocketConv2dConfig {
  uint32_t inputWidth = 0;
  uint32_t inputHeight = 0;
  uint32_t inputChannels = 0;
  uint32_t outputWidth = 0;
  uint32_t outputHeight = 0;
  uint32_t outputChannels = 0;
  uint32_t weightsWidth = 0;
  uint32_t weightsHeight = 0;
  uint32_t stride = 0;
  bool depthwise = false;
  uint32_t inputZeroPoint = 0;
  uint32_t outputZeroPoint = 0;
  uint32_t weightsZeroPoint = 0;
  float inputScale = 1.0f;
  float weightsScale = 1.0f;
  float outputScale = 1.0f;
  uint32_t truncateBits = 0;
  iree_hal_rocket_Activation_enum_t activation =
      iree_hal_rocket_Activation_NONE;
  uint32_t activationCmp = 0;
  iree_hal_rocket_Precision_enum_t precision = iree_hal_rocket_Precision_INT8;
  std::vector<iree_hal_rocket_Conv2DDimension_enum_t> runtimeDimensions;
};

struct RocketFullyConnectedConfig {
  uint32_t m = 0;
  uint32_t k = 0;
  uint32_t n = 0;
  uint32_t inputZeroPoint = 0;
  uint32_t outputZeroPoint = 0;
  uint32_t weightsZeroPoint = 0;
  float inputScale = 1.0f;
  float weightsScale = 1.0f;
  float outputScale = 1.0f;
  uint32_t truncateBits = 0;
  iree_hal_rocket_Activation_enum_t activation =
      iree_hal_rocket_Activation_NONE;
  uint32_t activationCmp = 0;
  iree_hal_rocket_Precision_enum_t precision = iree_hal_rocket_Precision_INT8;
};

struct RocketPoolingConfig {
  uint32_t inputWidth = 0;
  uint32_t inputHeight = 0;
  uint32_t channels = 0;
  uint32_t outputWidth = 0;
  uint32_t outputHeight = 0;
  uint32_t kernelWidth = 0;
  uint32_t kernelHeight = 0;
  uint32_t strideX = 0;
  uint32_t strideY = 0;
  uint32_t padLeft = 0;
  uint32_t padTop = 0;
  uint32_t padRight = 0;
  uint32_t padBottom = 0;
  iree_hal_rocket_PoolingMethod_enum_t method =
      iree_hal_rocket_PoolingMethod_MAX;
  iree_hal_rocket_Precision_enum_t precision = iree_hal_rocket_Precision_INT8;
  std::vector<iree_hal_rocket_PoolingDimension_enum_t> runtimeDimensions;
};

struct RocketMatmulConfig {
  uint32_t m = 0;
  uint32_t k = 0;
  uint32_t n = 0;
  uint32_t inputZeroPoint = 0;
  uint32_t outputZeroPoint = 0;
  uint32_t weightsZeroPoint = 0;
  float inputScale = 1.0f;
  float weightsScale = 1.0f;
  float outputScale = 1.0f;
  uint32_t truncateBits = 0;
  iree_hal_rocket_Activation_enum_t activation =
      iree_hal_rocket_Activation_NONE;
  uint32_t activationCmp = 0;
  iree_hal_rocket_Precision_enum_t precision = iree_hal_rocket_Precision_INT8;
  std::vector<iree_hal_rocket_MatmulDimension_enum_t> runtimeDimensions;
};

LogicalResult
parseActivationAndPrecision(DictionaryAttr config,
                            iree_hal_rocket_Activation_enum_t &activationValue,
                            iree_hal_rocket_Precision_enum_t &precisionValue,
                            llvm::function_ref<InFlightDiagnostic()> diagFn) {
  StringRef activation =
      llvm::cast<StringAttr>(config.get("activation")).getValue();
  if (activation == "none") {
    activationValue = iree_hal_rocket_Activation_NONE;
  } else if (activation == "relu") {
    activationValue = iree_hal_rocket_Activation_RELU;
  } else if (activation == "relux") {
    activationValue = iree_hal_rocket_Activation_RELUX;
  } else {
    diagFn() << "rocket backend: unrecognized 'activation' config value '"
             << activation << "' (expected none/relu/relux)";
    return failure();
  }

  StringRef precision =
      llvm::cast<StringAttr>(config.get("precision")).getValue();
  if (precision == "int8") {
    precisionValue = iree_hal_rocket_Precision_INT8;
  } else if (precision == "fp16") {
    precisionValue = iree_hal_rocket_Precision_FP16;
  } else if (precision == "int8_accumulator") {
    precisionValue = iree_hal_rocket_Precision_INT8_ACCUMULATOR;
  } else {
    diagFn() << "rocket backend: unrecognized 'precision' config value '"
             << precision << "' (expected int8/fp16/int8_accumulator)";
    return failure();
  }
  return success();
}

// Reads the config dict back into a RocketConv2dConfig. Returns
// std::nullopt (and emits a clear diagnostic via `diagFn`) if any required
// key is missing or the wrong attribute kind -- this IS the defensive
// check guarding against some OTHER op ever getting silently routed to
// "rocket" (e.g. if generic dispatch-region formation ever decided to,
// since nothing in DispatchCreation is target-aware -- see this plugin's
// design notes): a real conv2d dispatch spliced in by the transform script
// always carries all 20 keys; anything else won't.
std::optional<RocketConv2dConfig> buildRocketConv2dConfigFromTarget(
    DictionaryAttr config, llvm::function_ref<InFlightDiagnostic()> diagFn) {
  if (!config) {
    diagFn() << "rocket backend requires a non-empty executable target "
                "config dict (got none) -- only hand-spliced conv2d "
                "dispatches from the rocket transform script are supported "
                "in v1";
    return std::nullopt;
  }
  for (const char *key : kRequiredConfigKeys) {
    if (!config.get(key)) {
      diagFn() << "rocket backend v1 only supports hand-spliced conv2d "
                  "dispatches; executable target config is missing "
                  "required key '"
               << key << "'";
      return std::nullopt;
    }
  }

  auto getU32 = [&](StringRef key) -> uint32_t {
    return static_cast<uint32_t>(
        llvm::cast<IntegerAttr>(config.get(key)).getInt());
  };
  auto getF32 = [&](StringRef key) -> float {
    return llvm::cast<FloatAttr>(config.get(key)).getValueAsDouble();
  };

  RocketConv2dConfig shape;
  shape.inputWidth = getU32("input_width");
  shape.inputHeight = getU32("input_height");
  shape.inputChannels = getU32("input_channels");
  shape.outputWidth = getU32("output_width");
  shape.outputHeight = getU32("output_height");
  shape.outputChannels = getU32("output_channels");
  shape.weightsWidth = getU32("weights_width");
  shape.weightsHeight = getU32("weights_height");
  shape.stride = getU32("stride");
  shape.depthwise = llvm::cast<BoolAttr>(config.get("depthwise")).getValue();
  shape.inputZeroPoint = getU32("input_zero_point");
  shape.outputZeroPoint = getU32("output_zero_point");
  shape.weightsZeroPoint = getU32("weights_zero_point");
  shape.inputScale = getF32("input_scale");
  shape.weightsScale = getF32("weights_scale");
  shape.outputScale = getF32("output_scale");
  shape.truncateBits = getU32("truncate_bits");
  shape.activationCmp = getU32("activation_cmp");

  if (failed(parseActivationAndPrecision(config, shape.activation,
                                         shape.precision, diagFn))) {
    return std::nullopt;
  }
  if (shape.precision == iree_hal_rocket_Precision_INT8_ACCUMULATOR &&
      shape.activation != iree_hal_rocket_Activation_NONE) {
    diagFn() << "rocket backend: int8_accumulator convolution cannot fuse "
                "an activation";
    return std::nullopt;
  }
  if (shape.precision == iree_hal_rocket_Precision_INT8_ACCUMULATOR &&
      (shape.inputZeroPoint != 0 || shape.outputZeroPoint != 0 ||
       shape.weightsZeroPoint != 0)) {
    diagFn() << "rocket backend: int8_accumulator convolution currently "
                "requires zero input, weight, and output zero-points";
    return std::nullopt;
  }

  std::array<bool, 8> isRuntimeDimension = {};
  if (Attribute runtimeDimensionsAttr = config.get("runtime_dimensions")) {
    auto runtimeDimensions = llvm::dyn_cast<ArrayAttr>(runtimeDimensionsAttr);
    if (!runtimeDimensions) {
      diagFn() << "rocket backend: optional 'runtime_dimensions' config "
                  "value must be an array of strings";
      return std::nullopt;
    }

    for (Attribute dimensionAttr : runtimeDimensions) {
      auto dimensionName = llvm::dyn_cast<StringAttr>(dimensionAttr);
      if (!dimensionName) {
        diagFn() << "rocket backend: every 'runtime_dimensions' entry must "
                    "be a string";
        return std::nullopt;
      }

      std::optional<iree_hal_rocket_Conv2DDimension_enum_t> dimension;
      StringRef name = dimensionName.getValue();
      if (name == "input_width") {
        dimension = iree_hal_rocket_Conv2DDimension_INPUT_WIDTH;
      } else if (name == "input_height") {
        dimension = iree_hal_rocket_Conv2DDimension_INPUT_HEIGHT;
      } else if (name == "input_channels") {
        dimension = iree_hal_rocket_Conv2DDimension_INPUT_CHANNELS;
      } else if (name == "output_channels") {
        dimension = iree_hal_rocket_Conv2DDimension_OUTPUT_CHANNELS;
      } else if (name == "weights_width") {
        dimension = iree_hal_rocket_Conv2DDimension_WEIGHTS_WIDTH;
      } else if (name == "weights_height") {
        dimension = iree_hal_rocket_Conv2DDimension_WEIGHTS_HEIGHT;
      } else {
        // Deliberately includes "output_width"/"output_height": the runtime
        // always derives those from the six settable dimensions plus stride
        // and padding, so their Conv2DDimension values are retired (see
        // rocket_executable_def.fbs) and the driver rejects any executable
        // listing them.
        diagFn() << "rocket backend: unknown runtime Conv2D dimension '" << name
                 << "'";
        return std::nullopt;
      }

      size_t dimensionIndex = static_cast<size_t>(*dimension);
      if (isRuntimeDimension[dimensionIndex]) {
        diagFn() << "rocket backend: duplicate runtime Conv2D dimension '"
                 << name << "'";
        return std::nullopt;
      }
      isRuntimeDimension[dimensionIndex] = true;
      shape.runtimeDimensions.push_back(*dimension);
    }
  }

  // The six dimensions a dispatch may supply. 'output_width'/'output_height'
  // are deliberately absent: the runtime derives them, so they carry no
  // template obligation in either direction -- a dynamically-shaped conv has
  // no compile-time output extent to state, and zero is the expected value
  // there.
  struct SettableDimension {
    iree_hal_rocket_Conv2DDimension_enum_t dimension;
    StringRef name;
    uint32_t value;
  };
  const std::array<SettableDimension, 6> dimensions = {{
      {iree_hal_rocket_Conv2DDimension_INPUT_WIDTH, "input_width",
       shape.inputWidth},
      {iree_hal_rocket_Conv2DDimension_INPUT_HEIGHT, "input_height",
       shape.inputHeight},
      {iree_hal_rocket_Conv2DDimension_INPUT_CHANNELS, "input_channels",
       shape.inputChannels},
      {iree_hal_rocket_Conv2DDimension_OUTPUT_CHANNELS, "output_channels",
       shape.outputChannels},
      {iree_hal_rocket_Conv2DDimension_WEIGHTS_WIDTH, "weights_width",
       shape.weightsWidth},
      {iree_hal_rocket_Conv2DDimension_WEIGHTS_HEIGHT, "weights_height",
       shape.weightsHeight},
  }};
  for (const auto &[dimension, name, value] : dimensions) {
    const bool isRuntime = isRuntimeDimension[static_cast<size_t>(dimension)];
    if (isRuntime && value != 0) {
      diagFn() << "rocket backend: runtime Conv2D dimension '" << name
               << "' must use 0 as its executable template value";
      return std::nullopt;
    }
    if (!isRuntime && value == 0) {
      diagFn() << "rocket backend: zero Conv2D dimension '" << name
               << "' must be listed in 'runtime_dimensions'";
      return std::nullopt;
    }
  }

  return shape;
}

std::optional<RocketFullyConnectedConfig>
buildRocketFullyConnectedConfigFromTarget(
    DictionaryAttr config, llvm::function_ref<InFlightDiagnostic()> diagFn) {
  if (!config) {
    diagFn() << "rocket fully-connected backend requires a non-empty "
                "executable target config dict";
    return std::nullopt;
  }
  for (const char *key : kRequiredFcConfigKeys) {
    if (!config.get(key)) {
      diagFn() << "rocket fully-connected executable target config is "
                  "missing required key '"
               << key << "'";
      return std::nullopt;
    }
  }

  auto getU32 = [&](StringRef key) -> uint32_t {
    return static_cast<uint32_t>(
        llvm::cast<IntegerAttr>(config.get(key)).getInt());
  };
  auto getF32 = [&](StringRef key) -> float {
    return llvm::cast<FloatAttr>(config.get(key)).getValueAsDouble();
  };

  RocketFullyConnectedConfig shape;
  shape.m = getU32("m");
  shape.k = getU32("k");
  shape.n = getU32("n");
  if (shape.m == 0 || shape.k == 0 || shape.n == 0) {
    diagFn() << "rocket fully-connected dimensions m/k/n must be nonzero";
    return std::nullopt;
  }
  shape.inputZeroPoint = getU32("input_zero_point");
  shape.outputZeroPoint = getU32("output_zero_point");
  shape.weightsZeroPoint = getU32("weights_zero_point");
  shape.inputScale = getF32("input_scale");
  shape.weightsScale = getF32("weights_scale");
  shape.outputScale = getF32("output_scale");
  shape.truncateBits = getU32("truncate_bits");
  shape.activationCmp = getU32("activation_cmp");
  if (failed(parseActivationAndPrecision(config, shape.activation,
                                         shape.precision, diagFn))) {
    return std::nullopt;
  }
  if (shape.precision == iree_hal_rocket_Precision_INT8_ACCUMULATOR) {
    diagFn() << "rocket fully-connected backend does not support "
                "int8_accumulator precision";
    return std::nullopt;
  }
  return shape;
}

std::optional<RocketPoolingConfig> buildRocketPoolingConfigFromTarget(
    DictionaryAttr config, llvm::function_ref<InFlightDiagnostic()> diagFn) {
  if (!config) {
    diagFn() << "rocket pooling backend requires a non-empty executable "
                "target config dict";
    return std::nullopt;
  }
  for (const char *key : kRequiredPoolingConfigKeys) {
    if (!config.get(key)) {
      diagFn() << "rocket pooling executable target config is missing "
                  "required key '"
               << key << "'";
      return std::nullopt;
    }
  }

  auto getU32 = [&](StringRef key) -> uint32_t {
    return static_cast<uint32_t>(
        llvm::cast<IntegerAttr>(config.get(key)).getInt());
  };

  RocketPoolingConfig shape;
  shape.inputWidth = getU32("input_width");
  shape.inputHeight = getU32("input_height");
  shape.channels = getU32("channels");
  shape.outputWidth = getU32("output_width");
  shape.outputHeight = getU32("output_height");
  shape.kernelWidth = getU32("kernel_width");
  shape.kernelHeight = getU32("kernel_height");
  shape.strideX = getU32("stride_x");
  shape.strideY = getU32("stride_y");
  shape.padLeft = getU32("pad_left");
  shape.padTop = getU32("pad_top");
  shape.padRight = getU32("pad_right");
  shape.padBottom = getU32("pad_bottom");

  StringRef method = llvm::cast<StringAttr>(config.get("method")).getValue();
  if (method == "avg") {
    shape.method = iree_hal_rocket_PoolingMethod_AVG;
  } else if (method == "max") {
    shape.method = iree_hal_rocket_PoolingMethod_MAX;
  } else if (method == "min") {
    shape.method = iree_hal_rocket_PoolingMethod_MIN;
  } else {
    // Deliberately no "sum": the PPU has no sum mode, and linalg's
    // pooling_*_sum plus its divide is what a compiler turns into "avg"
    // before it gets here. See rocket_executable_def.fbs.
    diagFn() << "rocket backend: unrecognized pooling 'method' config value '"
             << method << "' (expected avg/max/min)";
    return std::nullopt;
  }

  StringRef precision =
      llvm::cast<StringAttr>(config.get("precision")).getValue();
  if (precision == "int8") {
    shape.precision = iree_hal_rocket_Precision_INT8;
  } else if (precision == "fp16") {
    shape.precision = iree_hal_rocket_Precision_FP16;
  } else {
    // int8_accumulator has nothing to mean for a reduction that carries its
    // operand format straight through.
    diagFn() << "rocket backend: unrecognized pooling 'precision' config "
                "value '"
             << precision << "' (expected int8/fp16)";
    return std::nullopt;
  }

  std::array<bool, 7> isRuntimeDimension = {};
  if (Attribute runtimeDimensionsAttr = config.get("runtime_dimensions")) {
    auto runtimeDimensions = llvm::dyn_cast<ArrayAttr>(runtimeDimensionsAttr);
    if (!runtimeDimensions) {
      diagFn() << "rocket backend: optional 'runtime_dimensions' config "
                  "value must be an array of strings";
      return std::nullopt;
    }
    for (Attribute dimensionAttr : runtimeDimensions) {
      auto dimensionName = llvm::dyn_cast<StringAttr>(dimensionAttr);
      if (!dimensionName) {
        diagFn() << "rocket backend: every 'runtime_dimensions' entry must "
                    "be a string";
        return std::nullopt;
      }
      std::optional<iree_hal_rocket_PoolingDimension_enum_t> dimension;
      StringRef name = dimensionName.getValue();
      if (name == "input_width") {
        dimension = iree_hal_rocket_PoolingDimension_INPUT_WIDTH;
      } else if (name == "input_height") {
        dimension = iree_hal_rocket_PoolingDimension_INPUT_HEIGHT;
      } else if (name == "channels") {
        dimension = iree_hal_rocket_PoolingDimension_CHANNELS;
      } else if (name == "kernel_width") {
        dimension = iree_hal_rocket_PoolingDimension_KERNEL_WIDTH;
      } else if (name == "kernel_height") {
        dimension = iree_hal_rocket_PoolingDimension_KERNEL_HEIGHT;
      } else if (name == "stride_x") {
        dimension = iree_hal_rocket_PoolingDimension_STRIDE_X;
      } else if (name == "stride_y") {
        dimension = iree_hal_rocket_PoolingDimension_STRIDE_Y;
      } else {
        // Padding is deliberately not settable per dispatch, and the output
        // extents are derived by the runtime rather than stated.
        diagFn() << "rocket backend: unknown runtime pooling dimension '"
                 << name << "'";
        return std::nullopt;
      }
      size_t dimensionIndex = static_cast<size_t>(*dimension);
      if (isRuntimeDimension[dimensionIndex]) {
        diagFn() << "rocket backend: duplicate runtime pooling dimension '"
                 << name << "'";
        return std::nullopt;
      }
      isRuntimeDimension[dimensionIndex] = true;
      shape.runtimeDimensions.push_back(*dimension);
    }
  }

  struct SettableDimension {
    iree_hal_rocket_PoolingDimension_enum_t dimension;
    StringRef name;
    uint32_t value;
  };
  const std::array<SettableDimension, 7> dimensions = {{
      {iree_hal_rocket_PoolingDimension_INPUT_WIDTH, "input_width",
       shape.inputWidth},
      {iree_hal_rocket_PoolingDimension_INPUT_HEIGHT, "input_height",
       shape.inputHeight},
      {iree_hal_rocket_PoolingDimension_CHANNELS, "channels", shape.channels},
      {iree_hal_rocket_PoolingDimension_KERNEL_WIDTH, "kernel_width",
       shape.kernelWidth},
      {iree_hal_rocket_PoolingDimension_KERNEL_HEIGHT, "kernel_height",
       shape.kernelHeight},
      {iree_hal_rocket_PoolingDimension_STRIDE_X, "stride_x", shape.strideX},
      {iree_hal_rocket_PoolingDimension_STRIDE_Y, "stride_y", shape.strideY},
  }};
  for (const auto &[dimension, name, value] : dimensions) {
    const bool isRuntime = isRuntimeDimension[static_cast<size_t>(dimension)];
    if (isRuntime && value != 0) {
      diagFn() << "rocket backend: runtime pooling dimension '" << name
               << "' must use 0 as its executable template value";
      return std::nullopt;
    }
    if (!isRuntime && value == 0) {
      diagFn() << "rocket backend: zero pooling dimension '" << name
               << "' must be listed in 'runtime_dimensions'";
      return std::nullopt;
    }
  }

  // The output extents follow from the input geometry, so a dynamic pool
  // has none to state and a static one must state the right ones. The
  // runtime derives them either way and rejects a static disagreement; this
  // catches the compile-time half of the same contract.
  const bool isDynamic = !shape.runtimeDimensions.empty();
  if (isDynamic && (shape.outputWidth != 0 || shape.outputHeight != 0)) {
    diagFn() << "rocket backend: a pooling executable with runtime "
                "dimensions must use 0 for output_width/output_height, "
                "which the runtime derives";
    return std::nullopt;
  }
  if (!isDynamic && (shape.outputWidth == 0 || shape.outputHeight == 0)) {
    diagFn() << "rocket backend: a static pooling executable must state "
                "nonzero output_width/output_height";
    return std::nullopt;
  }

  return shape;
}

std::optional<RocketMatmulConfig> buildRocketMatmulConfigFromTarget(
    DictionaryAttr config, llvm::function_ref<InFlightDiagnostic()> diagFn) {
  if (!config) {
    diagFn() << "rocket matmul backend requires a non-empty executable "
                "target config dict";
    return std::nullopt;
  }
  for (const char *key : kRequiredMatmulConfigKeys) {
    if (!config.get(key)) {
      diagFn() << "rocket matmul executable target config is missing "
                  "required key '"
               << key << "'";
      return std::nullopt;
    }
  }

  auto getU32 = [&](StringRef key) -> uint32_t {
    return static_cast<uint32_t>(
        llvm::cast<IntegerAttr>(config.get(key)).getInt());
  };
  auto getF32 = [&](StringRef key) -> float {
    return llvm::cast<FloatAttr>(config.get(key)).getValueAsDouble();
  };

  RocketMatmulConfig shape;
  shape.m = getU32("m");
  shape.k = getU32("k");
  shape.n = getU32("n");
  shape.inputZeroPoint = getU32("input_zero_point");
  shape.outputZeroPoint = getU32("output_zero_point");
  shape.weightsZeroPoint = getU32("weights_zero_point");
  shape.inputScale = getF32("input_scale");
  shape.weightsScale = getF32("weights_scale");
  shape.outputScale = getF32("output_scale");
  shape.truncateBits = getU32("truncate_bits");
  shape.activationCmp = getU32("activation_cmp");
  if (failed(parseActivationAndPrecision(config, shape.activation,
                                         shape.precision, diagFn))) {
    return std::nullopt;
  }
  if (shape.precision == iree_hal_rocket_Precision_INT8_ACCUMULATOR) {
    // The exact accumulator writer is validated for Conv2D only; this
    // lowering has its own output packing.
    diagFn() << "rocket matmul backend does not support int8_accumulator "
                "precision";
    return std::nullopt;
  }

  std::array<bool, 3> isRuntimeDimension = {};
  if (Attribute runtimeDimensionsAttr = config.get("runtime_dimensions")) {
    auto runtimeDimensions = llvm::dyn_cast<ArrayAttr>(runtimeDimensionsAttr);
    if (!runtimeDimensions) {
      diagFn() << "rocket backend: optional 'runtime_dimensions' config "
                  "value must be an array of strings";
      return std::nullopt;
    }
    for (Attribute dimensionAttr : runtimeDimensions) {
      auto dimensionName = llvm::dyn_cast<StringAttr>(dimensionAttr);
      if (!dimensionName) {
        diagFn() << "rocket backend: every 'runtime_dimensions' entry must "
                    "be a string";
        return std::nullopt;
      }
      std::optional<iree_hal_rocket_MatmulDimension_enum_t> dimension;
      StringRef name = dimensionName.getValue();
      if (name == "m") {
        dimension = iree_hal_rocket_MatmulDimension_M;
      } else if (name == "k") {
        dimension = iree_hal_rocket_MatmulDimension_K;
      } else if (name == "n") {
        dimension = iree_hal_rocket_MatmulDimension_N;
      } else {
        diagFn() << "rocket backend: unknown runtime matmul dimension '"
                 << name << "'";
        return std::nullopt;
      }
      size_t dimensionIndex = static_cast<size_t>(*dimension);
      if (isRuntimeDimension[dimensionIndex]) {
        diagFn() << "rocket backend: duplicate runtime matmul dimension '"
                 << name << "'";
        return std::nullopt;
      }
      isRuntimeDimension[dimensionIndex] = true;
      shape.runtimeDimensions.push_back(*dimension);
    }
  }

  struct SettableDimension {
    iree_hal_rocket_MatmulDimension_enum_t dimension;
    StringRef name;
    uint32_t value;
  };
  const std::array<SettableDimension, 3> dimensions = {{
      {iree_hal_rocket_MatmulDimension_M, "m", shape.m},
      {iree_hal_rocket_MatmulDimension_K, "k", shape.k},
      {iree_hal_rocket_MatmulDimension_N, "n", shape.n},
  }};
  for (const auto &[dimension, name, value] : dimensions) {
    const bool isRuntime = isRuntimeDimension[static_cast<size_t>(dimension)];
    if (isRuntime && value != 0) {
      diagFn() << "rocket backend: runtime matmul dimension '" << name
               << "' must use 0 as its executable template value";
      return std::nullopt;
    }
    if (!isRuntime && value == 0) {
      diagFn() << "rocket backend: zero matmul dimension '" << name
               << "' must be listed in 'runtime_dimensions'";
      return std::nullopt;
    }
  }

  return shape;
}

class RocketTargetDevice final : public TargetDevice {
public:
  RocketTargetDevice(const RocketOptions & /*options*/) {}

  IREE::HAL::DeviceTargetAttr
  getDefaultDeviceTarget(MLIRContext *context,
                         const TargetRegistry &targetRegistry) const final {
    Builder b(context);
    auto deviceConfigAttr = b.getDictionaryAttr({});
    auto executableConfigAttr = b.getDictionaryAttr({});
    SmallVector<IREE::HAL::ExecutableTargetAttr> executableTargetAttrs;
    targetRegistry.getTargetBackend("rocket")->getDefaultExecutableTargets(
        context, "rocket", executableConfigAttr, executableTargetAttrs);
    return IREE::HAL::DeviceTargetAttr::get(context, b.getStringAttr("rocket"),
                                            deviceConfigAttr,
                                            executableTargetAttrs);
  }
};

class RocketTargetBackend final : public TargetBackend {
public:
  RocketTargetBackend(const RocketOptions &options) : options(options) {}

  std::string getLegacyDefaultDeviceID() const final { return "rocket"; }

  void getDefaultExecutableTargets(
      MLIRContext *context, StringRef deviceID, DictionaryAttr deviceConfigAttr,
      SmallVectorImpl<IREE::HAL::ExecutableTargetAttr> &executableTargetAttrs)
      const final {
    Builder b(context);
    // No default config here -- real conv2d shapes are stamped onto each
    // individual hal.executable.variant by the transform script at match
    // time, not derivable generically at this "what targets exist" query
    // point (this function has no specific op in hand yet).
    executableTargetAttrs.push_back(b.getAttr<IREE::HAL::ExecutableTargetAttr>(
        b.getStringAttr("rocket"), b.getStringAttr("rocket-flatbuffer-v1"),
        b.getDictionaryAttr({})));
  }

  // Pure virtual in TargetBackend; intentionally empty. The inner module
  // is already in its final form by the time this backend ever sees
  // it -- see this file's top doc comment for why there is no real
  // lowering for this backend to do.
  void buildTranslationPassPipeline(IREE::HAL::ExecutableTargetAttr targetAttr,
                                    OpPassManager &passManager) final {}

  LogicalResult serializeExecutable(const SerializationOptions &serOptions,
                                    IREE::HAL::ExecutableVariantOp variantOp,
                                    OpBuilder &executableBuilder) final {
    if (variantOp.getTarget().getFormat() != "rocket-flatbuffer-v1") {
      return variantOp.emitOpError()
             << "unsupported Rocket executable format '"
             << variantOp.getTarget().getFormat().getValue()
             << "'; expected 'rocket-flatbuffer-v1'";
    }

    auto diagFn = [&]() { return variantOp.emitOpError(); };
    DictionaryAttr config = variantOp.getTarget().getConfiguration();
    StringRef kernel = "conv2d";
    if (Attribute kernelAttr = config ? config.get("kernel") : Attribute{}) {
      auto kernelString = llvm::dyn_cast<StringAttr>(kernelAttr);
      if (!kernelString) {
        return variantOp.emitOpError()
               << "rocket executable target 'kernel' must be a string";
      }
      kernel = kernelString.getValue();
    }
    if (kernel != "conv2d" && kernel != "fully_connected" &&
        kernel != "pooling" && kernel != "matmul") {
      return variantOp.emitOpError()
             << "unsupported Rocket kernel '" << kernel
             << "'; expected 'conv2d', 'pooling', 'matmul' or the "
                "deprecated 'fully_connected'";
    }

    std::optional<RocketConv2dConfig> convShape;
    // Deprecated: nothing emits this any more. "Fully connected" is not an
    // operation in the linalg dialect, so no matcher can produce one;
    // `matmul` is the same lowering under the name the input dialect uses.
    // Kept because an existing hand-authored spec may still say it.
    std::optional<RocketFullyConnectedConfig> fcShape;
    std::optional<RocketPoolingConfig> poolingShape;
    std::optional<RocketMatmulConfig> matmulShape;
    if (kernel == "fully_connected") {
      fcShape = buildRocketFullyConnectedConfigFromTarget(config, diagFn);
    } else if (kernel == "pooling") {
      poolingShape = buildRocketPoolingConfigFromTarget(config, diagFn);
    } else if (kernel == "matmul") {
      matmulShape = buildRocketMatmulConfigFromTarget(config, diagFn);
    } else {
      convShape = buildRocketConv2dConfigFromTarget(config, diagFn);
    }
    if (!convShape && !fcShape && !poolingShape && !matmulShape) {
      return failure();
    }

    auto exportOps = llvm::to_vector(variantOp.getExportOps());
    if (exportOps.size() != 1) {
      return variantOp.emitOpError()
             << "rocket-flatbuffer-v1 currently requires exactly one export "
                "per executable variant, but found "
             << exportOps.size();
    }
    IREE::HAL::ExecutableExportOp exportOp = exportOps.front();
    auto ordinalAttr = exportOp.getOrdinalAttr();
    if (!ordinalAttr || ordinalAttr.getInt() != 0) {
      return exportOp.emitOpError()
             << "rocket-flatbuffer-v1 requires its single export to have "
                "ordinal 0";
    }
    int64_t pipelineConstantCount = exportOp.getLayoutAttr().getConstants();
    size_t runtimeDimensionCount = 0;
    if (convShape) {
      runtimeDimensionCount = convShape->runtimeDimensions.size();
    } else if (poolingShape) {
      runtimeDimensionCount = poolingShape->runtimeDimensions.size();
    } else if (matmulShape) {
      runtimeDimensionCount = matmulShape->runtimeDimensions.size();
    }
    if (pipelineConstantCount != static_cast<int64_t>(runtimeDimensionCount)) {
      return exportOp.emitOpError()
             << "Rocket pipeline layout declares " << pipelineConstantCount
             << " push constants, but the executable target declares "
             << runtimeDimensionCount << " runtime dimensions";
    }

    FlatbufferBuilder builder;
    if (iree_hal_rocket_ExecutableDef_start_as_root(builder)) {
      return variantOp.emitOpError()
             << "failed to start Rocket executable FlatBuffer";
    }
    iree_hal_rocket_KernelDef_union_ref_t kernelRef;
    if (fcShape) {
      auto fcRef = iree_hal_rocket_FullyConnectedDef_create(
          builder, fcShape->m, fcShape->k, fcShape->n, fcShape->inputZeroPoint,
          fcShape->outputZeroPoint, fcShape->weightsZeroPoint,
          fcShape->inputScale, fcShape->weightsScale, fcShape->outputScale,
          fcShape->truncateBits, fcShape->activation, fcShape->activationCmp,
          fcShape->precision);
      if (!fcRef) {
        return variantOp.emitOpError()
               << "failed to build Rocket fully-connected definition";
      }
      kernelRef = iree_hal_rocket_KernelDef_as_FullyConnectedDef(fcRef);
    } else if (poolingShape) {
      iree_hal_rocket_PoolingDimension_vec_ref_t runtimeDimensionsRef = 0;
      if (!poolingShape->runtimeDimensions.empty()) {
        runtimeDimensionsRef = iree_hal_rocket_PoolingDimension_vec_create(
            builder, poolingShape->runtimeDimensions.data(),
            poolingShape->runtimeDimensions.size());
        if (!runtimeDimensionsRef) {
          return variantOp.emitOpError()
                 << "failed to build Rocket pooling runtime-dimension vector";
        }
      }
      if (iree_hal_rocket_PoolingDef_start(builder) ||
          iree_hal_rocket_PoolingDef_input_width_add(builder,
                                                     poolingShape->inputWidth) ||
          iree_hal_rocket_PoolingDef_input_height_add(
              builder, poolingShape->inputHeight) ||
          iree_hal_rocket_PoolingDef_channels_add(builder,
                                                  poolingShape->channels) ||
          iree_hal_rocket_PoolingDef_output_width_add(
              builder, poolingShape->outputWidth) ||
          iree_hal_rocket_PoolingDef_output_height_add(
              builder, poolingShape->outputHeight) ||
          iree_hal_rocket_PoolingDef_kernel_width_add(
              builder, poolingShape->kernelWidth) ||
          iree_hal_rocket_PoolingDef_kernel_height_add(
              builder, poolingShape->kernelHeight) ||
          iree_hal_rocket_PoolingDef_stride_x_add(builder,
                                                  poolingShape->strideX) ||
          iree_hal_rocket_PoolingDef_stride_y_add(builder,
                                                  poolingShape->strideY) ||
          iree_hal_rocket_PoolingDef_pad_left_add(builder,
                                                  poolingShape->padLeft) ||
          iree_hal_rocket_PoolingDef_pad_top_add(builder,
                                                 poolingShape->padTop) ||
          iree_hal_rocket_PoolingDef_pad_right_add(builder,
                                                   poolingShape->padRight) ||
          iree_hal_rocket_PoolingDef_pad_bottom_add(builder,
                                                    poolingShape->padBottom) ||
          iree_hal_rocket_PoolingDef_method_add(builder,
                                                poolingShape->method) ||
          iree_hal_rocket_PoolingDef_precision_add(builder,
                                                   poolingShape->precision) ||
          (runtimeDimensionsRef &&
           iree_hal_rocket_PoolingDef_runtime_dimensions_add(
               builder, runtimeDimensionsRef))) {
        return variantOp.emitOpError()
               << "failed to build Rocket pooling definition";
      }
      auto poolingRef = iree_hal_rocket_PoolingDef_end(builder);
      if (!poolingRef) {
        return variantOp.emitOpError()
               << "failed to finish Rocket pooling definition";
      }
      kernelRef = iree_hal_rocket_KernelDef_as_PoolingDef(poolingRef);
    } else if (matmulShape) {
      iree_hal_rocket_MatmulDimension_vec_ref_t runtimeDimensionsRef = 0;
      if (!matmulShape->runtimeDimensions.empty()) {
        runtimeDimensionsRef = iree_hal_rocket_MatmulDimension_vec_create(
            builder, matmulShape->runtimeDimensions.data(),
            matmulShape->runtimeDimensions.size());
        if (!runtimeDimensionsRef) {
          return variantOp.emitOpError()
                 << "failed to build Rocket matmul runtime-dimension vector";
        }
      }
      if (iree_hal_rocket_MatmulDef_start(builder) ||
          iree_hal_rocket_MatmulDef_m_add(builder, matmulShape->m) ||
          iree_hal_rocket_MatmulDef_k_add(builder, matmulShape->k) ||
          iree_hal_rocket_MatmulDef_n_add(builder, matmulShape->n) ||
          iree_hal_rocket_MatmulDef_input_zero_point_add(
              builder, matmulShape->inputZeroPoint) ||
          iree_hal_rocket_MatmulDef_output_zero_point_add(
              builder, matmulShape->outputZeroPoint) ||
          iree_hal_rocket_MatmulDef_weights_zero_point_add(
              builder, matmulShape->weightsZeroPoint) ||
          iree_hal_rocket_MatmulDef_input_scale_add(builder,
                                                    matmulShape->inputScale) ||
          iree_hal_rocket_MatmulDef_weights_scale_add(
              builder, matmulShape->weightsScale) ||
          iree_hal_rocket_MatmulDef_output_scale_add(
              builder, matmulShape->outputScale) ||
          iree_hal_rocket_MatmulDef_truncate_bits_add(
              builder, matmulShape->truncateBits) ||
          iree_hal_rocket_MatmulDef_activation_add(builder,
                                                   matmulShape->activation) ||
          iree_hal_rocket_MatmulDef_activation_cmp_add(
              builder, matmulShape->activationCmp) ||
          iree_hal_rocket_MatmulDef_precision_add(builder,
                                                  matmulShape->precision) ||
          (runtimeDimensionsRef &&
           iree_hal_rocket_MatmulDef_runtime_dimensions_add(
               builder, runtimeDimensionsRef))) {
        return variantOp.emitOpError()
               << "failed to build Rocket matmul definition";
      }
      auto matmulRef = iree_hal_rocket_MatmulDef_end(builder);
      if (!matmulRef) {
        return variantOp.emitOpError()
               << "failed to finish Rocket matmul definition";
      }
      kernelRef = iree_hal_rocket_KernelDef_as_MatmulDef(matmulRef);
    } else {
      iree_hal_rocket_Conv2DDimension_vec_ref_t runtimeDimensionsRef = 0;
      if (!convShape->runtimeDimensions.empty()) {
        runtimeDimensionsRef = iree_hal_rocket_Conv2DDimension_vec_create(
            builder, convShape->runtimeDimensions.data(),
            convShape->runtimeDimensions.size());
        if (!runtimeDimensionsRef) {
          return variantOp.emitOpError()
                 << "failed to build Rocket runtime-dimension vector";
        }
      }

      if (iree_hal_rocket_Conv2DDef_start(builder) ||
          iree_hal_rocket_Conv2DDef_input_width_add(builder,
                                                    convShape->inputWidth) ||
          iree_hal_rocket_Conv2DDef_input_height_add(builder,
                                                     convShape->inputHeight) ||
          iree_hal_rocket_Conv2DDef_input_channels_add(
              builder, convShape->inputChannels) ||
          iree_hal_rocket_Conv2DDef_output_width_add(builder,
                                                     convShape->outputWidth) ||
          iree_hal_rocket_Conv2DDef_output_height_add(
              builder, convShape->outputHeight) ||
          iree_hal_rocket_Conv2DDef_output_channels_add(
              builder, convShape->outputChannels) ||
          iree_hal_rocket_Conv2DDef_weights_width_add(
              builder, convShape->weightsWidth) ||
          iree_hal_rocket_Conv2DDef_weights_height_add(
              builder, convShape->weightsHeight) ||
          iree_hal_rocket_Conv2DDef_stride_add(builder, convShape->stride) ||
          iree_hal_rocket_Conv2DDef_depthwise_add(builder,
                                                  convShape->depthwise) ||
          iree_hal_rocket_Conv2DDef_input_zero_point_add(
              builder, convShape->inputZeroPoint) ||
          iree_hal_rocket_Conv2DDef_output_zero_point_add(
              builder, convShape->outputZeroPoint) ||
          iree_hal_rocket_Conv2DDef_weights_zero_point_add(
              builder, convShape->weightsZeroPoint) ||
          iree_hal_rocket_Conv2DDef_input_scale_add(builder,
                                                    convShape->inputScale) ||
          iree_hal_rocket_Conv2DDef_weights_scale_add(
              builder, convShape->weightsScale) ||
          iree_hal_rocket_Conv2DDef_output_scale_add(builder,
                                                     convShape->outputScale) ||
          iree_hal_rocket_Conv2DDef_truncate_bits_add(
              builder, convShape->truncateBits) ||
          iree_hal_rocket_Conv2DDef_activation_add(builder,
                                                   convShape->activation) ||
          iree_hal_rocket_Conv2DDef_activation_cmp_add(
              builder, convShape->activationCmp) ||
          iree_hal_rocket_Conv2DDef_precision_add(builder,
                                                  convShape->precision) ||
          (runtimeDimensionsRef &&
           iree_hal_rocket_Conv2DDef_runtime_dimensions_add(
               builder, runtimeDimensionsRef)) ||
          iree_hal_rocket_Conv2DDef_pad_top_add(builder, 0) ||
          iree_hal_rocket_Conv2DDef_pad_left_add(builder, 0)) {
        return variantOp.emitOpError()
               << "failed to populate Rocket convolution definition";
      }
      auto convRef = iree_hal_rocket_Conv2DDef_end(builder);
      if (!convRef) {
        return variantOp.emitOpError()
               << "failed to finish Rocket convolution definition";
      }
      kernelRef = iree_hal_rocket_KernelDef_as_Conv2DDef(convRef);
    }
    auto exportNameRef = builder.createString(exportOp.getName());
    if (!exportNameRef) {
      return exportOp.emitOpError() << "failed to build Rocket export name";
    }
    auto exportRef =
        iree_hal_rocket_ExportDef_create(builder, exportNameRef, kernelRef);
    if (!exportRef) {
      return exportOp.emitOpError()
             << "failed to build Rocket export definition";
    }
    auto exportsRef =
        iree_hal_rocket_ExportDef_vec_create(builder, &exportRef, 1);
    if (!exportsRef) {
      return variantOp.emitOpError() << "failed to build Rocket exports vector";
    }
    if (iree_hal_rocket_ExecutableDef_exports_add(builder, exportsRef) ||
        !iree_hal_rocket_ExecutableDef_end_as_root(builder)) {
      return variantOp.emitOpError()
             << "failed to finish Rocket executable FlatBuffer";
    }

    auto binaryOp = IREE::HAL::ExecutableBinaryOp::create(
        executableBuilder, variantOp.getLoc(), variantOp.getSymName(),
        variantOp.getTarget().getFormat(),
        builder.getHeaderPrefixedBufferAttr(
            executableBuilder.getContext(),
            iree_hal_rocket_ExecutableDef_file_identifier,
            /*version=*/0));
    binaryOp.setMimeTypeAttr(
        executableBuilder.getStringAttr("application/x-flatbuffers"));

    return success();
  }

private:
  const RocketOptions &options;
};

struct RocketSession final
    : PluginSession<RocketSession, RocketOptions,
                    PluginActivationPolicy::DefaultActivated> {
  void populateHALTargetDevices(IREE::HAL::TargetDeviceList &targets) final {
    // #hal.device.target<"rocket", ...
    targets.add("rocket", [&]() {
      return std::make_shared<RocketTargetDevice>(options);
    });
  }
  void populateHALTargetBackends(IREE::HAL::TargetBackendList &targets) final {
    // #hal.executable.target<"rocket", ...
    targets.add("rocket", [&]() {
      return std::make_shared<RocketTargetBackend>(options);
    });
  }

#ifdef ROCKET_ENABLE_ONNX_INPUT
  void extendInputConversionPreprocessingPassPipeline(
      OpPassManager &passManager,
      InputDialectOptions::Type inputType) override {
    // This hook is the only place upstream of the torch plugin's onnx->torch
    // conversion, which is where onnx.ConvInteger has to be rewritten: by the
    // time --iree-preprocessing-pass-pipeline runs, the unconverted
    // torch.operator has already failed to legalize. The enum cannot
    // distinguish "onnx" from any other plugin-provided input type (they all
    // arrive as Type::plugin), so the pass is added for every input type and
    // no-ops when the module holds no onnx.ConvInteger.
    passManager.addPass(createRocketExpandOnnxConvIntegerPass());
  }
#endif // ROCKET_ENABLE_ONNX_INPUT
};

} // namespace
} // namespace mlir::iree_compiler::IREE::HAL

extern "C" bool iree_register_compiler_plugin_hal_target_rocket(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<mlir::iree_compiler::IREE::HAL::RocketSession>(
      "hal_target_rocket");
  return true;
}

IREE_DEFINE_COMPILER_OPTION_FLAGS(
    mlir::iree_compiler::IREE::HAL::RocketOptions);
