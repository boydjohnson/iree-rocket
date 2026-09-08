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

// The whole shape of a unary element-wise task: geometry plus which ALU
// opcode. 'operand' is deliberately optional -- it is only meaningful for
// add_scalar, and requiring every other op to spell a zero would invite a
// producer to spell a nonzero one.
//
// There is no 'precision' key. The unary EW task shape is fp16 only, which
// is why the wire table has no precision field either; see the schema's own
// comment on ElementwiseUnaryDef.
constexpr std::array<const char *, 4> kRequiredElementwiseUnaryConfigKeys = {
    "width",
    "height",
    "channels",
    "op",
};

// The two-tensor form: geometry plus which operator. No 'precision' key,
// for a narrower reason than the unary kernel's -- EwAddShape does carry an
// int8 branch, but its EW_CVT_SCALE/OUT_CVT_SCALE ratio semantics are
// inferred rather than confirmed, and MUL has no int8 recipe in any capture.
constexpr std::array<const char *, 4> kRequiredElementwiseBinaryConfigKeys = {
    "width",
    "height",
    "channels",
    "op",
};

// Geometry, which curve, and the quantization that gets an input into the
// table's fixed domain. Also no 'precision' key: the LUT path is int8 by
// construction.
constexpr std::array<const char *, 8> kRequiredElementwiseLutConfigKeys = {
    "width",       "height",           "channels",  "fn",
    "input_scale", "output_scale",     "input_zero_point",
    "output_zero_point",
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
  std::vector<iree_hal_rocket_Conv2DQuantParam_enum_t> runtimeQuantization;
  // Symmetric spatial padding; see the parser for why it is optional and
  // what "symmetric" means on this hardware.
  uint32_t padTop = 0;
  uint32_t padLeft = 0;
  // Residual epilogue: one EW task adds a fourth binding to the conv's
  // output cube and applies `epilogueActivation` in the EW core. Optional
  // keys, like the padding, for the same reason.
  bool epilogueAdd = false;
  // One trailing push constant carries the compiler's count of Rocket
  // dispatches that read the result (Conv2DDef.runtime_dense_readers).
  bool runtimeDenseReaders = false;
  iree_hal_rocket_Activation_enum_t epilogueActivation =
      iree_hal_rocket_Activation_NONE;
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

struct RocketElementwiseUnaryConfig {
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t channels = 0;
  iree_hal_rocket_EwUnaryOp_enum_t op = iree_hal_rocket_EwUnaryOp_ABS;
  // IEEE-754 binary32 bit pattern; only meaningful for add_scalar.
  uint32_t operand = 0;
  std::vector<iree_hal_rocket_ElementwiseDimension_enum_t> runtimeDimensions;
};

struct RocketElementwiseBinaryConfig {
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t channels = 0;
  iree_hal_rocket_EwBinaryOp_enum_t op = iree_hal_rocket_EwBinaryOp_ADD;
  std::vector<iree_hal_rocket_ElementwiseDimension_enum_t> runtimeDimensions;
};

struct RocketElementwiseLutConfig {
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t channels = 0;
  iree_hal_rocket_LutFn_enum_t fn = iree_hal_rocket_LutFn_SIGMOID;
  // Decoded (real) zero points as a two's-complement int32 in a uint32 --
  // Conv2DQuantParam's documented convention. The runtime applies the 0x80
  // bias LutShape takes.
  uint32_t inputZeroPoint = 0;
  uint32_t outputZeroPoint = 0;
  float inputScale = 1.0f;
  float outputScale = 1.0f;
  std::vector<iree_hal_rocket_ElementwiseDimension_enum_t> runtimeDimensions;
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

  // Padding is optional, unlike every other conv key: it was added after the
  // 14 shipped targets were written and they do not spell it. Absent means
  // zero, which is what those targets have always serialized.
  //
  // The hardware applies these *symmetrically* -- `Shape::output_width` is
  // `w + 2 * pad_left`, matched against all 150 strided programs in the
  // vendor corpus -- so `pad_top` also pads the bottom and `pad_left` also
  // pads the right, exactly as `Conv2DDef` documents. There is no register
  // for a trailing-only pad: `CNA_PAD_CON0` has `pad_top` and `pad_left` and
  // nothing else, so an asymmetric pad cannot be expressed here at all and a
  // producer must leave it materialized.
  auto getOptionalU32 = [&](StringRef key) -> uint32_t {
    Attribute attr = config.get(key);
    if (!attr) {
      return 0;
    }
    return static_cast<uint32_t>(llvm::cast<IntegerAttr>(attr).getInt());
  };

  RocketConv2dConfig shape;
  shape.padTop = getOptionalU32("pad_top");
  shape.padLeft = getOptionalU32("pad_left");
  if (Attribute attr = config.get("epilogue_add")) {
    shape.epilogueAdd = llvm::cast<BoolAttr>(attr).getValue();
  }
  if (Attribute attr = config.get("runtime_dense_readers")) {
    auto boolAttr = llvm::dyn_cast<BoolAttr>(attr);
    if (!boolAttr) {
      diagFn() << "rocket backend: optional 'runtime_dense_readers' config "
                  "value must be a bool";
      return std::nullopt;
    }
    shape.runtimeDenseReaders = boolAttr.getValue();
  }
  if (Attribute attr = config.get("epilogue_activation")) {
    StringRef activation = llvm::cast<StringAttr>(attr).getValue();
    if (activation == "none") {
      shape.epilogueActivation = iree_hal_rocket_Activation_NONE;
    } else if (activation == "relu") {
      shape.epilogueActivation = iree_hal_rocket_Activation_RELU;
    } else {
      diagFn() << "rocket backend: unrecognized 'epilogue_activation' config "
                  "value '"
               << activation << "' (expected none/relu)";
      return std::nullopt;
    }
  }
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

  // Quantization parameters a dispatch may supply, listed separately from the
  // dimensions above. A convolution's scale and zero points are
  // per-convolution calibration data while an executable target is shared by
  // every dispatch that imports it, so a requantized int8 conv can only be
  // served by one executable if these arrive as push constants.
  //
  // The payload is a bit pattern rather than a number, because the schema
  // fields are uint32 and neither value fits that unsigned reading: the scale
  // is an IEEE-754 binary32 and a zero point is a signed int32. See
  // Conv2DQuantParam in rocket_executable_def.fbs, which is the one statement
  // of that convention; this serializer and rocket-hal-driver's
  // RuntimeConv2dQuantParam are its two implementations.
  std::array<bool, 3> isRuntimeQuantParam = {};
  if (Attribute runtimeQuantizationAttr = config.get("runtime_quantization")) {
    auto runtimeQuantization =
        llvm::dyn_cast<ArrayAttr>(runtimeQuantizationAttr);
    if (!runtimeQuantization) {
      diagFn() << "rocket backend: optional 'runtime_quantization' config "
                  "value must be an array of strings";
      return std::nullopt;
    }
    if (!runtimeQuantization.empty() &&
        shape.precision != iree_hal_rocket_Precision_INT8) {
      diagFn() << "rocket backend: 'runtime_quantization' is only meaningful "
                  "for requantized int8 convolution -- fp16 does not "
                  "requantize, and int8_accumulator bypasses the stage that "
                  "would consume these";
      return std::nullopt;
    }

    for (Attribute paramAttr : runtimeQuantization) {
      auto paramName = llvm::dyn_cast<StringAttr>(paramAttr);
      if (!paramName) {
        diagFn() << "rocket backend: every 'runtime_quantization' entry must "
                    "be a string";
        return std::nullopt;
      }

      std::optional<iree_hal_rocket_Conv2DQuantParam_enum_t> param;
      StringRef name = paramName.getValue();
      if (name == "output_scale") {
        param = iree_hal_rocket_Conv2DQuantParam_OUTPUT_SCALE;
      } else if (name == "input_zero_point") {
        param = iree_hal_rocket_Conv2DQuantParam_INPUT_ZERO_POINT;
      } else if (name == "output_zero_point") {
        param = iree_hal_rocket_Conv2DQuantParam_OUTPUT_ZERO_POINT;
      } else {
        // 'input_scale'/'weights_scale' are deliberately absent. They reach
        // the hardware only through pack_int8_bias_to_bs's bias
        // normalization and through their product with the output scale, and
        // the requantized int8 target keeps both at 1.0 so the single runtime
        // output scale carries the whole requantization ratio.
        diagFn() << "rocket backend: unknown runtime Conv2D quantization "
                    "parameter '"
                 << name << "'";
        return std::nullopt;
      }

      size_t paramIndex = static_cast<size_t>(*param);
      if (isRuntimeQuantParam[paramIndex]) {
        diagFn() << "rocket backend: duplicate runtime Conv2D quantization "
                    "parameter '"
                 << name << "'";
        return std::nullopt;
      }
      isRuntimeQuantParam[paramIndex] = true;
      shape.runtimeQuantization.push_back(*param);
    }
  }

  // The same template obligation the dimensions carry: a field a dispatch
  // supplies must be zero here, so a missing push constant cannot be mistaken
  // for calibration data. Zero is not a legal output scale, which is what
  // makes it usable as that field's sentinel -- unlike the zero points, where
  // zero is an ordinary value and only the list says who owns it.
  struct SettableQuantParam {
    iree_hal_rocket_Conv2DQuantParam_enum_t param;
    StringRef name;
    bool isZero;
  };
  const std::array<SettableQuantParam, 3> quantParams = {{
      {iree_hal_rocket_Conv2DQuantParam_OUTPUT_SCALE, "output_scale",
       shape.outputScale == 0.0f},
      {iree_hal_rocket_Conv2DQuantParam_INPUT_ZERO_POINT, "input_zero_point",
       shape.inputZeroPoint == 0},
      {iree_hal_rocket_Conv2DQuantParam_OUTPUT_ZERO_POINT, "output_zero_point",
       shape.outputZeroPoint == 0},
  }};
  for (const auto &[param, name, isZero] : quantParams) {
    const bool isRuntime = isRuntimeQuantParam[static_cast<size_t>(param)];
    if (isRuntime && !isZero) {
      diagFn() << "rocket backend: runtime Conv2D quantization parameter '"
               << name << "' must use 0 as its executable template value";
      return std::nullopt;
    }
    if (!isRuntime && param == iree_hal_rocket_Conv2DQuantParam_OUTPUT_SCALE &&
        isZero) {
      diagFn() << "rocket backend: 'output_scale' is zero and not listed in "
                  "'runtime_quantization'";
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

// Both element-wise tables carry the same three-entry dimension list, so
// they parse it the same way. Returns false on any diagnostic already
// emitted.
bool parseElementwiseRuntimeDimensions(
    DictionaryAttr config,
    std::vector<iree_hal_rocket_ElementwiseDimension_enum_t> &into,
    std::array<bool, 3> &isRuntimeDimension,
    llvm::function_ref<InFlightDiagnostic()> diagFn) {
  Attribute runtimeDimensionsAttr = config.get("runtime_dimensions");
  if (!runtimeDimensionsAttr) {
    return true;
  }
  auto runtimeDimensions = llvm::dyn_cast<ArrayAttr>(runtimeDimensionsAttr);
  if (!runtimeDimensions) {
    diagFn() << "rocket backend: optional 'runtime_dimensions' config value "
                "must be an array of strings";
    return false;
  }
  for (Attribute dimensionAttr : runtimeDimensions) {
    auto dimensionName = llvm::dyn_cast<StringAttr>(dimensionAttr);
    if (!dimensionName) {
      diagFn() << "rocket backend: every 'runtime_dimensions' entry must be "
                  "a string";
      return false;
    }
    std::optional<iree_hal_rocket_ElementwiseDimension_enum_t> dimension;
    StringRef name = dimensionName.getValue();
    if (name == "width") {
      dimension = iree_hal_rocket_ElementwiseDimension_WIDTH;
    } else if (name == "height") {
      dimension = iree_hal_rocket_ElementwiseDimension_HEIGHT;
    } else if (name == "channels") {
      dimension = iree_hal_rocket_ElementwiseDimension_CHANNELS;
    } else {
      diagFn() << "rocket backend: unknown runtime element-wise dimension '"
               << name << "'";
      return false;
    }
    size_t dimensionIndex = static_cast<size_t>(*dimension);
    if (isRuntimeDimension[dimensionIndex]) {
      diagFn() << "rocket backend: duplicate runtime element-wise dimension '"
               << name << "'";
      return false;
    }
    isRuntimeDimension[dimensionIndex] = true;
    into.push_back(*dimension);
  }
  return true;
}

// The template/runtime contract, identical for both element-wise kinds: a
// listed dimension must be zero here and arrive per dispatch, an unlisted
// one must be nonzero. rocket-hal-driver's
// `validate_elementwise_template` is the runtime half of the same rule.
bool checkElementwiseDimensions(
    const std::array<bool, 3> &isRuntimeDimension, uint32_t width,
    uint32_t height, uint32_t channels,
    llvm::function_ref<InFlightDiagnostic()> diagFn) {
  struct SettableDimension {
    iree_hal_rocket_ElementwiseDimension_enum_t dimension;
    StringRef name;
    uint32_t value;
  };
  const std::array<SettableDimension, 3> dimensions = {{
      {iree_hal_rocket_ElementwiseDimension_WIDTH, "width", width},
      {iree_hal_rocket_ElementwiseDimension_HEIGHT, "height", height},
      {iree_hal_rocket_ElementwiseDimension_CHANNELS, "channels", channels},
  }};
  for (const auto &[dimension, name, value] : dimensions) {
    const bool isRuntime = isRuntimeDimension[static_cast<size_t>(dimension)];
    if (isRuntime && value != 0) {
      diagFn() << "rocket backend: runtime element-wise dimension '" << name
               << "' must use 0 as its executable template value";
      return false;
    }
    if (!isRuntime && value == 0) {
      diagFn() << "rocket backend: zero element-wise dimension '" << name
               << "' must be listed in 'runtime_dimensions'";
      return false;
    }
  }
  return true;
}

std::optional<RocketElementwiseUnaryConfig>
buildRocketElementwiseUnaryConfigFromTarget(
    DictionaryAttr config, llvm::function_ref<InFlightDiagnostic()> diagFn) {
  if (!config) {
    diagFn() << "rocket element-wise backend requires a non-empty executable "
                "target config dict";
    return std::nullopt;
  }
  for (const char *key : kRequiredElementwiseUnaryConfigKeys) {
    if (!config.get(key)) {
      diagFn() << "rocket element-wise executable target config is missing "
                  "required key '"
               << key << "'";
      return std::nullopt;
    }
  }

  auto getU32 = [&](StringRef key) -> uint32_t {
    return static_cast<uint32_t>(
        llvm::cast<IntegerAttr>(config.get(key)).getInt());
  };

  RocketElementwiseUnaryConfig shape;
  shape.width = getU32("width");
  shape.height = getU32("height");
  shape.channels = getU32("channels");

  StringRef op = llvm::cast<StringAttr>(config.get("op")).getValue();
  if (op == "abs") {
    shape.op = iree_hal_rocket_EwUnaryOp_ABS;
  } else if (op == "neg") {
    shape.op = iree_hal_rocket_EwUnaryOp_NEG;
  } else if (op == "floor") {
    shape.op = iree_hal_rocket_EwUnaryOp_FLOOR;
  } else if (op == "ceil") {
    shape.op = iree_hal_rocket_EwUnaryOp_CEIL;
  } else if (op == "add_scalar") {
    shape.op = iree_hal_rocket_EwUnaryOp_ADD_SCALAR;
  } else {
    diagFn() << "rocket backend: unrecognized element-wise unary 'op' config "
                "value '"
             << op << "' (expected abs/neg/floor/ceil/add_scalar)";
    return std::nullopt;
  }

  if (Attribute operandAttr = config.get("operand")) {
    auto operand = llvm::dyn_cast<IntegerAttr>(operandAttr);
    if (!operand) {
      diagFn() << "rocket backend: 'operand' must be an integer holding an "
                  "IEEE-754 binary32 bit pattern";
      return std::nullopt;
    }
    shape.operand = static_cast<uint32_t>(operand.getInt());
  }
  // The runtime refuses this too, because `build_unary_regcmd` asserts it.
  // Catching it here turns a runtime rejection into a compile error.
  if (shape.operand != 0 &&
      shape.op != iree_hal_rocket_EwUnaryOp_ADD_SCALAR) {
    diagFn() << "rocket backend: 'operand' is only meaningful for the "
                "add_scalar element-wise op";
    return std::nullopt;
  }

  std::array<bool, 3> isRuntimeDimension = {};
  if (!parseElementwiseRuntimeDimensions(config, shape.runtimeDimensions,
                                         isRuntimeDimension, diagFn)) {
    return std::nullopt;
  }
  if (!checkElementwiseDimensions(isRuntimeDimension, shape.width,
                                  shape.height, shape.channels, diagFn)) {
    return std::nullopt;
  }
  return shape;
}

std::optional<RocketElementwiseBinaryConfig>
buildRocketElementwiseBinaryConfigFromTarget(
    DictionaryAttr config, llvm::function_ref<InFlightDiagnostic()> diagFn) {
  if (!config) {
    diagFn() << "rocket element-wise backend requires a non-empty executable "
                "target config dict";
    return std::nullopt;
  }
  for (const char *key : kRequiredElementwiseBinaryConfigKeys) {
    if (!config.get(key)) {
      diagFn() << "rocket element-wise executable target config is missing "
                  "required key '"
               << key << "'";
      return std::nullopt;
    }
  }

  auto getU32 = [&](StringRef key) -> uint32_t {
    return static_cast<uint32_t>(
        llvm::cast<IntegerAttr>(config.get(key)).getInt());
  };

  RocketElementwiseBinaryConfig shape;
  shape.width = getU32("width");
  shape.height = getU32("height");
  shape.channels = getU32("channels");

  StringRef op = llvm::cast<StringAttr>(config.get("op")).getValue();
  if (op == "add") {
    shape.op = iree_hal_rocket_EwBinaryOp_ADD;
  } else if (op == "sub") {
    shape.op = iree_hal_rocket_EwBinaryOp_SUB;
  } else if (op == "mul") {
    shape.op = iree_hal_rocket_EwBinaryOp_MUL;
  } else if (op == "max") {
    shape.op = iree_hal_rocket_EwBinaryOp_MAX;
  } else if (op == "min") {
    shape.op = iree_hal_rocket_EwBinaryOp_MIN;
  } else {
    // 'div' is deliberately not here: ew_alu_algo=3 is the one
    // TRM-documented binary opcode with no hardware evidence in this
    // project, and the wire enum does not carry it either.
    diagFn() << "rocket backend: unrecognized element-wise binary 'op' config "
                "value '"
             << op << "' (expected add/sub/mul/max/min)";
    return std::nullopt;
  }

  std::array<bool, 3> isRuntimeDimension = {};
  if (!parseElementwiseRuntimeDimensions(config, shape.runtimeDimensions,
                                         isRuntimeDimension, diagFn)) {
    return std::nullopt;
  }
  if (!checkElementwiseDimensions(isRuntimeDimension, shape.width,
                                  shape.height, shape.channels, diagFn)) {
    return std::nullopt;
  }
  return shape;
}

std::optional<RocketElementwiseLutConfig>
buildRocketElementwiseLutConfigFromTarget(
    DictionaryAttr config, llvm::function_ref<InFlightDiagnostic()> diagFn) {
  if (!config) {
    diagFn() << "rocket LUT backend requires a non-empty executable target "
                "config dict";
    return std::nullopt;
  }
  for (const char *key : kRequiredElementwiseLutConfigKeys) {
    if (!config.get(key)) {
      diagFn() << "rocket LUT executable target config is missing required "
                  "key '"
               << key << "'";
      return std::nullopt;
    }
  }

  auto getU32 = [&](StringRef key) -> uint32_t {
    return static_cast<uint32_t>(
        llvm::cast<IntegerAttr>(config.get(key)).getInt());
  };
  auto getI64 = [&](StringRef key) -> int64_t {
    return llvm::cast<IntegerAttr>(config.get(key)).getInt();
  };
  auto getF32 = [&](StringRef key) -> float {
    return llvm::cast<FloatAttr>(config.get(key)).getValueAsDouble();
  };

  RocketElementwiseLutConfig shape;
  shape.width = getU32("width");
  shape.height = getU32("height");
  shape.channels = getU32("channels");

  StringRef fn = llvm::cast<StringAttr>(config.get("fn")).getValue();
  if (fn == "sigmoid") {
    shape.fn = iree_hal_rocket_LutFn_SIGMOID;
  } else if (fn == "tanh") {
    shape.fn = iree_hal_rocket_LutFn_TANH;
  } else if (fn == "exp") {
    shape.fn = iree_hal_rocket_LutFn_EXP;
  } else if (fn == "square") {
    shape.fn = iree_hal_rocket_LutFn_SQUARE;
  } else if (fn == "erf") {
    shape.fn = iree_hal_rocket_LutFn_ERF;
  } else if (fn == "sqrt") {
    shape.fn = iree_hal_rocket_LutFn_SQRT;
  } else if (fn == "rsqrt") {
    shape.fn = iree_hal_rocket_LutFn_RSQRT;
  } else if (fn == "log") {
    shape.fn = iree_hal_rocket_LutFn_LOG;
  } else if (fn == "reciprocal") {
    shape.fn = iree_hal_rocket_LutFn_RECIPROCAL;
  } else {
    diagFn() << "rocket backend: unrecognized LUT 'fn' config value '" << fn
             << "'";
    return std::nullopt;
  }

  // Only -128, -2, 0 and 127 have a confirmed BN_ALU operand;
  // `build_lut_regcmd` asserts on the rest and the runtime refuses them, so
  // this is the compile-time half of that same rule.
  int64_t inputZeroPoint = getI64("input_zero_point");
  int64_t outputZeroPoint = getI64("output_zero_point");
  if (inputZeroPoint != -128 && inputZeroPoint != -2 && inputZeroPoint != 0 &&
      inputZeroPoint != 127) {
    diagFn() << "rocket backend: LUT 'input_zero_point' " << inputZeroPoint
             << " has no confirmed BN_ALU operand (expected -128, -2, 0 or "
                "127)";
    return std::nullopt;
  }
  if (outputZeroPoint < -128 || outputZeroPoint > 127) {
    diagFn() << "rocket backend: LUT 'output_zero_point' " << outputZeroPoint
             << " does not fit an int8";
    return std::nullopt;
  }
  shape.inputZeroPoint = static_cast<uint32_t>(
      static_cast<int32_t>(inputZeroPoint));
  shape.outputZeroPoint = static_cast<uint32_t>(
      static_cast<int32_t>(outputZeroPoint));

  shape.inputScale = getF32("input_scale");
  shape.outputScale = getF32("output_scale");
  if (!(shape.inputScale > 0.0f) || !(shape.outputScale > 0.0f)) {
    diagFn() << "rocket backend: LUT scales must be finite and positive";
    return std::nullopt;
  }

  std::array<bool, 3> isRuntimeDimension = {};
  if (!parseElementwiseRuntimeDimensions(config, shape.runtimeDimensions,
                                         isRuntimeDimension, diagFn)) {
    return std::nullopt;
  }
  if (!checkElementwiseDimensions(isRuntimeDimension, shape.width,
                                  shape.height, shape.channels, diagFn)) {
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
        kernel != "pooling" && kernel != "matmul" &&
        kernel != "elementwise_unary" && kernel != "elementwise_lut" &&
        kernel != "elementwise_binary") {
      return variantOp.emitOpError()
             << "unsupported Rocket kernel '" << kernel
             << "'; expected 'conv2d', 'pooling', 'matmul', "
                "'elementwise_unary', 'elementwise_binary', "
                "'elementwise_lut' or the "
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
    std::optional<RocketElementwiseUnaryConfig> ewUnaryShape;
    std::optional<RocketElementwiseBinaryConfig> ewBinaryShape;
    std::optional<RocketElementwiseLutConfig> lutShape;
    if (kernel == "fully_connected") {
      fcShape = buildRocketFullyConnectedConfigFromTarget(config, diagFn);
    } else if (kernel == "pooling") {
      poolingShape = buildRocketPoolingConfigFromTarget(config, diagFn);
    } else if (kernel == "matmul") {
      matmulShape = buildRocketMatmulConfigFromTarget(config, diagFn);
    } else if (kernel == "elementwise_unary") {
      ewUnaryShape = buildRocketElementwiseUnaryConfigFromTarget(config, diagFn);
    } else if (kernel == "elementwise_binary") {
      ewBinaryShape =
          buildRocketElementwiseBinaryConfigFromTarget(config, diagFn);
    } else if (kernel == "elementwise_lut") {
      lutShape = buildRocketElementwiseLutConfigFromTarget(config, diagFn);
    } else {
      convShape = buildRocketConv2dConfigFromTarget(config, diagFn);
    }
    if (!convShape && !fcShape && !poolingShape && !matmulShape &&
        !ewUnaryShape && !ewBinaryShape && !lutShape) {
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
    } else if (ewUnaryShape) {
      runtimeDimensionCount = ewUnaryShape->runtimeDimensions.size();
    } else if (ewBinaryShape) {
      runtimeDimensionCount = ewBinaryShape->runtimeDimensions.size();
    } else if (lutShape) {
      runtimeDimensionCount = lutShape->runtimeDimensions.size();
    }
    // Only Conv2DDef carries runtime quantization; pooling has none to carry
    // and matmul's own lowering does not use the dispatch-supplied scale.
    size_t runtimeQuantizationCount =
        convShape ? convShape->runtimeQuantization.size() : 0;
    // Dimensions first, then quantization parameters -- one flat push-constant
    // sequence in that order, which is the order rocket-hal-driver's
    // `Conv2dExecutable::resolve_shape` consumes them in.
    // And last, the dense-reader count, when the convolution target asks for
    // one (`runtime_dense_readers`); the driver reads it after the others.
    size_t denseReaderCount =
        convShape && convShape->runtimeDenseReaders ? 1 : 0;
    size_t runtimeConstantCount =
        runtimeDimensionCount + runtimeQuantizationCount + denseReaderCount;
    if (pipelineConstantCount != static_cast<int64_t>(runtimeConstantCount)) {
      return exportOp.emitOpError()
             << "Rocket pipeline layout declares " << pipelineConstantCount
             << " push constants, but the executable target declares "
             << runtimeDimensionCount << " runtime dimensions, "
             << runtimeQuantizationCount << " runtime quantization parameters"
             << " and " << denseReaderCount << " dense-reader count";
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
    } else if (ewUnaryShape) {
      iree_hal_rocket_ElementwiseDimension_vec_ref_t runtimeDimensionsRef = 0;
      if (!ewUnaryShape->runtimeDimensions.empty()) {
        runtimeDimensionsRef = iree_hal_rocket_ElementwiseDimension_vec_create(
            builder, ewUnaryShape->runtimeDimensions.data(),
            ewUnaryShape->runtimeDimensions.size());
        if (!runtimeDimensionsRef) {
          return variantOp.emitOpError()
                 << "failed to build Rocket element-wise runtime-dimension "
                    "vector";
        }
      }
      if (iree_hal_rocket_ElementwiseUnaryDef_start(builder) ||
          iree_hal_rocket_ElementwiseUnaryDef_width_add(builder,
                                                        ewUnaryShape->width) ||
          iree_hal_rocket_ElementwiseUnaryDef_height_add(
              builder, ewUnaryShape->height) ||
          iree_hal_rocket_ElementwiseUnaryDef_channels_add(
              builder, ewUnaryShape->channels) ||
          iree_hal_rocket_ElementwiseUnaryDef_op_add(builder,
                                                     ewUnaryShape->op) ||
          iree_hal_rocket_ElementwiseUnaryDef_operand_add(
              builder, ewUnaryShape->operand) ||
          (runtimeDimensionsRef &&
           iree_hal_rocket_ElementwiseUnaryDef_runtime_dimensions_add(
               builder, runtimeDimensionsRef))) {
        return variantOp.emitOpError()
               << "failed to build Rocket element-wise unary definition";
      }
      auto ewRef = iree_hal_rocket_ElementwiseUnaryDef_end(builder);
      if (!ewRef) {
        return variantOp.emitOpError()
               << "failed to finish Rocket element-wise unary definition";
      }
      kernelRef = iree_hal_rocket_KernelDef_as_ElementwiseUnaryDef(ewRef);
    } else if (ewBinaryShape) {
      iree_hal_rocket_ElementwiseDimension_vec_ref_t runtimeDimensionsRef = 0;
      if (!ewBinaryShape->runtimeDimensions.empty()) {
        runtimeDimensionsRef = iree_hal_rocket_ElementwiseDimension_vec_create(
            builder, ewBinaryShape->runtimeDimensions.data(),
            ewBinaryShape->runtimeDimensions.size());
        if (!runtimeDimensionsRef) {
          return variantOp.emitOpError()
                 << "failed to build Rocket element-wise runtime-dimension "
                    "vector";
        }
      }
      if (iree_hal_rocket_ElementwiseBinaryDef_start(builder) ||
          iree_hal_rocket_ElementwiseBinaryDef_width_add(
              builder, ewBinaryShape->width) ||
          iree_hal_rocket_ElementwiseBinaryDef_height_add(
              builder, ewBinaryShape->height) ||
          iree_hal_rocket_ElementwiseBinaryDef_channels_add(
              builder, ewBinaryShape->channels) ||
          iree_hal_rocket_ElementwiseBinaryDef_op_add(builder,
                                                      ewBinaryShape->op) ||
          (runtimeDimensionsRef &&
           iree_hal_rocket_ElementwiseBinaryDef_runtime_dimensions_add(
               builder, runtimeDimensionsRef))) {
        return variantOp.emitOpError()
               << "failed to build Rocket element-wise binary definition";
      }
      auto ewRef = iree_hal_rocket_ElementwiseBinaryDef_end(builder);
      if (!ewRef) {
        return variantOp.emitOpError()
               << "failed to finish Rocket element-wise binary definition";
      }
      kernelRef = iree_hal_rocket_KernelDef_as_ElementwiseBinaryDef(ewRef);
    } else if (lutShape) {
      iree_hal_rocket_ElementwiseDimension_vec_ref_t runtimeDimensionsRef = 0;
      if (!lutShape->runtimeDimensions.empty()) {
        runtimeDimensionsRef = iree_hal_rocket_ElementwiseDimension_vec_create(
            builder, lutShape->runtimeDimensions.data(),
            lutShape->runtimeDimensions.size());
        if (!runtimeDimensionsRef) {
          return variantOp.emitOpError()
                 << "failed to build Rocket LUT runtime-dimension vector";
        }
      }
      if (iree_hal_rocket_ElementwiseLutDef_start(builder) ||
          iree_hal_rocket_ElementwiseLutDef_width_add(builder,
                                                      lutShape->width) ||
          iree_hal_rocket_ElementwiseLutDef_height_add(builder,
                                                       lutShape->height) ||
          iree_hal_rocket_ElementwiseLutDef_channels_add(builder,
                                                         lutShape->channels) ||
          iree_hal_rocket_ElementwiseLutDef_fn_add(builder, lutShape->fn) ||
          iree_hal_rocket_ElementwiseLutDef_input_zero_point_add(
              builder, lutShape->inputZeroPoint) ||
          iree_hal_rocket_ElementwiseLutDef_output_zero_point_add(
              builder, lutShape->outputZeroPoint) ||
          iree_hal_rocket_ElementwiseLutDef_input_scale_add(
              builder, lutShape->inputScale) ||
          iree_hal_rocket_ElementwiseLutDef_output_scale_add(
              builder, lutShape->outputScale) ||
          (runtimeDimensionsRef &&
           iree_hal_rocket_ElementwiseLutDef_runtime_dimensions_add(
               builder, runtimeDimensionsRef))) {
        return variantOp.emitOpError()
               << "failed to build Rocket LUT definition";
      }
      auto lutRef = iree_hal_rocket_ElementwiseLutDef_end(builder);
      if (!lutRef) {
        return variantOp.emitOpError()
               << "failed to finish Rocket LUT definition";
      }
      kernelRef = iree_hal_rocket_KernelDef_as_ElementwiseLutDef(lutRef);
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
      iree_hal_rocket_Conv2DQuantParam_vec_ref_t runtimeQuantizationRef = 0;
      if (!convShape->runtimeQuantization.empty()) {
        runtimeQuantizationRef = iree_hal_rocket_Conv2DQuantParam_vec_create(
            builder, convShape->runtimeQuantization.data(),
            convShape->runtimeQuantization.size());
        if (!runtimeQuantizationRef) {
          return variantOp.emitOpError()
                 << "failed to build Rocket runtime-quantization vector";
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
          iree_hal_rocket_Conv2DDef_pad_top_add(builder, convShape->padTop) ||
          iree_hal_rocket_Conv2DDef_pad_left_add(builder,
                                                 convShape->padLeft) ||
          (runtimeQuantizationRef &&
           iree_hal_rocket_Conv2DDef_runtime_quantization_add(
               builder, runtimeQuantizationRef)) ||
          iree_hal_rocket_Conv2DDef_epilogue_add_add(builder,
                                                     convShape->epilogueAdd) ||
          iree_hal_rocket_Conv2DDef_epilogue_activation_add(
              builder, convShape->epilogueActivation) ||
          iree_hal_rocket_Conv2DDef_runtime_dense_readers_add(
              builder, convShape->runtimeDenseReaders)) {
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
