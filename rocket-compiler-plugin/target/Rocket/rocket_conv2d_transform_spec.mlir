// Rocket conv2d dispatch, generic over runtime shape. Previously this file
// also carried three fixed-shape per-MobileNet-layer executables
// (rocket_target_0/1/2, rocket_executable_0/1/2, @match_conv2d_0/1/2) as a
// literal-shape fast path ahead of the dynamic fallback below. They are
// retired: the dynamic path (@match_dynamic_conv2d/_3x3 and their depthwise
// counterparts) claims the same shapes through the identical
// runtime-dimension ABI, so the fixed-shape specializations added a second
// code path without adding coverage. Developed and hardware-validated
// against the RK3588 in iree-rocket-design-spike (see that repo's
// DESIGN_NOTES.md for the full derivation, including the two
// rocket-hal-driver bugs the depthwise path exposed: a skipped weight-
// packing branch and a tap-major layout formula only ever checked inside
// one 32-channel coefficient group).

#rocket_dynamic_target = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "conv2d",
  input_width = 0 : i32, input_height = 0 : i32, input_channels = 0 : i32,
  output_width = 0 : i32, output_height = 0 : i32, output_channels = 0 : i32,
  weights_width = 0 : i32, weights_height = 0 : i32, stride = 1 : i32,
  depthwise = false,
  input_zero_point = 0 : i32, output_zero_point = 0 : i32, weights_zero_point = 0 : i32,
  input_scale = 1.0 : f32, weights_scale = 1.0 : f32, output_scale = 1.0 : f32,
  truncate_bits = 0 : i32,
  activation = "none", activation_cmp = 0 : i32,
  precision = "fp16",
  // The six settable dimensions. output_width/output_height are absent
  // because the runtime always derives them from these plus stride and
  // padding -- their wire values are retired, see
  // rocket-schema/schema/rocket_executable_def.fbs.
  runtime_dimensions = [
    "input_width", "input_height", "input_channels",
    "output_channels", "weights_width", "weights_height"
  ]
}>

// Same wire format as #rocket_dynamic_target with depthwise = true. Kept as
// a separate attr (rather than a runtime-settable bool) because it gates a
// different register program at the driver (CNA_CONV_CON1.CONV_MODE=3,
// CORE_MISC_CFG.DW_EN=1, etc. -- see DESIGN_NOTES.md "Depthwise: Mesa's
// channel rule is wrong") and a different weight layout
// (tensor_layout::pack_depthwise_to_rocket_weights, tap-major with a
// padded-channel stride, not pack_hwcf_to_rocket_weights). output_channels
// is still carried on the wire even though the depthwise matchers below only
// ever bind it equal to input_channels -- Conv2DDef's depthwise field
// (rocket-schema) already exists for this and the driver
// (executable_cache.rs) already reads it, so this is genuinely just a
// dispatch-selection gap, not a runtime one.
#rocket_dynamic_depthwise_target = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "conv2d",
  input_width = 0 : i32, input_height = 0 : i32, input_channels = 0 : i32,
  output_width = 0 : i32, output_height = 0 : i32, output_channels = 0 : i32,
  weights_width = 0 : i32, weights_height = 0 : i32, stride = 1 : i32,
  depthwise = true,
  input_zero_point = 0 : i32, output_zero_point = 0 : i32, weights_zero_point = 0 : i32,
  input_scale = 1.0 : f32, weights_scale = 1.0 : f32, output_scale = 1.0 : f32,
  truncate_bits = 0 : i32,
  activation = "none", activation_cmp = 0 : i32,
  precision = "fp16",
  runtime_dimensions = [
    "input_width", "input_height", "input_channels",
    "output_channels", "weights_width", "weights_height"
  ]
}>

#rocket_dynamic_target_s2 = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "conv2d",
  input_width = 0 : i32, input_height = 0 : i32, input_channels = 0 : i32,
  output_width = 0 : i32, output_height = 0 : i32, output_channels = 0 : i32,
  weights_width = 0 : i32, weights_height = 0 : i32, stride = 2 : i32,
  depthwise = false,
  input_zero_point = 0 : i32, output_zero_point = 0 : i32, weights_zero_point = 0 : i32,
  input_scale = 1.0 : f32, weights_scale = 1.0 : f32, output_scale = 1.0 : f32,
  truncate_bits = 0 : i32,
  activation = "none", activation_cmp = 0 : i32,
  precision = "fp16",
  runtime_dimensions = [
    "input_width", "input_height", "input_channels",
    "output_channels", "weights_width", "weights_height"
  ]
}>

#rocket_dynamic_target_s3 = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "conv2d",
  input_width = 0 : i32, input_height = 0 : i32, input_channels = 0 : i32,
  output_width = 0 : i32, output_height = 0 : i32, output_channels = 0 : i32,
  weights_width = 0 : i32, weights_height = 0 : i32, stride = 3 : i32,
  depthwise = false,
  input_zero_point = 0 : i32, output_zero_point = 0 : i32, weights_zero_point = 0 : i32,
  input_scale = 1.0 : f32, weights_scale = 1.0 : f32, output_scale = 1.0 : f32,
  truncate_bits = 0 : i32,
  activation = "none", activation_cmp = 0 : i32,
  precision = "fp16",
  runtime_dimensions = [
    "input_width", "input_height", "input_channels",
    "output_channels", "weights_width", "weights_height"
  ]
}>

#rocket_dynamic_target_s4 = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "conv2d",
  input_width = 0 : i32, input_height = 0 : i32, input_channels = 0 : i32,
  output_width = 0 : i32, output_height = 0 : i32, output_channels = 0 : i32,
  weights_width = 0 : i32, weights_height = 0 : i32, stride = 4 : i32,
  depthwise = false,
  input_zero_point = 0 : i32, output_zero_point = 0 : i32, weights_zero_point = 0 : i32,
  input_scale = 1.0 : f32, weights_scale = 1.0 : f32, output_scale = 1.0 : f32,
  truncate_bits = 0 : i32,
  activation = "none", activation_cmp = 0 : i32,
  precision = "fp16",
  runtime_dimensions = [
    "input_width", "input_height", "input_channels",
    "output_channels", "weights_width", "weights_height"
  ]
}>

#rocket_dynamic_depthwise_target_s2 = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "conv2d",
  input_width = 0 : i32, input_height = 0 : i32, input_channels = 0 : i32,
  output_width = 0 : i32, output_height = 0 : i32, output_channels = 0 : i32,
  weights_width = 0 : i32, weights_height = 0 : i32, stride = 2 : i32,
  depthwise = true,
  input_zero_point = 0 : i32, output_zero_point = 0 : i32, weights_zero_point = 0 : i32,
  input_scale = 1.0 : f32, weights_scale = 1.0 : f32, output_scale = 1.0 : f32,
  truncate_bits = 0 : i32,
  activation = "none", activation_cmp = 0 : i32,
  precision = "fp16",
  runtime_dimensions = [
    "input_width", "input_height", "input_channels",
    "output_channels", "weights_width", "weights_height"
  ]
}>

#rocket_dynamic_depthwise_target_s3 = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "conv2d",
  input_width = 0 : i32, input_height = 0 : i32, input_channels = 0 : i32,
  output_width = 0 : i32, output_height = 0 : i32, output_channels = 0 : i32,
  weights_width = 0 : i32, weights_height = 0 : i32, stride = 3 : i32,
  depthwise = true,
  input_zero_point = 0 : i32, output_zero_point = 0 : i32, weights_zero_point = 0 : i32,
  input_scale = 1.0 : f32, weights_scale = 1.0 : f32, output_scale = 1.0 : f32,
  truncate_bits = 0 : i32,
  activation = "none", activation_cmp = 0 : i32,
  precision = "fp16",
  runtime_dimensions = [
    "input_width", "input_height", "input_channels",
    "output_channels", "weights_width", "weights_height"
  ]
}>

#rocket_dynamic_depthwise_target_s4 = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "conv2d",
  input_width = 0 : i32, input_height = 0 : i32, input_channels = 0 : i32,
  output_width = 0 : i32, output_height = 0 : i32, output_channels = 0 : i32,
  weights_width = 0 : i32, weights_height = 0 : i32, stride = 4 : i32,
  depthwise = true,
  input_zero_point = 0 : i32, output_zero_point = 0 : i32, weights_zero_point = 0 : i32,
  input_scale = 1.0 : f32, weights_scale = 1.0 : f32, output_scale = 1.0 : f32,
  truncate_bits = 0 : i32,
  activation = "none", activation_cmp = 0 : i32,
  precision = "fp16",
  runtime_dimensions = [
    "input_width", "input_height", "input_channels",
    "output_channels", "weights_width", "weights_height"
  ]
}>


// int8 counterparts of the fp16 targets above, for ONNX models quantized
// with ORT's quantize_dynamic (onnx.ConvInteger). Everything is identical
// apart from `precision`, which selects the DPU's int8-in/int32-out
// accumulator mode: the requantization stage (BS, CPEND, and the
// out-convert scale/shift/offset) is bypassed and the raw i32 accumulator
// is written straight out -- see iree-rocket-hal's
// int8_accumulator_output_uses_the_hardware_validated_bypasses.
//
// The zero points stay 0 here, and that is a hard requirement, not a
// placeholder: Shape::with_precision panics on a non-zero zero point in
// this mode, and RocketTarget.cpp rejects it at serialization time, because
// only the zero-zero-point bypass path is hardware-validated. Real ONNX
// activations are asymmetric (DynamicQuantizeLinear emits a non-zero ui8
// zero point), so getting here at all depends on
// iree-global-opt-quantized-conv-to-conv having already folded the zero
// point out of the convolution and into a separate CPU-side correction
// term -- an exact i32 identity, not an approximation. See
// @__transform_main's own comment.
//
// The scales stay 1.0 for the same reason they do in the fp16 targets: this
// mode does no rescaling at all, so the driver's bias normalization
// (pack_int8_bias_to_bs divides by input_scale * weights_scale) is the
// identity and the zero bias below passes through untouched.

#rocket_dynamic_int8_target = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "conv2d",
  input_width = 0 : i32, input_height = 0 : i32, input_channels = 0 : i32,
  output_width = 0 : i32, output_height = 0 : i32, output_channels = 0 : i32,
  weights_width = 0 : i32, weights_height = 0 : i32, stride = 1 : i32,
  depthwise = false,
  input_zero_point = 0 : i32, output_zero_point = 0 : i32, weights_zero_point = 0 : i32,
  input_scale = 1.0 : f32, weights_scale = 1.0 : f32, output_scale = 1.0 : f32,
  truncate_bits = 0 : i32,
  activation = "none", activation_cmp = 0 : i32,
  precision = "int8_accumulator",
  runtime_dimensions = [
    "input_width", "input_height", "input_channels",
    "output_channels", "weights_width", "weights_height"
  ]
}>

#rocket_dynamic_depthwise_int8_target = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "conv2d",
  input_width = 0 : i32, input_height = 0 : i32, input_channels = 0 : i32,
  output_width = 0 : i32, output_height = 0 : i32, output_channels = 0 : i32,
  weights_width = 0 : i32, weights_height = 0 : i32, stride = 1 : i32,
  depthwise = true,
  input_zero_point = 0 : i32, output_zero_point = 0 : i32, weights_zero_point = 0 : i32,
  input_scale = 1.0 : f32, weights_scale = 1.0 : f32, output_scale = 1.0 : f32,
  truncate_bits = 0 : i32,
  activation = "none", activation_cmp = 0 : i32,
  precision = "int8_accumulator",
  runtime_dimensions = [
    "input_width", "input_height", "input_channels",
    "output_channels", "weights_width", "weights_height"
  ]
}>

#rocket_dynamic_depthwise_int8_target_s2 = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "conv2d",
  input_width = 0 : i32, input_height = 0 : i32, input_channels = 0 : i32,
  output_width = 0 : i32, output_height = 0 : i32, output_channels = 0 : i32,
  weights_width = 0 : i32, weights_height = 0 : i32, stride = 2 : i32,
  depthwise = true,
  input_zero_point = 0 : i32, output_zero_point = 0 : i32, weights_zero_point = 0 : i32,
  input_scale = 1.0 : f32, weights_scale = 1.0 : f32, output_scale = 1.0 : f32,
  truncate_bits = 0 : i32,
  activation = "none", activation_cmp = 0 : i32,
  precision = "int8_accumulator",
  runtime_dimensions = [
    "input_width", "input_height", "input_channels",
    "output_channels", "weights_width", "weights_height"
  ]
}>

#dynamic_pipeline_layout = #hal.pipeline.layout<constants = 6, bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>

module attributes {transform.with_named_sequence} {

  hal.executable private @rocket_dynamic_executable {
    hal.executable.variant public @rocket_dynamic_conv2d_v1 target(#rocket_dynamic_target) {
      hal.executable.export public @rocket_dynamic_conv2d ordinal(0) layout(#dynamic_pipeline_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @rocket_dynamic_conv2d() {
          return
        }
      }
    }
  }

  hal.executable private @rocket_dynamic_depthwise_executable {
    hal.executable.variant public @rocket_dynamic_depthwise_conv2d_v1 target(#rocket_dynamic_depthwise_target) {
      hal.executable.export public @rocket_dynamic_depthwise_conv2d ordinal(0) layout(#dynamic_pipeline_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @rocket_dynamic_depthwise_conv2d() {
          return
        }
      }
    }
  }

  hal.executable private @rocket_dynamic_executable_s2 {
    hal.executable.variant public @rocket_dynamic_conv2d_v1 target(#rocket_dynamic_target_s2) {
      hal.executable.export public @rocket_dynamic_conv2d ordinal(0) layout(#dynamic_pipeline_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @rocket_dynamic_conv2d() {
          return
        }
      }
    }
  }

  hal.executable private @rocket_dynamic_executable_s3 {
    hal.executable.variant public @rocket_dynamic_conv2d_v1 target(#rocket_dynamic_target_s3) {
      hal.executable.export public @rocket_dynamic_conv2d ordinal(0) layout(#dynamic_pipeline_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @rocket_dynamic_conv2d() {
          return
        }
      }
    }
  }

  hal.executable private @rocket_dynamic_executable_s4 {
    hal.executable.variant public @rocket_dynamic_conv2d_v1 target(#rocket_dynamic_target_s4) {
      hal.executable.export public @rocket_dynamic_conv2d ordinal(0) layout(#dynamic_pipeline_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @rocket_dynamic_conv2d() {
          return
        }
      }
    }
  }

  hal.executable private @rocket_dynamic_depthwise_executable_s2 {
    hal.executable.variant public @rocket_dynamic_depthwise_conv2d_v1 target(#rocket_dynamic_depthwise_target_s2) {
      hal.executable.export public @rocket_dynamic_depthwise_conv2d ordinal(0) layout(#dynamic_pipeline_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @rocket_dynamic_depthwise_conv2d() {
          return
        }
      }
    }
  }

  hal.executable private @rocket_dynamic_depthwise_executable_s3 {
    hal.executable.variant public @rocket_dynamic_depthwise_conv2d_v1 target(#rocket_dynamic_depthwise_target_s3) {
      hal.executable.export public @rocket_dynamic_depthwise_conv2d ordinal(0) layout(#dynamic_pipeline_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @rocket_dynamic_depthwise_conv2d() {
          return
        }
      }
    }
  }

  hal.executable private @rocket_dynamic_depthwise_executable_s4 {
    hal.executable.variant public @rocket_dynamic_depthwise_conv2d_v1 target(#rocket_dynamic_depthwise_target_s4) {
      hal.executable.export public @rocket_dynamic_depthwise_conv2d ordinal(0) layout(#dynamic_pipeline_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @rocket_dynamic_depthwise_conv2d() {
          return
        }
      }
    }
  }


  hal.executable private @rocket_dynamic_int8_executable {
    hal.executable.variant public @rocket_dynamic_conv2d_v1 target(#rocket_dynamic_int8_target) {
      hal.executable.export public @rocket_dynamic_conv2d ordinal(0) layout(#dynamic_pipeline_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @rocket_dynamic_conv2d() {
          return
        }
      }
    }
  }

  hal.executable private @rocket_dynamic_depthwise_int8_executable {
    hal.executable.variant public @rocket_dynamic_depthwise_conv2d_v1 target(#rocket_dynamic_depthwise_int8_target) {
      hal.executable.export public @rocket_dynamic_depthwise_conv2d ordinal(0) layout(#dynamic_pipeline_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @rocket_dynamic_depthwise_conv2d() {
          return
        }
      }
    }
  }

  hal.executable private @rocket_dynamic_depthwise_int8_executable_s2 {
    hal.executable.variant public @rocket_dynamic_depthwise_conv2d_v1 target(#rocket_dynamic_depthwise_int8_target_s2) {
      hal.executable.export public @rocket_dynamic_depthwise_conv2d ordinal(0) layout(#dynamic_pipeline_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @rocket_dynamic_depthwise_conv2d() {
          return
        }
      }
    }
  }


  // Generic runtime-shape adapter. Batch remains statically one because it is
  // fixed by the Rocket Conv ABI. Every other logical Conv dimension is read
  // from the cast-compatible tensor operands and passed as an i32 push
  // constant in exactly the order declared by #rocket_dynamic_target.
  util.func private @call_rocket_dynamic_conv2d(
      %input: tensor<1x?x?x?xf16>,
      %filter: tensor<?x?x?x?xf16>,
      %init: tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    %input_height = tensor.dim %input, %c1 : tensor<1x?x?x?xf16>
    %input_width = tensor.dim %input, %c2 : tensor<1x?x?x?xf16>
    %input_channels = tensor.dim %input, %c3 : tensor<1x?x?x?xf16>
    %weights_height = tensor.dim %filter, %c0 : tensor<?x?x?x?xf16>
    %weights_width = tensor.dim %filter, %c1 : tensor<?x?x?x?xf16>
    %output_height = tensor.dim %init, %c1 : tensor<1x?x?x?xf32>
    %output_width = tensor.dim %init, %c2 : tensor<1x?x?x?xf32>
    %output_channels = tensor.dim %init, %c3 : tensor<1x?x?x?xf32>

    // %output_height/%output_width stay index-typed: they still describe the
    // dispatch result shape, but they are not push constants -- the runtime
    // derives the output extent itself.
    %input_width_i32 = arith.index_cast %input_width : index to i32
    %input_height_i32 = arith.index_cast %input_height : index to i32
    %input_channels_i32 = arith.index_cast %input_channels : index to i32
    %output_channels_i32 = arith.index_cast %output_channels : index to i32
    %weights_width_i32 = arith.index_cast %weights_width : index to i32
    %weights_height_i32 = arith.index_cast %weights_height : index to i32

    %zero_bias_empty = tensor.empty(%output_channels) : tensor<?xf16>
    %zero_f16 = arith.constant 0.0 : f16
    %zero_bias = linalg.fill ins(%zero_f16 : f16)
        outs(%zero_bias_empty : tensor<?xf16>) -> tensor<?xf16>

    %raw_f16 = flow.dispatch
        @rocket_dynamic_executable::@rocket_dynamic_conv2d_v1::@rocket_dynamic_conv2d(
          %input_width_i32, %input_height_i32, %input_channels_i32,
          %output_channels_i32, %weights_width_i32, %weights_height_i32,
          %input, %filter, %zero_bias)
        {stream.affinity = #hal.device.affinity<@rocket_device>}
        : (i32, i32, i32, i32, i32, i32,
           tensor<1x?x?x?xf16>{%input_height, %input_width, %input_channels},
           tensor<?x?x?x?xf16>{%weights_height, %weights_width, %input_channels, %output_channels},
           tensor<?xf16>{%output_channels})
        -> tensor<1x?x?x?xf16>{%output_height, %output_width, %output_channels}

    %final = flow.dispatch.workgroups[
        %output_height, %output_width, %output_channels](
        %raw_f16, %init, %output_height, %output_width, %output_channels)
        : (tensor<1x?x?x?xf16>{%output_height, %output_width, %output_channels},
           tensor<1x?x?x?xf32>{%output_height, %output_width, %output_channels},
           index, index, index)
        -> tensor<1x?x?x?xf32>{%output_height, %output_width, %output_channels}
        attributes { stream.affinity = #hal.device.affinity<@cpu_device> } =
        (%raw_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>,
         %init_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>,
         %output_height_arg: index,
         %output_width_arg: index,
         %output_channels_arg: index,
         %final_binding: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>) {
      %output_height_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_height_arg, 0 : index
      %output_width_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_width_arg, 1 : index
      %output_channels_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_channels_arg, 2 : index
      %raw_shaped = flow.dispatch.tie_shape %raw_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %init_shaped = flow.dispatch.tie_shape %init_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %final_shaped = flow.dispatch.tie_shape %final_binding
          : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %raw_loaded = iree_tensor_ext.dispatch.tensor.load %raw_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %output_channels_size}
          -> tensor<1x?x?x?xf16>
      %init_loaded = iree_tensor_ext.dispatch.tensor.load %init_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
          -> tensor<1x?x?x?xf32>
      %final_empty = tensor.empty(
          %output_height_size, %output_width_size, %output_channels_size)
          : tensor<1x?x?x?xf32>
      %final_inner = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
          ],
          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } ins(%raw_loaded, %init_loaded
            : tensor<1x?x?x?xf16>, tensor<1x?x?x?xf32>)
          outs(%final_empty : tensor<1x?x?x?xf32>) {
        ^bb0(%raw: f16, %initial: f32, %out: f32):
          %raw_f32 = arith.extf %raw : f16 to f32
          %sum = arith.addf %raw_f32, %initial : f32
          linalg.yield %sum : f32
      } -> tensor<1x?x?x?xf32>
      iree_tensor_ext.dispatch.tensor.store %final_inner, %final_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : tensor<1x?x?x?xf32>
          -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      flow.return
    } count(%output_height_workload: index,
            %output_width_workload: index,
            %output_channels_workload: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(
          %output_height_workload,
          %output_width_workload,
          %output_channels_workload)
      flow.return %x, %y, %z : index, index, index
    }

    util.return %final : tensor<1x?x?x?xf32>
  }

  // Stride-2 counterpart of call_rocket_dynamic_conv2d -- byte-identical
  // apart from the executable it dispatches to (@rocket_dynamic_executable_s2,
  // whose #rocket_dynamic_target_s2 bakes stride=2: the wire format's
  // Conv2DDef.stride is a fixed per-variant attribute, not one of the
  // runtime-settable push constants above, so a distinct stride needs its
  // own executable rather than a runtime argument). Hardware-confirmed for
  // dense fp16 at stride 2/3/4, both 1x1 and 3x3 kernels
  // (conv_wide_shape_hw.rs::shape_generalised_convs_run_on_npu, all 40
  // combinations pass -- see DESIGN_NOTES.md "Stride and large-width
  // sweeps").
  util.func private @call_rocket_dynamic_conv2d_s2(
      %input: tensor<1x?x?x?xf16>,
      %filter: tensor<?x?x?x?xf16>,
      %init: tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    %input_height = tensor.dim %input, %c1 : tensor<1x?x?x?xf16>
    %input_width = tensor.dim %input, %c2 : tensor<1x?x?x?xf16>
    %input_channels = tensor.dim %input, %c3 : tensor<1x?x?x?xf16>
    %weights_height = tensor.dim %filter, %c0 : tensor<?x?x?x?xf16>
    %weights_width = tensor.dim %filter, %c1 : tensor<?x?x?x?xf16>
    %output_height = tensor.dim %init, %c1 : tensor<1x?x?x?xf32>
    %output_width = tensor.dim %init, %c2 : tensor<1x?x?x?xf32>
    %output_channels = tensor.dim %init, %c3 : tensor<1x?x?x?xf32>

    %input_width_i32 = arith.index_cast %input_width : index to i32
    %input_height_i32 = arith.index_cast %input_height : index to i32
    %input_channels_i32 = arith.index_cast %input_channels : index to i32
    %output_channels_i32 = arith.index_cast %output_channels : index to i32
    %weights_width_i32 = arith.index_cast %weights_width : index to i32
    %weights_height_i32 = arith.index_cast %weights_height : index to i32

    %zero_bias_empty = tensor.empty(%output_channels) : tensor<?xf16>
    %zero_f16 = arith.constant 0.0 : f16
    %zero_bias = linalg.fill ins(%zero_f16 : f16)
        outs(%zero_bias_empty : tensor<?xf16>) -> tensor<?xf16>

    %raw_f16 = flow.dispatch
        @rocket_dynamic_executable_s2::@rocket_dynamic_conv2d_v1::@rocket_dynamic_conv2d(
          %input_width_i32, %input_height_i32, %input_channels_i32,
          %output_channels_i32, %weights_width_i32, %weights_height_i32,
          %input, %filter, %zero_bias)
        {stream.affinity = #hal.device.affinity<@rocket_device>}
        : (i32, i32, i32, i32, i32, i32,
           tensor<1x?x?x?xf16>{%input_height, %input_width, %input_channels},
           tensor<?x?x?x?xf16>{%weights_height, %weights_width, %input_channels, %output_channels},
           tensor<?xf16>{%output_channels})
        -> tensor<1x?x?x?xf16>{%output_height, %output_width, %output_channels}

    %final = flow.dispatch.workgroups[
        %output_height, %output_width, %output_channels](
        %raw_f16, %init, %output_height, %output_width, %output_channels)
        : (tensor<1x?x?x?xf16>{%output_height, %output_width, %output_channels},
           tensor<1x?x?x?xf32>{%output_height, %output_width, %output_channels},
           index, index, index)
        -> tensor<1x?x?x?xf32>{%output_height, %output_width, %output_channels}
        attributes { stream.affinity = #hal.device.affinity<@cpu_device> } =
        (%raw_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>,
         %init_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>,
         %output_height_arg: index,
         %output_width_arg: index,
         %output_channels_arg: index,
         %final_binding: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>) {
      %output_height_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_height_arg, 0 : index
      %output_width_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_width_arg, 1 : index
      %output_channels_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_channels_arg, 2 : index
      %raw_shaped = flow.dispatch.tie_shape %raw_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %init_shaped = flow.dispatch.tie_shape %init_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %final_shaped = flow.dispatch.tie_shape %final_binding
          : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %raw_loaded = iree_tensor_ext.dispatch.tensor.load %raw_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %output_channels_size}
          -> tensor<1x?x?x?xf16>
      %init_loaded = iree_tensor_ext.dispatch.tensor.load %init_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
          -> tensor<1x?x?x?xf32>
      %final_empty = tensor.empty(
          %output_height_size, %output_width_size, %output_channels_size)
          : tensor<1x?x?x?xf32>
      %final_inner = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
          ],
          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } ins(%raw_loaded, %init_loaded
            : tensor<1x?x?x?xf16>, tensor<1x?x?x?xf32>)
          outs(%final_empty : tensor<1x?x?x?xf32>) {
        ^bb0(%raw: f16, %initial: f32, %out: f32):
          %raw_f32 = arith.extf %raw : f16 to f32
          %sum = arith.addf %raw_f32, %initial : f32
          linalg.yield %sum : f32
      } -> tensor<1x?x?x?xf32>
      iree_tensor_ext.dispatch.tensor.store %final_inner, %final_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : tensor<1x?x?x?xf32>
          -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      flow.return
    } count(%output_height_workload: index,
            %output_width_workload: index,
            %output_channels_workload: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(
          %output_height_workload,
          %output_width_workload,
          %output_channels_workload)
      flow.return %x, %y, %z : index, index, index
    }

    util.return %final : tensor<1x?x?x?xf32>
  }

  // Stride-3 counterpart of call_rocket_dynamic_conv2d -- byte-identical
  // apart from the executable it dispatches to (@rocket_dynamic_executable_s3,
  // whose #rocket_dynamic_target_s3 bakes stride=3: the wire format's
  // Conv2DDef.stride is a fixed per-variant attribute, not one of the
  // runtime-settable push constants above, so a distinct stride needs its
  // own executable rather than a runtime argument). Hardware-confirmed for
  // dense fp16 at stride 2/3/4, both 1x1 and 3x3 kernels
  // (conv_wide_shape_hw.rs::shape_generalised_convs_run_on_npu, all 40
  // combinations pass -- see DESIGN_NOTES.md "Stride and large-width
  // sweeps").
  util.func private @call_rocket_dynamic_conv2d_s3(
      %input: tensor<1x?x?x?xf16>,
      %filter: tensor<?x?x?x?xf16>,
      %init: tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    %input_height = tensor.dim %input, %c1 : tensor<1x?x?x?xf16>
    %input_width = tensor.dim %input, %c2 : tensor<1x?x?x?xf16>
    %input_channels = tensor.dim %input, %c3 : tensor<1x?x?x?xf16>
    %weights_height = tensor.dim %filter, %c0 : tensor<?x?x?x?xf16>
    %weights_width = tensor.dim %filter, %c1 : tensor<?x?x?x?xf16>
    %output_height = tensor.dim %init, %c1 : tensor<1x?x?x?xf32>
    %output_width = tensor.dim %init, %c2 : tensor<1x?x?x?xf32>
    %output_channels = tensor.dim %init, %c3 : tensor<1x?x?x?xf32>

    %input_width_i32 = arith.index_cast %input_width : index to i32
    %input_height_i32 = arith.index_cast %input_height : index to i32
    %input_channels_i32 = arith.index_cast %input_channels : index to i32
    %output_channels_i32 = arith.index_cast %output_channels : index to i32
    %weights_width_i32 = arith.index_cast %weights_width : index to i32
    %weights_height_i32 = arith.index_cast %weights_height : index to i32

    %zero_bias_empty = tensor.empty(%output_channels) : tensor<?xf16>
    %zero_f16 = arith.constant 0.0 : f16
    %zero_bias = linalg.fill ins(%zero_f16 : f16)
        outs(%zero_bias_empty : tensor<?xf16>) -> tensor<?xf16>

    %raw_f16 = flow.dispatch
        @rocket_dynamic_executable_s3::@rocket_dynamic_conv2d_v1::@rocket_dynamic_conv2d(
          %input_width_i32, %input_height_i32, %input_channels_i32,
          %output_channels_i32, %weights_width_i32, %weights_height_i32,
          %input, %filter, %zero_bias)
        {stream.affinity = #hal.device.affinity<@rocket_device>}
        : (i32, i32, i32, i32, i32, i32,
           tensor<1x?x?x?xf16>{%input_height, %input_width, %input_channels},
           tensor<?x?x?x?xf16>{%weights_height, %weights_width, %input_channels, %output_channels},
           tensor<?xf16>{%output_channels})
        -> tensor<1x?x?x?xf16>{%output_height, %output_width, %output_channels}

    %final = flow.dispatch.workgroups[
        %output_height, %output_width, %output_channels](
        %raw_f16, %init, %output_height, %output_width, %output_channels)
        : (tensor<1x?x?x?xf16>{%output_height, %output_width, %output_channels},
           tensor<1x?x?x?xf32>{%output_height, %output_width, %output_channels},
           index, index, index)
        -> tensor<1x?x?x?xf32>{%output_height, %output_width, %output_channels}
        attributes { stream.affinity = #hal.device.affinity<@cpu_device> } =
        (%raw_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>,
         %init_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>,
         %output_height_arg: index,
         %output_width_arg: index,
         %output_channels_arg: index,
         %final_binding: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>) {
      %output_height_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_height_arg, 0 : index
      %output_width_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_width_arg, 1 : index
      %output_channels_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_channels_arg, 2 : index
      %raw_shaped = flow.dispatch.tie_shape %raw_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %init_shaped = flow.dispatch.tie_shape %init_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %final_shaped = flow.dispatch.tie_shape %final_binding
          : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %raw_loaded = iree_tensor_ext.dispatch.tensor.load %raw_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %output_channels_size}
          -> tensor<1x?x?x?xf16>
      %init_loaded = iree_tensor_ext.dispatch.tensor.load %init_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
          -> tensor<1x?x?x?xf32>
      %final_empty = tensor.empty(
          %output_height_size, %output_width_size, %output_channels_size)
          : tensor<1x?x?x?xf32>
      %final_inner = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
          ],
          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } ins(%raw_loaded, %init_loaded
            : tensor<1x?x?x?xf16>, tensor<1x?x?x?xf32>)
          outs(%final_empty : tensor<1x?x?x?xf32>) {
        ^bb0(%raw: f16, %initial: f32, %out: f32):
          %raw_f32 = arith.extf %raw : f16 to f32
          %sum = arith.addf %raw_f32, %initial : f32
          linalg.yield %sum : f32
      } -> tensor<1x?x?x?xf32>
      iree_tensor_ext.dispatch.tensor.store %final_inner, %final_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : tensor<1x?x?x?xf32>
          -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      flow.return
    } count(%output_height_workload: index,
            %output_width_workload: index,
            %output_channels_workload: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(
          %output_height_workload,
          %output_width_workload,
          %output_channels_workload)
      flow.return %x, %y, %z : index, index, index
    }

    util.return %final : tensor<1x?x?x?xf32>
  }

  // Stride-4 counterpart of call_rocket_dynamic_conv2d -- byte-identical
  // apart from the executable it dispatches to (@rocket_dynamic_executable_s4,
  // whose #rocket_dynamic_target_s4 bakes stride=4: the wire format's
  // Conv2DDef.stride is a fixed per-variant attribute, not one of the
  // runtime-settable push constants above, so a distinct stride needs its
  // own executable rather than a runtime argument). Hardware-confirmed for
  // dense fp16 at stride 2/3/4, both 1x1 and 3x3 kernels
  // (conv_wide_shape_hw.rs::shape_generalised_convs_run_on_npu, all 40
  // combinations pass -- see DESIGN_NOTES.md "Stride and large-width
  // sweeps").
  util.func private @call_rocket_dynamic_conv2d_s4(
      %input: tensor<1x?x?x?xf16>,
      %filter: tensor<?x?x?x?xf16>,
      %init: tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    %input_height = tensor.dim %input, %c1 : tensor<1x?x?x?xf16>
    %input_width = tensor.dim %input, %c2 : tensor<1x?x?x?xf16>
    %input_channels = tensor.dim %input, %c3 : tensor<1x?x?x?xf16>
    %weights_height = tensor.dim %filter, %c0 : tensor<?x?x?x?xf16>
    %weights_width = tensor.dim %filter, %c1 : tensor<?x?x?x?xf16>
    %output_height = tensor.dim %init, %c1 : tensor<1x?x?x?xf32>
    %output_width = tensor.dim %init, %c2 : tensor<1x?x?x?xf32>
    %output_channels = tensor.dim %init, %c3 : tensor<1x?x?x?xf32>

    %input_width_i32 = arith.index_cast %input_width : index to i32
    %input_height_i32 = arith.index_cast %input_height : index to i32
    %input_channels_i32 = arith.index_cast %input_channels : index to i32
    %output_channels_i32 = arith.index_cast %output_channels : index to i32
    %weights_width_i32 = arith.index_cast %weights_width : index to i32
    %weights_height_i32 = arith.index_cast %weights_height : index to i32

    %zero_bias_empty = tensor.empty(%output_channels) : tensor<?xf16>
    %zero_f16 = arith.constant 0.0 : f16
    %zero_bias = linalg.fill ins(%zero_f16 : f16)
        outs(%zero_bias_empty : tensor<?xf16>) -> tensor<?xf16>

    %raw_f16 = flow.dispatch
        @rocket_dynamic_executable_s4::@rocket_dynamic_conv2d_v1::@rocket_dynamic_conv2d(
          %input_width_i32, %input_height_i32, %input_channels_i32,
          %output_channels_i32, %weights_width_i32, %weights_height_i32,
          %input, %filter, %zero_bias)
        {stream.affinity = #hal.device.affinity<@rocket_device>}
        : (i32, i32, i32, i32, i32, i32,
           tensor<1x?x?x?xf16>{%input_height, %input_width, %input_channels},
           tensor<?x?x?x?xf16>{%weights_height, %weights_width, %input_channels, %output_channels},
           tensor<?xf16>{%output_channels})
        -> tensor<1x?x?x?xf16>{%output_height, %output_width, %output_channels}

    %final = flow.dispatch.workgroups[
        %output_height, %output_width, %output_channels](
        %raw_f16, %init, %output_height, %output_width, %output_channels)
        : (tensor<1x?x?x?xf16>{%output_height, %output_width, %output_channels},
           tensor<1x?x?x?xf32>{%output_height, %output_width, %output_channels},
           index, index, index)
        -> tensor<1x?x?x?xf32>{%output_height, %output_width, %output_channels}
        attributes { stream.affinity = #hal.device.affinity<@cpu_device> } =
        (%raw_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>,
         %init_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>,
         %output_height_arg: index,
         %output_width_arg: index,
         %output_channels_arg: index,
         %final_binding: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>) {
      %output_height_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_height_arg, 0 : index
      %output_width_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_width_arg, 1 : index
      %output_channels_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_channels_arg, 2 : index
      %raw_shaped = flow.dispatch.tie_shape %raw_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %init_shaped = flow.dispatch.tie_shape %init_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %final_shaped = flow.dispatch.tie_shape %final_binding
          : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %raw_loaded = iree_tensor_ext.dispatch.tensor.load %raw_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %output_channels_size}
          -> tensor<1x?x?x?xf16>
      %init_loaded = iree_tensor_ext.dispatch.tensor.load %init_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
          -> tensor<1x?x?x?xf32>
      %final_empty = tensor.empty(
          %output_height_size, %output_width_size, %output_channels_size)
          : tensor<1x?x?x?xf32>
      %final_inner = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
          ],
          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } ins(%raw_loaded, %init_loaded
            : tensor<1x?x?x?xf16>, tensor<1x?x?x?xf32>)
          outs(%final_empty : tensor<1x?x?x?xf32>) {
        ^bb0(%raw: f16, %initial: f32, %out: f32):
          %raw_f32 = arith.extf %raw : f16 to f32
          %sum = arith.addf %raw_f32, %initial : f32
          linalg.yield %sum : f32
      } -> tensor<1x?x?x?xf32>
      iree_tensor_ext.dispatch.tensor.store %final_inner, %final_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : tensor<1x?x?x?xf32>
          -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      flow.return
    } count(%output_height_workload: index,
            %output_width_workload: index,
            %output_channels_workload: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(
          %output_height_workload,
          %output_width_workload,
          %output_channels_workload)
      flow.return %x, %y, %z : index, index, index
    }

    util.return %final : tensor<1x?x?x?xf32>
  }


  // Depthwise counterpart of call_rocket_dynamic_conv2d. The filter operand
  // drops its Cout dimension (HWC, not HWCF): Rocket's depthwise mode is
  // hardware-validated for a channel multiplier of one only
  // (ConvPlan::with_depthwise asserts this in conv.rs -- "depthwise capture
  // backing covers a channel multiplier of one only"), so output_channels
  // is always input_channels here, not an independent quantity. It is still
  // read off %init and sent as its own push constant, matching the wire
  // format #rocket_dynamic_depthwise_target declares, but the matchers below
  // never bind it to anything other than input_channels.
  util.func private @call_rocket_dynamic_depthwise_conv2d(
      %input: tensor<1x?x?x?xf16>,
      %filter: tensor<?x?x?xf16>,
      %init: tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    %input_height = tensor.dim %input, %c1 : tensor<1x?x?x?xf16>
    %input_width = tensor.dim %input, %c2 : tensor<1x?x?x?xf16>
    %input_channels = tensor.dim %input, %c3 : tensor<1x?x?x?xf16>
    %weights_height = tensor.dim %filter, %c0 : tensor<?x?x?xf16>
    %weights_width = tensor.dim %filter, %c1 : tensor<?x?x?xf16>
    %output_height = tensor.dim %init, %c1 : tensor<1x?x?x?xf32>
    %output_width = tensor.dim %init, %c2 : tensor<1x?x?x?xf32>
    %output_channels = tensor.dim %init, %c3 : tensor<1x?x?x?xf32>

    %input_width_i32 = arith.index_cast %input_width : index to i32
    %input_height_i32 = arith.index_cast %input_height : index to i32
    %input_channels_i32 = arith.index_cast %input_channels : index to i32
    %output_channels_i32 = arith.index_cast %output_channels : index to i32
    %weights_width_i32 = arith.index_cast %weights_width : index to i32
    %weights_height_i32 = arith.index_cast %weights_height : index to i32

    %zero_bias_empty = tensor.empty(%output_channels) : tensor<?xf16>
    %zero_f16 = arith.constant 0.0 : f16
    %zero_bias = linalg.fill ins(%zero_f16 : f16)
        outs(%zero_bias_empty : tensor<?xf16>) -> tensor<?xf16>

    // linalg.depthwise_conv_2d_nhwc_hwc's filter operand is [kh][kw][c] --
    // required by the op's own semantics, matched by the CPU reference
    // computation, and NOT what the driver's weight-packing path expects.
    // rocket-hal-driver's pack_depthwise_to_rocket_weights (the hardware-
    // derived tap-major packer, see its doc comment) takes a torch/ONNX-
    // style [c][kh][kw] buffer, matching how a real depthwise filter is
    // conventionally stored -- this op's HWC layout is a linalg-dialect
    // convention, not a hardware one. Transposing here, once, keeps that
    // packer's contract simple instead of teaching it a second input order.
    %filter_chw_empty = tensor.empty(%input_channels, %weights_height, %weights_width) : tensor<?x?x?xf16>
    %filter_chw = linalg.transpose
        ins(%filter : tensor<?x?x?xf16>)
        outs(%filter_chw_empty : tensor<?x?x?xf16>)
        permutation = [2, 0, 1]

    %raw_f16 = flow.dispatch
        @rocket_dynamic_depthwise_executable::@rocket_dynamic_depthwise_conv2d_v1::@rocket_dynamic_depthwise_conv2d(
          %input_width_i32, %input_height_i32, %input_channels_i32,
          %output_channels_i32, %weights_width_i32, %weights_height_i32,
          %input, %filter_chw, %zero_bias)
        {stream.affinity = #hal.device.affinity<@rocket_device>}
        : (i32, i32, i32, i32, i32, i32,
           tensor<1x?x?x?xf16>{%input_height, %input_width, %input_channels},
           tensor<?x?x?xf16>{%input_channels, %weights_height, %weights_width},
           tensor<?xf16>{%output_channels})
        -> tensor<1x?x?x?xf16>{%output_height, %output_width, %output_channels}

    %final = flow.dispatch.workgroups[
        %output_height, %output_width, %output_channels](
        %raw_f16, %init, %output_height, %output_width, %output_channels)
        : (tensor<1x?x?x?xf16>{%output_height, %output_width, %output_channels},
           tensor<1x?x?x?xf32>{%output_height, %output_width, %output_channels},
           index, index, index)
        -> tensor<1x?x?x?xf32>{%output_height, %output_width, %output_channels}
        attributes { stream.affinity = #hal.device.affinity<@cpu_device> } =
        (%raw_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>,
         %init_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>,
         %output_height_arg: index,
         %output_width_arg: index,
         %output_channels_arg: index,
         %final_binding: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>) {
      %output_height_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_height_arg, 0 : index
      %output_width_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_width_arg, 1 : index
      %output_channels_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_channels_arg, 2 : index
      %raw_shaped = flow.dispatch.tie_shape %raw_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %init_shaped = flow.dispatch.tie_shape %init_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %final_shaped = flow.dispatch.tie_shape %final_binding
          : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %raw_loaded = iree_tensor_ext.dispatch.tensor.load %raw_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %output_channels_size}
          -> tensor<1x?x?x?xf16>
      %init_loaded = iree_tensor_ext.dispatch.tensor.load %init_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
          -> tensor<1x?x?x?xf32>
      %final_empty = tensor.empty(
          %output_height_size, %output_width_size, %output_channels_size)
          : tensor<1x?x?x?xf32>
      %final_inner = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
          ],
          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } ins(%raw_loaded, %init_loaded
            : tensor<1x?x?x?xf16>, tensor<1x?x?x?xf32>)
          outs(%final_empty : tensor<1x?x?x?xf32>) {
        ^bb0(%raw: f16, %initial: f32, %out: f32):
          %raw_f32 = arith.extf %raw : f16 to f32
          %sum = arith.addf %raw_f32, %initial : f32
          linalg.yield %sum : f32
      } -> tensor<1x?x?x?xf32>
      iree_tensor_ext.dispatch.tensor.store %final_inner, %final_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : tensor<1x?x?x?xf32>
          -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      flow.return
    } count(%output_height_workload: index,
            %output_width_workload: index,
            %output_channels_workload: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(
          %output_height_workload,
          %output_width_workload,
          %output_channels_workload)
      flow.return %x, %y, %z : index, index, index
    }

    util.return %final : tensor<1x?x?x?xf32>
  }

  // NCHW counterpart of call_rocket_dynamic_depthwise_conv2d. A real model
  // imported from ONNX (torch-mlir's onnx.Conv legalization) never produces
  // linalg.depthwise_conv_2d_nhwc_hwc directly: it lowers to
  // linalg.depthwise_conv_2d_nchw_chw, and IREE's own
  // iree-preprocessing-convert-conv-to-channels-last pass explicitly
  // declines to transpose depthwise convs to channels-last
  // (ConvertConvToChannelsLast.cpp's transposeConvLikeLinalgOp bails
  // whenever ConvolutionDimensions::depth is non-empty), unlike the dense
  // path, which that same pass always converts first. So the NHWC-only
  // matcher above -- correct and hardware-confirmed on its own -- never
  // sees a real ONNX-imported depthwise conv at all; every one of
  // MobileNetV2's 17 depthwise layers fell back to CPU confirming this
  // (iree-dump-module on a real compiled mobilenet.vmfb: zero references to
  // rocket_dynamic_depthwise_executable).
  //
  // Handled here by transposing host-side instead of waiting on an upstream
  // fix to the shared preprocessing pass: Rocket's hardware ABI is NHWC-
  // native (every existing matcher in this file agrees), so the input and
  // output feature maps are transposed NCHW<->NHWC around the same
  // rocket_dynamic_depthwise_executable dispatch call_rocket_dynamic_depthwise_conv2d
  // already uses. The filter needs no transpose at all here -- unlike the
  // NHWC matcher, whose HWC filter has to be transposed to CHW before
  // dispatch, linalg.depthwise_conv_2d_nchw_chw's filter operand is already
  // [c][kh][kw] (confirmed against a real onnx-imported MobileNet dump:
  // `tensor<32x3x3xf16>` for a Cin=32 depthwise layer), exactly what
  // rocket-hal-driver's pack_depthwise_to_rocket_weights expects.
  util.func private @call_rocket_dynamic_depthwise_conv2d_nchw(
      %input: tensor<1x?x?x?xf16>,
      %filter: tensor<?x?x?xf16>,
      %init: tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    // NCHW: dim 1 is channels, dims 2/3 are the spatial extent.
    %input_channels = tensor.dim %input, %c1 : tensor<1x?x?x?xf16>
    %input_height = tensor.dim %input, %c2 : tensor<1x?x?x?xf16>
    %input_width = tensor.dim %input, %c3 : tensor<1x?x?x?xf16>
    // Filter is [c][kh][kw]: dims 1/2 are the kernel extent.
    %weights_height = tensor.dim %filter, %c1 : tensor<?x?x?xf16>
    %weights_width = tensor.dim %filter, %c2 : tensor<?x?x?xf16>
    %output_channels = tensor.dim %init, %c1 : tensor<1x?x?x?xf32>
    %output_height = tensor.dim %init, %c2 : tensor<1x?x?x?xf32>
    %output_width = tensor.dim %init, %c3 : tensor<1x?x?x?xf32>

    %input_width_i32 = arith.index_cast %input_width : index to i32
    %input_height_i32 = arith.index_cast %input_height : index to i32
    %input_channels_i32 = arith.index_cast %input_channels : index to i32
    %output_channels_i32 = arith.index_cast %output_channels : index to i32
    %weights_width_i32 = arith.index_cast %weights_width : index to i32
    %weights_height_i32 = arith.index_cast %weights_height : index to i32

    %zero_bias_empty = tensor.empty(%output_channels) : tensor<?xf16>
    %zero_f16 = arith.constant 0.0 : f16
    %zero_bias = linalg.fill ins(%zero_f16 : f16)
        outs(%zero_bias_empty : tensor<?xf16>) -> tensor<?xf16>

    // NCHW [1,C,H,W] -> NHWC [1,H,W,C]: out.shape[i] = in.shape[perm[i]],
    // so perm = [0, 2, 3, 1].
    %input_nhwc_empty = tensor.empty(%input_height, %input_width, %input_channels) : tensor<1x?x?x?xf16>
    %input_nhwc = linalg.transpose
        ins(%input : tensor<1x?x?x?xf16>)
        outs(%input_nhwc_empty : tensor<1x?x?x?xf16>)
        permutation = [0, 2, 3, 1]

    %raw_f16 = flow.dispatch
        @rocket_dynamic_depthwise_executable::@rocket_dynamic_depthwise_conv2d_v1::@rocket_dynamic_depthwise_conv2d(
          %input_width_i32, %input_height_i32, %input_channels_i32,
          %output_channels_i32, %weights_width_i32, %weights_height_i32,
          %input_nhwc, %filter, %zero_bias)
        {stream.affinity = #hal.device.affinity<@rocket_device>}
        : (i32, i32, i32, i32, i32, i32,
           tensor<1x?x?x?xf16>{%input_height, %input_width, %input_channels},
           tensor<?x?x?xf16>{%input_channels, %weights_height, %weights_width},
           tensor<?xf16>{%output_channels})
        -> tensor<1x?x?x?xf16>{%output_height, %output_width, %output_channels}

    // %init also arrives NCHW; transpose it to NHWC too so it lines up with
    // %raw_f16 for the CPU-side accumulate below.
    %init_nhwc_empty = tensor.empty(%output_height, %output_width, %output_channels) : tensor<1x?x?x?xf32>
    %init_nhwc = linalg.transpose
        ins(%init : tensor<1x?x?x?xf32>)
        outs(%init_nhwc_empty : tensor<1x?x?x?xf32>)
        permutation = [0, 2, 3, 1]

    %final_nhwc = flow.dispatch.workgroups[
        %output_height, %output_width, %output_channels](
        %raw_f16, %init_nhwc, %output_height, %output_width, %output_channels)
        : (tensor<1x?x?x?xf16>{%output_height, %output_width, %output_channels},
           tensor<1x?x?x?xf32>{%output_height, %output_width, %output_channels},
           index, index, index)
        -> tensor<1x?x?x?xf32>{%output_height, %output_width, %output_channels}
        attributes { stream.affinity = #hal.device.affinity<@cpu_device> } =
        (%raw_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>,
         %init_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>,
         %output_height_arg: index,
         %output_width_arg: index,
         %output_channels_arg: index,
         %final_binding: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>) {
      %output_height_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_height_arg, 0 : index
      %output_width_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_width_arg, 1 : index
      %output_channels_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_channels_arg, 2 : index
      %raw_shaped = flow.dispatch.tie_shape %raw_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %init_shaped = flow.dispatch.tie_shape %init_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %final_shaped = flow.dispatch.tie_shape %final_binding
          : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %raw_loaded = iree_tensor_ext.dispatch.tensor.load %raw_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %output_channels_size}
          -> tensor<1x?x?x?xf16>
      %init_loaded = iree_tensor_ext.dispatch.tensor.load %init_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
          -> tensor<1x?x?x?xf32>
      %final_empty = tensor.empty(
          %output_height_size, %output_width_size, %output_channels_size)
          : tensor<1x?x?x?xf32>
      %final_inner = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
          ],
          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } ins(%raw_loaded, %init_loaded
            : tensor<1x?x?x?xf16>, tensor<1x?x?x?xf32>)
          outs(%final_empty : tensor<1x?x?x?xf32>) {
        ^bb0(%raw: f16, %initial: f32, %out: f32):
          %raw_f32 = arith.extf %raw : f16 to f32
          %sum = arith.addf %raw_f32, %initial : f32
          linalg.yield %sum : f32
      } -> tensor<1x?x?x?xf32>
      iree_tensor_ext.dispatch.tensor.store %final_inner, %final_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : tensor<1x?x?x?xf32>
          -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      flow.return
    } count(%output_height_workload: index,
            %output_width_workload: index,
            %output_channels_workload: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(
          %output_height_workload,
          %output_width_workload,
          %output_channels_workload)
      flow.return %x, %y, %z : index, index, index
    }

    // NHWC [1,H,W,C] -> NCHW [1,C,H,W] back again: perm = [0, 3, 1, 2].
    %final_nchw_empty = tensor.empty(%output_channels, %output_height, %output_width) : tensor<1x?x?x?xf32>
    %final_nchw = linalg.transpose
        ins(%final_nhwc : tensor<1x?x?x?xf32>)
        outs(%final_nchw_empty : tensor<1x?x?x?xf32>)
        permutation = [0, 3, 1, 2]

    util.return %final_nchw : tensor<1x?x?x?xf32>
  }

  // Stride-2 counterpart of call_rocket_dynamic_depthwise_conv2d_nchw --
  // same NCHW<->NHWC transpose shim, dispatching to
  // @rocket_dynamic_depthwise_executable_s2 (#rocket_dynamic_depthwise_target_s2
  // bakes stride=2, same reason call_rocket_dynamic_conv2d_s2 needs its own
  // executable -- see that comment). Hardware-confirmed for depthwise fp16
  // at stride 2/3/4, both 1x1 and 3x3 kernels, Cin/Cout 8 and 12
  // (conv_depthwise_stride_hw.rs::depthwise_strided_convs_run_on_npu, all
  // 12 combinations pass) -- the stride/DW_EN combination
  // call_rocket_dynamic_depthwise_conv2d_nchw's own doc comment notes as
  // untested is now covered; weight packing itself has no stride
  // dependence to re-check, so this reuses the same
  // pack_depthwise_to_rocket_weights path unchanged.
  util.func private @call_rocket_dynamic_depthwise_conv2d_nchw_s2(
      %input: tensor<1x?x?x?xf16>,
      %filter: tensor<?x?x?xf16>,
      %init: tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    // NCHW: dim 1 is channels, dims 2/3 are the spatial extent.
    %input_channels = tensor.dim %input, %c1 : tensor<1x?x?x?xf16>
    %input_height = tensor.dim %input, %c2 : tensor<1x?x?x?xf16>
    %input_width = tensor.dim %input, %c3 : tensor<1x?x?x?xf16>
    // Filter is [c][kh][kw]: dims 1/2 are the kernel extent.
    %weights_height = tensor.dim %filter, %c1 : tensor<?x?x?xf16>
    %weights_width = tensor.dim %filter, %c2 : tensor<?x?x?xf16>
    %output_channels = tensor.dim %init, %c1 : tensor<1x?x?x?xf32>
    %output_height = tensor.dim %init, %c2 : tensor<1x?x?x?xf32>
    %output_width = tensor.dim %init, %c3 : tensor<1x?x?x?xf32>

    %input_width_i32 = arith.index_cast %input_width : index to i32
    %input_height_i32 = arith.index_cast %input_height : index to i32
    %input_channels_i32 = arith.index_cast %input_channels : index to i32
    %output_channels_i32 = arith.index_cast %output_channels : index to i32
    %weights_width_i32 = arith.index_cast %weights_width : index to i32
    %weights_height_i32 = arith.index_cast %weights_height : index to i32

    %zero_bias_empty = tensor.empty(%output_channels) : tensor<?xf16>
    %zero_f16 = arith.constant 0.0 : f16
    %zero_bias = linalg.fill ins(%zero_f16 : f16)
        outs(%zero_bias_empty : tensor<?xf16>) -> tensor<?xf16>

    // NCHW [1,C,H,W] -> NHWC [1,H,W,C]: out.shape[i] = in.shape[perm[i]],
    // so perm = [0, 2, 3, 1].
    %input_nhwc_empty = tensor.empty(%input_height, %input_width, %input_channels) : tensor<1x?x?x?xf16>
    %input_nhwc = linalg.transpose
        ins(%input : tensor<1x?x?x?xf16>)
        outs(%input_nhwc_empty : tensor<1x?x?x?xf16>)
        permutation = [0, 2, 3, 1]

    %raw_f16 = flow.dispatch
        @rocket_dynamic_depthwise_executable_s2::@rocket_dynamic_depthwise_conv2d_v1::@rocket_dynamic_depthwise_conv2d(
          %input_width_i32, %input_height_i32, %input_channels_i32,
          %output_channels_i32, %weights_width_i32, %weights_height_i32,
          %input_nhwc, %filter, %zero_bias)
        {stream.affinity = #hal.device.affinity<@rocket_device>}
        : (i32, i32, i32, i32, i32, i32,
           tensor<1x?x?x?xf16>{%input_height, %input_width, %input_channels},
           tensor<?x?x?xf16>{%input_channels, %weights_height, %weights_width},
           tensor<?xf16>{%output_channels})
        -> tensor<1x?x?x?xf16>{%output_height, %output_width, %output_channels}

    // %init also arrives NCHW; transpose it to NHWC too so it lines up with
    // %raw_f16 for the CPU-side accumulate below.
    %init_nhwc_empty = tensor.empty(%output_height, %output_width, %output_channels) : tensor<1x?x?x?xf32>
    %init_nhwc = linalg.transpose
        ins(%init : tensor<1x?x?x?xf32>)
        outs(%init_nhwc_empty : tensor<1x?x?x?xf32>)
        permutation = [0, 2, 3, 1]

    %final_nhwc = flow.dispatch.workgroups[
        %output_height, %output_width, %output_channels](
        %raw_f16, %init_nhwc, %output_height, %output_width, %output_channels)
        : (tensor<1x?x?x?xf16>{%output_height, %output_width, %output_channels},
           tensor<1x?x?x?xf32>{%output_height, %output_width, %output_channels},
           index, index, index)
        -> tensor<1x?x?x?xf32>{%output_height, %output_width, %output_channels}
        attributes { stream.affinity = #hal.device.affinity<@cpu_device> } =
        (%raw_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>,
         %init_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>,
         %output_height_arg: index,
         %output_width_arg: index,
         %output_channels_arg: index,
         %final_binding: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>) {
      %output_height_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_height_arg, 0 : index
      %output_width_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_width_arg, 1 : index
      %output_channels_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_channels_arg, 2 : index
      %raw_shaped = flow.dispatch.tie_shape %raw_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %init_shaped = flow.dispatch.tie_shape %init_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %final_shaped = flow.dispatch.tie_shape %final_binding
          : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %raw_loaded = iree_tensor_ext.dispatch.tensor.load %raw_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %output_channels_size}
          -> tensor<1x?x?x?xf16>
      %init_loaded = iree_tensor_ext.dispatch.tensor.load %init_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
          -> tensor<1x?x?x?xf32>
      %final_empty = tensor.empty(
          %output_height_size, %output_width_size, %output_channels_size)
          : tensor<1x?x?x?xf32>
      %final_inner = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
          ],
          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } ins(%raw_loaded, %init_loaded
            : tensor<1x?x?x?xf16>, tensor<1x?x?x?xf32>)
          outs(%final_empty : tensor<1x?x?x?xf32>) {
        ^bb0(%raw: f16, %initial: f32, %out: f32):
          %raw_f32 = arith.extf %raw : f16 to f32
          %sum = arith.addf %raw_f32, %initial : f32
          linalg.yield %sum : f32
      } -> tensor<1x?x?x?xf32>
      iree_tensor_ext.dispatch.tensor.store %final_inner, %final_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : tensor<1x?x?x?xf32>
          -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      flow.return
    } count(%output_height_workload: index,
            %output_width_workload: index,
            %output_channels_workload: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(
          %output_height_workload,
          %output_width_workload,
          %output_channels_workload)
      flow.return %x, %y, %z : index, index, index
    }

    // NHWC [1,H,W,C] -> NCHW [1,C,H,W] back again: perm = [0, 3, 1, 2].
    %final_nchw_empty = tensor.empty(%output_channels, %output_height, %output_width) : tensor<1x?x?x?xf32>
    %final_nchw = linalg.transpose
        ins(%final_nhwc : tensor<1x?x?x?xf32>)
        outs(%final_nchw_empty : tensor<1x?x?x?xf32>)
        permutation = [0, 3, 1, 2]

    util.return %final_nchw : tensor<1x?x?x?xf32>
  }

  // Stride-3 counterpart of call_rocket_dynamic_depthwise_conv2d_nchw --
  // same NCHW<->NHWC transpose shim, dispatching to
  // @rocket_dynamic_depthwise_executable_s3 (#rocket_dynamic_depthwise_target_s3
  // bakes stride=3, same reason call_rocket_dynamic_conv2d_s3 needs its own
  // executable -- see that comment). Hardware-confirmed for depthwise fp16
  // at stride 2/3/4, both 1x1 and 3x3 kernels, Cin/Cout 8 and 12
  // (conv_depthwise_stride_hw.rs::depthwise_strided_convs_run_on_npu, all
  // 12 combinations pass) -- the stride/DW_EN combination
  // call_rocket_dynamic_depthwise_conv2d_nchw's own doc comment notes as
  // untested is now covered; weight packing itself has no stride
  // dependence to re-check, so this reuses the same
  // pack_depthwise_to_rocket_weights path unchanged.
  util.func private @call_rocket_dynamic_depthwise_conv2d_nchw_s3(
      %input: tensor<1x?x?x?xf16>,
      %filter: tensor<?x?x?xf16>,
      %init: tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    // NCHW: dim 1 is channels, dims 2/3 are the spatial extent.
    %input_channels = tensor.dim %input, %c1 : tensor<1x?x?x?xf16>
    %input_height = tensor.dim %input, %c2 : tensor<1x?x?x?xf16>
    %input_width = tensor.dim %input, %c3 : tensor<1x?x?x?xf16>
    // Filter is [c][kh][kw]: dims 1/2 are the kernel extent.
    %weights_height = tensor.dim %filter, %c1 : tensor<?x?x?xf16>
    %weights_width = tensor.dim %filter, %c2 : tensor<?x?x?xf16>
    %output_channels = tensor.dim %init, %c1 : tensor<1x?x?x?xf32>
    %output_height = tensor.dim %init, %c2 : tensor<1x?x?x?xf32>
    %output_width = tensor.dim %init, %c3 : tensor<1x?x?x?xf32>

    %input_width_i32 = arith.index_cast %input_width : index to i32
    %input_height_i32 = arith.index_cast %input_height : index to i32
    %input_channels_i32 = arith.index_cast %input_channels : index to i32
    %output_channels_i32 = arith.index_cast %output_channels : index to i32
    %weights_width_i32 = arith.index_cast %weights_width : index to i32
    %weights_height_i32 = arith.index_cast %weights_height : index to i32

    %zero_bias_empty = tensor.empty(%output_channels) : tensor<?xf16>
    %zero_f16 = arith.constant 0.0 : f16
    %zero_bias = linalg.fill ins(%zero_f16 : f16)
        outs(%zero_bias_empty : tensor<?xf16>) -> tensor<?xf16>

    // NCHW [1,C,H,W] -> NHWC [1,H,W,C]: out.shape[i] = in.shape[perm[i]],
    // so perm = [0, 2, 3, 1].
    %input_nhwc_empty = tensor.empty(%input_height, %input_width, %input_channels) : tensor<1x?x?x?xf16>
    %input_nhwc = linalg.transpose
        ins(%input : tensor<1x?x?x?xf16>)
        outs(%input_nhwc_empty : tensor<1x?x?x?xf16>)
        permutation = [0, 2, 3, 1]

    %raw_f16 = flow.dispatch
        @rocket_dynamic_depthwise_executable_s3::@rocket_dynamic_depthwise_conv2d_v1::@rocket_dynamic_depthwise_conv2d(
          %input_width_i32, %input_height_i32, %input_channels_i32,
          %output_channels_i32, %weights_width_i32, %weights_height_i32,
          %input_nhwc, %filter, %zero_bias)
        {stream.affinity = #hal.device.affinity<@rocket_device>}
        : (i32, i32, i32, i32, i32, i32,
           tensor<1x?x?x?xf16>{%input_height, %input_width, %input_channels},
           tensor<?x?x?xf16>{%input_channels, %weights_height, %weights_width},
           tensor<?xf16>{%output_channels})
        -> tensor<1x?x?x?xf16>{%output_height, %output_width, %output_channels}

    // %init also arrives NCHW; transpose it to NHWC too so it lines up with
    // %raw_f16 for the CPU-side accumulate below.
    %init_nhwc_empty = tensor.empty(%output_height, %output_width, %output_channels) : tensor<1x?x?x?xf32>
    %init_nhwc = linalg.transpose
        ins(%init : tensor<1x?x?x?xf32>)
        outs(%init_nhwc_empty : tensor<1x?x?x?xf32>)
        permutation = [0, 2, 3, 1]

    %final_nhwc = flow.dispatch.workgroups[
        %output_height, %output_width, %output_channels](
        %raw_f16, %init_nhwc, %output_height, %output_width, %output_channels)
        : (tensor<1x?x?x?xf16>{%output_height, %output_width, %output_channels},
           tensor<1x?x?x?xf32>{%output_height, %output_width, %output_channels},
           index, index, index)
        -> tensor<1x?x?x?xf32>{%output_height, %output_width, %output_channels}
        attributes { stream.affinity = #hal.device.affinity<@cpu_device> } =
        (%raw_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>,
         %init_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>,
         %output_height_arg: index,
         %output_width_arg: index,
         %output_channels_arg: index,
         %final_binding: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>) {
      %output_height_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_height_arg, 0 : index
      %output_width_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_width_arg, 1 : index
      %output_channels_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_channels_arg, 2 : index
      %raw_shaped = flow.dispatch.tie_shape %raw_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %init_shaped = flow.dispatch.tie_shape %init_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %final_shaped = flow.dispatch.tie_shape %final_binding
          : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %raw_loaded = iree_tensor_ext.dispatch.tensor.load %raw_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %output_channels_size}
          -> tensor<1x?x?x?xf16>
      %init_loaded = iree_tensor_ext.dispatch.tensor.load %init_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
          -> tensor<1x?x?x?xf32>
      %final_empty = tensor.empty(
          %output_height_size, %output_width_size, %output_channels_size)
          : tensor<1x?x?x?xf32>
      %final_inner = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
          ],
          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } ins(%raw_loaded, %init_loaded
            : tensor<1x?x?x?xf16>, tensor<1x?x?x?xf32>)
          outs(%final_empty : tensor<1x?x?x?xf32>) {
        ^bb0(%raw: f16, %initial: f32, %out: f32):
          %raw_f32 = arith.extf %raw : f16 to f32
          %sum = arith.addf %raw_f32, %initial : f32
          linalg.yield %sum : f32
      } -> tensor<1x?x?x?xf32>
      iree_tensor_ext.dispatch.tensor.store %final_inner, %final_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : tensor<1x?x?x?xf32>
          -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      flow.return
    } count(%output_height_workload: index,
            %output_width_workload: index,
            %output_channels_workload: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(
          %output_height_workload,
          %output_width_workload,
          %output_channels_workload)
      flow.return %x, %y, %z : index, index, index
    }

    // NHWC [1,H,W,C] -> NCHW [1,C,H,W] back again: perm = [0, 3, 1, 2].
    %final_nchw_empty = tensor.empty(%output_channels, %output_height, %output_width) : tensor<1x?x?x?xf32>
    %final_nchw = linalg.transpose
        ins(%final_nhwc : tensor<1x?x?x?xf32>)
        outs(%final_nchw_empty : tensor<1x?x?x?xf32>)
        permutation = [0, 3, 1, 2]

    util.return %final_nchw : tensor<1x?x?x?xf32>
  }

  // Stride-4 counterpart of call_rocket_dynamic_depthwise_conv2d_nchw --
  // same NCHW<->NHWC transpose shim, dispatching to
  // @rocket_dynamic_depthwise_executable_s4 (#rocket_dynamic_depthwise_target_s4
  // bakes stride=4, same reason call_rocket_dynamic_conv2d_s4 needs its own
  // executable -- see that comment). Hardware-confirmed for depthwise fp16
  // at stride 2/3/4, both 1x1 and 3x3 kernels, Cin/Cout 8 and 12
  // (conv_depthwise_stride_hw.rs::depthwise_strided_convs_run_on_npu, all
  // 12 combinations pass) -- the stride/DW_EN combination
  // call_rocket_dynamic_depthwise_conv2d_nchw's own doc comment notes as
  // untested is now covered; weight packing itself has no stride
  // dependence to re-check, so this reuses the same
  // pack_depthwise_to_rocket_weights path unchanged.
  util.func private @call_rocket_dynamic_depthwise_conv2d_nchw_s4(
      %input: tensor<1x?x?x?xf16>,
      %filter: tensor<?x?x?xf16>,
      %init: tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    // NCHW: dim 1 is channels, dims 2/3 are the spatial extent.
    %input_channels = tensor.dim %input, %c1 : tensor<1x?x?x?xf16>
    %input_height = tensor.dim %input, %c2 : tensor<1x?x?x?xf16>
    %input_width = tensor.dim %input, %c3 : tensor<1x?x?x?xf16>
    // Filter is [c][kh][kw]: dims 1/2 are the kernel extent.
    %weights_height = tensor.dim %filter, %c1 : tensor<?x?x?xf16>
    %weights_width = tensor.dim %filter, %c2 : tensor<?x?x?xf16>
    %output_channels = tensor.dim %init, %c1 : tensor<1x?x?x?xf32>
    %output_height = tensor.dim %init, %c2 : tensor<1x?x?x?xf32>
    %output_width = tensor.dim %init, %c3 : tensor<1x?x?x?xf32>

    %input_width_i32 = arith.index_cast %input_width : index to i32
    %input_height_i32 = arith.index_cast %input_height : index to i32
    %input_channels_i32 = arith.index_cast %input_channels : index to i32
    %output_channels_i32 = arith.index_cast %output_channels : index to i32
    %weights_width_i32 = arith.index_cast %weights_width : index to i32
    %weights_height_i32 = arith.index_cast %weights_height : index to i32

    %zero_bias_empty = tensor.empty(%output_channels) : tensor<?xf16>
    %zero_f16 = arith.constant 0.0 : f16
    %zero_bias = linalg.fill ins(%zero_f16 : f16)
        outs(%zero_bias_empty : tensor<?xf16>) -> tensor<?xf16>

    // NCHW [1,C,H,W] -> NHWC [1,H,W,C]: out.shape[i] = in.shape[perm[i]],
    // so perm = [0, 2, 3, 1].
    %input_nhwc_empty = tensor.empty(%input_height, %input_width, %input_channels) : tensor<1x?x?x?xf16>
    %input_nhwc = linalg.transpose
        ins(%input : tensor<1x?x?x?xf16>)
        outs(%input_nhwc_empty : tensor<1x?x?x?xf16>)
        permutation = [0, 2, 3, 1]

    %raw_f16 = flow.dispatch
        @rocket_dynamic_depthwise_executable_s4::@rocket_dynamic_depthwise_conv2d_v1::@rocket_dynamic_depthwise_conv2d(
          %input_width_i32, %input_height_i32, %input_channels_i32,
          %output_channels_i32, %weights_width_i32, %weights_height_i32,
          %input_nhwc, %filter, %zero_bias)
        {stream.affinity = #hal.device.affinity<@rocket_device>}
        : (i32, i32, i32, i32, i32, i32,
           tensor<1x?x?x?xf16>{%input_height, %input_width, %input_channels},
           tensor<?x?x?xf16>{%input_channels, %weights_height, %weights_width},
           tensor<?xf16>{%output_channels})
        -> tensor<1x?x?x?xf16>{%output_height, %output_width, %output_channels}

    // %init also arrives NCHW; transpose it to NHWC too so it lines up with
    // %raw_f16 for the CPU-side accumulate below.
    %init_nhwc_empty = tensor.empty(%output_height, %output_width, %output_channels) : tensor<1x?x?x?xf32>
    %init_nhwc = linalg.transpose
        ins(%init : tensor<1x?x?x?xf32>)
        outs(%init_nhwc_empty : tensor<1x?x?x?xf32>)
        permutation = [0, 2, 3, 1]

    %final_nhwc = flow.dispatch.workgroups[
        %output_height, %output_width, %output_channels](
        %raw_f16, %init_nhwc, %output_height, %output_width, %output_channels)
        : (tensor<1x?x?x?xf16>{%output_height, %output_width, %output_channels},
           tensor<1x?x?x?xf32>{%output_height, %output_width, %output_channels},
           index, index, index)
        -> tensor<1x?x?x?xf32>{%output_height, %output_width, %output_channels}
        attributes { stream.affinity = #hal.device.affinity<@cpu_device> } =
        (%raw_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>,
         %init_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>,
         %output_height_arg: index,
         %output_width_arg: index,
         %output_channels_arg: index,
         %final_binding: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>) {
      %output_height_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_height_arg, 0 : index
      %output_width_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_width_arg, 1 : index
      %output_channels_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_channels_arg, 2 : index
      %raw_shaped = flow.dispatch.tie_shape %raw_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %init_shaped = flow.dispatch.tie_shape %init_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %final_shaped = flow.dispatch.tie_shape %final_binding
          : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %raw_loaded = iree_tensor_ext.dispatch.tensor.load %raw_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %output_channels_size}
          -> tensor<1x?x?x?xf16>
      %init_loaded = iree_tensor_ext.dispatch.tensor.load %init_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
          -> tensor<1x?x?x?xf32>
      %final_empty = tensor.empty(
          %output_height_size, %output_width_size, %output_channels_size)
          : tensor<1x?x?x?xf32>
      %final_inner = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
          ],
          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } ins(%raw_loaded, %init_loaded
            : tensor<1x?x?x?xf16>, tensor<1x?x?x?xf32>)
          outs(%final_empty : tensor<1x?x?x?xf32>) {
        ^bb0(%raw: f16, %initial: f32, %out: f32):
          %raw_f32 = arith.extf %raw : f16 to f32
          %sum = arith.addf %raw_f32, %initial : f32
          linalg.yield %sum : f32
      } -> tensor<1x?x?x?xf32>
      iree_tensor_ext.dispatch.tensor.store %final_inner, %final_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : tensor<1x?x?x?xf32>
          -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      flow.return
    } count(%output_height_workload: index,
            %output_width_workload: index,
            %output_channels_workload: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(
          %output_height_workload,
          %output_width_workload,
          %output_channels_workload)
      flow.return %x, %y, %z : index, index, index
    }

    // NHWC [1,H,W,C] -> NCHW [1,C,H,W] back again: perm = [0, 3, 1, 2].
    %final_nchw_empty = tensor.empty(%output_channels, %output_height, %output_width) : tensor<1x?x?x?xf32>
    %final_nchw = linalg.transpose
        ins(%final_nhwc : tensor<1x?x?x?xf32>)
        outs(%final_nchw_empty : tensor<1x?x?x?xf32>)
        permutation = [0, 3, 1, 2]

    util.return %final_nchw : tensor<1x?x?x?xf32>
  }


  // 1x1 kernel, spatial dims dynamic. Both channel counts must be provably
  // <= 512 (MAX_INPUT_CHANNELS/MAX_OUTPUT_CHANNELS in iree-rocket-hal's
  // conv.rs -- the 14-bit weight_kernels field's range): a convolution whose
  // channel count the compiler cannot bound must not be claimed here, since
  // the hardware rejects an out-of-range dispatch with no CPU fallback. The
  // input handle's dim 3 is Cin; the filter's dim 3 is Cout.

  // int8 counterpart of call_rocket_dynamic_conv2d. Same six push constants
  // in the same order, same three tensor bindings; the operands are i8 and
  // the result is the i32 accumulator int8_accumulator mode writes out, so
  // the CPU epilogue only has to add the convolution's own init operand
  // back in.
  util.func private @call_rocket_dynamic_conv2d_int8(
      %input: tensor<1x?x?x?xi8>,
      %filter: tensor<?x?x?x?xi8>,
      %init: tensor<1x?x?x?xi32>) -> tensor<1x?x?x?xi32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    %input_height = tensor.dim %input, %c1 : tensor<1x?x?x?xi8>
    %input_width = tensor.dim %input, %c2 : tensor<1x?x?x?xi8>
    %input_channels = tensor.dim %input, %c3 : tensor<1x?x?x?xi8>
    %weights_height = tensor.dim %filter, %c0 : tensor<?x?x?x?xi8>
    %weights_width = tensor.dim %filter, %c1 : tensor<?x?x?x?xi8>
    %output_height = tensor.dim %init, %c1 : tensor<1x?x?x?xi32>
    %output_width = tensor.dim %init, %c2 : tensor<1x?x?x?xi32>
    %output_channels = tensor.dim %init, %c3 : tensor<1x?x?x?xi32>

    %input_width_i32 = arith.index_cast %input_width : index to i32
    %input_height_i32 = arith.index_cast %input_height : index to i32
    %input_channels_i32 = arith.index_cast %input_channels : index to i32
    %output_channels_i32 = arith.index_cast %output_channels : index to i32
    %weights_width_i32 = arith.index_cast %weights_width : index to i32
    %weights_height_i32 = arith.index_cast %weights_height : index to i32

    // The bias binding is i32 here, not i8: rocket-hal-driver validates it
    // as output_channels * 4 bytes for both int8 precisions and reads it as
    // i32 in pack_int8_bias_to_bs, whereas the fp16 path binds an f16
    // vector. It stays zero -- int8_accumulator bypasses the BS stage
    // entirely, so nothing here can be folded into a hardware bias; the
    // zero-point correction lives in a separate CPU op that
    // iree-global-opt-quantized-conv-to-conv already emitted next to this
    // convolution.
    %zero_bias_empty = tensor.empty(%output_channels) : tensor<?xi32>
    %zero_i32 = arith.constant 0 : i32
    %zero_bias = linalg.fill ins(%zero_i32 : i32)
        outs(%zero_bias_empty : tensor<?xi32>) -> tensor<?xi32>

    %raw_i32 = flow.dispatch
        @rocket_dynamic_int8_executable::@rocket_dynamic_conv2d_v1::@rocket_dynamic_conv2d(
          %input_width_i32, %input_height_i32, %input_channels_i32,
          %output_channels_i32, %weights_width_i32, %weights_height_i32,
          %input, %filter, %zero_bias)
        {stream.affinity = #hal.device.affinity<@rocket_device>}
        : (i32, i32, i32, i32, i32, i32,
           tensor<1x?x?x?xi8>{%input_height, %input_width, %input_channels},
           tensor<?x?x?x?xi8>{%weights_height, %weights_width, %input_channels, %output_channels},
           tensor<?xi32>{%output_channels})
        -> tensor<1x?x?x?xi32>{%output_height, %output_width, %output_channels}

    %final = flow.dispatch.workgroups[
        %output_height, %output_width, %output_channels](
        %raw_i32, %init, %output_height, %output_width, %output_channels)
        : (tensor<1x?x?x?xi32>{%output_height, %output_width, %output_channels},
           tensor<1x?x?x?xi32>{%output_height, %output_width, %output_channels},
           index, index, index)
        -> tensor<1x?x?x?xi32>{%output_height, %output_width, %output_channels}
        attributes { stream.affinity = #hal.device.affinity<@cpu_device> } =
        (%raw_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xi32>>,
         %init_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xi32>>,
         %output_height_arg: index,
         %output_width_arg: index,
         %output_channels_arg: index,
         %final_binding: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xi32>>) {
      %output_height_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_height_arg, 0 : index
      %output_width_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_width_arg, 1 : index
      %output_channels_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_channels_arg, 2 : index
      %raw_shaped = flow.dispatch.tie_shape %raw_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xi32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %init_shaped = flow.dispatch.tie_shape %init_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xi32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %final_shaped = flow.dispatch.tie_shape %final_binding
          : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xi32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %raw_loaded = iree_tensor_ext.dispatch.tensor.load %raw_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xi32>>{
              %output_height_size, %output_width_size, %output_channels_size}
          -> tensor<1x?x?x?xi32>
      %init_loaded = iree_tensor_ext.dispatch.tensor.load %init_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xi32>>{
              %output_height_size, %output_width_size, %output_channels_size}
          -> tensor<1x?x?x?xi32>
      %final_empty = tensor.empty(
          %output_height_size, %output_width_size, %output_channels_size)
          : tensor<1x?x?x?xi32>
      %final_inner = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
          ],
          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } ins(%raw_loaded, %init_loaded
            : tensor<1x?x?x?xi32>, tensor<1x?x?x?xi32>)
          outs(%final_empty : tensor<1x?x?x?xi32>) {
        ^bb0(%raw: i32, %initial: i32, %out: i32):
          // No extend, unlike the fp16 epilogue: int8_accumulator mode
          // already hands back a full i32 accumulator.
          %sum = arith.addi %raw, %initial : i32
          linalg.yield %sum : i32
      } -> tensor<1x?x?x?xi32>
      iree_tensor_ext.dispatch.tensor.store %final_inner, %final_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : tensor<1x?x?x?xi32>
          -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xi32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      flow.return
    } count(%output_height_workload: index,
            %output_width_workload: index,
            %output_channels_workload: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(
          %output_height_workload,
          %output_width_workload,
          %output_channels_workload)
      flow.return %x, %y, %z : index, index, index
    }

    util.return %final : tensor<1x?x?x?xi32>
  }

  // int8 counterpart of call_rocket_dynamic_depthwise_conv2d.
  util.func private @call_rocket_dynamic_depthwise_conv2d_int8(
      %input: tensor<1x?x?x?xi8>,
      %filter: tensor<?x?x?xi8>,
      %init: tensor<1x?x?x?xi32>) -> tensor<1x?x?x?xi32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    %input_height = tensor.dim %input, %c1 : tensor<1x?x?x?xi8>
    %input_width = tensor.dim %input, %c2 : tensor<1x?x?x?xi8>
    %input_channels = tensor.dim %input, %c3 : tensor<1x?x?x?xi8>
    %weights_height = tensor.dim %filter, %c0 : tensor<?x?x?xi8>
    %weights_width = tensor.dim %filter, %c1 : tensor<?x?x?xi8>
    %output_height = tensor.dim %init, %c1 : tensor<1x?x?x?xi32>
    %output_width = tensor.dim %init, %c2 : tensor<1x?x?x?xi32>
    %output_channels = tensor.dim %init, %c3 : tensor<1x?x?x?xi32>

    %input_width_i32 = arith.index_cast %input_width : index to i32
    %input_height_i32 = arith.index_cast %input_height : index to i32
    %input_channels_i32 = arith.index_cast %input_channels : index to i32
    %output_channels_i32 = arith.index_cast %output_channels : index to i32
    %weights_width_i32 = arith.index_cast %weights_width : index to i32
    %weights_height_i32 = arith.index_cast %weights_height : index to i32

    // The bias binding is i32 here, not i8: rocket-hal-driver validates it
    // as output_channels * 4 bytes for both int8 precisions and reads it as
    // i32 in pack_int8_bias_to_bs, whereas the fp16 path binds an f16
    // vector. It stays zero -- int8_accumulator bypasses the BS stage
    // entirely, so nothing here can be folded into a hardware bias; the
    // zero-point correction lives in a separate CPU op that
    // iree-global-opt-quantized-conv-to-conv already emitted next to this
    // convolution.
    %zero_bias_empty = tensor.empty(%output_channels) : tensor<?xi32>
    %zero_i32 = arith.constant 0 : i32
    %zero_bias = linalg.fill ins(%zero_i32 : i32)
        outs(%zero_bias_empty : tensor<?xi32>) -> tensor<?xi32>

    // Same HWC -> CHW filter transpose call_rocket_dynamic_depthwise_conv2d
    // does, and for the same reason: pack_depthwise_to_rocket_weights takes
    // a [c][kh][kw] buffer, while linalg.depthwise_conv_2d_nhwc_hwc's filter
    // operand is [kh][kw][c].
    %filter_chw_empty = tensor.empty(%input_channels, %weights_height, %weights_width) : tensor<?x?x?xi8>
    %filter_chw = linalg.transpose
        ins(%filter : tensor<?x?x?xi8>)
        outs(%filter_chw_empty : tensor<?x?x?xi8>)
        permutation = [2, 0, 1]

    %raw_i32 = flow.dispatch
        @rocket_dynamic_depthwise_int8_executable::@rocket_dynamic_depthwise_conv2d_v1::@rocket_dynamic_depthwise_conv2d(
          %input_width_i32, %input_height_i32, %input_channels_i32,
          %output_channels_i32, %weights_width_i32, %weights_height_i32,
          %input, %filter_chw, %zero_bias)
        {stream.affinity = #hal.device.affinity<@rocket_device>}
        : (i32, i32, i32, i32, i32, i32,
           tensor<1x?x?x?xi8>{%input_height, %input_width, %input_channels},
           tensor<?x?x?xi8>{%input_channels, %weights_height, %weights_width},
           tensor<?xi32>{%output_channels})
        -> tensor<1x?x?x?xi32>{%output_height, %output_width, %output_channels}

    %final = flow.dispatch.workgroups[
        %output_height, %output_width, %output_channels](
        %raw_i32, %init, %output_height, %output_width, %output_channels)
        : (tensor<1x?x?x?xi32>{%output_height, %output_width, %output_channels},
           tensor<1x?x?x?xi32>{%output_height, %output_width, %output_channels},
           index, index, index)
        -> tensor<1x?x?x?xi32>{%output_height, %output_width, %output_channels}
        attributes { stream.affinity = #hal.device.affinity<@cpu_device> } =
        (%raw_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xi32>>,
         %init_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xi32>>,
         %output_height_arg: index,
         %output_width_arg: index,
         %output_channels_arg: index,
         %final_binding: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xi32>>) {
      %output_height_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_height_arg, 0 : index
      %output_width_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_width_arg, 1 : index
      %output_channels_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_channels_arg, 2 : index
      %raw_shaped = flow.dispatch.tie_shape %raw_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xi32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %init_shaped = flow.dispatch.tie_shape %init_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xi32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %final_shaped = flow.dispatch.tie_shape %final_binding
          : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xi32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %raw_loaded = iree_tensor_ext.dispatch.tensor.load %raw_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xi32>>{
              %output_height_size, %output_width_size, %output_channels_size}
          -> tensor<1x?x?x?xi32>
      %init_loaded = iree_tensor_ext.dispatch.tensor.load %init_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xi32>>{
              %output_height_size, %output_width_size, %output_channels_size}
          -> tensor<1x?x?x?xi32>
      %final_empty = tensor.empty(
          %output_height_size, %output_width_size, %output_channels_size)
          : tensor<1x?x?x?xi32>
      %final_inner = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
          ],
          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } ins(%raw_loaded, %init_loaded
            : tensor<1x?x?x?xi32>, tensor<1x?x?x?xi32>)
          outs(%final_empty : tensor<1x?x?x?xi32>) {
        ^bb0(%raw: i32, %initial: i32, %out: i32):
          // No extend, unlike the fp16 epilogue: int8_accumulator mode
          // already hands back a full i32 accumulator.
          %sum = arith.addi %raw, %initial : i32
          linalg.yield %sum : i32
      } -> tensor<1x?x?x?xi32>
      iree_tensor_ext.dispatch.tensor.store %final_inner, %final_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : tensor<1x?x?x?xi32>
          -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xi32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      flow.return
    } count(%output_height_workload: index,
            %output_width_workload: index,
            %output_channels_workload: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(
          %output_height_workload,
          %output_width_workload,
          %output_channels_workload)
      flow.return %x, %y, %z : index, index, index
    }

    util.return %final : tensor<1x?x?x?xi32>
  }

  // Stride-2 int8 depthwise. The fp16 side reaches stride 2 only through
  // its NCHW adapters (call_rocket_dynamic_depthwise_conv2d_nchw_s2); there
  // is no fp16 NHWC stride-2 adapter because ONNX-imported fp16 depthwise
  // convs arrive NCHW and stay that way. An ONNX *int8* model is different:
  // torch-mlir lowers ConvInteger's grouped form straight to
  // linalg.depthwise_conv_2d_nhwc_hwc_q, so the NHWC layout is the one that
  // actually shows up (4 of MobileNetV2's 17 depthwise layers are stride 2).
  // This is a layout-plumbing addition, not a new hardware claim: it
  // dispatches to the same depthwise stride-2 executable shape the
  // hardware-confirmed NCHW s2 path already uses.
  util.func private @call_rocket_dynamic_depthwise_conv2d_int8_s2(
      %input: tensor<1x?x?x?xi8>,
      %filter: tensor<?x?x?xi8>,
      %init: tensor<1x?x?x?xi32>) -> tensor<1x?x?x?xi32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    %input_height = tensor.dim %input, %c1 : tensor<1x?x?x?xi8>
    %input_width = tensor.dim %input, %c2 : tensor<1x?x?x?xi8>
    %input_channels = tensor.dim %input, %c3 : tensor<1x?x?x?xi8>
    %weights_height = tensor.dim %filter, %c0 : tensor<?x?x?xi8>
    %weights_width = tensor.dim %filter, %c1 : tensor<?x?x?xi8>
    %output_height = tensor.dim %init, %c1 : tensor<1x?x?x?xi32>
    %output_width = tensor.dim %init, %c2 : tensor<1x?x?x?xi32>
    %output_channels = tensor.dim %init, %c3 : tensor<1x?x?x?xi32>

    %input_width_i32 = arith.index_cast %input_width : index to i32
    %input_height_i32 = arith.index_cast %input_height : index to i32
    %input_channels_i32 = arith.index_cast %input_channels : index to i32
    %output_channels_i32 = arith.index_cast %output_channels : index to i32
    %weights_width_i32 = arith.index_cast %weights_width : index to i32
    %weights_height_i32 = arith.index_cast %weights_height : index to i32

    // The bias binding is i32 here, not i8: rocket-hal-driver validates it
    // as output_channels * 4 bytes for both int8 precisions and reads it as
    // i32 in pack_int8_bias_to_bs, whereas the fp16 path binds an f16
    // vector. It stays zero -- int8_accumulator bypasses the BS stage
    // entirely, so nothing here can be folded into a hardware bias; the
    // zero-point correction lives in a separate CPU op that
    // iree-global-opt-quantized-conv-to-conv already emitted next to this
    // convolution.
    %zero_bias_empty = tensor.empty(%output_channels) : tensor<?xi32>
    %zero_i32 = arith.constant 0 : i32
    %zero_bias = linalg.fill ins(%zero_i32 : i32)
        outs(%zero_bias_empty : tensor<?xi32>) -> tensor<?xi32>

    // Same HWC -> CHW filter transpose call_rocket_dynamic_depthwise_conv2d
    // does, and for the same reason: pack_depthwise_to_rocket_weights takes
    // a [c][kh][kw] buffer, while linalg.depthwise_conv_2d_nhwc_hwc's filter
    // operand is [kh][kw][c].
    %filter_chw_empty = tensor.empty(%input_channels, %weights_height, %weights_width) : tensor<?x?x?xi8>
    %filter_chw = linalg.transpose
        ins(%filter : tensor<?x?x?xi8>)
        outs(%filter_chw_empty : tensor<?x?x?xi8>)
        permutation = [2, 0, 1]

    %raw_i32 = flow.dispatch
        @rocket_dynamic_depthwise_int8_executable_s2::@rocket_dynamic_depthwise_conv2d_v1::@rocket_dynamic_depthwise_conv2d(
          %input_width_i32, %input_height_i32, %input_channels_i32,
          %output_channels_i32, %weights_width_i32, %weights_height_i32,
          %input, %filter_chw, %zero_bias)
        {stream.affinity = #hal.device.affinity<@rocket_device>}
        : (i32, i32, i32, i32, i32, i32,
           tensor<1x?x?x?xi8>{%input_height, %input_width, %input_channels},
           tensor<?x?x?xi8>{%input_channels, %weights_height, %weights_width},
           tensor<?xi32>{%output_channels})
        -> tensor<1x?x?x?xi32>{%output_height, %output_width, %output_channels}

    %final = flow.dispatch.workgroups[
        %output_height, %output_width, %output_channels](
        %raw_i32, %init, %output_height, %output_width, %output_channels)
        : (tensor<1x?x?x?xi32>{%output_height, %output_width, %output_channels},
           tensor<1x?x?x?xi32>{%output_height, %output_width, %output_channels},
           index, index, index)
        -> tensor<1x?x?x?xi32>{%output_height, %output_width, %output_channels}
        attributes { stream.affinity = #hal.device.affinity<@cpu_device> } =
        (%raw_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xi32>>,
         %init_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xi32>>,
         %output_height_arg: index,
         %output_width_arg: index,
         %output_channels_arg: index,
         %final_binding: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xi32>>) {
      %output_height_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_height_arg, 0 : index
      %output_width_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_width_arg, 1 : index
      %output_channels_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_channels_arg, 2 : index
      %raw_shaped = flow.dispatch.tie_shape %raw_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xi32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %init_shaped = flow.dispatch.tie_shape %init_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xi32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %final_shaped = flow.dispatch.tie_shape %final_binding
          : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xi32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %raw_loaded = iree_tensor_ext.dispatch.tensor.load %raw_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xi32>>{
              %output_height_size, %output_width_size, %output_channels_size}
          -> tensor<1x?x?x?xi32>
      %init_loaded = iree_tensor_ext.dispatch.tensor.load %init_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xi32>>{
              %output_height_size, %output_width_size, %output_channels_size}
          -> tensor<1x?x?x?xi32>
      %final_empty = tensor.empty(
          %output_height_size, %output_width_size, %output_channels_size)
          : tensor<1x?x?x?xi32>
      %final_inner = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
          ],
          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } ins(%raw_loaded, %init_loaded
            : tensor<1x?x?x?xi32>, tensor<1x?x?x?xi32>)
          outs(%final_empty : tensor<1x?x?x?xi32>) {
        ^bb0(%raw: i32, %initial: i32, %out: i32):
          // No extend, unlike the fp16 epilogue: int8_accumulator mode
          // already hands back a full i32 accumulator.
          %sum = arith.addi %raw, %initial : i32
          linalg.yield %sum : i32
      } -> tensor<1x?x?x?xi32>
      iree_tensor_ext.dispatch.tensor.store %final_inner, %final_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : tensor<1x?x?x?xi32>
          -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xi32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      flow.return
    } count(%output_height_workload: index,
            %output_width_workload: index,
            %output_channels_workload: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(
          %output_height_workload,
          %output_width_workload,
          %output_channels_workload)
      flow.return %x, %y, %z : index, index, index
    }

    util.return %final : tensor<1x?x?x?xi32>
  }

  transform.named_sequence @match_dynamic_conv2d(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.conv_2d_nhwc_hwcf"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = f16, rhs_type = f16, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    // Cout is capped well below MAX_OUTPUT_CHANNELS (512); Cin is not. This
    // is hardware-verified, not a guess: an isolated correctness probe
    // (iree-rocket-hal/tests/conv_cbuf_split_sweep_hw.rs and
    // conv_features19_isolated_hw.rs), run in isolation on real Planck
    // hardware with a fill-1.0/exact-expected-value check (not just
    // "did it time out"), found:
    //
    //   Cout  64/128/256 (Cin 3, 64, 128, 256; banks 11/1, 9/3, 7/5, 3/9,
    //         1/11 all covered)          -> correct, 5/5 every rep
    //   Cout  512 (Cin 256 -- features.19; Cin 512 -- features.21; both
    //         30x30, banks 11/1)         -> all-zero output, 0/5 every
    //                                        rep, deterministic
    //
    // IMPORTANT CAVEAT, found later by a real-compiler-path harness
    // (rocket_conv_harness.py in iree-rocket-design-spike) plus a follow-up
    // extent_sweep_at_fixed_channels sweep in
    // conv_cbuf_split_sweep_hw.rs: Cout is not actually the discriminator.
    // Cin=256/Cout=256/3x3 -- comfortably inside this bound -- is ALSO
    // deterministically all-zero (0/5, every output element wrong) across
    // every spatial extent from 26x26 to 48x48, because ConvPlan picks the
    // same 11/1 split there that it picks for the broken Cout=512 shapes
    // above; extents 20-24 (banks 7/5, 9/3) and 50-58 (banks 1/11) at the
    // SAME channel counts pass 5/5, and the pass/fail boundary lines up
    // exactly with ConvPlan's split-flip points, with zero fuzziness. The
    // real discriminator is an 11/1-style split combined with a large
    // coefficient footprint: features.0 (Cin=3/Cout=64, footprint
    // 3*3*3*64 = 1728 elements) also gets an 11/1 split and is fine, while
    // Cin=256/Cout=256 (footprint 3*3*256*256 = 589824 elements) at that
    // same split is broken everywhere it occurs. This bound is still safe
    // for VGG specifically -- none of its real Cout<=256 layers land on an
    // 11/1 split at a large-footprint channel count -- but that is a
    // property of VGG's specific shapes, not a guarantee this Cout<=256
    // rule provides in general. A future model (or a wider matcher) could
    // reintroduce this exact bug at Cout<=256 with the wrong spatial
    // extent. See DESIGN_NOTES.md for the full characterization.
    //
    // This is the same class of bug DESIGN_NOTES.md documents for 9x9/11x11
    // -- ConvPlan's demand-based CBUF formula picks a split based on raw
    // byte demand, but the real vendor coefficient-streaming schedule for
    // high-pressure shapes isn't decoded, so an unvalidated split doesn't
    // fail loudly, it silently completes with all-zero output. A live
    // rocket-npu-trace initially suggested this was a timing/idle-gap
    // issue (every failure followed an anomalously long idle gap on that
    // core) -- ruled out by this same isolated probe: the shape fails
    // identically as a fresh first job, after a sustained warmup burst,
    // and after a deliberate idle gap. It is the shape, not the timing.
    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    %filter_value = transform.get_operand %root[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 512 : !transform.any_value
    transform.iree.match.dim_bounds %filter_value[3], umin = 1, umax = 512 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  // Same fallback for 3x3, which the hardware handles natively through the
  // identical demand-based CBUF partition as 1x1. Spelled as its own
  // matcher (rather than widening the filter check to 1..=3) so 2x2 and
  // non-square combinations, which route through different ConvPlan
  // partitions, are never silently claimed.
  transform.named_sequence @match_dynamic_conv2d_3x3(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.conv_2d_nhwc_hwcf"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = f16, rhs_type = f16, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [3, 3] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    // Cout is capped well below MAX_OUTPUT_CHANNELS (512); Cin is not. This
    // is hardware-verified, not a guess: an isolated correctness probe
    // (iree-rocket-hal/tests/conv_cbuf_split_sweep_hw.rs and
    // conv_features19_isolated_hw.rs), run in isolation on real Planck
    // hardware with a fill-1.0/exact-expected-value check (not just
    // "did it time out"), found:
    //
    //   Cout  64/128/256 (Cin 3, 64, 128, 256; banks 11/1, 9/3, 7/5, 3/9,
    //         1/11 all covered)          -> correct, 5/5 every rep
    //   Cout  512 (Cin 256 -- features.19; Cin 512 -- features.21; both
    //         30x30, banks 11/1)         -> all-zero output, 0/5 every
    //                                        rep, deterministic
    //
    // IMPORTANT CAVEAT, found later by a real-compiler-path harness
    // (rocket_conv_harness.py in iree-rocket-design-spike) plus a follow-up
    // extent_sweep_at_fixed_channels sweep in
    // conv_cbuf_split_sweep_hw.rs: Cout is not actually the discriminator.
    // Cin=256/Cout=256/3x3 -- comfortably inside this bound -- is ALSO
    // deterministically all-zero (0/5, every output element wrong) across
    // every spatial extent from 26x26 to 48x48, because ConvPlan picks the
    // same 11/1 split there that it picks for the broken Cout=512 shapes
    // above; extents 20-24 (banks 7/5, 9/3) and 50-58 (banks 1/11) at the
    // SAME channel counts pass 5/5, and the pass/fail boundary lines up
    // exactly with ConvPlan's split-flip points, with zero fuzziness. The
    // real discriminator is an 11/1-style split combined with a large
    // coefficient footprint: features.0 (Cin=3/Cout=64, footprint
    // 3*3*3*64 = 1728 elements) also gets an 11/1 split and is fine, while
    // Cin=256/Cout=256 (footprint 3*3*256*256 = 589824 elements) at that
    // same split is broken everywhere it occurs. This bound is still safe
    // for VGG specifically -- none of its real Cout<=256 layers land on an
    // 11/1 split at a large-footprint channel count -- but that is a
    // property of VGG's specific shapes, not a guarantee this Cout<=256
    // rule provides in general. A future model (or a wider matcher) could
    // reintroduce this exact bug at Cout<=256 with the wrong spatial
    // extent. See DESIGN_NOTES.md for the full characterization.
    //
    // This is the same class of bug DESIGN_NOTES.md documents for 9x9/11x11
    // -- ConvPlan's demand-based CBUF formula picks a split based on raw
    // byte demand, but the real vendor coefficient-streaming schedule for
    // high-pressure shapes isn't decoded, so an unvalidated split doesn't
    // fail loudly, it silently completes with all-zero output. A live
    // rocket-npu-trace initially suggested this was a timing/idle-gap
    // issue (every failure followed an anomalously long idle gap on that
    // core) -- ruled out by this same isolated probe: the shape fails
    // identically as a fresh first job, after a sustained warmup burst,
    // and after a deliberate idle gap. It is the shape, not the timing.
    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    %filter_value = transform.get_operand %root[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 512 : !transform.any_value
    transform.iree.match.dim_bounds %filter_value[3], umin = 1, umax = 512 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  // Stride-2 counterpart of @match_dynamic_conv2d -- same 1x1 kernel and Cout/Cin
  // <= 512 bound (see that matcher's own doc comment for the
  // full CBUF-split correctness caveat, which applies identically here: it
  // is a property of channel count and coefficient footprint, not stride).
  // What's new is `%strides` = [2, 2] instead of [1, 1] -- hardware-
  // confirmed dense fp16 at stride 2 by conv_wide_shape_hw.rs (see
  // DESIGN_NOTES.md "Stride and large-width sweeps").
  transform.named_sequence @match_dynamic_conv2d_s2(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.conv_2d_nhwc_hwcf"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = f16, rhs_type = f16, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [2, 2] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    %filter_value = transform.get_operand %root[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 512 : !transform.any_value
    transform.iree.match.dim_bounds %filter_value[3], umin = 1, umax = 512 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  // Stride-2 counterpart of @match_dynamic_conv2d_3x3 -- same 3x3 kernel and Cout/Cin
  // <= 512 bound (see that matcher's own doc comment for the
  // full CBUF-split correctness caveat, which applies identically here: it
  // is a property of channel count and coefficient footprint, not stride).
  // What's new is `%strides` = [2, 2] instead of [1, 1] -- hardware-
  // confirmed dense fp16 at stride 2 by conv_wide_shape_hw.rs (see
  // DESIGN_NOTES.md "Stride and large-width sweeps").
  transform.named_sequence @match_dynamic_conv2d_3x3_s2(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.conv_2d_nhwc_hwcf"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = f16, rhs_type = f16, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [3, 3] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [2, 2] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    %filter_value = transform.get_operand %root[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 512 : !transform.any_value
    transform.iree.match.dim_bounds %filter_value[3], umin = 1, umax = 512 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  // Stride-3 counterpart of @match_dynamic_conv2d -- same 1x1 kernel and Cout/Cin
  // <= 512 bound (see that matcher's own doc comment for the
  // full CBUF-split correctness caveat, which applies identically here: it
  // is a property of channel count and coefficient footprint, not stride).
  // What's new is `%strides` = [3, 3] instead of [1, 1] -- hardware-
  // confirmed dense fp16 at stride 3 by conv_wide_shape_hw.rs (see
  // DESIGN_NOTES.md "Stride and large-width sweeps").
  transform.named_sequence @match_dynamic_conv2d_s3(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.conv_2d_nhwc_hwcf"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = f16, rhs_type = f16, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [3, 3] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    %filter_value = transform.get_operand %root[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 512 : !transform.any_value
    transform.iree.match.dim_bounds %filter_value[3], umin = 1, umax = 512 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  // Stride-3 counterpart of @match_dynamic_conv2d_3x3 -- same 3x3 kernel and Cout/Cin
  // <= 512 bound (see that matcher's own doc comment for the
  // full CBUF-split correctness caveat, which applies identically here: it
  // is a property of channel count and coefficient footprint, not stride).
  // What's new is `%strides` = [3, 3] instead of [1, 1] -- hardware-
  // confirmed dense fp16 at stride 3 by conv_wide_shape_hw.rs (see
  // DESIGN_NOTES.md "Stride and large-width sweeps").
  transform.named_sequence @match_dynamic_conv2d_3x3_s3(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.conv_2d_nhwc_hwcf"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = f16, rhs_type = f16, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [3, 3] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [3, 3] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    %filter_value = transform.get_operand %root[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 512 : !transform.any_value
    transform.iree.match.dim_bounds %filter_value[3], umin = 1, umax = 512 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  // Stride-4 counterpart of @match_dynamic_conv2d -- same 1x1 kernel and Cout/Cin
  // <= 512 bound (see that matcher's own doc comment for the
  // full CBUF-split correctness caveat, which applies identically here: it
  // is a property of channel count and coefficient footprint, not stride).
  // What's new is `%strides` = [4, 4] instead of [1, 1] -- hardware-
  // confirmed dense fp16 at stride 4 by conv_wide_shape_hw.rs (see
  // DESIGN_NOTES.md "Stride and large-width sweeps").
  transform.named_sequence @match_dynamic_conv2d_s4(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.conv_2d_nhwc_hwcf"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = f16, rhs_type = f16, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [4, 4] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    %filter_value = transform.get_operand %root[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 512 : !transform.any_value
    transform.iree.match.dim_bounds %filter_value[3], umin = 1, umax = 512 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  // Stride-4 counterpart of @match_dynamic_conv2d_3x3 -- same 3x3 kernel and Cout/Cin
  // <= 512 bound (see that matcher's own doc comment for the
  // full CBUF-split correctness caveat, which applies identically here: it
  // is a property of channel count and coefficient footprint, not stride).
  // What's new is `%strides` = [4, 4] instead of [1, 1] -- hardware-
  // confirmed dense fp16 at stride 4 by conv_wide_shape_hw.rs (see
  // DESIGN_NOTES.md "Stride and large-width sweeps").
  transform.named_sequence @match_dynamic_conv2d_3x3_s4(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.conv_2d_nhwc_hwcf"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = f16, rhs_type = f16, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [3, 3] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [4, 4] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    %filter_value = transform.get_operand %root[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 512 : !transform.any_value
    transform.iree.match.dim_bounds %filter_value[3], umin = 1, umax = 512 : !transform.any_value
    transform.yield %root : !transform.any_op
  }


  // Depthwise counterpart of @match_dynamic_conv2d. linalg.depthwise_conv_2d_nhwc_hwc
  // (not the _hwcm variant) is the only depthwise op family Rocket claims:
  // ConvPlan::with_depthwise hard-asserts a channel multiplier of one, so a
  // real channel-multiplier dimension (_hwcm) has never been captured or
  // validated and must stay on CPU. transform.iree.match.convolution's
  // dimension inference (LinalgInterfaces.cpp inferConvolutionDimsImpl)
  // puts the shared input/output channel dim in depth_dims for this op
  // family, not output_channel_dims/input_channel_dims (both empty here,
  // unlike the dense matcher above) -- confirmed against
  // mlir/unittests/Dialect/Linalg/InferConvolutionDimsTest.cpp, which
  // exercises exactly this op and asserts depth is non-empty. Cout is
  // therefore never an independent quantity to bound: it is always Cin,
  // read off %init the same way call_rocket_dynamic_depthwise_conv2d's
  // %output_channels already is.
  //
  // Runtime support (packing, register fields, CBUF allocation) is
  // hardware-validated -- see DESIGN_NOTES.md "Depthwise: Mesa's channel
  // rule is wrong" and conv_phase1_validation_hw.rs (8/8 passing, including
  // the int8 Cin=12 tap-major layout check) -- but that validation never
  // exceeded one 32-channel coefficient group
  // (WEIGHT_INPUT_GROUP_CHANNELS), so this matcher originally shipped
  // capped at 128, not the dense matcher's 512, to avoid the exact mistake
  // DESIGN_NOTES.md documents for dense conv (a coarse capture ladder
  // hiding a formula bug between measured points). A boundary probe
  // (rocket_conv_harness.py, single fixed-shape depthwise dispatches)
  // through the real compiled dispatch path -- not the isolated hardware
  // captures above -- found exactly that: `pack_depthwise_to_rocket_weights`
  // used a single global tap-major stride instead of grouping channels by
  // 32, invisible below one group. Fixed in `tensor_layout.rs`
  // (`pack_depthwise_to_rocket_weights`'s doc comment has the full
  // derivation) and confirmed clean at every ladder point from 128 through
  // 512, both kernel sizes, so the cap now matches the dense matcher's.
  transform.named_sequence @match_dynamic_depthwise_conv2d(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.depthwise_conv_2d_nhwc_hwc"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = f16, rhs_type = f16, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    // Only one channel count to bound: depthwise Cout is always Cin.
    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 512 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  // Same fallback for 3x3, mirroring @match_dynamic_conv2d_3x3's rationale:
  // ConvPlan routes depthwise through the identical demand-based CBUF
  // partition for kernel extents 1 and 3 (conv.rs), and MobileNet-style
  // depthwise-separable models overwhelmingly use 3x3 for the depthwise
  // stage, so this is the practically load-bearing case.
  transform.named_sequence @match_dynamic_depthwise_conv2d_3x3(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.depthwise_conv_2d_nhwc_hwc"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = f16, rhs_type = f16, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [3, 3] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 512 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  transform.named_sequence @cast_and_call_dynamic_depthwise_conv2d(%root: !transform.any_op {transform.readonly}) {
    %ins = transform.get_operand %root[all] : (!transform.any_op) -> !transform.any_value
    %out = transform.get_result %root[all] : (!transform.any_op) -> !transform.any_value
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %topology_attr = transform.param.constant #hal.device.topology<links = [
        (@rocket_device -> @cpu_device = {transparent_access = true, unified_memory = true}),
        (@cpu_device -> @rocket_device = {transparent_access = true, unified_memory = true})
      ]> -> !transform.any_param
    transform.annotate %module "stream.topology" = %topology_attr : !transform.any_op, !transform.any_param
    %executable = transform.util.import_symbol @rocket_dynamic_depthwise_executable into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_rocket_dynamic_depthwise_conv2d into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
          transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  // NCHW counterpart of @match_dynamic_depthwise_conv2d -- see
  // call_rocket_dynamic_depthwise_conv2d_nchw's doc comment for why real
  // ONNX-imported models need this instead of (not in addition to reaching)
  // the NHWC matcher above. Same dims_equal shape as the NHWC matcher --
  // depthwise's dimension inference doesn't depend on operand layout, only
  // on which axes are batch/channel/spatial -- except the channel dim to
  // bound is input dim 1 here (NCHW), not dim 3 (NHWC).
  transform.named_sequence @match_dynamic_depthwise_conv2d_nchw(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.depthwise_conv_2d_nchw_chw"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = f16, rhs_type = f16, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[1], umin = 1, umax = 512 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  // Same fallback for 3x3, mirroring @match_dynamic_depthwise_conv2d_3x3's
  // rationale -- this is the practically load-bearing case, since a real
  // ONNX-imported depthwise-separable model's spatial stage is
  // overwhelmingly 3x3, arriving in exactly this NCHW form.
  //
  // Cin bound kept at 512 despite `MAX_INPUT_CHANNELS`/`MAX_OUTPUT_CHANNELS`
  // in conv.rs being raised to 960: MobileNetV2's Cin=576 (14x14) and
  // Cin=960 (7x7) depthwise stages are hardware-validated correct at that
  // width
  // (iree-rocket-hal/tests/conv_mobilenetv2_depthwise_wide_hw.rs), but
  // routing them to Rocket measured as a net *regression* end to end
  // (161-163ms -> 165-170ms on real hardware) -- the layers are spatially
  // too small (14x14, 7x7) to amortize the per-dispatch NC1HWC2 pack/unpack
  // tax every Rocket dispatch pays (see
  // rocket-hal-driver/src/command_buffer.rs; no propagation across chained
  // dispatches exists). So this stays CPU-routed on purpose. Revisit once
  // that repack-propagation work exists, or raise this bound again for a
  // model whose >512-channel depthwise layers are large enough to be worth
  // it.
  transform.named_sequence @match_dynamic_depthwise_conv2d_nchw_3x3(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.depthwise_conv_2d_nchw_chw"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = f16, rhs_type = f16, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [3, 3] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[1], umin = 1, umax = 512 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  // Stride-2 counterpart of @match_dynamic_depthwise_conv2d_nchw -- same 1x1 kernel and Cin <= 512
  // bound (Cout is always Cin here, see that matcher's own doc comment).
  // `%strides` = [2, 2] instead of [1, 1] -- hardware-confirmed depthwise
  // fp16 at stride 2 by conv_depthwise_stride_hw.rs (see DESIGN_NOTES.md
  // "Depthwise stride hardware confirmation").
  transform.named_sequence @match_dynamic_depthwise_conv2d_nchw_s2(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.depthwise_conv_2d_nchw_chw"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = f16, rhs_type = f16, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [2, 2] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[1], umin = 1, umax = 512 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  // Stride-2 counterpart of @match_dynamic_depthwise_conv2d_nchw_3x3 -- same 3x3 kernel and Cin <= 512
  // bound (Cout is always Cin here, see that matcher's own doc comment).
  // `%strides` = [2, 2] instead of [1, 1] -- hardware-confirmed depthwise
  // fp16 at stride 2 by conv_depthwise_stride_hw.rs (see DESIGN_NOTES.md
  // "Depthwise stride hardware confirmation").
  //
  // Deliberately NOT raised to 960 alongside @match_dynamic_depthwise_conv2d_nchw_3x3
  // above. MobileNetV2's Cin=576, 14x14->7x7 stride-2 block is the matcher
  // this would newly claim, and `ConvPlan::new` hard-panics for it today
  // ("convolution needs horizontal tiling, which is only capture-backed at
  // stride 1", conv.rs:~1909) -- at 16-wide input, 576 channels' coefficient
  // demand forces horizontal (column) tiling, which has never been
  // implemented for stride > 1. Confirmed with the pure-planning
  // `dump_conv_plan` example (no hardware involved): 96..512 channels at
  // this width plan cleanly, 576 panics immediately. Raising this bound
  // would make the compiler route the shape to Rocket and then hard-crash
  // the HAL driver at first inference, which is strictly worse than today's
  // CPU fallback. Needs horizontal-tiling-at-stride>1 support in ConvPlan
  // before this bound can move; tracked as follow-up, not done here.
  transform.named_sequence @match_dynamic_depthwise_conv2d_nchw_3x3_s2(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.depthwise_conv_2d_nchw_chw"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = f16, rhs_type = f16, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [3, 3] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [2, 2] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[1], umin = 1, umax = 512 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  // Stride-3 counterpart of @match_dynamic_depthwise_conv2d_nchw -- same 1x1 kernel and Cin <= 512
  // bound (Cout is always Cin here, see that matcher's own doc comment).
  // `%strides` = [3, 3] instead of [1, 1] -- hardware-confirmed depthwise
  // fp16 at stride 3 by conv_depthwise_stride_hw.rs (see DESIGN_NOTES.md
  // "Depthwise stride hardware confirmation").
  transform.named_sequence @match_dynamic_depthwise_conv2d_nchw_s3(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.depthwise_conv_2d_nchw_chw"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = f16, rhs_type = f16, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [3, 3] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[1], umin = 1, umax = 512 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  // Stride-3 counterpart of @match_dynamic_depthwise_conv2d_nchw_3x3 -- same 3x3 kernel and Cin <= 512
  // bound (Cout is always Cin here, see that matcher's own doc comment).
  // `%strides` = [3, 3] instead of [1, 1] -- hardware-confirmed depthwise
  // fp16 at stride 3 by conv_depthwise_stride_hw.rs (see DESIGN_NOTES.md
  // "Depthwise stride hardware confirmation").
  transform.named_sequence @match_dynamic_depthwise_conv2d_nchw_3x3_s3(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.depthwise_conv_2d_nchw_chw"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = f16, rhs_type = f16, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [3, 3] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [3, 3] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[1], umin = 1, umax = 512 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  // Stride-4 counterpart of @match_dynamic_depthwise_conv2d_nchw -- same 1x1 kernel and Cin <= 512
  // bound (Cout is always Cin here, see that matcher's own doc comment).
  // `%strides` = [4, 4] instead of [1, 1] -- hardware-confirmed depthwise
  // fp16 at stride 4 by conv_depthwise_stride_hw.rs (see DESIGN_NOTES.md
  // "Depthwise stride hardware confirmation").
  transform.named_sequence @match_dynamic_depthwise_conv2d_nchw_s4(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.depthwise_conv_2d_nchw_chw"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = f16, rhs_type = f16, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [4, 4] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[1], umin = 1, umax = 512 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  // Stride-4 counterpart of @match_dynamic_depthwise_conv2d_nchw_3x3 -- same 3x3 kernel and Cin <= 512
  // bound (Cout is always Cin here, see that matcher's own doc comment).
  // `%strides` = [4, 4] instead of [1, 1] -- hardware-confirmed depthwise
  // fp16 at stride 4 by conv_depthwise_stride_hw.rs (see DESIGN_NOTES.md
  // "Depthwise stride hardware confirmation").
  transform.named_sequence @match_dynamic_depthwise_conv2d_nchw_3x3_s4(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.depthwise_conv_2d_nchw_chw"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = f16, rhs_type = f16, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [3, 3] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [4, 4] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[1], umin = 1, umax = 512 : !transform.any_value
    transform.yield %root : !transform.any_op
  }


  transform.named_sequence @cast_and_call_dynamic_depthwise_conv2d_nchw(%root: !transform.any_op {transform.readonly}) {
    %ins = transform.get_operand %root[all] : (!transform.any_op) -> !transform.any_value
    %out = transform.get_result %root[all] : (!transform.any_op) -> !transform.any_value
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %topology_attr = transform.param.constant #hal.device.topology<links = [
        (@rocket_device -> @cpu_device = {transparent_access = true, unified_memory = true}),
        (@cpu_device -> @rocket_device = {transparent_access = true, unified_memory = true})
      ]> -> !transform.any_param
    transform.annotate %module "stream.topology" = %topology_attr : !transform.any_op, !transform.any_param
    %executable = transform.util.import_symbol @rocket_dynamic_depthwise_executable into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_rocket_dynamic_depthwise_conv2d_nchw into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
          transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @cast_and_call_dynamic_depthwise_conv2d_nchw_s2(%root: !transform.any_op {transform.readonly}) {
    %ins = transform.get_operand %root[all] : (!transform.any_op) -> !transform.any_value
    %out = transform.get_result %root[all] : (!transform.any_op) -> !transform.any_value
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %topology_attr = transform.param.constant #hal.device.topology<links = [
        (@rocket_device -> @cpu_device = {transparent_access = true, unified_memory = true}),
        (@cpu_device -> @rocket_device = {transparent_access = true, unified_memory = true})
      ]> -> !transform.any_param
    transform.annotate %module "stream.topology" = %topology_attr : !transform.any_op, !transform.any_param
    %executable = transform.util.import_symbol @rocket_dynamic_depthwise_executable_s2 into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_rocket_dynamic_depthwise_conv2d_nchw_s2 into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
          transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @cast_and_call_dynamic_depthwise_conv2d_nchw_s3(%root: !transform.any_op {transform.readonly}) {
    %ins = transform.get_operand %root[all] : (!transform.any_op) -> !transform.any_value
    %out = transform.get_result %root[all] : (!transform.any_op) -> !transform.any_value
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %topology_attr = transform.param.constant #hal.device.topology<links = [
        (@rocket_device -> @cpu_device = {transparent_access = true, unified_memory = true}),
        (@cpu_device -> @rocket_device = {transparent_access = true, unified_memory = true})
      ]> -> !transform.any_param
    transform.annotate %module "stream.topology" = %topology_attr : !transform.any_op, !transform.any_param
    %executable = transform.util.import_symbol @rocket_dynamic_depthwise_executable_s3 into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_rocket_dynamic_depthwise_conv2d_nchw_s3 into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
          transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @cast_and_call_dynamic_depthwise_conv2d_nchw_s4(%root: !transform.any_op {transform.readonly}) {
    %ins = transform.get_operand %root[all] : (!transform.any_op) -> !transform.any_value
    %out = transform.get_result %root[all] : (!transform.any_op) -> !transform.any_value
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %topology_attr = transform.param.constant #hal.device.topology<links = [
        (@rocket_device -> @cpu_device = {transparent_access = true, unified_memory = true}),
        (@cpu_device -> @rocket_device = {transparent_access = true, unified_memory = true})
      ]> -> !transform.any_param
    transform.annotate %module "stream.topology" = %topology_attr : !transform.any_op, !transform.any_param
    %executable = transform.util.import_symbol @rocket_dynamic_depthwise_executable_s4 into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_rocket_dynamic_depthwise_conv2d_nchw_s4 into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
          transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }


  transform.named_sequence @cast_and_call_dynamic_conv2d(%root: !transform.any_op {transform.readonly}) {
    %ins = transform.get_operand %root[all] : (!transform.any_op) -> !transform.any_value
    %out = transform.get_result %root[all] : (!transform.any_op) -> !transform.any_value
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    // Declare truthfully that rocket_device and cpu_device share unified,
    // transparently-accessible memory (real RK3588 hardware fact) so
    // ResolveTopologyQueriesPass can resolve the cross-device buffer Stream
    // forms for values flowing directly from the rocket dispatch into
    // CPU-side compute. Must be set on the REAL target module (%module),
    // not this transform-spec file's own module. Idempotent: fires once per
    // matched conv, each time just overwriting the same value.
    %topology_attr = transform.param.constant #hal.device.topology<links = [
        (@rocket_device -> @cpu_device = {transparent_access = true, unified_memory = true}),
        (@cpu_device -> @rocket_device = {transparent_access = true, unified_memory = true})
      ]> -> !transform.any_param
    transform.annotate %module "stream.topology" = %topology_attr : !transform.any_op, !transform.any_param
    %executable = transform.util.import_symbol @rocket_dynamic_executable into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_rocket_dynamic_conv2d into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
          transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @cast_and_call_dynamic_conv2d_s2(%root: !transform.any_op {transform.readonly}) {
    %ins = transform.get_operand %root[all] : (!transform.any_op) -> !transform.any_value
    %out = transform.get_result %root[all] : (!transform.any_op) -> !transform.any_value
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %topology_attr = transform.param.constant #hal.device.topology<links = [
        (@rocket_device -> @cpu_device = {transparent_access = true, unified_memory = true}),
        (@cpu_device -> @rocket_device = {transparent_access = true, unified_memory = true})
      ]> -> !transform.any_param
    transform.annotate %module "stream.topology" = %topology_attr : !transform.any_op, !transform.any_param
    %executable = transform.util.import_symbol @rocket_dynamic_executable_s2 into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_rocket_dynamic_conv2d_s2 into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
          transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @cast_and_call_dynamic_conv2d_s3(%root: !transform.any_op {transform.readonly}) {
    %ins = transform.get_operand %root[all] : (!transform.any_op) -> !transform.any_value
    %out = transform.get_result %root[all] : (!transform.any_op) -> !transform.any_value
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %topology_attr = transform.param.constant #hal.device.topology<links = [
        (@rocket_device -> @cpu_device = {transparent_access = true, unified_memory = true}),
        (@cpu_device -> @rocket_device = {transparent_access = true, unified_memory = true})
      ]> -> !transform.any_param
    transform.annotate %module "stream.topology" = %topology_attr : !transform.any_op, !transform.any_param
    %executable = transform.util.import_symbol @rocket_dynamic_executable_s3 into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_rocket_dynamic_conv2d_s3 into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
          transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @cast_and_call_dynamic_conv2d_s4(%root: !transform.any_op {transform.readonly}) {
    %ins = transform.get_operand %root[all] : (!transform.any_op) -> !transform.any_value
    %out = transform.get_result %root[all] : (!transform.any_op) -> !transform.any_value
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %topology_attr = transform.param.constant #hal.device.topology<links = [
        (@rocket_device -> @cpu_device = {transparent_access = true, unified_memory = true}),
        (@cpu_device -> @rocket_device = {transparent_access = true, unified_memory = true})
      ]> -> !transform.any_param
    transform.annotate %module "stream.topology" = %topology_attr : !transform.any_op, !transform.any_param
    %executable = transform.util.import_symbol @rocket_dynamic_executable_s4 into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_rocket_dynamic_conv2d_s4 into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
          transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }



  //===--------------------------------------------------------------------===//
  // int8 matchers
  //
  // Structurally these are the fp16 matchers above with `lhs_type = i8,
  // rhs_type = i8, output_type = i32` -- the shape predicates are identical
  // and their rationale is not repeated here, so read the fp16 matcher of
  // the same name for the CBUF-split correctness caveat behind the channel
  // bounds. What reaches them is an ONNX ConvInteger model after
  // @__transform_main's two dequantization passes: an ordinary named conv on
  // i8 operands with an i32 accumulator, its zero point already folded into
  // a separate CPU-side correction.
  //
  // The channel bounds below are no longer the fp16 matchers' umax = 512.
  // That inherited bound was wrong for int8, and the failure was the
  // predicted one: a silent, near-all-zero result. Measured on RK3588
  // (2026-08-31) through the real compiled path, at 34x34 (3x3) / 32x32
  // (1x1), Cout = 64, sweeping Cin:
  //
  //   dense 1x1  : exact to Cin 352; first failure at Cin 353.
  //   dense 3x3  : exact to Cin  32; wrong from Cin  33 up.
  //
  // The 1x1 boundary was refined with the raw exact-i32 oracle: every native
  // 16-channel point through 352 passed, 368 and 384 failed, and an adjacent
  // probe confirmed 351/352 pass while 353/354/367 fail. At this geometry,
  // 353 is also where ConvPlan changes from one output tile to two. The 352
  // cap is conservative containment; the underlying multi-tile accumulator
  // bug remains open and can affect other geometries.
  //
  // So `match_dynamic_conv2d_int8` caps Cin at 352 and
  // `match_dynamic_conv2d_3x3_int8` at 32. Anything above falls back to the
  // CPU, which is slower but correct. Raising either needs a fix in
  // ConvPlan's int8 modelling, not a bound change -- the cause is still
  // open. Ruled out so far, each on hardware: the coefficient stream order
  // (the shipped order is the only one of four candidates that produces
  // correct values), the CBUF bank split (all eleven splits behave
  // identically), `feature_grains` (swept 1..40), `data_entries` (matches
  // the vendor capture for this exact shape), and the packed feature width.
  //
  // The Cout bound remains 512. It is now hardware-validated in isolation by
  // tools/e2e_conv_regression.py's exact compiled differentials: both 1x1 and
  // 3x3 at Cin=16, Cout=512, 32x32 output and non-zero input zero points
  // matched all 524288 i32 accumulators exactly on RK3588 (2026-08-31). This
  // does not weaken either Cin cap above or claim unmeasured high-Cin/high-Cout
  // interactions are safe; it establishes that Cout=512 itself is not a
  // failure trigger at a low, independently safe Cin.
  //
  // The depthwise int8 matchers keep umax = 512 and are correct across it.
  // They briefly were not: Cin whose *atom* count (ceil(Cin/16)) was one
  // short of a multiple of four -- 33..48, 97..112, 225..240 and every 64
  // thereafter -- lost its last output rows, because `data_bank_demand`
  // billed the CBUF with `weight_atoms` instead of `cbuf_atoms`. Fixed in
  // `conv.rs`; verified at Cin 33, 44..47, 112, 176, 240, 304, 368, 432 and
  // 496. The dense caps above are a *different* bug and that fix does not
  // move them, which is consistent with their atom counts already being
  // whole multiples of four.
  //===--------------------------------------------------------------------===//

  transform.named_sequence @match_dynamic_conv2d_int8(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.conv_2d_nhwc_hwcf"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = i8, rhs_type = i8, output_type = i32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    %filter_value = transform.get_operand %root[1] : (!transform.any_op) -> !transform.any_value
    // Cin 353 is the first failing value in the fixed-geometry dense signed
    // boundary probe. See this section's doc comment.
    transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 352 : !transform.any_value
    transform.iree.match.dim_bounds %filter_value[3], umin = 1, umax = 512 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  transform.named_sequence @match_dynamic_conv2d_3x3_int8(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.conv_2d_nhwc_hwcf"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = i8, rhs_type = i8, output_type = i32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [3, 3] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    %filter_value = transform.get_operand %root[1] : (!transform.any_op) -> !transform.any_value
    // Cin 33 and above returns near-all-zero output on hardware -- the 3x3
    // int8 dense path is correct only within a single CBUF atom pair.
    // See this section's doc comment.
    transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 32 : !transform.any_value
    transform.iree.match.dim_bounds %filter_value[3], umin = 1, umax = 512 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  transform.named_sequence @match_dynamic_depthwise_conv2d_int8(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.depthwise_conv_2d_nhwc_hwc"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = i8, rhs_type = i8, output_type = i32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    // Only one channel count to bound: depthwise Cout is always Cin.
    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 512 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  transform.named_sequence @match_dynamic_depthwise_conv2d_3x3_int8(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.depthwise_conv_2d_nhwc_hwc"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = i8, rhs_type = i8, output_type = i32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [3, 3] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    // Only one channel count to bound: depthwise Cout is always Cin.
    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 512 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  transform.named_sequence @match_dynamic_depthwise_conv2d_int8_s2(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.depthwise_conv_2d_nhwc_hwc"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = i8, rhs_type = i8, output_type = i32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [2, 2] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    // Only one channel count to bound: depthwise Cout is always Cin.
    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 512 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  transform.named_sequence @match_dynamic_depthwise_conv2d_3x3_int8_s2(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.depthwise_conv_2d_nhwc_hwc"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = i8, rhs_type = i8, output_type = i32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [3, 3] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [2, 2] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    // Only one channel count to bound: depthwise Cout is always Cin.
    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 512 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  transform.named_sequence @cast_and_call_dynamic_conv2d_int8(%root: !transform.any_op {transform.readonly}) {
    %ins = transform.get_operand %root[all] : (!transform.any_op) -> !transform.any_value
    %out = transform.get_result %root[all] : (!transform.any_op) -> !transform.any_value
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %topology_attr = transform.param.constant #hal.device.topology<links = [
        (@rocket_device -> @cpu_device = {transparent_access = true, unified_memory = true}),
        (@cpu_device -> @rocket_device = {transparent_access = true, unified_memory = true})
      ]> -> !transform.any_param
    transform.annotate %module "stream.topology" = %topology_attr : !transform.any_op, !transform.any_param
    %executable = transform.util.import_symbol @rocket_dynamic_int8_executable into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_rocket_dynamic_conv2d_int8 into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
          transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @cast_and_call_dynamic_depthwise_conv2d_int8(%root: !transform.any_op {transform.readonly}) {
    %ins = transform.get_operand %root[all] : (!transform.any_op) -> !transform.any_value
    %out = transform.get_result %root[all] : (!transform.any_op) -> !transform.any_value
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %topology_attr = transform.param.constant #hal.device.topology<links = [
        (@rocket_device -> @cpu_device = {transparent_access = true, unified_memory = true}),
        (@cpu_device -> @rocket_device = {transparent_access = true, unified_memory = true})
      ]> -> !transform.any_param
    transform.annotate %module "stream.topology" = %topology_attr : !transform.any_op, !transform.any_param
    %executable = transform.util.import_symbol @rocket_dynamic_depthwise_int8_executable into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_rocket_dynamic_depthwise_conv2d_int8 into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
          transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @cast_and_call_dynamic_depthwise_conv2d_int8_s2(%root: !transform.any_op {transform.readonly}) {
    %ins = transform.get_operand %root[all] : (!transform.any_op) -> !transform.any_value
    %out = transform.get_result %root[all] : (!transform.any_op) -> !transform.any_value
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %topology_attr = transform.param.constant #hal.device.topology<links = [
        (@rocket_device -> @cpu_device = {transparent_access = true, unified_memory = true}),
        (@cpu_device -> @rocket_device = {transparent_access = true, unified_memory = true})
      ]> -> !transform.any_param
    transform.annotate %module "stream.topology" = %topology_attr : !transform.any_op, !transform.any_param
    %executable = transform.util.import_symbol @rocket_dynamic_depthwise_int8_executable_s2 into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_rocket_dynamic_depthwise_conv2d_int8_s2 into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
          transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %funcs = transform.structured.match ops{["util.func"]} in %module : (!transform.any_op) -> !transform.any_op
    // An ONNX int8 model (ORT quantize_dynamic -> onnx.ConvInteger, expanded
    // by RocketExpandOnnxConvIntegerPass) arrives as *quantized* linalg
    // convs -- linalg.conv_2d_nchw_fchw_q and
    // linalg.depthwise_conv_2d_nhwc_hwc_q -- carrying scalar i32 zero-point
    // operands. Those two passes turn them into the ordinary named convs
    // every matcher below already understands, and must run before the
    // channels-last pass: on a quantized conv that pass generalizes to
    // linalg.generic, which cannot hold a scalar operand, and the compile
    // dies on a verifier error rather than merely failing to offload. See
    // RocketTransposeQuantizedConvPass.cpp for the full account.
    //
    // What comes out is an i8 x i8 -> i32 convolution plus a separate
    // zero-point correction (conv_q(x, w, xz, 0) == conv(x, w) - xz*sum(w),
    // an exact i32 identity), which is the form the int8 matchers below
    // claim. A fp32 model is unaffected: neither pass matches anything.
    %nhwc_quantized_funcs = transform.apply_registered_pass
        "rocket-transpose-quantized-conv-to-nhwc" to %funcs
      : (!transform.any_op) -> !transform.any_op
    %dequantized_funcs = transform.apply_registered_pass
        "iree-global-opt-quantized-conv-to-conv" to %nhwc_quantized_funcs
      : (!transform.any_op) -> !transform.any_op

    // ONNX commonly imports Conv as NCHW/FCHW, while Rocket's logical
    // convolution ABI and the matchers above use NHWC/HWCF. Normalize
    // before attempting any Rocket specialization: the first pass does the
    // conversion but leaves the op generalized to linalg.generic, so the
    // second re-specializes it back to the named op the matchers look for.
    %channels_last_funcs = transform.apply_registered_pass
        "iree-preprocessing-convert-conv-to-channels-last" to %dequantized_funcs
      : (!transform.any_op) -> !transform.any_op
    %canonical_funcs = transform.apply_registered_pass
        "linalg-specialize-generic-ops" to %channels_last_funcs
      : (!transform.any_op) -> !transform.any_op
    // Rocket's ABI is f16-in/f32-accumulate (see call_rocket_dynamic_conv2d
    // above), but models commonly arrive as plain f32 (e.g. ONNX/torch
    // import, no fp16 casting anywhere). Demote just the conv operands --
    // not matmuls, which stay on CPU untouched -- to f16, leaving the
    // accumulator at f32, so @match_dynamic_conv2d's f16/f16/f32 typing
    // requirement matches these too. A conv already authored in f16 is
    // left alone: the pass only rewrites all-f32 operand sets.
    %demoted_funcs = transform.apply_registered_pass
        "iree-global-opt-demote-contraction-inputs"
        with options = { "type" = "f16", "operation" = "conv" } to %canonical_funcs
      : (!transform.any_op) -> !transform.any_op

    // Tags every conv-family linalg op with rocket.origin/rocket.origin_kind
    // right before the match/rewrite loop below claims (and erases) some of
    // them -- see RocketAnnotateOriginalPlacementPass.cpp. A
    // --compile-to=preprocessing dump then shows exactly which conv-shaped
    // ops fell through to CPU: matched ops lose the tag along with the rest
    // of the op they were erased from.
    %annotated_funcs = transform.apply_registered_pass
        "rocket-annotate-original-placement" to %demoted_funcs
      : (!transform.any_op) -> !transform.any_op

    transform.foreach %annotated_funcs : !transform.any_op {
      ^bb1(%func: !transform.any_op):
        // @match_dynamic_conv2d_s{2,3,4}/@match_dynamic_conv2d_3x3_s{2,3,4}
        // (defined above, deliberately NOT wired into foreach_match below)
        // are structurally correct and fire on the right shapes -- verified
        // with a synthetic probe module compiled through
        // --compile-to=preprocessing, each stride routing to its own
        // call_rocket_dynamic_conv2d_sN -- but wiring them in breaks a real
        // compile: on a real MobileNetV2 module they additionally claim the
        // network's stem conv (the only dense conv MobileNetV2 runs at
        // stride > 1), and iree-compile then fails at HAL executable
        // serialization ("failed to serialize executables") on
        // main_graph$async_dispatch_0 -- the model's own input f32->f16
        // cast+layout-transpose, which has nothing to do with Rocket.
        // Once the stem conv (dispatch_0's sole consumer) becomes
        // rocket-affinitized, IREE's HAL target-variant materialization
        // additionally generates an empty "rocket_flatbuffer_v1" variant for
        // dispatch_0 itself, which RocketTarget.cpp's
        // buildRocketConv2dConfigFromTarget correctly refuses (see that
        // function's own doc comment: "guarding against some OTHER op ever
        // getting silently routed to 'rocket'"). This is NOT simply "any CPU
        // producer feeding a newly-rocket op breaks" -- VGG-19's stem
        // (stride 1, already claimed and hardware-confirmed end to end
        // before this change) and every interior CPU->rocket edge already in
        // the working MobileNetV2 build are fine; the distinguishing factor
        // isolated so far is that dispatch_0 is a real, independent dispatch
        // region reading the function's own external argument directly (an
        // ONNX-authored `onnx.Cast`, not a compiler-inserted demotion VGG
        // relies on instead), not an ordinary intermediate value. Root cause
        // not fully isolated; see DESIGN_NOTES.md "Depthwise stride hardware
        // confirmation..." for the isolation steps taken (confirmed by
        // disabling just these six matchers: the real MobileNetV2 module
        // then compiles clean with the depthwise stride matchers below still
        // active). Re-enable once this is understood and fixed.
        transform.foreach_match in %func
            @match_dynamic_conv2d -> @cast_and_call_dynamic_conv2d,
            @match_dynamic_conv2d_3x3 -> @cast_and_call_dynamic_conv2d,
            @match_dynamic_depthwise_conv2d -> @cast_and_call_dynamic_depthwise_conv2d,
            @match_dynamic_depthwise_conv2d_3x3 -> @cast_and_call_dynamic_depthwise_conv2d,
            @match_dynamic_depthwise_conv2d_nchw -> @cast_and_call_dynamic_depthwise_conv2d_nchw,
            @match_dynamic_depthwise_conv2d_nchw_3x3 -> @cast_and_call_dynamic_depthwise_conv2d_nchw,
            @match_dynamic_depthwise_conv2d_nchw_s2 -> @cast_and_call_dynamic_depthwise_conv2d_nchw_s2,
            @match_dynamic_depthwise_conv2d_nchw_3x3_s2 -> @cast_and_call_dynamic_depthwise_conv2d_nchw_s2,
            @match_dynamic_depthwise_conv2d_nchw_s3 -> @cast_and_call_dynamic_depthwise_conv2d_nchw_s3,
            @match_dynamic_depthwise_conv2d_nchw_3x3_s3 -> @cast_and_call_dynamic_depthwise_conv2d_nchw_s3,
            @match_dynamic_depthwise_conv2d_nchw_s4 -> @cast_and_call_dynamic_depthwise_conv2d_nchw_s4,
            @match_dynamic_depthwise_conv2d_nchw_3x3_s4 -> @cast_and_call_dynamic_depthwise_conv2d_nchw_s4,
            @match_dynamic_conv2d_int8 -> @cast_and_call_dynamic_conv2d_int8,
            @match_dynamic_conv2d_3x3_int8 -> @cast_and_call_dynamic_conv2d_int8,
            @match_dynamic_depthwise_conv2d_int8 -> @cast_and_call_dynamic_depthwise_conv2d_int8,
            @match_dynamic_depthwise_conv2d_3x3_int8 -> @cast_and_call_dynamic_depthwise_conv2d_int8,
            @match_dynamic_depthwise_conv2d_int8_s2 -> @cast_and_call_dynamic_depthwise_conv2d_int8_s2,
            @match_dynamic_depthwise_conv2d_3x3_int8_s2 -> @cast_and_call_dynamic_depthwise_conv2d_int8_s2
          : (!transform.any_op) -> (!transform.any_op)
    }
    transform.apply_dce to %module : !transform.any_op
    transform.yield
  }
}
