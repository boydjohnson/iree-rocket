// RUN: iree-opt %s --pass-pipeline='builtin.module(rocket-mark-dense-readers)' \
// RUN:   | FileCheck %s

// The trailing push constant every Rocket convolution target declares with
// `runtime_dense_readers` is rewritten from the shim's literal 0 to the
// number of Rocket dispatches that read the result -- through a reshape --
// or left at 0 when anything else reads it.

#rocket_target = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "conv2d",
  input_width = 0 : i32, input_height = 0 : i32, input_channels = 0 : i32,
  output_width = 0 : i32, output_height = 0 : i32, output_channels = 0 : i32,
  weights_width = 0 : i32, weights_height = 0 : i32, stride = 1 : i32,
  depthwise = false,
  runtime_dense_readers = true,
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

#layout = #hal.pipeline.layout<constants = 7, bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>

module {
  hal.executable private @rocket_dynamic_executable {
    hal.executable.variant public @rocket_dynamic_conv2d_v1 target(#rocket_target) {
      hal.executable.export public @rocket_dynamic_conv2d ordinal(0) layout(#layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
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

  // CHECK-LABEL: util.func public @chain
  util.func public @chain(%input: tensor<1x8x8x16xf16>, %filter: tensor<1x1x16x32xf16>, %bias: tensor<32xf16>) -> tensor<1x8x8x32xf16> {
    %c8 = arith.constant 8 : i32
    %c16 = arith.constant 16 : i32
    %c32 = arith.constant 32 : i32
    %c1 = arith.constant 1 : i32
    %zero = arith.constant 0 : i32
    // The first dispatch's only readers are the reshape below and, through
    // it, the second Rocket dispatch: one reader. The pass materializes each
    // count immediately before its dispatch, after the shim's literal zero.
    // CHECK: arith.constant 0 : i32
    // CHECK-NEXT: %[[ONE:.+]] = arith.constant 1 : i32
    // CHECK-NEXT: flow.dispatch @rocket_dynamic_executable::@rocket_dynamic_conv2d_v1::@rocket_dynamic_conv2d(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %[[ONE]],
    %a = flow.dispatch @rocket_dynamic_executable::@rocket_dynamic_conv2d_v1::@rocket_dynamic_conv2d(%c8, %c8, %c16, %c32, %c1, %c1, %zero, %input, %filter, %bias) : (i32, i32, i32, i32, i32, i32, i32, tensor<1x8x8x16xf16>, tensor<1x1x16x32xf16>, tensor<32xf16>) -> tensor<1x8x8x32xf16>
    %a_r = flow.tensor.reshape %a : tensor<1x8x8x32xf16> -> tensor<1x8x8x32xf16>
    // The second dispatch's result is returned: not a Rocket reader, so 0.
    // CHECK: flow.tensor.reshape
    // CHECK-NEXT: %[[ZERO:.+]] = arith.constant 0 : i32
    // CHECK-NEXT: flow.dispatch @rocket_dynamic_executable::@rocket_dynamic_conv2d_v1::@rocket_dynamic_conv2d(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %[[ZERO]],
    %b = flow.dispatch @rocket_dynamic_executable::@rocket_dynamic_conv2d_v1::@rocket_dynamic_conv2d(%c8, %c8, %c32, %c32, %c1, %c1, %zero, %a_r, %filter, %bias) : (i32, i32, i32, i32, i32, i32, i32, tensor<1x8x8x32xf16>, tensor<1x1x16x32xf16>, tensor<32xf16>) -> tensor<1x8x8x32xf16>
    util.return %b : tensor<1x8x8x32xf16>
  }

  // CHECK-LABEL: util.func public @fan_out
  util.func public @fan_out(%input: tensor<1x8x8x16xf16>, %filter: tensor<1x1x16x32xf16>, %bias: tensor<32xf16>) -> (tensor<1x8x8x32xf16>, tensor<1x8x8x32xf16>) {
    %c8 = arith.constant 8 : i32
    %c16 = arith.constant 16 : i32
    %c32 = arith.constant 32 : i32
    %c1 = arith.constant 1 : i32
    %zero = arith.constant 0 : i32
    // Two Rocket readers: the count is 2, and the driver will only skip the
    // dense write if both chain on one command buffer.
    // CHECK: arith.constant 0 : i32
    // CHECK-NEXT: %[[TWO:.+]] = arith.constant 2 : i32
    // CHECK-NEXT: flow.dispatch @rocket_dynamic_executable::@rocket_dynamic_conv2d_v1::@rocket_dynamic_conv2d(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %[[TWO]],
    %a = flow.dispatch @rocket_dynamic_executable::@rocket_dynamic_conv2d_v1::@rocket_dynamic_conv2d(%c8, %c8, %c16, %c32, %c1, %c1, %zero, %input, %filter, %bias) : (i32, i32, i32, i32, i32, i32, i32, tensor<1x8x8x16xf16>, tensor<1x1x16x32xf16>, tensor<32xf16>) -> tensor<1x8x8x32xf16>
    %b = flow.dispatch @rocket_dynamic_executable::@rocket_dynamic_conv2d_v1::@rocket_dynamic_conv2d(%c8, %c8, %c32, %c32, %c1, %c1, %zero, %a, %filter, %bias) : (i32, i32, i32, i32, i32, i32, i32, tensor<1x8x8x32xf16>, tensor<1x1x16x32xf16>, tensor<32xf16>) -> tensor<1x8x8x32xf16>
    %c = flow.dispatch @rocket_dynamic_executable::@rocket_dynamic_conv2d_v1::@rocket_dynamic_conv2d(%c8, %c8, %c32, %c32, %c1, %c1, %zero, %a, %filter, %bias) : (i32, i32, i32, i32, i32, i32, i32, tensor<1x8x8x32xf16>, tensor<1x1x16x32xf16>, tensor<32xf16>) -> tensor<1x8x8x32xf16>
    util.return %b, %c : tensor<1x8x8x32xf16>, tensor<1x8x8x32xf16>
  }

  // CHECK-LABEL: util.func public @mixed_reader
  util.func public @mixed_reader(%input: tensor<1x8x8x16xf16>, %filter: tensor<1x1x16x32xf16>, %bias: tensor<32xf16>) -> (tensor<1x8x8x32xf16>, tensor<1x8x8x32xf16>) {
    %c8 = arith.constant 8 : i32
    %c16 = arith.constant 16 : i32
    %c32 = arith.constant 32 : i32
    %c1 = arith.constant 1 : i32
    %zero = arith.constant 0 : i32
    // One Rocket reader and one escape: any non-Rocket reader zeroes the
    // count, however many Rocket readers there are besides.
    // CHECK: arith.constant 0 : i32
    // CHECK-NEXT: %[[Z:.+]] = arith.constant 0 : i32
    // CHECK-NEXT: flow.dispatch @rocket_dynamic_executable::@rocket_dynamic_conv2d_v1::@rocket_dynamic_conv2d(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %[[Z]],
    %a = flow.dispatch @rocket_dynamic_executable::@rocket_dynamic_conv2d_v1::@rocket_dynamic_conv2d(%c8, %c8, %c16, %c32, %c1, %c1, %zero, %input, %filter, %bias) : (i32, i32, i32, i32, i32, i32, i32, tensor<1x8x8x16xf16>, tensor<1x1x16x32xf16>, tensor<32xf16>) -> tensor<1x8x8x32xf16>
    %b = flow.dispatch @rocket_dynamic_executable::@rocket_dynamic_conv2d_v1::@rocket_dynamic_conv2d(%c8, %c8, %c32, %c32, %c1, %c1, %zero, %a, %filter, %bias) : (i32, i32, i32, i32, i32, i32, i32, tensor<1x8x8x32xf16>, tensor<1x1x16x32xf16>, tensor<32xf16>) -> tensor<1x8x8x32xf16>
    util.return %a, %b : tensor<1x8x8x32xf16>, tensor<1x8x8x32xf16>
  }
}
