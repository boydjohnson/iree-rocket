// RUN: iree-opt %s --pass-pipeline='builtin.module(rocket-pack-weights)' \
// RUN:   | FileCheck %s

// A Rocket convolution whose filter is a `util.global.load immutable` of an
// initialized global -- what IREE's const-eval leaves at the flow phase --
// has that global packed into the CNA's coefficient layout at compile time:
// a new i8 global of the packed size replaces the filter operand (its
// dynamic dims leave the dispatch with it), and the dispatch is retargeted
// at a clone of its executable whose target config says
// `weights_packed = true`. A filter that is a function argument, or a
// dispatch whose dimensions are not constants, is left to the runtime
// packer. COMPILER_ROADMAP.md 6.3.

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
  // The clone's target attr is the original's plus the flag; it prints
  // first, as an alias, and the clone follows its source.
  // CHECK: #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {{{.*}}weights_packed = true
  // CHECK: hal.executable private @rocket_dynamic_executable {
  // CHECK: hal.executable private @rocket_dynamic_executable_packed {
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

  // 3x3, Cin 24 -> Cout 40 at fp16: neither channel count fills its padding
  // unit. Cin pads to 32 (three 8-channel atoms round up to four) and Cout
  // stays 40, so the stream is 3 * 3 * 32 * 40 * 2 = 23040 bytes from a
  // 17280-byte filter. The original global has no other reader and goes.
  // CHECK-NOT: util.global private @__constant_filter =
  // CHECK: util.global private @__constant_filter_rocket_packed {inlining_policy = #util.inline.never} = dense{{.*}} : tensor<23040xi8>
  util.global private @__constant_filter {inlining_policy = #util.inline.never} = dense<1.500000e+00> : tensor<3x3x24x40xf16>

  // CHECK-LABEL: util.func public @constant_filter
  util.func public @constant_filter(%input: tensor<1x8x8x24xf16>, %bias: tensor<40xf16>) -> tensor<1x8x8x40xf16> {
    %c8 = arith.constant 8 : i32
    %c24 = arith.constant 24 : i32
    %c40 = arith.constant 40 : i32
    %c3 = arith.constant 3 : i32
    %zero = arith.constant 0 : i32
    %c3_idx = arith.constant 3 : index
    %c24_idx = arith.constant 24 : index
    %c40_idx = arith.constant 40 : index
    %filter = util.global.load immutable @__constant_filter : tensor<3x3x24x40xf16>
    %dynamic = flow.tensor.reshape %filter : tensor<3x3x24x40xf16> -> tensor<?x?x?x?xf16>{%c3_idx, %c3_idx, %c24_idx, %c40_idx}
    // CHECK: %[[PACKED:.+]] = util.global.load immutable @__constant_filter_rocket_packed : tensor<23040xi8>
    // CHECK: flow.dispatch @rocket_dynamic_executable_packed::@rocket_dynamic_conv2d_v1::@rocket_dynamic_conv2d(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %[[PACKED]], %{{.+}}) : (i32, i32, i32, i32, i32, i32, i32, tensor<1x8x8x24xf16>, tensor<23040xi8>, tensor<40xf16>) -> tensor<1x8x8x40xf16>
    %out = flow.dispatch @rocket_dynamic_executable::@rocket_dynamic_conv2d_v1::@rocket_dynamic_conv2d(%c8, %c8, %c24, %c40, %c3, %c3, %zero, %input, %dynamic, %bias) : (i32, i32, i32, i32, i32, i32, i32, tensor<1x8x8x24xf16>, tensor<?x?x?x?xf16>{%c3_idx, %c3_idx, %c24_idx, %c40_idx}, tensor<40xf16>) -> tensor<1x8x8x40xf16>
    util.return %out : tensor<1x8x8x40xf16>
  }

  // A filter that arrives as an argument: nothing to pack, the dispatch
  // keeps its executable and its operand.
  // CHECK-LABEL: util.func public @argument_filter
  // CHECK: flow.dispatch @rocket_dynamic_executable::@rocket_dynamic_conv2d_v1::@rocket_dynamic_conv2d(%{{.+}}) : (i32, i32, i32, i32, i32, i32, i32, tensor<1x8x8x24xf16>, tensor<3x3x24x40xf16>, tensor<40xf16>)
  util.func public @argument_filter(%input: tensor<1x8x8x24xf16>, %filter: tensor<3x3x24x40xf16>, %bias: tensor<40xf16>) -> tensor<1x8x8x40xf16> {
    %c8 = arith.constant 8 : i32
    %c24 = arith.constant 24 : i32
    %c40 = arith.constant 40 : i32
    %c3 = arith.constant 3 : i32
    %zero = arith.constant 0 : i32
    %out = flow.dispatch @rocket_dynamic_executable::@rocket_dynamic_conv2d_v1::@rocket_dynamic_conv2d(%c8, %c8, %c24, %c40, %c3, %c3, %zero, %input, %filter, %bias) : (i32, i32, i32, i32, i32, i32, i32, tensor<1x8x8x24xf16>, tensor<3x3x24x40xf16>, tensor<40xf16>) -> tensor<1x8x8x40xf16>
    util.return %out : tensor<1x8x8x40xf16>
  }

  // A constant filter behind a dimension that is not a constant: the packer
  // cannot know the geometry, so the runtime keeps packing.
  util.global private @__constant_other {inlining_policy = #util.inline.never} = dense<2.500000e+00> : tensor<3x3x24x40xf16>
  // CHECK-LABEL: util.func public @dynamic_width
  // CHECK: util.global.load immutable @__constant_other : tensor<3x3x24x40xf16>
  // CHECK: flow.dispatch @rocket_dynamic_executable::@rocket_dynamic_conv2d_v1::@rocket_dynamic_conv2d[
  util.func public @dynamic_width(%input: tensor<1x?x8x24xf16>, %width: i32, %bias: tensor<40xf16>) -> tensor<1x?x8x40xf16> {
    %c8 = arith.constant 8 : i32
    %c24 = arith.constant 24 : i32
    %c40 = arith.constant 40 : i32
    %c3 = arith.constant 3 : i32
    %zero = arith.constant 0 : i32
    %c1 = arith.constant 1 : index
    %filter = util.global.load immutable @__constant_other : tensor<3x3x24x40xf16>
    %w = arith.index_cast %width : i32 to index
    %h = tensor.dim %input, %c1 : tensor<1x?x8x24xf16>
    %out = flow.dispatch @rocket_dynamic_executable::@rocket_dynamic_conv2d_v1::@rocket_dynamic_conv2d[%h](%c8, %width, %c24, %c40, %c3, %c3, %zero, %input, %filter, %bias) : (i32, i32, i32, i32, i32, i32, i32, tensor<1x?x8x24xf16>{%h}, tensor<3x3x24x40xf16>, tensor<40xf16>) -> tensor<1x?x8x40xf16>{%h}
    util.return %out : tensor<1x?x8x40xf16>
  }
}
