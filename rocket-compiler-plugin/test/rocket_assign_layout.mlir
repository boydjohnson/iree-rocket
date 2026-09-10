// RUN: iree-opt %s --pass-pipeline='builtin.module(rocket-assign-layout)' \
// RUN:   | FileCheck %s

// The trailing layout push constant every Rocket target declares with
// `runtime_layout` is rewritten from the shim's literal 0 to
// `packed_inputs | (packed_readers << 16)`: bit i of the low half set when
// input binding i reads its producer's cube in place, the high half the
// number of Rocket dispatches that read this result that way. Every edge
// and every reader count also lands in `rocket.layout_decisions` on the
// function, for the audit. COMPILER_ROADMAP.md 6.2.

#rocket_target = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "conv2d",
  input_width = 0 : i32, input_height = 0 : i32, input_channels = 0 : i32,
  output_width = 0 : i32, output_height = 0 : i32, output_channels = 0 : i32,
  weights_width = 0 : i32, weights_height = 0 : i32, stride = 1 : i32,
  depthwise = false,
  runtime_layout = true,
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

#pooling_target = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "pooling",
  runtime_layout = true,
  input_width = 0 : i32, input_height = 0 : i32, channels = 0 : i32,
  output_width = 0 : i32, output_height = 0 : i32,
  kernel_width = 0 : i32, kernel_height = 0 : i32,
  stride_x = 2 : i32, stride_y = 2 : i32,
  pad_left = 0 : i32, pad_top = 0 : i32, pad_right = 0 : i32, pad_bottom = 0 : i32,
  method = "max",
  precision = "fp16",
  runtime_dimensions = ["input_width", "input_height", "channels",
                        "kernel_width", "kernel_height"]
}>

#conv_layout = #hal.pipeline.layout<constants = 7, bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>
#pool_layout = #hal.pipeline.layout<constants = 6, bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>

module {
  hal.executable private @conv_exec {
    hal.executable.variant public @v1 target(#rocket_target) {
      hal.executable.export public @conv ordinal(0) layout(#conv_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @conv() {
          return
        }
      }
    }
  }
  hal.executable private @pool_exec {
    hal.executable.variant public @v1 target(#pooling_target) {
      hal.executable.export public @pool ordinal(0) layout(#pool_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @pool() {
          return
        }
      }
    }
  }

  // conv A (16x16, 64 -> 64 fp16) -> reshape -> conv B (64 -> 128): B's
  // input reads A's cube (bit 0), and A's only reader takes it (count 1).
  // B's result is returned, so B's count is 0; B's input from A is packed.
  // CHECK-LABEL: util.func public @chain
  // CHECK-SAME: rocket.layout_decisions = [
  // CHECK-SAME: {binding = 0 : i64, edge = "input", kind = "conv2d", loc = {{.+}}, producer = "argument", reason = "the value is a function argument", site = "conv_exec::conv", verdict = "dense"}
  // CHECK-SAME: {binding = 0 : i64, edge = "input", kind = "conv2d", loc = {{.+}}, producer = "conv_exec::conv", reason = "geometries identical", site = "conv_exec::conv", verdict = "packed"}
  util.func public @chain(%input: tensor<1x16x16x64xf16>, %f1: tensor<1x1x64x64xf16>, %f2: tensor<1x1x64x128xf16>, %bias: tensor<64xf16>, %bias2: tensor<128xf16>) -> tensor<1x16x16x128xf16> {
    %c16 = arith.constant 16 : i32
    %c64 = arith.constant 64 : i32
    %c128 = arith.constant 128 : i32
    %c1 = arith.constant 1 : i32
    %zero = arith.constant 0 : i32
    // A: input is an argument (bit 0 clear); one packed reader -> 1 << 16.
    // CHECK: %[[A_WORD:.+]] = arith.constant 65536 : i32
    // CHECK: flow.dispatch @conv_exec::@v1::@conv(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %[[A_WORD]],
    %a = flow.dispatch @conv_exec::@v1::@conv(%c16, %c16, %c64, %c64, %c1, %c1, %zero, %input, %f1, %bias) : (i32, i32, i32, i32, i32, i32, i32, tensor<1x16x16x64xf16>, tensor<1x1x64x64xf16>, tensor<64xf16>) -> tensor<1x16x16x64xf16>
    %a_r = flow.tensor.reshape %a : tensor<1x16x16x64xf16> -> tensor<1x16x16x64xf16>
    // B: input packed (bit 0); returned, so no packed readers.
    // CHECK: %[[B_WORD:.+]] = arith.constant 1 : i32
    // CHECK: flow.dispatch @conv_exec::@v1::@conv(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %[[B_WORD]],
    %b = flow.dispatch @conv_exec::@v1::@conv(%c16, %c16, %c64, %c128, %c1, %c1, %zero, %a_r, %f2, %bias2) : (i32, i32, i32, i32, i32, i32, i32, tensor<1x16x16x64xf16>, tensor<1x1x64x128xf16>, tensor<128xf16>) -> tensor<1x16x16x128xf16>
    util.return %b : tensor<1x16x16x128xf16>
  }

  // A 7x7 max pool feeding a conv: the PPU strides its surfaces by 52, the
  // conv packs at 49, so the edge is dense with that reason, and the pool's
  // reader count is 0. The pool's own input from a conv on 14x14 (196
  // pixels, a multiple of four) is packed.
  // CHECK-LABEL: util.func public @stride_mismatch
  // CHECK-SAME: producer = "pool_exec::pool", reason = "producer surfaces 52 pixels apart, consumer packs at 49", site = "conv_exec::conv", verdict = "dense"
  util.func public @stride_mismatch(%input: tensor<1x14x14x64xf16>, %f1: tensor<1x1x64x64xf16>, %bias: tensor<64xf16>) -> tensor<1x7x7x64xf16> {
    %c14 = arith.constant 14 : i32
    %c7 = arith.constant 7 : i32
    %c64 = arith.constant 64 : i32
    %c2 = arith.constant 2 : i32
    %c1 = arith.constant 1 : i32
    %zero = arith.constant 0 : i32
    // CHECK: %[[C_WORD:.+]] = arith.constant 65536 : i32
    // CHECK: flow.dispatch @conv_exec::@v1::@conv(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %[[C_WORD]],
    %c = flow.dispatch @conv_exec::@v1::@conv(%c14, %c14, %c64, %c64, %c1, %c1, %zero, %input, %f1, %bias) : (i32, i32, i32, i32, i32, i32, i32, tensor<1x14x14x64xf16>, tensor<1x1x64x64xf16>, tensor<64xf16>) -> tensor<1x14x14x64xf16>
    // Pool: input packed (bit 0), no packed readers (the conv below cannot
    // take a 52-stride cube).
    // CHECK: %[[P_WORD:.+]] = arith.constant 1 : i32
    // CHECK: flow.dispatch @pool_exec::@v1::@pool(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %[[P_WORD]],
    %p = flow.dispatch @pool_exec::@v1::@pool(%c14, %c14, %c64, %c2, %c2, %zero, %c) : (i32, i32, i32, i32, i32, i32, tensor<1x14x14x64xf16>) -> tensor<1x7x7x64xf16>
    // CHECK: %[[D_WORD:.+]] = arith.constant 0 : i32
    // CHECK: flow.dispatch @conv_exec::@v1::@conv(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %[[D_WORD]],
    %d = flow.dispatch @conv_exec::@v1::@conv(%c7, %c7, %c64, %c64, %c1, %c1, %zero, %p, %f1, %bias) : (i32, i32, i32, i32, i32, i32, i32, tensor<1x7x7x64xf16>, tensor<1x1x64x64xf16>, tensor<64xf16>) -> tensor<1x7x7x64xf16>
    util.return %d : tensor<1x7x7x64xf16>
  }

  // Two Rocket readers of one result: both packed, count 2. A third reader
  // that is not a Rocket dispatch would make it 0.
  // CHECK-LABEL: util.func public @fan_out
  util.func public @fan_out(%input: tensor<1x8x8x64xf16>, %f1: tensor<1x1x64x64xf16>, %bias: tensor<64xf16>) -> (tensor<1x8x8x64xf16>, tensor<1x8x8x64xf16>) {
    %c8 = arith.constant 8 : i32
    %c64 = arith.constant 64 : i32
    %c1 = arith.constant 1 : i32
    %zero = arith.constant 0 : i32
    // CHECK: %[[E_WORD:.+]] = arith.constant 131072 : i32
    // CHECK: flow.dispatch @conv_exec::@v1::@conv(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %[[E_WORD]],
    %e = flow.dispatch @conv_exec::@v1::@conv(%c8, %c8, %c64, %c64, %c1, %c1, %zero, %input, %f1, %bias) : (i32, i32, i32, i32, i32, i32, i32, tensor<1x8x8x64xf16>, tensor<1x1x64x64xf16>, tensor<64xf16>) -> tensor<1x8x8x64xf16>
    %x = flow.dispatch @conv_exec::@v1::@conv(%c8, %c8, %c64, %c64, %c1, %c1, %zero, %e, %f1, %bias) : (i32, i32, i32, i32, i32, i32, i32, tensor<1x8x8x64xf16>, tensor<1x1x64x64xf16>, tensor<64xf16>) -> tensor<1x8x8x64xf16>
    %y = flow.dispatch @conv_exec::@v1::@conv(%c8, %c8, %c64, %c64, %c1, %c1, %zero, %e, %f1, %bias) : (i32, i32, i32, i32, i32, i32, i32, tensor<1x8x8x64xf16>, tensor<1x1x64x64xf16>, tensor<64xf16>) -> tensor<1x8x8x64xf16>
    util.return %x, %y : tensor<1x8x8x64xf16>, tensor<1x8x8x64xf16>
  }
}
