// RUN: iree-compile %s --compile-mode=hal-executable -o %t.rkt1
//
// The serialization path for `kernel = "matmul"`, which supersedes
// `fully_connected`: "fully connected" is not an operation in MLIR's linalg
// dialect, so no matcher can ever produce one, while `linalg.matmul` is what
// both MobileNetV2 models actually hand this backend.
//
// MobileNetV2's classifier: [1,1792] x [1792,1001]. K = 1792 is exactly the
// HAL's `MAX_INPUT_CHANNELS`, which was raised to it after the height-one
// geometry a matmul lowers to was measured on hardware.

#rocket_matmul_target = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "matmul",
  m = 1 : i32, k = 1792 : i32, n = 1001 : i32,
  input_zero_point = 0 : i32, output_zero_point = 0 : i32, weights_zero_point = 0 : i32,
  input_scale = 1.0 : f32, weights_scale = 1.0 : f32, output_scale = 1.0 : f32,
  truncate_bits = 0 : i32,
  activation = "none", activation_cmp = 0 : i32,
  precision = "fp16"
}>

// Input, weights, bias, output -- the same four the convolution lowering
// binds, because that is what this lowers to.
#matmul_layout = #hal.pipeline.layout<constants = 0, bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>

module {
  hal.executable public @rocket_matmul {
    hal.executable.variant public @rocket_matmul_v1 target(#rocket_matmul_target) {
      hal.executable.export public @rocket_matmul ordinal(0) layout(#matmul_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @rocket_matmul() {
          return
        }
      }
    }
  }
}
