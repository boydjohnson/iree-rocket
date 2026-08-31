// RUN: iree-compile %s --compile-mode=hal-executable -o %t.rkt1
//
// Serializer smoke test for ConvInteger's signed-int8-to-i32 output mode.

#rocket_target = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "conv2d",
  input_width = 4 : i32, input_height = 4 : i32, input_channels = 1 : i32,
  output_width = 4 : i32, output_height = 4 : i32, output_channels = 8 : i32,
  weights_width = 1 : i32, weights_height = 1 : i32, stride = 1 : i32,
  depthwise = false,
  input_zero_point = 0 : i32, output_zero_point = 0 : i32, weights_zero_point = 0 : i32,
  input_scale = 1.0 : f32, weights_scale = 1.0 : f32, output_scale = 1.0 : f32,
  truncate_bits = 0 : i32,
  activation = "none", activation_cmp = 0 : i32,
  precision = "int8_accumulator"
}>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>

module {
  hal.executable public @rocket_int8_accumulator {
    hal.executable.variant public @rocket_int8_accumulator_v1 target(#rocket_target) {
      hal.executable.export public @conv_integer ordinal(0) layout(#pipeline_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @conv_integer() {
          return
        }
      }
    }
  }
}
