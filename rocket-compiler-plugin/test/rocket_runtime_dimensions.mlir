// RUN: iree-compile %s --compile-mode=hal-executable -o %t.rkt1
//
// This exercises the Rocket serializer directly. The ordered runtime dimension
// list maps the four pipeline constants to input_width, input_height,
// input_channels, and output_channels. Runtime fields use zero in the
// executable template; the driver replaces them with nonzero push constants at
// dispatch. output_width/output_height stay zero here and carry no meaning:
// the runtime always derives the output extent, which is why they are not
// listable in 'runtime_dimensions' (see rocket_runtime_dimensions_invalid.mlir).

#rocket_target = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "conv2d",
  input_width = 0 : i32, input_height = 0 : i32, input_channels = 0 : i32,
  output_width = 0 : i32, output_height = 0 : i32, output_channels = 0 : i32,
  weights_width = 1 : i32, weights_height = 1 : i32, stride = 1 : i32,
  depthwise = false,
  input_zero_point = 0 : i32, output_zero_point = 0 : i32, weights_zero_point = 0 : i32,
  input_scale = 1.0 : f32, weights_scale = 1.0 : f32, output_scale = 1.0 : f32,
  truncate_bits = 0 : i32,
  activation = "none", activation_cmp = 0 : i32,
  precision = "fp16",
  runtime_dimensions = ["input_width", "input_height", "input_channels", "output_channels"]
}>

#pipeline_layout = #hal.pipeline.layout<constants = 4, bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>

module {
  hal.executable public @rocket_dynamic_conv {
    hal.executable.variant public @rocket_dynamic_conv_v1 target(#rocket_target) {
      hal.executable.export public @rocket_dynamic_conv ordinal(0) layout(#pipeline_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @rocket_dynamic_conv() {
          return
        }
      }
    }
  }
}
