// RUN: not iree-compile %s --compile-mode=hal-executable -o %t.rkt1 2>&1 | FileCheck %s
//
// output_width/output_height are always derived by the runtime from the six
// settable dimensions plus stride and padding, so their Conv2DDimension wire
// values are retired (see rocket-schema/schema/rocket_executable_def.fbs) and
// the driver rejects any executable listing them. Reject them here instead of
// emitting a FlatBuffer no driver will load.

// CHECK: unknown runtime Conv2D dimension 'output_width'

#rocket_target = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "conv2d",
  input_width = 0 : i32, input_height = 0 : i32, input_channels = 32 : i32,
  output_width = 0 : i32, output_height = 0 : i32, output_channels = 16 : i32,
  weights_width = 1 : i32, weights_height = 1 : i32, stride = 1 : i32,
  depthwise = false,
  input_zero_point = 0 : i32, output_zero_point = 0 : i32, weights_zero_point = 0 : i32,
  input_scale = 1.0 : f32, weights_scale = 1.0 : f32, output_scale = 1.0 : f32,
  truncate_bits = 0 : i32,
  activation = "none", activation_cmp = 0 : i32,
  precision = "fp16",
  runtime_dimensions = ["input_width", "input_height", "output_width"]
}>

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
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
