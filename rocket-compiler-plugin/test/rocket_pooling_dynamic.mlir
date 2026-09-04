// RUN: iree-compile %s --compile-mode=hal-executable -o %t.rkt1
//
// The dynamic counterpart of rocket_pooling.mlir: one executable serving many
// shapes, with the geometry arriving as push constants. Every field the
// vector lists is zero in the template, and so are the output extents, which
// the runtime derives per dispatch rather than reading -- a dynamic pool has
// no compile-time output extent to state.
//
// Padding is deliberately not settable this way: it is 0..=7 on this
// hardware, its meaning depends on the pooling method, and no measured model
// varies it per dispatch.

#rocket_dynamic_pooling_target = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "pooling",
  input_width = 0 : i32, input_height = 0 : i32, channels = 0 : i32,
  output_width = 0 : i32, output_height = 0 : i32,
  kernel_width = 0 : i32, kernel_height = 0 : i32,
  stride_x = 2 : i32, stride_y = 2 : i32,
  pad_left = 0 : i32, pad_top = 0 : i32, pad_right = 0 : i32, pad_bottom = 0 : i32,
  method = "max",
  precision = "int8",
  runtime_dimensions = ["input_width", "input_height", "channels", "kernel_width", "kernel_height"]
}>

// Input and output only: a pool has no weights and no bias.
#dynamic_pooling_layout = #hal.pipeline.layout<constants = 5, bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>

module {
  hal.executable public @rocket_dynamic_pooling {
    hal.executable.variant public @rocket_dynamic_pooling_v1 target(#rocket_dynamic_pooling_target) {
      hal.executable.export public @rocket_dynamic_pooling ordinal(0) layout(#dynamic_pooling_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @rocket_dynamic_pooling() {
          return
        }
      }
    }
  }
}
