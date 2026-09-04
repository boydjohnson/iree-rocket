// RUN: not iree-compile %s --compile-mode=hal-executable -o %t.rkt1 2>&1 | FileCheck %s
//
// A pool's output extents follow from its input geometry, kernel, stride and
// padding, so a dynamically-shaped pool has none to state at compile time --
// the runtime derives them per dispatch. Stating one anyway means the
// producer believes something the runtime will not read, which is worth
// failing on rather than silently ignoring.

// CHECK: must use 0 for output_width/output_height

#rocket_target = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "pooling",
  input_width = 0 : i32, input_height = 0 : i32, channels = 0 : i32,
  output_width = 1 : i32, output_height = 1 : i32,
  kernel_width = 7 : i32, kernel_height = 7 : i32,
  stride_x = 1 : i32, stride_y = 1 : i32,
  pad_left = 0 : i32, pad_top = 0 : i32, pad_right = 0 : i32, pad_bottom = 0 : i32,
  method = "avg",
  precision = "fp16",
  runtime_dimensions = ["input_width", "input_height", "channels"]
}>

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>

module {
  hal.executable public @rocket_pooling {
    hal.executable.variant public @rocket_pooling_v1 target(#rocket_target) {
      hal.executable.export public @rocket_pooling ordinal(0) layout(#pipeline_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @rocket_pooling() {
          return
        }
      }
    }
  }
}
