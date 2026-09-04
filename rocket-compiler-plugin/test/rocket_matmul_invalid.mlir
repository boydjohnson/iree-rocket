// RUN: not iree-compile %s --compile-mode=hal-executable -o %t.rkt1 2>&1 | FileCheck %s
//
// The exact int32 accumulator writer is hardware-validated for Conv2D only --
// this lowering has its own output packing -- so the runtime refuses an
// accumulator matmul. Refuse it here too rather than emitting a FlatBuffer
// no driver will load.

// CHECK: rocket matmul backend does not support int8_accumulator precision

#rocket_target = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "matmul",
  m = 4 : i32, k = 32 : i32, n = 16 : i32,
  input_zero_point = 0 : i32, output_zero_point = 0 : i32, weights_zero_point = 0 : i32,
  input_scale = 1.0 : f32, weights_scale = 1.0 : f32, output_scale = 1.0 : f32,
  truncate_bits = 0 : i32,
  activation = "none", activation_cmp = 0 : i32,
  precision = "int8_accumulator"
}>

#pipeline_layout = #hal.pipeline.layout<constants = 0, bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>

module {
  hal.executable public @rocket_matmul {
    hal.executable.variant public @rocket_matmul_v1 target(#rocket_target) {
      hal.executable.export public @rocket_matmul ordinal(0) layout(#pipeline_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
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
