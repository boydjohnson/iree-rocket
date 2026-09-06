// RUN: not iree-compile %s --compile-mode=hal-executable -o %t.rkt1 2>&1 | FileCheck %s
//
// A dimension listed in `runtime_dimensions` is supplied per dispatch, so
// the executable template must hold zero for it. A nonzero template value
// means the producer promised to push a value it also baked in, and the two
// could disagree.

// CHECK: runtime element-wise dimension 'width' must use 0 as its executable template value

#rocket_ew_target = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "elementwise_unary",
  width = 4 : i32, height = 4 : i32, channels = 16 : i32,
  op = "abs",
  runtime_dimensions = ["width"]
}>

// One push constant, for the one runtime dimension.
#elementwise_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>


module {
  hal.executable public @rocket_elementwise_unary {
    hal.executable.variant public @rocket_elementwise_unary_v1 target(#rocket_ew_target) {
      hal.executable.export public @rocket_elementwise_unary ordinal(0) layout(#elementwise_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @rocket_elementwise_unary() {
          return
        }
      }
    }
  }
}
