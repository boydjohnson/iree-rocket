// RUN: not iree-compile %s --compile-mode=hal-executable -o %t.rkt1 2>&1 | FileCheck %s
//
// `ew_alu_algo = 3` (Div) is the one TRM-documented binary opcode with no
// hardware evidence anywhere in this project. It is absent from
// iree-rocket-hal's own EwBinaryOp and from the wire enum, so naming it here
// must be a compile error rather than something the serializer invents an
// encoding for.

// CHECK: unrecognized element-wise binary 'op' config value 'div'

#rocket_ew_binary_target = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "elementwise_binary",
  width = 4 : i32, height = 4 : i32, channels = 16 : i32,
  op = "div"
}>

#elementwise_binary_layout = #hal.pipeline.layout<constants = 0, bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>

module {
  hal.executable public @rocket_elementwise_binary {
    hal.executable.variant public @rocket_elementwise_binary_v1 target(#rocket_ew_binary_target) {
      hal.executable.export public @rocket_elementwise_binary ordinal(0) layout(#elementwise_binary_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @rocket_elementwise_binary() {
          return
        }
      }
    }
  }
}
