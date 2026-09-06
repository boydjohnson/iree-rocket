// RUN: not iree-compile %s --compile-mode=hal-executable -o %t.rkt1 2>&1 | FileCheck %s
//
// `lut_bn_alu`'s BN_ALU operand formula is exact by construction at zero and
// confirmed against captures at -128, -2 and 127. There are known-bad
// captures at 42, and `build_lut_regcmd` asserts on anything outside that
// set. The runtime refuses it at the decode boundary; refuse it here too,
// rather than emitting a FlatBuffer that can only fail later.

// CHECK: LUT 'input_zero_point' 42 has no confirmed BN_ALU operand

#rocket_lut_target = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "elementwise_lut",
  width = 4 : i32, height = 4 : i32, channels = 16 : i32,
  fn = "sigmoid",
  input_zero_point = 42 : i32, output_zero_point = 0 : i32,
  input_scale = 0.03125 : f32, output_scale = 0.00390625 : f32
}>

// Input and output only: no weights, no bias.
#elementwise_layout = #hal.pipeline.layout<constants = 0, bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>

module {
  hal.executable public @rocket_elementwise_lut {
    hal.executable.variant public @rocket_elementwise_lut_v1 target(#rocket_lut_target) {
      hal.executable.export public @rocket_elementwise_lut ordinal(0) layout(#elementwise_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @rocket_elementwise_lut() {
          return
        }
      }
    }
  }
}
