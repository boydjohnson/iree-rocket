// RUN: iree-compile %s --compile-mode=hal-executable -o %t.rkt1
//
// The serialization path for a static `kernel = "elementwise_lut"`.
// rocket_elementwise_lut_dynamic.mlir is the push-constant-driven
// counterpart, and rocket_elementwise_unary.mlir is the ALU sibling.
//
// No `precision` key, for the mirror image of the unary kernel's reason: the
// LUT path is int8 by construction. The curve is evaluated on the
// dequantized real value, so `input_scale`/`input_zero_point` are not
// optional metadata -- they are how an input reaches the table's fixed
// domain at all.
//
// Zero points are the **decoded** values, not the 0x80-biased raw register
// form `LutShape` takes; the runtime applies that bias. Only -128, -2, 0 and
// 127 are accepted, which is the set `lut_bn_alu`'s operand formula is
// confirmed against.
//
// 7x5x48 is an odd, non-square, unaligned area (35 pixels) across three int8
// surfaces -- the shape `lut_multi_surface_hw.rs` carries for the same
// reason.
#rocket_lut_target = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "elementwise_lut",
  width = 7 : i32, height = 5 : i32, channels = 48 : i32,
  fn = "tanh",
  input_zero_point = 0 : i32, output_zero_point = 0 : i32,
  input_scale = 0.03125 : f32, output_scale = 0.0078125 : f32
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
