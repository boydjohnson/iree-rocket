// RUN: iree-compile %s --compile-mode=hal-executable -o %t.rkt1
//
// The push-constant-driven counterpart to rocket_elementwise_lut.mlir: every
// dimension is supplied per dispatch, so all three are zero in the template
// and the pipeline layout declares three constants.
//
// The contract is the one every other kernel here holds -- a listed
// dimension must be zero in the template and an unlisted one must be
// nonzero. `checkElementwiseDimensions` is the compile-time half and
// rocket-hal-driver's `validate_elementwise_template` the runtime half.
#rocket_lut_dynamic_target = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "elementwise_lut",
  width = 0 : i32, height = 0 : i32, channels = 0 : i32,
  fn = "sqrt",
  input_zero_point = 0 : i32, output_zero_point = 0 : i32,
  input_scale = 0.0078125 : f32, output_scale = 0.0078125 : f32,
  runtime_dimensions = ["width", "height", "channels"]
}>

// Three push constants, one per runtime dimension, in that order.
#dynamic_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>


module {
  hal.executable public @rocket_elementwise_lut_dynamic {
    hal.executable.variant public @rocket_elementwise_lut_dynamic_v1 target(#rocket_lut_dynamic_target) {
      hal.executable.export public @rocket_elementwise_lut_dynamic ordinal(0) layout(#dynamic_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @rocket_elementwise_lut_dynamic() {
          return
        }
      }
    }
  }
}
