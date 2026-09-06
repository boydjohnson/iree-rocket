// RUN: iree-compile %s --compile-mode=hal-executable -o %t.rkt1
//
// The serialization path for `kernel = "elementwise_unary"`, one of the two
// element-wise kernels ROADMAP Phase 1 added. Like rocket_pooling.mlir this
// hand-writes the `hal.executable` rather than going through a matcher -- it
// tests the target backend, not op matching.
//
// One input, one output. There is no reduction, so the two cubes have
// identical geometry and there is no output extent to state or to derive.
//
// No `precision` key: the unary EW task shape is fp16 only, which is why the
// wire table has no precision field either. `iree-rocket-hal`'s
// `EwUnaryShape` ships no int8 branch because no capture confirms an int8
// zero-point/scale recipe for it.
//
// 14x14x64 is the top of `ew_unary_multi_surface_hw.rs`'s geometry ladder:
// eight fp16 surfaces, which is what makes it worth carrying here rather
// than a single-surface shape that could not exercise a surface stride.
#rocket_ew_target = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "elementwise_unary",
  width = 14 : i32, height = 14 : i32, channels = 64 : i32,
  op = "floor"
}>

// Input and output only: no weights, no bias.
#elementwise_layout = #hal.pipeline.layout<constants = 0, bindings = [
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
