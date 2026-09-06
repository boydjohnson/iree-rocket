// RUN: iree-compile %s --compile-mode=hal-executable -o %t.rkt1
//
// The serialization path for `kernel = "elementwise_binary"`, the two-tensor
// EW form. Unlike every other Rocket kernel this has three bindings: the
// primary operand (fetched by DPU_RDMA's main feed), the second tensor
// (fetched by ERDMA), and the output.
//
// Both operands and the result share one geometry -- this op does not
// broadcast, and a producer that wants broadcasting must materialize it --
// which is why there is a single width/height/channels rather than a
// per-operand set.
//
// No `precision` key, for a narrower reason than the unary kernel's.
// `EwAddShape` does carry an int8 branch, but its EW_CVT_SCALE and
// OUT_CVT_SCALE ratio semantics are inferred from register shape rather than
// confirmed against a known value, and MUL has no int8 recipe in any capture
// at all. The wire declines to carry that inference.
//
// 197x1x768 is ViT's own token-by-embedding shape, which is where the
// element-wise sites in a real model actually are: 123 Add, 74 Mul and 25
// Sub across the graph.
#rocket_ew_binary_target = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "elementwise_binary",
  width = 197 : i32, height = 1 : i32, channels = 768 : i32,
  op = "mul"
}>

// Two inputs and one output.
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
