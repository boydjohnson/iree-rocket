// RUN: iree-compile %s --compile-mode=hal-executable -o %t.rkt1
//
// The serialization path for a static `kernel = "pooling"` executable. There
// is no matcher yet (that is the transform spec's job), so this hand-writes
// the `hal.executable` the way rocket_runtime_dimensions.mlir does for
// convolution -- it tests the target backend, not op matching.
// rocket_pooling_dynamic.mlir is the push-constant-driven counterpart.

// MobileNetV2's global average pool, the shape PoolingDef was added to
// carry: 7x7 over 1792 channels down to one pixel, unpadded.
#rocket_pooling_target = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "pooling",
  input_width = 7 : i32, input_height = 7 : i32, channels = 1792 : i32,
  output_width = 1 : i32, output_height = 1 : i32,
  kernel_width = 7 : i32, kernel_height = 7 : i32,
  stride_x = 1 : i32, stride_y = 1 : i32,
  pad_left = 0 : i32, pad_top = 0 : i32, pad_right = 0 : i32, pad_bottom = 0 : i32,
  method = "avg",
  precision = "fp16"
}>

// Input and output only: a pool has no weights and no bias.
#pooling_layout = #hal.pipeline.layout<constants = 0, bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>

module {
  hal.executable public @rocket_pooling {
    hal.executable.variant public @rocket_pooling_v1 target(#rocket_pooling_target) {
      hal.executable.export public @rocket_pooling ordinal(0) layout(#pooling_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
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
