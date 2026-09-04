// RUN: not iree-compile %s --compile-mode=hal-executable -o %t.rkt1 2>&1 | FileCheck %s
//
// MLIR's linalg dialect has no average pool: an ONNX AveragePool arrives as
// linalg.pooling_*_sum plus a separate divide. It is tempting to carry that
// "sum" through to the hardware, and the hardware cannot do it -- the PPU has
// no sum mode, only a multiply by fp16(65536/k), which cannot encode a
// divisor of one. Recognizing sum-plus-divide as an average is the matcher's
// job, and the wire format has no SUM to fall back on.

// CHECK: unrecognized pooling 'method' config value 'sum'

#rocket_target = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "pooling",
  input_width = 7 : i32, input_height = 7 : i32, channels = 64 : i32,
  output_width = 1 : i32, output_height = 1 : i32,
  kernel_width = 7 : i32, kernel_height = 7 : i32,
  stride_x = 1 : i32, stride_y = 1 : i32,
  pad_left = 0 : i32, pad_top = 0 : i32, pad_right = 0 : i32, pad_bottom = 0 : i32,
  method = "sum",
  precision = "fp16"
}>

#pipeline_layout = #hal.pipeline.layout<constants = 0, bindings = [
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
