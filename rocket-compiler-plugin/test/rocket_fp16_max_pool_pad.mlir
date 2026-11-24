// RUN: iree-compile %s \
// RUN:   --iree-preprocessing-transform-spec-filename=%S/../target/Rocket/rocket_conv2d_transform_spec.mlir \
// RUN:   --iree-hal-target-device=rocket_device=rocket \
// RUN:   --iree-hal-target-device=cpu_device=local \
// RUN:   --iree-hal-local-target-device-backends=llvm-cpu \
// RUN:   --iree-llvmcpu-target-cpu=generic \
// RUN:   --iree-hal-default-device=cpu_device \
// RUN:   --iree-hal-indirect-command-buffers=false \
// RUN:   --compile-to=preprocessing \
// RUN:   --mlir-print-op-generic=false \
// RUN:   -o - | FileCheck %s

// A ResNet stem's max pool as it arrives on the fp16 path: the -inf pad in
// the channels-last rank-5 form, collapsed, transposed back to NCHW for a
// `pooling_nchw_max`. rocket-fold-conv-pad turns it into the NHWC pool over
// the NHWC tensor and marks the pad, and @match_pooling_nhwc_max_s2_pad1
// hands the padding to the PPU: the dispatch reads the unpadded 112x112.
// CHECK-LABEL: util.func public @stem_pool
// CHECK-NOT: tensor.pad
// CHECK: flow.dispatch @rocket_pooling_max_pad1_executable_s2::{{.*}}(%c112_i32, %c112_i32, %c64_i32, %c3_i32, %c3_i32
// CHECK-NOT: tensor.pad
// CHECK: util.return
util.func public @stem_pool(%x: tensor<1x112x112x64xf32>) -> tensor<1x64x56x56xf32> {
  %neg_inf = arith.constant 0xFF800000 : f32
  %e = tensor.expand_shape %x [[0], [1, 2], [3], [4]] output_shape [1, 1, 112, 112, 64] : tensor<1x112x112x64xf32> into tensor<1x1x112x112x64xf32>
  %p = tensor.pad %e low[0, 0, 1, 1, 0] high[0, 0, 1, 1, 0] {
  ^bb0(%a: index, %b: index, %c: index, %d: index, %f: index):
    tensor.yield %neg_inf : f32
  } : tensor<1x1x112x112x64xf32> to tensor<1x1x114x114x64xf32>
  %c = tensor.collapse_shape %p [[0], [1, 2], [3], [4]] : tensor<1x1x114x114x64xf32> into tensor<1x114x114x64xf32>
  %t_empty = tensor.empty() : tensor<1x64x114x114xf32>
  %t = linalg.transpose ins(%c : tensor<1x114x114x64xf32>) outs(%t_empty : tensor<1x64x114x114xf32>) permutation = [0, 3, 1, 2]
  %empty = tensor.empty() : tensor<1x64x56x56xf32>
  %init = linalg.fill ins(%neg_inf : f32) outs(%empty : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
  %w = tensor.empty() : tensor<3x3xf32>
  %r = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%t, %w : tensor<1x64x114x114xf32>, tensor<3x3xf32>) outs(%init : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
  util.return %r : tensor<1x64x56x56xf32>
}

// A zero pad is not a max pool's identity: left alone, so the CPU pads and
// the plain NCHW matcher claims the pool.
// CHECK-LABEL: util.func public @zero_pad_declines
// CHECK: tensor.pad
// CHECK-NOT: rocket_pooling_max_pad1_executable_s2
// CHECK: util.return
util.func public @zero_pad_declines(%x: tensor<1x112x112x64xf32>) -> tensor<1x64x56x56xf32> {
  %zero = arith.constant 0.0 : f32
  %neg_inf = arith.constant 0xFF800000 : f32
  %e = tensor.expand_shape %x [[0], [1, 2], [3], [4]] output_shape [1, 1, 112, 112, 64] : tensor<1x112x112x64xf32> into tensor<1x1x112x112x64xf32>
  %p = tensor.pad %e low[0, 0, 1, 1, 0] high[0, 0, 1, 1, 0] {
  ^bb0(%a: index, %b: index, %c: index, %d: index, %f: index):
    tensor.yield %zero : f32
  } : tensor<1x1x112x112x64xf32> to tensor<1x1x114x114x64xf32>
  %c = tensor.collapse_shape %p [[0], [1, 2], [3], [4]] : tensor<1x1x114x114x64xf32> into tensor<1x114x114x64xf32>
  %t_empty = tensor.empty() : tensor<1x64x114x114xf32>
  %t = linalg.transpose ins(%c : tensor<1x114x114x64xf32>) outs(%t_empty : tensor<1x64x114x114xf32>) permutation = [0, 3, 1, 2]
  %empty = tensor.empty() : tensor<1x64x56x56xf32>
  %init = linalg.fill ins(%neg_inf : f32) outs(%empty : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
  %w = tensor.empty() : tensor<3x3xf32>
  %r = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%t, %w : tensor<1x64x114x114xf32>, tensor<3x3xf32>) outs(%init : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
  util.return %r : tensor<1x64x56x56xf32>
}
