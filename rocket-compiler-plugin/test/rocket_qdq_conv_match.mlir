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

// A QDQ convolution as the IREE 3.14 ONNX importer emits it -- NCHW, f32,
// between `dequantize_affine` and `quantize_affine` -- runs through the whole
// spec and lands on the requantized int8 executables, not the fp16 ones.
// `rocket-fuse-qdq-conv` is what gets it there; without it these convolutions
// demote to f16 and keep their quantize/dequantize on the CPU.

#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#scalar4 = affine_map<(d0, d1, d2, d3) -> ()>
#map1 = affine_map<(d0) -> (d0)>
#scalar1 = affine_map<(d0) -> ()>

// 1x1, S8S8, with a bias.
// CHECK-LABEL: util.func public @qdq_1x1_s8s8
// CHECK-NOT:   iree_linalg_ext.quantize_affine
// CHECK:       flow.dispatch @rocket_dynamic_int8_requant_executable
// CHECK-NOT:   iree_linalg_ext.quantize_affine
// CHECK:       util.return
func.func @qdq_1x1_s8s8(%x: tensor<1x64x8x8xi8>) -> tensor<1x64x8x8xi8> {
  %wq = arith.constant dense<3> : tensor<64x64x1x1xi8>
  %bq = arith.constant dense<1000> : tensor<64xi32>
  %sx = arith.constant 2.000000e-02 : f32
  %sw = arith.constant 4.000000e-03 : f32
  %sb = arith.constant 8.000000e-05 : f32
  %sy = arith.constant 5.000000e-02 : f32
  %zx = arith.constant -128 : i64
  %zy = arith.constant -128 : i64
  %e0 = tensor.empty() : tensor<1x64x8x8xf32>
  %xf = iree_linalg_ext.dequantize_affine <indexing_maps = [#map4, #scalar4, #scalar4, #map4], quant_min = -128, quant_max = 127>
      ins(%x, %sx, %zx : tensor<1x64x8x8xi8>, f32, i64) outs(%e0 : tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
  %e1 = tensor.empty() : tensor<64x64x1x1xf32>
  %wf = iree_linalg_ext.dequantize_affine <indexing_maps = [#map4, #scalar4, #map4], quant_min = -128, quant_max = 127>
      ins(%wq, %sw : tensor<64x64x1x1xi8>, f32) outs(%e1 : tensor<64x64x1x1xf32>) -> tensor<64x64x1x1xf32>
  %e2 = tensor.empty() : tensor<64xf32>
  %bf = iree_linalg_ext.dequantize_affine <indexing_maps = [#map1, #scalar1, #map1], quant_min = -2147483648, quant_max = 2147483647>
      ins(%bq, %sb : tensor<64xi32>, f32) outs(%e2 : tensor<64xf32>) -> tensor<64xf32>
  %bb = linalg.broadcast ins(%bf : tensor<64xf32>) outs(%e0 : tensor<1x64x8x8xf32>) dimensions = [0, 2, 3]
  %acc = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%xf, %wf : tensor<1x64x8x8xf32>, tensor<64x64x1x1xf32>) outs(%bb : tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
  %e8 = tensor.empty() : tensor<1x64x8x8xi8>
  %q = iree_linalg_ext.quantize_affine <indexing_maps = [#map4, #scalar4, #scalar4, #map4], quant_min = -128, quant_max = 127>
      ins(%acc, %sy, %zy : tensor<1x64x8x8xf32>, f32, i64) outs(%e8 : tensor<1x64x8x8xi8>) -> tensor<1x64x8x8xi8>
  return %q : tensor<1x64x8x8xi8>
}

// 3x3 with "same" padding, after a ReLU: zero point -128, which is the one
// fill the pad1 requantized target bakes, so the CNA does the padding too.
// CHECK-LABEL: util.func public @qdq_3x3_pad1
// CHECK-NOT:   tensor.pad
// CHECK:       flow.dispatch @rocket_dynamic_int8_requant_pad1_executable
// CHECK-NOT:   iree_linalg_ext.quantize_affine
// CHECK:       util.return
func.func @qdq_3x3_pad1(%x: tensor<1x64x8x8xi8>) -> tensor<1x64x8x8xi8> {
  %wq = arith.constant dense<-2> : tensor<64x64x3x3xi8>
  %bq = arith.constant dense<77> : tensor<64xi32>
  %sx = arith.constant 2.000000e-02 : f32
  %sw = arith.constant 4.000000e-03 : f32
  %sb = arith.constant 8.000000e-05 : f32
  %sy = arith.constant 5.000000e-02 : f32
  %zx = arith.constant -128 : i64
  %zy = arith.constant -128 : i64
  %zero = arith.constant 0.000000e+00 : f32
  %e0 = tensor.empty() : tensor<1x64x8x8xf32>
  %xf = iree_linalg_ext.dequantize_affine <indexing_maps = [#map4, #scalar4, #scalar4, #map4], quant_min = -128, quant_max = 127>
      ins(%x, %sx, %zx : tensor<1x64x8x8xi8>, f32, i64) outs(%e0 : tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
  %xp = tensor.pad %xf low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a: index, %b: index, %c: index, %d: index):
    tensor.yield %zero : f32
  } : tensor<1x64x8x8xf32> to tensor<1x64x10x10xf32>
  %e1 = tensor.empty() : tensor<64x64x3x3xf32>
  %wf = iree_linalg_ext.dequantize_affine <indexing_maps = [#map4, #scalar4, #map4], quant_min = -128, quant_max = 127>
      ins(%wq, %sw : tensor<64x64x3x3xi8>, f32) outs(%e1 : tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
  %e2 = tensor.empty() : tensor<64xf32>
  %bf = iree_linalg_ext.dequantize_affine <indexing_maps = [#map1, #scalar1, #map1], quant_min = -2147483648, quant_max = 2147483647>
      ins(%bq, %sb : tensor<64xi32>, f32) outs(%e2 : tensor<64xf32>) -> tensor<64xf32>
  %bb = linalg.broadcast ins(%bf : tensor<64xf32>) outs(%e0 : tensor<1x64x8x8xf32>) dimensions = [0, 2, 3]
  %acc = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%xp, %wf : tensor<1x64x10x10xf32>, tensor<64x64x3x3xf32>) outs(%bb : tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
  %e8 = tensor.empty() : tensor<1x64x8x8xi8>
  %q = iree_linalg_ext.quantize_affine <indexing_maps = [#map4, #scalar4, #scalar4, #map4], quant_min = -128, quant_max = 127>
      ins(%acc, %sy, %zy : tensor<1x64x8x8xf32>, f32, i64) outs(%e8 : tensor<1x64x8x8xi8>) -> tensor<1x64x8x8xi8>
  return %q : tensor<1x64x8x8xi8>
}

// Per-channel weight scales are declined, so this keeps today's fp16 path:
// an f16 Rocket convolution with the quantization on the CPU.
// CHECK-LABEL: util.func public @qdq_1x1_per_channel_stays_fp16
// CHECK-NOT:   int8_requant
// CHECK:       iree_linalg_ext.quantize_affine
// CHECK:       util.return
func.func @qdq_1x1_per_channel_stays_fp16(%x: tensor<1x64x8x8xi8>) -> tensor<1x64x8x8xi8> {
  %wq = arith.constant dense<3> : tensor<64x64x1x1xi8>
  %sw = arith.constant dense<4.000000e-03> : tensor<64xf32>
  %sx = arith.constant 2.000000e-02 : f32
  %sy = arith.constant 5.000000e-02 : f32
  %zx = arith.constant -128 : i64
  %zy = arith.constant -128 : i64
  %zero = arith.constant 0.000000e+00 : f32
  %e0 = tensor.empty() : tensor<1x64x8x8xf32>
  %xf = iree_linalg_ext.dequantize_affine <indexing_maps = [#map4, #scalar4, #scalar4, #map4], quant_min = -128, quant_max = 127>
      ins(%x, %sx, %zx : tensor<1x64x8x8xi8>, f32, i64) outs(%e0 : tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
  %e1 = tensor.empty() : tensor<64x64x1x1xf32>
  %wf = iree_linalg_ext.dequantize_affine <indexing_maps = [#map4, affine_map<(d0, d1, d2, d3) -> (d0)>, #map4], quant_min = -128, quant_max = 127>
      ins(%wq, %sw : tensor<64x64x1x1xi8>, tensor<64xf32>) outs(%e1 : tensor<64x64x1x1xf32>) -> tensor<64x64x1x1xf32>
  %init = linalg.fill ins(%zero : f32) outs(%e0 : tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
  %acc = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%xf, %wf : tensor<1x64x8x8xf32>, tensor<64x64x1x1xf32>) outs(%init : tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
  %e8 = tensor.empty() : tensor<1x64x8x8xi8>
  %q = iree_linalg_ext.quantize_affine <indexing_maps = [#map4, #scalar4, #scalar4, #map4], quant_min = -128, quant_max = 127>
      ins(%acc, %sy, %zy : tensor<1x64x8x8xf32>, f32, i64) outs(%e8 : tensor<1x64x8x8xi8>) -> tensor<1x64x8x8xi8>
  return %q : tensor<1x64x8x8xi8>
}
