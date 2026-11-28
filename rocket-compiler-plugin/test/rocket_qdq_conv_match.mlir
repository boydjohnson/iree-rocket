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

// `rocket-drop-unit-channel-scales`: the per-tensor convolutions' factors
// are all 1, so their executable is the spec's own, rewritten without
// `per_channel_scale` and its fourth binding; the per-channel convolution
// below runs a `_per_channel` clone that keeps both.
// CHECK:       hal.executable private @rocket_dynamic_int8_requant_executable {
// CHECK-NOT:     per_channel_scale
// CHECK:       hal.executable private @rocket_dynamic_int8_requant_executable_per_channel {
// CHECK:         per_channel_scale = true

// 1x1, S8S8, with a bias.
// CHECK-LABEL: util.func public @qdq_1x1_s8s8
// CHECK-NOT:   iree_linalg_ext.quantize_affine
// CHECK:       flow.dispatch @rocket_dynamic_int8_requant_executable::
// CHECK-SAME:  tensor<?xi32>{%c64}) ->
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

// Per-channel weight scales (ORT's per_channel=True): the requantization
// factor per channel, 0.25 to 1 of the largest here, is its own binding
// after the bias, and the convolution is requantized on the NPU.
// CHECK-LABEL: util.func public @qdq_1x1_per_channel(
// CHECK:       flow.dispatch @rocket_dynamic_int8_requant_executable_per_channel::
// CHECK-SAME:  tensor<?xi32>{%c64}, tensor<?xf32>{%c64})
// CHECK-NOT:   iree_linalg_ext.quantize_affine
// CHECK:       util.return
func.func @qdq_1x1_per_channel(%x: tensor<1x64x8x8xi8>) -> tensor<1x64x8x8xi8> {
  %wq = arith.constant dense<3> : tensor<64x64x1x1xi8>
  %sw = arith.constant dense<[4.000000e-03, 8.000000e-03, 1.200000e-02, 1.600000e-02, 4.000000e-03, 8.000000e-03, 1.200000e-02, 1.600000e-02, 4.000000e-03, 8.000000e-03, 1.200000e-02, 1.600000e-02, 4.000000e-03, 8.000000e-03, 1.200000e-02, 1.600000e-02, 4.000000e-03, 8.000000e-03, 1.200000e-02, 1.600000e-02, 4.000000e-03, 8.000000e-03, 1.200000e-02, 1.600000e-02, 4.000000e-03, 8.000000e-03, 1.200000e-02, 1.600000e-02, 4.000000e-03, 8.000000e-03, 1.200000e-02, 1.600000e-02, 4.000000e-03, 8.000000e-03, 1.200000e-02, 1.600000e-02, 4.000000e-03, 8.000000e-03, 1.200000e-02, 1.600000e-02, 4.000000e-03, 8.000000e-03, 1.200000e-02, 1.600000e-02, 4.000000e-03, 8.000000e-03, 1.200000e-02, 1.600000e-02, 4.000000e-03, 8.000000e-03, 1.200000e-02, 1.600000e-02, 4.000000e-03, 8.000000e-03, 1.200000e-02, 1.600000e-02, 4.000000e-03, 8.000000e-03, 1.200000e-02, 1.600000e-02, 4.000000e-03, 8.000000e-03, 1.200000e-02, 1.600000e-02]> : tensor<64xf32>
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

// One channel's factor is 1/1000 of the largest. A lone 16-bit multiplier
// lane could not carry that within a code; with the BS shift lane it is
// mantissa 16384 at shift 24, so the convolution stays on int8.
// CHECK-LABEL: util.func public @qdq_1x1_per_channel_small_factor_int8
// CHECK-NOT:   iree_linalg_ext.quantize_affine
// CHECK:       flow.dispatch @rocket_dynamic_int8_requant_executable_per_channel::
// CHECK-NOT:   iree_linalg_ext.quantize_affine
// CHECK:       util.return
func.func @qdq_1x1_per_channel_small_factor_int8(%x: tensor<1x64x8x8xi8>) -> tensor<1x64x8x8xi8> {
  %wq = arith.constant dense<3> : tensor<64x64x1x1xi8>
  %sw = arith.constant dense<[4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-06, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03, 4.000000e-03]> : tensor<64xf32>
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

// One channel's factor is 1e-15 of the largest, past even the BS shift
// lane's reach (kMinChannelScale, 2^-49), and the channel is not dead: its
// conv term reaches 0.4 * 4e-7 * 192 * 255 = 0.008 codes, over
// kDeadChannelTerm. So this keeps the fp16 path: an f16 Rocket convolution
// with the quantization on the CPU.
// CHECK-LABEL: util.func public @qdq_1x1_per_channel_tiny_factor_stays_fp16
// CHECK-NOT:   int8_requant
// CHECK:       iree_linalg_ext.quantize_affine
// CHECK:       util.return
func.func @qdq_1x1_per_channel_tiny_factor_stays_fp16(%x: tensor<1x64x8x8xi8>) -> tensor<1x64x8x8xi8> {
  %wq = arith.constant dense<3> : tensor<64x64x1x1xi8>
  %sw = arith.constant dense<[4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e-07, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08, 4.000000e+08]> : tensor<64xf32>
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

// A 3x3 depthwise at stride 1 and 2 with pad 1 after a ReLU, as in every
// MobileNetV2 depthwise layer: the pad goes to the CNA on the depthwise
// requantized pad1 targets.
// CHECK-LABEL: util.func public @qdq_depthwise_pad1
// CHECK-NOT:   tensor.pad
// CHECK:       flow.dispatch @rocket_dynamic_depthwise_int8_requant_pad1_executable
// CHECK-NOT:   tensor.pad
// CHECK:       flow.dispatch @rocket_dynamic_depthwise_int8_requant_pad1_s2_executable
// CHECK-NOT:   iree_linalg_ext.quantize_affine
// CHECK:       util.return
func.func @qdq_depthwise_pad1(%x: tensor<1x64x8x8xi8>) -> tensor<1x64x4x4xi8> {
  %wq = arith.constant dense<5> : tensor<64x1x3x3xi8>
  %bq = arith.constant dense<-300> : tensor<64xi32>
  %sx = arith.constant 2.000000e-02 : f32
  %sw = arith.constant 4.000000e-03 : f32
  %sb = arith.constant 8.000000e-05 : f32
  %sy = arith.constant 5.000000e-02 : f32
  %zx = arith.constant -128 : i64
  %zero = arith.constant 0.000000e+00 : f32
  %e0 = tensor.empty() : tensor<1x64x8x8xf32>
  %e1 = tensor.empty() : tensor<64x1x3x3xf32>
  %wf = iree_linalg_ext.dequantize_affine <indexing_maps = [#map4, #scalar4, #map4], quant_min = -128, quant_max = 127>
      ins(%wq, %sw : tensor<64x1x3x3xi8>, f32) outs(%e1 : tensor<64x1x3x3xf32>) -> tensor<64x1x3x3xf32>
  %wc = tensor.collapse_shape %wf [[0, 1], [2], [3]] : tensor<64x1x3x3xf32> into tensor<64x3x3xf32>
  %e2 = tensor.empty() : tensor<64xf32>
  %bf = iree_linalg_ext.dequantize_affine <indexing_maps = [#map1, #scalar1, #map1], quant_min = -2147483648, quant_max = 2147483647>
      ins(%bq, %sb : tensor<64xi32>, f32) outs(%e2 : tensor<64xf32>) -> tensor<64xf32>

  // Stride 1.
  %xf = iree_linalg_ext.dequantize_affine <indexing_maps = [#map4, #scalar4, #scalar4, #map4], quant_min = -128, quant_max = 127>
      ins(%x, %sx, %zx : tensor<1x64x8x8xi8>, f32, i64) outs(%e0 : tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
  %xp = tensor.pad %xf low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a: index, %b: index, %c: index, %d: index):
    tensor.yield %zero : f32
  } : tensor<1x64x8x8xf32> to tensor<1x64x10x10xf32>
  %bb = linalg.broadcast ins(%bf : tensor<64xf32>) outs(%e0 : tensor<1x64x8x8xf32>) dimensions = [0, 2, 3]
  %acc = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%xp, %wc : tensor<1x64x10x10xf32>, tensor<64x3x3xf32>) outs(%bb : tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
  %e8 = tensor.empty() : tensor<1x64x8x8xi8>
  %q = iree_linalg_ext.quantize_affine <indexing_maps = [#map4, #scalar4, #scalar4, #map4], quant_min = -128, quant_max = 127>
      ins(%acc, %sy, %zx : tensor<1x64x8x8xf32>, f32, i64) outs(%e8 : tensor<1x64x8x8xi8>) -> tensor<1x64x8x8xi8>

  // Stride 2, reading the first one's codes.
  %yf = iree_linalg_ext.dequantize_affine <indexing_maps = [#map4, #scalar4, #scalar4, #map4], quant_min = -128, quant_max = 127>
      ins(%q, %sy, %zx : tensor<1x64x8x8xi8>, f32, i64) outs(%e0 : tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
  %yp = tensor.pad %yf low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a: index, %b: index, %c: index, %d: index):
    tensor.yield %zero : f32
  } : tensor<1x64x8x8xf32> to tensor<1x64x10x10xf32>
  %e3 = tensor.empty() : tensor<1x64x4x4xf32>
  %bb2 = linalg.broadcast ins(%bf : tensor<64xf32>) outs(%e3 : tensor<1x64x4x4xf32>) dimensions = [0, 2, 3]
  %sb2 = arith.constant 2.000000e-04 : f32
  %bf2 = iree_linalg_ext.dequantize_affine <indexing_maps = [#map1, #scalar1, #map1], quant_min = -2147483648, quant_max = 2147483647>
      ins(%bq, %sb2 : tensor<64xi32>, f32) outs(%e2 : tensor<64xf32>) -> tensor<64xf32>
  %bb3 = linalg.broadcast ins(%bf2 : tensor<64xf32>) outs(%e3 : tensor<1x64x4x4xf32>) dimensions = [0, 2, 3]
  %acc2 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
      ins(%yp, %wc : tensor<1x64x10x10xf32>, tensor<64x3x3xf32>) outs(%bb3 : tensor<1x64x4x4xf32>) -> tensor<1x64x4x4xf32>
  %e9 = tensor.empty() : tensor<1x64x4x4xi8>
  %q2 = iree_linalg_ext.quantize_affine <indexing_maps = [#map4, #scalar4, #scalar4, #map4], quant_min = -128, quant_max = 127>
      ins(%acc2, %sy, %zx : tensor<1x64x4x4xf32>, f32, i64) outs(%e9 : tensor<1x64x4x4xi8>) -> tensor<1x64x4x4xi8>
  return %q2 : tensor<1x64x4x4xi8>
}


// A QDQ classifier, `Gemm(transB = 1)` with per-channel weight scales, as the
// importer writes it (the bias add still a `linalg.generic`): it runs as the
// int8 1x1 convolution on the requantized executable, keeping its factors.
// CHECK-LABEL: util.func public @qdq_fc_per_channel
// CHECK-NOT:   linalg.matmul
// CHECK:       flow.dispatch @rocket_dynamic_int8_requant_executable_per_channel::
// CHECK-NOT:   iree_linalg_ext.quantize_affine
// CHECK:       util.return
func.func @qdq_fc_per_channel(%x: tensor<1x64xi8>) -> tensor<1x32xi8> {
  %wq = arith.constant dense<3> : tensor<32x64xi8>
  %sw = arith.constant dense<[4.000000e-03, 8.000000e-03, 1.200000e-02, 1.600000e-02, 4.000000e-03, 8.000000e-03, 1.200000e-02, 1.600000e-02, 4.000000e-03, 8.000000e-03, 1.200000e-02, 1.600000e-02, 4.000000e-03, 8.000000e-03, 1.200000e-02, 1.600000e-02, 4.000000e-03, 8.000000e-03, 1.200000e-02, 1.600000e-02, 4.000000e-03, 8.000000e-03, 1.200000e-02, 1.600000e-02, 4.000000e-03, 8.000000e-03, 1.200000e-02, 1.600000e-02, 4.000000e-03, 8.000000e-03, 1.200000e-02, 1.600000e-02]> : tensor<32xf32>
  %bq = arith.constant dense<100> : tensor<32xi32>
  %sb = arith.constant dense<[1.000000e-03, 2.000000e-03, 3.000000e-03, 4.000000e-03, 1.000000e-03, 2.000000e-03, 3.000000e-03, 4.000000e-03, 1.000000e-03, 2.000000e-03, 3.000000e-03, 4.000000e-03, 1.000000e-03, 2.000000e-03, 3.000000e-03, 4.000000e-03, 1.000000e-03, 2.000000e-03, 3.000000e-03, 4.000000e-03, 1.000000e-03, 2.000000e-03, 3.000000e-03, 4.000000e-03, 1.000000e-03, 2.000000e-03, 3.000000e-03, 4.000000e-03, 1.000000e-03, 2.000000e-03, 3.000000e-03, 4.000000e-03]> : tensor<32xf32>
  %sx = arith.constant 2.500000e-01 : f32
  %sy = arith.constant 1.000000e+00 : f32
  %zx = arith.constant -128 : i64
  %zy = arith.constant -70 : i64
  %zero = arith.constant 0.000000e+00 : f32
  %e0 = tensor.empty() : tensor<1x64xf32>
  %xf = iree_linalg_ext.dequantize_affine <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], quant_min = -128, quant_max = 127>
      ins(%x, %sx, %zx : tensor<1x64xi8>, f32, i64) outs(%e0 : tensor<1x64xf32>) -> tensor<1x64xf32>
  %e1 = tensor.empty() : tensor<32x64xf32>
  %wf = iree_linalg_ext.dequantize_affine <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], quant_min = -128, quant_max = 127>
      ins(%wq, %sw : tensor<32x64xi8>, tensor<32xf32>) outs(%e1 : tensor<32x64xf32>) -> tensor<32x64xf32>
  %e2 = tensor.empty() : tensor<64x32xf32>
  %wt = linalg.transpose ins(%wf : tensor<32x64xf32>) outs(%e2 : tensor<64x32xf32>) permutation = [1, 0]
  %e3 = tensor.empty() : tensor<1x32xf32>
  %init = linalg.fill ins(%zero : f32) outs(%e3 : tensor<1x32xf32>) -> tensor<1x32xf32>
  %y = linalg.matmul ins(%xf, %wt : tensor<1x64xf32>, tensor<64x32xf32>) outs(%init : tensor<1x32xf32>) -> tensor<1x32xf32>
  %e4 = tensor.empty() : tensor<32xf32>
  %bf = iree_linalg_ext.dequantize_affine <indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], quant_min = -2147483648, quant_max = 2147483647>
      ins(%bq, %sb : tensor<32xi32>, tensor<32xf32>) outs(%e4 : tensor<32xf32>) -> tensor<32xf32>
  %yb = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
      ins(%y, %bf : tensor<1x32xf32>, tensor<32xf32>) outs(%e3 : tensor<1x32xf32>) {
  ^bb0(%a: f32, %b: f32, %o: f32):
    %s = arith.addf %a, %b : f32
    linalg.yield %s : f32
  } -> tensor<1x32xf32>
  %e5 = tensor.empty() : tensor<1x32xi8>
  %q = iree_linalg_ext.quantize_affine <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], quant_min = -128, quant_max = 127>
      ins(%yb, %sy, %zy : tensor<1x32xf32>, f32, i64) outs(%e5 : tensor<1x32xi8>) -> tensor<1x32xi8>
  return %q : tensor<1x32xi8>
}
