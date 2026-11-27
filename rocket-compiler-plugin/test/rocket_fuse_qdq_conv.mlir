// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(rocket-fuse-qdq-conv))" %s | FileCheck %s

// An ONNX QDQ convolution as IREE 3.14 has it after the channels-last
// conversion, copied in shape from stock `quantize_static` ResNet50: the
// activation and the constant weights come through
// `iree_linalg_ext.dequantize_affine`, the bias is a dequantized i32
// broadcast into the f32 accumulator's init, and the result goes back to
// NCHW and through `quantize_affine`. Signed activations (S8S8, ORT's
// default), input zero point -128.

#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#scalar4 = affine_map<(d0, d1, d2, d3) -> ()>
#map1 = affine_map<(d0) -> (d0)>
#scalar1 = affine_map<(d0) -> ()>

// CHECK-LABEL: func.func @qdq_conv_s8s8
//   The requantization's scalars: scale = 0.02 * 0.004 / 0.05 = 0.0016,
//   output zero point -128 as it is (signed), the int8 clamp bounds.
// CHECK-DAG:   %[[HI:.+]] = arith.constant 1.270000e+02 : f32
// CHECK-DAG:   %[[LO:.+]] = arith.constant -1.280000e+02 : f32
// CHECK-DAG:   %[[SCALE:.+]] = arith.constant 1.600000e-03 : f32
// CHECK-DAG:   %[[ZP:.+]] = arith.constant -128 : i32
//   The convolution reads the activation's codes, transposed on i8, and the
//   weight codes, over a zero init.
// CHECK:       %[[XT:.+]] = linalg.transpose ins(%[[X:.+]] : tensor<1x64x8x8xi8>)
// CHECK:       %[[WT:.+]] = linalg.transpose ins(%{{.+}} : tensor<32x64x1x1xi8>)
// CHECK:       %[[INIT:.+]] = linalg.fill ins(%{{.+}} : i32) outs(%{{.+}} : tensor<1x8x8x32xi32>)
// CHECK:       %[[ACC:.+]] = linalg.conv_2d_nhwc_hwcf
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>
// CHECK-SAME:      ins(%[[XT]], %[[WT]] : tensor<1x8x8x64xi8>, tensor<1x1x64x32xi8>)
// CHECK-SAME:      outs(%[[INIT]] : tensor<1x8x8x32xi32>)
//   sum_k(w), then bias - zx * sum_k(w) with zx = -128.
// CHECK:       %[[SUM:.+]] = linalg.generic {{.*}}iterator_types = ["reduction", "reduction", "reduction", "parallel"]
// CHECK-SAME:      ins(%[[WT]] : tensor<1x1x64x32xi8>)
// CHECK:         arith.extsi
// CHECK:       %[[BIAS:.+]] = linalg.generic
// CHECK-SAME:      ins(%{{.+}}, %[[SUM]] : tensor<32xi32>, tensor<32xi32>)
// CHECK:         arith.muli %{{.+}}, %[[ZP]]
// CHECK:         arith.subi
//   The canonical requantization.
// CHECK:       %[[Q:.+]] = linalg.generic
// CHECK-SAME:      ins(%[[ACC]], %[[BIAS]], %[[SCALE]], %[[ZP]], %[[LO]], %[[HI]] : tensor<1x8x8x32xi32>, tensor<32xi32>, f32, i32, f32, f32)
// CHECK:         arith.addi
// CHECK:         arith.sitofp
// CHECK:         arith.mulf
// CHECK:         math.roundeven
// CHECK:         arith.addf
// CHECK:         arith.maximumf
// CHECK:         arith.minimumf
// CHECK:         arith.fptosi
//   Back to NCHW on i8. No encoding shift for signed codes.
// CHECK-NOT:   arith.addi %{{.+}}, %c-128_i8
// CHECK:       %[[OUT:.+]] = linalg.transpose ins(%[[Q]] : tensor<1x8x8x32xi8>)
// CHECK-SAME:      permutation = [0, 3, 1, 2]
// CHECK:       return %[[OUT]]
// CHECK-NOT:   iree_linalg_ext.quantize_affine
func.func @qdq_conv_s8s8(%x: tensor<1x64x8x8xi8>) -> tensor<1x32x8x8xi8> {
  %wq = arith.constant dense<3> : tensor<32x64x1x1xi8>
  %bq = arith.constant dense<1000> : tensor<32xi32>
  %sx = arith.constant 2.000000e-02 : f32
  %sw = arith.constant 4.000000e-03 : f32
  %sb = arith.constant 8.000000e-05 : f32
  %sy = arith.constant 5.000000e-02 : f32
  %zx = arith.constant -128 : i64
  %zy = arith.constant -128 : i64
  %e0 = tensor.empty() : tensor<1x64x8x8xf32>
  %xf = iree_linalg_ext.dequantize_affine <indexing_maps = [#map4, #scalar4, #scalar4, #map4], quant_min = -128, quant_max = 127>
      ins(%x, %sx, %zx : tensor<1x64x8x8xi8>, f32, i64) outs(%e0 : tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
  %e1 = tensor.empty() : tensor<32x64x1x1xf32>
  %wf = iree_linalg_ext.dequantize_affine <indexing_maps = [#map4, #scalar4, #map4], quant_min = -128, quant_max = 127>
      ins(%wq, %sw : tensor<32x64x1x1xi8>, f32) outs(%e1 : tensor<32x64x1x1xf32>) -> tensor<32x64x1x1xf32>
  %e2 = tensor.empty() : tensor<32xf32>
  %bf = iree_linalg_ext.dequantize_affine <indexing_maps = [#map1, #scalar1, #map1], quant_min = -2147483648, quant_max = 2147483647>
      ins(%bq, %sb : tensor<32xi32>, f32) outs(%e2 : tensor<32xf32>) -> tensor<32xf32>
  %e3 = tensor.empty() : tensor<1x32x8x8xf32>
  %bb = linalg.broadcast ins(%bf : tensor<32xf32>) outs(%e3 : tensor<1x32x8x8xf32>) dimensions = [0, 2, 3]
  %e4 = tensor.empty() : tensor<1x8x8x64xf32>
  %xt = linalg.transpose ins(%xf : tensor<1x64x8x8xf32>) outs(%e4 : tensor<1x8x8x64xf32>) permutation = [0, 2, 3, 1]
  %e5 = tensor.empty() : tensor<1x1x64x32xf32>
  %wt = linalg.transpose ins(%wf : tensor<32x64x1x1xf32>) outs(%e5 : tensor<1x1x64x32xf32>) permutation = [2, 3, 1, 0]
  %e6 = tensor.empty() : tensor<1x8x8x32xf32>
  %init = linalg.transpose ins(%bb : tensor<1x32x8x8xf32>) outs(%e6 : tensor<1x8x8x32xf32>) permutation = [0, 2, 3, 1]
  %acc = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%xt, %wt : tensor<1x8x8x64xf32>, tensor<1x1x64x32xf32>) outs(%init : tensor<1x8x8x32xf32>) -> tensor<1x8x8x32xf32>
  %e7 = tensor.empty() : tensor<1x32x8x8xf32>
  %nchw = linalg.transpose ins(%acc : tensor<1x8x8x32xf32>) outs(%e7 : tensor<1x32x8x8xf32>) permutation = [0, 3, 1, 2]
  %e8 = tensor.empty() : tensor<1x32x8x8xi8>
  %q = iree_linalg_ext.quantize_affine <indexing_maps = [#map4, #scalar4, #scalar4, #map4], quant_min = -128, quant_max = 127>
      ins(%nchw, %sy, %zy : tensor<1x32x8x8xf32>, f32, i64) outs(%e8 : tensor<1x32x8x8xi8>) -> tensor<1x32x8x8xi8>
  return %q : tensor<1x32x8x8xi8>
}

// -----

// Unsigned activations (U8S8), a 3x3 convolution with "same" padding in the
// rank-5 expand/pad/collapse spelling the channels-last conversion gives a
// padded NCHW input, and no bias. The codes shift into the signed encoding
// first, the pad fills with the input zero point in that encoding
// (u8 0 -> -128, which `rocket-fold-conv-pad` can hand to the CNA), and the
// result shifts back. A zero point of 0 in the signed encoding would need no
// correction; -128 does, so the folded bias is `0 - (-128) * sum_k(w)`.
//
// The rank-5 round trip comes out as one rank-4 NHWC pad: that is the form
// the int8 pad1 requantized template claims, and a rank-5 pad would leave
// the convolution to the int8_accumulator path (measured: it did).

#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#scalar4 = affine_map<(d0, d1, d2, d3) -> ()>

// CHECK-LABEL: func.func @qdq_conv_u8s8_padded_no_bias
//   Output zero point 10 (ui8) is -118 signed.
// CHECK-DAG:   %[[ZP:.+]] = arith.constant -118 : i32
// CHECK-DAG:   %[[SHIFT:.+]] = arith.constant -128 : i8
// CHECK:       %[[XS:.+]] = linalg.generic
// CHECK-SAME:      ins(%{{.+}} : tensor<1x16x8x8xi8>)
// CHECK:         arith.addi %{{.+}}, %[[SHIFT]]
// CHECK:       %[[XT:.+]] = linalg.transpose ins(%[[XS]] : tensor<1x16x8x8xi8>)
// CHECK-NOT:   tensor.expand_shape
// CHECK:       %[[XP:.+]] = tensor.pad %[[XT]] low[0, 1, 1, 0] high[0, 1, 1, 0]
// CHECK:         tensor.yield %[[SHIFT]] : i8
// CHECK-NEXT:  } : tensor<1x8x8x16xi8> to tensor<1x10x10x16xi8>
// CHECK-NOT:   tensor.collapse_shape
// CHECK:       %[[ACC:.+]] = linalg.conv_2d_nhwc_hwcf
// CHECK-SAME:      ins(%[[XP]], %{{.+}} : tensor<1x10x10x16xi8>, tensor<3x3x16x16xi8>)
// CHECK:       %[[SUM:.+]] = linalg.generic {{.*}}iterator_types = ["reduction", "reduction", "reduction", "parallel"]
// CHECK:       %[[BIAS:.+]] = linalg.generic
// CHECK-SAME:      ins(%[[SUM]] : tensor<16xi32>)
// CHECK:       %[[Q:.+]] = linalg.generic
// CHECK-SAME:      ins(%[[ACC]], %[[BIAS]], %{{.+}}, %[[ZP]]
// CHECK:       %[[QS:.+]] = linalg.generic
// CHECK-SAME:      ins(%[[Q]] : tensor<1x8x8x16xi8>)
// CHECK:         arith.addi %{{.+}}, %[[SHIFT]]
// CHECK:       linalg.transpose ins(%[[QS]] : tensor<1x8x8x16xi8>)
// CHECK-NOT:   iree_linalg_ext.quantize_affine
func.func @qdq_conv_u8s8_padded_no_bias(%x: tensor<1x16x8x8xi8>) -> tensor<1x16x8x8xi8> {
  %wq = arith.constant dense<-2> : tensor<16x16x3x3xi8>
  %sx = arith.constant 2.000000e-02 : f32
  %sw = arith.constant 4.000000e-03 : f32
  %sy = arith.constant 5.000000e-02 : f32
  %zx = arith.constant 0 : i64
  %zy = arith.constant 10 : i64
  %zero = arith.constant 0.000000e+00 : f32
  %e0 = tensor.empty() : tensor<1x16x8x8xf32>
  %xf = iree_linalg_ext.dequantize_affine <indexing_maps = [#map4, #scalar4, #scalar4, #map4], input_unsigned>
      ins(%x, %sx, %zx : tensor<1x16x8x8xi8>, f32, i64) outs(%e0 : tensor<1x16x8x8xf32>) -> tensor<1x16x8x8xf32>
  %e1 = tensor.empty() : tensor<16x16x3x3xf32>
  %wf = iree_linalg_ext.dequantize_affine <indexing_maps = [#map4, #scalar4, #map4], quant_min = -128, quant_max = 127>
      ins(%wq, %sw : tensor<16x16x3x3xi8>, f32) outs(%e1 : tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
  %e4 = tensor.empty() : tensor<1x8x8x16xf32>
  %xt = linalg.transpose ins(%xf : tensor<1x16x8x8xf32>) outs(%e4 : tensor<1x8x8x16xf32>) permutation = [0, 2, 3, 1]
  %xe = tensor.expand_shape %xt [[0], [1, 2], [3], [4]] output_shape [1, 1, 8, 8, 16] : tensor<1x8x8x16xf32> into tensor<1x1x8x8x16xf32>
  %xp = tensor.pad %xe low[0, 0, 1, 1, 0] high[0, 0, 1, 1, 0] {
  ^bb0(%a: index, %b: index, %c: index, %d: index, %e: index):
    tensor.yield %zero : f32
  } : tensor<1x1x8x8x16xf32> to tensor<1x1x10x10x16xf32>
  %xc = tensor.collapse_shape %xp [[0], [1, 2], [3], [4]] : tensor<1x1x10x10x16xf32> into tensor<1x10x10x16xf32>
  %e5 = tensor.empty() : tensor<3x3x16x16xf32>
  %wt = linalg.transpose ins(%wf : tensor<16x16x3x3xf32>) outs(%e5 : tensor<3x3x16x16xf32>) permutation = [2, 3, 1, 0]
  %e6 = tensor.empty() : tensor<1x8x8x16xf32>
  %init = linalg.fill ins(%zero : f32) outs(%e6 : tensor<1x8x8x16xf32>) -> tensor<1x8x8x16xf32>
  %acc = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%xc, %wt : tensor<1x10x10x16xf32>, tensor<3x3x16x16xf32>) outs(%init : tensor<1x8x8x16xf32>) -> tensor<1x8x8x16xf32>
  %e7 = tensor.empty() : tensor<1x16x8x8xf32>
  %nchw = linalg.transpose ins(%acc : tensor<1x8x8x16xf32>) outs(%e7 : tensor<1x16x8x8xf32>) permutation = [0, 3, 1, 2]
  %e8 = tensor.empty() : tensor<1x16x8x8xi8>
  %q = iree_linalg_ext.quantize_affine <indexing_maps = [#map4, #scalar4, #scalar4, #map4], quant_min = 0, quant_max = 255, storage_unsigned>
      ins(%nchw, %sy, %zy : tensor<1x16x8x8xf32>, f32, i64) outs(%e8 : tensor<1x16x8x8xi8>) -> tensor<1x16x8x8xi8>
  return %q : tensor<1x16x8x8xi8>
}

// -----

// Input zero point 0 in the signed encoding: no correction term, so no
// filter reduction, and the bias goes to the requantization as it is.

#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#scalar4 = affine_map<(d0, d1, d2, d3) -> ()>
#map1 = affine_map<(d0) -> (d0)>
#scalar1 = affine_map<(d0) -> ()>

// CHECK-LABEL: func.func @qdq_conv_zero_input_zero_point
// CHECK:       %[[BQ:.+]] = arith.constant dense<-7> : tensor<16xi32>
// CHECK:       %[[ACC:.+]] = linalg.conv_2d_nhwc_hwcf
// CHECK-NOT:   "reduction"
// CHECK:       linalg.generic
// CHECK-SAME:      ins(%[[ACC]], %[[BQ]],
// CHECK-NOT:   iree_linalg_ext.quantize_affine
func.func @qdq_conv_zero_input_zero_point(%x: tensor<1x8x4x4xi8>) -> tensor<1x16x4x4xi8> {
  %wq = arith.constant dense<1> : tensor<16x8x1x1xi8>
  %bq = arith.constant dense<-7> : tensor<16xi32>
  %sx = arith.constant 2.500000e-01 : f32
  %sw = arith.constant 5.000000e-01 : f32
  %sb = arith.constant 1.250000e-01 : f32
  %sy = arith.constant 1.000000e+00 : f32
  %e0 = tensor.empty() : tensor<1x8x4x4xf32>
  %xf = iree_linalg_ext.dequantize_affine <indexing_maps = [#map4, #scalar4, #map4], quant_min = -128, quant_max = 127>
      ins(%x, %sx : tensor<1x8x4x4xi8>, f32) outs(%e0 : tensor<1x8x4x4xf32>) -> tensor<1x8x4x4xf32>
  %e1 = tensor.empty() : tensor<16x8x1x1xf32>
  %wf = iree_linalg_ext.dequantize_affine <indexing_maps = [#map4, #scalar4, #map4], quant_min = -128, quant_max = 127>
      ins(%wq, %sw : tensor<16x8x1x1xi8>, f32) outs(%e1 : tensor<16x8x1x1xf32>) -> tensor<16x8x1x1xf32>
  %e2 = tensor.empty() : tensor<16xf32>
  %bf = iree_linalg_ext.dequantize_affine <indexing_maps = [#map1, #scalar1, #map1], quant_min = -2147483648, quant_max = 2147483647>
      ins(%bq, %sb : tensor<16xi32>, f32) outs(%e2 : tensor<16xf32>) -> tensor<16xf32>
  %e3 = tensor.empty() : tensor<1x16x4x4xf32>
  %bb = linalg.broadcast ins(%bf : tensor<16xf32>) outs(%e3 : tensor<1x16x4x4xf32>) dimensions = [0, 2, 3]
  %e4 = tensor.empty() : tensor<1x4x4x8xf32>
  %xt = linalg.transpose ins(%xf : tensor<1x8x4x4xf32>) outs(%e4 : tensor<1x4x4x8xf32>) permutation = [0, 2, 3, 1]
  %e5 = tensor.empty() : tensor<1x1x8x16xf32>
  %wt = linalg.transpose ins(%wf : tensor<16x8x1x1xf32>) outs(%e5 : tensor<1x1x8x16xf32>) permutation = [2, 3, 1, 0]
  %e6 = tensor.empty() : tensor<1x4x4x16xf32>
  %init = linalg.transpose ins(%bb : tensor<1x16x4x4xf32>) outs(%e6 : tensor<1x4x4x16xf32>) permutation = [0, 2, 3, 1]
  %acc = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%xt, %wt : tensor<1x4x4x8xf32>, tensor<1x1x8x16xf32>) outs(%init : tensor<1x4x4x16xf32>) -> tensor<1x4x4x16xf32>
  %e7 = tensor.empty() : tensor<1x16x4x4xf32>
  %nchw = linalg.transpose ins(%acc : tensor<1x4x4x16xf32>) outs(%e7 : tensor<1x16x4x4xf32>) permutation = [0, 3, 1, 2]
  %e8 = tensor.empty() : tensor<1x16x4x4xi8>
  %q = iree_linalg_ext.quantize_affine <indexing_maps = [#map4, #scalar4, #map4], quant_min = -128, quant_max = 127>
      ins(%nchw, %sy : tensor<1x16x4x4xf32>, f32) outs(%e8 : tensor<1x16x4x4xi8>) -> tensor<1x16x4x4xi8>
  return %q : tensor<1x16x4x4xi8>
}

// -----

// Declined: per-channel weight scales. The requantization ratio would be
// per channel, and the executable ABI carries one `output_scale`.

#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#scalar4 = affine_map<(d0, d1, d2, d3) -> ()>
#chan4 = affine_map<(d0, d1, d2, d3) -> (d0)>

// CHECK-LABEL: func.func @qdq_conv_per_channel_declined
// CHECK:       linalg.conv_2d_nhwc_hwcf {{.*}} -> tensor<1x4x4x16xf32>
// CHECK:       iree_linalg_ext.quantize_affine
func.func @qdq_conv_per_channel_declined(%x: tensor<1x8x4x4xi8>, %sw: tensor<16xf32>) -> tensor<1x16x4x4xi8> {
  %wq = arith.constant dense<1> : tensor<16x8x1x1xi8>
  %sx = arith.constant 2.500000e-01 : f32
  %sy = arith.constant 1.000000e+00 : f32
  %zero = arith.constant 0.000000e+00 : f32
  %e0 = tensor.empty() : tensor<1x8x4x4xf32>
  %xf = iree_linalg_ext.dequantize_affine <indexing_maps = [#map4, #scalar4, #map4], quant_min = -128, quant_max = 127>
      ins(%x, %sx : tensor<1x8x4x4xi8>, f32) outs(%e0 : tensor<1x8x4x4xf32>) -> tensor<1x8x4x4xf32>
  %e1 = tensor.empty() : tensor<16x8x1x1xf32>
  %wf = iree_linalg_ext.dequantize_affine <indexing_maps = [#map4, #chan4, #map4], quant_min = -128, quant_max = 127>
      ins(%wq, %sw : tensor<16x8x1x1xi8>, tensor<16xf32>) outs(%e1 : tensor<16x8x1x1xf32>) -> tensor<16x8x1x1xf32>
  %e4 = tensor.empty() : tensor<1x4x4x8xf32>
  %xt = linalg.transpose ins(%xf : tensor<1x8x4x4xf32>) outs(%e4 : tensor<1x4x4x8xf32>) permutation = [0, 2, 3, 1]
  %e5 = tensor.empty() : tensor<1x1x8x16xf32>
  %wt = linalg.transpose ins(%wf : tensor<16x8x1x1xf32>) outs(%e5 : tensor<1x1x8x16xf32>) permutation = [2, 3, 1, 0]
  %e6 = tensor.empty() : tensor<1x4x4x16xf32>
  %init = linalg.fill ins(%zero : f32) outs(%e6 : tensor<1x4x4x16xf32>) -> tensor<1x4x4x16xf32>
  %acc = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%xt, %wt : tensor<1x4x4x8xf32>, tensor<1x1x8x16xf32>) outs(%init : tensor<1x4x4x16xf32>) -> tensor<1x4x4x16xf32>
  %e7 = tensor.empty() : tensor<1x16x4x4xf32>
  %nchw = linalg.transpose ins(%acc : tensor<1x4x4x16xf32>) outs(%e7 : tensor<1x16x4x4xf32>) permutation = [0, 3, 1, 2]
  %e8 = tensor.empty() : tensor<1x16x4x4xi8>
  %q = iree_linalg_ext.quantize_affine <indexing_maps = [#map4, #scalar4, #map4], quant_min = -128, quant_max = 127>
      ins(%nchw, %sy : tensor<1x16x4x4xf32>, f32) outs(%e8 : tensor<1x16x4x4xi8>) -> tensor<1x16x4x4xi8>
  return %q : tensor<1x16x4x4xi8>
}

// -----

// Declined: a bias whose scale is not `sx * sw` (here 2x it). Folding it
// would need a rescale of the bias codes, which is not exact.

#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#scalar4 = affine_map<(d0, d1, d2, d3) -> ()>
#map1 = affine_map<(d0) -> (d0)>
#scalar1 = affine_map<(d0) -> ()>

// CHECK-LABEL: func.func @qdq_conv_bias_scale_declined
// CHECK:       linalg.conv_2d_nhwc_hwcf {{.*}} -> tensor<1x4x4x16xf32>
// CHECK:       iree_linalg_ext.quantize_affine
func.func @qdq_conv_bias_scale_declined(%x: tensor<1x8x4x4xi8>) -> tensor<1x16x4x4xi8> {
  %wq = arith.constant dense<1> : tensor<16x8x1x1xi8>
  %bq = arith.constant dense<-7> : tensor<16xi32>
  %sx = arith.constant 2.500000e-01 : f32
  %sw = arith.constant 5.000000e-01 : f32
  %sb = arith.constant 2.500000e-01 : f32
  %sy = arith.constant 1.000000e+00 : f32
  %e0 = tensor.empty() : tensor<1x8x4x4xf32>
  %xf = iree_linalg_ext.dequantize_affine <indexing_maps = [#map4, #scalar4, #map4], quant_min = -128, quant_max = 127>
      ins(%x, %sx : tensor<1x8x4x4xi8>, f32) outs(%e0 : tensor<1x8x4x4xf32>) -> tensor<1x8x4x4xf32>
  %e1 = tensor.empty() : tensor<16x8x1x1xf32>
  %wf = iree_linalg_ext.dequantize_affine <indexing_maps = [#map4, #scalar4, #map4], quant_min = -128, quant_max = 127>
      ins(%wq, %sw : tensor<16x8x1x1xi8>, f32) outs(%e1 : tensor<16x8x1x1xf32>) -> tensor<16x8x1x1xf32>
  %e2 = tensor.empty() : tensor<16xf32>
  %bf = iree_linalg_ext.dequantize_affine <indexing_maps = [#map1, #scalar1, #map1], quant_min = -2147483648, quant_max = 2147483647>
      ins(%bq, %sb : tensor<16xi32>, f32) outs(%e2 : tensor<16xf32>) -> tensor<16xf32>
  %e3 = tensor.empty() : tensor<1x16x4x4xf32>
  %bb = linalg.broadcast ins(%bf : tensor<16xf32>) outs(%e3 : tensor<1x16x4x4xf32>) dimensions = [0, 2, 3]
  %e4 = tensor.empty() : tensor<1x4x4x8xf32>
  %xt = linalg.transpose ins(%xf : tensor<1x8x4x4xf32>) outs(%e4 : tensor<1x4x4x8xf32>) permutation = [0, 2, 3, 1]
  %e5 = tensor.empty() : tensor<1x1x8x16xf32>
  %wt = linalg.transpose ins(%wf : tensor<16x8x1x1xf32>) outs(%e5 : tensor<1x1x8x16xf32>) permutation = [2, 3, 1, 0]
  %e6 = tensor.empty() : tensor<1x4x4x16xf32>
  %init = linalg.transpose ins(%bb : tensor<1x16x4x4xf32>) outs(%e6 : tensor<1x4x4x16xf32>) permutation = [0, 2, 3, 1]
  %acc = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%xt, %wt : tensor<1x4x4x8xf32>, tensor<1x1x8x16xf32>) outs(%init : tensor<1x4x4x16xf32>) -> tensor<1x4x4x16xf32>
  %e7 = tensor.empty() : tensor<1x16x4x4xf32>
  %nchw = linalg.transpose ins(%acc : tensor<1x4x4x16xf32>) outs(%e7 : tensor<1x16x4x4xf32>) permutation = [0, 3, 1, 2]
  %e8 = tensor.empty() : tensor<1x16x4x4xi8>
  %q = iree_linalg_ext.quantize_affine <indexing_maps = [#map4, #scalar4, #map4], quant_min = -128, quant_max = 127>
      ins(%nchw, %sy : tensor<1x16x4x4xf32>, f32) outs(%e8 : tensor<1x16x4x4xi8>) -> tensor<1x16x4x4xi8>
  return %q : tensor<1x16x4x4xi8>
}
