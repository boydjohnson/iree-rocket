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
//   Per-tensor weights: a per-channel factor of exactly 1 everywhere.
// CHECK-DAG:   %[[CS:.+]] = arith.constant dense<1.000000e+00> : tensor<32xf32>
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
// CHECK-SAME:      ins(%[[ACC]], %[[BIAS]], %[[CS]], %[[SCALE]], %[[ZP]], %[[LO]], %[[HI]] : tensor<1x8x8x32xi32>, tensor<32xi32>, tensor<32xf32>, f32, i32, f32, f32)
// CHECK:         arith.addi
// CHECK:         arith.sitofp
// CHECK:         arith.mulf
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
// CHECK-SAME:      ins(%[[ACC]], %[[BIAS]], %{{.+}}, %{{.+}}, %[[ZP]]
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

// Declined: per-channel weight scales that are not constants (a function
// argument here). The per-channel factors are compile-time data: the
// per-tensor `output_scale` is their maximum.

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

// Per-channel weight scales (ORT's per_channel=True) and the per-channel
// bias scale `sx * sw[c]` that goes with them. The ratios `sx * sw[c] / sy`
// are 0.001 to 0.004; the per-tensor scale is the largest, 0.004, and each
// channel's factor is its ratio over that, 1/4 to 1.

#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#scalar4 = affine_map<(d0, d1, d2, d3) -> ()>
#chan4 = affine_map<(d0, d1, d2, d3) -> (d0)>
#map1 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func.func @qdq_conv_per_channel
// CHECK-DAG:   %[[SCALE:.+]] = arith.constant 4.000000e-03 : f32
// CHECK-DAG:   %[[CS:.+]] = arith.constant dense<[2.500000e-01, 5.000000e-01, 7.500000e-01, 1.000000e+00, 2.500000e-01, {{.*}}]> : tensor<16xf32>
// CHECK:       %[[ACC:.+]] = linalg.conv_2d_nhwc_hwcf
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : tensor<1x4x4x8xi8>, tensor<1x1x8x16xi8>)
// CHECK:       linalg.generic
// CHECK-SAME:      ins(%[[ACC]], %{{.+}}, %[[CS]], %[[SCALE]], %{{.+}}, %{{.+}}, %{{.+}} : tensor<1x4x4x16xi32>, tensor<16xi32>, tensor<16xf32>, f32, i32, f32, f32)
// CHECK-NOT:   iree_linalg_ext.quantize_affine
func.func @qdq_conv_per_channel(%x: tensor<1x8x4x4xi8>) -> tensor<1x16x4x4xi8> {
  %wq = arith.constant dense<1> : tensor<16x8x1x1xi8>
  %sw = arith.constant dense<[4.000000e-03, 8.000000e-03, 1.200000e-02, 1.600000e-02, 4.000000e-03, 8.000000e-03, 1.200000e-02, 1.600000e-02, 4.000000e-03, 8.000000e-03, 1.200000e-02, 1.600000e-02, 4.000000e-03, 8.000000e-03, 1.200000e-02, 1.600000e-02]> : tensor<16xf32>
  %bq = arith.constant dense<100> : tensor<16xi32>
  %sb = arith.constant dense<[1.000000e-03, 2.000000e-03, 3.000000e-03, 4.000000e-03, 1.000000e-03, 2.000000e-03, 3.000000e-03, 4.000000e-03, 1.000000e-03, 2.000000e-03, 3.000000e-03, 4.000000e-03, 1.000000e-03, 2.000000e-03, 3.000000e-03, 4.000000e-03]> : tensor<16xf32>
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
  %e2 = tensor.empty() : tensor<16xf32>
  %bf = iree_linalg_ext.dequantize_affine <indexing_maps = [#map1, #map1, #map1], quant_min = -2147483648, quant_max = 2147483647>
      ins(%bq, %sb : tensor<16xi32>, tensor<16xf32>) outs(%e2 : tensor<16xf32>) -> tensor<16xf32>
  %e3 = tensor.empty() : tensor<1x16x4x4xf32>
  %bb = linalg.broadcast ins(%bf : tensor<16xf32>) outs(%e3 : tensor<1x16x4x4xf32>) dimensions = [0, 2, 3]
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

// One channel's factor is 1/400 of the largest: below what a lone 16-bit
// multiplier lane carries within a code, inside what the BS shift lane
// carries at 15 bits. The factor reaches the canonical form as it is.

#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#scalar4 = affine_map<(d0, d1, d2, d3) -> ()>
#chan4 = affine_map<(d0, d1, d2, d3) -> (d0)>

// CHECK-LABEL: func.func @qdq_conv_per_channel_small_factor
// CHECK:       arith.constant dense<[2.500000e-01, 5.000000e-01, 7.500000e-01, 0.00249999971, 2.500000e-01, {{.*}}]> : tensor<16xf32>
// CHECK:       linalg.conv_2d_nhwc_hwcf {{.*}} -> tensor<1x4x4x16xi32>
// CHECK-NOT:   iree_linalg_ext.quantize_affine
func.func @qdq_conv_per_channel_small_factor(%x: tensor<1x8x4x4xi8>) -> tensor<1x16x4x4xi8> {
  %wq = arith.constant dense<1> : tensor<16x8x1x1xi8>
  %sw = arith.constant dense<[4.000000e-03, 8.000000e-03, 1.200000e-02, 4.000000e-05, 4.000000e-03, 8.000000e-03, 1.200000e-02, 1.600000e-02, 4.000000e-03, 8.000000e-03, 1.200000e-02, 1.600000e-02, 4.000000e-03, 8.000000e-03, 1.200000e-02, 1.600000e-02]> : tensor<16xf32>
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

// Declined: one channel's factor is 1e-15 of the largest, below
// `kMinChannelScale` (2^-49), past what the BS shift lane reaches. Nor is
// the channel dead: its conv term reaches 1e-5 * 8 * 128 = 0.01 codes, over
// `kDeadChannelTerm` (1/256). The convolution keeps its float form.

#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#scalar4 = affine_map<(d0, d1, d2, d3) -> ()>
#chan4 = affine_map<(d0, d1, d2, d3) -> (d0)>

// CHECK-LABEL: func.func @qdq_conv_per_channel_tiny_factor_declined
// CHECK:       linalg.conv_2d_nhwc_hwcf {{.*}} -> tensor<1x4x4x16xf32>
// CHECK:       iree_linalg_ext.quantize_affine
func.func @qdq_conv_per_channel_tiny_factor_declined(%x: tensor<1x8x4x4xi8>) -> tensor<1x16x4x4xi8> {
  %wq = arith.constant dense<1> : tensor<16x8x1x1xi8>
  %sw = arith.constant dense<[4.000000e+10, 8.000000e+10, 1.200000e+10, 4.000000e-05, 4.000000e+10, 8.000000e+10, 1.200000e+10, 1.600000e+10, 4.000000e+10, 8.000000e+10, 1.200000e+10, 1.600000e+10, 4.000000e+10, 8.000000e+10, 1.200000e+10, 1.600000e+10]> : tensor<16xf32>
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

// A dead channel: channel 3's factor is 2.5e-19 of the largest, past what
// the BS shift lane reaches, but its whole conv term is under
// 1e-21 * 8 * 128 codes, so its output is its bias alone:
// round(1e-21 * 3000000) = 0 codes. The fold zeroes its filter and gives it
// a factor of 1 and a bias of 0. The other channels are as they were.

#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#scalar4 = affine_map<(d0, d1, d2, d3) -> ()>
#chan4 = affine_map<(d0, d1, d2, d3) -> (d0)>
#map1 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func.func @qdq_conv_per_channel_dead_channel_folded
// CHECK-DAG:   %[[SCALE:.+]] = arith.constant 4.000000e-03 : f32
// CHECK-DAG:   %[[CS:.+]] = arith.constant dense<[2.500000e-01, 5.000000e-01, 7.500000e-01, 1.000000e+00, 2.500000e-01, {{.*}}]> : tensor<16xf32>
// CHECK-DAG:   %[[BIAS:.+]] = arith.constant dense<[100, 100, 100, 0, 100, {{.*}}]> : tensor<16xi32>
// Channel 3 (bytes 24 to 31) zeroed, the rest the splat 1 they were:
// CHECK-DAG:   %[[W:.+]] = arith.constant dense<"0x0101010101010101010101010101010101010101010101010000000000000000010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101"> : tensor<16x8x1x1xi8>
// CHECK:       linalg.transpose ins(%[[W]]
// CHECK:       %[[ACC:.+]] = linalg.conv_2d_nhwc_hwcf
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : tensor<1x4x4x8xi8>, tensor<1x1x8x16xi8>)
// CHECK:       linalg.generic
// CHECK-SAME:      ins(%[[ACC]], %[[BIAS]], %[[CS]], %[[SCALE]], %{{.+}}, %{{.+}}, %{{.+}} : tensor<1x4x4x16xi32>, tensor<16xi32>, tensor<16xf32>, f32, i32, f32, f32)
// CHECK-NOT:   iree_linalg_ext.quantize_affine
func.func @qdq_conv_per_channel_dead_channel_folded(%x: tensor<1x8x4x4xi8>) -> tensor<1x16x4x4xi8> {
  %wq = arith.constant dense<1> : tensor<16x8x1x1xi8>
  %sw = arith.constant dense<[4.000000e-03, 8.000000e-03, 1.200000e-02, 4.000000e-21, 4.000000e-03, 8.000000e-03, 1.200000e-02, 1.600000e-02, 4.000000e-03, 8.000000e-03, 1.200000e-02, 1.600000e-02, 4.000000e-03, 8.000000e-03, 1.200000e-02, 1.600000e-02]> : tensor<16xf32>
  %bq = arith.constant dense<[100, 100, 100, 3000000, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]> : tensor<16xi32>
  %sb = arith.constant dense<[1.000000e-03, 2.000000e-03, 3.000000e-03, 1.000000e-21, 1.000000e-03, 2.000000e-03, 3.000000e-03, 4.000000e-03, 1.000000e-03, 2.000000e-03, 3.000000e-03, 4.000000e-03, 1.000000e-03, 2.000000e-03, 3.000000e-03, 4.000000e-03]> : tensor<16xf32>
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
  %e2 = tensor.empty() : tensor<16xf32>
  %bf = iree_linalg_ext.dequantize_affine <indexing_maps = [#map1, #map1, #map1], quant_min = -2147483648, quant_max = 2147483647>
      ins(%bq, %sb : tensor<16xi32>, tensor<16xf32>) outs(%e2 : tensor<16xf32>) -> tensor<16xf32>
  %e3 = tensor.empty() : tensor<1x16x4x4xf32>
  %bb = linalg.broadcast ins(%bf : tensor<16xf32>) outs(%e3 : tensor<1x16x4x4xf32>) dimensions = [0, 2, 3]
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

// -----

// A depthwise convolution, which the channels-last conversion leaves in the
// importer's NCHW form: `depthwise_conv_2d_nchw_chw` over an NCHW pad of the
// dequantized activation, a collapsed [C, 1, kh, kw] filter, and the bias
// broadcast in NCHW. It comes out NHWC: transpose, then a rank-4 NHWC pad
// filled with the input zero point, an HWC filter, and the same
// requantization as the dense case, transposed back.

#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#scalar4 = affine_map<(d0, d1, d2, d3) -> ()>
#map1 = affine_map<(d0) -> (d0)>
#scalar1 = affine_map<(d0) -> ()>

// CHECK-LABEL: func.func @qdq_depthwise_s2_pad1
// CHECK-DAG:   %[[FILL:.+]] = arith.constant -128 : i8
// CHECK:       %[[XT:.+]] = linalg.transpose ins(%{{.+}} : tensor<1x32x8x8xi8>)
// CHECK-SAME:      permutation = [0, 2, 3, 1]
// CHECK:       %[[XP:.+]] = tensor.pad %[[XT]] low[0, 1, 1, 0] high[0, 1, 1, 0]
// CHECK:         tensor.yield %[[FILL]] : i8
// CHECK-NEXT:  } : tensor<1x8x8x32xi8> to tensor<1x10x10x32xi8>
// CHECK:       %[[WC:.+]] = tensor.collapse_shape %{{.+}} {{\[}}[0, 1], [2], [3]] : tensor<32x1x3x3xi8> into tensor<32x3x3xi8>
// CHECK:       %[[WT:.+]] = linalg.transpose ins(%[[WC]] : tensor<32x3x3xi8>)
// CHECK-SAME:      permutation = [1, 2, 0]
// CHECK:       %[[ACC:.+]] = linalg.depthwise_conv_2d_nhwc_hwc
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>
// CHECK-SAME:      ins(%[[XP]], %[[WT]] : tensor<1x10x10x32xi8>, tensor<3x3x32xi8>)
// CHECK-SAME:      -> tensor<1x4x4x32xi32>
//   sum_k(w) over (kh, kw), per channel.
// CHECK:       %[[SUM:.+]] = linalg.generic {{.*}}iterator_types = ["reduction", "reduction", "parallel"]
// CHECK-SAME:      ins(%[[WT]] : tensor<3x3x32xi8>)
// CHECK:       %[[BIAS:.+]] = linalg.generic
// CHECK-SAME:      ins(%{{.+}}, %[[SUM]] : tensor<32xi32>, tensor<32xi32>)
// CHECK:       %[[Q:.+]] = linalg.generic
// CHECK-SAME:      ins(%[[ACC]], %[[BIAS]],
// CHECK:       %[[OUT:.+]] = linalg.transpose ins(%[[Q]] : tensor<1x4x4x32xi8>)
// CHECK-SAME:      permutation = [0, 3, 1, 2]
// CHECK:       return %[[OUT]] : tensor<1x32x4x4xi8>
// CHECK-NOT:   iree_linalg_ext.quantize_affine
func.func @qdq_depthwise_s2_pad1(%x: tensor<1x32x8x8xi8>) -> tensor<1x32x4x4xi8> {
  %wq = arith.constant dense<5> : tensor<32x1x3x3xi8>
  %bq = arith.constant dense<-300> : tensor<32xi32>
  %sx = arith.constant 2.000000e-02 : f32
  %sw = arith.constant 4.000000e-03 : f32
  %sb = arith.constant 8.000000e-05 : f32
  %sy = arith.constant 5.000000e-02 : f32
  %zx = arith.constant -128 : i64
  %zy = arith.constant -128 : i64
  %zero = arith.constant 0.000000e+00 : f32
  %e0 = tensor.empty() : tensor<1x32x8x8xf32>
  %xf = iree_linalg_ext.dequantize_affine <indexing_maps = [#map4, #scalar4, #scalar4, #map4], quant_min = -128, quant_max = 127>
      ins(%x, %sx, %zx : tensor<1x32x8x8xi8>, f32, i64) outs(%e0 : tensor<1x32x8x8xf32>) -> tensor<1x32x8x8xf32>
  %xp = tensor.pad %xf low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a: index, %b: index, %c: index, %d: index):
    tensor.yield %zero : f32
  } : tensor<1x32x8x8xf32> to tensor<1x32x10x10xf32>
  %e1 = tensor.empty() : tensor<32x1x3x3xf32>
  %wf = iree_linalg_ext.dequantize_affine <indexing_maps = [#map4, #scalar4, #map4], quant_min = -128, quant_max = 127>
      ins(%wq, %sw : tensor<32x1x3x3xi8>, f32) outs(%e1 : tensor<32x1x3x3xf32>) -> tensor<32x1x3x3xf32>
  %wc = tensor.collapse_shape %wf [[0, 1], [2], [3]] : tensor<32x1x3x3xf32> into tensor<32x3x3xf32>
  %e2 = tensor.empty() : tensor<32xf32>
  %bf = iree_linalg_ext.dequantize_affine <indexing_maps = [#map1, #scalar1, #map1], quant_min = -2147483648, quant_max = 2147483647>
      ins(%bq, %sb : tensor<32xi32>, f32) outs(%e2 : tensor<32xf32>) -> tensor<32xf32>
  %e3 = tensor.empty() : tensor<1x32x4x4xf32>
  %bb = linalg.broadcast ins(%bf : tensor<32xf32>) outs(%e3 : tensor<1x32x4x4xf32>) dimensions = [0, 2, 3]
  %acc = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
      ins(%xp, %wc : tensor<1x32x10x10xf32>, tensor<32x3x3xf32>) outs(%bb : tensor<1x32x4x4xf32>) -> tensor<1x32x4x4xf32>
  %e8 = tensor.empty() : tensor<1x32x4x4xi8>
  %q = iree_linalg_ext.quantize_affine <indexing_maps = [#map4, #scalar4, #scalar4, #map4], quant_min = -128, quant_max = 127>
      ins(%acc, %sy, %zy : tensor<1x32x4x4xf32>, f32, i64) outs(%e8 : tensor<1x32x4x4xi8>) -> tensor<1x32x4x4xi8>
  return %q : tensor<1x32x4x4xi8>
}

// -----

// The depthwise twin of @qdq_conv_per_channel: one weight scale per channel
// on the [C, 1, kh, kw] filter, which the collapse to [C, kh, kw] keeps on
// dimension 0, and a per-channel bias scale. Ratios 0.0016 to 0.0064, so
// the per-tensor scale is 0.0064 and the factors are 1/4 to 1.

#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#scalar4 = affine_map<(d0, d1, d2, d3) -> ()>
#map1 = affine_map<(d0) -> (d0)>
#scalar1 = affine_map<(d0) -> ()>
#chan4 = affine_map<(d0, d1, d2, d3) -> (d0)>

// CHECK-LABEL: func.func @qdq_depthwise_per_channel
// CHECK-DAG:   %[[SCALE:.+]] = arith.constant 6.400000e-03 : f32
// CHECK-DAG:   %[[CS:.+]] = arith.constant dense<[2.500000e-01, 5.000000e-01, 7.500000e-01, 1.000000e+00, {{.*}}]> : tensor<32xf32>
// CHECK:       %[[ACC:.+]] = linalg.depthwise_conv_2d_nhwc_hwc
// CHECK:       linalg.generic {{.*}}ins(%[[ACC]], %{{.+}}, %[[CS]], %[[SCALE]], %{{.+}}, %{{.+}}, %{{.+}} : tensor<1x4x4x32xi32>, tensor<32xi32>, tensor<32xf32>, f32, i32, f32, f32)
// CHECK-NOT:   iree_linalg_ext.quantize_affine
func.func @qdq_depthwise_per_channel(%x: tensor<1x32x8x8xi8>) -> tensor<1x32x4x4xi8> {
  %wq = arith.constant dense<5> : tensor<32x1x3x3xi8>
  %bq = arith.constant dense<-300> : tensor<32xi32>
  %sx = arith.constant 2.000000e-02 : f32
  %sw = arith.constant dense<[4.000000e-03, 8.000000e-03, 1.200000e-02, 1.600000e-02, 4.000000e-03, 8.000000e-03, 1.200000e-02, 1.600000e-02, 4.000000e-03, 8.000000e-03, 1.200000e-02, 1.600000e-02, 4.000000e-03, 8.000000e-03, 1.200000e-02, 1.600000e-02, 4.000000e-03, 8.000000e-03, 1.200000e-02, 1.600000e-02, 4.000000e-03, 8.000000e-03, 1.200000e-02, 1.600000e-02, 4.000000e-03, 8.000000e-03, 1.200000e-02, 1.600000e-02, 4.000000e-03, 8.000000e-03, 1.200000e-02, 1.600000e-02]> : tensor<32xf32>
  %sb = arith.constant dense<[8.000000e-05, 1.600000e-04, 2.400000e-04, 3.200000e-04, 8.000000e-05, 1.600000e-04, 2.400000e-04, 3.200000e-04, 8.000000e-05, 1.600000e-04, 2.400000e-04, 3.200000e-04, 8.000000e-05, 1.600000e-04, 2.400000e-04, 3.200000e-04, 8.000000e-05, 1.600000e-04, 2.400000e-04, 3.200000e-04, 8.000000e-05, 1.600000e-04, 2.400000e-04, 3.200000e-04, 8.000000e-05, 1.600000e-04, 2.400000e-04, 3.200000e-04, 8.000000e-05, 1.600000e-04, 2.400000e-04, 3.200000e-04]> : tensor<32xf32>
  %sy = arith.constant 5.000000e-02 : f32
  %zx = arith.constant -128 : i64
  %zy = arith.constant -128 : i64
  %zero = arith.constant 0.000000e+00 : f32
  %e0 = tensor.empty() : tensor<1x32x8x8xf32>
  %xf = iree_linalg_ext.dequantize_affine <indexing_maps = [#map4, #scalar4, #scalar4, #map4], quant_min = -128, quant_max = 127>
      ins(%x, %sx, %zx : tensor<1x32x8x8xi8>, f32, i64) outs(%e0 : tensor<1x32x8x8xf32>) -> tensor<1x32x8x8xf32>
  %xp = tensor.pad %xf low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a: index, %b: index, %c: index, %d: index):
    tensor.yield %zero : f32
  } : tensor<1x32x8x8xf32> to tensor<1x32x10x10xf32>
  %e1 = tensor.empty() : tensor<32x1x3x3xf32>
  %wf = iree_linalg_ext.dequantize_affine <indexing_maps = [#map4, #chan4, #map4], quant_min = -128, quant_max = 127>
      ins(%wq, %sw : tensor<32x1x3x3xi8>, tensor<32xf32>) outs(%e1 : tensor<32x1x3x3xf32>) -> tensor<32x1x3x3xf32>
  %wc = tensor.collapse_shape %wf [[0, 1], [2], [3]] : tensor<32x1x3x3xf32> into tensor<32x3x3xf32>
  %e2 = tensor.empty() : tensor<32xf32>
  %bf = iree_linalg_ext.dequantize_affine <indexing_maps = [#map1, #map1, #map1], quant_min = -2147483648, quant_max = 2147483647>
      ins(%bq, %sb : tensor<32xi32>, tensor<32xf32>) outs(%e2 : tensor<32xf32>) -> tensor<32xf32>
  %e3 = tensor.empty() : tensor<1x32x4x4xf32>
  %bb = linalg.broadcast ins(%bf : tensor<32xf32>) outs(%e3 : tensor<1x32x4x4xf32>) dimensions = [0, 2, 3]
  %acc = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
      ins(%xp, %wc : tensor<1x32x10x10xf32>, tensor<32x3x3xf32>) outs(%bb : tensor<1x32x4x4xf32>) -> tensor<1x32x4x4xf32>
  %e8 = tensor.empty() : tensor<1x32x4x4xi8>
  %q = iree_linalg_ext.quantize_affine <indexing_maps = [#map4, #scalar4, #scalar4, #map4], quant_min = -128, quant_max = 127>
      ins(%acc, %sy, %zy : tensor<1x32x4x4xf32>, f32, i64) outs(%e8 : tensor<1x32x4x4xi8>) -> tensor<1x32x4x4xi8>
  return %q : tensor<1x32x4x4xi8>
}

// -----

// The depthwise fold, on @qdq_depthwise_per_channel with channel 1's weight
// scale cut to 8e-24: a factor of 5e-22, past the BS shift lane's reach,
// but a dead channel, its conv term at most 3.2e-24 * 45 * 255 codes (the
// input zero point -128 lets |x - zx| reach 255). Its bias, -625000000 at a
// scale of 1.6e-25, is 0 codes, so the fold zeroes its taps (bytes 9 to 17
// of the [C, 1, 3, 3] filter) and gives it a bias of 0 and a factor of 1.

#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#scalar4 = affine_map<(d0, d1, d2, d3) -> ()>
#map1 = affine_map<(d0) -> (d0)>
#scalar1 = affine_map<(d0) -> ()>
#chan4 = affine_map<(d0, d1, d2, d3) -> (d0)>

// CHECK-LABEL: func.func @qdq_depthwise_dead_channel_folded
// CHECK-DAG:   %[[SCALE:.+]] = arith.constant 6.400000e-03 : f32
// CHECK-DAG:   %[[CS:.+]] = arith.constant dense<[2.500000e-01, 1.000000e+00, 7.500000e-01, 1.000000e+00, {{.*}}]> : tensor<32xf32>
// CHECK-DAG:   %[[W:.+]] = arith.constant dense<"0x050505050505050505000000000000000000050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505"> : tensor<32x1x3x3xi8>
// CHECK-DAG:   %[[B:.+]] = arith.constant dense<[-300, 0, -300, -300, {{.*}}]> : tensor<32xi32>
// CHECK:       tensor.collapse_shape %[[W]]
// CHECK:       %[[ACC:.+]] = linalg.depthwise_conv_2d_nhwc_hwc
// CHECK:       linalg.generic {{.*}}ins(%[[B]], %{{.+}} : tensor<32xi32>, tensor<32xi32>)
// CHECK:       linalg.generic {{.*}}ins(%[[ACC]], %{{.+}}, %[[CS]], %[[SCALE]], %{{.+}}, %{{.+}}, %{{.+}} : tensor<1x4x4x32xi32>, tensor<32xi32>, tensor<32xf32>, f32, i32, f32, f32)
// CHECK-NOT:   iree_linalg_ext.quantize_affine
func.func @qdq_depthwise_dead_channel_folded(%x: tensor<1x32x8x8xi8>) -> tensor<1x32x4x4xi8> {
  %wq = arith.constant dense<5> : tensor<32x1x3x3xi8>
  %bq = arith.constant dense<[-300, -625000000, -300, -300, -300, -300, -300, -300, -300, -300, -300, -300, -300, -300, -300, -300, -300, -300, -300, -300, -300, -300, -300, -300, -300, -300, -300, -300, -300, -300, -300, -300]> : tensor<32xi32>
  %sx = arith.constant 2.000000e-02 : f32
  %sw = arith.constant dense<[4.000000e-03, 8.000000e-24, 1.200000e-02, 1.600000e-02, 4.000000e-03, 8.000000e-03, 1.200000e-02, 1.600000e-02, 4.000000e-03, 8.000000e-03, 1.200000e-02, 1.600000e-02, 4.000000e-03, 8.000000e-03, 1.200000e-02, 1.600000e-02, 4.000000e-03, 8.000000e-03, 1.200000e-02, 1.600000e-02, 4.000000e-03, 8.000000e-03, 1.200000e-02, 1.600000e-02, 4.000000e-03, 8.000000e-03, 1.200000e-02, 1.600000e-02, 4.000000e-03, 8.000000e-03, 1.200000e-02, 1.600000e-02]> : tensor<32xf32>
  %sb = arith.constant dense<[8.000000e-05, 1.600000e-25, 2.400000e-04, 3.200000e-04, 8.000000e-05, 1.600000e-04, 2.400000e-04, 3.200000e-04, 8.000000e-05, 1.600000e-04, 2.400000e-04, 3.200000e-04, 8.000000e-05, 1.600000e-04, 2.400000e-04, 3.200000e-04, 8.000000e-05, 1.600000e-04, 2.400000e-04, 3.200000e-04, 8.000000e-05, 1.600000e-04, 2.400000e-04, 3.200000e-04, 8.000000e-05, 1.600000e-04, 2.400000e-04, 3.200000e-04, 8.000000e-05, 1.600000e-04, 2.400000e-04, 3.200000e-04]> : tensor<32xf32>
  %sy = arith.constant 5.000000e-02 : f32
  %zx = arith.constant -128 : i64
  %zy = arith.constant -128 : i64
  %zero = arith.constant 0.000000e+00 : f32
  %e0 = tensor.empty() : tensor<1x32x8x8xf32>
  %xf = iree_linalg_ext.dequantize_affine <indexing_maps = [#map4, #scalar4, #scalar4, #map4], quant_min = -128, quant_max = 127>
      ins(%x, %sx, %zx : tensor<1x32x8x8xi8>, f32, i64) outs(%e0 : tensor<1x32x8x8xf32>) -> tensor<1x32x8x8xf32>
  %xp = tensor.pad %xf low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a: index, %b: index, %c: index, %d: index):
    tensor.yield %zero : f32
  } : tensor<1x32x8x8xf32> to tensor<1x32x10x10xf32>
  %e1 = tensor.empty() : tensor<32x1x3x3xf32>
  %wf = iree_linalg_ext.dequantize_affine <indexing_maps = [#map4, #chan4, #map4], quant_min = -128, quant_max = 127>
      ins(%wq, %sw : tensor<32x1x3x3xi8>, tensor<32xf32>) outs(%e1 : tensor<32x1x3x3xf32>) -> tensor<32x1x3x3xf32>
  %wc = tensor.collapse_shape %wf [[0, 1], [2], [3]] : tensor<32x1x3x3xf32> into tensor<32x3x3xf32>
  %e2 = tensor.empty() : tensor<32xf32>
  %bf = iree_linalg_ext.dequantize_affine <indexing_maps = [#map1, #map1, #map1], quant_min = -2147483648, quant_max = 2147483647>
      ins(%bq, %sb : tensor<32xi32>, tensor<32xf32>) outs(%e2 : tensor<32xf32>) -> tensor<32xf32>
  %e3 = tensor.empty() : tensor<1x32x4x4xf32>
  %bb = linalg.broadcast ins(%bf : tensor<32xf32>) outs(%e3 : tensor<1x32x4x4xf32>) dimensions = [0, 2, 3]
  %acc = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
      ins(%xp, %wc : tensor<1x32x10x10xf32>, tensor<32x3x3xf32>) outs(%bb : tensor<1x32x4x4xf32>) -> tensor<1x32x4x4xf32>
  %e8 = tensor.empty() : tensor<1x32x4x4xi8>
  %q = iree_linalg_ext.quantize_affine <indexing_maps = [#map4, #scalar4, #scalar4, #map4], quant_min = -128, quant_max = 127>
      ins(%acc, %sy, %zy : tensor<1x32x4x4xf32>, f32, i64) outs(%e8 : tensor<1x32x4x4xi8>) -> tensor<1x32x4x4xi8>
  return %q : tensor<1x32x4x4xi8>
}
