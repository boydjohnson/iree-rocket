// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(rocket-fuse-int8-global-avg-pool))" %s | FileCheck %s

// ONNX GlobalAveragePool between a DequantizeLinear and a QuantizeLinear, as
// stock `quantize_static` ResNet50 has it before its classifier: the codes
// arrive as the NCHW transpose of an NHWC producer, then a 7x7 sum pool and
// a divide by 49. The rewrite sums the codes in i32 straight off the NHWC
// tensor (no transpose survives), subtracts `49 * zx = -6272`, scales by
// `sx / (49 * sy) = 0.04 / 0.98`, and requantizes once.

#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#scalar4 = affine_map<(d0, d1, d2, d3) -> ()>

// CHECK-LABEL: func.func @qdq_global_avg_pool
// CHECK-SAME:      %[[X:.+]]: tensor<1x7x7x16xi8>
// CHECK-DAG:   %[[OFFSET:.+]] = arith.constant -6272 : i32
// CHECK-DAG:   %[[RATIO:.+]] = arith.constant 0.0408163{{[0-9]*}} : f32
// CHECK-DAG:   %[[ZP:.+]] = arith.constant -1.280000e+02 : f32
// CHECK-NOT:   linalg.transpose
// CHECK-NOT:   pooling_nchw_sum
// CHECK:       %[[SUM:.+]] = linalg.generic
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "reduction", "reduction"]
// CHECK-SAME:      ins(%[[X]] : tensor<1x7x7x16xi8>) outs(%{{.+}} : tensor<1x16xi32>)
// CHECK:         arith.extsi
// CHECK:         arith.addi
// CHECK:       %[[Q:.+]] = linalg.generic {{.*}} ins(%[[SUM]] : tensor<1x16xi32>) outs(%{{.+}} : tensor<1x16xi8>)
// CHECK:         %[[C:.+]] = arith.subi %{{.+}}, %[[OFFSET]] : i32
// CHECK:         %[[R:.+]] = arith.sitofp %[[C]]
// CHECK:         %[[S:.+]] = arith.mulf %[[R]], %[[RATIO]] : f32
// CHECK:         %[[E:.+]] = math.roundeven %[[S]]
// CHECK:         arith.addf %[[E]], %[[ZP]] : f32
// CHECK:       %[[OUT:.+]] = tensor.expand_shape %[[Q]] {{\[}}[0], [1, 2, 3]] output_shape [1, 16, 1, 1] : tensor<1x16xi8> into tensor<1x16x1x1xi8>
// CHECK-NOT:   iree_linalg_ext.quantize_affine
// CHECK:       return %[[OUT]]
func.func @qdq_global_avg_pool(%x: tensor<1x7x7x16xi8>) -> tensor<1x16x1x1xi8> {
  %sx = arith.constant 4.000000e-02 : f32
  %zx = arith.constant -128 : i64
  %sy = arith.constant 2.000000e-02 : f32
  %zy = arith.constant -128 : i64
  %zero = arith.constant 0.000000e+00 : f32
  %k = arith.constant 4.900000e+01 : f32
  %et = tensor.empty() : tensor<1x16x7x7xi8>
  %nchw = linalg.transpose ins(%x : tensor<1x7x7x16xi8>) outs(%et : tensor<1x16x7x7xi8>) permutation = [0, 3, 1, 2]
  %e0 = tensor.empty() : tensor<1x16x7x7xf32>
  %xf = iree_linalg_ext.dequantize_affine <indexing_maps = [#map4, #scalar4, #scalar4, #map4], quant_min = -128, quant_max = 127>
      ins(%nchw, %sx, %zx : tensor<1x16x7x7xi8>, f32, i64) outs(%e0 : tensor<1x16x7x7xf32>) -> tensor<1x16x7x7xf32>
  %e1 = tensor.empty() : tensor<1x16x1x1xf32>
  %init = linalg.fill ins(%zero : f32) outs(%e1 : tensor<1x16x1x1xf32>) -> tensor<1x16x1x1xf32>
  %window = tensor.empty() : tensor<7x7xf32>
  %sum = linalg.pooling_nchw_sum {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%xf, %window : tensor<1x16x7x7xf32>, tensor<7x7xf32>) outs(%init : tensor<1x16x1x1xf32>) -> tensor<1x16x1x1xf32>
  %mean = linalg.elementwise <div> indexing_maps = [#map4, #scalar4, #map4] ins(%sum, %k : tensor<1x16x1x1xf32>, f32) outs(%e1 : tensor<1x16x1x1xf32>) -> tensor<1x16x1x1xf32>
  %e2 = tensor.empty() : tensor<1x16x1x1xi8>
  %q = iree_linalg_ext.quantize_affine <indexing_maps = [#map4, #scalar4, #scalar4, #map4], quant_min = -128, quant_max = 127>
      ins(%mean, %sy, %zy : tensor<1x16x1x1xf32>, f32, i64) outs(%e2 : tensor<1x16x1x1xi8>) -> tensor<1x16x1x1xi8>
  return %q : tensor<1x16x1x1xi8>
}

// -----

// Declined: a 3x3 window over a 7x7 image is a pool with spatial output,
// not a global one, and keeps its float form.

#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#scalar4 = affine_map<(d0, d1, d2, d3) -> ()>

// CHECK-LABEL: func.func @qdq_windowed_avg_pool_declined
// CHECK:       linalg.pooling_nchw_sum
// CHECK:       iree_linalg_ext.quantize_affine
func.func @qdq_windowed_avg_pool_declined(%x: tensor<1x16x7x7xi8>) -> tensor<1x16x5x5xi8> {
  %sx = arith.constant 4.000000e-02 : f32
  %zx = arith.constant -128 : i64
  %sy = arith.constant 2.000000e-02 : f32
  %zy = arith.constant -128 : i64
  %zero = arith.constant 0.000000e+00 : f32
  %k = arith.constant 9.000000e+00 : f32
  %e0 = tensor.empty() : tensor<1x16x7x7xf32>
  %xf = iree_linalg_ext.dequantize_affine <indexing_maps = [#map4, #scalar4, #scalar4, #map4], quant_min = -128, quant_max = 127>
      ins(%x, %sx, %zx : tensor<1x16x7x7xi8>, f32, i64) outs(%e0 : tensor<1x16x7x7xf32>) -> tensor<1x16x7x7xf32>
  %e1 = tensor.empty() : tensor<1x16x5x5xf32>
  %init = linalg.fill ins(%zero : f32) outs(%e1 : tensor<1x16x5x5xf32>) -> tensor<1x16x5x5xf32>
  %window = tensor.empty() : tensor<3x3xf32>
  %sum = linalg.pooling_nchw_sum {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%xf, %window : tensor<1x16x7x7xf32>, tensor<3x3xf32>) outs(%init : tensor<1x16x5x5xf32>) -> tensor<1x16x5x5xf32>
  %mean = linalg.elementwise <div> indexing_maps = [#map4, #scalar4, #map4] ins(%sum, %k : tensor<1x16x5x5xf32>, f32) outs(%e1 : tensor<1x16x5x5xf32>) -> tensor<1x16x5x5xf32>
  %e2 = tensor.empty() : tensor<1x16x5x5xi8>
  %q = iree_linalg_ext.quantize_affine <indexing_maps = [#map4, #scalar4, #scalar4, #map4], quant_min = -128, quant_max = 127>
      ins(%mean, %sy, %zy : tensor<1x16x5x5xf32>, f32, i64) outs(%e2 : tensor<1x16x5x5xi8>) -> tensor<1x16x5x5xi8>
  return %q : tensor<1x16x5x5xi8>
}

// -----

// Declined: a global sum divided by something other than the window size
// is not an average.

#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#scalar4 = affine_map<(d0, d1, d2, d3) -> ()>

// CHECK-LABEL: func.func @qdq_global_sum_wrong_divisor_declined
// CHECK:       linalg.pooling_nchw_sum
// CHECK:       iree_linalg_ext.quantize_affine
func.func @qdq_global_sum_wrong_divisor_declined(%x: tensor<1x16x7x7xi8>) -> tensor<1x16x1x1xi8> {
  %sx = arith.constant 4.000000e-02 : f32
  %zx = arith.constant -128 : i64
  %sy = arith.constant 2.000000e-02 : f32
  %zy = arith.constant -128 : i64
  %zero = arith.constant 0.000000e+00 : f32
  %k = arith.constant 4.800000e+01 : f32
  %e0 = tensor.empty() : tensor<1x16x7x7xf32>
  %xf = iree_linalg_ext.dequantize_affine <indexing_maps = [#map4, #scalar4, #scalar4, #map4], quant_min = -128, quant_max = 127>
      ins(%x, %sx, %zx : tensor<1x16x7x7xi8>, f32, i64) outs(%e0 : tensor<1x16x7x7xf32>) -> tensor<1x16x7x7xf32>
  %e1 = tensor.empty() : tensor<1x16x1x1xf32>
  %init = linalg.fill ins(%zero : f32) outs(%e1 : tensor<1x16x1x1xf32>) -> tensor<1x16x1x1xf32>
  %window = tensor.empty() : tensor<7x7xf32>
  %sum = linalg.pooling_nchw_sum {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%xf, %window : tensor<1x16x7x7xf32>, tensor<7x7xf32>) outs(%init : tensor<1x16x1x1xf32>) -> tensor<1x16x1x1xf32>
  %mean = linalg.elementwise <div> indexing_maps = [#map4, #scalar4, #map4] ins(%sum, %k : tensor<1x16x1x1xf32>, f32) outs(%e1 : tensor<1x16x1x1xf32>) -> tensor<1x16x1x1xf32>
  %e2 = tensor.empty() : tensor<1x16x1x1xi8>
  %q = iree_linalg_ext.quantize_affine <indexing_maps = [#map4, #scalar4, #scalar4, #map4], quant_min = -128, quant_max = 127>
      ins(%mean, %sy, %zy : tensor<1x16x1x1xf32>, f32, i64) outs(%e2 : tensor<1x16x1x1xi8>) -> tensor<1x16x1x1xi8>
  return %q : tensor<1x16x1x1xi8>
}
