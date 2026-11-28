// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(rocket-fold-constant-dequant))" %s | FileCheck %s
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(rocket-fold-constant-dequant{offload=false}))" %s | FileCheck %s --check-prefix=NOOFFLOAD

// A QDQ Gemm as IREE 3.14 has it after input conversion, copied in shape
// from stock `quantize_static` ResNet50's FC layer: a symmetric i8 weight
// with a scalar scale (ORT's zero point 0 is dropped by the importer),
// transposed into the matmul's B operand. The dequantization becomes the
// f32 constant it computes; the transpose and the matmul are untouched.

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#scalar2 = affine_map<(d0, d1) -> ()>

// CHECK-LABEL: func.func @gemm_weight
// CHECK-NOT:   iree_linalg_ext.dequantize_affine
// CHECK:       %[[W:.+]] = arith.constant dense<{{\[}}[5.000000e-01, -1.000000e+00], [1.500000e+00, 6.350000e+01], [-6.400000e+01, 0.000000e+00], [2.500000e+00, 3.000000e+00]]> : tensor<4x2xf32>
// CHECK:       %[[WT:.+]] = linalg.transpose ins(%[[W]] : tensor<4x2xf32>)
// CHECK:       linalg.matmul ins(%{{.+}}, %[[WT]] : tensor<1x2xf32>, tensor<2x4xf32>)
//   `rocket-compiler --no-offload` turns the fold off: the CPU baseline
//   keeps the dequantization, which its matmul fuses.
// NOOFFLOAD-LABEL: func.func @gemm_weight
// NOOFFLOAD:       iree_linalg_ext.dequantize_affine
// NOOFFLOAD:       linalg.matmul
func.func @gemm_weight(%x: tensor<1x2xf32>) -> tensor<1x4xf32> {
  %codes = arith.constant dense<[[1, -2], [3, 127], [-128, 0], [5, 6]]> : tensor<4x2xi8>
  %scale = arith.constant 5.000000e-01 : f32
  %zero = arith.constant 0.000000e+00 : f32
  %e0 = tensor.empty() : tensor<4x2xf32>
  %w = iree_linalg_ext.dequantize_affine <indexing_maps = [#id2, #scalar2, #id2], quant_min = -128, quant_max = 127> ins(%codes, %scale : tensor<4x2xi8>, f32) outs(%e0 : tensor<4x2xf32>) -> tensor<4x2xf32>
  %e1 = tensor.empty() : tensor<2x4xf32>
  %wt = linalg.transpose ins(%w : tensor<4x2xf32>) outs(%e1 : tensor<2x4xf32>) permutation = [1, 0]
  %e2 = tensor.empty() : tensor<1x4xf32>
  %acc = linalg.fill ins(%zero : f32) outs(%e2 : tensor<1x4xf32>) -> tensor<1x4xf32>
  %y = linalg.matmul ins(%x, %wt : tensor<1x2xf32>, tensor<2x4xf32>) outs(%acc : tensor<1x4xf32>) -> tensor<1x4xf32>
  return %y : tensor<1x4xf32>
}

// -----

// The weight as a real model stores it, a `dense_resource` blob (i8 1, -2,
// 3, -128 after the 4-byte alignment prefix), straight into B.

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#scalar2 = affine_map<(d0, d1) -> ()>

// CHECK-LABEL: func.func @resource_weight
// CHECK-NOT:   iree_linalg_ext.dequantize_affine
// CHECK:       %[[W:.+]] = arith.constant dense<{{\[}}[2.000000e+00, -4.000000e+00], [6.000000e+00, -2.560000e+02]]> : tensor<2x2xf32>
// CHECK:       linalg.matmul ins(%{{.+}}, %[[W]] : tensor<1x2xf32>, tensor<2x2xf32>)
func.func @resource_weight(%x: tensor<1x2xf32>, %acc: tensor<1x2xf32>) -> tensor<1x2xf32> {
  %codes = arith.constant dense_resource<fc_weight> : tensor<2x2xi8>
  %scale = arith.constant 2.000000e+00 : f32
  %e0 = tensor.empty() : tensor<2x2xf32>
  %w = iree_linalg_ext.dequantize_affine <indexing_maps = [#id2, #scalar2, #id2]> ins(%codes, %scale : tensor<2x2xi8>, f32) outs(%e0 : tensor<2x2xf32>) -> tensor<2x2xf32>
  %y = linalg.matmul ins(%x, %w : tensor<1x2xf32>, tensor<2x2xf32>) outs(%acc : tensor<1x2xf32>) -> tensor<1x2xf32>
  return %y : tensor<1x2xf32>
}

{-#
  dialect_resources: {
    builtin: {
      fc_weight: "0x0400000001FE0380"
    }
  }
#-}

// -----

// Asymmetric unsigned codes: `input_unsigned` reads -1 as 255 and -128 as
// 128, `zp_unsigned` reads the zero point -128 as 128, and the difference
// is taken in i16 before the scale: 127, -128, 0, -118 times 0.25.

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#scalar2 = affine_map<(d0, d1) -> ()>

// CHECK-LABEL: func.func @unsigned_zero_point
// CHECK-NOT:   iree_linalg_ext.dequantize_affine
// CHECK:       %[[W:.+]] = arith.constant dense<{{\[}}[3.175000e+01, -3.200000e+01], [0.000000e+00, -2.950000e+01]]> : tensor<2x2xf32>
// CHECK:       linalg.matmul ins(%{{.+}}, %[[W]] : tensor<1x2xf32>, tensor<2x2xf32>)
func.func @unsigned_zero_point(%x: tensor<1x2xf32>, %acc: tensor<1x2xf32>) -> tensor<1x2xf32> {
  %codes = arith.constant dense<[[-1, 0], [-128, 10]]> : tensor<2x2xi8>
  %scale = arith.constant 2.500000e-01 : f32
  %zp = arith.constant -128 : i8
  %e0 = tensor.empty() : tensor<2x2xf32>
  %w = iree_linalg_ext.dequantize_affine <indexing_maps = [#id2, #scalar2, #scalar2, #id2], input_unsigned, zp_unsigned> ins(%codes, %scale, %zp : tensor<2x2xi8>, f32, i8) outs(%e0 : tensor<2x2xf32>) -> tensor<2x2xf32>
  %y = linalg.matmul ins(%x, %w : tensor<1x2xf32>, tensor<2x2xf32>) outs(%acc : tensor<1x2xf32>) -> tensor<1x2xf32>
  return %y : tensor<1x2xf32>
}

// -----

// A per-channel convolution filter -- the case `rocket-fuse-qdq-conv`
// declines -- after the channels-last conversion: FCHW codes with one scale
// per output channel, transposed to HWCF. Scales 1, 0.5, 0.25 by channel.

#id4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#channel4 = affine_map<(d0, d1, d2, d3) -> (d0)>

// CHECK-LABEL: func.func @per_channel_conv_filter
// CHECK-NOT:   iree_linalg_ext.dequantize_affine
// CHECK:       %[[W:.+]] = arith.constant dense<{{\[}}{{\[}}{{\[}}[1.000000e+00]], {{\[}}[-1.000000e+00]]], {{\[}}{{\[}}[1.000000e+00]], {{\[}}[-1.000000e+00]]], {{\[}}{{\[}}[7.500000e-01]], {{\[}}[-7.500000e-01]]]]> : tensor<3x2x1x1xf32>
// CHECK:       %[[WT:.+]] = linalg.transpose ins(%[[W]] : tensor<3x2x1x1xf32>)
// CHECK:       linalg.conv_2d_nhwc_hwcf
// CHECK-SAME:      ins(%{{.+}}, %[[WT]] : tensor<1x4x4x2xf32>, tensor<1x1x2x3xf32>)
func.func @per_channel_conv_filter(%x: tensor<1x4x4x2xf32>, %acc: tensor<1x4x4x3xf32>) -> tensor<1x4x4x3xf32> {
  %codes = arith.constant dense<[[[[1]], [[-1]]], [[[2]], [[-2]]], [[[3]], [[-3]]]]> : tensor<3x2x1x1xi8>
  %scale = arith.constant dense<[1.000000e+00, 5.000000e-01, 2.500000e-01]> : tensor<3xf32>
  %e0 = tensor.empty() : tensor<3x2x1x1xf32>
  %w = iree_linalg_ext.dequantize_affine <indexing_maps = [#id4, #channel4, #id4]> ins(%codes, %scale : tensor<3x2x1x1xi8>, tensor<3xf32>) outs(%e0 : tensor<3x2x1x1xf32>) -> tensor<3x2x1x1xf32>
  %e1 = tensor.empty() : tensor<1x1x2x3xf32>
  %wt = linalg.transpose ins(%w : tensor<3x2x1x1xf32>) outs(%e1 : tensor<1x1x2x3xf32>) permutation = [2, 3, 1, 0]
  %y = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%x, %wt : tensor<1x4x4x2xf32>, tensor<1x1x2x3xf32>) outs(%acc : tensor<1x4x4x3xf32>) -> tensor<1x4x4x3xf32>
  return %y : tensor<1x4x4x3xf32>
}

// -----

// An f16 result from an f32 scale: the product is rounded to f32 first and
// then to f16, as the decomposition's mulf + truncf do. 3 * 0.1f is
// 0.300000012 in f32, 0.30005 in f16 (and -0.1 is -0.099976).

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#scalar2 = affine_map<(d0, d1) -> ()>

// CHECK-LABEL: func.func @f16_result
// CHECK:       %[[W:.+]] = arith.constant dense<{{\[}}[3.000490e-01, -9.997550e-02]]> : tensor<1x2xf16>
// CHECK:       linalg.matmul ins(%{{.+}}, %[[W]] : tensor<1x1xf16>, tensor<1x2xf16>)
func.func @f16_result(%x: tensor<1x1xf16>, %acc: tensor<1x2xf32>) -> tensor<1x2xf32> {
  %codes = arith.constant dense<[[3, -1]]> : tensor<1x2xi8>
  %scale = arith.constant 1.000000e-01 : f32
  %e0 = tensor.empty() : tensor<1x2xf16>
  %w = iree_linalg_ext.dequantize_affine <indexing_maps = [#id2, #scalar2, #id2]> ins(%codes, %scale : tensor<1x2xi8>, f32) outs(%e0 : tensor<1x2xf16>) -> tensor<1x2xf16>
  %y = linalg.matmul ins(%x, %w : tensor<1x1xf16>, tensor<1x2xf16>) outs(%acc : tensor<1x2xf32>) -> tensor<1x2xf32>
  return %y : tensor<1x2xf32>
}

// -----

// Declined: the dequantized constant is the matmul's A operand, not its
// weight.

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#scalar2 = affine_map<(d0, d1) -> ()>

// CHECK-LABEL: func.func @decline_lhs
// CHECK:       iree_linalg_ext.dequantize_affine
func.func @decline_lhs(%b: tensor<2x2xf32>, %acc: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %codes = arith.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi8>
  %scale = arith.constant 5.000000e-01 : f32
  %e0 = tensor.empty() : tensor<2x2xf32>
  %a = iree_linalg_ext.dequantize_affine <indexing_maps = [#id2, #scalar2, #id2]> ins(%codes, %scale : tensor<2x2xi8>, f32) outs(%e0 : tensor<2x2xf32>) -> tensor<2x2xf32>
  %y = linalg.matmul ins(%a, %b : tensor<2x2xf32>, tensor<2x2xf32>) outs(%acc : tensor<2x2xf32>) -> tensor<2x2xf32>
  return %y : tensor<2x2xf32>
}

// -----

// Declined: besides the weight, the dequantized tensor has a use that is
// not one (here it is also returned), so folding it would only widen that
// use's constant.

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#scalar2 = affine_map<(d0, d1) -> ()>

// CHECK-LABEL: func.func @decline_other_use
// CHECK:       iree_linalg_ext.dequantize_affine
func.func @decline_other_use(%x: tensor<1x2xf32>, %acc: tensor<1x2xf32>) -> (tensor<1x2xf32>, tensor<2x2xf32>) {
  %codes = arith.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi8>
  %scale = arith.constant 5.000000e-01 : f32
  %e0 = tensor.empty() : tensor<2x2xf32>
  %w = iree_linalg_ext.dequantize_affine <indexing_maps = [#id2, #scalar2, #id2]> ins(%codes, %scale : tensor<2x2xi8>, f32) outs(%e0 : tensor<2x2xf32>) -> tensor<2x2xf32>
  %y = linalg.matmul ins(%x, %w : tensor<1x2xf32>, tensor<2x2xf32>) outs(%acc : tensor<1x2xf32>) -> tensor<1x2xf32>
  return %y, %w : tensor<1x2xf32>, tensor<2x2xf32>
}

// -----

// Declined: the codes are not a constant (a dynamically quantized weight).

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#scalar2 = affine_map<(d0, d1) -> ()>

// CHECK-LABEL: func.func @decline_non_constant
// CHECK:       iree_linalg_ext.dequantize_affine
func.func @decline_non_constant(%codes: tensor<2x2xi8>, %x: tensor<1x2xf32>, %acc: tensor<1x2xf32>) -> tensor<1x2xf32> {
  %scale = arith.constant 5.000000e-01 : f32
  %e0 = tensor.empty() : tensor<2x2xf32>
  %w = iree_linalg_ext.dequantize_affine <indexing_maps = [#id2, #scalar2, #id2]> ins(%codes, %scale : tensor<2x2xi8>, f32) outs(%e0 : tensor<2x2xf32>) -> tensor<2x2xf32>
  %y = linalg.matmul ins(%x, %w : tensor<1x2xf32>, tensor<2x2xf32>) outs(%acc : tensor<1x2xf32>) -> tensor<1x2xf32>
  return %y : tensor<1x2xf32>
}
