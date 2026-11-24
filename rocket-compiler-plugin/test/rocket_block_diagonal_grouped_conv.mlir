// RUN: iree-opt %s --pass-pipeline='builtin.module(util.func(rocket-block-diagonal-grouped-conv))' | FileCheck %s
// RUN: iree-opt %s --pass-pipeline='builtin.module(util.func(rocket-block-diagonal-grouped-conv{offload=false}))' | FileCheck %s --check-prefix=NOOFFLOAD

// Two groups of two channels, as the ONNX importer spells a grouped
// convolution: the dense input viewed [N, G, Cg, H, W], the filter [G, Fg,
// Cg, kh, kw]. It becomes a dense convolution whose filter is zero off the
// diagonal blocks -- folded to a constant here.
//   group 0: out 0..1 read in 0..1;   group 1: out 2..3 read in 2..3.
// CHECK-LABEL: util.func public @two_groups
// CHECK-NOT: linalg.conv_2d_ngchw_gfchw
// CHECK: %[[W:.+]] = arith.constant dense<{{\[\[\[\[}}1.000000e+00]], {{\[\[}}2.000000e+00]], {{\[\[}}0.000000e+00]], {{\[\[}}0.000000e+00]]], {{\[\[\[}}3.000000e+00]], {{\[\[}}4.000000e+00]], {{\[\[}}0.000000e+00]], {{\[\[}}0.000000e+00]]], {{\[\[\[}}0.000000e+00]], {{\[\[}}0.000000e+00]], {{\[\[}}5.000000e+00]], {{\[\[}}6.000000e+00]]], {{\[\[\[}}0.000000e+00]], {{\[\[}}0.000000e+00]], {{\[\[}}7.000000e+00]], {{\[\[}}8.000000e+00]]]]> : tensor<4x4x1x1xf32>
// CHECK: %[[IN:.+]] = tensor.collapse_shape %{{.+}} {{\[\[}}0], [1, 2], [3], [4]] : tensor<1x2x2x8x8xf32> into tensor<1x4x8x8xf32>
// CHECK: %[[CONV:.+]] = linalg.conv_2d_nchw_fchw {{.+}} ins(%[[IN]], %[[W]] : tensor<1x4x8x8xf32>, tensor<4x4x1x1xf32>)
// CHECK: tensor.expand_shape %[[CONV]] {{\[\[}}0], [1, 2], [3], [4]] output_shape [1, 2, 2, 8, 8]
// NOOFFLOAD-LABEL: util.func public @two_groups
// NOOFFLOAD: linalg.conv_2d_ngchw_gfchw
util.func public @two_groups(%x: tensor<1x2x2x8x8xf32>, %init: tensor<1x2x2x8x8xf32>) -> tensor<1x2x2x8x8xf32> {
  %w = arith.constant dense<[[[[[1.0]], [[2.0]]], [[[3.0]], [[4.0]]]], [[[[5.0]], [[6.0]]], [[[7.0]], [[8.0]]]]]> : tensor<2x2x2x1x1xf32>
  %0 = linalg.conv_2d_ngchw_gfchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%x, %w : tensor<1x2x2x8x8xf32>, tensor<2x2x2x1x1xf32>) outs(%init : tensor<1x2x2x8x8xf32>) -> tensor<1x2x2x8x8xf32>
  util.return %0 : tensor<1x2x2x8x8xf32>
}

// An fp16 import widens its f16 filter before the convolution; the constant
// is folded through that `extf` and the importer's expand_shape.
// CHECK-LABEL: util.func public @widened_filter
// CHECK: arith.constant dense<{{.+}}> : tensor<128x128x3x3xf32>
// CHECK: linalg.conv_2d_nchw_fchw {{.+}} ins(%{{.+}}, %{{.+}} : tensor<1x128x58x58xf32>, tensor<128x128x3x3xf32>)
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
util.func public @widened_filter(%x: tensor<1x32x4x58x58xf32>, %init: tensor<1x32x4x56x56xf32>) -> tensor<1x32x4x56x56xf32> {
  %w16 = arith.constant dense<1.0> : tensor<128x4x3x3xf16>
  %e = tensor.empty() : tensor<128x4x3x3xf32>
  %w32 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%w16 : tensor<128x4x3x3xf16>) outs(%e : tensor<128x4x3x3xf32>) {
  ^bb0(%in: f16, %out: f32):
    %v = arith.extf %in : f16 to f32
    linalg.yield %v : f32
  } -> tensor<128x4x3x3xf32>
  %grouped = tensor.expand_shape %w32 [[0, 1], [2], [3], [4]] output_shape [32, 4, 4, 3, 3] : tensor<128x4x3x3xf32> into tensor<32x4x4x3x3xf32>
  %0 = linalg.conv_2d_ngchw_gfchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%x, %grouped : tensor<1x32x4x58x58xf32>, tensor<32x4x4x3x3xf32>) outs(%init : tensor<1x32x4x56x56xf32>) -> tensor<1x32x4x56x56xf32>
  util.return %0 : tensor<1x32x4x56x56xf32>
}

// Declined: a non-constant filter (it would be expanded on the CPU every
// run), a 5x5 kernel, and a dense extent above the 3x3 envelope (Cin 2048).
// CHECK-LABEL: util.func public @declines
// CHECK-COUNT-3: linalg.conv_2d_ngchw_gfchw
// CHECK-NOT: linalg.conv_2d_nchw_fchw
util.func public @declines(%x: tensor<1x2x2x8x8xf32>, %w: tensor<2x2x2x1x1xf32>, %init: tensor<1x2x2x8x8xf32>, %x5: tensor<1x2x2x12x12xf32>, %init5: tensor<1x2x2x8x8xf32>, %xw: tensor<1x32x64x10x10xf32>, %initw: tensor<1x32x64x8x8xf32>) -> (tensor<1x2x2x8x8xf32>, tensor<1x2x2x8x8xf32>, tensor<1x32x64x8x8xf32>) {
  %w5 = arith.constant dense<1.0> : tensor<2x2x2x5x5xf32>
  %ww = arith.constant dense<1.0> : tensor<32x64x64x3x3xf32>
  %0 = linalg.conv_2d_ngchw_gfchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%x, %w : tensor<1x2x2x8x8xf32>, tensor<2x2x2x1x1xf32>) outs(%init : tensor<1x2x2x8x8xf32>) -> tensor<1x2x2x8x8xf32>
  %1 = linalg.conv_2d_ngchw_gfchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%x5, %w5 : tensor<1x2x2x12x12xf32>, tensor<2x2x2x5x5xf32>) outs(%init5 : tensor<1x2x2x8x8xf32>) -> tensor<1x2x2x8x8xf32>
  %2 = linalg.conv_2d_ngchw_gfchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%xw, %ww : tensor<1x32x64x10x10xf32>, tensor<32x64x64x3x3xf32>) outs(%initw : tensor<1x32x64x8x8xf32>) -> tensor<1x32x64x8x8xf32>
  util.return %0, %1, %2 : tensor<1x2x2x8x8xf32>, tensor<1x2x2x8x8xf32>, tensor<1x32x64x8x8xf32>
}
