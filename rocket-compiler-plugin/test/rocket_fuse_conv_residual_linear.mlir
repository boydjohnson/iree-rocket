// RUN: iree-opt %s --pass-pipeline='builtin.module(rocket-fuse-conv-relu6)' \
// RUN:   | FileCheck %s

// `conv(x) + skip` with nothing clamping the sum -- MobileNetV2's linear
// bottlenecks -- becomes the conv over a zero init and one epilogue generic
// {addf bias; addf skip}, the form @match_dynamic_conv2d_residual* claim
// (conv tiles + one EW task, the EW core's ReLU off). An f32 import arrives
// with no truncf and an f32 skip, which the epilogue adds as is (the shim
// narrows it); an f16 import's skip is f16 and is widened in the body.

// CHECK-LABEL: util.func public @linear_residual_f32_import
// CHECK: tensor.collapse_shape %{{.+}} {{\[}}[0], [1, 2], [3], [4]]
// CHECK: %[[FILL:.+]] = linalg.fill
// CHECK: %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf
// CHECK-SAME: outs(%[[FILL]]
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[CONV]], %{{.+}}, %{{.+}} : tensor<1x14x14x64xf32>, tensor<64xf32>, tensor<1x14x14x64xf32>)
// CHECK: arith.addf
// CHECK-NEXT: arith.addf
// CHECK-NEXT: linalg.yield
// CHECK: tensor.expand_shape
// CHECK-NOT: linalg.add
util.func public @linear_residual_f32_import(%input: tensor<1x14x14x384xf16>,
    %filter: tensor<1x1x384x64xf16>, %bias: tensor<64xf32>, %skip: tensor<1x1x14x14x64xf32>) -> tensor<1x1x14x14x64xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %nchw_empty = tensor.empty() : tensor<1x64x14x14xf32>
  %bias_nchw = linalg.broadcast ins(%bias : tensor<64xf32>)
      outs(%nchw_empty : tensor<1x64x14x14xf32>) dimensions = [0, 2, 3]
  %init_empty = tensor.empty() : tensor<1x14x14x64xf32>
  %init = linalg.transpose ins(%bias_nchw : tensor<1x64x14x14xf32>)
      outs(%init_empty : tensor<1x14x14x64xf32>) permutation = [0, 2, 3, 1]
  %conv = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, rocket.f16_demoted, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x14x14x384xf16>, tensor<1x1x384x64xf16>)
      outs(%init : tensor<1x14x14x64xf32>) -> tensor<1x14x14x64xf32>
  %expanded = tensor.expand_shape %conv [[0], [1, 2], [3], [4]] output_shape [1, 1, 14, 14, 64]
      : tensor<1x14x14x64xf32> into tensor<1x1x14x14x64xf32>
  %sum_empty = tensor.empty() : tensor<1x1x14x14x64xf32>
  %sum = linalg.add ins(%skip, %expanded : tensor<1x1x14x14x64xf32>, tensor<1x1x14x14x64xf32>) outs(%sum_empty : tensor<1x1x14x14x64xf32>) -> tensor<1x1x14x14x64xf32>
  util.return %sum : tensor<1x1x14x14x64xf32>
}

// The NCHW spelling: the conv's result and the skip are both transposed to
// NCHW before the add (the first add of a stage, after the channels-last
// conversion). The skip is transposed back to the conv's layout, the
// epilogue runs NHWC, and the transpose is re-applied to its result.
//
// CHECK-LABEL: util.func public @linear_residual_nchw_f32
// CHECK: %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf
// CHECK: %[[EPILOGUE:.+]] = linalg.generic
// CHECK-SAME: ins(%[[CONV]], %{{.+}}, %{{.+}} : tensor<1x28x28x32xf32>, tensor<32xf32>, tensor<1x28x28x32xf32>)
// CHECK: linalg.transpose ins(%[[EPILOGUE]] : tensor<1x28x28x32xf32>)
// CHECK-SAME: permutation = [0, 3, 1, 2]
// CHECK-NOT: linalg.add
util.func public @linear_residual_nchw_f32(%input: tensor<1x28x28x384xf16>,
    %filter: tensor<1x1x384x32xf16>, %bias: tensor<32xf32>, %skip: tensor<1x32x28x28xf32>)
    -> tensor<1x32x28x28xf32> {
  %nchw_empty = tensor.empty() : tensor<1x32x28x28xf32>
  %bias_nchw = linalg.broadcast ins(%bias : tensor<32xf32>)
      outs(%nchw_empty : tensor<1x32x28x28xf32>) dimensions = [0, 2, 3]
  %init_empty = tensor.empty() : tensor<1x28x28x32xf32>
  %init = linalg.transpose ins(%bias_nchw : tensor<1x32x28x28xf32>)
      outs(%init_empty : tensor<1x28x28x32xf32>) permutation = [0, 2, 3, 1]
  %conv = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, rocket.f16_demoted, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x28x28x384xf16>, tensor<1x1x384x32xf16>)
      outs(%init : tensor<1x28x28x32xf32>) -> tensor<1x28x28x32xf32>
  %t_empty = tensor.empty() : tensor<1x32x28x28xf32>
  %t = linalg.transpose ins(%conv : tensor<1x28x28x32xf32>)
      outs(%t_empty : tensor<1x32x28x28xf32>) permutation = [0, 3, 1, 2]
  %sum_empty = tensor.empty() : tensor<1x32x28x28xf32>
  %sum = linalg.add ins(%t, %skip : tensor<1x32x28x28xf32>, tensor<1x32x28x28xf32>)
      outs(%sum_empty : tensor<1x32x28x28xf32>) -> tensor<1x32x28x28xf32>
  util.return %sum : tensor<1x32x28x28xf32>
}

// Cout 24 (MobileNetV2's 56x56 stage) is not a multiple of the EW cube's 16
// channels, so the add stays where it is.
//
// CHECK-LABEL: util.func public @linear_residual_cout24_declines
// CHECK: linalg.add
util.func public @linear_residual_cout24_declines(%input: tensor<1x56x56x384xf16>,
    %filter: tensor<1x1x384x24xf16>, %bias: tensor<24xf32>, %skip: tensor<1x1x56x56x24xf32>) -> tensor<1x1x56x56x24xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %nchw_empty = tensor.empty() : tensor<1x24x56x56xf32>
  %bias_nchw = linalg.broadcast ins(%bias : tensor<24xf32>)
      outs(%nchw_empty : tensor<1x24x56x56xf32>) dimensions = [0, 2, 3]
  %init_empty = tensor.empty() : tensor<1x56x56x24xf32>
  %init = linalg.transpose ins(%bias_nchw : tensor<1x24x56x56xf32>)
      outs(%init_empty : tensor<1x56x56x24xf32>) permutation = [0, 2, 3, 1]
  %conv = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, rocket.f16_demoted, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x56x56x384xf16>, tensor<1x1x384x24xf16>)
      outs(%init : tensor<1x56x56x24xf32>) -> tensor<1x56x56x24xf32>
  %expanded = tensor.expand_shape %conv [[0], [1, 2], [3], [4]] output_shape [1, 1, 56, 56, 24]
      : tensor<1x56x56x24xf32> into tensor<1x1x56x56x24xf32>
  %sum_empty = tensor.empty() : tensor<1x1x56x56x24xf32>
  %sum = linalg.add ins(%skip, %expanded : tensor<1x1x56x56x24xf32>, tensor<1x1x56x56x24xf32>) outs(%sum_empty : tensor<1x1x56x56x24xf32>) -> tensor<1x1x56x56x24xf32>
  util.return %sum : tensor<1x1x56x56x24xf32>
}

// A sum something ReLUs is not linear: left alone, so it cannot be split
// into an NPU add and a CPU clamp.
//
// CHECK-LABEL: util.func public @residual_then_relu_declines_linear
// CHECK: linalg.add
util.func public @residual_then_relu_declines_linear(%input: tensor<1x14x14x384xf16>,
    %filter: tensor<1x1x384x64xf16>, %bias: tensor<64xf32>, %skip: tensor<1x1x14x14x64xf32>) -> tensor<1x1x14x14x64xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %nchw_empty = tensor.empty() : tensor<1x64x14x14xf32>
  %bias_nchw = linalg.broadcast ins(%bias : tensor<64xf32>)
      outs(%nchw_empty : tensor<1x64x14x14xf32>) dimensions = [0, 2, 3]
  %init_empty = tensor.empty() : tensor<1x14x14x64xf32>
  %init = linalg.transpose ins(%bias_nchw : tensor<1x64x14x14xf32>)
      outs(%init_empty : tensor<1x14x14x64xf32>) permutation = [0, 2, 3, 1]
  %conv = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, rocket.f16_demoted, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x14x14x384xf16>, tensor<1x1x384x64xf16>)
      outs(%init : tensor<1x14x14x64xf32>) -> tensor<1x14x14x64xf32>
  %expanded = tensor.expand_shape %conv [[0], [1, 2], [3], [4]] output_shape [1, 1, 14, 14, 64]
      : tensor<1x14x14x64xf32> into tensor<1x1x14x14x64xf32>
  %sum_empty = tensor.empty() : tensor<1x1x14x14x64xf32>
  %sum = linalg.add ins(%skip, %expanded : tensor<1x1x14x14x64xf32>, tensor<1x1x14x14x64xf32>) outs(%sum_empty : tensor<1x1x14x14x64xf32>) -> tensor<1x1x14x14x64xf32>
  %relu_empty = tensor.empty() : tensor<1x1x14x14x64xf32>
  %relu = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
      ins(%sum : tensor<1x1x14x14x64xf32>) outs(%relu_empty : tensor<1x1x14x14x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %0 = arith.cmpf ugt, %in, %zero : f32
    %1 = arith.select %0, %in, %zero : f32
    linalg.yield %1 : f32
  } -> tensor<1x1x14x14x64xf32>
  util.return %relu : tensor<1x1x14x14x64xf32>
}
