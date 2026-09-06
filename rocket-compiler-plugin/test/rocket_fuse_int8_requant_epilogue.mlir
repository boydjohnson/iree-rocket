// RUN: iree-opt --pass-pipeline="builtin.module(func.func(rocket-fuse-int8-requant-epilogue))" %s | FileCheck %s

// The five-op requantization epilogue an ONNX QLinearConv arrives as, after
// iree-global-opt-quantized-conv-to-conv and the channels-last conversion.
// This is copied from mobilenetv2.static-int8.onnx rather than invented: the
// bias is the convolution's init, the zero-point correction reduces the
// convolution's own filter, the layout transpose sits on the i32 tensor, and
// the narrowing clamps to the ui8 range and uses arith.fptoui.

// CHECK-LABEL: func.func @qlinearconv_epilogue
//   The convolution keeps its operands and loses its bias to a zero init.
// CHECK:       %[[INIT:.+]] = linalg.fill ins(%{{.+}} : i32) outs(%{{.+}} : tensor<1x4x4x144xi32>)
// CHECK:       %[[ACC:.+]] = linalg.conv_2d_nhwc_hwcf
// CHECK-SAME:      outs(%[[INIT]] : tensor<1x4x4x144xi32>)
//   The zero-point correction is now a rank-1 expression over the bias and
//   the filter reduction, which constant evaluation folds to a literal.
// CHECK:       %[[BIAS:.+]] = linalg.generic {{.*}}iterator_types = ["parallel"]{{.*}}outs(%{{.+}} : tensor<144xi32>)
//   One elementwise generic, with the scale, zero point and clamp bounds as
//   operands rather than body constants -- the form the requantized matcher
//   is written against.
// CHECK:       linalg.generic
// CHECK-SAME:      ins(%[[ACC]], %[[BIAS]]
// CHECK:         arith.addi
// CHECK:         arith.sitofp
// CHECK:         arith.mulf
// CHECK:         math.roundeven
// CHECK:         arith.addf
// CHECK:         arith.maximumf
// CHECK:         arith.minimumf
// CHECK:         arith.fptosi
//   The signed-to-unsigned shift the DPU's output range needs, then the
//   layout transpose -- now moving i8 rather than i32.
// CHECK:       linalg.generic
// CHECK:         arith.addi %{{.+}}, %c-128_i8
// CHECK:       linalg.transpose
// CHECK-SAME:      permutation = [0, 3, 1, 2]
//   Nothing of the original epilogue survives.
// CHECK-NOT:   arith.fptoui
func.func @qlinearconv_epilogue(
    %input: tensor<1x4x4x24xi8>,
    %filter: tensor<1x1x24x144xi8>,
    %bias: tensor<144xi32>) -> tensor<1x144x4x4xi8> {
  %c0_i32 = arith.constant 0 : i32
  %c-2_i32 = arith.constant -2 : i32
  %acc_scale = arith.constant 1.500000e-03 : f32
  %out_scale = arith.constant 2.500000e-02 : f32
  %out_zp = arith.constant 1.220000e+02 : f32
  %lo = arith.constant 0.000000e+00 : f32
  %hi = arith.constant 2.550000e+02 : f32

  // sum_k(w), reduced from this convolution's own filter.
  %sum_empty = tensor.empty() : tensor<144xi32>
  %sum_init = linalg.fill ins(%c0_i32 : i32) outs(%sum_empty : tensor<144xi32>) -> tensor<144xi32>
  %sum = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1, d2, d3, d0)>,
                       affine_map<(d0, d1, d2, d3) -> (d0)>],
      iterator_types = ["parallel", "reduction", "reduction", "reduction"]}
      ins(%filter : tensor<1x1x24x144xi8>) outs(%sum_init : tensor<144xi32>) {
    ^bb0(%in: i8, %out: i32):
      %e = arith.extsi %in : i8 to i32
      %a = arith.addi %e, %out : i32
      linalg.yield %a : i32
  } -> tensor<144xi32>

  // The bias arrives as the convolution's accumulator init.
  %acc_empty = tensor.empty() : tensor<1x4x4x144xi32>
  %bias_bcast = linalg.broadcast ins(%bias : tensor<144xi32>)
      outs(%acc_empty : tensor<1x4x4x144xi32>) dimensions = [0, 1, 2]
  %acc = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x4x4x24xi8>, tensor<1x1x24x144xi8>)
      outs(%bias_bcast : tensor<1x4x4x144xi32>) -> tensor<1x4x4x144xi32>

  // acc - sum_k(w) * x_zp
  %corr_empty = tensor.empty() : tensor<1x4x4x144xi32>
  %sum_bcast_empty = tensor.empty() : tensor<1x4x4x144xi32>
  %sum_bcast = linalg.broadcast ins(%sum : tensor<144xi32>)
      outs(%sum_bcast_empty : tensor<1x4x4x144xi32>) dimensions = [0, 1, 2]
  %corr = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%acc, %sum_bcast : tensor<1x4x4x144xi32>, tensor<1x4x4x144xi32>)
      outs(%corr_empty : tensor<1x4x4x144xi32>) {
    ^bb0(%in: i32, %in_sum: i32, %out: i32):
      %m = arith.muli %in_sum, %c-2_i32 : i32
      %s = arith.subi %in, %m : i32
      linalg.yield %s : i32
  } -> tensor<1x4x4x144xi32>

  // NHWC -> NCHW, on the i32 tensor.
  %t_empty = tensor.empty() : tensor<1x144x4x4xi32>
  %transposed = linalg.transpose ins(%corr : tensor<1x4x4x144xi32>)
      outs(%t_empty : tensor<1x144x4x4xi32>) permutation = [0, 3, 1, 2]

  // sitofp, multiply by x_scale * w_scale.
  %real_empty = tensor.empty() : tensor<1x144x4x4xf32>
  %real = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%transposed : tensor<1x144x4x4xi32>) outs(%real_empty : tensor<1x144x4x4xf32>) {
    ^bb0(%in: i32, %out: f32):
      %f = arith.sitofp %in : i32 to f32
      %m = arith.mulf %f, %acc_scale : f32
      linalg.yield %m : f32
  } -> tensor<1x144x4x4xf32>

  // Divide by y_scale, round, offset, clamp to the ui8 range, narrow.
  %out_empty = tensor.empty() : tensor<1x144x4x4xi8>
  %quantized = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%real : tensor<1x144x4x4xf32>) outs(%out_empty : tensor<1x144x4x4xi8>) {
    ^bb0(%in: f32, %out: i8):
      %d = arith.divf %in, %out_scale : f32
      %r = math.roundeven %d : f32
      %z = arith.addf %r, %out_zp : f32
      %l = arith.maximumf %z, %lo : f32
      %c = arith.minimumf %l, %hi : f32
      %n = arith.fptoui %c : f32 to i8
      linalg.yield %n : i8
  } -> tensor<1x144x4x4xi8>

  return %quantized : tensor<1x144x4x4xi8>
}

// A convolution whose accumulator has a second reader cannot be absorbed --
// the rewrite deletes the epilogue, and the other reader would lose its
// operand. Declined, and the original chain is left standing.

// CHECK-LABEL: func.func @accumulator_has_two_readers
// CHECK:       linalg.conv_2d_nhwc_hwcf
// CHECK:       arith.fptoui
func.func @accumulator_has_two_readers(
    %input: tensor<1x4x4x24xi8>,
    %filter: tensor<1x1x24x144xi8>,
    %bias: tensor<144xi32>) -> (tensor<1x4x4x144xi8>, tensor<1x4x4x144xi32>) {
  %c0_i32 = arith.constant 0 : i32
  %c-2_i32 = arith.constant -2 : i32
  %acc_scale = arith.constant 1.500000e-03 : f32
  %out_scale = arith.constant 2.500000e-02 : f32
  %out_zp = arith.constant 1.220000e+02 : f32
  %lo = arith.constant 0.000000e+00 : f32
  %hi = arith.constant 2.550000e+02 : f32

  %sum_empty = tensor.empty() : tensor<144xi32>
  %sum_init = linalg.fill ins(%c0_i32 : i32) outs(%sum_empty : tensor<144xi32>) -> tensor<144xi32>
  %sum = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1, d2, d3, d0)>,
                       affine_map<(d0, d1, d2, d3) -> (d0)>],
      iterator_types = ["parallel", "reduction", "reduction", "reduction"]}
      ins(%filter : tensor<1x1x24x144xi8>) outs(%sum_init : tensor<144xi32>) {
    ^bb0(%in: i8, %out: i32):
      %e = arith.extsi %in : i8 to i32
      %a = arith.addi %e, %out : i32
      linalg.yield %a : i32
  } -> tensor<144xi32>

  %acc_empty = tensor.empty() : tensor<1x4x4x144xi32>
  %bias_bcast = linalg.broadcast ins(%bias : tensor<144xi32>)
      outs(%acc_empty : tensor<1x4x4x144xi32>) dimensions = [0, 1, 2]
  %acc = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x4x4x24xi8>, tensor<1x1x24x144xi8>)
      outs(%bias_bcast : tensor<1x4x4x144xi32>) -> tensor<1x4x4x144xi32>

  %corr_empty = tensor.empty() : tensor<1x4x4x144xi32>
  %sum_bcast_empty = tensor.empty() : tensor<1x4x4x144xi32>
  %sum_bcast = linalg.broadcast ins(%sum : tensor<144xi32>)
      outs(%sum_bcast_empty : tensor<1x4x4x144xi32>) dimensions = [0, 1, 2]
  %corr = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%acc, %sum_bcast : tensor<1x4x4x144xi32>, tensor<1x4x4x144xi32>)
      outs(%corr_empty : tensor<1x4x4x144xi32>) {
    ^bb0(%in: i32, %in_sum: i32, %out: i32):
      %m = arith.muli %in_sum, %c-2_i32 : i32
      %s = arith.subi %in, %m : i32
      linalg.yield %s : i32
  } -> tensor<1x4x4x144xi32>

  %real_empty = tensor.empty() : tensor<1x4x4x144xf32>
  %real = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%corr : tensor<1x4x4x144xi32>) outs(%real_empty : tensor<1x4x4x144xf32>) {
    ^bb0(%in: i32, %out: f32):
      %f = arith.sitofp %in : i32 to f32
      %m = arith.mulf %f, %acc_scale : f32
      linalg.yield %m : f32
  } -> tensor<1x4x4x144xf32>

  %out_empty = tensor.empty() : tensor<1x4x4x144xi8>
  %quantized = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%real : tensor<1x4x4x144xf32>) outs(%out_empty : tensor<1x4x4x144xi8>) {
    ^bb0(%in: f32, %out: i8):
      %d = arith.divf %in, %out_scale : f32
      %r = math.roundeven %d : f32
      %z = arith.addf %r, %out_zp : f32
      %l = arith.maximumf %z, %lo : f32
      %c = arith.minimumf %l, %hi : f32
      %n = arith.fptoui %c : f32 to i8
      linalg.yield %n : i8
  } -> tensor<1x4x4x144xi8>

  return %quantized, %corr : tensor<1x4x4x144xi8>, tensor<1x4x4x144xi32>
}
