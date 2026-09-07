// RUN: iree-opt %s --pass-pipeline='builtin.module(rocket-fuse-conv-relu6)' \
// RUN:   | FileCheck %s

// The pass in isolation. It rewrites an fp16 convolution and the ReLU6 that
// follows it into the two-op form @match_dynamic_conv2d_relu6 claims:
//
//   * the per-channel bias is lifted out of the convolution's init and into
//     the epilogue generic, because that is where the hardware computes it
//     (accumulate -> BS bias -> BN activation -> OUT_CVT), and a clamp in BN
//     sees the biased value only if the bias is on the BS plane;
//   * the bounds move from broadcast tensors to scalar operands, because a
//     value captured from outside a generic's region can never match --
//     `cast_compatible_dag_from_root` builds its value mapping only from the
//     ops it walked;
//   * the channels-last `tensor.expand_shape` moves after the clamp, because
//     the DAG matcher compares whole attribute dictionaries and a reshape's
//     output shape is a static attribute that differs at every site.
//
// The two positive cases below are the forms that actually occur in the
// pipeline, not the tidied ones. Matching only the folded spellings -- a bare
// `linalg.broadcast` init and splat `arith.constant` bounds -- is how the
// first version of this pass read correctly on an end-of-preprocessing dump
// and fused zero of MobileNetV2's 18 sites.

// The real form: the bias init is transpose(broadcast(bias)), because
// `iree-preprocessing-convert-conv-to-channels-last` puts a transpose in
// front of the broadcast rather than rewriting it, and the bounds are
// broadcasts of rank-0 constants.

// CHECK-LABEL: util.func public @conv_relu6_pipeline_form
// CHECK-DAG: %[[LOW:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[HIGH:.+]] = arith.constant 6.000000e+00 : f32
// The convolution now accumulates over a zero init.
// CHECK: %[[FILL:.+]] = linalg.fill
// CHECK: %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf
// CHECK-SAME: outs(%[[FILL]]
// The bias and the bounds are operands of the epilogue.
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[CONV]], %{{.+}}, %[[LOW]], %[[HIGH]]
// CHECK-SAME: tensor<1x112x112x144xf32>, tensor<144xf32>, f32, f32)
// CHECK: arith.addf
// CHECK: arith.maximumf
// CHECK: arith.minimumf
// CHECK-NOT: arith.cmpf
// The reshape is re-emitted after it, so everything downstream keeps its rank.
// CHECK: tensor.expand_shape
// CHECK-SAME: into tensor<1x1x112x112x144xf32>
util.func public @conv_relu6_pipeline_form(%input: tensor<1x112x112x24xf16>,
                                           %filter: tensor<1x1x24x144xf16>,
                                           %bias: tensor<144xf32>)
    -> tensor<1x1x112x112x144xf32> {
  %lo_scalar = arith.constant dense<0.000000e+00> : tensor<f32>
  %hi_scalar = arith.constant dense<6.000000e+00> : tensor<f32>
  %lo_empty = tensor.empty() : tensor<1x1x112x112x144xf32>
  %hi_empty = tensor.empty() : tensor<1x1x112x112x144xf32>
  %low = linalg.broadcast ins(%lo_scalar : tensor<f32>)
      outs(%lo_empty : tensor<1x1x112x112x144xf32>) dimensions = [0, 1, 2, 3, 4]
  %high = linalg.broadcast ins(%hi_scalar : tensor<f32>)
      outs(%hi_empty : tensor<1x1x112x112x144xf32>) dimensions = [0, 1, 2, 3, 4]
  %bias_empty = tensor.empty() : tensor<1x144x112x112xf32>
  %bias_nchw = linalg.broadcast ins(%bias : tensor<144xf32>)
      outs(%bias_empty : tensor<1x144x112x112xf32>) dimensions = [0, 2, 3]
  %init_empty = tensor.empty() : tensor<1x112x112x144xf32>
  %init = linalg.transpose ins(%bias_nchw : tensor<1x144x112x112xf32>)
      outs(%init_empty : tensor<1x112x112x144xf32>) permutation = [0, 2, 3, 1]
  %conv = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, rocket.f16_demoted, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x112x112x24xf16>, tensor<1x1x24x144xf16>)
      outs(%init : tensor<1x112x112x144xf32>) -> tensor<1x112x112x144xf32>
  %expanded = tensor.expand_shape %conv [[0], [1, 2], [3], [4]]
      output_shape [1, 1, 112, 112, 144]
      : tensor<1x112x112x144xf32> into tensor<1x1x112x112x144xf32>
  %empty = tensor.empty() : tensor<1x1x112x112x144xf32>
  %clamped = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
      ins(%expanded, %low, %high : tensor<1x1x112x112x144xf32>,
          tensor<1x1x112x112x144xf32>, tensor<1x1x112x112x144xf32>)
      outs(%empty : tensor<1x1x112x112x144xf32>) {
  ^bb0(%in: f32, %l: f32, %h: f32, %out: f32):
    %0 = arith.cmpf ult, %in, %l : f32
    %1 = arith.select %0, %l, %in : f32
    %2 = arith.cmpf ugt, %1, %h : f32
    %3 = arith.select %2, %h, %1 : f32
    linalg.yield %3 : f32
  } -> tensor<1x1x112x112x144xf32>
  util.return %clamped : tensor<1x1x112x112x144xf32>
}

// The folded spellings -- a bare broadcast init and splat constant bounds,
// with no reshape -- also fuse, because a later canonicalisation can produce
// them and a model without the channels-last expansion arrives this way.

// CHECK-LABEL: util.func public @conv_relu6_folded_form
// CHECK: linalg.fill
// CHECK: linalg.conv_2d_nhwc_hwcf
// CHECK: arith.addf
// CHECK: arith.maximumf
// CHECK-NOT: tensor.expand_shape
util.func public @conv_relu6_folded_form(%input: tensor<1x14x14x88xf16>,
                                         %filter: tensor<1x1x88x528xf16>,
                                         %bias: tensor<528xf32>)
    -> tensor<1x14x14x528xf32> {
  %low = arith.constant dense<0.000000e+00> : tensor<1x14x14x528xf32>
  %high = arith.constant dense<6.000000e+00> : tensor<1x14x14x528xf32>
  %init_empty = tensor.empty() : tensor<1x14x14x528xf32>
  %init = linalg.broadcast ins(%bias : tensor<528xf32>)
      outs(%init_empty : tensor<1x14x14x528xf32>) dimensions = [0, 1, 2]
  %conv = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x14x14x88xf16>, tensor<1x1x88x528xf16>)
      outs(%init : tensor<1x14x14x528xf32>) -> tensor<1x14x14x528xf32>
  %empty = tensor.empty() : tensor<1x14x14x528xf32>
  %clamped = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%conv, %low, %high : tensor<1x14x14x528xf32>,
          tensor<1x14x14x528xf32>, tensor<1x14x14x528xf32>)
      outs(%empty : tensor<1x14x14x528xf32>) {
  ^bb0(%in: f32, %l: f32, %h: f32, %out: f32):
    %0 = arith.cmpf ult, %in, %l : f32
    %1 = arith.select %0, %l, %in : f32
    %2 = arith.cmpf ugt, %1, %h : f32
    %3 = arith.select %2, %h, %1 : f32
    linalg.yield %3 : f32
  } -> tensor<1x14x14x528xf32>
  util.return %clamped : tensor<1x14x14x528xf32>
}

// A ceiling other than 6.0 is left alone, and that is load-bearing rather
// than a missing feature: the wire carries `activation_cmp` as a static
// attribute on the executable target, so the canonical form encodes exactly
// one ceiling and the matcher -- which matches structure, not constants --
// cannot tell two ceilings apart. Producing the canonical form here for a
// different ceiling would compile it as 6.0.

// CHECK-LABEL: util.func public @conv_relu1_declines
// CHECK: arith.cmpf ult
// CHECK: arith.cmpf ugt
// CHECK-NOT: arith.maximumf
util.func public @conv_relu1_declines(%input: tensor<1x14x14x88xf16>,
                                      %filter: tensor<1x1x88x528xf16>,
                                      %bias: tensor<528xf32>)
    -> tensor<1x14x14x528xf32> {
  %low = arith.constant dense<0.000000e+00> : tensor<1x14x14x528xf32>
  %high = arith.constant dense<1.000000e+00> : tensor<1x14x14x528xf32>
  %init_empty = tensor.empty() : tensor<1x14x14x528xf32>
  %init = linalg.broadcast ins(%bias : tensor<528xf32>)
      outs(%init_empty : tensor<1x14x14x528xf32>) dimensions = [0, 1, 2]
  %conv = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x14x14x88xf16>, tensor<1x1x88x528xf16>)
      outs(%init : tensor<1x14x14x528xf32>) -> tensor<1x14x14x528xf32>
  %empty = tensor.empty() : tensor<1x14x14x528xf32>
  %clamped = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%conv, %low, %high : tensor<1x14x14x528xf32>,
          tensor<1x14x14x528xf32>, tensor<1x14x14x528xf32>)
      outs(%empty : tensor<1x14x14x528xf32>) {
  ^bb0(%in: f32, %lo: f32, %hi: f32, %out: f32):
    %0 = arith.cmpf ult, %in, %lo : f32
    %1 = arith.select %0, %lo, %in : f32
    %2 = arith.cmpf ugt, %1, %hi : f32
    %3 = arith.select %2, %hi, %1 : f32
    linalg.yield %3 : f32
  } -> tensor<1x14x14x528xf32>
  util.return %clamped : tensor<1x14x14x528xf32>
}

// A convolution whose result has a second reader is left alone: the rewrite
// deletes what it walks through, so absorbing an intermediate that something
// else reads would change that reader's value.

// CHECK-LABEL: util.func public @conv_relu6_second_reader_declines
// CHECK: arith.cmpf ult
// CHECK-NOT: arith.maximumf
util.func public @conv_relu6_second_reader_declines(%input: tensor<1x14x14x88xf16>,
                                                    %filter: tensor<1x1x88x528xf16>,
                                                    %bias: tensor<528xf32>)
    -> (tensor<1x14x14x528xf32>, tensor<1x14x14x528xf32>) {
  %low = arith.constant dense<0.000000e+00> : tensor<1x14x14x528xf32>
  %high = arith.constant dense<6.000000e+00> : tensor<1x14x14x528xf32>
  %init_empty = tensor.empty() : tensor<1x14x14x528xf32>
  %init = linalg.broadcast ins(%bias : tensor<528xf32>)
      outs(%init_empty : tensor<1x14x14x528xf32>) dimensions = [0, 1, 2]
  %conv = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x14x14x88xf16>, tensor<1x1x88x528xf16>)
      outs(%init : tensor<1x14x14x528xf32>) -> tensor<1x14x14x528xf32>
  %empty = tensor.empty() : tensor<1x14x14x528xf32>
  %clamped = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%conv, %low, %high : tensor<1x14x14x528xf32>,
          tensor<1x14x14x528xf32>, tensor<1x14x14x528xf32>)
      outs(%empty : tensor<1x14x14x528xf32>) {
  ^bb0(%in: f32, %lo: f32, %hi: f32, %out: f32):
    %0 = arith.cmpf ult, %in, %lo : f32
    %1 = arith.select %0, %lo, %in : f32
    %2 = arith.cmpf ugt, %1, %hi : f32
    %3 = arith.select %2, %hi, %1 : f32
    linalg.yield %3 : f32
  } -> tensor<1x14x14x528xf32>
  util.return %clamped, %conv : tensor<1x14x14x528xf32>, tensor<1x14x14x528xf32>
}

// An f32 convolution is left alone. Only the fp16 path fuses: the BN stage
// sits before OUT_CVT, so an int8 ceiling is in post-BS accumulator units
// (`Activation::clamped_int8`) and is a different encoding entirely.

// CHECK-LABEL: util.func public @conv_relu6_f32_declines
// CHECK: arith.cmpf ult
// CHECK-NOT: arith.maximumf
util.func public @conv_relu6_f32_declines(%input: tensor<1x14x14x88xf32>,
                                          %filter: tensor<1x1x88x528xf32>,
                                          %bias: tensor<528xf32>)
    -> tensor<1x14x14x528xf32> {
  %low = arith.constant dense<0.000000e+00> : tensor<1x14x14x528xf32>
  %high = arith.constant dense<6.000000e+00> : tensor<1x14x14x528xf32>
  %init_empty = tensor.empty() : tensor<1x14x14x528xf32>
  %init = linalg.broadcast ins(%bias : tensor<528xf32>)
      outs(%init_empty : tensor<1x14x14x528xf32>) dimensions = [0, 1, 2]
  %conv = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x14x14x88xf32>, tensor<1x1x88x528xf32>)
      outs(%init : tensor<1x14x14x528xf32>) -> tensor<1x14x14x528xf32>
  %empty = tensor.empty() : tensor<1x14x14x528xf32>
  %clamped = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%conv, %low, %high : tensor<1x14x14x528xf32>,
          tensor<1x14x14x528xf32>, tensor<1x14x14x528xf32>)
      outs(%empty : tensor<1x14x14x528xf32>) {
  ^bb0(%in: f32, %lo: f32, %hi: f32, %out: f32):
    %0 = arith.cmpf ult, %in, %lo : f32
    %1 = arith.select %0, %lo, %in : f32
    %2 = arith.cmpf ugt, %1, %hi : f32
    %3 = arith.select %2, %hi, %1 : f32
    linalg.yield %3 : f32
  } -> tensor<1x14x14x528xf32>
  util.return %clamped : tensor<1x14x14x528xf32>
}
