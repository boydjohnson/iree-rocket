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

// The NCHW depthwise form. The channels-last conversion leaves a depthwise
// convolution in NCHW and puts the layout change on its *result*, so the
// chain is conv -> transpose -> expand_shape -> clamp and the bias broadcast
// retains dimension 1 rather than the last one. Both differences have to
// reach the output: the clamp lands in NCHW with a `(d1)` bias map, and the
// transpose is re-emitted after it.
//
// Reachable only when the depthwise convolutions are demoted to f16, which
// is off by default -- see RocketDemoteConvInputsPass's scope comment for
// the measurement and the two lines that turn it on.

// CHECK-LABEL: util.func public @depthwise_conv_relu6_nchw
// CHECK: %[[FILL:.+]] = linalg.fill
// CHECK: %[[CONV:.+]] = linalg.depthwise_conv_2d_nchw_chw
// CHECK-SAME: outs(%[[FILL]]
// The clamp is in NCHW, so the bias map has to be (d1) rather than the NHWC
// (d3). That is not spelled as a CHECK because MLIR hoists the map into a
// `#mapN` alias whose number is not stable -- it is enforced instead by the
// op verifying at all: a 144-channel bias read through (d3) against a
// 1x144x56x56 tensor is `inferred input/output operand #1 has shape's
// dimension #0 to be 56, but found 144`, which is exactly how the first cut
// of this failed in the real pipeline.
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[CONV]]
// CHECK-SAME: tensor<1x144x56x56xf32>, tensor<144xf32>, f32, f32)
// CHECK: arith.addf
// CHECK: arith.maximumf
// CHECK: arith.minimumf
// CHECK-NOT: arith.cmpf
// CHECK: linalg.transpose
// CHECK: tensor.expand_shape
util.func public @depthwise_conv_relu6_nchw(%input: tensor<1x144x113x113xf16>,
                                            %filter: tensor<144x3x3xf16>,
                                            %bias: tensor<144xf32>)
    -> tensor<1x1x56x56x144xf32> {
  %lo_scalar = arith.constant dense<0.000000e+00> : tensor<f32>
  %hi_scalar = arith.constant dense<6.000000e+00> : tensor<f32>
  %lo_empty = tensor.empty() : tensor<1x1x56x56x144xf32>
  %hi_empty = tensor.empty() : tensor<1x1x56x56x144xf32>
  %low = linalg.broadcast ins(%lo_scalar : tensor<f32>)
      outs(%lo_empty : tensor<1x1x56x56x144xf32>) dimensions = [0, 1, 2, 3, 4]
  %high = linalg.broadcast ins(%hi_scalar : tensor<f32>)
      outs(%hi_empty : tensor<1x1x56x56x144xf32>) dimensions = [0, 1, 2, 3, 4]
  %init_empty = tensor.empty() : tensor<1x144x56x56xf32>
  %init = linalg.broadcast ins(%bias : tensor<144xf32>)
      outs(%init_empty : tensor<1x144x56x56xf32>) dimensions = [0, 2, 3]
  %conv = linalg.depthwise_conv_2d_nchw_chw
      {dilations = dense<1> : vector<2xi64>, rocket.f16_demoted, strides = dense<2> : vector<2xi64>}
      ins(%input, %filter : tensor<1x144x113x113xf16>, tensor<144x3x3xf16>)
      outs(%init : tensor<1x144x56x56xf32>) -> tensor<1x144x56x56xf32>
  %nhwc_empty = tensor.empty() : tensor<1x56x56x144xf32>
  %nhwc = linalg.transpose ins(%conv : tensor<1x144x56x56xf32>)
      outs(%nhwc_empty : tensor<1x56x56x144xf32>) permutation = [0, 2, 3, 1]
  %expanded = tensor.expand_shape %nhwc [[0], [1, 2], [3], [4]]
      output_shape [1, 1, 56, 56, 144]
      : tensor<1x56x56x144xf32> into tensor<1x1x56x56x144xf32>
  %empty = tensor.empty() : tensor<1x1x56x56x144xf32>
  %clamped = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
      ins(%expanded, %low, %high : tensor<1x1x56x56x144xf32>,
          tensor<1x1x56x56x144xf32>, tensor<1x1x56x56x144xf32>)
      outs(%empty : tensor<1x1x56x56x144xf32>) {
  ^bb0(%in: f32, %l: f32, %h: f32, %out: f32):
    %0 = arith.cmpf ult, %in, %l : f32
    %1 = arith.select %0, %l, %in : f32
    %2 = arith.cmpf ugt, %1, %h : f32
    %3 = arith.select %2, %h, %1 : f32
    linalg.yield %3 : f32
  } -> tensor<1x1x56x56x144xf32>
  util.return %clamped : tensor<1x1x56x56x144xf32>
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

// The f16-import chain, as ResNet50 arrives: the bias init is
// transpose(broadcast(extf(bias))), the accumulator is narrowed to f16 by a
// generic, and the ReLU is a cmpf ugt / select in f16 against a *captured*
// scalar constant. The clamp moves onto the BN stage as maximumf on the f32
// accumulator, the narrow is re-emitted after the fused epilogue, and the
// f16 clamp is gone.

// CHECK-LABEL: util.func public @conv_relu_f16_import
// CHECK: %[[FILL:.+]] = linalg.fill
// CHECK: %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf
// CHECK-SAME: outs(%[[FILL]]
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[CONV]], %{{.+}}, %{{.+}} : tensor<1x56x56x64xf32>, tensor<64xf32>, f32)
// CHECK: arith.addf
// CHECK-NEXT: arith.maximumf
// CHECK-NOT: arith.minimumf
// CHECK: tensor.expand_shape
// CHECK: arith.truncf
// CHECK-NOT: arith.select
util.func public @conv_relu_f16_import(%input: tensor<1x56x56x64xf16>,
                                       %filter: tensor<1x1x64x64xf16>,
                                       %bias_f16: tensor<64xf16>)
    -> tensor<1x1x56x56x64xf16> {
  %cst = arith.constant 0.000000e+00 : f16
  %bias_empty = tensor.empty() : tensor<64xf32>
  %bias = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%bias_f16 : tensor<64xf16>) outs(%bias_empty : tensor<64xf32>) {
  ^bb0(%in: f16, %out: f32):
    %0 = arith.extf %in : f16 to f32
    linalg.yield %0 : f32
  } -> tensor<64xf32>
  %nchw_empty = tensor.empty() : tensor<1x64x56x56xf32>
  %bias_nchw = linalg.broadcast ins(%bias : tensor<64xf32>)
      outs(%nchw_empty : tensor<1x64x56x56xf32>) dimensions = [0, 2, 3]
  %init_empty = tensor.empty() : tensor<1x56x56x64xf32>
  %init = linalg.transpose ins(%bias_nchw : tensor<1x64x56x56xf32>)
      outs(%init_empty : tensor<1x56x56x64xf32>) permutation = [0, 2, 3, 1]
  %conv = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x56x56x64xf16>, tensor<1x1x64x64xf16>)
      outs(%init : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %expanded = tensor.expand_shape %conv [[0], [1, 2], [3], [4]]
      output_shape [1, 1, 56, 56, 64]
      : tensor<1x56x56x64xf32> into tensor<1x1x56x56x64xf32>
  %narrow_empty = tensor.empty() : tensor<1x1x56x56x64xf16>
  %narrowed = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
      ins(%expanded : tensor<1x1x56x56x64xf32>)
      outs(%narrow_empty : tensor<1x1x56x56x64xf16>) {
  ^bb0(%in: f32, %out: f16):
    %0 = arith.truncf %in : f32 to f16
    linalg.yield %0 : f16
  } -> tensor<1x1x56x56x64xf16>
  %relu_empty = tensor.empty() : tensor<1x1x56x56x64xf16>
  %relu = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
      ins(%narrowed : tensor<1x1x56x56x64xf16>)
      outs(%relu_empty : tensor<1x1x56x56x64xf16>) {
  ^bb0(%in: f16, %out: f16):
    %0 = arith.cmpf ugt, %in, %cst : f16
    %1 = arith.select %0, %in, %cst : f16
    linalg.yield %1 : f16
  } -> tensor<1x1x56x56x64xf16>
  util.return %relu : tensor<1x1x56x56x64xf16>
}

// A convolution with a per-channel bias and no activation: the bias alone
// moves onto the BS plane, so the shim no longer adds it on the CPU. The
// consumers are untouched.

// CHECK-LABEL: util.func public @conv_bias_only
// CHECK: %[[FILL:.+]] = linalg.fill
// CHECK: %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf
// CHECK-SAME: outs(%[[FILL]]
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[CONV]], %{{.+}} : tensor<1x56x56x256xf32>, tensor<256xf32>)
// CHECK: arith.addf
// CHECK-NEXT: linalg.yield
// CHECK: tensor.expand_shape
util.func public @conv_bias_only(%input: tensor<1x56x56x64xf16>,
                                 %filter: tensor<1x1x64x256xf16>,
                                 %bias: tensor<256xf32>)
    -> tensor<1x1x56x56x256xf32> {
  %nchw_empty = tensor.empty() : tensor<1x256x56x56xf32>
  %bias_nchw = linalg.broadcast ins(%bias : tensor<256xf32>)
      outs(%nchw_empty : tensor<1x256x56x56xf32>) dimensions = [0, 2, 3]
  %init_empty = tensor.empty() : tensor<1x56x56x256xf32>
  %init = linalg.transpose ins(%bias_nchw : tensor<1x256x56x56xf32>)
      outs(%init_empty : tensor<1x56x56x256xf32>) permutation = [0, 2, 3, 1]
  %conv = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x56x56x64xf16>, tensor<1x1x64x256xf16>)
      outs(%init : tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
  %expanded = tensor.expand_shape %conv [[0], [1, 2], [3], [4]]
      output_shape [1, 1, 56, 56, 256]
      : tensor<1x56x56x256xf32> into tensor<1x1x56x56x256xf32>
  util.return %expanded : tensor<1x1x56x56x256xf32>
}

// A 7x7 filter has no fused target, so it is left with its bias init: a
// convolution the matchers cannot claim keeps the form the CPU fuses best.

// CHECK-LABEL: util.func public @conv_bias_7x7_declines
// CHECK-NOT: linalg.fill
// CHECK: linalg.conv_2d_nhwc_hwcf
// CHECK-NOT: arith.addf
util.func public @conv_bias_7x7_declines(%input: tensor<1x230x230x3xf16>,
                                         %filter: tensor<7x7x3x64xf16>,
                                         %bias: tensor<64xf32>)
    -> tensor<1x112x112x64xf32> {
  %init_empty = tensor.empty() : tensor<1x112x112x64xf32>
  %init = linalg.broadcast ins(%bias : tensor<64xf32>)
      outs(%init_empty : tensor<1x112x112x64xf32>) dimensions = [0, 1, 2]
  %conv = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
      ins(%input, %filter : tensor<1x230x230x3xf16>, tensor<7x7x3x64xf16>)
      outs(%init : tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32>
  util.return %conv : tensor<1x112x112x64xf32>
}

// The residual block's tail on the f16-import chain: conv, narrow, then an
// add of the skip with a ReLU on the sum. The skip is brought to the
// convolution's rank with the inverse reshape, the bias, the skip and the
// ReLU all move into one epilogue generic on the f32 accumulator, and the
// narrow is re-emitted after it.

// CHECK-LABEL: util.func public @conv_residual_relu_f16_import
// CHECK: tensor.collapse_shape %[[SKIP:.+]] {{\[}}[0], [1, 2], [3], [4]]
// CHECK: %[[FILL:.+]] = linalg.fill
// CHECK: %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf
// CHECK-SAME: outs(%[[FILL]]
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[CONV]], %{{.+}}, %{{.+}}, %{{.+}} : tensor<1x56x56x256xf32>, tensor<256xf32>, tensor<1x56x56x256xf16>, f32)
// CHECK: arith.addf
// CHECK-NEXT: arith.extf
// CHECK-NEXT: arith.addf
// CHECK-NEXT: arith.maximumf
// CHECK: tensor.expand_shape
// CHECK: arith.truncf
// CHECK-NOT: arith.select
util.func public @conv_residual_relu_f16_import(%input: tensor<1x56x56x64xf16>,
                                                %filter: tensor<1x1x64x256xf16>,
                                                %bias: tensor<256xf32>,
                                                %skip: tensor<1x1x56x56x256xf16>)
    -> tensor<1x1x56x56x256xf16> {
  %cst = arith.constant 0.000000e+00 : f16
  %nchw_empty = tensor.empty() : tensor<1x256x56x56xf32>
  %bias_nchw = linalg.broadcast ins(%bias : tensor<256xf32>)
      outs(%nchw_empty : tensor<1x256x56x56xf32>) dimensions = [0, 2, 3]
  %init_empty = tensor.empty() : tensor<1x56x56x256xf32>
  %init = linalg.transpose ins(%bias_nchw : tensor<1x256x56x56xf32>)
      outs(%init_empty : tensor<1x56x56x256xf32>) permutation = [0, 2, 3, 1]
  %conv = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x56x56x64xf16>, tensor<1x1x64x256xf16>)
      outs(%init : tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
  %expanded = tensor.expand_shape %conv [[0], [1, 2], [3], [4]]
      output_shape [1, 1, 56, 56, 256]
      : tensor<1x56x56x256xf32> into tensor<1x1x56x56x256xf32>
  %narrow_empty = tensor.empty() : tensor<1x1x56x56x256xf16>
  %narrowed = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
      ins(%expanded : tensor<1x1x56x56x256xf32>)
      outs(%narrow_empty : tensor<1x1x56x56x256xf16>) {
  ^bb0(%in: f32, %out: f16):
    %0 = arith.truncf %in : f32 to f16
    linalg.yield %0 : f16
  } -> tensor<1x1x56x56x256xf16>
  %sum_empty = tensor.empty() : tensor<1x1x56x56x256xf16>
  %sum = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
      ins(%narrowed, %skip : tensor<1x1x56x56x256xf16>, tensor<1x1x56x56x256xf16>)
      outs(%sum_empty : tensor<1x1x56x56x256xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %0 = arith.addf %a, %b : f16
    %1 = arith.cmpf ugt, %0, %cst : f16
    %2 = arith.select %1, %0, %cst : f16
    linalg.yield %2 : f16
  } -> tensor<1x1x56x56x256xf16>
  util.return %sum : tensor<1x1x56x56x256xf16>
}

// An already-f16 import narrows the accumulator before it clamps, so the
// clamp arrives one domain lower with a `truncf` generic between it and the
// convolution, and with each bound narrowed *inside* the region. Before the
// walk stepped through that narrowing this fused nothing at all -- which is
// the whole ReLU6 gap on an f16-imported MobileNetV2, 34 standalone clamp
// dispatches. What leaves puts the clamp back in f32 on the raw accumulator,
// with the narrowing re-emitted after it: the same function, because
// rounding to f16 is monotone and 0.0 and 6.0 are exactly representable
// there.

// CHECK-LABEL: util.func public @conv_relu6_f16_import
//   The convolution now accumulates over a zero init, bias lifted out.
// CHECK:       %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:       %[[FILL:.+]] = linalg.fill ins(%[[ZERO]]
// CHECK:       %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf
// CHECK-SAME:      outs(%[[FILL]]
//   Bias and both bounds are scalar/1-D operands on the epilogue, and the
//   body is the short form in f32.
// CHECK:       linalg.generic
// CHECK-SAME:      ins(%[[CONV]]
// CHECK:         arith.addf
// CHECK:         arith.maximumf
// CHECK:         arith.minimumf
//   The narrowing is re-emitted last, so the result type is unchanged.
// CHECK:       arith.truncf
// CHECK-NOT:   arith.cmpf
util.func public @conv_relu6_f16_import(
    %input: tensor<1x14x14x96xf16>,
    %filter: tensor<1x1x96x576xf16>,
    %bias: tensor<576xf32>) -> tensor<1x1x14x14x576xf16> {
  %lo = arith.constant dense<0.000000e+00> : tensor<f32>
  %hi = arith.constant dense<6.000000e+00> : tensor<f32>
  %acc = tensor.empty() : tensor<1x14x14x576xf32>
  %init = linalg.broadcast ins(%bias : tensor<576xf32>)
      outs(%acc : tensor<1x14x14x576xf32>) dimensions = [0, 1, 2]
  %conv = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x14x14x96xf16>, tensor<1x1x96x576xf16>)
      outs(%init : tensor<1x14x14x576xf32>) -> tensor<1x14x14x576xf32>
  %expanded = tensor.expand_shape %conv [[0], [1, 2], [3], [4]]
      output_shape [1, 1, 14, 14, 576]
      : tensor<1x14x14x576xf32> into tensor<1x1x14x14x576xf32>
  %narrow_empty = tensor.empty() : tensor<1x1x14x14x576xf16>
  %narrowed = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
      ins(%expanded : tensor<1x1x14x14x576xf32>)
      outs(%narrow_empty : tensor<1x1x14x14x576xf16>) {
  ^bb0(%in: f32, %out: f16):
    %0 = arith.truncf %in : f32 to f16
    linalg.yield %0 : f16
  } -> tensor<1x1x14x14x576xf16>
  %clamp_empty = tensor.empty() : tensor<1x1x14x14x576xf16>
  %clamped = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> ()>,
                       affine_map<(d0, d1, d2, d3, d4) -> ()>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
      ins(%narrowed, %lo, %hi
          : tensor<1x1x14x14x576xf16>, tensor<f32>, tensor<f32>)
      outs(%clamp_empty : tensor<1x1x14x14x576xf16>) {
  ^bb0(%x: f16, %l: f32, %u: f32, %out: f16):
    %lf = arith.truncf %l : f32 to f16
    %p = arith.cmpf ult, %x, %lf : f16
    %q = arith.select %p, %lf, %x : f16
    %uf = arith.truncf %u : f32 to f16
    %r = arith.cmpf ugt, %q, %uf : f16
    %s = arith.select %r, %uf, %q : f16
    linalg.yield %s : f16
  } -> tensor<1x1x14x14x576xf16>
  util.return %clamped : tensor<1x1x14x14x576xf16>
}
