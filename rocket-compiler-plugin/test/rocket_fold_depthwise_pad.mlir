// RUN: iree-opt %s --pass-pipeline='builtin.module(rocket-fold-conv-pad)' \
// RUN:   | FileCheck %s

// A depthwise convolution stays NCHW through the channels-last conversion,
// which pads its input while it is still NHWC: the pad arrives two ops above
// the convolution, `transpose(collapse(pad(x)))`, where neither the mark nor
// a DAG template can see it. For the form a padded template claims -- 3x3,
// pad 1, stride 1 or 2, the fused ReLU6 after it -- the pass moves the pad
// below the transpose (a zero pad commutes with a layout change) and marks
// the convolution. MobileNetV2's seventeen depthwise convolutions are all
// this form; before, each kept a CPU pad copy between two NPU dispatches.

// The rebuilt collapse folds against the import's expand, so the transpose
// reads the producer's NHWC result directly -- the shim's own transpose then
// cancels against it, and the two NPU dispatches become adjacent.
//
// CHECK-LABEL: util.func public @depthwise_pad_moves_below_transpose(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9]+]]: tensor<1x56x56x144xf16>
// CHECK-NOT: tensor.pad
// CHECK: %[[NCHW:.+]] = linalg.transpose ins(%[[INPUT]] : tensor<1x56x56x144xf16>)
// CHECK-SAME: permutation = [0, 3, 1, 2]
// CHECK: %[[PAD:.+]] = tensor.pad %[[NCHW]] low[0, 0, 1, 1] high[0, 0, 1, 1]
// The fill is sunk into the region, as for every claimed pad: a template
// cannot match a region yielding a value defined outside it.
// CHECK-NEXT: ^bb0
// CHECK-NEXT: arith.constant 0.000000e+00 : f16
// CHECK-NEXT: tensor.yield
// CHECK: tensor<1x144x56x56xf16> to tensor<1x144x58x58xf16>
// CHECK: linalg.depthwise_conv_2d_nchw_chw
// CHECK-SAME: rocket.pad_left = 1 : i64
// CHECK-SAME: rocket.pad_top = 1 : i64
// CHECK-SAME: ins(%[[PAD]], %{{.+}} : tensor<1x144x58x58xf16>, tensor<144x3x3xf16>)
util.func public @depthwise_pad_moves_below_transpose(
    %input: tensor<1x56x56x144xf16>, %filter: tensor<144x3x3xf16>,
    %bias: tensor<144xf32>) -> tensor<1x144x56x56xf32> {
  %zero = arith.constant 0.000000e+00 : f16
  %low = arith.constant 0.000000e+00 : f32
  %high = arith.constant 6.000000e+00 : f32
  %expanded = tensor.expand_shape %input [[0], [1, 2], [3], [4]] output_shape [1, 1, 56, 56, 144]
      : tensor<1x56x56x144xf16> into tensor<1x1x56x56x144xf16>
  %padded = tensor.pad %expanded low[0, 0, 1, 1, 0] high[0, 0, 1, 1, 0] {
  ^bb0(%i0: index, %i1: index, %i2: index, %i3: index, %i4: index):
    tensor.yield %zero : f16
  } : tensor<1x1x56x56x144xf16> to tensor<1x1x58x58x144xf16>
  %collapsed = tensor.collapse_shape %padded [[0], [1, 2], [3], [4]]
      : tensor<1x1x58x58x144xf16> into tensor<1x58x58x144xf16>
  %nchw_empty = tensor.empty() : tensor<1x144x58x58xf16>
  %nchw = linalg.transpose ins(%collapsed : tensor<1x58x58x144xf16>)
      outs(%nchw_empty : tensor<1x144x58x58xf16>) permutation = [0, 3, 1, 2]
  %acc_empty = tensor.empty() : tensor<1x144x56x56xf32>
  %acc_init = linalg.fill ins(%low : f32) outs(%acc_empty : tensor<1x144x56x56xf32>) -> tensor<1x144x56x56xf32>
  %conv = linalg.depthwise_conv_2d_nchw_chw
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%nchw, %filter : tensor<1x144x58x58xf16>, tensor<144x3x3xf16>)
      outs(%acc_init : tensor<1x144x56x56xf32>) -> tensor<1x144x56x56xf32>
  %out_empty = tensor.empty() : tensor<1x144x56x56xf32>
  %clamped = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d1)>,
                       affine_map<(d0, d1, d2, d3) -> ()>,
                       affine_map<(d0, d1, d2, d3) -> ()>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%conv, %bias, %low, %high : tensor<1x144x56x56xf32>, tensor<144xf32>, f32, f32)
      outs(%out_empty : tensor<1x144x56x56xf32>) {
  ^bb0(%raw: f32, %b: f32, %lo: f32, %hi: f32, %unused: f32):
    %biased = arith.addf %raw, %b : f32
    %low_clamped = arith.maximumf %biased, %lo : f32
    %result = arith.minimumf %low_clamped, %hi : f32
    linalg.yield %result : f32
  } -> tensor<1x144x56x56xf32>
  util.return %clamped : tensor<1x144x56x56xf32>
}

// Stride 2 takes the same path: MobileNetV2's four stride-2 depthwise pads
// are symmetric in a torch export (pad 1 each side), unlike the SAME_UPPER
// ones an older ONNX import produced.
//
// CHECK-LABEL: util.func public @depthwise_pad_moves_below_transpose_s2
// CHECK: tensor.pad %{{.+}} low[0, 0, 1, 1] high[0, 0, 1, 1]
// CHECK: linalg.depthwise_conv_2d_nchw_chw
// CHECK-SAME: rocket.pad_left = 1 : i64
// CHECK-SAME: strides = dense<2> : vector<2xi64>
util.func public @depthwise_pad_moves_below_transpose_s2(
    %input: tensor<1x56x56x144xf16>, %filter: tensor<144x3x3xf16>,
    %bias: tensor<144xf32>) -> tensor<1x144x28x28xf32> {
  %zero = arith.constant 0.000000e+00 : f16
  %low = arith.constant 0.000000e+00 : f32
  %high = arith.constant 6.000000e+00 : f32
  %expanded = tensor.expand_shape %input [[0], [1, 2], [3], [4]] output_shape [1, 1, 56, 56, 144]
      : tensor<1x56x56x144xf16> into tensor<1x1x56x56x144xf16>
  %padded = tensor.pad %expanded low[0, 0, 1, 1, 0] high[0, 0, 1, 1, 0] {
  ^bb0(%i0: index, %i1: index, %i2: index, %i3: index, %i4: index):
    tensor.yield %zero : f16
  } : tensor<1x1x56x56x144xf16> to tensor<1x1x58x58x144xf16>
  %collapsed = tensor.collapse_shape %padded [[0], [1, 2], [3], [4]]
      : tensor<1x1x58x58x144xf16> into tensor<1x58x58x144xf16>
  %nchw_empty = tensor.empty() : tensor<1x144x58x58xf16>
  %nchw = linalg.transpose ins(%collapsed : tensor<1x58x58x144xf16>)
      outs(%nchw_empty : tensor<1x144x58x58xf16>) permutation = [0, 3, 1, 2]
  %acc_empty = tensor.empty() : tensor<1x144x28x28xf32>
  %acc_init = linalg.fill ins(%low : f32) outs(%acc_empty : tensor<1x144x28x28xf32>) -> tensor<1x144x28x28xf32>
  %conv = linalg.depthwise_conv_2d_nchw_chw
      {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
      ins(%nchw, %filter : tensor<1x144x58x58xf16>, tensor<144x3x3xf16>)
      outs(%acc_init : tensor<1x144x28x28xf32>) -> tensor<1x144x28x28xf32>
  %out_empty = tensor.empty() : tensor<1x144x28x28xf32>
  %clamped = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d1)>,
                       affine_map<(d0, d1, d2, d3) -> ()>,
                       affine_map<(d0, d1, d2, d3) -> ()>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%conv, %bias, %low, %high : tensor<1x144x28x28xf32>, tensor<144xf32>, f32, f32)
      outs(%out_empty : tensor<1x144x28x28xf32>) {
  ^bb0(%raw: f32, %b: f32, %lo: f32, %hi: f32, %unused: f32):
    %biased = arith.addf %raw, %b : f32
    %low_clamped = arith.maximumf %biased, %lo : f32
    %result = arith.minimumf %low_clamped, %hi : f32
    linalg.yield %result : f32
  } -> tensor<1x144x28x28xf32>
  util.return %clamped : tensor<1x144x28x28xf32>
}

// No padded template claims a 5x5, so its pad stays where it was and the
// convolution is not marked -- a mark would cost it the plain ReLU6 matcher
// (attribute-dictionary inequality) and buy nothing.
//
// CHECK-LABEL: util.func public @depthwise_5x5_pad_stays
// CHECK: tensor.pad %{{.+}} low[0, 0, 2, 2, 0] high[0, 0, 2, 2, 0]
// CHECK: linalg.transpose
// CHECK-NOT: tensor.pad
// CHECK: linalg.depthwise_conv_2d_nchw_chw
// CHECK-NOT: rocket.pad_top
// CHECK-SAME: ins(
util.func public @depthwise_5x5_pad_stays(
    %input: tensor<1x56x56x144xf16>, %filter: tensor<144x5x5xf16>,
    %bias: tensor<144xf32>) -> tensor<1x144x56x56xf32> {
  %zero = arith.constant 0.000000e+00 : f16
  %low = arith.constant 0.000000e+00 : f32
  %high = arith.constant 6.000000e+00 : f32
  %expanded = tensor.expand_shape %input [[0], [1, 2], [3], [4]] output_shape [1, 1, 56, 56, 144]
      : tensor<1x56x56x144xf16> into tensor<1x1x56x56x144xf16>
  %padded = tensor.pad %expanded low[0, 0, 2, 2, 0] high[0, 0, 2, 2, 0] {
  ^bb0(%i0: index, %i1: index, %i2: index, %i3: index, %i4: index):
    tensor.yield %zero : f16
  } : tensor<1x1x56x56x144xf16> to tensor<1x1x60x60x144xf16>
  %collapsed = tensor.collapse_shape %padded [[0], [1, 2], [3], [4]]
      : tensor<1x1x60x60x144xf16> into tensor<1x60x60x144xf16>
  %nchw_empty = tensor.empty() : tensor<1x144x60x60xf16>
  %nchw = linalg.transpose ins(%collapsed : tensor<1x60x60x144xf16>)
      outs(%nchw_empty : tensor<1x144x60x60xf16>) permutation = [0, 3, 1, 2]
  %acc_empty = tensor.empty() : tensor<1x144x56x56xf32>
  %acc_init = linalg.fill ins(%low : f32) outs(%acc_empty : tensor<1x144x56x56xf32>) -> tensor<1x144x56x56xf32>
  %conv = linalg.depthwise_conv_2d_nchw_chw
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%nchw, %filter : tensor<1x144x60x60xf16>, tensor<144x5x5xf16>)
      outs(%acc_init : tensor<1x144x56x56xf32>) -> tensor<1x144x56x56xf32>
  %out_empty = tensor.empty() : tensor<1x144x56x56xf32>
  %clamped = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d1)>,
                       affine_map<(d0, d1, d2, d3) -> ()>,
                       affine_map<(d0, d1, d2, d3) -> ()>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%conv, %bias, %low, %high : tensor<1x144x56x56xf32>, tensor<144xf32>, f32, f32)
      outs(%out_empty : tensor<1x144x56x56xf32>) {
  ^bb0(%raw: f32, %b: f32, %lo: f32, %hi: f32, %unused: f32):
    %biased = arith.addf %raw, %b : f32
    %low_clamped = arith.maximumf %biased, %lo : f32
    %result = arith.minimumf %low_clamped, %hi : f32
    linalg.yield %result : f32
  } -> tensor<1x144x56x56xf32>
  util.return %clamped : tensor<1x144x56x56xf32>
}

// Without the fused clamp no padded template applies either (the padded
// depthwise targets are ReLU6 ones), so this is left exactly as imported.
//
// CHECK-LABEL: util.func public @depthwise_without_clamp_pad_stays
// CHECK: tensor.pad %{{.+}} low[0, 0, 1, 1, 0] high[0, 0, 1, 1, 0]
// CHECK: linalg.transpose
// CHECK-NOT: tensor.pad
// CHECK: linalg.depthwise_conv_2d_nchw_chw
// CHECK-NOT: rocket.pad_top
// CHECK-SAME: ins(
util.func public @depthwise_without_clamp_pad_stays(
    %input: tensor<1x56x56x144xf16>, %filter: tensor<144x3x3xf16>) -> tensor<1x144x56x56xf32> {
  %zero = arith.constant 0.000000e+00 : f16
  %low = arith.constant 0.000000e+00 : f32
  %expanded = tensor.expand_shape %input [[0], [1, 2], [3], [4]] output_shape [1, 1, 56, 56, 144]
      : tensor<1x56x56x144xf16> into tensor<1x1x56x56x144xf16>
  %padded = tensor.pad %expanded low[0, 0, 1, 1, 0] high[0, 0, 1, 1, 0] {
  ^bb0(%i0: index, %i1: index, %i2: index, %i3: index, %i4: index):
    tensor.yield %zero : f16
  } : tensor<1x1x56x56x144xf16> to tensor<1x1x58x58x144xf16>
  %collapsed = tensor.collapse_shape %padded [[0], [1, 2], [3], [4]]
      : tensor<1x1x58x58x144xf16> into tensor<1x58x58x144xf16>
  %nchw_empty = tensor.empty() : tensor<1x144x58x58xf16>
  %nchw = linalg.transpose ins(%collapsed : tensor<1x58x58x144xf16>)
      outs(%nchw_empty : tensor<1x144x58x58xf16>) permutation = [0, 3, 1, 2]
  %acc_empty = tensor.empty() : tensor<1x144x56x56xf32>
  %acc_init = linalg.fill ins(%low : f32) outs(%acc_empty : tensor<1x144x56x56xf32>) -> tensor<1x144x56x56xf32>
  %conv = linalg.depthwise_conv_2d_nchw_chw
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%nchw, %filter : tensor<1x144x58x58xf16>, tensor<144x3x3xf16>)
      outs(%acc_init : tensor<1x144x56x56xf32>) -> tensor<1x144x56x56xf32>
  util.return %conv : tensor<1x144x56x56xf32>
}
