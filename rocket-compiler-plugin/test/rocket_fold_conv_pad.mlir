// RUN: iree-opt %s --pass-pipeline='builtin.module(rocket-fold-conv-pad)' \
// RUN:   | FileCheck %s

// The pass in isolation. It records on a convolution what the `tensor.pad`
// in front of it was, so a matcher can hand that padding to the CNA instead
// of leaving it as the full-tensor copy IREE materializes (a `slow_memcpy`
// in an `audit` listing -- 16 of them on ResNet50 fp16, all feeding
// offloaded convolutions).
//
// The pad sits two ops back: the channels-last conversion pads in the rank-5
// `1x1xHxWxC` form and collapses to rank 4 for the convolution.

// The fill is also sunk into the pad's region. That is not cosmetic: the DAG
// matcher compares regions structurally, and a region yielding a value
// defined outside itself can never match a template, since equivalence has
// nothing to map it to. The inline constant is also what pins the fill to
// zero on the matcher's side.
//
// CHECK-LABEL: util.func public @conv_symmetric_pad
// CHECK: tensor.pad
// CHECK: arith.constant 0.000000e+00 : f16
// CHECK-NEXT: tensor.yield
// CHECK: linalg.conv_2d_nhwc_hwcf
// CHECK-SAME: rocket.pad_left = 1 : i64
// CHECK-SAME: rocket.pad_top = 1 : i64
util.func public @conv_symmetric_pad(%input: tensor<1x1x56x56x64xf16>,
                                     %filter: tensor<3x3x64x64xf16>,
                                     %init: tensor<1x56x56x64xf32>)
    -> tensor<1x56x56x64xf32> {
  %zero = arith.constant 0.000000e+00 : f16
  %padded = tensor.pad %input low[0, 0, 1, 1, 0] high[0, 0, 1, 1, 0] {
  ^bb0(%a: index, %b: index, %c: index, %d: index, %e: index):
    tensor.yield %zero : f16
  } : tensor<1x1x56x56x64xf16> to tensor<1x1x58x58x64xf16>
  %collapsed = tensor.collapse_shape %padded [[0], [1, 2], [3], [4]]
      : tensor<1x1x58x58x64xf16> into tensor<1x58x58x64xf16>
  %conv = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%collapsed, %filter : tensor<1x58x58x64xf16>, tensor<3x3x64x64xf16>)
      outs(%init : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  util.return %conv : tensor<1x56x56x64xf32>
}

// An asymmetric pad is declined, and that is a **hardware** limit rather than
// a missing wire field: `CNA_PAD_CON0` has `pad_top` and `pad_left` and
// nothing else, and the hardware applies each to *both* sides --
// `Shape::output_width` is `(w + 2 * pad_left - kw) / stride + 1`, matched
// against all 150 strided programs in the vendor corpus. Recording
// `low[0] high[1]` as a symmetric pad of any amount would shift every output
// window and silently compute a different convolution. This is what ONNX
// emits for `auto_pad = SAME_UPPER` at stride 2, so it is the common case,
// not a corner one -- five of MobileNetV2's eighteen pads are this.

// CHECK-LABEL: util.func public @conv_asymmetric_pad_declines
// CHECK: linalg.conv_2d_nhwc_hwcf
// CHECK-NOT: rocket.pad_top
util.func public @conv_asymmetric_pad_declines(%input: tensor<1x1x224x224x3xf16>,
                                               %filter: tensor<3x3x3x64xf16>,
                                               %init: tensor<1x112x112x64xf32>)
    -> tensor<1x112x112x64xf32> {
  %zero = arith.constant 0.000000e+00 : f16
  %padded = tensor.pad %input low[0, 0, 0, 0, 0] high[0, 0, 1, 1, 0] {
  ^bb0(%a: index, %b: index, %c: index, %d: index, %e: index):
    tensor.yield %zero : f16
  } : tensor<1x1x224x224x3xf16> to tensor<1x1x225x225x3xf16>
  %collapsed = tensor.collapse_shape %padded [[0], [1, 2], [3], [4]]
      : tensor<1x1x225x225x3xf16> into tensor<1x225x225x3xf16>
  %conv = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
      ins(%collapsed, %filter : tensor<1x225x225x3xf16>, tensor<3x3x3x64xf16>)
      outs(%init : tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32>
  util.return %conv : tensor<1x112x112x64xf32>
}

// A nonzero fill is declined: the CNA fills with `CNA_PAD_CON1.pad_value`,
// which every fp16 target leaves at zero and which is not on the wire.

// CHECK-LABEL: util.func public @conv_nonzero_pad_value_declines
// CHECK: linalg.conv_2d_nhwc_hwcf
// CHECK-NOT: rocket.pad_top
util.func public @conv_nonzero_pad_value_declines(%input: tensor<1x1x56x56x64xf16>,
                                                  %filter: tensor<3x3x64x64xf16>,
                                                  %init: tensor<1x56x56x64xf32>)
    -> tensor<1x56x56x64xf32> {
  %one = arith.constant 1.000000e+00 : f16
  %padded = tensor.pad %input low[0, 0, 1, 1, 0] high[0, 0, 1, 1, 0] {
  ^bb0(%a: index, %b: index, %c: index, %d: index, %e: index):
    tensor.yield %one : f16
  } : tensor<1x1x56x56x64xf16> to tensor<1x1x58x58x64xf16>
  %collapsed = tensor.collapse_shape %padded [[0], [1, 2], [3], [4]]
      : tensor<1x1x58x58x64xf16> into tensor<1x58x58x64xf16>
  %conv = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%collapsed, %filter : tensor<1x58x58x64xf16>, tensor<3x3x64x64xf16>)
      outs(%init : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  util.return %conv : tensor<1x56x56x64xf32>
}

// A pad on the channel axis is not padding in the convolution's sense.

// CHECK-LABEL: util.func public @conv_channel_pad_declines
// CHECK: linalg.conv_2d_nhwc_hwcf
// CHECK-NOT: rocket.pad_top
util.func public @conv_channel_pad_declines(%input: tensor<1x1x56x56x60xf16>,
                                            %filter: tensor<3x3x64x64xf16>,
                                            %init: tensor<1x54x54x64xf32>)
    -> tensor<1x54x54x64xf32> {
  %zero = arith.constant 0.000000e+00 : f16
  %padded = tensor.pad %input low[0, 0, 0, 0, 0] high[0, 0, 0, 0, 4] {
  ^bb0(%a: index, %b: index, %c: index, %d: index, %e: index):
    tensor.yield %zero : f16
  } : tensor<1x1x56x56x60xf16> to tensor<1x1x56x56x64xf16>
  %collapsed = tensor.collapse_shape %padded [[0], [1, 2], [3], [4]]
      : tensor<1x1x56x56x64xf16> into tensor<1x56x56x64xf16>
  %conv = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%collapsed, %filter : tensor<1x56x56x64xf16>, tensor<3x3x64x64xf16>)
      outs(%init : tensor<1x54x54x64xf32>) -> tensor<1x54x54x64xf32>
  util.return %conv : tensor<1x54x54x64xf32>
}

// A convolution `rocket-fuse-conv-relu6` has already claimed is left alone.
// This pass runs after it -- the sink has to be the last thing before the
// match loop, because every greedy pattern driver hoists the constant back
// out of `tensor.pad`'s non-isolated region -- and marking such a convolution
// would change the attribute dictionary the ReLU6 template compares, costing
// the activation fusion and gaining nothing, since no template claims a pad
// and a clamp together.

// CHECK-LABEL: util.func public @conv_with_fused_relu6_declines
// CHECK: linalg.conv_2d_nhwc_hwcf
// CHECK-NOT: rocket.pad_top
util.func public @conv_with_fused_relu6_declines(%input: tensor<1x1x56x56x64xf16>,
                                                 %filter: tensor<3x3x64x64xf16>,
                                                 %bias: tensor<64xf32>,
                                                 %init: tensor<1x56x56x64xf32>)
    -> tensor<1x56x56x64xf32> {
  %zero = arith.constant 0.000000e+00 : f16
  %lo = arith.constant 0.000000e+00 : f32
  %hi = arith.constant 6.000000e+00 : f32
  %padded = tensor.pad %input low[0, 0, 1, 1, 0] high[0, 0, 1, 1, 0] {
  ^bb0(%a: index, %b: index, %c: index, %d: index, %e: index):
    tensor.yield %zero : f16
  } : tensor<1x1x56x56x64xf16> to tensor<1x1x58x58x64xf16>
  %collapsed = tensor.collapse_shape %padded [[0], [1, 2], [3], [4]]
      : tensor<1x1x58x58x64xf16> into tensor<1x58x58x64xf16>
  %conv = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%collapsed, %filter : tensor<1x58x58x64xf16>, tensor<3x3x64x64xf16>)
      outs(%init : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %empty = tensor.empty() : tensor<1x56x56x64xf32>
  %activated = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d3)>,
                       affine_map<(d0, d1, d2, d3) -> ()>,
                       affine_map<(d0, d1, d2, d3) -> ()>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%conv, %bias, %lo, %hi : tensor<1x56x56x64xf32>, tensor<64xf32>, f32, f32)
      outs(%empty : tensor<1x56x56x64xf32>) {
  ^bb0(%raw: f32, %b: f32, %l: f32, %h: f32, %out: f32):
    %biased = arith.addf %raw, %b : f32
    %low_clamped = arith.maximumf %biased, %l : f32
    %clamped = arith.minimumf %low_clamped, %h : f32
    linalg.yield %clamped : f32
  } -> tensor<1x56x56x64xf32>
  util.return %activated : tensor<1x56x56x64xf32>
}
