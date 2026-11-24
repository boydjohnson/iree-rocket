// RUN: iree-opt %s --pass-pipeline='builtin.module(rocket-demote-conv-inputs-to-f16)' \
// RUN:   | FileCheck %s --check-prefix=DEMOTE
// RUN: iree-opt %s --pass-pipeline='builtin.module(rocket-demote-conv-inputs-to-f16,rocket-promote-unclaimed-conv-inputs)' \
// RUN:   | FileCheck %s --check-prefix=PROMOTE
// RUN: iree-opt %s --pass-pipeline='builtin.module(rocket-demote-conv-inputs-to-f16,rocket-fold-conv-pad,rocket-promote-unclaimed-conv-inputs)' \
// RUN:   | FileCheck %s --check-prefix=FOLDED

// An f32 import's "same" padding is a zero tensor.pad in front of the
// convolution (in the channels-last 1x1xHxWxC form, collapsed back for the
// op). The demote narrows *through* it -- truncf on the pad's source, the
// pad rebuilt in f16 -- so `pad -> collapse -> conv` survives for
// rocket-fold-conv-pad and the pad-1 matchers, and the truncf sits on the
// producer's result where its widen can cancel it. A nonzero fill is left
// as an ordinary truncf over the padded value. The promote pass undoes the
// demotion through the pad exactly, rebuilding the f32 pad.

// DEMOTE-LABEL: util.func public @zero_pad
// DEMOTE: %[[NARROW:.+]] = linalg.generic
// DEMOTE-SAME: ins(%{{.+}} : tensor<1x1x8x8x16xf32>)
// DEMOTE: arith.truncf
// DEMOTE: %[[PAD:.+]] = tensor.pad %[[NARROW]] low[0, 0, 1, 1, 0] high[0, 0, 1, 1, 0]
// DEMOTE: tensor.yield %{{.+}} : f16
// DEMOTE: tensor<1x1x8x8x16xf16> to tensor<1x1x10x10x16xf16>
// DEMOTE: %[[COLLAPSED:.+]] = tensor.collapse_shape %[[PAD]]
// DEMOTE: linalg.conv_2d_nhwc_hwcf
// DEMOTE-SAME: rocket.f16_demoted
// DEMOTE-SAME: ins(%[[COLLAPSED]], %{{.+}} : tensor<1x10x10x16xf16>, tensor<3x3x16x32xf16>)

// PROMOTE-LABEL: util.func public @zero_pad
// PROMOTE-NOT: xf16
// PROMOTE: tensor.pad %{{.+}} low[0, 0, 1, 1, 0] high[0, 0, 1, 1, 0]
// PROMOTE: tensor.yield %{{.+}} : f32
// PROMOTE: linalg.conv_2d_nhwc_hwcf
// PROMOTE-SAME: ins(%{{.+}}, %{{.+}} : tensor<1x10x10x16xf32>, tensor<3x3x16x32xf32>)
// PROMOTE-NOT: xf16
util.func public @zero_pad(%input: tensor<1x8x8x16xf32>, %filter: tensor<3x3x16x32xf32>) -> tensor<1x8x8x32xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %expanded = tensor.expand_shape %input [[0], [1, 2], [3], [4]] output_shape [1, 1, 8, 8, 16] : tensor<1x8x8x16xf32> into tensor<1x1x8x8x16xf32>
  %padded = tensor.pad %expanded low[0, 0, 1, 1, 0] high[0, 0, 1, 1, 0] {
  ^bb0(%i0: index, %i1: index, %i2: index, %i3: index, %i4: index):
    tensor.yield %zero : f32
  } : tensor<1x1x8x8x16xf32> to tensor<1x1x10x10x16xf32>
  %collapsed = tensor.collapse_shape %padded [[0], [1, 2], [3], [4]] : tensor<1x1x10x10x16xf32> into tensor<1x10x10x16xf32>
  %empty = tensor.empty() : tensor<1x8x8x32xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x8x8x32xf32>) -> tensor<1x8x8x32xf32>
  %conv = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%collapsed, %filter : tensor<1x10x10x16xf32>, tensor<3x3x16x32xf32>)
      outs(%init : tensor<1x8x8x32xf32>) -> tensor<1x8x8x32xf32>
  util.return %conv : tensor<1x8x8x32xf32>
}

// DEMOTE-LABEL: util.func public @nonzero_pad
// DEMOTE: tensor.pad
// DEMOTE: tensor.yield %{{.+}} : f32
// DEMOTE: %[[NARROW:.+]] = linalg.generic
// DEMOTE-SAME: ins(%{{.+}} : tensor<1x10x10x16xf32>)
// DEMOTE: linalg.conv_2d_nhwc_hwcf
// DEMOTE-SAME: ins(%[[NARROW]]
util.func public @nonzero_pad(%input: tensor<1x8x8x16xf32>, %filter: tensor<3x3x16x32xf32>) -> tensor<1x8x8x32xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %one = arith.constant 1.000000e+00 : f32
  %expanded = tensor.expand_shape %input [[0], [1, 2], [3], [4]] output_shape [1, 1, 8, 8, 16] : tensor<1x8x8x16xf32> into tensor<1x1x8x8x16xf32>
  %padded = tensor.pad %expanded low[0, 0, 1, 1, 0] high[0, 0, 1, 1, 0] {
  ^bb0(%i0: index, %i1: index, %i2: index, %i3: index, %i4: index):
    tensor.yield %one : f32
  } : tensor<1x1x8x8x16xf32> to tensor<1x1x10x10x16xf32>
  %collapsed = tensor.collapse_shape %padded [[0], [1, 2], [3], [4]] : tensor<1x1x10x10x16xf32> into tensor<1x10x10x16xf32>
  %empty = tensor.empty() : tensor<1x8x8x32xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x8x8x32xf32>) -> tensor<1x8x8x32xf32>
  %conv = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%collapsed, %filter : tensor<1x10x10x16xf32>, tensor<3x3x16x32xf32>)
      outs(%init : tensor<1x8x8x32xf32>) -> tensor<1x8x8x32xf32>
  util.return %conv : tensor<1x8x8x32xf32>
}

// A depthwise convolution stays NCHW, so its padded input is
// `transpose(collapse(pad(x)))`. The demote goes through the transpose too:
// the truncf lands on `x` (where the producer's widen cancels it), and the
// pad, collapse and transpose are rebuilt in f16 above the convolution, so
// rocket-fold-conv-pad can move the pad below the transpose and mark it. The
// promote undoes either form exactly: as the demote left it (PROMOTE), and
// after the fold pass moved the pad (FOLDED) -- the shape an unclaimed
// convolution would reach it in.

// DEMOTE-LABEL: util.func public @depthwise_zero_pad
// DEMOTE: %[[NARROW:.+]] = linalg.generic
// DEMOTE-SAME: ins(%{{.+}} : tensor<1x1x8x8x16xf32>)
// DEMOTE: arith.truncf
// DEMOTE: %[[PAD:.+]] = tensor.pad %[[NARROW]] low[0, 0, 1, 1, 0] high[0, 0, 1, 1, 0]
// DEMOTE: %[[COLLAPSED:.+]] = tensor.collapse_shape %[[PAD]]
// DEMOTE: %[[NCHW:.+]] = linalg.transpose ins(%[[COLLAPSED]] : tensor<1x10x10x16xf16>)
// DEMOTE-SAME: permutation = [0, 3, 1, 2]
// DEMOTE: linalg.depthwise_conv_2d_nchw_chw
// DEMOTE-SAME: rocket.f16_demoted
// DEMOTE-SAME: ins(%[[NCHW]], %{{.+}} : tensor<1x16x10x10xf16>, tensor<16x3x3xf16>)

// PROMOTE-LABEL: util.func public @depthwise_zero_pad
// PROMOTE-NOT: xf16
// PROMOTE: linalg.depthwise_conv_2d_nchw_chw
// PROMOTE-SAME: ins(%{{.+}}, %{{.+}} : tensor<1x16x10x10xf32>, tensor<16x3x3xf32>)
// PROMOTE-NOT: xf16

// FOLDED-LABEL: util.func public @depthwise_zero_pad
// FOLDED-NOT: xf16
// FOLDED: %[[PAD:.+]] = tensor.pad %{{.+}} low[0, 0, 1, 1] high[0, 0, 1, 1]
// FOLDED: tensor.yield %{{.+}} : f32
// FOLDED: linalg.depthwise_conv_2d_nchw_chw
// FOLDED-SAME: ins(%[[PAD]], %{{.+}} : tensor<1x16x10x10xf32>, tensor<16x3x3xf32>)
// FOLDED-NOT: xf16
util.func public @depthwise_zero_pad(%input: tensor<1x8x8x16xf32>, %filter: tensor<16x3x3xf32>,
                                     %bias: tensor<16xf32>) -> tensor<1x16x8x8xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %six = arith.constant 6.000000e+00 : f32
  %expanded = tensor.expand_shape %input [[0], [1, 2], [3], [4]] output_shape [1, 1, 8, 8, 16] : tensor<1x8x8x16xf32> into tensor<1x1x8x8x16xf32>
  %padded = tensor.pad %expanded low[0, 0, 1, 1, 0] high[0, 0, 1, 1, 0] {
  ^bb0(%i0: index, %i1: index, %i2: index, %i3: index, %i4: index):
    tensor.yield %zero : f32
  } : tensor<1x1x8x8x16xf32> to tensor<1x1x10x10x16xf32>
  %collapsed = tensor.collapse_shape %padded [[0], [1, 2], [3], [4]] : tensor<1x1x10x10x16xf32> into tensor<1x10x10x16xf32>
  %nchw_empty = tensor.empty() : tensor<1x16x10x10xf32>
  %nchw = linalg.transpose ins(%collapsed : tensor<1x10x10x16xf32>)
      outs(%nchw_empty : tensor<1x16x10x10xf32>) permutation = [0, 3, 1, 2]
  %empty = tensor.empty() : tensor<1x16x8x8xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x16x8x8xf32>) -> tensor<1x16x8x8xf32>
  %conv = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%nchw, %filter : tensor<1x16x10x10xf32>, tensor<16x3x3xf32>)
      outs(%init : tensor<1x16x8x8xf32>) -> tensor<1x16x8x8xf32>
  %out_empty = tensor.empty() : tensor<1x16x8x8xf32>
  %clamped = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d1)>,
                       affine_map<(d0, d1, d2, d3) -> ()>,
                       affine_map<(d0, d1, d2, d3) -> ()>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%conv, %bias, %zero, %six : tensor<1x16x8x8xf32>, tensor<16xf32>, f32, f32)
      outs(%out_empty : tensor<1x16x8x8xf32>) {
  ^bb0(%raw: f32, %b: f32, %lo: f32, %hi: f32, %unused: f32):
    %biased = arith.addf %raw, %b : f32
    %low_clamped = arith.maximumf %biased, %lo : f32
    %result = arith.minimumf %low_clamped, %hi : f32
    linalg.yield %result : f32
  } -> tensor<1x16x8x8xf32>
  util.return %clamped : tensor<1x16x8x8xf32>
}
