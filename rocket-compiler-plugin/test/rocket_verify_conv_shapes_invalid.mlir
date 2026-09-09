// RUN: iree-opt %s --split-input-file --verify-diagnostics --pass-pipeline='builtin.module(rocket-verify-conv-shapes)'

// Each failure mode of rocket-verify-conv-shapes, with the record
// rocket-record-conv-attrs would have left written by hand so the
// disagreement can be staged without a broken pass to produce it. The
// extents are symbolic wherever the check does not need them: the point of
// the attribute check is that it holds there.

// `strides` lost across a rebuild -- the stride-2 stem conv arriving as
// stride 1.
util.func public @strides_dropped(
    %input: tensor<1x?x?x3xf32>,
    %filter: tensor<3x3x3x16xf32>,
    %init: tensor<1x?x?x16xf32>) -> tensor<1x?x?x16xf32> {
  // expected-error @+1 {{rocket-verify-conv-shapes: linalg.conv_2d_nhwc_hwcf 'strides' changed since rocket-record-conv-attrs: recorded [2, 2], now [1, 1]}}
  %result = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>,
       rocket.recorded_attrs = {dilations = array<i64: 1, 1>, strides = array<i64: 2, 2>}}
      ins(%input, %filter : tensor<1x?x?x3xf32>, tensor<3x3x3x16xf32>)
      outs(%init : tensor<1x?x?x16xf32>) -> tensor<1x?x?x16xf32>
  util.return %result : tensor<1x?x?x16xf32>
}

// -----

// `indexing_maps` lost across a rebuild -- a transposed matmul arriving as
// an untransposed one.
util.func public @indexing_maps_dropped(
    %lhs: tensor<?x?xf32>,
    %rhs: tensor<?x?xf32>,
    %init: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error @+1 {{rocket-verify-conv-shapes: linalg.matmul 'indexing_maps' changed since rocket-record-conv-attrs}}
  %result = linalg.matmul
      {rocket.recorded_attrs = {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                                 affine_map<(d0, d1, d2) -> (d1, d2)>,
                                                 affine_map<(d0, d1, d2) -> (d0, d1)>]}}
      ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
  util.return %result : tensor<?x?xf32>
}

// -----

// `cast` compared on its effective value: an op with no `cast` is
// cast_signed, which is not the cast_unsigned it was recorded with.
util.func public @cast_dropped(
    %lhs: tensor<?x?xi8>,
    %rhs: tensor<?x?xi8>,
    %init: tensor<?x?xi32>) -> tensor<?x?xi32> {
  // expected-error @+1 {{rocket-verify-conv-shapes: linalg.matmul 'cast' changed since rocket-record-conv-attrs: recorded #linalg.type_fn<cast_unsigned>, now #linalg.type_fn<cast_signed>}}
  %result = linalg.matmul
      {rocket.recorded_attrs = {cast = #linalg.type_fn<cast_unsigned>}}
      ins(%lhs, %rhs : tensor<?x?xi8>, tensor<?x?xi8>)
      outs(%init : tensor<?x?xi32>) -> tensor<?x?xi32>
  util.return %result : tensor<?x?xi32>
}

// -----

// Under the mark, an op with no record at all is an error: whatever rebuilt
// it dropped its discardable attributes too, so nothing can be checked.
module attributes {rocket.conv_attrs_recorded} {
  util.func public @record_lost(
      %input: tensor<1x?x?x3xf32>,
      %filter: tensor<3x3x3x16xf32>,
      %init: tensor<1x?x?x16xf32>) -> tensor<1x?x?x16xf32> {
    // expected-error @+1 {{rocket-verify-conv-shapes: linalg.conv_2d_nhwc_hwcf carries no 'rocket.recorded_attrs'}}
    %result = linalg.conv_2d_nhwc_hwcf
        {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
        ins(%input, %filter : tensor<1x?x?x3xf32>, tensor<3x3x3x16xf32>)
        outs(%init : tensor<1x?x?x16xf32>) -> tensor<1x?x?x16xf32>
    util.return %result : tensor<1x?x?x16xf32>
  }
}

// -----

// The arithmetic check is still there for static extents: attributes
// intact, output type wrong. 8 - (3 - 1) - 1 + 1 = 6 rows, not 4.
util.func public @static_extent_mismatch(
    %input: tensor<1x8x8x3xf32>,
    %filter: tensor<3x3x3x16xf32>,
    %init: tensor<1x4x4x16xf32>) -> tensor<1x4x4x16xf32> {
  // expected-error @+1 {{rocket-verify-conv-shapes: linalg.conv_2d_nhwc_hwcf output extent 4 on spatial axis 0 disagrees with its operands: input 8, filter 3, stride 1, dilation 1 imply 6}}
  %result = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x8x8x3xf32>, tensor<3x3x3x16xf32>)
      outs(%init : tensor<1x4x4x16xf32>) -> tensor<1x4x4x16xf32>
  util.return %result : tensor<1x4x4x16xf32>
}

// -----

// Not an error: a default spelled explicitly is the same op. The record
// omits `strides` (all ones), the op carries `dense<1>`; no mark, so the
// unrecorded static conv beside it is only extent-checked.
util.func public @explicit_default_agrees(
    %input: tensor<1x?x?x3xf32>,
    %filter: tensor<3x3x3x16xf32>,
    %init: tensor<1x?x?x16xf32>) -> tensor<1x?x?x16xf32> {
  %result = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>,
       rocket.recorded_attrs = {dilations = array<i64: 1, 1>}}
      ins(%input, %filter : tensor<1x?x?x3xf32>, tensor<3x3x3x16xf32>)
      outs(%init : tensor<1x?x?x16xf32>) -> tensor<1x?x?x16xf32>
  util.return %result : tensor<1x?x?x16xf32>
}
