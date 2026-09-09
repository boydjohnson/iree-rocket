// RUN: iree-opt %s --mlir-print-local-scope --pass-pipeline='builtin.module(rocket-record-conv-attrs,rocket-demote-conv-inputs-to-f16,rocket-verify-conv-shapes)' \
// RUN:   | FileCheck %s

// The record/verify pair around the plugin's own demotion, on the shapes the
// old extent check could not see: a symbolic-extent strided convolution and
// a symbolic-extent transposed matmul. The demote rebuilds both ops and
// carries `strides`/`dilations`/`indexing_maps` across, so verify is
// silent, and it strips the record and the mark on its way out -- nothing
// downstream (the DAG matchers compare whole attribute dictionaries) may see
// either.

// CHECK-NOT: rocket.conv_attrs_recorded

// CHECK-LABEL: util.func public @dynamic_strided_conv
// CHECK: linalg.conv_2d_nhwc_hwcf
// CHECK-SAME: dilations = dense<1>
// CHECK-SAME: strides = dense<2>
// CHECK-SAME: ins(%{{.+}}, %{{.+}} : tensor<1x?x?x3xf16>, tensor<3x3x3x16xf16>)
// CHECK-NOT: rocket.recorded_attrs
util.func public @dynamic_strided_conv(
    %input: tensor<1x?x?x3xf32>,
    %filter: tensor<3x3x3x16xf32>,
    %init: tensor<1x?x?x16xf32>) -> tensor<1x?x?x16xf32> {
  %result = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
      ins(%input, %filter : tensor<1x?x?x3xf32>, tensor<3x3x3x16xf32>)
      outs(%init : tensor<1x?x?x16xf32>) -> tensor<1x?x?x16xf32>
  util.return %result : tensor<1x?x?x16xf32>
}

// CHECK-LABEL: util.func public @dynamic_transposed_matmul
// CHECK: linalg.matmul
// CHECK-SAME: indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>]
// CHECK-SAME: ins(%{{.+}}, %{{.+}} : tensor<?x?xf16>, tensor<?x?xf16>)
// CHECK-NOT: rocket.recorded_attrs
util.func public @dynamic_transposed_matmul(
    %lhs: tensor<?x?xf32>,
    %rhs: tensor<?x?xf32>,
    %init: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %result = linalg.matmul
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>]
      ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
  util.return %result : tensor<?x?xf32>
}

// An op the demote leaves alone (already f16) keeps its record until verify
// strips it; a healthy static conv still passes the extent check too.
// CHECK-LABEL: util.func public @static_f16_conv
// CHECK: linalg.conv_2d_nhwc_hwcf
// CHECK-SAME: strides = dense<2>
// CHECK-NOT: rocket.recorded_attrs
util.func public @static_f16_conv(
    %input: tensor<1x225x225x3xf16>,
    %filter: tensor<3x3x3x16xf16>,
    %init: tensor<1x112x112x16xf32>) -> tensor<1x112x112x16xf32> {
  %result = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
      ins(%input, %filter : tensor<1x225x225x3xf16>, tensor<3x3x3x16xf16>)
      outs(%init : tensor<1x112x112x16xf32>) -> tensor<1x112x112x16xf32>
  util.return %result : tensor<1x112x112x16xf32>
}
