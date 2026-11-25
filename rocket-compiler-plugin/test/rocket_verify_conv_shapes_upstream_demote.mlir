// RUN: iree-opt %s --pass-pipeline='builtin.module(rocket-record-conv-attrs,iree-global-opt-demote-contraction-inputs{type=f16 operation=conv},rocket-verify-conv-shapes)' 2>&1 \
// RUN:   | FileCheck %s

// The regression the tripwire exists for, on the shape it used to be blind
// to. `iree-global-opt-demote-contraction-inputs` rebuilds a named
// convolution through linalg::getPrunedAttributeList, which elides
// `strides` and `dilations`, so a stride-2 convolution comes out stride 1 --
// the bug that once read a 114x114 corner of MobileNetV2's stem input
// (RocketDemoteConvInputsPass.cpp). With every extent symbolic there is no
// output extent to re-derive, so the old arithmetic check would have passed
// this; the attribute record does not.
//
// IREE 3.14 rewrote the demote pass (DemoteContractionInputs.cpp) and it
// keeps `strides` now, so this test flipped from expecting the tripwire's
// error to expecting a clean pass: if a later IREE drops the attribute
// again, the CHECK-NOT below is what turns red. The tripwire itself is
// covered on a hand-built mismatch by rocket_verify_conv_shapes.mlir.

// CHECK-NOT: error
// CHECK: linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
// CHECK-SAME: tensor<1x?x?x3xf16>, tensor<3x3x3x16xf16>

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
