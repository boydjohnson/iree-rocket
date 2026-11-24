// RUN: iree-opt %s --pass-pipeline='builtin.module(util.func(rocket-demote-conv-inputs-to-f16))' | FileCheck %s

// Depthwise demotion is an experiment gate (ROCKET_DEMOTE_DEPTHWISE), off by
// default because MobileNetV2's small 3x3 depthwise convolutions lose on the
// NPU. A 7x7 stride-1 one -- ConvNeXt's -- is demoted regardless: ~5x the
// work per dispatch, and ConvNeXt-T measured 113 -> 103 ms with all 18 on
// the NPU.

// CHECK-LABEL: util.func public @convnext_depthwise
// CHECK: linalg.depthwise_conv_2d_nchw_chw {{.*}}rocket.f16_demoted{{.*}} ins(%{{.+}}, %{{.+}} : tensor<1x96x62x62xf16>, tensor<96x7x7xf16>)
util.func public @convnext_depthwise(%x: tensor<1x96x62x62xf32>, %w: tensor<96x7x7xf32>, %init: tensor<1x96x56x56xf32>) -> tensor<1x96x56x56xf32> {
  %0 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%x, %w : tensor<1x96x62x62xf32>, tensor<96x7x7xf32>) outs(%init : tensor<1x96x56x56xf32>) -> tensor<1x96x56x56xf32>
  util.return %0 : tensor<1x96x56x56xf32>
}

// CHECK-LABEL: util.func public @nhwc_depthwise
// CHECK: linalg.depthwise_conv_2d_nhwc_hwc {{.*}}rocket.f16_demoted{{.*}} ins(%{{.+}}, %{{.+}} : tensor<1x20x20x384xf16>, tensor<7x7x384xf16>)
util.func public @nhwc_depthwise(%x: tensor<1x20x20x384xf32>, %w: tensor<7x7x384xf32>, %init: tensor<1x14x14x384xf32>) -> tensor<1x14x14x384xf32> {
  %0 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%x, %w : tensor<1x20x20x384xf32>, tensor<7x7x384xf32>) outs(%init : tensor<1x14x14x384xf32>) -> tensor<1x14x14x384xf32>
  util.return %0 : tensor<1x14x14x384xf32>
}

// A 3x3 depthwise (MobileNetV2's) and a strided 7x7 stay f32.
// CHECK-LABEL: util.func public @stays_f32
// CHECK-NOT: rocket.f16_demoted
// CHECK: util.return
util.func public @stays_f32(%x: tensor<1x32x18x18xf32>, %w3: tensor<32x3x3xf32>, %i3: tensor<1x32x16x16xf32>, %w7: tensor<32x7x7xf32>, %i7: tensor<1x32x6x6xf32>) -> (tensor<1x32x16x16xf32>, tensor<1x32x6x6xf32>) {
  %0 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%x, %w3 : tensor<1x32x18x18xf32>, tensor<32x3x3xf32>) outs(%i3 : tensor<1x32x16x16xf32>) -> tensor<1x32x16x16xf32>
  %1 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%x, %w7 : tensor<1x32x18x18xf32>, tensor<32x7x7xf32>) outs(%i7 : tensor<1x32x6x6xf32>) -> tensor<1x32x6x6xf32>
  util.return %0, %1 : tensor<1x32x16x16xf32>, tensor<1x32x6x6xf32>
}
