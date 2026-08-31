// RUN: iree-compile %s \
// RUN:   --iree-preprocessing-transform-spec-filename=%S/../target/Rocket/rocket_conv2d_transform_spec.mlir \
// RUN:   --iree-hal-target-device=rocket_device=rocket \
// RUN:   --iree-hal-target-device=cpu_device=local \
// RUN:   --iree-hal-local-target-device-backends=llvm-cpu \
// RUN:   --iree-llvmcpu-target-cpu=generic \
// RUN:   --iree-hal-default-device=cpu_device \
// RUN:   --iree-hal-indirect-command-buffers=false \
// RUN:   --compile-to=preprocessing \
// RUN:   --mlir-print-op-generic=false \
// RUN:   -o - | FileCheck %s

// Matcher-boundary coverage for the hardware-derived int8 limits. Each
// accepted shape has an immediately-adjacent rejected shape so widening a
// matcher cannot silently route a known-bad convolution to Rocket, and
// tightening one cannot silently lose the largest measured-good shape.

// CHECK-LABEL: util.func public @dense_1x1_cin_384_matched
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf
// CHECK: util.call @call_rocket_dynamic_conv2d_int8
func.func @dense_1x1_cin_384_matched(
    %input: tensor<1x4x4x384xi8>,
    %filter: tensor<1x1x384x64xi8>,
    %init: tensor<1x4x4x64xi32>) -> tensor<1x4x4x64xi32> {
  %result = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x4x4x384xi8>, tensor<1x1x384x64xi8>)
      outs(%init : tensor<1x4x4x64xi32>) -> tensor<1x4x4x64xi32>
  return %result : tensor<1x4x4x64xi32>
}

// CHECK-LABEL: util.func public @dense_1x1_cin_385_falls_back
// CHECK-NOT: util.call @call_rocket_dynamic_conv2d_int8
// CHECK: linalg.conv_2d_nhwc_hwcf
func.func @dense_1x1_cin_385_falls_back(
    %input: tensor<1x4x4x385xi8>,
    %filter: tensor<1x1x385x64xi8>,
    %init: tensor<1x4x4x64xi32>) -> tensor<1x4x4x64xi32> {
  %result = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x4x4x385xi8>, tensor<1x1x385x64xi8>)
      outs(%init : tensor<1x4x4x64xi32>) -> tensor<1x4x4x64xi32>
  return %result : tensor<1x4x4x64xi32>
}

// CHECK-LABEL: util.func public @dense_3x3_cin_32_matched
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf
// CHECK: util.call @call_rocket_dynamic_conv2d_int8
func.func @dense_3x3_cin_32_matched(
    %input: tensor<1x6x6x32xi8>,
    %filter: tensor<3x3x32x64xi8>,
    %init: tensor<1x4x4x64xi32>) -> tensor<1x4x4x64xi32> {
  %result = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x6x6x32xi8>, tensor<3x3x32x64xi8>)
      outs(%init : tensor<1x4x4x64xi32>) -> tensor<1x4x4x64xi32>
  return %result : tensor<1x4x4x64xi32>
}

// CHECK-LABEL: util.func public @dense_3x3_cin_33_falls_back
// CHECK-NOT: util.call @call_rocket_dynamic_conv2d_int8
// CHECK: linalg.conv_2d_nhwc_hwcf
func.func @dense_3x3_cin_33_falls_back(
    %input: tensor<1x6x6x33xi8>,
    %filter: tensor<3x3x33x64xi8>,
    %init: tensor<1x4x4x64xi32>) -> tensor<1x4x4x64xi32> {
  %result = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x6x6x33xi8>, tensor<3x3x33x64xi8>)
      outs(%init : tensor<1x4x4x64xi32>) -> tensor<1x4x4x64xi32>
  return %result : tensor<1x4x4x64xi32>
}

// CHECK-LABEL: util.func public @dense_1x1_cout_512_matched
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf
// CHECK: util.call @call_rocket_dynamic_conv2d_int8
func.func @dense_1x1_cout_512_matched(
    %input: tensor<1x4x4x16xi8>,
    %filter: tensor<1x1x16x512xi8>,
    %init: tensor<1x4x4x512xi32>) -> tensor<1x4x4x512xi32> {
  %result = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x4x4x16xi8>, tensor<1x1x16x512xi8>)
      outs(%init : tensor<1x4x4x512xi32>) -> tensor<1x4x4x512xi32>
  return %result : tensor<1x4x4x512xi32>
}

// CHECK-LABEL: util.func public @dense_1x1_cout_513_falls_back
// CHECK-NOT: util.call @call_rocket_dynamic_conv2d_int8
// CHECK: linalg.conv_2d_nhwc_hwcf
func.func @dense_1x1_cout_513_falls_back(
    %input: tensor<1x4x4x16xi8>,
    %filter: tensor<1x1x16x513xi8>,
    %init: tensor<1x4x4x513xi32>) -> tensor<1x4x4x513xi32> {
  %result = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x4x4x16xi8>, tensor<1x1x16x513xi8>)
      outs(%init : tensor<1x4x4x513xi32>) -> tensor<1x4x4x513xi32>
  return %result : tensor<1x4x4x513xi32>
}

// CHECK-LABEL: util.func public @dense_3x3_cout_512_matched
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf
// CHECK: util.call @call_rocket_dynamic_conv2d_int8
func.func @dense_3x3_cout_512_matched(
    %input: tensor<1x6x6x16xi8>,
    %filter: tensor<3x3x16x512xi8>,
    %init: tensor<1x4x4x512xi32>) -> tensor<1x4x4x512xi32> {
  %result = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x6x6x16xi8>, tensor<3x3x16x512xi8>)
      outs(%init : tensor<1x4x4x512xi32>) -> tensor<1x4x4x512xi32>
  return %result : tensor<1x4x4x512xi32>
}

// CHECK-LABEL: util.func public @dense_3x3_cout_513_falls_back
// CHECK-NOT: util.call @call_rocket_dynamic_conv2d_int8
// CHECK: linalg.conv_2d_nhwc_hwcf
func.func @dense_3x3_cout_513_falls_back(
    %input: tensor<1x6x6x16xi8>,
    %filter: tensor<3x3x16x513xi8>,
    %init: tensor<1x4x4x513xi32>) -> tensor<1x4x4x513xi32> {
  %result = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x6x6x16xi8>, tensor<3x3x16x513xi8>)
      outs(%init : tensor<1x4x4x513xi32>) -> tensor<1x4x4x513xi32>
  return %result : tensor<1x4x4x513xi32>
}

// Depthwise has one shared channel bound because Cout is structurally Cin.

// CHECK-LABEL: util.func public @depthwise_3x3_channels_512_matched
// CHECK-NOT: linalg.depthwise_conv_2d_nhwc_hwc
// CHECK: util.call @call_rocket_dynamic_depthwise_conv2d_int8
func.func @depthwise_3x3_channels_512_matched(
    %input: tensor<1x6x6x512xi8>,
    %filter: tensor<3x3x512xi8>,
    %init: tensor<1x4x4x512xi32>) -> tensor<1x4x4x512xi32> {
  %result = linalg.depthwise_conv_2d_nhwc_hwc
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x6x6x512xi8>, tensor<3x3x512xi8>)
      outs(%init : tensor<1x4x4x512xi32>) -> tensor<1x4x4x512xi32>
  return %result : tensor<1x4x4x512xi32>
}

// CHECK-LABEL: util.func public @depthwise_3x3_channels_513_falls_back
// CHECK-NOT: util.call @call_rocket_dynamic_depthwise_conv2d_int8
// CHECK: linalg.depthwise_conv_2d_nhwc_hwc
func.func @depthwise_3x3_channels_513_falls_back(
    %input: tensor<1x6x6x513xi8>,
    %filter: tensor<3x3x513xi8>,
    %init: tensor<1x4x4x513xi32>) -> tensor<1x4x4x513xi32> {
  %result = linalg.depthwise_conv_2d_nhwc_hwc
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x6x6x513xi8>, tensor<3x3x513xi8>)
      outs(%init : tensor<1x4x4x513xi32>) -> tensor<1x4x4x513xi32>
  return %result : tensor<1x4x4x513xi32>
}
