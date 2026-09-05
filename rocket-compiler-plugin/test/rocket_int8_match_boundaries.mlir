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
//
// The dense int8 bounds moved twice on 2026-09-03: first from 352 (1x1) and
// 32 (3x3) to 512, then to the raised HAL ceilings -- 1x1 Cin 1344 / Cout
// 1792, 3x3 Cin 1152 (the coefficient working set binds before the channel
// rules at k=3). Those two numbers were containment for a DPU output-writer bug
// (`mc_surf_out`, see the transform spec's int8 section); with the writer and
// its readback corrected there is no coefficient-per-channel ceiling left,
// and both bounds are now the HAL's `MAX_INT8_INPUT_CHANNELS`. 513 falls back
// because the *channel padding* rules are unmeasured above 512, which is a
// different limit from the one that was lifted.

// CHECK-LABEL: util.func public @dense_1x1_cin_1344_matched
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf
// CHECK: flow.dispatch @rocket_dynamic_int8_executable
func.func @dense_1x1_cin_1344_matched(
    %input: tensor<1x4x4x1344xi8>,
    %filter: tensor<1x1x1344x64xi8>,
    %init: tensor<1x4x4x64xi32>) -> tensor<1x4x4x64xi32> {
  %result = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x4x4x1344xi8>, tensor<1x1x1344x64xi8>)
      outs(%init : tensor<1x4x4x64xi32>) -> tensor<1x4x4x64xi32>
  return %result : tensor<1x4x4x64xi32>
}

// CHECK-LABEL: util.func public @dense_1x1_cin_1345_falls_back
// CHECK-NOT: flow.dispatch @rocket_dynamic_int8_executable
// CHECK: linalg.conv_2d_nhwc_hwcf
func.func @dense_1x1_cin_1345_falls_back(
    %input: tensor<1x4x4x1345xi8>,
    %filter: tensor<1x1x1345x64xi8>,
    %init: tensor<1x4x4x64xi32>) -> tensor<1x4x4x64xi32> {
  %result = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x4x4x1345xi8>, tensor<1x1x1345x64xi8>)
      outs(%init : tensor<1x4x4x64xi32>) -> tensor<1x4x4x64xi32>
  return %result : tensor<1x4x4x64xi32>
}

// CHECK-LABEL: util.func public @dense_3x3_cin_1152_matched
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf
// CHECK: flow.dispatch @rocket_dynamic_int8_executable
func.func @dense_3x3_cin_1152_matched(
    %input: tensor<1x6x6x1152xi8>,
    %filter: tensor<3x3x1152x64xi8>,
    %init: tensor<1x4x4x64xi32>) -> tensor<1x4x4x64xi32> {
  %result = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x6x6x1152xi8>, tensor<3x3x1152x64xi8>)
      outs(%init : tensor<1x4x4x64xi32>) -> tensor<1x4x4x64xi32>
  return %result : tensor<1x4x4x64xi32>
}

// CHECK-LABEL: util.func public @dense_3x3_cin_1153_falls_back
// CHECK-NOT: flow.dispatch @rocket_dynamic_int8_executable
// CHECK: linalg.conv_2d_nhwc_hwcf
func.func @dense_3x3_cin_1153_falls_back(
    %input: tensor<1x6x6x1153xi8>,
    %filter: tensor<3x3x1153x64xi8>,
    %init: tensor<1x4x4x64xi32>) -> tensor<1x4x4x64xi32> {
  %result = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x6x6x1153xi8>, tensor<3x3x1153x64xi8>)
      outs(%init : tensor<1x4x4x64xi32>) -> tensor<1x4x4x64xi32>
  return %result : tensor<1x4x4x64xi32>
}

// CHECK-LABEL: util.func public @dense_1x1_cout_512_matched
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf
// CHECK: flow.dispatch @rocket_dynamic_int8_executable
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

// The 1x1 matcher's Cout bound is 768, not 512 -- it was raised when the
// vendor corpus reached that far. This pair was left at 512/513 and so
// asserted a fallback that has not happened for some time; it failed
// identically before and after the 2026-09-03 Cin change. Corrected to the
// real boundary.

// CHECK-LABEL: util.func public @dense_1x1_cout_1792_matched
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf
// CHECK: flow.dispatch @rocket_dynamic_int8_executable
func.func @dense_1x1_cout_1792_matched(
    %input: tensor<1x4x4x16xi8>,
    %filter: tensor<1x1x16x1792xi8>,
    %init: tensor<1x4x4x1792xi32>) -> tensor<1x4x4x1792xi32> {
  %result = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x4x4x16xi8>, tensor<1x1x16x1792xi8>)
      outs(%init : tensor<1x4x4x1792xi32>) -> tensor<1x4x4x1792xi32>
  return %result : tensor<1x4x4x1792xi32>
}

// CHECK-LABEL: util.func public @dense_1x1_cout_1793_falls_back
// CHECK-NOT: flow.dispatch @rocket_dynamic_int8_executable
// CHECK: linalg.conv_2d_nhwc_hwcf
func.func @dense_1x1_cout_1793_falls_back(
    %input: tensor<1x4x4x16xi8>,
    %filter: tensor<1x1x16x1793xi8>,
    %init: tensor<1x4x4x1793xi32>) -> tensor<1x4x4x1793xi32> {
  %result = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x4x4x16xi8>, tensor<1x1x16x1793xi8>)
      outs(%init : tensor<1x4x4x1793xi32>) -> tensor<1x4x4x1793xi32>
  return %result : tensor<1x4x4x1793xi32>
}

// CHECK-LABEL: util.func public @dense_3x3_cout_512_matched
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf
// CHECK: flow.dispatch @rocket_dynamic_int8_executable
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
// CHECK-NOT: flow.dispatch @rocket_dynamic_int8_executable
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

// CHECK-LABEL: util.func public @depthwise_3x3_channels_1344_matched
// CHECK-NOT: linalg.depthwise_conv_2d_nhwc_hwc
// CHECK: flow.dispatch @rocket_dynamic_depthwise_int8_executable
func.func @depthwise_3x3_channels_1344_matched(
    %input: tensor<1x6x6x1344xi8>,
    %filter: tensor<3x3x1344xi8>,
    %init: tensor<1x4x4x1344xi32>) -> tensor<1x4x4x1344xi32> {
  %result = linalg.depthwise_conv_2d_nhwc_hwc
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x6x6x1344xi8>, tensor<3x3x1344xi8>)
      outs(%init : tensor<1x4x4x1344xi32>) -> tensor<1x4x4x1344xi32>
  return %result : tensor<1x4x4x1344xi32>
}

// CHECK-LABEL: util.func public @depthwise_3x3_channels_1345_falls_back
// CHECK-NOT: flow.dispatch @rocket_dynamic_depthwise_int8_executable
// CHECK: linalg.depthwise_conv_2d_nhwc_hwc
func.func @depthwise_3x3_channels_1345_falls_back(
    %input: tensor<1x6x6x1345xi8>,
    %filter: tensor<3x3x1345xi8>,
    %init: tensor<1x4x4x1345xi32>) -> tensor<1x4x4x1345xi32> {
  %result = linalg.depthwise_conv_2d_nhwc_hwc
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x6x6x1345xi8>, tensor<3x3x1345xi8>)
      outs(%init : tensor<1x4x4x1345xi32>) -> tensor<1x4x4x1345xi32>
  return %result : tensor<1x4x4x1345xi32>
}
