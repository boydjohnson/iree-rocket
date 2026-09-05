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

// Boundary coverage for the fp16 dense matchers. Each accepted shape has an
// immediately-adjacent rejected one, so widening a matcher cannot silently
// route an uncharacterized convolution to Rocket and tightening one cannot
// silently lose the largest measured-good shape.
//
// Raised 2026-09-03 from Cin 512 / Cout 528 to the HAL's `MAX_INPUT_CHANNELS`
// 1344 and `MAX_OUTPUT_CHANNELS` 1792, on board evidence (fp16 k=1 exact at
// 14x14 for Cin 256..1792 and at 7x7 Cin 448 for Cout 528..2048) plus the
// fp16 vendor corpus in `conv_vendor_fixture_wide.rs`. The 3x3 matcher stops
// at Cin 1152 instead: at k=3 the coefficient working set binds first and
// `ConvPlan` refuses Cin >= 1216 outright.

// CHECK-LABEL: util.func public @dense_1x1_cin_1344_matched
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf
// CHECK: flow.dispatch @rocket_dynamic_executable
func.func @dense_1x1_cin_1344_matched(
    %input: tensor<1x?x?x1344xf16>,
    %filter: tensor<1x1x1344x64xf16>,
    %init: tensor<1x?x?x64xf32>) -> tensor<1x?x?x64xf32> {
  %result = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x?x?x1344xf16>, tensor<1x1x1344x64xf16>)
      outs(%init : tensor<1x?x?x64xf32>) -> tensor<1x?x?x64xf32>
  return %result : tensor<1x?x?x64xf32>
}

// CHECK-LABEL: util.func public @dense_1x1_cin_1345_falls_back
// CHECK-NOT: flow.dispatch @rocket_dynamic_executable
// CHECK: linalg.conv_2d_nhwc_hwcf
func.func @dense_1x1_cin_1345_falls_back(
    %input: tensor<1x?x?x1345xf16>,
    %filter: tensor<1x1x1345x64xf16>,
    %init: tensor<1x?x?x64xf32>) -> tensor<1x?x?x64xf32> {
  %result = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x?x?x1345xf16>, tensor<1x1x1345x64xf16>)
      outs(%init : tensor<1x?x?x64xf32>) -> tensor<1x?x?x64xf32>
  return %result : tensor<1x?x?x64xf32>
}

// CHECK-LABEL: util.func public @dense_1x1_cout_1792_matched
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf
// CHECK: flow.dispatch @rocket_dynamic_executable
func.func @dense_1x1_cout_1792_matched(
    %input: tensor<1x?x?x448xf16>,
    %filter: tensor<1x1x448x1792xf16>,
    %init: tensor<1x?x?x1792xf32>) -> tensor<1x?x?x1792xf32> {
  %result = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x?x?x448xf16>, tensor<1x1x448x1792xf16>)
      outs(%init : tensor<1x?x?x1792xf32>) -> tensor<1x?x?x1792xf32>
  return %result : tensor<1x?x?x1792xf32>
}

// CHECK-LABEL: util.func public @dense_1x1_cout_1793_falls_back
// CHECK-NOT: flow.dispatch @rocket_dynamic_executable
// CHECK: linalg.conv_2d_nhwc_hwcf
func.func @dense_1x1_cout_1793_falls_back(
    %input: tensor<1x?x?x448xf16>,
    %filter: tensor<1x1x448x1793xf16>,
    %init: tensor<1x?x?x1793xf32>) -> tensor<1x?x?x1793xf32> {
  %result = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x?x?x448xf16>, tensor<1x1x448x1793xf16>)
      outs(%init : tensor<1x?x?x1793xf32>) -> tensor<1x?x?x1793xf32>
  return %result : tensor<1x?x?x1793xf32>
}

// CHECK-LABEL: util.func public @dense_3x3_cin_1152_matched
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf
// CHECK: flow.dispatch @rocket_dynamic_executable
func.func @dense_3x3_cin_1152_matched(
    %input: tensor<1x?x?x1152xf16>,
    %filter: tensor<3x3x1152x64xf16>,
    %init: tensor<1x?x?x64xf32>) -> tensor<1x?x?x64xf32> {
  %result = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x?x?x1152xf16>, tensor<3x3x1152x64xf16>)
      outs(%init : tensor<1x?x?x64xf32>) -> tensor<1x?x?x64xf32>
  return %result : tensor<1x?x?x64xf32>
}

// CHECK-LABEL: util.func public @dense_3x3_cin_1153_falls_back
// CHECK-NOT: flow.dispatch @rocket_dynamic_executable
// CHECK: linalg.conv_2d_nhwc_hwcf
func.func @dense_3x3_cin_1153_falls_back(
    %input: tensor<1x?x?x1153xf16>,
    %filter: tensor<3x3x1153x64xf16>,
    %init: tensor<1x?x?x64xf32>) -> tensor<1x?x?x64xf32> {
  %result = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x?x?x1153xf16>, tensor<3x3x1153x64xf16>)
      outs(%init : tensor<1x?x?x64xf32>) -> tensor<1x?x?x64xf32>
  return %result : tensor<1x?x?x64xf32>
}
