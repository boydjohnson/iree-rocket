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

// The 528 boundary is the largest output width validated for the FP16
// stride-1 pointwise matcher: MobileNetV2's 14x14, 88-to-528 layers passed
// the hardware oracle for counting, selector, and dense inputs.  Keep the
// adjacent 529 shape on CPU until it has equivalent hardware coverage.

// CHECK-LABEL: util.func public @mobilenetv2_1x1_cout_528_matched
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf
// CHECK: util.call @call_rocket_dynamic_conv2d
func.func @mobilenetv2_1x1_cout_528_matched(
    %input: tensor<1x?x?x88xf16>,
    %filter: tensor<1x1x88x528xf16>,
    %init: tensor<1x?x?x528xf32>) -> tensor<1x?x?x528xf32> {
  %result = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x?x?x88xf16>, tensor<1x1x88x528xf16>)
      outs(%init : tensor<1x?x?x528xf32>) -> tensor<1x?x?x528xf32>
  return %result : tensor<1x?x?x528xf32>
}

// CHECK-LABEL: util.func public @mobilenetv2_1x1_cout_529_falls_back
// CHECK-NOT: util.call @call_rocket_dynamic_conv2d
// CHECK: linalg.conv_2d_nhwc_hwcf
func.func @mobilenetv2_1x1_cout_529_falls_back(
    %input: tensor<1x?x?x88xf16>,
    %filter: tensor<1x1x88x529xf16>,
    %init: tensor<1x?x?x529xf32>) -> tensor<1x?x?x529xf32> {
  %result = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x?x?x88xf16>, tensor<1x1x88x529xf16>)
      outs(%init : tensor<1x?x?x529xf32>) -> tensor<1x?x?x529xf32>
  return %result : tensor<1x?x?x529xf32>
}
