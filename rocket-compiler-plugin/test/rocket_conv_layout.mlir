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
// RUN: iree-compile %s \
// RUN:   --iree-preprocessing-transform-spec-filename=%S/../target/Rocket/rocket_conv2d_transform_spec.mlir \
// RUN:   --iree-hal-target-device=rocket_device=rocket \
// RUN:   --iree-hal-target-device=cpu_device=local \
// RUN:   --iree-hal-local-target-device-backends=llvm-cpu \
// RUN:   --iree-llvmcpu-target-cpu=generic \
// RUN:   --iree-hal-default-device=cpu_device \
// RUN:   --iree-hal-indirect-command-buffers=false \
// RUN:   -o %t.vmfb

// CHECK-LABEL: util.func private @call_rocket_dynamic_conv2d(
// CHECK: tensor.dim
// CHECK: arith.index_cast
// CHECK: flow.dispatch @rocket_dynamic_executable::@rocket_dynamic_conv2d_v1::@rocket_dynamic_conv2d(
// One i32 push constant per settable Conv2D dimension. Six, not eight: the
// runtime derives output_width/output_height rather than accepting them.
// CHECK-SAME: : (i32, i32, i32, i32, i32, i32,
// CHECK: flow.dispatch.workgroups
// CHECK-SAME: stream.affinity = #hal.device.affinity<@cpu_device>

// CHECK: util.func private @call_rocket_conv2d_0
// CHECK: flow.dispatch @rocket_executable_0::@rocket_conv2d_v1_0::@rocket_conv2d_0

// This shape intentionally does not match a current Rocket hardware
// specialization. It isolates the layout-normalization stage that runs before
// dispatch selection. The kernel is 5x5 rather than 3x3 precisely because
// both 1x1 and 3x3 now have a fallback that would claim it.

// CHECK-LABEL: util.func public @regular_conv
// CHECK-NOT: linalg.conv_2d_nchw_fchw
// CHECK: linalg.transpose
// CHECK-SAME: permutation = [0, 2, 3, 1]
// CHECK: linalg.transpose
// CHECK-SAME: permutation = [2, 3, 1, 0]
// CHECK: linalg.conv_2d_nhwc_hwcf
// CHECK-SAME: tensor<1x8x8x3xf16>, tensor<5x5x3x4xf16>
// CHECK: linalg.transpose
// CHECK-SAME: permutation = [0, 3, 1, 2]
util.func public @regular_conv(
    %input: tensor<1x3x8x8xf16>,
    %filter: tensor<4x3x5x5xf16>,
    %init: tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32> {
  %result = linalg.conv_2d_nchw_fchw {
      dilations = dense<1> : vector<2xi64>,
      strides = dense<1> : vector<2xi64>
    } ins(%input, %filter : tensor<1x3x8x8xf16>, tensor<4x3x5x5xf16>)
      outs(%init : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
  util.return %result : tensor<1x4x4x4xf32>
}

// A supported Rocket shape proves that normalization happens before matching:
// the source is NCHW/FCHW, but the existing NHWC/HWCF specialization sees it
// and replaces it with the Rocket dispatch.

// CHECK-LABEL: util.func public @supported_nchw_conv
// CHECK-NOT: linalg.conv_2d_nchw_fchw
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf
// CHECK: util.call @call_rocket_conv2d_0
// CHECK: linalg.transpose
// CHECK-SAME: permutation = [0, 3, 1, 2]
util.func public @supported_nchw_conv(
    %input: tensor<1x32x112x112xf16>,
    %filter: tensor<16x32x1x1xf16>,
    %init: tensor<1x16x112x112xf32>) -> tensor<1x16x112x112xf32> {
  %result = linalg.conv_2d_nchw_fchw {
      dilations = dense<1> : vector<2xi64>,
      strides = dense<1> : vector<2xi64>
    } ins(%input, %filter
      : tensor<1x32x112x112xf16>, tensor<16x32x1x1xf16>)
      outs(%init : tensor<1x16x112x112xf32>)
      -> tensor<1x16x112x112xf32>
  util.return %result : tensor<1x16x112x112xf32>
}

// Dynamic spatial dimensions cannot select one of the literal-shape
// executables above. They instead use the runtime-shape adapter, which obtains
// dimensions with tensor.dim and passes them as i32 dispatch constants.

// CHECK-LABEL: util.func public @dynamic_nhwc_conv
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf
// CHECK: util.call @call_rocket_dynamic_conv2d
util.func public @dynamic_nhwc_conv(
    %input: tensor<1x?x?x32xf16>,
    %filter: tensor<1x1x32x16xf16>,
    %init: tensor<1x?x?x16xf32>) -> tensor<1x?x?x16xf32> {
  %result = linalg.conv_2d_nhwc_hwcf {
      dilations = dense<1> : vector<2xi64>,
      strides = dense<1> : vector<2xi64>
    } ins(%input, %filter
      : tensor<1x?x?x32xf16>, tensor<1x1x32x16xf16>)
      outs(%init : tensor<1x?x?x16xf32>)
      -> tensor<1x?x?x16xf32>
  util.return %result : tensor<1x?x?x16xf32>
}

// Spatial dimensions may be dynamic, but channel counts may not: the Rocket
// convolution accepts at most 512 input and 512 output channels, and an
// unbounded `?` cannot be shown to fit. Since a dispatch that turns out to
// exceed the bound fails outright rather than falling back, this stays on the
// CPU.

// CHECK-LABEL: util.func public @fully_dynamic_nhwc_conv
// CHECK-NOT: util.call @call_rocket_dynamic_conv2d
// CHECK: linalg.conv_2d_nhwc_hwcf
util.func public @fully_dynamic_nhwc_conv(
    %input: tensor<1x?x?x?xf16>,
    %filter: tensor<1x1x?x?xf16>,
    %init: tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32> {
  %result = linalg.conv_2d_nhwc_hwcf {
      dilations = dense<1> : vector<2xi64>,
      strides = dense<1> : vector<2xi64>
    } ins(%input, %filter
      : tensor<1x?x?x?xf16>, tensor<1x1x?x?xf16>)
      outs(%init : tensor<1x?x?x?xf32>)
      -> tensor<1x?x?x?xf32>
  util.return %result : tensor<1x?x?x?xf32>
}

// 3x3 goes down the same runtime-dimension fallback as 1x1: ConvPlan gives
// both extents the identical demand-based CBUF partition. Spatial dimensions
// stay dynamic; the channel counts are static and within the 512 bound.

// CHECK-LABEL: util.func public @dynamic_3x3_nhwc_conv
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf
// CHECK: util.call @call_rocket_dynamic_conv2d
util.func public @dynamic_3x3_nhwc_conv(
    %input: tensor<1x?x?x32xf16>,
    %filter: tensor<3x3x32x16xf16>,
    %init: tensor<1x?x?x16xf32>) -> tensor<1x?x?x16xf32> {
  %result = linalg.conv_2d_nhwc_hwcf {
      dilations = dense<1> : vector<2xi64>,
      strides = dense<1> : vector<2xi64>
    } ins(%input, %filter
      : tensor<1x?x?x32xf16>, tensor<3x3x32x16xf16>)
      outs(%init : tensor<1x?x?x16xf32>)
      -> tensor<1x?x?x16xf32>
  util.return %result : tensor<1x?x?x16xf32>
}

// A 5x5 kernel is not claimed. The hardware supports it, but only under the
// extra fp16/stride-1 conditions of conv.rs's assert_large_kernel_plan_case,
// and no fallback executable states them.

// CHECK-LABEL: util.func public @dynamic_5x5_nhwc_conv
// CHECK-NOT: util.call @call_rocket_dynamic_conv2d
// CHECK: linalg.conv_2d_nhwc_hwcf
util.func public @dynamic_5x5_nhwc_conv(
    %input: tensor<1x?x?x32xf16>,
    %filter: tensor<5x5x32x16xf16>,
    %init: tensor<1x?x?x16xf32>) -> tensor<1x?x?x16xf32> {
  %result = linalg.conv_2d_nhwc_hwcf {
      dilations = dense<1> : vector<2xi64>,
      strides = dense<1> : vector<2xi64>
    } ins(%input, %filter
      : tensor<1x?x?x32xf16>, tensor<5x5x32x16xf16>)
      outs(%init : tensor<1x?x?x16xf32>)
      -> tensor<1x?x?x16xf32>
  util.return %result : tensor<1x?x?x16xf32>
}
