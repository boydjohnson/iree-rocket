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
// CHECK-SAME: : (i32, i32, i32, i32, i32, i32, i32, i32,
// CHECK: flow.dispatch.workgroups
// CHECK-SAME: stream.affinity = #hal.device.affinity<@cpu_device>

// CHECK: util.func private @call_rocket_conv2d_0
// CHECK: flow.dispatch @rocket_executable_0::@rocket_conv2d_v1_0::@rocket_conv2d_0

// This shape intentionally does not match a current Rocket hardware
// specialization. It isolates the layout-normalization stage that runs before
// dispatch selection.

// CHECK-LABEL: util.func public @regular_conv
// CHECK-NOT: linalg.conv_2d_nchw_fchw
// CHECK: linalg.transpose
// CHECK-SAME: permutation = [0, 2, 3, 1]
// CHECK: linalg.transpose
// CHECK-SAME: permutation = [2, 3, 1, 0]
// CHECK: linalg.conv_2d_nhwc_hwcf
// CHECK-SAME: tensor<1x8x8x3xf16>, tensor<3x3x3x4xf16>
// CHECK: linalg.transpose
// CHECK-SAME: permutation = [0, 3, 1, 2]
util.func public @regular_conv(
    %input: tensor<1x3x8x8xf16>,
    %filter: tensor<4x3x3x3xf16>,
    %init: tensor<1x4x6x6xf32>) -> tensor<1x4x6x6xf32> {
  %result = linalg.conv_2d_nchw_fchw {
      dilations = dense<1> : vector<2xi64>,
      strides = dense<1> : vector<2xi64>
    } ins(%input, %filter : tensor<1x3x8x8xf16>, tensor<4x3x3x3xf16>)
      outs(%init : tensor<1x4x6x6xf32>) -> tensor<1x4x6x6xf32>
  util.return %result : tensor<1x4x6x6xf32>
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

// Channels may also be dynamic. Filter height/width stay statically 1x1 so
// this fallback does not claim unsupported regular 3x3 convolutions.

// CHECK-LABEL: util.func public @fully_dynamic_nhwc_conv
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf
// CHECK: util.call @call_rocket_dynamic_conv2d
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
