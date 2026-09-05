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

// A supported Rocket shape proves that normalization happens before
// matching: the source is NCHW/FCHW, but the existing NHWC/HWCF dynamic
// specialization sees it (after channels-last normalization converts it)
// and replaces it with the Rocket dispatch, then transposes the result back
// to NCHW to match this function's own declared return type.

// The adapter's own shape is checked here rather than on a
// `util.func private @call_rocket_dynamic_conv2d`: @__transform_main inlines
// the wrappers, so everything the adapter builds -- the tensor.dim/index_cast
// push constants and the CPU-side accumulate epilogue -- now lands in the
// function that used to call it.
// CHECK-LABEL: util.func public @supported_nchw_conv
// CHECK-NOT: linalg.conv_2d_nchw_fchw
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf
// CHECK: flow.dispatch @rocket_dynamic_executable::@rocket_dynamic_conv2d_v1::@rocket_dynamic_conv2d(
// One i32 push constant per settable Conv2D dimension. Six, not eight: the
// runtime derives output_width/output_height rather than accepting them.
// CHECK-SAME: : (i32, i32, i32, i32, i32, i32,
// CHECK: flow.dispatch.workgroups
// CHECK-SAME: stream.affinity = #hal.device.affinity<@cpu_device>
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
// CHECK: flow.dispatch @rocket_dynamic_executable
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
// CHECK-NOT: flow.dispatch @rocket_dynamic_executable
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
// CHECK: flow.dispatch @rocket_dynamic_executable
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
// CHECK-NOT: flow.dispatch @rocket_dynamic_executable
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

// Depthwise's filter has no Cout dimension at all (one filter per input
// channel, channel multiplier one): tensor<1x1x32xf16>, not
// tensor<1x1x32x16xf16> the way the dense case above is. Cout is always
// Cin, matched structurally rather than bounded independently.

// CHECK-LABEL: util.func public @dynamic_nhwc_depthwise_conv
// CHECK-NOT: linalg.depthwise_conv_2d_nhwc_hwc
// CHECK: flow.dispatch @rocket_dynamic_depthwise_executable
util.func public @dynamic_nhwc_depthwise_conv(
    %input: tensor<1x?x?x32xf16>,
    %filter: tensor<1x1x32xf16>,
    %init: tensor<1x?x?x32xf32>) -> tensor<1x?x?x32xf32> {
  %result = linalg.depthwise_conv_2d_nhwc_hwc {
      dilations = dense<1> : tensor<2xi64>,
      strides = dense<1> : tensor<2xi64>
    } ins(%input, %filter : tensor<1x?x?x32xf16>, tensor<1x1x32xf16>)
      outs(%init : tensor<1x?x?x32xf32>) -> tensor<1x?x?x32xf32>
  util.return %result : tensor<1x?x?x32xf32>
}

// 3x3 goes down the same runtime-dimension fallback as 1x1, mirroring the
// dense case: ConvPlan gives both extents the identical demand-based CBUF
// partition.

// CHECK-LABEL: util.func public @dynamic_3x3_nhwc_depthwise_conv
// CHECK-NOT: linalg.depthwise_conv_2d_nhwc_hwc
// CHECK: flow.dispatch @rocket_dynamic_depthwise_executable
util.func public @dynamic_3x3_nhwc_depthwise_conv(
    %input: tensor<1x?x?x32xf16>,
    %filter: tensor<3x3x32xf16>,
    %init: tensor<1x?x?x32xf32>) -> tensor<1x?x?x32xf32> {
  %result = linalg.depthwise_conv_2d_nhwc_hwc {
      dilations = dense<1> : tensor<2xi64>,
      strides = dense<1> : tensor<2xi64>
    } ins(%input, %filter : tensor<1x?x?x32xf16>, tensor<3x3x32xf16>)
      outs(%init : tensor<1x?x?x32xf32>) -> tensor<1x?x?x32xf32>
  util.return %result : tensor<1x?x?x32xf32>
}

// Unlike dense conv, IREE's channels-last preprocessing pass never converts
// a depthwise NCHW op to NHWC (ConvertConvToChannelsLast.cpp's
// transposeConvLikeLinalgOp bails whenever ConvolutionDimensions::depth is
// non-empty) -- and this is the form a real ONNX-imported model actually
// produces, not linalg.depthwise_conv_2d_nhwc_hwc above. So there is a
// second, NCHW-native matcher rather than a shared normalization stage: it
// claims linalg.depthwise_conv_2d_nchw_chw directly and transposes the
// input/output feature maps around the same Rocket dispatch the NHWC
// matcher uses. The filter needs no transpose -- [c][kh][kw] here is
// already what the driver's depthwise weight packer expects.

// CHECK-LABEL: util.func public @dynamic_nchw_depthwise_conv
// CHECK-NOT: linalg.depthwise_conv_2d_nchw_chw
// The NCHW matcher reuses the NHWC executable; only the transposes around it
// differ, which is why the marker below is the same one @dynamic_nhwc_
// depthwise_conv checks.
// CHECK: flow.dispatch @rocket_dynamic_depthwise_executable
util.func public @dynamic_nchw_depthwise_conv(
    %input: tensor<1x32x?x?xf16>,
    %filter: tensor<32x3x3xf16>,
    %init: tensor<1x32x?x?xf32>) -> tensor<1x32x?x?xf32> {
  %result = linalg.depthwise_conv_2d_nchw_chw {
      dilations = dense<1> : tensor<2xi64>,
      strides = dense<1> : tensor<2xi64>
    } ins(%input, %filter : tensor<1x32x?x?xf16>, tensor<32x3x3xf16>)
      outs(%init : tensor<1x32x?x?xf32>) -> tensor<1x32x?x?xf32>
  util.return %result : tensor<1x32x?x?xf32>
}

// A 5x5 depthwise kernel is not claimed, mirroring the dense 5x5 case above:
// no fallback executable states the extra conditions this extent would need.

// CHECK-LABEL: util.func public @dynamic_5x5_nhwc_depthwise_conv
// CHECK-NOT: flow.dispatch @rocket_dynamic_depthwise_executable
// CHECK: linalg.depthwise_conv_2d_nhwc_hwc
util.func public @dynamic_5x5_nhwc_depthwise_conv(
    %input: tensor<1x?x?x32xf16>,
    %filter: tensor<5x5x32xf16>,
    %init: tensor<1x?x?x32xf32>) -> tensor<1x?x?x32xf32> {
  %result = linalg.depthwise_conv_2d_nhwc_hwc {
      dilations = dense<1> : tensor<2xi64>,
      strides = dense<1> : tensor<2xi64>
    } ins(%input, %filter : tensor<1x?x?x32xf16>, tensor<5x5x32xf16>)
      outs(%init : tensor<1x?x?x32xf32>) -> tensor<1x?x?x32xf32>
  util.return %result : tensor<1x?x?x32xf32>
}

// An unbounded channel count cannot be shown to fit the hardware's 512
// limit, mirroring @fully_dynamic_nhwc_conv above -- depthwise only has one
// channel count to bound (Cout is always Cin), but the same reasoning
// applies to it.

// CHECK-LABEL: util.func public @fully_dynamic_nhwc_depthwise_conv
// CHECK-NOT: flow.dispatch @rocket_dynamic_depthwise_executable
// CHECK: linalg.depthwise_conv_2d_nhwc_hwc
util.func public @fully_dynamic_nhwc_depthwise_conv(
    %input: tensor<1x?x?x?xf16>,
    %filter: tensor<3x3x?xf16>,
    %init: tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32> {
  %result = linalg.depthwise_conv_2d_nhwc_hwc {
      dilations = dense<1> : tensor<2xi64>,
      strides = dense<1> : tensor<2xi64>
    } ins(%input, %filter : tensor<1x?x?x?xf16>, tensor<3x3x?xf16>)
      outs(%init : tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32>
  util.return %result : tensor<1x?x?x?xf32>
}

// A channel multiplier above one (_hwcm, not _hwc -- note the filter's extra
// trailing dimension and the output's extra rank) is never claimed.
// ConvPlan::with_depthwise hard-asserts a channel multiplier of one; a real
// multiplier has never been captured or validated on this hardware, so it
// must stay on CPU regardless of how small or dynamic the shape is.

// CHECK-LABEL: util.func public @depthwise_channel_multiplier_conv
// CHECK-NOT: flow.dispatch @rocket_dynamic_depthwise_executable
// CHECK: linalg.depthwise_conv_2d_nhwc_hwcm
util.func public @depthwise_channel_multiplier_conv(
    %input: tensor<1x10x10x8xf16>,
    %filter: tensor<3x3x8x2xf16>,
    %init: tensor<1x8x8x8x2xf32>) -> tensor<1x8x8x8x2xf32> {
  %result = linalg.depthwise_conv_2d_nhwc_hwcm {
      dilations = dense<1> : tensor<2xi64>,
      strides = dense<1> : tensor<2xi64>
    } ins(%input, %filter : tensor<1x10x10x8xf16>, tensor<3x3x8x2xf16>)
      outs(%init : tensor<1x8x8x8x2xf32>) -> tensor<1x8x8x8x2xf32>
  util.return %result : tensor<1x8x8x8x2xf32>
}
