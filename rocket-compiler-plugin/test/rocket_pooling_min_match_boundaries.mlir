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

// Boundary coverage for the two min-pool matchers, on the same contract as
// its max and average neighbours: every accepted shape has an immediately
// adjacent rejected one.
//
// There are only two matchers, not four, because linalg defines
// `pooling_nchw_max` but no `pooling_nchw_min` -- there is no NCHW min op for
// a matcher to claim. ONNX has no MinPool operator at all, which is
// presumably why the dialect never grew one.
//
// The zero padding these executables bake is load-bearing here in a way it is
// not for max. `PoolingMethod::pad_fill_value` has no measured identity for
// min at *any* precision, so `required_pad_fill` returns None for a padded
// min and the driver refuses the executable outright. An unpadded pool never
// reads the field, and the driver derives `padded` from the executable's own
// pad fields, so tiling a wide input cannot reintroduce it.

// ---------------------------------------------------------------- accepted

// CHECK-LABEL: util.func public @min_nhwc_s2
// CHECK-NOT: linalg.pooling_nhwc_min
// CHECK: flow.dispatch @rocket_pooling_min_executable_s2
util.func public @min_nhwc_s2(
    %input: tensor<1x14x14x64xf32>,
    %init: tensor<1x7x7x64xf32>) -> tensor<1x7x7x64xf32> {
  %window = tensor.empty() : tensor<2x2xf32>
  %result = linalg.pooling_nhwc_min {
      dilations = dense<1> : vector<2xi64>,
      strides = dense<2> : vector<2xi64>
    } ins(%input, %window : tensor<1x14x14x64xf32>, tensor<2x2xf32>)
      outs(%init : tensor<1x7x7x64xf32>) -> tensor<1x7x7x64xf32>
  util.return %result : tensor<1x7x7x64xf32>
}

// Stride 1 routes to the stride-1 executable: stride is baked, one per value,
// so picking the wrong one is a wrong answer rather than a decline.
// CHECK-LABEL: util.func public @min_nhwc_s1
// CHECK-NOT: @rocket_pooling_min_executable_s2
// CHECK: flow.dispatch @rocket_pooling_min_executable
util.func public @min_nhwc_s1(
    %input: tensor<1x8x8x64xf32>,
    %init: tensor<1x7x7x64xf32>) -> tensor<1x7x7x64xf32> {
  %window = tensor.empty() : tensor<2x2xf32>
  %result = linalg.pooling_nhwc_min {
      dilations = dense<1> : vector<2xi64>,
      strides = dense<1> : vector<2xi64>
    } ins(%input, %window : tensor<1x8x8x64xf32>, tensor<2x2xf32>)
      outs(%init : tensor<1x7x7x64xf32>) -> tensor<1x7x7x64xf32>
  util.return %result : tensor<1x7x7x64xf32>
}

// 8x8 is MAX_DIRECT_KERNEL, hardware-confirmed.
// CHECK-LABEL: util.func public @min_kernel_8x8_accepted
// CHECK: flow.dispatch @rocket_pooling_min_executable
util.func public @min_kernel_8x8_accepted(
    %input: tensor<1x8x8x64xf32>,
    %init: tensor<1x1x1x64xf32>) -> tensor<1x1x1x64xf32> {
  %window = tensor.empty() : tensor<8x8xf32>
  %result = linalg.pooling_nhwc_min {
      dilations = dense<1> : vector<2xi64>,
      strides = dense<1> : vector<2xi64>
    } ins(%input, %window : tensor<1x8x8x64xf32>, tensor<8x8xf32>)
      outs(%init : tensor<1x1x1x64xf32>) -> tensor<1x1x1x64xf32>
  util.return %result : tensor<1x1x1x64xf32>
}

// ---------------------------------------------------------------- rejected

// 9x9 is one past MAX_DIRECT_KERNEL.
// CHECK-LABEL: util.func public @min_kernel_9x9_rejected
// CHECK: linalg.pooling_nhwc_min
util.func public @min_kernel_9x9_rejected(
    %input: tensor<1x9x9x64xf32>,
    %init: tensor<1x1x1x64xf32>) -> tensor<1x1x1x64xf32> {
  %window = tensor.empty() : tensor<9x9xf32>
  %result = linalg.pooling_nhwc_min {
      dilations = dense<1> : vector<2xi64>,
      strides = dense<1> : vector<2xi64>
    } ins(%input, %window : tensor<1x9x9x64xf32>, tensor<9x9xf32>)
      outs(%init : tensor<1x1x1x64xf32>) -> tensor<1x1x1x64xf32>
  util.return %result : tensor<1x1x1x64xf32>
}

// A 1x1 window is an identity, excluded for the same reason as max: claiming
// it would spend a dispatch, a pack and a compaction to copy a tensor.
// CHECK-LABEL: util.func public @min_kernel_1x1_rejected
// CHECK: linalg.pooling_nhwc_min
util.func public @min_kernel_1x1_rejected(
    %input: tensor<1x8x8x64xf32>,
    %init: tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32> {
  %window = tensor.empty() : tensor<1x1xf32>
  %result = linalg.pooling_nhwc_min {
      dilations = dense<1> : vector<2xi64>,
      strides = dense<1> : vector<2xi64>
    } ins(%input, %window : tensor<1x8x8x64xf32>, tensor<1x1xf32>)
      outs(%init : tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
  util.return %result : tensor<1x8x8x64xf32>
}

// Stride 3 has no measurement; `pooling_oracle_hw.rs` covers 1 and 2.
// CHECK-LABEL: util.func public @min_stride_3_rejected
// CHECK: linalg.pooling_nhwc_min
util.func public @min_stride_3_rejected(
    %input: tensor<1x14x14x64xf32>,
    %init: tensor<1x5x5x64xf32>) -> tensor<1x5x5x64xf32> {
  %window = tensor.empty() : tensor<2x2xf32>
  %result = linalg.pooling_nhwc_min {
      dilations = dense<1> : vector<2xi64>,
      strides = dense<3> : vector<2xi64>
    } ins(%input, %window : tensor<1x14x14x64xf32>, tensor<2x2xf32>)
      outs(%init : tensor<1x5x5x64xf32>) -> tensor<1x5x5x64xf32>
  util.return %result : tensor<1x5x5x64xf32>
}

// The unsigned-integer reduction is a different op and stays unclaimed: this
// path is f32 in, fp16 on the hardware, so there is nothing for it to mean.
// CHECK-LABEL: util.func public @min_unsigned_falls_back
// CHECK: linalg.pooling_nhwc_min_unsigned
util.func public @min_unsigned_falls_back(
    %input: tensor<1x14x14x64xi8>,
    %init: tensor<1x7x7x64xi8>) -> tensor<1x7x7x64xi8> {
  %window = tensor.empty() : tensor<2x2xi8>
  %result = linalg.pooling_nhwc_min_unsigned {
      dilations = dense<1> : vector<2xi64>,
      strides = dense<2> : vector<2xi64>
    } ins(%input, %window : tensor<1x14x14x64xi8>, tensor<2x2xi8>)
      outs(%init : tensor<1x7x7x64xi8>) -> tensor<1x7x7x64xi8>
  util.return %result : tensor<1x7x7x64xi8>
}
