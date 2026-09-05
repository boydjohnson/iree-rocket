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

// Boundary coverage for the four max-pool matchers, on the same contract as
// rocket_pooling_match_boundaries.mlir: every accepted shape has an
// immediately-adjacent rejected one, so widening a matcher cannot silently
// claim something the hardware has no measurement for, and tightening one
// cannot silently lose the largest measured-good shape.
//
// Unlike the average pool, max needs no correction after the hardware: the
// PPU computes the maximum directly. What the shim still does is widen f16
// back to f32 and fold in the accumulator initialiser, because linalg
// defines a max pool as `O = max(O, I)`.

// ---------------------------------------------------------------- accepted

// NHWC stride 2 -- the shape a real model has, and the layout the hardware
// wants, so this shim transposes nothing.
// CHECK-LABEL: util.func public @max_nhwc_s2
// CHECK-NOT: linalg.pooling_nhwc_max
// CHECK: flow.dispatch @rocket_pooling_max_executable_s2
util.func public @max_nhwc_s2(
    %input: tensor<1x14x14x64xf32>,
    %init: tensor<1x7x7x64xf32>) -> tensor<1x7x7x64xf32> {
  %window = tensor.empty() : tensor<2x2xf32>
  %result = linalg.pooling_nhwc_max {
      dilations = dense<1> : vector<2xi64>,
      strides = dense<2> : vector<2xi64>
    } ins(%input, %window : tensor<1x14x14x64xf32>, tensor<2x2xf32>)
      outs(%init : tensor<1x7x7x64xf32>) -> tensor<1x7x7x64xf32>
  util.return %result : tensor<1x7x7x64xf32>
}

// NHWC stride 1, which routes to the stride-1 executable rather than the
// stride-2 one. Stride is baked per executable, so picking the wrong one is
// a wrong answer, not a decline.
// CHECK-LABEL: util.func public @max_nhwc_s1
// CHECK-NOT: @rocket_pooling_max_executable_s2
// CHECK: flow.dispatch @rocket_pooling_max_executable
util.func public @max_nhwc_s1(
    %input: tensor<1x8x8x64xf32>,
    %init: tensor<1x7x7x64xf32>) -> tensor<1x7x7x64xf32> {
  %window = tensor.empty() : tensor<2x2xf32>
  %result = linalg.pooling_nhwc_max {
      dilations = dense<1> : vector<2xi64>,
      strides = dense<1> : vector<2xi64>
    } ins(%input, %window : tensor<1x8x8x64xf32>, tensor<2x2xf32>)
      outs(%init : tensor<1x7x7x64xf32>) -> tensor<1x7x7x64xf32>
  util.return %result : tensor<1x7x7x64xf32>
}

// NCHW, the layout ONNX imports. iree-preprocessing-convert-conv-to-channels-
// last does not touch pooling ops, so this needs its own matcher and shim.
// CHECK-LABEL: util.func public @max_nchw_s2
// CHECK-NOT: linalg.pooling_nchw_max
// CHECK: flow.dispatch @rocket_pooling_max_executable_s2
util.func public @max_nchw_s2(
    %input: tensor<1x64x14x14xf32>,
    %init: tensor<1x64x7x7xf32>) -> tensor<1x64x7x7xf32> {
  %window = tensor.empty() : tensor<2x2xf32>
  %result = linalg.pooling_nchw_max {
      dilations = dense<1> : vector<2xi64>,
      strides = dense<2> : vector<2xi64>
    } ins(%input, %window : tensor<1x64x14x14xf32>, tensor<2x2xf32>)
      outs(%init : tensor<1x64x7x7xf32>) -> tensor<1x64x7x7xf32>
  util.return %result : tensor<1x64x7x7xf32>
}

// An 8x8 window is MAX_DIRECT_KERNEL, hardware-confirmed, with 16x16
// rejected by the hardware outright.
// CHECK-LABEL: util.func public @max_kernel_8x8_accepted
// CHECK: flow.dispatch @rocket_pooling_max_executable
util.func public @max_kernel_8x8_accepted(
    %input: tensor<1x8x8x64xf32>,
    %init: tensor<1x1x1x64xf32>) -> tensor<1x1x1x64xf32> {
  %window = tensor.empty() : tensor<8x8xf32>
  %result = linalg.pooling_nhwc_max {
      dilations = dense<1> : vector<2xi64>,
      strides = dense<1> : vector<2xi64>
    } ins(%input, %window : tensor<1x8x8x64xf32>, tensor<8x8xf32>)
      outs(%init : tensor<1x1x1x64xf32>) -> tensor<1x1x1x64xf32>
  util.return %result : tensor<1x1x1x64xf32>
}

// ---------------------------------------------------------------- rejected

// 9x9 is one past MAX_DIRECT_KERNEL.
// CHECK-LABEL: util.func public @max_kernel_9x9_rejected
// CHECK: linalg.pooling_nhwc_max
util.func public @max_kernel_9x9_rejected(
    %input: tensor<1x9x9x64xf32>,
    %init: tensor<1x1x1x64xf32>) -> tensor<1x1x1x64xf32> {
  %window = tensor.empty() : tensor<9x9xf32>
  %result = linalg.pooling_nhwc_max {
      dilations = dense<1> : vector<2xi64>,
      strides = dense<1> : vector<2xi64>
    } ins(%input, %window : tensor<1x9x9x64xf32>, tensor<9x9xf32>)
      outs(%init : tensor<1x1x1x64xf32>) -> tensor<1x1x1x64xf32>
  util.return %result : tensor<1x1x1x64xf32>
}

// A 1x1 window is programmable -- the average pool's floor of 2 comes from
// its fp16 reciprocal, which max does not compute -- but it is an identity,
// and claiming it would spend an NPU dispatch, a pack and a compaction to
// copy a tensor. Excluded deliberately, not by a hardware limit.
// CHECK-LABEL: util.func public @max_kernel_1x1_rejected
// CHECK: linalg.pooling_nhwc_max
util.func public @max_kernel_1x1_rejected(
    %input: tensor<1x8x8x64xf32>,
    %init: tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32> {
  %window = tensor.empty() : tensor<1x1xf32>
  %result = linalg.pooling_nhwc_max {
      dilations = dense<1> : vector<2xi64>,
      strides = dense<1> : vector<2xi64>
    } ins(%input, %window : tensor<1x8x8x64xf32>, tensor<1x1xf32>)
      outs(%init : tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
  util.return %result : tensor<1x8x8x64xf32>
}

// Stride 3. The PPU's register range reaches 16, but `pooling_oracle_hw.rs`
// measures 1 and 2, and this repo does not claim a shape ahead of its
// measurement.
// CHECK-LABEL: util.func public @max_stride_3_rejected
// CHECK: linalg.pooling_nhwc_max
util.func public @max_stride_3_rejected(
    %input: tensor<1x14x14x64xf32>,
    %init: tensor<1x5x5x64xf32>) -> tensor<1x5x5x64xf32> {
  %window = tensor.empty() : tensor<2x2xf32>
  %result = linalg.pooling_nhwc_max {
      dilations = dense<1> : vector<2xi64>,
      strides = dense<3> : vector<2xi64>
    } ins(%input, %window : tensor<1x14x14x64xf32>, tensor<2x2xf32>)
      outs(%init : tensor<1x5x5x64xf32>) -> tensor<1x5x5x64xf32>
  util.return %result : tensor<1x5x5x64xf32>
}

// Dilation is not supported by the pooling builder at all.
// CHECK-LABEL: util.func public @max_dilated_rejected
// CHECK: linalg.pooling_nhwc_max
util.func public @max_dilated_rejected(
    %input: tensor<1x10x10x64xf32>,
    %init: tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32> {
  %window = tensor.empty() : tensor<2x2xf32>
  %result = linalg.pooling_nhwc_max {
      dilations = dense<2> : vector<2xi64>,
      strides = dense<1> : vector<2xi64>
    } ins(%input, %window : tensor<1x10x10x64xf32>, tensor<2x2xf32>)
      outs(%init : tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
  util.return %result : tensor<1x8x8x64xf32>
}

// Min pooling is a different reduction, and these matchers must not claim it:
// their executables bake method = "max", which would return the wrong end of
// every window. Since @match_pooling_nhwc_min landed, a min pool is claimed --
// but by its own executable, and asserting *which* is the point.
// rocket_pooling_min_match_boundaries.mlir carries min's own bounds.
// CHECK-LABEL: util.func public @min_pool_uses_the_min_executable
// CHECK-NOT: @rocket_pooling_max_executable
// CHECK: flow.dispatch @rocket_pooling_min_executable
util.func public @min_pool_uses_the_min_executable(
    %input: tensor<1x8x8x64xf32>,
    %init: tensor<1x4x4x64xf32>) -> tensor<1x4x4x64xf32> {
  %window = tensor.empty() : tensor<2x2xf32>
  %result = linalg.pooling_nhwc_min {
      dilations = dense<1> : vector<2xi64>,
      strides = dense<2> : vector<2xi64>
    } ins(%input, %window : tensor<1x8x8x64xf32>, tensor<2x2xf32>)
      outs(%init : tensor<1x4x4x64xf32>) -> tensor<1x4x4x64xf32>
  util.return %result : tensor<1x4x4x64xf32>
}
