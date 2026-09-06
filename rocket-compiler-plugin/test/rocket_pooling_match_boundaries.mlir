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

// Boundary coverage for @match_pooling_nchw_sum_avg. Each accepted shape has
// an adjacent rejected one, so widening the matcher cannot silently claim
// something the hardware has no measurement for.
//
// linalg has no average pool: an ONNX AveragePool arrives as a sum pool plus
// a separate divide, and the PPU has no sum mode. The matched replacement
// therefore runs the hardware's *average* and multiplies it back up by
// kh*kw, leaving the model's own divide in place.

// MobileNetV2's global average pool: 7x7 over 1792 channels, stride 1.
// CHECK-LABEL: util.func public @global_avg_pool_7x7
// CHECK-NOT: linalg.pooling_nchw_sum
// CHECK: flow.dispatch @rocket_pooling_executable
util.func public @global_avg_pool_7x7(
    %input: tensor<1x1792x7x7xf32>,
    %init: tensor<1x1792x1x1xf32>) -> tensor<1x1792x1x1xf32> {
  %window = tensor.empty() : tensor<7x7xf32>
  %result = linalg.pooling_nchw_sum {
      dilations = dense<1> : vector<2xi64>,
      strides = dense<1> : vector<2xi64>
    } ins(%input, %window : tensor<1x1792x7x7xf32>, tensor<7x7xf32>)
      outs(%init : tensor<1x1792x1x1xf32>) -> tensor<1x1792x1x1xf32>
  util.return %result : tensor<1x1792x1x1xf32>
}

// An 8x8 window is the largest the PPU programs directly -- MAX_DIRECT_KERNEL,
// hardware-confirmed, with a 16x16 window rejected outright.
// CHECK-LABEL: util.func public @avg_pool_8x8_matches
// CHECK: flow.dispatch @rocket_pooling_executable
util.func public @avg_pool_8x8_matches(
    %input: tensor<1x64x8x8xf32>,
    %init: tensor<1x64x1x1xf32>) -> tensor<1x64x1x1xf32> {
  %window = tensor.empty() : tensor<8x8xf32>
  %result = linalg.pooling_nchw_sum {
      dilations = dense<1> : vector<2xi64>,
      strides = dense<1> : vector<2xi64>
    } ins(%input, %window : tensor<1x64x8x8xf32>, tensor<8x8xf32>)
      outs(%init : tensor<1x64x1x1xf32>) -> tensor<1x64x1x1xf32>
  util.return %result : tensor<1x64x1x1xf32>
}

// 9x9 is one past it and stays on the CPU.
// CHECK-LABEL: util.func public @avg_pool_9x9_falls_back
// CHECK: linalg.pooling_nchw_sum
util.func public @avg_pool_9x9_falls_back(
    %input: tensor<1x64x9x9xf32>,
    %init: tensor<1x64x1x1xf32>) -> tensor<1x64x1x1xf32> {
  %window = tensor.empty() : tensor<9x9xf32>
  %result = linalg.pooling_nchw_sum {
      dilations = dense<1> : vector<2xi64>,
      strides = dense<1> : vector<2xi64>
    } ins(%input, %window : tensor<1x64x9x9xf32>, tensor<9x9xf32>)
      outs(%init : tensor<1x64x1x1xf32>) -> tensor<1x64x1x1xf32>
  util.return %result : tensor<1x64x1x1xf32>
}

// A 1x1 window is below the floor, and the floor is arithmetic rather than
// arbitrary: an fp16 average's reciprocal is fp16(65536/k), and k=1 needs
// 65536, past fp16's 65504 ceiling.
// CHECK-LABEL: util.func public @avg_pool_1x1_falls_back
// CHECK: linalg.pooling_nchw_sum
util.func public @avg_pool_1x1_falls_back(
    %input: tensor<1x64x8x8xf32>,
    %init: tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32> {
  %window = tensor.empty() : tensor<1x1xf32>
  %result = linalg.pooling_nchw_sum {
      dilations = dense<1> : vector<2xi64>,
      strides = dense<1> : vector<2xi64>
    } ins(%input, %window : tensor<1x64x8x8xf32>, tensor<1x1xf32>)
      outs(%init : tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
  util.return %result : tensor<1x64x8x8xf32>
}

// Stride 2 is not claimed: the executable bakes stride 1, so a strided pool
// would be computed at the wrong geometry rather than declined.
// CHECK-LABEL: util.func public @avg_pool_stride2_falls_back
// CHECK: linalg.pooling_nchw_sum
util.func public @avg_pool_stride2_falls_back(
    %input: tensor<1x64x16x16xf32>,
    %init: tensor<1x64x7x7xf32>) -> tensor<1x64x7x7xf32> {
  %window = tensor.empty() : tensor<3x3xf32>
  %result = linalg.pooling_nchw_sum {
      dilations = dense<1> : vector<2xi64>,
      strides = dense<2> : vector<2xi64>
    } ins(%input, %window : tensor<1x64x16x16xf32>, tensor<3x3xf32>)
      outs(%init : tensor<1x64x7x7xf32>) -> tensor<1x64x7x7xf32>
  util.return %result : tensor<1x64x7x7xf32>
}

// Max pooling is a different op, and this matcher must not claim it: the
// executable it dispatches to bakes method = "avg", which would silently
// average a tensor the model asked to reduce by maximum.
//
// Since @match_pooling_nchw_max landed, a max pool is claimed -- but by its
// own executable. Asserting *which* one is the point: a matcher that started
// routing max pools into @rocket_pooling_executable would still be "claimed"
// and still be wrong.
// CHECK-LABEL: util.func public @max_pool_uses_the_max_executable
// CHECK-NOT: @rocket_pooling_executable
// CHECK: flow.dispatch @rocket_pooling_max_executable
util.func public @max_pool_uses_the_max_executable(
    %input: tensor<1x64x8x8xf32>,
    %init: tensor<1x64x4x4xf32>) -> tensor<1x64x4x4xf32> {
  %window = tensor.empty() : tensor<2x2xf32>
  %result = linalg.pooling_nchw_max {
      dilations = dense<1> : vector<2xi64>,
      strides = dense<2> : vector<2xi64>
    } ins(%input, %window : tensor<1x64x8x8xf32>, tensor<2x2xf32>)
      outs(%init : tensor<1x64x4x4xf32>) -> tensor<1x64x4x4xf32>
  util.return %result : tensor<1x64x4x4xf32>
}

// ------------------------------------------------------------------- NHWC

// The NHWC counterpart, claimed by @match_pooling_nhwc_sum_avg since
// 2026-09-05. It shares @rocket_pooling_executable with the NCHW shim -- the
// executable takes NC1HWC2 cubes and knows nothing about the logical layout
// its caller started from -- so only the shim differs, and it is the cheaper
// one: NHWC is the hardware's own layout, so it transposes nothing.
// CHECK-LABEL: util.func public @avg_pool_nhwc_matches
// CHECK-NOT: linalg.pooling_nhwc_sum
// CHECK: flow.dispatch @rocket_pooling_executable
util.func public @avg_pool_nhwc_matches(
    %input: tensor<1x7x7x1792xf32>,
    %init: tensor<1x1x1x1792xf32>) -> tensor<1x1x1x1792xf32> {
  %window = tensor.empty() : tensor<7x7xf32>
  %result = linalg.pooling_nhwc_sum {
      dilations = dense<1> : vector<2xi64>,
      strides = dense<1> : vector<2xi64>
    } ins(%input, %window : tensor<1x7x7x1792xf32>, tensor<7x7xf32>)
      outs(%init : tensor<1x1x1x1792xf32>) -> tensor<1x1x1x1792xf32>
  util.return %result : tensor<1x1x1x1792xf32>
}

// Stride 2 is rejected for the average in *both* layouts, and this is the one
// remaining asymmetry in the pooling matchers: max and min both carry stride
// 2, and `avg 2x2s2` is measured against the oracle in pooling_oracle_hw.rs,
// so this is a missing executable rather than a missing measurement.
// CHECK-LABEL: util.func public @avg_pool_nhwc_s2_rejected
// CHECK: linalg.pooling_nhwc_sum
util.func public @avg_pool_nhwc_s2_rejected(
    %input: tensor<1x8x8x64xf32>,
    %init: tensor<1x4x4x64xf32>) -> tensor<1x4x4x64xf32> {
  %window = tensor.empty() : tensor<2x2xf32>
  %result = linalg.pooling_nhwc_sum {
      dilations = dense<1> : vector<2xi64>,
      strides = dense<2> : vector<2xi64>
    } ins(%input, %window : tensor<1x8x8x64xf32>, tensor<2x2xf32>)
      outs(%init : tensor<1x4x4x64xf32>) -> tensor<1x4x4x64xf32>
  util.return %result : tensor<1x4x4x64xf32>
}
