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

// Boundary coverage for @match_rocket_matmul. A matmul reaches this hardware
// as a height-one 1x1 convolution -- M is the convolution width, K its input
// channels and N its output channels -- so the bounds here are the HAL's
// channel ceilings under different names.

// MobileNetV2's classifier, and exactly the shape the ceilings were measured
// at: K = 1792 is MAX_INPUT_CHANNELS.
// CHECK-LABEL: util.func public @classifier_matches
// CHECK-NOT: linalg.matmul
// CHECK: util.call @call_rocket_matmul
util.func public @classifier_matches(
    %lhs: tensor<1x1792xf32>,
    %rhs: tensor<1792x1001xf32>,
    %init: tensor<1x1001xf32>) -> tensor<1x1001xf32> {
  %result = linalg.matmul
      ins(%lhs, %rhs : tensor<1x1792xf32>, tensor<1792x1001xf32>)
      outs(%init : tensor<1x1001xf32>) -> tensor<1x1001xf32>
  util.return %result : tensor<1x1001xf32>
}

// One channel past it, and the whole matmul stays on the CPU rather than
// reaching a driver that would refuse it.
// CHECK-LABEL: util.func public @k_past_the_ceiling_falls_back
// CHECK: linalg.matmul
util.func public @k_past_the_ceiling_falls_back(
    %lhs: tensor<1x1793xf32>,
    %rhs: tensor<1793x64xf32>,
    %init: tensor<1x64xf32>) -> tensor<1x64xf32> {
  %result = linalg.matmul
      ins(%lhs, %rhs : tensor<1x1793xf32>, tensor<1793x64xf32>)
      outs(%init : tensor<1x64xf32>) -> tensor<1x64xf32>
  util.return %result : tensor<1x64xf32>
}

// N is the output-channel count, bounded by MAX_OUTPUT_CHANNELS at the same
// 1792.
// CHECK-LABEL: util.func public @n_past_the_ceiling_falls_back
// CHECK: linalg.matmul
util.func public @n_past_the_ceiling_falls_back(
    %lhs: tensor<1x64xf32>,
    %rhs: tensor<64x1793xf32>,
    %init: tensor<1x1793xf32>) -> tensor<1x1793xf32> {
  %result = linalg.matmul
      ins(%lhs, %rhs : tensor<1x64xf32>, tensor<64x1793xf32>)
      outs(%init : tensor<1x1793xf32>) -> tensor<1x1793xf32>
  util.return %result : tensor<1x1793xf32>
}

// M becomes the convolution's width. 32 is where the hardware ladder stops,
// so it is where the matcher stops.
// CHECK-LABEL: util.func public @m_32_matches
// CHECK: util.call @call_rocket_matmul
util.func public @m_32_matches(
    %lhs: tensor<32x64xf32>,
    %rhs: tensor<64x64xf32>,
    %init: tensor<32x64xf32>) -> tensor<32x64xf32> {
  %result = linalg.matmul
      ins(%lhs, %rhs : tensor<32x64xf32>, tensor<64x64xf32>)
      outs(%init : tensor<32x64xf32>) -> tensor<32x64xf32>
  util.return %result : tensor<32x64xf32>
}

// CHECK-LABEL: util.func public @m_33_falls_back
// CHECK: linalg.matmul
util.func public @m_33_falls_back(
    %lhs: tensor<33x64xf32>,
    %rhs: tensor<64x64xf32>,
    %init: tensor<33x64xf32>) -> tensor<33x64xf32> {
  %result = linalg.matmul
      ins(%lhs, %rhs : tensor<33x64xf32>, tensor<64x64xf32>)
      outs(%init : tensor<33x64xf32>) -> tensor<33x64xf32>
  util.return %result : tensor<33x64xf32>
}

// The case a name-only matcher would get wrong. `linalg.matmul` expresses a
// transposed operand by overriding its indexing maps rather than by being a
// different op, and a transposed B is a memory layout the height-one
// convolution lowering cannot pack. Same shapes as the classifier otherwise.
// CHECK-LABEL: util.func public @transposed_rhs_falls_back
// CHECK: linalg.matmul
util.func public @transposed_rhs_falls_back(
    %lhs: tensor<1x1792xf32>,
    %rhs: tensor<1001x1792xf32>,
    %init: tensor<1x1001xf32>) -> tensor<1x1001xf32> {
  %result = linalg.matmul
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>]
      ins(%lhs, %rhs : tensor<1x1792xf32>, tensor<1001x1792xf32>)
      outs(%init : tensor<1x1001xf32>) -> tensor<1x1001xf32>
  util.return %result : tensor<1x1001xf32>
}

// A batched matmul is a different contraction with a batch dimension the
// Rocket ABI has no room for -- batch is fixed at one throughout.
// CHECK-LABEL: util.func public @batch_matmul_falls_back
// CHECK: linalg.batch_matmul
util.func public @batch_matmul_falls_back(
    %lhs: tensor<4x8x64xf32>,
    %rhs: tensor<4x64x64xf32>,
    %init: tensor<4x8x64xf32>) -> tensor<4x8x64xf32> {
  %result = linalg.batch_matmul
      ins(%lhs, %rhs : tensor<4x8x64xf32>, tensor<4x64x64xf32>)
      outs(%init : tensor<4x8x64xf32>) -> tensor<4x8x64xf32>
  util.return %result : tensor<4x8x64xf32>
}
