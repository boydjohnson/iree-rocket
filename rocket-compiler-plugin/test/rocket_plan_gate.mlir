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
// RUN:   -o - 2>/dev/null | FileCheck %s

// The shared planner gates admission (COMPILER_ROADMAP.md section 2). The
// matchers' own bounds cover channels only; a shape inside those bounds
// that the planner refuses used to compile to a Rocket dispatch the runtime
// then rejected with a bare INVALID_ARGUMENT. Now it falls back to the CPU,
// tagged with the refusal, and an admitted shape still offloads.

// A dense-layout (Cin <= 4) row of 4098 pixels does not fit one CBUF data
// bank, so the plan would need column tiles, which only NC1HWC2 surfaces
// have capture backing for. The 3x3 stride-1 matcher's bounds admit it.
// CHECK-LABEL: util.func public @wide_dense_row_falls_back
// CHECK-NOT: flow.dispatch @rocket_dynamic
// CHECK: linalg.conv_2d_nhwc_hwcf
// CHECK-SAME: rocket.plan_refused = "unvalidated_configuration"
func.func @wide_dense_row_falls_back(
    %input: tensor<1x6x4098x3xf16>,
    %filter: tensor<3x3x3x8xf16>,
    %init: tensor<1x4x4096x8xf32>) -> tensor<1x4x4096x8xf32> {
  %result = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x6x4098x3xf16>, tensor<3x3x3x8xf16>)
      outs(%init : tensor<1x4x4096x8xf32>) -> tensor<1x4x4096x8xf32>
  return %result : tensor<1x4x4096x8xf32>
}

// The same convolution at a width one bank holds is admitted and offloads.
// CHECK-LABEL: util.func public @narrow_dense_row_offloads
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf
// CHECK-NOT: rocket.plan_refused
// CHECK: flow.dispatch @rocket_dynamic
func.func @narrow_dense_row_offloads(
    %input: tensor<1x6x226x3xf16>,
    %filter: tensor<3x3x3x8xf16>,
    %init: tensor<1x4x224x8xf32>) -> tensor<1x4x224x8xf32> {
  %result = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x6x226x3xf16>, tensor<3x3x3x8xf16>)
      outs(%init : tensor<1x4x224x8xf32>) -> tensor<1x4x224x8xf32>
  return %result : tensor<1x4x224x8xf32>
}

// A symbolic extent is deferred to the runtime, not refused: the dynamic
// matchers keep admitting it.
// CHECK-LABEL: util.func public @dynamic_row_still_offloads
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf
// CHECK: flow.dispatch @rocket_dynamic
func.func @dynamic_row_still_offloads(
    %input: tensor<1x?x?x3xf16>,
    %filter: tensor<3x3x3x8xf16>,
    %init: tensor<1x?x?x8xf32>) -> tensor<1x?x?x8xf32> {
  %result = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x?x?x3xf16>, tensor<3x3x3x8xf16>)
      outs(%init : tensor<1x?x?x8xf32>) -> tensor<1x?x?x8xf32>
  return %result : tensor<1x?x?x8xf32>
}

// The matmul matcher carries the same check: a wide-M matmul whose K makes
// every row exceed what the planner tiles is refused through the FC path.
// CHECK-LABEL: util.func public @refused_matmul_falls_back
// CHECK-NOT: flow.dispatch @rocket_matmul
// CHECK: linalg.matmul
// CHECK-SAME: rocket.plan_refused
func.func @refused_matmul_falls_back(
    %lhs: tensor<4x3585xf16>,
    %rhs: tensor<3585x64xf16>,
    %init: tensor<4x64xf32>) -> tensor<4x64xf32> {
  %result = linalg.matmul
      ins(%lhs, %rhs : tensor<4x3585xf16>, tensor<3585x64xf16>)
      outs(%init : tensor<4x64xf32>) -> tensor<4x64xf32>
  return %result : tensor<4x64xf32>
}
