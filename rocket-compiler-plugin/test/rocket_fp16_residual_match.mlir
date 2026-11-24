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

// The whole spec on a linear residual bottleneck as an f32 import reaches
// it: the 1x1 projection (demoted), its bias, and the block's skip add with
// nothing after it. rocket-fuse-conv-relu6's FuseConvResidualLinear builds
// the epilogue and @match_dynamic_conv2d_residual_demoted_wide claims conv,
// bias and add as one dispatch (conv tiles + one EW task), so no add is left
// for the CPU.

// CHECK-LABEL: util.func public @linear_residual_claimed
// CHECK-NOT: linalg.add
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf
// CHECK: flow.dispatch @rocket_dynamic_residual_executable::
// CHECK-NOT: linalg.add
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf
// CHECK: util.return
func.func @linear_residual_claimed(%input: tensor<1x14x14x384xf16>,
    %filter: tensor<1x1x384x64xf16>, %bias: tensor<64xf32>, %skip: tensor<1x1x14x14x64xf32>) -> tensor<1x1x14x14x64xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %nchw_empty = tensor.empty() : tensor<1x64x14x14xf32>
  %bias_nchw = linalg.broadcast ins(%bias : tensor<64xf32>)
      outs(%nchw_empty : tensor<1x64x14x14xf32>) dimensions = [0, 2, 3]
  %init_empty = tensor.empty() : tensor<1x14x14x64xf32>
  %init = linalg.transpose ins(%bias_nchw : tensor<1x64x14x14xf32>)
      outs(%init_empty : tensor<1x14x14x64xf32>) permutation = [0, 2, 3, 1]
  %conv = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, rocket.f16_demoted, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x14x14x384xf16>, tensor<1x1x384x64xf16>)
      outs(%init : tensor<1x14x14x64xf32>) -> tensor<1x14x14x64xf32>
  %expanded = tensor.expand_shape %conv [[0], [1, 2], [3], [4]] output_shape [1, 1, 14, 14, 64]
      : tensor<1x14x14x64xf32> into tensor<1x1x14x14x64xf32>
  %sum_empty = tensor.empty() : tensor<1x1x14x14x64xf32>
  %sum = linalg.add ins(%skip, %expanded : tensor<1x1x14x14x64xf32>, tensor<1x1x14x14x64xf32>) outs(%sum_empty : tensor<1x1x14x14x64xf32>) -> tensor<1x1x14x14x64xf32>
  return %sum : tensor<1x1x14x14x64xf32>
}
