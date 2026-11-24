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

// The whole spec on an f16 convolution with its "same" pad and fused ReLU6,
// depthwise first and dense below: the pad-1 depthwise matchers claim pad, convolution and
// clamp together, so neither a tensor.pad nor the convolution is left for
// the CPU. (The rocket-fold-conv-pad unit test covers the NCHW-transpose
// form a real import arrives in; this is the claim itself.)

// CHECK-LABEL: util.func public @depthwise_pad1_relu6_claimed
// CHECK-NOT: tensor.pad
// CHECK-NOT: linalg.depthwise_conv_2d_nchw_chw
// CHECK: flow.dispatch @rocket_dynamic_depthwise_pad1_relu6_executable::
// CHECK-NOT: tensor.pad
// CHECK-NOT: linalg.depthwise_conv_2d_nchw_chw
// CHECK: util.return

func.func @depthwise_pad1_relu6_claimed(%input: tensor<1x144x56x56xf16>, %filter: tensor<144x3x3xf16>,
                  %bias: tensor<144xf32>) -> tensor<1x144x56x56xf32> {
  %zero = arith.constant 0.000000e+00 : f16
  %low = arith.constant 0.000000e+00 : f32
  %high = arith.constant 6.000000e+00 : f32
  %padded = tensor.pad %input low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%i0: index, %i1: index, %i2: index, %i3: index):
    tensor.yield %zero : f16
  } : tensor<1x144x56x56xf16> to tensor<1x144x58x58xf16>
  %acc_empty = tensor.empty() : tensor<1x144x56x56xf32>
  %acc_init = linalg.fill ins(%low : f32) outs(%acc_empty : tensor<1x144x56x56xf32>) -> tensor<1x144x56x56xf32>
  %conv = linalg.depthwise_conv_2d_nchw_chw
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%padded, %filter : tensor<1x144x58x58xf16>, tensor<144x3x3xf16>)
      outs(%acc_init : tensor<1x144x56x56xf32>) -> tensor<1x144x56x56xf32>
  %out_empty = tensor.empty() : tensor<1x144x56x56xf32>
  %clamped = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d1)>,
                       affine_map<(d0, d1, d2, d3) -> ()>,
                       affine_map<(d0, d1, d2, d3) -> ()>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%conv, %bias, %low, %high : tensor<1x144x56x56xf32>, tensor<144xf32>, f32, f32)
      outs(%out_empty : tensor<1x144x56x56xf32>) {
  ^bb0(%raw: f32, %b: f32, %lo: f32, %hi: f32, %unused: f32):
    %biased = arith.addf %raw, %b : f32
    %low_clamped = arith.maximumf %biased, %lo : f32
    %result = arith.minimumf %low_clamped, %hi : f32
    linalg.yield %result : f32
  } -> tensor<1x144x56x56xf32>
  return %clamped : tensor<1x144x56x56xf32>
}

// CHECK-LABEL: util.func public @depthwise_pad1_relu6_s2_claimed
// CHECK-NOT: tensor.pad
// CHECK: flow.dispatch @rocket_dynamic_depthwise_pad1_relu6_executable_s2::
// CHECK-NOT: tensor.pad
// CHECK: util.return

func.func @depthwise_pad1_relu6_s2_claimed(%input: tensor<1x144x56x56xf16>, %filter: tensor<144x3x3xf16>,
                  %bias: tensor<144xf32>) -> tensor<1x144x28x28xf32> {
  %zero = arith.constant 0.000000e+00 : f16
  %low = arith.constant 0.000000e+00 : f32
  %high = arith.constant 6.000000e+00 : f32
  %padded = tensor.pad %input low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%i0: index, %i1: index, %i2: index, %i3: index):
    tensor.yield %zero : f16
  } : tensor<1x144x56x56xf16> to tensor<1x144x58x58xf16>
  %acc_empty = tensor.empty() : tensor<1x144x28x28xf32>
  %acc_init = linalg.fill ins(%low : f32) outs(%acc_empty : tensor<1x144x28x28xf32>) -> tensor<1x144x28x28xf32>
  %conv = linalg.depthwise_conv_2d_nchw_chw
      {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
      ins(%padded, %filter : tensor<1x144x58x58xf16>, tensor<144x3x3xf16>)
      outs(%acc_init : tensor<1x144x28x28xf32>) -> tensor<1x144x28x28xf32>
  %out_empty = tensor.empty() : tensor<1x144x28x28xf32>
  %clamped = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d1)>,
                       affine_map<(d0, d1, d2, d3) -> ()>,
                       affine_map<(d0, d1, d2, d3) -> ()>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%conv, %bias, %low, %high : tensor<1x144x28x28xf32>, tensor<144xf32>, f32, f32)
      outs(%out_empty : tensor<1x144x28x28xf32>) {
  ^bb0(%raw: f32, %b: f32, %lo: f32, %hi: f32, %unused: f32):
    %biased = arith.addf %raw, %b : f32
    %low_clamped = arith.maximumf %biased, %lo : f32
    %result = arith.minimumf %low_clamped, %hi : f32
    linalg.yield %result : f32
  } -> tensor<1x144x28x28xf32>
  return %clamped : tensor<1x144x28x28xf32>
}

// The dense twins: MobileNetV2's stem, a 3x3 stride-2 Cin 3 convolution
// with its pad and ReLU6, claimed whole by the pad-1 dense ReLU6 matcher.
// Before, the stem's pad ran fused into the input transpose on the CPU and
// its clamp as a CPU elementwise -- no dense stride-2 ReLU6 matcher existed.

// CHECK-LABEL: util.func public @dense_stem_pad1_relu6_s2_claimed
// CHECK-NOT: tensor.pad
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf
// CHECK: flow.dispatch @rocket_dynamic_pad1_relu6_executable_s2::
// CHECK-NOT: tensor.pad
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf
// CHECK: util.return

func.func @dense_stem_pad1_relu6_s2_claimed(%input: tensor<1x1x224x224x3xf16>, %filter: tensor<3x3x3x32xf16>,
                  %bias: tensor<32xf32>) -> tensor<1x112x112x32xf32> {
  %zero = arith.constant 0.000000e+00 : f16
  %low = arith.constant 0.000000e+00 : f32
  %high = arith.constant 6.000000e+00 : f32
  %padded = tensor.pad %input low[0, 0, 1, 1, 0] high[0, 0, 1, 1, 0] {
  ^bb0(%i0: index, %i1: index, %i2: index, %i3: index, %i4: index):
    tensor.yield %zero : f16
  } : tensor<1x1x224x224x3xf16> to tensor<1x1x226x226x3xf16>
  %collapsed = tensor.collapse_shape %padded [[0], [1, 2], [3], [4]]
      : tensor<1x1x226x226x3xf16> into tensor<1x226x226x3xf16>
  %acc_empty = tensor.empty() : tensor<1x112x112x32xf32>
  %acc_init = linalg.fill ins(%low : f32) outs(%acc_empty : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
  %conv = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
      ins(%collapsed, %filter : tensor<1x226x226x3xf16>, tensor<3x3x3x32xf16>)
      outs(%acc_init : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
  %out_empty = tensor.empty() : tensor<1x112x112x32xf32>
  %clamped = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d3)>,
                       affine_map<(d0, d1, d2, d3) -> ()>,
                       affine_map<(d0, d1, d2, d3) -> ()>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%conv, %bias, %low, %high : tensor<1x112x112x32xf32>, tensor<32xf32>, f32, f32)
      outs(%out_empty : tensor<1x112x112x32xf32>) {
  ^bb0(%raw: f32, %b: f32, %lo: f32, %hi: f32, %unused: f32):
    %biased = arith.addf %raw, %b : f32
    %low_clamped = arith.maximumf %biased, %lo : f32
    %result = arith.minimumf %low_clamped, %hi : f32
    linalg.yield %result : f32
  } -> tensor<1x112x112x32xf32>
  return %clamped : tensor<1x112x112x32xf32>
}

// CHECK-LABEL: util.func public @dense_pad1_relu6_s1_claimed
// CHECK-NOT: tensor.pad
// CHECK: flow.dispatch @rocket_dynamic_pad1_relu6_executable::
// CHECK-NOT: tensor.pad
// CHECK: util.return

func.func @dense_pad1_relu6_s1_claimed(%input: tensor<1x1x56x56x64xf16>, %filter: tensor<3x3x64x64xf16>,
                  %bias: tensor<64xf32>) -> tensor<1x56x56x64xf32> {
  %zero = arith.constant 0.000000e+00 : f16
  %low = arith.constant 0.000000e+00 : f32
  %high = arith.constant 6.000000e+00 : f32
  %padded = tensor.pad %input low[0, 0, 1, 1, 0] high[0, 0, 1, 1, 0] {
  ^bb0(%i0: index, %i1: index, %i2: index, %i3: index, %i4: index):
    tensor.yield %zero : f16
  } : tensor<1x1x56x56x64xf16> to tensor<1x1x58x58x64xf16>
  %collapsed = tensor.collapse_shape %padded [[0], [1, 2], [3], [4]]
      : tensor<1x1x58x58x64xf16> into tensor<1x58x58x64xf16>
  %acc_empty = tensor.empty() : tensor<1x56x56x64xf32>
  %acc_init = linalg.fill ins(%low : f32) outs(%acc_empty : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %conv = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%collapsed, %filter : tensor<1x58x58x64xf16>, tensor<3x3x64x64xf16>)
      outs(%acc_init : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %out_empty = tensor.empty() : tensor<1x56x56x64xf32>
  %clamped = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d3)>,
                       affine_map<(d0, d1, d2, d3) -> ()>,
                       affine_map<(d0, d1, d2, d3) -> ()>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%conv, %bias, %low, %high : tensor<1x56x56x64xf32>, tensor<64xf32>, f32, f32)
      outs(%out_empty : tensor<1x56x56x64xf32>) {
  ^bb0(%raw: f32, %b: f32, %lo: f32, %hi: f32, %unused: f32):
    %biased = arith.addf %raw, %b : f32
    %low_clamped = arith.maximumf %biased, %lo : f32
    %result = arith.minimumf %low_clamped, %hi : f32
    linalg.yield %result : f32
  } -> tensor<1x56x56x64xf32>
  return %clamped : tensor<1x56x56x64xf32>
}
