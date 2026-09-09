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

// Boundary coverage for the fp16 dense matchers. Each accepted shape has an
// immediately-adjacent rejected one, so widening a matcher cannot silently
// route an uncharacterized convolution to Rocket and tightening one cannot
// silently lose the largest measured-good shape.
//
// Raised 2026-09-03 from Cin 512 / Cout 528 to the HAL's `MAX_INPUT_CHANNELS`
// 1344 and `MAX_OUTPUT_CHANNELS` 1792, on board evidence plus the fp16 vendor
// corpus in `conv_vendor_fixture_wide.rs`, and again on 2026-09-06 to **3584**
// on both axes -- the raise that puts a transformer MLP (K = N = 3072) inside
// the matchers. That sweep is in `MAX_INPUT_CHANNELS`' doc comment: k=1 exact
// at 14x14 Cout 64 for Cin 1792..3584 and on to 8192, at 7x7 Cin 448 for Cout
// 1792..4096, ragged on both axes, and under the `onehot` read map at
// Cout == Cin.
//
// The 3x3 matcher stops at Cin 1152 and did not move: at k=3 the coefficient
// working set binds first and `ConvPlan` refuses Cin >= 1216 outright, so the
// channel ceiling is not what governs there.
//
// Since 2026-09-09 these ceilings are not in the spec at all. They are one
// table in `rocket-core`'s `admission` module, reached from each matcher's
// `transform.rocket.match.admitted` line, and moving them there collapsed
// the per-matcher drift the old `dim_bounds` had accumulated: the stride-2
// and stride-1 rows of a given kernel now share an envelope, because the
// epilogue a matcher fuses (bias on the BS plane, activation on the BN
// plane) sits downstream of the MAC array and does not touch the channel
// path. The cases below the depthwise ones are the corners that moved.

// CHECK-LABEL: util.func public @dense_1x1_cin_4096_matched
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf
// CHECK: flow.dispatch @rocket_dynamic_executable
func.func @dense_1x1_cin_4096_matched(
    %input: tensor<1x?x?x4096xf16>,
    %filter: tensor<1x1x4096x64xf16>,
    %init: tensor<1x?x?x64xf32>) -> tensor<1x?x?x64xf32> {
  %result = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x?x?x4096xf16>, tensor<1x1x4096x64xf16>)
      outs(%init : tensor<1x?x?x64xf32>) -> tensor<1x?x?x64xf32>
  return %result : tensor<1x?x?x64xf32>
}

// CHECK-LABEL: util.func public @dense_1x1_cin_4097_falls_back
// CHECK-NOT: flow.dispatch @rocket_dynamic_executable
// CHECK: linalg.conv_2d_nhwc_hwcf
func.func @dense_1x1_cin_4097_falls_back(
    %input: tensor<1x?x?x4097xf16>,
    %filter: tensor<1x1x4097x64xf16>,
    %init: tensor<1x?x?x64xf32>) -> tensor<1x?x?x64xf32> {
  %result = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x?x?x4097xf16>, tensor<1x1x4097x64xf16>)
      outs(%init : tensor<1x?x?x64xf32>) -> tensor<1x?x?x64xf32>
  return %result : tensor<1x?x?x64xf32>
}

// CHECK-LABEL: util.func public @dense_1x1_cout_4096_matched
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf
// CHECK: flow.dispatch @rocket_dynamic_executable
func.func @dense_1x1_cout_4096_matched(
    %input: tensor<1x?x?x448xf16>,
    %filter: tensor<1x1x448x4096xf16>,
    %init: tensor<1x?x?x4096xf32>) -> tensor<1x?x?x4096xf32> {
  %result = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x?x?x448xf16>, tensor<1x1x448x4096xf16>)
      outs(%init : tensor<1x?x?x4096xf32>) -> tensor<1x?x?x4096xf32>
  return %result : tensor<1x?x?x4096xf32>
}

// CHECK-LABEL: util.func public @dense_1x1_cout_4097_falls_back
// CHECK-NOT: flow.dispatch @rocket_dynamic_executable
// CHECK: linalg.conv_2d_nhwc_hwcf
func.func @dense_1x1_cout_4097_falls_back(
    %input: tensor<1x?x?x448xf16>,
    %filter: tensor<1x1x448x4097xf16>,
    %init: tensor<1x?x?x4097xf32>) -> tensor<1x?x?x4097xf32> {
  %result = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x?x?x448xf16>, tensor<1x1x448x4097xf16>)
      outs(%init : tensor<1x?x?x4097xf32>) -> tensor<1x?x?x4097xf32>
  return %result : tensor<1x?x?x4097xf32>
}

// CHECK-LABEL: util.func public @dense_3x3_cin_1152_matched
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf
// CHECK: flow.dispatch @rocket_dynamic_executable
func.func @dense_3x3_cin_1152_matched(
    %input: tensor<1x?x?x1152xf16>,
    %filter: tensor<3x3x1152x64xf16>,
    %init: tensor<1x?x?x64xf32>) -> tensor<1x?x?x64xf32> {
  %result = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x?x?x1152xf16>, tensor<3x3x1152x64xf16>)
      outs(%init : tensor<1x?x?x64xf32>) -> tensor<1x?x?x64xf32>
  return %result : tensor<1x?x?x64xf32>
}

// CHECK-LABEL: util.func public @dense_3x3_cin_1153_falls_back
// CHECK-NOT: flow.dispatch @rocket_dynamic_executable
// CHECK: linalg.conv_2d_nhwc_hwcf
func.func @dense_3x3_cin_1153_falls_back(
    %input: tensor<1x?x?x1153xf16>,
    %filter: tensor<3x3x1153x64xf16>,
    %init: tensor<1x?x?x64xf32>) -> tensor<1x?x?x64xf32> {
  %result = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x?x?x1153xf16>, tensor<3x3x1153x64xf16>)
      outs(%init : tensor<1x?x?x64xf32>) -> tensor<1x?x?x64xf32>
  return %result : tensor<1x?x?x64xf32>
}

// -----------------------------------------------------------------------
// The consolidated rows. Each of these was admitted to a *different*
// ceiling before the table moved, purely by which matcher happened to claim
// it, and each pair here is one channel apart across the new one.

// A plain stride-2 1x1 with nothing fused after it. This was bounded at Cin
// 512 while the same convolution with a ReLU after it was bounded at 3584.
// CHECK-LABEL: util.func public @dense_1x1_s2_cin_4096_matched
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf
// CHECK: flow.dispatch @rocket_dynamic_executable_s2
func.func @dense_1x1_s2_cin_4096_matched(
    %input: tensor<1x?x?x4096xf16>,
    %filter: tensor<1x1x4096x64xf16>,
    %init: tensor<1x?x?x64xf32>) -> tensor<1x?x?x64xf32> {
  %result = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
      ins(%input, %filter : tensor<1x?x?x4096xf16>, tensor<1x1x4096x64xf16>)
      outs(%init : tensor<1x?x?x64xf32>) -> tensor<1x?x?x64xf32>
  return %result : tensor<1x?x?x64xf32>
}

// CHECK-LABEL: util.func public @dense_1x1_s2_cin_4097_falls_back
// CHECK-NOT: flow.dispatch @rocket_dynamic_executable
// CHECK: linalg.conv_2d_nhwc_hwcf
func.func @dense_1x1_s2_cin_4097_falls_back(
    %input: tensor<1x?x?x4097xf16>,
    %filter: tensor<1x1x4097x64xf16>,
    %init: tensor<1x?x?x64xf32>) -> tensor<1x?x?x64xf32> {
  %result = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
      ins(%input, %filter : tensor<1x?x?x4097xf16>, tensor<1x1x4097x64xf16>)
      outs(%init : tensor<1x?x?x64xf32>) -> tensor<1x?x?x64xf32>
  return %result : tensor<1x?x?x64xf32>
}

// 3x3 `Cout`, which the padded matchers already admitted to the dense
// ceiling while the unpadded ones stopped at 1792. `Cout` charges no
// feature residency, so the kernel does not bound it.
// CHECK-LABEL: util.func public @dense_3x3_cout_4096_matched
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf
// CHECK: flow.dispatch @rocket_dynamic_executable
func.func @dense_3x3_cout_4096_matched(
    %input: tensor<1x?x?x64xf16>,
    %filter: tensor<3x3x64x4096xf16>,
    %init: tensor<1x?x?x4096xf32>) -> tensor<1x?x?x4096xf32> {
  %result = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x?x?x64xf16>, tensor<3x3x64x4096xf16>)
      outs(%init : tensor<1x?x?x4096xf32>) -> tensor<1x?x?x4096xf32>
  return %result : tensor<1x?x?x4096xf32>
}

// CHECK-LABEL: util.func public @dense_3x3_cout_4097_falls_back
// CHECK-NOT: flow.dispatch @rocket_dynamic_executable
// CHECK: linalg.conv_2d_nhwc_hwcf
func.func @dense_3x3_cout_4097_falls_back(
    %input: tensor<1x?x?x64xf16>,
    %filter: tensor<3x3x64x4097xf16>,
    %init: tensor<1x?x?x4097xf32>) -> tensor<1x?x?x4097xf32> {
  %result = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x?x?x64xf16>, tensor<3x3x64x4097xf16>)
      outs(%init : tensor<1x?x?x4097xf32>) -> tensor<1x?x?x4097xf32>
  return %result : tensor<1x?x?x4097xf32>
}

// 3x3 at stride 2, which the plain matcher bounded at Cin 512 and the
// ReLU-fused one at 1152. The 1152 is `ConvPlan`'s own coefficient limit,
// so it is the number that means something.
// CHECK-LABEL: util.func public @dense_3x3_s2_cin_1152_matched
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf
// CHECK: flow.dispatch @rocket_dynamic_executable_s2
func.func @dense_3x3_s2_cin_1152_matched(
    %input: tensor<1x?x?x1152xf16>,
    %filter: tensor<3x3x1152x64xf16>,
    %init: tensor<1x?x?x64xf32>) -> tensor<1x?x?x64xf32> {
  %result = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
      ins(%input, %filter : tensor<1x?x?x1152xf16>, tensor<3x3x1152x64xf16>)
      outs(%init : tensor<1x?x?x64xf32>) -> tensor<1x?x?x64xf32>
  return %result : tensor<1x?x?x64xf32>
}

// CHECK-LABEL: util.func public @dense_3x3_s2_cin_1153_falls_back
// CHECK-NOT: flow.dispatch @rocket_dynamic_executable
// CHECK: linalg.conv_2d_nhwc_hwcf
func.func @dense_3x3_s2_cin_1153_falls_back(
    %input: tensor<1x?x?x1153xf16>,
    %filter: tensor<3x3x1153x64xf16>,
    %init: tensor<1x?x?x64xf32>) -> tensor<1x?x?x64xf32> {
  %result = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
      ins(%input, %filter : tensor<1x?x?x1153xf16>, tensor<3x3x1153x64xf16>)
      outs(%init : tensor<1x?x?x64xf32>) -> tensor<1x?x?x64xf32>
  return %result : tensor<1x?x?x64xf32>
}

// Stride 3 keeps the 512 the s3/s4 matchers always had, and the
// consolidation deliberately stops before it: no sweep has been taken at a
// stride above 2. It is not observable here, because those two matchers are
// defined in the spec but no `foreach_match` list invokes them, so a
// stride-3 convolution reaches the CPU either way -- which is what this
// pair pins. Admitting one is a two-part change: the envelope entry *and*
// the loop entry.
// CHECK-LABEL: util.func public @dense_1x1_s3_falls_back
// CHECK-NOT: flow.dispatch @rocket_dynamic_executable
// CHECK: linalg.conv_2d_nhwc_hwcf
func.func @dense_1x1_s3_falls_back(
    %input: tensor<1x?x?x512xf16>,
    %filter: tensor<1x1x512x64xf16>,
    %init: tensor<1x?x?x64xf32>) -> tensor<1x?x?x64xf32> {
  %result = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<3> : vector<2xi64>}
      ins(%input, %filter : tensor<1x?x?x512xf16>, tensor<1x1x512x64xf16>)
      outs(%init : tensor<1x?x?x64xf32>) -> tensor<1x?x?x64xf32>
  return %result : tensor<1x?x?x64xf32>
}

// -----------------------------------------------------------------------
// fp16 depthwise, raised 512 -> 1536 on 2026-09-09. The old 512 was where
// the depthwise matchers were first written and nothing had gone back to
// it: `ConvPlan` plans fp16 depthwise to `MAX_DEPTHWISE_CHANNELS` (1792),
// the int8 depthwise rung had already moved to 1344, and MobileNetV2's own
// C=576 and C=960 depthwise convolutions sat above it and ran on the CPU.
//
// 1536 rather than 1792 because 1536 is what the compiled end-to-end gate
// measures: `tools/e2e_conv_regression.py`'s `depthwise_fp16_c576`,
// `_c960`, `_c1536` and `_c1536_s2` compile a Rocket and a CPU module from
// the same MLIR and compare them on `planck` -- max|error| 1.6e-4 to 2.4e-4,
// 0 mismatches, at atol 1e-3.

// CHECK-LABEL: util.func public @depthwise_3x3_channels_1536_matched
// CHECK-NOT: linalg.depthwise_conv_2d_nhwc_hwc
// CHECK: flow.dispatch @rocket_dynamic_depthwise_executable
func.func @depthwise_3x3_channels_1536_matched(
    %input: tensor<1x?x?x1536xf16>,
    %filter: tensor<3x3x1536xf16>,
    %init: tensor<1x?x?x1536xf32>) -> tensor<1x?x?x1536xf32> {
  %result = linalg.depthwise_conv_2d_nhwc_hwc
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x?x?x1536xf16>, tensor<3x3x1536xf16>)
      outs(%init : tensor<1x?x?x1536xf32>) -> tensor<1x?x?x1536xf32>
  return %result : tensor<1x?x?x1536xf32>
}

// CHECK-LABEL: util.func public @depthwise_3x3_channels_1537_falls_back
// CHECK-NOT: flow.dispatch @rocket_dynamic_depthwise_executable
// CHECK: linalg.depthwise_conv_2d_nhwc_hwc
func.func @depthwise_3x3_channels_1537_falls_back(
    %input: tensor<1x?x?x1537xf16>,
    %filter: tensor<3x3x1537xf16>,
    %init: tensor<1x?x?x1537xf32>) -> tensor<1x?x?x1537xf32> {
  %result = linalg.depthwise_conv_2d_nhwc_hwc
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x?x?x1537xf16>, tensor<3x3x1537xf16>)
      outs(%init : tensor<1x?x?x1537xf32>) -> tensor<1x?x?x1537xf32>
  return %result : tensor<1x?x?x1537xf32>
}

// MobileNetV2's own widest, which the old ceiling left on the CPU.
// CHECK-LABEL: util.func public @depthwise_3x3_channels_960_matched
// CHECK-NOT: linalg.depthwise_conv_2d_nhwc_hwc
// CHECK: flow.dispatch @rocket_dynamic_depthwise_executable
func.func @depthwise_3x3_channels_960_matched(
    %input: tensor<1x?x?x960xf16>,
    %filter: tensor<3x3x960xf16>,
    %init: tensor<1x?x?x960xf32>) -> tensor<1x?x?x960xf32> {
  %result = linalg.depthwise_conv_2d_nhwc_hwc
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x?x?x960xf16>, tensor<3x3x960xf16>)
      outs(%init : tensor<1x?x?x960xf32>) -> tensor<1x?x?x960xf32>
  return %result : tensor<1x?x?x960xf32>
}
