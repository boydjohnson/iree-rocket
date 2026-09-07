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

// The requantized int8 path claims a convolution *and* its requantization
// epilogue, replacing both with one dispatch that returns i8. This pins the
// canonical form that match is written against, which is also the form the
// ONNX QLinearConv fusion has to produce: a plain i8 x i8 -> i32 convolution
// over a zero init, then a single elementwise generic that adds the
// per-channel bias, scales, rounds, offsets by the output zero point, clamps
// and narrows.
//
// Two details of that form are load-bearing rather than stylistic, and the
// negative cases below are what stop them regressing silently:
//
//   * The scale, the zero point and the int8 clamp bounds are *operands* of
//     the generic. `transform.iree.match.cast_compatible_dag_from_root`
//     compares regions under a value mapping built only from the ops it
//     walked, so a value captured from the enclosing region can never match,
//     and a constant written into the body does not stay there -- the
//     canonicalizer hoists it out before the matcher runs.
//   * `Cin` goes to 512 at both kernel sizes, well past the int8_accumulator
//     matchers' 352 (1x1) and 32 (3x3). Those caps are a property of that
//     mode's 384-coefficient-bytes-per-output-channel DPU limit, and this
//     path does not have it.

// CHECK-LABEL: util.func public @requant_1x1_cin512_matched
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf
// CHECK: flow.dispatch @rocket_dynamic_int8_requant_executable
func.func @requant_1x1_cin512_matched(
    %input: tensor<1x4x4x512xi8>,
    %filter: tensor<1x1x512x64xi8>,
    %bias: tensor<64xi32>) -> tensor<1x4x4x64xi8> {
  %zero = arith.constant 0 : i32
  %scale = arith.constant 1.500000e-03 : f32
  %zp = arith.constant -6 : i32
  %int8_min = arith.constant -1.280000e+02 : f32
  %int8_max = arith.constant 1.270000e+02 : f32
  %acc_empty = tensor.empty() : tensor<1x4x4x64xi32>
  %acc_init = linalg.fill ins(%zero : i32) outs(%acc_empty : tensor<1x4x4x64xi32>) -> tensor<1x4x4x64xi32>
  %acc = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%input, %filter : tensor<1x4x4x512xi8>, tensor<1x1x512x64xi8>)
      outs(%acc_init : tensor<1x4x4x64xi32>) -> tensor<1x4x4x64xi32>
  %out_empty = tensor.empty() : tensor<1x4x4x64xi8>
  %out = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d3)>,
                       affine_map<(d0, d1, d2, d3) -> ()>,
                       affine_map<(d0, d1, d2, d3) -> ()>,
                       affine_map<(d0, d1, d2, d3) -> ()>,
                       affine_map<(d0, d1, d2, d3) -> ()>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%acc, %bias, %scale, %zp, %int8_min, %int8_max
          : tensor<1x4x4x64xi32>, tensor<64xi32>, f32, i32, f32, f32)
      outs(%out_empty : tensor<1x4x4x64xi8>) {
    ^bb0(%raw: i32, %channel_bias: i32, %s: f32, %z: i32, %low: f32, %high: f32, %unused: i8):
      %biased = arith.addi %raw, %channel_bias : i32
      %real = arith.sitofp %biased : i32 to f32
      %scaled = arith.mulf %real, %s : f32
      %rounded = math.roundeven %scaled : f32
      %zf = arith.sitofp %z : i32 to f32
      %offset = arith.addf %rounded, %zf : f32
      %low_clamped = arith.maximumf %offset, %low : f32
      %clamped = arith.minimumf %low_clamped, %high : f32
      %narrowed = arith.fptosi %clamped : f32 to i8
      linalg.yield %narrowed : i8
  } -> tensor<1x4x4x64xi8>
  return %out : tensor<1x4x4x64xi8>
}

// CHECK-LABEL: util.func public @requant_3x3_cin512_matched
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf
// CHECK: flow.dispatch @rocket_dynamic_int8_requant_executable
func.func @requant_3x3_cin512_matched(
    %input: tensor<1x6x6x512xi8>,
    %filter: tensor<3x3x512x64xi8>,
    %bias: tensor<64xi32>) -> tensor<1x4x4x64xi8> {
  %zero = arith.constant 0 : i32
  %scale = arith.constant 5.000000e-04 : f32
  %zp = arith.constant 0 : i32
  %int8_min = arith.constant -1.280000e+02 : f32
  %int8_max = arith.constant 1.270000e+02 : f32
  %acc_empty = tensor.empty() : tensor<1x4x4x64xi32>
  %acc_init = linalg.fill ins(%zero : i32) outs(%acc_empty : tensor<1x4x4x64xi32>) -> tensor<1x4x4x64xi32>
  %acc = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%input, %filter : tensor<1x6x6x512xi8>, tensor<3x3x512x64xi8>)
      outs(%acc_init : tensor<1x4x4x64xi32>) -> tensor<1x4x4x64xi32>
  %out_empty = tensor.empty() : tensor<1x4x4x64xi8>
  %out = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d3)>,
                       affine_map<(d0, d1, d2, d3) -> ()>,
                       affine_map<(d0, d1, d2, d3) -> ()>,
                       affine_map<(d0, d1, d2, d3) -> ()>,
                       affine_map<(d0, d1, d2, d3) -> ()>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%acc, %bias, %scale, %zp, %int8_min, %int8_max
          : tensor<1x4x4x64xi32>, tensor<64xi32>, f32, i32, f32, f32)
      outs(%out_empty : tensor<1x4x4x64xi8>) {
    ^bb0(%raw: i32, %channel_bias: i32, %s: f32, %z: i32, %low: f32, %high: f32, %unused: i8):
      %biased = arith.addi %raw, %channel_bias : i32
      %real = arith.sitofp %biased : i32 to f32
      %scaled = arith.mulf %real, %s : f32
      %rounded = math.roundeven %scaled : f32
      %zf = arith.sitofp %z : i32 to f32
      %offset = arith.addf %rounded, %zf : f32
      %low_clamped = arith.maximumf %offset, %low : f32
      %clamped = arith.minimumf %low_clamped, %high : f32
      %narrowed = arith.fptosi %clamped : f32 to i8
      linalg.yield %narrowed : i8
  } -> tensor<1x4x4x64xi8>
  return %out : tensor<1x4x4x64xi8>
}

// Cin 1536 is past the requantized matchers' measured ceiling of 1344 --
// itself the widest Cin MobileNetV2-static-int8 asks for, which is where the
// bound was raised to on 2026-09-06 -- so the requantized loop must decline
// it. This case pins one thing only: that exceeding the bound leaves the
// *requantized* path alone. Where the convolution lands afterwards is the
// accumulator matchers' business, and `rocket_int8_match_boundaries.mlir` is
// where that bound is pinned.
//
// It lands on the accumulator path since 2026-09-06, when
// `MAX_INT8_INPUT_CHANNELS` went to 3584 and `@match_dynamic_conv2d_int8`
// followed it: this convolution used to fall out of both loops and stay on
// the CPU. The positive check is the accumulator dispatch rather than a
// surviving `linalg.conv_2d_nhwc_hwcf` for exactly that reason -- the
// assertion here is about which of the two int8 loops claims it, not about
// whether anything does.
// CHECK-LABEL: util.func public @requant_1x1_cin1536_declined
// CHECK-NOT: @rocket_dynamic_int8_requant_executable
// CHECK: flow.dispatch @rocket_dynamic_int8_executable
func.func @requant_1x1_cin1536_declined(
    %input: tensor<1x4x4x1536xi8>,
    %filter: tensor<1x1x1536x64xi8>,
    %bias: tensor<64xi32>) -> tensor<1x4x4x64xi8> {
  %zero = arith.constant 0 : i32
  %scale = arith.constant 1.500000e-03 : f32
  %zp = arith.constant 0 : i32
  %int8_min = arith.constant -1.280000e+02 : f32
  %int8_max = arith.constant 1.270000e+02 : f32
  %acc_empty = tensor.empty() : tensor<1x4x4x64xi32>
  %acc_init = linalg.fill ins(%zero : i32) outs(%acc_empty : tensor<1x4x4x64xi32>) -> tensor<1x4x4x64xi32>
  %acc = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%input, %filter : tensor<1x4x4x1536xi8>, tensor<1x1x1536x64xi8>)
      outs(%acc_init : tensor<1x4x4x64xi32>) -> tensor<1x4x4x64xi32>
  %out_empty = tensor.empty() : tensor<1x4x4x64xi8>
  %out = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d3)>,
                       affine_map<(d0, d1, d2, d3) -> ()>,
                       affine_map<(d0, d1, d2, d3) -> ()>,
                       affine_map<(d0, d1, d2, d3) -> ()>,
                       affine_map<(d0, d1, d2, d3) -> ()>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%acc, %bias, %scale, %zp, %int8_min, %int8_max
          : tensor<1x4x4x64xi32>, tensor<64xi32>, f32, i32, f32, f32)
      outs(%out_empty : tensor<1x4x4x64xi8>) {
    ^bb0(%raw: i32, %channel_bias: i32, %s: f32, %z: i32, %low: f32, %high: f32, %unused: i8):
      %biased = arith.addi %raw, %channel_bias : i32
      %real = arith.sitofp %biased : i32 to f32
      %scaled = arith.mulf %real, %s : f32
      %rounded = math.roundeven %scaled : f32
      %zf = arith.sitofp %z : i32 to f32
      %offset = arith.addf %rounded, %zf : f32
      %low_clamped = arith.maximumf %offset, %low : f32
      %clamped = arith.minimumf %low_clamped, %high : f32
      %narrowed = arith.fptosi %clamped : f32 to i8
      linalg.yield %narrowed : i8
  } -> tensor<1x4x4x64xi8>
  return %out : tensor<1x4x4x64xi8>
}

// The depthwise twin, which is where the model's remaining i32 epilogues
// were. Same canonical form, a `linalg.depthwise_conv_2d_nhwc_hwc` producer
// and a rank-3 filter.
//
// This case exists because the first version of the depthwise matcher checked
// for a 1x1 kernel and every depthwise convolution in MobileNetV2 is 3x3. It
// declined the entire model and the accumulator matchers claimed the
// convolutions straight back, which is indistinguishable from the matcher not
// existing -- nothing failed, placement just did not move. A matched-case test
// is the only thing that catches that.

// CHECK-LABEL: util.func public @requant_depthwise_3x3_matched
// CHECK-NOT: linalg.depthwise_conv_2d_nhwc_hwc
// CHECK: flow.dispatch @rocket_dynamic_depthwise_int8_requant_executable
func.func @requant_depthwise_3x3_matched(
    %input: tensor<1x6x6x48xi8>,
    %filter: tensor<3x3x48xi8>,
    %bias: tensor<48xi32>) -> tensor<1x4x4x48xi8> {
  %zero = arith.constant 0 : i32
  %scale = arith.constant 1.500000e-03 : f32
  %zp = arith.constant -6 : i32
  %int8_min = arith.constant -1.280000e+02 : f32
  %int8_max = arith.constant 1.270000e+02 : f32
  %acc_empty = tensor.empty() : tensor<1x4x4x48xi32>
  %acc_init = linalg.fill ins(%zero : i32) outs(%acc_empty : tensor<1x4x4x48xi32>) -> tensor<1x4x4x48xi32>
  %acc = linalg.depthwise_conv_2d_nhwc_hwc
      {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%input, %filter : tensor<1x6x6x48xi8>, tensor<3x3x48xi8>)
      outs(%acc_init : tensor<1x4x4x48xi32>) -> tensor<1x4x4x48xi32>
  %out_empty = tensor.empty() : tensor<1x4x4x48xi8>
  %out = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d3)>,
                       affine_map<(d0, d1, d2, d3) -> ()>,
                       affine_map<(d0, d1, d2, d3) -> ()>,
                       affine_map<(d0, d1, d2, d3) -> ()>,
                       affine_map<(d0, d1, d2, d3) -> ()>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%acc, %bias, %scale, %zp, %int8_min, %int8_max
          : tensor<1x4x4x48xi32>, tensor<48xi32>, f32, i32, f32, f32)
      outs(%out_empty : tensor<1x4x4x48xi8>) {
    ^bb0(%raw: i32, %channel_bias: i32, %s: f32, %z: i32, %low: f32, %high: f32, %unused: i8):
      %biased = arith.addi %raw, %channel_bias : i32
      %real = arith.sitofp %biased : i32 to f32
      %scaled = arith.mulf %real, %s : f32
      %rounded = math.roundeven %scaled : f32
      %zf = arith.sitofp %z : i32 to f32
      %offset = arith.addf %rounded, %zf : f32
      %low_clamped = arith.maximumf %offset, %low : f32
      %clamped = arith.minimumf %low_clamped, %high : f32
      %narrowed = arith.fptosi %clamped : f32 to i8
      linalg.yield %narrowed : i8
  } -> tensor<1x4x4x48xi8>
  return %out : tensor<1x4x4x48xi8>
}
