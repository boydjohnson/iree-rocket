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

// Cin 528 is one 16-channel atom past the requantized matchers' measured
// plain-int8 ceiling of 512, so the requantized loop must decline it. The
// convolution is then still standing and is offered to the int8_accumulator
// matchers, which -- since the per-channel coefficient limit that used to cap
// them at Cin 352 was retracted as a measurement artifact -- now claim it.
// So this case pins one thing only: that exceeding the bound leaves the
// *requantized* path alone. Where it lands afterwards is the accumulator
// matchers' own business, and `rocket_int8_match_boundaries.mlir` is where
// that bound is pinned.
// CHECK-LABEL: util.func public @requant_1x1_cin528_declined
// CHECK-NOT: @rocket_dynamic_int8_requant_executable
// CHECK: flow.dispatch @rocket_dynamic_int8_executable
func.func @requant_1x1_cin528_declined(
    %input: tensor<1x4x4x528xi8>,
    %filter: tensor<1x1x528x64xi8>,
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
      ins(%input, %filter : tensor<1x4x4x528xi8>, tensor<1x1x528x64xi8>)
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
