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

// A requantized int8 3x3 with "same" padding, in the form the ONNX importer
// produces for a QLinearConv: an NCHW `tensor.pad` filled with the input
// zero point in front of `linalg.conv_2d_nchw_fchw_q`, then the
// dequantize/requantize generics. -128 is the signed encoding of a `ui8`
// zero point of 0, what every ReLU output carries once the quantizer folds
// the ReLU into the conv.
//
// Three passes have to agree for the CNA to do the padding:
// rocket-transpose-quantized-conv-to-nhwc moves the pad behind its
// transpose (so it sits directly in front of the NHWC conv),
// rocket-fold-conv-pad marks the pair, and the pad1 matcher claims pad,
// conv and epilogue together and hands the dispatch the *unpadded* tensor.
// The executable bakes the fill as `input_zero_point = -128`, which the
// driver programs into CNA_PAD_CON1.pad_value; ResNet50 int8 is
// bit-identical with the pad on the CPU and on the CNA.

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: util.func public @pad1_3x3
// CHECK-NOT: tensor.pad
// CHECK: flow.dispatch @rocket_dynamic_int8_requant_pad1_executable::
// The dispatch reads the 56x56 source, not a 58x58 copy.
// CHECK-SAME: tensor<1x?x?x?xi8>{%c56, %c56, %c64}
// CHECK-NOT: tensor.pad
// CHECK: util.return

util.func public @pad1_3x3(%x: tensor<1x64x56x56xi8>, %bias: tensor<64xi32>) -> tensor<1x64x56x56xi8> {
  %fill = arith.constant -128 : i8
  %x_zp = arith.constant -128 : i32
  %w_zp = arith.constant 0 : i32
  %weights = arith.constant dense<3> : tensor<64x64x3x3xi8>
  %acc_scale = arith.constant 2.000000e-04 : f32
  %out_scale = arith.constant 3.125000e-02 : f32
  %zero = arith.constant 0.000000e+00 : f32
  %max = arith.constant 2.550000e+02 : f32
  %padded = tensor.pad %x low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a: index, %b: index, %c: index, %d: index):
    tensor.yield %fill : i8
  } : tensor<1x64x56x56xi8> to tensor<1x64x58x58xi8>
  %acc_empty = tensor.empty() : tensor<1x64x56x56xi32>
  %acc_init = linalg.broadcast ins(%bias : tensor<64xi32>) outs(%acc_empty : tensor<1x64x56x56xi32>) dimensions = [0, 2, 3]
  %acc = linalg.conv_2d_nchw_fchw_q {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%padded, %weights, %x_zp, %w_zp : tensor<1x64x58x58xi8>, tensor<64x64x3x3xi8>, i32, i32)
      outs(%acc_init : tensor<1x64x56x56xi32>) -> tensor<1x64x56x56xi32>
  %real_empty = tensor.empty() : tensor<1x64x56x56xf32>
  %real = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%acc : tensor<1x64x56x56xi32>) outs(%real_empty : tensor<1x64x56x56xf32>) {
  ^bb0(%in: i32, %o: f32):
    %f = arith.sitofp %in : i32 to f32
    %s = arith.mulf %f, %acc_scale : f32
    linalg.yield %s : f32
  } -> tensor<1x64x56x56xf32>
  %q_empty = tensor.empty() : tensor<1x64x56x56xi8>
  %q = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%real : tensor<1x64x56x56xf32>) outs(%q_empty : tensor<1x64x56x56xi8>) {
  ^bb0(%in: f32, %o: i8):
    %d = arith.divf %in, %out_scale : f32
    %r = math.roundeven %d : f32
    %z = arith.addf %r, %zero : f32
    %lo = arith.maximumf %z, %zero : f32
    %hi = arith.minimumf %lo, %max : f32
    %n = arith.fptoui %hi : f32 to i8
    linalg.yield %n : i8
  } -> tensor<1x64x56x56xi8>
  util.return %q : tensor<1x64x56x56xi8>
}

// The stride-2 twin: ResNet50 v1.5 puts its downsampling on the first 3x3
// of each stage, with the same symmetric pad of 1.
// CHECK-LABEL: util.func public @pad1_3x3_s2
// CHECK-NOT: tensor.pad
// CHECK: flow.dispatch @rocket_dynamic_int8_requant_pad1_s2_executable::
// CHECK: util.return

util.func public @pad1_3x3_s2(%x: tensor<1x64x56x56xi8>, %bias: tensor<64xi32>) -> tensor<1x64x28x28xi8> {
  %fill = arith.constant -128 : i8
  %x_zp = arith.constant -128 : i32
  %w_zp = arith.constant 0 : i32
  %weights = arith.constant dense<3> : tensor<64x64x3x3xi8>
  %acc_scale = arith.constant 2.000000e-04 : f32
  %out_scale = arith.constant 3.125000e-02 : f32
  %zero = arith.constant 0.000000e+00 : f32
  %max = arith.constant 2.550000e+02 : f32
  %padded = tensor.pad %x low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a: index, %b: index, %c: index, %d: index):
    tensor.yield %fill : i8
  } : tensor<1x64x56x56xi8> to tensor<1x64x58x58xi8>
  %acc_empty = tensor.empty() : tensor<1x64x28x28xi32>
  %acc_init = linalg.broadcast ins(%bias : tensor<64xi32>) outs(%acc_empty : tensor<1x64x28x28xi32>) dimensions = [0, 2, 3]
  %acc = linalg.conv_2d_nchw_fchw_q {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
      ins(%padded, %weights, %x_zp, %w_zp : tensor<1x64x58x58xi8>, tensor<64x64x3x3xi8>, i32, i32)
      outs(%acc_init : tensor<1x64x28x28xi32>) -> tensor<1x64x28x28xi32>
  %real_empty = tensor.empty() : tensor<1x64x28x28xf32>
  %real = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%acc : tensor<1x64x28x28xi32>) outs(%real_empty : tensor<1x64x28x28xf32>) {
  ^bb0(%in: i32, %o: f32):
    %f = arith.sitofp %in : i32 to f32
    %s = arith.mulf %f, %acc_scale : f32
    linalg.yield %s : f32
  } -> tensor<1x64x28x28xf32>
  %q_empty = tensor.empty() : tensor<1x64x28x28xi8>
  %q = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%real : tensor<1x64x28x28xf32>) outs(%q_empty : tensor<1x64x28x28xi8>) {
  ^bb0(%in: f32, %o: i8):
    %d = arith.divf %in, %out_scale : f32
    %r = math.roundeven %d : f32
    %z = arith.addf %r, %zero : f32
    %lo = arith.maximumf %z, %zero : f32
    %hi = arith.minimumf %lo, %max : f32
    %n = arith.fptoui %hi : f32 to i8
    linalg.yield %n : i8
  } -> tensor<1x64x28x28xi8>
  util.return %q : tensor<1x64x28x28xi8>
}

// Any other fill is left alone: the targets bake -128 and nothing else, so a
// pad filled with -127 (a `ui8` zero point of 1) stays materialized and the
// unpadded requantized matcher claims the conv reading it.
// CHECK-LABEL: util.func public @pad1_3x3_other_fill_declines
// CHECK: tensor.pad
// CHECK: flow.dispatch @rocket_dynamic_int8_requant_executable::
// CHECK-NOT: rocket_dynamic_int8_requant_pad1
// CHECK: util.return

util.func public @pad1_3x3_other_fill_declines(%x: tensor<1x64x56x56xi8>, %bias: tensor<64xi32>) -> tensor<1x64x56x56xi8> {
  %fill = arith.constant -127 : i8
  %x_zp = arith.constant -127 : i32
  %w_zp = arith.constant 0 : i32
  %weights = arith.constant dense<3> : tensor<64x64x3x3xi8>
  %acc_scale = arith.constant 2.000000e-04 : f32
  %out_scale = arith.constant 3.125000e-02 : f32
  %zero = arith.constant 0.000000e+00 : f32
  %max = arith.constant 2.550000e+02 : f32
  %padded = tensor.pad %x low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a: index, %b: index, %c: index, %d: index):
    tensor.yield %fill : i8
  } : tensor<1x64x56x56xi8> to tensor<1x64x58x58xi8>
  %acc_empty = tensor.empty() : tensor<1x64x56x56xi32>
  %acc_init = linalg.broadcast ins(%bias : tensor<64xi32>) outs(%acc_empty : tensor<1x64x56x56xi32>) dimensions = [0, 2, 3]
  %acc = linalg.conv_2d_nchw_fchw_q {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%padded, %weights, %x_zp, %w_zp : tensor<1x64x58x58xi8>, tensor<64x64x3x3xi8>, i32, i32)
      outs(%acc_init : tensor<1x64x56x56xi32>) -> tensor<1x64x56x56xi32>
  %real_empty = tensor.empty() : tensor<1x64x56x56xf32>
  %real = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%acc : tensor<1x64x56x56xi32>) outs(%real_empty : tensor<1x64x56x56xf32>) {
  ^bb0(%in: i32, %o: f32):
    %f = arith.sitofp %in : i32 to f32
    %s = arith.mulf %f, %acc_scale : f32
    linalg.yield %s : f32
  } -> tensor<1x64x56x56xf32>
  %q_empty = tensor.empty() : tensor<1x64x56x56xi8>
  %q = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%real : tensor<1x64x56x56xf32>) outs(%q_empty : tensor<1x64x56x56xi8>) {
  ^bb0(%in: f32, %o: i8):
    %d = arith.divf %in, %out_scale : f32
    %r = math.roundeven %d : f32
    %z = arith.addf %r, %zero : f32
    %lo = arith.maximumf %z, %zero : f32
    %hi = arith.minimumf %lo, %max : f32
    %n = arith.fptoui %hi : f32 to i8
    linalg.yield %n : i8
  } -> tensor<1x64x56x56xi8>
  util.return %q : tensor<1x64x56x56xi8>
}

// A -128 pad in front of a 3x3 the requantized path does *not* take stays
// materialized. Here the bias is a splat constant, the broadcast folds into
// the init, `rocket-fuse-int8-requant-epilogue` does not apply, and the conv
// takes the int8_accumulator path reading the padded tensor. The fold
// leaves such a pad unmarked: only a conv feeding the requantized epilogue
// has a pad1 matcher to claim it.
// CHECK-LABEL: util.func public @accumulator_3x3_keeps_npu
// CHECK: tensor.pad
// CHECK: flow.dispatch @rocket_dynamic_int8_executable::
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf
// CHECK: util.return

util.func public @accumulator_3x3_keeps_npu(%x: tensor<1x64x56x56xi8>) -> tensor<1x64x56x56xi8> {
  %fill = arith.constant -128 : i8
  %x_zp = arith.constant -128 : i32
  %w_zp = arith.constant 0 : i32
  %weights = arith.constant dense<3> : tensor<64x64x3x3xi8>
  %bias = arith.constant dense<-700> : tensor<64xi32>
  %acc_scale = arith.constant 2.000000e-04 : f32
  %out_scale = arith.constant 3.125000e-02 : f32
  %zero = arith.constant 0.000000e+00 : f32
  %max = arith.constant 2.550000e+02 : f32
  %padded = tensor.pad %x low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a: index, %b: index, %c: index, %d: index):
    tensor.yield %fill : i8
  } : tensor<1x64x56x56xi8> to tensor<1x64x58x58xi8>
  %acc_empty = tensor.empty() : tensor<1x64x56x56xi32>
  %acc_init = linalg.broadcast ins(%bias : tensor<64xi32>) outs(%acc_empty : tensor<1x64x56x56xi32>) dimensions = [0, 2, 3]
  %acc = linalg.conv_2d_nchw_fchw_q {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%padded, %weights, %x_zp, %w_zp : tensor<1x64x58x58xi8>, tensor<64x64x3x3xi8>, i32, i32)
      outs(%acc_init : tensor<1x64x56x56xi32>) -> tensor<1x64x56x56xi32>
  %real_empty = tensor.empty() : tensor<1x64x56x56xf32>
  %real = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%acc : tensor<1x64x56x56xi32>) outs(%real_empty : tensor<1x64x56x56xf32>) {
  ^bb0(%in: i32, %o: f32):
    %f = arith.sitofp %in : i32 to f32
    %s = arith.mulf %f, %acc_scale : f32
    linalg.yield %s : f32
  } -> tensor<1x64x56x56xf32>
  %q_empty = tensor.empty() : tensor<1x64x56x56xi8>
  %q = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%real : tensor<1x64x56x56xf32>) outs(%q_empty : tensor<1x64x56x56xi8>) {
  ^bb0(%in: f32, %o: i8):
    %d = arith.divf %in, %out_scale : f32
    %r = math.roundeven %d : f32
    %z = arith.addf %r, %zero : f32
    %lo = arith.maximumf %z, %zero : f32
    %hi = arith.minimumf %lo, %max : f32
    %n = arith.fptoui %hi : f32 to i8
    linalg.yield %n : i8
  } -> tensor<1x64x56x56xi8>
  util.return %q : tensor<1x64x56x56xi8>
}
