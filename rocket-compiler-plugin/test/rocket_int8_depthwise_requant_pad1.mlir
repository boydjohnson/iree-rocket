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

// A requantized int8 depthwise 3x3 with "same" padding, in the form the ONNX
// importer produces for a grouped QLinearConv: the depthwise op is already
// NHWC (`linalg.depthwise_conv_2d_nhwc_hwc_q`), but its pad is done in NCHW,
// in front of the importer's own NCHW->NHWC transpose -- so the transpose
// sits between the pad and the convolution. -128 is the signed encoding of a
// `ui8` zero point of 0, which every expand conv's output carries once the
// quantizer folds its ReLU6 (`Clip`) into the conv.
//
// Three passes have to agree for the CNA to do the padding, as on the dense
// pad1 path (rocket_int8_requant_pad1.mlir):
// rocket-transpose-quantized-conv-to-nhwc moves the pad below the transpose
// (so it sits directly in front of the convolution), rocket-fold-conv-pad
// marks the pair, and the depthwise pad1 matcher claims pad, convolution and
// epilogue together and hands the dispatch the *unpadded* tensor. Before,
// every depthwise input in an int8 MobileNetV2 was a CPU pad copy plus a
// transpose.

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: util.func public @depthwise_pad1
// CHECK-NOT: tensor.pad
// CHECK: flow.dispatch @rocket_dynamic_depthwise_int8_requant_pad1_executable::
// The dispatch reads the 56x56 source, not a 58x58 copy.
// CHECK-SAME: tensor<1x?x?x?xi8>{%c56, %c56, %c144}
// CHECK-NOT: tensor.pad
// CHECK: util.return

util.func public @depthwise_pad1(%x: tensor<1x144x56x56xi8>, %bias: tensor<144xi32>) -> tensor<1x144x56x56xi8> {
  %fill = arith.constant -128 : i8
  %x_zp = arith.constant -128 : i32
  %w_zp = arith.constant 0 : i32
  %weights = arith.constant dense<3> : tensor<144x3x3xi8>
  %acc_scale = arith.constant 2.000000e-04 : f32
  %out_scale = arith.constant 3.125000e-02 : f32
  %zero = arith.constant 0.000000e+00 : f32
  %max = arith.constant 2.550000e+02 : f32
  %padded = tensor.pad %x low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a: index, %b: index, %c: index, %d: index):
    tensor.yield %fill : i8
  } : tensor<1x144x56x56xi8> to tensor<1x144x58x58xi8>
  %nhwc_empty = tensor.empty() : tensor<1x58x58x144xi8>
  %nhwc = linalg.transpose ins(%padded : tensor<1x144x58x58xi8>) outs(%nhwc_empty : tensor<1x58x58x144xi8>) permutation = [0, 2, 3, 1]
  %hwc_empty = tensor.empty() : tensor<3x3x144xi8>
  %hwc = linalg.transpose ins(%weights : tensor<144x3x3xi8>) outs(%hwc_empty : tensor<3x3x144xi8>) permutation = [1, 2, 0]
  %acc_empty = tensor.empty() : tensor<1x56x56x144xi32>
  %acc_init = linalg.broadcast ins(%bias : tensor<144xi32>) outs(%acc_empty : tensor<1x56x56x144xi32>) dimensions = [0, 1, 2]
  %acc = linalg.depthwise_conv_2d_nhwc_hwc_q {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%nhwc, %hwc, %x_zp, %w_zp : tensor<1x58x58x144xi8>, tensor<3x3x144xi8>, i32, i32)
      outs(%acc_init : tensor<1x56x56x144xi32>) -> tensor<1x56x56x144xi32>
  %nchw_empty = tensor.empty() : tensor<1x144x56x56xi32>
  %nchw = linalg.transpose ins(%acc : tensor<1x56x56x144xi32>) outs(%nchw_empty : tensor<1x144x56x56xi32>) permutation = [0, 3, 1, 2]
  %real_empty = tensor.empty() : tensor<1x144x56x56xf32>
  %real = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%nchw : tensor<1x144x56x56xi32>) outs(%real_empty : tensor<1x144x56x56xf32>) {
  ^bb0(%in: i32, %o: f32):
    %f = arith.sitofp %in : i32 to f32
    %s = arith.mulf %f, %acc_scale : f32
    linalg.yield %s : f32
  } -> tensor<1x144x56x56xf32>
  %q_empty = tensor.empty() : tensor<1x144x56x56xi8>
  %q = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%real : tensor<1x144x56x56xf32>) outs(%q_empty : tensor<1x144x56x56xi8>) {
  ^bb0(%in: f32, %o: i8):
    %d = arith.divf %in, %out_scale : f32
    %r = math.roundeven %d : f32
    %z = arith.addf %r, %zero : f32
    %lo = arith.maximumf %z, %zero : f32
    %hi = arith.minimumf %lo, %max : f32
    %n = arith.fptoui %hi : f32 to i8
    linalg.yield %n : i8
  } -> tensor<1x144x56x56xi8>
  util.return %q : tensor<1x144x56x56xi8>
}

// The stride-2 twin: MobileNetV2's four downsampling depthwise layers. These
// took the int8 accumulator path before, with a CPU i32 -> i8 requantization
// behind each; on the pad1 s2 executable the DPU requantizes and the result
// is the final i8 tensor.
// CHECK-LABEL: util.func public @depthwise_pad1_s2
// CHECK-NOT: tensor.pad
// CHECK: flow.dispatch @rocket_dynamic_depthwise_int8_requant_pad1_s2_executable::
// CHECK-SAME: tensor<1x?x?x?xi8>{%c112, %c112, %c96}
// CHECK-NOT: rocket_dynamic_depthwise_int8_executable_s2
// CHECK: util.return

util.func public @depthwise_pad1_s2(%x: tensor<1x96x112x112xi8>, %bias: tensor<96xi32>) -> tensor<1x96x56x56xi8> {
  %fill = arith.constant -128 : i8
  %x_zp = arith.constant -128 : i32
  %w_zp = arith.constant 0 : i32
  %weights = arith.constant dense<3> : tensor<96x3x3xi8>
  %acc_scale = arith.constant 2.000000e-04 : f32
  %out_scale = arith.constant 3.125000e-02 : f32
  %zero = arith.constant 0.000000e+00 : f32
  %max = arith.constant 2.550000e+02 : f32
  %padded = tensor.pad %x low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a: index, %b: index, %c: index, %d: index):
    tensor.yield %fill : i8
  } : tensor<1x96x112x112xi8> to tensor<1x96x114x114xi8>
  %nhwc_empty = tensor.empty() : tensor<1x114x114x96xi8>
  %nhwc = linalg.transpose ins(%padded : tensor<1x96x114x114xi8>) outs(%nhwc_empty : tensor<1x114x114x96xi8>) permutation = [0, 2, 3, 1]
  %hwc_empty = tensor.empty() : tensor<3x3x96xi8>
  %hwc = linalg.transpose ins(%weights : tensor<96x3x3xi8>) outs(%hwc_empty : tensor<3x3x96xi8>) permutation = [1, 2, 0]
  %acc_empty = tensor.empty() : tensor<1x56x56x96xi32>
  %acc_init = linalg.broadcast ins(%bias : tensor<96xi32>) outs(%acc_empty : tensor<1x56x56x96xi32>) dimensions = [0, 1, 2]
  %acc = linalg.depthwise_conv_2d_nhwc_hwc_q {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
      ins(%nhwc, %hwc, %x_zp, %w_zp : tensor<1x114x114x96xi8>, tensor<3x3x96xi8>, i32, i32)
      outs(%acc_init : tensor<1x56x56x96xi32>) -> tensor<1x56x56x96xi32>
  %nchw_empty = tensor.empty() : tensor<1x96x56x56xi32>
  %nchw = linalg.transpose ins(%acc : tensor<1x56x56x96xi32>) outs(%nchw_empty : tensor<1x96x56x56xi32>) permutation = [0, 3, 1, 2]
  %real_empty = tensor.empty() : tensor<1x96x56x56xf32>
  %real = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%nchw : tensor<1x96x56x56xi32>) outs(%real_empty : tensor<1x96x56x56xf32>) {
  ^bb0(%in: i32, %o: f32):
    %f = arith.sitofp %in : i32 to f32
    %s = arith.mulf %f, %acc_scale : f32
    linalg.yield %s : f32
  } -> tensor<1x96x56x56xf32>
  %q_empty = tensor.empty() : tensor<1x96x56x56xi8>
  %q = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%real : tensor<1x96x56x56xf32>) outs(%q_empty : tensor<1x96x56x56xi8>) {
  ^bb0(%in: f32, %o: i8):
    %d = arith.divf %in, %out_scale : f32
    %r = math.roundeven %d : f32
    %z = arith.addf %r, %zero : f32
    %lo = arith.maximumf %z, %zero : f32
    %hi = arith.minimumf %lo, %max : f32
    %n = arith.fptoui %hi : f32 to i8
    linalg.yield %n : i8
  } -> tensor<1x96x56x56xi8>
  util.return %q : tensor<1x96x56x56xi8>
}

// Any other fill is left alone: the targets bake -128 and nothing else. The
// pad still moves to the NHWC side (a pure layout change, right for any
// constant fill), but it is not marked, so it stays materialized and the
// unpadded requantized depthwise matcher claims the convolution reading it.
// CHECK-LABEL: util.func public @depthwise_pad1_other_fill_declines
// CHECK: tensor.pad
// CHECK: flow.dispatch @rocket_dynamic_depthwise_int8_requant_executable::
// CHECK-NOT: rocket_dynamic_depthwise_int8_requant_pad1
// CHECK: util.return

util.func public @depthwise_pad1_other_fill_declines(%x: tensor<1x144x56x56xi8>, %bias: tensor<144xi32>) -> tensor<1x144x56x56xi8> {
  %fill = arith.constant -127 : i8
  %x_zp = arith.constant -127 : i32
  %w_zp = arith.constant 0 : i32
  %weights = arith.constant dense<3> : tensor<144x3x3xi8>
  %acc_scale = arith.constant 2.000000e-04 : f32
  %out_scale = arith.constant 3.125000e-02 : f32
  %zero = arith.constant 0.000000e+00 : f32
  %max = arith.constant 2.550000e+02 : f32
  %padded = tensor.pad %x low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a: index, %b: index, %c: index, %d: index):
    tensor.yield %fill : i8
  } : tensor<1x144x56x56xi8> to tensor<1x144x58x58xi8>
  %nhwc_empty = tensor.empty() : tensor<1x58x58x144xi8>
  %nhwc = linalg.transpose ins(%padded : tensor<1x144x58x58xi8>) outs(%nhwc_empty : tensor<1x58x58x144xi8>) permutation = [0, 2, 3, 1]
  %hwc_empty = tensor.empty() : tensor<3x3x144xi8>
  %hwc = linalg.transpose ins(%weights : tensor<144x3x3xi8>) outs(%hwc_empty : tensor<3x3x144xi8>) permutation = [1, 2, 0]
  %acc_empty = tensor.empty() : tensor<1x56x56x144xi32>
  %acc_init = linalg.broadcast ins(%bias : tensor<144xi32>) outs(%acc_empty : tensor<1x56x56x144xi32>) dimensions = [0, 1, 2]
  %acc = linalg.depthwise_conv_2d_nhwc_hwc_q {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%nhwc, %hwc, %x_zp, %w_zp : tensor<1x58x58x144xi8>, tensor<3x3x144xi8>, i32, i32)
      outs(%acc_init : tensor<1x56x56x144xi32>) -> tensor<1x56x56x144xi32>
  %nchw_empty = tensor.empty() : tensor<1x144x56x56xi32>
  %nchw = linalg.transpose ins(%acc : tensor<1x56x56x144xi32>) outs(%nchw_empty : tensor<1x144x56x56xi32>) permutation = [0, 3, 1, 2]
  %real_empty = tensor.empty() : tensor<1x144x56x56xf32>
  %real = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%nchw : tensor<1x144x56x56xi32>) outs(%real_empty : tensor<1x144x56x56xf32>) {
  ^bb0(%in: i32, %o: f32):
    %f = arith.sitofp %in : i32 to f32
    %s = arith.mulf %f, %acc_scale : f32
    linalg.yield %s : f32
  } -> tensor<1x144x56x56xf32>
  %q_empty = tensor.empty() : tensor<1x144x56x56xi8>
  %q = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%real : tensor<1x144x56x56xf32>) outs(%q_empty : tensor<1x144x56x56xi8>) {
  ^bb0(%in: f32, %o: i8):
    %d = arith.divf %in, %out_scale : f32
    %r = math.roundeven %d : f32
    %z = arith.addf %r, %zero : f32
    %lo = arith.maximumf %z, %zero : f32
    %hi = arith.minimumf %lo, %max : f32
    %n = arith.fptoui %hi : f32 to i8
    linalg.yield %n : i8
  } -> tensor<1x144x56x56xi8>
  util.return %q : tensor<1x144x56x56xi8>
}
