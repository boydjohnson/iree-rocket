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

// A quantized residual tail as the ONNX importer emits it for an Add whose
// operands both come through DequantizeLinear (tools/quantize_onnx_int8.py
// --quantize-residual-skip): two `ui8` NCHW dequantizations -- one with a
// zero point, one spelled `uitofp` because its zero point is 0 -- an add, the
// importer's ReLU, and a `ui8` QuantizeLinear. rocket-fuse-int8-residual-add
// puts it into the signed NHWC form @match_elementwise_add_int8_relu claims,
// and the NPU's int8 EW add runs it.

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: util.func public @residual_add_relu
// The zero points reach the dispatch in the signed encoding: 163 - 128 for
// `a`, 0 - 128 for `b`, and the output's 0 - 128.
// CHECK: flow.dispatch @rocket_elementwise_add_int8_relu_executable::
// CHECK-SAME: %c35_i32
// CHECK-SAME: %c-128_i32
// CHECK-NOT: arith.roundeven
// CHECK: util.return
util.func public @residual_add_relu(%a: tensor<1x64x8x8xi8>, %b: tensor<1x64x8x8xi8>) -> tensor<1x64x8x8xi8> {
  %a_scale = arith.constant 3.500000e-02 : f32
  %b_scale = arith.constant 4.200000e-02 : f32
  %out_scale = arith.constant 7.000000e-02 : f32
  %a_zp = arith.constant 163 : i32
  %zero = arith.constant 0.000000e+00 : f32
  %max = arith.constant 2.550000e+02 : f32
  %f_empty = tensor.empty() : tensor<1x64x8x8xf32>
  %fa = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%a : tensor<1x64x8x8xi8>) outs(%f_empty : tensor<1x64x8x8xf32>) {
  ^bb0(%in: i8, %out: f32):
    %w = arith.extui %in : i8 to i32
    %c = arith.subi %w, %a_zp : i32
    %f = arith.sitofp %c : i32 to f32
    %s = arith.mulf %f, %a_scale : f32
    linalg.yield %s : f32
  } -> tensor<1x64x8x8xf32>
  %fb = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%b : tensor<1x64x8x8xi8>) outs(%f_empty : tensor<1x64x8x8xf32>) {
  ^bb0(%in: i8, %out: f32):
    %f = arith.uitofp %in : i8 to f32
    %s = arith.mulf %f, %b_scale : f32
    linalg.yield %s : f32
  } -> tensor<1x64x8x8xf32>
  %sum = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%fa, %fb : tensor<1x64x8x8xf32>, tensor<1x64x8x8xf32>) outs(%f_empty : tensor<1x64x8x8xf32>) {
  ^bb0(%x: f32, %y: f32, %out: f32):
    %r = arith.addf %x, %y : f32
    linalg.yield %r : f32
  } -> tensor<1x64x8x8xf32>
  %relu = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%sum : tensor<1x64x8x8xf32>) outs(%f_empty : tensor<1x64x8x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %p = arith.cmpf ugt, %in, %zero : f32
    %r = arith.select %p, %in, %zero : f32
    linalg.yield %r : f32
  } -> tensor<1x64x8x8xf32>
  %q_empty = tensor.empty() : tensor<1x64x8x8xi8>
  %q = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%relu : tensor<1x64x8x8xf32>) outs(%q_empty : tensor<1x64x8x8xi8>) {
  ^bb0(%in: f32, %out: i8):
    %d = arith.divf %in, %out_scale : f32
    %r = math.roundeven %d : f32
    %z = arith.addf %r, %zero : f32
    %lo = arith.maximumf %z, %zero : f32
    %hi = arith.minimumf %lo, %max : f32
    %n = arith.fptoui %hi : f32 to i8
    linalg.yield %n : i8
  } -> tensor<1x64x8x8xi8>
  util.return %q : tensor<1x64x8x8xi8>
}

// Without the ReLU there is no target (a linear bottleneck's add): left on
// the CPU, untouched. The output zero point is 128, not quant_min (0): a ui8
// QuantizeLinear with zero point 0 clamps every negative sum to the code of
// 0, which is a ReLU folded into it, and is claimed (see the QDQ cases
// below).
// CHECK-LABEL: util.func public @residual_add_no_relu_declines
// CHECK-NOT: rocket_elementwise_add_int8_relu
// CHECK: util.return
util.func public @residual_add_no_relu_declines(%a: tensor<1x64x8x8xi8>, %b: tensor<1x64x8x8xi8>) -> tensor<1x64x8x8xi8> {
  %a_scale = arith.constant 3.500000e-02 : f32
  %b_scale = arith.constant 4.200000e-02 : f32
  %out_scale = arith.constant 7.000000e-02 : f32
  %zero = arith.constant 0.000000e+00 : f32
  %max = arith.constant 2.550000e+02 : f32
  %out_zp = arith.constant 1.280000e+02 : f32
  %f_empty = tensor.empty() : tensor<1x64x8x8xf32>
  %fa = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%a : tensor<1x64x8x8xi8>) outs(%f_empty : tensor<1x64x8x8xf32>) {
  ^bb0(%in: i8, %out: f32):
    %f = arith.uitofp %in : i8 to f32
    %s = arith.mulf %f, %a_scale : f32
    linalg.yield %s : f32
  } -> tensor<1x64x8x8xf32>
  %fb = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%b : tensor<1x64x8x8xi8>) outs(%f_empty : tensor<1x64x8x8xf32>) {
  ^bb0(%in: i8, %out: f32):
    %f = arith.uitofp %in : i8 to f32
    %s = arith.mulf %f, %b_scale : f32
    linalg.yield %s : f32
  } -> tensor<1x64x8x8xf32>
  %sum = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%fa, %fb : tensor<1x64x8x8xf32>, tensor<1x64x8x8xf32>) outs(%f_empty : tensor<1x64x8x8xf32>) {
  ^bb0(%x: f32, %y: f32, %out: f32):
    %r = arith.addf %x, %y : f32
    linalg.yield %r : f32
  } -> tensor<1x64x8x8xf32>
  %q_empty = tensor.empty() : tensor<1x64x8x8xi8>
  %q = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%sum : tensor<1x64x8x8xf32>) outs(%q_empty : tensor<1x64x8x8xi8>) {
  ^bb0(%in: f32, %out: i8):
    %d = arith.divf %in, %out_scale : f32
    %r = math.roundeven %d : f32
    %z = arith.addf %r, %out_zp : f32
    %lo = arith.maximumf %z, %zero : f32
    %hi = arith.minimumf %lo, %max : f32
    %n = arith.fptoui %hi : f32 to i8
    linalg.yield %n : i8
  } -> tensor<1x64x8x8xi8>
  util.return %q : tensor<1x64x8x8xi8>
}

// The same tail as IREE 3.14's importer spells it: DequantizeLinear and
// QuantizeLinear as `iree_linalg_ext.{de,}quantize_affine` (an `i64` zero
// point for `a`, none -- symmetric -- for `b` and the output) and the add as
// `linalg.elementwise <add>`. Before the pass accepted this form, every int8
// residual add in a real model silently stayed on the CPU.
// CHECK-LABEL: util.func public @residual_add_relu_affine_qdq
// CHECK: flow.dispatch @rocket_elementwise_add_int8_relu_executable::
// CHECK-SAME: %c35_i32
// CHECK-SAME: %c-128_i32
// CHECK-NOT: iree_linalg_ext.quantize_affine
// CHECK: util.return
util.func public @residual_add_relu_affine_qdq(%a: tensor<1x64x8x8xi8>, %b: tensor<1x64x8x8xi8>) -> tensor<1x64x8x8xi8> {
  %a_scale = arith.constant 3.500000e-02 : f32
  %b_scale = arith.constant 4.200000e-02 : f32
  %out_scale = arith.constant 7.000000e-02 : f32
  %a_zp = arith.constant 163 : i64
  %zero = arith.constant 0.000000e+00 : f32
  %f_empty = tensor.empty() : tensor<1x64x8x8xf32>
  %fa = iree_linalg_ext.dequantize_affine <indexing_maps = [#map, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> ()>, #map], quant_min = 0, quant_max = 255, input_unsigned> ins(%a, %a_scale, %a_zp : tensor<1x64x8x8xi8>, f32, i64) outs(%f_empty : tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
  %fb = iree_linalg_ext.dequantize_affine <indexing_maps = [#map, affine_map<(d0, d1, d2, d3) -> ()>, #map], quant_min = 0, quant_max = 255, input_unsigned> ins(%b, %b_scale : tensor<1x64x8x8xi8>, f32) outs(%f_empty : tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
  %sum = linalg.elementwise <add> ins(%fa, %fb : tensor<1x64x8x8xf32>, tensor<1x64x8x8xf32>) outs(%f_empty : tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
  %relu = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%sum : tensor<1x64x8x8xf32>) outs(%f_empty : tensor<1x64x8x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %p = arith.cmpf ugt, %in, %zero : f32
    %r = arith.select %p, %in, %zero : f32
    linalg.yield %r : f32
  } -> tensor<1x64x8x8xf32>
  %q_empty = tensor.empty() : tensor<1x64x8x8xi8>
  %q = iree_linalg_ext.quantize_affine <indexing_maps = [#map, affine_map<(d0, d1, d2, d3) -> ()>, #map], quant_min = 0, quant_max = 255, storage_unsigned> ins(%relu, %out_scale : tensor<1x64x8x8xf32>, f32) outs(%q_empty : tensor<1x64x8x8xi8>) -> tensor<1x64x8x8xi8>
  util.return %q : tensor<1x64x8x8xi8>
}

// The tail as ONNX Runtime's default `quantize_static` (QDQ, S8S8) writes it,
// taken from a stock QDQ ResNet50: signed `i8` codes in [-128, 127], and no
// ReLU -- ORT deleted it and gave the QuantizeLinear `zero_point ==
// quant_min`, so the clamp already sends every negative sum to the code of 0.
// The signed codes need no `addi -128`, and their zero points (-29, 5, -128)
// reach the dispatch as they are.
// CHECK-LABEL: util.func public @residual_add_folded_relu_qdq_s8
// CHECK-NOT: arith.addi
// CHECK: flow.dispatch @rocket_elementwise_add_int8_relu_executable::
// CHECK-SAME: %c-29_i32
// CHECK-SAME: %c5_i32
// CHECK-SAME: %c-128_i32
// CHECK-NOT: iree_linalg_ext.quantize_affine
// CHECK: util.return
util.func public @residual_add_folded_relu_qdq_s8(%a: tensor<1x64x8x8xi8>, %b: tensor<1x64x8x8xi8>) -> tensor<1x64x8x8xi8> {
  %a_scale = arith.constant 3.500000e-02 : f32
  %b_scale = arith.constant 4.200000e-02 : f32
  %out_scale = arith.constant 7.000000e-02 : f32
  %a_zp = arith.constant -29 : i64
  %b_zp = arith.constant 5 : i64
  %out_zp = arith.constant -128 : i64
  %f_empty = tensor.empty() : tensor<1x64x8x8xf32>
  %fa = iree_linalg_ext.dequantize_affine <indexing_maps = [#map, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> ()>, #map], quant_min = -128, quant_max = 127> ins(%a, %a_scale, %a_zp : tensor<1x64x8x8xi8>, f32, i64) outs(%f_empty : tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
  %fb = iree_linalg_ext.dequantize_affine <indexing_maps = [#map, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> ()>, #map], quant_min = -128, quant_max = 127> ins(%b, %b_scale, %b_zp : tensor<1x64x8x8xi8>, f32, i64) outs(%f_empty : tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
  %sum = linalg.elementwise <add> ins(%fa, %fb : tensor<1x64x8x8xf32>, tensor<1x64x8x8xf32>) outs(%f_empty : tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
  %q_empty = tensor.empty() : tensor<1x64x8x8xi8>
  %q = iree_linalg_ext.quantize_affine <indexing_maps = [#map, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> ()>, #map], quant_min = -128, quant_max = 127> ins(%sum, %out_scale, %out_zp : tensor<1x64x8x8xf32>, f32, i64) outs(%q_empty : tensor<1x64x8x8xi8>) -> tensor<1x64x8x8xi8>
  util.return %q : tensor<1x64x8x8xi8>
}

// Signed codes with the ReLU still explicit: the output zero point (-7) is
// not quant_min, and that is fine because the ReLU says so itself.
// CHECK-LABEL: util.func public @residual_add_relu_qdq_s8
// CHECK: flow.dispatch @rocket_elementwise_add_int8_relu_executable::
// CHECK-SAME: %c-7_i32
// CHECK-NOT: iree_linalg_ext.quantize_affine
// CHECK: util.return
util.func public @residual_add_relu_qdq_s8(%a: tensor<1x64x8x8xi8>, %b: tensor<1x64x8x8xi8>) -> tensor<1x64x8x8xi8> {
  %a_scale = arith.constant 3.500000e-02 : f32
  %b_scale = arith.constant 4.200000e-02 : f32
  %out_scale = arith.constant 7.000000e-02 : f32
  %a_zp = arith.constant -29 : i64
  %out_zp = arith.constant -7 : i64
  %zero = arith.constant 0.000000e+00 : f32
  %f_empty = tensor.empty() : tensor<1x64x8x8xf32>
  %fa = iree_linalg_ext.dequantize_affine <indexing_maps = [#map, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> ()>, #map], quant_min = -128, quant_max = 127> ins(%a, %a_scale, %a_zp : tensor<1x64x8x8xi8>, f32, i64) outs(%f_empty : tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
  %fb = iree_linalg_ext.dequantize_affine <indexing_maps = [#map, affine_map<(d0, d1, d2, d3) -> ()>, #map], quant_min = -128, quant_max = 127> ins(%b, %b_scale : tensor<1x64x8x8xi8>, f32) outs(%f_empty : tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
  %sum = linalg.elementwise <add> ins(%fa, %fb : tensor<1x64x8x8xf32>, tensor<1x64x8x8xf32>) outs(%f_empty : tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
  %relu = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%sum : tensor<1x64x8x8xf32>) outs(%f_empty : tensor<1x64x8x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %p = arith.cmpf ugt, %in, %zero : f32
    %r = arith.select %p, %in, %zero : f32
    linalg.yield %r : f32
  } -> tensor<1x64x8x8xf32>
  %q_empty = tensor.empty() : tensor<1x64x8x8xi8>
  %q = iree_linalg_ext.quantize_affine <indexing_maps = [#map, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> ()>, #map], quant_min = -128, quant_max = 127> ins(%relu, %out_scale, %out_zp : tensor<1x64x8x8xf32>, f32, i64) outs(%q_empty : tensor<1x64x8x8xi8>) -> tensor<1x64x8x8xi8>
  util.return %q : tensor<1x64x8x8xi8>
}

// A linear residual add (MobileNetV2's): no ReLU and an output zero point
// above quant_min, so negative sums must survive. There is no target for it;
// it stays a CPU DQ -> add -> Q.
// CHECK-LABEL: util.func public @residual_add_linear_qdq_s8_declined
// CHECK-NOT: rocket_elementwise_add_int8_relu_executable
// CHECK: iree_linalg_ext.quantize_affine
// CHECK: util.return
util.func public @residual_add_linear_qdq_s8_declined(%a: tensor<1x64x8x8xi8>, %b: tensor<1x64x8x8xi8>) -> tensor<1x64x8x8xi8> {
  %a_scale = arith.constant 3.500000e-02 : f32
  %b_scale = arith.constant 4.200000e-02 : f32
  %out_scale = arith.constant 7.000000e-02 : f32
  %a_zp = arith.constant -29 : i64
  %out_zp = arith.constant -7 : i64
  %f_empty = tensor.empty() : tensor<1x64x8x8xf32>
  %fa = iree_linalg_ext.dequantize_affine <indexing_maps = [#map, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> ()>, #map], quant_min = -128, quant_max = 127> ins(%a, %a_scale, %a_zp : tensor<1x64x8x8xi8>, f32, i64) outs(%f_empty : tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
  %fb = iree_linalg_ext.dequantize_affine <indexing_maps = [#map, affine_map<(d0, d1, d2, d3) -> ()>, #map], quant_min = -128, quant_max = 127> ins(%b, %b_scale : tensor<1x64x8x8xi8>, f32) outs(%f_empty : tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
  %sum = linalg.elementwise <add> ins(%fa, %fb : tensor<1x64x8x8xf32>, tensor<1x64x8x8xf32>) outs(%f_empty : tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
  %q_empty = tensor.empty() : tensor<1x64x8x8xi8>
  %q = iree_linalg_ext.quantize_affine <indexing_maps = [#map, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> ()>, #map], quant_min = -128, quant_max = 127> ins(%sum, %out_scale, %out_zp : tensor<1x64x8x8xf32>, f32, i64) outs(%q_empty : tensor<1x64x8x8xi8>) -> tensor<1x64x8x8xi8>
  util.return %q : tensor<1x64x8x8xi8>
}

// A symmetric [-127, 127] clamp is a different quantization: -128 is not a
// code it can produce, so the canonical full-range form would be wrong for
// it. Declined even with zero point == quant_min.
// CHECK-LABEL: util.func public @residual_add_narrow_range_declined
// CHECK-NOT: rocket_elementwise_add_int8_relu_executable
// CHECK: iree_linalg_ext.quantize_affine
// CHECK: util.return
util.func public @residual_add_narrow_range_declined(%a: tensor<1x64x8x8xi8>, %b: tensor<1x64x8x8xi8>) -> tensor<1x64x8x8xi8> {
  %a_scale = arith.constant 3.500000e-02 : f32
  %b_scale = arith.constant 4.200000e-02 : f32
  %out_scale = arith.constant 7.000000e-02 : f32
  %out_zp = arith.constant -127 : i64
  %f_empty = tensor.empty() : tensor<1x64x8x8xf32>
  %fa = iree_linalg_ext.dequantize_affine <indexing_maps = [#map, affine_map<(d0, d1, d2, d3) -> ()>, #map], quant_min = -127, quant_max = 127> ins(%a, %a_scale : tensor<1x64x8x8xi8>, f32) outs(%f_empty : tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
  %fb = iree_linalg_ext.dequantize_affine <indexing_maps = [#map, affine_map<(d0, d1, d2, d3) -> ()>, #map], quant_min = -127, quant_max = 127> ins(%b, %b_scale : tensor<1x64x8x8xi8>, f32) outs(%f_empty : tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
  %sum = linalg.elementwise <add> ins(%fa, %fb : tensor<1x64x8x8xf32>, tensor<1x64x8x8xf32>) outs(%f_empty : tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
  %q_empty = tensor.empty() : tensor<1x64x8x8xi8>
  %q = iree_linalg_ext.quantize_affine <indexing_maps = [#map, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> ()>, #map], quant_min = -127, quant_max = 127> ins(%sum, %out_scale, %out_zp : tensor<1x64x8x8xf32>, f32, i64) outs(%q_empty : tensor<1x64x8x8xi8>) -> tensor<1x64x8x8xi8>
  util.return %q : tensor<1x64x8x8xi8>
}
