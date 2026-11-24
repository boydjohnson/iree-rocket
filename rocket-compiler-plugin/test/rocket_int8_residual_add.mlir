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
// the CPU, untouched.
// CHECK-LABEL: util.func public @residual_add_no_relu_declines
// CHECK-NOT: rocket_elementwise_add_int8_relu
// CHECK: util.return
util.func public @residual_add_no_relu_declines(%a: tensor<1x64x8x8xi8>, %b: tensor<1x64x8x8xi8>) -> tensor<1x64x8x8xi8> {
  %a_scale = arith.constant 3.500000e-02 : f32
  %b_scale = arith.constant 4.200000e-02 : f32
  %out_scale = arith.constant 7.000000e-02 : f32
  %zero = arith.constant 0.000000e+00 : f32
  %max = arith.constant 2.550000e+02 : f32
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
    %z = arith.addf %r, %zero : f32
    %lo = arith.maximumf %z, %zero : f32
    %hi = arith.minimumf %lo, %max : f32
    %n = arith.fptoui %hi : f32 to i8
    linalg.yield %n : i8
  } -> tensor<1x64x8x8xi8>
  util.return %q : tensor<1x64x8x8xi8>
}
