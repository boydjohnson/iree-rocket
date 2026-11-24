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

// A quantized ResNet stem's max pool as the ONNX importer emits it: QOperator
// leaves MaxPool in float, so a `ui8` dequantization, a -inf pad, the pool
// and a QuantizeLinear. tools/quantize_onnx_int8.py gives the QuantizeLinear
// the DequantizeLinear's own parameters, and then rocket-fuse-int8-max-pool
// turns the four into an int8 max pool on the codes -- exactly -- which the
// NPU's int8 pooling runs.

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: util.func public @stem_max_pool
// The pad survives on the CPU at int8, filled with the smallest code.
// CHECK: tensor.pad {{.*}} low[0, 1, 1, 0] high[0, 1, 1, 0]
// CHECK: tensor.yield %c-128_i8 : i8
// CHECK: flow.dispatch @rocket_pooling_max_int8_executable_s2::
// CHECK-NOT: linalg.pooling_nchw_max
// CHECK: util.return
util.func public @stem_max_pool(%x: tensor<1x64x16x16xi8>) -> tensor<1x64x8x8xi8> {
  %scale = arith.constant 1.370000e-01 : f32
  %zero = arith.constant 0.000000e+00 : f32
  %max = arith.constant 2.550000e+02 : f32
  %neg_inf = arith.constant 0xFF800000 : f32
  %f_empty = tensor.empty() : tensor<1x64x16x16xf32>
  %f = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%x : tensor<1x64x16x16xi8>) outs(%f_empty : tensor<1x64x16x16xf32>) {
  ^bb0(%in: i8, %out: f32):
    %v = arith.uitofp %in : i8 to f32
    %s = arith.mulf %v, %scale : f32
    linalg.yield %s : f32
  } -> tensor<1x64x16x16xf32>
  %padded = tensor.pad %f low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a: index, %b: index, %c: index, %d: index):
    tensor.yield %neg_inf : f32
  } : tensor<1x64x16x16xf32> to tensor<1x64x18x18xf32>
  %p_empty = tensor.empty() : tensor<1x64x8x8xf32>
  %p_init = linalg.fill ins(%neg_inf : f32) outs(%p_empty : tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
  %window = tensor.empty() : tensor<3x3xf32>
  %pooled = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%padded, %window : tensor<1x64x18x18xf32>, tensor<3x3xf32>) outs(%p_init : tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
  %q_empty = tensor.empty() : tensor<1x64x8x8xi8>
  %q = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%pooled : tensor<1x64x8x8xf32>) outs(%q_empty : tensor<1x64x8x8xi8>) {
  ^bb0(%in: f32, %out: i8):
    %d = arith.divf %in, %scale : f32
    %r = math.roundeven %d : f32
    %z = arith.addf %r, %zero : f32
    %lo = arith.maximumf %z, %zero : f32
    %hi = arith.minimumf %lo, %max : f32
    %n = arith.fptoui %hi : f32 to i8
    linalg.yield %n : i8
  } -> tensor<1x64x8x8xi8>
  util.return %q : tensor<1x64x8x8xi8>
}

// A QuantizeLinear with its own scale is a max pool *and* a requantization:
// not rewritten, so it keeps the fp16 path it had.
// CHECK-LABEL: util.func public @requantizing_max_pool_declines
// CHECK-NOT: rocket_pooling_max_int8
// CHECK: util.return
util.func public @requantizing_max_pool_declines(%x: tensor<1x64x16x16xi8>) -> tensor<1x64x8x8xi8> {
  %scale = arith.constant 1.370000e-01 : f32
  %out_scale = arith.constant 1.500000e-01 : f32
  %zero = arith.constant 0.000000e+00 : f32
  %max = arith.constant 2.550000e+02 : f32
  %neg_inf = arith.constant 0xFF800000 : f32
  %f_empty = tensor.empty() : tensor<1x64x16x16xf32>
  %f = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%x : tensor<1x64x16x16xi8>) outs(%f_empty : tensor<1x64x16x16xf32>) {
  ^bb0(%in: i8, %out: f32):
    %v = arith.uitofp %in : i8 to f32
    %s = arith.mulf %v, %scale : f32
    linalg.yield %s : f32
  } -> tensor<1x64x16x16xf32>
  %padded = tensor.pad %f low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a: index, %b: index, %c: index, %d: index):
    tensor.yield %neg_inf : f32
  } : tensor<1x64x16x16xf32> to tensor<1x64x18x18xf32>
  %p_empty = tensor.empty() : tensor<1x64x8x8xf32>
  %p_init = linalg.fill ins(%neg_inf : f32) outs(%p_empty : tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
  %window = tensor.empty() : tensor<3x3xf32>
  %pooled = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%padded, %window : tensor<1x64x18x18xf32>, tensor<3x3xf32>) outs(%p_init : tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
  %q_empty = tensor.empty() : tensor<1x64x8x8xi8>
  %q = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%pooled : tensor<1x64x8x8xf32>) outs(%q_empty : tensor<1x64x8x8xi8>) {
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
