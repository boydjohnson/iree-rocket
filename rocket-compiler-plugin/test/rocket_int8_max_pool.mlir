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
// NPU's int8 pooling runs. Its -128 pad is marked by rocket-fold-conv-pad and
// claimed with the pool, so the PPU pads: the dispatch reads the unpadded
// 16x16 tensor.

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: util.func public @stem_max_pool
// CHECK-NOT: tensor.pad
// CHECK: flow.dispatch @rocket_pooling_max_int8_pad1_executable_s2::{{.*}}(%c16_i32, %c16_i32, %c64_i32, %c3_i32, %c3_i32
// CHECK-NOT: linalg.pooling_nchw_max
// CHECK-NOT: tensor.pad
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

// Padding 2 on a 3x3 window is below the kernel, so rocket-fold-conv-pad
// marks it, but no target bakes pad 2: the pad1 matcher declines on the DAG
// and the unpadded int8 matcher claims the pool alone, with the pad staying a
// CPU tensor.pad. Correct either way -- the unpadded executable pads nothing.
// CHECK-LABEL: util.func public @pad2_stays_on_the_cpu
// CHECK: tensor.pad {{.*}} low[0, 2, 2, 0] high[0, 2, 2, 0]
// CHECK: flow.dispatch @rocket_pooling_max_int8_executable_s2::
// CHECK: util.return
util.func public @pad2_stays_on_the_cpu(%x: tensor<1x64x16x16xi8>) -> tensor<1x64x9x9xi8> {
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
  %padded = tensor.pad %f low[0, 0, 2, 2] high[0, 0, 2, 2] {
  ^bb0(%a: index, %b: index, %c: index, %d: index):
    tensor.yield %neg_inf : f32
  } : tensor<1x64x16x16xf32> to tensor<1x64x20x20xf32>
  %p_empty = tensor.empty() : tensor<1x64x9x9xf32>
  %p_init = linalg.fill ins(%neg_inf : f32) outs(%p_empty : tensor<1x64x9x9xf32>) -> tensor<1x64x9x9xf32>
  %window = tensor.empty() : tensor<3x3xf32>
  %pooled = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%padded, %window : tensor<1x64x20x20xf32>, tensor<3x3xf32>) outs(%p_init : tensor<1x64x9x9xf32>) -> tensor<1x64x9x9xf32>
  %q_empty = tensor.empty() : tensor<1x64x9x9xi8>
  %q = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%pooled : tensor<1x64x9x9xf32>) outs(%q_empty : tensor<1x64x9x9xi8>) {
  ^bb0(%in: f32, %out: i8):
    %d = arith.divf %in, %scale : f32
    %r = math.roundeven %d : f32
    %z = arith.addf %r, %zero : f32
    %lo = arith.maximumf %z, %zero : f32
    %hi = arith.minimumf %lo, %max : f32
    %n = arith.fptoui %hi : f32 to i8
    linalg.yield %n : i8
  } -> tensor<1x64x9x9xi8>
  util.return %q : tensor<1x64x9x9xi8>
}

// @stem_max_pool as IREE 3.14's importer spells its DequantizeLinear and
// QuantizeLinear: `iree_linalg_ext.{de,}quantize_affine`, here with a shared
// `i64` zero point.
// CHECK-LABEL: util.func public @stem_max_pool_affine_qdq
// CHECK-NOT: tensor.pad
// CHECK: flow.dispatch @rocket_pooling_max_int8_pad1_executable_s2::{{.*}}(%c16_i32, %c16_i32, %c64_i32, %c3_i32, %c3_i32
// CHECK-NOT: linalg.pooling_nchw_max
// CHECK-NOT: iree_linalg_ext.quantize_affine
// CHECK: util.return
util.func public @stem_max_pool_affine_qdq(%x: tensor<1x64x16x16xi8>) -> tensor<1x64x8x8xi8> {
  %scale = arith.constant 1.370000e-01 : f32
  %zp = arith.constant 7 : i64
  %neg_inf = arith.constant 0xFF800000 : f32
  %f_empty = tensor.empty() : tensor<1x64x16x16xf32>
  %f = iree_linalg_ext.dequantize_affine <indexing_maps = [#map, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> ()>, #map], quant_min = 0, quant_max = 255, input_unsigned> ins(%x, %scale, %zp : tensor<1x64x16x16xi8>, f32, i64) outs(%f_empty : tensor<1x64x16x16xf32>) -> tensor<1x64x16x16xf32>
  %padded = tensor.pad %f low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a: index, %b: index, %c: index, %d: index):
    tensor.yield %neg_inf : f32
  } : tensor<1x64x16x16xf32> to tensor<1x64x18x18xf32>
  %p_empty = tensor.empty() : tensor<1x64x8x8xf32>
  %p_init = linalg.fill ins(%neg_inf : f32) outs(%p_empty : tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
  %window = tensor.empty() : tensor<3x3xf32>
  %pooled = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%padded, %window : tensor<1x64x18x18xf32>, tensor<3x3xf32>) outs(%p_init : tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
  %q_empty = tensor.empty() : tensor<1x64x8x8xi8>
  %q = iree_linalg_ext.quantize_affine <indexing_maps = [#map, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> ()>, #map], quant_min = 0, quant_max = 255, storage_unsigned> ins(%pooled, %scale, %zp : tensor<1x64x8x8xf32>, f32, i64) outs(%q_empty : tensor<1x64x8x8xi8>) -> tensor<1x64x8x8xi8>
  util.return %q : tensor<1x64x8x8xi8>
}

// ResNet50's stem pool as ONNX Runtime's default `quantize_static` (QDQ,
// S8S8) writes it: signed `i8` codes, the conv's folded-ReLU zero point -128
// shared by both sides. Signed codes are the max pool's own encoding, so no
// shift appears.
// CHECK-LABEL: util.func public @stem_max_pool_qdq_s8
// CHECK-NOT: arith.addi
// CHECK-NOT: tensor.pad
// CHECK: flow.dispatch @rocket_pooling_max_int8_pad1_executable_s2::{{.*}}(%c16_i32, %c16_i32, %c64_i32, %c3_i32, %c3_i32
// CHECK-NOT: linalg.pooling_nchw_max
// CHECK-NOT: iree_linalg_ext.quantize_affine
// CHECK: util.return
util.func public @stem_max_pool_qdq_s8(%x: tensor<1x64x16x16xi8>) -> tensor<1x64x8x8xi8> {
  %scale = arith.constant 2.407370e-01 : f32
  %zp = arith.constant -128 : i64
  %neg_inf = arith.constant 0xFF800000 : f32
  %f_empty = tensor.empty() : tensor<1x64x16x16xf32>
  %f = iree_linalg_ext.dequantize_affine <indexing_maps = [#map, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> ()>, #map], quant_min = -128, quant_max = 127> ins(%x, %scale, %zp : tensor<1x64x16x16xi8>, f32, i64) outs(%f_empty : tensor<1x64x16x16xf32>) -> tensor<1x64x16x16xf32>
  %padded = tensor.pad %f low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a: index, %b: index, %c: index, %d: index):
    tensor.yield %neg_inf : f32
  } : tensor<1x64x16x16xf32> to tensor<1x64x18x18xf32>
  %p_empty = tensor.empty() : tensor<1x64x8x8xf32>
  %p_init = linalg.fill ins(%neg_inf : f32) outs(%p_empty : tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
  %window = tensor.empty() : tensor<3x3xf32>
  %pooled = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%padded, %window : tensor<1x64x18x18xf32>, tensor<3x3xf32>) outs(%p_init : tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
  %q_empty = tensor.empty() : tensor<1x64x8x8xi8>
  %q = iree_linalg_ext.quantize_affine <indexing_maps = [#map, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> ()>, #map], quant_min = -128, quant_max = 127> ins(%pooled, %scale, %zp : tensor<1x64x8x8xf32>, f32, i64) outs(%q_empty : tensor<1x64x8x8xi8>) -> tensor<1x64x8x8xi8>
  util.return %q : tensor<1x64x8x8xi8>
}

// A ui8 DequantizeLinear feeding a signed QuantizeLinear with the same scale
// and zero point is not a round trip -- code 200 would come back clamped --
// so it is left alone.
// (The float pool then takes the fp16 path, between the CPU's Q and DQ.)
// CHECK-LABEL: util.func public @max_pool_mixed_signedness_declined
// CHECK-NOT: rocket_pooling_max_int8
// CHECK: iree_linalg_ext.quantize_affine
// CHECK: util.return
util.func public @max_pool_mixed_signedness_declined(%x: tensor<1x64x16x16xi8>) -> tensor<1x64x8x8xi8> {
  %scale = arith.constant 2.407370e-01 : f32
  %zp = arith.constant 0 : i64
  %neg_inf = arith.constant 0xFF800000 : f32
  %f_empty = tensor.empty() : tensor<1x64x16x16xf32>
  %f = iree_linalg_ext.dequantize_affine <indexing_maps = [#map, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> ()>, #map], quant_min = 0, quant_max = 255, input_unsigned> ins(%x, %scale, %zp : tensor<1x64x16x16xi8>, f32, i64) outs(%f_empty : tensor<1x64x16x16xf32>) -> tensor<1x64x16x16xf32>
  %padded = tensor.pad %f low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%a: index, %b: index, %c: index, %d: index):
    tensor.yield %neg_inf : f32
  } : tensor<1x64x16x16xf32> to tensor<1x64x18x18xf32>
  %p_empty = tensor.empty() : tensor<1x64x8x8xf32>
  %p_init = linalg.fill ins(%neg_inf : f32) outs(%p_empty : tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
  %window = tensor.empty() : tensor<3x3xf32>
  %pooled = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%padded, %window : tensor<1x64x18x18xf32>, tensor<3x3xf32>) outs(%p_init : tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
  %q_empty = tensor.empty() : tensor<1x64x8x8xi8>
  %q = iree_linalg_ext.quantize_affine <indexing_maps = [#map, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> ()>, #map], quant_min = -128, quant_max = 127> ins(%pooled, %scale, %zp : tensor<1x64x8x8xf32>, f32, i64) outs(%q_empty : tensor<1x64x8x8xi8>) -> tensor<1x64x8x8xi8>
  util.return %q : tensor<1x64x8x8xi8>
}
