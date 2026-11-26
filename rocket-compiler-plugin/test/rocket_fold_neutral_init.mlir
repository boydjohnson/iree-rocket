// RUN: iree-opt %s --pass-pipeline='builtin.module(rocket-fold-neutral-init)' \
// RUN:   | FileCheck %s

// A shim's widen `OP(extf(raw), init)` loses its init term when the init is
// a fill of OP's neutral element, and keeps it otherwise.

#id2 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: util.func public @add_zero
util.func public @add_zero(%raw: tensor<4x8xf16>) -> tensor<4x8xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<4x8xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<4x8xf32>) -> tensor<4x8xf32>
  %out_empty = tensor.empty() : tensor<4x8xf32>
  // CHECK: linalg.generic
  // CHECK-SAME: ins(%{{.+}} : tensor<4x8xf16>)
  // CHECK: arith.extf
  // CHECK-NOT: arith.addf
  // CHECK: linalg.yield
  %wide = linalg.generic {indexing_maps = [#id2, #id2, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%raw, %init : tensor<4x8xf16>, tensor<4x8xf32>) outs(%out_empty : tensor<4x8xf32>) {
  ^bb0(%r: f16, %i: f32, %o: f32):
    %w = arith.extf %r : f16 to f32
    %s = arith.addf %w, %i : f32
    linalg.yield %s : f32
  } -> tensor<4x8xf32>
  util.return %wide : tensor<4x8xf32>
}

// CHECK-LABEL: util.func public @max_neg_inf_through_transpose
util.func public @max_neg_inf_through_transpose(%raw: tensor<4x8xf16>) -> tensor<4x8xf32> {
  %lowest = arith.constant 0xFF800000 : f32
  %empty = tensor.empty() : tensor<8x4xf32>
  %init_t = linalg.fill ins(%lowest : f32) outs(%empty : tensor<8x4xf32>) -> tensor<8x4xf32>
  %t_empty = tensor.empty() : tensor<4x8xf32>
  %init = linalg.transpose ins(%init_t : tensor<8x4xf32>) outs(%t_empty : tensor<4x8xf32>) permutation = [1, 0]
  %out_empty = tensor.empty() : tensor<4x8xf32>
  // CHECK: linalg.generic
  // CHECK-SAME: ins(%{{.+}} : tensor<4x8xf16>)
  // CHECK-NOT: arith.maximumf
  // CHECK: linalg.yield
  %wide = linalg.generic {indexing_maps = [#id2, #id2, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%raw, %init : tensor<4x8xf16>, tensor<4x8xf32>) outs(%out_empty : tensor<4x8xf32>) {
  ^bb0(%r: f16, %i: f32, %o: f32):
    %w = arith.extf %r : f16 to f32
    %m = arith.maximumf %w, %i : f32
    linalg.yield %m : f32
  } -> tensor<4x8xf32>
  util.return %wide : tensor<4x8xf32>
}

// CHECK-LABEL: util.func public @add_one_is_kept
util.func public @add_one_is_kept(%raw: tensor<4x8xf16>) -> tensor<4x8xf32> {
  %one = arith.constant 1.000000e+00 : f32
  %empty = tensor.empty() : tensor<4x8xf32>
  %init = linalg.fill ins(%one : f32) outs(%empty : tensor<4x8xf32>) -> tensor<4x8xf32>
  %out_empty = tensor.empty() : tensor<4x8xf32>
  // CHECK: linalg.generic
  // CHECK-SAME: ins(%{{.+}}, %{{.+}} : tensor<4x8xf16>, tensor<4x8xf32>)
  // CHECK: arith.addf
  %wide = linalg.generic {indexing_maps = [#id2, #id2, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%raw, %init : tensor<4x8xf16>, tensor<4x8xf32>) outs(%out_empty : tensor<4x8xf32>) {
  ^bb0(%r: f16, %i: f32, %o: f32):
    %w = arith.extf %r : f16 to f32
    %s = arith.addf %w, %i : f32
    linalg.yield %s : f32
  } -> tensor<4x8xf32>
  util.return %wide : tensor<4x8xf32>
}

// CHECK-LABEL: util.func public @max_zero_is_kept
util.func public @max_zero_is_kept(%raw: tensor<4x8xf16>) -> tensor<4x8xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<4x8xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<4x8xf32>) -> tensor<4x8xf32>
  %out_empty = tensor.empty() : tensor<4x8xf32>
  // CHECK: linalg.generic
  // CHECK: arith.maximumf
  %wide = linalg.generic {indexing_maps = [#id2, #id2, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%raw, %init : tensor<4x8xf16>, tensor<4x8xf32>) outs(%out_empty : tensor<4x8xf32>) {
  ^bb0(%r: f16, %i: f32, %o: f32):
    %w = arith.extf %r : f16 to f32
    %m = arith.maximumf %w, %i : f32
    linalg.yield %m : f32
  } -> tensor<4x8xf32>
  util.return %wide : tensor<4x8xf32>
}

// IREE 3.14's arith no longer folds truncf(extf(x)), so the pass cancels a
// bare widen against the consumer's narrow itself -- directly, and through
// the one reshape a shim's caller puts between them.

#id3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL: util.func public @cancel_widen_narrow
util.func public @cancel_widen_narrow(%raw: tensor<4x8xf16>) -> tensor<4x8xf16> {
  // CHECK-NOT: arith.extf
  // CHECK-NOT: arith.truncf
  // CHECK: util.return %arg0
  %e32 = tensor.empty() : tensor<4x8xf32>
  %wide = linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%raw : tensor<4x8xf16>) outs(%e32 : tensor<4x8xf32>) {
  ^bb0(%r: f16, %o: f32):
    %w = arith.extf %r : f16 to f32
    linalg.yield %w : f32
  } -> tensor<4x8xf32>
  %e16 = tensor.empty() : tensor<4x8xf16>
  %narrow = linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%wide : tensor<4x8xf32>) outs(%e16 : tensor<4x8xf16>) {
  ^bb0(%w: f32, %o: f16):
    %n = arith.truncf %w : f32 to f16
    linalg.yield %n : f16
  } -> tensor<4x8xf16>
  util.return %narrow : tensor<4x8xf16>
}

// CHECK-LABEL: util.func public @cancel_through_expand
util.func public @cancel_through_expand(%raw: tensor<4x8xf16>) -> tensor<1x4x8xf16> {
  // CHECK-NOT: arith.truncf
  // CHECK: %[[E:.+]] = tensor.expand_shape %arg0 {{\[\[}}0, 1], [2]] output_shape [1, 4, 8] : tensor<4x8xf16> into tensor<1x4x8xf16>
  // CHECK: util.return %[[E]]
  %e32 = tensor.empty() : tensor<4x8xf32>
  %wide = linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%raw : tensor<4x8xf16>) outs(%e32 : tensor<4x8xf32>) {
  ^bb0(%r: f16, %o: f32):
    %w = arith.extf %r : f16 to f32
    linalg.yield %w : f32
  } -> tensor<4x8xf32>
  %x = tensor.expand_shape %wide [[0, 1], [2]] output_shape [1, 4, 8] : tensor<4x8xf32> into tensor<1x4x8xf32>
  %e16 = tensor.empty() : tensor<1x4x8xf16>
  %narrow = linalg.generic {indexing_maps = [#id3, #id3], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%x : tensor<1x4x8xf32>) outs(%e16 : tensor<1x4x8xf16>) {
  ^bb0(%w: f32, %o: f16):
    %n = arith.truncf %w : f32 to f16
    linalg.yield %n : f16
  } -> tensor<1x4x8xf16>
  util.return %narrow : tensor<1x4x8xf16>
}

// A narrow to a different type than the widen's source is a real conversion
// and stays.
// CHECK-LABEL: util.func public @keep_bf16_narrow
util.func public @keep_bf16_narrow(%raw: tensor<4x8xf16>) -> tensor<4x8xbf16> {
  // CHECK: arith.extf
  // CHECK: arith.truncf
  %e32 = tensor.empty() : tensor<4x8xf32>
  %wide = linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%raw : tensor<4x8xf16>) outs(%e32 : tensor<4x8xf32>) {
  ^bb0(%r: f16, %o: f32):
    %w = arith.extf %r : f16 to f32
    linalg.yield %w : f32
  } -> tensor<4x8xf32>
  %e16 = tensor.empty() : tensor<4x8xbf16>
  %narrow = linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%wide : tensor<4x8xf32>) outs(%e16 : tensor<4x8xbf16>) {
  ^bb0(%w: f32, %o: bf16):
    %n = arith.truncf %w : f32 to bf16
    linalg.yield %n : bf16
  } -> tensor<4x8xbf16>
  util.return %narrow : tensor<4x8xbf16>
}
