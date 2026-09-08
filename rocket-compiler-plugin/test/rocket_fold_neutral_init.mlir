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
