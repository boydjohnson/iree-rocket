// RUN: iree-opt %s --pass-pipeline='builtin.module(util.func(rocket-unbatch-matmul))' \
// RUN:   | FileCheck %s

// ISSUES.md C15. A static-batch `linalg.batch_matmul` becomes one
// `linalg.matmul` per batch element, so the existing matmul matcher and
// lowering claim them; the batched form reaches neither, because
// `readRocketCandidate` reads row-major `linalg.matmul` only.
//
// The pass is off by default -- the transform spec carries its line behind
// `//@ROCKET_BATCH_MATMUL@` and `rocket-compiler --batch-matmul` splices it
// in -- so these cases gate the rewrite, not the placement.

// ViT-B/16 attention, `Q K^T`: twelve heads at sequence 197, head dim 64.
// Twelve matmuls, three slices each in and one out.
// CHECK-LABEL: util.func public @vit_attention_qk
// CHECK-COUNT-12: linalg.matmul
// CHECK-NOT: linalg.batch_matmul
// CHECK: util.return
util.func public @vit_attention_qk(
    %q: tensor<12x197x64xf32>,
    %k: tensor<12x64x197xf32>,
    %init: tensor<12x197x197xf32>) -> tensor<12x197x197xf32> {
  %0 = linalg.batch_matmul
      ins(%q, %k : tensor<12x197x64xf32>, tensor<12x64x197xf32>)
      outs(%init : tensor<12x197x197xf32>) -> tensor<12x197x197xf32>
  util.return %0 : tensor<12x197x197xf32>
}

// The slices are rank-reducing: a [1, M, K] read comes back as [M, K], which
// is what `linalg.matmul` takes. A non-reducing slice would leave a rank-3
// operand and the matmul would not verify.
// CHECK-LABEL: util.func public @slices_are_rank_reducing
// CHECK: tensor.extract_slice %{{.*}}[0, 0, 0] [1, 8, 4] [1, 1, 1] : tensor<2x8x4xf32> to tensor<8x4xf32>
// CHECK: tensor.insert_slice %{{.*}} into %{{.*}}[0, 0, 0] [1, 8, 8] [1, 1, 1] : tensor<8x8xf32> into tensor<2x8x8xf32>
util.func public @slices_are_rank_reducing(
    %a: tensor<2x8x4xf32>,
    %b: tensor<2x4x8xf32>,
    %init: tensor<2x8x8xf32>) -> tensor<2x8x8xf32> {
  %0 = linalg.batch_matmul
      ins(%a, %b : tensor<2x8x4xf32>, tensor<2x4x8xf32>)
      outs(%init : tensor<2x8x8xf32>) -> tensor<2x8x8xf32>
  util.return %0 : tensor<2x8x8xf32>
}

// A unit batch is the transform spec's own job -- it collapses one into a
// plain matmul with a reshape, which is strictly cheaper than a slice pair --
// so this pass leaves it alone rather than racing it.
// CHECK-LABEL: util.func public @unit_batch_is_left_to_the_spec
// CHECK: linalg.batch_matmul
util.func public @unit_batch_is_left_to_the_spec(
    %a: tensor<1x8x8xf32>,
    %b: tensor<1x8x8xf32>,
    %init: tensor<1x8x8xf32>) -> tensor<1x8x8xf32> {
  %0 = linalg.batch_matmul
      ins(%a, %b : tensor<1x8x8xf32>, tensor<1x8x8xf32>)
      outs(%init : tensor<1x8x8xf32>) -> tensor<1x8x8xf32>
  util.return %0 : tensor<1x8x8xf32>
}

// A dynamic batch cannot be unrolled at all -- the batch decides how many ops
// to emit -- and a dynamic M/K/N cannot be sliced at constant sizes.
// CHECK-LABEL: util.func public @dynamic_batch_is_untouched
// CHECK: linalg.batch_matmul
util.func public @dynamic_batch_is_untouched(
    %a: tensor<?x8x8xf32>,
    %b: tensor<?x8x8xf32>,
    %init: tensor<?x8x8xf32>) -> tensor<?x8x8xf32> {
  %0 = linalg.batch_matmul
      ins(%a, %b : tensor<?x8x8xf32>, tensor<?x8x8xf32>)
      outs(%init : tensor<?x8x8xf32>) -> tensor<?x8x8xf32>
  util.return %0 : tensor<?x8x8xf32>
}

// CHECK-LABEL: util.func public @dynamic_extent_is_untouched
// CHECK: linalg.batch_matmul
util.func public @dynamic_extent_is_untouched(
    %a: tensor<4x?x8xf32>,
    %b: tensor<4x8x8xf32>,
    %init: tensor<4x?x8xf32>) -> tensor<4x?x8xf32> {
  %0 = linalg.batch_matmul
      ins(%a, %b : tensor<4x?x8xf32>, tensor<4x8x8xf32>)
      outs(%init : tensor<4x?x8xf32>) -> tensor<4x?x8xf32>
  util.return %0 : tensor<4x?x8xf32>
}

// Above the bound the IR growth is not worth paying at compile time for a
// result nobody has measured, so a large batch stays batched.
// CHECK-LABEL: util.func public @batch_above_the_bound_is_untouched
// CHECK: linalg.batch_matmul
util.func public @batch_above_the_bound_is_untouched(
    %a: tensor<65x8x8xf32>,
    %b: tensor<65x8x8xf32>,
    %init: tensor<65x8x8xf32>) -> tensor<65x8x8xf32> {
  %0 = linalg.batch_matmul
      ins(%a, %b : tensor<65x8x8xf32>, tensor<65x8x8xf32>)
      outs(%init : tensor<65x8x8xf32>) -> tensor<65x8x8xf32>
  util.return %0 : tensor<65x8x8xf32>
}
