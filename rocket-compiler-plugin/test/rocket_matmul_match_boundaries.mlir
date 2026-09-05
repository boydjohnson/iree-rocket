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

// Boundary coverage for @match_rocket_matmul. A matmul reaches this hardware
// as a height-one 1x1 convolution -- M is the convolution width, K its input
// channels and N its output channels -- so the bounds here are the HAL's
// channel ceilings under different names.

// MobileNetV2's classifier, and exactly the shape the ceilings were measured
// at: K = 1792 is MAX_INPUT_CHANNELS.
//
// Both narrowings must land *here*, in the caller, not in the operands the
// wrapper builds for itself: a truncf that reaches the dispatch from inside
// @call_rocket_matmul would re-narrow the constant weights on every inference
// (ISSUES.md P6 item 2). @__transform_main now inlines the wrappers, so the
// two truncf ops and the dispatch land in the same function and only their
// *order* distinguishes the two cases -- truncf before the dispatch, on the
// dispatch's f16 tensor operands, is what says the demotion happened before
// the match rather than after it.
// CHECK-LABEL: util.func public @classifier_matches
// CHECK-NOT: linalg.matmul
// CHECK: arith.truncf
// CHECK: arith.truncf
// CHECK: flow.dispatch @rocket_matmul_executable
// CHECK-SAME: tensor<?x?xf16>{{.*}}tensor<?x?xf16>
util.func public @classifier_matches(
    %lhs: tensor<1x1792xf32>,
    %rhs: tensor<1792x1001xf32>,
    %init: tensor<1x1001xf32>) -> tensor<1x1001xf32> {
  %result = linalg.matmul
      ins(%lhs, %rhs : tensor<1x1792xf32>, tensor<1792x1001xf32>)
      outs(%init : tensor<1x1001xf32>) -> tensor<1x1001xf32>
  util.return %result : tensor<1x1001xf32>
}

// One channel past it, and the whole matmul stays on the CPU rather than
// reaching a driver that would refuse it -- in f32, because
// rocket-promote-unclaimed-conv-inputs gives back what the demotion took.
// Running an unclaimed matmul in f16 would be pure loss.
// CHECK-LABEL: util.func public @k_past_the_ceiling_falls_back
// CHECK: linalg.matmul
// CHECK-SAME: ins(%{{.*}}, %{{.*}} : tensor<1x1793xf32>, tensor<1793x64xf32>)
util.func public @k_past_the_ceiling_falls_back(
    %lhs: tensor<1x1793xf32>,
    %rhs: tensor<1793x64xf32>,
    %init: tensor<1x64xf32>) -> tensor<1x64xf32> {
  %result = linalg.matmul
      ins(%lhs, %rhs : tensor<1x1793xf32>, tensor<1793x64xf32>)
      outs(%init : tensor<1x64xf32>) -> tensor<1x64xf32>
  util.return %result : tensor<1x64xf32>
}

// N is the output-channel count, bounded by MAX_OUTPUT_CHANNELS at the same
// 1792.
// CHECK-LABEL: util.func public @n_past_the_ceiling_falls_back
// CHECK: linalg.matmul
util.func public @n_past_the_ceiling_falls_back(
    %lhs: tensor<1x64xf32>,
    %rhs: tensor<64x1793xf32>,
    %init: tensor<1x1793xf32>) -> tensor<1x1793xf32> {
  %result = linalg.matmul
      ins(%lhs, %rhs : tensor<1x64xf32>, tensor<64x1793xf32>)
      outs(%init : tensor<1x1793xf32>) -> tensor<1x1793xf32>
  util.return %result : tensor<1x1793xf32>
}

// M becomes the convolution's width, and what bounds it is the 11-bit
// `CNA_DATA_SIZE0.datain_width`. It was 32 -- the vendor FC ladder's extent
// -- until ISSUES.md C10 was resolved on 2026-09-05: above
// `(K/32 - 1) * M <= 2047` CBUF entries a single row read its last 32
// channels from the wrong place, and the planner now splits such a row into
// column tiles instead. ViT-B/16's M 197 at K 768 is the shape that matters.
// CHECK-LABEL: util.func public @m_197_matches
// CHECK: flow.dispatch @rocket_matmul_executable
util.func public @m_197_matches(
    %lhs: tensor<197x768xf32>,
    %rhs: tensor<768x768xf32>,
    %init: tensor<197x768xf32>) -> tensor<197x768xf32> {
  %result = linalg.matmul
      ins(%lhs, %rhs : tensor<197x768xf32>, tensor<768x768xf32>)
      outs(%init : tensor<197x768xf32>) -> tensor<197x768xf32>
  util.return %result : tensor<197x768xf32>
}

// CHECK-LABEL: util.func public @m_2047_matches
// CHECK: flow.dispatch @rocket_matmul_executable
util.func public @m_2047_matches(
    %lhs: tensor<2047x64xf32>,
    %rhs: tensor<64x64xf32>,
    %init: tensor<2047x64xf32>) -> tensor<2047x64xf32> {
  %result = linalg.matmul
      ins(%lhs, %rhs : tensor<2047x64xf32>, tensor<64x64xf32>)
      outs(%init : tensor<2047x64xf32>) -> tensor<2047x64xf32>
  util.return %result : tensor<2047x64xf32>
}

// CHECK-LABEL: util.func public @m_2048_falls_back
// CHECK: linalg.matmul
util.func public @m_2048_falls_back(
    %lhs: tensor<2048x64xf32>,
    %rhs: tensor<64x64xf32>,
    %init: tensor<2048x64xf32>) -> tensor<2048x64xf32> {
  %result = linalg.matmul
      ins(%lhs, %rhs : tensor<2048x64xf32>, tensor<64x64xf32>)
      outs(%init : tensor<2048x64xf32>) -> tensor<2048x64xf32>
  util.return %result : tensor<2048x64xf32>
}

// The case a name-only matcher would get wrong. `linalg.matmul` expresses a
// transposed operand by overriding its indexing maps rather than by being a
// different op, and a transposed B is a memory layout the height-one
// convolution lowering cannot pack. Same shapes as the classifier otherwise.
//
// This is also the case that proves the demotion carries `indexing_maps`
// across its rebuild: `getPrunedAttributeList` elides every inherent
// attribute, and a matmul rebuilt without its maps is an *untransposed* one
// -- which would then match, and be packed with the operand transposed. The
// maps below are checked on the way out for exactly that reason.
// CHECK-LABEL: util.func public @transposed_rhs_falls_back
// CHECK: linalg.matmul
// CHECK-SAME: indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>]
// CHECK-SAME: ins(%{{.*}}, %{{.*}} : tensor<1x1792xf32>, tensor<1001x1792xf32>)
util.func public @transposed_rhs_falls_back(
    %lhs: tensor<1x1792xf32>,
    %rhs: tensor<1001x1792xf32>,
    %init: tensor<1x1001xf32>) -> tensor<1x1001xf32> {
  %result = linalg.matmul
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>]
      ins(%lhs, %rhs : tensor<1x1792xf32>, tensor<1001x1792xf32>)
      outs(%init : tensor<1x1001xf32>) -> tensor<1x1001xf32>
  util.return %result : tensor<1x1001xf32>
}

// A batched matmul is a different contraction with a batch dimension the
// Rocket ABI has no room for -- batch is fixed at one throughout.
// CHECK-LABEL: util.func public @batch_matmul_falls_back
// CHECK: linalg.batch_matmul
util.func public @batch_matmul_falls_back(
    %lhs: tensor<4x8x64xf32>,
    %rhs: tensor<4x64x64xf32>,
    %init: tensor<4x8x64xf32>) -> tensor<4x8x64xf32> {
  %result = linalg.batch_matmul
      ins(%lhs, %rhs : tensor<4x8x64xf32>, tensor<4x64x64xf32>)
      outs(%init : tensor<4x8x64xf32>) -> tensor<4x8x64xf32>
  util.return %result : tensor<4x8x64xf32>
}

// Except when that batch is one. ONNX MatMul over a `[1, tokens, features]`
// activation imports as exactly this, and it is every projection in a
// transformer; @__transform_main collapses the unit batch (generalize, fold
// unit dims through reshapes, re-specialize) before the match loop, so it
// reaches the matcher as the `linalg.matmul` above and dispatches.
// CHECK-LABEL: util.func public @unit_batch_matmul_matches
// CHECK-NOT: linalg.batch_matmul
// CHECK: flow.dispatch @rocket_matmul_executable
util.func public @unit_batch_matmul_matches(
    %lhs: tensor<1x197x768xf32>,
    %rhs: tensor<1x768x768xf32>,
    %init: tensor<1x197x768xf32>) -> tensor<1x197x768xf32> {
  %result = linalg.batch_matmul
      ins(%lhs, %rhs : tensor<1x197x768xf32>, tensor<1x768x768xf32>)
      outs(%init : tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  util.return %result : tensor<1x197x768xf32>
}

// ------------------------------------------------------------------- GEMV

// linalg.matvec and linalg.vecmat reach this same matcher, because
// rocket-expand-gemv-to-matmul raises them into linalg.matmul with a unit
// extent before the match loop runs. Nothing here knows they were ever
// anything else -- which is the point of raising rather than adding matchers.

// matvec pins N = 1.
// CHECK-LABEL: util.func public @matvec_reaches_the_matmul_matcher
// CHECK-NOT: linalg.matvec
// CHECK: flow.dispatch @rocket_matmul_executable
util.func public @matvec_reaches_the_matmul_matcher(
    %a: tensor<197x768xf32>,
    %y: tensor<768xf32>,
    %init: tensor<197xf32>) -> tensor<197xf32> {
  %result = linalg.matvec
      ins(%a, %y : tensor<197x768xf32>, tensor<768xf32>)
      outs(%init : tensor<197xf32>) -> tensor<197xf32>
  util.return %result : tensor<197xf32>
}

// vecmat pins M = 1.
// CHECK-LABEL: util.func public @vecmat_reaches_the_matmul_matcher
// CHECK-NOT: linalg.vecmat
// CHECK: flow.dispatch @rocket_matmul_executable
util.func public @vecmat_reaches_the_matmul_matcher(
    %y: tensor<768xf32>,
    %a: tensor<768x1000xf32>,
    %init: tensor<1000xf32>) -> tensor<1000xf32> {
  %result = linalg.vecmat
      ins(%y, %a : tensor<768xf32>, tensor<768x1000xf32>)
      outs(%init : tensor<1000xf32>) -> tensor<1000xf32>
  util.return %result : tensor<1000xf32>
}

// A raised GEMV is still bound by K: this one is one past the 1792 ceiling
// and must fall back like any other oversized matmul.
// CHECK-LABEL: util.func public @matvec_k_1793_rejected
// CHECK: linalg.matmul
util.func public @matvec_k_1793_rejected(
    %a: tensor<197x1793xf32>,
    %y: tensor<1793xf32>,
    %init: tensor<197xf32>) -> tensor<197xf32> {
  %result = linalg.matvec
      ins(%a, %y : tensor<197x1793xf32>, tensor<1793xf32>)
      outs(%init : tensor<197xf32>) -> tensor<197xf32>
  util.return %result : tensor<197xf32>
}

// linalg.dot is not raised, so it never reaches the matcher at all: it
// reduces to a scalar, and a dispatch plus a weight pack plus an output
// compaction to produce one number is not a trade worth making.
// CHECK-LABEL: util.func public @dot_falls_back
// CHECK: linalg.dot
util.func public @dot_falls_back(
    %x: tensor<768xf32>,
    %y: tensor<768xf32>,
    %init: tensor<f32>) -> tensor<f32> {
  %result = linalg.dot
      ins(%x, %y : tensor<768xf32>, tensor<768xf32>)
      outs(%init : tensor<f32>) -> tensor<f32>
  util.return %result : tensor<f32>
}
