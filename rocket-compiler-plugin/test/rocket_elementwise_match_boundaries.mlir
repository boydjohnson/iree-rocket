// RUN: sed 's|^ *//@ROCKET_ELEMENTWISE@||' %S/../target/Rocket/rocket_conv2d_transform_spec.mlir > %t.spec.mlir
// RUN: iree-compile %s \
// RUN:   --iree-preprocessing-transform-spec-filename=%t.spec.mlir \
// RUN:   --iree-hal-target-device=rocket_device=rocket \
// RUN:   --iree-hal-target-device=cpu_device=local \
// RUN:   --iree-hal-local-target-device-backends=llvm-cpu \
// RUN:   --iree-llvmcpu-target-cpu=generic \
// RUN:   --iree-hal-default-device=cpu_device \
// RUN:   --iree-hal-indirect-command-buffers=false \
// RUN:   --compile-to=preprocessing \
// RUN:   --mlir-print-op-generic=false \
// RUN:   -o - | FileCheck %s

// Boundary coverage for the three two-tensor element-wise matchers, on the
// same contract as rocket_pooling_match_boundaries.mlir: every accepted shape
// has an immediately-adjacent rejected one, so widening a matcher cannot
// silently claim something the hardware has no measurement for, and
// tightening one cannot silently lose the largest measured-good shape.
//
// The `sed` in the first RUN line is not incidental. These matchers ship
// **commented out** with a `//@ROCKET_ELEMENTWISE@` marker, because
// ISSUES.md P8 measured that at the current per-dispatch cost more offload
// sites make a model slower. `rocket-compiler --elementwise` uncomments
// exactly those lines; this test does the same thing the same way, so it
// covers the marker convention as well as the matchers.
//
// Inputs are written as `linalg.generic`, which is the form an ONNX
// Add/Mul/Sub actually reaches the pipeline as. `@__transform_main` runs
// `linalg-specialize-generic-ops` before the match loop and turns them into
// the named `linalg.add`/`mul`/`sub` the matchers are written against -- so
// writing the generic form here exercises that step too, and a change to it
// shows up as a lost match rather than as a silent CPU fallback.
//
// The bounds are board measurements, not conventions:
//   tokens   <= 197   ViT's sequence length
//   channels <= 3072  ViT-base's MLP hidden width, 384 fp16 surfaces
// both from `iree-rocket-hal/tests/ew_binary_hw.rs`, which is bit-exact at
// each. Its ladder previously stopped at 64 channels.

// ---------------------------------------------------------------- accepted

// ViT's own attention-block shape: 197 tokens of 768 channels, 96 fp16 surfaces.
// CHECK-LABEL: util.func public @mul_vit_attention
// CHECK-NOT: linalg.mul
// CHECK: flow.dispatch @rocket_elementwise_mul_executable
util.func public @mul_vit_attention(%lhs: tensor<1x197x768xf32>, %rhs: tensor<1x197x768xf32>) -> tensor<1x197x768xf32> {
  %init = tensor.empty() : tensor<1x197x768xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%lhs, %rhs : tensor<1x197x768xf32>, tensor<1x197x768xf32>)
      outs(%init : tensor<1x197x768xf32>) {
    ^bb0(%a: f32, %b: f32, %out: f32):
      %v = arith.mulf %a, %b : f32
      linalg.yield %v : f32
  } -> tensor<1x197x768xf32>
  util.return %result : tensor<1x197x768xf32>
}

// Add at the same shape. ViT carries 123 of these.
// CHECK-LABEL: util.func public @add_vit_attention
// CHECK-NOT: linalg.add
// CHECK: flow.dispatch @rocket_elementwise_add_executable
util.func public @add_vit_attention(%lhs: tensor<1x197x768xf32>, %rhs: tensor<1x197x768xf32>) -> tensor<1x197x768xf32> {
  %init = tensor.empty() : tensor<1x197x768xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%lhs, %rhs : tensor<1x197x768xf32>, tensor<1x197x768xf32>)
      outs(%init : tensor<1x197x768xf32>) {
    ^bb0(%a: f32, %b: f32, %out: f32):
      %v = arith.addf %a, %b : f32
      linalg.yield %v : f32
  } -> tensor<1x197x768xf32>
  util.return %result : tensor<1x197x768xf32>
}

// Sub at the same shape.
// CHECK-LABEL: util.func public @sub_vit_attention
// CHECK-NOT: linalg.sub
// CHECK: flow.dispatch @rocket_elementwise_sub_executable
util.func public @sub_vit_attention(%lhs: tensor<1x197x768xf32>, %rhs: tensor<1x197x768xf32>) -> tensor<1x197x768xf32> {
  %init = tensor.empty() : tensor<1x197x768xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%lhs, %rhs : tensor<1x197x768xf32>, tensor<1x197x768xf32>)
      outs(%init : tensor<1x197x768xf32>) {
    ^bb0(%a: f32, %b: f32, %out: f32):
      %v = arith.subf %a, %b : f32
      linalg.yield %v : f32
  } -> tensor<1x197x768xf32>
  util.return %result : tensor<1x197x768xf32>
}

// The channel ceiling exactly: ViT-base's MLP hidden width, 384 surfaces.
// CHECK-LABEL: util.func public @mul_vit_mlp
// CHECK-NOT: linalg.mul
// CHECK: flow.dispatch @rocket_elementwise_mul_executable
util.func public @mul_vit_mlp(%lhs: tensor<1x197x3072xf32>, %rhs: tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32> {
  %init = tensor.empty() : tensor<1x197x3072xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%lhs, %rhs : tensor<1x197x3072xf32>, tensor<1x197x3072xf32>)
      outs(%init : tensor<1x197x3072xf32>) {
    ^bb0(%a: f32, %b: f32, %out: f32):
      %v = arith.mulf %a, %b : f32
      linalg.yield %v : f32
  } -> tensor<1x197x3072xf32>
  util.return %result : tensor<1x197x3072xf32>
}

// The floor of both bounds -- one pixel, one channel.
// CHECK-LABEL: util.func public @mul_smallest
// CHECK-NOT: linalg.mul
// CHECK: flow.dispatch @rocket_elementwise_mul_executable
util.func public @mul_smallest(%lhs: tensor<1x1x1xf32>, %rhs: tensor<1x1x1xf32>) -> tensor<1x1x1xf32> {
  %init = tensor.empty() : tensor<1x1x1xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%lhs, %rhs : tensor<1x1x1xf32>, tensor<1x1x1xf32>)
      outs(%init : tensor<1x1x1xf32>) {
    ^bb0(%a: f32, %b: f32, %out: f32):
      %v = arith.mulf %a, %b : f32
      linalg.yield %v : f32
  } -> tensor<1x1x1xf32>
  util.return %result : tensor<1x1x1xf32>
}

// ---------------------------------------------------------------- rejected

// One token past the measured 197. Nothing wider has been run on hardware.
// CHECK-LABEL: util.func public @mul_tokens_over
// CHECK: linalg.mul
// CHECK-NOT: flow.dispatch @rocket_elementwise_mul_executable
util.func public @mul_tokens_over(%lhs: tensor<1x198x768xf32>, %rhs: tensor<1x198x768xf32>) -> tensor<1x198x768xf32> {
  %init = tensor.empty() : tensor<1x198x768xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%lhs, %rhs : tensor<1x198x768xf32>, tensor<1x198x768xf32>)
      outs(%init : tensor<1x198x768xf32>) {
    ^bb0(%a: f32, %b: f32, %out: f32):
      %v = arith.mulf %a, %b : f32
      linalg.yield %v : f32
  } -> tensor<1x198x768xf32>
  util.return %result : tensor<1x198x768xf32>
}

// One channel past the measured 3072.
// CHECK-LABEL: util.func public @mul_channels_over
// CHECK: linalg.mul
// CHECK-NOT: flow.dispatch @rocket_elementwise_mul_executable
util.func public @mul_channels_over(%lhs: tensor<1x197x3073xf32>, %rhs: tensor<1x197x3073xf32>) -> tensor<1x197x3073xf32> {
  %init = tensor.empty() : tensor<1x197x3073xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%lhs, %rhs : tensor<1x197x3073xf32>, tensor<1x197x3073xf32>)
      outs(%init : tensor<1x197x3073xf32>) {
    ^bb0(%a: f32, %b: f32, %out: f32):
      %v = arith.mulf %a, %b : f32
      linalg.yield %v : f32
  } -> tensor<1x197x3073xf32>
  util.return %result : tensor<1x197x3073xf32>
}

// A real batch. The cube has three extents and the leading 1 is what makes
// `width = tokens, height = 1` a sound mapping; batch 2 would need a fourth.
// CHECK-LABEL: util.func public @mul_batched
// CHECK: linalg.mul
// CHECK-NOT: flow.dispatch @rocket_elementwise_mul_executable
util.func public @mul_batched(%lhs: tensor<2x197x768xf32>, %rhs: tensor<2x197x768xf32>) -> tensor<2x197x768xf32> {
  %init = tensor.empty() : tensor<2x197x768xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%lhs, %rhs : tensor<2x197x768xf32>, tensor<2x197x768xf32>)
      outs(%init : tensor<2x197x768xf32>) {
    ^bb0(%a: f32, %b: f32, %out: f32):
      %v = arith.mulf %a, %b : f32
      linalg.yield %v : f32
  } -> tensor<2x197x768xf32>
  util.return %result : tensor<2x197x768xf32>
}
