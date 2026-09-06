// RUN: iree-opt %s --pass-pipeline='builtin.module(rocket-expand-gemv-to-matmul)' \
// RUN:   | FileCheck %s

// The pass in isolation. A GEMV is a matmul with one extent pinned to 1, and
// everything downstream of this point already handles a matmul with a unit
// extent, so raising is cheaper than teaching the demotion, the matcher, the
// shim and the executable about two more ops.
//
// Both reshapes are pure metadata on a contiguous tensor, so this costs
// nothing at runtime.

// matvec pins N: the vector is the rhs and becomes [k, 1], and the
// accumulator becomes [m, 1] -- the real extent on dim 0 in both, which is
// the part that is easy to get backwards.
// CHECK-LABEL: util.func public @matvec
// CHECK-NOT: linalg.matvec
// CHECK: %[[RHS:.+]] = tensor.expand_shape %{{.+}} {{\[}}[0, 1]] output_shape [768, 1]
// CHECK-SAME: tensor<768xf32> into tensor<768x1xf32>
// CHECK: %[[INIT:.+]] = tensor.expand_shape %{{.+}} {{\[}}[0, 1]] output_shape [197, 1]
// CHECK-SAME: tensor<197xf32> into tensor<197x1xf32>
// CHECK: %[[MM:.+]] = linalg.matmul ins(%{{.+}}, %[[RHS]] : tensor<197x768xf32>, tensor<768x1xf32>)
// CHECK-SAME: outs(%[[INIT]] : tensor<197x1xf32>)
// CHECK: tensor.collapse_shape %[[MM]] {{\[}}[0, 1]] : tensor<197x1xf32> into tensor<197xf32>
util.func public @matvec(
    %a: tensor<197x768xf32>,
    %y: tensor<768xf32>,
    %init: tensor<197xf32>) -> tensor<197xf32> {
  %result = linalg.matvec
      ins(%a, %y : tensor<197x768xf32>, tensor<768xf32>)
      outs(%init : tensor<197xf32>) -> tensor<197xf32>
  util.return %result : tensor<197xf32>
}

// vecmat pins M instead: the vector is the lhs and becomes [1, k], the
// accumulator [1, n] -- the real extent on dim 1 in both.
// CHECK-LABEL: util.func public @vecmat
// CHECK-NOT: linalg.vecmat
// CHECK: %[[LHS:.+]] = tensor.expand_shape %{{.+}} {{\[}}[0, 1]] output_shape [1, 768]
// CHECK-SAME: tensor<768xf32> into tensor<1x768xf32>
// CHECK: %[[INIT:.+]] = tensor.expand_shape %{{.+}} {{\[}}[0, 1]] output_shape [1, 1000]
// CHECK-SAME: tensor<1000xf32> into tensor<1x1000xf32>
// CHECK: %[[MM:.+]] = linalg.matmul ins(%[[LHS]], %{{.+}} : tensor<1x768xf32>, tensor<768x1000xf32>)
// CHECK-SAME: outs(%[[INIT]] : tensor<1x1000xf32>)
// CHECK: tensor.collapse_shape %[[MM]] {{\[}}[0, 1]] : tensor<1x1000xf32> into tensor<1000xf32>
util.func public @vecmat(
    %y: tensor<768xf32>,
    %a: tensor<768x1000xf32>,
    %init: tensor<1000xf32>) -> tensor<1000xf32> {
  %result = linalg.vecmat
      ins(%y, %a : tensor<768xf32>, tensor<768x1000xf32>)
      outs(%init : tensor<1000xf32>) -> tensor<1000xf32>
  util.return %result : tensor<1000xf32>
}

// linalg.dot is left alone deliberately: it reduces two vectors to a scalar,
// so offloading it would spend a dispatch, a weight pack and an output
// compaction to produce one number. Same reasoning as the 1x1 pool identity.
// CHECK-LABEL: util.func public @dot_is_left_alone
// CHECK: linalg.dot
// CHECK-NOT: linalg.matmul
util.func public @dot_is_left_alone(
    %x: tensor<768xf32>,
    %y: tensor<768xf32>,
    %init: tensor<f32>) -> tensor<f32> {
  %result = linalg.dot
      ins(%x, %y : tensor<768xf32>, tensor<768xf32>)
      outs(%init : tensor<f32>) -> tensor<f32>
  util.return %result : tensor<f32>
}
