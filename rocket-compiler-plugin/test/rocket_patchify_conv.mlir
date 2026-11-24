// RUN: iree-opt %s --pass-pipeline='builtin.module(util.func(rocket-patchify-conv-to-matmul))' | FileCheck %s

// A ViT patch embedding: 16x16 stride 16 over a 224x224x3 image, which is a
// [196, 768] x [768, 768] matmul once the image is rearranged patch-major.
// CHECK-LABEL: util.func public @vit_patch_embedding
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf
// CHECK: %[[PATCHES:.+]] = tensor.expand_shape %{{.+}} {{\[\[}}0], [1, 2], [3, 4], [5]] output_shape [1, 14, 16, 14, 16, 3]
// CHECK: %[[MAJOR:.+]] = linalg.transpose ins(%[[PATCHES]] : tensor<1x14x16x14x16x3xf16>) {{.+}} permutation = [0, 1, 3, 2, 4, 5]
// CHECK: %[[LHS:.+]] = tensor.collapse_shape %[[MAJOR]] {{\[\[}}0, 1, 2], [3, 4, 5]] : tensor<1x14x14x16x16x3xf16> into tensor<196x768xf16>
// CHECK: %[[RHS:.+]] = tensor.collapse_shape %{{.+}} {{\[\[}}0, 1, 2], [3]] : tensor<16x16x3x768xf16> into tensor<768x768xf16>
// CHECK: %[[ACC:.+]] = tensor.collapse_shape %{{.+}} {{\[\[}}0, 1, 2], [3]] : tensor<1x14x14x768xf32> into tensor<196x768xf32>
// CHECK: %[[PRODUCT:.+]] = linalg.matmul ins(%[[LHS]], %[[RHS]] : tensor<196x768xf16>, tensor<768x768xf16>) outs(%[[ACC]] : tensor<196x768xf32>)
// CHECK: tensor.expand_shape %[[PRODUCT]] {{\[\[}}0, 1, 2], [3]] output_shape [1, 14, 14, 768]
util.func public @vit_patch_embedding(%x: tensor<1x224x224x3xf16>, %w: tensor<16x16x3x768xf16>, %init: tensor<1x14x14x768xf32>) -> tensor<1x14x14x768xf32> {
  %0 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<16> : tensor<2xi64>} ins(%x, %w : tensor<1x224x224x3xf16>, tensor<16x16x3x768xf16>) outs(%init : tensor<1x14x14x768xf32>) -> tensor<1x14x14x768xf32>
  util.return %0 : tensor<1x14x14x768xf32>
}

// A floor-mode remainder: 57 rows at stride 2 is 28 output rows, and the
// 57th row is never read, so it is cropped before the rearrangement.
// CHECK-LABEL: util.func public @downsample_with_remainder
// CHECK: tensor.extract_slice %{{.+}}[0, 0, 0, 0] [1, 56, 56, 96] [1, 1, 1, 1]
// CHECK: linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<784x384xf16>, tensor<384x192xf16>)
util.func public @downsample_with_remainder(%x: tensor<1x57x57x96xf16>, %w: tensor<2x2x96x192xf16>, %init: tensor<1x28x28x192xf32>) -> tensor<1x28x28x192xf32> {
  %0 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%x, %w : tensor<1x57x57x96xf16>, tensor<2x2x96x192xf16>) outs(%init : tensor<1x28x28x192xf32>) -> tensor<1x28x28x192xf32>
  util.return %0 : tensor<1x28x28x192xf32>
}

// Overlapping windows (stride below the kernel), dilation, and a 1x1 are
// all left alone.
// CHECK-LABEL: util.func public @not_patchify
// CHECK-COUNT-3: linalg.conv_2d_nhwc_hwcf
// CHECK-NOT: linalg.matmul
util.func public @not_patchify(%x: tensor<1x16x16x8xf16>, %w3: tensor<3x3x8x8xf16>, %w1: tensor<1x1x8x8xf16>, %i7: tensor<1x7x7x8xf32>, %i4: tensor<1x4x4x8xf32>, %i16: tensor<1x16x16x8xf32>) -> (tensor<1x7x7x8xf32>, tensor<1x4x4x8xf32>, tensor<1x16x16x8xf32>) {
  %overlap = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%x, %w3 : tensor<1x16x16x8xf16>, tensor<3x3x8x8xf16>) outs(%i7 : tensor<1x7x7x8xf32>) -> tensor<1x7x7x8xf32>
  %dilated = linalg.conv_2d_nhwc_hwcf {dilations = dense<2> : tensor<2xi64>, strides = dense<3> : tensor<2xi64>} ins(%x, %w3 : tensor<1x16x16x8xf16>, tensor<3x3x8x8xf16>) outs(%i4 : tensor<1x4x4x8xf32>) -> tensor<1x4x4x8xf32>
  %pointwise = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%x, %w1 : tensor<1x16x16x8xf16>, tensor<1x1x8x8xf16>) outs(%i16 : tensor<1x16x16x8xf32>) -> tensor<1x16x16x8xf32>
  util.return %overlap, %dilated, %pointwise : tensor<1x7x7x8xf32>, tensor<1x4x4x8xf32>, tensor<1x16x16x8xf32>
}
