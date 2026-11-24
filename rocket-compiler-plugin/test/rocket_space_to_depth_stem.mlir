// RUN: iree-opt %s --pass-pipeline='builtin.module(util.func(rocket-space-to-depth-stem))' | FileCheck %s

// ResNet's stem: 7x7 stride 2 over a padded 230x230x3 image, 112x112x64 out.
// Folded 2x2 into channels it is a 4x4 stride-1 convolution over 115x115x12,
// with the kernel zero-padded to 8x8 and regrouped the same way.
// CHECK-LABEL: util.func public @resnet_stem
// CHECK: %[[SPLIT:.+]] = tensor.expand_shape %{{.+}} {{\[\[}}0], [1, 2], [3, 4], [5]] output_shape [1, 115, 2, 115, 2, 3]
// CHECK: %[[GROUPED:.+]] = linalg.transpose ins(%[[SPLIT]] : tensor<1x115x2x115x2x3xf16>) {{.+}} permutation = [0, 1, 3, 2, 4, 5]
// CHECK: %[[FOLDED:.+]] = tensor.collapse_shape %[[GROUPED]] {{\[\[}}0], [1], [2], [3, 4, 5]] : tensor<1x115x115x2x2x3xf16> into tensor<1x115x115x12xf16>
// CHECK: %[[PADDED:.+]] = tensor.pad %{{.+}} low[0, 0, 0, 0] high[1, 1, 0, 0]
// CHECK: tensor<7x7x3x64xf16> to tensor<8x8x3x64xf16>
// CHECK: %[[KSPLIT:.+]] = tensor.expand_shape %[[PADDED]] {{\[\[}}0, 1], [2, 3], [4], [5]] output_shape [4, 2, 4, 2, 3, 64]
// CHECK: linalg.transpose ins(%[[KSPLIT]] : tensor<4x2x4x2x3x64xf16>) {{.+}} permutation = [0, 2, 1, 3, 4, 5]
// CHECK: %[[KERNEL:.+]] = tensor.collapse_shape %{{.+}} {{\[\[}}0], [1], [2, 3, 4], [5]] : tensor<4x4x2x2x3x64xf16> into tensor<4x4x12x64xf16>
// CHECK: linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%[[FOLDED]], %[[KERNEL]] : tensor<1x115x115x12xf16>, tensor<4x4x12x64xf16>)
util.func public @resnet_stem(%x: tensor<1x230x230x3xf16>, %w: tensor<7x7x3x64xf16>, %init: tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32> {
  %0 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%x, %w : tensor<1x230x230x3xf16>, tensor<7x7x3x64xf16>) outs(%init : tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32>
  util.return %0 : tensor<1x112x112x64xf32>
}

// An input one row and column short of the padded kernel's reach (229: the
// 7x7 kernel never reads further) is zero-padded, and one with a remainder
// the convolution never reads is cropped.
// CHECK-LABEL: util.func public @short_and_long
// CHECK: tensor.pad %{{.+}} low[0, 0, 0, 0] high[0, 1, 1, 0]
// CHECK: tensor.extract_slice %{{.+}}[0, 0, 0, 0] [1, 230, 230, 3] [1, 1, 1, 1]
util.func public @short_and_long(%short: tensor<1x229x229x3xf16>, %long: tensor<1x231x231x3xf16>, %w: tensor<7x7x3x64xf16>, %init: tensor<1x112x112x64xf32>) -> (tensor<1x112x112x64xf32>, tensor<1x112x112x64xf32>) {
  %0 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%short, %w : tensor<1x229x229x3xf16>, tensor<7x7x3x64xf16>) outs(%init : tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32>
  %1 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%long, %w : tensor<1x231x231x3xf16>, tensor<7x7x3x64xf16>) outs(%init : tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32>
  util.return %0, %1 : tensor<1x112x112x64xf32>, tensor<1x112x112x64xf32>
}

// Not a narrow float 7x7 stride-2 stem: integer, eight channels, stride 1.
// CHECK-LABEL: util.func public @declines
// CHECK-COUNT-3: linalg.conv_2d_nhwc_hwcf {{.+}} ins(%{{.+}}, %{{.+}} : tensor<1x{{(230x230x3xi8|230x230x8xf16|118x118x3xf16)}}
util.func public @declines(%i8: tensor<1x230x230x3xi8>, %wi8: tensor<7x7x3x64xi8>, %ii: tensor<1x112x112x64xi32>, %wide: tensor<1x230x230x8xf16>, %w8: tensor<7x7x8x64xf16>, %s1: tensor<1x118x118x3xf16>, %w: tensor<7x7x3x64xf16>, %init: tensor<1x112x112x64xf32>) -> (tensor<1x112x112x64xi32>, tensor<1x112x112x64xf32>, tensor<1x112x112x64xf32>) {
  %0 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%i8, %wi8 : tensor<1x230x230x3xi8>, tensor<7x7x3x64xi8>) outs(%ii : tensor<1x112x112x64xi32>) -> tensor<1x112x112x64xi32>
  %1 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%wide, %w8 : tensor<1x230x230x8xf16>, tensor<7x7x8x64xf16>) outs(%init : tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32>
  %2 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%s1, %w : tensor<1x118x118x3xf16>, tensor<7x7x3x64xf16>) outs(%init : tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32>
  util.return %0, %1, %2 : tensor<1x112x112x64xi32>, tensor<1x112x112x64xf32>, tensor<1x112x112x64xf32>
}
