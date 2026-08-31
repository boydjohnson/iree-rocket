// RUN: iree-opt --pass-pipeline="builtin.module(rocket-expand-onnx-conv-integer)" %s \
// RUN:   | FileCheck %s --check-prefix=TORCH
// RUN: iree-compile %s --iree-input-type=onnx --compile-to=input -o - \
// RUN:   | FileCheck %s --check-prefix=LINALG

// onnx.ConvInteger has no torch-mlir OnnxToTorch pattern at any opset, so
// without RocketExpandOnnxConvIntegerPass every one of these fails input
// conversion with "failed to legalize operation 'torch.operator'". The
// second RUN line is the one that matters for the ONNX Runtime
// quantize_dynamic route (DynamicQuantizeLinear + ConvInteger): it asserts
// the expansion lands on linalg's *quantized* convolutions, i8 operands with
// an i32 accumulator, rather than a widened i32 convolution.

module {
  // Dense 3x3/s2 with asymmetric ONNX pads, the MobileNetV2 stem. Asymmetric
  // pads have no aten.convolution spelling, so the pass materializes them as
  // an aten.constant_pad_nd filled with the input zero point, ahead of the
  // quantized cast.
  // TORCH-LABEL: func.func @conv_integer_dense
  // TORCH: %[[XZP:.+]] = torch.aten.item %arg2
  // TORCH: %[[WZP:.+]] = torch.aten.item %arg3
  // TORCH: %[[PAD:.+]] = torch.aten.constant_pad_nd %arg0, %{{.+}}, %[[XZP]]
  // TORCH: %[[QX:.+]] = torch.aten._make_per_tensor_quantized_tensor %[[PAD]], %{{.+}}, %[[XZP]] {{.*}} -> !torch.vtensor<[1,3,225,225],!torch.quint8>
  // TORCH: %[[QW:.+]] = torch.aten._make_per_tensor_quantized_tensor %arg1, %{{.+}}, %[[WZP]] {{.*}} -> !torch.vtensor<[48,3,3,3],!torch.qint8>
  // TORCH: %[[CONV:.+]] = torch.aten.convolution %[[QX]], %[[QW]]{{.*}} -> !torch.vtensor<[1,48,112,112],!torch.qint32>
  // TORCH: torch.aten.int_repr %[[CONV]] : !torch.vtensor<[1,48,112,112],!torch.qint32> -> !torch.vtensor<[1,48,112,112],si32>
  // TORCH-NOT: onnx.ConvInteger

  // LINALG-LABEL: util.func public @conv_integer_dense
  // LINALG: linalg.conv_2d_nchw_fchw_q
  // LINALG-SAME: ins(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} : tensor<1x3x225x225xi8>, tensor<48x3x3x3xi8>, i32, i32)
  // LINALG-SAME: outs(%{{.+}} : tensor<1x48x112x112xi32>) -> tensor<1x48x112x112xi32>
  func.func @conv_integer_dense(%x: !torch.vtensor<[1,3,224,224],ui8>, %w: !torch.vtensor<[48,3,3,3],si8>, %xz: !torch.vtensor<[],ui8>, %wz: !torch.vtensor<[],si8>) -> !torch.vtensor<[1,48,112,112],si32> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64} {
    %0 = torch.operator "onnx.ConvInteger"(%x, %w, %xz, %wz) {torch.onnx.dilations = [1 : si64, 1 : si64], torch.onnx.group = 1 : si64, torch.onnx.kernel_shape = [3 : si64, 3 : si64], torch.onnx.pads = [0 : si64, 0 : si64, 1 : si64, 1 : si64], torch.onnx.strides = [2 : si64, 2 : si64]} : (!torch.vtensor<[1,3,224,224],ui8>, !torch.vtensor<[48,3,3,3],si8>, !torch.vtensor<[],ui8>, !torch.vtensor<[],si8>) -> !torch.vtensor<[1,48,112,112],si32>
    return %0 : !torch.vtensor<[1,48,112,112],si32>
  }

  // Depthwise (group == input channels) with symmetric pads, which are folded
  // into aten.convolution's own padding operand instead: Torch->Linalg already
  // pads a quantized convolution with the input zero point, which is what ONNX
  // requires of ConvInteger.
  // TORCH-LABEL: func.func @conv_integer_depthwise
  // TORCH-NOT: torch.aten.constant_pad_nd
  // TORCH: torch.aten.convolution
  // TORCH-NOT: onnx.ConvInteger

  // LINALG-LABEL: util.func public @conv_integer_depthwise
  // LINALG: linalg.depthwise_conv_2d_nhwc_hwc_q
  func.func @conv_integer_depthwise(%x: !torch.vtensor<[1,48,112,112],ui8>, %w: !torch.vtensor<[48,1,3,3],si8>, %xz: !torch.vtensor<[],ui8>, %wz: !torch.vtensor<[],si8>) -> !torch.vtensor<[1,48,112,112],si32> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64} {
    %0 = torch.operator "onnx.ConvInteger"(%x, %w, %xz, %wz) {torch.onnx.dilations = [1 : si64, 1 : si64], torch.onnx.group = 48 : si64, torch.onnx.kernel_shape = [3 : si64, 3 : si64], torch.onnx.pads = [1 : si64, 1 : si64, 1 : si64, 1 : si64], torch.onnx.strides = [1 : si64, 1 : si64]} : (!torch.vtensor<[1,48,112,112],ui8>, !torch.vtensor<[48,1,3,3],si8>, !torch.vtensor<[],ui8>, !torch.vtensor<[],si8>) -> !torch.vtensor<[1,48,112,112],si32>
    return %0 : !torch.vtensor<[1,48,112,112],si32>
  }

  // 1x1 with both zero points omitted -- ONNX defaults them to 0 -- and a
  // dynamic batch, the shape iree-import-onnx produces before batch pinning.
  // TORCH-LABEL: func.func @conv_integer_no_zero_points
  // TORCH-NOT: torch.aten.item
  // TORCH: torch.aten.convolution
  // TORCH-NOT: onnx.ConvInteger

  // LINALG-LABEL: util.func public @conv_integer_no_zero_points
  // LINALG: linalg.conv_2d_nchw_fchw_q
  func.func @conv_integer_no_zero_points(%x: !torch.vtensor<[?,48,112,112],ui8>, %w: !torch.vtensor<[24,48,1,1],si8>) -> !torch.vtensor<[?,24,112,112],si32> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64} {
    %0 = torch.operator "onnx.ConvInteger"(%x, %w) {torch.onnx.dilations = [1 : si64, 1 : si64], torch.onnx.group = 1 : si64, torch.onnx.kernel_shape = [1 : si64, 1 : si64], torch.onnx.pads = [0 : si64, 0 : si64, 0 : si64, 0 : si64], torch.onnx.strides = [1 : si64, 1 : si64]} : (!torch.vtensor<[?,48,112,112],ui8>, !torch.vtensor<[24,48,1,1],si8>) -> !torch.vtensor<[?,24,112,112],si32>
    return %0 : !torch.vtensor<[?,24,112,112],si32>
  }

  // auto_pad=SAME_UPPER, which ONNX leaves implicit: the pass resolves it to
  // the same asymmetric [0,0,1,1] the stem spells out explicitly above.
  // TORCH-LABEL: func.func @conv_integer_same_upper
  // TORCH: torch.aten.constant_pad_nd
  // TORCH-NOT: onnx.ConvInteger

  // LINALG-LABEL: util.func public @conv_integer_same_upper
  // LINALG: tensor<1x3x225x225xi8>
  // LINALG: linalg.conv_2d_nchw_fchw_q
  func.func @conv_integer_same_upper(%x: !torch.vtensor<[1,3,224,224],ui8>, %w: !torch.vtensor<[48,3,3,3],si8>, %xz: !torch.vtensor<[],ui8>, %wz: !torch.vtensor<[],si8>) -> !torch.vtensor<[1,48,112,112],si32> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64} {
    %0 = torch.operator "onnx.ConvInteger"(%x, %w, %xz, %wz) {torch.onnx.auto_pad = "SAME_UPPER", torch.onnx.dilations = [1 : si64, 1 : si64], torch.onnx.group = 1 : si64, torch.onnx.kernel_shape = [3 : si64, 3 : si64], torch.onnx.strides = [2 : si64, 2 : si64]} : (!torch.vtensor<[1,3,224,224],ui8>, !torch.vtensor<[48,3,3,3],si8>, !torch.vtensor<[],ui8>, !torch.vtensor<[],si8>) -> !torch.vtensor<[1,48,112,112],si32>
    return %0 : !torch.vtensor<[1,48,112,112],si32>
  }
}
