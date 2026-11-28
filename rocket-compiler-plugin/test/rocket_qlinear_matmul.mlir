// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(rocket-transpose-quantized-conv-to-nhwc))" %s | FileCheck %s --check-prefix=REWRITE
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
// RUN:   -o - | FileCheck %s --check-prefix=SPEC

// An ONNX QLinearMatMul, as `tools/quantize_onnx_int8.py --quantize-classifier`
// writes ResNet50's classifier and torch-mlir imports it: the `ui8`
// activation shifted to signed, a `linalg.quantized_matmul` against the
// constant `[K, N]` weight with the shifted zero point -128, then the same
// five-op requantization epilogue a QLinearConv has (scale, then divide by
// the output scale, round, offset, clamp to the `ui8` range, `fptoui`).
//
// `rocket-transpose-quantized-conv-to-nhwc` writes it as the 1x1
// convolution of height one `fc.rs` lowers an FC to; through the whole
// spec, `quantized-conv-to-conv` and `rocket-fuse-int8-requant-epilogue`
// then take it to the requantized int8 1x1 executable, with the collapse
// back to `[M, N]` after the fused epilogue, on i8.

#map2 = affine_map<(d0, d1) -> (d0, d1)>

// REWRITE-LABEL: util.func public @qlinear_matmul
// REWRITE:       %[[X:.+]] = tensor.expand_shape %{{.+}} {{\[}}[0, 1, 2], [3]] output_shape [1, 1, 1, 256] : tensor<1x256xi8> into tensor<1x1x1x256xi8>
//   The weight's expand folds into the constant: the filter is a literal
//   `tensor<1x1x256x64xi8>`.
// REWRITE:       %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf_q
// REWRITE-SAME:      ins(%[[X]], %{{.+}}, %{{.+}}, %{{.+}} : tensor<1x1x1x256xi8>, tensor<1x1x256x64xi8>, i32, i32)
// REWRITE:       tensor.collapse_shape %[[CONV]] {{\[}}[0, 1, 2], [3]] : tensor<1x1x1x64xi32> into tensor<1x64xi32>
// REWRITE-NOT:   linalg.quantized_matmul

// SPEC-LABEL: util.func public @qlinear_matmul
// SPEC-NOT:   linalg.quantized_matmul
// SPEC:       flow.dispatch @rocket_dynamic_int8_requant_executable::
// SPEC-NOT:   arith.fptoui
// SPEC:       util.return
util.func public @qlinear_matmul(%x: tensor<1x256xi8>) -> tensor<1x64xi8> {
  %shift = arith.constant -128 : i8
  %x_zp = arith.constant -128 : i32
  %w_zp = arith.constant 0 : i32
  %c0 = arith.constant 0 : i32
  %weights = arith.constant dense<3> : tensor<256x64xi8>
  %acc_scale = arith.constant 2.000000e-04 : f32
  %out_scale = arith.constant 3.125000e-02 : f32
  %out_zp = arith.constant 5.900000e+01 : f32
  %zero = arith.constant 0.000000e+00 : f32
  %max = arith.constant 2.550000e+02 : f32
  %s_empty = tensor.empty() : tensor<1x256xi8>
  %signed = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]}
      ins(%x : tensor<1x256xi8>) outs(%s_empty : tensor<1x256xi8>) {
  ^bb0(%in: i8, %o: i8):
    %a = arith.addi %in, %shift : i8
    linalg.yield %a : i8
  } -> tensor<1x256xi8>
  %acc_empty = tensor.empty() : tensor<1x64xi32>
  %acc_init = linalg.fill ins(%c0 : i32) outs(%acc_empty : tensor<1x64xi32>) -> tensor<1x64xi32>
  %acc = linalg.quantized_matmul ins(%signed, %weights, %x_zp, %w_zp : tensor<1x256xi8>, tensor<256x64xi8>, i32, i32)
      outs(%acc_init : tensor<1x64xi32>) -> tensor<1x64xi32>
  %real_empty = tensor.empty() : tensor<1x64xf32>
  %real = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]}
      ins(%acc : tensor<1x64xi32>) outs(%real_empty : tensor<1x64xf32>) {
  ^bb0(%in: i32, %o: f32):
    %f = arith.sitofp %in : i32 to f32
    %s = arith.mulf %f, %acc_scale : f32
    linalg.yield %s : f32
  } -> tensor<1x64xf32>
  %q_empty = tensor.empty() : tensor<1x64xi8>
  %q = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]}
      ins(%real : tensor<1x64xf32>) outs(%q_empty : tensor<1x64xi8>) {
  ^bb0(%in: f32, %o: i8):
    %d = arith.divf %in, %out_scale : f32
    %r = math.roundeven %d : f32
    %z = arith.addf %r, %out_zp : f32
    %lo = arith.maximumf %z, %zero : f32
    %hi = arith.minimumf %lo, %max : f32
    %n = arith.fptoui %hi : f32 to i8
    linalg.yield %n : i8
  } -> tensor<1x64xi8>
  util.return %q : tensor<1x64xi8>
}

// -----

// Declined: a dynamic `M` would need an `output_shape` for the expand, and no
// Rocket matcher takes a dynamic extent anyway.

// REWRITE-LABEL: util.func public @dynamic_quantized_matmul_declined
// REWRITE:       linalg.quantized_matmul
// REWRITE-NOT:   linalg.conv_2d_nhwc_hwcf_q
util.func public @dynamic_quantized_matmul_declined(%x: tensor<?x256xi8>, %init: tensor<?x64xi32>) -> tensor<?x64xi32> {
  %x_zp = arith.constant -128 : i32
  %w_zp = arith.constant 0 : i32
  %weights = arith.constant dense<3> : tensor<256x64xi8>
  %acc = linalg.quantized_matmul ins(%x, %weights, %x_zp, %w_zp : tensor<?x256xi8>, tensor<256x64xi8>, i32, i32)
      outs(%init : tensor<?x64xi32>) -> tensor<?x64xi32>
  util.return %acc : tensor<?x64xi32>
}
