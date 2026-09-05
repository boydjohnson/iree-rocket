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
// RUN: iree-compile %s \
// RUN:   --iree-preprocessing-transform-spec-filename=%S/../target/Rocket/rocket_conv2d_transform_spec.mlir \
// RUN:   --iree-hal-target-device=rocket_device=rocket \
// RUN:   --iree-hal-target-device=cpu_device=local \
// RUN:   --iree-hal-local-target-device-backends=llvm-cpu \
// RUN:   --iree-llvmcpu-target-cpu=generic \
// RUN:   --iree-hal-default-device=cpu_device \
// RUN:   --iree-hal-indirect-command-buffers=false \
// RUN:   --compile-to=executable-targets \
// RUN:   -o - | FileCheck %s --check-prefix=TARGET

// The int8 offload path, entered by an ONNX ConvInteger model after
// RocketExpandOnnxConvIntegerPass: quantized linalg convs with a *non-zero*
// runtime input zero point (ORT's quantize_dynamic never produces zero) and
// a constant-zero weight zero point.
//
// @__transform_main dequantizes these before matching --
// rocket-transpose-quantized-conv-to-nhwc puts the dense one in NHWC, then
// iree-global-opt-quantized-conv-to-conv folds the zero point out into a
// separate CPU correction -- so what the matchers see is an ordinary named
// conv on i8 with an i32 accumulator. Nothing here may reach the fp16
// executables.

// @__transform_main inlines the @call_rocket_* wrappers, so what each adapter
// builds is checked inside the function that used to call it rather than in a
// `util.func private` of its own.

// Each conv is claimed, and the dense one reaches its matcher only because
// rocket-transpose-quantized-conv-to-nhwc moved it out of NCHW first.
// CHECK-LABEL: util.func public @dense_1x1
// CHECK: linalg.transpose {{.*}} permutation = [0, 2, 3, 1]
// CHECK: linalg.transpose {{.*}} permutation = [2, 3, 1, 0]
// The bias binding is i32, not i8: rocket-hal-driver reads it as
// output_channels * i32 for both int8 precisions.
// CHECK: linalg.fill ins(%{{.+}} : i32) outs(%{{.+}} : tensor<24xi32>)
// CHECK: flow.dispatch @rocket_dynamic_int8_executable::@rocket_dynamic_conv2d_v1::@rocket_dynamic_conv2d(
// One i32 push constant per settable Conv2D dimension, as in the fp16 path.
// CHECK-SAME: : (i32, i32, i32, i32, i32, i32,
// int8_accumulator already hands back i32, so the epilogue only adds the
// convolution's init operand -- no widening step like the fp16 path's extf.
// It is a plain linalg.generic, not a flow.dispatch.workgroups, so that
// dispatch-region formation can fuse it with the requantization that follows
// (ISSUES.md P8).
// CHECK: arith.addi
// CHECK-NOT: arith.extf

// CHECK-LABEL: util.func public @depthwise_3x3_s1
// The HWC -> CHW filter transpose the depthwise weight packer requires. It is
// on a constant here, which is the point of inlining: in the caller const-eval
// hoists it into an initializer instead of running it every inference.
// CHECK: linalg.transpose {{.*}} permutation = [2, 0, 1]
// CHECK: flow.dispatch @rocket_dynamic_depthwise_int8_executable::

// CHECK-LABEL: util.func public @depthwise_3x3_s2
// CHECK: linalg.transpose {{.*}} permutation = [2, 0, 1]
// CHECK: flow.dispatch @rocket_dynamic_depthwise_int8_executable_s2::

// No quantized conv survives anywhere: every one was either folded and
// claimed, or would have broken the compile outright.
// CHECK-NOT: linalg.conv_2d_nchw_fchw_q
// CHECK-NOT: linalg.depthwise_conv_2d_nhwc_hwc_q

// The serialized variants must select the int8 accumulator mode with zero
// zero-points -- RocketTarget.cpp refuses any other value for this
// precision, and iree-rocket-hal only validated the zero-zero-point bypass
// on hardware. That constraint is exactly why the zero-point fold above has
// to happen before dispatch selection.
// TARGET: hal.executable.variant public @rocket_dynamic_conv2d_v1 target(<"rocket", "rocket-flatbuffer-v1"
// TARGET-SAME: input_zero_point = 0 : i32
// TARGET-SAME: precision = "int8_accumulator"
// TARGET-SAME: weights_zero_point = 0 : i32

module {
  func.func @dense_1x1(%in: tensor<1x48x14x14xi8>, %f: tensor<24x48x1x1xi8>, %izp: i32) -> tensor<1x24x14x14xi32> {
    %zero = arith.constant 0 : i32
    %empty = tensor.empty() : tensor<1x24x14x14xi32>
    %init = linalg.fill ins(%zero : i32) outs(%empty : tensor<1x24x14x14xi32>) -> tensor<1x24x14x14xi32>
    %0 = linalg.conv_2d_nchw_fchw_q {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%in, %f, %izp, %zero : tensor<1x48x14x14xi8>, tensor<24x48x1x1xi8>, i32, i32)
      outs(%init : tensor<1x24x14x14xi32>) -> tensor<1x24x14x14xi32>
    return %0 : tensor<1x24x14x14xi32>
  }

  func.func @depthwise_3x3_s1(%in: tensor<1x16x16x48xi8>, %f: tensor<3x3x48xi8>, %izp: i32) -> tensor<1x14x14x48xi32> {
    %zero = arith.constant 0 : i32
    %empty = tensor.empty() : tensor<1x14x14x48xi32>
    %init = linalg.fill ins(%zero : i32) outs(%empty : tensor<1x14x14x48xi32>) -> tensor<1x14x14x48xi32>
    %0 = linalg.depthwise_conv_2d_nhwc_hwc_q {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%in, %f, %izp, %zero : tensor<1x16x16x48xi8>, tensor<3x3x48xi8>, i32, i32)
      outs(%init : tensor<1x14x14x48xi32>) -> tensor<1x14x14x48xi32>
    return %0 : tensor<1x14x14x48xi32>
  }

  func.func @depthwise_3x3_s2(%in: tensor<1x29x29x48xi8>, %f: tensor<3x3x48xi8>, %izp: i32) -> tensor<1x14x14x48xi32> {
    %zero = arith.constant 0 : i32
    %empty = tensor.empty() : tensor<1x14x14x48xi32>
    %init = linalg.fill ins(%zero : i32) outs(%empty : tensor<1x14x14x48xi32>) -> tensor<1x14x14x48xi32>
    %0 = linalg.depthwise_conv_2d_nhwc_hwc_q {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
      ins(%in, %f, %izp, %zero : tensor<1x29x29x48xi8>, tensor<3x3x48xi8>, i32, i32)
      outs(%init : tensor<1x14x14x48xi32>) -> tensor<1x14x14x48xi32>
    return %0 : tensor<1x14x14x48xi32>
  }
}
