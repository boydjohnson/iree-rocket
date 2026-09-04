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
// RUN:   -o - | FileCheck %s --check-prefix=ORIGINAL
// RUN: iree-compile %s \
// RUN:   --iree-preprocessing-transform-spec-filename=%S/../target/Rocket/rocket_conv2d_transform_spec.mlir \
// RUN:   --iree-hal-target-device=rocket_device=rocket \
// RUN:   --iree-hal-target-device=cpu_device=local \
// RUN:   --iree-hal-local-target-device-backends=llvm-cpu \
// RUN:   --iree-llvmcpu-target-cpu=generic \
// RUN:   --iree-hal-default-device=cpu_device \
// RUN:   --iree-hal-indirect-command-buffers=false \
// RUN:   --compile-to=executable-targets \
// RUN:   --mlir-print-op-generic=false \
// RUN:   -o - | iree-opt --pass-pipeline="builtin.module(rocket-annotate-final-placement)" \
// RUN:   | FileCheck %s --check-prefix=FINAL

// RocketAnnotateOriginalPlacementPass runs right before the transform spec's
// match/rewrite loop (see the "rocket-annotate-original-placement" call in
// @__transform_main). A matched op is erased and replaced by a Rocket
// dispatch, taking its rocket.origin/rocket.origin_kind tags with it; an
// unmatched op keeps them, so a --compile-to=preprocessing dump shows
// exactly which conv-shaped ops fell through to CPU and why (shape, or a
// deliberately disabled matcher -- see @stride2_dense_conv_disabled_matcher
// below).

// ORIGINAL-LABEL: util.func public @matched_dynamic_conv
// ORIGINAL-NOT: rocket.origin
// ORIGINAL: util.call @call_rocket_dynamic_conv2d
util.func public @matched_dynamic_conv(
    %input: tensor<1x?x?x32xf16>,
    %filter: tensor<1x1x32x16xf16>,
    %init: tensor<1x?x?x16xf32>) -> tensor<1x?x?x16xf32> {
  %result = linalg.conv_2d_nhwc_hwcf {
      dilations = dense<1> : vector<2xi64>,
      strides = dense<1> : vector<2xi64>
    } ins(%input, %filter
      : tensor<1x?x?x32xf16>, tensor<1x1x32x16xf16>)
      outs(%init : tensor<1x?x?x16xf32>)
      -> tensor<1x?x?x16xf32>
  util.return %result : tensor<1x?x?x16xf32>
}

// A 5x5 kernel is never claimed by any matcher (see rocket_conv_layout.mlir's
// @dynamic_5x5_nhwc_conv) -- rocket.origin_kind records the structural shape
// regardless of why it didn't match.

// ORIGINAL-LABEL: util.func public @unmatched_5x5_conv
// ORIGINAL: linalg.conv_2d_nhwc_hwcf
// ORIGINAL-SAME: rocket.origin = "candidate"
// ORIGINAL-SAME: rocket.origin_kind = "dense_conv2d_5x5_s1"
util.func public @unmatched_5x5_conv(
    %input: tensor<1x?x?x32xf16>,
    %filter: tensor<5x5x32x16xf16>,
    %init: tensor<1x?x?x16xf32>) -> tensor<1x?x?x16xf32> {
  %result = linalg.conv_2d_nhwc_hwcf {
      dilations = dense<1> : vector<2xi64>,
      strides = dense<1> : vector<2xi64>
    } ins(%input, %filter
      : tensor<1x?x?x32xf16>, tensor<5x5x32x16xf16>)
      outs(%init : tensor<1x?x?x16xf32>)
      -> tensor<1x?x?x16xf32>
  util.return %result : tensor<1x?x?x16xf32>
}

// Stride 2 is claimed, through its own matcher and its own executable. This
// case used to assert the opposite: @match_dynamic_conv2d_s2 was written but
// deliberately left out of @__transform_main's foreach_match, and this
// function was named for that. It has since been wired in, and the stride-3
// case below is what carries the disabled-matcher story now.

// ORIGINAL-LABEL: util.func public @matched_stride2_dense_conv
// ORIGINAL-NOT: rocket.origin
// ORIGINAL: util.call @call_rocket_dynamic_conv2d_s2
util.func public @matched_stride2_dense_conv(
    %input: tensor<1x?x?x32xf16>,
    %filter: tensor<1x1x32x16xf16>,
    %init: tensor<1x?x?x16xf32>) -> tensor<1x?x?x16xf32> {
  %result = linalg.conv_2d_nhwc_hwcf {
      dilations = dense<1> : vector<2xi64>,
      strides = dense<2> : vector<2xi64>
    } ins(%input, %filter
      : tensor<1x?x?x32xf16>, tensor<1x1x32x16xf16>)
      outs(%init : tensor<1x?x?x16xf32>)
      -> tensor<1x?x?x16xf32>
  util.return %result : tensor<1x?x?x16xf32>
}

// @match_dynamic_conv2d_s3 is structurally correct for this exact shape but
// deliberately not wired into @__transform_main's foreach_match, along with
// @match_dynamic_conv2d_s4 and both of their 3x3 counterparts.
// rocket.origin_kind makes that audit-able straight from the IR: this op is
// tagged as a Rocket-shaped candidate, and --stays-- on CPU.
//
// This is the check that catches a matcher being enabled without anyone
// updating the placement expectations -- which is exactly what happened to
// the stride-2 case above.

// ORIGINAL-LABEL: util.func public @stride3_dense_conv_disabled_matcher
// ORIGINAL: linalg.conv_2d_nhwc_hwcf
// ORIGINAL-SAME: rocket.origin = "candidate"
// ORIGINAL-SAME: rocket.origin_kind = "dense_conv2d_1x1_s3"
util.func public @stride3_dense_conv_disabled_matcher(
    %input: tensor<1x?x?x32xf16>,
    %filter: tensor<1x1x32x16xf16>,
    %init: tensor<1x?x?x16xf32>) -> tensor<1x?x?x16xf32> {
  %result = linalg.conv_2d_nhwc_hwcf {
      dilations = dense<1> : vector<2xi64>,
      strides = dense<3> : vector<2xi64>
    } ins(%input, %filter
      : tensor<1x?x?x32xf16>, tensor<1x1x32x16xf16>)
      outs(%init : tensor<1x?x?x16xf32>)
      -> tensor<1x?x?x16xf32>
  util.return %result : tensor<1x?x?x16xf32>
}

// RocketAnnotateFinalPlacementPass runs separately, over the
// --compile-to=executable-targets dump (one phase before --compile-to=hal,
// which already serializes each hal.executable.variant into an opaque
// hal.executable.binary and discards #hal.executable.target along with it).
// It reads the resolved backend straight off each variant: the two matched
// ops above produced Rocket executables; every CPU fallback -- the unmatched
// ops, and the elementwise accumulate dispatch each Rocket call still needs
// on the CPU side -- targets "llvm-cpu" and is tagged "cpu".
//
// There is one accumulate dispatch rather than two: the stride-1 and
// stride-2 calls need the identical f16-to-f32 body, and IREE deduplicates
// identical executables, so it is named after whichever call survived.

// FINAL: hal.executable private @call_rocket_dynamic_conv2d_s2_dispatch_0 attributes {rocket.final = "cpu"}
// FINAL: hal.executable.variant public @embedded_elf_x86_64
// FINAL-SAME: attributes {rocket.final = "cpu"}

// FINAL: hal.executable private @rocket_dynamic_executable_s2 attributes {rocket.final = "rocket"}
// FINAL: hal.executable.variant public @rocket_dynamic_conv2d_v1
// FINAL-SAME: attributes {rocket.final = "rocket"}

// FINAL: hal.executable private @rocket_dynamic_executable attributes {rocket.final = "rocket"}
// FINAL: hal.executable.variant public @rocket_dynamic_conv2d_v1
// FINAL-SAME: attributes {rocket.final = "rocket"}

// FINAL: hal.executable private @unmatched_5x5_conv_dispatch_0 attributes {rocket.final = "cpu"}
// FINAL: hal.executable.variant public @embedded_elf_x86_64
// FINAL-SAME: attributes {rocket.final = "cpu"}

// FINAL: hal.executable private @stride3_dense_conv_disabled_matcher_dispatch_0 attributes {rocket.final = "cpu"}
// FINAL: hal.executable.variant public @embedded_elf_x86_64
// FINAL-SAME: attributes {rocket.final = "cpu"}
