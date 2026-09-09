// RUN: iree-opt %s --pass-pipeline='builtin.module(util.func(rocket-plan-candidates))' 2>/dev/null \
// RUN:   | FileCheck %s
// RUN: iree-opt %s --pass-pipeline='builtin.module(util.func(rocket-plan-candidates))' 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=REMARK

// The shared planner's verdict on each candidate, recorded on the function
// (first RUN) and, for refusals and deferrals, as a remark on the op
// (second RUN; diagnostics precede the module, so they are checked apart).
// The shapes sit on the matcher boundaries so this doubles as the parity
// corpus for COMPILER_ROADMAP.md section 2: a shape the matchers claim must
// be one the planner accepts, and here it is.
//
// Each record also carries what section 3's report prints: the precision
// rung the decision was reached on, a `limit` class saying whether a refusal
// is the shape, the semantics, a hardware bound or an unmeasured
// configuration, and the hardware-job and column counts of an accepted plan
// -- one dispatch site, `jobs` standalone hardware jobs.

// The channel ceiling itself is accepted; at 7x7 the coefficient working
// set splits the CBUF 5/7 and the rows tile.
// CHECK-LABEL: util.func public @pointwise_at_the_channel_ceiling
// CHECK-SAME: rocket.plan_decisions = [{columns = 1 : i64, decision = "tiled", detail = "cbuf 5/7, tiles {{[0-9]+}}, columns 1", jobs = {{[0-9]+}} : i64, kind = "dense_conv2d", layout = "nhwc", limit = "none", loc = #loc{{[0-9]*}}, precision = "fp16", shape = "7x7 Cin 3584 Cout 3584 k1x1 s1", status = "ok"}]
util.func public @pointwise_at_the_channel_ceiling(
    %input: tensor<1x7x7x3584xf16>,
    %filter: tensor<1x1x3584x3584xf16>,
    %init: tensor<1x7x7x3584xf32>) -> tensor<1x7x7x3584xf32> {
  %0 = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x7x7x3584xf16>, tensor<1x1x3584x3584xf16>)
      outs(%init : tensor<1x7x7x3584xf32>) -> tensor<1x7x7x3584xf32>
  util.return %0 : tensor<1x7x7x3584xf32>
}

// A small pointwise conv is one job.
// CHECK-LABEL: util.func public @pointwise_is_direct
// CHECK-SAME: decision = "direct", detail = "cbuf {{[0-9]+}}/{{[0-9]+}}"
// CHECK-SAME: jobs = 1 : i64
// CHECK-SAME: limit = "none"
// CHECK-SAME: precision = "fp16", shape = "14x14 Cin 88 Cout 528 k1x1 s1"
util.func public @pointwise_is_direct(
    %input: tensor<1x14x14x88xf16>,
    %filter: tensor<1x1x88x528xf16>,
    %init: tensor<1x14x14x528xf32>) -> tensor<1x14x14x528xf32> {
  %0 = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x14x14x88xf16>, tensor<1x1x88x528xf16>)
      outs(%init : tensor<1x14x14x528xf32>) -> tensor<1x14x14x528xf32>
  util.return %0 : tensor<1x14x14x528xf32>
}

// One past the ceiling: register-representable, no capture backing. The
// matchers' dim_bounds decline it too; that agreement is the point.
// REMARK: remark: rocket-plan: cpu [unvalidated_configuration] dense_conv2d 7x7 Cin 3585 Cout 64 k1x1 s1: input channels must be 1..=3584
// CHECK-LABEL: util.func public @pointwise_past_the_channel_ceiling
// CHECK-SAME: decision = "cpu"
// CHECK-SAME: limit = "validation"
// CHECK-SAME: status = "unvalidated_configuration"
util.func public @pointwise_past_the_channel_ceiling(
    %input: tensor<1x7x7x3585xf16>,
    %filter: tensor<1x1x3585x64xf16>,
    %init: tensor<1x7x7x64xf32>) -> tensor<1x7x7x64xf32> {
  %0 = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x7x7x3585xf16>, tensor<1x1x3585x64xf16>)
      outs(%init : tensor<1x7x7x64xf32>) -> tensor<1x7x7x64xf32>
  util.return %0 : tensor<1x7x7x64xf32>
}

// A stride-2 3x3 over an explicitly padded 225x225 stem input: several row
// tiles, so the decision is "tiled" with the split spelled out.
// CHECK-LABEL: util.func public @strided_stem_is_tiled
// CHECK-SAME: decision = "tiled", detail = "cbuf {{[0-9]+}}/{{[0-9]+}}, tiles {{[0-9]+}}, columns 1"
// CHECK-SAME: shape = "225x225 Cin 3 Cout 32 k3x3 s2"
util.func public @strided_stem_is_tiled(
    %input: tensor<1x225x225x3xf16>,
    %filter: tensor<3x3x3x32xf16>,
    %init: tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32> {
  %0 = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
      ins(%input, %filter : tensor<1x225x225x3xf16>, tensor<3x3x3x32xf16>)
      outs(%init : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
  util.return %0 : tensor<1x112x112x32xf32>
}

// The stride-dropping bug's signature: a 225x225 input whose op claims a
// 112x112 output at stride 1. linalg's verifier only checks the input is
// large enough, so it passes; the planner derives 223x223 and disputes it.
// REMARK: remark: rocket-plan: cpu [invalid_shape] dense_conv2d 225x225 Cin 32 Cout 32 k3x3 s1: output extent 112x112 disagrees with the 223x223 the planner derives
// CHECK-LABEL: util.func public @output_extent_disagrees
// CHECK-SAME: decision = "cpu"
// CHECK-SAME: limit = "shape"
// CHECK-SAME: status = "invalid_shape"
util.func public @output_extent_disagrees(
    %input: tensor<1x225x225x32xf16>,
    %filter: tensor<3x3x32x32xf16>,
    %init: tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32> {
  %0 = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x225x225x32xf16>, tensor<3x3x32x32xf16>)
      outs(%init : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
  util.return %0 : tensor<1x112x112x32xf32>
}

// A symbolic extent is the runtime's to plan.
// REMARK: remark: rocket-plan: deferred [dynamic] dense_conv2d
// CHECK-LABEL: util.func public @dynamic_is_deferred
// CHECK-SAME: decision = "deferred"
util.func public @dynamic_is_deferred(
    %input: tensor<1x?x?x3xf16>,
    %filter: tensor<3x3x3x16xf16>,
    %init: tensor<1x?x?x16xf32>) -> tensor<1x?x?x16xf32> {
  %0 = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
      ins(%input, %filter : tensor<1x?x?x3xf16>, tensor<3x3x3x16xf16>)
      outs(%init : tensor<1x?x?x16xf32>) -> tensor<1x?x?x16xf32>
  util.return %0 : tensor<1x?x?x16xf32>
}

// An f32 op the demotion left alone is a form problem, not a planner one.
// REMARK: remark: rocket-plan: cpu [form] depthwise_conv2d 113x113 Cin 96 Cout 96 k3x3 s2: f32 operands
// CHECK-LABEL: util.func public @f32_depthwise_is_a_form_problem
// CHECK-SAME: decision = "cpu"
// CHECK-SAME: limit = "semantics"
// A form problem is caught before any rung is named, so there is none to
// record.
// CHECK-SAME: precision = ""
// CHECK-SAME: status = "form"
util.func public @f32_depthwise_is_a_form_problem(
    %input: tensor<1x96x113x113xf32>,
    %filter: tensor<96x3x3xf32>,
    %init: tensor<1x96x56x56xf32>) -> tensor<1x96x56x56xf32> {
  %0 = linalg.depthwise_conv_2d_nchw_chw
      {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
      ins(%input, %filter : tensor<1x96x113x113xf32>, tensor<96x3x3xf32>)
      outs(%init : tensor<1x96x56x56xf32>) -> tensor<1x96x56x56xf32>
  util.return %0 : tensor<1x96x56x56xf32>
}

// An f16 depthwise is planned like any other convolution.
// CHECK-LABEL: util.func public @f16_depthwise_is_planned
// CHECK-SAME: decision = "{{direct|tiled}}"
// CHECK-SAME: kind = "depthwise_conv2d"
// CHECK-SAME: shape = "113x113 Cin 96 Cout 96 k3x3 s2"
util.func public @f16_depthwise_is_planned(
    %input: tensor<1x96x113x113xf16>,
    %filter: tensor<96x3x3xf16>,
    %init: tensor<1x96x56x56xf32>) -> tensor<1x96x56x56xf32> {
  %0 = linalg.depthwise_conv_2d_nchw_chw
      {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
      ins(%input, %filter : tensor<1x96x113x113xf16>, tensor<96x3x3xf16>)
      outs(%init : tensor<1x96x56x56xf32>) -> tensor<1x96x56x56xf32>
  util.return %0 : tensor<1x96x56x56xf32>
}

// ViT's projection: M past the sweep's 32 splits into column tiles under
// the 11-bit slab-base bound (ISSUES.md C10), which the planner reports.
// CHECK-LABEL: util.func public @vit_projection_matmul
// CHECK-SAME: decision = "tiled", detail = "cbuf {{[0-9]+}}/{{[0-9]+}}, tiles {{[0-9]+}}, columns {{[2-9]}}"
// CHECK-SAME: kind = "matmul", layout = "row_major"
// CHECK-SAME: shape = "197x768 x 768x768"
util.func public @vit_projection_matmul(
    %lhs: tensor<197x768xf16>,
    %rhs: tensor<768x768xf16>,
    %init: tensor<197x768xf32>) -> tensor<197x768xf32> {
  %0 = linalg.matmul
      ins(%lhs, %rhs : tensor<197x768xf16>, tensor<768x768xf16>)
      outs(%init : tensor<197x768xf32>) -> tensor<197x768xf32>
  util.return %0 : tensor<197x768xf32>
}

// A transposed matmul is outside the FC lowering's form.
// REMARK: remark: rocket-plan: cpu [form] matmul {{.*}}: user-defined indexing maps
// CHECK-LABEL: util.func public @transposed_matmul_is_a_form_problem
// CHECK-SAME: status = "form"
util.func public @transposed_matmul_is_a_form_problem(
    %lhs: tensor<1x1792xf16>,
    %rhs: tensor<1001x1792xf16>,
    %init: tensor<1x1001xf32>) -> tensor<1x1001xf32> {
  %0 = linalg.matmul
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>]
      ins(%lhs, %rhs : tensor<1x1792xf16>, tensor<1001x1792xf16>)
      outs(%init : tensor<1x1001xf32>) -> tensor<1x1001xf32>
  util.return %0 : tensor<1x1001xf32>
}

// The gap section 3 closed. `ConvPlan` plans an fp16 depthwise convolution
// to Cin 1792, but the admission envelope stops at 1536 -- the extent the
// compiled end-to-end gate proves -- so every matcher will decline this and
// it stays on the CPU. Recorded as "direct" before the envelope was asked
// here, which is how six of MobileNetV2's candidates read as offloaded
// while the module held no dispatch for them.
//
// It must NOT be tagged `rocket.plan_refused`: the tag makes *every* matcher
// decline, and the envelope is indexed by the rung the matcher claims for,
// which this pass can only guess at. Only planner and form refusals tag.
// REMARK: remark: rocket-plan: cpu [unvalidated_configuration] depthwise_conv2d 7x7 Cin 1792 Cout 1792 k3x3 s1
// CHECK-LABEL: util.func public @depthwise_past_the_measured_envelope
// CHECK-SAME: decision = "cpu"
// CHECK-SAME: limit = "validation"
// CHECK-SAME: status = "unvalidated_configuration"
// CHECK-NOT: rocket.plan_refused
util.func public @depthwise_past_the_measured_envelope(
    %input: tensor<1x1792x7x7xf16>,
    %filter: tensor<1792x3x3xf16>,
    %init: tensor<1x1792x5x5xf32>) -> tensor<1x1792x5x5xf32> {
  %0 = linalg.depthwise_conv_2d_nchw_chw
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x1792x7x7xf16>, tensor<1792x3x3xf16>)
      outs(%init : tensor<1x1792x5x5xf32>) -> tensor<1x1792x5x5xf32>
  util.return %0 : tensor<1x1792x5x5xf32>
}

// An `i8 x i8 -> i32` convolution is on whichever int8 rung the matcher that
// claims it says, and the two rungs have separate envelopes: at 3x3 the
// accumulator stops at Cout 512 and the requantized path reaches 768. The
// operand types name the accumulator, so asking only that rung would report
// this as CPU. Both are tried, and the rung that admits is the one recorded.
// CHECK-LABEL: util.func public @int8_admitted_only_on_the_requant_rung
// CHECK-SAME: decision = "{{direct|tiled}}"
// CHECK-SAME: precision = "int8_requant"
// CHECK-SAME: shape = "14x14 Cin 512 Cout 640 k3x3 s1"
// CHECK-SAME: status = "ok"
util.func public @int8_admitted_only_on_the_requant_rung(
    %input: tensor<1x14x14x512xi8>,
    %filter: tensor<3x3x512x640xi8>,
    %init: tensor<1x12x12x640xi32>) -> tensor<1x12x12x640xi32> {
  %0 = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x14x14x512xi8>, tensor<3x3x512x640xi8>)
      outs(%init : tensor<1x12x12x640xi32>) -> tensor<1x12x12x640xi32>
  util.return %0 : tensor<1x12x12x640xi32>
}

// A fill beside the candidates has one input and must simply be skipped.
// CHECK-LABEL: util.func public @fill_is_not_a_candidate
// CHECK-NOT: rocket.plan_decisions
util.func public @fill_is_not_a_candidate(%init: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %zero = arith.constant 0.0 : f32
  %0 = linalg.fill ins(%zero : f32) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  util.return %0 : tensor<4x4xf32>
}
