// Rocket conv2d dispatch, generic over runtime shape. Previously this file
// also carried three fixed-shape per-MobileNet-layer executables
// (rocket_target_0/1/2, rocket_executable_0/1/2, @match_conv2d_0/1/2) as a
// literal-shape fast path ahead of the dynamic fallback below. They are
// retired: the dynamic path (@match_dynamic_conv2d/_3x3 and their depthwise
// counterparts) claims the same shapes through the identical
// runtime-dimension ABI, so the fixed-shape specializations added a second
// code path without adding coverage. Developed and hardware-validated
// against the RK3588 in iree-rocket-design-spike (see that repo's
// DESIGN_NOTES.md for the full derivation, including the two
// rocket-hal-driver bugs the depthwise path exposed: a skipped weight-
// packing branch and a tap-major layout formula only ever checked inside
// one 32-channel coefficient group).

#rocket_dynamic_target = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "conv2d",
  input_width = 0 : i32, input_height = 0 : i32, input_channels = 0 : i32,
  output_width = 0 : i32, output_height = 0 : i32, output_channels = 0 : i32,
  weights_width = 0 : i32, weights_height = 0 : i32, stride = 1 : i32,
  depthwise = false,
  input_zero_point = 0 : i32, output_zero_point = 0 : i32, weights_zero_point = 0 : i32,
  input_scale = 1.0 : f32, weights_scale = 1.0 : f32, output_scale = 1.0 : f32,
  truncate_bits = 0 : i32,
  activation = "none", activation_cmp = 0 : i32,
  precision = "fp16",
  // The six settable dimensions. output_width/output_height are absent
  // because the runtime always derives them from these plus stride and
  // padding -- their wire values are retired, see
  // rocket-schema/schema/rocket_executable_def.fbs.
  runtime_dimensions = [
    "input_width", "input_height", "input_channels",
    "output_channels", "weights_width", "weights_height"
  ]
}>

// Same wire format as #rocket_dynamic_target with depthwise = true. Kept as
// a separate attr (rather than a runtime-settable bool) because it gates a
// different register program at the driver (CNA_CONV_CON1.CONV_MODE=3,
// CORE_MISC_CFG.DW_EN=1, etc. -- see DESIGN_NOTES.md "Depthwise: Mesa's
// channel rule is wrong") and a different weight layout
// (tensor_layout::pack_depthwise_to_rocket_weights, tap-major with a
// padded-channel stride, not pack_hwcf_to_rocket_weights). output_channels
// is still carried on the wire even though the depthwise matchers below only
// ever bind it equal to input_channels -- Conv2DDef's depthwise field
// (rocket-schema) already exists for this and the driver
// (executable_cache.rs) already reads it, so this is genuinely just a
// dispatch-selection gap, not a runtime one.
#rocket_dynamic_depthwise_target = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "conv2d",
  input_width = 0 : i32, input_height = 0 : i32, input_channels = 0 : i32,
  output_width = 0 : i32, output_height = 0 : i32, output_channels = 0 : i32,
  weights_width = 0 : i32, weights_height = 0 : i32, stride = 1 : i32,
  depthwise = true,
  input_zero_point = 0 : i32, output_zero_point = 0 : i32, weights_zero_point = 0 : i32,
  input_scale = 1.0 : f32, weights_scale = 1.0 : f32, output_scale = 1.0 : f32,
  truncate_bits = 0 : i32,
  activation = "none", activation_cmp = 0 : i32,
  precision = "fp16",
  runtime_dimensions = [
    "input_width", "input_height", "input_channels",
    "output_channels", "weights_width", "weights_height"
  ]
}>

#rocket_dynamic_target_s2 = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "conv2d",
  input_width = 0 : i32, input_height = 0 : i32, input_channels = 0 : i32,
  output_width = 0 : i32, output_height = 0 : i32, output_channels = 0 : i32,
  weights_width = 0 : i32, weights_height = 0 : i32, stride = 2 : i32,
  depthwise = false,
  input_zero_point = 0 : i32, output_zero_point = 0 : i32, weights_zero_point = 0 : i32,
  input_scale = 1.0 : f32, weights_scale = 1.0 : f32, output_scale = 1.0 : f32,
  truncate_bits = 0 : i32,
  activation = "none", activation_cmp = 0 : i32,
  precision = "fp16",
  runtime_dimensions = [
    "input_width", "input_height", "input_channels",
    "output_channels", "weights_width", "weights_height"
  ]
}>

#rocket_dynamic_target_s3 = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "conv2d",
  input_width = 0 : i32, input_height = 0 : i32, input_channels = 0 : i32,
  output_width = 0 : i32, output_height = 0 : i32, output_channels = 0 : i32,
  weights_width = 0 : i32, weights_height = 0 : i32, stride = 3 : i32,
  depthwise = false,
  input_zero_point = 0 : i32, output_zero_point = 0 : i32, weights_zero_point = 0 : i32,
  input_scale = 1.0 : f32, weights_scale = 1.0 : f32, output_scale = 1.0 : f32,
  truncate_bits = 0 : i32,
  activation = "none", activation_cmp = 0 : i32,
  precision = "fp16",
  runtime_dimensions = [
    "input_width", "input_height", "input_channels",
    "output_channels", "weights_width", "weights_height"
  ]
}>

#rocket_dynamic_target_s4 = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "conv2d",
  input_width = 0 : i32, input_height = 0 : i32, input_channels = 0 : i32,
  output_width = 0 : i32, output_height = 0 : i32, output_channels = 0 : i32,
  weights_width = 0 : i32, weights_height = 0 : i32, stride = 4 : i32,
  depthwise = false,
  input_zero_point = 0 : i32, output_zero_point = 0 : i32, weights_zero_point = 0 : i32,
  input_scale = 1.0 : f32, weights_scale = 1.0 : f32, output_scale = 1.0 : f32,
  truncate_bits = 0 : i32,
  activation = "none", activation_cmp = 0 : i32,
  precision = "fp16",
  runtime_dimensions = [
    "input_width", "input_height", "input_channels",
    "output_channels", "weights_width", "weights_height"
  ]
}>

#rocket_dynamic_depthwise_target_s2 = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "conv2d",
  input_width = 0 : i32, input_height = 0 : i32, input_channels = 0 : i32,
  output_width = 0 : i32, output_height = 0 : i32, output_channels = 0 : i32,
  weights_width = 0 : i32, weights_height = 0 : i32, stride = 2 : i32,
  depthwise = true,
  input_zero_point = 0 : i32, output_zero_point = 0 : i32, weights_zero_point = 0 : i32,
  input_scale = 1.0 : f32, weights_scale = 1.0 : f32, output_scale = 1.0 : f32,
  truncate_bits = 0 : i32,
  activation = "none", activation_cmp = 0 : i32,
  precision = "fp16",
  runtime_dimensions = [
    "input_width", "input_height", "input_channels",
    "output_channels", "weights_width", "weights_height"
  ]
}>

#rocket_dynamic_depthwise_target_s3 = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "conv2d",
  input_width = 0 : i32, input_height = 0 : i32, input_channels = 0 : i32,
  output_width = 0 : i32, output_height = 0 : i32, output_channels = 0 : i32,
  weights_width = 0 : i32, weights_height = 0 : i32, stride = 3 : i32,
  depthwise = true,
  input_zero_point = 0 : i32, output_zero_point = 0 : i32, weights_zero_point = 0 : i32,
  input_scale = 1.0 : f32, weights_scale = 1.0 : f32, output_scale = 1.0 : f32,
  truncate_bits = 0 : i32,
  activation = "none", activation_cmp = 0 : i32,
  precision = "fp16",
  runtime_dimensions = [
    "input_width", "input_height", "input_channels",
    "output_channels", "weights_width", "weights_height"
  ]
}>

#rocket_dynamic_depthwise_target_s4 = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "conv2d",
  input_width = 0 : i32, input_height = 0 : i32, input_channels = 0 : i32,
  output_width = 0 : i32, output_height = 0 : i32, output_channels = 0 : i32,
  weights_width = 0 : i32, weights_height = 0 : i32, stride = 4 : i32,
  depthwise = true,
  input_zero_point = 0 : i32, output_zero_point = 0 : i32, weights_zero_point = 0 : i32,
  input_scale = 1.0 : f32, weights_scale = 1.0 : f32, output_scale = 1.0 : f32,
  truncate_bits = 0 : i32,
  activation = "none", activation_cmp = 0 : i32,
  precision = "fp16",
  runtime_dimensions = [
    "input_width", "input_height", "input_channels",
    "output_channels", "weights_width", "weights_height"
  ]
}>


// int8 counterparts of the fp16 targets above, for ONNX models quantized
// with ORT's quantize_dynamic (onnx.ConvInteger). Everything is identical
// apart from `precision`, which selects the DPU's int8-in/int32-out
// accumulator mode: the requantization stage (BS, CPEND, and the
// out-convert scale/shift/offset) is bypassed and the raw i32 accumulator
// is written straight out -- see iree-rocket-hal's
// int8_accumulator_output_uses_the_hardware_validated_bypasses.
//
// The zero points stay 0 here, and that is a hard requirement, not a
// placeholder: Shape::with_precision panics on a non-zero zero point in
// this mode, and RocketTarget.cpp rejects it at serialization time, because
// only the zero-zero-point bypass path is hardware-validated. Real ONNX
// activations are asymmetric (DynamicQuantizeLinear emits a non-zero ui8
// zero point), so getting here at all depends on
// iree-global-opt-quantized-conv-to-conv having already folded the zero
// point out of the convolution and into a separate CPU-side correction
// term -- an exact i32 identity, not an approximation. See
// @__transform_main's own comment.
//
// The scales stay 1.0 for the same reason they do in the fp16 targets: this
// mode does no rescaling at all, so the driver's bias normalization
// (pack_int8_bias_to_bs divides by input_scale * weights_scale) is the
// identity and the zero bias below passes through untouched.

#rocket_dynamic_int8_target = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "conv2d",
  input_width = 0 : i32, input_height = 0 : i32, input_channels = 0 : i32,
  output_width = 0 : i32, output_height = 0 : i32, output_channels = 0 : i32,
  weights_width = 0 : i32, weights_height = 0 : i32, stride = 1 : i32,
  depthwise = false,
  input_zero_point = 0 : i32, output_zero_point = 0 : i32, weights_zero_point = 0 : i32,
  input_scale = 1.0 : f32, weights_scale = 1.0 : f32, output_scale = 1.0 : f32,
  truncate_bits = 0 : i32,
  activation = "none", activation_cmp = 0 : i32,
  precision = "int8_accumulator",
  runtime_dimensions = [
    "input_width", "input_height", "input_channels",
    "output_channels", "weights_width", "weights_height"
  ]
}>

#rocket_dynamic_depthwise_int8_target = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "conv2d",
  input_width = 0 : i32, input_height = 0 : i32, input_channels = 0 : i32,
  output_width = 0 : i32, output_height = 0 : i32, output_channels = 0 : i32,
  weights_width = 0 : i32, weights_height = 0 : i32, stride = 1 : i32,
  depthwise = true,
  input_zero_point = 0 : i32, output_zero_point = 0 : i32, weights_zero_point = 0 : i32,
  input_scale = 1.0 : f32, weights_scale = 1.0 : f32, output_scale = 1.0 : f32,
  truncate_bits = 0 : i32,
  activation = "none", activation_cmp = 0 : i32,
  precision = "int8_accumulator",
  runtime_dimensions = [
    "input_width", "input_height", "input_channels",
    "output_channels", "weights_width", "weights_height"
  ]
}>

#rocket_dynamic_depthwise_int8_target_s2 = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "conv2d",
  input_width = 0 : i32, input_height = 0 : i32, input_channels = 0 : i32,
  output_width = 0 : i32, output_height = 0 : i32, output_channels = 0 : i32,
  weights_width = 0 : i32, weights_height = 0 : i32, stride = 2 : i32,
  depthwise = true,
  input_zero_point = 0 : i32, output_zero_point = 0 : i32, weights_zero_point = 0 : i32,
  input_scale = 1.0 : f32, weights_scale = 1.0 : f32, output_scale = 1.0 : f32,
  truncate_bits = 0 : i32,
  activation = "none", activation_cmp = 0 : i32,
  precision = "int8_accumulator",
  runtime_dimensions = [
    "input_width", "input_height", "input_channels",
    "output_channels", "weights_width", "weights_height"
  ]
}>

// The indexing maps of an untransposed row-major matmul: A[m,k], B[k,n],
// C[m,n]. `linalg.matmul` expresses a transpose or a broadcast by overriding
// these rather than by being a different op, so pinning them is what keeps
// @match_rocket_matmul from claiming an operand layout the lowering cannot
// pack.
#rocket_matmul_lhs = affine_map<(d0, d1, d2) -> (d0, d2)>
#rocket_matmul_rhs = affine_map<(d0, d1, d2) -> (d2, d1)>
#rocket_matmul_out = affine_map<(d0, d1, d2) -> (d0, d1)>

// The PPU pooling engine, driven per dispatch. MLIR's linalg dialect has no
// average pool: an ONNX AveragePool or GlobalAveragePool arrives as
// linalg.pooling_*_sum followed by a separate divide, and the PPU has no sum
// mode of its own (its average is a multiply by fp16(65536/k), which cannot
// encode a divisor of one). So the hardware computes the *average* and
// @call_rocket_pooling_avg_nchw multiplies it back up by kh*kw, leaving the
// model's own divide to do what it was already going to do. That is one
// elementwise pass over the pooled result -- 1792 values on MobileNetV2 --
// and it costs nothing to keep the matcher a single-op match rather than a
// two-op DAG.
//
// Stride and padding are baked: every measured model pools with stride 1 or
// with stride equal to the kernel, and the PPU's padding is a 3-bit field
// whose meaning depends on the method. Kernel extent, channels and the input
// extent arrive as push constants.
#rocket_pooling_avg_target = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "pooling",
  input_width = 0 : i32, input_height = 0 : i32, channels = 0 : i32,
  output_width = 0 : i32, output_height = 0 : i32,
  kernel_width = 0 : i32, kernel_height = 0 : i32,
  stride_x = 1 : i32, stride_y = 1 : i32,
  pad_left = 0 : i32, pad_top = 0 : i32, pad_right = 0 : i32, pad_bottom = 0 : i32,
  method = "avg",
  precision = "fp16",
  runtime_dimensions = ["input_width", "input_height", "channels",
                        "kernel_width", "kernel_height"]
}>

// Max pooling: the same PPU engine, a different reduction. Everything the
// average above has to correct for, this does not -- the hardware computes
// the maximum directly, so the shim only widens f16 back to f32 and folds in
// the accumulator initialiser, which linalg defines into the reduction
// itself (`O = max(O, I)`, confirmed by generalizing the named op).
//
// Padding stays baked at zero, and for max that is what makes the method
// safe rather than merely conventional. `PoolingMethod::pad_fill_value`
// (pooling.rs) has a measured pad fill for max only at fp16 (0xFC00), none
// at int8, and none for min at any precision; an *unpadded* pool never reads
// the field, which is why an unmeasured combination is still runnable there.
// Model-level padding arrives as a separate tensor.pad the CPU runs, so the
// executable never sees a padded window.
//
// Stride is baked, one executable per value, matching the convolution
// variants. Both exist here because -- unlike the average pool, where every
// measured model pools with stride 1 -- a max pool is overwhelmingly stride
// 2. Stride 2 is measured against the oracle in `pooling_oracle_hw.rs`.
// Nothing above 2 is, so there is deliberately no s3/s4 target.
#rocket_pooling_max_target = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "pooling",
  input_width = 0 : i32, input_height = 0 : i32, channels = 0 : i32,
  output_width = 0 : i32, output_height = 0 : i32,
  kernel_width = 0 : i32, kernel_height = 0 : i32,
  stride_x = 1 : i32, stride_y = 1 : i32,
  pad_left = 0 : i32, pad_top = 0 : i32, pad_right = 0 : i32, pad_bottom = 0 : i32,
  method = "max",
  precision = "fp16",
  runtime_dimensions = ["input_width", "input_height", "channels",
                        "kernel_width", "kernel_height"]
}>

#rocket_pooling_max_target_s2 = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "pooling",
  input_width = 0 : i32, input_height = 0 : i32, channels = 0 : i32,
  output_width = 0 : i32, output_height = 0 : i32,
  kernel_width = 0 : i32, kernel_height = 0 : i32,
  stride_x = 2 : i32, stride_y = 2 : i32,
  pad_left = 0 : i32, pad_top = 0 : i32, pad_right = 0 : i32, pad_bottom = 0 : i32,
  method = "max",
  precision = "fp16",
  runtime_dimensions = ["input_width", "input_height", "channels",
                        "kernel_width", "kernel_height"]
}>

// Min pooling: the third PPU reduction, and the one whose zero padding is
// not merely conventional but *required*. `PoolingMethod::pad_fill_value`
// has no measured identity for min at any precision -- unlike max, which has
// 0xFC00 at fp16 -- so `required_pad_fill` returns None for a padded min and
// the driver refuses the executable outright (executable_cache.rs). Baking
// all four pad fields to zero is what keeps that path unreachable, and the
// driver derives `padded` from these fields alone, so tiling a wide input
// cannot reintroduce it.
//
// NHWC only, and that is linalg's asymmetry rather than a choice here: the
// dialect defines pooling_nchw_max but no pooling_nchw_min, so there is no
// NCHW min op for a matcher to claim. ONNX has no MinPool operator at all,
// which is presumably why.
//
// `pooling_nhwc_min_unsigned` is also unclaimed: it is the unsigned-integer
// reduction, and this path is f32 in, fp16 on the hardware.
#rocket_pooling_min_target = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "pooling",
  input_width = 0 : i32, input_height = 0 : i32, channels = 0 : i32,
  output_width = 0 : i32, output_height = 0 : i32,
  kernel_width = 0 : i32, kernel_height = 0 : i32,
  stride_x = 1 : i32, stride_y = 1 : i32,
  pad_left = 0 : i32, pad_top = 0 : i32, pad_right = 0 : i32, pad_bottom = 0 : i32,
  method = "min",
  precision = "fp16",
  runtime_dimensions = ["input_width", "input_height", "channels",
                        "kernel_width", "kernel_height"]
}>

#rocket_pooling_min_target_s2 = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "pooling",
  input_width = 0 : i32, input_height = 0 : i32, channels = 0 : i32,
  output_width = 0 : i32, output_height = 0 : i32,
  kernel_width = 0 : i32, kernel_height = 0 : i32,
  stride_x = 2 : i32, stride_y = 2 : i32,
  pad_left = 0 : i32, pad_top = 0 : i32, pad_right = 0 : i32, pad_bottom = 0 : i32,
  method = "min",
  precision = "fp16",
  runtime_dimensions = ["input_width", "input_height", "channels",
                        "kernel_width", "kernel_height"]
}>

// The matmul engine. There is no matmul *hardware*: `fc::Shape` lowers
// [M,K] x [K,N] to a height-one 1x1 convolution, with M the convolution
// width, K the input channels and N the output channels -- a mapping
// established over 160 captured ONNX `Linear` models. What is new is that
// the wire format now names the operation the input dialect actually has.
// "Fully connected" is not an op in linalg, which is why FullyConnectedDef
// sat unused since the day it was written.
//
// M, K and N all arrive as push constants, so one executable serves every
// matmul shape inside the channel ceilings.
#rocket_matmul_target = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "matmul",
  m = 0 : i32, k = 0 : i32, n = 0 : i32,
  input_zero_point = 0 : i32, output_zero_point = 0 : i32, weights_zero_point = 0 : i32,
  input_scale = 1.0 : f32, weights_scale = 1.0 : f32, output_scale = 1.0 : f32,
  truncate_bits = 0 : i32,
  activation = "none", activation_cmp = 0 : i32,
  precision = "fp16",
  runtime_dimensions = ["m", "k", "n"]
}>

// Input, weights, bias, output -- the convolution binding convention, since
// that is what this lowers to. The bias is zero-filled by the caller.
#matmul_pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>

// A pool has no weights and no bias: input and output only.
// Two-tensor element-wise, one target per operator. Everything but `op` is
// identical, and all three extents arrive as push constants -- an
// element-wise op has no weights and no reduction, so geometry is the only
// thing that varies per dispatch.
//
// fp16 only. `EwAddShape` has an int8 branch but its EW_CVT_SCALE and
// OUT_CVT_SCALE ratio semantics are inferred rather than confirmed, so the
// wire format carries no precision field at all and there is nothing to
// state here.
#rocket_elementwise_add_target = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "elementwise_binary",
  width = 0 : i32, height = 0 : i32, channels = 0 : i32,
  op = "add",
  runtime_dimensions = ["width", "height", "channels"]
}>

#rocket_elementwise_sub_target = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "elementwise_binary",
  width = 0 : i32, height = 0 : i32, channels = 0 : i32,
  op = "sub",
  runtime_dimensions = ["width", "height", "channels"]
}>

#rocket_elementwise_mul_target = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "elementwise_binary",
  width = 0 : i32, height = 0 : i32, channels = 0 : i32,
  op = "mul",
  runtime_dimensions = ["width", "height", "channels"]
}>

// Three push constants (width, height, channels) and three bindings: the
// primary operand, the second tensor, and the output. Every other Rocket
// kernel has two bindings; this is the only one that reads two tensors.
#elementwise_binary_pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>

#pooling_pipeline_layout = #hal.pipeline.layout<constants = 5, bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>

// Requantized int8: the DPU's own BS/CPEND/out-convert stages run, so the
// dispatch returns quantized i8 and no CPU epilogue requantizes behind it.
//
// This is the path that lifts the dense channel caps. The int8_accumulator
// target above is exact only while a convolution stays under 384 coefficient
// bytes per output channel -- past that the DPU writes one pixel stripe of
// the tile and stops, measured on RK3588 2026-09-03 and independent of
// tiling, Cout, geometry and the CBUF split, so there is nothing left to
// program around. Plain requantized int8 has no such limit and is
// hardware-exact at every Cin measured through 512 at both 1x1 and 3x3.
//
// Fused ReLU6, the fp16 activation MobileNetV2 puts after 18 of its
// offloaded convolutions.
//
// `activation_cmp` is the f32 bit pattern of the ceiling -- 6.0 is
// `0x40C00000` -- because the fp16 accumulator is float and the BN stage
// compares against it in the accumulator's own units, before `OUT_CVT`
// narrows. That is `Activation::clamped_fp16` in the HAL, confirmed there at
// three ceilings.
//
// **The ceiling is static here, and `rocket-fuse-conv-relu6` is what makes
// that sound.** A transform matcher matches structure, not constant values,
// so nothing below can tell a ceiling of 6.0 from any other. What guarantees
// this attribute is right is that the pass only ever produces the canonical
// form this target's matchers claim when the ceiling is exactly 6.0. Widening
// the pass without making `activation_cmp` a runtime push constant --
// `runtime_quantization` is the mechanism, it already carries `output_scale`
// -- would compile every other ceiling as 6.0.
//
// The bias is a real binding here rather than the zero fill every other fp16
// conv target passes, and it has to be: the hardware order is accumulate ->
// BS (bias) -> BN (activation) -> OUT_CVT, so the clamp sees the biased
// value only when the bias is on the BS plane. Measured in
// `conv_fp16_bias_activation_hw` -- bias alone, bias + ReLU and bias + ReLU6
// are each exact over 1024 outputs with `acc + bias` crossing both ends of
// the range.
// Convolution with the model's "same" padding done by the CNA, stride 1.
//
// A model's padding arrives as an explicit `tensor.pad` and IREE forms it as
// its own dispatch -- `slow_memcpy` in an `audit` listing. On ResNet50 fp16
// that is 16 dispatch sites feeding offloaded convolutions, ~12 MB per
// inference to copy a tensor into one two rows and columns larger. The CNA
// pads for free: `CNA_PAD_CON0` costs no cycles and no DMA.
//
// **`pad = 1` is baked, and it is the only value worth baking.** The
// hardware's pad fields are 4 bits, but the matchers stop at 3x3, and 3x3
// "same" padding is pad 1; pad 2 and 3 belong to 5x5 and 7x7, which no
// matcher claims. `rocket-fold-conv-pad` records the amount it verified on
// the convolution and the matchers below pin it to 1 twice over -- in the
// `rocket.pad_top`/`rocket.pad_left` attributes they compare, and in the
// `tensor.pad` amounts in the DAG template.
//
// **Symmetric only, and that is the hardware, not the wire.** `CNA_PAD_CON0`
// has `pad_top` and `pad_left` and nothing else, and each applies to *both*
// sides: `Shape::output_width` is `(w + 2 * pad_left - kw) / stride + 1`,
// matched against all 150 strided programs in the vendor corpus. A
// `low[0] high[1]` pad -- ONNX's `auto_pad = SAME_UPPER` at stride 2 -- has
// no expression here and stays materialized.
#rocket_dynamic_pad1_target = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "conv2d",
  input_width = 0 : i32, input_height = 0 : i32, input_channels = 0 : i32,
  output_width = 0 : i32, output_height = 0 : i32, output_channels = 0 : i32,
  weights_width = 0 : i32, weights_height = 0 : i32, stride = 1 : i32,
  depthwise = false,
  input_zero_point = 0 : i32, output_zero_point = 0 : i32, weights_zero_point = 0 : i32,
  input_scale = 1.0 : f32, weights_scale = 1.0 : f32, output_scale = 1.0 : f32,
  truncate_bits = 0 : i32,
  activation = "none", activation_cmp = 0 : i32,
  precision = "fp16",
  pad_top = 1 : i32, pad_left = 1 : i32,
  runtime_dimensions = [
    "input_width", "input_height", "input_channels",
    "output_channels", "weights_width", "weights_height"
  ]
}>

// Convolution with the model's "same" padding done by the CNA, stride 2.
//
// A model's padding arrives as an explicit `tensor.pad` and IREE forms it as
// its own dispatch -- `slow_memcpy` in an `audit` listing. On ResNet50 fp16
// that is 16 dispatch sites feeding offloaded convolutions, ~12 MB per
// inference to copy a tensor into one two rows and columns larger. The CNA
// pads for free: `CNA_PAD_CON0` costs no cycles and no DMA.
//
// **`pad = 1` is baked, and it is the only value worth baking.** The
// hardware's pad fields are 4 bits, but the matchers stop at 3x3, and 3x3
// "same" padding is pad 1; pad 2 and 3 belong to 5x5 and 7x7, which no
// matcher claims. `rocket-fold-conv-pad` records the amount it verified on
// the convolution and the matchers below pin it to 1 twice over -- in the
// `rocket.pad_top`/`rocket.pad_left` attributes they compare, and in the
// `tensor.pad` amounts in the DAG template.
//
// **Symmetric only, and that is the hardware, not the wire.** `CNA_PAD_CON0`
// has `pad_top` and `pad_left` and nothing else, and each applies to *both*
// sides: `Shape::output_width` is `(w + 2 * pad_left - kw) / stride + 1`,
// matched against all 150 strided programs in the vendor corpus. A
// `low[0] high[1]` pad -- ONNX's `auto_pad = SAME_UPPER` at stride 2 -- has
// no expression here and stays materialized.
#rocket_dynamic_pad1_target_s2 = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "conv2d",
  input_width = 0 : i32, input_height = 0 : i32, input_channels = 0 : i32,
  output_width = 0 : i32, output_height = 0 : i32, output_channels = 0 : i32,
  weights_width = 0 : i32, weights_height = 0 : i32, stride = 2 : i32,
  depthwise = false,
  input_zero_point = 0 : i32, output_zero_point = 0 : i32, weights_zero_point = 0 : i32,
  input_scale = 1.0 : f32, weights_scale = 1.0 : f32, output_scale = 1.0 : f32,
  truncate_bits = 0 : i32,
  activation = "none", activation_cmp = 0 : i32,
  precision = "fp16",
  pad_top = 1 : i32, pad_left = 1 : i32,
  runtime_dimensions = [
    "input_width", "input_height", "input_channels",
    "output_channels", "weights_width", "weights_height"
  ]
}>

#rocket_dynamic_relu6_target = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "conv2d",
  input_width = 0 : i32, input_height = 0 : i32, input_channels = 0 : i32,
  output_width = 0 : i32, output_height = 0 : i32, output_channels = 0 : i32,
  weights_width = 0 : i32, weights_height = 0 : i32, stride = 1 : i32,
  depthwise = false,
  input_zero_point = 0 : i32, output_zero_point = 0 : i32, weights_zero_point = 0 : i32,
  input_scale = 1.0 : f32, weights_scale = 1.0 : f32, output_scale = 1.0 : f32,
  truncate_bits = 0 : i32,
  activation = "relux", activation_cmp = 1086324736 : i32,   // 0x40C00000, f32 6.0
  precision = "fp16",
  runtime_dimensions = [
    "input_width", "input_height", "input_channels",
    "output_channels", "weights_width", "weights_height"
  ]
}>

// Depthwise twin of #rocket_dynamic_relu6_target, stride 1. Same reasoning
// throughout -- the ceiling is static and `rocket-fuse-conv-relu6` is what
// makes that sound, and the bias is a real binding because BN clamps after
// BS. Depthwise is a *different register program* (`CNA_CONV_CON1.CONV_MODE
// = 3`, `CORE_MISC_CFG.DW_EN = 1`, tap-major coefficients), so "the dense
// path works" is not evidence for it: `conv_fp16_bias_activation_hw` runs
// bias alone, bias + ReLU and bias + ReLU6 under both programs and all six
// are exact.
#rocket_dynamic_depthwise_relu6_target = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "conv2d",
  input_width = 0 : i32, input_height = 0 : i32, input_channels = 0 : i32,
  output_width = 0 : i32, output_height = 0 : i32, output_channels = 0 : i32,
  weights_width = 0 : i32, weights_height = 0 : i32, stride = 1 : i32,
  depthwise = true,
  input_zero_point = 0 : i32, output_zero_point = 0 : i32, weights_zero_point = 0 : i32,
  input_scale = 1.0 : f32, weights_scale = 1.0 : f32, output_scale = 1.0 : f32,
  truncate_bits = 0 : i32,
  activation = "relux", activation_cmp = 1086324736 : i32,   // 0x40C00000, f32 6.0
  precision = "fp16",
  runtime_dimensions = [
    "input_width", "input_height", "input_channels",
    "output_channels", "weights_width", "weights_height"
  ]
}>

// Depthwise twin of #rocket_dynamic_relu6_target, stride 2. Same reasoning
// throughout -- the ceiling is static and `rocket-fuse-conv-relu6` is what
// makes that sound, and the bias is a real binding because BN clamps after
// BS. Depthwise is a *different register program* (`CNA_CONV_CON1.CONV_MODE
// = 3`, `CORE_MISC_CFG.DW_EN = 1`, tap-major coefficients), so "the dense
// path works" is not evidence for it: `conv_fp16_bias_activation_hw` runs
// bias alone, bias + ReLU and bias + ReLU6 under both programs and all six
// are exact.
#rocket_dynamic_depthwise_relu6_target_s2 = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "conv2d",
  input_width = 0 : i32, input_height = 0 : i32, input_channels = 0 : i32,
  output_width = 0 : i32, output_height = 0 : i32, output_channels = 0 : i32,
  weights_width = 0 : i32, weights_height = 0 : i32, stride = 2 : i32,
  depthwise = true,
  input_zero_point = 0 : i32, output_zero_point = 0 : i32, weights_zero_point = 0 : i32,
  input_scale = 1.0 : f32, weights_scale = 1.0 : f32, output_scale = 1.0 : f32,
  truncate_bits = 0 : i32,
  activation = "relux", activation_cmp = 1086324736 : i32,   // 0x40C00000, f32 6.0
  precision = "fp16",
  runtime_dimensions = [
    "input_width", "input_height", "input_channels",
    "output_channels", "weights_width", "weights_height"
  ]
}>

// The scales here are not the model's. `pack_int8_bias_to_bs` normalizes the
// i32 bias by `input_scale * weights_scale`, and a QLinearConv bias is
// already in accumulator units, so holding both at 1.0 passes it through
// untouched and leaves `output_scale` carrying the whole requantization
// ratio: pass `y_scale / (x_scale * w_scale)`, and the driver derives
// `multiplier = 1 * 1 / output_scale`.
//
// `output_scale` and `output_zero_point` are zero here and listed in
// `runtime_quantization` because they are per-convolution calibration data
// while this executable is shared by every convolution that imports it --
// they arrive as push constants after the six dimensions. Zero is not a
// legal scale, which is what makes it a usable sentinel; see Conv2DQuantParam
// in rocket_executable_def.fbs for the bit-pattern convention.
//
// `input_zero_point` stays static at zero and is deliberately *not* a runtime
// parameter: it only reaches the hardware as CNA_PAD_CON1.pad_value, and
// every Rocket dispatch is unpadded (pad_top/pad_left are hardcoded zero in
// RocketTarget.cpp) because padding is materialized by an explicit tensor.pad
// ahead of the convolution. The input zero point's real contribution,
// `-x_zp * sum_k(w)`, is folded into the bias at compile time instead.
#rocket_dynamic_int8_requant_target = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "conv2d",
  input_width = 0 : i32, input_height = 0 : i32, input_channels = 0 : i32,
  output_width = 0 : i32, output_height = 0 : i32, output_channels = 0 : i32,
  weights_width = 0 : i32, weights_height = 0 : i32, stride = 1 : i32,
  depthwise = false,
  input_zero_point = 0 : i32, output_zero_point = 0 : i32, weights_zero_point = 0 : i32,
  input_scale = 1.0 : f32, weights_scale = 1.0 : f32, output_scale = 0.0 : f32,
  truncate_bits = 0 : i32,
  activation = "none", activation_cmp = 0 : i32,
  precision = "int8",
  runtime_dimensions = [
    "input_width", "input_height", "input_channels",
    "output_channels", "weights_width", "weights_height"
  ],
  runtime_quantization = ["output_scale", "output_zero_point"]
}>

// Requantized int8 depthwise: the same trade the dense requantized target
// makes, for the op that needs it most. All 13 of MobileNetV2's offloaded
// depthwise convolutions were on `int8_accumulator`, so each returned `i32`
// and had its requantization run as a CPU pass over a full activation tensor
// -- one `elementwise_i32xi32xi32xi8` per layer, the largest single category
// of CPU dispatch left in the model after the dense path was converted.
//
// Hardware-validated before any of this was written
// (`conv_depthwise_requant_hw.rs`): bit-exact at Cin 64 and 128 and at
// MobileNetV2's own 48, 144, 192 and 288, 884736 elements with none wrong.
// That test also settles the output layout question -- a requantized
// depthwise writes 16-byte atoms, not the 256-byte ones the accumulator
// path's depthwise uses.
//
// Stride 1 only, deliberately. That is what the 13 offloaded depthwise
// convolutions use; the model's stride-2 depthwise layers do not reach a
// Rocket matcher today at all, so a stride-2 requantized target would be
// dead weight until they do.
#rocket_dynamic_depthwise_int8_requant_target = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "conv2d",
  input_width = 0 : i32, input_height = 0 : i32, input_channels = 0 : i32,
  output_width = 0 : i32, output_height = 0 : i32, output_channels = 0 : i32,
  weights_width = 0 : i32, weights_height = 0 : i32, stride = 1 : i32,
  depthwise = true,
  input_zero_point = 0 : i32, output_zero_point = 0 : i32, weights_zero_point = 0 : i32,
  input_scale = 1.0 : f32, weights_scale = 1.0 : f32, output_scale = 0.0 : f32,
  truncate_bits = 0 : i32,
  activation = "none", activation_cmp = 0 : i32,
  precision = "int8",
  runtime_dimensions = [
    "input_width", "input_height", "input_channels",
    "output_channels", "weights_width", "weights_height"
  ],
  runtime_quantization = ["output_scale", "output_zero_point"]
}>

#dynamic_pipeline_layout = #hal.pipeline.layout<constants = 6, bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>

// Six dimensions plus the two runtime quantization parameters. The count has
// to match the target's two lists exactly -- RocketTarget.cpp checks it and
// the driver reads the constants in the same order, dimensions first.
#dynamic_requant_pipeline_layout = #hal.pipeline.layout<constants = 8, bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>

module attributes {transform.with_named_sequence} {

  hal.executable private @rocket_dynamic_executable {
    hal.executable.variant public @rocket_dynamic_conv2d_v1 target(#rocket_dynamic_target) {
      hal.executable.export public @rocket_dynamic_conv2d ordinal(0) layout(#dynamic_pipeline_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @rocket_dynamic_conv2d() {
          return
        }
      }
    }
  }

  hal.executable private @rocket_matmul_executable {
    hal.executable.variant public @rocket_matmul_v1 target(#rocket_matmul_target) {
      hal.executable.export public @rocket_matmul ordinal(0) layout(#matmul_pipeline_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @rocket_matmul() {
          return
        }
      }
    }
  }

  hal.executable private @rocket_pooling_executable {
    hal.executable.variant public @rocket_pooling_avg_v1 target(#rocket_pooling_avg_target) {
      hal.executable.export public @rocket_pooling_avg ordinal(0) layout(#pooling_pipeline_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @rocket_pooling_avg() {
          return
        }
      }
    }
  }

  hal.executable private @rocket_elementwise_add_executable {
    hal.executable.variant public @rocket_elementwise_add_v1 target(#rocket_elementwise_add_target) {
      hal.executable.export public @rocket_elementwise_add ordinal(0) layout(#elementwise_binary_pipeline_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @rocket_elementwise_add() {
          return
        }
      }
    }
  }

  hal.executable private @rocket_elementwise_sub_executable {
    hal.executable.variant public @rocket_elementwise_sub_v1 target(#rocket_elementwise_sub_target) {
      hal.executable.export public @rocket_elementwise_sub ordinal(0) layout(#elementwise_binary_pipeline_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @rocket_elementwise_sub() {
          return
        }
      }
    }
  }

  hal.executable private @rocket_elementwise_mul_executable {
    hal.executable.variant public @rocket_elementwise_mul_v1 target(#rocket_elementwise_mul_target) {
      hal.executable.export public @rocket_elementwise_mul ordinal(0) layout(#elementwise_binary_pipeline_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @rocket_elementwise_mul() {
          return
        }
      }
    }
  }

  hal.executable private @rocket_pooling_max_executable {
    hal.executable.variant public @rocket_pooling_max_v1 target(#rocket_pooling_max_target) {
      hal.executable.export public @rocket_pooling_max ordinal(0) layout(#pooling_pipeline_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @rocket_pooling_max() {
          return
        }
      }
    }
  }

  hal.executable private @rocket_pooling_max_executable_s2 {
    hal.executable.variant public @rocket_pooling_max_v1 target(#rocket_pooling_max_target_s2) {
      hal.executable.export public @rocket_pooling_max ordinal(0) layout(#pooling_pipeline_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @rocket_pooling_max() {
          return
        }
      }
    }
  }

  hal.executable private @rocket_pooling_min_executable {
    hal.executable.variant public @rocket_pooling_min_v1 target(#rocket_pooling_min_target) {
      hal.executable.export public @rocket_pooling_min ordinal(0) layout(#pooling_pipeline_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @rocket_pooling_min() {
          return
        }
      }
    }
  }

  hal.executable private @rocket_pooling_min_executable_s2 {
    hal.executable.variant public @rocket_pooling_min_v1 target(#rocket_pooling_min_target_s2) {
      hal.executable.export public @rocket_pooling_min ordinal(0) layout(#pooling_pipeline_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @rocket_pooling_min() {
          return
        }
      }
    }
  }

  hal.executable private @rocket_dynamic_depthwise_executable {
    hal.executable.variant public @rocket_dynamic_depthwise_conv2d_v1 target(#rocket_dynamic_depthwise_target) {
      hal.executable.export public @rocket_dynamic_depthwise_conv2d ordinal(0) layout(#dynamic_pipeline_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @rocket_dynamic_depthwise_conv2d() {
          return
        }
      }
    }
  }

  hal.executable private @rocket_dynamic_pad1_executable {
    hal.executable.variant public @rocket_dynamic_conv2d_v1 target(#rocket_dynamic_pad1_target) {
      hal.executable.export public @rocket_dynamic_conv2d ordinal(0) layout(#dynamic_pipeline_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @rocket_dynamic_conv2d() {
          return
        }
      }
    }
  }

  hal.executable private @rocket_dynamic_pad1_executable_s2 {
    hal.executable.variant public @rocket_dynamic_conv2d_v1 target(#rocket_dynamic_pad1_target_s2) {
      hal.executable.export public @rocket_dynamic_conv2d ordinal(0) layout(#dynamic_pipeline_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @rocket_dynamic_conv2d() {
          return
        }
      }
    }
  }

  hal.executable private @rocket_dynamic_relu6_executable {
    hal.executable.variant public @rocket_dynamic_conv2d_v1 target(#rocket_dynamic_relu6_target) {
      hal.executable.export public @rocket_dynamic_conv2d ordinal(0) layout(#dynamic_pipeline_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @rocket_dynamic_conv2d() {
          return
        }
      }
    }
  }

  hal.executable private @rocket_dynamic_depthwise_relu6_executable {
    hal.executable.variant public @rocket_dynamic_depthwise_conv2d_v1 target(#rocket_dynamic_depthwise_relu6_target) {
      hal.executable.export public @rocket_dynamic_depthwise_conv2d ordinal(0) layout(#dynamic_pipeline_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @rocket_dynamic_depthwise_conv2d() {
          return
        }
      }
    }
  }

  hal.executable private @rocket_dynamic_depthwise_relu6_executable_s2 {
    hal.executable.variant public @rocket_dynamic_depthwise_conv2d_v1 target(#rocket_dynamic_depthwise_relu6_target_s2) {
      hal.executable.export public @rocket_dynamic_depthwise_conv2d ordinal(0) layout(#dynamic_pipeline_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @rocket_dynamic_depthwise_conv2d() {
          return
        }
      }
    }
  }

  hal.executable private @rocket_dynamic_executable_s2 {
    hal.executable.variant public @rocket_dynamic_conv2d_v1 target(#rocket_dynamic_target_s2) {
      hal.executable.export public @rocket_dynamic_conv2d ordinal(0) layout(#dynamic_pipeline_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @rocket_dynamic_conv2d() {
          return
        }
      }
    }
  }

  hal.executable private @rocket_dynamic_executable_s3 {
    hal.executable.variant public @rocket_dynamic_conv2d_v1 target(#rocket_dynamic_target_s3) {
      hal.executable.export public @rocket_dynamic_conv2d ordinal(0) layout(#dynamic_pipeline_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @rocket_dynamic_conv2d() {
          return
        }
      }
    }
  }

  hal.executable private @rocket_dynamic_executable_s4 {
    hal.executable.variant public @rocket_dynamic_conv2d_v1 target(#rocket_dynamic_target_s4) {
      hal.executable.export public @rocket_dynamic_conv2d ordinal(0) layout(#dynamic_pipeline_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @rocket_dynamic_conv2d() {
          return
        }
      }
    }
  }

  hal.executable private @rocket_dynamic_depthwise_executable_s2 {
    hal.executable.variant public @rocket_dynamic_depthwise_conv2d_v1 target(#rocket_dynamic_depthwise_target_s2) {
      hal.executable.export public @rocket_dynamic_depthwise_conv2d ordinal(0) layout(#dynamic_pipeline_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @rocket_dynamic_depthwise_conv2d() {
          return
        }
      }
    }
  }

  hal.executable private @rocket_dynamic_depthwise_executable_s3 {
    hal.executable.variant public @rocket_dynamic_depthwise_conv2d_v1 target(#rocket_dynamic_depthwise_target_s3) {
      hal.executable.export public @rocket_dynamic_depthwise_conv2d ordinal(0) layout(#dynamic_pipeline_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @rocket_dynamic_depthwise_conv2d() {
          return
        }
      }
    }
  }

  hal.executable private @rocket_dynamic_depthwise_executable_s4 {
    hal.executable.variant public @rocket_dynamic_depthwise_conv2d_v1 target(#rocket_dynamic_depthwise_target_s4) {
      hal.executable.export public @rocket_dynamic_depthwise_conv2d ordinal(0) layout(#dynamic_pipeline_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @rocket_dynamic_depthwise_conv2d() {
          return
        }
      }
    }
  }


  hal.executable private @rocket_dynamic_int8_executable {
    hal.executable.variant public @rocket_dynamic_conv2d_v1 target(#rocket_dynamic_int8_target) {
      hal.executable.export public @rocket_dynamic_conv2d ordinal(0) layout(#dynamic_pipeline_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @rocket_dynamic_conv2d() {
          return
        }
      }
    }
  }

  hal.executable private @rocket_dynamic_int8_requant_executable {
    hal.executable.variant public @rocket_dynamic_conv2d_v1 target(#rocket_dynamic_int8_requant_target) {
      hal.executable.export public @rocket_dynamic_conv2d ordinal(0) layout(#dynamic_requant_pipeline_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @rocket_dynamic_conv2d() {
          return
        }
      }
    }
  }

  hal.executable private @rocket_dynamic_depthwise_int8_requant_executable {
    hal.executable.variant public @rocket_dynamic_depthwise_conv2d_v1 target(#rocket_dynamic_depthwise_int8_requant_target) {
      hal.executable.export public @rocket_dynamic_depthwise_conv2d ordinal(0) layout(#dynamic_requant_pipeline_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @rocket_dynamic_depthwise_conv2d() {
          return
        }
      }
    }
  }

  hal.executable private @rocket_dynamic_depthwise_int8_executable {
    hal.executable.variant public @rocket_dynamic_depthwise_conv2d_v1 target(#rocket_dynamic_depthwise_int8_target) {
      hal.executable.export public @rocket_dynamic_depthwise_conv2d ordinal(0) layout(#dynamic_pipeline_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @rocket_dynamic_depthwise_conv2d() {
          return
        }
      }
    }
  }

  hal.executable private @rocket_dynamic_depthwise_int8_executable_s2 {
    hal.executable.variant public @rocket_dynamic_depthwise_conv2d_v1 target(#rocket_dynamic_depthwise_int8_target_s2) {
      hal.executable.export public @rocket_dynamic_depthwise_conv2d ordinal(0) layout(#dynamic_pipeline_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @rocket_dynamic_depthwise_conv2d() {
          return
        }
      }
    }
  }


  // The replacement for a matched linalg.matmul.
  //
  // No reshaping: the dispatch's operands are the 2-D matrices as they
  // stand. The tensor types here only fix each binding's size, and the
  // runtime derives the geometry from the M/K/N push constants -- it is
  // `fc::Shape` that knows M is a convolution width and K its input
  // channels, not this file. A is [M,K] row-major and B is [K,N], which is
  // already a 1x1 HWCF filter; the matcher pins the indexing maps so a
  // transposed operand cannot arrive here claiming to be one.
  //
  // The bias binding is zero-filled. MobileNetV2's bias add is a separate
  // linalg.generic and stays on the CPU: folding it would cost a matcher
  // that claims two ops for one elementwise pass over 1001 floats.
  //
  // Both matrix operands arrive already f16: RocketDemoteConvInputsPass
  // narrows a matmul exactly as it narrows a convolution, and
  // @match_rocket_matmul requires the result. This function used to do the
  // narrowing itself, which looked equivalent and was not -- it is never
  // inlined (every dispatch formed inside it is named
  // `call_rocket_matmul_dispatch_N`), so a truncf here is invisible to
  // const-expr hoisting and re-narrows the *constant* classifier weights on
  // every inference: 1.79M elements of CPU work into a fresh transient
  // buffer, which then misses the runtime's packed-coefficient cache every
  // time as well. Demoted in the caller instead, const-eval folds it into an
  // initializer, as it already did for every convolution's weights.
  util.func private @call_rocket_matmul(
      %lhs: tensor<?x?xf16>,
      %rhs: tensor<?x?xf16>,
      %init: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %m = tensor.dim %lhs, %c0 : tensor<?x?xf16>
    %k = tensor.dim %lhs, %c1 : tensor<?x?xf16>
    %n = tensor.dim %rhs, %c1 : tensor<?x?xf16>

    %m_i32 = arith.index_cast %m : index to i32
    %k_i32 = arith.index_cast %k : index to i32
    %n_i32 = arith.index_cast %n : index to i32

    %zero_bias_empty = tensor.empty(%n) : tensor<?xf16>
    %zero_f16 = arith.constant 0.0 : f16
    %zero_bias = linalg.fill ins(%zero_f16 : f16)
        outs(%zero_bias_empty : tensor<?xf16>) -> tensor<?xf16>

    %raw_f16 = flow.dispatch
        @rocket_matmul_executable::@rocket_matmul_v1::@rocket_matmul(
          %m_i32, %k_i32, %n_i32,
          %lhs, %rhs, %zero_bias)
        {stream.affinity = #hal.device.affinity<@rocket_device>}
        : (i32, i32, i32,
           tensor<?x?xf16>{%m, %k},
           tensor<?x?xf16>{%k, %n},
           tensor<?xf16>{%n})
        -> tensor<?x?xf16>{%m, %n}

    // Widen and accumulate on the CPU, explicitly -- an op consuming the
    // Rocket result otherwise inherits its affinity and is formed into an
    // executable for a device with no config to serialize.
    %final = flow.dispatch.workgroups[%m, %n](%raw_f16, %init, %m, %n)
        : (tensor<?x?xf16>{%m, %n}, tensor<?x?xf32>{%m, %n}, index, index)
        -> tensor<?x?xf32>{%m, %n}
        attributes { stream.affinity = #hal.device.affinity<@cpu_device> } =
        (%raw_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf16>>,
         %init_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>,
         %m_arg: index,
         %n_arg: index,
         %final_binding: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>) {
      %m_size = iree_tensor_ext.dispatch.workload.ordinal %m_arg, 0 : index
      %n_size = iree_tensor_ext.dispatch.workload.ordinal %n_arg, 1 : index
      %raw_shaped = flow.dispatch.tie_shape %raw_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf16>>{%m_size, %n_size}
      %init_shaped = flow.dispatch.tie_shape %init_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>{%m_size, %n_size}
      %final_shaped = flow.dispatch.tie_shape %final_binding
          : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>{%m_size, %n_size}
      %raw_loaded = iree_tensor_ext.dispatch.tensor.load %raw_shaped,
          offsets = [0, 0], sizes = [%m_size, %n_size], strides = [1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf16>>{%m_size, %n_size}
          -> tensor<?x?xf16>
      %init_loaded = iree_tensor_ext.dispatch.tensor.load %init_shaped,
          offsets = [0, 0], sizes = [%m_size, %n_size], strides = [1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>{%m_size, %n_size}
          -> tensor<?x?xf32>
      %final_empty = tensor.empty(%m_size, %n_size) : tensor<?x?xf32>
      %final_inner = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>
          ],
          iterator_types = ["parallel", "parallel"]
        } ins(%raw_loaded, %init_loaded : tensor<?x?xf16>, tensor<?x?xf32>)
          outs(%final_empty : tensor<?x?xf32>) {
        ^bb0(%raw: f16, %initial: f32, %out: f32):
          %raw_f32 = arith.extf %raw : f16 to f32
          %sum = arith.addf %raw_f32, %initial : f32
          linalg.yield %sum : f32
      } -> tensor<?x?xf32>
      iree_tensor_ext.dispatch.tensor.store %final_inner, %final_shaped,
          offsets = [0, 0], sizes = [%m_size, %n_size], strides = [1, 1]
          : tensor<?x?xf32>
          -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>{%m_size, %n_size}
      flow.return
    } count(%m_workload: index, %n_workload: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(
          %m_workload, %n_workload)
      flow.return %x, %y, %z : index, index, index
    }

    util.return %final : tensor<?x?xf32>
  }

  // The replacement for a matched linalg.pooling_nchw_sum.
  //
  // Three things happen around the dispatch, and each is here because the
  // hardware and the input dialect disagree about something:
  //
  //   * NCHW -> NHWC and back. The PPU reads and writes NC1HWC2 cubes built
  //     from NHWC, and pooling arrives NCHW because that is what
  //     torch-mlir emits and, unlike dense convolution, nothing upstream
  //     converts it (the same reason the depthwise NCHW shim above exists).
  //   * f32 -> f16 and back. Nothing demotes pooling inputs the way
  //     RocketDemoteConvInputsPass demotes convolution and matmul ones, so the
  //     truncation is explicit here.
  //   * a multiply by kh*kw. The op is a *sum* pool and the hardware
  //     computes an *average*, so the result is scaled back up to the sum
  //     the consumer expects. The model's own divide then produces the
  //     average it was always going to.
  //
  // The `outs` operand is an accumulator initialiser, so it is added rather
  // than ignored -- the same contract the convolution shims honour.
  util.func private @call_rocket_pooling_avg_nchw(
      %input: tensor<1x?x?x?xf32>,
      %window: tensor<?x?xf32>,
      %init: tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    // NCHW: dim 1 is channels, dims 2/3 are the spatial extent.
    %channels = tensor.dim %input, %c1 : tensor<1x?x?x?xf32>
    %input_height = tensor.dim %input, %c2 : tensor<1x?x?x?xf32>
    %input_width = tensor.dim %input, %c3 : tensor<1x?x?x?xf32>
    // The window operand carries no values, only [kh, kw].
    %kernel_height = tensor.dim %window, %c0 : tensor<?x?xf32>
    %kernel_width = tensor.dim %window, %c1 : tensor<?x?xf32>
    %output_height = tensor.dim %init, %c2 : tensor<1x?x?x?xf32>
    %output_width = tensor.dim %init, %c3 : tensor<1x?x?x?xf32>

    %input_width_i32 = arith.index_cast %input_width : index to i32
    %input_height_i32 = arith.index_cast %input_height : index to i32
    %channels_i32 = arith.index_cast %channels : index to i32
    %kernel_width_i32 = arith.index_cast %kernel_width : index to i32
    %kernel_height_i32 = arith.index_cast %kernel_height : index to i32

    // NCHW [1,C,H,W] -> NHWC [1,H,W,C]: out.shape[i] = in.shape[perm[i]].
    %input_nhwc_empty = tensor.empty(%input_height, %input_width, %channels) : tensor<1x?x?x?xf32>
    %input_nhwc = linalg.transpose
        ins(%input : tensor<1x?x?x?xf32>)
        outs(%input_nhwc_empty : tensor<1x?x?x?xf32>)
        permutation = [0, 2, 3, 1]

    %input_f16_empty = tensor.empty(%input_height, %input_width, %channels) : tensor<1x?x?x?xf16>
    %input_f16 = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } ins(%input_nhwc : tensor<1x?x?x?xf32>)
        outs(%input_f16_empty : tensor<1x?x?x?xf16>) {
      ^bb0(%value: f32, %out: f16):
        %narrowed = arith.truncf %value : f32 to f16
        linalg.yield %narrowed : f16
    } -> tensor<1x?x?x?xf16>

    %averaged = flow.dispatch
        @rocket_pooling_executable::@rocket_pooling_avg_v1::@rocket_pooling_avg(
          %input_width_i32, %input_height_i32, %channels_i32,
          %kernel_width_i32, %kernel_height_i32,
          %input_f16)
        {stream.affinity = #hal.device.affinity<@rocket_device>}
        : (i32, i32, i32, i32, i32,
           tensor<1x?x?x?xf16>{%input_height, %input_width, %channels})
        -> tensor<1x?x?x?xf16>{%output_height, %output_width, %channels}

    // %init arrives NCHW too; line it up with the NHWC result.
    %init_nhwc_empty = tensor.empty(%output_height, %output_width, %channels) : tensor<1x?x?x?xf32>
    %init_nhwc = linalg.transpose
        ins(%init : tensor<1x?x?x?xf32>)
        outs(%init_nhwc_empty : tensor<1x?x?x?xf32>)
        permutation = [0, 2, 3, 1]

    // Undo the hardware's divide -- the matched op is a sum pool -- widen
    // back to f32, and add the accumulator initialiser.
    //
    // Explicitly a CPU dispatch, like the convolution shims' accumulate and
    // for the same reason: an op that consumes the Rocket result inherits
    // its affinity, gets formed into an executable for the rocket device,
    // and that executable has no conv2d config to serialize. The failure is
    // a serialization error naming a missing `input_width`, which is a long
    // way from the cause.
    %taps = arith.muli %kernel_height, %kernel_width : index

    %final_nhwc = flow.dispatch.workgroups[
        %output_height, %output_width, %channels](
        %averaged, %init_nhwc, %taps, %output_height, %output_width, %channels)
        : (tensor<1x?x?x?xf16>{%output_height, %output_width, %channels},
           tensor<1x?x?x?xf32>{%output_height, %output_width, %channels},
           index, index, index, index)
        -> tensor<1x?x?x?xf32>{%output_height, %output_width, %channels}
        attributes { stream.affinity = #hal.device.affinity<@cpu_device> } =
        (%averaged_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>,
         %init_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>,
         %taps_arg: index,
         %output_height_arg: index,
         %output_width_arg: index,
         %channels_arg: index,
         %final_binding: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>) {
      %output_height_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_height_arg, 0 : index
      %output_width_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_width_arg, 1 : index
      %channels_size = iree_tensor_ext.dispatch.workload.ordinal
          %channels_arg, 2 : index
      %averaged_shaped = flow.dispatch.tie_shape %averaged_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %channels_size}
      %init_shaped = flow.dispatch.tie_shape %init_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %channels_size}
      %final_shaped = flow.dispatch.tie_shape %final_binding
          : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %channels_size}
      %averaged_loaded = iree_tensor_ext.dispatch.tensor.load %averaged_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %channels_size}
          -> tensor<1x?x?x?xf16>
      %init_loaded = iree_tensor_ext.dispatch.tensor.load %init_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %channels_size}
          -> tensor<1x?x?x?xf32>
      %taps_i32 = arith.index_cast %taps_arg : index to i32
      %taps_f32 = arith.sitofp %taps_i32 : i32 to f32
      %final_empty = tensor.empty(
          %output_height_size, %output_width_size, %channels_size)
          : tensor<1x?x?x?xf32>
      %final_inner = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
          ],
          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } ins(%averaged_loaded, %init_loaded
            : tensor<1x?x?x?xf16>, tensor<1x?x?x?xf32>)
          outs(%final_empty : tensor<1x?x?x?xf32>) {
        ^bb0(%average: f16, %initial: f32, %out: f32):
          %average_f32 = arith.extf %average : f16 to f32
          %sum = arith.mulf %average_f32, %taps_f32 : f32
          %accumulated = arith.addf %sum, %initial : f32
          linalg.yield %accumulated : f32
      } -> tensor<1x?x?x?xf32>
      iree_tensor_ext.dispatch.tensor.store %final_inner, %final_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %channels_size],
          strides = [1, 1, 1, 1]
          : tensor<1x?x?x?xf32>
          -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %channels_size}
      flow.return
    } count(%output_height_workload: index,
            %output_width_workload: index,
            %channels_workload: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(
          %output_height_workload,
          %output_width_workload,
          %channels_workload)
      flow.return %x, %y, %z : index, index, index
    }

    // NHWC [1,H,W,C] -> NCHW [1,C,H,W].
    %final_nchw_empty = tensor.empty(%channels, %output_height, %output_width) : tensor<1x?x?x?xf32>
    %final_nchw = linalg.transpose
        ins(%final_nhwc : tensor<1x?x?x?xf32>)
        outs(%final_nchw_empty : tensor<1x?x?x?xf32>)
        permutation = [0, 3, 1, 2]

    util.return %final_nchw : tensor<1x?x?x?xf32>
  }


  // Two-tensor element-wise `add`, rank 3. The shape a compiled model
  // actually presents: ONNX `Add`/`Mul`/`Sub` reach the spec as a
  // `linalg.generic` over `1x?x?` with an `arith.addf` body (verified
  // by compiling a torch-onnx probe to `--compile-to=preprocessing`), not as
  // a `linalg.add` named op and not as `linalg.elementwise`.
  //
  // `1 x T x C` maps to the hardware cube as width = T, height = 1,
  // channels = C: the trailing dimension is innermost in memory and is what
  // NC1HWC2 packs into feature atoms, and the leading 1 is the batch the
  // matcher pins. For ViT that is 197 tokens of 768 (or 3072) channels.
  //
  // There is no init to fold in, unlike the pooling shims: `O = A add B`
  // writes every element, so the `outs` operand is a pure destination and
  // the CPU epilogue only has to widen f16 back to f32.
  util.func private @call_rocket_elementwise_add(
      %lhs: tensor<1x?x?xf32>,
      %rhs: tensor<1x?x?xf32>,
      %init: tensor<1x?x?xf32>) -> tensor<1x?x?xf32> {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    %tokens = tensor.dim %lhs, %c1 : tensor<1x?x?xf32>
    %channels = tensor.dim %lhs, %c2 : tensor<1x?x?xf32>

    %width_i32 = arith.index_cast %tokens : index to i32
    %channels_i32 = arith.index_cast %channels : index to i32
    // Height is 1: a rank-3 token tensor is one row of `tokens` pixels.
    %height_i32 = arith.constant 1 : i32

    // The EW datapath is fp16; the model's tensors are f32.
    %lhs_f16_empty = tensor.empty(%tokens, %channels) : tensor<1x?x?xf16>
    %lhs_f16 = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>
        ],
        iterator_types = ["parallel", "parallel", "parallel"]
      } ins(%lhs : tensor<1x?x?xf32>)
        outs(%lhs_f16_empty : tensor<1x?x?xf16>) {
      ^bb0(%value: f32, %out: f16):
        %narrowed = arith.truncf %value : f32 to f16
        linalg.yield %narrowed : f16
    } -> tensor<1x?x?xf16>

    %rhs_f16_empty = tensor.empty(%tokens, %channels) : tensor<1x?x?xf16>
    %rhs_f16 = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>
        ],
        iterator_types = ["parallel", "parallel", "parallel"]
      } ins(%rhs : tensor<1x?x?xf32>)
        outs(%rhs_f16_empty : tensor<1x?x?xf16>) {
      ^bb0(%value: f32, %out: f16):
        %narrowed = arith.truncf %value : f32 to f16
        linalg.yield %narrowed : f16
    } -> tensor<1x?x?xf16>

    %combined = flow.dispatch
        @rocket_elementwise_add_executable::@rocket_elementwise_add_v1::@rocket_elementwise_add(
          %width_i32, %height_i32, %channels_i32,
          %lhs_f16, %rhs_f16)
        {stream.affinity = #hal.device.affinity<@rocket_device>}
        : (i32, i32, i32,
           tensor<1x?x?xf16>{%tokens, %channels},
           tensor<1x?x?xf16>{%tokens, %channels})
        -> tensor<1x?x?xf16>{%tokens, %channels}

    // Explicitly a CPU dispatch, for the reason the pooling shims record: an
    // op that consumes the Rocket result inherits its affinity, gets formed
    // into an executable for the rocket device, and that executable has no
    // element-wise config to serialize. The failure is a serialization error
    // naming a missing `width`, a long way from the cause.
    %final = flow.dispatch.workgroups[%tokens, %channels](
        %combined, %tokens, %channels)
        : (tensor<1x?x?xf16>{%tokens, %channels}, index, index)
        -> tensor<1x?x?xf32>{%tokens, %channels}
        attributes { stream.affinity = #hal.device.affinity<@cpu_device> } =
        (%combined_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?xf16>>,
         %tokens_arg: index,
         %channels_arg: index,
         %final_binding: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?xf32>>) {
      %tokens_size = iree_tensor_ext.dispatch.workload.ordinal %tokens_arg, 0 : index
      %channels_size = iree_tensor_ext.dispatch.workload.ordinal %channels_arg, 1 : index
      %combined_shaped = flow.dispatch.tie_shape %combined_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?xf16>>{
              %tokens_size, %channels_size}
      %final_shaped = flow.dispatch.tie_shape %final_binding
          : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?xf32>>{
              %tokens_size, %channels_size}
      %combined_loaded = iree_tensor_ext.dispatch.tensor.load %combined_shaped,
          offsets = [0, 0, 0],
          sizes = [1, %tokens_size, %channels_size],
          strides = [1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?xf16>>{
              %tokens_size, %channels_size}
          -> tensor<1x?x?xf16>
      %final_empty = tensor.empty(%tokens_size, %channels_size) : tensor<1x?x?xf32>
      %final_inner = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
            affine_map<(d0, d1, d2) -> (d0, d1, d2)>
          ],
          iterator_types = ["parallel", "parallel", "parallel"]
        } ins(%combined_loaded : tensor<1x?x?xf16>)
          outs(%final_empty : tensor<1x?x?xf32>) {
        ^bb0(%value: f16, %out: f32):
          %widened = arith.extf %value : f16 to f32
          linalg.yield %widened : f32
      } -> tensor<1x?x?xf32>
      iree_tensor_ext.dispatch.tensor.store %final_inner, %final_shaped,
          offsets = [0, 0, 0],
          sizes = [1, %tokens_size, %channels_size],
          strides = [1, 1, 1]
          : tensor<1x?x?xf32>
          -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?xf32>>{
              %tokens_size, %channels_size}
      flow.return
    } count(%tokens_workload: index, %channels_workload: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(
          %tokens_workload, %channels_workload)
      flow.return %x, %y, %z : index, index, index
    }

    util.return %final : tensor<1x?x?xf32>
  }

  // Two-tensor element-wise `subtract`, rank 3. The shape a compiled model
  // actually presents: ONNX `Add`/`Mul`/`Sub` reach the spec as a
  // `linalg.generic` over `1x?x?` with an `arith.subf` body (verified
  // by compiling a torch-onnx probe to `--compile-to=preprocessing`), not as
  // a `linalg.add` named op and not as `linalg.elementwise`.
  //
  // `1 x T x C` maps to the hardware cube as width = T, height = 1,
  // channels = C: the trailing dimension is innermost in memory and is what
  // NC1HWC2 packs into feature atoms, and the leading 1 is the batch the
  // matcher pins. For ViT that is 197 tokens of 768 (or 3072) channels.
  //
  // There is no init to fold in, unlike the pooling shims: `O = A subtract B`
  // writes every element, so the `outs` operand is a pure destination and
  // the CPU epilogue only has to widen f16 back to f32.
  util.func private @call_rocket_elementwise_sub(
      %lhs: tensor<1x?x?xf32>,
      %rhs: tensor<1x?x?xf32>,
      %init: tensor<1x?x?xf32>) -> tensor<1x?x?xf32> {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    %tokens = tensor.dim %lhs, %c1 : tensor<1x?x?xf32>
    %channels = tensor.dim %lhs, %c2 : tensor<1x?x?xf32>

    %width_i32 = arith.index_cast %tokens : index to i32
    %channels_i32 = arith.index_cast %channels : index to i32
    // Height is 1: a rank-3 token tensor is one row of `tokens` pixels.
    %height_i32 = arith.constant 1 : i32

    // The EW datapath is fp16; the model's tensors are f32.
    %lhs_f16_empty = tensor.empty(%tokens, %channels) : tensor<1x?x?xf16>
    %lhs_f16 = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>
        ],
        iterator_types = ["parallel", "parallel", "parallel"]
      } ins(%lhs : tensor<1x?x?xf32>)
        outs(%lhs_f16_empty : tensor<1x?x?xf16>) {
      ^bb0(%value: f32, %out: f16):
        %narrowed = arith.truncf %value : f32 to f16
        linalg.yield %narrowed : f16
    } -> tensor<1x?x?xf16>

    %rhs_f16_empty = tensor.empty(%tokens, %channels) : tensor<1x?x?xf16>
    %rhs_f16 = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>
        ],
        iterator_types = ["parallel", "parallel", "parallel"]
      } ins(%rhs : tensor<1x?x?xf32>)
        outs(%rhs_f16_empty : tensor<1x?x?xf16>) {
      ^bb0(%value: f32, %out: f16):
        %narrowed = arith.truncf %value : f32 to f16
        linalg.yield %narrowed : f16
    } -> tensor<1x?x?xf16>

    %combined = flow.dispatch
        @rocket_elementwise_sub_executable::@rocket_elementwise_sub_v1::@rocket_elementwise_sub(
          %width_i32, %height_i32, %channels_i32,
          %lhs_f16, %rhs_f16)
        {stream.affinity = #hal.device.affinity<@rocket_device>}
        : (i32, i32, i32,
           tensor<1x?x?xf16>{%tokens, %channels},
           tensor<1x?x?xf16>{%tokens, %channels})
        -> tensor<1x?x?xf16>{%tokens, %channels}

    // Explicitly a CPU dispatch, for the reason the pooling shims record: an
    // op that consumes the Rocket result inherits its affinity, gets formed
    // into an executable for the rocket device, and that executable has no
    // element-wise config to serialize. The failure is a serialization error
    // naming a missing `width`, a long way from the cause.
    %final = flow.dispatch.workgroups[%tokens, %channels](
        %combined, %tokens, %channels)
        : (tensor<1x?x?xf16>{%tokens, %channels}, index, index)
        -> tensor<1x?x?xf32>{%tokens, %channels}
        attributes { stream.affinity = #hal.device.affinity<@cpu_device> } =
        (%combined_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?xf16>>,
         %tokens_arg: index,
         %channels_arg: index,
         %final_binding: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?xf32>>) {
      %tokens_size = iree_tensor_ext.dispatch.workload.ordinal %tokens_arg, 0 : index
      %channels_size = iree_tensor_ext.dispatch.workload.ordinal %channels_arg, 1 : index
      %combined_shaped = flow.dispatch.tie_shape %combined_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?xf16>>{
              %tokens_size, %channels_size}
      %final_shaped = flow.dispatch.tie_shape %final_binding
          : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?xf32>>{
              %tokens_size, %channels_size}
      %combined_loaded = iree_tensor_ext.dispatch.tensor.load %combined_shaped,
          offsets = [0, 0, 0],
          sizes = [1, %tokens_size, %channels_size],
          strides = [1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?xf16>>{
              %tokens_size, %channels_size}
          -> tensor<1x?x?xf16>
      %final_empty = tensor.empty(%tokens_size, %channels_size) : tensor<1x?x?xf32>
      %final_inner = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
            affine_map<(d0, d1, d2) -> (d0, d1, d2)>
          ],
          iterator_types = ["parallel", "parallel", "parallel"]
        } ins(%combined_loaded : tensor<1x?x?xf16>)
          outs(%final_empty : tensor<1x?x?xf32>) {
        ^bb0(%value: f16, %out: f32):
          %widened = arith.extf %value : f16 to f32
          linalg.yield %widened : f32
      } -> tensor<1x?x?xf32>
      iree_tensor_ext.dispatch.tensor.store %final_inner, %final_shaped,
          offsets = [0, 0, 0],
          sizes = [1, %tokens_size, %channels_size],
          strides = [1, 1, 1]
          : tensor<1x?x?xf32>
          -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?xf32>>{
              %tokens_size, %channels_size}
      flow.return
    } count(%tokens_workload: index, %channels_workload: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(
          %tokens_workload, %channels_workload)
      flow.return %x, %y, %z : index, index, index
    }

    util.return %final : tensor<1x?x?xf32>
  }

  // Two-tensor element-wise `multiply`, rank 3. The shape a compiled model
  // actually presents: ONNX `Add`/`Mul`/`Sub` reach the spec as a
  // `linalg.generic` over `1x?x?` with an `arith.mulf` body (verified
  // by compiling a torch-onnx probe to `--compile-to=preprocessing`), not as
  // a `linalg.add` named op and not as `linalg.elementwise`.
  //
  // `1 x T x C` maps to the hardware cube as width = T, height = 1,
  // channels = C: the trailing dimension is innermost in memory and is what
  // NC1HWC2 packs into feature atoms, and the leading 1 is the batch the
  // matcher pins. For ViT that is 197 tokens of 768 (or 3072) channels.
  //
  // There is no init to fold in, unlike the pooling shims: `O = A multiply B`
  // writes every element, so the `outs` operand is a pure destination and
  // the CPU epilogue only has to widen f16 back to f32.
  util.func private @call_rocket_elementwise_mul(
      %lhs: tensor<1x?x?xf32>,
      %rhs: tensor<1x?x?xf32>,
      %init: tensor<1x?x?xf32>) -> tensor<1x?x?xf32> {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    %tokens = tensor.dim %lhs, %c1 : tensor<1x?x?xf32>
    %channels = tensor.dim %lhs, %c2 : tensor<1x?x?xf32>

    %width_i32 = arith.index_cast %tokens : index to i32
    %channels_i32 = arith.index_cast %channels : index to i32
    // Height is 1: a rank-3 token tensor is one row of `tokens` pixels.
    %height_i32 = arith.constant 1 : i32

    // The EW datapath is fp16; the model's tensors are f32.
    %lhs_f16_empty = tensor.empty(%tokens, %channels) : tensor<1x?x?xf16>
    %lhs_f16 = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>
        ],
        iterator_types = ["parallel", "parallel", "parallel"]
      } ins(%lhs : tensor<1x?x?xf32>)
        outs(%lhs_f16_empty : tensor<1x?x?xf16>) {
      ^bb0(%value: f32, %out: f16):
        %narrowed = arith.truncf %value : f32 to f16
        linalg.yield %narrowed : f16
    } -> tensor<1x?x?xf16>

    %rhs_f16_empty = tensor.empty(%tokens, %channels) : tensor<1x?x?xf16>
    %rhs_f16 = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>
        ],
        iterator_types = ["parallel", "parallel", "parallel"]
      } ins(%rhs : tensor<1x?x?xf32>)
        outs(%rhs_f16_empty : tensor<1x?x?xf16>) {
      ^bb0(%value: f32, %out: f16):
        %narrowed = arith.truncf %value : f32 to f16
        linalg.yield %narrowed : f16
    } -> tensor<1x?x?xf16>

    %combined = flow.dispatch
        @rocket_elementwise_mul_executable::@rocket_elementwise_mul_v1::@rocket_elementwise_mul(
          %width_i32, %height_i32, %channels_i32,
          %lhs_f16, %rhs_f16)
        {stream.affinity = #hal.device.affinity<@rocket_device>}
        : (i32, i32, i32,
           tensor<1x?x?xf16>{%tokens, %channels},
           tensor<1x?x?xf16>{%tokens, %channels})
        -> tensor<1x?x?xf16>{%tokens, %channels}

    // Explicitly a CPU dispatch, for the reason the pooling shims record: an
    // op that consumes the Rocket result inherits its affinity, gets formed
    // into an executable for the rocket device, and that executable has no
    // element-wise config to serialize. The failure is a serialization error
    // naming a missing `width`, a long way from the cause.
    %final = flow.dispatch.workgroups[%tokens, %channels](
        %combined, %tokens, %channels)
        : (tensor<1x?x?xf16>{%tokens, %channels}, index, index)
        -> tensor<1x?x?xf32>{%tokens, %channels}
        attributes { stream.affinity = #hal.device.affinity<@cpu_device> } =
        (%combined_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?xf16>>,
         %tokens_arg: index,
         %channels_arg: index,
         %final_binding: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?xf32>>) {
      %tokens_size = iree_tensor_ext.dispatch.workload.ordinal %tokens_arg, 0 : index
      %channels_size = iree_tensor_ext.dispatch.workload.ordinal %channels_arg, 1 : index
      %combined_shaped = flow.dispatch.tie_shape %combined_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?xf16>>{
              %tokens_size, %channels_size}
      %final_shaped = flow.dispatch.tie_shape %final_binding
          : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?xf32>>{
              %tokens_size, %channels_size}
      %combined_loaded = iree_tensor_ext.dispatch.tensor.load %combined_shaped,
          offsets = [0, 0, 0],
          sizes = [1, %tokens_size, %channels_size],
          strides = [1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?xf16>>{
              %tokens_size, %channels_size}
          -> tensor<1x?x?xf16>
      %final_empty = tensor.empty(%tokens_size, %channels_size) : tensor<1x?x?xf32>
      %final_inner = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
            affine_map<(d0, d1, d2) -> (d0, d1, d2)>
          ],
          iterator_types = ["parallel", "parallel", "parallel"]
        } ins(%combined_loaded : tensor<1x?x?xf16>)
          outs(%final_empty : tensor<1x?x?xf32>) {
        ^bb0(%value: f16, %out: f32):
          %widened = arith.extf %value : f16 to f32
          linalg.yield %widened : f32
      } -> tensor<1x?x?xf32>
      iree_tensor_ext.dispatch.tensor.store %final_inner, %final_shaped,
          offsets = [0, 0, 0],
          sizes = [1, %tokens_size, %channels_size],
          strides = [1, 1, 1]
          : tensor<1x?x?xf32>
          -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?xf32>>{
              %tokens_size, %channels_size}
      flow.return
    } count(%tokens_workload: index, %channels_workload: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(
          %tokens_workload, %channels_workload)
      flow.return %x, %y, %z : index, index, index
    }

    util.return %final : tensor<1x?x?xf32>
  }

  // Max pool, NHWC, stride 1. NHWC is the hardware's own layout, so unlike the
  // average pool's NCHW shim there is nothing to transpose on either side.
  util.func private @call_rocket_pooling_max_nhwc(
      %input: tensor<1x?x?x?xf32>,
      %window: tensor<?x?xf32>,
      %init: tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    // NHWC: dims 1 and 2 are the spatial extent, dim 3 is channels.
    %input_height = tensor.dim %input, %c1 : tensor<1x?x?x?xf32>
    %input_width = tensor.dim %input, %c2 : tensor<1x?x?x?xf32>
    %channels = tensor.dim %input, %c3 : tensor<1x?x?x?xf32>
    // The window operand carries no values, only [kh, kw].
    %kernel_height = tensor.dim %window, %c0 : tensor<?x?xf32>
    %kernel_width = tensor.dim %window, %c1 : tensor<?x?xf32>
    %output_height = tensor.dim %init, %c1 : tensor<1x?x?x?xf32>
    %output_width = tensor.dim %init, %c2 : tensor<1x?x?x?xf32>

    %input_width_i32 = arith.index_cast %input_width : index to i32
    %input_height_i32 = arith.index_cast %input_height : index to i32
    %channels_i32 = arith.index_cast %channels : index to i32
    %kernel_width_i32 = arith.index_cast %kernel_width : index to i32
    %kernel_height_i32 = arith.index_cast %kernel_height : index to i32

    // The PPU pools in fp16; the model's tensor is f32.
    %input_f16_empty = tensor.empty(%input_height, %input_width, %channels) : tensor<1x?x?x?xf16>
    %input_f16 = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } ins(%input : tensor<1x?x?x?xf32>)
        outs(%input_f16_empty : tensor<1x?x?x?xf16>) {
      ^bb0(%value: f32, %out: f16):
        %narrowed = arith.truncf %value : f32 to f16
        linalg.yield %narrowed : f16
    } -> tensor<1x?x?x?xf16>

    %pooled = flow.dispatch
        @rocket_pooling_max_executable::@rocket_pooling_max_v1::@rocket_pooling_max(
          %input_width_i32, %input_height_i32, %channels_i32,
          %kernel_width_i32, %kernel_height_i32,
          %input_f16)
        {stream.affinity = #hal.device.affinity<@rocket_device>}
        : (i32, i32, i32, i32, i32,
           tensor<1x?x?x?xf16>{%input_height, %input_width, %channels})
        -> tensor<1x?x?x?xf16>{%output_height, %output_width, %channels}

    // Widen back to f32 and fold in the accumulator initialiser. linalg
    // defines a max pool as `O = max(O, I)`, so the init is part of the
    // reduction rather than merely a destination, and dropping it would
    // silently change the answer for any model that seeds it with something
    // other than -inf.
    //
    // Explicitly a CPU dispatch, for the reason @call_rocket_pooling_avg_nchw
    // records: an op that consumes the Rocket result inherits its affinity,
    // gets formed into an executable for the rocket device, and that
    // executable has no pooling config to serialize. The failure is a
    // serialization error naming a missing `input_width`, a long way from
    // the cause.
    %final_nhwc = flow.dispatch.workgroups[
        %output_height, %output_width, %channels](
        %pooled, %init, %output_height, %output_width, %channels)
        : (tensor<1x?x?x?xf16>{%output_height, %output_width, %channels},
           tensor<1x?x?x?xf32>{%output_height, %output_width, %channels},
           index, index, index)
        -> tensor<1x?x?x?xf32>{%output_height, %output_width, %channels}
        attributes { stream.affinity = #hal.device.affinity<@cpu_device> } =
        (%pooled_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>,
         %init_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>,
         %output_height_arg: index,
         %output_width_arg: index,
         %channels_arg: index,
         %final_binding: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>) {
      %output_height_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_height_arg, 0 : index
      %output_width_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_width_arg, 1 : index
      %channels_size = iree_tensor_ext.dispatch.workload.ordinal
          %channels_arg, 2 : index
      %pooled_shaped = flow.dispatch.tie_shape %pooled_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %channels_size}
      %init_shaped = flow.dispatch.tie_shape %init_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %channels_size}
      %final_shaped = flow.dispatch.tie_shape %final_binding
          : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %channels_size}
      %pooled_loaded = iree_tensor_ext.dispatch.tensor.load %pooled_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %channels_size}
          -> tensor<1x?x?x?xf16>
      %init_loaded = iree_tensor_ext.dispatch.tensor.load %init_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %channels_size}
          -> tensor<1x?x?x?xf32>
      %final_empty = tensor.empty(
          %output_height_size, %output_width_size, %channels_size)
          : tensor<1x?x?x?xf32>
      %final_inner = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
          ],
          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } ins(%pooled_loaded, %init_loaded
            : tensor<1x?x?x?xf16>, tensor<1x?x?x?xf32>)
          outs(%final_empty : tensor<1x?x?x?xf32>) {
        ^bb0(%pooled_value: f16, %initial: f32, %out: f32):
          %pooled_f32 = arith.extf %pooled_value : f16 to f32
          %accumulated = arith.maximumf %pooled_f32, %initial : f32
          linalg.yield %accumulated : f32
      } -> tensor<1x?x?x?xf32>
      iree_tensor_ext.dispatch.tensor.store %final_inner, %final_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %channels_size],
          strides = [1, 1, 1, 1]
          : tensor<1x?x?x?xf32>
          -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %channels_size}
      flow.return
    } count(%output_height_workload: index,
            %output_width_workload: index,
            %channels_workload: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(
          %output_height_workload,
          %output_width_workload,
          %channels_workload)
      flow.return %x, %y, %z : index, index, index
    }

    util.return %final_nhwc : tensor<1x?x?x?xf32>
  }

  // Max pool, NHWC, stride 2 -- the shape a real model actually has. Identical
  // to the stride-1 shim but for the executable it dispatches to; stride is
  // baked per executable, matching the convolution variants.
  util.func private @call_rocket_pooling_max_nhwc_s2(
      %input: tensor<1x?x?x?xf32>,
      %window: tensor<?x?xf32>,
      %init: tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    // NHWC: dims 1 and 2 are the spatial extent, dim 3 is channels.
    %input_height = tensor.dim %input, %c1 : tensor<1x?x?x?xf32>
    %input_width = tensor.dim %input, %c2 : tensor<1x?x?x?xf32>
    %channels = tensor.dim %input, %c3 : tensor<1x?x?x?xf32>
    // The window operand carries no values, only [kh, kw].
    %kernel_height = tensor.dim %window, %c0 : tensor<?x?xf32>
    %kernel_width = tensor.dim %window, %c1 : tensor<?x?xf32>
    %output_height = tensor.dim %init, %c1 : tensor<1x?x?x?xf32>
    %output_width = tensor.dim %init, %c2 : tensor<1x?x?x?xf32>

    %input_width_i32 = arith.index_cast %input_width : index to i32
    %input_height_i32 = arith.index_cast %input_height : index to i32
    %channels_i32 = arith.index_cast %channels : index to i32
    %kernel_width_i32 = arith.index_cast %kernel_width : index to i32
    %kernel_height_i32 = arith.index_cast %kernel_height : index to i32

    // The PPU pools in fp16; the model's tensor is f32.
    %input_f16_empty = tensor.empty(%input_height, %input_width, %channels) : tensor<1x?x?x?xf16>
    %input_f16 = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } ins(%input : tensor<1x?x?x?xf32>)
        outs(%input_f16_empty : tensor<1x?x?x?xf16>) {
      ^bb0(%value: f32, %out: f16):
        %narrowed = arith.truncf %value : f32 to f16
        linalg.yield %narrowed : f16
    } -> tensor<1x?x?x?xf16>

    %pooled = flow.dispatch
        @rocket_pooling_max_executable_s2::@rocket_pooling_max_v1::@rocket_pooling_max(
          %input_width_i32, %input_height_i32, %channels_i32,
          %kernel_width_i32, %kernel_height_i32,
          %input_f16)
        {stream.affinity = #hal.device.affinity<@rocket_device>}
        : (i32, i32, i32, i32, i32,
           tensor<1x?x?x?xf16>{%input_height, %input_width, %channels})
        -> tensor<1x?x?x?xf16>{%output_height, %output_width, %channels}

    // Widen back to f32 and fold in the accumulator initialiser. linalg
    // defines a max pool as `O = max(O, I)`, so the init is part of the
    // reduction rather than merely a destination, and dropping it would
    // silently change the answer for any model that seeds it with something
    // other than -inf.
    //
    // Explicitly a CPU dispatch, for the reason @call_rocket_pooling_avg_nchw
    // records: an op that consumes the Rocket result inherits its affinity,
    // gets formed into an executable for the rocket device, and that
    // executable has no pooling config to serialize. The failure is a
    // serialization error naming a missing `input_width`, a long way from
    // the cause.
    %final_nhwc = flow.dispatch.workgroups[
        %output_height, %output_width, %channels](
        %pooled, %init, %output_height, %output_width, %channels)
        : (tensor<1x?x?x?xf16>{%output_height, %output_width, %channels},
           tensor<1x?x?x?xf32>{%output_height, %output_width, %channels},
           index, index, index)
        -> tensor<1x?x?x?xf32>{%output_height, %output_width, %channels}
        attributes { stream.affinity = #hal.device.affinity<@cpu_device> } =
        (%pooled_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>,
         %init_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>,
         %output_height_arg: index,
         %output_width_arg: index,
         %channels_arg: index,
         %final_binding: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>) {
      %output_height_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_height_arg, 0 : index
      %output_width_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_width_arg, 1 : index
      %channels_size = iree_tensor_ext.dispatch.workload.ordinal
          %channels_arg, 2 : index
      %pooled_shaped = flow.dispatch.tie_shape %pooled_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %channels_size}
      %init_shaped = flow.dispatch.tie_shape %init_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %channels_size}
      %final_shaped = flow.dispatch.tie_shape %final_binding
          : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %channels_size}
      %pooled_loaded = iree_tensor_ext.dispatch.tensor.load %pooled_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %channels_size}
          -> tensor<1x?x?x?xf16>
      %init_loaded = iree_tensor_ext.dispatch.tensor.load %init_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %channels_size}
          -> tensor<1x?x?x?xf32>
      %final_empty = tensor.empty(
          %output_height_size, %output_width_size, %channels_size)
          : tensor<1x?x?x?xf32>
      %final_inner = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
          ],
          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } ins(%pooled_loaded, %init_loaded
            : tensor<1x?x?x?xf16>, tensor<1x?x?x?xf32>)
          outs(%final_empty : tensor<1x?x?x?xf32>) {
        ^bb0(%pooled_value: f16, %initial: f32, %out: f32):
          %pooled_f32 = arith.extf %pooled_value : f16 to f32
          %accumulated = arith.maximumf %pooled_f32, %initial : f32
          linalg.yield %accumulated : f32
      } -> tensor<1x?x?x?xf32>
      iree_tensor_ext.dispatch.tensor.store %final_inner, %final_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %channels_size],
          strides = [1, 1, 1, 1]
          : tensor<1x?x?x?xf32>
          -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %channels_size}
      flow.return
    } count(%output_height_workload: index,
            %output_width_workload: index,
            %channels_workload: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(
          %output_height_workload,
          %output_width_workload,
          %channels_workload)
      flow.return %x, %y, %z : index, index, index
    }

    util.return %final_nhwc : tensor<1x?x?x?xf32>
  }

  // Max pool, NCHW, stride 1. ONNX imports max pools NCHW, and
  // iree-preprocessing-convert-conv-to-channels-last does not touch pooling
  // ops -- verified here: a linalg.pooling_nchw_max survives the spec's
  // preprocessing unchanged. So this layout needs its own shim rather than
  // being normalized into the NHWC one above.
  util.func private @call_rocket_pooling_max_nchw(
      %input: tensor<1x?x?x?xf32>,
      %window: tensor<?x?xf32>,
      %init: tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    // NCHW: dim 1 is channels, dims 2 and 3 are the spatial extent.
    %channels = tensor.dim %input, %c1 : tensor<1x?x?x?xf32>
    %input_height = tensor.dim %input, %c2 : tensor<1x?x?x?xf32>
    %input_width = tensor.dim %input, %c3 : tensor<1x?x?x?xf32>
    // The window operand carries no values, only [kh, kw].
    %kernel_height = tensor.dim %window, %c0 : tensor<?x?xf32>
    %kernel_width = tensor.dim %window, %c1 : tensor<?x?xf32>
    %output_height = tensor.dim %init, %c2 : tensor<1x?x?x?xf32>
    %output_width = tensor.dim %init, %c3 : tensor<1x?x?x?xf32>

    %input_width_i32 = arith.index_cast %input_width : index to i32
    %input_height_i32 = arith.index_cast %input_height : index to i32
    %channels_i32 = arith.index_cast %channels : index to i32
    %kernel_width_i32 = arith.index_cast %kernel_width : index to i32
    %kernel_height_i32 = arith.index_cast %kernel_height : index to i32

    // NCHW [1,C,H,W] -> NHWC [1,H,W,C]: out.shape[i] = in.shape[perm[i]].
    %input_nhwc_empty = tensor.empty(%input_height, %input_width, %channels) : tensor<1x?x?x?xf32>
    %input_nhwc = linalg.transpose
        ins(%input : tensor<1x?x?x?xf32>)
        outs(%input_nhwc_empty : tensor<1x?x?x?xf32>)
        permutation = [0, 2, 3, 1]

    // The PPU pools in fp16; the model's tensor is f32.
    %input_f16_empty = tensor.empty(%input_height, %input_width, %channels) : tensor<1x?x?x?xf16>
    %input_f16 = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } ins(%input_nhwc : tensor<1x?x?x?xf32>)
        outs(%input_f16_empty : tensor<1x?x?x?xf16>) {
      ^bb0(%value: f32, %out: f16):
        %narrowed = arith.truncf %value : f32 to f16
        linalg.yield %narrowed : f16
    } -> tensor<1x?x?x?xf16>

    %pooled = flow.dispatch
        @rocket_pooling_max_executable::@rocket_pooling_max_v1::@rocket_pooling_max(
          %input_width_i32, %input_height_i32, %channels_i32,
          %kernel_width_i32, %kernel_height_i32,
          %input_f16)
        {stream.affinity = #hal.device.affinity<@rocket_device>}
        : (i32, i32, i32, i32, i32,
           tensor<1x?x?x?xf16>{%input_height, %input_width, %channels})
        -> tensor<1x?x?x?xf16>{%output_height, %output_width, %channels}

    // %init arrives NCHW too; line it up with the NHWC result.
    %init_nhwc_empty = tensor.empty(%output_height, %output_width, %channels) : tensor<1x?x?x?xf32>
    %init_nhwc = linalg.transpose
        ins(%init : tensor<1x?x?x?xf32>)
        outs(%init_nhwc_empty : tensor<1x?x?x?xf32>)
        permutation = [0, 2, 3, 1]

    // Widen back to f32 and fold in the accumulator initialiser. linalg
    // defines a max pool as `O = max(O, I)`, so the init is part of the
    // reduction rather than merely a destination, and dropping it would
    // silently change the answer for any model that seeds it with something
    // other than -inf.
    //
    // Explicitly a CPU dispatch, for the reason @call_rocket_pooling_avg_nchw
    // records: an op that consumes the Rocket result inherits its affinity,
    // gets formed into an executable for the rocket device, and that
    // executable has no pooling config to serialize. The failure is a
    // serialization error naming a missing `input_width`, a long way from
    // the cause.
    %final_nhwc = flow.dispatch.workgroups[
        %output_height, %output_width, %channels](
        %pooled, %init_nhwc, %output_height, %output_width, %channels)
        : (tensor<1x?x?x?xf16>{%output_height, %output_width, %channels},
           tensor<1x?x?x?xf32>{%output_height, %output_width, %channels},
           index, index, index)
        -> tensor<1x?x?x?xf32>{%output_height, %output_width, %channels}
        attributes { stream.affinity = #hal.device.affinity<@cpu_device> } =
        (%pooled_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>,
         %init_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>,
         %output_height_arg: index,
         %output_width_arg: index,
         %channels_arg: index,
         %final_binding: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>) {
      %output_height_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_height_arg, 0 : index
      %output_width_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_width_arg, 1 : index
      %channels_size = iree_tensor_ext.dispatch.workload.ordinal
          %channels_arg, 2 : index
      %pooled_shaped = flow.dispatch.tie_shape %pooled_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %channels_size}
      %init_shaped = flow.dispatch.tie_shape %init_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %channels_size}
      %final_shaped = flow.dispatch.tie_shape %final_binding
          : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %channels_size}
      %pooled_loaded = iree_tensor_ext.dispatch.tensor.load %pooled_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %channels_size}
          -> tensor<1x?x?x?xf16>
      %init_loaded = iree_tensor_ext.dispatch.tensor.load %init_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %channels_size}
          -> tensor<1x?x?x?xf32>
      %final_empty = tensor.empty(
          %output_height_size, %output_width_size, %channels_size)
          : tensor<1x?x?x?xf32>
      %final_inner = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
          ],
          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } ins(%pooled_loaded, %init_loaded
            : tensor<1x?x?x?xf16>, tensor<1x?x?x?xf32>)
          outs(%final_empty : tensor<1x?x?x?xf32>) {
        ^bb0(%pooled_value: f16, %initial: f32, %out: f32):
          %pooled_f32 = arith.extf %pooled_value : f16 to f32
          %accumulated = arith.maximumf %pooled_f32, %initial : f32
          linalg.yield %accumulated : f32
      } -> tensor<1x?x?x?xf32>
      iree_tensor_ext.dispatch.tensor.store %final_inner, %final_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %channels_size],
          strides = [1, 1, 1, 1]
          : tensor<1x?x?x?xf32>
          -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %channels_size}
      flow.return
    } count(%output_height_workload: index,
            %output_width_workload: index,
            %channels_workload: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(
          %output_height_workload,
          %output_width_workload,
          %channels_workload)
      flow.return %x, %y, %z : index, index, index
    }

    // NHWC [1,H,W,C] -> NCHW [1,C,H,W].
    %final_nchw_empty = tensor.empty(%channels, %output_height, %output_width) : tensor<1x?x?x?xf32>
    %final_nchw = linalg.transpose
        ins(%final_nhwc : tensor<1x?x?x?xf32>)
        outs(%final_nchw_empty : tensor<1x?x?x?xf32>)
        permutation = [0, 3, 1, 2]

    util.return %final_nchw : tensor<1x?x?x?xf32>
  }

  // Max pool, NCHW, stride 2.
  util.func private @call_rocket_pooling_max_nchw_s2(
      %input: tensor<1x?x?x?xf32>,
      %window: tensor<?x?xf32>,
      %init: tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    // NCHW: dim 1 is channels, dims 2 and 3 are the spatial extent.
    %channels = tensor.dim %input, %c1 : tensor<1x?x?x?xf32>
    %input_height = tensor.dim %input, %c2 : tensor<1x?x?x?xf32>
    %input_width = tensor.dim %input, %c3 : tensor<1x?x?x?xf32>
    // The window operand carries no values, only [kh, kw].
    %kernel_height = tensor.dim %window, %c0 : tensor<?x?xf32>
    %kernel_width = tensor.dim %window, %c1 : tensor<?x?xf32>
    %output_height = tensor.dim %init, %c2 : tensor<1x?x?x?xf32>
    %output_width = tensor.dim %init, %c3 : tensor<1x?x?x?xf32>

    %input_width_i32 = arith.index_cast %input_width : index to i32
    %input_height_i32 = arith.index_cast %input_height : index to i32
    %channels_i32 = arith.index_cast %channels : index to i32
    %kernel_width_i32 = arith.index_cast %kernel_width : index to i32
    %kernel_height_i32 = arith.index_cast %kernel_height : index to i32

    // NCHW [1,C,H,W] -> NHWC [1,H,W,C]: out.shape[i] = in.shape[perm[i]].
    %input_nhwc_empty = tensor.empty(%input_height, %input_width, %channels) : tensor<1x?x?x?xf32>
    %input_nhwc = linalg.transpose
        ins(%input : tensor<1x?x?x?xf32>)
        outs(%input_nhwc_empty : tensor<1x?x?x?xf32>)
        permutation = [0, 2, 3, 1]

    // The PPU pools in fp16; the model's tensor is f32.
    %input_f16_empty = tensor.empty(%input_height, %input_width, %channels) : tensor<1x?x?x?xf16>
    %input_f16 = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } ins(%input_nhwc : tensor<1x?x?x?xf32>)
        outs(%input_f16_empty : tensor<1x?x?x?xf16>) {
      ^bb0(%value: f32, %out: f16):
        %narrowed = arith.truncf %value : f32 to f16
        linalg.yield %narrowed : f16
    } -> tensor<1x?x?x?xf16>

    %pooled = flow.dispatch
        @rocket_pooling_max_executable_s2::@rocket_pooling_max_v1::@rocket_pooling_max(
          %input_width_i32, %input_height_i32, %channels_i32,
          %kernel_width_i32, %kernel_height_i32,
          %input_f16)
        {stream.affinity = #hal.device.affinity<@rocket_device>}
        : (i32, i32, i32, i32, i32,
           tensor<1x?x?x?xf16>{%input_height, %input_width, %channels})
        -> tensor<1x?x?x?xf16>{%output_height, %output_width, %channels}

    // %init arrives NCHW too; line it up with the NHWC result.
    %init_nhwc_empty = tensor.empty(%output_height, %output_width, %channels) : tensor<1x?x?x?xf32>
    %init_nhwc = linalg.transpose
        ins(%init : tensor<1x?x?x?xf32>)
        outs(%init_nhwc_empty : tensor<1x?x?x?xf32>)
        permutation = [0, 2, 3, 1]

    // Widen back to f32 and fold in the accumulator initialiser. linalg
    // defines a max pool as `O = max(O, I)`, so the init is part of the
    // reduction rather than merely a destination, and dropping it would
    // silently change the answer for any model that seeds it with something
    // other than -inf.
    //
    // Explicitly a CPU dispatch, for the reason @call_rocket_pooling_avg_nchw
    // records: an op that consumes the Rocket result inherits its affinity,
    // gets formed into an executable for the rocket device, and that
    // executable has no pooling config to serialize. The failure is a
    // serialization error naming a missing `input_width`, a long way from
    // the cause.
    %final_nhwc = flow.dispatch.workgroups[
        %output_height, %output_width, %channels](
        %pooled, %init_nhwc, %output_height, %output_width, %channels)
        : (tensor<1x?x?x?xf16>{%output_height, %output_width, %channels},
           tensor<1x?x?x?xf32>{%output_height, %output_width, %channels},
           index, index, index)
        -> tensor<1x?x?x?xf32>{%output_height, %output_width, %channels}
        attributes { stream.affinity = #hal.device.affinity<@cpu_device> } =
        (%pooled_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>,
         %init_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>,
         %output_height_arg: index,
         %output_width_arg: index,
         %channels_arg: index,
         %final_binding: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>) {
      %output_height_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_height_arg, 0 : index
      %output_width_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_width_arg, 1 : index
      %channels_size = iree_tensor_ext.dispatch.workload.ordinal
          %channels_arg, 2 : index
      %pooled_shaped = flow.dispatch.tie_shape %pooled_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %channels_size}
      %init_shaped = flow.dispatch.tie_shape %init_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %channels_size}
      %final_shaped = flow.dispatch.tie_shape %final_binding
          : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %channels_size}
      %pooled_loaded = iree_tensor_ext.dispatch.tensor.load %pooled_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %channels_size}
          -> tensor<1x?x?x?xf16>
      %init_loaded = iree_tensor_ext.dispatch.tensor.load %init_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %channels_size}
          -> tensor<1x?x?x?xf32>
      %final_empty = tensor.empty(
          %output_height_size, %output_width_size, %channels_size)
          : tensor<1x?x?x?xf32>
      %final_inner = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
          ],
          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } ins(%pooled_loaded, %init_loaded
            : tensor<1x?x?x?xf16>, tensor<1x?x?x?xf32>)
          outs(%final_empty : tensor<1x?x?x?xf32>) {
        ^bb0(%pooled_value: f16, %initial: f32, %out: f32):
          %pooled_f32 = arith.extf %pooled_value : f16 to f32
          %accumulated = arith.maximumf %pooled_f32, %initial : f32
          linalg.yield %accumulated : f32
      } -> tensor<1x?x?x?xf32>
      iree_tensor_ext.dispatch.tensor.store %final_inner, %final_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %channels_size],
          strides = [1, 1, 1, 1]
          : tensor<1x?x?x?xf32>
          -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %channels_size}
      flow.return
    } count(%output_height_workload: index,
            %output_width_workload: index,
            %channels_workload: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(
          %output_height_workload,
          %output_width_workload,
          %channels_workload)
      flow.return %x, %y, %z : index, index, index
    }

    // NHWC [1,H,W,C] -> NCHW [1,C,H,W].
    %final_nchw_empty = tensor.empty(%channels, %output_height, %output_width) : tensor<1x?x?x?xf32>
    %final_nchw = linalg.transpose
        ins(%final_nhwc : tensor<1x?x?x?xf32>)
        outs(%final_nchw_empty : tensor<1x?x?x?xf32>)
        permutation = [0, 3, 1, 2]

    util.return %final_nchw : tensor<1x?x?x?xf32>
  }

  // Min pool, NHWC, stride 1. Derived from the max shim above and identical
  // to it but for the reduction it dispatches and the `arith.minimumf` that
  // folds in the initialiser -- linalg defines a min pool as `O = min(O, I)`,
  // confirmed by generalizing the named op.
  util.func private @call_rocket_pooling_min_nhwc(
      %input: tensor<1x?x?x?xf32>,
      %window: tensor<?x?xf32>,
      %init: tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    // NHWC: dims 1 and 2 are the spatial extent, dim 3 is channels.
    %input_height = tensor.dim %input, %c1 : tensor<1x?x?x?xf32>
    %input_width = tensor.dim %input, %c2 : tensor<1x?x?x?xf32>
    %channels = tensor.dim %input, %c3 : tensor<1x?x?x?xf32>
    // The window operand carries no values, only [kh, kw].
    %kernel_height = tensor.dim %window, %c0 : tensor<?x?xf32>
    %kernel_width = tensor.dim %window, %c1 : tensor<?x?xf32>
    %output_height = tensor.dim %init, %c1 : tensor<1x?x?x?xf32>
    %output_width = tensor.dim %init, %c2 : tensor<1x?x?x?xf32>

    %input_width_i32 = arith.index_cast %input_width : index to i32
    %input_height_i32 = arith.index_cast %input_height : index to i32
    %channels_i32 = arith.index_cast %channels : index to i32
    %kernel_width_i32 = arith.index_cast %kernel_width : index to i32
    %kernel_height_i32 = arith.index_cast %kernel_height : index to i32

    // The PPU pools in fp16; the model's tensor is f32.
    %input_f16_empty = tensor.empty(%input_height, %input_width, %channels) : tensor<1x?x?x?xf16>
    %input_f16 = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } ins(%input : tensor<1x?x?x?xf32>)
        outs(%input_f16_empty : tensor<1x?x?x?xf16>) {
      ^bb0(%value: f32, %out: f16):
        %narrowed = arith.truncf %value : f32 to f16
        linalg.yield %narrowed : f16
    } -> tensor<1x?x?x?xf16>

    %pooled = flow.dispatch
        @rocket_pooling_min_executable::@rocket_pooling_min_v1::@rocket_pooling_min(
          %input_width_i32, %input_height_i32, %channels_i32,
          %kernel_width_i32, %kernel_height_i32,
          %input_f16)
        {stream.affinity = #hal.device.affinity<@rocket_device>}
        : (i32, i32, i32, i32, i32,
           tensor<1x?x?x?xf16>{%input_height, %input_width, %channels})
        -> tensor<1x?x?x?xf16>{%output_height, %output_width, %channels}

    // Widen back to f32 and fold in the accumulator initialiser. linalg
    // defines a min pool as `O = min(O, I)`, so the init is part of the
    // reduction rather than merely a destination, and dropping it would
    // silently change the answer for any model that seeds it with something
    // other than +inf.
    //
    // Explicitly a CPU dispatch, for the reason @call_rocket_pooling_avg_nchw
    // records: an op that consumes the Rocket result inherits its affinity,
    // gets formed into an executable for the rocket device, and that
    // executable has no pooling config to serialize. The failure is a
    // serialization error naming a missing `input_width`, a long way from
    // the cause.
    %final_nhwc = flow.dispatch.workgroups[
        %output_height, %output_width, %channels](
        %pooled, %init, %output_height, %output_width, %channels)
        : (tensor<1x?x?x?xf16>{%output_height, %output_width, %channels},
           tensor<1x?x?x?xf32>{%output_height, %output_width, %channels},
           index, index, index)
        -> tensor<1x?x?x?xf32>{%output_height, %output_width, %channels}
        attributes { stream.affinity = #hal.device.affinity<@cpu_device> } =
        (%pooled_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>,
         %init_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>,
         %output_height_arg: index,
         %output_width_arg: index,
         %channels_arg: index,
         %final_binding: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>) {
      %output_height_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_height_arg, 0 : index
      %output_width_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_width_arg, 1 : index
      %channels_size = iree_tensor_ext.dispatch.workload.ordinal
          %channels_arg, 2 : index
      %pooled_shaped = flow.dispatch.tie_shape %pooled_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %channels_size}
      %init_shaped = flow.dispatch.tie_shape %init_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %channels_size}
      %final_shaped = flow.dispatch.tie_shape %final_binding
          : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %channels_size}
      %pooled_loaded = iree_tensor_ext.dispatch.tensor.load %pooled_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %channels_size}
          -> tensor<1x?x?x?xf16>
      %init_loaded = iree_tensor_ext.dispatch.tensor.load %init_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %channels_size}
          -> tensor<1x?x?x?xf32>
      %final_empty = tensor.empty(
          %output_height_size, %output_width_size, %channels_size)
          : tensor<1x?x?x?xf32>
      %final_inner = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
          ],
          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } ins(%pooled_loaded, %init_loaded
            : tensor<1x?x?x?xf16>, tensor<1x?x?x?xf32>)
          outs(%final_empty : tensor<1x?x?x?xf32>) {
        ^bb0(%pooled_value: f16, %initial: f32, %out: f32):
          %pooled_f32 = arith.extf %pooled_value : f16 to f32
          %accumulated = arith.minimumf %pooled_f32, %initial : f32
          linalg.yield %accumulated : f32
      } -> tensor<1x?x?x?xf32>
      iree_tensor_ext.dispatch.tensor.store %final_inner, %final_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %channels_size],
          strides = [1, 1, 1, 1]
          : tensor<1x?x?x?xf32>
          -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %channels_size}
      flow.return
    } count(%output_height_workload: index,
            %output_width_workload: index,
            %channels_workload: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(
          %output_height_workload,
          %output_width_workload,
          %channels_workload)
      flow.return %x, %y, %z : index, index, index
    }

    util.return %final_nhwc : tensor<1x?x?x?xf32>
  }

  // Min pool, NHWC, stride 2.
  util.func private @call_rocket_pooling_min_nhwc_s2(
      %input: tensor<1x?x?x?xf32>,
      %window: tensor<?x?xf32>,
      %init: tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    // NHWC: dims 1 and 2 are the spatial extent, dim 3 is channels.
    %input_height = tensor.dim %input, %c1 : tensor<1x?x?x?xf32>
    %input_width = tensor.dim %input, %c2 : tensor<1x?x?x?xf32>
    %channels = tensor.dim %input, %c3 : tensor<1x?x?x?xf32>
    // The window operand carries no values, only [kh, kw].
    %kernel_height = tensor.dim %window, %c0 : tensor<?x?xf32>
    %kernel_width = tensor.dim %window, %c1 : tensor<?x?xf32>
    %output_height = tensor.dim %init, %c1 : tensor<1x?x?x?xf32>
    %output_width = tensor.dim %init, %c2 : tensor<1x?x?x?xf32>

    %input_width_i32 = arith.index_cast %input_width : index to i32
    %input_height_i32 = arith.index_cast %input_height : index to i32
    %channels_i32 = arith.index_cast %channels : index to i32
    %kernel_width_i32 = arith.index_cast %kernel_width : index to i32
    %kernel_height_i32 = arith.index_cast %kernel_height : index to i32

    // The PPU pools in fp16; the model's tensor is f32.
    %input_f16_empty = tensor.empty(%input_height, %input_width, %channels) : tensor<1x?x?x?xf16>
    %input_f16 = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } ins(%input : tensor<1x?x?x?xf32>)
        outs(%input_f16_empty : tensor<1x?x?x?xf16>) {
      ^bb0(%value: f32, %out: f16):
        %narrowed = arith.truncf %value : f32 to f16
        linalg.yield %narrowed : f16
    } -> tensor<1x?x?x?xf16>

    %pooled = flow.dispatch
        @rocket_pooling_min_executable_s2::@rocket_pooling_min_v1::@rocket_pooling_min(
          %input_width_i32, %input_height_i32, %channels_i32,
          %kernel_width_i32, %kernel_height_i32,
          %input_f16)
        {stream.affinity = #hal.device.affinity<@rocket_device>}
        : (i32, i32, i32, i32, i32,
           tensor<1x?x?x?xf16>{%input_height, %input_width, %channels})
        -> tensor<1x?x?x?xf16>{%output_height, %output_width, %channels}

    // Widen back to f32 and fold in the accumulator initialiser. linalg
    // defines a min pool as `O = min(O, I)`, so the init is part of the
    // reduction rather than merely a destination, and dropping it would
    // silently change the answer for any model that seeds it with something
    // other than +inf.
    //
    // Explicitly a CPU dispatch, for the reason @call_rocket_pooling_avg_nchw
    // records: an op that consumes the Rocket result inherits its affinity,
    // gets formed into an executable for the rocket device, and that
    // executable has no pooling config to serialize. The failure is a
    // serialization error naming a missing `input_width`, a long way from
    // the cause.
    %final_nhwc = flow.dispatch.workgroups[
        %output_height, %output_width, %channels](
        %pooled, %init, %output_height, %output_width, %channels)
        : (tensor<1x?x?x?xf16>{%output_height, %output_width, %channels},
           tensor<1x?x?x?xf32>{%output_height, %output_width, %channels},
           index, index, index)
        -> tensor<1x?x?x?xf32>{%output_height, %output_width, %channels}
        attributes { stream.affinity = #hal.device.affinity<@cpu_device> } =
        (%pooled_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>,
         %init_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>,
         %output_height_arg: index,
         %output_width_arg: index,
         %channels_arg: index,
         %final_binding: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>) {
      %output_height_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_height_arg, 0 : index
      %output_width_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_width_arg, 1 : index
      %channels_size = iree_tensor_ext.dispatch.workload.ordinal
          %channels_arg, 2 : index
      %pooled_shaped = flow.dispatch.tie_shape %pooled_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %channels_size}
      %init_shaped = flow.dispatch.tie_shape %init_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %channels_size}
      %final_shaped = flow.dispatch.tie_shape %final_binding
          : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %channels_size}
      %pooled_loaded = iree_tensor_ext.dispatch.tensor.load %pooled_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %channels_size}
          -> tensor<1x?x?x?xf16>
      %init_loaded = iree_tensor_ext.dispatch.tensor.load %init_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %channels_size}
          -> tensor<1x?x?x?xf32>
      %final_empty = tensor.empty(
          %output_height_size, %output_width_size, %channels_size)
          : tensor<1x?x?x?xf32>
      %final_inner = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
          ],
          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } ins(%pooled_loaded, %init_loaded
            : tensor<1x?x?x?xf16>, tensor<1x?x?x?xf32>)
          outs(%final_empty : tensor<1x?x?x?xf32>) {
        ^bb0(%pooled_value: f16, %initial: f32, %out: f32):
          %pooled_f32 = arith.extf %pooled_value : f16 to f32
          %accumulated = arith.minimumf %pooled_f32, %initial : f32
          linalg.yield %accumulated : f32
      } -> tensor<1x?x?x?xf32>
      iree_tensor_ext.dispatch.tensor.store %final_inner, %final_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %channels_size],
          strides = [1, 1, 1, 1]
          : tensor<1x?x?x?xf32>
          -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %channels_size}
      flow.return
    } count(%output_height_workload: index,
            %output_width_workload: index,
            %channels_workload: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(
          %output_height_workload,
          %output_width_workload,
          %channels_workload)
      flow.return %x, %y, %z : index, index, index
    }

    util.return %final_nhwc : tensor<1x?x?x?xf32>
  }


  // Average pool, NHWC. The layout counterpart of @call_rocket_pooling_avg_nchw
  // above, and the same operation: run the hardware's *average* and multiply
  // it back up by kh*kw, because linalg has no average pool and the PPU has no
  // sum mode. What NHWC drops is the three transposes -- input, initialiser
  // and result -- since this is already the layout the hardware wants.
  //
  // It shares @rocket_pooling_executable with the NCHW shim rather than
  // getting one of its own: the executable takes NC1HWC2 cubes and knows
  // nothing about the logical layout its caller started from, so only the
  // shim differs.
  util.func private @call_rocket_pooling_avg_nhwc(
      %input: tensor<1x?x?x?xf32>,
      %window: tensor<?x?xf32>,
      %init: tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    // NHWC: dims 1 and 2 are the spatial extent, dim 3 is channels.
    %input_height = tensor.dim %input, %c1 : tensor<1x?x?x?xf32>
    %input_width = tensor.dim %input, %c2 : tensor<1x?x?x?xf32>
    %channels = tensor.dim %input, %c3 : tensor<1x?x?x?xf32>
    // The window operand carries no values, only [kh, kw].
    %kernel_height = tensor.dim %window, %c0 : tensor<?x?xf32>
    %kernel_width = tensor.dim %window, %c1 : tensor<?x?xf32>
    %output_height = tensor.dim %init, %c1 : tensor<1x?x?x?xf32>
    %output_width = tensor.dim %init, %c2 : tensor<1x?x?x?xf32>

    %input_width_i32 = arith.index_cast %input_width : index to i32
    %input_height_i32 = arith.index_cast %input_height : index to i32
    %channels_i32 = arith.index_cast %channels : index to i32
    %kernel_width_i32 = arith.index_cast %kernel_width : index to i32
    %kernel_height_i32 = arith.index_cast %kernel_height : index to i32

    // The PPU pools in fp16; the model's tensor is f32.
    %input_f16_empty = tensor.empty(%input_height, %input_width, %channels) : tensor<1x?x?x?xf16>
    %input_f16 = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } ins(%input : tensor<1x?x?x?xf32>)
        outs(%input_f16_empty : tensor<1x?x?x?xf16>) {
      ^bb0(%value: f32, %out: f16):
        %narrowed = arith.truncf %value : f32 to f16
        linalg.yield %narrowed : f16
    } -> tensor<1x?x?x?xf16>

    %averaged = flow.dispatch
        @rocket_pooling_executable::@rocket_pooling_avg_v1::@rocket_pooling_avg(
          %input_width_i32, %input_height_i32, %channels_i32,
          %kernel_width_i32, %kernel_height_i32,
          %input_f16)
        {stream.affinity = #hal.device.affinity<@rocket_device>}
        : (i32, i32, i32, i32, i32,
           tensor<1x?x?x?xf16>{%input_height, %input_width, %channels})
        -> tensor<1x?x?x?xf16>{%output_height, %output_width, %channels}

    %taps = arith.muli %kernel_height, %kernel_width : index

    %final_nhwc = flow.dispatch.workgroups[
        %output_height, %output_width, %channels](
        %averaged, %init, %taps, %output_height, %output_width, %channels)
        : (tensor<1x?x?x?xf16>{%output_height, %output_width, %channels},
           tensor<1x?x?x?xf32>{%output_height, %output_width, %channels},
           index, index, index, index)
        -> tensor<1x?x?x?xf32>{%output_height, %output_width, %channels}
        attributes { stream.affinity = #hal.device.affinity<@cpu_device> } =
        (%averaged_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>,
         %init_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>,
         %taps_arg: index,
         %output_height_arg: index,
         %output_width_arg: index,
         %channels_arg: index,
         %final_binding: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>) {
      %output_height_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_height_arg, 0 : index
      %output_width_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_width_arg, 1 : index
      %channels_size = iree_tensor_ext.dispatch.workload.ordinal
          %channels_arg, 2 : index
      %averaged_shaped = flow.dispatch.tie_shape %averaged_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %channels_size}
      %init_shaped = flow.dispatch.tie_shape %init_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %channels_size}
      %final_shaped = flow.dispatch.tie_shape %final_binding
          : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %channels_size}
      %averaged_loaded = iree_tensor_ext.dispatch.tensor.load %averaged_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %channels_size}
          -> tensor<1x?x?x?xf16>
      %init_loaded = iree_tensor_ext.dispatch.tensor.load %init_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %channels_size}
          -> tensor<1x?x?x?xf32>
      %taps_i32 = arith.index_cast %taps_arg : index to i32
      %taps_f32 = arith.sitofp %taps_i32 : i32 to f32
      %final_empty = tensor.empty(
          %output_height_size, %output_width_size, %channels_size)
          : tensor<1x?x?x?xf32>
      %final_inner = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
          ],
          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } ins(%averaged_loaded, %init_loaded
            : tensor<1x?x?x?xf16>, tensor<1x?x?x?xf32>)
          outs(%final_empty : tensor<1x?x?x?xf32>) {
        ^bb0(%average: f16, %initial: f32, %out: f32):
          %average_f32 = arith.extf %average : f16 to f32
          %sum = arith.mulf %average_f32, %taps_f32 : f32
          %accumulated = arith.addf %sum, %initial : f32
          linalg.yield %accumulated : f32
      } -> tensor<1x?x?x?xf32>
      iree_tensor_ext.dispatch.tensor.store %final_inner, %final_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %channels_size],
          strides = [1, 1, 1, 1]
          : tensor<1x?x?x?xf32>
          -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %channels_size}
      flow.return
    } count(%output_height_workload: index,
            %output_width_workload: index,
            %channels_workload: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(
          %output_height_workload,
          %output_width_workload,
          %channels_workload)
      flow.return %x, %y, %z : index, index, index
    }

    util.return %final_nhwc : tensor<1x?x?x?xf32>
  }

  // The padded-convolution shim, stride 1.
  //
  // Identical to @call_rocket_dynamic_conv2d except for what arrives: the
  // input is the **unpadded** tensor, because the DAG match claimed the
  // `tensor.pad` along with the convolution and yielded the pad's source as
  // the leaf. The CNA does the padding instead, from the target's static
  // `pad_top`/`pad_left`.
  //
  // It arrives rank 5 -- `1x1xHxWxC` -- because the channels-last conversion
  // pads in that form and collapses to rank 4 for the convolution, and the
  // collapse is inside the claimed subgraph. Collapsing here rather than
  // matching a rank-4 pad keeps the template to the shape models actually
  // produce.
  //
  // The dimensions pushed are the *unpadded* extents, which is what the
  // runtime wants: it derives the output extent itself from those plus the
  // stride and the pad it was given.
  util.func private @call_rocket_dynamic_conv2d_pad1(
      %input: tensor<1x1x?x?x?xf16>,
      %filter: tensor<?x?x?x?xf16>,
      %init: tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    %input_nhwc = tensor.collapse_shape %input [[0], [1, 2], [3], [4]]
        : tensor<1x1x?x?x?xf16> into tensor<1x?x?x?xf16>

    %input_height = tensor.dim %input_nhwc, %c1 : tensor<1x?x?x?xf16>
    %input_width = tensor.dim %input_nhwc, %c2 : tensor<1x?x?x?xf16>
    %input_channels = tensor.dim %input_nhwc, %c3 : tensor<1x?x?x?xf16>
    %weights_height = tensor.dim %filter, %c0 : tensor<?x?x?x?xf16>
    %weights_width = tensor.dim %filter, %c1 : tensor<?x?x?x?xf16>
    %output_height = tensor.dim %init, %c1 : tensor<1x?x?x?xf32>
    %output_width = tensor.dim %init, %c2 : tensor<1x?x?x?xf32>
    %output_channels = tensor.dim %init, %c3 : tensor<1x?x?x?xf32>

    %input_width_i32 = arith.index_cast %input_width : index to i32
    %input_height_i32 = arith.index_cast %input_height : index to i32
    %input_channels_i32 = arith.index_cast %input_channels : index to i32
    %output_channels_i32 = arith.index_cast %output_channels : index to i32
    %weights_width_i32 = arith.index_cast %weights_width : index to i32
    %weights_height_i32 = arith.index_cast %weights_height : index to i32

    %zero_bias_empty = tensor.empty(%output_channels) : tensor<?xf16>
    %zero_f16 = arith.constant 0.0 : f16
    %zero_bias = linalg.fill ins(%zero_f16 : f16)
        outs(%zero_bias_empty : tensor<?xf16>) -> tensor<?xf16>

    %raw_f16 = flow.dispatch
        @rocket_dynamic_pad1_executable::@rocket_dynamic_conv2d_v1::@rocket_dynamic_conv2d(
          %input_width_i32, %input_height_i32, %input_channels_i32,
          %output_channels_i32, %weights_width_i32, %weights_height_i32,
          %input_nhwc, %filter, %zero_bias)
        {stream.affinity = #hal.device.affinity<@rocket_device>}
        : (i32, i32, i32, i32, i32, i32,
           tensor<1x?x?x?xf16>{%input_height, %input_width, %input_channels},
           tensor<?x?x?x?xf16>{%weights_height, %weights_width, %input_channels, %output_channels},
           tensor<?xf16>{%output_channels})
        -> tensor<1x?x?x?xf16>{%output_height, %output_width, %output_channels}

    %final = flow.dispatch.workgroups[
        %output_height, %output_width, %output_channels](
        %raw_f16, %init, %output_height, %output_width, %output_channels)
        : (tensor<1x?x?x?xf16>{%output_height, %output_width, %output_channels},
           tensor<1x?x?x?xf32>{%output_height, %output_width, %output_channels},
           index, index, index)
        -> tensor<1x?x?x?xf32>{%output_height, %output_width, %output_channels}
        attributes { stream.affinity = #hal.device.affinity<@cpu_device> } =
        (%raw_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>,
         %init_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>,
         %output_height_arg: index,
         %output_width_arg: index,
         %output_channels_arg: index,
         %final_binding: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>) {
      %output_height_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_height_arg, 0 : index
      %output_width_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_width_arg, 1 : index
      %output_channels_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_channels_arg, 2 : index
      %raw_shaped = flow.dispatch.tie_shape %raw_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %init_shaped = flow.dispatch.tie_shape %init_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %final_shaped = flow.dispatch.tie_shape %final_binding
          : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %raw_loaded = iree_tensor_ext.dispatch.tensor.load %raw_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %output_channels_size}
          -> tensor<1x?x?x?xf16>
      %init_loaded = iree_tensor_ext.dispatch.tensor.load %init_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
          -> tensor<1x?x?x?xf32>
      %final_empty = tensor.empty(
          %output_height_size, %output_width_size, %output_channels_size)
          : tensor<1x?x?x?xf32>
      %final_inner = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
          ],
          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } ins(%raw_loaded, %init_loaded
            : tensor<1x?x?x?xf16>, tensor<1x?x?x?xf32>)
          outs(%final_empty : tensor<1x?x?x?xf32>) {
        ^bb0(%raw: f16, %initial: f32, %out: f32):
          %raw_f32 = arith.extf %raw : f16 to f32
          %sum = arith.addf %raw_f32, %initial : f32
          linalg.yield %sum : f32
      } -> tensor<1x?x?x?xf32>
      iree_tensor_ext.dispatch.tensor.store %final_inner, %final_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : tensor<1x?x?x?xf32>
          -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      flow.return
    } count(%output_height_workload: index,
            %output_width_workload: index,
            %output_channels_workload: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(
          %output_height_workload,
          %output_width_workload,
          %output_channels_workload)
      flow.return %x, %y, %z : index, index, index
    }

    util.return %final : tensor<1x?x?x?xf32>
  }

  // The padded-convolution shim, stride 2.
  //
  // Identical to @call_rocket_dynamic_conv2d except for what arrives: the
  // input is the **unpadded** tensor, because the DAG match claimed the
  // `tensor.pad` along with the convolution and yielded the pad's source as
  // the leaf. The CNA does the padding instead, from the target's static
  // `pad_top`/`pad_left`.
  //
  // It arrives rank 5 -- `1x1xHxWxC` -- because the channels-last conversion
  // pads in that form and collapses to rank 4 for the convolution, and the
  // collapse is inside the claimed subgraph. Collapsing here rather than
  // matching a rank-4 pad keeps the template to the shape models actually
  // produce.
  //
  // The dimensions pushed are the *unpadded* extents, which is what the
  // runtime wants: it derives the output extent itself from those plus the
  // stride and the pad it was given.
  util.func private @call_rocket_dynamic_conv2d_pad1_s2(
      %input: tensor<1x1x?x?x?xf16>,
      %filter: tensor<?x?x?x?xf16>,
      %init: tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    %input_nhwc = tensor.collapse_shape %input [[0], [1, 2], [3], [4]]
        : tensor<1x1x?x?x?xf16> into tensor<1x?x?x?xf16>

    %input_height = tensor.dim %input_nhwc, %c1 : tensor<1x?x?x?xf16>
    %input_width = tensor.dim %input_nhwc, %c2 : tensor<1x?x?x?xf16>
    %input_channels = tensor.dim %input_nhwc, %c3 : tensor<1x?x?x?xf16>
    %weights_height = tensor.dim %filter, %c0 : tensor<?x?x?x?xf16>
    %weights_width = tensor.dim %filter, %c1 : tensor<?x?x?x?xf16>
    %output_height = tensor.dim %init, %c1 : tensor<1x?x?x?xf32>
    %output_width = tensor.dim %init, %c2 : tensor<1x?x?x?xf32>
    %output_channels = tensor.dim %init, %c3 : tensor<1x?x?x?xf32>

    %input_width_i32 = arith.index_cast %input_width : index to i32
    %input_height_i32 = arith.index_cast %input_height : index to i32
    %input_channels_i32 = arith.index_cast %input_channels : index to i32
    %output_channels_i32 = arith.index_cast %output_channels : index to i32
    %weights_width_i32 = arith.index_cast %weights_width : index to i32
    %weights_height_i32 = arith.index_cast %weights_height : index to i32

    %zero_bias_empty = tensor.empty(%output_channels) : tensor<?xf16>
    %zero_f16 = arith.constant 0.0 : f16
    %zero_bias = linalg.fill ins(%zero_f16 : f16)
        outs(%zero_bias_empty : tensor<?xf16>) -> tensor<?xf16>

    %raw_f16 = flow.dispatch
        @rocket_dynamic_pad1_executable_s2::@rocket_dynamic_conv2d_v1::@rocket_dynamic_conv2d(
          %input_width_i32, %input_height_i32, %input_channels_i32,
          %output_channels_i32, %weights_width_i32, %weights_height_i32,
          %input_nhwc, %filter, %zero_bias)
        {stream.affinity = #hal.device.affinity<@rocket_device>}
        : (i32, i32, i32, i32, i32, i32,
           tensor<1x?x?x?xf16>{%input_height, %input_width, %input_channels},
           tensor<?x?x?x?xf16>{%weights_height, %weights_width, %input_channels, %output_channels},
           tensor<?xf16>{%output_channels})
        -> tensor<1x?x?x?xf16>{%output_height, %output_width, %output_channels}

    %final = flow.dispatch.workgroups[
        %output_height, %output_width, %output_channels](
        %raw_f16, %init, %output_height, %output_width, %output_channels)
        : (tensor<1x?x?x?xf16>{%output_height, %output_width, %output_channels},
           tensor<1x?x?x?xf32>{%output_height, %output_width, %output_channels},
           index, index, index)
        -> tensor<1x?x?x?xf32>{%output_height, %output_width, %output_channels}
        attributes { stream.affinity = #hal.device.affinity<@cpu_device> } =
        (%raw_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>,
         %init_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>,
         %output_height_arg: index,
         %output_width_arg: index,
         %output_channels_arg: index,
         %final_binding: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>) {
      %output_height_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_height_arg, 0 : index
      %output_width_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_width_arg, 1 : index
      %output_channels_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_channels_arg, 2 : index
      %raw_shaped = flow.dispatch.tie_shape %raw_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %init_shaped = flow.dispatch.tie_shape %init_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %final_shaped = flow.dispatch.tie_shape %final_binding
          : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %raw_loaded = iree_tensor_ext.dispatch.tensor.load %raw_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %output_channels_size}
          -> tensor<1x?x?x?xf16>
      %init_loaded = iree_tensor_ext.dispatch.tensor.load %init_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
          -> tensor<1x?x?x?xf32>
      %final_empty = tensor.empty(
          %output_height_size, %output_width_size, %output_channels_size)
          : tensor<1x?x?x?xf32>
      %final_inner = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
          ],
          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } ins(%raw_loaded, %init_loaded
            : tensor<1x?x?x?xf16>, tensor<1x?x?x?xf32>)
          outs(%final_empty : tensor<1x?x?x?xf32>) {
        ^bb0(%raw: f16, %initial: f32, %out: f32):
          %raw_f32 = arith.extf %raw : f16 to f32
          %sum = arith.addf %raw_f32, %initial : f32
          linalg.yield %sum : f32
      } -> tensor<1x?x?x?xf32>
      iree_tensor_ext.dispatch.tensor.store %final_inner, %final_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : tensor<1x?x?x?xf32>
          -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      flow.return
    } count(%output_height_workload: index,
            %output_width_workload: index,
            %output_channels_workload: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(
          %output_height_workload,
          %output_width_workload,
          %output_channels_workload)
      flow.return %x, %y, %z : index, index, index
    }

    util.return %final : tensor<1x?x?x?xf32>
  }

  // The ReLU6 shim. Same dispatch as @call_rocket_dynamic_conv2d with two
  // differences, both of which are the point of the target:
  //
  //   * the bias binding carries the model's real per-channel bias instead
  //     of a zero fill, so the DPU's BS plane adds it before BN clamps; and
  //   * the CPU epilogue only widens f16 -> f32. It no longer adds the bias,
  //     because the hardware already did, and it no longer needs a separate
  //     clamp dispatch, which is the 43.8 MB of f32 traffic per inference
  //     this whole path exists to remove.
  //
  // `%low` and `%high` are unused. They are parameters because the DAG match
  // yields every leaf of the subgraph it claimed, and keeping them in the
  // signature documents where the ceiling went: into the target's static
  // `activation_cmp`, guaranteed to be 6.0 by `rocket-fuse-conv-relu6`.
  //
  // The bias is narrowed to f16 here because that is the binding type the
  // Conv2D ABI gives it; the driver widens it straight back to f32 for
  // BRDMA (`pack_fp16_bias_to_rocket`), so the round trip costs one f16
  // rounding per output channel and nothing per element.
  util.func private @call_rocket_dynamic_conv2d_relu6(
      %input: tensor<1x?x?x?xf16>,
      %filter: tensor<?x?x?x?xf16>,
      %acc_init: tensor<1x?x?x?xf32>,
      %bias: tensor<?xf32>,
      %low: f32,
      %high: f32,
      %init: tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    %input_height = tensor.dim %input, %c1 : tensor<1x?x?x?xf16>
    %input_width = tensor.dim %input, %c2 : tensor<1x?x?x?xf16>
    %input_channels = tensor.dim %input, %c3 : tensor<1x?x?x?xf16>
    %weights_height = tensor.dim %filter, %c0 : tensor<?x?x?x?xf16>
    %weights_width = tensor.dim %filter, %c1 : tensor<?x?x?x?xf16>
    %output_height = tensor.dim %init, %c1 : tensor<1x?x?x?xf32>
    %output_width = tensor.dim %init, %c2 : tensor<1x?x?x?xf32>
    %output_channels = tensor.dim %init, %c3 : tensor<1x?x?x?xf32>

    %input_width_i32 = arith.index_cast %input_width : index to i32
    %input_height_i32 = arith.index_cast %input_height : index to i32
    %input_channels_i32 = arith.index_cast %input_channels : index to i32
    %output_channels_i32 = arith.index_cast %output_channels : index to i32
    %weights_width_i32 = arith.index_cast %weights_width : index to i32
    %weights_height_i32 = arith.index_cast %weights_height : index to i32

    %bias_empty = tensor.empty(%output_channels) : tensor<?xf16>
    %bias_f16 = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%bias : tensor<?xf32>) outs(%bias_empty : tensor<?xf16>) {
      ^bb0(%value: f32, %out: f16):
        %narrowed = arith.truncf %value : f32 to f16
        linalg.yield %narrowed : f16
    } -> tensor<?xf16>

    %raw_f16 = flow.dispatch
        @rocket_dynamic_relu6_executable::@rocket_dynamic_conv2d_v1::@rocket_dynamic_conv2d(
          %input_width_i32, %input_height_i32, %input_channels_i32,
          %output_channels_i32, %weights_width_i32, %weights_height_i32,
          %input, %filter, %bias_f16)
        {stream.affinity = #hal.device.affinity<@rocket_device>}
        : (i32, i32, i32, i32, i32, i32,
           tensor<1x?x?x?xf16>{%input_height, %input_width, %input_channels},
           tensor<?x?x?x?xf16>{%weights_height, %weights_width, %input_channels, %output_channels},
           tensor<?xf16>{%output_channels})
        -> tensor<1x?x?x?xf16>{%output_height, %output_width, %output_channels}

    %final = flow.dispatch.workgroups[
        %output_height, %output_width, %output_channels](
        %raw_f16, %output_height, %output_width, %output_channels)
        : (tensor<1x?x?x?xf16>{%output_height, %output_width, %output_channels},
           index, index, index)
        -> tensor<1x?x?x?xf32>{%output_height, %output_width, %output_channels}
        attributes { stream.affinity = #hal.device.affinity<@cpu_device> } =
        (%raw_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>,
         %output_height_arg: index,
         %output_width_arg: index,
         %output_channels_arg: index,
         %final_binding: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>) {
      %output_height_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_height_arg, 0 : index
      %output_width_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_width_arg, 1 : index
      %output_channels_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_channels_arg, 2 : index
      %raw_shaped = flow.dispatch.tie_shape %raw_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %final_shaped = flow.dispatch.tie_shape %final_binding
          : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %raw_loaded = iree_tensor_ext.dispatch.tensor.load %raw_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %output_channels_size}
          -> tensor<1x?x?x?xf16>
      %final_empty = tensor.empty(
          %output_height_size, %output_width_size, %output_channels_size)
          : tensor<1x?x?x?xf32>
      %final_inner = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
          ],
          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } ins(%raw_loaded : tensor<1x?x?x?xf16>)
          outs(%final_empty : tensor<1x?x?x?xf32>) {
        ^bb0(%raw: f16, %out: f32):
          %raw_f32 = arith.extf %raw : f16 to f32
          linalg.yield %raw_f32 : f32
      } -> tensor<1x?x?x?xf32>
      iree_tensor_ext.dispatch.tensor.store %final_inner, %final_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : tensor<1x?x?x?xf32>
          -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      flow.return
    } count(%output_height_workload: index,
            %output_width_workload: index,
            %output_channels_workload: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(
          %output_height_workload,
          %output_width_workload,
          %output_channels_workload)
      flow.return %x, %y, %z : index, index, index
    }

    util.return %final : tensor<1x?x?x?xf32>
  }

  // Generic runtime-shape adapter. Batch remains statically one because it is
  // fixed by the Rocket Conv ABI. Every other logical Conv dimension is read
  // from the cast-compatible tensor operands and passed as an i32 push
  // constant in exactly the order declared by #rocket_dynamic_target.
  util.func private @call_rocket_dynamic_conv2d(
      %input: tensor<1x?x?x?xf16>,
      %filter: tensor<?x?x?x?xf16>,
      %init: tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    %input_height = tensor.dim %input, %c1 : tensor<1x?x?x?xf16>
    %input_width = tensor.dim %input, %c2 : tensor<1x?x?x?xf16>
    %input_channels = tensor.dim %input, %c3 : tensor<1x?x?x?xf16>
    %weights_height = tensor.dim %filter, %c0 : tensor<?x?x?x?xf16>
    %weights_width = tensor.dim %filter, %c1 : tensor<?x?x?x?xf16>
    %output_height = tensor.dim %init, %c1 : tensor<1x?x?x?xf32>
    %output_width = tensor.dim %init, %c2 : tensor<1x?x?x?xf32>
    %output_channels = tensor.dim %init, %c3 : tensor<1x?x?x?xf32>

    // %output_height/%output_width stay index-typed: they still describe the
    // dispatch result shape, but they are not push constants -- the runtime
    // derives the output extent itself.
    %input_width_i32 = arith.index_cast %input_width : index to i32
    %input_height_i32 = arith.index_cast %input_height : index to i32
    %input_channels_i32 = arith.index_cast %input_channels : index to i32
    %output_channels_i32 = arith.index_cast %output_channels : index to i32
    %weights_width_i32 = arith.index_cast %weights_width : index to i32
    %weights_height_i32 = arith.index_cast %weights_height : index to i32

    %zero_bias_empty = tensor.empty(%output_channels) : tensor<?xf16>
    %zero_f16 = arith.constant 0.0 : f16
    %zero_bias = linalg.fill ins(%zero_f16 : f16)
        outs(%zero_bias_empty : tensor<?xf16>) -> tensor<?xf16>

    %raw_f16 = flow.dispatch
        @rocket_dynamic_executable::@rocket_dynamic_conv2d_v1::@rocket_dynamic_conv2d(
          %input_width_i32, %input_height_i32, %input_channels_i32,
          %output_channels_i32, %weights_width_i32, %weights_height_i32,
          %input, %filter, %zero_bias)
        {stream.affinity = #hal.device.affinity<@rocket_device>}
        : (i32, i32, i32, i32, i32, i32,
           tensor<1x?x?x?xf16>{%input_height, %input_width, %input_channels},
           tensor<?x?x?x?xf16>{%weights_height, %weights_width, %input_channels, %output_channels},
           tensor<?xf16>{%output_channels})
        -> tensor<1x?x?x?xf16>{%output_height, %output_width, %output_channels}

    %final = flow.dispatch.workgroups[
        %output_height, %output_width, %output_channels](
        %raw_f16, %init, %output_height, %output_width, %output_channels)
        : (tensor<1x?x?x?xf16>{%output_height, %output_width, %output_channels},
           tensor<1x?x?x?xf32>{%output_height, %output_width, %output_channels},
           index, index, index)
        -> tensor<1x?x?x?xf32>{%output_height, %output_width, %output_channels}
        attributes { stream.affinity = #hal.device.affinity<@cpu_device> } =
        (%raw_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>,
         %init_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>,
         %output_height_arg: index,
         %output_width_arg: index,
         %output_channels_arg: index,
         %final_binding: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>) {
      %output_height_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_height_arg, 0 : index
      %output_width_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_width_arg, 1 : index
      %output_channels_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_channels_arg, 2 : index
      %raw_shaped = flow.dispatch.tie_shape %raw_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %init_shaped = flow.dispatch.tie_shape %init_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %final_shaped = flow.dispatch.tie_shape %final_binding
          : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %raw_loaded = iree_tensor_ext.dispatch.tensor.load %raw_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %output_channels_size}
          -> tensor<1x?x?x?xf16>
      %init_loaded = iree_tensor_ext.dispatch.tensor.load %init_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
          -> tensor<1x?x?x?xf32>
      %final_empty = tensor.empty(
          %output_height_size, %output_width_size, %output_channels_size)
          : tensor<1x?x?x?xf32>
      %final_inner = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
          ],
          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } ins(%raw_loaded, %init_loaded
            : tensor<1x?x?x?xf16>, tensor<1x?x?x?xf32>)
          outs(%final_empty : tensor<1x?x?x?xf32>) {
        ^bb0(%raw: f16, %initial: f32, %out: f32):
          %raw_f32 = arith.extf %raw : f16 to f32
          %sum = arith.addf %raw_f32, %initial : f32
          linalg.yield %sum : f32
      } -> tensor<1x?x?x?xf32>
      iree_tensor_ext.dispatch.tensor.store %final_inner, %final_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : tensor<1x?x?x?xf32>
          -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      flow.return
    } count(%output_height_workload: index,
            %output_width_workload: index,
            %output_channels_workload: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(
          %output_height_workload,
          %output_width_workload,
          %output_channels_workload)
      flow.return %x, %y, %z : index, index, index
    }

    util.return %final : tensor<1x?x?x?xf32>
  }

  // Stride-2 counterpart of call_rocket_dynamic_conv2d -- byte-identical
  // apart from the executable it dispatches to (@rocket_dynamic_executable_s2,
  // whose #rocket_dynamic_target_s2 bakes stride=2: the wire format's
  // Conv2DDef.stride is a fixed per-variant attribute, not one of the
  // runtime-settable push constants above, so a distinct stride needs its
  // own executable rather than a runtime argument). Hardware-confirmed for
  // dense fp16 at stride 2/3/4, both 1x1 and 3x3 kernels
  // (conv_wide_shape_hw.rs::shape_generalised_convs_run_on_npu, all 40
  // combinations pass -- see DESIGN_NOTES.md "Stride and large-width
  // sweeps").
  util.func private @call_rocket_dynamic_conv2d_s2(
      %input: tensor<1x?x?x?xf16>,
      %filter: tensor<?x?x?x?xf16>,
      %init: tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    %input_height = tensor.dim %input, %c1 : tensor<1x?x?x?xf16>
    %input_width = tensor.dim %input, %c2 : tensor<1x?x?x?xf16>
    %input_channels = tensor.dim %input, %c3 : tensor<1x?x?x?xf16>
    %weights_height = tensor.dim %filter, %c0 : tensor<?x?x?x?xf16>
    %weights_width = tensor.dim %filter, %c1 : tensor<?x?x?x?xf16>
    %output_height = tensor.dim %init, %c1 : tensor<1x?x?x?xf32>
    %output_width = tensor.dim %init, %c2 : tensor<1x?x?x?xf32>
    %output_channels = tensor.dim %init, %c3 : tensor<1x?x?x?xf32>

    %input_width_i32 = arith.index_cast %input_width : index to i32
    %input_height_i32 = arith.index_cast %input_height : index to i32
    %input_channels_i32 = arith.index_cast %input_channels : index to i32
    %output_channels_i32 = arith.index_cast %output_channels : index to i32
    %weights_width_i32 = arith.index_cast %weights_width : index to i32
    %weights_height_i32 = arith.index_cast %weights_height : index to i32

    %zero_bias_empty = tensor.empty(%output_channels) : tensor<?xf16>
    %zero_f16 = arith.constant 0.0 : f16
    %zero_bias = linalg.fill ins(%zero_f16 : f16)
        outs(%zero_bias_empty : tensor<?xf16>) -> tensor<?xf16>

    %raw_f16 = flow.dispatch
        @rocket_dynamic_executable_s2::@rocket_dynamic_conv2d_v1::@rocket_dynamic_conv2d(
          %input_width_i32, %input_height_i32, %input_channels_i32,
          %output_channels_i32, %weights_width_i32, %weights_height_i32,
          %input, %filter, %zero_bias)
        {stream.affinity = #hal.device.affinity<@rocket_device>}
        : (i32, i32, i32, i32, i32, i32,
           tensor<1x?x?x?xf16>{%input_height, %input_width, %input_channels},
           tensor<?x?x?x?xf16>{%weights_height, %weights_width, %input_channels, %output_channels},
           tensor<?xf16>{%output_channels})
        -> tensor<1x?x?x?xf16>{%output_height, %output_width, %output_channels}

    %final = flow.dispatch.workgroups[
        %output_height, %output_width, %output_channels](
        %raw_f16, %init, %output_height, %output_width, %output_channels)
        : (tensor<1x?x?x?xf16>{%output_height, %output_width, %output_channels},
           tensor<1x?x?x?xf32>{%output_height, %output_width, %output_channels},
           index, index, index)
        -> tensor<1x?x?x?xf32>{%output_height, %output_width, %output_channels}
        attributes { stream.affinity = #hal.device.affinity<@cpu_device> } =
        (%raw_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>,
         %init_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>,
         %output_height_arg: index,
         %output_width_arg: index,
         %output_channels_arg: index,
         %final_binding: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>) {
      %output_height_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_height_arg, 0 : index
      %output_width_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_width_arg, 1 : index
      %output_channels_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_channels_arg, 2 : index
      %raw_shaped = flow.dispatch.tie_shape %raw_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %init_shaped = flow.dispatch.tie_shape %init_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %final_shaped = flow.dispatch.tie_shape %final_binding
          : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %raw_loaded = iree_tensor_ext.dispatch.tensor.load %raw_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %output_channels_size}
          -> tensor<1x?x?x?xf16>
      %init_loaded = iree_tensor_ext.dispatch.tensor.load %init_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
          -> tensor<1x?x?x?xf32>
      %final_empty = tensor.empty(
          %output_height_size, %output_width_size, %output_channels_size)
          : tensor<1x?x?x?xf32>
      %final_inner = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
          ],
          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } ins(%raw_loaded, %init_loaded
            : tensor<1x?x?x?xf16>, tensor<1x?x?x?xf32>)
          outs(%final_empty : tensor<1x?x?x?xf32>) {
        ^bb0(%raw: f16, %initial: f32, %out: f32):
          %raw_f32 = arith.extf %raw : f16 to f32
          %sum = arith.addf %raw_f32, %initial : f32
          linalg.yield %sum : f32
      } -> tensor<1x?x?x?xf32>
      iree_tensor_ext.dispatch.tensor.store %final_inner, %final_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : tensor<1x?x?x?xf32>
          -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      flow.return
    } count(%output_height_workload: index,
            %output_width_workload: index,
            %output_channels_workload: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(
          %output_height_workload,
          %output_width_workload,
          %output_channels_workload)
      flow.return %x, %y, %z : index, index, index
    }

    util.return %final : tensor<1x?x?x?xf32>
  }

  // Stride-3 counterpart of call_rocket_dynamic_conv2d -- byte-identical
  // apart from the executable it dispatches to (@rocket_dynamic_executable_s3,
  // whose #rocket_dynamic_target_s3 bakes stride=3: the wire format's
  // Conv2DDef.stride is a fixed per-variant attribute, not one of the
  // runtime-settable push constants above, so a distinct stride needs its
  // own executable rather than a runtime argument). Hardware-confirmed for
  // dense fp16 at stride 2/3/4, both 1x1 and 3x3 kernels
  // (conv_wide_shape_hw.rs::shape_generalised_convs_run_on_npu, all 40
  // combinations pass -- see DESIGN_NOTES.md "Stride and large-width
  // sweeps").
  util.func private @call_rocket_dynamic_conv2d_s3(
      %input: tensor<1x?x?x?xf16>,
      %filter: tensor<?x?x?x?xf16>,
      %init: tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    %input_height = tensor.dim %input, %c1 : tensor<1x?x?x?xf16>
    %input_width = tensor.dim %input, %c2 : tensor<1x?x?x?xf16>
    %input_channels = tensor.dim %input, %c3 : tensor<1x?x?x?xf16>
    %weights_height = tensor.dim %filter, %c0 : tensor<?x?x?x?xf16>
    %weights_width = tensor.dim %filter, %c1 : tensor<?x?x?x?xf16>
    %output_height = tensor.dim %init, %c1 : tensor<1x?x?x?xf32>
    %output_width = tensor.dim %init, %c2 : tensor<1x?x?x?xf32>
    %output_channels = tensor.dim %init, %c3 : tensor<1x?x?x?xf32>

    %input_width_i32 = arith.index_cast %input_width : index to i32
    %input_height_i32 = arith.index_cast %input_height : index to i32
    %input_channels_i32 = arith.index_cast %input_channels : index to i32
    %output_channels_i32 = arith.index_cast %output_channels : index to i32
    %weights_width_i32 = arith.index_cast %weights_width : index to i32
    %weights_height_i32 = arith.index_cast %weights_height : index to i32

    %zero_bias_empty = tensor.empty(%output_channels) : tensor<?xf16>
    %zero_f16 = arith.constant 0.0 : f16
    %zero_bias = linalg.fill ins(%zero_f16 : f16)
        outs(%zero_bias_empty : tensor<?xf16>) -> tensor<?xf16>

    %raw_f16 = flow.dispatch
        @rocket_dynamic_executable_s3::@rocket_dynamic_conv2d_v1::@rocket_dynamic_conv2d(
          %input_width_i32, %input_height_i32, %input_channels_i32,
          %output_channels_i32, %weights_width_i32, %weights_height_i32,
          %input, %filter, %zero_bias)
        {stream.affinity = #hal.device.affinity<@rocket_device>}
        : (i32, i32, i32, i32, i32, i32,
           tensor<1x?x?x?xf16>{%input_height, %input_width, %input_channels},
           tensor<?x?x?x?xf16>{%weights_height, %weights_width, %input_channels, %output_channels},
           tensor<?xf16>{%output_channels})
        -> tensor<1x?x?x?xf16>{%output_height, %output_width, %output_channels}

    %final = flow.dispatch.workgroups[
        %output_height, %output_width, %output_channels](
        %raw_f16, %init, %output_height, %output_width, %output_channels)
        : (tensor<1x?x?x?xf16>{%output_height, %output_width, %output_channels},
           tensor<1x?x?x?xf32>{%output_height, %output_width, %output_channels},
           index, index, index)
        -> tensor<1x?x?x?xf32>{%output_height, %output_width, %output_channels}
        attributes { stream.affinity = #hal.device.affinity<@cpu_device> } =
        (%raw_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>,
         %init_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>,
         %output_height_arg: index,
         %output_width_arg: index,
         %output_channels_arg: index,
         %final_binding: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>) {
      %output_height_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_height_arg, 0 : index
      %output_width_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_width_arg, 1 : index
      %output_channels_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_channels_arg, 2 : index
      %raw_shaped = flow.dispatch.tie_shape %raw_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %init_shaped = flow.dispatch.tie_shape %init_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %final_shaped = flow.dispatch.tie_shape %final_binding
          : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %raw_loaded = iree_tensor_ext.dispatch.tensor.load %raw_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %output_channels_size}
          -> tensor<1x?x?x?xf16>
      %init_loaded = iree_tensor_ext.dispatch.tensor.load %init_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
          -> tensor<1x?x?x?xf32>
      %final_empty = tensor.empty(
          %output_height_size, %output_width_size, %output_channels_size)
          : tensor<1x?x?x?xf32>
      %final_inner = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
          ],
          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } ins(%raw_loaded, %init_loaded
            : tensor<1x?x?x?xf16>, tensor<1x?x?x?xf32>)
          outs(%final_empty : tensor<1x?x?x?xf32>) {
        ^bb0(%raw: f16, %initial: f32, %out: f32):
          %raw_f32 = arith.extf %raw : f16 to f32
          %sum = arith.addf %raw_f32, %initial : f32
          linalg.yield %sum : f32
      } -> tensor<1x?x?x?xf32>
      iree_tensor_ext.dispatch.tensor.store %final_inner, %final_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : tensor<1x?x?x?xf32>
          -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      flow.return
    } count(%output_height_workload: index,
            %output_width_workload: index,
            %output_channels_workload: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(
          %output_height_workload,
          %output_width_workload,
          %output_channels_workload)
      flow.return %x, %y, %z : index, index, index
    }

    util.return %final : tensor<1x?x?x?xf32>
  }

  // Stride-4 counterpart of call_rocket_dynamic_conv2d -- byte-identical
  // apart from the executable it dispatches to (@rocket_dynamic_executable_s4,
  // whose #rocket_dynamic_target_s4 bakes stride=4: the wire format's
  // Conv2DDef.stride is a fixed per-variant attribute, not one of the
  // runtime-settable push constants above, so a distinct stride needs its
  // own executable rather than a runtime argument). Hardware-confirmed for
  // dense fp16 at stride 2/3/4, both 1x1 and 3x3 kernels
  // (conv_wide_shape_hw.rs::shape_generalised_convs_run_on_npu, all 40
  // combinations pass -- see DESIGN_NOTES.md "Stride and large-width
  // sweeps").
  util.func private @call_rocket_dynamic_conv2d_s4(
      %input: tensor<1x?x?x?xf16>,
      %filter: tensor<?x?x?x?xf16>,
      %init: tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    %input_height = tensor.dim %input, %c1 : tensor<1x?x?x?xf16>
    %input_width = tensor.dim %input, %c2 : tensor<1x?x?x?xf16>
    %input_channels = tensor.dim %input, %c3 : tensor<1x?x?x?xf16>
    %weights_height = tensor.dim %filter, %c0 : tensor<?x?x?x?xf16>
    %weights_width = tensor.dim %filter, %c1 : tensor<?x?x?x?xf16>
    %output_height = tensor.dim %init, %c1 : tensor<1x?x?x?xf32>
    %output_width = tensor.dim %init, %c2 : tensor<1x?x?x?xf32>
    %output_channels = tensor.dim %init, %c3 : tensor<1x?x?x?xf32>

    %input_width_i32 = arith.index_cast %input_width : index to i32
    %input_height_i32 = arith.index_cast %input_height : index to i32
    %input_channels_i32 = arith.index_cast %input_channels : index to i32
    %output_channels_i32 = arith.index_cast %output_channels : index to i32
    %weights_width_i32 = arith.index_cast %weights_width : index to i32
    %weights_height_i32 = arith.index_cast %weights_height : index to i32

    %zero_bias_empty = tensor.empty(%output_channels) : tensor<?xf16>
    %zero_f16 = arith.constant 0.0 : f16
    %zero_bias = linalg.fill ins(%zero_f16 : f16)
        outs(%zero_bias_empty : tensor<?xf16>) -> tensor<?xf16>

    %raw_f16 = flow.dispatch
        @rocket_dynamic_executable_s4::@rocket_dynamic_conv2d_v1::@rocket_dynamic_conv2d(
          %input_width_i32, %input_height_i32, %input_channels_i32,
          %output_channels_i32, %weights_width_i32, %weights_height_i32,
          %input, %filter, %zero_bias)
        {stream.affinity = #hal.device.affinity<@rocket_device>}
        : (i32, i32, i32, i32, i32, i32,
           tensor<1x?x?x?xf16>{%input_height, %input_width, %input_channels},
           tensor<?x?x?x?xf16>{%weights_height, %weights_width, %input_channels, %output_channels},
           tensor<?xf16>{%output_channels})
        -> tensor<1x?x?x?xf16>{%output_height, %output_width, %output_channels}

    %final = flow.dispatch.workgroups[
        %output_height, %output_width, %output_channels](
        %raw_f16, %init, %output_height, %output_width, %output_channels)
        : (tensor<1x?x?x?xf16>{%output_height, %output_width, %output_channels},
           tensor<1x?x?x?xf32>{%output_height, %output_width, %output_channels},
           index, index, index)
        -> tensor<1x?x?x?xf32>{%output_height, %output_width, %output_channels}
        attributes { stream.affinity = #hal.device.affinity<@cpu_device> } =
        (%raw_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>,
         %init_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>,
         %output_height_arg: index,
         %output_width_arg: index,
         %output_channels_arg: index,
         %final_binding: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>) {
      %output_height_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_height_arg, 0 : index
      %output_width_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_width_arg, 1 : index
      %output_channels_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_channels_arg, 2 : index
      %raw_shaped = flow.dispatch.tie_shape %raw_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %init_shaped = flow.dispatch.tie_shape %init_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %final_shaped = flow.dispatch.tie_shape %final_binding
          : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %raw_loaded = iree_tensor_ext.dispatch.tensor.load %raw_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %output_channels_size}
          -> tensor<1x?x?x?xf16>
      %init_loaded = iree_tensor_ext.dispatch.tensor.load %init_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
          -> tensor<1x?x?x?xf32>
      %final_empty = tensor.empty(
          %output_height_size, %output_width_size, %output_channels_size)
          : tensor<1x?x?x?xf32>
      %final_inner = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
          ],
          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } ins(%raw_loaded, %init_loaded
            : tensor<1x?x?x?xf16>, tensor<1x?x?x?xf32>)
          outs(%final_empty : tensor<1x?x?x?xf32>) {
        ^bb0(%raw: f16, %initial: f32, %out: f32):
          %raw_f32 = arith.extf %raw : f16 to f32
          %sum = arith.addf %raw_f32, %initial : f32
          linalg.yield %sum : f32
      } -> tensor<1x?x?x?xf32>
      iree_tensor_ext.dispatch.tensor.store %final_inner, %final_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : tensor<1x?x?x?xf32>
          -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      flow.return
    } count(%output_height_workload: index,
            %output_width_workload: index,
            %output_channels_workload: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(
          %output_height_workload,
          %output_width_workload,
          %output_channels_workload)
      flow.return %x, %y, %z : index, index, index
    }

    util.return %final : tensor<1x?x?x?xf32>
  }


  // Depthwise counterpart of call_rocket_dynamic_conv2d. The filter operand
  // drops its Cout dimension (HWC, not HWCF): Rocket's depthwise mode is
  // hardware-validated for a channel multiplier of one only
  // (ConvPlan::with_depthwise asserts this in conv.rs -- "depthwise capture
  // backing covers a channel multiplier of one only"), so output_channels
  // is always input_channels here, not an independent quantity. It is still
  // read off %init and sent as its own push constant, matching the wire
  // format #rocket_dynamic_depthwise_target declares, but the matchers below
  // never bind it to anything other than input_channels.
  util.func private @call_rocket_dynamic_depthwise_conv2d(
      %input: tensor<1x?x?x?xf16>,
      %filter: tensor<?x?x?xf16>,
      %init: tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    %input_height = tensor.dim %input, %c1 : tensor<1x?x?x?xf16>
    %input_width = tensor.dim %input, %c2 : tensor<1x?x?x?xf16>
    %input_channels = tensor.dim %input, %c3 : tensor<1x?x?x?xf16>
    %weights_height = tensor.dim %filter, %c0 : tensor<?x?x?xf16>
    %weights_width = tensor.dim %filter, %c1 : tensor<?x?x?xf16>
    %output_height = tensor.dim %init, %c1 : tensor<1x?x?x?xf32>
    %output_width = tensor.dim %init, %c2 : tensor<1x?x?x?xf32>
    %output_channels = tensor.dim %init, %c3 : tensor<1x?x?x?xf32>

    %input_width_i32 = arith.index_cast %input_width : index to i32
    %input_height_i32 = arith.index_cast %input_height : index to i32
    %input_channels_i32 = arith.index_cast %input_channels : index to i32
    %output_channels_i32 = arith.index_cast %output_channels : index to i32
    %weights_width_i32 = arith.index_cast %weights_width : index to i32
    %weights_height_i32 = arith.index_cast %weights_height : index to i32

    %zero_bias_empty = tensor.empty(%output_channels) : tensor<?xf16>
    %zero_f16 = arith.constant 0.0 : f16
    %zero_bias = linalg.fill ins(%zero_f16 : f16)
        outs(%zero_bias_empty : tensor<?xf16>) -> tensor<?xf16>

    // linalg.depthwise_conv_2d_nhwc_hwc's filter operand is [kh][kw][c] --
    // required by the op's own semantics, matched by the CPU reference
    // computation, and NOT what the driver's weight-packing path expects.
    // rocket-hal-driver's pack_depthwise_to_rocket_weights (the hardware-
    // derived tap-major packer, see its doc comment) takes a torch/ONNX-
    // style [c][kh][kw] buffer, matching how a real depthwise filter is
    // conventionally stored -- this op's HWC layout is a linalg-dialect
    // convention, not a hardware one. Transposing here, once, keeps that
    // packer's contract simple instead of teaching it a second input order.
    %filter_chw_empty = tensor.empty(%input_channels, %weights_height, %weights_width) : tensor<?x?x?xf16>
    %filter_chw = linalg.transpose
        ins(%filter : tensor<?x?x?xf16>)
        outs(%filter_chw_empty : tensor<?x?x?xf16>)
        permutation = [2, 0, 1]

    %raw_f16 = flow.dispatch
        @rocket_dynamic_depthwise_executable::@rocket_dynamic_depthwise_conv2d_v1::@rocket_dynamic_depthwise_conv2d(
          %input_width_i32, %input_height_i32, %input_channels_i32,
          %output_channels_i32, %weights_width_i32, %weights_height_i32,
          %input, %filter_chw, %zero_bias)
        {stream.affinity = #hal.device.affinity<@rocket_device>}
        : (i32, i32, i32, i32, i32, i32,
           tensor<1x?x?x?xf16>{%input_height, %input_width, %input_channels},
           tensor<?x?x?xf16>{%input_channels, %weights_height, %weights_width},
           tensor<?xf16>{%output_channels})
        -> tensor<1x?x?x?xf16>{%output_height, %output_width, %output_channels}

    %final = flow.dispatch.workgroups[
        %output_height, %output_width, %output_channels](
        %raw_f16, %init, %output_height, %output_width, %output_channels)
        : (tensor<1x?x?x?xf16>{%output_height, %output_width, %output_channels},
           tensor<1x?x?x?xf32>{%output_height, %output_width, %output_channels},
           index, index, index)
        -> tensor<1x?x?x?xf32>{%output_height, %output_width, %output_channels}
        attributes { stream.affinity = #hal.device.affinity<@cpu_device> } =
        (%raw_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>,
         %init_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>,
         %output_height_arg: index,
         %output_width_arg: index,
         %output_channels_arg: index,
         %final_binding: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>) {
      %output_height_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_height_arg, 0 : index
      %output_width_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_width_arg, 1 : index
      %output_channels_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_channels_arg, 2 : index
      %raw_shaped = flow.dispatch.tie_shape %raw_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %init_shaped = flow.dispatch.tie_shape %init_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %final_shaped = flow.dispatch.tie_shape %final_binding
          : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %raw_loaded = iree_tensor_ext.dispatch.tensor.load %raw_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %output_channels_size}
          -> tensor<1x?x?x?xf16>
      %init_loaded = iree_tensor_ext.dispatch.tensor.load %init_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
          -> tensor<1x?x?x?xf32>
      %final_empty = tensor.empty(
          %output_height_size, %output_width_size, %output_channels_size)
          : tensor<1x?x?x?xf32>
      %final_inner = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
          ],
          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } ins(%raw_loaded, %init_loaded
            : tensor<1x?x?x?xf16>, tensor<1x?x?x?xf32>)
          outs(%final_empty : tensor<1x?x?x?xf32>) {
        ^bb0(%raw: f16, %initial: f32, %out: f32):
          %raw_f32 = arith.extf %raw : f16 to f32
          %sum = arith.addf %raw_f32, %initial : f32
          linalg.yield %sum : f32
      } -> tensor<1x?x?x?xf32>
      iree_tensor_ext.dispatch.tensor.store %final_inner, %final_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : tensor<1x?x?x?xf32>
          -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      flow.return
    } count(%output_height_workload: index,
            %output_width_workload: index,
            %output_channels_workload: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(
          %output_height_workload,
          %output_width_workload,
          %output_channels_workload)
      flow.return %x, %y, %z : index, index, index
    }

    util.return %final : tensor<1x?x?x?xf32>
  }

  // NCHW counterpart of call_rocket_dynamic_depthwise_conv2d. A real model
  // imported from ONNX (torch-mlir's onnx.Conv legalization) never produces
  // linalg.depthwise_conv_2d_nhwc_hwc directly: it lowers to
  // linalg.depthwise_conv_2d_nchw_chw, and IREE's own
  // iree-preprocessing-convert-conv-to-channels-last pass explicitly
  // declines to transpose depthwise convs to channels-last
  // (ConvertConvToChannelsLast.cpp's transposeConvLikeLinalgOp bails
  // whenever ConvolutionDimensions::depth is non-empty), unlike the dense
  // path, which that same pass always converts first. So the NHWC-only
  // matcher above -- correct and hardware-confirmed on its own -- never
  // sees a real ONNX-imported depthwise conv at all; every one of
  // MobileNetV2's 17 depthwise layers fell back to CPU confirming this
  // (iree-dump-module on a real compiled mobilenet.vmfb: zero references to
  // rocket_dynamic_depthwise_executable).
  //
  // Handled here by transposing host-side instead of waiting on an upstream
  // fix to the shared preprocessing pass: Rocket's hardware ABI is NHWC-
  // native (every existing matcher in this file agrees), so the input and
  // output feature maps are transposed NCHW<->NHWC around the same
  // rocket_dynamic_depthwise_executable dispatch call_rocket_dynamic_depthwise_conv2d
  // already uses. The filter needs no transpose at all here -- unlike the
  // NHWC matcher, whose HWC filter has to be transposed to CHW before
  // dispatch, linalg.depthwise_conv_2d_nchw_chw's filter operand is already
  // [c][kh][kw] (confirmed against a real onnx-imported MobileNet dump:
  // `tensor<32x3x3xf16>` for a Cin=32 depthwise layer), exactly what
  // rocket-hal-driver's pack_depthwise_to_rocket_weights expects.
  // ReLU6 twin of @call_rocket_dynamic_depthwise_conv2d_nchw. Stride 1.
  //
  // Three differences from that shim, all the same trade the dense ReLU6
  // shim makes: the bias binding carries the model's real per-channel bias
  // rather than a zero fill, the CPU epilogue only widens f16 -> f32, and
  // the result stays NCHW because `rocket-fuse-conv-relu6` leaves the
  // model's own NCHW -> NHWC transpose downstream of the clamp it moved.
  //
  // `%low` and `%high` are unused: the ceiling lives in the target's static
  // `activation_cmp`, and the pass guarantees it is 6.0.
  util.func private @call_rocket_dynamic_depthwise_conv2d_nchw_relu6(
      %input: tensor<1x?x?x?xf16>,
      %filter: tensor<?x?x?xf16>,
      %acc_init: tensor<1x?x?x?xf32>,
      %bias: tensor<?xf32>,
      %low: f32,
      %high: f32,
      %init: tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32> {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    %input_channels = tensor.dim %input, %c1 : tensor<1x?x?x?xf16>
    %input_height = tensor.dim %input, %c2 : tensor<1x?x?x?xf16>
    %input_width = tensor.dim %input, %c3 : tensor<1x?x?x?xf16>
    %weights_height = tensor.dim %filter, %c1 : tensor<?x?x?xf16>
    %weights_width = tensor.dim %filter, %c2 : tensor<?x?x?xf16>
    %output_channels = tensor.dim %init, %c1 : tensor<1x?x?x?xf32>
    %output_height = tensor.dim %init, %c2 : tensor<1x?x?x?xf32>
    %output_width = tensor.dim %init, %c3 : tensor<1x?x?x?xf32>

    %input_width_i32 = arith.index_cast %input_width : index to i32
    %input_height_i32 = arith.index_cast %input_height : index to i32
    %input_channels_i32 = arith.index_cast %input_channels : index to i32
    %output_channels_i32 = arith.index_cast %output_channels : index to i32
    %weights_width_i32 = arith.index_cast %weights_width : index to i32
    %weights_height_i32 = arith.index_cast %weights_height : index to i32

    %bias_empty = tensor.empty(%output_channels) : tensor<?xf16>
    %bias_f16 = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%bias : tensor<?xf32>) outs(%bias_empty : tensor<?xf16>) {
      ^bb0(%value: f32, %out: f16):
        %narrowed = arith.truncf %value : f32 to f16
        linalg.yield %narrowed : f16
    } -> tensor<?xf16>

    %input_nhwc_empty = tensor.empty(%input_height, %input_width, %input_channels) : tensor<1x?x?x?xf16>
    %input_nhwc = linalg.transpose
        ins(%input : tensor<1x?x?x?xf16>)
        outs(%input_nhwc_empty : tensor<1x?x?x?xf16>)
        permutation = [0, 2, 3, 1]

    %raw_f16 = flow.dispatch
        @rocket_dynamic_depthwise_relu6_executable::@rocket_dynamic_depthwise_conv2d_v1::@rocket_dynamic_depthwise_conv2d(
          %input_width_i32, %input_height_i32, %input_channels_i32,
          %output_channels_i32, %weights_width_i32, %weights_height_i32,
          %input_nhwc, %filter, %bias_f16)
        {stream.affinity = #hal.device.affinity<@rocket_device>}
        : (i32, i32, i32, i32, i32, i32,
           tensor<1x?x?x?xf16>{%input_height, %input_width, %input_channels},
           tensor<?x?x?xf16>{%input_channels, %weights_height, %weights_width},
           tensor<?xf16>{%output_channels})
        -> tensor<1x?x?x?xf16>{%output_height, %output_width, %output_channels}

    %final_nhwc = flow.dispatch.workgroups[
        %output_height, %output_width, %output_channels](
        %raw_f16, %output_height, %output_width, %output_channels)
        : (tensor<1x?x?x?xf16>{%output_height, %output_width, %output_channels},
           index, index, index)
        -> tensor<1x?x?x?xf32>{%output_height, %output_width, %output_channels}
        attributes { stream.affinity = #hal.device.affinity<@cpu_device> } =
        (%raw_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>,
         %output_height_arg: index,
         %output_width_arg: index,
         %output_channels_arg: index,
         %final_binding: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>) {
      %output_height_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_height_arg, 0 : index
      %output_width_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_width_arg, 1 : index
      %output_channels_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_channels_arg, 2 : index
      %raw_shaped = flow.dispatch.tie_shape %raw_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %final_shaped = flow.dispatch.tie_shape %final_binding
          : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %raw_loaded = iree_tensor_ext.dispatch.tensor.load %raw_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %output_channels_size}
          -> tensor<1x?x?x?xf16>
      %final_empty = tensor.empty(
          %output_height_size, %output_width_size, %output_channels_size)
          : tensor<1x?x?x?xf32>
      %final_inner = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
          ],
          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } ins(%raw_loaded : tensor<1x?x?x?xf16>)
          outs(%final_empty : tensor<1x?x?x?xf32>) {
        ^bb0(%raw: f16, %out: f32):
          %raw_f32 = arith.extf %raw : f16 to f32
          linalg.yield %raw_f32 : f32
      } -> tensor<1x?x?x?xf32>
      iree_tensor_ext.dispatch.tensor.store %final_inner, %final_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : tensor<1x?x?x?xf32>
          -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      flow.return
    } count(%output_height_workload: index,
            %output_width_workload: index,
            %output_channels_workload: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(
          %output_height_workload,
          %output_width_workload,
          %output_channels_workload)
      flow.return %x, %y, %z : index, index, index
    }

    %final_nchw_empty = tensor.empty(%output_channels, %output_height, %output_width) : tensor<1x?x?x?xf32>
    %final_nchw = linalg.transpose
        ins(%final_nhwc : tensor<1x?x?x?xf32>)
        outs(%final_nchw_empty : tensor<1x?x?x?xf32>)
        permutation = [0, 3, 1, 2]

    util.return %final_nchw : tensor<1x?x?x?xf32>
  }

  // ReLU6 twin of @call_rocket_dynamic_depthwise_conv2d_nchw_s2. Stride 2; the target bakes the stride, as every other strided executable here does.
  //
  // Three differences from that shim, all the same trade the dense ReLU6
  // shim makes: the bias binding carries the model's real per-channel bias
  // rather than a zero fill, the CPU epilogue only widens f16 -> f32, and
  // the result stays NCHW because `rocket-fuse-conv-relu6` leaves the
  // model's own NCHW -> NHWC transpose downstream of the clamp it moved.
  //
  // `%low` and `%high` are unused: the ceiling lives in the target's static
  // `activation_cmp`, and the pass guarantees it is 6.0.
  util.func private @call_rocket_dynamic_depthwise_conv2d_nchw_relu6_s2(
      %input: tensor<1x?x?x?xf16>,
      %filter: tensor<?x?x?xf16>,
      %acc_init: tensor<1x?x?x?xf32>,
      %bias: tensor<?xf32>,
      %low: f32,
      %high: f32,
      %init: tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32> {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    %input_channels = tensor.dim %input, %c1 : tensor<1x?x?x?xf16>
    %input_height = tensor.dim %input, %c2 : tensor<1x?x?x?xf16>
    %input_width = tensor.dim %input, %c3 : tensor<1x?x?x?xf16>
    %weights_height = tensor.dim %filter, %c1 : tensor<?x?x?xf16>
    %weights_width = tensor.dim %filter, %c2 : tensor<?x?x?xf16>
    %output_channels = tensor.dim %init, %c1 : tensor<1x?x?x?xf32>
    %output_height = tensor.dim %init, %c2 : tensor<1x?x?x?xf32>
    %output_width = tensor.dim %init, %c3 : tensor<1x?x?x?xf32>

    %input_width_i32 = arith.index_cast %input_width : index to i32
    %input_height_i32 = arith.index_cast %input_height : index to i32
    %input_channels_i32 = arith.index_cast %input_channels : index to i32
    %output_channels_i32 = arith.index_cast %output_channels : index to i32
    %weights_width_i32 = arith.index_cast %weights_width : index to i32
    %weights_height_i32 = arith.index_cast %weights_height : index to i32

    %bias_empty = tensor.empty(%output_channels) : tensor<?xf16>
    %bias_f16 = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%bias : tensor<?xf32>) outs(%bias_empty : tensor<?xf16>) {
      ^bb0(%value: f32, %out: f16):
        %narrowed = arith.truncf %value : f32 to f16
        linalg.yield %narrowed : f16
    } -> tensor<?xf16>

    %input_nhwc_empty = tensor.empty(%input_height, %input_width, %input_channels) : tensor<1x?x?x?xf16>
    %input_nhwc = linalg.transpose
        ins(%input : tensor<1x?x?x?xf16>)
        outs(%input_nhwc_empty : tensor<1x?x?x?xf16>)
        permutation = [0, 2, 3, 1]

    %raw_f16 = flow.dispatch
        @rocket_dynamic_depthwise_relu6_executable_s2::@rocket_dynamic_depthwise_conv2d_v1::@rocket_dynamic_depthwise_conv2d(
          %input_width_i32, %input_height_i32, %input_channels_i32,
          %output_channels_i32, %weights_width_i32, %weights_height_i32,
          %input_nhwc, %filter, %bias_f16)
        {stream.affinity = #hal.device.affinity<@rocket_device>}
        : (i32, i32, i32, i32, i32, i32,
           tensor<1x?x?x?xf16>{%input_height, %input_width, %input_channels},
           tensor<?x?x?xf16>{%input_channels, %weights_height, %weights_width},
           tensor<?xf16>{%output_channels})
        -> tensor<1x?x?x?xf16>{%output_height, %output_width, %output_channels}

    %final_nhwc = flow.dispatch.workgroups[
        %output_height, %output_width, %output_channels](
        %raw_f16, %output_height, %output_width, %output_channels)
        : (tensor<1x?x?x?xf16>{%output_height, %output_width, %output_channels},
           index, index, index)
        -> tensor<1x?x?x?xf32>{%output_height, %output_width, %output_channels}
        attributes { stream.affinity = #hal.device.affinity<@cpu_device> } =
        (%raw_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>,
         %output_height_arg: index,
         %output_width_arg: index,
         %output_channels_arg: index,
         %final_binding: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>) {
      %output_height_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_height_arg, 0 : index
      %output_width_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_width_arg, 1 : index
      %output_channels_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_channels_arg, 2 : index
      %raw_shaped = flow.dispatch.tie_shape %raw_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %final_shaped = flow.dispatch.tie_shape %final_binding
          : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %raw_loaded = iree_tensor_ext.dispatch.tensor.load %raw_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %output_channels_size}
          -> tensor<1x?x?x?xf16>
      %final_empty = tensor.empty(
          %output_height_size, %output_width_size, %output_channels_size)
          : tensor<1x?x?x?xf32>
      %final_inner = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
          ],
          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } ins(%raw_loaded : tensor<1x?x?x?xf16>)
          outs(%final_empty : tensor<1x?x?x?xf32>) {
        ^bb0(%raw: f16, %out: f32):
          %raw_f32 = arith.extf %raw : f16 to f32
          linalg.yield %raw_f32 : f32
      } -> tensor<1x?x?x?xf32>
      iree_tensor_ext.dispatch.tensor.store %final_inner, %final_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : tensor<1x?x?x?xf32>
          -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      flow.return
    } count(%output_height_workload: index,
            %output_width_workload: index,
            %output_channels_workload: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(
          %output_height_workload,
          %output_width_workload,
          %output_channels_workload)
      flow.return %x, %y, %z : index, index, index
    }

    %final_nchw_empty = tensor.empty(%output_channels, %output_height, %output_width) : tensor<1x?x?x?xf32>
    %final_nchw = linalg.transpose
        ins(%final_nhwc : tensor<1x?x?x?xf32>)
        outs(%final_nchw_empty : tensor<1x?x?x?xf32>)
        permutation = [0, 3, 1, 2]

    util.return %final_nchw : tensor<1x?x?x?xf32>
  }

  util.func private @call_rocket_dynamic_depthwise_conv2d_nchw(
      %input: tensor<1x?x?x?xf16>,
      %filter: tensor<?x?x?xf16>,
      %init: tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    // NCHW: dim 1 is channels, dims 2/3 are the spatial extent.
    %input_channels = tensor.dim %input, %c1 : tensor<1x?x?x?xf16>
    %input_height = tensor.dim %input, %c2 : tensor<1x?x?x?xf16>
    %input_width = tensor.dim %input, %c3 : tensor<1x?x?x?xf16>
    // Filter is [c][kh][kw]: dims 1/2 are the kernel extent.
    %weights_height = tensor.dim %filter, %c1 : tensor<?x?x?xf16>
    %weights_width = tensor.dim %filter, %c2 : tensor<?x?x?xf16>
    %output_channels = tensor.dim %init, %c1 : tensor<1x?x?x?xf32>
    %output_height = tensor.dim %init, %c2 : tensor<1x?x?x?xf32>
    %output_width = tensor.dim %init, %c3 : tensor<1x?x?x?xf32>

    %input_width_i32 = arith.index_cast %input_width : index to i32
    %input_height_i32 = arith.index_cast %input_height : index to i32
    %input_channels_i32 = arith.index_cast %input_channels : index to i32
    %output_channels_i32 = arith.index_cast %output_channels : index to i32
    %weights_width_i32 = arith.index_cast %weights_width : index to i32
    %weights_height_i32 = arith.index_cast %weights_height : index to i32

    %zero_bias_empty = tensor.empty(%output_channels) : tensor<?xf16>
    %zero_f16 = arith.constant 0.0 : f16
    %zero_bias = linalg.fill ins(%zero_f16 : f16)
        outs(%zero_bias_empty : tensor<?xf16>) -> tensor<?xf16>

    // NCHW [1,C,H,W] -> NHWC [1,H,W,C]: out.shape[i] = in.shape[perm[i]],
    // so perm = [0, 2, 3, 1].
    %input_nhwc_empty = tensor.empty(%input_height, %input_width, %input_channels) : tensor<1x?x?x?xf16>
    %input_nhwc = linalg.transpose
        ins(%input : tensor<1x?x?x?xf16>)
        outs(%input_nhwc_empty : tensor<1x?x?x?xf16>)
        permutation = [0, 2, 3, 1]

    %raw_f16 = flow.dispatch
        @rocket_dynamic_depthwise_executable::@rocket_dynamic_depthwise_conv2d_v1::@rocket_dynamic_depthwise_conv2d(
          %input_width_i32, %input_height_i32, %input_channels_i32,
          %output_channels_i32, %weights_width_i32, %weights_height_i32,
          %input_nhwc, %filter, %zero_bias)
        {stream.affinity = #hal.device.affinity<@rocket_device>}
        : (i32, i32, i32, i32, i32, i32,
           tensor<1x?x?x?xf16>{%input_height, %input_width, %input_channels},
           tensor<?x?x?xf16>{%input_channels, %weights_height, %weights_width},
           tensor<?xf16>{%output_channels})
        -> tensor<1x?x?x?xf16>{%output_height, %output_width, %output_channels}

    // %init also arrives NCHW; transpose it to NHWC too so it lines up with
    // %raw_f16 for the CPU-side accumulate below.
    %init_nhwc_empty = tensor.empty(%output_height, %output_width, %output_channels) : tensor<1x?x?x?xf32>
    %init_nhwc = linalg.transpose
        ins(%init : tensor<1x?x?x?xf32>)
        outs(%init_nhwc_empty : tensor<1x?x?x?xf32>)
        permutation = [0, 2, 3, 1]

    %final_nhwc = flow.dispatch.workgroups[
        %output_height, %output_width, %output_channels](
        %raw_f16, %init_nhwc, %output_height, %output_width, %output_channels)
        : (tensor<1x?x?x?xf16>{%output_height, %output_width, %output_channels},
           tensor<1x?x?x?xf32>{%output_height, %output_width, %output_channels},
           index, index, index)
        -> tensor<1x?x?x?xf32>{%output_height, %output_width, %output_channels}
        attributes { stream.affinity = #hal.device.affinity<@cpu_device> } =
        (%raw_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>,
         %init_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>,
         %output_height_arg: index,
         %output_width_arg: index,
         %output_channels_arg: index,
         %final_binding: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>) {
      %output_height_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_height_arg, 0 : index
      %output_width_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_width_arg, 1 : index
      %output_channels_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_channels_arg, 2 : index
      %raw_shaped = flow.dispatch.tie_shape %raw_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %init_shaped = flow.dispatch.tie_shape %init_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %final_shaped = flow.dispatch.tie_shape %final_binding
          : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %raw_loaded = iree_tensor_ext.dispatch.tensor.load %raw_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %output_channels_size}
          -> tensor<1x?x?x?xf16>
      %init_loaded = iree_tensor_ext.dispatch.tensor.load %init_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
          -> tensor<1x?x?x?xf32>
      %final_empty = tensor.empty(
          %output_height_size, %output_width_size, %output_channels_size)
          : tensor<1x?x?x?xf32>
      %final_inner = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
          ],
          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } ins(%raw_loaded, %init_loaded
            : tensor<1x?x?x?xf16>, tensor<1x?x?x?xf32>)
          outs(%final_empty : tensor<1x?x?x?xf32>) {
        ^bb0(%raw: f16, %initial: f32, %out: f32):
          %raw_f32 = arith.extf %raw : f16 to f32
          %sum = arith.addf %raw_f32, %initial : f32
          linalg.yield %sum : f32
      } -> tensor<1x?x?x?xf32>
      iree_tensor_ext.dispatch.tensor.store %final_inner, %final_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : tensor<1x?x?x?xf32>
          -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      flow.return
    } count(%output_height_workload: index,
            %output_width_workload: index,
            %output_channels_workload: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(
          %output_height_workload,
          %output_width_workload,
          %output_channels_workload)
      flow.return %x, %y, %z : index, index, index
    }

    // NHWC [1,H,W,C] -> NCHW [1,C,H,W] back again: perm = [0, 3, 1, 2].
    %final_nchw_empty = tensor.empty(%output_channels, %output_height, %output_width) : tensor<1x?x?x?xf32>
    %final_nchw = linalg.transpose
        ins(%final_nhwc : tensor<1x?x?x?xf32>)
        outs(%final_nchw_empty : tensor<1x?x?x?xf32>)
        permutation = [0, 3, 1, 2]

    util.return %final_nchw : tensor<1x?x?x?xf32>
  }

  // Stride-2 counterpart of call_rocket_dynamic_depthwise_conv2d_nchw --
  // same NCHW<->NHWC transpose shim, dispatching to
  // @rocket_dynamic_depthwise_executable_s2 (#rocket_dynamic_depthwise_target_s2
  // bakes stride=2, same reason call_rocket_dynamic_conv2d_s2 needs its own
  // executable -- see that comment). Hardware-confirmed for depthwise fp16
  // at stride 2/3/4, both 1x1 and 3x3 kernels, Cin/Cout 8 and 12
  // (conv_depthwise_stride_hw.rs::depthwise_strided_convs_run_on_npu, all
  // 12 combinations pass) -- the stride/DW_EN combination
  // call_rocket_dynamic_depthwise_conv2d_nchw's own doc comment notes as
  // untested is now covered; weight packing itself has no stride
  // dependence to re-check, so this reuses the same
  // pack_depthwise_to_rocket_weights path unchanged.
  util.func private @call_rocket_dynamic_depthwise_conv2d_nchw_s2(
      %input: tensor<1x?x?x?xf16>,
      %filter: tensor<?x?x?xf16>,
      %init: tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    // NCHW: dim 1 is channels, dims 2/3 are the spatial extent.
    %input_channels = tensor.dim %input, %c1 : tensor<1x?x?x?xf16>
    %input_height = tensor.dim %input, %c2 : tensor<1x?x?x?xf16>
    %input_width = tensor.dim %input, %c3 : tensor<1x?x?x?xf16>
    // Filter is [c][kh][kw]: dims 1/2 are the kernel extent.
    %weights_height = tensor.dim %filter, %c1 : tensor<?x?x?xf16>
    %weights_width = tensor.dim %filter, %c2 : tensor<?x?x?xf16>
    %output_channels = tensor.dim %init, %c1 : tensor<1x?x?x?xf32>
    %output_height = tensor.dim %init, %c2 : tensor<1x?x?x?xf32>
    %output_width = tensor.dim %init, %c3 : tensor<1x?x?x?xf32>

    %input_width_i32 = arith.index_cast %input_width : index to i32
    %input_height_i32 = arith.index_cast %input_height : index to i32
    %input_channels_i32 = arith.index_cast %input_channels : index to i32
    %output_channels_i32 = arith.index_cast %output_channels : index to i32
    %weights_width_i32 = arith.index_cast %weights_width : index to i32
    %weights_height_i32 = arith.index_cast %weights_height : index to i32

    %zero_bias_empty = tensor.empty(%output_channels) : tensor<?xf16>
    %zero_f16 = arith.constant 0.0 : f16
    %zero_bias = linalg.fill ins(%zero_f16 : f16)
        outs(%zero_bias_empty : tensor<?xf16>) -> tensor<?xf16>

    // NCHW [1,C,H,W] -> NHWC [1,H,W,C]: out.shape[i] = in.shape[perm[i]],
    // so perm = [0, 2, 3, 1].
    %input_nhwc_empty = tensor.empty(%input_height, %input_width, %input_channels) : tensor<1x?x?x?xf16>
    %input_nhwc = linalg.transpose
        ins(%input : tensor<1x?x?x?xf16>)
        outs(%input_nhwc_empty : tensor<1x?x?x?xf16>)
        permutation = [0, 2, 3, 1]

    %raw_f16 = flow.dispatch
        @rocket_dynamic_depthwise_executable_s2::@rocket_dynamic_depthwise_conv2d_v1::@rocket_dynamic_depthwise_conv2d(
          %input_width_i32, %input_height_i32, %input_channels_i32,
          %output_channels_i32, %weights_width_i32, %weights_height_i32,
          %input_nhwc, %filter, %zero_bias)
        {stream.affinity = #hal.device.affinity<@rocket_device>}
        : (i32, i32, i32, i32, i32, i32,
           tensor<1x?x?x?xf16>{%input_height, %input_width, %input_channels},
           tensor<?x?x?xf16>{%input_channels, %weights_height, %weights_width},
           tensor<?xf16>{%output_channels})
        -> tensor<1x?x?x?xf16>{%output_height, %output_width, %output_channels}

    // %init also arrives NCHW; transpose it to NHWC too so it lines up with
    // %raw_f16 for the CPU-side accumulate below.
    %init_nhwc_empty = tensor.empty(%output_height, %output_width, %output_channels) : tensor<1x?x?x?xf32>
    %init_nhwc = linalg.transpose
        ins(%init : tensor<1x?x?x?xf32>)
        outs(%init_nhwc_empty : tensor<1x?x?x?xf32>)
        permutation = [0, 2, 3, 1]

    %final_nhwc = flow.dispatch.workgroups[
        %output_height, %output_width, %output_channels](
        %raw_f16, %init_nhwc, %output_height, %output_width, %output_channels)
        : (tensor<1x?x?x?xf16>{%output_height, %output_width, %output_channels},
           tensor<1x?x?x?xf32>{%output_height, %output_width, %output_channels},
           index, index, index)
        -> tensor<1x?x?x?xf32>{%output_height, %output_width, %output_channels}
        attributes { stream.affinity = #hal.device.affinity<@cpu_device> } =
        (%raw_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>,
         %init_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>,
         %output_height_arg: index,
         %output_width_arg: index,
         %output_channels_arg: index,
         %final_binding: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>) {
      %output_height_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_height_arg, 0 : index
      %output_width_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_width_arg, 1 : index
      %output_channels_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_channels_arg, 2 : index
      %raw_shaped = flow.dispatch.tie_shape %raw_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %init_shaped = flow.dispatch.tie_shape %init_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %final_shaped = flow.dispatch.tie_shape %final_binding
          : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %raw_loaded = iree_tensor_ext.dispatch.tensor.load %raw_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %output_channels_size}
          -> tensor<1x?x?x?xf16>
      %init_loaded = iree_tensor_ext.dispatch.tensor.load %init_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
          -> tensor<1x?x?x?xf32>
      %final_empty = tensor.empty(
          %output_height_size, %output_width_size, %output_channels_size)
          : tensor<1x?x?x?xf32>
      %final_inner = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
          ],
          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } ins(%raw_loaded, %init_loaded
            : tensor<1x?x?x?xf16>, tensor<1x?x?x?xf32>)
          outs(%final_empty : tensor<1x?x?x?xf32>) {
        ^bb0(%raw: f16, %initial: f32, %out: f32):
          %raw_f32 = arith.extf %raw : f16 to f32
          %sum = arith.addf %raw_f32, %initial : f32
          linalg.yield %sum : f32
      } -> tensor<1x?x?x?xf32>
      iree_tensor_ext.dispatch.tensor.store %final_inner, %final_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : tensor<1x?x?x?xf32>
          -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      flow.return
    } count(%output_height_workload: index,
            %output_width_workload: index,
            %output_channels_workload: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(
          %output_height_workload,
          %output_width_workload,
          %output_channels_workload)
      flow.return %x, %y, %z : index, index, index
    }

    // NHWC [1,H,W,C] -> NCHW [1,C,H,W] back again: perm = [0, 3, 1, 2].
    %final_nchw_empty = tensor.empty(%output_channels, %output_height, %output_width) : tensor<1x?x?x?xf32>
    %final_nchw = linalg.transpose
        ins(%final_nhwc : tensor<1x?x?x?xf32>)
        outs(%final_nchw_empty : tensor<1x?x?x?xf32>)
        permutation = [0, 3, 1, 2]

    util.return %final_nchw : tensor<1x?x?x?xf32>
  }

  // Stride-3 counterpart of call_rocket_dynamic_depthwise_conv2d_nchw --
  // same NCHW<->NHWC transpose shim, dispatching to
  // @rocket_dynamic_depthwise_executable_s3 (#rocket_dynamic_depthwise_target_s3
  // bakes stride=3, same reason call_rocket_dynamic_conv2d_s3 needs its own
  // executable -- see that comment). Hardware-confirmed for depthwise fp16
  // at stride 2/3/4, both 1x1 and 3x3 kernels, Cin/Cout 8 and 12
  // (conv_depthwise_stride_hw.rs::depthwise_strided_convs_run_on_npu, all
  // 12 combinations pass) -- the stride/DW_EN combination
  // call_rocket_dynamic_depthwise_conv2d_nchw's own doc comment notes as
  // untested is now covered; weight packing itself has no stride
  // dependence to re-check, so this reuses the same
  // pack_depthwise_to_rocket_weights path unchanged.
  util.func private @call_rocket_dynamic_depthwise_conv2d_nchw_s3(
      %input: tensor<1x?x?x?xf16>,
      %filter: tensor<?x?x?xf16>,
      %init: tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    // NCHW: dim 1 is channels, dims 2/3 are the spatial extent.
    %input_channels = tensor.dim %input, %c1 : tensor<1x?x?x?xf16>
    %input_height = tensor.dim %input, %c2 : tensor<1x?x?x?xf16>
    %input_width = tensor.dim %input, %c3 : tensor<1x?x?x?xf16>
    // Filter is [c][kh][kw]: dims 1/2 are the kernel extent.
    %weights_height = tensor.dim %filter, %c1 : tensor<?x?x?xf16>
    %weights_width = tensor.dim %filter, %c2 : tensor<?x?x?xf16>
    %output_channels = tensor.dim %init, %c1 : tensor<1x?x?x?xf32>
    %output_height = tensor.dim %init, %c2 : tensor<1x?x?x?xf32>
    %output_width = tensor.dim %init, %c3 : tensor<1x?x?x?xf32>

    %input_width_i32 = arith.index_cast %input_width : index to i32
    %input_height_i32 = arith.index_cast %input_height : index to i32
    %input_channels_i32 = arith.index_cast %input_channels : index to i32
    %output_channels_i32 = arith.index_cast %output_channels : index to i32
    %weights_width_i32 = arith.index_cast %weights_width : index to i32
    %weights_height_i32 = arith.index_cast %weights_height : index to i32

    %zero_bias_empty = tensor.empty(%output_channels) : tensor<?xf16>
    %zero_f16 = arith.constant 0.0 : f16
    %zero_bias = linalg.fill ins(%zero_f16 : f16)
        outs(%zero_bias_empty : tensor<?xf16>) -> tensor<?xf16>

    // NCHW [1,C,H,W] -> NHWC [1,H,W,C]: out.shape[i] = in.shape[perm[i]],
    // so perm = [0, 2, 3, 1].
    %input_nhwc_empty = tensor.empty(%input_height, %input_width, %input_channels) : tensor<1x?x?x?xf16>
    %input_nhwc = linalg.transpose
        ins(%input : tensor<1x?x?x?xf16>)
        outs(%input_nhwc_empty : tensor<1x?x?x?xf16>)
        permutation = [0, 2, 3, 1]

    %raw_f16 = flow.dispatch
        @rocket_dynamic_depthwise_executable_s3::@rocket_dynamic_depthwise_conv2d_v1::@rocket_dynamic_depthwise_conv2d(
          %input_width_i32, %input_height_i32, %input_channels_i32,
          %output_channels_i32, %weights_width_i32, %weights_height_i32,
          %input_nhwc, %filter, %zero_bias)
        {stream.affinity = #hal.device.affinity<@rocket_device>}
        : (i32, i32, i32, i32, i32, i32,
           tensor<1x?x?x?xf16>{%input_height, %input_width, %input_channels},
           tensor<?x?x?xf16>{%input_channels, %weights_height, %weights_width},
           tensor<?xf16>{%output_channels})
        -> tensor<1x?x?x?xf16>{%output_height, %output_width, %output_channels}

    // %init also arrives NCHW; transpose it to NHWC too so it lines up with
    // %raw_f16 for the CPU-side accumulate below.
    %init_nhwc_empty = tensor.empty(%output_height, %output_width, %output_channels) : tensor<1x?x?x?xf32>
    %init_nhwc = linalg.transpose
        ins(%init : tensor<1x?x?x?xf32>)
        outs(%init_nhwc_empty : tensor<1x?x?x?xf32>)
        permutation = [0, 2, 3, 1]

    %final_nhwc = flow.dispatch.workgroups[
        %output_height, %output_width, %output_channels](
        %raw_f16, %init_nhwc, %output_height, %output_width, %output_channels)
        : (tensor<1x?x?x?xf16>{%output_height, %output_width, %output_channels},
           tensor<1x?x?x?xf32>{%output_height, %output_width, %output_channels},
           index, index, index)
        -> tensor<1x?x?x?xf32>{%output_height, %output_width, %output_channels}
        attributes { stream.affinity = #hal.device.affinity<@cpu_device> } =
        (%raw_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>,
         %init_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>,
         %output_height_arg: index,
         %output_width_arg: index,
         %output_channels_arg: index,
         %final_binding: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>) {
      %output_height_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_height_arg, 0 : index
      %output_width_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_width_arg, 1 : index
      %output_channels_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_channels_arg, 2 : index
      %raw_shaped = flow.dispatch.tie_shape %raw_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %init_shaped = flow.dispatch.tie_shape %init_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %final_shaped = flow.dispatch.tie_shape %final_binding
          : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %raw_loaded = iree_tensor_ext.dispatch.tensor.load %raw_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %output_channels_size}
          -> tensor<1x?x?x?xf16>
      %init_loaded = iree_tensor_ext.dispatch.tensor.load %init_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
          -> tensor<1x?x?x?xf32>
      %final_empty = tensor.empty(
          %output_height_size, %output_width_size, %output_channels_size)
          : tensor<1x?x?x?xf32>
      %final_inner = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
          ],
          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } ins(%raw_loaded, %init_loaded
            : tensor<1x?x?x?xf16>, tensor<1x?x?x?xf32>)
          outs(%final_empty : tensor<1x?x?x?xf32>) {
        ^bb0(%raw: f16, %initial: f32, %out: f32):
          %raw_f32 = arith.extf %raw : f16 to f32
          %sum = arith.addf %raw_f32, %initial : f32
          linalg.yield %sum : f32
      } -> tensor<1x?x?x?xf32>
      iree_tensor_ext.dispatch.tensor.store %final_inner, %final_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : tensor<1x?x?x?xf32>
          -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      flow.return
    } count(%output_height_workload: index,
            %output_width_workload: index,
            %output_channels_workload: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(
          %output_height_workload,
          %output_width_workload,
          %output_channels_workload)
      flow.return %x, %y, %z : index, index, index
    }

    // NHWC [1,H,W,C] -> NCHW [1,C,H,W] back again: perm = [0, 3, 1, 2].
    %final_nchw_empty = tensor.empty(%output_channels, %output_height, %output_width) : tensor<1x?x?x?xf32>
    %final_nchw = linalg.transpose
        ins(%final_nhwc : tensor<1x?x?x?xf32>)
        outs(%final_nchw_empty : tensor<1x?x?x?xf32>)
        permutation = [0, 3, 1, 2]

    util.return %final_nchw : tensor<1x?x?x?xf32>
  }

  // Stride-4 counterpart of call_rocket_dynamic_depthwise_conv2d_nchw --
  // same NCHW<->NHWC transpose shim, dispatching to
  // @rocket_dynamic_depthwise_executable_s4 (#rocket_dynamic_depthwise_target_s4
  // bakes stride=4, same reason call_rocket_dynamic_conv2d_s4 needs its own
  // executable -- see that comment). Hardware-confirmed for depthwise fp16
  // at stride 2/3/4, both 1x1 and 3x3 kernels, Cin/Cout 8 and 12
  // (conv_depthwise_stride_hw.rs::depthwise_strided_convs_run_on_npu, all
  // 12 combinations pass) -- the stride/DW_EN combination
  // call_rocket_dynamic_depthwise_conv2d_nchw's own doc comment notes as
  // untested is now covered; weight packing itself has no stride
  // dependence to re-check, so this reuses the same
  // pack_depthwise_to_rocket_weights path unchanged.
  util.func private @call_rocket_dynamic_depthwise_conv2d_nchw_s4(
      %input: tensor<1x?x?x?xf16>,
      %filter: tensor<?x?x?xf16>,
      %init: tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    // NCHW: dim 1 is channels, dims 2/3 are the spatial extent.
    %input_channels = tensor.dim %input, %c1 : tensor<1x?x?x?xf16>
    %input_height = tensor.dim %input, %c2 : tensor<1x?x?x?xf16>
    %input_width = tensor.dim %input, %c3 : tensor<1x?x?x?xf16>
    // Filter is [c][kh][kw]: dims 1/2 are the kernel extent.
    %weights_height = tensor.dim %filter, %c1 : tensor<?x?x?xf16>
    %weights_width = tensor.dim %filter, %c2 : tensor<?x?x?xf16>
    %output_channels = tensor.dim %init, %c1 : tensor<1x?x?x?xf32>
    %output_height = tensor.dim %init, %c2 : tensor<1x?x?x?xf32>
    %output_width = tensor.dim %init, %c3 : tensor<1x?x?x?xf32>

    %input_width_i32 = arith.index_cast %input_width : index to i32
    %input_height_i32 = arith.index_cast %input_height : index to i32
    %input_channels_i32 = arith.index_cast %input_channels : index to i32
    %output_channels_i32 = arith.index_cast %output_channels : index to i32
    %weights_width_i32 = arith.index_cast %weights_width : index to i32
    %weights_height_i32 = arith.index_cast %weights_height : index to i32

    %zero_bias_empty = tensor.empty(%output_channels) : tensor<?xf16>
    %zero_f16 = arith.constant 0.0 : f16
    %zero_bias = linalg.fill ins(%zero_f16 : f16)
        outs(%zero_bias_empty : tensor<?xf16>) -> tensor<?xf16>

    // NCHW [1,C,H,W] -> NHWC [1,H,W,C]: out.shape[i] = in.shape[perm[i]],
    // so perm = [0, 2, 3, 1].
    %input_nhwc_empty = tensor.empty(%input_height, %input_width, %input_channels) : tensor<1x?x?x?xf16>
    %input_nhwc = linalg.transpose
        ins(%input : tensor<1x?x?x?xf16>)
        outs(%input_nhwc_empty : tensor<1x?x?x?xf16>)
        permutation = [0, 2, 3, 1]

    %raw_f16 = flow.dispatch
        @rocket_dynamic_depthwise_executable_s4::@rocket_dynamic_depthwise_conv2d_v1::@rocket_dynamic_depthwise_conv2d(
          %input_width_i32, %input_height_i32, %input_channels_i32,
          %output_channels_i32, %weights_width_i32, %weights_height_i32,
          %input_nhwc, %filter, %zero_bias)
        {stream.affinity = #hal.device.affinity<@rocket_device>}
        : (i32, i32, i32, i32, i32, i32,
           tensor<1x?x?x?xf16>{%input_height, %input_width, %input_channels},
           tensor<?x?x?xf16>{%input_channels, %weights_height, %weights_width},
           tensor<?xf16>{%output_channels})
        -> tensor<1x?x?x?xf16>{%output_height, %output_width, %output_channels}

    // %init also arrives NCHW; transpose it to NHWC too so it lines up with
    // %raw_f16 for the CPU-side accumulate below.
    %init_nhwc_empty = tensor.empty(%output_height, %output_width, %output_channels) : tensor<1x?x?x?xf32>
    %init_nhwc = linalg.transpose
        ins(%init : tensor<1x?x?x?xf32>)
        outs(%init_nhwc_empty : tensor<1x?x?x?xf32>)
        permutation = [0, 2, 3, 1]

    %final_nhwc = flow.dispatch.workgroups[
        %output_height, %output_width, %output_channels](
        %raw_f16, %init_nhwc, %output_height, %output_width, %output_channels)
        : (tensor<1x?x?x?xf16>{%output_height, %output_width, %output_channels},
           tensor<1x?x?x?xf32>{%output_height, %output_width, %output_channels},
           index, index, index)
        -> tensor<1x?x?x?xf32>{%output_height, %output_width, %output_channels}
        attributes { stream.affinity = #hal.device.affinity<@cpu_device> } =
        (%raw_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>,
         %init_binding: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>,
         %output_height_arg: index,
         %output_width_arg: index,
         %output_channels_arg: index,
         %final_binding: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>) {
      %output_height_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_height_arg, 0 : index
      %output_width_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_width_arg, 1 : index
      %output_channels_size = iree_tensor_ext.dispatch.workload.ordinal
          %output_channels_arg, 2 : index
      %raw_shaped = flow.dispatch.tie_shape %raw_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %init_shaped = flow.dispatch.tie_shape %init_binding
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %final_shaped = flow.dispatch.tie_shape %final_binding
          : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      %raw_loaded = iree_tensor_ext.dispatch.tensor.load %raw_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf16>>{
              %output_height_size, %output_width_size, %output_channels_size}
          -> tensor<1x?x?x?xf16>
      %init_loaded = iree_tensor_ext.dispatch.tensor.load %init_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
          -> tensor<1x?x?x?xf32>
      %final_empty = tensor.empty(
          %output_height_size, %output_width_size, %output_channels_size)
          : tensor<1x?x?x?xf32>
      %final_inner = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
          ],
          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } ins(%raw_loaded, %init_loaded
            : tensor<1x?x?x?xf16>, tensor<1x?x?x?xf32>)
          outs(%final_empty : tensor<1x?x?x?xf32>) {
        ^bb0(%raw: f16, %initial: f32, %out: f32):
          %raw_f32 = arith.extf %raw : f16 to f32
          %sum = arith.addf %raw_f32, %initial : f32
          linalg.yield %sum : f32
      } -> tensor<1x?x?x?xf32>
      iree_tensor_ext.dispatch.tensor.store %final_inner, %final_shaped,
          offsets = [0, 0, 0, 0],
          sizes = [1, %output_height_size, %output_width_size, %output_channels_size],
          strides = [1, 1, 1, 1]
          : tensor<1x?x?x?xf32>
          -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x?x?xf32>>{
              %output_height_size, %output_width_size, %output_channels_size}
      flow.return
    } count(%output_height_workload: index,
            %output_width_workload: index,
            %output_channels_workload: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(
          %output_height_workload,
          %output_width_workload,
          %output_channels_workload)
      flow.return %x, %y, %z : index, index, index
    }

    // NHWC [1,H,W,C] -> NCHW [1,C,H,W] back again: perm = [0, 3, 1, 2].
    %final_nchw_empty = tensor.empty(%output_channels, %output_height, %output_width) : tensor<1x?x?x?xf32>
    %final_nchw = linalg.transpose
        ins(%final_nhwc : tensor<1x?x?x?xf32>)
        outs(%final_nchw_empty : tensor<1x?x?x?xf32>)
        permutation = [0, 3, 1, 2]

    util.return %final_nchw : tensor<1x?x?x?xf32>
  }


  // 1x1 kernel, spatial dims dynamic. Both channel counts must be provably
  // <= 512 (MAX_INPUT_CHANNELS/MAX_OUTPUT_CHANNELS in iree-rocket-hal's
  // conv.rs -- the 14-bit weight_kernels field's range): a convolution whose
  // channel count the compiler cannot bound must not be claimed here, since
  // the hardware rejects an out-of-range dispatch with no CPU fallback. The
  // input handle's dim 3 is Cin; the filter's dim 3 is Cout.

  // int8 counterpart of call_rocket_dynamic_conv2d. Same six push constants
  // in the same order, same three tensor bindings; the operands are i8 and
  // the result is the i32 accumulator int8_accumulator mode writes out, so
  // the CPU epilogue only has to add the convolution's own init operand
  // back in.
  util.func private @call_rocket_dynamic_conv2d_int8(
      %input: tensor<1x?x?x?xi8>,
      %filter: tensor<?x?x?x?xi8>,
      %init: tensor<1x?x?x?xi32>) -> tensor<1x?x?x?xi32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    %input_height = tensor.dim %input, %c1 : tensor<1x?x?x?xi8>
    %input_width = tensor.dim %input, %c2 : tensor<1x?x?x?xi8>
    %input_channels = tensor.dim %input, %c3 : tensor<1x?x?x?xi8>
    %weights_height = tensor.dim %filter, %c0 : tensor<?x?x?x?xi8>
    %weights_width = tensor.dim %filter, %c1 : tensor<?x?x?x?xi8>
    %output_height = tensor.dim %init, %c1 : tensor<1x?x?x?xi32>
    %output_width = tensor.dim %init, %c2 : tensor<1x?x?x?xi32>
    %output_channels = tensor.dim %init, %c3 : tensor<1x?x?x?xi32>

    %input_width_i32 = arith.index_cast %input_width : index to i32
    %input_height_i32 = arith.index_cast %input_height : index to i32
    %input_channels_i32 = arith.index_cast %input_channels : index to i32
    %output_channels_i32 = arith.index_cast %output_channels : index to i32
    %weights_width_i32 = arith.index_cast %weights_width : index to i32
    %weights_height_i32 = arith.index_cast %weights_height : index to i32

    // The bias binding is i32 here, not i8: rocket-hal-driver validates it
    // as output_channels * 4 bytes for both int8 precisions and reads it as
    // i32 in pack_int8_bias_to_bs, whereas the fp16 path binds an f16
    // vector. It stays zero -- int8_accumulator bypasses the BS stage
    // entirely, so nothing here can be folded into a hardware bias; the
    // zero-point correction lives in a separate CPU op that
    // iree-global-opt-quantized-conv-to-conv already emitted next to this
    // convolution.
    %zero_bias_empty = tensor.empty(%output_channels) : tensor<?xi32>
    %zero_i32 = arith.constant 0 : i32
    %zero_bias = linalg.fill ins(%zero_i32 : i32)
        outs(%zero_bias_empty : tensor<?xi32>) -> tensor<?xi32>

    %raw_i32 = flow.dispatch
        @rocket_dynamic_int8_executable::@rocket_dynamic_conv2d_v1::@rocket_dynamic_conv2d(
          %input_width_i32, %input_height_i32, %input_channels_i32,
          %output_channels_i32, %weights_width_i32, %weights_height_i32,
          %input, %filter, %zero_bias)
        {stream.affinity = #hal.device.affinity<@rocket_device>}
        : (i32, i32, i32, i32, i32, i32,
           tensor<1x?x?x?xi8>{%input_height, %input_width, %input_channels},
           tensor<?x?x?x?xi8>{%weights_height, %weights_width, %input_channels, %output_channels},
           tensor<?xi32>{%output_channels})
        -> tensor<1x?x?x?xi32>{%output_height, %output_width, %output_channels}

    // The epilogue is a plain `linalg.generic`, not a hand-written
    // `flow.dispatch.workgroups`, and the `inline` pass at the end of
    // @__transform_main puts it in the caller. Both halves matter, and only
    // together: a pre-formed dispatch is opaque to dispatch-region formation,
    // so nothing downstream can fuse into it, and a `linalg.generic` left
    // inside a `util.func private` that is never inlined just becomes its own
    // dispatch anyway (the trap ISSUES.md P6 item 2 records for the
    // classifier matmul's truncf). Inlined and generic, IREE fuses this add
    // with the zero-point correction and requantization that follow it in
    // main_graph -- on MobileNetV2 int8 that is 265 dispatch sites down to
    // 145. It needs no `stream.affinity`: rocket-pin-unclaimed-dispatches
    // stamps the CPU default onto every dispatch this spec did not claim,
    // which is what the explicit @cpu_device annotation here used to be for.
    %final_empty_outer = tensor.empty(
        %output_height, %output_width, %output_channels) : tensor<1x?x?x?xi32>
    %final = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } ins(%raw_i32, %init : tensor<1x?x?x?xi32>, tensor<1x?x?x?xi32>)
        outs(%final_empty_outer : tensor<1x?x?x?xi32>) {
      ^bb0(%raw: i32, %initial: i32, %out: i32):
        // No extend, unlike the fp16 epilogue: int8_accumulator mode
        // already hands back a full i32 accumulator.
        %sum = arith.addi %raw, %initial : i32
        linalg.yield %sum : i32
    } -> tensor<1x?x?x?xi32>

    util.return %final : tensor<1x?x?x?xi32>
  }

  // Requantized int8 dense conv. Unlike every other adapter here this one has
  // no CPU epilogue at all: the DPU's BS stage adds the bias, its out-convert
  // stage applies the multiplier and output zero point, and the dispatch
  // returns quantized i8 directly.
  //
  // Two push constants follow the usual six dimensions, in the order the
  // target's `runtime_quantization` lists them. The scale travels as its
  // IEEE-754 bit pattern because a push constant is a uint32 -- see
  // Conv2DQuantParam in rocket_executable_def.fbs, which is where that
  // convention is written down.
  //
  // The bias binding is i32 and, unlike the accumulator adapter's, it is not
  // zero: this is the real per-output-channel bias, already in accumulator
  // units (`bias / (input_scale * weights_scale)`, which is what a
  // QLinearConv bias already is) with the input zero point's
  // `-x_zp * sum_k(w)` correction folded in by the producer.
  //
  // The argument list is the matched DAG's inputs, in the order its block
  // arguments declare them -- including `%acc_init`, the convolution's zero
  // init, and the two int8 clamp bounds, which the hardware applies itself
  // and which are here only because the CPU form has to name them somewhere.
  util.func private @call_rocket_dynamic_conv2d_int8_requant(
      %input: tensor<1x?x?x?xi8>,
      %filter: tensor<?x?x?x?xi8>,
      %acc_init: tensor<1x?x?x?xi32>,
      %bias: tensor<?xi32>,
      %output_scale: f32,
      %output_zero_point: i32,
      %int8_min: f32,
      %int8_max: f32,
      %init: tensor<1x?x?x?xi8>) -> tensor<1x?x?x?xi8> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    %input_height = tensor.dim %input, %c1 : tensor<1x?x?x?xi8>
    %input_width = tensor.dim %input, %c2 : tensor<1x?x?x?xi8>
    %input_channels = tensor.dim %input, %c3 : tensor<1x?x?x?xi8>
    %weights_height = tensor.dim %filter, %c0 : tensor<?x?x?x?xi8>
    %weights_width = tensor.dim %filter, %c1 : tensor<?x?x?x?xi8>
    %output_height = tensor.dim %init, %c1 : tensor<1x?x?x?xi8>
    %output_width = tensor.dim %init, %c2 : tensor<1x?x?x?xi8>
    %output_channels = tensor.dim %init, %c3 : tensor<1x?x?x?xi8>

    %input_width_i32 = arith.index_cast %input_width : index to i32
    %input_height_i32 = arith.index_cast %input_height : index to i32
    %input_channels_i32 = arith.index_cast %input_channels : index to i32
    %output_channels_i32 = arith.index_cast %output_channels : index to i32
    %weights_width_i32 = arith.index_cast %weights_width : index to i32
    %weights_height_i32 = arith.index_cast %weights_height : index to i32
    // The epilogue's operand is the requantization *multiplier*
    // `(x_scale * w_scale) / y_scale`, while the schema field is an output
    // scale that the driver divides into `input_scale * weights_scale` -- both
    // of which this target holds at 1.0. So what travels is the reciprocal.
    // Keeping the multiplier in the IR and inverting here means the canonical
    // form stays the one a reader of the convolution would write.
    %one = arith.constant 1.000000e+00 : f32
    %schema_output_scale = arith.divf %one, %output_scale : f32
    %output_scale_i32 = arith.bitcast %schema_output_scale : f32 to i32

    %quantized = flow.dispatch
        @rocket_dynamic_int8_requant_executable::@rocket_dynamic_conv2d_v1::@rocket_dynamic_conv2d(
          %input_width_i32, %input_height_i32, %input_channels_i32,
          %output_channels_i32, %weights_width_i32, %weights_height_i32,
          %output_scale_i32, %output_zero_point,
          %input, %filter, %bias)
        {stream.affinity = #hal.device.affinity<@rocket_device>}
        : (i32, i32, i32, i32, i32, i32, i32, i32,
           tensor<1x?x?x?xi8>{%input_height, %input_width, %input_channels},
           tensor<?x?x?x?xi8>{%weights_height, %weights_width, %input_channels, %output_channels},
           tensor<?xi32>{%output_channels})
        -> tensor<1x?x?x?xi8>{%output_height, %output_width, %output_channels}

    util.return %quantized : tensor<1x?x?x?xi8>
  }

  // Requantized depthwise. Unlike @call_rocket_dynamic_depthwise_conv2d_int8
  // there is no epilogue at all below the dispatch: the DPU's BS plane adds
  // the per-channel bias and its out-convert stage applies the multiplier and
  // output zero point, so what comes back is the final `i8` tensor. That
  // missing epilogue is the whole point -- it is one full-size `i32` pass per
  // depthwise layer that stops existing.
  //
  // The HWC -> CHW filter transpose stays, because it is the layout
  // `pack_depthwise_to_rocket_weights` reads and it const-evals away over a
  // constant filter.
  util.func private @call_rocket_dynamic_depthwise_conv2d_int8_requant(
      %input: tensor<1x?x?x?xi8>,
      %filter: tensor<?x?x?xi8>,
      %acc_init: tensor<1x?x?x?xi32>,
      %bias: tensor<?xi32>,
      %output_scale: f32,
      %output_zero_point: i32,
      %int8_min: f32,
      %int8_max: f32,
      %init: tensor<1x?x?x?xi8>) -> tensor<1x?x?x?xi8> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    %input_height = tensor.dim %input, %c1 : tensor<1x?x?x?xi8>
    %input_width = tensor.dim %input, %c2 : tensor<1x?x?x?xi8>
    %input_channels = tensor.dim %input, %c3 : tensor<1x?x?x?xi8>
    %weights_height = tensor.dim %filter, %c0 : tensor<?x?x?xi8>
    %weights_width = tensor.dim %filter, %c1 : tensor<?x?x?xi8>
    %output_height = tensor.dim %init, %c1 : tensor<1x?x?x?xi8>
    %output_width = tensor.dim %init, %c2 : tensor<1x?x?x?xi8>
    %output_channels = tensor.dim %init, %c3 : tensor<1x?x?x?xi8>

    %input_width_i32 = arith.index_cast %input_width : index to i32
    %input_height_i32 = arith.index_cast %input_height : index to i32
    %input_channels_i32 = arith.index_cast %input_channels : index to i32
    %output_channels_i32 = arith.index_cast %output_channels : index to i32
    %weights_width_i32 = arith.index_cast %weights_width : index to i32
    %weights_height_i32 = arith.index_cast %weights_height : index to i32

    // Same reciprocal convention as the dense requantized shim: the IR keeps
    // the multiplier a reader of the convolution would write, and the schema
    // field is the output scale the driver divides into `input_scale *
    // weights_scale`, both held at 1.0 by the target.
    %one = arith.constant 1.000000e+00 : f32
    %schema_output_scale = arith.divf %one, %output_scale : f32
    %output_scale_i32 = arith.bitcast %schema_output_scale : f32 to i32

    %filter_chw_empty = tensor.empty(%input_channels, %weights_height, %weights_width) : tensor<?x?x?xi8>
    %filter_chw = linalg.transpose
        ins(%filter : tensor<?x?x?xi8>)
        outs(%filter_chw_empty : tensor<?x?x?xi8>)
        permutation = [2, 0, 1]

    %quantized = flow.dispatch
        @rocket_dynamic_depthwise_int8_requant_executable::@rocket_dynamic_depthwise_conv2d_v1::@rocket_dynamic_depthwise_conv2d(
          %input_width_i32, %input_height_i32, %input_channels_i32,
          %output_channels_i32, %weights_width_i32, %weights_height_i32,
          %output_scale_i32, %output_zero_point,
          %input, %filter_chw, %bias)
        {stream.affinity = #hal.device.affinity<@rocket_device>}
        : (i32, i32, i32, i32, i32, i32, i32, i32,
           tensor<1x?x?x?xi8>{%input_height, %input_width, %input_channels},
           tensor<?x?x?xi8>{%input_channels, %weights_height, %weights_width},
           tensor<?xi32>{%output_channels})
        -> tensor<1x?x?x?xi8>{%output_height, %output_width, %output_channels}

    util.return %quantized : tensor<1x?x?x?xi8>
  }

  // int8 counterpart of call_rocket_dynamic_depthwise_conv2d.
  util.func private @call_rocket_dynamic_depthwise_conv2d_int8(
      %input: tensor<1x?x?x?xi8>,
      %filter: tensor<?x?x?xi8>,
      %init: tensor<1x?x?x?xi32>) -> tensor<1x?x?x?xi32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    %input_height = tensor.dim %input, %c1 : tensor<1x?x?x?xi8>
    %input_width = tensor.dim %input, %c2 : tensor<1x?x?x?xi8>
    %input_channels = tensor.dim %input, %c3 : tensor<1x?x?x?xi8>
    %weights_height = tensor.dim %filter, %c0 : tensor<?x?x?xi8>
    %weights_width = tensor.dim %filter, %c1 : tensor<?x?x?xi8>
    %output_height = tensor.dim %init, %c1 : tensor<1x?x?x?xi32>
    %output_width = tensor.dim %init, %c2 : tensor<1x?x?x?xi32>
    %output_channels = tensor.dim %init, %c3 : tensor<1x?x?x?xi32>

    %input_width_i32 = arith.index_cast %input_width : index to i32
    %input_height_i32 = arith.index_cast %input_height : index to i32
    %input_channels_i32 = arith.index_cast %input_channels : index to i32
    %output_channels_i32 = arith.index_cast %output_channels : index to i32
    %weights_width_i32 = arith.index_cast %weights_width : index to i32
    %weights_height_i32 = arith.index_cast %weights_height : index to i32

    // The bias binding is i32 here, not i8: rocket-hal-driver validates it
    // as output_channels * 4 bytes for both int8 precisions and reads it as
    // i32 in pack_int8_bias_to_bs, whereas the fp16 path binds an f16
    // vector. It stays zero -- int8_accumulator bypasses the BS stage
    // entirely, so nothing here can be folded into a hardware bias; the
    // zero-point correction lives in a separate CPU op that
    // iree-global-opt-quantized-conv-to-conv already emitted next to this
    // convolution.
    %zero_bias_empty = tensor.empty(%output_channels) : tensor<?xi32>
    %zero_i32 = arith.constant 0 : i32
    %zero_bias = linalg.fill ins(%zero_i32 : i32)
        outs(%zero_bias_empty : tensor<?xi32>) -> tensor<?xi32>

    // Same HWC -> CHW filter transpose call_rocket_dynamic_depthwise_conv2d
    // does, and for the same reason: pack_depthwise_to_rocket_weights takes
    // a [c][kh][kw] buffer, while linalg.depthwise_conv_2d_nhwc_hwc's filter
    // operand is [kh][kw][c].
    %filter_chw_empty = tensor.empty(%input_channels, %weights_height, %weights_width) : tensor<?x?x?xi8>
    %filter_chw = linalg.transpose
        ins(%filter : tensor<?x?x?xi8>)
        outs(%filter_chw_empty : tensor<?x?x?xi8>)
        permutation = [2, 0, 1]

    %raw_i32 = flow.dispatch
        @rocket_dynamic_depthwise_int8_executable::@rocket_dynamic_depthwise_conv2d_v1::@rocket_dynamic_depthwise_conv2d(
          %input_width_i32, %input_height_i32, %input_channels_i32,
          %output_channels_i32, %weights_width_i32, %weights_height_i32,
          %input, %filter_chw, %zero_bias)
        {stream.affinity = #hal.device.affinity<@rocket_device>}
        : (i32, i32, i32, i32, i32, i32,
           tensor<1x?x?x?xi8>{%input_height, %input_width, %input_channels},
           tensor<?x?x?xi8>{%input_channels, %weights_height, %weights_width},
           tensor<?xi32>{%output_channels})
        -> tensor<1x?x?x?xi32>{%output_height, %output_width, %output_channels}

    // The epilogue is a plain `linalg.generic`, not a hand-written
    // `flow.dispatch.workgroups`, and the `inline` pass at the end of
    // @__transform_main puts it in the caller. Both halves matter, and only
    // together: a pre-formed dispatch is opaque to dispatch-region formation,
    // so nothing downstream can fuse into it, and a `linalg.generic` left
    // inside a `util.func private` that is never inlined just becomes its own
    // dispatch anyway (the trap ISSUES.md P6 item 2 records for the
    // classifier matmul's truncf). Inlined and generic, IREE fuses this add
    // with the zero-point correction and requantization that follow it in
    // main_graph -- on MobileNetV2 int8 that is 265 dispatch sites down to
    // 145. It needs no `stream.affinity`: rocket-pin-unclaimed-dispatches
    // stamps the CPU default onto every dispatch this spec did not claim,
    // which is what the explicit @cpu_device annotation here used to be for.
    %final_empty_outer = tensor.empty(
        %output_height, %output_width, %output_channels) : tensor<1x?x?x?xi32>
    %final = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } ins(%raw_i32, %init : tensor<1x?x?x?xi32>, tensor<1x?x?x?xi32>)
        outs(%final_empty_outer : tensor<1x?x?x?xi32>) {
      ^bb0(%raw: i32, %initial: i32, %out: i32):
        // No extend, unlike the fp16 epilogue: int8_accumulator mode
        // already hands back a full i32 accumulator.
        %sum = arith.addi %raw, %initial : i32
        linalg.yield %sum : i32
    } -> tensor<1x?x?x?xi32>

    util.return %final : tensor<1x?x?x?xi32>
  }

  // Stride-2 int8 depthwise. The fp16 side reaches stride 2 only through
  // its NCHW adapters (call_rocket_dynamic_depthwise_conv2d_nchw_s2); there
  // is no fp16 NHWC stride-2 adapter because ONNX-imported fp16 depthwise
  // convs arrive NCHW and stay that way. An ONNX *int8* model is different:
  // torch-mlir lowers ConvInteger's grouped form straight to
  // linalg.depthwise_conv_2d_nhwc_hwc_q, so the NHWC layout is the one that
  // actually shows up (4 of MobileNetV2's 17 depthwise layers are stride 2).
  // This is a layout-plumbing addition, not a new hardware claim: it
  // dispatches to the same depthwise stride-2 executable shape the
  // hardware-confirmed NCHW s2 path already uses.
  util.func private @call_rocket_dynamic_depthwise_conv2d_int8_s2(
      %input: tensor<1x?x?x?xi8>,
      %filter: tensor<?x?x?xi8>,
      %init: tensor<1x?x?x?xi32>) -> tensor<1x?x?x?xi32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    %input_height = tensor.dim %input, %c1 : tensor<1x?x?x?xi8>
    %input_width = tensor.dim %input, %c2 : tensor<1x?x?x?xi8>
    %input_channels = tensor.dim %input, %c3 : tensor<1x?x?x?xi8>
    %weights_height = tensor.dim %filter, %c0 : tensor<?x?x?xi8>
    %weights_width = tensor.dim %filter, %c1 : tensor<?x?x?xi8>
    %output_height = tensor.dim %init, %c1 : tensor<1x?x?x?xi32>
    %output_width = tensor.dim %init, %c2 : tensor<1x?x?x?xi32>
    %output_channels = tensor.dim %init, %c3 : tensor<1x?x?x?xi32>

    %input_width_i32 = arith.index_cast %input_width : index to i32
    %input_height_i32 = arith.index_cast %input_height : index to i32
    %input_channels_i32 = arith.index_cast %input_channels : index to i32
    %output_channels_i32 = arith.index_cast %output_channels : index to i32
    %weights_width_i32 = arith.index_cast %weights_width : index to i32
    %weights_height_i32 = arith.index_cast %weights_height : index to i32

    // The bias binding is i32 here, not i8: rocket-hal-driver validates it
    // as output_channels * 4 bytes for both int8 precisions and reads it as
    // i32 in pack_int8_bias_to_bs, whereas the fp16 path binds an f16
    // vector. It stays zero -- int8_accumulator bypasses the BS stage
    // entirely, so nothing here can be folded into a hardware bias; the
    // zero-point correction lives in a separate CPU op that
    // iree-global-opt-quantized-conv-to-conv already emitted next to this
    // convolution.
    %zero_bias_empty = tensor.empty(%output_channels) : tensor<?xi32>
    %zero_i32 = arith.constant 0 : i32
    %zero_bias = linalg.fill ins(%zero_i32 : i32)
        outs(%zero_bias_empty : tensor<?xi32>) -> tensor<?xi32>

    // Same HWC -> CHW filter transpose call_rocket_dynamic_depthwise_conv2d
    // does, and for the same reason: pack_depthwise_to_rocket_weights takes
    // a [c][kh][kw] buffer, while linalg.depthwise_conv_2d_nhwc_hwc's filter
    // operand is [kh][kw][c].
    %filter_chw_empty = tensor.empty(%input_channels, %weights_height, %weights_width) : tensor<?x?x?xi8>
    %filter_chw = linalg.transpose
        ins(%filter : tensor<?x?x?xi8>)
        outs(%filter_chw_empty : tensor<?x?x?xi8>)
        permutation = [2, 0, 1]

    %raw_i32 = flow.dispatch
        @rocket_dynamic_depthwise_int8_executable_s2::@rocket_dynamic_depthwise_conv2d_v1::@rocket_dynamic_depthwise_conv2d(
          %input_width_i32, %input_height_i32, %input_channels_i32,
          %output_channels_i32, %weights_width_i32, %weights_height_i32,
          %input, %filter_chw, %zero_bias)
        {stream.affinity = #hal.device.affinity<@rocket_device>}
        : (i32, i32, i32, i32, i32, i32,
           tensor<1x?x?x?xi8>{%input_height, %input_width, %input_channels},
           tensor<?x?x?xi8>{%input_channels, %weights_height, %weights_width},
           tensor<?xi32>{%output_channels})
        -> tensor<1x?x?x?xi32>{%output_height, %output_width, %output_channels}

    // The epilogue is a plain `linalg.generic`, not a hand-written
    // `flow.dispatch.workgroups`, and the `inline` pass at the end of
    // @__transform_main puts it in the caller. Both halves matter, and only
    // together: a pre-formed dispatch is opaque to dispatch-region formation,
    // so nothing downstream can fuse into it, and a `linalg.generic` left
    // inside a `util.func private` that is never inlined just becomes its own
    // dispatch anyway (the trap ISSUES.md P6 item 2 records for the
    // classifier matmul's truncf). Inlined and generic, IREE fuses this add
    // with the zero-point correction and requantization that follow it in
    // main_graph -- on MobileNetV2 int8 that is 265 dispatch sites down to
    // 145. It needs no `stream.affinity`: rocket-pin-unclaimed-dispatches
    // stamps the CPU default onto every dispatch this spec did not claim,
    // which is what the explicit @cpu_device annotation here used to be for.
    %final_empty_outer = tensor.empty(
        %output_height, %output_width, %output_channels) : tensor<1x?x?x?xi32>
    %final = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } ins(%raw_i32, %init : tensor<1x?x?x?xi32>, tensor<1x?x?x?xi32>)
        outs(%final_empty_outer : tensor<1x?x?x?xi32>) {
      ^bb0(%raw: i32, %initial: i32, %out: i32):
        // No extend, unlike the fp16 epilogue: int8_accumulator mode
        // already hands back a full i32 accumulator.
        %sum = arith.addi %raw, %initial : i32
        linalg.yield %sum : i32
    } -> tensor<1x?x?x?xi32>

    util.return %final : tensor<1x?x?x?xi32>
  }

  // The matmul matcher.
  //
  // f16/f16/f32, like every convolution matcher here: a matmul reaches this
  // point already demoted by RocketDemoteConvInputsPass, and
  // RocketPromoteUnclaimedConvInputsPass gives f32 back to whatever this
  // declines. It used to match f32 and narrow inside @call_rocket_matmul --
  // see that function for why that was quietly expensive.
  //
  // `transform.iree.match.contraction` is the op that can check indexing
  // maps, which is the whole difficulty here: `linalg.matmul` carries a
  // transpose or a broadcast as an attribute rather than as a different op
  // name, and a transposed B is a different memory layout that the
  // height-one convolution lowering cannot pack. Pinning the three maps
  // declines those without having to enumerate them.
  //
  // The bounds are the HAL's, and they are the reason Phase 5 of the plan
  // ran before this matcher was written: K becomes the convolution's input
  // channels and N its output channels, so `MAX_INPUT_CHANNELS` and
  // `MAX_OUTPUT_CHANNELS` bound them -- at 1792 when this was written,
  // exactly MobileNetV2's classifier, measured at that shape rather than
  // inferred from the 14x14 sweep that already reached 1792 at a different
  // geometry. M becomes the convolution *width*, which no constant bounds.
  //
  // Both are **3584** since 2026-09-06, which is what puts a transformer's
  // MLP on the NPU: ViT-B/16 and Qwen3 are both K = N = 3072, and ViT's QKV
  // projection is N = 2304. The sweep behind the raise is in
  // `MAX_INPUT_CHANNELS`' doc comment and includes these shapes at this
  // geometry -- `197x1` with K 3072 N 768, and K 768 N 2304 and 3072 --
  // rather than only the 14x14 conv one.
  //
  // M was bounded at 32 -- where the vendor FC ladder stopped -- until
  // 2026-09-05, and for a while that was holding a hardware fault at bay: a
  // single input row wider than `(K/32 - 1) * M <= 2047` CBUF entries read
  // its last 32 channels from the wrong place (ISSUES.md C10). The planner
  // now bounds the row itself (`Shape::max_tile_input_width`) and splits a
  // wider matmul into column tiles, board-validated at M 90, 128, 197 and
  // 296 (K 768 and 256), 1035 (K 96) and 2000 (K 32). What bounds M now is
  // `CNA_DATA_SIZE0.datain_width`, an 11-bit field: 2047. ViT-B/16's M 197
  // at K 768 plans as three columns. Whether the tall `1 x M` geometry the
  // notes use is the better lowering above 32 is ISSUES.md D1.
  transform.named_sequence @match_rocket_matmul(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.matmul"] : !transform.any_op
    %batch, %m, %n, %k = transform.iree.match.contraction %root,
        lhs_type = f16, rhs_type = f16, output_type = f32,
        indexing_maps = [#rocket_matmul_lhs, #rocket_matmul_rhs, #rocket_matmul_out]
        : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [] : !transform.param<i64>

    %lhs_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    %rhs_value = transform.get_operand %root[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %lhs_value[0], umin = 1, umax = 2047 : !transform.any_value
    transform.iree.match.dim_bounds %lhs_value[1], umin = 1, umax = 3584 : !transform.any_value
    transform.iree.match.dim_bounds %rhs_value[1], umin = 1, umax = 3584 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  // The average-pool matcher.
  //
  // `transform.iree.match.convolution` works on a pooling op -- they
  // implement `LinalgConvolutionOpInterface` too -- but it reports their
  // dimensions differently from a convolution, and guessing wrong is a
  // silent decline. Measured against a real `linalg.pooling_nchw_sum` with
  // `iree-opt` before this was written:
  //
  //   batch    [1, C]   the channel is a pure parallel dim, so it lands here
  //   out_img  [oh, ow]
  //   out_ch   []       a pool has no output-channel dimension at all
  //   in_ch    []
  //   depth    []       and no depth dimension either, unlike a depthwise conv
  //   filter   [kh, kw] from the shape-only window operand
  //
  // Bounds. The kernel must be 2..=8: 8 is `MAX_DIRECT_KERNEL`, which the
  // hardware confirms and a 16x16 window is rejected at, and 2 is the floor
  // because an fp16 average's reciprocal is `fp16(65536/k)` and `k = 1`
  // needs 65536, past fp16's ceiling. Extents and channels stop at the PPU's
  // 13-bit 8192. Stride is 1 because that is what the executable bakes.
  //
  // Wider images are not excluded: `PoolingPlan` splits them into tiles the
  // hardware is measured to run, including the narrow ones an overlapping
  // window needs (see `overlapping_window_width_limits`).

  // The max-pool matchers.
  //
  // Dimension buckets are @match_pooling_nchw_sum_avg's, unchanged: a pool's
  // channel is a pure parallel dimension, so it joins the *batch* group in
  // both layouts rather than becoming an out_ch, and `out_ch`/`in_ch`/`depth`
  // are all empty. That is a property of the reduction's shape, not of the
  // data layout, which is why NHWC and NCHW read identically here.
  //
  // The bounds are the average pool's too, including the kernel floor of 2.
  // For the average that floor is forced -- its reciprocal is fp16(65536/k)
  // and k=1 needs 65536, past fp16's 65504. A max pool computes no
  // reciprocal, so 1 would be programmable; it is excluded anyway because a
  // 1x1 stride-1 max pool is an identity, and claiming one would spend a
  // whole NPU dispatch, a pack and a compaction to copy a tensor.
  //
  // Stride 1 and 2 only: 2 is measured against the oracle in
  // `pooling_oracle_hw.rs`, and nothing above it is.


  // ONNX `Add` on a rank-3 token tensor, as it actually reaches this loop.
  //
  // The op form here is not obvious and was established by running a
  // torch-onnx probe through this very spec. The ONNX path produces a
  // `linalg.generic` with an `arith.addf` body -- but `@__transform_main`
  // runs `linalg-specialize-generic-ops` twice, with canonicalization
  // between, and by the time this loop runs the op has been specialized into
  // the named `linalg.add`. A matcher written against the generic form
  // matches nothing and says nothing about why. (It is also not
  // `linalg.elementwise`, which this build's linalg does define.)
  //
  // Matching the named op means no `cast_compatible_dag_from_root` is
  // needed: the operation name already says the body is exactly this
  // arithmetic, so there is no subgraph to describe and none of that op's
  // silent-decline traps to fall into.
  transform.named_sequence @match_elementwise_add_f32(
      %root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.add"] : !transform.any_op

    %lhs_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    %rhs_value = transform.get_operand %root[1] : (!transform.any_op) -> !transform.any_value

    // Rank 3, `1 x tokens x channels`, mapped to the hardware cube as
    // width = tokens, height = 1, channels = channels: the trailing
    // dimension is innermost in memory and is what NC1HWC2 packs into
    // feature atoms. Requiring the leading extent to be 1 is what makes
    // that mapping sound -- a real batch would need a fourth extent the
    // cube does not have.
    transform.iree.match.dim_bounds %lhs_value[0], umin = 1, umax = 1 : !transform.any_value
    // Tokens. Board-measured to 197 (ViT's sequence length) by
    // `ew_binary_hw`; nothing wider has been run, so nothing wider is
    // admitted.
    transform.iree.match.dim_bounds %lhs_value[1], umin = 1, umax = 197 : !transform.any_value
    // Channels. Board-measured to 3072, ViT-base's MLP hidden width -- 384
    // fp16 surfaces. That ladder's previous ceiling was 64 channels, which
    // would have been a careless bound to set this from.
    transform.iree.match.dim_bounds %lhs_value[2], umin = 1, umax = 3072 : !transform.any_value

    // The second operand must be the same cube, not a broadcast: this kernel
    // has one geometry for both inputs and the result.
    transform.iree.match.dim_bounds %rhs_value[0], umin = 1, umax = 1 : !transform.any_value
    transform.iree.match.dim_bounds %rhs_value[1], umin = 1, umax = 197 : !transform.any_value
    transform.iree.match.dim_bounds %rhs_value[2], umin = 1, umax = 3072 : !transform.any_value

    transform.yield %root : !transform.any_op
  }

  // ONNX `Sub` on a rank-3 token tensor, as it actually reaches this loop.
  //
  // The op form here is not obvious and was established by running a
  // torch-onnx probe through this very spec. The ONNX path produces a
  // `linalg.generic` with an `arith.subf` body -- but `@__transform_main`
  // runs `linalg-specialize-generic-ops` twice, with canonicalization
  // between, and by the time this loop runs the op has been specialized into
  // the named `linalg.sub`. A matcher written against the generic form
  // matches nothing and says nothing about why. (It is also not
  // `linalg.elementwise`, which this build's linalg does define.)
  //
  // Matching the named op means no `cast_compatible_dag_from_root` is
  // needed: the operation name already says the body is exactly this
  // arithmetic, so there is no subgraph to describe and none of that op's
  // silent-decline traps to fall into.
  transform.named_sequence @match_elementwise_sub_f32(
      %root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.sub"] : !transform.any_op

    %lhs_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    %rhs_value = transform.get_operand %root[1] : (!transform.any_op) -> !transform.any_value

    // Rank 3, `1 x tokens x channels`, mapped to the hardware cube as
    // width = tokens, height = 1, channels = channels: the trailing
    // dimension is innermost in memory and is what NC1HWC2 packs into
    // feature atoms. Requiring the leading extent to be 1 is what makes
    // that mapping sound -- a real batch would need a fourth extent the
    // cube does not have.
    transform.iree.match.dim_bounds %lhs_value[0], umin = 1, umax = 1 : !transform.any_value
    // Tokens. Board-measured to 197 (ViT's sequence length) by
    // `ew_binary_hw`; nothing wider has been run, so nothing wider is
    // admitted.
    transform.iree.match.dim_bounds %lhs_value[1], umin = 1, umax = 197 : !transform.any_value
    // Channels. Board-measured to 3072, ViT-base's MLP hidden width -- 384
    // fp16 surfaces. That ladder's previous ceiling was 64 channels, which
    // would have been a careless bound to set this from.
    transform.iree.match.dim_bounds %lhs_value[2], umin = 1, umax = 3072 : !transform.any_value

    // The second operand must be the same cube, not a broadcast: this kernel
    // has one geometry for both inputs and the result.
    transform.iree.match.dim_bounds %rhs_value[0], umin = 1, umax = 1 : !transform.any_value
    transform.iree.match.dim_bounds %rhs_value[1], umin = 1, umax = 197 : !transform.any_value
    transform.iree.match.dim_bounds %rhs_value[2], umin = 1, umax = 3072 : !transform.any_value

    transform.yield %root : !transform.any_op
  }

  // ONNX `Mul` on a rank-3 token tensor, as it actually reaches this loop.
  //
  // The op form here is not obvious and was established by running a
  // torch-onnx probe through this very spec. The ONNX path produces a
  // `linalg.generic` with an `arith.mulf` body -- but `@__transform_main`
  // runs `linalg-specialize-generic-ops` twice, with canonicalization
  // between, and by the time this loop runs the op has been specialized into
  // the named `linalg.mul`. A matcher written against the generic form
  // matches nothing and says nothing about why. (It is also not
  // `linalg.elementwise`, which this build's linalg does define.)
  //
  // Matching the named op means no `cast_compatible_dag_from_root` is
  // needed: the operation name already says the body is exactly this
  // arithmetic, so there is no subgraph to describe and none of that op's
  // silent-decline traps to fall into.
  transform.named_sequence @match_elementwise_mul_f32(
      %root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.mul"] : !transform.any_op

    %lhs_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    %rhs_value = transform.get_operand %root[1] : (!transform.any_op) -> !transform.any_value

    // Rank 3, `1 x tokens x channels`, mapped to the hardware cube as
    // width = tokens, height = 1, channels = channels: the trailing
    // dimension is innermost in memory and is what NC1HWC2 packs into
    // feature atoms. Requiring the leading extent to be 1 is what makes
    // that mapping sound -- a real batch would need a fourth extent the
    // cube does not have.
    transform.iree.match.dim_bounds %lhs_value[0], umin = 1, umax = 1 : !transform.any_value
    // Tokens. Board-measured to 197 (ViT's sequence length) by
    // `ew_binary_hw`; nothing wider has been run, so nothing wider is
    // admitted.
    transform.iree.match.dim_bounds %lhs_value[1], umin = 1, umax = 197 : !transform.any_value
    // Channels. Board-measured to 3072, ViT-base's MLP hidden width -- 384
    // fp16 surfaces. That ladder's previous ceiling was 64 channels, which
    // would have been a careless bound to set this from.
    transform.iree.match.dim_bounds %lhs_value[2], umin = 1, umax = 3072 : !transform.any_value

    // The second operand must be the same cube, not a broadcast: this kernel
    // has one geometry for both inputs and the result.
    transform.iree.match.dim_bounds %rhs_value[0], umin = 1, umax = 1 : !transform.any_value
    transform.iree.match.dim_bounds %rhs_value[1], umin = 1, umax = 197 : !transform.any_value
    transform.iree.match.dim_bounds %rhs_value[2], umin = 1, umax = 3072 : !transform.any_value

    transform.yield %root : !transform.any_op
  }

  transform.named_sequence @cast_and_call_elementwise_add(%root: !transform.any_op {transform.readonly}) {
    %ins = transform.get_operand %root[all] : (!transform.any_op) -> !transform.any_value
    %out = transform.get_result %root[all] : (!transform.any_op) -> !transform.any_value
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %topology_attr = transform.param.constant #hal.device.topology<links = [
        (@rocket_device -> @cpu_device = {transparent_access = true, unified_memory = true}),
        (@cpu_device -> @rocket_device = {transparent_access = true, unified_memory = true})
      ]> -> !transform.any_param
    transform.annotate %module "stream.topology" = %topology_attr : !transform.any_op, !transform.any_param
    %executable = transform.util.import_symbol @rocket_elementwise_add_executable into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_rocket_elementwise_add into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
          transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @cast_and_call_elementwise_sub(%root: !transform.any_op {transform.readonly}) {
    %ins = transform.get_operand %root[all] : (!transform.any_op) -> !transform.any_value
    %out = transform.get_result %root[all] : (!transform.any_op) -> !transform.any_value
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %topology_attr = transform.param.constant #hal.device.topology<links = [
        (@rocket_device -> @cpu_device = {transparent_access = true, unified_memory = true}),
        (@cpu_device -> @rocket_device = {transparent_access = true, unified_memory = true})
      ]> -> !transform.any_param
    transform.annotate %module "stream.topology" = %topology_attr : !transform.any_op, !transform.any_param
    %executable = transform.util.import_symbol @rocket_elementwise_sub_executable into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_rocket_elementwise_sub into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
          transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @cast_and_call_elementwise_mul(%root: !transform.any_op {transform.readonly}) {
    %ins = transform.get_operand %root[all] : (!transform.any_op) -> !transform.any_value
    %out = transform.get_result %root[all] : (!transform.any_op) -> !transform.any_value
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %topology_attr = transform.param.constant #hal.device.topology<links = [
        (@rocket_device -> @cpu_device = {transparent_access = true, unified_memory = true}),
        (@cpu_device -> @rocket_device = {transparent_access = true, unified_memory = true})
      ]> -> !transform.any_param
    transform.annotate %module "stream.topology" = %topology_attr : !transform.any_op, !transform.any_param
    %executable = transform.util.import_symbol @rocket_elementwise_mul_executable into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_rocket_elementwise_mul into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
          transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @match_pooling_nhwc_max(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.pooling_nhwc_max"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = f32, rhs_type = f32, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    %window_value = transform.get_operand %root[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[1], umin = 1, umax = 8192 : !transform.any_value
    transform.iree.match.dim_bounds %input_value[2], umin = 1, umax = 8192 : !transform.any_value
    transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 8192 : !transform.any_value
    transform.iree.match.dim_bounds %window_value[0], umin = 2, umax = 8 : !transform.any_value
    transform.iree.match.dim_bounds %window_value[1], umin = 2, umax = 8 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  transform.named_sequence @match_pooling_nhwc_max_s2(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.pooling_nhwc_max"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = f32, rhs_type = f32, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [2, 2] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    %window_value = transform.get_operand %root[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[1], umin = 1, umax = 8192 : !transform.any_value
    transform.iree.match.dim_bounds %input_value[2], umin = 1, umax = 8192 : !transform.any_value
    transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 8192 : !transform.any_value
    transform.iree.match.dim_bounds %window_value[0], umin = 2, umax = 8 : !transform.any_value
    transform.iree.match.dim_bounds %window_value[1], umin = 2, umax = 8 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  transform.named_sequence @match_pooling_nchw_max(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.pooling_nchw_max"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = f32, rhs_type = f32, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    %window_value = transform.get_operand %root[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[1], umin = 1, umax = 8192 : !transform.any_value
    transform.iree.match.dim_bounds %input_value[2], umin = 1, umax = 8192 : !transform.any_value
    transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 8192 : !transform.any_value
    transform.iree.match.dim_bounds %window_value[0], umin = 2, umax = 8 : !transform.any_value
    transform.iree.match.dim_bounds %window_value[1], umin = 2, umax = 8 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  transform.named_sequence @match_pooling_nchw_max_s2(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.pooling_nchw_max"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = f32, rhs_type = f32, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [2, 2] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    %window_value = transform.get_operand %root[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[1], umin = 1, umax = 8192 : !transform.any_value
    transform.iree.match.dim_bounds %input_value[2], umin = 1, umax = 8192 : !transform.any_value
    transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 8192 : !transform.any_value
    transform.iree.match.dim_bounds %window_value[0], umin = 2, umax = 8 : !transform.any_value
    transform.iree.match.dim_bounds %window_value[1], umin = 2, umax = 8 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  transform.named_sequence @match_pooling_nchw_sum_avg(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.pooling_nchw_sum"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = f32, rhs_type = f32, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    %window_value = transform.get_operand %root[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[1], umin = 1, umax = 8192 : !transform.any_value
    transform.iree.match.dim_bounds %input_value[2], umin = 1, umax = 8192 : !transform.any_value
    transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 8192 : !transform.any_value
    transform.iree.match.dim_bounds %window_value[0], umin = 2, umax = 8 : !transform.any_value
    transform.iree.match.dim_bounds %window_value[1], umin = 2, umax = 8 : !transform.any_value
    transform.yield %root : !transform.any_op
  }



  // The NHWC average matcher. Same bounds and same dimension buckets as
  // @match_pooling_nchw_sum_avg -- a pool's channel is a pure parallel
  // dimension in either layout, so the buckets do not move -- differing only
  // in the op name and in which shim claims it.
  //
  // Stride 1 only, matching its NCHW counterpart. Stride 2 *is* measured for
  // the average (`avg 2x2s2` in pooling_oracle_hw.rs) and max and min both
  // carry it, so this is the one remaining asymmetry in the pooling matchers;
  // it needs a second executable and is deliberately left for its own change.
  transform.named_sequence @match_pooling_nhwc_sum_avg(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.pooling_nhwc_sum"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = f32, rhs_type = f32, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    %window_value = transform.get_operand %root[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[1], umin = 1, umax = 8192 : !transform.any_value
    transform.iree.match.dim_bounds %input_value[2], umin = 1, umax = 8192 : !transform.any_value
    transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 8192 : !transform.any_value
    transform.iree.match.dim_bounds %window_value[0], umin = 2, umax = 8 : !transform.any_value
    transform.iree.match.dim_bounds %window_value[1], umin = 2, umax = 8 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  // The min-pool matchers. Bounds are the max ones exactly -- the reduction
  // does not change any geometric limit -- and there is no NCHW pair because
  // linalg has no pooling_nchw_min to match.

  transform.named_sequence @match_pooling_nhwc_min(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.pooling_nhwc_min"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = f32, rhs_type = f32, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    %window_value = transform.get_operand %root[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[1], umin = 1, umax = 8192 : !transform.any_value
    transform.iree.match.dim_bounds %input_value[2], umin = 1, umax = 8192 : !transform.any_value
    transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 8192 : !transform.any_value
    transform.iree.match.dim_bounds %window_value[0], umin = 2, umax = 8 : !transform.any_value
    transform.iree.match.dim_bounds %window_value[1], umin = 2, umax = 8 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  transform.named_sequence @match_pooling_nhwc_min_s2(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.pooling_nhwc_min"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = f32, rhs_type = f32, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [2, 2] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    %window_value = transform.get_operand %root[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[1], umin = 1, umax = 8192 : !transform.any_value
    transform.iree.match.dim_bounds %input_value[2], umin = 1, umax = 8192 : !transform.any_value
    transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 8192 : !transform.any_value
    transform.iree.match.dim_bounds %window_value[0], umin = 2, umax = 8 : !transform.any_value
    transform.iree.match.dim_bounds %window_value[1], umin = 2, umax = 8 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  transform.named_sequence @match_dynamic_conv2d(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.conv_2d_nhwc_hwcf"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = f16, rhs_type = f16, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    // Cin remains capped at 512 and Cout is capped well below
    // MAX_OUTPUT_CHANNELS (528). This is hardware-verified, not a guess: an
    // is hardware-verified, not a guess: an isolated correctness probe
    // (iree-rocket-hal/tests/conv_cbuf_split_sweep_hw.rs and
    // conv_features19_isolated_hw.rs), run in isolation on real Planck
    // hardware with a fill-1.0/exact-expected-value check (not just
    // "did it time out"), found:
    //
    //   Cout  64/128/256 (Cin 3, 64, 128, 256; banks 11/1, 9/3, 7/5, 3/9,
    //         1/11 all covered)          -> correct, 5/5 every rep
    //   Cout  512 (Cin 256 -- features.19; Cin 512 -- features.21; both
    //         30x30, banks 11/1)         -> all-zero output, 0/5 every
    //                                        rep, deterministic
    //
    // IMPORTANT CAVEAT, found later by a real-compiler-path harness
    // (rocket_conv_harness.py in iree-rocket-design-spike) plus a follow-up
    // extent_sweep_at_fixed_channels sweep in
    // conv_cbuf_split_sweep_hw.rs: Cout is not actually the discriminator.
    // Cin=256/Cout=256/3x3 -- comfortably inside this bound -- is ALSO
    // deterministically all-zero (0/5, every output element wrong) across
    // every spatial extent from 26x26 to 48x48, because ConvPlan picks the
    // same 11/1 split there that it picks for the broken Cout=512 shapes
    // above; extents 20-24 (banks 7/5, 9/3) and 50-58 (banks 1/11) at the
    // SAME channel counts pass 5/5, and the pass/fail boundary lines up
    // exactly with ConvPlan's split-flip points, with zero fuzziness. The
    // real discriminator is an 11/1-style split combined with a large
    // coefficient footprint: features.0 (Cin=3/Cout=64, footprint
    // 3*3*3*64 = 1728 elements) also gets an 11/1 split and is fine, while
    // Cin=256/Cout=256 (footprint 3*3*256*256 = 589824 elements) at that
    // same split is broken everywhere it occurs. This bound is still safe
    // for VGG specifically -- none of its real Cout<=256 layers land on an
    // 11/1 split at a large-footprint channel count -- but that is a
    // property of VGG's specific shapes, not a guarantee this Cout<=256
    // rule provides in general. A future model (or a wider matcher) could
    // reintroduce this exact bug at Cout<=256 with the wrong spatial
    // extent. See DESIGN_NOTES.md for the full characterization.
    //
    // This is the same class of bug DESIGN_NOTES.md documents for 9x9/11x11
    // -- ConvPlan's demand-based CBUF formula picks a split based on raw
    // byte demand, but the real vendor coefficient-streaming schedule for
    // high-pressure shapes isn't decoded, so an unvalidated split doesn't
    // fail loudly, it silently completes with all-zero output. A live
    // rocket-npu-trace initially suggested this was a timing/idle-gap
    // issue (every failure followed an anomalously long idle gap on that
    // core) -- ruled out by this same isolated probe: the shape fails
    // identically as a fresh first job, after a sustained warmup burst,
    // and after a deliberate idle gap. It is the shape, not the timing.
    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    %filter_value = transform.get_operand %root[1] : (!transform.any_op) -> !transform.any_value
    // The HAL's `MAX_INPUT_CHANNELS`, raised 512 -> 1344 (2026-09-03) on
    // hardware: fp16 k=1 is exact at 14x14 Cout 64 for Cin 256..1792 across
    // one to five tiles, and the fp16 vendor corpus above the old ceiling
    // agrees (conv_vendor_fixture_wide.rs). The 2026-08-28 attempt at 960 was
    // reverted for a CBUF-split divergence that the 2026-09-02 group-division
    // fix removed; see MAX_INPUT_CHANNELS' doc comment.
    //
    // Raised again 1344 -> 3584 on 2026-09-06, with the constant. k=1 is the
    // kernel the sweep covers and the only one this bound governs: at k=3
    // the coefficient working set binds far below either number, and
    // @match_dynamic_conv2d_3x3 keeps its own 1152. Board evidence, quiet
    // board, `Selectors` for addressing and `Counting` for lane coverage at
    // every point, plus the `onehot` read map at Cout == Cin: Cin 1792
    // through 3584 in 256-channel steps and on to 8192, ragged 1793..4095,
    // 56x56 multi-tile to 3584. Full list in MAX_INPUT_CHANNELS' doc comment.
    transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 3584 : !transform.any_value
    // MobileNetV2's four 14x14, Cin=88, Cout=528 pointwise convolutions
    // pass the three hardware-oracle patterns with a 2/10 CBUF split. Keep
    // this narrow expansion local to the stride-1 1x1 matcher; the 3x3 and
    // strided matchers retain their separately characterized 512 limit.
    // The HAL's `MAX_OUTPUT_CHANNELS`, raised 528 -> 1792, then 1792 -> 3584
    // (2026-09-06). Measured exact at 7x7 Cin 448 for Cout 528, 640, 768,
    // 1024, 1344, 1792, 2048, and then 2304, 2560, 3072, 3584 and 4096, with
    // the CBUF split flat at 2d/10w over the whole range -- the high-channel
    // divergence is indexed by `Cin`, not `Cout`. Ragged Cout 1793, 2049,
    // 2313, 3073, 3585 and 4095 are exact too. The old 528 was a narrow
    // expansion for MobileNetV2's Cin=88/Cout=528 pointwise convolutions.
    transform.iree.match.dim_bounds %filter_value[3], umin = 1, umax = 3584 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  // Same fallback for 3x3, which the hardware handles natively through the
  // identical demand-based CBUF partition as 1x1. Spelled as its own
  // matcher (rather than widening the filter check to 1..=3) so 2x2 and
  // non-square combinations, which route through different ConvPlan
  // partitions, are never silently claimed.
  transform.named_sequence @match_dynamic_conv2d_3x3(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.conv_2d_nhwc_hwcf"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = f16, rhs_type = f16, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [3, 3] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    // Cout is capped well below MAX_OUTPUT_CHANNELS (512); Cin is not. This
    // is hardware-verified, not a guess: an isolated correctness probe
    // (iree-rocket-hal/tests/conv_cbuf_split_sweep_hw.rs and
    // conv_features19_isolated_hw.rs), run in isolation on real Planck
    // hardware with a fill-1.0/exact-expected-value check (not just
    // "did it time out"), found:
    //
    //   Cout  64/128/256 (Cin 3, 64, 128, 256; banks 11/1, 9/3, 7/5, 3/9,
    //         1/11 all covered)          -> correct, 5/5 every rep
    //   Cout  512 (Cin 256 -- features.19; Cin 512 -- features.21; both
    //         30x30, banks 11/1)         -> all-zero output, 0/5 every
    //                                        rep, deterministic
    //
    // IMPORTANT CAVEAT, found later by a real-compiler-path harness
    // (rocket_conv_harness.py in iree-rocket-design-spike) plus a follow-up
    // extent_sweep_at_fixed_channels sweep in
    // conv_cbuf_split_sweep_hw.rs: Cout is not actually the discriminator.
    // Cin=256/Cout=256/3x3 -- comfortably inside this bound -- is ALSO
    // deterministically all-zero (0/5, every output element wrong) across
    // every spatial extent from 26x26 to 48x48, because ConvPlan picks the
    // same 11/1 split there that it picks for the broken Cout=512 shapes
    // above; extents 20-24 (banks 7/5, 9/3) and 50-58 (banks 1/11) at the
    // SAME channel counts pass 5/5, and the pass/fail boundary lines up
    // exactly with ConvPlan's split-flip points, with zero fuzziness. The
    // real discriminator is an 11/1-style split combined with a large
    // coefficient footprint: features.0 (Cin=3/Cout=64, footprint
    // 3*3*3*64 = 1728 elements) also gets an 11/1 split and is fine, while
    // Cin=256/Cout=256 (footprint 3*3*256*256 = 589824 elements) at that
    // same split is broken everywhere it occurs. This bound is still safe
    // for VGG specifically -- none of its real Cout<=256 layers land on an
    // 11/1 split at a large-footprint channel count -- but that is a
    // property of VGG's specific shapes, not a guarantee this Cout<=256
    // rule provides in general. A future model (or a wider matcher) could
    // reintroduce this exact bug at Cout<=256 with the wrong spatial
    // extent. See DESIGN_NOTES.md for the full characterization.
    //
    // This is the same class of bug DESIGN_NOTES.md documents for 9x9/11x11
    // -- ConvPlan's demand-based CBUF formula picks a split based on raw
    // byte demand, but the real vendor coefficient-streaming schedule for
    // high-pressure shapes isn't decoded, so an unvalidated split doesn't
    // fail loudly, it silently completes with all-zero output. A live
    // rocket-npu-trace initially suggested this was a timing/idle-gap
    // issue (every failure followed an anomalously long idle gap on that
    // core) -- ruled out by this same isolated probe: the shape fails
    // identically as a fresh first job, after a sustained warmup burst,
    // and after a deliberate idle gap. It is the shape, not the timing.
    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    %filter_value = transform.get_operand %root[1] : (!transform.any_op) -> !transform.any_value
    // 1152, not `MAX_INPUT_CHANNELS` (3584 since 2026-09-06): at a 3x3
    // kernel the coefficient working set binds first and `ConvPlan` refuses
    // Cin >= 1216 outright, which would reach the driver and panic rather
    // than fall back. fp16 k=3
    // is exact at 28x28 Cout 64 for Cin 512..1152, including the 1/11 split
    // at 1152. Same reasoning as `@match_dynamic_conv2d_3x3_int8`.
    transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 1152 : !transform.any_value
    transform.iree.match.dim_bounds %filter_value[3], umin = 1, umax = 1792 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  // Stride-2 counterpart of @match_dynamic_conv2d -- same 1x1 kernel and Cout/Cin
  // <= 512 bound (see that matcher's own doc comment for the
  // full CBUF-split correctness caveat, which applies identically here: it
  // is a property of channel count and coefficient footprint, not stride).
  // What's new is `%strides` = [2, 2] instead of [1, 1] -- hardware-
  // confirmed dense fp16 at stride 2 by conv_wide_shape_hw.rs (see
  // DESIGN_NOTES.md "Stride and large-width sweeps").
  transform.named_sequence @match_dynamic_conv2d_s2(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.conv_2d_nhwc_hwcf"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = f16, rhs_type = f16, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [2, 2] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    %filter_value = transform.get_operand %root[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 512 : !transform.any_value
    transform.iree.match.dim_bounds %filter_value[3], umin = 1, umax = 512 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  // Stride-2 counterpart of @match_dynamic_conv2d_3x3 -- same 3x3 kernel and Cout/Cin
  // <= 512 bound (see that matcher's own doc comment for the
  // full CBUF-split correctness caveat, which applies identically here: it
  // is a property of channel count and coefficient footprint, not stride).
  // What's new is `%strides` = [2, 2] instead of [1, 1] -- hardware-
  // confirmed dense fp16 at stride 2 by conv_wide_shape_hw.rs (see
  // DESIGN_NOTES.md "Stride and large-width sweeps").
  transform.named_sequence @match_dynamic_conv2d_3x3_s2(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.conv_2d_nhwc_hwcf"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = f16, rhs_type = f16, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [3, 3] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [2, 2] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    %filter_value = transform.get_operand %root[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 512 : !transform.any_value
    transform.iree.match.dim_bounds %filter_value[3], umin = 1, umax = 512 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  // Stride-3 counterpart of @match_dynamic_conv2d -- same 1x1 kernel and Cout/Cin
  // <= 512 bound (see that matcher's own doc comment for the
  // full CBUF-split correctness caveat, which applies identically here: it
  // is a property of channel count and coefficient footprint, not stride).
  // What's new is `%strides` = [3, 3] instead of [1, 1] -- hardware-
  // confirmed dense fp16 at stride 3 by conv_wide_shape_hw.rs (see
  // DESIGN_NOTES.md "Stride and large-width sweeps").
  transform.named_sequence @match_dynamic_conv2d_s3(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.conv_2d_nhwc_hwcf"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = f16, rhs_type = f16, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [3, 3] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    %filter_value = transform.get_operand %root[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 512 : !transform.any_value
    transform.iree.match.dim_bounds %filter_value[3], umin = 1, umax = 512 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  // Stride-3 counterpart of @match_dynamic_conv2d_3x3 -- same 3x3 kernel and Cout/Cin
  // <= 512 bound (see that matcher's own doc comment for the
  // full CBUF-split correctness caveat, which applies identically here: it
  // is a property of channel count and coefficient footprint, not stride).
  // What's new is `%strides` = [3, 3] instead of [1, 1] -- hardware-
  // confirmed dense fp16 at stride 3 by conv_wide_shape_hw.rs (see
  // DESIGN_NOTES.md "Stride and large-width sweeps").
  transform.named_sequence @match_dynamic_conv2d_3x3_s3(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.conv_2d_nhwc_hwcf"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = f16, rhs_type = f16, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [3, 3] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [3, 3] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    %filter_value = transform.get_operand %root[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 512 : !transform.any_value
    transform.iree.match.dim_bounds %filter_value[3], umin = 1, umax = 512 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  // Stride-4 counterpart of @match_dynamic_conv2d -- same 1x1 kernel and Cout/Cin
  // <= 512 bound (see that matcher's own doc comment for the
  // full CBUF-split correctness caveat, which applies identically here: it
  // is a property of channel count and coefficient footprint, not stride).
  // What's new is `%strides` = [4, 4] instead of [1, 1] -- hardware-
  // confirmed dense fp16 at stride 4 by conv_wide_shape_hw.rs (see
  // DESIGN_NOTES.md "Stride and large-width sweeps").
  transform.named_sequence @match_dynamic_conv2d_s4(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.conv_2d_nhwc_hwcf"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = f16, rhs_type = f16, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [4, 4] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    %filter_value = transform.get_operand %root[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 512 : !transform.any_value
    transform.iree.match.dim_bounds %filter_value[3], umin = 1, umax = 512 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  // Stride-4 counterpart of @match_dynamic_conv2d_3x3 -- same 3x3 kernel and Cout/Cin
  // <= 512 bound (see that matcher's own doc comment for the
  // full CBUF-split correctness caveat, which applies identically here: it
  // is a property of channel count and coefficient footprint, not stride).
  // What's new is `%strides` = [4, 4] instead of [1, 1] -- hardware-
  // confirmed dense fp16 at stride 4 by conv_wide_shape_hw.rs (see
  // DESIGN_NOTES.md "Stride and large-width sweeps").
  transform.named_sequence @match_dynamic_conv2d_3x3_s4(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.conv_2d_nhwc_hwcf"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = f16, rhs_type = f16, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [3, 3] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [4, 4] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    %filter_value = transform.get_operand %root[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 512 : !transform.any_value
    transform.iree.match.dim_bounds %filter_value[3], umin = 1, umax = 512 : !transform.any_value
    transform.yield %root : !transform.any_op
  }


  // Depthwise counterpart of @match_dynamic_conv2d. linalg.depthwise_conv_2d_nhwc_hwc
  // (not the _hwcm variant) is the only depthwise op family Rocket claims:
  // ConvPlan::with_depthwise hard-asserts a channel multiplier of one, so a
  // real channel-multiplier dimension (_hwcm) has never been captured or
  // validated and must stay on CPU. transform.iree.match.convolution's
  // dimension inference (LinalgInterfaces.cpp inferConvolutionDimsImpl)
  // puts the shared input/output channel dim in depth_dims for this op
  // family, not output_channel_dims/input_channel_dims (both empty here,
  // unlike the dense matcher above) -- confirmed against
  // mlir/unittests/Dialect/Linalg/InferConvolutionDimsTest.cpp, which
  // exercises exactly this op and asserts depth is non-empty. Cout is
  // therefore never an independent quantity to bound: it is always Cin,
  // read off %init the same way call_rocket_dynamic_depthwise_conv2d's
  // %output_channels already is.
  //
  // Runtime support (packing, register fields, CBUF allocation) is
  // hardware-validated -- see DESIGN_NOTES.md "Depthwise: Mesa's channel
  // rule is wrong" and conv_phase1_validation_hw.rs (8/8 passing, including
  // the int8 Cin=12 tap-major layout check) -- but that validation never
  // exceeded one 32-channel coefficient group
  // (WEIGHT_INPUT_GROUP_CHANNELS), so this matcher originally shipped
  // capped at 128, not the dense matcher's 512, to avoid the exact mistake
  // DESIGN_NOTES.md documents for dense conv (a coarse capture ladder
  // hiding a formula bug between measured points). A boundary probe
  // (rocket_conv_harness.py, single fixed-shape depthwise dispatches)
  // through the real compiled dispatch path -- not the isolated hardware
  // captures above -- found exactly that: `pack_depthwise_to_rocket_weights`
  // used a single global tap-major stride instead of grouping channels by
  // 32, invisible below one group. Fixed in `tensor_layout.rs`
  // (`pack_depthwise_to_rocket_weights`'s doc comment has the full
  // derivation) and confirmed clean at every ladder point from 128 through
  // 512, both kernel sizes, so the cap now matches the dense matcher's.
  transform.named_sequence @match_dynamic_depthwise_conv2d(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.depthwise_conv_2d_nhwc_hwc"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = f16, rhs_type = f16, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    // Only one channel count to bound: depthwise Cout is always Cin.
    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 512 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  // Same fallback for 3x3, mirroring @match_dynamic_conv2d_3x3's rationale:
  // ConvPlan routes depthwise through the identical demand-based CBUF
  // partition for kernel extents 1 and 3 (conv.rs), and MobileNet-style
  // depthwise-separable models overwhelmingly use 3x3 for the depthwise
  // stage, so this is the practically load-bearing case.
  transform.named_sequence @match_dynamic_depthwise_conv2d_3x3(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.depthwise_conv_2d_nhwc_hwc"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = f16, rhs_type = f16, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [3, 3] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 512 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  transform.named_sequence @cast_and_call_dynamic_depthwise_conv2d(%root: !transform.any_op {transform.readonly}) {
    %ins = transform.get_operand %root[all] : (!transform.any_op) -> !transform.any_value
    %out = transform.get_result %root[all] : (!transform.any_op) -> !transform.any_value
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %topology_attr = transform.param.constant #hal.device.topology<links = [
        (@rocket_device -> @cpu_device = {transparent_access = true, unified_memory = true}),
        (@cpu_device -> @rocket_device = {transparent_access = true, unified_memory = true})
      ]> -> !transform.any_param
    transform.annotate %module "stream.topology" = %topology_attr : !transform.any_op, !transform.any_param
    %executable = transform.util.import_symbol @rocket_dynamic_depthwise_executable into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_rocket_dynamic_depthwise_conv2d into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
          transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  // NCHW counterpart of @match_dynamic_depthwise_conv2d -- see
  // call_rocket_dynamic_depthwise_conv2d_nchw's doc comment for why real
  // ONNX-imported models need this instead of (not in addition to reaching)
  // the NHWC matcher above. Same dims_equal shape as the NHWC matcher --
  // depthwise's dimension inference doesn't depend on operand layout, only
  // on which axes are batch/channel/spatial -- except the channel dim to
  // bound is input dim 1 here (NCHW), not dim 3 (NHWC).
  transform.named_sequence @match_dynamic_depthwise_conv2d_nchw(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.depthwise_conv_2d_nchw_chw"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = f16, rhs_type = f16, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[1], umin = 1, umax = 512 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  // Same fallback for 3x3, mirroring @match_dynamic_depthwise_conv2d_3x3's
  // rationale -- this is the practically load-bearing case, since a real
  // ONNX-imported depthwise-separable model's spatial stage is
  // overwhelmingly 3x3, arriving in exactly this NCHW form.
  //
  // Cin bound kept at 512 despite `MAX_INPUT_CHANNELS`/`MAX_OUTPUT_CHANNELS`
  // in conv.rs being raised to 960: MobileNetV2's Cin=576 (14x14) and
  // Cin=960 (7x7) depthwise stages are hardware-validated correct at that
  // width
  // (iree-rocket-hal/tests/conv_mobilenetv2_depthwise_wide_hw.rs), but
  // routing them to Rocket measured as a net *regression* end to end
  // (161-163ms -> 165-170ms on real hardware) -- the layers are spatially
  // too small (14x14, 7x7) to amortize the per-dispatch NC1HWC2 pack/unpack
  // tax every Rocket dispatch pays (see
  // rocket-hal-driver/src/command_buffer.rs; no propagation across chained
  // dispatches exists). So this stays CPU-routed on purpose. Revisit once
  // that repack-propagation work exists, or raise this bound again for a
  // model whose >512-channel depthwise layers are large enough to be worth
  // it.
  transform.named_sequence @match_dynamic_depthwise_conv2d_nchw_3x3(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.depthwise_conv_2d_nchw_chw"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = f16, rhs_type = f16, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [3, 3] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[1], umin = 1, umax = 512 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  // Stride-2 counterpart of @match_dynamic_depthwise_conv2d_nchw -- same 1x1 kernel and Cin <= 512
  // bound (Cout is always Cin here, see that matcher's own doc comment).
  // `%strides` = [2, 2] instead of [1, 1] -- hardware-confirmed depthwise
  // fp16 at stride 2 by conv_depthwise_stride_hw.rs (see DESIGN_NOTES.md
  // "Depthwise stride hardware confirmation").
  transform.named_sequence @match_dynamic_depthwise_conv2d_nchw_s2(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.depthwise_conv_2d_nchw_chw"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = f16, rhs_type = f16, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [2, 2] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[1], umin = 1, umax = 512 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  // Stride-2 counterpart of @match_dynamic_depthwise_conv2d_nchw_3x3 -- same 3x3 kernel and Cin <= 512
  // bound (Cout is always Cin here, see that matcher's own doc comment).
  // `%strides` = [2, 2] instead of [1, 1] -- hardware-confirmed depthwise
  // fp16 at stride 2 by conv_depthwise_stride_hw.rs (see DESIGN_NOTES.md
  // "Depthwise stride hardware confirmation").
  //
  // Deliberately NOT raised to 960 alongside @match_dynamic_depthwise_conv2d_nchw_3x3
  // above. MobileNetV2's Cin=576, 14x14->7x7 stride-2 block is the matcher
  // this would newly claim, and `ConvPlan::new` hard-panics for it today
  // ("convolution needs horizontal tiling, which is only capture-backed at
  // stride 1", conv.rs:~1909) -- at 16-wide input, 576 channels' coefficient
  // demand forces horizontal (column) tiling, which has never been
  // implemented for stride > 1. Confirmed with the pure-planning
  // `dump_conv_plan` example (no hardware involved): 96..512 channels at
  // this width plan cleanly, 576 panics immediately. Raising this bound
  // would make the compiler route the shape to Rocket and then hard-crash
  // the HAL driver at first inference, which is strictly worse than today's
  // CPU fallback. Needs horizontal-tiling-at-stride>1 support in ConvPlan
  // before this bound can move; tracked as follow-up, not done here.
  transform.named_sequence @match_dynamic_depthwise_conv2d_nchw_3x3_s2(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.depthwise_conv_2d_nchw_chw"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = f16, rhs_type = f16, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [3, 3] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [2, 2] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[1], umin = 1, umax = 512 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  // Stride-3 counterpart of @match_dynamic_depthwise_conv2d_nchw -- same 1x1 kernel and Cin <= 512
  // bound (Cout is always Cin here, see that matcher's own doc comment).
  // `%strides` = [3, 3] instead of [1, 1] -- hardware-confirmed depthwise
  // fp16 at stride 3 by conv_depthwise_stride_hw.rs (see DESIGN_NOTES.md
  // "Depthwise stride hardware confirmation").
  transform.named_sequence @match_dynamic_depthwise_conv2d_nchw_s3(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.depthwise_conv_2d_nchw_chw"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = f16, rhs_type = f16, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [3, 3] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[1], umin = 1, umax = 512 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  // Stride-3 counterpart of @match_dynamic_depthwise_conv2d_nchw_3x3 -- same 3x3 kernel and Cin <= 512
  // bound (Cout is always Cin here, see that matcher's own doc comment).
  // `%strides` = [3, 3] instead of [1, 1] -- hardware-confirmed depthwise
  // fp16 at stride 3 by conv_depthwise_stride_hw.rs (see DESIGN_NOTES.md
  // "Depthwise stride hardware confirmation").
  transform.named_sequence @match_dynamic_depthwise_conv2d_nchw_3x3_s3(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.depthwise_conv_2d_nchw_chw"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = f16, rhs_type = f16, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [3, 3] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [3, 3] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[1], umin = 1, umax = 512 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  // Stride-4 counterpart of @match_dynamic_depthwise_conv2d_nchw -- same 1x1 kernel and Cin <= 512
  // bound (Cout is always Cin here, see that matcher's own doc comment).
  // `%strides` = [4, 4] instead of [1, 1] -- hardware-confirmed depthwise
  // fp16 at stride 4 by conv_depthwise_stride_hw.rs (see DESIGN_NOTES.md
  // "Depthwise stride hardware confirmation").
  transform.named_sequence @match_dynamic_depthwise_conv2d_nchw_s4(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.depthwise_conv_2d_nchw_chw"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = f16, rhs_type = f16, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [4, 4] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[1], umin = 1, umax = 512 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  // Stride-4 counterpart of @match_dynamic_depthwise_conv2d_nchw_3x3 -- same 3x3 kernel and Cin <= 512
  // bound (Cout is always Cin here, see that matcher's own doc comment).
  // `%strides` = [4, 4] instead of [1, 1] -- hardware-confirmed depthwise
  // fp16 at stride 4 by conv_depthwise_stride_hw.rs (see DESIGN_NOTES.md
  // "Depthwise stride hardware confirmation").
  transform.named_sequence @match_dynamic_depthwise_conv2d_nchw_3x3_s4(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.depthwise_conv_2d_nchw_chw"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = f16, rhs_type = f16, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [3, 3] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [4, 4] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[1], umin = 1, umax = 512 : !transform.any_value
    transform.yield %root : !transform.any_op
  }


  transform.named_sequence @cast_and_call_rocket_matmul(%root: !transform.any_op {transform.readonly}) {
    %ins = transform.get_operand %root[all] : (!transform.any_op) -> !transform.any_value
    %out = transform.get_result %root[all] : (!transform.any_op) -> !transform.any_value
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %topology_attr = transform.param.constant #hal.device.topology<links = [
        (@rocket_device -> @cpu_device = {transparent_access = true, unified_memory = true}),
        (@cpu_device -> @rocket_device = {transparent_access = true, unified_memory = true})
      ]> -> !transform.any_param
    transform.annotate %module "stream.topology" = %topology_attr : !transform.any_op, !transform.any_param
    %executable = transform.util.import_symbol @rocket_matmul_executable into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_rocket_matmul into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
          transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }




  transform.named_sequence @cast_and_call_pooling_avg_nhwc(%root: !transform.any_op {transform.readonly}) {
    %ins = transform.get_operand %root[all] : (!transform.any_op) -> !transform.any_value
    %out = transform.get_result %root[all] : (!transform.any_op) -> !transform.any_value
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %topology_attr = transform.param.constant #hal.device.topology<links = [
        (@rocket_device -> @cpu_device = {transparent_access = true, unified_memory = true}),
        (@cpu_device -> @rocket_device = {transparent_access = true, unified_memory = true})
      ]> -> !transform.any_param
    transform.annotate %module "stream.topology" = %topology_attr : !transform.any_op, !transform.any_param
    %executable = transform.util.import_symbol @rocket_pooling_executable into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_rocket_pooling_avg_nhwc into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
          transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @cast_and_call_pooling_min_nhwc(%root: !transform.any_op {transform.readonly}) {
    %ins = transform.get_operand %root[all] : (!transform.any_op) -> !transform.any_value
    %out = transform.get_result %root[all] : (!transform.any_op) -> !transform.any_value
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %topology_attr = transform.param.constant #hal.device.topology<links = [
        (@rocket_device -> @cpu_device = {transparent_access = true, unified_memory = true}),
        (@cpu_device -> @rocket_device = {transparent_access = true, unified_memory = true})
      ]> -> !transform.any_param
    transform.annotate %module "stream.topology" = %topology_attr : !transform.any_op, !transform.any_param
    %executable = transform.util.import_symbol @rocket_pooling_min_executable into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_rocket_pooling_min_nhwc into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
          transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @cast_and_call_pooling_min_nhwc_s2(%root: !transform.any_op {transform.readonly}) {
    %ins = transform.get_operand %root[all] : (!transform.any_op) -> !transform.any_value
    %out = transform.get_result %root[all] : (!transform.any_op) -> !transform.any_value
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %topology_attr = transform.param.constant #hal.device.topology<links = [
        (@rocket_device -> @cpu_device = {transparent_access = true, unified_memory = true}),
        (@cpu_device -> @rocket_device = {transparent_access = true, unified_memory = true})
      ]> -> !transform.any_param
    transform.annotate %module "stream.topology" = %topology_attr : !transform.any_op, !transform.any_param
    %executable = transform.util.import_symbol @rocket_pooling_min_executable_s2 into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_rocket_pooling_min_nhwc_s2 into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
          transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @cast_and_call_pooling_max_nhwc(%root: !transform.any_op {transform.readonly}) {
    %ins = transform.get_operand %root[all] : (!transform.any_op) -> !transform.any_value
    %out = transform.get_result %root[all] : (!transform.any_op) -> !transform.any_value
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %topology_attr = transform.param.constant #hal.device.topology<links = [
        (@rocket_device -> @cpu_device = {transparent_access = true, unified_memory = true}),
        (@cpu_device -> @rocket_device = {transparent_access = true, unified_memory = true})
      ]> -> !transform.any_param
    transform.annotate %module "stream.topology" = %topology_attr : !transform.any_op, !transform.any_param
    %executable = transform.util.import_symbol @rocket_pooling_max_executable into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_rocket_pooling_max_nhwc into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
          transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @cast_and_call_pooling_max_nhwc_s2(%root: !transform.any_op {transform.readonly}) {
    %ins = transform.get_operand %root[all] : (!transform.any_op) -> !transform.any_value
    %out = transform.get_result %root[all] : (!transform.any_op) -> !transform.any_value
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %topology_attr = transform.param.constant #hal.device.topology<links = [
        (@rocket_device -> @cpu_device = {transparent_access = true, unified_memory = true}),
        (@cpu_device -> @rocket_device = {transparent_access = true, unified_memory = true})
      ]> -> !transform.any_param
    transform.annotate %module "stream.topology" = %topology_attr : !transform.any_op, !transform.any_param
    %executable = transform.util.import_symbol @rocket_pooling_max_executable_s2 into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_rocket_pooling_max_nhwc_s2 into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
          transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @cast_and_call_pooling_max_nchw(%root: !transform.any_op {transform.readonly}) {
    %ins = transform.get_operand %root[all] : (!transform.any_op) -> !transform.any_value
    %out = transform.get_result %root[all] : (!transform.any_op) -> !transform.any_value
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %topology_attr = transform.param.constant #hal.device.topology<links = [
        (@rocket_device -> @cpu_device = {transparent_access = true, unified_memory = true}),
        (@cpu_device -> @rocket_device = {transparent_access = true, unified_memory = true})
      ]> -> !transform.any_param
    transform.annotate %module "stream.topology" = %topology_attr : !transform.any_op, !transform.any_param
    %executable = transform.util.import_symbol @rocket_pooling_max_executable into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_rocket_pooling_max_nchw into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
          transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @cast_and_call_pooling_max_nchw_s2(%root: !transform.any_op {transform.readonly}) {
    %ins = transform.get_operand %root[all] : (!transform.any_op) -> !transform.any_value
    %out = transform.get_result %root[all] : (!transform.any_op) -> !transform.any_value
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %topology_attr = transform.param.constant #hal.device.topology<links = [
        (@rocket_device -> @cpu_device = {transparent_access = true, unified_memory = true}),
        (@cpu_device -> @rocket_device = {transparent_access = true, unified_memory = true})
      ]> -> !transform.any_param
    transform.annotate %module "stream.topology" = %topology_attr : !transform.any_op, !transform.any_param
    %executable = transform.util.import_symbol @rocket_pooling_max_executable_s2 into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_rocket_pooling_max_nchw_s2 into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
          transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @cast_and_call_pooling_avg_nchw(%root: !transform.any_op {transform.readonly}) {
    %ins = transform.get_operand %root[all] : (!transform.any_op) -> !transform.any_value
    %out = transform.get_result %root[all] : (!transform.any_op) -> !transform.any_value
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %topology_attr = transform.param.constant #hal.device.topology<links = [
        (@rocket_device -> @cpu_device = {transparent_access = true, unified_memory = true}),
        (@cpu_device -> @rocket_device = {transparent_access = true, unified_memory = true})
      ]> -> !transform.any_param
    transform.annotate %module "stream.topology" = %topology_attr : !transform.any_op, !transform.any_param
    %executable = transform.util.import_symbol @rocket_pooling_executable into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_rocket_pooling_avg_nchw into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
          transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @cast_and_call_dynamic_depthwise_conv2d_nchw(%root: !transform.any_op {transform.readonly}) {
    %ins = transform.get_operand %root[all] : (!transform.any_op) -> !transform.any_value
    %out = transform.get_result %root[all] : (!transform.any_op) -> !transform.any_value
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %topology_attr = transform.param.constant #hal.device.topology<links = [
        (@rocket_device -> @cpu_device = {transparent_access = true, unified_memory = true}),
        (@cpu_device -> @rocket_device = {transparent_access = true, unified_memory = true})
      ]> -> !transform.any_param
    transform.annotate %module "stream.topology" = %topology_attr : !transform.any_op, !transform.any_param
    %executable = transform.util.import_symbol @rocket_dynamic_depthwise_executable into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_rocket_dynamic_depthwise_conv2d_nchw into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
          transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @cast_and_call_dynamic_depthwise_conv2d_nchw_s2(%root: !transform.any_op {transform.readonly}) {
    %ins = transform.get_operand %root[all] : (!transform.any_op) -> !transform.any_value
    %out = transform.get_result %root[all] : (!transform.any_op) -> !transform.any_value
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %topology_attr = transform.param.constant #hal.device.topology<links = [
        (@rocket_device -> @cpu_device = {transparent_access = true, unified_memory = true}),
        (@cpu_device -> @rocket_device = {transparent_access = true, unified_memory = true})
      ]> -> !transform.any_param
    transform.annotate %module "stream.topology" = %topology_attr : !transform.any_op, !transform.any_param
    %executable = transform.util.import_symbol @rocket_dynamic_depthwise_executable_s2 into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_rocket_dynamic_depthwise_conv2d_nchw_s2 into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
          transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @cast_and_call_dynamic_depthwise_conv2d_nchw_s3(%root: !transform.any_op {transform.readonly}) {
    %ins = transform.get_operand %root[all] : (!transform.any_op) -> !transform.any_value
    %out = transform.get_result %root[all] : (!transform.any_op) -> !transform.any_value
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %topology_attr = transform.param.constant #hal.device.topology<links = [
        (@rocket_device -> @cpu_device = {transparent_access = true, unified_memory = true}),
        (@cpu_device -> @rocket_device = {transparent_access = true, unified_memory = true})
      ]> -> !transform.any_param
    transform.annotate %module "stream.topology" = %topology_attr : !transform.any_op, !transform.any_param
    %executable = transform.util.import_symbol @rocket_dynamic_depthwise_executable_s3 into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_rocket_dynamic_depthwise_conv2d_nchw_s3 into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
          transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @cast_and_call_dynamic_depthwise_conv2d_nchw_s4(%root: !transform.any_op {transform.readonly}) {
    %ins = transform.get_operand %root[all] : (!transform.any_op) -> !transform.any_value
    %out = transform.get_result %root[all] : (!transform.any_op) -> !transform.any_value
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %topology_attr = transform.param.constant #hal.device.topology<links = [
        (@rocket_device -> @cpu_device = {transparent_access = true, unified_memory = true}),
        (@cpu_device -> @rocket_device = {transparent_access = true, unified_memory = true})
      ]> -> !transform.any_param
    transform.annotate %module "stream.topology" = %topology_attr : !transform.any_op, !transform.any_param
    %executable = transform.util.import_symbol @rocket_dynamic_depthwise_executable_s4 into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_rocket_dynamic_depthwise_conv2d_nchw_s4 into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
          transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }


  transform.named_sequence @cast_and_call_dynamic_conv2d(%root: !transform.any_op {transform.readonly}) {
    %ins = transform.get_operand %root[all] : (!transform.any_op) -> !transform.any_value
    %out = transform.get_result %root[all] : (!transform.any_op) -> !transform.any_value
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    // Declare truthfully that rocket_device and cpu_device share unified,
    // transparently-accessible memory (real RK3588 hardware fact) so
    // ResolveTopologyQueriesPass can resolve the cross-device buffer Stream
    // forms for values flowing directly from the rocket dispatch into
    // CPU-side compute. Must be set on the REAL target module (%module),
    // not this transform-spec file's own module. Idempotent: fires once per
    // matched conv, each time just overwriting the same value.
    %topology_attr = transform.param.constant #hal.device.topology<links = [
        (@rocket_device -> @cpu_device = {transparent_access = true, unified_memory = true}),
        (@cpu_device -> @rocket_device = {transparent_access = true, unified_memory = true})
      ]> -> !transform.any_param
    transform.annotate %module "stream.topology" = %topology_attr : !transform.any_op, !transform.any_param
    %executable = transform.util.import_symbol @rocket_dynamic_executable into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_rocket_dynamic_conv2d into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
          transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @cast_and_call_dynamic_conv2d_s2(%root: !transform.any_op {transform.readonly}) {
    %ins = transform.get_operand %root[all] : (!transform.any_op) -> !transform.any_value
    %out = transform.get_result %root[all] : (!transform.any_op) -> !transform.any_value
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %topology_attr = transform.param.constant #hal.device.topology<links = [
        (@rocket_device -> @cpu_device = {transparent_access = true, unified_memory = true}),
        (@cpu_device -> @rocket_device = {transparent_access = true, unified_memory = true})
      ]> -> !transform.any_param
    transform.annotate %module "stream.topology" = %topology_attr : !transform.any_op, !transform.any_param
    %executable = transform.util.import_symbol @rocket_dynamic_executable_s2 into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_rocket_dynamic_conv2d_s2 into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
          transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @cast_and_call_dynamic_conv2d_s3(%root: !transform.any_op {transform.readonly}) {
    %ins = transform.get_operand %root[all] : (!transform.any_op) -> !transform.any_value
    %out = transform.get_result %root[all] : (!transform.any_op) -> !transform.any_value
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %topology_attr = transform.param.constant #hal.device.topology<links = [
        (@rocket_device -> @cpu_device = {transparent_access = true, unified_memory = true}),
        (@cpu_device -> @rocket_device = {transparent_access = true, unified_memory = true})
      ]> -> !transform.any_param
    transform.annotate %module "stream.topology" = %topology_attr : !transform.any_op, !transform.any_param
    %executable = transform.util.import_symbol @rocket_dynamic_executable_s3 into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_rocket_dynamic_conv2d_s3 into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
          transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @cast_and_call_dynamic_conv2d_s4(%root: !transform.any_op {transform.readonly}) {
    %ins = transform.get_operand %root[all] : (!transform.any_op) -> !transform.any_value
    %out = transform.get_result %root[all] : (!transform.any_op) -> !transform.any_value
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %topology_attr = transform.param.constant #hal.device.topology<links = [
        (@rocket_device -> @cpu_device = {transparent_access = true, unified_memory = true}),
        (@cpu_device -> @rocket_device = {transparent_access = true, unified_memory = true})
      ]> -> !transform.any_param
    transform.annotate %module "stream.topology" = %topology_attr : !transform.any_op, !transform.any_param
    %executable = transform.util.import_symbol @rocket_dynamic_executable_s4 into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_rocket_dynamic_conv2d_s4 into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
          transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }



  //===--------------------------------------------------------------------===//
  // int8 matchers
  //
  // Structurally these are the fp16 matchers above with `lhs_type = i8,
  // rhs_type = i8, output_type = i32` -- the shape predicates are identical
  // and their rationale is not repeated here, so read the fp16 matcher of
  // the same name for the CBUF-split correctness caveat behind the channel
  // bounds. What reaches them is an ONNX ConvInteger model after
  // @__transform_main's two dequantization passes: an ordinary named conv on
  // i8 operands with an i32 accumulator, its zero point already folded into
  // a separate CPU-side correction.
  //
  // The dense int8 Cin bounds were 352 (1x1) and 32 (3x3), containment for a
  // silent near-all-zero result first measured on 2026-08-31. **That cause is
  // found and fixed (2026-09-03), and both bounds are now the HAL's own
  // `MAX_INT8_INPUT_CHANNELS` of 512.**
  //
  // The cause was never ConvPlan's int8 modelling, which is why every
  // candidate ruled out at the time -- coefficient stream order, the CBUF
  // bank split across all eleven splits, `feature_grains` swept 1..40,
  // `data_entries`, the packed feature width -- came back clean. It was the
  // DPU *output writer*: the dense int8 accumulator drove `mc_surf_out = 1`,
  // the "2/4 surface serial" writer, which stops emitting once it runs out of
  // surfaces, and the host read it back as 32-channel 128-byte blocks to
  // match. Both are now the geometry `rocket-userspace`'s validated
  // int8 -> int32 program uses: `mc_surf_out = 0`, `size_e = 7`,
  // `surf_add = dataout_w * dataout_h * 8` per tile, read back as the C2=4
  // cube (16-byte atoms of four int32 lanes). See `Shape::bs_ow_size_e` and
  // `Shape::output_channel_block_bytes` in `iree-rocket-hal`.
  //
  // The old 352 boundary is explained exactly by that: "353 is also where
  // ConvPlan changes from one output tile to two" -- one tile fits inside the
  // surfaces the serial writer manages, two do not.
  //
  // Hardware after the fix, shipped path, `Dense` (non-degenerate) pattern,
  // 0 mismatches at every point: 1x1 at Cin 385, 512 (and 704 with the HAL
  // ceiling lifted), Cout 64 and 256, odd extents, one to three tiles; 3x3 at
  // Cin 33 and 256 (2304 coefficient bytes per output channel); and a 3x3
  // output extent with a 3x3 kernel, which used to be refused outright.
  // There is no coefficient-per-channel ceiling left to contain.
  //
  // 512 is the ceiling because `MAX_INT8_INPUT_CHANNELS` is 512 -- above it
  // the *channel padding* rules are unmeasured, and separately ConvPlan's
  // CBUF split is known to diverge from vendor captures for dense shapes
  // above Cin 384. Both are questions about planning, not about this writer.
  // Raising past 512 needs that split's sawtooth reset rule; see
  // `MAX_INPUT_CHANNELS`' doc comment in `conv.rs`.
  //
  // The Cout bound remains 512. It is now hardware-validated in isolation by
  // tools/e2e_conv_regression.py's exact compiled differentials: both 1x1 and
  // 3x3 at Cin=16, Cout=512, 32x32 output and non-zero input zero points
  // matched all 524288 i32 accumulators exactly on RK3588 (2026-08-31). This
  // does not weaken either Cin cap above or claim unmeasured high-Cin/high-Cout
  // interactions are safe; it establishes that Cout=512 itself is not a
  // failure trigger at a low, independently safe Cin.
  //
  // The depthwise int8 matchers keep umax = 512 and are correct across it.
  // They briefly were not: Cin whose *atom* count (ceil(Cin/16)) was one
  // short of a multiple of four -- 33..48, 97..112, 225..240 and every 64
  // thereafter -- lost its last output rows, because `data_bank_demand`
  // billed the CBUF with `weight_atoms` instead of `cbuf_atoms`. Fixed in
  // `conv.rs`; verified at Cin 33, 44..47, 112, 176, 240, 304, 368, 432 and
  // 496. The dense caps above are a *different* bug and that fix does not
  // move them, which is consistent with their atom counts already being
  // whole multiples of four.
  //===--------------------------------------------------------------------===//

  transform.named_sequence @match_dynamic_conv2d_int8(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.conv_2d_nhwc_hwcf"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = i8, rhs_type = i8, output_type = i32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    %filter_value = transform.get_operand %root[1] : (!transform.any_op) -> !transform.any_value
    // The HAL's `MAX_INT8_INPUT_CHANNELS`, raised 512 -> 1344 on hardware
    // evidence, then 1344 -> 3584 on 2026-09-06 with the rest of the rungs.
    // int8's own points at k=1: 14x14 Cout 64 at Cin 1792, 2304, 3072, 3584
    // and 4096 under `SelectorsAffine` and again under `Counting`, the
    // `onehot` read map at Cout == Cin 3584, and stride 2 at Cin 2304..4096.
    // 1344 was MobileNetV2's widest; a transformer's is 3072.
    transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 3584 : !transform.any_value
    // The HAL's `MAX_INT8_OUTPUT_CHANNELS`, split out from the shared
    // `MAX_OUTPUT_CHANNELS` at 1792, raised to 3584 on 2026-09-06. Measured
    // exact at 7x7 Cin 448 for Cout 768, 1024, 1280, 1536, 1792, 2048, and
    // then 2304, 3072, 3584 and 4096, with the CBUF split flat (7d/5w)
    // across the whole range -- the high-channel divergence is indexed by
    // `Cin`, not `Cout`.
    transform.iree.match.dim_bounds %filter_value[3], umin = 1, umax = 3584 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  transform.named_sequence @match_dynamic_conv2d_3x3_int8(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.conv_2d_nhwc_hwcf"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = i8, rhs_type = i8, output_type = i32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [3, 3] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    %filter_value = transform.get_operand %root[1] : (!transform.any_op) -> !transform.any_value
    // 1152, not `MAX_INT8_INPUT_CHANNELS` (3584 since 2026-09-06): at a 3x3
    // kernel the binding limit is the coefficient working set, not the
    // channel-padding rules.
    // `ConvPlan` plans and agrees with the vendor to Cin 1152 and **refuses**
    // Cin >= 1216 outright (the working set exceeds the eleven grantable CBUF
    // banks), so admitting past 1152 would reach the driver and panic rather
    // than fall back. 1152 is hardware-exact at Cout 64 and 448, including the
    // 1/11 splits at 1088 and 1152. The Cout bound stays 512: the corpus
    // backing above it was established against the 1x1 matcher, not this one.
    transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 1152 : !transform.any_value
    transform.iree.match.dim_bounds %filter_value[3], umin = 1, umax = 512 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  transform.named_sequence @match_dynamic_depthwise_conv2d_int8(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.depthwise_conv_2d_nhwc_hwc"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = i8, rhs_type = i8, output_type = i32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    // Only one channel count to bound: depthwise Cout is always Cin.
    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    // Raised 512 -> 1344 (2026-09-03) with the depthwise coefficient model
    // fix: the streamed working set was using the *dense* product
    // `kh*kw*Cin*64`, which scales with C and asked for 13 of eleven
    // grantable CBUF banks at C=1344. A depthwise output channel
    // accumulates over one input channel, so the contraction depth is 1.
    // See `Shape::streamed_contraction_channels`.
    transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 1344 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  transform.named_sequence @match_dynamic_depthwise_conv2d_3x3_int8(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.depthwise_conv_2d_nhwc_hwc"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = i8, rhs_type = i8, output_type = i32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [3, 3] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    // Only one channel count to bound: depthwise Cout is always Cin.
    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    // Raised 512 -> 1344 (2026-09-03) with the depthwise coefficient model
    // fix: the streamed working set was using the *dense* product
    // `kh*kw*Cin*64`, which scales with C and asked for 13 of eleven
    // grantable CBUF banks at C=1344. A depthwise output channel
    // accumulates over one input channel, so the contraction depth is 1.
    // See `Shape::streamed_contraction_channels`.
    transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 1344 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  transform.named_sequence @match_dynamic_depthwise_conv2d_int8_s2(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.depthwise_conv_2d_nhwc_hwc"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = i8, rhs_type = i8, output_type = i32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [2, 2] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    // Only one channel count to bound: depthwise Cout is always Cin.
    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    // Raised 512 -> 1344 (2026-09-03) with the depthwise coefficient model
    // fix: the streamed working set was using the *dense* product
    // `kh*kw*Cin*64`, which scales with C and asked for 13 of eleven
    // grantable CBUF banks at C=1344. A depthwise output channel
    // accumulates over one input channel, so the contraction depth is 1.
    // See `Shape::streamed_contraction_channels`.
    transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 1344 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  transform.named_sequence @match_dynamic_depthwise_conv2d_3x3_int8_s2(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.depthwise_conv_2d_nhwc_hwc"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = i8, rhs_type = i8, output_type = i32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [3, 3] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [2, 2] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    // Only one channel count to bound: depthwise Cout is always Cin.
    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    // Raised 512 -> 1344 (2026-09-03) with the depthwise coefficient model
    // fix: the streamed working set was using the *dense* product
    // `kh*kw*Cin*64`, which scales with C and asked for 13 of eleven
    // grantable CBUF banks at C=1344. A depthwise output channel
    // accumulates over one input channel, so the contraction depth is 1.
    // See `Shape::streamed_contraction_channels`.
    transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 1344 : !transform.any_value
    transform.yield %root : !transform.any_op
  }

  //===--------------------------------------------------------------------===//
  // Requantized int8 matchers
  //
  // These claim a convolution *and* its requantization epilogue, and hand
  // both to a dispatch that returns i8. That is the whole point: the
  // int8_accumulator path below returns the raw i32 accumulator and leaves a
  // CPU pass to requantize it, and it is capped at 384 coefficient bytes per
  // output channel by a DPU limitation with no programmable workaround, which
  // is what holds its dense matchers at Cin 352 (1x1) and 32 (3x3).
  //
  // The channel bounds here are the *plain* int8 measurements instead:
  // hardware-exact at every Cin through 512 at both kernel sizes, 14/14 at
  // 3x3 28x28 Cout 256 and through the compiled path at 1x1. Cout keeps the
  // 768 the vendor corpus and the Cout sweep support.
  //
  // The matched DAG is the canonical form: a plain i8 x i8 -> i32
  // convolution over a zero init, then one elementwise generic that adds the
  // per-channel bias, scales, rounds, offsets by the output zero point,
  // clamps to int8 and truncates. The int8 clamp bounds are *operands* of
  // that generic rather than constants in its body, and that is not a
  // stylistic choice: the DAG matcher compares regions under a value mapping
  // built only from the ops it walked, so a value captured from outside the
  // region can never match -- and a constant written inside the body does not
  // stay there, because the canonicalizer hoists it out before this loop
  // runs. Operands are the only form that survives both. Everything about it maps onto a hardware
  // stage -- the bias and the BS plane, the scale and OUT_CVT, the zero point
  // and OUT_CVT_OFFSET -- which is why the dispatch can replace the whole
  // subgraph rather than just the convolution. The scale and zero point are
  // *operands* of the generic rather than constants captured into its region,
  // so the DAG match yields them as inputs and they can travel to the
  // dispatch as push constants.
  //===--------------------------------------------------------------------===//

  //===--------------------------------------------------------------------===//
  // Padded convolution matchers
  //
  // These claim a convolution *and* the `tensor.pad` in front of it, so the
  // CNA does the padding instead of IREE materializing it as a full-tensor
  // copy. On ResNet50 fp16 that is 16 dispatch sites and ~12 MB per
  // inference; MobileNetV2 has almost none to give (17 of its 18 pads feed
  // depthwise convolutions that stay on the CPU, and the one that does not is
  // asymmetric).
  //
  // The matched DAG is what `rocket-fold-conv-pad` has certified: a
  // zero-filled, spatial-only, symmetric `tensor.pad` of exactly 1 in the
  // rank-5 channels-last form, collapsed to rank 4, feeding the convolution.
  // The pad amount is pinned twice -- by the `rocket.pad_top`/`_left`
  // attributes in the convolution's dictionary, which the DAG matcher
  // compares exactly, and by the `low`/`high` amounts on the template's own
  // `tensor.pad`. The *fill value* is pinned only by the pass, because the
  // DAG matcher compares attribute dictionaries and operands but never
  // regions -- a `tensor.pad` yielding something other than zero would match
  // this template. That is why the pass exists rather than the whole thing
  // being a matcher.
  //
  // Each stride needs two spellings, with and without `rocket.f16_demoted`,
  // and that is not redundancy. The marker is added by
  // `RocketDemoteConvInputsPass` only to convolutions it *narrowed*: a model
  // imported at f32 carries it, and one imported already f16 -- ResNet50's
  // fp16 export, where this lever actually pays -- does not. Since the DAG
  // matcher compares whole dictionaries, one spelling silently covers only
  // half the models.
  //===--------------------------------------------------------------------===//

  transform.named_sequence @match_dynamic_conv2d_3x3_pad1(
      %root: !transform.any_op {transform.readonly})
      -> (!transform.any_value, !transform.any_value) {
    transform.match.operation_name %root ["linalg.conv_2d_nhwc_hwcf"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = f16, rhs_type = f16, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [3, 3] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    // The bounds are the ones the unpadded 3x3 matchers already carry: this
    // changes where the padding happens, not how wide the convolution may be.
    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    %filter_value = transform.get_operand %root[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 1152 : !transform.any_value
    transform.iree.match.dim_bounds %filter_value[3], umin = 1, umax = 3584 : !transform.any_value

    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
      ^bb0(%input: tensor<1x1x?x?x?xf16>, %weights: tensor<?x?x?x?xf16>,
           %init: tensor<1x?x?x?xf32>):
        // Self-contained by construction: `rocket-fold-conv-pad` sinks the
        // fill into the region, because this matcher compares regions
        // structurally and one yielding a value defined outside it can never
        // match. The inline constant is also what pins the fill to zero.
        %padded = tensor.pad %input low[0, 0, 1, 1, 0] high[0, 0, 1, 1, 0] {
          ^bb1(%i0: index, %i1: index, %i2: index, %i3: index, %i4: index):
            %zero = arith.constant 0.000000e+00 : f16
            tensor.yield %zero : f16
        } : tensor<1x1x?x?x?xf16> to tensor<1x1x?x?x?xf16>
        %collapsed = tensor.collapse_shape %padded [[0], [1, 2], [3], [4]]
            : tensor<1x1x?x?x?xf16> into tensor<1x?x?x?xf16>
        %convolution = linalg.conv_2d_nhwc_hwcf
            {dilations = dense<1> : vector<2xi64>,
             rocket.pad_left = 1 : i64, rocket.pad_top = 1 : i64,
             strides = dense<1> : vector<2xi64>}
            ins(%collapsed, %weights : tensor<1x?x?x?xf16>, tensor<?x?x?x?xf16>)
            outs(%init : tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    transform.yield %ins, %outs : !transform.any_value, !transform.any_value
  }

  transform.named_sequence @match_dynamic_conv2d_3x3_pad1_demoted(
      %root: !transform.any_op {transform.readonly})
      -> (!transform.any_value, !transform.any_value) {
    transform.match.operation_name %root ["linalg.conv_2d_nhwc_hwcf"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = f16, rhs_type = f16, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [3, 3] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    // The bounds are the ones the unpadded 3x3 matchers already carry: this
    // changes where the padding happens, not how wide the convolution may be.
    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    %filter_value = transform.get_operand %root[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 1152 : !transform.any_value
    transform.iree.match.dim_bounds %filter_value[3], umin = 1, umax = 3584 : !transform.any_value

    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
      ^bb0(%input: tensor<1x1x?x?x?xf16>, %weights: tensor<?x?x?x?xf16>,
           %init: tensor<1x?x?x?xf32>):
        // Self-contained by construction: `rocket-fold-conv-pad` sinks the
        // fill into the region, because this matcher compares regions
        // structurally and one yielding a value defined outside it can never
        // match. The inline constant is also what pins the fill to zero.
        %padded = tensor.pad %input low[0, 0, 1, 1, 0] high[0, 0, 1, 1, 0] {
          ^bb1(%i0: index, %i1: index, %i2: index, %i3: index, %i4: index):
            %zero = arith.constant 0.000000e+00 : f16
            tensor.yield %zero : f16
        } : tensor<1x1x?x?x?xf16> to tensor<1x1x?x?x?xf16>
        %collapsed = tensor.collapse_shape %padded [[0], [1, 2], [3], [4]]
            : tensor<1x1x?x?x?xf16> into tensor<1x?x?x?xf16>
        %convolution = linalg.conv_2d_nhwc_hwcf
            {dilations = dense<1> : vector<2xi64>, rocket.f16_demoted,
             rocket.pad_left = 1 : i64, rocket.pad_top = 1 : i64,
             strides = dense<1> : vector<2xi64>}
            ins(%collapsed, %weights : tensor<1x?x?x?xf16>, tensor<?x?x?x?xf16>)
            outs(%init : tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    transform.yield %ins, %outs : !transform.any_value, !transform.any_value
  }

  transform.named_sequence @match_dynamic_conv2d_3x3_pad1_s2(
      %root: !transform.any_op {transform.readonly})
      -> (!transform.any_value, !transform.any_value) {
    transform.match.operation_name %root ["linalg.conv_2d_nhwc_hwcf"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = f16, rhs_type = f16, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [3, 3] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [2, 2] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    // The bounds are the ones the unpadded 3x3 matchers already carry: this
    // changes where the padding happens, not how wide the convolution may be.
    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    %filter_value = transform.get_operand %root[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 512 : !transform.any_value
    transform.iree.match.dim_bounds %filter_value[3], umin = 1, umax = 3584 : !transform.any_value

    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
      ^bb0(%input: tensor<1x1x?x?x?xf16>, %weights: tensor<?x?x?x?xf16>,
           %init: tensor<1x?x?x?xf32>):
        // Self-contained by construction: `rocket-fold-conv-pad` sinks the
        // fill into the region, because this matcher compares regions
        // structurally and one yielding a value defined outside it can never
        // match. The inline constant is also what pins the fill to zero.
        %padded = tensor.pad %input low[0, 0, 1, 1, 0] high[0, 0, 1, 1, 0] {
          ^bb1(%i0: index, %i1: index, %i2: index, %i3: index, %i4: index):
            %zero = arith.constant 0.000000e+00 : f16
            tensor.yield %zero : f16
        } : tensor<1x1x?x?x?xf16> to tensor<1x1x?x?x?xf16>
        %collapsed = tensor.collapse_shape %padded [[0], [1, 2], [3], [4]]
            : tensor<1x1x?x?x?xf16> into tensor<1x?x?x?xf16>
        %convolution = linalg.conv_2d_nhwc_hwcf
            {dilations = dense<1> : vector<2xi64>,
             rocket.pad_left = 1 : i64, rocket.pad_top = 1 : i64,
             strides = dense<2> : vector<2xi64>}
            ins(%collapsed, %weights : tensor<1x?x?x?xf16>, tensor<?x?x?x?xf16>)
            outs(%init : tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    transform.yield %ins, %outs : !transform.any_value, !transform.any_value
  }

  transform.named_sequence @match_dynamic_conv2d_3x3_pad1_demoted_s2(
      %root: !transform.any_op {transform.readonly})
      -> (!transform.any_value, !transform.any_value) {
    transform.match.operation_name %root ["linalg.conv_2d_nhwc_hwcf"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = f16, rhs_type = f16, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [3, 3] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [2, 2] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    // The bounds are the ones the unpadded 3x3 matchers already carry: this
    // changes where the padding happens, not how wide the convolution may be.
    %input_value = transform.get_operand %root[0] : (!transform.any_op) -> !transform.any_value
    %filter_value = transform.get_operand %root[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 512 : !transform.any_value
    transform.iree.match.dim_bounds %filter_value[3], umin = 1, umax = 3584 : !transform.any_value

    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
      ^bb0(%input: tensor<1x1x?x?x?xf16>, %weights: tensor<?x?x?x?xf16>,
           %init: tensor<1x?x?x?xf32>):
        // Self-contained by construction: `rocket-fold-conv-pad` sinks the
        // fill into the region, because this matcher compares regions
        // structurally and one yielding a value defined outside it can never
        // match. The inline constant is also what pins the fill to zero.
        %padded = tensor.pad %input low[0, 0, 1, 1, 0] high[0, 0, 1, 1, 0] {
          ^bb1(%i0: index, %i1: index, %i2: index, %i3: index, %i4: index):
            %zero = arith.constant 0.000000e+00 : f16
            tensor.yield %zero : f16
        } : tensor<1x1x?x?x?xf16> to tensor<1x1x?x?x?xf16>
        %collapsed = tensor.collapse_shape %padded [[0], [1, 2], [3], [4]]
            : tensor<1x1x?x?x?xf16> into tensor<1x?x?x?xf16>
        %convolution = linalg.conv_2d_nhwc_hwcf
            {dilations = dense<1> : vector<2xi64>, rocket.f16_demoted,
             rocket.pad_left = 1 : i64, rocket.pad_top = 1 : i64,
             strides = dense<2> : vector<2xi64>}
            ins(%collapsed, %weights : tensor<1x?x?x?xf16>, tensor<?x?x?x?xf16>)
            outs(%init : tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    transform.yield %ins, %outs : !transform.any_value, !transform.any_value
  }

  //===--------------------------------------------------------------------===//
  // Fused ReLU6 matchers
  //
  // These claim a convolution *and* the ReLU6 that follows it, and hand both
  // to a dispatch that clamps in the DPU's own BN stage. On MobileNetV2 fp16
  // that removes 18 standalone CPU dispatches -- 43.8 MB of f32 read and
  // written per inference to do three instructions per element -- because
  // nothing fuses across a Rocket dispatch boundary (ISSUES.md P8).
  //
  // The matched DAG is the canonical form `rocket-fuse-conv-relu6` produces:
  // a plain f16 x f16 -> f32 convolution over a zero init, then one
  // elementwise generic that adds the per-channel bias, clamps at 0 and
  // clamps at the ceiling. Every piece of it maps onto a hardware stage --
  // the bias and BS, the clamp and BN -- which is why the dispatch can
  // replace both ops rather than just the convolution.
  //
  // Three things about the form are load-bearing rather than stylistic, and
  // all three are the pass's doing:
  //
  //   * The bias and the bounds are *operands* of the generic. A value
  //     captured from outside the region can never match, because
  //     `cast_compatible_dag_from_root` builds its value mapping only from
  //     the ops it walked, and a constant written inside the body does not
  //     stay there -- the canonicaliser hoists it out.
  //   * The channels-last `tensor.expand_shape` is moved *after* the clamp.
  //     Its output shape is a static attribute that differs at every site,
  //     and the DAG matcher compares whole attribute dictionaries, so a
  //     template spanning one could match at most a single convolution.
  //   * The ceiling is not checked here and cannot be: a matcher matches
  //     structure, not constants. The pass only produces this form for a
  //     ceiling of exactly 6.0, which is what makes the target's static
  //     `activation_cmp` correct.
  //
  // Channel bounds are the plain fp16 ones -- this path changes what the BS
  // and BN stages do, not how wide the convolution may be.
  //===--------------------------------------------------------------------===//

  transform.named_sequence @match_dynamic_conv2d_relu6(
      %root: !transform.any_op {transform.readonly})
      -> (!transform.any_value, !transform.any_value) {
    transform.match.operation_name %root ["linalg.generic"] : !transform.any_op
    %conv = transform.get_producer_of_operand %root[0]
        : (!transform.any_op) -> !transform.any_op
    transform.match.operation_name %conv ["linalg.conv_2d_nhwc_hwcf"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %conv,
          lhs_type = f16, rhs_type = f16, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    %input_value = transform.get_operand %conv[0] : (!transform.any_op) -> !transform.any_value
    %filter_value = transform.get_operand %conv[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 3584 : !transform.any_value
    transform.iree.match.dim_bounds %filter_value[3], umin = 1, umax = 3584 : !transform.any_value

    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
      ^bb0(%input: tensor<1x?x?x?xf16>, %weights: tensor<?x?x?x?xf16>,
           %acc_init: tensor<1x?x?x?xf32>, %bias: tensor<?xf32>,
           %low: f32, %high: f32,
           %out_init: tensor<1x?x?x?xf32>):
        // `rocket.f16_demoted` is on the payload convolution --
        // RocketDemoteConvInputsPass marks every convolution it narrows --
        // and `cast_compatible_dag_from_root` compares whole attribute
        // dictionaries, so leaving it out of the template makes every
        // convolution decline.
        %accumulator = linalg.conv_2d_nhwc_hwcf
            {dilations = dense<1> : vector<2xi64>, rocket.f16_demoted,
             strides = dense<1> : vector<2xi64>}
            ins(%input, %weights : tensor<1x?x?x?xf16>, tensor<?x?x?x?xf16>)
            outs(%acc_init : tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32>
        %activated = linalg.generic {
            indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                             affine_map<(d0, d1, d2, d3) -> (d3)>,
                             affine_map<(d0, d1, d2, d3) -> ()>,
                             affine_map<(d0, d1, d2, d3) -> ()>,
                             affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
            ins(%accumulator, %bias, %low, %high
                : tensor<1x?x?x?xf32>, tensor<?xf32>, f32, f32)
            outs(%out_init : tensor<1x?x?x?xf32>) {
          ^bb1(%raw: f32, %channel_bias: f32, %lo: f32, %hi: f32, %unused: f32):
            %biased = arith.addf %raw, %channel_bias : f32
            %low_clamped = arith.maximumf %biased, %lo : f32
            %clamped = arith.minimumf %low_clamped, %hi : f32
            linalg.yield %clamped : f32
        } -> tensor<1x?x?x?xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    transform.yield %ins, %outs : !transform.any_value, !transform.any_value
  }

  // The 3x3 twin. Spelled separately for the same reason every other conv
  // matcher here is: 2x2 and non-square filters route through different
  // ConvPlan partitions and must never be claimed by widening a bound. The
  // Cin ceiling is @match_dynamic_conv2d_3x3's own 1152, not the 1x1 3584.
  transform.named_sequence @match_dynamic_conv2d_3x3_relu6(
      %root: !transform.any_op {transform.readonly})
      -> (!transform.any_value, !transform.any_value) {
    transform.match.operation_name %root ["linalg.generic"] : !transform.any_op
    %conv = transform.get_producer_of_operand %root[0]
        : (!transform.any_op) -> !transform.any_op
    transform.match.operation_name %conv ["linalg.conv_2d_nhwc_hwcf"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %conv,
          lhs_type = f16, rhs_type = f16, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [3, 3] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    %input_value = transform.get_operand %conv[0] : (!transform.any_op) -> !transform.any_value
    %filter_value = transform.get_operand %conv[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 1152 : !transform.any_value
    transform.iree.match.dim_bounds %filter_value[3], umin = 1, umax = 3584 : !transform.any_value

    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
      ^bb0(%input: tensor<1x?x?x?xf16>, %weights: tensor<?x?x?x?xf16>,
           %acc_init: tensor<1x?x?x?xf32>, %bias: tensor<?xf32>,
           %low: f32, %high: f32,
           %out_init: tensor<1x?x?x?xf32>):
        // `rocket.f16_demoted` is on the payload convolution --
        // RocketDemoteConvInputsPass marks every convolution it narrows --
        // and `cast_compatible_dag_from_root` compares whole attribute
        // dictionaries, so leaving it out of the template makes every
        // convolution decline.
        %accumulator = linalg.conv_2d_nhwc_hwcf
            {dilations = dense<1> : vector<2xi64>, rocket.f16_demoted,
             strides = dense<1> : vector<2xi64>}
            ins(%input, %weights : tensor<1x?x?x?xf16>, tensor<?x?x?x?xf16>)
            outs(%acc_init : tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32>
        %activated = linalg.generic {
            indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                             affine_map<(d0, d1, d2, d3) -> (d3)>,
                             affine_map<(d0, d1, d2, d3) -> ()>,
                             affine_map<(d0, d1, d2, d3) -> ()>,
                             affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
            ins(%accumulator, %bias, %low, %high
                : tensor<1x?x?x?xf32>, tensor<?xf32>, f32, f32)
            outs(%out_init : tensor<1x?x?x?xf32>) {
          ^bb1(%raw: f32, %channel_bias: f32, %lo: f32, %hi: f32, %unused: f32):
            %biased = arith.addf %raw, %channel_bias : f32
            %low_clamped = arith.maximumf %biased, %lo : f32
            %clamped = arith.minimumf %low_clamped, %hi : f32
            linalg.yield %clamped : f32
        } -> tensor<1x?x?x?xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    transform.yield %ins, %outs : !transform.any_value, !transform.any_value
  }

  // Depthwise ReLU6, stride 1. Roots at the clamp and claims the depthwise
  // convolution with it. stride 1
  //
  // The canonical form differs from the dense one in exactly one place: the
  // bias map is `(d1)`, not `(d3)`. The channels-last conversion leaves a
  // depthwise convolution in NCHW and puts the layout change on its *result*,
  // so the clamp `rocket-fuse-conv-relu6` moves in front of that transpose is
  // NCHW too, and the channel is dimension 1.
  transform.named_sequence @match_dynamic_depthwise_conv2d_nchw_relu6(
      %root: !transform.any_op {transform.readonly})
      -> (!transform.any_value, !transform.any_value) {
    transform.match.operation_name %root ["linalg.generic"] : !transform.any_op
    %conv = transform.get_producer_of_operand %root[0]
        : (!transform.any_op) -> !transform.any_op
    transform.match.operation_name %conv ["linalg.depthwise_conv_2d_nchw_chw"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %conv,
          lhs_type = f16, rhs_type = f16, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    // NCHW: the channel is dimension 1. Cout is always Cin for depthwise, so
    // there is only one bound to place, and it is the fp16 depthwise
    // matchers' own 512 rather than any dense ceiling.
    %input_value = transform.get_operand %conv[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[1], umin = 1, umax = 512 : !transform.any_value

    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
      ^bb0(%input: tensor<1x?x?x?xf16>, %weights: tensor<?x?x?xf16>,
           %acc_init: tensor<1x?x?x?xf32>, %bias: tensor<?xf32>,
           %low: f32, %high: f32,
           %out_init: tensor<1x?x?x?xf32>):
        %accumulator = linalg.depthwise_conv_2d_nchw_chw
            {dilations = dense<1> : vector<2xi64>, rocket.f16_demoted,
             strides = dense<1> : vector<2xi64>}
            ins(%input, %weights : tensor<1x?x?x?xf16>, tensor<?x?x?xf16>)
            outs(%acc_init : tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32>
        %activated = linalg.generic {
            indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                             affine_map<(d0, d1, d2, d3) -> (d1)>,
                             affine_map<(d0, d1, d2, d3) -> ()>,
                             affine_map<(d0, d1, d2, d3) -> ()>,
                             affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
            ins(%accumulator, %bias, %low, %high
                : tensor<1x?x?x?xf32>, tensor<?xf32>, f32, f32)
            outs(%out_init : tensor<1x?x?x?xf32>) {
          ^bb1(%raw: f32, %channel_bias: f32, %lo: f32, %hi: f32, %unused: f32):
            %biased = arith.addf %raw, %channel_bias : f32
            %low_clamped = arith.maximumf %biased, %lo : f32
            %clamped = arith.minimumf %low_clamped, %hi : f32
            linalg.yield %clamped : f32
        } -> tensor<1x?x?x?xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    transform.yield %ins, %outs : !transform.any_value, !transform.any_value
  }

  // Depthwise ReLU6, stride 2. Roots at the clamp and claims the depthwise
  // convolution with it. stride 2
  //
  // The canonical form differs from the dense one in exactly one place: the
  // bias map is `(d1)`, not `(d3)`. The channels-last conversion leaves a
  // depthwise convolution in NCHW and puts the layout change on its *result*,
  // so the clamp `rocket-fuse-conv-relu6` moves in front of that transpose is
  // NCHW too, and the channel is dimension 1.
  transform.named_sequence @match_dynamic_depthwise_conv2d_nchw_relu6_s2(
      %root: !transform.any_op {transform.readonly})
      -> (!transform.any_value, !transform.any_value) {
    transform.match.operation_name %root ["linalg.generic"] : !transform.any_op
    %conv = transform.get_producer_of_operand %root[0]
        : (!transform.any_op) -> !transform.any_op
    transform.match.operation_name %conv ["linalg.depthwise_conv_2d_nchw_chw"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %conv,
          lhs_type = f16, rhs_type = f16, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [2, 2] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    // NCHW: the channel is dimension 1. Cout is always Cin for depthwise, so
    // there is only one bound to place, and it is the fp16 depthwise
    // matchers' own 512 rather than any dense ceiling.
    %input_value = transform.get_operand %conv[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[1], umin = 1, umax = 512 : !transform.any_value

    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
      ^bb0(%input: tensor<1x?x?x?xf16>, %weights: tensor<?x?x?xf16>,
           %acc_init: tensor<1x?x?x?xf32>, %bias: tensor<?xf32>,
           %low: f32, %high: f32,
           %out_init: tensor<1x?x?x?xf32>):
        %accumulator = linalg.depthwise_conv_2d_nchw_chw
            {dilations = dense<1> : vector<2xi64>, rocket.f16_demoted,
             strides = dense<2> : vector<2xi64>}
            ins(%input, %weights : tensor<1x?x?x?xf16>, tensor<?x?x?xf16>)
            outs(%acc_init : tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32>
        %activated = linalg.generic {
            indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                             affine_map<(d0, d1, d2, d3) -> (d1)>,
                             affine_map<(d0, d1, d2, d3) -> ()>,
                             affine_map<(d0, d1, d2, d3) -> ()>,
                             affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
            ins(%accumulator, %bias, %low, %high
                : tensor<1x?x?x?xf32>, tensor<?xf32>, f32, f32)
            outs(%out_init : tensor<1x?x?x?xf32>) {
          ^bb1(%raw: f32, %channel_bias: f32, %lo: f32, %hi: f32, %unused: f32):
            %biased = arith.addf %raw, %channel_bias : f32
            %low_clamped = arith.maximumf %biased, %lo : f32
            %clamped = arith.minimumf %low_clamped, %hi : f32
            linalg.yield %clamped : f32
        } -> tensor<1x?x?x?xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    transform.yield %ins, %outs : !transform.any_value, !transform.any_value
  }

  transform.named_sequence @match_dynamic_conv2d_int8_requant(
      %root: !transform.any_op {transform.readonly})
      -> (!transform.any_value, !transform.any_value) {
    transform.match.operation_name %root ["linalg.generic"] : !transform.any_op
    %conv = transform.get_producer_of_operand %root[0]
        : (!transform.any_op) -> !transform.any_op
    transform.match.operation_name %conv ["linalg.conv_2d_nhwc_hwcf"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %conv,
          lhs_type = i8, rhs_type = i8, output_type = i32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    %input_value = transform.get_operand %conv[0] : (!transform.any_op) -> !transform.any_value
    %filter_value = transform.get_operand %conv[1] : (!transform.any_op) -> !transform.any_value
    // Raised 512 -> 816 on 2026-09-06. The ceiling is a *model* measurement,
    // not a shape one, and it is lower than every isolated test supports:
    // the HAL sweep `dtype_boundary_probe` is exact to Cin 1792 under both
    // patterns valid for this path (`selectors-affine` and `onehot`), and
    // `tools/e2e_conv_regression.py` is exact at Cin 1344 Cout 448 -- max
    // error 0, not 1 -- for the very convolution that breaks the model.
    // Admitting Cin 1344 moves MobileNetV2's logits from max|diff| 0.33 to
    // 5.01 and its argmax from 447 to 977, bisected to that one shape. Two
    // hypotheses are ruled out by measurement: it is not the shape (exact in
    // isolation) and not the folded bias magnitude (`requant_int8_1x1_
    // large_bias` puts a million-scale bias through the same shape and is
    // exact). The discriminator is unidentified, so the bound stops at the
    // widest Cin the model is *measured* correct at. See ISSUES.md.
    //
    // The accumulator path's own caps do not apply here and never did: they
    // came from the 384-coefficient-bytes-per-output-channel limit that
    // `int8_accumulator` has and this path does not.
    transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 816 : !transform.any_value
    // Cout has a *lower* bound of 32, and it is a measurement, not a
    // convention. MobileNetV2's `112x112 Cin 48 -> Cout 24` pointwise
    // convolution is wrong on this path: admitting it alone moves the
    // model's logits from max|diff| 0.40 against a CPU reference to 4.71,
    // with the mean rising from 0.07 to 0.99 against a logit standard
    // deviation of 1.17 -- the output stops being a classification. Every
    // other Cout the model asks for is exact, including 88, which is not a
    // multiple of the 16-channel atom either, so the rule is not "whole
    // atoms": 24 is simply below the smallest Cout measured correct (32).
    // Bisected on `planck` 2026-09-06 by admitting convolutions in Cin
    // order and then excluding this one shape, which restored the baseline
    // exactly. 25..31 are untested and excluded with it.
    transform.iree.match.dim_bounds %filter_value[3], umin = 32, umax = 1792 : !transform.any_value

    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
      ^bb0(%input: tensor<1x?x?x?xi8>, %weights: tensor<?x?x?x?xi8>,
           %acc_init: tensor<1x?x?x?xi32>, %bias: tensor<?xi32>,
           %output_scale: f32, %output_zero_point: i32,
           %int8_min: f32, %int8_max: f32,
           %out_init: tensor<1x?x?x?xi8>):
        %accumulator = linalg.conv_2d_nhwc_hwcf
            {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
            ins(%input, %weights : tensor<1x?x?x?xi8>, tensor<?x?x?x?xi8>)
            outs(%acc_init : tensor<1x?x?x?xi32>) -> tensor<1x?x?x?xi32>
        %quantized = linalg.generic {
            indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                             affine_map<(d0, d1, d2, d3) -> (d3)>,
                             affine_map<(d0, d1, d2, d3) -> ()>,
                             affine_map<(d0, d1, d2, d3) -> ()>,
                             affine_map<(d0, d1, d2, d3) -> ()>,
                             affine_map<(d0, d1, d2, d3) -> ()>,
                             affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
            ins(%accumulator, %bias, %output_scale, %output_zero_point,
                %int8_min, %int8_max
                : tensor<1x?x?x?xi32>, tensor<?xi32>, f32, i32, f32, f32)
            outs(%out_init : tensor<1x?x?x?xi8>) {
          ^bb1(%raw: i32, %channel_bias: i32, %scale: f32, %zero_point: i32,
               %low: f32, %high: f32, %unused: i8):
            %biased = arith.addi %raw, %channel_bias : i32
            %real = arith.sitofp %biased : i32 to f32
            %scaled = arith.mulf %real, %scale : f32
            %rounded = math.roundeven %scaled : f32
            %zero_point_f32 = arith.sitofp %zero_point : i32 to f32
            %offset = arith.addf %rounded, %zero_point_f32 : f32
            %low_clamped = arith.maximumf %offset, %low : f32
            %clamped = arith.minimumf %low_clamped, %high : f32
            %narrowed = arith.fptosi %clamped : f32 to i8
            linalg.yield %narrowed : i8
        } -> tensor<1x?x?x?xi8>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    transform.yield %ins, %outs : !transform.any_value, !transform.any_value
  }

  transform.named_sequence @match_dynamic_conv2d_3x3_int8_requant(
      %root: !transform.any_op {transform.readonly})
      -> (!transform.any_value, !transform.any_value) {
    transform.match.operation_name %root ["linalg.generic"] : !transform.any_op
    %conv = transform.get_producer_of_operand %root[0]
        : (!transform.any_op) -> !transform.any_op
    transform.match.operation_name %conv ["linalg.conv_2d_nhwc_hwcf"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %conv,
          lhs_type = i8, rhs_type = i8, output_type = i32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [3, 3] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    %input_value = transform.get_operand %conv[0] : (!transform.any_op) -> !transform.any_value
    %filter_value = transform.get_operand %conv[1] : (!transform.any_op) -> !transform.any_value
    // Raised 512 -> 768, gated by `requant_int8_3x3_cin768`. Lower than the
    // 1x1 ceiling because that is where this kernel's own compiled
    // differential stops, not because 3x3 is known to fail above it -- the
    // HAL sweep is exact at 3x3 Cin 1024 too. MobileNetV2's 3x3 convolutions
    // are all depthwise, so nothing in the measured models needs more.
    transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 768 : !transform.any_value
    // Cout has a *lower* bound of 32, and it is a measurement, not a
    // convention. MobileNetV2's `112x112 Cin 48 -> Cout 24` pointwise
    // convolution is wrong on this path: admitting it alone moves the
    // model's logits from max|diff| 0.40 against a CPU reference to 4.71,
    // with the mean rising from 0.07 to 0.99 against a logit standard
    // deviation of 1.17 -- the output stops being a classification. Every
    // other Cout the model asks for is exact, including 88, which is not a
    // multiple of the 16-channel atom either, so the rule is not "whole
    // atoms": 24 is simply below the smallest Cout measured correct (32).
    // Bisected on `planck` 2026-09-06 by admitting convolutions in Cin
    // order and then excluding this one shape, which restored the baseline
    // exactly. 25..31 are untested and excluded with it.
    transform.iree.match.dim_bounds %filter_value[3], umin = 32, umax = 768 : !transform.any_value

    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
      ^bb0(%input: tensor<1x?x?x?xi8>, %weights: tensor<?x?x?x?xi8>,
           %acc_init: tensor<1x?x?x?xi32>, %bias: tensor<?xi32>,
           %output_scale: f32, %output_zero_point: i32,
           %int8_min: f32, %int8_max: f32,
           %out_init: tensor<1x?x?x?xi8>):
        %accumulator = linalg.conv_2d_nhwc_hwcf
            {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
            ins(%input, %weights : tensor<1x?x?x?xi8>, tensor<?x?x?x?xi8>)
            outs(%acc_init : tensor<1x?x?x?xi32>) -> tensor<1x?x?x?xi32>
        %quantized = linalg.generic {
            indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                             affine_map<(d0, d1, d2, d3) -> (d3)>,
                             affine_map<(d0, d1, d2, d3) -> ()>,
                             affine_map<(d0, d1, d2, d3) -> ()>,
                             affine_map<(d0, d1, d2, d3) -> ()>,
                             affine_map<(d0, d1, d2, d3) -> ()>,
                             affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
            ins(%accumulator, %bias, %output_scale, %output_zero_point,
                %int8_min, %int8_max
                : tensor<1x?x?x?xi32>, tensor<?xi32>, f32, i32, f32, f32)
            outs(%out_init : tensor<1x?x?x?xi8>) {
          ^bb1(%raw: i32, %channel_bias: i32, %scale: f32, %zero_point: i32,
               %low: f32, %high: f32, %unused: i8):
            %biased = arith.addi %raw, %channel_bias : i32
            %real = arith.sitofp %biased : i32 to f32
            %scaled = arith.mulf %real, %scale : f32
            %rounded = math.roundeven %scaled : f32
            %zero_point_f32 = arith.sitofp %zero_point : i32 to f32
            %offset = arith.addf %rounded, %zero_point_f32 : f32
            %low_clamped = arith.maximumf %offset, %low : f32
            %clamped = arith.minimumf %low_clamped, %high : f32
            %narrowed = arith.fptosi %clamped : f32 to i8
            linalg.yield %narrowed : i8
        } -> tensor<1x?x?x?xi8>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    transform.yield %ins, %outs : !transform.any_value, !transform.any_value
  }

  // The depthwise twin of @match_dynamic_conv2d_int8_requant. Same epilogue,
  // same canonical form, a different convolution op and a rank-3 filter.
  //
  // `Cout` carries no lower bound here, unlike the dense requantized
  // matchers. That bound exists because dense `Cout` 24 is wrong on hardware;
  // a depthwise convolution's `Cout` is its `Cin`, and
  // `conv_depthwise_requant_hw.rs` measures 48 exact, which is the narrowest
  // MobileNetV2 asks for. The upper bound is the depthwise coefficient
  // model's own 1344, shared with the accumulator depthwise matchers.
  transform.named_sequence @match_dynamic_depthwise_conv2d_int8_requant(
      %root: !transform.any_op {transform.readonly})
      -> (!transform.any_value, !transform.any_value) {
    transform.match.operation_name %root ["linalg.generic"] : !transform.any_op
    %conv = transform.get_producer_of_operand %root[0]
        : (!transform.any_op) -> !transform.any_op
    transform.match.operation_name %conv ["linalg.depthwise_conv_2d_nhwc_hwc"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %conv,
          lhs_type = i8, rhs_type = i8, output_type = i32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    %input_value = transform.get_operand %conv[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 1344 : !transform.any_value

    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
      ^bb0(%input: tensor<1x?x?x?xi8>, %weights: tensor<?x?x?xi8>,
           %acc_init: tensor<1x?x?x?xi32>, %bias: tensor<?xi32>,
           %output_scale: f32, %output_zero_point: i32,
           %int8_min: f32, %int8_max: f32,
           %out_init: tensor<1x?x?x?xi8>):
        %accumulator = linalg.depthwise_conv_2d_nhwc_hwc
            {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
            ins(%input, %weights : tensor<1x?x?x?xi8>, tensor<?x?x?xi8>)
            outs(%acc_init : tensor<1x?x?x?xi32>) -> tensor<1x?x?x?xi32>
        %quantized = linalg.generic {
            indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                             affine_map<(d0, d1, d2, d3) -> (d3)>,
                             affine_map<(d0, d1, d2, d3) -> ()>,
                             affine_map<(d0, d1, d2, d3) -> ()>,
                             affine_map<(d0, d1, d2, d3) -> ()>,
                             affine_map<(d0, d1, d2, d3) -> ()>,
                             affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
            ins(%accumulator, %bias, %output_scale, %output_zero_point,
                %int8_min, %int8_max
                : tensor<1x?x?x?xi32>, tensor<?xi32>, f32, i32, f32, f32)
            outs(%out_init : tensor<1x?x?x?xi8>) {
          ^bb1(%raw: i32, %channel_bias: i32, %scale: f32, %zero_point: i32,
               %low: f32, %high: f32, %unused: i8):
            %biased = arith.addi %raw, %channel_bias : i32
            %real = arith.sitofp %biased : i32 to f32
            %scaled = arith.mulf %real, %scale : f32
            %rounded = math.roundeven %scaled : f32
            %zero_point_f32 = arith.sitofp %zero_point : i32 to f32
            %offset = arith.addf %rounded, %zero_point_f32 : f32
            %low_clamped = arith.maximumf %offset, %low : f32
            %clamped = arith.minimumf %low_clamped, %high : f32
            %narrowed = arith.fptosi %clamped : f32 to i8
            linalg.yield %narrowed : i8
        } -> tensor<1x?x?x?xi8>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    transform.yield %ins, %outs : !transform.any_value, !transform.any_value
  }

  // 3x3 counterpart, spelled out as its own matcher rather than widening the
  // kernel check to 1..=3, matching how every other kernel variant in this
  // file is written -- 2x2 and non-square combinations route through
  // different ConvPlan partitions and must never be claimed by accident.
  //
  // This is the one that matters for a real model: every depthwise
  // convolution in MobileNetV2 is 3x3. Writing only the 1x1 variant first was
  // a silent no-op -- the matcher declined every convolution in the model and
  // the accumulator matchers picked them up again, which looks exactly like
  // the matcher not existing.
  //
  // `Cout` carries no lower bound here, unlike the dense requantized
  // matchers. That bound exists because dense `Cout` 24 is wrong on hardware;
  // a depthwise convolution's `Cout` is its `Cin`, and
  // `conv_depthwise_requant_hw.rs` measures 48 exact, which is the narrowest
  // MobileNetV2 asks for. The upper bound is the depthwise coefficient
  // model's own 1344, shared with the accumulator depthwise matchers.
  transform.named_sequence @match_dynamic_depthwise_conv2d_3x3_int8_requant(
      %root: !transform.any_op {transform.readonly})
      -> (!transform.any_value, !transform.any_value) {
    transform.match.operation_name %root ["linalg.generic"] : !transform.any_op
    %conv = transform.get_producer_of_operand %root[0]
        : (!transform.any_op) -> !transform.any_op
    transform.match.operation_name %conv ["linalg.depthwise_conv_2d_nhwc_hwc"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %conv,
          lhs_type = i8, rhs_type = i8, output_type = i32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [-1, -1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [3, 3] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [-1] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>

    %input_value = transform.get_operand %conv[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 1344 : !transform.any_value

    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
      ^bb0(%input: tensor<1x?x?x?xi8>, %weights: tensor<?x?x?xi8>,
           %acc_init: tensor<1x?x?x?xi32>, %bias: tensor<?xi32>,
           %output_scale: f32, %output_zero_point: i32,
           %int8_min: f32, %int8_max: f32,
           %out_init: tensor<1x?x?x?xi8>):
        %accumulator = linalg.depthwise_conv_2d_nhwc_hwc
            {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
            ins(%input, %weights : tensor<1x?x?x?xi8>, tensor<?x?x?xi8>)
            outs(%acc_init : tensor<1x?x?x?xi32>) -> tensor<1x?x?x?xi32>
        %quantized = linalg.generic {
            indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                             affine_map<(d0, d1, d2, d3) -> (d3)>,
                             affine_map<(d0, d1, d2, d3) -> ()>,
                             affine_map<(d0, d1, d2, d3) -> ()>,
                             affine_map<(d0, d1, d2, d3) -> ()>,
                             affine_map<(d0, d1, d2, d3) -> ()>,
                             affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
            ins(%accumulator, %bias, %output_scale, %output_zero_point,
                %int8_min, %int8_max
                : tensor<1x?x?x?xi32>, tensor<?xi32>, f32, i32, f32, f32)
            outs(%out_init : tensor<1x?x?x?xi8>) {
          ^bb1(%raw: i32, %channel_bias: i32, %scale: f32, %zero_point: i32,
               %low: f32, %high: f32, %unused: i8):
            %biased = arith.addi %raw, %channel_bias : i32
            %real = arith.sitofp %biased : i32 to f32
            %scaled = arith.mulf %real, %scale : f32
            %rounded = math.roundeven %scaled : f32
            %zero_point_f32 = arith.sitofp %zero_point : i32 to f32
            %offset = arith.addf %rounded, %zero_point_f32 : f32
            %low_clamped = arith.maximumf %offset, %low : f32
            %clamped = arith.minimumf %low_clamped, %high : f32
            %narrowed = arith.fptosi %clamped : f32 to i8
            linalg.yield %narrowed : i8
        } -> tensor<1x?x?x?xi8>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    transform.yield %ins, %outs : !transform.any_value, !transform.any_value
  }

  // The depthwise rewriter. Same shape as the dense one, importing the
  // depthwise executable and shim instead.
  transform.named_sequence @cast_and_call_dynamic_depthwise_conv2d_int8_requant(
      %ins: !transform.any_value {transform.readonly},
      %out: !transform.any_value {transform.readonly}) {
    %root = transform.get_defining_op %out : (!transform.any_value) -> !transform.any_op
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %topology_attr = transform.param.constant #hal.device.topology<links = [
        (@rocket_device -> @cpu_device = {transparent_access = true, unified_memory = true}),
        (@cpu_device -> @rocket_device = {transparent_access = true, unified_memory = true})
      ]> -> !transform.any_param
    transform.annotate %module "stream.topology" = %topology_attr : !transform.any_op, !transform.any_param
    %executable = transform.util.import_symbol @rocket_dynamic_depthwise_int8_requant_executable into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_rocket_dynamic_depthwise_conv2d_int8_requant into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
          transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  // Shared by both requantized matchers. Unlike the other rewriters here it
  // takes the matched DAG's inputs and output rather than the root op: the
  // call replaces a two-op subgraph, so the values are what identify it.
  transform.named_sequence @cast_and_call_dynamic_conv2d_pad1(
      %ins: !transform.any_value {transform.readonly},
      %out: !transform.any_value {transform.readonly}) {
    %root = transform.get_defining_op %out : (!transform.any_value) -> !transform.any_op
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %topology_attr = transform.param.constant #hal.device.topology<links = [
        (@rocket_device -> @cpu_device = {transparent_access = true, unified_memory = true}),
        (@cpu_device -> @rocket_device = {transparent_access = true, unified_memory = true})
      ]> -> !transform.any_param
    transform.annotate %module "stream.topology" = %topology_attr : !transform.any_op, !transform.any_param
    %executable = transform.util.import_symbol @rocket_dynamic_pad1_executable into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_rocket_dynamic_conv2d_pad1 into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
          transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @cast_and_call_dynamic_conv2d_pad1_s2(
      %ins: !transform.any_value {transform.readonly},
      %out: !transform.any_value {transform.readonly}) {
    %root = transform.get_defining_op %out : (!transform.any_value) -> !transform.any_op
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %topology_attr = transform.param.constant #hal.device.topology<links = [
        (@rocket_device -> @cpu_device = {transparent_access = true, unified_memory = true}),
        (@cpu_device -> @rocket_device = {transparent_access = true, unified_memory = true})
      ]> -> !transform.any_param
    transform.annotate %module "stream.topology" = %topology_attr : !transform.any_op, !transform.any_param
    %executable = transform.util.import_symbol @rocket_dynamic_pad1_executable_s2 into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_rocket_dynamic_conv2d_pad1_s2 into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
          transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @cast_and_call_dynamic_conv2d_relu6(
      %ins: !transform.any_value {transform.readonly},
      %out: !transform.any_value {transform.readonly}) {
    %root = transform.get_defining_op %out : (!transform.any_value) -> !transform.any_op
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %topology_attr = transform.param.constant #hal.device.topology<links = [
        (@rocket_device -> @cpu_device = {transparent_access = true, unified_memory = true}),
        (@cpu_device -> @rocket_device = {transparent_access = true, unified_memory = true})
      ]> -> !transform.any_param
    transform.annotate %module "stream.topology" = %topology_attr : !transform.any_op, !transform.any_param
    %executable = transform.util.import_symbol @rocket_dynamic_relu6_executable into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_rocket_dynamic_conv2d_relu6 into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
          transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @cast_and_call_dynamic_depthwise_conv2d_nchw_relu6(
      %ins: !transform.any_value {transform.readonly},
      %out: !transform.any_value {transform.readonly}) {
    %root = transform.get_defining_op %out : (!transform.any_value) -> !transform.any_op
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %topology_attr = transform.param.constant #hal.device.topology<links = [
        (@rocket_device -> @cpu_device = {transparent_access = true, unified_memory = true}),
        (@cpu_device -> @rocket_device = {transparent_access = true, unified_memory = true})
      ]> -> !transform.any_param
    transform.annotate %module "stream.topology" = %topology_attr : !transform.any_op, !transform.any_param
    %executable = transform.util.import_symbol @rocket_dynamic_depthwise_relu6_executable into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_rocket_dynamic_depthwise_conv2d_nchw_relu6 into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
          transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @cast_and_call_dynamic_depthwise_conv2d_nchw_relu6_s2(
      %ins: !transform.any_value {transform.readonly},
      %out: !transform.any_value {transform.readonly}) {
    %root = transform.get_defining_op %out : (!transform.any_value) -> !transform.any_op
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %topology_attr = transform.param.constant #hal.device.topology<links = [
        (@rocket_device -> @cpu_device = {transparent_access = true, unified_memory = true}),
        (@cpu_device -> @rocket_device = {transparent_access = true, unified_memory = true})
      ]> -> !transform.any_param
    transform.annotate %module "stream.topology" = %topology_attr : !transform.any_op, !transform.any_param
    %executable = transform.util.import_symbol @rocket_dynamic_depthwise_relu6_executable_s2 into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_rocket_dynamic_depthwise_conv2d_nchw_relu6_s2 into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
          transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @cast_and_call_dynamic_conv2d_int8_requant(
      %ins: !transform.any_value {transform.readonly},
      %out: !transform.any_value {transform.readonly}) {
    %root = transform.get_defining_op %out : (!transform.any_value) -> !transform.any_op
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %topology_attr = transform.param.constant #hal.device.topology<links = [
        (@rocket_device -> @cpu_device = {transparent_access = true, unified_memory = true}),
        (@cpu_device -> @rocket_device = {transparent_access = true, unified_memory = true})
      ]> -> !transform.any_param
    transform.annotate %module "stream.topology" = %topology_attr : !transform.any_op, !transform.any_param
    %executable = transform.util.import_symbol @rocket_dynamic_int8_requant_executable into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_rocket_dynamic_conv2d_int8_requant into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
          transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @cast_and_call_dynamic_conv2d_int8(%root: !transform.any_op {transform.readonly}) {
    %ins = transform.get_operand %root[all] : (!transform.any_op) -> !transform.any_value
    %out = transform.get_result %root[all] : (!transform.any_op) -> !transform.any_value
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %topology_attr = transform.param.constant #hal.device.topology<links = [
        (@rocket_device -> @cpu_device = {transparent_access = true, unified_memory = true}),
        (@cpu_device -> @rocket_device = {transparent_access = true, unified_memory = true})
      ]> -> !transform.any_param
    transform.annotate %module "stream.topology" = %topology_attr : !transform.any_op, !transform.any_param
    %executable = transform.util.import_symbol @rocket_dynamic_int8_executable into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_rocket_dynamic_conv2d_int8 into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
          transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @cast_and_call_dynamic_depthwise_conv2d_int8(%root: !transform.any_op {transform.readonly}) {
    %ins = transform.get_operand %root[all] : (!transform.any_op) -> !transform.any_value
    %out = transform.get_result %root[all] : (!transform.any_op) -> !transform.any_value
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %topology_attr = transform.param.constant #hal.device.topology<links = [
        (@rocket_device -> @cpu_device = {transparent_access = true, unified_memory = true}),
        (@cpu_device -> @rocket_device = {transparent_access = true, unified_memory = true})
      ]> -> !transform.any_param
    transform.annotate %module "stream.topology" = %topology_attr : !transform.any_op, !transform.any_param
    %executable = transform.util.import_symbol @rocket_dynamic_depthwise_int8_executable into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_rocket_dynamic_depthwise_conv2d_int8 into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
          transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @cast_and_call_dynamic_depthwise_conv2d_int8_s2(%root: !transform.any_op {transform.readonly}) {
    %ins = transform.get_operand %root[all] : (!transform.any_op) -> !transform.any_value
    %out = transform.get_result %root[all] : (!transform.any_op) -> !transform.any_value
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %topology_attr = transform.param.constant #hal.device.topology<links = [
        (@rocket_device -> @cpu_device = {transparent_access = true, unified_memory = true}),
        (@cpu_device -> @rocket_device = {transparent_access = true, unified_memory = true})
      ]> -> !transform.any_param
    transform.annotate %module "stream.topology" = %topology_attr : !transform.any_op, !transform.any_param
    %executable = transform.util.import_symbol @rocket_dynamic_depthwise_int8_executable_s2 into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_rocket_dynamic_depthwise_conv2d_int8_s2 into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
          transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %funcs = transform.structured.match ops{["util.func"]} in %module : (!transform.any_op) -> !transform.any_op
    // An ONNX int8 model (ORT quantize_dynamic -> onnx.ConvInteger, expanded
    // by RocketExpandOnnxConvIntegerPass) arrives as *quantized* linalg
    // convs -- linalg.conv_2d_nchw_fchw_q and
    // linalg.depthwise_conv_2d_nhwc_hwc_q -- carrying scalar i32 zero-point
    // operands. Those two passes turn them into the ordinary named convs
    // every matcher below already understands, and must run before the
    // channels-last pass: on a quantized conv that pass generalizes to
    // linalg.generic, which cannot hold a scalar operand, and the compile
    // dies on a verifier error rather than merely failing to offload. See
    // RocketTransposeQuantizedConvPass.cpp for the full account.
    //
    // What comes out is an i8 x i8 -> i32 convolution plus a separate
    // zero-point correction (conv_q(x, w, xz, 0) == conv(x, w) - xz*sum(w),
    // an exact i32 identity), which is the form the int8 matchers below
    // claim. A fp32 model is unaffected: neither pass matches anything.
    %nhwc_quantized_funcs = transform.apply_registered_pass
        "rocket-transpose-quantized-conv-to-nhwc" to %funcs
      : (!transform.any_op) -> !transform.any_op
    %dequantized_funcs = transform.apply_registered_pass
        "iree-global-opt-quantized-conv-to-conv" to %nhwc_quantized_funcs
      : (!transform.any_op) -> !transform.any_op

    // ONNX commonly imports Conv as NCHW/FCHW, while Rocket's logical
    // convolution ABI and the matchers above use NHWC/HWCF. Normalize
    // before attempting any Rocket specialization: the first pass does the
    // conversion but leaves the op generalized to linalg.generic, so the
    // second re-specializes it back to the named op the matchers look for.
    %channels_last_funcs = transform.apply_registered_pass
        "iree-preprocessing-convert-conv-to-channels-last" to %dequantized_funcs
      : (!transform.any_op) -> !transform.any_op
    %specialized_funcs = transform.apply_registered_pass
        "linalg-specialize-generic-ops" to %channels_last_funcs
      : (!transform.any_op) -> !transform.any_op

    // A transformer's projections import as `linalg.batch_matmul` with a
    // unit batch -- ONNX MatMul over a `[1, tokens, features]` activation --
    // and @match_rocket_matmul only sees `linalg.matmul`. Nothing upstream
    // drops that batch on a *named* contraction (`linalg-fold-unit-extent-dims`
    // and IREE's dispatch-creation variant both leave it alone), so:
    // generalize just the batch matmuls, fold their unit dims through
    // reshapes, and re-specialize, which turns `1x197x768 x 1x768x768` into a
    // `197x768 x 768x768` `linalg.matmul` wrapped in collapse/expand shapes.
    // Convolutions are already named ops again by this point, so the fold
    // patterns do not touch their unit batch.
    //
    // The fold is scoped to functions that actually contain a batch matmul,
    // by walking out from the generalized contractions rather than over
    // every function. It is not a free rewrite for its neighbours: the
    // patterns also reach elementwise generics, and collapsing one of those
    // puts a `tensor.collapse_shape` between a convolution and its epilogue.
    // `@match_dynamic_conv2d_int8_requant` is a DAG match rooted at that
    // epilogue, so a reshape in between makes it decline -- measured, not
    // predicted: with the fold applied to every function the requantized
    // matcher claims nothing, and `rocket_int8_requant_match.mlir` fails.
    // Scoping it keeps ViT's collapse and leaves a quantized convolution
    // graph, which has no batch matmul at all, exactly as it was.
    transform.foreach %specialized_funcs : !transform.any_op {
      ^bb0(%func: !transform.any_op):
        %batch_matmuls = transform.structured.match ops{["linalg.batch_matmul"]}
            in %func : (!transform.any_op) -> !transform.any_op
        %generalized_batch_matmuls = transform.structured.generalize %batch_matmuls
          : (!transform.any_op) -> !transform.any_op
        // Empty when this function has no batch matmul, which is what scopes
        // the fold: `apply_patterns` over an empty handle rewrites nothing.
        // `deduplicate` keeps a function with several batch matmuls -- every
        // transformer block has one -- from being rewritten once per match.
        %owning_funcs = transform.get_parent_op %generalized_batch_matmuls
            {deduplicate, op_name = "util.func"}
          : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %owning_funcs {
          transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
        } : !transform.any_op
    }
    %canonical_funcs = transform.apply_registered_pass
        "linalg-specialize-generic-ops" to %specialized_funcs
      : (!transform.any_op) -> !transform.any_op
    // A GEMV is a matmul with one extent pinned to 1, and everything below
    // this point already handles a matmul with a unit extent -- the demotion
    // right after, @match_rocket_matmul (whose dim_bounds start at umin = 1),
    // @call_rocket_matmul and #rocket_matmul_target. So raise
    // linalg.matvec/vecmat into linalg.matmul rather than teaching each of
    // those about two more ops: the vector operand and the accumulator each
    // gain a unit dimension via tensor.expand_shape, and a collapse_shape
    // puts the result back to rank one. Both are pure metadata.
    //
    // This is the same move the batch-matmul fold above makes, run backwards:
    // there a degenerate dimension is removed so the matmul matcher can see
    // what is really there, here one is added for the same reason.
    //
    // linalg.dot is deliberately untouched -- see the pass -- because it
    // reduces to a scalar, and a dispatch plus a weight pack plus an output
    // compaction to produce one number is not a trade worth making.
    // Collapses an ONNX QLinearConv's five-op requantization epilogue into the
    // single generic @match_dynamic_conv2d_int8_requant is written against,
    // so the requantized path is reachable from a real model rather than only
    // from a hand-written canonical form. Has to run after
    // `iree-global-opt-quantized-conv-to-conv` (which is what produces the
    // chain) and after the channels-last conversion (it matches
    // `linalg.conv_2d_nhwc_hwcf`, and an NCHW convolution is not that op), and
    // before the match loops that claim convolutions.
    %requant_fused_funcs = transform.apply_registered_pass
        "rocket-fuse-int8-requant-epilogue" to %canonical_funcs
      : (!transform.any_op) -> !transform.any_op
    %gemv_funcs = transform.apply_registered_pass
        "rocket-expand-gemv-to-matmul" to %requant_fused_funcs
      : (!transform.any_op) -> !transform.any_op

    // Rocket's ABI is f16-in/f32-accumulate (see call_rocket_dynamic_conv2d
    // above), but models commonly arrive as plain f32 (e.g. ONNX/torch
    // import, no fp16 casting anywhere). Demote the conv and matmul operands
    // to f16, leaving the accumulator at f32, so @match_dynamic_conv2d's
    // f16/f16/f32 typing requirement matches these too. An op already
    // authored in f16 is left alone: the pass only rewrites all-f32 operand
    // sets.
    //
    // This is the plugin's own pass, not upstream's
    // "iree-global-opt-demote-contraction-inputs" that it used to call:
    // that one rebuilds the named op through
    // linalg::getPrunedAttributeList, which erases `strides` and
    // `dilations`, silently turning every strided convolution into a
    // stride-1 one. See RocketDemoteConvInputsPass.cpp -- it handles exactly
    // the same convolution set, so that part is a behaviour-preserving swap
    // apart from keeping those two attributes. linalg.matmul was added to it
    // on 2026-09-04 and has no counterpart upstream.
    %demoted_funcs = transform.apply_registered_pass
        "rocket-demote-conv-inputs-to-f16" to %gemv_funcs
      : (!transform.any_op) -> !transform.any_op

    // Tripwire for the above and anything like it: errors if any named
    // convolution's output extent disagrees with its own input/filter/
    // stride/dilation. Runs while padding is still explicit and nothing has
    // been tiled, so the relation is exact here. Never fires on a healthy
    // compile.
    %verified_funcs = transform.apply_registered_pass
        "rocket-verify-conv-shapes" to %demoted_funcs
      : (!transform.any_op) -> !transform.any_op

    // Puts an fp16 convolution and the ReLU6 after it into the two-op form
    // @match_dynamic_conv2d_relu6 claims: bias lifted out of the init and
    // into the epilogue generic (which is where the hardware computes it,
    // on the BS plane), bounds as scalar operands, and the channels-last
    // reshape moved out of the way. Like the requantized path below, this
    // has to run before `rocket-annotate-original-placement` -- the
    // `rocket.origin` tags that pass adds would make every convolution fail
    // the DAG match, which compares whole attribute dictionaries.
    %activated_funcs = transform.apply_registered_pass
        "rocket-fuse-conv-relu6" to %verified_funcs
      : (!transform.any_op) -> !transform.any_op

    // Tags every conv-family linalg op with rocket.origin/rocket.origin_kind
    // right before the match/rewrite loop below claims (and erases) some of
    // them -- see RocketAnnotateOriginalPlacementPass.cpp. A
    // --compile-to=preprocessing dump then shows exactly which conv-shaped
    // ops fell through to CPU: matched ops lose the tag along with the rest
    // of the op they were erased from.
    // Requantized int8 runs as its own pass over each function, ahead of
    // everything else, because it is the only matcher here that claims more
    // than the convolution op itself. Its root is the requantization generic,
    // which `foreach_match`'s walk reaches *after* the convolution -- so
    // sharing one loop with the int8_accumulator matchers would let those
    // claim the convolution first and leave the epilogue behind, whatever
    // order they are listed in. Rewriting the whole subgraph first leaves
    // nothing for them to match.
    //
    // It also has to run before `rocket-annotate-original-placement`, not
    // after like every other matcher: `transform.iree.match.cast_compatible_dag_from_root`
    // compares whole attribute dictionaries, so the `rocket.origin` tags that
    // pass adds would make every convolution fail to match the DAG template.
    // Nothing is lost by claiming these first -- an op the loop rewrites is
    // erased, which is exactly what that annotation's readers already expect
    // of a claimed convolution.
    // Records, on any convolution whose input is a symmetric, zero-filled,
    // spatial-only `tensor.pad`, what that pad was -- so the matchers below
    // can claim the pair and let the CNA do the padding.
    //
    // **This must be the last pass before the match loop.** It sinks the pad's
    // fill constant into the pad's region, because a region yielding a value
    // defined outside itself can never match a DAG template -- and *every*
    // greedy pattern driver hoists such a constant straight back out, since
    // `tensor.pad`'s region is not isolated-from-above. Running this before
    // `rocket-fuse-conv-relu6` undoes it: measured on ResNet50, 17 pads sunk
    // and 0 still sunk by the time the loop ran.
    //
    // Like the other subgraph-claiming matchers it also has to run before
    // `rocket-annotate-original-placement`, whose `rocket.origin` tags would
    // make every convolution fail the DAG match.
    %padded_funcs = transform.apply_registered_pass
        "rocket-fold-conv-pad" to %activated_funcs
      : (!transform.any_op) -> !transform.any_op

    %requantized_funcs = transform.foreach %padded_funcs : !transform.any_op -> !transform.any_op {
      ^bb0(%requant_func: !transform.any_op):
        %matched_func = transform.foreach_match in %requant_func
            @match_dynamic_conv2d_3x3_pad1 -> @cast_and_call_dynamic_conv2d_pad1,
            @match_dynamic_conv2d_3x3_pad1_demoted -> @cast_and_call_dynamic_conv2d_pad1,
            @match_dynamic_conv2d_3x3_pad1_s2 -> @cast_and_call_dynamic_conv2d_pad1_s2,
            @match_dynamic_conv2d_3x3_pad1_demoted_s2 -> @cast_and_call_dynamic_conv2d_pad1_s2,
            @match_dynamic_conv2d_relu6 -> @cast_and_call_dynamic_conv2d_relu6,
            @match_dynamic_conv2d_3x3_relu6 -> @cast_and_call_dynamic_conv2d_relu6,
            @match_dynamic_depthwise_conv2d_nchw_relu6 -> @cast_and_call_dynamic_depthwise_conv2d_nchw_relu6,
            @match_dynamic_depthwise_conv2d_nchw_relu6_s2 -> @cast_and_call_dynamic_depthwise_conv2d_nchw_relu6_s2,
            @match_dynamic_conv2d_int8_requant -> @cast_and_call_dynamic_conv2d_int8_requant,
            @match_dynamic_conv2d_3x3_int8_requant -> @cast_and_call_dynamic_conv2d_int8_requant,
            @match_dynamic_depthwise_conv2d_int8_requant -> @cast_and_call_dynamic_depthwise_conv2d_int8_requant,
            @match_dynamic_depthwise_conv2d_3x3_int8_requant -> @cast_and_call_dynamic_depthwise_conv2d_int8_requant
          : (!transform.any_op) -> (!transform.any_op)
        // `cast_and_call` rewires uses; it does not erase what it replaced.
        // Without this the convolution and its epilogue are still standing
        // when the main loop runs, and the int8_accumulator matcher claims
        // the dead convolution -- emitting a second, redundant NPU dispatch
        // whose result nothing reads.
        transform.apply_dce to %matched_func : !transform.any_op
        transform.yield %matched_func : !transform.any_op
    }

    %annotated_funcs = transform.apply_registered_pass
        "rocket-annotate-original-placement" to %requantized_funcs
      : (!transform.any_op) -> !transform.any_op

    transform.foreach %annotated_funcs : !transform.any_op {
      ^bb1(%func: !transform.any_op):

        // The stride-2 dense matchers below were disabled for a long time
        // because wiring them in broke the compile: they claim MobileNetV2's
        // stem conv (the only dense conv it runs at stride > 1), and
        // iree-compile then failed to serialize main_graph$async_dispatch_0,
        // the model's own input cast+transpose, because IREE's affinity
        // analysis pulled that CPU dispatch onto @rocket_device along with
        // its only consumer. That is fixed generally, not specially:
        // rocket-pin-unclaimed-dispatches pins every dispatch this spec did
        // not claim to the CPU (see RocketPinUnclaimedDispatchesPass.cpp).
        //
        // Turning them back on then exposed three real defects that the
        // disabling had been hiding, all since fixed and all now covered by
        // hardware regressions in conv2d_oracle_hw.rs:
        //
        //   * strides were being erased outright before matching, so a
        //     stride-2 conv was dispatched as stride 1 (see
        //     RocketDemoteConvInputsPass.cpp);
        //   * dense conv at stride > 1 was wrong whenever
        //     `(extent - kernel) % stride != 0` (ColumnTile::from_output_range);
        //   * the stem's Cin=3 feature buffer was never synced for device
        //     (rocket-hal-driver's command_buffer.rs).
        //
        // What is left is a genuine precision tradeoff, not a bug. Rocket's
        // ABI is f16-in/f32-accumulate, so a conv on the NPU runs its inputs
        // at half precision. For MobileNetV2's f32 stem that costs about
        // 0.35 max|err| on the final logits -- the stem feeds an int8
        // quantization step, and f16-level noise there crosses quantization
        // boundaries and propagates. Keeping the stem on the CPU instead
        // costs one offloaded dispatch out of 18 and buys back exact f32
        // (7.2e-07 against a plain f32 build). The isolated stem convolution
        // itself is correct on hardware to f16 epsilon, so this is the cost
        // of f16, not of the NPU being wrong.
        //
        // The three element-wise entries below are commented out on purpose,
        // and this marker is load-bearing: `rocket-compiler --elementwise`
        // uncomments exactly the lines carrying it, and nothing else does.
        // ROADMAP Phase 1 requires these matchers to land behind a flag
        // rather than in the default list, because ISSUES.md P8 measured that
        // at the current per-dispatch cost more offload sites make a model
        // slower -- an element-wise op does less arithmetic than its own
        // dispatch tax. Shipping them on by default would regress every
        // model P8 measured.
        //
        // They are written here, next to the entries they would join, rather
        // than injected from Rust, so that the enabled and disabled specs
        // differ by three characters per line and the list stays readable and
        // maintainable in one place.
        transform.foreach_match in %func
//@ROCKET_ELEMENTWISE@            @match_elementwise_add_f32 -> @cast_and_call_elementwise_add,
//@ROCKET_ELEMENTWISE@            @match_elementwise_sub_f32 -> @cast_and_call_elementwise_sub,
//@ROCKET_ELEMENTWISE@            @match_elementwise_mul_f32 -> @cast_and_call_elementwise_mul,
            @match_pooling_nchw_sum_avg -> @cast_and_call_pooling_avg_nchw,
            @match_pooling_nhwc_sum_avg -> @cast_and_call_pooling_avg_nhwc,
            @match_pooling_nhwc_max -> @cast_and_call_pooling_max_nhwc,
            @match_pooling_nhwc_max_s2 -> @cast_and_call_pooling_max_nhwc_s2,
            @match_pooling_nchw_max -> @cast_and_call_pooling_max_nchw,
            @match_pooling_nchw_max_s2 -> @cast_and_call_pooling_max_nchw_s2,
            @match_pooling_nhwc_min -> @cast_and_call_pooling_min_nhwc,
            @match_pooling_nhwc_min_s2 -> @cast_and_call_pooling_min_nhwc_s2,
            @match_rocket_matmul -> @cast_and_call_rocket_matmul,
            @match_dynamic_conv2d -> @cast_and_call_dynamic_conv2d,
            @match_dynamic_conv2d_3x3 -> @cast_and_call_dynamic_conv2d,
            @match_dynamic_conv2d_s2 -> @cast_and_call_dynamic_conv2d_s2,
            @match_dynamic_conv2d_3x3_s2 -> @cast_and_call_dynamic_conv2d_s2,
            @match_dynamic_depthwise_conv2d -> @cast_and_call_dynamic_depthwise_conv2d,
            @match_dynamic_depthwise_conv2d_3x3 -> @cast_and_call_dynamic_depthwise_conv2d,
            @match_dynamic_depthwise_conv2d_nchw -> @cast_and_call_dynamic_depthwise_conv2d_nchw,
            @match_dynamic_depthwise_conv2d_nchw_3x3 -> @cast_and_call_dynamic_depthwise_conv2d_nchw,
            @match_dynamic_depthwise_conv2d_nchw_s2 -> @cast_and_call_dynamic_depthwise_conv2d_nchw_s2,
            @match_dynamic_depthwise_conv2d_nchw_3x3_s2 -> @cast_and_call_dynamic_depthwise_conv2d_nchw_s2,
            @match_dynamic_depthwise_conv2d_nchw_s3 -> @cast_and_call_dynamic_depthwise_conv2d_nchw_s3,
            @match_dynamic_depthwise_conv2d_nchw_3x3_s3 -> @cast_and_call_dynamic_depthwise_conv2d_nchw_s3,
            @match_dynamic_depthwise_conv2d_nchw_s4 -> @cast_and_call_dynamic_depthwise_conv2d_nchw_s4,
            @match_dynamic_depthwise_conv2d_nchw_3x3_s4 -> @cast_and_call_dynamic_depthwise_conv2d_nchw_s4,
            @match_dynamic_conv2d_int8 -> @cast_and_call_dynamic_conv2d_int8,
            @match_dynamic_conv2d_3x3_int8 -> @cast_and_call_dynamic_conv2d_int8,
            @match_dynamic_depthwise_conv2d_int8 -> @cast_and_call_dynamic_depthwise_conv2d_int8,
            @match_dynamic_depthwise_conv2d_3x3_int8 -> @cast_and_call_dynamic_depthwise_conv2d_int8,
            @match_dynamic_depthwise_conv2d_int8_s2 -> @cast_and_call_dynamic_depthwise_conv2d_int8_s2,
            @match_dynamic_depthwise_conv2d_3x3_int8_s2 -> @cast_and_call_dynamic_depthwise_conv2d_int8_s2
          : (!transform.any_op) -> (!transform.any_op)
    }

    // Every convolution still standing here is one the loop above declined,
    // so the f16 demotion it was given for matching bought it nothing --
    // put its f32 inputs back rather than make the CPU run it in half
    // precision. Only this project's own demotion is reverted; see
    // RocketPromoteUnclaimedConvInputsPass.cpp. On MobileNetV2 this is the
    // stride-2 stem, worth 0.349 max|err| on the final logits.
    //
    // The dead truncf generics this leaves behind are what apply_dce below
    // is already there to remove.
    // Applied to the module, not to %annotated_funcs: the foreach above
    // consumes that handle, and re-matching just to hand the pass a
    // function-shaped handle would buy nothing -- the pass walks whatever it
    // is given.
    %promoted_module = transform.apply_registered_pass
        "rocket-promote-unclaimed-conv-inputs" to %module
      : (!transform.any_op) -> !transform.any_op

    // Inline the @call_rocket_* wrappers into their callers.
    //
    // `transform.util.cast_and_call` leaves a call, and IREE never inlines
    // these functions on its own -- ISSUES.md P6 item 2 found that the hard
    // way, when a `truncf` written inside @call_rocket_matmul ran as a
    // per-inference CPU dispatch instead of folding into an initializer.
    // Everything a wrapper does around its `flow.dispatch` has that problem:
    // the int8 epilogue add above, the HWC->CHW depthwise filter transpose
    // (13 dispatches per inference on MobileNetV2 int8, over constant
    // weights), and the zero-bias fills. Inlined, const-eval hoists the
    // constant ones into initializers and dispatch-region formation fuses
    // the rest into the neighbouring elementwise chain; the `flow.dispatch`
    // itself is unaffected, it keeps its @rocket_device affinity.
    %inlined_module = transform.apply_registered_pass
        "inline" to %promoted_module
      : (!transform.any_op) -> !transform.any_op

    transform.apply_dce to %inlined_module : !transform.any_op
    transform.yield
  }
}
