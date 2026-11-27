# Bring-your-own quantization, measured (2026-11-27)

The question: if a user quantizes a model themselves with stock ONNX Runtime
settings and hands it to `rocket-compiler`, what do they get, compared with
the same fp32 model run through `tools/quantize_onnx_int8.py`?

## Setup

- Sources: torchvision MobileNetV2 and ResNet50, fp32, opset 17
  (`survey/mobilenet_v2/mobilenet_v2.f32.onnx`,
  `survey/2026-11/resnet50/resnet50.f32.onnx`).
- Every variant gets the same batch pin, `quant_pre_process`, and the same 32
  synthetic calibration batches (`synthetic_batches`, mean 0.45, std 0.225).
  Only the `quantize_static` arguments differ.
- Variants:
  - `ours` is `tools/quantize_onnx_int8.py` with its defaults: QOperator,
    per-tensor, U8S8, `Conv Relu`.
  - `ours_skip` adds `--quantize-residual-skip` (ResNet50 only).
  - `ort_qdq` is `quantize_static` with every default: QDQ, per-tensor,
    S8S8, all op types.
  - `ort_qdq_pc` is the same with `per_channel=True`.
  - `ort_qop` is QOperator with the other defaults.
  - `ort_qop_pc` is QOperator with `per_channel=True`.
- Pipeline: `tools/import_onnx.py`, then `rocket-compiler compile` (main at
  `ebeb967`, IREE 3.14) for an NPU arm and a `--no-offload` arm.
- Board: orangepi on 7.3.6 with the patched `rocket.ko`, NPU at 600 MHz,
  `--task_topology_cpu_ids=4,5,6,7`.
  - **The CPU governor was `ondemand`**, not the `performance` the surveys
    used. Changing it needs sudo. That inflates every NPU arm (see
    `planck-measurement-environment`), so compare arms within this table
    only, not against earlier surveys.
  - Timing: two interleaved passes, median of 6 repetitions each.
- Artifacts: `survey/quant-byo-2026-11-27/` in the main checkout.

## Results

| model | variant | compiles | NPU sites / CPU sites | conv precision on NPU | NPU ms (p1/p2) | own `--no-offload` ms |
|---|---|---|---|---|---|---|
| MobileNetV2 | ours | yes | 54 / 91 | int8 | 50.6 / 47.7 | 50.3 / 51.7 |
| MobileNetV2 | ort_qdq | yes | 54 / 127 | **fp16** | 81.5 / 79.3 | 53.9 / 55.1 |
| MobileNetV2 | ort_qdq_pc | yes | 54 / 127 | **fp16** | 82.3 / 79.8 | 56.4 / 55.6 |
| MobileNetV2 | ort_qop, ort_qop_pc | **no** (both arms) | | | | |
| MobileNetV2 | fp16 build, for scale | | | fp16 | 16.2 | |
| ResNet50 | ours | yes | 56 / 31 | int8 | 65.2 / 65.6 | 481 / 488 |
| ResNet50 | ours_skip | yes | 56 / 23 | int8 | 69.6 / 69.4 | 493 / 480 |
| ResNet50 | ort_qdq | yes | 56 / 120 | **fp16** | 239 / 250 | 1565 / 1561 |
| ResNet50 | ort_qdq_pc | yes | 56 / 120 | **fp16** | 249 / 244 | 1586 / 1572 |
| ResNet50 | ort_qop, ort_qop_pc | **no** (both arms) | | | | |
| ResNet50 | fp16 build, for scale | | | fp16 | 77.3 | |

Every module that compiled ran, and its top-1 agreed with onnxruntime on the
quantized model. Synthetic calibration and a synthetic input mean that says
nothing about accuracy.

## What it shows

1. **Stock QOperator does not compile.** ORT's QOperator quantizer emits
   `com.microsoft` `QLinearAdd`, `QLinearGlobalAveragePool` and `QGemm` for
   the non-conv ops. The `QLinearAdd` (on si8 activations, ORT's default
   QInt8) fails torch-mlir legalization on both arms. QOperator is the form
   the Rocket matchers want, and our tool only reaches it by quantizing
   nothing but `Conv` and `Relu`.

2. **Stock QDQ, ORT's default, compiles and offloads every conv, at fp16.**
   IREE has no pass that turns `DQ -> conv -> Q` into an integer convolution.
   So the weights dequantize at compile time and the spec claims an f16
   conv. Every activation edge keeps its fake quantization on the CPU:
   - each conv's fp16 output is quantized to i8 (`..._f16xi8`) and
     dequantized straight back (`..._i8xf16`);
   - that adds 110 elementwise/transpose dispatches on ResNet50 and breaks
     every NPU-to-NPU edge.

   The cost:
   - MobileNetV2's NPU arm is 1.45x **slower than its own CPU build** and
     4.9x slower than the plain fp16 build.
   - ResNet50 is 3.7x slower than `ours`, and 3.1x slower than not
     quantizing at all.

3. **Per-channel changed nothing**, because on the QDQ path the weights are
   dequantized to f16 anyway. The torch-mlir per-channel fallback that
   `quantize_onnx_int8.py` guards against only bites on the QOperator path,
   which does not compile here for an earlier reason.

4. **Our tool is what makes int8 pay.** ResNet50 `ours` runs 7.4x faster than
   its own CPU arm. MobileNetV2 `ours` is only level with its CPU arm, and
   3x slower than fp16. The tool leaves ReLU6 (`Clip`) in float, so every
   MobileNetV2 edge carries a CPU requant. Stock ORT folds `Clip` by itself.
   (**Since fixed**: `Clip` is in the default now, and MobileNetV2 runs
   30.0 -> 21.0 ms. See *Folding ReLU6* below.)

5. **Regression found along the way: the int8 residual add no longer reaches
   the NPU.** `ours_skip` gains nothing over `ours`. Recompiling the very
   model that measured 50.6 -> 27.5 ms on 2026-11-24
   (`survey/relufuse/resnet50-224.int8s.mlir`) with today's compiler gives
   the same module. There is no `rocket_elementwise_add_int8_relu`
   executable, and the add runs as a CPU
   `elementwise_transpose_256x3136_i8` over two i8 inputs.
   - **Cause.** The IREE 3.14 bump (#100) changed how DequantizeLinear and
     QuantizeLinear lower: they now arrive as `iree_linalg_ext.dequantize_affine`
     / `quantize_affine`. `rocket-fuse-int8-residual-add` and
     `rocket-fuse-int8-max-pool` only knew the old `linalg.generic` form.
     The bump's int8 check compared the board set, which has no int8 skip.
     It is break 6 in `agent-docs/iree-3.14-bump.md`.
   - **Fixed** in the same change as this document: both passes accept the
     affine ops, and the lit tests gain cases in the 3.14 form.
   - **Re-measured on the board**, governor `performance` this time, so not
     comparable with the table above:
     - `ours_skip`: 41.1 / 43.8 ms -> **17.6 / 17.2 ms**, with 15 of 16
       residual adds on the NPU and CPU dispatches 23 -> 5. Top-5 equals
       onnxruntime's; max|err| against it is 0.115 (was 0.132).
     - `ours`: 40.3 / 40.9 -> 36.0 / 36.1 ms, from the int8 stem pool.
       Its output is bit-identical to the unfixed build.
     - The Nov-24 model recompiled: 18.9 / 17.8 ms.
     - The `--no-offload` arm is unchanged (1e-7). 0 NPU timeouts.

## Folding ReLU6 (`Clip`) into the conv (2026-11-27, later)

`tools/quantize_onnx_int8.py` now quantizes `Conv Relu Clip` by default, so
ORT folds MobileNetV2's 35 ReLU6s into the convolutions in front of them,
just as it already folded `Relu`.

- **How the fold works.** ORT deletes the `Clip` and gives the conv the
  `Clip`'s calibrated output range. Each of the 35 folded convs gets output
  zero point 0, and the top of its range is at most 6.0 (4.04 to 6.0 across
  them). So uint8 saturation does the clamp.
- **The verify step still passes.** All 52 `QLinearConv`s are per-tensor
  with zero weight zero points.
- **The graph after the fold** has 11 `QuantizeLinear` and 16
  `DequantizeLinear`, down from 45 and 50, and no `Clip`.
- **ResNet50 is unaffected.** Quantized with the new default, it is
  byte-identical to the `ours` model above.

**A guard came with it.** ORT's fold is two pieces of code that do not
check each other:

- `QLinearActivation` deletes *every* Relu or Clip that follows a quantized
  tensor.
- `adjust_tensor_ranges` is what makes the clamp real, and it runs only when
  the conv's output has exactly one consumer.
- A range always includes 0, so a `Clip` whose bounds exclude 0 also loses
  that bound.

In both cases the activation is silently gone. `unfoldable_activations`
leaves such nodes in float. On a three-conv toy (`Clip(1,6)` plus a Relu
whose input also feeds an Add), ORT alone dropped both clamps, max|err|
5.33 against float. With the guard, both stay and the error is 0.13.
MobileNetV2 and ResNet50 have no such node.

**Setup**

- Same source, batch pin and calibration as the table above: 32 synthetic
  batches, mean 0.45, std 0.225.
- Compiler: main at `7afc3cd`, using the `libIREECompiler.so` built for #116.
  Its plugin, `rocket-core` and `rocket-plan-ffi` sources are identical to
  main's.
- Board:
  - governor `performance`, NPU IRQs on cpu6, NPU at 600 MHz;
  - `--device=rocket --device=local-task --task_topology_cpu_ids=4,5,6,7`;
  - two interleaved passes, median of 6 repetitions each.
- **Do not compare these timings with the first table.** That table was
  measured under `ondemand`.
- Artifacts: `survey/quant-clip-2026-11-27/` in the main checkout.

| MobileNetV2 int8 | NPU sites / CPU sites | NPU->NPU edges | NPU ms (p1/p2) | own `--no-offload` ms (p1/p2) | top-5 vs ORT | NPU max\|err\| vs ORT |
|---|---|---|---|---|---|---|
| `Conv Relu` (old default) | 54 / 91 | 2 | 30.1 / 29.9 | 44.8 / 44.5 | 4 of 5 | 0.093 |
| `Conv Relu Clip` (new default) | 54 / 78 | 15 | **21.0 / 21.0** | 42.0 / 42.6 | 5 of 5 | 0.050 |
| fp16 build, for scale | | | 9.8 / 9.4 | | | |

- **Speed.** The fold is 1.43x faster than the old default, and 2.0x faster
  than its own CPU build (it was 1.5x).
- **Correctness.**
  - Both `--no-offload` arms match onnxruntime on their own quantized
    model exactly (max|err| 0.0000).
  - The new NPU arm agrees with onnxruntime's top-5 in order.
  - Its max\|err\| halves, because one rounding is gone from every edge.
- **The input is synthetic.** Calibration and input are both random, so
  none of this says anything about accuracy on real images.

**int8 is still 2.2x slower than fp16, and the rest is compiler work.**
The fold made 13 of the 17 depthwise -> project edges NPU->NPU (the other
2 of the 15 edges were already there). The remaining edges still run
through the CPU:

1. **The int8 depthwise pad is not folded.** Each depthwise 3x3 input
   still goes through a CPU transpose and pad (`transpose_*` plus
   `slow_memcpy`), so no expand -> depthwise edge chains.
   - fp16 has this fold (`#rocket_dynamic_depthwise_pad1_relu6_target`,
     memory `depthwise-pad-fold`: 30.8 -> 13.5 ms).
   - The int8 requantized depthwise path has no pad-1 target.
   - This is the biggest remaining item.
2. **The four stride-2 depthwise convs take the int8 accumulator path**
   (`rocket_dynamic_depthwise_int8_executable_s2`), so their requantization
   is a CPU `elementwise_*_i32xi32xi8`. There is no requantized stride-2
   depthwise target.
3. **The 10 residual adds stay float.** `--quantize-residual-skip` has not
   been tried on MobileNetV2.

(**Items 1 and 2 are fixed**: MobileNetV2 int8 runs 21.0 -> 9.1 ms, a
little faster than fp16. See *Folding the int8 depthwise pad* below.
Item 3 is now nearly all that is left on the CPU.)

## Folding the int8 depthwise pad (2026-11-27, later still)

Items 1 and 2 above, done together. Every int8 depthwise 3x3 now takes its
pad on the CNA and its requantization on the DPU, so each expand ->
depthwise -> project chain runs NPU to NPU.

**Why the pad was invisible.** The importer emits a grouped QLinearConv as
an NHWC `linalg.depthwise_conv_2d_nhwc_hwc_q`, but pads it in NCHW, in
front of its own NCHW->NHWC transpose:
`depthwise_q(transpose[0,2,3,1](pad_nchw(x, -128)))`. The transpose sits
between the pad and the conv, so `rocket-fold-conv-pad` never saw the pad.

**What changed**, modelled on the dense int8 pad1 work (#64):

1. `rocket-transpose-quantized-conv-to-nhwc` gains
   `SinkPadBelowQuantizedDepthwiseTranspose`. It rebuilds the chain as
   `pad_nhwc(transpose(x))`. A constant pad commutes with a permutation,
   so the tensor is the same. The transpose then sits next to the
   producer's own and the pair cancels.
2. `rocket-fold-conv-pad` marks a -128 pad in front of an NHWC depthwise
   3x3 that feeds the requant epilogue, at stride 1 or 2 and no dilation.
   Before, only the dense 3x3 could take an int8 mark.
3. The spec has `#rocket_dynamic_depthwise_int8_requant_pad1_target` and
   `_pad1_s2_target`. Both bake `input_zero_point = -128` and
   `pad_top/pad_left = 1`. The matchers are
   `@match_dynamic_depthwise_conv2d_3x3_pad1_int8_requant{,_s2}`. The
   stride-2 one is also the first requantized stride-2 depthwise, so those
   four layers leave the accumulator path and its CPU requant.
4. **A driver fix the hardware forced.** See below.

**The int8 depthwise CNA reads `pad_value` per byte lane.**

- **Symptom.** The first board run of the stride-1 fixture had its
  interior bit-exact, and 14,595 of 451,584 elements wrong. Every wrong
  element was a border pixel of an *odd* channel.
- **What the odd channels see.** Fitting the fill they actually got gives
  -1 (99.96% exact). That is byte 1 of the programmed `0xFFFFFF80`.
  Channel `2k` takes byte 0 and channel `2k + 1` takes byte 1.
- **Dense is different.** The dense pad1 fixture is exact with the same
  register value, so dense reads byte 0 for every channel.
- **The fix.** `cna_pad_value` (`iree-rocket-hal/src/rocket/conv.rs`)
  copies byte 0 into byte 1 for depthwise only, so -128 is written as
  `0xFFFF8080`. Zero points whose two low bytes already agree (0, -1) are
  unchanged.
- **Mesa and rocket-userspace** write the dense encoding for depthwise too.
  No earlier test here ran an int8 depthwise padded with a nonzero fill.

**Setup**

- Same model, input and board conditions as *Folding ReLU6* above:
  `survey/quant-clip-2026-11-27/imp/mnv2.clip.mlir`, governor
  `performance`, NPU IRQs on cpu6, 600 MHz,
  `--device=rocket --device=local-task --task_topology_cpu_ids=4,5,6,7`.
- `iree-benchmark-module --benchmark_repetitions=6`, median, two
  interleaved passes.
- **Compiler.** The #116 `libIREECompiler.so`, relinked with this
  change's two plugin objects. They were compiled with that build's own
  commands. Its other plugin, `rocket-core` and `rocket-plan-ffi` sources
  are identical to this branch's base. Built this way, ResNet50 int8
  compiles byte-identically to the unmodified lib.
- **Runtime.** `iree-run-module` and `iree-benchmark-module` are built from
  this branch, for the driver fix. Arm A gives the same output under the
  old and new runtimes.

| MobileNetV2 int8 (`Conv Relu Clip`) | NPU / CPU sites | NPU->NPU edges | ms (p1/p2) | vs ORT max\|err\| | top-5 vs ORT |
|---|---|---|---|---|---|
| A: before (main) | 54 / 78 | 15 | 20.8 / 20.6 | 0.050 | 5 of 5 |
| B: stride-1 fold only | 54 / 31 | 28 | 14.1 / 14.2 | 0.050 | 5 of 5 |
| C: stride 1 + stride 2 (this change) | 54 / 23 | 36 | **9.07 / 9.20** | 0.069 | 5 of 5 |
| own `--no-offload`, before | 0 / 64 | | 42.0 / 42.0 | 0.0000 | |
| own `--no-offload`, after | 0 / 64 | | 30.8 / 30.7 | 0.0000 | |
| fp16 build, for scale | | | 9.56 / 9.48 | | |

- **Correctness.**
  - **B is bit-identical to A** on the board. That is the check the dense
    pad1 work used: the same NPU executables, with the pad moved from a
    CPU copy to the CNA.
  - C differs from A because the stride-2 requantization moved from the
    CPU (ties to even) to the DPU (ties away from zero; memory
    `int8-tie-rule-half-away`). Its max\|err\| against onnxruntime is
    0.069, against A's 0.050, and its top-5 is the same and in order.
  - The two `--no-offload` arms are bit-identical to each other, and both
    match onnxruntime exactly.
- **Where the CPU sites went.**
  - A's 78 were 34 transposes, 17 pad copies, 17 i32 requants and 10
    others.
  - B removes the 34 transposes and 13 of the 17 pad copies.
  - C removes the last 4 pad copies and the 4 stride-2 CPU requants.
- **The CPU arm got faster too, 42.0 -> 30.8 ms.** The pad move makes
  the CPU build skip a transpose round trip per depthwise layer. Against
  that new baseline, the NPU build is 3.4x faster (it was 2.0x).
- **Gates.**
  - Lit: 56 of 56, including the new `rocket_int8_depthwise_requant_pad1.mlir`.
    That test fails on the unmodified lib.
  - Four e2e fixtures on the board, all exact to atol 1 (the tie):
    `requant_int8_dw3x3_pad1`, `requant_int8_dw3x3_pad1_s2`,
    `requant_int8_1x1_then_dw_pad1` (the expand -> depthwise edge), and the
    dense `requant_int8_3x3_pad1` control.
  - The whole compiled e2e gate (`tools/e2e_conv_regression.py
    --skip-raw`, 57 differentials) passes on the board with the fixed
    runtime.
- **What is left on the CPU.** 19 of the 23 sites belong to the 10 float
  residual adds (item 3). The rest are the input quantize and three head
  ops.
- **Artifacts.** `survey/dw-pad1-2026-11-27/` in the main checkout, and
  board `~/dw-pad1`. A to D are the arms in the table; `Dold` is the old
  `--no-offload` arm. `e2e/` holds the failing stride-1 run that exposed
  the byte-lane fill, and `relink.py` is the relink recipe.

## Implications for scope

- **"Bring your own pre-quantized ONNX" does not work today with either
  format ORT offers.** QOperator fails to build. QDQ builds, then runs
  slower than fp16, and on MobileNetV2 slower than the CPU.
- **The highest-value compiler item is a QDQ -> requantized-conv
  rewrite**: `Q(conv(DQ(x), DQ(w)) + b)` into the canonical form
  `rocket-fuse-int8-requant-epilogue` already produces. QDQ is ORT's default
  and the common interchange format. Per-channel weight scales come right
  behind: the BS plane can carry them (`iree-rocket-hal/src/rocket/conv.rs`,
  `pack_int8_bias_to_bs`), but the executable ABI is per-tensor.
- **The quantizer is part of the product either way.**
  - On both models, its choices (which ops to fold, which scales to share,
    the skip quantization) decide between 65 ms and 245 ms.
  - Folding `Clip` was a missing default there. It is the default now:
    MobileNetV2 runs 30.0 -> 21.0 ms, and 9.1 ms with the int8 depthwise
    pad fold.
