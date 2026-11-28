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
  rewrite** (done: `rocket-fuse-qdq-conv`, measured in the next section): `Q(conv(DQ(x), DQ(w)) + b)` into the canonical form
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

## Follow-up: `rocket-fuse-qdq-conv` (2026-11-27)

A plugin pass (`rocket-compiler-plugin/target/Rocket/RocketFuseQdqConvPass.cpp`)
now rewrites `Q(conv(DQ(x), DQ(w)) + DQ(b))` into the canonical requantized
int8 form. The `rocket_dynamic_int8_requant*` and
`rocket_dynamic_depthwise_int8_requant*` matchers then claim QDQ
convolutions the way they claim QOperator ones.

- **Where:** in the spec, right before `rocket-fuse-int8-requant-epilogue`
  (after the channels-last conversion, before the match loops). It anchors on
  `iree_linalg_ext.quantize_affine`.
- **What it emits:**
  - an i8 x i8 -> i32 NHWC convolution over a zero init;
  - a rank-1 folded bias `bq - zx * sum_k(w)`, constant-evaluated later;
  - the one canonical requant generic, with `scale = sx * sw / sy`.
- **Dense convolutions** arrive NHWC from the channels-last conversion. A
  padded input pads its codes with `zx`. The conversion's rank-5
  `expand -> pad -> collapse` is rebuilt as one rank-4 NHWC pad, because the
  int8 pad1 template only claims that form. The first version kept the
  rank-5 spelling, and every 3x3 fell to `int8_accumulator`.
- **Depthwise convolutions** stay in the importer's NCHW
  `depthwise_conv_2d_nchw_chw`. The pass emits the NHWC chain the QOperator
  path reaches after #119: `transpose -> pad_nhwc(zx) ->
  depthwise_conv_2d_nhwc_hwc`. So a -128 pad goes to the CNA on
  `#rocket_dynamic_depthwise_int8_requant_pad1{,_s2}_target`.
- **Signedness:** S8S8 (ORT's default) uses the codes as they are. U8S8 gets
  the same `addi -128` shifts the QOperator passes use.
- **Declined, so those convolutions keep the fp16 path:**
  - per-channel weight scales (the executable ABI's `output_scale` is one
    value; see Implications above);
  - a nonzero weight zero point;
  - a bias scale that is not `sx * sw` (checked to 2^-20 relative);
  - any clamp range narrower than the full 8-bit one.
- **Lit:**
  - `rocket_fuse_qdq_conv.mlir` runs the pass alone: S8S8; U8S8 with the
    rank-5 pad; a zero input zero point; a stride-2 padded depthwise; and
    the per-channel and bias-scale declines.
  - `rocket_qdq_conv_match.mlir` runs the importer's NCHW form through the
    whole spec onto `int8_requant`, `int8_requant_pad1` and both depthwise
    pad1 executables. A per-channel case stays fp16.
  - All 58 plugin lit tests pass.

### Placement (same `ort_qdq` imports as the first table)

| model | int8 convs | on | CPU sites before -> after |
|---|---|---|---|
| ResNet50 | 53 / 53 | dense requant, s2, pad1, pad1_s2 (the executables `ours` uses) | 120 -> 25 |
| MobileNetV2 | 52 / 52 | dense requant and s2; depthwise requant pad1 and pad1_s2 | 127 -> 16 |

- **ResNet50's 25 CPU sites** are 20 `elementwise_transpose`, 4
  `elementwise` and 1 `slow_memcpy`.
- **Most of them are the 16 residual adds**, which still run as
  `DQ -> add -> Q` in float. `ours` has 27 CPU sites; `ours_skip` has 5.

### Single-layer exactness

Two QDQ depthwise layers:
- 3x3, pad 1, stride 1 (96 channels) and stride 2 (144 channels);
- random weights, bias and input; `zx` -128.

Each compiled both ways and compared code for code with a float64
evaluation of the original Q/DQ arithmetic in numpy:

- **`--no-offload`** (the rewritten integer form on the CPU): **exact**, 0
  of 103,488 codes differ.
- **NPU**: at most 1 code off, on 0.021% (stride 1) and 0.025% (stride 2)
  of elements. Those are the ties the DPU rounds away from zero (memory
  `int8-tie-rule-half-away`).

### Board timing

Setup:
- orangepi, 7.3.6, patched `rocket.ko`, NPU at 600 MHz, governor
  `performance`, NPU IRQs on cpu6, `--task_topology_cpu_ids=4,5,6,7`.
- The runtime is built from this branch, so it has #119's depthwise
  pad-value fix.
- `iree-benchmark-module`, 7 repetitions with the first dropped. Two
  interleaved passes, each run gated on no other `iree-*` process on the
  board (a survey run had contaminated a first attempt).
- **Arms.**
  - `old` rows are the modules from the first table, re-timed here.
  - `ours` is the current `tools/quantize_onnx_int8.py` default (#118:
    `Conv Relu Clip`). For MobileNetV2 that is
    `survey/quant-clip-2026-11-27/imp/mnv2.clip.mlir`. For ResNet50 it is
    byte-identical to the first table's `ours`.
  - All arms are compiled with this branch.

| model | arm | ms (pass 1 / 2) |
|---|---|---|
| ResNet50 | `ort_qdq` NPU, **this pass** | **50.4 / 50.0** |
| ResNet50 | `ort_qdq` NPU, old (fp16 convs) | 136 / 136 |
| ResNet50 | `ort_qdq` `--no-offload`, this pass | 388 / 391 |
| ResNet50 | `ours` NPU | 39.7 / 35.3 |
| ResNet50 | `ours_skip` NPU (`--quantize-residual-skip`) | 17.5 / 17.9 |
| MobileNetV2 | `ort_qdq` NPU, **this pass** | **14.8 / 14.5** |
| MobileNetV2 | `ort_qdq` NPU, old | 49.8 / 49.4 |
| MobileNetV2 | `ort_qdq` `--no-offload`, this pass | 30.0 / 30.1 |
| MobileNetV2 | `ours` NPU | 10.1 / 9.7 |

- **ResNet50 QDQ.** 2.7x faster than before, and 7.8x faster than its own
  `--no-offload` build. Still 1.3x behind `ours`, and 2.8x behind
  `ours_skip`.
- **MobileNetV2 QDQ.** 3.4x faster than before. It was 1.45x slower than its
  CPU build and is now 2.1x faster than it. Still 1.5x behind `ours`.
- **The `--no-offload` arm runs the rewrite too**, so it is an int8 CPU
  baseline. The first table's fp32 fake-quant CPU arm is not comparable: it
  was timed under `ondemand`.

### Accuracy

Compared against onnxruntime on the same QDQ model, for both arms, on the
import's own input plus 4 synthetic inputs (mean 0.45, sd 0.225):

- **argmax is equal on all 10 NPU runs.** Top-5 overlap is 4 or 5 of 5.
- **ResNet50:** max|d| 0.10 to 0.20 against a logit sd of about 0.95, mean
  |d| about 0.025.
- **MobileNetV2:** max|d| 0.064 to 0.128 against an sd of 0.51 to 0.81, mean
  |d| 0.011 to 0.018.
- **Size of the errors.** Those are 2 to 6 steps of the output
  quantization (0.050 and 0.021). The int8 `--no-offload` arm sits at 1 to
  5 steps, and misses one ResNet50 argmax on a near-tie.
- **The NPU arm drifts further than the CPU arm** because the DPU breaks
  ties away from zero at every layer. The layer test above rules out
  anything else.
- **The QOperator path is untouched.** `ours` recompiled with this pass
  gives output bit-identical to the #116 build.

### What is left, by expected size

1. **QDQ residual add** (done for ResNet50, see the next section). None of
   ResNet50's 16 adds or MobileNetV2's 10 is rewritten.
   - ORT folds the ReLU into the add's QuantizeLinear (the zero point
     equals quant_min), and the activations are signed.
     `rocket-fuse-int8-residual-add` wants `ui8` and an explicit ReLU, so it
     matches neither.
   - `ours` vs `ours_skip` puts this at about 2x on ResNet50.
   - MobileNetV2's adds carry no ReLU at all, and there is no target for
     them yet in either path.
2. **QDQ max pool** (done, next section): the same signedness gap in
   `rocket-fuse-int8-max-pool`.
   ResNet50's stem pool runs fp16 between a CPU Q/DQ pair.
3. **Per-channel weight scales**: an executable-ABI change (above).

## Follow-up: signed QDQ residual add and max pool (2026-11-27)

`rocket-fuse-int8-residual-add` and `rocket-fuse-int8-max-pool`
(`RocketFuseInt8QdqPass.cpp`) now take the form ORT's default
`quantize_static` (QDQ, S8S8) writes. Two changes:

- **Signed codes.** The affine `dequantize_affine` / `quantize_affine`
  matchers accept signed `i8` as well as `ui8`. A signed operand or result
  gets no `addi -128` shift and keeps its zero point as is; each operand and
  the result is shifted only when its own codes are `ui8`. The clamp has to be
  the encoding's full range, [0, 255] or [-128, 127]. A symmetric [-127, 127]
  or 7-bit `reduce_range` clamp is declined.
- **The folded ReLU.** ORT deletes the Relu after a residual Add and gives
  the Add's QuantizeLinear `zero_point == quant_min` (-128). With that zero
  point the clamp sends every negative sum to the code of 0, so
  `Q(DQ(a) + DQ(b))` equals `Q(relu(DQ(a) + DQ(b)))` code for code, and the
  pass rewrites it into the same canonical add+ReLU generic. Without that
  equality (a linear add, MobileNetV2's) it still declines.
  - The same rule covers a `ui8` QuantizeLinear with zero point 0. The
    existing lit case that called such an add "linear" was a ReLU all along.
    It now uses zero point 128.
- **Max pool:** the DequantizeLinear and QuantizeLinear must also share
  signedness, besides scale and zero point.

**Lit.**
- `rocket_int8_residual_add.mlir`: S8S8 folded-ReLU (zero points -29, 5 and
  -128 reach the dispatch unchanged), S8S8 explicit ReLU, and two declines:
  S8S8 linear and a [-127, 127] clamp.
- `rocket_int8_max_pool.mlir`: the S8S8 stem pool onto
  `rocket_pooling_max_int8_pad1_executable_s2`, and a mixed-signedness
  decline.
- 58/58 plugin lit pass. The new positive cases fail on #120's compiler.

### Placement (the same `ort_qdq` imports)

| model | before (#120) | after |
|---|---|---|
| ResNet50 | 25 CPU sites; adds `DQ -> add -> Q` in float; stem pool fp16 | **6 CPU sites**; 16/16 adds on `rocket_elementwise_add_int8_relu`, pool on `rocket_pooling_max_int8_pad1_s2`; 85 NPU->NPU edges packed, 0 dense |
| MobileNetV2 | 16 CPU sites | unchanged (its 10 adds are linear) |

- The QOperator `ours_skip` ResNet50 recompiled with this pass has
  identical placement: 15 adds on the NPU, 5 CPU sites.

### Board timing

Same setup as the previous section: orangepi, 7.3.6, NPU at 600 MHz,
governor `performance`, `--task_topology_cpu_ids=4,5,6,7`, the runtime from
#120. `iree-benchmark-module`, 7 repetitions with the first dropped, every run
gated on no other `iree-*` process. Two runs of two passes each.

| ResNet50 arm | ms (run A pass 1 / 2; run B pass 1 / 2) |
|---|---|
| `ort_qdq` NPU, #120 | 49.8 / 52.6 |
| `ort_qdq` NPU, **this change** | 29.1 / 27.8; 27.3 / 26.8 |
| `ort_qdq` NPU, this change + `--iree-opt-const-expr-max-size-increase-threshold=67108864` | **17.2 / 16.8** |
| `ours_skip` NPU (`--quantize-residual-skip`) | 17.4 / 18.8; 17.1 / 17.6 |
| `ort_qdq` `--no-offload`, this change | 287.5 / 280 (#120: 389) |

- **The residual adds and the pool closed most of the gap:** 50 -> 27 ms.
- **The last ~10 ms was the FC layer, not the adds.** `ROCKET_PROFILE=1`
  per call:
  - the fp16 matmul is 7.8 ms here against 0.47 in `ours_skip`, of which
    5.3 ms is `pack.weights`;
  - `outside` (the CPU dispatches) is 5.4 ms against 2.0. Almost all of it
    is an `elementwise_transpose_2048x1000_i8xf16` dispatch.
- **Why the FC costs that much.** The Gemm's weight is `DQ(w_i8)`, and IREE's
  const-expr hoisting declines to precompute it: widening 2 MB of `i8` to
  `f16` grows it past the default 1 MiB size-increase threshold. So the
  weight is dequantized and transposed on the CPU every call, and the matmul
  sees a non-constant weight that it has to re-pack every call
  (`rocket-pack-weights: 1 with non-constant weights`).
- **Raising the threshold puts QDQ level with `ours_skip`:** 16.8 to 17.2 ms.
  The DQ dispatch disappears, all 54 weights are packed at compile time, and
  the output is bit-identical to the build without the flag. It is not a
  `rocket-compiler` default yet: the threshold applies to every model, and
  its effect on module size elsewhere is unmeasured.

### Accuracy

Against onnxruntime on the import's input plus the 4 synthetic inputs:
- **NPU: argmax equal on 5/5**, top-5 overlap 5/5 on all five, max|d| 0.10
  to 0.23, mean|d| about 0.022. #120's NPU build had max|d| 0.10 to 0.20 and
  top-5 4 or 5 of 5.
- **`--no-offload`: output bit-identical to #120's** on all five inputs. The
  integer rewrite of the add, the folded ReLU included, is exact. It keeps
  #120's one near-tie argmax miss on input 1.

### What is left

1. **FC weight hoisting:** decide whether `rocket-compiler` should raise
   `--iree-opt-const-expr-max-size-increase-threshold`, or fold the QDQ
   Gemm's `DQ(w)` itself. That is worth 10 ms on ResNet50 QDQ. *Done: the
   targeted fold, next section.*
2. **Linear int8 residual add** (MobileNetV2): needs a target without the
   ReLU.
3. **Per-channel weight scales:** the executable-ABI change above.


## Follow-up: folding a QDQ weight's `DQ(constant)` (2026-11-27)

`rocket-fold-constant-dequant` (`RocketFoldConstantDequantPass.cpp`) folds
an `iree_linalg_ext.dequantize_affine` of a constant into the f32 constant
it computes, when every use of its result is operand 1 of a contraction or a
convolution (B of a matmul, a filter), through at most six
transpose/reshape/cast ops. It runs in the spec right after
`rocket-fuse-qdq-conv`, which needs the per-tensor convolution weights as
codes, so what it sees is a Gemm/MatMul's weight and a per-channel
convolution's. After it the weight is an ordinary float constant, and the
transpose and f16 demote downstream shrink it, so IREE's hoisting and
const-eval take it as they take a float model's weights.

**Why a fold, not the threshold.** Option (a) was to make `rocket-compiler`
pass `--iree-opt-const-expr-max-size-increase-threshold=67108864`. That
applies to every const-expr in every model. The fold touches only a
dequantization whose every use is a weight, and the result could be checked
module by module: every survey model is byte-identical (below).

**Offload-only.** The first version also ran in `--no-offload`, and that
arm regressed on the per-channel models (median ms, two passes):

| `--no-offload` arm | #121 | folding in the CPU arm too |
|---|---|---|
| ResNet50 `ort_qdq_pc` | 1511.5 / 1512.5 | 2032 / 2036.5 |
| MobileNetV2 `ort_qdq_pc` | 50.2 / 49.9 | 57.9 / 57.9 |
| ResNet50 `ort_qdq` | 287.5 / 280 | 279 / 282 |
| MobileNetV2 `ort_qdq` | 30.05 / 30.4 | 30.3 / 30.6 |

A CPU convolution reads the i8 codes with the dequantization fused. The
folded f32 constant is four times the bytes (the per-channel ResNet50 CPU
module went from 26 to 102 MB). So the pass carries
`with options = {"offload" = true}`, which `rocket-compiler --no-offload`
flips to false, as it already does for
`rocket-block-diagonal-grouped-conv`. Every `--no-offload` module is now
byte-identical to #121's.

**Exactness.** The fold evaluates `DequantizeAffineOp::decomposeOperation`
element by element: extend, subtract the zero point in
`getZeroPointDifferenceType`, `sitofp`, multiply in the scale's type, then
convert to the result type, each rounded to nearest-even. Per-tensor and
per-channel scales, scalar or tensor zero points, `input_unsigned` /
`zp_unsigned`, `dense` and `dense_resource` codes. The NPU ResNet50 `ort_qdq`
module is byte-identical to #121's threshold-raised build except for the
spec path string embedded in `module.fb`.

**Lit.** `rocket_fold_constant_dequant.mlir`: the ResNet50-shaped Gemm
weight (transpose into B), a `dense_resource` weight, unsigned codes with an
unsigned zero point, a per-channel FCHW filter into
`conv_2d_nhwc_hwcf`, and an f16 result (f32 product, then f16). It also
covers three declines: the matmul's A operand, a second non-weight use, and
non-constant codes. A second RUN line with `offload=false` checks the
dequantization survives. 59/59 plugin lit pass. `cargo test -p
rocket-compiler` passes with the offload-only count raised to 2.

### Placement and module size (NPU arm)

| import | non-constant weights in `rocket-pack-weights` | vmfb MB | compile s |
|---|---|---|---|
| ResNet50 `ort_qdq` | 1 -> **0** | 25.7 -> 27.7 | 4.7 -> 5.0 |
| ResNet50 `ort_qdq_pc` | 5 -> **0** | 40.1 -> 51.3 | 7.3 -> 8.6 |
| MobileNetV2 `ort_qdq` | 1 -> **0** | 3.6 -> 4.9 | 4.1 -> 4.0 |
| MobileNetV2 `ort_qdq_pc` | 1 -> **0** | 5.9 -> 7.2 | 6.2 -> 5.6 |

- ResNet50 `ort_qdq` loses its `elementwise_transpose_2048x1000_i8xf16` CPU
  dispatch.
- The size growth is the weights now stored as packed f16 instead of i8.
- Compile times are single runs, three compiles in parallel on laplace.

**Unchanged, byte for byte** (old = #121's compiler and spec, new = this
change, both arms each):
- the 10 survey models (`survey/2026-11-27-main` imports): mobilenet_v2,
  resnet50, resnext50_32x4d, convnext_tiny, wide_resnet50_2, vit_b_16,
  blip, vit_l_16, bert_base, bert_base_384. None contains a
  quantize/dequantize op.
- the QOperator imports: ResNet50 `ours` and `ours_skip`, MobileNetV2
  `ours`.

### Board timing

orangepi, 7.3.6, NPU at its 600 MHz default (not re-read: the clock is only in debugfs, which needs a sudo password), governor `performance`,
`--task_topology_cpu_ids=4,5,6,7`, the #120 runtime. `bench_board.sh` from
the previous section: 7 repetitions with the first dropped, each run gated
on 30 s with no other `iree-*` process, from a script file whose command
line names no module. Median ms, two passes.

| NPU arm | #121 | this change |
|---|---|---|
| ResNet50 `ort_qdq` | 26.9 / 27.7 | **17.35 / 18.7** |
| ResNet50 `ours_skip` (QOperator, reference) | 17.6 / 17.7 | (unchanged module) |
| ResNet50 `ort_qdq_pc` | 134.5 / 135.5 | **76.85 / 77.35** |
| MobileNetV2 `ort_qdq` | 14.6 / 14.85 | **9.15 / 9.155** |
| MobileNetV2 `ort_qdq_pc` | 49.9 / 49.65 | **43.5 / 43.5** |

- ResNet50 QDQ is now level with the QOperator `ours_skip` build.
- `ROCKET_PROFILE=1` on ResNet50 `ort_qdq`, per call: the FC
  `matmul fp16 1x2048x1000` is 0.47 ms, down from 7.8, and there is no
  `pack.weights` phase. `outside` (the CPU dispatches) is 1.26 ms, down from
  5.4.
- Per-channel ResNet50 is still 77 ms because its convolutions stay fp16:
  the executable ABI carries one `output_scale` (section above).

### Accuracy

Against onnxruntime (`*.oracle.npy`):
- **ResNet50 `ort_qdq` NPU:** argmax equal on 5/5 (the import's input plus
  the 4 synthetic), top-5 overlap 5/5. The output is bit-identical to #121's
  threshold-raised build on all five.
- **The other three NPU arms:** argmax equal, on the import's input. The
  output is bit-identical to #121's.
- **`--no-offload`:** the module is unchanged, and so is the output. That
  includes #120's near-tie argmax miss on ResNet50 input 1.

### What is left

1. **Per-channel weight scales** on the int8 path, as before: the ABI
   change. It is now the whole gap on `ort_qdq_pc`. *Done for most layers:
   next section.*
2. **Linear int8 residual add** (MobileNetV2).
3. **A folded weight the planner then refuses** stays on the CPU as an f32
   constant in an offloading build, because the fold runs before
   placement. No current model has one: all four QDQ imports pack every
   weight.

## Follow-up: per-channel weight scales on the requantized path (2026-11-27)

ORT's `quantize_static(per_channel=True)` gives each output channel its own
weight scale, so the requantization ratio `sx * sw[c] / sy` is per channel.
The DPU already has a per-channel multiplier: each BS entry carries a 16-bit
`multiplier` lane next to its bias, read from memory (`bs_mul_src = 1`)
whenever the conv is quantized. Until now every lane held the unit value
`2^14`. The vendor uses it for exactly this
(`bs_buffer_matches_the_converted_models`: lanes `1024 * (c + 1)` for weight
scales `0.01 * (c + 1)`). What was missing was a way to carry the factors
from the compiler to the driver.

### The ABI

- **Canonical form.** The requantization generic takes one more operand,
  `channel_scale: tensor<Cout x f32>` indexed by the channel, and multiplies
  by it before the per-tensor scale. So channel `c` is requantized by
  `m[c] * scale`.
  - `rocket-fuse-qdq-conv` writes `scale = max_c(sx * sw[c] / sy)` and
    `m[c] = ratio[c] / scale`, in (0, 1].
  - Per-tensor weights, and every QOperator convolution
    (`rocket-fuse-int8-requant-epilogue`), get `m = 1` on every channel:
    exact in f32, and the unit lane they always had.
- **`Conv2DDef.per_channel_scale`.** This adds a fifth binding after the
  bias: `Cout` f32 factors. The bindings are then input, weights, bias,
  scales, output. The driver writes `round(2^14 * m[c])` into each
  multiplier lane (`pack_int8_bias_to_bs_per_channel`) and refuses a factor
  that rounds to 0 or passes the signed lane. The flag excludes
  `epilogue_add`. An executable without it reads `m = 1`, so an older module
  runs unchanged: #122's modules give bit-identical output on the new
  runtime.
- **Why a separate binding, not the factors after the biases in one
  binding.** That was the first version. Concatenating in the shim needs a
  `tensor.concat` on dynamic types. IREE does not constant-fold it, and its
  tensor-to-flow conversion replaces the value with a static
  `flow.tensor.update` without fixing the dispatch's dims (`value set has 7
  dynamic dimensions but only 8 dimension values`). A separate binding
  needs no op in the shim at all.
- **`rocket-drop-unit-channel-scales`.** This pass runs at the end of
  preprocessing. IREE does not store a splat `dense<1.0>` in the module; it
  materializes it with a fill on every call, between the Rocket dispatches
  it feeds. So for a per-tensor convolution the new binding costs:
  - ResNet50 `ours_skip`: 4 -> 5 NPU submissions and 4 -> 6 compactions per
    call;
  - same-session A/B: 17.8 -> 18.6 ms, `ort_qdq` 17.5 -> 17.8 ms.

  The pass rewrites the spec's executable in place for its unit-factor
  dispatches: no flag, four bindings. Dispatches with real factors move to
  a `_per_channel` clone when both kinds share the executable. **Every survey
  model and every per-tensor quantized model now compiles to executables and
  constants byte-identical to #122's.** Only `module.fb` differs, in 21 bytes
  of debug line numbers into the spec.

### The precision gate

The lane resolves `m` to `2^-14`, a relative error of at most `2^-15 / m`.
`rocket-fuse-qdq-conv` therefore declines a convolution when any channel's
`m` is below `kMinChannelScale = 1/128`. There the error is half a code on
the widest unclamped output, which keeps the path's one-code tolerance.
Measured on the stock per-channel models, before compiling:

| model | layers whose smallest `m` >= 1/128 | with near-dead channels |
|---|---|---|
| ResNet50 `ort_qdq_pc` | 37 / 53 | 16 (smallest `m` 9e-8) |
| MobileNetV2 `ort_qdq_pc` | 39 / 52 | 13 (smallest `m` 7e-8) |

The compiled placement agrees: ResNet50 37 int8 and 17 fp16 candidates (16
convolutions plus the FC), MobileNetV2 39 int8 and 14 fp16. The declined
layers keep #122's fp16 path. A "dead" channel has a scale around `1e-7` of
the layer's largest. If its whole conv term is below a fraction of a code,
the channel could be folded into its bias with unit `m`. That would admit
1 more ResNet50 layer and 10 more MobileNetV2 layers (by the bound
`sum|w| * 128 * ratio < 1/256`). The other 15 ResNet50 layers have small-`m`
channels whose conv term reaches 3 to 65 codes, which one 16-bit lane cannot
carry.

### Lit

- `rocket_fuse_qdq_conv.mlir`:
  - per-channel dense with a per-channel bias scale (output scale 0.004,
    factors 1/4 to 1);
  - per-channel depthwise on the `[C, 1, kh, kw]` filter;
  - a factor-below-1/128 decline;
  - the per-tensor case checks its `dense<1.0>` factors;
  - a non-constant per-channel scale still declines.
- `rocket_qdq_conv_match.mlir` (whole spec): the per-channel convolution
  lands on `rocket_dynamic_int8_requant_executable_per_channel` with its
  factors binding. The per-tensor ones land on the spec's own executable,
  without `per_channel_scale` and with four bindings. The tiny-factor case
  stays fp16.
- `rocket_int8_requant_match.mlir`: the canonical form with the factor
  operand. One case is non-unit and keeps the binding.
- `rocket-fold-conv-pad` counts the epilogue's seven inputs.
- 59/59 plugin lit, iree-rocket-hal 208 (two new: lanes and refusals),
  rocket-hal-driver 67 (`decodes_per_channel_scale`), rocket-compiler 56.

### Board

orangepi, 7.3.6, NPU at its 600 MHz default, governor `performance`,
`--task_topology_cpu_ids=4,5,6,7`, the runtime from this branch.
`bench_board.sh` as before: 6 kept repetitions, median ms, each run gated on
30 s with no other `iree-*` process.

| NPU arm | #122 | this change |
|---|---|---|
| ResNet50 `ort_qdq_pc` | 79 / 79.8 | **54.7 / 55.15**, final build 58.55 / 54.65 |
| MobileNetV2 `ort_qdq_pc` | 44.4 / 47.95 | **21.25 / 20.15**, final build 20.4 / 20.95 |
| ResNet50 `ours_skip`, same-session A/B | 18 / 17 / 17 / 16.85 | 17 / 17.25 / 17.6 / 17.3 |

- The per-tensor models' executables are byte-identical to #122's, so they
  were not re-timed beyond the `ours_skip` A/B.
- Per-channel ResNet50 is still 3x the per-tensor build. `ROCKET_PROFILE`:
  the 16 fp16 convolutions pay a dense pack of their int8-sourced input
  (11 ms/call) and an output compaction (21 ms/call) at each int8/fp16
  boundary. The dead-channel fold above is the next lever.

**e2e gate** (`tools/e2e_conv_regression.py --skip-raw`, every `requant_*`
fixture, all now in the new canonical form): 21/21 pass, max|error| <= 1, 0
mismatches past atol 1. `requant_int8_1x1_cin512` now carries factors 1/4
to 1 and checks the BS lane against the CPU reference: max|error| 1, 0/65536.

### Accuracy (against onnxruntime, the import's input)

| arm | argmax | top-5 | max\|d\| |
|---|---|---|---|
| ResNet50 `ort_qdq_pc` NPU | equal | 5/5 | 0.151 (#122: 0.101) |
| MobileNetV2 `ort_qdq_pc` NPU | equal | 5/5 | 0.064 (#122: 0.064) |
| ResNet50 `ort_qdq_pc` `--no-offload` | equal | 5/5 | 0.101 |
| MobileNetV2 `ort_qdq_pc` `--no-offload` | equal | 4/5 | 0.043 |

- **Per-tensor models:** bit-identical to #122 (ResNet50 `ort_qdq` on all
  five inputs, MobileNetV2 `ort_qdq`).
- **Per-channel `--no-offload` arms:** these change, because the integer
  rewrite runs there too, as it does for per-tensor QDQ since #120.

### What is left

1. **Dead-channel fold** (above): 1 more ResNet50 layer and 10 more
   MobileNetV2 layers onto int8. Done: see the next section.
2. **The 15 ResNet50 layers no 16-bit lane can carry:** a second
   per-channel multiply (the BN plane) or a split of the layer by channel
   magnitude.
3. **Linear int8 residual add** (MobileNetV2), as before.

## Follow-up: the dead-channel fold (2026-11-28)

The per-channel path declines a layer whose smallest factor `m[c]` is below
1/128. On the stock per-channel models most such channels are dead: their
weight scale is around 1e-7 of the layer's largest, so the whole conv term
is a tiny fraction of a code and the channel's output is its bias.
`rocket-fuse-qdq-conv` now folds those instead of declining the layer.

### The rule

For a channel with `m[c] < 1/128`, with `ratio[c] = sx * sw[c] / sy`:

- **Dead** when `ratio[c] * sum_k|w[c, k]| * D < 1/256` (`kDeadChannelTerm`).
  `D = max(127 - zx, zx + 128)` is the widest `|x - zx|` an 8-bit activation
  reaches (255 at `zx = -128`); a padded tap contributes `zx - zx = 0`. The
  folded output is then off from the exact one by at most one code, and
  only next to a rounding tie.
- **Folded** as the canonical form already computes it: the channel's
  filter is zeroed (a new constant), its factor is 1, and its bias becomes
  `b' = round(v / scale)`, with `v = round(ratio[c] * b[c])` (clamped to
  +-256) and `scale` the per-tensor scale. The zero filter also zeroes the
  channel's input-zero-point correction, so the bias reaches the generic as
  `b'`. The pass checks `|b' * scale - v| <= 1/4` (`kDeadBiasSlack`), well
  inside half a code, so neither rounding rule nor the hardware's scale
  representation moves the result.
- **Declined**, as before, when any small-factor channel is not dead.

Per-tensor weights never have a factor below 1, so per-tensor and QOperator
models are untouched by construction.

### Placement (compiled, stock `quantize_static` per-channel models)

| model | int8 layers, #123 | with the fold | fp16 candidates left |
|---|---|---|---|
| ResNet50 `ort_qdq_pc` | 37 / 53 | **38** | 16 (15 convs + FC) |
| MobileNetV2 `ort_qdq_pc` | 39 / 52 | **49** | 4 (3 convs + FC) |

That is exactly the prediction from the `sum|w| * 128 * ratio < 1/256`
survey in the section above. Hardware jobs: ResNet50 90 -> 88, MobileNetV2
68 -> 63.

### Board

orangepi, 7.3.6, NPU 600 MHz, governor `performance`,
`--task_topology_cpu_ids=4,5,6,7`, the runtime from main (unchanged by
this). `bench_board.sh`: 6 kept repetitions, median ms, two passes, each
run gated on 30 s with no other `iree-*` process.

| NPU arm | #123 | the fold |
|---|---|---|
| MobileNetV2 `ort_qdq_pc` | 20.35 / 20.35 | **10.6 / 10.4** |
| ResNet50 `ort_qdq_pc` | 55.05 / 53.8 | 54.8 / 53.4 |

MobileNetV2 per-channel is now within ~1.3 ms of its per-tensor build
(9.15 ms). ResNet50 moves one layer and is level; its remaining 15 fp16
layers have small-factor channels whose conv term reaches 3 to 65 codes.

### Accuracy (against onnxruntime)

| arm | argmax | top-5 | max\|d\| (#123) |
|---|---|---|---|
| ResNet50 `ort_qdq_pc` NPU | equal | 5/5 | 0.202 (0.151) |
| MobileNetV2 `ort_qdq_pc` NPU | equal | 5/5 | 0.086 (0.064) |
| ResNet50 `ort_qdq_pc` `--no-offload`, x86 host | equal | 5/5 | 0.101 (0.101) |
| MobileNetV2 `ort_qdq_pc` `--no-offload`, x86 host | equal | 5/5 | 0.043 (0.043) |

The `--no-offload` arms run the same rewrite, fold included, and their
error does not move. So the fold's arithmetic is faithful, and the NPU
arms' larger max|d| is the requantized hardware path's one-code tie
differences (it rounds ties away from zero) spread over more layers.

### Lit

`rocket_fuse_qdq_conv.mlir`:
- a dense dead channel folded: factor 1, filter bytes zeroed, bias 750;
- a depthwise dead channel folded with input zero point -128 (`D = 255`):
  bias -313, because the f32 scale sits just under 0.0064;
- the factor-below-1/128 decline now uses a channel that is not dead (term
  0.01 codes).

59/59 plugin lit.
