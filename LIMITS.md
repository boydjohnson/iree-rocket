# Tested limits

What this stack is *measured* to do, as of 2026-09-05. Every number here is
the extent of an actual measurement -- a vendor capture, a hardware sweep on
an RK3588, or both -- not the extent of what the register encodings could
express. The register fields are almost always wider: `CNA_WEIGHT_SIZE2.weight_kernels`
is 14 bits and could hold 16383 output channels, and the constant that governs
it sits at 1792 because that is where the evidence stops. Raise a limit with a
measurement, never ahead of one.

Read this alongside [ISSUES.md](ISSUES.md), which carries the open defects.
A limit here says "this was tested and works"; it does not say "everything
inside it is safe" -- see [Hazards inside the limits](#hazards-inside-the-limits).
A limit here also does not say the op is reachable at all: which *operations*
the compiler can claim, as opposed to which shapes of them, is
[ROADMAP.md](ROADMAP.md).

Nothing here is a performance statement. Throughput, the CPU baseline the
offload has to beat, and where the time actually goes are ISSUES.md P7/P8 and
the README's "The CPU-only baseline"; a shape being inside every bound below
says only that it computes the right answer.

## Where a limit can bind

A shape has to clear three independent gates, and they do not agree with each
other. Whichever is tightest for a given shape is the one that decides.

| Layer | Where | What it bounds | What happens past it |
|---|---|---|---|
| Compiler matchers | [`rocket_conv2d_transform_spec.mlir`](rocket-compiler-plugin/target/Rocket/rocket_conv2d_transform_spec.mlir) | Which ops in a model are claimed for the NPU at all | Silent, graceful CPU fallback |
| Wire format | [`rocket_executable_def.fbs`](rocket-schema/schema/rocket_executable_def.fbs) | Which precisions and ops a compiled `.vmfb` can even express | Serialization error at compile time |
| HAL planner | [`conv.rs`](iree-rocket-hal/src/rocket/conv.rs), [`pooling.rs`](iree-rocket-hal/src/rocket/pooling.rs) | Which shapes `ConvPlan`/`PoolingPlan` will program | Panic (loud) -- the matchers are set so this is unreachable from a compiled model |

The matcher bounds are deliberately *at or below* the HAL bounds. Where they
differ it is because the HAL constant governs one rule and something else binds
first: at a 3x3 kernel the matchers stop `Cin` at 1152 even though
`MAX_INT8_INPUT_CHANNELS` is 1344, because `ConvPlan` refuses `Cin >= 1216` at
k=3 outright (the coefficient working set exceeds the eleven grantable CBUF
banks) and that refusal is a panic, not a fallback.

## Convolution channel limits

These are the HAL constants in `iree-rocket-hal/src/rocket/conv.rs`. They are
shared between the dense and depthwise `Shape` constructors, and -- because a
matmul reaches this hardware as a height-one 1x1 convolution -- they are also
the matmul limits under different names.

| Precision | Element | `Cin` max | `Cout` max | Constant |
|---|---|---|---|---|
| fp16 | 2 B | **1792** | **1792** | `MAX_INPUT_CHANNELS` / `MAX_OUTPUT_CHANNELS` |
| bf16 | 2 B | 1792 | 1792 | shares the fp16 constants |
| int16 | 2 B | 1792 | 1792 | shares the fp16 constants |
| fp16 + fp32 accumulator | 2 B in / 4 B out | 1792 | 1792 | shares the fp16 constants |
| int8 (requantized) | 1 B | **1344** | **1792** | `MAX_INT8_INPUT_CHANNELS` / `MAX_INT8_OUTPUT_CHANNELS` |
| int8 + int32 accumulator | 1 B in / 4 B out | 1344 | 1792 | shares the int8 constants |
| int4 | 0.5 B | **1344** | **1792** | `MAX_INT4_INPUT_CHANNELS` / `MAX_INT4_OUTPUT_CHANNELS` |
| tf32 | 4 B | **1024** | **1792** | `MAX_TF32_INPUT_CHANNELS` / `MAX_TF32_OUTPUT_CHANNELS` |

`Cout` reaches the same 1792 at every width because an output channel does not
charge CBUF feature residency; `Cin` does, which is why tf32's 4-byte element
sits lower. The four `Cin` ceilings do not reduce to one quantity -- neither
`Cin * element_bytes` nor feature-atom count fits all of them -- so this is a
table of what was measured rather than a rule.

The sharing of the fp16 constants across the other 2-byte rungs is measured at
each width rather than argued from the element width alone:
`bf16_regression_matrix` (58/58), `int16_regression_matrix` (37/37) and
`fp16_accumulator_matrix` (52/52) each run `Cin` 512/1024/1344 at k=1, `Cout`
to 1792, ragged channel counts, 56x56 and 112x112 multi-tile, 5x5 and 7x7, and
stride 2. int4 is `int4_regression_matrix_matches_oracle` (51/51) and tf32 is
`tf32_regression_matrix_matches_oracle` (50/50). All six live in
`iree-rocket-hal/tests/conv2d_oracle_hw.rs`.

Extra constraints on top of the table:

- **int4 `Cin` must be a whole 32-channel feature atom.** A partial int4 atom
  is unmeasured and the ARGB dense path cannot address a nibble, so
  `Shape::with_precision` refuses rather than guesses
  (`int4_refuses_a_partial_feature_atom`).
- **Depthwise fixes `Cout == Cin`** (channel multiplier one only) and is
  restricted to the 1- and 2-byte rungs with a narrowed result. tf32, int4 and
  the fp32-result writers are refused: the coefficient grouping is a 64-*byte*
  run, which is only a channel count once an element width is fixed, and the
  depthwise output writer has a 256-byte write atom no fp32-result measurement
  covers.
- **`Cin <= 4` takes the dense ARGB path** rather than NC1HWC2 surfaces, in
  both precisions. This is a channel-count boundary, not a byte-width one.
- **int8 accumulator output requires all three zero points to be zero.** The
  transform spec folds the activation zero point into a CPU-side correction
  before the dispatch for exactly this reason.

`ROCKET_ALLOW_UNBACKED_CHANNELS` in the environment turns the channel assertions
off. It is a probe hatch for extending the measurement, not a supported mode.

## Kernel, stride, spatial and padding limits

| Property | Tested limit | Notes |
|---|---|---|
| Kernel extent | 1..=11 per axis | Vendor reference data stops at 11 |
| Automatic CBUF planning | 1x1 and 3x3 only | Anything else needs an explicit `ConvPlan` or a bank override |
| Stride | 1 and 2 measured at 1x1 and 3x3 | **Stride is 1 everywhere above 3x3** -- the capture sweep has no strided capture at any larger kernel, and `assert_large_kernel_plan_case` enforces it |
| Batch | 1 | The ABI fixes it; every matcher requires it |
| Dilation | 1 | No dilation support in the builder at all |
| Leading padding | 0..=15, and `pad < kernel extent` | The CNA's 4-bit pad fields. There are no *trailing* padding registers; the output extent implies bottom and right |
| Input rows per program | `CNA_CBUF_CON1.data_entries` <= 0x7fff, `feature_grains` <= 0x3ff | Usually the CBUF bank grant binds first; the planner tiles |
| Input row width per program | `(ceil(atoms/4) - 1) * in_cols <= 2047` for NC1HWC2 input (89 at fp16 `Cin` 768, 292 at 256, 37 at 1792; unbounded at one slab, `Cin` <= 32) | The CBUF's 11-bit entry-slab base, `MAX_ENTRY_SLAB_BASE`; measured exact at 2048 across fp16/bf16/int16/tf32/int8. `Shape::max_tile_input_width` bounds it and the planner splits columns. A row that does not fit its data grant is refused too (capacity 0), no longer forced through |

### Above 3x3, `Cin` is the cliff

Measured on `planck` 2026-09-04 with `dtype_boundary_probe`. These are **hangs**
(the watchdog kills the job), not wrong data, so they are contained by
`large_kernel_max_in_channels` refusing the shape up front:

| Kernel | fp16 / bf16 / int16 / fp16-acc | tf32 | int4 | int8 |
|---|---|---|---|---|
| 5x5 | no measured ceiling | no measured ceiling | no measured ceiling | **refused** |
| 7x7 | `Cin` 64 | `Cin` 32 | `Cin` 128 | **refused** |
| 9x9 | `Cin` 64 | refused (fp16-only) | refused (fp16-only) | **refused** |
| 11x11 | `Cin` 64 | refused (fp16-only) | refused (fp16-only) | **refused** |

At 5x5 nothing has failed at any width; the CBUF planner's own refusal is what
bounds it. 5x5 and 7x7 are planned precision-neutrally, subject to that `Cin`
ceiling; 9x9 and 11x11 key on a channel *count* whose byte footprint differs
fourfold across the rungs, so `assert_large_kernel_plan_case` admits fp16 only
there.

**int8 is refused above 3x3 outright**, and it is not a ceiling: at 5x5 and 7x7,
`Cin` 16, 32 and 64 alike come back with every output channel holding the same
value at a given pixel (~14600 of 16384 elements wrong). That is coefficients
not reaching their channels at all. Both int8 rungs share the packing, so both
are refused. ISSUES.md C9 carries the write-up.

## Matmul

A matmul is lowered as a convolution of height one with `K` as `Cin` and `N` as
`Cout` (`fc::Shape`), so it inherits the convolution channel limits directly.
Validated on `planck` 2026-09-04, 23 points, **0 mismatches and 0 device
timeouts**, under both the `Selectors` and `Counting` oracle patterns:

| Dimension | Tested range | Bound by |
|---|---|---|
| `M` (conv width at height one) | 1..=2047 in the compiled path; 1..=296 measured in the HAL, 197 end to end | `CNA_DATA_SIZE0.datain_width` is 11 bits. The vendor FC sweep covers 1, 2, 7, 16, 32 (three CBUF splits); above the row-width limit above the planner splits column tiles -- see below |
| `K` (conv `Cin`) | 1..=1792 | `MAX_INPUT_CHANNELS`; measured 512, 1024, 1344, 1792, 2048 |
| `N` (conv `Cout`) | 1..=1792 | `MAX_OUTPUT_CHANNELS`; measured 64, 512, 1001, 1792, 2048 |

MobileNetV2's classifier, `M=1 K=1792 N=1001`, is exact under both patterns and
again with the fp32 accumulator kept. `K = 1792` is why `MAX_INPUT_CHANNELS` was
raised from 1344 on 2026-09-04: the geometry that carries it -- a 1x1 spatial
"image" -- is not one any convolution sweep had run.
`fc_matmul_ladder_matches_the_fc_lowering` keeps the regression's cases
identical to what `fc::Shape::as_conv_shape` actually builds, so the ladder
cannot drift into measuring its own geometry.

**Above `M` 32 the row splits, and the split is measured.** A single input
row wider than `(K/32 - 1) * M <= 2047` CBUF entries reads its last 32
channels from the wrong place (ISSUES.md C10, resolved 2026-09-05); until that
day the `M <= 32` matcher bound was what kept it out of compiled models. The
planner now bounds the row (`Shape::max_tile_input_width`) and plans column
tiles past it, measured exact on `planck` with `dtype_boundary_probe` at `M`
90, 128 and 197 (`K` 768, two and three columns), 296 (`K` 256), 38 (`K`
1792), at bf16, int16, tf32 and both int8 paths, and under the `onehot` read
map at 197 and 296. The transposed `1 x M` geometry row-tiles instead and is
exact at the same points with fewer tiles (ISSUES.md D1).

**End to end, ViT-B/16 (2026-09-05).** With the matcher bound raised to the
register's 2047 and the spec collapsing ONNX's unit-batch `batch_matmul` to
`linalg.matmul`, `rocket-compiler` offloads the twelve `197x768x768` attention
out-projections (12 of 272 dispatch sites; QKV at N 2304 and the MLP at N/K
3072 exceed the 1792 channel caps). Against the `--no-offload` build on
`planck`, same input: max|err| 0.0014 on logits of magnitude 6.8, top-5
identical, **3.87 s vs 4.06 s per inference** (`iree-benchmark-module`, 3
repetitions each) -- the first configuration in this repo faster than its
like-for-like CPU arm, by 5%. Read with [[planck-measurement-environment]]'s
caveats; it is one input and one core allocation.

## Pooling

| Property | Tested limit |
|---|---|
| Extents (H, W, C) | 1..=8192, the PPU's 13-bit N-1 range |
| Kernel, directly programmed | 1..=8 per axis (`MAX_DIRECT_KERNEL`); a 16x16 window is rejected by the hardware |
| Kernel, matched from a model | 2..=8 for both methods. For the average the floor is forced -- its reciprocal is `fp16(65536/k)` and `k=1` needs 65536, past fp16's 65504 ceiling. For max `k=1` is programmable and excluded anyway: a 1x1 stride-1 max pool is an identity, and claiming it would spend a dispatch, a pack and a compaction to copy a tensor |
| Kernel or stride, register range | 1..=16 |
| Padding | 0..=7, the PPU's 3-bit range. Every matched executable bakes 0: model-level padding arrives as a separate `tensor.pad` the CPU runs |
| Stride, matched from a model | avg 1; max 1 and 2 -- the executable bakes it, one per value. Stride 2 is measured against the oracle in `pooling_oracle_hw.rs`; nothing above 2 is |
| Method, matched from a model | avg and max. **Min has no matcher**: `PoolingMethod::pad_fill_value` has no measured identity for min at any precision, and none for max at int8. An *unpadded* pool never reads that field, so min is runnable and simply unclaimed -- ROADMAP.md Phase 0 |

Pooling has a second, narrower limit: **direct tile width**, which is a hang
rather than wrong data past the boundary. Measured on `planck` 2026-09-04 with
`pooling_width_probe`, deterministic 3/3, and independent of image height and
of tiling:

| Kernel | Widest input that completes | First width that hangs |
|---|---|---|
| 3x3 | 34 | 35 |
| 4x4 | 34 | 36 |
| 5x5 | 20 | 24 |
| 8x8 | 20 | 24 |

Vertical stride is part of the rule, and `overlapping_window_width_limits` is
where it lives: 3x3 and 4x4 are narrowed **only at `stride_y == 1`**, which is
where all of their measured hangs are, because the vendor's own captured split
for a 256-wide 3x3 pool is two 127/129-column tiles at stride 2. 5x5 and wider
are narrowed whenever their windows overlap at all. A 2x2 window is unaffected
(40 wide at stride 1, and a 258-wide 2x2/s2 tiled case, are both exact).
`PoolingPlan` splits wider images into tiles the hardware is measured to run, so
a wide pool is slower than necessary and correct. This is a containment, not an
explanation -- the vendor corpus has direct 129-wide captures, so the hardware
can run widths this refuses.

## Datatypes: two different menus

The HAL crate and the compiled-model path do **not** support the same set.

`ConvPlan` (`iree-rocket-hal`, reachable from Rust and the hardware tests)
supports eight precisions: fp16, bf16, int16, fp16-with-fp32-accumulator, tf32,
int4, int8 (requantized), int8-with-int32-accumulator.

The executable wire format supports **three**: `INT8`, `FP16`,
`INT8_ACCUMULATOR`. Everything else is characterized at the HAL level and has no
path through `iree-compile` today. int16 in particular is explicitly waiting on
a full-iteration integer *output* writer before anything should depend on it.

## What a compiled model actually offloads

These are the enabled matchers -- the complete set of shapes that can leave the
CPU today. Everything else falls back silently and correctly.

| Op | Layout | Types | Kernel | Stride | `Cin` | `Cout` |
|---|---|---|---|---|---|---|
| conv | NHWC HWCF | f16/f16/f32 | 1x1 | 1 | 1..=1344 | 1..=1792 |
| conv | NHWC HWCF | f16/f16/f32 | 3x3 | 1 | 1..=1152 | 1..=1792 |
| conv | NHWC HWCF | f16/f16/f32 | 1x1, 3x3 | 2 | 1..=512 | 1..=512 |
| conv | NHWC HWCF | i8/i8/i32 | 1x1 | 1 | 1..=1344 | 1..=1792 |
| conv | NHWC HWCF | i8/i8/i32 | 3x3 | 1 | 1..=1152 | 1..=512 |
| depthwise | NHWC HWC | f16/f16/f32 | 1x1, 3x3 | 1 | 1..=512 | = `Cin` |
| depthwise | NCHW CHW | f16/f16/f32 | 1x1, 3x3 | 1, 2, 3, 4 | 1..=512 | = `Cin` |
| depthwise | NHWC HWC | i8/i8/i32 | 1x1, 3x3 | 1, 2 | 1..=1344 | = `Cin` |
| matmul | -- | f16/f16/f32 | -- | -- | `M` 1..=32, `K` 1..=1792 | `N` 1..=1792 |
| avg pool | NCHW sum | f32 | 2x2..=8x8 | 1 | H/W/C 1..=8192 | -- |
| max pool | NHWC | f32 | 2x2..=8x8 | 1, 2 | H/W/C 1..=8192 | -- |
| max pool | NCHW | f32 | 2x2..=8x8 | 1, 2 | H/W/C 1..=8192 | -- |

All of these additionally require batch 1 and dilation 1.

Stride-3 and stride-4 *dense* fp16 matchers exist in the spec but are **not** in
the `foreach_match` list. Depthwise NCHW carries strides 3 and 4; depthwise NHWC
fp16 has no strided matcher at all.

Every accepted bound in that table has an immediately-adjacent rejected shape
asserted in a lit test -- `rocket_int8_match_boundaries.mlir`,
`rocket_fp16_match_boundaries.mlir`, `rocket_matmul_match_boundaries.mlir`,
`rocket_pooling_match_boundaries.mlir`,
`rocket_pooling_max_match_boundaries.mlir` -- so widening a matcher cannot silently
route a known-bad shape to the NPU, and tightening one cannot silently lose the
largest measured-good shape.

The pooling rows additionally have an end-to-end differential behind them:
`tools/e2e_pooling_regression.py` compiles both methods twice and compares on
the board. Max is compared **exactly** and measured exact on `planck`
2026-09-05 (max|error| 0 across both layouts, both strides, the 8x8 ceiling
and a tiled width); the average carries genuine f16 error, 0.0057 worst case
at 49 taps, because the PPU's average is a multiply by `fp16(65536/k)` that
the shim multiplies back out.

What that adds up to on a real model, from `rocket-compiler audit` (2026-09-05):

| Model | Rocket dispatch sites | Of which convolutions | CPU dispatch sites |
|---|---|---|---|
| `mnv2.fp16.mlir` | 37 | 35 (34 stride-1 dense, 1 stride-2 stem) | 145 |
| `mnv2.int8.mlir` | 50 | 48 (34 dense, 13 depthwise, 1 stride-2 stem) | 95 |

Both also offload the classifier matmul and the average pool. MobileNetV2 fp16's
17 depthwise convolutions stay on the CPU by choice, not by a limit -- they are
correct on the NPU and 26% slower there (ISSUES.md P7).

## End-to-end validation

Two models have been run through the whole pipeline: **MobileNetV2** (fp16, and
int8 via ONNX Runtime `quantize_dynamic` / `onnx.ConvInteger`) and **VGG**
geometry (30x30, `Cin` 512, `Cout` 512, 3x3).

The board gate is:

```bash
python3 tools/e2e_conv_regression.py --board "<board name>"
```

It compiles each case twice -- once for the host CPU, once for Rocket -- runs
the Rocket build on the RK3588 through `iree-run-module`, and compares. int8
cases are compared **exactly** (atol = rtol = 0), since the whole path is
integer arithmetic; fp16 cases use 1e-2. Thirteen cases: dense f16 3x3
`Cin`/`Cout` 512, dense `Cin`=3 ARGB, depthwise 40-channel (crossing the
driver's 32-channel weight-packing group boundary), five int8 dense variants up
to `Cout` 768, int8 depthwise at stride 1 and 2, and three two-dispatch cases
that share one command buffer the way a real model does. All thirteen gate; the
mixed int8-then-fp16-depthwise case was a `known_failure` until 2026-09-05 and
is now an ordinary one (ISSUES.md C8).

Build the arms with `rocket-compiler`, not a bare `iree-compile`: the
`rocket-pin-unclaimed-dispatches` pass has to run between the `flow` and
`stream` phases, and without it Stream's affinity analysis places the int8
epilogue on `@rocket_device` and serialization fails on an op that is not a
convolution. See the README's "Placement pinning".

The MobileNetV2 fp16 stem carries a known **precision** cost rather than a
correctness one: Rocket's ABI is f16-in/f32-accumulate, so the f32 stem feeding
an int8 quantization step lands ~0.35-0.42 max|err| on the final logits against
a plain f32 build. Top-1 is stable except on near-ties. The isolated stem
convolution matches a CPU reference computing the same f16 arithmetic to f16
epsilon.

## Hazards inside the limits

Being inside every bound above is necessary, not sufficient. These are the
known ways a shape in range still misbehaves.

- **An 11/1 CBUF split with a large coefficient footprint was measured to
  produce silently all-zero output.** `Cin`=256/`Cout`=256/3x3 -- comfortably
  inside the matcher bounds -- was deterministically all-zero (0/5) at every
  spatial extent from 26x26 to 48x48, while extents 20-24 and 50-58 at the
  *same* channel counts passed 5/5, with the pass/fail boundary on `ConvPlan`'s
  split-flip points. `Cin`=3/`Cout`=64 also gets an 11/1 split and is fine, so
  the discriminator is the split *combined with* a large coefficient footprint.
  **Status is unclear and that is the hazard**: the finding survives only as a
  comment in `@match_dynamic_conv2d_3x3` in the transform spec, the
  `conv_cbuf_split_sweep_hw.rs` and `DESIGN_NOTES.md` that comment cites are
  both gone from this tree, and ISSUES.md tracks it nowhere. The 3x3 matcher's
  `Cout` bound has since been widened from 256 to 1792, so the "safe for VGG"
  reasoning in that comment no longer describes the bounds in force. Re-measure
  before trusting a 3x3 shape near an 11/1 split.
- **`Cin > 4` and `Cin <= 4` take different feature paths**, and the dense
  ARGB path silently corrupts multi-row fetches at some alignments;
  `Shape::dense_feature_offset_safe` is the hardware-measured guard.
- **The NPU's state is order-dependent between dispatches, and a hung job can
  leave the device sick for the next one.** `e2e_conv_regression.py` waits for a
  quiet NPU before it starts and warns when it gives up, and `--only FUNCTION`
  exists so one case can get a clean verdict. A failure in the first case after
  a hang is not evidence about that case.
- **A watchdog-killed job used to read as a wrong answer.** It no longer does --
  any SUBMIT -> PREP_BO round trip over `DISPATCH_TIMEOUT_FLOOR` is labelled a
  hung job (ISSUES.md C3) -- but every timing number taken before 2026-09-03
  absorbed silent hangs and is invalid.

Two hazards that used to be listed here are fixed, and are recorded in
ISSUES.md's **Resolved** section rather than repeated: the int8 offload hanging
after ~2 consecutive inferences in one process, and an int8 dispatch hanging a
following fp16 depthwise dispatch in the same command buffer. Both were the same
cause -- `brdma_data_use` left set on the int32-accumulator path -- fixed
2026-09-05 (C8).

## Changing a limit

1. Measure it on hardware -- a probe under `iree-rocket-hal/tests/*_hw.rs`, or
   `dtype_boundary_probe` / `accumulator_size_e_probe` / `pooling_width_probe`
   for a ladder. An "it did not time out" check is not enough: the worst
   failure mode in this stack is a job that completes successfully with
   all-zero output, so check exact expected values.
2. Raise the HAL constant and record the evidence in its doc comment -- date,
   board, probe, and the points measured. Every constant in `conv.rs` carries
   its own provenance; that is what makes a later raise reviewable.
3. Raise the matcher bound in the transform spec, and move the paired
   accepted/rejected shapes in the corresponding `*_match_boundaries.mlir`.
4. Add the shape to `tools/e2e_conv_regression.py` if it is a shape a real
   model reaches.

Set the constant at what a real model needs and the corpus reaches, not at the
furthest point that happened to pass. Several constants sit at 1792 where 2048
also measured clean, deliberately.
