# Tested limits

What this stack is *measured* to do, as of 2026-09-05. Every number here is
the extent of an actual measurement -- a vendor capture, a hardware sweep on
an RK3588, or both -- not the extent of what the register encodings could
express. The register fields are almost always wider: `CNA_WEIGHT_SIZE2.weight_kernels`
is 14 bits and could hold 16383 output channels, and the constant that governs
it sits at 3584 because that is where the evidence -- and the shape of a real
model -- stops. Raise a limit with a measurement, never ahead of one.

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
`MAX_INT8_INPUT_CHANNELS` is 3584, because `ConvPlan` refuses `Cin >= 1216` at
k=3 outright (the coefficient working set exceeds the eleven grantable CBUF
banks) and that refusal is a panic, not a fallback. The channel ceilings are a
k=1 measurement and only k=1 matchers follow them up.

## Convolution channel limits

These are the HAL constants in `iree-rocket-hal/src/rocket/conv.rs`. Dense
convolution and matmul share them -- a matmul reaches this hardware as a
height-one 1x1 convolution, so these are the matmul limits under different
names -- while depthwise has its own, lower ceiling (below).

| Precision | Element | `Cin` max | `Cout` max | Constant |
|---|---|---|---|---|
| fp16 | 2 B | **3584** | **3584** | `MAX_INPUT_CHANNELS` / `MAX_OUTPUT_CHANNELS` |
| bf16 | 2 B | 3584 | 3584 | shares the fp16 constants |
| int16 | 2 B | 3584 | 3584 | shares the fp16 constants |
| fp16 + fp32 accumulator | 2 B in / 4 B out | 3584 | 3584 | shares the fp16 constants |
| int8 (requantized) | 1 B | **3584** | **3584** | `MAX_INT8_INPUT_CHANNELS` / `MAX_INT8_OUTPUT_CHANNELS` |
| int8 + int32 accumulator | 1 B in / 4 B out | 3584 | 3584 | shares the int8 constants |
| int4 | 0.5 B | **3584** | **3584** | `MAX_INT4_INPUT_CHANNELS` / `MAX_INT4_OUTPUT_CHANNELS` |
| tf32 | 4 B | **3584** | **3584** | `MAX_TF32_INPUT_CHANNELS` / `MAX_TF32_OUTPUT_CHANNELS` |
| any, depthwise | 1-2 B | **1792** | = `Cin` | `MAX_DEPTHWISE_CHANNELS` |

**Every dense ceiling moved to 3584 on 2026-09-06**, from 1792 (fp16 family),
1344/1792 (int8, int4) and 1024/1792 (tf32). Until then the ceilings differed
per rung and the table above was a table of separate measurements; the ladders
now agree at every width, because at k=1 the binding quantity is not CBUF
feature residency. tf32 is the case that shows it: a 4-byte element charges
four times fp16's residency per channel and was expected to stop lowest, and
instead it reaches the same 3584 with roughly double the tile count (28 against
14 at `Cin` 3584) -- the residency turns into geometry rather than into a
refusal. `Cout` never charged residency at all, which is why it was already
equal across the rungs.

The raise was driven by transformer shapes: ViT-B/16 and Qwen3 both have
`K = N = 3072` MLPs and ViT's QKV projection is `N = 2304`, all of which the
old 1792 excluded. Measured on `planck` from a quiet board with
`dtype_boundary_probe`, one sweep per process, 0 mismatches and 0 device
timeouts at every point:

* `Cin` at k=1, 14x14 `Cout` 64: 1792 through 3584 in 256-channel steps, then
  3840, 4096, 4608, 5120, 6144 and **8192** at fp16, and 1792/2304/3072/3584/
  4096 at every other rung. Under `Selectors` (addressing) and `Counting`
  (every lane contributes).
* `Cout` at k=1, 7x7 `Cin` 448: 1792 through 4096 at every rung, CBUF split
  flat across the whole range.
* ragged counts on both axes: 1793, 2049, 2313, 3073, 3585, 4095.
* the `onehot` read map at `Cout == Cin` -- the instrument that says *where* a
  value was read from -- at 2313, 3072, 3584 and 3585, at every rung except
  int4, whose index encoding does not fit a nibble.
* the matmul geometry itself, `197x1`: `K` 3072 `N` 768, `K` 768 `N` 2304 and
  3072, and the read map at `K = N` 3072 and 3584.
* 56x56 multi-tile to `Cin` 3584 (224 tiles), and stride 2 at `Cin`
  1792..4096 and `Cout` 2304..3584.

4096 measured clean everywhere and 8192 measured clean at fp16 k=1; the
constants sit at 3584 on this repo's usual principle -- a limit is what a real
model needs and the corpus reaches. A ViT-L/16 MLP at 4096 needs the constant
moved, not another measurement.

**`Counting` cannot be read at fp16 above `Cin` 2048**, and the failure looks
exactly like a channel-padding fault: its expected output is the channel count
itself, fp16 spaces integers by two above 2048, so `Cin` 2049 returns 2048 with
`max|diff| = 1` at every pixel. The ragged counts above were taken with the
fp32 output container (`fp16acc`) for this reason. Same trap as bf16's
nine-significant-bit ceiling.

**Depthwise did not follow the raise.** It constructs with
`out_channels == in_channels`, so `MAX_INPUT_CHANNELS` used to bind it, and
letting it keep riding that constant would have doubled the depthwise range on
dense evidence. `MAX_DEPTHWISE_CHANNELS` freezes it at the 1792 it already had;
the depthwise evidence itself stops earlier still (vendor corpus to 1344,
hardware exactness to 1536, matchers at 1344). Depthwise has its own
coefficient grouping and its own output writer, each of which has been wrong at
a shape the dense path was right at.

The sharing of the fp16 constants across the other 2-byte rungs is measured at
each width rather than argued from the element width alone:
`bf16_regression_matrix` (63/63), `int16_regression_matrix` (42/42) and
`fp16_accumulator_matrix` (57/57) each run `Cin` 512/1024/1344/1792/3584 at
k=1, `Cout` to 3584, ragged channel counts, 56x56 and 112x112 multi-tile, 5x5
and 7x7, and stride 2. int4 is `int4_regression_matrix_matches_oracle` (54/54)
and tf32 is `tf32_regression_matrix_matches_oracle` (53/53). All six live in
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
| Input row width per program | `(ceil(atoms/4) - 1) * in_cols <= 2047` for NC1HWC2 input (89 at fp16 `Cin` 768, 292 at 256, 37 at 1792, 18 at 3584; unbounded at one slab, `Cin` <= 32) | The CBUF's 11-bit entry-slab base, `MAX_ENTRY_SLAB_BASE`; measured exact at 2048 across fp16/bf16/int16/tf32/int8. `Shape::max_tile_input_width` bounds it and the planner splits columns. A row that does not fit its data grant is refused too (capacity 0), no longer forced through |

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
| `M` (conv width at height one) | 1..=2047, measured end to end at 2047 by `tools/e2e_matmul_regression.py` (exact) as well as in the compiled matcher; 1..=296 measured in the HAL | `CNA_DATA_SIZE0.datain_width` is 11 bits. The vendor FC sweep covers 1, 2, 7, 16, 32 (three CBUF splits); above the row-width limit above the planner splits column tiles -- see below |
| `K` (conv `Cin`) | 1..=3584 | `MAX_INPUT_CHANNELS`; measured 512, 1024, 1344, 1792, 2048, and 2304..4096 in 2026-09-06's sweep |
| `N` (conv `Cout`) | 1..=3584 | `MAX_OUTPUT_CHANNELS`; measured 64, 512, 1001, 1792, 2048, and 2304..4096 |

MobileNetV2's classifier, `M=1 K=1792 N=1001`, is exact under both patterns and
again with the fp32 accumulator kept. `K = 1792` is why `MAX_INPUT_CHANNELS` was
raised from 1344 on 2026-09-04: the geometry that carries it -- a 1x1 spatial
"image" -- is not one any convolution sweep had run.
`fc_matmul_ladder_matches_the_fc_lowering` keeps the regression's cases
identical to what `fc::Shape::as_conv_shape` actually builds, so the ladder
cannot drift into measuring its own geometry.

`K` and `N` reached **3584** on 2026-09-06 with the channel ceilings, measured
at this geometry as well as the convolution one: `197x1` with `K` 3072 `N` 768,
`K` 768 `N` 2304 and 3072, and the `onehot` read map at `K = N` 3072 and 3584.
End to end, `tools/e2e_matmul_regression.py` runs `matmul_k_n_ceilings`
(`M` 8, `K` = `N` = 3584) and `matmul_vit_mlp` (`197x768x3072`) through
compiled modules on the board, both bit-exact against the CPU arm.

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
3072 exceeded the 1792 channel caps of that day -- the 2026-09-06 raise to 3584
admits all three, taking the model to **48 of 333** dispatch sites, every
matmul in all twelve encoder layers). Against the `--no-offload` build on
`planck`, same input: max|err| 0.0014 on logits of magnitude 6.8, top-5
identical, **3.87 s vs 4.06 s per inference** (`iree-benchmark-module`, 3
repetitions each) -- the first configuration in this repo faster than its
like-for-like CPU arm, by 5%. With the 3584 ceilings and all 48 sites the
same comparison is **873 ms vs 4283 ms** (7 repetitions, medians, arms measured
back to back), `max|err|` 0.0041 with identical top-5, and Qwen3-0.6B's prefill
goes 56 -> 196 sites for 30.3 s against 49.3 s. ROADMAP.md carries both
tables. Read with [[planck-measurement-environment]]'s caveats; it is one input
and one core allocation.

## Pooling

| Property | Tested limit |
|---|---|
| Extents (H, W, C) | 1..=8192, the PPU's 13-bit N-1 range |
| Kernel, directly programmed | 1..=8 per axis (`MAX_DIRECT_KERNEL`); a 16x16 window is rejected by the hardware |
| Kernel, matched from a model | 2..=8 for both methods. For the average the floor is forced -- its reciprocal is `fp16(65536/k)` and `k=1` needs 65536, past fp16's 65504 ceiling. For max `k=1` is programmable and excluded anyway: a 1x1 stride-1 max pool is an identity, and claiming it would spend a dispatch, a pack and a compaction to copy a tensor |
| Kernel or stride, register range | 1..=16 |
| Padding | 0..=7, the PPU's 3-bit range. Every matched executable bakes 0: model-level padding arrives as a separate `tensor.pad` the CPU runs |
| Stride, matched from a model | avg 1; max and min 1 and 2 -- the executable bakes it, one per value. Stride 2 is measured against the oracle in `pooling_oracle_hw.rs`, for the average too (`avg 2x2s2`), so the average's stride-1 ceiling is a missing executable rather than a missing measurement -- the one remaining asymmetry in these matchers. Nothing above 2 is measured at any method |
| Method, matched from a model | avg, max and min -- all three the PPU has. `PoolingMethod::pad_fill_value` has no measured identity for min at any precision, and none for max at int8, but an *unpadded* pool never reads that field and every matched executable bakes zero padding. The driver derives `padded` from those baked fields alone, so tiling cannot reintroduce it |
| Layouts, matched from a model | avg NHWC and NCHW; max NHWC and NCHW; **min NHWC only** -- linalg defines `pooling_nchw_max` but no `pooling_nchw_min`, so there is no NCHW min op to claim. `pooling_*_unsigned` is unclaimed at every method: it is the unsigned-integer reduction and this path is f32 in, fp16 on the hardware |

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

### int8 requantisation rounds half away from zero

`DPU_OUT_CVT` computes `(accumulator * SCALE) >> SHIFT`, and on an exact half it
rounds **away from zero**: `0.5 -> 1`, `1.5 -> 2`, `-0.5 -> -1`, `-1.5 -> -2`.
Measured by `tests/conv_requant_tie_rule_hw.rs`, which sweeps every `i8`
accumulator through the datapath at two shifts and classifies all 192 exact
ties; round-half-up and round-half-to-even each miss exactly half of them, and
the 320 non-tie accumulators are exact, which is the probe's validity gate.

Two things that look like they answer this and do not. `DPU_OUT_CVT_SHIFT`'s
`cvt_round` field documents `0 = odd-in-even-not (round-half-to-even)`, and the
driver has always left it 0 -- but setting it changes no tie at all, so the
field does not select the rule here. And `../rockchip-npu-notes` measured
round-half-to-even on **RK3576**, scoping RK3588 as predicted rather than
probed; the prediction does not hold, so the two parts differ on this.

The consequence for a caller: a framework that requantizes with banker's
rounding (QNNPACK's *precise* mode, and anything matching it) will differ from
this hardware by one LSB on roughly `2^-(SHIFT+1)` of a surface. That is the
documented source of the `max|error| 1` in the requantized e2e fixtures, and it
is a difference rather than a defect.

## What a compiled model actually offloads

These are the enabled matchers -- the complete set of shapes that can leave the
CPU today. Everything else falls back silently and correctly.

The `conv + ReLU6` rows claim a convolution *and* the clamp that follows it,
and are the only rows that fuse anything: the clamp runs in the DPU's BN
stage, so the CPU dispatch it would otherwise cost disappears. Three bounds
on them that are not in the table:

- **The ceiling must be exactly 6.0.** `activation_cmp` is a static attribute
  on the executable target, and a transform matcher matches structure rather
  than constant values, so what pins the ceiling is that
  `rocket-fuse-conv-relu6` produces the canonical form for no other one.
- **The bias must be a per-channel broadcast.** It moves onto the BS plane,
  which is what puts it before the BN clamp; a convolution whose init is
  anything else is declined and keeps its separate clamp dispatch.
- **Stride 1 only**, so a strided convolution with a ReLU6 -- MobileNetV2's
  stem is the one in that model -- offloads through the plain rows and keeps
  its clamp on the CPU.

The `conv + pad 1` rows claim a convolution *and* the `tensor.pad` in front of
it, so the CNA pads instead of IREE materializing a full-tensor copy. Bounds
that are not in the table:

- **The pad must be symmetric, and that is the hardware.** `CNA_PAD_CON0` has
  `pad_top` and `pad_left` and nothing else, and each applies to *both* sides:
  `Shape::output_width` is `(w + 2 * pad_left - kw) / stride + 1`, matched
  against all 150 strided programs in the vendor corpus. There is no
  trailing-pad register, so `low[0] high[1]` -- ONNX's `auto_pad = SAME_UPPER`
  at stride 2 -- stays materialized.
- **Exactly 1**, spatial axes only, zero fill. Pad 2 and 3 belong to 5x5 and
  7x7, which no matcher claims.
- **Not combined with a fused ReLU6.** A convolution with both folds the
  activation and keeps its pad; no target claims the pair.

Measured on ResNet50 fp16: all 16 pad sites folded, `slow_memcpy` executables
7 -> 0, 230 -> 223 ms at `taskset -c 4-7` and 203.5 -> 195.5 at `0-7`, with
output identical to the materialized-pad arm to five decimals. MobileNetV2 has
almost nothing to give here -- 17 of its 18 pads feed depthwise convolutions
that stay on the CPU, and the one that does not is asymmetric.

Measured on MobileNetV2 fp16: 17 of 18 fusable sites, 146 -> 133 ms at
`taskset -c 4-7` and 122.5 -> 116.5 at `0-7`, with max|diff| against a
`--no-offload` CPU arm of 0.0184 where the unfused offload arm is 0.0173 and
argmax and top-5 are unchanged. The residual difference is the bias being
narrowed to f16 for the Conv2D ABI's bias binding.

| Op | Layout | Types | Kernel | Stride | `Cin` | `Cout` |
|---|---|---|---|---|---|---|
| conv | NHWC HWCF | f16/f16/f32 | 1x1 | 1 | 1..=3584 | 1..=3584 |
| conv | NHWC HWCF | f16/f16/f32 | 3x3 | 1 | 1..=1152 | 1..=1792 |
| conv | NHWC HWCF | f16/f16/f32 | 1x1, 3x3 | 2 | 1..=512 | 1..=512 |
| conv + ReLU6 | NHWC HWCF | f16/f16/f32 | 1x1 | 1 | 1..=3584 | 1..=3584 |
| conv + pad 1 | NHWC HWCF | f16/f16/f32 | 3x3 | 1 | 1..=1152 | 1..=3584 |
| conv + pad 1 | NHWC HWCF | f16/f16/f32 | 3x3 | 2 | 1..=512 | 1..=3584 |
| conv + ReLU6 | NHWC HWCF | f16/f16/f32 | 3x3 | 1 | 1..=1152 | 1..=3584 |
| conv | NHWC HWCF | i8/i8/i32 | 1x1 | 1 | 1..=3584 | 1..=3584 |
| conv | NHWC HWCF | i8/i8/i32 | 3x3 | 1 | 1..=1152 | 1..=512 |
| depthwise | NHWC HWC | f16/f16/f32 | 1x1, 3x3 | 1 | 1..=512 | = `Cin` |
| depthwise | NCHW CHW | f16/f16/f32 | 1x1, 3x3 | 1, 2, 3, 4 | 1..=512 | = `Cin` |
| depthwise | NHWC HWC | i8/i8/i32 | 1x1, 3x3 | 1, 2 | 1..=1344 | = `Cin` |
| matmul | -- | f16/f16/f32 | -- | -- | `M` 1..=2047, `K` 1..=3584 | `N` 1..=3584 |
| matvec | -- | f16/f16/f32 | -- | -- | `M` 1..=2047, `K` 1..=3584 | `N` = 1 |
| vecmat | -- | f16/f16/f32 | -- | -- | `M` = 1, `K` 1..=3584 | `N` 1..=3584 |
| avg pool | NCHW sum | f32 | 2x2..=8x8 | 1 | H/W/C 1..=8192 | -- |
| avg pool | NHWC sum | f32 | 2x2..=8x8 | 1 | H/W/C 1..=8192 | -- |
| max pool | NHWC | f32 | 2x2..=8x8 | 1, 2 | H/W/C 1..=8192 | -- |
| max pool | NCHW | f32 | 2x2..=8x8 | 1, 2 | H/W/C 1..=8192 | -- |
| min pool | NHWC | f32 | 2x2..=8x8 | 1, 2 | H/W/C 1..=8192 | -- |

All of these additionally require batch 1 and dilation 1.

All three matmul rows have an end-to-end differential behind them:
`tools/e2e_matmul_regression.py`, eight cases, all passing on `planck`
2026-09-05. Seven are compared **exactly** using ternary fixtures -- `{-1, 0,
1}` entries are exact in f16 and the sums stay inside its integer-exact range
-- which is what lets a contraction be gated bit for bit rather than under a
tolerance. The eighth is the ViT shape with realistic magnitudes, at
max|error| 0.0011.

`matvec` and `vecmat` have no matcher of their own: `rocket-expand-gemv-to-matmul`
raises them into `linalg.matmul` with a unit extent before the match loop, so
the matmul matcher, shim and executable claim them unchanged. `linalg.dot` is
deliberately not raised -- it reduces to a scalar, and a dispatch plus a weight
pack plus an output compaction to produce one number is not a trade worth
making.

Stride-3 and stride-4 *dense* fp16 matchers exist in the spec but are **not** in
the `foreach_match` list. Depthwise NCHW carries strides 3 and 4; depthwise NHWC
fp16 has no strided matcher at all.

Every accepted bound in that table has an immediately-adjacent rejected shape
asserted in a lit test -- `rocket_int8_match_boundaries.mlir`,
`rocket_fp16_match_boundaries.mlir`, `rocket_matmul_match_boundaries.mlir`,
`rocket_pooling_match_boundaries.mlir`,
`rocket_pooling_max_match_boundaries.mlir`,
`rocket_pooling_min_match_boundaries.mlir` -- so widening a matcher cannot silently
route a known-bad shape to the NPU, and tightening one cannot silently lose the
largest measured-good shape.

**An offloaded max or min pool is exact only up to f16.** The shim demotes to
f16 before the hardware sees the tensor, so on arbitrary f32 activations the
pool returns `f16(x)` rather than `x`. On VGG that is worth **0.077 max|error|
on the logits** (top-1 unchanged), against 0 for the same model's int8
convolutions, which are bit-exact. The gate's `max_pool_nhwc_dense` case is
the isolated version of that cost, at 0.00024 on a 2x2 window.

The pooling rows additionally have an end-to-end differential behind them:
`tools/e2e_pooling_regression.py` compiles all three methods twice and
compares on the board -- fourteen cases, all passing on `planck` 2026-09-05.
Max and min are compared **exactly** and measure exact (max|error| 0 across
both layouts where they exist, both strides, the 8x8 ceiling, a tiled width,
and pools sharing a command buffer including a min-then-max transition),
because a max or min pool returns one of its inputs unchanged and the
fixtures are f16-exact. The average carries genuine f16 error, 0.0057 worst
case at 49 taps, because the PPU's average is a multiply by `fp16(65536/k)`
that the shim multiplies back out.

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
furthest point that happened to pass. The dense channel ceilings sit at 3584
where 4096 also measured clean at every rung and 8192 did at fp16 k=1,
deliberately -- 3584 is a rung above the widest transformer shape in the
corpus.
