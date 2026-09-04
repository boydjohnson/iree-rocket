# Issues

Findings from a review of this repo against
[`rockchip-npu-notes`](../rockchip-npu-notes) (read at commit `fbedbfc`), plus
live state read off `planck` on 2026-09-03.

Those notes are independent reverse-engineering of the same silicon through the
same mainline `rocket` driver, from a different stack (a userspace matmul
library, a ggml backend, a TFLite delegate). Where they and this repo agree, the
agreement is worth something because the two derivations are independent. Where
they disagree, one of the two is wrong, and several of the disagreements land on
questions this repo currently has open.

Every claim below is tagged with how it was established here: **[verified]** =
checked in this tree or on the board during this review; **[notes]** = asserted
by rockchip-npu-notes with its own HW evidence, not re-measured here;
**[hypothesis]** = my inference from combining the two.

Severity: **S1** wrong results reach a user, **S2** wrong results reach a
developer or a measurement, **S3** performance, **S4** hygiene.

Updated 2026-09-04 with C8 and M4, both found while re-running the offload
A/Bs under the pinned governor and relocated IRQs that M1 and M3 asked for.

---

## C1 (S1) — RESOLVED 2026-09-03: there is no coefficient-per-channel limit at any kernel size. The accumulator writes the wrong output cube, and the readback models the wrong one to match.

Settled by diffing against `../rocket-userspace`, the C library the notes were
written from. Two earlier attempts got this wrong; both are corrected below.

### The two writers

`rocket-userspace/src/npu_regcmd.c`'s `gen_matmul_int8` is a HW-validated,
bit-exact int8 x int8 -> **int32** program. Its DPU output writer against this
repo's `Int8Accumulator` (offsets verified identical against
`rocket-userspace/include/npu_hw.h`; **every other DPU register matches**):

| field | register | rocket-userspace | this repo |
|---|---|---|---|
| **`mc_surf_out`** | `DPU_DATA_FORMAT` 0x4010 bit 3 | **0** | **1** |
| **`size_e_{0,1,2}`** | `DPU_BS_OW_CFG` 0x4050 | **7** | **1** |
| **`surf_add`** | `DPU_SURFACE_ADD` 0x40C0 | **`dataout_h*dataout_w * 8`, per task** | **16** |

`rocket-userspace/include/npu_dpu.h:120`:

> `mc_surf_out   DPU_DATA_FORMAT bit3   0=16B/pixel one surface, 1=2/4 surf serial`

These are **two different writers**. This repo is in the serial one, which is
why `size_e` and `surf_add` read as inert/unhelpful there — in the serial writer
there is no surface stride for either to describe. `surf_add` is also derived
per **task**, so no constant could have expressed it on a tiled plan.

### What each writer does

Measured on `planck`, one shape per process, `ROCKET_PAD_*` set, device HEALTHY
throughout, with a **layout scanner** that scores candidate address maps against
the oracle (validated: it returns 100% on the shipped model for the shipped
writer).

| | shipped (`mc_surf_out=1`) | reference (`mc_surf_out=0`, `size_e=7`, `surf_add=dataout*8`) |
|---|---|---|
| output cube | 32-channel blocks, surface-major (128-byte atoms) | **C2=4 surface-major** (16-byte atoms, 4 int32 lanes) |
| coverage | **truncates** past ~384 coefficient bytes/channel | **100% at every shape tested** |
| correctness | exact where it writes | exact everywhere, *once read back as C2=4* |

C2=4 is exactly the int32 output cube
`rockchip-npu-notes/encodings/tile-layouts.md` documents (`int8xint8 | int32 |
4 B | output cube C2 = 4`). The scanner puts it at **100.0%** and every other
candidate (C2=8/16/32, pixel-major) in the 32–38% noise band.

### There is no channel cap, at either kernel size

Reference writer, C2=4 readback, `Dense` pattern, all **100.0% of lanes exact**:

| shape | tiles | CBUF split | coef bytes/channel |
|---|---|---|---|
| 32² Cin 32 Cout 64 k1 | 1 | 1d/11w | 32 |
| 32² Cin 385 Cout 64 k1 | 2 | 11d/1w | 400 |
| 32² Cin 512 Cout 64 k1 | 2 | 11d/1w | 512 |
| 32² Cin 704 Cout 64 k1 | 3 | 10d/2w | **704** |
| 8² Cin 1024 Cout 64 k1 | 1 | 2d/10w | **1024** |
| 32² Cin 385 Cout **256** k1 | 2 | 8d/4w | 400 |
| 33² Cin 128 Cout 64 k1 (odd extent) | 1 | 5d/7w | 128 |
| 32² Cin 33 Cout 64 **k3** | 1 | 2d/10w | 432 |
| 32² Cin 256 Cout 64 **k3** | 2 | 7d/5w | **2304** |
| 16² Cin 256 Cout 64 **k3** | 1 | 2d/10w | 2304 |

**So `MAX_ACCUMULATOR_COEFFICIENT_BYTES_PER_CHANNEL = 384` describes where the
serial writer runs out of surfaces, not a hardware limit.** Every supporting
observation in `accumulator-per-channel-coefficient-limit` is consistent with
this: fully-accumulated written pixels, a clean prefix per tile, small
single-surface shapes passing, and `DPU_SURFACE_ADD` driving the write pattern
exactly.

### Does k=3 differ from k=1?

The question that produced the answer. **Tiling differs; the channel ceiling does
not.**

- **Tiling genuinely differs.** k=3 costs 9x the coefficient bytes per output
  channel, so the CBUF split swings toward weight banks (32² Cin 32: **1d/11w**
  at k=3 against a data-heavy split at k=1), and the halo makes
  `in_rows = out_rows + 2`. Same logical shape, different tile count and
  different rows per tile.
- **The channel ceiling does not differ, because there isn't one.** k=3 at
  C=2304 is exact; k=1 at C=1024 is exact. C is not the variable — it never was.
- A k=1/k=3 register diff at the same shape shows **only CNA registers move**
  (`0x1010`, `0x1030`, `0x1034`, `0x1038`, `0x1068` — weight sizes, kernel dims,
  padding). The DPU program is byte-identical. So nothing about the writer is
  kernel-dependent, which is why one readback fix covers both.

### Two corrections to my own earlier reports

1. **"`size_e` is inert, hypothesis refuted"** — wrong. I swept one register of a
   three-register geometry. In the serial writer that null was guaranteed and
   meant nothing. The repo's earlier `ROCKET_ACC_SURF_ADD` sweep failed the same
   way, independently.
2. **"the reference writer is bit-exact at k=1, regresses at k=3"** — also wrong,
   in both halves. Both used `OraclePattern::Counting`, which sets every input
   and coefficient to 1; at a 1x1 kernel with no padding that makes **every
   output lane the same constant**, so any permutation is invisible and
   "0 mismatches" only proved coverage. Re-run with `Dense` (varies in y, x and
   channel), k=1 under the reference writer scores 42232 mismatches on the
   shipped readback model and 100% on C2=4. The k=3 "regression" was the same
   layout mismatch, visible there only because padding makes k=3 vary spatially.
   **`Counting` cannot validate addressing on any shape whose output is
   constant** — that is a trap in the oracle harness worth a comment.

### DONE 2026-09-03 (items 1-3; the transform-spec cap raise is deliberately not done)

**1. The writer.** Dense `Int8Accumulator` now programs `mc_surf_out = 0`,
`size_e = 7`, `surf_add = out_cols * out_rows * DENSE_ACCUMULATOR_SURF_MULT (8)`
**per tile** — legal because `programs_with_staged_accumulator_output` gives each
tile its own contiguous scratch range, so a tile really is a standalone image.
Depthwise accumulator output is untouched and stays on the serial writer with
its 256-byte write atom: the change is measured on dense shapes only.

**2. The readback.** `Shape::output_channel_block_bytes` returns 16 for dense
accumulator output instead of 128. Nothing else needed changing — both
assemblers (`assemble_staged_accumulator_output` and the driver's
`compact_tiled_accumulator_output`) were already generic over `block_bytes` and
already implemented surface-major-within-tile, which *is* the C2=4 model at
16-byte atoms. `output_scratch_bytes`, `output_row_stride` and the driver's
`source_block_bytes` all derive from it.

**3. The guards.** Deleted `MAX_ACCUMULATOR_COEFFICIENT_BYTES_PER_CHANNEL`, its
doc block, `accumulator_coefficient_bytes_per_channel`, the 3x3-output/3x3-kernel
refusal, `validate_accumulator_output_shape`, `parity_padded_out_channels`,
`needs_output_parity_padding`, and the now-unused `known_bad_shapes_allowed`
escape hatch. `parity_padded_shape` survives as the identity, kept because the
driver, the executable format and the oracle harness are all routed through it
and it is the right place for any future physical/logical divergence.

The parity rule was **re-tested, not assumed**: under the C2=4 cube
`blocks_per_pixel = padded_out_channels / 4` and `padded_out_channels` is always
a multiple of the 32-channel granule, so the block count is a multiple of 8 —
even by construction, at every shape. That is pinned by
`accumulator_block_count_is_even_by_construction_so_parity_cannot_bind`, and
confirmed on hardware at the shapes the rule was originally fitted to.

**Hardware validation, shipped path, no env overrides, `Dense` pattern.** Every
shape 100% written, **0 mismatches**:

| shape | tiles | CBUF | coef bytes/ch | note |
|---|---|---|---|---|
| 32² Cin 32 / 128 Cout 64 k1 | 1 | 1d/11w, 4d/8w | 32, 128 | always worked |
| 32² Cin 385 Cout 64 k1 | 2 | 11d/1w | 400 | **was refused** |
| 32² Cin 512 Cout 64 k1 | 2 | 11d/1w | 512 | **was refused** |
| 32² Cin 704 Cout 64 k1 | 3 | 10d/2w | 704 | **was refused** |
| 32² Cin 385 Cout 256 k1 | 2 | 8d/4w | 400 | **was refused** |
| 33² Cin 128 Cout 64 k1 | 1 | 5d/7w | 128 | odd extent |
| 32² Cin 33 Cout 64 k3 | 1 | 2d/10w | 432 | **was refused** |
| 32² Cin 256 Cout 64 k3 | 2 | 7d/5w | 2304 | **was refused** |
| **3² Cin 64 Cout 32 k3** | 1 | 1d/11w | 576 | **was refused outright** |
| 9² and 33² Cout 32 k1 | 1 | | 64 | old parity shapes |

Existing accumulator gates on the board, all green: regression matrix 9/9,
k1 Cin boundary 5/5, Cout padding sweep 42/42, Cout shape interaction 24/24,
k1 supported Cin atoms 24/24. Non-accumulator gates unaffected: the cartesian
sweep and `cbuf_residency_boundary` pass, and the fp16/requantized register
programs are byte-identical (`0x4010`/`0x4050`/`0x40c0` unchanged at both).

Two gates failed on the first back-to-back pass and are **the documented
per-process flakiness, not this change**: `dense_geometry_regression_matches_oracle`
(1 of 22 — the 34x34 Cin 8 Cout 16 K3 fp16 case `npu-wedges-after-failed-job`
already names) went 22/22 twice with a 2 s idle gap, and
`int8_neutral80_one_hot_four_way_confirmation_matches_oracle` went 4/4, 4/4, 3/4
isolated. The latter is on the **requantized** path, whose register program this
change leaves byte-identical. Worth recording that `neutral80` is a fourth
member of that flaky set.

Host side: 156 lib tests + all workspace tests pass; clippy delta measured by
stash/pop is **zero**.

### Item 4 DONE 2026-09-03: caps raised, MobileNetV2 re-audited — and the caps were not what was blocking it

**The caps.** Both dense int8 Cin bounds raised to the HAL's
`MAX_INT8_INPUT_CHANNELS`: `@match_dynamic_conv2d_int8` 352 -> **512**,
`@match_dynamic_conv2d_3x3_int8` 32 -> **512**. Cout bounds unchanged (768 for
1x1, 512 for 3x3). 512 is the ceiling because above it the *channel padding*
rules are unmeasured and ConvPlan's CBUF split is known to diverge from vendor
captures for dense shapes above Cin 384 — both planning questions, unrelated to
the writer. `rocket_int8_match_boundaries.mlir` updated to 512-matched /
513-falls-back at both kernels and **passes**, which is the proof the bounds are
live. It also had a **stale case** unrelated to this work:
`dense_1x1_cout_513_falls_back` asserted a fallback that stopped happening when
the 1x1 Cout bound was raised to 768; it fails identically at HEAD. Corrected to
768-matched / 769-falls-back.

**The re-audit: 22 dispatch sites -> rocket, before and after. The cap raise
gains nothing on this model.** Not a disappointment to explain away — the shape
table says so directly. MobileNetV2-static-int8's 52 convolutions have dense 1x1
Cin values 24, 32, 48, 88, 136, 144, 192, 224, 288, then a jump to 448, 528, 816,
1344. Only **448** lands in the newly-opened 353..512 window, and its Cout is
1792, so `Cout <= 768` still refuses it. The 26 convolutions still on the CPU
are blocked by two *HAL* ceilings, not by matcher caps:

| blocked by | sites | shapes |
|---|---|---|
| `MAX_INT8_INPUT_CHANNELS = 512` | 16 dense 1x1 | Cin 528, 816, 1344 |
| `MAX_OUTPUT_CHANNELS = 768` | 1 dense 1x1 | Cin 448 -> Cout 1792 |
| depthwise matcher `umax = 512` (HAL ceiling is the same 512) | 9 depthwise 3x3 | C 528, 816, 1344 |

The only dense 3x3 in the model is the Cin 3 stem, and it is stride 2, which the
int8 3x3 matcher does not admit anyway — so raising that bound from 32 to 512
also gains zero here. Both raises are still correct and will matter on models
whose channel counts land in the window.

**Correctness end-to-end, and this is the real result.** On the board, against a
CPU-only aarch64 build of the same MLIR on identical input:

| build | max\|err\| vs CPU | top-1 |
|---|---|---|
| rocket, as shipped | 2.596e-01 | match |
| rocket, stride-2 stem kept on CPU | **0.000e+00** | match |

**Bit-identical.** All 21 int8 sites (17 dense + 4 depthwise) are exact against
the CPU reference end-to-end; the entire 0.26 is the one offloaded f32 stride-2
stem running at f16, exactly as this README documents (0.35-0.42 expected). That
is the strongest evidence yet that the C1 writer/readback change is right: not a
probe, a whole model.

**Performance, and the honest number: the NPU path is ~3x slower than CPU-only
on this model.** `iree-benchmark-module`, app pinned to cpu4-5 in every arm,
three interleaved passes, items/s:

| arm | pass 1 | pass 2 | pass 3 |
|---|---|---|---|
| rocket, governor idle | 1.33 | 1.49 | 1.64 |
| rocket, A76 held ramped | 1.41 | 1.66 | 1.51 |
| CPU-only, governor idle | 4.48 | 4.48 | 4.49 |
| CPU-only, A76 held ramped | 4.48 | 4.47 | 4.48 |

Two things worth reading off it. The CPU arm is **flat to two decimal places**
across governor state, exactly as `cpu-governor-and-offload.md` predicts for a
multi-threaded CPU inference. And the governor effect on the *rocket* arm is
only ~3% here, far short of the notes' 3.2x — because a continuously-fed
benchmark loop never lets the cluster idle (cpu4 read 408 MHz only on the very
first cold measurement, 2400 MHz thereafter). M1 remains real, but it bites
workloads with idle gaps between invocations, not a tight benchmark loop. Do not
quote the 3.2x for this shape of measurement.

The ~3x deficit is fully accounted for by things already in this file and not by
anything the cap raise touches: 22 of 153 dispatch sites on the NPU with the 26
heaviest int8 convolutions ineligible, the NPU at 200 MHz (**M2**), a NC1HWC2
pack/unpack per dispatch (**P2**), and one submit plus one blocking `PREP_BO`
per *tile* with the completion IRQ on an A55 (**M3**, **P3**).

### What would actually move MobileNetV2

In rough order of leverage, none of which is a matcher bound:

1. **Raise `MAX_INT8_INPUT_CHANNELS` past 512 and `MAX_OUTPUT_CHANNELS` past
   768.** Worth 17 dense sites. **This does not need CBUF-split work** — see the
   correction below; it needs corpus extension and board validation.
2. **A depthwise coefficient model.** Worth 9 sites, and the one item here that
   does need a code change: with the ceilings lifted, depthwise C=1344 at k=3 is
   refused by `streamed_weight_bank_preference` ("wants 13 CBUF banks"), which is
   the *dense* coefficient formula applied to a depthwise shape whose real weight
   footprint is `C · kh · kw` bytes total. Depthwise at C=528 and 816 plans fine.
3. **P2 (cross-op chaining) and P3 (per-tile submit/sync)**, which attack the
   per-dispatch tax rather than the dispatch count. At 22 sites the tax is
   currently paid 22 times for 22 convolutions.
4. **M2**, the 200 MHz clock, which is a straight ~1.43x on the device half.

### Items 1-4 DONE 2026-09-03: ceilings raised, every int8 convolution in MobileNetV2 now offloads

The four things a lift needed, all completed. (My earlier claim that it needed
CBUF-split work was wrong; see the correction folded into item 1 below.)

**1. Vendor corpus extended past 768.** Built with `build_vendor_fixtures.py`
(spike repo, `rknn-convert` on PATH, no board). Two new checked-in corpora and
a test, `conv_vendor_fixture_wide.rs`:

| corpus | cases | content |
|---|---|---|
| `conv_vendor_fixtures_wide.json` | 86 | dense: Cin sweeps to 1792 at Cout 64/448/1792; a coarse Cin×Cout grid; MobileNetV2's own widest 1x1 convs at their real 14x14 and 7x7 extents |
| `conv_vendor_fixtures_depthwise.json` | 63 | **first committed depthwise corpus**: C 64..1344 at extents 7, 14, 28 |

Results: **dense 83 agree, 2 differ, 1 refusal edge; depthwise 63 agree, 0
differ, 0 refusals.** Both differences are hardware-validated as correct —
28x28 Cin 704 Cout 64 k3 (ConvPlan 4/8 vs vendor 5/7) and 14x14 Cin 816 Cout 136
k1 (5/7 vs 10/2, one of MobileNetV2's own). Both give the weights at least as
many banks as the vendor, the safe direction, and both are 0 mismatches on the
board. **The CBUF split was never the blocker**: the four points
`MAX_OUTPUT_CHANNELS`' doc names as divergent now reproduce the vendor exactly,
because that doc predates the 2026-09-02 group-division fix.

**2. Board validation.** `accumulator_size_e_probe`, `Dense` pattern, one shape
per process, **0 mismatches everywhere**:

- k=1, 14x14, Cout 64: Cin 512, 576, 640, 704, 768, 896, 1024, 1152, 1280, 1344,
  1408, 1536, 1792, **2048** — single- and multi-tile.
- k=3, 28x28, Cout 64/448: Cin to **1152**, including the 1/11 splits at 1088
  and 1152, and the vendor-divergent 704 at Cout 64/128/256.
- Cout, 7x7 Cin 448: 768, 1024, 1280, 1536, 1792, **2048** — split flat at 7d/5w
  throughout, confirming the divergence is indexed by Cin, not Cout.
- All seven of MobileNetV2's blocked dense 1x1 shapes at their real extents.

**3. The ceilings and the assertion.** `MAX_INT8_INPUT_CHANNELS` 512 → **1344**;
`MAX_INT8_OUTPUT_CHANNELS` **split out at 1792** rather than raising the shared
`MAX_OUTPUT_CHANNELS`, because the evidence is int8-only — fp16 keeps 768 and
512, mirroring the existing `max_in_channels` split. `Precision::max_out_channels`
added alongside it. `conv_vendor_fixture_channels_768`'s hardcoded
`supported == 96` is now per-precision (fp16 96/48, int8 144/0) with the Cin 704
divergence as an explicit hardware-validated allowlist entry, so a *new*
divergence still fails.

Matcher caps: 1x1 int8 Cin **1344** / Cout **1792**; 3x3 int8 Cin **1152** — not
1344, because at k=3 the coefficient working set binds first and `ConvPlan`
*refuses* Cin ≥ 1216, which would reach the driver and panic rather than fall
back. Depthwise int8 **1344**.

**4. The depthwise coefficient model.** The streamed working set used the dense
product `kh · kw · Cin · 64`, which scales with C: depthwise C=1344 at k=3 asked
for 13 of the eleven grantable banks and was refused. A depthwise output channel
accumulates over exactly **one** input channel, so the contraction depth is 1
and the working set does not scale with C at all —
`Shape::streamed_contraction_channels`. Purely additive: a 128-case sweep
(k=3 and k=5, extents 7/14/28/56, C 32..512) gives byte-identical plans before
and after, and the new depthwise corpus agrees with the vendor 63/63.

### The re-audit, and the result that matters

| build | rocket dispatch sites | dense | depthwise | stem |
|---|---|---|---|---|
| before | 22 | 17 | 4 | 1 |
| caps raised | 39 | 34 | 4 | 1 |
| + depthwise model | **48** | 34 | 13 | 1 |

**Zero int8 convolutions remain on the CPU.** And it is correct: against a
CPU-only aarch64 build on identical input, the 48-site build is **bit-identical**
(max\|err\| 0.000e+00) once the f32 stride-2 stem is kept on the CPU; as shipped
it is 2.596e-01, entirely that stem's f16 rounding, unchanged from the 22-site
build. All 47 int8 convolutions are exact end to end.

**But it is slower, and that is the finding.** `iree-benchmark-module`, app
pinned to cpu4-5, three interleaved passes, items/s:

| build | pass 1 | pass 2 | pass 3 |
|---|---|---|---|
| 22 sites | 1.42 | 1.50 | 1.22 |
| **48 sites** | **0.90** | **0.85** | **0.92** |
| CPU-only | 4.51 | 4.50 | 4.49 |

Offloading 26 more convolutions made the model **~35% slower**, and the reason
is in the audit: CPU dispatch sites went **131 → 209**. Those +78 are the
NC1HWC2 pack/unpack wrapper ops — one pair per newly-offloaded convolution. At
7x7 and 14x14 extents the convolutions are far too small to amortize a
per-dispatch host repack, which is exactly what `rocket-layout-repack-per-dispatch`
predicts and what the 512→960 depthwise experiment measured once before.

So the caps were worth raising — they are correct, validated, and they remove a
whole class of "can't offload this" — but **dispatch count is not the lever on
this model. P2 (cross-op chaining) and P3 (per-tile submit/sync) are**, and they
are now the only things between this and a net win. Until one of them lands, the
22-site configuration is the faster one to ship, which is a decision for whoever
owns the default, not something to bury.

### fp16 ceilings raised too, 2026-09-03 — and fp16 is the case where the NPU actually wins

Same treatment as int8, and one piece was already done: the wide and depthwise
corpora were **fp16-generated**, so the vendor evidence above 768 was fp16 all
along.

**Ceilings.** `MAX_INPUT_CHANNELS` 512 → **1344**, `MAX_OUTPUT_CHANNELS`
768 → **1792**. Matcher caps: 1x1 Cin 512→**1344** and Cout 528→**1792**;
3x3 Cin 512→**1152** (at k=3 the coefficient working set binds first and
`ConvPlan` refuses Cin ≥ 1216). `rocket_fp16_match_boundaries.mlir` rewritten
from a single Cout 528/529 pair to three matched/falls-back pairs and passes.
Both precisions are now 144/0 in `conv_vendor_fixture_channels_768`, with the
Cin 704 divergence allowlisted per precision (fp16 Cout [64], int8 [64, 128]).

**The 2026-08-28 "960 attempt" objection is gone.** That raise was reverted
because ConvPlan predicted 1/11 against the vendor's 6/6, 5/7, 4/8, 4/8 at Cin
576–768. The 2026-09-02 group-division fix reproduces all four exactly.

**Board, `ROCKET_ACC_PROBE_PRECISION=fp16`, 0 mismatches at every point:** k=1
14x14 Cout 64 at Cin 256…**1792** (one to five tiles); k=3 28x28 Cout 64 at Cin
512…**1152**; Cout at 7x7 Cin 448 for 528…**2048**, split flat at 2d/10w.

**Audit: 18 → 35 sites, zero dense convolutions left on the CPU.** The
depthwise ones stay, and *not* because of a channel cap: `RocketDemoteConvInputsPass`
deliberately excludes depthwise (reverted 2026-09-01, max\|err\| 3.5), so fp16
depthwise never reaches Rocket at all. That is a separate open bug, untouched
here. The fp16 depthwise matcher caps were therefore left at 512 rather than
raised into a path nothing can reach.

**Accuracy is fine.** Against a CPU-only aarch64 build: 18 sites max\|err\|
0.0127, 35 sites **0.0172**, top-1 and top-5 stable, against a top-2 logit gap
of 1.26. Two orders of magnitude below the int8 model's 0.26, because this model
has no quantization boundaries for f16 noise to cross.

**And here the NPU actually wins — at 18 sites, not 35:**

| build | pass 1 | pass 2 | pass 3 |
|---|---|---|---|
| **18 sites (pre-raise)** | **4.16** | **4.27** | **4.18** |
| 35 sites (raised) | 2.67 | 2.69 | 2.69 |
| CPU-only | 3.97 | 3.95 | 3.94 |

The 18-site fp16 configuration is **~6% faster than CPU-only** — the first
configuration in this repo that is a net win. Raising the caps then costs 36%,
and the mechanism is the same as int8's: CPU dispatch sites went **103 → 142**,
+39 pack/unpack wrappers for +17 convolutions.

So both cap raises are correct, validated, and currently **anti-optimizations**.
They are the right preparation for the layout-repack compiler pass — once a
Rocket/CPU split removes the per-dispatch repack, the wider caps are what let
the bigger convolutions ride it. Until then the fastest shipping configurations
are 22 sites (int8) and 18 sites (fp16).

## C2 (S2) — the requant oracle rounds half-away-from-zero; the hardware rounds half-to-even, and this repo's multiplier encoding makes ties reachable

**Two halves, both in this tree.**

`conv2d_oracle.rs:386` [verified]:

```rust
fn rounded_shift(value: i32, shift: u32) -> i32 {
    let half = 1i32 << (shift - 1);
    if value >= 0 { (value + half) >> shift } else { -((-value + half) >> shift) }
}
```

That is round-half-away-from-zero. `encodings/out-cvt-converter.md` [notes]
measured the tie rule over 40 exact ties at two shifts and both signs:

> `acc*SCALE >> SHIFT` rounds to nearest, and an exact half lands on the
> **even** side: 0.5 -> 0, 1.5 -> 2, −0.5 -> 0, −1.5 -> −2. ... banker's rounding,
> matching QNNPACK's *precise* requantization, and **not** the
> round-half-away-from-zero the ancestor IP's documentation specifies, nor the
> round-half-**up** that `(x + half) >> shift` gives and that every CPU model in
> this tree used to spell.

The notes scope this honestly: measured on RK3576, *predicted* for RK3588, and
they say so — *"no probe run there has separated truncation from round-to-even."*

**The second half is the part specific to this repo.** The notes argue ties are
unreachable in practice because the Mesa/QNNPACK derivation ends in
`MUL = ((bits>>9) & 0x7fff) + 1` with bit 14 forced, so `MUL` is always odd and
an odd multiplier moves an exact half off the tie. `Multiplier::from_ratio`
(`conv.rs:606`) does **not** use that derivation — it normalizes the mantissa
into `[2^14, 2^15)` and takes `scaled.round()`, which can and does land on even
values [verified]. `Multiplier::from_ratio(1.0 / 2^s)` returns
`scale = 16384, shift = 14 + s` — exactly the deliberately-chosen power-of-two
multiplier the notes' probe had to construct on purpose to reach a tie at all.
And `conv2d_oracle.rs:129` builds precisely that for the `Counting` and
`SelectorsAffine` patterns.

So on this stack, ties *are* reachable, on roughly `2^-(SHIFT+1)` of a surface,
and the model and the hardware disagree on half of them.

**Actions.**

1. Change `rounded_shift` to round half to even. It is a two-line change and it
   is right under either the notes' rule or the QNNPACK rule this hardware's
   scale derivation is copied from.
2. This repo can settle the RK3588 prediction the notes explicitly flag as open,
   because it already has a board-validated requantized int8 path
   (`requantized-int8-conv-path`). One probe: pick a scale making `MUL` exactly
   `2^14`, drive accumulators onto exact ties at two shifts and both signs,
   classify. That is a genuine contribution back to `../rockchip-npu-notes`.
3. Separately, consider adopting the QNNPACK `+1`/bit-14-forced derivation in
   `from_ratio` so the shipped path never sits on a tie, independent of which
   rounding rule wins. That is what the vendor emitters do.

---

## C3 (S1) — RESOLVED 2026-09-03: the hung-job dispatch guard is now in the tree, in both places

`npu-wedges-after-failed-job` states, as settled:

> **The clock is the discriminator, and it is now wired in.** `run_hardware_case_matrix`
> times every SUBMIT -> PREP_BO round trip and labels any failure over
> `DISPATCH_TIMEOUT_FLOOR` (150 ms)...
>
> `rocket-hal-driver` had the same blind spot and now carries the same guard:
> `queue_execute` times each individually fenced task and returns
> `IREE_STATUS_DEADLINE_EXCEEDED` past `HUNG_JOB_DISPATCH_FLOOR` (250 ms)...
> Before this, a hung job during a real inference silently produced a partial
> output buffer.

Neither symbol exists anywhere in the repo [verified: `grep -r` over all `.rs`
and `.sh`, excluding `target/`], `git status` is clean of tracked modifications,
and `queue_execute` (`rocket-hal-driver/src/device.rs:1426`) contains no
`Instant`/`elapsed` at all — it calls `prep_bo` with a flat
`DISPATCH_COMPLETION_TIMEOUT_NS = 10s` and checks only the ioctl return.

The underlying hazard the memory correctly root-caused is therefore **still
live**: the watchdog resets the core and signals the job fence *with an error*,
`PREP_BO` waits on the `dma_resv` fence, and a fence signaled with an error is
still signaled — so `PREP_BO` returns success and `iree-run-module` reads a
half-written output buffer with no indication anything went wrong.

Fix: re-implement the guard. Time each `submit` → `prep_bo` round trip and
return `IREE_STATUS_DEADLINE_EXCEEDED` past a floor. The measured separation is
enormous and independently corroborated: healthy dispatches under 3.4 ms, a
watchdog-killed one 507–534 ms (this repo's own measurement), and
`perf/pool-completion.md` independently reports the same 500 ms `JOB_TIMEOUT_MS`
+ scheduler tick floor for the same driver.

See also C7 — this is one of several instruments the memories describe that were
never committed.

### Resolution

Both halves are implemented and measured, prompted by a real instance:
`fp16_accumulator_matrix_matches_oracle` failed 3/3 under
`cargo nextest run --release -j1 -- --include-ignored` while the other 361
tests passed, and it was this hazard rather than a shape result.

* `run_hardware_case_matrix` (`tests/conv2d_oracle_hw.rs`) times each
  `SUBMIT` → `PREP_BO` round trip and labels a failure past
  `DISPATCH_TIMEOUT_FLOOR` (150 ms) as `DEVICE TIMEOUT, not a shape result`,
  counted and named in the summary but excluded from the verdict.
  `ROCKET_STRICT_DISPATCH=1` makes them fail instead;
  `ROCKET_DISPATCH_TIMES=1` prints every dispatch so the floor can be
  re-measured rather than trusted.
* `queue_execute` (`rocket-hal-driver/src/device.rs`) times each fenced task
  and returns `IREE_STATUS_DEADLINE_EXCEEDED` past `HUNG_JOB_DISPATCH_FLOOR`
  (250 ms) with one stderr line pointing at dmesg, so a killed job stops
  reaching the caller as a plausible-looking output buffer.

Floors chosen from measurement, not carried over: healthy dispatches are
3.13 ms at worst across MobileNetV2 fp16's 54 real dispatches and 58.5 ms at
worst across every hardware ladder in this repo (226x226, 28 tiles), against
~500 ms for a killed job. Verified: 12 consecutive clean `iree-run-module`
runs with no false positive, and 10 back-to-back harness runs where the hang
struck 5 times, was labelled every time, and left the gate green.

---

## C4 (S3, blocks P4) — multi-task jobs were written off on evidence that contradicts source-confirmed mainline behaviour

`rocket-hal-driver/src/device.rs:1563` [verified]:

> The mainline driver's IRQ-mediated transition between tasks in one
> `drm_rocket_job` is not reliable on RK3588: task 0 completes correctly, but
> every later split leaves its output rows untouched.

So every tile of every dispatch is submitted as its own single-task job with its
own blocking `PREP_BO`.

`encodings/cbuf-reuse.md` reads the same driver the other way [notes,
source-confirmed against v7.1, which is the kernel on `planck`]:

> Mainline `rocket` gives that for nothing: `rocket_job_handle_irq()` programs
> the next task of the **same** job and only signals the done fence once
> `next_task_idx` reaches `task_count`, so `core->in_flight_job` holds the core
> for the whole sequence.

and `encodings/regcmd-task-model.md` classifies gapped multi-task jobs as
**safe for all dtypes** (N kicks, N IRQs, one fence), reserving the integer
breakage for *contiguous chaining* (one kick), where the int32 CACC clears per
kick rather than per task.

Two candidate confounds for the repo's observation, and I ruled out the first:

- **The `rocket_batch_submit` kernel param.** The notes warn it is global, and
  that with it on, a *gapped* multi-task job mismatches the kernel and task 0
  streams into the gap — which is the repo's exact symptom. Not it:
  `/sys/module/rocket/` on `planck` exposes no `parameters/` directory at all
  [verified].
- **Incomplete per-task regcmd.** The notes' delta-regcmd probe produces
  precisely "task 0 correct, later tasks leave output untouched", caused by a
  task writing an incomplete ping-pong producer group. This repo emits a full
  self-contained program per tile with `S_POINTER = 0xE`
  (`conv.rs:3563`) [verified], so it should not be this — but the two failure
  descriptions match well enough that it is worth re-deriving rather than
  assuming.

Worth retesting, because it gates P4 (the CBUF reuse bits require an
uninterrupted job) and it removes one submit ioctl + one blocking fence wait per
tile.

---

## C5 (S2) — LUT tables carry `q = 0` entries at exactly the inputs models hit

`encodings/dpu-lut-activation.md` QUIRK 4 [notes]:

> A **zero-valued LUT table entry** trips a decode fault in the output
> converter: it emits a constant **~4.0**, not 0. ... **Fix: floor every
> shifted-table entry to `q>=1`.**

Scanning `iree-rocket-hal/src/rocket/lut_tables.rs` [verified] — `q = 0` entries
and where they sit:

| table | zero at index | what input that is |
|---|---|---|
| `TANH_LE` / `TANH_LO` | 512 / 0 | x → 0⁻ and x = 0 |
| `ERF_LE` / `ERF_LO` | 512 / 0 | x = 0 |
| `SQUARE_LE` / `SQUARE_LO` | 510–512 / 0–2 | x ≈ 0 |
| `SQRT_LE` / `SQRT_LO` | 0 / 0 | x = 0 |
| `RSQRT_LE` | 0 | x = 0 |
| `LOG_LE` / `LOG_LO` | 0 / 128 | x = 0, log(1) = 0 |

`SIGMOID_*`, `EXP_*`, `RECIPROCAL_*` are clean.

**Not confirmed as a live bug here**, and the counter-evidence is in this tree:
`lut_standalone_tanh_matches_oracle` drives fill = 0 (real input 0.0) and
asserts zero mismatches against the oracle. If tanh(0) were coming back as ~4.0
it would saturate and that assertion would fail. So either the quirk does not
reach this repo's int8-output LUT configuration, or the vendor-captured tables
decode differently from the notes' `build_lut_shifted` tables. The notes
themselves scope QUIRK 4 to the shifted-table build and flag the sigmoid/tanh
deep tail as *"flagged, not chased."*

What is genuinely untested here:

- The **fp16-output** LUT configuration, if this repo ever uses one.
- The **deep tails** (`TANH_LE[512]`, `SQUARE`, `LOG_LO[128]`), which no current
  test drives.
- **QUIRK 2**, a separate mux glitch: within ~±0.0015 of exactly 0, signed-output
  kinds emit a discrete `+128` spike. tanh, erf and log are all signed-output.
  The notes warn a sparse-linspace gate steps straight over the band and that
  only dense random sampling finds it. `lut_standalone_tanh_matches_oracle`
  drives a handful of discrete fills, so it would not see this.

Action: add a dense sweep near 0 for every signed-output kind, and drive the
tails at least once, before relying on the LUT path in a compiled model.

---

## C6 (S4) — `DPU_BS_OW_OP` is always zero, including depthwise

`conv.rs:3838` [verified]: `commands.push(zero::<DpuBsOwOp>())`, unconditionally.

`depthwise-conv.md` [notes, source-confirmed against Mesa `rkt_regcmd.c` and
HW-swept] lists it as one of six fields that must differ for depthwise:

> **`bs_ow_op = 0x80 − weight_zero_point`** (`DPU_BS_OW_OP`, so `128` for
> symmetric/zero-zp fp16 weights). ... the validated direct fp16 path bypasses
> BS and leaves it `0`, but the depthwise job needs the `128`.

This repo's fp16 depthwise is board-validated exact (`fp16-depthwise-exact-coverage`),
so this is a **recorded divergence, not a known defect** — the two derivations
came from different sources (RKNN captures here, Mesa there) and this one has HW
backing. Two reasons to keep it on the list anyway:

- It is one of the last un-reconciled register deltas between the two stacks on
  the depthwise path, and `fp16-depthwise-int8-mix-corrupts` is still open.
- The repo also sets `od_bypass = 1` for fp16 where Mesa sets it to 0 for
  depthwise — the same divergence in the same register. Both are the BS/OW
  stage. If the mix bug is ever traced to BS state, this is where to look.

---

## C7 (S2) — several instruments the memories describe as shipped do not exist

`grep -r` over `.rs`/`.sh`/`.py`, excluding `target/` [verified]. Present:
`ROCKET_PROBE_ONLY`, `ROCKET_PROBE_RESUME_AT`, `run_hardware_case_matrix`,
`ROCKET_DUMP_PRECISION`, `accumulator_written_lanes_probe`,
`accumulator_written_region_map`, `accumulator_per_channel_threshold_probe`,
`MAX_ACCUMULATOR_COEFFICIENT_BYTES_PER_CHANNEL`.

Also present, and I initially got this wrong: **`ROCKET_PAD_INPUT` / `_WEIGHTS` /
`_BIAS` and `ROCKET_POISON_*` do exist.** They are built with
`format!("ROCKET_PAD_{which}")` (`conv2d_oracle_hw.rs:83`), so a literal grep
for the full name finds nothing. Corrected after using them during the C1 work.
Worth remembering as a search hazard in this file generally.

Genuinely absent:

| named in a memory as built | actually in tree |
|---|---|
| `accumulator_truncation_anatomy` | no |
| `ROCKET_ANATOMY_SHAPES` / `_REPEATS` / `_PRECISION` / `_BANKS` / `_CANARY` | no |
| `ROCKET_TILE_LIMIT` | no |
| `ROCKET_PAD_OUTPUT` | added during C1 |
| `ROCKET_IOVA_LEDGER` | no |
| `DISPATCH_TIMEOUT_FLOOR`, `HUNG_JOB_DISPATCH_FLOOR` | no |
| `tools/npu_hang_survey.sh` | no |

The measurements those tools produced are recorded in detail and are probably
sound — the C1 work reproduced the `accumulator-per-channel-coefficient-limit`
numbers exactly (6144 + 512 written bytes at `@0` and `@229376`, 63884
mismatches), which is good evidence the vanished harness was faithful. The tools
themselves were built in a working tree that was never committed. The cost is
that the memories instruct a future session to reproduce findings with
instruments that are not there, and one of them (C3) is a correctness guard the
memory believes is protecting production.

Either recommit them or amend the memories to say they were lost. I have amended
the memories that made load-bearing claims.

---

## C8 (S2) — the int8 offload path hangs the NPU after ~2 consecutive inferences, and every int8 timing ever recorded absorbed those hangs

Found 2026-09-04 while re-running the offload A/Bs, and found only because C3's
guard now exists to refuse the result.

Both int8 builds — 22 NPU sites and 48 — abort under `iree-benchmark-module`
with `HUNG_JOB_DISPATCH_FLOOR` firing at **505–532 ms** [verified]. Both fp16
builds run the identical harness with **zero** hangs across 34 and 19
iterations. Single `iree-run-module` invocations of the same int8 module are
**8/8 clean**. The discriminator is *repeated invocation in one process*, not
the shapes:

| `--benchmark_min_time` | iterations | hangs |
|---|---|---|
| 0.001s / 0.05s | 1 | 0 |
| 0.5s | 2 | 0 (sometimes aborts) |
| >= 1s | aborts, or 4 iterations with 1 hang | 1 |

Stochastic past ~2 inferences, matching the order-dependence in
`npu-wedges-after-failed-job`. **The wedge crosses process boundaries**: a
single-shot `fp16.prerise` run that had just completed 34 clean benchmark
iterations hung immediately after the int8 arms hung.

**This retroactively invalidates the int8 performance numbers.** They were taken
before C3's guard existed, so hung dispatches were absorbed silently. Against a
CPU-only int8 build measured at 131–133 ms:

    22 sites  1.42 items/s =  704 ms/iter   ~= baseline + one ~510 ms hang
    48 sites  0.90 items/s = 1111 ms/iter   ~= baseline + two

So **"more offload is worse" plausibly measured more hangs, not more offload
cost.** Consistent-with, not proven — the old CPU arm was also an NCHW build
(M4). Correctness is unaffected: the int8 path remains bit-exact.

The first two inferences are always right, which points at **state not reset
between inferences** rather than a shape or layout bug. Probably the same defect
as `fp16-depthwise-int8-mix-corrupts`, also a ~510 ms deterministic hang.

### Localised 2026-09-04: it is the `Int8Accumulator` -> `Fp16` transition, and it is not the program

Ablation, same model and pipeline, varying only which matchers may fire
(3 trials each, `--benchmark_min_time=2s`, canary run after every trial):

| arm | NPU sites | result |
|---|---|---|
| `int8.stemonly` (f16 stem only) | 1 | clean, 21 iterations |
| `int8.dw` (int8 depthwise only) | 4 | clean, 17–18 iterations |
| `int8.dense` (int8 dense only) | 17 | clean, 10–11 iterations |
| `int8.nostem` (dense + depthwise, **no** f16 stem) | 21 | **clean, 8–9 iterations** |
| `int8.stem_dw` (stem + depthwise) | 5 | clean, 15–18 iterations |
| `int8.stem_dense` (stem + dense) | 18 | **hangs 3/3** |
| `int8.prerise` (everything) | 22 | **hangs 3/3** |

So neither precision hangs alone, depthwise is not involved, and the minimal
mix is **one f16 dense conv plus the int8 dense convs**.

**Which dispatch hangs.** `ROCKET_DISPATCH_TIMES=1` with a precision tag: the
model is 28 tasks per inference — 6 tasks of the f16 stem, then 22 int8
accumulator tasks. Inference 1 completes all 28 at 0.15–5.21 ms. The hang is
always the **first f16 job after the int8 run**, within its first three tasks
(task 3 in 4/5 runs, task 1 in 1/5), at 519–530 ms. The reverse transition is
safe: every int8 job that follows the f16 stem inside inference 1 is fine, so
**the asymmetry is `int8acc -> fp16`, not a mix per se.**

An f16-only build is clean 9/9 in isolation, but it is *not* immune once the
device has been wedged: one f16 trial run immediately after a hanging int8 arm
logged a hang of its own. Measure f16 from a quiet device or the rate is
meaningless.

**It is not the program.** An FNV-1a hash over every task's program words is
byte-identical across all executions (`0xc310e9aa0aced33a`), with identical
regcmd IOVAs (`0x37d000..0x382000`) and identical in/out BO handles. The same
program that hung ran six times cleanly minutes earlier. The only variable is
device state left by the intervening int8 jobs.

**Three hypotheses eliminated**, each with a direct experiment:

- **Settling time.** Forcing the existing `DEPTHWISE_TO_DENSE_QUIESCENCE` 1 ms
  dwell before *every* dispatch does not help (still 3/3 hangs). The knob was
  confirmed live by its cost: 280 -> 301 ms on a clean arm, ~1.2 ms x 17
  dispatches.
- **GEM handle / IOVA recycling.** Leaking every per-tile regcmd BO, so no
  handle or IOVA is ever recycled, does not help (still 3/3).
- **A register the f16 program fails to re-initialise.** Both programs write
  **exactly the same 126 registers** — the set difference is empty in both
  directions.

So the stale state is not reachable through the registers either program
writes. The remaining candidates are hardware state the regcmd does not cover:
the accumulator path's write-back FSM, or CBUF/CACC state. Next step is to diff
the precision transition against `../rocket-userspace`, the known-good C
emitter, rather than sweep registers — the lesson from C1's method note.

Diagnostics used are committed (precision tag threaded into `DispatchJob`,
program hash, register-set dump with values, and the two probe knobs), rather
than left in a working tree the way C7's were.


### The `../rocket-userspace` diff, 2026-09-04: the conv register program is exonerated

Done as C1's method note prescribes -- diff the whole program against the
known-good emitter rather than sweep. Result is a **definite negative**, which
is worth as much as a hit: it removes the conv regcmd from suspicion entirely.

**1. The register *sets* are identical.** Extracting every `NPUOP(..., REG)` in
`gen_conv2d_task` (npu_regcmd.c:2220-2523) and resolving the names through
`npu_hw.h` gives **124** distinct registers. This repo's conv program writes
**124** plus the PC trailer. The set difference is **empty in both directions**
-- so is the difference between this repo's own fp16 and int8 programs
(126 vs 126, verified with `ROCKET_DUMP_REGSET`).

**2. The one value divergence is load-bearing here, and is not a defect.**
`DPU_RDMA_FEATURE_MODE_CFG` (0x5044) differs on two fields:

| field | this repo (fp16) | this repo (int8) | `gen_conv2d_task` |
|---|---|---|---|
| BURST_LEN [14:11] | 15 | 15 | 15 |
| MRDMA_DISABLE bit4 | 1 | 1 | 1 |
| MRDMA_FP16TOFP32 bit3 | 0 | 0 | `fp32tofp16_en` |
| IN_PRECISION [17:15] | 2 (fp16) | 0 | 0 |
| PROC_PRECISION [7:5] | 2 (fp16) | 0 | 0 |

The reference's header says those precision fields are *"Left at 0 (=int8) for
the plain-conv path because the whole RDMA block is off there"*, and this repo
arms neither MRDMA nor ERDMA -- so adopting the reference's spelling looked
obvious. **It is wrong here.** Clearing `in_precision`/`proc_precision` makes
this repo's *fp16* path hang on its own, 3/3, with no int8 in the model at all;
restoring them restores a clean 3/3. Adding the missing `mrdma_fp16tofp32_en`
on top of the existing fields is harmless (fp16 stays clean 3/3) but does not
fix the mix. So the two stacks genuinely diverge on this register and this
repo's spelling is the one its own fp16 path requires. Recorded, not "fixed".

**3. The only registers this repo never writes are the PPU block** -- 26 of
them, 0x6xxx/0x7xxx, written in the reference only by `gen_pool_fp16`. Nothing
here routes to pooling (P5), so neither precision touches them and they cannot
be what differentiates the two.

**What the diff did yield is the mechanism's precondition.**
`tests/regcmd_persist_rocket.c` establishes, deterministically on RK3588:

> the NPU register file is **NOT cleared between jobs/processes** ... the
> register file persists globally (not reset on job/process boundaries).

and the npu_regcmd.c header documents the exact failure signature being chased,
for a different cause:

> the DPU read-DMA engine stays armed waiting for a main-RDMA feed that never
> arrives: **the DPU never raises completion, and the job watchdog reports "NPU
> job timed out" with the output left untouched.**

Global register persistence is what makes a hang conditional on what ran
before, and it is why the same program hashes identically and still fails. But
since both programs write the same 124 registers with self-consistent values,
**the carried-over state is not in the registers either program writes.** That
leaves hardware state the regcmd does not address at all -- CBUF contents, CACC,
or the DPU write-back FSM -- and those are the next place to look, not the
program.



### CBUF and the rest of the program, 2026-09-04: also clean

Following the register diff, the remaining parts of the program were checked
with `ROCKET_DUMP_REGSET` (values) against both `../rocket-userspace` and this
repo's own two precisions. Nothing here is wrong either.

**The PC trailer and enable mask are identical.** Both precisions emit
`0x41:0x0000=0x0` then `0x81:0x0008=0x1d`. `gen_conv2d_task`
(npu_regcmd.c:2521) emits `NPUOP(OP_ENABLE, 0x1D, PC_OPERATION_ENABLE)` -- the
same word. The trailer cannot be the differentiator.

**CBUF bank programming is correct and is not precision-specific.**
`CNA_CBUF_CON0` decodes as the reference's
`(weight_bank << 4) | data_bank | reuse bits`:

| program | weight_bank | data_bank | reuse | fc_data_bank |
|---|---|---|---|---|
| fp16 stem | 1 | 11 | 0 | 0 |
| int8, 6 distinct splits | 3, 10, 5, 7, **1**, 8 | 9, 2, 7, 5, **11**, 4 | 0 | 0 |

The int8-only arm uses **six different bank splits within one inference**,
including `0x001b` -- weight 1 / data 11, byte-identical to the split the fp16
stem uses -- and runs clean 10-12 iterations, 3/3. So re-partitioning the CBUF
between jobs is demonstrably safe, and the fp16 stem's split is not even
distinctive. Reuse bits are 0 in every program (P4 is still un-attempted, as
recorded), and `FC_DATA_BANK[10:8]` is 0 everywhere, which is what P4's warning
requires.

**`CNA_CBUF_CON1` is a documented width divergence, not a defect.** The fp16
stem programs `data_entries = 9512` against the int8 convs' 21-168. The
reference masks this field to 13 bits (`& 0x1FFF`), which would truncate 9512
to 1320 -- but this repo's field is 15 bits on the strength of vendor corpora
that used bit 13 at 11,264 and bit 14 through 25,600 (`cna.rs` `data_entries`).
9512 is inside the observed vendor range, so it is in-family; the reference
simply never emitted a feature map big enough to find the wider field.

**Of the 40 registers whose values differ between the two precisions**, every
one is accounted for by shape (cube extents, strides, DMA sizes) or by
precision (`DPU_DATA_FORMAT`, `BS_MUL_CFG`, `OUT_CVT_SCALE`, the RDMA precision
fields above). None is a mode bit left set by one path and unread by the other.

**Net: the regcmd is exonerated end to end** -- coverage, values, CBUF
partitioning, and the enable trailer. Combined with the byte-identical program
hash across the working and hanging executions, no part of what this repo sends
the device explains the hang. What is left is state the regcmd does not
address, and finding it needs a different class of tool than program diffing:
hardware read-back between jobs, or a core reset inserted at the
`int8acc -> fp16` boundary to see whether it clears.

**One limit worth stating.** MobileNetV2 gives exactly one fp16 conv in the
int8 model (the stem), so "the *first* fp16 job after int8 hangs" and "*this
shape* hangs after int8" are not separated by any experiment run here. A model
with several fp16 convs among int8 ones would separate them, and is the cheapest
next probe.


---

## C9 (S2) — above 3x3 the conv path has two faults the fp16 capture sweep could not have seen: a `Cin` cliff that hangs at every width, and an int8 program that computes wrong values at every shape

Found 2026-09-04 while extending the datatype ladders past their first-light
shapes (bf16, int16, int4, tf32, fp16-f32out). Both are **guarded now** rather
than fixed: `large_kernel_max_in_channels` refuses what hardware does not do,
so a program that used to hang is a loud panic instead. Neither is reachable
from the compiler, whose matchers stop at 3x3.

The same sweep found two tf32 faults that *are* fixed, both also hangs rather
than wrong data, and both now board-validated over the whole ladder:
`Precision::out_channel_granule` (tf32 was the one rung whose granule was not
a multiple of 16, and every padded `Cout` at `8 (mod 16)` hung) and
`streamed_weight_bank_preference_for_group` (its coefficient working set was
calibrated at 1- and 2-byte widths and starved the 4-byte stream, so tf32 k=3
`Cin` 576-896 planned 5/7 and hung where the same *footprint* at fp16 plans
1/11 and is exact). Neither is in the table below.

### The `Cin` cliff [verified]

At 7x7, 9x9 and 11x11, a convolution is exact up to a per-width `Cin` and
**hangs the NPU above it** -- a watchdog kill at ~500 ms, `prep_bo` returning
success over an error-signalled fence, i.e. the C3 signature. Measured with
`dtype_boundary_probe`, `Selectors`, one shape per case:

| kernel | precision | exact | hangs |
|---|---|---|---|
| 7x7 | fp16 | `Cin` 64 | 72, 80, 88, 96, 128, 192 |
| 9x9 | fp16 | 64 | 96 |
| 11x11 | fp16 | 64 | 96 |
| 7x7 | tf32 | 32 | 48, 64, 96 |
| 7x7 | int4 | 128 | 160, 192, 224, 256, 288, 384 |
| 7x7 | bf16, int16 | 64 | — (ladder stops at the fp16 ceiling) |

Three things it is **not**: extent-dependent (the fp16 cliff sits between 64
and 72 at 8x8, 16x16 and 32x32 alike), `Cout`-dependent (7x7 fp16 at `Cin` 32
is exact at `Cout` 64, 128, 160, 192 and 256, up to a *larger* coefficient
footprint than the hanging shapes), or a CBUF-split artifact (9x9 `Cin` 64
takes 6/6 and 11x11 takes 3/9, and both hang one step later). The ceilings do
not reduce to one quantity either: `Cin * element_bytes` fits fp16 and tf32 at
128 bytes and misses int4 at 64; feature atoms fit those two at 8 and miss
int4 at 4.

**Why it was invisible:** `conv_kernel_size_hw.rs`, the only above-3x3
coverage, sweeps `Cin` 16, 24, 32, 48 and 64 -- it stops exactly at the last
value that works. 5x5 is unaffected at every width tried (fp16 and bf16 to
`Cin` 320, tf32 to 192).

### int8 above 3x3 [verified]

At 5x5 and 7x7, int8 returns **the same value in every output channel of a
pixel** -- `want 2 got -13`, ~14,600 of 16,384 elements wrong, max|diff| 30-43
-- at `Cin` 16, 32 and 64 alike, on a healthy device with a passing canary.
That is coefficients not reaching their channels, not a starved stream. No
int8 capture above 3x3 exists to say what the program should be, so both int8
rungs are refused there rather than guessed at.

The gate that used to hide all of this refused *every* non-fp16 precision above
3x3, on the grounds that the capture sweep was fp16. Half of that was
over-broad -- 5x5 and 7x7 take `demand_based_cbuf_partition`, which is stated
in bytes and shared with 1x1 and 3x3 at every precision -- and the other half
was masking a fault fp16 has too.

---

## M1 (S2) — `planck` runs `ondemand` with a 408 MHz A76 floor, which is the worst case the notes measured

Read off the board [verified, 2026-09-03]:

```
cpu0: gov=ondemand min=408000 max=1800000 cur=1800000
cpu4: gov=ondemand min=408000 max=2400000 cur=408000
cpu6: gov=ondemand min=408000 max=2400000 cur=408000
```

`perf/cpu-governor-and-offload.md` [notes] is exactly this configuration:

> A workload that hands its heavy arithmetic to the NPU spends that time
> blocked, with its threads off the run queue. A load-sampling CPU governor
> reads that as idle and drops the big cores toward their floor, and the half of
> the work that never left the host, the cube scatter and gather, then runs at
> that floor. **The NPU arm pays the penalty and the CPU-only arm does not.**

with measured penalties keyed to the A76 floor: **1.27x at a 1200 MHz floor,
3.2x at a 408 MHz floor**. `planck` is the 408 MHz board. Both big clusters were
sitting at 408 MHz when I read them.

This repo is unusually exposed to it, because the host share of an offloaded
dispatch here is the NC1HWC2 pack and the output compaction — pure
memory-bound host work, per dispatch (`rocket-layout-repack-per-dispatch`).

**Consequence for a conclusion already drawn.** `depthwise-channel-cap-960`
records the 512→960 cap raise as *"HW-validated correct but measured slightly
slower on MobileNetV2"* (161–163 ms → 165–170 ms), and reads that as the repack
tax not being amortized. That reading may well be right, but the measurement
cannot support it as taken: the arm that moved work onto the NPU is the arm the
governor penalizes, and the delta being explained (~3%) is far inside the
1.27–3.2x envelope the governor can move.

Action: pin the governor (or just raise `scaling_min_freq` on the A76 cluster)
for the duration of any A/B, restore it after, and record both
`scaling_governor` and `scaling_min_freq` alongside any NPU-vs-CPU number.
Re-run the depthwise-960 A/B under a pinned governor before treating that lever
as closed.

---

## M2 (S3) — the NPU is running at 200 MHz

`perf/clock.md` [notes]: the RK3588 compute clock `scmi_clk_npu` boots pinned at
200 MHz, one fifth of silicon max, because there is no NPU devfreq under
mainline `rocket` and the DT pins `assigned-clock-rates = <200000000>`. 200 MHz
is the vendor's idle `POWER_DOWN_FREQ`; nothing ever ramps it back up.

Consistent with the board [verified]: `/sys/class/devfreq` on `planck` contains
only `fb000000.gpu` — there is no NPU devfreq node.

Two consequences:

1. ~1.43x is sitting on the table (the notes' measured 600 MHz figure; 900 MHz
   buys nothing more and is dangerous). It requires a driver-side change —
   `clk_set_rate` inside `rocket_device_runtime_resume()`, after the power domain
   is up. **Both obvious shortcuts hang the box**: a DT `assigned-clock-rates`
   override hangs the boot, and a standalone out-of-tree `clk_set_rate` module
   at idle wedges the live SCMI firmware. The notes carry a working patch shape
   (`rocket-clk`, built as a module so recovery is `rmmod`).
2. It biases every offload decision in this repo toward "don't offload". At
   1/5 clock the device half of a dispatch is ~5x inflated while the host half
   (the pack/compact) is not, so a marginal layer looks worse than it is at the
   real operating point. Compounds with M1, which inflates the host half in the
   other direction.

---

## M3 (S3) — all three NPU completion IRQs are being serviced on cpu0, an A55 little core

From the board [verified]:

```
 82:  46744  0 0 0 0 0 0 0   GICv3 142 Level   fdab9000.iommu, fdab0000.npu
 83:  61146  0 0 0 0 0 0 0   GICv3 143 Level   fdaca000.iommu, fdac0000.npu
 84:   5276  0 0 0 0 0 0 0   GICv3 144 Level   fdada000.iommu, fdad0000.npu
```

Affinity is `0-7` on all three, and the effective delivery is the lowest CPU in
the mask = cpu0, an A55. `perf/iova-and-multicore.md` [notes] measured this:
binding the NPU IRQs to a big core takes the per-submit round trip 41.5 → 33.5 µs
(−19%), and co-locating the waiting thread on that same big core another
41.5 → 27–28 µs (−33% total).

This repo pays one submit + one blocking `PREP_BO` **per tile**, not per
dispatch (`device.rs:1570`), so this term scales with tile count, and a
multi-tile MobileNetV2 layer pays it many times over.

Free to test: `echo 6 > /proc/irq/{82,83,84}/smp_affinity_list`, no code change.

---

## M4 (S2) — every NPU-vs-CPU comparison in this repo used an NCHW CPU baseline that is 2.8x slower than the rocket pipeline's own CPU code

Measured on `planck` 2026-09-04 [verified], governor pinned to `performance`,
NPU IRQs on cpu6, app on cpu4-5, three interleaved passes, MobileNetV2
(`mnv2.fp16.mlir`), 224x224 input:

    fp16.cpu.nchw    3.93  3.93  3.95 items/s   <- the historical "CPU-only" baseline
    fp16.cpu        10.90 10.91 10.88           <- like-for-like CPU-only
    fp16.prerise     4.28  4.34  4.22           <- 18 NPU sites
    fp16.raised      2.71  2.68  2.82           <- 35 NPU sites

`rocket_conv2d_transform_spec.mlir` applies
`iree-preprocessing-convert-conv-to-channels-last` followed by
`linalg-specialize-generic-ops` **before** the matcher loop, so the whole model
is NHWC whether or not anything offloads. A CPU-only build made with plain
`iree-compile` never gets that and stays NCHW, and IREE's CPU backend is 2.8x
slower on NCHW MobileNetV2.

Isolated by bisection: deleting only those two `apply_registered_pass` lines
from the spec takes the rocket-pipeline CPU build from **10.94 to 3.94
items/s**, reproducing the historical baseline exactly. The dispatch names are
the tell — `matmul_like_528x14x14x88_f32` (NCHW) against
`matmul_like_14x14x528x88_f32` (NHWC), same 55 dispatches either way.

Passing `--iree-preprocessing-pass-pipeline='builtin.module(iree-preprocessing-convert-conv-to-channels-last)'`
to plain `iree-compile` does **not** reproduce it (still 3.96 items/s): at that
point the model is still torch-level and there are no linalg convs to
transpose. The spec gets it because it runs as a transform spec after linalg
conversion.

**Consequence.** The `fp16-channel-caps-raised` headline is retracted. The
18-site fp16 build is **2.5x slower** than CPU-only, not 6% faster, and no
configuration in this repo has ever beaten a like-for-like CPU build. The
correctness of the arms is not in question — both NPU arms agree with the NHWC
CPU arm to 0.025 max|err| with identical top-5, which they could not do if the
baseline were miscomputing.

**And M1/M3 turned out not to bind.** With the governor pinned *and* the IRQs
moved to cpu6, both fp16 NPU arms land within noise of their pre-fix values
(18 sites 4.22–4.34 against 4.16–4.27; 35 sites 2.68–2.82 against 2.67–2.69).
The notes' 3.2x needs idle gaps between invocations, which a benchmark loop
never provides. Pin them anyway — they are free and they remove the argument —
but the confound that actually mattered was the baseline's conv layout.

**Action.** Build the CPU-only arm with the *same* rocket-compiler pipeline and
a transform spec whose matchers cannot fire: rewrite every
`transform.iree.match.dim_bounds ... umin = N, umax = M` to
`umin = 999999, umax = 999999` (34 of them; verified to yield zero offload).
Never compare against a plain `iree-compile` build.

---

## P1 (S3) — one fd is one scheduler entity is one core

`rocket-hal-driver` opens `/dev/accel/accel0` exactly once
(`device.rs:335`) [verified], and `iree-rocket-hal`'s `submit_jobs` documents
the intent as:

> N *jobs* are independent work items the kernel scheduler can place on
> different cores -- that is the only lever userspace has

`perf/iova-and-multicore.md` §"Multicore: N fds for N cores, not a core-mask"
[notes] says that specific thing does not work, and says it was measured as the
first attempt:

> The driver creates **one `drm_sched` per core** but **one scheduling *entity*
> per fd**, and a DRM entity pins to one core while it has queued work. ... So
> **one fd with many jobs serializes onto a single core.** A first probe that
> submitted N jobs in one submit on one fd scaled 1.00 / 1.99 / 3.07x: it did
> *not* spread across cores (that 3.07 was an artifact of job batching, not
> multicore).
>
> **Driving N threads, each with its own fd/entity**, makes the kernel dispatch
> across all 3 cores: measured **39.5 -> 84.3 -> 116.1 -> 120.9 jobs/s** at
> 1/2/3/4 threads.

So the doc comment on `submit_jobs` is wrong about the mechanism, and the
`submit_jobs` entry point cannot deliver multicore however it is called. The
lever is N file descriptors, one per worker thread.

The notes add two constraints worth designing against: multicore only helps a
**multi-tile** conv (a single-CBUF-pass conv is one job on one core), and it is
**mutually exclusive with P2** — a cube must be one contiguous BO on one fd, so
a cube-chained sequence is inherently single-fd. Pick per shape.

Also worth knowing while you are there: each fd carries its own independent 4 GB
IOVA window, so N fds also multiply addressable device memory.

---

## P2 (S3) — cross-op chaining is HW-proven for fp16, and this repo's fp16 output cube is already the right layout

`rocket-layout-repack-per-dispatch` records the per-dispatch NC1HWC2 pack/unpack
as debt with no mechanism to avoid it, and correctly says no scaffolding exists.
`encodings/cross-op-chaining.md` [notes] supplies the hardware fact that makes
the fix legal:

> For an fp16 matmul whose output is the default fp16-narrowed cube
> (`fp32tofp16=1`), the **output cube and the input feature cube are the same
> layout**: both `feat_idx`, channel atom C2=8. ... The host does not need to
> de-tile `C1` to row-major and re-scatter it; the second matmul can read the
> first's output BO at the same IOVA.

proven two ways: A's raw output BO is byte-identical to `C1` de-tiled then
re-packed (`memcmp`, 0/4096 lanes differ), and B reading A's output BO directly
is bit-exact to the host round-trip.

**This repo is already in that configuration** [verified]:
`DpuOutCvtScale.fp32tofp16_en` is set whenever `quantization.is_none()`
(`conv.rs:3912`), and `Precision::Fp16::output_element_bytes()` is 2
(`conv.rs:476`), so C2 = 16/2 = 8 on both the input and the output side. The
aliasing precondition holds today, for free.

It is fp16-only, and the notes tabulate why: int8 (int32 C2=4 out vs int8 C2=16
in) and the fp32-accumulator fp16 output (C2=4) both mismatch. Element-wise ops
preserve the cube, so a `conv → activation → conv` run chains too.

Also carry the two costs the notes measured, so this is not oversold: matched
tiling pins the consumer's K-tile to the producer's N-tile, which fragments the
consumer's K-accumulation (they measured pack −30% but wait +14%); and the whole
chain is single-fd, so it forfeits P1. On their transform-bound encoder it was
net-positive; on compute-bound LLM prefill the ceiling was 2–3% and not worth
building. Which regime MobileNetV2's back-to-back 1x1 convs sit in is the thing
to measure first — and measure it under a pinned governor (M1), because the term
being removed is exactly the host-side term the governor penalizes.

---

## P3 (S3) — the full output BO is cache-synced once per tile, and a regcmd BO is allocated and mapped per tile

`perf/bo-sync-cost.md` [notes]:

> Both walk the BO's **entire** scatter-gather list and do per-page cache
> maintenance. There is **no offset/length** in the uAPI, so you always sync the
> whole BO, even if the NPU only touched a small live sub-region. So the sync
> cost is **∝ the allocated BO size (page count)**, *not* ∝ the bytes actually
> used.

They measured an oversized, repeatedly-synced output BO at ~20% of wall, and
right-sizing it at +11%.

In `device.rs:1570` [verified] the dispatch loop is, per tile:
`submit(...)` then `for out_handle in job.out_bo_handles { prep_bo(...) }`. So an
N-tile dispatch syncs **the whole output BO N times**, even though each tile
writes a disjoint slice of it. The repo's own comment notes that IREE packs
multiple bindings into one combined transient buffer — which makes the synced BO
potentially much larger than the dispatch's own output, and the notes' cost model
says the sync bills the whole thing.

Separately, the same loop does a fresh `OwnedBuffer::new` (CREATE_BO + mmap,
rounded up to 4096) plus a `fini_bo` for **each tile's regcmd**, on every
dispatch, on every inference (`device.rs:1509`). `regcmd-task-model.md` [notes]
recommends the opposite: cache the full regcmd per
`(tile geometry, precision, accumulate)` and patch only the address fields,
since the program is self-contained and address-only-different between tiles.

Both are cheap to fix and both compound with M3 (each tile is also an IRQ round
trip on a little core).

---

## P4 (S3) — the CBUF operand-reuse bits are never set

`conv.rs:3524` sets only `weight_bank` and `data_bank` on `CNA_CBUF_CON0`; no
call site anywhere sets `.weight_reuse()` or `.data_reuse()` [verified], though
both builders exist in `builders/cna.rs`.

`encodings/cbuf-reuse.md` [notes] measured **DATA_REUSE at +7% in-model** on
Gemma prefill, with the NPU `wait` bucket dropping 21% and everything else flat —
and argues the wait drop is itself the proof the bit is honored.

Three preconditions, all of which this repo would have to arrange:

- The run of tiles must be **one uninterrupted job**, so the operand is still
  resident. Blocked by C4 today.
- Only **one** bit can pay at a time (a 1-D task order makes only one operand
  identical to the previous task); pick by reuse depth.
- The tile loop must be ordered so the shared operand is adjacent.

The notes also carry a sharp warning about validating it: with the vendor driver,
which does not hold the core across tasks, reuse-on corrupted 29 of 120 runs with
*plausible* output, and a periodic-input gate reported clean for months. Validate
with aperiodic inputs and interleaved repeats, never a single run.

While in that register: `FC_DATA_BANK[10:8]` is live on the conv datapath and any
non-zero value corrupts the output. This repo leaves it 0 [verified] — correct,
and worth not touching.

---

## P5 (S3, latent) — a pool submit on this board costs ~507 ms and a core reset

`iree-rocket-hal/src/rocket/pooling.rs` exists and `command_buffer.rs` can
dispatch `UkernelShape::Pooling`, but nothing in
`rocket_conv2d_transform_spec.mlir` routes to it [verified] — no pooling matcher
exists, so no compiled model reaches this path today.

Before wiring GlobalAvgPool (the obvious next MobileNetV2 lever), read
`perf/pool-completion.md` [notes]:

> A pooling program on the RK3588 raises **no DPU interrupt**. A driver that arms
> only the DPU pair therefore masks off the only completion such a job can raise:
> nothing signals, `drm_sched` retires the job at its 500 ms deadline, and the
> core is reset. **Every pool submit costs ~507 ms and one reset.** The answer is
> correct throughout... Only the wall shows it.

Measured: 506–533 ms per pool call, twelve submits producing twelve
`NPU job timed out` lines, against 0.027 s for the same program on the vendor
driver.

`planck` is the affected configuration [verified]: mainline `rocket.ko` on
7.1.0-edge, no module parameters, well below the DRM interface 1.3 that carries
the `DRM_ROCKET_JOB_PPU_DONE` fix. And this repo's `DISPATCH_COMPLETION_TIMEOUT_NS`
is 10 s, so the submit would return *successfully* half a second later with the
correct answer — the failure mode is pure latency with nothing to attribute it
to. (C3's dispatch clock would catch it.)

Conclusion: keep pooling off the compiled path until either the kernel carries
the PPU-completion patch, or the pool is encoded DPU-fed
(`FLYING_MODE=1` + `DPU_FLYIN=1`) so a DPU completion arrives — which the notes
list as untested.

---

## D1 (S4) — the FC lowering here and in the notes use opposite geometries, and this one may be better

`iree-rocket-hal/src/rocket/fc.rs` maps, from a sweep of 160 RKNN-compiled ONNX
`Linear` models (666 programs) and HW-validated at M=7:

> M is the convolution width and the physical height is exactly one

`matmul-as-conv.md` maps the other way — M becomes the conv's spatial **height**,
width is 1 — and records two hard constraints that follow from it:

- **Feature height < 4 mis-computes at every dtype.** At height 1 the result is
  uncorrelated with the reference (cosine ~0.01–0.06). `M%4==0` is the real
  constraint; `M==1` only "works" because software pads it to 4.
- **The int8 CBUF bank-slack resonance fires only at `datain_width == 1`**
  (`cbuf-bank-slack.md`): int8 feature DMA over-reads by one bank and garbles the
  tail rows, and `IW>=2` shapes use a different descriptor and never resonate.

This repo's height-1/width-M mapping sidesteps **both**: its height is always 1
by construction rather than being M, and its `datain_width` is M, never 1. That
looks like a genuine advantage of the capture-derived mapping, and it is worth
feeding back to the notes — their M%4 padding may be avoidable by transposing
the mapping.

Caveat before claiming it: `fc.rs` is validated at M=7/K=16/N=32-33, a single
small point. If FC is going to carry real shapes, sweep it.

---

## D2 (S4) — the int8 CBUF bank slack was found twice, from two directions

`cbuf-bank-slack.md` [notes] prescribes `data_bank = min(fd_banks + 1, 11)` for
the int8 feature cube, from a `(Mtile, Ktile)` resonance with no closed form.

`Shape::data_bank_demand` (`conv.rs:1498`) [verified] arrives somewhere similar
by a different route: it bills the *rounded* `cbuf_atoms` rather than the exact
int8 channel count, with its own doc comment recording that billing the exact
count under-grants a bank at Cin 33..48, 97..112, 225..240 and every 64
thereafter, measured on RK3588 at Cin 48, 112 and 240 where the byte shortfall
predicts 3.88/3.88/0.12 lost rows and the hardware loses exactly 4, 4 and 1.

Same phenomenon, two independent derivations, both HW-backed. Reconciling them
is cheap and would either raise confidence in the CBUF planner or find a gap in
one of them — this repo's rule is width-dependent while the notes' is a flat +1,
so they must disagree somewhere.

---

## Suggested order

Revised 2026-09-04. C1 and C3 are landed and verified in the tree. M1 and M3
were applied on the board and turned out **not** to bind (see M4), which
demotes them; M2 is the only platform item with an open case. The re-audit they
were meant to enable found two larger things, and both are now at the top.

1. **C8** — the int8 offload path cannot complete a benchmark loop, and the
   first two inferences are always right. That is a state-reset bug, not a
   shape bug, and it currently blocks every int8 performance question. Likely
   the same defect as `fp16-depthwise-int8-mix-corrupts`.
2. **M4** — already established; what remains is to stop using the NCHW
   baseline. Cheap, and it is the precondition for any offload decision being
   meaningful.
3. **P3 → C4 → P4 → P1** — the dispatch-path cost stack, roughly in increasing
   order of work. This is where the offload deficit actually lives: a
   like-for-like CPU build is 2.5x faster than the best NPU configuration, so
   the per-dispatch and per-tile taxes are the whole game.
4. **C2** — small fix, plus a probe that settles a question the notes leave open.
5. **M2** — ~1.43x on the device half, but it needs a driver-side
   `clk_set_rate` and both shortcuts hang the box.
6. **P2** — the big structural one; measure the regime first.
7. **C5, C6, C7, D1, D2** — hygiene and reconciliation.

## Method note

Two things went wrong here repeatedly, and both are cheap to avoid.

**Single-knob sweeps of a multi-register geometry.** Three attempts missed C1
this way: the repo's `ROCKET_ACC_SURF_ADD` sweep, my `ROCKET_ACC_SIZE_E` sweep,
and the `surf_add` half of my first joint attempt. When a mode bit
(`mc_surf_out`) reinterprets what its neighbours mean, a null result from moving
one of them says nothing at all. Diffing against a known-good emitter is what
broke it open; no amount of further sweeping would have.

**Degenerate test patterns.** `OraclePattern::Counting` sets every input and
coefficient to 1, so any shape whose output is constant across pixels and
channels — every 1x1 kernel without padding — cannot detect a permuted layout,
only unwritten lanes. It reported "0 mismatches" for a writer that was putting
every lane in the wrong place. Use `Dense` (varies in y, x, channel) for
anything that tests addressing, and treat a 100%-pass on `Counting` at k=1 as
evidence of coverage only. Better still, score the layout explicitly:
`ROCKET_ACC_LAYOUT_SCAN=1` names the cube instead of leaving "wrong somehow".
