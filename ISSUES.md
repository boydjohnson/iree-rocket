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

This file is what is still open. [LIMITS.md](LIMITS.md) is the complement:
what the stack is measured to do, and which layer enforces each bound.

Trimmed 2026-09-05: issues that are settled were cut down to one entry each
in **Resolved** at the end, which keeps their IDs resolvable without keeping
their narratives. Everything above that section is open. Evidence a resolved
issue produced that open work still depends on was moved into the open issue
that needs it, not deleted -- M4's phase profile and dispatch-family counts now
live in P8.

---

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

## C10 (S2) — a height-one convolution is never tiled along its width, and past a `K`-dependent width the matmul lowering returns silently wrong data

**[verified]** on `planck` 2026-09-05 with `dtype_boundary_probe`, which runs
exactly `fc::Shape::as_conv_shape`'s geometry (width `M`, height 1, k=1,
pad 0). One shape per process, `Selectors`.

The matmul matcher caps `M` at 32, and the transform spec explains that cap as
bookkeeping: "M becomes the convolution *width*, which no constant bounds; 32
is where the ladder stops, so it is where this stops." That reads as an
invitation to raise it. It is not one -- above 32 the hardware returns wrong
values, and the boundary moves with `K`:

| `K` | last `M` exact | first `M` wrong |
|---|---|---|
| 1792 | 37 | 38 |
| 768 | 88 | 90 |
| 256 | 288 | 296 |
| 64 | 384 (no failure found) | -- |

`M x K` at the last passing point is 66,304 / 67,584 / 73,728 -- roughly
constant, which is the signature of a single feature row that stops fitting
the granted CBUF data banks.

These are **wrong values, never a timeout**, and the corruption grows with `M`:
at `K` 768 the first failures are 8 output channels wrong at every `x`
(`M` 90, 720 of 5,760 elements, several reading 0 where a coefficient should
have landed), and by `M` 128 every element of every channel is wrong. At
`K` 256 `M` 296 the first column is still exact and the corruption starts
further along the row. `Counting` cannot see any of it -- its output is
spatially constant -- which is why the ladder's own pattern choice matters
here; `Selectors` and `Dense` both catch it, and disagree on which shapes
above the boundary survive (`Dense` passes `M` 197 where `Selectors` fails it),
so neither alone bounds the fault.

**Why the planner does not stop it.** `ConvPlan` plans a height-one shape as
`tiles=1, in_rows=1` at every width tried, up to `M` 512. Row tiling has no
freedom at height one, and column tiling is not reachable:
`captured_column_partition` returns `Some` only for three hard-coded vendor
captures -- width exactly 256, height exactly 32, `Cout` exactly 64, fp16,
unpadded, at 9x9 or 11x11. Every other shape gets `None`. So nothing bounds a
matmul's feature footprint, and past the point where its single row stops
fitting, the program is emitted anyway.

**The capacity model does not predict the boundary either.** At the last
passing point the charged footprint is 81-90% of the granted data banks, so
`max_tile_input_rows_for_width_and_data_banks` believes every one of these
shapes fits, and its trailing `.max(1)` would force one row through even if it
did not. Whatever the missing overhead is, it is not in that formula. The bank
split is also non-monotonic in `M` at fixed `K` (`K` 768 takes 7 data banks at
`M` 16, 2 at `M` 32, 9 at `M` 197), which is worth understanding before
trusting any threshold derived from it.

**Not reachable from a compiled model today** -- the `M <= 32` matcher bound
contains it, which is why this is S2 and not S1. What it blocks is transformer
offload: ViT-B/16 wants `M` 197 at `K` 768, four times past that `K`'s
boundary. Raising the matmul `M` bound before this is fixed would route known-
wrong shapes to the NPU.

**What would fix it:** width tiling at height one. The machinery exists --
`plan_grid`, `balanced_column_widths`, and the `horizontally_tiled` register
path are all written and board-validated -- it is only the gate that is
hard-coded to three captures.

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

**Resized 2026-09-04.** The round trip was 53 ms per inference when this was
written; P6 item 3 found that most of that was the transforms running on the
little cluster plus a redundant zero pass, and it is now 26 ms (`compact` 18,
`pack.input` 8.4) against a 146 ms model. Still the largest thing in the
driver, and still worth doing — but the "measure the regime first" advice
above now has a second half: measure it *after* P6 item 3, or the case for
chaining will look twice as strong as it is. MobileNetV2 fp16 does have
adjacent NPU dispatch pairs (roughly 15 of 37 dispatches follow another NPU
dispatch immediately), so the mechanism has somewhere to apply here.

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

## P6 (S3) — the command-buffer `record` phase still runs on a little core

Items 1-3 are done (packed-coefficient caching, the classifier matmul's
per-inference re-narrowing, and the host-side transforms' core placement --
together 198 -> 146 ms on MobileNetV2 fp16); see **Resolved**. What they left
behind is one residual and one pointer.

Per inference after: `pack.input` 8.4 ms at 1023 MB/s (was 16 ms at 538),
`compact` 18 ms at 694 MB/s (was 38 ms at 330), `record` 15 ms, `outside`
88 ms, NPU 22 ms — 13.7% of wall. Logits bit-identical on both models, board
gate green.

**P2 is still open and still worth what it was**, but it is now worth 26 ms
per inference rather than 53, and the next person should read this item and the
M1/M3 entry under **Resolved** before quoting either number. The `record`
phase's 15 ms is also still on a little core: it runs at command-buffer record time, outside `queue_execute`'s
guard, and a guard per `dispatch` call measured as noise because 37
back-to-back set/restore pairs migrate the thread off the big cluster between
every one of them. One guard held across a whole command buffer's recording
is the way to get it.

---

## P7 (S2) — MobileNetV2 fp16's 17 depthwise convolutions stay on the CPU, and offloading them today makes the model 26% slower

They are the whole of P6's `outside` term: ten executables over 17 dispatch
sites, `112x112x48` down to `7x7x1344`, all `linalg.depthwise_conv_2d_nhwc_hwc`
at f32. They stay on the CPU for one reason —
`RocketDemoteConvInputsPass` deliberately excludes depthwise, so an f32
depthwise never becomes the f16/f16/f32 the matchers require and no depthwise
matcher can ever fire.

### The recorded reason for that exclusion does not apply to this model

The pass's scope comment, and `depthwise-f16-demote-breaks-model`, say
demoting depthwise makes MobileNetV2 wrong: max|err| 3.5 on the logits, top-1
incorrect on every input, with every isolation passing. That was measured on
**static-int8** — a command buffer mixing fp16 depthwise with int8 dispatches,
which is C8's territory, not a statement about depthwise.

Measured on the plain fp16 model, 2026-09-04, against a CPU-only aarch64 f32
build of the same MLIR:

```text
  37 sites (today)          max|err| 0.0192   top-1 and top-5 stable
  44 sites (+7 depthwise)   max|err| 0.0500   top-1 and top-5 stable
```

against a top-2 logit gap of 0.257, and byte-identical across five consecutive
single-shot runs with no hang. **Correctness is not the blocker for fp16.**

### It is slower

```text
  147/149 ms   37 sites
  185/188 ms   44 sites
```

Only 7 of the 17 match at all: the fp16 depthwise matchers cap at `Cin <= 512`
and ten of these are 528, 816 or 1344 (the HAL itself is exact to 1536, see
`depthwise-channel-ceiling-split`). Those seven cost 32.7 ms of driver time per
inference, of which 9.6 ms is the NPU:

```text
  op                                     total   rec  pk.in   npu  cmpct
  113x113x144->144 k3x3 s2 dw            10.13  2.46   3.19  2.85   1.22
  114x114x48->48   k3x3 s1 dw             6.12  1.76   1.26  2.04   0.90
  58x58x192->192   k3x3 s1 dw             6.10  1.37   1.11  1.71   1.72
  30x30x288->288   k3x3 s1 dw (x2)        5.08  1.13   0.97  1.52   1.19
  57x57x192->192   k3x3 s2 dw             3.70  0.86   1.11  1.03   0.51
  29x29x288->288   k3x3 s2 dw             1.54  0.38   0.44  0.45   0.15
```

Two costs on top of that, both of which only exist because these are the ops
being moved:

- **`quiesce` 7.4 ms per inference**, 1.06 ms x 7. That is
  `DEPTHWISE_TO_DENSE_QUIESCENCE`, the dwell after a depthwise completion and
  before a dense submit. A model whose depthwise and dense convolutions
  alternate pays it at every boundary.
- **`outside` *rises* 88 -> 118 ms.** Offloading these does not remove CPU
  work, it adds some: the matched depthwise needs explicit padding
  (`112x112 -> 114x114`), which IREE forms as its own CPU dispatch. The
  padding is why the shapes above read 113/114/58/30 rather than
  112/56/28.

So the arithmetic is: 9.6 ms of NPU work bought with 23 ms of driver host
time, 7.4 ms of hardware dwell and 30 ms of extra CPU padding. The
depthwise convolutions are the cheapest ops in the model per byte moved —
one filter per channel, no Cout reduction — which is exactly the profile that
loses to a per-dispatch layout round trip.

### What would change the verdict, in order

1. **P2's cross-op chaining.** These are the widest spatial extents in the
   model, so their round trip is the most expensive one there is: `pack.input`
   plus `compact` is 12.3 of the 32.7 ms. A depthwise sitting between two
   Rocket 1x1 convolutions is also the ideal chaining shape — producer and
   consumer both on the NPU.
2. **The pad.** Folding it into the dispatch (the driver already pads on the
   input packing path) removes the added CPU dispatch and its buffer.
3. **The quiesce dwell.** 7.4 ms is pure empirical caution; it is C8's
   neighbour and should be re-derived rather than kept at a millisecond.
4. **The `Cin <= 512` matcher cap**, last, because raising it only adds more
   of a currently-losing trade until the three above land. The HAL is exact
   to 1536.

To reproduce the 44-site build: add
`DemoteInputsToF16<linalg::DepthwiseConv2DNhwcHwcOp>` and
`...<linalg::DepthwiseConv2DNchwChwOp>` to `RocketDemoteConvInputsPass`, and
the `PromoteInputsToF32` counterparts to the promote pass. Two lines each, and
nothing else in the tree needs to change — the matchers, the executables, the
weight packing and the HAL are all already there and all already correct.

---

## P8 (S2) — the offload's cost is a flat per-dispatch tax that *parallelises*; the dispatches the offload adds are not it, and `taskset -c 4,5` doubles the deficit it reports

Measured on `planck` 2026-09-05 [verified], one MobileNetV2 int8 model
(`mnv2.int8.mlir`) through the rocket-compiler pipeline, governor
`performance`, NPU IRQs on cpu6, the post-C8 board binary, and every arm gated
on all three NPU cores reading `suspended` before it starts. This was started
as M4's lever 1 ("layout propagation ... this is the whole game") and it
refutes it.

### The law: cost is linear in offloaded convolutions and flat per convolution

Same model, same pipeline, only the matcher list changed, app on cpu4-5:

| arm | NPU sites | ms/inference | over CPU-only | per NPU site |
|---|---:|---:|---:|---:|
| `int8.cpu` (M4's like-for-like build) | 0 | 132 | — | — |
| `int8.dwonly` (depthwise matchers only) | 13 | 211 | 79 | **6.1 ms** |
| `int8.denseonly` (dense int8 matchers only) | 34 | 398 | 266 | **7.8 ms** |
| `int8.base` (everything) | 50 | 500 | 368 | **7.4 ms** |

`132 + 7.4 x sites` fits all four. The tax does not care *which* convolution:
a 7x7x1344 depthwise costs what a 112x112x144 1x1 dense costs. It also matches
`ROCKET_PROFILE`'s own `outside` figure exactly -- 7.46 ms averaged over 549
dispatches.

### Three things that should have removed it, and did not

Each is a real, working mechanism, verified in the IR; each is worth close to
nothing on the clock.

| change | dispatch sites | ms/inference (cpu4-5) |
|---|---:|---:|
| `int8.base` | 265 | 501–514 |
| + transpose propagation before the matcher loop | 263 | 498–501 |
| + `inline` on the `@call_rocket_*` wrappers | 249 | 478–495 |
| + int8 epilogue as `linalg.generic`, fused | **145** | 479–487 |
| epilogue **deleted outright** (wrong results, an upper bound) | 171 | 467–483 |

**Scope, stated precisely.** M4's lever 1 bundles two different mechanisms
under one name: the *compiler*-level NCHW/NHWC transposes stranded around each
opaque Rocket call (its "172 transposes and 39 memcpys"), and the *driver*-level
NC1HWC2 round trip between chained NPU dispatches (its "820 ms of compaction",
which is P2's cube aliasing). **Only the first is tested here.** The second is
untouched and still open — but the profile now bounds it: `compact` is 1.03 ms
of the ~10 ms an offloaded convolution costs, so P2 cannot be more than about a
tenth of the deficit even if it removes the round trip entirely.

1. **The transposes are free.** `iree-global-opt-propagate-linalg-transpose`,
   run inside the transform spec *before* the matcher loop claims the convs,
   really does make the whole activation chain NHWC:
   `broadcast -> transpose -> conv -> transpose -> requant(NCHW)` becomes
   `conv -> requant(NHWC)`, and `elementwise_transpose_144x12544_i32xi32xi8`
   becomes `elementwise_12544x144_i32xi32xi8`. M4 counted 172 of those. Removing
   them is worth **0%**, because a transpose fused into an elementwise op that
   has to run anyway moves the same bytes in a different order. (It also costs
   the classifier matmul its offload: propagation folds the constant RHS
   transpose into the matmul's indexing maps, giving the N-K form that
   `@match_rocket_matmul` correctly declines. So it is not landed.)
2. **The dispatch count is not the cost.** Cutting 120 of 265 sites buys 4%.
3. **Neither is the epilogue.** Deleting the `%raw + %init` add and the
   full-size `linalg.broadcast` that feeds it -- the ceiling on any fusion or
   hardware-bias fix for it -- buys 7%.

So "17 offloaded convolutions cost +112 net dispatches, and that is what the
5079 ms of `outside` is" (M4) is **wrong** — `outside` contains no driver
compaction at all, and the dispatches it does contain are not what costs. The dispatch count and the cost
happen to correlate because both scale with the number of offloaded
convolutions; ablating one without the other separates them.

### What does move it: CPU slots

| | cpu4,5 | cpu4-7 | cpu0-7 |
|---|---:|---:|---:|
| `int8.base` | 501 / 509 ms | 321 / 318 ms | 262 / 265 ms |
| `int8.cpu` | 130–132 ms | 130–131 ms | — |

**The CPU-only arm does not care how many cores it gets and the offloaded arm
nearly halves.** `ROCKET_PROFILE` says where: per NPU dispatch, `outside` goes
7.46 -> 3.36 -> 2.83 ms across the three, while every phase the driver owns is
flat (`compact` 1.03 / 1.06 / 0.88, `record` 0.72 / 0.81 / 0.48, `wait.npu`
0.71 / 0.75 / 0.55). None of the win is in this driver.

The reading that fits: the CPU-only build is a serial chain of large fused
kernels with few workgroups each, so it is latency-bound and two A76s are
already enough; the offloaded build's extra work is unfused per-element passes
over `i32` tensors, which is embarrassingly parallel and starves on two cores.
The offload is losing to *unfused epilogue work*, not to dispatch overhead --
and every offload number this repo has ever quoted was taken at `taskset -c
4,5`, which is the configuration that punishes it most. **The deficit is 3.9x
at two cores and 2.0x at eight.** Quote both, or quote the core count.

### 284 threads per inference, and pooling them is a wash

`strace -f -c` over one inference counts **284 `clone3`** against 50 NPU
dispatches: `device::run_after_wait` does a `std::thread::spawn` for every
`queue_*` operation whose wait is not already satisfied, and `queue_alloca` /
`queue_dealloca` / barriers vastly outnumber `queue_execute`. Each carries a
stack `mmap`, an `mprotect`, `set_robust_list`/`rseq`, and a `munmap`.

Replacing the spawn with a condvar worker pool (same blocking semantics -- a
new worker whenever every existing one is busy, so nothing serializes) cut it
to **282 threads for a whole 11-inference run** instead of ~3200, and measured
**-3% at cpu4-5 and +3% at cpu4-7**, consistently, three passes each. Not
landed. The likely reason it does not help is that a freshly spawned thread
starts runnable on the waker's own CPU while a parked one has to be woken and
possibly migrated; the creation cost it saves was never the problem. Recorded
so the idea does not get re-derived.

### What landed

The int8 epilogue is a plain `linalg.generic` instead of a hand-written
`flow.dispatch.workgroups`, and `@__transform_main` runs `inline` on the
`@call_rocket_*` wrappers. Both halves are needed: a pre-formed dispatch is
opaque to dispatch-region formation, and a `linalg.generic` left inside a
never-inlined `util.func` just becomes its own dispatch (P6 item 2's trap).
Together IREE fuses the epilogue with the zero-point correction and
requantization that follow it, const-eval hoists the depthwise HWC->CHW filter
transpose that had been running as 13 CPU dispatches per inference over
constant weights, and MobileNetV2 int8 goes **265 -> 145 dispatch sites**.

Measured 2026-09-05 with each arm run **first** in half the passes, because
the original A/B always ran `landed` second and this board drifts within a
sequence. Six runs of each arm at each core count, medians:

```text
                    cpu4-5                          cpu4-7
  int8.base     498 506 509 506 506 507  -> 506   318 316 315 319 318 313  -> 317
  int8.landed   446 488 480 475 500 476  -> 478   285 291 292 286 281 288  -> 287
                        -5.5%                            -9.5%
```

The sign is the same in both orders (base-first: 506 vs 480 and 317 vs 289;
landed-first: 506 vs 476 and 317 vs 285), and at cpu4-7 the two distributions
do not overlap at all. `int8.base` is the tighter of the two, 313-319 ms across
six runs.

**Where the difference is, per NPU dispatch, at cpu4-7:**

```text
  phase          int8.base   int8.landed
  outside          3.512        2.970      <- all of it
  execute          2.802        2.775
    record         0.871        0.842
    compact        1.101        1.117
    wait.npu       0.745        0.736
    pack.input     0.214        0.215
    quiesce        1.062        1.061
```

Exactly as the change predicts: it removes CPU dispatches and touches nothing
the driver does, so the gain is entirely in `outside` and every driver phase is
flat. The weight cache is the independent confirmation that the inlining
worked -- **478 misses over 34 inferences becomes 49 over 35**, one per binding
on the first inference only. Those 429 recurring misses were the depthwise
filter transposes: each inference produced a fresh transient buffer for the
transposed filter, so the packed-coefficient cache could never hit it.

**The per-op table is unchanged**, as it should be: it only covers phases the
driver owns, and this change touches none of them. It does refine P2's bound,
though, because compaction is *concentrated* rather than spread. Worst op,
`int8.base` at cpu4-7, 35 calls:

```text
  conv int8acc 112x112x24->144 k1x1 s1   31.5 ms/call   npu 6.9   compact 18.1
```

18.1 ms of compaction on one op against a 1.10 ms average over all 1700
dispatches -- this single convolution is a third of the model's entire
`compact` total. So "P2 caps at ~10%" is right for the model and wrong for the
op: a cube-aliasing fix would be worth little on the 7x7 tail and most of
`112x112x24->144`. Pick the shapes before building it. (These figures are
within noise of the ones M4 recorded at cpu4-5 -- 32 ms/call, 7 ms NPU, 17.7 ms
compaction -- which is itself the point: nothing about this phase moved.)

**Instrument warning.** `ROCKET_PROFILE=1` is not safe for an A/B at cpu4-5. It
makes `int8.base` *faster* (478 ms median profiled against 506 unprofiled) and
erases the whole difference (base 476/478/480 against landed 479/480/485,
three interleaved passes). That is consistent with the rest of this issue --
two CPUs is a scheduling-starved configuration, so an instrument that adds a
mutex and a yield per phase changes the thing it is measuring. At cpu4-7 the
profile and the un-profiled A/B agree. Use `ROCKET_PROFILE` for composition,
and a plain interleaved run for totals.

Logits are **bit-identical** to the pre-change build (all 1001, max|diff| 0.0)
— the change is purely structural. fp16 is unaffected in placement (37
NPU sites either way, 148 -> 145 CPU sites from the inline alone); its own
epilogue still carries a genuine `f16 -> f32` widen and was left alone.

### The ranked levers this leaves

1. **Stop materializing `i32` activations and stop leaving their epilogues
   unfused.** This is the whole 7.4 ms. The structural version is the
   requantized int8 path (`requantized-int8-conv-path`): it returns `i8` with
   the bias on the BS plane and no CPU epilogue at all, which removes the
   `i32` tensor rather than fusing passes over it. What landed above is the
   cheap half of the same idea.
2. ~~**Re-take every offload number at a realistic core allocation.**~~ Done
   2026-09-05; see the re-measurement table below. It confirms this issue's law
   across a second precision and shows the 7.4 ms constant is the `4,5` value
   of one that falls to 1.6 ms at `0-7`.
3. **P2 (the driver-level NC1HWC2 round trip) is still open** and is the one
   part of M4's lever 1 this did not test. Size it against `compact`'s 1.10 ms
   average out of ~10 ms per convolution before building it -- but note the
   cost is concentrated, not spread: one convolution
   (`112x112x24->144 k1x1`) carries 18.1 ms of compaction per call and a third
   of the model's total, so the lever is shape-selective.
4. Compiler-level transposes, dispatch counts, and thread churn are all
   measured and all worth ~nothing. Do not spend on them again.

### Moved here from M4, 2026-09-05

M4 is resolved and its narrative is gone, but three of its measurements are
evidence for the open question this issue is about, so they are kept here
rather than in the ledger. The first two were taken at `cpu4-5` before the
allocation rule below existed; read them with that in mind.

### The first hang-free int8 numbers, 2026-09-05, and where the time goes

With ISSUES.md C8 fixed the int8 offload finally completes a benchmark loop, so
these are the first int8 figures that are not contaminated by watchdog kills or
by a per-boundary dwell. Both arms are MobileNetV2 (`mnv2.int8.mlir`) through
the **same** rocket-compiler pipeline -- the CPU arm built with the matcher
`dim_bounds` rewritten to `umin = umax = 999999`, which is what
`--no-offload` now automates --
and both are aarch64 modules run on `planck` with the governor on
`performance`, NPU IRQs on cpu6 and the app on cpu4-5:

    int8.npu   17 NPU sites of 265 dispatch sites   484-515 ms   2.07 items/s
    int8.cpu    0 NPU sites of  64 dispatch sites   131-132 ms   7.60 items/s

**The offload is 3.7x slower than the like-for-like CPU build.** Five
consecutive 5 s runs of the NPU arm, zero hangs.

`ROCKET_PROFILE=1` says why, and it is not the NPU:

    outside      5079.0 ms   IREE's non-driver work: the CPU dispatches
    execute      2195.0 ms   the driver's own per-dispatch work
      compact     820.0 ms   DPU atomic slots -> IREE's dense ABI buffer
      wait.npu    772.8 ms   the hardware jobs themselves
      record      627.7 ms
      quiesce     207.0 ms   195 depthwise<->dense dwells at 1.06 ms
      pack.input  153.4 ms
      pack.weights 78.0 ms
    wall         7901.7 ms   host 7112.3, npu 789.4, npu share 10.0%

Two things fall out. **The device is 10% of the wall clock**, so nothing about
the NPU's 200 MHz clock (M2) or its IRQ placement (M3) can move this much.
And **output compaction alone (820 ms) costs more than every hardware job put
together (773 ms)** -- the per-dispatch NC1HWC2 round trip is the offload's
actual price, exactly as `rocket-layout-repack-per-dispatch` predicted.

The per-op table makes it concrete. The worst op,
`conv int8acc 112x112x24->144 k1x1 s1`, spends 32 ms per call: 7 ms on the NPU
and **17.7 ms in compaction**. Offloading it is a loss no matter how fast the
DPU gets.

So the lever is layout propagation between chained NPU dispatches, not clock,
not IRQ affinity, and not more offload sites -- more sites at this cost make it
worse, which is what the 17-site build against 0-site's 3.7x is already saying.

**Corrected 2026-09-05.** "More sites make it worse" holds and is now
quantified: a flat 7.4 ms per offloaded convolution, whichever convolution it
is. "The lever is layout propagation" does not: it was built and measures 0%.
The 3.7x is also configuration-dependent — it is 2.0x when the process is not
confined to two CPUs.

### What the 64% `outside` actually is: fusion the offload destroys

`outside` is IREE's own dispatches, and the question it raises is whether the
answer is more HAL coverage. Counting dispatch entry points by family in the
two arms -- the same two modules, `strings`ed for
`main_graph$async_dispatch_N_<family>` -- says no:

| family | NPU build | CPU-only | delta |
|---|---:|---:|---:|
| `elementwise_transpose` | 172 | 0 | **+172** |
| `matmul_like` | 0 | 136 | **-136** |
| `elementwise` | 96 | 36 | +60 |
| `conv` | 16 | 72 | -56 |
| `slow_memcpy` | 39 | 0 | **+39** |
| `transpose` | 24 | 4 | +20 |
| `elementwise_broadcast` | 20 | 0 | +20 |
| `matmul` / `reduction` | 0 | 8 | -8 |
| **total** | **368** | **256** | **+112** |

**These are not ops we failed to offload. They are ops offloading created.**
The CPU-only build fuses conv + bias + dequant/requant + activation into 136
`matmul_like` kernels; the NPU build has none, because nothing fuses across a
Rocket dispatch boundary. Every epilogue becomes its own dispatch, and 172 of
them carry a transpose because the NPU wants a different layout than its
neighbours. `slow_memcpy` x39 is IREE's own name for the unfused fallback copy:
39 dispatches that exist only to move bytes, none of which the CPU build needs.

So 17 offloaded convolutions cost **+112 net dispatches**, and that is what the
5079 ms of `outside` is.

**Does the HAL already support these?** Partly, and it does not help. The HAL
has `build_add_regcmd`, `build_unary_regcmd` and `build_conv_then_add_regcmd`
(`iree-rocket-hal/src/rocket/elementwise.rs`), but the transform spec matches
none of them -- every `@match_*` is a convolution except
`@match_pooling_nchw_sum_avg` and `@match_rocket_matmul`. Adding an elementwise
or transpose matcher would treat the symptom: each one is another Rocket
dispatch and another pack/compact round trip, and at the measured 17.7 ms of
compaction against a 7 ms convolution an elementwise op would be almost pure
overhead.

**The ranked lever list this supports** — item 1 was built and measured on
2026-09-05 and is **refuted** by this issue's own measurements above, which
replace this list:

1. ~~**Layout propagation between chained dispatches.** Removes the 172
   transposes and the 39 memcpys at their source *and* the driver's 820 ms of
   compaction. This is the whole game; see
   `rocket-layout-repack-per-dispatch`.~~ The mechanism works — the transposes
   really do disappear — and it is worth **0%**. Cutting 120 of 265 dispatch
   sites is worth 4%. The cost is a flat 7.4 ms per *offloaded convolution*
   that scales with available CPUs, not with dispatches. Only the
   compiler-level half of this item was tested; the driver-level NC1HWC2 round
   trip (P2) is still open, and the profile bounds it at ~10%.
2. **Epilogue fusion into the Rocket dispatch.** The requantized int8 path
   already does this for requantization -- that is why it returns i8 with no
   CPU epilogue -- and `build_conv_then_add_regcmd` is the same idea for a
   residual add. It recovers part of the `matmul_like` fusion the offload lost.
3. Standalone elementwise/transpose matchers, **not before (1)**: each adds a
   boundary rather than removing one.

Stated plainly, because every past instinct in this repo has been the
opposite: at the current per-dispatch cost, **more offload sites make the model
slower**. The 17-site build losing to the 0-site build by 3.7x is not a
coincidence, it is +112 dispatches.

### Re-measured 2026-09-05: both precisions, both arms, three core allocations

Every arm built by the flag, `planck`, governor `performance` on both A76
clusters, NPU IRQs on cpu6, post-C8 board binary (`62e3ca3d`), every run gated
on all three NPU cores reading `suspended`, three interleaved passes at
`--benchmark_min_time=5s` (two at `0-7`). **Zero hangs in 32 runs.** Medians:

| arm | sites (rocket / cpu) | `4,5` | `4-7` | `0-7` |
|---|---|---:|---:|---:|
| `int8.cpu` (`--no-offload`) | 0 / 64 | 132 ms | 132 ms | 156 ms |
| `int8` offload | 50 / 95 | 482 ms | 282 ms | 240 ms |
| **int8 deficit** | | **3.65x** | **2.14x** | **1.53x** |
| `fp16.cpu` (`--no-offload`) | 0 / 55 | 91.9 ms | 91.4 ms | 82.0 ms |
| `fp16` offload | 37 / 145 | 313 ms | 168 ms | 140-169 ms |
| **fp16 deficit** | | **3.41x** | **1.84x** | **1.71x** |

Three things fall out, none of which the old numbers could have shown.

**The deficit is a property of the measurement as much as of the offload.**
Same four binaries, same board, same minute: int8 is 3.65x slower or 1.53x
slower depending only on which cores the process may use. Any single figure
quoted without its allocation is arbitrary within a 2.4x band. The standing
"3.7x" was the worst cell in this table.

**The CPU baselines do not scale and the NPU arms do.** Going from two A76s to
four moves `int8.cpu` 132 -> 132 ms and `fp16.cpu` 91.9 -> 91.4 ms -- nothing --
while the offload arms move 482 -> 282 and 313 -> 168. The extra cores are not
speeding up the model; they are absorbing the offload's own overhead. That is
this issue's "the cost parallelises" seen from the other side, and it is why
`taskset -c 4,5` inflates the deficit: it starves the overhead, not the work.

**The flat per-site tax holds across precisions, and its constant is set by
the core allocation.** The law above was established within int8 alone.
Dividing (offload - baseline) by offloaded sites here:

| allocation | int8 (50 sites) | fp16 (37 sites) |
|---|---:|---:|
| `4,5` | 7.00 ms | 5.98 ms |
| `4-7` | 3.00 ms | 2.07 ms |
| `0-7` | 1.67 ms | 1.57 ms |

The two precisions agree to within 15% at every allocation and to within 6% at
`0-7`, so the tax is indifferent to precision as well as to shape. And the
constant itself falls **4.4x** between the narrowest and widest allocation.
The 7.4 ms is not a hardware or driver constant; it is the two-A76 value of
one.

**What is still true.** No configuration beats the like-for-like CPU build, at
either precision, at any allocation -- the best cell in the table is still 1.5x
slower. M4's conclusion is unchanged; only its magnitude was wrong, and it was
wrong in the pessimistic direction by up to 2.4x.

**Rule, restated.** Quote the allocation next to the number, run both arms at
the same one, and prefer `0-7` or `4-7` -- `4,5` measures a starved machine.
`~/bench/m4sweep.sh` on planck takes the whole table.

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

## Resolved

What was settled and how, newest first, in place of the narratives — those are
in this file's git history (`git log -p ISSUES.md`). Everything cited below is
something that still exists: a commit, a file, or a memory.

**M4 (S2) — 2026-09-05. Every NPU-vs-CPU number before 2026-09-04 was measured
against the wrong CPU baseline.** The spec runs
`iree-preprocessing-convert-conv-to-channels-last` before its match loop, so a
rocket-pipeline build is NHWC whether or not anything offloads, while a plain
`iree-compile` build stays NCHW and is 2.8x slower on MobileNetV2. Isolated by
bisection (10.94 -> 3.94 items/s on deleting just those two
`apply_registered_pass` lines). `rocket-compiler --no-offload` now builds the
like-for-like arm and refuses to emit a spec that would offload anyway; its
output is bit-identical to the hand-edited builds it replaces. Commits
`411b7cc`, `df07d56`; README "The CPU-only baseline"; memory
`nhwc-cpu-baseline-trap`. The numbers it produced are in P8.

**C8 (S1) — 2026-09-05. An int8 dispatch hung the next fp16 one.** The cause
was `brdma_data_use` left set on the int32-accumulator path, which fetches
through a plane the BS stage bypasses. Not runtime-PM, which was the leading
hypothesis for a day and had a working mitigation, and not the regcmd, which
was exonerated end to end against `../rocket-userspace`. Commits `59eaec0`
(root cause), `dfd1410` (fix), `cd80179` (gating case); memory
`c8-runtime-pm-poisoning-lead`.

**P6 items 1-3 (S2) — 2026-09-04. MobileNetV2 fp16 spent 30% of its time
repacking constant weights.** Packed coefficients are now cached across
inferences (1.47x); the classifier matmul's operands are no longer re-narrowed
every inference; and the host-side layout transforms ask for the big cluster
(`rocket-hal-driver/src/cpu_affinity.rs`) after `layout_bench` showed identical
code running 13.8 ms on A76 and 52.4 ms on A55. Together 198 -> 146 ms. Commits
`021f0de`, `9f6426c`, `1bc3f62`. P6 stays open for what is left.

**P5 (S3) — 2026-09-04. A pool submit does not cost 507 ms on this board.** The
loaded module already unmasks the PPU_0/PPU_1 completion interrupts, and a pool
dispatch takes 2.6-3.9 ms. The section's other half is stale too: a pooling
matcher now exists and routes (`@match_pooling_nchw_sum_avg`). Memory
`planck-measurement-environment`. What it keeps: any *other* board running a
module without that patch still pays the 507 ms.

**M1 (S2) and M3 (S3) — applied 2026-09-03, measured not to bind.** The A76
governors are pinned to `performance` and the NPU IRQs are on cpu6; both are
free and are now the standing board configuration. But with them applied, both
fp16 arms land within noise of their pre-fix values -- the 3.2x the notes
measured needs idle gaps between invocations, which a benchmark loop never
provides. The confound that did bind was the baseline's conv layout (M4).
Memory `planck-measurement-environment`. **Residual:** the
`depthwise-channel-cap-960` A/B was taken under `ondemand` and has still not
been re-run pinned, so that lever is not actually closed.

**C3 (S1) — 2026-09-03. A watchdog-killed NPU job read as a shape result.** The
dispatch clock is now in both `run_hardware_case_matrix` and
`rocket-hal-driver`: any SUBMIT -> PREP_BO round trip over
`DISPATCH_TIMEOUT_FLOOR` is labelled a hung job rather than a wrong answer.
Commit `0dbbaae`; memory `npu-wedges-after-failed-job`.

**C1 (S1) — 2026-09-03. There is no coefficient-per-channel limit at any kernel
size.** The accumulator was writing the wrong output cube and the readback
modelled the wrong one to match, which looked like a channel cap. Both were
fixed, the caps were raised, and MobileNetV2 was re-audited: every int8
convolution in it now offloads. The retraction is recorded in memory
`accumulator-per-channel-coefficient-limit`; the method lesson it produced is
in **Method note** below.

---

## Suggested order

Revised 2026-09-05, after M4 closed. The offload deficit against a
like-for-like CPU build is **1.5x (int8) / 1.7x (fp16)** on a full machine --
smaller than the 3.7x this list used to be ranked against, so read P8 before
spending on anything below it.

1. **P8** — read first. It measures three things this list previously assumed
   were the cost (compiler-level transposes, dispatch count, thread churn) and
   finds all three worth ~nothing; it lands the epilogue-fusion half of the
   fix; and it points at the requantized int8 path as the structural one. It
   also carries the numbers of record, and the rule that a deficit quoted
   without its core allocation is arbitrary within 2.4x.
2. **The requantized int8 path** — P8's lever 1, and the only structural item.
   It returns `i8` with the bias on the BS plane and no CPU epilogue, removing
   the `i32` activation tensor rather than fusing passes over it. Memory
   `requantized-int8-conv-path` has the board-validated shapes.
3. **P6 → P3 → C4 → P4 → P1** — the dispatch-path cost stack, roughly in
   increasing order of work. P6's residual is one guard held across a whole
   command buffer's recording; the rest is per-tile taxes.
4. **C2** — small fix, plus a probe that settles a question the notes leave
   open.
5. **M2** — ~1.43x on the device half, but the device is ~10% of wall (see
   P8's phase profile), it needs a driver-side `clk_set_rate`, and both
   shortcuts hang the box. Low ceiling for the risk.
6. **P2** — the driver-level NC1HWC2 round trip, the one part of the old
   "layout propagation" lever P8 did not test; bounded at ~10% and
   shape-selective (one convolution carries a third of the model's
   compaction). **P7** is the case that most needs it.
7. **C9, C5, C6, C7, D1, D2** — limitations, hygiene and reconciliation. C9 is
   the only S2 among them: above 3x3 there is a `Cin` cliff that hangs at every
   precision and an int8 program that is wrong at every shape, both now behind
   loud refusals rather than fixed.

---

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
