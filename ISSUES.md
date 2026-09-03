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

---

## C1 (S1) — CONFIRMED 2026-09-03 (second pass, against `../rocket-userspace`): the accumulator uses the wrong output writer, and the "384 coefficient-bytes-per-channel hardware limit" is an artifact of it

**Retracting my first-pass refutation.** I swept `size_e` alone, found it inert,
and closed this. That was the exact mistake
`rockchip-npu-notes/encodings/k-accumulation.md` warns about — *"Three registers
must be right at once... no single-knob sweep can converge"* — and the repo's
earlier `ROCKET_ACC_SURF_ADD` sweep fell into it independently. The two sweeps
were each moving one knob of a three-knob geometry.

### The diff, from the reference implementation

`rocket-userspace/src/npu_regcmd.c`'s `gen_matmul_int8` is a HW-validated,
bit-exact int8 x int8 -> **int32** program. Its DPU output writer against this
repo's `Int8Accumulator`, at 32x32 Cin=384 Cout=64 k1 (offsets confirmed
identical against `rocket-userspace/include/npu_hw.h`):

| field | register | rocket-userspace | this repo | same? |
|---|---|---|---|---|
| `burst_len`/`conv_mode`/`output_mode`/`flying_mode` | `DPU_FEATURE_MODE_CFG` 0x400C | `0xf`/0/2/0 | `0x1e4` = same | yes |
| `out`/`in`/`proc_precision` | `DPU_DATA_FORMAT` 0x4010 | int32/int8/int8 | same | yes |
| **`mc_surf_out`** | `DPU_DATA_FORMAT` bit 3 | **0** | **1** | **no** |
| **`size_e_{0,1,2}`** | `DPU_BS_OW_CFG` 0x4050 | **7** | **1** | **no** |
| **`surf_add`** | `DPU_SURFACE_ADD` 0x40C0 | **`dataout_h*dataout_w * 8`** | **16** | **no** |
| `od_bypass`, BS/BN/EW bypasses, OUT_CVT identity | — | all bypassed | same | yes |

`rocket-userspace/include/npu_dpu.h:120` documents the field that makes the
other two readable:

> `mc_surf_out   DPU_DATA_FORMAT bit3   0=16B/pixel one surface, 1=2/4 surf serial`

So these are **two different writers**, not two tunings of one. This repo is in
the serial mode, which is why `size_e` and `surf_add` read as inert or unhelpful
there — in the serial writer there is no surface stride for either to describe.
`gen_matmul_int8` carries the warning verbatim: `size_e=3`/`surf*4` *"halves the
surface stride, leaving every output column past the first few surfaces as the
`0xAA` sentinel"* — this path's exact signature.

Note also that `surf_add` is derived from the **task's** `dataout_h * dataout_w`,
not the image's. On a height-tiled plan every tile has its own `out_rows`, so
**no constant `ROCKET_ACC_SURF_ADD` could ever have reproduced it**, jointly or
otherwise. That is why the earlier constant sweep was not merely underpowered
but structurally unable to find this.

### Result on hardware

`planck`, one shape per process, `ROCKET_PAD_*` set, device HEALTHY before and
after (verified with a no-override run — an early "SICK" reading was my own
canary running under the override, not the device):

**1x1 kernel — the reference writer is a strict improvement, with no cap:**

| shape | shipped writer | reference writer |
|---|---|---|
| 32² Cin 8 / 32 / 64 / 128, Cout 64 | 100%, 0 mismatches | 100%, **0 mismatches** |
| 32² Cin **385**, Cout 64 | 2.5–8.5% written, 59974–63884 mism. | 100%, **0 mismatches** |
| 32² Cin **512**, Cout 64 | truncated | 100%, **0 mismatches** |
| 32² Cin **704**, Cout 64 | 2.3% written, 64000 mism. | 100%, **0 mismatches** |
| 32² Cin 385, Cout **128** | 2.4% written, 127884 mism. | 100%, **0 mismatches** |
| 32² Cin 385, Cout **256** | 2.3% written, 256070 mism. | 100%, **0 mismatches** |

The surface-multiplier sweep confirms 8 is the value and behaves as the
reference documents: at Cin 384, mult 8 → 100%/0, mult 4 → 75%/16384 mism.,
mult 2 → 62.5%/24576, mult 1 → 56.2%/28672.

**So `MAX_ACCUMULATOR_COEFFICIENT_BYTES_PER_CHANNEL = 384` is not a hardware
limit at 1x1.** It is a description of where the serial writer runs out of
surfaces. `accumulator-per-channel-coefficient-limit`'s central conclusion is
wrong, and its supporting observations were all consistent with this all along:
fully-accumulated written pixels, truncation that is a clean prefix per tile,
small single-surface shapes passing, and `DPU_SURFACE_ADD` driving the write
pattern exactly.

**3x3 kernel — the reference writer regresses, and this is unresolved.** It
writes 100% but 12128 lanes are wrong, with the *identical* count at Cin 16, 32,
33 and 256 (max|diff| 80 / 160 / 165 / 1280, scaling with Cin — computed-wrong
values, not sentinel), including at shapes the shipped writer computes exactly.
Sweeping the multiplier does not recover it (mult 4 → 25480, mult 16 → 38832,
mult 8 → 12128 is the best), and `size_e=3` collapses every arm to sentinel.
Something else in the k≥3 geometry is tied to the writer mode. So this is **not
a drop-in replacement**: k=1 wants the reference writer, k≥3 currently wants the
shipped one.

### Why this matters

The transform spec caps `@match_dynamic_conv2d_int8` at **Cin ≤ 352** and
`@match_dynamic_conv2d_3x3_int8` at **Cin ≤ 32**, and
`int8-dense-conv-caps` calls those the biggest offload lever open. MobileNetV2
is overwhelmingly **1x1 pointwise** convolutions — exactly the kernel where this
is now measured bit-exact to at least Cin 704 at Cout up to 256. Lifting the 1x1
cap no longer needs the `ConvInteger`-requantization-fusion work to land first.

### Suggested next steps

1. Make the writer selectable per kernel size: reference writer (`mc_surf_out=0`,
   `size_e=7`, `surf_add = out_rows*out_cols*8` **per tile**) for 1x1, shipped
   writer for k≥3, until the k≥3 case is understood. Both are now expressible.
2. Re-derive `output_atom_bytes` / `output_row_stride` / the staging partition
   for the reference writer. Note the existing assembler already reproduces it
   bit-exactly at 1x1, so the 128-byte block model and the reference layout
   coincide there — do not assume that holds at k≥3.
3. Then raise the 1x1 int8 cap in the transform spec and re-audit MobileNetV2.
4. Re-test the output-parity rule under the reference writer. My parity shapes
   (3² and 9² at Cout 32) passed under *both* writers, so I did not reproduce
   the failure and cannot say whether it survives; use the memory's `ROCKET_PROBE_ONLY`
   one-hot protocol rather than my `Counting` pattern.

### What landed

- `ROCKET_ACC_SIZE_E` / `ROCKET_ACC_MC_SURF_OUT` / `ROCKET_ACC_SURF_MULT`
  sentinels (all gated by `ROCKET_ACC_SIZE_E_MIN_CIN`), which together express
  the reference writer; `ROCKET_ACC_SURF_MULT` applies the per-task rule rather
  than a constant.
- `ROCKET_PAD_OUTPUT` + a `pad_written` count in the oracle harness.
- `accumulator_size_e_probe`: env-driven, one shape per process, canary either
  side, prints elapsed so a hang is not misread as a shape result.
- Compiled path verified byte-identical with no env set (`0x4010`/`0x4050`/`0x40c0`
  unchanged); 159 host unit tests pass; no new clippy warnings.

**Method note worth keeping.** A wrong `size_e` on the *requantized* int8 path
(where `OD_BYPASS` is clear) hangs the NPU rather than returning wrong data:
1050/1102 ms dispatches against 30–41 ms healthy, `PREP_BO` returning success
either way. That is C3's argument measured, and it is also how I know the
shipped `size_e=1` is load-bearing on that path.

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

## C3 (S1) — the hung-job dispatch guard the memory says is shipped is not in the tree

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

1. **C1** — confirmed and now the biggest item: the 1x1 int8 accumulator has no coefficient cap once the output writer matches `rocket-userspace`'s. Land the per-kernel writer selection, then raise the transform spec's 1x1 Cin cap and re-audit MobileNetV2. The k>=3 regression is the open sub-problem.
2. **C3** — a live silent-wrong-results path in production, and the smallest fix here. C1 measured its argument twice over: a 30x wall-clock separation between a watchdog-killed job and a healthy one, with `PREP_BO` returning success either way.
3. **M1 + M2 + M3** — free or nearly free, and they change what every subsequent measurement means. Do these before re-running any A/B.
4. **C2** — small fix, plus a probe that settles a question the notes leave open.
5. **P3 → C4 → P4 → P1** — the dispatch-path cost stack, roughly in increasing order of work.
6. **P2** — the big structural one; measure the regime first, under a pinned governor.

## Method note

Two independent sweeps missed C1 the same way — the repo's `ROCKET_ACC_SURF_ADD`
sweep and my `ROCKET_ACC_SIZE_E` sweep — because each moved one register of a
three-register geometry and read "no effect" as "eliminated". The reference
implementation is what broke it open, not more sweeping. When a subsystem has a
mode bit (`mc_surf_out`) that reinterprets the meaning of its other fields, no
single-knob sweep of those fields can converge, and a null result from one says
nothing. Prefer diffing against a known-good emitter over sweeping, wherever a
known-good emitter exists.
