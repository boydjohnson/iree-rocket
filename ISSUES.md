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
[ROADMAP.md](ROADMAP.md) is the third: which MLIR operations the hardware could
run but no compiled model can reach yet. One of its phases is gated on an
issue here by name: P8's measured per-dispatch cost is why its coverage
matchers land behind a flag. C5, which used to gate the LUT path in a compiled
model, was resolved 2026-09-06, and C2 and P6 on 2026-09-07 -- see
**Resolved**. C12 and C13 (2026-09-08) are there too: the one hazard
LIMITS.md and DYNAMIC_SHAPES.md carried that this file did not, and the
chained-dispatch fault behind P8's `Cin` 1344 anomaly and the `Cout` 24
rule. ROADMAP's fused-activation row landed the same day, which is
what moved P7's and P2's numbers below. **M2 was resolved 2026-09-09** -- the
NPU now runs at 600 MHz, not 200 -- which means every board number in this
file taken before that date has an inflated `wait.npu` and an understated host
share; see **Suggested order** for what that reranks.

Trimmed 2026-09-05: issues that are settled were cut down to one entry each
in **Resolved** at the end, which keeps their IDs resolvable without keeping
their narratives. Everything above that section is open. Evidence a resolved
issue produced that open work still depends on was moved into the open issue
that needs it, not deleted -- M4's phase profile and dispatch-family counts now
live in P8.

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

**Progress 2026-09-07** (`rocket-hal-driver/MULTICORE.md` §9-§11). The
hardware term is measured: N opens scale 1 / 2.0 / 3.0x on three cores for
both a conv and a matmul, and the fc host phases pipeline to 4.0x. The driver
now has the N-context worker pool (`ROCKET_NPU_CORES`), placement per command
buffer, the copy hop for the four direct-binding sites, a per-context weight
cache and a device-global time-based depthwise dwell -- bit-exact at any N,
gated on four models and the three e2e gates. It buys nothing yet: IREE
orders command buffers on one device timeline and the two-device
partitioning gives this device one dispatch per command buffer, so `overlap`
is 0.0 % on ViT at N=3. Dispatch-level placement is therefore not the lever
on these programs; the notes' "multicore only helps a multi-tile conv" above
is exactly right, and the remaining work is M2, splitting one dispatch's CBUF
tiles across contexts. Still open.

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

**Resized again 2026-09-07.** 24.4 ms (`compact` 17.7, `pack.input` 6.7)
against a **127 ms** model — 19% of wall, and the second-largest term after
P7's `outside`. See P7 for the profile. Two things to carry into any attempt:

- **The cost is concentrated, not spread.** One convolution,
  `112x112x24->144 k1x1`, is 5.4 ms of compaction per inference on its own —
  30% of the whole `compact` budget. P8's lever 3 said this and the current
  profile confirms it: the lever is shape-selective, and a chaining
  implementation that pays a fixed cost per dispatch to save an average one
  will lose.
- **The reach is gated on P7.** Chaining needs producer *and* consumer on the
  NPU, and this model alternates dense (NPU) with depthwise (CPU). The
  biggest compaction, the one above, feeds a *depthwise* convolution — so it
  is not chainable at all until P7 moves. Sizing P2 against the whole 24.4 ms
  overstates what it can reach today.

---

**Sized on ResNet50 fp16, 2026-09-08 -- the model where the NPU wins.**
The reach of chaining is the set of edges where one NPU dispatch feeds
another with nothing between them, and on the current ResNet50 build that
set is **empty**: all 51 NPU results feed the plain fp16 shim's CPU epilogue
(`extf` + bias add against a full-size hoisted f32 bias tensor), then a
second CPU dispatch (ReLU + `truncf`) before the next convolution, and the
bottleneck outputs go through a CPU residual add as well. Measured from the
per-op profile (`taskset -c 4-7`, rebuilt runtime, 245 ms/inference:
`outside` 100.1, `wait.npu` 79.2, `compact` 28.6, `pack.input` 15.7) and
split by the role each shape plays in a bottleneck:

| edge class | calls/inf | `pack.input` | `compact` | `npu` |
|---|---:|---:|---:|---:|
| conv1 (1x1 in, reads the block input) | 16 | 8.6 | 4.1 | 23.8 |
| conv2 (3x3) | 16 | 3.5 | 2.1 | 33.2 |
| conv3 (1x1 out, feeds the residual add) | 13 | 1.3 | 8.9 | 10.7 |
| conv3 + downsample at 56x56x64->256 | 4 | 1.1 | 12.6 | 6.0 |
| downsample 56x56x256->512 s2 | 1 | 1.2 | 0.9 | 4.0 |

So the order of work, with what each step is worth per inference:

1. **Fuse the fp16 epilogue into the dispatch** -- bias on the BS plane for
   every convolution and ReLU on the BN stage where the model has one. This
   is the ReLU6 machinery generalised to `relu` and to bias-only, on the
   f16-import chain (`truncf` then a `cmpf ugt`/`select` clamp) rather than
   the demoted one. It is not chaining, but it is the prerequisite for it
   *and* the larger lever: it removes two full-tensor CPU passes behind each
   of 51 convolutions, most of the 100 ms `outside`, and leaves conv1 ->
   conv2 -> conv3 as direct NPU -> NPU edges once the shim's `extf` and the
   graph's `truncf` cancel.
2. **Chaining on those edges** (the driver keeps conv1's and conv2's output
   cube and conv2 and conv3 read it): `pack.input` 4.7 + `compact` 6.2 =
   **10.9 ms**, 4.5 % of wall. Bounded by the fact that the *wide* tensors
   (conv3's Cout 256..2048 outputs, 21.5 of the 28.6 ms of compaction) all
   cross the residual add.
3. **Residual add on the NPU** (`build_conv_then_add_regcmd`, ROADMAP Phase
   3's epilogue field) makes every edge direct: the whole `pack.input` +
   `compact` term, **~44 ms**, 18 % of wall, plus the 16 CPU adds.

Nothing in this list is worth building before step 1, and step 1 pays on
its own.

**Step 1 landed 2026-09-08.** `rocket-fuse-conv-relu6` gained two patterns:
bias-only (every dense fp16 1x1/3x3 convolution seeded with a per-channel
bias now hands it to the BS plane through the plain executables' bias
binding, which had only ever carried zeros) and bias + ReLU on the
f16-import chain (`truncf` then `cmpf ugt`/`select`, the clamp moving onto
the BN stage through four new `relu` targets: stride 1 and 2, with and
without the folded pad 1). The shims widen with a `linalg.generic` rather
than a pre-formed dispatch, so the widen and the model's own narrow cancel
once the wrapper is inlined -- that is what makes the edge direct. Two
match-loop facts fell out: an epilogue-rooted matcher must live in the
first loop, and a conv-rooted one in the second, because the walk reaches
the convolution before its epilogue and a conv-rooted matcher in the first
loop pre-empts everything behind it (the pad-1 matchers moved for this).

ResNet50 fp16 on `planck`, `taskset -c 4-7`, 3 repetitions, medians:

| | dispatches (NPU / CPU) | NPU results feeding an NPU dispatch | ms |
|---|---|---|---:|
| before | 51 / 130 | 0 of 51 | 230 |
| after | 53 / 22 | **32 of 53** | **162** |

1.42x on the model, max|diff| 0.0116 against onnxruntime (the CPU arm is
0.0151), same argmax and top-5; the CPU arm is 1035 ms. MobileNetV2 fp16
(f32 import, the demoted spelling) is unchanged in placement at 37 sites,
129 ms, max|diff| 0.028 against onnxruntime with the same top-5. Every
conv1 -> conv2 -> conv3 edge is now NPU -> NPU with nothing between, which
is exactly the set step 2 chains; what still crosses the CPU is the 16
residual adds, the 7x7 stem, the padded max pool and the head.

**Re-sized after step 1** (`ROCKET_PROFILE`, same build, per inference;
the profiler inflates wall to 237 ms, composition only): `wait.npu` 108.2
(46 %), `outside` 47.9, `compact` 37.0, `pack.input` 20.6. By role, the
edges step 2 can chain are conv1 -> conv2 and conv2 -> conv3:

| edge class | `pack.input` | `compact` |
|---|---:|---:|
| conv1 (reads the block input, from the CPU add) | 9.2 | 3.7 |
| conv2 | 3.2 | 3.9 |
| conv3 (feeds the CPU residual add) | 1.6 | 9.7 |
| conv3 + downsample at 56x56x64->256 | 1.5 | 15.0 |

So **step 2 is worth 12.4 ms**, 5 % of the profiled wall and at most 7 % of
the real one: conv2's and conv3's packing plus conv1's and conv2's
compaction. Three quarters of the remaining `compact` is conv3 writing the
wide block output for the residual add, and half of `pack.input` is conv1
reading it back, so **step 3 (the residual add on the NPU) is now the
larger lever by four to one**: it unlocks ~57 ms of pack and compact plus
the 16 CPU adds, against step 2's 12. Build step 3 first, and step 2 on
the edges it leaves.

**Step 3 landed 2026-09-08.** `Conv2DDef.epilogue_add` and
`epilogue_activation` put the residual add and its ReLU in the DPU's EW
core after the convolution's own tiles, inside the same dispatch, with the
skip as a fourth binding; the driver runs the tiles, then one EW task on
the same in-order queue (no fan-out for such a dispatch), and compacts the
sum. `conv_residual_add_hw` validated the multi-tile cube handoff and the
EW-stage ReLU exactly before any wire or compiler work. The pass rewrites
the f16-import chain `conv -> truncf -> linalg.add(skip) -> relu` (a named
`linalg.add`, since specialisation runs first) into one epilogue generic
over the accumulator, with the skip collapsed to the convolution's rank.

ResNet50 fp16 on `planck`, `taskset -c 4-7`, medians:

| | dispatches (NPU / CPU) | NPU results feeding an NPU dispatch | ms |
|---|---|---|---:|
| after step 1 | 53 / 22 | 32 of 53 | 162 |
| after step 3 | 53 / 8 | **51 of 53** | 164 |

Correctness is unchanged (max|diff| 0.0116 against onnxruntime, same top-5)
and the wall is flat, and the profile says exactly why: `outside` fell
47.9 -> 26.4 ms (the sixteen CPU adds), but each residual dispatch now
packs its skip tensor (`pack.input` 20.6 -> 29.5) and its EW task is
hardware time (`wait.npu` 108 -> 131). An even trade on its own. What it
buys is the reach of step 2: `pack.input` + `compact` is now **61.5 ms**
per inference and 51 of the 53 edges it sits on are NPU -> NPU, so the
driver chain -- consumer reads the producer's cube, no compaction, no
repack, the skip too -- is worth up to 37 % of the model. That is the next
thing to build, and it is what P2 always was.

What stays on the CPU: the 7x7 stem, the padded max pool, the head's
pooling transpose and the classifier epilogue. MobileNetV2's residual adds
have no ReLU after them (linear bottlenecks) and are not claimed; a
`NONE`-activation twin of the target is one matcher away.

**Step 2 landed 2026-09-08 -- the input half.** A dispatch whose input was
written by an earlier dispatch on the same command buffer now reads that
dispatch's output cube in place (`OutputCube`, `chainable_cube`,
`ROCKET_CHAIN=0` to turn it off, `=debug` to trace every edge). No wire or
compiler change: the producer already writes feature-atomic NC1HWC2 surfaces
into scratch, and the repack the consumer used to do reproduces those exact
bytes, so the consumer simply points its regcmd at the producer's scratch BO
and the pack is skipped. This is the aliasing `encodings/cross-op-chaining.md`
proved legal and the section above has carried since.

The claim is `pack(compact(cube)) == cube`, and it holds when the two
dispatches name the identical device byte range, the pixel counts are equal
(surfaces are `pixels * 16` apart, so a producer with physical height padding
strides differently), the logical pixel widths are equal with no channel
padding beyond them, and the pixel is a whole number of 16-byte atoms -- the
last two because a repack zeroes padding lanes where a producer leaves its
padding channels. Fanned-out and accumulator dispatches offer no cube: the
first has no single scratch (each context wrote its own tiles), the second
writes 128-byte blocks. `tensor_layout.rs`'s `chain_identity_tests` pins the
identity and each way it fails, since nothing at runtime can check it.

ResNet50 fp16 on `planck`, `taskset -c 4-7`, medians of 3, same
`resnet50.res.vmfb` with the feature only turned off and on:

| | `pack.input` | `compact` | ms |
|---|---:|---:|---:|
| chaining off | 24.0 | 26.0 | 169 |
| chaining on | **0.6** | 25.6 | **145** |

**1.17x**, and the output is bit-identical to the same build with chaining
off and to the step-3 reference -- byte for byte, not within a tolerance,
which is the point of the identity above. 66 of the 68 conv operand repacks
are gone: all 50 feature inputs and all 16 residual skips. The 2 that remain
have no NPU producer on their command buffer (the stem, behind the CPU max
pool). MobileNetV2 fp16 is 134 -> 132 ms and bit-identical, with 1 of 34
edges chained -- its depthwise convolutions sit on the CPU between every pair
of dense ones, so its reach is gated on P7, exactly as the 2026-09-07 sizing
above said. **Corrected 2026-09-08 by measuring it** (P7): with the seven
admissible depthwise offloaded the chain still takes that same one edge.
The model's edges are behind the CPU residual add (10), the explicit pad
in front of every depthwise (7) and the whole-atom rule (8 of the 14 direct
NPU -> NPU edges, at 88/136/24 channels); P7 alone opens nothing.

~~**What is left of P2 is the compaction, and it needs something this layer
does not have.**~~ `compact` was unmoved at 25.6 ms, 18% of wall, because the
producer still wrote the dense IREE buffer: nothing in a command buffer can
prove that buffer has no other reader -- a later CPU dispatch, a later
submission, or the model's own output -- and eliding the write on a guess is
silent corruption. **Landed 2026-09-08 as step 4 below**, with the signal
coming from the compiler. The other open edges are the elementwise, pooling
and matmul dispatch kinds, which record no cube yet: on a matmul-heavy model
(ViT, Qwen3) that is where the same lever would apply.

**A trap worth keeping.** The first cut matched producers by
`iree_hal_buffer_t` pointer alone and chained **zero** residual skips: IREE
packs neighbouring transients into one allocation, so the most recent write
to that buffer is usually a different tensor at a different offset, and every
skip -- the wide tensors, `Cout` 512..2048 -- was declined as if blocked. The
fix is to compare device byte *ranges* and skip writes that miss: 0 of 16
skips became 16 of 16. `ROCKET_CHAIN=debug` printed the offsets that said so.

**Step 4 landed 2026-09-08 -- the compaction half.** A dispatch skips
writing its dense output buffer when every reader of its result took the
output cube in place. The proof that there is no other reader is a count,
not a flag, and it comes from the compiler: `rocket-mark-dense-readers`
(run by `rocket-compiler` at the flow phase, next to the placement pin)
walks each Rocket convolution's result once dispatch regions are final and
counts the Rocket dispatches that read it, through `flow.tensor.reshape`;
any other reader -- a CPU dispatch, `util.return`, a tied operand -- makes
the count 0. The count rides as one more push constant
(`Conv2DDef.runtime_dense_readers`, the trailing constant of every
convolution target, literal 0 in every shim). The driver tallies, per
recorded dispatch, how many later dispatches on the same command buffer
chained to its cube and whether anything read its dense bytes instead (a
consumer that declined, a kind that records no cube, a copy), and
`apply_ops` drops the compaction iff `chained == count > 0` and no dense
read was seen (`compaction_elidable`). Every failure mode keeps the write:
a reader on a *later* command buffer leaves the tally short, which is why a
count survives IREE's partitioning where a boolean would not; an executable
compiled without the constant says 0. A driver-only version was considered
and rejected: a transient from `queue_alloca` is not proof of a single
command buffer, because IREE also allocas a result consumed by a later
execute region.

`planck`, `taskset -c 4-7`, medians of 4 interleaved passes, same binary
and same `.vmfb`, `ROCKET_LAZY_COMPACT=1` against `=0`:

| model | kept / skipped per inference | off | on | |
|---|---|---:|---:|---:|
| ResNet50 fp16, 224x224 | 2 / 51 | 142 ms | **114 ms** | **1.25x** |
| ResNet50 fp16, 512x512 | 2 / 51 | 858 ms | **626 ms** | **1.37x** |
| VGG19 fp16 | 5 / 11 | 533 ms | 502 ms | 1.06x |
| MobileNetV2 fp16 (f16 import) | 45 / 2 | 87 ms | 88 ms | noise |

Output bit-identical on and off for all four, and identical to the same
model compiled before this change. ResNet50-224's profile: `compact` 0.45
ms x 2650 calls -> 0.26 ms x 118 over the run, `wait.npu` share 62 -> 74 %.
The compiler's own census matches the driver's: 51 of ResNet50's 52
convolutions are read only by Rocket dispatches (66 reader edges, the 16
residual skips counted twice), VGG19 11 of 16 (the other five feed max
pools, which are Rocket dispatches that cannot chain), MobileNetV2 2 of 46.
The two kept on ResNet50 are the stem (read by the CPU max pool) and the
last conv (read by the head).

With step 2 and step 4 together, ResNet50-224 is 169 -> 114 ms, and what
the driver still does on the host for it is ~10 % of wall (`outside` is the
CPU stem/pool/head). ~~What is left of P2 is breadth, not depth: the
elementwise, pooling and matmul kinds publish no cube, so nothing chains
through them and nothing feeding them elides.~~ Step 5, below.

**Step 5 landed 2026-09-08 -- breadth.** Matmul, pooling and element-wise
dispatches now publish an output cube, take a producer's cube for their
inputs, and carry the reader count (`runtime_dense_readers` on
`MatmulDef`, `PoolingDef` and the three `Elementwise*Def`s; every shim
passes it). `OutputCube` grew a `surface_pixel_count`: the PPU strides its
surfaces by the pixel count rounded up to four, so a pool's cube matches a
conv's only when the count is a multiple of four (VGG's 224/112/56/28
images all are; a 7x7 is not), and `chain_identity_tests` pins the
mismatch. Matmul's cube is width M, height 1, no row padding; the `[K,N]`
operand is a coefficient stream and never chains.

That was the smaller half. The larger one was that on every plain-conv,
pool and matmul edge nothing could have chained anyway, because the shims
widened their f16 result inside a `flow.dispatch.workgroups` that also
folded in linalg's accumulator init -- opaque to fusion, so a CPU dispatch
sat between the two Rocket dispatches on every such edge. Only the fused
bias/ReLU shims (ResNet50's) used a plain generic, which is why ResNet50
chained and MobileNetV2's f32 import did not. Two compiler changes fix it:
every shim's widen is a plain `linalg.generic` (the placement pin, which
postdates the workgroups form, keeps it off the NPU), and
`rocket-fold-neutral-init`, run after inlining, drops the `+ init` /
`max(.., init)` term when the init is a `linalg.fill` of the neutral element
-- which arith itself never does, since `x + 0.0` is not `x` at `-0.0` --
leaving a bare `extf` that the consumer's `truncf` cancels. And on an f32
import the demote pass now narrows *through* a zero `tensor.pad`
(`pad(truncf(x))`, not `truncf(pad(x))`), so `rocket-fold-conv-pad` and
the pad-1 matchers see the same `pad -> conv` an f16 import gives them; the
promote pass rebuilds the f32 pad for anything left unclaimed.

Two chain fixtures joined the conv gate, both bit-exact and both chaining
end to end with the middle compaction skipped: `fp16_conv_pool_conv_chain`
(pool reads conv A's cube, conv B reads the pool's) and
`fp16_matmul_chain`. On the models, `planck`, `taskset -c 4-7`, chain on
vs `ROCKET_CHAIN=0`, medians:

| model | NPU -> NPU edges chained | compactions skipped | chain off | on | |
|---|---|---|---:|---:|---:|
| VGG (f32 import, 16 conv + 5 pool), 10 s runs x3 | **20 of 20** (11 conv, 5 conv->pool, 4 pool->conv) | 20 of 21 | 676 ms | **617 ms** | **1.10x** |
| VGG19 (f16 import, 16 conv; pools on CPU), 10 s x3 | 11 of 16 | 11 of 16 | 544 | **498** | 1.09x |
| ResNet50 fp16 224x224 | 66 of 68 (unchanged) | 51 of 53 | 142 | 115 | 1.24x |
| MobileNetV2 fp16 (f16 import) | 1 -> 6 of 47 | 2 | 87 | 88 | noise |

All bit-identical chain on/off and against the earlier builds. The first
row is the one this step is about: before it, the same model had 50 CPU
dispatch sites (15 explicit pads, 16 widens, the rest shims' narrows) and
5 chained edges; it now has 4 CPU sites, all of them the classifier, whose
K = 25088 is past the matmul caps -- that is the `outside` 195 ms per
inference in its profile and the whole reason the model is not faster.
`pack.input` and `compact` are 0.0 and 0.1 ms per inference on it.

Two things measured on the way that are not wins: a 3 s
`--benchmark_min_time` on a 600 ms model is six iterations and the arms
overlap by 15 % run to run -- the 10 s runs above are what settled both
VGGs -- and the CNA padding the input itself was not faster than IREE's CPU
pad plus a repack on this VGG (617 ms either way); it is the chaining the
fold enables that pays, not the pad itself. Once during these measurements
the board wedged (a benchmark left the NPU busy and the next process hung
in `PREP_BO`, uninterruptible); it did not reproduce after a reboot in six
further runs of both arms and is recorded rather than explained.

What is left is coverage, not chaining: VGG19's torchvision export pools in
f16 and the pooling matchers are f32-only, so its five pools stay on the
CPU (and break the chain between blocks); ViT/Qwen3's matmul edges chain
where the M/K/N caps admit both sides; and a matmul or pool whose consumer
is the CPU (every classifier) keeps its dense write, as it must.

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

**Progress 2026-09-07** (`rocket-hal-driver/MULTICORE.md` §12). The second
half is gone: `rocket-hal-driver/src/scratch_pool.rs` keeps every
driver-private GEM buffer -- regcmd, input, bias, output, and the multicore
replicas -- on a free list keyed on (file, size class), so a dispatch no
longer pays `CREATE_BO` + `mmap` + first-touch faults per tile. Worth ViT
1203 -> ~1050 ms, MobileNetV2 requant 290 -> 255, MobileNetV2 fp16 166 ->
146 at one context, which makes it the largest single win of the multicore
series. Also, every task of a dispatch is now submitted before any is waited
for, so the per-tile `PREP_BO` no longer idles the core between tiles. The
first half -- the whole-BO cache sync, ∝ pages not bytes -- stands, and is
now the dominant cost of a fanned-out replica (`stage` 0.47 ms per ViT
dispatch is mostly `fini_bo` over a 512 KiB input replica).

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

## P7 (S2) — MobileNetV2 fp16's 17 depthwise convolutions stay on the CPU; at 600 MHz offloading them is 1.08x *faster* at the default allocation and 1.02x slower at four workers — but the whole fp16 offload loses to a four-worker CPU baseline, so the demote stays off (re-measured 2026-09-09)

They are the whole of the `outside` term (70.9 ms of a 127 ms model; see
the profile below): ten executables over 17 dispatch
sites, `112x112x48` down to `7x7x1344`, all `linalg.depthwise_conv_2d_nhwc_hwc`
at f32. They stay on the CPU for one reason —
`RocketDemoteConvInputsPass` deliberately excludes depthwise, so an f32
depthwise never becomes the f16/f16/f32 the matchers require and no depthwise
matcher can ever fire.

### The fp16 depthwise channel ceiling moved 512 -> 1536 (2026-09-09), and this verdict did not

Separate from the demote gate above, and worth keeping apart from it. The
*already-f16* MobileNetV2 import needs no demotion, so its depthwise
convolutions reach the matchers directly -- and six of the seventeen were
still on the CPU, because the fp16 depthwise admission ceiling was 512 while
the model's own widest are C=576 and C=960. That 512 was where the depthwise
matchers were first written and nothing had gone back to it: `ConvPlan`
plans fp16 depthwise to `MAX_DEPTHWISE_CHANNELS` (1792), and the int8
depthwise rung had already moved to 1344.

Raised to 1536 in `rocket-core`'s `admission` table, backed end to end:
`tools/e2e_conv_regression.py` gained `depthwise_fp16_c576`, `_c960`,
`_c1536` and `_c1536_s2` (the last NCHW, the only layout the stride-2 fp16
depthwise matcher exists in), each compiled to a Rocket and a CPU module from
the same MLIR and compared on `planck`: max|error| 1.6e-4 to 2.4e-4, 0
mismatches, at atol 1e-3.

MobileNetV2 fp16 goes 47 -> 53 dispatch sites, all 17 depthwise convolutions
now offloaded, max|diff| 0.0156 against its own `--no-offload` arm with top-1
and top-5 unchanged. The wall time says the same thing this section already
said, on `planck` at 600 MHz, `performance` governor, medians of three:

| arm | sites | 4 A76 workers | all 8 cores |
|---|---|---|---|
| NPU, depthwise <= 512 | 47 | 58.9 ms | 77.5 ms |
| NPU, depthwise <= 1536 | 53 | 63.5 ms | 77.3 ms |
| `--no-offload` | 0 | **47.0 ms** | 81.4 ms |

The six extra sites cost 4.6 ms at four workers and nothing at eight. A
depthwise convolution here is still the cheapest op in the model and still
does not pay for its own dispatch, which is what the rest of this section is
about; the raise changes what is *admissible*, not that verdict. It is kept
because admission is a statement about what has been measured, and because a
model whose depthwise convolutions are not this cheap -- a wide MobileNetV3
or EfficientNet stage -- can now reach the NPU at all. Nothing in that class
has been measured here.

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

### Re-measured 2026-09-07 with depthwise ReLU6 fused: the verdict holds, at a quarter of the cost

This is item 0 below, done. The hypothesis was right in direction and too
small to change the answer.

|   | ms | vs 37 sites |
|---|---:|---:|
| 37 sites (dense ReLU6 fused) | 133.0 | — |
| 44 sites, depthwise clamps on the CPU | 142.5 | 1.071x slower |
| 44 sites, depthwise ReLU6 fused into BN | **140.0** | **1.053x slower** |

Six interleaved passes each, `taskset -c 4-7`, governor `performance`,
medians. Accuracy at 44 sites is max|diff| 0.0320 against a `--no-offload`
CPU arm (0.0184 at 37 sites), top-1 and top-5 stable.

**So the 26% is now 5.3%**, and two separate things did that. Most of it is
M2's scratch pool, which removed the per-dispatch allocation the seven extra
depthwise dispatches were each paying -- the 186-vs-148 above predates it.
The rest, 2.5 ms of the 9.5 ms gap, is the depthwise clamps: the section
below was right that offloading un-fuses them and that the recorded
`outside` rise contains them, but they are a fifth of the gap, not the bulk
of it.

**What is in the tree.** All of the depthwise fusion machinery, and it is
hardware-validated: `rocket-fuse-conv-relu6` handles
`DepthwiseConv2DNchwChwOp` (the NCHW chain is
`conv -> transpose -> expand_shape -> clamp`, and the bias broadcast retains
dimension 1, so the canonical clamp is NCHW with a `(d1)` bias map),
`#rocket_dynamic_depthwise_relu6_target` and its stride-2 twin exist with
their shims and matchers, and `conv_fp16_bias_activation_hw` runs bias alone,
bias + ReLU and bias + ReLU6 under the depthwise register program as well as
the dense one -- all six exact. What is *not* in the tree is the demote,
which stays off: `RocketDemoteConvInputsPass`'s scope comment carries the two
lines that turn it on and this table.

**What is left of the gap** is items 2-4 below, unchanged: the explicit pad
IREE materializes as its own dispatch, the `DEPTHWISE_TO_DENSE_QUIESCENCE`
dwell, and the `Cin` 512 matcher cap. Item 1, P2's chaining, is the one that
would move it most and is also the one gated on this item -- see P2.

### The 2026-09-07 profile, and why the earlier accounting understated the clamps

Two things happened after the measurement above, and both move it.

**The `outside` rise has a second cause the measurement never named.** It is
attributed entirely to the explicit padding IREE materializes. But IREE
currently fuses each depthwise convolution's ReLU6 **into the depthwise
dispatch itself** — read the flow IR and one dispatch does depthwise + bias +
clamp + `truncf` to f16 in a single pass. Offloading the depthwise breaks
that fusion and hands back 17 more standalone CPU clamp dispatches. Those are
inside the 88 -> 118 ms rise and were paid but not attributed. The mechanism
to fuse them into the NPU instead now exists for dense convolutions
(`rocket-fuse-conv-relu6`, `#rocket_dynamic_relu6_target`); extending it to
depthwise is an increment on work that has landed, not new ground —
though it does need a depthwise arm of `conv_fp16_bias_activation_hw` first,
because that test covers the dense path only.

**The denominator moved.** The 147/149 ms baseline is now 133 ms (the dense
ReLU6 fusion), so the same absolute cost is a larger fraction, and P6's
`record` term — which is 2.46 + 1.76 + 1.37 + ... of the per-op table above —
has largely gone with the M2 scratch pool.

### The current profile, 2026-09-07

`ROCKET_PROFILE=1`, `taskset -c 4-7`, m2 driver, ~39 inferences, per
inference. Both arms are the same binary and the same input; the fused arm is
the ReLU6 build.

| phase | 37 sites, unfused | 37 sites, ReLU6 fused |
|---|---:|---:|
| `outside` | 79.6 | **70.9** |
| `wait.npu` | 25.9 | 25.4 |
| `compact` | 17.7 | 17.7 |
| `pack.input` | 6.6 | 6.7 |
| `record` | 1.6 | 1.5 |
| wall | 136.7 | 127.2 |

The whole 9.4 ms is `outside` and every driver phase is flat, which is what
removing a CPU dispatch should look like — and it is also the evidence that
moving the bias onto the BS plane and turning BN on costs the hardware
nothing (`wait.npu` -0.6 ms, noise).

**What this profile says about where to spend.** `outside` is **54% of the
model** and P7 is what it is made of: these 17 convolutions. `compact` plus
`pack.input` is 24.4 ms, 19%, and that is P2 — but P2's cross-op chaining
only pays between *adjacent* NPU dispatches, and this model alternates dense
(NPU) with depthwise (CPU), so P2's reach is itself gated on this item. The
knot is P7 -> P2 -> ROADMAP's conv padding row, and P7 is the end to pull.

### Re-measured 2026-09-08 with the driver chain on: chaining reaches nothing here, and the verdict holds

P2 step 2 landed (the consumer reads its NPU producer's output cube in
place), and item 1 below said that was the lever that would move this most.
Measured, it moves it by nothing, and the reason is structural rather than a
tuning.

Same two-line demote (`DemoteInputsToF16<linalg::DepthwiseConv2D{NhwcHwc,
NchwChw}Op>` and the promote twins), same source model rebuilt for the run
(`mnv2.fp16.mlir` is no longer on disk; the width-1.4 f32-constant form is
the fp16 export widened with `widen_to_f32.py`, batch pinned, imported at
opset 20 -- 37 sites, and 44 with the demote, exactly the 2026-09-07
placement). `planck`, six interleaved passes, medians, governor
`performance`, NPU IRQs on cpu6. All three arms are the same binary; the
third is the 44-site build with `ROCKET_CHAIN=0`.

|   | `taskset -c 4-7` | + `--task_topology_cpu_ids=4,5,6,7` |
|---|---:|---:|
| 37 sites | **130.5** ms | **108** ms |
| 44 sites, chain on | 137.0 (1.050x slower) | 127.5 (**1.18x slower**) |
| 44 sites, chain off | 136.5 | 126.5 |

Correctness is unchanged: max|diff| 0.0078 against the `--no-offload` arm
at 44 sites (0.0073 at 37), same top-5, and chain on/off are bit-identical.

**Chaining took one edge in both arms -- the same one.** `ROCKET_CHAIN=debug`
on the 44-site build: 1 taken (the 7x7x448 -> 1792 head), 32 declined for
*no producer on this command buffer*, 8 declined because the consumer's
pixel is not a whole number of 32-byte atoms (88-, 136- and 24-channel
project outputs, which pack 176 -> 192, 272 -> 288 and 48 -> 64 bytes).
Classifying every Rocket call's input operand in the post-match IR says what
the 32 are:

| Rocket conv inputs (44-site build) | fed by | chainable today |
|---|---|---|
| 10 expand 1x1 | `linalg.add` -- the residual add, on the CPU | no |
| 10 project 1x1 | the CPU clamp after a still-unclaimed wide depthwise | no |
| 7 depthwise 3x3 | `tensor.pad` -- the explicit pad, on the CPU | no |
| 7 project 1x1 | the offloaded depthwise, directly | only if whole-atom |
| 7 expand 1x1 | a project conv, directly (no residual) | only if whole-atom |

So the reach was never "17 depthwise convolutions sitting between NPU pairs".
Offloading the seven admissible ones opens at most 14 direct NPU -> NPU
edges, 8 of which the whole-atom rule declines, and every other edge on the
model is behind one of three CPU ops: the residual add (10 edges), the pad
(7), and the clamps of the depthwise the `Cin` 512 cap leaves behind (10).
On this model P2's reach is gated on those three, not on P7 alone; the
2026-09-07 statement above that "P2's reach is itself gated on this item"
was the right direction and too small a claim.

**The hardware term is the binding one, and chaining cannot touch it.**
Whole-run profile, 37 vs 44 sites at `taskset -c 4-7`: `outside` -1030 ms
(the CPU depthwise and their clamps, gone), against `wait.npu` **+633**,
`pack.input` +354, `compact` +265, `record` +47. The seven depthwise
convolutions cost 60 % of the CPU time they replace *in NPU time alone*, at
the 200 MHz M2 leaves the clock at; the layout round trip is the other 600.
A perfect chain -- every pack and compact on those seven edges free -- would
leave 44 sites at roughly 125 ms against 130, a 4 % win at best, and the
four-worker column says the CPU side has more headroom than that
(108 -> ~127 is the CPU depthwise getting four workers while the NPU path
gains nothing).

What this changes in the list below: item 1 is done and worth ~0 until the
pad, the residual add and the whole-atom rule move; item 2 (the pad) is now
also a P2 blocker and the first thing to build; and M2 is the only lever
that touches the term that actually binds. The demote stays off.

### Re-measured 2026-09-09 at 600 MHz: the clock is worth 8-10 points, and it is not the thing that decides this

M2 is resolved (see **Resolved**), and it was the term this issue named as
binding. Governor `performance`, four interleaved passes, medians,
`mnv2_14.f32.mlir`, against a `--no-offload` arm from the same pipeline:

| allocation | clk | base (37) | dw (44) | nooff | dw vs base |
|---|---|---:|---:|---:|---|
| default topology | 200 MHz | 99.4 | 99.8 | 108.0 | 1.004x slower |
| default topology | 600 MHz | 87.5 | **81.0** | 108.0 | **1.080x faster** |
| four workers | 200 MHz | 83.1 | 93.2 | **54.5** | 1.122x slower |
| four workers | 600 MHz | 73.1 | 74.7 | **54.5** | 1.022x slower |

The clock moves the depthwise arm 8-10 points relative to the baseline in
*both* allocations, enough to flip the sign at the default one. That is what
this issue predicted, and the prediction was the reason it ranked the clock
first. `max|diff|` vs `--no-offload` is 0.0051 (dw) and 0.0050 (base), top-1
stable at both clocks, zero faults in ~60 runs.

**The verdict holds anyway, and the reason has moved.** At four workers
`--no-offload` is **54.5 ms** against a best NPU arm of 73.1 — the CPU is
**1.34x faster**, at either clock. The CPU baseline nearly doubles from four
workers (108.0 → 54.5, 1.98x) while every NPU arm gains 8-14%. So this is no
longer "depthwise loses to the layout round trip"; the whole fp16 offload on
this model is underwater once the CPU is allocated fairly, and whether the
depthwise demote is on is a detail inside a losing trade. The 81.0-vs-108.0
row is the only arm here that beats its CPU baseline and it does so only at an
allocation that starves the CPU of workers — `taskset` does not set IREE's
worker count. This is M4's baseline trap and P8's "what moves it: CPU slots"
arriving together.

**Two corrections to the 2026-09-08 entry above.** Its edge census is stale:
compaction now skips 67 dense output writes in the 37-site build against
**464** in the 44-site one, because the residual-add-on-NPU and lazy-compaction
work landed the same day it was taken. And **governor `ondemand` cannot
measure this**: it read the four-worker control at 1.078x against the
documented 1.18x and gave the default-topology control the wrong *sign*. Under
`performance` both controls reproduce. See the method note.

**Reproducing it is now one compiler build.** `ROCKET_DEMOTE_DEPTHWISE=1`
gates the demote in `RocketDemoteConvInputsPass`; the `PromoteInputsToF32`
twins are registered unconditionally because they only fire on ops carrying
`kDemotedAttrName`. Verify by dispatch-site count (37 off, 44 on) rather than
by trusting the variable, and pass
`--llvmcpu-target-triple aarch64-linux-gnu` or the vmfb is x86. Board script:
`planck:~/p7ab.sh`.

---

### What would change the verdict, in order

0. ~~**Re-measure the 44-site arm with depthwise ReLU6 fused**~~ — done
   2026-09-07, see above. Worth 2.5 ms of the 9.5 ms gap; the verdict holds
   at 1.053x rather than 1.26x.
1. ~~**P2's cross-op chaining.**~~ Built (P2 step 2) and measured
   2026-09-08 above: it takes the same single edge with or without the
   depthwise offloaded, because every depthwise input sits behind the
   explicit pad and 8 of the 14 direct edges fail the whole-atom rule. It
   cannot reach the `wait.npu` term, which is the one that binds.
2. **The pad.** Folding it into the dispatch (the driver already pads on the
   input packing path) removes the added CPU dispatch and its buffer -- and,
   since 2026-09-08, it is what stands between every offloaded depthwise and
   its producer's cube, so it is a P2 blocker on this model too.
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

### Measured 2026-09-06: the requantized path makes the int8 offload FASTER than the CPU

This is P8's lever 1, built and measured. **MobileNetV2-static-int8 with 29 of
its 34 dense convolutions on the requantized path is 1.54x faster than a
like-for-like CPU build on a full machine**, where the accumulator build it
replaces was 1.5x *slower*. Same model, same input, same pipeline; the only
difference between the two offload arms is `rocket-fuse-int8-requant-epilogue`
and the raised channel bounds.

Protocol, because a number here means nothing without it: `planck`, governor
`performance` on both A76 clusters, `iree-benchmark-module
--benchmark_min_time=3s`, six interleaved passes with the arm order rotating
each pass so no arm is ever always second, a quiet-NPU wait and a 2 s dwell
before every run. The CPU arm is `rocket-compiler --no-offload`, not plain
`iree-compile` (see the NHWC baseline trap). Medians of six:

| cpus | cpu-only | accumulator | requantized | requant vs cpu | requant vs accumulator |
|---|---:|---:|---:|---:|---:|
| `4,5` | 277.5 ms | 485.0 ms | 371.5 ms | 1.34x slower | **-23.4%** |
| `4-7` | 277.0 ms | 283.0 ms | 198.0 ms | **1.40x faster** | **-30.0%** |
| `0-7` | 267.0 ms | 240.5 ms | 173.5 ms | **1.54x faster** | **-27.9%** |

Spread is tight (the `0-7` requantized arm is 172-177 ms across six passes)
and the CPU arm is 267 ms in all six.

**Where the win is, from `ROCKET_PROFILE` at `0-7`** -- composition only, one
run each, which is what that instrument is for:

| phase | accumulator | requantized | delta |
|---|---:|---:|---:|
| `compact` | 50.3 ms | 21.0 ms | **-29.3** |
| `outside` | 182.5 ms | 162.5 ms | -20.0 |
| `wait.npu` | 43.4 ms | 30.6 ms | -12.8 |
| `record` | 34.4 ms | 26.2 ms | -8.3 |
| `pack.weights` | 65.8 ms | 65.8 ms | 0 |

Both arms issue the same 47 NPU dispatches, so none of this is dispatch count.
The largest single term is **compaction, more than halved**, which is the
mechanism working exactly as stated: the accumulator path writes `i32` through
128-byte native accumulator blocks, the requantized path writes `i8` into
16-byte slots, so there is a quarter of the output traffic to compact. `outside`
falls because the CPU requantization epilogues are gone, and the device itself
is quicker writing `i8`.

**What this does and does not overturn.** P8's law -- cost is flat per offloaded
dispatch and it parallelises -- still holds in shape: at `4,5` the requantized
arm is still 1.34x slower than the CPU, and the offload still needs cores. What
changed is the constant. The right reading is P8's own: the deficit was `i32`
activation traffic and unfused epilogues, and removing them removes it.
`pack.weights` at 65.8 ms is now the largest driver-side phase in both arms and
is untouched by this work -- it is the next thing to look at, and the weight
cache reports 0 hits over 49 misses, which is worth a look on its own.

### Where the remaining time is, and two levers measured to be worth nothing

Measured 2026-09-06 on the requantized build, after it started beating the CPU.

**`pack.weights` is a cold-start cost, not a steady-state one.** The single-run
profile shows 65.8 ms and reads like the largest driver phase; over 35
inferences it runs **49 times in total**, once per weight binding, with the
weight cache reporting **1666 hits, 49 misses**. It is doing exactly what
`weight_cache.rs` says it does. The visible effect is on the first inference
only:

    1 iteration    262 ms
    3 iterations   214 ms
    50 iterations  176 ms

So packing the coefficients at compile time -- expressing the blocked layout
as MLIR on the constant filter and letting const-eval fold it, with a
`Conv2DDef` flag telling the driver to skip its packer -- would buy about 68
of the ~86 ms cold-start penalty and 7.5 MiB of driver cache, and **nothing at
all in steady state**. Worth doing for first-inference latency; not a
throughput lever. Weigh it against carrying a second implementation of the
blocked coefficient layout, which is the kind of duplication that produced the
depthwise tap-major bug.

**Transpose propagation is still worth nothing, re-measured under conditions
that had changed.** P8 measured `iree-global-opt-propagate-linalg-transpose`
at 0% when every transpose fused into an `i32` epilogue that had to run
anyway. Those epilogues are now gone for 29 convolutions, so the premise
looked different and it was re-run. It is not:

| placement | 4-7 | 0-7 |
|---|---:|---:|
| shipped (29 requant / 5 accumulator) | 198.0 ms | 172.0 ms |
| + propagation, after the fusion pass | 196.0 ms | 175.5 ms |

-1.0% and **+2.0%**, six interleaved passes each, and it costs the classifier
matmul its offload exactly as P8 recorded (propagation folds the constant RHS
transpose into the matmul's indexing maps). Running it *before* the fusion
pass is worse still: it rearranges the epilogue chain enough that
`rocket-fuse-int8-requant-epilogue` declines, and placement inverts to 5
requantized / 29 accumulator. The transposes remain fused into elementwise
passes that run regardless. **Do not spend on this a third time.**

**What the CPU is actually doing**, 93 dispatch sites in the shipped build,
by kind:

| count | kind |
|---:|---|
| 13 | `elementwise_i32xi32xi32xi8` |
| 12 | `elementwise_transpose_i8` |
| 10 | `slow_memcpy` |
| 9 | `elementwise_i8` |
| 5 | `transpose_i8` |

The largest category was **one `i32` epilogue per depthwise convolution** --
all 13 depthwise convolutions were still on the `int8_accumulator` path, each
materializing an `i32` activation tensor and requantizing it on the CPU, the
exact cost the dense path had just shed for a 28% win. **That lever was taken
the same day; the next section has the result.** This breakdown is left as it
was measured, because it is what pointed at it.

### Requantized depthwise, 2026-09-06: 1.80x faster than the CPU

The follow-through on the section above. All 13 of MobileNetV2's offloaded
depthwise convolutions moved from `int8_accumulator` to the requantized path,
so the `elementwise_i32xi32xi32xi8` per layer -- the largest remaining CPU
dispatch category -- is gone. 42 of the model's 47 convolution dispatches are
now requantized.

| cpus | cpu-only | dense requant | + depthwise | vs cpu |
|---|---:|---:|---:|---:|
| `4,5` | 277.5 ms | 375.0 ms | 279.0 ms | **1.01x -- parity** |
| `4-7` | 277.0 ms | 196.5 ms | 158.5 ms | **1.75x faster** |
| `0-7` | 267.0 ms | 171.5 ms | 148.5 ms | **1.80x faster** |

Same protocol as the dense measurement. -13% to -26% against the dense-only
build. The two-core column is the one to notice: this configuration has
punished the offload in every measurement this repo has ever taken -- 3.9x
slower at its worst, 1.34x after the dense conversion -- and it is now level.
Accuracy is unchanged (max|diff| 0.3298, same argmax and top-5).

The hardware was measured first (`conv_depthwise_requant_hw.rs`, bit-exact at
six widths including all four MobileNetV2 asks for), so nothing here was built
on hope.

**A failure mode worth naming, because it is silent.** The first depthwise
matcher checked for a 1x1 kernel, copied from the dense 1x1 variant, and every
depthwise convolution in this model is 3x3. It declined everything and the
accumulator matchers claimed the convolutions straight back: no error, no test
failure, placement simply unchanged -- exactly what "the matcher does not
exist" looks like. The only instrument that catches it is a matched-case test
asserting the dispatch reaches the *specific* executable, which
`rocket_int8_requant_match.mlir` now carries for both dense and depthwise.

### Cin 1344 is exact in every isolated test and wrong inside the model

**Resolved 2026-09-08 as C13 -- see Resolved.** It was never this
convolution. The section below is kept as it was measured, because its
method (every shape-level instrument exact, the model wrong) is what pointed
at the chain rather than the dispatch. The bound now ships at 1344 and the
`Cout` floor at 16.

Open, 2026-09-06. Raising the requantized matchers' `Cin` bound from 512 to
1344 puts 32 of MobileNetV2-static-int8's 34 dense convolutions on the
requantized path and **breaks the model**: logits go from max|diff| 0.33
against a CPU arm to 5.01, mean 0.07 to 0.92, and the argmax moves from 447
to 977. Bisected by raising the bound one step at a time -- `Cin` 528 and 816
are both clean (max 0.43 and 0.33, argmax correct) -- to exactly one
convolution: **7x7 `Cin` 1344 -> `Cout` 448**.

That convolution is exact everywhere it is tested on its own:

| instrument | result |
|---|---|
| `dtype_boundary_probe`, `selectors-affine` | exact |
| `dtype_boundary_probe`, `onehot` (the addressing-sensitive read map) | exact |
| `dtype_boundary_probe`, `Cin` ladder to 1792, `Cout` ladder to 2048 | exact |
| `e2e_conv_regression` at the identical shape | **max error 0**, not 1 |
| same shape with a million-scale bias (`requant_int8_1x1_large_bias`) | exact |

So neither the shape, the addressing, nor the folded bias magnitude is the
discriminator, and the compiled differential -- which is the strongest
instrument here, since it runs the real compiler against a CPU reference on
real data -- is *bit-exact* for the convolution that ruins the model. What
differs in the model and not in the fixture is unidentified.

The bound therefore ships at `Cin` 816: the widest the model is measured
correct at, well below everything the isolated tests support. `Cout` was
raised 768 -> 1792 in the same session and is clean on the model.

**Do not raise `Cin` past 816 on isolated evidence.** This issue exists
because isolated evidence said 1792 and the model said 816. The next step is
an instrument that can see *which* dispatch first diverges inside a real
model -- comparing intermediate tensors, not logits -- because every
shape-level instrument this repo has already says the shape is fine.

### The ranked levers this leaves

1. ~~**Stop materializing `i32` activations and stop leaving their epilogues
   unfused.**~~ **Done and measured 2026-09-06. This was the whole 7.4 ms, and
   removing it reversed the deficit: MobileNetV2-static-int8 is now 1.80x
   faster than a like-for-like CPU build on a full machine and level with it
   at two cores, from 1.5x slower.** The structural version is the requantized
   int8 path -- `i8` out, bias on the BS plane, no CPU epilogue at all -- which
   removes the `i32` tensor rather than fusing passes over it.

   Three things had to happen and all three are in the tree: the path itself,
   which had been built on 2026-09-03 and left unmerged while both this file
   and ROADMAP.md ranked work against it; `rocket-fuse-int8-requant-epilogue`,
   which puts a real quantized model into the canonical form its matchers
   claim; and the same treatment for depthwise. 42 of the model's 47
   convolution dispatches are requantized. The two measured sections above
   carry the numbers and the protocol.

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

**The fp16 half of the phase numbers below is superseded** — P7 carries the
2026-09-07 profile, in which `record` is 1.5 ms rather than 15 and `outside`
is 70.9 rather than 88. The int8 figures here have not been retaken.


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
   **Scope of the refutation, stated 2026-09-09 so it is not over-read:**
   what was built and measured was transpose propagation
   (`iree-global-opt-propagate-linalg-transpose`) in a build whose every
   NPU result went through an `i32` CPU epilogue. It refutes *that*. It does
   not bear on the driver chain P2 later landed (which is worth 1.24x on
   ResNet50) nor on the compiler owning the packed layout as an edge
   encoding, which is COMPILER_ROADMAP.md section 6 and has not been
   measured at all.
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

## C11 (S2) — VGG int8 aborts under a repeated benchmark loop, on `main` as well as with the pooling matchers

Found 2026-09-05 while benchmarking VGG for the Phase 0 pooling work. **Not
introduced by that work**: the arm built from `main`'s own transform spec,
with no pooling offloaded at all, hangs identically.

`iree-benchmark-module` on `bench-vgg/vgg.int8.mlir`, `taskset -c 4-7`,
governors `performance`:

| `--benchmark_min_time` | main arm | branch arm |
|---|---|---|
| 0.001s | 982 ms, clean | 792 ms, clean |
| 0.5s | 997 ms, clean | 799 ms, clean |
| 2s | **abort**, hung-job floor | **abort**, hung-job floor |
| 5s | **abort**, hung-job floor | **abort**, hung-job floor |

Single `iree-run-module` invocations are clean and produce correct logits, so
the discriminator is repeated invocation in one process -- the same shape as
C8, but **not the same cause**: C8's `ROCKET_PM_DWELL=suspend
ROCKET_PM_DWELL_AT=transition` workaround does not clear this (still 1 hang,
both arms), and C8 itself is resolved.

VGG's int8 convolutions reach the NPU through the same `int8_accumulator`
path MobileNetV2 uses, and MobileNetV2's int8 arm no longer hangs, so
whatever this is either scales with something VGG has more of, or is a
different mechanism wearing the same symptom.

**What this blocks.** VGG can only be timed one iteration per process, which
means every VGG number in this file and in LIMITS.md includes a cold weight
cache. The comparisons are still valid -- every arm pays it equally -- but the
absolute numbers are pessimistic for the NPU arms and should not be quoted
against a steady-state figure from another model.

Next step: `ROCKET_DISPATCH_TIMES` to name which dispatch hits the floor and
at which iteration, then whether it is the int8 path alone (build a VGG arm
with only the conv matchers, no pooling) or the mix.

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

It has a cost of its own, found while closing C10: a wide row hits the CBUF's
11-bit slab base at `(K/32 - 1) * M > 2047` (M 89 at K 768), and past that the
planner splits it into column tiles — three of them for ViT-B/16's M 197 at
K 768. The notes' `1 x M` geometry has no wide row and row-tiles instead: two
tiles at the same shape here, since row capacity is 192 at 9 banks. Both
geometries are now measured exact on `planck` 2026-09-05 at M 90 and 197
(`dtype_boundary_probe` at `90x1`/`197x1` and `1x90`/`1x197`), so the choice
above M 32 is a performance question, not a correctness one, and the tall form
has fewer tiles. `whisper-encoder.md` records the notes hitting exactly this
wall with a width-on-time 1D conv and transposing to time-on-height.

The compiled path now carries a real shape: with the matcher bound raised to
2047 and ONNX's unit-batch `batch_matmul` collapsed in the spec, ViT-B/16's
twelve `197x768x768` out-projections offload as three-column plans, max|err|
0.0014 against the `--no-offload` arm and 5% faster than it (LIMITS.md,
Matmul). Whether the tall geometry would be faster still at those twelve
sites is the open half of this issue.

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

## C14 (S2) — the HuggingFace float16-converter ViT mis-imports through torch-mlir; the f32 import of the same model is exact

**[verified]** `onnx-community/vit-base-patch16-224-ONNX` ships both
`model.onnx` (f32) and `model_fp16.onnx`. ONNX Runtime runs both correctly and
they agree with each other to max|diff| 0.0400 with identical top-1 and top-5,
which is what fp16 costs on this model. Imported with `iree-import-onnx` and
compiled by `rocket-compiler --no-offload` -- **no NPU involved at all, host
CPU only** -- the two diverge completely:

| arm | vs ORT f32 | top-1 |
|---|---:|---|
| IREE, f32 import | **max\|diff\| 0.0000** | 767 (correct) |
| IREE, fp16 import | max\|diff\| 6.0187 | 600 (wrong) |

So the fault is in importing or compiling the converter's output, not in the
model (ORT computes it correctly) and not in this backend (the NPU is not in
the picture; with offload on, the NPU arm is bit-identical to this same wrong
CPU arm, which is how the fault was isolated).

Both graphs were preprocessed identically per README "ONNX models": all four
`dim_param`s pinned (`batch_size` 1, `num_channels` 3, `height`/`width` 224 --
ViT is unusual in leaving all four symbolic, not just batch), `value_info`
cleared, `shape_inference.infer_shapes` re-run, `checker.check_model` clean.
Same input bytes to both, fed as raw `.bin` (`--input=@x.npy` reads a npy
header as data -- see the note in the profiling section).

**What to do about it, today:** import the f32 model. The transform spec
demotes every all-f32 named convolution and matmul to f16 itself and
`rocket-promote-unclaimed-conv-inputs` restores f32 on whatever the match
loop did not claim, so the f32 import is the *better* arm regardless -- it is
exact against the oracle, and the fp16 the NPU actually runs is chosen per
operation rather than for the whole graph. Measured on `planck`: ViT f32
import, 73 NPU dispatch sites, max|diff| **0.0037** against both its own
`--no-offload` arm and the ORT f32 oracle, top-1 and top-5 unchanged -- an
order of magnitude *inside* the 0.0400 that fp16 costs in ORT alone.

This is the same shape of hazard as the note in the fp16 export recipe ("use
torch `.half()`, never the float16 converter"), but a different mechanism: the
converter did not corrupt the model here, the importer mishandled what it
produced. Root cause not yet localized -- the next step is a layer-wise
comparison to find where the two imports first diverge.

## C15 (S3) — RESOLVED as *implemented and off by default*: attention offloads correctly and is slower on both models measured (ViT-B 1.16x, BERT-384 1.41x); the CPU was barely spending anything on it

**[verified]** `rocket-compiler audit` on ViT-B/16 (f32 import) reports 74
convolution/matmul candidates: 73 matmuls accepted and one refusal, the
`224x224 Cin 3 Cout 768 k16x16 s16` patch-embed stem. But the section 3
*reconciliation* -- the half that reads final placement rather than the
decision record -- showed 25 contraction-shaped CPU dispatch sites, 24 of them
`batch_matmul`: twelve layers times `Q K^T` and `attn V`, the attention core.
They were absent from the decision record entirely, because
`readRocketCandidate` reads row-major `linalg.matmul` and these are
`linalg.batch_matmul` with a real head batch.

Worth noting what found this. The decision record alone said "74 candidates,
73 accepted" and read as near-total coverage; only reconciling it against
final placement showed 24 contraction sites the record had never heard of.

**Implemented** as `rocket-unbatch-matmul` (`RocketUnbatchMatmulPass.cpp`),
which splits a static-batch `linalg.batch_matmul` into one `linalg.matmul`
per batch element so the existing matmul path claims them, behind
`rocket-compiler --batch-matmul`. There is no way to give the descriptor a
batch instead: every batch element has its own right-hand operand and a
convolution shares one coefficient set across every pixel, so B independent
matmuls is what the hardware can run.

It works, and it is **correct**: ViT-B/16 goes 73 -> **361 NPU dispatch
sites**, the reconciliation drops to a single contraction-shaped CPU dispatch
(the k16x16 stem), and the logits are max|diff| **0.0074** against the ONNX
Runtime f32 oracle with top-1 and top-5 unchanged.

**And it is 1.16x slower**, so it stays off. `planck`, 8 workers,
`iree-benchmark-module`:

| arm | sites | ms |
|---|---:|---:|
| NPU, attention on CPU (default) | 73 | **592** |
| NPU, attention offloaded (`--batch-matmul`) | 361 | 687 |
| `--batch-matmul --no-offload` (like-for-like CPU) | 0 | 3471 |

**Why, from `ROCKET_PROFILE=1` on both arms** -- and this is the number that
settles it rather than the wall time:

| phase | 73 sites | 361 sites | delta |
|---|---:|---:|---:|
| `outside` (everything not this driver) | 208.8 | 205.8 | **-3.0** |
| `record` | 117.0 | 158.7 | +41.7 |
| `wait.npu` | 130.3 | 163.3 | +33.0 |
| `compact` | 93.1 | 128.7 | +35.6 |
| `pack.input` | 28.2 | 43.2 | +15.0 |

Offloading the entire attention core bought **3 ms** of CPU time and cost 125.
The CPU was barely spending anything on it: 24 dispatch sites, ~1.4 % of
`outside`. A FLOP count agrees -- attention is roughly 4 % of this model's
arithmetic, against the projections and the MLP -- but the measured share is
lower still, because those matmuls are small and the CPU runs small
contractions well.

Two details worth keeping. `wait.npu` went *up* 33 ms: the NPU takes longer
doing the attention matmuls than not doing them, because `[197,64] x
[64,197]` is a poor shape for the array. And these are the only matmuls in
the model whose *right-hand operand is an activation* -- attention multiplies
two things the model just computed -- so their coefficient packs can never
hit the weight cache the way a projection's do (288 fresh packs per
inference, ~0.16 ms each, ~45 ms).

**What would change the verdict.** `record` is 20 % of this model's wall at 73
sites and grows with dispatch count; that is COMPILER_ROADMAP section 4's
target, and section 4's status note now carries this measurement. If
compile-time plan materialization removed `record`, this trade would be
roughly 125 - 42 = 83 ms of cost against 3 ms of benefit -- still losing.
The honest conclusion is that attention offload needs the *shape* to get
better (a batched descriptor, or fusing the pair so the intermediate never
lands), not the dispatch tax to get cheaper.

**Confirmed on a second model, 2026-09-09: BERT-base at 384 tokens, 1.41x
slower.** The ViT measurement above could have been an artifact of one model,
one head width, or the per-dispatch cost as it stood that week -- so it was
re-run through `tools/model_survey.py` on a graph with no convolution in it at
all, whose attention has twelve heads against ViT-B's twelve at a wider
sequence, and after the epilogue-fusion and lazy-compaction work landed.
`planck`, f32 import, medians of 7 warm repetitions:

| arm | candidates | NPU sites | contraction on CPU | cpus 4-7 |
|---|---:|---:|---:|---:|
| NPU, attention on CPU (default) | 72 | 72 | 24 | **636 ms** |
| NPU, attention offloaded (`--batch-matmul`) | 360 | 360 | **0** | 894 ms |
| `--no-offload` (like-for-like CPU) | 72 | 0 | 96 | 5074 ms |

Same shape of result, larger margin. It is again *correct* -- every one of the
360 candidates offloads, `max|err|` 0.0165 against the f32 oracle on a
distribution of sd 0.60, argmax and top-5 unchanged -- and again the failure
is dispatch granularity rather than the matcher or the hardware. 24 sites
become 288, each a `384x64 x 64x384` too small to amortize its own dispatch,
and `ROCKET_PROFILE` on the default arm puts the per-dispatch host tax
(`pack.input` + `compact` + `record` + submit) at about 1.6 ms averaged over
72 dispatches -- so 288 more of them is a few hundred milliseconds of new cost
against attention work the CPU was not spending much on. The CPU dispatch
count *rises* too, 199 -> 1063, because splitting the batch multiplies the
surrounding reshape and transpose glue.

The same profile says where this model's time actually is, which is not
attention: of 636 ms, **314 ms (48%) is `outside`** -- layernorm, GELU,
softmax and the transposes -- **206 ms (32%) is `wait.npu`**, and **110 ms
(17%) is the NC1HWC2 round trip** (`compact` 69, `pack.input` 41). One
asymmetry in there is worth its own look: the MLP down-projection
(`384x3072x768`, `K` = 3072) takes **9.87 ms** per dispatch on the hardware
against the up-projection's (`384x768x3072`, `N` = 3072) **2.99 ms** for
identical arithmetic, because `K` becomes `Cin` and charges CBUF feature
residency while `N` becomes `Cout` and charges none. The up-projection pays
instead on the way out, 31 ms of `compact` per inference against 7.5.

## Resolved

What was settled and how, newest first, in place of the narratives — those are
in this file's git history (`git log -p ISSUES.md`). Everything cited below is
something that still exists: a commit, a file, or a memory.

**M2 (S3) — 2026-09-09. The NPU ran at 200 MHz; it now runs at 600, and the
lever was a driver patch that was already built.** `scmi_clk_npu` boots pinned
at 200 MHz (the vendor's idle `POWER_DOWN_FREQ`) because mainline `rocket` has
no NPU devfreq and the DT sets `assigned-clock-rates = <200000000>`. A patched
`rocket.ko` — `clk_set_rate` plus a coupled `regulator_set_voltage` in
runtime-resume, exposed as the module parameters `rocket_npu_clk_hz` and
`rocket_npu_uv` — is installed on `planck` and the board now defaults to
600 MHz at 0.80 V. Measured, bit-identical output, zero faults:

| model | 200 MHz | 600 MHz | speedup | npu share @200 | `wait.npu` mean/dispatch |
|---|---:|---:|---|---:|---|
| ResNet50-224 fp16 | ~123 ms | ~59.6 ms | **2.06x** | 76.0% | 1.888 → 0.656 ms (2.88x) |
| MobileNetV2 fp16 | ~87.5 ms | ~78.2 ms | 1.12x | 34.0% | 0.674 → 0.417 ms (1.62x) |
| MobileNetV2 int8 | ~81.3 ms | ~72.8 ms | 1.12x | 26.9% | 0.445 → 0.287 ms (1.55x) |

This issue estimated ~1.43x and ranked itself sixth of seven on "low ceiling
for the risk". The ceiling was 2.06x on the model where the NPU is the
bottleneck, and the risk was zero: the two shortcuts it warned about (a DT
override, a standalone `clk_set_rate` module) are genuinely dangerous, but the
in-driver version is not. What the estimate missed is that the payoff is
Amdahl-bound per model — `wait.npu` scales with the clock almost exactly
(2.88x against a 3x ratio) when dispatches are large, and barely at all
(~1.6x) when they are small enough that fixed submission latency dominates.

**Three consequences for the rest of this file.** Every board number recorded
before 2026-09-09 is a 200 MHz number, and their host/NPU balance no longer
holds — ResNet50's NPU share is 60%, not 76%. ROADMAP's per-op bar ("a site is
worth offloading when the op does more work than the dispatch tax") moved in
the offload-favouring direction for every op at once. And it makes multicore
fan-out *less* attractive, not more: `MULTICORE.md` §12 found fan-out pays only
above ~1 ms of hardware time per dispatch, and raising the clock shrinks
exactly that term.

Two traps. The suspend-time park to `ROCKET_NPU_POWER_DOWN_HZ` is itself gated
on `rocket_npu_clk_hz != 0`, so writing `0` while raised strands the clock at
600 and the "stock" arm silently runs fast — use an explicit `200000000`
baseline, and to truly restore stock write `200000000`, wait for every core to
read `suspended`, then write `0`. And `updates/` only wins over
`kernel/drivers/` after `depmod -a` runs; a KO dropped there without it is
inert. Board script: `planck:~/clkab.sh`. Memory:
`npu-clock-lever-600mhz`.

**C13 (S1) — 2026-09-08. A dispatch that consumed another dispatch's output
in the same command buffer read it before it was written.** `apply_ops`
walked the whole recorded command buffer up front -- packing every
dispatch's input, then handing `queue_execute` the list of jobs to submit,
wait on and compact -- so the second of two chained dispatches packed the
transient before the first had compacted into it, and the hardware saw an
all-zero input. IREE emits exactly that command buffer whenever two Rocket
dispatches have nothing on the CPU between them: one `hal.command_buffer`,
two dispatches, an `execution_barrier` the driver records as a no-op.

This is what P8's `Cin` 1344 anomaly and the requantized path's "`Cout` 24
is wrong" rule both were. Those two convolutions -- `1344 -> 448` and
`48 -> 24` -- are MobileNetV2's only two projection layers whose output feeds
the next convolution alone; every other projection also feeds a residual
add, and every expansion or depthwise output passes through a CPU ReLU6.
With both convolutions on the requantized path, the producer's s8->u8 shift,
the consumer's u8->s8 shift and the transpose pair fold away and the two
dispatches touch. Nothing else in any measured model does: fp16 convolutions
keep a CPU bias or clamp between them, ViT's matmuls have adds between them,
and the accumulator path always leaves an `i32` epilogue on the CPU. So the
bug hid behind two channel bounds that happened to exclude exactly those
edges.

Found by elimination on the board. Every value-shaped hypothesis was refuted
first with a new `magnitude` oracle pattern (constant input, unit weights,
optional BS-plane bias): accumulators to 877,824, a cancelling bias of
either magnitude, 0x80 feature bytes and the exact 7x7 `Cin` 1344 `Cout`
448 shape are all exact on the requantized path, both signs. Then the
structural one reproduced at `Cin` 64: `requant_int8_chain`, two 1x1
convolutions with the first's output as the second's input, 4048/4096
wrong (max error 181), and its output is bit-for-bit `requant(0 + bias)` --
a zero input. `requant_int8_chain_cpu_between`, the same pair with one CPU
clamp between them, is exact.

Fixed in `rocket-hal-driver`: `apply_ops_until_dispatch` applies the
recorded ops in call order and stops at each dispatch, and `queue_execute`
runs that dispatch to completion before asking for the next. M2's batched
tile submission inside a dispatch is untouched. Gates: the full compiled
conv gate (28 cases, chain and control included), matmul and pooling e2e,
and MobileNetV2-static-int8 with the requant bounds widened to `Cin` 1344
and `Cout` >= 16 -- max|diff| **0.35** against the CPU arm, same argmax and
top-5, where the same build was 5.01 / 4.71 and a different class. The CPU
arm itself is bit-identical to onnxruntime. Both bounds ship widened.

Two things worth keeping. **A hazard that hides behind a shape bound looks
like a shape rule**: both "rules" were measurements of the same bug, taken
on the only two shapes that exercised it. **And the model was the only
instrument that could see it**, because every fixture in every gate ran one
dispatch per function; chaining is now in the gate.

**C12 (S1) — 2026-09-08. The "11/1 silent-zero" 3x3 hazard was a
watchdog-killed job on a planner that no longer exists.** LIMITS.md's
*Hazards inside the limits* carried, and DYNAMIC_SHAPES.md DS3 escalated,
a finding this file never tracked: `Cin`=256/`Cout`=256/3x3 at 26x26 through
48x48 deterministically all-zero on an 11/1 CBUF split, recorded in
`@match_dynamic_conv2d_3x3`'s comment against a `Cout<=256` rule since widened
to 1792, with both evidence files gone from the tree. Re-measured on `planck`,
one extent per process: the current planner grants the shape **7/5 at every
extent from 20 to 58** (the streamed working set is five banks), and every
one is exact -- fp16 under `selectors` and `dense`, int8 under
`selectors-affine`. Forcing the old split back with `ROCKET_CBUF_SPLIT` at
30x30 brackets the mechanism: **9/3 and 8/4 exact, 10/2 and 11/1 a device
timeout** with the output unwritten (`0xa5a5` sentinel, `tile_mismatches`
covering every element), and the device clean afterwards. So the fault was a
starved coefficient grant -- the same class as C9 -- and the "all-zero
output" was the pre-C3 harness zero-filling a killed job and reading it as a
shape result. `streamed_weight_bank_preference` has prevented the grant since
it landed; `dense_k3_plan_never_starves_the_streamed_coefficient_working_set`
now pins it across both precisions, `Cin` 64..512, `Cout` 64..512 and every
even extent 8..64; no wire field can force a split, so a compiled model can
only reach `ConvPlan::new`. No runtime refusal was needed. Spec comments (both
copies), LIMITS.md and DYNAMIC_SHAPES.md corrected. Not a fix, a closure --
which is the point: a hazard nothing tracked cost a doc a severity it had not
had for a month.

**C9 (S2) — 2026-09-07. Neither half was what it looked like.** The `Cin`
cliff above 3x3 — a watchdog kill at ~500 ms, read as a hardware ceiling — was
our own CBUF split. The above-3x3 policies are read off the fp16 capture sweep
and never consulted `streamed_weight_bank_preference`; 7x7's fallback
hardcoded `(8, 4)`, four coefficient banks regardless of the stream. Forcing
every partition at the first hanging shape settles it: 7x7 fp16 `Cin` 72 is
exact at 1/11 through 7/5 and hangs only from 8/4 down.
`unstarved_large_kernel_partition` raises the captured grant to the streamed
preference and never lowers it; ceilings roughly triple (7x7 64 -> **208** for
the 2-byte family and int8, 32 -> **96** tf32, 128 -> **224** int4; 9x9 64 ->
**128**; 11x11 unchanged at 64). Same fault as the tf32 k=3 hang: **a grant,
not a size, is what breaks.** What is left is real — at 7x7 `Cin` 224 no
partition works, and 11x11 above 64 hangs at 1/11, the maximum grant — so the
old "ceilings fit `Cin * element_bytes`" reading is withdrawn; it was a fit to
our starvation boundary.

The **int8 half is retracted outright**: there was never a fault there. The
probe had no `SelectorsAffine` branch that day, so int8 fell through to
`Selectors`, which is not the int8 ABI and fails identically at 1x1 and 3x3.
It looked kernel-specific only because 1x1/3x3 are gated by the ladders and
never went through the probe.

Gates: `large_kernel_ceiling_matrix_matches_oracle` (19 cases, every precision
at its old cliff width and its new ceiling), `int8_large_kernel_matrix_matches_oracle`
(36), `conv_kernel_size_hw` (27 shapes), `tools/e2e_conv_regression.py`. Table
and reasoning in `large_kernel_max_in_channels` and LIMITS.md; memory
[[large-kernels-above-3x3]]. Two method lessons, both cheap and both missed:
**run the instrument at a shape already known good** (int8 at 1x1 refutes the
coefficient story in two minutes), and **when a knob is suspected, turn it**
(`ROCKET_CBUF_SPLIT` existed the whole time; "not a CBUF-split artifact" was
inferred from the planner's choices, never measured by forcing one).

**P6 (S3) — 2026-09-07. The residual closed itself.** Items 1-3 (packed
coefficient caching, the classifier matmul's per-inference re-narrowing, and
the host transforms' core placement) took MobileNetV2 fp16 from 198 to 146 ms
and are in this ledger already. What P6 kept open was one thing: `record`'s
15 ms per inference, still on a little core, with "one guard held across a
whole command buffer's recording" named as the fix.

**Do not build that guard.** `record` is now **1.5 ms** per inference, not 15
(`ROCKET_PROFILE=1`, `taskset -c 4-7`, m2 driver, 2026-09-07). M2's
per-context scratch pool removed it as a side effect -- a dispatch no longer
pays `CREATE_BO` + `mmap` + first-touch faults per tile, and recording was
mostly those faults. `rocket-hal-driver/MULTICORE.md` §12 predicted exactly
this shape (`record` 1.4 -> 0.22 ms per ViT dispatch) without noticing it
also discharged this item.

The phase profile P6 carried is superseded; P7 has the current one.

**C2 (S2) — 2026-09-07. Measured, and the finding is the opposite of the
one this issue proposed: RK3588 rounds half *away from zero*, so the oracle
was right and the change C2 asked for would have introduced the bug.**
`tests/conv_requant_tie_rule_hw.rs` classifies the rule instead of assuming
it. A 1x1 convolution at `Cin` 1 makes each output pixel's accumulator that
pixel's input byte, so one 16x16 job sweeps every `i8` accumulator through
`DPU_OUT_CVT`; coefficients are 1 and the BS plane is `BsEntry::default()`,
whose `2^14` multiplier and `>> BS_MULTIPLIER_SHIFT` are exact, so the only
rounding in the datapath is the one under test. Off a tie all three candidate
rules agree, which the probe uses as its own validity gate.

| shift | non-tie exact | ties | half-up wrong | half-away wrong | half-even wrong |
|---|---|---|---|---|---|
| 1 | 128/128 | 128 | 64 | **0** | 64 |
| 2 | 192/192 | 64 | 32 | **0** | 32 |

Both signs, 192 ties, no exceptions. Three consequences:

- **`conv2d_oracle.rs`'s `rounded_shift` is correct as written** and now says
  so with the measurement behind it. C2's action 1 is withdrawn.
- **`../rockchip-npu-notes`' RK3588 prediction is falsified.** Its
  `encodings/out-cvt-converter.md` measured round-half-to-even on RK3576 over
  40 ties and explicitly scoped RK3588 as predicted, never probed — *"no probe
  run there has separated truncation from round-to-even."* It has now been
  probed and the two parts differ. This is the contribution back that C2
  wanted; it is just the other answer.
- **`DPU_OUT_CVT_SHIFT.cvt_round` does not select the tie rule on this path.**
  It documents `0 = odd-in-even-not (round-half-to-even)` and `1 = carry 1 no
  matter what`, and `conv.rs` has always left it 0 — which is why half-to-even
  looked like the safe reading. Setting it changes **nothing** about the ties
  (still 0/128 wrong for half-away) and moves exactly one non-tie value, the
  most negative input: `-128 >> 1` returns `-65` rather than `-64`. So the
  register documentation does not describe this silicon here, and the bit is
  worth leaving clear for that stray value alone.

C2's action 3 — adopting the QNNPACK `+1`/bit-14-forced derivation in
`Multiplier::from_ratio` so the shipped path never sits on a tie — is dropped
rather than done. Its entire motivation was "independent of which rounding
rule wins"; the rule is now known and the model matches it, so forcing every
multiplier odd would perturb every shipped scale to buy nothing. Ties stay
reachable and stay correctly modelled.

One real defect fell out of it. `conv_kernel_shape_hw.rs` attributed its
rounding model to `conv_int8_probe_hw`, which measured the BS *gain* and never
drove a tie; the attribution is now the probe above. The model itself was
never wrong there — that harness's accumulator is a `usize`, so half-up and
half-away-from-zero coincide, and its `Int8` tolerance of 1.0 is wide enough
that the site could not have told them apart either way.

**C5 (S2) — 2026-09-06. Neither LUT quirk reaches this stack; the one real
defect the sweep found was a doc comment.** C5 asked whether QUIRK 4 (a `q = 0`
table entry mis-decoding to a garbage `~4.0`) and QUIRK 2 (a discrete `+128`
mux spike within ~±0.0015 of `x = 0` on signed-output kinds) apply to this
crate's int8-output `build_lut_regcmd`, and noted the existing gates could not
have seen either: each drives 6-13 hand-picked *uniform* fills per kind, and
the notes say a sparse gate steps straight over QUIRK 2's band.
`tests/lut_zero_join_hw.rs` answers both, board-run on `planck`, all green.

The mechanism that makes it cheap: `push_lut_tables_and_config` sets
`LE_START=-16384, LE_END=0, LO_START=0, LO_END=16384`, so **`LE[512]` and
`LO[0]` are both fixed-domain index 0, i.e. real `x = 0`** — the LE/LO join.
Every `q = 0` entry C5 lists except `LOG_LO[128]` (`log 1 = 0`) sits exactly
there, so QUIRK 4's target and QUIRK 2's band are the same neighbourhood and
one dense probe tests both.

- **QUIRK 2 does not occur.** At `input_scale = 1/32768` all 256 int8 codes
  collapse into `x` in `[-0.0039, +0.0039]` — ~100 of them inside the quoted
  band, and *every* code within the single 32-wide table cell on each side of
  the join. `tanh`, `erf` (the signed-output kinds C5 names), `square`,
  `sigmoid` and `exp`: **0 spikes, worst error 1.08 LSB**, and `tanh`/`erf`
  return exactly byte 0 at `x = 0`. int8 input is also why this is *complete*
  rather than a denser-sample-please: the reachable inputs are a lattice of
  spacing `input_scale`, so exactly one of them is `x = 0` and there is
  nothing finer to probe.
- **QUIRK 4 does not occur.** The `square` near-zero probe drives nothing but
  the six `q = 0` entries (`SQUARE_LE[510..512]`, `SQUARE_LO[0..2]`) and
  returns byte 0 at every code, worst error 1.5e-5. Stronger still:
  `SQRT_LE`/`RSQRT_LE`/`LOG_LE` are all-zero *placeholder* tables — 513
  consecutive `q = 0` entries each — and a negative input to `sqrt` returns
  ~0, not garbage, for all 128 negative codes. Every saturation the sweep does
  observe is a documented in-table clamp (`RSQRT_LO`/`LOG_LO`/`RECIPROCAL_*`
  at their Q15 limits), never a `q = 0` decode.
- **Tails driven.** The full-code sweep runs all 256 codes per kind in **one
  NPU job** (the LUT is pointwise and both cubes share geometry, so element
  `i` out is `f(element i in)`), covering both saturating ends of all nine
  kinds. 9 of 10 sweeps pass at **≤ 1 LSB** first time.

**The one real finding: `LutTable::log`'s domain was documented an order of
magnitude too wide.** Both doc comments claimed accuracy for `x` in
`[0.02, e)` while the same paragraph said the ceiling is `log(e) = 1.0` — a
Q15 encoding that cannot hold `log x` past `+1` cannot hold it past `−1`
either, so the floor is `1/e`, not 0.02. Hardware agreed with the table: codes
2..23 returned a flat `−1.0`, the `−32768` clamp, exactly as `LOG_LO[0..47]`
holds it. The true window is `[1/e, e)`, entries `48..=347`, `x` in
`[0.375, 2.711]`. `lut_log_hw.rs` already said `[1/e, e)` and its fills
already started at code 24; only the two library doc comments were wrong, and
both are fixed. A caller trusting the old bound and feeding `x = 0.05` got a
silent `−1.0`.

Method note for anything that reuses the patterned harness:
`lut_ramp_agrees_with_uniform_fill` is not decoration. It asserts that a
monotone input ramp yields monotone output *and* that the ramp's answer at four
codes equals the established uniform-fill harness's answer at those same codes.
Without it, a DPU_RDMA/DPU_WDMA walk-order mismatch would have failed every
oracle in the file for a harness reason and read as a hardware hazard.

**C10 (S2) — 2026-09-05. A wide input row returned silently wrong data, and
the planner never split it.** Two faults, both in `conv.rs`, neither in the
CBUF capacity arithmetic the issue first blamed. (1) A surface-layout line is
held in the CBUF in 32-channel slabs (one 64-byte entry of four atoms per
pixel) laid slab-major, and each slab's base offset is **11 bits**:
`(ceil(atoms/4) - 1) * in_cols` past 2047 wraps to the front of the line, so
the last slab of every pixel is read from slab 0. Boundary exact at 2048 at
every depth tried (fp16 K 96..1792, K 72/40, bf16, int16, tf32, int8
accumulator), per line not per tile (90x2 fails like 90x1), and independent of
the bank grant (fails at 9/3 as at 5/7); the `onehot` read map decodes the wrap
exactly. `Shape::max_tile_input_width` now bounds `in_cols`, `plan_grid`
declines a wider column, and the emitter refuses one. (2)
`max_tile_input_rows_for_width_and_data_banks` ended in `.max(1)`, forcing a
row through a grant it did not fit — what a 3x3 at height one hit, because the
coefficient floor leaves it 3-4 data banks (86x1 K 768: 2064 entries against
2048, wrong; 85x1 exact; the same 86x1 at 6/6 exact). It returns zero now.
Both fixes engage the existing column tiling, which is now board-validated at
1x1 and 3x3: M 90, 128, 197, 296 and every 3x3 point above, plus two clean
read maps through the split path. The matmul `M <= 32` matcher bound is no
longer load-bearing for correctness; see D1 for the geometry question that
remains. Memory `cbuf-entry-slab-base-limit` has the rule and the three
harness traps found on the way.

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
`021f0de`, `9f6426c`, `1bc3f62`. The residual P6 kept open after these
closed itself in turn -- see the 2026-09-07 entry above.

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

Revised 2026-09-09, after M2 (the 200 MHz clock) was resolved. **Read the
denominator warning first: every profile share below this line that predates
2026-09-09 was measured with the NPU at 200 MHz.** Raising it to 600 shrinks
`wait.npu` by 1.5-2.9x depending on dispatch size and leaves every host term
untouched, so host work is now a larger fraction of every model than the
numbers in this file say. ResNet50's NPU share is 60%, not 76%.

The 2026-09-09 re-measurement of P7 also produced the single most important
number in this list, and it is not about P7: **at four workers MobileNetV2
fp16's `--no-offload` arm is 54.5 ms against a best NPU arm of 73.1.** The CPU
baseline nearly doubles from four workers while every NPU arm gains 8-14%. On
this model the offload is underwater at a fair allocation, at either clock.
Rank work against that, not against the default-topology rows.

1. **P3's first half — the whole-BO cache sync, ∝ pages not bytes.** Promoted
   from 4. It is pure host cost, so the clock change made it a strictly larger
   share of every model, and it is independently named as the next lever by
   `MULTICORE.md` §12, where `fini_bo` over a 512 KiB replica input BO is most
   of the 0.47 ms per fanned-out ViT dispatch. The uAPI has no offset/length,
   so the fix is to stop syncing whole combined transient buffers rather than
   to sync them faster.
2. **P8's allocation finding, made actionable.** The 54.5-vs-73.1 result says
   the per-dispatch tax still dominates this model once the CPU is allowed to
   use its cores. P8 measured the constant at 7.4 ms on two A76s and 1.6 ms on
   eight and identified CPU slots as the only thing that moves it; nothing has
   attacked the constant itself since. This is the difference between "the NPU
   helps" and "the NPU helps only when the CPU is handicapped".
3. **P2's remaining matcher coverage** — f16 pooling and the classifier
   matmuls past the caps. P2 itself is closed; steps 2, 4 and 5 landed
   2026-09-08 and take ResNet50 169 -> 114 ms bit-identically.
4. **~~P7~~ — re-measured 2026-09-09 and closed as "not the deciding term".**
   The clock was worth the 8-10 points it was predicted to be worth, and flips
   the sign at the default allocation (1.080x faster at 600 MHz). It does not
   change the verdict, because the verdict is now set by item 2 rather than by
   anything about depthwise. `ROCKET_DEMOTE_DEPTHWISE=1` keeps the arm one
   compiler build away. Revisit when the offload is competitive at four
   workers.
5. **C4 → P4 → P1** — the rest of the dispatch-path stack, in increasing
   order of work.
6. **C11** — VGG int8 aborts under a repeated benchmark loop, which is why
   every VGG number in this repo includes a cold weight cache. Blocks a
   measurement rather than a user.
7. **C6, C7, D1, D2** — limitations, hygiene and reconciliation.

Done and in **Resolved**: the requantized int8 path (2026-09-06), C2
(2026-09-07), P6 (2026-09-07), C9 (2026-09-07), C12 and C13 (2026-09-08),
M2 (2026-09-09).

**Naming hazard.** Two different things in this repo are called M2: this
file's M2 was the 200 MHz clock (now resolved), and `MULTICORE.md`'s M2 is the
scratch pool and tile fan-out. Several references above and below are to the
latter. Check which one a sentence means before acting on it.

---

## Method note

Two things went wrong here repeatedly, and both are cheap to avoid.

**Single-knob sweeps of a multi-register geometry.** Three attempts missed C1
this way: the repo's `ROCKET_ACC_SURF_ADD` sweep, my `ROCKET_ACC_SIZE_E` sweep,
and the `surf_add` half of my first joint attempt. When a mode bit
(`mc_surf_out`) reinterprets what its neighbours mean, a null result from moving
one of them says nothing at all. Diffing against a known-good emitter is what
broke it open; no amount of further sweeping would have.

**Measuring on `planck` under governor `ondemand`.** The 2026-09-09 P7
re-test was run twice. Under `ondemand` it read the four-worker 200 MHz
control at 1.078x against this file's documented 1.18x, and gave the
default-topology 200 MHz control the wrong *sign* (1.02x faster against a
documented 1.05x slower) -- which nearly produced a "the verdict flipped"
conclusion from what was really CPU frequency noise. Under `performance` both
controls reproduce. The A76 cluster idles at a 408 MHz floor under `ondemand`,
so any host-bound arm is measured at whatever clock the governor happened to
pick. Check `scaling_governor` before quoting a number, and re-run a control
that disagrees with a recorded result rather than reporting the disagreement
as a finding.

**Degenerate test patterns.** `OraclePattern::Counting` sets every input and
coefficient to 1, so any shape whose output is constant across pixels and
channels — every 1x1 kernel without padding — cannot detect a permuted layout,
only unwritten lanes. It reported "0 mismatches" for a writer that was putting
every lane in the wrong place. Use `Dense` (varies in y, x, channel) for
anything that tests addressing, and treat a 100%-pass on `Counting` at k=1 as
evidence of coverage only. Better still, score the layout explicitly:
`ROCKET_ACC_LAYOUT_SCAN=1` names the cube instead of leaving "wrong somehow".
