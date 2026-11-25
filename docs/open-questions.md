# Open questions

A numbered list of what this stack does not yet know, across four layers:
the RK3588 NPU itself, the mainline `accel/rocket` kernel driver, the
iree-rocket runtime (`iree-rocket-hal` + `rocket-hal-driver`), and the
iree-rocket compiler (`rocket-compiler-plugin`, `rocket-compiler`,
`rocket-core`'s admission and planning as the compiler uses them). A fifth
section collects questions about measurement method.

Written against `main` at `b46c625` (2026-09-10); section 3.1, 4.9 and 5.1
reconciled against `2fe6ded` (2026-11-25). Each question names where it came
from in brackets, so it can be checked against its source rather than taken
on trust. A question here is *open*: if one of these is answered, move the
answer into ISSUES.md, LIMITS.md or the relevant roadmap and delete the
question, rather than annotating it here.

**The 2026-11-25 pass checked the questions whose premises had moved, not all
72.** What it found: one question answered in code (4.9.2), two resting on a
number that has since inverted (3.1.2, 5.1.1), and one stating a board
condition that is no longer true (5.1.3). The rest were read and left alone --
most of this file still holds, and a question citing an issue now in ISSUES'
**Resolved** section is usually citing it as background rather than being
stale.

How this relates to the other documents: [ISSUES.md](../ISSUES.md) carries
open defects with their evidence, [LIMITS.md](../LIMITS.md) what is measured
to work, [ROADMAP.md](../ROADMAP.md) and
[COMPILER_ROADMAP.md](../COMPILER_ROADMAP.md) planned work,
[DYNAMIC_SHAPES.md](../DYNAMIC_SHAPES.md) symbolic dimensions, and
[MULTICORE.md](../rocket-hal-driver/MULTICORE.md) the multi-context driver.
This file is the questions those documents leave open, pulled into one place
and arranged by layer. Many issues show up here as several questions, one per
layer they touch.

---

## 1. RK3588 NPU hardware

### 1.1 Completion and drain

1.1.1 What hardware signal actually reports that the DPU has drained a
depthwise write-back? The 1 ms depthwise-to-dense dwell exists because the
output fence appears to signal before the write-back finishes, and it costs
about 23% of the fp16 depthwise arm's wall. [`device.rs`
`DEPTHWISE_TO_DENSE_QUIESCENCE`, ISSUES P7]

- 1.1.1.1 Is the dwell still needed at all? Answered 2026-11-24: no
  transition the driver makes shows the hazard -- 60000 depthwise -> dense
  transitions without it, including in-job ones the kernel starts
  microseconds after the depthwise retires, plus 500 bit-identical ConvNeXt
  inferences. It is opt-in now (`ROCKET_QUIESCE_DEPTHWISE=1`). [`device.rs`
  `depthwise_quiesce_enabled`, ISSUES Resolved]
- 1.1.1.2 Can anyone reproduce the original failure (a dense job after a
  depthwise one writing only its first 16 channels) often enough to quote a
  rate to test a fix against? [`device.rs`]

1.1.2 Can a convolution retire on a PPU interrupt bit? The kernel's IRQ path
wakes on any of `DPU_0`, `DPU_1`, `PPU_0`, `PPU_1`, clears all seventeen bits
and signals the fence without recording which bit fired. A conv enables no
PPU stage, but both PPU bits are armed for every job. [`device.rs`, the dwell
doc comment]

1.1.3 What state does a hung (watchdog-killed) job leave behind, and why does
it make the device "sick" for the next job, sometimes in the next process?
[LIMITS "Hazards inside the limits", ISSUES Resolved C16; memory `npu-wedges-after-failed-job` "the wedge crosses processes"]

- 1.1.3.1 Is there a reset that clears it short of a reboot? Once a board
  hung uninterruptibly in `PREP_BO` after a benchmark left the NPU busy, and
  it did not reproduce after reboot. [ISSUES P2 step 5]
- 1.1.3.2 How long is the dwell between runs that the board protocol actually
  needs, and is that a property of the hardware or of the kernel's timeout
  handling? [ISSUES Resolved C16, `npu-wedges-after-failed-job` memory]

### 1.2 State that leaks between jobs

1.2.2 What is VGG int8's repeated-loop abort? C16 turned out to be a pooling
program (an unread trailing input row), and VGG's main arm offloads no
pooling at all, so it is probably a different mechanism. [ISSUES C11]

1.2.3 Which registers does each emitter (conv, depthwise, pooling,
element-wise, LUT) leave unwritten, relative to the union of all registers any
emitter writes? C8 is "a register one emitter sets and another
does not reset". An explicit audit might find the next one before a model
does. [ISSUES C8, Resolved C16; inferred]

### 1.3 CBUF

1.3.1 What actually bounds the large kernels? At 7x7 `Cin` 224 *no* CBUF
partition works (1/11 computes wrong values, the other ten hang), and 11x11
above `Cin` 64 hangs even at the maximum grant. The old "ceiling fits
`Cin * element_bytes`" reading was withdrawn because it fit the planner's
starvation boundary instead. [LIMITS "Above 3x3, `Cin` is the cliff", ISSUES C9]

1.3.2 Where do the two int8 CBUF bank-slack rules disagree? This repo bills
rounded `cbuf_atoms`, which is width-dependent. The notes prescribe a flat
`data_bank = min(fd_banks + 1, 11)`. Both are hardware-backed, so they must
disagree somewhere. [ISSUES D2]

1.3.3 Is a `Cout`-sliced tile worth building into `ConvPlan` for ~8% on Wide
ResNet50 (two layers, 1.44-1.46x each at 200 MHz) and nothing measured on
ResNet50 or MobileNetV2? `cout_group_probe` measured the scheme exact and
showed the task-count overhead that limits it to deep-tiled layers. Does the
answer change at 600 MHz, where compute shrinks but per-task overhead may
not? [ISSUES P4]

1.3.4 Does `ConvPlan`'s CBUF split still diverge from the vendor captures for
dense shapes above `Cin` 384, and does the "sawtooth reset rule" the spec
comment asks for exist? (The comment was written when
`MAX_INT8_INPUT_CHANNELS` was 512; it is 4096 now.)
[`rocket_conv2d_transform_spec.mlir` ~line 6500]

1.3.5 Why does BERT's MLP down-projection (`K` = 3072) take 9.87 ms on the
hardware against 2.99 ms for the up-projection (`N` = 3072) at identical
arithmetic? The working explanation is that `K` is `Cin` and charges feature
residency. Is that the whole story, and can the planner choose a geometry
that avoids it? [ISSUES C15]

### 1.4 PPU (pooling)

1.4.1 Why does a pooling window three or more rows tall, overlapping at
stride 1, hang past a measured width (3x3: 34 columns; 5x5: 20)? The vendor
corpus has direct 129-wide 3x3 captures, and this repo's register *set*
matches `rocket-userspace`'s `gen_pool_fp16`, so some other value differs.
[LIMITS "Pooling", `pooling-and-matmul-plan.md` "Still open"]

1.4.2 ~~What is the correct pad-fill identity for min pooling at any
precision, and for max pooling at int8?~~ **Answered 2026-11-24: none is
needed.** The PPU's max and min ignore `PPU_PADDING_VALUE` at both
precisions and clamp the window to the image (`*_pad_fill_probe` in
`pooling_oracle_hw.rs`; `avg_pad_fill_control_probe` shows the field live
for the average in the same windows). A pad as wide as the kernel hangs the
job instead. [ISSUES Resolved, `pooling.rs` `pad_fill_value`]

1.4.3 Is pooling at stride above 2 correct? Nothing above 2 has been measured
at any method. [LIMITS "Pooling"]

### 1.5 Numerics and the DPU output path

1.5.1 If `DPU_OUT_CVT_SHIFT.cvt_round` does not select the requantization tie
rule, what does it do? Setting it changes no tie, and it moves only
`-128 >> 1` from `-64` to `-65`. [LIMITS "int8 requantisation rounds half away
from zero", ISSUES C2]

1.5.2 RK3576 rounds half to even and RK3588 rounds half away from zero. What
other datapath behaviours differ between the two parts that the notes predict
for RK3588 but never probed? [ISSUES C2]

1.5.3 Does the DPU write zeros or stale bytes into an output cube's padding
lanes? If stale, fp16 garbage can be NaN, which a zero weight does not cancel.
The compile-time layout contract for non-whole-atom edges waits on this.
[COMPILER_ROADMAP 6.4]

1.5.4 Can a CNA-fed consumer be programmed at a channel count that is not a
multiple of 16 lanes (for example `Cin` = 24 at fp16) without reading the
padding surface? This is what MobileNetV2's whole-atom chain declines at Cout
24, 88 and 136 turn on. [COMPILER_ROADMAP 6.1]

1.5.6 For depthwise, which is right: this repo's `DPU_BS_OW_OP = 0` with
`od_bypass = 1`, or Mesa's `bs_ow_op = 0x80 - weight_zero_point` with
`od_bypass = 0`? This repo is board-exact for fp16 depthwise. Does the
divergence matter for int8 depthwise with nonzero weight zero points?
[ISSUES C6]

1.5.7 What are the reserved registers at `0x3030` (CORE) and `0x40c4` (DPU)
that every vendor and Mesa program writes as zero? [`builders/core.rs`,
`builders/dpu.rs`]

1.5.8 Is the EW unary `Add`'s scalar operand (`EW_OP_VALUE_0`, "from
configure register") read as a raw IEEE fp32? No capture uses that operand
source, and the encoding is marked UNCONFIRMED. Has `ew_round_hw.rs` settled
it? [`iree-rocket-hal/src/rocket/elementwise.rs`]

### 1.6 Capability that has never been measured

1.6.1 Does the hardware support stride above 1 at kernels above 3x3 beyond
the 7x7 stride-2 image stem? That one is measured exact on the dense path
(`Cin` 1..=4, `Cout` 32..=128, every rung; ISSUES Resolved, 2026-11-23).
Surface-layout inputs (`Cin` > 4), 5x5 and 9x9+, and stride 3 or more are
still unmeasured. [LIMITS "Kernel, stride, spatial and padding limits"]

1.6.2 Is there any hardware path to dilation, a depthwise channel multiplier
above one, or a batch dimension? Today the builder has none, depthwise fixes
`Cout == Cin`, and the wire format has no batch. [LIMITS, DYNAMIC_SHAPES DS2]

1.6.3 What does a full-iteration integer *output* writer for int16 look like?
int16 is explicitly waiting on one before anything should depend on it.
[LIMITS "Datatypes: two different menus"]

1.6.4 Is 600 MHz the right operating point? It is the patched driver's
default at 0.80 V. Is a higher OPP safe on this board's regulator and
thermals, and how much would it be worth now that ResNet50's NPU share is
about 60%? [ISSUES M2 (resolved)]

1.6.5 Is there an integer-safe form of contiguous task chaining? The notes
say the int32 CACC clears per kick rather than per task, which is what breaks
integer contiguous chains. [ISSUES C4, `regcmd-task-model.md` in the notes]

---

## 2. `accel/rocket` kernel driver (mainline)

2.1 What made multi-task jobs fail on v7.1: that kernel's IRQ-mediated
task transition, or an incomplete per-task regcmd, which produces the
identical symptom in the notes? On 7.3.5 they are exact: 58/58 multi-tile
cases as single jobs, up to 64 tasks, and every conv matrix in one-job mode
(ISSUES C4, 2026-11-23), and `rocket-hal-driver` now submits a dispatch's
tiles as one job per file, bit-identical on five models. The v7.1 kernel is
gone from the board, so this matters only if an older kernel has to be
supported. [ISSUES Resolved C4]

2.2 Would upstream accept per-core scheduling entities with a core hint in
`drm_rocket_submit.reserved`? That is the "30-line kernel patch" that would
let one file use every core, instead of N files each with its own memory.
[MULTICORE §7]

2.3 Is there an upstream path to sharing a BO across files (PRIME/dma-buf
import, or a shared IOMMU domain)? Today a job on file *k* can name only BOs
created on file *k*, which forces replica copies and collides with packed
IREE buffers. [MULTICORE §2, COMPILER_ROADMAP 6.1]

2.4 Would upstream accept an offset/length on `PREP_BO`/`FINI_BO`? Cache
maintenance is ∝ the BO's page count, not the bytes touched. The constant
buffer no longer pays it per dispatch (ISSUES P3, resolved 2026-11-23); what
still does is the output `PREP_BO` and a fanned-out replica's staged input.
[ISSUES P3]

- 2.4.1 Are NPU BOs mapped cached on the CPU side? Would an uncached or
  write-combined mapping make the remaining syncs unnecessary rather than
  cheaper? [ISSUES P3; inferred]

2.5 Could `SUBMIT` return an out-fence (sync_file or syncobj)? Today it is
fire-and-forget, `PREP_BO` is the only completion primitive, and there is no
status ioctl or register read. That rules out a userspace drain check (1.1.1)
and makes IREE semaphore integration thread-per-wait. [`device.rs` dwell
comment, ISSUES P8 "284 threads per inference"]

2.6 What is the upstream status of the two out-of-tree changes `planck` has
run: PPU completion and the `rocket_npu_clk_hz`/`rocket_npu_uv` clock and
regulator parameters? Which results in this repo would not reproduce on a
stock kernel? [MULTICORE §7, ISSUES M2]

Part answered 2026-11-23, when `planck` was re-flashed to Armbian
`7.3.5-edge-rockchip64` (reached as `ssh orangepi`). Its stock `rocket`
(upstream plus Armbian's RK3576 series) arms only the DPU interrupts, so
every pooling job is watchdog-killed at ~500 ms. Single-tile pools still
come back right, but `tiled_pooling_matches_oracle` fails, because only the
first task runs before the kill. The board now runs that module plus the
*unconditional* PPU patch (`drm-misc-next` `0001`, which arms PPU and DPU
for every job), not the flag-gated `093`/`PPU_DONE` one: this repo never sets
`DRM_ROCKET_JOB_PPU_DONE`.

The clock patch (`081`) is back on it since 2026-11-23, rebased onto
Armbian's module (one hunk hand-merged: its suspend path counts clocks
with `soc->num_clks`) and loaded with `rocket_npu_clk_hz=600000000`; so is
`088` (reset before IOMMU detach, see 2.7). At 600 MHz: gate 409/409,
every board model bit-identical to its 200 MHz output, ResNet50-224 fp16
~105 -> 45 ms and int8 150 -> 125 ms (cpu 4-7, `#60` build), MobileNetV2
fp16 63 -> 55.5 ms. The rebased patches are in `~/projects/patches/rocket/`
with an `-armbian-7.3.5` suffix.

- 2.6.1 Is NPU devfreq (OPP table plus governor) the right upstream shape for
  the clock fix, in place of module parameters? [ISSUES M2]
- 2.6.2 Can the suspend-time park bug be fixed so that writing `0` really
  restores stock? Today `rocket_npu_clk_hz = 0` while raised strands the
  clock at 600 MHz. [ISSUES M2]

2.7 What does the kernel do to a core after a job timeout? Does it fully
reset the core (CNA, CORE, DPU, PPU state and IOMMU), or only stop it? This
is the kernel half of 1.1.3. [LIMITS hazards, ISSUES Resolved C16]

Part answered 2026-11-23 (ISSUES C17): Armbian's `rocket_reset` detached
the IOMMU *before* resetting the still-running core, so the stall handshake
timed out on every timeout, and one burst was followed by page-table
corruption in the submitting process. With `088` (reset first) a timeout
recovers with nothing logged but `NPU job timed out`, and the next job on
that core runs correctly. Whether CNA/DPU/PPU state survives the reset is
still open.

2.8 Is the per-file 4 GB IOVA window ever a constraint in practice, for
example for an LLM's weights on one file? [ISSUES P1]

---

## 3. iree-rocket runtime (`iree-rocket-hal`, `rocket-hal-driver`)

### 3.1 Host cost

3.1.2 What is the per-dispatch tax made of now, and what is the largest host
term? The premise this question carried is gone: MobileNetV2 fp16 at four
workers is **10.0 ms offloaded against 54.2 ms `--no-offload`** (2026-11-25),
not the 73.1-against-54.5 that made the tax look decisive, and it reaches
that with *more* offloaded sites (54 against 37-44). The question survives in
a narrower form, because the tax is still the thing worth attacking on a
dispatch-heavy model: ISSUES C15's marginal attribution on a 338-site arm
puts `compact` first at ~24.5 ms an inference, `record` second at ~18.7 and
`pack.input` at ~9.1, where the same 50-site arm pays 3.0, 3.2 and 3.3. So:
is the NC1HWC2 round trip now the binding host term, and does P8's flat
per-dispatch model still describe it? [ISSUES P7 2026-11-25, C15, P8]

3.1.3 Would pre-warming the scratch pool at executable load, from sizes the
compiler already knows, remove the 15 to 45 ms per-process cold `record`
cost? [COMPILER_ROADMAP 4]

3.1.4 Is the depthwise dwell (1.1.1) the whole remaining cost of offloading
fp16 depthwise? With the dwell off (the default since 2026-11-24) the
depthwise arm landed within 1 to 2.5% of the arm that leaves depthwise on
the CPU. Re-measured 2026-11-24: the demote (on by default now) ties
`--no-offload` on MobileNetV2 fp16, 30.8 ms each, base 35.3; what is left is
the CPU pad in front of each depthwise. [ISSUES Resolved]
[`device.rs` `depthwise_quiesce_enabled`]

### 3.2 Multicore

3.2.1 Does tile fan-out across contexts still pay at 600 MHz? MULTICORE
found it pays only above about 1 ms of hardware time per dispatch. The clock
raise shrinks exactly that term. [MULTICORE §12, ISSUES M2]

3.2.2 Should `ROCKET_NPU_CORES` be chosen per model, or per dispatch from
its hardware time, instead of as a process-wide setting? The current advice
is "1 by default, 3 for ViT-shaped models". [MULTICORE §12]

3.2.3 Who owns a packed tensor when there are N contexts? Fan-out and
compile-time packed IREE buffers are in tension, and the second form of
COMPILER_ROADMAP 6.2 cannot be designed until this is decided.
[COMPILER_ROADMAP 6.1, 6.5]

3.2.4 For each shape, which wins: P1's multi-fd parallelism or P2's
single-fd cube chaining? The two are mutually exclusive. [ISSUES P1]

### 3.3 Failure semantics

3.3.1 What should an out-of-envelope dispatch do at runtime? Today it is a
hard `INVALID_ARGUMENT`, so a model's input data can fail an inference that
compiled cleanly. The options are a hard error, a runtime CPU fallback, or
carrying the envelope in compile-time assumptions. [DYNAMIC_SHAPES DS3]

3.3.2 Should semaphore failure propagate through
`iree_async_semaphore_fail()`? `async_on_fail` is a no-op TODO.
[`rocket-hal-driver/src/semaphore.rs:203`]

3.3.3 When `HUNG_JOB_DISPATCH_FLOOR` refuses a buffer after a hang, should
the driver attempt recovery (dwell, reset, retry) or always fail the
inference? [ISSUES Resolved C16, C3]

### 3.4 Correctness hazards still inside the limits

3.4.1 Where exactly does the dense ARGB path (`Cin <= 4`) corrupt multi-row
fetches, and is `Shape::dense_feature_offset_safe` a complete guard or only a
measured one? [LIMITS "Hazards inside the limits"]

A second dense hazard, found and fixed 2026-11-23 (ISSUES C17): the feature
DMA fetches `data_entries` pixels per tile, the row padded to the atom
width, so it reads past the tensor's end and hangs when that page is
unmapped. `ConvPlan::dense_input_read_bytes` is measured for full-width
tiles only; its column-tile half is a conservative bound, not a
measurement.

3.4.2 Is there a tool that compares *intermediate* tensors between the NPU
and CPU arms of a real model, to find the first dispatch that diverges? P8
asked for one after C13 showed that a model can be the only instrument that
sees a bug. [ISSUES P8 "Cin 1344 is exact in every isolated test",
C13]

---

## 4. iree-rocket compiler

### 4.1 Cost policy

4.1.1 What cost model should decide whether an admitted, plannable candidate
goes to the NPU? Admission and hardware legality are separate today, but a
cost policy is not. The report has a `cost` class that nothing produces.
[COMPILER_ROADMAP 2 and 3 "Still open"]

- 4.1.1.1 What inputs should it use? ROADMAP's per-op bar ("does more work
  than the dispatch tax") depends on core allocation, clock and chaining, and
  each of those has moved the verdict at least once. [ROADMAP "Recommended
  order", ISSUES P7, P8]
- 4.1.1.2 Should the policy target a specific CPU allocation? Several
  verdicts flip between `taskset -c 4-7` and `0-7`. [ISSUES P7, P8]

### 4.2 Coverage gaps on measured models

4.2.1 What should happen to the operations that are too big for any single
dispatch: VGG's classifier (`K` = 25088), Qwen3's LM head (`N` = 151936) and
other classifiers past the caps? Is decomposition into already-gated slices
worth its per-dispatch cost? [ISSUES Suggested order 3, COMPILER_ROADMAP 5]

4.2.2 Should the pooling matchers accept f16? VGG19's torchvision export
pools in f16, so its five pools stay on the CPU and break the chain between
blocks. [ISSUES P2 step 5]

4.2.3 Which stems should be claimed, and how? ResNet50's 7x7 stride-2 stem is
claimed at int8 (2026-11-23) and left on the CPU at fp16, where the NPU was
slower at 200 MHz -- should it be admitted at 600 MHz? (Re-measured
2026-11-23 at 600 MHz: slower on cpu 4-7, 53.6/48.8 vs 45.0/44.4 ms, faster
on all eight, 45.7/44.4 vs 47.4/50.4 -- still mixed, still out.) ViT's 16x16 stride-16
patch embedding and ResNet50 fp16's padded max pool still have no matcher.
[LIMITS ViT-L/16, ISSUES P2, C15, Resolved stem]

4.2.4 Should MobileNetV2's linear-bottleneck residual adds get a
`NONE`-activation twin of the epilogue-add target? It is described as "one
matcher away". [ISSUES P2 step 3]

4.2.5 Why are the stride-3 and stride-4 dense fp16 matchers written but not
in `foreach_match`, and why does depthwise NHWC fp16 have no strided matcher
when NCHW does? [LIMITS "What a compiled model actually offloads", ROADMAP
ledger]

4.2.6 When should int8 matmul, int8 pooling, pooling padding, unequal pooling
strides, average pooling at stride 2, and the runtime input zero point be
lowered? Each is already plumbed below the compiler. [ROADMAP "Expressible
and decoded, but not lowered"]

4.2.7 Should the compiler handle the N-K matmul form, so that transpose
propagation stops costing the classifier its offload? [ISSUES P8 "The
transposes are free"]

4.2.8 Does the canonical requant form need to accept `arith.fptoui` as well
as `arith.fptosi`? The model narrows with the former. [ROADMAP Phase 3]

### 4.3 Wire format

4.3.1 When, and in what order, should bf16, int16, fp16-with-fp32-accumulator,
tf32 and int4 get wire encodings? All five are board-validated in the HAL and
unreachable from `iree-compile`. [LIMITS "two different menus", ROADMAP
ledger]

4.3.2 Should the wire format express "two tasks, one job" (conv → LUT, and a
general conv → EW beyond `epilogue_add`)? [ROADMAP Gap 1, Phase 3]

4.3.3 Should per-output-channel weight zero points replace the single scalar
`weights_zero_point`? [ROADMAP ledger]

4.3.4 Should a wire field allow an explicit CBUF plan or bank override for
kernels above 3x3? Today automatic planning covers only 1x1 and 3x3. Would
such a field undo C12's "no wire field can force a split" guarantee?
[ROADMAP ledger, ISSUES C12]

4.3.5 When can the legacy `rocket-conv2d-v1` decoder be removed?
[`rocket-schema/docs/compatibility.md` "Migration"]

### 4.4 Plans and layout at compile time

4.4.1 Is COMPILER_ROADMAP section 4 (serialized tile plans) still worth
building? Its performance case was retracted. What remains is proven
register-program equality and a target/policy identity in the executable.
[COMPILER_ROADMAP 4]

4.4.2 Should serialized plans and the layout encoding share one version
field and target/policy identity, and which one defines it? [COMPILER_ROADMAP
6.5]

4.4.3 How should the second form of 6.2 work, so that a reader on a *later*
command buffer can chain and `runtime_dense_readers` can leave the wire? It
is blocked on 3.2.3. [COMPILER_ROADMAP 6.6 step 4]

4.4.4 Should a compile-time padding decision replace MobileNetV2's eight
whole-atom declines, and what is it worth? It is unsized and depends on
1.5.3 and 1.5.4. [COMPILER_ROADMAP 6.4]

### 4.5 Reduction (`K`) splitting

4.5.1 What are the accumulation semantics for a split reduction? That means
the precision partial sums combine at, per-slice zero-point correction,
integer overflow behaviour, and floating-point ordering and tolerance. The
CBUF grant at `Cin` 16384 is the only structural wall left.
[COMPILER_ROADMAP 5]

4.5.2 If CPU-side accumulation is the first implementation, does the mixed
plan ever beat leaving the op on the CPU? [COMPILER_ROADMAP 5]

### 4.6 Dynamic shapes

4.6.1 Where should a symbolic channel count's bound come from (a frontend
`util.assume.int`, an explicit compile-time bound, or something else), so
that the admission query has a range to check? [DYNAMIC_SHAPES DS1]

4.6.2 For the pooling and element-wise matchers, which still use
`dim_bounds`, should `flow.tensor.tie_shape` get a `ValueBoundsOpInterface`
external model, upstream or in the plugin? [DYNAMIC_SHAPES DS1]

4.6.3 Is a symbolic batch worth a wire field and a `conv::Shape` axis, or is
a per-batch dispatch loop (or folding N into H) enough? [DYNAMIC_SHAPES DS2]

4.6.4 How should `rocket-compiler audit` report "did this shape offload"
when the answer depends on runtime extents? [DYNAMIC_SHAPES DS5]

### 4.7 Attention and transformers

4.7.1 Is there a batched descriptor, or a fused `Q K^T` → `attn V` pair, that
would make attention worth offloading? Unbatching is correct and 1.16x
(ViT-B) to 1.41x (BERT-384) slower, and the CPU spends little on attention.
[ISSUES C15]

4.7.2 On BERT, 48% of the wall is `outside` (layernorm, GELU, softmax,
transposes). Which of these have a credible NPU path, given that softmax's
reductions have no PPU path outside a pooling window? [ISSUES C15, ROADMAP
"What this roadmap does not propose"]

4.7.3 Would the tall `1 x M` matmul geometry be faster than this repo's
height-1/width-`M` mapping for `M` above 32? Both are exact, and the tall
form needs fewer tiles. [ISSUES D1]

### 4.8 Import and frontend

4.8.1 Where do the HuggingFace float16-converter ViT and its f32 import
first diverge under `iree-import-onnx`? Is the bug in torch-mlir, in IREE, or
in the converter's output? [ISSUES C14]

4.8.2 Should the f32 import plus per-op demotion be the documented default
for every model, given that it was the better arm on ViT? [ISSUES C14]

### 4.9 Ceiling policy

4.9.1 Should the dense ceilings rise to 8192 at fp16 `k=1`, and `M` to 8192?
Both were measured clean at fp16 and stop at 4096 only because that is the
widest extent measured at every rung. Should ceilings be per rung instead of
precision-independent? [LIMITS, COMPILER_ROADMAP 5]

~~4.9.2 Should requantized int8 dense `Cin` stay at 1344?~~ **Answered in the
tree, 2026-11-24: it is 2048.** Raised on ResNet50-224 int8, whose layer4
`7x7 Cin 2048 -> 512` bottleneck inputs were its last convolutions on the
accumulator path and the only `conv -> conv` edges left that could not chain.
Gated by `int8_requant_cin2048_matches_oracle` and by the model against its
`--no-offload` arm; `Cout` went to 2048 the day before on the same model.
`admission.rs` carries the full rationale. Not deleted yet only because
LIMITS.md has no requantized-dense ceiling table to move it into -- that
table is the actual outstanding work here. [`admission.rs`, ISSUES Resolved]

---

## 5. Measurement method and documentation

### 5.1 Measurements that predate a change in their denominator

5.1.1 Which recorded conclusions depend on 200 MHz numbers and need
re-measuring at 600 MHz? Every board number before 2026-09-09 has an
inflated `wait.npu`. This question named **P8's law** as its first
candidate, and that one has now been collected: the law's MobileNetV2
measurement is overturned at 600 MHz *and* a fair allocation (ISSUES P7,
2026-11-25), though the clock is not what did it -- P2, P3 and the pad fold
are. **P1's "buys nothing yet" and the D1 geometry question are still
uncollected**, and so is every profile *share* quoted in this file from
before 2026-09-09, since raising the clock shrinks `wait.npu` by 1.5-2.9x
and leaves host terms untouched. [ISSUES header, Suggested order]

5.1.2 Which numbers were taken under `ondemand` before the method note
existed? [ISSUES "Method note"]

5.1.3 How representative is `planck`? Every board number comes from one
board, one kernel build and one input per model. Have any results been
checked on a second RK3588 (or RK3588S) board? [LIMITS ViT-B/16 note,
`planck-measurement-environment` memory]

Still one board, but now two kernels. Since 2026-11-23, `planck` runs Armbian
`7.3.5-edge-rockchip64` and is reached as `ssh orangepi`. The full
`iree-rocket-hal` hardware gate passes 400/400 there, in both debug and
`--release`, with the PPU patch in place (see 2.6).

Two corrections to the sentence that used to end this paragraph (2026-11-25).
**Model-level numbers have been re-taken on the new OS**: the 2026-11-24
ten-model survey ran there and every model beat its own `--no-offload` arm by
2.6x to 35x. And **the board does not run at 200 MHz** -- it runs at 600,
which is what the sentence had backwards. `/etc/modprobe.d/rocket.conf` sets
`rocket_npu_clk_hz=600000000` and
`/sys/module/rocket/parameters/rocket_npu_clk_hz` reads it back, so a board
number taken here since 2026-09-09 is a 600 MHz number unless the parameter
says otherwise. The second-board question is untouched: still one board, one
kernel build per OS, one input per model.
