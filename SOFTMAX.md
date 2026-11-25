# Online softmax on Rocket

A HAL-first design for `softmax` on the RK3588 NPU: which register-level
mechanisms it needs, which of those this repo already drives, which are
documented but have never been run here, and what the recurrence should
actually look like once the hardware's own shape constraints are taken as
given. Written 2026-11-24 against `main` at `74787f0`.

Read alongside the other four. [LIMITS.md](LIMITS.md) says what the stack is
*measured* to do; [ISSUES.md](ISSUES.md) carries the open defects;
[ROADMAP.md](ROADMAP.md) says which operations the hardware could run but the
compiler cannot reach; [COMPILER_ROADMAP.md](COMPILER_ROADMAP.md) is the
compiler-side plumbing. This file answers
[`docs/open-questions.md`](docs/open-questions.md) 4.7.2 -- "which of
layernorm/GELU/softmax/transposes have a credible NPU path, given that
softmax's reductions have no PPU path outside a pooling window" -- and
contradicts ROADMAP.md's "What this roadmap does not propose" entry for
`linalg.softmax`, which says to revisit once the pieces exist. They exist.

Evidence tags follow ISSUES.md: **[verified]** = read in this tree or in a
cited capture during this design pass; **[hypothesis]** = inference not
measured. Nothing below has been run on hardware as part of this document.

---

## The headline

Three conclusions, in the order they constrain the design.

1. **The softmax axis must be the width axis.** Every reduction this
   hardware has -- the PPU's pooling window, a strided convolution -- runs
   over space, not over channels. The FC/matmul lowering already puts the
   matmul's `M` on width at physical height 1 ([`fc.rs`][fc], and
   `rocket_core::fc`), so a score matrix can be *made* to arrive with the
   reduced axis on width by choosing which operand is the weight tensor.
   Once it does, the per-row statistics come out as a width-1 cube whose
   channel index is the row index -- which is exactly the index the DPU's
   two per-channel operand planes are addressed by. The whole design follows
   from that one alignment. The vendor's own compiled softmax does the same
   thing: its max reduction is a five-stage tree of PPU max pools whose
   *width* field shrinks 16, 11, 10, 9 while its channel count stays pinned
   at 32 [verified, `rknpu-spelunking/NOTES.md` Follow-up 5].

2. **The capability this design stands on is the BN operand plane, and it
   works** [verified on an RK3588, 2026-11-24, probe S1]. The DPU has two
   per-channel operand planes between the accumulator and the output
   converter: BS (add + multiply, fed by BRDMA from `BS_BASE_ADDR`) and BN
   (add + multiply, fed by NRDMA from `BN_BASE_ADDR`). This repo drove only
   BS -- a convolution's fp16 bias ([`tensor_layout.rs`][tl]'s
   `pack_fp16_bias_to_rocket`) -- and wrote `DpuRdmaNrdmaCfg` and
   `DpuRdmaBnBaseAddr` as zero in every builder it had. The BN plane is
   what broadcasts a per-row maximum back across the row and what applies
   the per-row reciprocal, and [`bn_plane.rs`][bn] now drives it: the ALU
   operand is a flat IEEE `f32` per channel, the MUL operand a flat IEEE
   `fp16`, each constant across every spatial position and indexed by
   logical channel past the first atom, applied as `x * m + a`. §7 has the
   measurements and §9 the one question left open.

3. **Online-ness pays for itself in fused attention and nowhere else.** The
   textbook reason for an online softmax -- never materialise the score
   matrix -- does not transfer: every task boundary on this device is a DRAM
   boundary, and the driver's NPU→NPU chain removes the *repack*, not the
   round trip (ISSUES.md's P2 steps 2 and 4). A
   standalone `linalg.softmax` is strictly better as two passes over the
   scores than as a streaming recurrence, because the streaming rescale
   lands on the score block itself and costs exactly what the second pass
   would have. Tile combination does not need it either: the HAL has to
   column-tile the reduction anyway (the PPU's direct path stops at 129
   input columns, and `Shape::max_tile_input_width` splits a matmul whose
   `M` overruns the 11-bit CBUF slab base, ISSUES C10), but if each tile
   writes its partial maximum into a disjoint width slice of one shared
   cube, merging them is **one more level of the same pooling tree** rather
   than a recurrence at all (§5.1). What is left is the one genuine serial
   dependency: a fused attention block's `[d_v, queries]` output
   accumulator, where the rescale is two orders of magnitude less traffic
   than the `[keys, queries]` score block it replaces. That, and only that,
   is the shape worth writing a recurrence for.

---

## 1. Shape: why width, and what that costs

### 1.1 The layout that makes everything line up

Take attention with `M` queries, `N` keys, head width `d`. Score matrix
`S[q, k]`. The FC lowering maps a matmul `A[m, k] x W[k, n]` to a 1x1
convolution with `m` on width, physical height 1, `k` on input channels and
`n` on output channels [verified, [`fc.rs`][fc] module docs, 160 captured
models, hardware-validated at M 90/128/197/296].

So compute the scores **transposed**:

```
S^T = K · Q^T            activations K  [N keys, d]    -> width = N, Cin  = d
                         weights     Q^T [d, M queries] -> Cout  = M
                         output      S^T [N keys, M queries] -> width = N, C = M
```

Nothing is transposed in memory to get here -- `S^T` versus `S` is only a
question of which of `Q` and `K` is handed to the weight path. Now:

* the softmax reduction is over keys, which is **width**;
* each row's statistic is a **width-1, M-channel** cube;
* physical height is 1, so a per-channel value is constant over the entire
  remaining axis -- a per-channel plane *is* a broadcast along the reduced
  axis. At height > 1 it would not be, which is probably why the vendor's own
  32x32x32 softmax capture does not use the plane.

And the second matmul consumes the probabilities as its **weight** tensor,
which keeps the queries on channels:

```
O^T = V^T · P^T          activations V^T [d_v, N keys]     -> width = d_v, Cin = N
                         weights     P^T [N keys, M queries] -> Cout = M
                         output      O^T [d_v, M queries]
```

The row sum is then free: **append one all-ones column to `V^T`**. Output
column `d_v` of `O^T` holds `sum_k P^T[k, q] = d[q]`, computed by the DPU in
its own accumulator precision, with no extra dispatch and no pooling tree at
all. The normalisation afterwards is a per-channel multiply over a
`[d_v, M]` cube -- small -- rather than over `[N, M]`.

### 1.2 What it costs

Two real costs, both pre-existing and both ISSUES C15's territory, not this
design's:

* `Q^T` and `P^T` reach the hardware through the **weight** path, so they
  need runtime coefficient packing (`rocket_core::weights::WeightPlan`,
  COMPILER_ROADMAP.md section 6.3). For `Q^T` that is per-invocation work
  on `d x M` values; for `P^T` on `N x M`. Weight packing is currently a
  cold-start cost because weights are constants; here it is not.
* `V^T` and the final `O^T` are transposed relative to the surrounding
  network's convention, so an attention block in this form has a transpose
  at its boundary unless the whole network is kept transposed. Attention on
  this device alternates axis conventions; one transpose per block is the
  floor.

A **standalone** `linalg.softmax` over the last axis of a 2-D tensor pays
neither: its input already has the reduced axis contiguous, which is width,
and its output has the same shape as its input.

---

## 2. The primitives, and their status

| Step | Mechanism | Status |
|---|---|---|
| row max | PPU max pool, kernel `k x 1`, stride `k x 1` | **exists**, and exact at atol 0 over `N` 8..512 as a tree in one job, probe S2 (`reduce_width_hw.rs`) |
| max over tiles | `EwBinaryOp::Max` on two width-1 cubes | **exists**; oracle-checked on `planck` by `ew_binary_hw.rs:346` and again across ViT's shapes at `:421`. The `EwBinaryOp::Max` doc comment still says "not measured here yet" -- that comment is stale and worth deleting. Needed only for §6's fused form; §5.1's combine does not use it |
| subtract the max | BN ALU operand from NRDMA, per channel | **exists**, as of probe S1: flat IEEE `f32` at `channel * 4`, broadcast over the cube, `bn_plane_probe_hw.rs` |
| `exp` | DPU LUT, `LutTable::exp_with_decaying_tail(1/20)` into `build_lut_q15_regcmd` | **exists**, int8 in and Q15 `int16` out, measured within **1.12%** total variation of an exact softmax on real ViT-B/16 attention rows (§3.4). The shipped flat tail is 6.69% and unusable |
| square (range extension) | `EwBinaryOp::Mul` with both operands the same buffer | **exists**, `ew_binary_hw.rs`, `ew_square_hw.rs` |
| row sum | depthwise all-ones **3x3 at stride 3** on a height-1 cube; or the all-ones column on the second matmul (§1.1) | **exists**, and exact at atol 0 over `N` 9..512 as a tree in one job, probe S2 (`reduce_width_hw.rs`). The window is 3x3 and not `1 x k` for the reason in §2.2. The avg-pool route is gone at fp16, §2.1 |
| reciprocal | the **host**: `M` f32 divides on a width-1 vector | **measured** at 0.006 ms for 64 rows, plus one 0.037-0.059 ms submission boundary (S4). The LUT cannot take `d`'s 200:1 range through an int8 input, §4 |
| divide by the sum | BN MUL operand from NRDMA, per channel | **exists**, as of probe S1: flat IEEE `fp16` at `channel * 2`, shift 0, same file |

Nothing in that column is a reverse-engineering problem any more. The two
rows that were "documented, never run here" when this was written are the
ones S1 measured; everything else was already gated.

### 2.1 Padding semantics are free for both reductions

A reduction tree over an extent that is not a multiple of the kernel needs a
partial last window, and both reductions get the right answer without the
caller doing anything:

* **max**: a padded tap is **clamp-to-image** -- the 35-bit
  `PPU_PADDING_VALUE` is inert for max and min, measured 2026-11-24 across
  thirteen encodings at int8 and six at fp16 (`pooling_oracle_hw.rs`'s
  `*_pad_fill_probe`) [verified, `pooling.rs:105`]. A partial window
  therefore returns the max of its real taps.
  Measured: exact at atol 0 at `N = 197`, whose stages pad by 3 then 7
  (probe S2).
* **sum via avg**: the PPU divides by `kh * kw` whether or not a tap was
  padding (count-include-pad) and the avg fill is zero, so `avg * kh * kw`
  would be the sum of the real taps exactly [verified, same doc comment] --
  **except that an fp16 average cannot be programmed at all here.** This
  paragraph used to end with a two-level tree recovering the sum as
  `avg_final * 256`; probe S2 found that shape refuses at program-build
  time and every other height-1 average with it.

  The reason is the axis *not* being reduced. `RECIP_KERNEL` holds the fp16
  bit pattern of `65536 / k` on each axis [verified, `pooling.rs:636`, and
  the bug this fixed], so the identity divisor a height-1 window needs on
  the height axis is `65536 / 1`, past fp16's 65504 ceiling. int8 has no
  such limit and programs 65536 plainly, which is how we know this is a
  precision restriction and not a geometry one
  (`an_fp16_average_cannot_reduce_width_only`,
  `an_int8_average_can_reduce_width_only`). It is the same ceiling that
  makes a plain **sum** pool unencodable, so there is still no
  `PoolingMethod::Sum` to add -- but now there is no fp16 average to carry
  a factor out of either. §2.2's two DPU routes are the sum, not a
  preference over a third.

### 2.2 Prefer the DPU for the sum

This section used to rank the DPU above an fp16 PPU average on accuracy.
After S2 the ranking is moot: there is no fp16 PPU average to rank against
(§2.1). Both remaining routes run on the DPU:

* the all-ones column of §1.1 (fused attention): the DPU accumulates, which
  is fp32 for fp16 inputs (LIMITS.md's precision table);
* a **depthwise** convolution with an all-ones `1 x k` kernel at stride `k`
  (standalone softmax): depthwise is exact to `Cin` 1536 at fp16 and int8
  (LIMITS.md), strides are correct since the
  demote-pass fix (ISSUES.md), and the kernel is a
  compile-time constant so it packs once.

So: the PPU max pool for the max, where S2 shows it is exact and where
there is no alternative anyway, and the DPU for the sum, where at fp16
there is now no alternative either.

**The window is 3x3 at stride 3, not `1 x k`,** and that is the planner's
choice rather than a preference. Asked directly, it refuses a `1 x k`
window at any stride but 1 ("non-square kernels currently have capture
backing only at stride 1"), and at stride 1 such a window is a sliding box
filter rather than a reduction. It refuses an *even* square window at any
stride but 1 as well, which rules out the 2x2/s2 halving tree and 4x4 and
8x8 with it. What is left is the one strided square kernel the corpus
backs, 3x3 at stride 3, laid on a height-1 cube: its vertical taps are
padding, a convolution's padding is zero, so the window sums three real
columns. `only_an_odd_square_window_can_stride` pins both refusals, so a
later widening of the admission rules shows up as a choice rather than as
a surprise -- an 8x8 window would reduce a row in two stages where this
takes six.

The horizontal geometry needs care the max tree did not. A convolution has
leading padding registers and no trailing ones -- the output extent implies
the right edge -- and that extent is floor-mode, so a row whose length is
not a multiple of 3 silently drops its tail unless the *leading* pad makes
up the difference. With `p = (3 - N % 3) % 3` the windows tile `[-p, N - 1]`
exactly: disjoint, every real column covered once, and `p <= 2` stays under
the kernel. One all-ones coefficient buffer serves every stage of the tree,
since the channel count and kernel do not change
(`one_all_ones_coefficient_buffer_serves_every_sum_stage`).

---

## 3. `exp`: an int8 block, and the domain is not the problem

`LutTable::exp()` carries the real captured vendor `EXP_LE` table and
`EXP_LUT_BN_SCALE_K = 3276.7184`, fit against that table's own content
[verified, `activation.rs:176`]. Its `LO` half (`x >= 0`) is a documented
placeholder returning a constant 1.0, and -- as that doc comment explains --
is not fixable in the Q15 entry encoding, because `exp(x) > 1` for any
`x > 0`. So:

> **The max this design subtracts must be a true upper bound, not an
> approximation.** A surrogate that is ever below the real maximum sends some
> element into the placeholder half, which silently returns 1.0 instead of
> something greater than 1. The error does not announce itself and the later
> rescale cannot correct it. Every "cheap approximate max" shortcut is
> therefore closed. Use a real reduction.

### 3.1 What probe S3 found, and what it overturns

The rest of this section was written before the `exp` step had been run on
anything but int8, and S3 overturned most of it. Two measured facts, the
second decisive:

**The LUT block is int8, and this repo's build of it is int8 everywhere.**
`build_lut_regcmd` programs neither `in_precision`, `out_precision`,
`proc_precision` nor `mrdma_fp16tofp32_en` anywhere, so all four sit at 0 --
"Integer 8bit" -- and [`LutShape`][act] carries int8 zero points and
quantization scales to match. Every table in `activation.rs` has only ever
run that way, `exp` included. Softmax's `exp` consumes a matmul's **fp16**
score cube, so `build_lut_fp16_regcmd` sets those four fields the way
`build_unary_regcmd` sets them for its own fp16 datapath. The task runs, in
milliseconds, and `exp(0)` comes back exactly 1.0 -- so the fp16 cube does
reach the table.

**But `OUT_CVT` is an integer multiply-shift, so a Q15 table entry cannot
leave the task as a fraction.** The entries are `real_value * 32768`, a
fixed-point encoding independent of any op's quantization, and recovering a
real number means multiplying by `1/32768`. At `(scale 1, shift 15)` the
entry for `exp(0)`, 32768, comes out as exactly 1.0 while the LE table's
edge entry, 221, comes out as exactly **0** -- `221 >> 15` in integer
arithmetic, against 0.00674 in real arithmetic. That is decisive on its own
and independent of everything below
(`out_cvt_cannot_turn_a_q15_entry_into_a_fraction`).

So **the `exp` step's output is a fixed-point Q15 integer, not an fp16
probability**, and §7 has to deal with that. The promising direction is to
carry it: softmax is a ratio, so a constant factor of 32768 in both the
numerator and the denominator cancels, and the sum reduction of §2.2 could
run on the Q15 values as well. What that costs is the design's remaining
open question -- it is now the first thing S5 has to answer, ahead of §4's
reciprocal.

A second problem is recorded but not diagnosed: at `input_scale = 1` the
index reaches the table's edge by `x` of about `-1.2e-4`, where the
multiplier below predicts `-5.0`, so the effective multiplier is four
orders of magnitude too large. Two mechanisms were tried and neither fits
(see `the_fp16_index_does_not_track_the_real_value`). It is not on the
critical path while the converter finding stands, since both would have to
be solved to keep `exp` in fp16 and only one of them has to be solved to
move to Q15.

### 3.3 The envelope, and the slope that makes it usable

§3.1 left the Q15 output as §9's open question. Measuring the envelope on
the **int8** path -- the one `lut_exp_hw.rs` has gated against an oracle --
turned out to matter more than the output format, and to have an answer.

**The domain edge is exactly `-5.0`, and below it the shipped table holds a
flat `6.74e-3`** [verified, `lut_exp_envelope_hw.rs`: the output tracks
`exp` down to `-5.00` and is a constant 28 counts at `1/4096` from there to
`-8.0`]. That is not an accuracy problem, it is a correctness one. A softmax
over `N` keys puts up to `N * 6.74e-3` of spurious mass in its denominator
-- 2.6 at `N = 384`, against a true denominator of order 1 -- and nothing
downstream recovers it.

**The flat clamp is a programmed choice, and undoing it fixes the tail.**
`LutTable::exp()` ships `le_slope_uflow_scale = 0`; `sigmoid()` carries a
real slope, "consistent with sigmoid's tail not being fully flat even far
from zero". Measured, that register drives a straight line in the table's
index space, and it runs *through* zero into negative output if the slope is
too steep -- `(4420, 14)` crosses zero at `-5.25` and hits int8's floor by
`-6.5`. A negative probability is worse than the flat clamp, and the EW
stage's ReLU does not stop it: `ew_relu_bypass = 0` on a LUT task is
**inert**, byte-identical output either way
(`the_ew_relu_is_inert_on_a_lut_task`).

So the slope is tuned to the input domain instead. At an int8 `input_scale`
the most negative real input is `-128 * input_scale`, and a slope taking the
edge entry's 221 counts to zero over exactly that distance leaves the tail
monotonically decreasing, non-negative everywhere, and exactly zero at the
bottom -- **for no extra task**. That is
`LutTable::exp_with_decaying_tail(input_scale)`; at `1/16` it is
`(368, 14)`, and the hardware reaches 0 at exactly `x = -8.0`
(`the_tuned_tail_decays_to_zero_and_never_below`).

### 3.4 The pair, tuned against real attention and measured

`input_scale` and the slope trade against each other -- the int8 input's
quantization wants a fine scale, since a code step of `s` puts up to `s/2`
of relative error on every weight, while the `-5` domain edge wants a
coarse one. Picking the pair at the domain's corner is not the same as
picking it where the data is, so it was picked against real data.

`fixtures/vit_b_16_attention_rows.f32` is 64 rows of real pre-softmax
attention logits, already reduced by their own row maximum: torchvision's
ViT-B/16 on one image, scores recomputed from each block's `in_proj_weight`
(`output_attentions` returns probabilities, not logits), sampled across
every layer. `tools/fixtures/dump_vit_attention_rows.py` regenerates it.
Its shape is what decides the pair: median `-3.44`, and **28% of all
entries fall past the table's `-5` edge**, 8.7% past `-8`, 1.4% past `-16`.
So the tail is not a corner case in a real transformer -- it is a quarter
of the data.

The figure of merit is **total variation** against an exact softmax, half
the L1 distance between the two distributions, which bounds how far the
attention output can move. Relative error on an individual weight is
useless here: it is dominated by weights of order 1e-9 whose absolute error
is nil, and reports ~100% at every scale including the best one.

| `1/s` | reach | slope | denominator err | TV, tuned | TV, flat clamp |
|---|---|---|---|---|---|
| 24 | -5.3 | 3315 | 1.98% | 2.13% | |
| **20** | **-6.4** | **789** | **0.70%** | **1.11%** | 6.69% |
| 16 | -8.0 | 368 | 1.38% | 1.60% | 6.77% |
| 12 | -10.7 | 195 | 3.51% | 3.25% | |

**Measured on the board at the chosen pair: 1.12% mean total variation,
against the host sweep's 1.11%** -- and 6.69% for the flat clamp, against
its predicted 6.69%. The hardware and the piecewise model agree on both
arms to a hundredth of a percentage point, which is what makes the sweep
usable for choosing the pair in the first place
(`the_tuned_pair_reaches_one_percent_on_real_attention_rows`,
`the_flat_clamp_on_real_attention_rows`). So the tuned slope is a measured
**6.0x**, not a predicted one.

Two caveats worth carrying. The optimum is shallow -- `1/16` costs 1.6% and
`1/24` 2.1% -- so the pair is not delicate, which matters because a
different model's logits will not have ViT-B/16's exact distribution. And
the *worst* row of the 64 is 14% rather than 1%: the mean is good and the
tail of the row distribution is not, so a model-level gate should report
both.

This also closes §9's Q15 question on the output side. The end-to-end
figure above is measured through `build_lut_q15_regcmd`, which writes the
table's Q15 integer to memory as `int16` with the converter at identity --
`out_precision = 1`, `size_e = 1`, `OUT_CVT (0, 1, 0)`. It worked first
try, and the 32768 never has to be divided out at all: the sum and the
division are a ratio, so it cancels.

### 3.2 The domain arithmetic, corrected

Everything here still describes the table; what has changed is that the
"stored value" is an int8 code rather than an fp16 number, and that the
conclusion at the end was wrong.

The table is indexed in a fixed-point domain `[-16384, 0)` for the `LE` half,
and the hardware maps a stored element to that domain by `index = x *
input_scale * bn_scale_k`. `input_scale` is an ordinary `f32` field on
[`LutShape`][act], not a constant. So the function computed is
`exp(x * input_scale)` over a stored-value domain of
`[-16384 / (input_scale * K_EXP), 0)`:

| `input_scale` | computes | stored-value domain | recovery |
|---|---|---|---|
| 1 | `exp(x)` | `[-5.0, 0]` | none |
| 1/2 | `exp(x/2)` | `[-10.0, 0]` | one `Mul(u, u)` |
| 1/4 | `exp(x/4)` | `[-20.0, 0]` | two squarings |

At `input_scale = 1` the domain is `[-5, 0]`: everything past `-5` flat-clamps
to `exp(-5) = 6.7e-3`, and for `N = 384` keys that is up to `2.6` of spurious
mass against a denominator of at least 1. **The default configuration is
unusable for softmax**, which is worth saying out loud because `exp()` reads
as ready.

`input_scale = 1/2` plus one squaring remains the right choice, but the
reason given here was wrong and worth correcting rather than deleting. The
claim was that the useful range is bounded by the Q15 *output* encoding at
about `x = -10.4` rather than by the input domain, so a wider domain buys
nothing. Arithmetic alone refutes it: the table's edge entry is
`round(exp(-5) * 32768) = 221`, nowhere near the Q15 floor of 1, so nothing
inside the table underflows and **the domain edge binds first at every
scale**. Below the edge the output is a flat clamp at that edge entry,
`exp(-5) = 6.74e-3`, not a decay toward zero -- which is what makes the
domain worth buying: at `input_scale = 1/2` with one squaring the clamp
becomes `(6.74e-3)^2 = 4.54e-5`, which over 384 keys is 0.017 of spurious
mass against a denominator of at least 1. `input_scale = 1/4` with two
squarings would take it to `2.1e-9`, and that is real accuracy rather than
domain the output cannot express.

Accuracy: entry spacing is `32 / K_EXP` in fixed-point domain units, so at
`input_scale = 1/2` the real-x step is `0.0195` and the 513-entry
(512-interval) table's interpolation error is about `5e-5` relative; squaring
doubles it to `1e-4`. fp16's own epsilon is `9.8e-4`, so the LUT is not the
limiting term.

**No table regeneration is needed for the domain.** It is a `LutShape`
field and one existing EW task -- though §3.1 means neither is reachable at
fp16 as things stand. (If a later pass wants to avoid the squaring dispatch, a
self-derived `EXP_LE` over `[-11, 0]` is the move -- the `LE` half has no
encoding obstacle at all, unlike `LO`, and `square()`/`erf()`/`sqrt()`
established the methodology. It is an optimisation, not a prerequisite.)

---

## 4. Normalisation: the multiply is free, the reciprocal is the host's

In the §1.1 layout the divide by the row sum is a **per-channel multiply**,
which is `bn_mul_src = 1` with a `1/d` vector in the NRDMA plane. In fused
attention it applies to the `[d_v, M]` output; standalone it is one DPU pass
over `[N, M]` with the LUT bypassed. Either way the multiply itself is free
-- it rides a stage that is otherwise bypassed, and S1 measured the operand
as a flat fp16 at `channel * 2`.

Producing `1/d` is the part this section was written to agonise over, and
probe **S4 settled it the other way from the ranking below: it runs on the
host.** Three measurements decide it.

**`d`'s real range.** Over the 64 real ViT-B/16 attention rows of §3.4,
`d = sum(exp(x - m))` runs from 1.03 to 122.8 with a median of 10.8, and
**44% of rows are at or past 16** -- the edge of `LutTable::reciprocal`'s
real data. The worst case is `N` itself, since a uniform row sums to `N`.
So the shipped table covers a bit over half of a real model's rows.

**Why a wider table does not help.** The blocker is the *input*, not the
table. An int8 code space is 128 linear steps and `d`'s range is 200:1, so
the step at `d = 1` is worth more than `d` itself, and a Newton iteration
cannot recover what the input quantization threw away because it multiplies
by the same quantized `d`. An int16 LUT input would have the codes and a
self-derived table over `[1, 256]` would have the range -- but the Newton
step then needs `y0 * d` as a *fraction*, and S3 measured that the LUT's
output converter is an integer multiply-shift, so a Q15 entry cannot leave
the task as one. The NPU route therefore needs two unprobed mechanisms and
a new table.

**What the host costs.** The reciprocal is `M` f32 divides on a width-1
vector: **0.006 ms** for 64 rows, measured. The one real objection was §5's
one-job structure -- the reciprocal sits between the sum and the normalize,
so computing it on the host splits one dispatch into two submissions -- and
a second submission inside the same dispatch is not a second IREE dispatch,
just one more `SUBMIT` and one more `PREP_BO` wait: **0.037 to 0.059 ms**
on the smallest real job (`the_extra_submission_boundary_costs`). Against
ISSUES P8's 1.6 to 7.4 ms of flat *per-dispatch* host cost, that is a
rounding error.

**Measured end to end.** §7's passes 2 and 3 on the real rows, with `exp`
and the normalizing multiply on the device and the reciprocal on the host,
come to **1.12% mean total variation** against an exact softmax -- the same
figure as §3.4's `exp` step alone, so the fp16 plane multiply adds nothing
measurable. Worst row 13.97%, which is §9's spread item rather than a new
one (`exp_then_a_host_reciprocal_normalizes_real_attention_rows`).

**One thing the plane will not take.** The multiplier has to be `1/d` in
*real* units, not `1/d_q15`. With `exp` emitting Q15, the summed counts run
to `32768 * N`, so `1/sum` is `1.5e-7` to `3e-5` -- below fp16's normal
range of `6.1e-5`, and `write_bn_mul_plane_fp16` refuses it rather than
silently flushing a channel's multiplier to zero. So the factor of
`1/32768` has to ride the Q15-to-fp16 conversion instead, where it is exact
because it is a power of two. That conversion is a pass over the whole
`[N, M]` cube (0.18 ms on the host for `64 x 197`), and moving it onto the
device is §9's open item -- not the reciprocal, which is 64 divides.

### 4.2 The Q15-to-fp16 bridge, and the DPU's two datapaths

§9 carried the Q15-to-fp16 conversion as the last host step and the only
one in §7 that is a pass over the whole `[N, M]` cube rather than a width-1
vector -- 0.18 ms for `64 x 197`, against the reciprocal's 0.004. Probe
**S4a** moved it onto the device (`build_q15_bridge_regcmd`), and it takes
**two** tasks rather than the one it looked like, for a reason worth
knowing beyond softmax.

**An `int16` input works, and it makes the datapath integer.** `exp`'s Q15
counts read back exactly as fp16 through `in_precision = 1` with
`mrdma_fp16tofp32_en` clear -- 221 to 221.0, 1 to 1.0 -- which is the first
integer input any task in this crate has taken. But with that input the
whole post-processing path is integer-valued: a BN MUL plane of `2^-13`
turns 16384 into 2.0 correctly and 4096 into **0** instead of 0.5. So the
int16 read can hand on exact integer counts and nothing else.

**An fp16 input gives the float datapath**, where the same plane is exact
down to at least `2^-16`, power-of-two subnormals included. So the constant
`2^-15` that turns a Q15 count into a probability goes in a *second* task,
on the fp16 path, and §7's pass 3 multiplies by `1/d` after it.

Neither shift stage can stand in for that plane. `BS_MUL` at shift 15 and
`OUT_CVT` both return 0 for a Q15 count, on the fp16 path as well as the
int16 one: every scaling stage in the DPU but the BN plane is an integer
multiply-then-right-shift, and an integer multiply-shift cannot produce a
value below 1. The plane's fp16 operand is the only true fractional
multiplier the DPU has.

And the two multiplies cannot be folded into one plane, which is what makes
this two tasks and not one: their product is `1/d_q15`, a non-power-of-two
subnormal worth about 2% of relative precision at a real row's median `d`
and 24% at the worst.

Two of this design's own bugs came out of it, both in code that reads fp16
back. `write_bn_mul_plane_fp16` encoded an exact subnormal four times too
small, which showed up on the board as a factor of four before it showed up
anywhere else, and `exact_power_of_two_subnormals_encode` now pins it. And
the `f16_to_f32` helper copied across **fifteen** hardware test files -- not
just this design's -- reported every subnormal a factor of four low, its
renormalising counter starting at -1 where it has to start at 1. That one is
pre-existing and harmless in a test whose values are of order 1; it is not
harmless in one reading a softmax's small weights, so it is fixed in all
fifteen. Neither of §3.4's or §4's measured total variations moves: both
come out at 1.12% after the fix, because a weight small enough to be
subnormal is too small to shift an L1 distance.

### 4.1 The routes this replaces

Kept because the reasoning is what a later widening of the LUT's input
would have to overturn, in the order they were ranked before S4 measured
anything:

1. **`rsqrt` then square then one Newton step.** `rsqrt` is real on
   `[1, 16)` too, but `1/sqrt(d)` over `d` in `[1, 384]` has an output
   range of only 20:1 against the reciprocal's 384:1, so a widened
   `RSQRT_LE` keeps about `6e-4` relative accuracy where a widened
   reciprocal table would give `1.2%`. Square it, then
   `y1 = y0 * (2 - d * y0)` squares the error to about `1.4e-6`.
2. **One Newton step on a widened reciprocal table**, landing at about
   `1.4e-4` from a `1.2%` seed.
3. **Log-domain normalisation**, `softmax(x) = exp(x - m - ln d)`, folding
   the denominator into the same BN ALU operand as the max. It removes the
   multiply entirely but needs a self-derived `log` table whose output
   exceeds Q15 for `d > e`, and it makes the `exp` pass depend on two
   serial reductions.
4. **The host**, which is where S4 put it.

All four assumed the LUT could take a usable input for `d`. It cannot, and
that is the finding rather than the ranking.

---

## 5. Job structure: one job, an ordered task array

The reduction tree looks embarrassingly parallel, and the arithmetic is. The
hardware's unit of parallelism is not the unit the arithmetic suggests,
though, so it is worth being exact about what a `RocketJob` buys.

**Tasks within one job run sequentially on a single core.** `drm_rocket_job`
has no core field; core assignment is the kernel's, and a file's
`drm_sched_entity` "picks the least-loaded core only when its own queue is
empty, so while it has queued jobs it stays put" [verified,
`submit_jobs`'s doc comment and MULTICORE.md §2]. N tasks in one job buy
capacity, not parallelism. **N jobs** are what the scheduler can place on
different cores, and `submit_jobs` exists for exactly that.

**But intra-job ordering is free, and it is a real dependency edge.** The
kernel dispatches task `k+1` only after task `k`'s own hardware completion
IRQ fires, so a buffer written by one task and read by the next needs no
CPU-side wait, no `prep_bo`/`fini_bo`, and does not even have to appear in
the job's handle lists -- "there is no kernel-mediated sync point between
tasks in the same job for the CPU to wait on anyway" [verified,
`submit_tasks`'s doc comment; `build_conv_then_lut_regcmd` is the
hardware-validated instance]. A whole dependent chain is therefore one job.

**And cross-job buffers do not exist.** A fanned-out job lives on its own
file, each file gets a fresh IOMMU domain and `drm_mm`, a BO is mapped into
exactly one file's domain at `CREATE_BO`, there is no PRIME and no
`gem_prime_import_sg_table`, and `SUBMIT` resolves handles with
`drm_gem_objects_lookup(file, ...)` [verified, MULTICORE.md §2 against
mainline `rocket`]. So the N partial maxima of a fanned-out reduction land in
N *different* IOVA domains, and no single task can read them. M2 solves the
same problem for convolution tiles by replicating the inputs per context and
**gathering on the host** (`compact_atomic_output_rect`, MULTICORE.md §12);
there is no device-side alternative.

That rules the plain version of "N buffers, then combine on device" out, and
points at something better:

### 5.1 The combine is a tree level, not a recurrence

Have each tile task write its partial maximum into a **disjoint width slice
of one shared cube** on the same file -- which is what
`OutputPlacement::SharedImage` already does for convolution tiles, via each
tile's `output_offset`. The partial maxima then sit contiguously as a
`[tiles, M]` cube, and the combine is **one more PPU max-pool level over
that cube**: a single task with `kernel = tiles x 1`, no `EwBinaryOp::Max`
chain, no accumulator, no new primitive. The tree does not care that its
elements came from different tasks.

**Measured** (probe S2, `the_combine_is_one_more_tree_level_over_one_shared_cube`):
at `N = 384` the first stage plans 3 column tiles, every column of the
shared cube holds the max of its own window whichever tile wrote it, and
two further stages finish the row -- all as one job's ordered task array,
exact at atol 0. The sequential execution inside the job is exactly the
ordering the tree needs, and `PoolingPlan::programs_with_buffers` already
emits tiles that write disjoint column offsets of one output buffer, so
this needed no new HAL mechanism.

It did need one HAL *fix*. `PoolingPlan::new` derived its tile count from
the output-width cap alone, but a tile emitting `w` columns reads
`(w - 1) * stride + kernel` input columns, so a stride as wide as the
kernel makes the input cap the binding one: an 8x1 window at stride 8 over
384 columns balanced to a single 48-column tile and then read 384 input
columns against a limit of 129, tripping the function's own `assert!`
rather than splitting. Every extent above 128 was affected, which is every
transformer extent. The cap is now the lower of the two, which can only
narrow a split and leaves every capture-backed one alone (at 3x3/s2 over
256 columns both caps come to 64; at 2x2/s2 over 258 both come to 128).

Whole-row cost, for the record: 512 columns reduce in 3 stages and 6 tasks,
197 in 3 stages and 4 tasks.

So: **structure A -- one dispatch, one job, one ordered task array holding
the whole of §7.** One flat per-dispatch host cost, which is the term that
actually dominates (ISSUES P8's law, re-measured: the cost is flat per
offloaded dispatch and set by core allocation, 1.6 ms on eight A76s to 7.4 ms
on two). Serial on one core, every dependency free.

### 5.2 When to fan out anyway

**Structure B**: fan the per-tile `S^T = K_j Q^T` + max-pool work out across
contexts as N jobs, gather the partial maxima **on the host**, then submit a
second job for the global max, the `exp` pass and the sum.

The reason this is worth considering here, when M2 found the gather cost ate
the fan-out win on MobileNetV2, is that the thing being gathered is a
width-1, `M`-channel vector per tile -- a few kilobytes, not an output
rectangle. The one structurally expensive part of M2's replication model is
cheap for a reduction, because a reduction's result is small by definition.
What it still pays is M2's other costs: a replica of `K_j` and of the packed
`Q^T` coefficients per context, `fini_bo` over the replica input BO's pages,
and a second submission's round trip. On ViT at N=3 the hardware term fell
8.4 to 3.4 ms per dispatch while the host term grew; which side wins for a
softmax-shaped dispatch is a measurement, not a derivation.

Build A first. B is a strict refinement of A -- same programs, same plans,
different grouping -- and `ROCKET_FANOUT=0` already exists to A/B it.

### 5.3 The cost of a task is not zero

Every task boundary inside a job costs a completion IRQ, and on `planck` the
NPU IRQs land on an A55 while the driver's host work defaults to the slow
cluster unless pinned (the measurement-environment note in ISSUES.md). A
reduction tree of many tiny tasks pays one IRQ each. Two consequences for
§7: prefer the widest pooling kernel that the direct path accepts over a
deeper tree of narrow ones, and prefer the all-ones-column sum of §1.1,
which costs **zero** extra tasks, over any sum tree at all.

---

## 6. Where the recurrence goes

§5.1 removed the statistics from the recurrence: a `[tiles, M]` cube of
partial maxima reduces in one pooling task, so the max and the sum need no
running state. What remains is the output accumulator of a *fused* attention
block, where the dependency is real -- a tile's contribution has to be
rescaled by however much the maximum moved after it was computed.

Let a tile `j` be one column tile of keys. Write `m_j` for the tile max,
`M_j` for the running max, and `O_j` for the running `[d_v + 1, M]` output
accumulator.

```
M_j = max(M_{j-1}, m_j)                       EwBinaryOp::Max, width-1 cube
r_j = exp((M_{j-1} - M_j) / 2)^2              EW Sub + LUT + Mul, width-1 cube
O_j = O_{j-1} * r_j + V_j^T · P_j^T           BN MUL plane on the accumulate
```

with the denominator carried as column `d_v` of `O_j` (§1.1), so the sum
telescopes through the same recurrence as the output and needs no separate
state. The first two steps run on width-1, `M`-channel cubes -- a few
hundred elements. The per-tile score block is read once.

Note what this costs that §5.1 does not: the chain is serial in `j` by
construction, so a fused attention block built this way **cannot** use
structure B's fan-out for the phase that owns most of the arithmetic. The
two-pass form can. That trade is the thing to measure before committing to
a fused op.

Costs, for one head of BERT-384 (`M = N = 384`, `d = d_v = 64`), fp16:

| | passes over `[N, M]` | width-1 tasks | `[d_v, M]` tasks |
|---|---|---|---|
| three-pass safe softmax | 3 | `2 * ceil(log_8 N)` reductions | 0 |
| two-pass (§7) | 2 | same | 1 |
| online, standalone | 1 | `3 * tiles` | `tiles` rescales of `[N, M]` -- i.e. pass 2 by another name |
| online, fused attention | 1 | `3 * tiles` | `tiles` rescales of `[d_v, M]` |

The third row is the point. For a standalone softmax the streaming rescale
touches the score block itself, so one pass plus `tiles` rescales is two
passes with extra dispatches. **Build the two-pass form for
`linalg.softmax`, and the online form only inside a fused attention op.**

---

## 7. The two-pass form, concretely

For `linalg.softmax` over the last axis of `[M, N]`, laid out as width `N`,
height 1, channels `M`. "Pass" below means a pass over the data, not a job:
the whole of this is **one job's ordered task array** under structure A
(§5), with every intermediate living in device memory the CPU never touches.

```
pass 1   PPU max pool, kernel k x 1, per tile  [N, M] -> [tiles, M]
           tiles write disjoint width slices of one shared cube (§5.1)
         PPU max pool, kernel tiles x 1        [tiles, M] -> [1, M]    m
         HOST: negate m, scale by the exp task's own BN multiplier,
           write it as i32 per channel (§4.2, S5b)
pass 2   DPU LUT task, flying mode:            <-- Q15 int16 output, §3.1
           BN ALU operand <- NRDMA, -m per channel: the subtraction
           happens *here*, not in a stage of its own (S5b)
           input_scale = 1/20, exp_with_decaying_tail (§3.3, §3.4)
           -> P in Q15                         [N, M] -> [N, M]
         depthwise all-ones 3x3/s3 conv tree   [N, M] -> [1, M]        d
pass 3   bridge task 1: int16 -> fp16         [N, M] -> [N, M]
           an integer read cannot also scale (§4.2)
         bridge task 2: BN MUL plane = 2^-15   [N, M] -> [N, M]
         reciprocal of d: HOST (§4)            [1, M] -> [1, M]
         DPU task, BN MUL operand <- NRDMA     [N, M] -> [N, M]        y
           the operand is 1/d in real units, not 1/d_q15: see §4
```

Pass 2's `Mul` and pass 3's multiply are the two dispatches a self-derived
`EXP_LE` over `[-11, 0]` and a fused epilogue would remove; both are
optimisations with the primitive set unchanged.

A note on the plane operands' formats, now measured rather than assumed
(S1). Pass 2's ALU plane is **one `f32` per programmed channel, dense at
`channel * 4`** -- the same form the fp16 BS bias takes
(`rocket_fp16_bias_storage_size` = `channels * 4`,
[`tensor_layout.rs`][tl]). Pass 3's MUL plane is **one `fp16` per channel at
`channel * 2`**, half the pitch, which is a real trap for a helper written
to serve both: `bn_plane_bytes` and `bn_mul_plane_bytes` are separate
functions for that reason, and a test loads a MUL plane with a sentinel
over the bytes an ALU-sized one would have occupied.

Getting the reduction's output *into* a plane is the step this section
originally waved at and S2 turned into a real one. The argument was that a
width-1, height-1 cube's NC1HWC2 layout collapses to dense -- each surface
is one 16-byte atom, surfaces are consecutive, so channel `c` lands at
`c * element_bytes` -- leaving only an fp16-to-fp32 widen. That holds for
the **DPU's** writer. It does not hold for the **PPU's**: that path rounds
both cube strides up to four pixels, so a width-1 pooling output puts
channel `c` at `(c / 8) * 64 + (c % 8) * 2`, with 48 bytes of nothing
between consecutive surfaces
[verified, `the_reduced_vector_is_four_pixel_strided_not_dense`].

So the max tree's output is a strided gather away from an ALU plane
operand, not a widen away. **That bridge is built and measured** (S2a):
`build_plane_gather_regcmd` is one DPU pass-through that reads the pooled
cube through `RDMA_SURF_NOTCH` -- "number of pixels from the end of this
process's feature map to the end of the underlying shape's" -- and writes a
width-1 cube, which is dense because `DST_SURF_STRIDE` counts atoms. One
task closes both the pitch and the element type, and §8 has what it cost to
find. The two alternatives that bullet listed are not needed: stopping the
tree at 4 columns to finish on the DPU, or gathering `M` values on the host
and paying a round trip inside §5's one-job structure.

The sum side needs no bridge. A conv's output cube has no four-pixel
rounding -- its surface stride is `width * height` atoms -- so the sum
tree's width-1 fp16 output is already dense at `channel * 2`, which is
exactly the MUL plane's format.

---

## 8. HAL work plan

Each step lands behind an oracle-based hardware test in
`iree-rocket-hal/tests`, following the pattern `lut_exp_hw.rs` and
`pooling_oracle_hw.rs` set. Steps S1 and S2 are the only ones that can fail
for a reason this document cannot predict.

**S1 -- the BN operand plane. DONE 2026-11-24** (`bn_plane.rs`,
`tests/bn_plane_probe_hw.rs`, run on the board). Seven questions, each a
test that decides between named candidates rather than asserting a guess,
so the run reports the answer whichever way it goes. What it found:

* **The plane is read.** A zeroed plane with BN engaged is the exact
  identity -- which is what makes anything else attributable to the
  plane's contents rather than to engaging BN -- and a loaded one changes
  the output.
* **The ALU operand is a flat IEEE `f32` at `channel * 4`**, the same form
  the fp16 BS bias takes. The fixed-point `i32` forms BS uses at int8, at
  shifts 0 and 14, were both ruled out in the same run.
* **The MUL operand is a flat IEEE `fp16` at `channel * 2`**, shift 0 --
  *not* the same width, and not fixed point. This one was found the hard
  way and is the probe earning its structure: every candidate in the first
  sweep failed, but failed in a pattern that named the answer (every even
  channel exactly zero, every odd one scaled by the fp16 reading of the
  preceding `f32`'s upper half, which is what a 4-byte-pitch plane read
  two bytes at a time looks like). That it is a float and not a
  fixed-point value with a 6-bit shift matters for §4: a row reciprocal
  `1/d` for `d` anywhere in `[1, N]` is nothing as fp16, so the operand's
  range stops being a constraint on the design.
* **It broadcasts**, at `16x1`, `4x4` and `1x1` -- stated as "the same
  addend at every pixel" rather than as an oracle match, because that is
  the claim the max-subtraction rests on.
* **It indexes every programmed channel**, checked at 32 channels against
  the two layouts that would otherwise pass an 8-channel test (repeating
  per fp16 atom, or per 16-channel programmed granule).
* **The stage order is `x * m + a`.** The addend is not scaled by the
  multiplier -- which independently corroborates `activation.rs`'s
  `lut_bn_alu`, whose formula pre-scales the register-sourced ALU operand
  by `bn_mul_operand` for exactly that reason.
* **C8 does not reproduce.** A plain `build_unary_regcmd` job submitted
  straight after an NRDMA fetch, with nothing in between, is still
  correct. That was worth checking rather than assuming: leaving a DMA
  operand fetch enabled with no consumer is what poisoned the core and
  hung the *next* job until the power domain cycled, and it was BRDMA,
  this DMA's sibling ([`conv.rs:1353`][conv], ISSUES C8). `bn_plane.rs`
  derives `nrdma_data_use` from the same enum that engages the BN
  sub-stages, so a fetch and its consumer cannot disagree by construction.

Two things are worth keeping from how it went. The builder's own bypass
words came out as `BS_CFG=0x53` and `EW_CFG=0x383` -- the exact pair on
every plain pass-through DPU copy in the vendor's compiled softmax
(`rknpu-spelunking/NOTES.md` Follow-up 5), which a unit test now asserts;
so this task shape was capture-backed before it ever ran. And the vendor
does not use this plane in its own softmax (`BN_BASE_ADDR` was captured as
0 on its one `exp` dispatch), which was the reason to expect trouble and
turned out not to matter -- the same shape of result as the `mulf`
investigation, where the reference emitter's claim about the EW operand
path was simply not true of this register program.

**S2 -- width reduction geometry. DONE 2026-11-24** (`reduce_width_hw.rs`,
plus a `PoolingPlan` fix and two `pooling.rs` unit tests, run on the
board). Four hardware tests, all green:

* **The `k x 1` reduction is exact at atol 0** at `N` 8, 16, 64, 128, 197,
  384 and 512 -- a max of fp16-exact values is exact, so there is nothing
  for a tolerance to absorb. 197 is the case that earns its place: ViT's
  token count, not a multiple of 8, and wide enough to need several column
  tiles per stage, so one case covers the partial window, the split and the
  tree depth. Inputs vary along both axes, per the `Counting` trap.
* **The whole tree fits in one job**, every stage and every tile in one
  ordered task array, with no CPU wait between stages and no fence on the
  intermediates. 512 columns is 3 stages and 6 tasks; 197 is 3 and 4.
* **The combine is one more tree level over one shared cube** (§5.1),
  checked by reading the intermediate rather than inferring it from a right
  answer: at `N = 384` three column tiles write disjoint slices and every
  column holds the max of its own window.
* **Multi-surface channel counts work** (16 and 32), which the 8-channel
  cases could not have shown -- eight fp16 channels are one atom and one
  surface, so they would pass with the surface stride ignored entirely.

Two findings that change the design rather than confirm it.

**`PoolingPlan::new` could not plan a strided reduction at all.** It took
its tile count from the output-width cap alone, so an 8x1 window at stride
8 over 384 columns balanced to one 48-column tile that then read 384 input
columns against the 129 limit -- and tripped the function's own `assert!`.
Every extent above 128 panicked at plan time. §5.1 has the fix and why it
leaves the capture-backed splits alone.

**An fp16 average pool cannot reduce along one axis only**, which removes
§2.2's third route to a row sum. `average_reciprocal` is asked for the
*height* axis's identity divisor `65536 / 1`, past fp16's 65504 ceiling, so
every height-1 average shape refuses before it reaches the device. int8
programs 65536 plainly, so it is a precision restriction, not a geometry
one. Both halves are pinned by unit tests that need no board
(`an_fp16_average_cannot_reduce_width_only`,
`an_int8_average_can_reduce_width_only`). The sum now has exactly two
routes, both on the DPU.

**The sum route is gated too**, in the same file: a depthwise all-ones
window reduces a row **exactly, at atol 0**, at `N` of 9, 27, 64, 197, 384
and 512, as a tree in one job. Every input is a multiple of 0.25 in
`[0, 1]` -- the range `exp(x - m)` actually produces -- so every partial
sum is a multiple of 0.25 and exact in fp16 well past the total, which is
what lets this be gated exactly rather than to a tolerance that could hide
a dropped or double-counted column. It also checks the fp16 vertical pad
fill as a side effect: a nonzero one would add three taps to every window.
The window had to be 3x3 at stride 3 and the geometry needs a per-stage
leading pad; §2.2 has both and why.

Whole-row cost at `N = 384`, both halves: the max is 3 stages and 6 tasks,
the sum is 6 stages and 6 tasks. The sum's extra depth is the price of a
3-wide window where the PPU could take 8.

**S2a -- the plane bridge. DONE 2026-11-24**
(`bn_plane::build_plane_gather_regcmd`, three more tests in
`bn_plane_probe_hw.rs`). One DPU pass-through turns the PPU's
four-pixel-strided width-1 cube into a dense plane operand, and the two
halves were tested separately so a failure would localise:

* **`RDMA_SURF_NOTCH` reads the strided cube**, first try, at 32 channels
  -- four source surfaces, with a `-99.0` sentinel in the slack pixels so
  an ignored notch would be obvious rather than merely wrong.
* **The fp32 writer took a sweep.** Reading
  `Precision::Fp16Accumulator`'s recipe off `conv.rs` gave `size_e = 3`
  (its "float rule, size_e = output bytes - 1"), and that **hung at the
  watchdog**, 522 ms, twice. Sweeping `size_e` alone then showed 0 and 1
  running and writing channel 0 *correctly* as `f32` at `channel * 4` with
  28 of 32 channels wrong -- exactly one atom's worth right, which pointed
  at the surface stride rather than the element type.

  `conv.rs` says why one field at a time reads as "nothing helps": the
  4-byte writer's `(mc_surf_out, size_e, surf_add)` are one configuration
  and "move together or not at all". Sweeping the triple one combination
  per process -- a hung job can leave the core sick -- found
  **`(0, 1, dst_surf_stride * 2)`**, and the neighbours confirm it is an
  optimum rather than luck: `surf_add` at twice that writes every other
  surface, at the same value with `size_e = 0` it stops after the first,
  and `size_e` 3 or 7 hangs. The 2 is the multiplier `conv.rs`'s int4 note
  predicted when it called 8 the integer quirk, "not the float path's
  natural 1 and 2x". `the_fp32_gather_programs_the_measured_writer_triple`
  pins all of it without a board.
* **The edge works end to end**: a gathered `f32` plane feeds the BN ALU
  stage in one job, with no host write between the two tasks. That is §7's
  pass-2 edge, and it is the claim S2a exists to support rather than either
  half on its own.

**S3 -- `exp` over the real domain. DONE 2026-11-24, and it is the one
probe that came back negative** (`activation::build_lut_fp16_regcmd`,
`lut_exp_fp16_hw.rs`). This step was written as "a test, not new code",
which was wrong twice over.

The first surprise cost nothing: the LUT path in this crate is **int8**, on
both the DPU and the RDMA side, and none of its six tables has ever run
otherwise. So the probe had to build the fp16 variant before it could
measure anything. That variant works as far as the table: the job completes
in milliseconds and `exp(0)` is exactly 1.0.

The second is the finding. **`OUT_CVT` is an integer multiply-shift**, so
the table's `real_value * 32768` entries cannot be divided back down inside
the task: at `(1, 15)` the entry for `exp(0)` survives as 1.0 while the LE
edge entry, 221, becomes exactly 0. The `exp` step's output is a Q15
integer, not an fp16 probability. §3.1 has the consequence and the
direction out of it -- softmax is a ratio, so a constant 32768 cancels --
and §9 carries it as the design's open question.

A separate mis-indexing was measured and deliberately not diagnosed: the
index reaches the table edge four orders of magnitude earlier than the
multiplier predicts. Both would have to be solved to keep `exp` in fp16;
only the converter has to be solved to move to Q15, so that is where the
effort belongs.

What S3 did confirm from §3 is the *shape* of the domain argument, and it
corrected its conclusion: the clamp below the edge is the table's edge
entry, `exp(-5)`, not a decay to zero and not the Q15 floor, so buying
domain with `input_scale` buys real accuracy. §3.2 has the corrected
numbers.

**S4 -- the reciprocal. DONE 2026-11-24, and it inverts §4's own ranking**
(`reciprocal_route_hw.rs`). The reciprocal runs on the host, for three
measured reasons: `d` spans 1.03 to 122.8 on real ViT-B/16 rows with 44% at
or past `LutTable::reciprocal`'s `[1, 16)` edge; the blocker is the int8
*input*'s 128 linear codes against a 200:1 range, which a wider table does
not fix and a Newton step cannot recover from; and the host costs 0.006 ms
for 64 rows plus one 0.037-0.059 ms submission boundary, against ISSUES
P8's 1.6 to 7.4 ms flat per-dispatch cost. §4 has the detail and keeps the
four ranked NPU routes, all of which assumed an input the LUT does not
have.

It also measured §7's passes 2 and 3 end to end on the real rows --
**1.12% mean total variation**, the same as §3.4's `exp` alone, so the fp16
plane multiply costs nothing measurable -- and turned up two things on the
way.

**The two-byte writer's surface multiplier is 2, not 1.** At 64 channels
the Q15 `int16` output wrote only 40 of them with
`surf_add = surface_stride`, and all 64 with `* 2`
(`the_q15_writer_channel_map`). That is the same value the fp32 plane
gather needed (S2a), and the rule that fits every builder in the crate is
the narrowing bit: `surf_add` is `surface_stride` when `fp32tofp16_en` is
*set*, and twice it when the writer emits the wider datapath element.
`conv.rs`'s int4 note predicted the pair -- "the float path's natural 1 and
2x". Every earlier Q15 run used one channel, where an output surface stride
cannot be observed at all, which is why S3 did not catch it.

**The plane will not take `1/d_q15`.** It is `1.5e-7` to `3e-5`, below
fp16's normal range, and `write_bn_mul_plane_fp16` refuses it rather than
flushing a channel's multiplier to zero. The `1/32768` rides the
Q15-to-fp16 conversion instead, exactly, since it is a power of two.

**S4a -- the Q15-to-fp16 bridge. DONE 2026-11-24**
(`bn_plane::build_q15_bridge_regcmd`, four more tests in
`reciprocal_route_hw.rs`). The last host step in §7 moves onto the device
as two tasks. §4.2 has the finding that makes it two -- the DPU has an
integer datapath for an integer input and a float one for an fp16 input,
and only the BN plane multiplies by a true fraction -- plus the two
fp16-reading bugs it turned up, one of them pre-existing in fifteen test
files.

**S5 -- the op. DONE 2026-11-24** (`rocket_core::softmax`,
`iree_rocket_hal::rocket::softmax`, `softmax_oracle_hw.rs`).

**The assembled op is within 1.12% total variation of an exact softmax on
64 real ViT-B/16 attention rows** -- the same figure as §3.4's `exp` step
alone and §4's passes 2 and 3, so the device's own maximum, the int8 score
quantisation and the two reduction trees add nothing measurable on top.
Worst row 13.97%, which is §9's spread item. 13 device tasks across three
submissions, 2 host steps, 92160 bytes of scratch.

`rocket_core::softmax::Plan::try_new` is §7 as a plan: the ordered stages,
both reduction trees with their per-level extents and padding, and the
costs §4 and §5 argue about reported before anything runs. Column tiling
stays with the planners that own it, `PoolingPlan` for a max level and
`ConvPlan` for a sum level, as `fc` delegates to `conv`. Its refusals mark
where the *measurement* stops rather than where the hardware does -- an
`exp_input_scale` other than §3.4's `1/20`, more than 64 rows, more than
512 keys -- and all three come back `UnvalidatedConfiguration`.

`iree_rocket_hal::rocket::softmax` emits them and owns the two things a
plan cannot: the scratch layout and the submission boundaries.

Three things the assembly taught that no single probe could.

**The physical width has to be a multiple of four, and that is a
correctness rule rather than padding for tidiness.** The score buffer is
read by the PPU, whose cube rounds each surface stride up to four pixels,
*and* by the `exp` task's DPU_RDMA, whose does not. At 197 keys those
strides are 200 and 197 atoms, so every channel past the first atom lands
somewhere different for the two readers. `Shape::physical_keys` rounds up
and the trailing columns carry `PAD_CODE` -- the most negative int8, which
never wins a maximum and whose `exp` is exactly zero under §3.3's tuned
tail, so it changes neither reduction. The failure is quiet without it: a
row whose maximum sits in the first atom still comes out right, which is
how the first run's maxima assertion passed while everything downstream was
zero.

**A BO listed in a job's `in_bo_handles` *and* its `out_bo_handles` is not
written.** This cost the longest bisection of the whole design. The `exp`
task returned all zeros while the identical program -- byte-identical,
asserted against a direct build -- wrote correct values elsewhere, and what
differed was that the harness passed every buffer as an input, the output
among them. Each step of the bisection ruled out something likelier: the
input codes read back correctly, the task is exact at this width and row
count standalone and at every width from 197 to 256, a preceding PPU job
does not disturb it, one task per job behaves like one grouped job, and an
output offset of 15360 into a shared allocation is fine. Nothing in this
repo documented the rule; `submit_tasks`'s doc comment says a buffer only
tasks within the job touch "does not need to appear in either list", which
is not the same as saying both is harmful. It is worth a test of its own in
`device.rs`.

**And one of my own tools lied twice.** `ls -t` over the cargo target
directory picked a stale cross-compiled binary, so two board runs reported
results from code that no longer existed -- including one that appeared to
clear a hypothesis. Take the path from `cargo --message-format=json`.

Only after S5: the schema def, the driver's dispatch, the transform-spec
matcher. Those follow COMPILER_ROADMAP's existing grooves and are not
designed here. Note that `linalg.softmax` decomposes by default, so the
matcher has a choice to make that ROADMAP.md's entry already names -- match
the single op before decomposition, or match the decomposed chain. Match the
single op: the decomposition's `linalg.reduce` over a channel-mapped axis is
precisely the form this hardware cannot run.

---

## 9. What would falsify this

* ~~**Nothing here is a performance measurement.**~~ Measured now, and
  **it is about 4.4x slower than the CPU** -- §10.

* ~~**S1 comes back dead or fixed-point-only.**~~ Answered 2026-11-24: the
  plane carries a per-channel float on both sub-stages. The fallback this
  bullet named -- an ERDMA broadcast operand, or a materialised broadcast
  tensor and the full-size pass it costs -- is not needed.
* **The combined ALU-plus-MUL plane layout is open**, the one thing S1 and
  S2a did not settle. Each stream's own stride is measured, but with both engaged
  the MUL stream is fetched two bytes further on than `Mul` alone reads it,
  so they are not simply both based at `BN_BASE_ADDR`. §7 engages one
  sub-stage per task and does not need it; a fused op that wanted
  `relu(a * x + b)` in one pass would, and should start from
  `bn_plane_add_then_mul_fit`'s dump rather than a fresh sweep.
* ~~**The Q15 `exp` floor is not where §3 computes it.**~~ Settled by S3,
  against the design: the floor is not the binding constraint at all, and
  the real one is worse. `OUT_CVT` is an integer multiply-shift, so the
  `exp` step cannot emit a fractional value -- its output is a Q15 integer.
* ~~**Whether §7 can run on Q15 probabilities.**~~ Closed on the output
  side by §3.4: `build_lut_q15_regcmd` writes the Q15 integer as `int16`
  with the converter at identity, it worked first try, and the end-to-end
  softmax measured through it is within 1.12% of exact. The 32768 never
  has to be divided out, because the sum and the division are a ratio.
  What is left of this question is only the final divide: S1 measured the
  BN MUL plane's operand as fp16, so `1/d` has to become an fp16 number
  somewhere, which is §4's problem rather than this one.
* ~~**The `exp` step's accuracy.**~~ Measured: 1.12% mean total variation
  against exact softmax on real ViT-B/16 attention rows, at
  `input_scale = 1/20` with the tuned slope (§3.4). The table does not need
  regenerating.
* ~~**The Q15-to-fp16 bridge.**~~ Closed by S4a: `build_q15_bridge_regcmd`,
  two device tasks rather than the one it looked like. §4.2 has it.
* **The worst row is 14%, not 1%.** The mean is good and the spread is not,
  on the same 64 real rows. A model-level gate should report both, and if
  the bad rows turn out to be the ones that matter -- a near-uniform
  attention row is exactly where a denominator error moves every weight --
  then the answer is a finer `input_scale` for that layer rather than a
  different table.
* **The fused-attention win does not survive measurement.** C15 measured
  unbatched attention at 1.16x (ViT-B) to 1.41x (BERT-384) *slower*, and the
  CPU spends little on attention. This design removes softmax and the
  denominator from that accounting but adds runtime weight packing for `Q^T`
  and `P^T`, which C15 did not pay. The honest prediction is that standalone
  softmax lands behind a flag like Phase 1 and 2's matchers, and that the
  fused form is the only one that could beat the CPU. Measure the
  `--no-offload` arm alongside, on a full machine, at 600 MHz
  and against `rocket-compiler --no-offload`
  rather than a stock NHWC CPU build (ISSUES.md's P8 and the baseline
  correction recorded with it).

[fc]: iree-rocket-hal/src/rocket/fc.rs
[act]: iree-rocket-hal/src/rocket/activation.rs
[tl]: iree-rocket-hal/src/rocket/tensor_layout.rs
[dpu]: iree-rocket-hal/src/rocket/builders/dpu.rs
[conv]: iree-rocket-hal/src/rocket/conv.rs
[bn]: iree-rocket-hal/src/rocket/bn_plane.rs

---

## 10. What it costs, measured

The accuracy question is settled and the performance one is not close.
`performance` governor, A76 at 2.4 GHz, `rocket_npu_clk_hz` at 600 MHz,
pinned to the big cluster, 20 iterations after 3 warm-ups per sample,
buffers and programs built once outside the loop, five samples
(`softmax_wall_time_against_the_cpu`).

| | ms per softmax, `[64 rows, 197 keys]` |
|---|---|
| **NPU, the whole op** | **0.46 - 0.51**, median 0.49 |
| &nbsp;&nbsp;max tree, 3 tasks | 0.134 |
| &nbsp;&nbsp;host: maximum into the plane | 0.003 |
| &nbsp;&nbsp;`exp` + 2 bridge + sum tree, 8 tasks | 0.258 |
| &nbsp;&nbsp;host: `1/d` | 0.006 |
| &nbsp;&nbsp;normalise, 1 task | 0.061 |
| **CPU, plain scalar Rust, same shape** | **0.108 - 0.111** |

The NPU arm varies about 10% run to run; the CPU arm is stable to 3%. The
breakdown is one representative sample rather than a median of each row.

**The op is 4.1 to 4.7x the CPU's time**, median **4.4x** -- and the CPU
arm is a scalar Rust loop rather than IREE's codegen, which ISSUES P10
profiled and which gets a further 1.39x from a wider vector tile. So the
real gap is larger than 4.4x, not smaller.

**The cost is the task count, not the arithmetic.** 0.49 ms over 13 tasks
is **35 to 39 microseconds a task**, and S4 measured a bare
submit-and-wait on the smallest possible job at 0.037 to 0.059 ms. So each
task costs about what an *empty* one does: the op spends its time on
completion IRQs, and §5.3's note that "every task boundary costs a
completion IRQ" turns out to be the whole story rather than a caveat. The
two host steps, which §4 argued about at length, come to **0.009 ms of
0.49** -- under 2%, and S4's conclusion was right for a reason that barely
matters.

### 10.1 Which levers are left

Not accuracy, and not the host. Task count:

* **Rows per task.** 64 rows is not a hardware bound, it is the widest
  per-channel plane any probe has driven (§8's S5 refusal). A ViT head is
  197 rows, so the same three submissions could carry 3x the work for
  roughly the same 13 IRQs. This is the biggest single lever and it is a
  measurement, not a mechanism.
* **A wider sum window.** The sum tree is 5 of the 13 tasks because the
  planner backs only a 3x3 stride-3 square window (§2.2). An 8x8 at stride
  8 would reduce a row in two levels instead of five -- three tasks, about
  0.11 ms --
  and `only_an_odd_square_window_can_stride` pins the refusal so that a
  capture which widened it would show up here.
* **Fusion into attention.** §1.1's whole argument: inside an attention
  block the comparison is not against a CPU softmax in isolation but
  against the CPU doing softmax *plus* the round trips an NPU-resident
  block avoids. ISSUES C15 measured unbatched attention offload as a loss
  (1.16x to 1.41x slower) before softmax was available at all, so this is
  a reason to re-measure C15 rather than a reason to expect a win.

### 10.2 What this means for the op

It stays **off by default**, like ROADMAP.md's Phase 1 and 2 matchers and
for the same reason -- P8's law, which says an operation doing less work
than its own dispatch costs belongs behind a flag until something moves the
dispatch cost. Softmax is a very pure case of it: 12,608 exponentials is
about 25 KB of traffic and 13 IRQs.

That is not an argument against the work. The decomposition is correct,
exact where it can be and 1.12% where it cannot, and every mechanism in it
is measured and reusable -- the operand planes, the Q15 bridge, the int16
read, the tuned `exp` tail and the width-mapped reduction are all general.
It is an argument about *where* a softmax should run today, and the answer
is the CPU until the task count comes down.
