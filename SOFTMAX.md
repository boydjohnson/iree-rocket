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
| row max | PPU max pool, kernel `k x 1`, stride `k x 1` | **exists**, hardware-exact; `Max=1` encoding confirmed, `pooling_oracle_hw.rs` |
| max over tiles | `EwBinaryOp::Max` on two width-1 cubes | **exists**; oracle-checked on `planck` by `ew_binary_hw.rs:346` and again across ViT's shapes at `:421`. The `EwBinaryOp::Max` doc comment still says "not measured here yet" -- that comment is stale and worth deleting. Needed only for §6's fused form; §5.1's combine does not use it |
| subtract the max | BN ALU operand from NRDMA, per channel | **exists**, as of probe S1: flat IEEE `f32` at `channel * 4`, broadcast over the cube, `bn_plane_probe_hw.rs` |
| `exp` | DPU LUT, `LutTable::exp()` | **exists**, `lut_exp_hw.rs` against an oracle; domain needs §3's `input_scale` choice |
| square (range extension) | `EwBinaryOp::Mul` with both operands the same buffer | **exists**, `ew_binary_hw.rs`, `ew_square_hw.rs` |
| row sum | all-ones column on the second matmul (§1.1); or depthwise all-ones `1 x k` conv; or PPU avg pool x k | **exists** three ways |
| reciprocal | `LutTable::reciprocal()` + one Newton step | **exists**, `lut_reciprocal_hw.rs`; domain needs §4 |
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
* **sum via avg**: the PPU divides by `kh * kw` whether or not a tap was
  padding (count-include-pad) and the avg fill is zero, so `avg * kh * kw`
  is the sum of the real taps exactly [verified, same doc comment].

So a two-level tree over `N = 197` at `k = 8, 8, 4` recovers the exact sum as
`avg_final * 256`, with every per-level divisor a power of two (exact in
fp16). The scale factor folds into the reciprocal or into the BN multiplier;
it never needs its own pass.

What is *not* free: a plain **sum** pool is not encodable at fp16. The
`RECIP_KERNEL` field holds the fp16 bit pattern of `65536 / k`
[verified, `pooling.rs:636`, and the bug this fixed], so a multiplier of 1.0
would need `65536`, past fp16's 65504 ceiling -- which is the same reason
`average_reciprocal` panics at `k = 1`. Do not add a `PoolingMethod::Sum`;
carry the factor.

### 2.2 Prefer the DPU for the sum

The PPU accumulates an average in its own internal precision over up to 64
taps per level. For a sum of `N` terms each in `(0, 1]` that is good enough
for fp16 but it is not free of error, and it has no oracle here. Two better
routes, both already validated paths in this repo:

* the all-ones column of §1.1 (fused attention): the DPU accumulates, which
  is fp32 for fp16 inputs (LIMITS.md's precision table);
* a **depthwise** convolution with an all-ones `1 x k` kernel at stride `k`
  (standalone softmax): depthwise is exact to `Cin` 1536 at fp16 and int8
  (LIMITS.md), strides are correct since the
  demote-pass fix (ISSUES.md), and the kernel is a
  compile-time constant so it packs once.

Use the PPU max pool for the max, where there is no alternative, and the DPU
for the sum, where there is.

---

## 3. `exp`: the domain is already a free parameter

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

`input_scale = 1/2` plus one squaring is the right choice, and the reason it
is *exactly* right is that it also fixes the output side. A table entry is
Q15 (`32767` = 1.0), so the smallest nonzero `exp` the table can return is
`3.05e-5` -- the useful input range is bounded by the output encoding at
about `x = -10.4`, not by the input domain. Squaring `exp(x/2)` pushes the
representable floor to `exp(-20.8) = 9e-10` and the clamp edge to `-20`,
which reach the same place. Going to `input_scale = 1/4` buys domain the
output encoding cannot express.

Accuracy: entry spacing is `32 / K_EXP` in fixed-point domain units, so at
`input_scale = 1/2` the real-x step is `0.0195` and the 513-entry
(512-interval) table's interpolation error is about `5e-5` relative; squaring
doubles it to `1e-4`. fp16's own epsilon is `9.8e-4`, so the LUT is not the
limiting term.

**No table regeneration is needed.** This is a `LutShape` field and one
existing EW task. (If a later pass wants to avoid the squaring dispatch, a
self-derived `EXP_LE` over `[-11, 0]` is the move -- the `LE` half has no
encoding obstacle at all, unlike `LO`, and `square()`/`erf()`/`sqrt()`
established the methodology. It is an optimisation, not a prerequisite.)

---

## 4. Normalisation, and the reciprocal's real difficulty

In the §1.1 layout the divide by the row sum is a **per-channel multiply**,
which is `bn_mul_src = 1` with a `1/d` vector in the NRDMA plane. In fused
attention it applies to the `[d_v, M]` output; standalone it is one DPU pass
over `[N, M]` with the LUT bypassed. Either way the multiply itself is free
-- it rides a stage that is otherwise bypassed.

Producing `1/d` is the awkward part. `d` lies in `[1, N]` when the subtracted
value is the true max, so its dynamic range is `N`-wide and no static
prescaling brings it into a fixed window. `LutTable::reciprocal()` holds real
data only for `|x|` in `[1, 16)` [verified, `activation.rs:291`], which
covers `N <= 16` and nothing a transformer uses.

Ranked options:

1. **`rsqrt` then square then one Newton step.** `rsqrt` is real on
   `[1, 16)` too, but `1/sqrt(d)` over `d` in `[1, 384]` has an output range
   of only 20:1 against the reciprocal's 384:1, so a self-derived `RSQRT_LE`
   widened to `[1, 512]` keeps about `6e-4` relative accuracy where a
   widened reciprocal table would give `1.2%`. Square it (`1.2e-3`), then one
   Newton step `y1 = y0 * (2 - d * y0)` squares the error to `1.4e-6`. Three
   EW tasks on a length-`M` vector -- tens of microseconds at most. Uses only
   existing validated primitives plus one self-derived table.
2. **One Newton step on a widened reciprocal table.** Simpler (one new table,
   no `rsqrt`), lands at about `1.4e-4` from a `1.2%` seed. Probably good
   enough at fp16 and the first thing to try.
3. **Log-domain normalisation**: `softmax(x) = exp(x - m - ln d)`, folding the
   denominator into the same BN ALU operand as the max. Elegant -- it removes
   the multiply entirely -- but needs a self-derived `log` table whose output
   exceeds Q15 for `d > e`, so it needs the `out_cvt` scalar scale to carry a
   factor, and it makes the exp pass depend on two serial reductions. Note
   it, do not build it first.
4. **CPU.** `M` floats across a dispatch boundary. It is the fallback the
   vendor may well take: the reciprocal step is **not present as any on-chip
   dispatch** in a complete, self-consistent 36864-byte capture of the
   vendor's own compiled softmax -- every `BN_CFG`/`BN_ALU_CFG`/`BN_MUL_CFG`/
   `LUT_CFG` write in the whole buffer is the single confirmed `exp`
   dispatch, and all five PPU dispatches are max-only with
   `RECIP_KERNEL = 0` [verified, `rknpu-spelunking/NOTES.md` Follow-up 5, a
   strong negative result rather than a gap in searching].

That last point is the one place this design knowingly departs from the
vendor. The repo's house rule for exactly this situation is on the record:
diff against the reference emitter, do not obey it -- `rocket-userspace`
asserts that the EW unit cannot read a second operand without a conv main
feed, built its entire elementwise runtime around manufacturing one, and is
simply wrong about this register program (ROADMAP's `mulf` section). The BN
plane is the same shape of claim: documented in the TRM, absent from the
vendor's softmax, and cheap to test.

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

For `N = 384` keys the PPU's 129-column direct limit gives 3 tiles, so the
combine is one task at `kernel 3 x 1`. Tiles stay at or below 8 until
`N > 1032`, past which the combine is itself two levels. The sequential
execution inside the job is exactly the ordering the tree needs.

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
pass 2   DPU LUT task, flying mode:
           BN ALU operand <- NRDMA, -m per channel
           input_scale = 1/2, LutTable::exp()
           -> u = exp((x - m)/2)               [N, M] -> [N, M]
         EwBinaryOp::Mul(u, u) -> P            [N, M] -> [N, M]
         depthwise all-ones 1 x k conv tree    [N, M] -> [1, M]        d
pass 3   reciprocal of d (§4)                  [1, M] -> [1, M]
         DPU task, BN MUL operand <- NRDMA     [N, M] -> [N, M]        y
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

The plane wants a dense vector and the reduction produces a cube, but at
width 1 and height 1 those are the same thing: each NC1HWC2 surface is
exactly one 16-byte atom and surfaces are consecutive, so channel `c` lands
at byte `c * element_bytes`. A width-1 height-1 fp32 cube *is* a dense fp32
vector. What still has to be confirmed is the element-type widen on the way
there -- the max tree writes fp16 and the ALU plane wants fp32 (the MUL
plane does not, which is the easier direction) -- so S2 should check the
widen's output placement rather than assume it.

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

**S2 -- width reduction geometry.** `reduce_width_hw.rs`. Max pool at
`kernel 8 x 1`, `stride 8 x 1`, height 1, against a non-uniform input and a
host oracle, at the channel counts and extents a transformer actually uses
(`M` 197/384/512, `N` the same), including a non-multiple extent so the
partial-window clamp-to-image path is exercised rather than reasoned about.
Then the same for the depthwise all-ones sum, and the §5.1 combine: that
`PoolingPlan`'s column tiling at `N > 129` can be pointed at disjoint width
slices of one shared cube, and that a second pooling task over that cube
agrees with a host max over the whole row. Submit the tile tasks and the
combine as **one multi-task job** -- that is the structure under test, not
just the arithmetic. `Counting`-style uniform patterns cannot test this -- use
`Dense` -- a `Counting` pattern
makes a 1x1 output constant, so it proves coverage and never addressing.

**S3 -- `exp` over the real domain.** Extend `lut_exp_hw.rs` with
`input_scale = 1/2` plus `EwBinaryOp::Mul(u, u)`, oracle-checked across
`x - m` in `[-21, 0]`, and assert the clamp edge and the Q15 floor land where
§3 says. This is a test, not new code.

**S4 -- the reciprocal.** `softmax_reciprocal_hw.rs`: widened table, Newton
step, oracle across `d` in `[1, 512]`. Pick option 2 from §4 first; fall back
to option 1 if `1.4e-4` is not enough.

**S5 -- the op.** `softmax.rs` in `iree-rocket-hal/src/rocket`, holding a
`SoftmaxShape { width, height: 1, channels, precision }` and a
`SoftmaxPlan` that returns the ordered task array of §7, plus a
`SoftmaxBuffers` naming the scratch cubes. Shape refusals go through
`PlanError` with a code, not a panic, per
COMPILER_ROADMAP.md section 1's `try_*` contract -- and the planning
itself belongs in `rocket-core` so the compiler can ask the same question
the runtime does. `softmax_oracle_hw.rs` gates the whole thing against a
host softmax at `atol = 1e-3`.

Only after S5: the schema def, the driver's dispatch, the transform-spec
matcher. Those follow COMPILER_ROADMAP's existing grooves and are not
designed here. Note that `linalg.softmax` decomposes by default, so the
matcher has a choice to make that ROADMAP.md's entry already names -- match
the single op before decomposition, or match the decomposed chain. Match the
single op: the decomposition's `linalg.reduce` over a channel-mapped axis is
precisely the form this hardware cannot run.

---

## 9. What would falsify this

* ~~**S1 comes back dead or fixed-point-only.**~~ Answered 2026-11-24: the
  plane carries a per-channel float on both sub-stages. The fallback this
  bullet named -- an ERDMA broadcast operand, or a materialised broadcast
  tensor and the full-size pass it costs -- is not needed.
* **The combined ALU-plus-MUL plane layout is open**, the one thing S1 did
  not settle. Each stream's own stride is measured, but with both engaged
  the MUL stream is fetched two bytes further on than `Mul` alone reads it,
  so they are not simply both based at `BN_BASE_ADDR`. §7 engages one
  sub-stage per task and does not need it; a fused op that wanted
  `relu(a * x + b)` in one pass would, and should start from
  `bn_plane_add_then_mul_fit`'s dump rather than a fresh sweep.
* **The Q15 `exp` floor is not where §3 computes it.** The `out_cvt` scalar
  scale may interact with the entry encoding in a way the existing tests do
  not pin down. S3 settles it.
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
