# Op coverage roadmap

Which MLIR operations this stack could execute on the NPU but does not, what it
would cost to reach each one, and the order that keeps the work honest. Written
2026-09-05 against the `linalg`, `math` and `arith` dialect definitions vendored
under
[`iree-build/iree-src/third_party/llvm-project/mlir/include/mlir/Dialect/`](iree-build/iree-src/third_party/llvm-project/mlir/include/mlir/Dialect/):
46 `math` ops, 55 `arith` ops, and `linalg`'s 23 named element-wise ops
alongside its convolution, pooling, contraction and structured sets.

Read alongside its two neighbours. [LIMITS.md](LIMITS.md) says what the stack is
*measured* to do and which layer enforces each bound; [ISSUES.md](ISSUES.md)
carries the open defects. This file is the third question -- what is *not*
reached yet -- and it defers to both: a phase here that contradicts a measurement
there is wrong, and two of them below are gated on open issues by name.

Nothing here is a throughput proposal. The distinction matters more in this
document than anywhere else in the repo, because most of the work below buys
**coverage** and P8 has already measured that coverage of this particular shape
buys **negative** throughput. See the next section before reading the phases.

---

## The warning that orders everything below

[ISSUES.md P8](ISSUES.md) measured it on MobileNetV2: 17 offloaded convolutions
cost **+112 net dispatches**, because nothing fuses across a Rocket dispatch
boundary. The CPU-only build fuses conv + bias + dequant + activation into 136
`matmul_like` kernels; the NPU build has none of them, and every epilogue
becomes its own dispatch. That is what the 5079 ms of `outside` is.

P8's own ranked lever list puts *"standalone elementwise/transpose matchers"* at
**#3, explicitly "not before (1)"** -- and (1), layout propagation, was built and
**refuted**, worth 0%. Its closing sentence is the one to hold onto:

> at the current per-dispatch cost, **more offload sites make the model slower**.

So this roadmap separates two things the repo has previously conflated:

- **Coverage work** (Phases 1 and 2) makes an op *expressible and correct*. It
  is worth doing -- an op that cannot be compiled at all cannot later be fused --
  but its matchers must land behind a flag, not in the default
  `transform.foreach_match` list.
- **Throughput work** (Phase 3) *removes* a dispatch boundary rather than adding
  one, and is the only phase here with a positive performance story.

A phase that ships coverage matchers switched on by default would regress every
model measured in P8. That is the failure mode this ordering exists to prevent.

---

## Three gaps, not one

### Gap 1 -- the HAL has hardware-validated capability the wire format cannot express

`iree-rocket-hal` already builds, and has board-tested on `planck`, a set of
operations that no compiled `.vmfb` can reach:

| Capability | Builder | Hardware test |
|---|---|---|
| 9 LUT curves: sigmoid, tanh, exp, square, erf, sqrt, rsqrt, log, reciprocal | [`activation.rs`](iree-rocket-hal/src/rocket/activation.rs) `build_lut_regcmd` | `lut_zero_join_hw` (all 256 int8 codes, every kind), `lut_hw`, `lut_exp_hw`, `lut_erf_hw`, `lut_sqrt_hw`, `lut_rsqrt_hw`, `lut_log_hw`, `lut_reciprocal_hw`, `ew_square_hw` |
| EW binary add / subtract / **multiply** / max / min, standalone or chained | [`elementwise.rs`](iree-rocket-hal/src/rocket/elementwise.rs) `build_add_regcmd`, `EwBinaryOp` | `ew_binary_hw` (all five, bit-exact), `conv_with_add_hw` |
| EW unary abs / neg / floor / ceil, and add-with-scalar | `elementwise.rs` `build_unary_regcmd` | `ew_unary_hw`, `ew_round_hw` |
| conv → LUT as two tasks in one job | `activation.rs` `build_conv_then_lut_regcmd` | `conv_then_lut_hw` |
| conv → EW add as two tasks in one job | `elementwise.rs` `build_conv_then_add_regcmd` | `conv_with_add_hw` |

~~[`rocket_executable_def.fbs`](rocket-schema/schema/rocket_executable_def.fbs)
has four `KernelDef` union members -- `Conv2DDef`, `FullyConnectedDef`
(deprecated, never emitted by any compiler), `PoolingDef`, `MatmulDef`. **None
of the element-wise or LUT work above is expressible.**~~

**Closed 2026-09-06 by Phase 1.** Three union members were appended --
`ElementwiseUnaryDef` (5), `ElementwiseLutDef` (6) and
`ElementwiseBinaryDef` (7) -- with decode and dispatch arms in
`rocket-hal-driver` and serializer arms in `RocketTarget.cpp`. The two-tensor
form is reachable from a compiled model behind `--elementwise`; the other two
are expressible but no measured model contains an op that would use them (see
Phase 1's op census). Chained conv->LUT and conv->EW as one job remain
unexpressible: the wire format has no way to say "two tasks, one job".

### Gap 2 -- matchers

Every `@match_*` in
[`rocket_conv2d_transform_spec.mlir`](rocket-compiler-plugin/target/Rocket/rocket_conv2d_transform_spec.mlir)
is a convolution except `@match_rocket_matmul` and
`@match_pooling_nchw_sum_avg`.

### Gap 3 -- one capability is fully plumbed and blocked *only* by a matcher

Min and max pooling exist at every other layer: `PoolingMethod::{MAX,MIN}` in
the wire schema, the decode arm in
[`executable_cache.rs:287`](rocket-hal-driver/src/executable_cache.rs), and
`PoolingMethod::{Max,Min}` in [`pooling.rs`](iree-rocket-hal/src/rocket/pooling.rs)
with the per-precision pad-fill identities already worked out
(`pad_fill_value`: max/fp16 is `0xFC00`, min and int8-max take the
"no fill" path). The compiler simply never asked for them.

**Both were closed on 2026-09-05** -- six matchers and six shims, no change
below the compiler at all, which is the evidence that this diagnosis was
right. All three PPU reductions are now reachable from a compiled model.

---

## What the hardware can and cannot reach

Three menus, one per dialect. "Have" means a hardware-validated HAL builder
exists; the op is still unreachable from `iree-compile` until Phase 1.

### `math` -- roughly 22 of 46 reachable

The LUT block is a 513-entry interpolated fp16 table, so *any* unary fp16
function with a bounded domain and a bounded range is one table plus one
hardware oracle test.

| Class | Ops | Cost |
|---|---|---|
| Have (LUT) | `erf` `exp` `log` `sqrt` `rsqrt` `tanh` | Wire format only |
| Have (EW ALU) | `absf` `ceil` `floor`, and `round`/`roundeven` composed as `floor(x + 0.5)` | Wire format only |
| New LUT, low risk -- monotone, bounded on a restricted domain, same self-derivation as `square`/`erf` | `exp2` `log2` `log10` `log1p` `expm1` `erfc` `cbrt` `atan` `asin` `acos` `atanh` | One table + one hw test each |
| New LUT, harder -- needs range reduction, or has no output ceiling to bound the table against | `sin` `cos` `tan` `sinh` `cosh` `asinh` `acosh` | Domain design first |
| Composable | `clampf` (the existing RELUX activation, or EW max → min), `trunc` (sign-aware, needs a select), `sincos` (two tables, two results) | Case by case |
| Unblocked 2026-09-06 by `mulf` -- now composable rather than blocked | `powf` (`exp(b*log(a))`, two LUTs and a MUL) `fpowi` `ipowi` `fma` `atan2` | Composition + one hw test each |
| No hardware path -- integer, bit-manipulation, or predicate-returning | `absi` `copysign` `ctlz` `cttz` `ctpop` `isfinite` `isinf` `isnan` `isnormal` | Refuse; document |

### `arith` -- roughly 11 of 55 reachable

The honest framing: `arith` ops are not tensor ops. They arrive inside
`linalg.generic` / `linalg.map` / `linalg.elementwise` bodies. Supporting
`arith` means matching those bodies, which is a `linalg` matcher problem.

| Class | Ops | Notes |
|---|---|---|
| EW ALU, hardware-confirmed opcodes | `addf` (algo 2), `subf` (4), `negf` (6) | `2` and `4` confirmed both precisions by the 47-model conv+add sweep; `5`/`6`/`7`/`8` confirmed by `ew_unary_hw`'s CPU-oracle tests |
| EW ALU, hardware-confirmed 2026-09-06 | `maximumf`/`maxnumf` (algo 0), `minimumf`/`minnumf` (1), `mulf` (the MUL sub-unit, not the ALU) | `tests/ew_binary_hw.rs`, fp16, bit-exact over non-uniform operands |
| EW ALU, TRM-documented but **untested in this repo** | `divf` (algo 3) | Needs a hardware oracle test before anything depends on it, same bar the other opcodes cleared |
| Already handled elsewhere | `constant` → `RecordedOp::Fill`; `extf`/`truncf` → the packer's job, fold into an adjacent dispatch rather than offload | No new work |
| No hardware path | ~35 integer and predicate ops: `andi` `ori` `xori` `shli` `shrsi` `shrui` `divsi` `divui` `remsi` `remui` `cmpi` `cmpf` `select` `extui` `extsi` `sitofp` `fptosi` `index_cast` `bitcast` … | Refuse; document |

### `linalg` -- 19 of 23 named element-wise ops reachable

This is the striking result of the survey: **`linalg`'s named unary and binary
element-wise set is very nearly a one-to-one match for the HAL's existing
menu.**

- **Reachable**: `copy` `exp` `log` `abs` `ceil` `floor` `negf` `reciprocal`
  `round` `sqrt` `rsqrt` `square` `tanh` `erf` `add` `sub` `div` `max` `min`
  `mul` (unblocked 2026-09-06, see below)
- **Blocked**: `select` (needs a predicate), `div_unsigned` (integer).
  `powf` is no longer blocked, but it is a composition (`exp(b*log(a))`),
  not a single hardware op

Matchers must cover `linalg.elementwise` (the unified `ElementwiseKind` op) and
`linalg.map` alongside the named ops, because upstream canonicalization moves
freely between all three forms -- the spec already fights this for contractions,
where `linalg-specialize-generic-ops` has to be re-run after
`fold_unit_extent_dims_via_reshapes`.

Beyond element-wise, `linalg` also offers three things reachable with no HAL
work at all -- see Phase 0.

### ~~`arith.mulf` is the single biggest blocker~~ -- RESOLVED 2026-09-06

**`mulf` works. It was one register field, and the recorded failure was a
red herring.**

The claim this section used to make was that `arith.mulf` is blocked by a
known, unresolved hardware failure recorded in
[`ew_square_hw.rs`](iree-rocket-hal/tests/ew_square_hw.rs): a prior
`build_square_regcmd` used DPU MUL mode with ERDMA self-aliased to the
primary input address and produced all-zero output for every input. That
note also said the failing configuration was specifically the *self-aliased*
one and that a genuine two-tensor MUL through ERDMA's ordinary path had
never been tried.

It has now been tried, and it works. `EwBinaryOp::Mul` in
[`elementwise.rs`](iree-rocket-hal/src/rocket/elementwise.rs) selects the EW
core's MUL sub-unit instead of its ALU -- `ew_op_type=1`, `ew_alu_algo`
cleared, `ew_op_cvt_bypass=1`, everything else byte-identical to the
already-shipping add -- and
[`ew_binary_hw.rs`](iree-rocket-hal/tests/ew_binary_hw.rs) measures it on
`planck` as **bit-exact against a host oracle**, over operands that are
distinct at every element, across four cube shapes (4x4x16, 3x3x16 -- below
the builder's own `EW_SURF_STRIDE` floor of 12 -- 7x5x24, and 14x14x64, an
8-surface cube of 12544 elements). No tolerance: `==`.

The same round confirmed four other things worth recording, because each one
was carrying an "untested"/"unconfirmed" label somewhere in this repo:

| Claim | Status before | Status now |
|---|---|---|
| `EwBinaryOp::Mul` (`ew_op_type=1`) | Believed hardware-dead | Bit-exact, 4 geometries |
| `Max`/`Min` (`ew_alu_algo` 0/1) | TRM-documented, untested | Bit-exact |
| `build_add_regcmd` standalone, no producing conv | "Would reuse ... directly", never run | Bit-exact, all five ops |
| `build_add_regcmd`/`build_conv_then_add_regcmd` on silicon at all | "NOT YET RUN ON REAL HARDWARE" | Both files green on `planck` |

**Two lessons about the evidence, both worth more than the result.**

*The recorded failure never applied.* `build_square_regcmd` aliased ERDMA to
the input, so the MUL's two operands were the same buffer -- there was never
a second tensor. That is one configuration of one op; it says nothing about
MUL as such, and this section read it as a hardware limit for months.

*The independent reference stack is wrong here, and following it would have
cost the whole investigation.* `../rocket-userspace` states plainly, in
`rocket_activation.c`, `API.md` and `tests/ew_mul_rocket.c`, that this NPU's
EW unit reads its second operand *only* combined with a conv/CACC main feed
(ERDMA `EW_BASE` + MRDMA `SRC_BASE` + `COMB_USE(5)`), and that "a pure
flying-MRDMA-main + ERDMA-operand pair reads the operand as 0". Its whole
element-wise runtime is built around that belief: every binary op goes
through an **identity matmul** to manufacture a conv main feed. On the
register program in this repo that is simply not so -- `COMB_USE` is 0 here,
`SURF_NOTCH`/`EW_SURF_NOTCH` are 0, the main feed is a plain flying MRDMA
read from memory, and the operand is fetched correctly for every position of
an 8-surface cube. Taking the reference at its word would have meant
building an identity-matmul path and never testing the one-field change that
actually works. The standing advice to diff against that
emitter still holds; this is the counter-example that says diff, do not
obey.

**What this unblocks, and what it does not.** `linalg.mul`, `powf` (as
`exp(b*log(a))`, two existing LUTs and a MUL), `fma`, `atan2` and softmax
scaling are all now expressible. None of that is free throughput: an
element-wise MUL is exactly the kind of op P8's law puts on the wrong side
of the dispatch-tax line, so its matcher belongs behind a flag like the rest
of Phase 1/2, and the fusion story (Phase 3) is still where the performance
is.

**What is still open.** int8 MUL is a loud refusal in the builder, not a
measurement -- the EW/OUT `_CVT` scale semantics for a multiply are not in
any capture this repo has. `divf` (`ew_alu_algo=3`) remains the one
TRM-documented binary opcode with no hardware evidence.

---

## Phases

### Phase 0 -- free coverage: compiler only -- COMPLETE 2026-09-05

No schema, driver or HAL change. Every item was already plumbed end to end and
blocked solely by the compiler.

All three items landed and are board-validated. The diagnosis held: not one
line below the compiler changed for any of them. The one correction is item
3, which needed a new pass rather than only spec edits.

1. ~~**Max pooling.**~~ **Landed 2026-09-05.** Four matchers
   (`linalg.pooling_nhwc_max` and `pooling_nchw_max`, strides 1 and 2), two
   executables, four shims. Nothing below the compiler changed: the schema,
   `executable_cache.rs`'s decode arm and `pooling.rs` already carried
   `PoolingMethod::Max`. Padding stays baked at zero, which is what makes the
   method safe -- `pad_fill_value` has a measured identity for max only at
   fp16, and an unpadded pool never reads the field.
   **Board-validated 2026-09-05** by `tools/e2e_pooling_regression.py`: all
   six compiled max cases are **bit-exact** against the CPU (max|error| 0),
   both layouts, both strides, the 8x8 kernel ceiling, a width that forces
   `PoolingPlan` to tile, and two pools sharing a command buffer.
   ~~**Min pooling.**~~ **Landed 2026-09-05** too, and board-validated the
   same way: two matchers (NHWC only -- linalg has no `pooling_nchw_min`),
   two executables, two shims, all three compiled cases bit-exact plus a
   min-then-max command buffer. `pad_fill_value` has no measured identity for
   min at any precision, which makes it unpadded-only -- and every matched
   executable already is, with `padded` derived from the executable's own
   baked fields so tiling cannot reintroduce it.
2. ~~**NHWC average pool.**~~ **Landed 2026-09-05.** One matcher, one shim,
   no new executable -- it shares `@rocket_pooling_executable` with the NCHW
   form, since the executable takes NC1HWC2 cubes and knows nothing about the
   layout its caller started from. Board-validated at max|error| 0.0053, in
   line with the NCHW shim's 0.0057 on the same shape.
   This leaves **one asymmetry** in the pooling matchers: max and min carry
   strides 1 and 2, the average only stride 1. `avg 2x2s2` is measured
   against the oracle in `pooling_oracle_hw.rs`, so that is a missing
   executable rather than a missing measurement, and it is the cheapest
   pooling work left.
3. ~~**`linalg.matvec` / `vecmat` / `dot`.**~~ **Landed 2026-09-05**, but not
   the way this item proposed. Rather than three matchers and three shims,
   one new pass -- `rocket-expand-gemv-to-matmul` -- raises `matvec` and
   `vecmat` into `linalg.matmul` with a unit extent, and everything
   downstream claims them unchanged: the f16 demotion, `@match_rocket_matmul`
   (whose `dim_bounds` already start at `umin = 1`), `@call_rocket_matmul`
   and `#rocket_matmul_target`. It is the batch-matmul unit-dim fold the spec
   already performs, run backwards.

   That is the one place Phase 0's "compiler only, no C++" framing was wrong:
   the item is still compiler-only, but it needed a pass rather than spec
   edits, because nothing upstream raises rank.

   `linalg.dot` is deliberately excluded: it reduces two vectors to a scalar,
   so a dispatch plus a weight pack plus an output compaction would produce
   one number a CPU computes in a few hundred multiply-adds -- the same
   reasoning that keeps a 1x1 stride-1 pool off the NPU.

   Board-validated on `planck`: matvec max|error| 5.2e-04, vecmat 8.2e-04 at
   K = 768, both consistent with f16 demotion of that contraction and nothing
   else.

Each needs a boundary lit test on the existing pattern -- one accepted shape and
its immediately-adjacent rejected neighbour, as in
[`rocket_pooling_match_boundaries.mlir`](rocket-compiler-plugin/test/rocket_pooling_match_boundaries.mlir).

> **Hard constraint on every new matcher, in every phase.**
> [`rocket-compiler/src/spec.rs`](rocket-compiler/src/spec.rs)'s `neutralize`
> refuses to build a `--no-offload` spec if any matcher named in the
> `foreach_match` list constrains no dimension with
> `transform.iree.match.dim_bounds`. A matcher without one silently breaks the
> baseline arm -- which, per the NHWC-baseline correction, is the only valid
> comparison this repo has. This is deliberate: the check exists because the
> spec grows matchers over time and nothing else would notice.

**Buys**: real op coverage, no new hardware risk. **Costs**: per P8, possibly
throughput. Gate behind a flag and measure both arms.

### Phase 1 -- a wire format for what the HAL already validated -- LANDED 2026-09-06

All four layers are in and the two-tensor path is board-validated end to end.
What follows is the original plan; the corrections it needed are marked
inline.

**Landed:**

| Kernel | Wire tag | Driver | Serializer | Matcher |
|---|---|---|---|---|
| `ElementwiseUnaryDef` (abs/neg/floor/ceil/add_scalar, fp16) | 5 | yes | yes | none -- no model uses these ops |
| `ElementwiseLutDef` (9 curves, int8) | 6 | yes | yes | none -- the crate's LUT path is int8 and ViT's 61 `sqrt` are f32 |
| `ElementwiseBinaryDef` (add/sub/mul/max/min, fp16) | 7 | yes | yes | `linalg.add`/`mul`/`sub` rank 3, behind `--elementwise` |

**Two things the plan got wrong, both found by checking rather than
assuming.**

1. *"`iree-rocket-hal` -- no new work."* `build_lut_regcmd`'s
   `DPU_DST_SURF_STRIDE`/`DPU_SURFACE_ADD` were `width * height *
   task_channels`; those registers count 16-byte feature atoms, so the
   channel factor made the stride 16x too large and every cube past one
   surface came back with only its first surface written. Every LUT test in
   the crate used `channels <= 16`, which is the one channel count that
   cannot see it. Fixed, and gated by `lut_multi_surface_hw.rs`;
   `build_unary_regcmd` had the same blind spot (`channels: 1`, uniform
   fill) but not the bug, now gated by `ew_unary_multi_surface_hw.rs`.
2. *"matchers for the `linalg` named element-wise ops, plus
   `linalg.elementwise` and `linalg.map`."* An ONNX `Add` arrives as a
   `linalg.generic` with an `arith.addf` body, and
   `linalg-specialize-generic-ops` -- which `@__transform_main` runs twice
   before the match loop -- turns it into the named `linalg.add`. So the
   named op is right, but for a reason the plan did not state, and a matcher
   written against `linalg.elementwise` or against the generic form matches
   nothing silently. Established with a torch-onnx probe compiled through
   this spec.

**A third correction, to the ordering rather than the content.** The plan put
`ElementwiseBinaryDef` last. An op census says it is the only one of the three
with a target in a compiled model:

  ViT              Add 123  Mul 74  Sqrt 61  Div 49  Sub 25  Pow 25
  MobileNetV2 fp16 Conv 52  Clip 35  Add 10
  MobileNetV2 int8 QLinearConv 47  Clip 35  Add 10

`abs`/`neg`/`floor`/`ceil` appear in none of them. The unary and LUT wire
formats are in and tested, but nothing compiles to them yet, and the honest
next step for the LUT half is an fp16 LUT in the HAL (see Phase 2).

**Measured on ViT, 2026-09-06.** Both correctness and throughput, on
`planck`, governor `performance` on both A76 clusters, NPU IRQs on cpu6,
`iree-benchmark-module --benchmark_repetitions=7`, every arm built by
`rocket-compiler` so the baseline is like-for-like ([ISSUES.md M4](ISSUES.md)).

Correct: against the `--no-offload` arm the 184-site build is `max|err|`
**0.0060** on logits with a standard deviation of 0.90, same predicted class.
The 12-site matmul arm is 0.0015, so the extra error is the f16 round trip on
172 more sites and nothing else.

Slower, and by how much depends on the core allocation, which is why it is a
column:

| cpus | `--no-offload` | 12 matmul sites | 184 sites (`--elementwise`) |
|---|---|---|---|
| 0-7 | 3973 ms | **3709 ms** (1.07x faster) | 4655 ms (**1.17x slower**) |
| 4-7 | 8598 ms | 7933 ms (1.08x faster) | 8857 ms (1.03x slower) |
| 4,5 | 8778 ms | 7999 ms (1.10x faster) | 8848 ms (1.01x slower) |

**`ROCKET_PROFILE` says why, and it is not the NPU.** At the full machine the
NPU is **4.2%** of wall (206 ms of 4901). The +1209 ms the element-wise sites
add splits almost evenly between two host costs:

- **+554 ms inside the driver**, dominated by the NC1HWC2 round trip:
  `compact` 243 ms, `pack.input` 147 ms (356 calls for 184 dispatches -- a
  two-tensor op packs both operands), `record` 129 ms.
- **+535 ms in `outside`**, which is the `truncf`/`extf` CPU dispatches the
  shims add. ViT's CPU dispatch sites go 260 -> 553.

Per site, `ew Add 1x197x3072` costs 9.07 ms of which the hardware is 2.18 ms;
the other 6.9 ms is host layout work. That is
[ISSUES.md P2](ISSUES.md)'s per-dispatch repack, paid at every link because
nothing propagates a packed layout between dispatches. **No cut point in the
shape distribution rescues it** -- even the widest op ViT has is net-negative
-- so this is not a bounds-tuning problem.

**Conclusion: `--elementwise` stays off by default.** It is P8's law confirmed
on a second model and a second op family, with the mechanism named. The lever
that would change the answer is layout propagation (P2), not a wider matcher.

One incidental finding: 25 of the matched sites are `1x197x1` -- a single
channel padded to a 16-channel atom. They cost only 7.9 ms in total, but a
channel floor would skip them for free.

---

### Phase 1 -- the original plan

**[`rocket-schema`](rocket-schema/schema/rocket_executable_def.fbs)**

- New enums `EwOp` (`Add` `Sub` `Div` `Max` `Min` `Abs` `Neg` `Floor` `Ceil`
  `AddScalar`) and `LutFn` (`Sigmoid` `Tanh` `Exp` `Square` `Erf` `Sqrt` `Rsqrt`
  `Log` `Reciprocal`). Wire values must stay independent of the `ew_alu_algo`
  register encoding, exactly as `Precision` and `PoolingMethod` already are.
  There is a concrete reason beyond convention: the conv+add sweep found real
  `rknn-toolkit2` compiles route int8 subtraction as `algo = 2` (Add) with a
  *negated scale* rather than `algo = 4`, so the register opcode is genuinely
  not a function of the logical operation alone.
- New tables `ElementwiseUnaryDef`, `ElementwiseBinaryDef`, `LutDef`: width,
  height, channels, precision, `runtime_dimensions` on the established
  contract, and for `AddScalar` the raw `f32::to_bits` operand.
- Append to the `KernelDef` union **only at the end**. Union tags are wire ABI,
  which is exactly why the deprecated `FullyConnectedDef` still holds its slot.

**[`rocket-hal-driver`](rocket-hal-driver/)**

- `UkernelShape::{ElementwiseUnary, ElementwiseBinary, Lut}` variants in
  [`executable.rs:510`](rocket-hal-driver/src/executable.rs) -- the enum's own
  doc comment already anticipates this ("Extend this enum … as more of
  iree-rocket-hal's `build_*_regcmd`/`Plan` types gain HAL-level wiring").
- Decode arms in
  [`executable_cache.rs`](rocket-hal-driver/src/executable_cache.rs) beside the
  existing four.
- Dispatch arms in
  [`command_buffer.rs`](rocket-hal-driver/src/command_buffer.rs) beside
  `UkernelShape::Conv2d` (1352), `Matmul` (1733), `Pooling` (2022).
- **The real work is `InputPacking` / `OutputCompaction`, not the plumbing.**
  These ops are NC1HWC2 in and out with identical geometry -- no reduction --
  so they pay the same host-side round trip a convolution does. Note
  `output_compaction` is currently `None` for pooling, with the atomic-slot
  mismatch flagged in its own doc comment as an unfixed follow-up risk; verify
  what pooling actually does before assuming either answer.
- [`profile.rs`](rocket-hal-driver/src/profile.rs): the phase set already
  covers these (they pack, submit, wait and compact like anything else), but
  the op labels need extending so a mixed model's per-op table stays readable.

**[`iree-rocket-hal`](iree-rocket-hal/)** -- ~~no new work~~. This was wrong;
see correction 1 above. `build_lut_regcmd` had a real surface-stride bug that
only a multi-surface cube could see, and both unary builders needed a
multi-surface gate before anything could rely on them.

**[`rocket-compiler-plugin`](rocket-compiler-plugin/)** -- matchers for the
`linalg` named element-wise ops, ~~plus `linalg.elementwise` and
`linalg.map`~~, behind a flag. See correction 2: the named ops are what
arrives, but only because `linalg-specialize-generic-ops` runs first, and
`linalg.elementwise` is not produced by this pipeline at all.

There was no "Phase 0 flag" to put them behind -- Phase 0 introduced none --
so `--elementwise` is it. The entries ship commented out in the spec with a
`//@ROCKET_ELEMENTWISE@` marker that `spec::enable_elementwise` uncomments,
which keeps the conservative list in force for anything that reads the spec
without going through `rocket-compiler`.

> **Precision.** EW unary is fp16-only by design and must stay that way.
> `EwUnaryShape`'s doc comment states the standard: there is no capture
> confirming an int8 zero-point/scale recipe for this task shape, and this crate
> does not ship an int8 branch with zero hardware evidence behind it.

> **~~Gated on ISSUES.md C5~~ -- cleared 2026-09-06.** C5 asked for *"a dense
> sweep near 0 for every signed-output kind, and drive the tails at least once,
> **before relying on the LUT path in a compiled model**."* That is
> [`lut_zero_join_hw.rs`](iree-rocket-hal/tests/lut_zero_join_hw.rs), board-run
> on `planck`: neither QUIRK 2 (the `+128` mux spike at `x~0`) nor QUIRK 4 (a
> `q = 0` entry decoding to `~4.0`) reaches this crate's int8-output LUT path.
> All 256 int8 codes now gate every kind, 9 of 10 at ≤ 1 LSB. Phase 1 may rely
> on the LUT path. See ISSUES.md **Resolved**, and note the one real defect the
> sweep found: `LutTable::log` is accurate only on `[1/e, e)`, not the
> `[0.02, e)` its doc comment used to claim -- below `x = 0.375` it returns a
> silent `-1.0` clamp.

### Phase 2 -- new LUT tables

Purely additive to `iree-rocket-hal`. Follow the `LutTable::square()` /
`LutTable::erf()` methodology exactly: self-derived table, an explicitly
documented domain restriction, one `*_hw.rs` oracle test run on `planck`.

- **Tier 1**: `exp2` `log2` `log10` `log1p` `expm1` `erfc` `cbrt` `atan`
- **Tier 2**, domain design first: `asin` `acos` `atanh`, then the trig and
  inverse-hyperbolic set

C5 used to gate this even harder than Phase 1, on the grounds that adding nine
tables generated the same way would propagate the same `q = 0` decode question
ninefold. It is cleared (2026-09-06): `q = 0` entries decode as 0 on this
hardware, including the 513-entry all-zero placeholder tables
`SQRT_LE`/`RSQRT_LE`/`LOG_LE`, so a self-derived table's zeros are safe.

What C5 leaves behind for this phase is a *domain* discipline, not a decode
one. The defect it actually found was `LutTable::log`'s documented domain being
an order of magnitude too wide, silently returning a `-1.0` clamp below
`x = 0.375`. Every table here is Q15 and holds `|f(x)| <= 1.0`; state each new
kind's accurate window as the range where that actually holds, and add the kind
to [`lut_zero_join_hw.rs`](iree-rocket-hal/tests/lut_zero_join_hw.rs)'s
full-code sweep, which drives all 256 input codes in a single NPU job and would
have caught it.

Board protocol applies -- accumulator results are order-dependent and the NPU
can go sick until reboot, so measure one shape per process.

### Phase 3 -- fusion, which is where the throughput actually is

This is P8's lever **#2**, and the only phase in this document that makes a
model faster. **Confirmed 2026-09-06, with a number: 1.80x faster than the
CPU arm on a full machine and level with it at two cores, from 1.5x slower.** See the requantized-path note
at the end of this section.

- `build_conv_then_lut_regcmd` and `build_conv_then_add_regcmd` are built,
  board-tested, and unreachable. Give them a wire representation as an
  **optional epilogue field on `Conv2DDef`**, not a separate union member: it is
  one dispatch that runs two hardware tasks, and modelling it as two kernels
  would recreate the boundary the phase exists to remove.
- Compiler side: a DAG matcher over `conv → elementwise` and `conv → lut` via
  `transform.iree.match.cast_compatible_dag_from_root`. Read the recorded
  DAG-matcher traps first -- there are three silent ways that op declines to
  match, and the debugging route is `iree-opt` on the isolated pattern.
- This recovers part of the `matmul_like` fusion the offload destroyed: 136
  fused kernels in the CPU-only build against 0 in the NPU build.

The requantized int8 path is the existence proof that this works -- it already
fuses requantization into the conv dispatch, which is why it returns `i8` with
no CPU epilogue at all.

**Started 2026-09-06, and the first move was to put that existence proof in the
tree.** It was not there: the whole path -- `Conv2DQuantParam` and
`runtime_quantization` on the wire, the driver's `RuntimeConv2dQuantParam`,
`#rocket_dynamic_int8_requant_target`, both requantized matchers, the lit test
and the e2e fixtures -- had been sitting unmerged on
`feature/more-mobilenetv2-convs` since 2026-09-03 while `main` moved twelve
commits past it. Both this file and ISSUES.md P8 were ranking work against
code no checkout contained. It is now on `main` and board-validated: the full
`tools/e2e_conv_regression.py --board planck` gate is green, including
`requant_int8_1x1_cin512` and `requant_int8_3x3_cin256` at max|error| 1 with
0 mismatches, which is the documented tie-rounding difference and not a defect.

**What is left is one compiler pass, and its shape is now known.** Measured on
`mobilenetv2.static-int8.onnx` (47 `onnx.QLinearConv`): after the spec's
`iree-global-opt-quantized-conv-to-conv`, all 34 dense convolutions still go
to the *accumulator* executable, because the requantized matcher's canonical
form is one convolution plus **one** elementwise generic and what the model
actually produces is a chain of five:

1. add the per-channel `i32` bias,
2. reduce the (constant) filter to `sum_k(w)` and subtract `x_zp * sum_k(w)`,
3. transpose NHWC -> NCHW,
4. `sitofp` and multiply by `x_scale * w_scale`,
5. divide by `y_scale`, round, offset by `y_zp`, clamp and narrow to 8 bits.

Steps 1 and 2 fold to a single constant per-channel bias, because the filter is
constant. Steps 4 and 5 collapse into the canonical generic. That is the pass.

**Built 2026-09-06 as `rocket-fuse-int8-requant-epilogue`, and a real model
reaches the requantized path for the first time: 15 of MobileNetV2's dense
convolutions.** It is as accurate as the accumulator build it replaces
(max|diff| 0.334 against a CPU arm, against the accumulator build's 0.396,
same top-5). The bounds were then raised on measurement the same day -- `Cin` 512 -> 816
and `Cout` 768 -> 1792 -- taking it to **29 of 34**, still at baseline
accuracy (max|diff| 0.336, argmax and top-5 unchanged).

**And it is faster, which is the first time anything in this document has
been.** Measured on `planck` over six interleaved passes: **1.54x faster than
a like-for-like CPU build on a full machine** (173.5 ms against 267.0),
1.40x faster on four A76s, and 23-30% faster than the accumulator build it
replaces at every core allocation. **Extending the same treatment to the 13
depthwise convolutions took it to 1.80x faster (148.5 ms) and to parity at
two cores**, the allocation that had punished the offload in every earlier
measurement. The accumulator build was 1.5x *slower*
than the same CPU arm. `ROCKET_PROFILE` puts the largest single term in
compaction, more than halved, which is the mechanism doing exactly what it
was supposed to: `i8` output instead of `i32` is a quarter of the bytes to
compact. ISSUES.md carries the table and the protocol. `Cin` stops at 816
rather than the 1792 every isolated instrument supports because the model
says so: see ISSUES.md's "Cin 1344 is exact in every isolated test and wrong
inside the model".

The model also found a hardware limit no fixture had: **`Cout = 24` is wrong
on the requantized path.** Admitting that one convolution moves the model's
logits from max|diff| 0.40 to 4.71 and the mean from 0.07 to 0.99 against a
logit standard deviation of 1.17 -- the output stops being a classification.
`Cout` 88 is exact and is not a whole number of 16-channel atoms either, so
the rule is not "whole atoms"; 24 is simply below the smallest `Cout`
measured correct. Both requantized matchers now carry `umin = 32`.

Two questions that looked like blockers and are not: the activations are ONNX
`ui8`, but `quantized-conv-to-conv` has already folded the unsigned-to-signed
shift into the zero point (30 of the 34 carry `x_zp = -128`), so the bytes the
NPU sees are already right; and the model's `Clip` (ReLU6) happens in `f32`
after requantization, so it stays a CPU pass and does not have to be part of
the fused form. The one real difference left to reconcile is that the model
narrows with `arith.fptoui` where the canonical form pins `arith.fptosi`.

### Phase 4 -- write the refusals down

Extend LIMITS.md's *"Datatypes: two different menus"* section with an op menu on
the same pattern: what the HAL supports, what the wire format supports, what the
matchers claim. Record the ~44 no-path ops explicitly. The point is that the
next pass over this question does not re-derive the same dead ends -- the same
service the existing "two menus" section performs for precisions.

---

## Recommended order

**~~Phase 0~~ → ~~`mulf` investigation~~ → ~~Phase 3~~ → ~~C5~~ → ~~Phase 1~~ → Phase 2.**

Phase 0, the `mulf` investigation, Phase 3, C5 and Phase 1 are done and
board-validated; only Phase 2 stands.

**Phase 0 produced the first counter-example to the warning above, and it is
worth reading before Phases 1 and 2.** P8's law -- cost is flat per offloaded
dispatch, so more sites make a model slower -- was measured on MobileNetV2,
where the added sites are small convolutions whose pack-and-compact overhead
rivals their arithmetic. VGG's five max pools are the opposite: large windows
over large images, each saving far more CPU work than the ~7 ms dispatch tax
costs. Offloading them is worth 212 ms of 1018, about 42 ms per site.

So the law is not "more sites are worse". It is "a site is worth offloading
when the op does more work than the dispatch tax", and MobileNetV2's
convolutions happen to sit on the wrong side of that line while VGG's pools
sit firmly on the right one. Phases 1 and 2 should be judged per op against
that bar rather than blocked by a blanket rule -- an element-wise op is
back on the wrong side, which is what P8 lever #3 was really saying.

| Step | Why here |
|---|---|
| ~~Phase 0~~ | Done 2026-09-05, and **measured**. MobileNetV2 and ViT are unaffected (dispatch-site counts identical). VGG's five `onnx.MaxPool` sites now offload and it is **1.26x faster** for it -- 1018 ms to 806 ms median, five interleaved passes. See below: this is the first counter-example to P8's law |
| ~~`mulf`~~ | Done 2026-09-06. It was bounded, and it was one register field: `EwBinaryOp::Mul` is bit-exact on `planck`, as are `Max` and `Min`. It changes the scope of Phases 1 and 2 by *removing* their only hardware unknown -- what is left there is matcher and wire-format work, not RE |
| ~~Phase 3~~ | Done 2026-09-06 (#26), and it delivered the throughput story it was ranked for: MobileNetV2-static-int8 is **1.80x faster** than a like-for-like CPU build on a full machine and level with it at two cores, from 1.5x slower. The one prediction here that was wrong is "needs no new schema breadth" -- carrying per-convolution calibration to a shared executable took a new `Conv2DQuantParam` enum and a `runtime_quantization` vector on `Conv2DDef` |
| ~~C5~~ | Done 2026-09-06. It blocked both remaining phases by its own stated action item; `lut_zero_join_hw.rs` discharges it and both LUT quirks are shown not to reach this stack. It cost the phases nothing -- no table or register changed. Its one real finding was a wrong domain in `LutTable::log`'s doc comment, which is a standard Phase 2 should hold itself to |
| ~~Phase 1~~ | Done 2026-09-06. All four layers, three kernel kinds, and `linalg.add`/`mul`/`sub` reachable from a compiled model behind `--elementwise`. It cost two HAL fixes the plan said would not be needed and one wrong assumption about the op form -- see the phase for both. ViT goes 12 -> 184 offloaded sites with the flag; the throughput question is open and is why the flag exists |
| Phase 2 | Breadth: new curves, the most additive and least urgent work here |

Phases 1 and 2 land their matchers **behind a flag**, with the `--no-offload`
arm measured alongside on every model. P8's measurement stands until something
displaces it: at the current per-dispatch cost, switching them on by default
would make MobileNetV2 slower.

Phase 3 moved that reference point and did not remove it. The int8 dispatch
tax is much lower than it was -- the model beats the CPU now -- but what
Phase 3 removed was `i32` activation traffic and unfused epilogues, which is
not a cost a standalone element-wise matcher has in the first place. An
element-wise op still does less work than its own dispatch costs, so the flag
stands and the per-op bar above is still how to judge one.

## What this roadmap does not propose

- **Integer and predicate element-wise ops.** ~35 of `arith`, 9 of `math`. No
  EW ALU evidence exists for any of them, and inventing a recipe without a
  capture or an oracle is the one thing this repo's element-wise modules
  explicitly refuse to do.
- **1-D and 3-D convolutions**, `linalg.conv_1d*` and `conv_3d*`. The builder is
  2-D throughout; a 3-D convolution is not a shape this hardware has.
- **`linalg.softmax`** as a single op. It decomposes into a max reduction, a
  subtract, `exp`, a sum reduction and a divide -- four of which land in Phase 1
  and 2, but the reductions have no PPU path outside the pooling window. Revisit
  once the pieces exist.
- **`linalg.pack` / `unpack`, `transpose`, `broadcast`.** These are layout ops,
  and P8's lever (1) already measured the layout-propagation approach at **0%**.
  Adding them as dispatches makes the layout problem worse, not better.
