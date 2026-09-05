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
| 9 LUT curves: sigmoid, tanh, exp, square, erf, sqrt, rsqrt, log, reciprocal | [`activation.rs`](iree-rocket-hal/src/rocket/activation.rs) `build_lut_regcmd` | `lut_hw`, `lut_exp_hw`, `lut_erf_hw`, `lut_sqrt_hw`, `lut_rsqrt_hw`, `lut_log_hw`, `lut_reciprocal_hw`, `ew_square_hw` |
| EW binary add / subtract | [`elementwise.rs`](iree-rocket-hal/src/rocket/elementwise.rs) `build_add_regcmd` | `conv_with_add_hw` |
| EW unary abs / neg / floor / ceil, and add-with-scalar | `elementwise.rs` `build_unary_regcmd` | `ew_unary_hw`, `ew_round_hw` |
| conv → LUT as two tasks in one job | `activation.rs` `build_conv_then_lut_regcmd` | `conv_then_lut_hw` |
| conv → EW add as two tasks in one job | `elementwise.rs` `build_conv_then_add_regcmd` | `conv_with_add_hw` |

[`rocket_executable_def.fbs`](rocket-schema/schema/rocket_executable_def.fbs)
has four `KernelDef` union members -- `Conv2DDef`, `FullyConnectedDef`
(deprecated, never emitted by any compiler), `PoolingDef`, `MatmulDef`. **None
of the element-wise or LUT work above is expressible.**

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
| **Blocked on `mulf`** | `powf` `fpowi` `ipowi` `fma` `atan2` | See below |
| No hardware path -- integer, bit-manipulation, or predicate-returning | `absi` `copysign` `ctlz` `cttz` `ctpop` `isfinite` `isinf` `isnan` `isnormal` | Refuse; document |

### `arith` -- roughly 11 of 55 reachable

The honest framing: `arith` ops are not tensor ops. They arrive inside
`linalg.generic` / `linalg.map` / `linalg.elementwise` bodies. Supporting
`arith` means matching those bodies, which is a `linalg` matcher problem.

| Class | Ops | Notes |
|---|---|---|
| EW ALU, hardware-confirmed opcodes | `addf` (algo 2), `subf` (4), `negf` (6) | `2` and `4` confirmed both precisions by the 47-model conv+add sweep; `5`/`6`/`7`/`8` confirmed by `ew_unary_hw`'s CPU-oracle tests |
| EW ALU, TRM-documented but **untested in this repo** | `maximumf`/`maxnumf` (algo 0), `minimumf`/`minnumf` (1), `divf` (3) | Needs a hardware oracle test before anything depends on it, same bar the unary opcodes cleared |
| Already handled elsewhere | `constant` → `RecordedOp::Fill`; `extf`/`truncf` → the packer's job, fold into an adjacent dispatch rather than offload | No new work |
| No hardware path | ~35 integer and predicate ops: `andi` `ori` `xori` `shli` `shrsi` `shrui` `divsi` `divui` `remsi` `remui` `cmpi` `cmpf` `select` `extui` `extsi` `sitofp` `fptosi` `index_cast` `bitcast` … | Refuse; document |

### `linalg` -- 19 of 23 named element-wise ops reachable

This is the striking result of the survey: **`linalg`'s named unary and binary
element-wise set is very nearly a one-to-one match for the HAL's existing
menu.**

- **Reachable**: `copy` `exp` `log` `abs` `ceil` `floor` `negf` `reciprocal`
  `round` `sqrt` `rsqrt` `square` `tanh` `erf` `add` `sub` `div` `max` `min`
- **Blocked**: `mul` (see below), `powf` (needs `mul`), `select` (needs a
  predicate), `div_unsigned` (integer)

Matchers must cover `linalg.elementwise` (the unified `ElementwiseKind` op) and
`linalg.map` alongside the named ops, because upstream canonicalization moves
freely between all three forms -- the spec already fights this for contractions,
where `linalg-specialize-generic-ops` has to be re-run after
`fold_unit_extent_dims_via_reshapes`.

Beyond element-wise, `linalg` also offers three things reachable with no HAL
work at all -- see Phase 0.

### `arith.mulf` is the single biggest blocker

Unblocking it opens `powf`, `fma`, `atan2`, `linalg.mul` and softmax scaling in
one move. It is currently blocked by a **known, unresolved hardware failure**
recorded in
[`ew_square_hw.rs`](iree-rocket-hal/tests/ew_square_hw.rs):

> A prior attempt used DPU MUL mode with ERDMA self-aliased to the primary input
> address (`build_square_regcmd`, since removed from `elementwise.rs`) --
> hardware-confirmed to produce all-zero output for every input, root cause not
> resolved.

`square` was shipped as a LUT instead, which was the right call for that op and
leaves the general case untouched. Note the failing configuration is
specifically the *self-aliased* one; a genuine two-tensor MUL through ERDMA's
ordinary path -- the configuration `build_add_regcmd` already uses successfully
for algo 2 and 4 -- has never been tried. That makes this a bounded
investigation, not open-ended reverse engineering, and it should run before the
breadth work rather than after.

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

### Phase 1 -- a wire format for what the HAL already validated

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

**[`iree-rocket-hal`](iree-rocket-hal/)** -- no new work. Wire the existing
`build_unary_regcmd` / `build_add_regcmd` / `build_lut_regcmd`.

**[`rocket-compiler-plugin`](rocket-compiler-plugin/)** -- matchers for the
`linalg` named element-wise ops, plus `linalg.elementwise` and `linalg.map`,
behind the Phase 0 flag.

> **Precision.** EW unary is fp16-only by design and must stay that way.
> `EwUnaryShape`'s doc comment states the standard: there is no capture
> confirming an int8 zero-point/scale recipe for this task shape, and this crate
> does not ship an int8 branch with zero hardware evidence behind it.

> **Gated on [ISSUES.md C5](ISSUES.md).** C5's own action item is *"add a dense
> sweep near 0 for every signed-output kind, and drive the tails at least once,
> **before relying on the LUT path in a compiled model**."* Phase 1 is precisely
> the step that starts relying on it. C5 also names QUIRK 2 -- a discrete `+128`
> spike within ~±0.0015 of zero on signed-output kinds, which `tanh`, `erf` and
> `log` all are -- and warns that a sparse-linspace gate steps straight over the
> band. Do C5's sweep first.

### Phase 2 -- new LUT tables

Purely additive to `iree-rocket-hal`. Follow the `LutTable::square()` /
`LutTable::erf()` methodology exactly: self-derived table, an explicitly
documented domain restriction, one `*_hw.rs` oracle test run on `planck`.

- **Tier 1**: `exp2` `log2` `log10` `log1p` `expm1` `erfc` `cbrt` `atan`
- **Tier 2**, domain design first: `asin` `acos` `atanh`, then the trig and
  inverse-hyperbolic set

C5 gates this even harder than Phase 1: adding nine tables generated the same
way propagates the same `q = 0` decode question ninefold. Fix C5 first.

Board protocol applies -- accumulator results are order-dependent and the NPU
can go sick until reboot, so measure one shape per process.

### Phase 3 -- fusion, which is where the throughput actually is

This is P8's lever **#2**, and the only phase in this document that makes a
model faster.

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

### Phase 4 -- write the refusals down

Extend LIMITS.md's *"Datatypes: two different menus"* section with an op menu on
the same pattern: what the HAL supports, what the wire format supports, what the
matchers claim. Record the ~44 no-path ops explicitly. The point is that the
next pass over this question does not re-derive the same dead ends -- the same
service the existing "two menus" section performs for precisions.

---

## Recommended order

**~~Phase 0~~ → `mulf` investigation → Phase 3 → C5 → Phase 1 → Phase 2.**

Phase 0 is done and board-validated; the rest stands.

| Step | Why here |
|---|---|
| ~~Phase 0~~ | Done 2026-09-05. Free coverage, no hardware risk, no new wire format. One caveat it did not honour: its own text said to gate the matchers behind a flag and measure both arms, and they went into `foreach_match` unflagged. MobileNetV2 and ViT are unaffected (dispatch-site counts identical), but VGG's five `onnx.MaxPool` sites now offload and that has not been benchmarked |
| `mulf` | Highest-leverage single unblock; bounded (one untried ERDMA configuration), and its answer changes the scope of everything after it |
| Phase 3 | The only phase with a positive throughput story, and it needs no new schema breadth |
| C5 | Blocks both remaining phases by its own stated action item |
| Phase 1 | Breadth: makes the validated HAL capability expressible |
| Phase 2 | Breadth: new curves, the most additive and least urgent work here |

Phases 1 and 2 land their matchers **behind a flag**, with the `--no-offload`
arm measured alongside on every model. P8's measurement stands until something
displaces it: at the current per-dispatch cost, switching them on by default
would make MobileNetV2 slower.

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
