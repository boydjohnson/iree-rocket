# Rocket compiler roadmap: shared planning and compile-time tiling

Status: proposed, grounded in the checkout on 2026-09-08. This document
describes future work; it does not change compiler or runtime behavior.

## Direction

Extract the pure shape, layout, hardware-limit, and convolution planning logic
from `iree-rocket-hal` into a new `rocket-core` crate. Both the compiler and
runtime should ask the same planner for a legal execution plan and receive an
explanation when no supported plan exists. This is a reasonable foundation for
compile-time tiling and useful CPU-placement diagnostics.

Use `Result<Plan, PlanError>`, with `PlanError` implementing `Display` and
`std::error::Error`. A descriptive String is the user-facing result, but retain
structured reason codes and values internally. The compiler must distinguish
an invalid operation, an unsupported lowering, a capacity limit that tiling
can resolve, and a policy whose hardware behavior has not been validated.
It must not infer those distinctions by parsing error text.

There are three separate deliveries:

1. Compute the existing HAL tile plans at compile time and explain placement.
2. Add compiler transformations for operations larger than the existing
   logical-shape limits, including output-channel tiling and eventually
   reduction tiling.
3. Own tensor layout and weight packing at compile time, so an executable
   states which of its bindings are packed and which dense, the constant
   filters ship prepacked, and activations are repacked only where a CPU
   dispatch meets an NPU one (section 6, added 2026-09-09).

Moving the planner alone delivers neither new reduction semantics nor automatic
CPU fallback after an NPU dispatch has already been selected.

## Current implementation and the gaps

| Area | Current code | Consequence |
|---|---|---|
| Convolution planning | `iree-rocket-hal/src/rocket/conv.rs`: `Shape`, `ConvPlan`, `Tile2D`, CBUF allocation, row/column partitions, accumulator staging | Tiling already exists, but pure planning is mixed with register emission and many unsupported cases panic. |
| Matmul planning | `iree-rocket-hal/src/rocket/fc.rs`: `Shape`, `Plan` | `[M,K] × [K,N]` maps to width M, height 1, Cin K, Cout N in a 1×1 convolution. Column tiles split M; this does not split K or N. |
| Runtime validation | `iree-rocket-hal/src/rocket/executable_format.rs`: `validate_conv_shape` | Rebuilds and parity-pads the shape, trial-plans under `catch_unwind`, then returns a generic `&'static str`, losing the actual refusal. |
| Dispatch execution | `rocket-hal-driver/src/executable.rs`, `command_buffer.rs` | Resolves runtime dimensions and plans execution; conv and matmul command recording also catch planning panics. |
| Compiler selection | `rocket-compiler-plugin/target/Rocket/rocket_conv2d_transform_spec.mlir` | Shape bounds and semantic matchers decide offload separately from the planner. The matmul matcher currently bounds M at 2047 and K/N at 3584. |
| Placement report | `rocket-compiler/src/report.rs` | Reports final executables and dispatch sites, but lacks reliable per-original-op rejection reasons. Its comments document original attributes being lost during optimization. |
| Compiler integration | C++ MLIR plugin plus Rust `rocket-compiler` embedding wrapper | Adding a Rust dependency to the wrapper is insufficient: the plugin must call the shared planner where the IR is inspected. |
| Wire format | `rocket-schema/schema/rocket_executable_def.fbs` | Conv and matmul definitions support runtime dimensions; there is no complete serialized compile-time tile-plan contract. |

Preserve `RocketVerifyConvShapesPass.cpp` as an early semantic tripwire. An
inconsistent convolution extent is a compilation error even if CPU fallback is
enabled: moving an incorrectly rewritten operation to CPU does not repair it.
Since 2026-09-09 (DYNAMIC_SHAPES.md DS4) it also checks each op's
shape-defining attributes against a record taken before the demotion, so it
holds on symbolic extents as well.

## 1. Extract a pure, fallible planning library

**Status 2026-09-09: milestone A landed.** `rocket-core` is in the workspace
with no dependencies; `conv.rs` holds the descriptors, limits, layout
geometry, CBUF partitioning, tiles and the pure `ConvPlan`, `fc.rs` the
matmul mapping, `error.rs` the `PlanError`/`PlanErrorCode` refusal, and
`policy.rs` the two environment overrides. Every panicking constructor has a
`try_*` twin (`Shape::try_with_precision`, `try_with_padding`,
`try_with_activation`, `try_with_depthwise`, `ConvPlan::try_new`,
`try_with_cbuf_banks`, `fc::Shape::try_new`, `fc::Plan::try_new`) and the
panicking one is now a wrapper with the same message. `ConvPlan::try_new`
checks the kernel against the padded extents and the coefficient footprint
against 32 bits before planning, and `Shape::try_with_precision` refuses a
surface that overflows the address space, so the planning path is free of
arithmetic panics on malformed descriptors. The HAL's `conv::ConvPlan` is a
newtype over the core plan that derefs to it and adds `programs()`;
`validate_conv_shape` returns the actual `PlanError`, with a `catch_unwind`
kept only as an `Internal` backstop. Evidence: the HAL's 177 unit tests
(register hashes) and the vendor-fixture scorers pass unchanged; 19 new host
refusal/overflow tests in `rocket-core`; old and new `iree-run-module`
produce bit-identical outputs on planck for fp16 (both widths) and int8
MobileNetV2. Still to do under this section: threading a `PlanningPolicy`
argument in place of `policy.rs`'s environment reads, and the pooling
planner, which stays in the HAL.

Add `rocket-core` to the Cargo workspace. It must build and test on the host
without a device, DRM, IREE, MLIR, or register submission dependencies.

Move the following together, retaining the evidence comments and fixture links:

- Logical conv/matmul descriptors, precision and relevant quantization metadata,
  layout geometry, channel padding, and checked size calculations.
- Hardware field limits, CBUF capacity and split policies, measured safety
  restrictions, parity padding, tile geometry, and scratch-layout planning.
- The pure portion of `ConvPlan` and the FC-to-convolution mapping.

Keep DMA addresses, relocation, register builders, device access, buffer
allocation, and submission in the HAL. Split methods such as `programs()` from
the plan data; HAL emission consumes a validated core plan. Use temporary
re-exports/adapters to avoid rewriting every HAL caller in the extraction PR.
Keep schema serialization in `rocket-schema` or an adapter to avoid a dependency
cycle. Core should not depend on the executable wire format.

Replace input-dependent asserts and panics throughout the reachable planning
path, not just `ConvPlan::new`. Validate positive dimensions, layout and dtype,
groups/depthwise rules, kernel/stride/dilation/padding, output extents,
quantization, integer conversions, byte counts, offsets, and allocation budgets
with checked arithmetic. Separate logical extents from padded physical extents.
Compiler input dimensions should remain wide integers until checked conversion;
do not truncate an MLIR dimension into a register-sized field.

A proposed API, with names to be finalized during extraction:

```rust
pub fn plan_conv(
    op: &ConvDesc, target: &TargetCaps, policy: &PlanningPolicy,
) -> Result<ConvPlan, PlanError>;

pub fn plan_matmul(
    op: &MatmulDesc, target: &TargetCaps, policy: &PlanningPolicy,
) -> Result<MatmulPlan, PlanError>;
```

Plans carry logical/programmed shapes, tile input and output regions, halos,
physical padding, CBUF partitions, scratch requirements, output placement, and
the constraints responsible for splitting. A multi-tile plan is success, not
an error. A refusal carries an operation description, stable code, actual and
allowed values where applicable, and a precise explanation. Distinguish
`InvalidShape`, `UnsupportedSemantics`, `CapacityExceeded`,
`UnvalidatedConfiguration`, and `PlanningBudgetExceeded` (illustrative names).

Keep hardware legality separate from conservative compiler admission policy
and performance policy. A register-representable configuration is not necessarily
hardware-proven; existing comments explicitly distinguish derived partitions
from board-validated ones. Do not silently broaden admission during extraction.
An explicit CBUF override is a diagnostic/research option, not an automatic way
around a refusal.

Acceptance: existing accepted cases produce identical tile geometry, scratch
layouts, and emitted register programs. Refusal cases return specific errors
without panics. Runtime validation and emission use the same normalized plan,
including parity padding and accumulator staging.

## 2. Call the planner during compiler selection

**Status 2026-09-09: the link, the decision record, and the admission gate
have landed.** `rocket-plan-ffi` (`include/rocket_plan.h`, ABI version 1)
exposes `rocket_plan_conv`/`rocket_plan_matmul` over fixed-width
descriptors with 64-bit extents, a caller-owned message buffer,
`struct_size` checks and `catch_unwind` at the edge; the plugin's CMake
builds it with cargo into its own target directory and links it into
`libIREECompiler.so`. `rocket-plan-candidates` runs right after
`rocket-verify-conv-shapes` -- before the first claiming loop, so it sees
every candidate -- records one decision per convolution/matmul (`direct`,
`tiled`, `cpu`, `deferred`, with the planner's status and message) in a
`rocket.plan_decisions` attribute on the function with a location-only
remark per refusal, and tags each *refused* op `rocket.plan_refused`.
Every shape-admitting sequence matcher in the spec (25 of them) now carries
`transform.rocket.match.admitted %root`, the plugin's own transform-dialect
matcher (`RocketTransformOps.td`), which declines a tagged op; the DAG
matchers decline it by attribute-dictionary inequality. Admission is thus
the matchers' own bounds AND the planner's acceptance -- a strict
narrowing, deliberately: no `dim_bounds` was removed, so nothing got
broader and `spec::neutralize` is untouched. What it closes: a shape the
bounds admit but the planner refuses -- a dense-layout row too wide for one
CBUF bank, say -- used to compile to a dispatch the runtime rejected with a
bare `INVALID_ARGUMENT`; it now falls back to the CPU
(`rocket_plan_gate.mlir`). Parity: MobileNetV2 fp16, 53 candidates, 0
refusals, 47 sites unchanged; ResNet50 fp16, 54 candidates, 1 refusal (the
7x7 stride-2 stem) that the matchers already left on the CPU, 53 sites
unchanged; the boundary corpora unchanged. `rocket-compiler compile
--no-offload` now also verifies on the placed module that zero Rocket
executables exist and refuses to write otherwise, which is what made the
next step safe.

**Status 2026-09-09: the `dim_bounds` are gone from the convolution and
matmul matchers, and the ceilings they spelled are one table in
`rocket-core`.** `admission.rs` holds `ConvAdmission`/`MatmulAdmission` and
a `conv_ceilings` table keyed by precision rung, depthwise, kernel and
stride, carrying the evidence comment that set each entry; `rocket-plan-ffi`
exposes it as `rocket_admit_conv`/`rocket_admit_matmul` (ABI version 2, no
spatial extents and no calibration numbers in the descriptor, so it answers
on a convolution whose height and width are still symbolic);
`transform.rocket.match.admitted` gained an optional `precision` attribute
and now asks *both* questions -- the planner's refusal tag and the admission
envelope -- so every convolution and matmul matcher carries exactly one
shape predicate and no numbers. `RocketPlanQuery.{h,cpp}` is the one linalg
reader the pass and the matcher share, so the two cannot disagree about
which operand is the filter.

Three things this closed. The `precision` attribute makes the two int8
lowerings ask different questions from the same IR: a requantized
convolution and an accumulator one are both `i8 x i8 -> i32`, and the
matcher now says which rung it is claiming for. A dynamic channel count is
declined with a message instead of `dim_bounds`' silent
`computeConstantBound` failure (DYNAMIC_SHAPES.md DS1). And the per-matcher
drift is gone: the old table admitted a stride-2 1x1 convolution to `Cin`
3584 when a ReLU followed it and to 512 when nothing did, and the epilogue
a matcher fuses -- bias on the BS plane, activation on the BN plane -- is
downstream of the MAC array and does not touch the channel path, so each
class now takes the largest value of its row.

`spec::neutralize` moved with them: `--no-offload` adds a `no_offload`
attribute to each `transform.rocket.match.admitted` and still rewrites the
`dim_bounds` the pooling and element-wise matchers keep, and it refuses a
spec whose loop contains a matcher carrying neither. Parity: MobileNetV2
fp16 47 sites (53 candidates, 0 refusals), ResNet50 fp16 53 sites (54
candidates, 1 refusal -- the 7x7 stride-2 stem the matchers already left on
the CPU), VGG19 21 sites, `--no-offload` 0 sites with and without
`--elementwise`; all unchanged, so no measured model sits in a widened
corner.

**Status 2026-09-09: the first deliberate raise off the shared table.** With
the ceilings in one place it was visible that fp16 depthwise sat at 512
while `ConvPlan` plans it to 1792 and the int8 depthwise rung was already at
1344 -- so six of MobileNetV2's seventeen depthwise convolutions were on the
CPU for want of a number nobody had revisited. Raised to 1536, the extent the
compiled end-to-end gate now proves: `tools/e2e_conv_regression.py` gained
`depthwise_fp16_c576`, `_c960`, `_c1536` and `_c1536_s2`, each compiled twice
from one MLIR and compared on `planck` at atol 1e-3 (max|error| 1.6e-4 to
2.4e-4, 0 mismatches). MobileNetV2 fp16 goes 47 -> 53 sites, max|diff| 0.0156
against its own `--no-offload` arm with top-1 and top-5 unchanged, and 4.6 ms
slower at four workers / flat at eight -- the ISSUES.md P7 verdict on cheap
depthwise dispatches, unchanged. That gap between "characterized" and "worth
offloading" is the third policy this section's opening asks for and the one
still missing: admission and hardware legality are now separate, cost policy
is not.

Still open here: the cost policy above and the dynamic-shape policy. The
strict-offload option landed with section 3.

Expose core through a small versioned C ABI adapter, built as a host Rust static
library and linked into the C++ plugin through its CMake build. Use fixed-width
descriptors and explicit status/result ownership. Never expose Rust Strings,
Vecs, or enums directly over the ABI. Provide release functions or caller-owned
buffers, and ensure no panic unwinds across C++.

Integrate a planning pass or transform extension after layout/precision and
epilogue normalization, before replacing candidates with Rocket dispatches and
pinning unclaimed work to CPU. Initially keep semantic matching in MLIR:
indexing maps, supported op forms, initialization/accumulation, dtype conversion,
and fusion patterns are not just shape checks. Replace duplicated hardware
shape predicates with planner queries incrementally, using parity tests first.

Record a decision for each candidate:

- Direct Rocket plan.
- Tiled Rocket plan, with split axes, tile count, and limiting constraint.
- Valid operation left on CPU, with semantic, planning, or performance reason.
- Invalid operation or required-offload failure, with a compilation error.

Do not report every matcher attempt as a refusal: aggregate alternatives and
report the selected outcome or the decisive failure. Record unsupported forms
that never reach the planner too, such as unsupported matmul indexing maps.

Static shapes can be fully planned. For dynamic shapes, either prove a guarded
plan from bounds or explicitly report deferred runtime planning. Until guarded
CPU/NPU branching exists, unknown legality should conservatively select CPU
under mixed-device policy; preserving an existing dynamic NPU path must state
its runtime constraints and possible runtime error. Runtime rejection alone
cannot reroute the already compiled dispatch to CPU.

Acceptance: compiler and runtime agree on a shared boundary corpus, and the
plugin works both through `rocket-compiler` and direct IREE compiler invocation.
Update `spec::neutralize`/`--no-offload`: it currently disables matchers by
rewriting dimension bounds, which will no longer be sufficient once planner
queries replace those bounds. Preserve the like-for-like CPU baseline pipeline.

## 3. Explain placement reliably

**Status 2026-09-09: landed.** `rocket-compiler audit` now prints one line
per convolution and matmul candidate -- source location, kind, layout,
shape, precision rung, decision, limit class and the planner's own message
-- then the placement report, then a reconciliation of the two.
`--report-json` writes the same record machine-readably from `audit` or
`compile`, and `--strict-offload` fails the compile instead of falling
back.

Where each half comes from, and why not from one place. The decision record
is read at the end of the **preprocessing** phase, the phase
`rocket-plan-candidates` runs in (`rocket-compiler/src/decisions.rs`); the
placement report is read at `executable-targets` as before. The record does
survive to `executable-targets` on every model measured here, so one dump
would have worked -- but whether a discardable attribute survives is a
property of which passes IREE happens to run, and `report.rs` documents the
`rocket.origin` tags that did not. `audit` and a reporting `compile`
therefore stop at one more phase; the resulting `.vmfb` is byte-identical to
one built without stopping (MobileNetV2 fp16, sha256 equal across plain,
`--report-json` and `--strict-offload` builds).

What the reconciliation is allowed to say. A Rocket dispatch has no per-op
name -- the spec splices a handful of fixed executables across every matched
shape -- and a CPU dispatch's name encodes its own loop ranges, not the
candidate's logical shape (a convolution's are its *output* extents plus the
kernel, with the input padding already folded away). So the join is by count
and by op *kind*, never by a claimed per-op identity, and the report says
"at least N" where N is a floor. What it does separate is the three cases a
reader acts on: the planner refused, the admission envelope has no evidence
for the shape class, or both said yes and a *matcher* still declined -- the
last being a semantic or fusion gap in the transform spec, and the only one
of the three that closes without new hardware measurements.

Three defects this found and fixed. (1) The decision record asked only the
planner, never the admission envelope, so a candidate the envelope declines
-- which every matcher then declines -- was recorded as `direct`. The pass
now asks both, and for an `i8 x i8 -> i32` operation it tries *both* int8
rungs before calling the class unmeasured, because which rung applies is
decided by the matcher that claims it and not by the operand types. The
`rocket.plan_refused` tag is deliberately **not** extended to admission
refusals: the tag makes every matcher decline, and this pass cannot know
which rung a matcher is claiming for. (2) `report.rs` read only bare MLIR
symbols, so every executable in a module whose entry point is
`torch-jit-export$async` -- i.e. every ONNX import through torch-mlir --
came back unnamed with zero dispatch sites. (3) `rocket_plan_precision_name`
was added to the C ABI (version 3) so a report names a rung in the same
spelling the transform spec's `precision` attribute uses.

Evidence: 39 plugin lit tests pass, including `rocket_plan_candidates.mlir`
with the new fields, an admission-declined depthwise proving it is *not*
tagged refused, and an int8 3x3 admitted only on the requantized rung; 38
`rocket-compiler` unit tests including the reconciliation and strict-offload
cases; MobileNetV2 fp16 unchanged at 53 candidates -> 53 Rocket dispatch
sites -> 77 hardware jobs, with `--strict-offload` passing on it and failing
with a named reason on a module carrying a Cin-3585 convolution.

Still open here: the cost policy. `limit` has a `cost` class and the report
prints it, but nothing produces one -- a candidate that plans and admits but
would be slower on the NPU (ISSUES.md P7's depthwise convolutions) is still
recorded as accepted. That is the same gap section 2 closes with, and it
needs a cost model before a report can honestly say so.

Extend `rocket-compiler` reporting with a structured decision report generated
at selection time and reconciled with final placement. Preserve origin IDs via
explicit rewrite mappings or a dedicated report sink; do not assume arbitrary
op attributes survive IREE passes. Track one-to-many mappings when an operation
is tiled, fused, or decomposed.

Offer readable remarks and an optional machine-readable report. Include source
location/origin, op kind and shape, precision, placement, reason code/message,
tile summary, and whether a limit is hardware, validation policy, or cost policy.
Keep original-op counts, compiled dispatch-site counts, and hardware-job counts
separate: one dispatch can already execute several HAL tiles.

Illustrative messages (not claims about newly supported shapes):

```text
matmul at model.mlir:42 -> Rocket, tiled along M
  full row exceeds the CBUF slab-address limit; selected legal column tiles

conv at model.mlir:81 -> CPU [UnvalidatedConfiguration]
  no validated automatic CBUF partition for this kernel/precision/channel combination

conv at model.mlir:96 -> error [InvalidShape]
  output height disagrees with input, kernel, stride, dilation, and padding
```

For numeric refusals, print the actual dimension/footprint and the applicable
limit. Say “no supported plan under the current policy” when that is what is
known, rather than claiming the hardware can never execute the operation.
Add an explicit strict-offload option if desired, with documented scope, so
requested conv/matmul candidates fail compilation instead of falling back.

## 4. Materialize existing tile plans at compile time

**Status 2026-09-09: not started, and deliberately so.** Section 3's report
made the cost of what this section removes measurable, and it is small.
Runtime planning happens inside `Phase::Record` (`ConvPlan::new` at
`command_buffer.rs`), which `ROCKET_PROFILE=1` puts at **1.6 ms of a 136.7 ms**
MobileNetV2 fp16 inference -- 1.2% of wall, and tile search is only a
fraction of that phase. Against `outside` at 79.6 ms and `compact` +
`pack.input` at 24.4 ms, serializing the plan is not where the time is.

Two of the three things this section was also going to buy have since been
delivered by section 2: the compiler already asks the same planner the
runtime asks, and already refuses at compile time what the runtime would
reject. What remains unique to section 4 is register-program equality as a
*proven* property rather than a shared-crate argument, and a target/policy
identity in the executable.

**That number has since been re-taken, and it is much larger.** ViT-B/16
(f32 import, 73 NPU dispatch sites) spends **117 ms of a 592 ms inference in
`record` -- 20% of wall**, at 1.6 ms per dispatch against MobileNetV2's
0.043 ms. Planning cost scales with dispatch count *and* with how much
search each shape needs, and a transformer's wide matmuls need far more of
both than a MobileNet's convolutions. So the 1.2% above is the bottom of the
range, not the middle of it, and section 4 is worth roughly an order of
magnitude more on the models this repository is now measuring.

What that does *not* do is rescue attention offload (ISSUES.md C15): removing
`record` entirely would still leave 83 ms of cost against 3 ms of benefit
there. But it does change section 4's own case, and it makes the ordering
argument concrete -- every future increase in dispatch count is taxed at
1.6 ms until this is done.

**Qwen3-0.6B answers that, and it complicates the picture** rather than
confirming it. Three models, `ROCKET_PROFILE=1` on `planck`, `record` being
the phase `ConvPlan::new` runs in:

| model | NPU sites | record ms | ms/dispatch | wall ms | record % | `outside` % |
|---|---:|---:|---:|---:|---:|---:|
| MobileNetV2 fp16 | 37 | 1.6 | 0.043 | 136.7 | 1.2% | 58% |
| ViT-B/16 f32 | 73 | 117.0 | 1.603 | 592 | **19.8%** | 35% |
| Qwen3-0.6B f32 (prefill 128) | 196 | 449.8 | 2.295 | 24080 | 1.9% | **96%** |

Two separate things, and conflating them is what made the ViT number look
decisive:

- **Cost per dispatch is a property of the shape class**, and it scales
  cleanly: 0.043 ms for a MobileNet convolution, 1.6 ms for a ViT projection,
  2.3 ms for a Qwen3 one. Wider matmuls give the CBUF partition search more
  to do. So section 4 removes more work per dispatch the bigger the
  operations are.
- **Share of wall is a property of the model**, and it does not track
  dispatch count at all. Qwen3 has 2.7x ViT's dispatches and a *tenth* of the
  relative planning cost, because 96% of its wall is `outside` -- CPU work
  this backend never touches. Its `npu share` is 1.3%.

So the honest statement of section 4's value is: it is worth roughly a fifth
of a model whose NPU half is the bottleneck, and roughly nothing on a model
bottlenecked elsewhere. Qwen3 is bottlenecked elsewhere for a reason worth
naming -- its single refused candidate is the LM head,
`128x1024 x 1024x151936`, whose `N` is 37x the channel ceiling and which
therefore runs on the CPU. Offloading that would cut `outside` sharply and
raise `record`'s share with it, so the two items are coupled: section 4 gets
more valuable exactly as section 5's N-splitting (or a very large ceiling
raise) succeeds.

Nothing here changes the conclusion that section 4 is not the first thing to
build. It does change the reason: not "the saving is 1.2%" but "the saving is
between 2% and 20% depending on where the model's time actually goes, and the
cheapest way to raise it is to offload more of the model first".

First preserve the current execution model: one logical Rocket dispatch owns
multiple standalone hardware jobs. Serialize the selected static plan with the
executable rather than immediately introducing an MLIR dispatch per tile.
This makes planning genuinely compile-time while reusing HAL packing,
accumulator staging, and scheduling.

Extend the schema with optional versioned plan data and a target/policy identity.
Keep old executable decoding supported; old files and supported dynamic cases
use the shared runtime planner. Validate decoded plans for bounds, coverage,
scratch size, target compatibility, and safe address arithmetic before emission.
Reject unsupported plan versions clearly; never trust serialized tile offsets.
Specify how compiler/runtime policy mismatches are handled.

Preserve disjoint logical output coverage, overlapping input halos, physical
padding, channel surfaces, and scratch compaction. Physical output writes can
require staging even when logical tile regions do not overlap. Avoid replanning
static shapes at every dispatch; addresses and buffer lifetimes remain runtime
concerns. Keep packing ordered after producer completion, including NPU-to-NPU
edges (the existing C13 regression is relevant).

Acceptance: decoded compile-time plans reproduce runtime-planned results and
register programs; static execution no longer performs tile search. Dynamic
execution retains explicit validation. Measure executable size and compiler
time as well as runtime planning savings.

## 5. Tile operations beyond today's logical-shape limits

**Status 2026-09-09: the premise was wrong, and the coverage it was for was
delivered another way.** This section assumes operations exist that the
hardware cannot execute in one dispatch and that decomposition is what
reaches them. Swept with the capture-backing ceilings lifted
(`rocket-core/examples/find_structural_limits.rs`), the planner already
plans every shape steps 1 and 2 were written for:

| axis | planner |
|---|---|
| conv `Cout` to 131072 | ok, 1 tile |
| matmul `N` to 65536 | ok, 3 tiles |
| matmul `M` to 65536 | ok, 737 tiles -- `ConvPlan` *already* splits `M` |
| conv `Cin` / matmul `K` | ok to 8192; refuses at 16384 |

Not one refusal in that sweep is structural; even the `Cin` 16384 one is a
CBUF *grant* message, not a register field. Every limit that kept these
shapes on the CPU was an `admission.rs` ceiling -- `UnvalidatedConfiguration`,
the class section 3's report prints as "register-representable, not
measured". Step 1's spatial-conv half had no failing case at all: a
1024x1024 image plans into 341 hardware jobs today, and 4096 wide and 8192
tall both plan.

**What was done instead**: the ceilings were measured and raised, which is
this repository's standing recipe for exactly this situation. Dense `Cin` and
`Cout` 3584 -> 4096 at every precision rung, and matmul `M` 2047 -> 4096.
LIMITS.md carries the ladders, the `onehot`-read-map reasoning behind the `M`
sweep, and the compiled gates (`matmul_m_4096`, `matmul_k_n_4096`,
`dense_fp16_cout4096`). Zero lines of new compiler code, and the work stays
one dispatch with one pack and one compact -- where MLIR-level slicing would
have paid the 19%-of-wall per-dispatch tax once per slice.

**What is still genuinely this section's**, and what it now means:

1. **Decomposition as an alternative to measurement.** Slicing an oversized
   operation into slices that each sit inside an *already-gated* class buys
   coverage with no new hardware campaign. That is a real motivation and the
   one thing raising a ceiling cannot offer -- but it is a trade against the
   per-dispatch cost, not a way to reach the unreachable, and it should be
   written up as such rather than as "beyond today's limits".
2. **`K`/reduction splitting** (step 3 below) is untouched and remains the
   only axis where a real wall exists at a reachable extent: the CBUF grant
   at `Cin` 16384. It still needs accumulation semantics, per-slice
   zero-point correction, and defined overflow and floating-point ordering,
   none of which exist.
3. **The next ceiling raise** is cheaper than either: fp16 `k=1` measured
   clean to 8192 in the 2026-09-06 corpus and `M` to 8192 in the 2026-09-09
   one. Both stop at 4096 only because that is the widest extent measured at
   *every* rung.

The step list below is kept as written, because its halo, coverage, tail and
zero-point requirements are all still correct for whoever builds
decomposition. Only its premise -- that these shapes are otherwise
unreachable -- has been falsified.

This needs a higher-level decomposition planner and compiler rewrite support.
Keep whole-operation descriptors distinct from legal hardware-task descriptors:
constructing today's bounded `Shape` first would reject the large operation
before decomposition has a chance to succeed. Validate every resulting task
through the same core planner.

Implement in this order:

1. **Spatial conv and M matmul splits beyond current whole-shape bounds.**
   Generate bounded slices or a compact loop representation, preserve global
   tensor strides, and assemble the output. For convolution, derive input
   regions from output coordinates, stride, dilation, and kernel footprint;
   apply padding only at true image boundaries. Keep batch and depthwise
   semantics explicit. Existing HAL tiling within accepted shapes is the
   reference, not proof that arbitrarily large parent shapes already work.
2. **Output-channel/N splits.** Slice weights and per-channel bias/scales,
   preserve packing alignment and tails, and assemble disjoint output channels.
   For grouped/depthwise convolution, preserve each output group's input
   association. No reduction is needed, but layout and epilogue handling still
   need implementation and hardware validation.
3. **Input-channel/K reduction splits.** Defer until accumulation semantics
   are implemented. Partial dot products must combine at the required precision;
   bias, activation, clamping, and requantization happen once after reduction.
   Do not sum separately narrowed fp16 or requantized int8 outputs. Define
   integer overflow behavior and floating-point tolerances/order explicitly.
   Account for zero-point correction per slice. If CPU accumulation is the
   initial implementation, report and measure that mixed plan honestly.

Choose between compiler-generated slices/loops and extending the runtime plan
executor for these new decompositions before expanding the schema further.
Neither path may hide unsupported reduction behavior behind larger matcher bounds.

Bound search time, tile count, scratch, and generated IR size. Start with
deterministic legal candidates; add a cost model after correctness. Consider
packing, repeated weights, halo traffic, submission, compaction, and CPU epilogues,
not only MAC count. Existing `ROADMAP.md` and `ISSUES.md` show why more offloaded
dispatches need not make a model faster.

## 6. Own the layout at compile time: packed tensors, boundary-only repacks, prepacked weights

**Status 2026-09-10: steps 1 and 2 of 6.6 -- the layout contract (6.1) and
compile-time weight packing (6.3) -- landed; steps 3 and 4 not started.** This section exists because the end state the
repository is working toward was only ever stated by halves. Put in one
place, for a model whose input extents are static:

1. the compiler knows which operations run on the CPU and which on the NPU
   -- **landed**, sections 2 and 3;
2. the compiler knows, per tensor edge, whether the tensor is in the NPU's
   packed `NC1HWC2` cube layout or in IREE's dense row-major one, and the
   executable says so rather than the driver guessing -- **not started**;
3. every constant filter is stored in the `.vmfb` already in the CNA's
   blocked coefficient order, so the driver never packs a weight --
   **landed**, 6.3;
4. activations are packed and unpacked only where a CPU dispatch meets an
   NPU one, never between two NPU dispatches -- **the runtime mechanism
   landed as ISSUES.md P2 steps 2, 4 and 5; the compile-time form has not
   been designed**.

Items 2 to 4 are one piece of work, and this section is its design and
order. The reason it belongs in this file and not in ISSUES.md is that
every piece of it is a compiler decision that the runtime currently makes
by inspection, and the argument of section 2 -- ask one planner, once,
at compile time, and let the runtime check rather than rediscover -- is
exactly the argument here.

### What the runtime does today, and why it is discovery rather than knowledge

The driver already achieves item 4 on most edges, by proving at record time
an identity the compiler could have stated. `chainable_cube`
(`rocket-hal-driver/src/command_buffer.rs`, ISSUES.md P2 step 2) matches a
dispatch's input binding against the *device byte range* of every earlier
dispatch's output on the same command buffer, and if the producer's
`OutputCube` has the same pixel count, the same surface stride
(`surface_pixel_count`, which the PPU rounds up to four) and a whole number
of 16-byte atoms per pixel, the consumer reads the producer's scratch BO in
place and skips its pack. `compaction_elidable` then skips the producer's
dense write when the number of consumers that chained equals the number of
Rocket readers the compiler counted (`rocket-mark-dense-readers`,
`Conv2DDef.runtime_dense_readers` and its twins on `MatmulDef`,
`PoolingDef` and the three `Elementwise*Def`s). It is bit-identical on and
off (`ROCKET_CHAIN`, `ROCKET_LAZY_COMPACT`) and it is gated end to end by
`fp16_conv_pool_conv_chain`, `fp16_matmul_chain` and `requant_int8_chain`.

That is the mechanism working well, and the numbers say so: ResNet50 fp16
chains 66 of 68 operand repacks and elides 51 of 53 compactions, 169 -> 114
ms; VGG f32 20 of 20 edges. It is also the reason this section is not a
throughput proposal on those models -- there is little left to remove.

What it is instead is a set of decisions made in the wrong place, each with
a cost that shows up somewhere other than the benchmark:

| Runtime decision today | What it costs |
|---|---|
| A packed cube exists only as a driver-private scratch BO, pooled per context (`scratch_pool.rs`) and matched by byte range | A consumer on a *later* command buffer cannot chain, by construction. `rocket-mark-dense-readers` had to be a count rather than a flag purely to survive IREE's partitioning of the program, and its own header says so. |
| Chainability is decided by geometry equality at record time | The whole-atom rule declines 8 of MobileNetV2's 14 direct NPU -> NPU edges (Cout 88, 136, 24). The compiler knows every one of those channel counts at compile time and could have padded the producer's programmed width, chosen a different consumer, or reported the edge as unchainable in the placement audit. Today the audit cannot mention layout at all. |
| The dense write is elided when a runtime tally matches a compile-time count | Correct, but two mechanisms for one fact. A reader the pass does not understand is "always write", silently, and nothing in section 3's reconciliation can say which results are written and which are not. |
| Weights are packed by `apply_ops` on first use and cached across inferences (`weight_cache.rs`) | 65.8 ms and 7.5 MiB per cold start on MobileNetV2 int8 (P8: 49 misses, 1666 hits over 35 inferences); a generation counter and a "nothing writes it before this dispatch" rule that every new way of writing a buffer must be taught about. |
| Two environment variables can change what a `.vmfb` does | Not a correctness risk -- both arms are bit-identical -- but a `.vmfb` is not self-describing, and section 4's acceptance ("static execution no longer performs tile search") has a layout twin that is not met: static execution still performs layout search. |

The `truncf`/`extf` cancellation is the same story one layer up. A Rocket
shim widens its f16 result with a plain `linalg.generic` so that the
consumer's own `truncf` folds against it and the edge becomes direct
(P2 step 5, `rocket-fold-neutral-init`). That works, and it is what made
chaining reach VGG's plain convolutions -- but it is IREE's canonicalizer
being relied on to discover that the two dispatches agree on an element
type. An encoding on the edge would state it.

### 6.1 The layout contract

The cube geometry has to be written down as a function of things the
compiler knows, and nothing else. From `OutputCube` and the identity
`chainable_cube` checks, that function is:

- inputs: dispatch kind (conv, matmul, pooling, element-wise), precision
  rung, logical width, height and channel count (or `M` and `N` for a
  matmul, which is width `M`, height 1);
- outputs: `pixel_count`, `surface_pixel_count` (equal to `pixel_count`,
  except the PPU rounds it up to a multiple of four), `bytes_per_pixel`
  (channels rounded up to the rung's 16-byte atom: C2 = 8 lanes at fp16,
  16 at int8), and therefore the physical byte size, which is larger than
  the dense tensor whenever the channel count is not a multiple of C2 or
  the PPU rounding applies.

And the things it must **not** depend on, each of which is a runtime
freedom today and becomes a constraint:

- **Tiling.** A multi-tile conv writes one cube; `Tile2D`'s row and column
  partitions are internal to it. This holds today and section 4's serialized
  plan does not change it.
- **Fan-out** (MULTICORE.md M2). A fanned-out dispatch writes per-context
  tiles and publishes no cube. Under a compile-time layout a fanned-out
  dispatch must either assemble one cube or be declared dense-out; "declines
  to chain" is no longer an available answer.
- **Command-buffer partitioning.** A packed tensor that lives in the
  dispatch's own IREE result buffer, rather than driver scratch, is readable
  by any later command buffer on the same device file. That is the change
  that removes the count/flag distinction. It collides with P1 and
  `rocket-no-cross-fd-bo-sharing`: a job names only BOs created on its own
  file, so a packed tensor is only reachable from the context that owns the
  IREE allocation. Multicore (N contexts) and compile-time layout therefore
  need one decision about who allocates, and it should be made before
  either grows further.
- **The accumulator int8 rung** has no identity at all -- it writes
  128-byte blocks at 4 lanes per atom against a 16-lane input -- and stays
  dense-out. The requantized rung does (i8 out, 16 lanes) and is gated.

Write the contract as a pure function in `rocket-core` beside `admission.rs`
(`layout.rs`: `cube_geometry(kind, precision, w, h, c) -> CubeGeometry`),
expose it over `rocket-plan-ffi` (ABI version 4), and make the driver's
`OutputCube` construction and `chainable_cube`'s comparison both call it,
so the compiler and runtime cannot compute two geometries for one shape.
That is section 1's extraction pattern applied to layout, and it is the
prerequisite for everything below because it turns "does the producer's cube
match" into "is the encoding equal".

**Landed 2026-09-10.** `rocket_core::layout` holds `CubeKind`,
`CubeGeometry`, `cube_geometry`, `packed_channels` and
`chain_identity -> Result<(), ChainRefusal>`, with `conv::Shape::
input_cube_geometry` / `output_cube_geometry` as the compiler-facing entry
points (`None` where the shape reads dense ARGB or writes accumulator
blocks). `rocket-plan-ffi` is ABI 4: `rocket_plan_cube_geometry`,
`rocket_plan_chain_identity` and a `ROCKET_PLAN_LAYOUT_MISMATCH` status
whose message is the failing condition. In the driver `OutputCube` carries
a `CubeGeometry`, all six `chainable_cube` call sites and five cube offers
build theirs through `cube_geometry`, `chainable_cube` itself is
`chain_identity` plus the two things only the runtime knows (which recorded
write produced the bytes, and whether it published a cube), and
`PoolingShape::programmed_channels` reads `packed_channels`. Every
`chain_identity_tests` case now asserts the pure verdict beside the bytes
it was already pinning.

Acceptance, planck 2026-09-10, before/after `iree-run-module` built from the
same HEAD with only this change stashed: bit-identical outputs on ResNet50
(the P2 build, 66 chains taken / 3 declined on both, per-site sets equal),
MobileNetV2 (2 taken / 34 declined, same per-site reasons), Wide ResNet50,
CLIP, BLIP, ViT-L/16, and int8 ResNet50 and MobileNetV2 (int8 ResNet50 three
alternating isolated runs; its one watchdog trip was the thirteenth
consecutive NPU process, not this change, and did not reproduce alone). The
four chain fixtures pass through the e2e gate with the ABI-4 plugin.

Moving the numbers exposed one correction to the contract as it was stated
above. The channel padding is a property of the **reader**, not the rung:
the CNA-fed kinds (conv, matmul, element-wise) pad to 16 *lanes* whatever
the element width -- two atoms at fp16 -- while the PPU programs whole atoms
only, 8 lanes at fp16 and 16 at int8. `bytes_per_pixel` is still channels
times element bytes, but the identity's consumer-side condition is "no
padding under *the consumer's* rule", so a conv output at Cout 24 fp16 (48
bytes, three whole atoms) feeds a pool in place and not a conv. That is what
MobileNetV2's "whole-atom" declines at Cout 24, 88 and 136 actually are, and
it changes 6.2's padding decision: the producer's atoms were never the
problem, the consumer's 16-lane programming is. Whether a conv can be
programmed at `Cin = 24` rather than 32 without reading the padding
surface is the hardware question that decision now turns on.

### 6.2 Layout assignment in the compiler

Once each dispatch operand and result can carry an encoding, assignment is a
walk over a graph the compiler already has. `rocket-mark-dense-readers`
runs at the flow phase after `rocket-pin-unclaimed-dispatches`, when every
reader of a dispatch result is a final SSA use; the same walk assigns
layouts:

- an edge from a Rocket dispatch to a Rocket dispatch whose cube geometries
  agree is packed on both sides -- no pack, no compact, no runtime check;
- an edge into a Rocket dispatch from anything else (a CPU dispatch, a
  function argument, a constant) is packed on the consumer side only: a
  pack happens here and nowhere else;
- an edge out of a Rocket dispatch to anything else (a CPU dispatch,
  `util.return`, a tied operand, a copy) is dense on the consumer side: a
  compact happens here;
- an edge between two Rocket dispatches whose geometries disagree (the PPU
  rounding, a channel count that is not whole-atom) is the decision point
  the runtime does not have today: pad the producer's programmed width so
  they agree, or keep it dense and *say so in the audit*.

The result is written into the executable, per binding: `packed_in`,
`packed_out` (or an encoding id) on `Conv2DDef` and its twins, replacing
`runtime_dense_readers` -- a boolean is enough once the packed tensor is an
IREE buffer, because the reason the count existed was the driver's
inability to see past its command buffer. The driver's `chainable_cube`
becomes a check: an input declared packed whose producer cube is missing or
mismatched is an `INTERNAL` failure, not a fallback to repacking, exactly
as section 2 treats a shape the planner refused.

Two choices are deliberately deferred, and each has a cheaper first form:

- **Who allocates the packed buffer.** First form: the packed tensor stays
  in driver scratch and the compiler-assigned layout is honoured only within
  a command buffer, so the driver still falls back to a dense write when an
  edge crosses one. That is bit-identical to today with the guesswork
  replaced by a declaration, and it is enough to remove both environment
  variables. Second form: the dispatch's IREE result buffer *is* the cube,
  sized by the encoding, which is what lets the layout cross command buffers
  and what makes the fan-out and multicore questions above unavoidable.
  IREE's encoding attribute interface (a storage-size hook on the tensor
  type) is the mechanism to evaluate for the second form; do not assume it
  composes with the plugin's transform-spec pipeline until a lit test shows
  it does.
- **Who executes the pack at a CPU boundary.** First form: the driver, as
  today -- it is one memcpy-shaped transform per boundary and P8 has
  measured that a standalone dispatch for it would cost more than it saves.
  Second form: an explicit pack op in IR, which is only worth building if it
  can fuse into the adjacent CPU dispatch (the widen/narrow it replaces
  already does), and which ROADMAP.md's closing section was right to refuse
  as a *standalone* dispatch.

The pipeline shape section 2 established applies: nothing widens. An edge
the compiler cannot prove packable is dense, the audit prints why, and
`--strict-offload` gains a layout mode that fails on any NPU -> NPU edge
left dense.

### 6.3 Weights packed at compile time

The packers are `pack_hwcf_to_rocket_weights`, `_padded`, `_int4`,
`_affine_i8` and `pack_depthwise_to_rocket_weights` in
`iree-rocket-hal/src/rocket/tensor_layout.rs`, with the nesting
`output_block -> input_group -> filter_y -> filter_x -> output_lane ->
input_lane` and the per-rung padding rules in `rocket_weight_storage_size`.
They depend on the logical filter shape, the programmed (padded) channel
counts and the rung -- all compile-time facts for a static model -- and on
nothing the runtime knows that the compiler does not.

**The compiler already controls the bytes in that binding.** What IREE
hands the driver is not the model's weight; it is a derived constant. The
ONNX filter passes through `convert-conv-to-channels-last` (to HWCF),
`rocket-demote-conv-inputs` (f32 -> f16) and, for depthwise, the shim's own
HWC -> CHW `linalg.transpose`, and IREE's const-eval folds the whole chain
into a `util.global` initializer -- the spec relies on it by name ("it is
the layout `pack_depthwise_to_rocket_weights` reads and it const-evals away
over a constant filter"). The CNA packing is one more link in a chain that
exists, not a new mechanism. `RocketTarget.cpp` is *not* where it happens:
the serializer emits executable definitions, and the constants are IREE's
rodata, which it never touches.

P8's objection to doing this at compile time was a second implementation
of the blocked layout, "the kind of duplication that produced the depthwise
tap-major bug". That objection is right and it chooses between the two
routes that exist:

1. **Express the packing as linalg in the shim's caller** -- pad `Cin` to
   32-lane groups and `Cout` to 16- or 32-lane blocks, `expand_shape`,
   `transpose` to `block, group, ky, kx, out_lane, in_lane`,
   `collapse_shape` -- and let const-eval fold it as it folds the demote.
   Hoisting is free and a non-constant weight degrades to a CPU dispatch
   instead of failing. But it is a second spelling of the layout unless the
   plugin *generates* the ops from `rocket-core`'s layout parameters, and it
   cannot express int4 nibble packing.
2. **A flow-phase pass that calls the packer over FFI.** By the flow phase
   the filter operand is a `util.global.load` of an immutable initialized
   global; the pass rewrites the initializer through `rocket-core`'s packer
   and sets `weights_packed` on the def. `rocket-mark-dense-readers` already
   runs there by name, so the slot exists; a non-constant weight keeps the
   flag unset and the runtime packer.

Take route 2: **move, do not copy.** The packers go into `rocket-core` (the
same `layout.rs`; their one import, `AccumulatorOutputTile`, is already
core data) and `rocket-plan-ffi` exposes one entry point per rung. The
driver, seeing the flag, binds the weight buffer directly and skips
`weight_cache`; an executable without the flag packs as before, so old
`.vmfb`s keep working. Two recorded traps apply verbatim: nothing may be
placed inside the never-inlined `@call_*` wrappers, or it becomes a
per-inference CPU dispatch (ISSUES.md P6 item 2, the bug the demote hit);
and `pack_hwcf_to_rocket_weights_affine_i8` fills padding lanes with each
channel's weight zero point, not zero, so any route-1 `tensor.pad` would be
per-channel on that rung.

Two things fall out for free. The wire's one scalar `weights_zero_point`,
broadcast across every channel by the driver (ROADMAP.md's ledger), stops
being a limitation: `pack_hwcf_to_rocket_weights_affine_i8` takes one zero
point per `Cout`, and a compile-time packer can hand it the per-channel
vector the model actually has. And the driver's "nothing is about to write
this buffer before the dispatch runs" rule, `WeightPacking`'s reason for
deferred packing, has no compile-time counterpart to maintain.

What it buys, stated the way P8 sized it: about 68 of the ~86 ms
first-inference penalty and 7.5 MiB of driver cache on MobileNetV2 int8,
nothing in steady state, and a `.vmfb` that grows by the group padding (16
or 32 lanes per block, 32 per input group). It is worth doing for
first-inference latency and because it is item 3 of the end state; it is
not a benchmark lever and should not be measured as one.

**Landed 2026-09-10, by route 2.** The packers moved -- not copied -- into
`rocket_core::weights`, and the one new thing there is `WeightPlan`: the
packer *selection* the driver's `apply_ops` used to make inline (tap-major
for depthwise, the affine int8 packer when the rung carries a zero point,
the plain one otherwise), as a value both sides build from the shape. The
driver's `WeightPacking` carries a `WeightPlan` and `apply_ops` calls
`plan.pack`; `rocket-plan-ffi` is ABI 5 with `rocket_pack_conv_weights` and
`rocket_pack_matmul_weights` over the same plan. The flow-phase pass is
`rocket-pack-weights`, after `rocket-mark-dense-readers`: it follows the
weights operand through the shim's `flow.tensor.reshape` to its
`util.global.load immutable`, packs the initializer, stores the stream in a
new i8 global, and retargets the dispatch at a clone of its executable whose
config carries `weights_packed = true`. That flag is an executable property
(`Conv2DDef.weights_packed`, `MatmulDef.weights_packed`; schema appended,
older `.vmfb`s read as false), which is why it needs no push constant, no
pipeline-layout change and no edit to the 54 shim dispatch sites -- the
first design sketch above had it as a def field for that reason. The driver
binds a flagged executable's weights binding directly after checking it is
at least `WeightPlan::packed_bytes` long; the runtime packer and
`weight_cache` are untouched by an unflagged one. A dispatch whose filter is
not a constant, or whose dimensions are not, is left to the runtime packer
and counted in the pass's remark.

Acceptance, planck 2026-09-10, fp16 (P8's int8 model has no MLIR in the
tree; the survey's fp16 ones do). MobileNetV2: 36 of 36 weight-bearing
dispatches packed, 6,811,072 -> 6,818,560 coefficient bytes, `.vmfb`
+7.5 KB (+0.1%), `pack.weights` (39.8 ms) and the 6.5 MiB weight cache
gone, cold single-inference wall 184-195 -> 146-149 ms. Wide ResNet50: 53
of 53 packed, coefficient bytes unchanged (every channel count fills its
padding unit), `.vmfb` 15 bytes smaller, `pack.weights` (137 MB at 168 MB/s,
815 ms) gone, cold wall 1290 -> 551 ms. Outputs bit-identical packed
against unpacked, and the unpacked `.vmfb` bit-identical on the runtime
before and after. The one cost that appeared: the packed binding is
cache-synced as an ordinary input (`sync.inputs` 0.3 -> 5.5 ms on
MobileNetV2), where the packer's scratch never was -- P3's whole-BO sync,
now on the weights too. Four constant-filter fixtures
(`fp16_conv_packed_weights`, `fp16_conv3x3_packed_weights`,
`requant_int8_packed_weights`, `depthwise_fp16_packed_weights`) gate the
direct-bind path in `tools/e2e_conv_regression.py`, and the gate refuses to
run if any of them stops dispatching at a `_packed` executable.

Not done, and worth saying: the fp16 bias is still packed at dispatch
(`pack.bias`, 0.2 ms on MobileNetV2 -- not worth a pass), and the
per-channel weight zero point the affine packer can take is still the
wire's one scalar, broadcast.

### 6.4 What the whole section is worth

Honestly: little on the models this repository benchmarks, and that is
fine, because the section is about what the compiler *knows*, not what the
board does per iteration. ResNet50 and VGG already chain nearly every edge
at runtime, and the compile-time form is bit-identical to them by
construction. What changes is:

- **First inference**: ~68 ms on MobileNetV2 int8 (6.3); proportionally
  more on models with more weight bytes.
- **The edges the runtime identity cannot take**: MobileNetV2's 8
  whole-atom declines become a compile-time padding decision; a reader on a
  later command buffer becomes chainable under the second form of 6.2.
  Neither is sized yet, and MobileNetV2's own verdict (P7) says its offload
  loses for other reasons, so do not expect either to move a headline.
- **The audit** can finally say, per edge, packed or dense and why -- the
  layout half of section 3, which today has no vocabulary for it.
- **Two runtime heuristics and two environment variables retire**, and a
  `.vmfb` describes its own layout the way it already describes its own
  placement.

The whole-atom rule deserves one caution. A producer leaves its padding
lanes as the hardware wrote them, and a consumer's repack zeroes them; the
identity fails when the two differ. Whether the DPU writes zeros or stale
bytes into padding lanes is a hardware question, and if it is stale bytes
then fp16 garbage can be NaN, which a zero weight does not neutralise. A
compile-time contract for a non-whole-atom edge therefore needs a hardware
measurement first (`Selectors`-pattern producer, read the padding lanes),
not a compiler rule. Until then the compiler's answer for such an edge is
dense, reported.

### 6.5 Coupling

- **DYNAMIC_SHAPES.md.** Scope is static extents only: an encoding fixes a
  physical size, and a symbolic channel count has none. DS1's "a symbolic
  model offloads nothing" already keeps such an op off the NPU, so nothing
  here needs a dynamic story yet; state that, do not build one.
- **Section 4.** The serialized plan and the layout encoding are the same
  kind of thing -- a compile-time decision the executable carries and the
  runtime validates -- and should share a version field and a target/policy
  identity. Whichever lands first defines the versioning; the other reuses
  it.
- **MULTICORE.md M2 and P1.** See 6.1: fan-out and packed IREE buffers are
  in tension, and the second form of 6.2 cannot be designed without settling
  which context owns a packed tensor.
- **The chain gate.** `fp16_conv_pool_conv_chain`, `fp16_matmul_chain`,
  `requant_int8_chain` and `requant_int8_chain_cpu_between` are the
  acceptance fixtures for every step below, unchanged: each must stay
  bit-exact with the middle compaction skipped, and the last must keep its
  dense write.

### 6.6 Order

1. **The contract (6.1).** `rocket-core::layout`, ABI v4, driver computes
   `OutputCube` through it. No behaviour change; `chain_identity_tests`
   gain the pure verdict beside each byte-level case (they cannot move: the
   bytes need the HAL's packer). Bit-identical on every model. **Landed
   2026-09-10**; see 6.1.
2. **Weights (6.3).** Independent of the rest and the only step with a
   number attached. Acceptance: packed bytes byte-identical to the driver
   packer on every conv fixture in `tools/e2e_conv_regression.py`, the
   first-inference latency before and after, the `.vmfb` size delta.
   **Landed 2026-09-10**; see 6.3. Byte identity is by construction (one
   `WeightPlan` on both sides) and the fixtures gate the bound path.
3. **Declared layout, first form (6.2).** Encoding assigned in the compiler,
   written per binding, driver checks instead of guesses, within a command
   buffer. `ROCKET_CHAIN` and `ROCKET_LAZY_COMPACT` are removed, and
   `runtime_dense_readers` with them. Acceptance: bit-identical to the
   current chain on every model in the survey harness; the audit prints
   every edge; `--strict-offload` layout mode on ResNet50 passes.
4. **Packed IREE buffers, second form (6.2)**, after the multicore
   ownership decision. Acceptance: a later-command-buffer reader chains; the
   count is gone from the wire.

Steps 1 and 2 are small and can go first without prejudice to anything in
sections 4 and 5. Step 3 is the one that delivers the end state's items 2
and 4 for a single command buffer, which on every model measured here is
the whole model.

## Validation and delivery order

| Milestone | Required evidence |
|---|---|
| A: core extraction and fallible APIs | Host boundary/refusal tests; existing vendor fixtures and register hashes unchanged; checked overflow and malformed-descriptor tests. |
| B: compiler planning and explanations | MLIR tests for accepted/tiled/rejected candidates, unsupported semantics, dynamic policy, strict mode, provenance, and `--no-offload`; ABI ownership/error tests. |
| C: serialized static plans | Schema compatibility and malformed-plan tests; compile/runtime plan equivalence; no repeated search for static dispatches. |
| D: expanded spatial/M and N tiling | Tail/halo/coverage and scratch tests; CPU-reference comparisons and RK3588 execution on both sides of each newly admitted boundary. |
| E: reduction tiling and tuning | Nonzero initial accumulators, bias/activation/quantization tests; precision/overflow tests; full-model correctness and end-to-end performance measurements. |
| F: layout contract and prepacked weights (6.1, 6.3) | **Met 2026-09-10.** `OutputCube` computed through `rocket-core::layout` with the driver bit-identical on every surveyed model; packed weight bytes byte-identical to the driver packer on every conv fixture; first-inference latency and `.vmfb` size before and after (6.3). |
| G: compiler-declared layout (6.2) | Per-binding encoding on the wire; driver checks and never repacks a declared-packed input; the chain fixtures bit-exact with `ROCKET_CHAIN`/`ROCKET_LAZY_COMPACT` removed; the audit names every NPU -> NPU edge packed or dense with a reason; `--strict-offload` layout mode. |

Use addressing-sensitive dense/selector inputs, odd channels, multi-surface
outputs, stride and padding cases, and adjacent NPU producers/consumers.
Check no gaps or duplicate logical output writes and no out-of-bounds physical
accesses. Follow the repository's board-test protocol, including one shape per
process where required. Reuse the existing HAL conv/FC fixtures, plugin boundary
tests, and `tools/e2e_conv_regression.py` before adding new test infrastructure.

Benchmark representative large conv and transformer matmul shapes against
`rocket-compiler --no-offload` under the same pipeline and core allocation.
Record correctness, total latency, host packing/compaction, hardware jobs,
scratch peak, executable size, and compile time. Expand default admission only
with supporting evidence; keep speculative tiling policies opt-in.

The first implementation PR should deliver milestone A without widening any
limits. The next should connect the planner to compiler selection and reporting.
That provides useful diagnostics early and establishes a trustworthy foundation
for compile-time plan serialization and new tiling strategies.
