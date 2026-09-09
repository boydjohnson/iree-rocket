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

There are two separate deliveries:

1. Compute the existing HAL tile plans at compile time and explain placement.
2. Add compiler transformations for operations larger than the existing
   logical-shape limits, including output-channel tiling and eventually
   reduction tiling.

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

Still open here: the strict-offload option, the cost policy above, and the
dynamic-shape policy.

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

## Validation and delivery order

| Milestone | Required evidence |
|---|---|
| A: core extraction and fallible APIs | Host boundary/refusal tests; existing vendor fixtures and register hashes unchanged; checked overflow and malformed-descriptor tests. |
| B: compiler planning and explanations | MLIR tests for accepted/tiled/rejected candidates, unsupported semantics, dynamic policy, strict mode, provenance, and `--no-offload`; ABI ownership/error tests. |
| C: serialized static plans | Schema compatibility and malformed-plan tests; compile/runtime plan equivalence; no repeated search for static dispatches. |
| D: expanded spatial/M and N tiling | Tail/halo/coverage and scratch tests; CPU-reference comparisons and RK3588 execution on both sides of each newly admitted boundary. |
| E: reduction tiling and tuning | Nonzero initial accumulators, bias/activation/quantization tests; precision/overflow tests; full-model correctness and end-to-end performance measurements. |

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
