# Symbolic tensor dimensions

What it would take to compile a model whose tensor dimensions are symbolic
(`tensor<1x?x?x?xf16>`) through `rocket-compiler` and run it under
`iree-run-module` with the extents supplied at invocation time. Written
2026-09-07 against `main` at `965708a`, which was already three commits behind
`main` when it was written. Every citation below was re-checked on 2026-09-07
against `main` at `68f281c` and is given for *that* tree; the review that
produced the corrections is recorded inline.

Read alongside the other three. [LIMITS.md](LIMITS.md) says what the stack is
*measured* to do and which layer enforces each bound; [ISSUES.md](ISSUES.md)
carries the open defects; [ROADMAP.md](ROADMAP.md) says which operations the
hardware could run but the compiler cannot reach. This file is a fourth
question that cuts across all three: not *which* op, but whether an op's
**extents** have to be known when the `.vmfb` is written.

Evidence tags follow ISSUES.md: **[verified]** = read in this tree during this
review; **[hypothesis]** = inference not re-measured. Severity follows it too:
**S1** wrong results reach a user, **S2** wrong results reach a developer or a
measurement, **S3** performance, **S4** hygiene.

The headline is that this is **not** a from-scratch feature. The wire format,
the serializer, the transform spec's dispatch helpers and the entire runtime
were built around runtime-supplied extents already. Every gap below is in the
compiler's *matchers* and *guards*, not in the plumbing they feed.

---

## What already works

**The wire format** [verified]. Every kernel def in
[`rocket_executable_def.fbs`](rocket-schema/schema/rocket_executable_def.fbs)
carries a `runtime_dimensions` vector: an ordered map from dispatch
push-constant ordinal to a shape field, where a listed field must be `0` in
the executable template and is replaced before validation. `Conv2DDef` adds
`runtime_quantization` on the same contract. Output extents are deliberately
*not* settable -- `Conv2DDimension` values 3 and 4 are retired -- because the
runtime derives them, so no dispatch can state an output shape the register
program was not built for.

**The serializer** [verified]. [`RocketTarget.cpp:415`](rocket-compiler-plugin/target/Rocket/RocketTarget.cpp:415)
parses `runtime_dimensions` off the `#hal.executable.target` config for conv,
and again at `:732` (pooling), `:840` (element-wise, in the shared
`parseElementwiseRuntimeDimensions` helper the three element-wise builders
call) and `:1199` (matmul). It enforces the template contract in both
directions: a listed dimension whose
template value is nonzero is an error, and so is a zero dimension that is not
listed.

**The transform spec's dispatch helpers** [verified].
[`@call_rocket_dynamic_conv2d`](rocket-compiler-plugin/target/Rocket/rocket_conv2d_transform_spec.mlir:3279)
is *already written against* `tensor<1x?x?x?xf16>`. It reads the six settable
extents with `tensor.dim`, `arith.index_cast`s them to `i32`, and passes them
as the push constants `#dynamic_pipeline_layout` declares (`constants = 6`).
Its f32 accumulate epilogue is a properly dynamic `flow.dispatch.workgroups`
with a three-dimension workload and `iree_tensor_ext.dispatch.workload.ordinal`
plumbing. The same is true of every other helper: `@call_rocket_matmul` takes
`tensor<?x?xf16>` with no static dimension at all, and the pooling and
element-wise helpers take `tensor<1x?x?x?xf32>` / `tensor<1x?x?xf32>`.

**The runtime** [verified].
[`Conv2dExecutable::resolve_shape`](rocket-hal-driver/src/executable.rs:237)
substitutes the push constants into the template per dispatch, derives the
multiplier from a runtime output scale when one is supplied, and runs
`validate_conv_shape` on the result. `PoolingExecutable::resolve_shape`
(`:446`) does the same and derives the output extents with `floor_output_extent`.
Register commands, NC1HWC2 storage sizing and the NHWC pack all happen per
dispatch in `command_buffer.rs` against the *resolved* shape, not a compiled-in
one.

**The weight cache does not care** [verified]. `weight_cache::Geometry`
(`rocket-hal-driver/src/weight_cache.rs:106`) keys on filter height/width,
channel counts, element size, depthwise, padded channels, zero point and
scratch length -- **no spatial extent**. A model whose H/W vary per invocation
therefore still hits the cache from the second inference onward.

So: the runtime and the wire format need approximately nothing. Everything
below is compiler-side.

---

## DS1 (S2) — `dim_bounds` cannot prove a bound on an unbounded `?`, so a symbolic model offloads *nothing*, silently

This is the blocker, and its failure mode is the bad one: the compile
succeeds, the `.vmfb` runs, and every convolution is on the CPU with no
diagnostic anywhere.

Every matcher in the `foreach_match` list constrains at least one dimension
with `transform.iree.match.dim_bounds`. That is not incidental -- `spec.rs`
depends on it, and `spec::neutralize` *refuses* a spec containing a matcher
with no `dim_bounds` at all, because such a matcher would still fire and
quietly corrupt the `--no-offload` baseline.

`MatchDimBoundsOp::matchValue`
(`iree-build/iree-src/compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp:656`)
[verified] resolves both ends through
`ValueBoundsConstraintSet::computeConstantBound`:

```cpp
auto constantUb = ValueBoundsConstraintSet::computeConstantBound(
    presburger::BoundType::UB, {current, /*dim=*/dim},
    /*stopCondition=*/nullptr, ValueBoundsOptions{/*closedUB=*/true});
if (failed(constantUb)) {
  return emitSilenceableError()
         << "failed to compute constant upper bound for dim " << dim;
}
```

For a dynamic dimension with no constraining producer that call fails. The
result is a *silenceable* error, so `foreach_match` treats it as "this matcher
declines" rather than as a compile failure -- which is exactly why nothing is
reported.

`@match_dynamic_conv2d` alone carries two of these
(`transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 3584` on
Cin at spec `:5923` and the same on the filter's Cout at `:5935`), and the
strided, 3x3, depthwise and int8 variants each carry their own.

### The mechanism to fix it exists, but is not wired end to end

`util.assume.int` has a `ValueBoundsOpInterface` external model --
`UtilAssumeIntValueBoundsOpInterface` in
`compiler/src/iree/compiler/ExternalInterfaces/UtilExternalModels.cpp:487`
[verified] -- which feeds `getUnionedUnsignedRange` straight into
`cstr.bound(result) >= *min` / `<= *max`. So an assumed `umin`/`umax` on a
dimension *value* is visible to the constraint set.

What is not established is the chain from that index value to the *tensor
value's* dim, which is what `dim_bounds` asks about. Grepping
`compiler/src/iree/compiler/` for `ValueBoundsOpInterface` implementations
returns only the Util (`util.assume.int`), Codegen
(`LoadFromBufferOp`, `Codegen/ExternalInterfaces/UtilExternalModels.cpp:21`),
TensorExt (`DispatchTensorLoadOp`, `DispatchWorkloadOrdinalOp`) and HAL
(workgroup ID/count) external models [verified] -- **nothing for
`flow.tensor.tie_shape`**, which is the op that normally ties an assumed dim
back onto a dynamically-shaped tensor at this stage of the pipeline.

Two ways out, and they are genuinely different amounts of work:

1. **Register a `ValueBoundsOpInterface` external model for
   `flow.tensor.tie_shape`** (upstream, or in the Rocket plugin), so
   `assume.int` bounds carried by an imported model reach the matchers
   unchanged. Then the frontend has to actually emit them -- a torch export
   with `dynamic_shapes` bounds, or hand-written `util.assume.int` on the
   entry function's dims.
2. **Re-spell the matchers to constrain `tensor.dim` results** instead of
   tensor values, which `assume.int` bounds already reach. Cheaper, but it
   touches every matcher and loses `dim_bounds`' tensor-dim addressing, so
   `spec::neutralize`'s invariant has to be restated in whatever the new
   spelling is.

Either way the acceptance test is a lit test asserting that a `?`-shaped conv
*does* match -- absence of offload is the thing that has to fail loudly.

---

## DS2 (S3, large) — the batch dimension must be statically 1, and the wire format has no batch at all

`transform.iree.match.convolution` builds its dimension params from
`linalgOp.getStaticLoopRanges()`
(`PreprocessingExtensions.cpp:562`) [verified], so a dynamic loop range
arrives as `ShapedType::kDynamic` wrapped in an `I64IntegerAttr`.
`MatchDimsEqualOp` compares with `rhs == -1 || lhs == rhs` (`:614`), which
means:

- `dims_equal %out_img, [-1, -1]`, `%out_ch, [-1]`, `%in_ch, [-1]` — the `-1`
  is a wildcard and passes a dynamic dimension. **H, W, Cin and Cout are
  already free.**
- `dims_equal %batch, [1]` — fails outright for a symbolic batch.

That is not just a matcher limit. `Conv2DDef` has no batch field, `conv::Shape`
has no batch concept, and every `call_rocket_*` helper spells its leading
extent as a literal `1`. Making N symbolic means a new wire field or a
per-batch dispatch loop, a new `conv::Shape` axis or an N-into-H folding, new
matchers, and new helper signatures -- with the same "is this hardware-verified"
bar every other wire field in this repo has had to clear.

`@call_rocket_matmul` is the exception worth noting: it is `tensor<?x?xf16>`
with M, K and N all runtime dimensions [verified], so the matmul path has no
batch problem to solve.

Recommendation: treat symbolic batch as out of scope for the first pass. A
symbolic *spatial and channel* model with `N == 1` is the useful 90%, and it
needs none of this.

---

## DS3 (S1) — the hardware envelope moves from compile time to dispatch time, and one bound is checked at neither -- already, today

Today the *channel* ceilings are compile-time matcher facts.
`@match_dynamic_conv2d` bounds Cin and Cout at 3584 (matching
`conv::MAX_INPUT_CHANNELS` / `MAX_OUTPUT_CHANNELS`,
`iree-rocket-hal/src/rocket/conv.rs:297` and `:495`); the 3x3 matcher keeps a
separate Cin 1152 / Cout 1792 (spec `:6013` and `:6014`); stride lives in the
executable variant. A model outside *those* bounds does not match and runs on
the CPU -- correct, just slower.

The spatial envelope is a different story, and it is the one that matters
below: the dense conv matchers carry `dim_bounds` on Cin and Cout only, and no
bound on H or W at all [verified] -- every `%input_value[1]`/`[2]` bound in the
spec belongs to a pooling or NCHW-depthwise matcher. So for spatial extent the
"move to dispatch time" this section describes has *already happened*, for
static models, and the rest of this section is a statement about the stack as
it stands rather than a consequence of symbolic shapes.

With symbolic extents there is nothing left to check at compile time. The check
lands at dispatch, in
[`validate_conv_shape`](iree-rocket-hal/src/rocket/executable_format.rs:341)
[verified], which trial-plans the shape under `catch_unwind` and turns a
planner panic into `Err`. The driver's response is terminal:

```rust
let (resolved_shape, kernels) = match executable.resolve_shape(constants) {
    Ok(resolved) => resolved,
    Err(_) => {
        return status::from_code(
            crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
        );
    }
};
```

(`rocket-hal-driver/src/command_buffer.rs:1929`) [verified]. There is **no
runtime CPU fallback**. An out-of-envelope invocation is a hard
`iree-run-module` failure, not a slow success.

### The part that was S1 -- measured and closed 2026-09-08

This section used to carry a live S1: `@match_dynamic_conv2d_3x3`'s own
comment recorded `Cin=256`/`Cout=256`/3x3 at 26x26 through 48x48 as
deterministically all-zero on an 11/1 CBUF split, the `Cout<=256` rule that
comment reasoned about had since been widened to 1792, and the evidence files
it cited were gone from the tree -- so the shape was inside the bounds in
force with static extents and nothing tracked it.

Re-measured on `planck` (ISSUES.md **C12**): the current planner grants the
shape a 7/5 split at every extent from 20 to 58 and every one is exact at
fp16 and int8. Forcing the old 11/1 back is a watchdog kill with the output
unwritten, not a silent completion, so the original "all-zero" was a killed
job read as a result before C3 existed -- the same class as C9's large-kernel
cliff, a starved coefficient grant. `validate_conv_shape` never needed a
refusal for it: `streamed_weight_bank_preference` is what keeps the grant
honest, `dense_k3_plan_never_starves_the_streamed_coefficient_working_set`
pins it, and no wire field can force a split. What survives for symbolic
shapes is only the general point above: the spatial envelope is checked at
dispatch time and an out-of-envelope invocation is a hard error.

### And a policy decision

Once the check is honest, decide what an out-of-envelope dispatch should do:

- **Hard error** (today's behaviour). Simple, and consistent with "wrong
  numbers are worse than no numbers", but it means a model's *input data* can
  fail an inference that compiled cleanly.
- **Runtime fallback**. Needs a CPU path the driver can reach at dispatch
  time, which does not exist and is a substantial piece of work.
- **Carry the envelope in the assumptions.** If DS1 is solved via
  `util.assume.int`, the assumed `umax` can be made to reproduce the matcher's
  guarantee exactly -- the model then either declares itself in-envelope at
  compile time or does not match at all, and the runtime check stays a
  backstop. This is the cheapest correct answer and it falls out of DS1 for
  free.

---

## DS4 (S2) — `RocketVerifyConvShapesPass` goes blind exactly when shapes are symbolic

The pass exists to catch an earlier pass having rewritten a convolution into a
different one -- concretely, the
`iree-global-opt-demote-contraction-inputs` bug documented at length in
[`RocketDemoteConvInputsPass.cpp`](rocket-compiler-plugin/target/Rocket/RocketDemoteConvInputsPass.cpp:16),
which silently dropped `strides`/`dilations` and turned MobileNetV2's stride-2
stem conv into a stride-1 one reading a 114x114 corner of its input. Its own
header calls silently returning wrong numbers "the outcome most worth
preventing".

It works by re-deriving the output extent from the input, filter, stride and
dilation and comparing. So:

```cpp
if (ShapedType::isDynamic(input) || ShapedType::isDynamic(filter) ||
    ShapedType::isDynamic(output)) {
  continue;
}
```

([`RocketVerifyConvShapesPass.cpp:110`](rocket-compiler-plugin/target/Rocket/RocketVerifyConvShapesPass.cpp:110))
[verified] -- and the guard is gone for every op it was written to protect.

The check has to be restated in a shape-independent form: verify that the
rebuilt op still carries the `strides`, `dilations`, `indexing_maps` and `cast`
attributes of the op it replaced, rather than verifying an arithmetic
consequence of them. That is arguably the better check anyway, since it names
the actual defect instead of one of its symptoms.

---

## DS5 (S4) — the compile-time audit gets weaker, and `--no-offload` needs re-checking

Two smaller consequences of moving the decision to runtime:

- **`rocket-compiler audit`** reports executable and dispatch-site counts. With
  symbolic extents those counts stay meaningful (a site is a site), but "did
  this shape offload" is no longer answerable from the module alone. Worth
  saying so in the audit output rather than letting the number be read the old
  way.
- **`spec::neutralize`** rewrites every `dim_bounds` interval to
  `umin = umax = 999999` to build the CPU baseline. If DS1 is solved by
  re-spelling the matchers away from `dim_bounds` (option 2), that mechanism
  stops working and `--no-offload` silently starts offloading -- the precise
  failure ISSUES.md M4 exists to prevent. `neutralize` already refuses a
  matcher with no `dim_bounds`, so it would fail loudly rather than lie
  [verified] -- but it would fail, and the neutralization has to be ported to
  whatever the new spelling is. Option 1 leaves it untouched.

---

## DS6 (S4) — things that are *not* problems, recorded so they are not re-litigated

- **Stride and padding** are per-executable variants (`#rocket_dynamic_target_s2`
  and friends), but both come from static op attributes even in a symbolic
  model, so the existing s1/s2/s3/s4 fan-out keeps working unchanged
  [verified].
- **`RocketExpandGemvToMatmulPass`** already handles a dynamic vector extent:
  it emits a `tensor.DimOp` when `sourceType.isDynamicDim(0)` and an index
  attribute otherwise
  ([`RocketExpandGemvToMatmulPass.cpp:70`](rocket-compiler-plugin/target/Rocket/RocketExpandGemvToMatmulPass.cpp:70))
  [verified].
- **`pin_unclaimed_dispatches`** runs at the `flow` phase and pins everything
  the spec did not claim to the CPU device. The extra shape-arithmetic
  dispatches a symbolic model produces are exactly "unclaimed", so they land on
  the CPU, which is correct [verified].
- **`iree-run-module` needs no change.** Extents are supplied the usual way,
  `--input=1x224x224x3xf16=@input.bin`. What has to change is upstream of the
  compiler: the model must be *imported* with dynamic dims in the entry
  signature (`iree-import-onnx` on a model with symbolic dim params, or a torch
  export with `dynamic_shapes`).
- **`RocketExpandOnnxConvIntegerPass`** handles a dynamic kernel extent by
  falling back to the `torch.onnx.kernel_shape` attribute, and warns out only
  when that attribute is missing or the wrong length
  ([`:204`](rocket-compiler-plugin/target/Rocket/RocketExpandOnnxConvIntegerPass.cpp:204))
  [verified] -- a narrower refusal than "declines a dynamic kernel", and better
  news for a symbolic import, since ONNX carries `kernel_shape` on the op.
  Weights are constants in every model measured here either way, so this is not
  a gap.

---

## Recommended order

1. ~~**DS3's silent-zero refusal first**~~ -- closed 2026-09-08 without a
   refusal: re-measured exact at every extent under the current planner's
   grant, and the original observation was a watchdog-killed job (ISSUES.md
   C12). Nothing on this list is S1 any more.
2. **DS1**, preferring the `tie_shape` external model (option 1) over
   re-spelling the matchers, because it leaves `spec::neutralize` and the
   `--no-offload` baseline intact (DS5). Acceptance is a lit test that a
   `?`-shaped conv matches, plus an end-to-end run on the board.
3. **DS3's policy decision**, which falls out of (2): make the assumed `umax`
   reproduce the matcher envelope so the compile-time decision stays
   compile-time.
4. **DS4**, restating the verify pass on attributes rather than extents.
5. **DS5**, whichever half (1) turned out to need.
6. **DS2 last, separately.** It is a wire-format change and a `conv::Shape`
   change, and `N == 1` covers every model this repo currently measures.

---

## What this file does not claim

That symbolic shapes are *worth* it. [ISSUES.md P8](ISSUES.md) `:554` measured
the offload's cost as a flat per-dispatch tax *that parallelises*, and its
constant has moved since: on a full machine the fp16 dense models beat their
own `--no-offload` baselines, so "more sites is slower" is the starved-core
int8 reading, not a general law. What survives for this file is the weaker
claim, which is enough: nothing here changes a single dispatch's cost -- a
symbolic model runs the same register program the static one does. The case
for this work is **coverage**: serving a model whose input resolution or
sequence length is not fixed at compile time, without recompiling per shape. Read ROADMAP.md's opening warning
before treating it as a throughput proposal.
