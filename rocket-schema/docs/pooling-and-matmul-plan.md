# Pooling and matmul: what Linalg gives us, and what the schema needs

Plan of record for branch `schema-pooling-and-matmul`. Written 2026-09-04
against `mlir-linalg-ops.md` (LLVM `59824b6`, the revision IREE vendors) in
`../iree-rocket-design-spike`, the HAL's `pooling.rs` and `fc.rs`, and the
actual lowering of both MobileNetV2 models in this repo.

Two gaps, and they are different in kind:

* **Pooling** exists in the HAL (`PoolingShape`, `PoolingPlan`, Avg/Max/Min)
  and in the runtime (`UkernelShape::Pooling`, a dispatch path in
  `command_buffer.rs`), and is unreachable because `KernelDef` has no
  `PoolingDef`. The only way to build one today is legacy tag `1`, a
  hardcoded 4x4 test shape that is not hardware-validated. So this is
  *plumbing a built engine through the wire format*.
* **Matmul** is the reverse. `FullyConnectedDef` is in the schema, the
  runtime decodes it, and `RocketTarget.cpp` accepts `kernel =
  "fully_connected"` -- but **nothing ever emits it**, because there is no FC
  op in Linalg to match. `grep` finds no FC matcher in the transform spec.
  So this is *retargeting a wire format at the op that actually arrives*.

## What the models actually produce

Both models, lowered with the repo's own compiler
(`iree-build/build/tools/iree-compile --compile-to=preprocessing`, which is
the stage the transform spec runs at):

| | mnv2.fp16 | mobilenetv2.static-int8 |
| --- | --- | --- |
| `linalg.conv_2d_nchw_fchw` | 35 | 1 (+34 `_q`) |
| `linalg.depthwise_conv_2d_nchw_chw` | 17 | 4 (+13 `nhwc_hwc_q`) |
| `linalg.pooling_nchw_sum` | **1** | **1** |
| `linalg.matmul` | **1** | **1** |

The tail of both models is identical and **unquantized f32**, even in the
"static-int8" one:

```mlir
%119 = tensor.empty() : tensor<7x7xf32>                       // window: shape only
%120 = linalg.pooling_nchw_sum {dilations = 1, strides = 1}
         ins(%116, %119 : tensor<1x1792x7x7xf32>, tensor<7x7xf32>)
         outs(%118 : tensor<1x1792x1x1xf32>)                  // zero-filled init
%121 = linalg.generic ... arith.divf %in, %cst_108            // divide by 49
%collapsed = tensor.collapse_shape %121 ...                   // 1x1792x1x1 -> 1x1792
%transposed = linalg.transpose ins(%cst_89 : tensor<1001x1792xf32>) permutation = [1, 0]
%125 = linalg.matmul ins(%collapsed, %transposed : tensor<1x1792xf32>, tensor<1792x1001xf32>)
         outs(%124 : tensor<1x1001xf32>)                      // zero-filled init
%126 = linalg.generic ... arith.addf %in, %in_369             // bias add
```

That is the whole target surface for this branch: one 7x7 global average
pool over 1792 channels, and one `[1,1792] x [1792,1001]` matmul. Every
number below is checked against those two shapes.

## What Linalg forces on the design

Five things, none of them negotiable, and three of them change the schema.

**1. There is no average pool in Linalg.** The pooling family is
`sum` / `max` / `min` (plus `_unsigned` integer variants). ONNX
`AveragePool` and `GlobalAveragePool` arrive as **`pooling_*_sum` followed by
a separate `divf`**, exactly as above. The PPU's native modes are Avg, Max
and Min.

**2. And the PPU cannot do a bare sum.** It has no divider; average is a
multiply by a per-axis reciprocal held as `fp16(65536/k)`, with
`avg = sum * recip_w * recip_h * 2^-32`. A sum needs the product to be 1,
i.e. both fields at 65536, which is past fp16's 65504 ceiling. (This is the
same constraint that makes `k = 1` unrepresentable and forces `k >= 2`.)

  So: **`PoolingMethod` on the wire is AVG/MAX/MIN, never SUM.** Recognising
  `sum + div(kh*kw)` as an average is the *compiler's* job, and a sum-pool
  whose divisor is missing or is not exactly `kh*kw` must stay on the CPU.
  Putting SUM in the schema would create a wire state the runtime cannot
  execute.

**3. The window is a shape-only operand.** `ins(input, %window)` where
`%window` is a `tensor.empty()` -- it carries no values, only `kh x kw`. A
matcher reads the kernel from its *type*, not from an attribute. Strides and
dilations are attributes; dilations must be 1 (the PPU has no dilation).

**4. Pooling arrives NCHW, and NCHW has no `min`.** The named set is
`pooling_nchw_max` and `pooling_nchw_sum` only -- `min` exists at `nwc`,
`nhwc` and `ndhwc` but **not** `nchw`. So the HAL's Min is reachable only
from an NHWC model or a `linalg.generic`. Not a blocker for MobileNetV2
(which needs Avg), but it means the Min rung ships without a named op to
match, and the matcher set should not pretend otherwise.

**5. `linalg.matmul` carries optional `indexing_maps`.** Transpose and
broadcast are expressed by overriding the maps rather than by separate ops
(the `matmul_transpose_a/b` names are gone). The model above has the default
maps and a materialised `linalg.transpose` on the constant weight, which is
the easy case. A matcher must **check the maps are the default** rather than
assume it; a transposed-B matmul is a different memory layout and would need
its own packing.

## What the HAL already gives us

`iree-rocket-hal/src/rocket/pooling.rs` -- `PoolingShape` + `PoolingPlan`,
1187 lines, capture-derived and partly hardware-validated:

| knob | limit | source |
| --- | --- | --- |
| method | Avg (0), Max (1), Min (2) | RK3588 probe, encoding corrected from an earlier guess |
| precision | Int8, Fp16 | vendor captures `(proc, in)` = (0,1) / (2,2) |
| spatial extent | 1..=8192 | 13-bit N-1 field, vendor's own 8193 diagnostic |
| kernel, stride | 1..=16 by field; **kernel <= 8 in practice** | `MAX_DIRECT_KERNEL`: HW confirms 8x8, rejects a directly programmed 16x16 |
| padding | 0..=7 | 3-bit field |
| channels | preserved; rounded to one 16-byte atom (C8 fp16 / C16 int8) | every direct vendor program |
| width tiling | 129 in / 64 out per task, 256/128 for the exact fp16 2x2s2 path, 130/65 int8 | `PoolingPlan`, capture + HW |

MobileNetV2's pool is **7x7, stride 1 (or 7), pad 0, 1792 channels, one
output pixel** -- kernel 7 is inside the direct-kernel 8, the extents are
trivial, and it plans as a single PPU task. It fits today.

`fc.rs` -- `fc::Shape { m, k, n }` lowering `[M,K] x [K,N]` to a 1x1
convolution: **M is the conv width, height is 1, K is Cin, N is Cout**. 160
ONNX `Linear` models were swept to establish that this is what the vendor
toolchain itself emits, and it is HW-validated at fp16 M=7/K=16/N=32 and
int8 M=7/K=16/N=33. So the matmul lowering needs no new register work at
all -- only a wire format and a matcher.

## The one hardware limit in the way

`fc::Shape` inherits the convolution channel ceilings, and MobileNetV2's
matmul is `K = 1792`:

```
K  = 1792  ->  Cin  1792  >  MAX_INPUT_CHANNELS = 1344   (fp16)   was BLOCKED
N  = 1001  ->  Cout 1001  <= MAX_OUTPUT_CHANNELS = 1792            ok
M  = 1     ->  conv width 1, height 1                              ok, but degenerate
```

**RESOLVED 2026-09-04 -- this is Phase 5, and it is done.** The partial
measurement already existed (`MAX_INPUT_CHANNELS`' own doc records fp16 k=1
exact at `Cin` 256..1792 at 14x14), but not at *this* geometry: a 1x1 spatial
"image", which is what an M=1 matmul degenerates to and which no conv ladder
had ever run. Measured on `planck`, 23 exploratory points then a 20-case
ladder, **0 mismatches and 0 device timeouts** at every one:

* `K` at M=1, N=64: 512, 1024, 1344, 1792, **2048** -- under `Selectors` and
  again under `Counting`;
* `N` at M=1, K=1792: 64, 512, **1001**, 1792, 2048;
* `M` at K=1792, N=64: 1, 2, 7, 16, 32, across the CBUF split changing
  7/5 -> 2/10 -> 4/8 with nothing else about the shape moving;
* the classifier itself, `M=1 K=1792 N=1001`, under both patterns and again
  with the fp32 accumulator kept.

`MAX_INPUT_CHANNELS` is now **1792** (not the 2048 also measured -- 1792 is
what a real model needs). `fc_matmul_geometry_matches_oracle` is the
regression, and `fc_matmul_ladder_matches_the_fc_lowering` checks the
ladder's cases field-for-field against `fc::Shape::as_conv_shape`, so a
ladder that drifted from the production lowering fails on the host rather
than passing while measuring its own geometry.

**So the matmul matcher's `dim_bounds` are `K <= 1792`, `N <= 1792`.** The
`M` bound is the open one: 32 is the widest measured, and nothing says the
next value fails -- it is simply where the ladder stops.

## Schema changes

All of these are compatible within `RKT1` by `docs/compatibility.md`'s own
rules: adding a union member and appending tables are explicitly listed as
safe, and no existing field, enum value or default moves.

```fbs
// Wire-format values, independent of the PPU's register encoding.
// SUM is deliberately absent: the PPU has no sum mode (its average is a
// multiply by fp16(65536/k), which cannot encode a divisor of 1), so a
// sum-pool is a compiler-side pattern, not a runtime capability.
enum PoolingMethod : ubyte {
  AVG = 0,
  MAX = 1,
  MIN = 2,
}

// Pooling shape fields that may be supplied by dispatch push constants.
// Numeric values are wire-format ABI and must not be changed or reused.
enum PoolingDimension : ubyte {
  INPUT_WIDTH = 0,
  INPUT_HEIGHT = 1,
  CHANNELS = 2,
  KERNEL_WIDTH = 3,
  KERNEL_HEIGHT = 4,
  STRIDE_X = 5,
  STRIDE_Y = 6,
}

// One standalone PPU + PPU_RDMA pooling job.
table PoolingDef {
  input_width:uint32;
  input_height:uint32;
  channels:uint32;          // preserved by pooling; one field, not two

  output_width:uint32;      // carried and re-derived, as Conv2DDef does
  output_height:uint32;

  kernel_width:uint32;
  kernel_height:uint32;
  stride_x:uint32;
  stride_y:uint32;

  pad_left:uint32;
  pad_top:uint32;
  pad_right:uint32;
  pad_bottom:uint32;

  method:PoolingMethod = MAX;
  precision:Precision = INT8;

  runtime_dimensions:[PoolingDimension];
}

// Matmul shape fields that may be supplied by dispatch push constants.
enum MatmulDimension : ubyte {
  M = 0,
  K = 1,
  N = 2,
}

// One row-major [M,K] x [K,N] -> [M,N] matmul. Supersedes
// FullyConnectedDef, which no compiler ever emitted: `linalg.matmul` is
// what IREE actually hands us, and "fully connected" is not an op in the
// Linalg dialect at all.
table MatmulDef {
  m:uint32;
  k:uint32;
  n:uint32;

  input_zero_point:uint32;
  output_zero_point:uint32;
  weights_zero_point:uint32;

  input_scale:float = 1.0;
  weights_scale:float = 1.0;
  output_scale:float = 1.0;

  truncate_bits:uint32;
  activation:Activation = NONE;
  activation_cmp:uint32;
  precision:Precision = INT8;

  runtime_dimensions:[MatmulDimension];   // M, K, N
}

union KernelDef {
  Conv2DDef,
  FullyConnectedDef,   // deprecated, never emitted; kept so union tags do not move
  PoolingDef,
  MatmulDef,
}
```

Four deliberate choices in that, each of which could reasonably have gone the
other way:

**`pad_value` is not on the wire.** `PoolingShape` has one, and its own doc
says picking a sane value is "the caller's responsibility". A wire field
there is a way for a producer to send a max-pool a pad of `0` and silently
change the answer at the border. The runtime should derive it from
`(method, precision)`: the reduction's identity element -- `-inf` for max,
`+inf` for min, `0` for avg -- which is one match arm, not an ABI.

**`channels`, singular.** `PoolingShape` carries `input_channels` and
`output_channels` and then asserts they are equal. A pool preserves channels
by definition; one field cannot disagree with itself.

**`FullyConnectedDef` stays in the union.** Compatibility policy says
existing numeric values must never be reused, and the union tag is a numeric
value. It stops being emitted and gets a deprecation comment; the runtime
keeps decoding it until old vmfbs are gone. `MatmulDef` is not a rename --
it takes the `runtime_dimensions` treatment FC never had.

**`runtime_dimensions` on both.** This is the choice worth arguing about
(see Open decisions). The conv path started with three static executables
and retired them for one dynamic executable driven by push constants; the
same pressure applies here, or the transform spec needs one declared
executable per `(method, precision, kernel, stride, pad)` tuple.

## Compiler side

`RocketTarget.cpp` builds the flatbuffer from the **executable target
attribute's configuration dictionary**, not by reading ops -- `kernel =
"conv2d"` today, with every shape field a dictionary entry. So the compiler
work is three mechanical pieces plus one real one:

1. `kernel = "pooling"` and `kernel = "matmul"` accepted alongside
   `"conv2d"`; `buildRocketPoolingConfigFromTarget` /
   `buildRocketMatmulConfigFromTarget` mirroring the conv2d builder; the
   corresponding `iree_hal_rocket_*Def_*_add` calls. Bounded and dull.
2. Executable declarations in `rocket_conv2d_transform_spec.mlir` --
   `#rocket_pooling_avg_fp16_target` and friends -- with the pipeline layout
   declaring the right push-constant count, which `serializeExecutable`
   already cross-checks against `runtime_dimensions.size()`.
3. `util.func` dispatch helpers, following `@call_rocket_dynamic_conv2d`:
   read dims with `tensor.dim`, `arith.index_cast` to i32, `flow.dispatch`,
   and the f16/f32 conversion `flow.dispatch.workgroups` on `@cpu_device`.
4. **The matchers**, which is the part with actual design in it.

### The average-pool matcher

The pattern to claim is a *pair*, not an op:

```
linalg.pooling_nchw_sum(%in, %window) outs(%zero_fill)   ->  %sum
linalg.generic { divf %in, %c }                          ->  %avg     where %c == kh*kw
```

Requirements the matcher has to check, in the order that fails cheapest:

* op name `linalg.pooling_nchw_sum` (and later `pooling_nhwc_sum`);
* `dilations == [1,1]`;
* window type gives `kh, kw` both in `2..=8` (`MAX_DIRECT_KERNEL`, and
  `k >= 2` because `fp16(65536/1)` overflows);
* strides in `1..=16`;
* the `outs` init is a zero `linalg.fill` -- a nonzero init is an
  accumulate, which the PPU cannot do;
* the sole consumer is a `divf` by a **splat constant equal to
  `kh * kw`**. Not `kh*kw` and the average is count-exclude-pad or a
  scaled variant, and we must decline;
* padding: MobileNetV2's is zero. Nonzero padding is admissible only for
  `AVG` when the divisor is `kh*kw` (the PPU is count-include-pad), and
  freely for `MAX`. `docs` for `pooling_nhwc_sum` say nothing about pad
  semantics because Linalg has no padding on pool ops at all -- it is
  materialised as a `tensor.pad` producer, which is a *third* op to match
  and is out of scope for v1.

The DAG-matching hazards are already written down in this project's notes
(`transform-dag-matcher-traps`): `cast_compatible_dag_from_root` declines
silently in at least three ways, and `iree-opt` on the matcher alone is how
to debug it. Expect the pair-matching to be the schedule risk in Phase 3,
not the schema or the runtime.

An honest fallback if pair-matching fights back: match the sum-pool alone,
emit `method = AVG`, and **leave the `divf` on the CPU as a multiply by
`kh*kw`** -- since the PPU already divided, the CPU op becomes a
compensating multiply rather than disappearing. It costs one elementwise
pass over `1x1792x1x1` (3.5 KB) and removes the pair-matching problem
entirely. Do this first, and treat true fusion as an optimisation.

### The matmul matcher

Simpler, because `fc.rs` already fixes the lowering:

* op name `linalg.matmul`;
* `indexing_maps` absent or default -- i.e. **not** transposed or broadcast;
* rank-2 operands, element types f16/f16 -> f32 (mirroring the conv
  matchers' `lhs_type`/`rhs_type`/`output_type` checks), or i8/i8 -> i32 for
  the quantized rung later;
* `outs` init is a zero fill (no accumulate);
* `dim_bounds`: `K <= MAX_INPUT_CHANNELS`, `N <= MAX_OUTPUT_CHANNELS`. With
  today's constants MobileNetV2's `K = 1792` **fails this**, deliberately and
  loudly, until Phase 5. `M` has no constant to check against -- it becomes
  the convolution *width*, which `ConvPlan` bounds by column partitioning
  rather than by a ceiling -- so the matcher should bound it at whatever the
  Phase 5 ladder validates and no further.

`linalg.batch_matmul`, `linalg.quantized_matmul` and the `matmul` forms with
non-default indexing maps are all out of scope for v1 and must be *excluded
by the matcher*, not left to the runtime.

## Runtime side

Least work of the three:

* `executable_cache.rs`: two new `schema::KernelDef` arms.
  `PoolingDef -> UkernelShape::Pooling` (deriving `pad_value` from method
  and precision, then `PoolingShape::validate()` behind `catch_unwind`, the
  same way the FC arm calls `fc::Shape::new`), and
  `MatmulDef -> UkernelShape::Matmul`, which can be `fc::Shape` renamed.
* `command_buffer.rs`: the `UkernelShape::Pooling` dispatch arm **already
  exists** (2 bindings, 0 constants, `PoolingPlan::programs_with_buffers`).
  Two things it needs: push-constant handling if `runtime_dimensions` lands,
  and the layout question below.
* The legacy tag `1` hardcoded pooling shape should be retired once
  `PoolingDef` works -- it is documented as not hardware-validated and it is
  the only thing that has ever built a `UkernelShape::Pooling`.

### The layout question, which is the real one

The conv dispatch arm sets `input_packing`, `weight_packing`,
`bias_packing` and `output_compaction`; the pooling arm sets **all four to
`None`** -- it assumes its buffers are already NC1HWC2. Nothing in the
compiler produces that. So v1 must add the same per-dispatch host repack the
conv path uses, and it is worth being clear-eyed that this is the tax
`rocket-layout-repack-per-dispatch` documents and that the 22-vs-48-site
measurement charged 35% for.

For this shape the tax is small in absolute terms -- `1x1792x7x7` fp16 is
175 KB in and 3.5 KB out -- but the pool sits two ops downstream of a conv
whose output the driver has *just unpacked*. Pack -> conv -> unpack -> pack
-> pool -> unpack is four repacks where two would do. That is exactly the
cross-op chaining ISSUES.md P2 describes as HW-proven for fp16, and pooling
is the cheapest possible place to prove it: a pool consuming a conv's
already-packed output needs no new register work, just a decision not to
round-trip. **Recommend building v1 with the repack and treating the
conv->pool chain as the first chaining experiment**, rather than blocking
v1 on it.

Two ops downstream, not one, because a `Clip` sits between them -- and that
Clip is the *third* thing already in the schema that nothing emits. Every
target config in the transform spec says `activation = "none"`, while
`Activation::RELUX` and `activation_cmp` have been in `Conv2DDef` and in the
HAL (`Activation::Clamped`, hardware-tested in
`conv_activation_fused_hw.rs`) the whole time. MobileNetV2's 35 `onnx.Clip`
ops are relu6. Claiming the Clip into the conv it follows is independently
worth doing -- 35 CPU dispatches and their repacks -- and it is also what
makes conv -> pool *adjacent*, which is the precondition for the chaining
experiment above. It is not on this branch's critical path, but it is the
same shape of gap and the same fix, and whoever writes the pooling matcher
will already have the machinery in their head.

## Accuracy note, because it lands next to the classifier

The PPU average multiplies by `fp16(65536/k)` per axis, ~3-4 significant
figures. For `k = 7` that is `9362.28...` rounded into an fp16 grid whose
spacing is 8 in `[8192, 16384)` -- roughly 0.05% per axis, ~0.1% on the
7x7 pool. The vendor's own runtime has the same error (this repo is
bit-identical to it on the same reciprocal), but a CPU f32 reference will
not match bit-exactly, and the very next op is the 1001-way classifier whose
top-2 logit gap on this model is 1.26. The e2e check must be top-1/top-5
stability plus a max|err| bound, not bit-exactness -- the same standard the
fp16 conv offload already uses.

## Phasing

| phase | content | gate |
| --- | --- | --- |
| 0 | `.fbs` additions, regenerate `rocket_executable_def_generated.rs`, extend `rocket-schema/tests/compiler_fixture.rs` round-trips | host tests; no behaviour change |
| 1 | runtime decode arms + `pad_value` derivation + `UkernelShape::Matmul`; retire legacy tag 1 | `cts/`, driver unit tests |
| 2 | `RocketTarget.cpp` config builders and serialization for both kernels | new lit tests under `rocket-compiler-plugin/test/` |
| 3 | transform-spec executables, dispatch helpers and matchers (avg-pool first, matmul second) | lit tests mirroring `rocket_fp16_match_boundaries.mlir` |
| 4 | e2e on both MobileNetV2 models: pool and matmul each offloaded once | `tools/e2e_conv_regression.py`, top-1/top-5 + max\|err\| |
| 5 | **DONE** -- `MAX_INPUT_CHANNELS` 1344 -> 1792 on the FC geometry | `fc_matmul_geometry_matches_oracle`, 20/20 on `planck` |

Phase 5 was the one that needed the board before the compiler work was worth
anything, and it ran first for that reason: the matcher's `dim_bounds` are
now written once, at a number that was measured rather than inferred.

## Open decisions

1. **Dynamic or static pooling executables.** `runtime_dimensions` mirrors
   the conv path and keeps the executable count at one per
   `(method, precision)`; static executables are simpler but need a spec
   edit per new pool geometry. Recommendation: dynamic, with kernel and
   stride as push constants, padding staying an executable property (it is
   0..=7 and interacts with method semantics).
2. **Match the sum+div pair, or offload the sum-pool as AVG and leave a
   compensating multiply on the CPU.** Recommendation: ship the second,
   measure, then fuse.
3. **Does `MatmulDef` carry a bias binding?** The conv path binds a
   zero-filled bias today and MobileNetV2's bias add is a separate
   `linalg.generic`. Folding it costs a fourth binding and a matcher that
   claims two ops; leaving it costs one elementwise pass over 1001 floats.
   Recommendation: leave it, revisit with the pool's `divf`.
