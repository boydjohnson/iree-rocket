# iree-rocket

Mono-repo for building [IREE](https://github.com/iree-org/iree) with support
for the Rocket NPU backend (RK3588). This repository produces:

- `iree-compile` — with the Rocket compiler target registered.
- `iree-run-module` / `iree-benchmark-module` — with the Rocket HAL driver
  statically linked in.

## Layout

| Path | Role |
|---|---|
| [`rocket-schema`](rocket-schema) | Canonical FlatBuffers schema for Rocket executables; shared by the compiler plugin (C++) and the runtime crates (Rust). |
| [`rocket-plan-ffi`](rocket-plan-ffi) | Versioned C ABI (`include/rocket_plan.h`) over `rocket-core`'s planner, built as a host staticlib by the compiler plugin's CMake and linked into `libIREECompiler.so`; how the compiler asks the same planner the runtime asks. |
| [`rocket-core`](rocket-core) | Pure, dependency-free Rust crate: convolution/matmul descriptors, hardware limits, layout geometry, CBUF partitioning and tile planning, with fallible `try_*` APIs that return a `PlanError` code instead of panicking. Shared by the runtime and, eventually, the compiler. |
| [`iree-rocket-hal`](iree-rocket-hal) | Low-level Rust crate: ioctl/mmap access to the RK3588 NPU and register command building. Consumes `rocket-core`'s plans and re-exports its planner under `rocket::conv`. |
| [`rocket-hal-driver`](rocket-hal-driver) | Rust `staticlib` implementing IREE's HAL driver interface, statically linked into IREE via `iree_register_external_hal_driver()`. Depends on `iree-rocket-hal` and `rocket-schema`. Includes HAL CTS wiring under `cts/`. |
| [`rocket-compiler-plugin`](rocket-compiler-plugin) | C++ IREE compiler target plugin ("Rocket"), loaded via `IREE_CMAKE_PLUGIN_PATHS`. Serializes executables using `rocket-schema`'s FlatBuffer format. |
| [`rocket-compiler`](rocket-compiler) | Rust driver over `libIREECompiler.so`. Applies the Rocket transform spec and the device flags it expects, and can audit why each convolution and matmul is where it is. |
| `iree-build/iree-src` | `iree-org/iree` as a pinned git submodule. |
| `iree-build` | CMake configuration used to build IREE with the Rocket driver/plugin. |

Three documents sit alongside this one: [LIMITS.md](LIMITS.md) is what the
stack is *measured* to do -- the channel, kernel, stride and precision bounds,
and which of them each layer enforces -- [ISSUES.md](ISSUES.md) is what is
still open, and [ROADMAP.md](ROADMAP.md) is which MLIR operations the hardware
could run but the compiler cannot yet reach.

## Building

Each of the three artifacts has its own build directory under `iree-build/`,
since each configures a different CMake source root (the vendored `iree-src`
directly, vs. one of two small wrapper projects that register the Rocket HAL
driver before IREE's own `add_subdirectory` runs):

```sh
# iree-compile, with the Rocket compiler target registered. Needs `cargo` on
# PATH: the plugin links rocket-plan-ffi (the shared planner's C ABI), which
# the CMake build compiles with cargo into its own target directory.
./iree-build/configure-compiler-host.sh
cmake --build iree-build/build

# iree-run-module / iree-benchmark-module, host build, with HAL CTS
(cd iree-build/host && cmake --preset runtime-host && cmake --build --preset runtime-host)

# iree-run-module / iree-benchmark-module, cross-compiled for the RK3588 board.
# Build compiler-host first: this configuration points IREE_HOST_BIN_DIR at
# iree-build/build/tools for codegen tools (e.g. iree-c-embed-data) that
# IREE's build runs on the host even when cross-compiling the runtime.
./iree-build/configure-runtime-aarch64.sh
cmake --build iree-build/host-aarch64/build
```

`iree-build/host/CMakePresets.json` and `iree-build/host-aarch64/CMakePresets.json`
each live next to the wrapper `CMakeLists.txt` they configure (CMake presets
are always rooted at the directory containing the `CMakeLists.txt`, so a
single repo-root `CMakePresets.json` can't span these plus the vendored
`iree-src` tree). `configure-runtime-aarch64.sh` is the command-line equivalent
of the aarch64 configure preset, while `configure-compiler-host.sh` covers the
compiler case, which has no wrapper project at all.

The Rust crates (`rocket-core`, `rocket-plan-ffi`, `rocket-schema`, `iree-rocket-hal`, `rocket-hal-driver`) form
a single Cargo workspace and can be built/checked independently of the CMake
builds above:

```sh
cargo build --workspace
cargo test --workspace
```

## Compiling a model

`rocket-compiler` wraps `libIREECompiler.so` with the Rocket transform spec and
the device flags that spec hardcodes, so a model does not have to be compiled
by hand. It is also the only way to compile a model correctly: it runs the
pipeline in two stages so it can pin placement in the middle (see **Placement
pinning** below), which a single `iree-compile` invocation cannot do.

```sh
export IREE_COMPILER_LIB=iree-build/build/lib/libIREECompiler.so
cargo run -p rocket-compiler -- compile --input model.mlir --output model.vmfb

# What actually ended up on the NPU:
cargo run -p rocket-compiler -- audit --input model.mlir
```

`audit` answers "why is this operation where it is". It prints one line per
convolution and matmul candidate -- source location, kind, layout, shape,
precision rung, the decision, and the decisive reason -- then the executable
and dispatch-site counts, then a reconciliation of the two:

```text
Placement decisions (2 candidate(s) at selection time):
  dense_conv2d at model.mlir:7:12 -> Rocket, direct [fp16]
      nhwc 14x14 Cin 88 Cout 528 k1x1 s1
      cbuf 2/10
  dense_conv2d at model.mlir:13:12 -> CPU [unvalidated_configuration] [fp16]
      nhwc 7x7 Cin 3585 Cout 64 k1x1 s1
      validation policy: register-representable, not measured
      input channels must be 1..=3584; beyond that the channel padding has no
      capture backing at this precision
  1 direct, 0 tiled, 1 cpu, 0 deferred; 1 hardware job(s) planned
  ...
Reconciliation:
  1 accepted candidate(s) -> 1 Rocket dispatch site(s) running 1 hardware job(s)
  4 CPU dispatch site(s), 1 of them convolution- or matmul-shaped
    - main_dispatch_3 (1 site(s)): matmul_like
```

Three counts, deliberately kept apart. **Candidates** are original
operations. **Dispatch sites** are what actually ran where -- the number that
answers "how much of the model ran on the NPU", because the transform spec
routes every matched convolution through a handful of fixed executables (so
an executable count understates the NPU badly) while IREE deduplicates
identical CPU dispatches (which overstates the CPU side). **Hardware jobs**
are standalone NPU jobs: one dispatch already runs several when the planner
tiled it.

The decisions come from the shared planner (`rocket-core`) at selection time,
recorded by `rocket-plan-candidates` before the match loop erases the
candidates it claims; the placement comes from the compiled module. Neither
alone explains anything. Together they separate the planner refusing, the
admission envelope having no evidence for the shape class, and *both* saying
yes while a matcher still declined -- which is a semantic or fusion gap in
the transform spec, and the only one of the three that closes without new
hardware measurements. Each refusal names its class: `shape`, `semantics`,
`hardware` (permanent), or `validation` (nobody has measured it yet).

`--report-json <path>`, on `audit` or `compile`, writes the same thing
machine-readably. `--strict-offload`, on either, fails the compile instead of
falling back: any candidate the planner, the envelope or the op's form ruled
out, any accepted candidate with no Rocket dispatch to account for it, and
any CPU dispatch whose export name says it is running a convolution or a
matmul. It cannot see an operation IREE fused into a larger element-wise
dispatch, so a pass means "nothing observably left on the CPU", not a proof.

### `--batch-matmul`: attention on the NPU, off by default

`linalg.batch_matmul` reaches no matcher -- the matmul path reads row-major
`linalg.matmul` -- so a transformer's attention core stays on the CPU.
`--batch-matmul` splices in `rocket-unbatch-matmul`, which splits a
static-batch contraction into one matmul per batch element. On ViT-B/16 that
takes the model from 73 NPU dispatch sites to 361 and leaves nothing
contraction-shaped on the CPU but the patch-embed stem, at max|diff| 0.0074
against an ONNX Runtime oracle.

It is off because it is **1.16x slower** (592 ms -> 687 ms on `planck` at
eight workers). Offloading the whole attention core bought 3 ms of CPU time
and cost 125 ms of `record`, `compact`, `pack.input` and NPU time -- the CPU
was spending almost nothing on it. ISSUES.md C15 has the phase tables. Like
`--elementwise`, the flag exists so both arms can be measured rather than
argued about.

### The CPU-only baseline: `--no-offload`

An NPU-vs-CPU comparison needs a CPU arm built by the *same* pipeline. A
module compiled with plain `iree-compile` is not one: `@__transform_main` runs
`iree-preprocessing-convert-conv-to-channels-last` and
`linalg-specialize-generic-ops` before its match loop, so anything through
`rocket-compiler` is NHWC whether or not a single convolution offloads, and
IREE's CPU backend is **2.8x slower** on NCHW MobileNetV2. Every NPU-vs-CPU
number this repo quoted before 2026-09-04 was measured against that slower
build; see ISSUES.md M4 for the bisection.

`--no-offload` builds the baseline correctly. It rewrites every matcher in
the loop so it declines: the convolution and matmul matchers get a
`no_offload` attribute on their `transform.rocket.match.admitted` line, and
the pooling and element-wise ones -- which have no planner to ask, so they
still carry bounds -- get `transform.iree.match.dim_bounds` set to
`umin = umax = 999999`, a bound no real dimension meets. The passes around
the loop, the device topology and the placement pin stay exactly as the
offload arm sees them. A matcher carrying neither hook fails the build
rather than quietly offloading:

```sh
# The arm under test, and its like-for-like baseline.
cargo run -p rocket-compiler -- compile --input mnv2.int8.mlir \
    --output mnv2.int8.npu.vmfb --llvmcpu-target-triple aarch64-linux-gnu
cargo run -p rocket-compiler -- compile --input mnv2.int8.mlir --no-offload \
    --output mnv2.int8.cpu.vmfb --llvmcpu-target-triple aarch64-linux-gnu
```

On `mnv2.int8.mlir` that is 50 Rocket dispatch sites against 0, and 95 CPU
sites against 64 -- the offload's own overhead, visible in `audit` before
anything runs. Confirm it with `audit --no-offload`, which must report
`0 dispatch site(s) -> rocket`.

The flag refuses to produce a spec it cannot vouch for: if a matcher in the
`foreach_match` list constrains no dimension, rewriting the bounds would not
stop it, so the build fails rather than hand back a "baseline" that quietly
offloads part of the model.

Two things this does not fix, both of which still invalidate a comparison:
run both arms at the same core allocation, and do not use `taskset -c 4,5` --
the offload deficit it reports is roughly double the one a full machine sees
(ISSUES.md P8).

### Opt-in coverage: `--elementwise`

The two-tensor element-wise matchers (`linalg.add`, `mul`, `sub` on a rank-3
`1 x tokens x channels` tensor) ship **disabled**. They are written into
`rocket_conv2d_transform_spec.mlir`'s `foreach_match` list but commented out
behind a `//@ROCKET_ELEMENTWISE@` marker, so anything that reads the spec
directly -- a bare `iree-compile` included -- gets the conservative list.
`--elementwise` uncomments exactly those lines.

They are off because they are measured to be slower, not because they are
provisional. On ViT at the full machine (2026-09-06, `performance` governor,
NPU IRQs on a big core, 7 repetitions):

| cpus | `--no-offload` | 12 matmul sites | 184 sites (`--elementwise`) |
|---|---|---|---|
| 0-7 | 3973 ms | 3709 ms (1.07x faster) | 4655 ms (**1.17x slower**) |

Correct either way -- `max|err|` 0.0060 on logits with a standard deviation of
0.90, same predicted class -- and `ROCKET_PROFILE=1` shows the NPU is only
**4.2%** of wall. The extra time is the per-dispatch NC1HWC2 round trip plus
the `truncf`/`extf` CPU dispatches the shims add around each offloaded op
(ViT's CPU sites go 260 to 553). No matcher bound fixes that; layout
propagation would. See ROADMAP.md's Phase 1 and ISSUES.md P2/P8.

The flag composes with `--no-offload`, and only in one order: the entries are
enabled first and neutralized second, so the element-wise matchers are checked
for a defeatable predicate and disarmed along with the rest, and the baseline
arm runs the identical pipeline.

```sh
cargo run -p rocket-compiler -- audit --input vit.mlir --elementwise
cargo run -p rocket-compiler -- audit --input vit.mlir --elementwise --no-offload
```

### Placement pinning

The Rocket backend has no code generator. `serializeExecutable` only knows how
to read the config dict that `rocket_conv2d_transform_spec.mlir` stamps onto
the hand-authored executables it splices in, so the NPU can only ever run
dispatches the spec itself created -- every one of which carries an explicit
`stream.affinity = #hal.device.affinity<@rocket_device>`.

Dispatches IREE forms on its own carry no affinity, and Stream's affinity
analysis places them by propagating through consumers. A dispatch whose result
is used *only* by a Rocket dispatch therefore gets pulled onto the NPU, and
serialization then fails on something that is not a convolution at all. The
observed case is the explicit padding for an int8 depthwise convolution, a
112x112x48 -> 114x114x48 copy dispatch that IREE names `..._slow_memcpy`;
nothing pulls it back toward the CPU, because its destination is a fresh
`flow.tensor.splat` and the Rocket consumer is the only constraint the analysis
can see. `--iree-hal-default-device` does not help: the module-level
`stream.affinity.default` it sets only applies where the analysis finds
nothing.

`rocket-pin-unclaimed-dispatches` (in the compiler plugin) makes the placement
explicit instead, stamping that default onto every `flow.dispatch` that has no
affinity of its own. It has to run between the `flow` and `stream` phases --
after dispatch regions are formed and outlined, before the affinities are
consumed -- and no plugin hook exists that late, so `rocket-compiler` drives it
by name: `--compile-to=flow`, the pass, then `--compile-from=flow`. Both
`compile` and `audit` do this, so the report matches what a `.vmfb` would get.

Compiling by hand with `iree-compile` in one shot skips the pass, and since
the transform spec's int8 epilogue became a fusible `linalg.generic` (see
ISSUES.md P8) that now breaks a **single convolution** too, not just a whole
model: the epilogue dispatch's only producer is the Rocket dispatch, so
Stream's affinity analysis places it on `@rocket_device` and serialization
fails on an op that is not a convolution. `tools/e2e_conv_regression.py` runs
the three-stage build for exactly this reason; use `rocket-compiler`, or
replicate its `--compile-to=flow` / `iree-opt
--pass-pipeline=builtin.module(rocket-pin-unclaimed-dispatches)` /
`--compile-from=flow` sequence.

### Stride-2 dense convolution

The stride-2 dense matchers are enabled, so MobileNetV2's stem convolution
runs on the NPU: one more offloaded convolution, 35 rather than 34 on
`mnv2.fp16.mlir`. They were
disabled for a long time behind a compile failure that
`rocket-pin-unclaimed-dispatches` now fixes; turning them on then exposed
three genuine defects, all since fixed and covered by hardware regressions.

What remains is a precision tradeoff worth knowing about. Rocket's ABI is
f16-in/f32-accumulate, so an offloaded convolution runs its inputs at half
precision. MobileNetV2's stem is f32 and feeds an int8 quantization step, so
f16-level noise there crosses quantization boundaries and propagates: the
model lands about 0.35-0.42 max|err| on its final logits against a plain f32
build, where keeping the stem on the CPU is exact (7e-07). Top-1 is stable
across inputs except on near-ties -- on one measured input whose top-2 gap was
0.07, well inside that perturbation, the top two classes swapped.

This is the cost of f16, not of the NPU being wrong: the isolated stem
convolution matches a CPU reference computing the same f16 arithmetic to f16
epsilon. To trade the dispatch back for exactness, drop
`@match_dynamic_conv2d_s2` and `@match_dynamic_conv2d_3x3_s2` from the
`foreach_match` list in the transform spec.

### Convolution shape integrity

`rocket-demote-conv-inputs-to-f16` (in the compiler plugin) demotes all-f32
named 2-D convolution inputs to f16 for Rocket's f16-in/f32-accumulate ABI. It
exists because the upstream pass the transform spec used to call for this,
`iree-global-opt-demote-contraction-inputs`, rebuilds the named op through
`linalg::getPrunedAttributeList`, which elides the op's own declared attribute
names -- including `strides` and `dilations`. Every strided or dilated
convolution it touched silently became a stride-1 one.

That is a correctness bug independent of Rocket: linalg drives a convolution's
iteration space from its output, so the rewritten op still verifies and still
lowers, it just computes a different convolution over a corner of its input.
On MobileNetV2 it turned the stride-2 stem conv into a nominal stride-1 conv,
which then also matched `@match_dynamic_conv2d_3x3` (which requires stride 1)
and was dispatched to the NPU with the wrong stride.

`linalg.matmul` is demoted by the same pass, for a different reason. The
transform spec's `@call_rocket_matmul` used to narrow its own operands, which
looks equivalent and is not: that function is never inlined, so the truncf is
invisible to const-expr hoisting and re-narrows the *constant* classifier
weights on every inference -- 1.79M elements of CPU work into a fresh
transient buffer, which then misses the runtime's packed-coefficient cache
every time as well. Demoted in the caller instead, const-eval folds it into an
initializer. Both passes carry `indexing_maps` and `cast` across the rebuild
alongside `strides`/`dilations`, since for a matmul the indexing maps are what
distinguishes a plain matmul from a transposed one.

Demotion has to precede the match loop, because the matchers require
f16/f16/f32 typing -- but it cannot know which operations the loop will
claim, and deciding that up front would mean re-implementing the matchers'
eligibility predicates in C++ and keeping the two in sync. So the spec demotes
every all-f32 named convolution and matmul, matches, and then
`rocket-promote-unclaimed-conv-inputs` restores f32 on whatever is left:
anything still holding a `linalg.conv_2d_*` or `linalg.matmul` after
`foreach_match` is by definition unclaimed. Without it an unclaimed convolution runs on the CPU in
half precision when f32 was free -- on MobileNetV2 that is the stride-2 stem,
worth 0.349 max|err| on the final logits. Only the plugin's own demotion is
reverted: both passes agree on a `rocket.f16_demoted` tag, so a model that
authored its own f16 convolution is untouched.

`rocket-record-conv-attrs` and `rocket-verify-conv-shapes` are the tripwire
for anything like it. Record, run right before the demotion, stamps every
named convolution and matmul with its `strides`, `dilations`,
`indexing_maps` and `cast`; verify, run right after, errors if any op's
attributes no longer agree with that record or if an op has lost the record
-- a check that holds on symbolic shapes too -- and, where the extents are
static, if a convolution's output spatial extent disagrees with its own
input, filter, stride and dilation. Both run immediately before the
match/rewrite loop, while padding is still explicit and nothing has been
tiled, so the relation is exact there. It should never fire.

### ONNX models

`tools/import_onnx.py` does the whole import, and the three things below are
why it exists rather than a bare `iree-import-onnx` call:

```sh
tools/import_onnx.py model.onnx --out-dir vit \
    --dim batch_size=1 --dim num_channels=3 --dim height=224 --dim width=224
```

Pin the symbolic dimensions before importing. `iree-import-onnx` will happily
import a model whose batch is a symbolic `dim_param`, but the Rocket ABI fixes
batch at one and every matcher in the transform spec requires it, so a
dynamic-batch model compiles cleanly and offloads **nothing**. Clear each
`dim_param` on the graph inputs and outputs, drop `graph.value_info`, and
re-run `shape_inference.infer_shapes` before `iree-import-onnx`. Not every
model leaves only batch symbolic -- ViT-B/16 leaves all four input dims that
way, so its channel count and spatial extents need pinning too, or the
patch-embed convolution is still symbolic after the batch is fixed.

`tools/import_onnx.py` also handles ONNX Runtime *optimized* exports -- the
`com.microsoft` contrib ops (`GroupQueryAttention`, `RotaryEmbedding`,
`SimplifiedLayerNormalization`, `SkipSimplifiedLayerNormalization`) that
`onnx-community/Qwen3-0.6B-ONNX` and models like it publish, often with no
plain-op variant. torch-mlir supports all four, but five rewrites are needed
around them and the script applies them when it sees such a node: pin
`value_info` rather than clearing it (the opposite of the plain path, because
`infer_shapes` cannot type a fused op), drop trailing empty optional node
inputs, split `SkipSimplifiedLayerNormalization`, route rank-3
`RotaryEmbedding` through the rank-4 entry point (torch-mlir's rank-3 path
reshapes where it must transpose, and the logits come out uncorrelated), and
pass `--large-model` so `onnx.checker` is skipped. Each is documented in the
script with the evidence behind it.

Build an ONNX Runtime oracle *before* importing. Without a reference from the
model's own runtime, a later difference cannot be attributed to the NPU rather
than to the import -- and at least one shipped model is mis-imported today
(ISSUES.md C14: a float16-converted ViT that ONNX Runtime runs correctly and
that IREE gets wrong on the host CPU with no NPU involved). Prefer the f32
file when a model ships both: the transform spec demotes convolutions and
matmuls to f16 itself and restores f32 on whatever the match loop leaves
behind, so the f32 import chooses precision per operation rather than for the
whole graph, and it is the arm measured exact against the oracle.

int8 models quantized with ONNX Runtime's `quantize_dynamic` are supported.
They import as `onnx.ConvInteger`, which upstream torch-mlir cannot lower at
all -- `RocketExpandOnnxConvIntegerPass` in the compiler plugin supplies the
expansion, so these models need this repository's `iree-compile`, not a stock
one. Their convolutions reach the NPU through the `int8_accumulator` precision
(int8 in, int32 accumulator out, requantization bypassed); the transform spec
folds the activation zero point into a CPU-side correction first, because that
hardware mode is only validated for zero zero-points.

## Profiling a run

The driver pays a host-side cost per dispatch that no per-job timer sees: the
input is repacked NHWC -> NC1HWC2, the weights are repacked into the CNA's
blocked coefficient order, and the DPU's atomic-slot output is compacted back
into IREE's dense buffer -- once per dispatch, on every inference.
`ROCKET_PROFILE` times each of those phases separately and prints a per-phase
table plus a per-op breakdown at exit:

```sh
ROCKET_PROFILE=1 ./iree-run-module --module=model.vmfb --function=main_graph \
  --device=rocket --device=local-task --input=1x3x224x224xf32=0.1
```

`ROCKET_PROFILE=trace` additionally prints a line per phase as it happens.
The `outside` row is time spent outside this driver entirely (the CPU
dispatches), so the two halves of a mixed model can be compared directly.
See ISSUES.md's P7 for the current MobileNetV2 fp16 numbers and what they
say; P6, which used to carry them, is resolved.

Packed coefficients are cached across inferences (once per weight binding and
geometry rather than once per dispatch), which is worth 1.47x on MobileNetV2
fp16 in `iree-benchmark-module`. The report's `weight cache` line says how well
it is working; `rocket-hal-driver/src/weight_cache.rs` documents why a reuse is
safe. Three knobs:

| Variable | Effect |
|---|---|
| `ROCKET_WEIGHT_CACHE=0` | Disable; pack on every dispatch, as before. |
| `ROCKET_WEIGHT_CACHE=verify` | Pack anyway on a hit and compare against the cached bytes the regcmd reads, failing loudly on any difference. |
| `ROCKET_WEIGHT_CACHE_MB=N` | Byte budget, default 256 MiB per NPU context. |

The driver's host-side work is memory-bound, so on a big.LITTLE part it runs
several times slower on the little cluster -- 52.4 ms against 13.8 ms for
MobileNetV2 fp16's layout transforms on RK3588. The NPU worker thread that
runs every `queue_execute` therefore asks the scheduler for the
highest-`cpu_capacity` CPUs once, for its life; the profile's `host time by
cpu` line shows where it actually landed. On a machine whose cores are all the
same this finds nothing to prefer and does nothing.

| Variable | Effect |
|---|---|
| `ROCKET_HOST_CPUS=off` | Never change affinity. |
| `ROCKET_HOST_CPUS=0-3,7` | Use this CPU list instead of the highest-capacity one. |

`queue_execute` does not run the command buffer on the calling thread: it
queues it to a worker that owns one open of `/dev/accel/accel0` and runs
units in submission order, signalling IREE's semaphores when each is done
(`rocket-hal-driver/src/pool.rs`; the design and its measurements are in
`rocket-hal-driver/MULTICORE.md`). With `ROCKET_NPU_CORES=N` there are N
such workers, each on its own open of the device -- its own DRM scheduler
entity, so its own NPU core -- and each command buffer is placed on one of
them when it is created. Results are bit-identical at any N; whether N > 1
is faster depends on whether IREE gives the device independent command
buffers to run at once, which the profile's `overlap` line reports.

A dispatch with several CBUF tiles spreads them over the sibling contexts
(each tile is an independent job): the siblings get a copy of the input
rows their tiles read, the bias, and the packed coefficients (cached per
context), the tiles are submitted to all files before any is waited for,
and the output is gathered from each context's scratch. Driver-private
scratch buffers are pooled across command buffers, which is worth ~1.2x on
its own at one context.

| Variable | Effect |
|---|---|
| `ROCKET_NPU_CORES=N` | Worker contexts, 1..=8. Default 1. |
| `ROCKET_NPU_CORES=auto` | One per NPU core (3 on RK3588). |
| `ROCKET_FANOUT=0` | Keep every tile of a dispatch on its command buffer's own context. |
| `ROCKET_PIN_WORKERS=0` | Let workers float over the big cluster instead of one core each. |
| `ROCKET_SCRATCH_POOL=0` | Allocate and free every scratch buffer instead of pooling. |
| `ROCKET_SCRATCH_POOL_MB=N` | Bytes the scratch free lists may hold, default 256 per context. |

Two dispatches on one command buffer that share a tensor do not go through
the dense buffer between them. The consumer reads the producer's NC1HWC2
output cube in place instead of repacking (the *chain*), and the producer
skips writing the dense buffer at all when the compiler counted every reader
of its result as a Rocket dispatch and every one of them chained (*lazy
compaction*). Every dispatch kind takes part -- convolution, matmul, pooling
and element-wise -- with one geometric caveat: the PPU strides its surfaces
by the pixel count rounded up to four, so a pool only chains at a multiple
of four pixels. The count travels as the dispatch's last push constant
(`rocket-mark-dense-readers`, run by `rocket-compiler` at the flow phase);
a reader on another command buffer, a CPU reader, or a consumer that could
not chain all keep the write. Two compiler details make the edges adjacent
in the first place: each shim widens its f16 result with a plain generic
whose accumulator-init term `rocket-fold-neutral-init` drops when the init
is a zero (or -inf) fill, so the consumer's narrow cancels it, and the
demote pass narrows an f32 import *through* its zero pads so the pad-folding
matchers still see `pad -> conv`. The profile's `compaction:` line counts
the writes skipped; `pack.input` and `compact` are the phases that shrink.

| Variable | Effect |
|---|---|
| `ROCKET_CHAIN=0` | Always repack a consumer's input from the dense buffer. Implies no lazy compaction. |
| `ROCKET_CHAIN=debug` | Print every chain edge taken and why each other one was declined. |
| `ROCKET_LAZY_COMPACT=0` | Always write the dense output buffer. |
| `ROCKET_LAZY_COMPACT=debug` | Print every dispatch's kept/skipped decision with the reader counts behind it. |

`iree-rocket-hal`'s `layout_bench` and `gem_bandwidth` examples measure the
transforms and the GEM mapping directly, which is a much faster way to test a
hypothesis about either than a model run:

```sh
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
  cargo build -p iree-rocket-hal --release \
  --target aarch64-unknown-linux-gnu --example layout_bench
scp target/aarch64-unknown-linux-gnu/release/examples/layout_bench planck:/tmp/
ssh planck '/tmp/layout_bench 20 cold'
```

## Board convolution regression gate

The convolution regression command checks the low-level ConvPlan/NPU path and
dense plus depthwise compiler-to-VMFB-to-public-driver convolutions against
independently compiled CPU results. It cross-builds the Rust probe, stages
temporary files over SSH, runs them on the RK3588, and fails on any numerical
mismatch:

```sh
python3 tools/e2e_conv_regression.py --board "<board name>"
```

It requires Python with NumPy, `ssh`/`scp` access to the board, the aarch64
Rust target and cross-linker, and the host/aarch64 IREE builds described above.
The compiled cases include the previously problematic VGG geometry (30x30,
Cin=512, Cout=512, 3x3) and a 40-channel 3x3 depthwise convolution that crosses
the driver's 32-channel weight-packing group boundary.

`tools/e2e_matmul_regression.py` is the same gate for matmul, with the same
flags:

```sh
python3 tools/e2e_matmul_regression.py --board "<board name>"
```

Its raw half runs the FC oracle tests from both `fc_hw` and `fc_phase3_hw`;
its compiled half covers the ViT projection shape, MobileNetV2's classifier,
the `K`/`N` channel ceilings at 3584, ViT-B/16's `197x768x3072` MLP
projection, `M` at the matcher's 2047, `linalg.matvec` and `vecmat` through
the GEMV raising, and two matmuls sharing a command buffer. **Eight of its
nine cases are compared exactly.** A contraction can be gated exactly if its
fixtures are ternary -- entries from `{-1, 0, 1}` are exact in f16, every
product is, and the sums stay far inside f16's integer-exact range (measured
|C|max 79 at `K` 768, 118 at 1792, against a ceiling of 2048; a ternary sum
grows as the square root of `K`, so `K` 3584 stays well under it). The ninth
case is the same shape with realistic magnitudes and a tolerance, since
ternary data never rounds and so cannot see a precision fault.

`tools/e2e_pooling_regression.py` is the same gate for pooling, with the same
flags:

```sh
python3 tools/e2e_pooling_regression.py --board "<board name>"
```

Its raw half runs the PPU oracle tests (max, padded max, min, average, tiled);
its compiled half covers the average pool in NCHW and max pooling in both
layouts at both strides, plus a tiled width and two pools sharing one command
buffer. **Max pools are compared exactly.** A max pool returns one of its
inputs unchanged, so fixtures generated in f16 and widened to f32 survive the
shim's demote-and-widen round trip bit for bit, and any difference at all is a
real fault. Averages cannot be exact -- the PPU's average is a multiply by
`fp16(65536/k)` that the shim multiplies back out -- and take the `--atol`
/`--rtol` defaults.

## Whole-model survey

`tools/model_survey.py` runs one model end to end -- export, import, both
compile arms, correctness against an ONNX Runtime oracle, and a timing on the
board -- and writes a machine-readable `result.json` next to the artifacts.
`tools/export_onnx.py` is the registry of models it exports, so a re-export is
reproducible rather than remembered.

```sh
tools/model_survey.py run --model wide_resnet50_2 --board planck
tools/model_survey.py run --model vit_l_16 --board planck --repetitions 7
tools/model_survey.py summarize
```

Each stage is skipped when its output already exists, so an interrupted run
resumes and a re-timing costs only the timing. A model that is not in the
registry brings its own file and its own pinning:
`--onnx qwen3.onnx --dim batch_size=1 --dim sequence_length=128`.

Three things it is careful about, each of which has cost this repository a
result before:

**The baseline is `--no-offload`, never a stock `iree-compile` build** -- see
"The CPU-only baseline" above. The survey builds both arms from the same MLIR
with the same device flags and reports the ratio between them.

**The core allocation is a column, not a footnote.** `--cpu-ids` takes one
`--task_topology_cpu_ids` value per timing column and defaults to `4,5,6,7`
(the A76 cluster) and all eight. `taskset` is *not* the knob: IREE builds its
task topology from the machine's cpuinfo rather than from the process affinity
mask, so a `taskset -c 4-7` run thinks it has eight cores and puts about two of
them to work. The governor, the NPU IRQ affinity and the NPU's runtime state
are read off the board and recorded in `result.json` next to the numbers.

**The first repetition is discarded.** An offloaded arm packs its weights on
the first invocation only, which on a ResNet-scale model is a 945 ms
first sample against a 124 ms steady state -- large enough to move a median
taken over three. `--benchmark_repetitions` is raised by one behind the scenes
so the requested count still survives.

`summarize` is why the survey is worth running on more than one model. It
pools every `result.json` and prints one histogram of *why* candidates stayed
on the CPU, across all of them, ranked by count and tagged with each reason's
class:

```text
Why candidates stayed on the CPU, every model pooled:

    24  unvalidated_configuration [validation] dense_conv2d
          in resnet50, vgg19, wide_resnet50_2
          e.g. wide_resnet50_2: 230x230 Cin 3 Cout 64 k7x7 s2 [fp16]
               automatic planning above 3x3 currently has capture backing only
               at stride 1
```

A wall-clock number says one model got faster. A pooled reason histogram says
which *compiler* lever is worth building next and roughly what it is worth --
`shape` and `semantics` close with compiler work, `validation` needs a board
measurement, `hardware` never closes.

## Precision-transition probe

`tools/c8_precision_transition_probe.py` isolates ISSUES.md's C8: an int8
dispatch makes a following fp16 dispatch hang, and this says which fp16 jobs
are affected. It builds one module per invocation holding only the selected
cases, each a `func.func` whose Rocket dispatches run in the order listed, and
runs each case in its own process on the board with a quiet gap and a canary.

```sh
python3 tools/c8_precision_transition_probe.py --board planck
python3 tools/c8_precision_transition_probe.py --board planck --repeat 3 \
    --only int8_then_k1 --only int8_then_k3
```

`--only` names cases from `CASES`; `--repeat` runs each several times, which is
what tells a real result from the device's order-dependent flake. Every
convolution is checked to reach a Rocket matcher before anything runs, so a
case that quietly fell through to the CPU fails loudly instead of reporting a
clean row. Outputs are compared against a CPU build unless
`--skip-differential`.

Adding a case is two lines: an entry in `INT8` or `F16` for the shape, and one
in `CASES` for the sequence. The MLIR, the fixtures and the argument plumbing
are generated from those.

## Submodules

After cloning, initialize `iree-src` (and its own third-party submodules):

```sh
git submodule update --init --recursive
```
