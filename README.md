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
| [`iree-rocket-hal`](iree-rocket-hal) | Low-level Rust crate: ioctl/mmap access to the RK3588 NPU and register command building. |
| [`rocket-hal-driver`](rocket-hal-driver) | Rust `staticlib` implementing IREE's HAL driver interface, statically linked into IREE via `iree_register_external_hal_driver()`. Depends on `iree-rocket-hal` and `rocket-schema`. Includes HAL CTS wiring under `cts/`. |
| [`rocket-compiler-plugin`](rocket-compiler-plugin) | C++ IREE compiler target plugin ("Rocket"), loaded via `IREE_CMAKE_PLUGIN_PATHS`. Serializes executables using `rocket-schema`'s FlatBuffer format. |
| [`rocket-compiler`](rocket-compiler) | Rust driver over `libIREECompiler.so`. Applies the Rocket transform spec and the device flags it expects, and can audit what ended up on the NPU. |
| `iree-build/iree-src` | `iree-org/iree` as a pinned git submodule. |
| `iree-build` | CMake configuration used to build IREE with the Rocket driver/plugin. |

## Building

Each of the three artifacts has its own build directory under `iree-build/`,
since each configures a different CMake source root (the vendored `iree-src`
directly, vs. one of two small wrapper projects that register the Rocket HAL
driver before IREE's own `add_subdirectory` runs):

```sh
# iree-compile, with the Rocket compiler target registered
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

The Rust crates (`rocket-schema`, `iree-rocket-hal`, `rocket-hal-driver`) form
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

`audit` reports both executable and dispatch-site counts. Dispatch sites are
the number that answers "how much of the model ran on the NPU": the transform
spec routes every matched convolution through a handful of fixed executables,
so an executable count alone understates NPU placement badly, while IREE
deduplicates identical CPU dispatches, which overstates the CPU side.

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

Compiling by hand with `iree-compile` in one shot skips the pass. Single
convolutions (what `tools/e2e_conv_regression.py` compiles) have nothing to
mis-place and are unaffected, but a whole model can be.

### Stride-2 dense convolution

The stride-2 dense matchers are enabled, so MobileNetV2's stem convolution
runs on the NPU (18 offloaded dispatch sites rather than 17). They were
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

Demotion has to precede the match loop, because the matchers require
f16/f16/f32 typing -- but it cannot know which convolutions the loop will
claim, and deciding that up front would mean re-implementing the matchers'
eligibility predicates in C++ and keeping the two in sync. So the spec demotes
every all-f32 named convolution, matches, and then
`rocket-promote-unclaimed-conv-inputs` restores f32 on whatever is left:
anything still holding a `linalg.conv_2d_*` after `foreach_match` is by
definition unclaimed. Without it an unclaimed convolution runs on the CPU in
half precision when f32 was free -- on MobileNetV2 that is the stride-2 stem,
worth 0.349 max|err| on the final logits. Only the plugin's own demotion is
reverted: both passes agree on a `rocket.f16_demoted` tag, so a model that
authored its own f16 convolution is untouched.

`rocket-verify-conv-shapes` is the tripwire for anything like it: it errors if
a named convolution's output spatial extent disagrees with its own input,
filter, stride and dilation. It runs immediately before the match/rewrite
loop, while padding is still explicit and nothing has been tiled, so the
relation is exact there. It should never fire.

### ONNX models

Pin the batch dimension before importing. `iree-import-onnx` will happily
import a model whose batch is a symbolic `dim_param`, but the Rocket ABI fixes
batch at one and every matcher in the transform spec requires it, so a
dynamic-batch model compiles cleanly and offloads **nothing**. Clear each
`dim_param` on the graph inputs and outputs to 1, drop `graph.value_info`, and
re-run `shape_inference.infer_shapes` before `iree-import-onnx`.

int8 models quantized with ONNX Runtime's `quantize_dynamic` are supported.
They import as `onnx.ConvInteger`, which upstream torch-mlir cannot lower at
all -- `RocketExpandOnnxConvIntegerPass` in the compiler plugin supplies the
expansion, so these models need this repository's `iree-compile`, not a stock
one. Their convolutions reach the NPU through the `int8_accumulator` precision
(int8 in, int32 accumulator out, requantization bypassed); the transform spec
folds the activation zero point into a CPU-side correction first, because that
hardware mode is only validated for zero zero-points.

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

## Submodules

After cloning, initialize `iree-src` (and its own third-party submodules):

```sh
git submodule update --init --recursive
```
