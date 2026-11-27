# iree-rocket

[![CI](https://github.com/boydjohnson/iree-rocket/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/boydjohnson/iree-rocket/actions/workflows/ci.yml)
[![Nightly artifacts](https://github.com/boydjohnson/iree-rocket/actions/workflows/nightly.yml/badge.svg?branch=main)](https://github.com/boydjohnson/iree-rocket/actions/workflows/nightly.yml)

An [IREE](https://github.com/iree-org/iree) backend for the RK3588 NPU, built
on the **mainline `accel/rocket` kernel driver** -- no RKNN SDK, no vendor
runtime, no closed blobs anywhere in the path. An ONNX or MLIR model is
compiled ahead of time into a `.vmfb` and executed by `iree-run-module` with
the NPU registered as an IREE HAL device, so a model's convolutions and
matmuls run on the accelerator while everything else runs on IREE's own CPU
backend in the same process.

Apache-2.0 with the LLVM exception, like IREE itself.

## Where the project stands

Ten ONNX models compile, offload and run correctly on an RK3588, each
measured against a CPU baseline built by the **same compiler pipeline** with
offload disabled. Medians on four Cortex-A76 cores, NPU at 600 MHz, on IREE
v3.14.0 ([the full run](agent-docs/iree-3.14-bump.md#10-model-survey-vs-survey2026-11-25-main-orangepi-600-mhz-performance),
with the eight-core column, the per-op site counts and the previous
toolchain's numbers beside each):

| model | NPU | CPU baseline | |
|---|---|---|---|
| mobilenet_v2 fp16 | 10.3 ms | 41.8 ms | 4.1x |
| resnet50 fp16 | 51.6 ms | 818 ms | 15.9x |
| resnext50_32x4d fp16 | 59.7 ms | 455 ms | 7.6x |
| wide_resnet50_2 fp16 | 86.1 ms | 2083 ms | 24.2x |
| convnext_tiny fp16 | 90.2 ms | 247 ms | 2.7x |
| vit_b_16 fp16 | 189 ms | 1667 ms | 8.8x |
| vit_l_16 fp16 | 641 ms | 11385 ms | 17.8x |
| blip fp16 | 779 ms | 4980 ms | 6.4x |
| bert_base f32 | 103 ms | 1397 ms | 13.6x |
| bert_base_384 f32 | 403 ms | 5248 ms | 13.0x |

All twenty arms agreed with an ONNX Runtime oracle on the predicted class,
with no NPU timeouts, and the candidate, accepted and NPU-site counts are
identical to the previous toolchain's on all ten models.

Two things to read with that table. **The ratios move when either arm moves,
and the IREE 3.14 bump moved the CPU arm more than the NPU arm.** Every NPU
arm is equal or faster than it was on 3.12 (transformers 3-7% faster), while
the CNN baselines got 9-23% faster on their own, so MobileNetV2's ratio fell
from 5.6x to 4.1x on a *faster CPU*, not a slower NPU.

**The precision of the baseline is part of the claim.** MobileNetV2's
baseline above is the model's own fp16 export, and IREE's CPU codegen is
1.76x worse at fp16 than f32 on an A76. The precision-matched comparison is
against the f32 import, which was **3.1x** (10.0 ms against 30.9) when it was
last measured, on the 3.12 toolchain -- it has not been re-run on 3.14, where
both arms have moved. Quote it with its date, or quote the fp16 row and say
so.

**On the NPU today:** dense and depthwise 2-D convolution at stride 1 and 2,
matmul (including `matvec`/`vecmat`), max/min/average pooling, in fp16 and
int8, with constant filters packed at compile time and NPU-to-NPU dispatch
chaining that keeps intermediate tensors in the hardware's own layout.
Two-tensor element-wise and batched attention are implemented and **off by
default** because they are measured slower -- the flags exist so either arm
can be measured rather than argued about.

**Not reached yet:** softmax, layernorm, the reductions and transposes around
them, and element-wise at a profitable cost. [`ROADMAP.md`](ROADMAP.md) is the
operation-by-operation account of what the hardware could run, what it would
take, and in what order.

## Try it on a board

The compiler and the board runtime both come prebuilt, so nothing below
builds IREE. Full version, with the ONNX Runtime oracle and the baseline arm,
in [`agent-docs/onnx-to-numbers.md`](agent-docs/onnx-to-numbers.md).

> **First, the board.** Every result above was measured on the mainline
> `rocket` module **plus three out-of-tree patches** (PPU completion, the NPU
> clock parameters, reset-before-IOMMU-detach). On a stock module, pooling
> jobs are killed by the watchdog at ~500 ms, timings roughly double, and a
> job timeout can corrupt the submitting process. All three are in
> [`kernel-patches/`](kernel-patches) with instructions;
> [`agent-docs/board-setup.md`](agent-docs/board-setup.md) says what each
> does, what breaks without it, and how to check what your board has.

```sh
# 1. The last green nightly: an x86_64 Linux compiler and an aarch64 runtime.
run=$(gh run list --repo boydjohnson/iree-rocket --workflow=nightly.yml \
    --status=success --limit=1 --json databaseId -q '.[0].databaseId')
gh run download --repo boydjohnson/iree-rocket "$run" \
    -n iree-compile-x86_64 -n iree-runtime-aarch64
# Two -n flags put each artifact in its own directory; each tarball unpacks
# into that same directory name.
tar -xzf iree-compile-x86_64/iree-compile-x86_64.tar.gz
tar -xzf iree-runtime-aarch64/iree-runtime-aarch64.tar.gz

# 2. An ONNX model, imported with its shapes pinned and an oracle beside it.
tools/export_onnx.py mobilenet_v2 --out-dir models
tools/import_onnx.py models/mobilenet_v2.fp16.onnx --out-dir mnv2

# 3. Compile for the board, and ask where each convolution and matmul went.
iree-compile-x86_64/bin/rocket-compiler compile \
    --input mnv2/mobilenet_v2.fp16.mlir --output mnv2/mnv2.npu.vmfb \
    --llvmcpu-target-triple aarch64-linux-gnu
iree-compile-x86_64/bin/rocket-compiler audit \
    --input mnv2/mobilenet_v2.fp16.mlir

# 4. Run it on the board.
scp iree-runtime-aarch64/bin/iree-run-module mnv2/mnv2.npu.vmfb \
    mnv2/mobilenet_v2.fp16.input.bin board:~/
ssh board './iree-run-module --module=mnv2.npu.vmfb --function=main_graph \
    --device=rocket --device=local-task --task_topology_cpu_ids=4,5,6,7 \
    --input=1x3x224x224xf16=@mobilenet_v2.fp16.input.bin \
    --output=@out.npy'
```

`audit` is the command worth running first on any model of your own: it prints
one line per convolution and matmul -- shape, precision, where it went, and
the decisive reason -- then reconciles that against the dispatches the
compiled module actually contains. A refusal names its class, so `shape` and
`semantics` are compiler work, `validation` means nobody has measured that
configuration yet, and `hardware` never closes.

To run models from your own program instead of `iree-run-module`, the
nightly's `iree-rocket-runtime-aarch64` artifact is the runtime as a C/C++
library; [`rocket-runtime`](rocket-runtime) has the API and a minimal example,
and [`rocket-runtime/rust`](rocket-runtime/rust) the Rust bindings.

## Known limitations

- **The board needs a patched kernel module.** All three changes are in
  [`kernel-patches/`](kernel-patches), with what each fixes and how to apply
  it. None is upstream yet, so this remains the biggest obstacle to
  reproducing anything here.
- **There is no code generator for this target.** The NPU runs only the
  dispatches the transform spec creates, so an operation reaches it only if a
  matcher claims it; unclaimed work stays on the CPU and still runs correctly.
- **An offloaded convolution runs its inputs at f16** (the ABI is
  f16-in/f32-accumulate). Where an f32 stem feeds int8 quantization, as in
  MobileNetV2, that is ~0.35-0.42 max|err| on the final logits, with the
  predicted class stable except on near-ties.
- **Two known-wrong cases are open:** a HuggingFace float16-converted ViT
  mis-imports through torch-mlir -- on the host CPU, with no NPU involved
  (ISSUES.md C14) -- and VGG int8 aborts under a repeated benchmark loop
  (ISSUES.md C11).
- **[`LIMITS.md`](LIMITS.md) records measurements, not capability.** A shape
  outside it is untested rather than unsupported; the register fields are
  almost always wider than the constants.
- **No tagged release yet**, and nightly artifacts expire after 14 days, so a
  link to a nightly run is not a durable reference.

## How this fits upstream

Nothing here is a fork. The compiler side is an out-of-tree IREE target
plugin loaded through `IREE_CMAKE_PLUGIN_PATHS`, and the runtime side is an
external HAL driver registered through IREE's own
`iree_register_external_hal_driver()`. IREE itself is a pinned commit of
`iree-org/iree` as a submodule, with no patches applied by the build.

The kernel side is mainline `accel/rocket` plus three patches that are not
upstream, carried in [`kernel-patches/`](kernel-patches) -- two of them Greg
Tomko's, from the external [`gregordinary/patches`](https://github.com/gregordinary/patches)
series, rebased onto Armbian 7.3.5 and carried to 7.3.6. Getting all three upstream is the useful
next step for anyone working on that driver. **Unconditional
PPU completion**: upstream arms only the DPU interrupts, so a standalone
PPU-flying pooling job never reaches the completion path and is retired by
the watchdog instead. **NPU clock control**: the SoC boots the NPU pinned at
200 MHz with no devfreq, and whether the right upstream shape is an OPP table
and a governor rather than the module parameters used here is an open
question. **Reset before IOMMU detach** on a job timeout, so in-flight DMA
cannot land in pages userspace has already freed.

What would help most from the hardware side is evidence for the
configurations the admission policy currently refuses as *unmeasured* rather
than *impossible*: register documentation or vendor captures for the shape
classes listed in [`LIMITS.md`](LIMITS.md), since every bound there is the
extent of an actual measurement and the register encodings are wider.

## Repository layout

| Path | Role |
|---|---|
| [`rocket-core`](rocket-core) | The planner: convolution/matmul descriptors, hardware limits, layout geometry, CBUF partitioning and tiling. Pure Rust, no dependencies, shared by the compiler and the runtime. |
| [`rocket-plan-ffi`](rocket-plan-ffi) | A versioned C ABI over that planner, so the compiler asks the same planner -- and packs the same coefficient bytes -- the runtime does. |
| [`iree-rocket-hal`](iree-rocket-hal) | ioctl/mmap access to the NPU and register command building. |
| [`rocket-hal-driver`](rocket-hal-driver) | The IREE HAL driver, statically linked into IREE's tools. Includes HAL CTS wiring. |
| [`rocket-runtime`](rocket-runtime) | The runtime as a library for C, C++ and Rust programs: `rocket/runtime.h`, a session over the NPU+CPU device pair every module needs (IREE's own session API takes one device), packaged as `libiree_rocket_runtime.{so,a}` with pkg-config and CMake files, plus Rust bindings in [`rocket-runtime/rust`](rocket-runtime/rust). |
| [`rocket-compiler-plugin`](rocket-compiler-plugin) | The IREE compiler target plugin ("Rocket"), in C++, plus the transform spec that matches and rewrites operations. |
| [`rocket-compiler`](rocket-compiler) | The compiler driver: applies the transform spec and the device flags it expects, and audits where every convolution and matmul went. |
| [`rocket-schema`](rocket-schema) | The FlatBuffers executable format shared by both sides. |
| [`tools`](tools) | ONNX import/export, the board regression gates, the model survey, the precision probe. |
| [`kernel-patches`](kernel-patches) | The out-of-tree `accel/rocket` patches the measured configuration runs, for Armbian 7.3.6 (and 7.3.5, unchanged). GPL-2.0, unlike the rest of this repository, and two of the three come from [`gregordinary/patches`](https://github.com/gregordinary/patches). |
| [`iree-build`](iree-build) | CMake configuration, and `iree-src`, IREE as a pinned submodule. |

Each component's detailed role is in
[`agent-docs/developer-tools.md`](agent-docs/developer-tools.md#repository-layout).

## Continuous integration

| Workflow | Status | What it gates |
|---|---|---|
| [`ci.yml`](.github/workflows/ci.yml) | [![CI](https://github.com/boydjohnson/iree-rocket/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/boydjohnson/iree-rocket/actions/workflows/ci.yml) | Every push and pull request: `cargo check`, `build`, `test` and `fmt` over the Rust workspace. Rust only -- it does not build IREE. |
| [`nightly.yml`](.github/workflows/nightly.yml) | [![Nightly artifacts](https://github.com/boydjohnson/iree-rocket/actions/workflows/nightly.yml/badge.svg?branch=main)](https://github.com/boydjohnson/iree-rocket/actions/workflows/nightly.yml) | Nightly, and the reason the prebuilt binaries above exist: the compiler with the Rocket plugin, the board runtime with the Rocket driver, the embeddable runtime library, the HAL CTS binaries and the crates' board test binaries. |

The hardware tests are `#[ignore]`d in CI -- nothing in GitHub Actions has an
NPU -- so correctness on the device is gated by three end-to-end regression
suites (convolution, matmul, pooling) run by hand against a real board, each
comparing against independently compiled CPU results, most cases exactly
rather than within a tolerance. Job-by-job detail is in
[`agent-docs/ci.md`](agent-docs/ci.md).

## Documents

[`LIMITS.md`](LIMITS.md) is what the stack is *measured* to do and which layer
enforces each bound. [`ISSUES.md`](ISSUES.md) is every open defect with the
measurements behind it. [`ROADMAP.md`](ROADMAP.md) is which operations the
hardware could reach and in what order; [`COMPILER_ROADMAP.md`](COMPILER_ROADMAP.md)
is the compiler-side plan, with [`DYNAMIC_SHAPES.md`](DYNAMIC_SHAPES.md) and
[`SOFTMAX.md`](SOFTMAX.md) as single-topic notes.

[`agent-docs/`](agent-docs) holds the detailed working documents --
[board setup](agent-docs/board-setup.md), the
[ONNX-to-board walkthrough](agent-docs/onnx-to-numbers.md),
[developer tools](agent-docs/developer-tools.md), [CI](agent-docs/ci.md), the
[IREE 3.14 bump record](agent-docs/iree-3.14-bump.md) behind the numbers
above, the [Mali/panvk Vulkan spike](agent-docs/vulkan-panvk-spike.md),
open questions and dated survey results. They are written for
contributors rather than for a first read.

## Building from source

Clone with submodules (`git submodule update --init --recursive`), then
`cargo test --workspace` for the Rust side, which needs no IREE build. The
three CMake configurations -- the compiler, the host runtime and the aarch64
runtime -- are in
[`agent-docs/developer-tools.md`](agent-docs/developer-tools.md#building).
