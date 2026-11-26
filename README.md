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
offload disabled. Medians on four Cortex-A76 cores, NPU at 600 MHz
([full survey, 2026-11-25](agent-docs/survey-2026-11-25-main.md), including
the eight-core column and the per-op site counts):

| model | NPU | CPU baseline | |
|---|---|---|---|
| mobilenet_v2 fp16 | 9.6 ms | 54.3 ms | 5.6x |
| resnet50 fp16 | 52.0 ms | 894 ms | 17.2x |
| resnext50_32x4d fp16 | 55.6 ms | 502 ms | 9.0x |
| wide_resnet50_2 fp16 | 87.9 ms | 1904 ms | 21.7x |
| convnext_tiny fp16 | 94.1 ms | 252 ms | 2.7x |
| vit_b_16 fp16 | 202 ms | 1664 ms | 8.2x |
| vit_l_16 fp16 | 671 ms | 11430 ms | 17.0x |
| blip fp16 | 817 ms | 4818 ms | 5.9x |
| bert_base f32 | 106 ms | 1241 ms | 11.7x |
| bert_base_384 f32 | 412 ms | 4846 ms | 11.8x |

All twenty arms agreed with an ONNX Runtime oracle on the predicted class,
with no NPU timeouts. One caveat on the first row, because it is the one most
often quoted: MobileNetV2's baseline there is the model's fp16 export, and
IREE's CPU codegen is 1.76x worse at fp16 than f32 on an A76. Against the
**f32 import** -- the precision-matched comparison -- the same offload is
10.0 ms against 30.9, i.e. **3.1x**. Both are real; only a labelled one is
useful.

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
> job timeout can corrupt the submitting process. The patches are not
> published yet; [`agent-docs/board-setup.md`](agent-docs/board-setup.md)
> says what each does, what breaks without it, and how to check what your
> board has.

```sh
# 1. The last green nightly: an x86_64 Linux compiler and an aarch64 runtime.
run=$(gh run list --repo boydjohnson/iree-rocket --workflow=nightly.yml \
    --status=success --limit=1 --json databaseId -q '.[0].databaseId')
gh run download --repo boydjohnson/iree-rocket "$run" \
    -n iree-compile-x86_64 -n iree-runtime-aarch64
tar -xzf iree-compile-x86_64.tar.gz && tar -xzf iree-runtime-aarch64.tar.gz

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
    --device=rocket --device=local-task \
    --input=1x3x224x224xf16=@mobilenet_v2.fp16.input.bin \
    --output=@out.npy'
```

`audit` is the command worth running first on any model of your own: it prints
one line per convolution and matmul -- shape, precision, where it went, and
the decisive reason -- then reconciles that against the dispatches the
compiled module actually contains. A refusal names its class, so `shape` and
`semantics` are compiler work, `validation` means nobody has measured that
configuration yet, and `hardware` never closes.

## Known limitations

- **The board needs a patched kernel module**, as above. This is the single
  biggest obstacle to reproducing anything here, and the patches are not
  published yet.
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
upstream, and getting them there is the useful next step for anyone working on
that driver: **unconditional PPU completion** (upstream arms only the DPU
interrupts, so tiled pooling cannot complete), **NPU clock control** (the SoC
boots the NPU pinned at 200 MHz with no devfreq -- whether the right upstream
shape is an OPP table and a governor rather than the module parameters used
here is an open question), and **reset before IOMMU detach** on a job timeout.

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
| [`rocket-compiler-plugin`](rocket-compiler-plugin) | The IREE compiler target plugin ("Rocket"), in C++, plus the transform spec that matches and rewrites operations. |
| [`rocket-compiler`](rocket-compiler) | The compiler driver: applies the transform spec and the device flags it expects, and audits where every convolution and matmul went. |
| [`rocket-schema`](rocket-schema) | The FlatBuffers executable format shared by both sides. |
| [`tools`](tools) | ONNX import/export, the board regression gates, the model survey, the precision probe. |
| [`iree-build`](iree-build) | CMake configuration, and `iree-src`, IREE as a pinned submodule. |

Each component's detailed role is in
[`agent-docs/developer-tools.md`](agent-docs/developer-tools.md#repository-layout).

## Continuous integration

| Workflow | Status | What it gates |
|---|---|---|
| [`ci.yml`](.github/workflows/ci.yml) | [![CI](https://github.com/boydjohnson/iree-rocket/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/boydjohnson/iree-rocket/actions/workflows/ci.yml) | Every push and pull request: `cargo check`, `build`, `test` and `fmt` over the Rust workspace. Rust only -- it does not build IREE. |
| [`nightly.yml`](.github/workflows/nightly.yml) | [![Nightly artifacts](https://github.com/boydjohnson/iree-rocket/actions/workflows/nightly.yml/badge.svg?branch=main)](https://github.com/boydjohnson/iree-rocket/actions/workflows/nightly.yml) | Nightly, and the reason the prebuilt binaries above exist: the compiler with the Rocket plugin, the board runtime with the Rocket driver, the HAL CTS binaries and the crates' board test binaries. |

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
[developer tools](agent-docs/developer-tools.md), [CI](agent-docs/ci.md),
open questions and dated survey results. They are written for contributors
rather than for a first read.

## Building from source

Clone with submodules (`git submodule update --init --recursive`), then
`cargo test --workspace` for the Rust side, which needs no IREE build. The
three CMake configurations -- the compiler, the host runtime and the aarch64
runtime -- are in
[`agent-docs/developer-tools.md`](agent-docs/developer-tools.md#building).
