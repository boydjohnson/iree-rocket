# From an ONNX model to numbers on the board

An end-to-end pass with **no local IREE build**: the compiler and the board
runtime both come from the `Nightly artifacts` workflow, so the only thing
built here is the model. Export or bring an ONNX file, import it with its
dimensions pinned and an ONNX Runtime oracle beside it, compile the offload
arm *and* its baseline, run both on the RK3588, compare against the oracle,
then time them.

The build-it-yourself path, and the reasoning behind every flag used below,
are in [`developer-tools.md`](developer-tools.md). Measured results from this
procedure are at the end.

## What it needs

| Where | What |
|---|---|
| Host | Python with `onnx`, `onnxruntime`, `numpy` (a `uv venv` is enough), and `iree-import-onnx` on `PATH` from IREE's own Python packages. `gh` for the artifacts, `ssh`/`scp` to the board. |
| Host | x86_64 Linux with glibc 2.39 or newer. The nightly's floor is the runner image's glibc; see `max-glibc` in the CI rundown. |
| Board | RK3588 with the rocket kernel module loaded and `/dev/accel/accel0` present, reachable over SSH. Debian 13's glibc 2.41 is above the same floor. |
| Neither | A local IREE build. Nothing below runs `cmake`. |

`tools/export_onnx.py` additionally needs `torch` and `torchvision` (and
`transformers` for the BERT and BLIP entries); skip that step by bringing your
own `.onnx`.

## 1. Fetch the nightly artifacts

Take them from the last *successful* nightly rather than the latest run, so a
run that lost its runner mid-build does not send you to a half-finished one:

```sh
run=$(gh run list --repo boydjohnson/iree-rocket --workflow=nightly.yml \
    --status=success --limit=1 --json databaseId -q '.[0].databaseId')
gh run download --repo boydjohnson/iree-rocket "$run" \
    -n iree-compile-x86_64 -n iree-runtime-aarch64
tar -xzf iree-compile-x86_64.tar.gz
tar -xzf iree-runtime-aarch64.tar.gz
```

That is ~99 MB and ~9 MB respectively. `iree-compile-x86_64/` holds
`bin/rocket-compiler`, `bin/iree-compile`, `bin/iree-lld`,
`lib/libIREECompiler.so` and the transform spec under `share/rocket/`;
`rocket-compiler` finds all three relative to its own executable, so no
environment variable is needed. `iree-runtime-aarch64/bin/` holds
`iree-run-module` and `iree-benchmark-module` with the Rocket HAL driver
statically linked.

Each tarball carries a `VERSION.txt` naming the repo SHA, the `iree-src` pin
and the run URL. **Record it next to any number you keep** -- a measurement
without it cannot be reproduced or bisected later. Artifacts expire with the
run's retention, 14 days by default.

## 2. Get an ONNX model

From the registry, which is why a re-export is reproducible rather than
remembered:

```sh
tools/export_onnx.py mobilenet_v2 --out-dir models        # fp16, opset 17
tools/export_onnx.py resnet50 --out-dir models --precision f32
```

The registry covers `mobilenet_v2`, `resnet50`, `resnext50_32x4d`,
`wide_resnet50_2`, `vgg19`, `densenet121`, `efficientnet_b0`, `inception_v3`,
`convnext_tiny`, `vit_b_16`, `vit_l_16`, `blip`, `bert_base` and
`bert_base_384`. It writes `models/<model>.<precision>.onnx`, ONNX
Runtime-constant-folded in place.

Prefer the **f32** file when a model ships both. The transform spec demotes
convolutions and matmuls to f16 itself and restores f32 on whatever the match
loop leaves behind, so an f32 import chooses precision per operation rather
than for the whole graph, and it is the arm measured exact against the oracle.

## 3. Import it, pinned, with an oracle

```sh
tools/import_onnx.py models/mobilenet_v2.fp16.onnx --out-dir mnv2
```

Three things happen here, each of which has cost this repository a result
before: every `dim_param` is pinned, an ONNX Runtime oracle is computed
*before* the import rather than after, and the sample input is written as raw
`.bin` as well as `.npy`.

A registry export has static shapes already, so it needs no `--dim`. A model
downloaded from elsewhere usually does: `iree-import-onnx` will happily import
a graph whose batch is a `dim_param`, but the Rocket ABI fixes batch at one and
every matcher requires it, so a dynamic-batch model compiles cleanly and
offloads **nothing**. An unpinned `dim_param` is an error here rather than a
guess. Not every model leaves only batch symbolic -- ViT-B/16 as published
leaves all four input dims that way:

```sh
tools/import_onnx.py vit-base-patch16-224.onnx --out-dir vit \
    --dim batch_size=1 --dim num_channels=3 --dim height=224 --dim width=224
```

| `mnv2/` | What it is |
|---|---|
| `mobilenet_v2.fp16.mlir` | The import. Everything below takes this as input. |
| `mobilenet_v2.fp16.oracle.npy` | ONNX Runtime's output for the sample input, f32. The reference every run is scored against. |
| `mobilenet_v2.fp16.input.bin` | The sample input as raw bytes, which is what `--input=@file` reads. |
| `mobilenet_v2.fp16.input.npy` | The same tensor for host-side numpy work. |
| `mobilenet_v2.fp16.inputs.json` | Name, file, dtype and shape per graph input, **in the graph's own order** -- `iree-run-module` takes its inputs positionally, and a multi-input model whose arguments get swapped runs and produces plausible garbage. |

## 4. Compile both arms

An NPU-vs-CPU comparison needs a CPU arm built by the *same* pipeline, so
compile two modules from the one import. `--no-offload` neutralizes every
matcher and leaves the passes, the device topology and the placement pin
exactly as the offload arm sees them. A module from plain `iree-compile` is
not a baseline: it is still NCHW, because the transform spec's own
preprocessing is what makes a model NHWC, and IREE's CPU backend is **2.8x
slower** on NCHW MobileNetV2. Every NPU-vs-CPU number this repo quoted before
2026-09-04 was measured against that slower build (ISSUES.md M4).

```sh
C=iree-compile-x86_64/bin/rocket-compiler
M=mnv2/mobilenet_v2.fp16

"$C" compile --input "$M.mlir" --output "$M.npu.vmfb" \
    --llvmcpu-target-triple aarch64-linux-gnu
"$C" compile --input "$M.mlir" --output "$M.cpu.vmfb" --no-offload \
    --llvmcpu-target-triple aarch64-linux-gnu
```

Before running anything, ask where the model went and why:

```sh
"$C" audit --input "$M.mlir" --report-json "$M.placement.json"
```

`audit` prints one line per convolution and matmul candidate -- shape,
precision rung, the decision and the decisive reason -- then the dispatch-site
counts and a reconciliation of the two. Three counts, deliberately kept apart:
**candidates** are original operations, **dispatch sites** are what actually
ran where (the number that answers "how much of the model ran on the NPU"),
and **hardware jobs** are standalone NPU jobs, since one tiled dispatch
already runs several. A refusal names its class: `shape`, `semantics`,
`hardware` (permanent) or `validation` (nobody has measured it yet). Add
`--strict-offload` to fail the compile rather than fall back.

Sanity check worth running once per model: `audit --input "$M.mlir"
--no-offload` must report `0 dispatch site(s) -> rocket`.

## 5. Run both on the board

```sh
scp iree-runtime-aarch64/bin/iree-run-module "$M".{npu,cpu}.vmfb \
    "$M.input.bin" planck:~/mnv2/
ssh planck 'cd mnv2 && chmod +x iree-run-module && for arm in npu cpu; do
    ./iree-run-module --module=mobilenet_v2.fp16.$arm.vmfb \
      --function=main_graph --device=rocket --device=local-task \
      --input=1x3x224x224xf16=@mobilenet_v2.fp16.input.bin \
      --output=@mobilenet_v2.fp16.$arm.out.npy
done'
```

Both devices are given in both arms: the Rocket driver and IREE's own
`local-task` CPU backend, in that order, because the transform spec's device
topology expects them that way. The input flag is the typed
`shapexdtype=@file.bin` form -- IREE reads a typed operand's file as **raw
bytes**, so handing it a `.npy` shifts every element by the header and the
model quietly computes something else. The element type comes from
`inputs.json`'s `dtype`, the shape from its `shape`, and the function name
from the `func.func` in the `.mlir` (`main_graph` for an ONNX import).

Then score each arm against the oracle, not against zero and not against each
other:

```sh
scp planck:~/mnv2/'*.out.npy' mnv2/
python3 - <<'PY'
import numpy as np
ref = np.load("mnv2/mobilenet_v2.fp16.oracle.npy").astype(np.float32).ravel()
for arm in ("npu", "cpu"):
    got = np.load(f"mnv2/mobilenet_v2.fp16.{arm}.out.npy").astype(np.float32).ravel()
    print(f"{arm:4s} max|diff| {np.abs(got - ref).max():.4g}  "
          f"argmax {got.argmax()} vs {ref.argmax()}  ref sd {ref.std():.3f}")
PY
```

Read the error against the reference's own spread, not in the abstract. On
ViT that spread is a standard deviation of 0.90 and the offload arm's
max|err| on logits is 0.0060, which is the f16 ABI doing its job rather than
a fault; the first thing to check either way is that the argmax agrees.

## 6. Time them

`iree-benchmark-module`, same two devices, with two rules that are not
optional:

```sh
scp iree-runtime-aarch64/bin/iree-benchmark-module planck:~/mnv2/
ssh planck 'cd mnv2 && chmod +x iree-benchmark-module && for arm in npu cpu; do
    ./iree-benchmark-module --module=mobilenet_v2.fp16.$arm.vmfb \
      --function=main_graph --device=rocket --device=local-task \
      --task_topology_cpu_ids=4,5,6,7 \
      --input=1x3x224x224xf16=@mobilenet_v2.fp16.input.bin \
      --benchmark_repetitions=6 --benchmark_min_time=1x
done'
```

**The core allocation is `--task_topology_cpu_ids`, never `taskset`.** IREE
builds its task topology from the machine's cpuinfo rather than from the
process affinity mask, so a `taskset -c 4-7` run thinks it has eight cores and
puts about two of them to work -- and the offload deficit it reports is
roughly double what a full machine sees (ISSUES.md P8). Time each arm at the
same allocation, and report more than one: `4,5,6,7` is the A76 cluster,
`0,1,2,3,4,5,6,7` the whole machine, and the two can disagree about which arm
wins.

**Discard the first repetition.** An offloaded arm packs its weights on the
first invocation only, which on a ResNet-scale model was a ~1021 ms first
sample against a 124 ms steady state (`tools/model_survey.py`'s own note;
`developer-tools.md` records 945 ms for the same comparison, so take the
magnitude rather than the digits) -- either way large enough to swallow the
thing being measured. Ask for one more repetition than you want and take the
median of the rest.

Board state belongs with the numbers, because it moves them: the `cpufreq`
governor, the NPU's clock and kernel-module variant, and the NPU IRQ
affinity. `tools/model_survey.py` reads all of it off the board and records it
in `result.json`; by hand, record it yourself.

## 7. Where the time went

```sh
ssh planck 'cd mnv2 && ROCKET_PROFILE=1 ./iree-run-module \
    --module=mobilenet_v2.fp16.npu.vmfb --function=main_graph \
    --device=rocket --device=local-task \
    --input=1x3x224x224xf16=@mobilenet_v2.fp16.input.bin'
```

At exit the driver prints a phase table --
`phase / calls / total ms / avg ms / max ms / MB/s` -- then `wall`, then
`host`, `npu` and the **npu share**, then a per-op breakdown. The phases that
matter are the host-side ones the hardware timers never see: `pack.input`
(NHWC -> NC1HWC2), `pack.weights`, `compact` (the DPU's atomic slots back into
IREE's dense buffer) and `wait.npu` itself, with `outside` being IREE's
non-driver work, i.e. the CPU dispatches. Two conclusions this repo reached
from that table rather than from wall clock: output compaction once cost more
than every hardware job put together, and on one model the device was 10% of
the wall, which ruled out the clock and the IRQ placement as explanations in a
single run. `ROCKET_PROFILE=trace` adds a line per phase as it happens.

The `weight cache`, `compaction:`, `host time by cpu` and `overlap` lines say
whether the caches and the worker placement are doing their jobs;
[`developer-tools.md`](developer-tools.md#profiling-a-run) has every knob that
changes them.

## 8. The whole thing in one command

`tools/model_survey.py` is steps 2 through 6 as one resumable run -- export,
import, both arms, correctness against the oracle, and a timing per core
allocation -- writing a machine-readable `result.json` next to the artifacts.
Point its four tool paths at the unpacked nightly and it needs no local build
either:

```sh
tools/model_survey.py run --model mobilenet_v2 --board planck \
    --rocket-compiler "$PWD/iree-compile-x86_64/bin/rocket-compiler" \
    --compiler-lib "$PWD/iree-compile-x86_64/lib/libIREECompiler.so" \
    --board-runtime "$PWD/iree-runtime-aarch64/bin/iree-run-module" \
    --board-benchmark "$PWD/iree-runtime-aarch64/bin/iree-benchmark-module"

tools/model_survey.py summarize
```

Each stage is skipped when its output already exists, so an interrupted run
resumes and a re-timing costs only the timing. It discards the warm-up
repetition, defaults to the two core allocations above, and reads the board
conditions itself. A model outside the registry brings its own file and its
own pinning: `--onnx qwen3.onnx --dim batch_size=1 --dim sequence_length=128`.

`summarize` pools every `result.json` and prints one histogram of *why*
candidates stayed on the CPU, ranked and tagged with each reason's class. That
is the output worth reading across models: a wall-clock number says one model
got faster, while the histogram says which compiler lever is worth building
next and roughly what it is worth -- `shape` and `semantics` close with
compiler work, `validation` needs a board measurement, `hardware` never
closes.

## Measured results

Ten models through exactly this procedure, from
[`survey-2026-11-25-main.md`](survey-2026-11-25-main.md): `main` at 3d04335,
board `orangepi` (RK3588), Armbian 7.3.5, `rocket.ko` PPU + 600 MHz clock +
088, governor `performance`. Medians with the warm-up discarded; `x` is the
offload arm against its own `--no-offload` baseline at the same allocation.
All 20 arms agreed with the f32 oracle on argmax, with 0 NPU timeouts.

| model | NPU sites | CPU sites | NPU 4xA76 | CPU 4xA76 | x | NPU 8c | CPU 8c | x |
|---|---|---|---|---|---|---|---|---|
| mobilenet_v2 fp16 | 54 | 5 | 9.6 ms | 54.3 ms | 5.64 | 9.8 ms | 57.7 ms | 5.91 |
| resnet50 fp16 | 56 | 21 | 52.0 ms | 894 ms | 17.2 | 52.6 ms | 1011 ms | 19.2 |
| resnext50_32x4d fp16 | 56 | 21 | 55.6 ms | 502 ms | 9.03 | 57.3 ms | 685 ms | 12.0 |
| wide_resnet50_2 fp16 | 55 | 9 | 87.9 ms | 1904 ms | 21.7 | 87.6 ms | 3012 ms | 34.4 |
| convnext_tiny fp16 | 60 | 216 | 94.1 ms | 252 ms | 2.68 | 98.1 ms | 252 ms | 2.57 |
| vit_b_16 fp16 | 50 | 249 | 202 ms | 1664 ms | 8.24 | 192 ms | 1327 ms | 6.91 |
| vit_l_16 fp16 | 98 | 489 | 671 ms | 11430 ms | 17.0 | 686 ms | 13871 ms | 20.2 |
| blip fp16 | 170 | 621 | 817 ms | 4818 ms | 5.90 | 740 ms | 4005 ms | 5.41 |
| bert_base f32 import | 72 | 199 | 106 ms | 1241 ms | 11.7 | 105 ms | 1356 ms | 12.9 |
| bert_base_384 f32 import | 72 | 199 | 412 ms | 4846 ms | 11.8 | 458 ms | 3875 ms | 8.46 |

Four things to read with that table:

**The ratio is against `--no-offload`, and the precision of the baseline is
part of the claim.** MobileNetV2's 5.64x is against the **fp16 export**
baseline, which is the arm the survey builds for it. Against the **f32
import** -- the precision-matched comparison, and the fairer baseline -- the
same offload is 10.0 ms against 30.9, i.e. **3.1x** (ISSUES.md P7). Both
numbers are sound; an unlabelled one is not, and this repository has had to
correct exactly that confusion.

**Sites, not executables.** The transform spec routes every matched
convolution through a handful of fixed executables, so an executable count
understates the NPU badly, while IREE deduplicates identical CPU dispatches,
which overstates the CPU side.

**A model's CPU column is where the next work is.** The convolution networks
leave almost nothing behind (mobilenet_v2: 4 element-wise and 1 transpose),
while the transformers leave hundreds of element-wise, reduction, `rsqrt`,
`softmax`, transpose and `batch_matmul` sites -- which is why their ratios are
large and their absolute times are not. Of every candidate in all ten models,
exactly one was *declined*: BLIP's 768x30524 LM head, on the Cout cap
(`unvalidated_configuration`).

**Both allocations, because they disagree.** wide_resnet50_2 goes 21.7x to
34.4x between four cores and eight, and BLIP's CPU arm is 9% *faster* on
eight. Quoting one allocation without saying which is how a number becomes
unreproducible.
