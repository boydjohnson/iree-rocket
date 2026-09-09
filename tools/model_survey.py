#!/usr/bin/env python3
"""Drive one model end to end -- export, import, compile, correctness, timing.

The repository already has a gate for *shapes* (`tools/e2e_conv_regression.py`)
and one for *matmul geometry* (`tools/e2e_matmul_regression.py`). This is the
gate for *models*: it takes a whole network, runs it through the product
pipeline the README documents, and reports three things that only a real model
can say.

**Where the model ran**, from `rocket-compiler --report-json`: candidates,
accepted, dispatch sites on each side, and the reason code for every candidate
that stayed on the CPU. This is the survey's real product. A wall-clock number
says one model got faster; a reason histogram aggregated over a dozen models
says which *compiler* lever is worth building next, ranked by how many real
candidates it would unblock. `summarize` prints exactly that.

**Whether it is right**, against an ONNX Runtime oracle built from the same
`.onnx` file by `tools/import_onnx.py`. Both arms are compared against the
oracle, not just the NPU one, because an IREE-side import bug shows up in the
CPU arm too and blaming the NPU for it has cost this repository a day before
(ISSUES.md C14).

**What it costs**, against the `--no-offload` arm and never against a stock
`iree-compile` build. A module built by plain `iree-compile` never runs the
spec's channels-last conversion and is 2.8x slower for that reason alone, so a
speedup quoted against one is measuring the layout, not the NPU (ISSUES.md M4).

## Conditions, which are part of the measurement

Board timings are quoted with their core allocation because the same arm moves
by 2.4x across allocations (ISSUES.md P8). The knob is
`--task_topology_cpu_ids`, *not* `taskset`: IREE builds its task topology from
the machine's cpuinfo rather than from the process affinity mask, so a
`taskset -c 4-7` run thinks it has eight cores and puts about two to work. Each
allocation is a column in the output, and the governor, the NPU IRQ affinity
and the NPU's runtime state are recorded next to the numbers rather than
assumed.

Every measurement waits for the NPU to go quiet first. A hung job leaves the
device sick across processes for seconds, which shows up as the *first* case
of the next run failing for reasons of its own (ISSUES.md, the wedge protocol).

## Usage

    tools/model_survey.py run --model wide_resnet50_2 --board planck
    tools/model_survey.py run --model vit_l_16 --board planck --arm npu --arm cpu
    tools/model_survey.py summarize

    # A model that is not in tools/export_onnx.py's registry: bring the .onnx
    # and pin whatever dims it leaves symbolic.
    tools/model_survey.py run --model qwen3 --onnx qwen3.onnx \
        --dim batch_size=1 --dim sequence_length=128 --board planck

Stages are skipped when their outputs already exist, so an interrupted run
resumes and a re-timing costs nothing but the timing. `--force STAGE` redoes
one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGES = ("export", "import", "compile", "run", "bench")

# An arm is a name and the `rocket-compiler` flags that build it. `cpu` is the
# only correct baseline: same pipeline, same device topology, same placement
# pin, every matcher neutralized. The two opt-in arms exist because the README
# says to measure them rather than argue about them -- both were losses when
# last measured, on a different per-dispatch cost than today's.
ARMS: dict[str, list[str]] = {
    "npu": [],
    "cpu": ["--no-offload"],
    "npu-bmm": ["--batch-matmul"],
    "npu-ew": ["--elementwise"],
}

DEFAULT_ARMS = ("npu", "cpu")
DEFAULT_CPU_IDS = ("4,5,6,7", "0,1,2,3,4,5,6,7")

# numpy dtype name -> the element type `--input=` wants in front of `=@file`.
IREE_ELEMENT_TYPES = {
    "float16": "f16",
    "float32": "f32",
    "float64": "f64",
    "int8": "i8",
    "int16": "i16",
    "int32": "i32",
    "int64": "i64",
    "uint8": "i8",
    "bool": "i1",
}


# --------------------------------------------------------------------------
# process plumbing


def command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print(f"  $ {command_text(command)}", flush=True)
    subprocess.run(command, check=True, env=env)


def capture(command: list[str], *, check: bool = True) -> str:
    result = subprocess.run(
        command, check=check, capture_output=True, text=True
    )
    return result.stdout


def capture_both(command: list[str]) -> tuple[int, str]:
    print(f"  $ {command_text(command)}", flush=True)
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# board


def board_conditions(host: str) -> dict[str, str]:
    """Reads the platform state that biases every number below.

    One ssh round trip, and recorded verbatim rather than interpreted: a
    result whose conditions are missing cannot be compared with the next one,
    and the governor in particular has silently moved between sessions.
    """
    probe = "; ".join(
        [
            'echo "governor=$(for c in 0 4 6; do printf "cpu%s:%s/%s " $c '
            '"$(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor)" '
            '"$(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_min_freq)"; done)"',
            'echo "npu_irq=$(for i in 82 83 84; do printf "%s:%s " $i '
            '"$(cat /proc/irq/$i/smp_affinity_list 2>/dev/null)"; done)"',
            'echo "npu_status=$(cat /sys/devices/platform/*.npu/power/runtime_status '
            '2>/dev/null | tr "\\n" " ")"',
            'echo "npu_clock=$(cat /sys/kernel/debug/clk/clk_npu/clk_rate 2>/dev/null)"',
            'echo "uname=$(uname -r)"',
        ]
    )
    text = capture(["ssh", host, probe], check=False)
    conditions = {}
    for line in text.splitlines():
        key, _, value = line.partition("=")
        conditions[key.strip()] = value.strip()
    return conditions


def wait_for_quiet_npu(host: str, budget_seconds: int = 30) -> None:
    """Blocks until every NPU core's runtime-PM state reads suspended.

    Lifted from `tools/e2e_conv_regression.py`, and for the same reason: the
    sick-device state crosses processes, so a benchmark started right after a
    hung job measures the recovery rather than the model.
    """
    probe = (
        "for i in $(seq 1 %d); do "
        "s=$(cat /sys/devices/platform/*.npu/power/runtime_status 2>/dev/null); "
        '[ -n "$s" ] || { echo missing; exit 0; }; '
        'case "$s" in *active*) sleep 1;; *) echo quiet; exit 0;; esac; '
        "done; echo busy"
    ) % budget_seconds
    state = (capture(["ssh", host, probe], check=False).splitlines() or ["missing"])[
        -1
    ].strip()
    if state == "busy":
        print(
            f"  WARNING: the NPU was still active after {budget_seconds}s. A job "
            "hung recently and the device may still be sick; re-run from a quiet "
            "board before believing what follows.",
            flush=True,
        )


def remote_path(remote_dir: str) -> str:
    """Strips a leading `~/`, because scp will not expand one.

    Modern OpenSSH moves data over the SFTP protocol, which has no shell to
    expand a tilde: `scp x host:~/dir/` fails with `dest open "dir/"` while
    `ssh host mkdir -p '~/dir'` cheerfully creates a directory *named* `~`.
    An SFTP relative path is already resolved against the login home, so
    dropping the prefix is both the fix and what the caller meant.
    """
    return remote_dir[2:] if remote_dir.startswith("~/") else remote_dir


# Rust the board binaries statically link. A binary older than any of it is
# running last week's driver, which is not a hypothetical: ViT-L/16's 1024x4096
# MLP aborted on the board with "output channels must be 1..=3584" from an
# `iree-benchmark-module` built the day before the ceiling moved to 4096, while
# the freshly built `iree-run-module` beside it ran the same module correctly.
DRIVER_SOURCES = ("rocket-core/src", "iree-rocket-hal/src", "rocket-hal-driver/src")


def check_binary_is_current(binary: Path) -> None:
    """Warns when a board binary predates the driver source it links."""
    if not binary.exists():
        raise SystemExit(f"error: {binary} does not exist; build it first")
    newest = 0.0
    newest_path = None
    for directory in DRIVER_SOURCES:
        for source in (ROOT / directory).rglob("*.rs"):
            stamp = source.stat().st_mtime
            if stamp > newest:
                newest, newest_path = stamp, source
    if newest > binary.stat().st_mtime:
        print(
            f"  WARNING: {binary.name} is older than {newest_path}. It "
            "statically links the Rust driver, so it is running the old one -- "
            "rebuild it (cmake --build iree-build/host-aarch64/build --target "
            f"{binary.name}) before believing a result.",
            flush=True,
        )


def sync(host: str, remote_dir: str, paths: list[Path]) -> None:
    """Copies only what the board does not already have, by content.

    A ViT-L arm is 600 MB and there are two of them; re-copying an unchanged
    pair on every timing run costs more than the timing. The remote digests
    come back in one round trip, and a stale board binary is the failure this
    guards against -- `iree-run-module` statically links the Rust driver, so a
    board still running last week's copy silently tests last week's code.
    """
    remote_dir = remote_path(remote_dir)
    run(["ssh", host, f"mkdir -p {shlex.quote(remote_dir)}"])
    names = " ".join(shlex.quote(f"{remote_dir}/{p.name}") for p in paths)
    remote = capture(
        ["ssh", host, f"md5sum {names} 2>/dev/null || true"], check=False
    )
    have = {}
    for line in remote.splitlines():
        digest, _, name = line.partition("  ")
        have[Path(name.strip()).name] = digest.strip()

    stale = [p for p in paths if have.get(p.name) != md5(p)]
    for path in stale:
        size = path.stat().st_size / 1e6
        print(f"  copying {path.name} ({size:.1f} MB)", flush=True)
        run(["scp", "-q", str(path), f"{host}:{remote_dir}/"])
    fresh = len(paths) - len(stale)
    if fresh:
        print(f"  {fresh} file(s) already current on the board", flush=True)


# --------------------------------------------------------------------------
# model plumbing


def function_signature(mlir: Path) -> tuple[str, int]:
    """Returns the entry function's name and how many results it has.

    Read by streaming rather than by loading the file: an imported ViT-L is
    over a gigabyte of textual IR, nearly all of it constants, and the
    signature is in the first few lines.
    """
    pattern = re.compile(r"func\.func @([A-Za-z0-9_.$]+)\(")
    with mlir.open("r", errors="replace") as handle:
        for index, line in enumerate(handle):
            match = pattern.search(line)
            if not match:
                if index > 20000:
                    break
                continue
            name = match.group(1)
            arrow = line.rfind("->")
            results = 1
            if arrow != -1:
                tail = line[arrow:]
                results = max(1, tail.count("tensor<"))
            return name, results
    raise SystemExit(f"error: no func.func found in {mlir}")


def input_flags(work: Path, stem: str, remote: bool) -> list[str]:
    """Builds one `--input=` per graph input, in the graph's own order.

    The typed `shapexdtype=@file.bin` form, never a bare `@file.npy`: IREE
    reads a typed operand's file as *raw bytes*, so handing it an npy shifts
    every element by the header and the model quietly computes something else
    (the trap is in the repository's notes twice).
    """
    manifest = json.loads((work / f"{stem}.inputs.json").read_text())
    flags = []
    for entry in manifest:
        dtype = IREE_ELEMENT_TYPES.get(entry["dtype"])
        if dtype is None:
            raise SystemExit(f"error: no IREE element type for {entry['dtype']}")
        shape = "x".join(str(d) for d in entry["shape"])
        name = entry["file"] if remote else str(work / entry["file"])
        flags.append(f"--input={shape}x{dtype}=@{name}")
    return flags


def compare(candidate: Path, reference: Path) -> dict[str, object]:
    """Scores one output against another, in float32 whatever they arrived as.

    Reports both an absolute error and the spread of the reference, because
    an absolute error alone is meaningless across models: 0.29 was a *pass*
    on BLIP's logits (sd 2.00) and would be a gross failure on a softmax.
    """
    import numpy as np

    a = np.asarray(np.load(candidate), dtype=np.float32).reshape(-1)
    b = np.asarray(np.load(reference), dtype=np.float32).reshape(-1)
    if a.shape != b.shape:
        return {"ok": False, "detail": f"shape {a.shape} vs {b.shape}"}
    error = np.abs(a - b)
    order_a = np.argsort(a)[::-1][:5]
    order_b = np.argsort(b)[::-1][:5]
    return {
        "ok": True,
        "max_abs_err": float(error.max()),
        "mean_abs_err": float(error.mean()),
        "reference_sd": float(b.std()),
        "argmax": int(order_b[0]),
        "argmax_matches": bool(order_a[0] == order_b[0]),
        "top5_matches": bool(list(order_a) == list(order_b)),
    }


# --------------------------------------------------------------------------
# stages


def stage_export(model: str, work: Path, onnx_path: Path | None, precision: str) -> Path:
    if onnx_path is not None:
        return onnx_path
    target = work / f"{model}.{precision}.onnx"
    if target.exists():
        print(f"  {target.name} present, skipping export")
        return target
    run(
        [
            sys.executable,
            str(ROOT / "tools/export_onnx.py"),
            model,
            "--out-dir",
            str(work),
            "--precision",
            precision,
        ]
    )
    return target


def stage_import(
    onnx_path: Path, work: Path, dims: list[str], oracle: str
) -> Path:
    stem = onnx_path.stem
    mlir = work / f"{stem}.mlir"
    if mlir.exists() and (work / f"{stem}.inputs.json").exists():
        print(f"  {mlir.name} present, skipping import")
        return mlir
    command = [
        sys.executable,
        str(ROOT / "tools/import_onnx.py"),
        str(onnx_path),
        "--out-dir",
        str(work),
    ]
    for dim in dims:
        command += ["--dim", dim]
    if oracle != "same":
        command.append("--skip-oracle")
    run(command)
    return mlir


def build_f32_oracle(
    model: str, work: Path, stem: str, oracle_onnx: Path | None
) -> Path:
    """Runs the reference in f32 on the same inputs and saves it as the oracle.

    An fp16 model is the wrong thing to hand ONNX Runtime's CPU provider. It
    loads one in 1.6 seconds and then runs it with no fp16 kernels at all --
    ViT-L/16 did not finish a single inference in eighty minutes, single
    threaded throughout, while the same graph in f32 takes about a second.
    So the oracle comes from the f32 export of the same model, fed the *same*
    seeded sample widened to f32, which is exact.

    This is also the better reference. Comparing an fp16 pipeline against an
    fp16 reference measures agreement between two roundings; comparing it
    against f32 measures the error the model actually has. Both arms are
    scored against it, so the offload is still judged against the CPU arm on
    identical terms.
    """
    import numpy as np
    import onnxruntime as ort

    reference = oracle_onnx
    if reference is None:
        reference = work / f"{model}.f32.onnx"
        if not reference.exists():
            run(
                [
                    sys.executable,
                    str(ROOT / "tools/export_onnx.py"),
                    model,
                    "--out-dir",
                    str(work),
                    "--precision",
                    "f32",
                ]
            )
    print(f"  oracle from {reference.name}")

    manifest = json.loads((work / f"{stem}.inputs.json").read_text())
    session = ort.InferenceSession(
        str(reference), providers=["CPUExecutionProvider"]
    )
    if len(session.get_inputs()) != len(manifest):
        raise SystemExit(
            "error: the oracle model's inputs do not match the imported "
            f"model's: {[i.name for i in session.get_inputs()]} vs "
            f"{[e['name'] for e in manifest]}"
        )
    feed = {}
    for entry, model_input in zip(manifest, session.get_inputs()):
        sample = np.load(work / entry["file"].replace(".bin", ".npy"))
        if "float" in model_input.type:
            sample = sample.astype(np.float32)
        feed[model_input.name] = sample
    outputs = session.run(None, feed)
    primary = np.asarray(outputs[0], dtype=np.float32)
    path = work / f"{stem}.oracle.npy"
    np.save(path, primary)
    flat = primary.reshape(-1)
    print(
        f"    {primary.shape} argmax={int(flat.argmax())} "
        f"min={flat.min():.4f} max={flat.max():.4f}"
    )
    return path


def stage_compile(
    mlir: Path,
    work: Path,
    stem: str,
    arms: list[str],
    compiler: list[str],
    compiler_lib: Path,
    triple: str,
) -> dict[str, dict]:
    env = dict(os.environ)
    env["IREE_COMPILER_LIB"] = str(compiler_lib)
    reports = {}
    for arm in arms:
        # Named by stem, not by arm alone: the fp16 and f32 imports of one
        # model share a directory, and `npu.vmfb` for both means the second
        # compile silently overwrites the first -- and the board, which keeps
        # its copies by name, then benchmarks whichever landed last.
        vmfb = work / f"{stem}.{arm}.vmfb"
        report = work / f"{stem}.{arm}.report.json"
        if not (vmfb.exists() and report.exists()):
            run(
                [
                    *compiler,
                    "compile",
                    "--input",
                    str(mlir),
                    "--output",
                    str(vmfb),
                    "--llvmcpu-target-triple",
                    triple,
                    "--report-json",
                    str(report),
                    *ARMS[arm],
                ],
                env=env,
            )
        else:
            print(f"  {vmfb.name} present, skipping compile")
        reports[arm] = json.loads(report.read_text())
    return reports


def stage_run(
    host: str,
    remote_dir: str,
    work: Path,
    stem: str,
    arms: list[str],
    runtime: Path,
    function: str,
    results: int,
) -> dict[str, dict]:
    check_binary_is_current(runtime)
    manifest = json.loads((work / f"{stem}.inputs.json").read_text())
    payload = [runtime] + [work / e["file"] for e in manifest]
    payload += [work / f"{stem}.{arm}.vmfb" for arm in arms]
    sync(host, remote_dir, payload)

    oracle = work / f"{stem}.oracle.npy"
    correctness = {}
    for arm in arms:
        wait_for_quiet_npu(host)
        outputs = [f"--output=@{stem}.{arm}.out.npy"] + ["--output="] * (results - 1)
        remote_command = " && ".join(
            [
                f"cd {shlex.quote(remote_dir)}",
                f"chmod +x {runtime.name}",
                command_text(
                    [
                        f"./{runtime.name}",
                        f"--module={stem}.{arm}.vmfb",
                        f"--function={function}",
                        "--device=rocket",
                        "--device=local-task",
                        *input_flags(work, stem, remote=True),
                        *outputs,
                    ]
                ),
            ]
        )
        code, text = capture_both(["ssh", host, remote_command])
        if code != 0:
            print(text.strip()[-2000:], flush=True)
            correctness[arm] = {"ok": False, "detail": "the module did not run"}
            continue
        local = work / f"{stem}.{arm}.out.npy"
        run(
            [
                "scp",
                "-q",
                f"{host}:{remote_dir}/{stem}.{arm}.out.npy",
                str(local),
            ]
        )
        correctness[arm] = (
            compare(local, oracle)
            if oracle.exists()
            else {"ok": False, "detail": "no oracle"}
        )

    # The arm-to-arm comparison separates an offload bug from an import bug:
    # a difference the CPU arm shares with the NPU arm is not the NPU's.
    if len(arms) > 1 and all(correctness[a].get("ok") for a in arms):
        first, *rest = arms
        for arm in rest:
            correctness[f"{arm}_vs_{first}"] = compare(
                work / f"{stem}.{arm}.out.npy", work / f"{stem}.{first}.out.npy"
            )
    return correctness


BENCH_TIME = re.compile(r"^(\S+)\s+([0-9.]+)\s+ms", re.MULTILINE)


def stage_bench(
    host: str,
    remote_dir: str,
    work: Path,
    stem: str,
    arms: list[str],
    benchmark: Path,
    function: str,
    cpu_ids: list[str],
    repetitions: int,
) -> dict[str, dict]:
    check_binary_is_current(benchmark)
    sync(host, remote_dir, [benchmark])
    timings: dict[str, dict] = {arm: {} for arm in arms}
    for ids in cpu_ids:
        for arm in arms:
            wait_for_quiet_npu(host)
            remote_command = " && ".join(
                [
                    f"cd {shlex.quote(remote_dir)}",
                    f"chmod +x {benchmark.name}",
                    command_text(
                        [
                            f"./{benchmark.name}",
                            f"--module={stem}.{arm}.vmfb",
                            f"--function={function}",
                            "--device=rocket",
                            "--device=local-task",
                            f"--task_topology_cpu_ids={ids}",
                            *input_flags(work, stem, remote=True),
                            f"--benchmark_repetitions={repetitions + 1}",
                            "--benchmark_min_time=1x",
                        ]
                    ),
                ]
            )
            code, text = capture_both(["ssh", host, remote_command])
            samples = [
                float(value)
                for label, value in BENCH_TIME.findall(text)
                if not label.endswith(("_mean", "_median", "_stddev", "_cv"))
            ]
            if code != 0 or len(samples) < 2:
                print(text.strip()[-2000:], flush=True)
                timings[arm][ids] = {"median_ms": None, "samples": samples}
                continue
            # The first repetition is a warm-up and is discarded, not
            # averaged in. It is not noise: an offloaded arm pays its weight
            # packing on the first invocation only (ISSUES.md, `pack.weights`
            # is cold-start), and it is large enough to swallow the thing
            # being measured -- 1021 ms against a 124 ms steady state on the
            # first model run through this script. `--benchmark_repetitions`
            # is raised by one so the requested number still survives.
            warm = samples[1:]
            timings[arm][ids] = {
                "median_ms": statistics.median(warm),
                "min_ms": min(warm),
                "max_ms": max(warm),
                "warmup_ms": samples[0],
                "samples": warm,
            }
            print(
                f"  {arm:8s} cpus={ids:16s} {statistics.median(warm):9.1f} ms "
                f"(min {min(warm):.1f}, max {max(warm):.1f}, "
                f"warm-up {samples[0]:.1f} discarded)",
                flush=True,
            )
    return timings


# --------------------------------------------------------------------------
# reporting


def arm_row(report: dict) -> dict[str, int]:
    summary = report["summary"]
    placement = report["placement"]
    return {
        "candidates": summary["candidates"],
        "accepted": summary["direct"] + summary["tiled"],
        "npu_sites": placement["rocket"]["dispatch_sites"],
        "cpu_sites": placement["cpu"]["dispatch_sites"],
        "contraction_cpu_sites": placement["cpu"]["contraction_dispatch_sites"],
        "hardware_jobs": summary["hardware_jobs"],
    }


def print_model_table(result: dict) -> None:
    print()
    print(f"== {result['model']} ({result['precision']}) ==")
    for key, value in result["board"].items():
        print(f"  {key}: {value}")
    print()
    print(
        f"  {'arm':10s} {'cand':>5s} {'acc':>5s} {'npu':>5s} {'cpu':>6s} "
        f"{'conv/mm on cpu':>15s}"
    )
    for arm, data in result["arms"].items():
        row = data["placement"]
        print(
            f"  {arm:10s} {row['candidates']:5d} {row['accepted']:5d} "
            f"{row['npu_sites']:5d} {row['cpu_sites']:6d} "
            f"{row['contraction_cpu_sites']:15d}"
        )
    print()
    for arm, data in result["arms"].items():
        correctness = data.get("correctness") or {}
        if correctness.get("ok"):
            print(
                f"  {arm:10s} max|err| {correctness['max_abs_err']:.4g} "
                f"(reference sd {correctness['reference_sd']:.4g}), "
                f"argmax {'==' if correctness['argmax_matches'] else '!='}, "
                f"top5 {'==' if correctness['top5_matches'] else '!='}"
            )
        elif correctness:
            print(f"  {arm:10s} correctness: {correctness.get('detail')}")
    print()
    baseline = result["arms"].get("cpu", {}).get("bench") or {}
    for arm, data in result["arms"].items():
        for ids, timing in (data.get("bench") or {}).items():
            median = timing["median_ms"]
            if median is None:
                print(f"  {arm:10s} cpus={ids:16s} ABORTED")
                continue
            reference = (baseline.get(ids) or {}).get("median_ms")
            ratio = (
                f"  {reference / median:6.2f}x vs cpu"
                if reference and arm != "cpu"
                else "              "
            )
            spread = (
                f"  [{timing['min_ms']:.0f}-{timing['max_ms']:.0f}]"
                if timing.get("min_ms") is not None
                else ""
            )
            print(f"  {arm:10s} cpus={ids:16s} {median:9.1f} ms{ratio}{spread}")


def summarize(work_root: Path, limit: int) -> None:
    """Aggregates every model's placement report into one reason histogram.

    The point of running a dozen models. One model's refusals are anecdotes
    about that model; the same reason code appearing across six of them, with
    a site count attached, is a ranked work queue for the compiler -- and each
    row already carries its class (`shape`, `semantics`, `hardware`,
    `validation`), which says whether closing it needs a matcher, a planner
    change, or new hardware measurement.
    """
    results = sorted(work_root.glob("*/*.result.json"))
    if not results:
        raise SystemExit(f"error: no *.result.json under {work_root}")

    print(
        f"{'model':22s} {'prec':5s} {'arm':8s} {'npu':>5s} {'cpu':>6s} "
        f"{'ms':>9s} {'vs cpu':>8s}"
    )
    for path in results:
        result = json.loads(path.read_text())
        baseline = result["arms"].get("cpu", {}).get("bench") or {}
        for arm, data in result["arms"].items():
            row = data["placement"]
            timings = data.get("bench") or {}
            ids = next(iter(timings), None)
            median = (timings.get(ids) or {}).get("median_ms") if ids else None
            reference = (baseline.get(ids) or {}).get("median_ms") if ids else None
            ratio = (
                f"{reference / median:7.2f}x"
                if reference and median and arm != "cpu"
                else "       -"
            )
            shown = f"{median:9.1f}" if median else "        -"
            print(
                f"{result['model']:22s} {result['precision']:5s} {arm:8s} "
                f"{row['npu_sites']:5d} {row['cpu_sites']:6d} {shown} {ratio}"
            )

    buckets: dict[tuple[str, str, str], dict] = {}
    for path in results:
        result = json.loads(path.read_text())
        model = f"{result['model']}.{result['precision']}"
        npu = result["arms"].get("npu")
        if not npu or "report" not in npu:
            continue
        report = json.loads((path.parent / npu["report"]).read_text())
        for candidate in report["candidates"]:
            if candidate["decision"] != "cpu":
                continue
            key = (candidate["status"], candidate["limit"], candidate["kind"])
            bucket = buckets.setdefault(
                key, {"count": 0, "models": set(), "examples": []}
            )
            bucket["count"] += 1
            bucket["models"].add(model)
            if len(bucket["examples"]) < 2:
                bucket["examples"].append(
                    f"{model}: {candidate['shape']} [{candidate['precision']}] "
                    f"{candidate['detail'][:90]}"
                )

    print()
    print("Why candidates stayed on the CPU, every model pooled:")
    print()
    ordered = sorted(buckets.items(), key=lambda kv: -kv[1]["count"])
    for (status, limit_class, kind), bucket in ordered[:limit]:
        models = ", ".join(sorted(bucket["models"]))
        print(f"  {bucket['count']:4d}  {status} [{limit_class}] {kind}")
        print(f"        in {models}")
        for example in bucket["examples"]:
            print(f"        e.g. {example}")
    print()
    print(
        "  class: `shape`/`semantics` close with compiler work, `validation` "
        "needs a board measurement, `hardware` never closes."
    )


# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    survey = sub.add_parser("run", help="survey one model end to end")
    survey.add_argument("--model", required=True)
    survey.add_argument(
        "--onnx", type=Path, help="a model not in tools/export_onnx.py's registry"
    )
    survey.add_argument("--precision", choices=("fp16", "f32"), default="fp16")
    survey.add_argument(
        "--oracle",
        choices=("f32", "same", "none"),
        default="f32",
        help="reference to score both arms against: the f32 export of the "
        "same model (default), the imported model itself, or none",
    )
    survey.add_argument(
        "--oracle-onnx",
        type=Path,
        help="an explicit f32 reference, for a model outside the registry",
    )
    survey.add_argument("--dim", action="append", default=[], metavar="NAME=VALUE")
    survey.add_argument("--arm", action="append", choices=sorted(ARMS), default=[])
    survey.add_argument("--board", default="planck")
    survey.add_argument("--work-root", type=Path, default=ROOT / "survey")
    survey.add_argument("--remote-root", default="~/survey")
    survey.add_argument(
        "--cpu-ids",
        action="append",
        default=[],
        help="one --task_topology_cpu_ids value per timing column",
    )
    survey.add_argument("--repetitions", type=int, default=5)
    survey.add_argument(
        "--compiler-lib",
        type=Path,
        default=ROOT / "iree-build/build/lib/libIREECompiler.so",
    )
    survey.add_argument(
        "--rocket-compiler",
        default="cargo run -p rocket-compiler --release --",
        help="how to invoke rocket-compiler",
    )
    survey.add_argument(
        "--board-runtime",
        type=Path,
        default=ROOT / "iree-build/host-aarch64/build/iree/tools/iree-run-module",
    )
    survey.add_argument(
        "--board-benchmark",
        type=Path,
        default=ROOT / "iree-build/host-aarch64/build/iree/tools/iree-benchmark-module",
    )
    survey.add_argument("--triple", default="aarch64-linux-gnu")
    survey.add_argument(
        "--skip", action="append", choices=STAGES, default=[], help="skip a stage"
    )

    digest = sub.add_parser("summarize", help="pool every result.json")
    digest.add_argument("--work-root", type=Path, default=ROOT / "survey")
    digest.add_argument("--limit", type=int, default=20)

    args = parser.parse_args()
    if args.command == "summarize":
        summarize(args.work_root, args.limit)
        return

    arms = args.arm or list(DEFAULT_ARMS)
    cpu_ids = args.cpu_ids or list(DEFAULT_CPU_IDS)
    work = args.work_root / args.model
    work.mkdir(parents=True, exist_ok=True)
    remote_dir = remote_path(f"{args.remote_root}/{args.model}")

    print(f"[export] {args.model}")
    onnx_path = stage_export(args.model, work, args.onnx, args.precision)
    stem = onnx_path.stem

    print(f"[import] {onnx_path.name}")
    mlir = stage_import(onnx_path, work, args.dim, args.oracle)
    function, results = function_signature(mlir)
    print(f"  entry function @{function}, {results} result(s)")

    if args.oracle == "f32" and args.precision != "f32":
        oracle_path = work / f"{stem}.oracle.npy"
        if oracle_path.exists():
            print(f"[oracle] {oracle_path.name} present, skipping")
        else:
            print("[oracle] f32 reference")
            build_f32_oracle(args.model, work, stem, args.oracle_onnx)

    print(f"[compile] {', '.join(arms)}")
    reports = stage_compile(
        mlir,
        work,
        stem,
        arms,
        shlex.split(args.rocket_compiler),
        args.compiler_lib,
        args.triple,
    )

    # A skipped stage must not erase what a previous run measured: the common
    # case for `--skip run` is re-timing a model whose correctness is already
    # established, and silently dropping the correctness block from
    # `result.json` would make `summarize` report it as never checked.
    result_path = work / f"{stem}.result.json"
    previous = {}
    if result_path.exists():
        previous = json.loads(result_path.read_text()).get("arms", {})

    result = {
        "model": args.model,
        "precision": args.precision,
        "onnx": str(onnx_path),
        "oracle": args.oracle,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "board": board_conditions(args.board),
        "cpu_ids": cpu_ids,
        "arms": {
            arm: {
                "placement": arm_row(reports[arm]),
                "flags": ARMS[arm],
                "report": f"{stem}.{arm}.report.json",
                **{
                    key: value
                    for key, value in previous.get(arm, {}).items()
                    if key in ("correctness", "bench")
                },
            }
            for arm in arms
        },
    }

    if "run" not in args.skip:
        print("[run] correctness against the ONNX Runtime oracle")
        correctness = stage_run(
            args.board,
            remote_dir,
            work,
            stem,
            arms,
            args.board_runtime,
            function,
            results,
        )
        for arm in arms:
            result["arms"][arm]["correctness"] = correctness.get(arm)
        result["cross_arm"] = {
            k: v for k, v in correctness.items() if k not in arms
        }
    elif result_path.exists():
        result["cross_arm"] = json.loads(result_path.read_text()).get(
            "cross_arm", {}
        )

    if "bench" not in args.skip:
        print("[bench] iree-benchmark-module")
        timings = stage_bench(
            args.board,
            remote_dir,
            work,
            stem,
            arms,
            args.board_benchmark,
            function,
            cpu_ids,
            args.repetitions,
        )
        for arm in arms:
            result["arms"][arm]["bench"] = timings.get(arm)

    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print_model_table(result)
    print(f"\n  wrote {result_path}")


if __name__ == "__main__":
    main()
