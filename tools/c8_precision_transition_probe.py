#!/usr/bin/env python3
"""C8 probe: separate "the first fp16 job after int8 hangs" from "this fp16
shape hangs after int8".

ISSUES.md's C8 localises the int8-offload hang to the `int8_accumulator ->
fp16` transition, and stops at a stated limit: MobileNetV2 has exactly one
fp16 convolution in its int8 build (the stem), so no experiment there can tell
a *position* property ("the first fp16 job after an int8 run hangs") from a
*shape* property ("this particular fp16 convolution hangs after int8"). This
builds the module that separates them.

Two things make it different from the model runs C8 was found with:

  * Several distinct fp16 dense shapes are victims, not one -- a 1x1, a 3x3, a
    3x3 stride-2, and two Cin=3 cases that take the driver's ARGB feature path
    (the path MobileNetV2's stem uses, and a suspect in its own right since it
    is the only one that hands an IREE buffer to the NPU directly).
  * The int8 convolutions come *first inside the same function*, so the
    transition happens between two dispatches of one command buffer rather
    than only across an inference boundary. C8's model needed a benchmark loop
    to reach the transition at all (the stem runs first, so int8 -> fp16 only
    happens at inference 2); here a single `iree-run-module` reaches it.

Read the result like this:

  * every `int8_then_*` case hangs           -> position. Any fp16 job is
                                                poisoned by a preceding int8
                                                run; the shape is irrelevant.
  * only some hang                           -> shape. Compare which: `stem`
                                                and `argb_s1` against the rest
                                                separates the Cin<=4 ARGB
                                                feature path from stride and
                                                kernel.
  * `int8_then_k1_then_stem` hangs on `stem` -> not *first*-after; the poison
    while `int8_then_k1` is clean               outlives one intervening fp16
                                                job.
  * `int8_then_stem_then_k1` hangs on `stem` -> first-after, and the `k1`
    and nothing else                            never runs to say more.

What it found, 2026-09-04 (ISSUES.md's C8 carries the tables): shape, and the
shape variable is the fp16 job's **output size**, bracketed at
256 KiB < X <= 272 KiB for the three-convolution aggressor here. Not the input
(`bigin` vs `bigout`), not `Cout`/extent/kernel/stride (`wide253`/`wide506`),
not position (`int8_then_k3_then_k1` hangs on the `k1`, past a clean fp16 job).
The int8 side is a dose rather than a switch: one small int8 convolution
poisons nothing, and one `q1` hangs the `stem` but not `k1`.

Every case runs in its own process with an idle gap, because per-process
contamination is a known property of this device (see the `npu-wedges-after-
failed-job` note): results are order-dependent once anything has hung. A
known-good fp16 canary runs after any failure to say whether the board itself
went sick, which would make every later row meaningless.
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REMOTE_PREFIX = "/tmp/iree-rocket-c8-probe."


class Int8Conv(NamedTuple):
    """A `linalg.conv_2d_nchw_fchw_q`, the form an ONNX ConvInteger arrives in."""

    cin: int
    height: int
    width: int
    cout: int
    kernel: int
    zero_point: int

    @property
    def out_height(self) -> int:
        return self.height - self.kernel + 1

    @property
    def out_width(self) -> int:
        return self.width - self.kernel + 1


class F16Conv(NamedTuple):
    """A `linalg.conv_2d_nhwc_hwcf` in the f16/f16/f32 typing the matchers want."""

    cin: int
    height: int
    width: int
    cout: int
    kernel: int
    stride: int

    @property
    def out_height(self) -> int:
        return (self.height - self.kernel) // self.stride + 1

    @property
    def out_width(self) -> int:
        return (self.width - self.kernel) // self.stride + 1


# The int8 aggressors. All three are already hardware-exact gate cases in
# e2e_conv_regression.py, so a wrong number here is about the mix and not
# about the int8 path. Three rather than one because C8's minimal hanging mix
# was one fp16 conv plus *the* int8 dense convs (17 of them); `int8x1_then_*`
# below is the dose control that says whether one is enough.
INT8: dict[str, Int8Conv] = {
    "q1": Int8Conv(cin=64, height=32, width=32, cout=128, kernel=1, zero_point=7),
    "q2": Int8Conv(cin=32, height=34, width=34, cout=64, kernel=3, zero_point=11),
    "q3": Int8Conv(cin=16, height=32, width=32, cout=512, kernel=1, zero_point=13),
    # A deliberately tiny aggressor: 4 KiB of output against q3's 2 MiB. If
    # this poisons an fp16 job just as well, the size threshold belongs to the
    # victim alone and the int8 side only has to have run.
    "qs": Int8Conv(cin=16, height=8, width=8, cout=16, kernel=1, zero_point=5),
    # The aggressor-side counterpart of `bigin`/`bigout`: one int8 job with a
    # large input and a small output, one with the reverse. Which of the two
    # poisons says whether the int8 side keys on the same quantity the fp16
    # side does.
    "q_bigin": Int8Conv(cin=256, height=32, width=32, cout=16, kernel=1, zero_point=5),
    "q_bigout": Int8Conv(cin=16, height=32, width=32, cout=128, kernel=1, zero_point=5),
}

# The fp16 victims, chosen to vary one property at a time against `stem`:
#   stem     -- MobileNetV2's stem geometry, the only fp16 job C8 ever saw hang
#   argb_s1  -- same Cin=3 ARGB feature path, stride 1: isolates the path
#   k3s2     -- same 3x3 stride 2, Cin 32: isolates the stride/kernel
#   k1, k3   -- ordinary staged-feature convs, neither property
F16: dict[str, F16Conv] = {
    "stem": F16Conv(cin=3, height=225, width=225, cout=32, kernel=3, stride=2),
    "argb_s1": F16Conv(cin=3, height=34, width=34, cout=16, kernel=3, stride=1),
    "k1": F16Conv(cin=64, height=32, width=32, cout=128, kernel=1, stride=1),
    "k3": F16Conv(cin=32, height=34, width=34, cout=64, kernel=3, stride=1),
    "k3s2": F16Conv(cin=32, height=113, width=113, cout=64, kernel=3, stride=2),
    # The 2x2 that separates "the fp16 job's input is large" from "its output
    # is large", once the first round showed the hang tracks size rather than
    # kernel, stride or Cin. `bigin` has k1's exact input (128 KiB) with k3's
    # clean output (256 KiB); `bigout` inverts it -- an input well under the
    # smallest that ever hung, feeding a 1 MiB output.
    "bigin": F16Conv(cin=64, height=32, width=32, cout=64, kernel=1, stride=1),
    "bigout": F16Conv(cin=16, height=32, width=32, cout=256, kernel=1, stride=1),
    # Bisect between the largest clean output (256 KiB) and the smallest that
    # hangs (512 KiB), holding the input at 32 KiB so only the output moves.
    "out320": F16Conv(cin=16, height=32, width=32, cout=80, kernel=1, stride=1),
    "out384": F16Conv(cin=16, height=32, width=32, cout=96, kernel=1, stride=1),
    "out448": F16Conv(cin=16, height=32, width=32, cout=112, kernel=1, stride=1),
    # The same two sizes reached by a different geometry -- wide and shallow
    # instead of narrow and deep. If the boundary is the output's size these
    # land on the same side as their byte-equal counterparts; if it is really
    # about Cout or about extent, they do not.
    "wide253": F16Conv(cin=16, height=45, width=45, cout=32, kernel=1, stride=1),
    "wide506": F16Conv(cin=16, height=45, width=45, cout=64, kernel=1, stride=1),
    # Resolve the boundary finer than Cout can. Cout is padded to 16-channel
    # atoms at fp16, so at 32x32 the smallest step is 64 KiB; moving the
    # spatial extent instead steps by ~16 KiB. 33x33 is the first size above
    # 256 KiB, 34x34 the next, 36x36 comfortably past it.
    "px33": F16Conv(cin=16, height=33, width=33, cout=64, kernel=1, stride=1),
    "px34": F16Conv(cin=16, height=34, width=34, cout=64, kernel=1, stride=1),
    "px36": F16Conv(cin=16, height=36, width=36, cout=64, kernel=1, stride=1),
}

AGGRESSOR = ("q1", "q2", "q3")

# (case name, op sequence). Names are keys of INT8 or F16.
CASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    # Controls: each half alone, from a quiet device.
    ("int8_only", AGGRESSOR),
    ("f16_stem", ("stem",)),
    ("f16_argb_s1", ("argb_s1",)),
    ("f16_k1", ("k1",)),
    ("f16_k3", ("k3",)),
    ("f16_k3s2", ("k3s2",)),
    ("f16_all", ("stem", "argb_s1", "k1", "k3", "k3s2")),
    # The experiment: one fp16 shape after the same int8 run, five times.
    ("int8_then_stem", AGGRESSOR + ("stem",)),
    ("int8_then_argb_s1", AGGRESSOR + ("argb_s1",)),
    ("int8_then_k1", AGGRESSOR + ("k1",)),
    ("int8_then_k3", AGGRESSOR + ("k3",)),
    ("int8_then_k3s2", AGGRESSOR + ("k3s2",)),
    # Dose: is one int8 dispatch enough to poison the transition?
    ("int8x1_then_stem", ("q1", "stem")),
    # Position: does an intervening fp16 job absorb the poison, or outlive it?
    ("int8_then_k1_then_stem", AGGRESSOR + ("k1", "stem")),
    ("int8_then_stem_then_k1", AGGRESSOR + ("stem", "k1")),
    # The model's own shape: fp16 first (clean, per C8's inference 1), then the
    # transition, then fp16 again -- MobileNetV2's inference 1 -> 2 boundary
    # folded into one command buffer.
    ("stem_then_int8_then_stem", ("stem",) + AGGRESSOR + ("stem",)),
    # Input size against output size, with their controls.
    ("f16_bigin", ("bigin",)),
    ("f16_bigout", ("bigout",)),
    ("int8_then_bigin", AGGRESSOR + ("bigin",)),
    ("int8_then_bigout", AGGRESSOR + ("bigout",)),
    # Does an fp16 job that survives the transition consume the poison? `k3`
    # is clean after int8 and `k1` is not, so this asks whether `k1` is still
    # unsafe once a safe fp16 job has run in between.
    ("int8_then_k3_then_k1", AGGRESSOR + ("k3", "k1")),
    ("int8_then_out320", AGGRESSOR + ("out320",)),
    ("int8_then_out384", AGGRESSOR + ("out384",)),
    ("int8_then_out448", AGGRESSOR + ("out448",)),
    ("int8_then_wide253", AGGRESSOR + ("wide253",)),
    ("int8_then_wide506", AGGRESSOR + ("wide506",)),
    ("int8_then_px33", AGGRESSOR + ("px33",)),
    ("int8_then_px34", AGGRESSOR + ("px34",)),
    ("int8_then_px36", AGGRESSOR + ("px36",)),
    ("int8small_then_k1", ("qs", "k1")),
    ("int8small_then_k3", ("qs", "k3")),
    ("int8bigin_then_k1", ("q_bigin", "k1")),
    ("int8bigout_then_k1", ("q_bigout", "k1")),
    # `q1` alone poisons the stem but the trio was needed for `k1`, and the
    # two aggressor variants above changed shape and count at once. These
    # separate them: same single aggressor against both victims, and the
    # small-Cin aggressor against the sensitive victim.
    ("int8x1_then_k1", ("q1", "k1")),
    ("int8bigout_then_stem", ("q_bigout", "stem")),
)

# Run after any failing case to tell "this case hangs" from "the board is now
# sick and every later row is noise".
CANARY = "f16_k1"


def arg_names(case: str, ops: Sequence[str]) -> list[str]:
    """One (input, filter, init) triple per op, unique even when ops repeat."""
    names: list[str] = []
    for index, op in enumerate(ops):
        for role in ("in", "flt", "init"):
            names.append(f"{case}_{index}_{op}_{role}")
    return names


def emit_function(case: str, ops: Sequence[str]) -> str:
    """One `func.func` whose dispatches run in the order the ops are listed.

    The convolutions are mutually independent -- nothing flows from one to the
    next -- so a difference between two cases cannot be explained by one op
    feeding another, only by what the device carried between the jobs.
    """
    names = arg_names(case, ops)
    signature: list[str] = []
    results: list[str] = []
    result_types: list[str] = []
    body: list[str] = []

    for index, op in enumerate(ops):
        in_name, flt_name, init_name = names[3 * index : 3 * index + 3]
        if op in INT8:
            conv = INT8[op]
            in_type = f"tensor<1x{conv.cin}x{conv.height}x{conv.width}xi8>"
            flt_type = f"tensor<{conv.cout}x{conv.cin}x{conv.kernel}x{conv.kernel}xi8>"
            out_type = f"tensor<1x{conv.cout}x{conv.out_height}x{conv.out_width}xi32>"
            body.append(
                f"  %izp{index} = arith.constant {conv.zero_point} : i32\n"
                f"  %kzp{index} = arith.constant 0 : i32\n"
                f"  %r{index} = linalg.conv_2d_nchw_fchw_q\n"
                f"      {{dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}}\n"
                f"      ins(%{in_name}, %{flt_name}, %izp{index}, %kzp{index} :"
                f" {in_type}, {flt_type}, i32, i32)\n"
                f"      outs(%{init_name} : {out_type}) -> {out_type}"
            )
        else:
            conv = F16[op]
            in_type = f"tensor<1x{conv.height}x{conv.width}x{conv.cin}xf16>"
            flt_type = (
                f"tensor<{conv.kernel}x{conv.kernel}x{conv.cin}x{conv.cout}xf16>"
            )
            out_type = f"tensor<1x{conv.out_height}x{conv.out_width}x{conv.cout}xf32>"
            body.append(
                f"  %r{index} = linalg.conv_2d_nhwc_hwcf\n"
                f"      {{dilations = dense<1> : tensor<2xi64>,"
                f" strides = dense<{conv.stride}> : tensor<2xi64>}}\n"
                f"      ins(%{in_name}, %{flt_name} : {in_type}, {flt_type})\n"
                f"      outs(%{init_name} : {out_type}) -> {out_type}"
            )
        signature += [
            f"%{in_name}: {in_type}",
            f"%{flt_name}: {flt_type}",
            f"%{init_name}: {out_type}",
        ]
        results.append(f"%r{index}")
        result_types.append(out_type)

    joined_results = ", ".join(results)
    joined_types = ", ".join(result_types)
    return (
        f"func.func @{case}({', '.join(signature)}) -> ({joined_types}) {{\n"
        + "\n".join(body)
        + f"\n  return {joined_results} : {joined_types}\n}}\n"
    )


def build_mlir(cases: Sequence[tuple[str, tuple[str, ...]]]) -> str:
    return "\n".join(emit_function(case, ops) for case, ops in cases)


def write_fixtures(work_dir: Path, cases: Sequence[tuple[str, tuple[str, ...]]]) -> None:
    """One .npy per argument. Values vary in every dimension on purpose.

    A hang needs no particular data, but these cases are also compared against
    a CPU build, and uniform data cannot see a displaced value -- only a
    missing one (see ISSUES.md's method note on `OraclePattern::Counting`).
    """
    rng = np.random.default_rng(20260904)
    for case, ops in cases:
        names = arg_names(case, ops)
        for index, op in enumerate(ops):
            in_name, flt_name, init_name = names[3 * index : 3 * index + 3]
            if op in INT8:
                conv = INT8[op]
                np.save(
                    work_dir / f"{in_name}.npy",
                    rng.integers(-128, 128, (1, conv.cin, conv.height, conv.width), dtype=np.int8),
                )
                np.save(
                    work_dir / f"{flt_name}.npy",
                    rng.integers(
                        -128, 128, (conv.cout, conv.cin, conv.kernel, conv.kernel), dtype=np.int8
                    ),
                )
                np.save(
                    work_dir / f"{init_name}.npy",
                    np.zeros((1, conv.cout, conv.out_height, conv.out_width), dtype=np.int32),
                )
            else:
                conv = F16[op]
                np.save(
                    work_dir / f"{in_name}.npy",
                    rng.uniform(-0.25, 0.25, (1, conv.height, conv.width, conv.cin)).astype(
                        np.float16
                    ),
                )
                np.save(
                    work_dir / f"{flt_name}.npy",
                    rng.uniform(-0.5, 0.5, (conv.kernel, conv.kernel, conv.cin, conv.cout)).astype(
                        np.float16
                    ),
                )
                np.save(
                    work_dir / f"{init_name}.npy",
                    np.zeros((1, conv.out_height, conv.out_width, conv.cout), dtype=np.float32),
                )


def run(command: Sequence[str], summary: str | None = None, quiet: bool = False) -> None:
    print(f"+ {summary or shlex.join(str(part) for part in command)}", flush=True)
    subprocess.run(
        [str(part) for part in command],
        check=True,
        stdout=subprocess.DEVNULL if quiet else None,
    )


def compile_modules(
    work_dir: Path,
    compiler: Path,
    transform_spec: Path,
    cases: Sequence[tuple[str, tuple[str, ...]]],
) -> None:
    """Compiles only the selected cases.

    Only the selected ones because IREE deduplicates dispatches across the
    whole module, and at seventeen functions it collided two differently sized
    elementwise-transpose dispatches and refused to compile
    (`invalid Read access range [0 to 524288 for 524288] of resource with size
    262144`). Nothing about that is what this probe measures, and a module
    holding just the cases being run cannot hit it.
    """
    source = work_dir / "probe.mlir"
    source.write_text(build_mlir(cases))
    run(
        [
            compiler,
            source,
            "-o",
            work_dir / "cpu.vmfb",
            "--iree-hal-target-backends=llvm-cpu",
            "--iree-llvmcpu-target-cpu=generic",
        ]
    )
    rocket_flags = [
        f"--iree-preprocessing-transform-spec-filename={transform_spec}",
        "--iree-hal-target-device=rocket_device=rocket",
        "--iree-hal-target-device=cpu_device=local",
        "--iree-hal-local-target-device-backends=llvm-cpu",
        "--iree-hal-default-device=cpu_device",
        "--iree-hal-indirect-command-buffers=false",
        "--iree-llvmcpu-target-cpu=generic",
    ]
    run(
        [
            compiler,
            source,
            "-o",
            work_dir / "rocket.vmfb",
            "--iree-llvmcpu-target-triple=aarch64-linux-gnu",
            *rocket_flags,
        ]
    )
    preprocessing = work_dir / "preprocessing.mlir"
    run(
        [
            compiler,
            source,
            "-o",
            preprocessing,
            *rocket_flags,
            "--compile-to=preprocessing",
            "--mlir-print-op-generic=false",
        ],
        summary="iree-compile --compile-to=preprocessing (matcher check)",
        quiet=True,
    )
    verify_every_conv_offloaded(preprocessing.read_text(), cases)


def verify_every_conv_offloaded(
    preprocessing_text: str, cases: Sequence[tuple[str, tuple[str, ...]]]
) -> None:
    """Refuses to run a probe whose convolutions fell through to the CPU.

    Without this a case that stopped matching would still "pass": it would run
    on the CPU on both sides of the differential, agree with itself, and never
    hang -- reporting the absence of the very transition being measured.
    """
    for case, ops in cases:
        match = re.search(
            rf"util\.func public @{re.escape(case)}\b(?P<body>.*?)"
            r"(?=\n\s*util\.func (?:public|private) @|\Z)",
            preprocessing_text,
            re.DOTALL,
        )
        if match is None:
            raise SystemExit(f"{case}: not found in the preprocessing dump")
        body = match.group("body")
        int8_calls = len(re.findall(r"util\.call @call_rocket_dynamic_conv2d_int8\b", body))
        f16_calls = len(
            re.findall(r"util\.call @call_rocket_dynamic_conv2d(?!_int8)\w*", body)
        )
        want_int8 = sum(1 for op in ops if op in INT8)
        want_f16 = len(ops) - want_int8
        if int8_calls != want_int8 or f16_calls != want_f16:
            raise SystemExit(
                f"{case}: expected {want_int8} int8 and {want_f16} fp16 Rocket calls, "
                f"found {int8_calls} and {f16_calls} -- a convolution stopped reaching "
                "its matcher, so this case would measure nothing"
            )


def output_names(case: str, ops: Sequence[str]) -> list[str]:
    return [f"{case}_out{index}.npy" for index in range(len(ops))]


class Outcome(NamedTuple):
    case: str
    ran: bool
    hung: bool
    dispatches: list[str]
    error: str
    mismatch: str


def run_cpu_reference(
    work_dir: Path, host_runtime: Path, case: str, names: Sequence[str], outputs: Sequence[str]
) -> None:
    run(
        [
            host_runtime,
            f"--module={work_dir / 'cpu.vmfb'}",
            f"--function={case}",
            "--device=local-task",
            *(f"--input=@{work_dir / name}.npy" for name in names),
            *(f"--output=@{work_dir / name}" for name in outputs),
        ]
    )


def compare_case(work_dir: Path, case: str, ops: Sequence[str], atol: float, rtol: float) -> str:
    """Returns a description of the worst disagreement, or "" when every op matched.

    The int8 halves are compared exactly: the accumulator path is integer end
    to end, so any difference at all is a defect rather than rounding. The fp16
    halves get a tolerance because Rocket's ABI is f16-in/f32-accumulate and
    the CPU build is computing the same convolution in f32.
    """
    worst = ""
    for index, op in enumerate(ops):
        name = f"{case}_out{index}.npy"
        cpu = np.load(work_dir / f"cpu_{name}")
        rocket = np.load(work_dir / name)
        if cpu.shape != rocket.shape:
            return f"op {index} ({op}): shape {cpu.shape} vs {rocket.shape}"
        if op in INT8:
            wrong = int(np.count_nonzero(cpu != rocket))
            if wrong:
                worst = f"op {index} ({op}): {wrong} of {cpu.size} int32 lanes differ"
        else:
            error = np.abs(cpu.astype(np.float64) - rocket.astype(np.float64))
            allowed = atol + rtol * np.abs(cpu.astype(np.float64))
            wrong = int(np.count_nonzero(error > allowed))
            if wrong:
                worst = (
                    f"op {index} ({op}): max|err| {error.max():.4g}, "
                    f"{wrong} of {error.size} outside tolerance"
                )
    return worst


def run_case_on_board(
    host: str,
    remote_dir: str,
    runtime_name: str,
    case: str,
    names: Sequence[str],
    outputs: Sequence[str],
    idle_seconds: float,
    board_env: Sequence[str] = (),
) -> Outcome:
    """Runs one case in its own process, after an idle gap.

    Its own process and its own gap because the device is order-dependent
    across jobs and across processes both; sharing either would make a clean
    row unattributable.
    """
    command = " && ".join(
        [
            f"cd {shlex.quote(remote_dir)}",
            f"sleep {idle_seconds}",
            shlex.join(
                [
                    "env",
                    "ROCKET_DISPATCH_TIMES=1",
                    *board_env,
                    f"./{runtime_name}",
                    "--module=rocket.vmfb",
                    f"--function={case}",
                    "--device=rocket",
                    "--device=local-task",
                    *(f"--input=@{name}.npy" for name in names),
                    *(f"--output=@{name}" for name in outputs),
                ]
            ),
        ]
    )
    completed = subprocess.run(
        ["ssh", host, command], capture_output=True, text=True, check=False
    )
    stderr = completed.stderr
    dispatches = [
        line.strip()
        for line in stderr.splitlines()
        if line.startswith("rocket: dispatch ")
    ]
    hung = "hung-job floor" in stderr
    ran = completed.returncode == 0
    error = "" if ran else stderr.strip().splitlines()[-1] if stderr.strip() else "no output"
    return Outcome(case, ran, hung, dispatches, error, "")


def canary_recovers(
    host: str, remote_dir: str, runtime_name: str, idle_seconds: float
) -> bool:
    """Runs a known-good fp16 case until it passes, or gives up.

    A hang leaves the device unusable for a window rather than permanently:
    the canary immediately after one reads SICK and the same canary a few
    seconds later runs in 0.41 ms. Retrying with a growing gap is what tells
    those apart, and the difference matters -- "this case hangs" is a result,
    while "the board is still sick from the last case" makes every later row
    unattributable.
    """
    for attempt, extra in enumerate((0.0, 5.0, 15.0), start=1):
        canary = run_case_on_board(
            host,
            remote_dir,
            runtime_name,
            CANARY,
            arg_names(CANARY, dict(CASES)[CANARY]),
            [],
            idle_seconds + extra,
        )
        if canary.ran:
            print(f"   canary {CANARY}: healthy (attempt {attempt})")
            return True
        print(f"   canary {CANARY}: SICK (attempt {attempt})")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", required=True, help="SSH host of the RK3588")
    parser.add_argument("--compiler", type=Path, default=ROOT / "iree-build/build/tools/iree-compile")
    parser.add_argument(
        "--host-runtime", type=Path, default=ROOT / "iree-build/build/tools/iree-run-module"
    )
    parser.add_argument(
        "--board-runtime",
        type=Path,
        default=ROOT / "iree-build/host-aarch64/build/iree/tools/iree-run-module",
    )
    parser.add_argument(
        "--transform-spec",
        type=Path,
        default=ROOT / "rocket-compiler-plugin/target/Rocket/rocket_conv2d_transform_spec.mlir",
    )
    parser.add_argument(
        "--idle-seconds",
        type=float,
        default=2.0,
        help="quiet gap before each case; the regime where the environmental flake "
        "essentially never fires",
    )
    parser.add_argument("--only", action="append", help="run just these cases")
    parser.add_argument("--repeat", type=int, default=1, help="trials per case")
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument(
        "--skip-differential",
        action="store_true",
        help="only ask whether each case completes, not whether it was right",
    )
    parser.add_argument("--keep-remote", action="store_true")
    parser.add_argument(
        "--board-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra environment for the board's iree-run-module, e.g. "
        "ROCKET_PM_DWELL=suspend (repeatable)",
    )
    args = parser.parse_args()

    selected = [
        (case, ops)
        for case, ops in CASES
        if args.only is None or case in args.only
    ]
    if not selected:
        raise SystemExit("no cases selected")

    remote_dir = subprocess.run(
        ["ssh", args.board, f"mktemp -d {REMOTE_PREFIX}XXXXXX"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if not remote_dir.startswith(REMOTE_PREFIX):
        raise SystemExit(f"board returned an unexpected path: {remote_dir!r}")
    print(f"board staging directory: {remote_dir}")

    results: list[Outcome] = []
    try:
        with tempfile.TemporaryDirectory(prefix="iree-rocket-c8-probe-") as temporary:
            work_dir = Path(temporary)
            # The canary has to be in the module too, since it runs from the
            # same .vmfb whenever a case fails.
            canary_entry = (CANARY, dict(CASES)[CANARY])
            compiled = list(selected)
            if canary_entry not in compiled:
                compiled.append(canary_entry)
            write_fixtures(work_dir, compiled)
            compile_modules(work_dir, args.compiler, args.transform_spec, compiled)

            staged = [args.board_runtime, work_dir / "rocket.vmfb"]
            staged += sorted(work_dir.glob("*_in.npy"))
            staged += sorted(work_dir.glob("*_flt.npy"))
            staged += sorted(work_dir.glob("*_init.npy"))
            run(
                ["scp", "-q", *(str(path) for path in staged), f"{args.board}:{remote_dir}/"],
                summary=f"scp {len(staged)} files to {args.board}:{remote_dir}/",
            )
            runtime_name = args.board_runtime.name
            run(["ssh", args.board, f"chmod +x {shlex.quote(remote_dir)}/{runtime_name}"])

            for case, ops in selected:
                names = arg_names(case, ops)
                outputs = [] if args.skip_differential else output_names(case, ops)
                if outputs:
                    run_cpu_reference(
                        work_dir,
                        args.host_runtime,
                        case,
                        names,
                        [f"cpu_{name}" for name in outputs],
                    )
                for trial in range(args.repeat):
                    outcome = run_case_on_board(
                        args.board,
                        remote_dir,
                        runtime_name,
                        case,
                        names,
                        outputs,
                        args.idle_seconds,
                        args.board_env,
                    )
                    if outcome.ran and outputs:
                        run(
                            [
                                "scp",
                                "-q",
                                *(
                                    f"{args.board}:{remote_dir}/{name}"
                                    for name in outputs
                                ),
                                str(work_dir),
                            ],
                            summary=f"scp {len(outputs)} outputs back from {args.board}",
                        )
                        outcome = outcome._replace(
                            mismatch=compare_case(work_dir, case, ops, args.atol, args.rtol)
                        )
                    results.append(outcome)
                    label = "hung" if outcome.hung else ("ok" if outcome.ran else "failed")
                    if outcome.mismatch:
                        label = "WRONG"
                    print(f"\n== {case} trial {trial + 1}: {label}")
                    for line in outcome.dispatches:
                        print(f"   {line}")
                    if outcome.mismatch:
                        print(f"   {outcome.mismatch}")
                    if not outcome.ran:
                        print(f"   {outcome.error}")
                        if not canary_recovers(
                            args.board, remote_dir, runtime_name, args.idle_seconds
                        ):
                            print(
                                "   the device stayed wedged; every later row would be "
                                "noise. Reboot the board and resume with --only."
                            )
                            return
    finally:
        if args.keep_remote:
            print(f"keeping board artifacts at {remote_dir}")
        else:
            subprocess.run(
                ["ssh", args.board, f"rm -rf -- {shlex.quote(remote_dir)}"], check=False
            )
        print("\n== summary ==")
        for outcome in results:
            label = "HUNG" if outcome.hung else ("ok" if outcome.ran else "failed")
            if outcome.mismatch:
                label = "WRONG"
            note = f"  {outcome.mismatch}" if outcome.mismatch else ""
            print(
                f"{outcome.case:28} {label:6} "
                f"{len(outcome.dispatches)} NPU dispatches{note}"
            )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(f"command failed with exit code {error.returncode}", file=sys.stderr)
        sys.exit(error.returncode or 1)
