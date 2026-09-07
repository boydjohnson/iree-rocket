#!/usr/bin/env python3
"""Run matmul regression gates on an RK3588 board.

The third gate in the `e2e_*_regression.py` family, closing the hole the conv
and pooling ones left: matmul has been offloaded since the ViT work, and until
now nothing ran a compiled matmul `.vmfb` on hardware.

Two independent checks, as in its neighbours:

1. iree-rocket-hal's raw FC oracle tests -- the height-one 1x1-convolution
   lowering `fc::Shape` builds -- run one per process on the board. These live
   in two test binaries, `fc_hw` and `fc_phase3_hw`, so both are cross-built.
2. Matmuls compiled twice from MLIR: once for the host CPU and once for
   Rocket. The Rocket VMFB is executed through iree-run-module on the board
   and compared with the CPU VMFB.

**Most cases here are exact** (atol = rtol = 0), which is not obvious for a
contraction and is worth stating plainly. The trick is the fixture: entries
drawn from {-1, 0, 1} are exactly representable in f16, every product is
exactly representable, and the sums stay small -- measured |C|max is 79 at
K = 768 and 118 at K = 1792, against f16's integer-exact ceiling of 2048; a
ternary sum grows as the square root of K, so the K = 3584 cases stay inside it
too, and they came back bit-exact. So
the whole path is lossless and the hardware must return the CPU's answer
bit for bit. A single wrong accumulator lane, a mis-packed weight column or a
displaced output element then shows up as a nonzero difference rather than
hiding under a tolerance.

Ternary data is dense, not sparse: every element participates, so this is a
real contraction and not a coverage-only pattern (see the note in the HAL
about `Counting` patterns proving coverage and nothing else).

What ternary data cannot exercise is f16 *mantissa* behaviour, since it never
rounds. `matmul_vit_dense` is the same shape with uniform f32 values and the
--atol/--rtol tolerances, so the two together cover both failure modes: an
addressing fault the exact cases catch precisely, and a precision fault only
realistic magnitudes reach.

`linalg.matvec` and `linalg.vecmat` are here too. They have no matcher of
their own -- `rocket-expand-gemv-to-matmul` raises them into `linalg.matmul`
with a unit extent before the match loop -- so what these cases really gate is
that raising, including the dimension mapping that is easy to write backwards
(matvec pins N, vecmat pins M).

`--only FUNCTION` (repeatable) runs a subset. The NPU's state is
order-dependent enough that one case can leave the device sick for the next,
so a filter is how a single case gets a clean verdict.

It requires Python numpy, ssh/scp access to the board, the aarch64 Rust
target, and built host/aarch64 IREE tools in their normal repository
locations.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile

import numpy as np

from typing import NamedTuple, Sequence


ROOT = Path(__file__).resolve().parents[1]

# `KEY=VALUE` settings prefixed (via `env`) to every board run of the compiled
# module, from `--board-env`. How a driver knob such as `ROCKET_NPU_CORES=3`
# reaches the board without a shell profile the ssh session would not read.
BOARD_ENV: list[str] = []

# (test binary, test name). Two binaries, unlike the conv and pooling gates:
# the FC oracle is split across `fc_hw` (the packing and column-independence
# checks) and `fc_phase3_hw` (the ones that assert the job actually runs on
# the NPU rather than falling back).
RAW_TESTS = (
    ("fc_hw", "fc_uniform_fill_columns_agree"),
    ("fc_hw", "fc_output_tracks_input"),
    ("fc_hw", "fc_packed_weights_select_one_output_channel"),
    ("fc_hw", "fc_columns_are_independent"),
    ("fc_hw", "fc_fp16_distinct_weights_follow_channels"),
    ("fc_phase3_hw", "fp16_height_one_fc_runs_on_npu"),
    ("fc_phase3_hw", "odd_int8_n_height_one_fc_runs_on_npu"),
)
REMOTE_PREFIX = "/tmp/iree-rocket-matmul-regression."

MATMUL_MLIR = """\
// A ViT projection: 197 tokens by 768 features. This is the shape the matmul
// offload was built for -- ONNX MatMul over a [1, tokens, features]
// activation, which the transform spec collapses from linalg.batch_matmul to
// this -- and the one that first beat the CPU.
func.func @matmul_vit(%lhs: tensor<197x768xf32>, %rhs: tensor<768x768xf32>, %init: tensor<197x768xf32>) -> tensor<197x768xf32> {
  %0 = linalg.matmul
      ins(%lhs, %rhs : tensor<197x768xf32>, tensor<768x768xf32>)
      outs(%init : tensor<197x768xf32>) -> tensor<197x768xf32>
  return %0 : tensor<197x768xf32>
}

// The same shape with realistic magnitudes rather than ternary integers, and
// the only case here that needs a tolerance. Ternary data never rounds, so it
// cannot see a precision fault; this can, and nothing else here does.
func.func @matmul_vit_dense(%lhs: tensor<197x768xf32>, %rhs: tensor<768x768xf32>, %init: tensor<197x768xf32>) -> tensor<197x768xf32> {
  %0 = linalg.matmul
      ins(%lhs, %rhs : tensor<197x768xf32>, tensor<768x768xf32>)
      outs(%init : tensor<197x768xf32>) -> tensor<197x768xf32>
  return %0 : tensor<197x768xf32>
}

// MobileNetV2's classifier, and the M = 1 edge: a single row of output, which
// is the degenerate end of the conv-width mapping fc::Shape uses (M is the
// convolution width at height one).
func.func @matmul_classifier(%lhs: tensor<1x1792xf32>, %rhs: tensor<1792x1001xf32>, %init: tensor<1x1001xf32>) -> tensor<1x1001xf32> {
  %0 = linalg.matmul
      ins(%lhs, %rhs : tensor<1x1792xf32>, tensor<1792x1001xf32>)
      outs(%init : tensor<1x1001xf32>) -> tensor<1x1001xf32>
  return %0 : tensor<1x1001xf32>
}

// Both channel ceilings at once: K = 3584 is MAX_INPUT_CHANNELS and N = 3584
// is MAX_OUTPUT_CHANNELS. M stays small so a failure characterises the
// channel limits rather than the row-width one below. Both were 1792 until
// 2026-09-06; a transformer MLP at 3072 is what moved them.
func.func @matmul_k_n_ceilings(%lhs: tensor<8x3584xf32>, %rhs: tensor<3584x3584xf32>, %init: tensor<8x3584xf32>) -> tensor<8x3584xf32> {
  %0 = linalg.matmul
      ins(%lhs, %rhs : tensor<8x3584xf32>, tensor<3584x3584xf32>)
      outs(%init : tensor<8x3584xf32>) -> tensor<8x3584xf32>
  return %0 : tensor<8x3584xf32>
}

// The shape the raise was for: ViT-B/16's first MLP projection, 768 -> 3072
// at M = 197, which the old 1792 ceiling kept on the CPU. Ternary and so
// exact, at an M that column-tiles.
func.func @matmul_vit_mlp(%lhs: tensor<197x768xf32>, %rhs: tensor<768x3072xf32>, %init: tensor<197x3072xf32>) -> tensor<197x3072xf32> {
  %0 = linalg.matmul
      ins(%lhs, %rhs : tensor<197x768xf32>, tensor<768x3072xf32>)
      outs(%init : tensor<197x3072xf32>) -> tensor<197x3072xf32>
  return %0 : tensor<197x3072xf32>
}

// M at the matcher's accepted ceiling, 2047 -- the register's own limit,
// `CNA_DATA_SIZE0.datain_width` being 11 bits. LIMITS.md records this as the
// widest *matched* M while noting only 296 was measured in the HAL and 197
// end to end, so this case is the one that closes that gap: above the
// row-width limit the planner has to split column tiles, and nothing has
// checked it does so correctly through a compiled module.
func.func @matmul_m_2047(%lhs: tensor<2047x64xf32>, %rhs: tensor<64x64xf32>, %init: tensor<2047x64xf32>) -> tensor<2047x64xf32> {
  %0 = linalg.matmul
      ins(%lhs, %rhs : tensor<2047x64xf32>, tensor<64x64xf32>)
      outs(%init : tensor<2047x64xf32>) -> tensor<2047x64xf32>
  return %0 : tensor<2047x64xf32>
}

// linalg.matvec: A[m,k] * y[k] -> z[m]. No matcher of its own --
// rocket-expand-gemv-to-matmul raises it into a matmul with N = 1 -- so what
// this gates is the raising, and specifically that the vector operand became
// [k, 1] and the accumulator [m, 1] rather than the transposes of those.
func.func @matvec(%a: tensor<197x768xf32>, %y: tensor<768xf32>, %init: tensor<197xf32>) -> tensor<197xf32> {
  %0 = linalg.matvec
      ins(%a, %y : tensor<197x768xf32>, tensor<768xf32>)
      outs(%init : tensor<197xf32>) -> tensor<197xf32>
  return %0 : tensor<197xf32>
}

// linalg.vecmat: y[k] * A[k,n] -> z[n], raised with M = 1 instead. The mirror
// of the case above, and the reason both are here: the two pin opposite
// extents, so a mapping written backwards passes one and fails the other.
func.func @vecmat(%y: tensor<768xf32>, %a: tensor<768x1000xf32>, %init: tensor<1000xf32>) -> tensor<1000xf32> {
  %0 = linalg.vecmat
      ins(%y, %a : tensor<768xf32>, tensor<768x1000xf32>)
      outs(%init : tensor<1000xf32>) -> tensor<1000xf32>
  return %0 : tensor<1000xf32>
}

// Two Rocket dispatches in one function, so they share a command buffer --
// what a real transformer does at every layer, and what no single-matmul case
// covers. The two are independent, so a difference cannot be explained by one
// feeding the other. Both are exact, so it cannot be explained by tolerance
// either.
func.func @mixed_matmul_then_matmul(%a: tensor<197x768xf32>, %b: tensor<768x768xf32>, %ai: tensor<197x768xf32>, %c: tensor<8x1792xf32>, %d: tensor<1792x512xf32>, %ci: tensor<8x512xf32>) -> (tensor<197x768xf32>, tensor<8x512xf32>) {
  %0 = linalg.matmul
      ins(%a, %b : tensor<197x768xf32>, tensor<768x768xf32>)
      outs(%ai : tensor<197x768xf32>) -> tensor<197x768xf32>
  %1 = linalg.matmul
      ins(%c, %d : tensor<8x1792xf32>, tensor<1792x512xf32>)
      outs(%ci : tensor<8x512xf32>) -> tensor<8x512xf32>
  return %0, %1 : tensor<197x768xf32>, tensor<8x512xf32>
}
"""


def command_text(command: list[str]) -> str:
    return shlex.join(str(part) for part in command)


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print(f"+ {command_text(command)}", flush=True)
    subprocess.run(command, check=True, env=env)


def run_allowing_failure(command: list[str]) -> bool:
    """Runs `command`, returning whether it succeeded instead of raising."""
    print(f"+ {command_text(command)}", flush=True)
    return subprocess.run(command, check=False).returncode == 0


def capture(command: list[str]) -> str:
    print(f"+ {command_text(command)}", flush=True)
    return subprocess.check_output(command, text=True).strip()


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise SystemExit(f"{description} not found: {path}")


def build_raw_test(name: str, linker: str) -> Path:
    command = [
        "cargo",
        "test",
        "-p",
        "iree-rocket-hal",
        "--release",
        "--target",
        "aarch64-unknown-linux-gnu",
        "--test",
        name,
        "--no-run",
        "--message-format=json",
    ]
    env = os.environ.copy()
    env["CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER"] = linker
    print(f"+ {command_text(command)}", flush=True)
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        text=True,
    )
    executable: Path | None = None
    assert process.stdout is not None
    for line in process.stdout:
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        target = message.get("target", {})
        candidate = message.get("executable")
        if target.get("name") == name and candidate:
            executable = Path(candidate)
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    if executable is None:
        raise SystemExit(f"cargo did not report the {name} executable")
    return executable


def run_raw_gate(host: str, remote_dir: str, linker: str) -> None:
    # Build each distinct binary once, then run every test in it as its own
    # process: a case cannot then inherit device state from the one before it
    # in the same address space.
    for binary in sorted({name for name, _ in RAW_TESTS}):
        executable = build_raw_test(binary, linker)
        remote_executable = f"{remote_dir}/{binary}"
        run(["scp", str(executable), f"{host}:{remote_executable}"])
        test_commands = [
            command_text(
                [
                    remote_executable,
                    "--exact",
                    test,
                    "--ignored",
                    "--nocapture",
                    "--test-threads=1",
                ]
            )
            for name, test in RAW_TESTS
            if name == binary
        ]
        remote_command = " && ".join(
            [f"chmod +x {shlex.quote(remote_executable)}", *test_commands]
        )
        run(["ssh", host, remote_command])


def ternary(rng: np.random.Generator, *shape: int) -> np.ndarray:
    """Fixture values from {-1, 0, 1}, as f32.

    Exactly representable in f16, every pairwise product likewise, and the
    sums stay far inside f16's integer-exact range (|C|max measures 79 at
    K = 768 and 118 at K = 1792 against a ceiling of 2048). That is what lets
    every case using this be compared at atol = rtol = 0.

    Dense, not sparse: every element participates, so this is a real
    contraction rather than a pattern that only proves coverage.
    """
    return rng.integers(-1, 2, size=shape).astype(np.float32)


def write_compiled_fixture(work_dir: Path) -> None:
    (work_dir / "matmul.mlir").write_text(MATMUL_MLIR)
    rng = np.random.default_rng(20260905)

    def zeros(*shape: int) -> np.ndarray:
        return np.zeros(shape, dtype=np.float32)

    np.save(work_dir / "vit_lhs.npy", ternary(rng, 197, 768))
    np.save(work_dir / "vit_rhs.npy", ternary(rng, 768, 768))
    np.save(work_dir / "vit_init.npy", zeros(197, 768))

    # The one non-ternary fixture: realistic magnitudes, so f16 rounding is
    # actually exercised. Kept small enough that the K = 768 accumulation
    # stays well inside f16's range.
    np.save(
        work_dir / "vit_dense_lhs.npy",
        rng.uniform(-0.25, 0.25, size=(197, 768)).astype(np.float32),
    )
    np.save(
        work_dir / "vit_dense_rhs.npy",
        rng.uniform(-0.25, 0.25, size=(768, 768)).astype(np.float32),
    )
    np.save(work_dir / "vit_dense_init.npy", zeros(197, 768))

    np.save(work_dir / "cls_lhs.npy", ternary(rng, 1, 1792))
    np.save(work_dir / "cls_rhs.npy", ternary(rng, 1792, 1001))
    np.save(work_dir / "cls_init.npy", zeros(1, 1001))

    np.save(work_dir / "ceil_lhs.npy", ternary(rng, 8, 3584))
    np.save(work_dir / "ceil_rhs.npy", ternary(rng, 3584, 3584))
    np.save(work_dir / "ceil_init.npy", zeros(8, 3584))
    np.save(work_dir / "mlp_lhs.npy", ternary(rng, 197, 768))
    np.save(work_dir / "mlp_rhs.npy", ternary(rng, 768, 3072))
    np.save(work_dir / "mlp_init.npy", zeros(197, 3072))

    np.save(work_dir / "m2047_lhs.npy", ternary(rng, 2047, 64))
    np.save(work_dir / "m2047_rhs.npy", ternary(rng, 64, 64))
    np.save(work_dir / "m2047_init.npy", zeros(2047, 64))

    np.save(work_dir / "matvec_a.npy", ternary(rng, 197, 768))
    np.save(work_dir / "matvec_y.npy", ternary(rng, 768))
    np.save(work_dir / "matvec_init.npy", zeros(197))

    np.save(work_dir / "vecmat_y.npy", ternary(rng, 768))
    np.save(work_dir / "vecmat_a.npy", ternary(rng, 768, 1000))
    np.save(work_dir / "vecmat_init.npy", zeros(1000))

    np.save(work_dir / "mixed_c.npy", ternary(rng, 8, 1792))
    np.save(work_dir / "mixed_d.npy", ternary(rng, 1792, 512))
    np.save(work_dir / "mixed_ci.npy", zeros(8, 512))


# The device flags the transform spec hardcodes, shared by every Rocket
# compile below. Kept in one place so the three-stage build and the
# preprocessing dump cannot drift apart.
ROCKET_DEVICE_FLAGS = [
    "--iree-hal-target-device=rocket_device=rocket",
    "--iree-hal-target-device=cpu_device=local",
    "--iree-hal-local-target-device-backends=llvm-cpu",
    "--iree-hal-default-device=cpu_device",
    "--iree-hal-indirect-command-buffers=false",
    "--iree-llvmcpu-target-cpu=generic",
]

# Every compiled function must reach the matmul executable. Without this a
# case whose shape quietly stopped matching would still "pass": it would run
# entirely on the CPU on both sides of the differential and agree with itself.
# There is only one executable to name here, unlike the pooling gate -- which
# is the point of raising GEMVs rather than giving them their own.
EXPECTED_EXECUTABLE = "rocket_matmul_executable"
EXPECTED_FUNCTIONS = (
    "matmul_vit",
    "matmul_vit_dense",
    "matmul_classifier",
    "matmul_k_n_ceilings",
    "matmul_vit_mlp",
    "matmul_m_2047",
    "matvec",
    "vecmat",
    "mixed_matmul_then_matmul",
)


def compile_modules(
    work_dir: Path, compiler: Path, opt: Path, transform_spec: Path
) -> None:
    source = work_dir / "matmul.mlir"
    run(
        [
            str(compiler),
            str(source),
            "-o",
            str(work_dir / "cpu.vmfb"),
            "--iree-hal-target-backends=llvm-cpu",
            "--iree-llvmcpu-target-cpu=generic",
        ]
    )

    # Three stages, not one, because `rocket-pin-unclaimed-dispatches` has to
    # run between the flow and stream phases and no plugin hook exists that
    # late -- exactly what `rocket-compiler` does for a model (see README,
    # "Placement pinning"). @call_rocket_matmul ends in a CPU epilogue whose
    # only producer is the Rocket dispatch, so without the pin, Stream's
    # affinity analysis places that epilogue on @rocket_device and
    # serialization dies on an op that is not a matmul.
    flow = work_dir / "rocket_flow.mlir"
    pinned = work_dir / "rocket_flow_pinned.mlir"
    run(
        [
            str(compiler),
            str(source),
            "-o",
            str(flow),
            f"--iree-preprocessing-transform-spec-filename={transform_spec}",
            "--iree-llvmcpu-target-triple=aarch64-linux-gnu",
            *ROCKET_DEVICE_FLAGS,
            "--compile-to=flow",
        ]
    )
    run(
        [
            str(opt),
            str(flow),
            "-o",
            str(pinned),
            "--pass-pipeline=builtin.module(rocket-pin-unclaimed-dispatches)",
        ]
    )
    run(
        [
            str(compiler),
            str(pinned),
            "-o",
            str(work_dir / "rocket.vmfb"),
            f"--iree-preprocessing-transform-spec-filename={transform_spec}",
            "--iree-llvmcpu-target-triple=aarch64-linux-gnu",
            *ROCKET_DEVICE_FLAGS,
            "--compile-from=flow",
        ]
    )
    preprocessing = work_dir / "rocket_preprocessing.mlir"
    run(
        [
            str(compiler),
            str(source),
            "-o",
            str(preprocessing),
            f"--iree-preprocessing-transform-spec-filename={transform_spec}",
            *ROCKET_DEVICE_FLAGS,
            "--compile-to=preprocessing",
            "--mlir-print-op-generic=false",
        ]
    )
    preprocessing_text = preprocessing.read_text()
    for function in EXPECTED_FUNCTIONS:
        match = re.search(
            rf"util\.func public @{re.escape(function)}\b(?P<body>.*?)"
            r"(?=\n\s*util\.func (?:public|private) @|\Z)",
            preprocessing_text,
            re.DOTALL,
        )
        # The marker is the dispatch, not the call: @__transform_main inlines
        # the @call_rocket_* wrappers, so `util.call` no longer survives to
        # preprocessing. The dispatch is the better check anyway -- it is the
        # thing that actually reaches the NPU.
        if match is None or (
            f"flow.dispatch @{EXPECTED_EXECUTABLE}" not in match.group("body")
        ):
            raise SystemExit(
                f"{function} no longer reaches {EXPECTED_EXECUTABLE}; refusing to "
                "run a CPU-versus-CPU differential"
            )
    rocket_bytes = (work_dir / "rocket.vmfb").read_bytes()
    if b"rocket-flatbuffer-v1" not in rocket_bytes or b"RKT1" not in rocket_bytes:
        raise SystemExit("compiled module contains no serialized Rocket executable")


def run_cpu_reference(
    work_dir: Path,
    host_runtime: Path,
    function: str,
    input_names: Sequence[str],
    output_names: Sequence[str],
) -> None:
    run(
        [
            str(host_runtime),
            f"--module={work_dir / 'cpu.vmfb'}",
            f"--function={function}",
            "--device=local-task",
            *(f"--input=@{work_dir / name}" for name in input_names),
            *(f"--output=@{work_dir / name}" for name in output_names),
        ]
    )


def run_rocket_module(
    host: str,
    remote_dir: str,
    work_dir: Path,
    board_runtime: Path,
    function: str,
    input_names: Sequence[str],
    output_names: Sequence[str],
    tolerate_failure: bool = False,
) -> bool:
    """Runs one function on the board, returning whether it executed.

    `tolerate_failure` is for a `known_failure` case, which may fail by not
    running at all rather than by computing the wrong numbers -- a job that
    hangs is killed by the watchdog and `iree-run-module` refuses the result
    rather than handing back a half-written buffer. Raising there would abort
    the whole gate over a failure it is meant to tolerate.
    """
    staged = [board_runtime, work_dir / "rocket.vmfb"]
    staged += [work_dir / name for name in input_names]
    run(["scp", *(str(path) for path in staged), f"{host}:{remote_dir}/"])
    runtime_name = board_runtime.name
    remote_command = " && ".join(
        [
            f"cd {shlex.quote(remote_dir)}",
            f"chmod +x {shlex.quote(runtime_name)}",
            command_text(
                [
                    *(["env", *BOARD_ENV] if BOARD_ENV else []),
                    f"./{runtime_name}",
                    "--module=rocket.vmfb",
                    f"--function={function}",
                    "--device=rocket",
                    "--device=local-task",
                    *(f"--input=@{name}" for name in input_names),
                    *(f"--output=@{name}" for name in output_names),
                ]
            ),
        ]
    )
    if tolerate_failure:
        if not run_allowing_failure(["ssh", host, remote_command]):
            return False
    else:
        run(["ssh", host, remote_command])
    for name in output_names:
        run(["scp", f"{host}:{remote_dir}/{name}", str(work_dir / name)])
    return True


def compare_outputs(
    work_dir: Path, cpu_name: str, rocket_name: str, atol: float, rtol: float
) -> bool:
    """Prints the differential and returns whether it matched.

    Returns rather than exiting so a known failure can be reported without
    taking the gate down with it; the caller decides what a mismatch means.
    """
    cpu = np.load(work_dir / cpu_name).astype(np.float64)
    rocket = np.load(work_dir / rocket_name).astype(np.float64)
    if cpu.shape != rocket.shape:
        raise SystemExit(f"output shape mismatch: CPU {cpu.shape}, Rocket {rocket.shape}")
    absolute_error = np.abs(cpu - rocket)
    allowed_error = atol + rtol * np.abs(cpu)
    mismatches = int(np.count_nonzero(absolute_error > allowed_error))
    max_error = float(absolute_error.max(initial=0.0))
    print(
        f"compiled VMFB differential ({rocket_name}): "
        f"max|error|={max_error:.8g}, mismatches={mismatches}/{cpu.size}, "
        f"atol={atol}, rtol={rtol}"
    )
    return mismatches == 0


class Case(NamedTuple):
    """One compiled differential case.

    `known_failure`, when set, is the reason this case is expected to differ
    from the CPU reference. Such a case is run and reported but does not fail
    the gate -- and fails it if it ever *passes*, so the list cannot rot.
    """

    function: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    atol: float | tuple[float, ...]
    rtol: float | tuple[float, ...]
    known_failure: str | None = None

    def tolerances(self, index: int) -> tuple[float, float]:
        atol = self.atol[index] if isinstance(self.atol, tuple) else self.atol
        rtol = self.rtol[index] if isinstance(self.rtol, tuple) else self.rtol
        return atol, rtol


def wait_for_quiet_npu(host: str, budget_seconds: float = 20.0) -> None:
    """Blocks until every NPU core's runtime-PM state reads `suspended`.

    A hung job leaves the device sick for seconds, and the state crosses
    processes -- so a gate started right after one (a deliberate repro, an
    aborted run, another session) can have its *first* case fail for reasons
    that have nothing to do with it. Waiting here turns "the first case hangs"
    into either a clean start or an explicit complaint about the board.
    """
    probe = (
        "for i in $(seq 1 %d); do "
        "s=$(cat /sys/devices/platform/*.npu/power/runtime_status 2>/dev/null); "
        '[ -n "$s" ] || { echo missing; exit 0; }; '
        'case "$s" in *active*) sleep 1;; *) echo quiet; exit 0;; esac; '
        "done; echo busy"
    ) % int(budget_seconds)
    state = capture(["ssh", host, probe]).splitlines()[-1].strip()
    if state == "quiet":
        return
    if state == "missing":
        print(
            "  note: no NPU runtime_status on the board; skipping the quiet-device "
            "check",
            flush=True,
        )
        return
    print(
        f"  WARNING: the NPU was still active after {budget_seconds:.0f}s. A job "
        "hung recently and the device may still be sick, which shows up as the "
        "FIRST case failing for no reason of its own. Re-run from a quiet board "
        "before believing a failure below.",
        flush=True,
    )


def build_cases(atol: float, rtol: float) -> list[Case]:
    """The compiled differential cases.

    Everything but `matmul_vit_dense` is exact; see this module's docstring
    for why a contraction can be. The dense case is the same shape with
    realistic magnitudes, and is the only one that needs a tolerance.
    """
    return [
        Case(
            "matmul_vit",
            ("vit_lhs.npy", "vit_rhs.npy", "vit_init.npy"),
            ("matmul_vit_out.npy",),
            0.0,
            0.0,
        ),
        Case(
            "matmul_vit_dense",
            ("vit_dense_lhs.npy", "vit_dense_rhs.npy", "vit_dense_init.npy"),
            ("matmul_vit_dense_out.npy",),
            atol,
            rtol,
        ),
        Case(
            "matmul_classifier",
            ("cls_lhs.npy", "cls_rhs.npy", "cls_init.npy"),
            ("matmul_classifier_out.npy",),
            0.0,
            0.0,
        ),
        Case(
            "matmul_k_n_ceilings",
            ("ceil_lhs.npy", "ceil_rhs.npy", "ceil_init.npy"),
            ("matmul_k_n_ceilings_out.npy",),
            0.0,
            0.0,
        ),
        Case(
            "matmul_vit_mlp",
            ("mlp_lhs.npy", "mlp_rhs.npy", "mlp_init.npy"),
            ("matmul_vit_mlp_out.npy",),
            0.0,
            0.0,
        ),
        Case(
            "matmul_m_2047",
            ("m2047_lhs.npy", "m2047_rhs.npy", "m2047_init.npy"),
            ("matmul_m_2047_out.npy",),
            0.0,
            0.0,
        ),
        Case(
            "matvec",
            ("matvec_a.npy", "matvec_y.npy", "matvec_init.npy"),
            ("matvec_out.npy",),
            0.0,
            0.0,
        ),
        Case(
            "vecmat",
            ("vecmat_y.npy", "vecmat_a.npy", "vecmat_init.npy"),
            ("vecmat_out.npy",),
            0.0,
            0.0,
        ),
        # Two Rocket dispatches sharing a command buffer.
        Case(
            "mixed_matmul_then_matmul",
            (
                "vit_lhs.npy",
                "vit_rhs.npy",
                "vit_init.npy",
                "mixed_c.npy",
                "mixed_d.npy",
                "mixed_ci.npy",
            ),
            ("mixed_matmul_then_matmul_0.npy", "mixed_matmul_then_matmul_1.npy"),
            0.0,
            0.0,
        ),
    ]


def run_compiled_gate(
    host: str,
    remote_dir: str,
    work_dir: Path,
    compiler: Path,
    opt: Path,
    host_runtime: Path,
    board_runtime: Path,
    transform_spec: Path,
    atol: float,
    rtol: float,
    only: list[str] | None = None,
) -> None:
    cases = build_cases(atol, rtol)
    # The two lists are written separately and would otherwise drift: a case
    # with no entry in EXPECTED_FUNCTIONS runs with nothing checking that it
    # still reaches a matcher, which is the exact failure -- a silent
    # CPU-versus-CPU differential -- that check exists to prevent.
    described = set(EXPECTED_FUNCTIONS)
    present = {case.function for case in cases}
    if described != present:
        raise SystemExit(
            "EXPECTED_FUNCTIONS and build_cases disagree: "
            f"cases not checked for a dispatch {sorted(present - described)}, "
            f"checked functions with no case {sorted(described - present)}"
        )

    write_compiled_fixture(work_dir)
    compile_modules(work_dir, compiler, opt, transform_spec)
    wait_for_quiet_npu(host)
    if only:
        wanted = set(only)
        unknown = wanted - {case.function for case in cases}
        if unknown:
            raise SystemExit(f"--only matched no case: {sorted(unknown)}")
        cases = [case for case in cases if case.function in wanted]
    unexpected_passes = []
    for case in cases:
        cpu_names = tuple(
            f"{case.function}_cpu_{index}.npy" for index in range(len(case.outputs))
        )
        run_cpu_reference(work_dir, host_runtime, case.function, case.inputs, cpu_names)
        executed = run_rocket_module(
            host,
            remote_dir,
            work_dir,
            board_runtime,
            case.function,
            case.inputs,
            case.outputs,
            tolerate_failure=case.known_failure is not None,
        )
        if not executed:
            matched = False
            print(f"  {case.function}: did not execute (see the error above)")
        else:
            # `all()` over a generator would stop at the first mismatch; a
            # list keeps every output's numbers on screen, which is the whole
            # point of a two-dispatch case.
            results = [
                compare_outputs(work_dir, cpu_name, rocket_name, *case.tolerances(index))
                for index, (cpu_name, rocket_name) in enumerate(
                    zip(cpu_names, case.outputs)
                )
            ]
            matched = all(results)
        if case.known_failure:
            if matched:
                unexpected_passes.append(case.function)
                print(f"  {case.function}: UNEXPECTEDLY PASSED -- see below")
            else:
                print(f"  {case.function}: known failure, not gating. {case.known_failure}")
        elif not matched:
            raise SystemExit("compiled Rocket matmul differs from the CPU reference")

    # A known failure that starts passing is itself a gate failure: the entry
    # is now a lie, and the bug it documents deserves a real assertion rather
    # than a comment nobody re-checks.
    if unexpected_passes:
        raise SystemExit(
            "known-failure case(s) now pass and must be promoted to real gate "
            f"cases: {', '.join(unexpected_passes)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--board-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="environment setting for every board run of the compiled module (repeatable), "
        "e.g. ROCKET_NPU_CORES=3",
    )
    parser.add_argument(
        "--board",
        required=True,
        help="SSH host name or config alias identifying the RK3588 board",
    )
    parser.add_argument(
        "--compiler",
        type=Path,
        default=ROOT / "iree-build/build/tools/iree-compile",
    )
    parser.add_argument(
        "--opt",
        type=Path,
        default=ROOT / "iree-build/build/tools/iree-opt",
        help="iree-opt with the Rocket plugin registered; runs "
        "rocket-pin-unclaimed-dispatches between the flow and stream phases, "
        "which a single iree-compile invocation cannot do",
    )
    parser.add_argument(
        "--host-runtime",
        type=Path,
        default=ROOT / "iree-build/build/tools/iree-run-module",
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
    parser.add_argument("--cross-linker", default="aarch64-linux-gnu-gcc")
    parser.add_argument(
        "--atol",
        type=float,
        default=0.05,
        help="absolute tolerance for matmul_vit_dense, the only inexact case; "
        "every other case is compared exactly and ignores this",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=0.02,
        help="relative tolerance for matmul_vit_dense, the only inexact case; "
        "every other case is compared exactly and ignores this",
    )
    parser.add_argument(
        "--only",
        action="append",
        metavar="FUNCTION",
        help="run only these compiled cases, by function name; repeatable. "
        "The NPU's state is order-dependent enough that one case can leave "
        "the device sick for the next, so a filter is how a single case gets "
        "a clean verdict (see ISSUES.md and npu-wedges-after-failed-job).",
    )
    parser.add_argument("--skip-raw", action="store_true")
    parser.add_argument("--skip-compiled", action="store_true")
    parser.add_argument(
        "--keep-remote",
        action="store_true",
        help="leave staged artifacts in the printed remote temporary directory",
    )
    args = parser.parse_args()
    for setting in args.board_env:
        if "=" not in setting:
            parser.error(f"--board-env expects KEY=VALUE, got {setting!r}")
    BOARD_ENV.extend(args.board_env)

    if args.skip_raw and args.skip_compiled:
        raise SystemExit("both gates were skipped")
    if not args.skip_compiled:
        require_file(args.compiler, "iree-compile")
        require_file(args.opt, "iree-opt")
        require_file(args.host_runtime, "host iree-run-module")
        require_file(args.board_runtime, "aarch64 iree-run-module")
        require_file(args.transform_spec, "Rocket transform spec")

    remote_dir = capture(["ssh", args.board, f"mktemp -d {REMOTE_PREFIX}XXXXXX"])
    if not remote_dir.startswith(REMOTE_PREFIX):
        raise SystemExit(f"board returned an unexpected temporary path: {remote_dir!r}")
    print(f"board staging directory: {remote_dir}")

    try:
        if not args.skip_raw:
            print("\n== raw fc::Shape/NPU oracle tests ==")
            run_raw_gate(args.board, remote_dir, args.cross_linker)
        if not args.skip_compiled:
            print("\n== compiled VMFB CPU differential ==")
            with tempfile.TemporaryDirectory(
                prefix="iree-rocket-matmul-regression-"
            ) as temporary:
                run_compiled_gate(
                    args.board,
                    remote_dir,
                    Path(temporary),
                    args.compiler,
                    args.opt,
                    args.host_runtime,
                    args.board_runtime,
                    args.transform_spec,
                    args.atol,
                    args.rtol,
                    args.only,
                )
    finally:
        if args.keep_remote:
            print(f"keeping board artifacts at {remote_dir}")
        else:
            run(["ssh", args.board, f"rm -rf -- {shlex.quote(remote_dir)}"])

    completed = []
    if not args.skip_raw:
        completed.append("raw oracle")
    if not args.skip_compiled:
        completed.append("compiled VMFB differential")
    print(f"\nPASS: matmul {' and '.join(completed)} regression gate(s)")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(f"command failed with exit code {error.returncode}", file=sys.stderr)
        sys.exit(error.returncode or 1)
