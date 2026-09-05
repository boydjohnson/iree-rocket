#!/usr/bin/env python3
"""Run pooling regression gates on an RK3588 board.

The pooling counterpart of `e2e_conv_regression.py`, with the same two-gate
shape:

1. iree-rocket-hal's raw PoolingPlan/PPU oracle tests -- max, padded max, min,
   average and tiled -- run one per process on the board.
2. Pooling compiled twice from MLIR: once for the host CPU and once for
   Rocket. The Rocket VMFB is executed through iree-run-module on the board
   and compared with the CPU VMFB.

The command exits nonzero if building, board execution, or comparison fails.

**Max pooling is gated exactly** (atol = rtol = 0), which the convolution gate
can only do for its int8 cases. The reason is that a max pool returns one of
its inputs unchanged: the shim demotes f32 to f16 on the way in and widens
back on the way out, so if every fixture value is exactly representable in
f16 -- which `f16_exact` guarantees by generating in f16 and widening -- the
round trip is lossless and the hardware must return the identical element the
CPU picked. Any difference at all is then a real fault, and the failure mode
worth catching is a *displaced* value rather than a slightly wrong one.

That is not a hope about the hardware: every max case here returned max|error|
exactly 0 on `planck` on 2026-09-05, the tiled 64x64 one included.

**Average pooling cannot be exact**, and not because of the demotion. The PPU
has no sum mode: its average is a multiply by fp16(65536/k), and the shim
multiplies that back up by k to recover the sum `linalg.pooling_nchw_sum`
asks for. That reciprocal round trip carries genuine f16 error, and it scales
with k -- which is why the 7x7 global pool is the loosest case here and takes
the --atol/--rtol defaults.

Those defaults have room. Measured on `planck`, 2026-09-05: the 49-tap global
pool comes back at max|error| 0.0057 and the 2x2 at 0.00051, against an
allowance of about 0.095. So the tolerance sits roughly 17x above the error
the hardware actually produces rather than being sized to just admit it.

Do not tighten it from a simulation. Modelling this path in numpy -- demote,
sum in f16, scale by fp16(65536/k), multiply back by k in f32 -- predicts
0.0027 for the global pool, about half what the board returns, because the
hardware's f16 accumulation order is not numpy's. The 2x2 agrees closely
(0.00049 predicted against 0.00051 measured); it is the deep 49-tap
accumulation where the model drifts, which is exactly where a tightened
tolerance would start failing for no reason.

**Min pooling is compared exactly too**, by the same argument as max -- a min
pool also returns one of its inputs unchanged. It is NHWC-only because linalg
defines `pooling_nchw_max` but no `pooling_nchw_min`, so the layout has no op
for a matcher to claim.

`min_pool_wide` earns its place: min is the method with *no* measured
pad-fill identity at any precision, so `required_pad_fill` refuses a padded
min outright. The executables bake zero padding and the driver derives
`padded` from those fields alone, which means tiling cannot reintroduce it --
and that case is what says so on hardware rather than on inspection.

Unlike the convolution harness, the window operand is a `tensor.empty()`
inside each function rather than a fixture: a pool's second operand carries no
values, only [kh, kw]. So every function here takes (input, init) and the
fixtures are two arrays per case, not three.

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

# `pooling_width_probe` is deliberately excluded: it is a probe that walks
# input widths until the hardware hangs, so it is expected to fail and would
# take the gate down with it. Its findings live in LIMITS.md's direct
# tile-width table instead.
RAW_TESTS = (
    "max_pooling_matches_oracle",
    "padded_max_pooling_matches_oracle",
    "min_pooling_matches_oracle",
    "average_pooling_matches_oracle",
    "tiled_pooling_matches_oracle",
)
REMOTE_PREFIX = "/tmp/iree-rocket-pooling-regression."

POOLING_MLIR = """\
// MobileNetV2's global average pool: 7x7 over 1792 channels down to one
// pixel. This is the shape PoolingDef was added to carry, and the only
// pooling shape a shipped model has ever driven through this backend.
//
// linalg has no average pool. An ONNX AveragePool arrives as a sum pool plus
// a separate divide, so the matched replacement runs the hardware's *average*
// and multiplies it back up by kh*kw -- which is where this case's error
// comes from, and why it is the loosest one here.
func.func @avg_pool_global(%input: tensor<1x1792x7x7xf32>, %init: tensor<1x1792x1x1xf32>) -> tensor<1x1792x1x1xf32> {
  %window = tensor.empty() : tensor<7x7xf32>
  %0 = linalg.pooling_nchw_sum
      {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%input, %window : tensor<1x1792x7x7xf32>, tensor<7x7xf32>)
      outs(%init : tensor<1x1792x1x1xf32>) -> tensor<1x1792x1x1xf32>
  return %0 : tensor<1x1792x1x1xf32>
}

// A small average pool with a 2x2 window -- the matcher's kernel floor, which
// exists because an fp16 average's reciprocal is fp16(65536/k) and k=1 needs
// 65536, past fp16's 65504 ceiling. Two taps instead of 49 also means far
// less accumulated f16 error than the case above, so a failure here is much
// harder to explain away as tolerance.
func.func @avg_pool_2x2(%input: tensor<1x64x16x16xf32>, %init: tensor<1x64x15x15xf32>) -> tensor<1x64x15x15xf32> {
  %window = tensor.empty() : tensor<2x2xf32>
  %0 = linalg.pooling_nchw_sum
      {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%input, %window : tensor<1x64x16x16xf32>, tensor<2x2xf32>)
      outs(%init : tensor<1x64x15x15xf32>) -> tensor<1x64x15x15xf32>
  return %0 : tensor<1x64x15x15xf32>
}

// Max pool, NHWC, stride 2 -- the shape a real network actually has, and the
// hardware's own layout, so this shim transposes nothing.
func.func @max_pool_nhwc_s2(%input: tensor<1x14x14x64xf32>, %init: tensor<1x7x7x64xf32>) -> tensor<1x7x7x64xf32> {
  %window = tensor.empty() : tensor<2x2xf32>
  %0 = linalg.pooling_nhwc_max
      {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
      ins(%input, %window : tensor<1x14x14x64xf32>, tensor<2x2xf32>)
      outs(%init : tensor<1x7x7x64xf32>) -> tensor<1x7x7x64xf32>
  return %0 : tensor<1x7x7x64xf32>
}

// The same pool in NCHW, the layout ONNX imports. This one *does* transpose,
// on both sides, and its shim costs three CPU dispatches against the NHWC
// shim's two. Running both is how a transpose-permutation error shows up as a
// difference between two cases rather than as a uniformly wrong gate.
func.func @max_pool_nchw_s2(%input: tensor<1x64x14x14xf32>, %init: tensor<1x64x7x7xf32>) -> tensor<1x64x7x7xf32> {
  %window = tensor.empty() : tensor<2x2xf32>
  %0 = linalg.pooling_nchw_max
      {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
      ins(%input, %window : tensor<1x64x14x14xf32>, tensor<2x2xf32>)
      outs(%init : tensor<1x64x7x7xf32>) -> tensor<1x64x7x7xf32>
  return %0 : tensor<1x64x7x7xf32>
}

// Stride 1, which routes to a *different executable*: stride is baked, one
// per value. Picking the wrong one is a wrong answer rather than a decline,
// so both strides need a differential of their own.
func.func @max_pool_nhwc_s1(%input: tensor<1x8x8x64xf32>, %init: tensor<1x7x7x64xf32>) -> tensor<1x7x7x64xf32> {
  %window = tensor.empty() : tensor<2x2xf32>
  %0 = linalg.pooling_nhwc_max
      {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%input, %window : tensor<1x8x8x64xf32>, tensor<2x2xf32>)
      outs(%init : tensor<1x7x7x64xf32>) -> tensor<1x7x7x64xf32>
  return %0 : tensor<1x7x7x64xf32>
}

func.func @max_pool_nchw_s1(%input: tensor<1x64x8x8xf32>, %init: tensor<1x64x7x7xf32>) -> tensor<1x64x7x7xf32> {
  %window = tensor.empty() : tensor<2x2xf32>
  %0 = linalg.pooling_nchw_max
      {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%input, %window : tensor<1x64x8x8xf32>, tensor<2x2xf32>)
      outs(%init : tensor<1x64x7x7xf32>) -> tensor<1x64x7x7xf32>
  return %0 : tensor<1x64x7x7xf32>
}

// An 8x8 window is MAX_DIRECT_KERNEL: hardware-confirmed, with a 16x16 window
// rejected by the hardware outright. 9x9 is covered as a CPU-fallback
// boundary by rocket_pooling_max_match_boundaries.mlir, so the accepted side
// needs an exact differential here or the ceiling could regress on its own.
func.func @max_pool_kernel_8x8(%input: tensor<1x8x8x64xf32>, %init: tensor<1x1x1x64xf32>) -> tensor<1x1x1x64xf32> {
  %window = tensor.empty() : tensor<8x8xf32>
  %0 = linalg.pooling_nhwc_max
      {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%input, %window : tensor<1x8x8x64xf32>, tensor<8x8xf32>)
      outs(%init : tensor<1x1x1x64xf32>) -> tensor<1x1x1x64xf32>
  return %0 : tensor<1x1x1x64xf32>
}

// Wide enough that PoolingPlan has to split it into tiles rather than program
// one direct pass -- the compiled-path counterpart of the raw
// `tiled_pooling_matches_oracle`. LIMITS.md's direct tile-width table is why
// this matters: past the boundary the hardware *hangs* rather than returning
// wrong data, and the planner tiling correctly is the only thing standing
// between a model and that hang.
func.func @max_pool_wide(%input: tensor<1x64x64x32xf32>, %init: tensor<1x32x32x32xf32>) -> tensor<1x32x32x32xf32> {
  %window = tensor.empty() : tensor<2x2xf32>
  %0 = linalg.pooling_nhwc_max
      {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
      ins(%input, %window : tensor<1x64x64x32xf32>, tensor<2x2xf32>)
      outs(%init : tensor<1x32x32x32xf32>) -> tensor<1x32x32x32xf32>
  return %0 : tensor<1x32x32x32xf32>
}

// Min pool, NHWC, both strides. There is no NCHW pair: linalg defines
// pooling_nchw_max but no pooling_nchw_min, so the layout simply has no op.
//
// Exact for the same reason max is -- a min pool also returns one of its
// inputs unchanged -- so the f16-exact fixtures make the whole path lossless.
func.func @min_pool_nhwc_s2(%input: tensor<1x14x14x64xf32>, %init: tensor<1x7x7x64xf32>) -> tensor<1x7x7x64xf32> {
  %window = tensor.empty() : tensor<2x2xf32>
  %0 = linalg.pooling_nhwc_min
      {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
      ins(%input, %window : tensor<1x14x14x64xf32>, tensor<2x2xf32>)
      outs(%init : tensor<1x7x7x64xf32>) -> tensor<1x7x7x64xf32>
  return %0 : tensor<1x7x7x64xf32>
}

func.func @min_pool_nhwc_s1(%input: tensor<1x8x8x64xf32>, %init: tensor<1x7x7x64xf32>) -> tensor<1x7x7x64xf32> {
  %window = tensor.empty() : tensor<2x2xf32>
  %0 = linalg.pooling_nhwc_min
      {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%input, %window : tensor<1x8x8x64xf32>, tensor<2x2xf32>)
      outs(%init : tensor<1x7x7x64xf32>) -> tensor<1x7x7x64xf32>
  return %0 : tensor<1x7x7x64xf32>
}

// A min pool wide enough to force PoolingPlan to tile. This matters more for
// min than for max: min is the method with no measured pad-fill identity at
// any precision, so if tiling ever introduced padding of its own the driver
// would have to refuse it. It does not -- `padded` is derived from the
// executable's own baked pad fields -- and this is the case that says so.
func.func @min_pool_wide(%input: tensor<1x64x64x32xf32>, %init: tensor<1x32x32x32xf32>) -> tensor<1x32x32x32xf32> {
  %window = tensor.empty() : tensor<2x2xf32>
  %0 = linalg.pooling_nhwc_min
      {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
      ins(%input, %window : tensor<1x64x64x32xf32>, tensor<2x2xf32>)
      outs(%init : tensor<1x32x32x32xf32>) -> tensor<1x32x32x32xf32>
  return %0 : tensor<1x32x32x32xf32>
}

// The two functions below put *two* Rocket dispatches in one function, so they
// share a command buffer -- what a real model does, and what no single-pool
// case here covers. The pools are independent (no data flows between them),
// so a difference cannot be explained by one feeding the other.
//
// `mixed_avg_then_max` is the one that matters: it crosses a *method*
// boundary within one command buffer, changing the PPU's reduction and its
// pad-fill identity between two submits. That class of transition is exactly
// where this hardware has misbehaved before -- ISSUES.md C8 was a register
// left set across a precision change, and the depthwise-to-dense quiescence
// dwell exists for the same reason -- so the pooling analogue is worth
// asserting rather than assuming.
func.func @mixed_max_then_max(%a: tensor<1x14x14x64xf32>, %ai: tensor<1x7x7x64xf32>, %b: tensor<1x8x8x64xf32>, %bi: tensor<1x7x7x64xf32>) -> (tensor<1x7x7x64xf32>, tensor<1x7x7x64xf32>) {
  %wa = tensor.empty() : tensor<2x2xf32>
  %wb = tensor.empty() : tensor<2x2xf32>
  %0 = linalg.pooling_nhwc_max
      {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
      ins(%a, %wa : tensor<1x14x14x64xf32>, tensor<2x2xf32>)
      outs(%ai : tensor<1x7x7x64xf32>) -> tensor<1x7x7x64xf32>
  %1 = linalg.pooling_nhwc_max
      {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%b, %wb : tensor<1x8x8x64xf32>, tensor<2x2xf32>)
      outs(%bi : tensor<1x7x7x64xf32>) -> tensor<1x7x7x64xf32>
  return %0, %1 : tensor<1x7x7x64xf32>, tensor<1x7x7x64xf32>
}

func.func @mixed_avg_then_max(%a: tensor<1x64x16x16xf32>, %ai: tensor<1x64x15x15xf32>, %b: tensor<1x14x14x64xf32>, %bi: tensor<1x7x7x64xf32>) -> (tensor<1x64x15x15xf32>, tensor<1x7x7x64xf32>) {
  %wa = tensor.empty() : tensor<2x2xf32>
  %wb = tensor.empty() : tensor<2x2xf32>
  %0 = linalg.pooling_nchw_sum
      {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%a, %wa : tensor<1x64x16x16xf32>, tensor<2x2xf32>)
      outs(%ai : tensor<1x64x15x15xf32>) -> tensor<1x64x15x15xf32>
  %1 = linalg.pooling_nhwc_max
      {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
      ins(%b, %wb : tensor<1x14x14x64xf32>, tensor<2x2xf32>)
      outs(%bi : tensor<1x7x7x64xf32>) -> tensor<1x7x7x64xf32>
  return %0, %1 : tensor<1x64x15x15xf32>, tensor<1x7x7x64xf32>
}

// Min and max in one command buffer -- the two reductions that sit at
// opposite ends of the same window, and the pair most likely to expose a PPU
// register left set across the transition. Both halves are exact, so this
// case cannot be explained away by tolerance.
func.func @mixed_min_then_max(%a: tensor<1x14x14x64xf32>, %ai: tensor<1x7x7x64xf32>, %b: tensor<1x8x8x64xf32>, %bi: tensor<1x7x7x64xf32>) -> (tensor<1x7x7x64xf32>, tensor<1x7x7x64xf32>) {
  %wa = tensor.empty() : tensor<2x2xf32>
  %wb = tensor.empty() : tensor<2x2xf32>
  %0 = linalg.pooling_nhwc_min
      {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
      ins(%a, %wa : tensor<1x14x14x64xf32>, tensor<2x2xf32>)
      outs(%ai : tensor<1x7x7x64xf32>) -> tensor<1x7x7x64xf32>
  %1 = linalg.pooling_nhwc_max
      {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%b, %wb : tensor<1x8x8x64xf32>, tensor<2x2xf32>)
      outs(%bi : tensor<1x7x7x64xf32>) -> tensor<1x7x7x64xf32>
  return %0, %1 : tensor<1x7x7x64xf32>, tensor<1x7x7x64xf32>
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


def build_raw_test(linker: str) -> Path:
    command = [
        "cargo",
        "test",
        "-p",
        "iree-rocket-hal",
        "--release",
        "--target",
        "aarch64-unknown-linux-gnu",
        "--test",
        "pooling_oracle_hw",
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
        if target.get("name") == "pooling_oracle_hw" and candidate:
            executable = Path(candidate)
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    if executable is None:
        raise SystemExit("cargo did not report the pooling_oracle_hw executable")
    return executable


def run_raw_gate(host: str, remote_dir: str, linker: str) -> None:
    executable = build_raw_test(linker)
    remote_executable = f"{remote_dir}/pooling_oracle_hw"
    run(["scp", str(executable), f"{host}:{remote_executable}"])
    # One `--exact` invocation per test, chained: each is its own process, so
    # a case cannot inherit device state from the one before it in the same
    # address space. The NPU can still be left sick across processes, which is
    # what `wait_for_quiet_npu` is for.
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
        for test in RAW_TESTS
    ]
    remote_command = " && ".join(
        [f"chmod +x {shlex.quote(remote_executable)}", *test_commands]
    )
    run(["ssh", host, remote_command])


# A max pool returns one of its inputs unchanged, so a fixture whose every
# value is exactly representable in f16 survives the shim's f32 -> f16 -> f32
# round trip bit for bit -- which is what lets every max case below be gated
# at atol = rtol = 0. Generating in f16 and widening is the whole trick.
def f16_exact(rng: np.random.Generator, *shape: int) -> np.ndarray:
    return rng.uniform(-1.0, 1.0, size=shape).astype(np.float16).astype(np.float32)


# The identity for a max reduction. Finite rather than -inf: the fixtures live
# in [-1, 1], so this is indistinguishable from -inf for the reduction while
# keeping infinities out of the differential entirely. linalg defines a max
# pool as `O = max(O, I)`, so this value really is read -- it is not merely a
# destination buffer.
MAX_POOL_INIT = -1.0e30

# The identity for a min reduction, by the same argument.
MIN_POOL_INIT = 1.0e30


def write_compiled_fixture(work_dir: Path) -> None:
    (work_dir / "pooling.mlir").write_text(POOLING_MLIR)
    rng = np.random.default_rng(20260905)

    def max_init(*shape: int) -> np.ndarray:
        return np.full(shape, MAX_POOL_INIT, dtype=np.float32)

    # Average pools accumulate, so their initialiser is zero and their inputs
    # do not need to be f16-exact -- the reciprocal round trip dominates any
    # demotion error anyway.
    np.save(
        work_dir / "avg_global_input.npy",
        rng.uniform(-0.25, 0.25, size=(1, 1792, 7, 7)).astype(np.float32),
    )
    np.save(work_dir / "avg_global_init.npy", np.zeros((1, 1792, 1, 1), dtype=np.float32))

    np.save(
        work_dir / "avg_2x2_input.npy",
        rng.uniform(-0.25, 0.25, size=(1, 64, 16, 16)).astype(np.float32),
    )
    np.save(work_dir / "avg_2x2_init.npy", np.zeros((1, 64, 15, 15), dtype=np.float32))

    # Every max fixture is f16-exact, which is what the exact tolerances below
    # depend on. Non-uniform across x, y *and* channel: a layout or addressing
    # fault displaces values rather than changing their count, so uniform data
    # cannot see it (see the `Counting`-pattern note -- a constant input only
    # ever proves coverage).
    np.save(work_dir / "max_nhwc_s2_input.npy", f16_exact(rng, 1, 14, 14, 64))
    np.save(work_dir / "max_nhwc_s2_init.npy", max_init(1, 7, 7, 64))

    np.save(work_dir / "max_nchw_s2_input.npy", f16_exact(rng, 1, 64, 14, 14))
    np.save(work_dir / "max_nchw_s2_init.npy", max_init(1, 64, 7, 7))

    np.save(work_dir / "max_nhwc_s1_input.npy", f16_exact(rng, 1, 8, 8, 64))
    np.save(work_dir / "max_nhwc_s1_init.npy", max_init(1, 7, 7, 64))

    np.save(work_dir / "max_nchw_s1_input.npy", f16_exact(rng, 1, 64, 8, 8))
    np.save(work_dir / "max_nchw_s1_init.npy", max_init(1, 64, 7, 7))

    np.save(work_dir / "max_8x8_input.npy", f16_exact(rng, 1, 8, 8, 64))
    np.save(work_dir / "max_8x8_init.npy", max_init(1, 1, 1, 64))

    np.save(work_dir / "max_wide_input.npy", f16_exact(rng, 1, 64, 64, 32))
    np.save(work_dir / "max_wide_init.npy", max_init(1, 32, 32, 32))

    # Min fixtures. f16-exact for the same reason the max ones are: a min pool
    # also returns one of its inputs unchanged, so the whole path is lossless
    # and the comparison can be exact.
    def min_init(*shape: int) -> np.ndarray:
        return np.full(shape, MIN_POOL_INIT, dtype=np.float32)

    np.save(work_dir / "min_nhwc_s2_input.npy", f16_exact(rng, 1, 14, 14, 64))
    np.save(work_dir / "min_nhwc_s2_init.npy", min_init(1, 7, 7, 64))

    np.save(work_dir / "min_nhwc_s1_input.npy", f16_exact(rng, 1, 8, 8, 64))
    np.save(work_dir / "min_nhwc_s1_init.npy", min_init(1, 7, 7, 64))

    np.save(work_dir / "min_wide_input.npy", f16_exact(rng, 1, 64, 64, 32))
    np.save(work_dir / "min_wide_init.npy", min_init(1, 32, 32, 32))

    np.save(work_dir / "mixed_min_a.npy", f16_exact(rng, 1, 14, 14, 64))
    np.save(work_dir / "mixed_min_ai.npy", min_init(1, 7, 7, 64))

    # Shared operands for the two-dispatch cases.
    np.save(work_dir / "mixed_max_a.npy", f16_exact(rng, 1, 14, 14, 64))
    np.save(work_dir / "mixed_max_ai.npy", max_init(1, 7, 7, 64))
    np.save(work_dir / "mixed_max_b.npy", f16_exact(rng, 1, 8, 8, 64))
    np.save(work_dir / "mixed_max_bi.npy", max_init(1, 7, 7, 64))

    np.save(
        work_dir / "mixed_avg_a.npy",
        rng.uniform(-0.25, 0.25, size=(1, 64, 16, 16)).astype(np.float32),
    )
    np.save(work_dir / "mixed_avg_ai.npy", np.zeros((1, 64, 15, 15), dtype=np.float32))


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

# Which executable each compiled function must reach. Without this a case
# whose shape quietly stopped matching would still "pass": it would run
# entirely on the CPU on both sides of the differential and agree with itself.
EXPECTED_EXECUTABLE = {
    "avg_pool_global": "rocket_pooling_executable",
    "avg_pool_2x2": "rocket_pooling_executable",
    "max_pool_nhwc_s2": "rocket_pooling_max_executable_s2",
    "max_pool_nchw_s2": "rocket_pooling_max_executable_s2",
    "max_pool_nhwc_s1": "rocket_pooling_max_executable",
    "max_pool_nchw_s1": "rocket_pooling_max_executable",
    "max_pool_kernel_8x8": "rocket_pooling_max_executable",
    "max_pool_wide": "rocket_pooling_max_executable_s2",
    "min_pool_nhwc_s2": "rocket_pooling_min_executable_s2",
    "min_pool_nhwc_s1": "rocket_pooling_min_executable",
    "min_pool_wide": "rocket_pooling_min_executable_s2",
    "mixed_max_then_max": "rocket_pooling_max_executable",
    "mixed_avg_then_max": "rocket_pooling_executable",
    "mixed_min_then_max": "rocket_pooling_min_executable_s2",
}


def compile_modules(
    work_dir: Path, compiler: Path, opt: Path, transform_spec: Path
) -> None:
    source = work_dir / "pooling.mlir"
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
    # "Placement pinning"). This matters more for pooling than for
    # convolution: every pooling shim ends in a CPU epilogue whose only
    # producer is the Rocket dispatch, so without the pin, Stream's affinity
    # analysis places that epilogue on @rocket_device and serialization dies
    # on an op that is not a pool.
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
    for function, executable in EXPECTED_EXECUTABLE.items():
        match = re.search(
            rf"util\.func public @{re.escape(function)}\b(?P<body>.*?)"
            r"(?=\n\s*util\.func (?:public|private) @|\Z)",
            preprocessing_text,
            re.DOTALL,
        )
        # The marker is the dispatch, not the call: @__transform_main inlines
        # the @call_rocket_* wrappers, so `util.call` no longer survives to
        # preprocessing.
        #
        # Matched with a negative lookahead rather than `in`, for the reason
        # the convolution gate records: the stride-1 executable's name is a
        # prefix of the stride-2 one, so a plain substring test for it can
        # never fail -- the stride-2 executable alone would satisfy it, and
        # the check whose whole job is to notice a case that stopped reaching
        # its matcher would be silently defeated.
        pattern = re.escape(f"flow.dispatch @{executable}") + r"(?!_s2)"
        if match is None or not re.search(pattern, match.group("body")):
            raise SystemExit(
                f"{function} no longer reaches {executable}; refusing to run a "
                "CPU-versus-CPU differential"
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
    running at all rather than by computing the wrong numbers -- a pool past
    the direct tile-width boundary hangs, the watchdog kills the job, and
    `iree-run-module` refuses the result rather than handing back a
    half-written buffer. Raising there would abort the whole gate over a
    failure it is meant to tolerate.
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
    # Only reached when the module actually ran, so there are outputs to
    # fetch. A case that did not execute wrote none, and the caller must not
    # go on to compare them.
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
    # Scalar, or one entry per output. A two-dispatch case needs the latter:
    # `mixed_avg_then_max` has one output that is exact and one that is not,
    # and a tolerance loose enough for the average would hide a displaced
    # element in the max.
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

    A board that never settles is reported and not waited on forever: the gate
    still runs, because a missing sysfs path should not stop a developer's
    run, but the warning says what to suspect.
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

    Max pools are compared exactly; see this module's docstring for why that
    is available here and not in the convolution gate. Averages take the
    --atol/--rtol defaults, because the PPU's average is a multiply by
    fp16(65536/k) that the shim multiplies back out.
    """
    return [
        Case(
            "avg_pool_global",
            ("avg_global_input.npy", "avg_global_init.npy"),
            ("avg_pool_global_out.npy",),
            atol,
            rtol,
        ),
        Case(
            "avg_pool_2x2",
            ("avg_2x2_input.npy", "avg_2x2_init.npy"),
            ("avg_pool_2x2_out.npy",),
            atol,
            rtol,
        ),
        Case(
            "max_pool_nhwc_s2",
            ("max_nhwc_s2_input.npy", "max_nhwc_s2_init.npy"),
            ("max_pool_nhwc_s2_out.npy",),
            0.0,
            0.0,
        ),
        Case(
            "max_pool_nchw_s2",
            ("max_nchw_s2_input.npy", "max_nchw_s2_init.npy"),
            ("max_pool_nchw_s2_out.npy",),
            0.0,
            0.0,
        ),
        Case(
            "max_pool_nhwc_s1",
            ("max_nhwc_s1_input.npy", "max_nhwc_s1_init.npy"),
            ("max_pool_nhwc_s1_out.npy",),
            0.0,
            0.0,
        ),
        Case(
            "max_pool_nchw_s1",
            ("max_nchw_s1_input.npy", "max_nchw_s1_init.npy"),
            ("max_pool_nchw_s1_out.npy",),
            0.0,
            0.0,
        ),
        Case(
            "max_pool_kernel_8x8",
            ("max_8x8_input.npy", "max_8x8_init.npy"),
            ("max_pool_kernel_8x8_out.npy",),
            0.0,
            0.0,
        ),
        Case(
            "max_pool_wide",
            ("max_wide_input.npy", "max_wide_init.npy"),
            ("max_pool_wide_out.npy",),
            0.0,
            0.0,
        ),
        Case(
            "min_pool_nhwc_s2",
            ("min_nhwc_s2_input.npy", "min_nhwc_s2_init.npy"),
            ("min_pool_nhwc_s2_out.npy",),
            0.0,
            0.0,
        ),
        Case(
            "min_pool_nhwc_s1",
            ("min_nhwc_s1_input.npy", "min_nhwc_s1_init.npy"),
            ("min_pool_nhwc_s1_out.npy",),
            0.0,
            0.0,
        ),
        Case(
            "min_pool_wide",
            ("min_wide_input.npy", "min_wide_init.npy"),
            ("min_pool_wide_out.npy",),
            0.0,
            0.0,
        ),
        # Two Rocket dispatches sharing a command buffer.
        Case(
            "mixed_max_then_max",
            ("mixed_max_a.npy", "mixed_max_ai.npy", "mixed_max_b.npy", "mixed_max_bi.npy"),
            ("mixed_max_then_max_0.npy", "mixed_max_then_max_1.npy"),
            0.0,
            0.0,
        ),
        # Per-output tolerances: the average half carries real f16 error, the
        # max half must still be exact.
        Case(
            "mixed_avg_then_max",
            ("mixed_avg_a.npy", "mixed_avg_ai.npy", "mixed_max_a.npy", "mixed_max_ai.npy"),
            ("mixed_avg_then_max_0.npy", "mixed_avg_then_max_1.npy"),
            (atol, 0.0),
            (rtol, 0.0),
        ),
        Case(
            "mixed_min_then_max",
            ("mixed_min_a.npy", "mixed_min_ai.npy", "mixed_max_b.npy", "mixed_max_bi.npy"),
            ("mixed_min_then_max_0.npy", "mixed_min_then_max_1.npy"),
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
    # with no entry in EXPECTED_EXECUTABLE runs with nothing checking that it
    # still reaches a matcher, which is the exact failure -- a silent
    # CPU-versus-CPU differential -- that map exists to prevent. An entry with
    # no case is the milder half, an assertion about something that never
    # runs, but it is just as much a lie about what this gate covers.
    described = set(EXPECTED_EXECUTABLE)
    present = {case.function for case in cases}
    if described != present:
        raise SystemExit(
            "EXPECTED_EXECUTABLE and build_cases disagree: "
            f"cases with no expected executable {sorted(present - described)}, "
            f"expected executables with no case {sorted(described - present)}"
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
            # Only reachable for a known failure -- every other case still
            # raises inside `run`. Not executing is a way of not matching, and
            # which way it failed is worth printing: if this case ever goes
            # back to returning numbers, the entry needs re-reading.
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
            raise SystemExit("compiled Rocket pool differs from the CPU reference")

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
        help="absolute tolerance for the average-pool cases; max pools are "
        "always compared exactly and ignore this",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=0.02,
        help="relative tolerance for the average-pool cases; max pools are "
        "always compared exactly and ignore this",
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
            print("\n== raw PoolingPlan/PPU oracle matrices ==")
            run_raw_gate(args.board, remote_dir, args.cross_linker)
        if not args.skip_compiled:
            print("\n== compiled VMFB CPU differential ==")
            with tempfile.TemporaryDirectory(
                prefix="iree-rocket-pooling-regression-"
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
    print(f"\nPASS: PoolingPlan/PPU {' and '.join(completed)} regression gate(s)")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(f"command failed with exit code {error.returncode}", file=sys.stderr)
        sys.exit(error.returncode or 1)
