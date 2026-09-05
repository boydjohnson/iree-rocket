#!/usr/bin/env python3
"""Run dense and depthwise convolution regression gates on an RK3588 board.

This runs two independent checks:

1. iree-rocket-hal's raw ConvPlan/NPU dense-coefficient and exact int32
   accumulator oracle matrices.
2. Dense and depthwise convolutions compiled twice from MLIR:
   once for the host CPU and once for Rocket. The Rocket VMFB is executed
   through iree-run-module on the board and compared with the CPU VMFB.

The command exits nonzero if building, board execution, or comparison fails.
Some cases put two Rocket dispatches in one function so they share a command
buffer, which is what a real model does. One of those, `mixed_int8_then_depthwise`,
was a `known_failure` until 2026-09-05 -- it hung the NPU, 8 of 8 runs -- and is
now an ordinary gating case: ISSUES.md C8's fix (BRDMA left fetching on the
int32-accumulator path) was its cause too, and it has been exact 3 of 3 since.
A `known_failure` entry is still run and reported without failing the gate, and
the gate *does* fail if one starts passing, so such an entry cannot go stale.

`--only FUNCTION` (repeatable) runs a subset. The NPU's state is order-dependent
enough that one case can leave the device sick for the next, so a filter is how
a single case gets a clean verdict.

It requires Python numpy, ssh/scp access to the board, the aarch64 Rust target,
and built host/aarch64 IREE tools in their normal repository locations.
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
RAW_TESTS = (
    "dense_coefficient_vgg_blocks_match_oracle",
    "int8_accumulator_regression_matrix_matches_oracle",
)
REMOTE_PREFIX = "/tmp/iree-rocket-conv-regression."

CONV_MLIR = """\
func.func @main(%input: tensor<1x32x32x512xf16>, %filter: tensor<3x3x512x512xf16>, %init: tensor<1x30x30x512xf32>) -> tensor<1x30x30x512xf32> {
  %0 = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%input, %filter : tensor<1x32x32x512xf16>, tensor<3x3x512x512xf16>)
      outs(%init : tensor<1x30x30x512xf32>) -> tensor<1x30x30x512xf32>
  return %0 : tensor<1x30x30x512xf32>
}

// Cin=3 exercises the dense (Cin<=4) ARGB feature path -- the only one that
// hands an IREE buffer to the NPU directly instead of staging it through
// driver-owned scratch, and so the only one needing its own
// sync-for-device before the dispatch reads it.
//
// Deliberately written as f32 NCHW, unlike the f16 NHWC cases above. The
// spec's channels-last and f16-demotion passes then put a CPU dispatch
// immediately before the Rocket one, so the feature buffer the NPU reads was
// just written by the CPU and is still dirty in cache. That producer edge is
// the trigger: the same convolution taking its input straight from an entry
// point argument passes either way, because IREE's upload path syncs it. It
// is also the real network-stem shape.
func.func @dense_cin3(%input: tensor<1x3x34x34xf32>, %filter: tensor<16x3x3x3xf32>, %init: tensor<1x16x32x32xf32>) -> tensor<1x16x32x32xf32> {
  %0 = linalg.conv_2d_nchw_fchw
      {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%input, %filter : tensor<1x3x34x34xf32>, tensor<16x3x3x3xf32>)
      outs(%init : tensor<1x16x32x32xf32>) -> tensor<1x16x32x32xf32>
  return %0 : tensor<1x16x32x32xf32>
}

func.func @depthwise(%input: tensor<1x8x8x40xf16>, %filter: tensor<3x3x40xf16>, %init: tensor<1x6x6x40xf32>) -> tensor<1x6x6x40xf32> {
  %0 = linalg.depthwise_conv_2d_nhwc_hwc
      {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%input, %filter : tensor<1x8x8x40xf16>, tensor<3x3x40xf16>)
      outs(%init : tensor<1x6x6x40xf32>) -> tensor<1x6x6x40xf32>
  return %0 : tensor<1x6x6x40xf32>
}

// The int8 cases below are written in the *quantized* form an ONNX
// ConvInteger model arrives in, with a non-zero input zero point and a zero
// weight zero point (what ORT's quantize_dynamic always produces). That is
// deliberate: it is the only way to exercise the whole int8 offload chain --
// rocket-transpose-quantized-conv-to-nhwc, then
// iree-global-opt-quantized-conv-to-conv folding the zero point into a CPU
// correction, then the int8_accumulator dispatch. Handing the harness an
// already-folded plain i8 convolution would skip every part of that.
//
// The zero points are constants rather than function arguments only so these
// keep the (input, filter, init) signature the rest of the harness expects;
// the fold does not care whether they are constant.
//
// Dense arrives NCHW, matching what torch-mlir emits for ConvInteger, so the
// transpose pass is on the path. Depthwise arrives NHWC, for the same reason.

func.func @dense_int8(%input: tensor<1x64x32x32xi8>, %filter: tensor<128x64x1x1xi8>, %init: tensor<1x128x32x32xi32>) -> tensor<1x128x32x32xi32> {
  %izp = arith.constant 7 : i32
  %kzp = arith.constant 0 : i32
  %0 = linalg.conv_2d_nchw_fchw_q
      {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%input, %filter, %izp, %kzp : tensor<1x64x32x32xi8>, tensor<128x64x1x1xi8>, i32, i32)
      outs(%init : tensor<1x128x32x32xi32>) -> tensor<1x128x32x32xi32>
  return %0 : tensor<1x128x32x32xi32>
}

// This sits exactly on the dense 3x3 matcher's measured-good Cin ceiling.
// Cin=33 is hardware-proven wrong and covered as a CPU-fallback boundary by
// rocket_int8_match_boundaries.mlir; Cin=32 must remain an exact Rocket
// differential so the accepted side cannot regress independently.
func.func @dense_int8_3x3(%input: tensor<1x32x34x34xi8>, %filter: tensor<64x32x3x3xi8>, %init: tensor<1x64x32x32xi32>) -> tensor<1x64x32x32xi32> {
  %izp = arith.constant 11 : i32
  %kzp = arith.constant 0 : i32
  %0 = linalg.conv_2d_nchw_fchw_q
      {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%input, %filter, %izp, %kzp : tensor<1x32x34x34xi8>, tensor<64x32x3x3xi8>, i32, i32)
      outs(%init : tensor<1x64x32x32xi32>) -> tensor<1x64x32x32xi32>
  return %0 : tensor<1x64x32x32xi32>
}

// These two cases isolate the matcher's currently accepted Cout=512 edge.
// Cin stays deliberately small so a failure characterizes the output-channel
// limit rather than either of the independently measured Cin limits above.
// Cout above the old 512 cap. 88->528 is a real MobileNetV2 shape (four of
// them), and 768 is the corpus extent -- both were rejected outright before
// MAX_OUTPUT_CHANNELS was raised, so this is the guard on that range.
func.func @dense_int8_cout768_1x1(%input: tensor<1x88x32x32xi8>, %filter: tensor<768x88x1x1xi8>, %init: tensor<1x768x32x32xi32>) -> tensor<1x768x32x32xi32> {
  %izp = arith.constant 7 : i32
  %kzp = arith.constant 0 : i32
  %0 = linalg.conv_2d_nchw_fchw_q
      {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%input, %filter, %izp, %kzp : tensor<1x88x32x32xi8>, tensor<768x88x1x1xi8>, i32, i32)
      outs(%init : tensor<1x768x32x32xi32>) -> tensor<1x768x32x32xi32>
  return %0 : tensor<1x768x32x32xi32>
}

func.func @dense_int8_cout512_1x1(%input: tensor<1x16x32x32xi8>, %filter: tensor<512x16x1x1xi8>, %init: tensor<1x512x32x32xi32>) -> tensor<1x512x32x32xi32> {
  %izp = arith.constant 13 : i32
  %kzp = arith.constant 0 : i32
  %0 = linalg.conv_2d_nchw_fchw_q
      {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%input, %filter, %izp, %kzp : tensor<1x16x32x32xi8>, tensor<512x16x1x1xi8>, i32, i32)
      outs(%init : tensor<1x512x32x32xi32>) -> tensor<1x512x32x32xi32>
  return %0 : tensor<1x512x32x32xi32>
}

func.func @dense_int8_cout512_3x3(%input: tensor<1x16x34x34xi8>, %filter: tensor<512x16x3x3xi8>, %init: tensor<1x512x32x32xi32>) -> tensor<1x512x32x32xi32> {
  %izp = arith.constant -9 : i32
  %kzp = arith.constant 0 : i32
  %0 = linalg.conv_2d_nchw_fchw_q
      {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%input, %filter, %izp, %kzp : tensor<1x16x34x34xi8>, tensor<512x16x3x3xi8>, i32, i32)
      outs(%init : tensor<1x512x32x32xi32>) -> tensor<1x512x32x32xi32>
  return %0 : tensor<1x512x32x32xi32>
}

func.func @depthwise_int8(%input: tensor<1x34x34x64xi8>, %filter: tensor<3x3x64xi8>, %init: tensor<1x32x32x64xi32>) -> tensor<1x32x32x64xi32> {
  %izp = arith.constant -5 : i32
  %kzp = arith.constant 0 : i32
  %0 = linalg.depthwise_conv_2d_nhwc_hwc_q
      {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%input, %filter, %izp, %kzp : tensor<1x34x34x64xi8>, tensor<3x3x64xi8>, i32, i32)
      outs(%init : tensor<1x32x32x64xi32>) -> tensor<1x32x32x64xi32>
  return %0 : tensor<1x32x32x64xi32>
}

// The three functions below put *two* Rocket dispatches in one function, so
// they share a command buffer -- what a real model does and what no other
// case here covers. Each convolution is independent (no data flows between
// them), so a difference cannot be explained by one feeding the other.
//
// The depthwise ones are authored directly in f16 rather than f32: the
// matchers require f16/f16/f32 typing, and rocket-demote-conv-inputs-to-f16
// deliberately does not demote depthwise (see its scope comment), so writing
// f32 here would leave the depthwise on the CPU and test nothing.
//
// `mixed_int8_then_depthwise` used to hang the NPU here (8/8). Its cause was
// ISSUES.md C8 -- BRDMA left fetching on the int32-accumulator path -- and it
// gates normally since that was fixed on 2026-09-05.
func.func @mixed_int8_then_depthwise(%qi: tensor<1x64x32x32xi8>, %qf: tensor<128x64x1x1xi8>, %qo: tensor<1x128x32x32xi32>, %di: tensor<1x144x113x113xf16>, %df: tensor<144x3x3xf16>, %do: tensor<1x144x56x56xf32>) -> (tensor<1x128x32x32xi32>, tensor<1x144x56x56xf32>) {
  %izp = arith.constant 7 : i32
  %kzp = arith.constant 0 : i32
  %0 = linalg.conv_2d_nchw_fchw_q
      {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%qi, %qf, %izp, %kzp : tensor<1x64x32x32xi8>, tensor<128x64x1x1xi8>, i32, i32)
      outs(%qo : tensor<1x128x32x32xi32>) -> tensor<1x128x32x32xi32>
  %1 = linalg.depthwise_conv_2d_nchw_chw
      {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
      ins(%di, %df : tensor<1x144x113x113xf16>, tensor<144x3x3xf16>)
      outs(%do : tensor<1x144x56x56xf32>) -> tensor<1x144x56x56xf32>
  return %0, %1 : tensor<1x128x32x32xi32>, tensor<1x144x56x56xf32>
}

// The same two convolutions with the depthwise submitted first. Passes,
// which is what makes the failure above an ordering property rather than a
// property of either convolution.
func.func @mixed_depthwise_then_int8(%qi: tensor<1x64x32x32xi8>, %qf: tensor<128x64x1x1xi8>, %qo: tensor<1x128x32x32xi32>, %di: tensor<1x144x113x113xf16>, %df: tensor<144x3x3xf16>, %do: tensor<1x144x56x56xf32>) -> (tensor<1x128x32x32xi32>, tensor<1x144x56x56xf32>) {
  %izp = arith.constant 7 : i32
  %kzp = arith.constant 0 : i32
  %1 = linalg.depthwise_conv_2d_nchw_chw
      {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
      ins(%di, %df : tensor<1x144x113x113xf16>, tensor<144x3x3xf16>)
      outs(%do : tensor<1x144x56x56xf32>) -> tensor<1x144x56x56xf32>
  %0 = linalg.conv_2d_nchw_fchw_q
      {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%qi, %qf, %izp, %kzp : tensor<1x64x32x32xi8>, tensor<128x64x1x1xi8>, i32, i32)
      outs(%qo : tensor<1x128x32x32xi32>) -> tensor<1x128x32x32xi32>
  return %0, %1 : tensor<1x128x32x32xi32>, tensor<1x144x56x56xf32>
}

// Dense-then-depthwise again, but both fp16, so the *only* difference from
// `mixed_int8_then_depthwise` is the first dispatch's precision. Passes,
// which is what makes the failure a precision-mix property rather than a
// "two dispatches in one command buffer" one.
func.func @mixed_fp16_then_depthwise(%qi: tensor<1x64x32x32xf16>, %qf: tensor<128x64x1x1xf16>, %qo: tensor<1x128x32x32xf32>, %di: tensor<1x144x113x113xf16>, %df: tensor<144x3x3xf16>, %do: tensor<1x144x56x56xf32>) -> (tensor<1x128x32x32xf32>, tensor<1x144x56x56xf32>) {
  %0 = linalg.conv_2d_nchw_fchw
      {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%qi, %qf : tensor<1x64x32x32xf16>, tensor<128x64x1x1xf16>)
      outs(%qo : tensor<1x128x32x32xf32>) -> tensor<1x128x32x32xf32>
  %1 = linalg.depthwise_conv_2d_nchw_chw
      {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
      ins(%di, %df : tensor<1x144x113x113xf16>, tensor<144x3x3xf16>)
      outs(%do : tensor<1x144x56x56xf32>) -> tensor<1x144x56x56xf32>
  return %0, %1 : tensor<1x128x32x32xf32>, tensor<1x144x56x56xf32>
}

func.func @depthwise_int8_s2(%input: tensor<1x33x33x64xi8>, %filter: tensor<3x3x64xi8>, %init: tensor<1x16x16x64xi32>) -> tensor<1x16x16x64xi32> {
  %izp = arith.constant -5 : i32
  %kzp = arith.constant 0 : i32
  %0 = linalg.depthwise_conv_2d_nhwc_hwc_q
      {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
      ins(%input, %filter, %izp, %kzp : tensor<1x33x33x64xi8>, tensor<3x3x64xi8>, i32, i32)
      outs(%init : tensor<1x16x16x64xi32>) -> tensor<1x16x16x64xi32>
  return %0 : tensor<1x16x16x64xi32>
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
        "conv2d_oracle_hw",
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
        if target.get("name") == "conv2d_oracle_hw" and candidate:
            executable = Path(candidate)
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    if executable is None:
        raise SystemExit("cargo did not report the conv2d_oracle_hw executable")
    return executable


def run_raw_gate(host: str, remote_dir: str, linker: str) -> None:
    executable = build_raw_test(linker)
    remote_executable = f"{remote_dir}/conv2d_oracle_hw"
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
        for test in RAW_TESTS
    ]
    remote_command = " && ".join(
        [f"chmod +x {shlex.quote(remote_executable)}", *test_commands]
    )
    run(["ssh", host, remote_command])


def write_compiled_fixture(work_dir: Path) -> None:
    (work_dir / "conv.mlir").write_text(CONV_MLIR)
    rng = np.random.default_rng(20260830)
    weights = rng.uniform(-0.5, 0.5, size=(3, 3, 512, 512)).astype(np.float16)
    np.save(work_dir / "kernel.npy", weights)

    input_tensor = np.zeros((1, 32, 32, 512), dtype=np.float16)
    input_tensor[:, 1:31, 1:31, :] = rng.uniform(
        -0.25, 0.25, size=(1, 30, 30, 512)
    ).astype(np.float16)
    np.save(work_dir / "input.npy", input_tensor)
    np.save(work_dir / "init.npy", np.zeros((1, 30, 30, 512), dtype=np.float32))

    # Non-uniform across x, y *and* channel: a dense-path addressing or
    # staleness fault displaces values rather than changing their count, so
    # uniform data cannot see it.
    np.save(
        work_dir / "dense_cin3_kernel.npy",
        rng.uniform(-0.5, 0.5, size=(16, 3, 3, 3)).astype(np.float32),
    )
    np.save(
        work_dir / "dense_cin3_input.npy",
        rng.uniform(-0.25, 0.25, size=(1, 3, 34, 34)).astype(np.float32),
    )
    np.save(
        work_dir / "dense_cin3_init.npy", np.zeros((1, 16, 32, 32), dtype=np.float32)
    )

    # Shared operands for the three mixed-precision two-dispatch cases. The
    # int8 and fp16 dense halves are separate arrays because the two
    # functions take different element types for the same argument slot.
    np.save(work_dir / "mixed_qi_i8.npy", rng.integers(-128, 128, (1, 64, 32, 32), dtype=np.int8))
    np.save(work_dir / "mixed_qf_i8.npy", rng.integers(-128, 128, (128, 64, 1, 1), dtype=np.int8))
    np.save(work_dir / "mixed_qo_i32.npy", np.zeros((1, 128, 32, 32), dtype=np.int32))
    np.save(
        work_dir / "mixed_qi_f16.npy",
        rng.uniform(-0.5, 0.5, size=(1, 64, 32, 32)).astype(np.float16),
    )
    np.save(
        work_dir / "mixed_qf_f16.npy",
        rng.uniform(-0.5, 0.5, size=(128, 64, 1, 1)).astype(np.float16),
    )
    np.save(work_dir / "mixed_qo_f32.npy", np.zeros((1, 128, 32, 32), dtype=np.float32))
    np.save(
        work_dir / "mixed_di.npy",
        rng.uniform(-0.5, 0.5, size=(1, 144, 113, 113)).astype(np.float16),
    )
    np.save(
        work_dir / "mixed_df.npy",
        rng.uniform(-0.5, 0.5, size=(144, 3, 3)).astype(np.float16),
    )
    np.save(work_dir / "mixed_do.npy", np.zeros((1, 144, 56, 56), dtype=np.float32))

    depthwise_weights = rng.uniform(-0.5, 0.5, size=(3, 3, 40)).astype(np.float16)
    depthwise_input = rng.uniform(-0.25, 0.25, size=(1, 8, 8, 40)).astype(np.float16)
    np.save(work_dir / "depthwise_kernel.npy", depthwise_weights)
    np.save(work_dir / "depthwise_input.npy", depthwise_input)
    np.save(work_dir / "depthwise_init.npy", np.zeros((1, 6, 6, 40), dtype=np.float32))

    # int8 fixtures. The values span the full signed range so a wrong
    # zero-point fold or a mis-permuted transpose cannot cancel out, and the
    # accumulators stay far inside i32.
    def i8(*shape: int) -> np.ndarray:
        return rng.integers(-128, 128, size=shape, dtype=np.int8)

    np.save(work_dir / "dense_int8_input.npy", i8(1, 64, 32, 32))
    np.save(work_dir / "dense_int8_kernel.npy", i8(128, 64, 1, 1))
    np.save(
        work_dir / "dense_int8_init.npy", np.zeros((1, 128, 32, 32), dtype=np.int32)
    )

    np.save(work_dir / "dense_int8_3x3_input.npy", i8(1, 32, 34, 34))
    np.save(work_dir / "dense_int8_3x3_kernel.npy", i8(64, 32, 3, 3))
    np.save(
        work_dir / "dense_int8_3x3_init.npy",
        np.zeros((1, 64, 32, 32), dtype=np.int32),
    )

    np.save(work_dir / "dense_int8_cout768_1x1_input.npy", i8(1, 88, 32, 32))
    np.save(work_dir / "dense_int8_cout768_1x1_kernel.npy", i8(768, 88, 1, 1))
    np.save(
        work_dir / "dense_int8_cout768_1x1_init.npy",
        np.zeros((1, 768, 32, 32), dtype=np.int32),
    )
    np.save(work_dir / "dense_int8_cout512_1x1_input.npy", i8(1, 16, 32, 32))
    np.save(work_dir / "dense_int8_cout512_1x1_kernel.npy", i8(512, 16, 1, 1))
    np.save(
        work_dir / "dense_int8_cout512_1x1_init.npy",
        np.zeros((1, 512, 32, 32), dtype=np.int32),
    )

    np.save(work_dir / "dense_int8_cout512_3x3_input.npy", i8(1, 16, 34, 34))
    np.save(work_dir / "dense_int8_cout512_3x3_kernel.npy", i8(512, 16, 3, 3))
    np.save(
        work_dir / "dense_int8_cout512_3x3_init.npy",
        np.zeros((1, 512, 32, 32), dtype=np.int32),
    )

    np.save(work_dir / "depthwise_int8_input.npy", i8(1, 34, 34, 64))
    np.save(work_dir / "depthwise_int8_kernel.npy", i8(3, 3, 64))
    np.save(
        work_dir / "depthwise_int8_init.npy",
        np.zeros((1, 32, 32, 64), dtype=np.int32),
    )

    np.save(work_dir / "depthwise_int8_s2_input.npy", i8(1, 33, 33, 64))
    np.save(work_dir / "depthwise_int8_s2_kernel.npy", i8(3, 3, 64))
    np.save(
        work_dir / "depthwise_int8_s2_init.npy",
        np.zeros((1, 16, 16, 64), dtype=np.int32),
    )


def compile_modules(work_dir: Path, compiler: Path, transform_spec: Path) -> None:
    source = work_dir / "conv.mlir"
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
    run(
        [
            str(compiler),
            str(source),
            "-o",
            str(work_dir / "rocket.vmfb"),
            f"--iree-preprocessing-transform-spec-filename={transform_spec}",
            "--iree-llvmcpu-target-triple=aarch64-linux-gnu",
            "--iree-hal-target-device=rocket_device=rocket",
            "--iree-hal-target-device=cpu_device=local",
            "--iree-hal-local-target-device-backends=llvm-cpu",
            "--iree-hal-default-device=cpu_device",
            "--iree-hal-indirect-command-buffers=false",
            "--iree-llvmcpu-target-cpu=generic",
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
            "--iree-hal-target-device=rocket_device=rocket",
            "--iree-hal-target-device=cpu_device=local",
            "--iree-hal-local-target-device-backends=llvm-cpu",
            "--iree-hal-default-device=cpu_device",
            "--iree-hal-indirect-command-buffers=false",
            "--iree-llvmcpu-target-cpu=generic",
            "--compile-to=preprocessing",
            "--mlir-print-op-generic=false",
        ]
    )
    preprocessing_text = preprocessing.read_text()
    for function in (
        "dense_int8_3x3",
        "dense_int8_cout512_1x1",
        "dense_int8_cout512_3x3",
    ):
        match = re.search(
            rf"util\.func public @{re.escape(function)}\b(?P<body>.*?)"
            r"(?=\n\s*util\.func (?:public|private) @|\Z)",
            preprocessing_text,
            re.DOTALL,
        )
        if match is None or "util.call @call_rocket_dynamic_conv2d_int8" not in match.group(
            "body"
        ):
            raise SystemExit(
                f"{function} no longer reaches its Rocket matcher; refusing to run "
                "a CPU-versus-CPU differential"
            )
    rocket_bytes = (work_dir / "rocket.vmfb").read_bytes()
    if b"rocket-flatbuffer-v1" not in rocket_bytes or b"RKT1" not in rocket_bytes:
        raise SystemExit("compiled module contains no serialized Rocket executable")
    # Without this, a case whose shape quietly stops matching its matcher
    # would still "pass": it would just run entirely on the CPU on both
    # sides of the differential and agree with itself.
    # Matched with a negative lookahead rather than `in`: the stride-1
    # depthwise name is a prefix of the stride-2 one, so a plain substring
    # test for it can never fail -- the stride-2 executable alone satisfies
    # it. That silently defeated this check, whose whole job is to notice a
    # case that stopped reaching its matcher.
    for executable in (
        b"rocket_dynamic_int8_executable",
        b"rocket_dynamic_depthwise_int8_executable",
        b"rocket_dynamic_depthwise_int8_executable_s2",
        # The fp16 pair the mixed two-dispatch cases need.
        b"rocket_dynamic_executable",
        b"rocket_dynamic_depthwise_executable_s2",
    ):
        if not re.search(re.escape(executable) + rb"(?!_s2)", rocket_bytes):
            raise SystemExit(
                f"compiled module has no {executable.decode()}: the int8 case for it "
                "no longer reaches a Rocket matcher, so the gate would test nothing"
            )


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
    running at all rather than by computing the wrong numbers. Since the
    driver grew a hung-job deadline, `mixed_int8_then_depthwise` does
    exactly that: it hangs the NPU, the watchdog kills the job at ~500 ms,
    and `iree-run-module` now correctly refuses the result instead of
    handing back a half-written buffer. Raising there would abort the whole
    gate over a failure it is meant to tolerate.
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
    # its int8 half is exact integer arithmetic while its fp16 half is not,
    # and a tolerance loose enough for the fp16 output would hide a real
    # error of tens of thousands in an i32 accumulator.
    atol: float | tuple[float, ...]
    rtol: float | tuple[float, ...]
    known_failure: str | None = None

    def tolerances(self, index: int) -> tuple[float, float]:
        atol = self.atol[index] if isinstance(self.atol, tuple) else self.atol
        rtol = self.rtol[index] if isinstance(self.rtol, tuple) else self.rtol
        return atol, rtol


MIXED_INT8_INPUTS = (
    "mixed_qi_i8.npy",
    "mixed_qf_i8.npy",
    "mixed_qo_i32.npy",
    "mixed_di.npy",
    "mixed_df.npy",
    "mixed_do.npy",
)

MIXED_FP16_INPUTS = (
    "mixed_qi_f16.npy",
    "mixed_qf_f16.npy",
    "mixed_qo_f32.npy",
    "mixed_di.npy",
    "mixed_df.npy",
    "mixed_do.npy",
)


def wait_for_quiet_npu(host: str, budget_seconds: float = 20.0) -> None:
    """Blocks until every NPU core's runtime-PM state reads `suspended`.

    A hung job leaves the device sick for seconds, and the state crosses
    processes -- so a gate started right after one (a deliberate repro, an
    aborted run, another session) can have its *first* case fail for reasons
    that have nothing to do with it. That happened on 2026-09-05: the compiled
    gate aborted on case 0, a pure fp16 512-channel 3x3, which then passed 3 of
    3 in isolation and passes in the full gate from a quiet device. Waiting
    here turns "the first case hangs" into either a clean start or an explicit
    complaint about the board.

    Cores read `suspended` once the driver's autosuspend has cycled the power
    domain, which is also what clears the wide-output poisoning ISSUES.md C8
    describes. A board that never settles is reported and not waited on
    forever: the gate still runs, because a missing sysfs path should not stop
    a developer's run, but the warning says what to suspect.
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


def run_compiled_gate(
    host: str,
    remote_dir: str,
    work_dir: Path,
    compiler: Path,
    host_runtime: Path,
    board_runtime: Path,
    transform_spec: Path,
    atol: float,
    rtol: float,
    only: list[str] | None = None,
) -> None:
    write_compiled_fixture(work_dir)
    compile_modules(work_dir, compiler, transform_spec)
    wait_for_quiet_npu(host)
    # int8 cases are compared exactly (atol=rtol=0), not with the fp16
    # tolerances: the whole path is integer arithmetic, so any difference at
    # all is a bug. That matters more than it sounds -- the failure mode the
    # CBUF sweeps found is an all-zero output that completes successfully,
    # and a tolerance wide enough for fp16 rounding would still catch that,
    # but an exact check also catches a single wrong accumulator lane.
    cases = [
        Case(
            "main",
            ("input.npy", "kernel.npy", "init.npy"),
            ("out_rocket.npy",),
            atol,
            rtol,
        ),
        Case(
            "depthwise",
            ("depthwise_input.npy", "depthwise_kernel.npy", "depthwise_init.npy"),
            ("depthwise_out_rocket.npy",),
            atol,
            rtol,
        ),
        Case(
            "dense_int8",
            ("dense_int8_input.npy", "dense_int8_kernel.npy", "dense_int8_init.npy"),
            ("dense_int8_out_rocket.npy",),
            0.0,
            0.0,
        ),
        Case(
            "dense_cin3",
            ("dense_cin3_input.npy", "dense_cin3_kernel.npy", "dense_cin3_init.npy"),
            ("dense_cin3_out_rocket.npy",),
            1e-2,
            1e-2,
        ),
        Case(
            "dense_int8_3x3",
            ("dense_int8_3x3_input.npy", "dense_int8_3x3_kernel.npy", "dense_int8_3x3_init.npy"),
            ("dense_int8_3x3_out_rocket.npy",),
            0.0,
            0.0,
        ),
        Case(
            "dense_int8_cout768_1x1",
            (
                "dense_int8_cout768_1x1_input.npy",
                "dense_int8_cout768_1x1_kernel.npy",
                "dense_int8_cout768_1x1_init.npy",
            ),
            ("dense_int8_cout768_1x1_out_rocket.npy",),
            0.0,
            0.0,
        ),
        Case(
            "dense_int8_cout512_1x1",
            ("dense_int8_cout512_1x1_input.npy", "dense_int8_cout512_1x1_kernel.npy", "dense_int8_cout512_1x1_init.npy"),
            ("dense_int8_cout512_1x1_out_rocket.npy",),
            0.0,
            0.0,
        ),
        Case(
            "dense_int8_cout512_3x3",
            ("dense_int8_cout512_3x3_input.npy", "dense_int8_cout512_3x3_kernel.npy", "dense_int8_cout512_3x3_init.npy"),
            ("dense_int8_cout512_3x3_out_rocket.npy",),
            0.0,
            0.0,
        ),
        Case(
            "depthwise_int8",
            ("depthwise_int8_input.npy", "depthwise_int8_kernel.npy", "depthwise_int8_init.npy"),
            ("depthwise_int8_out_rocket.npy",),
            0.0,
            0.0,
        ),
        Case(
            "depthwise_int8_s2",
            ("depthwise_int8_s2_input.npy", "depthwise_int8_s2_kernel.npy", "depthwise_int8_s2_init.npy"),
            ("depthwise_int8_s2_out_rocket.npy",),
            0.0,
            0.0,
        ),
        # Two Rocket dispatches sharing a command buffer. Tolerances are
        # per output: the int8 half is exact, the fp16 depthwise half is not.
        Case(
            "mixed_fp16_then_depthwise",
            MIXED_FP16_INPUTS,
            ("mixed_fp16_then_depthwise_dense.npy", "mixed_fp16_then_depthwise_dw.npy"),
            1e-2,
            1e-2,
        ),
        Case(
            "mixed_depthwise_then_int8",
            MIXED_INT8_INPUTS,
            ("mixed_depthwise_then_int8_dense.npy", "mixed_depthwise_then_int8_dw.npy"),
            (0.0, 1e-2),
            (0.0, 1e-2),
        ),
        Case(
            "mixed_int8_then_depthwise",
            MIXED_INT8_INPUTS,
            ("mixed_int8_then_depthwise_dense.npy", "mixed_int8_then_depthwise_dw.npy"),
            (0.0, 1e-2),
            (0.0, 1e-2),
        ),
    ]
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
            # raises inside `run`. Not executing is a way of not matching,
            # and which way it failed is worth printing: if this case ever
            # goes back to returning numbers, the entry needs re-reading.
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
            raise SystemExit("compiled Rocket convolution differs from the CPU reference")

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
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--rtol", type=float, default=0.02)
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
        require_file(args.host_runtime, "host iree-run-module")
        require_file(args.board_runtime, "aarch64 iree-run-module")
        require_file(args.transform_spec, "Rocket transform spec")

    remote_dir = capture(
        ["ssh", args.board, f"mktemp -d {REMOTE_PREFIX}XXXXXX"]
    )
    if not remote_dir.startswith(REMOTE_PREFIX):
        raise SystemExit(f"board returned an unexpected temporary path: {remote_dir!r}")
    print(f"board staging directory: {remote_dir}")

    try:
        if not args.skip_raw:
            print("\n== raw ConvPlan/NPU oracle matrices ==")
            run_raw_gate(args.board, remote_dir, args.cross_linker)
        if not args.skip_compiled:
            print("\n== compiled VMFB CPU differential ==")
            with tempfile.TemporaryDirectory(
                prefix="iree-rocket-conv-regression-"
            ) as temporary:
                run_compiled_gate(
                    args.board,
                    remote_dir,
                    Path(temporary),
                    args.compiler,
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
    print(f"\nPASS: ConvPlan/NPU {' and '.join(completed)} regression gate(s)")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(f"command failed with exit code {error.returncode}", file=sys.stderr)
        sys.exit(error.returncode or 1)
