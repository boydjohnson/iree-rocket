#!/usr/bin/env python3
"""Run the dense VGG convolution regression gates on an RK3588 board.

This runs two independent checks:

1. iree-rocket-hal's five-case raw ConvPlan/NPU dense-coefficient oracle.
2. A 30x30, Cin=512, Cout=512, 3x3 convolution compiled twice from MLIR:
   once for the host CPU and once for Rocket. The Rocket VMFB is executed
   through iree-run-module on the board and compared with the CPU VMFB.

The command exits nonzero if building, board execution, or comparison fails.
It requires Python numpy, ssh/scp access to the board, the aarch64 Rust target,
and built host/aarch64 IREE tools in their normal repository locations.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RAW_TEST = "dense_coefficient_vgg_blocks_match_oracle"
REMOTE_PREFIX = "/tmp/iree-rocket-conv-regression."

CONV_MLIR = """\
func.func @main(%input: tensor<1x32x32x512xf16>, %filter: tensor<3x3x512x512xf16>, %init: tensor<1x30x30x512xf32>) -> tensor<1x30x30x512xf32> {
  %0 = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%input, %filter : tensor<1x32x32x512xf16>, tensor<3x3x512x512xf16>)
      outs(%init : tensor<1x30x30x512xf32>) -> tensor<1x30x30x512xf32>
  return %0 : tensor<1x30x30x512xf32>
}
"""


def command_text(command: list[str]) -> str:
    return shlex.join(str(part) for part in command)


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print(f"+ {command_text(command)}", flush=True)
    subprocess.run(command, check=True, env=env)


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
    remote_command = (
        f"chmod +x {shlex.quote(remote_executable)} && "
        f"{shlex.quote(remote_executable)} --exact {RAW_TEST} "
        "--ignored --nocapture --test-threads=1"
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
    rocket_bytes = (work_dir / "rocket.vmfb").read_bytes()
    if b"rocket-flatbuffer-v1" not in rocket_bytes or b"RKT1" not in rocket_bytes:
        raise SystemExit("compiled module contains no serialized Rocket executable")


def run_cpu_reference(work_dir: Path, host_runtime: Path) -> None:
    run(
        [
            str(host_runtime),
            f"--module={work_dir / 'cpu.vmfb'}",
            "--function=main",
            "--device=local-task",
            f"--input=@{work_dir / 'input.npy'}",
            f"--input=@{work_dir / 'kernel.npy'}",
            f"--input=@{work_dir / 'init.npy'}",
            f"--output=@{work_dir / 'out_cpu.npy'}",
        ]
    )


def run_rocket_module(
    host: str, remote_dir: str, work_dir: Path, board_runtime: Path
) -> None:
    staged = [
        board_runtime,
        work_dir / "rocket.vmfb",
        work_dir / "input.npy",
        work_dir / "kernel.npy",
        work_dir / "init.npy",
    ]
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
                    "--function=main",
                    "--device=rocket",
                    "--device=local-task",
                    "--input=@input.npy",
                    "--input=@kernel.npy",
                    "--input=@init.npy",
                    "--output=@out_rocket.npy",
                ]
            ),
        ]
    )
    run(["ssh", host, remote_command])
    run(["scp", f"{host}:{remote_dir}/out_rocket.npy", str(work_dir / "out_rocket.npy")])


def compare_outputs(work_dir: Path, atol: float, rtol: float) -> None:
    cpu = np.load(work_dir / "out_cpu.npy").astype(np.float64)
    rocket = np.load(work_dir / "out_rocket.npy").astype(np.float64)
    if cpu.shape != rocket.shape:
        raise SystemExit(f"output shape mismatch: CPU {cpu.shape}, Rocket {rocket.shape}")
    absolute_error = np.abs(cpu - rocket)
    allowed_error = atol + rtol * np.abs(cpu)
    mismatches = int(np.count_nonzero(absolute_error > allowed_error))
    max_error = float(absolute_error.max(initial=0.0))
    print(
        "compiled VMFB differential: "
        f"max|error|={max_error:.8g}, mismatches={mismatches}/{cpu.size}, "
        f"atol={atol}, rtol={rtol}"
    )
    if mismatches:
        raise SystemExit("compiled Rocket convolution differs from the CPU reference")


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
) -> None:
    write_compiled_fixture(work_dir)
    compile_modules(work_dir, compiler, transform_spec)
    run_cpu_reference(work_dir, host_runtime)
    run_rocket_module(host, remote_dir, work_dir, board_runtime)
    compare_outputs(work_dir, atol, rtol)


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
            print("\n== raw dense-coefficient VGG oracle ==")
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
    print(f"\nPASS: dense VGG {' and '.join(completed)} regression gate(s)")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(f"command failed with exit code {error.returncode}", file=sys.stderr)
        sys.exit(error.returncode or 1)
