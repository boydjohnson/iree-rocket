#!/usr/bin/env python3
"""Places one model on the NPU, on the Mali GPU, on both, and on neither.

Section 6 of `agent-docs/vulkan-panvk-spike.md` left two things open: a module
that genuinely places work on both the `rocket` and the `vulkan-spirv`
backends, and any performance number at all for the GPU. This gate is the
answer to both, including the part of the first one that does not work.

Every arm is built from the *same* MLIR: a convolution the Rocket matchers
claim followed by a softmax no Rocket matcher will ever claim, so which half
goes where is a property of the model rather than of a flag.

    cpu     everything on llvm-cpu             --no-offload
    npu     rocket conv + llvm-cpu softmax     (the shipping configuration)
    gpu     everything on vulkan-spirv         --no-offload --cpu-device-backend=vulkan
    mixed   rocket conv + vulkan-spirv softmax --cpu-device-backend=vulkan

The first three run. **`mixed` does not**, and the gate asserts that it
compiles with the placement it claims and then records how its run fails
rather than passing or exiting: IREE 3.14 allocates a module's buffers from
one device's allocator and hands them to the other device unchanged, and
neither HAL driver can use the other's buffers. The Vulkan HAL type-checks
and returns FAILED_PRECONDITION ("buffer is not backed by the Vulkan HAL");
the Rocket HAL casts without checking and segfaults. The `mixed` arm is kept
because the compile half *is* the answer to the first open item and should
not silently rot, and because the day a driver learns to take the other's
buffers this arm is the test that says so.

Why four arms and not two: `--no-offload` is the only honest CPU baseline
here, because the transform spec rewrites conv layout whether or not anything
offloads (ISSUES.md M4), and a stock `iree-compile` module is 2.8x slower for
that reason alone.

Board conditions are read before and after every timing and printed with the
numbers. The Mali devfreq idles at 200 MHz against a 1000 MHz ceiling and
`simple_ondemand` is the only governor the kernel offers, so the clock is
pinned by raising `min_freq` to `max_freq` and then *confirmed* by reading
`cur_freq` back -- a floor the OPP table refuses is accepted silently.

Two numbers are reported per arm: a process-per-run wall time, which is what
a caller pays, and an `iree-benchmark-module` loop figure. They are kept
apart rather than averaged because a loop benchmark on this board overstates
by roughly 1.4-1.9x: it keeps every page resident and the real module touches
them once.

    python3 tools/mixed_device_gate.py --board orangepi
    python3 tools/mixed_device_gate.py --board orangepi --case conv_softmax_56
    python3 tools/mixed_device_gate.py --board orangepi --skip-timing

Needs an `iree-compile` carrying *both* the `vulkan-spirv` and `rocket`
target backends (spike doc section 5 has the configure line) and a board
runtime built with `-DROCKET_WITH_VULKAN=ON` and
`iree-build/patches/0001-vulkan-flush-invalidate-noncoherent-mappings.patch`
applied -- without the patch the GPU arm returns wrong data and does not say
so. Point `--compiler-lib` and `--board-runtime` at them.
"""

from __future__ import annotations

import argparse
import os
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

# The Mali's Vulkan device and llvmpipe's are both enumerated, and selecting
# by index is how you silently measure llvmpipe instead (spike doc section 4).
# llvmpipe's UUID is its driver version string in ASCII, so its first four
# bytes spell "mesa"; the Mali's do not.
LLVMPIPE_UUID_PREFIX = "6d657361"

# The arms that run. `mixed` is handled separately -- see the module docstring.
ARMS = ("cpu", "npu", "gpu")
MIXED_ARM = "mixed"
ALL_ARMS = ARMS + (MIXED_ARM,)

# Arms whose non-Rocket half is compiled for the GPU.
VULKAN_ARMS = ("gpu", MIXED_ARM)
# Arms that put every candidate on the non-Rocket device.
NO_OFFLOAD_ARMS = ("cpu", "gpu")


@dataclass(frozen=True)
class Case:
    """One fixture: a Rocket-claimable conv feeding a softmax it declines."""

    name: str
    # Conv geometry. There is no padding: the input is already the padded
    # size, which keeps the fixture a single conv with nothing fused in.
    input_hw: int
    channels: int
    kernel: int
    note: str

    @property
    def output_hw(self) -> int:
        return self.input_hw - self.kernel + 1

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (1, self.input_hw, self.input_hw, self.channels)

    @property
    def filter_shape(self) -> tuple[int, ...]:
        return (self.kernel, self.kernel, self.channels, self.channels)

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (1, self.output_hw, self.output_hw, self.channels)

    def softmax_only_mlir(self) -> str:
        """The same softmax with no convolution in front of it.

        Subtracting this from the whole-model time attributes the cost to a
        phase instead of leaving a wall-clock ratio. The input is the conv's
        f32 output shape, so the softmax dispatch is byte-for-byte the one
        the full module runs.
        """
        out = "x".join(str(d) for d in self.output_shape)
        return f"""\
// {self.name}_softmax: the whole model's softmax phase, on its own
func.func @main(%x: tensor<{out}xf32>) -> tensor<{out}xf32> {{
  %empty = tensor.empty() : tensor<{out}xf32>
  %softmax = linalg.softmax dimension(3)
      ins(%x : tensor<{out}xf32>)
      outs(%empty : tensor<{out}xf32>) -> tensor<{out}xf32>
  return %softmax : tensor<{out}xf32>
}}
"""

    def mlir(self) -> str:
        """The case as MLIR.

        `%init` is a function argument rather than a `linalg.fill` because
        that is the form `tools/e2e_conv_regression.py`'s known-offloadable
        fixtures use; the caller passes zeros, which matters because
        `linalg.conv_2d_nhwc_hwcf` accumulates into its `outs`.

        The softmax's `outs` is a `tensor.empty()` on purpose. Handing it a
        function argument instead makes IREE's vulkan-spirv codegen try to
        promote the whole tensor to workgroup memory and then fail workgroup
        distribution verification, so the shape of the fixture decides
        whether the GPU arms compile at all -- worth not rediscovering.
        """
        inp = "x".join(str(d) for d in self.input_shape)
        flt = "x".join(str(d) for d in self.filter_shape)
        out = "x".join(str(d) for d in self.output_shape)
        return f"""\
// {self.name}: {self.note}
func.func @main(%input: tensor<{inp}xf16>,
                %filter: tensor<{flt}xf16>,
                %init: tensor<{out}xf32>) -> tensor<{out}xf32> {{
  %conv = linalg.conv_2d_nhwc_hwcf
      {{dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}}
      ins(%input, %filter : tensor<{inp}xf16>, tensor<{flt}xf16>)
      outs(%init : tensor<{out}xf32>) -> tensor<{out}xf32>
  %empty = tensor.empty() : tensor<{out}xf32>
  %softmax = linalg.softmax dimension(3)
      ins(%conv : tensor<{out}xf32>)
      outs(%empty : tensor<{out}xf32>) -> tensor<{out}xf32>
  return %softmax : tensor<{out}xf32>
}}
"""


CASES: tuple[Case, ...] = (
    Case(
        name="conv_softmax_32",
        input_hw=32,
        channels=512,
        kernel=3,
        note="the dense fp16 3x3 shape the board conv gate already offloads",
    ),
    Case(
        name="conv_softmax_56",
        input_hw=58,
        channels=256,
        kernel=3,
        note="4x the softmax, so the GPU half is not pure dispatch overhead",
    ),
)


@dataclass
class Conditions:
    """What the board was while a number was taken."""

    kernel: str = ""
    governors: str = ""
    npu_clk: str = ""
    npu_param_clk: str = ""
    gpu_governor: str = ""
    gpu_cur_freq: str = ""
    gpu_target_freq: str = ""
    gpu_min_freq: str = ""
    gpu_max_freq: str = ""
    # devfreq `trans_stat` residency: frequency (Hz, as text) -> ms spent
    # there since boot.
    gpu_residency_ms: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"kernel={self.kernel} cpu_governors=[{self.governors}] "
            f"npu_clk={self.npu_clk} (module param {self.npu_param_clk}) "
            f"gpu_target={self.gpu_target_freq} cur={self.gpu_cur_freq} "
            f"min={self.gpu_min_freq} max={self.gpu_max_freq} "
            f"governor={self.gpu_governor}"
        )

    @property
    def gpu_pinned(self) -> bool:
        """Whether the Mali can only run at its ceiling.

        Neither `cur_freq` nor `target_freq` is the test: devfreq reports the
        200 MHz OPP for both whenever the GPU is powered down, so an idle
        board reads 200 MHz however the knobs are set, and both only read
        1000 MHz while a job is actually in flight. What pins the clock is
        the floor meeting the ceiling -- `simple_ondemand` then has nothing
        left to choose -- and what confirms it *stayed* pinned across a run
        is the `trans_stat` residency, which `residency_report` reads.
        """
        return bool(self.gpu_max_freq) and self.gpu_min_freq == self.gpu_max_freq

    @property
    def cpu_pinned(self) -> bool:
        return bool(self.governors) and all(
            part.endswith(":performance") for part in self.governors.split()
        )


@dataclass
class ArmResult:
    arm: str
    vmfb: Path
    rocket_sites: int = 0
    other_sites: int = 0
    other_backend: str = ""
    ran: bool = True
    failure: str = ""
    wall_ms: list[float] = field(default_factory=list)
    loop_ms: float | None = None

    @property
    def wall_median(self) -> float | None:
        return statistics.median(self.wall_ms) if self.wall_ms else None


def command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print(f"$ {command_text(command)}", flush=True)
    subprocess.run(command, check=True, env=env)


def capture(command: list[str], *, env: dict[str, str] | None = None) -> str:
    print(f"$ {command_text(command)}", flush=True)
    return subprocess.run(
        command, check=True, capture_output=True, text=True, env=env
    ).stdout


def capture_both(command: list[str], *, env: dict[str, str] | None = None) -> str:
    """stdout+stderr, for tools that diagnose on stderr (rocket-compiler does)."""
    print(f"$ {command_text(command)}", flush=True)
    done = subprocess.run(command, check=True, capture_output=True, text=True, env=env)
    return done.stdout + done.stderr


def ssh_capture(host: str, script: str) -> str:
    return capture(["ssh", host, script])


def ssh_run(host: str, script: str) -> None:
    run(["ssh", host, script])


def read_conditions(host: str) -> Conditions:
    """Everything that moves a number on this board, in one round trip."""
    script = (
        "uname -r; "
        'for c in 0 4 6; do printf "cpu%s:%s " $c '
        '"$(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor)"; done; echo; '
        "cat /sys/kernel/debug/clk/clk_npu/clk_rate 2>/dev/null || echo unreadable; "
        "cat /sys/module/rocket/parameters/rocket_npu_clk_hz 2>/dev/null || echo absent; "
        "cat /sys/class/devfreq/fb000000.gpu/governor; "
        "cat /sys/class/devfreq/fb000000.gpu/cur_freq; "
        "cat /sys/class/devfreq/fb000000.gpu/target_freq; "
        "cat /sys/class/devfreq/fb000000.gpu/min_freq; "
        "cat /sys/class/devfreq/fb000000.gpu/max_freq; "
        "echo ---; "
        "cat /sys/class/devfreq/fb000000.gpu/trans_stat"
    )
    text = ssh_capture(host, script)
    head, _, stats = text.partition("\n---\n")
    lines = [line.strip() for line in head.splitlines()]
    if len(lines) < 9:
        raise SystemExit(f"could not read board conditions; got {lines!r}")
    return Conditions(
        kernel=lines[0],
        governors=lines[1],
        npu_clk=lines[2],
        npu_param_clk=lines[3],
        gpu_governor=lines[4],
        gpu_cur_freq=lines[5],
        gpu_target_freq=lines[6],
        gpu_min_freq=lines[7],
        gpu_max_freq=lines[8],
        gpu_residency_ms=parse_trans_stat(stats),
    )


def parse_trans_stat(text: str) -> dict[str, int]:
    """devfreq `trans_stat` -> {frequency in Hz: ms spent there since boot}.

    Each data row is `[*]<freq>: <n> <n> ... <time_ms>`; the `*` marks the
    frequency devfreq currently reports. The header row has no colon-prefixed
    frequency and is skipped by the int() guard.
    """
    residency: dict[str, int] = {}
    for line in text.splitlines():
        head, sep, rest = line.partition(":")
        if not sep:
            continue
        freq = head.strip().lstrip("*")
        columns = rest.split()
        if not freq.isdigit() or not columns:
            continue
        try:
            residency[freq] = int(columns[-1])
        except ValueError:
            continue
    return residency


def residency_report(before: Conditions, after: Conditions) -> str:
    """How the GPU's time in the measured window was spread over its OPPs.

    This is what says the clock *stayed* pinned, rather than that it was
    pinned when the run started. 200 MHz is devfreq's reading while the GPU
    is powered down, so time there is idle time between arms, not a throttle;
    any time at an OPP in between would be.
    """
    deltas = {
        freq: after.gpu_residency_ms.get(freq, 0) - ms
        for freq, ms in before.gpu_residency_ms.items()
    }
    moved = {freq: ms for freq, ms in deltas.items() if ms > 0}
    if not moved:
        return "GPU devfreq residency: no transitions recorded in the window"
    floor, ceiling = after.gpu_min_freq, after.gpu_max_freq
    intermediate = {
        freq: ms for freq, ms in moved.items() if freq not in (floor, ceiling, "200000000")
    }
    parts = ", ".join(
        f"{int(freq) // 1_000_000} MHz {ms} ms" for freq, ms in sorted(moved.items())
    )
    verdict = (
        "nothing between the suspend floor and the ceiling"
        if not intermediate
        else "THROTTLED: "
        + ", ".join(
            f"{int(freq) // 1_000_000} MHz {ms} ms"
            for freq, ms in sorted(intermediate.items())
        )
    )
    return f"GPU devfreq residency over the window: {parts} -- {verdict}"


def pin_clocks(host: str) -> bool:
    """Pins both clocks for the run; returns False if it could not.

    The GPU devfreq offers only `simple_ondemand` -- no `performance`, no
    `userspace` -- so the only lever is the frequency floor: raising
    `min_freq` to `max_freq` leaves the governor nothing to scale down to.

    Both writes need root. Failing is reported rather than raised, so a
    correctness-only run still completes on a board without passwordless
    sudo; `main` refuses to *time* an unpinned board, because an unpinned
    Mali sits at a fifth of its ceiling and the number would mean nothing.
    """
    script = (
        "set -e; "
        "for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do "
        'echo performance | sudo -n tee "$g" >/dev/null; done; '
        "max=$(cat /sys/class/devfreq/fb000000.gpu/max_freq); "
        'echo "$max" | sudo -n tee /sys/class/devfreq/fb000000.gpu/min_freq >/dev/null; '
        "cat /sys/class/devfreq/fb000000.gpu/cur_freq"
    )
    print(f"$ ssh {host} {script!r}", flush=True)
    done = subprocess.run(["ssh", host, script], capture_output=True, text=True)
    if done.returncode != 0:
        print(
            f"warning: could not pin the clocks on {host}: "
            f"{done.stderr.strip() or done.stdout.strip()}",
            file=sys.stderr,
        )
        return False
    print(done.stdout.strip())
    return True


def restore_clocks(host: str, before: Conditions) -> None:
    """Puts the governor and the GPU floor back where they were."""
    governor = "ondemand"
    for part in before.governors.split():
        if ":" in part:
            governor = part.split(":", 1)[1]
            break
    script = (
        "for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do "
        f'echo {shlex.quote(governor)} | sudo -n tee "$g" >/dev/null; done; '
        f"echo {shlex.quote(before.gpu_min_freq or '200000000')} | "
        "sudo -n tee /sys/class/devfreq/fb000000.gpu/min_freq >/dev/null"
    )
    subprocess.run(["ssh", host, script], capture_output=True, text=True)


def wait_for_quiet_npu(host: str, budget_seconds: float = 30.0) -> None:
    """Blocks until every NPU core reads `suspended`.

    `active` on an idle board means a job hung recently and the device is
    still recovering; the state crosses processes, so a timing started now
    measures the recovery (agent-docs/board-setup.md).
    """
    deadline = time.monotonic() + budget_seconds
    script = "cat /sys/devices/platform/*.npu/power/runtime_status | tr '\\n' ' '"
    while True:
        status = subprocess.run(
            ["ssh", host, script], capture_output=True, text=True
        ).stdout.split()
        if status and all(state == "suspended" for state in status):
            return
        if time.monotonic() >= deadline:
            print(
                f"warning: NPU did not quiesce within {budget_seconds:.0f}s "
                f"(runtime_status: {' '.join(status)}); continuing anyway",
                file=sys.stderr,
            )
            return
        time.sleep(1.0)


def discover_gpu_device(host: str, runtime_name: str, remote_dir: str) -> str:
    """The Mali's `vulkan://GPU-...` URI, never an index.

    Both the Mali and llvmpipe are enumerated. llvmpipe's UUID is ASCII
    ("mesa..."), so it is identified by content rather than by position.
    """
    script = " && ".join(
        [f"cd {shlex.quote(remote_dir)}", f"./{runtime_name} --list_devices"]
    )
    uris = [
        line.strip()
        for line in ssh_capture(host, script).splitlines()
        if line.strip().startswith("vulkan://")
    ]
    mali = [uri for uri in uris if LLVMPIPE_UUID_PREFIX not in uri.lower()]
    if not mali:
        raise SystemExit(
            "found no non-llvmpipe Vulkan device on the board; "
            f"--list_devices gave {uris!r}. Pass --gpu-device-uri explicitly."
        )
    if len(mali) > 1:
        raise SystemExit(
            f"more than one candidate Vulkan device {mali!r}; "
            "pass --gpu-device-uri to say which."
        )
    return mali[0]


def arm_compile_flags(args: argparse.Namespace, arm: str) -> list[str]:
    flags: list[str] = []
    if arm in VULKAN_ARMS:
        flags += [
            "--cpu-device-name=gpu_device",
            "--cpu-device-backend=vulkan",
            f"--iree-flag=--iree-vulkan-target={args.vulkan_target}",
        ]
    else:
        # Both CPU-hosted arms need an aarch64 llvm-cpu half, and a fresh
        # compiler tree has no iree-lld of its own (spike doc section 6.4).
        flags.append(f"--llvmcpu-target-triple={args.llvmcpu_target_triple}")
        if args.linker:
            flags.append(
                f"--iree-flag=--iree-llvmcpu-embedded-linker-path={args.linker}"
            )
    if arm in NO_OFFLOAD_ARMS:
        flags.append("--no-offload")
    return flags


def compile_arm(
    args: argparse.Namespace,
    case: Case,
    arm: str,
    work_dir: Path,
    mlir: Path,
    suffix: str = "",
) -> ArmResult:
    """Compiles one arm and reads its placement off the audit, not the exit code."""
    vmfb = work_dir / f"{case.name}_{arm}{suffix}.vmfb"
    extra = arm_compile_flags(args, arm)
    env = dict(os.environ)
    env["IREE_COMPILER_LIB"] = str(args.compiler_lib)
    run(
        [
            str(args.rocket_compiler),
            "compile",
            f"--input={mlir}",
            f"--output={vmfb}",
            *extra,
        ],
        env=env,
    )
    report = capture_both(
        [str(args.rocket_compiler), "audit", f"--input={mlir}", *extra], env=env
    )

    result = ArmResult(arm=arm, vmfb=vmfb)
    for line in report.splitlines():
        stripped = line.strip()
        if "dispatch site(s) -> " not in stripped:
            continue
        count = int(stripped.split("executable(s),")[1].split("dispatch site(s)")[0])
        label = stripped.rsplit("-> ", 1)[1]
        if label == "rocket":
            result.rocket_sites = count
        else:
            result.other_sites = count
            result.other_backend = label
    return result


# arm -> (expected rocket dispatch sites, or None for "at least one";
#         expected non-Rocket backend label).
EXPECTED_PLACEMENT = {
    "cpu": (0, "cpu"),
    "npu": (None, "cpu"),
    "gpu": (0, "vulkan-spirv"),
    MIXED_ARM: (None, "vulkan-spirv"),
}


def check_placement(case: Case, results: dict[str, ArmResult]) -> None:
    """Refuses to go on unless each arm is the arm it claims to be.

    Exit 0 from a multi-device compile line is not evidence of anything: a
    two-device flag line with no affinities exits 0 and silently drops the
    second device (spike doc section 5). What the audit read off the
    `executable-targets` IR is the evidence.
    """
    failures: list[str] = []
    for arm, (rocket_sites, backend) in EXPECTED_PLACEMENT.items():
        result = results[arm]
        if rocket_sites is None:
            if result.rocket_sites < 1:
                failures.append(f"{arm} arm has no rocket dispatch site")
        elif result.rocket_sites != rocket_sites:
            failures.append(
                f"{arm} arm has {result.rocket_sites} rocket site(s), "
                f"expected {rocket_sites}"
            )
        if result.other_backend != backend:
            failures.append(
                f"{arm} arm's non-Rocket half is {result.other_backend!r}, "
                f"expected {backend!r}"
            )
        if result.other_sites < 1:
            failures.append(f"{arm} arm has no {backend} dispatch site")
    if results["npu"].rocket_sites != results[MIXED_ARM].rocket_sites:
        failures.append(
            f"npu arm has {results['npu'].rocket_sites} rocket site(s) and mixed "
            f"{results[MIXED_ARM].rocket_sites}: the two are not the same split"
        )
    if failures:
        raise SystemExit(f"{case.name}: " + "; ".join(failures))
    for arm in ALL_ARMS:
        result = results[arm]
        print(
            f"{case.name}: {arm:5} placement -> {result.rocket_sites} rocket + "
            f"{result.other_sites} {result.other_backend} dispatch site(s)"
        )


def write_inputs(case: Case, work_dir: Path, seed: int) -> list[tuple[str, str]]:
    """Writes the three inputs as raw `.bin`; returns (spec, filename) pairs.

    Raw bytes with an explicit type, never `.npy`: `iree-run-module` reads
    `--input=@file` as raw bytes even for a numpy container, so an npy header
    would shift the whole tensor by 64 f16 elements and the run would still
    exit 0. The output direction is asymmetric -- `--output=@f.npy` does
    write real npy -- which is what makes an all-npy harness look
    self-consistent and be wrong.
    """
    rng = np.random.default_rng(seed)
    activations = rng.standard_normal(case.input_shape).astype(np.float16)
    # Scaled so a k*k*C dot product does not saturate f16 on the way in: the
    # point of the fixture is placement, not a precision cliff.
    filters = (rng.standard_normal(case.filter_shape) * 0.02).astype(np.float16)
    init = np.zeros(case.output_shape, dtype=np.float32)

    entries = []
    for name, array in (
        ("input.bin", activations),
        ("filter.bin", filters),
        ("init.bin", init),
    ):
        path = work_dir / f"{case.name}_{name}"
        array.tofile(path)
        shape = "x".join(str(d) for d in array.shape)
        dtype = "f16" if array.dtype == np.float16 else "f32"
        entries.append((f"{shape}x{dtype}=@{path.name}", path.name))
    return entries


def arm_devices(arm: str, gpu_uri: str) -> list[str]:
    """Which HAL devices an arm's module needs.

    The `gpu` arm's module declares only `@gpu_device` -- the unused Rocket
    global is dropped -- so it takes the Mali alone. The `mixed` arm needs
    both, and the order decides *which* driver is handed the other's buffers
    and therefore how it fails; the GPU goes first because that is the order
    `#hal.device.optimal<[@gpu_device, @rocket_device]>` already puts the
    allocator-selection pairs in.
    """
    if arm == "gpu":
        return [f"--device={gpu_uri}"]
    if arm == MIXED_ARM:
        return [f"--device={gpu_uri}", "--device=rocket"]
    return ["--device=rocket", "--device=local-task"]


def arm_runtime_flags(arm: str, cpu_ids: str) -> list[str]:
    """`--task_topology_cpu_ids` on every arm that has a local-task device.

    IREE builds its task topology from cpuinfo, not from the process's
    affinity mask, and its default puts workers on whichever cluster it likes
    -- MobileNetV2 fp16 is 105 ms with the default and 60 ms pinned. The GPU
    and mixed arms have no local-task device, so the flag is inert there.
    """
    if arm in ("gpu", MIXED_ARM):
        return []
    return [f"--task_topology_cpu_ids={cpu_ids}"]


def run_command(
    result: ArmResult,
    inputs: list[tuple[str, str]],
    gpu_uri: str,
    cpu_ids: str,
    runtime_name: str,
    output_name: str | None,
) -> list[str]:
    return [
        f"./{runtime_name}",
        f"--module={result.vmfb.name}",
        "--function=main",
        *arm_devices(result.arm, gpu_uri),
        *arm_runtime_flags(result.arm, cpu_ids),
        *(f"--input={spec}" for spec, _ in inputs),
        *([f"--output=@{output_name}"] if output_name else []),
    ]


def run_arm(
    host: str,
    remote_dir: str,
    runtime_name: str,
    result: ArmResult,
    inputs: list[tuple[str, str]],
    output_name: str,
    gpu_uri: str,
    cpu_ids: str,
) -> bool:
    """Runs one arm; returns whether it produced an output.

    A failure is captured rather than raised, because the `mixed` arm is
    expected to fail and the point of this gate is to say exactly how.
    """
    command = run_command(result, inputs, gpu_uri, cpu_ids, runtime_name, output_name)
    # The remote exit status is echoed rather than taken from ssh, which
    # collapses everything to 255 and drops the signal: a segfault is
    # reported by the remote shell, not by the process, and does not reach
    # our stderr at all. 128+N is how a shell reports signal N.
    script = " && ".join(
        [f"cd {shlex.quote(remote_dir)}", command_text(command) + ' ; echo "__rc=$?"']
    )
    print(f"$ ssh {host} {script!r}", flush=True)
    done = subprocess.run(["ssh", host, script], capture_output=True, text=True)
    lines = (done.stdout + done.stderr).splitlines()
    status = next(
        (int(line.split("=", 1)[1]) for line in lines if line.startswith("__rc=")),
        done.returncode,
    )
    for line in lines:
        if not line.startswith("__rc="):
            print(line)
    if status == 0:
        return True
    result.ran = False
    diagnostic = next(
        (
            line.strip()
            for line in lines
            if line.strip()
            and not line.startswith(("EXEC ", "__rc="))
            and "iree-run-module" not in line
        ),
        "",
    )
    if not diagnostic and status > 128:
        diagnostic = f"killed by signal {status - 128}" + (
            " (SIGSEGV)" if status == 139 else ""
        )
    result.failure = f"exit {status}: {diagnostic or 'no diagnostic'}"
    print(f"{result.arm} arm did not run -- {result.failure}", file=sys.stderr)
    return False


def time_arm(
    host: str,
    remote_dir: str,
    runtime_name: str,
    benchmark_name: str,
    result: ArmResult,
    inputs: list[tuple[str, str]],
    gpu_uri: str,
    cpu_ids: str,
    repeats: int,
) -> None:
    """A process-per-run wall time and, separately, a loop figure."""
    base = run_command(result, inputs, gpu_uri, cpu_ids, runtime_name, None)[1:]
    # Timed on the board with `date`, not with `/usr/bin/time` (absent on
    # Armbian's minimal image) and not from the host, which would fold the
    # ssh round trip into every sample.
    timed = (
        "s=$(date +%s%N); "
        + command_text([f"./{runtime_name}", *base])
        + " >/dev/null 2>&1; rc=$?; e=$(date +%s%N); "
        'echo "__ms=$(( (e - s) / 1000000 )) __rc=$rc"'
    )
    for _ in range(repeats):
        wait_for_quiet_npu(host)
        script = " && ".join([f"cd {shlex.quote(remote_dir)}", timed])
        done = subprocess.run(
            ["ssh", host, script], capture_output=True, text=True, check=True
        )
        fields = dict(
            part.split("=", 1)
            for part in done.stdout.split()
            if part.startswith(("__ms=", "__rc="))
        )
        if fields.get("__rc") != "0":
            raise SystemExit(
                f"the {result.arm} arm failed while being timed "
                f"(exit {fields.get('__rc')}): {done.stdout}{done.stderr}"
            )
        result.wall_ms.append(float(fields["__ms"]))

    wait_for_quiet_npu(host)
    script = " && ".join(
        [
            f"cd {shlex.quote(remote_dir)}",
            command_text(
                [
                    f"./{benchmark_name}",
                    *base,
                    "--benchmark_repetitions=5",
                    "--benchmark_min_warmup_time=0.5",
                ]
            ),
        ]
    )
    done = subprocess.run(["ssh", host, script], capture_output=True, text=True)
    if done.returncode != 0:
        print(
            f"warning: iree-benchmark-module failed on the {result.arm} arm:\n"
            f"{done.stdout}\n{done.stderr}",
            file=sys.stderr,
        )
        return
    for line in done.stdout.splitlines():
        if "_median" in line and " ms " in line:
            parts = line.split()
            result.loop_ms = float(parts[parts.index("ms") - 1])
            break


def compare(
    label: str, reference: Path, candidate: Path, atol: float, rtol: float
) -> bool:
    ref = np.load(reference).astype(np.float64)
    got = np.load(candidate).astype(np.float64)
    if ref.shape != got.shape:
        raise SystemExit(f"{label}: shape mismatch {ref.shape} vs {got.shape}")
    error = np.abs(ref - got)
    allowed = atol + rtol * np.abs(ref)
    mismatches = int(np.count_nonzero(error > allowed))
    print(
        f"{label}: max|err|={error.max(initial=0.0):.6g} "
        f"mismatches={mismatches}/{ref.size} (atol={atol}, rtol={rtol})"
    )
    return mismatches == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--board", required=True, help="ssh host, e.g. orangepi")
    parser.add_argument(
        "--remote-dir",
        default="mixed-device",
        help="staging directory on the board, relative to $HOME unless absolute "
        "(a literal ~ is not expanded: every remote path here is quoted)",
    )
    parser.add_argument(
        "--compiler-lib",
        type=Path,
        default=os.environ.get("IREE_COMPILER_LIB"),
        help="libIREECompiler.so carrying BOTH vulkan-spirv and rocket",
    )
    parser.add_argument(
        "--rocket-compiler",
        type=Path,
        default=REPO / "target/release/rocket-compiler",
    )
    parser.add_argument(
        "--board-runtime",
        type=Path,
        default=None,
        help="aarch64 iree-run-module built with ROCKET_WITH_VULKAN=ON; when "
        "omitted, whatever is already in --remote-dir is used",
    )
    parser.add_argument(
        "--board-benchmark",
        type=Path,
        default=None,
        help="aarch64 iree-benchmark-module from the same build",
    )
    parser.add_argument(
        "--linker",
        default=str(REPO / "iree-build/build/tools/iree-lld"),
        help="iree-lld for the llvm-cpu arms; a fresh compiler tree has none",
    )
    parser.add_argument("--llvmcpu-target-triple", default="aarch64-linux-gnu")
    parser.add_argument(
        "--vulkan-target", default="valhall3", help="G610 is in the G710 family"
    )
    parser.add_argument("--gpu-device-uri", default=None, help="skip Mali discovery")
    parser.add_argument("--cpu-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--case", action="append", choices=[c.name for c in CASES])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261127)
    parser.add_argument("--skip-timing", action="store_true")
    parser.add_argument(
        "--no-pin",
        action="store_true",
        help="do not touch the clocks: for a board pinned by hand, whose state "
        "is still read and printed either side of every timing",
    )
    parser.add_argument("--work-dir", type=Path, default=None)
    # Both the NPU arm's convolution and the baseline's take fp16 inputs, so
    # they differ only in accumulation order and in the NPU's f16
    # accumulate. Loose enough for that, and far too tight to pass on a
    # buffer that was handed over wrong.
    parser.add_argument("--atol", type=float, default=2e-3)
    parser.add_argument("--rtol", type=float, default=2e-2)
    args = parser.parse_args()
    if not args.compiler_lib:
        raise SystemExit(
            "--compiler-lib (or IREE_COMPILER_LIB) must point at a "
            "libIREECompiler.so with both vulkan-spirv and rocket registered"
        )
    args.compiler_lib = Path(args.compiler_lib)
    if args.linker and not Path(args.linker).exists():
        print(f"note: {args.linker} is absent; leaving the linker to IREE", flush=True)
        args.linker = ""
    return args


def stage_runtime(args: argparse.Namespace, remote: str) -> tuple[str, str]:
    runtime_name = "iree-run-module"
    benchmark_name = "iree-benchmark-module"
    staged = []
    if args.board_runtime:
        staged.append(args.board_runtime)
        runtime_name = args.board_runtime.name
    if args.board_benchmark:
        staged.append(args.board_benchmark)
        benchmark_name = args.board_benchmark.name
    if staged:
        run(["scp", *(str(p) for p in staged), f"{args.board}:{remote}/"])
    ssh_run(
        args.board,
        f"cd {shlex.quote(remote)} && chmod +x {shlex.quote(runtime_name)} "
        f"{shlex.quote(benchmark_name)}",
    )
    return runtime_name, benchmark_name


def measure_softmax_phase(
    args: argparse.Namespace,
    case: Case,
    work_dir: Path,
    remote: str,
    runtime_name: str,
    benchmark_name: str,
    gpu_uri: str,
) -> dict[str, ArmResult]:
    """Times the softmax on its own, so a phase can be named rather than a ratio.

    Only the `cpu` and `gpu` arms are built: the NPU has no softmax at all,
    which is the whole reason this fixture pairs one with a convolution, so
    the NPU arm's softmax phase *is* the CPU one.
    """
    mlir = work_dir / f"{case.name}_softmax.mlir"
    mlir.write_text(case.softmax_only_mlir())
    shape = "x".join(str(d) for d in case.output_shape)
    rng = np.random.default_rng(args.seed + 1)
    name = f"{case.name}_softmax_in.bin"
    rng.standard_normal(case.output_shape).astype(np.float32).tofile(work_dir / name)
    inputs = [(f"{shape}xf32=@{name}", name)]

    phases: dict[str, ArmResult] = {}
    for arm in ("cpu", "gpu"):
        result = compile_arm(args, case, arm, work_dir, mlir, suffix="_softmax")
        if result.rocket_sites:
            raise SystemExit(
                f"{case.name}: the softmax-only {arm} module reached Rocket "
                f"({result.rocket_sites} site(s)); it is meant to have none"
            )
        phases[arm] = result
    run(
        [
            "scp",
            str(work_dir / name),
            *(str(phases[arm].vmfb) for arm in ("cpu", "gpu")),
            f"{args.board}:{remote}/",
        ]
    )
    for arm in ("cpu", "gpu"):
        time_arm(
            args.board,
            remote,
            runtime_name,
            benchmark_name,
            phases[arm],
            inputs,
            gpu_uri,
            args.cpu_ids,
            args.repeats,
        )
    return phases


def print_phases(
    case: Case, results: dict[str, ArmResult], phases: dict[str, ArmResult]
) -> None:
    """Attributes each arm's loop time to the conv and to the softmax.

    The subtraction is on the loop figures, not the wall ones, because the
    wall time of a separate module pays process and device setup a second
    time -- that cost belongs to the run, not to the phase.
    """
    print(f"\n{case.name} marginal cost per phase (ms, from the loop figures):")
    print(f"{'arm':6} {'total':>8} {'softmax':>9} {'conv (rest)':>12}  softmax runs on")
    softmax_for = {"cpu": "cpu", "npu": "cpu", "gpu": "gpu"}
    for arm in ARMS:
        total = results[arm].loop_ms
        softmax = phases[softmax_for[arm]].loop_ms
        if total is None or softmax is None:
            print(f"{arm:6} {'-':>8} {'-':>9} {'-':>12}")
            continue
        print(
            f"{arm:6} {total:>8.1f} {softmax:>9.1f} {total - softmax:>12.1f}"
            f"  {softmax_for[arm]}"
        )
    print(
        "The NPU runs no softmax -- nothing in the Rocket spec claims one -- so "
        "the npu arm's softmax phase is the CPU's, and the Mali's own softmax "
        "number is what decides whether offloading the op the NPU refuses is "
        "worth anything."
    )


def print_timings(
    case: Case,
    results: dict[str, ArmResult],
    conditions: Conditions,
    residency: str,
) -> None:
    print(f"\n{case.name} timings (ms) -- {conditions.summary()}")
    print(residency)
    print(f"{'arm':6} {'wall median':>12} {'wall range':>20} {'loop median':>12}")
    for arm in ARMS:
        result = results[arm]
        spread = (
            f"{min(result.wall_ms):.0f}-{max(result.wall_ms):.0f} "
            f"(n={len(result.wall_ms)})"
            if result.wall_ms
            else "-"
        )
        median = f"{result.wall_median:.0f}" if result.wall_median else "-"
        loop = f"{result.loop_ms:.3f}" if result.loop_ms is not None else "-"
        print(f"{arm:6} {median:>12} {spread:>20} {loop:>12}")
    print(
        "wall = one process per run, which is what a caller pays. "
        "loop = iree-benchmark-module, a resident-working-set ceiling that "
        "overstates by roughly 1.4-1.9x on this board."
    )


def main() -> None:
    args = parse_args()
    cases = [c for c in CASES if not args.case or c.name in args.case]
    work_dir = args.work_dir or Path(
        os.environ.get("TMPDIR", "/tmp")
    ) / f"mixed-device-{os.getpid()}"
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"work dir: {work_dir}")

    remote = args.remote_dir
    ssh_run(args.board, f"mkdir -p {shlex.quote(remote)}")
    runtime_name, benchmark_name = stage_runtime(args, remote)

    gpu_uri = args.gpu_device_uri or discover_gpu_device(
        args.board, runtime_name, remote
    )
    print(f"Mali Vulkan device: {gpu_uri}")

    before = read_conditions(args.board)
    print(f"board before: {before.summary()}")
    pinned_by_us = False
    if not args.no_pin:
        pinned_by_us = pin_clocks(args.board)
    conditions = read_conditions(args.board) if pinned_by_us else before
    if not args.skip_timing and not (conditions.gpu_pinned and conditions.cpu_pinned):
        raise SystemExit(
            "refusing to time an unpinned board. The Mali devfreq floor reads "
            f"min={conditions.gpu_min_freq} against max={conditions.gpu_max_freq} "
            f"and the CPU governors read [{conditions.governors}]. Pin them with\n"
            "  ssh BOARD 'sudo sh -c \"for g in /sys/devices/system/cpu/cpu*/"
            'cpufreq/scaling_governor; do echo performance > $g; done; cat '
            "/sys/class/devfreq/fb000000.gpu/max_freq > "
            "/sys/class/devfreq/fb000000.gpu/min_freq\"'\n"
            "and re-run with --no-pin, or run --skip-timing for the correctness "
            "half alone, for which the clocks do not matter."
        )

    failed: list[str] = []
    try:
        for case in cases:
            print(f"\n=== {case.name}: {case.note} ===")
            mlir = work_dir / f"{case.name}.mlir"
            mlir.write_text(case.mlir())

            results = {
                arm: compile_arm(args, case, arm, work_dir, mlir) for arm in ALL_ARMS
            }
            check_placement(case, results)

            inputs = write_inputs(case, work_dir, args.seed)
            run(
                [
                    "scp",
                    *(str(work_dir / name) for _, name in inputs),
                    *(str(results[arm].vmfb) for arm in ALL_ARMS),
                    f"{args.board}:{remote}/",
                ]
            )

            outputs: dict[str, Path] = {}
            for arm in ALL_ARMS:
                output_name = f"{case.name}_{arm}.npy"
                wait_for_quiet_npu(args.board)
                if not run_arm(
                    args.board,
                    remote,
                    runtime_name,
                    results[arm],
                    inputs,
                    output_name,
                    gpu_uri,
                    args.cpu_ids,
                ):
                    if arm != MIXED_ARM:
                        failed.append(f"{case.name}: the {arm} arm did not run")
                    continue
                run(
                    [
                        "scp",
                        f"{args.board}:{remote}/{output_name}",
                        str(work_dir / output_name),
                    ]
                )
                outputs[arm] = work_dir / output_name

            if results[MIXED_ARM].ran:
                print(
                    f"{case.name}: NOTE -- the mixed NPU+Mali arm ran, and was not "
                    "expected to. If a HAL driver has learnt to take the other's "
                    "buffers, this gate's expectations and section 6 of the spike "
                    "doc both need updating."
                )
            for arm in ("npu", "gpu"):
                if arm in outputs and "cpu" in outputs:
                    if not compare(
                        f"{case.name} {arm} vs cpu-only",
                        outputs["cpu"],
                        outputs[arm],
                        args.atol,
                        args.rtol,
                    ):
                        failed.append(f"{case.name}: {arm} differs from the CPU arm")

            if not args.skip_timing:
                before_timing = read_conditions(args.board)
                for arm in ARMS:
                    if results[arm].ran:
                        time_arm(
                            args.board,
                            remote,
                            runtime_name,
                            benchmark_name,
                            results[arm],
                            inputs,
                            gpu_uri,
                            args.cpu_ids,
                            args.repeats,
                        )
                phases = measure_softmax_phase(
                    args,
                    case,
                    work_dir,
                    remote,
                    runtime_name,
                    benchmark_name,
                    gpu_uri,
                )
                after_timing = read_conditions(args.board)
                print_timings(
                    case,
                    results,
                    conditions,
                    residency_report(before_timing, after_timing),
                )
                print_phases(case, results, phases)
    finally:
        after = read_conditions(args.board)
        print(f"\nboard after: {after.summary()}")
        if pinned_by_us:
            restore_clocks(args.board, before)
            print("clocks restored")

    if failed:
        raise SystemExit("FAILED: " + "; ".join(failed))
    print("\nmixed-device gate passed")


if __name__ == "__main__":
    main()
