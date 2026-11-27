# Board setup (RK3588)

What the board has to be before any number from this repository reproduces,
how to check what it currently is, and how to read the three failure
signatures a wrong setup produces. [`onnx-to-numbers.md`](onnx-to-numbers.md)
starts where this page ends.

**Read the kernel section first.** The measured results in this repository were
taken on a `rocket` module carrying three out-of-tree patches. On a stock
module, pooling jobs are killed by the watchdog, every time is roughly double,
and a job timeout can corrupt the submitting process. None of that is subtle
and all of it looks like a bug in this stack.

## The measured baseline

| | What was measured |
|---|---|
| SoC | RK3588. Two boards: `planck` and `orangepi` (the same host re-flashed; `planck` earlier ran 7.1.0-edge). |
| OS | Armbian `7.3.5-edge-rockchip64` for every measurement; the board has run `7.3.6-edge-rockchip64` with the same three patches since 2026-11-27 ([`kernel-patches/`](../kernel-patches) covers both). Debian 13 userland, glibc 2.41, which is above the nightly artifacts' 2.39 floor. |
| Kernel driver | Mainline `rocket` (`drivers/accel/rocket`), as Armbian ships it -- upstream plus Armbian's RK3576 series -- **plus the three patches below**. |
| Device node | `/dev/accel/accel0`. The driver opens it once per worker context. |
| NPU clock | 600 MHz at 0.80 V, set through the patched module's parameters. Mainline boots it pinned at 200 MHz. |
| Governor | `performance` on all three CPU clusters. |

Anything else is untested rather than unsupported. Armbian's module carries
an RK3576 series, so the driver reaches that part too; nothing in this
repository has been measured on it.

## The three kernel patches

All three are in [`kernel-patches/`](../kernel-patches) with instructions for
applying them, and that directory's README is the detailed account --
including that `081` and `088` are Greg Tomko's, from the external
[`gregordinary/patches`](https://github.com/gregordinary/patches) series, and
only rebased here. None of
them is upstream: the PPU change is patch 1 of a local series on top of
`drm-misc-next`, and getting it and a devfreq-shaped clock fix upstream is
the open item -- [`open-questions.md`](open-questions.md) 2.6.

| Patch | What it does | Without it |
|---|---|---|
| **Unconditional PPU completion** (`0001`, written against `drm-misc-next`, rebased onto Armbian) | Arms the PPU *and* DPU interrupts for every job, and accepts the PPU bits in the IRQ handler's raw-status check. The flag-gated `093` variant is not enough, because this repo never sets `DRM_ROCKET_JOB_PPU_DONE`. | Armbian's stock module arms only the DPU interrupts, so **every pooling job is watchdog-killed at ~500 ms**. A single-tile pool still returns the right answer; a tiled one fails (`tiled_pooling_matches_oracle`), because only its first task runs before the kill. |
| **Clock parameters** (`081`) | `clk_set_rate` on runtime-resume and a park back to 200 MHz on suspend, exposed as the writable module parameters `rocket_npu_clk_hz` (loaded as `600000000`) and `rocket_autosuspend_ms` (1000; stock 50 ms fires *inside* an inference once the park exists). The Armbian rebase carries no voltage parameter -- `rocket_npu_uv` is in the base series' `082`, which is not part of this configuration. One hunk is hand-merged: the suspend path counts clocks with `soc->num_clks`. | `scmi_clk_npu` stays pinned at 200 MHz -- mainline `rocket` has no NPU devfreq and the DT sets `assigned-clock-rates = <200000000>`. ResNet50-224 fp16 is ~105 ms rather than 45, int8 150 rather than 125, MobileNetV2 fp16 63 rather than 55.5. Output is bit-identical at either clock; only the time changes (ISSUES.md M2). |
| **Reset before IOMMU detach** (`088`) | Resets the core before detaching the IOMMU, instead of after. | Armbian's `rocket_reset` detaches the IOMMU while the core is still running, so the stall handshake times out on every timeout and one burst can be followed by **page-table corruption in the submitting process** (ISSUES.md C17). With it, a timeout recovers with nothing logged but `NPU job timed out`. |

Two traps when installing a module by hand: `updates/` only wins over
`kernel/drivers/` after `depmod -a`, so a `.ko` dropped there without it is
inert; and the suspend-time park is itself gated on `rocket_npu_clk_hz != 0`,
so writing `0` while the clock is raised **strands it at 600 MHz** and a
"stock" comparison silently runs fast. To really restore stock: write
`200000000`, wait for every core to read `suspended`, then write `0`.

The upstream shape of the clock fix is an open question -- devfreq with an
OPP table rather than module parameters ([`open-questions.md`](open-questions.md)
2.6.1) -- as is whether the park bug can be fixed so `0` restores stock
(2.6.2).

## What does this board actually have?

One round trip, the same probe `tools/model_survey.py` records into every
`result.json`, plus the module parameters:

```sh
ssh planck 'ls -l /dev/accel/accel0
  uname -r
  for c in 0 4 6; do printf "cpu%s:%s " $c \
    "$(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor)"; done; echo
  for i in 82 83 84; do printf "irq%s:%s " $i \
    "$(cat /proc/irq/$i/smp_affinity_list 2>/dev/null)"; done; echo
  cat /sys/devices/platform/*.npu/power/runtime_status | tr "\n" " "; echo
  cat /sys/kernel/debug/clk/clk_npu/clk_rate 2>/dev/null
  grep . /sys/module/rocket/parameters/* 2>/dev/null'
```

What each line has to say:

- **`/dev/accel/accel0` exists.** If not, the module is not loaded (or `depmod`
  was never run after dropping it into `updates/`).
- **`clk_npu/clk_rate` reads 600000000.** 200000000 means the clock patch is
  absent or the parameter is unset, and every timing will be roughly double.
- **`rocket_npu_clk_hz` appears under `/sys/module/rocket/parameters/`.** If the
  file is not there, the module is stock: no clock control, and the pooling
  behaviour below is what you will see. `rocket_autosuspend_ms` should read
  1000 beside it.
- **The governors read `performance`.** `ondemand` makes every host-side phase
  noisy; the governor has silently moved between sessions on these boards
  before, which is why the survey records it next to the numbers.
- **The NPU IRQs are on a big core.** 82/83/84 on RK3588; pinning them to the
  A76 cluster is what the measured runs did.
- **`runtime_status` reads `suspended` on every core.** `active` on an idle
  board means a job hung recently and the device is still sick -- the state
  crosses processes, so a benchmark started now measures the recovery rather
  than the model. `tools/model_survey.py` blocks on exactly this check for up
  to 30 s before each arm.

## Verify the stack before trusting a model

Two of the nightly's artifacts exist for this, and both run on the board with
no host involvement:

```sh
# The crates' own tests, including the hardware ones (all #[ignore]d by default).
for t in rocket-rust-tests-aarch64/bin/*; do
  "./$t" || echo "FAILED: $t"
  "./$t" --ignored || echo "FAILED (hardware): $t"
done

# IREE's HAL CTS against the real rocket driver.
for t in rocket-cts-aarch64/bin/*; do "./$t" || echo "FAILED: $t"; done
```

Then, from the host, the three end-to-end gates. Each cross-builds a probe,
stages it over SSH, runs it on the board and compares against independently
compiled CPU results, so a pass covers the compiler, the driver and the
hardware together:

```sh
python3 tools/e2e_conv_regression.py    --board planck
python3 tools/e2e_matmul_regression.py  --board planck
python3 tools/e2e_pooling_regression.py --board planck
```

The pooling gate is the one that fails loudest on a stock module, since that
is where the PPU interrupt matters. Eight of the matmul gate's nine cases and
every max-pool case are compared **exactly**, not within a tolerance ---
details in [`developer-tools.md`](developer-tools.md#board-convolution-regression-gate).

## Failure signatures

| What you see | What it is |
|---|---|
| Any pooling job dies at ~500 ms; `iree-run-module` hangs then reports a timeout | The PPU patch is missing. Single-tile pools still return correct results, which makes this easy to misread as a tiling bug. |
| Everything runs, every time is ~2x the published one | The NPU is at 200 MHz. Check `clk_npu/clk_rate`, not just the module parameter. |
| One process times out once, then every later job in **that** process times out; kernel log shows IOMMU faults | C17 with the stock reset path. `088` is the kernel half; the userspace half (sizing the dense feature read) is already in `rocket-core`. |
| A benchmark is wildly slower than a single run of the same module | The device may still be sick from an earlier hang, or the governor is not `performance`. Check `runtime_status` first. |
| A module you just installed has no effect | `depmod -a` was not run, so `kernel/drivers/` still wins over `updates/`. |
| `iree-run-module` runs but nothing lands on the NPU | Not a board problem: ask the compiler. `rocket-compiler audit` names the reason per candidate, and `--no-offload` is what a genuinely empty arm looks like. |

## Conditions belong with every number

The governor, the NPU clock, the IRQ affinity, the kernel release and the
module's parameters all move results, and all of them have moved silently
between sessions on these boards. `tools/model_survey.py` reads them off the
board and writes them into `result.json` beside the timings; by hand, record
them yourself. A measurement whose conditions are missing cannot be compared
with the next one -- which is why every dated survey in this directory
carries its board, clock, governor and module variant in the first paragraph.
