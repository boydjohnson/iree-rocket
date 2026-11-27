# Kernel patches — Armbian `7.3.5-edge-rockchip64`

Every measured result in this repository was taken on an RK3588 running
mainline `accel/rocket` **as Armbian 7.3.5 ships it, plus the three changes
below**. All three are here as patch files. None of them is upstream.

On a stock `7.3.5` module the stack does not merely run slower: every pooling
job is killed by the watchdog, so most of the surveyed models fail outright.
If you are bringing up a board, read
[`agent-docs/board-setup.md`](../agent-docs/board-setup.md) alongside this.

> ### Two of these three come from someone else's project
>
> `081` and `088` were written by **Greg Tomko** and are maintained in
> **[`gregordinary/patches`](https://github.com/gregordinary/patches)**, an
> external, GPL-2.0 collection of out-of-tree patch sets and kernel-version
> build profiles for the boot2deb Debian-device builder. They are *not* work
> of this project, and this repository is not their home: file problems with
> them, and look for newer versions, there.
>
> What is vendored here are **rebases onto Armbian 7.3.5**, which that
> project does not carry -- it has the originals, against other kernels.
> Only `0001`, the PPU change, originates with this project.
>
> Each rebased file carries Greg's original `From:`, `Date:`, `Subject:`,
> commit message and `Signed-off-by:`, with the rebase recorded as a
> bracketed note above the diff -- so `git am` credits the right author, and
> the measurements and reasoning in those messages (the refcounted clock
> park, the autosuspend sweep, the stall-timeout A/B on RK3588 silicon) come
> with them. `git format-patch` had rewritten both to name the rebaser and
> dropped the bodies; that is repaired here.

| Change | Base it was written against | Without it |
|---|---|---|
| [`0001-accel-rocket-Add-PPU-status-checks-to-irq-handler.patch`](0001-accel-rocket-Add-PPU-status-checks-to-irq-handler.patch) | `drm-misc-next` | Armbian's module arms only the DPU interrupts, so **every pooling job is watchdog-killed at ~500 ms**. A single-tile pool still returns the right answer; a tiled one does not, because only its first task runs before the kill. |
| [`081-rocket-drv-npu-clk-armbian-7.3.5.patch`](081-rocket-drv-npu-clk-armbian-7.3.5.patch) | Armbian 7.3.5 | The NPU stays pinned at 200 MHz, the rate the DT assigns and mainline has no devfreq to change. ResNet50-224 fp16 runs ~105 ms rather than 45, int8 150 rather than 125. Output is bit-identical at either clock; only the time changes. |
| [`088-rocket-drv-reset-before-iommu-detach-armbian-7.3.5.patch`](088-rocket-drv-reset-before-iommu-detach-armbian-7.3.5.patch) | Armbian 7.3.5 | A job timeout detaches the IOMMU group while the core is still mid-DMA, the `rk_iommu` stall handshake times out, and in-flight DMA can land in pages the kernel has already reused -- observed as page-table corruption in the submitting process after a burst of timeouts. |

## The patches

**`0001`, PPU completion** (this project)**.** Two hunks, five lines: `rocket_job_hw_submit`
unmasks and clears `PPU_0`/`PPU_1` alongside `DPU_0`/`DPU_1`, and
`rocket_job_irq_handler` accepts those bits in its raw-status check instead
of returning `IRQ_NONE`. It is unconditional -- every job arms both blocks --
which is what this stack needs, because a standalone PPU-flying pooling job
(`PPU_RDMA -> PPU`, bypassing CNA/CORE/DPU entirely; see `iree-rocket-hal`'s
`build_pooling_regcmd`) raises only the PPU's own completion interrupt. Stock,
such a job never reaches the hardware completion path at all and is retired
500 ms later by `drm_sched`'s timeout, reset and resubmit short-circuit.

It was written against `drm-misc-next` rather than Armbian's module, so check
it applies before building: it touches `rocket_job.c`, the same file `088`
does. A functionally equivalent change, plus diagnostic register reads and a
richer timeout log line, also exists in the maintainer's `7.1-rockchip` tree,
which is what `MULTICORE.md` refers to.

**`081`, the clock lever** (Greg Tomko, rebased here)**.** Adds
`clk_set_rate` on runtime-resume and a park
back to the vendor's 200 MHz `POWER_DOWN_FREQ` on suspend, driven by two
writable module parameters:

| Parameter | Default | What it is |
|---|---|---|
| `rocket_npu_clk_hz` | 0 (leave the DT rate alone) | The rate to request on resume. The measured configuration loads `rocket_npu_clk_hz=600000000`. |
| `rocket_autosuspend_ms` | 1000 | Runtime-PM autosuspend delay. Stock is 50 ms, which once the park exists fires *inside* a single inference rather than between inferences. |

This rebase does **not** carry a voltage parameter. `rocket_npu_uv` belongs to
`082` in the base series, which has not been rebased onto Armbian's module and
is not part of the configuration measured here. One hunk of `081` is
hand-merged for 7.3.5: its suspend path counts clocks with `soc->num_clks`.
Touches `rocket_core.c`, `rocket_core.h`, `rocket_drv.c`.

**`088`, reset before detach** (Greg Tomko, rebased here)**.** Moves
`iommu_detach_group` to *after*
`rocket_core_reset` and under `job_lock`, so a wedged core is reset before its
IOMMU is touched, and the detach that follows forces a clean re-attach of the
page-table base the reset wiped. 17 insertions in `rocket_job.c`. The
userspace half of the same defect -- a dense feature DMA reading past the end
of its input -- is already fixed in `rocket-core`
(`ConvPlan::dense_input_read_bytes`); this is the kernel half that keeps a
timeout from corrupting memory.

## Do not substitute `093`

The base series linked below also carries
`093-rocket-drv-ppu-completion.patch`. That is the **flag-gated** variant: it
retires a job on the PPU only when userspace sets `DRM_ROCKET_JOB_PPU_DONE`,
and this stack never sets that flag. Taking `093` in place of `0001` leaves
pooling exactly as broken as stock, which is a confusing way to lose an
afternoon.

## Applying them

Against an Armbian `7.3.5-edge-rockchip64` kernel source tree, with the
`rocket` driver present at `drivers/accel/rocket/`:

```sh
for p in /path/to/kernel-patches/0*.patch; do
  git apply --check "$p" && git apply "$p"
done
# then build the module out of tree, or rebuild the kernel package
```

Installing a module by hand has one trap: `updates/` only wins over
`kernel/drivers/` after `depmod -a`, so a `.ko` dropped there without it is
inert.

```sh
sudo cp rocket.ko /lib/modules/$(uname -r)/updates/
sudo depmod -a && sudo modprobe -r rocket && \
  sudo modprobe rocket rocket_npu_clk_hz=600000000
```

Then check that it took:

```sh
cat /sys/kernel/debug/clk/clk_npu/clk_rate        # expect 600000000
grep . /sys/module/rocket/parameters/*            # expect the two parameters
cat /sys/devices/platform/*.npu/power/runtime_status
```

**Restoring stock is not simply writing `0`.** The suspend-time park is itself
gated on `rocket_npu_clk_hz != 0`, so writing `0` while the clock is raised
strands it at 600 MHz and a "stock" comparison then silently runs fast. Write
`200000000`, wait for every core to read `suspended`, and only then write `0`.

## Provenance, licence and upstream status

Every file here is **GPL-2.0**, as a derivative work of Linux. This
repository's own Apache-2.0-with-LLVM-exception licence does not cover this
directory.

Authorship as the files themselves declare it, which `git mailinfo` will
confirm:

| File | Author | Where it comes from |
|---|---|---|
| `0001-accel-rocket-Add-PPU-status-checks-to-irq-handler.patch` | this project's author | Patch 1 of a local series on top of `drm-misc-next`. |
| `081-rocket-drv-npu-clk-armbian-7.3.5.patch` | **Greg Tomko** | [`gregordinary/patches`](https://github.com/gregordinary/patches), `rocket/081-rocket-drv-npu-clk.patch` (2026-06-30), rebased here onto Armbian 7.3.5 + the RK3576 series. |
| `088-rocket-drv-reset-before-iommu-detach-armbian-7.3.5.patch` | **Greg Tomko** | [`gregordinary/patches`](https://github.com/gregordinary/patches), `rocket/088-rocket-drv-reset-before-iommu-detach.patch` (2026-07-31), rebased here onto Armbian 7.3.5. |

That project carries more of the series than the three changes this stack
needs -- voltage coupling, IOMMU keep-attach, batched submit, the flag-gated
PPU completion and others -- so read it before assuming a problem here is a
problem there.

**None of the three is upstream.** `0001` exists only on local branches of a
`drm-misc-next` checkout, so a reader should not expect to find it in a
released kernel, and the other two are out-of-tree by construction.

Two questions about the final shape of the clock fix are open rather than
settled:

- Is devfreq with an OPP table and a governor the right upstream form of the
  clock fix, instead of module parameters?
- Can the suspend-time park be fixed so that writing `0` genuinely restores
  stock?

Both are tracked in
[`agent-docs/open-questions.md`](../agent-docs/open-questions.md) 2.6.1 and
2.6.2. Getting the PPU change and a devfreq-shaped clock fix upstream is the
most useful thing anyone working on this driver could do for this project.
