# Kernel patches — Armbian `7.3.6-edge-rockchip64`

Every measured result in this repository was taken on an RK3588 running
mainline `accel/rocket` **as Armbian 7.3.5 ships it, plus the three changes
below**. All three are here as patch files. None of them is upstream.

The board (`ssh orangepi`) has run Armbian **`7.3.6-edge-rockchip64`** since
2026-11-27, with the same three changes. These files apply unchanged to
either kernel, with plain `git apply` or `git am`, and reproduce the module
sources built on the board for each, byte for byte. On 7.3.6 the patched
module has so far been checked end to end with MobileNetV2 fp16 on the NPU
(within 0.0049 of the CPU-only module, same top-1), not with the full gate, so
the timings elsewhere in this repository are still 7.3.5 numbers.

On a stock module, at either version, the stack does not merely run slower:
every pooling job is killed by the watchdog, so most of the surveyed models fail outright.
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
> What is vendored here are **rebases onto Armbian 7.3.5, carried to
> 7.3.6**, which that project does not carry -- it has the originals,
> against other kernels.
> Only `0001`, the PPU change, originates with this project.
>
> Each rebased file carries Greg's original `From:`, `Date:`, `Subject:`,
> commit message and `Signed-off-by:`, with the rebase recorded as a
> bracketed note above the diff (one per kernel it was carried to) -- so `git am` credits the right author, and
> the measurements and reasoning in those messages (the refcounted clock
> park, the autosuspend sweep, the stall-timeout A/B on RK3588 silicon) come
> with them. `git format-patch` had rewritten both to name the rebaser and
> dropped the bodies; that is repaired here.

| Change | Base it was written against | Without it |
|---|---|---|
| [`0001-accel-rocket-Add-PPU-status-checks-to-irq-handler.patch`](0001-accel-rocket-Add-PPU-status-checks-to-irq-handler.patch) | `drm-misc-next`; rebased here onto Armbian 7.3.6 | Armbian's module arms only the DPU interrupts, so **every pooling job is watchdog-killed at ~500 ms**. A single-tile pool still returns the right answer; a tiled one does not, because only its first task runs before the kill. |
| [`081-rocket-drv-npu-clk-armbian-7.3.6.patch`](081-rocket-drv-npu-clk-armbian-7.3.6.patch) | Armbian 7.3.5; unchanged for 7.3.6 | The NPU stays pinned at 200 MHz, the rate the DT assigns and mainline has no devfreq to change. ResNet50-224 fp16 runs ~105 ms rather than 45, int8 150 rather than 125. Output is bit-identical at either clock; only the time changes. |
| [`088-rocket-drv-reset-before-iommu-detach-armbian-7.3.6.patch`](088-rocket-drv-reset-before-iommu-detach-armbian-7.3.6.patch) | Armbian 7.3.5; unchanged for 7.3.6 | A job timeout detaches the IOMMU group while the core is still mid-DMA, the `rk_iommu` stall handshake times out, and in-flight DMA can land in pages the kernel has already reused -- observed as page-table corruption in the submitting process after a burst of timeouts. |

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

It was written against `drm-misc-next`, and that original does **not** apply
to Armbian's module: the RK3576 series puts a `task_con_16bit` branch right
after the submit hunk, so `git apply` rejects it (GNU `patch` takes it only
at fuzz 2). Until 2026-11-27 this directory vendored that original, and the
old loop below printed one error for it and carried on, leaving pooling
broken. The file here is rebased onto Armbian and records the original base
in a bracketed note; the two hunks are the same five lines. A functionally
equivalent change, plus diagnostic register reads and a richer timeout log
line, also exists in the maintainer's `7.1-rockchip` tree,
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
hand-merged for Armbian: its suspend path counts clocks with `soc->num_clks`.
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

Against an Armbian `7.3.6-edge-rockchip64` (or `7.3.5`) kernel source tree,
with the `rocket` driver present at `drivers/accel/rocket/`:

```sh
for p in /path/to/kernel-patches/0*.patch; do
  git apply --check "$p" && git apply "$p" || { echo "FAILED: $p"; break; }
done
# then build the module out of tree, or rebuild the kernel package
```

Building out of tree, as the board does, means copying `drivers/accel/rocket/`
from a source tree **of the running kernel's version** into a directory of its
own, applying the patches there with `git apply -p4` (their paths start at
the kernel root), and building against that kernel's headers from Armbian's
`linux-headers-edge-rockchip64` package:
`make -C /lib/modules/$(uname -r)/build M=$PWD modules`. A driver directory
copied from 7.3.5 does not compile against 7.3.6 headers -- see
[the next section](#when-armbian-bumps-the-kernel).

Installing a module by hand has one trap: `updates/` only wins over
`kernel/drivers/` after `depmod -a`, so a `.ko` dropped there without it is
inert. The module parameter comes from `/etc/modprobe.d/` (the board has
`options rocket rocket_npu_clk_hz=600000000` in `rocket-clk.conf`), and the
initramfs carries that file but not `rocket.ko`, so nothing needs
regenerating.

```sh
sudo install -m 644 rocket.ko /lib/modules/$(uname -r)/updates/rocket.ko
sudo depmod -a
sudo reboot
```

**Reboot rather than `modprobe -r rocket`.** Unloading `rocket` also unloads
`drm_shmem_helper`, and on a kernel already tainted by NPU-timeout IOMMU
warnings -- exactly the state a stock module leaves behind -- reloading that
helper has failed BTF validation (`BTF: -22`), after which `rocket` would not
load at all. A reboot is the reliable reload.

### The 7.3.6 module on the board

| | |
|---|---|
| Source tree | `orangepi:~/rocket-ko/armbian-ppu-clk-088-7.3.6/` (the 7.3.5 tree, `~/rocket-ko/armbian-ppu-clk-088/`, plus the stable sync below) |
| Installed at | `/lib/modules/7.3.6-edge-rockchip64/updates/rocket.ko` |
| sha256 | `5d07d3cfa2a0c8159dccbfcb795c17311363321fcbe1fba42cd4ba56b71afb6e` |
| Loaded with | `rocket_npu_clk_hz` = `600000000` |

Rebuilt from these files on 2026-11-27 (the stock 7.3.6 driver plus the
three patches, `git apply -p4`, then the `make` line above), the module
compiles cleanly. It matches the installed one in every section except the
debug info and `.note.gnu.build-id`. Those two differ only because the build
directory's path is baked into the debug info, and the build-id is a hash
over all of it. **So a rebuild's sha256 never matches the installed
module's, even when the code is the same.** Compare code, not files:
`readelf -sW` FUNC sizes (48 here), or every non-`.debug*` section with
`objcopy -O binary --only-section=<s>`.

## When Armbian bumps the kernel

Each new Armbian kernel package installs a fresh `/lib/modules/<version>/`
with **no `updates/`**, so the board silently boots the stock module again.
The symptom is unmistakable: pooling dispatches take ~500 ms and the HAL
refuses them at its hung-job floor, so MobileNetV2 fails while a model
without NPU pooling (ViT-B/16) still runs. That is how 7.3.6 was found on
2026-11-27.

### What changed from 7.3.5 to 7.3.6

Upstream stable made exactly two changes to `drivers/accel/rocket` between
`v7.3.5` and `v7.3.6`, and Armbian's RK3576 series did not change:

1. `rocket_gem.c` adds `#include <drm/drm_timeout.h>`, because
   `drm_timeout_abs_to_jiffies` moved out of `drm_utils.h`. This is why a
   7.3.5 driver directory no longer compiles against 7.3.6 headers.
2. `rocket_job.c` drops `drm_sched_fini(&core->sched);` (and its blank line)
   from `rocket_job_init`'s `err_sched` error path.

Neither touches a hunk of the three patches, which is why the same files
apply to both.

### The recipe

1. **Diff the stable tags for the driver.** Fetch each file under
   `drivers/accel/rocket/` at both tags from
   `https://raw.githubusercontent.com/gregkh/linux/<tag>/drivers/accel/rocket/<file>`
   and diff them. git.kernel.org's anti-bot wall blocks scripted fetches, so
   use the GitHub mirror. Check Armbian's
   `patch/kernel/archive/rockchip64-<series>/rk3576-npu-*.patch` too: Armbian's
   module is upstream *plus* that series, not pristine upstream.
2. **Build an unpatched base** -- the previous unpatched driver tree plus
   that stable delta -- against the new kernel's headers.
3. **Confirm the base is stock.** The unpatched rebuild must match the
   `rocket.ko` Armbian ships for the new kernel, function for function:
   `readelf -sW` on both, compare every `FUNC` symbol's size. For 7.3.6 all
   48 matched. A mismatch means the base is wrong; do not patch on top of it.
4. **Apply the three patches** with the loop above. If a patch stops applying,
   rebase it, add a bracketed note recording the new kernel, and rename the
   file's suffix to match.
5. **Build the patched module.** When checking it against an earlier build,
   compare sections or FUNC sizes, not sha256 (see above).
6. **Install to `updates/`, `depmod -a`, reboot** -- as above, never
   `modprobe -r`.
7. **Check it took**, below, then run a pooling model end to end on the NPU.

## Checking the module

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
| `0001-accel-rocket-Add-PPU-status-checks-to-irq-handler.patch` | this project's author | Patch 1 of a local series on top of `drm-misc-next`, rebased here onto Armbian. |
| `081-rocket-drv-npu-clk-armbian-7.3.6.patch` | **Greg Tomko** | [`gregordinary/patches`](https://github.com/gregordinary/patches), `rocket/081-rocket-drv-npu-clk.patch` (2026-06-30), rebased here onto Armbian 7.3.5 + the RK3576 series, unchanged for 7.3.6. |
| `088-rocket-drv-reset-before-iommu-detach-armbian-7.3.6.patch` | **Greg Tomko** | [`gregordinary/patches`](https://github.com/gregordinary/patches), `rocket/088-rocket-drv-reset-before-iommu-detach.patch` (2026-07-31), rebased here onto Armbian 7.3.5, unchanged for 7.3.6. |

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
