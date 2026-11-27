# panthor/panvk spike: IREE's Vulkan backend on the Mali-G610 (2026-11-26)

Can the RK3588's Mali-G610 run IREE through mainline `panthor` + Mesa `panvk`,
and can that coexist with iree-rocket? The board is the same one the NPU work
uses, so a second accelerator backend is either a comparison baseline or a
fallback for ops the NPU refuses.

**Verdict: yes for IREE, and the blocker was ours, not ARM's.** panvk compute
is correct and now exposes `shaderFloat16`; the thing that produced wrong
answers was a cache-maintenance bug in *IREE's* Vulkan HAL, fixed by a local
patch. One compiler now carries both `vulkan-spirv` and `rocket`, and its
Vulkan output is bit-identical to the stock wheel's. The iree-rocket half is
**still not answered**: a module that actually places work on both backends
needs device affinities plus rocket-compiler's conv dispatches, and there are
**no performance numbers at all** -- every result here is a correctness
result.

Status marker: written 2026-11-27 against `claude/mali-vulkan`. Section 5 is
the combined compiler, which landed the same day; section 6 is what remains
open. Related: [open-questions.md](open-questions.md) for what the stack does
not know, [../LIMITS.md](../LIMITS.md) for what is measured to work.

## 1. The stack, and one trap

`orangepi` is `planck` re-flashed to Armbian 7.3.5. `panthor` is loaded and
bound (`card1` + `renderD128`, `fb000000.gpu`), with CSF firmware in
`/lib/firmware/arm/mali/arch10.8/`; `rockchip-drm` is `card0` and is display
only.

Userspace is **Debian trixie-backports `mesa-vulkan-drivers=26.1.6-1~bpo13+1`**
plus `vulkan-tools`, installed system-wide 2026-11-26. It was chosen over
Armbian's untested Mesa 27.0 nightly.

> **Trap:** apt's candidate is still that Armbian nightly -- priority 500 beats
> backports' 100 -- so a plain `apt upgrade` silently replaces the tested panvk
> unless it is held.

`vulkaninfo` reports Mali-G610 MC4, **Vulkan 1.4.354**, conformance 1.4.1.2,
`shaderFloat16 = true`, `uniformAndStorageBuffer8BitAccess = true`, int8 dot
product, `vulkanMemoryModel`, subgroup size fixed at 16, 32 KB shared memory,
7 descriptor sets, 256 B push constants, no float64. **The f16 gap recorded in
September was Mesa 25.0.7, not the hardware.** llvmpipe is the other Vulkan
device, so always select by name or UUID, never index.

GPU devfreq is `simple_ondemand`, idling at 200 MHz against a 1000 MHz ceiling
-- the same starved-clock trap as the NPU's. **Nothing below is a timing, and
the clock has not been pinned.**

## 2. The bug: IREE 3.14 never flushes non-coherent mappings

IREE 3.14.0 (`5332c68`) on the Mali ran without error and returned wrong data:
**~5-20% of a 1M-element `a*b+a` came back holding the input `a`**, in 64-byte
(cache-line) runs, a different count every run; matmuls were off by 0.4. No
error was reported at any level. llvmpipe on the same vmfb was exact.

It is IREE's bug, not panvk's:

| Step | What happens |
|---|---|
| panvk's memory type 1 | `DEVICE_LOCAL \| HOST_VISIBLE \| HOST_CACHED`, and **not** `HOST_COHERENT` |
| IREE's allocator | ranks `HOST_CACHED` above `HOST_COHERENT`, so it picks exactly that type |
| `iree_hal_vulkan_buffer_map_range` | ignores `memory_access` entirely; never invalidates |
| `iree_hal_vulkan_buffer_unmap_range` | never flushes (it `(void)`s the offset and length) |
| `iree_hal_buffer_view_generate_buffer` | its in-situ path -- how `iree-run-module --input=...=@file` uploads -- maps `DISCARD_WRITE`, writes, unmaps, and so never flushes |

Dirty CPU lines then evict *on top of* the GPU's results, which is why the
corruption is cache-line granular and varies run to run.

**The fix** is `iree-build/patches/0001-vulkan-flush-invalidate-noncoherent-mappings.patch`:
flush on unmap of a write mapping, invalidate on map for read. Both are no-ops
for coherent memory, so nothing else regresses.

**The invalidate alone did not fix it.** On ARM, invalidate is clean+invalidate,
which writes the stale lines back -- so the flush on unmap is the load-bearing
half. Worth remembering before trusting a half-fix here.

A standalone probe (`~/vkprobe/cacheprobe.c` on the board: CPU fills, flush,
GPU writes, invalidate, read) is exact on panvk type 1 and reproduces the bug
**only** when the invalidate is dropped. That is what separates IREE's bug from
a driver bug.

### Correctness gate (board, `~/vk`)

| Case | Result |
|---|---|
| `a*b+a`, 1048576 f32, 5 runs | exact, 5/5 (`x_fix1..5.bin` vs `cmp.py`) |
| matmul 256x512 * 512x384, f32 | **bit-identical** to the CPU |
| same, f16 | within 1 ulp of the CPU |

## 3. Build plumbing (committed, `6d4cca2`)

`ROCKET_WITH_VULKAN` (default **OFF**) gates `IREE_HAL_DRIVER_VULKAN` in the
aarch64 runtime build. It stays off by default because the host `iree-compile`
has no vulkan-spirv target, so the Vulkan executable tests cannot be
generated; build only the tool targets:

```
cmake -S iree-build/host-aarch64 -B iree-build/host-aarch64/build-vulkan \
  -G Ninja -DCMAKE_TOOLCHAIN_FILE=iree-build/aarch64_linux_gnu.cmake \
  -DIREE_HOST_BIN_DIR=iree-build/build/tools -DIREE_BUILD_COMPILER=OFF \
  -DROCKET_WITH_VULKAN=ON
ninja -C iree-build/host-aarch64/build-vulkan iree-run-module iree-benchmark-module
```

The patch must be re-applied by hand after anything touches the submodule --
`git submodule update` drops it silently, and the build then goes back to
returning wrong answers *without failing*:

```
git -C iree-build/iree-src apply ../patches/0001-vulkan-flush-invalidate-noncoherent-mappings.patch
```

It has **not** been sent upstream.

## 4. The two HAL drivers coexist

That build carries the rocket external driver as well, so **one binary**
enumerates both. From the deployed runtime on the board:

```
$ ./iree-run-module --list_drivers      # header lines elided
      local-sync: Local execution using a lightweight inline synchronous queue
      local-task: Local execution using the IREE multithreading task system
          vulkan: Vulkan HAL Rewrite Scaffold
          rocket: Rockchip RK3588 NPU (accel/rocket)
$ ./iree-run-module --list_devices
local-sync://
local-task://
vulkan://GPU-b5130000050067a80000000000000000   # the Mali
vulkan://GPU-6d65736132372e302e307e6769743200   # llvmpipe, ignore
rocket://
```

Select the Mali by UUID; the second Vulkan device is llvmpipe. Compile with
`--iree-hal-target-device=vulkan --iree-vulkan-target=valhall3` (G610 is in the
G710 family).

## 5. The combined compiler (landed 2026-11-27)

The split was only ever our build configuration. Configuring `iree-src` with
`IREE_TARGET_BACKEND_VULKAN_SPIRV=ON` alongside
`IREE_CMAKE_PLUGIN_PATHS=<worktree>/rocket-compiler-plugin` registers both as
static plugins, and the build (8144 steps, 51m35s on 16 cores, no ccache)
yields one `libIREECompiler.so` plus `iree-compile` that reports:

```
Registered target backends:
  llvm-cpu   rocket   vmvx   vmvx-inline   vulkan-spirv
```

Built into a scratch tree, **not** `iree-build/build`, which is stale against
its own sources and would trigger a ~7400-step rebuild of the compiler
everything else depends on. There is no shortcut through dynamic plugins: the
rocket plugin is registered statically, and IREE's dynamically-loaded-plugin
mechanism is experimental and off in our build
(`IREE_EXPERIMENTAL_COMPILER_DYNAMIC_PLUGINS=OFF`), so the stock wheel cannot
be taught the rocket target. One combined build is the route.

### It is a drop-in for the wheel on the Vulkan side

`smoke.mlir` compiled by *our* build, run on the Mali through the patched
runtime, against the CPU references and against the wheel-compiled vmfbs:

| Case | vs CPU | vs wheel-compiled vmfb |
|---|---|---|
| `@add`, 1048576 f32 | 0 elements over 1e-5, 0 stale inputs, max\|err\| 9.54e-7 | -- |
| `@mm`, f32 | **bit-identical** (0 of 98304 differ) | **bit-identical** |
| `@mm16`, f16 | not bit-exact: 5415 of 98304 differ (5.5%), max\|err\| 2^-10 | **bit-identical** |

Both matmuls match the wheel bit for bit, so adding the rocket plugin
perturbs vulkan-spirv codegen not at all.

> **Correction to an earlier shorthand.** "f16 matmul within 1 ulp of the CPU"
> is loose and should not be used as a gate. All values lie in [0, 1.006], and
> the worst absolute disagreement is 2^-10 -- one f16 step at the *top* of that
> range. But 3394 elements differ by more than one step at their *own*
> magnitude, concentrated where the 512-long dot product cancels toward zero.
> That is a different accumulation order, not a defect, and since our output is
> bit-identical to the wheel's it is inherent to the vulkan-spirv lowering
> rather than to this build. f32 *is* bit-exact; do not expect it at f16.

### The rocket backend is reached, and refuses this module by design

`--iree-hal-target-device=rocket` on `smoke.mlir` exits 74 with the backend's
own diagnostic:

```
'hal.executable.variant' op rocket backend v1 only supports hand-spliced
conv2d dispatches; executable target config is missing required key 'input_width'
```

That is the expected refusal, and it confirms the plugin is registered and
reached: the NPU path is driven by rocket-compiler's two-stage pin/claim flow
over conv dispatches, not by handing arbitrary elementwise or matmul IR to the
backend.

### Trap: a two-device compile can be silently single-device

```
--iree-hal-target-device=gpu=vulkan --iree-hal-target-device=npu=rocket
```

**exits 0** -- and produces a module with no rocket in it whatsoever.
`--compile-to=hal` shows one device global (`@gpu`), zero occurrences of
`rocket`, and only `vulkan-spirv` executable targets: with no affinities every
dispatch goes to the first device and the second is dropped silently. So exit 0
from a multi-device flag line is **not** evidence of a mixed module.

(A second false friend while checking this: grepping the IR for `npu` matches
inside `input`. Grep for the backend name, not the device name.)

## 6. What is still open

1. **A genuinely mixed module.** Needs explicit device affinities *and*
   rocket-compiler's hand-spliced conv dispatches -- both, per the two findings
   above. Having the backends in one compiler was the cheap half.
2. **Any performance number.** Correctness only, on an add and one matmul
   shape. No GPU-vs-NPU-vs-CPU comparison exists, and the devfreq must be
   pinned before one is quoted.
3. **Whether the GPU is worth it for what the NPU refuses.** The actual payoff
   question, and entirely untouched.
4. `iree-lld` is absent from a fresh build tree, so an LLVMCPU arm needs
   `--iree-llvmcpu-embedded-linker-path` pointed at the existing build's copy.

### Reproducing the correctness gate

The original smoke MLIR was lost with an ephemeral scratchpad. The shapes were
recovered from the `.bin` files still in `~/vk`: solving the three file sizes
gives 256x512 * 512x384 for the matmuls and 1048576 elements for the add, and
`cmp.py`'s reference is `a*b+a`. The rebuilt `smoke.mlir` with `@add`, `@mm`
and `@mm16` therefore compares against the identical reference the
wheel-compiled vmfbs were judged by -- which is what makes the "bit-identical
to the wheel" rows in section 5 meaningful.

It has now been lost once; **the rebuilt module and the compiler configure line
belong in the repo** rather than in a scratch directory, if this spike goes any
further.

Pass inputs as raw `.bin` with an explicit type (`--input=1048576xf32=@a.bin`),
never `@x.npy` -- IREE reads an npy as raw bytes and shifts the tensor.
