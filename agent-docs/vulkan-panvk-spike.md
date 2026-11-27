# panthor/panvk spike: IREE's Vulkan backend on the Mali-G610 (2026-11-26)

Can the RK3588's Mali-G610 run IREE through mainline `panthor` + Mesa `panvk`,
and can that coexist with iree-rocket? The board is the same one the NPU work
uses, so a second accelerator backend is either a comparison baseline or a
fallback for ops the NPU refuses.

**Verdict: yes for IREE, and the blocker was ours, not ARM's.** panvk compute
is correct and now exposes `shaderFloat16`; the thing that produced wrong
answers was a cache-maintenance bug in *IREE's* Vulkan HAL, fixed by a local
patch. One compiler now carries both `vulkan-spirv` and `rocket`, and its
Vulkan output is bit-identical to the stock wheel's.

**And the answer to "is the GPU worth it" is no, by an order of magnitude.**
A module split across the NPU and the Mali now *compiles* with both
backends in it, and does not *run*: IREE 3.14 expects the two HAL drivers to
share buffers and neither can (section 6). But the question that mattered
does not need the mixed module -- the whole model on the Mali, the whole
model on the NPU, and the whole model on the CPU all run and all agree, and
on this fixture the NPU is **10.8-12.2x faster than the Mali**, while the
Mali's own softmax -- the op the NPU refuses, which is the only reason to
want it -- is **slower than the eight-core CPU's** (section 7).

Status marker: sections 1-5 written 2026-11-26/27 against
`claude/mali-vulkan`; sections 6-8 added 2026-11-27 and are the mixed-module
and performance half. Related: [open-questions.md](open-questions.md) for
what the stack does not know, [../LIMITS.md](../LIMITS.md) for what is
measured to work, and `tools/mixed_device_gate.py`, which reproduces every
number in sections 6 and 7.

## 1. The stack, and one trap

`orangepi` is `planck` re-flashed to Armbian 7.3.5 -- 7.3.6 since 2026-11-27,
which is what sections 6 and 7 were measured on, with no change to any of
this. `panthor` is loaded and
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
-- the same starved-clock trap as the NPU's. **Nothing in sections 1-5 is a
timing, and the clock was not pinned for any of them**; section 7 is where it
is pinned, and where how to pin it (and how to tell) is worked out.

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

## 6. The mixed module: the compile is the easy half (2026-11-27)

`tools/mixed_device_gate.py` builds four arms from one MLIR -- a convolution
the Rocket matchers claim followed by a softmax no Rocket matcher will ever
claim -- so which half goes where is a property of the model, not of a flag:

| arm | how | rocket dispatch sites | the rest |
|---|---|---|---|
| `cpu` | `--no-offload` | 0 | 2 llvm-cpu |
| `npu` | the shipping configuration | 1 (30 hardware jobs) | 2 llvm-cpu |
| `gpu` | `--no-offload --cpu-device-backend=vulkan` | 0 | 2 vulkan-spirv |
| `mixed` | `--cpu-device-backend=vulkan` | 1 (30 hardware jobs) | 2 **vulkan-spirv** |

That table is `rocket-compiler audit` reading the `executable-targets` IR, not
an exit code -- section 5's trap is why. The first three arms run and agree.
**`mixed` does not run**, and the rest of this section is why, because the
reason is not where the spike expected it.

### What it took to compile (one flag, no new mechanism)

`rocket-compiler` grew `--cpu-device-backend`, which changes only the IREE
target backend behind the device the CLI already called `cpu_device` and
already placed every unclaimed dispatch on:

```
rocket-compiler compile --input model.mlir --output model.vmfb \
  --cpu-device-name=gpu_device --cpu-device-backend=vulkan \
  --iree-flag=--iree-vulkan-target=valhall3
```

`--cpu-device-name` is now load-bearing rather than documentation:
`resolve_transform_spec` rewrites the device symbols the transform spec
hardcodes (`spec::rename_device`, 114 references), so renaming the device no
longer leaves the spec's `#hal.device.affinity<@cpu_device>` and
`#hal.device.topology<links = ...>` dangling. Without the rename a GPU
would be called `@cpu_device` throughout the IR, which is exactly the kind of
label this repository has been burned by before.

Nothing else was needed, because the three mechanisms that make a split work
were already in place for the NPU/CPU split:

- the transform spec stamps `stream.affinity =
  #hal.device.affinity<@rocket_device>` on every conv dispatch it splices in,
  so the NPU half names its own device;
- `rocket-pin-unclaimed-dispatches` stamps `stream.affinity.default` on every
  dispatch that has no affinity, and that default is
  `--iree-hal-default-device`, which the CLI points at the non-Rocket device.
  Make that device `vulkan` and the remainder of the model is SPIR-V;
- the spec's `stream.topology` names the same two globals, and the rename
  carries it along.

The audit's non-Rocket bucket also used to be labelled `cpu` unconditionally.
It now reads the backend off the `hal.executable.variant` line, so a Mali
dispatch is reported as `vulkan-spirv` rather than as a CPU one.

### Why it does not run

IREE 3.14 does not insert a device-to-device transfer between two unrelated
allocators. It allocates each cross-device buffer **once, from one device**,
and hands the same `iree_hal_buffer_t` to the other device's command buffer.
Four steps establish that, in order:

**1. "Don't share" is not an option IREE offers.** With no
`transparent_access` link between the two devices in `stream.topology`, the
compile fails outright:

```
error: failed to legalize operation 'hal.allocator.resolve_memory_properties':
  affinity = #hal.device.optimal<[#hal.device.affinity<@gpu_device>,
                                 #hal.device.affinity<@rocket_device>]>
```

`ResolveTopologyQueries.cpp` resolves an `optimal` affinity only when every
device in it has the same device ID, or when one of them has
`transparent_access` to all the others; otherwise the pass fails and the
conversion stops. The affinity itself is unavoidable --
`DeviceAffinityAttr::joinOR` produces `#hal.device.optimal` for any resource
two devices touch.

**2. With the link, the program shares one buffer.** The spec's topology
already declares `transparent_access` both ways between its two globals, and
the rename gives the GPU that link, so the compile succeeds. The VM program
then does this, in this order:

```
%props   = hal.allocator.resolve_memory_properties
             affinity = #hal.device.optimal<[@gpu_device, @rocket_device]>
%dev     = hal.allocator.select %props, [(@gpu_device, q), (@rocket_device, q)]
%shared  = hal.device.queue.alloca %dev, %props          # one buffer
...      = hal.command_buffer.dispatch   <rocket conv>   # on @rocket_device
...      = hal.command_buffer.copy_buffer <conv out> -> %shared
...      = hal.command_buffer.dispatch   <gpu softmax>   # on @gpu_device, reads %shared
```

`hal.allocator.select` is resolved at *runtime*, and with no
`allocator_select` policy installed `iree_hal_module_allocator_select` takes
the first pair -- the GPU, because that is the order
`#hal.device.optimal` puts them in. So the shared buffer is a Vulkan buffer
that the Rocket command buffer writes into.

**3. Neither driver can take the other's buffers.**

- `rocket-hal-driver` resolves every binding with
  `&*(buffer as *const RocketBuffer)` to read `dma_address` and `host_ptr`.
  There is no type check anywhere. `device.rs` says so in as many words --
  *"no cross-driver import support ... it's the only HAL driver in the process
  in any realistic near-term usage"* -- and `allocator.rs`'s `import_buffer`
  is a `status_stub!`. A foreign buffer is a wild pointer.
- IREE's Vulkan HAL *does* check, and refuses.

**4. Even the ABI buffers are the wrong kind.** `iree-run-module` allocates a
module's inputs and outputs from a single device (`--device_lead_allocator`),
so whichever device is not the lead is handed foreign I/O. Both orders fail,
and the two failures are the two halves of step 3:

```
--device=vulkan://GPU-b513... --device=rocket
  EXEC @main
  Segmentation fault                         # Rocket casts a Vulkan buffer

--device=rocket --device=vulkan://GPU-b513...
  EXEC @main
  iree-src/runtime/src/iree/hal/drivers/vulkan/buffer.c:366:
    FAILED_PRECONDITION; buffer is not backed by the Vulkan HAL;
    while invoking native function hal.fence.await
```

Tested with and without a cross-device *transient* (a `linalg.fill` init
instead of an `%init` argument removes the only other shared tensor): both
orders fail identically either way, so this is not a property of the fixture.

**The segfault is ours, and is a bug on its own.** A driver that blind-casts
every binding is fine while it is the only driver in the process and is a
latent crash the moment it is not. It should return
`IREE_STATUS_INVALID_ARGUMENT` for a buffer whose vtable is not its own --
cheap, and it turns this whole class of failure into a diagnostic.

### What is actually missing

Not affinity wiring, and not compiler work: the compiler already emits a
correct two-device program. The gap is in the runtime, and it is one of two
things.

1. **The Rocket HAL stages foreign bindings.** Check every binding against
   this driver's buffer vtable and copy a foreign one through a scratch BO on
   the way in and back out. The machinery half-exists -- `stage_direct`
   already stages a binding whose GEM mapping stops short of what the feature
   DMA over-reads -- but the type check, the output copy-back, and an answer
   for the identity questions that currently key on `dma_address`
   (`binding_extent`, `chainable_cube`, the compaction bookkeeping) do not.
   `DRM_ROCKET_CREATE_BO` cannot wrap a host pointer -- there is no userptr
   ioctl in `drivers/accel/rocket` -- so this is a **copy, not an import**: a
   whole tensor at every device boundary, in both directions.
2. **The Vulkan HAL imports Rocket BOs.** IREE's Vulkan allocator does
   implement `import_buffer` for
   `IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION` through
   `VK_EXT_external_memory_host`, and a Rocket BO is permanently host-mapped,
   so the raw material is there. But nothing in IREE's generated code ever
   *calls* `import_buffer` for a cross-device buffer -- it passes the handle
   straight through -- so this route needs compiler work too, on top of
   whatever panvk does or does not expose.

The first is cheaper and lives entirely in this repository. Neither is "wire
up affinities", which is where this section stops -- and, given section 7, it
is work with no measured payoff behind it.

## 7. The first GPU numbers (2026-11-27)

Board `orangepi`, Armbian `7.3.6-edge-rockchip64`, all three CPU clusters
`performance`, NPU at 600 MHz (`rocket_npu_clk_hz=600000000`), Mali devfreq
floor raised to its 1000 MHz ceiling. `simple_ondemand` is the **only**
governor the kernel offers for the GPU, so `min_freq = max_freq` is the lever;
`cur_freq` and `target_freq` are *not* the check, since devfreq reports the
200 MHz OPP for both whenever the GPU is powered down, and an idle board
therefore reads 200 MHz however the knobs are set. What confirms the clock
**stayed** pinned is `trans_stat`, read either side of each case's timings:
11796 ms and 11432 ms at 1000 MHz respectively, with no residency at any OPP
between the suspend floor and the ceiling.

`tools/mixed_device_gate.py --board orangepi`, five runs per arm, NPU
quiesced to `suspended` before each.

| | conv_softmax_32 | | conv_softmax_56 | |
|---|---|---|---|---|
| **arm** | wall (ms) | loop (ms) | wall (ms) | loop (ms) |
| `cpu` (8 cores, `--no-offload`) | 1766 (1635-1784) | 1594 | 220 (218-227) | 186 |
| `npu` | **70** (67-76) | **17.6** | **57** (55-60) | **10.6** |
| `gpu` (Mali) | 756 (751-761) | 628 | 692 (686-703) | 561 |

`wall` is one process per run, timed on the board, which is what a caller
pays. `loop` is `iree-benchmark-module`'s median over 5 repetitions -- **a
loop benchmark, which on this board overstates by roughly 1.4-1.9x** because
it keeps every page resident while a real module touches them once. Both are
given rather than one, because their difference is itself a phase.

Shapes: `conv_softmax_32` is 30x30 output, Cin = Cout = 512, k3x3 s1 (2.12
GMAC) with a channel softmax over 460800 f32; `conv_softmax_56` is 56x56,
Cin = Cout = 256 (1.85 GMAC) with a softmax over 802816. Both convolutions
are fp16 in, f32 out; both are claimed by the Rocket matchers (30 and 12
hardware jobs). Every arm was checked against the `--no-offload` arm on the
same inputs: `npu` within max|err| 3.11e-4 and 2.15e-4, `gpu` within 2.09e-7
and 7.45e-8 -- the GPU arm is essentially exact against the CPU, which is
also the standing check that the non-coherent-mapping patch is still applied.

### Marginal cost per phase

Subtracting a softmax-only module of the identical shape from each arm's loop
figure attributes the time instead of leaving a ratio:

| | conv_softmax_32 | | | conv_softmax_56 | | |
|---|---|---|---|---|---|---|
| **arm** | total | softmax | conv | total | softmax | conv |
| `cpu` | 1594.0 | 0.7 | 1593.3 | 186.0 | 1.3 | 184.7 |
| `npu` | 17.6 | 0.7 (on CPU) | **16.9** | 10.6 | 1.3 (on CPU) | **9.3** |
| `gpu` | 628.0 | 1.1 | 626.9 | 561.0 | 1.1 | 559.9 |

And per-process setup, which is `wall - loop`: 172/34 ms for `cpu`, 52/46 ms
for `npu`, **128/131 ms** for `gpu`. Creating a Vulkan device and its
pipelines costs about 130 ms every process, three times the NPU's
first-touch cost and more than the entire `npu` arm.

### What the numbers say

**1. The op the NPU refuses is not worth a device boundary.** The softmax --
the whole reason to want a second accelerator -- is 0.7 ms and 1.3 ms on the
CPU, and 1.1 ms on the Mali both times. The Mali is 1.6x *slower* on the
smaller one and 1.2x faster on the larger, which is to say both are dispatch
overhead and neither is computing. Against that, moving it to the GPU costs a
cross-device copy of 1.8 MB or 3.2 MB (section 6's `copy_buffer`) plus the
130 ms of Vulkan setup. This is the same shape as the answer already recorded
for `--elementwise` and for attention: at the current per-dispatch cost, an
op that does less arithmetic than its own dispatch tax should stay where it
is. It answers open item 3 in the negative.

**2. The NPU beats the Mali by 36-60x on the convolution**, which is the
honest comparison between the two accelerators, since it is the only phase
either of them really runs. 16.9 ms and 9.3 ms against 627 ms and 560 ms.

**3. But the Mali number is a codegen result, not a hardware ceiling, and
must not be quoted as one.** 627 ms for 4.25 GFLOP is 6.8 GFLOP/s, and 560 ms
for 3.70 GFLOP is 6.6 GFLOP/s -- the same throughput for two different
shapes, on a GPU whose fp32 peak is in the hundreds of GFLOP/s. IREE's
`vulkan-spirv` convolution lowering for `valhall3` is evidently untuned
rather than the G610 being slow: a shape-insensitive ~6.7 GFLOP/s is what an
unvectorised, untiled loop looks like. Anyone who wants a real
Mali-vs-NPU number has to fix that first, and the NPU's 36-60x is a lower
bound on how much headroom would have to be found before the question
changes.

**4. The `cpu` arm's `conv_softmax_32` figure is a cliff, not a baseline.**
1593 ms for 2.12 GMAC is 2.7 GFLOP/s, against 20.0 GFLOP/s for
`conv_softmax_56`'s 1.85 GMAC -- a 7.4x per-MAC difference between Cin = Cout
= 512 and Cin = Cout = 256 on the same eight cores with the same flags. So
the 94x NPU-over-CPU ratio on that case measures an llvm-cpu tiling cliff at
512 channels, not the NPU; `conv_softmax_56`'s 20x is the number to believe.
Flagged here rather than chased: it is an IREE CPU codegen question, not a
Rocket one.

## 8. What is still open

1. **A genuinely mixed module.** *Compile: closed* (section 6) -- one flag,
   `--cpu-device-backend=vulkan`, and the audit proves 1 `rocket` plus 2
   `vulkan-spirv` dispatch sites in one module. *Run: open, and it is a
   runtime problem, not an affinity or compiler one* -- IREE 3.14 shares one
   buffer between the two devices and neither HAL driver can take the
   other's. Section 6's last two subsections say exactly what is missing.
2. **Any performance number.** *Closed* (section 7): three arms, both
   fixtures, clock state recorded and residency-confirmed, phases attributed.
3. **Whether the GPU is worth it for what the NPU refuses.** *Answered, no*
   (section 7). The softmax is ~1 ms on either device, so a device boundary
   costs more than the op; and on the one phase that does real work the NPU
   is 36-60x faster. Re-open it only if IREE's `vulkan-spirv` conv lowering
   for Valhall stops running at a shape-insensitive 6.7 GFLOP/s.
4. `iree-lld` is absent from a fresh build tree, so an llvm-cpu arm compiled
   against the combined library needs
   `--iree-flag=--iree-llvmcpu-embedded-linker-path=iree-build/build/tools/iree-lld`.
   Still true; `tools/mixed_device_gate.py` takes `--linker` for it.
5. **The Rocket HAL should type-check its bindings.** `command_buffer.rs`
   casts every `iree_hal_buffer_t*` to `RocketBuffer` with no check, so a
   foreign buffer is a segfault rather than a status. That is how section 6's
   failure presents, and it will be how any future multi-device or
   external-buffer mistake presents. Returning
   `IREE_STATUS_INVALID_ARGUMENT` when the buffer's vtable is not this
   driver's is cheap and turns a crash into a diagnostic.
6. **Whether a Rocket BO can be handed to panvk as a dma-buf.** Not tried.
   The pieces look present: panvk exposes `VK_EXT_external_memory_host` *and*
   `VK_EXT_external_memory_dma_buf`, and a Rocket BO is a `drm_gem_shmem`
   object whose `drm_gem_object_funcs` provide `pin`/`unpin`/`get_sg_table`,
   which is what `drm_gem_prime_export` needs. What is unchecked is whether
   `DRM_IOCTL_PRIME_HANDLE_TO_FD` is reachable at all on an accel node
   (`accel/rocket` declares only `DRIVER_COMPUTE_ACCEL | DRIVER_GEM`) -- and,
   even if it is, IREE's generated code never calls `import_buffer` for a
   cross-device buffer, so this route needs compiler work too.
7. **The combined compiler is a second full build, not a shipping
   configuration.** `iree-build/configure-compiler-vulkan.sh` now records the
   line, into `iree-build/build-vulkan` rather than `build`, for the reasons
   in its header. ~52 minutes on 16 cores. Nothing in CI builds it.
8. The Vulkan HAL patch is still not upstream, and
   `git submodule update` still drops it silently (section 3).

### Reproducing the correctness gate

The original smoke MLIR was lost with an ephemeral scratchpad. The shapes were
recovered from the `.bin` files still in `~/vk`: solving the three file sizes
gives 256x512 * 512x384 for the matmuls and 1048576 elements for the add, and
`cmp.py`'s reference is `a*b+a`. The rebuilt `smoke.mlir` with `@add`, `@mm`
and `@mm16` therefore compares against the identical reference the
wheel-compiled vmfbs were judged by -- which is what makes the "bit-identical
to the wheel" rows in section 5 meaningful.

Sections 6 and 7 do not depend on that scratch directory: their fixtures are
generated by `tools/mixed_device_gate.py` from the shapes in its own `CASES`
table, and the compiler configure line is
`iree-build/configure-compiler-vulkan.sh`.

Pass inputs as raw `.bin` with an explicit type (`--input=1048576xf32=@a.bin`),
never `@x.npy` -- IREE reads an npy as raw bytes and shifts the tensor. The
output direction is asymmetric: `--output=@y.npy` does write real npy.
