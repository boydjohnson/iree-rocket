# Multicore: choosing the NPU core count at runtime

Design note, 2026-09-06. Status: proposal, nothing implemented. Resolves
ISSUES.md P1 ("one fd is one scheduler entity is one core").

## 1. What the driver does today

- **One `open()` of `/dev/accel/accel0`** (`device.rs` `create`). The
  allocator gets a `try_clone()` of it. A dup shares the `struct file`, so
  it shares the DRM file's GEM handle namespace, IOVA domain *and scheduler
  entity* -- a dup is not a second core. Every IREE buffer, every scratch
  BO, every regcmd BO, every weight-cache packing lives on this one file.
- **`queue_execute` replays a command buffer serially** (`device.rs`
  ~1506-1800): for each recorded dispatch, for each CBUF row-split task,
  `SUBMIT` then a blocking `PREP_BO` on the output handles, all under the
  device-wide `last_dpu_mode` mutex. No two hardware jobs are ever in
  flight at once, whatever IREE does.
- **"Thread per dispatch"** is `run_after_wait` (`device.rs` ~441): when a
  queue op's wait semaphores are not already satisfied it spawns a detached
  `std::thread` to wait, do the work and signal. Under
  `--device=rocket --device=local-task` every NPU command buffer waits on a
  CPU-produced semaphore, so this is effectively one OS thread per NPU
  dispatch. Those threads then serialize on the mutex above, so they buy
  nothing except letting `queue_execute` return promptly. Host-side layout
  work (pack / compact) runs on whichever thread called, after
  `prefer_fast_cpus()`.

## 2. What the kernel gives us (mainline `rocket`, read from `../rknpu-spelunking/mainline-rocket-driver`)

| Fact | Where | Consequence |
|---|---|---|
| One `drm_gpu_scheduler` per core; `rocket_job_open` creates **one `drm_sched_entity` per file, initialised over all `num_cores` schedulers** | `rocket_job.c:508-526` | An entity picks the least-loaded core only when its own queue is empty; while it has queued jobs it stays put. So "1 fd = 1 core" is more precisely **"1 fd = 1 in-order queue that occupies at most one core at a time"**. N >= 3 files with work queued saturate 3 cores. There is no core-selection field in the uAPI (`drm_rocket_submit.reserved` must be 0, `rocket_job.c:~630`). |
| **Fresh IOMMU domain + `drm_mm` per file** | `rocket_drv.c:72-99` | IOVAs are per file, each a 0..4 GB window. |
| **IOVA mapping happens only in `CREATE_BO`** | `rocket_gem.c:62-96` | A BO is mapped into exactly one file's domain, at creation. |
| `driver_features = DRIVER_COMPUTE_ACCEL \| DRIVER_GEM` -- **no PRIME, no `gem_prime_import_sg_table`, no shared domain** | `rocket_drv.c:149` | There is no way to make a BO created on file A visible to a job on file B. |
| `SUBMIT` resolves handles with `drm_gem_objects_lookup(file, ...)` | `rocket_job.c:570-580` | A job can only name BOs in its own file's handle table. |
| Each core has its own IOMMU group; the job's domain is attached to whichever core runs it | `rocket_job.c:327` | Any file's domain can run on any core -- the load balancing is real. |
| The core index is folded into the regcmd only kernel-side (`extra_bit = 0x10000000 * core->index`) | `rocket_job.c:125` | Our regcmd programs are core-agnostic. Nothing in userspace needs to know which core ran a job. |

The notes' measurement (`../rockchip-npu-notes/perf/iova-and-multicore.md`):
1/2/3/4 threads with their own fd = 1.0 / 2.13 / 2.94 / 3.06x jobs/s on
NPU-bound 64-task jobs; T=4 edges above T=3 because the extra queue fills
each worker's pack/submit/readback bubble.

**The constraint the whole design is built around:** *a job submitted on
file k may reference only BOs created on file k.* No kernel-side sharing
exists on mainline.

## 3. What a job references today, and why the constraint is survivable

`command_buffer.rs` `dispatch_impl` bakes DMA addresses into the regcmd at
record time. The buffers a job touches:

| Operand | Usual source | Direct IREE-buffer exceptions (`addr(&refs[i])`) |
|---|---|---|
| regcmd | private BO allocated in `queue_execute` on `d.file` | none |
| input | packed NC1HWC2 scratch (`InputPacking`) | dense ARGB layouts, Cin 1..=4: `command_buffer.rs:1369` |
| weights | `weight_cache` packing BO | `StagedWeights::direct`, `:1467` (already-blocked / no-pack kinds) |
| bias | scratch (`:1500-1554`) | `:1566` |
| output | compaction scratch (`OutputCompaction`) | `:1864` and the no-compaction case |
| EW / pooling / LUT operands | `input_scratch` / `output_scratch` (`:1889-2294`) | none found |

So the NPU already reads and writes **driver-private scratch almost
everywhere**; the IREE buffers on file 0 are touched by the *host* (pack in,
compact out), not by the hardware. That is exactly what makes a multi-file
design cheap: a worker file needs its own scratch, weight copies and regcmd
BOs, and never needs an IREE buffer. The four direct sites are the only
places that would need a copy hop, and only when the dispatch is placed on a
worker other than file 0.

## 4. What multicore can and cannot buy (set expectations first)

Multicore shortens one profile phase, `wait.npu`, and only for work that can
be in flight concurrently. Measured shares of wall on planck:

| Model | `wait.npu` share | Source |
|---|---|---|
| MobileNetV2 int8 accumulator, 17 sites | 10.0 % | memory `int8-perf-and-phase-profile` |
| MobileNetV2 static-int8 requantized (the 1.80x arm) | 30.6 of ~176 ms, ~17 % | memory `requant-offload-beats-cpu` |
| ViT element-wise, 172 sites | 4.2 % | memory `vit-elementwise-offload-measured` |

Amdahl on `wait` alone caps the win at ~1.07x (requant MobileNetV2, N=3) and
~1.03x on the accumulator model. That is not worth a worker pool.

What a worker pool *also* buys, and what turns this into a real lever:

1. **Host phases overlap NPU time.** Today pack -> submit -> wait -> compact
   is one serial chain on one thread. With W workers each owning a file,
   worker A's compaction runs while worker B's job is on the NPU. On the
   requant arm `compact + pack.input + record` is ~55 ms against `wait`
   30 ms; overlapping them is worth more than the NPU term itself.
2. **Host phases parallelise across A76s.** Per-band input packing and
   per-band compaction (see §6, level 2) split the same bytes across W
   pinned threads; `layout_bench` says these are pure memory movement at
   ~10 GB/s, so they scale until DRAM saturates.
3. **`outside` does not move.** 64 % of the accumulator model's wall is
   IREE's own CPU dispatches. Nothing here touches it; quote the core count
   (`taskset`) with every number, per memory `offload-cost-is-flat-per-dispatch`.

Honest target: **1.15-1.3x on the requant MobileNetV2 arm at N=3**, more on
matmul-heavy ViT where per-site NPU time is a larger share, and a *loss* on
any model whose dispatches are all single-task and strictly chained if the
pool adds a hand-off per dispatch. §8 measures that before anything lands.

## 5. Design

### 5.1 The knob

```text
ROCKET_NPU_CORES=N     N worker contexts (files), 1..=8. Default 1.
ROCKET_NPU_CORES=auto  one per NPU core, read by counting the `*.npu`
                       platform devices under /sys/bus/platform/devices
                       (3 on RK3588, 2 on RK3576, 1 on RK3568); the kernel
                       does not export num_cores.
```

Read once at device create, same convention as every other `ROCKET_*`
knob. Also accepted as a device-URI parameter (`rocket://?cores=3`) once
`driver.rs` parses params at all; today it ignores them
(`driver.rs:91`). N > core count is allowed on purpose: the notes' knee is
one above the core count. **N=1 must reproduce today's numbers** -- it is
the regression gate for every milestone.

### 5.2 `NpuContext`: everything that is per-file

```rust
struct NpuContext {
    id: ContextId,            // 0..N, 0 is the allocator's file
    file: std::fs::File,      // its own open(), never a try_clone
    scratch: ScratchPool,     // input/bias/output scratch BOs, by size class
    regcmd: RegcmdPool,       // the per-task program BOs queue_execute allocates today
    weights: WeightCacheSlice,// packings on this file; see 5.5
    last_job_done: Instant,   // for the quiescence rule, 5.6
}
```

Context 0 wraps the file the device already opens; IREE buffers stay there
untouched. Contexts 1..N are fresh `open()`s validated with
`is_rocket_device`. Each context is owned by exactly one worker thread; no
context is ever used from two threads, which is what makes the per-file
in-order queue a meaningful unit.

**The rule, enforced by type:** a `ContextBo { ctx: ContextId, handle, dma_address, .. }`
and a `submit(ctx: &NpuContext, ...)` that takes only `ContextBo`s of the
same `ctx` (debug-asserted). The four direct sites in §3 become
`stage_binding(ctx, &ref)` which returns the IREE buffer itself when
`ctx.id == 0` and a copy into `ctx.scratch` otherwise -- so with N=1 no byte
moves that does not move today.

### 5.3 Worker pool replaces thread-per-op

- `N` OS threads created at device create, each owning one `NpuContext`,
  each pinned for its lifetime to a distinct CPU from the highest-
  `cpu_capacity` set (`cpu_affinity::preferred()`, honouring
  `ROCKET_HOST_CPUS`). Pinning once beats `prefer_fast_cpus()` per
  execute: the worker never wakes on an A55 after `PREP_BO`.
- `queue_execute` no longer runs work on the caller's thread and no longer
  spawns. It builds `Unit`s (§5.4), records their semaphore lists, and
  enqueues. A unit becomes runnable when its wait semaphores are satisfied;
  the pool has one dispatcher thread blocking on the union of pending waits
  (`iree_hal_semaphore_wait` on each, in a small select loop) that moves
  units to the ready queue. `queue_alloca`/`dealloca`/`fill`/`update`/
  `copy`/`host_call` keep `run_after_wait` -- they are host memcpys and are
  not the problem.
- Placement policy is a trait, `Placement`, with two implementations at
  first: `RoundRobin` and `LeastQueued` (pick the context with the shortest
  ready queue). Kept as a trait so a kernel-side placement (§7) slots in.
- Completion: the worker signals the unit's `signal_semaphore_list` after
  its last host phase (compaction) finishes. Because the hardware never
  touches an IREE buffer, **the host is the only fence IREE ever sees**, and
  no cross-file synchronisation object is needed anywhere. That is the
  payoff of §3.

### 5.4 Units of work: two levels

**Level 1 -- dispatch-level (milestone M1).** A `Unit` is one recorded
dispatch (all of its CBUF row tasks), run start to finish on one context.
Parallelism comes from dispatches whose semaphores are independently
satisfied: separate command buffers (every NPU dispatch is its own command
buffer under the two-device setup) and, within one command buffer, groups
separated by recorded `execution_barrier`s. Today the driver records
barriers as no-ops and replays in call order; M1 honours them as
group boundaries (dispatches between two barriers are independent, per
IREE's own semantics). Chained CNN dispatches stay serial at this level;
ViT's Q/K/V projections and the residual branches fan out.

**Level 2 -- task-level (milestone M2).** A multi-task dispatch (CBUF
height split, `regcmd_tasks.len() > 1`) is split across contexts: task *t*
gets its own output scratch (its row band), a packed input band (its rows
plus the kernel halo), the weights and bias on that context, and its own
regcmd BO. Compaction gathers from up to N scratches into the dense IREE
buffer -- it already walks output rows, so this is a change of source
pointer per band, not a new pass. The tasks write disjoint output rows and
each reloads its weights (the mainline driver's inter-task IRQ transition
is unreliable on RK3588, which is why they are already separate jobs), so
nothing about the hardware programs changes; only who submits them.

Level 2 is where a chained CNN benefits at all, and it is also where host
packing parallelises: band packing per worker is the same total bytes as
today's whole-tensor pack, done on W threads.

The record/execute split constrains both levels: `dispatch_impl` allocates
scratch on `cb.fd` and bakes its address at record time. Two ways out:

- **(a) Assign at record time.** `RocketCommandBuffer` gets a context
  cursor; each dispatch (M1) or task (M2) takes its scratch from the
  context it is assigned to, and the address baked is that context's. The
  unit is then bound to that context for the command buffer's life.
  Cheapest to build: `cb.fd` becomes `cb.ctx(i).file`. Load balance is
  blind (a re-executed command buffer always lands the same way), and a
  command buffer recorded once and replayed cannot rebalance.
- **(b) Assign at execute time with regcmd relocation.** Builders emit,
  next to the program, a relocation list `(task, word index, operand,
  offset)` for every address-carrying register (CNA feature/weight/bias
  addresses, DPU dst, DPU-RDMA src, PPU src/dst). Scratch is taken from the
  chosen context's pool at execute time and the words are patched before
  the program is copied into the regcmd BO. Better balance, and it makes
  scratch pooling per context natural (today every dispatch allocates and
  frees its scratch).

Recommendation: **(a) for M1, (b) for M2** -- M2 needs per-band scratch
anyway, and by then the relocation list is the smaller change.

### 5.5 Weight cache per context

`weight_cache` keys on `(iree_hal_buffer_t*, generation, Geometry)` and
holds one packing BO on file 0. Under N contexts the key gains
`ContextId`. A miss on context k with a hit on context 0 is filled by
`memcpy` from the file-0 packing (9.8 GB/s, `examples/gem_bandwidth`)
rather than by re-running the pack. `ROCKET_WEIGHT_CACHE_MB` stays the
total budget; the per-context copies count against it. Cold-start cost is
N x today's for the weights that get fanned out, warm cost zero. Level 1
only copies weights for dispatches placed off context 0; level 2 copies
every multi-task dispatch's weights to every context it spans.

### 5.6 The hazards that change shape under concurrency

- **Depthwise -> dense DPU write-back quiescence** (`DEPTHWISE_TO_DENSE_QUIESCENCE`,
  1 ms, hardware-validated on one core). The hazard is per-core DPU state,
  and userspace cannot see which core a context landed on. Rule: keep a
  device-global `last_depthwise_completed: Instant`; any dense submit on
  any context dwells until 1 ms has passed since it. Conservative, cheap,
  and it degrades to exactly today's behaviour at N=1. It does *not* cover
  a depthwise job still *in flight* on core A while a dense job starts on
  core B -- the hazard was write-back after the fence, on the same DPU, so
  different cores should be safe, but that is an assumption. M1 ships with
  `dpu_mode_multicore_hw.rs`: three contexts interleaving depthwise and
  dense jobs at the shapes that originally failed, gated bit-exactly.
- **Hung-job floor and the wedge protocol** (memory
  `npu-wedges-after-failed-job`). `rocket_reset` is per core; one wedged
  job takes down one core's queue, and the DRM scheduler will migrate an
  idle entity elsewhere afterwards. The clock floor stays per job, per
  worker. Diagnostics (`ROCKET_DISPATCH_TIMES`, the profiler) gain a
  `ctx=` column. Sweep-style tests keep one shape per process; the
  order-dependence that note records is more exposed, not less, with three
  queues.
- **Runtime PM.** Each core autosuspends independently; the first job on a
  cold core pays the resume. Nothing to design, but the microbenchmark in
  §8 must warm all N contexts before timing.
- **IRQ affinity** stays an admin setting: the notes' recipe for a pool is
  irqs 69/70/71 (planck: 82/83/84) to cpu5/6/7 with the workers on the same
  three. Document, don't automate.

### 5.7 Profile

`profile.rs` gains a `Queue` phase (unit ready -> worker picked it up) and
a `ctx` dimension on `Wait`; the exit tables print per-context `wait`
totals and the fraction of wall during which >= 2 jobs were in flight
("overlap"). Overlap is the number that says whether N > 1 did anything.

## 6. Milestones

| | Scope | Gate |
|---|---|---|
| **M0** (done, §10) | `NpuContext` + worker pool at N=1. `queue_execute` enqueues instead of running/spawning; `run_after_wait` stays for the host-memcpy ops. No second file yet. | MobileNetV2 fp16/int8/requant and ViT within noise of today's numbers on planck, `taskset -c 4-7` and `0-7`. This is the regression gate for everything after it. |
| **M1** (built, §11) | N contexts, dispatch-level placement (5.4 level 1, assignment (a)), barrier groups honoured, `stage_binding` copy hop at the four direct sites, per-context weight cache, global quiescence rule, `dpu_mode_multicore_hw`. | Bit-exact against N=1 on every e2e model; `overlap` > 0 on ViT. |
| **M2** (built, §12) | Task-level fan-out (5.4 level 2) with regcmd relocation (b), band packing, gather compaction, per-context scratch pools. | Bit-exact; requant MobileNetV2 faster than N=1 at N=3 on a full machine; `layout_bench`-style microbench shows pack/compact scaling. |
| **M3** (optional) | Kernel-side placement, §7. | -- |

## 7. The alternative: a 30-line kernel patch

planck already runs an out-of-tree `rocket.ko` (memory
`planck-measurement-environment`: `/lib/modules/7.1.0-edge-rockchip64/updates/rocket.ko`,
built from `../linux/drivers/accel/rocket` plus the PPU_DONE patch). If a
local uAPI extension is acceptable, the per-file memory constraint
disappears:

- `rocket_job_open`: create `num_cores` entities per file, entity *i*
  bound to scheduler *i* only (plus keep the existing floating entity as
  index `-1`).
- `rocket_ioctl_submit`: accept `drm_rocket_submit.reserved` as
  `core_hint + 1` (0 keeps today's meaning) and pick the entity.
- Detectable at runtime: stock kernels reject `reserved != 0` with
  `-EINVAL`, so the driver probes once and falls back to §5.

With that, one file holds all memory, tasks of one dispatch write disjoint
rows of *one* output scratch, no weight copies, no `stage_binding`, and the
depthwise quiescence rule becomes per-core again. The userspace design
above keeps `Placement` as a trait precisely so this becomes a third
implementation rather than a rewrite. It is not the recommended first step
only because it forks the uAPI; it is the better end state.

## 8. Measurement plan (before M1 is judged) -- steps 1 and 2 done, see §9

1. `iree-rocket-hal/examples/multicore_bench`: one multi-task conv shape
   (e.g. the requant model's `112x112x24->144 k1x1`) and one ViT matmul,
   run as independent jobs on N=1..4 contexts, warm, reporting jobs/s and
   per-context `wait`. Expect the notes' 1 / 2.1 / 2.9 / 3.1x on the
   hardware term. If it is not there, nothing above is worth building.
2. Same shapes with the host phases included (pack -> submit -> wait ->
   compact per worker): measures overlap, the term §4 says is the real
   lever.
3. E2E at N=1 vs N=3 on the three models, six interleaved passes, each arm
   first in half of them (the board drifts), governor `performance`,
   `taskset` stated. Quote `overlap` next to wall.

## 9. Measured: §8 steps 1 and 2 (planck, 2026-09-06)

`iree-rocket-hal/examples/multicore_bench` exists now. It opens the device
N times, gives each file its own copy of one dispatch's BOs, and has N
threads (pinned to cpu4-7) replay production's per-tile `submit` +
`prep_bo` chain. Every context is verified bit-exactly after warm-up *and*
after the timed concurrent run. Board state: cpu4-7 governor
`performance`, cpu0-3 `ondemand`, NPU IRQs 82-84 on cpu6, nothing else
running. 200 jobs per context, 3 passes; the ranges below span the passes.
Both shapes are 3 CBUF tiles per job.

**Step 1, hardware term** (submit + wait only), jobs/s and ratio to N=1:

| Shape | N=1 | N=2 | N=3 | N=4 | N=5 | N=6 |
|---|---|---|---|---|---|---|
| conv 112x112x24->144 k1 fp16 | 316-372 | 617-748 (1.95-2.03x) | 924-1135 (2.5-3.0x) | 1075-1218 (2.9-3.5x) | ~1160 | ~1210 |
| fc 197x768 @ 768x768 fp16 | 549-577 | 1018-1093 (1.8-2.0x) | 1530-1678 (2.8-3.1x) | 1518-1650 (2.7-3.0x) | -- | -- |

The go/no-go of §8 is **go**: three files reach ~3x on both shapes, the
notes' 1 / 2.1 / 2.9 / 3.1 reproduced. `overlap` reads 98-100% for every
N >= 2. The conv keeps climbing past three cores (a fourth queue is worth
+10-15%, five and six a few percent more): with a per-tile blocking wait,
each queue leaves a submit -> IRQ -> wake -> submit bubble on its core that
another queue's job fills. The fc, whose tiles are longer, gains nothing
from a fourth queue and its per-context `wait` rises instead.

**Step 2, host phases included** (pack -> submit -> wait -> compact per
worker per job):

| Shape | N=1 | N=2 | N=3 | N=4 | pack+compact / job | overlap |
|---|---|---|---|---|---|---|
| conv | 137-167 | 236-251 (1.5-1.8x) | 324-342 (1.9-2.5x) | 404-412 (2.5-3.0x) | 2.9-3.6 ms at N=1 -> 5.5 ms at N=4 | 35-58% |
| fc | 308-318 | 615-637 (1.95-2.04x) | 936-971 (2.9-3.15x) | 1232-1246 (3.9-4.0x) | 1.03-1.13 ms, flat | 47 / 67 / 96% |

The fc is the clean case of §4's lever: at N=4 the host pack/compact runs
on the fourth A76 while three jobs are on three cores, and the pipeline
delivers **4.0x on 3 cores**. The conv shows the limit §4 named: its output
compaction moves 3.5 MB per job, and four workers compacting at once push
the per-job host cost from ~3 ms to 5.5 ms -- DRAM saturates before the
NPU does. Per-band packing (level 2) does not change the bytes, so on
output-heavy shapes the host term, not the NPU term, will bound N.

**Control, `--shared-fd`** (N dups of one `open()`, otherwise identical):

| Shape | N=2 | N=3 | N=4 | per-context wait |
|---|---|---|---|---|
| conv | 1.30x | 1.27x | 1.27x | 5.0 / 7.8 / 10.4 ms -- one queue, N deep |
| fc | 1.00-1.04x | 0.96-0.98x | 1.01-1.02x | 3.5 / 5.6 / 7.1 ms |

§2's constraint holds exactly: a dup is one scheduler entity, and N
threads on it get one core, however many jobs are queued. The conv's flat
1.3x is the queue-depth effect on that one core -- a non-empty entity
queue removes the per-tile bubble -- and it is a lever **independent of
multicore**: submitting a dispatch's tiles back to back and waiting once
(they write disjoint rows) would buy it at N=1 today. Note it in ISSUES.md
rather than folding it into M0.

**Consequences for the milestones.** M1 as specified is justified by the
hardware term alone. The fc numbers say `--host-phases` overlap is real
and reaches the full core count without level 2; the conv numbers say
level 2's band packing is worth less than §5.4 hoped on output-heavy
shapes, because the bytes are the bound. Re-check §4's "1.15-1.3x" target
against these before M2 is scoped.

## 10. M0 as built (2026-09-07)

`rocket-hal-driver/src/pool.rs`. What changed, against §1:

- **`queue_execute` enqueues.** It retains the command buffer, wraps
  today's body in a `Unit` and hands it to the pool; the caller's thread
  returns at once. The unit's wait semaphores are waited on by the worker,
  not the caller, and its signal semaphores are signalled (or failed with
  the status) by the worker when the last host phase is done. `run_after_wait`
  and its synchronous fast path survive unchanged for `queue_alloca`/
  `dealloca`/`fill`/`update`/`copy`/`host_call`.
- **One `NpuContext`, one worker.** Context 0 is a `try_clone()` of the
  device's file, so every IREE buffer handle is valid on it and no copy hop
  exists yet. The worker thread is named `rocket-npu-0`, asks for the big
  cluster once (`cpu_affinity::prefer_fast_cpus`) and keeps that for its
  life, replacing the per-`queue_execute` guard.
- **In-order per worker.** Units run in enqueue order, each waiting on its
  own semaphores first. IREE never makes a submission wait on a later
  submission to the same queue (an in-order hardware queue would deadlock
  too), so with one placement this is today's semantics exactly, minus the
  thread per dispatch. With N workers and units only ever waiting on
  earlier submissions, the globally earliest unfinished unit always has
  its dependencies complete or running, so the same argument carries to M1.
- **Errors route through the signal list.** Before, a `queue_execute`
  whose waits were already satisfied returned its failure synchronously.
  Now every failure (a hung-job refusal, a `SUBMIT` error) fails the
  signal semaphores, which is how any asynchronous queue reports one. A
  unit with no signal semaphore that fails prints a line instead, since
  nothing else can carry it.
- **`ROCKET_NPU_CORES`** is parsed (`N` or `auto`), and anything but 1 is
  refused with a message until M1. `Placement` is a trait with
  `RoundRobin` as its first implementation.
- **`device::destroy` drains the pool** before the profiler prints, so a
  unit still in flight completes and is counted. `profile` gains a `queue`
  row: enqueue to pickup, overlapping `outside`, excluded from wall.
- **The DPU-mode mutex stays**, device-global, as §5.6 wants.

Correctness: `iree-run-module` output bit-identical to the pre-M0 binary
on `mnv2.fp16`, `mnv2.int8`, `mnv2.static-int8` and `vit.npu.caps3584`
(max|diff| 0.0, same argmax), `taskset -c 4-7`.

Performance gate (`tools/bench/binab.sh`: pre-M0 vs M0
`iree-benchmark-module`, same model files, 4 interleaved passes with the
binary order alternating, `--benchmark_min_time=3s`, governor
`performance` on cpu4-7, NPU IRQs on cpu6, every NPU core `suspended`
before each run). Medians of real_time, ms:

| Arm | cpus | pre-M0 | M0 | M0 / pre-M0 |
|---|---|---|---|---|
| mnv2.fp16 | 4-7 | 167.5 | 160.0 | 0.955 |
| mnv2.int8 (accumulator) | 4-7 | 344.0 | 333.5 | 0.969 |
| mnv2.static-int8 (requant) | 4-7 | 305.5 | 297.5 | 0.974 |
| vit.npu.caps3584 | 4-7 | 1205.0 | 1198.0 | 0.994 |
| mnv2.fp16 | 0-7 | 156.5 | 150.0 | 0.958 |
| mnv2.int8 (accumulator) | 0-7 | 290.0 | 285.0 | 0.983 |
| mnv2.static-int8 (requant) | 0-7 | 255.0 | 250.0 | 0.980 |
| vit.npu.caps3584 | 0-7 | 875.0 | 870.5 | 0.995 |

Zero hangs in 64 runs. Every arm is at or slightly under the pre-M0
number, so the gate ("within noise") holds with a little to spare: what
the pool removes is a thread spawn plus two affinity syscalls per
dispatch and the wake of a fresh thread onto whichever CPU the scheduler
liked, and MobileNetV2's ~50 dispatches per inference make that 2-4 %.

HAL CTS on the board (`rocket_{queue,command_buffer,core,buffer,file}_tests`):
the command-buffer, core, buffer and file suites pass; the queue suite has
the same eight failures before and after M0 (seven `QueueAllocaTest`
explicit-pool cases and `QueueHostCallTest.AsyncCallback`, whose
`IREE_STATUS_DEFERRED` contract `queue_host_call` has never implemented).
Not M0's, and not fixed here.

The three e2e gates (`tools/e2e_{conv,matmul,pooling}_regression.py
--board planck`) pass on the M0 runtime, including every mixed
two-dispatches-in-one-command-buffer case.

## 11. M1 as built (2026-09-07)

What landed, against §5:

- **N contexts.** `ROCKET_NPU_CORES=N|auto` works. Context 0 is a dup of
  the device file; contexts 1..N are fresh `open()`s checked with
  `is_rocket_device`. `NpuContext` is an `Arc` shared between its worker
  and the command buffers placed on it, since scratch is allocated on the
  context's file at record time.
- **Placement at command-buffer creation** (§5.4 assignment (a)):
  `create_command_buffer` takes `pool.place()` (round robin) and the
  command buffer carries its context for its life; `queue_execute` sends
  the unit to that context's worker. Workers with company pin to distinct
  big cores (`cpu_affinity::pin_worker`); a lone worker keeps the cluster,
  which is what M0 was gated with.
- **The copy hop, `stage_direct`.** The four sites where the hardware read
  an IREE buffer directly (conv ARGB input, unpacked coefficients, unpacked
  bias, matmul bias) hand the binding through on context 0 and record a
  `StagedCopy` into the context's own scratch anywhere else. `apply_ops`
  applies the copies (`stage` profile phase) before the input sync. No
  handle list ever names a foreign BO, which the kernel would reject.
- **Weight cache per context.** `weight_cache::Key` gains `context`. A
  miss on context k first looks for the same packing on any other context
  (`lookup_other_context`) and, if found, records a byte copy of it and
  publishes the copy under k's key when it lands -- `memcpy` at ~10 GB/s
  instead of a re-pack. The profile's weight-cache line counts these as
  `copied across contexts`.
- **The hazard rule is time-based and device-global.** The old
  `last_dpu_mode` mutex was held across the whole submission and would
  have serialised every context. It is now a `last_depthwise_done`
  instant: a dense submit on any context waits until 1 ms after the most
  recent depthwise completion on any context, computed under the lock and
  slept outside it. At N=1 this is the old dwell minus whatever compaction
  already spent of it.
- **`dpu_mode_multicore_hw`** (new hw test) settles §5.6's assumption:
  depthwise on one file and dense on two others, 60 jobs each, four in
  flight per queue so the entities stay on their cores, no dwell anywhere
  -- exact, 0 wrong. The same-queue alternation control without dwell is
  also 0 wrong of 120 at these shapes, which says nothing about the
  original intermittent failure but is recorded.
- **Barrier groups** are not split: with the two-device partitioning every
  NPU command buffer this repo produces holds exactly one dispatch (49
  units for ViT's 48 sites), so there is nothing inside a command buffer
  to fan out. The hook is where `queue_execute` builds its unit.
- **Diagnostics.** `ROCKET_DISPATCH_TIMES` lines carry `ctx=`; the profile
  prints `wait.npu by context` and `overlap` (share of the span with two
  or more waits in flight) whenever more than one context was used, and
  the three e2e scripts take `--board-env KEY=VALUE` so a driver knob
  reaches the board.

**The finding that matters.** ViT at `ROCKET_NPU_CORES=3`, `ROCKET_PROFILE=1`:

```text
wait.npu by context: ctx0 127.6 ms, ctx1 125.2 ms, ctx2 124.6 ms
overlap: 0.0% of the span between the first and last wait had >= 2 jobs in flight
```

Placement works -- the hardware time splits evenly three ways -- and
nothing overlaps, because IREE orders command buffers on one device
timeline: each NPU command buffer waits on the semaphore the previous one
signals, whether or not the two are data-dependent. Concurrency in IREE's
model lives *inside* a command buffer, between barriers, and the
two-device partitioning gives this device one dispatch per command buffer.
§4's hope that ViT's Q/K/V projections would fan out at this level was
wrong for the program as compiled. So M1 is correct and buys nothing on
the real models by itself; what it buys is the machinery M2 needs. The
lever is task-level fan-out (§5.4 level 2): every ViT matmul and most
MobileNetV2 convolutions are 3-12 CBUF tiles, and those are independent
jobs today, just submitted one after another on one context.

Gate (`tools/bench/binab.sh`, one binary, `A_ENV=ROCKET_NPU_CORES=1`
against `B_ENV=ROCKET_NPU_CORES=3`, 4 interleaved passes, medians of
real_time in ms):

| Arm | cpus | N=1 | N=3 | N=3 / N=1 |
|---|---|---|---|---|
| mnv2.fp16 | 4-7 | 166.5 | 175.0 | 1.051 |
| mnv2.int8 (accumulator) | 4-7 | 317.0 | 314.5 | 0.992 |
| mnv2.static-int8 (requant) | 4-7 | 289.0 | 295.0 | 1.021 |
| vit.npu.caps3584 | 4-7 | 1190.5 | 1186.0 | 0.996 |
| mnv2.fp16 | 0-7 | 138.5 | 142.0 | 1.025 |
| mnv2.int8 (accumulator) | 0-7 | 269.0 | 274.0 | 1.019 |
| mnv2.static-int8 (requant) | 0-7 | 242.5 | 242.0 | 0.998 |
| vit.npu.caps3584 | 0-7 | 866.5 | 864.5 | 0.998 |

Zero hangs in 64 runs; output bit-identical to the pre-M0 binary at N=1,
3 and 4 on all four models; the conv, matmul and pooling e2e gates pass at
`--board-env ROCKET_NPU_CORES=3`. The 2-5 % the fp16 arm loses at N=3 is
the price of three pinned workers with nothing to overlap: the copy hop
for its ARGB stem input, the coefficient fan-out, and a worker that can
no longer drift to an idle core. It is the cost M2 has to earn back
before anything else.

## 12. M2 as built (2026-09-07)

Task-level fan-out, §5.4 level 2, in the simplest form that keeps every
tile program identical to the single-context one:

- **Replicas, not bands of scratch.** A multi-tile conv or matmul on a
  command buffer whose context has siblings gets a `Replica` on each of
  up to `tiles - 1` siblings: that context's own input, bias, weights and
  output buffers. Tile `t` runs on context `t mod K` and its program is
  the unrelocated plan program bound to that context's four addresses
  (`relocate` / the new `relocate_staged_accumulator`), so one plan serves
  every context and the hardware sees exactly the programs it saw before.
- **Copies at execute time**, after the home packing has produced the
  bytes: only the input rows (plus halo) the replica's tiles read, per
  NC1HWC2 plane (`InputBand`, `BandGeometry`); the bias; and the packed
  coefficients through the per-context `weight_cache` (a hit on the
  sibling, else a copy of the home packing published there).
- **All tasks submitted before any is waited for**, each on its own file;
  waits in order; the `dispatch` time and hung-job floor are per dispatch.
  On one context this is the queue-depth effect of §9 for free.
- **Gather compaction**: each tile's rectangle is compacted from the
  scratch of the context that ran it (`compact_atomic_output_rect`, or the
  tile's own entry of the staged accumulator layout).
- **Per-context scratch pool** (`scratch_pool.rs`): every driver-private
  GEM buffer -- input, bias, output, regcmd, replicas -- comes from a free
  list keyed on (file, size class) instead of a fresh `CREATE_BO` + `mmap`
  + first-touch faults per dispatch. `ROCKET_FANOUT=0`,
  `ROCKET_SCRATCH_POOL=0`, `ROCKET_PIN_WORKERS=0` switch each piece off.

Bit-identical to the pre-M0 binary at N=1, 3 and 4 on all four models,
with and without fan-out.

**What the profiles said on the way** (`iree-benchmark-module`, 5 s,
`taskset -c 4-7`, per-call averages):

- The first cut, replicas allocated fresh per dispatch, cost ViT ~3 ms of
  `record` and ~0.7 ms of `stage` per fanned-out dispatch, and the 256 MiB
  weight budget refused a third of ViT's coefficients once three contexts
  each needed a copy. The hardware term still fell 8.1 -> 3.4 ms per
  dispatch; the host cost ate it. Hence the pool and a budget that scales
  with the context count.
- **The pool alone is the biggest single win of the whole series at one
  context**: ViT 1203 -> 990-1098 ms, MobileNetV2 requant 290 -> 255 ms,
  MobileNetV2 fp16 166 -> 146 ms. `record` fell from 1.4 to 0.22 ms per
  ViT dispatch. This is ISSUES.md P3's "regcmd BO allocated and mapped per
  tile", paid by every scratch buffer, and it was never about multicore.
- With the pool, ViT at N=3 is 922 against 1098 in the same session
  (`wait.npu` 8.4 -> 3.4 ms per dispatch), MobileNetV2 requant 252 against
  255, MobileNetV2 fp16 155 against 146. What the fan-out still pays per
  dispatch: `stage` 0.47 ms (ViT) -- not the copy, which the bands cut to
  a third, but `fini_bo` over the replica input BO's pages, ∝ BO size
  (P3 again); `record` +0.3 ms warm; and `outside` +0.7 ms, IREE's own CPU
  dispatches running slower next to two more cores of DMA and two more
  pinned workers. MobileNetV2 fp16's dispatches average 0.75 ms of
  hardware time, so three files' worth of submit and wait round trips
  cost more than the split saves; §4 predicted exactly this.
- Pinning workers to distinct cores versus letting them float over the
  cluster is within noise either way (924 vs 908 ms on ViT).

**Gate** (`tools/bench/binab.sh`, 3 interleaved passes, medians of
real_time in ms, governor `performance`, NPU IRQs on cpu6, every core
`suspended` before each run). First, what M2 does at one context -- the
scratch pool and batched tile submission, against the M1 binary:

| Arm | cpus | M1 N=1 | M2 N=1 | M2 / M1 |
|---|---|---|---|---|
| mnv2.fp16 | 4-7 | 166.0 | 144.0 | 0.867 |
| mnv2.int8 (accumulator) | 4-7 | 319.0 | 283.0 | 0.887 |
| mnv2.static-int8 (requant) | 4-7 | 290.0 | 259.0 | 0.893 |
| vit.npu.caps3584 | 4-7 | 1225.0 | 1150.0 | 0.939 |
| mnv2.fp16 | 0-7 | 140.0 | 124.0 | 0.886 |
| mnv2.int8 (accumulator) | 0-7 | 271.0 | 243.0 | 0.897 |
| mnv2.static-int8 (requant) | 0-7 | 240.0 | 218.0 | 0.908 |
| vit.npu.caps3584 | 0-7 | 867.0 | 834.0 | 0.962 |

Then the fan-out, the M2 binary at `ROCKET_NPU_CORES=1` against `3`:

| Arm | cpus | N=1 | N=3 | N=3 / N=1 |
|---|---|---|---|---|
| mnv2.fp16 | 4-7 | 144.0 | 145.0 | 1.007 |
| mnv2.int8 (accumulator) | 4-7 | 285.0 | 289.0 | 1.014 |
| mnv2.static-int8 (requant) | 4-7 | 256.0 | 258.0 | 1.008 |
| vit.npu.caps3584 | 4-7 | 1134.0 | 930.0 | **0.820** |
| mnv2.fp16 | 0-7 | 123.0 | 120.0 | 0.976 |
| mnv2.int8 (accumulator) | 0-7 | 241.0 | 241.0 | 1.000 |
| mnv2.static-int8 (requant) | 0-7 | 219.0 | 212.0 | 0.968 |
| vit.npu.caps3584 | 0-7 | 833.0 | 620.0 | **0.744** |

Zero hangs in 96 runs; the conv, matmul and pooling e2e gates pass at
`--board-env ROCKET_NPU_CORES=3`. Against the pre-M0 driver, ViT on eight
cores is 875 -> 620 ms (1.41x) and MobileNetV2 fp16 156 -> 120 ms
(1.30x), the latter entirely from the pool and batching.

**Where this leaves the milestones.** M2's gate asked for the requant
MobileNetV2 to be faster at N=3 than at N=1 on a full machine: it is, by
3 %, which is noise-adjacent and honest. The model where fan-out pays is
the one whose dispatches carry several milliseconds of hardware time each;
MobileNetV2's carry under one, and three files' worth of round trips is
the price of splitting them. The default stays `ROCKET_NPU_CORES=1`; a
ViT-shaped model should run with `3`. The next levers, in order:
`fini_bo` over whole BOs (P3's first half, now the dominant replica cost),
`outside` growing 0.7 ms per dispatch at N=3 (IREE's CPU dispatches next
to two more cores of DMA), and §7's kernel-side placement, which would
remove the replica copies altogether.
