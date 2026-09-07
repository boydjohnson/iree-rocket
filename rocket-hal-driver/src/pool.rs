//! The NPU worker pool: `queue_execute` enqueues, a worker runs.
//!
//! MULTICORE.md §5.2-5.3, milestone M0. Before this, `queue_execute` ran a
//! command buffer on whichever thread called it (or on a detached thread
//! spawned per call when its wait semaphores were still pending), and every
//! such thread then serialised on the device-wide DPU-mode mutex. Now every
//! NPU command buffer becomes a [`Unit`] handed to one of N long-lived
//! worker threads, each owning one [`NpuContext`] -- one `open()` of
//! `/dev/accel/accel0`, one DRM scheduler entity, one in-order queue that
//! occupies at most one NPU core at a time.
//!
//! Context 0 is a dup of the file every IREE buffer lives on, so a job on it
//! may name any of them. Contexts `1..N` (M1) are fresh `open()`s: a job on
//! one may only name BOs created on it, so every command buffer recorded
//! against such a context stages the few bindings the hardware would read
//! directly into its own scratch (`command_buffer::stage_direct`) and packs
//! its coefficients on it (`weight_cache`, keyed per context). The host is
//! the only fence IREE ever sees, so nothing cross-file is needed.
//!
//! Placement happens at command-buffer creation (`device::create_command_
//! buffer`, MULTICORE.md §5.4 assignment (a)): the context chosen then is
//! the file the command buffer's scratch is allocated on and the worker its
//! unit goes to. `queue_execute` returns as soon as its unit is queued; the
//! worker does the semaphore wait, the host layout work and the hardware
//! submission on one thread, pinned once for its life.
//!
//! Ordering. A worker runs its units strictly in the order they were
//! enqueued, waiting on each unit's semaphores before starting it. That is
//! safe because IREE only ever makes a queue submission wait on semaphores
//! signalled by earlier submissions or by other queues/the host -- a
//! submission that waited on a later one on the same queue would deadlock
//! any in-order hardware queue too. With one placement, in-order per worker
//! is therefore exactly today's semantics minus the thread churn.
//!
//! ```text
//! ROCKET_NPU_CORES=N      worker contexts, 1..=8. Default 1. N above the
//!                         core count is allowed: a fourth queue on three
//!                         cores fills each core's submit bubble (§9).
//! ROCKET_NPU_CORES=auto   one per NPU core (counted under
//!                         /sys/bus/platform/devices/*.npu).
//! ```

use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    thread::JoinHandle,
    time::Instant,
};

use crate::{
    bindings::{iree_hal_semaphore_list_t, iree_status_t},
    device::{OwnedSemaphoreList, fail_all, signal_all, wait_all},
    status,
};

/// One DRM file and its identity: MULTICORE.md §5.2. Shared between the
/// worker that submits on it and the command buffers recorded against it
/// (which allocate their scratch on it at record time). Scratch and regcmd
/// pools arrive with M2.
pub struct NpuContext {
    /// `0` is the context whose file the device allocator also uses, so
    /// every IREE buffer is visible to it. Contexts `1..N` are fresh
    /// `open()`s and see only what was created on them.
    pub id: usize,
    pub file: std::fs::File,
}

/// How many contexts a pool may have: `ROCKET_NPU_CORES` is capped here and
/// the profiler's per-context columns are sized by it.
pub const MAX_CONTEXTS: usize = 8;

/// Which worker a unit goes to. A trait so a kernel-side core hint
/// (MULTICORE.md §7) slots in as a third implementation.
pub trait Placement: Send + Sync {
    fn place(&self, workers: usize) -> usize;
}

/// Units go to workers in turn. With one worker this is the only policy
/// there is.
pub struct RoundRobin {
    next: AtomicUsize,
}

impl RoundRobin {
    pub fn new() -> RoundRobin {
        RoundRobin {
            next: AtomicUsize::new(0),
        }
    }
}

impl Default for RoundRobin {
    fn default() -> Self {
        Self::new()
    }
}

impl Placement for RoundRobin {
    fn place(&self, workers: usize) -> usize {
        self.next.fetch_add(1, Ordering::Relaxed) % workers
    }
}

/// One queued piece of work: wait for `wait`, run `work` on the worker's
/// context, then signal (or fail) `signal`.
struct Unit {
    wait: OwnedSemaphoreList,
    signal: OwnedSemaphoreList,
    work: Box<dyn FnOnce(&NpuContext) -> iree_status_t + Send>,
    enqueued: Instant,
}

struct Worker {
    context: Arc<NpuContext>,
    sender: Mutex<Option<mpsc::Sender<Unit>>>,
    handle: Mutex<Option<JoinHandle<()>>>,
}

pub struct WorkerPool {
    workers: Vec<Worker>,
    placement: Box<dyn Placement>,
}

/// Reads `ROCKET_NPU_CORES`; see the module doc comment.
pub fn requested_contexts() -> usize {
    match std::env::var("ROCKET_NPU_CORES") {
        Ok(value) if value == "auto" => npu_core_count().clamp(1, MAX_CONTEXTS),
        Ok(value) => match value.parse::<usize>() {
            Ok(n) if (1..=MAX_CONTEXTS).contains(&n) => n,
            _ => {
                eprintln!(
                    "rocket: ROCKET_NPU_CORES={value:?} is not 1..={MAX_CONTEXTS} or `auto`; using 1"
                );
                1
            }
        },
        Err(_) => 1,
    }
}

/// The NPU cores the platform has, counted the only way mainline `rocket`
/// exposes: one `*.npu` platform device per core (3 on RK3588, 2 on RK3576,
/// 1 on RK3568). The kernel does not export `num_cores`.
pub fn npu_core_count() -> usize {
    std::fs::read_dir("/sys/bus/platform/devices")
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .filter(|entry| entry.file_name().to_string_lossy().ends_with(".npu"))
                .count()
        })
        .unwrap_or(0)
}

impl WorkerPool {
    /// Spawns one worker per context. Each worker pins itself once, for its
    /// whole life (`cpu_affinity::pin_worker`): to the big cluster when it
    /// is alone, to its own big core when it has company.
    pub fn new(contexts: Vec<Arc<NpuContext>>, placement: Box<dyn Placement>) -> WorkerPool {
        assert!(
            !contexts.is_empty() && contexts.len() <= MAX_CONTEXTS,
            "a worker pool needs 1..={MAX_CONTEXTS} contexts"
        );
        let count = contexts.len();
        let workers = contexts
            .into_iter()
            .enumerate()
            .map(|(index, context)| {
                let (sender, receiver) = mpsc::channel::<Unit>();
                let worker_context = Arc::clone(&context);
                let handle = std::thread::Builder::new()
                    .name(format!("rocket-npu-{}", context.id))
                    .spawn(move || worker_main(worker_context, index, count, receiver))
                    .expect("spawn rocket NPU worker");
                Worker {
                    context,
                    sender: Mutex::new(Some(sender)),
                    handle: Mutex::new(Some(handle)),
                }
            })
            .collect();
        WorkerPool { workers, placement }
    }

    pub fn contexts(&self) -> usize {
        self.workers.len()
    }

    pub fn context(&self, id: usize) -> &Arc<NpuContext> {
        &self.workers[id].context
    }

    /// The context the next command buffer should be recorded against.
    pub fn place(&self) -> Arc<NpuContext> {
        Arc::clone(&self.workers[self.placement.place(self.workers.len())].context)
    }

    /// Queues `work` behind everything already queued on context `context`'s
    /// worker -- the context the command buffer was recorded against, since
    /// its scratch lives on that file. Returns immediately; the outcome
    /// reaches IREE through `signal_list` (signalled on success, failed with
    /// the status on error).
    ///
    /// # Safety
    ///
    /// The semaphore lists must be valid for the duration of this call; they
    /// are copied and retained before it returns. `work` runs on another
    /// thread, so everything it captures must be safe to use there.
    pub unsafe fn enqueue_on(
        &self,
        context: usize,
        wait_list: iree_hal_semaphore_list_t,
        signal_list: iree_hal_semaphore_list_t,
        work: impl FnOnce(&NpuContext) -> iree_status_t + Send + 'static,
    ) -> iree_status_t {
        let unit = Unit {
            wait: unsafe { OwnedSemaphoreList::new(wait_list) },
            signal: unsafe { OwnedSemaphoreList::new(signal_list) },
            work: Box::new(work),
            enqueued: Instant::now(),
        };
        let index = context.min(self.workers.len() - 1);
        let sender = self.workers[index]
            .sender
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match sender.as_ref().map(|sender| sender.send(unit)) {
            Some(Ok(())) => status::ok(),
            // The worker is gone: the pool is shutting down, or the worker
            // panicked. Either way nothing will ever signal this unit.
            _ => status::from_code(crate::bindings::iree_status_code_e_IREE_STATUS_UNAVAILABLE),
        }
    }

    /// Lets every worker drain its queue and exit. Idempotent; `Drop` calls
    /// it too, but `device::destroy` wants the drain to happen before the
    /// profiler prints and before the device's other resources go away.
    pub fn shutdown(&self) {
        for worker in &self.workers {
            drop(
                worker
                    .sender
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .take(),
            );
        }
        for worker in &self.workers {
            let handle = worker
                .handle
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .take();
            if let Some(handle) = handle {
                let _ = handle.join();
            }
        }
    }
}

impl Drop for WorkerPool {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn worker_main(
    context: Arc<NpuContext>,
    index: usize,
    workers: usize,
    receiver: mpsc::Receiver<Unit>,
) {
    // Held for the thread's life: the worker never wakes from `PREP_BO`
    // onto an A55, and with company never onto a sibling's core
    // (`cpu_affinity`, and MULTICORE.md §5.3).
    let _pinned = crate::cpu_affinity::pin_worker(index, workers);
    while let Ok(mut unit) = receiver.recv() {
        crate::profile::record(
            crate::profile::Phase::Queue,
            crate::profile::NO_OP,
            unit.enqueued.elapsed(),
            0,
        );
        let st = unsafe { wait_all(unit.wait.as_list()) };
        if !st.is_null() {
            unsafe { fail_all(unit.signal.as_list(), st) };
            continue;
        }
        let st = (unit.work)(&context);
        if st.is_null() {
            unsafe { signal_all(unit.signal.as_list()) };
        } else if unit.signal.is_empty() {
            // Nothing to route the failure through; the caller has already
            // returned. Say so rather than lose it.
            // Statuses in this crate are bare codes (`status.rs`), so there
            // is nothing to free, only something to say.
            eprintln!(
                "rocket: NPU unit on context {} failed (status code {}) with no signal \
                 semaphore to report it on",
                context.id,
                st as usize & 0x1f
            );
        } else {
            unsafe { fail_all(unit.signal.as_list(), st) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_robin_cycles_through_workers() {
        let placement = RoundRobin::new();
        let placed: Vec<usize> = (0..7).map(|_| placement.place(3)).collect();
        assert_eq!(placed, vec![0, 1, 2, 0, 1, 2, 0]);
    }

    #[test]
    fn a_single_worker_is_always_chosen() {
        let placement = RoundRobin::new();
        assert!((0..5).all(|_| placement.place(1) == 0));
    }
}
