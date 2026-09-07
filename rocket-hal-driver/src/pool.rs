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
//! M0 ships with N fixed at 1: context 0 is a dup of the file every IREE
//! buffer lives on, so nothing about which BO a job may name changes. What
//! changes is the threading: a `queue_execute` returns as soon as its unit
//! is queued, the worker pins itself to the big cluster once instead of per
//! call, and the semaphore wait, the host layout work and the hardware
//! submission all happen on the same thread in submission order. M1 adds
//! contexts 1..N (their own `open()`s, their own scratch) and a placement
//! policy that spreads units across them.
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
//! ROCKET_NPU_CORES=N      worker contexts, 1..=8. M0 accepts only 1.
//! ROCKET_NPU_CORES=auto   one per NPU core (counted under
//!                         /sys/bus/platform/devices/*.npu); M0 clamps to 1.
//! ```

use std::{
    sync::{
        Mutex,
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

/// Everything a worker owns that is bound to one DRM file: MULTICORE.md
/// §5.2. Scratch pools, regcmd pools and the per-context weight cache slice
/// arrive with M1/M2; at M0 the context is the file and its identity.
pub struct NpuContext {
    /// `0` is the context whose file the device allocator also uses, so
    /// every IREE buffer is visible to it. Contexts `1..N` are fresh
    /// `open()`s and see only what was created on them.
    pub id: usize,
    /// Owned by this context's worker thread and used from no other.
    pub file: std::fs::File,
}

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
    sender: Mutex<Option<mpsc::Sender<Unit>>>,
    handle: Mutex<Option<JoinHandle<()>>>,
}

pub struct WorkerPool {
    workers: Vec<Worker>,
    placement: Box<dyn Placement>,
}

/// Reads `ROCKET_NPU_CORES`; see the module doc comment.
pub fn requested_contexts() -> usize {
    let requested = match std::env::var("ROCKET_NPU_CORES") {
        Ok(value) if value == "auto" => npu_core_count().max(1),
        Ok(value) => match value.parse::<usize>() {
            Ok(n) if (1..=8).contains(&n) => n,
            _ => {
                eprintln!("rocket: ROCKET_NPU_CORES={value:?} is not 1..=8 or `auto`; using 1");
                1
            }
        },
        Err(_) => 1,
    };
    if requested > 1 {
        // M1 is where contexts 1..N get their own files and scratch.
        eprintln!("rocket: ROCKET_NPU_CORES={requested} is not implemented yet (M0); using 1");
        return 1;
    }
    requested
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
    /// Spawns one worker per context. Each worker asks for the big cluster
    /// once, for its whole life (`cpu_affinity`), which is what
    /// `prefer_fast_cpus` per `queue_execute` used to approximate.
    pub fn new(contexts: Vec<NpuContext>, placement: Box<dyn Placement>) -> WorkerPool {
        assert!(
            !contexts.is_empty(),
            "a worker pool needs at least one context"
        );
        let workers = contexts
            .into_iter()
            .map(|context| {
                let (sender, receiver) = mpsc::channel::<Unit>();
                let handle = std::thread::Builder::new()
                    .name(format!("rocket-npu-{}", context.id))
                    .spawn(move || worker_main(context, receiver))
                    .expect("spawn rocket NPU worker");
                Worker {
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

    /// Queues `work` behind everything already queued on the chosen worker.
    /// Returns immediately; the outcome reaches IREE through `signal_list`
    /// (signalled on success, failed with the status on error).
    ///
    /// # Safety
    ///
    /// The semaphore lists must be valid for the duration of this call; they
    /// are copied and retained before it returns. `work` runs on another
    /// thread, so everything it captures must be safe to use there.
    pub unsafe fn enqueue(
        &self,
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
        let index = self.placement.place(self.workers.len());
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

fn worker_main(context: NpuContext, receiver: mpsc::Receiver<Unit>) {
    // Held for the thread's life: the worker never wakes from `PREP_BO`
    // onto an A55 (`cpu_affinity`, and MULTICORE.md §5.3).
    let _fast_cpus = crate::cpu_affinity::prefer_fast_cpus();
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
