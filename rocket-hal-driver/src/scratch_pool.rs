//! Reuses driver-private scratch GEM buffers across command buffers.
//!
//! Every dispatch allocates a handful of scratch BOs at record time -- the
//! packed input, the bias, the output the hardware writes, one regcmd BO per
//! tile, and with `ROCKET_NPU_CORES` > 1 a replica of each on every sibling
//! context -- and frees them when the command buffer is destroyed. Each
//! allocation is a `CREATE_BO` (page allocation plus an IOMMU map), an
//! `mmap`, and then a page fault per 4 KiB page the first time the host
//! touches it. Measured on planck (MULTICORE.md §12) that was ~3 ms of
//! `record` and ~0.5 ms of `stage` per fanned-out ViT dispatch, more than
//! the hardware time the fan-out saved.
//!
//! A [`ScratchBuffer`] is an `OwnedBuffer` that goes back to a free list
//! keyed on `(file, size class)` when dropped instead of being closed, so
//! the next dispatch of the same shape on the same context gets a mapped,
//! faulted-in buffer for the price of a mutex. Sizes round up to a class so
//! near-equal requests share buffers.
//!
//! Nothing here relies on a fresh buffer's zero fill: every packer writes
//! its padding explicitly, the hardware's output is read back only where a
//! tile wrote it, and a regcmd BO is read only to its task's word count.
//!
//! ```text
//! ROCKET_SCRATCH_POOL=0      allocate and free every buffer, as before
//! ROCKET_SCRATCH_POOL_MB=N   bytes the free lists may hold, default 256 per context
//! ```

use std::{
    collections::HashMap,
    mem::ManuallyDrop,
    ops::Deref,
    os::fd::{AsFd, RawFd},
    sync::Mutex,
};

use iree_rocket_hal::rocket::device::OwnedBuffer;

/// A pooled scratch GEM buffer. Dereferences to the `OwnedBuffer` it wraps;
/// `size` is the class size, at least what was asked for.
pub struct ScratchBuffer {
    buffer: ManuallyDrop<OwnedBuffer>,
    fd: RawFd,
    class: usize,
}

impl Deref for ScratchBuffer {
    type Target = OwnedBuffer;

    fn deref(&self) -> &OwnedBuffer {
        &self.buffer
    }
}

/// An idle GEM buffer on a free list. `OwnedBuffer` carries a raw host
/// pointer and so is `!Send`; an idle buffer is touched by no thread until
/// it is handed out again, so moving it between threads through the pool is
/// sound.
struct Idle(OwnedBuffer);

unsafe impl Send for Idle {}

#[derive(Default)]
struct Pool {
    free: HashMap<(RawFd, usize), Vec<Idle>>,
    bytes: usize,
    reused: u64,
    allocated: u64,
}

fn pool() -> &'static Mutex<Pool> {
    static POOL: std::sync::OnceLock<Mutex<Pool>> = std::sync::OnceLock::new();
    POOL.get_or_init(|| Mutex::new(Pool::default()))
}

fn enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ROCKET_SCRATCH_POOL").map_or(true, |value| value != "0"))
}

/// How many NPU contexts allocate scratch, from `device::create`: the live
/// set of a fanned-out model is that many times larger, and a free list
/// that cannot hold it hands out fresh allocations every inference.
pub fn set_contexts(contexts: usize) {
    CONTEXTS.store(contexts.max(1), std::sync::atomic::Ordering::Relaxed);
}

static CONTEXTS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(1);

fn budget_bytes() -> usize {
    static BUDGET: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *BUDGET.get_or_init(|| {
        std::env::var("ROCKET_SCRATCH_POOL_MB")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(256 * CONTEXTS.load(std::sync::atomic::Ordering::Relaxed))
            .saturating_mul(1024 * 1024)
    })
}

/// The class a request rounds up to: pages below 64 KiB (a regcmd BO is a
/// page or two, and `fini_bo` walks every page a BO has), 64 KiB steps
/// below 256 KiB, 256 KiB steps below 4 MiB, 1 MiB steps above. Waste is
/// bounded at a quarter and falls with size; reuse is what matters.
fn class_of(bytes: usize) -> usize {
    const KIB: usize = 1024;
    let step = if bytes < 64 * KIB {
        4 * KIB
    } else if bytes < 256 * KIB {
        64 * KIB
    } else if bytes < 4096 * KIB {
        256 * KIB
    } else {
        1024 * KIB
    };
    bytes.max(1).div_ceil(step) * step
}

impl ScratchBuffer {
    /// A buffer of at least `bytes` on `fd`: from the free list when one of
    /// the class is there, freshly allocated otherwise. Same contract as
    /// [`OwnedBuffer::new`].
    ///
    /// # Safety
    ///
    /// `fd` must be an open Rocket DRM file and `file` the same file.
    pub unsafe fn new(fd: RawFd, bytes: usize, file: impl AsFd) -> ScratchBuffer {
        let class = class_of(bytes);
        if enabled() {
            let mut pool = pool()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(Idle(buffer)) = pool.free.get_mut(&(fd, class)).and_then(Vec::pop) {
                pool.bytes = pool.bytes.saturating_sub(class);
                pool.reused += 1;
                return ScratchBuffer {
                    buffer: ManuallyDrop::new(buffer),
                    fd,
                    class,
                };
            }
            pool.allocated += 1;
        }
        ScratchBuffer {
            buffer: ManuallyDrop::new(unsafe { OwnedBuffer::new(fd, class, file) }),
            fd,
            class,
        }
    }
}

impl Drop for ScratchBuffer {
    fn drop(&mut self) {
        // SAFETY: `buffer` is never touched again after this take.
        let buffer = unsafe { ManuallyDrop::take(&mut self.buffer) };
        if enabled() {
            let mut pool = pool()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if pool.bytes + self.class <= budget_bytes() {
                pool.bytes += self.class;
                pool.free
                    .entry((self.fd, self.class))
                    .or_default()
                    .push(Idle(buffer));
                return;
            }
        }
        drop(buffer);
    }
}

/// Closes every pooled buffer, from `device::destroy` -- before the files
/// they were created on go away.
pub fn clear() {
    let mut pool = pool()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    pool.free.clear();
    pool.bytes = 0;
}

/// Reuses and fresh allocations so far, for the `ROCKET_PROFILE` report.
pub fn stats() -> (u64, u64) {
    let pool = pool()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    (pool.reused, pool.allocated)
}

#[cfg(test)]
mod tests {
    use super::class_of;

    #[test]
    fn classes_round_up_and_never_down() {
        assert_eq!(class_of(1), 4 * 1024);
        assert_eq!(class_of(4097), 8 * 1024);
        assert_eq!(class_of(64 * 1024), 64 * 1024);
        assert_eq!(class_of(64 * 1024 + 1), 128 * 1024);
        assert_eq!(class_of(300 * 1024), 512 * 1024);
        assert_eq!(class_of(5 * 1024 * 1024 + 1), 6 * 1024 * 1024);
        for bytes in [1, 4096, 65_537, 250_000, 1_000_000, 9_000_000] {
            assert!(class_of(bytes) >= bytes);
        }
    }
}
