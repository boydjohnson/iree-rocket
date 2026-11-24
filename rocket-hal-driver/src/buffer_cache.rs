//! Reuses the GEM objects behind IREE's own buffers.
//!
//! A compiled program allocates its transients through `queue_alloca` and
//! frees them through `queue_dealloca` on every invocation: ResNet50-224
//! int8 makes 85 allocations a run, ~20 MB. Each is a `CREATE_BO` (shmem
//! pages plus an IOMMU map), an `mmap`, a page fault per 4 KiB page on
//! first touch, and on the way out a `munmap` and a `GEM_CLOSE` whose
//! page release and DMA unmap cost ~0.8 us per KiB -- 2.4 ms for a 3 MB
//! tensor. Measured on the board that is ~15 ms of `CREATE_BO` and ~20 ms
//! of free per inference, on the host's critical path.
//!
//! [`BufferCache`] keeps a freed allocation's GEM object, mapping and
//! IOVA on a free list keyed on its page-rounded size, and hands it to the
//! next allocation of exactly that size. A static-shape program repeats
//! its allocation sizes every invocation, so after the first the whole
//! transient set is reused. Sizes are not rounded to classes as in
//! [`crate::scratch_pool`]: a cache sync (`fini_bo`) walks every page a
//! GEM object has, and IREE buffers are synced far more than scratch.
//!
//! Nothing a buffer held in its previous life is visible through IREE:
//! IREE makes no zero-fill promise for a device allocation (programs emit
//! their own fills), the packed-weight cache forgets a buffer in
//! `buffer::destroy`, constant-sync state lives on the new
//! `RocketBuffer`, and every NPU job's output is `prep_bo`-synced for the
//! CPU after the job whatever buffer it is. Stale CPU cache lines from a
//! previous life are therefore covered by the same syncs as within one.
//!
//! The cache belongs to one allocator and its DRM file. It is shared with
//! every buffer through an `Arc`, so a buffer freed after its allocator is
//! gone finds the cache closed and frees its GEM object directly, as
//! before.
//!
//! ```text
//! ROCKET_BUFFER_CACHE=0      allocate and free every IREE buffer, as before
//! ROCKET_BUFFER_CACHE_MB=N   bytes the free lists may hold, default 256
//! ```

use std::{collections::HashMap, os::fd::RawFd, sync::Mutex};

use iree_rocket_hal::rocket::device;

/// A free GEM allocation. `device::Buffer` carries a raw host pointer and
/// so is `!Send`; an idle one is touched by no thread until it is handed
/// out again, so moving it between threads through the cache is sound.
struct Idle(device::Buffer);

unsafe impl Send for Idle {}

#[derive(Default)]
struct Inner {
    free: HashMap<usize, Vec<Idle>>,
    bytes: usize,
    closed: bool,
}

pub struct BufferCache {
    fd: RawFd,
    budget: usize,
    inner: Mutex<Inner>,
}

fn budget_from_env() -> usize {
    if std::env::var("ROCKET_BUFFER_CACHE").is_ok_and(|value| value == "0") {
        return 0;
    }
    std::env::var("ROCKET_BUFFER_CACHE_MB")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(256)
        .saturating_mul(1024 * 1024)
}

/// The size a request is allocated at: whole pages, at least one. A GEM
/// object is whole pages anyway; mapping all of them lets `mapped_bytes`
/// and `munmap` describe the same range.
pub fn page_size_of(bytes: usize) -> usize {
    bytes.max(1).next_multiple_of(4096)
}

static REUSED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static ALLOCATED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Reuses and fresh allocations so far over every cache, for the
/// `ROCKET_PROFILE` report.
pub fn stats() -> (u64, u64) {
    (
        REUSED.load(std::sync::atomic::Ordering::Relaxed),
        ALLOCATED.load(std::sync::atomic::Ordering::Relaxed),
    )
}

impl BufferCache {
    /// A cache for allocations on `fd`, which must stay open until
    /// [`BufferCache::close`].
    pub fn new(fd: RawFd) -> Self {
        Self {
            fd,
            budget: budget_from_env(),
            inner: Mutex::new(Inner::default()),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// An allocation of [`page_size_of`]`(bytes)`: a freed one of exactly
    /// that size when there is one, a new GEM object otherwise.
    ///
    /// # Safety
    ///
    /// `file` must be the open Rocket DRM file this cache was made for.
    pub unsafe fn take(&self, bytes: usize, file: impl std::os::fd::AsFd) -> device::Buffer {
        let size = page_size_of(bytes);
        if self.budget > 0 {
            let mut inner = self.lock();
            if let Some(Idle(buffer)) = inner.free.get_mut(&size).and_then(Vec::pop) {
                inner.bytes -= size;
                REUSED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return buffer;
            }
        }
        ALLOCATED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        unsafe { device::Buffer::new(self.fd, size, file) }
    }

    /// Returns an allocation [`BufferCache::take`] handed out: onto the
    /// free list while the cache is open and under budget, freed otherwise.
    ///
    /// # Safety
    ///
    /// `buffer` must have come from this cache's `take`, and nothing may
    /// use its mapping or handle afterwards.
    pub unsafe fn give(&self, buffer: device::Buffer) {
        {
            let mut inner = self.lock();
            if !inner.closed && inner.bytes + buffer.size <= self.budget {
                inner.bytes += buffer.size;
                inner
                    .free
                    .entry(buffer.size)
                    .or_default()
                    .push(Idle(buffer));
                return;
            }
        }
        unsafe { release(self.fd, &buffer) };
    }

    /// Frees every idle allocation and stops keeping new ones, from the
    /// allocator's `destroy` while its file is still open.
    pub fn close(&self) {
        let idle = {
            let mut inner = self.lock();
            inner.closed = true;
            inner.bytes = 0;
            std::mem::take(&mut inner.free)
        };
        for Idle(buffer) in idle.into_values().flatten() {
            unsafe { release(self.fd, &buffer) };
        }
    }
}

/// Drops both references to a GEM object: the mapping and the handle.
unsafe fn release(fd: RawFd, buffer: &device::Buffer) {
    let _ = unsafe { device::unmap_bo(buffer) };
    let _ = unsafe { device::close_bo(fd, buffer.handle) };
}

#[cfg(test)]
mod tests {
    use super::page_size_of;

    #[test]
    fn sizes_are_whole_pages_and_never_shrink() {
        assert_eq!(page_size_of(0), 4096);
        assert_eq!(page_size_of(4), 4096);
        assert_eq!(page_size_of(4096), 4096);
        assert_eq!(page_size_of(4097), 8192);
        assert_eq!(page_size_of(158_700), 159_744);
    }
}
