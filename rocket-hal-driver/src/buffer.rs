//! `iree_hal_buffer_vtable_t`, backed by `iree-rocket-hal`'s `Buffer`
//! (`DRM_ROCKET_CREATE_BO` + `mmap`, factored out to
//! `iree_rocket_hal::rocket::device` for this crate to share -- see
//! rknpu-spelunking/NOTES.md).
//!
//! Design: CREATE_BO gives a single physical allocation that's
//! *permanently* CPU-mapped (`mmap`'d once, at creation) and NPU-visible
//! via its `dma_address` -- unlike a discrete-memory GPU driver there's no
//! separate "map"/"unmap" step per access, so `map_range`/`unmap_range`
//! just hand out/release a view into the buffer's already-mmap'd
//! `host_ptr`. The two operations that DO need real ioctls are
//! `flush_range` ("writes on the host are expected to be visible to the
//! device after this returns" -- exactly `FINI_BO`'s CPU-done-writing
//! semantics) and `invalidate_range` ("writes on the device are expected
//! to be visible to the host after this returns" -- exactly `PREP_BO`'s
//! wait-for-device-completion semantics, the same call `rkt-basic.rs` uses
//! to read back its result).

use std::os::fd::RawFd;

use iree_rocket_hal::rocket::device;

use crate::{
    bindings::{
        iree_byte_span_t, iree_device_size_t, iree_hal_buffer_mapping_t, iree_hal_buffer_t,
        iree_hal_buffer_vtable_t, iree_hal_mapping_mode_t,
        iree_hal_memory_access_bits_t_IREE_HAL_MEMORY_ACCESS_WRITE, iree_hal_memory_access_t,
        iree_status_t,
    },
    status,
};

/// What every `iree_hal_buffer_t*` this driver hands out actually points
/// to. `base` (the real, fully-defined `iree_hal_buffer_t` -- see
/// `iree/hal/buffer.h`, filled via the real `iree_hal_buffer_initialize`)
/// must be the first field so casts between `*mut iree_hal_buffer_t` and
/// `*mut RocketBuffer` are valid -- the same "base struct at offset 0"
/// convention `iree_hal_resource_t` uses, just one level up.
#[repr(C)]
pub struct RocketBuffer {
    pub base: iree_hal_buffer_t,
    pub handle: u32,
    /// The 32-bit DMA-visible address -- what regcmd fields (e.g.
    /// `CnaFeatureDataAddr`) actually need, distinct from `handle` (a GEM
    /// handle, ioctl-only) and `host_ptr` (CPU virtual address, host-only).
    pub dma_address: u32,
    pub host_ptr: *mut u8,
    pub fd: RawFd,
    /// Set by `device::queue_dealloca`. The real GEM allocation isn't freed
    /// until the last retained HAL reference reaches `destroy`; this logical
    /// marker makes further queue ops against an already-dealloca'd buffer
    /// fail immediately (CTS's `QueueAllocaTest.DeallocaReleasesMemory`)
    /// instead of silently succeeding against memory the caller has already
    /// been told is gone.
    pub deallocated: std::sync::atomic::AtomicBool,
    /// Bumped on every write this driver can observe -- see
    /// `weight_cache`'s "Why a hit is safe". The packed-coefficient cache
    /// keys reuse on it, so a write that does not bump it is a correctness
    /// bug in that cache, not just a stale statistic.
    pub generation: crate::weight_cache::Generation,
    /// The `generation` at which this driver last ran `FINI_BO` over the
    /// whole buffer for an NPU read, or `u64::MAX` if it never has. Read and
    /// written only for buffers IREE declared immutable -- see
    /// [`constant_sync_due`].
    pub synced_generation: std::sync::atomic::AtomicU64,
    /// Bytes of IOVA the GEM object maps from `dma_address`: the allocation
    /// rounded up to whole pages, which can be more than IREE's view of the
    /// buffer. The hardware may read up to here without faulting; past it
    /// the page may be unmapped (see `command_buffer::stage_direct`).
    pub mapped_bytes: usize,
    /// Where `destroy` returns the GEM object (`buffer_cache.rs`).
    pub cache: std::sync::Arc<crate::buffer_cache::BufferCache>,
    /// Who owns the GEM object behind `handle` -- see [`Backing`].
    pub backing: Backing,
}

/// Who owns the GEM object a `RocketBuffer` names, which decides what
/// `destroy` does with it.
///
/// Every buffer the allocator hands out owns its own BO. The other two exist
/// for explicit `iree_hal_pool_t`s (`pool_backend.rs`): a pool carves its
/// reservations out of slabs, each slab one owned BO, and both kinds below
/// are windows into a slab -- `handle` is the slab's, `dma_address` and
/// `host_ptr` are offset into it. They stay `RocketBuffer`s, not IREE
/// subspans, because the dispatch path casts a binding's buffer straight to
/// `RocketBuffer` to read its DMA address.
pub enum Backing {
    /// The allocator's: `destroy` returns the BO to `cache`.
    Owned,
    /// A slab provider's `wrap_buffer` result: `parent` is the slab's own
    /// (retained) buffer, `release` the pool's callback, run on destroy.
    SlabView {
        parent: *mut iree_hal_buffer_t,
        release: crate::bindings::iree_hal_buffer_release_callback_t,
    },
    /// What `queue_alloca` returns for an explicit pool. The pool may be
    /// exhausted when the call is made, so the storage can arrive later --
    /// see [`crate::pool_backend::Pooled`].
    Pooled(Box<crate::pool_backend::Pooled>),
}

unsafe fn cast(buffer: *mut iree_hal_buffer_t) -> *mut RocketBuffer {
    buffer as *mut RocketBuffer
}

/// Whether `buffer` is a pooled allocation whose storage has not arrived yet
/// -- a `queue_alloca` against an exhausted pool that is still waiting for a
/// reservation. Such a buffer has no address: nothing may record it into a
/// dispatch or map it until its alloca signal fires.
///
/// # Safety
///
/// `buffer` must be a valid, non-null pointer to a live `RocketBuffer`.
pub unsafe fn is_pending(buffer: *mut iree_hal_buffer_t) -> bool {
    match unsafe { &(*cast(buffer)).backing } {
        Backing::Pooled(pooled) => !pooled.is_committed(),
        _ => false,
    }
}

fn pending_status() -> iree_status_t {
    status::from_code(crate::bindings::iree_status_code_e_IREE_STATUS_FAILED_PRECONDITION)
}

/// Not part of the vtable -- `device::queue_dealloca`/`queue_fill`/
/// `queue_update`/`queue_copy` call these directly to enforce the
/// dealloca contract described on `RocketBuffer::deallocated`.
///
/// # Safety
///
/// `buffer` must be a valid, non-null pointer to a `RocketBuffer` created
/// by this driver's allocator and still live.
pub unsafe fn is_deallocated(buffer: *mut iree_hal_buffer_t) -> bool {
    unsafe {
        (*cast(buffer))
            .deallocated
            .load(std::sync::atomic::Ordering::Acquire)
    }
}

/// The buffer's current write generation, for `weight_cache` keys.
///
/// # Safety
///
/// `buffer` must be a valid, non-null pointer to a `RocketBuffer` created
/// by this driver's allocator and still live.
pub unsafe fn generation(buffer: *mut iree_hal_buffer_t) -> u64 {
    unsafe { (*cast(buffer)).generation.current() }
}

/// Records that the buffer's contents have changed.
///
/// Called from `unmap_range` for every host write, and directly from
/// `device::queue_execute`'s output compaction, which writes `host_ptr`
/// without going through a mapping. Those are the only two ways bytes in an
/// IREE buffer change; a third would need a call here too.
///
/// # Safety
///
/// `buffer` must be a valid, non-null pointer to a `RocketBuffer` created
/// by this driver's allocator and still live.
pub unsafe fn note_write(buffer: *mut iree_hal_buffer_t) {
    unsafe { (*cast(buffer)).generation.bump() }
}

/// For an NPU read of `buffer` through GEM `handle`, on context 0 when `home`
/// is set: `None` when
/// the read must be synced the ordinary way (this is not that buffer, or it
/// is not a constant), `Some(None)` when it is a constant already synced at
/// its current generation, so the `FINI_BO` can be skipped, and
/// `Some(Some(generation))` when it is a constant that must be synced now and
/// then recorded with [`note_constant_synced`].
///
/// ISSUES.md P3: `FINI_BO` walks the whole BO, and with packed weights bound
/// directly the BO is IREE's combined constant buffer -- tens of MiB, synced
/// on every dispatch, 41% of ResNet50's wall. A constant only needs it once
/// per write. IREE gives every `Lifetime::Constant` resource
/// `IREE_HAL_BUFFER_USAGE_SHARING_IMMUTABLE` ("immutable once initialized;
/// implementations are allowed to avoid synchronization"), and every host
/// write this driver can see bumps `generation` (`note_write`).
///
/// The one write it cannot see is a CPU dispatch through `local-task`'s
/// persistent mapping, which never unmaps. That is why the sync is recorded
/// only here, at an NPU read, and not at `unmap_range`: a constant is
/// finished initializing before anything reads it, so the first NPU read
/// syncs whatever the CPU wrote, and an immutable buffer is not written
/// after that. `ROCKET_SYNC_CONSTANTS=1` restores the per-dispatch sync.
///
/// # Safety
///
/// `buffer` must be a valid, non-null pointer to a `RocketBuffer` created
/// by this driver's allocator and still live.
///
/// Only context 0 names an IREE buffer's own GEM handle: its file is the one
/// the allocator `dup`s (so `rb.fd` is a different number for the same open
/// file), and every other context stages bindings into its own scratch,
/// whose handle numbers can collide with this buffer's. Hence `home`, not an
/// fd comparison.
pub unsafe fn constant_sync_due(
    buffer: *mut iree_hal_buffer_t,
    home: bool,
    handle: u32,
) -> Option<Option<u64>> {
    let rb = unsafe { &*cast(buffer) };
    if !home || rb.handle != handle || sync_constants_every_dispatch() {
        return None;
    }
    let usage = unsafe { crate::bindings::iree_hal_buffer_allowed_usage(buffer) };
    if usage & crate::bindings::iree_hal_buffer_usage_bits_t_IREE_HAL_BUFFER_USAGE_SHARING_IMMUTABLE
        == 0
    {
        return None;
    }
    let generation = rb.generation.current();
    let synced = rb
        .synced_generation
        .load(std::sync::atomic::Ordering::Acquire);
    Some((synced != generation).then_some(generation))
}

/// Records that `buffer` was synced for device reads at `generation`.
///
/// # Safety
///
/// As [`constant_sync_due`].
pub unsafe fn note_constant_synced(buffer: *mut iree_hal_buffer_t, generation: u64) {
    unsafe {
        (*cast(buffer))
            .synced_generation
            .store(generation, std::sync::atomic::Ordering::Release)
    }
}

/// Whether to `FINI_BO` constant buffers before every NPU read, as before
/// ISSUES.md P3 (`ROCKET_SYNC_CONSTANTS=1`), for an A/B or to rule the skip
/// out of a wrong result.
fn sync_constants_every_dispatch() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ROCKET_SYNC_CONSTANTS").is_ok_and(|value| value != "0"))
}

/// # Safety
///
/// `buffer` must be a valid, non-null pointer to a `RocketBuffer` created
/// by this driver's allocator and still live.
pub unsafe fn mark_deallocated(buffer: *mut iree_hal_buffer_t) {
    unsafe {
        (*cast(buffer))
            .deallocated
            .store(true, std::sync::atomic::Ordering::Release)
    }
}

unsafe extern "C" fn recycle(buffer: *mut iree_hal_buffer_t) {
    // The real IREE helper (handles the pooling-allocator hack noted on
    // iree_hal_buffer_t::pooling_allocator) -- not something to
    // reimplement, just forward to it.
    unsafe { crate::bindings::iree_hal_buffer_recycle(buffer) }
}

unsafe extern "C" fn destroy(buffer: *mut iree_hal_buffer_t) {
    // Before the allocation goes away, so no later buffer landing on this
    // address can inherit its packed coefficients -- see `weight_cache`.
    crate::weight_cache::forget(buffer);
    let rb = unsafe { Box::from_raw(cast(buffer)) };
    let RocketBuffer {
        handle,
        dma_address,
        host_ptr,
        mapped_bytes,
        cache,
        backing,
        ..
    } = *rb;
    match backing {
        // The GEM object goes back to the allocator's cache, which keeps it
        // for the next allocation of its size or releases both of its
        // references (the mapping and the handle) -- by the mapping's real
        // length, which `allocation_size` is not: munmapping that leaves the
        // tail page mapped, and the VMA alone keeps the GEM object and its
        // IOVA alive.
        Backing::Owned => unsafe {
            cache.give(device::Buffer {
                handle,
                dma_address,
                size: mapped_bytes,
                host_ptr,
            })
        },
        // The slab owns the BO. The pool's callback returns the
        // reservation, then the slab loses this view's reference.
        Backing::SlabView { parent, release } => unsafe {
            if let Some(f) = release.fn_ {
                f(release.user_data, buffer);
            }
            crate::bindings::iree_hal_buffer_release(parent);
        },
        Backing::Pooled(pooled) => unsafe { (*pooled).destroy() },
    }
}

#[allow(unused_variables)]
unsafe extern "C" fn map_range(
    buffer: *mut iree_hal_buffer_t,
    mapping_mode: iree_hal_mapping_mode_t,
    memory_access: iree_hal_memory_access_t,
    local_byte_offset: iree_device_size_t,
    local_byte_length: iree_device_size_t,
    mapping: *mut iree_hal_buffer_mapping_t,
) -> iree_status_t {
    if unsafe { is_pending(buffer) } {
        return pending_status();
    }
    let rb = unsafe { &*cast(buffer) };
    unsafe {
        (*mapping).contents = iree_byte_span_t {
            data: rb.host_ptr.add(local_byte_offset),
            data_length: local_byte_length,
        };
        (*mapping).buffer = buffer;
        (*mapping).impl_.byte_offset = local_byte_offset;
        (*mapping).impl_.allowed_access = memory_access;
    }
    status::ok()
}

#[allow(unused_variables)]
unsafe extern "C" fn unmap_range(
    buffer: *mut iree_hal_buffer_t,
    local_byte_offset: iree_device_size_t,
    local_byte_length: iree_device_size_t,
    mapping: *mut iree_hal_buffer_mapping_t,
) -> iree_status_t {
    // The buffer stays mmap'd for its whole lifetime (see module doc
    // comment), so there's no real (un)mapping work to do here -- EXCEPT
    // for one real, load-bearing piece of semantics a genuine hardware
    // driver's unmap would naturally provide and this no-op previously
    // didn't: on this non-coherent device, a WRITE mapping's contents
    // aren't visible to the NPU until FINI_BO runs (see flush_range's own
    // doc comment). Most direct callers know this and call
    // iree_hal_buffer_mapping_flush_range/iree_hal_buffer_map_write
    // explicitly (which check IREE_HAL_MEMORY_TYPE_HOST_COHERENT and flush
    // when it's absent, as this driver's buffers always report) -- but
    // IREE's own `iree_hal_buffer_view_generate_buffer_in_situ`
    // (buffer_view_util.c, the exact code path `iree-run-module --input=`
    // splat/fill values go through) does NOT: it maps for
    // DISCARD_WRITE, runs its fill callback, and unmaps, relying entirely
    // on unmap to make the write visible -- which a true no-op unmap never
    // does. Confirmed as a real bug on hardware: input/weights buffers
    // populated this way looked correct to the CPU (same-address
    // same-core load-after-store is always coherent, mapping or not) but
    // the DPU read stale/pre-existing physical memory instead, producing a
    // deterministic, input-independent garbage result. Flushing here
    // whenever the mapping allowed any write access closes that gap for
    // every caller, not just this one -- and is a no-op (one extra
    // already-cheap FINI_BO ioctl) for the callers that already flush
    // explicitly themselves.
    //
    // This is also the one place every host write to an IREE buffer passes
    // through -- `iree_hal_buffer_map_fill`/`_write`/`_copy`, and therefore
    // the command-buffer ops, the device `queue_*` ops and
    // `iree_hal_file_read`, all reduce to a write mapping ending here -- so
    // it is where the packed-coefficient cache's generation is bumped. Same
    // condition, same reason: bytes changed.
    let allowed_access = unsafe { (*mapping).impl_.allowed_access };
    if allowed_access & (iree_hal_memory_access_bits_t_IREE_HAL_MEMORY_ACCESS_WRITE as u16) != 0 {
        unsafe { note_write(buffer) };
        let rb = unsafe { &*cast(buffer) };
        if unsafe { device::fini_bo(rb.fd, rb.handle) }.is_err() {
            return status::from_code(crate::bindings::iree_status_code_e_IREE_STATUS_INTERNAL);
        }
    }
    status::ok()
}

#[allow(unused_variables)]
unsafe extern "C" fn invalidate_range(
    buffer: *mut iree_hal_buffer_t,
    local_byte_offset: iree_device_size_t,
    local_byte_length: iree_device_size_t,
) -> iree_status_t {
    if unsafe { is_pending(buffer) } {
        return pending_status();
    }
    let rb = unsafe { &*cast(buffer) };
    // "Writes on the device are expected to be visible to the host after
    // this returns" -- PREP_BO's real semantics (rocket_gem.c waits on
    // the buffer's dma_resv reservation). By the time IREE calls this the
    // relevant queue-side semaphore wait has normally already completed,
    // so this should return immediately in practice; the timeout here is
    // just a generous backstop, not the primary synchronization -- see
    // NOTES.md for why it must be a relative-duration wrapper (device::
    // prep_bo), not a bare absolute deadline.
    match unsafe { device::prep_bo(rb.fd, rb.handle, 5_000_000_000) } {
        Ok(()) => status::ok(),
        Err(_) => {
            status::from_code(crate::bindings::iree_status_code_e_IREE_STATUS_DEADLINE_EXCEEDED)
        }
    }
}

#[allow(unused_variables)]
unsafe extern "C" fn flush_range(
    buffer: *mut iree_hal_buffer_t,
    local_byte_offset: iree_device_size_t,
    local_byte_length: iree_device_size_t,
) -> iree_status_t {
    if unsafe { is_pending(buffer) } {
        return pending_status();
    }
    let rb = unsafe { &*cast(buffer) };
    // "Writes on the host are expected to be visible to the device after
    // this returns" -- FINI_BO's real semantics (signals CPU access is
    // done, safe for the NPU to read).
    match unsafe { device::fini_bo(rb.fd, rb.handle) } {
        Ok(()) => status::ok(),
        Err(_) => status::from_code(crate::bindings::iree_status_code_e_IREE_STATUS_INTERNAL),
    }
}

pub static VTABLE: iree_hal_buffer_vtable_t = iree_hal_buffer_vtable_t {
    recycle: Some(recycle),
    destroy: Some(destroy),
    map_range: Some(map_range),
    unmap_range: Some(unmap_range),
    invalidate_range: Some(invalidate_range),
    flush_range: Some(flush_range),
};
