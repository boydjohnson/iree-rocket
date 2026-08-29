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
    /// eagerly (see `destroy`'s TODO on explicit GEM_CLOSE) -- this is a
    /// logical marker so further queue ops against an already-dealloca'd
    /// buffer correctly fail (CTS's `QueueAllocaTest.DeallocaReleasesMemory`)
    /// instead of silently succeeding against memory the caller has already
    /// been told is gone.
    pub deallocated: std::sync::atomic::AtomicBool,
}

unsafe fn cast(buffer: *mut iree_hal_buffer_t) -> *mut RocketBuffer {
    buffer as *mut RocketBuffer
}

/// Not part of the vtable -- `device::queue_dealloca`/`queue_fill`/
/// `queue_update`/`queue_copy` call these directly to enforce the
/// dealloca contract described on `RocketBuffer::deallocated`.
pub unsafe fn is_deallocated(buffer: *mut iree_hal_buffer_t) -> bool {
    unsafe {
        (*cast(buffer))
            .deallocated
            .load(std::sync::atomic::Ordering::Acquire)
    }
}

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
    let buffer = unsafe { Box::from_raw(cast(buffer)) };
    // TODO: no explicit DRM_IOCTL_GEM_CLOSE -- that's a generic DRM ioctl,
    // not part of the 4-ioctl rocket-specific UAPI iree-rocket-hal's
    // api.rs currently binds, and every hand-written test client in this
    // project (rkt-basic.rs et al) also relies on implicit cleanup at
    // process/fd-close time instead (see the harmless "Memory manager not
    // clean during takedown" dmesg warning in NOTES.md). Add real
    // GEM_CLOSE support to iree-rocket-hal if this ever needs to free
    // buffers mid-process rather than at device teardown.
    if !buffer.host_ptr.is_null() && buffer.base.allocation_size > 0 {
        let _ = unsafe {
            nix::sys::mman::munmap(
                std::ptr::NonNull::new_unchecked(buffer.host_ptr as *mut std::ffi::c_void),
                buffer.base.allocation_size,
            )
        };
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
    let allowed_access = unsafe { (*mapping).impl_.allowed_access };
    if allowed_access & (iree_hal_memory_access_bits_t_IREE_HAL_MEMORY_ACCESS_WRITE as u16) != 0 {
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
