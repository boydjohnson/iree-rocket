//! `iree_hal_allocator_vtable_t`. `allocate_buffer`/`deallocate_buffer` are
//! backed by our existing `Buffer` (`iree_rocket_hal::rocket::device`,
//! `DRM_ROCKET_CREATE_BO` + `mmap`). Virtual-memory reservation
//! (`virtual_memory_*`, `physical_memory_*`) isn't something the rocket
//! driver needs -- CREATE_BO already gives a CPU-mapped, DMA-capable
//! allocation directly, no separate reserve/map step -- so those stay
//! UNIMPLEMENTED indefinitely, not just for now.

use std::os::fd::AsRawFd;

use iree_rocket_hal::rocket::device;

use crate::{
    bindings::{
        iree_allocator_t, iree_device_size_t, iree_hal_allocator_memory_heap_t,
        iree_hal_allocator_statistics_t, iree_hal_allocator_t, iree_hal_allocator_vtable_t,
        iree_hal_buffer_compatibility_bits_t_IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE,
        iree_hal_buffer_compatibility_bits_t_IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH,
        iree_hal_buffer_compatibility_bits_t_IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER,
        iree_hal_buffer_compatibility_t, iree_hal_buffer_params_t,
        iree_hal_buffer_placement_flag_bits_t_IREE_HAL_BUFFER_PLACEMENT_FLAG_ASYNCHRONOUS,
        iree_hal_buffer_placement_t, iree_hal_buffer_release_callback_t, iree_hal_buffer_t,
        iree_hal_buffer_usage_bits_t_IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_RANDOM,
        iree_hal_buffer_usage_bits_t_IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT,
        iree_hal_buffer_usage_bits_t_IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED,
        iree_hal_external_buffer_flags_t, iree_hal_external_buffer_t,
        iree_hal_external_buffer_type_t, iree_hal_memory_advice_t, iree_hal_memory_protection_t,
        iree_hal_memory_type_bits_t_IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
        iree_hal_memory_type_bits_t_IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
        iree_hal_memory_type_bits_t_IREE_HAL_MEMORY_TYPE_OPTIMAL, iree_hal_physical_memory_t,
        iree_hal_queue_affinity_t, iree_hal_resource_t, iree_host_size_t, iree_status_t,
    },
    buffer::RocketBuffer,
    status,
};

/// What every `iree_hal_allocator_t*` this driver hands out actually
/// points to. `iree_hal_allocator_t` (unlike `iree_hal_buffer_t`) has no
/// public field definition at all -- its header only forward-declares it
/// -- so `resource` (not some `iree_hal_allocator_t` we can't even name a
/// concrete instance of) is the real base-at-offset-0 field, exactly like
/// every driver's own allocator struct in IREE itself
/// (`iree_hal_null_allocator_t`, mirrored at
/// rknpu-spelunking/iree-null-driver-reference/allocator.c).
#[repr(C)]
pub struct RocketAllocator {
    pub resource: iree_hal_resource_t,
    pub host_allocator: iree_allocator_t,
    /// The open `/dev/accel/accel0` handle -- kept alive for the
    /// allocator's lifetime; every `Buffer::new` mmap needs a live `File`.
    pub file: std::fs::File,
    /// The device that owns this allocator, stamped into every buffer's
    /// `iree_hal_buffer_placement_t` (see `allocate_buffer`). Not owned/
    /// retained -- the allocator is itself a field of `RocketDevice`, so
    /// it can never outlive its owning device. Set via `set_device` right
    /// after `device::create` has the device's final, boxed pointer (the
    /// allocator has to exist before that pointer does, since it's one of
    /// the fields used to construct `RocketDevice` in the first place).
    pub device: *mut crate::bindings::iree_hal_device_t,
}

/// Mirrors `iree_hal_null_allocator_create()`. Not called from anywhere
/// yet -- `driver::factory_try_create` (still UNIMPLEMENTED) is what will
/// eventually open `/dev/accel/accel0` and call this.
pub fn create(file: std::fs::File, host_allocator: iree_allocator_t) -> *mut iree_hal_allocator_t {
    let allocator = Box::new(RocketAllocator {
        resource: iree_hal_resource_t {
            ref_count: 1,
            vtable: &VTABLE as *const _ as *const std::ffi::c_void,
        },
        host_allocator,
        file,
        device: std::ptr::null_mut(),
    });
    Box::into_raw(allocator) as *mut iree_hal_allocator_t
}

/// See `RocketAllocator::device`'s doc comment for why this is a
/// separate, post-construction step rather than a `create` parameter.
pub unsafe fn set_device(
    allocator: *mut iree_hal_allocator_t,
    device: *mut crate::bindings::iree_hal_device_t,
) {
    unsafe { (*cast(allocator)).device = device };
}

unsafe fn cast(allocator: *mut iree_hal_allocator_t) -> *mut RocketAllocator {
    allocator as *mut RocketAllocator
}

unsafe extern "C" fn destroy(allocator: *mut iree_hal_allocator_t) {
    // Dropping the Box drops the embedded File, closing the fd -- the
    // kernel then cleans up any handles still outstanding as a backstop.
    // Normal HAL buffers and driver-private owned buffers close their GEM
    // handles eagerly when their own lifetimes end.
    unsafe {
        drop(Box::from_raw(cast(allocator)));
    }
}

unsafe extern "C" fn host_allocator(allocator: *const iree_hal_allocator_t) -> iree_allocator_t {
    let alloc = unsafe { &*cast(allocator as *mut iree_hal_allocator_t) };
    alloc.host_allocator
}

status_stub!(trim(allocator: *mut iree_hal_allocator_t) -> iree_status_t);

void_stub!(query_statistics(
    allocator: *mut iree_hal_allocator_t,
    out_statistics: *mut iree_hal_allocator_statistics_t,
));

status_stub!(query_memory_heaps(
    allocator: *mut iree_hal_allocator_t,
    capacity: iree_host_size_t,
    heaps: *mut iree_hal_allocator_memory_heap_t,
    out_count: *mut iree_host_size_t,
) -> iree_status_t);

#[allow(unused_variables)]
unsafe extern "C" fn query_buffer_compatibility(
    allocator: *mut iree_hal_allocator_t,
    params: *mut iree_hal_buffer_params_t,
    allocation_size: *mut iree_device_size_t,
) -> iree_hal_buffer_compatibility_t {
    unsafe {
        // Unified memory (RK3588's NPU shares system DRAM with the CPU,
        // not discrete VRAM) -- every buffer this driver hands out is both
        // genuinely CPU-mappable (permanently mmap'd, see buffer.rs) and
        // NPU-visible (via dma_address), so OPTIMAL always canonicalizes to
        // that concrete pair rather than just having its placeholder bit
        // cleared -- an allocate_buffer/dispatch caller that validates the
        // resulting memory_type (e.g. iree_hal_command_buffer_dispatch's
        // binding validation) needs DEVICE_VISIBLE actually set, not just
        // OPTIMAL cleared to 0. Deliberately not HOST_COHERENT -- this
        // memory is genuinely non-coherent (see buffer.rs's flush_range/
        // invalidate_range -> FINI_BO/PREP_BO doc comments).
        (*params).type_ &= !iree_hal_memory_type_bits_t_IREE_HAL_MEMORY_TYPE_OPTIMAL;
        (*params).type_ |= iree_hal_memory_type_bits_t_IREE_HAL_MEMORY_TYPE_HOST_VISIBLE
            | iree_hal_memory_type_bits_t_IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
        // Same real-hardware truth as the memory_type bits just above: since
        // every buffer this driver hands out is unconditionally, permanently
        // host-mmap'd (buffer.rs), it can always honor persistent/scoped/
        // random-access host mapping -- there is no allocation this driver
        // could produce that DOESN'T support this, so advertise it
        // unconditionally rather than only when a caller happens to ask for
        // it (mirrors iree_hal_heap_allocator_query_buffer_compatibility's
        // identical opportunistic-mapping stance in allocator_heap.c).
        // Real, load-bearing bug this fixes (found empirically on hardware,
        // multi-device rocket+local-sync program): a buffer allocated via
        // THIS allocator but later actually computed on by a DIFFERENT
        // device's queue_execute (e.g. local-sync, which genuinely needs a
        // persistent host pointer for inline CPU dispatch) failed
        // `iree_hal_buffer_validate_usage` with PERMISSION_DENIED --
        // "operation requires MAPPING_PERSISTENT" -- because this allocator
        // never granted it, unlike the heap allocator every other local
        // device already gets this from automatically.
        (*params).usage |= iree_hal_buffer_usage_bits_t_IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED
            | iree_hal_buffer_usage_bits_t_IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT
            | iree_hal_buffer_usage_bits_t_IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_RANDOM;
        // Guard the 0-byte corner case (real apps can hit this).
        if *allocation_size == 0 {
            *allocation_size = 4;
        }
    }
    iree_hal_buffer_compatibility_bits_t_IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE
        | iree_hal_buffer_compatibility_bits_t_IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER
        | iree_hal_buffer_compatibility_bits_t_IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH
}

unsafe extern "C" fn allocate_buffer(
    allocator: *mut iree_hal_allocator_t,
    params: *const iree_hal_buffer_params_t,
    allocation_size: iree_device_size_t,
    out_buffer: *mut *mut iree_hal_buffer_t,
) -> iree_status_t {
    let alloc = unsafe { &*cast(allocator) };
    let params = unsafe { &*params };

    // iree_hal_allocator_allocate_buffer (allocator.c) canonicalizes zero
    // fields (iree_hal_buffer_params_canonicalize) but never calls
    // query_buffer_compatibility -- that's only invoked by the separate
    // iree_hal_allocator_query_buffer_compatibility entry point, which
    // nothing in this driver's own dispatch path calls before allocating.
    // So a caller that passes plain IREE_HAL_MEMORY_TYPE_OPTIMAL (as
    // canonicalize's own default, and as CTS/hand-written tests commonly
    // do) reaches here with `params.type_` still just the OPTIMAL
    // placeholder bit -- no concrete HOST_VISIBLE/DEVICE_VISIBLE bits ever
    // get set on the resulting iree_hal_buffer_t. That's exactly what
    // surfaced as command_buffer.c's dispatch-binding validation rejecting
    // every real dispatch with "buffer has OPTIMAL, operation requires
    // DEVICE_VISIBLE" (PERMISSION_DENIED) the first time a real workgroup
    // count made it reach validation at all. Canonicalize here too, same
    // concrete bits query_buffer_compatibility above now sets.
    let mut memory_type = params.type_;
    memory_type &= !iree_hal_memory_type_bits_t_IREE_HAL_MEMORY_TYPE_OPTIMAL;
    memory_type |= iree_hal_memory_type_bits_t_IREE_HAL_MEMORY_TYPE_HOST_VISIBLE
        | iree_hal_memory_type_bits_t_IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;

    // Same real-hardware truth as query_buffer_compatibility above (and this
    // function's own memory_type canonicalization just above): every buffer
    // this driver hands out is unconditionally, permanently host-mmap'd
    // (buffer.rs), so it can always honor persistent/scoped/random-access
    // host mapping regardless of what the caller happened to request --
    // this is the entry point CTS/hand-written tests and a real compiled
    // .vmfb's `hal.device.queue.alloca` actually go through (unlike
    // query_buffer_compatibility, see this function's own top doc comment),
    // so it must carry this fix too, not just that other entry point. Fixes
    // a real bug found on hardware: a buffer allocated here but later
    // genuinely computed on by a DIFFERENT device's queue_execute (e.g.
    // local-sync, in a rocket+local-sync multi-device program) failed
    // `iree_hal_buffer_validate_usage` with PERMISSION_DENIED -- "operation
    // requires MAPPING_PERSISTENT" -- because this allocator never granted
    // it, unlike every other local device's heap allocator (allocator_heap.c)
    // already does automatically.
    let usage = params.usage
        | iree_hal_buffer_usage_bits_t_IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED
        | iree_hal_buffer_usage_bits_t_IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT
        | iree_hal_buffer_usage_bits_t_IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_RANDOM;

    // Same 0-byte guard as query_buffer_compatibility (real apps -- and
    // CTS's AllocatorTest.AllocateEmptyBuffer -- hit this): unlike that
    // query, iree_hal_allocator_allocate_buffer() dispatches straight to
    // this vtable slot with the raw, unclamped size, so allocate_buffer
    // itself must not pass 0 through to the kernel. The real
    // `accel/rocket` driver's CREATE_BO rejects a 0-byte request with
    // ENOSPC (drm_mm_insert_node_generic can't place a zero-length node),
    // and even if it didn't, iree_rocket_hal's own Buffer::new immediately
    // follows with `NonZeroUsize::new(size).unwrap()` for the mmap length,
    // which panics on exactly this input.
    let real_size = allocation_size.max(4);
    let raw = unsafe { device::Buffer::new(alloc.file.as_raw_fd(), real_size, &alloc.file) };

    let buffer = Box::new(RocketBuffer {
        // Filled in by iree_hal_buffer_initialize below -- this is just
        // reserving the memory; the real fields (resource/vtable/sizes/
        // memory_type/...) get set by that call, matching how every real
        // HAL driver's allocate_buffer works.
        base: unsafe { std::mem::zeroed() },
        handle: raw.handle,
        dma_address: raw.dma_address,
        host_ptr: raw.host_ptr,
        fd: alloc.file.as_raw_fd(),
        deallocated: std::sync::atomic::AtomicBool::new(false),
        generation: crate::weight_cache::Generation::default(),
    });
    let buffer_ptr = Box::into_raw(buffer);

    unsafe {
        let placement = iree_hal_buffer_placement_t {
            device: alloc.device,
            // This driver has exactly one real queue -- IREE_HAL_QUEUE_
            // AFFINITY_ANY (all bits set) gets resolved to that single
            // concrete queue (bit 0) here, matching what a real multi-
            // queue driver would do when actually placing a buffer.
            // iree_hal_queue_affinity_count(placement.queue_affinity) == 1
            // is a real, checked contract (CTS's QueueAllocaTest.
            // BufferMetadata), not just documentation -- leaving this as
            // 0 or as an unresolved wildcard both fail it.
            queue_affinity: 1,
            // Every buffer this driver hands out can be deallocated via
            // queue_dealloca (see device.rs) -- without this flag,
            // iree_hal_device_queue_dealloca() (device.c) treats the
            // buffer as synchronously-owned and silently substitutes a
            // no-op barrier (iree_hal_device_queue_barrier(), which calls
            // straight into queue_execute with a NULL command buffer)
            // instead of ever calling our real queue_dealloca. That NULL
            // command buffer is exactly what queue_execute segfaulted on
            // before it learned to tolerate it, and without this flag
            // queue_dealloca's deallocated-marking (see buffer.rs) would
            // never actually run for any test that goes through the
            // standard iree_hal_device_queue_dealloca API.
            flags:
                iree_hal_buffer_placement_flag_bits_t_IREE_HAL_BUFFER_PLACEMENT_FLAG_ASYNCHRONOUS,
            reserved: 0,
        };
        crate::bindings::iree_hal_buffer_initialize(
            placement,
            &mut (*buffer_ptr).base,
            real_size,
            0,
            allocation_size,
            memory_type,
            params.access,
            usage,
            &crate::buffer::VTABLE,
            &mut (*buffer_ptr).base,
        );
        *out_buffer = buffer_ptr as *mut iree_hal_buffer_t;
    }
    status::ok()
}

unsafe extern "C" fn deallocate_buffer(
    allocator: *mut iree_hal_allocator_t,
    buffer: *mut iree_hal_buffer_t,
) {
    let _ = allocator; // buffer::destroy doesn't need it back
    // Real IREE helper -- calls our buffer vtable's .destroy after
    // handling the base iree_hal_buffer_t bookkeeping (see
    // iree-null-driver-reference/allocator.c's deallocate_buffer, which
    // does exactly this).
    unsafe { crate::bindings::iree_hal_buffer_destroy(buffer) }
}

status_stub!(import_buffer(
    allocator: *mut iree_hal_allocator_t,
    params: *const iree_hal_buffer_params_t,
    external_buffer: *mut iree_hal_external_buffer_t,
    release_callback: iree_hal_buffer_release_callback_t,
    out_buffer: *mut *mut iree_hal_buffer_t,
) -> iree_status_t);

status_stub!(export_buffer(
    allocator: *mut iree_hal_allocator_t,
    buffer: *mut iree_hal_buffer_t,
    requested_type: iree_hal_external_buffer_type_t,
    requested_flags: iree_hal_external_buffer_flags_t,
    out_external_buffer: *mut iree_hal_external_buffer_t,
) -> iree_status_t);

#[allow(unused_variables)]
pub unsafe extern "C" fn supports_virtual_memory(allocator: *mut iree_hal_allocator_t) -> bool {
    false
}

status_stub!(virtual_memory_query_granularity(
    allocator: *mut iree_hal_allocator_t,
    params: iree_hal_buffer_params_t,
    out_minimum_page_size: *mut iree_device_size_t,
    out_recommended_page_size: *mut iree_device_size_t,
) -> iree_status_t);

status_stub!(virtual_memory_reserve(
    allocator: *mut iree_hal_allocator_t,
    queue_affinity: iree_hal_queue_affinity_t,
    size: iree_device_size_t,
    out_virtual_buffer: *mut *mut iree_hal_buffer_t,
) -> iree_status_t);

status_stub!(virtual_memory_release(
    allocator: *mut iree_hal_allocator_t,
    virtual_buffer: *mut iree_hal_buffer_t,
) -> iree_status_t);

status_stub!(physical_memory_allocate(
    allocator: *mut iree_hal_allocator_t,
    params: iree_hal_buffer_params_t,
    size: iree_device_size_t,
    host_allocator: iree_allocator_t,
    out_physical_memory: *mut *mut iree_hal_physical_memory_t,
) -> iree_status_t);

status_stub!(physical_memory_free(
    allocator: *mut iree_hal_allocator_t,
    physical_memory: *mut iree_hal_physical_memory_t,
) -> iree_status_t);

status_stub!(virtual_memory_map(
    allocator: *mut iree_hal_allocator_t,
    virtual_buffer: *mut iree_hal_buffer_t,
    virtual_offset: iree_device_size_t,
    physical_memory: *mut iree_hal_physical_memory_t,
    physical_offset: iree_device_size_t,
    size: iree_device_size_t,
) -> iree_status_t);

status_stub!(virtual_memory_unmap(
    allocator: *mut iree_hal_allocator_t,
    virtual_buffer: *mut iree_hal_buffer_t,
    virtual_offset: iree_device_size_t,
    size: iree_device_size_t,
) -> iree_status_t);

status_stub!(virtual_memory_protect(
    allocator: *mut iree_hal_allocator_t,
    virtual_buffer: *mut iree_hal_buffer_t,
    virtual_offset: iree_device_size_t,
    size: iree_device_size_t,
    queue_affinity: iree_hal_queue_affinity_t,
    protection: iree_hal_memory_protection_t,
) -> iree_status_t);

status_stub!(virtual_memory_advise(
    allocator: *mut iree_hal_allocator_t,
    virtual_buffer: *mut iree_hal_buffer_t,
    virtual_offset: iree_device_size_t,
    size: iree_device_size_t,
    queue_affinity: iree_hal_queue_affinity_t,
    advice: iree_hal_memory_advice_t,
) -> iree_status_t);

pub static VTABLE: iree_hal_allocator_vtable_t = iree_hal_allocator_vtable_t {
    destroy: Some(destroy),
    host_allocator: Some(host_allocator),
    trim: Some(trim),
    query_statistics: Some(query_statistics),
    query_memory_heaps: Some(query_memory_heaps),
    query_buffer_compatibility: Some(query_buffer_compatibility),
    allocate_buffer: Some(allocate_buffer),
    deallocate_buffer: Some(deallocate_buffer),
    import_buffer: Some(import_buffer),
    export_buffer: Some(export_buffer),
    supports_virtual_memory: Some(supports_virtual_memory),
    virtual_memory_query_granularity: Some(virtual_memory_query_granularity),
    virtual_memory_reserve: Some(virtual_memory_reserve),
    virtual_memory_release: Some(virtual_memory_release),
    physical_memory_allocate: Some(physical_memory_allocate),
    physical_memory_free: Some(physical_memory_free),
    virtual_memory_map: Some(virtual_memory_map),
    virtual_memory_unmap: Some(virtual_memory_unmap),
    virtual_memory_protect: Some(virtual_memory_protect),
    virtual_memory_advise: Some(virtual_memory_advise),
};
