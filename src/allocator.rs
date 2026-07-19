//! `iree_hal_allocator_vtable_t`. `allocate_buffer`/`import_buffer` are the
//! two slots that matter most for this driver -- they're where our
//! existing `Buffer` (CREATE_BO + mmap, in `iree-rocket-hal`) plugs in.
//! Virtual-memory reservation (`virtual_memory_*`, `physical_memory_*`)
//! isn't something the rocket driver needs -- CREATE_BO already gives a
//! CPU-mapped, DMA-capable allocation directly, no separate reserve/map
//! step -- so those stay UNIMPLEMENTED indefinitely, not just for now.

use crate::bindings::{
    iree_allocator_t, iree_device_size_t, iree_hal_allocator_memory_heap_t,
    iree_hal_allocator_statistics_t, iree_hal_allocator_t, iree_hal_allocator_vtable_t,
    iree_hal_buffer_compatibility_bits_t_IREE_HAL_BUFFER_COMPATIBILITY_NONE,
    iree_hal_buffer_compatibility_t, iree_hal_buffer_params_t, iree_hal_buffer_release_callback_t,
    iree_hal_buffer_t, iree_hal_external_buffer_flags_t, iree_hal_external_buffer_t,
    iree_hal_external_buffer_type_t, iree_hal_memory_advice_t, iree_hal_memory_protection_t,
    iree_hal_physical_memory_t, iree_hal_queue_affinity_t, iree_host_size_t,
};

void_stub!(destroy(allocator: *mut iree_hal_allocator_t));

#[allow(unused_variables)]
pub unsafe extern "C" fn host_allocator(
    allocator: *const iree_hal_allocator_t,
) -> iree_allocator_t {
    iree_allocator_t {
        self_: std::ptr::null_mut(),
        ctl: None,
    }
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
pub unsafe extern "C" fn query_buffer_compatibility(
    allocator: *mut iree_hal_allocator_t,
    params: *mut iree_hal_buffer_params_t,
    allocation_size: *mut iree_device_size_t,
) -> iree_hal_buffer_compatibility_t {
    iree_hal_buffer_compatibility_bits_t_IREE_HAL_BUFFER_COMPATIBILITY_NONE
}

// TODO: the real one -- allocate via iree-rocket-hal's Buffer::new
// (DRM_ROCKET_CREATE_BO + mmap) and wrap the result in an iree_hal_buffer_t
// (see buffer.rs).
status_stub!(allocate_buffer(
    allocator: *mut iree_hal_allocator_t,
    params: *const iree_hal_buffer_params_t,
    allocation_size: iree_device_size_t,
    out_buffer: *mut *mut iree_hal_buffer_t,
) -> iree_status_t);

void_stub!(deallocate_buffer(
    allocator: *mut iree_hal_allocator_t,
    buffer: *mut iree_hal_buffer_t,
));

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
