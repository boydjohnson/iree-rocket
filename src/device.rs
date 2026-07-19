//! `iree_hal_device_vtable_t` -- the biggest vtable (36 slots). Most return
//! `iree_status_t` and are handled by `status_stub!`; a handful return
//! other value types (`id`, `host_allocator`, `device_allocator`,
//! `topology_info`, `query_semaphore_compatibility`) and are written out by
//! hand below with a sensible default.
//!
//! `queue_execute` is the one slot with real design intent already: per
//! the research this crate started from, it should mirror
//! `sync_device.c`'s fully-synchronous pattern -- SUBMIT, then a blocking
//! PREP_BO wait (our existing `Buffer`/`SUBMIT`/`PREP_BO` primitives from
//! `iree-rocket-hal`), then signal a host-side semaphore for the timeline
//! value -- rather than the async/polling model
//! `iree/hal/utils/deferred_work_queue.h` assumes, which doesn't fit a
//! driver whose only completion signal is a blocking ioctl.

use crate::bindings::{
    iree_allocator_t, iree_const_byte_span_t, iree_device_size_t, iree_hal_alloca_flags_t,
    iree_hal_allocator_t, iree_hal_buffer_binding_table_t, iree_hal_buffer_params_t,
    iree_hal_buffer_ref_list_t, iree_hal_buffer_t, iree_hal_channel_params_t,
    iree_hal_channel_provider_t, iree_hal_channel_t,
    iree_hal_command_buffer_mode_t, iree_hal_command_buffer_t, iree_hal_command_category_t,
    iree_hal_copy_flags_t, iree_hal_dealloca_flags_t, iree_hal_device_capabilities_t,
    iree_hal_device_external_capture_options_t, iree_hal_device_profiling_options_t,
    iree_hal_device_t, iree_hal_device_topology_info_t, iree_hal_device_vtable_t,
    iree_hal_dispatch_config_t, iree_hal_dispatch_flags_t, iree_hal_event_flags_t,
    iree_hal_event_t, iree_hal_executable_cache_t, iree_hal_executable_function_t,
    iree_hal_executable_t, iree_hal_execute_flags_t, iree_hal_external_file_flags_t,
    iree_hal_file_t, iree_hal_fill_flags_t, iree_hal_host_call_flags_t, iree_hal_host_call_t,
    iree_hal_memory_access_t, iree_hal_pool_t, iree_hal_queue_affinity_t,
    iree_hal_queue_pool_backend_t, iree_hal_read_flags_t, iree_hal_semaphore_compatibility_bits_t_IREE_HAL_SEMAPHORE_COMPATIBILITY_NONE,
    iree_hal_semaphore_compatibility_t, iree_hal_semaphore_flags_t, iree_hal_semaphore_list_t,
    iree_hal_semaphore_t, iree_hal_topology_edge_t, iree_hal_update_flags_t,
    iree_hal_write_flags_t, iree_host_size_t, iree_io_file_handle_t,
    iree_string_view_t,
};

void_stub!(destroy(device: *mut iree_hal_device_t));

#[allow(unused_variables)]
pub unsafe extern "C" fn id(device: *mut iree_hal_device_t) -> iree_string_view_t {
    // TODO: return a real identifier ("rocket") once there's a device
    // instance carrying one -- see rkt_device.c-style identifier storage.
    iree_string_view_t {
        data: std::ptr::null(),
        size: 0,
    }
}

#[allow(unused_variables)]
pub unsafe extern "C" fn host_allocator(device: *mut iree_hal_device_t) -> iree_allocator_t {
    iree_allocator_t {
        self_: std::ptr::null_mut(),
        ctl: None,
    }
}

#[allow(unused_variables)]
pub unsafe extern "C" fn device_allocator(
    device: *mut iree_hal_device_t,
) -> *mut iree_hal_allocator_t {
    std::ptr::null_mut()
}

void_stub!(replace_device_allocator(
    device: *mut iree_hal_device_t,
    new_allocator: *mut iree_hal_allocator_t,
));

void_stub!(replace_channel_provider(
    device: *mut iree_hal_device_t,
    new_provider: *mut iree_hal_channel_provider_t,
));

status_stub!(trim(device: *mut iree_hal_device_t) -> iree_status_t);

status_stub!(query_i64(
    device: *mut iree_hal_device_t,
    category: iree_string_view_t,
    key: iree_string_view_t,
    out_value: *mut i64,
) -> iree_status_t);

status_stub!(query_capabilities(
    device: *mut iree_hal_device_t,
    out_capabilities: *mut iree_hal_device_capabilities_t,
) -> iree_status_t);

#[allow(unused_variables)]
pub unsafe extern "C" fn topology_info(
    device: *mut iree_hal_device_t,
) -> *const iree_hal_device_topology_info_t {
    std::ptr::null()
}

status_stub!(refine_topology_edge(
    src_device: *mut iree_hal_device_t,
    dst_device: *mut iree_hal_device_t,
    edge: *mut iree_hal_topology_edge_t,
) -> iree_status_t);

status_stub!(assign_topology_info(
    device: *mut iree_hal_device_t,
    topology_info: *const iree_hal_device_topology_info_t,
) -> iree_status_t);

status_stub!(create_channel(
    device: *mut iree_hal_device_t,
    queue_affinity: iree_hal_queue_affinity_t,
    params: iree_hal_channel_params_t,
    out_channel: *mut *mut iree_hal_channel_t,
) -> iree_status_t);

status_stub!(create_command_buffer(
    device: *mut iree_hal_device_t,
    mode: iree_hal_command_buffer_mode_t,
    command_categories: iree_hal_command_category_t,
    queue_affinity: iree_hal_queue_affinity_t,
    binding_capacity: iree_host_size_t,
    out_command_buffer: *mut *mut iree_hal_command_buffer_t,
) -> iree_status_t);

status_stub!(create_event(
    device: *mut iree_hal_device_t,
    queue_affinity: iree_hal_queue_affinity_t,
    flags: iree_hal_event_flags_t,
    out_event: *mut *mut iree_hal_event_t,
) -> iree_status_t);

status_stub!(create_executable_cache(
    device: *mut iree_hal_device_t,
    identifier: iree_string_view_t,
    out_executable_cache: *mut *mut iree_hal_executable_cache_t,
) -> iree_status_t);

status_stub!(import_file(
    device: *mut iree_hal_device_t,
    queue_affinity: iree_hal_queue_affinity_t,
    access: iree_hal_memory_access_t,
    handle: *mut iree_io_file_handle_t,
    flags: iree_hal_external_file_flags_t,
    out_file: *mut *mut iree_hal_file_t,
) -> iree_status_t);

status_stub!(create_semaphore(
    device: *mut iree_hal_device_t,
    queue_affinity: iree_hal_queue_affinity_t,
    initial_value: u64,
    flags: iree_hal_semaphore_flags_t,
    out_semaphore: *mut *mut iree_hal_semaphore_t,
) -> iree_status_t);

#[allow(unused_variables)]
pub unsafe extern "C" fn query_semaphore_compatibility(
    device: *mut iree_hal_device_t,
    semaphore: *mut iree_hal_semaphore_t,
) -> iree_hal_semaphore_compatibility_t {
    iree_hal_semaphore_compatibility_bits_t_IREE_HAL_SEMAPHORE_COMPATIBILITY_NONE
}

status_stub!(query_queue_pool_backend(
    device: *mut iree_hal_device_t,
    queue_affinity: iree_hal_queue_affinity_t,
    out_backend: *mut iree_hal_queue_pool_backend_t,
) -> iree_status_t);

status_stub!(queue_alloca(
    device: *mut iree_hal_device_t,
    queue_affinity: iree_hal_queue_affinity_t,
    wait_semaphore_list: iree_hal_semaphore_list_t,
    signal_semaphore_list: iree_hal_semaphore_list_t,
    pool: *mut iree_hal_pool_t,
    params: iree_hal_buffer_params_t,
    allocation_size: iree_device_size_t,
    flags: iree_hal_alloca_flags_t,
    out_buffer: *mut *mut iree_hal_buffer_t,
) -> iree_status_t);

status_stub!(queue_dealloca(
    device: *mut iree_hal_device_t,
    queue_affinity: iree_hal_queue_affinity_t,
    wait_semaphore_list: iree_hal_semaphore_list_t,
    signal_semaphore_list: iree_hal_semaphore_list_t,
    buffer: *mut iree_hal_buffer_t,
    flags: iree_hal_dealloca_flags_t,
) -> iree_status_t);

status_stub!(queue_fill(
    device: *mut iree_hal_device_t,
    queue_affinity: iree_hal_queue_affinity_t,
    wait_semaphore_list: iree_hal_semaphore_list_t,
    signal_semaphore_list: iree_hal_semaphore_list_t,
    target_buffer: *mut iree_hal_buffer_t,
    target_offset: iree_device_size_t,
    length: iree_device_size_t,
    pattern: *const std::ffi::c_void,
    pattern_length: iree_host_size_t,
    flags: iree_hal_fill_flags_t,
) -> iree_status_t);

status_stub!(queue_update(
    device: *mut iree_hal_device_t,
    queue_affinity: iree_hal_queue_affinity_t,
    wait_semaphore_list: iree_hal_semaphore_list_t,
    signal_semaphore_list: iree_hal_semaphore_list_t,
    source_buffer: *const std::ffi::c_void,
    source_offset: iree_host_size_t,
    target_buffer: *mut iree_hal_buffer_t,
    target_offset: iree_device_size_t,
    length: iree_device_size_t,
    flags: iree_hal_update_flags_t,
) -> iree_status_t);

status_stub!(queue_copy(
    device: *mut iree_hal_device_t,
    queue_affinity: iree_hal_queue_affinity_t,
    wait_semaphore_list: iree_hal_semaphore_list_t,
    signal_semaphore_list: iree_hal_semaphore_list_t,
    source_buffer: *mut iree_hal_buffer_t,
    source_offset: iree_device_size_t,
    target_buffer: *mut iree_hal_buffer_t,
    target_offset: iree_device_size_t,
    length: iree_device_size_t,
    flags: iree_hal_copy_flags_t,
) -> iree_status_t);

status_stub!(queue_read(
    device: *mut iree_hal_device_t,
    queue_affinity: iree_hal_queue_affinity_t,
    wait_semaphore_list: iree_hal_semaphore_list_t,
    signal_semaphore_list: iree_hal_semaphore_list_t,
    source_file: *mut iree_hal_file_t,
    source_offset: u64,
    target_buffer: *mut iree_hal_buffer_t,
    target_offset: iree_device_size_t,
    length: iree_device_size_t,
    flags: iree_hal_read_flags_t,
) -> iree_status_t);

status_stub!(queue_write(
    device: *mut iree_hal_device_t,
    queue_affinity: iree_hal_queue_affinity_t,
    wait_semaphore_list: iree_hal_semaphore_list_t,
    signal_semaphore_list: iree_hal_semaphore_list_t,
    source_buffer: *mut iree_hal_buffer_t,
    source_offset: iree_device_size_t,
    target_file: *mut iree_hal_file_t,
    target_offset: u64,
    length: iree_device_size_t,
    flags: iree_hal_write_flags_t,
) -> iree_status_t);

status_stub!(queue_host_call(
    device: *mut iree_hal_device_t,
    queue_affinity: iree_hal_queue_affinity_t,
    wait_semaphore_list: iree_hal_semaphore_list_t,
    signal_semaphore_list: iree_hal_semaphore_list_t,
    call: iree_hal_host_call_t,
    args: *const u64,
    flags: iree_hal_host_call_flags_t,
) -> iree_status_t);

status_stub!(queue_dispatch(
    device: *mut iree_hal_device_t,
    queue_affinity: iree_hal_queue_affinity_t,
    wait_semaphore_list: iree_hal_semaphore_list_t,
    signal_semaphore_list: iree_hal_semaphore_list_t,
    executable: *mut iree_hal_executable_t,
    function: iree_hal_executable_function_t,
    config: iree_hal_dispatch_config_t,
    constants: iree_const_byte_span_t,
    bindings: iree_hal_buffer_ref_list_t,
    flags: iree_hal_dispatch_flags_t,
) -> iree_status_t);

// TODO: this is the one slot with real design intent already -- see the
// module doc comment. Wire up: SUBMIT via iree-rocket-hal's ioctl bindings,
// blocking PREP_BO wait (absolute CLOCK_MONOTONIC deadline -- see
// rknpu-spelunking/NOTES.md for why that matters), then signal
// signal_semaphore_list. Still UNIMPLEMENTED until a real device/command
// buffer type exists to operate on.
status_stub!(queue_execute(
    device: *mut iree_hal_device_t,
    queue_affinity: iree_hal_queue_affinity_t,
    wait_semaphore_list: iree_hal_semaphore_list_t,
    signal_semaphore_list: iree_hal_semaphore_list_t,
    command_buffer: *mut iree_hal_command_buffer_t,
    binding_table: iree_hal_buffer_binding_table_t,
    flags: iree_hal_execute_flags_t,
) -> iree_status_t);

status_stub!(queue_flush(
    device: *mut iree_hal_device_t,
    queue_affinity: iree_hal_queue_affinity_t,
) -> iree_status_t);

status_stub!(profiling_begin(
    device: *mut iree_hal_device_t,
    options: *const iree_hal_device_profiling_options_t,
) -> iree_status_t);

status_stub!(profiling_flush(device: *mut iree_hal_device_t) -> iree_status_t);
status_stub!(profiling_end(device: *mut iree_hal_device_t) -> iree_status_t);

status_stub!(external_capture_begin(
    device: *mut iree_hal_device_t,
    options: *const iree_hal_device_external_capture_options_t,
) -> iree_status_t);

status_stub!(external_capture_end(device: *mut iree_hal_device_t) -> iree_status_t);

pub static VTABLE: iree_hal_device_vtable_t = iree_hal_device_vtable_t {
    destroy: Some(destroy),
    id: Some(id),
    host_allocator: Some(host_allocator),
    device_allocator: Some(device_allocator),
    replace_device_allocator: Some(replace_device_allocator),
    replace_channel_provider: Some(replace_channel_provider),
    trim: Some(trim),
    query_i64: Some(query_i64),
    query_capabilities: Some(query_capabilities),
    topology_info: Some(topology_info),
    refine_topology_edge: Some(refine_topology_edge),
    assign_topology_info: Some(assign_topology_info),
    create_channel: Some(create_channel),
    create_command_buffer: Some(create_command_buffer),
    create_event: Some(create_event),
    create_executable_cache: Some(create_executable_cache),
    import_file: Some(import_file),
    create_semaphore: Some(create_semaphore),
    query_semaphore_compatibility: Some(query_semaphore_compatibility),
    query_queue_pool_backend: Some(query_queue_pool_backend),
    queue_alloca: Some(queue_alloca),
    queue_dealloca: Some(queue_dealloca),
    queue_fill: Some(queue_fill),
    queue_update: Some(queue_update),
    queue_copy: Some(queue_copy),
    queue_read: Some(queue_read),
    queue_write: Some(queue_write),
    queue_host_call: Some(queue_host_call),
    queue_dispatch: Some(queue_dispatch),
    queue_execute: Some(queue_execute),
    queue_flush: Some(queue_flush),
    profiling_begin: Some(profiling_begin),
    profiling_flush: Some(profiling_flush),
    profiling_end: Some(profiling_end),
    external_capture_begin: Some(external_capture_begin),
    external_capture_end: Some(external_capture_end),
};
