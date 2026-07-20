//! `iree_hal_device_vtable_t`. `create` opens `/dev/accel/accel0` and
//! wires up the sub-objects (allocator, proactor-from-pool); `queue_execute`
//! is the real dispatch path: wait on `wait_semaphore_list`, pull the
//! regcmd program `command_buffer::dispatch` recorded, write it to a GEM
//! buffer, `SUBMIT`, blocking `PREP_BO`, then signal
//! `signal_semaphore_list` -- the synchronous pattern from
//! `local_sync/sync_device.c` that this crate's research phase identified
//! as the right model for a driver whose only completion signal is a
//! blocking ioctl rather than a native timeline/fence primitive.

use std::os::fd::AsRawFd;

use crate::bindings::{
    iree_allocator_t, iree_const_byte_span_t, iree_device_size_t, iree_hal_alloca_flags_t,
    iree_hal_allocator_t, iree_hal_buffer_binding_table_t, iree_hal_buffer_params_t,
    iree_hal_buffer_ref_list_t, iree_hal_buffer_t, iree_hal_channel_params_t,
    iree_hal_channel_provider_t, iree_hal_channel_t, iree_hal_command_buffer_mode_t,
    iree_hal_command_buffer_t, iree_hal_command_category_t, iree_hal_copy_flags_t,
    iree_hal_dealloca_flags_t, iree_hal_device_capabilities_t, iree_hal_device_create_params_t,
    iree_hal_device_external_capture_options_t, iree_hal_device_profiling_options_t,
    iree_hal_device_t, iree_hal_device_topology_info_t, iree_hal_device_vtable_t,
    iree_hal_dispatch_config_t, iree_hal_dispatch_flags_t, iree_hal_event_flags_t,
    iree_hal_event_t, iree_hal_executable_cache_t, iree_hal_executable_function_t,
    iree_hal_executable_t, iree_hal_execute_flags_t, iree_hal_external_file_flags_t,
    iree_hal_file_t, iree_hal_fill_flags_t, iree_hal_host_call_flags_t, iree_hal_host_call_t,
    iree_hal_memory_access_t, iree_hal_pool_t, iree_hal_queue_affinity_t,
    iree_hal_queue_pool_backend_t, iree_hal_read_flags_t, iree_hal_resource_t,
    iree_hal_semaphore_compatibility_bits_t_IREE_HAL_SEMAPHORE_COMPATIBILITY_ALL,
    iree_hal_semaphore_compatibility_t, iree_hal_semaphore_flags_t, iree_hal_semaphore_list_t,
    iree_hal_semaphore_t, iree_hal_topology_edge_t, iree_hal_update_flags_t,
    iree_hal_write_flags_t, iree_host_size_t, iree_io_file_handle_t,
    iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT, iree_status_code_e_IREE_STATUS_UNAVAILABLE,
    iree_status_t, iree_string_view_t, iree_timeout_t, iree_timeout_type_e_IREE_TIMEOUT_ABSOLUTE,
};
use crate::status;
use iree_rocket_hal::rocket::device as rocket_device;

const DEVICE_PATH: &str = "/dev/accel/accel0";

/// What every `iree_hal_device_t*` this driver hands out actually points
/// to. Opaque base type, `resource` at offset 0.
#[repr(C)]
pub struct RocketDevice {
    pub resource: iree_hal_resource_t,
    pub host_allocator: iree_allocator_t,
    pub identifier: Vec<u8>,
    pub file: std::fs::File,
    pub device_allocator: *mut iree_hal_allocator_t,
    pub proactor_pool: *mut crate::bindings::iree_async_proactor_pool_t,
    pub proactor: *mut crate::bindings::iree_async_proactor_t,
    /// Zeroed (not part of a topology) until `assign_topology_info` is
    /// called by `iree_hal_device_group_builder_finalize` -- mirrors
    /// iree-null-driver-reference/device.c's `topology_info` field. Does
    /// NOT retain/register with `topology_info.frontier.tracker` (unlike
    /// the null driver) -- frontier_tracker.h isn't in this crate's
    /// bindgen allowlist (see build.rs) and nothing exercised by this
    /// driver today needs cross-device causal tracking. Revisit if/when
    /// multi-device topologies or real frontier-based sync matter.
    pub topology_info: crate::bindings::iree_hal_device_topology_info_t,
}

unsafe fn cast(device: *mut iree_hal_device_t) -> *mut RocketDevice {
    device as *mut RocketDevice
}

/// Mirrors `iree_hal_null_device_create()`. `identifier` is borrowed only
/// for the duration of this call (copied into the device's own storage).
pub unsafe fn create(
    identifier: &[u8],
    create_params: *const iree_hal_device_create_params_t,
    host_allocator: iree_allocator_t,
    out_device: *mut *mut iree_hal_device_t,
) -> iree_status_t {
    if create_params.is_null() {
        return status::from_code(iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT as u32);
    }
    let proactor_pool = unsafe { (*create_params).proactor_pool };
    if proactor_pool.is_null() {
        // Real requirement, not a relaxation this driver invented -- see
        // module doc comment / iree-null-driver-reference/device.c's own
        // IREE_ASSERT_ARGUMENT(create_params->proactor_pool).
        return status::from_code(iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT as u32);
    }

    let file = match std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
    {
        Ok(f) => f,
        Err(_) => return status::from_code(iree_status_code_e_IREE_STATUS_UNAVAILABLE as u32),
    };
    let allocator_file = match file.try_clone() {
        Ok(f) => f,
        Err(_) => return status::from_code(iree_status_code_e_IREE_STATUS_UNAVAILABLE as u32),
    };

    unsafe {
        crate::bindings::iree_async_proactor_pool_retain(proactor_pool);
    }
    let mut proactor: *mut crate::bindings::iree_async_proactor_t = std::ptr::null_mut();
    let proactor_status =
        unsafe { crate::bindings::iree_async_proactor_pool_get(proactor_pool, 0, &mut proactor) };
    if !proactor_status.is_null() {
        unsafe {
            crate::bindings::iree_async_proactor_pool_release(proactor_pool);
        }
        return proactor_status;
    }

    let device_allocator = crate::allocator::create(allocator_file, host_allocator);

    let device = Box::new(RocketDevice {
        resource: iree_hal_resource_t {
            ref_count: 1,
            vtable: &VTABLE as *const _ as *const std::ffi::c_void,
        },
        host_allocator,
        identifier: identifier.to_vec(),
        file,
        device_allocator,
        proactor_pool,
        proactor,
        topology_info: unsafe { std::mem::zeroed() },
    });
    unsafe {
        *out_device = Box::into_raw(device) as *mut iree_hal_device_t;
    }
    status::ok()
}

unsafe extern "C" fn destroy(device: *mut iree_hal_device_t) {
    unsafe {
        let d = &*cast(device);
        crate::bindings::iree_hal_allocator_release(d.device_allocator);
        crate::bindings::iree_async_proactor_pool_release(d.proactor_pool);
        drop(Box::from_raw(cast(device)));
    }
}

unsafe extern "C" fn id(device: *mut iree_hal_device_t) -> iree_string_view_t {
    let d = unsafe { &*cast(device) };
    iree_string_view_t {
        data: d.identifier.as_ptr() as *const std::os::raw::c_char,
        size: d.identifier.len(),
    }
}

unsafe extern "C" fn host_allocator(device: *mut iree_hal_device_t) -> iree_allocator_t {
    unsafe { (*cast(device)).host_allocator }
}

unsafe extern "C" fn device_allocator(device: *mut iree_hal_device_t) -> *mut iree_hal_allocator_t {
    // Not retained -- matches iree-null-driver-reference/device.c's own
    // iree_hal_null_device_allocator (caller borrows it).
    unsafe { (*cast(device)).device_allocator }
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

#[allow(unused_variables)]
unsafe extern "C" fn query_capabilities(
    device: *mut iree_hal_device_t,
    out_capabilities: *mut iree_hal_device_capabilities_t,
) -> iree_status_t {
    // Zeroed = no special hardware capabilities declared yet -- mirrors
    // iree-null-driver-reference/device.c's query_capabilities exactly.
    // Enough for iree_hal_device_group_builder_finalize() to build a
    // single-device topology (numa_node=0, no import/export types) without
    // UNIMPLEMENTED bubbling up and leaving the CTS's cached device_group
    // half-built.
    unsafe {
        *out_capabilities = std::mem::zeroed();
    }
    status::ok()
}

unsafe extern "C" fn topology_info(
    device: *mut iree_hal_device_t,
) -> *const iree_hal_device_topology_info_t {
    let d = unsafe { &*cast(device) };
    &d.topology_info
}

#[allow(unused_variables)]
unsafe extern "C" fn refine_topology_edge(
    src_device: *mut iree_hal_device_t,
    dst_device: *mut iree_hal_device_t,
    edge: *mut iree_hal_topology_edge_t,
) -> iree_status_t {
    // Only called for same-driver device pairs; rocket has no multi-device
    // interconnect knowledge to contribute yet, so the capability-derived
    // edge stands as-is -- matches iree-null-driver-reference exactly.
    status::ok()
}

unsafe extern "C" fn assign_topology_info(
    device: *mut iree_hal_device_t,
    topology_info: *const iree_hal_device_topology_info_t,
) -> iree_status_t {
    let d = unsafe { &mut *cast(device) };
    d.topology_info = if topology_info.is_null() {
        unsafe { std::mem::zeroed() }
    } else {
        unsafe { std::ptr::read(topology_info) }
    };
    status::ok()
}

status_stub!(create_channel(
    device: *mut iree_hal_device_t,
    queue_affinity: iree_hal_queue_affinity_t,
    params: iree_hal_channel_params_t,
    out_channel: *mut *mut iree_hal_channel_t,
) -> iree_status_t);

#[allow(unused_variables)]
unsafe extern "C" fn create_command_buffer(
    device: *mut iree_hal_device_t,
    mode: iree_hal_command_buffer_mode_t,
    command_categories: iree_hal_command_category_t,
    queue_affinity: iree_hal_queue_affinity_t,
    binding_capacity: iree_host_size_t,
    out_command_buffer: *mut *mut iree_hal_command_buffer_t,
) -> iree_status_t {
    let d = unsafe { &*cast(device) };
    unsafe {
        *out_command_buffer = crate::command_buffer::create(
            d.device_allocator,
            mode,
            command_categories,
            queue_affinity,
            binding_capacity,
        );
    }
    status::ok()
}

status_stub!(create_event(
    device: *mut iree_hal_device_t,
    queue_affinity: iree_hal_queue_affinity_t,
    flags: iree_hal_event_flags_t,
    out_event: *mut *mut iree_hal_event_t,
) -> iree_status_t);

#[allow(unused_variables)]
unsafe extern "C" fn create_executable_cache(
    device: *mut iree_hal_device_t,
    identifier: iree_string_view_t,
    out_executable_cache: *mut *mut iree_hal_executable_cache_t,
) -> iree_status_t {
    unsafe {
        *out_executable_cache = crate::executable_cache::create();
    }
    status::ok()
}

status_stub!(import_file(
    device: *mut iree_hal_device_t,
    queue_affinity: iree_hal_queue_affinity_t,
    access: iree_hal_memory_access_t,
    handle: *mut iree_io_file_handle_t,
    flags: iree_hal_external_file_flags_t,
    out_file: *mut *mut iree_hal_file_t,
) -> iree_status_t);

#[allow(unused_variables)]
unsafe extern "C" fn create_semaphore(
    device: *mut iree_hal_device_t,
    queue_affinity: iree_hal_queue_affinity_t,
    initial_value: u64,
    flags: iree_hal_semaphore_flags_t,
    out_semaphore: *mut *mut iree_hal_semaphore_t,
) -> iree_status_t {
    let d = unsafe { &*cast(device) };
    unsafe {
        *out_semaphore = crate::semaphore::create(d.proactor, initial_value, d.host_allocator);
    }
    status::ok()
}

#[allow(unused_variables)]
unsafe extern "C" fn query_semaphore_compatibility(
    device: *mut iree_hal_device_t,
    semaphore: *mut iree_hal_semaphore_t,
) -> iree_hal_semaphore_compatibility_t {
    // We don't yet distinguish semaphores created by this device from
    // ones created elsewhere (no cross-driver import support) -- assume
    // full compatibility, matching every semaphore this driver could
    // plausibly be handed today (it's the only HAL driver in the process
    // in any realistic near-term usage).
    iree_hal_semaphore_compatibility_bits_t_IREE_HAL_SEMAPHORE_COMPATIBILITY_ALL
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

#[allow(unused_variables)]
unsafe extern "C" fn queue_execute(
    device: *mut iree_hal_device_t,
    queue_affinity: iree_hal_queue_affinity_t,
    wait_semaphore_list: iree_hal_semaphore_list_t,
    signal_semaphore_list: iree_hal_semaphore_list_t,
    command_buffer: *mut iree_hal_command_buffer_t,
    binding_table: iree_hal_buffer_binding_table_t,
    flags: iree_hal_execute_flags_t,
) -> iree_status_t {
    let d = unsafe { &*cast(device) };

    let infinite = iree_timeout_t {
        type_: iree_timeout_type_e_IREE_TIMEOUT_ABSOLUTE,
        nanos: i64::MAX, // IREE_TIME_INFINITE_FUTURE
    };
    unsafe {
        for i in 0..wait_semaphore_list.count as isize {
            let sem = *wait_semaphore_list.semaphores.offset(i);
            let value = *wait_semaphore_list.payload_values.offset(i);
            let st = crate::bindings::iree_hal_semaphore_wait(sem, value, infinite, 0);
            if !st.is_null() {
                return st;
            }
        }
    }

    let cmds = match unsafe { crate::command_buffer::regcmd(command_buffer) } {
        Some(c) if !c.is_empty() => c,
        _ => return status::from_code(iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT as u32),
    };

    let fd = d.file.as_raw_fd();
    let cmd_bytes = cmds.len() * std::mem::size_of::<u64>();
    let cmd_len = cmd_bytes.next_multiple_of(4096);
    let cmd_buf = unsafe { rocket_device::Buffer::new(fd, cmd_len, &d.file) };
    unsafe {
        let cmd_slice = std::slice::from_raw_parts_mut(cmd_buf.host_ptr as *mut u64, cmds.len());
        for (i, c) in cmds.iter().enumerate() {
            cmd_slice[i] = c.0;
        }
    }
    if unsafe { rocket_device::fini_bo(fd, cmd_buf.handle) }.is_err() {
        return status::from_code(iree_status_code_e_IREE_STATUS_UNAVAILABLE as u32);
    }

    // TODO: real input/output GEM handles from the command buffer's
    // recorded buffer bindings -- for now this only knows the regcmd
    // buffer's own handle, which is enough to prove SUBMIT/PREP_BO
    // round-trips but not enough for a real dispatch to complete
    // (matches this being an early, hardcoded-shape milestone -- see
    // executable.rs/command_buffer.rs's doc comments).
    let in_handles = [cmd_buf.handle];
    let out_handles: [u32; 0] = [];
    if unsafe {
        rocket_device::submit(
            fd,
            cmd_buf.dma_address,
            cmds.len() as u32,
            &in_handles,
            &out_handles,
        )
    }
    .is_err()
    {
        return status::from_code(iree_status_code_e_IREE_STATUS_UNAVAILABLE as u32);
    }

    if unsafe { rocket_device::prep_bo(fd, cmd_buf.handle, 2_000_000_000) }.is_err() {
        return status::from_code(iree_status_code_e_IREE_STATUS_UNAVAILABLE as u32);
    }

    unsafe {
        for i in 0..signal_semaphore_list.count as isize {
            let sem = *signal_semaphore_list.semaphores.offset(i);
            let value = *signal_semaphore_list.payload_values.offset(i);
            let st = crate::bindings::iree_hal_semaphore_signal(sem, value, std::ptr::null());
            if !st.is_null() {
                return st;
            }
        }
    }
    status::ok()
}

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
