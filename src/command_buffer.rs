//! `iree_hal_command_buffer_vtable_t`. `dispatch` is the one slot with
//! real design intent already: this is where a recorded conv op's shape
//! gets turned into a regcmd program via `iree_rocket_hal::rocket::regcmd::
//! build_conv_regcmd` (see NOTES.md in rknpu-spelunking for how that
//! function was derived/validated) and appended to the command buffer's
//! recorded state, to later be submitted as one job in `device::
//! queue_execute`. `collective` (multi-device reduce/broadcast/etc.) isn't
//! applicable to a single discrete NPU and stays UNIMPLEMENTED
//! indefinitely, not just for now.

use crate::bindings::{
    iree_const_byte_span_t, iree_hal_buffer_barrier_t, iree_hal_buffer_ref_list_t,
    iree_hal_buffer_ref_t, iree_hal_channel_t, iree_hal_collective_op_t,
    iree_hal_command_buffer_t, iree_hal_command_buffer_vtable_t, iree_hal_copy_flags_t,
    iree_hal_dispatch_config_t, iree_hal_dispatch_flags_t, iree_hal_event_t,
    iree_hal_execution_barrier_flags_t, iree_hal_execution_stage_t, iree_hal_executable_function_t,
    iree_hal_executable_t, iree_hal_fill_flags_t, iree_hal_label_color_t,
    iree_hal_label_location_t, iree_hal_memory_advise_flags_t, iree_hal_memory_barrier_t,
    iree_hal_update_flags_t, iree_device_size_t, iree_host_size_t,
    iree_string_view_t,
};

void_stub!(destroy(command_buffer: *mut iree_hal_command_buffer_t));

status_stub!(begin(command_buffer: *mut iree_hal_command_buffer_t) -> iree_status_t);
status_stub!(end(command_buffer: *mut iree_hal_command_buffer_t) -> iree_status_t);

status_stub!(begin_debug_group(
    command_buffer: *mut iree_hal_command_buffer_t,
    label: iree_string_view_t,
    label_color: iree_hal_label_color_t,
    location: *const iree_hal_label_location_t,
) -> iree_status_t);

status_stub!(end_debug_group(command_buffer: *mut iree_hal_command_buffer_t) -> iree_status_t);

status_stub!(execution_barrier(
    command_buffer: *mut iree_hal_command_buffer_t,
    source_stage_mask: iree_hal_execution_stage_t,
    target_stage_mask: iree_hal_execution_stage_t,
    flags: iree_hal_execution_barrier_flags_t,
    memory_barrier_count: iree_host_size_t,
    memory_barriers: *const iree_hal_memory_barrier_t,
    buffer_barrier_count: iree_host_size_t,
    buffer_barriers: *const iree_hal_buffer_barrier_t,
) -> iree_status_t);

status_stub!(signal_event(
    command_buffer: *mut iree_hal_command_buffer_t,
    event: *mut iree_hal_event_t,
    source_stage_mask: iree_hal_execution_stage_t,
) -> iree_status_t);

status_stub!(reset_event(
    command_buffer: *mut iree_hal_command_buffer_t,
    event: *mut iree_hal_event_t,
    source_stage_mask: iree_hal_execution_stage_t,
) -> iree_status_t);

status_stub!(wait_events(
    command_buffer: *mut iree_hal_command_buffer_t,
    event_count: iree_host_size_t,
    events: *mut *const iree_hal_event_t,
    source_stage_mask: iree_hal_execution_stage_t,
    target_stage_mask: iree_hal_execution_stage_t,
    memory_barrier_count: iree_host_size_t,
    memory_barriers: *const iree_hal_memory_barrier_t,
    buffer_barrier_count: iree_host_size_t,
    buffer_barriers: *const iree_hal_buffer_barrier_t,
) -> iree_status_t);

status_stub!(advise_buffer(
    command_buffer: *mut iree_hal_command_buffer_t,
    buffer_ref: iree_hal_buffer_ref_t,
    flags: iree_hal_memory_advise_flags_t,
    arg0: u64,
    arg1: u64,
) -> iree_status_t);

status_stub!(fill_buffer(
    command_buffer: *mut iree_hal_command_buffer_t,
    target_ref: iree_hal_buffer_ref_t,
    pattern: *const std::ffi::c_void,
    pattern_length: iree_host_size_t,
    flags: iree_hal_fill_flags_t,
) -> iree_status_t);

status_stub!(update_buffer(
    command_buffer: *mut iree_hal_command_buffer_t,
    source_buffer: *const std::ffi::c_void,
    source_offset: iree_host_size_t,
    target_ref: iree_hal_buffer_ref_t,
    flags: iree_hal_update_flags_t,
) -> iree_status_t);

status_stub!(copy_buffer(
    command_buffer: *mut iree_hal_command_buffer_t,
    source_ref: iree_hal_buffer_ref_t,
    target_ref: iree_hal_buffer_ref_t,
    flags: iree_hal_copy_flags_t,
) -> iree_status_t);

status_stub!(collective(
    command_buffer: *mut iree_hal_command_buffer_t,
    channel: *mut iree_hal_channel_t,
    op: iree_hal_collective_op_t,
    param: u32,
    send_ref: iree_hal_buffer_ref_t,
    recv_ref: iree_hal_buffer_ref_t,
    element_count: iree_device_size_t,
) -> iree_status_t);

// TODO: the real one -- this is where build_conv_regcmd() gets called.
// See module doc comment.
status_stub!(dispatch(
    command_buffer: *mut iree_hal_command_buffer_t,
    executable: *mut iree_hal_executable_t,
    function: iree_hal_executable_function_t,
    config: iree_hal_dispatch_config_t,
    constants: iree_const_byte_span_t,
    bindings: iree_hal_buffer_ref_list_t,
    flags: iree_hal_dispatch_flags_t,
) -> iree_status_t);

pub static VTABLE: iree_hal_command_buffer_vtable_t = iree_hal_command_buffer_vtable_t {
    destroy: Some(destroy),
    begin: Some(begin),
    end: Some(end),
    begin_debug_group: Some(begin_debug_group),
    end_debug_group: Some(end_debug_group),
    execution_barrier: Some(execution_barrier),
    signal_event: Some(signal_event),
    reset_event: Some(reset_event),
    wait_events: Some(wait_events),
    advise_buffer: Some(advise_buffer),
    fill_buffer: Some(fill_buffer),
    update_buffer: Some(update_buffer),
    copy_buffer: Some(copy_buffer),
    collective: Some(collective),
    dispatch: Some(dispatch),
};
