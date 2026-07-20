//! `iree_hal_command_buffer_vtable_t`. `dispatch` is the real regcmd
//! integration point: turns the executable's `ConvShape`
//! (`executable::shape`) plus this dispatch's buffer bindings into a
//! regcmd program via `iree_rocket_hal::rocket::regcmd::build_conv_regcmd`
//! and stashes it on the command buffer, to be submitted as one job by
//! `device::queue_execute`.
//!
//! Binding convention (since there's no real compiler emitting a binding
//! layout yet -- see executable.rs's module doc comment): binding 0 =
//! input, 1 = weights, 2 = bias, 3 = output. Entirely a placeholder of our
//! own choosing, not derived from anything.
//!
//! Only supports recording exactly one `dispatch` per command buffer right
//! now (matches `build_conv_regcmd`'s own single-task-only scope, see
//! regcmd.rs) -- a second `dispatch` call returns
//! `IREE_STATUS_UNIMPLEMENTED` rather than silently overwriting or
//! chaining tasks incorrectly. `collective` (multi-device reduce/
//! broadcast/etc.) isn't applicable to a single discrete NPU and stays
//! UNIMPLEMENTED indefinitely, not just for now.

use crate::bindings::{
    iree_const_byte_span_t, iree_device_size_t, iree_hal_buffer_barrier_t,
    iree_hal_buffer_ref_list_t, iree_hal_buffer_ref_t, iree_hal_channel_t,
    iree_hal_collective_op_t, iree_hal_command_buffer_mode_t, iree_hal_command_buffer_t,
    iree_hal_command_buffer_vtable_t, iree_hal_command_category_t, iree_hal_copy_flags_t,
    iree_hal_dispatch_config_t, iree_hal_dispatch_flags_t, iree_hal_event_t,
    iree_hal_executable_function_t, iree_hal_executable_t, iree_hal_execution_barrier_flags_t,
    iree_hal_execution_stage_t, iree_hal_fill_flags_t, iree_hal_label_color_t,
    iree_hal_label_location_t, iree_hal_memory_advise_flags_t, iree_hal_memory_barrier_t,
    iree_hal_queue_affinity_t, iree_hal_update_flags_t, iree_host_size_t, iree_status_t,
    iree_string_view_t,
};
use crate::buffer::RocketBuffer;
use crate::status;
use iree_rocket_hal::rocket::{
    builders::RegCmd,
    regcmd::{ConvBuffers, build_conv_regcmd},
};

/// What every `iree_hal_command_buffer_t*` this driver hands out actually
/// points to. `base` (the real, fully-defined `iree_hal_command_buffer_t`,
/// filled via `iree_hal_command_buffer_initialize`) must be the first
/// field, matching `buffer::RocketBuffer`'s convention.
#[repr(C)]
pub struct RocketCommandBuffer {
    pub base: iree_hal_command_buffer_t,
    /// Set by the one `dispatch` call this command buffer supports. `None`
    /// until then; `device::queue_execute` reads it back out.
    pub regcmd: Option<Vec<RegCmd>>,
}

unsafe fn cast(command_buffer: *mut iree_hal_command_buffer_t) -> *mut RocketCommandBuffer {
    command_buffer as *mut RocketCommandBuffer
}

/// Not part of the vtable -- `device::queue_execute` calls this directly
/// to get at the recorded regcmd program.
pub unsafe fn regcmd(command_buffer: *mut iree_hal_command_buffer_t) -> Option<&'static [RegCmd]> {
    unsafe { (*cast(command_buffer)).regcmd.as_deref() }
}

pub unsafe fn create(
    device_allocator: *mut crate::bindings::iree_hal_allocator_t,
    mode: iree_hal_command_buffer_mode_t,
    command_categories: iree_hal_command_category_t,
    queue_affinity: iree_hal_queue_affinity_t,
    binding_capacity: iree_host_size_t,
) -> *mut iree_hal_command_buffer_t {
    let cb = Box::new(RocketCommandBuffer {
        base: unsafe { std::mem::zeroed() }, // filled by iree_hal_command_buffer_initialize below
        regcmd: None,
    });
    let cb_ptr = Box::into_raw(cb);
    unsafe {
        crate::bindings::iree_hal_command_buffer_initialize(
            device_allocator,
            mode,
            command_categories,
            queue_affinity,
            binding_capacity,
            std::ptr::null_mut(), // validation_state -- not using IREE's validation utils yet
            &VTABLE,
            &mut (*cb_ptr).base,
        );
    }
    cb_ptr as *mut iree_hal_command_buffer_t
}

unsafe extern "C" fn destroy(command_buffer: *mut iree_hal_command_buffer_t) {
    unsafe { drop(Box::from_raw(cast(command_buffer))) }
}

// Real no-ops (not status_stub -- that returns UNIMPLEMENTED, which would
// break every real caller that expects begin/end to just work).
#[allow(unused_variables)]
unsafe extern "C" fn begin(command_buffer: *mut iree_hal_command_buffer_t) -> iree_status_t {
    status::ok()
}
#[allow(unused_variables)]
unsafe extern "C" fn end(command_buffer: *mut iree_hal_command_buffer_t) -> iree_status_t {
    status::ok()
}

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

#[allow(unused_variables)]
unsafe extern "C" fn dispatch(
    command_buffer: *mut iree_hal_command_buffer_t,
    executable: *mut iree_hal_executable_t,
    function: iree_hal_executable_function_t,
    config: iree_hal_dispatch_config_t,
    constants: iree_const_byte_span_t,
    bindings: iree_hal_buffer_ref_list_t,
    flags: iree_hal_dispatch_flags_t,
) -> iree_status_t {
    let cb = unsafe { &mut *cast(command_buffer) };
    if cb.regcmd.is_some() {
        // Only one dispatch per command buffer supported -- see module doc
        // comment.
        return status::unimplemented();
    }
    if bindings.count < 4 {
        return status::from_code(
            crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT as u32,
        );
    }

    let shape = unsafe { &*crate::executable::shape(executable) };
    let refs = unsafe { std::slice::from_raw_parts(bindings.values, bindings.count as usize) };
    let addr =
        |r: &iree_hal_buffer_ref_t| unsafe { (*(r.buffer as *mut RocketBuffer)).dma_address };

    let bufs = ConvBuffers {
        input_addr: addr(&refs[0]),
        weights_addr: addr(&refs[1]),
        bias_addr: addr(&refs[2]),
        output_addr: addr(&refs[3]),
    };
    cb.regcmd = Some(build_conv_regcmd(shape, &bufs));
    status::ok()
}

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
