//! `iree_hal_executable_vtable_t`. For this driver an "executable" isn't
//! compiled machine code -- it's a stored regcmd program (or, before real
//! MLIR codegen exists, a hardcoded `ConvShape` -- see
//! `executable_cache.rs`'s module doc comment for the staging plan).

use crate::bindings::{
    iree_hal_executable_function_info_t, iree_hal_executable_function_parameter_t,
    iree_hal_executable_function_t, iree_hal_executable_t, iree_hal_executable_vtable_t,
    iree_hal_buffer_t, iree_hal_queue_affinity_t, iree_host_size_t,
    iree_string_view_t,
};

void_stub!(destroy(executable: *mut iree_hal_executable_t));

#[allow(unused_variables)]
pub unsafe extern "C" fn function_count(executable: *mut iree_hal_executable_t) -> iree_host_size_t {
    0
}

status_stub!(function_info(
    executable: *mut iree_hal_executable_t,
    function: iree_hal_executable_function_t,
    out_info: *mut iree_hal_executable_function_info_t,
) -> iree_status_t);

status_stub!(function_parameters(
    executable: *mut iree_hal_executable_t,
    function: iree_hal_executable_function_t,
    capacity: iree_host_size_t,
    out_parameters: *mut iree_hal_executable_function_parameter_t,
) -> iree_status_t);

status_stub!(lookup_function_by_name(
    executable: *mut iree_hal_executable_t,
    name: iree_string_view_t,
    out_function: *mut iree_hal_executable_function_t,
) -> iree_status_t);

status_stub!(lookup_global_by_name(
    executable: *mut iree_hal_executable_t,
    name: iree_string_view_t,
    queue_affinity: iree_hal_queue_affinity_t,
    out_buffer: *mut *mut iree_hal_buffer_t,
) -> iree_status_t);

pub static VTABLE: iree_hal_executable_vtable_t = iree_hal_executable_vtable_t {
    destroy: Some(destroy),
    function_count: Some(function_count),
    function_info: Some(function_info),
    function_parameters: Some(function_parameters),
    lookup_function_by_name: Some(lookup_function_by_name),
    lookup_global_by_name: Some(lookup_global_by_name),
};
