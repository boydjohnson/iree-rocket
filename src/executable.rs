//! `iree_hal_executable_vtable_t`. For this driver an "executable" isn't
//! compiled machine code -- it's a stored `ConvShape`
//! (`iree_rocket_hal::rocket::regcmd::ConvShape`). No real MLIR codegen
//! target exists for this hardware yet (see the `custom_dispatch` research
//! this crate started from -- that mechanism doesn't fit a regcmd-bitstream
//! device), so `executable_cache::prepare_executable` currently produces
//! exactly one hardcoded shape regardless of what `executable_data` says --
//! a deliberate placeholder, not a real executable-format parser.

use crate::bindings::{
    iree_hal_buffer_t, iree_hal_executable_function_info_t,
    iree_hal_executable_function_parameter_t, iree_hal_executable_function_t,
    iree_hal_executable_t, iree_hal_executable_vtable_t, iree_hal_queue_affinity_t,
    iree_hal_resource_t, iree_host_size_t, iree_status_t, iree_string_view_t,
};
use crate::status;
use iree_rocket_hal::rocket::regcmd::ConvShape;

/// What every `iree_hal_executable_t*` this driver hands out actually
/// points to. `iree_hal_executable_t` is opaque (no public field
/// definition), so `resource` is the real base-at-offset-0 field.
#[repr(C)]
pub struct RocketExecutable {
    pub resource: iree_hal_resource_t,
    /// Exactly one "function" (ordinal 0) -- the hardcoded shape. A real
    /// executable format would carry N functions/entry points; this
    /// placeholder only ever has one.
    pub shape: ConvShape,
}

unsafe fn cast(executable: *mut iree_hal_executable_t) -> *mut RocketExecutable {
    executable as *mut RocketExecutable
}

pub fn create(shape: ConvShape) -> *mut iree_hal_executable_t {
    let executable = Box::new(RocketExecutable {
        resource: iree_hal_resource_t {
            ref_count: 1,
            vtable: &VTABLE as *const _ as *const std::ffi::c_void,
        },
        shape,
    });
    Box::into_raw(executable) as *mut iree_hal_executable_t
}

/// Not part of the vtable -- `command_buffer::dispatch` calls this
/// directly to get at the shape it needs for `build_conv_regcmd`.
pub unsafe fn shape(executable: *mut iree_hal_executable_t) -> *const ConvShape {
    unsafe { &(*cast(executable)).shape }
}

unsafe extern "C" fn destroy(executable: *mut iree_hal_executable_t) {
    unsafe { drop(Box::from_raw(cast(executable))) }
}

#[allow(unused_variables)]
unsafe extern "C" fn function_count(executable: *mut iree_hal_executable_t) -> iree_host_size_t {
    1
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

#[allow(unused_variables)]
unsafe extern "C" fn lookup_function_by_name(
    executable: *mut iree_hal_executable_t,
    name: iree_string_view_t,
    out_function: *mut iree_hal_executable_function_t,
) -> iree_status_t {
    // Only one function (ordinal 0) exists -- see module doc comment.
    unsafe {
        (*out_function).value = 0;
    }
    status::ok()
}

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
