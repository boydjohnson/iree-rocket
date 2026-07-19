//! `iree_hal_executable_cache_vtable_t`. `prepare_executable` is where real
//! integration work starts once the skeleton above is filled in: rather
//! than loading compiled code (what every real IREE HAL driver's
//! executable cache does), this needs to recognize a small,
//! driver-specific "executable format" -- initially just a hardcoded
//! `ConvShape` (see `iree_rocket_hal::rocket::regcmd::ConvShape`) rather
//! than anything IREE's compiler emits, since there's no MLIR codegen
//! target for this hardware yet (see the custom_dispatch research this
//! crate started from -- that mechanism doesn't fit a regcmd-bitstream
//! device, so this will need its own bespoke executable format, likely
//! IREE's `hal.executable.objects` data-carrying path rather than
//! precompiled code).

use crate::bindings::{
    iree_const_byte_span_t, iree_hal_executable_caching_mode_t, iree_hal_executable_cache_t,
    iree_hal_executable_cache_vtable_t, iree_hal_executable_params_t, iree_hal_executable_t,
    iree_host_size_t, iree_string_view_t,
};

void_stub!(destroy(executable_cache: *mut iree_hal_executable_cache_t));

status_stub!(infer_format(
    executable_cache: *mut iree_hal_executable_cache_t,
    caching_mode: iree_hal_executable_caching_mode_t,
    executable_data: iree_const_byte_span_t,
    executable_format_capacity: iree_host_size_t,
    executable_format: *mut std::os::raw::c_char,
    out_inferred_size: *mut iree_host_size_t,
) -> iree_status_t);

#[allow(unused_variables)]
pub unsafe extern "C" fn can_prepare_format(
    executable_cache: *mut iree_hal_executable_cache_t,
    caching_mode: iree_hal_executable_caching_mode_t,
    executable_format: iree_string_view_t,
) -> bool {
    false
}

// TODO: the real one -- see module doc comment.
status_stub!(prepare_executable(
    executable_cache: *mut iree_hal_executable_cache_t,
    executable_params: *const iree_hal_executable_params_t,
    out_executable: *mut *mut iree_hal_executable_t,
) -> iree_status_t);

pub static VTABLE: iree_hal_executable_cache_vtable_t = iree_hal_executable_cache_vtable_t {
    destroy: Some(destroy),
    infer_format: Some(infer_format),
    can_prepare_format: Some(can_prepare_format),
    prepare_executable: Some(prepare_executable),
};
