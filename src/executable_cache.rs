//! `iree_hal_executable_cache_vtable_t`. `prepare_executable` is the real
//! placeholder-executable-format entry point -- see `executable.rs`'s
//! module doc comment for why it currently ignores `executable_data`
//! entirely and always produces the same hardcoded `ConvShape` (matching
//! `rkt-basic.rs`'s validated 4x4 spatial, 1 channel, 1x1 kernel shape --
//! see rknpu-spelunking/NOTES.md for how that shape was derived and
//! hardware-confirmed correct).

use crate::bindings::{
    iree_const_byte_span_t, iree_hal_executable_cache_t, iree_hal_executable_cache_vtable_t,
    iree_hal_executable_caching_mode_t, iree_hal_executable_params_t, iree_hal_executable_t,
    iree_hal_resource_t, iree_host_size_t, iree_status_t, iree_string_view_t,
};
use crate::status;
use iree_rocket_hal::rocket::regcmd::ConvShape;

/// What every `iree_hal_executable_cache_t*` this driver hands out
/// actually points to. Opaque base type, `resource` at offset 0 like
/// `allocator.rs`/`semaphore.rs`.
#[repr(C)]
pub struct RocketExecutableCache {
    pub resource: iree_hal_resource_t,
}

unsafe fn cast(cache: *mut iree_hal_executable_cache_t) -> *mut RocketExecutableCache {
    cache as *mut RocketExecutableCache
}

pub fn create() -> *mut iree_hal_executable_cache_t {
    let cache = Box::new(RocketExecutableCache {
        resource: iree_hal_resource_t {
            ref_count: 1,
            vtable: &VTABLE as *const _ as *const std::ffi::c_void,
        },
    });
    Box::into_raw(cache) as *mut iree_hal_executable_cache_t
}

unsafe extern "C" fn destroy(executable_cache: *mut iree_hal_executable_cache_t) {
    unsafe { drop(Box::from_raw(cast(executable_cache))) }
}

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
    // No real executable-format parser -- accept anything, since
    // prepare_executable below ignores executable_data anyway (see module
    // doc comment). Once real codegen exists this needs to actually check
    // executable_format against whatever this driver's compiler target
    // emits.
    true
}

#[allow(unused_variables)]
unsafe extern "C" fn prepare_executable(
    executable_cache: *mut iree_hal_executable_cache_t,
    executable_params: *const iree_hal_executable_params_t,
    out_executable: *mut *mut iree_hal_executable_t,
) -> iree_status_t {
    // TODO: real shape extraction. This is rkt-basic.rs's validated shape
    // (see module doc comment) -- deliberately hardcoded, not parsed from
    // executable_params->executable_data (nothing produces a compiled
    // format for this driver yet).
    let shape = ConvShape {
        input_width: 4,
        input_height: 4,
        input_channels: 1,
        output_width: 4,
        output_height: 4,
        output_channels: 1,
        weights_width: 1,
        weights_height: 1,
        stride: 1,
        depthwise: false,
        input_zero_point: 0,
        output_zero_point: 0,
        weights_zero_point: 0,
        input_scale: 1.0,
        weights_scale: 1.0,
        output_scale: 1.0,
        truncate_bits: 0,
    };
    unsafe {
        *out_executable = crate::executable::create(shape);
    }
    status::ok()
}

pub static VTABLE: iree_hal_executable_cache_vtable_t = iree_hal_executable_cache_vtable_t {
    destroy: Some(destroy),
    infer_format: Some(infer_format),
    can_prepare_format: Some(can_prepare_format),
    prepare_executable: Some(prepare_executable),
};
