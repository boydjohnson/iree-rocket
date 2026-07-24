//! `iree_hal_executable_cache_vtable_t`. `prepare_executable` verifies and
//! decodes the compiler-produced `rocket-flatbuffer-v1` executable format.
//! The legacy one-byte test tags remain supported under the distinct
//! `rocket-conv2d-v1` format while old CTS inputs are migrated.
//!
//! Tag convention (entirely this driver's own invention, not derived from
//! anything -- there is no real compiler target yet, see module doc
//! comments throughout this crate): `executable_data`'s first byte, if
//! present, selects which of this driver's hardcoded shapes to produce --
//! `0` (or an empty/missing `executable_data`, preserving every existing
//! caller's behavior) for `UkernelShape::Conv2d` (matching `rkt-basic.rs`'s
//! validated 4x4 spatial, 1 channel, 1x1 kernel shape, `Precision::Int8` --
//! see rknpu-spelunking/NOTES.md), `1` for `UkernelShape::Pooling` (a
//! 4x4x1, 2x2 kernel/stride shape -- NOT yet hardware-validated, see
//! iree-rocket-hal's `build_pooling_regcmd` module doc comment and
//! `tests/pooling_hw.rs`/`tests/pooling_dispatch.rs`), `2` for
//! `UkernelShape::Conv2d` again but with `Precision::Fp16` -- same
//! geometry as tag `0`, matching iree-rocket-hal's `tests/conv_hw.rs`'s
//! `fp16_shape()`, the shape round 7's hardware fix confirmed produces a
//! bit-exact-correct fp16 conv (see
//! rknpu-spelunking's `project_conv_dtype_coverage` memory). Note tag `2`'s
//! buffers are `u16`-element (2 bytes/pixel) fp16 data, not `u8` int8 --
//! callers must size/fill bindings accordingly (see
//! `cts/conv_dispatch_test.cc`). Tags `0`/`1`/`2` have nothing to do with
//! any real serialized executable format; they exist purely so a test
//! harness (which fully controls `executable_data` itself) can select
//! which hardcoded ukernel to exercise through the real HAL API.
//!
//! **Tag `3` is different: it's the first tag backed by a real, versioned
//! wire format** (`iree_rocket_hal::rocket::executable_format`, see that
//! module's own doc comment for the exact byte layout) -- the rest of
//! `executable_data` after the tag byte is decoded into a real `ConvShape`
//! via `decode_conv_shape_v1`, then checked via `validate_conv_shape`. This
//! is real prep for a genuine IREE compiler `TargetBackend` (a separate,
//! not-yet-written C++ project) to eventually emit: unlike tags `0`-`2`,
//! this path is fallible -- malformed or unsupported bytes return a real
//! `IREE_STATUS_INVALID_ARGUMENT`, not a silent fallback to some other
//! hardcoded shape. Note this validation is deliberately not exhaustive of
//! every panic reachable from `build_conv_regcmd` -- see
//! `command_buffer.rs`'s `catch_unwind` wrapper around that call for the
//! backstop covering whatever gap remains.

use crate::bindings::{
    iree_const_byte_span_t, iree_hal_executable_cache_t, iree_hal_executable_cache_vtable_t,
    iree_hal_executable_caching_mode_t, iree_hal_executable_params_t, iree_hal_executable_t,
    iree_hal_resource_t, iree_host_size_t, iree_status_t, iree_string_view_t,
};
use crate::executable::UkernelShape;
use crate::status;
use iree_rocket_hal::rocket::executable_format::{
    CONV2D_V1_TAG, decode_conv_shape_v1, validate_conv_shape,
};
use iree_rocket_hal::rocket::regcmd::{
    Activation, ConvShape, PoolingMethod, PoolingShape, Precision,
};
use rocket_schema::rocket as schema;

const FLATBUFFER_FORMAT: &[u8] = b"rocket-flatbuffer-v1";
const LEGACY_FORMAT: &[u8] = b"rocket-conv2d-v1";
const IREE_FLATBUFFER_HEADER_SIZE: usize = 64;

fn decode_flatbuffer_conv_shape(data: &[u8]) -> Result<ConvShape, ()> {
    if data.len() < IREE_FLATBUFFER_HEADER_SIZE || &data[..4] != b"RKT1" {
        return Err(());
    }

    let version = u32::from_le_bytes(data[4..8].try_into().map_err(|_| ())?);
    if version != 0 {
        return Err(());
    }
    let content_size = usize::try_from(u64::from_le_bytes(data[8..16].try_into().map_err(|_| ())?))
        .map_err(|_| ())?;
    if content_size == 0 || content_size > data.len().saturating_sub(IREE_FLATBUFFER_HEADER_SIZE) {
        return Err(());
    }

    let flatbuffer = &data[IREE_FLATBUFFER_HEADER_SIZE..IREE_FLATBUFFER_HEADER_SIZE + content_size];
    if !schema::executable_def_buffer_has_identifier(flatbuffer) {
        return Err(());
    }
    let executable = schema::root_as_executable_def(flatbuffer).map_err(|_| ())?;
    let exports = executable.exports();
    if exports.len() != 1 {
        return Err(());
    }

    let export = exports.get(0);
    if export.kernel_type() != schema::KernelDef::Conv2DDef {
        return Err(());
    }
    let conv = export.kernel_as_conv_2ddef().ok_or(())?;
    let activation = match conv.activation() {
        schema::Activation::NONE => Activation::None,
        schema::Activation::RELU => Activation::Relu,
        schema::Activation::RELUX => Activation::Relux {
            cmp: conv.activation_cmp(),
        },
        _ => return Err(()),
    };
    let precision = match conv.precision() {
        schema::Precision::INT8 => Precision::Int8,
        schema::Precision::FP16 => Precision::Fp16,
        _ => return Err(()),
    };
    let shape = ConvShape {
        input_width: conv.input_width(),
        input_height: conv.input_height(),
        input_channels: conv.input_channels(),
        output_width: conv.output_width(),
        output_height: conv.output_height(),
        output_channels: conv.output_channels(),
        weights_width: conv.weights_width(),
        weights_height: conv.weights_height(),
        stride: conv.stride(),
        depthwise: conv.depthwise(),
        input_zero_point: conv.input_zero_point(),
        output_zero_point: conv.output_zero_point(),
        weights_zero_point: conv.weights_zero_point(),
        input_scale: conv.input_scale(),
        weights_scale: conv.weights_scale(),
        output_scale: conv.output_scale(),
        truncate_bits: conv.truncate_bits(),
        activation,
        precision,
    };
    validate_conv_shape(&shape).map_err(|_| ())?;
    Ok(shape)
}

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
    if executable_format.size == 0 || executable_format.data.is_null() {
        return false;
    }
    let format = unsafe {
        std::slice::from_raw_parts(
            executable_format.data as *const u8,
            executable_format.size as usize,
        )
    };
    format == FLATBUFFER_FORMAT || format == LEGACY_FORMAT
}

#[allow(unused_variables)]
unsafe extern "C" fn prepare_executable(
    executable_cache: *mut iree_hal_executable_cache_t,
    executable_params: *const iree_hal_executable_params_t,
    out_executable: *mut *mut iree_hal_executable_t,
) -> iree_status_t {
    unsafe {
        *out_executable = std::ptr::null_mut();
    }
    let params = unsafe { &*executable_params };
    if params.executable_format.size == 0 || params.executable_format.data.is_null() {
        return status::from_code(
            crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT as u32,
        );
    }
    let format = unsafe {
        std::slice::from_raw_parts(
            params.executable_format.data as *const u8,
            params.executable_format.size as usize,
        )
    };
    let data = params.executable_data;
    if data.data_length != 0 && data.data.is_null() {
        return status::from_code(
            crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT as u32,
        );
    }
    let bytes = if data.data_length == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(data.data, data.data_length as usize) }
    };

    if format == FLATBUFFER_FORMAT {
        let shape = match decode_flatbuffer_conv_shape(bytes) {
            Ok(shape) => shape,
            Err(()) => {
                return status::from_code(
                    crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT as u32,
                );
            }
        };
        unsafe {
            *out_executable = crate::executable::create(UkernelShape::Conv2d(shape));
        }
        return status::ok();
    }
    if format != LEGACY_FORMAT {
        return status::from_code(crate::bindings::iree_status_code_e_IREE_STATUS_NOT_FOUND as u32);
    }

    let tag = if data.data_length >= 1 {
        unsafe { *data.data }
    } else {
        0
    };

    if tag == CONV2D_V1_TAG {
        let payload = if data.data_length >= 1 {
            unsafe { std::slice::from_raw_parts(data.data.add(1), data.data_length as usize - 1) }
        } else {
            &[]
        };
        let shape = match decode_conv_shape_v1(payload) {
            Ok(shape) => shape,
            Err(_) => {
                return status::from_code(
                    crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT as u32,
                );
            }
        };
        if validate_conv_shape(&shape).is_err() {
            return status::from_code(
                crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT as u32,
            );
        }
        unsafe {
            *out_executable = crate::executable::create(UkernelShape::Conv2d(shape));
        }
        return status::ok();
    }

    let shape = match tag {
        1 => UkernelShape::Pooling(PoolingShape {
            // NOT hardware-validated -- see iree-rocket-hal's
            // build_pooling_regcmd module doc comment and
            // tests/pooling_hw.rs / tests/pooling_dispatch.rs.
            input_width: 4,
            input_height: 4,
            input_channels: 1,
            output_width: 2,
            output_height: 2,
            output_channels: 1,
            kernel_width: 2,
            kernel_height: 2,
            stride_x: 2,
            stride_y: 2,
            method: PoolingMethod::Max,
            pad_left: 0,
            pad_top: 0,
            pad_right: 0,
            pad_bottom: 0,
            pad_value: 0,
            activation: Activation::None,
        }),
        2 => UkernelShape::Conv2d(ConvShape {
            // Same geometry as tag 0's validated int8 shape, but
            // Precision::Fp16 -- matches iree-rocket-hal's
            // tests/conv_hw.rs's fp16_shape(), hardware-confirmed
            // bit-exact-correct (see module doc comment above).
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
            activation: Activation::None,
            precision: Precision::Fp16,
        }),
        _ => UkernelShape::Conv2d(ConvShape {
            // rkt-basic.rs's validated shape (see module doc comment).
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
            activation: Activation::None,
            precision: Precision::Int8,
        }),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_compiler_produced_flatbuffer() {
        let data = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../rocket-schema/testdata/mnv2_conv0.rkt1"
        ));
        let shape = decode_flatbuffer_conv_shape(data).unwrap();
        assert_eq!(shape.input_width, 112);
        assert_eq!(shape.input_height, 112);
        assert_eq!(shape.input_channels, 32);
        assert_eq!(shape.output_width, 112);
        assert_eq!(shape.output_height, 112);
        assert_eq!(shape.output_channels, 16);
        assert_eq!(shape.precision, Precision::Fp16);
    }

    #[test]
    fn rejects_truncated_compiler_flatbuffer() {
        let data = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../rocket-schema/testdata/mnv2_conv0.rkt1"
        ));
        assert!(decode_flatbuffer_conv_shape(&data[..data.len() - 1]).is_err());
    }
}
