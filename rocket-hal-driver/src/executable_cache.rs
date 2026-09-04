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
//! see rknpu-spelunking/NOTES.md). `1` is **retired**: it was a hardcoded
//! 4x4 pooling shape that was never hardware-validated, and the CTS test it
//! existed for is gone. `PoolingDef` in the RKT1 schema is how a pool is
//! described now, so tag 1 is rejected rather than falling through to the
//! convolution default and silently handing back the wrong kernel. `2` is
//! `UkernelShape::Conv2d` again but with `Precision::Fp16` -- same
//! geometry as tag `0`, the shape round 7's hardware fix confirmed produces a
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
//! `executable_data` after the tag byte is decoded into a real
//! `conv::Shape` + kernel extent via `decode_conv_shape_v1`, then checked
//! via `validate_conv_shape`. This is real prep for a genuine IREE compiler
//! `TargetBackend` (a separate, not-yet-written C++ project) to eventually
//! emit: unlike tags `0`-`2`, this path is fallible -- malformed or
//! unsupported bytes return a real `IREE_STATUS_INVALID_ARGUMENT`, not a
//! silent fallback to some other hardcoded shape. `validate_conv_shape`
//! shares its construction path with the real dispatch-time builder (see
//! that function's own doc comment), so `command_buffer.rs`'s
//! `catch_unwind` wrapper around the real build is now pure defense in
//! depth rather than a backstop for a known validation gap.

use crate::{
    bindings::{
        iree_const_byte_span_t, iree_hal_executable_cache_t, iree_hal_executable_cache_vtable_t,
        iree_hal_executable_caching_mode_t, iree_hal_executable_params_t, iree_hal_executable_t,
        iree_hal_resource_t, iree_host_size_t, iree_status_t, iree_string_view_t,
    },
    executable::{
        Conv2dExecutable, MatmulExecutable, PoolingExecutable, RuntimeConv2dDimension,
        RuntimeMatmulDimension, RuntimePoolingDimension, UkernelShape,
    },
    status,
};
use iree_rocket_hal::rocket::{
    conv::{self, Activation, Kernels, Multiplier, Precision, Quantization},
    executable_format::{CONV2D_V1_TAG, decode_conv_shape_v1, validate_conv_shape},
    fc,
    pooling::{PoolingMethod, PoolingPrecision, PoolingShape},
};
use rocket_schema::rocket as schema;

const FLATBUFFER_FORMAT: &[u8] = b"rocket-flatbuffer-v1";
const LEGACY_FORMAT: &[u8] = b"rocket-conv2d-v1";
const IREE_FLATBUFFER_HEADER_SIZE: usize = 64;

fn decode_activation(
    activation: schema::Activation,
    activation_cmp: u32,
) -> Result<Activation, ()> {
    match activation {
        schema::Activation::NONE => Ok(Activation::None),
        schema::Activation::RELU => Ok(Activation::Relu),
        schema::Activation::RELUX => Ok(Activation::Clamped {
            cmp: activation_cmp,
        }),
        _ => Err(()),
    }
}

/// Builds a [`conv::Precision`] from the schema's tag plus the surrounding
/// zero-point/scale fields (RKT1's `Conv2DDef`/`FullyConnectedDef` carry
/// these as separate fields, unlike `conv::Precision::Int8`'s own
/// `Quantization` payload). `Multiplier::from_ratio` can panic on a ratio
/// its normalized fixed-point form can't encode -- wrapped in `catch_unwind`
/// so a malformed but structurally valid FlatBuffer produces a clean decode
/// error instead of aborting the process at this `extern "C"` boundary.
fn decode_precision(
    precision: schema::Precision,
    input_zero_point: u32,
    output_zero_point: u32,
    weights_zero_point: u32,
    input_scale: f32,
    weights_scale: f32,
    output_scale: f32,
) -> Result<Precision, ()> {
    match precision {
        schema::Precision::INT8 | schema::Precision::INT8_ACCUMULATOR => {
            if !input_scale.is_finite()
                || !weights_scale.is_finite()
                || !output_scale.is_finite()
                || input_scale <= 0.0
                || weights_scale <= 0.0
                || output_scale <= 0.0
            {
                return Err(());
            }
            let ratio = f64::from(input_scale) * f64::from(weights_scale) / f64::from(output_scale);
            let multiplier =
                std::panic::catch_unwind(|| Multiplier::from_ratio(ratio)).map_err(|_| ())?;
            let quantization = Quantization {
                input_zero_point: input_zero_point as i32,
                output_zero_point: output_zero_point as i32,
                weight_zero_point: weights_zero_point as i32,
                input_scale,
                weights_scale,
                multiplier,
            };
            if precision == schema::Precision::INT8_ACCUMULATOR {
                if input_zero_point != 0 || output_zero_point != 0 || weights_zero_point != 0 {
                    return Err(());
                }
                Ok(Precision::Int8Accumulator(quantization))
            } else {
                Ok(Precision::Int8(quantization))
            }
        }
        schema::Precision::FP16 => Ok(Precision::Fp16),
        _ => Err(()),
    }
}

fn decode_flatbuffer_shape(data: &[u8]) -> Result<UkernelShape, ()> {
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
    match export.kernel_type() {
        schema::KernelDef::Conv2DDef => {
            let conv_def = export.kernel_as_conv_2ddef().ok_or(())?;
            let precision = decode_precision(
                conv_def.precision(),
                conv_def.input_zero_point(),
                conv_def.output_zero_point(),
                conv_def.weights_zero_point(),
                conv_def.input_scale(),
                conv_def.weights_scale(),
                conv_def.output_scale(),
            )?;
            let shape_template = conv::Shape {
                width: conv_def.input_width(),
                height: conv_def.input_height(),
                stride: conv_def.stride(),
                in_channels: conv_def.input_channels(),
                out_channels: conv_def.output_channels(),
                precision,
                padding: Some([conv_def.pad_top() as usize, conv_def.pad_left() as usize]),
                activation: decode_activation(conv_def.activation(), conv_def.activation_cmp())?,
                depthwise: conv_def.depthwise(),
            };
            let kernels: Kernels = [
                conv_def.weights_height() as usize,
                conv_def.weights_width() as usize,
            ];
            let mut runtime_dimensions = Vec::new();
            if let Some(dimensions) = conv_def.runtime_dimensions() {
                for index in 0..dimensions.len() {
                    runtime_dimensions.push(match dimensions.get(index) {
                        schema::Conv2DDimension::INPUT_WIDTH => RuntimeConv2dDimension::InputWidth,
                        schema::Conv2DDimension::INPUT_HEIGHT => {
                            RuntimeConv2dDimension::InputHeight
                        }
                        schema::Conv2DDimension::INPUT_CHANNELS => {
                            RuntimeConv2dDimension::InputChannels
                        }
                        schema::Conv2DDimension::OUTPUT_CHANNELS => {
                            RuntimeConv2dDimension::OutputChannels
                        }
                        schema::Conv2DDimension::WEIGHTS_WIDTH => {
                            RuntimeConv2dDimension::WeightsWidth
                        }
                        schema::Conv2DDimension::WEIGHTS_HEIGHT => {
                            RuntimeConv2dDimension::WeightsHeight
                        }
                        // Unknown to this runtime: either a value from a
                        // newer schema or one of the retired 3/4
                        // (OUTPUT_WIDTH/OUTPUT_HEIGHT, see the .fbs). The
                        // entry still consumed a push-constant ordinal on the
                        // producer's side, so skipping it would silently
                        // shift every later constant onto the wrong field --
                        // reject the executable instead.
                        _ => return Err(()),
                    });
                }
            }
            let executable = Conv2dExecutable {
                shape_template,
                kernels,
                runtime_dimensions,
            };
            executable.validate_template().map_err(|_| ())?;
            Ok(UkernelShape::Conv2d(executable))
        }
        // Deprecated, and decoded only because an older executable may
        // still carry one. It describes the same operation MatmulDef does,
        // so it produces the same runtime shape; what it cannot do is carry
        // runtime dimensions, which is one of the reasons it was replaced.
        schema::KernelDef::FullyConnectedDef => {
            let fc_def = export.kernel_as_fully_connected_def().ok_or(())?;
            let shape = decode_matmul_shape(
                fc_def.m(),
                fc_def.k(),
                fc_def.n(),
                fc_def.precision(),
                fc_def.input_zero_point(),
                fc_def.output_zero_point(),
                fc_def.weights_zero_point(),
                fc_def.input_scale(),
                fc_def.weights_scale(),
                fc_def.output_scale(),
                fc_def.activation(),
                fc_def.activation_cmp(),
            )?;
            let executable = MatmulExecutable::new_static(shape);
            executable.validate_template().map_err(|_| ())?;
            Ok(UkernelShape::Matmul(executable))
        }
        schema::KernelDef::MatmulDef => {
            let matmul_def = export.kernel_as_matmul_def().ok_or(())?;
            let shape = decode_matmul_shape(
                matmul_def.m(),
                matmul_def.k(),
                matmul_def.n(),
                matmul_def.precision(),
                matmul_def.input_zero_point(),
                matmul_def.output_zero_point(),
                matmul_def.weights_zero_point(),
                matmul_def.input_scale(),
                matmul_def.weights_scale(),
                matmul_def.output_scale(),
                matmul_def.activation(),
                matmul_def.activation_cmp(),
            )?;
            let mut runtime_dimensions = Vec::new();
            if let Some(dimensions) = matmul_def.runtime_dimensions() {
                for dimension in dimensions.iter() {
                    runtime_dimensions.push(match dimension {
                        schema::MatmulDimension::M => RuntimeMatmulDimension::M,
                        schema::MatmulDimension::K => RuntimeMatmulDimension::K,
                        schema::MatmulDimension::N => RuntimeMatmulDimension::N,
                        _ => return Err(()),
                    });
                }
            }
            let executable = MatmulExecutable {
                shape_template: shape,
                runtime_dimensions,
            };
            executable.validate_template().map_err(|_| ())?;
            Ok(UkernelShape::Matmul(executable))
        }
        schema::KernelDef::PoolingDef => {
            let pooling_def = export.kernel_as_pooling_def().ok_or(())?;
            let method = match pooling_def.method() {
                schema::PoolingMethod::AVG => PoolingMethod::Avg,
                schema::PoolingMethod::MAX => PoolingMethod::Max,
                schema::PoolingMethod::MIN => PoolingMethod::Min,
                _ => return Err(()),
            };
            let precision = match pooling_def.precision() {
                schema::Precision::INT8 => PoolingPrecision::Int8,
                schema::Precision::FP16 => PoolingPrecision::Fp16,
                // A pool has no weights and no requantization, so the
                // accumulator precision has nothing to mean here.
                _ => return Err(()),
            };
            let padded = pooling_def.pad_left()
                | pooling_def.pad_top()
                | pooling_def.pad_right()
                | pooling_def.pad_bottom()
                != 0;
            // The fill value is derived, never carried -- and where no
            // measurement says what it should be, a *padded* pool is
            // refused rather than filled with a plausible guess. An
            // unpadded pool never reads the field.
            let pad_value = match (method.pad_fill_value(precision), padded) {
                (Some(value), _) => value,
                (None, false) => 0,
                (None, true) => return Err(()),
            };
            let shape_template = PoolingShape {
                input_width: pooling_def.input_width(),
                input_height: pooling_def.input_height(),
                input_channels: pooling_def.channels(),
                output_width: pooling_def.output_width(),
                output_height: pooling_def.output_height(),
                output_channels: pooling_def.channels(),
                precision,
                kernel_width: pooling_def.kernel_width(),
                kernel_height: pooling_def.kernel_height(),
                stride_x: pooling_def.stride_x(),
                stride_y: pooling_def.stride_y(),
                method,
                pad_left: pooling_def.pad_left(),
                pad_top: pooling_def.pad_top(),
                pad_right: pooling_def.pad_right(),
                pad_bottom: pooling_def.pad_bottom(),
                pad_value,
            };
            let mut runtime_dimensions = Vec::new();
            if let Some(dimensions) = pooling_def.runtime_dimensions() {
                for dimension in dimensions.iter() {
                    runtime_dimensions.push(match dimension {
                        schema::PoolingDimension::INPUT_WIDTH => {
                            RuntimePoolingDimension::InputWidth
                        }
                        schema::PoolingDimension::INPUT_HEIGHT => {
                            RuntimePoolingDimension::InputHeight
                        }
                        schema::PoolingDimension::CHANNELS => RuntimePoolingDimension::Channels,
                        schema::PoolingDimension::KERNEL_WIDTH => {
                            RuntimePoolingDimension::KernelWidth
                        }
                        schema::PoolingDimension::KERNEL_HEIGHT => {
                            RuntimePoolingDimension::KernelHeight
                        }
                        schema::PoolingDimension::STRIDE_X => RuntimePoolingDimension::StrideX,
                        schema::PoolingDimension::STRIDE_Y => RuntimePoolingDimension::StrideY,
                        _ => return Err(()),
                    });
                }
            }
            let executable = PoolingExecutable {
                shape_template,
                runtime_dimensions,
            };
            executable.validate_template().map_err(|_| ())?;
            Ok(UkernelShape::Pooling(executable))
        }
        _ => Err(()),
    }
}

/// Shared by `MatmulDef` and the deprecated `FullyConnectedDef`, which carry
/// identical fields.
#[allow(clippy::too_many_arguments)]
fn decode_matmul_shape(
    m: u32,
    k: u32,
    n: u32,
    precision: schema::Precision,
    input_zero_point: u32,
    output_zero_point: u32,
    weights_zero_point: u32,
    input_scale: f32,
    weights_scale: f32,
    output_scale: f32,
    activation: schema::Activation,
    activation_cmp: u32,
) -> Result<fc::Shape, ()> {
    let precision = decode_precision(
        precision,
        input_zero_point,
        output_zero_point,
        weights_zero_point,
        input_scale,
        weights_scale,
        output_scale,
    )?;
    if precision.writes_accumulators() {
        // The exact accumulator path is hardware-validated for Conv2D only;
        // this lowering has separate output packing.
        return Err(());
    }
    let activation = decode_activation(activation, activation_cmp)?;
    // A dynamic executable states zeros here and fills them from push
    // constants, so an all-zero template is legal at this point;
    // `MatmulExecutable::validate_template` is what decides whether the
    // zeros were declared. `fc::Shape` is built as a literal rather than
    // through `fc::Shape::new` for that reason -- the constructor validates
    // eagerly, which a template cannot satisfy.
    Ok(fc::Shape {
        m,
        k,
        n,
        precision,
        activation,
    })
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
        std::slice::from_raw_parts(executable_format.data as *const u8, executable_format.size)
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
        return status::from_code(crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT);
    }
    let format = unsafe {
        std::slice::from_raw_parts(
            params.executable_format.data as *const u8,
            params.executable_format.size,
        )
    };
    let data = params.executable_data;
    if data.data_length != 0 && data.data.is_null() {
        return status::from_code(crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT);
    }
    let bytes = if data.data_length == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(data.data, data.data_length) }
    };

    if format == FLATBUFFER_FORMAT {
        let shape = match decode_flatbuffer_shape(bytes) {
            Ok(shape) => shape,
            Err(()) => {
                return status::from_code(
                    crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                );
            }
        };
        unsafe {
            *out_executable = crate::executable::create(shape);
        }
        return status::ok();
    }
    if format != LEGACY_FORMAT {
        return status::from_code(crate::bindings::iree_status_code_e_IREE_STATUS_NOT_FOUND);
    }

    let tag = if data.data_length >= 1 {
        unsafe { *data.data }
    } else {
        0
    };

    if tag == CONV2D_V1_TAG {
        let payload = if data.data_length >= 1 {
            unsafe { std::slice::from_raw_parts(data.data.add(1), data.data_length - 1) }
        } else {
            &[]
        };
        let (shape, kernels) = match decode_conv_shape_v1(payload) {
            Ok(decoded) => decoded,
            Err(_) => {
                return status::from_code(
                    crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                );
            }
        };
        if validate_conv_shape(&shape, kernels).is_err() {
            return status::from_code(
                crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
            );
        }
        unsafe {
            *out_executable = crate::executable::create(UkernelShape::Conv2d(
                Conv2dExecutable::new_static(shape, kernels),
            ));
        }
        return status::ok();
    }

    // Tag 1 was a hardcoded 4x4 pooling shape, and the only thing that had
    // ever built a `UkernelShape::Pooling`. It was never hardware-validated
    // -- its own comment said so -- and the CTS test it existed for
    // (`cts/pooling_dispatch_test.cc`) is gone. `PoolingDef` in the RKT1
    // schema replaces it, so this refuses rather than falling through to the
    // conv default below and silently handing back a convolution.
    if tag == 1 {
        return status::from_code(crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT);
    }

    let shape = match tag {
        2 => UkernelShape::Conv2d(Conv2dExecutable::new_static(
            conv::Shape {
                // Same geometry as tag 0's validated int8 shape, but
                // Precision::Fp16 -- hardware-confirmed bit-exact-correct
                // (see module doc comment above).
                width: 4,
                height: 4,
                stride: 1,
                in_channels: 1,
                out_channels: 1,
                precision: Precision::Fp16,
                padding: Some([0, 0]),
                activation: Activation::None,
                depthwise: false,
            },
            [1, 1],
        )),
        _ => UkernelShape::Conv2d(Conv2dExecutable::new_static(
            conv::Shape {
                // rkt-basic.rs's validated shape (see module doc comment).
                width: 4,
                height: 4,
                stride: 1,
                in_channels: 1,
                out_channels: 1,
                precision: Precision::Int8(Quantization {
                    input_zero_point: 0,
                    output_zero_point: 0,
                    weight_zero_point: 0,
                    input_scale: 1.0,
                    weights_scale: 1.0,
                    multiplier: Multiplier::from_ratio(1.0),
                }),
                padding: Some([0, 0]),
                activation: Activation::None,
                depthwise: false,
            },
            [1, 1],
        )),
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

    fn encode_accumulator_conv_executable(input_zero_point: i32) -> Vec<u8> {
        let mut builder = flatbuffers::FlatBufferBuilder::new();
        let name = builder.create_string("conv_integer");
        let conv = schema::Conv2DDef::create(
            &mut builder,
            &schema::Conv2DDefArgs {
                input_width: 4,
                input_height: 4,
                input_channels: 1,
                output_width: 4,
                output_height: 4,
                output_channels: 8,
                weights_width: 1,
                weights_height: 1,
                stride: 1,
                input_zero_point: input_zero_point as u32,
                input_scale: 0.25,
                weights_scale: 0.5,
                output_scale: 1.0,
                precision: schema::Precision::INT8_ACCUMULATOR,
                ..Default::default()
            },
        );
        let export = schema::ExportDef::create(
            &mut builder,
            &schema::ExportDefArgs {
                name: Some(name),
                kernel_type: schema::KernelDef::Conv2DDef,
                kernel: Some(conv.as_union_value()),
            },
        );
        let exports = builder.create_vector(&[export]);
        let executable = schema::ExecutableDef::create(
            &mut builder,
            &schema::ExecutableDefArgs {
                exports: Some(exports),
            },
        );
        schema::finish_executable_def_buffer(&mut builder, executable);

        let flatbuffer = builder.finished_data();
        let mut data = vec![0u8; IREE_FLATBUFFER_HEADER_SIZE];
        data[0..4].copy_from_slice(b"RKT1");
        data[8..16].copy_from_slice(&(flatbuffer.len() as u64).to_le_bytes());
        data.extend_from_slice(flatbuffer);
        data
    }

    fn encode_fc_executable() -> Vec<u8> {
        let mut builder = flatbuffers::FlatBufferBuilder::new();
        let name = builder.create_string("rocket_fc_0");
        let fc = schema::FullyConnectedDef::create(
            &mut builder,
            &schema::FullyConnectedDefArgs {
                m: 4,
                k: 32,
                n: 16,
                precision: schema::Precision::FP16,
                ..Default::default()
            },
        );
        let export = schema::ExportDef::create(
            &mut builder,
            &schema::ExportDefArgs {
                name: Some(name),
                kernel_type: schema::KernelDef::FullyConnectedDef,
                kernel: Some(fc.as_union_value()),
            },
        );
        let exports = builder.create_vector(&[export]);
        let executable = schema::ExecutableDef::create(
            &mut builder,
            &schema::ExecutableDefArgs {
                exports: Some(exports),
            },
        );
        schema::finish_executable_def_buffer(&mut builder, executable);

        let flatbuffer = builder.finished_data();
        let mut data = vec![0u8; IREE_FLATBUFFER_HEADER_SIZE];
        data[0..4].copy_from_slice(b"RKT1");
        data[8..16].copy_from_slice(&(flatbuffer.len() as u64).to_le_bytes());
        data.extend_from_slice(flatbuffer);
        data
    }

    fn encode_dynamic_conv_executable(
        dimensions: &[schema::Conv2DDimension],
        input_width: u32,
        padding: [u32; 2],
    ) -> Vec<u8> {
        let mut builder = flatbuffers::FlatBufferBuilder::new();
        let runtime_dimensions = builder.create_vector(dimensions);
        let name = builder.create_string("rocket_dynamic_conv");
        let conv = schema::Conv2DDef::create(
            &mut builder,
            &schema::Conv2DDefArgs {
                input_width,
                input_height: 0,
                input_channels: 32,
                output_width: 0,
                output_height: 0,
                output_channels: 16,
                weights_width: 5,
                weights_height: 3,
                stride: 1,
                precision: schema::Precision::FP16,
                runtime_dimensions: Some(runtime_dimensions),
                pad_top: padding[0],
                pad_left: padding[1],
                ..Default::default()
            },
        );
        let export = schema::ExportDef::create(
            &mut builder,
            &schema::ExportDefArgs {
                name: Some(name),
                kernel_type: schema::KernelDef::Conv2DDef,
                kernel: Some(conv.as_union_value()),
            },
        );
        let exports = builder.create_vector(&[export]);
        let executable = schema::ExecutableDef::create(
            &mut builder,
            &schema::ExecutableDefArgs {
                exports: Some(exports),
            },
        );
        schema::finish_executable_def_buffer(&mut builder, executable);

        let flatbuffer = builder.finished_data();
        let mut data = vec![0u8; IREE_FLATBUFFER_HEADER_SIZE];
        data[0..4].copy_from_slice(b"RKT1");
        data[8..16].copy_from_slice(&(flatbuffer.len() as u64).to_le_bytes());
        data.extend_from_slice(flatbuffer);
        data
    }

    #[test]
    fn decodes_compiler_produced_flatbuffer() {
        let data = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../rocket-schema/testdata/mnv2_conv0.rkt1"
        ));
        let shape = decode_flatbuffer_shape(data).unwrap();
        let UkernelShape::Conv2d(executable) = shape else {
            panic!("expected Conv2d");
        };
        let shape = executable.shape_template;
        let kernels = executable.kernels;
        assert_eq!(shape.width, 112);
        assert_eq!(shape.height, 112);
        assert_eq!(shape.in_channels, 32);
        assert_eq!(shape.output_width(kernels), 112);
        assert_eq!(shape.output_height(kernels), 112);
        assert_eq!(shape.out_channels, 16);
        assert_eq!(shape.precision, Precision::Fp16);
    }

    #[test]
    fn decodes_int8_accumulator_precision() {
        let UkernelShape::Conv2d(executable) =
            decode_flatbuffer_shape(&encode_accumulator_conv_executable(0)).unwrap()
        else {
            panic!("expected Conv2d");
        };
        let Precision::Int8Accumulator(quantization) = executable.shape_template.precision else {
            panic!("expected int8 accumulator precision");
        };
        assert_eq!(quantization.input_scale, 0.25);
        assert_eq!(quantization.weights_scale, 0.5);
        assert_eq!(
            executable.shape_template.precision.output_element_bytes(),
            4
        );
    }

    #[test]
    fn rejects_unvalidated_accumulator_zero_point() {
        assert!(decode_flatbuffer_shape(&encode_accumulator_conv_executable(1)).is_err());
    }

    #[test]
    fn decodes_and_resolves_runtime_conv_dimensions() {
        let data = encode_dynamic_conv_executable(
            &[
                schema::Conv2DDimension::INPUT_HEIGHT,
                schema::Conv2DDimension::INPUT_WIDTH,
            ],
            0,
            [0, 1],
        );
        let UkernelShape::Conv2d(executable) = decode_flatbuffer_shape(&data).unwrap() else {
            panic!("expected Conv2d");
        };
        let constants: Vec<u8> = [112u32, 96]
            .into_iter()
            .flat_map(u32::to_ne_bytes)
            .collect();
        let (shape, kernels) = executable.resolve_shape(&constants).unwrap();
        assert_eq!((shape.width, shape.height), (96, 112));
        assert_eq!(
            (shape.output_width(kernels), shape.output_height(kernels)),
            (94, 110)
        );
        assert_eq!(shape.padding, Some([0, 1]));
    }

    #[test]
    fn rejects_invalid_runtime_conv_dimension_mappings() {
        let duplicate = encode_dynamic_conv_executable(
            &[
                schema::Conv2DDimension::INPUT_WIDTH,
                schema::Conv2DDimension::INPUT_WIDTH,
                schema::Conv2DDimension::INPUT_HEIGHT,
            ],
            0,
            [0, 0],
        );
        assert!(decode_flatbuffer_shape(&duplicate).is_err());

        let unknown = encode_dynamic_conv_executable(
            &[
                schema::Conv2DDimension(99),
                schema::Conv2DDimension::INPUT_HEIGHT,
            ],
            0,
            [0, 0],
        );
        assert!(decode_flatbuffer_shape(&unknown).is_err());

        // 3 is the retired OUTPUT_WIDTH: a value an older serializer could
        // still emit, but one conv::Shape always derives rather than
        // accepts -- see the .fbs and RuntimeConv2dDimension's doc comment.
        let unsupported_output_dimension = encode_dynamic_conv_executable(
            &[
                schema::Conv2DDimension::INPUT_WIDTH,
                schema::Conv2DDimension::INPUT_HEIGHT,
                schema::Conv2DDimension(3),
            ],
            0,
            [0, 0],
        );
        assert!(decode_flatbuffer_shape(&unsupported_output_dimension).is_err());

        let nonzero_template = encode_dynamic_conv_executable(
            &[
                schema::Conv2DDimension::INPUT_WIDTH,
                schema::Conv2DDimension::INPUT_HEIGHT,
            ],
            96,
            [0, 0],
        );
        assert!(decode_flatbuffer_shape(&nonzero_template).is_err());
    }

    #[test]
    fn rejects_truncated_compiler_flatbuffer() {
        let data = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../rocket-schema/testdata/mnv2_conv0.rkt1"
        ));
        assert!(decode_flatbuffer_shape(&data[..data.len() - 1]).is_err());
    }

    fn wrap_executable(builder: flatbuffers::FlatBufferBuilder) -> Vec<u8> {
        let flatbuffer = builder.finished_data();
        let mut data = vec![0u8; IREE_FLATBUFFER_HEADER_SIZE];
        data[0..4].copy_from_slice(b"RKT1");
        data[8..16].copy_from_slice(&(flatbuffer.len() as u64).to_le_bytes());
        data.extend_from_slice(flatbuffer);
        data
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_pooling_executable(
        method: schema::PoolingMethod,
        precision: schema::Precision,
        input: (u32, u32),
        channels: u32,
        output: (u32, u32),
        kernel: (u32, u32),
        stride: (u32, u32),
        pad: [u32; 4],
        dimensions: &[schema::PoolingDimension],
    ) -> Vec<u8> {
        let mut builder = flatbuffers::FlatBufferBuilder::new();
        let runtime_dimensions =
            (!dimensions.is_empty()).then(|| builder.create_vector(dimensions));
        let name = builder.create_string("rocket_pooling_0");
        let pooling = schema::PoolingDef::create(
            &mut builder,
            &schema::PoolingDefArgs {
                input_width: input.0,
                input_height: input.1,
                channels,
                output_width: output.0,
                output_height: output.1,
                kernel_width: kernel.0,
                kernel_height: kernel.1,
                stride_x: stride.0,
                stride_y: stride.1,
                pad_left: pad[0],
                pad_top: pad[1],
                pad_right: pad[2],
                pad_bottom: pad[3],
                method,
                precision,
                runtime_dimensions,
            },
        );
        let export = schema::ExportDef::create(
            &mut builder,
            &schema::ExportDefArgs {
                name: Some(name),
                kernel_type: schema::KernelDef::PoolingDef,
                kernel: Some(pooling.as_union_value()),
            },
        );
        let exports = builder.create_vector(&[export]);
        let executable = schema::ExecutableDef::create(
            &mut builder,
            &schema::ExecutableDefArgs {
                exports: Some(exports),
            },
        );
        schema::finish_executable_def_buffer(&mut builder, executable);
        wrap_executable(builder)
    }

    fn encode_matmul_executable(
        m: u32,
        k: u32,
        n: u32,
        dimensions: &[schema::MatmulDimension],
    ) -> Vec<u8> {
        let mut builder = flatbuffers::FlatBufferBuilder::new();
        let runtime_dimensions =
            (!dimensions.is_empty()).then(|| builder.create_vector(dimensions));
        let name = builder.create_string("rocket_matmul_0");
        let matmul = schema::MatmulDef::create(
            &mut builder,
            &schema::MatmulDefArgs {
                m,
                k,
                n,
                precision: schema::Precision::FP16,
                runtime_dimensions,
                ..Default::default()
            },
        );
        let export = schema::ExportDef::create(
            &mut builder,
            &schema::ExportDefArgs {
                name: Some(name),
                kernel_type: schema::KernelDef::MatmulDef,
                kernel: Some(matmul.as_union_value()),
            },
        );
        let exports = builder.create_vector(&[export]);
        let executable = schema::ExecutableDef::create(
            &mut builder,
            &schema::ExecutableDefArgs {
                exports: Some(exports),
            },
        );
        schema::finish_executable_def_buffer(&mut builder, executable);
        wrap_executable(builder)
    }

    fn pooling_of(data: &[u8]) -> PoolingExecutable {
        match decode_flatbuffer_shape(data).expect("pooling executable must decode") {
            UkernelShape::Pooling(executable) => executable,
            _ => panic!("expected Pooling"),
        }
    }

    /// MobileNetV2's global average pool, the shape this table was added for.
    /// The cross-language gate `rocket-schema/docs/compatibility.md` asks
    /// for: a fixture the **C++ compiler actually produced**, read by the
    /// Rust runtime. Hand-built FlatBuffers above prove this decoder is
    /// self-consistent; only a producer artifact proves the two ends agree.
    ///
    /// Regenerate with:
    ///
    ///     iree-compile rocket-compiler-plugin/test/rocket_pooling.mlir \
    ///       --compile-mode=hal-executable -o rocket-schema/testdata/mnv2_pooling.rkt1
    #[test]
    fn decodes_the_compiler_produced_pooling_fixture() {
        let data = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../rocket-schema/testdata/mnv2_pooling.rkt1"
        ));
        let UkernelShape::Pooling(executable) = decode_flatbuffer_shape(data).unwrap() else {
            panic!("expected Pooling");
        };
        let shape = executable.shape_template;
        assert_eq!((shape.input_width, shape.input_height), (7, 7));
        assert_eq!(shape.input_channels, 1792);
        assert_eq!((shape.kernel_width, shape.kernel_height), (7, 7));
        assert_eq!(shape.method, PoolingMethod::Avg);
        assert_eq!(shape.precision, PoolingPrecision::Fp16);
        assert_eq!(shape.pad_value, 0);
        assert!(executable.runtime_dimensions.is_empty());
    }

    /// The same, for the classifier matmul -- and it doubles as the check
    /// that `MAX_INPUT_CHANNELS` really did reach 1792, since a K of 1792
    /// goes through `validate_conv_shape` on the way in.
    ///
    ///     iree-compile rocket-compiler-plugin/test/rocket_matmul.mlir \
    ///       --compile-mode=hal-executable -o rocket-schema/testdata/mnv2_matmul.rkt1
    #[test]
    fn decodes_the_compiler_produced_matmul_fixture() {
        let data = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../rocket-schema/testdata/mnv2_matmul.rkt1"
        ));
        let UkernelShape::Matmul(executable) = decode_flatbuffer_shape(data).unwrap() else {
            panic!("expected Matmul");
        };
        let shape = executable.shape_template;
        assert_eq!((shape.m, shape.k, shape.n), (1, 1792, 1001));
        assert_eq!(shape.precision, Precision::Fp16);
        assert!(executable.runtime_dimensions.is_empty());
    }

    #[test]
    fn decodes_pooling_flatbuffer() {
        let executable = pooling_of(&encode_pooling_executable(
            schema::PoolingMethod::AVG,
            schema::Precision::FP16,
            (7, 7),
            1792,
            (1, 1),
            (7, 7),
            (1, 1),
            [0; 4],
            &[],
        ));
        let shape = executable.shape_template;
        assert_eq!((shape.input_width, shape.input_height), (7, 7));
        assert_eq!(shape.input_channels, 1792);
        assert_eq!(shape.output_channels, shape.input_channels);
        assert_eq!((shape.output_width, shape.output_height), (1, 1));
        assert_eq!(shape.method, PoolingMethod::Avg);
        assert_eq!(shape.precision, PoolingPrecision::Fp16);
        assert_eq!(shape.pad_value, 0);
    }

    /// The fill value is derived, never carried: a padded fp16 max pool gets
    /// -inf without the producer having any say in it.
    #[test]
    fn derives_the_padded_max_pool_fill_value() {
        let executable = pooling_of(&encode_pooling_executable(
            schema::PoolingMethod::MAX,
            schema::Precision::FP16,
            (8, 8),
            32,
            (4, 4),
            (3, 3),
            (2, 2),
            [1, 1, 1, 1],
            &[],
        ));
        assert_eq!(executable.shape_template.pad_value, 0xFC00);
    }

    /// And where no measurement says what the fill should be, a padded pool
    /// is refused rather than filled with a plausible guess. Unpadded is
    /// fine: nothing reads the field.
    #[test]
    fn refuses_padded_pooling_with_an_unmeasured_fill_value() {
        let padded_int8_max = encode_pooling_executable(
            schema::PoolingMethod::MAX,
            schema::Precision::INT8,
            (8, 8),
            32,
            (4, 4),
            (3, 3),
            (2, 2),
            [1, 1, 1, 1],
            &[],
        );
        assert!(decode_flatbuffer_shape(&padded_int8_max).is_err());

        let padded_min = encode_pooling_executable(
            schema::PoolingMethod::MIN,
            schema::Precision::FP16,
            (8, 8),
            32,
            (4, 4),
            (3, 3),
            (2, 2),
            [1, 1, 1, 1],
            &[],
        );
        assert!(decode_flatbuffer_shape(&padded_min).is_err());

        let unpadded_min = encode_pooling_executable(
            schema::PoolingMethod::MIN,
            schema::Precision::FP16,
            (8, 8),
            32,
            (4, 4),
            (2, 2),
            (2, 2),
            [0; 4],
            &[],
        );
        assert_eq!(pooling_of(&unpadded_min).shape_template.pad_value, 0);
    }

    /// A static executable states its own output extents. They are not what
    /// the runtime uses -- it derives them -- but a disagreement means the
    /// two ends do not share a geometry model.
    #[test]
    fn refuses_pooling_whose_output_extent_disagrees() {
        let wrong = encode_pooling_executable(
            schema::PoolingMethod::AVG,
            schema::Precision::FP16,
            (7, 7),
            32,
            (2, 2),
            (7, 7),
            (1, 1),
            [0; 4],
            &[],
        );
        assert!(decode_flatbuffer_shape(&wrong).is_err());
    }

    #[test]
    fn resolves_pooling_runtime_dimensions() {
        let executable = pooling_of(&encode_pooling_executable(
            schema::PoolingMethod::AVG,
            schema::Precision::FP16,
            (0, 0),
            0,
            (0, 0),
            (0, 0),
            (1, 1),
            [0; 4],
            &[
                schema::PoolingDimension::INPUT_WIDTH,
                schema::PoolingDimension::INPUT_HEIGHT,
                schema::PoolingDimension::CHANNELS,
                schema::PoolingDimension::KERNEL_WIDTH,
                schema::PoolingDimension::KERNEL_HEIGHT,
            ],
        ));
        let mut constants = Vec::new();
        for value in [7u32, 7, 1792, 7, 7] {
            constants.extend_from_slice(&value.to_ne_bytes());
        }
        let shape = executable.resolve_shape(&constants).expect("resolves");
        assert_eq!((shape.input_width, shape.input_height), (7, 7));
        // One wire field sets both, because pooling preserves channels.
        assert_eq!(shape.input_channels, 1792);
        assert_eq!(shape.output_channels, 1792);
        // Derived, not carried.
        assert_eq!((shape.output_width, shape.output_height), (1, 1));

        assert!(executable.resolve_shape(&constants[..4]).is_err());
        let mut zeroed = constants.clone();
        zeroed[0..4].copy_from_slice(&0u32.to_ne_bytes());
        assert!(executable.resolve_shape(&zeroed).is_err());
    }

    /// A dimension listed in the vector has to be zero in the table, and one
    /// that is not listed has to be nonzero -- the same contract Conv2DDef
    /// has, checked here because a pooling template can now break it.
    #[test]
    fn refuses_inconsistent_pooling_templates() {
        let nonzero_runtime_field = encode_pooling_executable(
            schema::PoolingMethod::AVG,
            schema::Precision::FP16,
            (7, 0),
            0,
            (0, 0),
            (0, 0),
            (1, 1),
            [0; 4],
            &[
                schema::PoolingDimension::INPUT_WIDTH,
                schema::PoolingDimension::INPUT_HEIGHT,
                schema::PoolingDimension::CHANNELS,
                schema::PoolingDimension::KERNEL_WIDTH,
                schema::PoolingDimension::KERNEL_HEIGHT,
            ],
        );
        assert!(decode_flatbuffer_shape(&nonzero_runtime_field).is_err());

        let duplicate = encode_pooling_executable(
            schema::PoolingMethod::AVG,
            schema::Precision::FP16,
            (0, 0),
            32,
            (0, 0),
            (2, 2),
            (2, 2),
            [0; 4],
            &[
                schema::PoolingDimension::INPUT_WIDTH,
                schema::PoolingDimension::INPUT_WIDTH,
            ],
        );
        assert!(decode_flatbuffer_shape(&duplicate).is_err());
    }

    #[test]
    fn decodes_matmul_flatbuffer() {
        let shape = decode_flatbuffer_shape(&encode_matmul_executable(4, 32, 16, &[])).unwrap();
        let UkernelShape::Matmul(executable) = shape else {
            panic!("expected Matmul");
        };
        assert_eq!(
            (
                executable.shape_template.m,
                executable.shape_template.k,
                executable.shape_template.n
            ),
            (4, 32, 16)
        );
    }

    #[test]
    fn resolves_matmul_runtime_dimensions() {
        let shape = decode_flatbuffer_shape(&encode_matmul_executable(
            0,
            0,
            0,
            &[
                schema::MatmulDimension::M,
                schema::MatmulDimension::K,
                schema::MatmulDimension::N,
            ],
        ))
        .unwrap();
        let UkernelShape::Matmul(executable) = shape else {
            panic!("expected Matmul");
        };
        let mut constants = Vec::new();
        for value in [1u32, 1792, 1001] {
            constants.extend_from_slice(&value.to_ne_bytes());
        }
        let resolved = executable.resolve_shape(&constants).expect("resolves");
        assert_eq!((resolved.m, resolved.k, resolved.n), (1, 1792, 1001));

        assert!(executable.resolve_shape(&constants[..8]).is_err());
    }

    /// The channel ceilings are runtime semantics, not wire format: the
    /// schema round-trips a K the hardware cannot do, and this is what
    /// refuses it.
    #[test]
    fn refuses_a_matmul_past_the_channel_ceiling() {
        let too_wide = encode_matmul_executable(1, 4096, 64, &[]);
        assert!(decode_flatbuffer_shape(&too_wide).is_err());
    }

    /// The deprecated table still decodes, and lands on the same runtime
    /// shape `MatmulDef` does -- which is the whole reason there is one
    /// variant rather than two.
    #[test]
    fn decodes_fully_connected_flatbuffer() {
        let shape = decode_flatbuffer_shape(&encode_fc_executable()).unwrap();
        let UkernelShape::Matmul(executable) = shape else {
            panic!("expected Matmul");
        };
        let shape = executable.shape_template;
        assert_eq!((shape.m, shape.k, shape.n), (4, 32, 16));
        assert_eq!(shape.precision, Precision::Fp16);
        assert!(executable.runtime_dimensions.is_empty());
    }
}
