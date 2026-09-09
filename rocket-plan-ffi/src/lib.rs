//! Versioned C ABI over `rocket-core`'s planner. The header
//! (`include/rocket_plan.h`) is the contract; this file implements it and
//! nothing else. See the header's comment for the ownership and panic
//! rules, which every function here follows:
//!
//! - every pointer is checked for null and every `struct_size` for the size
//!   this build compiled against, before anything is read;
//! - the whole body runs under `catch_unwind`, so a planner panic becomes
//!   `ROCKET_PLAN_INTERNAL` rather than an abort in the compiler;
//! - the refusal message is copied into the caller's buffer, truncated and
//!   NUL-terminated; nothing allocated here outlives the call.
//!
//! The enum values are asserted against the Rust enums in the tests below,
//! so a reordering on either side fails on the host.

use std::{
    ffi::c_char,
    panic::{AssertUnwindSafe, catch_unwind},
};

use rocket_core::{
    admission::{self, ConvAdmission, MatmulAdmission},
    conv::{self, Activation, ConvPlan, Multiplier, Precision, Quantization},
    error::{PlanError, PlanErrorCode},
    fc,
    policy::{PlanningPolicy, with_policy},
};

pub const ROCKET_PLAN_ABI_VERSION: u32 = 3;

pub const ROCKET_PLAN_OK: u32 = 0;
pub const ROCKET_PLAN_INVALID_SHAPE: u32 = 1;
pub const ROCKET_PLAN_UNSUPPORTED_SEMANTICS: u32 = 2;
pub const ROCKET_PLAN_HARDWARE_LIMIT: u32 = 3;
pub const ROCKET_PLAN_CAPACITY_EXCEEDED: u32 = 4;
pub const ROCKET_PLAN_UNVALIDATED_CONFIGURATION: u32 = 5;
pub const ROCKET_PLAN_INTERNAL: u32 = 6;
pub const ROCKET_PLAN_INVALID_ARGUMENT: u32 = 7;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct rocket_plan_quantization_t {
    pub input_zero_point: i32,
    pub output_zero_point: i32,
    pub weight_zero_point: i32,
    pub input_scale: f32,
    pub weights_scale: f32,
    pub output_scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct rocket_plan_conv_desc_t {
    pub struct_size: u32,
    pub precision: u32,
    pub width: u64,
    pub height: u64,
    pub in_channels: u64,
    pub out_channels: u64,
    pub stride: u64,
    pub kernel_height: u64,
    pub kernel_width: u64,
    pub pad_top: i64,
    pub pad_left: i64,
    pub activation: u32,
    pub activation_ceiling: f32,
    pub depthwise: u8,
    pub reserved_: [u8; 7],
    pub quantization: rocket_plan_quantization_t,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct rocket_plan_matmul_desc_t {
    pub struct_size: u32,
    pub precision: u32,
    pub m: u64,
    pub k: u64,
    pub n: u64,
    pub activation: u32,
    pub activation_ceiling: f32,
    pub quantization: rocket_plan_quantization_t,
}

/// A shape class the compiler is asking about: no spatial extents, no
/// quantization numbers, because the admission envelope
/// (`rocket_core::admission`) is indexed by neither.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct rocket_plan_admission_desc_t {
    pub struct_size: u32,
    pub precision: u32,
    pub kernel_height: u64,
    pub kernel_width: u64,
    pub stride: u64,
    pub in_channels: u64,
    pub out_channels: u64,
    pub depthwise: u8,
    pub reserved_: [u8; 7],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct rocket_plan_matmul_admission_desc_t {
    pub struct_size: u32,
    pub precision: u32,
    pub m: u64,
    pub k: u64,
    pub n: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct rocket_plan_policy_t {
    pub struct_size: u32,
    pub allow_unbacked_channels: u8,
    pub allow_large_kernel_probing: u8,
    pub reserved_: [u8; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct rocket_plan_conv_plan_t {
    pub struct_size: u32,
    pub data_banks: u32,
    pub weight_banks: u32,
    pub tile_count: u32,
    pub column_count: u32,
    pub output_width: u32,
    pub output_height: u32,
    pub reserved_: u32,
    pub weight_bytes: u64,
    pub output_scratch_bytes: u64,
}

/// A boundary-level refusal: either the planner's, or a malformed call.
struct Refusal {
    status: u32,
    message: String,
}

impl From<PlanError> for Refusal {
    fn from(error: PlanError) -> Refusal {
        Refusal {
            status: status_of(error.code()),
            message: error.message().to_string(),
        }
    }
}

fn invalid(message: impl Into<String>) -> Refusal {
    Refusal {
        status: ROCKET_PLAN_INVALID_ARGUMENT,
        message: message.into(),
    }
}

pub fn status_of(code: PlanErrorCode) -> u32 {
    match code {
        PlanErrorCode::InvalidShape => ROCKET_PLAN_INVALID_SHAPE,
        PlanErrorCode::UnsupportedSemantics => ROCKET_PLAN_UNSUPPORTED_SEMANTICS,
        PlanErrorCode::HardwareLimit => ROCKET_PLAN_HARDWARE_LIMIT,
        PlanErrorCode::CapacityExceeded => ROCKET_PLAN_CAPACITY_EXCEEDED,
        PlanErrorCode::UnvalidatedConfiguration => ROCKET_PLAN_UNVALIDATED_CONFIGURATION,
        PlanErrorCode::Internal => ROCKET_PLAN_INTERNAL,
    }
}

pub fn status_name(status: u32) -> &'static str {
    match status {
        ROCKET_PLAN_OK => "ok",
        ROCKET_PLAN_INVALID_SHAPE => "invalid_shape",
        ROCKET_PLAN_UNSUPPORTED_SEMANTICS => "unsupported_semantics",
        ROCKET_PLAN_HARDWARE_LIMIT => "hardware_limit",
        ROCKET_PLAN_CAPACITY_EXCEEDED => "capacity_exceeded",
        ROCKET_PLAN_UNVALIDATED_CONFIGURATION => "unvalidated_configuration",
        ROCKET_PLAN_INTERNAL => "internal",
        ROCKET_PLAN_INVALID_ARGUMENT => "invalid_argument",
        _ => "unknown",
    }
}

fn narrow(value: u64, what: &str) -> Result<u32, Refusal> {
    u32::try_from(value).map_err(|_| {
        PlanError::new(
            PlanErrorCode::HardwareLimit,
            format!("{what} {value} does not fit the hardware's 32-bit fields"),
        )
        .into()
    })
}

fn quantization(q: &rocket_plan_quantization_t) -> Result<Quantization, Refusal> {
    let finite_positive = |v: f32| v.is_finite() && v > 0.0;
    if !finite_positive(q.input_scale)
        || !finite_positive(q.weights_scale)
        || !finite_positive(q.output_scale)
    {
        return Err(PlanError::new(
            PlanErrorCode::InvalidShape,
            "int8 scales must be finite and positive",
        )
        .into());
    }
    let ratio = f64::from(q.input_scale) * f64::from(q.weights_scale) / f64::from(q.output_scale);
    let multiplier = Multiplier::try_from_ratio(ratio)
        .map_err(|reason| PlanError::new(PlanErrorCode::HardwareLimit, reason))?;
    Ok(Quantization {
        input_zero_point: q.input_zero_point,
        output_zero_point: q.output_zero_point,
        weight_zero_point: q.weight_zero_point,
        input_scale: q.input_scale,
        weights_scale: q.weights_scale,
        multiplier,
    })
}

fn precision(code: u32, q: &rocket_plan_quantization_t) -> Result<Precision, Refusal> {
    Ok(match code {
        0 => Precision::Fp16,
        1 => Precision::Fp16Accumulator,
        2 => Precision::Bf16,
        3 => Precision::Int16,
        4 => Precision::Tf32,
        5 => Precision::Int4,
        6 => Precision::Int8(quantization(q)?),
        7 => Precision::Int8Accumulator(quantization(q)?),
        other => {
            return Err(invalid(format!(
                "precision {other} is not a rocket_plan_precision_e"
            )));
        }
    })
}

/// The precision for an admission query.
///
/// [`admission::admit_conv`] reads only which rung the precision is on, so
/// the int8 variants are given a unit requantization here rather than
/// making every caller carry calibration numbers it does not have yet. A
/// value that would change the answer must never be invented this way; a
/// value the answer does not depend on is exactly what a placeholder is
/// for.
fn admission_precision(code: u32) -> Result<Precision, Refusal> {
    let unit = rocket_plan_quantization_t {
        input_zero_point: 0,
        output_zero_point: 0,
        weight_zero_point: 0,
        input_scale: 1.0,
        weights_scale: 1.0,
        output_scale: 1.0,
    };
    precision(code, &unit)
}

fn activation(code: u32, ceiling: f32, precision: Precision) -> Result<Activation, Refusal> {
    Ok(match code {
        0 => Activation::None,
        1 => Activation::Relu,
        2 => match precision.quantization() {
            None => Activation::try_clamped_fp16(ceiling)?,
            Some(q) => Activation::try_clamped_int8(ceiling, q.input_scale, q.weights_scale)?,
        },
        other => {
            return Err(invalid(format!(
                "activation {other} is not a rocket_plan_activation_e"
            )));
        }
    })
}

fn policy_of(policy: *const rocket_plan_policy_t) -> Result<PlanningPolicy, Refusal> {
    if policy.is_null() {
        return Ok(PlanningPolicy::default());
    }
    // SAFETY: non-null; the caller promises a readable rocket_plan_policy_t,
    // and struct_size is checked before any other field is trusted.
    let policy = unsafe { &*policy };
    if policy.struct_size as usize != std::mem::size_of::<rocket_plan_policy_t>() {
        return Err(invalid(
            "rocket_plan_policy_t.struct_size does not match this ABI",
        ));
    }
    Ok(PlanningPolicy {
        allow_unbacked_channels: policy.allow_unbacked_channels != 0,
        allow_large_kernel_probing: policy.allow_large_kernel_probing != 0,
    })
}

fn summarize(plan: &ConvPlan) -> Result<rocket_plan_conv_plan_t, Refusal> {
    let shape = plan.shape();
    let kernels = plan.kernels();
    Ok(rocket_plan_conv_plan_t {
        struct_size: std::mem::size_of::<rocket_plan_conv_plan_t>() as u32,
        data_banks: plan.data_banks(),
        weight_banks: plan.weight_banks(),
        tile_count: plan.tiles().len() as u32,
        column_count: plan.output_column_widths().len() as u32,
        output_width: shape.output_width(kernels),
        output_height: shape.output_height(kernels),
        reserved_: 0,
        weight_bytes: u64::from(shape.try_weight_bytes(kernels)?),
        output_scratch_bytes: shape.output_scratch_bytes(kernels) as u64,
    })
}

fn plan_conv_checked(desc: &rocket_plan_conv_desc_t) -> Result<ConvPlan, Refusal> {
    let precision = precision(desc.precision, &desc.quantization)?;
    let mut shape = conv::Shape::try_with_precision(
        narrow(desc.width, "width")?,
        narrow(desc.height, "height")?,
        narrow(desc.stride, "stride")?,
        narrow(desc.in_channels, "input channels")?,
        narrow(desc.out_channels, "output channels")?,
        precision,
    )?;
    match (desc.pad_top, desc.pad_left) {
        (-1, -1) => {}
        (top, left) if top >= 0 && left >= 0 => {
            let pad = |value: i64, what: &str| {
                usize::try_from(value)
                    .ok()
                    .filter(|&v| v <= 15)
                    .ok_or_else(|| {
                        PlanError::new(
                            PlanErrorCode::HardwareLimit,
                            format!(
                                "{what} padding {value} does not fit the CNA's 4-bit pad field"
                            ),
                        )
                    })
            };
            shape = shape.try_with_padding([pad(top, "top")?, pad(left, "left")?])?;
        }
        _ => {
            return Err(invalid(
                "pad_top and pad_left must both be -1 (planner default) or both be non-negative",
            ));
        }
    }
    shape = shape.try_with_activation(activation(
        desc.activation,
        desc.activation_ceiling,
        precision,
    )?)?;
    if desc.depthwise != 0 {
        shape = shape.try_with_depthwise()?;
    }
    let kernels = [
        narrow(desc.kernel_height, "kernel height")? as usize,
        narrow(desc.kernel_width, "kernel width")? as usize,
    ];
    Ok(ConvPlan::try_new(shape, kernels)?)
}

fn plan_matmul_checked(desc: &rocket_plan_matmul_desc_t) -> Result<ConvPlan, Refusal> {
    let precision = precision(desc.precision, &desc.quantization)?;
    let shape = fc::Shape::try_new(
        narrow(desc.m, "m")?,
        narrow(desc.k, "k")?,
        narrow(desc.n, "n")?,
        precision,
    )?
    .with_activation(activation(
        desc.activation,
        desc.activation_ceiling,
        precision,
    )?);
    Ok(fc::Plan::try_new(shape)?.conv_plan().clone())
}

/// Copies `message` into the caller's buffer, truncated and NUL-terminated.
///
/// # Safety
/// `buffer` is either null or points at `capacity` writable bytes.
unsafe fn write_message(buffer: *mut c_char, capacity: usize, message: &str) {
    if buffer.is_null() || capacity == 0 {
        return;
    }
    let bytes = message.as_bytes();
    let mut length = bytes.len().min(capacity - 1);
    // Do not cut a UTF-8 sequence in half.
    while length > 0 && !message.is_char_boundary(length) {
        length -= 1;
    }
    // SAFETY: the caller promises `capacity` writable bytes; `length + 1 <=
    // capacity`.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.cast::<u8>(), length);
        *buffer.add(length) = 0;
    }
}

/// The shared boundary: checks the descriptor pointer and size, runs `plan`
/// under the policy and `catch_unwind`, and reports.
///
/// # Safety
/// Pointer contracts as documented in `rocket_plan.h`.
unsafe fn boundary<D>(
    desc: *const D,
    policy: *const rocket_plan_policy_t,
    out_plan: *mut rocket_plan_conv_plan_t,
    message: *mut c_char,
    message_capacity: usize,
    struct_size_of: impl Fn(&D) -> u32,
    plan: impl Fn(&D) -> Result<ConvPlan, Refusal>,
) -> u32 {
    let outcome = catch_unwind(AssertUnwindSafe(
        || -> Result<rocket_plan_conv_plan_t, Refusal> {
            if desc.is_null() {
                return Err(invalid("descriptor pointer is null"));
            }
            // SAFETY: non-null; the caller promises a readable descriptor whose
            // first field is struct_size, which is checked before the rest is
            // trusted.
            let desc = unsafe { &*desc };
            if struct_size_of(desc) as usize != std::mem::size_of::<D>() {
                return Err(invalid("descriptor struct_size does not match this ABI"));
            }
            if !out_plan.is_null() {
                // SAFETY: non-null; the caller promises a writable plan struct.
                let out = unsafe { &*out_plan };
                if out.struct_size as usize != std::mem::size_of::<rocket_plan_conv_plan_t>() {
                    return Err(invalid(
                        "rocket_plan_conv_plan_t.struct_size does not match this ABI",
                    ));
                }
            }
            let policy = policy_of(policy)?;
            let planned = with_policy(policy, || plan(desc))?;
            summarize(&planned)
        },
    ));
    match outcome {
        Ok(Ok(summary)) => {
            if !out_plan.is_null() {
                // SAFETY: checked above.
                unsafe { *out_plan = summary };
            }
            ROCKET_PLAN_OK
        }
        Ok(Err(refusal)) => {
            // SAFETY: caller's buffer contract.
            unsafe { write_message(message, message_capacity, &refusal.message) };
            refusal.status
        }
        Err(payload) => {
            let text = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .unwrap_or("non-string panic payload");
            // SAFETY: caller's buffer contract.
            unsafe {
                write_message(
                    message,
                    message_capacity,
                    &format!("planner panicked instead of refusing: {text}"),
                )
            };
            ROCKET_PLAN_INTERNAL
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rocket_plan_abi_version() -> u32 {
    ROCKET_PLAN_ABI_VERSION
}

#[unsafe(no_mangle)]
pub extern "C" fn rocket_plan_status_name(status: u32) -> *const c_char {
    // Every arm of `status_name` is a NUL-terminated literal via the table
    // below, which is what lets a `&'static str` be handed out as C string.
    const NAMES: [&std::ffi::CStr; 9] = [
        c"ok",
        c"invalid_shape",
        c"unsupported_semantics",
        c"hardware_limit",
        c"capacity_exceeded",
        c"unvalidated_configuration",
        c"internal",
        c"invalid_argument",
        c"unknown",
    ];
    NAMES[(status as usize).min(NAMES.len() - 1)].as_ptr()
}

#[unsafe(no_mangle)]
pub extern "C" fn rocket_plan_precision_name(precision: u32) -> *const c_char {
    // The spellings the transform spec's `precision` attribute uses, so a
    // placement report and the spec name the same rung the same way. The
    // two int8 entries are the whole reason this exists: `i8 x i8 -> i32`
    // in the IR is either of them, and a report that only said "int8"
    // would not say which lowering was asked about.
    const NAMES: [&std::ffi::CStr; 9] = [
        c"fp16",
        c"fp16_accumulator",
        c"bf16",
        c"int16",
        c"tf32",
        c"int4",
        c"int8_requant",
        c"int8_accumulator",
        c"unknown",
    ];
    NAMES[(precision as usize).min(NAMES.len() - 1)].as_ptr()
}

/// # Safety
/// See `rocket_plan.h`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rocket_plan_conv(
    desc: *const rocket_plan_conv_desc_t,
    policy: *const rocket_plan_policy_t,
    out_plan: *mut rocket_plan_conv_plan_t,
    message: *mut c_char,
    message_capacity: usize,
) -> u32 {
    // SAFETY: forwarded contract.
    unsafe {
        boundary(
            desc,
            policy,
            out_plan,
            message,
            message_capacity,
            |d| d.struct_size,
            plan_conv_checked,
        )
    }
}

/// # Safety
/// See `rocket_plan.h`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rocket_plan_matmul(
    desc: *const rocket_plan_matmul_desc_t,
    policy: *const rocket_plan_policy_t,
    out_plan: *mut rocket_plan_conv_plan_t,
    message: *mut c_char,
    message_capacity: usize,
) -> u32 {
    // SAFETY: forwarded contract.
    unsafe {
        boundary(
            desc,
            policy,
            out_plan,
            message,
            message_capacity,
            |d| d.struct_size,
            plan_matmul_checked,
        )
    }
}

/// The shared boundary for the admission queries: no policy, no plan, just
/// a verdict and a message. Same null/`struct_size`/`catch_unwind` rules as
/// [`boundary`].
///
/// # Safety
/// Pointer contracts as documented in `rocket_plan.h`.
unsafe fn admission_boundary<D>(
    desc: *const D,
    message: *mut c_char,
    message_capacity: usize,
    struct_size_of: impl Fn(&D) -> u32,
    admit: impl Fn(&D) -> Result<(), Refusal>,
) -> u32 {
    let outcome = catch_unwind(AssertUnwindSafe(|| -> Result<(), Refusal> {
        if desc.is_null() {
            return Err(invalid("descriptor pointer is null"));
        }
        // SAFETY: non-null; the caller promises a readable descriptor whose
        // first field is struct_size, checked before the rest is trusted.
        let desc = unsafe { &*desc };
        if struct_size_of(desc) as usize != std::mem::size_of::<D>() {
            return Err(invalid("descriptor struct_size does not match this ABI"));
        }
        admit(desc)
    }));
    match outcome {
        Ok(Ok(())) => ROCKET_PLAN_OK,
        Ok(Err(refusal)) => {
            // SAFETY: caller's buffer contract.
            unsafe { write_message(message, message_capacity, &refusal.message) };
            refusal.status
        }
        Err(payload) => {
            let text = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .unwrap_or("non-string panic payload");
            // SAFETY: caller's buffer contract.
            unsafe {
                write_message(
                    message,
                    message_capacity,
                    &format!("admission panicked instead of refusing: {text}"),
                )
            };
            ROCKET_PLAN_INTERNAL
        }
    }
}

/// # Safety
/// See `rocket_plan.h`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rocket_admit_conv(
    desc: *const rocket_plan_admission_desc_t,
    message: *mut c_char,
    message_capacity: usize,
) -> u32 {
    // SAFETY: forwarded contract.
    unsafe {
        admission_boundary(
            desc,
            message,
            message_capacity,
            |d| d.struct_size,
            |d| {
                Ok(admission::admit_conv(&ConvAdmission {
                    precision: admission_precision(d.precision)?,
                    depthwise: d.depthwise != 0,
                    kernel_height: d.kernel_height,
                    kernel_width: d.kernel_width,
                    stride: d.stride,
                    in_channels: d.in_channels,
                    out_channels: d.out_channels,
                })?)
            },
        )
    }
}

/// # Safety
/// See `rocket_plan.h`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rocket_admit_matmul(
    desc: *const rocket_plan_matmul_admission_desc_t,
    message: *mut c_char,
    message_capacity: usize,
) -> u32 {
    // SAFETY: forwarded contract.
    unsafe {
        admission_boundary(
            desc,
            message,
            message_capacity,
            |d| d.struct_size,
            |d| {
                Ok(admission::admit_matmul(&MatmulAdmission {
                    precision: admission_precision(d.precision)?,
                    m: d.m,
                    k: d.k,
                    n: d.n,
                })?)
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{ffi::CStr, mem::size_of};

    fn conv_desc(width: u64, height: u64, cin: u64, cout: u64, k: u64) -> rocket_plan_conv_desc_t {
        rocket_plan_conv_desc_t {
            struct_size: size_of::<rocket_plan_conv_desc_t>() as u32,
            precision: 0,
            width,
            height,
            in_channels: cin,
            out_channels: cout,
            stride: 1,
            kernel_height: k,
            kernel_width: k,
            pad_top: -1,
            pad_left: -1,
            activation: 0,
            activation_ceiling: 0.0,
            depthwise: 0,
            reserved_: [0; 7],
            quantization: rocket_plan_quantization_t::default(),
        }
    }

    fn call(desc: &rocket_plan_conv_desc_t) -> (u32, rocket_plan_conv_plan_t, String) {
        let mut plan = rocket_plan_conv_plan_t {
            struct_size: size_of::<rocket_plan_conv_plan_t>() as u32,
            ..Default::default()
        };
        let mut message = [0 as c_char; 256];
        let status = unsafe {
            rocket_plan_conv(
                desc,
                std::ptr::null(),
                &mut plan,
                message.as_mut_ptr(),
                message.len(),
            )
        };
        let text = unsafe { CStr::from_ptr(message.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        (status, plan, text)
    }

    #[test]
    fn the_status_codes_track_the_rust_enum() {
        for (code, status) in [
            (PlanErrorCode::InvalidShape, ROCKET_PLAN_INVALID_SHAPE),
            (
                PlanErrorCode::UnsupportedSemantics,
                ROCKET_PLAN_UNSUPPORTED_SEMANTICS,
            ),
            (PlanErrorCode::HardwareLimit, ROCKET_PLAN_HARDWARE_LIMIT),
            (
                PlanErrorCode::CapacityExceeded,
                ROCKET_PLAN_CAPACITY_EXCEEDED,
            ),
            (
                PlanErrorCode::UnvalidatedConfiguration,
                ROCKET_PLAN_UNVALIDATED_CONFIGURATION,
            ),
            (PlanErrorCode::Internal, ROCKET_PLAN_INTERNAL),
        ] {
            assert_eq!(status_of(code), status);
            let name = unsafe { CStr::from_ptr(rocket_plan_status_name(status)) };
            assert_eq!(name.to_str().unwrap(), status_name(status));
        }
        assert_eq!(rocket_plan_abi_version(), ROCKET_PLAN_ABI_VERSION);
    }

    #[test]
    fn the_precision_codes_track_the_rust_enum() {
        let q = rocket_plan_quantization_t {
            input_scale: 1.0,
            weights_scale: 1.0,
            output_scale: 1.0,
            ..Default::default()
        };
        let expected = [
            Precision::Fp16,
            Precision::Fp16Accumulator,
            Precision::Bf16,
            Precision::Int16,
            Precision::Tf32,
            Precision::Int4,
        ];
        for (code, want) in expected.into_iter().enumerate() {
            assert_eq!(precision(code as u32, &q).ok(), Some(want));
        }
        assert!(matches!(precision(6, &q), Ok(Precision::Int8(_))));
        assert!(matches!(
            precision(7, &q),
            Ok(Precision::Int8Accumulator(_))
        ));
        assert_eq!(
            precision(8, &q).err().map(|r| r.status),
            Some(ROCKET_PLAN_INVALID_ARGUMENT)
        );
    }

    /// The names are what a placement report prints, so they are pinned to
    /// the codes here rather than left to drift against the transform
    /// spec's `precision` attribute, which uses the same spellings.
    #[test]
    fn the_precision_names_track_the_codes() {
        let name = |code| {
            unsafe { CStr::from_ptr(rocket_plan_precision_name(code)) }
                .to_str()
                .unwrap()
        };
        assert_eq!(name(0), "fp16");
        assert_eq!(name(1), "fp16_accumulator");
        assert_eq!(name(2), "bf16");
        assert_eq!(name(3), "int16");
        assert_eq!(name(4), "tf32");
        assert_eq!(name(5), "int4");
        assert_eq!(name(6), "int8_requant");
        assert_eq!(name(7), "int8_accumulator");
        // Past the table, and far past it: neither may read out of bounds.
        assert_eq!(name(8), "unknown");
        assert_eq!(name(u32::MAX), "unknown");
    }

    #[test]
    fn a_planned_convolution_reports_its_geometry() {
        let (status, plan, _) = call(&conv_desc(56, 56, 64, 64, 3));
        assert_eq!(status, ROCKET_PLAN_OK);
        let reference = ConvPlan::new(conv::Shape::with_out_channels(56, 56, 1, 64, 64), [3, 3]);
        assert_eq!(plan.tile_count as usize, reference.tiles().len());
        assert_eq!(plan.data_banks, reference.data_banks());
        assert_eq!((plan.output_width, plan.output_height), (56, 56));
        assert_eq!(
            plan.weight_bytes,
            u64::from(reference.shape().weight_bytes([3, 3]))
        );
    }

    #[test]
    fn refusals_carry_the_status_and_the_message() {
        let (status, _, text) = call(&conv_desc(8, 8, conv::MAX_INPUT_CHANNELS as u64 + 1, 8, 1));
        assert_eq!(status, ROCKET_PLAN_UNVALIDATED_CONFIGURATION);
        assert!(text.contains("input channels must be"), "{text}");
        let (status, _, text) = call(&conv_desc(8, 8, 0, 8, 1));
        assert_eq!(status, ROCKET_PLAN_INVALID_SHAPE);
        assert!(text.contains("nonzero"), "{text}");
    }

    #[test]
    fn wide_dimensions_are_narrowed_with_a_refusal_not_a_truncation() {
        // 2^32 + 8 would read as 8 after a bare cast; it must refuse instead.
        let (status, _, text) = call(&conv_desc((1u64 << 32) + 8, 8, 3, 8, 1));
        assert_eq!(status, ROCKET_PLAN_HARDWARE_LIMIT);
        assert!(
            text.contains("does not fit the hardware's 32-bit fields"),
            "{text}"
        );
    }

    #[test]
    fn malformed_calls_are_invalid_arguments() {
        let mut desc = conv_desc(8, 8, 3, 8, 1);
        desc.struct_size = 4;
        assert_eq!(call(&desc).0, ROCKET_PLAN_INVALID_ARGUMENT);
        let status = unsafe {
            rocket_plan_conv(
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            )
        };
        assert_eq!(status, ROCKET_PLAN_INVALID_ARGUMENT);
        let mut desc = conv_desc(8, 8, 3, 8, 1);
        desc.precision = 99;
        assert_eq!(call(&desc).0, ROCKET_PLAN_INVALID_ARGUMENT);
        let mut desc = conv_desc(8, 8, 3, 8, 1);
        desc.pad_top = -1;
        desc.pad_left = 0;
        assert_eq!(call(&desc).0, ROCKET_PLAN_INVALID_ARGUMENT);
    }

    #[test]
    fn messages_are_truncated_at_a_char_boundary() {
        let desc = conv_desc(8, 8, 0, 8, 1);
        let mut message = [0x7f as c_char; 8];
        let status = unsafe {
            rocket_plan_conv(
                &desc,
                std::ptr::null(),
                std::ptr::null_mut(),
                message.as_mut_ptr(),
                message.len(),
            )
        };
        assert_eq!(status, ROCKET_PLAN_INVALID_SHAPE);
        let text = unsafe { CStr::from_ptr(message.as_ptr()) };
        assert_eq!(text.to_bytes().len(), 7);
    }

    #[test]
    fn the_policy_reaches_the_planner_and_does_not_leak() {
        let desc = conv_desc(8, 8, conv::MAX_INPUT_CHANNELS as u64 + 64, 8, 1);
        let policy = rocket_plan_policy_t {
            struct_size: size_of::<rocket_plan_policy_t>() as u32,
            allow_unbacked_channels: 1,
            ..Default::default()
        };
        let mut plan = rocket_plan_conv_plan_t {
            struct_size: size_of::<rocket_plan_conv_plan_t>() as u32,
            ..Default::default()
        };
        let status =
            unsafe { rocket_plan_conv(&desc, &policy, &mut plan, std::ptr::null_mut(), 0) };
        assert_eq!(status, ROCKET_PLAN_OK);
        assert_eq!(call(&desc).0, ROCKET_PLAN_UNVALIDATED_CONFIGURATION);
        assert_eq!(PlanningPolicy::current(), PlanningPolicy::from_env());
    }

    fn admission_desc(
        precision: u32,
        depthwise: bool,
        kernel: u64,
        stride: u64,
        cin: u64,
        cout: u64,
    ) -> rocket_plan_admission_desc_t {
        rocket_plan_admission_desc_t {
            struct_size: size_of::<rocket_plan_admission_desc_t>() as u32,
            precision,
            kernel_height: kernel,
            kernel_width: kernel,
            stride,
            in_channels: cin,
            out_channels: cout,
            depthwise: u8::from(depthwise),
            reserved_: [0; 7],
        }
    }

    fn admit(desc: &rocket_plan_admission_desc_t) -> (u32, String) {
        let mut message = [0i8; 256];
        // SAFETY: a live descriptor and a live buffer.
        let status = unsafe { rocket_admit_conv(desc, message.as_mut_ptr().cast(), message.len()) };
        let text = unsafe { CStr::from_ptr(message.as_ptr().cast()) }
            .to_string_lossy()
            .into_owned();
        (status, text)
    }

    #[test]
    fn admission_answers_without_spatial_extents_or_calibration() {
        // The point of the separate entry point: no width, no height, no
        // scales, and it still decides -- which is what makes it usable on
        // a convolution whose spatial extents are dynamic.
        assert_eq!(
            admit(&admission_desc(0, false, 1, 1, 3584, 3584)).0,
            ROCKET_PLAN_OK
        );
        let (status, message) = admit(&admission_desc(0, false, 1, 1, 3585, 64));
        assert_eq!(status, ROCKET_PLAN_UNVALIDATED_CONFIGURATION);
        assert!(
            message.contains("3585") && message.contains("3584"),
            "{message}"
        );
    }

    #[test]
    fn the_two_int8_precisions_get_different_admission_answers() {
        // Precision 6 is the requantized rung, 7 the accumulator one; the
        // same descriptor otherwise.
        assert_eq!(
            admit(&admission_desc(7, false, 1, 1, 3584, 64)).0,
            ROCKET_PLAN_OK
        );
        assert_eq!(
            admit(&admission_desc(6, false, 1, 1, 3584, 64)).0,
            ROCKET_PLAN_UNVALIDATED_CONFIGURATION
        );
        assert_eq!(
            admit(&admission_desc(6, false, 1, 1, 1344, 1792)).0,
            ROCKET_PLAN_OK
        );
    }

    #[test]
    fn malformed_admission_calls_are_invalid_arguments() {
        let mut wrong = admission_desc(0, false, 1, 1, 64, 64);
        wrong.struct_size = 3;
        assert_eq!(admit(&wrong).0, ROCKET_PLAN_INVALID_ARGUMENT);
        let mut bad_precision = admission_desc(99, false, 1, 1, 64, 64);
        bad_precision.struct_size = size_of::<rocket_plan_admission_desc_t>() as u32;
        assert_eq!(admit(&bad_precision).0, ROCKET_PLAN_INVALID_ARGUMENT);
        // SAFETY: a null descriptor is part of the documented contract.
        assert_eq!(
            unsafe { rocket_admit_conv(std::ptr::null(), std::ptr::null_mut(), 0) },
            ROCKET_PLAN_INVALID_ARGUMENT
        );
    }

    #[test]
    fn matmul_admission_bounds_m_and_the_channel_axes() {
        let desc = |m, k, n| rocket_plan_matmul_admission_desc_t {
            struct_size: size_of::<rocket_plan_matmul_admission_desc_t>() as u32,
            precision: 0,
            m,
            k,
            n,
        };
        let call = |d: &rocket_plan_matmul_admission_desc_t| {
            let mut message = [0i8; 256];
            // SAFETY: a live descriptor and a live buffer.
            unsafe { rocket_admit_matmul(d, message.as_mut_ptr().cast(), message.len()) }
        };
        assert_eq!(call(&desc(2047, 3584, 3584)), ROCKET_PLAN_OK);
        assert_eq!(
            call(&desc(2048, 64, 64)),
            ROCKET_PLAN_UNVALIDATED_CONFIGURATION
        );
        assert_eq!(
            call(&desc(64, 64, 3585)),
            ROCKET_PLAN_UNVALIDATED_CONFIGURATION
        );
    }

    #[test]
    fn matmuls_plan_through_the_fc_mapping() {
        let desc = rocket_plan_matmul_desc_t {
            struct_size: size_of::<rocket_plan_matmul_desc_t>() as u32,
            precision: 0,
            m: 197,
            k: 768,
            n: 768,
            activation: 0,
            activation_ceiling: 0.0,
            quantization: rocket_plan_quantization_t::default(),
        };
        let mut plan = rocket_plan_conv_plan_t {
            struct_size: size_of::<rocket_plan_conv_plan_t>() as u32,
            ..Default::default()
        };
        let status = unsafe {
            rocket_plan_matmul(&desc, std::ptr::null(), &mut plan, std::ptr::null_mut(), 0)
        };
        assert_eq!(status, ROCKET_PLAN_OK);
        assert_eq!((plan.output_width, plan.output_height), (197, 1));
        let reference = fc::Plan::new(fc::Shape::new(197, 768, 768, Precision::Fp16));
        assert_eq!(
            plan.tile_count as usize,
            reference.conv_plan().tiles().len()
        );
    }
}
