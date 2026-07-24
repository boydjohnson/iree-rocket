//! Real, versioned binary wire format for `ConvShape`, replacing
//! `rocket-hal-driver`'s one-byte "pick one of 3 hardcoded shapes" tag hack
//! with an actual encode/decode of shape/dtype/quantization data. This is
//! prep work for a real IREE compiler `TargetBackend` (not part of this
//! crate or `rocket-hal-driver` -- a separate, C++, out-of-tree project):
//! that backend's `serializeExecutable()` would emit exactly the bytes
//! `decode_conv_shape_v1` parses here.
//!
//! Layout, all little-endian, fixed total size (every `ConvShape` field is
//! a fixed-size scalar, so there is no variable-length data to encode):
//!
//! ```text
//! bytes 0..4:   format_version: u32        (CONV2D_V1_FORMAT_VERSION == 1)
//! bytes 4..8:   input_width: u32
//! bytes 8..12:  input_height: u32
//! bytes 12..16: input_channels: u32
//! bytes 16..20: output_width: u32
//! bytes 20..24: output_height: u32
//! bytes 24..28: output_channels: u32
//! bytes 28..32: weights_width: u32
//! bytes 32..36: weights_height: u32
//! bytes 36..40: stride: u32
//! bytes 40..44: depthwise: u32             (0 or 1)
//! bytes 44..48: input_zero_point: u32
//! bytes 48..52: output_zero_point: u32
//! bytes 52..56: weights_zero_point: u32
//! bytes 56..60: input_scale: f32 bits
//! bytes 60..64: weights_scale: f32 bits
//! bytes 64..68: output_scale: f32 bits
//! bytes 68..72: truncate_bits: u32
//! bytes 72..76: activation_tag: u32        (0=None, 1=Relu, 2=Relux)
//! bytes 76..80: activation_cmp: u32        (always present; only meaningful if activation_tag==2)
//! bytes 80..84: precision_tag: u32         (0=Int8, 1=Fp16 -- a wire-format-owned
//!                                           tag, NOT Precision's own hardware-
//!                                           register encoding, which is a
//!                                           separate, independently-changeable
//!                                           detail)
//! ```
//! Total payload length (excluding whatever outer selector byte the caller
//! uses -- `rocket-hal-driver`'s tag byte, in practice): 84 bytes.
//!
//! This is `rocket-hal-driver`'s tag value `3` in `executable_cache.rs`'s
//! tag convention -- see that module's doc comment.

#[cfg(test)]
use crate::rocket::regcmd::plan_conv_tasks;
use crate::rocket::regcmd::{
    Activation, ConvShape, Precision, compute_out_shift, compute_task_input_channels,
    compute_task_output_channels,
};

pub const CONV2D_V1_FORMAT_VERSION: u32 = 1;

/// `rocket-hal-driver::executable_cache`'s outer tag-byte value meaning "a
/// real encoded `ConvShape` (this module's format) follows in the rest of
/// `executable_data`". Defined here (not in `rocket-hal-driver`) since this
/// crate owns the wire format itself; `rocket-hal-driver` just re-exports/
/// matches on it.
pub const CONV2D_V1_TAG: u8 = 3;

const FIELD_COUNT_U32: usize = 20; // everything below format_version, in order.
const PAYLOAD_LEN: usize = 4 + FIELD_COUNT_U32 * 4;

#[derive(Debug, PartialEq, Eq)]
pub enum DecodeError {
    WrongLength { expected: usize, actual: usize },
    UnsupportedVersion(u32),
    InvalidActivationTag(u32),
    InvalidPrecisionTag(u32),
}

/// Encodes `shape` per this module's documented layout, NOT including any
/// outer selector byte (that's the caller's concern -- see
/// `rocket-hal-driver::executable_cache`). Built via an exhaustive struct
/// destructure (no `..`) so adding or removing a `ConvShape` field is a
/// compile error here rather than a silent wire-format gap.
pub fn encode_conv_shape_v1(shape: &ConvShape) -> Vec<u8> {
    let ConvShape {
        input_width,
        input_height,
        input_channels,
        output_width,
        output_height,
        output_channels,
        weights_width,
        weights_height,
        stride,
        depthwise,
        input_zero_point,
        output_zero_point,
        weights_zero_point,
        input_scale,
        weights_scale,
        output_scale,
        truncate_bits,
        activation,
        precision,
    } = *shape;

    let (activation_tag, activation_cmp) = match activation {
        Activation::None => (0u32, 0u32),
        Activation::Relu => (1u32, 0u32),
        Activation::Relux { cmp } => (2u32, cmp),
    };
    let precision_tag: u32 = match precision {
        Precision::Int8 => 0,
        Precision::Fp16 => 1,
    };

    let mut out = Vec::with_capacity(PAYLOAD_LEN);
    out.extend_from_slice(&CONV2D_V1_FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&input_width.to_le_bytes());
    out.extend_from_slice(&input_height.to_le_bytes());
    out.extend_from_slice(&input_channels.to_le_bytes());
    out.extend_from_slice(&output_width.to_le_bytes());
    out.extend_from_slice(&output_height.to_le_bytes());
    out.extend_from_slice(&output_channels.to_le_bytes());
    out.extend_from_slice(&weights_width.to_le_bytes());
    out.extend_from_slice(&weights_height.to_le_bytes());
    out.extend_from_slice(&stride.to_le_bytes());
    out.extend_from_slice(&(depthwise as u32).to_le_bytes());
    out.extend_from_slice(&input_zero_point.to_le_bytes());
    out.extend_from_slice(&output_zero_point.to_le_bytes());
    out.extend_from_slice(&weights_zero_point.to_le_bytes());
    out.extend_from_slice(&input_scale.to_bits().to_le_bytes());
    out.extend_from_slice(&weights_scale.to_bits().to_le_bytes());
    out.extend_from_slice(&output_scale.to_bits().to_le_bytes());
    out.extend_from_slice(&truncate_bits.to_le_bytes());
    out.extend_from_slice(&activation_tag.to_le_bytes());
    out.extend_from_slice(&activation_cmp.to_le_bytes());
    out.extend_from_slice(&precision_tag.to_le_bytes());
    debug_assert_eq!(out.len(), PAYLOAD_LEN);
    out
}

/// Decodes a `ConvShape` from `payload` -- everything AFTER the outer
/// selector byte (`rocket-hal-driver`'s tag byte). Never silently falls
/// back to a default shape on malformed input; every failure mode is a
/// real `DecodeError`.
pub fn decode_conv_shape_v1(payload: &[u8]) -> Result<ConvShape, DecodeError> {
    if payload.len() != PAYLOAD_LEN {
        return Err(DecodeError::WrongLength {
            expected: PAYLOAD_LEN,
            actual: payload.len(),
        });
    }

    let mut cursor = payload;
    let mut read_u32 = || -> u32 {
        let (head, tail) = cursor.split_at(4);
        cursor = tail;
        u32::from_le_bytes(head.try_into().unwrap())
    };

    let version = read_u32();
    if version != CONV2D_V1_FORMAT_VERSION {
        return Err(DecodeError::UnsupportedVersion(version));
    }

    let input_width = read_u32();
    let input_height = read_u32();
    let input_channels = read_u32();
    let output_width = read_u32();
    let output_height = read_u32();
    let output_channels = read_u32();
    let weights_width = read_u32();
    let weights_height = read_u32();
    let stride = read_u32();
    let depthwise = read_u32() != 0;
    let input_zero_point = read_u32();
    let output_zero_point = read_u32();
    let weights_zero_point = read_u32();
    let input_scale = f32::from_bits(read_u32());
    let weights_scale = f32::from_bits(read_u32());
    let output_scale = f32::from_bits(read_u32());
    let truncate_bits = read_u32();
    let activation_tag = read_u32();
    let activation_cmp = read_u32();
    let precision_tag = read_u32();

    let activation = match activation_tag {
        0 => Activation::None,
        1 => Activation::Relu,
        2 => Activation::Relux {
            cmp: activation_cmp,
        },
        other => return Err(DecodeError::InvalidActivationTag(other)),
    };
    let precision = match precision_tag {
        0 => Precision::Int8,
        1 => Precision::Fp16,
        other => return Err(DecodeError::InvalidPrecisionTag(other)),
    };

    Ok(ConvShape {
        input_width,
        input_height,
        input_channels,
        output_width,
        output_height,
        output_channels,
        weights_width,
        weights_height,
        stride,
        depthwise,
        input_zero_point,
        output_zero_point,
        weights_zero_point,
        input_scale,
        weights_scale,
        output_scale,
        truncate_bits,
        activation,
        precision,
    })
}

/// Structural validation of a decoded `ConvShape` before it's ever handed
/// to `build_conv_regcmd`. This is deliberately NOT exhaustive of every
/// panic reachable from that function (see `command_buffer.rs`'s
/// `catch_unwind` backstop in `rocket-hal-driver` for the rest) -- it
/// mirrors the checks that are cheap and safe to duplicate (direct
/// booleans on raw fields, or calls into the exact same pure helper
/// functions `build_conv_cna_core_dpu_dpu_rdma`'s own `assert!`s use, so
/// there is exactly one copy of each formula, not two that can drift).
///
/// **Known, deliberate gap, left to the `catch_unwind` backstop**:
/// `CNA_WEIGHT_SIZE1.weight_bytes_per_kernel` is a 19-bit field fed
/// `weights_width * weights_height * task_input_channels * bpe` -- with
/// every individual factor within the bounds this function already
/// checks, the *product* can still overflow 19 bits (e.g.
/// `weights_width=31, weights_height=31, task_input_channels=65520,
/// bpe=1` = 62,932,920, far past the 19-bit max of 524,287). Closing this
/// would need yet another extracted pure helper purely to re-derive
/// `weight_bytes_per_kernel`'s formula for validation -- left as the
/// concrete, real example of why the `catch_unwind` backstop exists,
/// rather than chased indefinitely (see this crate's own `rocket-hal-
/// driver::command_buffer`'s CTS test exercising exactly this shape).
pub fn validate_conv_shape(shape: &ConvShape) -> Result<(), &'static str> {
    let input_channels_real_is_one = shape.input_channels == 1;

    if input_channels_real_is_one && shape.output_channels > 1 {
        return Err(
            "input_channels==1 && output_channels>1 needs fill_task()'s \"wide \
             atomic\" branch, not implemented in build_conv_regcmd",
        );
    }
    // Precision::Fp16 multi-channel: EXPERIMENTAL, see
    // build_conv_cna_core_dpu_dpu_rdma's own matching comment -- this used
    // to unconditionally reject it here; relaxed in lockstep with that
    // function's own assert so the wire-format decode path doesn't reject
    // shapes the regcmd builder is now willing to attempt.

    // Input tensors that need 12 or more CBUF banks are accepted here:
    // `build_conv_regcmd_tasks` splits them along height. The exact task
    // plan is intentionally owned by regcmd.rs rather than duplicated in
    // wire-format validation.
    if compute_out_shift(shape) < 0 {
        return Err("unsupported output conversion scale: computed out_shift is negative");
    }

    // Every width/height/channel field is written to hardware as N-1
    // (register convention: a 0-indexed count) in at least one place in
    // build_conv_cna_core_dpu_dpu_rdma -- a 0 here underflows that
    // subtraction (silently wraps in release builds, since neither
    // Cargo.toml enables overflow-checks, then fails the resulting huge
    // value's Bits::<N>::new bit-width assert; panics directly in debug
    // builds instead). input_height's own >=4 check below already implies
    // input_height>=1; the rest need their own explicit check.
    if shape.input_width == 0 {
        return Err("input_width must be at least 1 (written to hardware as input_width-1)");
    }
    if shape.output_width == 0 || shape.output_height == 0 {
        return Err("output_width/output_height must be at least 1 (written to hardware as N-1)");
    }
    if shape.input_channels == 0 {
        return Err("input_channels must be at least 1 (written to hardware as input_channels-1)");
    }
    if shape.output_channels == 0 {
        return Err(
            "output_channels must be at least 1 (written to hardware as output_channels-1)",
        );
    }

    // CNA_DATA_SIZE0/CNA_DATA_SIZE2/CORE_DATAOUT_SIZE0 are 11-bit fields.
    const MAX_11_BIT: u32 = (1 << 11) - 1;
    if shape.input_width > MAX_11_BIT || shape.input_height > MAX_11_BIT {
        return Err("input_width/input_height exceed the 11-bit CNA_DATA_SIZE field width");
    }
    if shape.output_width > MAX_11_BIT || shape.output_height > MAX_11_BIT {
        return Err(
            "output_width/output_height exceed the 11-bit CNA_DATA_SIZE2/CORE_DATAOUT_SIZE0 field width",
        );
    }
    // CNA_WEIGHT_SIZE2's weights_width/height are 5-bit fields.
    const MAX_5_BIT: u32 = (1 << 5) - 1;
    if shape.weights_width > MAX_5_BIT || shape.weights_height > MAX_5_BIT {
        return Err("weights_width/weights_height exceed the 5-bit CNA_WEIGHT_SIZE2 field width");
    }
    // CNA_CONV_CON3's conv_x_stride/conv_y_stride are 3-bit fields.
    const MAX_3_BIT: u32 = (1 << 3) - 1;
    if shape.stride > MAX_3_BIT {
        return Err("stride exceeds the 3-bit CNA_CONV_CON3 stride field width");
    }
    // CORE_CLIP_TRUNCATE's clip_truncate is a 5-bit field.
    if shape.truncate_bits > MAX_5_BIT {
        return Err("truncate_bits exceeds the 5-bit CORE_CLIP_TRUNCATE field width");
    }
    // CNA_DATA_SIZE1.datain_channel is a 16-bit field.
    const MAX_16_BIT: u32 = (1 << 16) - 1;
    if compute_task_input_channels(shape) > MAX_16_BIT {
        return Err(
            "input_channels' atomic-rounded task_input_channels exceeds the 16-bit CNA_DATA_SIZE1 field width",
        );
    }
    // DPU_..._CHANNEL/DPU_WDMA_..._CHANNEL_WDMA are 13-bit fields, fed
    // task_output_channels - 1 -- the tightest of the channel-count-derived
    // constraints (also transitively bounds weight_kernels' looser 14-bit
    // limit, since task_output_channels' 32-multiple rounding is always
    // >= weights_kernels' 2-multiple rounding for the same output_channels).
    const MAX_13_BIT: u32 = (1 << 13) - 1;
    // task_output_channels is always >= 32 (output_channels.max(32)), so
    // this subtraction never underflows.
    if compute_task_output_channels(shape) - 1 > MAX_13_BIT {
        return Err(
            "output_channels' atomic-rounded task_output_channels exceeds the 13-bit DPU channel field width",
        );
    }

    // input_line_stride * (input_height/4 - 1) underflows u32 for
    // input_height < 4 (silent wraparound, not a panic -- see this
    // function's own module doc comment / the plan this was written
    // against for why this is closeable by validation, unlike the
    // genuinely-open data-dependent int8 identity-weight bug).
    if shape.input_height < 4 {
        return Err("input_height < 4 underflows build_conv_regcmd's input_surface_stride formula");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `rkt-basic.rs`'s validated shape (4x4x1 -> 4x4x1, 1x1 kernel,
    /// int8) -- the same known-good baseline shape used throughout this
    /// crate's own hardware tests.
    fn known_good_shape() -> ConvShape {
        ConvShape {
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
        }
    }

    #[test]
    fn round_trip_known_good_shape() {
        let shape = known_good_shape();
        let bytes = encode_conv_shape_v1(&shape);
        assert_eq!(bytes.len(), PAYLOAD_LEN);
        let decoded = decode_conv_shape_v1(&bytes).expect("decode should succeed");
        assert_eq!(decoded, shape);
    }

    #[test]
    fn round_trip_every_activation_variant() {
        for activation in [
            Activation::None,
            Activation::Relu,
            Activation::Relux { cmp: 42 },
        ] {
            let shape = ConvShape {
                activation,
                ..known_good_shape()
            };
            let bytes = encode_conv_shape_v1(&shape);
            let decoded = decode_conv_shape_v1(&bytes).expect("decode should succeed");
            assert_eq!(decoded, shape);
        }
    }

    #[test]
    fn round_trip_every_precision_variant() {
        for precision in [Precision::Int8, Precision::Fp16] {
            let shape = ConvShape {
                precision,
                ..known_good_shape()
            };
            let bytes = encode_conv_shape_v1(&shape);
            let decoded = decode_conv_shape_v1(&bytes).expect("decode should succeed");
            assert_eq!(decoded, shape);
        }
    }

    #[test]
    fn decode_rejects_wrong_length() {
        let shape = known_good_shape();
        let mut bytes = encode_conv_shape_v1(&shape);
        bytes.pop();
        assert_eq!(
            decode_conv_shape_v1(&bytes),
            Err(DecodeError::WrongLength {
                expected: PAYLOAD_LEN,
                actual: PAYLOAD_LEN - 1,
            })
        );
        bytes.push(0);
        bytes.push(0);
        assert_eq!(
            decode_conv_shape_v1(&bytes),
            Err(DecodeError::WrongLength {
                expected: PAYLOAD_LEN,
                actual: PAYLOAD_LEN + 1,
            })
        );
    }

    #[test]
    fn decode_rejects_unsupported_version() {
        let shape = known_good_shape();
        let mut bytes = encode_conv_shape_v1(&shape);
        bytes[0..4].copy_from_slice(&2u32.to_le_bytes());
        assert_eq!(
            decode_conv_shape_v1(&bytes),
            Err(DecodeError::UnsupportedVersion(2))
        );
    }

    #[test]
    fn decode_rejects_invalid_activation_tag() {
        let shape = known_good_shape();
        let mut bytes = encode_conv_shape_v1(&shape);
        // activation_tag is the 18th u32 field (0-indexed among the 21
        // post-version fields): version(1) + 17 plain fields before
        // activation_tag = offset 4 + 17*4 = 72.
        bytes[72..76].copy_from_slice(&99u32.to_le_bytes());
        assert_eq!(
            decode_conv_shape_v1(&bytes),
            Err(DecodeError::InvalidActivationTag(99))
        );
    }

    #[test]
    fn decode_rejects_invalid_precision_tag() {
        let shape = known_good_shape();
        let mut bytes = encode_conv_shape_v1(&shape);
        // precision_tag is the last u32: offset 4 + 20*4 = 84.
        let last = bytes.len() - 4;
        bytes[last..].copy_from_slice(&7u32.to_le_bytes());
        assert_eq!(
            decode_conv_shape_v1(&bytes),
            Err(DecodeError::InvalidPrecisionTag(7))
        );
    }

    #[test]
    fn validate_accepts_known_good_shape() {
        assert_eq!(validate_conv_shape(&known_good_shape()), Ok(()));
    }

    #[test]
    fn validate_accepts_shape_that_requires_cbuf_height_splitting() {
        let shape = ConvShape {
            input_width: 112,
            input_height: 112,
            input_channels: 32,
            output_width: 112,
            output_height: 112,
            output_channels: 16,
            precision: Precision::Fp16,
            ..known_good_shape()
        };
        assert!(plan_conv_tasks(&shape).unwrap().len() > 1);
        assert_eq!(validate_conv_shape(&shape), Ok(()));
    }

    #[test]
    fn validate_rejects_wide_atomic_combo() {
        let shape = ConvShape {
            output_channels: 2,
            ..known_good_shape()
        };
        assert!(validate_conv_shape(&shape).is_err());
    }

    #[test]
    fn validate_allows_fp16_multichannel() {
        // Was `validate_rejects_fp16_multichannel` -- this combination is
        // now EXPERIMENTALLY allowed through (see `validate_conv_shape`'s
        // own comment and `build_conv_cna_core_dpu_dpu_rdma`'s matching
        // one), pending real hardware validation. Not yet hardware-
        // confirmed correct -- this test only asserts the shape isn't
        // rejected at validation time, matching the relaxed gate.
        let shape = ConvShape {
            input_channels: 3,
            precision: Precision::Fp16,
            ..known_good_shape()
        };
        assert_eq!(validate_conv_shape(&shape), Ok(()));
    }

    #[test]
    fn validate_rejects_oversized_weights() {
        let shape = ConvShape {
            weights_width: 32,
            ..known_good_shape()
        };
        assert!(validate_conv_shape(&shape).is_err());
    }

    #[test]
    fn validate_rejects_small_input_height() {
        let shape = ConvShape {
            input_height: 3,
            ..known_good_shape()
        };
        assert!(validate_conv_shape(&shape).is_err());
    }

    #[test]
    fn validate_rejects_oversized_stride() {
        let shape = ConvShape {
            stride: 8,
            ..known_good_shape()
        };
        assert!(validate_conv_shape(&shape).is_err());
    }

    #[test]
    fn validate_rejects_task_input_channels_overflowing_16_bits() {
        // This shape isolates the 16-bit CNA_DATA_SIZE1.datain_channel
        // check.
        let shape = ConvShape {
            input_width: 1,
            input_channels: 65535,
            output_channels: 1,
            ..known_good_shape()
        };
        assert!(validate_conv_shape(&shape).is_err());
    }

    #[test]
    fn validate_rejects_task_output_channels_overflowing_13_bits() {
        // input_channels=2 (not 1) so the "wide atomic" input_channels==1
        // && output_channels>1 check doesn't fire first, isolating the
        // 13-bit DPU channel-field check on output_channels instead.
        let shape = ConvShape {
            input_channels: 2,
            output_channels: 100_000,
            ..known_good_shape()
        };
        assert!(validate_conv_shape(&shape).is_err());
    }

    #[test]
    fn validate_rejects_zero_input_width() {
        let shape = ConvShape {
            input_width: 0,
            ..known_good_shape()
        };
        assert!(validate_conv_shape(&shape).is_err());
    }

    #[test]
    fn validate_rejects_zero_output_width_or_height() {
        assert!(
            validate_conv_shape(&ConvShape {
                output_width: 0,
                ..known_good_shape()
            })
            .is_err()
        );
        assert!(
            validate_conv_shape(&ConvShape {
                output_height: 0,
                ..known_good_shape()
            })
            .is_err()
        );
    }

    #[test]
    fn validate_rejects_zero_input_channels() {
        let shape = ConvShape {
            input_channels: 0,
            ..known_good_shape()
        };
        assert!(validate_conv_shape(&shape).is_err());
    }

    #[test]
    fn validate_rejects_zero_output_channels() {
        let shape = ConvShape {
            output_channels: 0,
            ..known_good_shape()
        };
        assert!(validate_conv_shape(&shape).is_err());
    }

    /// The known, deliberate gap this module's doc comment documents:
    /// every individual factor is within bounds this function checks, but
    /// their product overflows `CNA_WEIGHT_SIZE1.weight_bytes_per_kernel`'s
    /// 19-bit field. `validate_conv_shape` does NOT catch this -- it's the
    /// concrete shape `rocket-hal-driver`'s CTS test uses to prove the
    /// `catch_unwind` backstop in `command_buffer.rs` actually converts
    /// the resulting panic into a graceful error instead of aborting.
    #[test]
    fn validate_does_not_catch_weight_bytes_per_kernel_overflow() {
        let shape = ConvShape {
            input_width: 1,
            input_height: 4,
            input_channels: 65520,
            output_width: 4,
            output_height: 4,
            output_channels: 1,
            weights_width: 31,
            weights_height: 31,
            ..known_good_shape()
        };
        assert!(compute_task_input_channels(&shape) <= u16::MAX as u32);
        assert_eq!(
            validate_conv_shape(&shape),
            Ok(()),
            "this shape is expected to pass validation despite being unsafe to build -- \
             see this test's own doc comment"
        );
    }

    /// Empirical confirmation of the previous test's claim: the exact same
    /// shape, passed to the real builder, does panic (not just "would
    /// theoretically overflow the bit-width by hand-calculation"). This is
    /// the shape `rocket-hal-driver`'s CTS `conv_dispatch_test.cc` uses to
    /// exercise the `catch_unwind` backstop end to end.
    #[test]
    #[should_panic]
    fn weight_bytes_per_kernel_overflow_shape_actually_panics() {
        use crate::rocket::regcmd::{ConvBuffers, build_conv_regcmd};

        let shape = ConvShape {
            input_width: 1,
            input_height: 4,
            input_channels: 65520,
            output_width: 4,
            output_height: 4,
            output_channels: 1,
            weights_width: 31,
            weights_height: 31,
            ..known_good_shape()
        };
        let bufs = ConvBuffers {
            input_addr: 0,
            weights_addr: 0,
            bias_addr: 0,
            output_addr: 0,
        };
        let _ = build_conv_regcmd(&shape, &bufs);
    }
}
