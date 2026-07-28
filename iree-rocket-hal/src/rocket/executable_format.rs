//! Real, versioned binary wire format for [`conv::Shape`] + [`Kernels`],
//! replacing `rocket-hal-driver`'s one-byte "pick one of 3 hardcoded shapes"
//! tag hack with an actual encode/decode of shape/dtype/quantization data.
//! This is prep work for a real IREE compiler `TargetBackend` (not part of
//! this crate or `rocket-hal-driver` -- a separate, C++, out-of-tree
//! project): that backend's `serializeExecutable()` would emit exactly the
//! bytes `decode_conv_shape_v1` parses here.
//!
//! Version 2 replaced version 1's `mesa_conv::ConvShape`-shaped payload with
//! one that mirrors the capture-derived [`conv::Shape`] directly: dropping
//! `output_width`/`output_height` (now always derived from
//! input/kernel/stride/padding rather than carried as separate,
//! possibly-inconsistent fields) and `weights_zero_point` (no such concept
//! in [`conv::Quantization`]), adding explicit `pad_top`/`pad_left` (v1 had
//! no padding fields at all, so it could only describe valid convolutions),
//! and replacing the three raw `f32` scale fields with the hardware's own
//! normalized `Multiplier{scale, shift}` pair. There is no real external v1
//! producer yet, so this was a clean break rather than a compatible
//! extension.
//!
//! Layout, all little-endian, fixed total size (every field is a
//! fixed-size scalar, so there is no variable-length data to encode):
//!
//! ```text
//! bytes 0..4:   format_version: u32        (CONV2D_V1_FORMAT_VERSION == 2)
//! bytes 4..8:   input_width: u32
//! bytes 8..12:  input_height: u32
//! bytes 12..16: input_channels: u32
//! bytes 16..20: output_channels: u32
//! bytes 20..24: weights_width: u32         (Kernels[1])
//! bytes 24..28: weights_height: u32        (Kernels[0])
//! bytes 28..32: stride: u32
//! bytes 32..36: depthwise: u32             (0 or 1)
//! bytes 36..40: pad_top: u32
//! bytes 40..44: pad_left: u32
//! bytes 44..48: input_zero_point: i32      (only meaningful if precision_tag==0)
//! bytes 48..52: output_zero_point: i32     (only meaningful if precision_tag==0)
//! bytes 52..56: multiplier_scale: u32      (only meaningful if precision_tag==0)
//! bytes 56..60: multiplier_shift: u32      (only meaningful if precision_tag==0)
//! bytes 60..64: activation_tag: u32        (0=None, 1=Relu, 2=Clamped)
//! bytes 64..68: activation_cmp: u32        (always present; only meaningful if activation_tag==2)
//! bytes 68..72: precision_tag: u32         (0=Int8, 1=Fp16 -- a wire-format-owned
//!                                           tag, NOT Precision's own hardware-
//!                                           register encoding, which is a
//!                                           separate, independently-changeable
//!                                           detail)
//! ```
//! Total payload length (excluding whatever outer selector byte the caller
//! uses -- `rocket-hal-driver`'s tag byte, in practice): 72 bytes.
//!
//! This is `rocket-hal-driver`'s tag value `3` in `executable_cache.rs`'s
//! tag convention -- see that module's doc comment.

use crate::rocket::conv::{self, Activation, Kernels, Multiplier, Precision, Quantization};

pub const CONV2D_V1_FORMAT_VERSION: u32 = 2;

/// `rocket-hal-driver::executable_cache`'s outer tag-byte value meaning "a
/// real encoded `ConvShape` (this module's format) follows in the rest of
/// `executable_data`". Defined here (not in `rocket-hal-driver`) since this
/// crate owns the wire format itself; `rocket-hal-driver` just re-exports/
/// matches on it.
pub const CONV2D_V1_TAG: u8 = 3;

const FIELD_COUNT_U32: usize = 17; // everything below format_version, in order.
const PAYLOAD_LEN: usize = 4 + FIELD_COUNT_U32 * 4;

#[derive(Debug, PartialEq, Eq)]
pub enum DecodeError {
    WrongLength { expected: usize, actual: usize },
    UnsupportedVersion(u32),
    InvalidActivationTag(u32),
    InvalidPrecisionTag(u32),
}

/// Encodes `shape`/`kernels` per this module's documented layout, NOT
/// including any outer selector byte (that's the caller's concern -- see
/// `rocket-hal-driver::executable_cache`). Built via an exhaustive struct
/// destructure (no `..`) so adding or removing a [`conv::Shape`] field is a
/// compile error here rather than a silent wire-format gap.
pub fn encode_conv_shape_v1(shape: &conv::Shape, kernels: Kernels) -> Vec<u8> {
    let conv::Shape {
        width: input_width,
        height: input_height,
        stride,
        in_channels: input_channels,
        out_channels: output_channels,
        precision,
        padding,
        activation,
        depthwise,
    } = *shape;
    let [weights_height, weights_width] = kernels;
    let [pad_top, pad_left] = padding.unwrap_or([0, 0]);

    let (activation_tag, activation_cmp) = match activation {
        Activation::None => (0u32, 0u32),
        Activation::Relu => (1u32, 0u32),
        Activation::Clamped { cmp } => (2u32, cmp),
    };
    let (precision_tag, input_zero_point, output_zero_point, multiplier) = match precision {
        Precision::Fp16 => (1u32, 0i32, 0i32, Multiplier { scale: 0, shift: 0 }),
        Precision::Int8(Quantization {
            input_zero_point,
            output_zero_point,
            multiplier,
        }) => (0u32, input_zero_point, output_zero_point, multiplier),
    };

    let mut out = Vec::with_capacity(PAYLOAD_LEN);
    out.extend_from_slice(&CONV2D_V1_FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&input_width.to_le_bytes());
    out.extend_from_slice(&input_height.to_le_bytes());
    out.extend_from_slice(&input_channels.to_le_bytes());
    out.extend_from_slice(&output_channels.to_le_bytes());
    out.extend_from_slice(&(weights_width as u32).to_le_bytes());
    out.extend_from_slice(&(weights_height as u32).to_le_bytes());
    out.extend_from_slice(&stride.to_le_bytes());
    out.extend_from_slice(&(depthwise as u32).to_le_bytes());
    out.extend_from_slice(&(pad_top as u32).to_le_bytes());
    out.extend_from_slice(&(pad_left as u32).to_le_bytes());
    out.extend_from_slice(&input_zero_point.to_le_bytes());
    out.extend_from_slice(&output_zero_point.to_le_bytes());
    out.extend_from_slice(&multiplier.scale.to_le_bytes());
    out.extend_from_slice(&multiplier.shift.to_le_bytes());
    out.extend_from_slice(&activation_tag.to_le_bytes());
    out.extend_from_slice(&activation_cmp.to_le_bytes());
    out.extend_from_slice(&precision_tag.to_le_bytes());
    debug_assert_eq!(out.len(), PAYLOAD_LEN);
    out
}

/// Decodes a [`conv::Shape`] + [`Kernels`] from `payload` -- everything
/// AFTER the outer selector byte (`rocket-hal-driver`'s tag byte). Never
/// silently falls back to a default shape on malformed input; every failure
/// mode is a real `DecodeError`. Purely structural: the returned shape may
/// still be geometrically or hardware-invalid (zero extents, oversized
/// channels, unsupported kernel/padding combination, ...) -- that is
/// [`validate_conv_shape`]'s separate job, deliberately not duplicated here.
pub fn decode_conv_shape_v1(payload: &[u8]) -> Result<(conv::Shape, Kernels), DecodeError> {
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
    let output_channels = read_u32();
    let weights_width = read_u32();
    let weights_height = read_u32();
    let stride = read_u32();
    let depthwise = read_u32() != 0;
    let pad_top = read_u32();
    let pad_left = read_u32();
    let input_zero_point = read_u32() as i32;
    let output_zero_point = read_u32() as i32;
    let multiplier_scale = read_u32();
    let multiplier_shift = read_u32();
    let activation_tag = read_u32();
    let activation_cmp = read_u32();
    let precision_tag = read_u32();

    let activation = match activation_tag {
        0 => Activation::None,
        1 => Activation::Relu,
        2 => Activation::Clamped {
            cmp: activation_cmp,
        },
        other => return Err(DecodeError::InvalidActivationTag(other)),
    };
    let precision = match precision_tag {
        0 => Precision::Int8(Quantization {
            input_zero_point,
            output_zero_point,
            multiplier: Multiplier {
                scale: multiplier_scale,
                shift: multiplier_shift,
            },
        }),
        1 => Precision::Fp16,
        other => return Err(DecodeError::InvalidPrecisionTag(other)),
    };

    let shape = conv::Shape {
        width: input_width,
        height: input_height,
        stride,
        in_channels: input_channels,
        out_channels: output_channels,
        precision,
        padding: Some([pad_top as usize, pad_left as usize]),
        activation,
        depthwise,
    };
    let kernels: Kernels = [weights_height as usize, weights_width as usize];
    Ok((shape, kernels))
}

/// Structural validation of a decoded shape/kernels pair before either is
/// ever handed to [`conv::ConvPlan::new`]. Rather than re-deriving
/// `conv::Shape`'s and `ConvPlan`'s own bounds by hand (channel ranges,
/// padding fitting the CNA's 4-bit fields, CBUF/kernel-plan capacity, ...),
/// this rebuilds the shape through the exact same constructor chain a real
/// caller would use (`Shape::with_precision`/`with_padding`/`with_depthwise`)
/// and trial-plans it, catching whatever that chain's own `assert!`s reject.
/// This is a single source of truth for those bounds -- there is exactly
/// one place each one lives, in `conv.rs` itself -- rather than two copies
/// that can drift.
pub fn validate_conv_shape(shape: &conv::Shape, kernels: Kernels) -> Result<(), &'static str> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut rebuilt = conv::Shape::with_precision(
            shape.width,
            shape.height,
            shape.stride,
            shape.in_channels,
            shape.out_channels,
            shape.precision,
        );
        if let Some(padding) = shape.padding {
            rebuilt = rebuilt.with_padding(padding);
        }
        rebuilt = rebuilt.with_activation(shape.activation);
        if shape.depthwise {
            rebuilt = rebuilt.with_depthwise();
        }
        let _ = conv::ConvPlan::new(rebuilt, kernels);
    }))
    .map_err(|_| "convolution shape is not supported by the capture-derived planner")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `rkt-basic.rs`'s validated shape (4x4x1 -> 4x4x1, 1x1 kernel,
    /// int8) -- the same known-good baseline shape used throughout this
    /// crate's own hardware tests. `padding: Some([0, 0])`, never `None` --
    /// decode always produces `Some`, so a round trip starting from `None`
    /// would never compare equal.
    fn known_good_shape() -> (conv::Shape, Kernels) {
        (
            conv::Shape {
                width: 4,
                height: 4,
                stride: 1,
                in_channels: 1,
                out_channels: 1,
                precision: Precision::Int8(Quantization {
                    input_zero_point: 0,
                    output_zero_point: 0,
                    multiplier: Multiplier::from_ratio(1.0),
                }),
                padding: Some([0, 0]),
                activation: Activation::None,
                depthwise: false,
            },
            [1, 1],
        )
    }

    #[test]
    fn round_trip_known_good_shape() {
        let (shape, kernels) = known_good_shape();
        let bytes = encode_conv_shape_v1(&shape, kernels);
        assert_eq!(bytes.len(), PAYLOAD_LEN);
        let decoded = decode_conv_shape_v1(&bytes).expect("decode should succeed");
        assert_eq!(decoded, (shape, kernels));
    }

    #[test]
    fn round_trip_every_activation_variant() {
        let (base, kernels) = known_good_shape();
        for activation in [
            Activation::None,
            Activation::Relu,
            Activation::Clamped { cmp: 42 },
        ] {
            let shape = conv::Shape { activation, ..base };
            let bytes = encode_conv_shape_v1(&shape, kernels);
            let decoded = decode_conv_shape_v1(&bytes).expect("decode should succeed");
            assert_eq!(decoded, (shape, kernels));
        }
    }

    #[test]
    fn round_trip_every_precision_variant() {
        let (base, kernels) = known_good_shape();
        for precision in [
            Precision::Fp16,
            Precision::Int8(Quantization {
                input_zero_point: -7,
                output_zero_point: 3,
                multiplier: Multiplier::from_ratio(0.25),
            }),
        ] {
            let shape = conv::Shape { precision, ..base };
            let bytes = encode_conv_shape_v1(&shape, kernels);
            let decoded = decode_conv_shape_v1(&bytes).expect("decode should succeed");
            assert_eq!(decoded, (shape, kernels));
        }
    }

    #[test]
    fn decode_rejects_wrong_length() {
        let (shape, kernels) = known_good_shape();
        let mut bytes = encode_conv_shape_v1(&shape, kernels);
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
        let (shape, kernels) = known_good_shape();
        let mut bytes = encode_conv_shape_v1(&shape, kernels);
        bytes[0..4].copy_from_slice(&3u32.to_le_bytes());
        assert_eq!(
            decode_conv_shape_v1(&bytes),
            Err(DecodeError::UnsupportedVersion(3))
        );
    }

    #[test]
    fn decode_rejects_invalid_activation_tag() {
        let (shape, kernels) = known_good_shape();
        let mut bytes = encode_conv_shape_v1(&shape, kernels);
        // activation_tag is at bytes 60..64 per this module's byte-layout
        // doc comment.
        bytes[60..64].copy_from_slice(&99u32.to_le_bytes());
        assert_eq!(
            decode_conv_shape_v1(&bytes),
            Err(DecodeError::InvalidActivationTag(99))
        );
    }

    #[test]
    fn decode_rejects_invalid_precision_tag() {
        let (shape, kernels) = known_good_shape();
        let mut bytes = encode_conv_shape_v1(&shape, kernels);
        // precision_tag is the last u32: offset 4 + 17*4 = 72, so the last
        // 4 bytes of the 72-byte payload.
        let last = bytes.len() - 4;
        bytes[last..].copy_from_slice(&7u32.to_le_bytes());
        assert_eq!(
            decode_conv_shape_v1(&bytes),
            Err(DecodeError::InvalidPrecisionTag(7))
        );
    }

    #[test]
    fn validate_accepts_known_good_shape() {
        let (shape, kernels) = known_good_shape();
        assert_eq!(validate_conv_shape(&shape, kernels), Ok(()));
    }

    #[test]
    fn validate_accepts_shape_that_requires_multiple_tiles() {
        // `conv.rs`'s own `plan_programs_with_buffers_relocates_every_tile`
        // test confirms this exact shape/kernel pair plans into 3 tiles.
        let shape = conv::Shape::with_out_channels(256, 32, 1, 32, 64).with_padding([2, 2]);
        let kernels: Kernels = [5, 5];
        assert!(conv::ConvPlan::new(shape, kernels).tiles().len() > 1);
        assert_eq!(validate_conv_shape(&shape, kernels), Ok(()));
    }

    #[test]
    fn validate_allows_fp16_multichannel() {
        // fp16 multi-channel is capture-backed up to
        // `conv::MAX_INPUT_CHANNELS` -- this just confirms a small
        // multi-channel fp16 shape isn't spuriously rejected.
        let (base, kernels) = known_good_shape();
        let shape = conv::Shape {
            in_channels: 3,
            precision: Precision::Fp16,
            ..base
        };
        assert_eq!(validate_conv_shape(&shape, kernels), Ok(()));
    }

    #[test]
    fn validate_rejects_unsupported_kernel_extent() {
        // `ConvPlan` only plans kernel extents 1..=11 (see conv.rs's module
        // doc comment) -- 32 is far outside that.
        let (shape, _) = known_good_shape();
        assert!(validate_conv_shape(&shape, [32, 32]).is_err());
    }

    #[test]
    fn validate_rejects_zero_stride() {
        let (base, kernels) = known_good_shape();
        let shape = conv::Shape { stride: 0, ..base };
        assert!(validate_conv_shape(&shape, kernels).is_err());
    }

    #[test]
    fn validate_rejects_oversized_padding() {
        // `Shape::with_padding` asserts each axis fits the CNA's 4-bit pad
        // fields (max 15).
        let (base, kernels) = known_good_shape();
        let shape = conv::Shape {
            padding: Some([16, 0]),
            ..base
        };
        assert!(validate_conv_shape(&shape, kernels).is_err());
    }

    #[test]
    fn validate_rejects_zero_input_width() {
        let (base, kernels) = known_good_shape();
        let shape = conv::Shape { width: 0, ..base };
        assert!(validate_conv_shape(&shape, kernels).is_err());
    }

    #[test]
    fn validate_rejects_zero_input_channels() {
        let (base, kernels) = known_good_shape();
        let shape = conv::Shape {
            in_channels: 0,
            ..base
        };
        assert!(validate_conv_shape(&shape, kernels).is_err());
    }

    #[test]
    fn validate_rejects_zero_output_channels() {
        let (base, kernels) = known_good_shape();
        let shape = conv::Shape {
            out_channels: 0,
            ..base
        };
        assert!(validate_conv_shape(&shape, kernels).is_err());
    }

    #[test]
    fn validate_rejects_output_channels_outside_capture_backed_range() {
        let (base, kernels) = known_good_shape();
        let shape = conv::Shape {
            out_channels: 100_000,
            ..base
        };
        assert!(validate_conv_shape(&shape, kernels).is_err());
    }

    #[test]
    fn validate_rejects_input_channels_outside_capture_backed_range() {
        // Historical note: under the mesa_conv-era wire format, this exact
        // channel count (65520) PASSED validate_conv_shape and only panicked
        // later inside `build_conv_regcmd`, since mesa_conv's validator only
        // checked individual register field widths, not conv.rs's own
        // capture-backed channel range. Since validate_conv_shape now
        // rebuilds the shape through `conv::Shape`'s own constructors, this
        // is rejected outright, at prepare_executable time rather than
        // dispatch time -- see `rocket-hal-driver/cts/conv_dispatch_test.cc`'s
        // `Tag3RejectsChannelCountOutsideCaptureBackedRange` for the
        // end-to-end proof of that behavior change.
        let (base, kernels) = known_good_shape();
        let shape = conv::Shape {
            in_channels: 65520,
            precision: Precision::Int8(Quantization {
                input_zero_point: 0,
                output_zero_point: 0,
                multiplier: Multiplier::from_ratio(1.0),
            }),
            ..base
        };
        assert!(validate_conv_shape(&shape, kernels).is_err());
    }

    #[test]
    fn validate_rejects_depthwise_channel_mismatch() {
        // `Shape::with_depthwise` requires `out_channels == in_channels`.
        let (base, kernels) = known_good_shape();
        let shape = conv::Shape {
            in_channels: 4,
            out_channels: 8,
            depthwise: true,
            ..base
        };
        assert!(validate_conv_shape(&shape, kernels).is_err());
    }
}
