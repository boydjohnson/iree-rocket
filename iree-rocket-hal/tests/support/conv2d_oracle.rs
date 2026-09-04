use iree_rocket_hal::rocket::{
    conv::{
        BsEntry, Kernels, Multiplier, Precision, Quantization, Shape, bs_buffer_bytes,
        write_bs_buffer,
    },
    tensor_layout::{
        pack_hwcf_to_rocket_weights, pack_hwcf_to_rocket_weights_affine_i8,
        pack_hwcf_to_rocket_weights_int4_padded, pack_hwcf_to_rocket_weights_padded,
        pack_nhwc_to_nc1hwc2_int4, rocket_weight_storage_size, rocket_weight_storage_size_bits,
    },
};

pub const FEATURE_ATOM_BYTES: usize = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OraclePrecision {
    Fp16,
    /// fp16 inputs and coefficients with the fp32 accumulator kept, rather
    /// than narrowed back to fp16 on the way out.
    Fp16Accumulator,
    /// Signed 4-bit integers, two to a byte, accumulating into int16.
    Int4,
    /// tf32: fp16's mantissa with fp32's range, in a 4-byte container,
    /// accumulating into fp32.
    Tf32,
    /// bfloat16, the 2-byte float with fp32 range and an 8-bit mantissa.
    /// Shares every layout with fp16, so the same fixtures drive it.
    Bf16,
    /// Signed 16-bit integer inputs and coefficients, on the fp16 layout.
    Int16,
    Int8,
    Int8Accumulator,
}

impl OraclePrecision {
    pub fn name(self) -> &'static str {
        match self {
            Self::Fp16 => "fp16",
            Self::Fp16Accumulator => "fp16-f32out",
            Self::Int4 => "int4",
            Self::Tf32 => "tf32",
            Self::Bf16 => "bf16",
            Self::Int16 => "int16",
            Self::Int8 => "int8",
            Self::Int8Accumulator => "int8-accumulator",
        }
    }

    /// Whether the precision requantizes, which is what decides whether the
    /// case needs an output shift and a populated BS buffer.
    pub fn is_quantized(self) -> bool {
        matches!(self, Self::Int8)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OraclePattern {
    /// Every logical input and coefficient is one. This exercises every Cin
    /// lane and reduces the expected accumulator to Cin times valid taps.
    Counting,
    /// Three signed coefficients per output channel select distinct HWCF
    /// positions. Inputs vary in y, x, and channel, exposing permutations.
    Selectors { phase: usize },
    /// Signed selectors represented as ordinary affine int8 weights.
    SelectorsAffine { phase: usize },
    /// The same logical one-hot filter, but with the raw byte encoding from
    /// the int8 neutral-coefficient probe: 0x80 background/padding and 0x00
    /// for the live logical +1. This is a diagnostic, not yet a general ABI.
    OneHotNeutral80 { phase: usize, signed_input: bool },
    /// One K1/Cin1 output per possible coefficient byte. Output channel `c`
    /// receives raw byte `c`, while every padded coefficient is 0x80. A
    /// magnified output conversion exposes the retained coefficient fraction.
    RawByteSweep,
    /// The same all-256-byte K1/Cin1 sweep under the ordinary unit output
    /// conversion and zero output point used by synthetic int8 convolution.
    RawByteSweepUnit,
    /// Every tap and channel carries a distinct nonzero weight, unlike
    /// `Counting`'s uniform 1s or `Selectors`'/`SelectorsAffine`'s three
    /// nonzero taps per output with everything else zero. Weight and input
    /// magnitudes are kept small enough (verified by hand, not just by
    /// convention, against the exact shapes it targets) that every fp16
    /// accumulator this can produce stays exactly representable in fp16 --
    /// the comparison this oracle does is bit-exact, tolerance 0.0, and a
    /// dense pattern has no small-term-count shortcut to fall back on if
    /// that stops holding.
    Dense { phase: usize },
    /// One nonzero coefficient per output channel, with a deliberately
    /// **wide** magnitude on one side of the product.
    ///
    /// This is the pattern that tells a datatype apart from a narrower one
    /// sharing its layout. Every other pattern here keeps its operands
    /// inside int8 and its accumulators inside fp16, so an int16 convolution
    /// that was really reading bytes, or a bf16 one that was really reading
    /// fp16, would still pass them. A single live tap keeps the accumulator
    /// equal to one product, so the magnitude can be pushed to the edge of
    /// the datatype without overflowing the output container.
    ///
    /// `wide_input` puts the wide magnitude on the feature side and holds
    /// the coefficient at one; clearing it swaps the two, which is what
    /// separates a wide feature path from a wide coefficient path.
    WideOperands { phase: usize, wide_input: bool },
}

/// The wide magnitude [`OraclePattern::WideOperands`] uses, per precision.
///
/// int16 wants a value past int8 and inside int16. bf16 wants one past
/// fp16's 65504 ceiling and exactly representable in bf16, which is what an
/// eight-bit mantissa times a power of two gives.
///
/// int4 is the exception: nothing is narrower than a nibble for it to be
/// mistaken for, so its wide cases test the nibble's own *endpoints*
/// instead. Every other pattern here stays inside [-3, 3], which leaves a
/// nibble's top bit -- and its asymmetric minimum -- unexercised, so a
/// sign-extension error in the packers or in the CORE would not show. The
/// caller negates on the same parity this switches on, so the two values
/// that reach hardware are exactly +7 and -8.
fn wide_magnitude(precision: OraclePrecision, index: usize, phase: usize) -> i32 {
    let step = ((index + phase * 7) % 13) as i32;
    match precision {
        OraclePrecision::Int16 => 3000 + 137 * step,
        // Eight significant bits times a power of two: exact in bf16, and
        // therefore also in tf32's wider ten-bit mantissa, while sitting an
        // order of magnitude past fp16's 65504 ceiling.
        OraclePrecision::Bf16 | OraclePrecision::Tf32 => (128 + 9 * step) << 12,
        OraclePrecision::Int4 => {
            if (index + phase) % 2 == 0 {
                7
            } else {
                8
            }
        }
        other => panic!("{other:?} has no wide-operand magnitude"),
    }
}

impl OraclePattern {
    pub fn name(self) -> &'static str {
        match self {
            Self::Counting => "counting",
            Self::Selectors { .. } => "selectors",
            Self::SelectorsAffine { .. } => "selectors-affine",
            Self::OneHotNeutral80 {
                signed_input: false,
                ..
            } => "onehot-neutral80-positive",
            Self::OneHotNeutral80 {
                signed_input: true, ..
            } => "onehot-neutral80-signed",
            Self::RawByteSweep => "raw-byte-sweep",
            Self::RawByteSweepUnit => "raw-byte-sweep-unit",
            Self::Dense { .. } => "dense",
            Self::WideOperands {
                wide_input: true, ..
            } => "wide-input",
            Self::WideOperands {
                wide_input: false, ..
            } => "wide-weight",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Conv2dCase {
    pub width: u32,
    pub height: u32,
    pub cin: u32,
    pub cout: u32,
    pub kernel: Kernels,
    pub stride: u32,
    pub padding: [usize; 2],
    pub precision: OraclePrecision,
    pub pattern: OraclePattern,
}

impl Conv2dCase {
    pub fn output_shift(self) -> u32 {
        if !self.precision.is_quantized()
            || matches!(
                self.pattern,
                OraclePattern::Selectors { .. }
                    | OraclePattern::SelectorsAffine { .. }
                    | OraclePattern::OneHotNeutral80 { .. }
                    | OraclePattern::RawByteSweep
                    | OraclePattern::RawByteSweepUnit
                    | OraclePattern::Dense { .. }
            )
        {
            return 0;
        }
        let peak = self.cin * (self.kernel[0] * self.kernel[1]) as u32;
        let mut shift = 0;
        while (peak >> shift) > 127 {
            shift += 1;
        }
        shift
    }

    pub fn shape(self) -> Shape {
        let precision = match self.precision {
            OraclePrecision::Fp16 => Precision::Fp16,
            OraclePrecision::Fp16Accumulator => Precision::Fp16Accumulator,
            OraclePrecision::Int4 => Precision::Int4,
            OraclePrecision::Tf32 => Precision::Tf32,
            OraclePrecision::Bf16 => Precision::Bf16,
            OraclePrecision::Int16 => Precision::Int16,
            OraclePrecision::Int8 => {
                let (output_zero_point, multiplier) = match self.pattern {
                    OraclePattern::RawByteSweep => (-128, Multiplier::for_unit_bs(128.0)),
                    OraclePattern::Counting | OraclePattern::SelectorsAffine { .. } => (
                        0,
                        Multiplier::from_ratio(1.0 / f64::from(1u32 << self.output_shift())),
                    ),
                    _ => (
                        0,
                        Multiplier::for_unit_bs(1.0 / f64::from(1u32 << self.output_shift())),
                    ),
                };
                Precision::Int8(Quantization {
                    input_zero_point: 0,
                    output_zero_point,
                    weight_zero_point: 0,
                    input_scale: 1.0,
                    weights_scale: 1.0,
                    multiplier,
                })
            }
            OraclePrecision::Int8Accumulator => Precision::Int8Accumulator(Quantization {
                input_zero_point: 0,
                output_zero_point: 0,
                weight_zero_point: 0,
                input_scale: 1.0,
                weights_scale: 1.0,
                multiplier: Multiplier::from_ratio(1.0),
            }),
        };
        Shape::with_precision(
            self.width,
            self.height,
            self.stride,
            self.cin,
            self.cout,
            precision,
        )
        .with_padding(self.padding)
    }

    pub fn label(self) -> String {
        format!(
            "{} {}x{} Cin={} Cout={} K={}x{} S={} P={}x{} {} >>{}",
            self.precision.name(),
            self.width,
            self.height,
            self.cin,
            self.cout,
            self.kernel[0],
            self.kernel[1],
            self.stride,
            self.padding[0],
            self.padding[1],
            self.pattern.name(),
            self.output_shift(),
        )
    }
}

pub struct Conv2dFixture {
    pub case: Conv2dCase,
    pub shape: Shape,
    pub input: Vec<u8>,
    pub weights: Vec<u8>,
    pub bias: Vec<u8>,
}

pub fn element_bytes(shape: Shape) -> usize {
    shape.precision.element_bytes() as usize
}

pub fn channels_per_atom(shape: Shape) -> usize {
    shape.precision.channels_per_atom() as usize
}

pub fn input_storage_bytes(shape: Shape) -> usize {
    let pixels = shape.width as usize * shape.height as usize;
    match shape.layout() {
        iree_rocket_hal::rocket::conv::FeatureLayout::Dense => {
            pixels * shape.in_channels as usize * element_bytes(shape)
        }
        iree_rocket_hal::rocket::conv::FeatureLayout::Surfaces => {
            shape.feature_atoms() as usize * pixels * FEATURE_ATOM_BYTES
        }
    }
}

pub fn output_storage_bytes(shape: Shape, kernels: Kernels) -> usize {
    shape.output_scratch_bytes(kernels)
}

pub fn feature_offset(shape: Shape, channel: usize, y: usize, x: usize) -> usize {
    let width = shape.width as usize;
    match shape.layout() {
        iree_rocket_hal::rocket::conv::FeatureLayout::Dense => {
            ((y * width + x) * shape.in_channels as usize + channel) * element_bytes(shape)
        }
        iree_rocket_hal::rocket::conv::FeatureLayout::Surfaces => {
            let atom_channels = channels_per_atom(shape);
            (channel / atom_channels) * width * shape.height as usize * FEATURE_ATOM_BYTES
                + (y * width + x) * FEATURE_ATOM_BYTES
                + (channel % atom_channels) * element_bytes(shape)
        }
    }
}

pub fn output_offset(shape: Shape, kernels: Kernels, channel: usize, y: usize, x: usize) -> usize {
    let atom_bytes = shape.output_atom_bytes() as usize;
    let element_bytes = shape.precision.output_element_bytes() as usize;
    let atom_channels = atom_bytes / element_bytes;
    let surface_bytes =
        shape.output_width(kernels) as usize * shape.output_height(kernels) as usize * atom_bytes;
    (channel / atom_channels) * surface_bytes
        + (y * shape.output_width(kernels) as usize + x) * atom_bytes
        + (channel % atom_channels) * element_bytes
}

pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    if value == 0.0 {
        return sign;
    }
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let fraction = bits & 0x7f_ffff;
    assert!(
        (1..31).contains(&exponent),
        "{value} is outside the fp16 normal range"
    );
    assert_eq!(fraction & 0x1fff, 0, "{value} is not exact in fp16");
    sign | ((exponent as u16) << 10) | ((fraction >> 13) as u16)
}

/// Encodes a value that must be exactly representable in bf16.
///
/// bf16 is the top 16 bits of an fp32, so the encoding is a truncation and
/// the exactness check is that the discarded 16 bits are zero -- the same
/// shape of assertion [`f32_to_f16`] makes, one mantissa width over. Keeping
/// it an assertion rather than a rounding is what lets the hardware
/// comparison run at tolerance 0.0: an inexact fixture fails on the host
/// instead of arriving on the board as an ambiguous near-miss.
pub fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    assert_eq!(bits & 0xffff, 0, "{value} is not exact in bf16");
    (bits >> 16) as u16
}

pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Whether an integer accumulator survives a round trip through bf16.
///
/// bf16 carries nine significant bits, so this is the predicate a bf16
/// hardware case has to satisfy for a tolerance-0.0 comparison to be fair.
pub fn is_exact_in_bf16(value: i32) -> bool {
    (value as f32).to_bits() & 0xffff == 0 && value as f32 as i32 == value
}

/// One bf16 ulp at `value` -- the spacing between adjacent bf16 numbers of
/// that magnitude.
///
/// This is the tolerance a bf16 comparison needs when the exact accumulator
/// has more than nine significant bits: the hardware converts an fp32
/// accumulator on the way out, and which way it breaks a tie is not
/// established, so a case that cannot be exact is held to one ulp rather
/// than to a guessed rounding mode. Any layout, addressing or coverage
/// error is orders of magnitude larger than this.
pub fn bf16_ulp(value: f32) -> f32 {
    if value == 0.0 {
        return f32::from_bits(1 << 16);
    }
    let exponent_bits = value.abs().to_bits() & 0x7f80_0000;
    // Eight explicit mantissa bits, so the step is 2^-8 of the binade.
    f32::from_bits(exponent_bits) / 256.0
}

/// Whether an integer is exactly representable in fp16.
///
/// The predicate the fp32-result ladder turns on: fp16 carries an 11-bit
/// significand, so it holds every integer to 2048 and then starts skipping.
/// A case whose accumulator fails this is one the ordinary fp16 output
/// could not have returned exactly, which is the whole reason to keep the
/// accumulator.
pub fn is_exact_in_fp16(value: i32) -> bool {
    let magnitude = value.unsigned_abs();
    if magnitude == 0 {
        return true;
    }
    magnitude <= 65504 && (32 - magnitude.leading_zeros()) <= 11
}

pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    let word = match exp {
        0 if frac == 0 => sign << 31,
        0x1f => (sign << 31) | 0x7f80_0000 | (frac << 13),
        0 => {
            let mut exponent = -1i32;
            let mut mantissa = frac;
            while mantissa & 0x400 == 0 {
                mantissa <<= 1;
                exponent -= 1;
            }
            (sign << 31) | (((exponent + 127 - 15) as u32) << 23) | ((mantissa & 0x3ff) << 13)
        }
        _ => (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13),
    };
    f32::from_bits(word)
}

/// A logical input element.
///
/// Logical values are `i32` rather than `i8` so a pattern can exercise an
/// operand wider than a byte. Every pattern that predates the wider
/// datatypes stays inside int8 -- `logical_values_fit_the_precision` checks
/// that, so an int8 case cannot silently truncate.
fn input_value(case: Conv2dCase, y: usize, x: usize, channel: usize) -> i32 {
    match case.pattern {
        OraclePattern::Counting => 1,
        OraclePattern::Selectors { phase }
        | OraclePattern::SelectorsAffine { phase }
        | OraclePattern::Dense { phase } => {
            ((y * 13 + x * 7 + channel * 3 + (y * x) % 5 + phase) % 7) as i32 - 3
        }
        OraclePattern::OneHotNeutral80 {
            phase,
            signed_input,
        } => {
            let linear = (y * case.width as usize + x) * case.cin as usize + channel;
            let magnitude = 1 + ((linear + phase * 17) % 61) as i32;
            if signed_input && (linear + phase) % 2 != 0 {
                -magnitude
            } else {
                magnitude
            }
        }
        OraclePattern::RawByteSweep | OraclePattern::RawByteSweepUnit => 1,
        OraclePattern::WideOperands { phase, wide_input } => {
            if !wide_input {
                // The narrow side alternates sign so a sign-extension bug on
                // the wide coefficient still shows up.
                return 1 - 2 * ((y + x + channel) % 2) as i32;
            }
            let linear = (y * case.width as usize + x) * case.cin as usize + channel;
            let magnitude = wide_magnitude(case.precision, linear, phase);
            if (linear + phase) % 2 == 0 {
                magnitude
            } else {
                -magnitude
            }
        }
    }
}

fn selector_weights(case: Conv2dCase, output_channel: usize) -> [(usize, i32); 3] {
    let slots = case.kernel[0] * case.kernel[1] * case.cin as usize;
    assert!(
        slots >= 3,
        "selector oracle needs at least three HWCF slots"
    );
    let phase = match case.pattern {
        OraclePattern::Selectors { phase } | OraclePattern::SelectorsAffine { phase } => phase,
        OraclePattern::Counting
        | OraclePattern::OneHotNeutral80 { .. }
        | OraclePattern::RawByteSweep
        | OraclePattern::RawByteSweepUnit
        | OraclePattern::Dense { .. }
        | OraclePattern::WideOperands { .. } => 0,
    };
    let first = (output_channel * 17 + phase * 11) % slots;
    let step = (slots / 3).max(1);
    [
        (first, 1),
        ((first + step) % slots, -1),
        ((first + 2 * step) % slots, 2),
    ]
}

fn one_hot_selector(case: Conv2dCase, output_channel: usize) -> usize {
    let slots = case.kernel[0] * case.kernel[1] * case.cin as usize;
    let phase = match case.pattern {
        OraclePattern::OneHotNeutral80 { phase, .. }
        | OraclePattern::WideOperands { phase, .. } => phase,
        OraclePattern::Counting
        | OraclePattern::Selectors { .. }
        | OraclePattern::SelectorsAffine { .. }
        | OraclePattern::RawByteSweep
        | OraclePattern::RawByteSweepUnit
        | OraclePattern::Dense { .. } => 0,
    };
    (output_channel * 17 + phase * 11) % slots
}

fn decode_selector(case: Conv2dCase, selector: usize) -> (usize, usize, usize) {
    let kernel_row = case.kernel[1] * case.cin as usize;
    let ky = selector / kernel_row;
    let remainder = selector % kernel_row;
    let kx = remainder / case.cin as usize;
    let channel = remainder % case.cin as usize;
    (ky, kx, channel)
}

fn weight_value(
    case: Conv2dCase,
    ky: usize,
    kx: usize,
    input_channel: usize,
    output_channel: usize,
) -> i32 {
    match case.pattern {
        OraclePattern::Counting => 1,
        OraclePattern::Selectors { .. } | OraclePattern::SelectorsAffine { .. } => {
            let selector = (ky * case.kernel[1] + kx) * case.cin as usize + input_channel;
            selector_weights(case, output_channel)
                .into_iter()
                .find_map(|(selected, weight)| (selected == selector).then_some(weight))
                .unwrap_or(0)
        }
        OraclePattern::OneHotNeutral80 { .. } => {
            let selector = (ky * case.kernel[1] + kx) * case.cin as usize + input_channel;
            i32::from(selector == one_hot_selector(case, output_channel))
        }
        OraclePattern::RawByteSweep | OraclePattern::RawByteSweepUnit => {
            assert_eq!((case.kernel, case.cin), ([1, 1], 1));
            i32::from(output_channel as u8 as i8)
        }
        OraclePattern::WideOperands { phase, wide_input } => {
            let selector = (ky * case.kernel[1] + kx) * case.cin as usize + input_channel;
            if selector != one_hot_selector(case, output_channel) {
                return 0;
            }
            if wide_input {
                1
            } else {
                let magnitude = wide_magnitude(case.precision, output_channel, phase);
                if (output_channel + phase) % 2 == 0 {
                    magnitude
                } else {
                    -magnitude
                }
            }
        }
        OraclePattern::Dense { phase } => {
            // Every tap gets one of four small nonzero values -- dense,
            // unlike Selectors' near-total sparsity, but kept small so the
            // accumulator stays exactly representable in fp16 (verified by
            // hand for the shapes `Dense` targets; see its doc comment).
            const VALUES: [i32; 4] = [-2, -1, 1, 2];
            let hash =
                (ky * 97) ^ (kx * 89) ^ (input_channel * 13) ^ (output_channel * 7) ^ (phase * 3);
            VALUES[hash % 4]
        }
    }
}

fn rounded_shift(value: i32, shift: u32) -> i32 {
    if shift == 0 {
        return value;
    }
    let half = 1i32 << (shift - 1);
    if value >= 0 {
        (value + half) >> shift
    } else {
        -((-value + half) >> shift)
    }
}

pub fn expected_accumulator(
    case: Conv2dCase,
    output_channel: usize,
    output_y: usize,
    output_x: usize,
) -> i32 {
    let input_origin_y = output_y as isize * case.stride as isize - case.padding[0] as isize;
    let input_origin_x = output_x as isize * case.stride as isize - case.padding[1] as isize;
    match case.pattern {
        OraclePattern::Counting => {
            let mut valid_taps = 0i32;
            for ky in 0..case.kernel[0] {
                let input_y = input_origin_y + ky as isize;
                if !(0..case.height as isize).contains(&input_y) {
                    continue;
                }
                for kx in 0..case.kernel[1] {
                    let input_x = input_origin_x + kx as isize;
                    if (0..case.width as isize).contains(&input_x) {
                        valid_taps += 1;
                    }
                }
            }
            valid_taps * case.cin as i32
        }
        OraclePattern::Selectors { .. } | OraclePattern::SelectorsAffine { .. } => {
            selector_weights(case, output_channel)
                .into_iter()
                .filter_map(|(selector, weight)| {
                    let (ky, kx, channel) = decode_selector(case, selector);
                    let input_y = input_origin_y + ky as isize;
                    let input_x = input_origin_x + kx as isize;
                    ((0..case.height as isize).contains(&input_y)
                        && (0..case.width as isize).contains(&input_x))
                    .then(|| {
                        i32::from(input_value(
                            case,
                            input_y as usize,
                            input_x as usize,
                            channel,
                        )) * i32::from(weight)
                    })
                })
                .sum()
        }
        OraclePattern::WideOperands { .. } => {
            let (ky, kx, channel) = decode_selector(case, one_hot_selector(case, output_channel));
            let input_y = input_origin_y + ky as isize;
            let input_x = input_origin_x + kx as isize;
            if (0..case.height as isize).contains(&input_y)
                && (0..case.width as isize).contains(&input_x)
            {
                input_value(case, input_y as usize, input_x as usize, channel)
                    * weight_value(case, ky, kx, channel, output_channel)
            } else {
                0
            }
        }
        OraclePattern::OneHotNeutral80 { .. } => {
            let (ky, kx, channel) = decode_selector(case, one_hot_selector(case, output_channel));
            let input_y = input_origin_y + ky as isize;
            let input_x = input_origin_x + kx as isize;
            if (0..case.height as isize).contains(&input_y)
                && (0..case.width as isize).contains(&input_x)
            {
                input_value(case, input_y as usize, input_x as usize, channel)
            } else {
                0
            }
        }
        OraclePattern::RawByteSweep | OraclePattern::RawByteSweepUnit => {
            i32::from(output_channel as u8 as i8)
        }
        OraclePattern::Dense { .. } => {
            let mut sum = 0i32;
            for ky in 0..case.kernel[0] {
                let input_y = input_origin_y + ky as isize;
                if !(0..case.height as isize).contains(&input_y) {
                    continue;
                }
                for kx in 0..case.kernel[1] {
                    let input_x = input_origin_x + kx as isize;
                    if !(0..case.width as isize).contains(&input_x) {
                        continue;
                    }
                    for channel in 0..case.cin as usize {
                        sum += i32::from(input_value(
                            case,
                            input_y as usize,
                            input_x as usize,
                            channel,
                        )) * i32::from(weight_value(case, ky, kx, channel, output_channel));
                    }
                }
            }
            sum
        }
    }
}

pub fn expected_output(
    case: Conv2dCase,
    output_channel: usize,
    output_y: usize,
    output_x: usize,
) -> i32 {
    let accumulator = expected_accumulator(case, output_channel, output_y, output_x);
    if case.precision.is_quantized() {
        rounded_shift(accumulator, case.output_shift()).clamp(-128, 127)
    } else {
        accumulator
    }
}

pub fn dense_reference(case: Conv2dCase, input: &[i32], weights: &[i32], bias: &[i32]) -> Vec<i32> {
    let shape = case.shape();
    let out_height = shape.output_height(case.kernel) as usize;
    let out_width = shape.output_width(case.kernel) as usize;
    assert_eq!(
        input.len(),
        case.height as usize * case.width as usize * case.cin as usize
    );
    assert_eq!(
        weights.len(),
        case.kernel[0] * case.kernel[1] * case.cin as usize * case.cout as usize
    );
    assert_eq!(bias.len(), case.cout as usize);

    let mut output = vec![0; out_height * out_width * case.cout as usize];
    for oy in 0..out_height {
        for ox in 0..out_width {
            for oc in 0..case.cout as usize {
                let mut accumulator = bias[oc];
                for ky in 0..case.kernel[0] {
                    let iy =
                        oy as isize * case.stride as isize + ky as isize - case.padding[0] as isize;
                    if !(0..case.height as isize).contains(&iy) {
                        continue;
                    }
                    for kx in 0..case.kernel[1] {
                        let ix = ox as isize * case.stride as isize + kx as isize
                            - case.padding[1] as isize;
                        if !(0..case.width as isize).contains(&ix) {
                            continue;
                        }
                        for ic in 0..case.cin as usize {
                            let input_index = ((iy as usize * case.width as usize + ix as usize)
                                * case.cin as usize)
                                + ic;
                            let weight_index = (((ky * case.kernel[1] + kx) * case.cin as usize
                                + ic)
                                * case.cout as usize)
                                + oc;
                            accumulator += input[input_index] * weights[weight_index];
                        }
                    }
                }
                output[(oy * out_width + ox) * case.cout as usize + oc] = accumulator;
            }
        }
    }
    output
}

fn logical_input(case: Conv2dCase) -> Vec<i32> {
    let mut input = vec![0; case.height as usize * case.width as usize * case.cin as usize];
    for y in 0..case.height as usize {
        for x in 0..case.width as usize {
            for channel in 0..case.cin as usize {
                input[(y * case.width as usize + x) * case.cin as usize + channel] =
                    input_value(case, y, x, channel);
            }
        }
    }
    input
}

fn logical_weights(case: Conv2dCase) -> Vec<i32> {
    let mut weights =
        vec![0; case.kernel[0] * case.kernel[1] * case.cin as usize * case.cout as usize];
    for ky in 0..case.kernel[0] {
        for kx in 0..case.kernel[1] {
            for input_channel in 0..case.cin as usize {
                for output_channel in 0..case.cout as usize {
                    let index = (((ky * case.kernel[1] + kx) * case.cin as usize + input_channel)
                        * case.cout as usize)
                        + output_channel;
                    weights[index] = weight_value(case, ky, kx, input_channel, output_channel);
                }
            }
        }
    }
    weights
}

fn uses_affine_int8_weights(case: Conv2dCase) -> bool {
    case.precision == OraclePrecision::Int8
        && matches!(
            case.pattern,
            OraclePattern::Counting | OraclePattern::SelectorsAffine { .. }
        )
}

fn affine_weight_zero_points(case: Conv2dCase) -> Vec<i8> {
    const ZERO_POINTS: [i8; 5] = [-127, -43, 0, 42, 125];
    (0..case.cout as usize)
        .map(|channel| ZERO_POINTS[channel % ZERO_POINTS.len()])
        .collect()
}

/// Encodes one logical value in the storage form its precision uses.
///
/// The single place a logical `i32` becomes hardware bytes, so each
/// datatype's representable range is checked once rather than separately for
/// the feature and the coefficient buffer. A value the datatype cannot hold
/// is an error rather than a silent truncation -- that is what keeps a
/// fixture bug from arriving on the board looking like a hardware result.
fn encode_element(precision: OraclePrecision, value: i32) -> Result<Vec<u8>, String> {
    match precision {
        // The operands are fp16 on both, so the exactness bound is fp16's
        // whichever container the result lands in.
        OraclePrecision::Fp16 | OraclePrecision::Fp16Accumulator => {
            let exact = i32::from(i16::try_from(value).unwrap_or(i16::MAX)) == value
                && (value as f32) as i32 == value
                && value.abs() <= 2048;
            if !exact {
                return Err(format!("logical value {value} is not exact in fp16"));
            }
            Ok(f32_to_f16(value as f32).to_le_bytes().to_vec())
        }
        OraclePrecision::Bf16 => {
            if !is_exact_in_bf16(value) {
                return Err(format!("logical value {value} is not exact in bf16"));
            }
            Ok(f32_to_bf16(value as f32).to_le_bytes().to_vec())
        }
        OraclePrecision::Int16 => {
            let value = i16::try_from(value)
                .map_err(|_| format!("logical value {value} is outside int16"))?;
            Ok(value.to_le_bytes().to_vec())
        }
        OraclePrecision::Int8 | OraclePrecision::Int8Accumulator => {
            let value = i8::try_from(value)
                .map_err(|_| format!("logical value {value} is outside int8"))?;
            Ok(vec![value as u8])
        }
        // tf32 is fed as raw fp32; the hardware rounds the mantissa.
        OraclePrecision::Tf32 => Ok((value as f32).to_le_bytes().to_vec()),
        // int4 is not byte-addressable, so it never reaches this path: its
        // fixtures go through the nibble packers instead.
        OraclePrecision::Int4 => Err("int4 elements are packed as nibbles".to_string()),
    }
}

/// The int4 fixture, built through the nibble packers.
///
/// Kept separate rather than threaded through the byte path: at half a byte
/// an element has no address of its own, so the per-element
/// `feature_offset`/`encode_element` route cannot express it. Everything
/// else -- the logical values, the reference, the comparison -- is shared,
/// so an int4 case is the same case as its int8 twin apart from the storage.
fn build_int4_fixture(case: Conv2dCase, shape: Shape) -> Result<Conv2dFixture, String> {
    let to_int4 = |values: Vec<i32>, what: &str| -> Result<Vec<i8>, String> {
        values
            .into_iter()
            .map(|value| {
                i8::try_from(value)
                    .ok()
                    .filter(|value| (-8..=7).contains(value))
                    .ok_or_else(|| format!("logical {what} {value} is outside int4"))
            })
            .collect()
    };

    let dense_input = to_int4(logical_input(case), "input")?;
    let mut input = vec![0; input_storage_bytes(shape)];
    pack_nhwc_to_nc1hwc2_int4(
        &dense_input,
        case.height as usize * case.width as usize,
        case.cin as usize,
        &mut input,
    )
    .map_err(str::to_string)?;

    let dense_weights = to_int4(logical_weights(case), "weight")?;
    let packed_len = rocket_weight_storage_size_bits(
        case.kernel[0],
        case.kernel[1],
        case.cin as usize,
        shape.out_channels as usize,
        4,
    )
    .map_err(str::to_string)?;
    let mut weights = vec![0; packed_len];
    pack_hwcf_to_rocket_weights_int4_padded(
        &dense_weights,
        case.kernel[0],
        case.kernel[1],
        case.cin as usize,
        case.cout as usize,
        shape.out_channels as usize,
        &mut weights,
    )
    .map_err(str::to_string)?;
    if weights.len() != shape.weight_bytes(case.kernel) as usize {
        return Err(format!(
            "packed int4 weight size {} does not match Shape::weight_bytes {}",
            weights.len(),
            shape.weight_bytes(case.kernel),
        ));
    }

    Ok(Conv2dFixture {
        case,
        shape,
        input,
        weights,
        bias: vec![0; shape.bs_buffer_bytes()],
    })
}

pub fn build_fixture(case: Conv2dCase) -> Result<Conv2dFixture, String> {
    let logical_shape = case.shape();
    let shape = logical_shape
        .parity_padded_shape(case.kernel)
        .map_err(str::to_string)?;
    build_fixture_for_shape(case, shape)
}

/// Builds the logical shape exactly as written, bypassing HAL rejection and
/// programmed-Cout padding. Only raw hardware-characterization probes should
/// use this; supported regressions must use [`build_fixture`].
pub fn build_raw_fixture(case: Conv2dCase) -> Result<Conv2dFixture, String> {
    build_fixture_for_shape(case, case.shape())
}

fn build_fixture_for_shape(case: Conv2dCase, shape: Shape) -> Result<Conv2dFixture, String> {
    if case.precision == OraclePrecision::Int4 {
        return build_int4_fixture(case, shape);
    }
    let logical_input = logical_input(case);
    let mut input = vec![0; input_storage_bytes(shape)];
    for y in 0..case.height as usize {
        for x in 0..case.width as usize {
            for channel in 0..case.cin as usize {
                let logical_index = (y * case.width as usize + x) * case.cin as usize + channel;
                let offset = feature_offset(shape, channel, y, x);
                let encoded = encode_element(case.precision, logical_input[logical_index])?;
                input[offset..offset + encoded.len()].copy_from_slice(&encoded);
            }
        }
    }

    let logical_weights = logical_weights(case);
    let weight_zero_points =
        uses_affine_int8_weights(case).then(|| affine_weight_zero_points(case));
    let element_bytes = element_bytes(shape);
    let mut dense_weight_bytes = Vec::with_capacity(logical_weights.len() * element_bytes);
    match case.precision {
        // int4 never reaches here -- `build_int4_fixture` returns earlier.
        OraclePrecision::Fp16
        | OraclePrecision::Bf16
        | OraclePrecision::Int16
        | OraclePrecision::Int4
        | OraclePrecision::Tf32
        | OraclePrecision::Fp16Accumulator => {
            for value in logical_weights {
                dense_weight_bytes.extend_from_slice(&encode_element(case.precision, value)?);
            }
        }
        OraclePrecision::Int8 | OraclePrecision::Int8Accumulator => {
            if let Some(zero_points) = &weight_zero_points {
                for (index, value) in logical_weights.into_iter().enumerate() {
                    let output_channel = index % case.cout as usize;
                    let raw = value + i32::from(zero_points[output_channel]);
                    let raw = i8::try_from(raw).map_err(|_| {
                        format!(
                            "logical weight {value} plus output {output_channel} zero point {} is outside int8",
                            zero_points[output_channel],
                        )
                    })?;
                    dense_weight_bytes.push(raw as u8);
                }
            } else {
                for value in logical_weights {
                    dense_weight_bytes.extend_from_slice(&encode_element(case.precision, value)?);
                }
            }
        }
    }
    let packed_len = rocket_weight_storage_size(
        case.kernel[0],
        case.kernel[1],
        case.cin as usize,
        shape.out_channels as usize,
        element_bytes,
    )
    .map_err(str::to_string)?;
    let mut weights = vec![0; packed_len];
    if let Some(zero_points) = &weight_zero_points {
        pack_hwcf_to_rocket_weights_affine_i8(
            &dense_weight_bytes,
            case.kernel[0],
            case.kernel[1],
            case.cin as usize,
            case.cout as usize,
            zero_points,
            &mut weights,
        )
        .map_err(str::to_string)?;
    } else if shape.out_channels != case.cout {
        pack_hwcf_to_rocket_weights_padded(
            &dense_weight_bytes,
            case.kernel[0],
            case.kernel[1],
            case.cin as usize,
            case.cout as usize,
            shape.out_channels as usize,
            element_bytes,
            &mut weights,
        )
        .map_err(str::to_string)?;
    } else {
        pack_hwcf_to_rocket_weights(
            &dense_weight_bytes,
            case.kernel[0],
            case.kernel[1],
            case.cin as usize,
            case.cout as usize,
            element_bytes,
            &mut weights,
        )
        .map_err(str::to_string)?;
    }
    if case.precision == OraclePrecision::Int8
        && matches!(case.pattern, OraclePattern::OneHotNeutral80 { .. })
    {
        for byte in &mut weights {
            *byte = match *byte {
                // The ordinary packer's logical zero, including every
                // padded lane, becomes the probed neutral coefficient.
                0 => 0x80,
                // Under the current unit-BS diagnostic configuration, raw
                // zero produced one positive input contribution.
                1 => 0x00,
                other => {
                    return Err(format!(
                        "neutral80 one-hot fixture produced unexpected packed byte 0x{other:02x}"
                    ));
                }
            };
        }
    }
    if case.precision == OraclePrecision::Int8
        && matches!(
            case.pattern,
            OraclePattern::RawByteSweep | OraclePattern::RawByteSweepUnit
        )
    {
        // Pack a parallel all-live mask through the same production layout
        // transform. This distinguishes a legitimate live raw 0x00 byte
        // from zero-filled physical padding without duplicating the layout.
        let dense_live_mask = vec![1u8; dense_weight_bytes.len()];
        let mut packed_live_mask = vec![0; packed_len];
        pack_hwcf_to_rocket_weights(
            &dense_live_mask,
            case.kernel[0],
            case.kernel[1],
            case.cin as usize,
            case.cout as usize,
            element_bytes,
            &mut packed_live_mask,
        )
        .map_err(str::to_string)?;
        for (byte, live) in weights.iter_mut().zip(packed_live_mask) {
            if live == 0 {
                *byte = 0x80;
            }
        }
    }
    if weights.len() != shape.weight_bytes(case.kernel) as usize {
        return Err(format!(
            "packed weight size {} does not match Shape::weight_bytes {}",
            weights.len(),
            shape.weight_bytes(case.kernel),
        ));
    }

    let bias = match case.precision {
        // Every unrequantized rung wants an all-zero BS buffer, which is
        // zero in each of their encodings.
        OraclePrecision::Fp16
        | OraclePrecision::Bf16
        | OraclePrecision::Int16
        | OraclePrecision::Int4
        | OraclePrecision::Tf32
        | OraclePrecision::Fp16Accumulator
        | OraclePrecision::Int8Accumulator => {
            vec![0; shape.bs_buffer_bytes()]
        }
        OraclePrecision::Int8 => {
            let channels = shape.padded_out_channels();
            let mut bytes = vec![0; bs_buffer_bytes(channels)];
            let entries = if let Some(zero_points) = &weight_zero_points {
                (0..channels as usize)
                    .map(|channel| {
                        if channel < zero_points.len() {
                            BsEntry {
                                bias: 0,
                                constant: -i16::from(zero_points[channel]),
                                multiplier: 1 << 14,
                            }
                        } else {
                            BsEntry {
                                bias: 0,
                                constant: 0,
                                multiplier: 0,
                            }
                        }
                    })
                    .collect::<Vec<_>>()
            } else {
                vec![BsEntry::default(); channels as usize]
            };
            write_bs_buffer(&mut bytes, &entries);
            bytes
        }
    };

    Ok(Conv2dFixture {
        case,
        shape,
        input,
        weights,
        bias,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logical_fixture(case: Conv2dCase) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
        (
            logical_input(case),
            logical_weights(case),
            vec![0; case.cout as usize],
        )
    }

    #[test]
    fn dense_reference_counts_padding_and_channels() {
        let case = Conv2dCase {
            width: 3,
            height: 2,
            cin: 2,
            cout: 2,
            kernel: [3, 3],
            stride: 1,
            padding: [1, 1],
            precision: OraclePrecision::Fp16,
            pattern: OraclePattern::Counting,
        };
        let (input, weights, bias) = logical_fixture(case);
        let output = dense_reference(case, &input, &weights, &bias);
        assert_eq!(output, vec![8, 8, 12, 12, 8, 8, 8, 8, 12, 12, 8, 8]);
    }

    #[test]
    fn selector_shortcut_matches_dense_reference() {
        for (precision, pattern) in [
            (OraclePrecision::Fp16, OraclePattern::Selectors { phase: 2 }),
            (OraclePrecision::Int8, OraclePattern::Selectors { phase: 2 }),
            (
                OraclePrecision::Int8Accumulator,
                OraclePattern::Selectors { phase: 2 },
            ),
            (
                OraclePrecision::Int8,
                OraclePattern::SelectorsAffine { phase: 2 },
            ),
        ] {
            let case = Conv2dCase {
                width: 5,
                height: 4,
                cin: 5,
                cout: 7,
                kernel: [3, 3],
                stride: 1,
                padding: [1, 1],
                precision,
                pattern,
            };
            let shape = case.shape();
            let (input, weights, bias) = logical_fixture(case);
            let dense = dense_reference(case, &input, &weights, &bias);
            for y in 0..shape.output_height(case.kernel) as usize {
                for x in 0..shape.output_width(case.kernel) as usize {
                    for channel in 0..case.cout as usize {
                        let index = (y * shape.output_width(case.kernel) as usize + x)
                            * case.cout as usize
                            + channel;
                        assert_eq!(
                            expected_accumulator(case, channel, y, x),
                            dense[index],
                            "{precision:?} {pattern:?} [{y}, {x}, {channel}]",
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn accumulator_output_preserves_i32_values_and_native_block_offsets() {
        let case = Conv2dCase {
            width: 2,
            height: 1,
            cin: 64,
            cout: 33,
            kernel: [1, 1],
            stride: 1,
            padding: [0, 0],
            precision: OraclePrecision::Int8Accumulator,
            pattern: OraclePattern::Counting,
        };
        let shape = case.shape();

        // 2x1 output, Cout 33 -> padded 64, so 16 surfaces of C2=4 int32
        // lanes over 2 pixels: surface stride 2*16 = 32 bytes.
        assert_eq!(expected_output(case, 0, 0, 0), 64);
        assert_eq!(output_offset(shape, case.kernel, 0, 0, 0), 0);
        assert_eq!(output_offset(shape, case.kernel, 3, 0, 0), 3 * 4);
        assert_eq!(output_offset(shape, case.kernel, 0, 0, 1), 16);
        assert_eq!(output_offset(shape, case.kernel, 4, 0, 0), 32);
        assert_eq!(output_offset(shape, case.kernel, 31, 0, 0), 7 * 32 + 3 * 4);
        assert_eq!(output_offset(shape, case.kernel, 32, 0, 0), 8 * 32);
        assert_eq!(output_offset(shape, case.kernel, 32, 0, 1), 8 * 32 + 16);
        // Unchanged: the same bytes, differently arranged.
        assert_eq!(output_storage_bytes(shape, case.kernel), 512);

        let fixture = build_fixture(case).expect("build accumulator fixture");
        assert_eq!(fixture.input.len(), input_storage_bytes(shape));
        assert_eq!(
            fixture.weights.len(),
            shape.weight_bytes(case.kernel) as usize
        );
        assert_eq!(fixture.bias.len(), shape.bs_buffer_bytes());
        assert!(matches!(shape.precision, Precision::Int8Accumulator(_)));
    }

    /// The programmed shape is now the logical one: no parity padding.
    ///
    /// 9x7 Cout 32 used to be widened to Cout 64, because the serial output
    /// writer needed `tile_pixels * blocks_per_pixel` even (63 pixels is odd,
    /// and Cout 32 was one 128-byte block). The corrected writer's C2=4 cube
    /// makes the block count even by construction, so nothing is padded and
    /// the surplus-coefficient machinery is unused here.
    #[test]
    fn accumulator_fixture_uses_the_hal_programmed_shape() {
        let case = Conv2dCase {
            width: 9,
            height: 7,
            cin: 8,
            cout: 32,
            kernel: [1, 1],
            stride: 1,
            padding: [0, 0],
            precision: OraclePrecision::Int8Accumulator,
            pattern: OraclePattern::Dense { phase: 0 },
        };
        let fixture = build_fixture(case).expect("build HAL accumulator fixture");
        assert_eq!(case.shape().out_channels, 32);
        assert_eq!(
            fixture.shape.out_channels, 32,
            "the programmed shape is the logical one; parity padding is gone"
        );
        assert_eq!(
            fixture.weights.len(),
            fixture.shape.weight_bytes(case.kernel) as usize
        );
        assert_eq!(fixture.bias.len(), fixture.shape.bs_buffer_bytes());

        // Formerly rejected outright (3x3 output extent, 3x3 kernel).
        let formerly_rejected = Conv2dCase {
            width: 3,
            height: 3,
            kernel: [3, 3],
            padding: [1, 1],
            ..case
        };
        assert!(build_fixture(formerly_rejected).is_ok());
        assert!(build_raw_fixture(formerly_rejected).is_ok());
    }

    #[test]
    fn affine_selector_fixture_centers_live_weights_padding_and_bs() {
        let case = Conv2dCase {
            width: 5,
            height: 4,
            cin: 3,
            cout: 5,
            kernel: [1, 1],
            stride: 1,
            padding: [0, 0],
            precision: OraclePrecision::Int8,
            pattern: OraclePattern::SelectorsAffine { phase: 2 },
        };
        let logical = logical_weights(case);
        let zero_points = affine_weight_zero_points(case);
        let fixture = build_fixture(case).expect("build affine selector fixture");

        for output_channel in 0..case.cout as usize {
            let packed = &fixture.weights[output_channel * 16..(output_channel + 1) * 16];
            for input_channel in 0..case.cin as usize {
                let logical_index = input_channel * case.cout as usize + output_channel;
                assert_eq!(
                    i32::from(packed[input_channel] as i8),
                    logical[logical_index] + i32::from(zero_points[output_channel]),
                    "live weight oc={output_channel} ic={input_channel}",
                );
            }
            assert!(
                packed[case.cin as usize..]
                    .iter()
                    .all(|&byte| byte == zero_points[output_channel] as u8),
                "neutral padding oc={output_channel}",
            );

            let lane = output_channel % 8;
            let constant =
                i16::from_le_bytes([fixture.bias[32 + lane * 2], fixture.bias[33 + lane * 2]]);
            let multiplier =
                i16::from_le_bytes([fixture.bias[48 + lane * 2], fixture.bias[49 + lane * 2]]);
            assert_eq!(constant, -i16::from(zero_points[output_channel]));
            assert_eq!(multiplier, 1 << 14);
        }
        assert!(
            fixture.weights[5 * 16..6 * 16]
                .iter()
                .all(|&byte| byte == 0)
        );
        match fixture.shape.precision {
            Precision::Int8(quantization) => {
                assert_eq!(quantization.multiplier, Multiplier::from_ratio(1.0),)
            }
            other => panic!("affine selector unexpectedly built {other:?}"),
        }
    }

    #[test]
    fn neutral80_one_hot_shortcut_matches_dense_reference() {
        for kernel in [1usize, 3] {
            for signed_input in [false, true] {
                let case = Conv2dCase {
                    width: 9,
                    height: 7,
                    cin: 3,
                    cout: 32,
                    kernel: [kernel, kernel],
                    stride: 1,
                    padding: [0, 0],
                    precision: OraclePrecision::Int8,
                    pattern: OraclePattern::OneHotNeutral80 {
                        phase: 2,
                        signed_input,
                    },
                };
                let shape = case.shape();
                let (input, weights, bias) = logical_fixture(case);
                let dense = dense_reference(case, &input, &weights, &bias);
                for y in 0..shape.output_height(case.kernel) as usize {
                    for x in 0..shape.output_width(case.kernel) as usize {
                        for channel in 0..case.cout as usize {
                            let index = (y * shape.output_width(case.kernel) as usize + x)
                                * case.cout as usize
                                + channel;
                            assert_eq!(
                                expected_accumulator(case, channel, y, x),
                                dense[index],
                                "K{kernel} signed={signed_input} [{y}, {x}, {channel}]",
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn neutral80_one_hot_fixture_encodes_live_and_padding_bytes() {
        let case = Conv2dCase {
            width: 9,
            height: 7,
            cin: 3,
            cout: 32,
            kernel: [3, 3],
            stride: 1,
            padding: [0, 0],
            precision: OraclePrecision::Int8,
            pattern: OraclePattern::OneHotNeutral80 {
                phase: 2,
                signed_input: false,
            },
        };
        let fixture = build_fixture(case).expect("build neutral80 fixture");
        assert_eq!(
            fixture.weights.iter().filter(|&&byte| byte == 0).count(),
            case.cout as usize,
        );
        assert!(fixture.weights.iter().all(|&byte| matches!(byte, 0 | 0x80)));
    }

    #[test]
    fn raw_byte_sweep_oracle_and_fixture_cover_all_256_values() {
        for pattern in [OraclePattern::RawByteSweep, OraclePattern::RawByteSweepUnit] {
            let case = Conv2dCase {
                width: 1,
                height: 1,
                cin: 1,
                cout: 256,
                kernel: [1, 1],
                stride: 1,
                padding: [0, 0],
                precision: OraclePrecision::Int8,
                pattern,
            };
            let (input, weights, bias) = logical_fixture(case);
            let dense = dense_reference(case, &input, &weights, &bias);
            let expected = (0u16..=255)
                .map(|raw| raw as u8 as i8 as i32)
                .collect::<Vec<_>>();
            assert_eq!(dense, expected);

            let fixture = build_fixture(case).expect("build raw-byte sweep fixture");
            assert_eq!(
                fixture.weights.len(),
                case.shape().weight_bytes(case.kernel) as usize
            );
            assert!(fixture.weights.iter().filter(|&&byte| byte == 0x80).count() > 256);
        }
    }

    #[test]
    fn fixture_storage_covers_dense_and_surface_layouts() {
        for precision in [
            OraclePrecision::Fp16,
            OraclePrecision::Int8,
            OraclePrecision::Int8Accumulator,
        ] {
            for cin in [3u32, 5, 17] {
                let case = Conv2dCase {
                    width: 7,
                    height: 5,
                    cin,
                    cout: 19,
                    kernel: [3, 3],
                    stride: 1,
                    padding: [1, 1],
                    precision,
                    pattern: OraclePattern::Selectors { phase: 0 },
                };
                let fixture = build_fixture(case).expect("build fixture");
                assert_eq!(fixture.input.len(), input_storage_bytes(fixture.shape));
                assert_eq!(
                    fixture.weights.len(),
                    fixture.shape.weight_bytes(case.kernel) as usize,
                );
                assert!(
                    output_offset(
                        fixture.shape,
                        case.kernel,
                        case.cout as usize - 1,
                        fixture.shape.output_height(case.kernel) as usize - 1,
                        fixture.shape.output_width(case.kernel) as usize - 1,
                    ) < output_storage_bytes(fixture.shape, case.kernel)
                );
            }
        }
    }
}
