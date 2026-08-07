use iree_rocket_hal::rocket::{
    conv::{
        BsEntry, Kernels, Multiplier, Precision, Quantization, Shape, bs_buffer_bytes,
        write_bs_buffer,
    },
    tensor_layout::{pack_hwcf_to_rocket_weights, rocket_weight_storage_size},
};

pub const FEATURE_ATOM_BYTES: usize = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OraclePrecision {
    Fp16,
    Int8,
}

impl OraclePrecision {
    pub fn name(self) -> &'static str {
        match self {
            Self::Fp16 => "fp16",
            Self::Int8 => "int8",
        }
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
}

impl OraclePattern {
    pub fn name(self) -> &'static str {
        match self {
            Self::Counting => "counting",
            Self::Selectors { .. } => "selectors",
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
        if self.precision == OraclePrecision::Fp16
            || matches!(self.pattern, OraclePattern::Selectors { .. })
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
            OraclePrecision::Int8 => Precision::Int8(Quantization {
                input_zero_point: 0,
                output_zero_point: 0,
                multiplier: Multiplier::for_unit_bs(1.0 / f64::from(1u32 << self.output_shift())),
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
    let surfaces = (shape.padded_out_channels() as usize).div_ceil(channels_per_atom(shape));
    surfaces
        * shape.output_width(kernels) as usize
        * shape.output_height(kernels) as usize
        * FEATURE_ATOM_BYTES
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
    let atom_channels = channels_per_atom(shape);
    let surface_bytes = shape.output_width(kernels) as usize
        * shape.output_height(kernels) as usize
        * FEATURE_ATOM_BYTES;
    (channel / atom_channels) * surface_bytes
        + (y * shape.output_width(kernels) as usize + x) * FEATURE_ATOM_BYTES
        + (channel % atom_channels) * element_bytes(shape)
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

fn input_value(case: Conv2dCase, y: usize, x: usize, channel: usize) -> i8 {
    match case.pattern {
        OraclePattern::Counting => 1,
        OraclePattern::Selectors { phase } => {
            ((y * 13 + x * 7 + channel * 3 + (y * x) % 5 + phase) % 7) as i8 - 3
        }
    }
}

fn selector_weights(case: Conv2dCase, output_channel: usize) -> [(usize, i8); 3] {
    let slots = case.kernel[0] * case.kernel[1] * case.cin as usize;
    assert!(
        slots >= 3,
        "selector oracle needs at least three HWCF slots"
    );
    let phase = match case.pattern {
        OraclePattern::Selectors { phase } => phase,
        OraclePattern::Counting => 0,
    };
    let first = (output_channel * 17 + phase * 11) % slots;
    let step = (slots / 3).max(1);
    [
        (first, 1),
        ((first + step) % slots, -1),
        ((first + 2 * step) % slots, 2),
    ]
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
) -> i8 {
    match case.pattern {
        OraclePattern::Counting => 1,
        OraclePattern::Selectors { .. } => {
            let selector = (ky * case.kernel[1] + kx) * case.cin as usize + input_channel;
            selector_weights(case, output_channel)
                .into_iter()
                .find_map(|(selected, weight)| (selected == selector).then_some(weight))
                .unwrap_or(0)
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
        OraclePattern::Selectors { .. } => selector_weights(case, output_channel)
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
            .sum(),
    }
}

pub fn expected_output(
    case: Conv2dCase,
    output_channel: usize,
    output_y: usize,
    output_x: usize,
) -> i32 {
    let accumulator = expected_accumulator(case, output_channel, output_y, output_x);
    match case.precision {
        OraclePrecision::Fp16 => accumulator,
        OraclePrecision::Int8 => rounded_shift(accumulator, case.output_shift()).clamp(-128, 127),
    }
}

pub fn dense_reference(case: Conv2dCase, input: &[i8], weights: &[i8], bias: &[i32]) -> Vec<i32> {
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
                            accumulator +=
                                i32::from(input[input_index]) * i32::from(weights[weight_index]);
                        }
                    }
                }
                output[(oy * out_width + ox) * case.cout as usize + oc] = accumulator;
            }
        }
    }
    output
}

fn logical_input(case: Conv2dCase) -> Vec<i8> {
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

fn logical_weights(case: Conv2dCase) -> Vec<i8> {
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

pub fn build_fixture(case: Conv2dCase) -> Result<Conv2dFixture, String> {
    let shape = case.shape();
    let logical_input = logical_input(case);
    let mut input = vec![0; input_storage_bytes(shape)];
    for y in 0..case.height as usize {
        for x in 0..case.width as usize {
            for channel in 0..case.cin as usize {
                let logical_index = (y * case.width as usize + x) * case.cin as usize + channel;
                let offset = feature_offset(shape, channel, y, x);
                match case.precision {
                    OraclePrecision::Fp16 => {
                        input[offset..offset + 2].copy_from_slice(
                            &f32_to_f16(f32::from(logical_input[logical_index])).to_le_bytes(),
                        );
                    }
                    OraclePrecision::Int8 => {
                        input[offset] = logical_input[logical_index] as u8;
                    }
                }
            }
        }
    }

    let logical_weights = logical_weights(case);
    let element_bytes = element_bytes(shape);
    let mut dense_weight_bytes = Vec::with_capacity(logical_weights.len() * element_bytes);
    match case.precision {
        OraclePrecision::Fp16 => {
            for value in logical_weights {
                dense_weight_bytes.extend_from_slice(&f32_to_f16(f32::from(value)).to_le_bytes());
            }
        }
        OraclePrecision::Int8 => {
            dense_weight_bytes.extend(logical_weights.into_iter().map(|value| value as u8));
        }
    }
    let packed_len = rocket_weight_storage_size(
        case.kernel[0],
        case.kernel[1],
        case.cin as usize,
        case.cout as usize,
        element_bytes,
    )
    .map_err(str::to_string)?;
    let mut weights = vec![0; packed_len];
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
    if weights.len() != shape.weight_bytes(case.kernel) as usize {
        return Err(format!(
            "packed weight size {} does not match Shape::weight_bytes {}",
            weights.len(),
            shape.weight_bytes(case.kernel),
        ));
    }

    let bias = match case.precision {
        OraclePrecision::Fp16 => vec![0; shape.bs_buffer_bytes()],
        OraclePrecision::Int8 => {
            let channels = shape.padded_out_channels();
            let mut bytes = vec![0; bs_buffer_bytes(channels)];
            write_bs_buffer(&mut bytes, &vec![BsEntry::default(); channels as usize]);
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

    fn logical_fixture(case: Conv2dCase) -> (Vec<i8>, Vec<i8>, Vec<i32>) {
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
        for precision in [OraclePrecision::Fp16, OraclePrecision::Int8] {
            let case = Conv2dCase {
                width: 5,
                height: 4,
                cin: 5,
                cout: 7,
                kernel: [3, 3],
                stride: 1,
                padding: [1, 1],
                precision,
                pattern: OraclePattern::Selectors { phase: 2 },
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
                            "{precision:?} [{y}, {x}, {channel}]",
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn fixture_storage_covers_dense_and_surface_layouts() {
        for precision in [OraclePrecision::Fp16, OraclePrecision::Int8] {
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
