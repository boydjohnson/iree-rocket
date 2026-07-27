//! Hardware validation for the remaining Phase 1 convolution combinations.
//!
//! These cases deliberately live together because they share the same
//! capture-derived `conv.rs` execution fixture, but each test closes a
//! separate gap called out in `DESIGN_NOTES.md`:
//!
//! - clamped int8 activation on silicon;
//! - the int8 depthwise coefficient layout;
//! - depthwise convolution combined with fused activation; and
//! - depthwise convolution submitted through `ConvPlan`'s multi-tile path.
//!
//! Cross-compile this test, copy the resulting binary to the RK3588 board,
//! and run the ignored tests there:
//!
//! ```text
//! CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!   cargo test --target aarch64-unknown-linux-gnu --release \
//!     --test conv_phase1_validation_hw --no-run
//!
//! ./conv_phase1_validation_hw-<hash> --ignored --nocapture
//! ```
//!
//! The inputs are chosen to keep the expected results exact. The int8 cases
//! allow the same one-LSB tolerance as `conv_int8_hw`: the emitted program is
//! capture-identical, but the measured output-conversion gain can differ by
//! one LSB above small accumulator values.

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    conv::{
        Activation, BsEntry, Buffers, ConvPlan, FeatureLayout, Kernels, Multiplier, Precision,
        Quantization, Shape, bs_buffer_bytes, write_bs_buffer,
    },
    device::{Buffer, JobDesc, close_bo, fini_bo, prep_bo, submit_jobs},
    tensor_layout::pack_depthwise_to_rocket_weights,
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const FEATURE_ATOM_BYTES: usize = 16;
const PAGE_BYTES: usize = 4096;
const KERNEL: Kernels = [3, 3];
const IMPULSE: usize = 16;

fn page_aligned_size(size: usize) -> usize {
    size.max(1).div_ceil(PAGE_BYTES) * PAGE_BYTES
}

fn f32_to_f16(value: f32) -> u16 {
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

fn f16_to_f32(bits: u16) -> f32 {
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

fn element_bytes(shape: Shape) -> usize {
    shape.precision.element_bytes() as usize
}

fn channels_per_atom(shape: Shape) -> usize {
    shape.precision.channels_per_atom() as usize
}

fn input_bytes(shape: Shape) -> usize {
    let pixels = shape.width as usize * shape.height as usize;
    match shape.layout() {
        FeatureLayout::Dense => pixels * shape.in_channels as usize * element_bytes(shape),
        FeatureLayout::Surfaces => shape.feature_atoms() as usize * pixels * FEATURE_ATOM_BYTES,
    }
}

fn output_bytes(shape: Shape, kernels: Kernels) -> usize {
    let surfaces = (shape.padded_out_channels() as usize).div_ceil(channels_per_atom(shape));
    surfaces
        * shape.output_width(kernels) as usize
        * shape.output_height(kernels) as usize
        * FEATURE_ATOM_BYTES
}

fn feature_offset(shape: Shape, channel: usize, y: usize, x: usize) -> usize {
    let width = shape.width as usize;
    match shape.layout() {
        FeatureLayout::Dense => {
            ((y * width + x) * shape.in_channels as usize + channel) * element_bytes(shape)
        }
        FeatureLayout::Surfaces => {
            let atom_channels = channels_per_atom(shape);
            (channel / atom_channels) * width * shape.height as usize * FEATURE_ATOM_BYTES
                + (y * width + x) * FEATURE_ATOM_BYTES
                + (channel % atom_channels) * element_bytes(shape)
        }
    }
}

fn output_offset(shape: Shape, kernels: Kernels, channel: usize, y: usize, x: usize) -> usize {
    let atom_channels = channels_per_atom(shape);
    let surface_bytes = shape.output_width(kernels) as usize
        * shape.output_height(kernels) as usize
        * FEATURE_ATOM_BYTES;
    (channel / atom_channels) * surface_bytes
        + (y * shape.output_width(kernels) as usize + x) * FEATURE_ATOM_BYTES
        + (channel % atom_channels) * element_bytes(shape)
}

fn int8_quantization() -> Precision {
    Precision::Int8(Quantization {
        input_zero_point: 0,
        output_zero_point: 0,
        multiplier: Multiplier::for_unit_bs(1.0),
    })
}

fn zero_bias(shape: Shape) -> Vec<u8> {
    match shape.precision {
        Precision::Fp16 => vec![0; shape.bs_buffer_bytes()],
        Precision::Int8(_) => {
            let channels = shape.padded_out_channels();
            let mut bytes = vec![0; bs_buffer_bytes(channels)];
            write_bs_buffer(&mut bytes, &vec![BsEntry::default(); channels as usize]);
            bytes
        }
    }
}

/// Executes every program selected by `ConvPlan` as a separate job in one
/// submission. The shared output BO is listed by every job; waiting on it
/// therefore waits for the entire tile set, as in `conv_tiled_hw`.
fn execute(
    shape: Shape,
    kernels: Kernels,
    input: &[u8],
    weights: &[u8],
    bias: &[u8],
) -> (usize, Vec<u8>) {
    assert_eq!(input.len(), input_bytes(shape), "input fixture size");
    assert_eq!(
        weights.len(),
        shape.weight_bytes(kernels) as usize,
        "weight fixture size"
    );
    assert!(
        bias.len() >= shape.bs_buffer_bytes(),
        "bias/BS fixture is shorter than the shape declares"
    );

    let plan = ConvPlan::new(shape, kernels);
    let tile_count = plan.tiles().len();
    let output_len = output_bytes(shape, kernels);

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let buf_input = Buffer::new(fd, page_aligned_size(input.len()), &file);
        ptr::write_bytes(buf_input.host_ptr, 0, buf_input.size);
        ptr::copy_nonoverlapping(input.as_ptr(), buf_input.host_ptr, input.len());

        let buf_weights = Buffer::new(fd, page_aligned_size(weights.len()), &file);
        ptr::write_bytes(buf_weights.host_ptr, 0, buf_weights.size);
        ptr::copy_nonoverlapping(weights.as_ptr(), buf_weights.host_ptr, weights.len());

        let buf_bias = Buffer::new(fd, page_aligned_size(bias.len()), &file);
        ptr::write_bytes(buf_bias.host_ptr, 0, buf_bias.size);
        ptr::copy_nonoverlapping(bias.as_ptr(), buf_bias.host_ptr, bias.len());

        let buf_output = Buffer::new(fd, page_aligned_size(output_len), &file);
        ptr::write_bytes(buf_output.host_ptr, 0, buf_output.size);

        let programs = plan.programs_with_buffers(Buffers {
            input: buf_input.dma_address,
            weights: buf_weights.dma_address,
            bias: buf_bias.dma_address,
            output: buf_output.dma_address,
        });
        let mut command_buffers = Vec::with_capacity(programs.len());
        for program in &programs {
            let command_bytes = program.len() * mem::size_of::<u64>();
            let buffer = Buffer::new(fd, page_aligned_size(command_bytes), &file);
            ptr::write_bytes(buffer.host_ptr, 0, buffer.size);
            let words = std::slice::from_raw_parts_mut(buffer.host_ptr as *mut u64, program.len());
            for (destination, command) in words.iter_mut().zip(program) {
                *destination = command.0;
            }
            command_buffers.push((buffer, program.len() as u32));
        }

        for handle in [
            buf_input.handle,
            buf_weights.handle,
            buf_bias.handle,
            buf_output.handle,
        ] {
            fini_bo(fd, handle).expect("failed to sync data BO for the NPU");
        }
        for (buffer, _) in &command_buffers {
            fini_bo(fd, buffer.handle).expect("failed to sync regcmd BO for the NPU");
        }

        let tasks: Vec<[(u32, u32); 1]> = command_buffers
            .iter()
            .map(|(buffer, count)| [(buffer.dma_address, *count)])
            .collect();
        let in_handles: Vec<[u32; 4]> = command_buffers
            .iter()
            .map(|(buffer, _)| {
                [
                    buffer.handle,
                    buf_input.handle,
                    buf_weights.handle,
                    buf_bias.handle,
                ]
            })
            .collect();
        let out_handles = [buf_output.handle];
        let jobs: Vec<JobDesc<'_>> = tasks
            .iter()
            .zip(&in_handles)
            .map(|(tasks, in_handles)| JobDesc {
                tasks,
                in_handles,
                out_handles: &out_handles,
            })
            .collect();

        submit_jobs(fd, &jobs).unwrap_or_else(|error| panic!("{shape:?} SUBMIT failed: {error}"));
        prep_bo(fd, buf_output.handle, 5_000_000_000)
            .unwrap_or_else(|error| panic!("{shape:?} did not complete: {error}"));

        let output = std::slice::from_raw_parts(buf_output.host_ptr, output_len).to_vec();

        for handle in [
            buf_input.handle,
            buf_weights.handle,
            buf_bias.handle,
            buf_output.handle,
        ] {
            close_bo(fd, handle).expect("failed to close data BO");
        }
        for (buffer, _) in &command_buffers {
            close_bo(fd, buffer.handle).expect("failed to close regcmd BO");
        }

        (tile_count, output)
    }
}

fn fill_int8_input(shape: Shape, value: i8) -> Vec<u8> {
    let mut input = vec![0; input_bytes(shape)];
    for channel in 0..shape.in_channels as usize {
        for y in 0..shape.height as usize {
            for x in 0..shape.width as usize {
                input[feature_offset(shape, channel, y, x)] = value as u8;
            }
        }
    }
    input
}

fn fill_fp16_input(shape: Shape, impulse: Option<(usize, usize)>) -> Vec<u8> {
    let mut input = vec![0; input_bytes(shape)];
    let one = f32_to_f16(1.0).to_le_bytes();
    for channel in 0..shape.in_channels as usize {
        for y in 0..shape.height as usize {
            for x in 0..shape.width as usize {
                if impulse.is_some_and(|position| position != (y, x)) {
                    continue;
                }
                let offset = feature_offset(shape, channel, y, x);
                input[offset..offset + 2].copy_from_slice(&one);
            }
        }
    }
    input
}

/// A positive coefficient unique across all real channels and taps. Cin 12
/// tops out at 108, so the same coordinate code is exactly representable in
/// both signed int8 and fp16.
fn coefficient(channel: usize, ky: usize, kx: usize) -> i32 {
    (channel * KERNEL[0] * KERNEL[1] + ky * KERNEL[1] + kx + 1) as i32
}

fn depthwise_weights(shape: Shape) -> Vec<u8> {
    let channels = shape.in_channels as usize;
    let bytes_per_element = element_bytes(shape);
    let mut dense = vec![0; channels * KERNEL[0] * KERNEL[1] * bytes_per_element];
    for channel in 0..channels {
        for ky in 0..KERNEL[0] {
            for kx in 0..KERNEL[1] {
                let offset = ((channel * KERNEL[0] + ky) * KERNEL[1] + kx) * bytes_per_element;
                match shape.precision {
                    Precision::Fp16 => {
                        dense[offset..offset + 2].copy_from_slice(
                            &f32_to_f16(coefficient(channel, ky, kx) as f32).to_le_bytes(),
                        );
                    }
                    Precision::Int8(_) => {
                        dense[offset] = i8::try_from(coefficient(channel, ky, kx))
                            .expect("test coefficient must fit int8")
                            as u8;
                    }
                }
            }
        }
    }

    let weight_bytes = shape.weight_bytes(KERNEL) as usize;
    let padded_channels = weight_bytes / (KERNEL[0] * KERNEL[1] * bytes_per_element);
    let mut packed = vec![0; weight_bytes];
    pack_depthwise_to_rocket_weights(
        &dense,
        KERNEL[0],
        KERNEL[1],
        channels,
        padded_channels,
        bytes_per_element,
        &mut packed,
    )
    .expect("depthwise packing failed");
    packed
}

fn assert_int8_output(
    label: &str,
    shape: Shape,
    kernels: Kernels,
    output: &[u8],
    expected: impl Fn(usize, usize, usize) -> i32,
) {
    let mut mismatches = 0;
    let mut samples = Vec::new();
    for channel in 0..shape.out_channels as usize {
        for y in 0..shape.output_height(kernels) as usize {
            for x in 0..shape.output_width(kernels) as usize {
                let offset = output_offset(shape, kernels, channel, y, x);
                let got = i32::from(output[offset] as i8);
                let want = expected(channel, y, x);
                if (got - want).abs() > 1 {
                    mismatches += 1;
                    if samples.len() < 12 {
                        samples.push(format!("[c{channel}, {y}, {x}] want {want} got {got}"));
                    }
                }
            }
        }
    }
    assert_eq!(
        mismatches,
        0,
        "{label}: {mismatches} values differ by more than one LSB:\n  {}",
        samples.join("\n  ")
    );
}

fn assert_fp16_output(
    label: &str,
    shape: Shape,
    kernels: Kernels,
    output: &[u8],
    expected: impl Fn(usize, usize, usize) -> f32,
) {
    let mut mismatches = 0;
    let mut samples = Vec::new();
    for channel in 0..shape.out_channels as usize {
        for y in 0..shape.output_height(kernels) as usize {
            for x in 0..shape.output_width(kernels) as usize {
                let offset = output_offset(shape, kernels, channel, y, x);
                let got = f16_to_f32(u16::from_le_bytes([output[offset], output[offset + 1]]));
                let want = expected(channel, y, x);
                if got != want {
                    mismatches += 1;
                    if samples.len() < 12 {
                        samples.push(format!("[c{channel}, {y}, {x}] want {want} got {got}"));
                    }
                }
            }
        }
    }
    assert_eq!(
        mismatches,
        0,
        "{label}: {mismatches} values are wrong:\n  {}",
        samples.join("\n  ")
    );
}

fn valid_taps(coordinate: usize, extent: usize) -> usize {
    3 - usize::from(coordinate == 0) - usize::from(coordinate + 1 == extent)
}

/// Keeps the ignored hardware cases honest on an ordinary development host.
/// If padding or planning policy changes, this fails before anybody copies a
/// binary to the board that no longer exercises the intended boundary.
#[test]
fn validation_shapes_still_exercise_the_phase1_gaps() {
    assert_eq!(
        Activation::clamped_int8(0.75, 0.5, 0.25),
        Activation::Clamped { cmp: 6 },
        "the int8 fixture must encode a nontrivial accumulator-unit ceiling"
    );

    let int8_depthwise =
        Shape::with_precision(32, 32, 1, 12, 12, int8_quantization()).with_depthwise();
    assert_eq!(
        int8_depthwise.weight_bytes(KERNEL) as usize / (KERNEL[0] * KERNEL[1]),
        16,
        "the int8 layout case must separate real Cin from its packed stride"
    );

    let tiled = Shape::with_out_channels(256, 64, 1, 12, 12).with_depthwise();
    assert!(
        ConvPlan::new(tiled, KERNEL).tiles().len() > 1,
        "the ConvPlan case must require more than one hardware job"
    );
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn clamped_int8_activation_runs_on_npu() {
    // The real-valued ceiling is 0.75, but BN sees accumulator units:
    // round(0.75 / (0.5 * 0.25)) == 6. With unit output conversion the raw
    // results are therefore 4 at corners, 6 at edges, and 6 in the interior.
    // A bypassed clamp returns 9 in the interior, while treating 0.75 as a
    // raw integer clamps almost everything to zero.
    let activation = Activation::clamped_int8(0.75, 0.5, 0.25);
    assert_eq!(activation, Activation::Clamped { cmp: 6 });
    let shape =
        Shape::with_precision(32, 32, 1, 1, 8, int8_quantization()).with_activation(activation);
    let input = fill_int8_input(shape, 1);
    let weights = vec![1; shape.weight_bytes(KERNEL) as usize];
    let (tiles, output) = execute(shape, KERNEL, &input, &weights, &zero_bias(shape));
    assert_eq!(
        tiles, 1,
        "the activation case should isolate one whole tile"
    );

    assert_int8_output(
        "clamped int8 activation",
        shape,
        KERNEL,
        &output,
        |_, y, x| {
            (valid_taps(y, shape.height as usize) * valid_taps(x, shape.width as usize)).min(6)
                as i32
        },
    );
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn int8_depthwise_layout_reproduces_its_filter() {
    // Cin 12 is deliberately not one whole int8 atom. The filter's real
    // channel count is 12 while the packed tap stride is 16, so this fails
    // under both a channel-major layout and a tap-major layout using Cin as
    // its stride.
    let shape = Shape::with_precision(32, 32, 1, 12, 12, int8_quantization()).with_depthwise();
    let mut input = vec![0; input_bytes(shape)];
    for channel in 0..shape.in_channels as usize {
        input[feature_offset(shape, channel, IMPULSE, IMPULSE)] = 1;
    }
    let weights = depthwise_weights(shape);
    assert_eq!(
        weights.len() / (KERNEL[0] * KERNEL[1]),
        16,
        "the test must distinguish real Cin 12 from packed stride 16"
    );
    let (_, output) = execute(shape, KERNEL, &input, &weights, &zero_bias(shape));

    assert_int8_output(
        "int8 depthwise layout",
        shape,
        KERNEL,
        &output,
        |channel, y, x| {
            if y + 1 >= IMPULSE && y <= IMPULSE + 1 && x + 1 >= IMPULSE && x <= IMPULSE + 1 {
                coefficient(channel, IMPULSE + 1 - y, IMPULSE + 1 - x)
            } else {
                0
            }
        },
    );
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn depthwise_with_fused_activation_runs_on_npu() {
    // The impulse exposes every coefficient separately. Values 1..49 pass
    // through and values 51..108 clamp, so this checks both halves of the
    // BN-stage activation while depthwise mode is enabled.
    const CEILING: f32 = 50.0;
    let shape = Shape::with_out_channels(32, 32, 1, 12, 12)
        .with_depthwise()
        .with_activation(Activation::clamped_fp16(CEILING));
    let input = fill_fp16_input(shape, Some((IMPULSE, IMPULSE)));
    let weights = depthwise_weights(shape);
    let (_, output) = execute(shape, KERNEL, &input, &weights, &zero_bias(shape));

    assert_fp16_output(
        "depthwise with fused activation",
        shape,
        KERNEL,
        &output,
        |channel, y, x| {
            if y + 1 >= IMPULSE && y <= IMPULSE + 1 && x + 1 >= IMPULSE && x <= IMPULSE + 1 {
                (coefficient(channel, IMPULSE + 1 - y, IMPULSE + 1 - x) as f32).min(CEILING)
            } else {
                0.0
            }
        },
    );
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn depthwise_conv_plan_tiles_run_on_npu() {
    // At 256x64 with two fp16 feature atoms per pixel, a whole-height tile
    // exceeds both the CBUF capacity and DATA_ENTRIES. Every output value is
    // nonzero, including tile-boundary rows, so a tile that is skipped or
    // reads the wrong halo cannot hide behind the zero-filled output BO.
    let shape = Shape::with_out_channels(256, 64, 1, 12, 12).with_depthwise();
    let input = fill_fp16_input(shape, None);
    let weights = depthwise_weights(shape);
    let (tiles, output) = execute(shape, KERNEL, &input, &weights, &zero_bias(shape));
    assert!(
        tiles > 1,
        "the test shape no longer exercises ConvPlan tiling"
    );

    assert_fp16_output(
        "depthwise ConvPlan tiling",
        shape,
        KERNEL,
        &output,
        |channel, y, x| {
            let mut sum = 0;
            for ky in 0..KERNEL[0] {
                let input_y = y + ky;
                if input_y == 0 || input_y > shape.height as usize {
                    continue;
                }
                for kx in 0..KERNEL[1] {
                    let input_x = x + kx;
                    if input_x == 0 || input_x > shape.width as usize {
                        continue;
                    }
                    sum += coefficient(channel, ky, kx);
                }
            }
            sum as f32
        },
    );
}
