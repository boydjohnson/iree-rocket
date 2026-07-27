//! Hardware validation of square and non-square convolution kernels, in both
//! precisions.
//!
//! This test is ignored on the development host because it needs the RK3588
//! NPU device. Cross-compile it, copy the printed test binary to the board,
//! and run it there:
//!
//!   CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!     cargo test --target aarch64-unknown-linux-gnu --release \
//!       --test conv_kernel_shape_hw --no-run
//!
//!   ./conv_kernel_shape_hw-<hash> --ignored --nocapture
//!
//! # What this is checking
//!
//! Every kernel capture before the rectangular sweep was square, so
//! `weight_width` and `weight_height` were never observed differing and the
//! builder carried a single kernel size. A sweep of 53 fp16 and 60 int8
//! non-square captures separates the two axes: `weight_width` and `pad_left`
//! follow the kernel's width, `weight_height`, `pad_top` and `feature_grains`
//! follow its height, and the coefficient footprint is `kh * kw`. This is the
//! hardware half of that -- it runs kernels the builder has never executed
//! and checks every output element.
//!
//! The even-kernel sweep adds a second ambiguity the odd captures could not
//! resolve: the extent is programmed verbatim rather than reconstructed as
//! `2 * pad + 1`, and padding is an independent per-axis input. The even
//! tests below cover square, rectangular, and mixed-parity kernels with both
//! the default `kernel / 2` padding and explicit asymmetric per-axis values.
//!
//! # Why the shapes come in mirrored pairs
//!
//! An axis confusion is exactly the bug this class of change invites, and a
//! square kernel cannot expose one. Every geometry below is run as a pair,
//! `kh x kw` and `kw x kh`, so a builder that programmed the width where the
//! height belongs would compute the mirrored convolution instead. That is
//! visible in the output because the border tap counts differ: at 3x7 an
//! output on the top edge but interior in x sees `2 * 7` taps, at 7x3 it
//! sees `4 * 3`. The corners alone would not separate them, so the check
//! covers every element rather than sampling.
//!
//! Both extents are also run against non-square *feature maps*, where a
//! swapped axis additionally changes the output extent and the tiling.
//!
//! # Scope
//!
//! Extents 1..=11 per axis and stride 1, which is what the combined odd and
//! even captures cover. Non-square kernels reach the CBUF allocator through
//! [`ConvPlan`], since `conv_2d_tile`'s automatic split has runtime backing
//! for 1x1 and 3x3 only. The last test covers
//! `ConvPlan::with_cbuf_banks`, the explicit-split path for the high-demand
//! shapes where the captured allocation stops following coefficient demand
//! and `ConvPlan::new` declines to guess.
//!
//! Input and weights are 1, with padded lanes zero, so each output is
//! `Cin * valid_y_taps * valid_x_taps` -- independent of the order the
//! hardware walks the coefficients in, and exact in fp16 at these sizes. The
//! int8 runs follow `conv_int8_hw`: the requantisation shift is a negative
//! power of two that divides the accumulator exactly, and the acceptance bar
//! is one LSB for the reasons documented there.

use std::{collections::BTreeSet, fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr, sync::Mutex};

use iree_rocket_hal::rocket::{
    builders::{
        RegCmd, RegisterMeta,
        cna::{CnaDcompAddr0, CnaFeatureDataAddr},
        dpu::DpuDstBaseAddr,
        dpu_rdma::DpuRdmaBsBaseAddr,
    },
    conv::{
        BsEntry, ConvPlan, FeatureLayout, Kernels, Multiplier, Padding, Precision, Quantization,
        Shape, write_bs_buffer,
    },
    device::{Buffer, JobDesc, close_bo, fini_bo, prep_bo, submit_jobs},
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const FEATURE_ATOM_BYTES: usize = 16;
const PAGE_BYTES: usize = 4096;
const FP16_ONE: u16 = 0x3c00;
const COMPLETION_TIMEOUT_NS: u64 = 10_000_000_000;
static NPU_TEST_LOCK: Mutex<()> = Mutex::new(());

fn page_aligned_size(size: usize) -> usize {
    size.div_ceil(PAGE_BYTES) * PAGE_BYTES
}

fn decode_identity(command: &RegCmd) -> (u32, u32) {
    ((command.0 >> 48) as u32, command.0 as u32 & 0xffff)
}

fn relocate<R: RegisterMeta>(commands: &mut [RegCmd], address: u32) {
    let matches: Vec<_> = commands
        .iter()
        .enumerate()
        .filter_map(|(index, command)| {
            (decode_identity(command) == (R::DOMAIN, R::OFFSET)).then_some(index)
        })
        .collect();
    assert_eq!(matches.len(), 1, "expected exactly one relocation site");
    let tile_offset = (commands[matches[0]].0 >> 16) as u32;
    commands[matches[0]] = RegCmd::new(R::DOMAIN, R::OFFSET, address + tile_offset);
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    let f32_bits = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            let subnormal = (frac as f32) * 2f32.powi(-24);
            return if sign == 1 { -subnormal } else { subnormal };
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (frac << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13)
    };
    f32::from_bits(f32_bits)
}

/// Taps of a kernel axis that land inside an image axis at one output
/// coordinate.
///
/// Expressing this as an interval intersection handles even kernels and
/// explicit padding directly. The old centred-radius formula was equivalent
/// only when `kernel == 2 * padding + 1`.
fn valid_taps(
    output_coordinate: usize,
    extent: usize,
    kernel: usize,
    padding: usize,
    stride: usize,
) -> usize {
    let window_first = (output_coordinate * stride) as isize - padding as isize;
    let window_last = window_first + kernel as isize;
    let input_first = window_first.max(0);
    let input_last = window_last.min(extent as isize);
    input_last.saturating_sub(input_first) as usize
}

/// The accumulator an output pixel should hold: `Cin` times the taps inside
/// the image on each axis. The kernel's height is charged against the image
/// height and its width against the width -- swapping either is the mistake
/// these runs exist to catch.
fn expected_accumulator(shape: Shape, kernels: Kernels, y: usize, x: usize) -> usize {
    let stride = shape.stride as usize;
    let [pad_top, pad_left] = shape.padding.unwrap_or([kernels[0] / 2, kernels[1] / 2]);
    shape.in_channels as usize
        * valid_taps(y, shape.height as usize, kernels[0], pad_top, stride)
        * valid_taps(x, shape.width as usize, kernels[1], pad_left, stride)
}

/// Writes the input feature map: every real channel 1, every padding lane 0.
unsafe fn fill_input(base: *mut u8, size: usize, shape: Shape, surfaces: usize) {
    unsafe {
        ptr::write_bytes(base, 0, size);
        let width = shape.width as usize;
        let height = shape.height as usize;
        let element_bytes = shape.precision.element_bytes() as usize;
        let channels_per_atom = shape.precision.channels_per_atom() as usize;
        for channel in 0..shape.in_channels as usize {
            let surface = channel / channels_per_atom;
            let lane = channel % channels_per_atom;
            if surface >= surfaces {
                continue;
            }
            for y in 0..height {
                for x in 0..width {
                    let offset = match shape.layout() {
                        FeatureLayout::Dense => {
                            ((y * width + x) * shape.in_channels as usize + channel) * element_bytes
                        }
                        FeatureLayout::Surfaces => {
                            surface * width * height * FEATURE_ATOM_BYTES
                                + (y * width + x) * FEATURE_ATOM_BYTES
                                + lane * element_bytes
                        }
                    };
                    match shape.precision {
                        Precision::Fp16 => ptr::write(base.add(offset) as *mut u16, FP16_ONE),
                        Precision::Int8(_) => ptr::write(base.add(offset), 1u8),
                    }
                }
            }
        }
    }
}

struct Failure {
    mismatches: usize,
    samples: Vec<String>,
    /// Every distinct `got - want`, so a systematic offset stays visible
    /// rather than being absorbed by the int8 tolerance.
    differences: BTreeSet<i32>,
}

/// Runs `plan` as one job per tile and checks every output element.
///
/// `shift` is the negative power of two the int8 output conversion applies,
/// and is ignored at fp16.
fn run(plan: &ConvPlan, shift: u32) -> Result<BTreeSet<i32>, Failure> {
    let shape = plan.shape();
    let kernels = plan.kernels();
    let width = shape.width as usize;
    let out_width = shape.output_width(kernels) as usize;
    let out_height = shape.output_height(kernels) as usize;
    let element_bytes = shape.precision.element_bytes() as usize;
    let channels_per_atom = shape.precision.channels_per_atom() as usize;
    let in_surfaces = (shape.weight_channels() as usize).div_ceil(channels_per_atom);
    // The DPU writes whole granules, so the destination has to hold the
    // padded channel count even when the caller only wants `out_channels`.
    let out_surfaces = (shape.padded_out_channels() as usize).div_ceil(channels_per_atom);

    let input_bytes = match shape.layout() {
        FeatureLayout::Dense => {
            width * shape.height as usize * shape.in_channels as usize * element_bytes
        }
        FeatureLayout::Surfaces => in_surfaces * width * shape.height as usize * FEATURE_ATOM_BYTES,
    };
    let output_bytes = out_surfaces * out_width * out_height * FEATURE_ATOM_BYTES;

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let buf_input = Buffer::new(fd, page_aligned_size(input_bytes), &file);
        fill_input(buf_input.host_ptr, buf_input.size, shape, in_surfaces);

        // Coefficients cover the padded input channel count and the real
        // output count, all ones. Padding channels multiply zeroed input, so
        // they contribute nothing whatever order they are read in.
        let weight_bytes = shape.weight_bytes(kernels) as usize;
        let buf_weights = Buffer::new(fd, page_aligned_size(weight_bytes), &file);
        ptr::write_bytes(buf_weights.host_ptr, 0, buf_weights.size);
        match shape.precision {
            Precision::Fp16 => {
                std::slice::from_raw_parts_mut(buf_weights.host_ptr as *mut u16, weight_bytes / 2)
                    .fill(FP16_ONE);
            }
            Precision::Int8(_) => ptr::write_bytes(buf_weights.host_ptr, 1, weight_bytes),
        }

        // At fp16 BRDMA fetches only the bias and a zeroed buffer is right.
        // At int8 it also fetches a per-channel multiplier, and a zeroed
        // buffer would multiply the whole tensor by zero.
        let bias_bytes = match shape.precision {
            Precision::Fp16 => PAGE_BYTES,
            Precision::Int8(_) => page_aligned_size(shape.bs_buffer_bytes()),
        };
        let buf_bias = Buffer::new(fd, bias_bytes, &file);
        ptr::write_bytes(buf_bias.host_ptr, 0, buf_bias.size);
        if matches!(shape.precision, Precision::Int8(_)) {
            let entries = vec![BsEntry::default(); shape.padded_out_channels() as usize];
            write_bs_buffer(
                std::slice::from_raw_parts_mut(buf_bias.host_ptr, buf_bias.size),
                &entries,
            );
        }

        let buf_output = Buffer::new(fd, page_aligned_size(output_bytes), &file);
        ptr::write_bytes(buf_output.host_ptr, 0, buf_output.size);

        let programs = plan.programs();
        let mut command_buffers = Vec::with_capacity(programs.len());
        for mut commands in programs {
            relocate::<CnaFeatureDataAddr>(&mut commands, buf_input.dma_address);
            relocate::<CnaDcompAddr0>(&mut commands, buf_weights.dma_address);
            relocate::<DpuRdmaBsBaseAddr>(&mut commands, buf_bias.dma_address);
            relocate::<DpuDstBaseAddr>(&mut commands, buf_output.dma_address);

            let command_bytes = commands.len() * mem::size_of::<u64>();
            let buffer = Buffer::new(fd, page_aligned_size(command_bytes), &file);
            ptr::write_bytes(buffer.host_ptr, 0, buffer.size);
            let words = std::slice::from_raw_parts_mut(buffer.host_ptr as *mut u64, commands.len());
            for (destination, command) in words.iter_mut().zip(&commands) {
                *destination = command.0;
            }
            command_buffers.push((buffer, commands.len() as u32));
        }

        for handle in [
            buf_input.handle,
            buf_weights.handle,
            buf_bias.handle,
            buf_output.handle,
        ] {
            fini_bo(fd, handle).expect("failed to sync BO for the NPU");
        }
        for (buffer, _) in &command_buffers {
            fini_bo(fd, buffer.handle).expect("failed to sync regcmd BO");
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

        submit_jobs(fd, &jobs)
            .unwrap_or_else(|error| panic!("{shape:?} {kernels:?} SUBMIT failed: {error}"));
        prep_bo(fd, buf_output.handle, COMPLETION_TIMEOUT_NS)
            .unwrap_or_else(|error| panic!("{shape:?} {kernels:?} did not complete: {error}"));

        let raw = std::slice::from_raw_parts(buf_output.host_ptr, output_bytes);
        let mut failure = Failure {
            mismatches: 0,
            samples: Vec::new(),
            differences: BTreeSet::new(),
        };
        // The int8 output conversion realises slightly more than the gain it
        // is asked for -- measured, documented in `conv_int8_hw`, and present
        // in the vendor's own programs -- so one LSB is the bar there. fp16
        // arithmetic on these values is exact and takes no tolerance.
        let tolerance: f32 = match shape.precision {
            Precision::Fp16 => 0.0,
            Precision::Int8(_) => 1.0,
        };
        for y in 0..out_height {
            for x in 0..out_width {
                let accumulator = expected_accumulator(shape, kernels, y, x);
                let want: i32 = match shape.precision {
                    Precision::Fp16 => accumulator as i32,
                    Precision::Int8(quantization) => {
                        // Rounds half away from zero, measured by
                        // `conv_int8_probe_hw`.
                        let rounded = if shift == 0 {
                            accumulator
                        } else {
                            (accumulator + (1 << (shift - 1))) >> shift
                        };
                        rounded as i32 + quantization.output_zero_point
                    }
                };
                // Only the real output channels are checked. What the
                // hardware leaves in the padding lanes of the last granule is
                // not something the vendor captures constrain.
                for channel in 0..shape.out_channels as usize {
                    let surface = channel / channels_per_atom;
                    let lane = channel % channels_per_atom;
                    let offset = surface * out_width * out_height * FEATURE_ATOM_BYTES
                        + (y * out_width + x) * FEATURE_ATOM_BYTES
                        + lane * element_bytes;
                    // Compared in f32, not truncated to an integer: a
                    // fractional fp16 result is a real failure and rounding
                    // it first would hide it.
                    let got = match shape.precision {
                        Precision::Fp16 => {
                            f16_to_f32(u16::from_le_bytes([raw[offset], raw[offset + 1]]))
                        }
                        Precision::Int8(_) => f32::from(raw[offset] as i8),
                    };
                    let difference = got - want as f32;
                    failure.differences.insert(difference as i32);
                    if difference.abs() > tolerance {
                        failure.mismatches += 1;
                        if failure.samples.len() < 8 {
                            failure
                                .samples
                                .push(format!("[{y}, {x}, {channel}] want {want} got {got}"));
                        }
                    }
                }
            }
        }

        for handle in [
            buf_input.handle,
            buf_weights.handle,
            buf_bias.handle,
            buf_output.handle,
        ] {
            let _ = close_bo(fd, handle);
        }
        for (buffer, _) in &command_buffers {
            let _ = close_bo(fd, buffer.handle);
        }

        if failure.mismatches == 0 {
            Ok(failure.differences)
        } else {
            Err(failure)
        }
    }
}

/// Smallest shift keeping `Cin * kh * kw` inside int8, so the expected value
/// stays exact and unsaturated.
fn shift_for(in_channels: u32, kernels: Kernels) -> u32 {
    let peak = in_channels * (kernels[0] * kernels[1]) as u32;
    let mut shift = 0;
    while (peak >> shift) > 127 {
        shift += 1;
    }
    shift
}

fn int8_precision(shift: u32) -> Precision {
    Precision::Int8(Quantization {
        input_zero_point: 0,
        output_zero_point: 0,
        // A negative power of two, so the requantisation divides exactly.
        multiplier: Multiplier::for_unit_bs(1.0 / f64::from(1u32 << shift)),
    })
}

fn attempt(plan: &ConvPlan, shift: u32, failures: &mut Vec<String>) {
    let shape = plan.shape();
    let kernels = plan.kernels();
    let padding = shape.padding.unwrap_or([kernels[0] / 2, kernels[1] / 2]);
    let precision = match shape.precision {
        Precision::Fp16 => "fp16",
        Precision::Int8(_) => "int8",
    };
    let label = format!(
        "{}x{} Cin {:>2} Cout {:>3} k{}x{} p{}x{} {precision} d{}/w{}",
        shape.width,
        shape.height,
        shape.in_channels,
        shape.out_channels,
        kernels[0],
        kernels[1],
        padding[0],
        padding[1],
        plan.data_banks(),
        plan.weight_banks(),
    );
    match run(plan, shift) {
        Ok(differences) => println!(
            "  ok   {label} {:>2} tile(s)  out {}x{}  weights {}B  got - want in {:?}",
            plan.tiles().len(),
            shape.output_width(kernels),
            shape.output_height(kernels),
            shape.weight_bytes(kernels),
            differences,
        ),
        Err(failure) => {
            println!(
                "  FAIL {label} {:>2} tile(s)  {} mismatches, got - want in {:?}",
                plan.tiles().len(),
                failure.mismatches,
                failure.differences
            );
            for sample in &failure.samples {
                println!("         {sample}");
            }
            failures.push(label);
        }
    }
}

fn assert_no_failures(failures: Vec<String>) {
    assert!(
        failures.is_empty(),
        "{} configuration(s) produced wrong output: {}",
        failures.len(),
        failures.join(", ")
    );
}

/// Geometries every precision runs, as `(width, height, Cin, Cout)`.
///
/// The 32x64 and 64x32 rows are the same map transposed, so a swapped kernel
/// axis changes the output extent as well as the tap counts. `Cin` 16 crosses
/// into NC1HWC2 surfaces, where the row strides differ.
const GEOMETRIES: [(u32, u32, u32, u32); 4] = [
    (32, 32, 3, 8),
    (64, 32, 3, 8),
    (32, 64, 3, 8),
    (32, 32, 16, 16),
];

/// Kernel shapes, each of which is also run mirrored.
const KERNEL_SHAPES: [(usize, usize); 5] = [(1, 3), (3, 5), (3, 7), (5, 7), (1, 11)];

/// Even-kernel cases as `(extents, explicit_padding)`.
///
/// The square cases cover every measured even extent. The default-padded
/// rectangular pairs cover both-even and mixed-parity axes in both
/// orientations. The final pair makes padding independent from the extent:
/// neither `[0, 2]` nor its mirror can be reconstructed as `kernel / 2`.
const EVEN_KERNEL_CASES: [(Kernels, Option<Padding>); 15] = [
    ([2, 2], None),
    ([4, 4], None),
    ([6, 6], None),
    ([8, 8], None),
    ([10, 10], None),
    ([2, 4], None),
    ([4, 2], None),
    ([6, 10], None),
    ([10, 6], None),
    ([3, 4], None),
    ([4, 3], None),
    ([5, 8], None),
    ([8, 5], None),
    ([4, 6], Some([0, 2])),
    ([6, 4], Some([2, 0])),
];

/// A dense square map and a non-square NC1HWC2 map.
///
/// The latter makes an axis swap change the output shape for the
/// explicit-padding cases and exercises the surface-layout DMA path.
const EVEN_GEOMETRIES: [(u32, u32, u32, u32); 2] = [(32, 32, 3, 8), (40, 24, 16, 16)];

/// Even kernels at the 256x32 `Cin` 32 pressure geometry, as
/// `(extents, Cout)`.
///
/// `EVEN_GEOMETRIES` runs at one or two banks of coefficient demand, so it
/// says nothing about the CBUF split. These are the demands the fill-in
/// captures measured: 4x8 at four banks, 6x8 at six, and 10x10 at five and
/// seven, the two that showed the extent was never what made 10x10 deviate.
const EVEN_PRESSURE_CASES: [(Kernels, u32); 4] =
    [([4, 8], 64), ([6, 8], 64), ([10, 10], 24), ([10, 10], 32)];

/// The int8 half, with `Cout` doubled so the coefficient demand matches its
/// fp16 twin. `6x8` is missing on purpose -- at demand 6 int8 leaves the
/// demand rule where fp16 does not, so `ConvPlan::new` refuses it.
const INT8_EVEN_PRESSURE_CASES: [(Kernels, u32); 3] =
    [([4, 8], 128), ([10, 10], 48), ([10, 10], 64)];

fn even_shape(
    width: u32,
    height: u32,
    in_channels: u32,
    out_channels: u32,
    precision: Precision,
    padding: Option<Padding>,
) -> Shape {
    let shape = Shape::with_precision(width, height, 1, in_channels, out_channels, precision);
    match padding {
        Some(padding) => shape.with_padding(padding),
        None => shape,
    }
}

/// Shapes `ConvPlan::new` refuses, with the split the vendor captured for
/// each. Above five banks of coefficient demand the captured allocation stops
/// following demand, and mirrored shapes stop agreeing.
const EXPLICIT_SPLITS: [(Kernels, u32, u32); 4] = [
    ([11, 5], 8, 4),
    ([5, 11], 5, 7),
    ([9, 7], 8, 4),
    ([7, 9], 4, 8),
];

/// Builds every plan the device tests submit, without a device.
///
/// Planning is pure, so a shape that cannot be planned is a host-side panic
/// that would otherwise only surface after cross-compiling and copying the
/// binary to the board. This runs by default and keeps that round trip for
/// real hardware failures.
#[test]
fn non_square_hardware_matrix_is_plannable() {
    let mut plans = 0;
    for (width, height, in_channels, out_channels) in GEOMETRIES {
        for (kh, kw) in KERNEL_SHAPES {
            for kernels in [[kh, kw], [kw, kh]] {
                let shift = shift_for(in_channels, kernels);
                for precision in [Precision::Fp16, int8_precision(shift)] {
                    let shape = Shape::with_precision(
                        width,
                        height,
                        1,
                        in_channels,
                        out_channels,
                        precision,
                    );
                    let plan = ConvPlan::new(shape, kernels);
                    assert!(
                        !plan.tiles().is_empty(),
                        "{width}x{height} Cin {in_channels} {kernels:?} planned no tiles"
                    );
                    assert!(
                        plan.programs().iter().all(|program| program.len() == 136),
                        "{width}x{height} Cin {in_channels} {kernels:?} program length"
                    );
                    // The output extent has to follow the kernel's own axes,
                    // which is what makes a mirrored pair a real test rather
                    // than the same run twice.
                    assert_eq!(
                        (shape.output_width(kernels), shape.output_height(kernels)),
                        (width, height),
                        "{width}x{height} {kernels:?} output extent"
                    );
                    plans += 1;
                }
            }
        }
    }

    for kernels in [[3usize, 9], [9, 3]] {
        let shift = shift_for(32, kernels);
        for precision in [Precision::Fp16, int8_precision(shift)] {
            let shape = Shape::with_precision(256, 32, 1, 32, 64, precision);
            assert!(!ConvPlan::new(shape, kernels).tiles().is_empty());
            plans += 1;
        }
    }

    for (kernels, data_banks, weight_banks) in EXPLICIT_SPLITS {
        let shape = Shape::with_out_channels(256, 32, 1, 32, 64);
        let plan = ConvPlan::with_cbuf_banks(shape, kernels, data_banks, weight_banks);
        assert!(!plan.tiles().is_empty(), "{kernels:?} planned no tiles");
        plans += 1;
    }

    // Geometries x shapes x mirrors x precisions, the wide tall-kernel pair
    // in both precisions, and the explicit splits.
    assert_eq!(
        plans,
        2 * GEOMETRIES.len() * KERNEL_SHAPES.len() * 2 + 2 * 2 + EXPLICIT_SPLITS.len()
    );
}

/// Builds the even-kernel device matrix on the host before the board run.
#[test]
fn even_kernel_hardware_matrix_is_plannable() {
    let mut plans = 0;
    for (width, height, in_channels, out_channels) in EVEN_GEOMETRIES {
        for (kernels, explicit_padding) in EVEN_KERNEL_CASES {
            let padding = explicit_padding.unwrap_or([kernels[0] / 2, kernels[1] / 2]);
            let shift = shift_for(in_channels, kernels);
            for precision in [Precision::Fp16, int8_precision(shift)] {
                let shape = even_shape(
                    width,
                    height,
                    in_channels,
                    out_channels,
                    precision,
                    explicit_padding,
                );
                let plan = ConvPlan::new(shape, kernels);
                assert!(
                    !plan.tiles().is_empty(),
                    "{width}x{height} Cin {in_channels} {kernels:?} \
                     padding {padding:?} planned no tiles"
                );
                assert!(
                    plan.programs().iter().all(|program| program.len() == 136),
                    "{width}x{height} Cin {in_channels} {kernels:?} \
                     padding {padding:?} program length"
                );
                assert_eq!(
                    (shape.output_width(kernels), shape.output_height(kernels)),
                    (
                        width + 2 * padding[1] as u32 - kernels[1] as u32 + 1,
                        height + 2 * padding[0] as u32 - kernels[0] as u32 + 1,
                    ),
                    "{width}x{height} {kernels:?} padding {padding:?} output extent"
                );
                plans += 1;
            }
        }
    }

    assert_eq!(plans, EVEN_GEOMETRIES.len() * EVEN_KERNEL_CASES.len() * 2);

    // The pressure cases, where the CBUF split rather than the geometry is
    // what is under test. These are planned per precision because the two
    // bounds differ there, so the fp16 and int8 sets are not the same shapes.
    for (kernels, out_channels) in EVEN_PRESSURE_CASES {
        let shape = Shape::with_out_channels(256, 32, 1, 32, out_channels);
        assert!(
            !ConvPlan::new(shape, kernels).tiles().is_empty(),
            "fp16 pressure {kernels:?} Cout {out_channels} planned no tiles"
        );
    }
    for (kernels, out_channels) in INT8_EVEN_PRESSURE_CASES {
        let shift = shift_for(32, kernels);
        let shape = Shape::with_precision(256, 32, 1, 32, out_channels, int8_precision(shift));
        assert!(
            !ConvPlan::new(shape, kernels).tiles().is_empty(),
            "int8 pressure {kernels:?} Cout {out_channels} planned no tiles"
        );
    }
}

#[test]
fn even_kernel_cpu_reference_counts_the_exact_window_intersection() {
    let default_padding = Shape::new(5, 5);
    assert_eq!(
        (
            default_padding.output_width([4, 4]),
            default_padding.output_height([4, 4]),
        ),
        (6, 6)
    );
    assert_eq!(expected_accumulator(default_padding, [4, 4], 0, 0), 12);
    assert_eq!(expected_accumulator(default_padding, [4, 4], 2, 2), 48);
    assert_eq!(expected_accumulator(default_padding, [4, 4], 5, 5), 12);

    let explicit_padding = Shape::new(5, 5).with_padding([0, 1]);
    assert_eq!(
        (
            explicit_padding.output_width([4, 4]),
            explicit_padding.output_height([4, 4]),
        ),
        (4, 2)
    );
    assert_eq!(expected_accumulator(explicit_padding, [4, 4], 0, 0), 36);
    assert_eq!(expected_accumulator(explicit_padding, [4, 4], 0, 1), 48);
    assert_eq!(expected_accumulator(explicit_padding, [4, 4], 1, 3), 36);
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn fp16_non_square_kernels_run_on_npu() {
    let _device_guard = NPU_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let mut failures = Vec::new();

    for (width, height, in_channels, out_channels) in GEOMETRIES {
        for (kh, kw) in KERNEL_SHAPES {
            for kernels in [[kh, kw], [kw, kh]] {
                let shape = Shape::with_out_channels(width, height, 1, in_channels, out_channels);
                attempt(&ConvPlan::new(shape, kernels), 0, &mut failures);
            }
        }
    }

    // A tall kernel on a wide map at Cin 32: the halo is ten rows against a
    // tile the CBUF caps well below the image height, so this exercises row
    // splitting driven by the kernel's height rather than its area.
    for kernels in [[3usize, 9], [9, 3]] {
        let shape = Shape::with_out_channels(256, 32, 1, 32, 64);
        attempt(&ConvPlan::new(shape, kernels), 0, &mut failures);
    }

    assert_no_failures(failures);
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn fp16_even_square_and_rectangular_kernels_run_on_npu() {
    let _device_guard = NPU_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let mut failures = Vec::new();

    for (width, height, in_channels, out_channels) in EVEN_GEOMETRIES {
        for (kernels, padding) in EVEN_KERNEL_CASES {
            let shape = even_shape(
                width,
                height,
                in_channels,
                out_channels,
                Precision::Fp16,
                padding,
            );
            attempt(&ConvPlan::new(shape, kernels), 0, &mut failures);
        }
    }

    // The pressure geometry, which the matrix above never reaches: at Cin 32
    // and these Cout the coefficient claim is five and seven banks rather
    // than one or two, so the CBUF split is doing real work. 10x10 is here
    // because it is newly plannable -- the fill-in captures show it follows
    // the demand rule below the ceiling, where a single Cout 64 capture had
    // previously made the whole extent look special.
    for (kernels, out_channels) in EVEN_PRESSURE_CASES {
        let shape = Shape::with_out_channels(256, 32, 1, 32, out_channels);
        attempt(&ConvPlan::new(shape, kernels), 0, &mut failures);
    }

    assert_no_failures(failures);
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn int8_non_square_kernels_run_on_npu() {
    let _device_guard = NPU_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let mut failures = Vec::new();

    for (width, height, in_channels, out_channels) in GEOMETRIES {
        for (kh, kw) in KERNEL_SHAPES {
            for kernels in [[kh, kw], [kw, kh]] {
                let shift = shift_for(in_channels, kernels);
                let shape = Shape::with_precision(
                    width,
                    height,
                    1,
                    in_channels,
                    out_channels,
                    int8_precision(shift),
                );
                attempt(&ConvPlan::new(shape, kernels), shift, &mut failures);
            }
        }
    }

    // The same wide, tall-kernel case as the fp16 test. An int8 coefficient
    // is one byte, so this asks for half the coefficient banks its fp16 twin
    // does and takes a different split for the same geometry.
    for kernels in [[3usize, 9], [9, 3]] {
        let shift = shift_for(32, kernels);
        let shape = Shape::with_precision(256, 32, 1, 32, 64, int8_precision(shift));
        attempt(&ConvPlan::new(shape, kernels), shift, &mut failures);
    }

    assert_no_failures(failures);
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn int8_even_square_and_rectangular_kernels_run_on_npu() {
    let _device_guard = NPU_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let mut failures = Vec::new();

    for (width, height, in_channels, out_channels) in EVEN_GEOMETRIES {
        for (kernels, padding) in EVEN_KERNEL_CASES {
            let shift = shift_for(in_channels, kernels);
            let shape = even_shape(
                width,
                height,
                in_channels,
                out_channels,
                int8_precision(shift),
                padding,
            );
            attempt(&ConvPlan::new(shape, kernels), shift, &mut failures);
        }
    }

    // The int8 pressure set is not the fp16 one with a doubled Cout. A 6x8
    // reaches demand 6 there, which int8 captures as 8/4 against the demand
    // rule's 6/6, so the planner refuses it and it is absent here. The square
    // 10x10 cases stay, because the even square bound holds in both
    // precisions.
    for (kernels, out_channels) in INT8_EVEN_PRESSURE_CASES {
        let shift = shift_for(32, kernels);
        let shape = Shape::with_precision(256, 32, 1, 32, out_channels, int8_precision(shift));
        attempt(&ConvPlan::new(shape, kernels), shift, &mut failures);
    }

    assert_no_failures(failures);
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn explicit_cbuf_split_runs_the_refused_non_square_kernels_on_npu() {
    let _device_guard = NPU_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let mut failures = Vec::new();

    // Above five banks of coefficient demand the captured split stops
    // following demand -- 11x5 takes 8/4 where its mirror 5x11 takes 5/7 --
    // so `ConvPlan::new` refuses these and the caller supplies the split.
    // The pairs below are the captured allocations for each shape, which is
    // the thing worth knowing runs: if the vendor's own choice did not work
    // here, the refusal would be hiding a deeper problem than a missing rule.
    for (kernels, data_banks, weight_banks) in EXPLICIT_SPLITS {
        let shape = Shape::with_out_channels(256, 32, 1, 32, 64);
        let plan = ConvPlan::with_cbuf_banks(shape, kernels, data_banks, weight_banks);
        attempt(&plan, 0, &mut failures);
    }

    assert_no_failures(failures);
}
