//! Hardware validation of quantized int8 convolution.
//!
//! This test is ignored on the development host because it needs the RK3588
//! NPU device. Cross-compile it, copy the printed test binary to the board,
//! and run it there:
//!
//!   CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!     cargo test --target aarch64-unknown-linux-gnu --release \
//!       --test conv_int8_hw --no-run
//!
//!   ./conv_int8_hw-<hash> --ignored --nocapture
//!
//! # What is under test
//!
//! Everything the int8 path does differently: the precision enum replicated
//! across four blocks, doubled channel granules on both axes, a coefficient
//! footprint at one byte per element, padding that contributes the input
//! zero point rather than zero, and a requantisation stage the fp16 path
//! bypasses entirely.
//!
//! It also tests the BS buffer, which is why this test could not be written
//! from the register corpus alone. At fp16 `brdma_data_use` is 1 and BRDMA
//! fetches only the bias, so every fp16 test zeroes that buffer and passes.
//! At int8 it is 7: BRDMA also fetches a per-channel multiplier, and a
//! zeroed buffer multiplies the whole tensor by zero. The layout came from
//! diffing converted models whose biases and per-channel weight magnitudes
//! were varied independently; `write_bs_buffer` encodes it.
//!
//! # Making the expected value exact
//!
//! Every input element is 1, every coefficient is 1, the input zero point is
//! 0, and the bias is 0, so the accumulator at an output pixel is exactly
//! `Cin * taps_inside_the_image` -- the same quantity the fp16 tests check,
//! and independent of the order the hardware walks the coefficients in.
//!
//! The BS multiplier is left at unit, so the only scaling is the output
//! conversion. Its multiplier is chosen as a negative power of two large
//! enough to keep the result inside int8 and small enough to divide the
//! accumulator exactly, which takes rounding out of the comparison: an
//! expected value of 36 is 36, not 36 give or take a half-LSB.
//!
//! The output zero point is 0 for the same reason. A nonzero one is tested
//! separately, as a constant offset on a case already known to pass.
//!
//! # Why the tolerance is one LSB and not zero
//!
//! The output conversion realises slightly more than the gain it is asked
//! for: sweeping the accumulator at unit gain, the output tracks it exactly
//! to 64 and runs one high from 65 to 126. Four models were fitted to that
//! and all four were wrong -- it is not a threshold in the accumulator, not
//! one in the output, not a fixed ratio, and not an additive offset on the
//! scale mantissa.
//!
//! What settles the question is that the regcmd this builder emits is
//! identical to the vendor's, raw 32-bit word for raw 32-bit word, across
//! all 33 int8 captures, and the BS buffer carries what the vendor writes
//! for uniform weights. The vendor's own programs therefore behave the same
//! way. The gain is the hardware's, not this builder's, and it stays inside
//! quantisation noise for scales derived from calibration -- which is why
//! nothing in the vendor toolchain would ever surface it.
//!
//! So exact equality is the wrong acceptance criterion here; it was mine,
//! and it is stricter than the hardware's own contract. One LSB is the
//! usual bar for quantized inference. The distribution of differences is
//! printed regardless, so a systematic offset stays visible rather than
//! being absorbed by the tolerance.

use std::{collections::BTreeSet, fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    builders::{
        RegCmd, RegisterMeta,
        cna::{CnaDcompAddr0, CnaFeatureDataAddr},
        dpu::DpuDstBaseAddr,
        dpu_rdma::DpuRdmaBsBaseAddr,
    },
    conv::{
        BsEntry, FeatureLayout, Kernels, Multiplier, Precision, Quantization, Shape, Tile,
        bs_buffer_bytes, conv_2d_tile, write_bs_buffer,
    },
    device::{Buffer, JobDesc, close_bo, fini_bo, prep_bo, submit_jobs},
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const FEATURE_ATOM_BYTES: usize = 16;
/// int8 lanes in one 16-byte atom, on both the input and output sides.
const CHANNELS_PER_ATOM: usize = 16;
const PAGE_BYTES: usize = 4096;

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

fn valid_taps(coordinate: usize, extent: usize, kernel: usize) -> usize {
    match kernel {
        1 => 1,
        3 => 3 - usize::from(coordinate == 0) - usize::from(coordinate + 1 == extent),
        _ => unreachable!("only 1x1 and 3x3 have vendor reference data"),
    }
}

/// Writes the input feature map: every real channel 1, every padding lane 0.
unsafe fn fill_input(base: *mut u8, size: usize, shape: Shape, surfaces: usize) {
    unsafe {
        ptr::write_bytes(base, 0, size);
        let width = shape.width as usize;
        let height = shape.height as usize;
        for channel in 0..shape.in_channels as usize {
            let surface = channel / CHANNELS_PER_ATOM;
            let lane = channel % CHANNELS_PER_ATOM;
            if surface >= surfaces {
                continue;
            }
            for y in 0..height {
                for x in 0..width {
                    let offset = match shape.layout() {
                        FeatureLayout::Dense => {
                            (y * width + x) * shape.in_channels as usize + channel
                        }
                        FeatureLayout::Surfaces => {
                            surface * width * height * FEATURE_ATOM_BYTES
                                + (y * width + x) * FEATURE_ATOM_BYTES
                                + lane
                        }
                    };
                    ptr::write(base.add(offset), 1u8);
                }
            }
        }
    }
}

struct Failure {
    mismatches: usize,
    samples: Vec<String>,
    /// Every distinct `got - want` seen. A single entry means a uniform
    /// offset, which is a very different diagnosis from a scatter, and it
    /// is worth seeing without having to run a second binary.
    differences: BTreeSet<i32>,
}

/// `shift` is the negative power of two the output conversion applies.
fn run(
    shape: Shape,
    kernels: Kernels,
    shift: u32,
    output_zero_point: i32,
) -> Result<BTreeSet<i32>, Failure> {
    let width = shape.width as usize;
    let height = shape.height as usize;
    let out_width = shape.output_width(kernels) as usize;
    let out_height = shape.output_height(kernels) as usize;
    let in_surfaces = (shape.weight_channels() as usize).div_ceil(CHANNELS_PER_ATOM);
    let out_surfaces = (shape.padded_out_channels() as usize).div_ceil(CHANNELS_PER_ATOM);

    let input_bytes = match shape.layout() {
        FeatureLayout::Dense => width * height * shape.in_channels as usize,
        FeatureLayout::Surfaces => in_surfaces * width * height * FEATURE_ATOM_BYTES,
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

        // One byte per coefficient, all 1. Padding channels multiply zeroed
        // input, so they contribute nothing whatever order they are read in.
        let weight_bytes = shape.weight_bytes(kernels) as usize;
        let buf_weights = Buffer::new(fd, page_aligned_size(weight_bytes), &file);
        ptr::write_bytes(buf_weights.host_ptr, 0, buf_weights.size);
        ptr::write_bytes(buf_weights.host_ptr, 1, weight_bytes);

        // The BS buffer. Zero bias at unit multiplier -- and emphatically not
        // a zeroed buffer, which would multiply everything by zero.
        let bs_bytes = bs_buffer_bytes(shape.out_channels);
        let buf_bs = Buffer::new(fd, page_aligned_size(bs_bytes), &file);
        ptr::write_bytes(buf_bs.host_ptr, 0, buf_bs.size);
        let entries = vec![BsEntry::default(); shape.out_channels as usize];
        write_bs_buffer(
            std::slice::from_raw_parts_mut(buf_bs.host_ptr, buf_bs.size),
            &entries,
        );

        let buf_output = Buffer::new(fd, page_aligned_size(output_bytes), &file);
        ptr::write_bytes(buf_output.host_ptr, 0, buf_output.size);

        let split = Tile::split(shape, kernels, shape.min_tiles(kernels));
        let mut command_buffers = Vec::with_capacity(split.len());
        for tile in &split {
            let mut commands = conv_2d_tile(shape, kernels, tile);
            relocate::<CnaFeatureDataAddr>(&mut commands, buf_input.dma_address);
            relocate::<CnaDcompAddr0>(&mut commands, buf_weights.dma_address);
            relocate::<DpuRdmaBsBaseAddr>(&mut commands, buf_bs.dma_address);
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
            buf_bs.handle,
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
                    buf_bs.handle,
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
        prep_bo(fd, buf_output.handle, 5_000_000_000)
            .unwrap_or_else(|error| panic!("{shape:?} {kernels:?} did not complete: {error}"));

        let raw = std::slice::from_raw_parts(buf_output.host_ptr as *const i8, output_bytes);
        let mut failure = Failure {
            mismatches: 0,
            samples: Vec::new(),
            differences: BTreeSet::new(),
        };
        for y in 0..out_height {
            for x in 0..out_width {
                let stride = shape.stride as usize;
                let accumulator = shape.in_channels as usize
                    * valid_taps(y * stride, height, kernels[0])
                    * valid_taps(x * stride, width, kernels[1]);
                // The hardware rounds half away from zero, measured
                // directly by `conv_int8_probe_hw`: a BS multiplier of 64
                // against a unit shift returns 1, not 0. Truncating here
                // reported Cin 17 as a hardware fault when the only thing
                // wrong was this line.
                let rounded = if shift == 0 {
                    accumulator
                } else {
                    (accumulator + (1 << (shift - 1))) >> shift
                };
                let want = rounded as i32 + output_zero_point;
                for channel in 0..shape.out_channels as usize {
                    let surface = channel / CHANNELS_PER_ATOM;
                    let lane = channel % CHANNELS_PER_ATOM;
                    let offset = surface * out_width * out_height * FEATURE_ATOM_BYTES
                        + (y * out_width + x) * FEATURE_ATOM_BYTES
                        + lane;
                    let got = i32::from(raw[offset]);
                    failure.differences.insert(got - want);
                    if (got - want).abs() > 1 {
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
            buf_bs.handle,
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

/// Picks the smallest shift keeping `Cin * k * k` inside int8, so the
/// expected value stays exact and unsaturated.
fn shift_for(in_channels: u32, kernel: usize) -> u32 {
    let peak = in_channels * (kernel * kernel) as u32;
    let mut shift = 0;
    while (peak >> shift) > 127 {
        shift += 1;
    }
    shift
}

fn attempt(
    in_channels: u32,
    out_channels: u32,
    width: u32,
    kernel: usize,
    output_zero_point: i32,
    failures: &mut Vec<String>,
) {
    let kernels = [kernel, kernel];
    let shift = shift_for(in_channels, kernel);
    let precision = Precision::Int8(Quantization {
        input_zero_point: 0,
        output_zero_point,
        // A negative power of two, so the requantisation divides exactly.
        // `for_unit_bs` cancels the gain the unit BS multiplier carries: the
        // BS stage shifts by 7, not the 14 the register suggests. That ratio
        // between the two stages is solid -- five (bs_multiplier, cvt) pairs
        // encoding the same nominal gain all agree. What it does not cover
        // is the residual the output stage adds, which is why the check
        // below allows one LSB.
        multiplier: Multiplier::for_unit_bs(1.0 / f64::from(1u32 << shift)),
    });
    let shape = Shape::with_precision(width, 32, 1, in_channels, out_channels, precision);
    let label = format!(
        "Cin {in_channels:>3} Cout {out_channels:>3} {width}x32 {kernels:?} >>{shift} zp {output_zero_point}"
    );
    match run(shape, kernels, shift, output_zero_point) {
        Ok(differences) => println!(
            "  ok   {label}  {:?} padded {}/{}  got - want in {:?}",
            shape.layout(),
            shape.padded_channels(),
            shape.padded_out_channels(),
            differences
        ),
        Err(failure) => {
            println!(
                "  FAIL {label}  {} mismatches, got - want in {:?}",
                failure.mismatches, failure.differences
            );
            for sample in &failure.samples {
                println!("         {sample}");
            }
            failures.push(label);
        }
    }
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn int8_convs_run_on_npu() {
    let mut failures = Vec::new();

    // Both sides of the dense/surface boundary, which int8 does not move.
    for in_channels in [1u32, 3, 4, 8, 16, 17, 32, 64] {
        for kernel in [1usize, 3] {
            attempt(in_channels, 8, 64, kernel, 0, &mut failures);
        }
    }

    // Output channels across the 32-wide granule, including values that are
    // not multiples of it.
    for out_channels in [1u32, 8, 16, 20, 32, 40, 64] {
        attempt(16, out_channels, 64, 3, 0, &mut failures);
    }

    // A nonzero output zero point, as a constant offset on a case that
    // already passes above.
    attempt(16, 8, 64, 3, -3, &mut failures);
    attempt(16, 8, 64, 3, 7, &mut failures);

    // Wide enough to tile, so the int8 capacity rule is exercised rather
    // than assumed. 256 wide at Cin 32 is the shape that caught the fp16
    // capacity bug.
    attempt(32, 8, 256, 3, 0, &mut failures);
    attempt(3, 8, 256, 3, 0, &mut failures);

    assert!(
        failures.is_empty(),
        "{} configuration(s) produced wrong output: {}",
        failures.len(),
        failures.join(", ")
    );
}
