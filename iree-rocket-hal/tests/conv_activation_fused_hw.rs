//! Hardware-in-the-loop tests for `conv.rs`'s fused activation, the BN-stage
//! path derived from the 30-capture activation sweep (see DESIGN_NOTES.md,
//! "Fused activation: the vendor uses BN, not BS").
//!
//! Distinct from `mesa_conv`'s BS-stage port of the same idea (its own
//! hardware coverage has since been retired as redundant with this file).
//! The two program different registers; this file is what says the
//! capture-derived one computes the right thing.
//!
//! # Why Cin 1
//!
//! With unit weights the unactivated output at a pixel is just its valid tap
//! count: 4 at a corner, 6 along an edge, 9 in the interior. A ceiling of 6
//! therefore clamps the interior while leaving the corners alone, so a
//! passing run distinguishes a real clamp from both "no clamp" and "clamp
//! everything". A larger Cin would push every pixel above the ceiling and
//! the test would pass without discriminating.
//!
//! Every expected value is exact in fp16.

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    conv::{Activation, Buffers, Kernels, Shape, Tile, conv_2d_tile, relocate},
    device::{Buffer, JobDesc, close_bo, fini_bo, prep_bo, submit_jobs},
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const OUTPUT_CHANNELS: usize = 8;
const FP16_BYTES: usize = 2;
const FEATURE_ATOM_BYTES: usize = 16;
const PAGE_BYTES: usize = 4096;
const FP16_ONE: u16 = 0x3c00;
const FP16_MINUS_ONE: u16 = 0xbc00;

fn page_aligned_size(size: usize) -> usize {
    size.div_ceil(PAGE_BYTES) * PAGE_BYTES
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
            let unbiased = (exponent + 127 - 15) as u32;
            (sign << 31) | (unbiased << 23) | ((mantissa & 0x3ff) << 13)
        }
        _ => (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13),
    };
    f32::from_bits(word)
}

fn valid_taps(coordinate: usize, extent: usize) -> usize {
    3 - usize::from(coordinate == 0) - usize::from(coordinate + 1 == extent)
}

struct Failure {
    mismatches: usize,
    samples: Vec<String>,
}

/// Runs one 32x32 Cin-1 3x3 convolution with `activation` fused, every input
/// element `input` and every coefficient 1.0, and checks each output against
/// `expected(unactivated_value)`.
fn run(activation: Activation, input: u16, expected: impl Fn(f32) -> f32) -> Result<(), Failure> {
    let kernels: Kernels = [3, 3];
    let shape = Shape::with_channels(32, 32, 1, 1).with_activation(activation);
    let width = shape.width as usize;
    let height = shape.height as usize;
    let out_width = shape.output_width(kernels) as usize;
    let out_height = shape.output_height(kernels) as usize;

    let input_bytes = width * height * shape.in_channels as usize * FP16_BYTES;
    let output_bytes = out_width * out_height * FEATURE_ATOM_BYTES * 2;

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let buf_input = Buffer::new(fd, page_aligned_size(input_bytes), &file);
        ptr::write_bytes(buf_input.host_ptr, 0, buf_input.size);
        std::slice::from_raw_parts_mut(buf_input.host_ptr as *mut u16, input_bytes / 2).fill(input);

        // Coefficients cover the padded channel count. Padding channels
        // multiply zeroed input, so they contribute nothing.
        let weight_bytes = kernels[0]
            * kernels[1]
            * shape.weight_channels() as usize
            * OUTPUT_CHANNELS
            * FP16_BYTES;
        let buf_weights = Buffer::new(fd, page_aligned_size(weight_bytes), &file);
        ptr::write_bytes(buf_weights.host_ptr, 0, buf_weights.size);
        std::slice::from_raw_parts_mut(buf_weights.host_ptr as *mut u16, weight_bytes / 2)
            .fill(FP16_ONE);

        let buf_bias = Buffer::new(fd, PAGE_BYTES, &file);
        ptr::write_bytes(buf_bias.host_ptr, 0, buf_bias.size);
        let buf_output = Buffer::new(fd, page_aligned_size(output_bytes), &file);
        ptr::write_bytes(buf_output.host_ptr, 0, buf_output.size);

        let mut commands = conv_2d_tile(shape, kernels, &Tile::whole(shape, kernels));
        relocate(
            &mut commands,
            Buffers {
                input: buf_input.dma_address,
                weights: buf_weights.dma_address,
                bias: buf_bias.dma_address,
                output: buf_output.dma_address,
            },
        );

        let command_bytes = commands.len() * mem::size_of::<u64>();
        let buf_commands = Buffer::new(fd, page_aligned_size(command_bytes), &file);
        ptr::write_bytes(buf_commands.host_ptr, 0, buf_commands.size);
        let words =
            std::slice::from_raw_parts_mut(buf_commands.host_ptr as *mut u64, commands.len());
        for (destination, command) in words.iter_mut().zip(&commands) {
            *destination = command.0;
        }

        for handle in [
            buf_input.handle,
            buf_weights.handle,
            buf_bias.handle,
            buf_output.handle,
            buf_commands.handle,
        ] {
            fini_bo(fd, handle).expect("failed to sync BO for the NPU");
        }

        let tasks = [(buf_commands.dma_address, commands.len() as u32)];
        let in_handles = [
            buf_commands.handle,
            buf_input.handle,
            buf_weights.handle,
            buf_bias.handle,
        ];
        let out_handles = [buf_output.handle];
        submit_jobs(
            fd,
            &[JobDesc {
                tasks: &tasks,
                in_handles: &in_handles,
                out_handles: &out_handles,
            }],
        )
        .unwrap_or_else(|error| panic!("{activation:?} SUBMIT failed: {error}"));
        prep_bo(fd, buf_output.handle, 5_000_000_000)
            .unwrap_or_else(|error| panic!("{activation:?} did not complete: {error}"));

        let raw = std::slice::from_raw_parts(buf_output.host_ptr, output_bytes);
        let mut failure = Failure {
            mismatches: 0,
            samples: Vec::new(),
        };
        let signed = f16_to_f32(input);
        for y in 0..out_height {
            for x in 0..out_width {
                let unactivated = signed * (valid_taps(y, height) * valid_taps(x, width)) as f32;
                let want = expected(unactivated);
                for channel in 0..OUTPUT_CHANNELS {
                    let offset = (y * out_width + x) * FEATURE_ATOM_BYTES + channel * FP16_BYTES;
                    let got = f16_to_f32(u16::from_le_bytes([raw[offset], raw[offset + 1]]));
                    if got != want {
                        failure.mismatches += 1;
                        if failure.samples.len() < 8 {
                            failure.samples.push(format!(
                                "[{y}, {x}, {channel}] unactivated {unactivated} want {want} got {got}"
                            ));
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
            buf_commands.handle,
        ] {
            let _ = close_bo(fd, handle);
        }

        if failure.mismatches == 0 {
            Ok(())
        } else {
            Err(failure)
        }
    }
}

fn check(name: &str, result: Result<(), Failure>) {
    if let Err(failure) = result {
        panic!(
            "{name}: {} mismatches\n  {}",
            failure.mismatches,
            failure.samples.join("\n  ")
        );
    }
}

/// The control. Without it a broken activation and a broken convolution look
/// the same.
#[test]
#[ignore = "requires a real RK3588 NPU at /dev/accel/accel0"]
fn unactivated_conv_returns_its_tap_count() {
    check(
        "none",
        run(Activation::None, FP16_ONE, |unactivated| unactivated),
    );
}

#[test]
#[ignore = "requires a real RK3588 NPU at /dev/accel/accel0"]
fn relu_passes_positive_output_through() {
    check(
        "relu positive",
        run(Activation::Relu, FP16_ONE, |unactivated| unactivated),
    );
}

/// Every unactivated value here is negative (-4, -6, -9), so a working relu
/// makes the whole output exactly zero and a bypassed one leaves it negative.
#[test]
#[ignore = "requires a real RK3588 NPU at /dev/accel/accel0"]
fn relu_clamps_negative_output_to_zero() {
    check(
        "relu negative",
        run(Activation::Relu, FP16_MINUS_ONE, |_| 0.0),
    );
}

/// The discriminating case: corners stay at 4 while the interior is cut from
/// 9 to 6. Passing rules out both a missing clamp and a clamp at the wrong
/// ceiling, which is what pins `BN_RELUX_CMP_VALUE` being the ceiling's raw
/// binary32 bit pattern.
#[test]
#[ignore = "requires a real RK3588 NPU at /dev/accel/accel0"]
fn relu6_clamps_only_what_exceeds_its_ceiling() {
    check(
        "relu6",
        run(Activation::clamped_fp16(6.0), FP16_ONE, |unactivated| {
            unactivated.min(6.0)
        }),
    );
}

/// A second ceiling, so a hardcoded 6.0 anywhere in the encoding path fails.
/// Corners (4) pass through, edges (6) and interior (9) both clamp to 5.
#[test]
#[ignore = "requires a real RK3588 NPU at /dev/accel/accel0"]
fn a_second_ceiling_clamps_at_its_own_value() {
    check(
        "clip5",
        run(Activation::clamped_fp16(5.0), FP16_ONE, |unactivated| {
            unactivated.min(5.0)
        }),
    );
}
