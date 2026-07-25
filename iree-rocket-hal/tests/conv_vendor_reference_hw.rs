//! Hardware-in-the-loop test for the bit-exact vendor reference in
//! `rocket::conv::conv_2d`.
//!
//! This test is ignored on the development host because it needs the RK3588
//! NPU device. Cross-compile it, copy the printed test binary to the board,
//! and run it there:
//!
//!   CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!     cargo test --target aarch64-unknown-linux-gnu --release \
//!       --test conv_vendor_reference_hw --no-run
//!
//!   ./conv_vendor_reference_hw-<hash> --ignored --nocapture
//!
//! The captured program is specifically a `32x32x3 -> 32x32x8` fp16
//! convolution. Its input is dense NHWC, not NC1HWC2: the alternative
//! height-split groups advance the input address by exactly
//! `rows * 32 * 3 * sizeof(fp16)` (`0xc00` for 16 rows in the 1x1 capture,
//! and `0xb40` for 15 rows in the overlapping 3x3 capture). Input and
//! weights are both 1.0 and bias is zero, so every output has a simple exact
//! value:
//!
//! - 1x1: `3`
//! - padded 3x3: `12` at corners, `18` at non-corner edges, `27` inside
//!
//! These are all exactly representable as fp16. Uniform weights intentionally
//! make this first hardware test independent of the still-narrow reference
//! builder's coefficient-order API while still proving input DMA, weight DMA,
//! padding, accumulation, output conversion, and output DMA end to end.

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    builders::{
        RegCmd, RegisterMeta,
        cna::{CnaDcompAddr0, CnaFeatureDataAddr},
        dpu::DpuDstBaseAddr,
        dpu_rdma::DpuRdmaBsBaseAddr,
    },
    conv::{Kernels, conv_2d},
    device::{Buffer, close_bo, fini_bo, prep_bo, submit},
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const WIDTH: usize = 32;
const HEIGHT: usize = 32;
const INPUT_CHANNELS: usize = 3;
const WEIGHT_INPUT_CHANNELS: usize = 8;
const OUTPUT_CHANNELS: usize = 8;
const FP16_BYTES: usize = 2;
const FEATURE_ATOM_BYTES: usize = 16;
const PAGE_BYTES: usize = 4096;

const INPUT_BYTES: usize = WIDTH * HEIGHT * INPUT_CHANNELS * FP16_BYTES;
// DPU_DATA_CUBE_CHANNEL/DPU_WDMA_SIZE_0 program 16 physical fp16 channels.
// They occupy two 16-byte NC1HWC2 surfaces; the eight real outputs are in
// the first surface and DST_SURF_STRIDE places the second after 0x4000 bytes.
const OUTPUT_BYTES: usize = WIDTH * HEIGHT * FEATURE_ATOM_BYTES * 2;

const FP16_ONE: u16 = 0x3c00;

fn page_aligned_size(byte_len: usize) -> usize {
    byte_len.max(1).next_multiple_of(PAGE_BYTES)
}

fn decode_identity(command: &RegCmd) -> (u32, u32) {
    ((command.0 >> 48) as u32, command.0 as u32 & 0xffff)
}

/// Replaces the one captured zero-valued address command for `R`.
///
/// Matching by typed register identity instead of a hardcoded vector index
/// makes the test fail clearly if the reference sequence is reordered or
/// unexpectedly gains a second write to the same address register.
fn relocate<R: RegisterMeta>(commands: &mut [RegCmd], address: u32) {
    assert_eq!(
        address & 0xf,
        0,
        "NPU DMA address for register {:#x}:{:#x} is not 16-byte aligned",
        R::DOMAIN,
        R::OFFSET
    );

    let matches: Vec<_> = commands
        .iter()
        .enumerate()
        .filter_map(|(index, command)| {
            (decode_identity(command) == (R::DOMAIN, R::OFFSET)).then_some(index)
        })
        .collect();
    assert_eq!(
        matches.len(),
        1,
        "expected exactly one {:#x}:{:#x} relocation, found {matches:?}",
        R::DOMAIN,
        R::OFFSET
    );
    commands[matches[0]] = RegCmd::new(R::DOMAIN, R::OFFSET, address);
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

fn valid_taps(coordinate: usize, extent: usize, kernel: usize) -> usize {
    match kernel {
        1 => 1,
        3 => 3 - usize::from(coordinate == 0) - usize::from(coordinate + 1 == extent),
        _ => unreachable!("conv_2d rejects kernels other than 1x1 and 3x3"),
    }
}

fn expected_output(kernels: Kernels, y: usize, x: usize) -> f32 {
    (INPUT_CHANNELS * valid_taps(y, HEIGHT, kernels[0]) * valid_taps(x, WIDTH, kernels[1])) as f32
}

fn run_vendor_reference_conv(kernels: Kernels) -> Vec<f32> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let buf_input = Buffer::new(fd, page_aligned_size(INPUT_BYTES), &file);
        ptr::write_bytes(buf_input.host_ptr, 0, buf_input.size);
        let input = std::slice::from_raw_parts_mut(
            buf_input.host_ptr as *mut u16,
            INPUT_BYTES / FP16_BYTES,
        );
        // The vendor feature DMA consumes three contiguous fp16 values per
        // NHWC pixel. It performs the physical C8 padding internally.
        input.fill(FP16_ONE);

        let weight_bytes =
            kernels[0] * kernels[1] * WEIGHT_INPUT_CHANNELS * OUTPUT_CHANNELS * FP16_BYTES;
        let buf_weights = Buffer::new(fd, page_aligned_size(weight_bytes), &file);
        ptr::write_bytes(buf_weights.host_ptr, 0, buf_weights.size);
        let weights =
            std::slice::from_raw_parts_mut(buf_weights.host_ptr as *mut u16, weight_bytes / 2);
        weights.fill(FP16_ONE);

        let buf_bias = Buffer::new(fd, PAGE_BYTES, &file);
        ptr::write_bytes(buf_bias.host_ptr, 0, buf_bias.size);

        let buf_output = Buffer::new(fd, OUTPUT_BYTES, &file);
        ptr::write_bytes(buf_output.host_ptr, 0, buf_output.size);

        let mut commands = conv_2d(kernels);
        relocate::<CnaFeatureDataAddr>(&mut commands, buf_input.dma_address);
        relocate::<CnaDcompAddr0>(&mut commands, buf_weights.dma_address);
        relocate::<DpuRdmaBsBaseAddr>(&mut commands, buf_bias.dma_address);
        relocate::<DpuDstBaseAddr>(&mut commands, buf_output.dma_address);

        let command_bytes = commands.len() * mem::size_of::<u64>();
        let buf_commands = Buffer::new(fd, page_aligned_size(command_bytes), &file);
        ptr::write_bytes(buf_commands.host_ptr, 0, buf_commands.size);
        let command_words =
            std::slice::from_raw_parts_mut(buf_commands.host_ptr as *mut u64, commands.len());
        for (destination, command) in command_words.iter_mut().zip(&commands) {
            *destination = command.0;
        }

        fini_bo(fd, buf_input.handle).expect("failed to sync input BO for the NPU");
        fini_bo(fd, buf_weights.handle).expect("failed to sync weight BO for the NPU");
        fini_bo(fd, buf_bias.handle).expect("failed to sync bias BO for the NPU");
        fini_bo(fd, buf_output.handle).expect("failed to sync output BO for the NPU");
        fini_bo(fd, buf_commands.handle).expect("failed to sync regcmd BO for the NPU");

        let input_handles = [
            buf_commands.handle,
            buf_input.handle,
            buf_weights.handle,
            buf_bias.handle,
        ];
        let output_handles = [buf_output.handle];
        submit(
            fd,
            buf_commands.dma_address,
            commands.len() as u32,
            &input_handles,
            &output_handles,
        )
        .unwrap_or_else(|error| panic!("{kernels:?} convolution SUBMIT ioctl failed: {error}"));

        prep_bo(fd, buf_output.handle, 2_000_000_000).unwrap_or_else(|error| {
            panic!("{kernels:?} convolution did not complete within two seconds: {error}")
        });

        let raw = std::slice::from_raw_parts(buf_output.host_ptr, OUTPUT_BYTES);
        let mut output = Vec::with_capacity(WIDTH * HEIGHT * OUTPUT_CHANNELS);
        for pixel in 0..WIDTH * HEIGHT {
            for channel in 0..OUTPUT_CHANNELS {
                let offset = pixel * FEATURE_ATOM_BYTES + channel * FP16_BYTES;
                output.push(f16_to_f32(u16::from_le_bytes([
                    raw[offset],
                    raw[offset + 1],
                ])));
            }
        }

        close_bo(fd, buf_input.handle).expect("failed to close input BO");
        close_bo(fd, buf_weights.handle).expect("failed to close weight BO");
        close_bo(fd, buf_bias.handle).expect("failed to close bias BO");
        close_bo(fd, buf_output.handle).expect("failed to close output BO");
        close_bo(fd, buf_commands.handle).expect("failed to close regcmd BO");

        output
    }
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn vendor_reference_convs_run_on_npu() {
    for kernels in [[1, 1], [3, 3]] {
        let actual = run_vendor_reference_conv(kernels);
        let mut mismatch_count = 0;
        let mut first_mismatches = Vec::new();

        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let expected = expected_output(kernels, y, x);
                for channel in 0..OUTPUT_CHANNELS {
                    let index = (y * WIDTH + x) * OUTPUT_CHANNELS + channel;
                    if actual[index] != expected {
                        mismatch_count += 1;
                        if first_mismatches.len() < 16 {
                            first_mismatches.push(format!(
                                "[{y}, {x}, {channel}]: expected {expected}, got {}",
                                actual[index]
                            ));
                        }
                    }
                }
            }
        }

        assert_eq!(
            mismatch_count,
            0,
            "{kernels:?} convolution had {mismatch_count} mismatches; first mismatches:\n{}",
            first_mismatches.join("\n")
        );
    }
}
