//! End-to-end depthwise convolution on real hardware, through
//! `tensor_layout::pack_depthwise_to_rocket_weights`.
//!
//! `conv_depthwise_probe_hw.rs` derived the coefficient layout by one-hot
//! probing; this checks that the packer built from that result actually
//! produces correct output for an ordinary filter.
//!
//! # The impulse trick
//!
//! The input is zero everywhere except a single 1.0 at one pixel, on every
//! channel. With `padding = 1` a 3x3 convolution then gives
//!
//! `output[c][y][x] = sum over (ky, kx) of input[c][y+ky-1][x+kx-1] * w[c][ky][kx]`
//!
//! whose only surviving term is the one landing on the impulse, at
//! `ky = IMPULSE + 1 - y` and `kx = IMPULSE + 1 - x`. So the 3x3 output
//! window centred on the impulse *is the filter, point-reflected*, and every
//! other output pixel is zero.
//!
//! That makes the test maximally discriminating for a packing bug: each
//! coefficient is read back individually and separately identified, so a tap
//! transposition, a channel mix-up, or a wrong row stride each show up as a
//! specific wrong value rather than as a plausible-looking sum. Giving every
//! coefficient a distinct value is what makes that work -- a filter of all
//! ones would pass under any permutation of its own taps.
//!
//! # Why Cin 12
//!
//! 12 channels pad to 16, so the packed row stride and the real channel
//! count differ. That is the case that distinguishes the two candidate
//! layouts, and the one a Cin-8 test cannot speak for. Cin 8 runs too, as
//! the case where they coincide.

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    conv::{Buffers, Kernels, Shape, Tile, conv_2d_tile, relocate},
    device::{Buffer, JobDesc, close_bo, fini_bo, prep_bo, submit_jobs},
    tensor_layout::pack_depthwise_to_rocket_weights,
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const FP16_BYTES: usize = 2;
const FEATURE_ATOM_BYTES: usize = 16;
const CHANNELS_PER_ATOM: usize = 8;
const PAGE_BYTES: usize = 4096;

const EXTENT: usize = 32;
const IMPULSE: usize = 16;
const KERNEL: Kernels = [3, 3];

fn page_aligned_size(size: usize) -> usize {
    size.div_ceil(PAGE_BYTES) * PAGE_BYTES
}

/// Exact fp16 encoding. Asserts exactness rather than rounding: every value
/// this test uses is a small integer, and a silent rounding would weaken the
/// comparison it exists to make.
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

/// The coefficient for one `(channel, ky, kx)`, distinct across the whole
/// filter so a misplaced byte names where it belongs.
fn coefficient(channel: usize, ky: usize, kx: usize) -> f32 {
    (channel * KERNEL[0] * KERNEL[1] + ky * KERNEL[1] + kx + 1) as f32
}

fn run(channels: u32) {
    let shape = Shape::with_out_channels(EXTENT as u32, EXTENT as u32, 1, channels, channels)
        .with_depthwise();
    let channels = channels as usize;
    let (kh, kw) = (KERNEL[0], KERNEL[1]);

    // The stride the packer needs, taken from the register program's own
    // coefficient footprint rather than recomputed here.
    let weight_bytes = shape.weight_bytes(KERNEL) as usize;
    let padded_channels = weight_bytes / (kh * kw * FP16_BYTES);
    assert!(
        padded_channels >= channels,
        "padded {padded_channels} < real {channels}"
    );

    let in_surfaces = (shape.weight_channels() as usize).div_ceil(CHANNELS_PER_ATOM);
    let out_surfaces = (shape.padded_out_channels() as usize).div_ceil(CHANNELS_PER_ATOM);

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        // Input: a single 1.0 per channel, at (IMPULSE, IMPULSE).
        let input = Buffer::new(
            fd,
            page_aligned_size(in_surfaces * EXTENT * EXTENT * FEATURE_ATOM_BYTES),
            &file,
        );
        ptr::write_bytes(input.host_ptr, 0, input.size);
        for channel in 0..channels {
            let offset = (channel / CHANNELS_PER_ATOM) * EXTENT * EXTENT * FEATURE_ATOM_BYTES
                + (IMPULSE * EXTENT + IMPULSE) * FEATURE_ATOM_BYTES
                + (channel % CHANNELS_PER_ATOM) * FP16_BYTES;
            ptr::write(input.host_ptr.add(offset) as *mut u16, f32_to_f16(1.0));
        }

        // Weights: dense [channels][kh][kw], then packed by the routine
        // under test.
        let mut dense = vec![0u8; channels * kh * kw * FP16_BYTES];
        for channel in 0..channels {
            for ky in 0..kh {
                for kx in 0..kw {
                    let index = ((channel * kh + ky) * kw + kx) * FP16_BYTES;
                    let bits = f32_to_f16(coefficient(channel, ky, kx));
                    dense[index..index + FP16_BYTES].copy_from_slice(&bits.to_le_bytes());
                }
            }
        }
        let mut packed = vec![0u8; weight_bytes];
        pack_depthwise_to_rocket_weights(
            &dense,
            kh,
            kw,
            channels,
            padded_channels,
            FP16_BYTES,
            &mut packed,
        )
        .expect("depthwise packing failed");

        let weights = Buffer::new(fd, page_aligned_size(weight_bytes), &file);
        ptr::write_bytes(weights.host_ptr, 0, weights.size);
        ptr::copy_nonoverlapping(packed.as_ptr(), weights.host_ptr, packed.len());

        let bias = Buffer::new(fd, PAGE_BYTES, &file);
        ptr::write_bytes(bias.host_ptr, 0, bias.size);
        let output = Buffer::new(
            fd,
            page_aligned_size(out_surfaces * EXTENT * EXTENT * FEATURE_ATOM_BYTES),
            &file,
        );
        ptr::write_bytes(output.host_ptr, 0, output.size);

        let mut words = conv_2d_tile(shape, KERNEL, &Tile::whole(shape, KERNEL));
        relocate(
            &mut words,
            Buffers {
                input: input.dma_address,
                weights: weights.dma_address,
                bias: bias.dma_address,
                output: output.dma_address,
            },
        );
        let commands = Buffer::new(
            fd,
            page_aligned_size(words.len() * mem::size_of::<u64>()),
            &file,
        );
        ptr::write_bytes(commands.host_ptr, 0, commands.size);
        let slots = std::slice::from_raw_parts_mut(commands.host_ptr as *mut u64, words.len());
        for (destination, command) in slots.iter_mut().zip(&words) {
            *destination = command.0;
        }

        for handle in [
            input.handle,
            weights.handle,
            bias.handle,
            output.handle,
            commands.handle,
        ] {
            fini_bo(fd, handle).expect("failed to sync BO for the NPU");
        }

        let tasks = [(commands.dma_address, words.len() as u32)];
        let in_handles = [commands.handle, input.handle, weights.handle, bias.handle];
        let out_handles = [output.handle];
        submit_jobs(
            fd,
            &[JobDesc {
                tasks: &tasks,
                in_handles: &in_handles,
                out_handles: &out_handles,
            }],
        )
        .unwrap_or_else(|error| panic!("Cin {channels}: SUBMIT failed: {error}"));
        prep_bo(fd, output.handle, 5_000_000_000)
            .unwrap_or_else(|error| panic!("Cin {channels}: did not complete: {error}"));

        let raw = std::slice::from_raw_parts(
            output.host_ptr,
            out_surfaces * EXTENT * EXTENT * FEATURE_ATOM_BYTES,
        );
        let at = |channel: usize, y: usize, x: usize| -> f32 {
            let offset = (channel / CHANNELS_PER_ATOM) * EXTENT * EXTENT * FEATURE_ATOM_BYTES
                + (y * EXTENT + x) * FEATURE_ATOM_BYTES
                + (channel % CHANNELS_PER_ATOM) * FP16_BYTES;
            f16_to_f32(u16::from_le_bytes([raw[offset], raw[offset + 1]]))
        };

        let mut mismatches = Vec::new();
        for channel in 0..channels {
            for y in 0..EXTENT {
                for x in 0..EXTENT {
                    // Only the window around the impulse survives, and there
                    // the filter appears point-reflected.
                    let want = if y + 1 >= IMPULSE
                        && y <= IMPULSE + 1
                        && x + 1 >= IMPULSE
                        && x <= IMPULSE + 1
                    {
                        coefficient(channel, IMPULSE + 1 - y, IMPULSE + 1 - x)
                    } else {
                        0.0
                    };
                    let got = at(channel, y, x);
                    if got != want && mismatches.len() < 12 {
                        mismatches.push(format!("[c{channel}, {y}, {x}] want {want} got {got}"));
                    }
                }
            }
        }

        for handle in [
            input.handle,
            weights.handle,
            bias.handle,
            output.handle,
            commands.handle,
        ] {
            let _ = close_bo(fd, handle);
        }

        assert!(
            mismatches.is_empty(),
            "Cin {channels} (packed stride {padded_channels}) depthwise output is wrong:\n  {}",
            mismatches.join("\n  ")
        );
    }
}

/// The case that distinguishes the two candidate layouts: 12 channels pad to
/// a packed row stride of 16, so a packer using the real channel count puts
/// every tap after the first in the wrong place.
#[test]
#[ignore = "requires a real RK3588 NPU at /dev/accel/accel0"]
fn depthwise_with_channel_padding_reproduces_its_filter() {
    run(12);
}

/// The case where the real and padded counts coincide, so it passes under
/// either stride. Kept because it is the shape the layout was first read at.
#[test]
#[ignore = "requires a real RK3588 NPU at /dev/accel/accel0"]
fn depthwise_at_a_whole_atom_reproduces_its_filter() {
    run(8);
}
