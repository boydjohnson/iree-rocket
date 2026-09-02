//! Exact fp16 depthwise convolution check on real RK3588 hardware.
//!
//! The fp16 depthwise path has never had a test that could fail on ordinary
//! data. What exists is:
//!
//!   * `conv_depthwise_hw.rs` -- an impulse: one 1.0 per channel, so only the
//!     window around it is nonzero and the result is the filter itself. It
//!     pins the coefficient layout and nothing about spatial addressing.
//!     Stride 1 only, one extent (32).
//!   * `conv_depthwise_stride_hw.rs` -- strides 2/3/4, but with *uniform*
//!     input and weights, so every output is a count of valid taps. Blind to
//!     which pixels were read as long as the right number were. Its own doc
//!     comment says as much.
//!
//! That is the same shape of gap `conv_depthwise_int8_exact_hw.rs` was
//! written to close on the int8 side, where the uniform probe stayed green
//! through two live bugs at once. This is its fp16 counterpart: every
//! (tap, channel) pair gets a distinct coefficient and the input varies in
//! y, x and channel, so neither a tap permutation nor a spatial or channel
//! one can cancel out, and it asserts per element rather than printing.
//!
//! Values are kept small on purpose. Inputs land in [-15, 15] and
//! coefficients in [-5, 5], so the largest accumulator nine taps can produce
//! is 675 -- well inside 2048, where fp16 still represents every integer
//! exactly. The comparison is therefore exact, tolerance zero, with no
//! rounding slack for a real error to hide in.
//!
//! # What the shapes are chosen to reach
//!
//! `pack_depthwise_to_rocket_weights` groups coefficients in
//! `DEPTHWISE_GROUP_BYTES` = 64-byte runs, which at fp16 is 32 channels per
//! group -- the int8 bug that motivated the int8 test was exactly a
//! mis-grouping at its own 64-channel boundary. Feature atoms hold 8 fp16
//! channels, so channel counts that are not a multiple of 8 exercise the
//! padding. The original probes use `padding = [0, 0]`; the MobileNetV2
//! cases below use its real 3x3 SAME geometry, `padding = [1, 1]`.

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    conv::{Buffers, ConvPlan, Kernels, Shape},
    device::{Buffer, JobDesc, close_bo, fini_bo, prep_bo, submit_jobs},
    tensor_layout::pack_depthwise_to_rocket_weights,
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const PAGE_BYTES: usize = 4096;
const FEATURE_ATOM_BYTES: usize = 16;
const CHANNELS_PER_ATOM: usize = 8;
const FP16_BYTES: usize = 2;
const KERNEL: Kernels = [3, 3];
const SENTINEL: u8 = 0xa5;

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
            (sign << 31) | (((exponent + 127 - 15) as u32) << 23) | ((mantissa & 0x3ff) << 13)
        }
        _ => (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13),
    };
    f32::from_bits(word)
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

/// Varies in all three axes, so a spatial or channel permutation shows up.
fn input_at(y: usize, x: usize, c: usize) -> f32 {
    (((y * 31 + x * 13 + c * 7) % 31) as i32 - 15) as f32
}

/// Distinct per (tap, channel) in both axes, so a tap permutation and a
/// channel permutation each show up.
fn weight_at(ky: usize, kx: usize, c: usize) -> f32 {
    (((ky * KERNEL[1] + kx) * 5 + c * 2) % 11) as i32 as f32 - 5.0
}

/// Host-side: the coefficient grouping each tested channel count produces.
///
/// `pack_depthwise_to_rocket_weights` splits `padded_channels` into 32-channel
/// groups at fp16 and gives the remainder a narrower tail group. Its doc
/// records validation at Cin 128, 256 and 144 -- all of which have at least
/// one *full* group. Everything below 32 is tail-only, which is a different
/// code path in the same function.
#[test]
fn fp16_depthwise_coefficient_grouping() {
    const GROUP_CHANNELS: usize = 32; // DEPTHWISE_GROUP_BYTES / FP16_BYTES
    for channels in [8usize, 12, 20, 24, 32, 40, 64, 96, 144, 192, 288] {
        let shape = Shape::with_out_channels(34, 34, 1, channels as u32, channels as u32)
            .with_padding([0, 0])
            .with_depthwise();
        let padded = shape.weight_bytes(KERNEL) as usize / (KERNEL[0] * KERNEL[1] * FP16_BYTES);
        let full = padded / GROUP_CHANNELS;
        let tail = padded - full * GROUP_CHANNELS;
        println!(
            "  Cin {channels:>3}: padded_channels {padded:>3}, full groups {full}, tail width {tail}"
        );
    }
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn fp16_depthwise_exact_within_one_coefficient_group() {
    for channels in [8, 24] {
        check_fp16_depthwise(channels, 34, 34, 1);
    }
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn fp16_depthwise_exact_across_coefficient_groups() {
    // 32 fp16 channels fill one 64-byte coefficient group; these span two,
    // three and four of them.
    for channels in [32, 40, 64, 96] {
        check_fp16_depthwise(channels, 34, 34, 1);
    }
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn fp16_depthwise_exact_at_partial_channel_atoms() {
    // Not multiples of 8, so the last feature atom is partly padding.
    for channels in [12, 20, 36] {
        check_fp16_depthwise(channels, 34, 34, 1);
    }
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn fp16_depthwise_exact_at_stride_two() {
    // MobileNetV2's own stride-2 depthwise geometries, the ones a compiled
    // model dispatches.
    for (channels, extent) in [(144, 113), (192, 57), (288, 29)] {
        check_fp16_depthwise(channels, extent, extent, 2);
    }
}

// These ten signatures account for every one of MobileNetV2's 17 depthwise
// placements that currently stay on CPU. Repeated model placements have the
// same isolated hardware signature, so each is measured once here. Unlike the
// older no-padding probes above, these use the model's 3x3 SAME geometry:
// logical input extent and a leading one-pixel hardware pad.
//
// The wide cases deliberately opt into Shape's characterization-only escape
// hatch. No compiled path sets it. A passing isolated case stays an ignored
// board regression until the model-level interaction is understood. The one
// failure below is behind `hardware-characterization`: C1344 needs 13 CBUF
// weight banks for a 3x3 kernel, above the 11-bank hardware ceiling.
macro_rules! mobilenetv2_depthwise_case {
    ($name:ident, $channels:expr, $extent:expr, $stride:expr) => {
        #[test]
        #[ignore = "needs /dev/accel/accel0 -- characterizes a CPU-routed fp16 MobileNetV2 depthwise shape"]
        fn $name() {
            unsafe { std::env::set_var("ROCKET_ALLOW_UNBACKED_CHANNELS", "1") };
            check_fp16_depthwise_with_padding($channels, $extent, $extent, $stride, [1, 1]);
        }
    };
}

// 112x112: one stride-1 C48 and one stride-2 C144 placement.
mobilenetv2_depthwise_case!(fp16_mobilenetv2_depthwise_112_c48_s1, 48, 112, 1);
mobilenetv2_depthwise_case!(fp16_mobilenetv2_depthwise_112_c144_s2, 144, 112, 2);
// 56x56: one stride-1 and one stride-2 C192 placement.
mobilenetv2_depthwise_case!(fp16_mobilenetv2_depthwise_56_c192_s1, 192, 56, 1);
mobilenetv2_depthwise_case!(fp16_mobilenetv2_depthwise_56_c192_s2, 192, 56, 2);
// 28x28: two stride-1 and one stride-2 C288 placements.
mobilenetv2_depthwise_case!(fp16_mobilenetv2_depthwise_28_c288_s1, 288, 28, 1);
mobilenetv2_depthwise_case!(fp16_mobilenetv2_depthwise_28_c288_s2, 288, 28, 2);
// 14x14: six stride-1 (C528 x4, C816 x2) and one stride-2 C816 placement.
mobilenetv2_depthwise_case!(fp16_mobilenetv2_depthwise_14_c528_s1, 528, 14, 1);
mobilenetv2_depthwise_case!(fp16_mobilenetv2_depthwise_14_c816_s1, 816, 14, 1);
mobilenetv2_depthwise_case!(fp16_mobilenetv2_depthwise_14_c816_s2, 816, 14, 2);
// 7x7: three stride-1 C1344 placements.
#[cfg(feature = "hardware-characterization")]
mobilenetv2_depthwise_case!(fp16_mobilenetv2_depthwise_7_c1344_s1, 1344, 7, 1);

fn check_fp16_depthwise(channels: usize, width: usize, height: usize, stride: u32) {
    check_fp16_depthwise_with_padding(channels, width, height, stride, [0, 0]);
}

fn check_fp16_depthwise_with_padding(
    channels: usize,
    width: usize,
    height: usize,
    stride: u32,
    padding: [usize; 2],
) {
    let shape = Shape::with_out_channels(
        width as u32,
        height as u32,
        stride,
        channels as u32,
        channels as u32,
    )
    .with_padding(padding)
    .with_depthwise();
    let (kh, kw) = (KERNEL[0], KERNEL[1]);
    let ow = shape.output_width(KERNEL) as usize;
    let oh = shape.output_height(KERNEL) as usize;

    let weight_bytes = shape.weight_bytes(KERNEL) as usize;
    let padded_channels = weight_bytes / (kh * kw * FP16_BYTES);
    let in_surfaces = (shape.weight_channels() as usize).div_ceil(CHANNELS_PER_ATOM);
    let out_surfaces = (shape.padded_out_channels() as usize).div_ceil(CHANNELS_PER_ATOM);
    let label = format!(
        "Cin {channels}, {width}x{height}, stride {stride}, padding {}x{}",
        padding[0], padding[1]
    );

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let input = Buffer::new(
            fd,
            page_aligned_size(in_surfaces * width * height * FEATURE_ATOM_BYTES),
            &file,
        );
        ptr::write_bytes(input.host_ptr, 0, input.size);
        for c in 0..channels {
            for y in 0..height {
                for x in 0..width {
                    let offset = (c / CHANNELS_PER_ATOM) * width * height * FEATURE_ATOM_BYTES
                        + (y * width + x) * FEATURE_ATOM_BYTES
                        + (c % CHANNELS_PER_ATOM) * FP16_BYTES;
                    ptr::write(
                        input.host_ptr.add(offset) as *mut u16,
                        f32_to_f16(input_at(y, x, c)),
                    );
                }
            }
        }

        // Dense [c][ky][kx] -- the order the compiler's HWC -> CHW transpose
        // hands the driver -- then packed by the routine under test.
        let mut dense = vec![0u8; channels * kh * kw * FP16_BYTES];
        for c in 0..channels {
            for ky in 0..kh {
                for kx in 0..kw {
                    let index = ((c * kh + ky) * kw + kx) * FP16_BYTES;
                    let bits = f32_to_f16(weight_at(ky, kx, c));
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
        .unwrap_or_else(|error| panic!("{label}: weight packing failed: {error}"));
        let weights = Buffer::new(fd, page_aligned_size(weight_bytes), &file);
        ptr::copy_nonoverlapping(packed.as_ptr(), weights.host_ptr, packed.len());

        // Zero bias, in the fp16 BRDMA format: plain fp16 values, so an
        // all-zero buffer *is* a zero bias. Deliberately not
        // `write_bs_buffer`/`BsEntry` -- that writes the quantized
        // bias/scale/shift triple the int8 path uses
        // (`BRDMA_DATA_USE_QUANTIZED`), and feeding it to an fp16 dispatch
        // corrupts a subset of channels rather than failing outright.
        let bs = Buffer::new(
            fd,
            page_aligned_size(shape.bs_buffer_bytes().max(PAGE_BYTES)),
            &file,
        );
        ptr::write_bytes(bs.host_ptr, 0, bs.size);

        let output_bytes = page_aligned_size(out_surfaces * ow * oh * FEATURE_ATOM_BYTES);
        let output = Buffer::new(fd, output_bytes, &file);
        ptr::write_bytes(output.host_ptr, SENTINEL, output.size);

        // Driven through ConvPlan, not Tile::whole: a real geometry does not
        // fit one tile, and Tile::whole would silently over-commit the CBUF
        // rather than split. This is also what the driver does, so the tiling
        // under test is the tiling a compiled model gets.
        let plan = ConvPlan::new(shape, KERNEL);
        let programs = plan.programs_with_buffers(Buffers {
            input: input.dma_address,
            weights: weights.dma_address,
            bias: bs.dma_address,
            output: output.dma_address,
        });
        let mut command_buffers = Vec::with_capacity(programs.len());
        for commands in &programs {
            let buffer = Buffer::new(
                fd,
                page_aligned_size(commands.len() * mem::size_of::<u64>()),
                &file,
            );
            ptr::write_bytes(buffer.host_ptr, 0, buffer.size);
            let words = std::slice::from_raw_parts_mut(buffer.host_ptr as *mut u64, commands.len());
            for (word, command) in words.iter_mut().zip(commands) {
                *word = command.0;
            }
            command_buffers.push((buffer, commands.len() as u32));
        }

        let data_handles = [input.handle, weights.handle, bs.handle, output.handle];
        for handle in data_handles {
            fini_bo(fd, handle).unwrap();
        }
        for (buffer, _) in &command_buffers {
            fini_bo(fd, buffer.handle).unwrap();
        }

        let tasks: Vec<[(u32, u32); 1]> = command_buffers
            .iter()
            .map(|(buffer, count)| [(buffer.dma_address, *count)])
            .collect();
        let in_handles: Vec<[u32; 4]> = command_buffers
            .iter()
            .map(|(buffer, _)| [buffer.handle, input.handle, weights.handle, bs.handle])
            .collect();
        let out_handles = [output.handle];
        let jobs: Vec<JobDesc<'_>> = tasks
            .iter()
            .zip(&in_handles)
            .map(|(tasks, in_handles)| JobDesc {
                tasks,
                in_handles,
                out_handles: &out_handles,
            })
            .collect();
        submit_jobs(fd, &jobs).unwrap_or_else(|error| {
            panic!("{label} ({} tile(s)): SUBMIT failed: {error}", jobs.len())
        });
        prep_bo(fd, output.handle, 5_000_000_000)
            .unwrap_or_else(|error| panic!("{label}: did not complete: {error}"));

        let raw = std::slice::from_raw_parts(output.host_ptr, output.size);
        let read = |oy: usize, ox: usize, c: usize| -> f32 {
            let offset = (c / CHANNELS_PER_ATOM) * ow * oh * FEATURE_ATOM_BYTES
                + (oy * ow + ox) * FEATURE_ATOM_BYTES
                + (c % CHANNELS_PER_ATOM) * FP16_BYTES;
            f16_to_f32(u16::from_le_bytes([raw[offset], raw[offset + 1]]))
        };

        let mut mismatches = 0usize;
        let mut first = None;
        for oy in 0..oh {
            for ox in 0..ow {
                for c in 0..channels {
                    let mut expected = 0.0f32;
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let iy = oy * stride as usize + ky;
                            let ix = ox * stride as usize + kx;
                            if let (Some(iy), Some(ix)) =
                                (iy.checked_sub(padding[0]), ix.checked_sub(padding[1]))
                                && iy < height
                                && ix < width
                            {
                                expected += input_at(iy, ix, c) * weight_at(ky, kx, c);
                            }
                        }
                    }
                    let actual = read(oy, ox, c);
                    if actual != expected {
                        mismatches += 1;
                        first.get_or_insert((oy, ox, c, expected, actual));
                    }
                }
            }
        }
        for handle in data_handles {
            let _ = close_bo(fd, handle);
        }
        for (buffer, _) in &command_buffers {
            let _ = close_bo(fd, buffer.handle);
        }
        if let Some((oy, ox, c, expected, actual)) = first {
            // A depthwise channel only ever mixes with itself, so a wrong
            // value that equals *another* channel's expected value is a
            // permutation, not arithmetic. Report which, because the two
            // have completely different causes.
            let want = |cc: usize| -> f32 {
                let mut sum = 0.0f32;
                for ky in 0..kh {
                    for kx in 0..kw {
                        let iy = oy * stride as usize + ky;
                        let ix = ox * stride as usize + kx;
                        if let (Some(iy), Some(ix)) =
                            (iy.checked_sub(padding[0]), ix.checked_sub(padding[1]))
                            && iy < height
                            && ix < width
                        {
                            sum += input_at(iy, ix, cc) * weight_at(ky, kx, cc);
                        }
                    }
                }
                sum
            };
            let mut mapping = Vec::new();
            for cc in 0..channels {
                let got = read(oy, ox, cc);
                if got != want(cc) {
                    let source: Vec<usize> = (0..channels).filter(|&s| want(s) == got).collect();
                    mapping.push(match source.as_slice() {
                        [only] => format!("{cc}<-{only}"),
                        [] => format!("{cc}<-?"),
                        _ => format!("{cc}<-amb"),
                    });
                }
            }
            panic!(
                "fp16 depthwise at {label}: {mismatches} of {} elements wrong; \
                 first at (y {oy}, x {ox}, c {c}): expected {expected}, got {actual}\n  \
                 at that pixel, wrong channels (dst<-src): {}",
                oh * ow * channels,
                mapping.join(" ")
            );
        }
    }
}
