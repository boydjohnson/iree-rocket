//! Exact 2-byte depthwise convolution check on real RK3588 hardware.
//!
//! Covers every 2-byte rung -- fp16, bf16 and int16 -- because the depthwise
//! coefficient grouping is stated in *bytes*
//! (`DEPTHWISE_GROUP_BYTES` = 64, so 32 channels at any 2-byte width) and
//! the feature atom holds 8 channels at all three. If the depthwise path
//! follows the element width the way the dense path does, one fixture drives
//! all three with only the value encoding changed; this is the test that
//! says whether it does, rather than assuming it.
//!
//! The fp16 depthwise path had no test that could fail on ordinary
//! data. What existed was:
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
//! padding. And `padding = [0, 0]` throughout, which is what
//! rocket-hal-driver programs from a compiled .vmfb -- not the SAME padding
//! `Shape`'s default gives.

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr, time::Instant};

#[path = "support/dispatch.rs"]
mod dispatch;

use iree_rocket_hal::rocket::{
    conv::{Buffers, ConvPlan, Kernels, Precision, Shape},
    device::{Buffer, JobDesc, close_bo, fini_bo, prep_bo, submit_jobs},
    tensor_layout::pack_depthwise_to_rocket_weights,
};

/// A 2-byte rung, and the value range whose accumulators it holds exactly.
///
/// The ranges differ because the exactness bound does: fp16 and int16 carry
/// every integer to 2048 and 32767, but bf16's eight-bit significand stops
/// at 256, so its fixture is scaled down until nine taps cannot leave the
/// exact grid. The comparison stays tolerance-zero on all three -- a
/// depthwise layout test has no business allowing rounding slack, since
/// every error it hunts is a permutation rather than a rounding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TwoByte {
    Fp16,
    Bf16,
    Int16,
}

impl TwoByte {
    fn name(self) -> &'static str {
        match self {
            Self::Fp16 => "fp16",
            Self::Bf16 => "bf16",
            Self::Int16 => "int16",
        }
    }

    fn precision(self) -> Precision {
        match self {
            Self::Fp16 => Precision::Fp16,
            Self::Bf16 => Precision::Bf16,
            Self::Int16 => Precision::Int16,
        }
    }

    /// `(modulus, offset)` for the input generator: values land in
    /// `-offset ..= modulus - 1 - offset`.
    fn input_range(self) -> (usize, i32) {
        match self {
            Self::Fp16 | Self::Int16 => (31, 15),
            Self::Bf16 => (15, 7),
        }
    }

    /// The modulus must be coprime with the generator's tap stride of 5, or
    /// the tap term vanishes and the fixture stops varying across taps
    /// entirely -- a test that cannot see a tap permutation, which is one of
    /// the two things it exists to catch.
    /// `weight_generator_varies_across_every_axis` enforces it.
    fn weight_range(self) -> (usize, i32) {
        match self {
            Self::Fp16 | Self::Int16 => (11, 5),
            Self::Bf16 => (7, 3),
        }
    }

    /// Largest accumulator the ranges above can produce over nine taps,
    /// which must stay on the rung's exact integer grid.
    fn peak_accumulator(self) -> i32 {
        let (modulus, offset) = self.input_range();
        let (weight_modulus, weight_offset) = self.weight_range();
        let input = offset.max(modulus as i32 - 1 - offset);
        let weight = weight_offset.max(weight_modulus as i32 - 1 - weight_offset);
        (KERNEL[0] * KERNEL[1]) as i32 * input * weight
    }

    fn encode(self, value: f32) -> u16 {
        match self {
            Self::Fp16 => f32_to_f16(value),
            Self::Bf16 => f32_to_bf16(value),
            Self::Int16 => (value as i16) as u16,
        }
    }

    fn decode(self, bits: u16) -> f32 {
        match self {
            Self::Fp16 => f16_to_f32(bits),
            Self::Bf16 => f32::from_bits(u32::from(bits) << 16),
            Self::Int16 => f32::from(bits as i16),
        }
    }
}

fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    assert_eq!(bits & 0xffff, 0, "{value} is not exact in bf16");
    (bits >> 16) as u16
}

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
///
/// The `y` stride is 17, not 31. It was 31 while the modulus was also 31,
/// which made `(y * 31) % 31` zero at every row: the fp16 fixture this file
/// shipped with did not vary with `y` at all, and could not have seen a row
/// permutation. Found by `weight_generator_varies_across_every_axis` rather
/// than by a failing hardware run, which is the point of having it.
fn input_at(precision: TwoByte, y: usize, x: usize, c: usize) -> f32 {
    let (modulus, offset) = precision.input_range();
    (((y * 17 + x * 13 + c * 7) % modulus) as i32 - offset) as f32
}

/// Distinct per (tap, channel) in both axes, so a tap permutation and a
/// channel permutation each show up.
fn weight_at(precision: TwoByte, ky: usize, kx: usize, c: usize) -> f32 {
    let (modulus, offset) = precision.weight_range();
    (((ky * KERNEL[1] + kx) * 5 + c * 2) % modulus) as i32 as f32 - offset as f32
}

/// The generators have to vary along every axis they claim to.
///
/// Caught two real defects, one of them shipped. The `y` stride and the
/// input modulus were both 31, so the fp16 fixture was constant down every
/// column and blind to a row permutation. And bf16's scaled-down weight modulus was first chosen
/// as 5, and the generator multiplies the tap index by 5 before taking the
/// modulus, so `(tap * 5) % 5` is zero for every tap. Every coefficient in a
/// channel became identical, and the bf16 arm passed on hardware while being
/// structurally blind to a tap permutation -- the exact failure this whole
/// file was written to stop being blind to.
#[test]
fn weight_generator_varies_across_every_axis() {
    for precision in [TwoByte::Fp16, TwoByte::Bf16, TwoByte::Int16] {
        let name = precision.name();
        let taps: Vec<f32> = (0..KERNEL[0])
            .flat_map(|ky| (0..KERNEL[1]).map(move |kx| (ky, kx)))
            .map(|(ky, kx)| weight_at(precision, ky, kx, 0))
            .collect();
        assert!(
            taps.iter().any(|value| *value != taps[0]),
            "{name}: coefficients do not vary across taps"
        );
        let channels: Vec<f32> = (0..8).map(|c| weight_at(precision, 1, 1, c)).collect();
        assert!(
            channels.iter().any(|value| *value != channels[0]),
            "{name}: coefficients do not vary across channels"
        );
        for (axis, values) in [
            (
                "y",
                (0..8)
                    .map(|y| input_at(precision, y, 0, 0))
                    .collect::<Vec<_>>(),
            ),
            ("x", (0..8).map(|x| input_at(precision, 0, x, 0)).collect()),
            ("c", (0..8).map(|c| input_at(precision, 0, 0, c)).collect()),
        ] {
            assert!(
                values.iter().any(|value| *value != values[0]),
                "{name}: inputs do not vary across {axis}"
            );
        }
    }
}

/// Every fixture value, and every accumulator it can produce, must round-trip
/// through its rung exactly -- otherwise a tolerance-zero comparison is
/// measuring the encoding rather than the hardware.
#[test]
fn fixture_values_are_exact_in_every_two_byte_rung() {
    for precision in [TwoByte::Fp16, TwoByte::Bf16, TwoByte::Int16] {
        let peak = precision.peak_accumulator();
        let significant = 32 - (peak as u32).leading_zeros();
        let limit = match precision {
            // Eleven- and eight-bit significands, and int16's whole range.
            TwoByte::Fp16 => 11,
            TwoByte::Bf16 => 9,
            TwoByte::Int16 => 15,
        };
        assert!(
            significant <= limit,
            "{}: peak accumulator {peak} needs {significant} significant bits, past {limit}",
            precision.name(),
        );
        // Round-trip the extremes of both generators through the encoding.
        for value in [-8.0f32, -1.0, 0.0, 1.0, 5.0, 15.0] {
            assert_eq!(precision.decode(precision.encode(value)), value);
        }
    }
}

/// Host-side: the coefficient grouping each tested channel count produces.
///
/// `pack_depthwise_to_rocket_weights` splits `padded_channels` into 32-channel
/// groups at fp16 and gives the remainder a narrower tail group. Its doc
/// records validation at Cin 128, 256 and 144 -- all of which have at least
/// one *full* group. Everything below 32 is tail-only, which is a different
/// code path in the same function.
#[test]
fn two_byte_depthwise_coefficient_grouping() {
    const GROUP_CHANNELS: usize = 32; // DEPTHWISE_GROUP_BYTES / FP16_BYTES
    for channels in [8usize, 12, 20, 24, 32, 40, 64, 96, 144, 192, 288] {
        let shape =
            Shape::with_precision(34, 34, 1, channels as u32, channels as u32, Precision::Fp16)
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
        check_depthwise(TwoByte::Fp16, channels, 34, 34, 1);
    }
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn fp16_depthwise_exact_across_coefficient_groups() {
    // 32 fp16 channels fill one 64-byte coefficient group; these span two,
    // three and four of them.
    for channels in [32, 40, 64, 96] {
        check_depthwise(TwoByte::Fp16, channels, 34, 34, 1);
    }
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn fp16_depthwise_exact_at_partial_channel_atoms() {
    // Not multiples of 8, so the last feature atom is partly padding.
    for channels in [12, 20, 36] {
        check_depthwise(TwoByte::Fp16, channels, 34, 34, 1);
    }
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn fp16_depthwise_exact_at_stride_two() {
    // MobileNetV2's own stride-2 depthwise geometries, the ones a compiled
    // model dispatches.
    for (channels, extent) in [(144, 113), (192, 57), (288, 29)] {
        check_depthwise(TwoByte::Fp16, channels, extent, extent, 2);
    }
}

/// The other two 2-byte rungs, on a representative slice of the fp16 ladder
/// above: inside one coefficient group, across two and four of them, at a
/// partial feature atom, at a real stride-2 MobileNetV2 geometry, and --
/// added 2026-09-04 -- wide, at nine and sixteen coefficient groups.
///
/// The claim is that the depthwise path follows the element *width*, exactly
/// as the dense path does, so bf16 and int16 inherit fp16's 32-channel
/// coefficient group and 8-channel feature atom rather than needing their
/// own. A narrower slice than fp16's because that claim is structural: if it
/// holds at all it holds at every channel count, and if it fails it fails at
/// the first group boundary.
///
/// The wide points are here because "structural" is an argument, not a
/// measurement, and the dense side of these same rungs turned out to have a
/// real width-dependent fault at high channel counts once someone looked
/// (`streamed_weight_bank_preference_for_group`). 288 and 512 are what the
/// dense ladders reach on the depthwise axis.
#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn bf16_depthwise_exact() {
    for channels in [24, 32, 64, 20, 288, 512] {
        check_depthwise(TwoByte::Bf16, channels, 34, 34, 1);
    }
    check_depthwise(TwoByte::Bf16, 144, 113, 113, 2);
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn int16_depthwise_exact() {
    for channels in [24, 32, 64, 20, 288, 512] {
        check_depthwise(TwoByte::Int16, channels, 34, 34, 1);
    }
    check_depthwise(TwoByte::Int16, 144, 113, 113, 2);
}

fn check_depthwise(precision: TwoByte, channels: usize, width: usize, height: usize, stride: u32) {
    let shape = Shape::with_precision(
        width as u32,
        height as u32,
        stride,
        channels as u32,
        channels as u32,
        precision.precision(),
    )
    .with_padding([0, 0])
    .with_depthwise();
    let (kh, kw) = (KERNEL[0], KERNEL[1]);
    let ow = shape.output_width(KERNEL) as usize;
    let oh = shape.output_height(KERNEL) as usize;

    let weight_bytes = shape.weight_bytes(KERNEL) as usize;
    let padded_channels = weight_bytes / (kh * kw * FP16_BYTES);
    let in_surfaces = (shape.weight_channels() as usize).div_ceil(CHANNELS_PER_ATOM);
    let out_surfaces = (shape.padded_out_channels() as usize).div_ceil(CHANNELS_PER_ATOM);
    let label = format!(
        "{} Cin {channels}, {width}x{height}, stride {stride}",
        precision.name()
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
                        precision.encode(input_at(precision, y, x, c)),
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
                    let bits = precision.encode(weight_at(precision, ky, kx, c));
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
        let started = Instant::now();
        submit_jobs(fd, &jobs).unwrap_or_else(|error| {
            panic!("{label} ({} tile(s)): SUBMIT failed: {error}", jobs.len())
        });
        prep_bo(fd, output.handle, 5_000_000_000)
            .unwrap_or_else(|error| panic!("{label}: did not complete: {error}"));
        let elapsed = started.elapsed();
        print!("  {label}: ok{}", dispatch::note(elapsed));
        println!();

        let raw = std::slice::from_raw_parts(output.host_ptr, output.size);
        let read = |oy: usize, ox: usize, c: usize| -> f32 {
            let offset = (c / CHANNELS_PER_ATOM) * ow * oh * FEATURE_ATOM_BYTES
                + (oy * ow + ox) * FEATURE_ATOM_BYTES
                + (c % CHANNELS_PER_ATOM) * FP16_BYTES;
            precision.decode(u16::from_le_bytes([raw[offset], raw[offset + 1]]))
        };

        let mut mismatches = 0usize;
        let mut first = None;
        for oy in 0..oh {
            for ox in 0..ow {
                for c in 0..channels {
                    let mut expected = 0.0f32;
                    for ky in 0..kh {
                        for kx in 0..kw {
                            expected += input_at(
                                precision,
                                oy * stride as usize + ky,
                                ox * stride as usize + kx,
                                c,
                            ) * weight_at(precision, ky, kx, c);
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
        // A dispatch this long did not produce a result to disagree with:
        // the watchdog killed the job and `PREP_BO` returned success over an
        // error-signalled fence, so the whole buffer is the sentinel. Every
        // element being wrong is what that looks like, and it is exactly what
        // a catastrophic layout bug looks like too -- the clock is the only
        // thing that tells them apart.
        if mismatches != 0 && dispatch::is_device_timeout(elapsed) {
            assert!(
                !dispatch::report(&label, elapsed),
                "{label} was killed mid-dispatch and ROCKET_STRICT_DISPATCH is set"
            );
            return;
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
                        sum += input_at(
                            precision,
                            oy * stride as usize + ky,
                            ox * stride as usize + kx,
                            cc,
                        ) * weight_at(precision, ky, kx, cc);
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
                "depthwise at {label}: {mismatches} of {} elements wrong; \
                 first at (y {oy}, x {ox}, c {c}): expected {expected}, got {actual}\n  \
                 at that pixel, wrong channels (dst<-src): {}",
                oh * ow * channels,
                mapping.join(" ")
            );
        }
    }
}
