//! Requantized int8 depthwise convolution on real RK3588 hardware.
//!
//! The dense requantized path -- `Precision::Int8`, where the DPU's own BS and
//! out-convert stages run and the dispatch returns `i8` -- made MobileNetV2
//! 1.54x faster than a like-for-like CPU build by removing the `i32`
//! activation tensor and its CPU epilogue (ISSUES.md). Every *depthwise*
//! convolution is still on `Precision::Int8Accumulator`, so all 13 of
//! MobileNetV2's depthwise layers still materialize `i32` and requantize on
//! the host: that is the largest single category of CPU dispatch left in the
//! model, one `elementwise_i32xi32xi32xi8` per layer.
//!
//! `Shape::with_depthwise` already accepts `Precision::Int8` (see
//! `depthwise_accepts_only_the_measured_element_widths`), so the combination
//! is expressible. Nothing has ever run it. This is that measurement, and
//! nothing downstream -- no executable target, no matcher, no fusion -- is
//! worth building until it passes.
//!
//! It is deliberately the requantized twin of `conv_depthwise_int8_exact_hw`,
//! same geometry and same coefficient scheme, so a failure here against a pass
//! there isolates the requantization stages rather than the depthwise datapath.
//! That file's two traps are inherited and still matter: every (tap, channel)
//! pair gets a distinct coefficient, so neither a tap permutation nor a channel
//! permutation can hide, and padding is `[0, 0]` because that is what
//! rocket-hal-driver programs from a compiled `.vmfb`.
//!
//! **The output layout is the thing most likely to be wrong**, and it is
//! checked rather than assumed. `Shape::output_atom_bytes` returns the doubled
//! 256-byte atom only for a depthwise convolution that *writes accumulators*;
//! a requantized depthwise writes `i8`, so it should fall through to the
//! ordinary channel-block size. If the hardware instead keeps the doubled
//! stride here, the readback comes back permuted and this test says so.
//!
//! Cross-compile for aarch64, copy to the board, run the ignored tests:
//!
//! ```text
//! CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!   cargo test -p iree-rocket-hal --release \
//!     --target aarch64-unknown-linux-gnu --test conv_depthwise_requant_hw --no-run
//!
//! ./conv_depthwise_requant_hw-<hash> --ignored --nocapture --test-threads=1
//! ```

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    conv::{
        Buffers, Multiplier, Precision, Quantization, Shape, Tile, conv_2d_tile,
        pack_int8_bias_to_bs, relocate,
    },
    device::{Buffer, JobDesc, close_bo, fini_bo, prep_bo, submit_jobs},
    tensor_layout::{pack_depthwise_to_rocket_weights, pack_nhwc_to_nc1hwc2},
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const PAGE_BYTES: usize = 4096;
const W: usize = 34;
const H: usize = 34;
const OW: usize = 32;
const OH: usize = 32;
const KERNEL: usize = 3;
const SENTINEL: u8 = 0xa5;

/// The requantization ratio the accumulator is multiplied by. Chosen so the
/// accumulators this input and filter produce land across most of the signed
/// byte's range without clipping at either end -- a case that saturates
/// everywhere would pass while computing almost nothing.
const RATIO: f64 = 1.0 / 96.0;
const OUTPUT_ZERO_POINT: i32 = -6;

fn page_aligned_size(size: usize) -> usize {
    size.div_ceil(PAGE_BYTES) * PAGE_BYTES
}

fn input_at(y: usize, x: usize, c: usize) -> i8 {
    (((y * 31 + x * 13 + c * 7) % 61) as i32 - 30) as i8
}

fn weight_at(ky: usize, kx: usize, c: usize) -> i8 {
    (((ky * KERNEL + kx) * 5 + c * 2) % 11) as i8 - 5
}

/// Per-channel bias, in accumulator units. Nonzero and channel-varying: a zero
/// bias would leave the BS plane untested, which is half of what the
/// requantized path turns on.
fn bias_at(c: usize) -> i32 {
    (c as i32 % 17) * 37 - 300
}

/// The device's own requantization, exactly:
/// `clamp(floor((acc * scale + 2^(shift-1)) >> shift) + zero_point)`, using
/// the encoded mantissa and shift rather than an `f32` multiply.
///
/// Fitted on all 131072 elements of the dense 1x1 case and exact there (see
/// `requantized-int8-conv-path`), so this is a bit-exact oracle rather than a
/// tolerance. An `f32` reference would differ at every exact tie and force a
/// tolerance that hides real errors.
fn device_requantize(accumulator: i64, multiplier: &Multiplier) -> i8 {
    let scaled = accumulator * i64::from(multiplier.scale);
    let rounding = 1i64 << (multiplier.shift - 1);
    let shifted = (scaled + rounding) >> multiplier.shift;
    let offset = shifted + i64::from(OUTPUT_ZERO_POINT);
    offset.clamp(-128, 127) as i8
}

fn check_requantized_depthwise(channels: usize) {
    let multiplier = Multiplier::from_ratio(RATIO);
    let precision = Precision::Int8(Quantization {
        input_zero_point: 0,
        output_zero_point: OUTPUT_ZERO_POINT,
        weight_zero_point: 0,
        // Held at 1.0 so `pack_int8_bias_to_bs` passes the accumulator-unit
        // bias through untouched and `multiplier` carries the whole ratio --
        // the same division of labour the compiler's requantized target uses.
        input_scale: 1.0,
        weights_scale: 1.0,
        multiplier,
    });
    let shape = Shape::with_precision(
        W as u32,
        H as u32,
        1,
        channels as u32,
        channels as u32,
        precision,
    )
    .with_padding([0, 0])
    .with_depthwise();
    let kernels = [KERNEL, KERNEL];
    assert_eq!(shape.output_width(kernels) as usize, OW);
    assert_eq!(shape.output_height(kernels) as usize, OH);

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .unwrap();
    let fd = file.as_raw_fd();

    unsafe {
        let input_bytes = W * H * channels;
        let input = Buffer::new(fd, page_aligned_size(input_bytes), &file);
        let mut dense_input = vec![0u8; input_bytes];
        for y in 0..H {
            for x in 0..W {
                for c in 0..channels {
                    dense_input[(y * W + x) * channels + c] = input_at(y, x, c) as u8;
                }
            }
        }
        let mut packed_input = vec![0u8; input_bytes];
        pack_nhwc_to_nc1hwc2(&dense_input, W * H, channels, &mut packed_input).unwrap();
        ptr::copy_nonoverlapping(packed_input.as_ptr(), input.host_ptr, packed_input.len());

        let weight_bytes = shape.weight_bytes(kernels) as usize;
        let weights = Buffer::new(fd, page_aligned_size(weight_bytes), &file);
        let mut dense_weights = vec![0u8; channels * KERNEL * KERNEL];
        for c in 0..channels {
            for ky in 0..KERNEL {
                for kx in 0..KERNEL {
                    dense_weights[(c * KERNEL + ky) * KERNEL + kx] = weight_at(ky, kx, c) as u8;
                }
            }
        }
        let mut packed_weights = vec![0u8; weight_bytes];
        pack_depthwise_to_rocket_weights(
            &dense_weights,
            KERNEL,
            KERNEL,
            channels,
            shape.depthwise_padded_channels() as usize,
            1,
            &mut packed_weights,
        )
        .unwrap();
        ptr::copy_nonoverlapping(
            packed_weights.as_ptr(),
            weights.host_ptr,
            packed_weights.len(),
        );

        // The BS plane carries the real per-channel bias here, unlike the
        // accumulator path's zeroed entries: this is the stage the requantized
        // path turns on.
        let bs = Buffer::new(fd, page_aligned_size(shape.bs_buffer_bytes()), &file);
        ptr::write_bytes(bs.host_ptr, 0, bs.size);
        let mut dense_bias = vec![0u8; channels * 4];
        for c in 0..channels {
            dense_bias[c * 4..c * 4 + 4].copy_from_slice(&bias_at(c).to_le_bytes());
        }
        pack_int8_bias_to_bs(
            &dense_bias,
            channels,
            shape.padded_out_channels() as usize,
            1.0,
            1.0,
            0,
            std::slice::from_raw_parts_mut(bs.host_ptr, bs.size),
        )
        .unwrap();

        let output_bytes = page_aligned_size(shape.output_scratch_bytes(kernels) + PAGE_BYTES);
        let output = Buffer::new(fd, output_bytes, &file);
        ptr::write_bytes(output.host_ptr, SENTINEL, output.size);

        let mut commands = conv_2d_tile(shape, kernels, &Tile::whole(shape, kernels));
        relocate(
            &mut commands,
            Buffers {
                input: input.dma_address,
                weights: weights.dma_address,
                bias: bs.dma_address,
                output: output.dma_address,
            },
        );
        let command_buffer = Buffer::new(
            fd,
            page_aligned_size(commands.len() * mem::size_of::<u64>()),
            &file,
        );
        let words =
            std::slice::from_raw_parts_mut(command_buffer.host_ptr as *mut u64, commands.len());
        for (word, command) in words.iter_mut().zip(&commands) {
            *word = command.0;
        }
        let handles = [
            input.handle,
            weights.handle,
            bs.handle,
            output.handle,
            command_buffer.handle,
        ];
        for handle in handles {
            fini_bo(fd, handle).unwrap();
        }
        submit_jobs(
            fd,
            &[JobDesc {
                tasks: &[(command_buffer.dma_address, commands.len() as u32)],
                in_handles: &[
                    command_buffer.handle,
                    input.handle,
                    weights.handle,
                    bs.handle,
                ],
                out_handles: &[output.handle],
            }],
        )
        .unwrap();
        prep_bo(fd, output.handle, 5_000_000_000).unwrap();

        let raw = std::slice::from_raw_parts(output.host_ptr, output.size);
        let written = raw
            .iter()
            .rposition(|&b| b != SENTINEL)
            .map_or(0, |i| i + 1);
        assert!(
            written <= shape.output_scratch_bytes(kernels),
            "DPU wrote {written} bytes past the {} byte allocation",
            shape.output_scratch_bytes(kernels)
        );

        // Surface-major over `output_atom_bytes` atoms, one byte per lane at
        // int8. Read straight from the Shape rather than a constant, because
        // whether a requantized depthwise keeps the accumulator path's doubled
        // atom is exactly what is unknown here.
        let atom_bytes = shape.output_atom_bytes() as usize;
        let lanes = atom_bytes;
        let pixels = OH * OW;
        let read = |oy: usize, ox: usize, c: usize| -> i8 {
            let offset = ((c / lanes) * pixels + oy * OW + ox) * atom_bytes + (c % lanes);
            raw[offset] as i8
        };

        let mut mismatches = 0usize;
        let mut first = None;
        let mut unwritten = 0usize;
        let mut saturated_low = 0usize;
        let mut saturated_high = 0usize;
        for oy in 0..OH {
            for ox in 0..OW {
                for c in 0..channels {
                    let mut accumulator = i64::from(bias_at(c));
                    for ky in 0..KERNEL {
                        for kx in 0..KERNEL {
                            accumulator += i64::from(input_at(oy + ky, ox + kx, c))
                                * i64::from(weight_at(ky, kx, c));
                        }
                    }
                    let expected = device_requantize(accumulator, &multiplier);
                    if expected == -128 {
                        saturated_low += 1;
                    }
                    if expected == 127 {
                        saturated_high += 1;
                    }
                    let actual = read(oy, ox, c);
                    if actual as u8 == SENTINEL {
                        unwritten += 1;
                    }
                    if actual != expected {
                        mismatches += 1;
                        first.get_or_insert((oy, ox, c, expected, actual));
                    }
                }
            }
        }
        for handle in handles {
            let _ = close_bo(fd, handle);
        }

        let total = pixels * channels;
        eprintln!(
            "requantized int8 depthwise Cin {channels}: atom {atom_bytes} B, \
             {mismatches}/{total} wrong, {unwritten} unwritten, \
             saturation {saturated_low} low / {saturated_high} high"
        );
        // A case that clamps everywhere would pass while computing nothing.
        assert!(
            saturated_low + saturated_high < total / 4,
            "Cin {channels}: {}/{total} outputs saturate -- RATIO makes this case vacuous",
            saturated_low + saturated_high
        );
        if let Some((oy, ox, c, expected, actual)) = first {
            panic!(
                "requantized int8 depthwise at Cin {channels}: {mismatches} of {total} \
                 elements wrong; first at (y {oy}, x {ox}, c {c}): expected {expected}, \
                 got {actual}"
            );
        }
    }
}

/// One output atom's worth of channels, the simplest case that can be right.
#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn requantized_int8_depthwise_within_one_output_atom() {
    check_requantized_depthwise(64);
}

/// Two atoms, which is the only way to pin the output surface stride -- the
/// bug that made the accumulator path's depthwise output come back permuted.
#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn requantized_int8_depthwise_across_two_output_atoms() {
    check_requantized_depthwise(128);
}

/// The channel counts MobileNetV2 actually asks of its depthwise layers, and
/// the CBUF atom-rounding cases (`ceil(Cin/16) % 4 == 3`) that under-granted a
/// data bank on the accumulator path.
#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn requantized_int8_depthwise_model_channel_counts() {
    for channels in [48, 144, 192, 288] {
        check_requantized_depthwise(channels);
    }
}
