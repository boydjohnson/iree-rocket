//! Exact int8 depthwise convolution check on real RK3588 hardware.
//!
//! This exists because `conv_depthwise_int8_probe_hw.rs` could not fail. That
//! probe drives the same path with a spatially uniform input and an all-ones
//! filter and only prints what comes back -- a filter whose taps are all
//! equal cannot observe a tap permutation, and an input constant across space
//! cannot observe a spatial one. Two bugs were live at once and the probe was
//! green throughout:
//!
//!   * `pack_depthwise_to_rocket_weights` grouped int8 coefficients 16
//!     channels at a time. The hardware groups 64 (`DEPTHWISE_GROUP_BYTES`),
//!     so every channel past the first group convolved with the wrong tap.
//!   * Depthwise accumulator output is written in 256-byte atoms, not the
//!     128-byte atoms the dense path uses, so the result came back permuted
//!     (see `Shape::output_atom_bytes`).
//!
//! So this test uses `padding = [0, 0]` -- what rocket-hal-driver actually
//! programs from a compiled `.vmfb`, unlike the probe's default SAME padding
//! -- gives every (tap, channel) pair a distinct coefficient so neither
//! permutation can hide, and asserts per element instead of printing. Two
//! channel counts: 64 fills a single output atom, 128 spans two, which is the
//! only way to pin the surface stride.

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    conv::{
        BsEntry, Buffers, Multiplier, Precision, Quantization, Shape, Tile, conv_2d_tile, relocate,
        write_bs_buffer,
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

fn page_aligned_size(size: usize) -> usize {
    size.div_ceil(PAGE_BYTES) * PAGE_BYTES
}

/// Small deterministic values: accumulators stay far inside i32, and the
/// coefficient for one (tap, channel) is distinct from its neighbours in both
/// axes, so a tap permutation and a channel permutation each show up.
fn input_at(y: usize, x: usize, c: usize) -> i8 {
    (((y * 31 + x * 13 + c * 7) % 61) as i32 - 30) as i8
}

fn weight_at(ky: usize, kx: usize, c: usize) -> i8 {
    (((ky * KERNEL + kx) * 5 + c * 2) % 11) as i8 - 5
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn int8_depthwise_exact_within_one_output_atom() {
    check_int8_depthwise(64);
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn int8_depthwise_exact_across_two_output_atoms() {
    check_int8_depthwise(128);
}

/// Channel counts whose *atom* count (`ceil(Cin / 16)`) is one short of a
/// multiple of four, where `Shape::cbuf_atoms` rounds up and everything else
/// at int8 stays exact.
///
/// `data_bank_demand` used to bill the CBUF with `weight_atoms` here, which
/// is the un-rounded count at int8, and under-granted a data bank. The tile
/// then read past the resident window and lost its last input rows -- 4 rows
/// at Cin 48 and 112, 1 row at Cin 240, exactly the shortfall the byte
/// arithmetic predicts. Nothing else in the suite covers this: fp16 cannot
/// reach it (its `weight_channels` is already quad-rounded, so the two atom
/// counts are equal) and every other int8 test happens to sit on a whole
/// multiple of four atoms.
#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn int8_depthwise_exact_where_the_cbuf_atom_charge_rounds_up() {
    for channels in [48, 112, 240] {
        assert_eq!(
            channels / 16 % 4,
            3,
            "Cin {channels} is not a rounding case"
        );
        check_int8_depthwise(channels);
    }
}

fn check_int8_depthwise(channels: usize) {
    let precision = Precision::Int8Accumulator(Quantization {
        input_zero_point: 0,
        output_zero_point: 0,
        weight_zero_point: 0,
        input_scale: 1.0,
        weights_scale: 1.0,
        multiplier: Multiplier::from_ratio(1.0),
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

        // `pack_depthwise_to_rocket_weights` takes `[c][ky][kx]` -- the order
        // the compiler's HWC -> CHW transpose hands the driver.
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

        let bs = Buffer::new(fd, page_aligned_size(shape.bs_buffer_bytes()), &file);
        ptr::write_bytes(bs.host_ptr, 0, bs.size);
        write_bs_buffer(
            std::slice::from_raw_parts_mut(bs.host_ptr, bs.size),
            &vec![BsEntry::default(); shape.padded_out_channels() as usize],
        );

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

        // Surface-major over `output_atom_bytes` atoms: atom `a` holds, for
        // every pixel, i32 lanes `a * lanes .. (a + 1) * lanes`.
        let atom_bytes = shape.output_atom_bytes() as usize;
        let lanes = atom_bytes / mem::size_of::<i32>();
        let pixels = OH * OW;
        let read = |oy: usize, ox: usize, c: usize| -> i32 {
            let offset = ((c / lanes) * pixels + oy * OW + ox) * atom_bytes
                + (c % lanes) * mem::size_of::<i32>();
            i32::from_le_bytes(raw[offset..offset + 4].try_into().unwrap())
        };

        let mut mismatches = 0usize;
        let mut first = None;
        for oy in 0..OH {
            for ox in 0..OW {
                for c in 0..channels {
                    let mut expected = 0i32;
                    for ky in 0..KERNEL {
                        for kx in 0..KERNEL {
                            expected +=
                                input_at(oy + ky, ox + kx, c) as i32 * weight_at(ky, kx, c) as i32;
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
        for handle in handles {
            let _ = close_bo(fd, handle);
        }
        if let Some((oy, ox, c, expected, actual)) = first {
            panic!(
                "int8 depthwise at Cin {channels}: {mismatches} of {} elements wrong; \
                 first at (y {oy}, x {ox}, c {c}): expected {expected}, got {actual}",
                pixels * channels
            );
        }
    }
}
