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
        BsEntry, Buffers, ConvPlan, Multiplier, Precision, Quantization, Shape, Tile, conv_2d_tile,
        relocate, write_bs_buffer,
    },
    device::{Buffer, JobDesc, close_bo, fini_bo, prep_bo, submit_jobs},
    tensor_layout::{pack_depthwise_to_rocket_weights, pack_nhwc_to_nc1hwc2},
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const PAGE_BYTES: usize = 4096;
/// The historical single-tile geometry every pre-sweep case in this file
/// uses. `check_int8_depthwise_at` takes an explicit extent instead.
const DEFAULT_W: usize = 34;
const DEFAULT_H: usize = 34;
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

/// The int8 counterpart of
/// `conv_depthwise_fp16_exact_hw.rs::fp16_depthwise_exact_above_the_dense_ceiling`,
/// walking the channel range that opened when `MAX_DEPTHWISE_CHANNELS` was
/// split out of the dense ceiling.
///
/// `MAX_INT8_DEPTHWISE_CHANNELS` is deliberately still 512 -- lower than the
/// fp16 bound -- because the int8 depthwise sweep stopped there, not because
/// anything above it was known to fail. This is the sweep that settles it.
///
/// The geometry is fixed at this file's 34x34 -> 32x32 and only the channel
/// count moves, which is the right axis: the bound is a channel bound, and
/// int8 depthwise packs coefficients in **64-channel** groups (twice fp16's
/// 32), so the step is 128 with deliberate non-multiples of 64 and of the
/// 16-channel int8 feature atom -- both of this path's known bugs were at a
/// group boundary, and a sweep sampling only aligned counts saw neither.
///
/// One case per process; these allocate several MB each.
///
/// ```text
/// for i in $(seq 0 N); do ROCKET_DEPTHWISE_SWEEP_ONLY=$i ./conv_depthwise_int8_exact_hw \
///     int8_depthwise_exact_above_the_dense_ceiling --ignored --nocapture; done
/// ```
#[test]
#[ignore = "needs /dev/accel/accel0 -- sweeps the int8 depthwise channel range above the dense ceiling"]
fn int8_depthwise_exact_above_the_dense_ceiling() {
    unsafe { std::env::set_var("ROCKET_ALLOW_UNBACKED_CHANNELS", "1") };

    let mut channels: Vec<usize> = (640..=1400).step_by(128).collect();
    // MobileNetV2's own int8 depthwise channel counts.
    channels.extend([528, 816, 1344]);
    // Controls at and below `MAX_INT8_DEPTHWISE_CHANNELS`, so a failure above
    // it is attributable to the channel count rather than to this harness.
    channels.extend([256, 272, 288, 320, 352, 384, 448, 496, 512]);
    // Two points past the range so the ceiling lands on a measured boundary.
    channels.extend([1408, 1536]);
    // Non-multiples of the 16-channel int8 feature atom are NOT swept here:
    // this harness sizes its packed input buffer at the dense byte count, so
    // `pack_nhwc_to_nc1hwc2` fails on them before any hardware runs. That is
    // a harness limit, not a device one -- see the fp16 sweep, which does
    // cover unaligned counts because its buffer is sized from the shape.
    channels.sort_unstable();
    channels.dedup();

    let only = std::env::var("ROCKET_DEPTHWISE_SWEEP_ONLY")
        .ok()
        .and_then(|value| value.parse::<usize>().ok());
    eprintln!("int8 depthwise channel sweep: {} case(s)", channels.len());
    for (index, channels) in channels.into_iter().enumerate() {
        if only.is_some_and(|wanted| wanted != index) {
            continue;
        }
        eprintln!("[{index}] Cin {channels} at {DEFAULT_W}x{DEFAULT_H}");
        check_int8_depthwise(channels);
    }
}

/// Isolates the variable behind the int8 depthwise channel cliff.
///
/// The channel sweep above is exact to `Cin` 288 and wrong from 320 -- and
/// 288 is the last count whose plan fits **one** row tile at 34x34, while
/// 320 is the first that needs two. Every int8 depthwise test that existed
/// before the sweep (`Cin` 48, 64, 112, 128, 240) is single-tile, which is
/// why none of them saw this.
///
/// So the suspect is the tile count, not the channel count. This holds the
/// channel count at a value the sweep proved exact and shrinks the extent
/// until the planner splits the image: if a low-`Cin` multi-tile case is
/// also wrong, the cliff is a row-tiling bug and a channel ceiling is the
/// wrong guard for it entirely.
///
/// The fp16 depthwise path is *not* affected -- its own sweep passes at
/// 112x112 with four row tiles -- so this is specific to int8 depthwise.
///
/// Run one case per process.
#[test]
#[ignore = "needs /dev/accel/accel0 -- isolates the int8 depthwise multi-tile cliff"]
fn int8_depthwise_multi_tile_is_wrong() {
    unsafe { std::env::set_var("ROCKET_ALLOW_UNBACKED_CHANNELS", "1") };
    // (channels, extent). 34 is the file's single-tile control at each
    // channel count; the taller extents force the planner to split rows
    // while holding the channel count fixed.
    let cases = [
        (64usize, 34usize),
        (64, 130),
        (64, 260),
        (240, 34),
        (240, 70),
        (240, 140),
    ];
    let only = std::env::var("ROCKET_DEPTHWISE_SWEEP_ONLY")
        .ok()
        .and_then(|value| value.parse::<usize>().ok());
    eprintln!("multi-tile isolation: {} case(s)", cases.len());
    for (index, (channels, extent)) in cases.into_iter().enumerate() {
        if only.is_some_and(|wanted| wanted != index) {
            continue;
        }
        eprintln!("[{index}] Cin {channels} at {extent}x{extent}");
        check_int8_depthwise_at(channels, extent, extent);
    }
}

/// The int8 counterpart of `fp16_depthwise_corpus_shapes_are_exact`: the
/// depthwise vendor corpus's 40 (extent, `Cin`) pairs, on hardware.
///
/// Geometry note: the corpus captures 3x3 SAME (pad 1) because that is what
/// the vendor's ONNX has, while this harness convolves **valid** (pad 0),
/// which is what `rocket-hal-driver` actually programs from a compiled
/// `.vmfb` -- IREE materializes padding as a separate `tensor.pad`. So these
/// are the corpus's extents and channel counts with the runtime's padding;
/// the output extent is two smaller and the tile structure is otherwise the
/// same. Each case prints its own tile count.
///
/// Run one case per process.
#[test]
#[ignore = "needs /dev/accel/accel0 -- runs the depthwise vendor corpus shapes"]
fn int8_depthwise_corpus_shapes_are_exact() {
    unsafe { std::env::set_var("ROCKET_ALLOW_UNBACKED_CHANNELS", "1") };
    let mut cases = Vec::new();
    for extent in [34usize, 56, 70, 112, 130] {
        for channels in (48usize..=384).step_by(48) {
            cases.push((channels, extent));
        }
    }
    let only = std::env::var("ROCKET_DEPTHWISE_SWEEP_ONLY")
        .ok()
        .and_then(|value| value.parse::<usize>().ok());
    eprintln!("int8 depthwise corpus: {} case(s)", cases.len());
    for (index, (channels, extent)) in cases.into_iter().enumerate() {
        if only.is_some_and(|wanted| wanted != index) {
            continue;
        }
        eprintln!("[{index}] Cin {channels} at {extent}x{extent} pad 0");
        check_int8_depthwise_at(channels, extent, extent);
    }
}

fn check_int8_depthwise(channels: usize) {
    check_int8_depthwise_at(channels, DEFAULT_W, DEFAULT_H);
}

/// `check_int8_depthwise` at an explicit input extent.
///
/// The extent is what decides how many row tiles `ConvPlan` needs, and the
/// tile count turned out to matter far more than the channel count -- see
/// `int8_depthwise_multi_tile_is_wrong`.
#[allow(non_snake_case)]
fn check_int8_depthwise_at(channels: usize, w: usize, h: usize) {
    let (W, H) = (w, h);
    let (OW, OH) = (w - (KERNEL - 1), h - (KERNEL - 1));
    let precision = Precision::Int8Accumulator(Quantization {
        input_zero_point: 0,
        output_zero_point: 0,
        weight_zero_point: 0,
        input_scale: 1.0,
        weights_scale: 1.0,
        multiplier: Multiplier::from_ratio(1.0),
    });
    // `depthwise_with_precision`, not `with_precision().with_depthwise()`:
    // the latter runs the *dense* channel bound first and cannot build the
    // wide shapes the sweep below characterizes.
    let shape = Shape::depthwise_with_precision(W as u32, H as u32, 1, channels as u32, precision)
        .with_padding([0, 0]);
    let kernels = [KERNEL, KERNEL];
    assert_eq!(shape.output_width(kernels) as usize, OW);
    assert_eq!(shape.output_height(kernels) as usize, OH);
    let tiles = ConvPlan::new(shape, kernels).tiles().len();
    eprintln!("  Cin {channels} at {W}x{H}: {tiles} tile(s)");

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
