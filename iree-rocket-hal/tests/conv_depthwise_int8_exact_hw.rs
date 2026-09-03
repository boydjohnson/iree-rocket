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

use std::{collections::BTreeMap, fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    conv::{
        AccumulatorOutputTile, BsEntry, Buffers, ConvPlan, Multiplier, Precision, Quantization,
        Shape, write_bs_buffer,
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

/// The smallest int8 depthwise plan that splits rows, for reading the
/// failure rather than counting it.
///
/// 45x45 `Cin` 16 is the smallest shape in the whole sweep whose plan needs
/// two row tiles: `out[0..40)` and `out[40..43)`, with a 32 KB input. Its
/// single-tile sibling at the same channel count is the control -- if the
/// control passes and this fails, the only variable is the split.
///
/// `report_mismatches` prints per-tile attribution, whether the bad lanes
/// were written at all, and whether the values are merely displaced from
/// another output row.
#[test]
#[ignore = "needs /dev/accel/accel0 -- minimal int8 depthwise multi-tile reproducer"]
fn int8_depthwise_smallest_multi_tile_repro() {
    unsafe { std::env::set_var("ROCKET_ALLOW_UNBACKED_CHANNELS", "1") };
    let only = std::env::var("ROCKET_DEPTHWISE_SWEEP_ONLY")
        .ok()
        .and_then(|value| value.parse::<usize>().ok());
    // (channels, extent, expected tiles). The single-tile controls come
    // first so a batch run establishes them before anything fails.
    let cases = [
        (16usize, 44usize, 1usize),
        (16, 45, 2),
        (32, 44, 1),
        (32, 45, 2),
    ];
    for (index, (channels, extent, tiles)) in cases.into_iter().enumerate() {
        if only.is_some_and(|wanted| wanted != index) {
            continue;
        }
        eprintln!("[{index}] Cin {channels} at {extent}x{extent} (expect {tiles} tile(s))");
        check_int8_depthwise_at(channels, extent, extent);
    }
}

/// Compacts staged accumulator scratch into the logical surface-major
/// output the comparison below reads.
///
/// A port of `conv2d_oracle_hw.rs`'s assembler. Each staged tile holds its
/// own pixels densely, surface-major within the tile; the logical surface is
/// surface-major over the whole image, so the copy has to re-stride both.
fn assemble_staged(
    shape: Shape,
    kernels: [usize; 2],
    scratch: &[u8],
    tiles: &[AccumulatorOutputTile],
) -> Vec<u8> {
    let output_width = shape.output_width(kernels) as usize;
    let output_pixels = output_width * shape.output_height(kernels) as usize;
    let block_bytes = shape.output_atom_bytes() as usize;
    let bytes_per_pixel =
        shape.padded_out_channels() as usize * shape.precision.output_element_bytes() as usize;
    let blocks_per_pixel = bytes_per_pixel.div_ceil(block_bytes);
    let mut output = vec![SENTINEL; output_pixels * blocks_per_pixel * block_bytes];
    for tile in tiles {
        let tile_pixels = tile.output_rows * tile.output_columns;
        for surface in 0..blocks_per_pixel {
            for row in 0..tile.output_rows {
                for column in 0..tile.output_columns {
                    let local = row * tile.output_columns + column;
                    let source =
                        tile.scratch_offset + (surface * tile_pixels + local) * block_bytes;
                    let destination = (surface * output_pixels
                        + (tile.output_row + row) * output_width
                        + tile.output_column
                        + column)
                        * block_bytes;
                    output[destination..destination + block_bytes]
                        .copy_from_slice(&scratch[source..source + block_bytes]);
                }
            }
        }
    }
    output
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

        // Sized for staged scratch, which can exceed the logical output: a
        // partial last tile still writes whole blocks.
        let output_bytes = page_aligned_size(
            shape.output_scratch_bytes(kernels).max(
                ConvPlan::new(shape, kernels)
                    .programs_with_staged_accumulator_output(Buffers {
                        input: 0,
                        weights: 0,
                        bias: 0,
                        output: 0,
                    })
                    .scratch_bytes,
            ) + PAGE_BYTES,
        );
        let output = Buffer::new(fd, output_bytes, &file);
        ptr::write_bytes(output.host_ptr, SENTINEL, output.size);

        // Driven through `ConvPlan`, not `Tile::whole`. This harness used to
        // program the whole image as one tile and submit a single job, which
        // silently **over-commits the CBUF** for any geometry the planner
        // would have split -- the CNA then reads past its resident window and
        // the output is wrong. That produced a perfect but entirely spurious
        // correlation between "ConvPlan wants more than one tile" and
        // "hardware returns wrong values", which read like a device bug and
        // was not one. The fp16 sibling has carried this comment for a while;
        // the int8 one had not caught up. Dispatching the plan is also what
        // the driver does, so the tiling under test is a compiled model's.
        let plan = ConvPlan::new(shape, kernels);
        // The **staged** accumulator entry point, which is what
        // `rocket-hal-driver` uses. `programs_with_buffers` places every tile
        // into a shared full-image surface, and its own doc says why that is
        // not interchangeable: a shared-image tile keeps the full output
        // geometry in its destination surface stride and row notch, so
        // moving only its base address is not enough. Staging gives each
        // tile a private contiguous range and the host compacts afterwards.
        let staged = plan.programs_with_staged_accumulator_output(Buffers {
            input: input.dma_address,
            weights: weights.dma_address,
            bias: bs.dma_address,
            output: output.dma_address,
        });
        assert!(
            staged.scratch_bytes <= output.size,
            "staged scratch needs {} bytes, allocated {}",
            staged.scratch_bytes,
            output.size
        );
        let programs = staged.programs;
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
        let handles: Vec<u32> = data_handles
            .iter()
            .copied()
            .chain(command_buffers.iter().map(|(buffer, _)| buffer.handle))
            .collect();
        submit_jobs(fd, &jobs).unwrap_or_else(|error| {
            panic!(
                "Cin {channels} at {W}x{H} ({} tile(s)): SUBMIT failed: {error}",
                jobs.len()
            )
        });
        prep_bo(fd, output.handle, 5_000_000_000).unwrap();

        let scratch = std::slice::from_raw_parts(output.host_ptr, output.size);
        let compacted = assemble_staged(shape, kernels, scratch, &staged.tiles);
        let raw = compacted.as_slice();
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

        let expected_at = |oy: usize, ox: usize, c: usize| -> i32 {
            let mut sum = 0i32;
            for ky in 0..KERNEL {
                for kx in 0..KERNEL {
                    sum += input_at(oy + ky, ox + kx, c) as i32 * weight_at(ky, kx, c) as i32;
                }
            }
            sum
        };

        let mut mismatches = 0usize;
        let mut first = None;
        // Per output row, so a failure can be attributed to a row tile.
        let mut wrong_in_row = vec![0usize; OH];
        // An i32 lane the DPU never wrote still holds the fill pattern.
        let sentinel_lane = i32::from_le_bytes([SENTINEL; 4]);
        let mut untouched = 0usize;
        for oy in 0..OH {
            for ox in 0..OW {
                for c in 0..channels {
                    let expected = expected_at(oy, ox, c);
                    let actual = read(oy, ox, c);
                    if actual != expected {
                        mismatches += 1;
                        wrong_in_row[oy] += 1;
                        if actual == sentinel_lane {
                            untouched += 1;
                        }
                        first.get_or_insert((oy, ox, c, expected, actual));
                    }
                }
            }
        }

        if mismatches > 0 {
            report_mismatches(
                channels,
                W,
                H,
                OW,
                OH,
                &shape,
                kernels,
                &wrong_in_row,
                untouched,
                &expected_at,
                &read,
            );
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

/// Prints what a failure looks like, rather than only how big it is.
///
/// The three questions worth asking of a wrong depthwise output, in order:
///
///   1. **Which row tile owns the bad rows?** `ConvPlan`'s tile boundaries
///      are printed alongside a per-row wrong count, so "tile 1 is entirely
///      wrong" and "the last row of every tile is wrong" look different.
///   2. **Was the lane written at all?** An untouched lane still holds
///      `SENTINEL`; that is a coverage failure, not an arithmetic one, and
///      it is what the dense accumulator cliff looks like.
///   3. **Is the value right but in the wrong place?** For a sample of bad
///      elements this searches the *other* output rows for a row whose
///      expected value equals what the device produced. A consistent row
///      delta means the tile read its input at the wrong offset, which is
///      the classic multi-tile addressing bug and tells you the stride.
#[allow(clippy::too_many_arguments, non_snake_case)]
fn report_mismatches(
    channels: usize,
    W: usize,
    H: usize,
    OW: usize,
    OH: usize,
    shape: &Shape,
    kernels: [usize; 2],
    wrong_in_row: &[usize],
    untouched: usize,
    expected_at: &dyn Fn(usize, usize, usize) -> i32,
    read: &dyn Fn(usize, usize, usize) -> i32,
) {
    let plan = ConvPlan::new(*shape, kernels);
    let per_row = OW * channels;
    eprintln!("\n--- int8 depthwise Cin {channels} at {W}x{H} (out {OW}x{OH}) ---");
    eprintln!(
        "  banks={}/{}  tiles={}",
        plan.data_banks(),
        plan.weight_banks(),
        plan.tiles().len()
    );
    for (index, tile) in plan.tiles().iter().enumerate() {
        let first = tile.rows.out_first as usize;
        let last = first + tile.rows.out_rows as usize;
        let wrong: usize = wrong_in_row[first..last.min(OH)].iter().sum();
        let rows_touched = wrong_in_row[first..last.min(OH)]
            .iter()
            .filter(|&&n| n > 0)
            .count();
        eprintln!(
            "  tile {index}: out[{first}..{last}) in[{}..{})  wrong {wrong}/{} ({rows_touched}/{} rows affected)",
            tile.rows.in_first,
            tile.rows.in_first + tile.rows.in_rows,
            (last.min(OH) - first) * per_row,
            last.min(OH) - first,
        );
    }
    eprintln!("  lanes still holding SENTINEL (never written): {untouched}");

    // Per 64-lane output atom plane. The depthwise accumulator writes
    // 256-byte atoms, so a channel count above 64 spans several planes and a
    // plane-stride error shows up as "plane 0 fine, the rest missing".
    let lanes_per_atom = 64usize;
    let planes = channels.div_ceil(lanes_per_atom);
    if planes > 1 {
        let sentinel_lane = i32::from_le_bytes([SENTINEL; 4]);
        for plane in 0..planes {
            let (mut wrong, mut blank, mut total) = (0usize, 0usize, 0usize);
            for oy in 0..OH {
                for ox in (0..OW).step_by(3) {
                    for c in plane * lanes_per_atom..((plane + 1) * lanes_per_atom).min(channels) {
                        total += 1;
                        let actual = read(oy, ox, c);
                        if actual != expected_at(oy, ox, c) {
                            wrong += 1;
                            if actual == sentinel_lane {
                                blank += 1;
                            }
                        }
                    }
                }
            }
            eprintln!(
                "    atom plane {plane} (c {}..{}): wrong {wrong}/{total}, of which never written {blank}",
                plane * lanes_per_atom,
                ((plane + 1) * lanes_per_atom).min(channels),
            );
        }
    }

    // Row-by-row, compressed to runs so a 130-row output stays readable.
    let mut runs: Vec<(usize, usize, bool)> = Vec::new();
    for (row, &count) in wrong_in_row.iter().enumerate() {
        let bad = count > 0;
        match runs.last_mut() {
            Some((_, end, kind)) if *kind == bad => *end = row,
            _ => runs.push((row, row, bad)),
        }
    }
    let summary: Vec<String> = runs
        .iter()
        .map(|(start, end, bad)| format!("{}{start}..={end}", if *bad { "BAD " } else { "ok " }))
        .collect();
    eprintln!("  rows: {}", summary.join(", "));

    // Which pixels in the first bad row survived? A tile that stops being
    // correct partway through -- rather than being wrong everywhere -- is a
    // different bug from one that reads its input at the wrong offset.
    if let Some(bad_row) = wrong_in_row.iter().position(|&n| n > 0) {
        let mut runs: Vec<(usize, usize, bool)> = Vec::new();
        for ox in 0..OW {
            let bad = (0..channels).any(|c| read(bad_row, ox, c) != expected_at(bad_row, ox, c));
            match runs.last_mut() {
                Some((_, end, kind)) if *kind == bad => *end = ox,
                _ => runs.push((ox, ox, bad)),
            }
        }
        let summary: Vec<String> = runs
            .iter()
            .map(|(start, end, bad)| {
                format!("{}{start}..={end}", if *bad { "BAD " } else { "ok " })
            })
            .collect();
        eprintln!("  row {bad_row} columns: {}", summary.join(", "));
        let correct_pixels: usize = runs
            .iter()
            .filter(|(_, _, bad)| !bad)
            .map(|(start, end, _)| end - start + 1)
            .sum();
        eprintln!("  correct pixels in row {bad_row}: {correct_pixels} of {OW}");
    }

    // Is the tile convolving a *shifted* input window? Try every small
    // (dy, dx) and see which one reproduces what the device returned.
    let mut shifts: BTreeMap<(i32, i32), usize> = BTreeMap::new();
    let mut shift_samples = 0usize;
    'shift: for oy in 0..OH {
        if wrong_in_row[oy] == 0 {
            continue;
        }
        for ox in (0..OW).step_by(3) {
            for c in (0..channels).step_by(3) {
                let actual = read(oy, ox, c);
                if actual == expected_at(oy, ox, c) {
                    continue;
                }
                shift_samples += 1;
                for dy in -4i32..=4 {
                    for dx in -4i32..=4 {
                        let (sy, sx) = (oy as i32 + dy, ox as i32 + dx);
                        if sy < 0 || sx < 0 || sy + 2 >= H as i32 || sx + 2 >= W as i32 {
                            continue;
                        }
                        if expected_at(sy as usize, sx as usize, c) == actual {
                            *shifts.entry((dy, dx)).or_default() += 1;
                        }
                    }
                }
                if shift_samples >= 120 {
                    break 'shift;
                }
            }
        }
    }
    eprintln!(
        "  input-window shifts reproducing the device value, {shift_samples} sampled: {shifts:?}"
    );

    // Does a bad row hold some *other* output row's data? Intersecting the
    // candidate source rows over several (column, channel) probes is far
    // stronger than matching one element at a time: a single i32 can collide
    // by chance, a dozen agreeing on the same source row cannot.
    let probes: Vec<(usize, usize)> = (0..OW)
        .step_by((OW / 6).max(1))
        .flat_map(|ox| {
            (0..channels)
                .step_by((channels / 3).max(1))
                .map(move |c| (ox, c))
        })
        .take(12)
        .collect();
    let mut displaced: Vec<(usize, i64)> = Vec::new();
    let mut unexplained = 0usize;
    for oy in 0..OH {
        if wrong_in_row[oy] == 0 {
            continue;
        }
        let mut candidates: Option<Vec<usize>> = None;
        for &(ox, c) in &probes {
            let actual = read(oy, ox, c);
            let matches: Vec<usize> = (0..OH)
                .filter(|&y| expected_at(y, ox, c) == actual)
                .collect();
            candidates = Some(match candidates {
                None => matches,
                Some(prev) => prev.into_iter().filter(|y| matches.contains(y)).collect(),
            });
            if candidates.as_ref().is_some_and(|list| list.is_empty()) {
                break;
            }
        }
        match candidates.as_deref() {
            Some([source]) => displaced.push((oy, *source as i64 - oy as i64)),
            _ => unexplained += 1,
        }
    }
    if displaced.is_empty() {
        eprintln!(
            "  displacement: no bad row holds another output row's data \
             ({unexplained} bad rows unexplained)"
        );
    } else {
        let mut by_delta: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
        for (row, delta) in displaced {
            by_delta.entry(delta).or_default().push(row);
        }
        eprintln!("  displacement, {unexplained} bad row(s) unexplained:");
        for (delta, rows) in by_delta {
            eprintln!(
                "    source = dest {delta:+}  for {} row(s), e.g. {:?}",
                rows.len(),
                &rows[..rows.len().min(8)]
            );
        }
    }
}
