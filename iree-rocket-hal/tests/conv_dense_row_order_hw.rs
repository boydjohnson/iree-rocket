//! Isolated hardware test: does a multi-row-tile dense-input dispatch write
//! each tile's rows to the correct output rows?
//!
//! `conv_dense_channel_order_hw.rs` ruled out a channel-order bug on the
//! Cin<=4 dense path, and `rocket-hal-driver`'s
//! `device::compaction_tests::compacts_eight_full_surfaces_cout_64` ruled
//! out a bug in the NC1HWC2-to-dense output interleaving at `features.0`'s
//! real Cout=64. Both of those tests, like every other hand-rolled hw test
//! in this repo, used images small enough to stay in one CBUF row-tile.
//! `features.0`'s real shape (226x226) does not: it splits into 5 tiles.
//! Every prior "5/5 passed" result for that exact shape, including the
//! original five-shape sweep, used a *uniform* fill -- which is exactly as
//! blind to a tile-boundary row-addressing bug (rows from the wrong tile,
//! or written to the wrong output offset) as it was to a channel swap: with
//! every row holding the same value, misrouting rows between tiles is
//! numerically invisible.
//!
//! This uses a 1x1 kernel (Cin=Cout=1, identity weight) so there is no
//! spatial *tap* order to reason about -- output row `y` must read input
//! row `y` exactly, one-to-one, no halo, no padding offset. Each input row
//! is filled with one of three distinguishable constants cycling by `y %
//! 3`, at a height (1200) that `ConvPlan` splits into 2 tiles. A tile-
//! boundary bug shows up as a row reading back the wrong phase of the
//! cycle, or a whole tile's rows shifted, rather than an aggregate
//! magnitude mismatch.
//!
//! The original failure was at 32x400 under the old dense CBUF accounting.
//! Vendor-matching ARGB pixel charging now correctly fits that shape in one
//! tile, so the primary multi-tile regression uses height=1200. The separate
//! diagnostic probes below retain the historical 32x400 shape.
//!
//!   CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!     cargo test --target aarch64-unknown-linux-gnu --release \
//!       --test conv_dense_row_order_hw --no-run
//!
//!   ./conv_dense_row_order_hw-<hash> --ignored --nocapture

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    conv::{Buffers, ConvPlan, Kernels, Shape},
    device::{Buffer, JobDesc, close_bo, fini_bo, prep_bo, submit_jobs},
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const FP16_BYTES: usize = 2;
// Raw hardware output (before compact_atomic_output, which this test -- like
// every other hand-rolled hw test -- never invokes) always writes one full
// 16-byte atomic slot per pixel per 8-channel surface, regardless of how
// many of the 8 lanes are real. Cout=1 here still costs the whole 16 bytes;
// only byte offset 0 within each pixel's slot is real data.
const FEATURE_ATOM_BYTES: usize = 16;
const PAGE_BYTES: usize = 4096;

// fp16 1.0, 2.0, 3.0 -- same distinguishable-constant idea as
// conv_dense_channel_order_hw.rs, cycled by row instead of by channel.
const ROW_VALUES: [u16; 3] = [0x3c00, 0x4000, 0x4200];

const WIDTH: u32 = 32;
const HEIGHT: u32 = 400; // Historical failure shape, now correctly one tile.
const MULTI_TILE_HEIGHT: u32 = 1200; // ConvPlan::new splits this into 2 tiles.
const REPS: usize = 3;

fn page_aligned_size(size: usize) -> usize {
    size.div_ceil(PAGE_BYTES) * PAGE_BYTES
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

struct Failure {
    mismatches: usize,
    samples: Vec<String>,
    timed_out: bool,
    tiles: usize,
    /// (y, x) of the first mismatch found, in row-major scan order --
    /// the diagnostic signal `dense_row_order_break_point_scales_with_width`
    /// actually wants; `samples` is for humans reading `--nocapture`.
    first_bad: Option<(usize, usize)>,
}

fn run(
    fd: i32,
    file: &std::fs::File,
    width: u32,
    height: u32,
    cin: u32,
    banks: Option<(u32, u32)>,
    timeout_ns: u64,
) -> Result<(), Failure> {
    let kernels: Kernels = [1, 1];
    let shape = Shape::with_out_channels(width, height, 1, cin, 1);
    debug_assert!(matches!(
        shape.layout(),
        iree_rocket_hal::rocket::conv::FeatureLayout::Dense
    ));

    let plan = match banks {
        Some((data_banks, weight_banks)) => {
            ConvPlan::with_cbuf_banks(shape, kernels, data_banks, weight_banks)
        }
        None => ConvPlan::new(shape, kernels),
    };
    let tiles = plan.tiles().len();
    let pixel_count = width as usize * height as usize;
    let input_bytes = pixel_count * cin as usize * FP16_BYTES;
    let output_bytes = pixel_count * FEATURE_ATOM_BYTES; // Cout=1: one surface, but a full atomic slot per pixel regardless.

    unsafe {
        let buf_input = Buffer::new(fd, page_aligned_size(input_bytes), file);
        let input_words =
            std::slice::from_raw_parts_mut(buf_input.host_ptr as *mut u16, input_bytes / 2);
        for y in 0..height as usize {
            let value = ROW_VALUES[y % ROW_VALUES.len()];
            for x in 0..width as usize {
                for c in 0..cin as usize {
                    input_words[(y * width as usize + x) * cin as usize + c] = value;
                }
            }
        }

        // Identity weight: single tap, only input channel 0 feeds output
        // channel 0 (weight 1.0), every other input channel weighted 0 --
        // channel order was already confirmed correct
        // (conv_dense_channel_order_hw.rs), so this keeps the expected
        // value the same regardless of `cin`: whatever row value channel 0
        // was filled with, unconditionally. The CNA's blocked weight layout
        // pads Cin up to a full 8-channel atom regardless of the real
        // count (tensor_layout::rocket_weight_storage_size), and with only
        // one output block and one input group (Cout=1, Cin<=8, 1x1
        // kernel) that padded layout is simply eight consecutive per-
        // channel weights in channel order -- so a zeroed buffer with only
        // channel 0's slot set is already correctly packed for any Cin in
        // 1..=8, the same as the Cin=1 tests already relied on.
        let buf_weights = Buffer::new(fd, PAGE_BYTES, file);
        ptr::write_bytes(buf_weights.host_ptr, 0, buf_weights.size);
        *(buf_weights.host_ptr as *mut u16) = 0x3c00;

        let buf_bias = Buffer::new(fd, PAGE_BYTES, file);
        ptr::write_bytes(buf_bias.host_ptr, 0, buf_bias.size);
        let buf_output = Buffer::new(fd, page_aligned_size(output_bytes), file);
        ptr::write_bytes(buf_output.host_ptr, 0, buf_output.size);

        let programs = plan.programs_with_buffers(Buffers {
            input: buf_input.dma_address,
            weights: buf_weights.dma_address,
            bias: buf_bias.dma_address,
            output: buf_output.dma_address,
        });
        let mut command_buffers = Vec::with_capacity(programs.len());
        for commands in &programs {
            let command_bytes = commands.len() * mem::size_of::<u64>();
            let buffer = Buffer::new(fd, page_aligned_size(command_bytes), file);
            ptr::write_bytes(buffer.host_ptr, 0, buffer.size);
            let words = std::slice::from_raw_parts_mut(buffer.host_ptr as *mut u64, commands.len());
            for (destination, command) in words.iter_mut().zip(commands.iter()) {
                *destination = command.0;
            }
            command_buffers.push((buffer, commands.len() as u32));
        }

        for handle in [
            buf_input.handle,
            buf_weights.handle,
            buf_bias.handle,
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
                    buf_bias.handle,
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
            .unwrap_or_else(|error| panic!("{width}x{height} SUBMIT failed: {error}"));

        let mut failure = Failure {
            mismatches: 0,
            samples: Vec::new(),
            timed_out: false,
            tiles,
            first_bad: None,
        };

        if let Err(error) = prep_bo(fd, buf_output.handle, timeout_ns) {
            failure.timed_out = true;
            failure
                .samples
                .push(format!("prep_bo did not complete: {error}"));
        } else {
            let raw = std::slice::from_raw_parts(buf_output.host_ptr, output_bytes);
            for y in 0..height as usize {
                let want = f16_to_f32(ROW_VALUES[y % ROW_VALUES.len()]);
                for x in 0..width as usize {
                    let offset = (y * width as usize + x) * FEATURE_ATOM_BYTES;
                    let got = f16_to_f32(u16::from_le_bytes([raw[offset], raw[offset + 1]]));
                    if got != want {
                        failure.mismatches += 1;
                        if failure.first_bad.is_none() {
                            failure.first_bad = Some((y, x));
                        }
                        if failure.samples.len() < 12 {
                            failure
                                .samples
                                .push(format!("[y={y}, x={x}] want {want} got {got}"));
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
        ] {
            let _ = close_bo(fd, handle);
        }
        for (buffer, _) in &command_buffers {
            let _ = close_bo(fd, buffer.handle);
        }

        if failure.mismatches == 0 && !failure.timed_out {
            Ok(())
        } else {
            Err(failure)
        }
    }
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn dense_row_order_survives_a_multi_tile_1x1_identity_conv() {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();

    let mut passed = 0;
    for i in 0..REPS {
        match run(fd, &file, WIDTH, MULTI_TILE_HEIGHT, 1, None, 5_000_000_000) {
            Ok(()) => {
                println!("rep {i}: ok");
                passed += 1;
            }
            Err(failure) => {
                println!(
                    "rep {i}: FAIL ({} mismatches, timed_out={}, tiles={}, first_bad={:?})",
                    failure.mismatches, failure.timed_out, failure.tiles, failure.first_bad
                );
                for sample in &failure.samples {
                    println!("         {sample}");
                }
            }
        }
    }

    println!("\n=== summary: dense_row_order_survives_a_multi_tile_1x1_identity_conv ===");
    println!("  {WIDTH}x{MULTI_TILE_HEIGHT}  {passed}/{REPS} passed");

    assert_eq!(
        passed, REPS,
        "row order broke across a tile boundary at least once -- see samples above \
         for which output row read back which input row's value"
    );
}

/// The first run of `dense_row_order_survives_a_multi_tile_1x1_identity_conv`
/// broke well inside tile 0 (tiles split at row 200; the failure started at
/// row 16, col 16 -- linear pixel 528), not at the tile boundary this file's
/// own doc comment was written to test. Diagnostic, not a correctness gate
/// (no assertion): sweeps `width` at fixed height=400 and reports each
/// width's first bad `(y, x)` and its row-major linear pixel index, to tell
/// apart a few hypotheses that all fit the one data point equally well --
/// a fixed row count (break always at y=16 regardless of width), a fixed
/// linear pixel/byte count (break always at linear index 528, so the row
/// shifts as width changes), or something tied to width itself (e.g. a
/// fixed column count, matching col=16 exactly half of WIDTH=32).
#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn dense_row_order_break_point_scales_with_width() {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();
    let height = 400;

    println!("\n=== break point vs width, height={height} fixed ===");
    for width in [8u32, 16, 32, 64, 128] {
        match run(fd, &file, width, height, 1, None, 5_000_000_000) {
            Ok(()) => println!("width={width:4}  ok (no break)"),
            Err(failure) => {
                let linear = failure.first_bad.map(|(y, x)| y * width as usize + x);
                println!(
                    "width={width:4}  tiles={}  mismatches={}  first_bad={:?}  linear_pixel={:?}",
                    failure.tiles, failure.mismatches, failure.first_bad, linear
                );
            }
        }
    }
}

/// A vendor capture (`~/projects/rknn-files/sweep/conv-w32-h256-k1-s1.rknn`,
/// Cin=3/Cout=8/32x256, the closest real vendor-compiled shape to this
/// file's own Cin=1/Cout=1/32x400) does the *entire* 256-row image in one
/// program (`feature_grains=256`, no row split at its 1-core plan) -- so a
/// large single-program row count is not inherently unsafe. But it does so
/// at `banks 8/4`, not the `11/1` the older `ConvPlan` picked automatically
/// for this file's historical failing shape. Every prior bug this investigation found
/// (`weight_banks<3` at large `Cin`) was about the *weight* side being
/// starved; `Cin=Cout=1`'s weight footprint here is 2 bytes, nowhere near
/// starved, so the mechanism can't be identical -- but an imbalanced split
/// is an imbalanced split, and this checks the same lever (explicit
/// `ConvPlan::with_cbuf_banks`, same escape hatch `weight_bank_floor_probe`
/// used) on the *data* side instead: does forcing a split closer to the
/// vendor's own (fewer data banks, more weight banks than the historical
/// automatic `11/1`) fix the corruption at this exact shape (32x400, break
/// at row 16 under that old automatic split)?
#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn dense_row_order_break_point_vs_cbuf_split() {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();

    println!("\n=== break point vs forced CBUF split, {WIDTH}x{HEIGHT} ===");
    for &(data_banks, weight_banks) in &[(11u32, 1u32), (9, 3), (8, 4), (6, 6), (4, 8), (1, 11)] {
        match run(
            fd,
            &file,
            WIDTH,
            HEIGHT,
            1,
            Some((data_banks, weight_banks)),
            5_000_000_000,
        ) {
            Ok(()) => println!("banks {data_banks:2}/{weight_banks:<2}  ok (no break)"),
            Err(failure) => {
                let linear = failure.first_bad.map(|(y, x)| y * WIDTH as usize + x);
                println!(
                    "banks {data_banks:2}/{weight_banks:<2}  tiles={}  mismatches={}  first_bad={:?}  linear_pixel={:?}",
                    failure.tiles, failure.mismatches, failure.first_bad, linear
                );
            }
        }
    }
}

/// The direct link back to where this whole investigation started:
/// `rocket_conv_harness.py` (a real-compiler-path harness in
/// iree-rocket-design-spike) found `features.0`'s exact shape (Cin=3,
/// Cout=64, 226x226, 3x3) producing a real, non-zero, deterministic
/// mismatch against CPU with genuine random data -- the finding that led
/// to this whole file. `features.0`'s shape automatically picks `banks
/// 11/1, tiles=5` (confirmed locally, matches DESIGN_NOTES.md's own VGG
/// table), each tile ~45-46 rows -- far past the ~16-row point
/// `dense_row_order_break_point_vs_cbuf_split` found corruption starting
/// at `11/1` for a smaller Cin. This runs the same per-row-distinguishable-
/// constant probe at `features.0`'s real Cin (3) and real spatial extent
/// (226x226), simplified to Cout=1 and a 1x1 kernel (channel order and
/// tap order are already independently confirmed elsewhere in this file
/// and in conv_dense_channel_order_hw.rs; this isolates row order at the
/// real shape's actual CBUF allocation) -- at the automatic `11/1` split,
/// at an explicit `1/11` (eliminated the corruption entirely for the
/// smaller Cin=1 case, but produces 57 tiny tiles here -- a first run
/// timed out at this file's normal 5-second `prep_bo` budget before ever
/// reading the output back, so this uses a longer one, purely for the
/// extra submit/wait round trips, not because 5 seconds of actual compute
/// was too tight), and at `8/4` -- the real vendor capture's own split for
/// Cin=3 (`~/projects/rknn-files/sweep/conv-w32-h256-k1-s1.rknn`, though at
/// a narrower width; 7 tiles here, confirmed locally, not 1).
#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn dense_row_order_at_features_0_real_shape() {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();
    let (width, height, cin) = (226u32, 226u32, 3u32);

    println!("\n=== features.0 real shape (Cin=3, 226x226): automatic vs forced split ===");
    let cases: [(&str, Option<(u32, u32)>, u64); 3] = [
        ("automatic (11/1 expected)", None, 5_000_000_000),
        (
            "forced 1/11 (57 tiles)",
            Some((1u32, 11u32)),
            30_000_000_000,
        ),
        (
            "forced 8/4 (vendor's Cin=3 split, 7 tiles)",
            Some((8, 4)),
            15_000_000_000,
        ),
    ];
    for (label, banks, timeout_ns) in cases {
        match run(fd, &file, width, height, cin, banks, timeout_ns) {
            Ok(()) => println!("{label}: ok (no break)"),
            Err(failure) => {
                let linear = failure.first_bad.map(|(y, x)| y * width as usize + x);
                println!(
                    "{label}: tiles={} mismatches={} first_bad={:?} linear_pixel={:?}",
                    failure.tiles, failure.mismatches, failure.first_bad, linear
                );
                for sample in &failure.samples {
                    println!("         {sample}");
                }
            }
        }
    }
}
