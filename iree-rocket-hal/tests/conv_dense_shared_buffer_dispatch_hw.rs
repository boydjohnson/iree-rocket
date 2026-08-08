//! Isolated hardware test: does `rocket-hal-driver`'s *actual* multi-tile
//! dispatch pattern -- not `iree-rocket-hal`'s own `submit_jobs` -- survive
//! a shape that needs several row tiles?
//!
//! `rocket_conv_harness.py` (iree-rocket-design-spike, `run1/`) found the
//! real `iree-compile` -> `rocket-hal-driver` -> `iree-run-module` pipeline
//! producing a deterministic, non-zero, wrong result for Cin=3 (dense/ARGB),
//! Cout=256, 3x3, 226x226 logical output -- and the corruption starts
//! exactly at the first row of `ConvPlan`'s sixth and *last* row tile
//! (`out_first=189`), with every row before it correct. See
//! DESIGN_NOTES.md, "The dense (Cin<=4) ARGB path silently corrupts
//! multi-row dispatches" and its follow-up sections, for the full
//! characterization.
//!
//! Every existing hand-rolled hw test in this crate -- including
//! `conv_dense_row_order_hw.rs`, which already probes this exact bug class
//! at smaller scale -- submits its tiles via `device::submit_jobs`: every
//! job goes into *one* `DRM_ROCKET_SUBMIT` ioctl call, then a single
//! `prep_bo` waits for all of them. That is not what production does.
//! `rocket-hal-driver/src/device.rs`'s `queue_execute` submits one tile at a
//! time -- `device::submit` followed immediately by `device::prep_bo`, in a
//! loop, all tiles sharing one driver-private scratch output buffer -- and
//! the comment already sitting on that loop documents a *related*,
//! previously-found RK3588 kernel-driver reliability issue (multi-task
//! transitions within one job are unreliable; the fix was to submit
//! individually-fenced jobs instead). Nothing has hardware-tested that
//! specific sequential-submit-on-a-shared-buffer pattern at 6 tiles. This
//! test reproduces it directly, using this crate's own primitives
//! (`device::submit` + `device::prep_bo` in a loop, one shared output BO),
//! rather than going through the real IREE HAL / `rocket-hal-driver` code
//! at all -- so a pass or fail here is attributable to this dispatch
//! *pattern* alone, independent of anything else in that driver crate.
//!
//! The shape matches `run1` exactly: `ConvPlan::new` for Cin=3 (dense),
//! Cout=256, 3x3, physically-padded 228x228 input (matching
//! `RocketTarget.cpp`'s convention of a pre-padded input and
//! `Shape::padding=[0,0]` -- see `rocket_conv_harness.py`'s own doc
//! comment) picks `banks 10/2`, 6 row tiles, `out_first` = 0, 38, 76, 114,
//! 152, 189.
//!
//! The weight is a one-hot identity: only the center tap (`ky=1, kx=1`),
//! input channel 0, output channel 0, is 1.0; every other coefficient
//! (including the other 8 taps and channels 1-255) is zero. With
//! `padding=[0,0]` the center tap of a 3x3 kernel reads input row `y+1`
//! (no boundary case, since it is never in the zero-padding region for any
//! output row) -- so output row `y`, channel 0, must equal input row `y+1`,
//! for every column, regardless of tiling. This keeps the same
//! spatial-tap-order and channel-order independence
//! `conv_dense_row_order_hw.rs` relies on, while exercising the real 3x3 /
//! Cout=256 register geometry and CBUF footprint (which depend on `kh*kw*
//! Cout`, not on which weights are actually nonzero) -- unlike the existing
//! tests, which never varied `Cout` above 1.
//!
//! Each physically-padded input row is filled with one of three
//! distinguishable fp16 constants, cycling by `row % 3`, so a row-order or
//! tile-boundary bug shows up as a wrong phase of the cycle rather than an
//! aggregate magnitude mismatch. Only output channel 0 is checked (weight
//! feeds nothing else), read directly out of the raw NC1HWC2 scratch layout
//! -- like every other hand-rolled hw test here, this bypasses
//! `compact_atomic_output` entirely.
//!
//!   CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!     cargo test --target aarch64-unknown-linux-gnu --release \
//!       --test conv_dense_shared_buffer_dispatch_hw --no-run
//!
//!   ./conv_dense_shared_buffer_dispatch_hw-<hash> --ignored --nocapture

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    conv::{Buffers, ConvPlan, Kernels, Shape},
    device::{Buffer, close_bo, fini_bo, prep_bo, submit},
    tensor_layout::{pack_hwcf_to_rocket_weights, rocket_weight_storage_size},
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const FP16_BYTES: usize = 2;
const FEATURE_ATOM_BYTES: usize = 16;
const PAGE_BYTES: usize = 4096;
const PER_TILE_TIMEOUT_NS: u64 = 5_000_000_000;

// Matches `run1` (iree-rocket-design-spike) and `features.0`-scale
// production shapes: Cin=3 (dense/ARGB), Cout=256, 3x3, physically-padded
// 228x228 input -> logical 226x226 output. `ConvPlan::new` picks banks
// 10/2, 6 row tiles, at this exact shape (checked with `vgg_cbuf_split`'s
// technique before writing this test).
const CIN: u32 = 3;
const COUT: u32 = 256;
const PADDED: u32 = 228;
const OUTPUT: u32 = 226; // PADDED - kernel + 1, with padding=[0,0].
const KERNEL: usize = 3;

// fp16 1.0, 2.0, 3.0 -- same distinguishable-constant convention as
// conv_dense_row_order_hw.rs, cycled by physically-padded input row.
const ROW_VALUES: [u16; 3] = [0x3c00, 0x4000, 0x4200];
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
    failing_tile: Option<usize>,
    tiles: usize,
    /// (y, x) of the first mismatch, row-major scan order over the logical
    /// 226x226 output -- the number that should land on `out_first=189` if
    /// this reproduces `run1`'s break.
    first_bad: Option<(usize, usize)>,
}

fn run(fd: i32, file: &std::fs::File) -> Result<(), Failure> {
    let kernels: Kernels = [KERNEL, KERNEL];
    let shape = Shape::with_out_channels(PADDED, PADDED, 1, CIN, COUT).with_padding([0, 0]);
    debug_assert!(matches!(
        shape.layout(),
        iree_rocket_hal::rocket::conv::FeatureLayout::Dense
    ));
    assert_eq!(shape.output_width(kernels), OUTPUT);
    assert_eq!(shape.output_height(kernels), OUTPUT);

    let plan = ConvPlan::new(shape, kernels);
    let tiles = plan.tiles().len();

    unsafe {
        // Physically-padded dense NHWC input: every row filled with one of
        // three distinguishable constants, cycling by row.
        let input_bytes = PADDED as usize * PADDED as usize * CIN as usize * FP16_BYTES;
        let buf_input = Buffer::new(fd, page_aligned_size(input_bytes), file);
        let input_words =
            std::slice::from_raw_parts_mut(buf_input.host_ptr as *mut u16, input_bytes / 2);
        for y in 0..PADDED as usize {
            let value = ROW_VALUES[y % ROW_VALUES.len()];
            for x in 0..PADDED as usize {
                for c in 0..CIN as usize {
                    input_words[(y * PADDED as usize + x) * CIN as usize + c] = value;
                }
            }
        }

        // One-hot identity weight: center tap (ky=1, kx=1), input channel 0,
        // output channel 0 = 1.0fp16; everything else zero. Dense HWCF
        // layout, matching rocket_conv_harness.py's ONNX weight tensor and
        // pack_hwcf_to_rocket_weights's expected input.
        let dense_weight_elems = KERNEL * KERNEL * CIN as usize * COUT as usize;
        let mut dense_weights = vec![0u16; dense_weight_elems];
        let center_index = (1 * KERNEL + 1) * CIN as usize * COUT as usize + 0 * COUT as usize + 0;
        dense_weights[center_index] = 0x3c00; // fp16 1.0
        let dense_weight_bytes: Vec<u8> =
            dense_weights.iter().flat_map(|w| w.to_le_bytes()).collect();

        let packed_weight_bytes =
            rocket_weight_storage_size(KERNEL, KERNEL, CIN as usize, COUT as usize, FP16_BYTES)
                .expect("weight storage size");
        let buf_weights = Buffer::new(fd, page_aligned_size(packed_weight_bytes), file);
        ptr::write_bytes(buf_weights.host_ptr, 0, buf_weights.size);
        let weight_dst = std::slice::from_raw_parts_mut(buf_weights.host_ptr, packed_weight_bytes);
        pack_hwcf_to_rocket_weights(
            &dense_weight_bytes,
            KERNEL,
            KERNEL,
            CIN as usize,
            COUT as usize,
            FP16_BYTES,
            weight_dst,
        )
        .expect("weight packing");

        let buf_bias = Buffer::new(fd, page_aligned_size(shape.bs_buffer_bytes()), file);
        ptr::write_bytes(buf_bias.host_ptr, 0, buf_bias.size);

        // One shared output scratch buffer for every tile -- exactly
        // command_buffer.rs's `Conv2d` dispatch (one `scratch` per whole
        // convolution, all tile programs write into it at their own
        // relocated offset).
        let output_bytes = shape.output_scratch_bytes(kernels);
        let buf_output = Buffer::new(fd, page_aligned_size(output_bytes), file);
        ptr::write_bytes(buf_output.host_ptr, 0, buf_output.size);

        let programs = plan.programs_with_buffers(Buffers {
            input: buf_input.dma_address,
            weights: buf_weights.dma_address,
            bias: buf_bias.dma_address,
            output: buf_output.dma_address,
        });
        assert_eq!(programs.len(), tiles);

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

        let out_handles = [buf_output.handle];
        let mut failure = Failure {
            mismatches: 0,
            samples: Vec::new(),
            timed_out: false,
            failing_tile: None,
            tiles,
            first_bad: None,
        };

        // The pattern under test: rocket-hal-driver's device.rs submits and
        // waits for one tile at a time, sequentially, all of them sharing
        // this one output BO handle -- not device::submit_jobs's batched
        // one-ioctl-for-everything approach every other tiled hw test here
        // uses.
        'tiles: for (index, (buffer, count)) in command_buffers.iter().enumerate() {
            let in_handles = [
                buffer.handle,
                buf_input.handle,
                buf_weights.handle,
                buf_bias.handle,
            ];
            if submit(fd, buffer.dma_address, *count, &in_handles, &out_handles).is_err() {
                failure.timed_out = true;
                failure.failing_tile = Some(index);
                failure.samples.push(format!("tile {index}: SUBMIT failed"));
                break 'tiles;
            }
            if prep_bo(fd, buf_output.handle, PER_TILE_TIMEOUT_NS).is_err() {
                failure.timed_out = true;
                failure.failing_tile = Some(index);
                failure
                    .samples
                    .push(format!("tile {index}: prep_bo did not complete"));
                break 'tiles;
            }
        }

        if failure.timed_out {
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
            return Err(failure);
        }

        // Raw NC1HWC2 scratch layout, channel 0 of surface 0 only: byte
        // offset `(y * OUTPUT + x) * FEATURE_ATOM_BYTES`, same formula
        // command_buffer.rs's `compact_atomic_output` uses for its first
        // surface, bypassed here like every other hand-rolled hw test.
        let raw = std::slice::from_raw_parts(buf_output.host_ptr, output_bytes);
        for y in 0..OUTPUT as usize {
            // Center tap (ky=1) with padding=[0,0] reads physically-padded
            // input row y+1.
            let want = f16_to_f32(ROW_VALUES[(y + 1) % ROW_VALUES.len()]);
            for x in 0..OUTPUT as usize {
                let offset = (y * OUTPUT as usize + x) * FEATURE_ATOM_BYTES;
                let got = f16_to_f32(u16::from_le_bytes([raw[offset], raw[offset + 1]]));
                if got != want {
                    failure.mismatches += 1;
                    if failure.first_bad.is_none() {
                        failure.first_bad = Some((y, x));
                    }
                    if failure.samples.len() < 16 {
                        failure
                            .samples
                            .push(format!("[y={y}, x={x}] want {want} got {got}"));
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

        if failure.mismatches == 0 {
            Ok(())
        } else {
            Err(failure)
        }
    }
}

/// `ConvPlan`'s tile boundaries for this exact shape, printed once so a
/// failing row can be checked against them by eye without a separate dump
/// tool. `run1` broke at exactly `out_first=189`, the start of tile 5 (the
/// sixth and last).
#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn print_tile_plan() {
    let kernels: Kernels = [KERNEL, KERNEL];
    let shape = Shape::with_out_channels(PADDED, PADDED, 1, CIN, COUT).with_padding([0, 0]);
    let plan = ConvPlan::new(shape, kernels);
    println!(
        "banks {}/{} tiles={}",
        plan.data_banks(),
        plan.weight_banks(),
        plan.tiles().len()
    );
    for (i, tile) in plan.tiles().iter().enumerate() {
        println!("  tile {i}: {:?}", tile.rows);
    }
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn dense_shared_buffer_dispatch_survives_run1_shape() {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();

    let mut passed = 0;
    for i in 0..REPS {
        match run(fd, &file) {
            Ok(()) => {
                println!("rep {i}: ok");
                passed += 1;
            }
            Err(failure) => {
                println!(
                    "rep {i}: FAIL ({} mismatches, timed_out={}, failing_tile={:?}, tiles={}, first_bad={:?})",
                    failure.mismatches,
                    failure.timed_out,
                    failure.failing_tile,
                    failure.tiles,
                    failure.first_bad
                );
                for sample in &failure.samples {
                    println!("         {sample}");
                }
            }
        }
    }

    println!("\n=== summary: dense_shared_buffer_dispatch_survives_run1_shape ===");
    println!(
        "  Cin={CIN} Cout={COUT} {KERNEL}x{KERNEL} padded {PADDED}x{PADDED}  {passed}/{REPS} passed"
    );
    println!(
        "  compare first_bad's y against tile boundaries from print_tile_plan (run1 broke at y=189, \
         the start of tile 5 of 6) -- an exact match there implicates this dispatch pattern; \
         no failure, or a failure at a different row, points back at ConvPlan/the register program instead"
    );

    assert_eq!(
        passed, REPS,
        "sequential submit+prep_bo on a shared output buffer broke at least once -- see samples \
         above for which output row read back wrong, and print_tile_plan for tile boundaries"
    );
}
