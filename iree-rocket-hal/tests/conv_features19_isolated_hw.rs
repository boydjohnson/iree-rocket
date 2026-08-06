//! Isolated hardware probe for the exact shape (`Cin=256, Cout=512`, padded
//! 30x30, 3x3, fp16) that a live rocket-npu-trace showed wedging the NPU
//! after 714 prior clean jobs -- always as job #715, always right after an
//! anomalously long (20-52ms) idle gap on the failing core, vs. sub-1ms
//! gaps everywhere else in steady state. A register-level diff against
//! features.10/features.16 (which succeeded) and against this shape's own
//! second tile found no malformed field -- everything differs only in
//! ways fully explained by ordinary shape-dependent formulas. So the two
//! remaining hypotheses this test is built to separate are:
//!
//!   1. This shape is simply broken regardless of context (register
//!      program looks fine statically, but something about it still
//!      doesn't work on real hardware).
//!   2. It only breaks after sustained prior activity plus an idle gap --
//!      i.e. a timing/power-state issue, not a shape-specific one.
//!
//! `features19_runs_clean_as_first_job` tests (1): the shape run
//! completely fresh, as the very first submission after opening the
//! device, no preceding load.
//!
//! `features19_survives_burst_then_idle_gap` tests (2): a burst of
//! features.16-shaped jobs (same family that succeeded for real, 168
//! tiles worth of submissions between them) to put the NPU under
//! sustained load, then a deliberate sleep spanning the 20-52ms idle
//! window the trace showed, then features.19.
//!
//! If (1) fails, the shape itself needs a different CBUF split or is
//! outside what this builder can plan correctly. If (1) passes but (2)
//! fails, that's a strong, directly-reproduced confirmation of the
//! timing/idle hypothesis over the shape hypothesis.
//!
//!   CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!     cargo test --target aarch64-unknown-linux-gnu --release \
//!       --test conv_features19_isolated_hw --no-run
//!
//!   ./conv_features19_isolated_hw-<hash> --ignored --nocapture

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr, time::Duration};

use iree_rocket_hal::rocket::{
    conv::{Buffers, ConvPlan, FeatureLayout, Kernels, Shape},
    device::{Buffer, JobDesc, close_bo, fini_bo, prep_bo, submit_jobs},
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const FP16_BYTES: usize = 2;
const FEATURE_ATOM_BYTES: usize = 16;
const CHANNELS_PER_ATOM: usize = 8;
const PAGE_BYTES: usize = 4096;
const FP16_ONE: u16 = 0x3c00;

// The real Cin=256/Cout=512, padded 30x30 shape from features.19 -- see
// vgg_cbuf_split.rs's doc comment for why the padded (not logical) extent
// is what's actually dispatched at runtime.
const FEATURES_19_EXTENT: u32 = 30;
const FEATURES_19_CIN: u32 = 256;
const FEATURES_19_COUT: u32 = 512;

// The shape that DID succeed in the real trace, immediately before
// features.19's block transition -- used here purely as synthetic load to
// put the NPU under sustained activity before the idle gap.
const WARMUP_EXTENT: u32 = 58;
const WARMUP_CIN: u32 = 256;
const WARMUP_COUT: u32 = 256;
const WARMUP_BURST_JOBS: usize = 32;

// Spans the 20.0ms-51.69ms idle_before window both real timeouts followed.
const IDLE_GAP: Duration = Duration::from_millis(60);

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

/// Writes the input feature map, real channels 1.0 and padding zero.
unsafe fn fill_input(base: *mut u8, size: usize, shape: Shape, surfaces: usize) {
    unsafe {
        ptr::write_bytes(base, 0, size);
        let width = shape.width as usize;
        let height = shape.height as usize;
        for channel in 0..shape.in_channels as usize {
            let surface = channel / CHANNELS_PER_ATOM;
            let lane = channel % CHANNELS_PER_ATOM;
            if surface >= surfaces {
                continue;
            }
            for y in 0..height {
                for x in 0..width {
                    let offset = match shape.layout() {
                        FeatureLayout::Dense => {
                            (y * width + x) * shape.in_channels as usize * FP16_BYTES
                                + channel * FP16_BYTES
                        }
                        FeatureLayout::Surfaces => {
                            surface * width * height * FEATURE_ATOM_BYTES
                                + (y * width + x) * FEATURE_ATOM_BYTES
                                + lane * FP16_BYTES
                        }
                    };
                    ptr::write((base.add(offset)) as *mut u16, FP16_ONE);
                }
            }
        }
    }
}

struct Failure {
    mismatches: usize,
    samples: Vec<String>,
    timed_out: bool,
}

/// Submits one convolution end to end (fresh buffers, fresh command
/// streams every call, matching the real driver's per-dispatch scratch
/// allocation) and checks its output. `timeout_ns` is kept generous (5s)
/// rather than the driver's real ~500ms watchdog, so a hung job here is
/// unambiguous rather than racing the same timeout the bug itself trips.
fn run(fd: i32, file: &std::fs::File, shape: Shape, kernels: Kernels) -> Result<(), Failure> {
    // Built from ConvPlan directly -- the exact same path
    // rocket-hal-driver/src/command_buffer.rs uses in production
    // (`ConvPlan::new(*shape, kernels).programs_with_buffers(bufs)`).
    // An earlier version of this test built tiles via
    // `Shape::min_tiles`/`Tile::split` instead (matching
    // conv_outchannel_hw.rs's older pattern) and that undercounts tiles
    // for shapes like features.16 (56 vs ConvPlan's real 168) -- confirmed
    // by direct comparison in examples/verify_tiling_match.rs. That
    // mismatch means oversized tiles whose input rows exceed what the
    // actual CBUF data-bank allocation holds, which doesn't fault, it
    // silently drops rows (see conv_outchannel_hw.rs's own doc comment on
    // this exact failure mode) -- so results from that version aren't
    // trustworthy evidence about production behavior, only about the test
    // harness's own bug. ConvPlan's tiling is CBUF-capacity-correct by
    // construction, so this version's tile count is not itself in question.
    let plan = ConvPlan::new(shape, kernels);
    eprintln!(
        "  [run] {}x{} {}->{} via ConvPlan: {} tile(s), banks {}/{}",
        shape.width,
        shape.height,
        shape.in_channels,
        shape.out_channels,
        plan.tiles().len(),
        plan.data_banks(),
        plan.weight_banks(),
    );
    let width = shape.width as usize;
    let height = shape.height as usize;
    let out_width = shape.output_width(kernels) as usize;
    let out_height = shape.output_height(kernels) as usize;
    let in_surfaces = (shape.weight_channels() / 8) as usize;
    let out_surfaces = (shape.padded_out_channels() / 8) as usize;

    let input_bytes = match shape.layout() {
        FeatureLayout::Dense => width * height * shape.in_channels as usize * FP16_BYTES,
        FeatureLayout::Surfaces => in_surfaces * width * height * FEATURE_ATOM_BYTES,
    };
    let output_bytes = out_surfaces * out_width * out_height * FEATURE_ATOM_BYTES;

    unsafe {
        let buf_input = Buffer::new(fd, page_aligned_size(input_bytes), file);
        fill_input(buf_input.host_ptr, buf_input.size, shape, in_surfaces);

        let weight_bytes = shape.weight_bytes(kernels) as usize;
        let buf_weights = Buffer::new(fd, page_aligned_size(weight_bytes), file);
        ptr::write_bytes(buf_weights.host_ptr, 0, buf_weights.size);
        std::slice::from_raw_parts_mut(buf_weights.host_ptr as *mut u16, weight_bytes / 2)
            .fill(FP16_ONE);

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
            .unwrap_or_else(|error| panic!("{shape:?} {kernels:?} SUBMIT failed: {error}"));

        let mut failure = Failure {
            mismatches: 0,
            samples: Vec::new(),
            timed_out: false,
        };

        if let Err(error) = prep_bo(fd, buf_output.handle, 5_000_000_000) {
            failure.timed_out = true;
            failure
                .samples
                .push(format!("prep_bo did not complete: {error}"));
        } else {
            let raw = std::slice::from_raw_parts(buf_output.host_ptr, output_bytes);
            // With explicit padding=[0,0] and an input buffer that already
            // covers the full physically-padded extent (matching real
            // production dispatch), every output position reads a full,
            // real kernel-sized window: there is no synthetic zero
            // boundary left for edge clipping to apply to. See
            // conv_cbuf_split_sweep_hw.rs's matching fix for the full
            // story -- an earlier version of this test kept
            // conv_outchannel_hw.rs's clipped-edge formula unchanged
            // despite switching padding conventions, which produced a
            // bogus mismatch at every edge/corner position.
            let want = (shape.in_channels as usize * kernels[0] * kernels[1]) as f32;
            for y in 0..out_height {
                for x in 0..out_width {
                    for channel in 0..shape.out_channels as usize {
                        let surface = channel / CHANNELS_PER_ATOM;
                        let lane = channel % CHANNELS_PER_ATOM;
                        let offset = surface * out_width * out_height * FEATURE_ATOM_BYTES
                            + (y * out_width + x) * FEATURE_ATOM_BYTES
                            + lane * FP16_BYTES;
                        let got = f16_to_f32(u16::from_le_bytes([raw[offset], raw[offset + 1]]));
                        if got != want {
                            failure.mismatches += 1;
                            if failure.samples.len() < 8 {
                                failure
                                    .samples
                                    .push(format!("[{y}, {x}, {channel}] want {want} got {got}"));
                            }
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

fn report(label: &str, result: Result<(), Failure>) -> bool {
    match result {
        Ok(()) => {
            println!("  ok   {label}");
            true
        }
        Err(failure) if failure.timed_out => {
            println!("  FAIL {label}: TIMED OUT (did not complete inside 5s)");
            for sample in &failure.samples {
                println!("         {sample}");
            }
            false
        }
        Err(failure) => {
            println!("  FAIL {label}: {} mismatches", failure.mismatches);
            for sample in &failure.samples {
                println!("         {sample}");
            }
            false
        }
    }
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn features19_runs_clean_as_first_job() {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();

    let shape = Shape::with_out_channels(
        FEATURES_19_EXTENT,
        FEATURES_19_EXTENT,
        1,
        FEATURES_19_CIN,
        FEATURES_19_COUT,
    )
    .with_padding([0, 0]);

    let ok = report(
        "features.19 shape, fresh process, first job, no preceding load",
        run(fd, &file, shape, [3, 3]),
    );
    assert!(
        ok,
        "features.19's shape failed even as the very first job -- points at the shape/CBUF split itself, not timing"
    );
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn features19_survives_burst_then_idle_gap() {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();

    let warmup_shape =
        Shape::with_out_channels(WARMUP_EXTENT, WARMUP_EXTENT, 1, WARMUP_CIN, WARMUP_COUT)
            .with_padding([0, 0]);

    println!(
        "  submitting {WARMUP_BURST_JOBS} warmup jobs ({}x{} {}->{}) to simulate sustained load...",
        WARMUP_EXTENT, WARMUP_EXTENT, WARMUP_CIN, WARMUP_COUT
    );
    // Deliberately does not stop at the first failure: this shape genuinely
    // succeeded (168 tiles, correctly) 168 times over in the real VGG
    // trace, so if it fails here too, the interesting question is whether
    // *only* the very first job ever submitted to a freshly opened device
    // fails (a cold-start/init quirk) or whether every single instance
    // fails identically (a deterministic per-shape bug). Panicking on job 0
    // would hide that distinction entirely.
    let mut warmup_outcomes = Vec::with_capacity(WARMUP_BURST_JOBS);
    for i in 0..WARMUP_BURST_JOBS {
        match run(fd, &file, warmup_shape, [3, 3]) {
            Ok(()) => {
                println!("  warmup job {i}: ok");
                warmup_outcomes.push(true);
            }
            Err(failure) => {
                println!(
                    "  warmup job {i}: FAIL ({} mismatches, timed_out={})",
                    failure.mismatches, failure.timed_out
                );
                warmup_outcomes.push(false);
            }
        }
    }
    let passed = warmup_outcomes.iter().filter(|ok| **ok).count();
    println!(
        "  warmup summary: {passed}/{WARMUP_BURST_JOBS} passed -- outcomes: {warmup_outcomes:?}"
    );
    println!("  warmup burst complete, sleeping {IDLE_GAP:?} to reproduce the idle gap...");
    std::thread::sleep(IDLE_GAP);

    let features_19_shape = Shape::with_out_channels(
        FEATURES_19_EXTENT,
        FEATURES_19_EXTENT,
        1,
        FEATURES_19_CIN,
        FEATURES_19_COUT,
    )
    .with_padding([0, 0]);

    let ok = report(
        "features.19 shape, after burst + idle gap",
        run(fd, &file, features_19_shape, [3, 3]),
    );
    assert!(
        ok,
        "features.19's shape failed after a burst + idle gap -- reproduces the live-trace wedge directly, confirming the timing/idle hypothesis"
    );
}
