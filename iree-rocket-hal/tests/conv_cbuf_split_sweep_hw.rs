//! Sweeps the five distinct CBUF data/weight bank splits VGG-19's own conv
//! layers exercise, each using that layer's real (padded) shape, run in
//! isolation -- to map out which splits are actually correct on real
//! hardware versus which silently produce wrong (all-zero) output despite
//! completing without a timeout.
//!
//! `conv_features19_isolated_hw.rs` originally found that both
//! features.16's shape (1 data / 11 weight banks) and features.19's shape
//! (11 data / 1 weight) failed 100% deterministically -- but that was
//! traced to a bug in the test itself (an edge-clipping expected-value
//! formula left over from a different padding convention), not the
//! hardware. After fixing it, the sweep below passed 5/5 on every split,
//! and features.16's warmup burst passed 32/32 -- except features.19's
//! shape, which *still* fails deterministically (all-zero output) with the
//! corrected formula, both as a fresh first job and after sustained load +
//! an idle gap. So this is a real, narrow, shape-specific bug, not a
//! timing issue and not a broad split-level issue.
//!
//! The first five distinct-shaped rocket-eligible VGG layers
//! (features.0, .2, .5, .7, .10) between them cover all five splits that
//! appear anywhere in the network before the repeats start:
//!
//!   features.0   padded 226x226   3->64    11/1
//!   features.2   padded 226x226  64->64     9/3
//!   features.5   padded 114x114  64->128     7/5
//!   features.7   padded 114x114 128->128    3/9
//!   features.10  padded  58x58  128->256    1/11
//!
//! features.21 is added as a sixth point specifically to localize the
//! features.19 bug: same 30x30 padded extent, same 11/1 split, but
//! Cin=512 instead of features.19's Cin=256 (Cout=512 both times). If it
//! also comes back all-zero, the bug spans the whole Cin~256-512/Cout=512
//! region at this split (and the real live trace's apparent success on
//! features.21/23/25 was likely also silently wrong, just never caught --
//! the eBPF tracer can only see timeouts, not bad output). If it passes,
//! the bug is narrower than expected -- specific to Cin=256 rather than
//! high Cin generally, at this Cout and split.
//!
//!   features.19  padded  30x30  256->512    11/1  (known broken)
//!   features.21  padded  30x30  512->512    11/1  (this addition)
//!
//!   CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!     cargo test --target aarch64-unknown-linux-gnu --release \
//!       --test conv_cbuf_split_sweep_hw --no-run
//!
//!   ./conv_cbuf_split_sweep_hw-<hash> --ignored --nocapture

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

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

// Repeats per shape: enough to catch flakiness without taking long, given
// conv_features19_isolated_hw.rs already found perfect (0/32) determinism
// for its two shapes.
const REPS: usize = 5;

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

/// Built from ConvPlan directly -- the same path command_buffer.rs uses in
/// production (`ConvPlan::new(*shape, kernels).programs_with_buffers(bufs)`).
/// `banks`, when set, overrides the demand-based split via
/// `ConvPlan::with_cbuf_banks` -- the escape hatch `conv.rs` documents for
/// shapes its own formula does not settle -- to probe bank counts the
/// automatic partition would never choose for this shape on its own.
fn run(
    fd: i32,
    file: &std::fs::File,
    shape: Shape,
    kernels: Kernels,
    banks: Option<(u32, u32)>,
) -> Result<(), Failure> {
    let plan = match banks {
        Some((data_banks, weight_banks)) => {
            ConvPlan::with_cbuf_banks(shape, kernels, data_banks, weight_banks)
        }
        None => ConvPlan::new(shape, kernels),
    };
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
            // production dispatch -- see the module doc comment), every
            // output position reads a full, real kernel-sized window: there
            // is no synthetic zero boundary left for valid_taps' edge
            // clipping to apply to. An earlier version of this test kept
            // conv_outchannel_hw.rs's clipped formula unchanged despite
            // switching padding conventions, which produced a bogus
            // mismatch at every edge/corner output position while masking
            // that interior positions (already unclipped either way)
            // matched correctly.
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
                            if failure.samples.len() < 4 {
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

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn cbuf_split_sweep() {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();
    let kernels = [3, 3];

    // (label, padded extent, Cin, Cout) -- the first five distinct-shaped
    // rocket-eligible VGG layers (one per split), plus features.21 to
    // localize the features.19 bug (same 30x30/11/1 as features.19, but
    // Cin=512 instead of 256).
    let layers: &[(&str, u32, u32, u32)] = &[
        ("features.0", 226, 3, 64),
        ("features.2", 226, 64, 64),
        ("features.5", 114, 64, 128),
        ("features.7", 114, 128, 128),
        ("features.10", 58, 128, 256),
        ("features.21", 30, 512, 512),
    ];

    let mut summary = Vec::new();
    for (label, extent, cin, cout) in layers {
        let shape = Shape::with_out_channels(*extent, *extent, 1, *cin, *cout).with_padding([0, 0]);
        let plan = ConvPlan::new(shape, kernels);
        println!(
            "\n=== {label}: padded {extent}x{extent} {cin}->{cout}  banks {}/{}  tiles={} ===",
            plan.data_banks(),
            plan.weight_banks(),
            plan.tiles().len()
        );
        let mut passed = 0;
        for i in 0..REPS {
            match run(fd, &file, shape, kernels, None) {
                Ok(()) => {
                    println!("  rep {i}: ok");
                    passed += 1;
                }
                Err(failure) => {
                    println!(
                        "  rep {i}: FAIL ({} mismatches, timed_out={})",
                        failure.mismatches, failure.timed_out
                    );
                    for sample in &failure.samples {
                        println!("           {sample}");
                    }
                }
            }
        }
        summary.push((*label, plan.data_banks(), plan.weight_banks(), passed));
    }

    println!("\n=== summary: split -> pass rate ===");
    for (label, data_banks, weight_banks, passed) in &summary {
        println!("  {label:<14} banks {data_banks}/{weight_banks}  {passed}/{REPS} passed");
    }

    let broken: Vec<_> = summary
        .iter()
        .filter(|(_, _, _, passed)| *passed != REPS)
        .map(|(label, d, w, passed)| format!("{label} ({d}/{w}, {passed}/{REPS})"))
        .collect();
    assert!(
        broken.is_empty(),
        "these splits produced wrong output at least once: {}",
        broken.join(", ")
    );
}

/// `rocket_conv_harness.py` (a separate real-compiler-path harness in
/// iree-rocket-design-spike) found Cin=256/Cout=256/3x3 at 30x30 -- a shape
/// that never occurs in real VGG -- comes back deterministically all-zero on
/// real hardware, despite Cin=256/Cout=256 at 58x58 (features.16) being
/// hardware-validated safe above. `ConvPlan` picks a *different* CBUF split
/// for the two extents (`banks 1/11` at 58x58, `banks 11/1` at 30x30) even
/// though the channel counts and weight footprint are identical -- so the
/// `cbuf_split_sweep` test above, which covers one shape per split, was
/// silently assuming split alone determines correctness and never actually
/// tested Cin=256/Cout=256 at the 11/1 split it picks for smaller extents.
///
/// A local, hardware-free `ConvPlan` sweep (see
/// iree-rocket-hal/examples/harness_shape_check.rs) at fixed
/// Cin=256/Cout=256/3x3 found the split changes twice as extent grows:
///
///   20..24   banks 7/5, 8/4, 9/3   (labels seen before, at *different*
///                                    channel counts, in cbuf_split_sweep)
///   26..48   banks 11/1            (features.19/21's split; 30x30 already
///                                    confirmed broken via the harness)
///   50..     banks 1/11            (features.16's split; validated safe,
///                                    but only ever tested at exactly 58x58)
///
/// This sweeps representative points from each region, plus both sides of
/// each flip, to see whether the whole 11/1 region is broken or only part of
/// it, and whether the 1/11 region is safe everywhere or only near 58x58.
#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn extent_sweep_at_fixed_channels() {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();
    let kernels = [3, 3];
    let cin = 256;
    let cout = 256;

    // (label, extent) -- extent alone varies; Cin/Cout/kernel stay fixed.
    let points: &[(&str, u32)] = &[
        ("pre-flip-1 (7/5)", 20),
        ("pre-flip-1 (9/3)", 24),
        ("just-past-flip-1 (11/1)", 26),
        ("harness shape, known broken (11/1)", 30),
        ("mid 11/1", 36),
        ("last-before-flip-2 (11/1)", 48),
        ("just-past-flip-2 (1/11)", 50),
        ("features.16 extent, known safe (1/11)", 58),
    ];

    let mut summary = Vec::new();
    for (label, extent) in points {
        let shape = Shape::with_out_channels(*extent, *extent, 1, cin, cout).with_padding([0, 0]);
        let plan = ConvPlan::new(shape, kernels);
        println!(
            "\n=== {label}: {extent}x{extent} {cin}->{cout}  banks {}/{}  tiles={} ===",
            plan.data_banks(),
            plan.weight_banks(),
            plan.tiles().len()
        );
        let mut passed = 0;
        for i in 0..REPS {
            match run(fd, &file, shape, kernels, None) {
                Ok(()) => {
                    println!("  rep {i}: ok");
                    passed += 1;
                }
                Err(failure) => {
                    println!(
                        "  rep {i}: FAIL ({} mismatches, timed_out={})",
                        failure.mismatches, failure.timed_out
                    );
                    for sample in &failure.samples {
                        println!("           {sample}");
                    }
                }
            }
        }
        summary.push((
            *label,
            *extent,
            plan.data_banks(),
            plan.weight_banks(),
            passed,
        ));
    }

    println!("\n=== summary: extent -> split -> pass rate (Cin=Cout=256, 3x3) ===");
    for (label, extent, data_banks, weight_banks, passed) in &summary {
        println!(
            "  {label:<40} {extent:4}x{extent:<4} banks {data_banks}/{weight_banks}  {passed}/{REPS} passed"
        );
    }

    let broken: Vec<_> = summary
        .iter()
        .filter(|(_, _, _, _, passed)| *passed != REPS)
        .map(|(label, extent, d, w, passed)| {
            format!("{label} ({extent}x{extent}, {d}/{w}, {passed}/{REPS})")
        })
        .collect();
    assert!(
        broken.is_empty(),
        "these extents produced wrong output at least once: {}",
        broken.join(", ")
    );
}

/// DESIGN_NOTES.md "The vendor's own CBUF split formula disagrees with
/// `ConvPlan`'s, and never reaches `11/1`" -- a real-compiler-path harness
/// plus a full 8x8 `Cin`x`Cout` cross product through the vendor's own
/// `rknn-convert` toolchain (not `ConvPlan`, no hardware involved in that
/// sweep) found the vendor's own CBUF partition formula never picks
/// `weight_banks` below 3, at any of 75 shapes tried, while `ConvPlan`
/// readily picks `weight_banks=1` (`11/1`) -- exactly the split already
/// hardware-proven all-zero above, both for `features.19`/`features.21` and
/// for the same channel counts at other extents.
///
/// That is a floor observed in a *different* compiler's choices, not a
/// hardware fact by itself. This test checks it directly: one fixed shape
/// (Cin=Cout=256, 3x3, 30x30 -- the harness's already-broken point, same as
/// `features.19`'s extent), `weight_banks` forced explicitly via
/// `ConvPlan::with_cbuf_banks` across the entire range the RK3588's 12 CBUF
/// banks allow (1 through 6, i.e. `weight_banks` from the maximally
/// weight-starved split up to parity with `data_banks`), independent of
/// whatever split the demand-based formula would have picked for this shape
/// on its own. `weight_banks=1` is a known-broken repeat, included as an
/// in-test sanity check that this harness path reproduces the earlier
/// result; the real question is `weight_banks` 2 through 6.
#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn weight_bank_floor_probe() {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();
    let kernels = [3, 3];
    let shape = Shape::with_out_channels(30, 30, 1, 256, 256).with_padding([0, 0]);

    let mut summary = Vec::new();
    for weight_banks in 1..=6u32 {
        let data_banks = 12 - weight_banks;
        println!("\n=== weight_banks={weight_banks} data_banks={data_banks} (forced) ===");
        let mut passed = 0;
        for i in 0..REPS {
            match run(fd, &file, shape, kernels, Some((data_banks, weight_banks))) {
                Ok(()) => {
                    println!("  rep {i}: ok");
                    passed += 1;
                }
                Err(failure) => {
                    println!(
                        "  rep {i}: FAIL ({} mismatches, timed_out={})",
                        failure.mismatches, failure.timed_out
                    );
                    for sample in &failure.samples {
                        println!("           {sample}");
                    }
                }
            }
        }
        summary.push((weight_banks, data_banks, passed));
    }

    println!("\n=== summary: forced weight_banks -> pass rate (Cin=Cout=256, 3x3, 30x30) ===");
    for (weight_banks, data_banks, passed) in &summary {
        println!("  weight_banks={weight_banks} data_banks={data_banks}  {passed}/{REPS} passed");
    }

    let broken: Vec<_> = summary
        .iter()
        .filter(|(_, _, passed)| *passed != REPS)
        .map(|(w, d, passed)| format!("weight_banks={w} (data_banks={d}, {passed}/{REPS})"))
        .collect();
    assert!(
        broken.is_empty(),
        "these forced weight_banks counts produced wrong output at least once: {}",
        broken.join(", ")
    );
}

/// Closes the loop `weight_bank_floor_probe` left open: that test forced
/// `weight_banks` explicitly via `ConvPlan::with_cbuf_banks`, so it never
/// exercised the fixed *automatic* formula (`WEIGHT_BANKS_FLOOR` in
/// `conv.rs`, applied inside `demand_based_cbuf_partition`) on the two real
/// shapes that started this whole investigation. Both `features.19`
/// (Cin=256, Cout=512) and `features.21` (Cin=512, Cout=512) were
/// hardware-proven all-zero at their old automatic split (`11/1`) earlier in
/// this file; this checks two things at once with `ConvPlan::new` and no
/// override: that the fixed formula actually moves them off `11/1` at all,
/// and that wherever it lands is correct on real hardware, not just
/// plausible from the vendor's own formula agreeing in spirit.
///
/// A local (non-hardware) check found the fixed formula puts *both* shapes
/// at `9/3` -- not `features.21`'s vendor-preferred `3/9` from
/// DESIGN_NOTES.md's rknn sweep, since the fix only clamps up to the
/// hardware-confirmed floor of 3, it does not reproduce the vendor's fuller
/// `Cin`-dependent curve. `9/3` at `Cin=512` is therefore its own claim, not
/// an inherited one from the `Cin=Cout=256` probe.
#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn fixed_formula_resolves_features_19_and_21() {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();
    let kernels = [3, 3];

    let layers: &[(&str, u32, u32)] = &[("features.19", 256, 512), ("features.21", 512, 512)];

    let mut summary = Vec::new();
    for (label, cin, cout) in layers {
        let shape = Shape::with_out_channels(30, 30, 1, *cin, *cout).with_padding([0, 0]);
        let plan = ConvPlan::new(shape, kernels);
        let (data_banks, weight_banks) = (plan.data_banks(), plan.weight_banks());
        println!(
            "\n=== {label}: Cin={cin} Cout={cout} 30x30, automatic banks {data_banks}/{weight_banks} ==="
        );
        assert_ne!(
            weight_banks, 1,
            "{label} still lands on the pre-fix weight_banks=1 -- WEIGHT_BANKS_FLOOR did not move it"
        );
        let mut passed = 0;
        for i in 0..REPS {
            match run(fd, &file, shape, kernels, None) {
                Ok(()) => {
                    println!("  rep {i}: ok");
                    passed += 1;
                }
                Err(failure) => {
                    println!(
                        "  rep {i}: FAIL ({} mismatches, timed_out={})",
                        failure.mismatches, failure.timed_out
                    );
                    for sample in &failure.samples {
                        println!("           {sample}");
                    }
                }
            }
        }
        summary.push((*label, data_banks, weight_banks, passed));
    }

    println!("\n=== summary: fixed automatic formula on the real VGG-19 shapes ===");
    for (label, data_banks, weight_banks, passed) in &summary {
        println!("  {label:<12} banks {data_banks}/{weight_banks}  {passed}/{REPS} passed");
    }

    let broken: Vec<_> = summary
        .iter()
        .filter(|(_, _, _, passed)| *passed != REPS)
        .map(|(label, d, w, passed)| format!("{label} ({d}/{w}, {passed}/{REPS})"))
        .collect();
    assert!(
        broken.is_empty(),
        "the fixed automatic formula still produced wrong output at least once: {}",
        broken.join(", ")
    );
}
