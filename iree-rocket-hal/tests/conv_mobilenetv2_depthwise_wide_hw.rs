//! Isolated hardware probes for two depthwise channel counts beyond
//! `MAX_OUTPUT_CHANNELS`/`MAX_INPUT_CHANNELS`'s current 512: MobileNetV2's
//! two expand-ratio-6 depthwise stages, Cin=Cout=576 and Cin=Cout=960, both
//! 3x3, both stride 1. These are the real shapes behind
//! `main_graph$async_dispatch_68` (576, 14x14) and `_dispatch_77` (960,
//! 7x7) in `mobilenetv2.vmfb`.
//!
//! STATUS: both tests here passed on real RK3588 hardware (0 mismatches,
//! first job, fresh device) when `MAX_OUTPUT_CHANNELS`/`MAX_INPUT_CHANNELS`
//! were temporarily raised to 960 to let their `Shape`s construct. That
//! raise was reverted (2026-08-28): those constants are shared between
//! dense and depthwise construction, and
//! `tests/conv_vendor_fixture_channels_768.rs` caught a real `ConvPlan`
//! divergence from vendor ground truth for *dense* shapes in the same
//! 513..960 range (predicted CBUF split 1/11, real vendor 6/6, 5/7, 4/8 --
//! see that constant's doc comment). As shipped, these two `#[ignore]`d
//! tests will panic immediately at `Shape::with_out_channels` (channel cap
//! assert), before ever touching hardware -- that's expected, not a
//! regression, until a depthwise-specific ceiling is added and wired into
//! `Shape`/`ConvPlan` separately from the dense-shared constants. The
//! hardware result itself is still valid; only the plumbing to reach 576/960
//! is currently reverted. Even once that's back, the matcher that would
//! route these shapes to Rocket
//! (`rocket_conv2d_transform_spec.mlir`'s `match_dynamic_depthwise_conv2d_nchw_3x3`)
//! is deliberately left at `umax = 512` for an unrelated reason: it measured
//! as a net latency regression on MobileNetV2 (161-163ms -> 165-170ms, see
//! that matcher's doc comment) due to per-dispatch layout-repack overhead,
//! so raising the Rust ceiling alone would not be enough to make this
//! matter for MobileNetV2 even if re-added.
//!
//! A third real shape, `_dispatch_74` (Cin=Cout=576, 14x14 -> 7x7,
//! **stride 2**), is deliberately NOT covered here even though it also
//! exceeds 512. `ConvPlan::new` hard-panics for it today ("convolution
//! needs horizontal tiling, which is only capture-backed at stride 1",
//! conv.rs:~1909): at 16-wide padded input, 576 channels' coefficient
//! demand forces horizontal (column) tiling, which has never been
//! implemented for stride > 1. This isn't a channel-cap question at all --
//! confirmed with the pure-planning `dump_conv_plan` example (no hardware
//! needed): sweeping 96..512 channels at this same width/stride plans
//! cleanly every time, 576 panics immediately, and the panic is on the
//! tiling strategy, unrelated to `MAX_INPUT_CHANNELS`/`MAX_OUTPUT_CHANNELS`.
//! `match_dynamic_depthwise_conv2d_nchw_3x3_s2`'s bound is intentionally
//! left at 512 -- raising it would let the compiler route this shape to
//! Rocket and then hard-crash the driver at first inference. That's a
//! separate follow-up (horizontal tiling at stride > 1 in `ConvPlan`), not
//! done here.
//!
//! Every prior cap-raise in this codebase (weight_banks_floor's 256..512
//! ladder, the dense matchers' Cout<=512 bound) turned out to hide a
//! shape-specific silent-corruption bug rather than a hard wall, so this
//! reuses `conv_features19_isolated_hw.rs`'s methodology exactly: build the
//! plan through the real `ConvPlan::new` -> `programs_with_buffers` path
//! (not a hand-rolled tile count), fill input and weights uniformly, and
//! check every output element against an exact expected value rather than
//! just "did it complete". Unlike features.19's probe, these two tests
//! aren't chasing a timing/idle-gap hypothesis -- they're a first-job,
//! fresh-device check of whether the shape/CBUF split itself is sound at
//! all, which is the load-bearing question for widening the cap.
//!
//! Both shapes plan to `data_banks=1, weight_banks=11` (`dump_conv_plan`
//! output). That numerically resembles the documented Cout=512 poison
//! pattern in `match_dynamic_conv2d_3x3`'s doc comment, but is not the same
//! failure mode on inspection: the documented bug is *weight*-starvation
//! (`weight_banks` too low for a large *coefficient* footprint --
//! `weight_banks_floor`'s whole reason to exist), and the known-bad dense
//! case there is `weight_banks=1` against a 589824-element coefficient
//! footprint (Cin=256*Cout=256*3*3). Here it's the reverse split
//! (`weight_banks=11`, generously high) and the depthwise coefficient
//! footprint is only Cin*3*3 = 5184 (576ch) or 8640 (960ch) elements, both
//! roughly two orders of magnitude smaller. So this is not a rediscovery of
//! the known bug. It is also not proof of safety: nobody has probed a
//! *data*-starved split (`data_banks=1`) against a large feature-map
//! footprint before, which is exactly what these two shapes are, and that
//! combination has no hardware evidence either way yet. Hence these tests,
//! rather than treating "different bug, different footprint" as good
//! enough.
//!
//! Depthwise's expected value has no channel-sum term (unlike dense conv):
//! each output channel only ever sums its own kernel window, so uniform
//! 1.0 input times uniform 1.0 weight gives `want = kernel_h * kernel_w`
//! regardless of Cin/Cout. The weight buffer is filled directly with the
//! raw all-ones bit pattern rather than going through
//! `pack_depthwise_to_rocket_weights` -- packing only reorders which byte
//! offset holds which (tap, channel) value, and every offset holds the
//! same value here, so the packing layout cannot affect the result. That
//! means this probe cannot by itself re-confirm the tap-major packing
//! formula (`conv_depthwise_hw.rs` and `conv_depthwise_probe_hw.rs` already
//! did that, at Cin/Cout 8 and 12); it is purely a CBUF-split/channel-count
//! probe, matching what raising `MAX_OUTPUT_CHANNELS` actually needs
//! validated.
//!
//! Padded (not logical) width/height is what's given to `Shape`, with
//! `.with_padding([0, 0])` telling the plan the input buffer already covers
//! the full padded extent -- same convention as `conv_features19_isolated_hw.rs`,
//! see its doc comment. All three shapes use MobileNetV2's real 3x3/SAME
//! padding of 1 on each side:
//!
//!   dispatch_68: Cin=Cout=576, logical 14x14 stride 1 -> padded 16x16 in, 14x14 out
//!   dispatch_74: Cin=Cout=576, logical 14x14 stride 2 -> padded 16x16 in,  7x7 out
//!   dispatch_77: Cin=Cout=960, logical  7x7 stride 1 -> padded  9x9 in,  7x7 out
//!
//!   CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!     cargo test --target aarch64-unknown-linux-gnu --release \
//!       --test conv_mobilenetv2_depthwise_wide_hw --no-run
//!
//!   ./conv_mobilenetv2_depthwise_wide_hw-<hash> --ignored --nocapture

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
    // rocket-hal-driver/src/command_buffer.rs uses in production. See
    // conv_features19_isolated_hw.rs's doc comment on `run` for why a
    // hand-rolled tile count (Tile::split without ConvPlan) is not
    // trustworthy evidence here.
    let plan = ConvPlan::new(shape, kernels);
    eprintln!(
        "  [run] {}x{} {}->{} depthwise={} via ConvPlan: {} tile(s), banks {}/{}",
        shape.width,
        shape.height,
        shape.in_channels,
        shape.out_channels,
        shape.depthwise,
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

        // Filled with the raw all-ones bit pattern rather than packed via
        // pack_depthwise_to_rocket_weights -- see this file's top doc
        // comment for why a uniform fill makes the packing layout
        // irrelevant to the result.
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
            // Depthwise: each output channel sums only its own kernel
            // window, no channel-count factor (contrast dense conv's
            // `in_channels * kernel_h * kernel_w` in
            // conv_features19_isolated_hw.rs). Padding=[0,0] plus an input
            // buffer that already covers the full physically-padded extent
            // means every output position reads a full, real kernel-sized
            // window -- no edge clipping to account for.
            let want = (kernels[0] * kernels[1]) as f32;
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

fn open_device() -> (std::fs::File, i32) {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();
    (file, fd)
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn depthwise_576ch_14x14_stride1_runs_clean() {
    let (file, fd) = open_device();
    // dispatch_68: features stage c=96,n=3 repeat blocks, expand 96*6=576.
    let shape = Shape::with_out_channels(16, 16, 1, 576, 576)
        .with_padding([0, 0])
        .with_depthwise();
    let ok = report(
        "Cin=Cout=576, padded 16x16 -> 14x14, stride 1, fresh process, first job",
        run(fd, &file, shape, [3, 3]),
    );
    assert!(
        ok,
        "576-channel depthwise stride-1 shape failed as the first job -- \
         the raised MAX_OUTPUT_CHANNELS cap is not safe for this shape"
    );
}

// dispatch_74 (Cin=Cout=576, 14x14 -> 7x7, stride 2) has no test here --
// `ConvPlan::new` hard-panics for it before any device I/O is possible. See
// this file's top doc comment.

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn depthwise_960ch_7x7_stride1_runs_clean() {
    let (file, fd) = open_device();
    // dispatch_77: features stage c=160,n=3 repeat blocks, expand 160*6=960.
    let shape = Shape::with_out_channels(9, 9, 1, 960, 960)
        .with_padding([0, 0])
        .with_depthwise();
    let ok = report(
        "Cin=Cout=960, padded 9x9 -> 7x7, stride 1, fresh process, first job",
        run(fd, &file, shape, [3, 3]),
    );
    assert!(
        ok,
        "960-channel depthwise stride-1 shape failed as the first job -- \
         the raised MAX_OUTPUT_CHANNELS cap is not safe for this shape"
    );
}
