//! SOFTMAX.md probe **S2**: reducing a row along the width axis with the
//! PPU, as a tree of `k x 1` pooling stages inside one job.
//!
//! This is the geometry SOFTMAX.md section 1 argues the whole design has to
//! use. Every reduction this hardware has runs over space, never channels,
//! so the softmax axis is mapped to width; a row's maximum and sum then
//! come out as a width-1 cube indexed by channel, which is what the BN
//! operand plane (probe S1) broadcasts back across the row. The vendor's
//! own compiled softmax reduces the same way -- five PPU max-pool stages
//! whose width field shrinks 16, 11, 10, 9 with channels pinned at 32
//! (`rknpu-spelunking/NOTES.md` Follow-up 5).
//!
//! Cross-compile this test, copy the resulting binary to the RK3588 board,
//! and run the ignored tests there:
//!
//! ```text
//! CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!   cargo test --target aarch64-unknown-linux-gnu --release \
//!     --test reduce_width_hw --no-run
//!
//! ./reduce_width_hw-<hash> --ignored --nocapture --test-threads=1
//! ```
//!
//! ## What is being asked
//!
//! 1. **Does a `k x 1` window at height 1 reduce along width**, at the
//!    extents a transformer actually uses (`N` 197/384/512), including ones
//!    that are not a multiple of the kernel.
//! 2. **Does the whole tree fit in one job.** Tasks within a job run in
//!    order on one core and the kernel dispatches task `k+1` only after
//!    task `k`'s completion IRQ, so a dependent chain needs no CPU-side
//!    wait and no fence -- SOFTMAX.md section 5's structure A. Every stage
//!    of every tree here is submitted as one ordered task array.
//! 3. **Is the combine just another tree level** (section 5.1): each
//!    column tile writes a disjoint width slice of one shared cube, and one
//!    more pooling task over that cube finishes the reduction. No
//!    `EwBinaryOp::Max` accumulator, no recurrence.
//! 4. **Are the partial-window semantics really free** (section 2.1): a
//!    padded max tap is clamp-to-image, so a non-multiple extent should
//!    need nothing from the caller. 197 columns is the case that matters.
//!
//!    The average half of that question **cannot be asked here**, and that
//!    is this probe's one negative result. An fp16 average pool refuses
//!    every height-1 shape at program-build time, whatever its width
//!    kernel: `average_reciprocal` is asked for the *height* axis's
//!    identity divisor `65536 / 1`, which is past fp16's 65504 ceiling. So
//!    SOFTMAX.md section 2.2's third route to a row sum -- "PPU avg pool
//!    times k" -- does not exist at fp16, though it does at int8.
//!    `pooling.rs`'s `an_fp16_average_cannot_reduce_width_only` and
//!    `an_int8_average_can_reduce_width_only` pin both halves without
//!    needing a device. The two routes the design actually prefers both run
//!    on the DPU and are untouched.
//! 5. **What layout does the reduced vector arrive in** -- the question
//!    section 7 answers by analogy and this answers by measurement.
//!
//! The max arm is gated at **atol 0**: every input value is exactly
//! representable in fp16 and a maximum of exact values is exact, so there
//! is nothing for a tolerance to absorb (the same reasoning as this repo's
//! other f16-exact fixtures). The average arm carries a small tolerance
//! because it divides.
//!
//! Inputs vary per channel *and* per column. A pattern that is constant
//! along the reduced axis cannot catch a tile offset, a surface stride or a
//! window that reduced the wrong columns -- it is the trap recorded against
//! `Counting`, which makes a 1x1 output constant and so proves coverage
//! only.

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr, time::Instant};

#[path = "support/dispatch.rs"]
mod dispatch;

use iree_rocket_hal::rocket::{
    conv::{self, Buffers, ConvPlan, Precision},
    device::{Buffer, JobDesc, close_bo, fini_bo, prep_bo, submit_jobs},
    pooling::{PoolingBuffers, PoolingMethod, PoolingPlan, PoolingPrecision, PoolingShape},
    tensor_layout::pack_depthwise_to_rocket_weights,
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const FEATURE_ATOM_BYTES: usize = 16;
const FP16_LANES_PER_ATOM: usize = 8;
const COMPLETION_TIMEOUT_NS: u64 = 2_000_000_000;
/// Distinguishes an unwritten output from a computed zero.
const OUTPUT_SENTINEL: u8 = 0x5A;
/// The widest kernel the direct pooling path accepts (`MAX_DIRECT_KERNEL`).
const MAX_KERNEL: u32 = 8;

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

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    let word = match exp {
        0 if frac == 0 => sign << 31,
        0x1f => (sign << 31) | 0x7f80_0000 | (frac << 13),
        0 => {
            // An fp16 subnormal is `mantissa * 2^-24`. Normalising it so
            // bit 10 is set after `j` shifts leaves `1.m * 2^(-14 - j)`, so
            // the f32 exponent field wants `113 - j` and this counter has
            // to start at **1**, not -1: `exponent + 127 - 15` is the field.
            // Starting at -1 reports every subnormal a factor of four low,
            // which is what it did until 2026-11-24 -- harmless in tests
            // whose values are of order 1, and not harmless in one reading
            // a softmax's small weights.
            let mut exponent = 1i32;
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

fn page_aligned_size(bytes: usize) -> usize {
    bytes.next_multiple_of(4096).max(4096)
}

/// The probe's input: a multiple of 0.25 in `[-12, 12]`, distinct along
/// both axes and with no repeat inside any one window (7 and 97 are
/// coprime, so the residues do not recur within 8 consecutive columns).
///
/// Every value is exact in fp16, and so is every partial sum of eight of
/// them, which is what lets the max arm assert at atol 0 and keeps the
/// average arm's error to its own divisions.
fn input_value(channel: usize, column: usize) -> f32 {
    0.25 * (((column * 7 + channel * 3) % 97) as f32) - 12.0
}

/// Byte offset of a logical element in a pooling cube.
///
/// Both cube strides are rounded up to **four pixels** by
/// `build_pooling_tile_task` -- the "four-pixel-aligned source and
/// destination surfaces" its register sequence is fixed by. A buffer laid
/// out any other way puts every surface past the first at the wrong offset,
/// which is the same rule `pooling_oracle_hw.rs` states for the gate.
fn cube_offset(channel: usize, pixel: usize, pixels: usize) -> usize {
    let surface_pixels = pixels.next_multiple_of(4);
    (channel / FP16_LANES_PER_ATOM) * surface_pixels * FEATURE_ATOM_BYTES
        + pixel * FEATURE_ATOM_BYTES
        + (channel % FP16_LANES_PER_ATOM) * 2
}

fn cube_bytes(channels: u32, pixels: usize) -> usize {
    let surfaces = (channels as usize).div_ceil(FP16_LANES_PER_ATOM);
    surfaces * pixels.next_multiple_of(4) * FEATURE_ATOM_BYTES
}

/// One `k x 1` width-reduction stage at physical height 1.
fn stage_shape(width: u32, channels: u32, kernel: u32, method: PoolingMethod) -> PoolingShape {
    let output_width = width.div_ceil(kernel);
    let pad_right = output_width * kernel - width;
    assert!(
        pad_right < kernel,
        "a window entirely in the padding hangs the PPU"
    );
    PoolingShape {
        input_width: width,
        input_height: 1,
        input_channels: channels,
        output_width,
        output_height: 1,
        output_channels: channels,
        precision: PoolingPrecision::Fp16,
        kernel_width: kernel,
        kernel_height: 1,
        stride_x: kernel,
        stride_y: 1,
        method,
        pad_left: 0,
        pad_top: 0,
        pad_right,
        pad_bottom: 0,
        pad_value: method.pad_fill_value(PoolingPrecision::Fp16),
    }
}

/// The stages that reduce `width` columns to one, widest kernel first.
///
/// Widest-first is deliberate: every stage boundary costs a completion IRQ
/// (SOFTMAX.md section 5.3), so a shallower tree of wider windows beats a
/// deeper one of narrow windows.
fn reduction_stages(width: u32, channels: u32, method: PoolingMethod) -> Vec<PoolingShape> {
    let mut stages = Vec::new();
    let mut remaining = width;
    while remaining > 1 {
        let kernel = MAX_KERNEL.min(remaining);
        let shape = stage_shape(remaining, channels, kernel, method);
        remaining = shape.output_width;
        stages.push(shape);
    }
    stages
}

struct Reduced {
    /// The final width-1 cube's raw bytes.
    final_raw: Vec<u8>,
    /// Each stage's output cube, raw, when `keep_intermediates` was set.
    stage_raw: Vec<Vec<u8>>,
    tasks: usize,
    elapsed: std::time::Duration,
}

/// Runs a whole reduction tree as **one job's ordered task array** and
/// returns the final cube.
///
/// Every stage's tiles and every stage boundary live in that one array:
/// tiles of a stage write disjoint column slices of their shared output
/// cube, and the next stage reads that cube. Nothing waits on the CPU in
/// between, and the intermediate buffers need no `prep_bo`/`fini_bo` unless
/// the caller asks to read them -- there is no kernel-mediated sync point
/// between tasks of one job for the CPU to wait on anyway, only the
/// hardware writes and each task's completion IRQ.
fn run_reduction(
    stages: &[PoolingShape],
    channels: u32,
    keep_intermediates: bool,
) -> Option<Reduced> {
    assert!(!stages.is_empty(), "nothing to reduce");
    let input_pixels = stages[0].input_width as usize;

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let input = Buffer::new(
            fd,
            page_aligned_size(cube_bytes(channels, input_pixels)),
            &file,
        );
        ptr::write_bytes(input.host_ptr, 0, input.size);
        for channel in 0..channels as usize {
            for column in 0..input_pixels {
                let at = cube_offset(channel, column, input_pixels);
                ptr::write(
                    input.host_ptr.add(at) as *mut u16,
                    f32_to_f16(input_value(channel, column)),
                );
            }
        }

        // One buffer per stage boundary. A stage's output cube and the next
        // stage's input cube are the same bytes, and their four-pixel
        // surface pitches agree by construction because the pixel count
        // does.
        let mut stage_buffers = Vec::with_capacity(stages.len());
        for stage in stages {
            let buffer = Buffer::new(
                fd,
                page_aligned_size(cube_bytes(channels, stage.output_width as usize)),
                &file,
            );
            ptr::write_bytes(buffer.host_ptr, OUTPUT_SENTINEL, buffer.size);
            stage_buffers.push(buffer);
        }

        let mut programs = Vec::new();
        for (index, stage) in stages.iter().enumerate() {
            let input_addr = if index == 0 {
                input.dma_address
            } else {
                stage_buffers[index - 1].dma_address
            };
            let plan = PoolingPlan::new(*stage);
            programs.extend(plan.programs_with_buffers(&PoolingBuffers {
                input_addr,
                output_addr: stage_buffers[index].dma_address,
            }));
        }

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

        fini_bo(fd, input.handle).unwrap();
        for buffer in &stage_buffers {
            fini_bo(fd, buffer.handle).unwrap();
        }
        for (buffer, _) in &command_buffers {
            fini_bo(fd, buffer.handle).unwrap();
        }

        let tasks: Vec<(u32, u32)> = command_buffers
            .iter()
            .map(|(buffer, count)| (buffer.dma_address, *count))
            .collect();
        let mut in_handles: Vec<u32> = command_buffers
            .iter()
            .map(|(buffer, _)| buffer.handle)
            .collect();
        in_handles.push(input.handle);
        // Only the cubes the CPU will read need a fence. The rest are
        // touched by this job alone.
        let out_handles: Vec<u32> = if keep_intermediates {
            stage_buffers.iter().map(|buffer| buffer.handle).collect()
        } else {
            vec![stage_buffers[stage_buffers.len() - 1].handle]
        };
        let jobs = [JobDesc {
            tasks: &tasks,
            in_handles: &in_handles,
            out_handles: &out_handles,
        }];

        let started = Instant::now();
        submit_jobs(fd, &jobs).unwrap_or_else(|error| {
            panic!(
                "SUBMIT failed for a {}-stage, {}-task reduction: {error}",
                stages.len(),
                tasks.len()
            )
        });
        for &handle in &out_handles {
            prep_bo(fd, handle, COMPLETION_TIMEOUT_NS).unwrap_or_else(|error| {
                panic!(
                    "a {}-stage, {}-task reduction did not complete: {error}",
                    stages.len(),
                    tasks.len()
                )
            });
        }
        let elapsed = started.elapsed();

        let final_raw =
            std::slice::from_raw_parts(stage_buffers[stage_buffers.len() - 1].host_ptr, {
                stage_buffers[stage_buffers.len() - 1].size
            })
            .to_vec();
        let stage_raw = if keep_intermediates {
            stage_buffers
                .iter()
                .map(|buffer| std::slice::from_raw_parts(buffer.host_ptr, buffer.size).to_vec())
                .collect()
        } else {
            Vec::new()
        };

        for (buffer, _) in &command_buffers {
            close_bo(fd, buffer.handle).unwrap();
        }
        close_bo(fd, input.handle).unwrap();
        for buffer in &stage_buffers {
            close_bo(fd, buffer.handle).unwrap();
        }

        if dispatch::is_device_timeout(elapsed) {
            let label = format!("{}-stage reduction, {} tasks", stages.len(), tasks.len());
            let strict = dispatch::report(&label, elapsed);
            assert!(!strict, "killed dispatch under ROCKET_STRICT_DISPATCH");
            return None;
        }
        Some(Reduced {
            final_raw,
            stage_raw,
            tasks: tasks.len(),
            elapsed,
        })
    }
}

/// Reads the single reduced value of channel `channel` from a width-1 cube.
fn read_reduced(raw: &[u8], channel: usize) -> f32 {
    let at = cube_offset(channel, 0, 1);
    f16_to_f32(u16::from_le_bytes([raw[at], raw[at + 1]]))
}

fn host_max(channel: usize, width: u32) -> f32 {
    (0..width as usize)
        .map(|column| input_value(channel, column))
        .fold(f32::NEG_INFINITY, f32::max)
}

/// Questions 1, 2 and 4 for the maximum, at **atol 0**.
///
/// 197 is the interesting width: it is ViT's token count, it is not a
/// multiple of 8, and it needs more than one column tile per stage, so one
/// case exercises the partial window, the tile split and the tree depth at
/// once.
#[test]
#[ignore = "requires an RK3588 NPU at /dev/accel/accel0"]
fn max_reduces_a_row_at_every_transformer_extent() {
    let channels = 8u32;
    for width in [8u32, 16, 64, 128, 197, 384, 512] {
        let stages = reduction_stages(width, channels, PoolingMethod::Max);
        let Some(reduced) = run_reduction(&stages, channels, false) else {
            continue;
        };
        println!(
            "  width {width}: {} stage(s), {} task(s){}",
            stages.len(),
            reduced.tasks,
            dispatch::note(reduced.elapsed)
        );
        for channel in 0..channels as usize {
            let got = read_reduced(&reduced.final_raw, channel);
            let want = host_max(channel, width);
            assert_eq!(
                got,
                want,
                "width {width} channel {channel}: a max of fp16-exact values is exact, so \
                 this is gated at atol 0 -- {} stage(s) over {width} columns gave {got}, \
                 the host says {want}",
                stages.len()
            );
        }
    }
}

/// Question 3, stated directly: the per-tile partials really do land in
/// disjoint width slices of one shared cube, and one more pooling task over
/// that cube finishes the job.
///
/// This is the claim that removes the recurrence from SOFTMAX.md's tile
/// combination, so it is checked by reading the intermediate cube rather
/// than inferred from the final answer being right.
#[test]
#[ignore = "requires an RK3588 NPU at /dev/accel/accel0"]
fn the_combine_is_one_more_tree_level_over_one_shared_cube() {
    let channels = 8u32;
    let width = 384u32;
    let stages = reduction_stages(width, channels, PoolingMethod::Max);
    let Some(reduced) = run_reduction(&stages, channels, true) else {
        return;
    };
    let first = stages[0];
    assert!(
        PoolingPlan::new(first).tiles().len() > 1,
        "this test is about the multi-tile case; width {width} planned as one tile"
    );
    println!(
        "  first stage: {} column tile(s) -> {} columns, then {} more stage(s)",
        PoolingPlan::new(first).tiles().len(),
        first.output_width,
        stages.len() - 1
    );

    // Every column of the first stage's shared cube is the max of its own
    // window, whichever tile wrote it. A tile that wrote at the wrong
    // offset would leave a sentinel somewhere, or duplicate a column.
    let kernel = first.kernel_width as usize;
    for channel in 0..channels as usize {
        for column in 0..first.output_width as usize {
            let at = cube_offset(channel, column, first.output_width as usize);
            let got = f16_to_f32(u16::from_le_bytes([
                reduced.stage_raw[0][at],
                reduced.stage_raw[0][at + 1],
            ]));
            let want = (0..kernel)
                .map(|tap| column * kernel + tap)
                .filter(|&source| source < width as usize)
                .map(|source| input_value(channel, source))
                .fold(f32::NEG_INFINITY, f32::max);
            assert_eq!(
                got, want,
                "channel {channel} column {column} of the shared cube: the tiles did not \
                 write disjoint slices of it correctly"
            );
        }
    }

    for channel in 0..channels as usize {
        assert_eq!(
            read_reduced(&reduced.final_raw, channel),
            host_max(channel, width),
            "channel {channel}: the combine over the shared cube disagrees with a host max"
        );
    }
}

/// The channel axis. Eight channels is one fp16 atom and one surface, so
/// every test above would pass with the surface stride ignored entirely.
#[test]
#[ignore = "requires an RK3588 NPU at /dev/accel/accel0"]
fn a_multi_surface_row_reduces_on_every_channel() {
    for channels in [16u32, 32] {
        let width = 197u32;
        let stages = reduction_stages(width, channels, PoolingMethod::Max);
        let Some(reduced) = run_reduction(&stages, channels, false) else {
            continue;
        };
        println!("  channels {channels}: {} task(s)", reduced.tasks);
        for channel in 0..channels as usize {
            assert_eq!(
                read_reduced(&reduced.final_raw, channel),
                host_max(channel, width),
                "channels {channels}, channel {channel} (surface {}, lane {})",
                channel / FP16_LANES_PER_ATOM,
                channel % FP16_LANES_PER_ATOM
            );
        }
    }
}

/// Question 5, and it corrects SOFTMAX.md section 7 rather than confirming
/// it.
///
/// Section 7 argues that a width-1, height-1 cube's NC1HWC2 layout
/// "collapses to dense" -- each surface is one atom, surfaces are
/// consecutive, so channel `c` sits at `c * element_bytes` -- which would
/// make the reduced vector almost what the BN ALU plane wants, needing only
/// an fp16-to-fp32 widen. That is true of the *DPU's* writer. It is **not**
/// true of the PPU's: `build_pooling_tile_task` rounds both cube strides up
/// to four pixels, so a width-1 output puts channel `c` at
/// `(c / 8) * 64 + (c % 8) * 2`, with 48 bytes of nothing between
/// consecutive surfaces.
///
/// So the max tree's output is a strided gather away from a plane operand,
/// not a widen away. This test pins the pitch so section 7's successor
/// knows which it is.
#[test]
#[ignore = "requires an RK3588 NPU at /dev/accel/accel0"]
fn the_reduced_vector_is_four_pixel_strided_not_dense() {
    let channels = 32u32;
    let width = 64u32;
    let stages = reduction_stages(width, channels, PoolingMethod::Max);
    let Some(reduced) = run_reduction(&stages, channels, false) else {
        return;
    };

    for channel in 0..channels as usize {
        let want = host_max(channel, width);
        let strided = cube_offset(channel, 0, 1);
        assert_eq!(
            strided,
            (channel / FP16_LANES_PER_ATOM) * 4 * FEATURE_ATOM_BYTES
                + (channel % FP16_LANES_PER_ATOM) * 2,
            "this test's own offset formula is the four-pixel one"
        );
        assert_eq!(
            f16_to_f32(u16::from_le_bytes([
                reduced.final_raw[strided],
                reduced.final_raw[strided + 1]
            ])),
            want,
            "channel {channel} is not at the four-pixel-strided offset"
        );

        // And the dense reading is wrong for every channel past the first
        // atom, which is the thing section 7 would have got wrong.
        if channel >= FP16_LANES_PER_ATOM {
            let dense = channel * 2;
            let at_dense = f16_to_f32(u16::from_le_bytes([
                reduced.final_raw[dense],
                reduced.final_raw[dense + 1],
            ]));
            assert_ne!(
                at_dense, want,
                "channel {channel} also reads correctly at the dense offset {dense}, so the \
                 pooling writer is not four-pixel strided after all and section 7's analogy \
                 holds -- which would be the better answer, but check it before believing it"
            );
        }
    }
    println!(
        "  width-1 cube at {channels} channels: surface pitch is 4 pixels (64 bytes), so the \
         reduced vector needs a gather rather than a widen to become a BN plane operand"
    );
}

// ===========================================================================
// The sum half: a depthwise all-ones window on the DPU.
// ===========================================================================
//
// SOFTMAX.md section 2.2's remaining routes to a row sum are both on the
// DPU, since section 2.1's fp16 PPU average turned out not to exist. This
// gates the one a *standalone* softmax can use: a depthwise convolution
// whose every coefficient is 1.0, applied as a non-overlapping window that
// tiles the row.
//
// Getting there took three refusals from the planner, each of which is a
// fact worth keeping:
//
// * **A `1 x k` window is refused at any stride but 1** -- "non-square
//   kernels currently have capture backing only at stride 1". At stride 1 it
//   is a sliding box filter, not a reduction.
// * **An even square window is refused at any stride but 1** too -- "even
//   kernels currently have capture backing only at stride 1" -- which rules
//   out the 2x2/s2 halving tree and 4x4, 8x8 with it.
// * So the window is **3x3 at stride 3**, the one strided square kernel the
//   corpus backs, on a height-1 cube. Its vertical taps are padding, and a
//   conv's padding is zero, so a 3x3 window on one row sums three real
//   columns -- which this file's oracle also makes a check of the fp16
//   vertical pad fill, since a nonzero one would show up as three extra
//   taps per window.
//
// The horizontal geometry needs care that the max tree did not. A conv has
// leading padding registers and no trailing ones (the output extent implies
// the right edge), and the output extent is floor-mode, so a row whose
// length is not a multiple of 3 drops its tail unless the *leading* pad
// makes up the difference. With `p = (3 - N % 3) % 3` the windows tile
// `[-p, N - 1]` exactly: disjoint, every real column covered once, and
// `p <= 2` stays under the kernel. That makes the stage count 5 or 6 where
// the max tree needs 3.

/// Kernel for the sum reduction: the only strided square window the
/// corpus backs.
const SUM_KERNEL: conv::Kernels = [3, 3];

/// One stage of the sum tree over a height-1 cube of `width` columns.
fn sum_stage_shape(width: u32, channels: u32) -> conv::Shape {
    let pad_left = (3 - width % 3) % 3;
    conv::Shape {
        width,
        height: 1,
        stride: 3,
        in_channels: channels,
        out_channels: channels,
        precision: Precision::Fp16,
        // [pad_top, pad_left]. The vertical pad is what lets a 3-row window
        // sit on a 1-row image at all; the horizontal pad is the coverage
        // term above.
        padding: Some([1, pad_left as usize]),
        activation: conv::Activation::None,
        depthwise: true,
    }
}

fn sum_stages(width: u32, channels: u32) -> Vec<conv::Shape> {
    let mut stages = Vec::new();
    let mut remaining = width;
    while remaining > 1 {
        let shape = sum_stage_shape(remaining, channels);
        let out = shape.output_width(SUM_KERNEL);
        assert_eq!(
            out,
            (remaining + (3 - remaining % 3) % 3) / 3,
            "the coverage formula and the planner's floor-mode extent disagree at {remaining}"
        );
        assert_eq!(shape.output_height(SUM_KERNEL), 1);
        remaining = out;
        stages.push(shape);
    }
    stages
}

/// A conv cube's surface stride is `width * height` atoms -- the DPU's own
/// writer, with none of the PPU's four-pixel rounding.
fn conv_cube_offset(channel: usize, column: usize, width: usize) -> usize {
    (channel / FP16_LANES_PER_ATOM) * width * FEATURE_ATOM_BYTES
        + column * FEATURE_ATOM_BYTES
        + (channel % FP16_LANES_PER_ATOM) * 2
}

fn conv_cube_bytes(channels: u32, width: usize) -> usize {
    let surfaces = (channels as usize).div_ceil(FP16_LANES_PER_ATOM);
    surfaces * width * FEATURE_ATOM_BYTES
}

/// The sum oracle's input: multiples of 0.25 in `[0, 1]`, which is the
/// range `exp(x - m)` actually produces, and which keeps every partial sum
/// a multiple of 0.25 and so exact in fp16 all the way to the total.
fn sum_input_value(channel: usize, column: usize) -> f32 {
    0.25 * (((column * 7 + channel * 3) % 5) as f32)
}

/// Runs a whole depthwise sum tree as one job's ordered task array.
fn run_sum_reduction(stages: &[conv::Shape], channels: u32) -> Option<Vec<f32>> {
    let input_width = stages[0].width as usize;
    let (kh, kw) = (SUM_KERNEL[0], SUM_KERNEL[1]);

    // Every stage has the same channel count and kernel, so one packed
    // all-ones coefficient buffer serves the whole tree -- which is also
    // why section 2.2 calls it a compile-time constant that packs once.
    let weight_bytes = stages[0].weight_bytes(SUM_KERNEL) as usize;
    let padded_channels = weight_bytes / (kh * kw * 2);
    for stage in stages {
        assert_eq!(
            stage.weight_bytes(SUM_KERNEL) as usize,
            weight_bytes,
            "a stage wants a different coefficient footprint, so one buffer cannot serve all"
        );
    }
    let mut dense = vec![0u8; channels as usize * kh * kw * 2];
    let one = f32_to_f16(1.0).to_le_bytes();
    for slot in dense.chunks_mut(2) {
        slot.copy_from_slice(&one);
    }
    let mut packed = vec![0u8; weight_bytes];
    pack_depthwise_to_rocket_weights(
        &dense,
        kh,
        kw,
        channels as usize,
        padded_channels,
        2,
        &mut packed,
    )
    .expect("all-ones depthwise packing failed");

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let input = Buffer::new(
            fd,
            page_aligned_size(conv_cube_bytes(channels, input_width)),
            &file,
        );
        ptr::write_bytes(input.host_ptr, 0, input.size);
        for channel in 0..channels as usize {
            for column in 0..input_width {
                let at = conv_cube_offset(channel, column, input_width);
                ptr::write(
                    input.host_ptr.add(at) as *mut u16,
                    f32_to_f16(sum_input_value(channel, column)),
                );
            }
        }

        let weights = Buffer::new(fd, page_aligned_size(weight_bytes), &file);
        ptr::write_bytes(weights.host_ptr, 0, weights.size);
        ptr::copy_nonoverlapping(packed.as_ptr(), weights.host_ptr, packed.len());
        // Zero bias: an fp16 convolution's BS ALU operand is an f32 per
        // channel, so a zeroed buffer is a zero bias.
        let bias = Buffer::new(fd, 4096, &file);
        ptr::write_bytes(bias.host_ptr, 0, bias.size);

        let mut stage_buffers = Vec::with_capacity(stages.len());
        for stage in stages {
            let width = stage.output_width(SUM_KERNEL) as usize;
            let buffer = Buffer::new(
                fd,
                page_aligned_size(conv_cube_bytes(channels, width)),
                &file,
            );
            ptr::write_bytes(buffer.host_ptr, OUTPUT_SENTINEL, buffer.size);
            stage_buffers.push(buffer);
        }

        let mut programs = Vec::new();
        for (index, stage) in stages.iter().enumerate() {
            let plan = ConvPlan::try_new(*stage, SUM_KERNEL).unwrap_or_else(|error| {
                panic!(
                    "the planner refused a {}-column sum stage: {error}",
                    stage.width
                )
            });
            assert_eq!(
                plan.tiles().len(),
                1,
                "a tiled sum stage would need its own output placement; {}-column stage                  planned {} tiles",
                stage.width,
                plan.tiles().len()
            );
            programs.extend(plan.programs_with_buffers(Buffers {
                input: if index == 0 {
                    input.dma_address
                } else {
                    stage_buffers[index - 1].dma_address
                },
                weights: weights.dma_address,
                bias: bias.dma_address,
                output: stage_buffers[index].dma_address,
            }));
        }

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

        for handle in [input.handle, weights.handle, bias.handle] {
            fini_bo(fd, handle).unwrap();
        }
        for buffer in &stage_buffers {
            fini_bo(fd, buffer.handle).unwrap();
        }
        for (buffer, _) in &command_buffers {
            fini_bo(fd, buffer.handle).unwrap();
        }

        let tasks: Vec<(u32, u32)> = command_buffers
            .iter()
            .map(|(buffer, count)| (buffer.dma_address, *count))
            .collect();
        let mut in_handles: Vec<u32> = command_buffers
            .iter()
            .map(|(buffer, _)| buffer.handle)
            .collect();
        in_handles.extend([input.handle, weights.handle, bias.handle]);
        let last = stage_buffers[stage_buffers.len() - 1].handle;
        let out_handles = [last];

        let started = Instant::now();
        submit_jobs(
            fd,
            &[JobDesc {
                tasks: &tasks,
                in_handles: &in_handles,
                out_handles: &out_handles,
            }],
        )
        .unwrap_or_else(|error| panic!("SUBMIT failed for a {}-stage sum: {error}", stages.len()));
        prep_bo(fd, last, COMPLETION_TIMEOUT_NS).unwrap_or_else(|error| {
            panic!(
                "a {}-stage, {}-task sum did not complete: {error}",
                stages.len(),
                tasks.len()
            )
        });
        let elapsed = started.elapsed();

        let final_buffer = &stage_buffers[stage_buffers.len() - 1];
        let raw = std::slice::from_raw_parts(final_buffer.host_ptr, final_buffer.size).to_vec();
        let sums = (0..channels as usize)
            .map(|channel| {
                let at = conv_cube_offset(channel, 0, 1);
                f16_to_f32(u16::from_le_bytes([raw[at], raw[at + 1]]))
            })
            .collect();

        for (buffer, _) in &command_buffers {
            close_bo(fd, buffer.handle).unwrap();
        }
        for handle in [input.handle, weights.handle, bias.handle] {
            close_bo(fd, handle).unwrap();
        }
        for buffer in &stage_buffers {
            close_bo(fd, buffer.handle).unwrap();
        }

        if dispatch::is_device_timeout(elapsed) {
            let label = format!("{}-stage sum, {} tasks", stages.len(), tasks.len());
            let strict = dispatch::report(&label, elapsed);
            assert!(!strict, "killed dispatch under ROCKET_STRICT_DISPATCH");
            return None;
        }
        println!(
            "  {} columns: {} stage(s), {} task(s){}",
            input_width,
            stages.len(),
            tasks.len(),
            dispatch::note(elapsed)
        );
        Some(sums)
    }
}

/// The sum half of the reduction, and the gap S2 left open.
///
/// Every input is a multiple of 0.25 in `[0, 1]`, so every partial sum is a
/// multiple of 0.25 and exact in fp16 to well past the total -- the tree
/// should be exact, and this asserts that rather than allowing a tolerance
/// to hide a dropped or double-counted column. A tail that the floor-mode
/// output extent dropped would show up as a short sum; an overlapping
/// window as a long one; a nonzero fp16 vertical pad fill as three extra
/// taps per window.
#[test]
#[ignore = "requires an RK3588 NPU at /dev/accel/accel0"]
fn a_depthwise_all_ones_window_sums_a_row() {
    let channels = 64u32;
    for width in [9u32, 27, 64, 197, 384, 512] {
        let stages = sum_stages(width, channels);
        let Some(sums) = run_sum_reduction(&stages, channels) else {
            continue;
        };
        for (channel, &got) in sums.iter().enumerate() {
            let want: f32 = (0..width as usize)
                .map(|column| sum_input_value(channel, column))
                .sum();
            assert_eq!(
                got,
                want,
                "width {width} channel {channel}: {} stage(s) of a 3x3/s3 all-ones depthwise                  window summed to {got}, the host says {want}. Every partial sum here is a                  multiple of 0.25 and exact in fp16, so this is gated exactly.",
                stages.len()
            );
        }
    }
}

/// The coefficient footprint is the same for every stage, which is what
/// lets one packed all-ones buffer serve the whole tree. Worth its own
/// assertion because it is a property of the shape family rather than of
/// any one stage, and it needs no device.
#[test]
fn one_all_ones_coefficient_buffer_serves_every_sum_stage() {
    for width in [197u32, 384, 512] {
        let stages = sum_stages(width, 64);
        assert!(
            stages.len() >= 5,
            "width {width} planned {} stages",
            stages.len()
        );
        let first = stages[0].weight_bytes(SUM_KERNEL);
        for stage in &stages {
            assert_eq!(stage.weight_bytes(SUM_KERNEL), first, "width {width}");
            assert!(
                ConvPlan::try_new(*stage, SUM_KERNEL).is_ok(),
                "width {width}"
            );
        }
    }
}

/// The two refusals that forced a 3x3 stride-3 window, pinned so a later
/// change to the admission rules shows up here as a choice rather than as
/// a surprise. Neither needs a device.
#[test]
fn only_an_odd_square_window_can_stride() {
    let base = sum_stage_shape(384, 64);
    for kernel in [[1usize, 2usize], [1, 3], [1, 8]] {
        assert!(
            ConvPlan::try_new(base, kernel).is_err(),
            "a non-square {kernel:?} window is admitted at stride 3 now; a 1 x k reduction              would be shallower than the 3x3 tree, so this is worth acting on"
        );
    }
    for kernel in [[2usize, 2usize], [4, 4], [8, 8]] {
        let mut shape = base;
        shape.stride = kernel[0] as u32;
        shape.padding = Some([kernel[0] / 2, 0]);
        assert!(
            ConvPlan::try_new(shape, kernel).is_err(),
            "an even {kernel:?} window is admitted at stride {} now; an 8x8 one would reduce              a row in two stages instead of six",
            kernel[0]
        );
    }
    assert!(ConvPlan::try_new(base, SUM_KERNEL).is_ok());
}
