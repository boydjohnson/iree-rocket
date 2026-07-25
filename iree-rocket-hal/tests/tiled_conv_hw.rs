//! Hardware-in-the-loop tests for `rocket::conv`'s tiled multi-task
//! convolution and the NPU core's ping-pong register groups.
//!
//! Not run by a plain `cargo test` -- this crate targets the board's NPU
//! (`/dev/accel/accel0`), which doesn't exist on the host doing the
//! building. Cross-compile the test binary and copy it over instead:
//!
//!   CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!     cargo test --target aarch64-unknown-linux-gnu --release \
//!       --test tiled_conv_hw --no-run
//!
//! then copy the resulting binary (path printed by `--no-run`, under
//! `target/aarch64-unknown-linux-gnu/release/deps/tiled_conv_hw-*`) to the
//! board and run it there as `./tiled_conv_hw-<hash> --ignored
//! --test-threads=1 --nocapture`. Single-threaded on purpose: these tests
//! share one NPU, and `--nocapture` because the wall-clock reports below
//! are the point of some of them.
//!
//! # What this file is isolating
//!
//! `conv_hw.rs::conv_fp16_c32_to_c16_position_values_height_splits` already
//! proves the *shape* used here computes correctly on real hardware when
//! its height splits are submitted as separate, individually-fenced DRM
//! jobs. Its doc comment also records the failure this file exists to chase:
//! the same regcmds submitted as multiple tasks in one `drm_rocket_job`
//! produced correct task-0 rows and left every later row at zero.
//!
//! So every test here runs the *same tensor with the same oracle* and
//! varies only how the task sequence is dispatched and how the ping-pong
//! registers are programmed.
//!
//! # Results (real RK3588, kernel 7.1.0-edge-rockchip64)
//!
//! | test | dispatch | result |
//! |---|---|---|
//! | `single_tile_matches_the_oracle` | one job, one tile | **pass** |
//! | `row_tiles_as_separate_jobs_match_the_oracle` | job per tile | **pass** |
//! | `late_tile_rows_do_not_appear_after_settling` | one job, 3 tasks | **pass**: zero rows 38..=111 before AND after |
//! | `row_tiles_as_kernel_tasks_without_ping_pong` | one job, 3 tasks | fail: rows 38..=111 zero |
//! | `row_tiles_as_kernel_tasks_with_ping_pong` | one job, 3 tasks | fail: identical |
//! | `row_tiles_as_kernel_tasks_with_explicit_pointers` | one job, 3 tasks | fail: identical |
//! | `reversed_task_order_...` | one job, 3 tasks reversed | fail: zero rows **0..=74** |
//! | `row_tiles_as_hardware_chain_*` | one task, PC-walked chain | fail: rows 38..=111 zero |
//! | `single_tile_with_*_interrupts_unmasked_*` | one job, one tile | fail: still ~500 ms (526 / 504) |
//! | `row_tiles_as_kernel_tasks_with_all_interrupts_unmasked` | one job, 3 tasks | fail: rows 38..=111 zero |
//! | `*_via_kick_domain_*` | mask override tagged `0x81` | not yet run |
//!
//! Every red test above has the same single root cause (below). The two green
//! ones are the real contract: `conv.rs`'s tiling and emission are correct,
//! and stay correct, on the dispatch path that works.
//!
//! Four conclusions, in the order they were established:
//!
//! **Ping-pong programming is not the variable.** All three
//! [`PointerMode`]s give byte-identical results, and `separate_jobs`
//! (passes) and `kernel_tasks` (fails) submit *byte-identical command
//! buffers*. The difference between pass and fail contains no regcmd
//! content at all. `rocket_job.c` also arms CNA/CORE `S_POINTER` itself
//! before every task, so this crate was never the deciding factor -- see
//! `conv.rs`'s module doc.
//!
//! **The failure is positional, and the later tiles never write.** Reversing
//! the submission order moves the surviving rows to 75..=111 -- tile 2's,
//! i.e. whichever task went *first*. And a 500 ms settle changes nothing, so
//! this is not the CPU reading before a late write lands.
//!
//! **Only the first task per job is ever dispatched, because the completion
//! IRQ never arrives.** An eBPF trace of the driver counted 69 jobs with
//! `ops=0, timeouts=69, resets=69`: every job times out in `drm_sched` and
//! is force-reset, and the hw_submit -> IRQ pairing never completes once.
//! `rocket_job_handle_irq()` is the only thing that re-enters
//! `rocket_job_hw_submit()` for task N+1, so with no interrupt there is no
//! task N+1. See `conv.rs`'s module doc for the full trace and its two
//! corollaries -- in particular, that the ~510 ms every test below reports
//! is the scheduler's timeout, and that the passing tests are passing
//! *through* the reset path rather than through clean completion.
//!
//! **The NPU never asserts its interrupt line.** `/proc/interrupts` shows the
//! three `*.npu` GIC lines byte-identical before and after a run, across three
//! runs -- so the driver's hard handler never even gets a chance to reject
//! anything, and `dmesg` carries nothing but the bare `NPU job timed out`
//! lines (no IOMMU faults, no reset failures). A 4x4 tile times out just like
//! a 112x112 one, so it is not the op running long. The driver is not at
//! fault by version either: mainline 7.1's rocket is byte-identical to v6.18
//! through the IRQ/submit path, and no Armbian `rockchip64-7.1` patch touches
//! it.
//!
//! Overriding `PC_INTERRUPT_MASK` from the regcmd changed nothing in either
//! polarity's favour -- but see `conv::PcWriteDomain`: those overrides used a
//! domain tag (`0x101`) that this crate has never confirmed reaches the
//! hardware, so that result is **inconclusive, not negative**, and
//! `single_tile_with_all_interrupts_unmasked_via_kick_domain_...` retries it
//! with the tag the kick uses.

use std::{
    fs::OpenOptions,
    mem,
    os::unix::io::AsRawFd,
    ptr,
    time::{Duration, Instant},
};

use iree_rocket_hal::rocket::{
    conv::{
        InterruptMask, PcWriteDomain, PingPong, PointerMode, RegisterAmount, Tiling,
        build_tiled_conv_regcmds, cycles_per_pixel, link_tiled_conv_regcmds, plan_tiled_conv,
    },
    device::{Buffer, close_bo, fini_bo, prep_bo, submit, submit_tasks},
    regcmd::{
        Activation, CONV_OUTPUT_ATOMIC_STRIDE, ConvBuffers, ConvShape, Precision,
        conv_output_scratch_bytes,
    },
    tensor_layout::{nc1hwc2_storage_size, pack_nhwc_to_nc1hwc2},
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const BIAS_SIZE: usize = 4096;
const JOB_TIMEOUT_NS: u64 = 2_000_000_000;

const INPUT_CHANNELS: u32 = 32;
const OUTPUT_CHANNELS: u32 = 16;
const WEIGHT: f32 = 1.0 / INPUT_CHANNELS as f32;
const BPE: usize = 2;

/// `conv_hw.rs`'s hardware-validated fp16 geometry: C32 -> C16, 1x1 kernel,
/// stride 1, so output spatial dims equal input and the valid-convolution
/// geometry `plan_tiled_conv` requires is satisfied for any size.
fn shape(width: u32, height: u32) -> ConvShape {
    ConvShape {
        input_width: width,
        input_height: height,
        input_channels: INPUT_CHANNELS,
        output_width: width,
        output_height: height,
        output_channels: OUTPUT_CHANNELS,
        weights_width: 1,
        weights_height: 1,
        stride: 1,
        depthwise: false,
        input_zero_point: 0,
        output_zero_point: 0,
        weights_zero_point: 0,
        input_scale: 1.0,
        weights_scale: 1.0,
        output_scale: 1.0,
        truncate_bits: 0,
        activation: Activation::None,
        precision: Precision::Fp16,
    }
}

/// Minimal IEEE-754 binary16 -> f32 decode, duplicated from `conv_hw.rs`
/// rather than shared (integration tests in `tests/` each carry their own
/// copy -- see that file's own note). Handles subnormals and inf/nan as
/// well as normals, so a wrong result shows up as the wrong number rather
/// than being flushed to zero.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 0x1) as u32;
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
        (sign << 31) | ((exp + (127 - 15)) << 23) | (frac << 13)
    };
    f32::from_bits(f32_bits)
}

fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = (bits >> 31) & 0x1;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x7f_ffff;
    let new_exp = exp - 127 + 15;
    assert!((1..31).contains(&new_exp), "value out of easy f16 range");
    ((sign << 15) | ((new_exp as u32) << 10) | (frac >> 13)) as u16
}

/// Coordinate-derived input value (`conv_hw.rs`'s position oracle). A
/// uniform fill cannot tell a tile that wrote the wrong rows from one that
/// wrote the right ones -- which is the entire failure mode under
/// investigation here -- so every element is a function of `(y, x, c)`, and
/// every value is an integer exactly representable in fp16.
fn input_value(y: u32, x: u32, channel: u32) -> f32 {
    let spatial = 1 + (y * 7 + x * 3) % 16;
    (spatial + channel) as f32
}

/// With all 32 weights equal to 1/32, `sum_c(spatial + c) / 32` is
/// `spatial + 15.5` -- exact in fp16 at this magnitude, so the assertions
/// below are equalities, not tolerances.
fn expected_value(y: u32, x: u32) -> f32 {
    let spatial = 1 + (y * 7 + x * 3) % 16;
    spatial as f32 + 15.5
}

fn page_aligned(byte_len: usize) -> usize {
    byte_len.max(1).next_multiple_of(4096)
}

/// Order the tiles are handed to the kernel in. Tiles read disjoint input
/// rows and write disjoint output rows (1x1 kernel, no halo), so a correct
/// implementation produces the identical tensor either way -- which makes
/// `Reversed` a clean probe for *position*-dependent failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Order {
    Forward,
    Reversed,
}

/// How the tile sequence reaches the hardware. See this file's doc comment
/// table for what each one isolates.
#[derive(Clone, Copy, Debug)]
enum Dispatch {
    /// One DRM job per tile, each fenced from the CPU before the next is
    /// submitted. The arrangement `conv_hw.rs` found working; ping-pong buys
    /// nothing here, since the PC is re-kicked from scratch every time.
    SeparateJobs,
    /// All tiles as one job's task array. `rocket_job.c` dispatches task N+1
    /// from task N's completion IRQ and only signals the job's `done_fence`
    /// once `next_task_idx == task_count`.
    KernelTasks(Order),
    /// Only tile 0 submitted; its regcmd's trailing PC link points at tile
    /// 1, and so on. Cannot work through the mainline driver, which pins
    /// `PC_TASK_CON.task_number` to 1 -- see `conv::RegisterAmount` and
    /// `conv.rs`'s module doc.
    HardwareChain(RegisterAmount),
}

struct Run {
    /// Output decoded to dense NHWC order.
    output: Vec<f32>,
    /// The same buffer re-fenced and re-decoded after a settling delay, when
    /// one was requested. Distinguishes "the later tiles never wrote" from
    /// "they wrote after the CPU looked".
    output_after_settle: Option<Vec<f32>>,
    /// Submit-to-completion wall clock, summed across jobs for
    /// `SeparateJobs`.
    elapsed: Duration,
    tile_output_rows: Vec<u32>,
    /// The design notes' MAC model for this plan (`conv::TiledConv::
    /// sequential_cycles`), for comparison against `elapsed`.
    estimated_cycles: u64,
}

/// The scaffolding: plans the tiling, packs the position-oracle input,
/// builds one command buffer per tile, dispatches per `dispatch`, and
/// decodes the DPU's feature-atomic output surfaces back to dense NHWC.
///
/// Every host-visible buffer gets `fini_bo` before submission (flush CPU
/// writes / invalidate for the device) and the output gets `prep_bo` after
/// (wait for the job's fence, then invalidate for the CPU). GEM handles are
/// released with `close_bo` at the end: this file allocates well over a
/// dozen buffers per test in a single process, which is exactly the
/// situation `close_bo`'s doc comment records producing stale results back
/// when nothing in this crate closed a handle.
fn run_tiled_conv(
    shape: &ConvShape,
    tiling: Tiling,
    ping_pong: PingPong,
    dispatch: Dispatch,
) -> Run {
    run_tiled_conv_settling(
        shape,
        tiling,
        ping_pong,
        dispatch,
        InterruptMask::default(),
        PcWriteDomain::default(),
        None,
    )
}

/// As [`run_tiled_conv`], with an explicit [`InterruptMask`] override.
fn run_tiled_conv_masked(
    shape: &ConvShape,
    tiling: Tiling,
    ping_pong: PingPong,
    dispatch: Dispatch,
    interrupt_mask: InterruptMask,
    pc_domain: PcWriteDomain,
) -> Run {
    run_tiled_conv_settling(
        shape,
        tiling,
        ping_pong,
        dispatch,
        interrupt_mask,
        pc_domain,
        None,
    )
}

/// As [`run_tiled_conv`], but when `settle` is set, the output buffer is
/// re-fenced with a second `prep_bo` after that delay and decoded a second
/// time into `Run::output_after_settle`.
///
/// `prep_bo` on an already-signaled fence returns immediately and just
/// re-invalidates the mapping for the CPU, so this is a genuine second look
/// at device memory rather than a re-read of the same cached bytes.
fn run_tiled_conv_settling(
    shape: &ConvShape,
    tiling: Tiling,
    ping_pong: PingPong,
    dispatch: Dispatch,
    interrupt_mask: InterruptMask,
    pc_domain: PcWriteDomain,
    settle: Option<Duration>,
) -> Run {
    let plan = plan_tiled_conv(shape, tiling, ping_pong).expect("tiled convolution plan");
    let tile_output_rows = plan.tiles.iter().map(|tile| tile.output_height).collect();
    let estimated_cycles = plan.sequential_cycles();

    let pixel_count = shape.input_width as usize * shape.input_height as usize;
    let input_bytes_per_pixel = shape.input_channels as usize * BPE;
    let input_scratch_len =
        nc1hwc2_storage_size(pixel_count, input_bytes_per_pixel).expect("input scratch size");
    let output_scratch_len = conv_output_scratch_bytes(shape);

    let mut dense_input = vec![0u8; pixel_count * input_bytes_per_pixel];
    for y in 0..shape.input_height {
        for x in 0..shape.input_width {
            for channel in 0..shape.input_channels {
                let index = ((y * shape.input_width + x) * shape.input_channels + channel) as usize;
                dense_input[index * BPE..index * BPE + BPE]
                    .copy_from_slice(&f32_to_f16_bits(input_value(y, x, channel)).to_le_bytes());
            }
        }
    }

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let buf_in = Buffer::new(fd, page_aligned(input_scratch_len), &file);
        ptr::write_bytes(buf_in.host_ptr, 0, buf_in.size);
        pack_nhwc_to_nc1hwc2(
            &dense_input,
            pixel_count,
            input_bytes_per_pixel,
            std::slice::from_raw_parts_mut(buf_in.host_ptr, input_scratch_len),
        )
        .expect("failed to pack test input as NC1HWC2");

        // 1x1 kernel, so the weight buffer is just Cin x Cout with no tap
        // dimension to get right -- uniform 1/32 across every real channel
        // of every real kernel.
        let weight_bytes = shape.input_channels as usize * shape.output_channels as usize * BPE;
        let buf_w = Buffer::new(fd, page_aligned(weight_bytes), &file);
        ptr::write_bytes(buf_w.host_ptr, 0, buf_w.size);
        let weight_slice =
            std::slice::from_raw_parts_mut(buf_w.host_ptr as *mut u16, weight_bytes / 2);
        weight_slice.fill(f32_to_f16_bits(WEIGHT));

        let buf_bias = Buffer::new(fd, BIAS_SIZE, &file);
        ptr::write_bytes(buf_bias.host_ptr, 0, BIAS_SIZE);

        let buf_out = Buffer::new(fd, page_aligned(output_scratch_len), &file);
        ptr::write_bytes(buf_out.host_ptr, 0, buf_out.size);

        let bufs = ConvBuffers {
            input_addr: buf_in.dma_address,
            weights_addr: buf_w.dma_address,
            bias_addr: buf_bias.dma_address,
            output_addr: buf_out.dma_address,
        };
        let mut tasks = build_tiled_conv_regcmds(&plan, &bufs, interrupt_mask, pc_domain)
            .expect("tiled convolution regcmds");

        // One command buffer per tile. Addresses have to exist before the
        // PC links can be patched, so allocate first, then link, then fill.
        let cmd_buffers: Vec<Buffer> = tasks
            .iter()
            .map(|cmds| Buffer::new(fd, page_aligned(cmds.len() * mem::size_of::<u64>()), &file))
            .collect();
        if let Dispatch::HardwareChain(amount) = dispatch {
            let addresses: Vec<u32> = cmd_buffers.iter().map(|buf| buf.dma_address).collect();
            link_tiled_conv_regcmds(&mut tasks, &addresses, amount)
                .expect("failed to link the tile chain");
        }
        for (cmds, buf_cmd) in tasks.iter().zip(&cmd_buffers) {
            let slot = std::slice::from_raw_parts_mut(buf_cmd.host_ptr as *mut u64, cmds.len());
            for (target, command) in slot.iter_mut().zip(cmds) {
                *target = command.0;
            }
        }

        fini_bo(fd, buf_in.handle).ok();
        fini_bo(fd, buf_w.handle).ok();
        fini_bo(fd, buf_bias.handle).ok();
        fini_bo(fd, buf_out.handle).ok();
        for buf_cmd in &cmd_buffers {
            fini_bo(fd, buf_cmd.handle).ok();
        }

        // Job-wide, not per-task: every command buffer any dispatched task
        // might fetch has to be retained for the job's duration, including
        // the chain's later tiles that the kernel never sees a task for.
        let mut in_handles: Vec<u32> = cmd_buffers.iter().map(|buf| buf.handle).collect();
        in_handles.extend([buf_in.handle, buf_w.handle, buf_bias.handle]);
        let out_handles = [buf_out.handle];
        let mut descriptors: Vec<(u32, u32)> = cmd_buffers
            .iter()
            .zip(&tasks)
            .map(|(buf, cmds)| (buf.dma_address, cmds.len() as u32))
            .collect();
        if matches!(dispatch, Dispatch::KernelTasks(Order::Reversed)) {
            descriptors.reverse();
        }

        let started = Instant::now();
        match dispatch {
            Dispatch::SeparateJobs => {
                for &(regcmd_addr, regcmd_count) in &descriptors {
                    submit(fd, regcmd_addr, regcmd_count, &in_handles, &out_handles)
                        .expect("per-tile SUBMIT ioctl failed");
                    prep_bo(fd, buf_out.handle, JOB_TIMEOUT_NS)
                        .expect("per-tile job did not complete within timeout");
                }
            }
            Dispatch::KernelTasks(_) => {
                submit_tasks(fd, &descriptors, &in_handles, &out_handles)
                    .expect("multi-task SUBMIT ioctl failed");
                prep_bo(fd, buf_out.handle, JOB_TIMEOUT_NS)
                    .expect("multi-task job did not complete within timeout");
            }
            Dispatch::HardwareChain(_) => {
                // Only tile 0: the PC follows the embedded links for the
                // rest, and `pc_interrupt_mask` applies to the last task in
                // the running group, so one fence covers the whole chain.
                let (regcmd_addr, regcmd_count) = descriptors[0];
                submit(fd, regcmd_addr, regcmd_count, &in_handles, &out_handles)
                    .expect("chained SUBMIT ioctl failed");
                prep_bo(fd, buf_out.handle, JOB_TIMEOUT_NS)
                    .expect("chained job did not complete within timeout");
            }
        }
        let elapsed = started.elapsed();

        let scratch = std::slice::from_raw_parts(buf_out.host_ptr, output_scratch_len);
        let output = decode_output(shape, scratch);

        // Second look at the same device memory: if the later tiles ran but
        // landed after the CPU's first read, their rows appear here.
        let output_after_settle = settle.map(|delay| {
            std::thread::sleep(delay);
            prep_bo(fd, buf_out.handle, JOB_TIMEOUT_NS).ok();
            decode_output(shape, scratch)
        });

        close_bo(fd, buf_in.handle).ok();
        close_bo(fd, buf_w.handle).ok();
        close_bo(fd, buf_bias.handle).ok();
        close_bo(fd, buf_out.handle).ok();
        for buf_cmd in &cmd_buffers {
            close_bo(fd, buf_cmd.handle).ok();
        }

        Run {
            output,
            output_after_settle,
            elapsed,
            tile_output_rows,
            estimated_cycles,
        }
    }
}

/// DPU write-back is 16-byte feature-atomic surfaces: surface S holds
/// channel bytes `[16S, 16S+16)` for every pixel, planar. Same decode as
/// `conv_hw.rs`'s position test.
fn decode_output(shape: &ConvShape, scratch: &[u8]) -> Vec<f32> {
    let output_pixels = shape.output_width as usize * shape.output_height as usize;
    let surface_stride = output_pixels * CONV_OUTPUT_ATOMIC_STRIDE as usize;
    let mut output = Vec::with_capacity(output_pixels * shape.output_channels as usize);
    for pixel in 0..output_pixels {
        for channel in 0..shape.output_channels as usize {
            let channel_byte = channel * BPE;
            let surface = channel_byte / CONV_OUTPUT_ATOMIC_STRIDE as usize;
            let byte_in_surface = channel_byte % CONV_OUTPUT_ATOMIC_STRIDE as usize;
            let offset = surface * surface_stride
                + pixel * CONV_OUTPUT_ATOMIC_STRIDE as usize
                + byte_in_surface;
            output.push(f16_to_f32(u16::from_le_bytes([
                scratch[offset],
                scratch[offset + 1],
            ])));
        }
    }
    output
}

/// Which output rows are entirely zero -- the signature of a tile that
/// never wrote anything, as distinct from one that wrote wrong values.
fn zero_rows(shape: &ConvShape, output: &[f32]) -> Vec<u32> {
    (0..shape.output_height)
        .filter(|&y| {
            (0..shape.output_width).all(|x| {
                (0..shape.output_channels).all(|channel| {
                    let index =
                        ((y * shape.output_width + x) * shape.output_channels + channel) as usize;
                    output[index] == 0.0
                })
            })
        })
        .collect()
}

/// Compact `first..last` summary of a row list, so a 74-row report reads as
/// one range instead of 74 numbers.
fn row_span(rows: &[u32]) -> String {
    match (rows.first(), rows.last()) {
        (Some(first), Some(last)) => format!("{} rows, {first}..={last}", rows.len()),
        _ => "none".to_string(),
    }
}

/// Asserts the whole tensor against the oracle, and reports *which rows*
/// disagree rather than just the first mismatch: "rows 0..37 correct, the
/// rest zero" is the specific signature this file is hunting, and it is
/// only visible if the failure message says so.
fn assert_matches_oracle(shape: &ConvShape, run: &Run, label: &str) {
    let mut bad_rows = Vec::new();
    let mut zero_rows = Vec::new();
    let mut first_mismatches = Vec::new();
    for y in 0..shape.output_height {
        let mut row_bad = false;
        let mut row_all_zero = true;
        for x in 0..shape.output_width {
            let expected = expected_value(y, x);
            for channel in 0..shape.output_channels {
                let index =
                    ((y * shape.output_width + x) * shape.output_channels + channel) as usize;
                let actual = run.output[index];
                if actual != 0.0 {
                    row_all_zero = false;
                }
                if actual != expected {
                    row_bad = true;
                    if first_mismatches.len() < 8 {
                        first_mismatches.push(format!(
                            "[{y}, {x}, {channel}]: expected {expected}, got {actual}"
                        ));
                    }
                }
            }
        }
        if row_bad {
            bad_rows.push(y);
            if row_all_zero {
                zero_rows.push(y);
            }
        }
    }

    assert!(
        bad_rows.is_empty(),
        "{label}: {} of {} output rows wrong ({} of them entirely zero); \
         tile output rows were {:?}. First mismatches:\n{}",
        bad_rows.len(),
        shape.output_height,
        zero_rows.len(),
        run.tile_output_rows,
        first_mismatches.join("\n")
    );
}

fn report(label: &str, shape: &ConvShape, run: &Run) {
    eprintln!(
        "{label}: tiles {:?}, {} cycles estimated ({} cycles/px), {:?} wall clock",
        run.tile_output_rows,
        run.estimated_cycles,
        cycles_per_pixel(shape),
        run.elapsed
    );
}

/// Cheapest possible anchor: one tile, so nothing about task sequencing is
/// in play and any failure is in `conv.rs`'s plan/emission itself.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn single_tile_matches_the_oracle() {
    let shape = shape(4, 4);
    let run = run_tiled_conv(
        &shape,
        Tiling::Tiles(1),
        PingPong::default(),
        Dispatch::SeparateJobs,
    );
    report("single_tile", &shape, &run);
    assert_eq!(run.tile_output_rows, vec![4]);
    assert_matches_oracle(&shape, &run, "single_tile");
}

/// Caller-driven row tiling over the dispatch path `conv_hw.rs` already
/// proved works. Isolates "does `plan_tiled_conv` tile correctly" from
/// every ping-pong and task-sequencing question.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn row_tiles_as_separate_jobs_match_the_oracle() {
    let shape = shape(112, 112);
    let run = run_tiled_conv(
        &shape,
        Tiling::Tiles(3),
        PingPong::default(),
        Dispatch::SeparateJobs,
    );
    report("separate_jobs", &shape, &run);
    assert_eq!(run.tile_output_rows, vec![38, 37, 37]);
    assert_matches_oracle(&shape, &run, "separate_jobs");
}

/// The known failure, reproduced deliberately: N tasks in one job with
/// ping-pong left exactly as this crate programs it today (DPU/DPU_RDMA
/// armed, CNA/CORE at reset, no `PC_TASK_CON`). Expected to fail with
/// tile 0's rows correct and the rest zero -- if it *passes*, the earlier
/// failure was something else and the rest of this file's premise needs
/// revisiting.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn row_tiles_as_kernel_tasks_without_ping_pong() {
    let shape = shape(112, 112);
    let run = run_tiled_conv(
        &shape,
        Tiling::Tiles(3),
        PingPong::off(),
        Dispatch::KernelTasks(Order::Forward),
    );
    report("kernel_tasks_ping_pong_off", &shape, &run);
    assert_matches_oracle(&shape, &run, "kernel_tasks_ping_pong_off");
}

/// Same dispatch, with CNA and CORE brought into the same armed state the
/// payload already puts DPU/DPU_RDMA in, plus `PC_TASK_CON`.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn row_tiles_as_kernel_tasks_with_ping_pong() {
    let shape = shape(112, 112);
    let run = run_tiled_conv(
        &shape,
        Tiling::Tiles(3),
        PingPong::default(),
        Dispatch::KernelTasks(Order::Forward),
    );
    report("kernel_tasks_auto_toggle", &shape, &run);
    assert_matches_oracle(&shape, &run, "kernel_tasks_auto_toggle");
}

/// Pointer alternation driven from the regcmd stream instead of by
/// hardware -- distinguishes "the toggle doesn't happen the way the TRM
/// implies" from "multi-task dispatch is broken for some other reason".
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn row_tiles_as_kernel_tasks_with_explicit_pointers() {
    let shape = shape(112, 112);
    let run = run_tiled_conv(
        &shape,
        Tiling::Tiles(3),
        PingPong {
            pointer_mode: PointerMode::ExplicitPerTask,
            executers: true,
            pc_task_fetch: true,
        },
        Dispatch::KernelTasks(Order::Forward),
    );
    report("kernel_tasks_explicit_pointer", &shape, &run);
    assert_matches_oracle(&shape, &run, "kernel_tasks_explicit_pointer");
}

//===========================================================================
// Diagnostics for the "first task only" failure. Both have now run on
// hardware; their answers are recorded in this file's doc comment and drove
// the eBPF tracing that found the root cause. They stay as regression
// witnesses -- if the interrupt path is ever fixed, both should flip.
//
// A caution for anyone repeating this reasoning: `prep_bo` returning
// successfully does NOT mean the job completed cleanly. It returns because
// the timeout-and-reset path force-completes the fence. An earlier revision
// of this comment argued from a successful `prep_bo` that every task must
// have been dispatched; the eBPF trace showed the opposite. Likewise, the
// ~510 ms per job says nothing about how many tasks ran -- it is the
// scheduler's timeout, and `pm_runtime_get_sync`/`iommu_attach_group` are
// per-job (`rocket_job_run`) rather than per-task (`hw_submit`) anyway.
//===========================================================================

/// Did the later tiles write *late* rather than not at all? Re-fences and
/// re-decodes the output buffer after a settling delay.
///
/// Asserts nothing about the first read (that failure is already covered
/// above); it asserts that the second read is no better than the first, i.e.
/// that this is genuinely not a CPU-looked-too-early problem.
///
/// **Hardware result: passes** -- zero rows are 38..=111 both before and
/// after 500 ms. The later tiles never write at all.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn late_tile_rows_do_not_appear_after_settling() {
    let shape = shape(112, 112);
    let run = run_tiled_conv_settling(
        &shape,
        Tiling::Tiles(3),
        PingPong::default(),
        Dispatch::KernelTasks(Order::Forward),
        InterruptMask::default(),
        PcWriteDomain::default(),
        Some(Duration::from_millis(500)),
    );
    let before = zero_rows(&shape, &run.output);
    let settled = run
        .output_after_settle
        .as_ref()
        .expect("a settling delay was requested");
    let after = zero_rows(&shape, settled);

    eprintln!(
        "settle probe: zero rows before = {}, after 500ms = {}",
        row_span(&before),
        row_span(&after)
    );

    assert_eq!(
        before,
        after,
        "output CHANGED after a 500ms settle: zero rows went from {} to {}. \
         The later tasks do write, just after the CPU's first read -- so the \
         fault is in completion signaling/fencing, not in the tile regcmds, \
         and this whole file's premise needs reframing.",
        row_span(&before),
        row_span(&after)
    );
}

/// Is the failure positional (only the *first* submitted task works) or
/// content-based (only *tile 0's* regcmd works)?
///
/// Tiles read and write disjoint rows, so submitting them in reverse is
/// still a correct program and a working implementation returns the same
/// full tensor. The diagnostic value is in *which* rows survive:
///
/// - tile 2's rows (75..=111) correct, 0..=74 zero -> positional: whichever
///   task goes first is the only one that runs to completion, and tile 0's
///   regcmd content is not special;
/// - tile 0's rows (0..=37) correct again -> content-based: something about
///   the first tile's own program (it is the one carrying the `PC_TASK_CON`
///   and `S_POINTER` preamble, and it is 134 words vs the others' 128) is
///   what makes it work;
/// - all rows correct -> reversal fixed it, which would point at the
///   input-side CBUF/DMA state carried between tasks rather than at
///   dispatch.
///
/// **Hardware result: the first case.** Zero rows came back as 0..=74, so
/// tile 2 -- submitted first -- is the one that ran. Purely positional. The
/// assertion below is still the correct one (a fixed driver must produce the
/// whole tensor regardless of order), so this test stays red until the
/// interrupt path is fixed.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn reversed_task_order_shows_whether_the_failure_is_positional() {
    let shape = shape(112, 112);
    let run = run_tiled_conv(
        &shape,
        Tiling::Tiles(3),
        PingPong::default(),
        Dispatch::KernelTasks(Order::Reversed),
    );
    report("kernel_tasks_reversed", &shape, &run);
    eprintln!(
        "reversed order: zero rows = {} (tile row spans: 0..=37, 38..=74, 75..=111)",
        row_span(&zero_rows(&shape, &run.output))
    );
    assert_matches_oracle(&shape, &run, "kernel_tasks_reversed");
}

//===========================================================================
// The interrupt path. `/proc/interrupts` on the board shows the NPU's three
// shared GIC lines (`fdab0000.npu` and siblings) at byte-identical counts
// before and after a full test run: not one interrupt is delivered. The NPU
// never asserts the line, so the driver's hard handler never runs and every
// job is force-completed ~500 ms later by the scheduler's timeout.
//
// A regcmd is fetched after the driver's AHB register writes, so userspace
// can overwrite `PC_INTERRUPT_MASK` before any block is enabled -- see
// `conv::InterruptMask`. If the register's polarity were the opposite of what
// the TRM documents, the driver would be masking off exactly the two DPU
// completion events it waits for, and these tests would fix it without a
// kernel change.
//
// The signal to watch is wall clock, not just correctness: a job that
// completes cleanly should finish in single-digit milliseconds, against the
// ~510 ms every other test here reports. Correct output alone proves
// nothing, since the reset path already delivers that.
//
// **Hardware result: ruled out.** Both overrides still time out (526 ms /
// 504 ms) and the interrupt counters stay put. These three tests are kept as
// the record of that, and stay red with everything else in this file.
//===========================================================================

/// Does un-masking every interrupt source make a *single-task* job complete
/// cleanly instead of being force-reset?
///
/// Deliberately one tile: this isolates the completion interrupt from
/// multi-task dispatch entirely. Output correctness is already established
/// for this shape, so the assertion that matters is the wall clock.
///
/// **Hardware result: no.** 526 ms, i.e. still the timeout.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn single_tile_with_all_interrupts_unmasked_completes_without_timeout() {
    let shape = shape(4, 4);
    let run = run_tiled_conv_masked(
        &shape,
        Tiling::Tiles(1),
        PingPong::default(),
        Dispatch::SeparateJobs,
        InterruptMask::All,
        PcWriteDomain::TargetPc,
    );
    report("single_tile / mask=All", &shape, &run);
    assert_matches_oracle(&shape, &run, "single_tile / mask=All");
    assert!(
        run.elapsed < Duration::from_millis(400),
        "took {:?}, i.e. still the ~500ms drm_sched timeout -- unmasking every \
         interrupt source did not make the NPU raise its completion IRQ, so \
         the mask polarity is not the explanation. Compare \
         `single_tile_matches_the_oracle`'s own ~517ms.",
        run.elapsed
    );
}

/// Same probe with only the two bits the driver intends. Distinguishes a
/// wrong mask *value* (this fails, `mask=All` passes -> inverted polarity)
/// from a driver write that never lands (both pass).
///
/// **Hardware result: neither.** 504 ms, same as `mask=All` -- both still hit
/// the timeout, so the mask is not the deciding register at all.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn single_tile_with_dpu_interrupts_unmasked_completes_without_timeout() {
    let shape = shape(4, 4);
    let run = run_tiled_conv_masked(
        &shape,
        Tiling::Tiles(1),
        PingPong::default(),
        Dispatch::SeparateJobs,
        InterruptMask::DpuOnly,
        PcWriteDomain::TargetPc,
    );
    report("single_tile / mask=DpuOnly", &shape, &run);
    assert_matches_oracle(&shape, &run, "single_tile / mask=DpuOnly");
    assert!(
        run.elapsed < Duration::from_millis(400),
        "took {:?}, i.e. still the ~500ms drm_sched timeout",
        run.elapsed
    );
}

/// The payoff, if either probe above works: with a real completion IRQ, the
/// driver's `rocket_job_handle_irq()` should re-enter `hw_submit` for tiles 1
/// and 2, and the whole tensor should land from one multi-task job.
///
/// **Hardware result: fails exactly as before**, zero rows 38..=111 -- as
/// expected once the single-tile probes above showed the mask is not the
/// issue.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn row_tiles_as_kernel_tasks_with_all_interrupts_unmasked() {
    let shape = shape(112, 112);
    let run = run_tiled_conv_masked(
        &shape,
        Tiling::Tiles(3),
        PingPong::default(),
        Dispatch::KernelTasks(Order::Forward),
        InterruptMask::All,
        PcWriteDomain::TargetPc,
    );
    report("kernel_tasks / mask=All", &shape, &run);
    eprintln!(
        "kernel_tasks / mask=All: zero rows = {}",
        row_span(&zero_rows(&shape, &run.output))
    );
    assert_matches_oracle(&shape, &run, "kernel_tasks / mask=All");
}

/// The mask override again, tagged the way the kick is (`0x81`) instead of
/// `target_PC | 1` (`0x101`).
///
/// This exists because the two probes above are **inconclusive, not
/// negative**: they emitted `PC_INTERRUPT_MASK` with the `0x101` tag, and this
/// crate has never had positive evidence that a `0x101`-tagged regcmd write
/// lands at all -- the only other ones it emits carry the value zero in the
/// path that works. The kick's `0x81` is the tag with real evidence behind it,
/// since nothing runs without it. See `conv::PcWriteDomain`.
///
/// So: if this one completes in single-digit milliseconds, the mask *was* the
/// problem all along and the earlier probes were simply not reaching the
/// register. If it still takes ~500 ms, then either tag is fine for a PC
/// write and the mask really is exonerated.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn single_tile_with_all_interrupts_unmasked_via_kick_domain_completes_without_timeout() {
    let shape = shape(4, 4);
    let run = run_tiled_conv_masked(
        &shape,
        Tiling::Tiles(1),
        PingPong::default(),
        Dispatch::SeparateJobs,
        InterruptMask::All,
        PcWriteDomain::Kick,
    );
    report("single_tile / mask=All domain=0x81", &shape, &run);
    assert_matches_oracle(&shape, &run, "single_tile / mask=All domain=0x81");
    assert!(
        run.elapsed < Duration::from_millis(400),
        "took {:?}: still the ~500ms drm_sched timeout even with the mask \
         override tagged 0x81 like the kick. Both domain tags now behave the \
         same, so the tag is not the confound and the mask really is \
         exonerated -- move on to whether the DPU ever signals done.",
        run.elapsed
    );
}

/// Multi-task counterpart, run only because it is nearly free once the
/// single-tile probe above exists.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn row_tiles_as_kernel_tasks_unmasked_via_kick_domain() {
    let shape = shape(112, 112);
    let run = run_tiled_conv_masked(
        &shape,
        Tiling::Tiles(3),
        PingPong::default(),
        Dispatch::KernelTasks(Order::Forward),
        InterruptMask::All,
        PcWriteDomain::Kick,
    );
    report("kernel_tasks / mask=All domain=0x81", &shape, &run);
    eprintln!(
        "kernel_tasks / mask=All domain=0x81: zero rows = {}",
        row_span(&zero_rows(&shape, &run.output))
    );
    assert_matches_oracle(&shape, &run, "kernel_tasks / mask=All domain=0x81");
}

/// The real target: one kernel task, the PC walking the tile chain itself
/// with `task_number = 3` and `task_pp_en` set.
///
/// Known to be unreachable through the mainline driver, which pins
/// `PC_TASK_CON.task_number` to 1 on every kick -- kept as a regression
/// witness for that, and so the emission side is ready if the driver gains
/// a task-count passthrough. `RegisterAmount::Driver` is the encoding the
/// driver itself uses; the other two are the superseded guesses.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn row_tiles_as_hardware_chain_driver_amount() {
    let shape = shape(112, 112);
    let run = run_tiled_conv(
        &shape,
        Tiling::Tiles(3),
        PingPong::default(),
        Dispatch::HardwareChain(RegisterAmount::Driver),
    );
    report("hardware_chain_driver_amount", &shape, &run);
    assert_matches_oracle(&shape, &run, "hardware_chain_driver_amount");
}

#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn row_tiles_as_hardware_chain_mesa_amount() {
    let shape = shape(112, 112);
    let run = run_tiled_conv(
        &shape,
        Tiling::Tiles(3),
        PingPong::default(),
        Dispatch::HardwareChain(RegisterAmount::MesaHalvedEven),
    );
    report("hardware_chain_mesa_amount", &shape, &run);
    assert_matches_oracle(&shape, &run, "hardware_chain_mesa_amount");
}

#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn row_tiles_as_hardware_chain_word_count_amount() {
    let shape = shape(112, 112);
    let run = run_tiled_conv(
        &shape,
        Tiling::Tiles(3),
        PingPong::default(),
        Dispatch::HardwareChain(RegisterAmount::KernelWordCount),
    );
    report("hardware_chain_word_count_amount", &shape, &run);
    assert_matches_oracle(&shape, &run, "hardware_chain_word_count_amount");
}

/// What ping-pong is *for*: overlapping tile N+1's register fetch with tile
/// N's compute should shrink the gap between measured wall clock and the
/// design notes' pure-compute cycle estimate. Reports rather than asserts a
/// speedup -- a 112x112 1x1 conv is ~1 cycle/pixel, so its register-fetch
/// overhead is a large fraction of the total and the ratio here is a
/// measurement to read, not a threshold to enforce.
///
/// Only runs the tilings that actually completed; a dispatch that fails the
/// correctness tests above has nothing meaningful to time.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn ping_pong_wall_clock_report() {
    let shape = shape(112, 112);
    for (label, ping_pong, dispatch) in [
        (
            "separate_jobs / armed",
            PingPong::default(),
            Dispatch::SeparateJobs,
        ),
        (
            "kernel_tasks / off",
            PingPong::off(),
            Dispatch::KernelTasks(Order::Forward),
        ),
        (
            "kernel_tasks / armed",
            PingPong::default(),
            Dispatch::KernelTasks(Order::Forward),
        ),
        (
            "hardware_chain / armed",
            PingPong::default(),
            Dispatch::HardwareChain(RegisterAmount::MesaHalvedEven),
        ),
    ] {
        let run = run_tiled_conv(&shape, Tiling::Tiles(3), ping_pong, dispatch);
        let correct = (0..shape.output_height).all(|y| {
            (0..shape.output_width).all(|x| {
                (0..shape.output_channels).all(|channel| {
                    let index =
                        ((y * shape.output_width + x) * shape.output_channels + channel) as usize;
                    run.output[index] == expected_value(y, x)
                })
            })
        });
        eprintln!(
            "{label}: {:?} wall clock, {} cycles estimated, output {}",
            run.elapsed,
            run.estimated_cycles,
            if correct { "correct" } else { "WRONG" }
        );
    }
}
