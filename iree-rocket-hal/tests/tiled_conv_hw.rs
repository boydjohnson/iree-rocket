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
//! registers are programmed. A failure is then attributable:
//!
//! | test | dispatch | ping-pong | what a failure means |
//! |---|---|---|---|
//! | `single_tile_matches_the_oracle` | one job | armed | `conv.rs`'s tiling/emission is broken independently of everything else |
//! | `row_tiles_as_separate_jobs_match_the_oracle` | job per tile | armed | caller-driven row tiling (as opposed to CBUF-driven splits) is wrong |
//! | `row_tiles_as_kernel_tasks_without_ping_pong` | one job, N tasks | off | reproduces the known failure -- expected to fail until the mechanism is understood |
//! | `row_tiles_as_kernel_tasks_with_ping_pong` | one job, N tasks | armed | CNA/CORE arming alone doesn't fix the kernel-walked path |
//! | `row_tiles_as_hardware_chain_*` | one task, PC-walked chain | armed | the PC's own task walk or the `PC_REGISTER_AMOUNTS` unit is wrong |
//!
//! The two `hardware_chain` variants differ only in
//! [`RegisterAmount`] -- see its doc comment for why that unit is an open
//! question rather than a settled one.

use std::{
    fs::OpenOptions,
    mem,
    os::unix::io::AsRawFd,
    ptr,
    time::{Duration, Instant},
};

use iree_rocket_hal::rocket::{
    conv::{
        PingPong, PointerMode, RegisterAmount, Tiling, build_tiled_conv_regcmds, cycles_per_pixel,
        link_tiled_conv_regcmds, plan_tiled_conv,
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

/// How the tile sequence reaches the hardware. See this file's doc comment
/// table for what each one isolates.
#[derive(Clone, Copy, Debug)]
enum Dispatch {
    /// One DRM job per tile, each fenced from the CPU before the next is
    /// submitted. The arrangement `conv_hw.rs` found working; ping-pong buys
    /// nothing here, since the PC is re-kicked from scratch every time.
    SeparateJobs,
    /// All tiles as one job's task array. Mainline `rocket_job.c` dispatches
    /// task N+1 from task N's completion IRQ.
    KernelTasks,
    /// Only tile 0 submitted; its regcmd's trailing PC link points at tile
    /// 1, and so on. The PC walks the chain itself -- the configuration
    /// ping-pong is actually for.
    HardwareChain(RegisterAmount),
}

struct Run {
    /// Output decoded to dense NHWC order.
    output: Vec<f32>,
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
        let mut tasks = build_tiled_conv_regcmds(&plan, &bufs).expect("tiled convolution regcmds");

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
        let descriptors: Vec<(u32, u32)> = cmd_buffers
            .iter()
            .zip(&tasks)
            .map(|(buf, cmds)| (buf.dma_address, cmds.len() as u32))
            .collect();

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
            Dispatch::KernelTasks => {
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

        // DPU write-back is 16-byte feature-atomic surfaces: surface S holds
        // channel bytes [16S, 16S+16) for every pixel, planar. Same decode
        // as `conv_hw.rs`'s position test.
        let scratch = std::slice::from_raw_parts(buf_out.host_ptr, output_scratch_len);
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

        close_bo(fd, buf_in.handle).ok();
        close_bo(fd, buf_w.handle).ok();
        close_bo(fd, buf_bias.handle).ok();
        close_bo(fd, buf_out.handle).ok();
        for buf_cmd in &cmd_buffers {
            close_bo(fd, buf_cmd.handle).ok();
        }

        Run {
            output,
            elapsed,
            tile_output_rows,
            estimated_cycles,
        }
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
        Dispatch::KernelTasks,
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
        Dispatch::KernelTasks,
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
        Dispatch::KernelTasks,
    );
    report("kernel_tasks_explicit_pointer", &shape, &run);
    assert_matches_oracle(&shape, &run, "kernel_tasks_explicit_pointer");
}

/// The real target: one kernel task, the PC walking the tile chain itself
/// with `task_number = 3` and `task_pp_en` set. The two amount conventions
/// are separate tests so a hang or a wrong result attributes to one of
/// them; see [`RegisterAmount`].
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
        ("kernel_tasks / off", PingPong::off(), Dispatch::KernelTasks),
        (
            "kernel_tasks / armed",
            PingPong::default(),
            Dispatch::KernelTasks,
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
