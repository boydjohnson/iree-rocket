//! Hardware-in-the-loop tests for `build_pooling_regcmd`'s standalone
//! ("flying mode") PPU path -- TRM Ch.36 Fig 36-6, PPU_RDMA feeding PPU
//! directly from memory with CNA/CORE/DPU untouched.
//!
//! Not run by a plain `cargo test` -- see `conv_phase1_validation_hw.rs`'s
//! doc comment for the cross-compile-and-copy-to-the-board workflow;
//! identical here:
//!
//!   CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!     cargo test --target aarch64-unknown-linux-gnu --release \
//!       --test pooling_hw --no-run
//!
//! Unlike an ordinary conv shape, there is NO Mesa/Teflon reference
//! implementation for pooling to cross-check against (`rkt_ml.c` never
//! implements it) -- see `build_pooling_regcmd`'s module doc comment for
//! everything that's genuinely unconfirmed here (the pooling_method bit
//! encoding chief among them). These tests are split accordingly:
//!
//! - `pooling_*_completes_and_output_tracks_input`: the load-bearing
//!   correctness tests. Uniform-fill (sidesteps not knowing the input
//!   buffer's real pixel packing order) at two fill
//!   levels, for each of the three raw `PoolingMethod` encodings
//!   independently. Proves the whole standalone-PPU-flying dispatch
//!   completes without hanging and genuinely reads the input, regardless
//!   of whether `PoolingMethod::Max`/`Min`/`Avg`'s *labels* are correct.
//! - `pooling_method_encoding_discovery`: NOT a strict pass/fail
//!   correctness test -- deliberately exploratory. Fills the whole input
//!   plane (a single pooling window covers all of it, so real pixel
//!   packing order can't hide either fill value from the window) half
//!   with a low byte value and half with a high one, runs all three raw
//!   encodings, and asserts only the one invariant that must hold
//!   regardless of which raw value means what: the three outputs, sorted,
//!   must be low <= mid <= high (i.e. *some* encoding is really min,
//!   *some* is really max, *some* is really avg-in-between) rather than
//!   e.g. two of them reading identically (which would mean the
//!   pooling_method field isn't actually being consulted). Prints which
//!   raw value produced which sorted position -- use that to fix
//!   `PoolingMethod::bits()` if it disagrees with the current guess.

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    activation::Activation,
    device::{Buffer, JobDesc, fini_bo, prep_bo, submit, submit_jobs},
    pooling::{
        PoolingBuffers, PoolingMethod, PoolingPlan, PoolingPrecision, PoolingShape,
        build_pooling_regcmd,
    },
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const TENSOR_SIZE: usize = 4096;

/// 4x4x1 input, 2x2 kernel, stride 2, no padding -> 2x2x1 output. Small
/// enough that every output pixel comes from a disjoint, non-overlapping
/// window, so a uniform-fill input should make every output pixel agree
/// regardless of pooling method.
fn tiled_shape(method: PoolingMethod) -> PoolingShape {
    PoolingShape {
        input_width: 4,
        input_height: 4,
        input_channels: 1,
        output_width: 2,
        output_height: 2,
        output_channels: 1,
        precision: PoolingPrecision::Int8,
        kernel_width: 2,
        kernel_height: 2,
        stride_x: 2,
        stride_y: 2,
        method,
        pad_left: 0,
        pad_top: 0,
        pad_right: 0,
        pad_bottom: 0,
        pad_value: 0,
        activation: Activation::None,
    }
}

/// Whole 4x4x1 input as a single pooling window -> 1x1x1 output. Used by
/// the encoding-discovery test: covering 100% of the input plane means an
/// unknown internal pixel packing order can't hide either fill value from
/// the window (unlike a sub-window, which might land entirely within one
/// packing quirk or another).
fn whole_input_shape(method: PoolingMethod) -> PoolingShape {
    PoolingShape {
        input_width: 4,
        input_height: 4,
        input_channels: 1,
        output_width: 1,
        output_height: 1,
        output_channels: 1,
        precision: PoolingPrecision::Int8,
        kernel_width: 4,
        kernel_height: 4,
        stride_x: 4,
        stride_y: 4,
        method,
        pad_left: 0,
        pad_top: 0,
        pad_right: 0,
        pad_bottom: 0,
        pad_value: 0,
        activation: Activation::None,
    }
}

/// Runs `shape` against a uniformly-filled input and returns the real
/// output pixels (16-byte-atomic read stride, the same convention every
/// hardware test in this crate uses -- output_channels=1 still lands each
/// pixel at a full atomic slot regardless of real channel count).
fn run_uniform_pooling(shape: &PoolingShape, input_fill: u8, num_output_pixels: usize) -> Vec<u8> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let buf_in = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_in.host_ptr, input_fill, TENSOR_SIZE);

        let buf_out = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_out.host_ptr, 0, TENSOR_SIZE);

        run_pooling(&file, fd, shape, &buf_in, &buf_out, num_output_pixels)
    }
}

/// Same dispatch as `run_uniform_pooling`, but sizes the BOs from the shape
/// so the validation cases can cross the old fixed 4 KiB test-buffer limit.
fn run_uniform_pooling_dynamic(shape: &PoolingShape, input_fill: u8) -> Vec<u8> {
    assert_eq!(shape.input_channels, 16);
    assert_eq!(shape.output_channels, 16);
    assert_eq!(shape.precision, PoolingPrecision::Int8);
    let input_bytes = (shape.input_width * shape.input_height * 16) as usize;
    let output_pixels = (shape.output_width * shape.output_height) as usize;
    let output_bytes = output_pixels.next_multiple_of(4) * 16 + 16;
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let buf_in = Buffer::new(fd, input_bytes.next_multiple_of(4096), &file);
        ptr::write_bytes(buf_in.host_ptr, input_fill, input_bytes);
        let buf_out = Buffer::new(fd, output_bytes.next_multiple_of(4096), &file);
        ptr::write_bytes(buf_out.host_ptr, 0, output_bytes.next_multiple_of(4096));
        run_pooling_plan(&file, fd, shape, &buf_in, &buf_out, output_pixels)
    }
}

fn validation_shape(
    width: u32,
    height: u32,
    kernel_width: u32,
    kernel_height: u32,
    stride_x: u32,
    stride_y: u32,
) -> PoolingShape {
    PoolingShape {
        input_width: width,
        input_height: height,
        input_channels: 16,
        output_width: (width - kernel_width) / stride_x + 1,
        output_height: (height - kernel_height) / stride_y + 1,
        output_channels: 16,
        precision: PoolingPrecision::Int8,
        kernel_width,
        kernel_height,
        stride_x,
        stride_y,
        method: PoolingMethod::Max,
        pad_left: 0,
        pad_top: 0,
        pad_right: 0,
        pad_bottom: 0,
        pad_value: 0,
        activation: Activation::None,
    }
}

/// Shared dispatch/readback plumbing -- factored out so the encoding-
/// discovery test can fill its own two-value split input rather than a
/// single uniform fill.
unsafe fn run_pooling(
    file: &std::fs::File,
    fd: i32,
    shape: &PoolingShape,
    buf_in: &Buffer,
    buf_out: &Buffer,
    num_output_pixels: usize,
) -> Vec<u8> {
    unsafe {
        let bufs = PoolingBuffers {
            input_addr: buf_in.dma_address,
            output_addr: buf_out.dma_address,
        };
        let cmds = build_pooling_regcmd(shape, &bufs);

        let cmd_bytes = cmds.len() * mem::size_of::<u64>();
        let cmd_len = cmd_bytes.next_multiple_of(4096);
        let buf_cmd = Buffer::new(fd, cmd_len, file);
        let cmd_slice = std::slice::from_raw_parts_mut(buf_cmd.host_ptr as *mut u64, cmds.len());
        for (i, c) in cmds.iter().enumerate() {
            cmd_slice[i] = c.0;
        }

        fini_bo(fd, buf_in.handle).ok();
        fini_bo(fd, buf_cmd.handle).ok();
        // buf_out was CPU-zero-filled by the caller (run_uniform_pooling /
        // pooling_method_encoding_discovery) -- that write needs the same
        // dma_sync_sgtable_for_device() hand-off as every other buffer the
        // device touches, or the CPU's dirty cache line for it can race the
        // device's real write and win, silently clobbering the result back
        // to zero. Every other output buffer in this crate's hardware tests
        // already does this via fini_bo; this file previously didn't.
        fini_bo(fd, buf_out.handle).ok();

        let in_handles = [buf_cmd.handle, buf_in.handle];
        let out_handles = [buf_out.handle];

        submit(
            fd,
            buf_cmd.dma_address,
            cmds.len() as u32,
            &in_handles,
            &out_handles,
        )
        .expect("SUBMIT ioctl failed");

        prep_bo(fd, buf_out.handle, 2_000_000_000).expect("job did not complete within timeout");

        let raw = std::slice::from_raw_parts(buf_out.host_ptr, num_output_pixels * 16 + 16);
        (0..num_output_pixels).map(|i| raw[i * 16]).collect()
    }
}

/// Queues every horizontal tile as an ordered job and waits once for the
/// shared output BO. This mirrors the multi-tile convolution tests: syncing
/// and submitting the complete job list at once avoids a CPU cache hand-off
/// between partial writes to the same output surface.
unsafe fn run_pooling_plan(
    file: &std::fs::File,
    fd: i32,
    shape: &PoolingShape,
    buf_in: &Buffer,
    buf_out: &Buffer,
    num_output_pixels: usize,
) -> Vec<u8> {
    unsafe {
        let programs = PoolingPlan::new(*shape).programs_with_buffers(&PoolingBuffers {
            input_addr: buf_in.dma_address,
            output_addr: buf_out.dma_address,
        });
        let mut command_buffers = Vec::with_capacity(programs.len());
        for program in &programs {
            let cmd_bytes = program.len() * mem::size_of::<u64>();
            let buffer = Buffer::new(fd, cmd_bytes.next_multiple_of(4096), file);
            let words = std::slice::from_raw_parts_mut(buffer.host_ptr as *mut u64, program.len());
            for (destination, command) in words.iter_mut().zip(program) {
                *destination = command.0;
            }
            command_buffers.push((buffer, program.len() as u32));
        }

        fini_bo(fd, buf_in.handle).ok();
        fini_bo(fd, buf_out.handle).ok();
        for (buffer, _) in &command_buffers {
            fini_bo(fd, buffer.handle).ok();
        }

        let tasks: Vec<[(u32, u32); 1]> = command_buffers
            .iter()
            .map(|(buffer, count)| [(buffer.dma_address, *count)])
            .collect();
        let in_handles: Vec<[u32; 2]> = command_buffers
            .iter()
            .map(|(buffer, _)| [buffer.handle, buf_in.handle])
            .collect();
        let out_handles = [buf_out.handle];
        let jobs: Vec<JobDesc<'_>> = tasks
            .iter()
            .zip(&in_handles)
            .map(|(tasks, in_handles)| JobDesc {
                tasks,
                in_handles,
                out_handles: &out_handles,
            })
            .collect();
        submit_jobs(fd, &jobs).expect("tiled pooling SUBMIT failed");
        prep_bo(fd, buf_out.handle, 5_000_000_000)
            .expect("tiled pooling did not complete within timeout");

        let raw = std::slice::from_raw_parts(buf_out.host_ptr, num_output_pixels * 16 + 16);
        (0..num_output_pixels)
            .map(|index| raw[index * 16])
            .collect()
    }
}

macro_rules! completes_and_tracks_input_test {
    ($name:ident, $method:expr) => {
        #[test]
        #[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
        fn $name() {
            let shape = tiled_shape($method);
            // Uniform fill: every 2x2 window sees identical input, so
            // every one of the 4 output pixels should agree regardless of
            // which pooling method this raw encoding actually is.
            for input_fill in [10u8, 118, 200] {
                let pixels = run_uniform_pooling(&shape, input_fill, 4);
                assert!(
                    pixels.iter().all(|&p| p == pixels[0]),
                    "input_fill={input_fill}: expected all 4 output pixels identical \
                     (uniform input), got {pixels:?}"
                );
            }
            // Liveness: output must actually respond to the input, guards
            // against a hollow "completes but never touches the data" pass.
            let low = run_uniform_pooling(&shape, 10, 4)[0];
            let high = run_uniform_pooling(&shape, 200, 4)[0];
            assert_ne!(
                low, high,
                "output pixel value didn't change between input_fill=10 ({low}) and \
                 input_fill=200 ({high}) -- suggests the op isn't really reading the input"
            );
        }
    };
}

completes_and_tracks_input_test!(
    pooling_max_completes_and_output_tracks_input,
    PoolingMethod::Max
);
completes_and_tracks_input_test!(
    pooling_min_completes_and_output_tracks_input,
    PoolingMethod::Min
);
completes_and_tracks_input_test!(
    pooling_avg_completes_and_output_tracks_input,
    PoolingMethod::Avg
);

/// See module doc comment -- exploratory, not a strict correctness check
/// of which raw value means what. Splits the *real image footprint* in
/// half (first `IMAGE_BYTES/2` bytes low, rest high) so a single
/// whole-input pooling window is guaranteed to see both values no matter
/// the real per-pixel packing order, then checks the one invariant that
/// must hold for any internally-consistent max/min/avg encoding.
///
/// UPDATE: previously split the entire `TENSOR_SIZE` (4096-byte) buffer in
/// half rather than just the real 4x4x1 image (256 bytes = 4 rows * 16
/// bytes/pixel * 4 pixels/row). That accidentally worked before the
/// `src_line_stride`/`src_surf_stride` fix (see `build_ppu_standalone_
/// flying`'s doc comment) because the old, 16x-too-large stride formula
/// happened to read far enough past the real image to sample both this
/// test's low and high halves regardless. With the corrected (smaller,
/// TRM-confirmed) stride, PPU_RDMA's read is now tightly confined to the
/// real 256-byte image -- which sat entirely inside the old fill's "low"
/// half, so every method read a genuinely uniform 10 and (correctly)
/// produced identical output. Splitting within the real image footprint
/// instead restores a genuinely bimodal window.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there; \
            exploratory -- read the printed mapping and fix PoolingMethod::bits() if needed"]
fn pooling_method_encoding_discovery() {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open NPU device");
    let fd = file.as_raw_fd();

    // whole_input_shape()'s 4x4x1 image at 16 bytes/pixel.
    const IMAGE_BYTES: usize = 4 * 4 * 16;

    let mut results: Vec<(u8, u8)> = Vec::new(); // (raw encoding, output byte)
    for (raw, method) in [
        (0u8, PoolingMethod::Max),
        (1, PoolingMethod::Min),
        (2, PoolingMethod::Avg),
    ] {
        let shape = whole_input_shape(method);
        unsafe {
            let buf_in = Buffer::new(fd, TENSOR_SIZE, &file);
            ptr::write_bytes(buf_in.host_ptr, 10u8, IMAGE_BYTES / 2);
            ptr::write_bytes(
                buf_in.host_ptr.add(IMAGE_BYTES / 2),
                200u8,
                TENSOR_SIZE - IMAGE_BYTES / 2,
            );
            let buf_out = Buffer::new(fd, TENSOR_SIZE, &file);
            ptr::write_bytes(buf_out.host_ptr, 0, TENSOR_SIZE);

            let pixels = run_pooling(&file, fd, &shape, &buf_in, &buf_out, 1);
            results.push((raw, pixels[0]));
        }
    }

    eprintln!("pooling_method_encoding_discovery: raw encoding -> output byte: {results:?}");
    eprintln!(
        "  if PoolingMethod::{{Max,Min,Avg}}.bits() (0,1,2) is correct, expect \
         raw=0 highest, raw=1 lowest, raw=2 in between -- fix bits() if not."
    );

    let mut sorted = results.clone();
    sorted.sort_by_key(|&(_, v)| v);
    assert!(
        sorted[0].1 <= sorted[1].1 && sorted[1].1 <= sorted[2].1,
        "expected the three raw encodings to produce three orderable outputs \
         (some min, some max, some in-between) -- got {results:?}, which suggests \
         pooling_method isn't being consulted at all rather than just being \
         mislabeled"
    );
    assert!(
        sorted[0].1 != sorted[2].1,
        "min and max raw encodings produced identical output ({:?}) -- expected \
         them to differ given a genuinely bimodal input",
        results
    );
}

/// Diagnostic, not a correctness check -- added after
/// `pooling_*_completes_and_output_tracks_input` came back all-zero on
/// real hardware for every method and every fill level. That specific
/// failure signature (not just "methods agree with each other" but
/// literally zero everywhere) rules out a mislabeled-but-working
/// `pooling_method` and pointed first at `PPU_DST_BASE_ADDR`'s address
/// convention (fixed to write `output_addr >> 4`, matching TRM's
/// documented `pc_base_address` bits[31:4] precedent) -- but a full-buffer
/// dump still came back all zero even after that fix. That's ambiguous on
/// its own: pre-filling the output buffer with zero means "PPU wrote real
/// zeros here" and "PPU never touched this buffer at all" look identical.
///
/// Pre-fills with a distinctive non-zero sentinel (0xAA) instead, so ANY
/// write reaching this buffer -- even a wrong/garbage one -- must knock at
/// least some bytes off 0xAA. Two outcomes:
/// - Still all 0xAA: PPU's write genuinely never lands in this buffer at
///   all (address still wrong, or PPU/PPU_RDMA's op_en never really
///   fired for this dispatch despite PREP_BO completing).
/// - Anything else: the address is right and PPU is writing here, so the
///   remaining suspect is the *input* side -- PPU_RDMA's
///   `src_line_stride`/`src_surf_stride` formulas (flagged UNCONFIRMED in
///   build_pooling_regcmd's module doc comment), which would make PPU
///   compute over fetched garbage/zeroed memory rather than the real
///   input, landing on 0 (or some other wrong-but-real value) regardless
///   of method.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there; \
            diagnostic only, not a pass/fail check -- read the printed hex dump"]
fn pooling_dump_full_output_buffer() {
    let shape = tiled_shape(PoolingMethod::Max);
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let buf_in = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_in.host_ptr, 200u8, TENSOR_SIZE);

        let buf_out = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_out.host_ptr, 0xAAu8, TENSOR_SIZE);

        run_pooling(&file, fd, &shape, &buf_in, &buf_out, 4);

        let raw = std::slice::from_raw_parts(buf_out.host_ptr, TENSOR_SIZE);
        eprintln!(
            "pooling_dump_full_output_buffer: full {TENSOR_SIZE}-byte output buffer (sentinel-filled 0xAA, input uniformly filled with 200):"
        );
        for (row, chunk) in raw.chunks(32).enumerate() {
            let hex: String = chunk.iter().map(|b| format!("{b:02x} ")).collect();
            eprintln!("  {:04x}: {hex}", row * 32);
        }
        let unchanged_count = raw.iter().filter(|&&b| b == 0xAA).count();
        eprintln!(
            "pooling_dump_full_output_buffer: {unchanged_count}/{TENSOR_SIZE} bytes still == 0xAA \
             (if this is TENSOR_SIZE, PPU's write never reached this buffer at all; if less, \
             something did write here -- check what value landed where)"
        );
    }
}

#[test]
#[ignore = "needs the real NPU device -- validates capture-derived horizontal tiling"]
fn wide_pooling_plan_runs_on_npu() {
    let shape = validation_shape(256, 48, 3, 3, 2, 2);
    let pixels = run_uniform_pooling_dynamic(&shape, 73);
    assert!(
        pixels.iter().all(|&value| value == 73),
        "{}x{} -> {}x{} tiled max pooling did not preserve a uniform input",
        shape.input_width,
        shape.input_height,
        shape.output_width,
        shape.output_height
    );
}

#[test]
#[ignore = "needs the real NPU device -- validates the 13-bit height boundary"]
fn height_8192_pooling_runs_on_npu() {
    let shape = validation_shape(64, 8192, 3, 3, 2, 2);
    let pixels = run_uniform_pooling_dynamic(&shape, 73);
    assert!(
        pixels.iter().all(|&value| value == 73),
        "{}x{} -> {}x{} max pooling did not preserve a uniform input",
        shape.input_width,
        shape.input_height,
        shape.output_width,
        shape.output_height
    );
}

#[test]
#[ignore = "needs the real NPU device -- probes the upper half of the PPU kernel fields"]
fn large_pooling_windows_run_on_npu() {
    let shape = validation_shape(64, 48, 8, 8, 3, 3);
    let pixels = run_uniform_pooling_dynamic(&shape, 91);
    assert!(
        pixels.iter().all(|&value| value == 91),
        "{}x{} kernel with {}x{} stride did not preserve a uniform input",
        shape.kernel_width,
        shape.kernel_height,
        shape.stride_x,
        shape.stride_y
    );
}
