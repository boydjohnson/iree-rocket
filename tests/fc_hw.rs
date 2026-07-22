//! Hardware-in-the-loop tests for `build_fc_regcmd` -- Phase 2 of the
//! ukernel roadmap. FC as a 1x1-kernel conv, M mapped onto `input_width`,
//! height fixed internally at `FC_SAFE_HEIGHT` (see `regcmd.rs`'s FC
//! section doc comment for the full rationale, especially the
//! `input_height >= 4` underflow risk this sidesteps).
//!
//! Not run by a plain `cargo test` -- see conv_hw.rs's doc comment for the
//! cross-compile-and-copy-to-the-board workflow; identical here:
//!
//!   CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!     cargo test --target aarch64-unknown-linux-gnu --release \
//!       --test fc_hw --no-run
//!
//! Smallest atomic-aligned shape per the roadmap plan: m=4, k=16, n=32
//! (k already a full FEATURE_ATOMIC_SIZE group, n a full task_output_
//! channels group -- no padding-related surprises to also debug on the
//! first hardware round).
//!
//! Two kinds of test, same uniform-fill-sidesteps-packing-order strategy
//! established by conv_hw.rs/pooling_hw.rs throughout this crate:
//! - `fc_uniform_fill_*`/`fc_output_tracks_input`: whole-input-buffer
//!   uniform fill, proves the dispatch completes and genuinely reads the
//!   input, regardless of any unknown per-row/per-channel weight packing.
//! - `fc_rows_are_independent`: NOT a uniform fill -- gives each of the 4
//!   logical M positions (columns, since M maps onto width) a distinct
//!   per-column fill value and checks the corresponding output columns
//!   differ from each other in the same relative order. This is the load-
//!   bearing check for the M-as-width mapping itself: if rows/columns were
//!   secretly cross-contaminating (wrong stride math, wrong CBUF geometry
//!   assumption), this is what would catch it -- a uniform fill alone
//!   cannot.

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    device::{Buffer, close_bo, fini_bo, prep_bo, submit},
    regcmd::{Activation, FcBuffers, FcShape, build_fc_regcmd},
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const TENSOR_SIZE: usize = 4096;

/// Smallest atomic-aligned FC shape: m=4 (maps onto input_width, matches
/// FC_SAFE_HEIGHT so the padded cube is square), k=16 (one full
/// FEATURE_ATOMIC_SIZE group, no channel padding), n=32 (one full
/// task_output_channels group, no output-channel padding).
fn small_shape() -> FcShape {
    FcShape {
        m: 4,
        k: 16,
        n: 32,
        input_zero_point: 0,
        output_zero_point: 0,
        weights_zero_point: 0,
        input_scale: 1.0,
        weights_scale: 1.0,
        output_scale: 1.0,
        truncate_bits: 0,
        activation: Activation::None,
    }
}

/// Runs `shape` with the whole input plane filled with `input_fill` and
/// the whole weight buffer filled with `weight_fill`, returning row 0's
/// `m` real output pixels (the only row with real, not garbage, input --
/// see regcmd.rs's FC section doc comment).
///
/// Read at stride 16 bytes/pixel, same convention as conv_hw.rs/
/// pooling_hw.rs -- output_channels padding still lands each pixel at a
/// full 16-byte-aligned atomic slot regardless of real channel count.
fn run_uniform_fc(shape: &FcShape, input_fill: u8, weight_fill: u8) -> Vec<u8> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let buf_in = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_in.host_ptr, input_fill, TENSOR_SIZE);

        let buf_w = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_w.host_ptr, weight_fill, TENSOR_SIZE);

        let buf_bias = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_bias.host_ptr, 0, TENSOR_SIZE);

        let buf_out = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_out.host_ptr, 0, TENSOR_SIZE);

        let pixels = run_fc(&file, fd, shape, &buf_in, &buf_w, &buf_bias, &buf_out);

        close_bo(fd, buf_in.handle).ok();
        close_bo(fd, buf_w.handle).ok();
        close_bo(fd, buf_bias.handle).ok();
        close_bo(fd, buf_out.handle).ok();

        pixels
    }
}

/// Shared dispatch/readback plumbing -- factored out so the per-row-
/// independence test can fill its own per-column pattern instead of a
/// single uniform fill.
unsafe fn run_fc(
    file: &std::fs::File,
    fd: i32,
    shape: &FcShape,
    buf_in: &Buffer,
    buf_w: &Buffer,
    buf_bias: &Buffer,
    buf_out: &Buffer,
) -> Vec<u8> {
    unsafe {
        let bufs = FcBuffers {
            input_addr: buf_in.dma_address,
            weights_addr: buf_w.dma_address,
            bias_addr: buf_bias.dma_address,
            output_addr: buf_out.dma_address,
        };
        let cmds = build_fc_regcmd(shape, &bufs);

        let cmd_bytes = cmds.len() * mem::size_of::<u64>();
        let cmd_len = cmd_bytes.next_multiple_of(4096);
        let buf_cmd = Buffer::new(fd, cmd_len, file);
        let cmd_slice = std::slice::from_raw_parts_mut(buf_cmd.host_ptr as *mut u64, cmds.len());
        for (i, c) in cmds.iter().enumerate() {
            cmd_slice[i] = c.0;
        }

        fini_bo(fd, buf_in.handle).ok();
        fini_bo(fd, buf_w.handle).ok();
        fini_bo(fd, buf_bias.handle).ok();
        fini_bo(fd, buf_out.handle).ok();
        fini_bo(fd, buf_cmd.handle).ok();

        let in_handles = [buf_cmd.handle, buf_in.handle, buf_w.handle, buf_bias.handle];
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
        close_bo(fd, buf_cmd.handle).ok();

        // Row 0 only (h=0) -- rows 1..FC_SAFE_HEIGHT-1 hold real but unread
        // output for garbage input, per regcmd.rs's FC section doc comment.
        // m=4 real output columns, each landing at a 16-byte-aligned slot.
        let raw = std::slice::from_raw_parts(buf_out.host_ptr, shape.m as usize * 16);
        (0..shape.m as usize).map(|i| raw[i * 16]).collect()
    }
}

/// Uniform-fill sanity check: every one of m=4 output columns should agree
/// (same input/weight fill everywhere, so no per-column difference should
/// be possible regardless of any unknown weight-buffer packing order).
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn fc_uniform_fill_columns_agree() {
    let shape = small_shape();
    for input_fill in [10u8, 118, 200] {
        let pixels = run_uniform_fc(&shape, input_fill, 2);
        assert!(
            pixels.iter().all(|&p| p == pixels[0]),
            "input_fill={input_fill}: expected all {} output columns identical \
             (uniform input/weights), got {pixels:?}",
            shape.m
        );
    }
}

/// A real matmul accumulation should actually respond to the input --
/// guards against a hollow "completes but never touches the data" pass.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn fc_output_tracks_input() {
    let shape = small_shape();
    let low = run_uniform_fc(&shape, 10, 2)[0];
    let high = run_uniform_fc(&shape, 200, 2)[0];
    assert_ne!(
        low, high,
        "output column value didn't change between input_fill=10 ({low}) and \
         input_fill=200 ({high}) -- suggests the op isn't really reading the input"
    );
}

/// Load-bearing check for the M-as-width mapping itself: gives each of the
/// 4 logical M positions (columns) a distinct fill value across its whole
/// K depth, and checks the m real output columns track that per-column
/// distinction in the same relative order (monotonically, since a higher
/// uniform input column should produce a higher accumulator everywhere
/// weights/bias are uniform). A uniform whole-buffer fill can't catch a
/// bug where M positions cross-contaminate (wrong per-column stride, wrong
/// CBUF geometry) -- this can.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn fc_columns_are_independent() {
    let shape = small_shape();
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open NPU device");
    let fd = file.as_raw_fd();

    // Column-major per-16-byte-atomic-pixel fill: FEATURE_ATOMIC_SIZE=16
    // bytes per (w, h) position regardless of real channel count, same
    // atomic convention as every other builder in this module. Column w
    // gets fill value col_fill(w), replicated across every row (only row 0
    // is real, but filling all rows identically is simplest and harmless).
    let col_fill = |w: u32| -> u8 { (10 + w * 60) as u8 }; // 10, 70, 130, 190

    unsafe {
        let buf_in = Buffer::new(fd, TENSOR_SIZE, &file);
        for h in 0..4u32 {
            for w in 0..shape.m {
                let row_base = (h * shape.m + w) as usize * 16;
                ptr::write_bytes(buf_in.host_ptr.add(row_base), col_fill(w), 16);
            }
        }

        let buf_w = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_w.host_ptr, 2, TENSOR_SIZE);

        let buf_bias = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_bias.host_ptr, 0, TENSOR_SIZE);

        let buf_out = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_out.host_ptr, 0, TENSOR_SIZE);

        let pixels = run_fc(&file, fd, &shape, &buf_in, &buf_w, &buf_bias, &buf_out);

        close_bo(fd, buf_in.handle).ok();
        close_bo(fd, buf_w.handle).ok();
        close_bo(fd, buf_bias.handle).ok();
        close_bo(fd, buf_out.handle).ok();

        eprintln!(
            "fc_columns_are_independent: per-column input fills {:?} -> output columns {pixels:?}",
            (0..shape.m).map(col_fill).collect::<Vec<_>>()
        );

        assert!(
            pixels.windows(2).all(|w| w[0] <= w[1]),
            "expected output columns to be non-decreasing (input fills strictly \
             increase per column: {:?}), got {pixels:?} -- suggests columns are \
             cross-contaminating rather than independent",
            (0..shape.m).map(col_fill).collect::<Vec<_>>()
        );
        assert!(
            pixels[0] != pixels[shape.m as usize - 1],
            "first and last output columns are identical ({}) despite very \
             different input fills ({} vs {}) -- suggests the op isn't really \
             reading per-column input",
            pixels[0],
            col_fill(0),
            col_fill(shape.m - 1)
        );
    }
}
