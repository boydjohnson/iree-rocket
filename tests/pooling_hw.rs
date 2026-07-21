//! Hardware-in-the-loop tests for `build_pooling_regcmd`'s standalone
//! ("flying mode") PPU path -- TRM Ch.36 Fig 36-6, PPU_RDMA feeding PPU
//! directly from memory with CNA/CORE/DPU untouched.
//!
//! Not run by a plain `cargo test` -- see conv_hw.rs's doc comment for the
//! cross-compile-and-copy-to-the-board workflow; identical here:
//!
//!   CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!     cargo test --target aarch64-unknown-linux-gnu --release \
//!       --test pooling_hw --no-run
//!
//! Unlike conv_hw.rs's conv shapes, there is NO Mesa/Teflon reference
//! implementation for pooling to cross-check against (`rkt_ml.c` never
//! implements it) -- see `build_pooling_regcmd`'s module doc comment for
//! everything that's genuinely unconfirmed here (the pooling_method bit
//! encoding chief among them). These tests are split accordingly:
//!
//! - `pooling_*_completes_and_output_tracks_input`: the load-bearing
//!   correctness tests. Uniform-fill (same trick as conv_hw.rs, sidesteps
//!   not knowing the input buffer's real pixel packing order) at two fill
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
    device::{Buffer, fini_bo, prep_bo, submit},
    regcmd::{PoolingBuffers, PoolingMethod, PoolingShape, build_pooling_regcmd},
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
    }
}

/// Runs `shape` against a uniformly-filled input and returns the real
/// output pixels (same 16-byte-atomic read stride as conv_hw.rs's
/// `run_uniform_conv` -- output_channels=1 still lands each pixel at a
/// full atomic slot regardless of real channel count).
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
/// of which raw value means what. Splits the input plane in half (first
/// TENSOR_SIZE/2 bytes low, rest high) so a single whole-input pooling
/// window is guaranteed to see both values no matter the real per-pixel
/// packing order, then checks the one invariant that must hold for any
/// internally-consistent max/min/avg encoding.
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

    let mut results: Vec<(u8, u8)> = Vec::new(); // (raw encoding, output byte)
    for (raw, method) in [
        (0u8, PoolingMethod::Max),
        (1, PoolingMethod::Min),
        (2, PoolingMethod::Avg),
    ] {
        let shape = whole_input_shape(method);
        unsafe {
            let buf_in = Buffer::new(fd, TENSOR_SIZE, &file);
            ptr::write_bytes(buf_in.host_ptr, 10u8, TENSOR_SIZE / 2);
            ptr::write_bytes(
                buf_in.host_ptr.add(TENSOR_SIZE / 2),
                200u8,
                TENSOR_SIZE - TENSOR_SIZE / 2,
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
