//! Hardware-in-the-loop tests for `build_conv_regcmd`'s 3x3-kernel path,
//! both single- and multi-input-channel.
//!
//! Not run by a plain `cargo test` -- this crate targets the board's NPU
//! (`/dev/accel/accel0`), which doesn't exist on the host doing the
//! building. Cross-compile the test binary and copy it over instead:
//!
//!   CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!     cargo test --target aarch64-unknown-linux-gnu --release \
//!       --test conv_hw --no-run
//!
//! then copy the resulting binary (path printed by `--no-run`, under
//! `target/aarch64-unknown-linux-gnu/release/deps/conv_hw-*`) to the
//! board and run it there as `./conv_hw-<hash> --ignored`.
//!
//! Same shape family and verification strategy as `rkt-shape-b.rs` (see
//! that file's doc comment and rknpu-spelunking/NOTES.md for the full
//! rationale) -- uniform input/weight fill so every tap, and now every
//! input channel too, contributes identically regardless of the
//! still-unknown per-tap/per-channel weight-buffer packing order,
//! sweeping the fill byte near the CVT stage's 128 zero-point. Turned
//! into real `assert!`s instead of eyeballed prints.
//!
//! `input_channels > 1` exercises a genuinely different path through
//! `build_conv_regcmd` than every prior hardware test in this repo
//! (rkt-basic.rs/rkt-shape-a*.rs/rkt-shape-b.rs all used
//! `input_channels: 1`): the `input_channels_real_is_one` branches in
//! CNA_CONV_CON1/CNA_CVT_CON0/CNA_CVT_CON5, and the multi-channel
//! `input_data_entries`/CBUF-geometry formulas, were never hit by real
//! hardware before this.

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    device::{Buffer, fini_bo, prep_bo, submit},
    regcmd::{ConvBuffers, ConvShape, build_conv_regcmd},
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const TENSOR_SIZE: usize = 4096;

/// Shape B's own config: 6x6x1 -> 4x4x1, 3x3 kernel, stride 1.
fn single_channel_shape() -> ConvShape {
    ConvShape {
        input_width: 6,
        input_height: 6,
        input_channels: 1,
        output_width: 4,
        output_height: 4,
        output_channels: 1,
        weights_width: 3,
        weights_height: 3,
        stride: 1,
        depthwise: false,
        input_zero_point: 0,
        output_zero_point: 0,
        weights_zero_point: 0,
        input_scale: 1.0,
        weights_scale: 1.0,
        output_scale: 1.0,
        truncate_bits: 0,
    }
}

/// Identical geometry to `single_channel_shape()`, except a real 3-channel
/// input (e.g. RGB) instead of 1 -- takes `build_conv_regcmd`'s
/// multi-channel branches (`input_channels_real_is_one == false`) instead
/// of the single-channel ones every other hardware test in this repo has
/// exercised so far.
fn multi_channel_shape() -> ConvShape {
    ConvShape {
        input_channels: 3,
        ..single_channel_shape()
    }
}

/// Runs `shape` with the whole input plane filled with `input_fill` and
/// the whole weight buffer filled with `weight_fill`, and returns the 16
/// real output pixels.
///
/// Read at stride 16 bytes/pixel, not stride 1 -- output_channels=1 pads
/// to task_output_channels=32, and the hardware writes each pixel as a
/// full 16-byte-aligned atomic slot regardless of real channel count
/// (see rknpu-spelunking/NOTES.md's "RESOLVED: the only pixel [0,0]
/// written mystery" section).
fn run_uniform_conv(shape: &ConvShape, input_fill: u8, weight_fill: u8) -> Vec<u8> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let buf_a = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_a.host_ptr, input_fill, TENSOR_SIZE);

        let buf_w = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_w.host_ptr, weight_fill, TENSOR_SIZE);

        let buf_bias = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_bias.host_ptr, 0, TENSOR_SIZE);

        let buf_c = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_c.host_ptr, 0, TENSOR_SIZE);

        let bufs = ConvBuffers {
            input_addr: buf_a.dma_address,
            weights_addr: buf_w.dma_address,
            bias_addr: buf_bias.dma_address,
            output_addr: buf_c.dma_address,
        };
        let cmds = build_conv_regcmd(shape, &bufs);

        let cmd_bytes = cmds.len() * mem::size_of::<u64>();
        let cmd_len = cmd_bytes.next_multiple_of(4096);
        let buf_cmd = Buffer::new(fd, cmd_len, &file);
        let cmd_slice = std::slice::from_raw_parts_mut(buf_cmd.host_ptr as *mut u64, cmds.len());
        for (i, c) in cmds.iter().enumerate() {
            cmd_slice[i] = c.0;
        }

        fini_bo(fd, buf_a.handle).ok();
        fini_bo(fd, buf_w.handle).ok();
        fini_bo(fd, buf_bias.handle).ok();
        fini_bo(fd, buf_c.handle).ok();
        fini_bo(fd, buf_cmd.handle).ok();

        let in_handles = [buf_cmd.handle, buf_a.handle, buf_w.handle, buf_bias.handle];
        let out_handles = [buf_c.handle];

        submit(
            fd,
            buf_cmd.dma_address,
            cmds.len() as u32,
            &in_handles,
            &out_handles,
        )
        .expect("SUBMIT ioctl failed");

        prep_bo(fd, buf_c.handle, 2_000_000_000).unwrap_or_else(|e| {
            panic!(
                "job did not complete within timeout (input_channels={}, input_fill={input_fill}): {e}",
                shape.input_channels
            )
        });

        let raw = std::slice::from_raw_parts(buf_c.host_ptr, 256);
        (0..16).map(|i| raw[i * 16]).collect()
    }
}

/// Shape B's own uniform-fill trick sidesteps not knowing the real
/// per-tap weight-buffer packing order: every one of the 9 taps
/// contributes identically, so every one of the 16 output pixels should
/// come out identical regardless of tap ordering.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn conv_3x3_uniform_fill_pixels_agree() {
    let shape = single_channel_shape();
    for input_fill in [10u8, 118, 200] {
        let pixels = run_uniform_conv(&shape, input_fill, 2);
        assert!(
            pixels.iter().all(|&p| p == pixels[0]),
            "input_fill={input_fill}: expected all 16 output pixels identical \
             (uniform input/weights), got {pixels:?}"
        );
    }
}

/// A real multi-tap accumulation should actually respond to the input --
/// guards against a hollow "completes but never touches the data" pass
/// (an all-zero or stuck-at-some-constant output would satisfy the
/// "all pixels agree" check above too).
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn conv_3x3_output_tracks_input() {
    let shape = single_channel_shape();
    let low = run_uniform_conv(&shape, 10, 2)[0];
    let high = run_uniform_conv(&shape, 200, 2)[0];
    assert_ne!(
        low, high,
        "output pixel value didn't change between input_fill=10 ({low}) and \
         input_fill=200 ({high}) -- suggests the op isn't really reading the input"
    );
}

/// Same "all 16 output pixels agree" check as
/// `conv_3x3_uniform_fill_pixels_agree`, but with a genuine 3-channel
/// input instead of 1 -- the uniform whole-buffer fill makes per-channel
/// packing order just as irrelevant as per-tap order was, so this proves
/// real multi-channel accumulation across `build_conv_regcmd`'s
/// previously-untested `input_channels_real_is_one == false` branches.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn conv_3x3_multi_channel_uniform_fill_pixels_agree() {
    let shape = multi_channel_shape();
    for input_fill in [10u8, 118, 200] {
        let pixels = run_uniform_conv(&shape, input_fill, 2);
        assert!(
            pixels.iter().all(|&p| p == pixels[0]),
            "input_channels=3, input_fill={input_fill}: expected all 16 output \
             pixels identical (uniform input/weights), got {pixels:?}"
        );
    }
}

/// Multi-channel counterpart to `conv_3x3_output_tracks_input`.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn conv_3x3_multi_channel_output_tracks_input() {
    let shape = multi_channel_shape();
    let low = run_uniform_conv(&shape, 10, 2)[0];
    let high = run_uniform_conv(&shape, 200, 2)[0];
    assert_ne!(
        low, high,
        "input_channels=3: output pixel value didn't change between \
         input_fill=10 ({low}) and input_fill=200 ({high}) -- suggests the op \
         isn't really reading the input"
    );
}
