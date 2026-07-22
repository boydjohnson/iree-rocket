//! Hardware-in-the-loop test for `build_conv_with_add_regcmd` -- Mesa's
//! `add_tensor` element-wise-add fusion, see `AddTensor`'s doc comment in
//! `regcmd.rs` for the full derivation (TRM's documented `ew_alu_algo`
//! opcodes, Mesa's `rkt_regcmd.c` source this is ported from, and the
//! live hardware capture of a standalone `x + y` model that confirmed
//! the resulting register values bit-exact -- see
//! rknpu-spelunking/NOTES.md's "Elementwise tensor-tensor ops" section).
//!
//! Not run by a plain `cargo test` -- see `conv_hw.rs`'s doc comment for
//! the cross-compile-and-copy-to-the-board workflow; identical here.
//!
//! **This is a genuinely first-round, exploratory test**, unlike most
//! other hardware tests in this repo: `build_conv_with_add_regcmd`'s own
//! doc comment flags a real, unresolved gap between what this function
//! assumes (one task, DPU_RDMA's `SRC_BASE_ADDR` and ERDMA's
//! `EW_BASE_ADDR` both readable directly, no preceding stage needed --
//! the same single-task shape `build_lut_regcmd` already proved for a
//! different op) and what the one real hardware capture behind this
//! code actually showed (a 3-task chain, not a single task). This test
//! picks the simplest concrete hypothesis it can -- `addition.src_addr ==
//! addition.ew_addr == buf_y`'s address -- purely to see whether the
//! job completes at all. A hang or wrong output here should NOT be read
//! as "the EW_CFG/EW_CVT_SCALE_VALUE recipe is wrong" (that part is
//! bit-exact confirmed already); suspect the task-count/data-flow gap
//! first.
//!
//! Conv shape mirrors `conv_then_lut_hw.rs`'s own choice: 1x1 kernel,
//! real (non-offset) zero points, to avoid the accumulator-saturation
//! problem a 3x3 kernel + offset zero points caused on this repo's first
//! LUT hardware rounds.

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    device::{Buffer, fini_bo, prep_bo, submit},
    regcmd::{
        Activation, AddTensor, ConvBuffers, ConvShape, build_conv_regcmd,
        build_conv_with_add_regcmd,
    },
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const TENSOR_SIZE: usize = 4096;

fn conv_shape() -> ConvShape {
    ConvShape {
        input_width: 4,
        input_height: 4,
        input_channels: 1,
        output_width: 4,
        output_height: 4,
        output_channels: 1,
        weights_width: 1,
        weights_height: 1,
        stride: 1,
        depthwise: false,
        input_zero_point: 0x80,
        output_zero_point: 0x80,
        weights_zero_point: 0x80,
        input_scale: 1.0,
        weights_scale: 1.0,
        output_scale: 1.0,
        truncate_bits: 0,
        activation: Activation::None,
    }
}

/// Runs `x + y` with the whole `x` plane filled with `x_fill`, the whole
/// `y` plane filled with `y_fill`, a 1x1 identity-ish weight
/// (`weight_fill`), and returns the 16 real output pixels.
fn run_uniform_conv_with_add(x_fill: u8, y_fill: u8, weight_fill: u8) -> Vec<u8> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let buf_x = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_x.host_ptr, x_fill, TENSOR_SIZE);

        let buf_y = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_y.host_ptr, y_fill, TENSOR_SIZE);

        let buf_w = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_w.host_ptr, weight_fill, TENSOR_SIZE);

        let buf_bias = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_bias.host_ptr, 0, TENSOR_SIZE);

        let buf_out = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_out.host_ptr, 0, TENSOR_SIZE);

        let bufs = ConvBuffers {
            input_addr: buf_x.dma_address,
            weights_addr: buf_w.dma_address,
            bias_addr: buf_bias.dma_address,
            output_addr: buf_out.dma_address,
        };
        // Simplest hypothesis for the still-unconfirmed src_addr/ew_addr
        // relationship -- see this file's top doc comment.
        let addition = AddTensor {
            src_addr: buf_y.dma_address,
            ew_addr: buf_y.dma_address,
            scale: 1.0,
            cvt_offset: 0,
        };
        let cmds = build_conv_with_add_regcmd(&conv_shape(), &bufs, &addition);

        let cmd_bytes = cmds.len() * mem::size_of::<u64>();
        let cmd_len = cmd_bytes.next_multiple_of(4096);
        let buf_cmd = Buffer::new(fd, cmd_len, &file);
        let cmd_slice = std::slice::from_raw_parts_mut(buf_cmd.host_ptr as *mut u64, cmds.len());
        for (i, c) in cmds.iter().enumerate() {
            cmd_slice[i] = c.0;
        }

        fini_bo(fd, buf_x.handle).ok();
        fini_bo(fd, buf_y.handle).ok();
        fini_bo(fd, buf_w.handle).ok();
        fini_bo(fd, buf_bias.handle).ok();
        fini_bo(fd, buf_out.handle).ok();
        fini_bo(fd, buf_cmd.handle).ok();

        let in_handles = [
            buf_cmd.handle,
            buf_x.handle,
            buf_y.handle,
            buf_w.handle,
            buf_bias.handle,
        ];
        let out_handles = [buf_out.handle];

        submit(
            fd,
            buf_cmd.dma_address,
            cmds.len() as u32,
            &in_handles,
            &out_handles,
        )
        .expect("SUBMIT ioctl failed");

        prep_bo(fd, buf_out.handle, 2_000_000_000).unwrap_or_else(|e| {
            panic!(
                "conv-with-add job did not complete within timeout (x_fill={x_fill}, \
                 y_fill={y_fill}) -- see this file's top doc comment on the unconfirmed \
                 src_addr/ew_addr data-flow before assuming the EW recipe itself is wrong: {e}"
            )
        });

        let raw = std::slice::from_raw_parts(buf_out.host_ptr, 256);
        (0..16).map(|i| raw[i * 16]).collect()
    }
}

/// Most basic possible check: does this new fused-EW single task even
/// complete without hanging the NPU? See this file's top doc comment on
/// what a failure here would/wouldn't tell us.
///
/// Confirmed on real hardware (2026-07-22): completes, `output=[128;
/// 16]` for every pixel (`x_fill=0, y_fill=0, weight_fill=64`). NOT
/// meaningful as a correctness check on its own -- `weight_fill=64`
/// (copied from `conv_then_lut_hw.rs`'s amplification-test convention,
/// not chosen for this test) decodes to a real weight of `64-128=-64`,
/// not an identity pass-through, so the conv stage here computes
/// `x_real * -64`, not `x_real` -- the observed constant `128` (real
/// output `0`) doesn't confirm or refute add semantics either way, just
/// that the job runs. See `conv_with_add_tracks_x_plus_y` below for an
/// actual numeric check with a real identity weight.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn conv_with_add_completes() {
    let out = run_uniform_conv_with_add(0, 0, 64);
    eprintln!("conv_with_add_completes: output={out:?}");
}

/// Real numeric check, unlike `conv_with_add_completes` above:
/// `weight_fill=0x81` decodes to a real weight of `+1` (`weights_zero_
/// point=0x80` in `conv_shape()`), so the 1x1-kernel, single-channel
/// conv stage computes `x_real * 1 = x_real` exactly, not some
/// arbitrary scaled/negated value -- the only thing left to produce a
/// non-identity result is the EW add itself. Holds `y_fill` fixed at
/// `128` (real `0`) and sweeps `x_fill` through moderate values
/// symmetric around the `128` zero point (avoiding the `-128` extreme
/// `conv_with_add_completes` used, which saturates/wraps differently)
/// -- if the EW unit is really adding `y_real` (here `0`) to the conv's
/// real accumulator (`x_real`), output should track `x_fill` 1:1 and
/// come out strictly increasing across the sweep.
///
/// **FAILS on real hardware, but NOT because of `AddTensor`**: confirmed
/// via `conv_plain_identity_tracks_x` (same identity weight/sweep,
/// plain `build_conv_regcmd`, no `AddTensor` at all) that a real-
/// weight-of-exactly-1 conv accumulator has a pre-existing precision/
/// rounding bug of its own -- unrelated to element-wise add, and NOT
/// something this session's changes caused (see that test's own doc
/// comment and `rknpu-spelunking/NOTES.md`'s "Elementwise tensor-tensor
/// ops" section). This test is left failing/`#[ignore]`d rather than
/// deleted or weakened -- it's a real, valid code path (nonzero
/// accumulator + EW add together) worth revisiting once the conv-side
/// bug is fixed, just not diagnostic for `AddTensor` on its own.
/// `conv_with_add_tracks_y_alone` below is the reliable check for that
/// (holds the accumulator at a clean `0` via `weight_real=0`, sidestepping
/// this bug entirely) -- it passes bit-exact.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn conv_with_add_tracks_x_plus_y() {
    let fills = [118u8, 121, 124, 127, 128, 129, 132, 135, 138];
    let mut prev: Option<u8> = None;

    for x_fill in fills {
        let raw = run_uniform_conv_with_add(x_fill, 128, 0x81)[0];
        eprintln!(
            "conv_with_add_tracks_x_plus_y: x_fill={x_fill} (real={}) y_fill=128 (real=0): raw={raw} (real={})",
            x_fill as i32 - 0x80,
            raw as i32 - 0x80
        );

        if let Some(prev_raw) = prev {
            assert!(
                raw > prev_raw,
                "conv-with-add output is not strictly increasing over the x_fill sweep: \
                 previous_raw={prev_raw}, current_raw={raw}, x_fill={x_fill}"
            );
        }
        prev = Some(raw);
    }
}

/// Isolation test: same shape, same identity weight (`weight_fill=
/// 0x81`), same `x_fill` sweep as `conv_with_add_tracks_x_plus_y`
/// above, but through the plain, already-hardware-validated
/// `build_conv_regcmd` -- NO `AddTensor` fusion at all. Added after
/// `conv_with_add_tracks_x_plus_y` FAILED on real hardware (`x_fill=118`
/// and `x_fill=121`, real accumulator values `-10` and `-7`, both came
/// back as the same `raw=127`) -- this test's only purpose is figuring
/// out whether that flatness lives in the shared conv/out_cvt path
/// every builder in this module reuses, or specifically in the new
/// EW-add wiring: if a PLAIN conv (no EW fusion) also comes back flat
/// for these fills, the bug predates this session's changes entirely;
/// if it correctly tracks `x_fill`, the bug is in `AddTensor`'s own
/// register wiring.
fn run_uniform_conv_plain(x_fill: u8, weight_fill: u8) -> Vec<u8> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let buf_x = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_x.host_ptr, x_fill, TENSOR_SIZE);

        let buf_w = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_w.host_ptr, weight_fill, TENSOR_SIZE);

        let buf_bias = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_bias.host_ptr, 0, TENSOR_SIZE);

        let buf_out = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_out.host_ptr, 0, TENSOR_SIZE);

        let bufs = ConvBuffers {
            input_addr: buf_x.dma_address,
            weights_addr: buf_w.dma_address,
            bias_addr: buf_bias.dma_address,
            output_addr: buf_out.dma_address,
        };
        let cmds = build_conv_regcmd(&conv_shape(), &bufs);

        let cmd_bytes = cmds.len() * mem::size_of::<u64>();
        let cmd_len = cmd_bytes.next_multiple_of(4096);
        let buf_cmd = Buffer::new(fd, cmd_len, &file);
        let cmd_slice = std::slice::from_raw_parts_mut(buf_cmd.host_ptr as *mut u64, cmds.len());
        for (i, c) in cmds.iter().enumerate() {
            cmd_slice[i] = c.0;
        }

        fini_bo(fd, buf_x.handle).ok();
        fini_bo(fd, buf_w.handle).ok();
        fini_bo(fd, buf_bias.handle).ok();
        fini_bo(fd, buf_out.handle).ok();
        fini_bo(fd, buf_cmd.handle).ok();

        let in_handles = [buf_cmd.handle, buf_x.handle, buf_w.handle, buf_bias.handle];
        let out_handles = [buf_out.handle];

        submit(
            fd,
            buf_cmd.dma_address,
            cmds.len() as u32,
            &in_handles,
            &out_handles,
        )
        .expect("SUBMIT ioctl failed");

        prep_bo(fd, buf_out.handle, 2_000_000_000).unwrap_or_else(|e| {
            panic!("plain conv job did not complete within timeout (x_fill={x_fill}): {e}")
        });

        let raw = std::slice::from_raw_parts(buf_out.host_ptr, 256);
        (0..16).map(|i| raw[i * 16]).collect()
    }
}

#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn conv_plain_identity_tracks_x() {
    let fills = [118u8, 121, 124, 127, 128, 129, 132, 135, 138];

    for x_fill in fills {
        let raw = run_uniform_conv_plain(x_fill, 0x81)[0];
        eprintln!(
            "conv_plain_identity_tracks_x: x_fill={x_fill} (real={}): raw={raw} (real={})",
            x_fill as i32 - 0x80,
            raw as i32 - 0x80
        );
    }
}

/// Isolates the EW-add path from the (now separately, pre-existingly
/// broken -- see `conv_plain_identity_tracks_x`'s real hardware result
/// and this file's top doc comment) conv accumulator/`out_cvt` math
/// entirely: `weight_fill=0x80` decodes to a REAL weight of exactly
/// `0` (`weights_zero_point=0x80`), so the conv accumulator is
/// `x_real * 0 = 0` regardless of `x_fill` -- `x_fill` genuinely
/// doesn't matter here and is held constant. If the EW unit really
/// adds `y_real` to a `0` accumulator, output should track `y_fill`
/// 1:1, independent of whatever precision issue affects nonzero
/// accumulator values.
///
/// **PASSES bit-exact on real hardware** (2026-07-22): `raw == y_fill`
/// (equivalently `real_output == y_real`) for every point in the sweep
/// (`y_fill` 118,121,124,127,128,129,132,135,138 -> `raw` identical to
/// each, i.e. real -10,-7,-4,-1,0,1,4,7,10 reproduced exactly). This is
/// the confirming numeric result for `AddTensor`/`build_conv_with_add_
/// regcmd`'s EW-add wiring -- the register recipe (`EW_CFG`, `ERDMA_
/// CFG`, `EW_BASE_ADDR`, `EW_CVT_OFFSET_VALUE`, `EW_CVT_SCALE_VALUE`)
/// is correct, not just non-hanging. Combined with `conv_with_add_
/// completes`, the single-fused-task shape (vs. the 3-task chain the
/// one captured model used) is now validated end-to-end for this
/// `weight_real=0` case; the conv-side precision bug affecting nonzero
/// accumulators remains a separate, open issue (see
/// `conv_plain_identity_tracks_x`).
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn conv_with_add_tracks_y_alone() {
    let fills = [118u8, 121, 124, 127, 128, 129, 132, 135, 138];
    let mut prev: Option<u8> = None;

    for y_fill in fills {
        let raw = run_uniform_conv_with_add(128, y_fill, 0x80)[0];
        eprintln!(
            "conv_with_add_tracks_y_alone: x_fill=128 (real=0, weight_real=0) y_fill={y_fill} (real={}): raw={raw} (real={})",
            y_fill as i32 - 0x80,
            raw as i32 - 0x80
        );

        if let Some(prev_raw) = prev {
            assert!(
                raw > prev_raw,
                "conv-with-add output is not strictly increasing over the y_fill sweep \
                 (accumulator held at 0 via weight_real=0): previous_raw={prev_raw}, \
                 current_raw={raw}, y_fill={y_fill}"
            );
        }
        prev = Some(raw);
    }
}
