//! Hardware-in-the-loop, oracle-based tests for `build_unary_regcmd`
//! (`elementwise.rs`'s single-tensor EW-ALU task: abs/negf/floor/ceil via
//! `ew_alu_algo` 5/6/7/8).
//!
//! Unlike the LUT-based ops (`sigmoid`/`tanh`/`exp`), these have no vendor
//! capture behind them at all -- there's nothing to reverse-engineer, the
//! TRM's opcode table plus this crate's own established `EW_CFG`/`DPU_RDMA`
//! register recipe (`build_add_regcmd`) is the whole derivation. The
//! correctness bar here is a computed CPU oracle (`f32::abs`/`-x`/
//! `f32::floor`/`f32::ceil`) compared against real hardware output, exact
//! tolerance -- these are exact integer-ish operations on fp16 inputs
//! chosen to be exactly representable before and after the op, so there is
//! no quantization slop to allow for (unlike a LUT-approximated
//! transcendental).
//!
//! Cross-compile this test, copy the resulting binary to the RK3588 board
//! (`planck`), and run the ignored tests there:
//!
//! ```text
//! CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!   cargo test --target aarch64-unknown-linux-gnu --release \
//!     --test ew_unary_hw --no-run
//!
//! ./ew_unary_hw-<hash> --ignored --nocapture
//! ```
//!
//! **First hardware round for this task shape** -- `build_unary_regcmd`
//! has never run on real silicon (see its own doc comment for exactly
//! which register choices are inferred vs. carried over confirmed from
//! `build_add_regcmd`). If these tests hang, suspect the `ew_op_src=0`/
//! zeroed-operand choice or the explicit `ERDMA_DISABLE=1` first --
//! neither is independently hardware-confirmed.
//!
//! **`channels=1` only, deliberately.** `build_add_regcmd`'s own
//! multi-channel output byte layout has never been confirmed on real
//! hardware either (its `DST_SURF_STRIDE`/`SURFACE_ADD` use `width*height`
//! with no channel factor, unlike `build_lut_regcmd`'s `width*height*
//! task_channels` -- see `elementwise.rs`'s module doc comment), so a
//! multi-channel oracle check here would be asserting against a guessed
//! layout on top of an unconfirmed register recipe. Staying at
//! `channels=1` collapses output to a flat, unambiguous `width*height`
//! array of raw fp16 values and isolates the one thing this file actually
//! sets out to prove: does each ALU opcode compute the right per-pixel
//! value. Same restraint `conv_with_add_hw.rs` used for its own first
//! hardware round. A multi-channel round is real follow-up work, not done
//! here.

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    device::{Buffer, fini_bo, prep_bo, submit},
    elementwise::{EwUnaryAlgo, EwUnaryBuffers, EwUnaryShape, build_unary_regcmd},
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const TENSOR_SIZE: usize = 4096;
const WIDTH: u32 = 4;
const HEIGHT: u32 = 4;

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
            let mut exponent = -1i32;
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

unsafe fn fill_f16(ptr: *mut u8, byte_len: usize, value: f32) {
    let word = f32_to_f16(value);
    unsafe {
        let slice = std::slice::from_raw_parts_mut(ptr as *mut u16, byte_len / 2);
        slice.fill(word);
    }
}

fn unary_shape(algo: EwUnaryAlgo) -> EwUnaryShape {
    EwUnaryShape {
        width: WIDTH,
        height: HEIGHT,
        channels: 1,
        algo,
        operand: 0,
    }
}

/// Builds and submits the single-task unary EW-ALU job, waits once on the
/// output, and returns the `width*height` real (decoded fp16) output
/// pixels in row-major order.
fn run_unary(x_fill: f32, algo: EwUnaryAlgo) -> Vec<f32> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let buf_in = Buffer::new(fd, TENSOR_SIZE, &file);
        fill_f16(buf_in.host_ptr, TENSOR_SIZE, x_fill);

        let buf_out = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_out.host_ptr, 0, TENSOR_SIZE);

        let bufs = EwUnaryBuffers {
            input_addr: buf_in.dma_address,
            output_addr: buf_out.dma_address,
        };
        let cmds = build_unary_regcmd(&unary_shape(algo), &bufs);

        let cmd_bytes = cmds.len() * mem::size_of::<u64>();
        let cmd_len = cmd_bytes.next_multiple_of(4096);
        let buf_cmd = Buffer::new(fd, cmd_len, &file);
        let cmd_slice = std::slice::from_raw_parts_mut(buf_cmd.host_ptr as *mut u64, cmds.len());
        for (i, c) in cmds.iter().enumerate() {
            cmd_slice[i] = c.0;
        }

        fini_bo(fd, buf_in.handle).ok();
        fini_bo(fd, buf_out.handle).ok();
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

        prep_bo(fd, buf_out.handle, 2_000_000_000).unwrap_or_else(|e| {
            panic!(
                "unary EW-ALU job did not complete within timeout (x_fill={x_fill}, \
                 algo={algo:?}) -- see this file's top doc comment, this is the first \
                 hardware round for this task shape: {e}"
            )
        });

        let raw =
            std::slice::from_raw_parts(buf_out.host_ptr as *const u16, (WIDTH * HEIGHT) as usize);
        let pixels = raw.iter().map(|&bits| f16_to_f32(bits)).collect();

        iree_rocket_hal::rocket::device::close_bo(fd, buf_in.handle).ok();
        iree_rocket_hal::rocket::device::close_bo(fd, buf_out.handle).ok();
        iree_rocket_hal::rocket::device::close_bo(fd, buf_cmd.handle).ok();

        pixels
    }
}

/// Generic oracle check: every one of the `width*height` output pixels
/// (uniform-fill inputs, so every pixel has the same expected value) must
/// match `expected` within `tolerance`.
fn assert_matches_oracle(label: &str, got: &[f32], expected: f32, tolerance: f32) {
    let mut mismatches = 0;
    let mut samples = Vec::new();
    for (i, &value) in got.iter().enumerate() {
        if (value - expected).abs() > tolerance {
            mismatches += 1;
            if samples.len() < 4 {
                samples.push(format!("[{i}] want {expected} got {value}"));
            }
        }
    }
    assert_eq!(
        mismatches,
        0,
        "{label}: {mismatches}/{} pixels differ from oracle by more than {tolerance}:\n  {}",
        got.len(),
        samples.join("\n  ")
    );
}

/// Most basic possible check: does the new task even complete without
/// hanging the NPU? See this file's top doc comment on what a failure
/// here would/wouldn't tell us.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn ew_unary_completes() {
    let out = run_unary(-2.5, EwUnaryAlgo::Abs);
    eprintln!("ew_unary_completes: output={out:?}");
}

#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn ew_unary_abs_matches_oracle() {
    for x in [-6.0f32, -4.5, -2.0, -0.5, 0.0, 0.5, 2.0, 4.5, 6.0] {
        let got = run_unary(x, EwUnaryAlgo::Abs);
        assert_matches_oracle(&format!("abs(x={x})"), &got, x.abs(), 0.0);
    }
}

#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn ew_unary_neg_matches_oracle() {
    for x in [-6.0f32, -4.5, -2.0, -0.5, 0.0, 0.5, 2.0, 4.5, 6.0] {
        let got = run_unary(x, EwUnaryAlgo::Neg);
        assert_matches_oracle(&format!("neg(x={x})"), &got, -x, 0.0);
    }
}

#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn ew_unary_floor_matches_oracle() {
    for x in [-4.5f32, -3.25, -1.5, -0.5, 0.0, 0.5, 1.5, 3.25, 4.5] {
        let got = run_unary(x, EwUnaryAlgo::Floor);
        assert_matches_oracle(&format!("floor(x={x})"), &got, x.floor(), 0.0);
    }
}

#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn ew_unary_ceil_matches_oracle() {
    for x in [-4.5f32, -3.25, -1.5, -0.5, 0.0, 0.5, 1.5, 3.25, 4.5] {
        let got = run_unary(x, EwUnaryAlgo::Ceil);
        assert_matches_oracle(&format!("ceil(x={x})"), &got, x.ceil(), 0.0);
    }
}
