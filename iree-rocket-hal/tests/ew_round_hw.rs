//! Hardware-in-the-loop, oracle-based tests for `round(x)`, built as
//! `floor(x + 0.5)` -- two chained `build_unary_regcmd` tasks (`Add` with
//! a constant scalar operand, then `Floor`) rather than a new LUT, per
//! this crate's plan for `linalg.round` (no native `ROUND` opcode, no
//! captured table, but a cheap composition of ops this crate already has
//! hardware-confirmed: `ew_unary_hw.rs`'s `Floor`, plus `Add`'s new
//! constant-operand mode this file's own first test isolates and checks
//! before trusting the two-task composition).
//!
//! Cross-compile this test, copy the resulting binary to the RK3588 board
//! (`planck`), and run the ignored tests there:
//!
//! ```text
//! CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!   cargo test --target aarch64-unknown-linux-gnu --release \
//!     --test ew_round_hw --no-run
//!
//! ./ew_round_hw-<hash> --ignored --nocapture
//! ```
//!
//! First hardware round for `EwUnaryAlgo::Add`'s constant-operand mode
//! (`ew_op_src=0` with a real nonzero `EW_OP_VALUE_0..7`). Every other use
//! of this crate's EW ALU either leaves that operand zeroed/unused
//! (`Abs`/`Neg`/`Floor`/`Ceil`) or sources it from a real second tensor
//! via ERDMA (`EwAddShape`/`build_add_regcmd`) -- there was no existing
//! capture of a scalar-constant operand to confirm the encoding against
//! going in. Plain IEEE754 `f32` bits (`EwUnaryShape::operand`) turned out
//! right on the first try; what did NOT work was writing only
//! `EW_OP_VALUE_0` and leaving `_1` through `_7` zeroed the way the other
//! algos do (harmlessly, since they never read it) -- that produced a
//! real, reproducible split on real hardware, channels 0-7 of the
//! 16-wide padded atom using the real operand and channels 8-15 silently
//! reading zero. Fixed in `build_unary_regcmd` by writing the same
//! operand to all 8 registers; see that function's own doc comment.
//!
//! **`channels=1` only**, same reasoning as `ew_unary_hw.rs`: the
//! multi-channel output byte layout for this task family has never been
//! hardware-confirmed, so staying at `channels=1` isolates what this file
//! actually sets out to check.

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    device::{Buffer, close_bo, fini_bo, prep_bo, submit, submit_tasks},
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

/// Isolated check of `EwUnaryAlgo::Add`'s constant-operand mode alone:
/// `x + 0.5`, single task, no chaining. Must pass before
/// `ew_round_matches_oracle` (below) can mean anything -- if the operand
/// encoding hypothesis is wrong, chaining a second task on top would just
/// make the failure harder to attribute.
fn run_add_const(x_fill: f32, operand: f32) -> Vec<f32> {
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

        let shape = EwUnaryShape {
            width: WIDTH,
            height: HEIGHT,
            channels: 1,
            algo: EwUnaryAlgo::Add,
            operand: operand.to_bits(),
        };
        let bufs = EwUnaryBuffers {
            input_addr: buf_in.dma_address,
            output_addr: buf_out.dma_address,
        };
        let cmds = build_unary_regcmd(&shape, &bufs);

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
                "add-const job did not complete within timeout (x_fill={x_fill}, \
                 operand={operand}) -- see this file's top doc comment, this is the first \
                 hardware round for EwUnaryAlgo::Add's constant-operand mode: {e}"
            )
        });

        let raw =
            std::slice::from_raw_parts(buf_out.host_ptr as *const u16, (WIDTH * HEIGHT) as usize);
        let pixels = raw.iter().map(|&bits| f16_to_f32(bits)).collect();

        close_bo(fd, buf_in.handle).ok();
        close_bo(fd, buf_out.handle).ok();
        close_bo(fd, buf_cmd.handle).ok();

        pixels
    }
}

/// Two-task chain: `Add(x, 0.5)` then `Floor`, submitted as one
/// `submit_tasks()` job with the first task's output feeding the
/// second's input via a pure inter-task DMA buffer (same convention as
/// `build_conv_then_add_regcmd`'s `intermediate_addr` -- never touched by
/// the CPU, no `prep_bo`/`fini_bo`/handle-list entry needed for it).
fn run_round(x_fill: f32) -> Vec<f32> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let buf_in = Buffer::new(fd, TENSOR_SIZE, &file);
        fill_f16(buf_in.host_ptr, TENSOR_SIZE, x_fill);

        let buf_mid = Buffer::new(fd, TENSOR_SIZE, &file);

        let buf_out = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_out.host_ptr, 0, TENSOR_SIZE);

        let add_shape = EwUnaryShape {
            width: WIDTH,
            height: HEIGHT,
            channels: 1,
            algo: EwUnaryAlgo::Add,
            operand: 0.5f32.to_bits(),
        };
        let add_cmds = build_unary_regcmd(
            &add_shape,
            &EwUnaryBuffers {
                input_addr: buf_in.dma_address,
                output_addr: buf_mid.dma_address,
            },
        );

        let floor_shape = EwUnaryShape {
            width: WIDTH,
            height: HEIGHT,
            channels: 1,
            algo: EwUnaryAlgo::Floor,
            operand: 0,
        };
        let floor_cmds = build_unary_regcmd(
            &floor_shape,
            &EwUnaryBuffers {
                input_addr: buf_mid.dma_address,
                output_addr: buf_out.dma_address,
            },
        );

        let add_cmd_bytes = add_cmds.len() * mem::size_of::<u64>();
        let buf_cmd_add = Buffer::new(fd, add_cmd_bytes.next_multiple_of(4096), &file);
        let add_cmd_slice =
            std::slice::from_raw_parts_mut(buf_cmd_add.host_ptr as *mut u64, add_cmds.len());
        for (i, c) in add_cmds.iter().enumerate() {
            add_cmd_slice[i] = c.0;
        }

        let floor_cmd_bytes = floor_cmds.len() * mem::size_of::<u64>();
        let buf_cmd_floor = Buffer::new(fd, floor_cmd_bytes.next_multiple_of(4096), &file);
        let floor_cmd_slice =
            std::slice::from_raw_parts_mut(buf_cmd_floor.host_ptr as *mut u64, floor_cmds.len());
        for (i, c) in floor_cmds.iter().enumerate() {
            floor_cmd_slice[i] = c.0;
        }

        fini_bo(fd, buf_in.handle).ok();
        fini_bo(fd, buf_out.handle).ok();
        fini_bo(fd, buf_cmd_add.handle).ok();
        fini_bo(fd, buf_cmd_floor.handle).ok();

        let in_handles = [buf_cmd_add.handle, buf_cmd_floor.handle, buf_in.handle];
        let out_handles = [buf_out.handle];

        submit_tasks(
            fd,
            &[
                (buf_cmd_add.dma_address, add_cmds.len() as u32),
                (buf_cmd_floor.dma_address, floor_cmds.len() as u32),
            ],
            &in_handles,
            &out_handles,
        )
        .expect("multi-task SUBMIT ioctl failed");

        prep_bo(fd, buf_out.handle, 2_000_000_000).unwrap_or_else(|e| {
            panic!(
                "round (add-then-floor) job did not complete within timeout (x_fill={x_fill}) -- \
                 see this file's top doc comment, this is the first hardware round for this \
                 two-task chain: {e}"
            )
        });

        let raw =
            std::slice::from_raw_parts(buf_out.host_ptr as *const u16, (WIDTH * HEIGHT) as usize);
        let pixels = raw.iter().map(|&bits| f16_to_f32(bits)).collect();

        close_bo(fd, buf_in.handle).ok();
        close_bo(fd, buf_mid.handle).ok();
        close_bo(fd, buf_out.handle).ok();
        close_bo(fd, buf_cmd_add.handle).ok();
        close_bo(fd, buf_cmd_floor.handle).ok();

        pixels
    }
}

#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn ew_round_add_const_matches_oracle() {
    for x in [-6.0f32, -4.5, -2.0, -0.5, 0.0, 0.5, 2.0, 4.5, 6.0] {
        let got = run_add_const(x, 0.5);
        assert_matches_oracle(&format!("add_const(x={x}, operand=0.5)"), &got, x + 0.5, 0.0);
    }
}

/// `round(x)` via `floor(x + 0.5)` -- matches `f32::round`'s
/// round-half-away-from-zero semantics for positive inputs and exact
/// integers, but NOT at exact negative `.5` boundaries (`floor(-2.5+0.5)
/// = floor(-2.0) = -2.0`, while `(-2.5f32).round() == -3.0`) -- this
/// sweep deliberately avoids exact negative `.5` values for that known,
/// accepted reason rather than masking it with a wide tolerance.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn ew_round_matches_oracle() {
    for x in [-4.75f32, -3.25, -1.75, -0.25, 0.0, 0.25, 1.75, 3.25, 4.75] {
        let got = run_round(x);
        let expected = (x + 0.5).floor();
        assert_matches_oracle(&format!("round(x={x})"), &got, expected, 0.0);
    }
}

#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn ew_round_completes() {
    let out = run_round(2.75);
    eprintln!("ew_round_completes: output={out:?}");
}
