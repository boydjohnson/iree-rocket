//! Hardware-in-the-loop, oracle-based test for `square(x) = x*x`, built
//! via the DPU EW core's MUL mode with its "operand from outside" (ERDMA)
//! source deliberately aliased to the same address as the primary input
//! -- see `build_square_regcmd`'s own doc comment for the full mechanism.
//! No LUT needed if this works: `x*x` is exact given `x` is exact in
//! fp16, unlike the LUT-approximated transcendentals.
//!
//! **RESULT: `ew_square_matches_oracle` FAILS on real hardware -- output
//! is all-zero for every input.** `ew_square_completes` passes (the job
//! runs cleanly, no hang), so this isn't a hang/timeout, it's a real
//! wrong-computation result. See `build_square_regcmd`'s doc comment
//! (`elementwise.rs`) for the live hypotheses and why this crate falls
//! back to a Group 4 LUT-based `square` instead of continuing to guess at
//! MUL mode from here. Left in the repo, failing, as a record of what was
//! tried -- same discipline as this project's other documented dead
//! ends/open findings.
//!
//! Cross-compile this test, copy the resulting binary to the RK3588 board
//! (`planck`), and run the ignored tests there:
//!
//! ```text
//! CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!   cargo test --target aarch64-unknown-linux-gnu --release \
//!     --test ew_square_hw --no-run
//!
//! ./ew_square_hw-<hash> --ignored --nocapture
//! ```
//!
//! First hardware round for MUL mode and for ERDMA self-aliasing. Neither
//! has any precedent in this crate: every other EW builder stays in ALU
//! mode, and ERDMA has only ever been pointed at a genuinely distinct
//! second tensor (`EwAddShape`/`build_add_regcmd`).
//!
//! **`channels=1` only**, same reasoning as `ew_unary_hw.rs`/
//! `ew_round_hw.rs`: the multi-channel output byte layout for this task
//! family has never been hardware-confirmed.

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    device::{Buffer, close_bo, fini_bo, prep_bo, submit},
    elementwise::{EwSquareBuffers, EwSquareShape, build_square_regcmd},
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

fn run_square(x_fill: f32) -> Vec<f32> {
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

        let shape = EwSquareShape {
            width: WIDTH,
            height: HEIGHT,
            channels: 1,
        };
        let bufs = EwSquareBuffers {
            input_addr: buf_in.dma_address,
            output_addr: buf_out.dma_address,
        };
        let cmds = build_square_regcmd(&shape, &bufs);

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
                "square job did not complete within timeout (x_fill={x_fill}) -- see this \
                 file's top doc comment, this is the first hardware round for MUL mode and \
                 ERDMA self-aliasing: {e}"
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

#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn ew_square_completes() {
    let out = run_square(-3.0);
    eprintln!("ew_square_completes: output={out:?}");
}

#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn ew_square_matches_oracle() {
    for x in [-6.0f32, -4.5, -2.0, -0.5, 0.0, 0.5, 2.0, 4.5, 6.0] {
        let got = run_square(x);
        assert_matches_oracle(&format!("square(x={x})"), &got, x * x, 0.0);
    }
}
