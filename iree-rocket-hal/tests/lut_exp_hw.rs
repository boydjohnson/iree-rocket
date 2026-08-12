//! Hardware-in-the-loop, oracle-based test for `exp(x)`, `x <= 0` only --
//! `LutTable::exp()` (`activation.rs`) via the standalone DPU LUT.
//!
//! **Deliberately restricted to `x <= 0`.** `EXP_LO` (the `x >= 0` half,
//! `lut_tables.rs`) is a placeholder, not real data -- `exp(x) > 1` for
//! any `x > 0` immediately exceeds this table format's representable
//! range, so there is no real `x > 0` table to test yet. This file only
//! validates what's real: `EXP_LE` (a genuine vendor capture) combined
//! with `EXP_LUT_BN_SCALE_K` (`activation.rs`, freshly corrected from a
//! borrowed, unconfirmed sigmoid constant to the value independently fit
//! against `EXP_LE`'s own real content) -- exactly softmax's own real
//! usage pattern (max-subtraction guarantees `x <= 0`).
//!
//! Cross-compile this test, copy the resulting binary to the RK3588 board
//! (`planck`), and run the ignored tests there:
//!
//! ```text
//! CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!   cargo test --target aarch64-unknown-linux-gnu --release \
//!     --test lut_exp_hw --no-run
//!
//! ./lut_exp_hw-<hash> --ignored --nocapture
//! ```
//!
//! First hardware round for `EXP_LUT_BN_SCALE_K` -- `LutTable::exp()` has
//! run on hardware before (via the conv-then-lut / softmax paths this
//! table was originally captured for) but never through an independent
//! oracle check of its own domain-scale correctness the way this file
//! does, the same gap that caught `tanh()`'s bug.

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    activation::{LutBuffers, LutShape, LutTable, build_lut_regcmd},
    device::{Buffer, close_bo, fini_bo, prep_bo, submit},
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const TENSOR_SIZE: usize = 4096;

/// Real zero point `0` for both input and output (`0x80` raw),
/// `input_scale=1/32` (covers real `x` down to `-4.0` at the most
/// negative fill, comfortably inside `EXP_LE`'s real domain edge
/// `16384/EXP_LUT_BN_SCALE_K ~= 5.0`), `output_scale=1/128` (`exp(x)` for
/// `x` in `[-4, 0]` ranges `[0.018, 1.0]`, needs headroom up to `1.0`).
fn exp_shape() -> LutShape {
    LutShape {
        width: 4,
        height: 4,
        channels: 16,
        input_zero_point: 0x80,
        output_zero_point: 0x80,
        input_scale: 1.0 / 32.0,
        output_scale: 1.0 / 128.0,
    }
}

fn run_uniform_standalone_lut(shape: &LutShape, table: LutTable, input_fill: u8) -> Vec<u8> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let buf_a = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_a.host_ptr, input_fill, TENSOR_SIZE);

        let buf_c = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_c.host_ptr, 0, TENSOR_SIZE);

        let bufs = LutBuffers {
            input_addr: buf_a.dma_address,
            output_addr: buf_c.dma_address,
        };
        let cmds = build_lut_regcmd(shape, &bufs, table);

        let cmd_bytes = cmds.len() * mem::size_of::<u64>();
        let cmd_len = cmd_bytes.next_multiple_of(4096);
        let buf_cmd = Buffer::new(fd, cmd_len, &file);
        let cmd_slice = std::slice::from_raw_parts_mut(buf_cmd.host_ptr as *mut u64, cmds.len());
        for (i, c) in cmds.iter().enumerate() {
            cmd_slice[i] = c.0;
        }

        fini_bo(fd, buf_a.handle).ok();
        fini_bo(fd, buf_c.handle).ok();
        fini_bo(fd, buf_cmd.handle).ok();

        let in_handles = [buf_cmd.handle, buf_a.handle];
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
                "standalone exp LUT job did not complete within timeout (input_fill=\
                 {input_fill}) -- DPU/MRDMA flying-mode LUT config may have hung the NPU: {e}"
            )
        });

        let raw = std::slice::from_raw_parts(buf_c.host_ptr, 256);
        let pixels = raw[..16].to_vec();

        close_bo(fd, buf_a.handle).ok();
        close_bo(fd, buf_c.handle).ok();
        close_bo(fd, buf_cmd.handle).ok();

        pixels
    }
}

#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn lut_exp_completes() {
    let shape = exp_shape();
    let out = run_uniform_standalone_lut(&shape, LutTable::exp(), 224);
    eprintln!("lut_exp_completes: output={out:?}");
}

#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn lut_exp_matches_oracle() {
    let shape = exp_shape();
    const TOLERANCE_LSB: f32 = 2.0;
    let tolerance = TOLERANCE_LSB * shape.output_scale;

    // Signed int8 fills covering real x in [-4.0, 0.0] only -- EXP_LO
    // (x > 0) is a known placeholder, deliberately not exercised here.
    let fills: [u8; 8] = [
        128, // -128 -> -4.0
        176, // -80  -> -2.5
        220, // -36  -> -1.125
        250, // -6   -> -0.1875
        255, // -1   -> -0.03125
        254, // -2   -> -0.0625
        192, // -64  -> -2.0
        0,   // 0    -> 0.0
    ];

    for &fill in &fills {
        let real_input = (fill as i8) as f32 * shape.input_scale;
        let expected = real_input.exp();

        let raw = run_uniform_standalone_lut(&shape, LutTable::exp(), fill);
        let mut mismatches = 0;
        let mut samples = Vec::new();
        for (i, &byte) in raw.iter().enumerate() {
            let got = (byte as i8) as f32 * shape.output_scale;
            if (got - expected).abs() > tolerance {
                mismatches += 1;
                if samples.len() < 4 {
                    samples.push(format!("[ch{i}] want {expected} got {got}"));
                }
            }
        }
        assert_eq!(
            mismatches,
            0,
            "exp(real_input={real_input}, fill={fill}): {mismatches}/16 channels differ from \
             oracle by more than {tolerance} ({TOLERANCE_LSB} LSB):\n  {}",
            samples.join("\n  ")
        );
    }
}
