//! Hardware-in-the-loop, oracle-based test for `log(x)` (natural log),
//! `x` in roughly `[0.37, 2.0]` only -- `LutTable::log()` (`activation.rs`)
//! via the standalone DPU LUT. Self-derived table, no vendor capture --
//! see `activation.rs`'s `LOG_BN_SCALE_K`/`lut_tables::LOG_LE`/`_LO` doc
//! comments for the generation methodology and why, like `rsqrt`, this
//! table is only accurate across part of its declared domain (`log` is
//! unbounded in both directions, clamping on both sides of the accurate
//! `[1/e, e)` window).
//!
//! Cross-compile this test, copy the resulting binary to the RK3588 board
//! (`planck`), and run the ignored tests there:
//!
//! ```text
//! CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!   cargo test --target aarch64-unknown-linux-gnu --release \
//!     --test lut_log_hw --no-run
//!
//! ./lut_log_hw-<hash> --ignored --nocapture
//! ```
//!
//! First hardware round for `LutTable::log()`. Only exercises `x` in
//! `[0.375, 1.984]` (comfortably inside the accurate `[1/e, e)` window) --
//! deliberately does not test the clamped regions near `x=0` or the
//! domain's far edge, since those are known clamps, not bugs to catch.

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    activation::{LutBuffers, LutShape, LutTable, build_lut_regcmd},
    device::{Buffer, close_bo, fini_bo, prep_bo, submit},
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const TENSOR_SIZE: usize = 4096;

/// Real zero point `0` for both input and output (`0x80` raw),
/// `input_scale=1/64` (max positive fill `127` -> real `x=1.984`, still
/// inside the accurate `[1/e, e)` window -- `LOG_BN_SCALE_K`'s doc
/// comment), `output_scale=1/128` (`log(x)` for `x` in `[0.375, 1.984]`
/// ranges `[-0.98, 0.69]`).
fn log_shape() -> LutShape {
    LutShape {
        width: 4,
        height: 4,
        channels: 16,
        input_zero_point: 0x80,
        output_zero_point: 0x80,
        input_scale: 1.0 / 64.0,
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
                "standalone log LUT job did not complete within timeout (input_fill=\
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
fn lut_log_completes() {
    let shape = log_shape();
    let out = run_uniform_standalone_lut(&shape, LutTable::log(), 64);
    eprintln!("lut_log_completes: output={out:?}");
}

#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn lut_log_matches_oracle() {
    let shape = log_shape();
    const TOLERANCE_LSB: f32 = 3.0;
    let tolerance = TOLERANCE_LSB * shape.output_scale;

    // Fills covering real x in [0.375, 1.984] -- comfortably inside the
    // accurate [1/e, e) window. Clamped regions deliberately not tested.
    let fills: [u8; 6] = [
        24,  // 24  -> 0.375
        32,  // 32  -> 0.5
        48,  // 48  -> 0.75
        64,  // 64  -> 1.0
        96,  // 96  -> 1.5
        127, // 127 -> 1.984375
    ];

    for &fill in &fills {
        let real_input = (fill as i8) as f32 * shape.input_scale;
        let expected = real_input.ln();

        let raw = run_uniform_standalone_lut(&shape, LutTable::log(), fill);
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
            "log(real_input={real_input}, fill={fill}): {mismatches}/16 channels differ from \
             oracle by more than {tolerance} ({TOLERANCE_LSB} LSB):\n  {}",
            samples.join("\n  ")
        );
    }
}
