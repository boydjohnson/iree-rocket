//! Hardware-in-the-loop, oracle-based test for `rsqrt(x) = 1/sqrt(x)`,
//! `x` in `[1, 16)` only -- `LutTable::rsqrt()` (`activation.rs`) via the
//! standalone DPU LUT. Self-derived table, no vendor capture -- see
//! `activation.rs`'s `RSQRT_BN_SCALE_K`/`lut_tables::RSQRT_LE`/`_LO` doc
//! comments for the generation methodology and why, unlike
//! `square`/`erf`/`sqrt`, this table is only accurate across PART of its
//! declared domain (`x < 1` clamps to `32767` rather than holding real
//! values -- `rsqrt(x) > 1.0` there, exceeding this encoding's ceiling).
//!
//! Cross-compile this test, copy the resulting binary to the RK3588 board
//! (`planck`), and run the ignored tests there:
//!
//! ```text
//! CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!   cargo test --target aarch64-unknown-linux-gnu --release \
//!     --test lut_rsqrt_hw --no-run
//!
//! ./lut_rsqrt_hw-<hash> --ignored --nocapture
//! ```
//!
//! First hardware round for `LutTable::rsqrt()`. Only exercises `x >= 1`
//! (the accurate region) -- deliberately does not test `x < 1`, since
//! that region is a KNOWN clamp, not a bug to catch.

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    activation::{LutBuffers, LutShape, LutTable, build_lut_regcmd},
    device::{Buffer, close_bo, fini_bo, prep_bo, submit},
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const TENSOR_SIZE: usize = 4096;

/// Real zero point `0` for both input and output (`0x80` raw),
/// `input_scale=1/8` (covers real `x` up to `15.875`, comfortably inside
/// this table's `[1, 16)` accurate domain -- `RSQRT_BN_SCALE_K`'s doc
/// comment), `output_scale=1/128` (`rsqrt(x)` for `x` in `[1, 16)` ranges
/// `(0.25, 1.0]`).
fn rsqrt_shape() -> LutShape {
    LutShape {
        width: 4,
        height: 4,
        channels: 16,
        input_zero_point: 0x80,
        output_zero_point: 0x80,
        input_scale: 1.0 / 8.0,
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
                "standalone rsqrt LUT job did not complete within timeout (input_fill=\
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
fn lut_rsqrt_completes() {
    let shape = rsqrt_shape();
    let out = run_uniform_standalone_lut(&shape, LutTable::rsqrt(), 32);
    eprintln!("lut_rsqrt_completes: output={out:?}");
}

#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn lut_rsqrt_matches_oracle() {
    let shape = rsqrt_shape();
    const TOLERANCE_LSB: f32 = 3.0;
    let tolerance = TOLERANCE_LSB * shape.output_scale;

    // Fills covering real x in [1.0, 15.875) -- this table's accurate
    // region. x < 1 (a known clamp, not a bug) deliberately not tested.
    let fills: [u8; 6] = [
        8,   // 8   -> 1.0
        16,  // 16  -> 2.0
        32,  // 32  -> 4.0
        64,  // 64  -> 8.0
        100, // 100 -> 12.5
        127, // 127 -> 15.875
    ];

    for &fill in &fills {
        let real_input = (fill as i8) as f32 * shape.input_scale;
        let expected = 1.0 / real_input.sqrt();

        let raw = run_uniform_standalone_lut(&shape, LutTable::rsqrt(), fill);
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
            "rsqrt(real_input={real_input}, fill={fill}): {mismatches}/16 channels differ from \
             oracle by more than {tolerance} ({TOLERANCE_LSB} LSB):\n  {}",
            samples.join("\n  ")
        );
    }
}
