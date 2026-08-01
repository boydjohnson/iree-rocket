//! Hardware-in-the-loop tests for `build_conv_then_add_regcmd` -- the real
//! `Conv2d(x) +/- w` pipeline, as two hardware tasks in one
//! `device::submit_tasks()` job (see that function's doc comment and
//! `elementwise.rs`'s `ConvThenAddBuffers` doc comment for why this is a
//! genuinely separate task rather than fused into the conv's own DPU pass).
//!
//! Not run by a plain `cargo test` -- see `conv_phase1_validation_hw.rs`'s
//! doc comment for the cross-compile-and-copy-to-the-board workflow;
//! identical here:
//!
//!   CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!     cargo test --target aarch64-unknown-linux-gnu --release \
//!       --test conv_with_add_hw --no-run
//!
//! **First hardware round for this exact two-task shape.** Unlike
//! `conv_then_lut_hw.rs` (whose two-task design was independently confirmed
//! by a live bpftrace trace of the vendor runtime before ever being coded),
//! this builder's task structure and register recipe were reconstructed
//! entirely from a static sweep's decoded captures
//! (`iree-rocket-design-spike`'s `sweep_convadd_generate.py`/
//! `sweep_convadd_diff.py`, see `DESIGN_NOTES.md`'s "Conv+add fusion sweep"
//! section, and `elementwise.rs`'s own doc comments for exactly which
//! fields are capture-confirmed vs. inferred) -- nothing here has run on
//! real silicon yet. If these tests hang, suspect the task-structure/
//! register-skeleton claims first (the confirmed-vs-inferred split in
//! `build_add_regcmd`'s doc comment is the place to start).
//!
//! **fp16 only.** `EwAddShape`'s int8 fields (`w_scale_ratio`/
//! `output_scale_ratio`) are new and have no independent numeric
//! confirmation (see their own doc comments) -- keeping this first real
//! hardware round fp16-only isolates the task-shape/register-skeleton
//! question (genuinely uncertain) from the int8 ratio-formula question
//! (deliberately deferred, not blocking this round).
//!
//! Conv shape mirrors `conv_then_lut_hw.rs`'s own choice: 1x1 kernel, single
//! channel, to keep the expected numeric results simple to reason about.

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    conv::{self, Kernels},
    device::{Buffer, close_bo, fini_bo, prep_bo, submit_tasks},
    elementwise::{ConvThenAddBuffers, EwAddShape, EwPrecision, build_conv_then_add_regcmd},
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const TENSOR_SIZE: usize = 4096;
const KERNELS: Kernels = [1, 1];
const EXTENT: u32 = 4;

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

fn conv_shape() -> conv::Shape {
    conv::Shape {
        width: EXTENT,
        height: EXTENT,
        stride: 1,
        in_channels: 1,
        out_channels: 1,
        precision: conv::Precision::Fp16,
        padding: Some([0, 0]),
        activation: conv::Activation::None,
        depthwise: false,
    }
}

fn add_shape(algo: u32) -> EwAddShape {
    EwAddShape {
        width: EXTENT,
        height: EXTENT,
        channels: 1,
        precision: EwPrecision::Fp16,
        algo,
        // int8-only fields, unused for fp16.
        output_zero_point: 0,
        w_cvt_offset: 0,
        w_scale_ratio: 1.0,
        output_scale_ratio: 1.0,
    }
}

/// Builds and submits the two-task `x <algo> w_fill` (real conv, real
/// weight `weight_fill`, `x_fill` input) job as one `submit_tasks()` job,
/// waits once on the final output, and returns the 16 real (decoded fp16)
/// output pixels.
fn run_conv_then_add(x_fill: f32, weight_fill: f32, w_fill: f32, algo: u32) -> Vec<f32> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let buf_in = Buffer::new(fd, TENSOR_SIZE, &file);
        fill_f16(buf_in.host_ptr, TENSOR_SIZE, x_fill);

        let buf_w = Buffer::new(fd, TENSOR_SIZE, &file);
        fill_f16(buf_w.host_ptr, TENSOR_SIZE, weight_fill);

        let buf_bias = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_bias.host_ptr, 0, TENSOR_SIZE); // real 0.0

        let buf_second = Buffer::new(fd, TENSOR_SIZE, &file);
        fill_f16(buf_second.host_ptr, TENSOR_SIZE, w_fill);

        // Pure inter-task DMA memory -- never read/written by the CPU, see
        // conv_then_lut_hw.rs's own ConvThenLutBuffers::intermediate_addr
        // handling for why this needs neither prep_bo/fini_bo nor a spot in
        // either handle list below.
        let buf_mid = Buffer::new(fd, TENSOR_SIZE, &file);

        let buf_out = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_out.host_ptr, 0, TENSOR_SIZE);

        let bufs = ConvThenAddBuffers {
            input_addr: buf_in.dma_address,
            weights_addr: buf_w.dma_address,
            bias_addr: buf_bias.dma_address,
            intermediate_addr: buf_mid.dma_address,
            w_addr: buf_second.dma_address,
            output_addr: buf_out.dma_address,
        };
        let (conv_cmds, add_cmds) =
            build_conv_then_add_regcmd(&conv_shape(), KERNELS, &add_shape(algo), &bufs);

        let conv_cmd_bytes = conv_cmds.len() * mem::size_of::<u64>();
        let buf_cmd_conv = Buffer::new(fd, conv_cmd_bytes.next_multiple_of(4096), &file);
        let conv_cmd_slice =
            std::slice::from_raw_parts_mut(buf_cmd_conv.host_ptr as *mut u64, conv_cmds.len());
        for (i, c) in conv_cmds.iter().enumerate() {
            conv_cmd_slice[i] = c.0;
        }

        let add_cmd_bytes = add_cmds.len() * mem::size_of::<u64>();
        let buf_cmd_add = Buffer::new(fd, add_cmd_bytes.next_multiple_of(4096), &file);
        let add_cmd_slice =
            std::slice::from_raw_parts_mut(buf_cmd_add.host_ptr as *mut u64, add_cmds.len());
        for (i, c) in add_cmds.iter().enumerate() {
            add_cmd_slice[i] = c.0;
        }

        fini_bo(fd, buf_in.handle).ok();
        fini_bo(fd, buf_w.handle).ok();
        fini_bo(fd, buf_bias.handle).ok();
        fini_bo(fd, buf_second.handle).ok();
        fini_bo(fd, buf_out.handle).ok();
        fini_bo(fd, buf_cmd_conv.handle).ok();
        fini_bo(fd, buf_cmd_add.handle).ok();

        let in_handles = [
            buf_cmd_conv.handle,
            buf_cmd_add.handle,
            buf_in.handle,
            buf_w.handle,
            buf_bias.handle,
            buf_second.handle,
        ];
        let out_handles = [buf_out.handle];

        submit_tasks(
            fd,
            &[
                (buf_cmd_conv.dma_address, conv_cmds.len() as u32),
                (buf_cmd_add.dma_address, add_cmds.len() as u32),
            ],
            &in_handles,
            &out_handles,
        )
        .expect("multi-task SUBMIT ioctl failed");

        prep_bo(fd, buf_out.handle, 2_000_000_000).unwrap_or_else(|e| {
            panic!(
                "conv-then-add job did not complete within timeout (x_fill={x_fill}, \
                 weight_fill={weight_fill}, w_fill={w_fill}) -- see this file's top doc \
                 comment, this is the first hardware round for this task shape: {e}"
            )
        });

        let raw = std::slice::from_raw_parts(buf_out.host_ptr as *const u16, 128);
        let pixels = raw[..16].iter().map(|&bits| f16_to_f32(bits)).collect();

        close_bo(fd, buf_in.handle).ok();
        close_bo(fd, buf_w.handle).ok();
        close_bo(fd, buf_bias.handle).ok();
        close_bo(fd, buf_second.handle).ok();
        close_bo(fd, buf_mid.handle).ok();
        close_bo(fd, buf_out.handle).ok();
        close_bo(fd, buf_cmd_conv.handle).ok();
        close_bo(fd, buf_cmd_add.handle).ok();

        pixels
    }
}

/// Most basic possible check: does the new multi-task job even complete
/// without hanging the NPU? See this file's top doc comment on what a
/// failure here would/wouldn't tell us.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn conv_then_add_completes() {
    let out = run_conv_then_add(1.0, 1.0, 1.0, 2);
    eprintln!("conv_then_add_completes: output={out:?}");
}

/// Real numeric check: `weight_fill=1.0` makes the 1x1-kernel,
/// single-channel conv stage compute `x * 1 = x` exactly, so the only
/// thing left to produce a non-identity result is the add itself. Holds
/// `w_fill` (the second tensor) fixed at `0.0` and sweeps `x_fill` -- if
/// the add really adds `w_real` (here `0`) to the conv's real output
/// (`x_real`), output should track `x_fill` 1:1 and come out strictly
/// increasing across the sweep.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn conv_then_add_tracks_x_plus_y() {
    let fills = [-6.0f32, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0];
    let mut prev: Option<f32> = None;

    for x_fill in fills {
        let raw = run_conv_then_add(x_fill, 1.0, 0.0, 2)[0];
        eprintln!("conv_then_add_tracks_x_plus_y: x_fill={x_fill}: output={raw}");

        if let Some(prev_raw) = prev {
            assert!(
                raw > prev_raw,
                "conv-then-add output is not strictly increasing over the x_fill sweep: \
                 previous={prev_raw}, current={raw}, x_fill={x_fill}"
            );
        }
        prev = Some(raw);
    }
}

/// Isolation test: `weight_fill=0.0` holds the conv accumulator at a clean
/// real `0` regardless of `x_fill` (held constant here), so the add's own
/// output should track `w_fill` 1:1, independent of the conv stage's own
/// math -- the fp16 analog of `conv_with_add_hw.rs`'s old
/// `conv_with_add_tracks_y_alone`, which passed bit-exact for the
/// (now-retired) single-task int8 shape.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn conv_then_add_tracks_w_alone() {
    let fills = [-6.0f32, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0];
    let mut prev: Option<f32> = None;

    for w_fill in fills {
        let raw = run_conv_then_add(1.0, 0.0, w_fill, 2)[0];
        eprintln!(
            "conv_then_add_tracks_w_alone: w_fill={w_fill} (accumulator held at 0): output={raw}"
        );

        if let Some(prev_raw) = prev {
            assert!(
                raw > prev_raw,
                "conv-then-add output is not strictly increasing over the w_fill sweep \
                 (accumulator held at 0 via weight_fill=0): previous={prev_raw}, current={raw}, \
                 w_fill={w_fill}"
            );
        }
        prev = Some(raw);
    }
}

/// Subtraction, via `algo=4` -- the TRM's real, distinct Minus opcode. The
/// conv+add sweep found fp16 subtraction always uses this opcode directly
/// (unlike int8, which reuses `algo=2` with a negated scale -- see
/// `EwAddShape::algo`'s doc comment), so this is the fp16-correct way to
/// test subtraction, not a scaled Add. Same weight=0 isolation as
/// `conv_then_add_tracks_w_alone`: accumulator held at `0`, so output
/// should track `-w_fill` (`0 - w_real`) as `w_fill` increases --
/// non-increasing across the sweep.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn conv_then_sub_tracks_neg_w() {
    let fills = [-6.0f32, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0];
    let mut prev: Option<f32> = None;

    for w_fill in fills {
        let raw = run_conv_then_add(1.0, 0.0, w_fill, 4)[0];
        eprintln!(
            "conv_then_sub_tracks_neg_w: w_fill={w_fill} (accumulator held at 0): output={raw}"
        );

        if let Some(prev_raw) = prev {
            assert!(
                raw < prev_raw,
                "conv-then-sub output is not strictly decreasing over the w_fill sweep \
                 (accumulator held at 0 via weight_fill=0): previous={prev_raw}, current={raw}, \
                 w_fill={w_fill}"
            );
        }
        prev = Some(raw);
    }
}
