//! Hardware-in-the-loop tests for `build_conv_then_lut_regcmd` -- the real
//! conv->sigmoid/tanh pipeline, as two hardware tasks in one
//! `device::submit_tasks()` job (see that function's doc comment and
//! `regcmd.rs`'s `ConvThenLutBuffers` doc comment for why this is a
//! genuinely separate task rather than fused into the conv's own DPU pass,
//! unlike `Relu`/`Relux` -- confirmed via live bpftrace tracing of the
//! vendor runtime itself, `rknpu-spelunking/NOTES.md`).
//!
//! Not run by a plain `cargo test` -- see `conv_hw.rs`'s doc comment for the
//! cross-compile-and-copy-to-the-board workflow; identical here:
//!
//!   CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!     cargo test --target aarch64-unknown-linux-gnu --release \
//!       --test conv_then_lut_hw --no-run
//!
//! This is the first real hardware exercise of TWO previously-separate
//! things at once: `device::submit_tasks()`'s multi-task job (never run on
//! real hardware before -- see its own doc comment) and the conv output
//! buffer's byte layout actually lining up with what `build_lut_regcmd`
//! expects to read as its own input (never chained before either, unlike
//! `build_pooling_via_dpu_bypass_regcmd`'s conv->pooling chain, which *has*
//! been hardware-proven). Both are genuinely new -- if these tests hang or
//! misbehave, check `submit_tasks` in isolation (e.g. resubmit an already-
//! proven single task twice via one multi-task job) before assuming the
//! LUT recipe itself is wrong.
//!
//! Conv shape follows `lut_hw.rs`'s own hard-won lesson (see its doc
//! comment): a 1x1 kernel and *real* (non-offset) zero points (`0x80`
//! decodes to a real zero point of `0`) avoid the accumulator-saturation
//! problem a 3x3 kernel + offset zero points produced on the first LUT
//! hardware rounds. `output_zero_point: 0x80` on the conv side matches
//! `input_zero_point: 0x80` on the LUT side -- consistent "real zero = 0"
//! interpretation of the same raw bytes across the task boundary.

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    device::{Buffer, close_bo, fini_bo, prep_bo, submit_tasks},
    regcmd::{
        Activation, ConvShape, ConvThenLutBuffers, LutShape, LutTable, build_conv_then_lut_regcmd,
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

/// `channels: 1` (not `lut_hw.rs`'s standalone `16`) to match `conv_shape`'s
/// real `output_channels: 1` byte-for-byte -- both sides pad to their own
/// engine's atomic granularity internally (DPU's WDMA vs. DPU_RDMA/MRDMA's
/// own `FEATURE_ATOMIC_SIZE`), but the *real*, unpadded channel count they
/// agree on must match for the LUT stage to read the conv stage's actual
/// output layout correctly -- unverified until the first hardware round
/// (see this file's top doc comment).
fn lut_shape(output_zero_point: u32) -> LutShape {
    LutShape {
        width: 4,
        height: 4,
        channels: 1,
        input_zero_point: 0x80,
        output_zero_point,
        input_scale: 1.0 / 32.0,
        output_scale: 1.0 / 256.0,
    }
}

/// Builds the two-task conv->LUT program, submits it as one multi-task
/// job, waits once on the final output, and returns the 16 real pixels
/// (16-byte-atomic stride, same convention as every other hw test in this
/// crate).
fn run_uniform_conv_then_lut(
    table: LutTable,
    output_zero_point: u32,
    input_fill: u8,
    weight_fill: u8,
) -> Vec<u8> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let buf_in = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_in.host_ptr, input_fill, TENSOR_SIZE);

        let buf_w = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_w.host_ptr, weight_fill, TENSOR_SIZE);

        let buf_bias = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_bias.host_ptr, 0, TENSOR_SIZE);

        // Pure inter-task DMA memory -- never read/written by the CPU, so
        // no ptr::write_bytes/fini_bo/prep_bo on it, and it doesn't need to
        // appear in either handle list below (see submit_tasks's doc
        // comment: GEM buffers are IOMMU-mapped at CREATE_BO time,
        // regardless of a job's handle lists -- those lists are for
        // cross-job scheduler fencing only, which this buffer doesn't need
        // since nothing outside this one job ever touches it).
        let buf_mid = Buffer::new(fd, TENSOR_SIZE, &file);

        let buf_out = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_out.host_ptr, 0, TENSOR_SIZE);

        let bufs = ConvThenLutBuffers {
            input_addr: buf_in.dma_address,
            weights_addr: buf_w.dma_address,
            bias_addr: buf_bias.dma_address,
            intermediate_addr: buf_mid.dma_address,
            output_addr: buf_out.dma_address,
        };
        let (conv_cmds, lut_cmds) =
            build_conv_then_lut_regcmd(&conv_shape(), &lut_shape(output_zero_point), table, &bufs);

        let conv_cmd_bytes = conv_cmds.len() * mem::size_of::<u64>();
        let buf_cmd_conv = Buffer::new(fd, conv_cmd_bytes.next_multiple_of(4096), &file);
        let conv_cmd_slice =
            std::slice::from_raw_parts_mut(buf_cmd_conv.host_ptr as *mut u64, conv_cmds.len());
        for (i, c) in conv_cmds.iter().enumerate() {
            conv_cmd_slice[i] = c.0;
        }

        let lut_cmd_bytes = lut_cmds.len() * mem::size_of::<u64>();
        let buf_cmd_lut = Buffer::new(fd, lut_cmd_bytes.next_multiple_of(4096), &file);
        let lut_cmd_slice =
            std::slice::from_raw_parts_mut(buf_cmd_lut.host_ptr as *mut u64, lut_cmds.len());
        for (i, c) in lut_cmds.iter().enumerate() {
            lut_cmd_slice[i] = c.0;
        }

        fini_bo(fd, buf_in.handle).ok();
        fini_bo(fd, buf_w.handle).ok();
        fini_bo(fd, buf_bias.handle).ok();
        fini_bo(fd, buf_out.handle).ok();
        fini_bo(fd, buf_cmd_conv.handle).ok();
        fini_bo(fd, buf_cmd_lut.handle).ok();

        let in_handles = [
            buf_cmd_conv.handle,
            buf_cmd_lut.handle,
            buf_in.handle,
            buf_w.handle,
            buf_bias.handle,
        ];
        let out_handles = [buf_out.handle];

        submit_tasks(
            fd,
            &[
                (buf_cmd_conv.dma_address, conv_cmds.len() as u32),
                (buf_cmd_lut.dma_address, lut_cmds.len() as u32),
            ],
            &in_handles,
            &out_handles,
        )
        .expect("multi-task SUBMIT ioctl failed");

        prep_bo(fd, buf_out.handle, 2_000_000_000).unwrap_or_else(|e| {
            panic!(
                "conv-then-lut job did not complete within timeout (input_fill={input_fill}, \
                 weight_fill={weight_fill}) -- either submit_tasks's multi-task job mechanism \
                 or the conv->LUT buffer handoff may be at fault, see this file's top doc \
                 comment: {e}"
            )
        });

        let raw = std::slice::from_raw_parts(buf_out.host_ptr, 256);
        let pixels = raw[..16].to_vec();

        close_bo(fd, buf_in.handle).ok();
        close_bo(fd, buf_w.handle).ok();
        close_bo(fd, buf_bias.handle).ok();
        close_bo(fd, buf_mid.handle).ok();
        close_bo(fd, buf_out.handle).ok();
        close_bo(fd, buf_cmd_conv.handle).ok();
        close_bo(fd, buf_cmd_lut.handle).ok();

        pixels
    }
}

/// Most basic possible check: does the new multi-task job (never run on
/// real hardware before this test) even complete without hanging the NPU?
/// See this file's top doc comment on isolating this mechanism from the
/// LUT recipe itself if it doesn't.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn conv_then_sigmoid_completes() {
    let out = run_uniform_conv_then_lut(LutTable::sigmoid(), 0, 0, 64);
    eprintln!("conv_then_sigmoid_completes: output={out:?}");
}

#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn conv_then_tanh_completes() {
    let out = run_uniform_conv_then_lut(LutTable::tanh(), 0x80, 0, 64);
    eprintln!("conv_then_tanh_completes: output={out:?}");
}

/// `LutTable::exp()` (softmax Phase 5's exp(x) step, see its own doc
/// comment) through this exact same conv->LUT machinery, unmodified --
/// the recipe is byte-identical in structure to sigmoid/tanh, just a
/// different table + `ew_op_bypass`. This only exercises the isolated
/// exp(x) LUT dispatch completing and producing an ordered curve, same
/// as `conv_then_sigmoid_completes`/`conv_then_tanh_completes` above --
/// it does NOT validate a real softmax data flow (max-subtraction into
/// this stage and the reciprocal/normalize step are still unconfirmed,
/// see `build_max_reduction_tree_regcmd`'s module doc comment in
/// `regcmd.rs`). `output_zero_point: 0` (not `0x80`) because exp(x) is
/// never negative, matching `conv_then_sigmoid_completes`'s own choice
/// for the same reason.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn conv_then_exp_completes() {
    let out = run_uniform_conv_then_lut(LutTable::exp(), 0, 0, 64);
    eprintln!("conv_then_exp_completes: output={out:?}");
}

/// Validates the actual intended use case end-to-end (not just the
/// isolated LUT stage `lut_hw.rs` already covers): conv's real output,
/// fed through the LUT stage as a genuinely separate task, still produces
/// an ordered sigmoid-shaped curve as the conv's input fill varies.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn conv_then_sigmoid_is_monotonic() {
    // Raw bytes labeled by their two's-complement int8 value (-6, -5, ...,
    // 5, 6) purely for readability -- same sweep as lut_hw.rs's
    // LUT_SWEEP_FILLS. The output byte itself must be read as UNSIGNED,
    // not two's-complement: this LutShape uses output_zero_point=0, so the
    // code is a direct round(sigmoid(x)/output_scale) in [0, 256) clamped
    // to a u8 -- sigmoid's range [0,1] is never negative, unlike tanh's
    // signed-around-0x80 convention that a two's-complement decode would
    // suit. Confirmed on real hardware (2026-07-22): the full sweep is a
    // step from raw=123 (all six negative fills) to raw=133 (all seven
    // non-negative fills) -- monotonic non-decreasing as unsigned bytes,
    // and only appears to violate monotonicity if mis-decoded as i8.
    let fills = [250u8, 251, 252, 253, 254, 255, 0, 1, 2, 3, 4, 5, 6];
    let mut prev = None;

    for input_fill in fills {
        let raw = run_uniform_conv_then_lut(LutTable::sigmoid(), 0, input_fill, 64)[0];
        eprintln!(
            "conv_then_sigmoid input_fill={input_fill} (int8={}): raw={raw}",
            input_fill as i8
        );

        if let Some(prev_raw) = prev {
            assert!(
                raw >= prev_raw,
                "conv-then-sigmoid output is not monotonic over the sweep: \
                 previous_raw={prev_raw}, current_raw={raw}, input_fill={input_fill} \
                 (int8={})",
                input_fill as i8
            );
        }
        prev = Some(raw);
    }
}
