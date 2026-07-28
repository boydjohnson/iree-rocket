//! Hardware-in-the-loop tests for `build_conv_then_pooling_regcmd` -- PPU
//! pipelined directly after a real DPU stage (TRM Ch.36 Fig 36-4/36-5's
//! "do some pipeline surface ops" step), the alternative path tried after
//! `build_pooling_regcmd`'s standalone/flying mode (TRM Fig 36-6) hung real
//! hardware (NPU job timeout + IOMMU "Enable stall request timed out" --
//! see that module's doc comment for the full story).
//!
//! Not run by a plain `cargo test` -- see conv_hw.rs's doc comment for the
//! cross-compile-and-copy-to-the-board workflow; identical here:
//!
//!   CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!     cargo test --target aarch64-unknown-linux-gnu --release \
//!       --test conv_then_pooling_hw --no-run
//!
//! Conv shape is `rkt-basic.rs`'s own validated 4x4x1 -> 4x4x1, 1x1 kernel
//! shape (see rknpu-spelunking/NOTES.md) -- reused here specifically
//! because it's already hardware-proven on its own, isolating this test to
//! whatever's new (the PPU pipelined-pooling stage) rather than also
//! re-litigating the conv path. Same uniform-fill strategy as conv_hw.rs/
//! pooling_hw.rs to sidestep not knowing the real weight/pixel packing
//! order: a uniform-filled conv produces a uniform conv output (already
//! proven by conv_hw.rs), so pooling on top of a uniform output should
//! also be uniform regardless of which pooling method or window landed
//! where.

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    activation::Activation,
    device::{Buffer, fini_bo, prep_bo, submit},
    mesa_conv::ConvShape,
    pooling::{
        ConvThenPoolingBuffers, PipelinedPoolingShape, PoolingMethod, PoolingPrecision,
        build_conv_then_pooling_regcmd,
    },
    regcmd::Precision,
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const TENSOR_SIZE: usize = 4096;

/// rkt-basic.rs's own validated shape.
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
        input_zero_point: 0,
        output_zero_point: 0,
        weights_zero_point: 0,
        input_scale: 1.0,
        weights_scale: 1.0,
        output_scale: 1.0,
        truncate_bits: 0,
        activation: Activation::None,
        precision: Precision::Int8,
    }
}

/// 2x2 kernel/stride, no padding, over the conv's 4x4x1 output -> 2x2x1
/// pooled result.
fn pooling_shape(method: PoolingMethod) -> PipelinedPoolingShape {
    PipelinedPoolingShape {
        kernel_width: 2,
        kernel_height: 2,
        stride_x: 2,
        stride_y: 2,
        method,
        pad_left: 0,
        pad_top: 0,
        pad_right: 0,
        pad_bottom: 0,
        pad_value: 0,
        output_width: 2,
        output_height: 2,
        output_channels: 1,
        precision: PoolingPrecision::Int8,
    }
}

/// Runs `pooling` on top of `conv_shape()` with a uniformly-filled input
/// and returns the 4 real output pixels (16-byte-atomic stride, same
/// convention as conv_hw.rs/pooling_hw.rs's readback).
fn run_uniform(pooling: &PipelinedPoolingShape, input_fill: u8, weight_fill: u8) -> Vec<u8> {
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

        let buf_out = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_out.host_ptr, 0, TENSOR_SIZE);

        let bufs = ConvThenPoolingBuffers {
            input_addr: buf_in.dma_address,
            weights_addr: buf_w.dma_address,
            bias_addr: buf_bias.dma_address,
            output_addr: buf_out.dma_address,
        };
        let cmds = build_conv_then_pooling_regcmd(&conv_shape(), pooling, &bufs);

        let cmd_bytes = cmds.len() * mem::size_of::<u64>();
        let cmd_len = cmd_bytes.next_multiple_of(4096);
        let buf_cmd = Buffer::new(fd, cmd_len, &file);
        let cmd_slice = std::slice::from_raw_parts_mut(buf_cmd.host_ptr as *mut u64, cmds.len());
        for (i, c) in cmds.iter().enumerate() {
            cmd_slice[i] = c.0;
        }

        fini_bo(fd, buf_in.handle).ok();
        fini_bo(fd, buf_w.handle).ok();
        fini_bo(fd, buf_bias.handle).ok();
        fini_bo(fd, buf_cmd.handle).ok();
        // buf_out was CPU-zero-filled above -- needs the same
        // dma_sync_sgtable_for_device() hand-off as every other buffer the
        // device touches, or the CPU's dirty cache line for it can race the
        // device's real write and win, silently clobbering the result back
        // to zero. conv_hw.rs's known-good buf_c handling already does
        // this; this file previously didn't.
        fini_bo(fd, buf_out.handle).ok();

        let in_handles = [buf_cmd.handle, buf_in.handle, buf_w.handle, buf_bias.handle];
        let out_handles = [buf_out.handle];

        submit(
            fd,
            buf_cmd.dma_address,
            cmds.len() as u32,
            &in_handles,
            &out_handles,
        )
        .expect("SUBMIT ioctl failed");

        prep_bo(fd, buf_out.handle, 2_000_000_000)
            .unwrap_or_else(|e| panic!("job did not complete within timeout: {e}"));

        let raw = std::slice::from_raw_parts(buf_out.host_ptr, 64);
        (0..4).map(|i| raw[i * 16]).collect()
    }
}

macro_rules! completes_and_tracks_input_test {
    ($name:ident, $method:expr) => {
        #[test]
        #[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
        fn $name() {
            let pooling = pooling_shape($method);
            for input_fill in [10u8, 118, 200] {
                let pixels = run_uniform(&pooling, input_fill, 2);
                assert!(
                    pixels.iter().all(|&p| p == pixels[0]),
                    "input_fill={input_fill}: expected all 4 pooled output pixels \
                     identical (uniform conv output), got {pixels:?}"
                );
            }
            let low = run_uniform(&pooling, 10, 2)[0];
            let high = run_uniform(&pooling, 200, 2)[0];
            assert_ne!(
                low, high,
                "pooled output pixel value didn't change between input_fill=10 ({low}) \
                 and input_fill=200 ({high}) -- suggests the pipeline isn't really \
                 reading the input"
            );
        }
    };
}

completes_and_tracks_input_test!(
    conv_then_pooling_max_completes_and_tracks_input,
    PoolingMethod::Max
);
completes_and_tracks_input_test!(
    conv_then_pooling_min_completes_and_tracks_input,
    PoolingMethod::Min
);
completes_and_tracks_input_test!(
    conv_then_pooling_avg_completes_and_tracks_input,
    PoolingMethod::Avg
);

/// Diagnostic, not a correctness check -- added after all three tests
/// above came back stuck at constant 0 regardless of input_fill, same
/// symptom class pooling_hw.rs's original standalone-mode failure had.
/// Unlike that case, this run completed in ~0.02s with no hang/IOMMU
/// stall, so whatever's wrong here is a config/addressing bug, not a
/// wedged core. Pre-fills the output buffer with a distinctive non-zero
/// sentinel (0xAA) instead of zero -- same technique that resolved the
/// ambiguity in pooling_hw.rs's `pooling_dump_full_output_buffer` -- so we
/// can tell "PPU wrote real zeros here" apart from "nothing ever wrote to
/// this buffer at all".
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there; \
            diagnostic only, not a pass/fail check -- read the printed hex dump"]
fn conv_then_pooling_dump_full_output_buffer() {
    let pooling = pooling_shape(PoolingMethod::Max);
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let buf_in = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_in.host_ptr, 200u8, TENSOR_SIZE);

        let buf_w = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_w.host_ptr, 2u8, TENSOR_SIZE);

        let buf_bias = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_bias.host_ptr, 0, TENSOR_SIZE);

        let buf_out = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_out.host_ptr, 0xAAu8, TENSOR_SIZE);

        eprintln!(
            "conv_then_pooling_dump_full_output_buffer: buf_out.dma_address = {:#x} \
             ({} decimal) -- raw would need PPU_DST_BASE_ADDR's 28-bit field to hold \
             this directly (fails if >= 0x1000_0000); shifted (>>4) always fits",
            buf_out.dma_address, buf_out.dma_address
        );

        let bufs = ConvThenPoolingBuffers {
            input_addr: buf_in.dma_address,
            weights_addr: buf_w.dma_address,
            bias_addr: buf_bias.dma_address,
            output_addr: buf_out.dma_address,
        };
        let cmds = build_conv_then_pooling_regcmd(&conv_shape(), &pooling, &bufs);

        let cmd_bytes = cmds.len() * mem::size_of::<u64>();
        let cmd_len = cmd_bytes.next_multiple_of(4096);
        let buf_cmd = Buffer::new(fd, cmd_len, &file);
        let cmd_slice = std::slice::from_raw_parts_mut(buf_cmd.host_ptr as *mut u64, cmds.len());
        for (i, c) in cmds.iter().enumerate() {
            cmd_slice[i] = c.0;
        }

        fini_bo(fd, buf_in.handle).ok();
        fini_bo(fd, buf_w.handle).ok();
        fini_bo(fd, buf_bias.handle).ok();
        fini_bo(fd, buf_cmd.handle).ok();
        // buf_out was CPU-sentinel-filled (0xAA) above -- see run_uniform's
        // comment on the identical fini_bo(buf_out) fix above.
        fini_bo(fd, buf_out.handle).ok();

        let in_handles = [buf_cmd.handle, buf_in.handle, buf_w.handle, buf_bias.handle];
        let out_handles = [buf_out.handle];

        submit(
            fd,
            buf_cmd.dma_address,
            cmds.len() as u32,
            &in_handles,
            &out_handles,
        )
        .expect("SUBMIT ioctl failed");

        prep_bo(fd, buf_out.handle, 2_000_000_000).expect("job did not complete within timeout");

        let raw = std::slice::from_raw_parts(buf_out.host_ptr, TENSOR_SIZE);
        eprintln!(
            "conv_then_pooling_dump_full_output_buffer: full {TENSOR_SIZE}-byte output \
             buffer (sentinel-filled 0xAA, input uniformly filled with 200, weight=2):"
        );
        for (row, chunk) in raw.chunks(32).enumerate() {
            let hex: String = chunk.iter().map(|b| format!("{b:02x} ")).collect();
            eprintln!("  {:04x}: {hex}", row * 32);
        }
        let unchanged_count = raw.iter().filter(|&&b| b == 0xAA).count();
        eprintln!(
            "conv_then_pooling_dump_full_output_buffer: {unchanged_count}/{TENSOR_SIZE} bytes \
             still == 0xAA (if this is TENSOR_SIZE, nothing ever wrote to this buffer; if \
             less, something did write here -- check what value landed where)"
        );
    }
}
