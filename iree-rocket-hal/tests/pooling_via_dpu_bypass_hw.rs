//! Hardware-in-the-loop tests for `build_pooling_via_dpu_bypass_regcmd` --
//! the real two-kick shape a vendor-compiled model actually emits (see
//! `pooling.rs`'s module doc comment above that function, and
//! rknpu-spelunking/NOTES.md's "Decoding a real regcmd program for a
//! pooling-only op"), as opposed to `build_pooling_regcmd`'s standalone-only
//! path (`pooling_hw.rs`). A third, on-chip pipelined shape was tried and
//! then retired -- see `pooling.rs`'s module doc comment for the sweep that
//! found no compiler evidence for it.
//!
//! Not run by a plain `cargo test` -- see `conv_hw.rs`'s doc comment for the
//! cross-compile-and-copy-to-the-board workflow; identical here:
//!
//!   CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!     cargo test --target aarch64-unknown-linux-gnu --release \
//!       --test pooling_via_dpu_bypass_hw --no-run
//!
//! Same uniform-whole-buffer-fill methodology as `pooling_hw.rs`/`conv_hw.rs`
//! throughout: the bypass conv stage's exact numeric transform doesn't need
//! to be a true arithmetic identity for these tests to be meaningful --
//! only a real, consistent, input-tracking pipeline (uniform input/weight
//! fill makes per-tap/per-pixel packing order irrelevant either way).
//!
//! First hardware round with these tests found (and this file was rewritten
//! after) that `build_pooling_via_dpu_bypass_regcmd`'s two kicks genuinely
//! need two separate `submit()`/`prep_bo()` round trips -- a single
//! combined-blob submission left the bypass stage's intermediate buffer
//! completely unwritten (sentinel-fill diagnostic showed 128/128 bytes
//! still at the 0xAA prefill). See `build_pooling_via_dpu_bypass_regcmd`'s
//! module doc comment in `pooling.rs` for the full story.

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    activation::Activation,
    conv::{self, Kernels, Multiplier, Quantization},
    device::{Buffer, close_bo, fini_bo, prep_bo, submit},
    pooling::{
        PoolingMethod, PoolingPrecision, PoolingShape, PoolingViaBypassBuffers,
        build_pooling_via_dpu_bypass_regcmd,
    },
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const TENSOR_SIZE: usize = 4096;

/// 1x1 kernel, matching every `bypass_shape()` call in this file.
const BYPASS_KERNELS: Kernels = [1, 1];

/// Bypass stage: 4x4x1 -> 4x4x1, 1x1 kernel, stride 1 -- a real (if
/// numerically arbitrary under uniform fill) CNA->CORE->DPU task sized to
/// match the pooling stage's own 4x4 input, reusing the same safe (>=4)
/// input_height this module's other hardware-validated conv shapes use
/// (see conv.rs's own `input_height/4 - 1`-derived underflow risk,
/// documented in the roadmap plan's Phase 2 notes).
fn bypass_shape() -> conv::Shape {
    conv::Shape {
        width: 4,
        height: 4,
        stride: 1,
        in_channels: 1,
        out_channels: 1,
        precision: conv::Precision::Int8(Quantization {
            input_zero_point: 0,
            output_zero_point: 0,
            multiplier: Multiplier::from_ratio(1.0),
        }),
        padding: Some([0, 0]),
        activation: conv::Activation::None,
        depthwise: false,
    }
}

/// Pooling stage: same 4x4x1 -> 2x2x1, 2x2 kernel, stride 2 geometry as
/// `pooling_hw.rs`'s `tiled_shape` -- its "input" is the bypass stage's
/// real memory output, not an external buffer.
fn pooling_shape(method: PoolingMethod) -> PoolingShape {
    pooling_shape_with_activation(method, Activation::None)
}

/// Same geometry as `pooling_shape`, with an explicit fused activation --
/// Phase 3 of the ukernel roadmap. `build_pooling_via_dpu_bypass_regcmd`
/// applies this to the bypass conv stage's own fused-activation stage (see
/// its own doc comment); `pooling_shape`'s `activation: Activation::None`
/// above is just the common case of this with `None`.
fn pooling_shape_with_activation(method: PoolingMethod, activation: Activation) -> PoolingShape {
    PoolingShape {
        input_width: 4,
        input_height: 4,
        input_channels: 1,
        output_width: 2,
        output_height: 2,
        output_channels: 1,
        precision: PoolingPrecision::Int8,
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
        activation,
    }
}

struct Bufs {
    file: std::fs::File,
    fd: i32,
    buf_in: Buffer,
    buf_w: Buffer,
    buf_bias: Buffer,
    buf_mid: Buffer,
    buf_out: Buffer,
}

/// Allocates and fills every buffer this shape needs, but does NOT submit
/// anything -- shared setup for both the real test (two submits) and the
/// diagnostic (which also sentinel-fills buf_mid/buf_out differently).
fn setup(input_fill: u8, weight_fill: u8, out_fill: u8) -> Bufs {
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

        let buf_mid = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_mid.host_ptr, out_fill, TENSOR_SIZE);

        let buf_out = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_out.host_ptr, out_fill, TENSOR_SIZE);

        fini_bo(fd, buf_in.handle).ok();
        fini_bo(fd, buf_w.handle).ok();
        fini_bo(fd, buf_bias.handle).ok();
        fini_bo(fd, buf_mid.handle).ok();
        fini_bo(fd, buf_out.handle).ok();

        Bufs {
            file,
            fd,
            buf_in,
            buf_w,
            buf_bias,
            buf_mid,
            buf_out,
        }
    }
}

impl Bufs {
    /// Releases every GEM handle this struct owns. `Buffer` has no `Drop`
    /// impl (nothing in `iree-rocket-hal` did before this session's
    /// `close_bo` addition -- see `device.rs`'s doc comment on it), so
    /// without an explicit call like this, every one of these 5 buffers
    /// leaks for the rest of the process's lifetime. Added after repeated
    /// calls to `run_uniform_pooling_via_bypass` within one test (up to 6x
    /// in `pooling_via_bypass_repeat_dispatch_dump`, each leaking 7 GEM
    /// handles including `run_two_stage`'s own regcmd buffers) were found
    /// to produce corrupted/stale results, while the identical dispatch
    /// run exactly once in a fresh process was clean.
    unsafe fn close(self) {
        unsafe {
            close_bo(self.fd, self.buf_in.handle).ok();
            close_bo(self.fd, self.buf_w.handle).ok();
            close_bo(self.fd, self.buf_bias.handle).ok();
            close_bo(self.fd, self.buf_mid.handle).ok();
            close_bo(self.fd, self.buf_out.handle).ok();
        }
    }
}

/// Runs the two-kick bypass-then-pool shape as two genuinely separate
/// `submit()`/`prep_bo()` round trips (see this file's top doc comment for
/// why) and returns the real output pixels (same 16-byte-atomic read
/// stride as `conv_hw.rs`/`pooling_hw.rs`).
fn run_uniform_pooling_via_bypass(
    method: PoolingMethod,
    input_fill: u8,
    weight_fill: u8,
) -> Vec<u8> {
    run_uniform_pooling_via_bypass_with_activation(
        method,
        input_fill,
        weight_fill,
        Activation::None,
    )
}

/// Same as `run_uniform_pooling_via_bypass`, with an explicit fused
/// activation on the pooling op (Phase 3, applied to the bypass conv
/// stage's own fused-activation stage by `build_pooling_via_dpu_bypass_regcmd`)
/// instead of always `None`.
fn run_uniform_pooling_via_bypass_with_activation(
    method: PoolingMethod,
    input_fill: u8,
    weight_fill: u8,
    activation: Activation,
) -> Vec<u8> {
    let b = setup(input_fill, weight_fill, 0);
    unsafe {
        run_two_stage(
            &b,
            &bypass_shape(),
            &pooling_shape_with_activation(method, activation),
        )
    };

    let raw = unsafe { std::slice::from_raw_parts(b.buf_out.host_ptr, 256) };
    let pixels = (0..4).map(|i| raw[i * 16]).collect();
    unsafe { b.close() };
    pixels
}

unsafe fn run_two_stage(b: &Bufs, bypass: &conv::Shape, pooling: &PoolingShape) {
    unsafe {
        let bufs = PoolingViaBypassBuffers {
            input_addr: b.buf_in.dma_address,
            weights_addr: b.buf_w.dma_address,
            bias_addr: b.buf_bias.dma_address,
            bypass_output_addr: b.buf_mid.dma_address,
            output_addr: b.buf_out.dma_address,
        };
        let (bypass_cmds, pooling_cmds) =
            build_pooling_via_dpu_bypass_regcmd(bypass, BYPASS_KERNELS, pooling, &bufs);

        // Stage 1: submit and fully wait for the bypass conv to finish
        // writing buf_mid before touching stage 2 at all.
        let cmd_bytes_1 = bypass_cmds.len() * mem::size_of::<u64>();
        let cmd_len_1 = cmd_bytes_1.next_multiple_of(4096);
        let buf_cmd_1 = Buffer::new(b.fd, cmd_len_1, &b.file);
        let cmd_slice_1 =
            std::slice::from_raw_parts_mut(buf_cmd_1.host_ptr as *mut u64, bypass_cmds.len());
        for (i, c) in bypass_cmds.iter().enumerate() {
            cmd_slice_1[i] = c.0;
        }
        fini_bo(b.fd, buf_cmd_1.handle).ok();

        let in_handles_1 = [
            buf_cmd_1.handle,
            b.buf_in.handle,
            b.buf_w.handle,
            b.buf_bias.handle,
        ];
        let out_handles_1 = [b.buf_mid.handle];
        submit(
            b.fd,
            buf_cmd_1.dma_address,
            bypass_cmds.len() as u32,
            &in_handles_1,
            &out_handles_1,
        )
        .expect("stage 1 SUBMIT ioctl failed");
        prep_bo(b.fd, b.buf_mid.handle, 2_000_000_000)
            .expect("stage 1 (bypass conv) did not complete within timeout");
        // Purely local to this stage, never needed again -- close it
        // immediately rather than leaking it for the rest of the process.
        close_bo(b.fd, buf_cmd_1.handle).ok();

        // Stage 2: submit only after stage 1 is fully done.
        let cmd_bytes_2 = pooling_cmds.len() * mem::size_of::<u64>();
        let cmd_len_2 = cmd_bytes_2.next_multiple_of(4096);
        let buf_cmd_2 = Buffer::new(b.fd, cmd_len_2, &b.file);
        let cmd_slice_2 =
            std::slice::from_raw_parts_mut(buf_cmd_2.host_ptr as *mut u64, pooling_cmds.len());
        for (i, c) in pooling_cmds.iter().enumerate() {
            cmd_slice_2[i] = c.0;
        }
        fini_bo(b.fd, buf_cmd_2.handle).ok();
        // buf_out was CPU-filled during setup(); needs the same dma_sync
        // hand-off as pooling_hw.rs's run_pooling (fini_bo'd there already,
        // during setup() -- re-affirmed here as a reminder this ordering
        // matters, not a second real action).

        let in_handles_2 = [buf_cmd_2.handle, b.buf_mid.handle];
        let out_handles_2 = [b.buf_out.handle];
        submit(
            b.fd,
            buf_cmd_2.dma_address,
            pooling_cmds.len() as u32,
            &in_handles_2,
            &out_handles_2,
        )
        .expect("stage 2 SUBMIT ioctl failed");
        prep_bo(b.fd, b.buf_out.handle, 2_000_000_000)
            .expect("stage 2 (pooling) did not complete within timeout");
        close_bo(b.fd, buf_cmd_2.handle).ok();
    }
}

macro_rules! completes_and_tracks_input_test {
    ($name:ident, $method:expr) => {
        #[test]
        #[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
        fn $name() {
            for input_fill in [10u8, 118, 200] {
                let pixels = run_uniform_pooling_via_bypass($method, input_fill, 2);
                assert!(
                    pixels.iter().all(|&p| p == pixels[0]),
                    "input_fill={input_fill}: expected all 4 output pixels identical \
                     (uniform input), got {pixels:?}"
                );
            }
            let low = run_uniform_pooling_via_bypass($method, 10, 2)[0];
            let high = run_uniform_pooling_via_bypass($method, 200, 2)[0];
            assert_ne!(
                low, high,
                "output pixel value didn't change between input_fill=10 ({low}) and \
                 input_fill=200 ({high}) -- suggests the op isn't really reading the input"
            );
        }
    };
}

completes_and_tracks_input_test!(
    pooling_via_bypass_max_completes_and_output_tracks_input,
    PoolingMethod::Max
);
completes_and_tracks_input_test!(
    pooling_via_bypass_min_completes_and_output_tracks_input,
    PoolingMethod::Min
);
completes_and_tracks_input_test!(
    pooling_via_bypass_avg_completes_and_output_tracks_input,
    PoolingMethod::Avg
);

/// Diagnostic, not a correctness check. Second round update: with the
/// DPU_RDMA kick fix, stage 1 now genuinely writes buf_mid (no longer
/// stuck at the 0xAA sentinel) -- but `pooling_via_bypass_min_...` then
/// found a *new* bug: pooling a uniform-input result produced
/// non-identical output pixels ([255,255,0,0]), the first real exercise
/// of PPU_RDMA's fetch-side stride math against a genuinely DPU-written
/// (not CPU-written) buffer. Dumps a wider region (512 bytes, covering the
/// full 4x4x1 image at 16 bytes/pixel plus the padded second
/// task_output_channels atomic group) of both buffers using `Min` (the
/// method that showed the clearest non-uniform symptom) to see whether
/// DPU's write-side layout is genuinely uniform across all 16 pixel
/// positions (bug is in PPU_RDMA's read-side stride formulas) or not
/// (bug is upstream, in the bypass conv's own write).
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there; \
            diagnostic only, not a pass/fail check -- read the printed hex dumps"]
fn pooling_via_bypass_dump_intermediate_and_output_buffers() {
    let b = setup(200, 2, 0xAA);
    unsafe { run_two_stage(&b, &bypass_shape(), &pooling_shape(PoolingMethod::Min)) };

    unsafe {
        prep_bo(b.fd, b.buf_mid.handle, 2_000_000_000).ok();
        for (label, buf) in [
            ("buf_mid (stage 1 output)", &b.buf_mid),
            ("buf_out (stage 2 output)", &b.buf_out),
        ] {
            let raw = std::slice::from_raw_parts(buf.host_ptr, 512);
            eprintln!("{label}, first 512 bytes:");
            for (row, chunk) in raw.chunks(32).enumerate() {
                let hex: String = chunk.iter().map(|b| format!("{b:02x} ")).collect();
                eprintln!("  {:04x}: {hex}", row * 32);
            }
            let unchanged = raw.iter().filter(|&&b| b == 0xAA).count();
            eprintln!("{label}: {unchanged}/512 bytes still == 0xAA");
        }
    }
    unsafe { b.close() };
}

/// Diagnostic, not a correctness check. Added after the *same* nominal
/// dispatch (Min pooling, input_fill=200, weight_fill=2) produced two
/// different results across separate test-process invocations even under
/// `--test-threads=1` (serial execution, ruling out cross-thread races):
/// [255,255,0,0] from `pooling_via_bypass_min_completes_and_output_tracks_input`
/// vs. a uniform 250 from `pooling_via_bypass_dump_intermediate_and_output_buffers`,
/// with `buf_mid` confirmed byte-identical (uniform, correct) in both
/// cases. Since the input to stage 2 is proven identical, whatever's
/// non-deterministic must be internal to PPU/PPU_RDMA's own hardware state
/// -- the leading hypothesis is the ping-pong buffering bits
/// (`PpuSPointer`/`PpuRdmaSPointer`'s `pointer_pp_mode`/`pp_en`, set
/// identically on every dispatch here) not being correctly synchronized
/// with the engine's actual internal toggle state, which may persist
/// across dispatches regardless of which process/fd issued them.
///
/// Repeats the identical dispatch (fresh buffers each time, same shape,
/// same fills) several times in a row within one process and prints each
/// run's first 4 output bytes, to see whether results alternate between a
/// small fixed set of states (supports the ping-pong theory) or vary
/// unpredictably (points to something else, e.g. a genuine race against
/// the job's actual hardware completion rather than just PREP_BO
/// returning).
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there; \
            diagnostic only, not a pass/fail check -- read the printed sequence"]
fn pooling_via_bypass_repeat_dispatch_dump() {
    for i in 0..6 {
        let pixels = run_uniform_pooling_via_bypass(PoolingMethod::Min, 200, 2);
        eprintln!("repeat #{i}: output pixels = {pixels:?}");
    }
}

/// Phase 3 of the ukernel roadmap: fused activation on a pooling op that
/// has a real DPU stage ahead of PPU (this bypass path). Same
/// domain-independent proof `conv_activation_hw.rs`'s
/// `conv_3x3_relux_cmp_zero_forces_constant_output` used for conv: `cmp: 0`
/// clamps to 0 in any numeric scale, so if the bypass stage is really
/// applying `pooling_shape.activation` (not silently dropping it),
/// every input_fill should read back as the same constant regardless of
/// what the un-clamped value would have been -- proves the fusion wiring
/// without needing to know this suite's real (placeholder scale=1.0)
/// numeric domain, same caveat as conv's own `cmp` open question.
///
/// The bypass stage's conv now goes through `conv.rs`'s capture-derived
/// builder rather than `mesa_conv`'s, which fuses activation via the BN
/// stage instead of BS (see `pooling.rs`'s own doc comment on
/// `build_pooling_via_dpu_bypass_regcmd`) -- `cmp: 0` clamping to a
/// constant zero is domain- and stage-invariant, so this specific test
/// still proves the fusion wiring works, but a real hardware run after
/// this migration is the first one to exercise BN-stage fusion in this
/// exact bypass-then-pool composition.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn pooling_via_bypass_relux_cmp_zero_forces_constant_output() {
    let mut outputs = Vec::new();
    for input_fill in [0u8, 50, 100, 150, 200, 255] {
        let pixels = run_uniform_pooling_via_bypass_with_activation(
            PoolingMethod::Max,
            input_fill,
            2,
            Activation::Relux { cmp: 0 },
        );
        assert!(
            pixels.iter().all(|&p| p == pixels[0]),
            "input_fill={input_fill}: expected all 4 output pixels identical \
             (uniform input), got {pixels:?}"
        );
        outputs.push(pixels[0]);
    }
    assert!(
        outputs.iter().all(|&o| o == outputs[0]),
        "Relux{{cmp: 0}} should force the same constant output regardless of input_fill \
         in any numeric domain, but got {outputs:?} across fills [0,50,100,150,200,255] -- \
         fused activation on the bypass stage isn't being applied, or \
         pooling_shape.activation isn't reaching it"
    );
}
