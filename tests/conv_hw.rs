//! Hardware-in-the-loop tests for `build_conv_regcmd`'s 3x3-kernel path,
//! both single- and multi-input-channel.
//!
//! Not run by a plain `cargo test` -- this crate targets the board's NPU
//! (`/dev/accel/accel0`), which doesn't exist on the host doing the
//! building. Cross-compile the test binary and copy it over instead:
//!
//!   CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!     cargo test --target aarch64-unknown-linux-gnu --release \
//!       --test conv_hw --no-run
//!
//! then copy the resulting binary (path printed by `--no-run`, under
//! `target/aarch64-unknown-linux-gnu/release/deps/conv_hw-*`) to the
//! board and run it there as `./conv_hw-<hash> --ignored`.
//!
//! Same shape family and verification strategy as `rkt-shape-b.rs` (see
//! that file's doc comment and rknpu-spelunking/NOTES.md for the full
//! rationale) -- uniform input/weight fill so every tap, and now every
//! input channel too, contributes identically regardless of the
//! still-unknown per-tap/per-channel weight-buffer packing order,
//! sweeping the fill byte near the CVT stage's 128 zero-point. Turned
//! into real `assert!`s instead of eyeballed prints.
//!
//! `input_channels > 1` exercises a genuinely different path through
//! `build_conv_regcmd` than every prior hardware test in this repo
//! (rkt-basic.rs/rkt-shape-a*.rs/rkt-shape-b.rs all used
//! `input_channels: 1`): the `input_channels_real_is_one` branches in
//! CNA_CONV_CON1/CNA_CVT_CON0/CNA_CVT_CON5, and the multi-channel
//! `input_data_entries`/CBUF-geometry formulas, were never hit by real
//! hardware before this.
//!
//! `conv_fp16_*` below exercises `ConvShape::precision = Precision::Fp16`
//! -- `build_conv_regcmd`'s native fp16 path, ported from `rkt-fp16.rs`'s
//! own 7-round hardware derivation (see that file's module doc comment
//! and rknpu-spelunking/NOTES.md's fp16 section). Reuses `rkt-fp16.rs`'s
//! exact proven geometry (4x4x1 -> 4x4x1, 1x1 kernel) rather than this
//! file's 3x3/6x6 "shape B" geometry -- `Precision::Fp16` is only
//! asserted-safe for `input_channels==1` in `build_conv_regcmd`, and only
//! this one geometry has ever actually been run on real hardware.

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    device::{Buffer, fini_bo, prep_bo, submit},
    regcmd::{
        Activation, CONV_OUTPUT_ATOMIC_STRIDE, ConvBuffers, ConvShape, Precision,
        build_conv_regcmd, build_conv_regcmd_tasks, conv_output_scratch_bytes, plan_conv_tasks,
    },
    tensor_layout::{nc1hwc2_storage_size, pack_nhwc_to_nc1hwc2},
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const TENSOR_SIZE: usize = 4096;

/// Shape B's own config: 6x6x1 -> 4x4x1, 3x3 kernel, stride 1.
fn single_channel_shape() -> ConvShape {
    ConvShape {
        input_width: 6,
        input_height: 6,
        input_channels: 1,
        output_width: 4,
        output_height: 4,
        output_channels: 1,
        weights_width: 3,
        weights_height: 3,
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

/// Identical geometry to `single_channel_shape()`, except a real 3-channel
/// input (e.g. RGB) instead of 1 -- takes `build_conv_regcmd`'s
/// multi-channel branches (`input_channels_real_is_one == false`) instead
/// of the single-channel ones every other hardware test in this repo has
/// exercised so far.
fn multi_channel_shape() -> ConvShape {
    ConvShape {
        input_channels: 3,
        ..single_channel_shape()
    }
}

/// Runs `shape` with the whole input plane filled with `input_fill` and
/// the whole weight buffer filled with `weight_fill`, and returns the 16
/// real output pixels.
///
/// Read at stride 16 bytes/pixel, not stride 1 -- output_channels=1 pads
/// to task_output_channels=32, and the hardware writes each pixel as a
/// full 16-byte-aligned atomic slot regardless of real channel count
/// (see rknpu-spelunking/NOTES.md's "RESOLVED: the only pixel [0,0]
/// written mystery" section).
fn run_uniform_conv(shape: &ConvShape, input_fill: u8, weight_fill: u8) -> Vec<u8> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let buf_a = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_a.host_ptr, input_fill, TENSOR_SIZE);

        let buf_w = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_w.host_ptr, weight_fill, TENSOR_SIZE);

        let buf_bias = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_bias.host_ptr, 0, TENSOR_SIZE);

        let buf_c = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_c.host_ptr, 0, TENSOR_SIZE);

        let bufs = ConvBuffers {
            input_addr: buf_a.dma_address,
            weights_addr: buf_w.dma_address,
            bias_addr: buf_bias.dma_address,
            output_addr: buf_c.dma_address,
        };
        let cmds = build_conv_regcmd(shape, &bufs);

        let cmd_bytes = cmds.len() * mem::size_of::<u64>();
        let cmd_len = cmd_bytes.next_multiple_of(4096);
        let buf_cmd = Buffer::new(fd, cmd_len, &file);
        let cmd_slice = std::slice::from_raw_parts_mut(buf_cmd.host_ptr as *mut u64, cmds.len());
        for (i, c) in cmds.iter().enumerate() {
            cmd_slice[i] = c.0;
        }

        fini_bo(fd, buf_a.handle).ok();
        fini_bo(fd, buf_w.handle).ok();
        fini_bo(fd, buf_bias.handle).ok();
        fini_bo(fd, buf_c.handle).ok();
        fini_bo(fd, buf_cmd.handle).ok();

        let in_handles = [buf_cmd.handle, buf_a.handle, buf_w.handle, buf_bias.handle];
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
                "job did not complete within timeout (input_channels={}, input_fill={input_fill}): {e}",
                shape.input_channels
            )
        });

        let raw = std::slice::from_raw_parts(buf_c.host_ptr, 256);
        (0..16).map(|i| raw[i * 16]).collect()
    }
}

/// Shape B's own uniform-fill trick sidesteps not knowing the real
/// per-tap weight-buffer packing order: every one of the 9 taps
/// contributes identically, so every one of the 16 output pixels should
/// come out identical regardless of tap ordering.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn conv_3x3_uniform_fill_pixels_agree() {
    let shape = single_channel_shape();
    for input_fill in [10u8, 118, 200] {
        let pixels = run_uniform_conv(&shape, input_fill, 2);
        assert!(
            pixels.iter().all(|&p| p == pixels[0]),
            "input_fill={input_fill}: expected all 16 output pixels identical \
             (uniform input/weights), got {pixels:?}"
        );
    }
}

/// A real multi-tap accumulation should actually respond to the input --
/// guards against a hollow "completes but never touches the data" pass
/// (an all-zero or stuck-at-some-constant output would satisfy the
/// "all pixels agree" check above too).
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn conv_3x3_output_tracks_input() {
    let shape = single_channel_shape();
    let low = run_uniform_conv(&shape, 10, 2)[0];
    let high = run_uniform_conv(&shape, 200, 2)[0];
    assert_ne!(
        low, high,
        "output pixel value didn't change between input_fill=10 ({low}) and \
         input_fill=200 ({high}) -- suggests the op isn't really reading the input"
    );
}

/// Same "all 16 output pixels agree" check as
/// `conv_3x3_uniform_fill_pixels_agree`, but with a genuine 3-channel
/// input instead of 1 -- the uniform whole-buffer fill makes per-channel
/// packing order just as irrelevant as per-tap order was, so this proves
/// real multi-channel accumulation across `build_conv_regcmd`'s
/// previously-untested `input_channels_real_is_one == false` branches.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn conv_3x3_multi_channel_uniform_fill_pixels_agree() {
    let shape = multi_channel_shape();
    for input_fill in [10u8, 118, 200] {
        let pixels = run_uniform_conv(&shape, input_fill, 2);
        assert!(
            pixels.iter().all(|&p| p == pixels[0]),
            "input_channels=3, input_fill={input_fill}: expected all 16 output \
             pixels identical (uniform input/weights), got {pixels:?}"
        );
    }
}

/// Multi-channel counterpart to `conv_3x3_output_tracks_input`.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn conv_3x3_multi_channel_output_tracks_input() {
    let shape = multi_channel_shape();
    let low = run_uniform_conv(&shape, 10, 2)[0];
    let high = run_uniform_conv(&shape, 200, 2)[0];
    assert_ne!(
        low, high,
        "input_channels=3: output pixel value didn't change between \
         input_fill=10 ({low}) and input_fill=200 ({high}) -- suggests the op \
         isn't really reading the input"
    );
}

//===========================================================================
// fp16 (`Precision::Fp16`) -- see this file's module doc comment.
//===========================================================================

/// `rkt-fp16.rs`'s exact proven geometry: 4x4x1 -> 4x4x1, 1x1 kernel,
/// stride 1 -- NOT shape B's 3x3/6x6 geometry above, which has never been
/// run with `Precision::Fp16` on real hardware.
fn fp16_shape() -> ConvShape {
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
        precision: Precision::Fp16,
    }
}

/// Minimal IEEE-754 binary16 -> f32 decode, duplicated from `rkt-fp16.rs`
/// rather than shared (see `conv_activation_hw.rs`'s `run_uniform_conv`
/// doc comment on why integration tests in `tests/` each carry their own
/// copy). Handles subnormals (round 6's intermediate hardware result,
/// before round 7's fix, was a real subnormal -- flushing those to zero
/// would hide a genuine non-NaN result) and inf/nan (`exp==0x1f`) as well
/// as normals.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 0x1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    let f32_bits = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            let subnormal = (frac as f32) * 2f32.powi(-24);
            return if sign == 1 { -subnormal } else { subnormal };
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (frac << 13)
    } else {
        let new_exp = exp + (127 - 15);
        (sign << 31) | (new_exp << 23) | (frac << 13)
    };
    f32::from_bits(f32_bits)
}

fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = (bits >> 31) & 0x1;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x7f_ffff;
    let new_exp = exp - 127 + 15;
    assert!((1..31).contains(&new_exp), "value out of easy f16 range");
    ((sign << 15) | ((new_exp as u32) << 10) | (frac >> 13)) as u16
}

unsafe fn fill_u16(ptr: *mut u8, byte_len: usize, val: u16) {
    let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u16, byte_len / 2) };
    slice.fill(val);
}

/// fp16 counterpart to `run_uniform_conv` -- same uniform-whole-buffer-fill
/// strategy, u16-element fills instead of u8, and pixels decoded from f16
/// bits instead of read raw. Same 16-byte-atomic-per-pixel stride as
/// `run_uniform_conv` (the two low bytes of each 16-byte slot hold the
/// real f16 value; see this file's module doc comment on
/// `run_uniform_conv` and rknpu-spelunking/NOTES.md's pixel-stride
/// section).
fn run_uniform_conv_fp16(shape: &ConvShape, input_fill: f32, weight_fill: f32) -> Vec<f32> {
    let input_bits = f32_to_f16_bits(input_fill);
    let weight_bits = f32_to_f16_bits(weight_fill);

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let buf_a = Buffer::new(fd, TENSOR_SIZE, &file);
        fill_u16(buf_a.host_ptr, TENSOR_SIZE, input_bits);

        let buf_w = Buffer::new(fd, TENSOR_SIZE, &file);
        fill_u16(buf_w.host_ptr, TENSOR_SIZE, weight_bits);

        let buf_bias = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_bias.host_ptr, 0, TENSOR_SIZE);

        let buf_c = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_c.host_ptr, 0, TENSOR_SIZE);

        let bufs = ConvBuffers {
            input_addr: buf_a.dma_address,
            weights_addr: buf_w.dma_address,
            bias_addr: buf_bias.dma_address,
            output_addr: buf_c.dma_address,
        };
        let cmds = build_conv_regcmd(shape, &bufs);

        let cmd_bytes = cmds.len() * mem::size_of::<u64>();
        let cmd_len = cmd_bytes.next_multiple_of(4096);
        let buf_cmd = Buffer::new(fd, cmd_len, &file);
        let cmd_slice = std::slice::from_raw_parts_mut(buf_cmd.host_ptr as *mut u64, cmds.len());
        for (i, c) in cmds.iter().enumerate() {
            cmd_slice[i] = c.0;
        }

        fini_bo(fd, buf_a.handle).ok();
        fini_bo(fd, buf_w.handle).ok();
        fini_bo(fd, buf_bias.handle).ok();
        fini_bo(fd, buf_c.handle).ok();
        fini_bo(fd, buf_cmd.handle).ok();

        let in_handles = [buf_cmd.handle, buf_a.handle, buf_w.handle, buf_bias.handle];
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
                "job did not complete within timeout (input_fill={input_fill}, \
                 weight_fill={weight_fill}): {e}"
            )
        });

        let raw = std::slice::from_raw_parts(buf_c.host_ptr, 256);
        (0..16)
            .map(|i| f16_to_f32(u16::from_le_bytes([raw[i * 16], raw[i * 16 + 1]])))
            .collect()
    }
}

/// The real correctness check fp16 gets that int8 never could: every pair
/// here is chosen so both operands AND their product are exactly
/// representable in fp16, so this asserts exact equality, not a
/// tolerance -- no rounding to hand-wave away. `(10.5, 0.25, 2.625)` is
/// the exact case that first broke the NaN this whole investigation
/// chased (round 7, see rknpu-spelunking/NOTES.md); `(3.0, 2.0, 6.0)` is
/// this repo's very first fp16 attempt's original input (which used to
/// come back as NaN `0x7c01` -- rounds 1-6 never got a real number out of
/// it at all).
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn conv_fp16_tracks_exact_product() {
    let shape = fp16_shape();
    for (input_fill, weight_fill, expected) in [
        (10.5f32, 0.25f32, 2.625f32),
        (3.0, 2.0, 6.0),
        (4.0, 4.0, 16.0),
    ] {
        let pixels = run_uniform_conv_fp16(&shape, input_fill, weight_fill);
        assert!(
            pixels.iter().all(|&p| p == expected),
            "input_fill={input_fill}, weight_fill={weight_fill}: expected all 16 \
             output pixels == {expected} exactly, got {pixels:?}"
        );
    }
}

/// fp16 counterpart to `conv_3x3_uniform_fill_pixels_agree` -- same
/// uniform-fill packing-order-agnostic check, non-exact fill values this
/// time (not every fp16 product needs to be exact for this particular
/// assertion, only that all 16 pixels see the identical computation).
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn conv_fp16_uniform_fill_pixels_agree() {
    let shape = fp16_shape();
    for input_fill in [1.5f32, 10.5, 100.0] {
        let pixels = run_uniform_conv_fp16(&shape, input_fill, 0.25);
        assert!(
            pixels.iter().all(|&p| p == pixels[0]),
            "input_fill={input_fill}: expected all 16 output pixels identical \
             (uniform input/weights), got {pixels:?}"
        );
    }
}

//===========================================================================
// fp16 multi-channel (`input_channels > 1`) -- MobileNetV2 Fp16 support
// investigation. EXPERIMENTAL: `build_conv_cna_core_dpu_dpu_rdma`'s
// multi-channel branch was previously `assert!`-blocked for
// `Precision::Fp16` (every fp16 hw test above uses `input_channels == 1`);
// see that function's own comment on the two real generalizations made
// (`compute_input_banks`/`input_data_entries` now scale by
// `Precision::bytes_per_element()`, mirroring Mesa's `calc_entries_per_
// slice()`'s own -- currently hardcoded-to-1 -- `bpe` factor) to allow this.
// Deliberately keeps `output_channels == 1` and the proven 1x1-kernel/4x4
// spatial geometry from `fp16_shape()` unchanged -- isolates "does
// multi-input-channel accumulation work for fp16" as the ONLY new variable,
// rather than also introducing an untested kernel size or the SEPARATE
// (rocket-hal-driver-side) output_channels>1 compaction gap in the same
// experiment.
//===========================================================================

/// `fp16_shape()`'s exact proven geometry, with `input_channels` bumped up
/// from 1 -- see this section's own module-level doc comment for why
/// nothing else changes.
fn fp16_multi_channel_shape(input_channels: u32) -> ConvShape {
    ConvShape {
        input_channels,
        ..fp16_shape()
    }
}

/// Same idea as `run_uniform_conv_fp16`, but for a real `input_channels`
/// that may be smaller than the hardware's padded `task_input_channels`
/// (`.max(16).next_multiple_of(16)`, see `compute_task_input_channels`):
/// uniformly filling the ENTIRE weight buffer (as `run_uniform_conv_fp16`
/// does) is only correct when `input_channels` is already a multiple of
/// 16 -- otherwise the DPU genuinely reads `task_input_channels` worth of
/// weight data per `CNA_WEIGHT_SIZE1`/`weight_bytes_per_kernel`
/// (`weights_width*weights_height*task_input_channels*bpe`, confirmed by
/// `build_conv_regcmd`'s own register construction), and a real compiled
/// model's weight tensor is zero-padded past its real channel count --
/// the padding channels contribute `input * 0 = 0` regardless of what's in
/// the input buffer there. First hardware round of this test uniformly
/// filled the WHOLE weight buffer (matching `run_uniform_conv_fp16`,
/// correct for `single_channel`/int8-multi-channel's existing tests only
/// because their fills happened to be int8-byte-uniform in a way that
/// didn't matter, or channel counts that were already 16-aligned) and got
/// exactly `42.0 == 16 * 2.625` for `input_channels=3` -- not garbage, a
/// clean confirmation the hardware/CVT/banking math is right and reads
/// exactly `task_input_channels=16` channels of real (uniformly-filled,
/// undifferentiated) weight data, i.e. a real test-methodology gap, not a
/// regcmd bug. This constructs the weight buffer the way a real compiler
/// would instead: `weight_fill` for the first `input_channels` real
/// channels of kernel 0 (1x1 kernel, so no per-tap layout to also get
/// right), zero for every padding channel and for kernel 1 (the
/// `weights_kernels`-padding kernel `output_channels=1` rounds up to,
/// irrelevant to the one real output channel this test reads back).
fn run_multi_channel_conv_fp16(shape: &ConvShape, input_fill: f32, weight_fill: f32) -> Vec<f32> {
    const FEATURE_ATOMIC_SIZE: u32 = 16;
    const BPE: usize = 2; // Precision::Fp16::bytes_per_element()

    let input_bits = f32_to_f16_bits(input_fill);
    let weight_bits = f32_to_f16_bits(weight_fill);
    let task_input_channels = shape
        .input_channels
        .max(FEATURE_ATOMIC_SIZE)
        .next_multiple_of(FEATURE_ATOMIC_SIZE) as usize;
    let weights_kernels = shape.output_channels.next_multiple_of(2) as usize;
    let bytes_per_kernel = task_input_channels * BPE; // 1x1 kernel: no tap dimension
    let real_channel_bytes = shape.input_channels as usize * BPE;

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let buf_a = Buffer::new(fd, TENSOR_SIZE, &file);
        fill_u16(buf_a.host_ptr, TENSOR_SIZE, input_bits);

        let buf_w = Buffer::new(fd, TENSOR_SIZE, &file);
        // Zero the whole buffer first (covers every padding channel AND
        // every padding kernel uniformly), then fill in only the real
        // channels of the one real kernel (kernel 0).
        ptr::write_bytes(buf_w.host_ptr, 0, TENSOR_SIZE);
        assert!(
            weights_kernels * bytes_per_kernel <= TENSOR_SIZE,
            "test buffer too small for this shape's real weight footprint"
        );
        fill_u16(buf_w.host_ptr, real_channel_bytes, weight_bits);

        let buf_bias = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_bias.host_ptr, 0, TENSOR_SIZE);

        let buf_c = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_c.host_ptr, 0, TENSOR_SIZE);

        let bufs = ConvBuffers {
            input_addr: buf_a.dma_address,
            weights_addr: buf_w.dma_address,
            bias_addr: buf_bias.dma_address,
            output_addr: buf_c.dma_address,
        };
        let cmds = build_conv_regcmd(shape, &bufs);

        let cmd_bytes = cmds.len() * mem::size_of::<u64>();
        let cmd_len = cmd_bytes.next_multiple_of(4096);
        let buf_cmd = Buffer::new(fd, cmd_len, &file);
        let cmd_slice = std::slice::from_raw_parts_mut(buf_cmd.host_ptr as *mut u64, cmds.len());
        for (i, c) in cmds.iter().enumerate() {
            cmd_slice[i] = c.0;
        }

        fini_bo(fd, buf_a.handle).ok();
        fini_bo(fd, buf_w.handle).ok();
        fini_bo(fd, buf_bias.handle).ok();
        fini_bo(fd, buf_c.handle).ok();
        fini_bo(fd, buf_cmd.handle).ok();

        let in_handles = [buf_cmd.handle, buf_a.handle, buf_w.handle, buf_bias.handle];
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
                "job did not complete within timeout (input_channels={}, \
                 input_fill={input_fill}, weight_fill={weight_fill}): {e}",
                shape.input_channels
            )
        });

        let raw = std::slice::from_raw_parts(buf_c.host_ptr, 256);
        (0..16)
            .map(|i| f16_to_f32(u16::from_le_bytes([raw[i * 16], raw[i * 16 + 1]])))
            .collect()
    }
}

/// Real (non-16-aligned-friendly) per-channel fill across every spatial
/// position -- with a 1x1 kernel this makes the expected output an EXACT
/// multiple of `conv_fp16_tracks_exact_product`'s own single-channel
/// result: `input_channels * input_fill * weight_fill`. This is
/// deliberately a stronger check than "all pixels agree" or "output
/// changed": if the multi-channel CVT/banking generalization silently
/// only reads channel 0 (or reads garbage past it), the result would NOT
/// scale linearly with `input_channels` the way a real per-channel
/// accumulation must.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn conv_fp16_multi_channel_tracks_exact_product() {
    for input_channels in [3u32, 16] {
        let shape = fp16_multi_channel_shape(input_channels);
        for (input_fill, weight_fill, single_channel_expected) in [
            (10.5f32, 0.25f32, 2.625f32),
            (3.0, 2.0, 6.0),
            (4.0, 4.0, 16.0),
        ] {
            let expected = single_channel_expected * input_channels as f32;
            let pixels = run_multi_channel_conv_fp16(&shape, input_fill, weight_fill);
            assert!(
                pixels.iter().all(|&p| p == expected),
                "input_channels={input_channels}, input_fill={input_fill}, \
                 weight_fill={weight_fill}: expected all 16 output pixels == \
                 {expected} exactly ({input_channels} channels x \
                 {single_channel_expected}), got {pixels:?}"
            );
        }
    }
}

/// Multi-channel counterpart to `conv_fp16_uniform_fill_pixels_agree` --
/// guards against a hollow "completes but doesn't really touch channel
/// data" pass the exact-product test's uniform fill could theoretically
/// still satisfy by coincidence (e.g. a stuck-at-zero accumulator that
/// happens to also fail the exact-product check loudly, but a weaker bug
/// might not).
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn conv_fp16_multi_channel_output_tracks_input() {
    let shape = fp16_multi_channel_shape(3);
    let low = run_uniform_conv_fp16(&shape, 10.5, 0.25)[0];
    let high = run_uniform_conv_fp16(&shape, 100.0, 0.25)[0];
    assert_ne!(
        low, high,
        "input_channels=3: output pixel value didn't change between \
         input_fill=10.5 ({low}) and input_fill=100.0 ({high}) -- suggests \
         the op isn't really reading the input"
    );
}

//===========================================================================
// Position-dependent fp16 input -- MobileNetV2's C32 -> C16 1x1 conv.
//
// Uniform input fills intentionally hide input-layout and task-offset bugs.
// These cases instead use dense NHWC input where every element is derived
// from its (y, x, c) coordinate. Uniform 1/32 weights make every output
// channel equal the input-channel average, so the reference remains simple
// while still detecting spatially misplaced or missing input data.
//===========================================================================

const POSITION_INPUT_CHANNELS: u32 = 32;
const POSITION_OUTPUT_CHANNELS: u32 = 16;
const POSITION_WEIGHT: f32 = 1.0 / POSITION_INPUT_CHANNELS as f32;

fn fp16_position_shape(width: u32, height: u32) -> ConvShape {
    ConvShape {
        input_width: width,
        input_height: height,
        input_channels: POSITION_INPUT_CHANNELS,
        output_width: width,
        output_height: height,
        output_channels: POSITION_OUTPUT_CHANNELS,
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
        precision: Precision::Fp16,
    }
}

/// Coordinate-derived dense NHWC input value.
///
/// The spatial term cycles through 1..=16 in a nontrivial row/column
/// pattern, while `channel` makes all 32 values within a pixel distinct.
/// Every value is an integer and therefore exactly representable in fp16.
fn fp16_position_input_value(y: u32, x: u32, channel: u32) -> f32 {
    let spatial = 1 + (y * 7 + x * 3) % 16;
    (spatial + channel) as f32
}

/// With all 32 weights equal to 1/32, sum_c(spatial + c) / 32 is
/// `spatial + 15.5`. The result and every intermediate product are exact
/// binary fractions within fp16's precision at this magnitude.
fn fp16_position_expected(y: u32, x: u32) -> f32 {
    let spatial = 1 + (y * 7 + x * 3) % 16;
    spatial as f32 + 15.5
}

fn page_aligned_size(byte_len: usize) -> usize {
    byte_len.max(1).next_multiple_of(4096)
}

/// Packs a dense NHWC C32 input into NC1HWC2, runs the fp16 convolution,
/// and decodes DPU's feature-atomic output surfaces into dense NHWC order.
///
/// Height splits are submitted as individually fenced DRM jobs. A real
/// RK3588 run of the same regcmds as multiple tasks in one `drm_rocket_job`
/// produced correct task-0 rows and left every later row at zero; separate
/// jobs produced the complete expected tensor.
fn run_position_conv_fp16(shape: &ConvShape) -> Vec<f32> {
    assert_eq!(shape.input_channels, POSITION_INPUT_CHANNELS);
    assert_eq!(shape.output_channels, POSITION_OUTPUT_CHANNELS);

    const BPE: usize = 2;
    let pixel_count = shape.input_width as usize * shape.input_height as usize;
    let input_bytes_per_pixel = shape.input_channels as usize * BPE;
    let input_bytes = pixel_count * input_bytes_per_pixel;
    let input_scratch_len = nc1hwc2_storage_size(pixel_count, input_bytes_per_pixel).unwrap();
    let weight_elements = shape.input_channels as usize * shape.output_channels as usize; // 1x1 HWIO
    let output_scratch_len = conv_output_scratch_bytes(shape);

    let mut dense_input = vec![0u8; input_bytes];
    for y in 0..shape.input_height {
        for x in 0..shape.input_width {
            for channel in 0..shape.input_channels {
                let index = ((y * shape.input_width + x) * shape.input_channels + channel) as usize;
                let offset = index * BPE;
                dense_input[offset..offset + BPE].copy_from_slice(
                    &f32_to_f16_bits(fp16_position_input_value(y, x, channel)).to_le_bytes(),
                );
            }
        }
    }

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let buf_a = Buffer::new(fd, page_aligned_size(input_scratch_len), &file);
        ptr::write_bytes(buf_a.host_ptr, 0, buf_a.size);
        let packed_input = std::slice::from_raw_parts_mut(buf_a.host_ptr, input_scratch_len);
        pack_nhwc_to_nc1hwc2(
            &dense_input,
            pixel_count,
            input_bytes_per_pixel,
            packed_input,
        )
        .expect("failed to pack test input as NC1HWC2");

        let buf_w = Buffer::new(fd, page_aligned_size(weight_elements * BPE), &file);
        ptr::write_bytes(buf_w.host_ptr, 0, buf_w.size);
        fill_u16(
            buf_w.host_ptr,
            weight_elements * BPE,
            f32_to_f16_bits(POSITION_WEIGHT),
        );

        let buf_bias = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_bias.host_ptr, 0, TENSOR_SIZE);

        let buf_c = Buffer::new(fd, page_aligned_size(output_scratch_len), &file);
        ptr::write_bytes(buf_c.host_ptr, 0, buf_c.size);

        let bufs = ConvBuffers {
            input_addr: buf_a.dma_address,
            weights_addr: buf_w.dma_address,
            bias_addr: buf_bias.dma_address,
            output_addr: buf_c.dma_address,
        };
        let regcmd_tasks =
            build_conv_regcmd_tasks(shape, &bufs).expect("failed to build convolution tasks");
        let mut cmd_buffers = Vec::with_capacity(regcmd_tasks.len());
        for cmds in &regcmd_tasks {
            let cmd_bytes = cmds.len() * mem::size_of::<u64>();
            let buf_cmd = Buffer::new(fd, page_aligned_size(cmd_bytes), &file);
            cmd_buffers.push(buf_cmd);
        }
        for (cmds, buf_cmd) in regcmd_tasks.iter().zip(&cmd_buffers) {
            let cmd_slice =
                std::slice::from_raw_parts_mut(buf_cmd.host_ptr as *mut u64, cmds.len());
            for (slot, command) in cmd_slice.iter_mut().zip(cmds) {
                *slot = command.0;
            }
            fini_bo(fd, buf_cmd.handle).ok();
        }

        fini_bo(fd, buf_a.handle).ok();
        fini_bo(fd, buf_w.handle).ok();
        fini_bo(fd, buf_bias.handle).ok();
        fini_bo(fd, buf_c.handle).ok();

        let task_descriptors: Vec<_> = cmd_buffers
            .iter()
            .zip(&regcmd_tasks)
            .map(|(buffer, commands)| (buffer.dma_address, commands.len() as u32))
            .collect();
        let mut in_handles: Vec<_> = cmd_buffers.iter().map(|buffer| buffer.handle).collect();
        in_handles.extend([buf_a.handle, buf_w.handle, buf_bias.handle]);
        let out_handles = [buf_c.handle];

        for &(regcmd_addr, regcmd_count) in &task_descriptors {
            submit(fd, regcmd_addr, regcmd_count, &in_handles, &out_handles)
                .expect("single-task SUBMIT ioctl failed");
            prep_bo(fd, buf_c.handle, 2_000_000_000)
                .expect("single-task convolution did not complete within timeout");
        }

        let scratch = std::slice::from_raw_parts(buf_c.host_ptr, output_scratch_len);
        let pixel_count = shape.output_width as usize * shape.output_height as usize;
        let surface_stride = pixel_count * CONV_OUTPUT_ATOMIC_STRIDE as usize;
        let mut output = Vec::with_capacity(pixel_count * shape.output_channels as usize);
        for pixel in 0..pixel_count {
            for channel in 0..shape.output_channels as usize {
                let channel_byte = channel * BPE;
                let surface = channel_byte / CONV_OUTPUT_ATOMIC_STRIDE as usize;
                let byte_in_surface = channel_byte % CONV_OUTPUT_ATOMIC_STRIDE as usize;
                let offset = surface * surface_stride
                    + pixel * CONV_OUTPUT_ATOMIC_STRIDE as usize
                    + byte_in_surface;
                output.push(f16_to_f32(u16::from_le_bytes([
                    scratch[offset],
                    scratch[offset + 1],
                ])));
            }
        }
        output
    }
}

fn assert_position_conv_output(shape: &ConvShape, actual: &[f32]) {
    let mut mismatches = Vec::new();
    for y in 0..shape.output_height {
        for x in 0..shape.output_width {
            let expected = fp16_position_expected(y, x);
            for channel in 0..shape.output_channels {
                let index =
                    ((y * shape.output_width + x) * shape.output_channels + channel) as usize;
                if actual[index] != expected && mismatches.len() < 16 {
                    mismatches.push(format!(
                        "[{y}, {x}, {channel}]: expected {expected}, got {}",
                        actual[index]
                    ));
                }
            }
        }
    }
    assert!(
        mismatches.is_empty(),
        "{} of the first mismatches:\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
}

#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn conv_fp16_c32_to_c16_position_values_single_task() {
    let shape = fp16_position_shape(4, 4);
    assert_eq!(
        plan_conv_tasks(&shape)
            .expect("single-task shape should be valid")
            .len(),
        1
    );
    let actual = run_position_conv_fp16(&shape);
    assert_position_conv_output(&shape, &actual);
}

#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn conv_fp16_c32_to_c16_position_values_height_splits() {
    let shape = fp16_position_shape(112, 112);
    assert_eq!(
        plan_conv_tasks(&shape)
            .expect("MobileNet-sized shape should be valid")
            .len(),
        3
    );
    let actual = run_position_conv_fp16(&shape);
    assert_position_conv_output(&shape, &actual);
}
