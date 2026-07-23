//! Hardware-in-the-loop tests for `ConvShape::activation`'s fused
//! relu/relux wiring, added on top of the already-hardware-validated
//! `build_conv_regcmd` 3x3-kernel path (see `conv_hw.rs`).
//!
//! Not run by a plain `cargo test` -- see `conv_hw.rs`'s doc comment for
//! the cross-compile/copy-to-board workflow (same device, same pattern,
//! just `--test conv_activation_hw`).
//!
//! DPU's BS core's relu/relux bypass bits (`DPU_BS_CFG.bs_relu_bypass`/
//! `bs_relux_en`, `DPU_BS_RELUX_CMP_VALUE`) were wired but permanently
//! bypassed before this session -- these tests are the first real hardware
//! exercise of that path. Working assumption under test here, not yet
//! confirmed: BS applies `relu(bias_add(x))` (ALU-then-RELU order), and
//! relu/relux only affect values that would otherwise be negative/
//! above-clamp -- everything else should be bit-identical to
//! `Activation::None`.
//!
//! Same uniform-whole-buffer-fill strategy as `conv_hw.rs` throughout, for
//! the same reason: sidesteps not knowing the real per-tap weight-buffer
//! packing order.
//!
//! `conv_fp16_relu_*` below is the fp16 counterpart, first real hardware
//! test of `Activation` combined with `ConvShape::precision =
//! Precision::Fp16` (see `conv_hw.rs`'s own `conv_fp16_*` tests and
//! rknpu-spelunking/NOTES.md's fp16 section) -- `DPU_BS_CFG`'s relu bits
//! and fp16's CVT/OUT_CVT/CPEND-bypass fields are set by independent code
//! paths in `build_conv_cna_core_dpu_dpu_rdma`, but this combination had
//! never actually been run on real hardware before these tests. fp16
//! gets a strictly stronger check here than int8's inequality-based tests
//! above: real negative/positive accumulator values chosen so the
//! expected post-relu output is known exactly, not just relative to
//! `Activation::None`'s own output.

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    device::{Buffer, fini_bo, prep_bo, submit},
    regcmd::{Activation, ConvBuffers, ConvShape, Precision, build_conv_regcmd},
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const TENSOR_SIZE: usize = 4096;

/// Same 6x6x1 -> 4x4x1, 3x3 kernel, stride 1 geometry as `conv_hw.rs`'s
/// `single_channel_shape()` -- only `activation` varies across these tests.
fn shape_with_activation(activation: Activation) -> ConvShape {
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
        activation,
        precision: Precision::Int8,
    }
}

/// Identical to `conv_hw.rs`'s `run_uniform_conv`, duplicated rather than
/// shared across test-binary boundaries (integration tests in `tests/`
/// each compile as their own crate -- no `mod` to share this from without
/// a `tests/common/mod.rs` restructure not otherwise needed yet).
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
            panic!("job did not complete within timeout (input_fill={input_fill}): {e}")
        });

        let raw = std::slice::from_raw_parts(buf_c.host_ptr, 256);
        (0..16).map(|i| raw[i * 16]).collect()
    }
}

/// Liveness check, same style as `conv_hw.rs`'s `conv_3x3_output_tracks_input`
/// -- guards against relu collapsing every output to one constant floor
/// value regardless of input (which would also make the "never decreases"
/// check below trivially true for the wrong reason).
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn conv_3x3_relu_output_tracks_input() {
    let shape = shape_with_activation(Activation::Relu);
    let low = run_uniform_conv(&shape, 10, 2)[0];
    let high = run_uniform_conv(&shape, 200, 2)[0];
    assert_ne!(
        low, high,
        "relu output pixel didn't change between input_fill=10 ({low}) and \
         input_fill=200 ({high}) -- suggests relu is clamping everything to \
         a constant floor rather than passing through positive values"
    );
}

/// Core semantic check: relu only ever clamps a would-be-negative
/// accumulator up to zero, never pulls a positive value down. So for any
/// given input/weight fill, `Activation::Relu`'s output must be >= what
/// `Activation::None` produces for the identical shape and fill -- true
/// whether or not that particular fill actually triggers clamping,
/// assuming DPU's output-conversion stage maps the (possibly-clamped)
/// internal accumulator to the final byte via a monotonically-increasing
/// affine transform (scale=1.0, positive shift, per this shape's config).
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn conv_3x3_relu_never_decreases_output() {
    let none_shape = shape_with_activation(Activation::None);
    let relu_shape = shape_with_activation(Activation::Relu);
    for input_fill in [10u8, 60, 118, 160, 200] {
        let none_out = run_uniform_conv(&none_shape, input_fill, 2)[0];
        let relu_out = run_uniform_conv(&relu_shape, input_fill, 2)[0];
        assert!(
            relu_out >= none_out,
            "input_fill={input_fill}: relu output ({relu_out}) is less than \
             the unclamped None output ({none_out}) -- relu should only ever \
             raise a would-be-negative accumulator to zero, never lower a \
             positive one"
        );
    }
}

/// Non-panicking discovery test (same spirit as `pooling_hw.rs`'s
/// `pooling_method_encoding_discovery`): dumps `Activation::None`'s output
/// across a (weight_fill, input_fill) grid so the "still linear" range vs.
/// the output conversion pipeline's own natural saturation ceiling can be
/// read off directly, instead of guessed at.
///
/// Iteration history: a first attempt picked input_fill=160/200
/// (weight_fill=2) as a "should differ" pair and found both already
/// pinned at 127. A follow-up sweep of input_fill=10..200 at weight_fill=2
/// found the *entire* 10..110 range flat at 128 and 130..200 flat at
/// 127 -- a two-level step function, not a ramp, meaning weight_fill=2
/// already drives the accumulator far outside the output's dynamic range
/// even at the smallest input_fill tried. This sweep goes much smaller on
/// both axes (weight_fill down to 1, input_fill down to 0) to find where
/// real gradation exists, since neither prior guess left any linear region
/// to probe a relux cmp against.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn conv_3x3_none_output_by_fill_discovery() {
    let shape = shape_with_activation(Activation::None);
    for weight_fill in [1u8, 2] {
        for input_fill in [0u8, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50] {
            let out = run_uniform_conv(&shape, input_fill, weight_fill)[0];
            eprintln!("weight_fill={weight_fill} input_fill={input_fill}: output={out}");
        }
    }
}

/// Second-round finding (weight_fill in {1,2}, input_fill 0..50, cmp in
/// {0,5,20,50,100}): every combination read flat at 128 -- the small `cmp`
/// values tried were three orders of magnitude below the accumulator scale
/// this shape actually operates at. Corroborating data point: the *only*
/// place any output change was ever observed at all (weight_fill=2,
/// input_fill 110->130) corresponds to an accumulator delta of roughly
/// 9*2*(130-110) = 360 units producing exactly 1 unit of output change
/// (128->127) -- i.e. ~360 accumulator-units per output LSB near that
/// transition, given `conv_scale=1.0`'s placeholder (uncalibrated)
/// quantization. `cmp` needs to be in that same thousands-range domain to
/// have any visible effect, and needs to be probed right around that one
/// known transition zone (accumulator ~=1980..2340) rather than at
/// arbitrary small fills.
///
/// This is the decisive version of that experiment: a fine `input_fill`
/// sweep across the known transition under `None` (to pin the natural
/// break exactly), plus the identical sweep under two `Relux` cmp values --
/// one set well *below* the natural transition's accumulator (1000; if
/// relux's clamp is real and independent of the output stage's own
/// saturation, this should pull the 128->127-style break *earlier* than
/// None's) and one well *above* it (3000; should be indistinguishable from
/// None across this whole range, since the natural ceiling is hit first --
/// a control for "did changing `activation` at all perturb anything it
/// shouldn't").
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn conv_3x3_relux_transition_point_discovery() {
    let fills = [100u8, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140];
    for (label, activation) in [
        ("none", Activation::None),
        ("relux_cmp=1000", Activation::Relux { cmp: 1000 }),
        ("relux_cmp=3000", Activation::Relux { cmp: 3000 }),
    ] {
        let shape = shape_with_activation(activation);
        for input_fill in fills {
            let out = run_uniform_conv(&shape, input_fill, 2)[0];
            eprintln!("{label} input_fill={input_fill}: output={out}");
        }
    }
}

/// Third-round finding: `None`, `Relux{cmp:1000}`, and `Relux{cmp:3000}`
/// produced bit-identical output at every one of the 11 fills swept above,
/// including right at the one known transition -- two `cmp` values three
/// decades apart making zero difference rules out "wrong cmp magnitude,"
/// pointing instead at `bs_relux_en`/`DPU_BS_RELUX_CMP_VALUE` not taking
/// effect at all.
///
/// This test sidesteps the whole "what numeric domain is cmp in" question:
/// `Relux { cmp: 0 }` should clamp *every* accumulator (whatever its real
/// units) to a single fixed point, so the output must become constant
/// across a wide input sweep if the wiring is real -- a falsifiable,
/// domain-independent prediction, unlike the cmp-magnitude guessing above.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn conv_3x3_relux_cmp_zero_forces_constant_output() {
    let shape = shape_with_activation(Activation::Relux { cmp: 0 });
    let outputs: Vec<u8> = [0u8, 50, 100, 150, 200, 255]
        .into_iter()
        .map(|input_fill| run_uniform_conv(&shape, input_fill, 2)[0])
        .collect();
    assert!(
        outputs.iter().all(|&o| o == outputs[0]),
        "expected Relux{{cmp: 0}} to clamp every accumulator to the same \
         single point regardless of input_fill (0/50/100/150/200/255), got \
         {outputs:?} -- since cmp=0 is unambiguous in any numeric domain, a \
         non-constant result here means bs_relux_en/DPU_BS_RELUX_CMP_VALUE \
         isn't actually taking effect on hardware, not just a miscalibrated \
         cmp value"
    );
}

//===========================================================================
// fp16 (`Precision::Fp16`) -- see this file's module doc comment.
//===========================================================================

/// `rkt-fp16.rs`'s/`conv_hw.rs`'s exact proven fp16 geometry (4x4x1 ->
/// 4x4x1, 1x1 kernel, stride 1) -- NOT this file's 6x6/3x3 int8 geometry
/// above, which has never run `Precision::Fp16` on real hardware, and
/// which `build_conv_regcmd` doesn't even permit for `Fp16` (asserts
/// `input_channels==1`).
fn fp16_shape_with_activation(activation: Activation) -> ConvShape {
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
        activation,
        precision: Precision::Fp16,
    }
}

/// Duplicated from `conv_hw.rs`'s own copy (see this file's `run_uniform_conv`
/// doc comment on why integration tests each carry their own copy).
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

/// fp16 counterpart to `run_uniform_conv`, duplicated from `conv_hw.rs`'s
/// own `run_uniform_conv_fp16`.
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

/// Real negative accumulator (`weight_fill=-2.0, input_fill=3.0` ->
/// `-6.0`, exactly representable, no relu): relu must clamp every pixel
/// to exactly `0.0`, not just "less than the unclamped value" the way
/// int8's `conv_3x3_relu_never_decreases_output` has to settle for --
/// fp16 lets this test know the exact pre-clamp value ahead of time.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn conv_fp16_relu_clamps_negative_to_zero() {
    let shape = fp16_shape_with_activation(Activation::Relu);
    let pixels = run_uniform_conv_fp16(&shape, 3.0, -2.0);
    assert!(
        pixels.iter().all(|&p| p == 0.0),
        "weight_fill=-2.0, input_fill=3.0 (unclamped product -6.0): expected \
         every relu-clamped output pixel == 0.0 exactly, got {pixels:?}"
    );
}

/// Counterpart with a genuinely positive accumulator
/// (`weight_fill=2.0, input_fill=3.0` -> `6.0`): relu must pass it
/// through completely unchanged, confirming the clamp-to-zero test above
/// isn't passing for the wrong reason (e.g. relu clamping everything
/// regardless of sign).
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn conv_fp16_relu_passes_positive_unclamped() {
    let shape = fp16_shape_with_activation(Activation::Relu);
    let pixels = run_uniform_conv_fp16(&shape, 3.0, 2.0);
    assert!(
        pixels.iter().all(|&p| p == 6.0),
        "weight_fill=2.0, input_fill=3.0: expected every relu-passthrough \
         output pixel == 6.0 exactly, got {pixels:?}"
    );
}
