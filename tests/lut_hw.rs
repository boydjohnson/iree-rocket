//! Hardware-in-the-loop tests for the DPU LUT block, Phase 4 of the
//! ukernel roadmap. The exact register recipe was reverse-engineered from
//! vendor-compiler (`rknn-toolkit2`) regcmd captures of standalone
//! sigmoid/tanh ONNX exports -- see `rknpu-spelunking/NOTES.md`'s
//! "Decoding the DPU LUT block" section.
//!
//! The standalone DPU/MRDMA LUT route is now the load-bearing path in this
//! file. The fused CNA/CORE->DPU `Activation::Lut` route remains useful as
//! a diagnostic, but hardware has not shown stable curve semantics there,
//! so fused tests below print observations instead of asserting activation
//! shape.
//!
//! Not run by a plain `cargo test` -- see `conv_hw.rs`'s doc comment for
//! the cross-compile/copy-to-board workflow (same device, same pattern,
//! just `--test lut_hw`).
//!
//! Genuinely unconfirmed going in, deliberately not assumed: what real
//! accumulator range the fixed `x` domain (`[-16384, 16384]`, see
//! `LutTable`'s doc comment) maps to for a placeholder `scale=1.0` shape
//! like the one used here -- unlike `conv_activation_hw.rs`'s relu/relux
//! tests (whose `cmp` clamp behavior is domain-independent at `cmp: 0`),
//! there's no equivalent domain-independent trick for a LUT lookup: it
//! either lands inside the table's dynamic range for a given fill or it
//! doesn't, so the fill sweep here is a *discovery* test, not a
//! pre-asserted transition point.
//!
//! Shape design note (revised after the first few hardware rounds all
//! showed sigmoid/tanh saturated identically): with a 3x3 kernel (9-tap
//! sum) and this crate's usual placeholder `input_zero_point:
//! 0`/`weights_zero_point: 0` (which decode to *real* zero points of
//! -128 each, via the same `wrapping_sub(0x80)` convention used
//! elsewhere in `regcmd.rs`), the accumulator for *any* nonzero fill
//! reaches many millions -- `lut_bn_mul`'s confirmed `multiplier =
//! input_scale * 2596.513` then maps that into the LUT's `x` domain,
//! which is only `[-16384, 16384]`, so every fill saturates deep into
//! one table edge where sigmoid and tanh both approach +-1 and become
//! numerically indistinguishable at 8-bit output resolution -- not a
//! regcmd bug, a test-shape mismatch. Fixed here with a 1x1-kernel shape
//! (single tap, no 9x summation -- same pattern already hardware-
//! validated for FC in `fc_hw.rs`) and *real* (non-offset) zero points
//! (`0x80` decodes to a real zero point of `0`), so the accumulator is
//! simply `input_int8 * weight_int8`. The first hardware run with
//! `input_scale=1.0` and `weight_fill=1` produced only a 127/128 split for
//! both sigmoid and tanh -- consistent with the LUT seeing tiny `x` values
//! around zero. This version uses a realistic quantized `input_scale`
//! (`1.0/32.0`, which puts BN's shift value in the same range as the
//! decoded w8a8 vendor captures) and a larger 1x1 weight (`int8 == 64`),
//! letting `LUT_SWEEP_FILLS` (raw bytes decoding to int8 -6..=6) cover the
//! LUT domain without relying on the unusual `input_scale=1.0` corner.
//! `output_scale` also moved off the usual `1.0` placeholder to
//! `1.0/256.0`: sigmoid's real output range is `[0, 1]` and tanh's is
//! `[-1, 1]` -- at `output_scale=1.0` (1 real unit per int8 code) *either*
//! range collapses to just 1-2 distinct output bytes regardless of curve
//! shape, which could still mask a real difference even with the
//! accumulator fix above. `1.0/256.0` spreads a `~2`-wide real range across
//! close to the full byte code space.

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    device::{Buffer, fini_bo, prep_bo, submit},
    regcmd::{
        Activation, ConvBuffers, ConvShape, LutBuffers, LutShape, LutTable, build_conv_regcmd,
        build_lut_regcmd,
    },
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const TENSOR_SIZE: usize = 4096;

fn shape_with_activation(activation: Activation) -> ConvShape {
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
        output_zero_point: 0,
        weights_zero_point: 0x80,
        input_scale: 1.0 / 32.0,
        weights_scale: 1.0,
        output_scale: 1.0 / 256.0,
        truncate_bits: 0,
        activation,
    }
}

fn standalone_lut_shape() -> LutShape {
    standalone_lut_shape_with(0x80, 0, 1.0 / 32.0, 1.0 / 256.0)
}

fn standalone_tanh_shape() -> LutShape {
    standalone_lut_shape_with(0x80, 0x80, 1.0 / 32.0, 1.0 / 256.0)
}

fn standalone_lut_shape_with(
    input_zero_point: u32,
    output_zero_point: u32,
    input_scale: f32,
    output_scale: f32,
) -> LutShape {
    LutShape {
        width: 4,
        height: 4,
        channels: 16,
        input_zero_point,
        output_zero_point,
        input_scale,
        output_scale,
    }
}

/// Raw bytes decoding (as two's-complement int8) to -6, -5, ..., 5, 6 --
/// see `shape_with_activation`'s doc comment for why this range traces
/// almost the entire LUT `x` domain given `weight_fill=64`.
const LUT_SWEEP_FILLS: [u8; 13] = [250, 251, 252, 253, 254, 255, 0, 1, 2, 3, 4, 5, 6];
const LUT_SWEEP_WEIGHT_FILL: u8 = 64;

/// Identical structure to `conv_activation_hw.rs`'s `run_uniform_conv`,
/// duplicated for the same reason (each `tests/*.rs` file is its own
/// crate, no shared `tests/common/mod.rs` yet).
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
                "job did not complete within timeout (input_fill={input_fill}, \
                 weight_fill={weight_fill}) -- LUT table upload or EW/LUT config \
                 may have hung the NPU: {e}"
            )
        });

        let raw = std::slice::from_raw_parts(buf_c.host_ptr, 256);
        (0..16).map(|i| raw[i * 16]).collect()
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
                "standalone LUT job did not complete within timeout (input_fill={input_fill}) -- \
                 DPU/MRDMA flying-mode LUT config may have hung the NPU: {e}"
            )
        });

        let raw = std::slice::from_raw_parts(buf_c.host_ptr, 256);
        raw[..16].to_vec()
    }
}

fn output_code(raw: u8) -> i16 {
    raw as i16
}

fn signed_output_code(raw: u8) -> i16 {
    raw as i8 as i16
}

fn reg_name(offset: u32) -> Option<&'static str> {
    match offset {
        0x400c => Some("DPU_FEATURE_MODE_CFG"),
        0x4010 => Some("DPU_DATA_FORMAT"),
        0x4020 => Some("DPU_DST_BASE_ADDR"),
        0x4024 => Some("DPU_DST_SURF_STRIDE"),
        0x4040 => Some("DPU_BS_CFG"),
        0x4060 => Some("DPU_BN_CFG"),
        0x4064 => Some("DPU_BN_ALU_CFG"),
        0x4068 => Some("DPU_BN_MUL_CFG"),
        0x4050 => Some("DPU_BS_OW_CFG"),
        0x4054 => Some("DPU_BS_OW_OP"),
        0x4058 => Some("DPU_WDMA_SIZE_0"),
        0x405c => Some("DPU_WDMA_SIZE_1"),
        0x4070 => Some("DPU_EW_CFG"),
        0x4080 => Some("DPU_OUT_CVT_OFFSET"),
        0x4084 => Some("DPU_OUT_CVT_SCALE"),
        0x4088 => Some("DPU_OUT_CVT_SHIFT"),
        0x40c0 => Some("DPU_SURFACE_ADD"),
        0x4100 => Some("DPU_LUT_ACCESS_CFG"),
        0x4108 => Some("DPU_LUT_CFG"),
        0x410c => Some("DPU_LUT_INFO"),
        0x4110 => Some("DPU_LUT_LE_START"),
        0x4114 => Some("DPU_LUT_LE_END"),
        0x4118 => Some("DPU_LUT_LO_START"),
        0x411c => Some("DPU_LUT_LO_END"),
        0x4120 => Some("DPU_LUT_LE_SLOPE_SCALE"),
        0x4124 => Some("DPU_LUT_LE_SLOPE_SHIFT"),
        0x4128 => Some("DPU_LUT_LO_SLOPE_SCALE"),
        0x412c => Some("DPU_LUT_LO_SLOPE_SHIFT"),
        0x500c => Some("DPU_RDMA_DATA_CUBE_WIDTH"),
        0x5010 => Some("DPU_RDMA_DATA_CUBE_HEIGHT"),
        0x5014 => Some("DPU_RDMA_DATA_CUBE_CHANNEL"),
        0x5018 => Some("DPU_RDMA_SRC_BASE_ADDR"),
        0x501c => Some("DPU_RDMA_BRDMA_CFG"),
        0x5020 => Some("DPU_RDMA_BS_BASE_ADDR"),
        0x5028 => Some("DPU_RDMA_NRDMA_CFG"),
        0x502c => Some("DPU_RDMA_BN_BASE_ADDR"),
        0x5034 => Some("DPU_RDMA_ERDMA_CFG"),
        0x5038 => Some("DPU_RDMA_EW_BASE_ADDR"),
        0x5040 => Some("DPU_RDMA_EW_SURF_STRIDE"),
        0x5044 => Some("DPU_RDMA_FEATURE_MODE_CFG"),
        0x5048 => Some("DPU_RDMA_SRC_DMA_CFG"),
        0x504c => Some("DPU_RDMA_SURF_NOTCH"),
        0x5064 => Some("DPU_RDMA_PAD_CFG"),
        0x5068 => Some("DPU_RDMA_WEIGHT"),
        0x506c => Some("DPU_RDMA_EW_SURF_NOTCH"),
        _ => None,
    }
}

fn print_mode_decode(name: &str, value: u32) {
    match name {
        "DPU_FEATURE_MODE_CFG" => eprintln!(
            "    decode: flying_mode={} output_mode={} conv_mode={} burst_len={}",
            value & 0x1,
            (value >> 1) & 0x3,
            (value >> 3) & 0x3,
            (value >> 5) & 0xf,
        ),
        "DPU_RDMA_FEATURE_MODE_CFG" => eprintln!(
            "    decode: flying_mode={} conv_mode={} mrdma_disable={} proc_precision={} burst_len={} in_precision={}",
            value & 0x1,
            (value >> 1) & 0x3,
            (value >> 4) & 0x1,
            (value >> 5) & 0x3,
            (value >> 11) & 0xf,
            (value >> 15) & 0x3,
        ),
        _ => {}
    }
}

fn dump_lut_regcmd_summary(shape: &ConvShape) {
    let bufs = ConvBuffers {
        input_addr: 0x1000,
        weights_addr: 0x2000,
        bias_addr: 0x3000,
        output_addr: 0x4000,
    };
    let cmds = build_conv_regcmd(shape, &bufs);

    eprintln!("selected LUT-path regcmds:");
    for cmd in cmds {
        let raw = cmd.0;
        let offset = (raw & 0xffff) as u32;
        if let Some(name) = reg_name(offset) {
            let domain = ((raw >> 48) & 0xffff) as u32;
            let value = ((raw >> 16) & 0xffff_ffff) as u32;
            eprintln!(
                "  domain=0x{domain:04x} offset=0x{offset:04x} {name:<24} value=0x{value:08x}"
            );
            print_mode_decode(name, value);
        }
    }
}

fn dump_standalone_lut_regcmd_summary(shape: &LutShape, table: LutTable) {
    let bufs = LutBuffers {
        input_addr: 0x1000,
        output_addr: 0x4000,
    };
    let cmds = build_lut_regcmd(shape, &bufs, table);

    eprintln!("selected standalone LUT-path regcmds:");
    for cmd in cmds {
        let raw = cmd.0;
        let offset = (raw & 0xffff) as u32;
        if let Some(name) = reg_name(offset) {
            let domain = ((raw >> 48) & 0xffff) as u32;
            let value = ((raw >> 16) & 0xffff_ffff) as u32;
            eprintln!(
                "  domain=0x{domain:04x} offset=0x{offset:04x} {name:<24} value=0x{value:08x}"
            );
            print_mode_decode(name, value);
        }
    }
}

fn dump_lut_access_data_prefix(shape: &LutShape, table: LutTable, limit: usize) {
    let bufs = LutBuffers {
        input_addr: 0x1000,
        output_addr: 0x4000,
    };
    let cmds = build_lut_regcmd(shape, &bufs, table);

    eprintln!("first {limit} DPU_LUT_ACCESS_DATA writes:");
    let mut count = 0;
    for cmd in cmds {
        let raw = cmd.0;
        let offset = (raw & 0xffff) as u32;
        if offset == 0x4104 {
            let value = ((raw >> 16) & 0xffff_ffff) as u32;
            eprintln!("  #{count:03}: value=0x{value:08x}");
            count += 1;
            if count >= limit {
                break;
            }
        }
    }
}

/// Most basic possible check: does the new table-upload-plus-real-EW-
/// config sequence even complete without hanging the NPU? This is
/// genuinely new register territory (never exercised before this test),
/// so a clean completion is not a given.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn lut_sigmoid_completes() {
    let shape = shape_with_activation(Activation::Lut(LutTable::sigmoid()));
    let out = run_uniform_conv(&shape, 0, LUT_SWEEP_WEIGHT_FILL);
    eprintln!("lut_sigmoid_completes: output={out:?}");
}

#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn lut_tanh_completes() {
    let shape = shape_with_activation(Activation::Lut(LutTable::tanh()));
    let out = run_uniform_conv(&shape, 0, LUT_SWEEP_WEIGHT_FILL);
    eprintln!("lut_tanh_completes: output={out:?}");
}

/// Non-panicking discovery sweep (same spirit as `conv_activation_hw.rs`'s
/// `conv_3x3_none_output_by_fill_discovery`): the real accumulator-to-LUT-
/// domain mapping is confirmed at the formula level (see
/// `lut_bn_mul`/`lut_out_cvt` in `regcmd.rs`) but not yet hardware-proven,
/// so this characterizes it empirically rather than asserting a specific
/// transition point.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn lut_fused_sigmoid_output_by_fill_discovery() {
    let shape = shape_with_activation(Activation::Lut(LutTable::sigmoid()));
    for input_fill in LUT_SWEEP_FILLS {
        let out = run_uniform_conv(&shape, input_fill, LUT_SWEEP_WEIGHT_FILL)[0];
        eprintln!(
            "input_fill={input_fill} (int8={}): output={out}",
            input_fill as i8
        );
    }
}

/// Diagnostic only: compare the same conv shape with no LUT, sigmoid LUT,
/// and tanh LUT. If all three columns match, the LUT path is not changing
/// the fused-conv output at all and the next investigation should focus on
/// DPU routing rather than table contents.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn lut_compare_none_sigmoid_tanh_by_fill_discovery() {
    let none_shape = shape_with_activation(Activation::None);
    let sigmoid_shape = shape_with_activation(Activation::Lut(LutTable::sigmoid()));
    let tanh_shape = shape_with_activation(Activation::Lut(LutTable::tanh()));

    for input_fill in LUT_SWEEP_FILLS {
        let none = run_uniform_conv(&none_shape, input_fill, LUT_SWEEP_WEIGHT_FILL)[0];
        let sigmoid = run_uniform_conv(&sigmoid_shape, input_fill, LUT_SWEEP_WEIGHT_FILL)[0];
        let tanh = run_uniform_conv(&tanh_shape, input_fill, LUT_SWEEP_WEIGHT_FILL)[0];
        eprintln!(
            "input_fill={input_fill} (int8={}): none={none} sigmoid={sigmoid} tanh={tanh}",
            input_fill as i8
        );
    }
}

/// Diagnostic only: print the key DPU registers emitted for the current
/// sigmoid shape so hardware results can be compared against captured RKNN
/// regcmds without a separate decoder step.
#[test]
#[ignore = "diagnostic dump only"]
fn lut_sigmoid_regcmd_summary_discovery() {
    let shape = shape_with_activation(Activation::Lut(LutTable::sigmoid()));
    dump_lut_regcmd_summary(&shape);
}

#[test]
#[ignore = "diagnostic dump only"]
fn lut_standalone_sigmoid_regcmd_summary_discovery() {
    let shape = standalone_lut_shape();
    dump_standalone_lut_regcmd_summary(&shape, LutTable::sigmoid());
}

#[test]
#[ignore = "diagnostic dump only"]
fn lut_standalone_tanh_access_data_prefix_discovery() {
    let shape = standalone_lut_shape();
    dump_lut_access_data_prefix(&shape, LutTable::tanh(), 8);
}

#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn lut_standalone_tanh_quantization_variants_discovery() {
    let variants = [
        ("sigmoid_style", standalone_lut_shape()),
        (
            "signed_output_1_256",
            standalone_lut_shape_with(0x80, 0x80, 1.0 / 32.0, 1.0 / 256.0),
        ),
        (
            "signed_output_1_128",
            standalone_lut_shape_with(0x80, 0x80, 1.0 / 32.0, 1.0 / 128.0),
        ),
        (
            "signed_input_output",
            standalone_lut_shape_with(0, 0x80, 1.0 / 32.0, 1.0 / 128.0),
        ),
    ];
    let fills = [250u8, 255, 0, 1, 6, 128];

    for (name, shape) in variants {
        eprintln!(
            "variant={name} input_zero_point={} output_zero_point={} input_scale={} output_scale={}",
            shape.input_zero_point, shape.output_zero_point, shape.input_scale, shape.output_scale
        );
        for input_fill in fills {
            let raw = run_uniform_standalone_lut(&shape, LutTable::tanh(), input_fill)[0];
            eprintln!(
                "  input_fill={input_fill:3} (i8={:4}): raw={raw:3} unsigned_code={} signed_code={}",
                input_fill as i8,
                output_code(raw),
                signed_output_code(raw)
            );
        }
    }
}

/// Standalone DPU/MRDMA LUT path matching the vendor compiler's decoded
/// sigmoid/tanh task routing (`DPU_FEATURE_MODE_CFG.flying_mode=1`,
/// `DPU_RDMA_FEATURE_MODE_CFG.mrdma_disable=0`). This is the stable
/// discriminative test while the fused CNA/CORE->DPU route remains
/// diagnostic.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn lut_standalone_sigmoid_and_tanh_differ_somewhere() {
    let shape = standalone_lut_shape();

    let mut any_diff = false;
    for input_fill in LUT_SWEEP_FILLS {
        let sig_out = run_uniform_standalone_lut(&shape, LutTable::sigmoid(), input_fill)[0];
        let tanh_out = run_uniform_standalone_lut(&shape, LutTable::tanh(), input_fill)[0];
        eprintln!(
            "standalone input_fill={input_fill} (int8={}): sigmoid={sig_out} tanh={tanh_out}",
            input_fill as i8
        );
        if sig_out != tanh_out {
            any_diff = true;
        }
    }
    assert!(
        any_diff,
        "standalone sigmoid and tanh produced identical output at every fill in \
         {LUT_SWEEP_FILLS:?}; vendor routing is now matched, so a full match here \
         points at table upload/order or DPU_RDMA input layout rather than fused-path routing"
    );
}

/// Load-bearing shape check for standalone sigmoid. The chosen output
/// conversion wraps the sigmoid output around byte zero, so signed code
/// order is the meaningful monotonicity check for this diagnostic shape.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn lut_standalone_sigmoid_is_monotonic_signed() {
    let shape = standalone_lut_shape();
    let mut prev = None;

    for input_fill in LUT_SWEEP_FILLS {
        let raw = run_uniform_standalone_lut(&shape, LutTable::sigmoid(), input_fill)[0];
        let code = signed_output_code(raw);
        eprintln!(
            "standalone input_fill={input_fill} (int8={}): sigmoid_raw={raw} sigmoid_code={code}",
            input_fill as i8
        );

        if let Some(prev_code) = prev {
            assert!(
                code >= prev_code,
                "standalone sigmoid output is not monotonic over the LUT sweep: \
                 previous_code={prev_code}, current_code={code}, input_fill={input_fill} \
                 (int8={})",
                input_fill as i8
            );
        }
        prev = Some(code);
    }
}

#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn lut_standalone_tanh_negative_zero_positive_are_ordered() {
    let shape = standalone_tanh_shape();

    let neg_raw = run_uniform_standalone_lut(&shape, LutTable::tanh(), 250)[0];
    let zero_raw = run_uniform_standalone_lut(&shape, LutTable::tanh(), 0)[0];
    let pos_raw = run_uniform_standalone_lut(&shape, LutTable::tanh(), 6)[0];

    let neg = signed_output_code(neg_raw);
    let zero = signed_output_code(zero_raw);
    let pos = signed_output_code(pos_raw);
    eprintln!(
        "standalone tanh shape check: neg_raw={neg_raw} neg_code={neg}, \
         zero_raw={zero_raw} zero_code={zero}, pos_raw={pos_raw} pos_code={pos}"
    );

    assert!(
        neg < zero,
        "standalone tanh negative-fill output should be below zero-fill output: \
         neg_raw={neg_raw} neg_code={neg}, zero_raw={zero_raw} zero_code={zero}"
    );
    assert!(
        zero < pos,
        "standalone tanh positive-fill output should be above zero-fill output: \
         zero_raw={zero_raw} zero_code={zero}, pos_raw={pos_raw} pos_code={pos}"
    );
}

/// Fused-path diagnostic only. The standalone route above is the
/// load-bearing LUT path; this sweep remains useful for checking whether
/// the fused CNA/CORE->DPU route is changing as register experiments
/// continue.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn lut_fused_sigmoid_and_tanh_by_fill_discovery() {
    let sigmoid_shape = shape_with_activation(Activation::Lut(LutTable::sigmoid()));
    let tanh_shape = shape_with_activation(Activation::Lut(LutTable::tanh()));

    for input_fill in LUT_SWEEP_FILLS {
        let sig_out = run_uniform_conv(&sigmoid_shape, input_fill, LUT_SWEEP_WEIGHT_FILL)[0];
        let tanh_out = run_uniform_conv(&tanh_shape, input_fill, LUT_SWEEP_WEIGHT_FILL)[0];
        eprintln!(
            "input_fill={input_fill} (int8={}): sigmoid={sig_out} tanh={tanh_out}",
            input_fill as i8
        );
    }
}

/// Fused-path diagnostic only. This intentionally does not assert
/// monotonicity because hardware runs have shown the fused route can
/// produce discontinuities even when the standalone LUT route is ordered.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn lut_fused_sigmoid_monotonicity_discovery() {
    let shape = shape_with_activation(Activation::Lut(LutTable::sigmoid()));
    let mut prev = None;

    for input_fill in LUT_SWEEP_FILLS {
        let raw = run_uniform_conv(&shape, input_fill, LUT_SWEEP_WEIGHT_FILL)[0];
        let code = output_code(raw);
        eprintln!(
            "input_fill={input_fill} (int8={}): sigmoid_raw={raw} sigmoid_code={code}",
            input_fill as i8
        );

        if let Some(prev_code) = prev {
            eprintln!("  delta_from_previous={}", code - prev_code);
        }
        prev = Some(code);
    }
}

/// Fused-path diagnostic only. The equivalent standalone tanh test above
/// is the load-bearing signed-shape assertion.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn lut_fused_tanh_negative_zero_positive_discovery() {
    let shape = shape_with_activation(Activation::Lut(LutTable::tanh()));

    let neg_raw = run_uniform_conv(&shape, 250, LUT_SWEEP_WEIGHT_FILL)[0];
    let zero_raw = run_uniform_conv(&shape, 0, LUT_SWEEP_WEIGHT_FILL)[0];
    let pos_raw = run_uniform_conv(&shape, 6, LUT_SWEEP_WEIGHT_FILL)[0];

    let neg = output_code(neg_raw);
    let zero = output_code(zero_raw);
    let pos = output_code(pos_raw);
    eprintln!(
        "tanh shape check: neg_raw={neg_raw} neg_code={neg}, \
         zero_raw={zero_raw} zero_code={zero}, pos_raw={pos_raw} pos_code={pos}"
    );
}
