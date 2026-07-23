//! NPU diagnostic: real fp16 dispatch attempt, informed by reading the
//! actual RK3588 TRM (chapter 36, "RKNN") -- see
//! rknpu-spelunking/NOTES.md's fp16 section for the full derivation.
//!
//! The first version of this test only patched CNA_CONV_CON1's
//! IN_PRECISION/PROC_PRECISION fields to 2 (fp16) and still saw int8-style
//! saturating output (`0x7f`, the exact same rail the known-good int8
//! sweep hits). The TRM explains why: precision/format state is spread
//! across *four* separate places, and `build_conv_regcmd` (faithfully
//! porting Mesa's int8-only driver) only ever configures one of them:
//!
//! 1. `CNA_CONV_CON1.IN_PRECISION`/`PROC_PRECISION` (0x100c, bits 4-6/7-9)
//!    -- already patched by the first version.
//! 2. `CORE_MISC_CFG.PROC_PRECISION` (0x3010, bits 8-10) -- same 3-bit
//!    enum, never touched by build_conv_regcmd (only sets qd_en/dw_en).
//! 3. `DPU_DATA_FORMAT.IN_PRECISION`/`OUT_PRECISION`/`PROC_PRECISION`
//!    (0x4010) -- DPU has its *own*, different 6-value enum
//!    (int8/int16/fp16/bf16/int32/fp32 -- no int4/tf32, but does have
//!    int32/fp32 unlike CNA/CORE's enum). `fp16` is still `2` in this
//!    enum too. build_conv_regcmd zeros this register entirely
//!    (hardcodes int8 here regardless of what CNA says).
//! 4. `DPU_RDMA_FEATURE_MODE_CFG.IN_PRECISION`/`PROC_PRECISION` (0x5044)
//!    -- also never touched.
//!
//! Plus two more pieces the TRM calls out specifically for fp16:
//! - `CNA_CVT_CON0.CVT_BYPASS` -- the per-channel input CVT stage is an
//!   integer requantization block (scale/offset/truncate, the thing that
//!   does the 128 zero-point centering); real fp16 data doesn't need it,
//!   so this patches bypass on instead of leaving the int8 truncate/scale/
//!   offset=65408 pattern `build_conv_regcmd` uses for the
//!   single-input-channel case.
//! - `DPU_OUT_CVT_SCALE.FP32TOFP16_EN` (0x4084, bit 16) -- documented in
//!   the TRM as part of how the accumulator gets converted to an fp16
//!   output. The `Register<DpuOutCvtScale>` setter for this already
//!   existed (bindgen-generated from Mesa's registers.xml) -- it just was
//!   never called anywhere.
//!
//! Every patch here is a post-processing rewrite over build_conv_regcmd's
//! output (matching by register offset), not a change to the shared,
//! int8-validated builder itself -- same technique the precision-only
//! probe used.
//!
//! Output readback: pixels are NOT tightly packed -- see
//! rknpu-spelunking/NOTES.md's "RESOLVED: the only pixel [0,0] written
//! mystery" section (rkt-shape-a3.rs/rkt-shape-b.rs). Each output pixel
//! occupies its own 16-byte-aligned atomic slot regardless of real
//! channel/precision width, so this reads 256 bytes and extracts u16
//! values at stride 16, not just the first 8 bytes.

use std::{fs::OpenOptions, mem, num::NonZeroUsize, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    api::{
        DRM_COMMAND_BASE, DRM_IOCTL_BASE, DRM_ROCKET_CREATE_BO, DRM_ROCKET_FINI_BO,
        DRM_ROCKET_PREP_BO, DRM_ROCKET_SUBMIT, drm_rocket_create_bo, drm_rocket_fini_bo,
        drm_rocket_job, drm_rocket_prep_bo, drm_rocket_submit, drm_rocket_task,
    },
    builders::{
        Bits, RegCmd, Register, RegisterMeta,
        cna::{
            CnaConvCon1, CnaCvtCon0, CnaCvtCon1, CnaCvtCon2, CnaCvtCon3, CnaCvtCon4, CnaCvtCon5,
            CnaPadCon1, CnaWeightSize0, CnaWeightSize1,
        },
        core::CoreMiscCfg,
        dpu::{
            DpuBsOwCfg, DpuBsOwOp, DpuDataFormat, DpuOutCvtOffset, DpuOutCvtScale, DpuOutCvtShift,
            DpuWdmaSize0,
        },
        dpu_rdma::DpuRdmaFeatureModeCfg,
    },
    debug::dump_cmds,
    regcmd::{Activation, ConvBuffers, ConvShape, build_conv_regcmd},
    registers::{
        CNA_WEIGHT_SIZE0_WEIGHT_BYTES__MASK, CNA_WEIGHT_SIZE1_WEIGHT_BYTES_PER_KERNEL__MASK,
    },
};
use nix::{
    ioctl_readwrite, ioctl_write_ptr,
    sys::mman::{MapFlags, ProtFlags, mmap},
    time::{ClockId, clock_gettime},
};

/// Applies `f` to every regcmd entry whose offset matches `R::OFFSET`
/// (there can be more than one -- Mesa/`build_conv_regcmd` write
/// CNA_CONV_CON1 twice), seeding the builder with the entry's *current*
/// encoded value first so `f`'s typed setters only touch their own
/// field(s), preserving whatever `build_conv_regcmd` already put in the
/// rest of the register -- same read-modify-write `patch_field` used to
/// do by hand, but through the same `Register<R>` setters
/// `build_conv_regcmd` itself uses, instead of raw offset/mask/shift
/// arithmetic against bindgen constants.
fn replace_reg<R: RegisterMeta>(cmds: &mut [RegCmd], mut f: impl FnMut(u32, &mut Register<R>)) {
    for cmd in cmds.iter_mut() {
        if (cmd.0 & 0xFFFF) as u32 == R::OFFSET {
            let orig = ((cmd.0 >> 16) & 0xFFFF_FFFF) as u32;
            let mut reg = Register::<R>::from_val(orig);
            f(orig, &mut reg);
            *cmd = reg.build();
        }
    }
}

/// All six patches described in the module doc comment, precision fields
/// all set to `2` (fp16 in both CNA/CORE's 8-value enum and DPU's
/// 6-value enum).
fn patch_for_fp16(cmds: &mut [RegCmd]) {
    replace_reg::<CnaConvCon1>(cmds, |_, r| {
        r.in_precision(Bits::new(2)).proc_precision(Bits::new(2));
    });
    replace_reg::<CoreMiscCfg>(cmds, |_, r| {
        r.proc_precision(Bits::new(2));
    });
    replace_reg::<DpuDataFormat>(cmds, |_, r| {
        r.in_precision(Bits::new(2))
            .out_precision(Bits::new(2))
            .proc_precision(Bits::new(2));
    });
    replace_reg::<DpuRdmaFeatureModeCfg>(cmds, |_, r| {
        r.in_precision(Bits::new(2)).proc_precision(Bits::new(2));
    });
    // build_conv_regcmd's `input_channels_real_is_one` branch (our shape)
    // never sets data_sign/cvt_type -- they stay 0 (unsigned /
    // mul-then-add). The *other* (multi-channel) branch explicitly sets
    // both to 1 alongside its own cvt_bypass. Matching that convention
    // here in case data_sign=0 (unsigned) is quietly corrupting fp16's
    // own sign bit somewhere downstream, even with the stage nominally
    // bypassed.
    // Round 7: the diff against conv.rknn's real ground truth flagged
    // CNA_CVT_CON0 as DIFFERING (real=0xb, ours=0x038e38eb) back in round
    // 5, but only the bypass/data_sign/cvt_type bits were ever patched --
    // the leftover cvt_truncate_0-3=14 fields build_conv_regcmd computes
    // for the (never taken, for fp16) non-bypassed int8 truncate formula
    // were left in place underneath them. Real ground truth has ALL FOUR
    // truncate fields at 0. Zeroing them now that round 6 (DPU_BS_OW_CFG/
    // OP bypass) got a real, non-NaN result on hardware for the first
    // time -- this was the one remaining un-actioned item from that same
    // diff table.
    replace_reg::<CnaCvtCon0>(cmds, |_, r| {
        r.cvt_bypass(Bits::new(1))
            .data_sign(Bits::new(1))
            .cvt_type(Bits::new(1))
            .cvt_truncate_0(Bits::new(0))
            .cvt_truncate_1(Bits::new(0))
            .cvt_truncate_2(Bits::new(0))
            .cvt_truncate_3(Bits::new(0));
    });
    // No documented bypass bit exists for the DPU_OUT_CVT block (TRM
    // 0x4080-0x4088) -- build_conv_regcmd's offset/scale/shift values are
    // computed via an int8-specific fixed-point requantization formula
    // (bit-tricking a float32's mantissa/exponent), which is very likely
    // why the first fp16 attempt (fp32tofp16_en set, but these three left
    // at their int8-formula values) came back as NaN (0x7c01) instead of
    // a real number. Neutralize all three to 0 -- the closest thing to
    // "off" available without a real bypass bit -- while leaving
    // fp32tofp16_en set. Unconfirmed whether scale=0 is actually a no-op
    // or itself degenerate (e.g. "multiply by 0") -- this is a genuine
    // guess, not a documented identity configuration.
    replace_reg::<DpuOutCvtScale>(cmds, |_, r| {
        r.fp32tofp16_en(Bits::new(1)).out_cvt_scale(Bits::new(0));
    });
    replace_reg::<DpuOutCvtOffset>(cmds, |_, r| {
        r.out_cvt_offset(Bits::new(0));
    });
    replace_reg::<DpuOutCvtShift>(cmds, |_, r| {
        r.cvt_type(Bits::new(0))
            .cvt_round(Bits::new(0))
            .minus_exp(Bits::new(0))
            .out_cvt_shift(Bits::new(0));
    });
}

/// Round 5, grounded in real vendor-compiled ground truth instead of a
/// guess: `conv.rknn` (this repo's very first captured regcmd program,
/// `config.conv.toml`'s `do_quantization = false` build) was always
/// assumed to be an int8 dispatch and set aside as uninteresting for
/// fp16. Diffing its decode (`conv_rknn_decode.txt`) against this file's
/// own dump properly (see rknpu-spelunking chat log, user prompted a
/// direct diff) shows `DPU_DATA_FORMAT = 0x48000002` -- in/out/proc
/// precision all decode to `2` (fp16) -- and `CNA_CVT_CON0 = 0xb`, i.e.
/// `cvt_bypass=1, data_sign=1, cvt_type=1` all set together, the EXACT
/// same 3 bits `patch_for_fp16` already sets. This looks like a genuine
/// real non-int8 (fp16-shaped) dispatch, not the int8 baseline it was
/// assumed to be -- and it disagrees with this file's own patches in
/// several concrete, previously-guessed-wrong places:
/// - `CNA_CVT_CON1-4`: real is `0x00010000` (`cvt_scale=1, cvt_offset=0`)
///   on ALL FOUR. This file's patches never touched these registers at
///   all -- `patch_for_fp16` only ever set `CNA_CVT_CON0`'s bypass/sign/
///   type bits, leaving `build_conv_regcmd`'s original int8-truncate
///   formula (`cvt_scale=16384, cvt_offset=65408`) sitting in CON1-4
///   completely unpatched, even though CON0 claims to bypass the whole
///   stage. If bypass doesn't fully ignore these fields, `16384` is a
///   huge stray multiplier feeding downstream -- a very plausible NaN
///   source that's gone unexamined for all 4 prior rounds.
/// - `CNA_CVT_CON5.per_channel_cvt_en`: real is `0` (disabled) --
///   `build_conv_regcmd`'s `input_channels_real_is_one` branch
///   unconditionally sets this to `65535` regardless of dtype, and no
///   prior round cleared it back to 0 to match a genuinely bypassed CVT.
/// - `CNA_PAD_CON1`: real is `0` -- `build_conv_regcmd` computes
///   `input_zero_point.wrapping_sub(0x80)` (`0xffffff80` for our
///   `zero_point=0` shape), the int8 zero-point-centering pad
///   convention, never overridden for fp16.
/// - `CORE_MISC_CFG.qd_en`: real is `0` -- `build_conv_regcmd` sets this
///   unconditionally (`misc_cfg_builder.qd_en(Bits::new(1))`) regardless
///   of dtype; the real dispatch leaves it clear.
/// - `DPU_OUT_CVT_SCALE.out_cvt_scale`: real is `1`, not `0` -- round 1-3
///   zeroed this as an explicit, flagged guess ("closest thing to off
///   available"); ground truth says the real identity value is `1`.
/// - `DPU_WDMA_SIZE_0.tp_precision`: real is `0` (unset) -- round 4's own
///   hypothesis is REFUTED by this ground truth (consistent with round 4
///   producing zero change on hardware). Not called from `main` anymore.
fn patch_from_conv_rknn_ground_truth(cmds: &mut [RegCmd]) {
    replace_reg::<CnaCvtCon1>(cmds, |_, r| {
        r.cvt_scale0(Bits::new(1)).cvt_offset0(Bits::new(0));
    });
    replace_reg::<CnaCvtCon2>(cmds, |_, r| {
        r.cvt_scale1(Bits::new(1)).cvt_offset1(Bits::new(0));
    });
    replace_reg::<CnaCvtCon3>(cmds, |_, r| {
        r.cvt_scale2(Bits::new(1)).cvt_offset2(Bits::new(0));
    });
    replace_reg::<CnaCvtCon4>(cmds, |_, r| {
        r.cvt_scale3(Bits::new(1)).cvt_offset3(Bits::new(0));
    });
    replace_reg::<CnaCvtCon5>(cmds, |_, r| {
        r.per_channel_cvt_en(Bits::new(0));
    });
    replace_reg::<CnaPadCon1>(cmds, |_, r| {
        r.pad_value(Bits::new(0));
    });
    replace_reg::<CoreMiscCfg>(cmds, |_, r| {
        r.qd_en(Bits::new(0));
    });
    replace_reg::<DpuOutCvtScale>(cmds, |_, r| {
        r.out_cvt_scale(Bits::new(1));
    });
    // Round 6, same technique, one more diff pass after round 5's patches
    // came back with the identical NaN: with every CVT/precision/OUT_CVT
    // register now bit-exact against real ground truth, `DPU_BS_OW_CFG`
    // (0x4050, bit 1 `od_bypass`, "if bypass CPEND") and `DPU_BS_OW_OP`
    // (0x4054, the CPEND operand) were the next -- and only remaining --
    // registers that still differed for a non-geometry reason: real has
    // `od_bypass=1`/`ow_op=0` (CPEND stage cleanly bypassed), while
    // `build_conv_regcmd`'s conv path (`regcmd.rs:993-1007`) never sets
    // `od_bypass` and unconditionally computes
    // `ow_op = 0x80 - weights_zero_point` -- the same 128 zero-point
    // convention from this whole investigation, now leaking into a bias
    // stage a real fp16-shaped dispatch turns off entirely. A sibling
    // function elsewhere in `regcmd.rs` (line ~1360) already knows this
    // exact bypass pattern (`od_bypass(1)` + zeroed `DpuBsOwOp`) for a
    // different op path -- this just wasn't applied here.
    replace_reg::<DpuBsOwCfg>(cmds, |_, r| {
        r.od_bypass(Bits::new(1));
    });
    replace_reg::<DpuBsOwOp>(cmds, |_, r| {
        r.ow_op(Bits::new(0));
    });
}

/// New hypothesis (round 4): `DPU_WDMA_SIZE_0.tp_precision` (0x4058, bit
/// 27, "Transpose precision: 0=8bit, 1=16bit") sits on the OUTPUT write-
/// DMA register -- the block that actually writes DPU's post-OUT_CVT
/// result out to memory -- and is completely separate from every
/// precision field the prior 3 rounds patched (those were all upstream
/// of OUT_CVT: CNA/CORE/DPU/DPU_RDMA input-side precision enums, CVT
/// bypass, fp32tofp16_en). A setter already existed (bindgen) but was
/// never called anywhere; the register's default/current value leaves
/// this bit 0 (8-bit), which would make WDMA write the fp16 OUT_CVT
/// result out using an 8-bit-wide packing convention -- a very plausible
/// source of a corrupted/NaN-looking readback even if everything upstream
/// (including the accumulator) were already numerically correct.
///
/// Ran on hardware (2026-07-22): bit-exact identical `0x7c01` NaN, no
/// change -- ruled out same as the weight-bytes hypothesis below. Kept
/// (unused) for the historical record: real `conv.rknn` ground truth
/// (see `patch_from_conv_rknn_ground_truth`) independently confirms
/// `tp_precision` stays 0 in a genuine non-int8 dispatch, so this
/// hypothesis was wrong on top of producing no hardware effect.
#[allow(dead_code)]
fn patch_wdma_precision_for_fp16(cmds: &mut [RegCmd]) {
    replace_reg::<DpuWdmaSize0>(cmds, |_, r| {
        r.tp_precision(Bits::new(1));
    });
}

/// New hypothesis, not part of the original 3-round investigation (see
/// rknpu-spelunking/NOTES.md's fp16 section): `CNA_WEIGHT_SIZE0.
/// WEIGHT_BYTES` and `CNA_WEIGHT_SIZE1.WEIGHT_BYTES_PER_KERNEL` are
/// computed by build_conv_regcmd as a pure element-count formula
/// (`weights_width * weights_height * channels * kernels`) with no
/// bytes-per-element factor at all -- i.e. hardcoded bpe=1. Every other
/// dtype-sensitive register in the original patch list was addressed
/// (precision enums, CVT bypass/sign, OUT_CVT), but these two byte-count
/// fields were flagged in NOTES.md as "not patched or ruled out" and
/// never touched in any of the 3 prior rounds. For fp16's 2-bytes/element
/// weight layout, CNA would be told to fetch half the real weight byte
/// count -- doubling both fields here to match. The mask constants are
/// still needed here (unlike everywhere else in this file) because
/// there's no builder "getter" to read `build_conv_regcmd`'s own computed
/// value back out before doubling it -- only the write side goes through
/// the typed setter.
///
/// Ran on hardware (2026-07-22): bit-exact identical `0x7c01` NaN, no
/// change -- ruled out.
fn patch_weight_bytes_for_fp16(cmds: &mut [RegCmd]) {
    replace_reg::<CnaWeightSize0>(cmds, |orig, r| {
        let doubled = extract_field(orig, CNA_WEIGHT_SIZE0_WEIGHT_BYTES__MASK) * 2;
        r.weight_bytes(Bits::new(doubled));
    });
    replace_reg::<CnaWeightSize1>(cmds, |orig, r| {
        let doubled = extract_field(orig, CNA_WEIGHT_SIZE1_WEIGHT_BYTES_PER_KERNEL__MASK) * 2;
        r.weight_bytes_per_kernel(Bits::new(doubled));
    });
}

fn extract_field(val: u32, mask: u32) -> u32 {
    (val & mask) >> mask.trailing_zeros()
}

/// Minimal IEEE-754 binary16 -> f32 decode. Handles the normal/zero/inf/nan
/// cases correctly (exp==0x1f is inf/nan, not a huge finite exponent --
/// the first version of this diagnostic got this wrong and silently
/// printed a bogus finite number for a NaN bit pattern). Subnormals are
/// not handled (flushed to zero) -- not needed for this diagnostic's
/// inputs.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 0x1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    let f32_bits = if exp == 0 {
        sign << 31
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

fn main() {
    let input_fill_f32: f32 = std::env::args()
        .nth(1)
        .map(|s| s.parse().expect("input fill must be a float"))
        .unwrap_or(2.5);
    let input_fill = f32_to_f16_bits(input_fill_f32);
    let weight_fill = f32_to_f16_bits(0.25);

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/accel/accel0")
        .expect("Failed to open device");
    let fd = file.as_raw_fd();

    println!(
        "--- NPU Diagnostic: fp16 dispatch, TRM-informed (CNA+CORE+DPU+DPU_RDMA precision, CVT bypass, fp32tofp16_en) ---"
    );
    println!(
        "input_fill = {input_fill_f32} (f16 bits 0x{input_fill:04x}), weight_fill = 2.0 (f16 bits 0x{weight_fill:04x})"
    );

    let tensor_size = 4096; // 4KB aligned

    unsafe {
        let buf_a = Buffer::new(fd, tensor_size, &file);
        fill_u16(buf_a.host_ptr, tensor_size, input_fill);

        let buf_w = Buffer::new(fd, tensor_size, &file);
        fill_u16(buf_w.host_ptr, tensor_size, weight_fill);

        let buf_c = Buffer::new(fd, tensor_size, &file);
        ptr::write_bytes(buf_c.host_ptr, 0, tensor_size);

        let buf_bias = Buffer::new(fd, tensor_size, &file);
        ptr::write_bytes(buf_bias.host_ptr, 0, tensor_size);

        println!(
            "Buffers: A@0x{:x}, W@0x{:x}, Bias@0x{:x}, C@0x{:x}",
            buf_a.dma_address, buf_w.dma_address, buf_bias.dma_address, buf_c.dma_address
        );

        let shape = ConvShape {
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
        };
        let bufs = ConvBuffers {
            input_addr: buf_a.dma_address,
            weights_addr: buf_w.dma_address,
            bias_addr: buf_bias.dma_address,
            output_addr: buf_c.dma_address,
        };
        let mut cmds = build_conv_regcmd(&shape, &bufs);
        patch_for_fp16(&mut cmds);
        patch_weight_bytes_for_fp16(&mut cmds);
        patch_from_conv_rknn_ground_truth(&mut cmds);

        dump_cmds("rkt-fp16", &cmds);

        let cmd_bytes = cmds.len() * mem::size_of::<u64>();
        let cmd_len = cmd_bytes.next_multiple_of(4096);
        let buf_cmd = Buffer::new(fd, cmd_len, &file);
        let cmd_slice = std::slice::from_raw_parts_mut(buf_cmd.host_ptr as *mut u64, cmds.len());
        for (i, c) in cmds.iter().enumerate() {
            cmd_slice[i] = c.0;
        }

        rocket_fini_bo(
            fd,
            &drm_rocket_fini_bo {
                handle: buf_a.handle,
                reserved: 0,
            },
        )
        .ok();
        rocket_fini_bo(
            fd,
            &drm_rocket_fini_bo {
                handle: buf_w.handle,
                reserved: 0,
            },
        )
        .ok();
        rocket_fini_bo(
            fd,
            &drm_rocket_fini_bo {
                handle: buf_bias.handle,
                reserved: 0,
            },
        )
        .ok();
        rocket_fini_bo(
            fd,
            &drm_rocket_fini_bo {
                handle: buf_c.handle,
                reserved: 0,
            },
        )
        .ok();
        rocket_fini_bo(
            fd,
            &drm_rocket_fini_bo {
                handle: buf_cmd.handle,
                reserved: 0,
            },
        )
        .ok();

        let task = drm_rocket_task {
            regcmd: buf_cmd.dma_address,
            regcmd_count: cmds.len() as u32,
        };

        let in_handles = vec![buf_cmd.handle, buf_a.handle, buf_w.handle, buf_bias.handle];
        let out_handles = vec![buf_c.handle];

        let job = drm_rocket_job {
            tasks: &task as *const _ as u64,
            in_bo_handles: in_handles.as_ptr() as u64,
            out_bo_handles: out_handles.as_ptr() as u64,
            task_count: 1,
            task_struct_size: mem::size_of::<drm_rocket_task>() as u32,
            in_bo_handle_count: in_handles.len() as u32,
            out_bo_handle_count: out_handles.len() as u32,
        };

        let mut submit = drm_rocket_submit {
            jobs: &job as *const _ as u64,
            job_count: 1,
            job_struct_size: mem::size_of::<drm_rocket_job>() as u32,
            reserved: 0,
        };

        println!("Submitting...");
        rocket_submit(fd, &mut submit).expect("IOCTL Failed");

        let now = clock_gettime(ClockId::CLOCK_MONOTONIC).expect("clock_gettime failed");
        let now_ns = now.tv_sec() as u64 * 1_000_000_000 + now.tv_nsec() as u64;
        let prep = drm_rocket_prep_bo {
            handle: buf_c.handle,
            reserved: 0,
            timeout_ns: (now_ns + 2_000_000_000) as i64,
        };

        match rocket_prep_bo(fd, &prep) {
            Ok(_) => {
                // Pixels aren't tightly packed -- each occupies its own
                // 16-byte-aligned atomic slot (see module doc comment).
                let raw = std::slice::from_raw_parts(buf_c.host_ptr, 256);
                println!("Done! First 256 output bytes, 32 per row:");
                for (row, chunk) in raw.chunks(32).enumerate() {
                    println!("  [{:4}..{:4}) {:?}", row * 32, row * 32 + 32, chunk);
                }
                let pixels_f16: Vec<u16> = (0..16)
                    .map(|i| u16::from_le_bytes([raw[i * 16], raw[i * 16 + 1]]))
                    .collect();
                let pixels_f32: Vec<f32> = pixels_f16.iter().map(|&b| f16_to_f32(b)).collect();
                println!(
                    "16 output pixels (stride 16) as f16 bits: {:04x?}",
                    pixels_f16
                );
                println!("16 output pixels decoded as f32: {:?}", pixels_f32);
            }
            Err(e) => {
                println!("TIMEOUT/error: {}", e);
            }
        }
    }
}

unsafe fn fill_u16(ptr: *mut u8, byte_len: usize, val: u16) {
    let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u16, byte_len / 2) };
    slice.fill(val);
}

ioctl_readwrite!(
    rocket_create_bo,
    DRM_IOCTL_BASE,
    DRM_COMMAND_BASE + DRM_ROCKET_CREATE_BO,
    drm_rocket_create_bo
);
ioctl_readwrite!(
    rocket_submit,
    DRM_IOCTL_BASE,
    DRM_COMMAND_BASE + DRM_ROCKET_SUBMIT,
    drm_rocket_submit
);
ioctl_write_ptr!(
    rocket_prep_bo,
    DRM_IOCTL_BASE,
    DRM_COMMAND_BASE + DRM_ROCKET_PREP_BO,
    drm_rocket_prep_bo
);
ioctl_write_ptr!(
    rocket_fini_bo,
    DRM_IOCTL_BASE,
    DRM_COMMAND_BASE + DRM_ROCKET_FINI_BO,
    drm_rocket_fini_bo
);

struct Buffer {
    handle: u32,
    dma_address: u32,
    size: usize,
    host_ptr: *mut u8,
}

impl Buffer {
    unsafe fn new(fd: i32, size: usize, file: &std::fs::File) -> Self {
        let mut create_params = drm_rocket_create_bo {
            size: size as u32,
            handle: 0,
            dma_address: 0,
            offset: 0,
        };
        rocket_create_bo(fd, &mut create_params).expect("Failed to create BO");
        let map_len = NonZeroUsize::new(size).unwrap();
        let map_addr = mmap(
            None,
            map_len,
            ProtFlags::PROT_READ | ProtFlags::PROT_WRITE,
            MapFlags::MAP_SHARED,
            file,
            create_params.offset as i64,
        )
        .expect("mmap failed");
        Buffer {
            handle: create_params.handle,
            dma_address: create_params.dma_address as u32,
            size,
            host_ptr: map_addr.as_ptr() as *mut u8,
        }
    }
}
