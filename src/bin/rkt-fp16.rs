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
    debug::dump_cmds,
    regcmd::{Activation, ConvBuffers, ConvShape, build_conv_regcmd},
    registers::{
        CNA_CVT_CON0_CVT_BYPASS, CNA_CVT_CON0_CVT_BYPASS__MASK, CNA_CVT_CON0_CVT_TYPE,
        CNA_CVT_CON0_CVT_TYPE__MASK, CNA_CVT_CON0_DATA_SIGN, CNA_CVT_CON0_DATA_SIGN__MASK,
        CORE_MISC_CFG_PROC_PRECISION, CORE_MISC_CFG_PROC_PRECISION__MASK,
        DPU_DATA_FORMAT_IN_PRECISION, DPU_DATA_FORMAT_IN_PRECISION__MASK,
        DPU_DATA_FORMAT_OUT_PRECISION, DPU_DATA_FORMAT_OUT_PRECISION__MASK,
        DPU_DATA_FORMAT_PROC_PRECISION, DPU_DATA_FORMAT_PROC_PRECISION__MASK,
        DPU_OUT_CVT_OFFSET_OUT_CVT_OFFSET__MASK, DPU_OUT_CVT_SCALE_FP32TOFP16_EN,
        DPU_OUT_CVT_SCALE_FP32TOFP16_EN__MASK, DPU_OUT_CVT_SCALE_OUT_CVT_SCALE__MASK,
        DPU_OUT_CVT_SHIFT_CVT_ROUND__MASK, DPU_OUT_CVT_SHIFT_CVT_TYPE__MASK,
        DPU_OUT_CVT_SHIFT_MINUS_EXP__MASK, DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT__MASK,
        DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION,
        DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION__MASK,
        DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION,
        DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION__MASK, REG_CNA_CONV_CON1, REG_CNA_CVT_CON0,
        REG_CORE_MISC_CFG, REG_DPU_DATA_FORMAT, REG_DPU_OUT_CVT_OFFSET, REG_DPU_OUT_CVT_SCALE,
        REG_DPU_OUT_CVT_SHIFT, REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG,
    },
};
use nix::{
    ioctl_readwrite, ioctl_write_ptr,
    sys::mman::{MapFlags, ProtFlags, mmap},
    time::{ClockId, clock_gettime},
};

/// ORs `shifted_value & mask` into every regcmd entry at `offset` (there
/// can be more than one -- Mesa/`build_conv_regcmd` write CNA_CONV_CON1
/// twice), after clearing `mask`'s bits first. Domain and offset are left
/// untouched.
fn patch_field(
    cmds: &mut [iree_rocket_hal::rocket::builders::RegCmd],
    offset: u64,
    mask: u32,
    shifted_value: u32,
) {
    for cmd in cmds.iter_mut() {
        if cmd.0 & 0xFFFF == offset {
            let domain = cmd.0 >> 48;
            let mut val = ((cmd.0 >> 16) & 0xFFFF_FFFF) as u32;
            val = (val & !mask) | (shifted_value & mask);
            cmd.0 = (domain << 48) | ((val as u64) << 16) | offset;
        }
    }
}

/// All six patches described in the module doc comment, precision fields
/// all set to `2` (fp16 in both CNA/CORE's 8-value enum and DPU's
/// 6-value enum).
fn patch_for_fp16(cmds: &mut [iree_rocket_hal::rocket::builders::RegCmd]) {
    unsafe {
        patch_field(
            cmds,
            REG_CNA_CONV_CON1 as u64,
            iree_rocket_hal::rocket::registers::CNA_CONV_CON1_IN_PRECISION__MASK
                | iree_rocket_hal::rocket::registers::CNA_CONV_CON1_PROC_PRECISION__MASK,
            iree_rocket_hal::rocket::registers::CNA_CONV_CON1_IN_PRECISION(2)
                | iree_rocket_hal::rocket::registers::CNA_CONV_CON1_PROC_PRECISION(2),
        );
        patch_field(
            cmds,
            REG_CORE_MISC_CFG as u64,
            CORE_MISC_CFG_PROC_PRECISION__MASK,
            CORE_MISC_CFG_PROC_PRECISION(2),
        );
        patch_field(
            cmds,
            REG_DPU_DATA_FORMAT as u64,
            DPU_DATA_FORMAT_IN_PRECISION__MASK
                | DPU_DATA_FORMAT_OUT_PRECISION__MASK
                | DPU_DATA_FORMAT_PROC_PRECISION__MASK,
            DPU_DATA_FORMAT_IN_PRECISION(2)
                | DPU_DATA_FORMAT_OUT_PRECISION(2)
                | DPU_DATA_FORMAT_PROC_PRECISION(2),
        );
        patch_field(
            cmds,
            REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG as u64,
            DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION__MASK
                | DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION__MASK,
            DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION(2)
                | DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION(2),
        );
        patch_field(
            cmds,
            REG_CNA_CVT_CON0 as u64,
            CNA_CVT_CON0_CVT_BYPASS__MASK,
            CNA_CVT_CON0_CVT_BYPASS(1),
        );
        // build_conv_regcmd's `input_channels_real_is_one` branch (our
        // shape) never sets data_sign/cvt_type -- they stay 0 (unsigned /
        // mul-then-add). The *other* (multi-channel) branch explicitly
        // sets both to 1 alongside its own cvt_bypass. Matching that
        // convention here in case data_sign=0 (unsigned) is quietly
        // corrupting fp16's own sign bit somewhere downstream, even with
        // the stage nominally bypassed.
        patch_field(
            cmds,
            REG_CNA_CVT_CON0 as u64,
            CNA_CVT_CON0_DATA_SIGN__MASK,
            CNA_CVT_CON0_DATA_SIGN(1),
        );
        patch_field(
            cmds,
            REG_CNA_CVT_CON0 as u64,
            CNA_CVT_CON0_CVT_TYPE__MASK,
            CNA_CVT_CON0_CVT_TYPE(1),
        );
        patch_field(
            cmds,
            REG_DPU_OUT_CVT_SCALE as u64,
            DPU_OUT_CVT_SCALE_FP32TOFP16_EN__MASK,
            DPU_OUT_CVT_SCALE_FP32TOFP16_EN(1),
        );

        // No documented bypass bit exists for the DPU_OUT_CVT block (TRM
        // 0x4080-0x4088) -- build_conv_regcmd's offset/scale/shift values
        // are computed via an int8-specific fixed-point requantization
        // formula (bit-tricking a float32's mantissa/exponent), which is
        // very likely why the first fp16 attempt (fp32tofp16_en set, but
        // these three left at their int8-formula values) came back as
        // NaN (0x7c01) instead of a real number. Neutralize all three to
        // 0 -- the closest thing to "off" available without a real bypass
        // bit -- while leaving fp32tofp16_en set. Unconfirmed whether
        // scale=0 is actually a no-op or itself degenerate (e.g.
        // "multiply by 0") -- this is a genuine guess, not a documented
        // identity configuration.
        patch_field(
            cmds,
            REG_DPU_OUT_CVT_OFFSET as u64,
            DPU_OUT_CVT_OFFSET_OUT_CVT_OFFSET__MASK,
            0,
        );
        patch_field(
            cmds,
            REG_DPU_OUT_CVT_SCALE as u64,
            DPU_OUT_CVT_SCALE_OUT_CVT_SCALE__MASK,
            0,
        );
        patch_field(
            cmds,
            REG_DPU_OUT_CVT_SHIFT as u64,
            DPU_OUT_CVT_SHIFT_CVT_TYPE__MASK
                | DPU_OUT_CVT_SHIFT_CVT_ROUND__MASK
                | DPU_OUT_CVT_SHIFT_MINUS_EXP__MASK
                | DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT__MASK,
            0,
        );
    }
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
        .unwrap_or(3.0);
    let input_fill = f32_to_f16_bits(input_fill_f32);
    let weight_fill = f32_to_f16_bits(2.0);

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
