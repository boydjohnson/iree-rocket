//! NPU diagnostic: real fp16 dispatch, now via `build_conv_regcmd`'s
//! native `Precision::Fp16` path -- see `Precision`'s doc comment in
//! `regcmd.rs` and rknpu-spelunking/NOTES.md's fp16 section for the full
//! derivation history.
//!
//! This file used to be the derivation vehicle itself: 7 rounds of
//! post-processing patches (`patch_field`/`replace_reg` rewriting
//! `build_conv_regcmd`'s int8-shaped output register-by-register) before
//! landing on a real, hardware-confirmed bit-exact result (`10.5 * 0.25
//! = 2.625`, exactly representable in fp16, no rounding to hand-wave
//! away). That whole recipe has since been ported into
//! `build_conv_cna_core_dpu_dpu_rdma` itself, parameterized by
//! `ConvShape::precision` -- this file is now just a plain client of
//! `build_conv_regcmd(&shape, &bufs)` with `precision: Precision::Fp16`,
//! same as every other hw diagnostic in this repo. Kept as a standalone
//! `src/bin` (not folded into `tests/conv_hw.rs`) since it still owns
//! its own f16<->f32 encode/decode helpers for command-line input.
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
    regcmd::{Activation, ConvBuffers, ConvShape, Precision, build_conv_regcmd},
};
use nix::{
    ioctl_readwrite, ioctl_write_ptr,
    sys::mman::{MapFlags, ProtFlags, mmap},
    time::{ClockId, clock_gettime},
};

/// Minimal IEEE-754 binary16 -> f32 decode. Handles the normal/zero/inf/nan
/// cases correctly (exp==0x1f is inf/nan, not a huge finite exponent --
/// the first version of this diagnostic got this wrong and silently
/// printed a bogus finite number for a NaN bit pattern). Subnormals ARE
/// decoded (round 6's intermediate result, before round 7's fix, landed
/// on a real subnormal value -- flushing those to zero would have hidden
/// that the NaN was actually gone).
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 0x1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    let f32_bits = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            // Subnormal: value = frac * 2^-24 (sign-adjusted). Normalize
            // by hand into an f32 bit pattern rather than special-casing
            // every caller.
            let subnormal = (frac as f32) * 2f32.powi(-24);
            let signed = if sign == 1 { -subnormal } else { subnormal };
            return signed;
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

fn main() {
    let input_fill_f32: f32 = std::env::args()
        .nth(1)
        .map(|s| s.parse().expect("input fill must be a float"))
        .unwrap_or(10.5);
    let weight_fill_f32: f32 = std::env::args()
        .nth(2)
        .map(|s| s.parse().expect("weight fill must be a float"))
        .unwrap_or(0.25);
    let input_fill = f32_to_f16_bits(input_fill_f32);
    let weight_fill = f32_to_f16_bits(weight_fill_f32);

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/accel/accel0")
        .expect("Failed to open device");
    let fd = file.as_raw_fd();

    println!(
        "--- NPU Diagnostic: fp16 dispatch via build_conv_regcmd's native Precision::Fp16 path ---"
    );
    println!(
        "input_fill = {input_fill_f32} (f16 bits 0x{input_fill:04x}), weight_fill = {weight_fill_f32} (f16 bits 0x{weight_fill:04x})"
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
            precision: Precision::Fp16,
        };
        let bufs = ConvBuffers {
            input_addr: buf_a.dma_address,
            weights_addr: buf_w.dma_address,
            bias_addr: buf_bias.dma_address,
            output_addr: buf_c.dma_address,
        };
        let cmds = build_conv_regcmd(&shape, &bufs);

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
                println!(
                    "expected (input_fill * weight_fill): {}",
                    input_fill_f32 * weight_fill_f32
                );
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
