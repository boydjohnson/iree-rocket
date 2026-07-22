//! NPU diagnostic: minimal 1x1 conv, 4x4 spatial, 1 input channel, 1 output
//! channel, stride 1, no padding, no depthwise. Regcmd construction lives
//! in `iree_rocket_hal::rocket::regcmd` (shared with rkt-job.rs/
//! rkt-simple-job.rs) -- see that module's doc comment and
//! rknpu-spelunking/NOTES.md for the full derivation/validation history
//! (a faithful field-for-field port of Mesa's real gallium driver,
//! `mesa-rocket-userspace/rkt_regcmd.c`'s `fill_first_regcmd()`).
//!
//! Also fixes a second, independent bug this file had: `PREP_BO`'s
//! `timeout_ns` needs to be an absolute `CLOCK_MONOTONIC` deadline
//! (`drm_timeout_abs_to_jiffies()` in `rocket_gem.c`), not a relative
//! duration -- a bare literal here always computed as already-expired,
//! producing `EBUSY` regardless of whether the job actually completed.

use std::{fs::OpenOptions, mem, num::NonZeroUsize, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    api::{
        DRM_COMMAND_BASE, DRM_IOCTL_BASE, DRM_ROCKET_CREATE_BO, DRM_ROCKET_FINI_BO,
        DRM_ROCKET_PREP_BO, DRM_ROCKET_SUBMIT, drm_rocket_create_bo, drm_rocket_fini_bo,
        drm_rocket_job, drm_rocket_prep_bo, drm_rocket_submit, drm_rocket_task,
    },
    debug::dump_cmds,
    regcmd::{Activation, ConvBuffers, ConvShape, build_conv_regcmd},
};
use nix::{
    ioctl_readwrite, ioctl_write_ptr,
    sys::mman::{MapFlags, ProtFlags, mmap},
    time::{ClockId, clock_gettime},
};

fn main() {
    // Sanity-check knob: input fill byte, default 10. The CNA CVT stage
    // subtracts an implicit 128 zero-point before the MAC, so values far
    // from 128 in either direction saturate the int8 output range --
    // sweeping values close to 128 is what actually distinguishes "real,
    // proportional MAC computation" from "hollow completion that just
    // saturates to a rail regardless of input" (see NOTES.md's sanity-
    // check sweep: 118-128 -> 128, 129 -> 2, 130-138 -> 127).
    let input_fill: u8 = std::env::args()
        .nth(1)
        .map(|s| s.parse().expect("input fill byte must be 0-255"))
        .unwrap_or(10);

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/accel/accel0")
        .expect("Failed to open device");
    let fd = file.as_raw_fd();

    println!("--- NPU Diagnostic: Minimal 1x1 Conv (shared regcmd builder) ---");
    println!("input_fill = {input_fill}");

    let tensor_size = 4096; // 4KB aligned

    unsafe {
        let buf_a = Buffer::new(fd, tensor_size, &file);
        ptr::write_bytes(buf_a.host_ptr, input_fill, tensor_size);

        let buf_w = Buffer::new(fd, tensor_size, &file);
        ptr::write_bytes(buf_w.host_ptr, 2, tensor_size);

        let buf_c = Buffer::new(fd, tensor_size, &file);
        ptr::write_bytes(buf_c.host_ptr, 0, tensor_size);

        // DPU's BS (bias-subtract) block is never actually bypassed by
        // Mesa's real driver -- it always runs its ALU against a real
        // biases buffer, so this has to exist and be zero-filled even
        // though this op has no logical bias.
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
        let cmds = build_conv_regcmd(&shape, &bufs);

        // Printed to stderr before touching hardware, so it's visible even
        // if SUBMIT/PREP_BO hangs afterward.
        dump_cmds("rkt-basic", &cmds);

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

        // Not doubled -- Mesa's rkt_ml_subgraph_invoke() sets
        // ktask->regcmd_count = task->regcfg_amount, the raw entry count
        // (cmds.len()), no `* 2`. See NOTES.md for the full story.
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
            timeout_ns: (now_ns + 2_000_000_000) as i64, // 2s from now, as an absolute deadline
        };

        match rocket_prep_bo(fd, &prep) {
            Ok(_) => {
                let res = *buf_c.host_ptr;
                println!("Done! Result[0]: {}", res);
            }
            Err(e) => {
                println!("TIMEOUT/error: {}", e);
            }
        }
    }
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
