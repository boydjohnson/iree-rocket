//! NPU diagnostic: 1x1 conv, 64x64 spatial, 64 input channels, 64 output
//! channels, stride 1, no padding, no depthwise. Regcmd construction lives
//! in `iree_rocket_hal::rocket::regcmd` (shared with rkt-basic.rs/
//! rkt-simple-job.rs) -- this file previously carried its own independent
//! (and substantially incomplete/wrong -- no preamble writes, no
//! DCOMP_AMOUNT0-15, per-block OPERATION_ENABLE writes instead of the
//! real single broadcast kick, etc.) regcmd builder. See that module's
//! doc comment and rknpu-spelunking/NOTES.md for the full derivation/
//! validation history.

use std::{fs::OpenOptions, mem, num::NonZeroUsize, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    api::{
        DRM_COMMAND_BASE, DRM_IOCTL_BASE, DRM_ROCKET_CREATE_BO, DRM_ROCKET_FINI_BO,
        DRM_ROCKET_PREP_BO, DRM_ROCKET_SUBMIT, drm_rocket_create_bo, drm_rocket_fini_bo,
        drm_rocket_job, drm_rocket_prep_bo, drm_rocket_submit, drm_rocket_task,
    },
    debug::dump_cmds,
    regcmd::{ConvBuffers, ConvShape, build_conv_regcmd},
};
use nix::{
    ioctl_readwrite,
    sys::mman::{MapFlags, ProtFlags, mmap},
    time::{ClockId, clock_gettime},
};

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
ioctl_readwrite!(
    rocket_prep_bo,
    DRM_IOCTL_BASE,
    DRM_COMMAND_BASE + DRM_ROCKET_PREP_BO,
    drm_rocket_prep_bo
);
ioctl_readwrite!(
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
        unsafe {
            let mut create_params = drm_rocket_create_bo {
                size: size as u32,
                handle: 0,
                dma_address: 0,
                offset: 0,
            };

            rocket_create_bo(fd, &mut create_params).expect("Failed to create BO");

            if create_params.dma_address > u32::MAX as u64 {
                panic!("Driver returned >32-bit DMA address!");
            }

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
}

fn main() {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/accel/accel0")
        .expect("Failed to open device");
    let fd = file.as_raw_fd();

    println!("--- NPU Diagnostic: 64x64x64 1x1 Conv (shared regcmd builder) ---");

    // Input = 10, Weight = 2 everywhere. Note: unlike the naive "raw
    // multiply" expectation this file used to check against (20), the
    // real datapath centers input around a 128 zero-point before the MAC
    // (see rkt-basic.rs's NOTES.md sanity-check sweep) -- the correct
    // expected value isn't simply input*weight. Not asserting a specific
    // expected value here for that reason; use rkt-basic.rs's input-fill
    // CLI arg sweep if you want to sanity-check correctness on a shape
    // that's cheap to sweep.
    let tensor_size = 64 * 64 * 64;

    unsafe {
        let buf_a = Buffer::new(fd, tensor_size, &file);
        ptr::write_bytes(buf_a.host_ptr, 10, tensor_size); // Input

        let buf_w = Buffer::new(fd, tensor_size, &file);
        ptr::write_bytes(buf_w.host_ptr, 2, tensor_size); // Weight -- whole buffer, not just a prefix:
        // WEIGHT_SIZE0 for this shape is weights_w*h * aligned_input_channels(64) *
        // weights_kernels(64) = 4096 bytes; the previous revision only filled the
        // first 64 bytes (`weight_tensor_size = 64`), leaving the rest of what CNA
        // actually reads as uninitialized garbage.

        let buf_c = Buffer::new(fd, tensor_size, &file);
        ptr::write_bytes(buf_c.host_ptr, 0, tensor_size); // Output

        // DPU's BS (bias-subtract) block is never actually bypassed by
        // Mesa's real driver -- it always runs its ALU against a real
        // biases buffer, so this has to exist and be zero-filled even
        // though this op has no logical bias.
        let buf_bias = Buffer::new(fd, tensor_size, &file);
        ptr::write_bytes(buf_bias.host_ptr, 0, tensor_size);

        println!(
            "Buffers: A@{:#x}, W@{:#x}, Bias@{:#x}, C@{:#x}",
            buf_a.dma_address, buf_w.dma_address, buf_bias.dma_address, buf_c.dma_address
        );

        let shape = ConvShape {
            input_width: 64,
            input_height: 64,
            input_channels: 64,
            output_width: 64,
            output_height: 64,
            output_channels: 64,
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
        };
        let bufs = ConvBuffers {
            input_addr: buf_a.dma_address,
            weights_addr: buf_w.dma_address,
            bias_addr: buf_bias.dma_address,
            output_addr: buf_c.dma_address,
        };
        let cmds = build_conv_regcmd(&shape, &bufs);
        dump_cmds("rkt-job", &cmds);

        let buf_cmd = Buffer::new(fd, 4096, &file);
        let cmd_slice = std::slice::from_raw_parts_mut(buf_cmd.host_ptr as *mut u64, cmds.len());
        for (i, c) in cmds.iter().enumerate() {
            cmd_slice[i] = c.0;
        }
        let mut fini_bo_a = drm_rocket_fini_bo {
            reserved: 0,
            handle: buf_a.handle,
        };
        let mut fini_bo_w = drm_rocket_fini_bo {
            reserved: 0,
            handle: buf_w.handle,
        };
        let mut fini_bo_bias = drm_rocket_fini_bo {
            reserved: 0,
            handle: buf_bias.handle,
        };
        let mut fini_bo_c = drm_rocket_fini_bo {
            reserved: 0,
            handle: buf_c.handle,
        };
        let mut fini_bo_cmd = drm_rocket_fini_bo {
            reserved: 0,
            handle: buf_cmd.handle,
        };

        rocket_fini_bo(fd, &mut fini_bo_a).expect("Flush A");
        rocket_fini_bo(fd, &mut fini_bo_w).expect("Flush W");
        rocket_fini_bo(fd, &mut fini_bo_bias).expect("Flush Bias");
        rocket_fini_bo(fd, &mut fini_bo_c).expect("Flush C");
        rocket_fini_bo(fd, &mut fini_bo_cmd).expect("Flush Cmd");

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
        rocket_submit(fd, &mut submit).expect("Submit failed");

        // rocket_gem.c's rocket_ioctl_prep_bo() converts timeout_ns via
        // drm_timeout_abs_to_jiffies() -- that takes an ABSOLUTE
        // CLOCK_MONOTONIC deadline, not a relative duration (standard DRM
        // wait-ioctl convention; see rkt-basic.rs / NOTES.md for the full
        // story).
        let now = clock_gettime(ClockId::CLOCK_MONOTONIC).expect("clock_gettime failed");
        let now_ns = now.tv_sec() as u64 * 1_000_000_000 + now.tv_nsec() as u64;
        let mut prep = drm_rocket_prep_bo {
            handle: buf_c.handle,
            reserved: 0,
            timeout_ns: (now_ns + 2_000_000_000) as i64, // 2s from now, as an absolute deadline
        };
        rocket_prep_bo(fd, &mut prep).expect("Wait failed");

        let result = *buf_c.host_ptr;
        println!("Result: {}", result);
    }
}
