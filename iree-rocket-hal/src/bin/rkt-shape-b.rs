//! NPU diagnostic: shape generalization test B for `build_conv_regcmd` --
//! a genuine 3x3 kernel (rkt-basic.rs only ever exercised 1x1), still
//! single input channel so weight-buffer layout ambiguity doesn't matter
//! (see below), stride 1, no padding: 6x6x1 input -> 4x4x1 output.
//!
//! Exercises real multi-tap accumulation and the `weights_width >= 3`
//! branch of `pad_con1` in regcmd.rs (never hit by rkt-basic.rs's 1x1
//! kernel), plus CBUF/weight-bank geometry for a 9-tap kernel instead of a
//! single tap.
//!
//! Verification strategy, mirroring rkt-basic.rs's original zero-point
//! sweep exactly (see rknpu-spelunking/NOTES.md): fill the *entire* input
//! plane with one uniform byte value and the *entire* weight buffer with
//! one uniform byte value (2, matching rkt-basic.rs's known-good
//! magnitude) -- same trick rkt-basic.rs itself uses to sidestep not
//! knowing the real per-tap weight-buffer packing order (`rkt_coefs.c`,
//! where Mesa would implement that, isn't available in this checkout, only
//! its header). With a uniform input and uniform weights, every one of the
//! 9 taps contributes identically regardless of what order the hardware
//! actually reads them in, and every one of the 16 output pixels should
//! come out identical (all windows see the same uniform input). Sweeping
//! the fill byte near the CVT stage's 128 zero-point and checking that (a)
//! all 16 output pixels agree with each other and (b) the sequence is
//! non-constant/saturates on both sides is the same "real computation, not
//! a hollow completion" signal rkt-basic.rs's single-byte version used --
//! just now covering a real 2D window and all output positions instead of
//! one.
//!
//! Reads output at stride 16 bytes/pixel, not stride 1 -- output_channels=1
//! pads to task_output_channels=32, and the hardware writes each pixel as
//! a full 16-byte-aligned atomic slot regardless of real channel count.
//! (First version of this test read stride 1 and appeared to show only
//! pixel [0,0] getting written; rkt-shape-a3.rs proved that was a test
//! bug, not a real one -- see rknpu-spelunking/NOTES.md's "RESOLVED: the
//! only pixel [0,0] written mystery" section.)

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

fn main() {
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

    println!("--- NPU Diagnostic: shape B, 3x3 kernel, 1 input channel (6x6 -> 4x4) ---");
    println!("input_fill = {input_fill}");

    let tensor_size = 4096;

    unsafe {
        let buf_a = Buffer::new(fd, tensor_size, &file);
        ptr::write_bytes(buf_a.host_ptr, input_fill, tensor_size);

        let buf_w = Buffer::new(fd, tensor_size, &file);
        ptr::write_bytes(buf_w.host_ptr, 2, tensor_size);

        let buf_c = Buffer::new(fd, tensor_size, &file);
        ptr::write_bytes(buf_c.host_ptr, 0, tensor_size);

        let buf_bias = Buffer::new(fd, tensor_size, &file);
        ptr::write_bytes(buf_bias.host_ptr, 0, tensor_size);

        println!(
            "Buffers: A@0x{:x}, W@0x{:x}, Bias@0x{:x}, C@0x{:x}",
            buf_a.dma_address, buf_w.dma_address, buf_bias.dma_address, buf_c.dma_address
        );

        let shape = ConvShape {
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
        };
        let bufs = ConvBuffers {
            input_addr: buf_a.dma_address,
            weights_addr: buf_w.dma_address,
            bias_addr: buf_bias.dma_address,
            output_addr: buf_c.dma_address,
        };
        let cmds = build_conv_regcmd(&shape, &bufs);

        dump_cmds("rkt-shape-b", &cmds);

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
                // Output pixels are NOT tightly packed 1 byte apart --
                // output_channels=1 pads to task_output_channels=32, and
                // the hardware writes each pixel as a full 16-byte-aligned
                // atomic slot regardless of real channel count (see
                // rknpu-spelunking/NOTES.md's "RESOLVED: the only pixel
                // [0,0] written mystery" section -- rkt-shape-a3.rs
                // confirmed this for the known-good 1x1-kernel shape).
                // Read the first 256 bytes and pull out the 16 real pixel
                // values at stride 16 instead of assuming stride 1.
                let raw = std::slice::from_raw_parts(buf_c.host_ptr, 256);
                let pixels: Vec<u8> = (0..16).map(|i| raw[i * 16]).collect();
                let all_same = pixels.iter().all(|&b| b == pixels[0]);
                println!("Done! First 256 output bytes, 32 per row:");
                for (row, chunk) in raw.chunks(32).enumerate() {
                    println!("  [{:4}..{:4}) {:?}", row * 32, row * 32 + 32, chunk);
                }
                println!("16 output pixels (stride 16): {:?}", pixels);
                println!(
                    "All 16 output pixels identical: {} (expected: true, since input/weights are uniform)",
                    all_same
                );
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
        unsafe {
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
}
