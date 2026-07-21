//! NPU diagnostic: rkt-shape-a.rs (identical shape to rkt-basic.rs's
//! "validated" 4x4x1/1x1-kernel config, but reading back all 16 output
//! bytes) showed the *same* "only pixel [0,0] written" result rkt-shape-
//! b.rs did -- confirming this is a pre-existing gap in
//! `build_conv_regcmd`, not something the 3x3 kernel introduced.
//!
//! This variant tests one concrete theory for that gap: `regcmd.rs` sets
//! `CNA_CONV_CON2.feature_grains` to `50 + stride + 1` (= 52 for stride
//! 1), copied verbatim from Mesa's `rkt_regcmd.c` -- which has its own
//! comment admitting it's a hack ("Magic: Seems to pass the most tests",
//! rknpu-spelunking/mesa-rocket-userspace/rkt_regcmd.c). The RK3588 TRM
//! (chapter 36, `CNA_CONV_CON2` register description) documents the
//! *intended* formula: "Feature data rows needs to be buffered before
//! convolution start. It's suggested to set this field as
//! y_stride+weight_height+1" -- 3 for this shape, not 52. Real models'
//! early layers have hundreds of input rows, so 52 is a trivial, harmless
//! amount of buffering for Mesa's real-world usage; our test inputs only
//! have 4-6 rows total, so asking CNA to buffer 52 before starting is
//! nonsensical and a very plausible explanation for "only the first
//! output atomic gets produced before something signals done".
//!
//! Patches `feature_grains` post-hoc (via the same RegCmd-rewriting
//! technique rkt-fp16.rs used for precision) to the TRM-documented
//! formula and re-runs the identical known-good shape, to see if all 16
//! output bytes get written instead of just byte 0.

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
    ioctl_readwrite, ioctl_write_ptr,
    sys::mman::{MapFlags, ProtFlags, mmap},
    time::{ClockId, clock_gettime},
};

const CONV_CON2_OFFSET: u64 = 0x1010;

/// CNA_CONV_CON2's FEATURE_GRAINS field (bits 13:4, 10 bits) --
/// `build_conv_regcmd` sets it to Mesa's `50 + stride + 1` magic value;
/// this overrides it to the TRM-documented `y_stride + weight_height + 1`
/// formula instead.
fn patch_feature_grains(cmds: &mut [iree_rocket_hal::rocket::builders::RegCmd], grains: u64) {
    assert!(grains < (1 << 10), "feature_grains is a 10-bit field");
    for cmd in cmds.iter_mut() {
        if cmd.0 & 0xFFFF == CONV_CON2_OFFSET {
            let domain = cmd.0 >> 48;
            let mut val = (cmd.0 >> 16) & 0xFFFF_FFFF;
            val &= !(0x3ff << 4);
            val |= grains << 4;
            cmd.0 = (domain << 48) | (val << 16) | CONV_CON2_OFFSET;
        }
    }
}

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

    println!(
        "--- NPU Diagnostic: shape A2, same shape as shape A but feature_grains patched to TRM formula (3, not Mesa's magic 52) ---"
    );
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
        };
        let bufs = ConvBuffers {
            input_addr: buf_a.dma_address,
            weights_addr: buf_w.dma_address,
            bias_addr: buf_bias.dma_address,
            output_addr: buf_c.dma_address,
        };
        let mut cmds = build_conv_regcmd(&shape, &bufs);
        // TRM formula: y_stride(1) + weight_height(1) + 1 = 3, vs. Mesa's
        // magic 50 + stride(1) + 1 = 52 that build_conv_regcmd uses by
        // default.
        patch_feature_grains(&mut cmds, 3);

        dump_cmds("rkt-shape-a2", &cmds);

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
                let out = std::slice::from_raw_parts(buf_c.host_ptr, 16);
                let all_same = out.iter().all(|&b| b == out[0]);
                println!("Done! Output 4x4 grid: {:?}", out);
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
