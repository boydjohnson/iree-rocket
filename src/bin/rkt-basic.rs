use std::{fs::OpenOptions, mem, num::NonZeroUsize, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::api::{
    DRM_COMMAND_BASE, DRM_IOCTL_BASE, DRM_ROCKET_CREATE_BO, DRM_ROCKET_FINI_BO,
    DRM_ROCKET_PREP_BO, DRM_ROCKET_SUBMIT, drm_rocket_create_bo, drm_rocket_fini_bo,
    drm_rocket_job, drm_rocket_prep_bo, drm_rocket_submit, drm_rocket_task,
};
use nix::{
    ioctl_readwrite, ioctl_write_ptr,
    sys::mman::{MapFlags, ProtFlags, mmap},
};

// --- Correct Hardware Domain IDs ---
const TARGET_CNA: u64 = 0x01;
const TARGET_CORE: u64 = 0x11;
const TARGET_DPU: u64 = 0x41;
const TARGET_GLOBAL: u64 = 0x81;
const TARGET_PC: u64 = 0x02;

// --- Correct Register Offsets from rkt_registers.h ---
const REG_CNA_CONV_CON0: u64 = 0x1010;
const REG_CNA_FEATURE_DATA_ADDR: u64 = 0x1070;
const REG_CNA_DCOMP_ADDR0: u64 = 0x1110;
const REG_CNA_OPERATION_ENABLE: u64 = 0x1008;

const REG_CORE_OPERATION_ENABLE: u64 = 0x3008;

const REG_DPU_OPERATION_ENABLE: u64 = 0x4008;
const REG_DPU_DST_BASE_ADDR: u64 = 0x4020;

const REG_GLOBAL_OPERATION_ENABLE: u64 = 0xf008;
const REG_PC_OPERATION_ENABLE: u64 = 0x0008;

fn main() {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/accel/accel0")
        .expect("Failed to open device");
    let fd = file.as_raw_fd();

    println!("--- NPU Diagnostic: Minimal 1x1 Conv (Fixed Offsets) ---");

    let tensor_size = 4096; // 4KB aligned

    unsafe {
        let buf_a = Buffer::new(fd, tensor_size, &file);
        ptr::write_bytes(buf_a.host_ptr, 10, tensor_size);

        let buf_w = Buffer::new(fd, tensor_size, &file);
        ptr::write_bytes(buf_w.host_ptr, 2, tensor_size);

        let buf_c = Buffer::new(fd, tensor_size, &file);
        ptr::write_bytes(buf_c.host_ptr, 0, tensor_size);

        println!(
            "Buffers: A@0x{:x}, W@0x{:x}, C@0x{:x}",
            buf_a.dma_address, buf_w.dma_address, buf_c.dma_address
        );

        let mut cmds = Vec::new();
        // 1. CNA
        cmds.push(RegCmd::new(TARGET_CNA, REG_CNA_CONV_CON0, 0)); 
        cmds.push(RegCmd::new(TARGET_CNA, REG_CNA_FEATURE_DATA_ADDR, buf_a.dma_address));
        cmds.push(RegCmd::new(TARGET_CNA, REG_CNA_DCOMP_ADDR0, buf_w.dma_address));
        cmds.push(RegCmd::new(TARGET_CNA, REG_CNA_OPERATION_ENABLE, 1));

        // 2. CORE
        cmds.push(RegCmd::new(TARGET_CORE, REG_CORE_OPERATION_ENABLE, 1));

        // 3. DPU
        cmds.push(RegCmd::new(TARGET_DPU, REG_DPU_DST_BASE_ADDR, buf_c.dma_address));
        cmds.push(RegCmd::new(TARGET_DPU, REG_DPU_OPERATION_ENABLE, 1));

        // 4. KICK
        cmds.push(RegCmd::new(TARGET_GLOBAL, REG_GLOBAL_OPERATION_ENABLE, 0x7F));
        cmds.push(RegCmd::new(TARGET_PC, REG_PC_OPERATION_ENABLE, 1));

        // Create Command Buffer
        let cmd_len = 4096;
        let buf_cmd = Buffer::new(fd, cmd_len, &file);
        let cmd_slice = std::slice::from_raw_parts_mut(buf_cmd.host_ptr as *mut u64, cmds.len());
        for (i, c) in cmds.iter().enumerate() {
            cmd_slice[i] = c.0;
        }

        rocket_fini_bo(fd, &drm_rocket_fini_bo { handle: buf_a.handle, reserved: 0 }).ok();
        rocket_fini_bo(fd, &drm_rocket_fini_bo { handle: buf_w.handle, reserved: 0 }).ok();
        rocket_fini_bo(fd, &drm_rocket_fini_bo { handle: buf_c.handle, reserved: 0 }).ok();
        rocket_fini_bo(fd, &drm_rocket_fini_bo { handle: buf_cmd.handle, reserved: 0 }).ok();

        // 4. Submit (Count Fix)
        let task = drm_rocket_task {
            regcmd: buf_cmd.dma_address,
            regcmd_count: cmds.len() as u32 * 2,
        };

        let in_handles = vec![buf_cmd.handle, buf_a.handle, buf_w.handle];
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

        let prep = drm_rocket_prep_bo {
            handle: buf_c.handle,
            reserved: 0,
            timeout_ns: 2_000_000_000,
        };

        match rocket_prep_bo(fd, &prep) {
            Ok(_) => {
                let res = *buf_c.host_ptr;
                println!("Done! Result[0]: {}", res);
            }
            Err(e) => {
                println!("❌ TIMEOUT (EBUSY): {}", e);
            }
        }
    }
}

ioctl_readwrite!(rocket_create_bo, DRM_IOCTL_BASE, DRM_COMMAND_BASE + DRM_ROCKET_CREATE_BO, drm_rocket_create_bo);
ioctl_readwrite!(rocket_submit, DRM_IOCTL_BASE, DRM_COMMAND_BASE + DRM_ROCKET_SUBMIT, drm_rocket_submit);
ioctl_write_ptr!(rocket_prep_bo, DRM_IOCTL_BASE, DRM_COMMAND_BASE + DRM_ROCKET_PREP_BO, drm_rocket_prep_bo);
ioctl_write_ptr!(rocket_fini_bo, DRM_IOCTL_BASE, DRM_COMMAND_BASE + DRM_ROCKET_FINI_BO, drm_rocket_fini_bo);

#[repr(C)]
pub struct RegCmd(u64);

impl RegCmd {
    pub fn new(target: u64, offset: u64, value: u32) -> Self {
        let packed: u64 = (target << 48) | ((value as u64) << 16) | (offset & 0xFFFF);
        RegCmd(packed)
    }
}

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
        let map_addr = mmap(None, map_len, ProtFlags::PROT_READ | ProtFlags::PROT_WRITE, MapFlags::MAP_SHARED, file, create_params.offset as i64).expect("mmap failed");
        Buffer { handle: create_params.handle, dma_address: create_params.dma_address as u32, size, host_ptr: map_addr.as_ptr() as *mut u8 }
    }
}
