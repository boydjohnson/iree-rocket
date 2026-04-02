use std::{fs::OpenOptions, mem, num::NonZeroUsize, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::api::{
    drm_rocket_create_bo, drm_rocket_fini_bo, drm_rocket_job, drm_rocket_prep_bo,
    drm_rocket_submit, drm_rocket_task,
};
use nix::sys::mman::{MapFlags, ProtFlags, mmap};

// --- Register Definitions ---
const TARGET_CNA: u64 = 0x01;
const TARGET_CORE: u64 = 0x11;
const TARGET_DPU: u64 = 0x41;
const TARGET_GLOBAL: u64 = 0x81;
const TARGET_PC: u64 = 0x02;

fn main() {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/accel/accel0")
        .expect("Failed to open device");
    let fd = file.as_raw_fd();

    println!("--- NPU Diagnostic: Minimal 1x1 Conv ---");

    // 1. Minimal Data (1 byte each)
    let tensor_size = 1;

    unsafe {
        // Input A (Value: 10)
        let buf_a = Buffer::new(fd, tensor_size, &file);
        ptr::write_bytes(buf_a.host_ptr, 10, tensor_size);

        // Weight W (Value: 2)
        let buf_w = Buffer::new(fd, tensor_size, &file);
        ptr::write_bytes(buf_w.host_ptr, 2, tensor_size);

        // Output C (Value: 0)
        let buf_c = Buffer::new(fd, tensor_size, &file);
        ptr::write_bytes(buf_c.host_ptr, 0, tensor_size);

        println!(
            "Buffers: A@0x{:x}, W@0x{:x}, C@0x{:x}",
            buf_a.dma_address, buf_w.dma_address, buf_c.dma_address
        );

        // 2. Build Safe Command Stream
        let cmds = build_minimal_conv(buf_a.dma_address, buf_w.dma_address, buf_c.dma_address);

        // Create Command Buffer
        let cmd_len = (cmds.len() * 8 + 4095) & !4095;
        let buf_cmd = Buffer::new(fd, cmd_len, &file);
        let cmd_slice = std::slice::from_raw_parts_mut(buf_cmd.host_ptr as *mut u64, cmds.len());
        for (i, c) in cmds.iter().enumerate() {
            cmd_slice[i] = c.0;
        }

        let fini_bo_a = drm_rocket_fini_bo {
            handle: buf_a.handle,
            reserved: 0,
        };
        let fini_bo_w = drm_rocket_fini_bo {
            handle: buf_w.handle,
            reserved: 0,
        };
        let fini_bo_c = drm_rocket_fini_bo {
            handle: buf_c.handle,
            reserved: 0,
        };
        let fini_bo_cmd = drm_rocket_fini_bo {
            handle: buf_cmd.handle,
            reserved: 0,
        };

        rocket_fini_bo(fd, &fini_bo_a).expect("Flush A");
        rocket_fini_bo(fd, &fini_bo_w).expect("Flush W");
        rocket_fini_bo(fd, &fini_bo_c).expect("Flush C");
        rocket_fini_bo(fd, &fini_bo_cmd).expect("Flush Cmd");

        // 4. Submit
        let task = drm_rocket_task {
            regcmd: buf_cmd.dma_address,
            regcmd_count: cmds.len() as u32,
        };

        // All Inputs + Cmd must be here
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
        match rocket_submit(fd, &mut submit) {
            Ok(_) => println!("Submission OK. Waiting..."),
            Err(e) => panic!("IOCTL Failed: {}", e),
        }

        // 5. Wait
        let prep = drm_rocket_prep_bo {
            handle: buf_c.handle,
            reserved: 0,
            timeout_ns: 2_000_000_000, // 2s timeout
        };

        match rocket_prep_bo(fd, &prep) {
            Ok(_) => {
                let res = *buf_c.host_ptr;
                println!("Done! Result: {}", res);
                if res == 20 {
                    println!("✅ Hardware Verified.");
                } else {
                    println!("⚠️ Wrong Result (Expected 20). Pipeline ran but math/format is off.");
                }
            }
            Err(_e) => {
                println!("❌ TIMEOUT (EBUSY). The Pipeline is stuck.");
                println!("Possible causes:");
                println!("1. Input/Weight Fetch Stall (Converter Scale=0?)");
                println!("2. Output Write Stall (Backpressure)");
                println!("3. Clock Gating (Global Kick failed)");
            }
        }
    }
}

// --- Minimal, Safe 1x1 Conv Builder ---
fn build_minimal_conv(input: u32, weight: u32, output: u32) -> Vec<RegCmd> {
    let mut cmds = Vec::new();

    // ================= CNA (Input) =================
    // Mode: Standard Conv (0), 1x1
    cmds.push(RegCmd::new(TARGET_CNA, 0x1000, 0)); // Standard
    cmds.push(RegCmd::new_raw(
        (TARGET_CNA << 48) | (((input as u64) & 0xFFFFFFFF) << 16) | 0x1004,
    ));
    cmds.push(RegCmd::new_raw(
        (TARGET_CNA << 48) | (((weight as u64) & 0xFFFFFFFF) << 16) | 0x1008,
    ));

    // Size: 1x1 (val = size - 1 = 0)
    cmds.push(RegCmd::new(TARGET_CNA, 0x100C, 0)); // Data Size (0,0)
    cmds.push(RegCmd::new(TARGET_CNA, 0x1010, 0)); // Data Ch (0)
    cmds.push(RegCmd::new(TARGET_CNA, 0x1014, 0)); // Weight Size (0,0)
    cmds.push(RegCmd::new(TARGET_CNA, 0x1018, 0)); // Weight Ch/Count (0)

    // Padding/Stride
    cmds.push(RegCmd::new(TARGET_CNA, 0x1020, 0)); // Pad
    cmds.push(RegCmd::new(TARGET_CNA, 0x1024, (1 << 16) | 1)); // Stride 1x1

    // *** NEW: Converters (Identity) ***
    // If these are 0, the pipeline might block input.
    // 0x1050: CVT_CON0 (Offset) -> 0
    cmds.push(RegCmd::new(TARGET_CNA, 0x1050, 0));
    // 0x1054: CVT_CON1 (Scale) -> 1 (Assuming integer scale, or 1.0 fixed)
    // Try 1. If this is fixed-point, 1 might be tiny, but non-zero!
    cmds.push(RegCmd::new(TARGET_CNA, 0x1054, 1));

    // ================= CORE (Compute) =================
    cmds.push(RegCmd::new(TARGET_CORE, 0x2000, 0)); // Standard Conv
    cmds.push(RegCmd::new(TARGET_CORE, 0x2010, 1)); // Truncate En

    // ================= DPU (Output) =================
    cmds.push(RegCmd::new(TARGET_DPU, 0x4004, 0x7)); // PP Mode
    cmds.push(RegCmd::new(TARGET_DPU, 0x4008, 0x1)); // Op En
    cmds.push(RegCmd::new(TARGET_DPU, 0x400C, 2)); // DRAM Mode

    cmds.push(RegCmd::new_raw(
        (TARGET_DPU << 48) | (((output as u64) & 0xFFFFFFFF) << 16) | 0x4020,
    ));
    cmds.push(RegCmd::new(TARGET_DPU, 0x4024, 1)); // Stride = 1

    cmds.push(RegCmd::new(TARGET_DPU, 0x4030, 0)); // W=0 (1)
    cmds.push(RegCmd::new(TARGET_DPU, 0x4034, 0)); // H=0 (1)
    cmds.push(RegCmd::new(TARGET_DPU, 0x403C, 0)); // C=0 (1)

    // Bypass
    cmds.push(RegCmd::new(TARGET_DPU, 0x4040, 0xF));
    cmds.push(RegCmd::new(TARGET_DPU, 0x4060, 0xF));
    cmds.push(RegCmd::new(TARGET_DPU, 0x4070, 0));

    // *** NEW: Output Converter ***
    cmds.push(RegCmd::new(TARGET_DPU, 0x4080, 0)); // Offset
    cmds.push(RegCmd::new(TARGET_DPU, 0x4084, 1)); // Scale = 1
    cmds.push(RegCmd::new(TARGET_DPU, 0x4088, 0)); // Shift

    // ================= Kick =================
    cmds.push(RegCmd::new_raw(0x0041000000000000)); // Barrier
    cmds.push(RegCmd::new(TARGET_GLOBAL, 0x0008, 0x7F)); // Global Kick

    cmds
}

nix::ioctl_readwrite!(rocket_create_bo, b'd', 0x40, drm_rocket_create_bo);
nix::ioctl_readwrite!(rocket_submit, b'd', 0x41, drm_rocket_submit);

nix::ioctl_write_ptr!(rocket_prep_bo, 'd', 0x42, drm_rocket_prep_bo);
nix::ioctl_write_ptr!(rocket_fini_bo, 'd', 0x43, drm_rocket_fini_bo);

#[repr(C)]
pub struct RegCmd(u64);

impl RegCmd {
    /// Mimics `emit_raw` from rkt_regcmd.c
    /// Packet Layout: [Target (16b) | Value (32b) | Offset (16b)]
    pub fn new(target: u64, offset: u64, value: u32) -> Self {
        let packed: u64 = (target << 48) | ((value as u64) << 16) | (offset & 0xFFFF);
        RegCmd(packed)
    }

    /// Append a raw 64-bit command (used for barriers/magic values)
    pub fn new_raw(raw: u64) -> Self {
        RegCmd(raw)
    }
}

struct Buffer {
    handle: u32,
    dma_address: u32, // NPU requires 32-bit addresses
    size: usize,
    host_ptr: *mut u8,
}

impl Buffer {
    // Helper to wrap the ioctl and mmap logic
    unsafe fn new(fd: i32, size: usize, file: &std::fs::File) -> Self { unsafe {
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
    }}
}
