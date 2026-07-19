use iree_rocket_hal::rocket::{
    api::{
        DRM_COMMAND_BASE, DRM_IOCTL_BASE, DRM_ROCKET_CREATE_BO, DRM_ROCKET_FINI_BO,
        DRM_ROCKET_PREP_BO, DRM_ROCKET_SUBMIT, drm_rocket_create_bo, drm_rocket_fini_bo,
        drm_rocket_job, drm_rocket_prep_bo, drm_rocket_submit, drm_rocket_task,
    },
    builders::{
        Bits, RegCmd, Register,
        cna::*,
        core::{
            CoreClipTruncate, CoreDataoutSize0, CoreDataoutSize1, CoreMiscCfg, CoreOperationEnable,
        },
        dpu::*,
        dpu_rdma::*,
        global::GlobalOperationEnable,
        pc::{PCBaseAddress, PCOperationEnable, PCRegisterAmounts},
    },
};
use std::{fs::OpenOptions, mem, num::NonZeroUsize, os::unix::io::AsRawFd, ptr};
// Import the generated register definitions
// We assume this module exposes the C macros as Rust functions/constants
use iree_rocket_hal::rocket::registers::*;
use nix::{
    ioctl_readwrite,
    sys::mman::{MapFlags, ProtFlags, mmap},
};

// 0x40 + 0 = 0x40
ioctl_readwrite!(
    rocket_create_bo,
    DRM_IOCTL_BASE,
    DRM_COMMAND_BASE + DRM_ROCKET_CREATE_BO,
    drm_rocket_create_bo
);

// 0x40 + 1 = 0x41
ioctl_readwrite!(
    rocket_submit,
    DRM_IOCTL_BASE,
    DRM_COMMAND_BASE + DRM_ROCKET_SUBMIT,
    drm_rocket_submit
);

// 0x40 + 2 = 0x42 (Changed to readwrite for safety/standard DRM pattern)
ioctl_readwrite!(
    rocket_prep_bo,
    DRM_IOCTL_BASE,
    DRM_COMMAND_BASE + DRM_ROCKET_PREP_BO,
    drm_rocket_prep_bo
);

// 0x40 + 3 = 0x43 (Changed to readwrite for safety/standard DRM pattern)
ioctl_readwrite!(
    rocket_fini_bo,
    DRM_IOCTL_BASE,
    DRM_COMMAND_BASE + DRM_ROCKET_FINI_BO,
    drm_rocket_fini_bo
);

struct Buffer {
    handle: u32,
    dma_address: u32, // NPU requires 32-bit addresses
    size: usize,
    host_ptr: *mut u8,
}

impl Buffer {
    // Helper to wrap the ioctl and mmap logic
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

    println!("--- NPU Diagnostic: Safe 1x1 Conv (Corrected Registers) ---");

    // 1. Data Setup (1 Byte)
    // Input = 10, Weight = 2, Expected Output = 20
    let tensor_size = 64 * 64 * 64;

    let weight_tensor_size = 64;

    unsafe {
        let buf_a = Buffer::new(fd, tensor_size, &file);
        ptr::write_bytes(buf_a.host_ptr, 10, tensor_size); // Input

        let buf_w = Buffer::new(fd, tensor_size, &file);
        ptr::write_bytes(buf_w.host_ptr, 2, weight_tensor_size); // Weight

        let buf_c = Buffer::new(fd, tensor_size, &file);
        ptr::write_bytes(buf_c.host_ptr, 0, tensor_size); // Output

        println!(
            "Buffers: A@{:#x}, W@{:#x}, C@{:#x}",
            buf_a.dma_address, buf_w.dma_address, buf_c.dma_address
        );

        let conv_op = ConvOp {
            input_w: 64,
            input_h: 64,
            input_c: 64,
            output_w: 64,
            output_h: 64,
            output_c: 64,
            weights_w: 1,
            weights_h: 1,
            stride_x: 1,
            stride_y: 1,
            padding_same: false,
            depthwise: false,
            reuse_weights: false,
            input_addr: buf_a.dma_address,
            weights_addr: buf_w.dma_address,
            output_addr: buf_c.dma_address,
            input_zp: 0,
            weights_zp: 0,
            output_zp: 0,
        };

        // 2. Build Corrected Command Stream
        let cmds = build_calculated_conv_cmd(&conv_op);

        // 3. Submit & Wait (Standard Boilerplate)
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
        rocket_fini_bo(fd, &mut fini_bo_c).expect("Flush C");
        rocket_fini_bo(fd, &mut fini_bo_cmd).expect("Flush Cmd");

        let task = drm_rocket_task {
            regcmd: buf_cmd.dma_address,
            regcmd_count: cmds.len() as u32,
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
        rocket_submit(fd, &mut submit).expect("Submit failed");

        let mut prep = drm_rocket_prep_bo {
            handle: buf_c.handle,
            reserved: 0,
            timeout_ns: i64::MAX,
        };
        rocket_prep_bo(fd, &mut prep).expect("Wait failed");

        let result = *buf_c.host_ptr;
        println!("Result: {}", result);
        if result == 20 {
            println!("✅ SUCCESS!");
        } else {
            println!("❌ FAILED");
        }
    }
}

pub fn build_calculated_conv_cmd(op: &ConvOp) -> Vec<RegCmd> {
    // 1. Calculate Bank Distribution (Math Helper)
    let task = ConvTaskConfig::calculate(op);

    let mut cmds = Vec::new();

    // ====================================================================
    // 1. CNA: Compute Configuration (1x1 Convolution)
    // ====================================================================

    // CBUF_CON0: Bank configuration
    cmds.push(
        Register::<CnaCbufCon0>::new()
            .weight_bank(Bits::new(task.weights_banks))
            .data_bank(Bits::new(task.input_banks))
            .weight_reuse(Bits::new(0))
            .build(),
    );

    // DCOMP: Decompression control -- buf_w holds raw, uncompressed weight
    // bytes, so this must actually bypass (the comment always said so, the
    // code never did): with wt_dec_bypass left at 0 and DCOMP_REGNUM=0,
    // decompression was active but had no descriptor, most likely
    // presenting CNA with an effective weight of zero -- consistent with
    // getting a clean 0 result instead of a timeout or garbage.
    cmds.push(Register::<CnaDcompRegnum>::new().build());
    cmds.push(
        Register::<CnaDcompCtrl>::new()
            .wt_dec_bypass(Bits::new(1))
            .build(),
    );

    // DATA_SIZE: Input Dimensions
    cmds.push(
        Register::<CnaDataSize0>::new()
            .datain_width(Bits::new(op.input_w))
            .datain_height(Bits::new(op.input_h))
            .build(),
    );
    cmds.push(
        Register::<CnaDataSize1>::new()
            .datain_channel_real(Bits::new(op.input_c))
            .datain_channel(Bits::new(op.input_c))
            .build(),
    );

    // DATA_SIZE: Output Dimensions
    cmds.push(
        Register::<CnaDataSize2>::new()
            .dataout_width(Bits::new(op.output_w)) // Note: CNA usually takes actual width
            .build(),
    );
    cmds.push(
        Register::<CnaDataSize3>::new()
            .dataout_atomics(Bits::new(task.atomic_count))
            .build(),
    );

    // WEIGHT_SIZE: 1x1 Kernel Configuration
    cmds.push(
        Register::<CnaWeightSize0>::new()
            .weight_bytes(Bits::new(
                op.weights_w * op.weights_h * op.input_c * task.weights_kernels,
            ))
            .build(),
    );
    cmds.push(
        Register::<CnaWeightSize1>::new()
            .weight_bytes_per_kernel(Bits::new(op.weights_w * op.weights_h * op.input_c))
            .build(),
    );
    cmds.push(
        Register::<CnaWeightSize2>::new()
            .weight_width(Bits::new(1)) // 1x1 Kernel
            .weight_height(Bits::new(1)) // 1x1 Kernel
            .weight_kernels(Bits::new(task.weights_kernels))
            .build(),
    );

    // CONV_CON: Strides and Padding
    cmds.push(Register::<CnaConvCon1>::new().build());
    cmds.push(
        Register::<CnaConvCon2>::new()
            .feature_grains(Bits::new(50 + 1 + 1)) // 1x1 stride + overhead
            .build(),
    );
    cmds.push(
        Register::<CnaConvCon3>::new()
            .conv_x_stride(Bits::new(1)) // Stride 1
            .conv_y_stride(Bits::new(1)) // Stride 1
            .build(),
    );

    // Pointers (Input and Weights)
    cmds.push(
        Register::<CnaFeatureDataAddr>::new()
            .feature_base_addr(Bits::new(op.input_addr))
            .build(),
    );
    cmds.push(
        Register::<CnaDcompAddr0>::new()
            .decompress_addr0(Bits::new(op.weights_addr))
            .build(),
    );

    // ====================================================================
    // 2. DPU: Write / Output Configuration (The Fix)
    // ====================================================================
    // According to REGISTERS.md, use DPU_DATA_CUBE for the Write Path

    // Destination Address
    cmds.push(
        Register::<DpuDstBaseAddr>::new()
            .dst_base_addr(Bits::new(op.output_addr))
            .build(),
    );

    // Output Shape (N-1)
    cmds.push(
        Register::<DpuDataCubeWidth>::new()
            .width(Bits::new(op.output_w - 1))
            .build(),
    );
    cmds.push(
        Register::<DpuDataCubeHeight>::new()
            .height(Bits::new(op.output_h - 1))
            .build(),
    );
    // orig_channel is the direct (non-N-1) real channel count -- a third
    // encoding variant living in the same register as channel's N-1;
    // rkt-simple-job.rs sets both, this was missing it (defaulted to 0).
    cmds.push(
        Register::<DpuDataCubeChannel>::new()
            .channel(Bits::new(op.output_c - 1))
            .orig_channel(Bits::new(op.output_c))
            .build(),
    );

    // Surface Stride (How many values to skip to reach next row, usually = Width)
    cmds.push(
        Register::<DpuDstSurfStride>::new()
            .dst_surf_stride(Bits::new(op.output_w))
            .build(),
    );

    // Bias-scale / batchnorm bypass -- previously left unwritten entirely
    // (only EW was bypassed), leaving these stages in an unconfirmed
    // power-on-reset state rather than a deliberate one. Both siblings
    // (rkt-basic.rs, rkt-simple-job.rs) bypass both explicitly.
    cmds.push(Register::<DpuBsCfg>::new().bs_bypass(Bits::new(1)).build());
    cmds.push(Register::<DpuBnCfg>::new().bn_bypass(Bits::new(1)).build());

    // Element-Wise / Post-Process Bypass (We just want raw multiplication)
    cmds.push(
        Register::<DpuEwCfg>::new()
            .ew_bypass(Bits::new(1))
            .ew_relu_bypass(Bits::new(1))
            .ew_lut_bypass(Bits::new(1))
            .build(),
    );

    // ====================================================================
    // 3. RDMA: Read Configuration
    // ====================================================================
    // According to REGISTERS.md, RDMA uses "DpuRdmaRdma*" prefix

    cmds.push(
        Register::<DpuRdmaSrcBaseAddr>::new()
            .src_base_addr(Bits::new(op.input_addr))
            .build(),
    );

    // Input Shape (N-1)
    cmds.push(
        Register::<DpuRdmaDataCubeWidth>::new()
            .width(Bits::new(op.input_w - 1))
            .build(),
    );
    cmds.push(
        Register::<DpuRdmaDataCubeHeight>::new()
            .height(Bits::new(op.input_h - 1))
            .build(),
    );
    cmds.push(
        Register::<DpuRdmaDataCubeChannel>::new()
            .channel(Bits::new(op.input_c - 1))
            .build(),
    );

    // Enable Read DMA (mrdma_disable = 0)
    cmds.push(
        Register::<DpuRdmaFeatureModeCfg>::new()
            .burst_len(Bits::new(15))
            .mrdma_disable(Bits::new(0))
            .build(),
    );

    // ====================================================================
    // 4. Submit
    // ====================================================================

    cmds.push(Register::<PCRegisterAmounts>::new().build());

    cmds.push(Register::<PCOperationEnable>::new().op_enable(true).build());

    // Kick the NPU
    cmds.push(
        Register::<GlobalOperationEnable>::new()
            .cna_op_en(Bits::new(1))
            .dpu_op_en(Bits::new(1))
            .dpu_rdma_op_en(Bits::new(1))
            .build(),
    );

    cmds
}

// Constants from rkt_ml.h
const CBUF_BANK_SIZE: u32 = 32768;
const CBUF_BANKS: u32 = 12;
const CBUF_ENTRIES_PER_BANK: u32 = 256;
const CBUF_ENTRY_SIZE: u32 = CBUF_BANK_SIZE / CBUF_ENTRIES_PER_BANK; // 128
const FEATURE_ATOMIC_SIZE: u32 = 16;
const WEIGHT_ATOMIC_SIZE: u32 = 32;
const ATOMIC_K_SIZE: u32 = 16;

// Helper: DIV_ROUND_UP
fn div_round_up(n: u32, d: u32) -> u32 {
    (n + d - 1) / d
}

// Helper: ALIGN (Assuming power of 2 for simplicity, or general rounding)
fn align(x: u32, a: u32) -> u32 {
    div_round_up(x, a) * a
}

// Helper: MAX2
fn max2(a: u32, b: u32) -> u32 {
    if a > b { a } else { b }
}

pub struct ConvOp {
    // Geometry
    pub input_w: u32,
    pub input_h: u32,
    pub input_c: u32,
    pub output_w: u32,
    pub output_h: u32,
    pub output_c: u32,
    pub weights_w: u32,
    pub weights_h: u32,

    // Stride & Padding
    pub stride_x: u32,
    pub stride_y: u32,
    pub padding_same: bool,

    // Modes
    pub depthwise: bool,
    pub reuse_weights: bool,

    // Addresses
    pub input_addr: u32,
    pub weights_addr: u32,
    pub output_addr: u32,

    // Quantization (Zero Points)
    pub input_zp: u32,
    pub output_zp: u32,
    pub weights_zp: u32,
}

#[derive(Debug, Default)]
pub struct ConvTaskConfig {
    // Bank Management
    pub input_banks: u32,
    pub weights_banks: u32,
    pub input_data_entries: u32,

    // Strides & Memory Layout
    pub input_line_stride: u32,
    pub input_surf_stride: u32,
    pub output_surf_stride: u32,

    // Geometry (Aligned/Adjusted)
    pub input_c_aligned: u32,
    pub output_c_aligned: u32,
    pub weights_kernels: u32,

    // Padding
    pub pad_top: u32,
    pub pad_bottom: u32,
    pub pad_left: u32,
    pub pad_right: u32,

    // Misc
    pub atomic_count: u32,
    pub surfaces_per_row: u32,
}

impl ConvTaskConfig {
    pub fn calculate(op: &ConvOp) -> Self {
        let mut task = ConvTaskConfig::default();

        // --------------------------------------------------------
        // 1. Calculate Padding (calc_explicit_padding)
        // --------------------------------------------------------
        if op.padding_same && op.weights_w > 1 {
            let pad_w = max2(
                (op.output_w - 1) * op.stride_x + op.weights_w - op.input_w,
                0,
            );
            let pad_h = max2(
                (op.output_h - 1) * op.stride_y + op.weights_h - op.input_h,
                0,
            );
            task.pad_left = pad_w / 2;
            task.pad_right = pad_w - task.pad_left;
            task.pad_top = pad_h / 2;
            task.pad_bottom = pad_h - task.pad_top;
        }

        // --------------------------------------------------------
        // 2. Calculate Banks & Entries
        // --------------------------------------------------------
        // calc_entries_per_slice
        let bpe = 1; // Byte per element (INT8)
        let atomics_per_entry = CBUF_ENTRY_SIZE / FEATURE_ATOMIC_SIZE; // 128/16 = 8
        let total_c_atomics = div_round_up(op.input_c * bpe, FEATURE_ATOMIC_SIZE);
        let last_c_atomics = total_c_atomics % atomics_per_entry;

        let int_c_entries = (total_c_atomics / atomics_per_entry) * op.input_w;
        let frac_c_entries = if last_c_atomics == 3 {
            op.input_w
        } else {
            div_round_up(last_c_atomics * op.input_w, atomics_per_entry)
        };
        let entries_per_slice = int_c_entries + frac_c_entries;

        // calc_input_banks
        task.input_banks = div_round_up(entries_per_slice * op.input_h, CBUF_ENTRIES_PER_BANK);

        // calc_weights_banks
        let mut w_bytes = op.weights_w * op.weights_h * op.input_c * bpe;
        if !op.depthwise {
            w_bytes *= op.output_c;
        }
        let w_entries = div_round_up(w_bytes, CBUF_ENTRY_SIZE);
        task.weights_banks = div_round_up(w_entries, CBUF_ENTRIES_PER_BANK) + 1;

        // --------------------------------------------------------
        // 3. Calculate Strides (fill_task logic)
        // --------------------------------------------------------

        // Input Line Stride
        // C logic: calc_line_stride(width) / 4 usually
        // calc_line_stride = width * 16 * 1
        let raw_line_stride = op.input_w * ATOMIC_K_SIZE * 1;
        task.input_line_stride = raw_line_stride / 4;

        // Input Surface Stride
        // C logic: float calc involved? usually line_stride * (height/4 - 1)
        // Approximate for integer math:
        let h_div_4 = (op.input_h as f32) / 4.0;
        task.input_surf_stride = ((task.input_line_stride as f32) * (h_div_4 - 1.0)) as u32;

        // Output Surface Stride
        let out_line_stride = op.output_w * ATOMIC_K_SIZE * 1;
        // C logic: (out_line * out_h) / FEATURE_ATOMIC_SIZE
        task.output_surf_stride = (out_line_stride * op.output_h) / FEATURE_ATOMIC_SIZE;

        // --------------------------------------------------------
        // 4. Input Data Entries (CBUF_CON1)
        // --------------------------------------------------------
        // Simplified logic from C fill_task
        if op.input_c == 1 {
            task.input_data_entries = op.input_w * op.input_h;
        } else {
            let c_chunks = div_round_up(op.input_c, FEATURE_ATOMIC_SIZE);
            task.input_data_entries = div_round_up(op.input_w * 2 * c_chunks, 8);
        }

        // --------------------------------------------------------
        // 5. Aligned Dimensions
        // --------------------------------------------------------
        task.input_c_aligned = align(max2(op.input_c, FEATURE_ATOMIC_SIZE), FEATURE_ATOMIC_SIZE);

        task.output_c_aligned = align(max2(op.output_c, 32), 32);
        if op.depthwise {
            // C code specific adjustment
            task.output_c_aligned = align(task.output_c_aligned, 64);
        }

        // Weights Kernels
        if op.depthwise {
            task.weights_kernels = 1;
        } else {
            task.weights_kernels = align(op.output_c, 2);
        }

        // Surfaces Per Row (SURFACE_ADD)
        task.surfaces_per_row = op.output_w * op.output_h * 2;
        if op.depthwise {
            task.surfaces_per_row *= 2;
        }

        // Atomic Count (DATA_SIZE3)
        // Not explicitly calculated in C snippet provided, but typically:
        task.atomic_count = op.output_w * op.output_h; // Fallback

        task
    }
}
