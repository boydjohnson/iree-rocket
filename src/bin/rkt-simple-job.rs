use std::{fs::OpenOptions, marker::PhantomData, num::NonZeroUsize, os::fd::AsRawFd};

use iree_rocket_hal::rocket::{
    api::{
        DRM_COMMAND_BASE, DRM_IOCTL_BASE, DRM_ROCKET_CREATE_BO, DRM_ROCKET_FINI_BO,
        DRM_ROCKET_PREP_BO, DRM_ROCKET_SUBMIT, drm_rocket_create_bo, drm_rocket_fini_bo,
        drm_rocket_job, drm_rocket_prep_bo, drm_rocket_submit, drm_rocket_task,
    },
    builders::{
        Bits, RegCmd, Register,
        cna::{
            CnaCbufCon0, CnaCbufCon1, CnaClkGate, CnaConvCon1, CnaConvCon2, CnaConvCon3,
            CnaCvtCon0, CnaDataSize0, CnaDataSize1, CnaDataSize2, CnaDataSize3, CnaDcompAddr0,
            CnaDcompCtrl, CnaDmaCon0, CnaDmaCon1, CnaDmaCon2, CnaFeatureDataAddr,
            CnaOperationEnable, CnaPadCon0, CnaSPointer, CnaWeightSize0, CnaWeightSize1,
            CnaWeightSize2,
        },
        core::{
            CoreClipTruncate, CoreDataoutSize0, CoreDataoutSize1, CoreMiscCfg, CoreOperationEnable,
            CoreSPointer,
        },
        dpu::{
            DpuBnCfg, DpuBsCfg, DpuDataCubeChannel, DpuDataCubeHeight, DpuDataCubeWidth,
            DpuDataFormat, DpuDstBaseAddr, DpuDstSurfStride, DpuEwCfg, DpuFeatureModeCfg,
            DpuOperationEnable, DpuOutCvtOffset, DpuOutCvtScale, DpuOutCvtShift, DpuSPointer,
            DpuWdmaSize0, DpuWdmaSize1,
        },
        global::GlobalOperationEnable,
        pc::{PCBaseAddress, PCInterruptClear, PCInterruptMask},
    },
    registers::{
        PC_OPERATION_ENABLE_OP_EN, PC_OPERATION_ENABLE_RESERVED_0, REG_PC_OPERATION_ENABLE,
    },
};
use nix::{
    ioctl_readwrite,
    sys::mman::{MapFlags, ProtFlags, mmap},
};

fn with_cpu_access<F>(fd: i32, buf: &Buffer, label: &str, mut f: F)
where
    F: FnMut(&mut [u8]),
{
    // Take CPU ownership + wait for any in-flight NPU access to finish

    let mut prep = drm_rocket_prep_bo {
        handle: buf.handle,

        reserved: 0,

        timeout_ns: i64::MAX,
    };

    println!("PREP_BO ({})", label);

    unsafe {
        let ret = rocket_prep_bo(fd, &mut prep).expect("prep_bo failed");

        match ret >= 0 {
            true => println!("PREP_BO succeeded"),

            false => println!("PREP_BO failed: {:?}", std::io::Error::last_os_error()),
        }
    }

    // Safe-ish view over the mmap'd region

    let slice = unsafe { std::slice::from_raw_parts_mut(buf.host_ptr, buf.size as usize) };

    f(slice);

    // Give ownership back to device and sync caches for NPU access

    let mut fini = drm_rocket_fini_bo {
        handle: buf.handle,

        reserved: 0,
    };

    println!("FINI_BO ({})", label);

    unsafe {
        let ret = rocket_fini_bo(fd, &mut fini).expect("fini_bo failed");

        match ret >= 0 {
            true => println!("FINI_BO succeeded"),

            false => println!("FINI_BO  failed: {:?}", std::io::Error::last_os_error()),
        }
    }
}

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

fn fill_buffer<T: NpuDataType>(fd: i32, buf: &Buffer, label: &str, value: T) {
    with_cpu_access(fd, buf, label, |slice| {
        let mut view = unsafe { BufferView::<T>::new(slice) };
        view.fill(value);
    });
}

/// Write structured data to a buffer (type-safe)
fn write_buffer<T: NpuDataType>(fd: i32, buf: &Buffer, label: &str, data: &[T]) {
    with_cpu_access(fd, buf, label, |slice| {
        let mut view = unsafe { BufferView::<T>::new(slice) };
        view.copy_from_slice(data);
    });
}

pub struct BufferView<'a, T: NpuDataType> {
    data: &'a mut [u8],
    _phantom: PhantomData<T>,
}

impl<'a, T: NpuDataType> BufferView<'a, T> {
    /// Create a typed view of the buffer
    ///
    /// # Safety
    /// - Buffer must be properly aligned for T
    /// - Buffer size must be a multiple of sizeof(T)
    pub unsafe fn new(data: &'a mut [u8]) -> Self {
        assert_eq!(
            data.len() % T::bytes_per_element(),
            0,
            "Buffer size must be multiple of element size"
        );

        Self {
            data,
            _phantom: PhantomData,
        }
    }

    /// Get the typed slice
    pub fn as_slice(&mut self) -> &mut [T] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.data.as_mut_ptr() as *mut T,
                self.data.len() / T::bytes_per_element(),
            )
        }
    }

    /// Fill the entire buffer with a single value
    pub fn fill(&mut self, value: T) {
        self.as_slice().fill(value);
    }

    /// Copy data into the buffer
    pub fn copy_from_slice(&mut self, src: &[T]) {
        let slice = self.as_slice();
        assert_eq!(slice.len(), src.len(), "Buffer size mismatch");
        slice.copy_from_slice(src);
    }

    /// Get the precision type
    pub fn precision() -> Precision {
        T::precision()
    }
}

pub trait NpuDataType: Copy {
    fn precision() -> Precision;
    fn bytes_per_element() -> usize;
}

impl NpuDataType for i8 {
    fn precision() -> Precision {
        Precision::Int8
    }
    fn bytes_per_element() -> usize {
        1
    }
}

impl NpuDataType for i16 {
    fn precision() -> Precision {
        Precision::Int16
    }
    fn bytes_per_element() -> usize {
        2
    }
}

impl NpuDataType for i32 {
    fn precision() -> Precision {
        Precision::Int32
    }
    fn bytes_per_element() -> usize {
        4
    }
}

fn main() {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/accel/accel0")
        .expect("Failed to open device");
    let fd = file.as_raw_fd();

    println!("--- NPU Diagnostic: Fixed CnaCore1x1Job ---");

    // 1. Data Setup (64x64x3 bytes)
    let tensor_size = 224 * 224 * 24;

    unsafe {
        let buf_a = Buffer::new(fd, tensor_size, &file); // Input
        let buf_w = Buffer::new(fd, tensor_size, &file); // Weights
        let buf_c = Buffer::new(fd, tensor_size, &file); // Output
        let buf_cmd = Buffer::new(fd, 4096, &file); // Command Stream

        println!(
            "Buffers:\n  A@{:#x}\n  W@{:#x}\n  C@{:#x}\n  CMD@{:#x}",
            buf_a.dma_address, buf_w.dma_address, buf_c.dma_address, buf_cmd.dma_address
        );

        fill_buffer::<i8>(fd, &buf_a, "fill A", 10);
        fill_buffer::<i8>(fd, &buf_w, "fill W", 2);
        fill_buffer::<i8>(fd, &buf_c, "zero C", 0);

        // 3. Build Command Stream
        let job_desc = CnaCore1x1Job::new(
            224u32,
            224u32,
            24u32,
            24u32,
            Precision::Int8,
            Precision::Int8,
            Precision::Int8,
            buf_c.dma_address as u64,
            buf_a.dma_address as u64,
            buf_w.dma_address as u64,
        )
        .unwrap();

        let cmds = job_desc.build_regcmds();

        with_cpu_access(fd, &buf_cmd, "write CMD", |raw| {
            let cmd_slice =
                std::slice::from_raw_parts_mut(raw.as_mut_ptr() as *mut u64, cmds.len());
            for (i, c) in cmds.iter().enumerate() {
                cmd_slice[i] = c.0;
            }
        });

        // 4. Submit Job
        // FIX: Multiply len by 2 to satisfy kernel's "(N+1)/2 - 1" logic
        let regcmd_count_val = (cmds.len() * 2) as u32;

        let task = drm_rocket_task {
            regcmd: buf_cmd.dma_address,
            regcmd_count: regcmd_count_val,
        };

        let in_handles = vec![buf_cmd.handle, buf_a.handle, buf_w.handle];
        let out_handles = vec![buf_c.handle];

        let job = drm_rocket_job {
            tasks: &task as *const _ as u64,
            in_bo_handles: in_handles.as_ptr() as u64,
            out_bo_handles: out_handles.as_ptr() as u64,
            task_count: 1,
            task_struct_size: std::mem::size_of::<drm_rocket_task>() as u32,
            in_bo_handle_count: in_handles.len() as u32,
            out_bo_handle_count: out_handles.len() as u32,
        };

        let mut submit = drm_rocket_submit {
            jobs: &job as *const _ as u64,
            job_count: 1,
            job_struct_size: std::mem::size_of::<drm_rocket_job>() as u32,
            reserved: 0,
        };

        println!("Submitting job (count={})...", regcmd_count_val);
        rocket_submit(fd, &mut submit).expect("Submit failed");

        // 5. Read Back Result
        let mut prep_out = drm_rocket_prep_bo {
            handle: buf_c.handle,
            reserved: 0,
            timeout_ns: i64::MAX,
        };
        rocket_prep_bo(fd, &mut prep_out).expect("prep_bo failed");

        let result = std::slice::from_raw_parts(buf_c.host_ptr, buf_c.size as usize);
        println!("Result[0]: {}", result[0]);

        // Expectation: 10 (Input) * 2 (Weight) = 20
        if result[0] == 20 {
            println!("✅ SUCCESS!");
        } else {
            println!("❌ FAILED");
        }
    }
}

use std::{error::Error as StdError, fmt};

// ============================================================================
// Error Types
// ============================================================================

#[derive(Debug)]
pub enum RknnError {
    InvalidChannels(String),
    InvalidAlignment(String),
    InvalidDimensions(String),
    InvalidPrecision(String),
}

impl fmt::Display for RknnError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            RknnError::InvalidChannels(s) => write!(f, "Invalid channels: {}", s),
            RknnError::InvalidAlignment(s) => write!(f, "Invalid alignment: {}", s),
            RknnError::InvalidDimensions(s) => write!(f, "Invalid dimensions: {}", s),
            RknnError::InvalidPrecision(s) => write!(f, "Invalid precision: {}", s),
        }
    }
}

impl StdError for RknnError {}

// ============================================================================
// Precision Types (TRM Section 36.3.2)
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    Int8 = 0,
    Int16 = 1,
    Fp16 = 2,
    BFloat16 = 3,
    Int32 = 4,
    Fp32 = 5,
    Int4 = 6,
}

impl Precision {
    pub fn bytes_per_element(&self) -> u32 {
        match self {
            Precision::Int4 => 1, // Packed
            Precision::Int8 => 1,
            Precision::Int16 | Precision::Fp16 | Precision::BFloat16 => 2,
            Precision::Int32 | Precision::Fp32 => 4,
        }
    }

    pub fn channel_alignment(&self) -> u32 {
        match self {
            Precision::Int4 => 16,
            Precision::Int8 => 8,
            Precision::Int16 | Precision::Fp16 | Precision::BFloat16 => 4,
            Precision::Int32 | Precision::Fp32 => 2,
        }
    }

    pub fn as_u32(&self) -> u32 {
        *self as u32
    }
}

// ============================================================================
// Convolution Mode (TRM Section 36.4.2)
// ============================================================================

#[derive(Debug, Clone, Copy)]
pub enum ConvMode {
    Direct = 0,    // Standard convolution
    Depthwise = 3, // Depthwise convolution
}

// ============================================================================
// Output Mode (TRM Section 36.4.2)
// ============================================================================

#[derive(Debug, Clone, Copy)]
pub enum OutputMode {
    ToPpu = 1,    // Bit 0: to PPU
    ToMemory = 2, // Bit 1: to memory
    Both = 3,     // Both bits set
}

// ============================================================================
// Register Command Structure (You'll have your own implementation)
// ============================================================================

// ============================================================================
// Main Job Structure
// ============================================================================

#[derive(Debug, Clone)]
pub struct CnaCore1x1Job {
    // Input/Output dimensions
    width: u32,
    height: u32,
    in_channels: u32,
    out_channels: u32,

    // Precision settings
    in_precision: Precision,
    proc_precision: Precision,
    out_precision: Precision,

    // Memory addresses (IOVA - I/O Virtual Address)
    dst_iova: u64,
    src_iova: u64,
    weight_iova: u64,

    // Convolution parameters
    conv_mode: ConvMode,
    output_mode: OutputMode,

    // Advanced options
    use_ping_pong: bool,
    register_group: u8,
    enable_cvt: bool,
    pad_top: u32,
    pad_left: u32,
}

impl CnaCore1x1Job {
    pub fn new(
        width: u32,
        height: u32,
        in_channels: u32,
        out_channels: u32,
        in_precision: Precision,
        proc_precision: Precision,
        out_precision: Precision,
        dst_iova: u64,
        src_iova: u64,
        weight_iova: u64,
    ) -> Result<Self, RknnError> {
        // Validate dimensions
        if width == 0 || height == 0 {
            return Err(RknnError::InvalidDimensions(
                "Width and height must be > 0".to_string(),
            ));
        }

        // Validate channel alignment based on precision
        let in_align = in_precision.channel_alignment();
        if in_channels % in_align != 0 {
            return Err(RknnError::InvalidChannels(format!(
                "Input channels ({}) must be multiple of {} for {:?}",
                in_channels, in_align, in_precision
            )));
        }

        let out_align = out_precision.channel_alignment();
        if out_channels % out_align != 0 {
            return Err(RknnError::InvalidChannels(format!(
                "Output channels ({}) must be multiple of {} for {:?}",
                out_channels, out_align, out_precision
            )));
        }

        // Validate address alignment (16-byte for weight and destination)
        if weight_iova & 0xF != 0 {
            return Err(RknnError::InvalidAlignment(
                "Weight address must be 16-byte aligned".to_string(),
            ));
        }

        if dst_iova & 0xF != 0 {
            return Err(RknnError::InvalidAlignment(
                "Destination address must be 16-byte aligned".to_string(),
            ));
        }

        Ok(Self {
            width,
            height,
            in_channels,
            out_channels,
            in_precision,
            proc_precision,
            out_precision,
            dst_iova,
            src_iova,
            weight_iova,
            conv_mode: ConvMode::Direct,
            output_mode: OutputMode::ToMemory,
            use_ping_pong: false,
            register_group: 0,
            enable_cvt: false,
            pad_top: 0,
            pad_left: 0,
        })
    }

    /// Set convolution mode
    pub fn with_conv_mode(mut self, mode: ConvMode) -> Self {
        self.conv_mode = mode;
        self
    }

    /// Set output mode
    pub fn with_output_mode(mut self, mode: OutputMode) -> Self {
        self.output_mode = mode;
        self
    }

    /// Enable ping-pong register operation
    pub fn with_ping_pong(mut self, enable: bool, group: u8) -> Self {
        self.use_ping_pong = enable;
        self.register_group = group & 1;
        self
    }

    /// Enable input conversion
    pub fn with_cvt(mut self, enable: bool) -> Self {
        self.enable_cvt = enable;
        self
    }

    /// Set padding
    pub fn with_padding(mut self, top: u32, left: u32) -> Self {
        self.pad_top = top;
        self.pad_left = left;
        self
    }

    /// Build complete register command sequence
    /// Following TRM Figure 36-4: Convolution Flow 1
    pub fn build_regcmds(&self) -> Vec<RegCmd> {
        let mut cmds = Vec::new();

        // ====================================================================
        // STEP 1: Interrupt Configuration (TRM Section 36.5.3)
        // ====================================================================

        cmds.push(Register::<PCBaseAddress>::new().pc_sel(true).build());

        // Clear all pending interrupts
        cmds.push(
            Register::<PCInterruptClear>::new()
                .dpu_0(true)
                .dpu_1(true)
                .core_0(true)
                .core_1(true)
                .cna_feature_0(true)
                .cna_weight_0(true)
                .build(),
        );

        // Enable relevant interrupt masks
        // Bits: [0-1] CNA feature, [2-3] CNA weight, [6-7] CORE, [8-9] DPU
        cmds.push(
            Register::<PCInterruptMask>::new()
                .cna_feature_0(true)
                .cna_feature_1(true)
                .cna_weight_0(true)
                .cna_weight_1(true)
                .core_0(true)
                .core_1(true)
                .dpu_0(true)
                .dpu_1(true)
                .build(),
        );

        // ====================================================================
        // STEP 2: Ping-Pong Register Configuration (TRM Section 36.5.1)
        // ====================================================================

        if self.use_ping_pong {
            // CNA ping-pong
            cmds.push(
                Register::<CnaSPointer>::new()
                    .pointer_pp_en(Bits::new(1))
                    .pointer(Bits::new(self.register_group as u32))
                    .build(),
            );

            // CORE ping-pong
            cmds.push(
                Register::<CoreSPointer>::new()
                    .pointer_pp_en(Bits::new(1))
                    .pointer(Bits::new(self.register_group as u32))
                    .build(),
            );

            // DPU ping-pong
            cmds.push(
                Register::<DpuSPointer>::new()
                    .pointer_pp_en(Bits::new(1))
                    .pointer(Bits::new(self.register_group as u32))
                    .build(),
            );
        }

        // ====================================================================
        // STEP 3: CNA Data Fetch Configuration (TRM Section 36.4.2)
        // ====================================================================

        let bpe = self.in_precision.bytes_per_element();

        // Input data dimensions (TRM: RKNN_cna_data_size0)
        cmds.push(
            Register::<CnaDataSize0>::new()
                .datain_width(Bits::new(self.width))
                .datain_height(Bits::new(self.height))
                .build(),
        );

        // Input channels (TRM: RKNN_cna_data_size1)
        // Note: datain_channel is "minus 1" for hardware
        cmds.push(
            Register::<CnaDataSize1>::new()
                .datain_channel(Bits::new(self.in_channels - 1))
                .datain_channel_real(Bits::new(self.in_channels))
                .build(),
        );

        // Output width (TRM: RKNN_cna_data_size2)
        cmds.push(
            Register::<CnaDataSize2>::new()
                .dataout_width(Bits::new(self.width))
                .build(),
        );

        // Output atomics (total output pixels) (TRM: RKNN_cna_data_size3)
        let dataout_atomics = self.width * self.height;
        cmds.push(
            Register::<CnaDataSize3>::new()
                .dataout_atomics(Bits::new(dataout_atomics))
                .surf_mode(Bits::new(0)) // 1 surface series
                .build(),
        );

        // Memory stride configuration (TRM: RKNN_cna_dma_con1/2)
        let line_stride = self.width * self.in_channels * bpe;
        let surf_stride = self.height * line_stride;

        cmds.push(
            Register::<CnaDmaCon1>::new()
                .line_stride(Bits::new(line_stride))
                .build(),
        );

        cmds.push(
            Register::<CnaDmaCon2>::new()
                .surf_stride(Bits::new(surf_stride))
                .build(),
        );

        // DMA burst configuration (TRM: RKNN_cna_dma_con0)
        cmds.push(
            Register::<CnaDmaCon0>::new()
                .data_burst_len(Bits::new(7)) // Burst8 (4'd7)
                .weight_burst_len(Bits::new(7)) // Burst8
                .ov4k_bypass(Bits::new(0)) // Enable 4K boundary split
                .build(),
        );

        // Feature data base address (TRM: RKNN_cna_feature_data_addr)
        // Full 32-bit address, no shift needed
        cmds.push(
            Register::<CnaFeatureDataAddr>::new()
                .feature_base_addr(Bits::new(self.src_iova as u32))
                .build(),
        );

        // ====================================================================
        // STEP 4: CNA Weight Configuration
        // ====================================================================

        let weight_bpe = self.proc_precision.bytes_per_element();
        let weight_bytes_per_kernel = self.in_channels * weight_bpe;
        let total_weight_bytes = weight_bytes_per_kernel * self.out_channels;

        // Total weight bytes (TRM: RKNN_cna_weight_size0)
        cmds.push(
            Register::<CnaWeightSize0>::new()
                .weight_bytes(Bits::new(total_weight_bytes))
                .build(),
        );

        // Bytes per kernel (TRM: RKNN_cna_weight_size1)
        cmds.push(
            Register::<CnaWeightSize1>::new()
                .weight_bytes_per_kernel(Bits::new(weight_bytes_per_kernel))
                .build(),
        );

        // Kernel dimensions: 1x1xout_channels (TRM: RKNN_cna_weight_size2)
        cmds.push(
            Register::<CnaWeightSize2>::new()
                .weight_width(Bits::new(1)) // 1x1 kernel
                .weight_height(Bits::new(1))
                .weight_kernels(Bits::new(self.out_channels))
                .build(),
        );

        // Weight address (TRM: RKNN_cna_dcomp_addr0)
        // Bits 31:4 used, so shift right by 4
        cmds.push(
            Register::<CnaDcompAddr0>::new()
                .decompress_addr0(Bits::new((self.weight_iova) as u32))
                .build(),
        );

        // Weight decompression control (TRM: RKNN_cna_dcomp_ctrl)
        // Bypass decompression for uncompressed weights
        cmds.push(
            Register::<CnaDcompCtrl>::new()
                .wt_dec_bypass(Bits::new(1))
                .decomp_control(Bits::new(0))
                .build(),
        );

        // ====================================================================
        // STEP 5: CNA Convolution Configuration
        // ====================================================================

        // Padding (TRM: RKNN_cna_pad_con0)
        cmds.push(
            Register::<CnaPadCon0>::new()
                .pad_top(Bits::new(self.pad_top))
                .pad_left(Bits::new(self.pad_left))
                .build(),
        );

        // Convolution control 1 (TRM: RKNN_cna_conv_con1)
        cmds.push(
            Register::<CnaConvCon1>::new()
                .conv_mode(Bits::new(self.conv_mode as u32))
                .in_precision(Bits::new(self.in_precision.as_u32()))
                .proc_precision(Bits::new(self.proc_precision.as_u32()))
                .build(),
        );

        // Convolution control 2 (TRM: RKNN_cna_conv_con2)
        // feature_grains: rows to buffer before conv starts
        // For 1x1: stride(1) + height(1) + 1 = 3
        let kernel_group = match self.proc_precision {
            Precision::Int8 => (self.out_channels + 31) / 32 - 1, // 32 kernels per group
            _ => (self.out_channels + 15) / 16 - 1,               // 16 kernels per group
        };

        cmds.push(
            Register::<CnaConvCon2>::new()
                .feature_grains(Bits::new(3))
                .kernel_group(Bits::new(kernel_group))
                .csc_do_en(Bits::new(0)) // Enable data scan
                .csc_wo_en(Bits::new(0)) // Enable weight scan
                .build(),
        );

        // Convolution control 3: stride and dilation (TRM: RKNN_cna_conv_con3)
        cmds.push(
            Register::<CnaConvCon3>::new()
                .conv_x_stride(Bits::new(1))
                .conv_y_stride(Bits::new(1))
                .atrous_x_dilation(Bits::new(0))
                .atrous_y_dilation(Bits::new(0))
                .nn_mode(Bits::new(0)) // Single core mode
                .build(),
        );

        // ====================================================================
        // STEP 6: CNA Internal Buffer Configuration (CBUF)
        // ====================================================================

        // Buffer bank allocation (TRM: RKNN_cna_cbuf_con0)
        // Bank 0 for feature data, Bank 7 for weights
        cmds.push(
            Register::<CnaCbufCon0>::new()
                .data_bank(Bits::new(0)) // Bank 0 for feature
                .weight_bank(Bits::new(1)) // Bank 7 for weight
                .data_reuse(Bits::new(0)) // No data reuse
                .weight_reuse(Bits::new(0)) // No weight reuse
                .build(),
        );

        // Data entries (TRM: RKNN_cna_cbuf_con1)
        // Number of bank spaces for one feature map row
        cmds.push(
            Register::<CnaCbufCon1>::new()
                .data_entries(Bits::new(1))
                .build(),
        );

        // ====================================================================
        // STEP 7: CNA Input Conversion (Optional)
        // ====================================================================

        if self.enable_cvt {
            // Enable input data conversion
            cmds.push(
                Register::<CnaCvtCon0>::new()
                    .cvt_bypass(Bits::new(0))
                    .cvt_type(Bits::new(0)) // Multiply first, then add
                    .round_type(Bits::new(1)) // Round up 0.5 to 1
                    .data_sign(Bits::new(1)) // Signed data
                    .build(),
            );
            // Note: You'd add scale/offset registers (CvtCon1-4) if needed
        } else {
            // Bypass input conversion
            cmds.push(
                Register::<CnaCvtCon0>::new()
                    .cvt_bypass(Bits::new(1))
                    .build(),
            );
        }

        // ====================================================================
        // STEP 8: CNA Clock Gating (Optional for power)
        // ====================================================================

        cmds.push(
            Register::<CnaClkGate>::new()
                .cna_feature_disable_clkgate(Bits::new(0))
                .cna_weight_disable_clkgate(Bits::new(0))
                .csc_disable_clkgate(Bits::new(0))
                .cbuf_cs_disable_clkgate(Bits::new(0))
                .build(),
        );

        // CNA operation enable - triggers CNA to start
        cmds.push(
            Register::<CnaOperationEnable>::new()
                .op_en(Bits::new(1))
                .build(),
        );

        // ====================================================================
        // STEP 9: CORE Configuration (MAC Array)
        // ====================================================================

        // Output dimensions (TRM: RKNN_core_dataout_size_0)
        cmds.push(
            Register::<CoreDataoutSize0>::new()
                .dataout_width(Bits::new(self.width))
                .dataout_height(Bits::new(self.height))
                .build(),
        );

        // Output channels (TRM: RKNN_core_dataout_size_1)
        // Hardware expects "minus 1"
        cmds.push(
            Register::<CoreDataoutSize1>::new()
                .dataout_channel(Bits::new(self.out_channels - 1))
                .build(),
        );

        // Processing configuration (TRM: RKNN_core_misc_cfg)
        let is_depthwise = matches!(self.conv_mode, ConvMode::Depthwise);
        cmds.push(
            Register::<CoreMiscCfg>::new()
                .proc_precision(Bits::new(self.proc_precision.as_u32()))
                .dw_en(Bits::new(if is_depthwise { 1 } else { 0 }))
                .qd_en(Bits::new(0)) // No quantization
                .build(),
        );

        // Truncation/rounding (TRM: RKNN_core_clip_truncate)
        cmds.push(
            Register::<CoreClipTruncate>::new()
                .clip_truncate(Bits::new(0))
                .round_type(Bits::new(1))
                .build(),
        );

        // CORE operation enable
        cmds.push(
            Register::<CoreOperationEnable>::new()
                .op_en(Bits::new(1))
                .build(),
        );

        // ====================================================================
        // STEP 10: DPU Configuration (Data Processing Unit)
        // ====================================================================

        // Feature mode: where data comes from and goes to
        // (TRM: RKNN_dpu_feature_mode_cfg)
        cmds.push(
            Register::<DpuFeatureModeCfg>::new()
                .flying_mode(Bits::new(0)) // Data from convolution (not RDMA)
                .output_mode(Bits::new(self.output_mode as u32))
                .conv_mode(Bits::new(self.conv_mode as u32))
                .burst_len(Bits::new(7)) // Burst8
                .nonalign(Bits::new(0))
                .build(),
        );

        // Data format and precision (TRM: RKNN_dpu_data_format)
        cmds.push(
            Register::<DpuDataFormat>::new()
                .proc_precision(Bits::new(self.proc_precision.as_u32()))
                .in_precision(Bits::new(self.in_precision.as_u32()))
                .out_precision(Bits::new(self.out_precision.as_u32()))
                .mc_surf_out(Bits::new(0)) // Single surface output
                .build(),
        );

        // Output address (TRM: RKNN_dpu_dst_base_addr)
        // Bits 31:4 used, so shift right by 4
        cmds.push(
            Register::<DpuDstBaseAddr>::new()
                .dst_base_addr(Bits::new((self.dst_iova) as u32))
                .build(),
        );

        // Output stride (TRM: RKNN_dpu_dst_surf_stride)
        let out_bpe = self.out_precision.bytes_per_element();
        let out_line_stride = self.width * self.out_channels * out_bpe;
        let out_surf_stride = self.height * out_line_stride;

        cmds.push(
            Register::<DpuDstSurfStride>::new()
                .dst_surf_stride(Bits::new(out_surf_stride >> 4))
                .build(),
        );

        // Output cube dimensions (TRM: RKNN_dpu_data_cube_*)
        // Hardware expects "minus 1" for width/height/channel
        cmds.push(
            Register::<DpuDataCubeWidth>::new()
                .width(Bits::new(self.width - 1))
                .build(),
        );

        cmds.push(
            Register::<DpuDataCubeHeight>::new()
                .height(Bits::new(self.height - 1))
                .minmax_ctl(Bits::new(0))
                .build(),
        );

        cmds.push(
            Register::<DpuDataCubeChannel>::new()
                .channel(Bits::new(self.out_channels - 1))
                .orig_channel(Bits::new(self.out_channels))
                .build(),
        );

        // ====================================================================
        // STEP 11: DPU Bypass Stages (No extra ops for basic convolution)
        // ====================================================================

        // Bypass Batch Scale (BS) core (TRM: RKNN_dpu_bs_cfg)
        cmds.push(Register::<DpuBsCfg>::new().bs_bypass(Bits::new(1)).build());

        // Bypass Batch Normalization (BN) core (TRM: RKNN_dpu_bn_cfg)
        cmds.push(Register::<DpuBnCfg>::new().bn_bypass(Bits::new(1)).build());

        // Bypass Element-Wise (EW) core (TRM: RKNN_dpu_ew_cfg)
        cmds.push(Register::<DpuEwCfg>::new().ew_bypass(Bits::new(1)).build());

        // Output conversion (bypass if not needed)
        cmds.push(
            Register::<DpuOutCvtOffset>::new()
                .out_cvt_offset(Bits::new(0))
                .build(),
        );

        cmds.push(
            Register::<DpuOutCvtScale>::new()
                .out_cvt_scale(Bits::new(0))
                .fp32tofp16_en(Bits::new(0))
                .build(),
        );

        cmds.push(
            Register::<DpuOutCvtShift>::new()
                .out_cvt_shift(Bits::new(0))
                .minus_exp(Bits::new(0))
                .cvt_round(Bits::new(1))
                .cvt_type(Bits::new(0))
                .build(),
        );

        // ====================================================================
        // STEP 12: DPU Write DMA Configuration
        // ====================================================================

        // WDMA size configuration (TRM: RKNN_dpu_wdma_size_*)
        cmds.push(
            Register::<DpuWdmaSize0>::new()
                .channel_wdma(Bits::new(self.out_channels - 1))
                .size_c_wdma(Bits::new(self.out_channels - 1))
                .tp_precision(Bits::new(if out_bpe == 1 { 0 } else { 1 }))
                .build(),
        );

        cmds.push(
            Register::<DpuWdmaSize1>::new()
                .width_wdma(Bits::new(self.width - 1))
                .height_wdma(Bits::new(self.height - 1))
                .build(),
        );

        // DPU operation enable
        cmds.push(
            Register::<DpuOperationEnable>::new()
                .op_en(Bits::new(1))
                .build(),
        );

        // ====================================================================
        // STEP 13: Global Enable - Start All Blocks
        // ====================================================================

        cmds.push(RegCmd::new_raw(0x0041000000000000));

        // This triggers the entire pipeline to start executing
        // (TRM: RKNN_global_operation_enable)
        cmds.push(RegCmd::new(0x81, REG_PC_OPERATION_ENABLE, unsafe {
            PC_OPERATION_ENABLE_RESERVED_0(14) | PC_OPERATION_ENABLE_OP_EN(1)
        }));

        cmds
    }
}
