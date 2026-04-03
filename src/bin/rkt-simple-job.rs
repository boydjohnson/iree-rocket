use std::{fs::OpenOptions, marker::PhantomData, num::NonZeroUsize, os::fd::AsRawFd};
use iree_rocket_hal::rocket::builders;
use iree_rocket_hal::rocket::{
    api::{
        DRM_COMMAND_BASE, DRM_IOCTL_BASE, DRM_ROCKET_CREATE_BO, DRM_ROCKET_FINI_BO,
        DRM_ROCKET_PREP_BO, DRM_ROCKET_SUBMIT, drm_rocket_create_bo, drm_rocket_fini_bo,
        drm_rocket_job, drm_rocket_prep_bo, drm_rocket_submit, drm_rocket_task,
    },
    builders::{
        Bits, RegCmd, Register,
        cna::{
            CnaCbufCon0, CnaCbufCon1, CnaConvCon1, CnaConvCon2, CnaConvCon3,
            CnaCvtCon0, CnaCvtCon1, CnaDataSize0, CnaDataSize1, CnaDataSize2, CnaDataSize3, CnaDcompAddr0,
            CnaDcompCtrl, CnaDmaCon0, CnaDmaCon1, CnaDmaCon2, CnaFeatureDataAddr,
            CnaOperationEnable, CnaPadCon0, CnaSPointer, CnaWeightSize0, CnaWeightSize1,
            CnaWeightSize2,
        },
        core::{
            CoreDataoutSize0, CoreDataoutSize1, CoreMiscCfg, CoreOperationEnable,
        },
        dpu::{
            DpuBnCfg, DpuBsCfg, DpuDataCubeChannel, DpuDataCubeHeight, DpuDataCubeWidth,
            DpuDataFormat, DpuDstBaseAddr, DpuDstSurfStride, DpuEwCfg, DpuFeatureModeCfg,
            DpuOperationEnable, DpuOutCvtOffset, DpuOutCvtScale, DpuOutCvtShift,
            DpuWdmaSize0, DpuWdmaSize1,
        },
        global::GlobalOperationEnable,
        pc::{PCBaseAddress, PCInterruptClear, PCInterruptMask, PCOperationEnable},
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
    let mut prep = drm_rocket_prep_bo {
        handle: buf.handle,
        reserved: 0,
        timeout_ns: 1_000_000_000,
    };

    println!("PREP_BO ({})", label);
    unsafe {
        rocket_prep_bo(fd, &mut prep).expect("PREP_BO failed");
    }

    let slice = unsafe { std::slice::from_raw_parts_mut(buf.host_ptr, buf.size as usize) };
    f(slice);

    let mut fini = drm_rocket_fini_bo {
        handle: buf.handle,
        reserved: 0,
    };

    println!("FINI_BO ({})", label);
    unsafe {
        rocket_fini_bo(fd, &mut fini).expect("FINI_BO failed");
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

fn fill_buffer<T: NpuDataType>(fd: i32, buf: &Buffer, label: &str, value: T) {
    with_cpu_access(fd, buf, label, |slice| {
        let mut view = unsafe { BufferView::<T>::new(slice) };
        view.fill(value);
    });
}

pub struct BufferView<'a, T: NpuDataType> {
    data: &'a mut [u8],
    _phantom: PhantomData<T>,
}

impl<'a, T: NpuDataType> BufferView<'a, T> {
    pub unsafe fn new(data: &'a mut [u8]) -> Self {
        assert_eq!(data.len() % T::bytes_per_element(), 0);
        Self { data, _phantom: PhantomData }
    }

    pub fn as_slice(&mut self) -> &mut [T] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.data.as_mut_ptr() as *mut T,
                self.data.len() / T::bytes_per_element(),
            )
        }
    }

    pub fn fill(&mut self, value: T) {
        self.as_slice().fill(value);
    }
}

pub trait NpuDataType: Copy {
    fn bytes_per_element() -> usize;
}

impl NpuDataType for i8 {
    fn bytes_per_element() -> usize { 1 }
}

fn main() {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/accel/accel0")
        .expect("Failed to open device");
    let fd = file.as_raw_fd();

    println!("--- NPU Diagnostic: Fixed CnaCore1x1Job ---");

    let tensor_size = 224 * 224 * 24;

    unsafe {
        let buf_a = Buffer::new(fd, tensor_size, &file);
        let buf_w = Buffer::new(fd, tensor_size, &file);
        let buf_c = Buffer::new(fd, tensor_size, &file);
        let buf_cmd = Buffer::new(fd, 4096, &file);

        println!(
            "Buffers:\n  A@{:#x}\n  W@{:#x}\n  C@{:#x}\n  CMD@{:#x}",
            buf_a.dma_address, buf_w.dma_address, buf_c.dma_address, buf_cmd.dma_address
        );

        fill_buffer::<i8>(fd, &buf_a, "fill A", 10);
        fill_buffer::<i8>(fd, &buf_w, "fill W", 2);
        fill_buffer::<i8>(fd, &buf_c, "zero C", 0);

        let job_desc = CnaCore1x1Job {
            width: 224, height: 224, in_channels: 24, out_channels: 24,
            dst_iova: buf_c.dma_address as u64,
            src_iova: buf_a.dma_address as u64,
            weight_iova: buf_w.dma_address as u64,
        };

        let cmds = job_desc.build_regcmds();

        with_cpu_access(fd, &buf_cmd, "write CMD", |raw| {
            let cmd_slice = std::slice::from_raw_parts_mut(raw.as_mut_ptr() as *mut u64, cmds.len());
            for (i, c) in cmds.iter().enumerate() {
                cmd_slice[i] = c.0;
            }
        });

        // FIX: Multiply by 2 to satisfy kernel formula (N+1)/2 - 1
        let regcmd_count_val = cmds.len() as u32 * 2;

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

        let mut prep_out = drm_rocket_prep_bo {
            handle: buf_c.handle,
            reserved: 0,
            timeout_ns: 2_000_000_000,
        };
        rocket_prep_bo(fd, &mut prep_out).expect("Job timeout or NPU error (EBUSY)");

        let result = std::slice::from_raw_parts(buf_c.host_ptr, buf_c.size as usize);
        println!("Result[0]: {}", result[0]);

        if result[0] == 20 {
            println!("✅ SUCCESS!");
        } else {
            println!("❌ FAILED - Expected 20, got {}", result[0]);
        }
    }
}

pub struct CnaCore1x1Job {
    width: u32, height: u32,
    in_channels: u32, out_channels: u32,
    dst_iova: u64, src_iova: u64, weight_iova: u64,
}

impl CnaCore1x1Job {
    pub fn build_regcmds(&self) -> Vec<RegCmd> {
        let mut cmds = Vec::new();

        // 1. Setup PC and Interrupts
        cmds.push(Register::<PCBaseAddress>::new().pc_sel(true).build());
        cmds.push(Register::<PCInterruptClear>::new().dpu_0(true).core_0(true).cna_feature_0(true).build());
        cmds.push(Register::<PCInterruptMask>::new().cna_feature_0(true).core_0(true).dpu_0(true).build());

        // 2. CNA Config
        cmds.push(Register::<CnaDataSize0>::new().datain_width(Bits::new(self.width)).datain_height(Bits::new(self.height)).build());
        cmds.push(Register::<CnaDataSize1>::new().datain_channel(Bits::new(self.in_channels - 1)).datain_channel_real(Bits::new(self.in_channels)).build());
        cmds.push(Register::<CnaDataSize2>::new().dataout_width(Bits::new(self.width)).build());
        cmds.push(Register::<CnaDataSize3>::new().dataout_atomics(Bits::new(self.width * self.height)).build());

        let line_stride = self.width * self.in_channels;
        cmds.push(Register::<CnaDmaCon1>::new().line_stride(Bits::new(line_stride)).build());
        cmds.push(Register::<CnaDmaCon2>::new().surf_stride(Bits::new(self.height * line_stride)).build());
        cmds.push(Register::<CnaDmaCon0>::new().data_burst_len(Bits::new(7)).weight_burst_len(Bits::new(7)).build());
        cmds.push(Register::<CnaFeatureDataAddr>::new().feature_base_addr(Bits::new(self.src_iova as u32)).build());

        // 3. Weights
        let weight_bytes = self.in_channels * self.out_channels;
        cmds.push(Register::<CnaWeightSize0>::new().weight_bytes(Bits::new(weight_bytes)).build());
        cmds.push(Register::<CnaWeightSize1>::new().weight_bytes_per_kernel(Bits::new(self.in_channels)).build());
        cmds.push(Register::<CnaWeightSize2>::new().weight_width(Bits::new(1)).weight_height(Bits::new(1)).weight_kernels(Bits::new(self.out_channels)).build());
        cmds.push(Register::<CnaDcompAddr0>::new().decompress_addr0(Bits::new(self.weight_iova as u32)).build());
        cmds.push(Register::<CnaDcompCtrl>::new().wt_dec_bypass(Bits::new(1)).build());

        // 4. Convolution Mode & Pipeline Kick
        cmds.push(Register::<CnaConvCon1>::new().conv_mode(Bits::new(0)).in_precision(Bits::new(0)).proc_precision(Bits::new(0)).build());
        cmds.push(Register::<CnaConvCon2>::new().feature_grains(Bits::new(3)).kernel_group(Bits::new((self.out_channels + 31) / 32 - 1)).build());
        cmds.push(Register::<CnaConvCon3>::new().conv_x_stride(Bits::new(1)).conv_y_stride(Bits::new(1)).build());
        cmds.push(Register::<CnaCbufCon0>::new().data_bank(Bits::new(0)).weight_bank(Bits::new(1)).build());
        cmds.push(Register::<CnaCbufCon1>::new().data_entries(Bits::new(1)).build());

        // Identity Converters (CNA)
        cmds.push(Register::<CnaCvtCon0>::new().cvt_bypass(Bits::new(0)).build());
        cmds.push(Register::<CnaCvtCon1>::new().cvt_scale0(Bits::new(1)).cvt_offset0(Bits::new(0)).build());

        // Block Kicks
        cmds.push(Register::<CnaOperationEnable>::new().op_en(Bits::new(1)).build());
        cmds.push(Register::<CoreDataoutSize0>::new().dataout_width(Bits::new(self.width)).dataout_height(Bits::new(self.height)).build());
        cmds.push(Register::<CoreDataoutSize1>::new().dataout_channel(Bits::new(self.out_channels - 1)).build());
        cmds.push(Register::<CoreMiscCfg>::new().proc_precision(Bits::new(0)).build());
        cmds.push(Register::<CoreOperationEnable>::new().op_en(Bits::new(1)).build());

        // DPU
        cmds.push(Register::<DpuFeatureModeCfg>::new().output_mode(Bits::new(2)).burst_len(Bits::new(7)).build());
        cmds.push(Register::<DpuDataFormat>::new().proc_precision(Bits::new(0)).in_precision(Bits::new(0)).out_precision(Bits::new(0)).build());
        cmds.push(Register::<DpuDstBaseAddr>::new().dst_base_addr(Bits::new(self.dst_iova as u32)).build());
        cmds.push(Register::<DpuDstSurfStride>::new().dst_surf_stride(Bits::new((self.width * self.out_channels) >> 4)).build());
        cmds.push(Register::<DpuDataCubeWidth>::new().width(Bits::new(self.width - 1)).build());
        cmds.push(Register::<DpuDataCubeHeight>::new().height(Bits::new(self.height - 1)).build());
        cmds.push(Register::<DpuDataCubeChannel>::new().channel(Bits::new(self.out_channels - 1)).orig_channel(Bits::new(self.out_channels)).build());
        cmds.push(Register::<DpuBsCfg>::new().bs_bypass(Bits::new(1)).build());
        cmds.push(Register::<DpuBnCfg>::new().bn_bypass(Bits::new(1)).build());
        cmds.push(Register::<DpuEwCfg>::new().ew_bypass(Bits::new(1)).build());
        cmds.push(Register::<DpuWdmaSize0>::new().channel_wdma(Bits::new(self.out_channels - 1)).size_c_wdma(Bits::new(self.out_channels - 1)).build());
        cmds.push(Register::<DpuWdmaSize1>::new().width_wdma(Bits::new(self.width - 1)).height_wdma(Bits::new(self.height - 1)).build());

        // Identity Converter (DPU)
        cmds.push(Register::<DpuOutCvtScale>::new().out_cvt_scale(Bits::new(1)).build());
        cmds.push(Register::<DpuOperationEnable>::new().op_en(Bits::new(1)).build());

        // 5. GLOBAL KICK (Broad Enable)
        cmds.push(RegCmd::new(builders::DOMAIN_GLOBAL, 0x0008, 0x7F));

        // Final Master Kick via PC
        cmds.push(Register::<PCOperationEnable>::new().op_enable(true).build());

        cmds
    }
}
