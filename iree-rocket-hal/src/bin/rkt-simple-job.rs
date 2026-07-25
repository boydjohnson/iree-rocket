//! NPU diagnostic: 1x1 conv, 224x224 spatial, 24 input channels, 24 output
//! channels, stride 1, no padding, no depthwise. Regcmd construction lives
//! in `iree_rocket_hal::rocket::regcmd` (shared with rkt-basic.rs/
//! rkt-job.rs) -- this file previously carried its own independent (and
//! substantially different/incomplete -- PCInterruptClear/PCInterruptMask
//! writes that don't correspond to anything in Mesa's real driver, a
//! `DOMAIN_GLOBAL` broadcast kick pattern superseded by the real PC-based
//! kick sequence, no biases buffer, etc.) regcmd builder. See that
//! module's doc comment and rknpu-spelunking/NOTES.md for the full
//! derivation/validation history.

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
    ioctl_readwrite,
    sys::mman::{MapFlags, ProtFlags, mmap},
    time::{ClockId, clock_gettime},
};
use std::{fs::OpenOptions, marker::PhantomData, num::NonZeroUsize, os::fd::AsRawFd};

// rocket_gem.c's rocket_ioctl_prep_bo() converts timeout_ns via
// drm_timeout_abs_to_jiffies() -- that takes an ABSOLUTE CLOCK_MONOTONIC
// deadline, not a relative duration (standard DRM wait-ioctl convention;
// see rkt-basic.rs / NOTES.md for the full story). A bare literal like
// `1_000_000_000` is interpreted as "1 second after the monotonic clock's
// zero point", always already in the past on a booted system, so the wait
// returns immediately with -EBUSY regardless of whether the job actually
// completes.
fn abs_timeout_ns(relative_ns: u64) -> i64 {
    let now = clock_gettime(ClockId::CLOCK_MONOTONIC).expect("clock_gettime failed");
    let now_ns = now.tv_sec() as u64 * 1_000_000_000 + now.tv_nsec() as u64;
    (now_ns + relative_ns) as i64
}

fn with_cpu_access<F>(fd: i32, buf: &Buffer, label: &str, mut f: F)
where
    F: FnMut(&mut [u8]),
{
    let mut prep = drm_rocket_prep_bo {
        handle: buf.handle,
        reserved: 0,
        timeout_ns: abs_timeout_ns(1_000_000_000),
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
        Self {
            data,
            _phantom: PhantomData,
        }
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
    fn bytes_per_element() -> usize {
        1
    }
}

fn main() {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/accel/accel0")
        .expect("Failed to open device");
    let fd = file.as_raw_fd();

    println!("--- NPU Diagnostic: 32x32x24 1x1 Conv (shared regcmd builder) ---");

    // Originally 224x224x24 -- that needs 49 CBUF input banks (only 12
    // exist total), which requires Mesa's multi-task splitting
    // (rkt_split_tasks()'s general branch, not just its single-task
    // shortcut). build_conv_regcmd() only implements the single-task
    // path (see its own scope doc comment) and now asserts loudly on
    // this rather than silently underflowing into a confusing panic
    // somewhere unrelated. Shrunk to 32x32 spatial, which fits in 1 bank,
    // while keeping 24 channels as this file's distinguishing shape from
    // rkt-job.rs's 64x64x64.
    let tensor_size = 32 * 32 * 24;

    unsafe {
        let buf_a = Buffer::new(fd, tensor_size, &file);
        let buf_w = Buffer::new(fd, tensor_size, &file);
        let buf_c = Buffer::new(fd, tensor_size, &file);
        // DPU's BS (bias-subtract) block is never actually bypassed by
        // Mesa's real driver -- it always runs its ALU against a real
        // biases buffer, so this has to exist and be zero-filled even
        // though this op has no logical bias.
        let buf_bias = Buffer::new(fd, tensor_size, &file);
        let buf_cmd = Buffer::new(fd, 4096, &file);

        println!(
            "Buffers:\n  A@{:#x}\n  W@{:#x}\n  Bias@{:#x}\n  C@{:#x}\n  CMD@{:#x}",
            buf_a.dma_address,
            buf_w.dma_address,
            buf_bias.dma_address,
            buf_c.dma_address,
            buf_cmd.dma_address
        );

        fill_buffer::<i8>(fd, &buf_a, "fill A", 10);
        fill_buffer::<i8>(fd, &buf_w, "fill W", 2);
        fill_buffer::<i8>(fd, &buf_c, "zero C", 0);
        fill_buffer::<i8>(fd, &buf_bias, "zero Bias", 0);

        let shape = ConvShape {
            input_width: 32,
            input_height: 32,
            input_channels: 24,
            output_width: 32,
            output_height: 32,
            output_channels: 24,
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
            precision: Precision::Int8,
        };
        let bufs = ConvBuffers {
            input_addr: buf_a.dma_address,
            weights_addr: buf_w.dma_address,
            bias_addr: buf_bias.dma_address,
            output_addr: buf_c.dma_address,
        };
        let cmds = build_conv_regcmd(&shape, &bufs);
        dump_cmds("rkt-simple-job", &cmds);

        with_cpu_access(fd, &buf_cmd, "write CMD", |raw| {
            let cmd_slice =
                std::slice::from_raw_parts_mut(raw.as_mut_ptr() as *mut u64, cmds.len());
            for (i, c) in cmds.iter().enumerate() {
                cmd_slice[i] = c.0;
            }
        });

        let regcmd_count_val = cmds.len() as u32;

        let task = drm_rocket_task {
            regcmd: buf_cmd.dma_address,
            regcmd_count: regcmd_count_val,
        };

        let in_handles = vec![buf_cmd.handle, buf_a.handle, buf_w.handle, buf_bias.handle];
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
            timeout_ns: abs_timeout_ns(2_000_000_000),
        };
        rocket_prep_bo(fd, &mut prep_out).expect("Job timeout or NPU error (EBUSY)");

        let result = std::slice::from_raw_parts(buf_c.host_ptr, buf_c.size as usize);
        println!("Result[0]: {}", result[0]);
    }
}
