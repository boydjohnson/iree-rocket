//! Shared `/dev/accel/accel0` primitives -- GEM buffer creation (CREATE_BO
//! + mmap), job submission, and the CPU/device sync ioctls (PREP_BO/
//! FINI_BO). Previously duplicated identically across `rkt-basic.rs`/
//! `rkt-job.rs`/`rkt-simple-job.rs`; factored out here so out-of-tree
//! consumers (`rocket-hal-driver`) can use the same primitives instead of
//! re-deriving them.

use std::{num::NonZeroUsize, ptr::NonNull};

use nix::{
    ioctl_readwrite, ioctl_write_ptr,
    sys::mman::{MapFlags, ProtFlags, mmap, munmap},
    time::{ClockId, clock_gettime},
};

use crate::rocket::api::{
    DRM_COMMAND_BASE, DRM_IOCTL_BASE, DRM_ROCKET_CREATE_BO, DRM_ROCKET_FINI_BO, DRM_ROCKET_PREP_BO,
    DRM_ROCKET_SUBMIT, drm_gem_close, drm_rocket_create_bo, drm_rocket_fini_bo, drm_rocket_job,
    drm_rocket_prep_bo, drm_rocket_submit, drm_rocket_task,
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
// GEM_CLOSE is a generic (core DRM, not rocket-specific) ioctl -- nr=0x09
// per every DRM driver's shared <drm/drm.h>, NOT offset by
// DRM_COMMAND_BASE the way every other ioctl in this module is (those are
// all rocket-driver-specific commands). No client of this crate has ever
// called this before this session -- every `Buffer::new()` GEM allocation
// in every test/bin in this repo has leaked its handle for the lifetime of
// the process, which turned out to matter for tests doing many repeated
// allocations in one process (see `close_bo`'s doc comment).
ioctl_write_ptr!(gem_close, DRM_IOCTL_BASE, 0x09, drm_gem_close);

pub struct Buffer {
    pub handle: u32,
    pub dma_address: u32,
    pub size: usize,
    pub host_ptr: *mut u8,
}

impl Buffer {
    pub unsafe fn new(fd: i32, size: usize, file: impl std::os::fd::AsFd) -> Self {
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

/// Submits a multi-task job -- tasks run in order, on the same core,
/// dispatched task-by-task only after each prior task's own hardware
/// completion IRQ fires (mainline `rocket_job.c`'s `rocket_job_run()`
/// dispatches just the first task; `rocket_job_handle_irq()` re-enters
/// `rocket_job_hw_submit()` for the next one once `next_task_idx <
/// task_count`, only signaling the job's `done_fence` after the last
/// task completes) -- a real, kernel-implemented mechanism
/// (`drm_rocket_job.task_count`/`.tasks` already support an array; no
/// uapi struct changes needed), just never exercised by this crate
/// before `build_conv_then_lut_regcmd` needed it.
///
/// `in_handles`/`out_handles` are job-wide, not per-task (matches
/// `drm_rocket_job`'s own field layout: one flat GEM-handle list covers
/// every task in the job) -- must include every buffer any task reads
/// from or writes to, including each task's own regcmd buffer handle in
/// `in_handles`. A buffer that only tasks *within this job* touch (never
/// read/written by the CPU, e.g. one task's output feeding the next
/// task's input) does not need to appear in either list -- there is no
/// kernel-mediated sync point between tasks in the same job for the CPU
/// to wait on anyway (only real hardware writes + the completion IRQ
/// mediate visibility from one task to the next).
pub unsafe fn submit_tasks(
    fd: i32,
    tasks: &[(u32, u32)], // (regcmd_dma_address, regcmd_count) per task, in order
    in_handles: &[u32],
    out_handles: &[u32],
) -> nix::Result<()> {
    let raw_tasks: Vec<drm_rocket_task> = tasks
        .iter()
        .map(|&(regcmd, regcmd_count)| drm_rocket_task {
            regcmd,
            regcmd_count,
        })
        .collect();
    let job = drm_rocket_job {
        tasks: raw_tasks.as_ptr() as u64,
        in_bo_handles: in_handles.as_ptr() as u64,
        out_bo_handles: out_handles.as_ptr() as u64,
        task_count: raw_tasks.len() as u32,
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
    unsafe {
        rocket_submit(fd, &mut submit)?;
    }
    Ok(())
}

/// One job in a multi-job submission: its task list plus the GEM handles it
/// reads from and writes to.
pub struct JobDesc<'a> {
    /// `(regcmd_dma_address, regcmd_count)` per task, in order.
    pub tasks: &'a [(u32, u32)],
    pub in_handles: &'a [u32],
    pub out_handles: &'a [u32],
}

/// Submits several independent jobs in one SUBMIT ioctl.
///
/// This is the difference that matters for multi-core execution. Tasks
/// within one job run sequentially on a single core (see [`submit_tasks`]),
/// so N tasks buy capacity, not parallelism. N *jobs* are independent work
/// items the kernel scheduler can place on different cores -- that is the
/// only lever userspace has, because `drm_rocket_job` has no core field and
/// core assignment is entirely the kernel's.
///
/// Whether the scheduler actually spreads them is not observable from
/// userspace: a successful submit says nothing about placement. Confirming
/// it needs the out-of-tree kprobe tracer's per-core operation counts.
///
/// Jobs submitted together have no ordering guarantee relative to each
/// other, so they must not depend on each other's output. For height-tiled
/// convolution the tiles write disjoint output rows, which satisfies that.
///
/// # Safety
///
/// `fd` must be an open `/dev/accel/accel0` descriptor, every task's
/// `regcmd_dma_address` must point at a live GEM buffer holding
/// `regcmd_count` valid register commands, and every handle listed must
/// name a live GEM buffer. The hardware reads and writes those buffers
/// asynchronously, so they must stay alive and untouched by the CPU until
/// the job completes.
pub unsafe fn submit_jobs(fd: i32, jobs: &[JobDesc<'_>]) -> nix::Result<()> {
    assert!(!jobs.is_empty(), "submit_jobs requires at least one job");

    // Each job's `tasks` pointer must stay valid for the ioctl, so the
    // converted task arrays are kept alive here rather than in a temporary.
    let raw_tasks: Vec<Vec<drm_rocket_task>> = jobs
        .iter()
        .map(|job| {
            job.tasks
                .iter()
                .map(|&(regcmd, regcmd_count)| drm_rocket_task {
                    regcmd,
                    regcmd_count,
                })
                .collect()
        })
        .collect();

    let raw_jobs: Vec<drm_rocket_job> = jobs
        .iter()
        .zip(&raw_tasks)
        .map(|(job, tasks)| drm_rocket_job {
            tasks: tasks.as_ptr() as u64,
            in_bo_handles: job.in_handles.as_ptr() as u64,
            out_bo_handles: job.out_handles.as_ptr() as u64,
            task_count: tasks.len() as u32,
            task_struct_size: std::mem::size_of::<drm_rocket_task>() as u32,
            in_bo_handle_count: job.in_handles.len() as u32,
            out_bo_handle_count: job.out_handles.len() as u32,
        })
        .collect();

    let mut submit = drm_rocket_submit {
        jobs: raw_jobs.as_ptr() as u64,
        job_count: raw_jobs.len() as u32,
        job_struct_size: std::mem::size_of::<drm_rocket_job>() as u32,
        reserved: 0,
    };
    unsafe {
        rocket_submit(fd, &mut submit)?;
    }
    Ok(())
}

/// Submits a single-task job built from an already-written regcmd GEM
/// buffer. `in_handles`/`out_handles` are the GEM handles the job reads
/// from/writes to -- must include the regcmd buffer's own handle in
/// `in_handles` (the kernel needs it retained for the job's duration).
pub unsafe fn submit(
    fd: i32,
    regcmd_dma_address: u32,
    regcmd_count: u32,
    in_handles: &[u32],
    out_handles: &[u32],
) -> nix::Result<()> {
    unsafe {
        submit_tasks(
            fd,
            &[(regcmd_dma_address, regcmd_count)],
            in_handles,
            out_handles,
        )
    }
}

pub unsafe fn fini_bo(fd: i32, handle: u32) -> nix::Result<()> {
    unsafe {
        rocket_fini_bo(
            fd,
            &drm_rocket_fini_bo {
                handle,
                reserved: 0,
            },
        )?;
    }
    Ok(())
}

/// `rocket_gem.c`'s `rocket_ioctl_prep_bo()` converts `timeout_ns` via
/// `drm_timeout_abs_to_jiffies()` -- that takes an ABSOLUTE
/// `CLOCK_MONOTONIC` deadline, not a relative duration (standard DRM
/// wait-ioctl convention). See rknpu-spelunking/NOTES.md for the full
/// story of the bug this caused when every one of this project's own
/// test clients got it wrong. `relative_timeout_ns` here is what callers
/// actually want to express ("wait up to N ns from now").
pub unsafe fn prep_bo(fd: i32, handle: u32, relative_timeout_ns: u64) -> nix::Result<()> {
    let now = clock_gettime(ClockId::CLOCK_MONOTONIC).expect("clock_gettime failed");
    let now_ns = now.tv_sec() as u64 * 1_000_000_000 + now.tv_nsec() as u64;
    unsafe {
        rocket_prep_bo(
            fd,
            &drm_rocket_prep_bo {
                handle,
                reserved: 0,
                timeout_ns: (now_ns + relative_timeout_ns) as i64,
            },
        )?;
    }
    Ok(())
}

/// Releases a GEM handle (`DRM_IOCTL_GEM_CLOSE`) -- the counterpart
/// `Buffer::new()` never had. Added after a real hardware investigation
/// (rknpu-spelunking's ukernel roadmap, Phase 0 pooling reconciliation)
/// found that test functions calling `Buffer::new()` many times in a row
/// within one process (each call leaking its handle -- nothing in this
/// crate has ever closed one before) started producing corrupted/stale
/// dispatch results after several repeated allocations, while the exact
/// same dispatch run once in a fresh process was clean. Does NOT unmap the
/// buffer's `host_ptr` -- callers that also `mmap`'d (i.e. everyone using
/// `Buffer::new()`) must call [`unmap_bo`] separately to release the VMA's
/// GEM-object reference; this only releases the file's GEM handle.
pub unsafe fn close_bo(fd: i32, handle: u32) -> nix::Result<()> {
    unsafe {
        gem_close(fd, &drm_gem_close { handle, pad: 0 })?;
    }
    Ok(())
}

/// Releases the CPU mapping created by [`Buffer::new`].
///
/// This is separate from [`close_bo`]: GEM_CLOSE drops the file's handle,
/// while `munmap` drops the VMA's independent reference to the GEM object.
/// Call both once no CPU pointer into the buffer remains. Otherwise the GEM
/// object, its IOMMU domain reference, and its IOVA allocation remain alive
/// until process exit even though the handle has been closed.
pub unsafe fn unmap_bo(buffer: &Buffer) -> nix::Result<()> {
    let mapping =
        NonNull::new(buffer.host_ptr.cast()).expect("Buffer::new returned a null mapping");
    unsafe { munmap(mapping, buffer.size) }
}
