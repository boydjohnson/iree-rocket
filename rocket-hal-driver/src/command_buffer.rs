//! `iree_hal_command_buffer_vtable_t`. `dispatch` is the real regcmd
//! integration point: turns the executable's `UkernelShape`
//! (`executable::shape`) plus this dispatch's buffer bindings into a
//! regcmd task list via the matching `iree_rocket_hal::rocket::regcmd::
//! build_*_regcmd` function and stashes it (regcmd tasks + the real GEM
//! handles it touches, see `RecordedOp::Dispatch`) on the command buffer,
//! to be submitted as one job by `device::queue_execute`.
//!
//! Binding convention is per-ukernel-kind: `Conv2d` is binding 0 = input,
//! 1 = weights, 2 = bias, 3 = output; `Pooling` is binding 0 = input,
//! 1 = output (no weights/bias). This started as an arbitrary placeholder
//! of our own choosing (no real compiler emitted a binding layout), but as
//! of `executable_cache.rs`'s tag `3` (a real, versioned wire format for
//! `ConvShape`, see that module's doc comment and
//! `iree_rocket_hal::rocket::executable_format`), it's a **frozen
//! cross-repo ABI contract**: a real IREE compiler `TargetBackend` (not
//! part of this crate, not yet written) would need to emit dispatches
//! respecting this exact binding order, since nothing here validates
//! binding *semantics* -- only binding *count* is checked (see the
//! `bindings.count < 4`/`< 2` guards below).
//!
//! `Conv2d`'s `build_conv_regcmd_tasks` call is wrapped in `catch_unwind` (see
//! below) as a backstop: `executable_cache.rs`'s `validate_conv_shape` is
//! deliberately not proven exhaustive of every panic reachable from
//! `build_conv_regcmd` (some of its internal `assert!`s, and every
//! register-field `Bits::<N>::new` bit-width check, are gated on formulas
//! derived from the shape rather than direct fields) -- without this,
//! any validation gap would abort the whole host process (an unwind
//! crossing a plain `extern "C"` boundary calls `process::abort()` since
//! Rust 1.71, uncatchable as an IREE status) instead of failing this one
//! dispatch gracefully.
//!
//! Supports recording multiple `dispatch` calls per command buffer --
//! `apply_ops` returns every recorded dispatch's regcmd program, in call
//! order, and `device::queue_execute` submits each as its own individually
//! fenced hardware job (the same "submit, then `prep_bo`-wait before the
//! next" sequencing already used for one dispatch's own CBUF-height-split
//! task list, since the mainline driver's inter-task IRQ transition isn't
//! reliable on RK3588 -- see that function's comment). Originally this
//! rejected a second recorded dispatch outright: real compiled programs
//! (a ResNet50 bottleneck block's shortcut projection and its first reduce
//! conv, both independent 1x1 convs reading the same upstream activation)
//! route two independent Rocket dispatches into one stream partition/
//! command buffer whenever nothing forces them apart, so "one only" broke
//! on the first real model exercising that shape, not just a hypothetical.
//! `collective` (multi-device reduce/broadcast/etc.) isn't applicable to a
//! single discrete NPU and stays UNIMPLEMENTED indefinitely, not just for
//! now.
//!
//! `fill_buffer`/`update_buffer`/`copy_buffer` all operate on our own
//! permanently-host-mapped `RocketBuffer`s (see buffer.rs), so they're
//! implemented purely host-side via IREE's own generic
//! `iree_hal_buffer_map_fill`/`_write`/`_copy` helpers (the same primitives
//! `iree/hal/local/inline_command_buffer.c` uses) -- no ioctl/hardware
//! involvement needed. Like `dispatch`, they're *recorded* here (as
//! `RecordedOp` entries, in call order) and only actually applied by
//! `device::queue_execute`'s `apply_ops` call, after the wait-semaphore
//! gate -- executing them immediately at record time (the way IREE's own
//! `ALLOW_INLINE_EXECUTION` mode does) would violate the wait-before-
//! execute contract every other command-buffer category here already
//! relies on. `update_buffer`'s source bytes are copied into the recorded
//! op immediately (the caller's source buffer isn't guaranteed to outlive
//! the recording call), matching `iree_hal_deferred_command_buffer_t`'s
//! identical requirement.

use std::os::fd::{AsRawFd, BorrowedFd, RawFd};

use crate::{
    bindings::{
        iree_const_byte_span_t, iree_device_size_t, iree_hal_buffer_barrier_t,
        iree_hal_buffer_ref_list_t, iree_hal_buffer_ref_t, iree_hal_buffer_t, iree_hal_channel_t,
        iree_hal_collective_op_t, iree_hal_command_buffer_mode_t, iree_hal_command_buffer_t,
        iree_hal_command_buffer_vtable_t, iree_hal_command_category_t, iree_hal_copy_flags_t,
        iree_hal_dispatch_config_t, iree_hal_dispatch_flags_t, iree_hal_event_t,
        iree_hal_executable_function_t, iree_hal_executable_t, iree_hal_execution_barrier_flags_t,
        iree_hal_execution_stage_t, iree_hal_fill_flags_t, iree_hal_label_color_t,
        iree_hal_label_location_t, iree_hal_memory_advise_flags_t, iree_hal_memory_barrier_t,
        iree_hal_queue_affinity_t, iree_hal_update_flags_t, iree_host_size_t, iree_status_t,
        iree_string_view_t,
    },
    buffer::RocketBuffer,
    executable::UkernelShape,
    status,
};
use iree_rocket_hal::rocket::{
    builders::RegCmd,
    conv::{Buffers, ConvPlan, FeatureLayout, Precision},
    device::OwnedBuffer as RocketOwnedBuffer,
    fc,
    pooling::{PoolingBuffers, PoolingPlan},
    tensor_layout::{
        nc1hwc2_storage_size, pack_depthwise_to_rocket_weights, pack_hwcf_to_rocket_weights,
        pack_nhwc_to_nc1hwc2_padded, rocket_weight_storage_size,
    },
};

#[derive(Clone, Copy)]
pub enum InputPackingLayout {
    Dense,
    Nc1hwc2,
}

/// Defers dense-NHWC to NC1HWC2 packing until command-buffer execution.
///
/// The scratch allocation and its DMA address are fixed while recording so
/// the regcmd can be built immediately. The copy itself must happen later,
/// after preceding recorded update/fill/copy operations have populated the
/// real IREE input buffer.
#[derive(Clone, Copy)]
pub struct InputPacking {
    pub input_buffer: *mut iree_hal_buffer_t,
    pub input_offset: iree_device_size_t,
    pub input_length: iree_device_size_t,
    pub scratch_ptr: *mut u8,
    pub scratch_length: usize,
    pub scratch_handle: u32,
    pub source_pixel_count: usize,
    pub packed_pixel_count: usize,
    pub bytes_per_pixel: usize,
    pub packed_bytes_per_pixel: usize,
    pub padding_byte: u8,
    pub layout: InputPackingLayout,
}

/// Defers logical HWCF-to-Rocket coefficient packing until execution.
///
/// The original binding can be populated by earlier recorded operations, so
/// the copy cannot happen while recording the dispatch. The regcmd points at
/// `scratch_handle`; `apply_ops` fills and flushes it immediately before the
/// hardware submission.
#[derive(Clone, Copy)]
pub struct WeightPacking {
    pub weight_buffer: *mut iree_hal_buffer_t,
    pub weight_offset: iree_device_size_t,
    pub weight_length: iree_device_size_t,
    pub scratch_ptr: *mut u8,
    pub scratch_length: usize,
    pub scratch_handle: u32,
    pub filter_height: usize,
    pub filter_width: usize,
    pub input_channels: usize,
    pub output_channels: usize,
    pub element_size: usize,
    /// One filter per input channel (no `(input, output)` pairing) packed
    /// tap-major via [`pack_depthwise_to_rocket_weights`] instead of
    /// [`pack_hwcf_to_rocket_weights`]'s blocked dense order. `output_channels`
    /// is unused in this mode -- Cout is always Cin, per
    /// `iree-rocket-hal`'s `Shape::with_depthwise` -- and `padded_channels`
    /// is read instead.
    pub depthwise: bool,
    /// Tap-major stride, only meaningful when `depthwise` is set. See
    /// `iree-rocket-hal`'s `Shape::depthwise_padded_channels`.
    pub padded_channels: usize,
}

/// Bridges the RK3588 DPU's atomic-slot output write-back (16-byte-aligned
/// slots regardless of dtype, `FEATURE_ATOMIC_SIZE=16`) to IREE's densely-
/// packed ABI output buffer -- see `iree-rocket-hal/src/rocket/conv.rs`'s
/// `Shape::output_scratch_bytes` doc comment and the "Conv2d output
/// compaction" investigation this fixes. `dispatch()` points the regcmd at
/// a driver-private scratch buffer instead of the real output buffer;
/// `queue_execute`, after its existing post-dispatch `prep_bo` wait
/// confirms the hardware write is complete, interleaves the 16-byte channel
/// surfaces into `output_buffer` (the real, dense IREE buffer, retained with
/// every other direct dispatch binding so it survives until `queue_execute`
/// runs).
#[derive(Clone, Copy)]
pub struct OutputCompaction {
    pub output_buffer: *mut iree_hal_buffer_t,
    pub output_offset: iree_device_size_t,
    pub output_length: iree_device_size_t,
    pub scratch_ptr: *mut u8,
    pub scratch_length: usize,
    pub source_pixel_count: usize,
    pub output_pixel_count: usize,
    pub bytes_per_pixel: usize,
}

/// One recorded command-buffer operation, in call order -- see module doc
/// comment for why these are recorded rather than applied immediately.
pub enum RecordedOp {
    Fill {
        target: iree_hal_buffer_ref_t,
        /// Patterns are always <= 8 bytes (`pattern_length` further below
        /// bounds it) -- matches `iree_hal_deferred_command_buffer_t`'s
        /// identical fixed-size inline pattern storage.
        pattern: [u8; 8],
        pattern_length: u8,
    },
    Update {
        target: iree_hal_buffer_ref_t,
        /// Copied out of the caller's `source_buffer` at record time --
        /// that pointer is only guaranteed valid for the duration of the
        /// `update_buffer` call itself.
        source: Vec<u8>,
    },
    Copy {
        source: iree_hal_buffer_ref_t,
        target: iree_hal_buffer_ref_t,
    },
    Dispatch {
        regcmd_tasks: Vec<Vec<RegCmd>>,
        /// Every direct binding supplied to the dispatch, retained exactly
        /// once at record time as required by the IREE HAL command-buffer
        /// contract. The command buffer releases them from `destroy()`.
        retained_bindings: Vec<*mut iree_hal_buffer_t>,
        /// Driver-private GEM allocations whose DMA addresses and host
        /// mappings are referenced by the packing/compaction descriptors and
        /// baked into `regcmd_tasks`. Owning them here keeps them alive until
        /// the command buffer has finished executing and closes/unmaps them
        /// on every destruction path.
        scratch_buffers: Vec<RocketOwnedBuffer>,
        /// GEM handles of every buffer this dispatch reads (bindings other
        /// than the output) -- must be listed in `drm_rocket_job.in_bo_handles`
        /// (device.rs's `queue_execute`) so the kernel driver's implicit
        /// fencing/dependency tracking actually knows the job touches them.
        /// Previously only the regcmd program's own GEM buffer was listed
        /// there, which happened to let SUBMIT/PREP_BO round-trip (proving
        /// the ioctl plumbing itself worked) but never told the kernel
        /// about the real input/weight/bias/output BOs at all -- see any
        /// hand-rolled hardware test in iree-rocket-hal's `tests/`
        /// directory (e.g. `conv_phase1_validation_hw.rs`) for the ioctl
        /// call shape that always did this correctly.
        in_bo_handles: Vec<u32>,
        /// GEM handles of every buffer this dispatch writes.
        out_bo_handles: Vec<u32>,
        /// Set for multi-channel Conv2d dispatches whose dense IREE input
        /// must be packed into NC1HWC2 before the NPU reads it.
        input_packing: Option<InputPacking>,
        /// Set for regular fp16 Conv2d dispatches whose logical HWCF filter
        /// must be packed into the CNA's blocked coefficient order.
        weight_packing: Option<WeightPacking>,
        /// Set only for `Conv2d` -- see `OutputCompaction`'s own doc comment.
        /// `None` for `Pooling` (unaffected today, flagged as a follow-up
        /// risk -- same DPU write-back stage almost certainly has the same
        /// atomic-slot mismatch, just not fixed here).
        output_compaction: Option<OutputCompaction>,
    },
}

/// What `apply_ops` hands back for each recorded `dispatch`, in call order --
/// the regcmd program plus the real BO handles it touches, so
/// `device::queue_execute` can build a correct `drm_rocket_job` instead of
/// submitting with only the regcmd buffer's own handle listed.
pub struct DispatchJob {
    pub regcmd_tasks: &'static [Vec<RegCmd>],
    pub in_bo_handles: &'static [u32],
    pub out_bo_handles: &'static [u32],
    pub output_compaction: Option<OutputCompaction>,
}

/// What every `iree_hal_command_buffer_t*` this driver hands out actually
/// points to. `base` (the real, fully-defined `iree_hal_command_buffer_t`,
/// filled via `iree_hal_command_buffer_initialize`) must be the first
/// field, matching `buffer::RocketBuffer`'s convention.
#[repr(C)]
pub struct RocketCommandBuffer {
    pub base: iree_hal_command_buffer_t,
    /// Every fill/update/copy/dispatch recorded so far, in call order.
    /// `device::queue_execute` replays these via `apply_ops`.
    pub ops: Vec<RecordedOp>,
    /// Backing storage for `base.validation_state`. IREE's generic
    /// command_buffer.c enables validation by default
    /// (`IREE_HAL_COMMAND_BUFFER_VALIDATION_ENABLE=1`) for any mode without
    /// `IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED` and unconditionally derefs
    /// `validation_state` in that case -- passing NULL (as this used to)
    /// segfaults the instant a validated command buffer is created (CTS's
    /// `EventTest.SignalAndReset` hit this first, once `create_event` started
    /// returning success and the test could reach `iree_hal_command_buffer_
    /// create`). A `Vec`'s heap buffer address is stable across the
    /// `RocketCommandBuffer` itself moving/being boxed, so storing it as a
    /// plain field (rather than mimicking the null driver reference's
    /// single-trailing-allocation trick) is fine.
    validation_state: Vec<u8>,
    /// A raw, borrowed device fd -- populated in `create()` from
    /// `device_allocator`'s own `RocketAllocator.file`. Needed so
    /// `dispatch()` can allocate a driver-private scratch GEM buffer for
    /// Conv2d output compaction (see `OutputCompaction`) at record time,
    /// before `build_conv_regcmd` bakes a DMA address into the regcmd
    /// program. Sound to use independently of `RocketAllocator.file`'s own
    /// lifetime: `RocketAllocator.file` and `RocketDevice.file` are
    /// `try_clone()`'d duplicates of the same open file description (DRM
    /// GEM handles are namespaced per underlying `struct file`, not per fd
    /// integer), and `device_allocator` is guaranteed to outlive every
    /// command buffer created against it.
    fd: RawFd,
}

unsafe fn cast(command_buffer: *mut iree_hal_command_buffer_t) -> *mut RocketCommandBuffer {
    command_buffer as *mut RocketCommandBuffer
}

/// Retains every direct dispatch binding once, including bindings that do
/// not need a packing bridge. IREE permits the caller to release its own
/// references as soon as `dispatch()` returns, so raw buffer pointers and GEM
/// handles recorded in the command buffer are only valid if the command
/// buffer owns corresponding references.
unsafe fn retain_direct_bindings(refs: &[iree_hal_buffer_ref_t]) -> Vec<*mut iree_hal_buffer_t> {
    refs.iter()
        .map(|binding| {
            unsafe { crate::bindings::iree_hal_buffer_retain(binding.buffer) };
            binding.buffer
        })
        .collect()
}

/// Not part of the vtable -- `device::queue_execute` calls this directly
/// after its wait-semaphore gate. Applies every recorded fill/update/copy
/// immediately (host-side, via IREE's generic `iree_hal_buffer_map_*`
/// helpers -- `buffer::map_range`/`unmap_range` already back those
/// correctly) and returns every recorded `dispatch`'s regcmd program, in
/// call order, for the caller to submit to hardware afterward.
pub unsafe fn apply_ops(
    command_buffer: *mut iree_hal_command_buffer_t,
) -> Result<Vec<DispatchJob>, iree_status_t> {
    let cb = unsafe { &*cast(command_buffer) };
    let mut dispatch_jobs = Vec::new();
    for op in &cb.ops {
        // Indirect bindings (buffer == NULL, real buffer resolved from
        // binding_table.buffer_slot -- see command_buffer.h's own doc
        // comment on iree_hal_buffer_ref_t) aren't resolved anywhere in
        // this file today. Found the hard way, as a real segfault on real
        // hardware (iree_hal_buffer_map_copy called with a garbage/null
        // buffer pointer) the first time an actual compiled `.vmfb`
        // reached this code -- this project's own hand-driven CTS tests
        // only ever construct direct bindings. Reject cleanly instead of
        // dereferencing a null buffer; real indirect-binding support would
        // need resolving `binding_table` here (queue_execute's caller
        // does have it available, per device.rs, but nothing plumbs it
        // into apply_ops today) -- out of scope for this fix.
        let indirect_ref = match op {
            RecordedOp::Fill { target, .. } => target.buffer.is_null(),
            RecordedOp::Update { target, .. } => target.buffer.is_null(),
            RecordedOp::Copy { source, target } => {
                source.buffer.is_null() || target.buffer.is_null()
            }
            RecordedOp::Dispatch { .. } => false, // rejected earlier, in dispatch() itself.
        };
        if indirect_ref {
            return Err(status::from_code(
                crate::bindings::iree_status_code_e_IREE_STATUS_UNIMPLEMENTED,
            ));
        }

        match op {
            RecordedOp::Fill {
                target,
                pattern,
                pattern_length,
            } => {
                let st = unsafe {
                    crate::bindings::iree_hal_buffer_map_fill(
                        target.buffer,
                        target.offset,
                        target.length,
                        pattern.as_ptr() as *const std::ffi::c_void,
                        *pattern_length as iree_host_size_t,
                    )
                };
                if !st.is_null() {
                    return Err(st);
                }
            }
            RecordedOp::Update { target, source } => {
                let st = unsafe {
                    crate::bindings::iree_hal_buffer_map_write(
                        target.buffer,
                        target.offset,
                        source.as_ptr() as *const std::ffi::c_void,
                        source.len() as iree_device_size_t,
                    )
                };
                if !st.is_null() {
                    return Err(st);
                }
            }
            RecordedOp::Copy { source, target } => {
                let st = unsafe {
                    crate::bindings::iree_hal_buffer_map_copy(
                        source.buffer,
                        source.offset,
                        target.buffer,
                        target.offset,
                        target.length,
                    )
                };
                if !st.is_null() {
                    return Err(st);
                }
            }
            RecordedOp::Dispatch {
                regcmd_tasks,
                in_bo_handles,
                out_bo_handles,
                input_packing,
                weight_packing,
                output_compaction,
                ..
            } => {
                if let Some(packing) = input_packing {
                    let dense_len = packing
                        .source_pixel_count
                        .checked_mul(packing.bytes_per_pixel)
                        .ok_or_else(|| {
                            status::from_code(
                                crate::bindings::iree_status_code_e_IREE_STATUS_INTERNAL,
                            )
                        })?;
                    if dense_len as u64 > packing.input_length as u64 {
                        return Err(status::from_code(
                            crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                        ));
                    }
                    let input = unsafe { &*(packing.input_buffer as *const RocketBuffer) };
                    let dense = unsafe {
                        std::slice::from_raw_parts(
                            input.host_ptr.add(packing.input_offset),
                            dense_len,
                        )
                    };
                    let padded_dense = if packing.source_pixel_count != packing.packed_pixel_count {
                        let padded_len = packing
                            .packed_pixel_count
                            .checked_mul(packing.bytes_per_pixel)
                            .ok_or_else(|| {
                                status::from_code(
                                    crate::bindings::iree_status_code_e_IREE_STATUS_INTERNAL,
                                )
                            })?;
                        let mut padded = vec![packing.padding_byte; padded_len];
                        padded[..dense_len].copy_from_slice(dense);
                        Some(padded)
                    } else {
                        None
                    };
                    let dense = padded_dense.as_deref().unwrap_or(dense);
                    let scratch = unsafe {
                        std::slice::from_raw_parts_mut(packing.scratch_ptr, packing.scratch_length)
                    };
                    let packing_result = match packing.layout {
                        InputPackingLayout::Dense => {
                            if dense.len() > scratch.len() {
                                Err("dense padded input exceeds its scratch buffer")
                            } else {
                                scratch[..dense.len()].copy_from_slice(dense);
                                Ok(dense.len())
                            }
                        }
                        InputPackingLayout::Nc1hwc2 => pack_nhwc_to_nc1hwc2_padded(
                            dense,
                            packing.packed_pixel_count,
                            packing.bytes_per_pixel,
                            packing.packed_bytes_per_pixel,
                            scratch,
                        ),
                    };
                    if packing_result.is_err() {
                        return Err(status::from_code(
                            crate::bindings::iree_status_code_e_IREE_STATUS_INTERNAL,
                        ));
                    }
                    if unsafe {
                        iree_rocket_hal::rocket::device::fini_bo(cb.fd, packing.scratch_handle)
                    }
                    .is_err()
                    {
                        return Err(status::from_code(
                            crate::bindings::iree_status_code_e_IREE_STATUS_INTERNAL,
                        ));
                    }
                }
                if let Some(packing) = weight_packing {
                    // Depthwise's dense_len has no Cout factor -- one filter
                    // per input channel, not a kernel set per output channel
                    // (packing.output_channels is unused in this mode; see
                    // WeightPacking's doc comment).
                    let dense_len = if packing.depthwise {
                        packing
                            .filter_height
                            .checked_mul(packing.filter_width)
                            .and_then(|value| value.checked_mul(packing.input_channels))
                            .and_then(|value| value.checked_mul(packing.element_size))
                    } else {
                        packing
                            .filter_height
                            .checked_mul(packing.filter_width)
                            .and_then(|value| value.checked_mul(packing.input_channels))
                            .and_then(|value| value.checked_mul(packing.output_channels))
                            .and_then(|value| value.checked_mul(packing.element_size))
                    }
                    .ok_or_else(|| {
                        status::from_code(crate::bindings::iree_status_code_e_IREE_STATUS_INTERNAL)
                    })?;
                    if dense_len as u64 > packing.weight_length as u64 {
                        return Err(status::from_code(
                            crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                        ));
                    }
                    let weights = unsafe { &*(packing.weight_buffer as *const RocketBuffer) };
                    let dense = unsafe {
                        std::slice::from_raw_parts(
                            weights.host_ptr.add(packing.weight_offset),
                            dense_len,
                        )
                    };
                    let scratch = unsafe {
                        std::slice::from_raw_parts_mut(packing.scratch_ptr, packing.scratch_length)
                    };
                    let pack_result = if packing.depthwise {
                        pack_depthwise_to_rocket_weights(
                            dense,
                            packing.filter_height,
                            packing.filter_width,
                            packing.input_channels,
                            packing.padded_channels,
                            packing.element_size,
                            scratch,
                        )
                    } else {
                        pack_hwcf_to_rocket_weights(
                            dense,
                            packing.filter_height,
                            packing.filter_width,
                            packing.input_channels,
                            packing.output_channels,
                            packing.element_size,
                            scratch,
                        )
                    };
                    if pack_result.is_err() {
                        return Err(status::from_code(
                            crate::bindings::iree_status_code_e_IREE_STATUS_INTERNAL,
                        ));
                    }
                    if unsafe {
                        iree_rocket_hal::rocket::device::fini_bo(cb.fd, packing.scratch_handle)
                    }
                    .is_err()
                    {
                        return Err(status::from_code(
                            crate::bindings::iree_status_code_e_IREE_STATUS_INTERNAL,
                        ));
                    }
                }
                dispatch_jobs.push(DispatchJob {
                    regcmd_tasks: regcmd_tasks.as_slice(),
                    in_bo_handles: in_bo_handles.as_slice(),
                    out_bo_handles: out_bo_handles.as_slice(),
                    output_compaction: *output_compaction,
                });
            }
        }
    }
    Ok(dispatch_jobs)
}

pub unsafe fn create(
    device_allocator: *mut crate::bindings::iree_hal_allocator_t,
    mode: iree_hal_command_buffer_mode_t,
    command_categories: iree_hal_command_category_t,
    queue_affinity: iree_hal_queue_affinity_t,
    binding_capacity: iree_host_size_t,
) -> *mut iree_hal_command_buffer_t {
    let validation_state_size = unsafe {
        crate::bindings::iree_hal_command_buffer_validation_state_size(mode, binding_capacity)
    };
    let fd = unsafe {
        (*(device_allocator as *mut crate::allocator::RocketAllocator))
            .file
            .as_raw_fd()
    };
    let cb = Box::new(RocketCommandBuffer {
        base: unsafe { std::mem::zeroed() }, // filled by iree_hal_command_buffer_initialize below
        ops: Vec::new(),
        validation_state: vec![0u8; validation_state_size],
        fd,
    });
    let cb_ptr = Box::into_raw(cb);
    unsafe {
        crate::bindings::iree_hal_command_buffer_initialize(
            device_allocator,
            mode,
            command_categories,
            queue_affinity,
            binding_capacity,
            (*cb_ptr).validation_state.as_mut_ptr() as *mut std::ffi::c_void,
            &VTABLE,
            &mut (*cb_ptr).base,
        );
    }
    cb_ptr as *mut iree_hal_command_buffer_t
}

unsafe extern "C" fn destroy(command_buffer: *mut iree_hal_command_buffer_t) {
    unsafe {
        let cb = Box::from_raw(cast(command_buffer));
        // Release exactly what fill_buffer/update_buffer/copy_buffer/
        // dispatch retained at record time -- see those functions' own
        // comments. Dispatch retains every direct binding, including buffers
        // used without a packing bridge, because IREE permits callers to
        // release their references immediately after recording. Dropping `cb`
        // after this loop also drops every driver-private `OwnedBuffer`, which
        // unmaps its VMA and closes its GEM handle.
        for op in &cb.ops {
            match op {
                RecordedOp::Fill { target, .. } => {
                    crate::bindings::iree_hal_buffer_release(target.buffer);
                }
                RecordedOp::Update { target, .. } => {
                    crate::bindings::iree_hal_buffer_release(target.buffer);
                }
                RecordedOp::Copy { source, target } => {
                    crate::bindings::iree_hal_buffer_release(source.buffer);
                    crate::bindings::iree_hal_buffer_release(target.buffer);
                }
                RecordedOp::Dispatch {
                    retained_bindings, ..
                } => {
                    for &buffer in retained_bindings {
                        crate::bindings::iree_hal_buffer_release(buffer);
                    }
                }
            }
        }
        drop(cb);
    }
}

// Real no-ops (not status_stub -- that returns UNIMPLEMENTED, which would
// break every real caller that expects begin/end to just work).
#[allow(unused_variables)]
unsafe extern "C" fn begin(command_buffer: *mut iree_hal_command_buffer_t) -> iree_status_t {
    status::ok()
}
#[allow(unused_variables)]
unsafe extern "C" fn end(command_buffer: *mut iree_hal_command_buffer_t) -> iree_status_t {
    status::ok()
}

status_stub!(begin_debug_group(
    command_buffer: *mut iree_hal_command_buffer_t,
    label: iree_string_view_t,
    label_color: iree_hal_label_color_t,
    location: *const iree_hal_label_location_t,
) -> iree_status_t);

status_stub!(end_debug_group(command_buffer: *mut iree_hal_command_buffer_t) -> iree_status_t);

// Real no-op, not status_stub -- CTS's TransientBufferTest.
// FillThenCopyInSingleCommandBuffer records fill/barrier/copy into one
// command buffer and expects it to succeed. `apply_ops` (device.rs)
// already replays every recorded op strictly in push order, so ordering
// between ops recorded before/after a barrier is already guaranteed by
// this driver's own execution model -- there's nothing left for a real
// barrier to enforce, same reasoning as signal_event/reset_event/
// wait_events below.
#[allow(unused_variables)]
unsafe extern "C" fn execution_barrier(
    command_buffer: *mut iree_hal_command_buffer_t,
    source_stage_mask: iree_hal_execution_stage_t,
    target_stage_mask: iree_hal_execution_stage_t,
    flags: iree_hal_execution_barrier_flags_t,
    memory_barrier_count: iree_host_size_t,
    memory_barriers: *const iree_hal_memory_barrier_t,
    buffer_barrier_count: iree_host_size_t,
    buffer_barriers: *const iree_hal_buffer_barrier_t,
) -> iree_status_t {
    status::ok()
}

// Real no-ops, not status_stub -- see event.rs's module doc comment: this
// driver's only real synchronization granularity is per-command-buffer
// (device::queue_execute's blocking SUBMIT/PREP_BO), so there's no finer
// in-command-buffer schedule for these to actually mark/enforce. CTS's
// EventTest exercises these through a real command buffer and expects
// success (an execution barrier is a correct, conservative treatment of
// wait_events, per iree_hal_null_command_buffer_wait_events's own comment).
#[allow(unused_variables)]
unsafe extern "C" fn signal_event(
    command_buffer: *mut iree_hal_command_buffer_t,
    event: *mut iree_hal_event_t,
    source_stage_mask: iree_hal_execution_stage_t,
) -> iree_status_t {
    status::ok()
}

#[allow(unused_variables)]
unsafe extern "C" fn reset_event(
    command_buffer: *mut iree_hal_command_buffer_t,
    event: *mut iree_hal_event_t,
    source_stage_mask: iree_hal_execution_stage_t,
) -> iree_status_t {
    status::ok()
}

#[allow(unused_variables)]
unsafe extern "C" fn wait_events(
    command_buffer: *mut iree_hal_command_buffer_t,
    event_count: iree_host_size_t,
    events: *mut *const iree_hal_event_t,
    source_stage_mask: iree_hal_execution_stage_t,
    target_stage_mask: iree_hal_execution_stage_t,
    memory_barrier_count: iree_host_size_t,
    memory_barriers: *const iree_hal_memory_barrier_t,
    buffer_barrier_count: iree_host_size_t,
    buffer_barriers: *const iree_hal_buffer_barrier_t,
) -> iree_status_t {
    status::ok()
}

status_stub!(advise_buffer(
    command_buffer: *mut iree_hal_command_buffer_t,
    buffer_ref: iree_hal_buffer_ref_t,
    flags: iree_hal_memory_advise_flags_t,
    arg0: u64,
    arg1: u64,
) -> iree_status_t);

#[allow(unused_variables)]
unsafe extern "C" fn fill_buffer(
    command_buffer: *mut iree_hal_command_buffer_t,
    target_ref: iree_hal_buffer_ref_t,
    pattern: *const std::ffi::c_void,
    pattern_length: iree_host_size_t,
    flags: iree_hal_fill_flags_t,
) -> iree_status_t {
    if pattern_length > 8 {
        return status::from_code(crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT);
    }
    let mut pattern_buf = [0u8; 8];
    unsafe {
        std::ptr::copy_nonoverlapping(
            pattern as *const u8,
            pattern_buf.as_mut_ptr(),
            pattern_length,
        );
    }
    // See RecordedOp's own doc comment: recorded ops hold onto their
    // buffer_ref's raw `buffer` pointer for later use in `apply_ops`,
    // which runs at queue_execute time -- potentially well after this
    // recording call returns and the caller drops its own reference.
    // Without retaining here, that's a real use-after-free (found the
    // hard way, as a real segfault on real hardware).
    unsafe {
        crate::bindings::iree_hal_buffer_retain(target_ref.buffer);
    }
    let cb = unsafe { &mut *cast(command_buffer) };
    cb.ops.push(RecordedOp::Fill {
        target: target_ref,
        pattern: pattern_buf,
        pattern_length: pattern_length as u8,
    });
    status::ok()
}

#[allow(unused_variables)]
unsafe extern "C" fn update_buffer(
    command_buffer: *mut iree_hal_command_buffer_t,
    source_buffer: *const std::ffi::c_void,
    source_offset: iree_host_size_t,
    target_ref: iree_hal_buffer_ref_t,
    flags: iree_hal_update_flags_t,
) -> iree_status_t {
    let len = target_ref.length;
    let mut source = vec![0u8; len];
    unsafe {
        let src = (source_buffer as *const u8).add(source_offset);
        std::ptr::copy_nonoverlapping(src, source.as_mut_ptr(), len);
    }
    // See fill_buffer's comment on why this retain is needed.
    unsafe {
        crate::bindings::iree_hal_buffer_retain(target_ref.buffer);
    }
    let cb = unsafe { &mut *cast(command_buffer) };
    cb.ops.push(RecordedOp::Update {
        target: target_ref,
        source,
    });
    status::ok()
}

#[allow(unused_variables)]
unsafe extern "C" fn copy_buffer(
    command_buffer: *mut iree_hal_command_buffer_t,
    source_ref: iree_hal_buffer_ref_t,
    target_ref: iree_hal_buffer_ref_t,
    flags: iree_hal_copy_flags_t,
) -> iree_status_t {
    // See fill_buffer's comment on why these retains are needed.
    unsafe {
        crate::bindings::iree_hal_buffer_retain(source_ref.buffer);
        crate::bindings::iree_hal_buffer_retain(target_ref.buffer);
    }
    let cb = unsafe { &mut *cast(command_buffer) };
    cb.ops.push(RecordedOp::Copy {
        source: source_ref,
        target: target_ref,
    });
    status::ok()
}

status_stub!(collective(
    command_buffer: *mut iree_hal_command_buffer_t,
    channel: *mut iree_hal_channel_t,
    op: iree_hal_collective_op_t,
    param: u32,
    send_ref: iree_hal_buffer_ref_t,
    recv_ref: iree_hal_buffer_ref_t,
    element_count: iree_device_size_t,
) -> iree_status_t);

#[allow(unused_variables)]
unsafe extern "C" fn dispatch(
    command_buffer: *mut iree_hal_command_buffer_t,
    executable: *mut iree_hal_executable_t,
    function: iree_hal_executable_function_t,
    config: iree_hal_dispatch_config_t,
    constants: iree_const_byte_span_t,
    bindings: iree_hal_buffer_ref_list_t,
    flags: iree_hal_dispatch_flags_t,
) -> iree_status_t {
    let cb = unsafe { &mut *cast(command_buffer) };
    let shape = unsafe { &*crate::executable::shape(executable) };
    let constants = if constants.data_length == 0 {
        &[]
    } else {
        if constants.data.is_null() {
            return status::from_code(
                crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
            );
        }
        unsafe { std::slice::from_raw_parts(constants.data, constants.data_length) }
    };
    let refs = unsafe { std::slice::from_raw_parts(bindings.values, bindings.count) };

    // Indirect bindings (iree_hal_buffer_ref_t.buffer == NULL, real buffer
    // resolved from a binding_table.buffer_slot at queue_execute time, not
    // known yet at this record-time call -- see command_buffer.h's own doc
    // comment on iree_hal_buffer_ref_t) are the default IREE compiles
    // programs with (`--iree-hal-indirect-command-buffers` defaults to
    // true). This driver's whole design builds the real regcmd (with
    // concrete DMA addresses) immediately here, at record time, which is
    // fundamentally incompatible with a binding whose concrete buffer isn't
    // known yet -- found the hard way, as a real segfault on real hardware,
    // the first time an actual compiled `.vmfb` (as opposed to this
    // project's own hand-driven CTS tests, which only ever construct direct
    // bindings via iree_hal_make_buffer_ref) reached this function. Rather
    // than dereference a null `RocketBuffer*` (undefined behavior), reject
    // indirect bindings with a clear, real error -- true support would need
    // deferring regcmd construction to queue_execute time, a larger
    // redesign out of scope for this fix.
    if let Some(r) = refs.iter().find(|r| r.buffer.is_null()) {
        let _ = r;
        return status::from_code(crate::bindings::iree_status_code_e_IREE_STATUS_UNIMPLEMENTED);
    }

    // `r.offset` is the byte offset of this binding WITHIN its underlying
    // buffer -- always 0 in every hand-driven CTS test so far (each of
    // those constructs its own dedicated buffer per binding via
    // iree_hal_make_buffer_ref(buf, 0, size)), but a REAL compiled IREE
    // program routinely sub-allocates multiple tensor arguments out of one
    // combined transient buffer at nonzero offsets (confirmed on real
    // hardware: input/weights bindings shared one buffer, weights at
    // offset=64). Forgetting to add it here silently pointed the regcmd's
    // weight-read register at byte 0 of that shared buffer (the INPUT
    // tensor's own data) instead of the real weight value 64 bytes in --
    // found via a hardware diagnostic dump, not by inspection.
    let addr = |r: &iree_hal_buffer_ref_t| unsafe {
        (*(r.buffer as *mut RocketBuffer)).dma_address + r.offset as u32
    };
    let handle = |r: &iree_hal_buffer_ref_t| unsafe { (*(r.buffer as *mut RocketBuffer)).handle };

    // Binding convention: per-ukernel-kind, since each kind's regcmd
    // builder needs a different set of buffers -- see module doc comment
    // for why this is now a frozen cross-repo ABI contract, not just a
    // placeholder.
    match shape {
        UkernelShape::Conv2d(executable) => {
            let (resolved_shape, kernels) = match executable.resolve_shape(constants) {
                Ok(resolved) => resolved,
                Err(_) => {
                    return status::from_code(
                        crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                    );
                }
            };
            let shape = &resolved_shape;
            if bindings.count < 4 {
                return status::from_code(
                    crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                );
            }
            let pixel_count = shape.width as usize * shape.height as usize;
            let input_bytes_per_pixel =
                shape.in_channels as usize * shape.precision.element_bytes() as usize;
            let packed_input_bytes_per_pixel = shape.in_channels.max(16).next_multiple_of(16)
                as usize
                * shape.precision.element_bytes() as usize;
            if !matches!(
                pixel_count.checked_mul(input_bytes_per_pixel),
                Some(value) if value as u64 <= refs[0].length as u64
            ) {
                return status::from_code(
                    crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                );
            }
            let mut scratch_buffers = Vec::with_capacity(3);

            // CNA's surface-layout path consumes 16-byte feature-atomic
            // NC1HWC2 surfaces. Shapes with 1..=4 channels use the hardware's
            // dense ARGB modes and must remain dense; packing Cin 2..=4 into
            // 16-byte slots makes those modes read padding as later pixels.
            let (input_addr, input_handle, input_packing) =
                if shape.layout() == FeatureLayout::Surfaces {
                    let scratch_bytes =
                        match nc1hwc2_storage_size(pixel_count, packed_input_bytes_per_pixel) {
                            Ok(value) => value,
                            Err(_) => {
                                return status::from_code(
                                crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                            );
                            }
                        };
                    let scratch = unsafe {
                        RocketOwnedBuffer::new(
                            cb.fd,
                            scratch_bytes.max(1),
                            BorrowedFd::borrow_raw(cb.fd),
                        )
                    };
                    let packed = (
                        scratch.dma_address,
                        scratch.handle,
                        Some(InputPacking {
                            input_buffer: refs[0].buffer,
                            input_offset: refs[0].offset,
                            input_length: refs[0].length,
                            scratch_ptr: scratch.host_ptr,
                            scratch_length: scratch_bytes,
                            scratch_handle: scratch.handle,
                            source_pixel_count: pixel_count,
                            packed_pixel_count: pixel_count,
                            bytes_per_pixel: input_bytes_per_pixel,
                            packed_bytes_per_pixel: packed_input_bytes_per_pixel,
                            padding_byte: 0,
                            layout: InputPackingLayout::Nc1hwc2,
                        }),
                    );
                    scratch_buffers.push(scratch);
                    packed
                } else {
                    (addr(&refs[0]), handle(&refs[0]), None)
                };
            // IREE's conv ABI supplies a logical HWCF filter. Regular fp16
            // convolution consumes a blocked coefficient stream instead:
            // output-block, input-group, X, Y, output-lane, input-lane.
            // This is independently deferred for the same reason as input
            // packing: an earlier recorded operation may populate weights.
            let element_size = shape.precision.element_bytes() as usize;
            let (weights_addr, weights_handle, weight_packing) =
                if shape.precision == Precision::Fp16 && !shape.depthwise {
                    if !matches!(
                        kernels[0]
                        .checked_mul(kernels[1])
                        .and_then(|value| value.checked_mul(shape.in_channels as usize))
                        .and_then(|value| value.checked_mul(shape.out_channels as usize))
                        .and_then(|value| value.checked_mul(element_size)),
                        Some(value) if value as u64 <= refs[1].length as u64
                    ) {
                        return status::from_code(
                            crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                        );
                    }
                    let scratch_bytes = match rocket_weight_storage_size(
                        kernels[0],
                        kernels[1],
                        shape.in_channels as usize,
                        shape.out_channels as usize,
                        element_size,
                    ) {
                        Ok(value) => value,
                        Err(_) => {
                            return status::from_code(
                                crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                            );
                        }
                    };
                    let scratch = unsafe {
                        RocketOwnedBuffer::new(
                            cb.fd,
                            scratch_bytes.max(1),
                            BorrowedFd::borrow_raw(cb.fd),
                        )
                    };
                    let packed = (
                        scratch.dma_address,
                        scratch.handle,
                        Some(WeightPacking {
                            weight_buffer: refs[1].buffer,
                            weight_offset: refs[1].offset,
                            weight_length: refs[1].length,
                            scratch_ptr: scratch.host_ptr,
                            scratch_length: scratch_bytes,
                            scratch_handle: scratch.handle,
                            filter_height: kernels[0],
                            filter_width: kernels[1],
                            input_channels: shape.in_channels as usize,
                            output_channels: shape.out_channels as usize,
                            element_size,
                            depthwise: false,
                            padded_channels: 0,
                        }),
                    );
                    scratch_buffers.push(scratch);
                    packed
                } else if shape.precision == Precision::Fp16 && shape.depthwise {
                    // One filter per input channel -- no Cout factor, unlike
                    // the dense branch above. The compiler-emitted dispatch
                    // (transform.0.mlir's call_rocket_dynamic_depthwise_conv2d)
                    // transposes the filter to [Cin][kh][kw] before this
                    // binding is populated, matching what
                    // pack_depthwise_to_rocket_weights expects -- see that
                    // function's doc comment.
                    if !matches!(
                        kernels[0]
                        .checked_mul(kernels[1])
                        .and_then(|value| value.checked_mul(shape.in_channels as usize))
                        .and_then(|value| value.checked_mul(element_size)),
                        Some(value) if value as u64 <= refs[1].length as u64
                    ) {
                        return status::from_code(
                            crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                        );
                    }
                    let scratch_bytes = shape.weight_bytes(kernels) as usize;
                    let scratch = unsafe {
                        RocketOwnedBuffer::new(
                            cb.fd,
                            scratch_bytes.max(1),
                            BorrowedFd::borrow_raw(cb.fd),
                        )
                    };
                    let packed = (
                        scratch.dma_address,
                        scratch.handle,
                        Some(WeightPacking {
                            weight_buffer: refs[1].buffer,
                            weight_offset: refs[1].offset,
                            weight_length: refs[1].length,
                            scratch_ptr: scratch.host_ptr,
                            scratch_length: scratch_bytes,
                            scratch_handle: scratch.handle,
                            filter_height: kernels[0],
                            filter_width: kernels[1],
                            input_channels: shape.in_channels as usize,
                            output_channels: shape.out_channels as usize,
                            element_size,
                            depthwise: true,
                            padded_channels: shape.depthwise_padded_channels() as usize,
                        }),
                    );
                    scratch_buffers.push(scratch);
                    packed
                } else {
                    (addr(&refs[1]), handle(&refs[1]), None)
                };
            // DPU's atomic-slot output write-back (16-byte-aligned slots
            // regardless of dtype) doesn't match IREE's densely-packed ABI
            // output buffer -- see `OutputCompaction`'s own doc comment.
            // Bridge it with a driver-private scratch buffer: the regcmd
            // writes there instead of the real output buffer, and
            // `queue_execute` compacts the real values into the real
            // buffer after the hardware write completes.
            let scratch_bytes = shape.output_scratch_bytes(kernels).max(1);
            let scratch = unsafe {
                RocketOwnedBuffer::new(cb.fd, scratch_bytes, BorrowedFd::borrow_raw(cb.fd))
            };
            let bufs = Buffers {
                input: input_addr,
                weights: weights_addr,
                bias: addr(&refs[2]),
                output: scratch.dma_address,
            };
            // catch_unwind backstop -- see module doc comment for exactly
            // why. `resolve_shape` already trial-plans this exact shape via
            // `validate_conv_shape` (which shares this same `ConvPlan::new`
            // call), so a panic here indicates a genuine internal
            // inconsistency rather than an ordinary user error. The builder
            // only returns fresh local vectors, so a panic mid-build leaves
            // no shared state half-mutated.
            let regcmd_tasks = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                ConvPlan::new(*shape, kernels).programs_with_buffers(bufs)
            })) {
                Ok(tasks) => tasks,
                Err(_) => {
                    return status::from_code(
                        crate::bindings::iree_status_code_e_IREE_STATUS_INTERNAL,
                    );
                }
            };
            let output_pixel_count =
                shape.output_width(kernels) as usize * shape.output_height(kernels) as usize;
            let output_compaction = Some(OutputCompaction {
                output_buffer: refs[3].buffer,
                output_offset: refs[3].offset,
                output_length: refs[3].length,
                scratch_ptr: scratch.host_ptr,
                scratch_length: scratch_bytes,
                source_pixel_count: output_pixel_count,
                output_pixel_count,
                bytes_per_pixel: shape.out_channels as usize
                    * shape.precision.element_bytes() as usize,
            });
            let output_handle = scratch.handle;
            scratch_buffers.push(scratch);
            let retained_bindings = unsafe { retain_direct_bindings(refs) };
            cb.ops.push(RecordedOp::Dispatch {
                regcmd_tasks,
                retained_bindings,
                scratch_buffers,
                in_bo_handles: vec![input_handle, weights_handle, handle(&refs[2])],
                out_bo_handles: vec![output_handle],
                input_packing,
                weight_packing,
                output_compaction,
            });
        }
        UkernelShape::FullyConnected(shape) => {
            if !constants.is_empty() {
                return status::from_code(
                    crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                );
            }
            if bindings.count < 4 {
                return status::from_code(
                    crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                );
            }

            let m = shape.m as usize;
            let k = shape.k as usize;
            let n = shape.n as usize;
            let element_size = shape.precision.element_bytes() as usize;
            let input_zero_point = match shape.precision {
                Precision::Fp16 => 0,
                Precision::Int8(quantization) => quantization.input_zero_point,
            };
            // fc.rs's real vendor-confirmed lowering has physical height
            // exactly one -- no `FC_PHYSICAL_HEIGHT` padding, so the
            // packed pixel count is just the logical row count `m`.
            let physical_pixel_count = m;
            let input_bytes_per_pixel = match k.checked_mul(element_size) {
                Some(value) => value,
                None => {
                    return status::from_code(
                        crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                    );
                }
            };
            let output_bytes_per_pixel = match n.checked_mul(element_size) {
                Some(value) => value,
                None => {
                    return status::from_code(
                        crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                    );
                }
            };
            let input_len = m.checked_mul(input_bytes_per_pixel);
            let weights_len = k
                .checked_mul(n)
                .and_then(|value| value.checked_mul(element_size));
            let bias_len = n.checked_mul(element_size);
            let output_len = m.checked_mul(output_bytes_per_pixel);
            if !matches!(input_len, Some(value) if value as u64 <= refs[0].length as u64)
                || !matches!(weights_len, Some(value) if value as u64 <= refs[1].length as u64)
                || !matches!(bias_len, Some(value) if value as u64 <= refs[2].length as u64)
                || !matches!(output_len, Some(value) if value as u64 <= refs[3].length as u64)
                || !(0..=u8::MAX as i32).contains(&input_zero_point)
            {
                return status::from_code(
                    crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                );
            }

            // FC is lowered by fc::Plan to a height-one 1x1 convolution
            // (see fc.rs's module doc comment) -- the public input already
            // is exactly `m` physical rows, so this only needs the same
            // NC1HWC2 channel blocking used by convolution, no row padding.
            let packed_input_bytes_per_pixel = k.max(16).next_multiple_of(16) * element_size;
            let (input_scratch_bytes, input_layout) = if k > 1 {
                match nc1hwc2_storage_size(physical_pixel_count, packed_input_bytes_per_pixel) {
                    Ok(value) => (value, InputPackingLayout::Nc1hwc2),
                    Err(_) => {
                        return status::from_code(
                            crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                        );
                    }
                }
            } else {
                match physical_pixel_count.checked_mul(input_bytes_per_pixel) {
                    Some(value) => (value, InputPackingLayout::Dense),
                    None => {
                        return status::from_code(
                            crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                        );
                    }
                }
            };
            if input_scratch_bytes > u32::MAX as usize {
                return status::from_code(
                    crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                );
            }
            let input_scratch = unsafe {
                RocketOwnedBuffer::new(
                    cb.fd,
                    input_scratch_bytes.max(1),
                    BorrowedFd::borrow_raw(cb.fd),
                )
            };
            let input_packing = Some(InputPacking {
                input_buffer: refs[0].buffer,
                input_offset: refs[0].offset,
                input_length: refs[0].length,
                scratch_ptr: input_scratch.host_ptr,
                scratch_length: input_scratch_bytes,
                scratch_handle: input_scratch.handle,
                source_pixel_count: m,
                packed_pixel_count: physical_pixel_count,
                bytes_per_pixel: input_bytes_per_pixel,
                packed_bytes_per_pixel: packed_input_bytes_per_pixel,
                padding_byte: input_zero_point as u8,
                layout: input_layout,
            });

            // FC weights arrive as a logical row-major [K,N] matrix, which
            // is exactly a 1x1 HWCF filter and therefore always needs the
            // CNA coefficient transform (for both int8 and fp16).
            let weight_scratch_bytes = match rocket_weight_storage_size(1, 1, k, n, element_size) {
                Ok(value) => value,
                Err(_) => {
                    return status::from_code(
                        crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                    );
                }
            };
            if weight_scratch_bytes > u32::MAX as usize {
                return status::from_code(
                    crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                );
            }
            let weight_scratch = unsafe {
                RocketOwnedBuffer::new(
                    cb.fd,
                    weight_scratch_bytes.max(1),
                    BorrowedFd::borrow_raw(cb.fd),
                )
            };
            let weight_packing = Some(WeightPacking {
                weight_buffer: refs[1].buffer,
                weight_offset: refs[1].offset,
                weight_length: refs[1].length,
                scratch_ptr: weight_scratch.host_ptr,
                scratch_length: weight_scratch_bytes,
                scratch_handle: weight_scratch.handle,
                filter_height: 1,
                filter_width: 1,
                input_channels: k,
                output_channels: n,
                element_size,
                depthwise: false,
                padded_channels: 0,
            });

            let output_scratch_bytes =
                match nc1hwc2_storage_size(physical_pixel_count, output_bytes_per_pixel) {
                    Ok(value) => value,
                    Err(_) => {
                        return status::from_code(
                            crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                        );
                    }
                };
            if output_scratch_bytes > u32::MAX as usize {
                return status::from_code(
                    crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                );
            }
            let output_scratch = unsafe {
                RocketOwnedBuffer::new(
                    cb.fd,
                    output_scratch_bytes.max(1),
                    BorrowedFd::borrow_raw(cb.fd),
                )
            };
            let bufs = Buffers {
                input: input_scratch.dma_address,
                weights: weight_scratch.dma_address,
                bias: addr(&refs[2]),
                output: output_scratch.dma_address,
            };
            let regcmd_tasks = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                fc::Plan::new(*shape).programs_with_buffers(bufs)
            })) {
                Ok(tasks) => tasks,
                Err(_) => {
                    return status::from_code(
                        crate::bindings::iree_status_code_e_IREE_STATUS_INTERNAL,
                    );
                }
            };

            let input_scratch_handle = input_scratch.handle;
            let weight_scratch_handle = weight_scratch.handle;
            let output_scratch_handle = output_scratch.handle;
            let output_compaction = Some(OutputCompaction {
                output_buffer: refs[3].buffer,
                output_offset: refs[3].offset,
                output_length: refs[3].length,
                scratch_ptr: output_scratch.host_ptr,
                scratch_length: output_scratch_bytes,
                // No row padding to discard (see physical_pixel_count's
                // own comment) -- source and output pixel counts are the
                // same real `m`.
                source_pixel_count: m,
                output_pixel_count: m,
                bytes_per_pixel: output_bytes_per_pixel,
            });
            let retained_bindings = unsafe { retain_direct_bindings(refs) };
            cb.ops.push(RecordedOp::Dispatch {
                regcmd_tasks,
                retained_bindings,
                scratch_buffers: vec![input_scratch, weight_scratch, output_scratch],
                in_bo_handles: vec![
                    input_scratch_handle,
                    weight_scratch_handle,
                    handle(&refs[2]),
                ],
                out_bo_handles: vec![output_scratch_handle],
                input_packing,
                weight_packing,
                output_compaction,
            });
        }
        UkernelShape::Pooling(shape) => {
            if !constants.is_empty() {
                return status::from_code(
                    crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                );
            }
            if bindings.count < 2 {
                return status::from_code(
                    crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                );
            }
            // 0=input, 1=output. Every horizontal tile is one direct
            // PPU/PPU_RDMA task; all tasks belong to this one dispatch/job.
            let bufs = PoolingBuffers {
                input_addr: addr(&refs[0]),
                output_addr: addr(&refs[1]),
            };
            let regcmd_tasks = PoolingPlan::new(*shape).programs_with_buffers(&bufs);
            let retained_bindings = unsafe { retain_direct_bindings(refs) };
            cb.ops.push(RecordedOp::Dispatch {
                regcmd_tasks,
                retained_bindings,
                scratch_buffers: Vec::new(),
                in_bo_handles: vec![handle(&refs[0])],
                out_bo_handles: vec![handle(&refs[1])],
                input_packing: None,
                weight_packing: None,
                output_compaction: None,
            });
        }
    }
    status::ok()
}

pub static VTABLE: iree_hal_command_buffer_vtable_t = iree_hal_command_buffer_vtable_t {
    destroy: Some(destroy),
    begin: Some(begin),
    end: Some(end),
    begin_debug_group: Some(begin_debug_group),
    end_debug_group: Some(end_debug_group),
    execution_barrier: Some(execution_barrier),
    signal_event: Some(signal_event),
    reset_event: Some(reset_event),
    wait_events: Some(wait_events),
    advise_buffer: Some(advise_buffer),
    fill_buffer: Some(fill_buffer),
    update_buffer: Some(update_buffer),
    copy_buffer: Some(copy_buffer),
    collective: Some(collective),
    dispatch: Some(dispatch),
};
