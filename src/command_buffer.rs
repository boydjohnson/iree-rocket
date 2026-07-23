//! `iree_hal_command_buffer_vtable_t`. `dispatch` is the real regcmd
//! integration point: turns the executable's `UkernelShape`
//! (`executable::shape`) plus this dispatch's buffer bindings into a
//! regcmd program via the matching `iree_rocket_hal::rocket::regcmd::
//! build_*_regcmd` function and stashes it (regcmd + the real GEM handles
//! it touches, see `RecordedOp::Dispatch`) on the command buffer, to be
//! submitted as one job by `device::queue_execute`.
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
//! `Conv2d`'s `build_conv_regcmd` call is wrapped in `catch_unwind` (see
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
//! Only supports recording exactly one `dispatch` per command buffer right
//! now (matches every current `build_*_regcmd` function's own single-
//! task-only scope, see regcmd.rs) -- a second `dispatch` call returns
//! `IREE_STATUS_UNIMPLEMENTED` rather than silently overwriting or
//! chaining tasks incorrectly. `collective` (multi-device reduce/
//! broadcast/etc.) isn't applicable to a single discrete NPU and stays
//! UNIMPLEMENTED indefinitely, not just for now.
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

use crate::bindings::{
    iree_const_byte_span_t, iree_device_size_t, iree_hal_buffer_barrier_t,
    iree_hal_buffer_ref_list_t, iree_hal_buffer_ref_t, iree_hal_channel_t,
    iree_hal_collective_op_t, iree_hal_command_buffer_mode_t, iree_hal_command_buffer_t,
    iree_hal_command_buffer_vtable_t, iree_hal_command_category_t, iree_hal_copy_flags_t,
    iree_hal_dispatch_config_t, iree_hal_dispatch_flags_t, iree_hal_event_t,
    iree_hal_executable_function_t, iree_hal_executable_t, iree_hal_execution_barrier_flags_t,
    iree_hal_execution_stage_t, iree_hal_fill_flags_t, iree_hal_label_color_t,
    iree_hal_label_location_t, iree_hal_memory_advise_flags_t, iree_hal_memory_barrier_t,
    iree_hal_queue_affinity_t, iree_hal_update_flags_t, iree_host_size_t, iree_status_t,
    iree_string_view_t,
};
use crate::buffer::RocketBuffer;
use crate::executable::UkernelShape;
use crate::status;
use iree_rocket_hal::rocket::{
    builders::RegCmd,
    regcmd::{ConvBuffers, PoolingBuffers, build_conv_regcmd, build_pooling_regcmd},
};

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
        regcmd: Vec<RegCmd>,
        /// GEM handles of every buffer this dispatch reads (bindings other
        /// than the output) -- must be listed in `drm_rocket_job.in_bo_handles`
        /// (device.rs's `queue_execute`) so the kernel driver's implicit
        /// fencing/dependency tracking actually knows the job touches them.
        /// Previously only the regcmd program's own GEM buffer was listed
        /// there, which happened to let SUBMIT/PREP_BO round-trip (proving
        /// the ioctl plumbing itself worked) but never told the kernel
        /// about the real input/weight/bias/output BOs at all -- see
        /// `conv_hw.rs` in iree-rocket-hal for the hand-rolled ioctl call
        /// that always did this correctly.
        in_bo_handles: Vec<u32>,
        /// GEM handles of every buffer this dispatch writes.
        out_bo_handles: Vec<u32>,
    },
}

/// What `apply_ops` hands back for the one recorded `dispatch`, if any --
/// the regcmd program plus the real BO handles it touches, so
/// `device::queue_execute` can build a correct `drm_rocket_job` instead of
/// submitting with only the regcmd buffer's own handle listed.
pub struct DispatchJob {
    pub regcmd: &'static [RegCmd],
    pub in_bo_handles: &'static [u32],
    pub out_bo_handles: &'static [u32],
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
}

unsafe fn cast(command_buffer: *mut iree_hal_command_buffer_t) -> *mut RocketCommandBuffer {
    command_buffer as *mut RocketCommandBuffer
}

/// Not part of the vtable -- `device::queue_execute` calls this directly
/// after its wait-semaphore gate. Applies every recorded fill/update/copy
/// immediately (host-side, via IREE's generic `iree_hal_buffer_map_*`
/// helpers -- `buffer::map_range`/`unmap_range` already back those
/// correctly) and returns the one recorded `dispatch`'s regcmd program, if
/// any, for the caller to submit to hardware afterward.
pub unsafe fn apply_ops(
    command_buffer: *mut iree_hal_command_buffer_t,
) -> Result<Option<DispatchJob>, iree_status_t> {
    let cb = unsafe { &*cast(command_buffer) };
    let mut dispatch_job = None;
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
                crate::bindings::iree_status_code_e_IREE_STATUS_UNIMPLEMENTED as u32,
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
                regcmd,
                in_bo_handles,
                out_bo_handles,
            } => {
                dispatch_job = Some(DispatchJob {
                    regcmd: regcmd.as_slice(),
                    in_bo_handles: in_bo_handles.as_slice(),
                    out_bo_handles: out_bo_handles.as_slice(),
                });
            }
        }
    }
    Ok(dispatch_job)
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
    let cb = Box::new(RocketCommandBuffer {
        base: unsafe { std::mem::zeroed() }, // filled by iree_hal_command_buffer_initialize below
        ops: Vec::new(),
        validation_state: vec![0u8; validation_state_size],
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
        // Release exactly what fill_buffer/update_buffer/copy_buffer
        // retained at record time -- see those functions' own comments.
        // Dispatch isn't included: it never retains a buffer reference
        // (only extracts DMA address/GEM handle immediately, at record
        // time, via addr()/handle() -- see dispatch()'s own doc comment).
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
                RecordedOp::Dispatch { .. } => {}
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
    if pattern_length as usize > 8 {
        return status::from_code(
            crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT as u32,
        );
    }
    let mut pattern_buf = [0u8; 8];
    unsafe {
        std::ptr::copy_nonoverlapping(
            pattern as *const u8,
            pattern_buf.as_mut_ptr(),
            pattern_length as usize,
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
    let len = target_ref.length as usize;
    let mut source = vec![0u8; len];
    unsafe {
        let src = (source_buffer as *const u8).add(source_offset as usize);
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
    if cb
        .ops
        .iter()
        .any(|op| matches!(op, RecordedOp::Dispatch { .. }))
    {
        // Only one dispatch per command buffer supported -- see module doc
        // comment.
        return status::unimplemented();
    }

    let shape = unsafe { &*crate::executable::shape(executable) };
    let refs = unsafe { std::slice::from_raw_parts(bindings.values, bindings.count as usize) };

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
        return status::from_code(
            crate::bindings::iree_status_code_e_IREE_STATUS_UNIMPLEMENTED as u32,
        );
    }

    let addr =
        |r: &iree_hal_buffer_ref_t| unsafe { (*(r.buffer as *mut RocketBuffer)).dma_address };
    let handle = |r: &iree_hal_buffer_ref_t| unsafe { (*(r.buffer as *mut RocketBuffer)).handle };

    // Binding convention: per-ukernel-kind, since each kind's regcmd
    // builder needs a different set of buffers -- see module doc comment
    // for why this is now a frozen cross-repo ABI contract, not just a
    // placeholder.
    match shape {
        UkernelShape::Conv2d(shape) => {
            if bindings.count < 4 {
                return status::from_code(
                    crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT as u32,
                );
            }
            // 0=input, 1=weights, 2=bias, 3=output.
            let bufs = ConvBuffers {
                input_addr: addr(&refs[0]),
                weights_addr: addr(&refs[1]),
                bias_addr: addr(&refs[2]),
                output_addr: addr(&refs[3]),
            };
            // catch_unwind backstop -- see module doc comment for exactly
            // why. Safe to assert unwind-safety here: build_conv_regcmd
            // only builds and returns a fresh local Vec<RegCmd>, so a
            // panic mid-build just drops that local value -- no shared or
            // global state is left half-mutated.
            let regcmd = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                build_conv_regcmd(shape, &bufs)
            })) {
                Ok(regcmd) => regcmd,
                Err(_) => {
                    return status::from_code(
                        crate::bindings::iree_status_code_e_IREE_STATUS_INTERNAL as u32,
                    );
                }
            };
            cb.ops.push(RecordedOp::Dispatch {
                regcmd,
                in_bo_handles: vec![handle(&refs[0]), handle(&refs[1]), handle(&refs[2])],
                out_bo_handles: vec![handle(&refs[3])],
            });
        }
        UkernelShape::Pooling(shape) => {
            if bindings.count < 2 {
                return status::from_code(
                    crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT as u32,
                );
            }
            // 0=input, 1=output -- no weights/bias for a standalone
            // pooling op (see iree-rocket-hal's build_pooling_regcmd).
            let bufs = PoolingBuffers {
                input_addr: addr(&refs[0]),
                output_addr: addr(&refs[1]),
            };
            cb.ops.push(RecordedOp::Dispatch {
                regcmd: build_pooling_regcmd(shape, &bufs),
                in_bo_handles: vec![handle(&refs[0])],
                out_bo_handles: vec![handle(&refs[1])],
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
