//! `iree_hal_semaphore_vtable_t`. Turned out to need real integration with
//! `iree/async/semaphore.h`'s `iree_async_semaphore_t` -- not the
//! from-scratch `Mutex`+`Condvar` originally sketched. `iree_hal_semaphore_t`
//! is opaque (no public field definition, like `iree_hal_allocator_t`), but
//! the vtable's embedded `iree_async_semaphore_vtable_t` operates on
//! `*mut iree_async_semaphore_t` specifically, and `iree_async_semaphore_
//! multi_wait()`'s own doc comment confirms the intended pattern: "HAL
//! semaphores are compatible via toll-free bridging (cast
//! `iree_hal_semaphore_t*` to `iree_async_semaphore_t*`)" -- i.e. embed a
//! real `iree_async_semaphore_t` at offset 0 (mirrored from
//! `iree-null-driver-reference/semaphore.c`) and delegate the actual
//! timeline-value tracking/wait-satisfaction logic to IREE's own exported
//! functions (`iree_async_semaphore_initialize`/`_advance_timeline`/
//! `_multi_wait`) rather than reimplementing it.
//!
//! Needs an `iree_async_proactor_t`, obtained from the
//! `iree_async_proactor_pool_t` a caller provides via
//! `iree_hal_device_create_params_t.proactor_pool` at device-creation time
//! (not something this driver creates itself -- see device.rs). A null
//! proactor is tolerated (just means `import_timepoint`/`export_timepoint`,
//! already UNIMPLEMENTED below, definitely can't work -- local
//! advance_timeline/multi_wait don't appear to need one for our
//! synchronous, non-imported/exported use).

use crate::bindings::{
    iree_allocator_t, iree_async_frontier_t, iree_async_proactor_t, iree_async_semaphore_t,
    iree_async_semaphore_vtable_t, iree_async_wait_flags_t,
    iree_async_wait_mode_e_IREE_ASYNC_WAIT_MODE_ALL, iree_hal_external_timepoint_flags_t,
    iree_hal_external_timepoint_t, iree_hal_external_timepoint_type_t, iree_hal_queue_affinity_t,
    iree_hal_semaphore_t, iree_hal_semaphore_vtable_t, iree_status_code_t, iree_status_t,
    iree_timeout_t,
};

/// What every `iree_hal_semaphore_t*` this driver hands out actually
/// points to. `async_sem` (a real `iree_async_semaphore_t`, initialized via
/// `iree_async_semaphore_initialize`) must be the first field -- see module
/// doc comment.
#[repr(C)]
pub struct RocketSemaphore {
    pub async_sem: iree_async_semaphore_t,
    pub host_allocator: iree_allocator_t,
}

unsafe fn cast(semaphore: *mut iree_hal_semaphore_t) -> *mut RocketSemaphore {
    semaphore as *mut RocketSemaphore
}

unsafe fn async_cast(semaphore: *mut iree_async_semaphore_t) -> *mut RocketSemaphore {
    semaphore as *mut RocketSemaphore
}

/// Mirrors `iree_hal_null_semaphore_create()`. Not called from anywhere
/// yet -- `device::create_semaphore` (still UNIMPLEMENTED) is what will
/// eventually call this with a proactor obtained from the device's pool.
pub unsafe fn create(
    proactor: *mut iree_async_proactor_t,
    initial_value: u64,
    host_allocator: iree_allocator_t,
) -> *mut iree_hal_semaphore_t {
    let semaphore = Box::new(RocketSemaphore {
        // frontier_capacity=0 -- we don't do causal frontier tracking
        // (only relevant for import/export_timepoint, which stay
        // UNIMPLEMENTED below), so there's no trailing storage to lay out
        // and frontier_offset is unused.
        async_sem: unsafe { std::mem::zeroed() },
        host_allocator,
    });
    let semaphore_ptr = Box::into_raw(semaphore);
    unsafe {
        // Must be VTABLE.async_ (the async_ field embedded in the full
        // iree_hal_semaphore_vtable_t), not the standalone ASYNC_VTABLE --
        // iree_hal_semaphore_release() reinterprets this same pointer as
        // iree_hal_semaphore_vtable_t* (toll-free bridging, see module doc
        // comment) and reads the trailing wait/import_timepoint/
        // export_timepoint fields past it. Pointing at ASYNC_VTABLE (only
        // 4 fields) reads garbage there and crashes.
        crate::bindings::iree_async_semaphore_initialize(
            &VTABLE.async_,
            proactor,
            initial_value,
            0,
            0,
            &mut (*semaphore_ptr).async_sem,
        );
    }
    semaphore_ptr as *mut iree_hal_semaphore_t
}

unsafe extern "C" fn async_destroy(semaphore: *mut iree_async_semaphore_t) {
    unsafe {
        let rs = async_cast(semaphore);
        crate::bindings::iree_async_semaphore_deinitialize(&mut (*rs).async_sem);
        drop(Box::from_raw(rs));
    }
}

unsafe extern "C" fn async_query(semaphore: *mut iree_async_semaphore_t) -> u64 {
    // timeline_value is a real C _Atomic field (iree_atomic_int64_t,
    // bindgen-erased to a plain i64) -- reinterpret as Rust's AtomicI64 to
    // read it correctly rather than a plain (potentially torn) load.
    unsafe {
        let value_ptr =
            &(*semaphore).timeline_value as *const i64 as *const std::sync::atomic::AtomicI64;
        (*value_ptr).load(std::sync::atomic::Ordering::SeqCst) as u64
    }
}

unsafe extern "C" fn async_signal(
    semaphore: *mut iree_async_semaphore_t,
    value: u64,
    frontier: *const iree_async_frontier_t,
) -> iree_status_t {
    let _ = frontier; // no causal frontier tracking (see module doc comment)
    unsafe {
        crate::bindings::iree_async_semaphore_advance_timeline(semaphore, value, std::ptr::null())
    }
}

unsafe extern "C" fn async_on_fail(
    semaphore: *mut iree_async_semaphore_t,
    status_code: iree_status_code_t,
) {
    // TODO: real failure propagation -- iree/async/semaphore.h has
    // iree_async_semaphore_fail() for exactly this, not yet wired up.
    let _ = (semaphore, status_code);
}

static ASYNC_VTABLE: iree_async_semaphore_vtable_t = iree_async_semaphore_vtable_t {
    destroy: Some(async_destroy),
    query: Some(async_query),
    signal: Some(async_signal),
    on_fail: Some(async_on_fail),
};

unsafe extern "C" fn wait(
    semaphore: *mut iree_hal_semaphore_t,
    value: u64,
    timeout: iree_timeout_t,
    flags: iree_async_wait_flags_t,
) -> iree_status_t {
    let rs = unsafe { cast(semaphore) };
    let mut async_ptr: *mut iree_async_semaphore_t = unsafe { &mut (*rs).async_sem };
    let host_allocator = unsafe { (*rs).host_allocator };
    unsafe {
        crate::bindings::iree_async_semaphore_multi_wait(
            iree_async_wait_mode_e_IREE_ASYNC_WAIT_MODE_ALL as u8,
            &mut async_ptr,
            &value,
            1,
            timeout,
            flags,
            host_allocator,
        )
    }
}

status_stub!(import_timepoint(
    semaphore: *mut iree_hal_semaphore_t,
    value: u64,
    queue_affinity: iree_hal_queue_affinity_t,
    external_timepoint: iree_hal_external_timepoint_t,
) -> iree_status_t);

status_stub!(export_timepoint(
    semaphore: *mut iree_hal_semaphore_t,
    value: u64,
    queue_affinity: iree_hal_queue_affinity_t,
    requested_type: iree_hal_external_timepoint_type_t,
    requested_flags: iree_hal_external_timepoint_flags_t,
    out_external_timepoint: *mut iree_hal_external_timepoint_t,
) -> iree_status_t);

pub static VTABLE: iree_hal_semaphore_vtable_t = iree_hal_semaphore_vtable_t {
    async_: ASYNC_VTABLE,
    wait: Some(wait),
    import_timepoint: Some(import_timepoint),
    export_timepoint: Some(export_timepoint),
};
