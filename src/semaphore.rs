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

/// Allocation layout for a `RocketSemaphore` plus its *trailing* frontier
/// storage, and the byte offset (from the allocation's start) at which
/// that storage begins. `frontier_capacity=0` (we don't do causal
/// frontier tracking -- only relevant for import/export_timepoint, which
/// stay UNIMPLEMENTED below) means there are no frontier *entries*, but
/// `iree_async_frontier_t` still needs its own header (`entry_count` +
/// `reserved`, 8 bytes) written somewhere -- `frontier_offset` tells
/// `iree_async_semaphore_initialize` where. This used to be passed as a
/// bare `0`, which put that header at the very start of the semaphore
/// (aliasing `RocketSemaphore.async_sem.ref_count`/`vtable`): harmless-
/// looking at first since nothing read those bytes as a frontier, but
/// `iree_async_frontier_initialize(frontier, 0)` unconditionally writes
/// `entry_count = 0` then `memset(reserved, 0, 7)` -- 8 zero bytes at
/// offset 0, permanently stomping `ref_count` back to 0 immediately after
/// `iree_atomic_ref_count_init` set it to 1 moments earlier in the same
/// `iree_async_semaphore_initialize` call. Every semaphore's true
/// refcount has been 0 from the instant of creation ever since -- silent
/// as long as only the single creator ever released it (decrementing from
/// 0 never satisfies the "was 1" destroy-trigger check, so the object just
/// leaks rather than double-frees), but device.rs's `run_after_wait`
/// retaining a semaphore independently is the first thing that ever
/// legitimately brought the count to a real 1 -- and its matching release
/// then correctly saw "was 1" and freed the object out from under the
/// CTS test's own still-live reference. This layout, mirroring
/// `iree_async_semaphore_layout()`'s own (non-bindgen'd, static-inline)
/// formula, places the frontier header in real trailing space after
/// `RocketSemaphore` instead.
fn semaphore_layout() -> (std::alloc::Layout, usize) {
    let base_size = std::mem::size_of::<RocketSemaphore>();
    let frontier_align = std::mem::align_of::<iree_async_frontier_t>();
    let frontier_offset = base_size.next_multiple_of(frontier_align);
    let total_size = frontier_offset + std::mem::size_of::<iree_async_frontier_t>();
    let align = std::mem::align_of::<RocketSemaphore>().max(frontier_align);
    (
        std::alloc::Layout::from_size_align(total_size, align).unwrap(),
        frontier_offset,
    )
}

/// Mirrors `iree_hal_null_semaphore_create()`. Not called from anywhere
/// yet -- `device::create_semaphore` (still UNIMPLEMENTED) is what will
/// eventually call this with a proactor obtained from the device's pool.
pub unsafe fn create(
    proactor: *mut iree_async_proactor_t,
    initial_value: u64,
    host_allocator: iree_allocator_t,
) -> *mut iree_hal_semaphore_t {
    let (layout, frontier_offset) = semaphore_layout();
    let semaphore_ptr = unsafe { std::alloc::alloc(layout) } as *mut RocketSemaphore;
    if semaphore_ptr.is_null() {
        std::alloc::handle_alloc_error(layout);
    }
    unsafe {
        std::ptr::write(
            semaphore_ptr,
            RocketSemaphore {
                async_sem: std::mem::zeroed(),
                host_allocator,
            },
        );
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
            frontier_offset as crate::bindings::iree_host_size_t,
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
        std::ptr::drop_in_place(rs);
        let (layout, _) = semaphore_layout();
        std::alloc::dealloc(rs as *mut u8, layout);
    }
}

unsafe extern "C" fn async_query(semaphore: *mut iree_async_semaphore_t) -> u64 {
    // Mirrors local_sync/sync_semaphore.c's iree_hal_sync_semaphore_query:
    // iree_hal_semaphore_query() (hal/semaphore.c) only ever looks at this
    // vtable's return value, not failure_status directly -- a failed
    // semaphore must encode that itself via
    // IREE_HAL_SEMAPHORE_FAILURE_VALUE_STATUS_BIT | (status as u64), or
    // callers see a hollow OK/0 forever (the CTS SemaphoreTest.Failure family
    // this was breaking).
    unsafe {
        let failure_ptr = &(*semaphore).failure_status as *const isize
            as *const std::sync::atomic::AtomicIsize;
        let failure = (*failure_ptr).load(std::sync::atomic::Ordering::Acquire);
        if failure != 0 {
            return crate::bindings::IREE_HAL_SEMAPHORE_FAILURE_VALUE_STATUS_BIT as u64
                | (failure as u64);
        }
        // timeline_value is a real C _Atomic field (iree_atomic_int64_t,
        // bindgen-erased to a plain i64) -- reinterpret as Rust's AtomicI64 to
        // read it correctly rather than a plain (potentially torn) load.
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
        // advance_timeline's own doc comment is explicit that it does NOT
        // dispatch timepoints -- callers must do that themselves after
        // signaling. Mirrors local_sync/sync_semaphore.c's
        // iree_hal_sync_semaphore_signal exactly. Omitting the
        // dispatch_timepoints call left any waiter blocked in
        // iree_async_semaphore_multi_wait's general (non-fast-path) case
        // parked on a timepoint that never fires -- CTS's
        // SemaphoreThreadTest.WaitLaterSignaledBeyond (waiter blocks before
        // the signaling thread runs) hung forever on exactly this.
        let status =
            crate::bindings::iree_async_semaphore_advance_timeline(semaphore, value, std::ptr::null());
        if status.is_null() {
            crate::bindings::iree_async_semaphore_dispatch_timepoints(semaphore, value);
        }
        status
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
