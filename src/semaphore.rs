//! `iree_hal_semaphore_vtable_t` (which embeds `iree_async_semaphore_vtable_t`
//! at `.async_`). Per the research this crate started from: our hardware's
//! completion signal is a blocking `PREP_BO` ioctl, not a native
//! timeline/fence primitive, so this should be backed by a plain host-side
//! futex/condvar-style counter that `device::queue_execute` bumps after a
//! successful blocking wait -- `local_sync`'s synchronous pattern is the
//! model here, not `deferred_work_queue.h`'s async polling.
//! `import_timepoint`/`export_timepoint` (interop with external
//! fence/sync-file handles) aren't needed for this driver and stay
//! UNIMPLEMENTED indefinitely, not just for now.

use crate::bindings::{
    iree_async_frontier_t, iree_async_semaphore_t, iree_async_semaphore_vtable_t,
    iree_async_wait_flags_t, iree_hal_external_timepoint_flags_t, iree_hal_external_timepoint_t,
    iree_hal_external_timepoint_type_t, iree_hal_queue_affinity_t, iree_hal_semaphore_t,
    iree_hal_semaphore_vtable_t, iree_status_code_t, iree_timeout_t,
};

void_stub!(async_destroy(semaphore: *mut iree_async_semaphore_t));

#[allow(unused_variables)]
pub unsafe extern "C" fn async_query(semaphore: *mut iree_async_semaphore_t) -> u64 {
    // TODO: the real one -- return the last-signaled timeline value (the
    // host-side counter device::queue_execute bumps after a successful
    // blocking PREP_BO wait).
    0
}

status_stub!(async_signal(
    semaphore: *mut iree_async_semaphore_t,
    value: u64,
    frontier: *const iree_async_frontier_t,
) -> iree_status_t);

void_stub!(async_on_fail(
    semaphore: *mut iree_async_semaphore_t,
    status_code: iree_status_code_t,
));

pub static ASYNC_VTABLE: iree_async_semaphore_vtable_t = iree_async_semaphore_vtable_t {
    destroy: Some(async_destroy),
    query: Some(async_query),
    signal: Some(async_signal),
    on_fail: Some(async_on_fail),
};

status_stub!(wait(
    semaphore: *mut iree_hal_semaphore_t,
    value: u64,
    timeout: iree_timeout_t,
    flags: iree_async_wait_flags_t,
) -> iree_status_t);

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
