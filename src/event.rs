//! `iree_hal_event_vtable_t`. Mirrors `iree_hal_null_event_create()`
//! (iree-null-driver-reference/event.c) exactly: events are a "WIP API"
//! per IREE's own header, and this driver's synchronization granularity is
//! per-command-buffer anyway (`device::queue_execute`'s blocking
//! SUBMIT/PREP_BO round-trip already orders whole command buffers relative
//! to each other via semaphores) -- there is no finer-grained in-command-
//! buffer schedule for an event to mark a point within. So the event object
//! itself carries no state at all, just a destroyable handle.
//!
//! Unlike the null driver reference, this crate's `command_buffer.rs`
//! `signal_event`/`reset_event`/`wait_events` are real no-op successes
//! (not `status_stub!`/UNIMPLEMENTED) -- CTS's `EventTest` exercises those
//! through a real command buffer and expects them to succeed, and treating
//! them as a no-op execution barrier is exactly what's correct given the
//! per-command-buffer sync granularity above.

use crate::bindings::{
    iree_allocator_t, iree_hal_event_flags_t, iree_hal_event_t, iree_hal_event_vtable_t,
    iree_hal_queue_affinity_t, iree_hal_resource_t, iree_status_t,
};
use crate::status;

#[repr(C)]
pub struct RocketEvent {
    pub resource: iree_hal_resource_t,
    pub host_allocator: iree_allocator_t,
}

unsafe fn cast(event: *mut iree_hal_event_t) -> *mut RocketEvent {
    event as *mut RocketEvent
}

#[allow(unused_variables)]
pub unsafe fn create(
    queue_affinity: iree_hal_queue_affinity_t,
    flags: iree_hal_event_flags_t,
    host_allocator: iree_allocator_t,
    out_event: *mut *mut iree_hal_event_t,
) -> iree_status_t {
    let event = Box::new(RocketEvent {
        resource: iree_hal_resource_t {
            ref_count: 1,
            vtable: &VTABLE as *const _ as *const std::ffi::c_void,
        },
        host_allocator,
    });
    unsafe {
        *out_event = Box::into_raw(event) as *mut iree_hal_event_t;
    }
    status::ok()
}

unsafe extern "C" fn destroy(event: *mut iree_hal_event_t) {
    unsafe { drop(Box::from_raw(cast(event))) }
}

pub static VTABLE: iree_hal_event_vtable_t = iree_hal_event_vtable_t {
    destroy: Some(destroy),
};
