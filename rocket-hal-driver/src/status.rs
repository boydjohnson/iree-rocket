//! `iree_status_t` is a tagged pointer (`iree/base/status.h`): a bare status
//! code with no message/payload is directly representable as
//! `(code & IREE_STATUS_CODE_MASK)` cast to the pointer type -- no
//! allocation needed, and the header explicitly documents this as valid
//! ("it's legal to construct an iree_status_t from an iree_status_code_t
//! directly"). `iree_status_from_code`/`iree_ok_status` are C macros
//! (can't cross FFI), so this reimplements the same bit trick directly
//! rather than calling `iree_make_status` (variadic, wants a va_list).

use crate::bindings::{
    iree_status_code_e_IREE_STATUS_DEFERRED, iree_status_code_e_IREE_STATUS_UNIMPLEMENTED,
    iree_status_t,
};

const IREE_STATUS_CODE_MASK: usize = 0x1F;

pub fn from_code(code: u32) -> iree_status_t {
    ((code as usize) & IREE_STATUS_CODE_MASK) as iree_status_t
}

pub fn ok() -> iree_status_t {
    std::ptr::null_mut()
}

pub fn unimplemented() -> iree_status_t {
    from_code(iree_status_code_e_IREE_STATUS_UNIMPLEMENTED)
}

/// The low 5 bits of the tagged pointer are the status code even for a
/// heap-allocated (message-carrying) status -- `status.h`'s own
/// `iree_status_code()` macro relies on the same allocations always being
/// aligned past `IREE_STATUS_CODE_MASK`, so this is safe to apply to any
/// `iree_status_t`, not just the bare-code ones `from_code` builds.
pub fn code(status: iree_status_t) -> usize {
    (status as usize) & IREE_STATUS_CODE_MASK
}

/// True for a callback result of `IREE_STATUS_DEFERRED`: the host-call
/// contract's signal that the callee has taken ownership of signaling (or
/// failing) `signal_semaphore_list` itself, later, and the caller must not
/// touch it.
pub fn is_deferred(status: iree_status_t) -> bool {
    code(status) == iree_status_code_e_IREE_STATUS_DEFERRED as usize
}
