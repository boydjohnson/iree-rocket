use std::fmt;

use rocket_runtime_sys as sys;

/// The code of an IREE status, the `IREE_STATUS_*` values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum StatusCode {
    Cancelled,
    Unknown,
    InvalidArgument,
    DeadlineExceeded,
    NotFound,
    AlreadyExists,
    PermissionDenied,
    ResourceExhausted,
    FailedPrecondition,
    Aborted,
    OutOfRange,
    Unimplemented,
    Internal,
    Unavailable,
    DataLoss,
    Unauthenticated,
    Deferred,
    Incompatible,
    /// A code this crate does not name.
    Other(u32),
}

impl StatusCode {
    fn from_raw(code: sys::iree_status_code_t) -> Self {
        use StatusCode::*;
        match code {
            sys::iree_status_code_e_IREE_STATUS_CANCELLED => Cancelled,
            sys::iree_status_code_e_IREE_STATUS_UNKNOWN => Unknown,
            sys::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT => InvalidArgument,
            sys::iree_status_code_e_IREE_STATUS_DEADLINE_EXCEEDED => DeadlineExceeded,
            sys::iree_status_code_e_IREE_STATUS_NOT_FOUND => NotFound,
            sys::iree_status_code_e_IREE_STATUS_ALREADY_EXISTS => AlreadyExists,
            sys::iree_status_code_e_IREE_STATUS_PERMISSION_DENIED => PermissionDenied,
            sys::iree_status_code_e_IREE_STATUS_RESOURCE_EXHAUSTED => ResourceExhausted,
            sys::iree_status_code_e_IREE_STATUS_FAILED_PRECONDITION => FailedPrecondition,
            sys::iree_status_code_e_IREE_STATUS_ABORTED => Aborted,
            sys::iree_status_code_e_IREE_STATUS_OUT_OF_RANGE => OutOfRange,
            sys::iree_status_code_e_IREE_STATUS_UNIMPLEMENTED => Unimplemented,
            sys::iree_status_code_e_IREE_STATUS_INTERNAL => Internal,
            sys::iree_status_code_e_IREE_STATUS_UNAVAILABLE => Unavailable,
            sys::iree_status_code_e_IREE_STATUS_DATA_LOSS => DataLoss,
            sys::iree_status_code_e_IREE_STATUS_UNAUTHENTICATED => Unauthenticated,
            sys::iree_status_code_e_IREE_STATUS_DEFERRED => Deferred,
            sys::iree_status_code_e_IREE_STATUS_INCOMPATIBLE => Incompatible,
            other => Other(other as u32),
        }
    }
}

/// A failed IREE status, or an argument this crate refused before calling
/// into IREE.
///
/// `message` is IREE's own formatting of the status: its code, the source
/// location, the message and every annotation added on the way up, the same
/// text `iree-run-module` prints.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    code: StatusCode,
    message: String,
}

impl Error {
    pub fn code(&self) -> StatusCode {
        self.code
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub(crate) fn new(code: StatusCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    /// Takes ownership of a failed `status` and frees it.
    ///
    /// # Safety
    ///
    /// `status` must be a non-OK status the caller owns and does not use again.
    unsafe fn from_status(status: sys::iree_status_t) -> Self {
        let code = StatusCode::from_raw(sys::iree_status_code(status));
        let allocator = sys::iree_allocator_system();
        let mut buffer: *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut length: sys::iree_host_size_t = 0;
        // SAFETY: status is live; IREE allocates `buffer` from `allocator`.
        let message = if unsafe {
            sys::iree_status_to_string(status, &allocator, &mut buffer, &mut length)
        } {
            // SAFETY: IREE returned `length` valid bytes at `buffer`.
            let bytes = unsafe { std::slice::from_raw_parts(buffer.cast::<u8>(), length) };
            let text = String::from_utf8_lossy(bytes).into_owned();
            unsafe { sys::iree_allocator_free(allocator, buffer.cast()) };
            text
        } else {
            format!("{code:?} (the status could not be formatted)")
        };
        // SAFETY: the caller hands over ownership.
        unsafe { sys::iree_status_free(status) };
        Self { code, message }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// Turns an owned `iree_status_t` into a `Result`, freeing it on failure.
pub(crate) fn check(status: sys::iree_status_t) -> Result<()> {
    if sys::iree_status_is_ok(status) {
        Ok(())
    } else {
        // SAFETY: every caller passes a status it was just handed and owns.
        Err(unsafe { Error::from_status(status) })
    }
}
