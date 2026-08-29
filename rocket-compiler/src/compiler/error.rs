use std::{ffi::CStr, fmt};

use crate::bindings::{IREECompilerApi, iree_compiler_error_t};

/// An error surfaced by an `iree_compiler_error_t*`-returning API call, or a
/// synthetic message for the `bool`-returning calls (parse/pipeline/pass
/// pipeline) which report failure by returning `false` and relying on
/// diagnostics already emitted via the invocation's console/callback
/// diagnostics -- there is no error object to extract a message from in that
/// case.
#[derive(Debug)]
pub struct CompilerError(String);

impl fmt::Display for CompilerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for CompilerError {}

impl CompilerError {
    pub(crate) fn message(message: impl Into<String>) -> Self {
        CompilerError(message.into())
    }

    /// # Safety
    /// `error` must be a non-null pointer returned by an `ireeCompiler*` call
    /// that has not already been destroyed.
    unsafe fn from_raw(api: &IREECompilerApi, error: *mut iree_compiler_error_t) -> Self {
        let message = unsafe {
            let ptr = api.ireeCompilerErrorGetMessage(error);
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        };
        unsafe { api.ireeCompilerErrorDestroy(error) };
        CompilerError(message)
    }
}

/// Converts an `iree_compiler_error_t*` return value into a `Result`,
/// destroying the error object if present.
///
/// # Safety
/// `error` must be either null or a pointer returned by an `ireeCompiler*`
/// call that has not already been destroyed.
pub(crate) unsafe fn check(
    api: &IREECompilerApi,
    error: *mut iree_compiler_error_t,
) -> Result<(), CompilerError> {
    if error.is_null() {
        Ok(())
    } else {
        Err(unsafe { CompilerError::from_raw(api, error) })
    }
}
