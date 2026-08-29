use std::{ffi::CString, os::raw::c_void, path::Path, ptr};

use crate::{
    bindings::iree_compiler_output_t,
    compiler::{
        error::{CompilerError, check},
        library::Library,
    },
};

/// Outputs are not bound to a session -- they can outlive it -- so this only
/// borrows the `Library` that owns the API entry points, not a `Session`.
pub struct Output<'lib> {
    library: &'lib Library,
    pub(crate) raw: *mut iree_compiler_output_t,
}

impl<'lib> Output<'lib> {
    pub fn open_file(library: &'lib Library, path: &Path) -> Result<Self, CompilerError> {
        let c_path = CString::new(path.to_string_lossy().as_bytes())
            .map_err(|_| CompilerError::message("output path must not contain NUL"))?;
        let mut raw: *mut iree_compiler_output_t = ptr::null_mut();
        let error = unsafe {
            library
                .api
                .ireeCompilerOutputOpenFile(c_path.as_ptr(), &mut raw)
        };
        unsafe { check(&library.api, error) }?;
        Ok(Output { library, raw })
    }

    pub fn open_membuffer(library: &'lib Library) -> Result<Self, CompilerError> {
        let mut raw: *mut iree_compiler_output_t = ptr::null_mut();
        let error = unsafe { library.api.ireeCompilerOutputOpenMembuffer(&mut raw) };
        unsafe { check(&library.api, error) }?;
        Ok(Output { library, raw })
    }

    /// Commits a file/persistent output. Without this, it is deleted on drop.
    pub fn keep(&self) {
        unsafe { self.library.api.ireeCompilerOutputKeep(self.raw) };
    }

    /// Maps the contents written to a membuffer-backed output. Only valid
    /// for outputs created via [`Output::open_membuffer`].
    pub fn map_memory(&self) -> Result<&[u8], CompilerError> {
        let mut contents: *mut c_void = ptr::null_mut();
        let mut size: u64 = 0;
        let error = unsafe {
            self.library
                .api
                .ireeCompilerOutputMapMemory(self.raw, &mut contents, &mut size)
        };
        unsafe { check(&self.library.api, error) }?;
        Ok(unsafe { std::slice::from_raw_parts(contents as *const u8, size as usize) })
    }
}

impl Drop for Output<'_> {
    fn drop(&mut self) {
        unsafe { self.library.api.ireeCompilerOutputDestroy(self.raw) };
    }
}
