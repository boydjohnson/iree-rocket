use std::{ffi::CString, path::Path, ptr};

use crate::{
    bindings::iree_compiler_source_t,
    compiler::{
        error::{CompilerError, check},
        library::Library,
        session::Session,
    },
};

pub struct Source<'lib> {
    library: &'lib Library,
    pub(crate) raw: *mut iree_compiler_source_t,
}

impl<'lib> Source<'lib> {
    pub fn open_file(session: &Session<'lib>, path: &Path) -> Result<Self, CompilerError> {
        let c_path = CString::new(path.to_string_lossy().as_bytes())
            .map_err(|_| CompilerError::message("input path must not contain NUL"))?;
        let mut raw: *mut iree_compiler_source_t = ptr::null_mut();
        let error = unsafe {
            session
                .library
                .api
                .ireeCompilerSourceOpenFile(session.raw, c_path.as_ptr(), &mut raw)
        };
        unsafe { check(&session.library.api, error) }?;
        Ok(Source {
            library: session.library,
            raw,
        })
    }
}

impl Drop for Source<'_> {
    fn drop(&mut self) {
        unsafe { self.library.api.ireeCompilerSourceDestroy(self.raw) };
    }
}
