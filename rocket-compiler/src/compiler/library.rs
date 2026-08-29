use std::ffi::{CString, OsStr};
use std::os::raw::c_char;

use crate::bindings::IREECompilerApi;

/// A loaded `libIREECompiler.so`. Global compiler state (`ireeCompilerGlobalInitialize`
/// / `ireeCompilerGlobalShutdown`) is tied to this value's lifetime -- one `Library`
/// is expected per process (this CLI never constructs more than one), so no
/// reference counting is needed to keep the init/shutdown calls balanced.
pub struct Library {
    pub(crate) api: IREECompilerApi,
}

impl Library {
    /// Loads `libIREECompiler.so` from `path` and initializes the compiler.
    ///
    /// # Safety
    /// The loaded library must actually be a build of IREE's compiler
    /// embedding API matching the vendored header this crate binds against.
    /// Loading an unrelated or ABI-incompatible library is undefined behavior.
    pub unsafe fn load(path: impl AsRef<OsStr>) -> Result<Self, libloading::Error> {
        let api = unsafe { IREECompilerApi::new(path)? };
        unsafe { api.ireeCompilerGlobalInitialize() };
        Ok(Library { api })
    }

    /// Parses `flags` (each entry a single CLI argument, no program name)
    /// as IREE's global command-line options -- the same mechanism
    /// `iree-compile` itself uses for everything except the input/output
    /// file arguments. Must be called at most once per process, before any
    /// [`crate::compiler::Session`] is created.
    pub fn setup_global_cl(&self, flags: &[String]) {
        let banner = CString::new("rocket-compiler").unwrap();
        let program_name = CString::new("rocket-compiler").unwrap();
        let cstrings: Vec<CString> = flags
            .iter()
            .map(|flag| CString::new(flag.as_str()).expect("flag must not contain NUL"))
            .collect();
        let mut argv: Vec<*const c_char> = std::iter::once(program_name.as_ptr())
            .chain(cstrings.iter().map(|c| c.as_ptr()))
            .collect();
        unsafe {
            self.api.ireeCompilerSetupGlobalCL(
                argv.len() as i32,
                argv.as_mut_ptr(),
                banner.as_ptr(),
                /*installSignalHandlers=*/ true,
            )
        };
    }
}

impl Drop for Library {
    fn drop(&mut self) {
        // Must run before `self.api`'s internal libloading::Library is
        // dropped (which dlcloses the .so) -- custom Drop::drop runs before
        // a struct's ordinary field drops, so this ordering is automatic.
        unsafe { self.api.ireeCompilerGlobalShutdown() };
    }
}
