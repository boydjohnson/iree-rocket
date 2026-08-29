use crate::{bindings::iree_compiler_session_t, compiler::library::Library};

/// Session options are bootstrapped from whatever was passed to
/// [`Library::setup_global_cl`] -- `ireeCompilerSessionSetFlags` only
/// accepts a curated subset of flags (not e.g.
/// `--iree-hal-indirect-command-buffers`), so unlike a long-running service
/// juggling many differently-configured sessions, this CLI (one subcommand
/// per process) sets flags globally once, the same way `iree-compile` itself
/// does, rather than per-session.
pub struct Session<'lib> {
    pub(crate) library: &'lib Library,
    pub(crate) raw: *mut iree_compiler_session_t,
}

impl<'lib> Session<'lib> {
    pub fn create(library: &'lib Library) -> Self {
        let raw = unsafe { library.api.ireeCompilerSessionCreate() };
        Session { library, raw }
    }
}

impl Drop for Session<'_> {
    fn drop(&mut self) {
        unsafe { self.library.api.ireeCompilerSessionDestroy(self.raw) };
    }
}
