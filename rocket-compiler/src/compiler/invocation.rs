use std::ffi::CString;

use crate::bindings::{
    iree_compiler_invocation_t, iree_compiler_pipeline_t_IREE_COMPILER_PIPELINE_STD,
};
use crate::compiler::error::{CompilerError, check};
use crate::compiler::output::Output;
use crate::compiler::session::Session;
use crate::compiler::source::Source;

pub struct Invocation<'lib> {
    library: &'lib crate::compiler::library::Library,
    raw: *mut iree_compiler_invocation_t,
}

/// Compilation pipelines exposed by `iree_compiler_pipeline_t`. Only `Std`
/// (IREE's full compilation pipeline, the one `--compile-to=<phase>` walks)
/// is needed for this CLI; the other three variants (HAL_EXECUTABLE,
/// PRECOMPILE, VM) are left unmodeled until something actually needs them.
pub enum Pipeline {
    Std,
}

impl<'lib> Invocation<'lib> {
    pub fn create(session: &Session<'lib>) -> Self {
        let raw = unsafe { session.library.api.ireeCompilerInvocationCreate(session.raw) };
        Invocation {
            library: session.library,
            raw,
        }
    }

    /// Enables default, pretty-printed diagnostics to the console -- the
    /// right choice for a command-line tool.
    pub fn enable_console_diagnostics(&self) {
        unsafe {
            self.library
                .api
                .ireeCompilerInvocationEnableConsoleDiagnostics(self.raw)
        };
    }

    /// Mnemonic IREEVMPipelinePhase name to stop compilation at, e.g.
    /// `"executable-targets"` (the CLI's `--compile-to=executable-targets`).
    /// Default is `"end"`.
    pub fn set_compile_to_phase(&self, phase: &str) {
        let c_phase = CString::new(phase).expect("phase name must not contain NUL");
        unsafe {
            self.library
                .api
                .ireeCompilerInvocationSetCompileToPhase(self.raw, c_phase.as_ptr())
        };
    }

    pub fn parse_source(&self, source: &Source<'lib>) -> Result<(), CompilerError> {
        let ok = unsafe {
            self.library
                .api
                .ireeCompilerInvocationParseSource(self.raw, source.raw)
        };
        if ok {
            Ok(())
        } else {
            Err(CompilerError::message(
                "failed to parse source (see diagnostics above)",
            ))
        }
    }

    pub fn run_pipeline(&self, pipeline: Pipeline) -> Result<(), CompilerError> {
        let raw_pipeline = match pipeline {
            Pipeline::Std => iree_compiler_pipeline_t_IREE_COMPILER_PIPELINE_STD,
        };
        let ok = unsafe {
            self.library
                .api
                .ireeCompilerInvocationPipeline(self.raw, raw_pipeline)
        };
        if ok {
            Ok(())
        } else {
            Err(CompilerError::message(
                "compilation pipeline failed (see diagnostics above)",
            ))
        }
    }

    /// Runs an arbitrary named pass/pass-pipeline against the invocation's
    /// current in-memory module -- the FFI equivalent of
    /// `iree-opt --pass-pipeline=<textPassPipeline>`. Passes registered by
    /// statically-linked plugins (e.g. this repo's `rocket-annotate-final-placement`)
    /// are resolved by name through MLIR's global pass registry, with no
    /// extra wiring needed on the Rust side.
    pub fn run_pass_pipeline(&self, text_pass_pipeline: &str) -> Result<(), CompilerError> {
        let c_text =
            CString::new(text_pass_pipeline).expect("pass pipeline text must not contain NUL");
        let ok = unsafe {
            self.library
                .api
                .ireeCompilerInvocationRunPassPipeline(self.raw, c_text.as_ptr())
        };
        if ok {
            Ok(())
        } else {
            Err(CompilerError::message(
                "pass pipeline failed (see diagnostics above)",
            ))
        }
    }

    /// Outputs the invocation's current state as textual IR.
    pub fn output_ir(&self, output: &Output<'lib>) -> Result<(), CompilerError> {
        let error = unsafe {
            self.library
                .api
                .ireeCompilerInvocationOutputIR(self.raw, output.raw)
        };
        unsafe { check(&self.library.api, error) }
    }

    /// Outputs the invocation's current state (after a full `Pipeline::Std`
    /// run) as serialized VM bytecode (a `.vmfb`).
    pub fn output_vm_bytecode(&self, output: &Output<'lib>) -> Result<(), CompilerError> {
        let error = unsafe {
            self.library
                .api
                .ireeCompilerInvocationOutputVMBytecode(self.raw, output.raw)
        };
        unsafe { check(&self.library.api, error) }
    }
}

impl Drop for Invocation<'_> {
    fn drop(&mut self) {
        unsafe { self.library.api.ireeCompilerInvocationDestroy(self.raw) };
    }
}
