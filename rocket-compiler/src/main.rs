mod bindings;
mod cli;
mod compiler;
mod report;

use std::{
    env,
    error::Error,
    path::{Path, PathBuf},
    process::ExitCode,
};

use clap::Parser;

use compiler::{Invocation, Library, Output, Pipeline, Session, Source};

fn main() -> ExitCode {
    let cli = cli::Cli::parse();
    let result = match &cli.command {
        cli::Command::Compile(args) => run_compile(args),
        cli::Command::Audit(args) => run_audit(args),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

fn resolve_lib_path(explicit: Option<&Path>) -> Result<PathBuf, Box<dyn Error>> {
    if let Some(path) = explicit {
        return Ok(path.to_path_buf());
    }
    if let Ok(path) = env::var("IREE_COMPILER_LIB") {
        return Ok(PathBuf::from(path));
    }
    Err("no --iree-compiler-lib given and IREE_COMPILER_LIB is not set".into())
}

fn default_transform_spec_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../rocket-compiler-plugin/target/Rocket/rocket_conv2d_transform_spec.mlir")
}

fn compile_flags(common: &cli::CommonArgs) -> Vec<String> {
    let transform_spec = common
        .transform_spec
        .clone()
        .unwrap_or_else(default_transform_spec_path);
    vec![
        format!(
            "--iree-preprocessing-transform-spec-filename={}",
            transform_spec.display()
        ),
        format!(
            "--iree-hal-target-device={}=rocket",
            common.rocket_device_name
        ),
        format!("--iree-hal-target-device={}=local", common.cpu_device_name),
        "--iree-hal-local-target-device-backends=llvm-cpu".to_string(),
        format!("--iree-llvmcpu-target-cpu={}", common.llvmcpu_target_cpu),
        format!("--iree-hal-default-device={}", common.cpu_device_name),
        "--iree-hal-indirect-command-buffers=false".to_string(),
    ]
}

fn run_compile(args: &cli::CompileArgs) -> Result<(), Box<dyn Error>> {
    let lib_path = resolve_lib_path(args.common.iree_compiler_lib.as_deref())?;
    let library = unsafe { Library::load(&lib_path) }?;

    library.setup_global_cl(&compile_flags(&args.common));
    let session = Session::create(&library);

    let source = Source::open_file(&session, &args.common.input)?;
    let invocation = Invocation::create(&session);
    invocation.enable_console_diagnostics();
    invocation.parse_source(&source)?;
    invocation.run_pipeline(Pipeline::Std)?;

    let output = Output::open_file(&library, &args.output)?;
    invocation.output_vm_bytecode(&output)?;
    output.keep();

    println!("wrote {}", args.output.display());
    Ok(())
}

fn run_audit(args: &cli::AuditArgs) -> Result<(), Box<dyn Error>> {
    let lib_path = resolve_lib_path(args.common.iree_compiler_lib.as_deref())?;
    let library = unsafe { Library::load(&lib_path) }?;

    library.setup_global_cl(&compile_flags(&args.common));
    let session = Session::create(&library);

    let source = Source::open_file(&session, &args.common.input)?;
    let invocation = Invocation::create(&session);
    invocation.enable_console_diagnostics();
    invocation.set_compile_to_phase("executable-targets");
    invocation.parse_source(&source)?;
    invocation.run_pipeline(Pipeline::Std)?;
    // Bare pass name, not "builtin.module(rocket-annotate-final-placement)":
    // the invocation's PassManager (unlike iree-opt's generic tool machinery)
    // is already module-anchored, so wrapping it re-nests one level too deep
    // and silently matches zero ops instead of erroring.
    invocation.run_pass_pipeline("rocket-annotate-final-placement")?;

    let output = Output::open_membuffer(&library)?;
    invocation.output_ir(&output)?;
    let ir_bytes = output.map_memory()?;
    let ir_text = String::from_utf8_lossy(ir_bytes);

    if let Some(path) = &args.emit_ir {
        std::fs::write(path, ir_text.as_bytes())?;
    }

    let report = report::PlacementReport::scan(&ir_text);
    print!("{report}");
    Ok(())
}
