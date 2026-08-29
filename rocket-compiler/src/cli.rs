use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "rocket-compiler",
    about = "Compile and audit Rocket placement for a model"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Full compile to a .vmfb.
    Compile(CompileArgs),
    /// Compile to executable-targets, run the placement-audit pass, and
    /// print a report of what ended up on Rocket vs CPU.
    Audit(AuditArgs),
}

#[derive(Args)]
pub struct CommonArgs {
    /// Path to libIREECompiler.so. Falls back to the IREE_COMPILER_LIB env var.
    #[arg(long)]
    pub iree_compiler_lib: Option<PathBuf>,

    /// Input MLIR file.
    #[arg(long)]
    pub input: PathBuf,

    /// Rocket transform spec .mlir file. Defaults to the one in the sibling
    /// rocket-compiler-plugin checkout.
    #[arg(long)]
    pub transform_spec: Option<PathBuf>,

    #[arg(long, default_value = "rocket_device")]
    pub rocket_device_name: String,

    #[arg(long, default_value = "cpu_device")]
    pub cpu_device_name: String,

    #[arg(long, default_value = "generic")]
    pub llvmcpu_target_cpu: String,
}

#[derive(Args)]
pub struct CompileArgs {
    #[command(flatten)]
    pub common: CommonArgs,

    /// Output .vmfb path.
    #[arg(long)]
    pub output: PathBuf,
}

#[derive(Args)]
pub struct AuditArgs {
    #[command(flatten)]
    pub common: CommonArgs,

    /// Optional path to also write the annotated textual IR to.
    #[arg(long)]
    pub emit_ir: Option<PathBuf>,
}
