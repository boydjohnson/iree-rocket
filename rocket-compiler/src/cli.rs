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
    /// Compile to executable-targets and print why each convolution and
    /// matmul candidate is where it is: the shared planner's decision for
    /// each, what ended up on Rocket vs CPU, and the reconciliation between
    /// them.
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

    /// Build the like-for-like CPU-only baseline: same pipeline, same device
    /// topology, same placement pin, but every matcher in the loop is
    /// rewritten so none of them can claim anything and nothing reaches the
    /// NPU. The convolution and matmul matchers are defeated through their
    /// `transform.rocket.match.admitted` line, the pooling and element-wise
    /// ones through their `dim_bounds`; a matcher carrying neither is a
    /// build failure rather than a silently offloading baseline.
    ///
    /// This is the only correct CPU arm for an NPU-vs-CPU comparison. A module
    /// built with plain `iree-compile` never runs the spec's channels-last
    /// conversion and is 2.8x slower for that reason alone -- see ISSUES.md M4.
    #[arg(long)]
    pub no_offload: bool,

    /// Enable the ROADMAP Phase 1 element-wise matchers, which are commented
    /// out in the shipped transform spec.
    ///
    /// Off by default because ISSUES.md P8 measured that at the current
    /// per-dispatch cost more offload sites make a model slower: an
    /// element-wise op does less arithmetic than its own dispatch tax. This
    /// exists so both arms can be measured, not because the default is
    /// provisional -- turn it on, measure against `--no-offload`, and let the
    /// number decide.
    ///
    /// Composes with `--no-offload`: the element-wise entries are enabled
    /// first and then neutralized with everything else, so the baseline arm
    /// runs the identical pipeline.
    #[arg(long)]
    pub elementwise: bool,

    /// Enable ISSUES.md C15: split a static-batch `linalg.batch_matmul` into
    /// one `linalg.matmul` per batch element, so the existing matmul path
    /// claims them. ViT's attention core is twenty-four such dispatch sites
    /// on a twelve-layer model.
    ///
    /// Off by default for the reason `--elementwise` is: it multiplies
    /// dispatch count by the batch -- ViT-B/16's twenty-four sites become
    /// two hundred and eighty-eight -- against a per-dispatch cost ISSUES.md
    /// P8 measured as flat, and which the ViT profile puts at 1.6 ms of
    /// `record` alone. This exists so both arms can be measured. Turn it on,
    /// measure against `--no-offload`, and let the number decide.
    ///
    /// Composes with `--no-offload` the same way `--elementwise` does: the
    /// pass is spliced in first and every matcher neutralized afterwards, so
    /// the baseline runs the identical pipeline.
    #[arg(long)]
    pub batch_matmul: bool,

    #[arg(long, default_value = "rocket_device")]
    pub rocket_device_name: String,

    #[arg(long, default_value = "cpu_device")]
    pub cpu_device_name: String,

    #[arg(long, default_value = "generic")]
    pub llvmcpu_target_cpu: String,

    /// Fail the compile if any convolution or matmul candidate did not reach
    /// the NPU, instead of letting it fall back to the CPU.
    ///
    /// Scope, which is narrower than "everything ran on the NPU". It fails on
    /// a candidate the shared planner refused, one the admission envelope has
    /// no evidence for, one whose form the Rocket lowering cannot express, an
    /// accepted candidate with no Rocket dispatch site to account for it, and
    /// any CPU dispatch whose export name says it is running a convolution or
    /// a matmul. It cannot see an operation IREE fused into a larger
    /// element-wise dispatch, which leaves no named evidence in the module --
    /// so a pass means "nothing observably left on the CPU", not a proof.
    ///
    /// Pointless with `--no-offload`, which exists to put everything on the
    /// CPU; the two together are rejected rather than made to contradict each
    /// other.
    #[arg(long)]
    pub strict_offload: bool,

    /// Fail the compile if any edge between two Rocket dispatches was left
    /// dense by `rocket-assign-layout`: the consumer would repack what the
    /// producer's cube already holds, because the two geometries differ (a
    /// pool's four-rounded surface stride, a channel count the consumer pads)
    /// or the producer publishes no cube. The audit names each such edge and
    /// the failing condition. COMPILER_ROADMAP.md 6.2.
    #[arg(long)]
    pub strict_layout: bool,

    /// Also write the placement audit as JSON to this path: one record per
    /// candidate with its location, kind, shape, precision, decision, reason
    /// code and message, tile summary and limit class, plus the dispatch-site
    /// and hardware-job counts and the reconciliation between them.
    #[arg(long)]
    pub report_json: Option<PathBuf>,

    /// LLVM target triple for the CPU half of the compile, e.g.
    /// `aarch64-linux-gnu` to build a module that runs on the board. Only
    /// affects the `llvm-cpu` executables; Rocket executables are serialized
    /// by the plugin and are target-independent. Defaults to the host triple.
    #[arg(long)]
    pub llvmcpu_target_triple: Option<String>,
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
