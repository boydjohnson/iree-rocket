mod audit;
mod bindings;
mod cli;
mod compiler;
mod decisions;
mod layout;
mod report;
mod spec;

use std::{
    collections::BTreeSet,
    env,
    error::Error,
    fs,
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

fn transform_spec_path(common: &cli::CommonArgs) -> PathBuf {
    common
        .transform_spec
        .clone()
        .unwrap_or_else(default_transform_spec_path)
}

/// The transform spec an invocation will actually be given, plus ownership of
/// it when `--no-offload` had to derive one.
struct SpecFile {
    path: PathBuf,
    temporary: bool,
}

impl SpecFile {
    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for SpecFile {
    fn drop(&mut self) {
        if self.temporary {
            let _ = fs::remove_file(&self.path);
        }
    }
}

/// Resolves the spec to compile with, deriving the no-offload variant when
/// asked. The derived spec goes to a file because IREE takes it by filename
/// (`--iree-preprocessing-transform-spec-filename`); it is removed when the
/// returned handle drops.
fn resolve_transform_spec(common: &cli::CommonArgs) -> Result<SpecFile, Box<dyn Error>> {
    let source = transform_spec_path(common);
    if !common.no_offload && !common.elementwise && !common.batch_matmul {
        return Ok(SpecFile {
            path: source,
            temporary: false,
        });
    }

    let mut text = fs::read_to_string(&source)
        .map_err(|err| format!("failed to read transform spec {}: {err}", source.display()))?;

    // Order matters, and only one way round is correct: enable first, then
    // neutralize. `neutralize` refuses a matcher in the `foreach_match` list
    // that constrains no dimension, and it rewrites the bounds of every
    // matcher it finds there -- so running it second means the element-wise
    // matchers are checked and defeated along with the rest, and
    // `--no-offload --elementwise` is a true baseline for
    // `--elementwise`. The other order would leave them live in the
    // "no-offload" arm.
    if common.elementwise {
        let enabled = spec::enable_elementwise(&text)?;
        eprintln!(
            "--elementwise: {} matcher entries enabled in {}",
            enabled.enabled,
            source.display()
        );
        text = enabled.text;
    }

    // Before neutralize for the reason elementwise is: the baseline arm has
    // to run the identical pipeline, so the pass is spliced in and *then*
    // every matcher is defeated.
    if common.batch_matmul {
        let enabled = spec::enable_batch_matmul(&text)?;
        eprintln!(
            "--batch-matmul: rocket-unbatch-matmul spliced in ({} line(s)) in {}",
            enabled.enabled,
            source.display()
        );
        text = enabled.text;
    }

    let mut suffix = if common.batch_matmul {
        "batch-matmul"
    } else {
        "elementwise"
    };
    if common.no_offload {
        suffix = "no-offload";
        let neutralized = spec::neutralize(&text)?;
        // Said out loud because a baseline that silently stopped being a
        // baseline is the failure this mode exists to prevent (ISSUES.md M4).
        eprintln!(
            "--no-offload: {} matchers defeated, {} dim_bounds rewritten in {}",
            neutralized.matchers,
            neutralized.rewritten,
            source.display()
        );
        text = neutralized.text;
    }

    let path = env::temp_dir().join(format!(
        "rocket-compiler-{suffix}-{}.mlir",
        std::process::id()
    ));
    fs::write(&path, &text).map_err(|err| format!("failed to write {}: {err}", path.display()))?;

    Ok(SpecFile {
        path,
        temporary: true,
    })
}

fn compile_flags(common: &cli::CommonArgs, transform_spec: &Path) -> Vec<String> {
    let mut flags = vec![
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
    ];
    // Left off entirely when unset so IREE keeps its own host-triple default,
    // rather than us guessing a spelling for the host here.
    if let Some(triple) = &common.llvmcpu_target_triple {
        flags.push(format!("--iree-llvmcpu-target-triple={triple}"));
    }
    flags
}

/// Collects the device globals a transform spec refers to: every `@symbol`
/// inside a `#hal.device.*` attribute, i.e. `#hal.device.affinity<@d>` and
/// the endpoints of `#hal.device.topology<links = [(@a -> @b = {...})]>`.
///
/// The spec hardcodes these names, but the device globals themselves are
/// created from `--iree-hal-target-device=<name>=...`, which this CLI derives
/// from `--rocket-device-name` / `--cpu-device-name`. Renaming either one
/// leaves the spec's references dangling.
fn spec_device_symbols(spec: &str) -> BTreeSet<&str> {
    const MARKER: &str = "#hal.device.";
    let mut symbols = BTreeSet::new();
    for (start, _) in spec.match_indices(MARKER) {
        let rest = &spec[start + MARKER.len()..];
        // Only step over the attribute's mnemonic (`affinity`, `topology`,
        // ...); if what follows isn't a `<...>` body this is not an attribute
        // we can read, and scanning on would run into unrelated text.
        let Some(open) = rest.find(|c: char| !(c.is_alphanumeric() || c == '_' || c == '.')) else {
            continue;
        };
        if !rest[open..].starts_with('<') {
            continue;
        }
        symbols.extend(scan_symbol_refs(attribute_body(&rest[open + 1..])));
    }
    symbols
}

/// Returns the text up to the `>` closing an already-opened `<`, tracking
/// nesting. `->` and `=>` are skipped: MLIR spells topology links with an
/// arrow, whose `>` does not close anything.
fn attribute_body(body: &str) -> &str {
    let mut depth = 1usize;
    let mut previous = ' ';
    for (i, c) in body.char_indices() {
        match c {
            '<' => depth += 1,
            '>' if previous != '-' && previous != '=' => {
                depth -= 1;
                if depth == 0 {
                    return &body[..i];
                }
            }
            _ => {}
        }
        previous = c;
    }
    body
}

/// Yields each bare `@symbol` in `text`, without the leading `@`. Stops a name
/// at the first character that can't appear in an unquoted MLIR symbol.
fn scan_symbol_refs(text: &str) -> impl Iterator<Item = &str> {
    text.match_indices('@').map(|(at, _)| {
        let start = at + 1;
        let end = text[start..]
            .find(|c: char| !(c.is_alphanumeric() || c == '_' || c == '$' || c == '.'))
            .map(|i| start + i)
            .unwrap_or(text.len());
        &text[start..end]
    })
}

/// Fails if the transform spec names a device global this invocation won't
/// create.
///
/// Worth checking up front because the failure is otherwise awful: these
/// references live in attributes, and attributes are not symbol-verified, so
/// a dangling one survives until a later pass resolves it and dereferences
/// null -- the compiler segfaults instead of reporting an error.
fn check_spec_device_names(
    common: &cli::CommonArgs,
    transform_spec: &Path,
) -> Result<(), Box<dyn Error>> {
    let spec = fs::read_to_string(transform_spec).map_err(|err| {
        format!(
            "failed to read transform spec {}: {err}",
            transform_spec.display()
        )
    })?;
    let declared = [
        common.rocket_device_name.as_str(),
        common.cpu_device_name.as_str(),
    ];
    let dangling: Vec<&str> = spec_device_symbols(&spec)
        .into_iter()
        .filter(|symbol| !declared.contains(symbol))
        .collect();
    if dangling.is_empty() {
        return Ok(());
    }
    let named: Vec<String> = dangling.iter().map(|s| format!("@{s}")).collect();
    Err(format!(
        "transform spec {spec_path} refers to device global(s) {dangling} that this \
         invocation does not create: it declares @{rocket} (--rocket-device-name) and \
         @{cpu} (--cpu-device-name). The spec hardcodes its device names, so those \
         flags must match it -- the defaults do. Compiling anyway would crash the \
         compiler rather than report an error.",
        spec_path = transform_spec.display(),
        dangling = named.join(", "),
        rocket = common.rocket_device_name,
        cpu = common.cpu_device_name,
    )
    .into())
}

/// The phase `rocket-plan-candidates` runs in, and therefore the only phase
/// at which its decision record is certainly still on the function.
///
/// The record does survive to `executable-targets` on every model measured
/// here, so reading it out of the placement dump would work today. It is
/// taken at its own phase anyway: whether a discardable attribute survives
/// is a property of which passes IREE happens to run, and `report.rs` above
/// documents the `rocket.origin` tags that did not. A placement report that
/// silently loses its "why" the next time a pass is added is exactly what
/// COMPILER_ROADMAP.md section 3 asks not to build.
const DECISIONS_PHASE: &str = "preprocessing";

/// Runs the pipeline up to the end of preprocessing, reads the decision
/// record off the IR, and leaves the invocation set to resume from there.
fn capture_decisions(
    library: &Library,
    invocation: &Invocation,
) -> Result<decisions::DecisionRecord, Box<dyn Error>> {
    invocation.set_compile_to_phase(DECISIONS_PHASE);
    invocation.run_pipeline(Pipeline::Std)?;
    let output = Output::open_membuffer(library)?;
    invocation.output_ir(&output)?;
    let ir_bytes = output.map_memory()?;
    let record = decisions::DecisionRecord::scan(&String::from_utf8_lossy(ir_bytes));
    invocation.set_compile_from_phase(DECISIONS_PHASE);
    Ok(record)
}

/// Runs the rest of the pipeline to `executable-targets`, tags placement,
/// and reads the resulting module. Leaves the invocation set to resume.
fn capture_placement(
    library: &Library,
    invocation: &Invocation,
    emit_ir: Option<&Path>,
) -> Result<report::PlacementReport, Box<dyn Error>> {
    invocation.set_compile_to_phase("executable-targets");
    invocation.run_pipeline(Pipeline::Std)?;
    // Bare pass name, not "builtin.module(rocket-annotate-final-placement)":
    // the invocation's PassManager (unlike iree-opt's generic tool machinery)
    // is already module-anchored, so wrapping it re-nests one level too deep
    // and silently matches zero ops instead of erroring.
    invocation.run_pass_pipeline("rocket-annotate-final-placement")?;
    let output = Output::open_membuffer(library)?;
    invocation.output_ir(&output)?;
    let ir_bytes = output.map_memory()?;
    let ir_text = String::from_utf8_lossy(ir_bytes);
    if let Some(path) = emit_ir {
        fs::write(path, ir_text.as_bytes())?;
    }
    let report = report::PlacementReport::scan(&ir_text);
    invocation.set_compile_from_phase("executable-targets");
    Ok(report)
}

/// The phase the Rocket placement pin runs at. `flow` is the last point where
/// every dispatch in the program is still a `flow.dispatch` carrying a plain
/// `stream.affinity` attribute: dispatch regions have been formed and
/// outlined, and Stream's affinity analysis -- which is what pulls an
/// IREE-formed dispatch onto the NPU when its only consumer is a Rocket one
/// -- has not run yet.
const PIN_PHASE: &str = "flow";

/// Registered by the compiler plugin; see RocketPinUnclaimedDispatchesPass.cpp.
const PIN_PASS: &str = "rocket-pin-unclaimed-dispatches";

/// Registered by the compiler plugin; see RocketAssignLayoutPass.cpp. Runs at
/// the same phase as the pin for the same reason: it needs every reader of a
/// Rocket dispatch's result to be a formed dispatch, and it needs the
/// dispatch's push constants to still be plain SSA operands. It writes the
/// `rocket.layout_decisions` record the audit prints (COMPILER_ROADMAP.md
/// 6.2).
const ASSIGN_LAYOUT_PASS: &str = "rocket-assign-layout";

/// Registered by the compiler plugin; see RocketPackWeightsPass.cpp. Runs
/// after the reader count, at the same phase: a filter is a
/// `util.global.load` of a folded constant by now, and the dispatch's
/// dimensions are still constant push-constant operands it can read.
/// COMPILER_ROADMAP.md 6.3.
const PACK_WEIGHTS_PASS: &str = "rocket-pack-weights";

/// Runs `Pipeline::Std` up to and including the `flow` phase and pins every
/// dispatch the Rocket transform spec did not explicitly claim to the
/// default (CPU) device, then leaves the invocation set to resume from
/// `flow`. The caller sets its own compile-to phase and runs the pipeline
/// again to continue.
///
/// This is split out of a single `Pipeline::Std` run because the Rocket
/// backend has no codegen at all -- `serializeExecutable` only knows how to
/// read the config dict the transform spec stamps onto its hand-authored
/// executables. Anything else that reaches it (an auto-formed pad copy, say)
/// fails to serialize, so "Rocket runs only what the spec put there" has to
/// be enforced rather than hoped for, and the only hook a plugin gets --
/// `extendPreprocessingPassPipeline` -- runs long before dispatches exist.
///
/// When `capture` is set the layout record is read off the IR right after
/// `rocket-assign-layout` wrote it -- at the phase that produced it, for the
/// same reason `capture_decisions` reads its record at preprocessing.
fn pin_unclaimed_dispatches(
    library: &Library,
    invocation: &Invocation,
    capture: bool,
) -> Result<layout::LayoutRecord, Box<dyn Error>> {
    invocation.set_compile_to_phase(PIN_PHASE);
    invocation.run_pipeline(Pipeline::Std)?;
    invocation.run_pass_pipeline(PIN_PASS)?;
    invocation.run_pass_pipeline(ASSIGN_LAYOUT_PASS)?;
    let record = if capture {
        let output = Output::open_membuffer(library)?;
        invocation.output_ir(&output)?;
        let ir_bytes = output.map_memory()?;
        layout::LayoutRecord::scan(&String::from_utf8_lossy(ir_bytes))
    } else {
        layout::LayoutRecord::default()
    };
    invocation.run_pass_pipeline(PACK_WEIGHTS_PASS)?;
    invocation.set_compile_from_phase(PIN_PHASE);
    Ok(record)
}

/// The `--no-offload` baseline's last line of defence: after placement is
/// pinned, refuse to emit a module with a single Rocket executable in it.
///
/// `spec::neutralize` defeats each matcher through whichever predicate it
/// carries -- `no_offload` on `transform.rocket.match.admitted` for the
/// convolution and matmul matchers, a rewritten `dim_bounds` for the pooling
/// and element-wise ones -- and refuses a matcher carrying neither. That
/// check is on the spec text; this one is on the module, and it is the
/// property the baseline actually needs. A "CPU-only" arm that quietly
/// offloads is exactly the error ISSUES.md M4 exists to prevent, and it must
/// fail loudly here rather than skew a measurement.
fn assert_nothing_offloaded(report: &report::PlacementReport) -> Result<(), Box<dyn Error>> {
    if report.rocket_executables.is_empty() {
        return Ok(());
    }
    let names: Vec<&str> = report
        .rocket_executables
        .iter()
        .map(|executable| executable.name.as_str())
        .collect();
    Err(format!(
        "--no-offload produced {} Rocket executable(s) ({}); the neutralized spec did not \
         defeat every matcher, so this is not a CPU-only baseline. Refusing to write it.",
        names.len(),
        names.join(", "),
    )
    .into())
}

/// Whether this invocation has to stop at `executable-targets` to look at
/// the module before finishing the compile.
fn needs_placement_pass(common: &cli::CommonArgs) -> bool {
    common.no_offload
        || common.strict_offload
        || common.strict_layout
        || common.report_json.is_some()
}

/// Writes the machine-readable report when one was asked for, and applies
/// the strict-offload gate.
fn deliver_audit(
    common: &cli::CommonArgs,
    audit: &audit::PlacementAudit,
) -> Result<(), Box<dyn Error>> {
    if let Some(path) = &common.report_json {
        fs::write(path, audit.to_json())
            .map_err(|err| format!("failed to write {}: {err}", path.display()))?;
        eprintln!("wrote placement report {}", path.display());
    }
    if common.strict_offload
        && let Some(failure) = audit.strict_offload_failure()
    {
        return Err(failure.into());
    }
    if common.strict_layout
        && let Some(failure) = audit.strict_layout_failure()
    {
        return Err(failure.into());
    }
    Ok(())
}

/// `--strict-offload` and `--no-offload` ask for opposite things; letting
/// both through would make the baseline arm fail on purpose.
fn check_offload_flags(common: &cli::CommonArgs) -> Result<(), Box<dyn Error>> {
    if common.no_offload && common.strict_offload {
        return Err(
            "--strict-offload and --no-offload are contradictory: the baseline arm \
                    exists to keep every candidate on the CPU"
                .into(),
        );
    }
    Ok(())
}

fn run_compile(args: &cli::CompileArgs) -> Result<(), Box<dyn Error>> {
    check_offload_flags(&args.common)?;
    let lib_path = resolve_lib_path(args.common.iree_compiler_lib.as_deref())?;
    let transform_spec = resolve_transform_spec(&args.common)?;
    check_spec_device_names(&args.common, transform_spec.path())?;
    let library = unsafe { Library::load(&lib_path) }?;

    library.setup_global_cl(&compile_flags(&args.common, transform_spec.path()));
    let session = Session::create(&library);

    let source = Source::open_file(&session, &args.common.input)?;
    let invocation = Invocation::create(&session);
    invocation.enable_console_diagnostics();
    invocation.parse_source(&source)?;
    // Only when something will read it: capturing the decision record costs
    // an extra full-module IR dump, and a plain compile has no use for one.
    let record = if needs_placement_pass(&args.common) {
        capture_decisions(&library, &invocation)?
    } else {
        decisions::DecisionRecord::default()
    };
    let layout =
        pin_unclaimed_dispatches(&library, &invocation, needs_placement_pass(&args.common))?;
    if needs_placement_pass(&args.common) {
        let placement = capture_placement(&library, &invocation, None)?;
        if args.common.no_offload {
            assert_nothing_offloaded(&placement)?;
        }
        deliver_audit(
            &args.common,
            &audit::PlacementAudit::new(record, layout, placement),
        )?;
    }
    invocation.set_compile_to_phase("end");
    invocation.run_pipeline(Pipeline::Std)?;

    let output = Output::open_file(&library, &args.output)?;
    invocation.output_vm_bytecode(&output)?;
    output.keep();

    println!("wrote {}", args.output.display());
    Ok(())
}

fn run_audit(args: &cli::AuditArgs) -> Result<(), Box<dyn Error>> {
    check_offload_flags(&args.common)?;
    let lib_path = resolve_lib_path(args.common.iree_compiler_lib.as_deref())?;
    let transform_spec = resolve_transform_spec(&args.common)?;
    check_spec_device_names(&args.common, transform_spec.path())?;
    let library = unsafe { Library::load(&lib_path) }?;

    library.setup_global_cl(&compile_flags(&args.common, transform_spec.path()));
    let session = Session::create(&library);

    let source = Source::open_file(&session, &args.common.input)?;
    let invocation = Invocation::create(&session);
    invocation.enable_console_diagnostics();
    invocation.parse_source(&source)?;
    // Read at the phase that wrote it, before anything downstream can drop
    // it; see DECISIONS_PHASE.
    let record = capture_decisions(&library, &invocation)?;
    // Same staging as `compile`, so the report describes the placement a
    // .vmfb from this input would actually get.
    let layout = pin_unclaimed_dispatches(&library, &invocation, true)?;
    let placement = capture_placement(&library, &invocation, args.emit_ir.as_deref())?;

    let audit = audit::PlacementAudit::new(record, layout, placement);
    print!("{audit}");
    deliver_audit(&args.common, &audit)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn common_args(rocket: &str, cpu: &str, triple: Option<&str>) -> cli::CommonArgs {
        cli::CommonArgs {
            iree_compiler_lib: None,
            input: PathBuf::from("model.mlir"),
            transform_spec: None,
            no_offload: false,
            elementwise: false,
            batch_matmul: false,
            strict_offload: false,
            strict_layout: false,
            report_json: None,
            rocket_device_name: rocket.to_string(),
            cpu_device_name: cpu.to_string(),
            llvmcpu_target_cpu: "generic".to_string(),
            llvmcpu_target_triple: triple.map(str::to_string),
        }
    }

    /// The baseline arm exists to keep everything on the CPU; asking it to
    /// also insist everything reached the NPU would fail every time.
    #[test]
    fn strict_offload_and_no_offload_are_rejected_together() {
        let mut args = common_args("rocket_device", "cpu_device", None);
        args.strict_offload = true;
        check_offload_flags(&args).expect("strict offload alone is fine");
        args.no_offload = true;
        let err = check_offload_flags(&args).expect_err("the pair must be rejected");
        assert!(err.to_string().contains("contradictory"), "{err}");
    }

    /// The extra `executable-targets` stop is only taken when something will
    /// read what it produces; a plain compile must not pay for it.
    #[test]
    fn a_plain_compile_takes_no_report_detour() {
        let mut args = common_args("rocket_device", "cpu_device", None);
        assert!(!needs_placement_pass(&args));
        args.report_json = Some(PathBuf::from("report.json"));
        assert!(needs_placement_pass(&args));
    }

    #[test]
    fn target_triple_flag_is_omitted_unless_requested() {
        let spec = PathBuf::from("spec.mlir");
        let flags = compile_flags(&common_args("rocket_device", "cpu_device", None), &spec);
        assert!(
            !flags
                .iter()
                .any(|f| f.starts_with("--iree-llvmcpu-target-triple")),
            "{flags:?}"
        );

        let flags = compile_flags(
            &common_args("rocket_device", "cpu_device", Some("aarch64-linux-gnu")),
            &spec,
        );
        assert!(
            flags.contains(&"--iree-llvmcpu-target-triple=aarch64-linux-gnu".to_string()),
            "{flags:?}"
        );
    }

    #[test]
    fn device_symbols_come_from_affinity_and_topology_attributes() {
        let spec = r#"
            %0 = transform.param.constant #hal.device.affinity<@rocket_device> -> !transform.any_param
            %1 = transform.param.constant #hal.device.topology<links = [
                (@rocket_device -> @cpu_device = {unified_memory = true}),
                (@cpu_device -> @rocket_device = {unified_memory = true})
              ]> -> !transform.any_param
        "#;
        let symbols = spec_device_symbols(spec);
        assert_eq!(
            symbols.into_iter().collect::<Vec<_>>(),
            vec!["cpu_device", "rocket_device"]
        );
    }

    #[test]
    fn topology_arrow_does_not_end_the_attribute_body() {
        // The `>` in `->` closes nothing; stopping there would hide every
        // device named after the first link's arrow.
        let spec = "#hal.device.topology<links = [(@a -> @b = {}), (@c -> @d = {})]>";
        let symbols = spec_device_symbols(spec);
        assert_eq!(
            symbols.into_iter().collect::<Vec<_>>(),
            vec!["a", "b", "c", "d"]
        );
    }

    #[test]
    fn symbols_outside_device_attributes_are_ignored() {
        let spec = r#"
            %r = flow.dispatch @rocket_executable::@entry::@rocket_conv2d(%x)
                {stream.affinity = #hal.device.affinity<@rocket_device>} : (tensor<1xf16>) -> tensor<1xf32>
        "#;
        let symbols = spec_device_symbols(spec);
        assert_eq!(
            symbols.into_iter().collect::<Vec<_>>(),
            vec!["rocket_device"]
        );
    }

    #[test]
    fn checked_in_spec_matches_the_default_device_names() {
        let spec = default_transform_spec_path();
        let args = common_args("rocket_device", "cpu_device", None);
        check_spec_device_names(&args, &spec).expect("defaults must match the shipped spec");
    }

    #[test]
    fn renaming_a_device_is_rejected_rather_than_left_to_segfault() {
        let spec = default_transform_spec_path();

        // `--cpu-device-name` is the dangerous one: the spec refers to
        // @cpu_device only from attributes, which are not symbol-verified.
        let err = check_spec_device_names(&common_args("rocket_device", "local", None), &spec)
            .expect_err("a renamed CPU device must be rejected");
        let message = err.to_string();
        assert!(message.contains("@cpu_device"), "{message}");
        assert!(!message.contains("@rocket_device,"), "{message}");

        let err = check_spec_device_names(&common_args("npu", "cpu_device", None), &spec)
            .expect_err("a renamed Rocket device must be rejected");
        assert!(err.to_string().contains("@rocket_device"), "{err}");
    }
}
