mod bindings;
mod cli;
mod compiler;
mod report;

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

/// The phase the Rocket placement pin runs at. `flow` is the last point where
/// every dispatch in the program is still a `flow.dispatch` carrying a plain
/// `stream.affinity` attribute: dispatch regions have been formed and
/// outlined, and Stream's affinity analysis -- which is what pulls an
/// IREE-formed dispatch onto the NPU when its only consumer is a Rocket one
/// -- has not run yet.
const PIN_PHASE: &str = "flow";

/// Registered by the compiler plugin; see RocketPinUnclaimedDispatchesPass.cpp.
const PIN_PASS: &str = "rocket-pin-unclaimed-dispatches";

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
fn pin_unclaimed_dispatches(invocation: &Invocation) -> Result<(), Box<dyn Error>> {
    invocation.set_compile_to_phase(PIN_PHASE);
    invocation.run_pipeline(Pipeline::Std)?;
    invocation.run_pass_pipeline(PIN_PASS)?;
    invocation.set_compile_from_phase(PIN_PHASE);
    Ok(())
}

fn run_compile(args: &cli::CompileArgs) -> Result<(), Box<dyn Error>> {
    let lib_path = resolve_lib_path(args.common.iree_compiler_lib.as_deref())?;
    let transform_spec = transform_spec_path(&args.common);
    check_spec_device_names(&args.common, &transform_spec)?;
    let library = unsafe { Library::load(&lib_path) }?;

    library.setup_global_cl(&compile_flags(&args.common, &transform_spec));
    let session = Session::create(&library);

    let source = Source::open_file(&session, &args.common.input)?;
    let invocation = Invocation::create(&session);
    invocation.enable_console_diagnostics();
    invocation.parse_source(&source)?;
    pin_unclaimed_dispatches(&invocation)?;
    invocation.set_compile_to_phase("end");
    invocation.run_pipeline(Pipeline::Std)?;

    let output = Output::open_file(&library, &args.output)?;
    invocation.output_vm_bytecode(&output)?;
    output.keep();

    println!("wrote {}", args.output.display());
    Ok(())
}

fn run_audit(args: &cli::AuditArgs) -> Result<(), Box<dyn Error>> {
    let lib_path = resolve_lib_path(args.common.iree_compiler_lib.as_deref())?;
    let transform_spec = transform_spec_path(&args.common);
    check_spec_device_names(&args.common, &transform_spec)?;
    let library = unsafe { Library::load(&lib_path) }?;

    library.setup_global_cl(&compile_flags(&args.common, &transform_spec));
    let session = Session::create(&library);

    let source = Source::open_file(&session, &args.common.input)?;
    let invocation = Invocation::create(&session);
    invocation.enable_console_diagnostics();
    invocation.parse_source(&source)?;
    // Same staging as `compile`, so the report describes the placement a
    // .vmfb from this input would actually get.
    pin_unclaimed_dispatches(&invocation)?;
    invocation.set_compile_to_phase("executable-targets");
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

#[cfg(test)]
mod tests {
    use super::*;

    fn common_args(rocket: &str, cpu: &str, triple: Option<&str>) -> cli::CommonArgs {
        cli::CommonArgs {
            iree_compiler_lib: None,
            input: PathBuf::from("model.mlir"),
            transform_spec: None,
            rocket_device_name: rocket.to_string(),
            cpu_device_name: cpu.to_string(),
            llvmcpu_target_cpu: "generic".to_string(),
            llvmcpu_target_triple: triple.map(str::to_string),
        }
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
