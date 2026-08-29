use std::fmt;

/// A placement audit built by scanning the textual IR emitted after running
/// `rocket-annotate-final-placement`.
///
/// The original design assumed `rocket.origin`/`rocket.origin_kind`
/// (stamped by `rocket-annotate-original-placement` during preprocessing)
/// would survive on ops that stay on CPU all the way to the
/// `executable-targets` phase, giving a single-run "original vs final"
/// report. Empirically that's not true: something between preprocessing and
/// executable-targets (specifically `FoldUnitExtentDimsPass`, traced
/// pass-by-pass) drops unrecognized string attributes -- confirmed by direct
/// inspection of the annotated IR, not something specific to this FFI path
/// (the subprocess `iree-compile | iree-opt` pipeline shows the same
/// absence). So per-op provenance instead comes from IREE's own
/// dispatch-region naming convention, which survives for free: each
/// `hal.executable.export` name already encodes the source function, a
/// per-function ordinal, op kind, shape, and dtypes (e.g.
/// `unmatched_5x5_conv_dispatch_0_conv_DxDx16x5x5x32_f16xf16xf32`) -- surfaced
/// here rather than re-derived. Note this only exists for CPU-routed
/// dispatches: the Rocket transform spec's hand-authored splice reuses one
/// of just two executable names (`rocket_dynamic_executable` /
/// `rocket_dynamic_depthwise_executable`) across every matched shape, so
/// there is currently no comparable per-op naming signal for Rocket-routed
/// ops.
#[derive(Default)]
pub struct PlacementExecutable {
    pub name: String,
    pub exports: Vec<String>,
}

#[derive(Default)]
pub struct PlacementReport {
    pub rocket_executables: Vec<PlacementExecutable>,
    pub cpu_executables: Vec<PlacementExecutable>,
}

impl PlacementReport {
    pub fn scan(ir_text: &str) -> Self {
        let mut report = PlacementReport::default();
        // Index into whichever bucket the executable we're currently inside
        // of belongs to, so nested `hal.executable.export` lines can be
        // attributed correctly. Always the last element pushed to that
        // bucket, since exports are only ever encountered nested inside
        // their own executable's still-open block.
        let mut current: Option<bool> = None; // Some(true) = rocket, Some(false) = cpu

        for line in ir_text.lines() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("hal.executable ") {
                current = None;
                let Some(name) = extract_symbol_name(line) else {
                    continue;
                };
                let is_rocket = match extract_attr(line, "rocket.final").as_deref() {
                    Some("rocket") => true,
                    Some("cpu") => false,
                    _ => continue,
                };
                let bucket = if is_rocket {
                    &mut report.rocket_executables
                } else {
                    &mut report.cpu_executables
                };
                bucket.push(PlacementExecutable {
                    name,
                    exports: Vec::new(),
                });
                current = Some(is_rocket);
            } else if trimmed.starts_with("hal.executable.export ") {
                if let (Some(is_rocket), Some(name)) = (current, extract_symbol_name(line)) {
                    let bucket = if is_rocket {
                        &mut report.rocket_executables
                    } else {
                        &mut report.cpu_executables
                    };
                    if let Some(exec) = bucket.last_mut() {
                        exec.exports.push(name);
                    }
                }
            }
        }

        report
    }
}

/// Pulls `key = "value"` out of a printed MLIR attribute dict line. Good
/// enough for the stable, single-line-per-op format this pass prints; not a
/// general MLIR attribute parser.
fn extract_attr(line: &str, key: &str) -> Option<String> {
    let needle = format!("{key} = \"");
    let start = line.find(&needle)? + needle.len();
    let end = line[start..].find('"')? + start;
    Some(line[start..end].to_string())
}

/// Pulls the `@symbol_name` off a `hal.executable private @name attributes
/// {...}` or `hal.executable.export public @name ordinal(...) ...` line.
/// Stops at the first character that can't appear in a bare (unquoted) MLIR
/// symbol name.
fn extract_symbol_name(line: &str) -> Option<String> {
    let start = line.find('@')? + 1;
    let end = line[start..]
        .find(|c: char| !(c.is_alphanumeric() || c == '_' || c == '$' || c == '.'))
        .map(|i| start + i)
        .unwrap_or(line.len());
    Some(line[start..end].to_string())
}

impl fmt::Display for PlacementReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Placement report:")?;
        write_bucket(f, "rocket", &self.rocket_executables)?;
        write_bucket(f, "cpu", &self.cpu_executables)?;
        Ok(())
    }
}

fn write_bucket(
    f: &mut fmt::Formatter<'_>,
    label: &str,
    executables: &[PlacementExecutable],
) -> fmt::Result {
    writeln!(f, "  {} executable(s) -> {label}", executables.len())?;
    for exec in executables {
        writeln!(f, "    - {}", exec.name)?;
        for export in &exec.exports {
            writeln!(f, "        {export}")?;
        }
    }
    Ok(())
}
