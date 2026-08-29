use std::fmt;

/// A placement audit built by scanning the textual IR emitted after running
/// `rocket-annotate-final-placement`.
///
/// The original design assumed `rocket.origin`/`rocket.origin_kind`
/// (stamped by `rocket-annotate-original-placement` during preprocessing)
/// would survive on ops that stay on CPU all the way to the
/// `executable-targets` phase, giving a single-run "original vs final"
/// report. Empirically that's not true: something between preprocessing and
/// executable-targets (most likely dispatch-region formation) drops
/// unrecognized string attributes -- confirmed by direct inspection of the
/// annotated IR, not something specific to this FFI path (the subprocess
/// `iree-compile | iree-opt` pipeline shows the same absence). So this only
/// reports what `rocket.final` actually tells you: where each dispatch's
/// executable ended up.
#[derive(Default)]
pub struct PlacementReport {
    pub rocket_executables: Vec<String>,
    pub cpu_executables: Vec<String>,
}

impl PlacementReport {
    pub fn scan(ir_text: &str) -> Self {
        let mut report = PlacementReport::default();
        for line in ir_text.lines() {
            let trimmed = line.trim_start();
            // "hal.executable " (trailing space) is the per-executable line;
            // "hal.executable.variant"/"hal.executable.export" also carry a
            // redundant copy of the same tag and would double-count if matched.
            if !trimmed.starts_with("hal.executable ") {
                continue;
            }
            let Some(name) = extract_symbol_name(line) else {
                continue;
            };
            match extract_attr(line, "rocket.final").as_deref() {
                Some("rocket") => report.rocket_executables.push(name),
                Some("cpu") => report.cpu_executables.push(name),
                _ => {}
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
/// {...}` line. Stops at the first character that can't appear in a bare
/// (unquoted) MLIR symbol name.
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
        writeln!(
            f,
            "  {} executable(s) -> rocket",
            self.rocket_executables.len()
        )?;
        for name in &self.rocket_executables {
            writeln!(f, "    - {name}")?;
        }
        writeln!(
            f,
            "  {} executable(s) -> cpu",
            self.cpu_executables.len()
        )?;
        for name in &self.cpu_executables {
            writeln!(f, "    - {name}")?;
        }
        Ok(())
    }
}
