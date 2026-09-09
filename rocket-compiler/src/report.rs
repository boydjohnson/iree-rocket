use std::{collections::BTreeMap, fmt};

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
/// dispatches: the Rocket transform spec's hand-authored splice reuses a
/// handful of fixed executable names (`rocket_dynamic_executable`,
/// `rocket_dynamic_depthwise_executable`, their stride and int8 variants)
/// across every matched shape, so there is no comparable per-op naming
/// signal for Rocket-routed ops.
///
/// That is exactly why `dispatches` is counted separately. An executable
/// count alone badly misreports Rocket placement in both directions: the
/// whole int8 path is three executables serving twenty convolutions, while
/// IREE deduplicates structurally identical CPU dispatches so that a single
/// CPU executable can stand in for fifty call sites. Counting
/// `stream.cmd.dispatch` sites is what answers "how much of the model
/// actually ran on the NPU".
#[derive(Default)]
pub struct PlacementExecutable {
    pub name: String,
    pub exports: Vec<String>,
    pub dispatches: usize,
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
        // Dispatch sites are far below the executable declarations they name,
        // so they are tallied by symbol here and folded into the buckets once
        // the whole module has been read.
        let mut dispatch_counts: BTreeMap<String, usize> = BTreeMap::new();

        for line in ir_text.lines() {
            let trimmed = line.trim_start();
            if let Some(name) = dispatch_target(trimmed) {
                *dispatch_counts.entry(name).or_default() += 1;
            }
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
                    dispatches: 0,
                });
                current = Some(is_rocket);
            } else if trimmed.starts_with("hal.executable.export ")
                && let (Some(is_rocket), Some(name)) = (current, extract_symbol_name(line))
            {
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

        for exec in report
            .rocket_executables
            .iter_mut()
            .chain(report.cpu_executables.iter_mut())
        {
            exec.dispatches = dispatch_counts.get(&exec.name).copied().unwrap_or(0);
        }

        report
    }

    pub fn rocket_dispatches(&self) -> usize {
        self.rocket_executables.iter().map(|e| e.dispatches).sum()
    }

    pub fn cpu_dispatches(&self) -> usize {
        self.cpu_executables.iter().map(|e| e.dispatches).sum()
    }

    /// CPU dispatch sites whose export name says they are running a
    /// convolution or a matmul: work that was a Rocket candidate and is not
    /// on the NPU. This is the placement side of the reconciliation, and the
    /// only per-dispatch evidence a module carries -- an operation IREE
    /// fused into a larger element-wise dispatch leaves none, so a zero here
    /// means "none named", not "none exist".
    pub fn contraction_cpu_dispatches(&self) -> usize {
        self.cpu_executables
            .iter()
            .filter(|executable| executable.is_contraction())
            .map(|executable| executable.dispatches)
            .sum()
    }

    /// The contraction-shaped CPU executables, most dispatch sites first, for
    /// naming names in a strict-offload failure.
    pub fn contraction_cpu_executables(&self) -> Vec<&PlacementExecutable> {
        let mut executables: Vec<&PlacementExecutable> = self
            .cpu_executables
            .iter()
            .filter(|executable| executable.is_contraction())
            .collect();
        executables.sort_by(|a, b| b.dispatches.cmp(&a.dispatches).then(a.name.cmp(&b.name)));
        executables
    }
}

impl PlacementExecutable {
    /// The op kinds of this executable's exports, as IREE named them.
    pub fn kinds(&self) -> Vec<String> {
        self.exports.iter().filter_map(|e| export_kind(e)).collect()
    }

    /// Whether any export of this executable is a convolution or matmul.
    pub fn is_contraction(&self) -> bool {
        self.kinds().iter().any(|kind| is_contraction_kind(kind))
    }
}

/// Pulls the executable symbol out of a
/// `stream.cmd.dispatch @executable::@variant::@export(...)` line, i.e. the
/// first `@name` up to its `::`. The quotes come off a quoted symbol so the
/// name matches the one read off the executable's own declaration.
fn dispatch_target(trimmed: &str) -> Option<String> {
    const MARKER: &str = "stream.cmd.dispatch @";
    let start = trimmed.find(MARKER)? + MARKER.len();
    let rest = &trimmed[start..];
    let end = rest.find("::")?;
    Some(rest[..end].trim_matches('"').to_string())
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
///
/// Both spellings, because IREE names an executable after the function that
/// produced it and an entry point whose name is not a bare MLIR identifier
/// is printed quoted -- `@"torch-jit-export$async_dispatch_0"`, which every
/// ONNX import through torch-mlir produces. Reading only the bare spelling
/// stopped at the opening quote and yielded an empty name, so every CPU
/// executable in such a module was reported as `- (0 dispatch site(s))`
/// with no name and no dispatch count at all.
fn extract_symbol_name(line: &str) -> Option<String> {
    let start = line.find('@')? + 1;
    let rest = &line[start..];
    if let Some(quoted) = rest.strip_prefix('"') {
        let end = quoted.find('"')?;
        return Some(quoted[..end].to_string());
    }
    let end = rest
        .find(|c: char| !(c.is_alphanumeric() || c == '_' || c == '$' || c == '.'))
        .unwrap_or(rest.len());
    Some(rest[..end].to_string())
}

/// The op kind IREE encoded in a dispatch export name.
///
/// An export is named `<function>_dispatch_<ordinal>_<kind>_<extents>_<dtypes>`
/// -- `main$async_dispatch_3_conv_56x56x144x3x3_f16xf16xf32`,
/// `..._matmul_like_196x64x192_f16xf16xf32`, `..._elementwise_401408_f16`,
/// `..._slow_memcpy`. The kind is what survives reliably; the extents are
/// the dispatch's loop ranges, not the candidate's logical shape (a
/// convolution's are its *output* extents plus the kernel, and its input
/// padding is already folded away), so they are not a key an original op
/// can be matched on and this deliberately does not try.
///
/// Returns the kind with its extents and dtypes stripped, or `None` when the
/// name does not have the `_dispatch_<n>_` infix -- which is every
/// hand-authored Rocket export.
fn export_kind(export: &str) -> Option<String> {
    const MARKER: &str = "_dispatch_";
    let tail = &export[export.rfind(MARKER)? + MARKER.len()..];
    let after_ordinal = tail.strip_prefix(|c: char| c.is_ascii_digit())?;
    let after_ordinal = after_ordinal.trim_start_matches(|c: char| c.is_ascii_digit());
    let kind = after_ordinal.strip_prefix('_')?;
    // Everything up to the first token that starts with a digit: that is the
    // extents group, and `matmul_like` shows why the kind is not just the
    // first token.
    let words: Vec<&str> = kind
        .split('_')
        .take_while(|word| !word.starts_with(|c: char| c.is_ascii_digit()))
        .collect();
    if words.is_empty() {
        return None;
    }
    Some(words.join("_"))
}

/// Whether a CPU dispatch is running a convolution or a matmul, i.e. work a
/// Rocket candidate would have been. Names the classes IREE gives
/// contraction dispatches; anything else (element-wise, transpose,
/// reduction, pooling, a memcpy) is not a candidate this report tracks.
fn is_contraction_kind(kind: &str) -> bool {
    matches!(
        kind,
        "conv" | "conv_2d" | "depthwise_conv" | "matmul" | "matmul_like" | "batch_matmul"
    )
}

impl fmt::Display for PlacementReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Placement report:")?;
        write_bucket(
            f,
            "rocket",
            &self.rocket_executables,
            self.rocket_dispatches(),
        )?;
        write_bucket(f, "cpu", &self.cpu_executables, self.cpu_dispatches())?;
        Ok(())
    }
}

fn write_bucket(
    f: &mut fmt::Formatter<'_>,
    label: &str,
    executables: &[PlacementExecutable],
    dispatches: usize,
) -> fmt::Result {
    writeln!(
        f,
        "  {} executable(s), {dispatches} dispatch site(s) -> {label}",
        executables.len()
    )?;
    for exec in executables {
        writeln!(
            f,
            "    - {} ({} dispatch site(s))",
            exec.name, exec.dispatches
        )?;
        for export in &exec.exports {
            writeln!(f, "        {export}")?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Shaped like the `executable-targets` IR after
    /// `rocket-annotate-final-placement`: executables carry the placement
    /// tag, and their dispatch sites appear much further down, inside
    /// `stream.cmd.execute` regions.
    const IR: &str = r#"
  hal.executable private @rocket_dynamic_int8_executable attributes {rocket.final = "rocket"} {
    hal.executable.variant public @rocket_dynamic_conv2d_v1 target(<"rocket", "rocket-flatbuffer-v1", {precision = "int8_accumulator"}>) {
      hal.executable.export public @rocket_dynamic_conv2d ordinal(0) layout(#layout) {
      } attributes {rocket.final = "rocket"}
    }
  }
  hal.executable private @rocket_dynamic_depthwise_int8_executable attributes {rocket.final = "rocket"} {
    hal.executable.variant public @rocket_dynamic_depthwise_conv2d_v1 target(<"rocket", "rocket-flatbuffer-v1", {precision = "int8_accumulator"}>) {
      hal.executable.export public @rocket_dynamic_depthwise_conv2d ordinal(0) layout(#layout) {
      } attributes {rocket.final = "rocket"}
    }
  }
  hal.executable private @main_graph$async_dispatch_7 attributes {rocket.final = "cpu"} {
    hal.executable.variant public @embedded_elf_x86_64 target(<"llvm-cpu", "embedded-elf-x86_64">) {
      hal.executable.export public @main_graph$async_dispatch_7_elementwise ordinal(0) layout(#layout) {
      }
    }
  }
  util.func public @main_graph$async() {
    %0 = stream.cmd.execute on(#hal.device.affinity<@rocket_device>) await(%t) => with(%a as %arg0: !stream.resource<transient>{%c1}) {
      stream.cmd.dispatch @rocket_dynamic_int8_executable::@rocket_dynamic_conv2d_v1::@rocket_dynamic_conv2d(%c1_i32 : i32) {
        wo %arg0[%c0 for %c1] : !stream.resource<transient>{%c1}
      }
    } => !stream.timepoint
    %1 = stream.cmd.execute on(#hal.device.affinity<@rocket_device>) await(%0) => with(%a as %arg0: !stream.resource<transient>{%c1}) {
      stream.cmd.dispatch @rocket_dynamic_int8_executable::@rocket_dynamic_conv2d_v1::@rocket_dynamic_conv2d(%c1_i32 : i32) {
        wo %arg0[%c0 for %c1] : !stream.resource<transient>{%c1}
      }
    } => !stream.timepoint
    %2 = stream.cmd.execute on(#hal.device.affinity<@rocket_device>) await(%1) => with(%a as %arg0: !stream.resource<transient>{%c1}) {
      stream.cmd.dispatch @rocket_dynamic_depthwise_int8_executable::@rocket_dynamic_depthwise_conv2d_v1::@rocket_dynamic_depthwise_conv2d(%c1_i32 : i32) {
        wo %arg0[%c0 for %c1] : !stream.resource<transient>{%c1}
      }
    } => !stream.timepoint
    %3 = stream.cmd.execute on(#hal.device.affinity<@cpu_device>) await(%2) => with(%a as %arg0: !stream.resource<transient>{%c1}) {
      stream.cmd.dispatch @main_graph$async_dispatch_7::@embedded_elf_x86_64::@main_graph$async_dispatch_7_elementwise(%c1_i32 : i32) {
        wo %arg0[%c0 for %c1] : !stream.resource<transient>{%c1}
      }
    } => !stream.timepoint
    util.return
  }
"#;

    #[test]
    fn executables_are_bucketed_by_their_final_placement_tag() {
        let report = PlacementReport::scan(IR);
        let rocket: Vec<&str> = report
            .rocket_executables
            .iter()
            .map(|e| e.name.as_str())
            .collect();
        assert_eq!(
            rocket,
            vec![
                "rocket_dynamic_int8_executable",
                "rocket_dynamic_depthwise_int8_executable"
            ]
        );
        assert_eq!(report.cpu_executables.len(), 1);
        assert_eq!(
            report.cpu_executables[0].name,
            "main_graph$async_dispatch_7"
        );
        assert_eq!(
            report.rocket_executables[0].exports,
            vec!["rocket_dynamic_conv2d"]
        );
    }

    /// The reason dispatch sites are counted at all: the int8 path routes
    /// many convolutions through very few executables, so an executable
    /// count alone reads as "nothing was offloaded".
    #[test]
    fn dispatch_sites_are_counted_per_executable() {
        let report = PlacementReport::scan(IR);
        assert_eq!(report.rocket_executables[0].dispatches, 2);
        assert_eq!(report.rocket_executables[1].dispatches, 1);
        assert_eq!(report.rocket_dispatches(), 3);
        assert_eq!(report.cpu_dispatches(), 1);
        assert_eq!(report.rocket_executables.len(), 2);
    }

    /// Every ONNX import through torch-mlir names its entry point
    /// `torch-jit-export$async`, whose `-` forces MLIR to quote every
    /// executable symbol derived from it. Reading only bare symbols reported
    /// the whole CPU side of such a module as unnamed executables with zero
    /// dispatch sites.
    const QUOTED_IR: &str = r#"
  hal.executable private @"torch-jit-export$async_dispatch_3" attributes {rocket.final = "cpu"} {
    hal.executable.variant public @embedded_elf_arm_64 target(<"llvm-cpu", "embedded-elf-arm_64">) {
      hal.executable.export public @"torch-jit-export$async_dispatch_3_conv_56x56x144x3x3_f16xf16xf32" ordinal(0) layout(#layout) {
      }
    }
  }
  hal.executable private @"torch-jit-export$async_dispatch_4" attributes {rocket.final = "cpu"} {
    hal.executable.variant public @embedded_elf_arm_64 target(<"llvm-cpu", "embedded-elf-arm_64">) {
      hal.executable.export public @"torch-jit-export$async_dispatch_4_elementwise_401408_f16" ordinal(0) layout(#layout) {
      }
    }
  }
  util.func public @"torch-jit-export$async"() {
    %0 = stream.cmd.execute with() {
      stream.cmd.dispatch @"torch-jit-export$async_dispatch_3"::@embedded_elf_arm_64::@"torch-jit-export$async_dispatch_3_conv_56x56x144x3x3_f16xf16xf32"() {
      }
    } => !stream.timepoint
    %1 = stream.cmd.execute with() {
      stream.cmd.dispatch @"torch-jit-export$async_dispatch_4"::@embedded_elf_arm_64::@"torch-jit-export$async_dispatch_4_elementwise_401408_f16"() {
      }
    } => !stream.timepoint
    util.return
  }
"#;

    #[test]
    fn a_quoted_executable_symbol_is_named_and_counted() {
        let report = PlacementReport::scan(QUOTED_IR);
        let names: Vec<&str> = report
            .cpu_executables
            .iter()
            .map(|e| e.name.as_str())
            .collect();
        assert_eq!(
            names,
            vec![
                "torch-jit-export$async_dispatch_3",
                "torch-jit-export$async_dispatch_4"
            ]
        );
        assert_eq!(report.cpu_dispatches(), 2);
    }

    /// The reconciliation signal: a convolution left on the CPU says so in
    /// its export name, an element-wise dispatch is not a candidate.
    #[test]
    fn contraction_dispatches_are_told_apart_from_the_rest() {
        let report = PlacementReport::scan(QUOTED_IR);
        assert_eq!(report.contraction_cpu_dispatches(), 1);
        let named: Vec<&str> = report
            .contraction_cpu_executables()
            .iter()
            .map(|e| e.name.as_str())
            .collect();
        assert_eq!(named, vec!["torch-jit-export$async_dispatch_3"]);
    }

    #[test]
    fn export_kinds_stop_before_the_extents() {
        assert_eq!(
            export_kind("main$async_dispatch_3_conv_56x56x144x3x3_f16xf16xf32").as_deref(),
            Some("conv")
        );
        // `matmul_like` is why the kind is not just the first token.
        assert_eq!(
            export_kind("main$async_dispatch_12_matmul_like_196x64x192_f16xf16xf32").as_deref(),
            Some("matmul_like")
        );
        assert_eq!(
            export_kind("main$async_dispatch_7_elementwise_401408_f16").as_deref(),
            Some("elementwise")
        );
        // No extents at all.
        assert_eq!(
            export_kind("main$async_dispatch_2_slow_memcpy").as_deref(),
            Some("slow_memcpy")
        );
        // A hand-authored Rocket export has no `_dispatch_<n>_` infix.
        assert_eq!(export_kind("rocket_dynamic_conv2d"), None);
        assert!(is_contraction_kind("conv"));
        assert!(is_contraction_kind("matmul_like"));
        assert!(!is_contraction_kind("elementwise"));
    }

    #[test]
    fn dispatch_target_reads_only_the_executable_symbol() {
        assert_eq!(
            dispatch_target("stream.cmd.dispatch @exe::@variant::@export(%c1_i32 : i32) {")
                .as_deref(),
            Some("exe")
        );
        // A `$` is part of an IREE-generated executable symbol.
        assert_eq!(
            dispatch_target("stream.cmd.dispatch @main_graph$async_dispatch_7::@v::@e(").as_deref(),
            Some("main_graph$async_dispatch_7")
        );
        assert_eq!(dispatch_target("stream.cmd.fill %c0_i8, %arg4"), None);
        // A quoted symbol must come back as the bare name the executable's
        // own declaration carries, or the two never join up.
        assert_eq!(
            dispatch_target("stream.cmd.dispatch @\"torch-jit-export$async_dispatch_3\"::@v::@e(")
                .as_deref(),
            Some("torch-jit-export$async_dispatch_3")
        );
    }

    #[test]
    fn the_report_surfaces_both_counts() {
        let text = PlacementReport::scan(IR).to_string();
        assert!(
            text.contains("2 executable(s), 3 dispatch site(s) -> rocket"),
            "{text}"
        );
        assert!(
            text.contains("1 executable(s), 1 dispatch site(s) -> cpu"),
            "{text}"
        );
        assert!(
            text.contains("rocket_dynamic_int8_executable (2 dispatch site(s))"),
            "{text}"
        );
    }
}
