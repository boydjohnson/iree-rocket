//! The selection-time decision record, read back out of the IR.
//!
//! `rocket-plan-candidates` (the compiler plugin) asks rocket-core what it
//! would do with every convolution and matmul in a function and writes one
//! dictionary per candidate into a `rocket.plan_decisions` array attribute
//! on the function. That attribute is the report sink COMPILER_ROADMAP.md
//! section 3 asks for: it is written where every candidate is still visible,
//! before the match/rewrite loop erases the ones it claims, and it survives
//! that erasure because it lives on the function rather than on the ops.
//!
//! This module reads it back. The attribute is captured at the end of the
//! **preprocessing** phase -- the phase the pass runs in -- rather than
//! wherever the caller happens to dump IR next. It does in fact survive to
//! `executable-targets` today, but that is a property of which passes IREE
//! currently runs on a `util.func`'s discardable attributes, and
//! `report.rs` already documents one attribute that did not survive. Taking
//! the record at the phase that produced it is what makes the report
//! reliable rather than lucky.
//!
//! The parser here is a scanner for the shape MLIR prints this one
//! attribute in, not a general attribute parser: an array of dictionaries
//! whose values are strings, `i64`s, and one location. It tracks nesting
//! and string literals, which is the whole of what it takes to not be
//! fooled by a `,` inside a `detail` string or a `]` inside a path.

use std::{collections::BTreeMap, fmt};

/// What the planner decided for one candidate, and why.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Decision {
    /// The function the candidate was found in.
    pub function: String,
    /// `dense_conv2d`, `depthwise_conv2d` or `matmul`.
    pub kind: String,
    /// `nhwc` / `nchw` for convolutions, `row_major` for a matmul.
    pub layout: String,
    /// `226x226 Cin 3 Cout 32 k3x3 s2`, or `197x768 x 768x768`.
    pub shape: String,
    /// The rung the decision was reached on (`fp16`, `int8_requant`, ...),
    /// empty when the operand types never named one.
    pub precision: String,
    /// `direct`, `tiled`, `cpu` or `deferred`.
    pub decision: String,
    /// The planner or admission status name, or `form` / `dynamic` / `ok`.
    pub status: String,
    /// Which kind of limit the status is: see [`Decision::limit_explanation`].
    pub limit: String,
    /// The planner's own message, or the tile summary for an accepted plan.
    pub detail: String,
    /// Standalone hardware jobs an accepted plan runs; 0 when not planned.
    pub jobs: i64,
    /// Column tiles across the output row; 1 means no horizontal split.
    pub columns: i64,
    /// `path:line:col`, or `unknown`.
    pub location: String,
}

impl Decision {
    /// Whether the compiler expects this candidate on the NPU.
    pub fn accepted(&self) -> bool {
        self.decision == "direct" || self.decision == "tiled"
    }

    /// The one sentence that turns a status into something a reader can act
    /// on. Only the hardware class is permanent; the rest are ours to move.
    pub fn limit_explanation(&self) -> &'static str {
        match self.limit.as_str() {
            "none" => "",
            "shape" => "the operation is malformed on its own terms",
            "semantics" => "the Rocket lowering does not express this operation",
            "hardware" => "a fixed hardware bound; no plan under the current policy",
            "validation" => "validation policy: register-representable, not measured",
            "cost" => "cost policy: offloading this was measured to be slower",
            _ => "the planner failed internally; this is a bug report",
        }
    }
}

/// Every candidate decision recorded during a compile, in walk order.
#[derive(Clone, Debug, Default)]
pub struct DecisionRecord {
    pub decisions: Vec<Decision>,
}

impl DecisionRecord {
    /// Reads every `rocket.plan_decisions` attribute in `ir_text`.
    pub fn scan(ir_text: &str) -> Self {
        let aliases = location_aliases(ir_text);
        let mut decisions = Vec::new();
        for line in ir_text.lines() {
            let Some(marker) = line.find(DECISIONS_ATTR) else {
                continue;
            };
            let function = symbol_name(&line[..marker]).unwrap_or_default();
            let Some(array) = balanced(&line[marker + DECISIONS_ATTR.len()..], '[', ']') else {
                continue;
            };
            for entry in top_level_groups(array, '{', '}') {
                decisions.push(parse_decision(&function, entry, &aliases));
            }
        }
        DecisionRecord { decisions }
    }

    pub fn is_empty(&self) -> bool {
        self.decisions.is_empty()
    }

    pub fn accepted(&self) -> usize {
        self.decisions.iter().filter(|d| d.accepted()).count()
    }

    pub fn count_of(&self, decision: &str) -> usize {
        self.decisions
            .iter()
            .filter(|d| d.decision == decision)
            .count()
    }

    /// Standalone hardware jobs across the accepted candidates. Deliberately
    /// not a dispatch-site count: one dispatch already runs several jobs.
    pub fn hardware_jobs(&self) -> i64 {
        self.decisions.iter().map(|d| d.jobs).sum()
    }

    /// The candidates the compiler does not expect on the NPU, which is what
    /// a placement question is usually about.
    pub fn not_accepted(&self) -> impl Iterator<Item = &Decision> {
        self.decisions.iter().filter(|d| !d.accepted())
    }
}

const DECISIONS_ATTR: &str = "rocket.plan_decisions = ";

/// Collects `#loc12 = loc("f.mlir":3:4)` alias definitions. MLIR prints a
/// location inline the first time and as an alias when it is reused, so a
/// record whose ops share a location would otherwise read as `#loc7`.
fn location_aliases(ir_text: &str) -> BTreeMap<String, String> {
    let mut aliases = BTreeMap::new();
    for line in ir_text.lines() {
        let trimmed = line.trim_start();
        let Some(rest) = trimmed.strip_prefix('#') else {
            continue;
        };
        let Some((name, body)) = rest.split_once(" = ") else {
            continue;
        };
        if !name.starts_with("loc") || !body.starts_with("loc(") {
            continue;
        }
        aliases.insert(format!("#{name}"), format_location(body));
    }
    aliases
}

fn parse_decision(function: &str, entry: &str, aliases: &BTreeMap<String, String>) -> Decision {
    let mut decision = Decision {
        function: function.to_string(),
        ..Decision::default()
    };
    for field in top_level_split(entry, ',') {
        let Some((key, value)) = field.split_once(" = ") else {
            continue;
        };
        let (key, value) = (key.trim(), value.trim());
        match key {
            "kind" => decision.kind = unquote(value),
            "layout" => decision.layout = unquote(value),
            "shape" => decision.shape = unquote(value),
            "precision" => decision.precision = unquote(value),
            "decision" => decision.decision = unquote(value),
            "status" => decision.status = unquote(value),
            "limit" => decision.limit = unquote(value),
            "detail" => decision.detail = unquote(value),
            "jobs" => decision.jobs = leading_integer(value),
            "columns" => decision.columns = leading_integer(value),
            "loc" => {
                decision.location = if value.starts_with('#') {
                    aliases
                        .get(value)
                        .cloned()
                        .unwrap_or_else(|| value.to_string())
                } else {
                    format_location(value)
                }
            }
            _ => {}
        }
    }
    decision
}

/// `loc("model.mlir":42:7)` -> `model.mlir:42:7`. A fused or callsite
/// location keeps its first file/line/column, which is the one a reader
/// wants; anything else reads as `unknown`.
fn format_location(text: &str) -> String {
    let Some(open) = text.find('"') else {
        return "unknown".to_string();
    };
    let rest = &text[open + 1..];
    let Some(close) = rest.find('"') else {
        return "unknown".to_string();
    };
    let file = &rest[..close];
    let after = rest[close + 1..].trim_start();
    let Some(numbers) = after.strip_prefix(':') else {
        return file.to_string();
    };
    let digits: String = numbers
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == ':')
        .collect();
    if digits.is_empty() {
        return file.to_string();
    }
    format!("{file}:{digits}")
}

/// The first `@name` or `@"name"` in `text`, without the `@` or quotes.
///
/// The quoted spelling is not a corner case: IREE names a dispatch
/// executable after its source function, and an ONNX import's entry point is
/// commonly `torch-jit-export$async`, whose `-` forces MLIR to quote the
/// symbol.
fn symbol_name(text: &str) -> Option<String> {
    let start = text.find('@')? + 1;
    let rest = &text[start..];
    if let Some(quoted) = rest.strip_prefix('"') {
        let end = quoted.find('"')?;
        return Some(quoted[..end].to_string());
    }
    let end = rest
        .find(|c: char| !(c.is_alphanumeric() || c == '_' || c == '$' || c == '.'))
        .unwrap_or(rest.len());
    Some(rest[..end].to_string())
}

fn unquote(value: &str) -> String {
    let Some(rest) = value.strip_prefix('"') else {
        return value.to_string();
    };
    let mut out = String::new();
    let mut chars = rest.chars();
    while let Some(c) = chars.next() {
        match c {
            '"' => break,
            '\\' => match chars.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some(other) => out.push(other),
                None => break,
            },
            other => out.push(other),
        }
    }
    out
}

/// `6 : i64` -> 6.
fn leading_integer(value: &str) -> i64 {
    let digits: String = value
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '-')
        .collect();
    digits.parse().unwrap_or(0)
}

/// The text between `open` and its matching `close`, given `text` starts at
/// or before the `open`. Skips string literals so a bracket inside one does
/// not count.
fn balanced(text: &str, open: char, close: char) -> Option<&str> {
    let start = text.find(open)?;
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (i, c) in text[start..].char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        match c {
            '"' => in_string = true,
            c if c == open => depth += 1,
            c if c == close => {
                depth -= 1;
                if depth == 0 {
                    return Some(&text[start + open.len_utf8()..start + i]);
                }
            }
            _ => {}
        }
    }
    None
}

/// Every `open`..`close` group at the top nesting level of `text`.
fn top_level_groups(text: &str, open: char, close: char) -> Vec<&str> {
    let mut groups = Vec::new();
    let mut cursor = 0usize;
    while cursor < text.len() {
        let Some(offset) = text[cursor..].find(open) else {
            break;
        };
        let start = cursor + offset;
        let Some(group) = balanced(&text[start..], open, close) else {
            break;
        };
        // `group` excludes both delimiters; resume past the closing one.
        cursor = start + open.len_utf8() + group.len() + close.len_utf8();
        groups.push(group);
    }
    groups
}

/// Splits `text` on `separator`, ignoring separators inside brackets,
/// parentheses, braces or string literals.
fn top_level_split(text: &str, separator: char) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    let mut start = 0usize;
    for (i, c) in text.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        match c {
            '"' => in_string = true,
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth = depth.saturating_sub(1),
            c if c == separator && depth == 0 => {
                parts.push(&text[start..i]);
                start = i + c.len_utf8();
            }
            _ => {}
        }
    }
    parts.push(&text[start..]);
    parts
}

impl fmt::Display for Decision {
    /// The one-candidate form COMPILER_ROADMAP.md section 3 spells out:
    /// where it came from, where it went, and the decisive reason.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let precision = if self.precision.is_empty() {
            String::new()
        } else {
            format!(" [{}]", self.precision)
        };
        match self.decision.as_str() {
            "direct" => writeln!(
                f,
                "  {} at {} -> Rocket, direct{precision}",
                self.kind, self.location
            )?,
            "tiled" => writeln!(
                f,
                "  {} at {} -> Rocket, tiled into {} hardware job(s), {} column tile(s){precision}",
                self.kind, self.location, self.jobs, self.columns
            )?,
            "deferred" => writeln!(
                f,
                "  {} at {} -> deferred to runtime planning{precision}",
                self.kind, self.location
            )?,
            _ => writeln!(
                f,
                "  {} at {} -> CPU [{}]{precision}",
                self.kind, self.location, self.status
            )?,
        }
        writeln!(f, "      {} {}", self.layout, self.shape)?;
        let explanation = self.limit_explanation();
        if !explanation.is_empty() {
            writeln!(f, "      {explanation}")?;
        }
        if !self.detail.is_empty() {
            writeln!(f, "      {}", self.detail)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One line, exactly as the plugin's attribute prints on a `util.func`
    /// whose ONNX-imported name has to be quoted.
    const IR: &str = r#"
#loc3 = loc("mnv2.mlir":166:11)
  util.func public @"torch-jit-export$async"(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub, rocket.plan_decisions = [{columns = 1 : i64, decision = "tiled", detail = "cbuf 11/1, tiles 6, columns 1", jobs = 6 : i64, kind = "dense_conv2d", layout = "nhwc", limit = "none", loc = loc("mnv2.mlir":134:10), precision = "fp16", shape = "226x226 Cin 3 Cout 32 k3x3 s2", status = "ok"}, {columns = 0 : i64, decision = "cpu", detail = "input channels must be 1..=512, not 960", jobs = 0 : i64, kind = "depthwise_conv2d", layout = "nchw", limit = "validation", loc = #loc3, precision = "fp16", shape = "9x9 Cin 960 Cout 960 k3x3 s1", status = "unvalidated_configuration"}]} {
    util.return %arg0 : !hal.buffer_view
  }
"#;

    #[test]
    fn every_field_of_every_candidate_is_read() {
        let record = DecisionRecord::scan(IR);
        assert_eq!(record.decisions.len(), 2);
        let first = &record.decisions[0];
        assert_eq!(first.function, "torch-jit-export$async");
        assert_eq!(first.kind, "dense_conv2d");
        assert_eq!(first.layout, "nhwc");
        assert_eq!(first.shape, "226x226 Cin 3 Cout 32 k3x3 s2");
        assert_eq!(first.precision, "fp16");
        assert_eq!(first.decision, "tiled");
        assert_eq!(first.status, "ok");
        assert_eq!(first.limit, "none");
        assert_eq!(first.detail, "cbuf 11/1, tiles 6, columns 1");
        assert_eq!(first.jobs, 6);
        assert_eq!(first.columns, 1);
        assert_eq!(first.location, "mnv2.mlir:134:10");
        assert!(first.accepted());
    }

    /// The `detail` string carries commas and the shape summary carries
    /// `x`; splitting the dictionary naively on `,` would tear both apart.
    #[test]
    fn a_comma_inside_a_detail_string_does_not_end_the_field() {
        let record = DecisionRecord::scan(IR);
        assert_eq!(
            record.decisions[1].detail,
            "input channels must be 1..=512, not 960"
        );
        assert_eq!(record.decisions[1].shape, "9x9 Cin 960 Cout 960 k3x3 s1");
    }

    /// A location MLIR chose to print as an alias must still name a file.
    #[test]
    fn an_aliased_location_is_resolved() {
        let record = DecisionRecord::scan(IR);
        assert_eq!(record.decisions[1].location, "mnv2.mlir:166:11");
    }

    #[test]
    fn the_counts_separate_candidates_from_hardware_jobs() {
        let record = DecisionRecord::scan(IR);
        assert_eq!(record.decisions.len(), 2);
        assert_eq!(record.accepted(), 1);
        assert_eq!(record.count_of("cpu"), 1);
        // Six jobs from one accepted candidate: one dispatch, six jobs.
        assert_eq!(record.hardware_jobs(), 6);
        assert_eq!(record.not_accepted().count(), 1);
    }

    #[test]
    fn a_refusal_prints_its_class_and_the_planners_own_message() {
        let text = record_text(&DecisionRecord::scan(IR).decisions[1]);
        assert!(
            text.contains("depthwise_conv2d at mnv2.mlir:166:11 -> CPU"),
            "{text}"
        );
        assert!(text.contains("[unvalidated_configuration]"), "{text}");
        assert!(
            text.contains("validation policy: register-representable, not measured"),
            "{text}"
        );
        assert!(
            text.contains("input channels must be 1..=512, not 960"),
            "{text}"
        );
    }

    #[test]
    fn a_module_with_no_candidates_reads_as_empty() {
        assert!(DecisionRecord::scan("util.func public @f() { util.return }").is_empty());
    }

    fn record_text(decision: &Decision) -> String {
        decision.to_string()
    }
}
