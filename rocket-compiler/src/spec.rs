//! Rewrites the Rocket transform spec into one whose matchers cannot fire.
//!
//! This exists for measurement, not for compilation. ISSUES.md M4 found that
//! every NPU-vs-CPU number this repo had ever quoted was measured against a
//! CPU-only module built with plain `iree-compile`, which is 2.8x slower than
//! it needs to be: `@__transform_main` runs
//! `iree-preprocessing-convert-conv-to-channels-last` and
//! `linalg-specialize-generic-ops` *before* the match loop, so a model
//! compiled through this pipeline is NHWC whether or not anything offloads,
//! and IREE's CPU backend is 2.8x slower on NCHW MobileNetV2. A baseline that
//! never saw the spec is therefore not a baseline for the offload -- it is a
//! measurement of the conv layout.
//!
//! The fix is to build the CPU arm with the *same* pipeline and only the
//! match loop defeated, so the passes around the loop, the device topology
//! and the placement pin are exactly what the offload arm sees.
//!
//! Every matcher in the loop carries one of two things this module can
//! defeat, and it is checked rather than assumed -- a matcher that carried
//! neither would still fire and the "CPU-only" baseline would quietly
//! offload part of the model:
//!
//! - `transform.rocket.match.admitted`, on every convolution and matmul
//!   matcher since the shape ceilings moved to `rocket-core`'s admission
//!   table (COMPILER_ROADMAP.md section 2). Adding `no_offload` to it makes
//!   the op decline unconditionally.
//! - `transform.iree.match.dim_bounds`, still on the pooling and
//!   element-wise matchers, which have no planner to ask. Rewriting every
//!   bound to `umin = umax = 999999` -- larger than any dimension a real
//!   model has -- makes those decline.

use std::{collections::BTreeSet, error::Error};

/// Prefix on the `foreach_match` entries that ROADMAP Phase 1's element-wise
/// matchers are commented out with in the shipped spec.
///
/// They are disabled in the file itself, not merely absent from a default
/// built here, because ISSUES.md P8 measured that at the current
/// per-dispatch cost more offload sites make a model slower: an element-wise
/// op does less arithmetic than its own dispatch tax. Anything that reads the
/// spec without going through this module -- a bare `iree-compile
/// --iree-preprocessing-transform-spec-filename=...` included -- therefore
/// gets the conservative list.
const ELEMENTWISE_MARKER: &str = "//@ROCKET_ELEMENTWISE@";

/// The marker on the batch-matmul unbatching pass, and the handle rename
/// that has to accompany uncommenting it.
const BATCH_MATMUL_MARKER: &str = "//@ROCKET_BATCH_MATMUL@";
const BATCH_MATMUL_CONSUMER: &str = r#""rocket-record-conv-attrs" to %gemv_funcs"#;
const BATCH_MATMUL_CONSUMER_ENABLED: &str = r#""rocket-record-conv-attrs" to %unbatched_funcs"#;

/// Result of enabling the batch-matmul unbatching pass.
#[derive(Debug)]
pub struct BatchMatmulSpec {
    pub text: String,
    /// How many marked lines were uncommented.
    pub enabled: usize,
}

/// Returns `spec` with `rocket-unbatch-matmul` spliced into the pass chain.
///
/// Two edits, not one, and both are checked. Uncommenting the marked lines
/// defines a new handle; the pass that consumed the old one has to be
/// repointed at it or the new pass runs on nothing and its result is
/// discarded -- which would look exactly like "the flag works and no
/// batch_matmul matched". The transform dialect would not complain: an
/// unused handle is legal, so this cannot be left to the compiler to catch.
///
/// Fails if either edit finds nothing, for the reason
/// [`enable_elementwise`] fails the same way: a flag that silently compiles
/// the default pipeline is worse than one that errors.
pub fn enable_batch_matmul(spec: &str) -> Result<BatchMatmulSpec, Box<dyn Error>> {
    let mut enabled = 0;
    let text = spec
        .lines()
        .map(
            |line| match line.trim_start().strip_prefix(BATCH_MATMUL_MARKER) {
                Some(rest) => {
                    enabled += 1;
                    rest.to_string()
                }
                None => line.to_string(),
            },
        )
        .collect::<Vec<_>>()
        .join("\n");

    if enabled == 0 {
        return Err(format!(
            "--batch-matmul found no `{BATCH_MATMUL_MARKER}` lines in the transform spec, so \
             it would compile exactly the default pass chain under a flag that claims otherwise"
        )
        .into());
    }

    let consumers = text.matches(BATCH_MATMUL_CONSUMER).count();
    if consumers != 1 {
        return Err(format!(
            "--batch-matmul expected exactly one `{BATCH_MATMUL_CONSUMER}` to repoint at the \
             unbatched handle, found {consumers}. Uncommenting the pass without repointing its \
             consumer leaves it running on a handle nobody reads, which the transform dialect \
             accepts silently."
        )
        .into());
    }
    let text = text.replace(BATCH_MATMUL_CONSUMER, BATCH_MATMUL_CONSUMER_ENABLED);

    Ok(BatchMatmulSpec {
        text: if spec.ends_with('\n') {
            format!("{text}\n")
        } else {
            text
        },
        enabled,
    })
}

/// Result of enabling the element-wise matchers, kept together so the caller
/// can report what it did rather than trusting it silently -- the same
/// reasoning as [`NeutralizedSpec`].
#[derive(Debug)]
pub struct ElementwiseSpec {
    pub text: String,
    /// How many marked `foreach_match` entries were uncommented.
    pub enabled: usize,
}

/// Returns `spec` with the marked element-wise `foreach_match` entries
/// uncommented, so they join the match loop.
///
/// Fails if the spec carries no marked entries at all. That would mean the
/// marker was renamed or the entries were deleted, and silently compiling a
/// spec with no element-wise matchers under `--elementwise` would look like
/// "the flag works and nothing matched" -- indistinguishable from a real
/// measurement of zero sites, which is exactly the confusion this check
/// exists to prevent.
pub fn enable_elementwise(spec: &str) -> Result<ElementwiseSpec, Box<dyn Error>> {
    let mut enabled = 0;
    let text = spec
        .lines()
        // `trim_start` so the marker does not have to sit at column 0. It
        // reads better indented with the entries it belongs to, and a rule
        // that depends on a file's leading whitespace is a rule that breaks
        // the first time someone reformats the spec.
        .map(
            |line| match line.trim_start().strip_prefix(ELEMENTWISE_MARKER) {
                Some(rest) => {
                    enabled += 1;
                    rest.to_string()
                }
                None => line.to_string(),
            },
        )
        .collect::<Vec<_>>()
        .join("\n");

    if enabled == 0 {
        return Err(format!(
            "--elementwise found no `{ELEMENTWISE_MARKER}` entries in the transform spec, so \
             it would compile exactly the default match loop under a flag that claims \
             otherwise"
        )
        .into());
    }

    Ok(ElementwiseSpec {
        text: if spec.ends_with('\n') {
            format!("{text}\n")
        } else {
            text
        },
        enabled,
    })
}

/// The sentinel every `dim_bounds` bound is rewritten to. Both ends are set,
/// so the interval is a single value no real dimension can take.
const NO_OFFLOAD_BOUND: &str = "999999";

const DIM_BOUNDS_OP: &str = "transform.iree.match.dim_bounds";

/// The plugin's own admission matcher, and the unit attribute that makes it
/// decline every operation. Both spellings are part of the contract with
/// `RocketTransformOps.td`; a rename there has to land here.
const ADMITTED_OP: &str = "transform.rocket.match.admitted";
const NO_OFFLOAD_ATTR: &str = "no_offload";

/// Result of neutering a spec, kept together so the caller can report what it
/// did rather than trusting it silently.
#[derive(Debug)]
pub struct NeutralizedSpec {
    pub text: String,
    /// How many matcher lines were rewritten: one per `dim_bounds` bounds
    /// pair and one per `transform.rocket.match.admitted`.
    pub rewritten: usize,
    /// The matcher names taken from the `foreach_match` list.
    pub matchers: usize,
}

/// Returns `spec` with every matcher in the `foreach_match` lists defeated.
///
/// Fails if any matcher in the lists carries neither an admission check nor
/// a `dim_bounds`: that matcher would still fire, and the caller would get a
/// "CPU-only" baseline that quietly offloads part of the model -- exactly
/// the class of error this whole path exists to prevent. It is checked
/// rather than assumed because the spec grows matchers over time and
/// nothing else would notice.
pub fn neutralize(spec: &str) -> Result<NeutralizedSpec, Box<dyn Error>> {
    let matchers = foreach_match_matchers(spec);
    if matchers.is_empty() {
        return Err(
            "found no matchers in the transform spec: its `transform.foreach_match` \
                    list could not be read, so a no-offload spec cannot be derived from it"
                .into(),
        );
    }

    let unconstrained: Vec<&str> = matchers
        .iter()
        .copied()
        .filter(|name| !sequence_is_defeatable(spec, name))
        .collect();
    if !unconstrained.is_empty() {
        return Err(format!(
            "cannot build a no-offload spec: matcher(s) {} carry neither `{ADMITTED_OP}` nor \
             `{DIM_BOUNDS_OP}`, so nothing in them can be rewritten to make them decline. The \
             baseline would silently offload. Give each one an admission check (every \
             convolution and matmul matcher has one) or a dim_bounds, or defeat it another \
             way.",
            unconstrained
                .iter()
                .map(|name| format!("@{name}"))
                .collect::<Vec<_>>()
                .join(", "),
        )
        .into());
    }

    let mut text = String::with_capacity(spec.len());
    let mut rewritten = 0usize;
    for line in spec.split_inclusive('\n') {
        // Prose mentions both op names -- the file's own header explains
        // where the ceilings went -- and a comment is not a predicate.
        if line.trim_start().starts_with("//") {
            text.push_str(line);
            continue;
        }
        if line.contains(ADMITTED_OP) {
            let rewrite = disable_admission(line).ok_or_else(|| {
                format!(
                    "unrecognised `{ADMITTED_OP}` spelling, expected an optional \
                     `{{...}}` attribute dictionary before the `:`: {}",
                    line.trim()
                )
            })?;
            rewritten += 1;
            text.push_str(&rewrite);
            continue;
        }
        if !line.contains(DIM_BOUNDS_OP) {
            text.push_str(line);
            continue;
        }
        let mut rewrite = line.to_string();
        let min = replace_bound(&mut rewrite, "umin");
        let max = replace_bound(&mut rewrite, "umax");
        if min && max {
            rewritten += 1;
        } else {
            return Err(format!(
                "unrecognised `{DIM_BOUNDS_OP}` spelling, expected `umin = N, umax = M`: {}",
                line.trim()
            )
            .into());
        }
        text.push_str(&rewrite);
    }

    if rewritten == 0 {
        return Err(format!(
            "transform spec contains neither `{ADMITTED_OP}` nor `{DIM_BOUNDS_OP}` to rewrite"
        )
        .into());
    }

    Ok(NeutralizedSpec {
        text,
        rewritten,
        matchers: matchers.len(),
    })
}

/// Adds the `no_offload` unit attribute to one `transform.rocket.match.admitted`
/// line, keeping any attribute it already carries.
///
/// The op's assembly is `admitted $handle attr-dict `:` type`, so the
/// dictionary is optional and sits between the handle and the colon.
fn disable_admission(line: &str) -> Option<String> {
    if line.contains(NO_OFFLOAD_ATTR) {
        return Some(line.to_string());
    }
    let at = line.find(ADMITTED_OP)?;
    let colon = line[at..].find(" : ").map(|i| at + i)?;
    let middle = &line[at + ADMITTED_OP.len()..colon];
    let replacement = match middle.trim_end().strip_suffix('}') {
        // Already has a dictionary: prepend into it.
        Some(head) => {
            let open = head.rfind('{')?;
            format!(
                "{}{{{NO_OFFLOAD_ATTR}, {}}}",
                &head[..open],
                head[open + 1..].trim()
            )
        }
        None => format!("{} {{{NO_OFFLOAD_ATTR}}}", middle.trim_end()),
    };
    Some(format!(
        "{}{ADMITTED_OP}{replacement}{}",
        &line[..at],
        &line[colon..]
    ))
}

/// Replaces the integer after `<key> = ` with the sentinel, in place.
fn replace_bound(line: &mut String, key: &str) -> bool {
    let Some(at) = line.find(key) else {
        return false;
    };
    let after = at + key.len();
    let rest = &line[after..];
    // Tolerate `umin = 1` and `umin=1` alike; the spec uses the former.
    let digits_at = after + rest.len() - rest.trim_start_matches([' ', '=']).len();
    let end = digits_at
        + line[digits_at..]
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(line.len() - digits_at);
    if end == digits_at {
        return false;
    }
    line.replace_range(digits_at..end, NO_OFFLOAD_BOUND);
    true
}

/// The matcher half of every `@matcher -> @action` pair in **every**
/// `transform.foreach_match` list in the spec.
///
/// Read from the lists rather than from the `@match_*` naming convention: the
/// spec defines matchers that no list uses (the s3/s4 dense ones), and a
/// matcher that is never invoked cannot offload anything, so demanding bounds
/// of it would fail the build for no reason.
///
/// All of them, not just the first: the spec has run more than one match loop
/// since the requantized int8 path landed, and a matcher in a later loop
/// offloads exactly as much as one in the first. Reading only the first list
/// left the rest unchecked, so a `--no-offload` baseline built from this spec
/// would have quietly offloaded them -- the precise failure this module
/// exists to prevent, and it went unnoticed because the shipped-spec test
/// asserted a *lower bound* on the matcher count that the first list alone
/// could not meet.
fn foreach_match_matchers(spec: &str) -> BTreeSet<&str> {
    const LOOP: &str = "transform.foreach_match";
    let mut matchers = BTreeSet::new();
    let mut search = 0usize;
    while let Some(offset) = spec[search..].find(LOOP) {
        let start = search + offset;
        // The list ends at the op's type signature; everything before it is
        // the `in %handle @a -> @b, ...` clause.
        let body = &spec[start..];
        let end = body.find(" : (").unwrap_or(body.len());
        for pair in body[..end].split(',') {
            let Some((lhs, _)) = pair.split_once("->") else {
                continue;
            };
            if let Some(name) = symbol_after_last_at(lhs) {
                matchers.insert(name);
            }
        }
        search = start + LOOP.len();
    }
    matchers
}

fn symbol_after_last_at(text: &str) -> Option<&str> {
    let at = text.rfind('@')?;
    let name = &text[at + 1..];
    let end = name
        .find(|c: char| !(c.is_alphanumeric() || c == '_' || c == '$' || c == '.'))
        .unwrap_or(name.len());
    (end > 0).then(|| &name[..end])
}

/// Whether `@name`'s `transform.named_sequence` body contains something
/// [`neutralize`] can rewrite into a refusal.
///
/// Delimited by the next `transform.named_sequence` declaration rather than by
/// brace matching: the spec's sequences are top-level and consecutive, and
/// brace counting would have to understand MLIR's string and attribute
/// literals to be correct.
fn sequence_is_defeatable(spec: &str, name: &str) -> bool {
    const DECL: &str = "transform.named_sequence @";
    let mut search = 0usize;
    while let Some(offset) = spec[search..].find(DECL) {
        let start = search + offset + DECL.len();
        let end = spec[start..]
            .find(|c: char| !(c.is_alphanumeric() || c == '_' || c == '$' || c == '.'))
            .map(|i| start + i)
            .unwrap_or(spec.len());
        if &spec[start..end] == name {
            let body_end = spec[end..]
                .find(DECL)
                .map(|i| end + i)
                .unwrap_or(spec.len());
            return spec[end..body_end]
                .lines()
                .filter(|line| !line.trim_start().starts_with("//"))
                .any(|line| line.contains(DIM_BOUNDS_OP) || line.contains(ADMITTED_OP));
        }
        search = start;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both halves of the edit, and the check that catches the half that is
    /// easy to forget: an uncommented pass whose consumer still reads the
    /// old handle runs on nothing, and the transform dialect accepts that
    /// silently because an unused handle is legal.
    #[test]
    fn enabling_batch_matmul_repoints_the_consumer_too() {
        let spec = concat!(
            "    %gemv_funcs = transform.apply_registered_pass\n",
            "        \"rocket-expand-gemv-to-matmul\" to %x\n",
            "//@ROCKET_BATCH_MATMUL@    %unbatched_funcs = transform.apply_registered_pass\n",
            "//@ROCKET_BATCH_MATMUL@        \"rocket-unbatch-matmul\" to %gemv_funcs\n",
            "    %recorded_funcs = transform.apply_registered_pass\n",
            "        \"rocket-record-conv-attrs\" to %gemv_funcs\n",
        );
        let enabled = super::enable_batch_matmul(spec).expect("both edits apply");
        assert_eq!(enabled.enabled, 2);
        assert!(
            enabled
                .text
                .contains("\"rocket-unbatch-matmul\" to %gemv_funcs")
        );
        assert!(
            enabled
                .text
                .contains("\"rocket-record-conv-attrs\" to %unbatched_funcs"),
            "{}",
            enabled.text
        );
        assert!(!enabled.text.contains("//@ROCKET_BATCH_MATMUL@"));
    }

    #[test]
    fn enabling_batch_matmul_refuses_a_spec_with_no_marker() {
        let err = super::enable_batch_matmul("nothing to see")
            .expect_err("a spec with no marked lines must fail");
        assert!(err.to_string().contains("--batch-matmul"), "{err}");
    }

    /// The consumer check is the point of the second half, so it gets its
    /// own case: markers present, nothing to repoint.
    #[test]
    fn enabling_batch_matmul_refuses_when_the_consumer_is_missing() {
        let spec = "//@ROCKET_BATCH_MATMUL@    %unbatched_funcs = x\n";
        let err = super::enable_batch_matmul(spec).expect_err("no consumer to repoint");
        assert!(err.to_string().contains("found 0"), "{err}");
    }

    /// The shipped spec must carry both halves, or the flag is dead.
    #[test]
    fn the_checked_in_spec_can_be_batch_matmul_enabled() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../rocket-compiler-plugin/target/Rocket/rocket_conv2d_transform_spec.mlir");
        let text = std::fs::read_to_string(&path).expect("the shipped spec");
        let enabled = super::enable_batch_matmul(&text).expect("the shipped spec must enable");
        assert_eq!(enabled.enabled, 3, "three marked lines");
    }

    const SPEC: &str = r#"
  transform.named_sequence @match_a(%arg: !transform.any_op) {
    transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 512 : !transform.any_value
    transform.yield %arg : !transform.any_op
  }
  transform.named_sequence @match_b(%arg: !transform.any_op) {
    transform.iree.match.dim_bounds %lhs_value[0], umin = 1, umax = 32 : !transform.any_value
    transform.iree.match.dim_bounds %lhs_value[1], umin = 2, umax = 1792 : !transform.any_value
    transform.yield %arg : !transform.any_op
  }
  transform.named_sequence @match_admitted(%arg: !transform.any_op) {
    transform.rocket.match.admitted %arg : !transform.any_op
    transform.yield %arg : !transform.any_op
  }
  transform.named_sequence @match_admitted_int8(%arg: !transform.any_op) {
    transform.rocket.match.admitted %arg {precision = "int8_requant"} : !transform.any_op
    transform.yield %arg : !transform.any_op
  }
  transform.named_sequence @cast_and_call_a(%arg: !transform.any_op) {
    transform.yield
  }
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    transform.foreach_match in %func
        @match_a -> @cast_and_call_a,
        @match_b -> @cast_and_call_a,
        @match_admitted -> @cast_and_call_a,
        @match_admitted_int8 -> @cast_and_call_a
      : (!transform.any_op) -> (!transform.any_op)
  }
"#;

    #[test]
    fn every_bound_becomes_the_sentinel() {
        let out = neutralize(SPEC).expect("spec is well formed");
        assert_eq!(out.rewritten, 5);
        assert_eq!(out.matchers, 4);
        assert!(!out.text.contains("umax = 512"), "{}", out.text);
        assert!(!out.text.contains("umin = 2"), "{}", out.text);
        assert_eq!(out.text.matches("umin = 999999, umax = 999999").count(), 3);
    }

    #[test]
    fn an_admission_check_is_defeated_by_the_attribute() {
        // The convolution and matmul matchers carry no bounds at all since
        // the ceilings moved to rocket-core; this is what stops them.
        let out = neutralize(SPEC).expect("spec is well formed");
        assert!(
            out.text
                .contains("transform.rocket.match.admitted %arg {no_offload} :"),
            "{}",
            out.text
        );
        // An attribute the matcher already carries survives alongside it --
        // dropping `precision` would change which envelope is asked about,
        // and a no-offload build must differ from the offload one only in
        // that nothing matches.
        assert!(
            out.text.contains(
                "transform.rocket.match.admitted %arg {no_offload, precision = \"int8_requant\"} :"
            ),
            "{}",
            out.text
        );
        // Idempotent: neutralizing an already-neutralized spec is a no-op
        // on these lines rather than a second attribute.
        let twice = neutralize(&out.text).expect("already neutralized");
        assert_eq!(twice.text, out.text);
    }

    #[test]
    fn nothing_but_the_bounds_changes() {
        let out = neutralize(SPEC).expect("spec is well formed");
        // Same line count, same everything outside the bounds themselves.
        assert_eq!(out.text.lines().count(), SPEC.lines().count());
        assert!(out.text.contains("transform.foreach_match in %func"));
        assert!(out.text.contains("%input_value[3]"));
    }

    #[test]
    fn only_matchers_in_the_foreach_list_are_read() {
        // @match_unused has no dim_bounds, but nothing invokes it, so it
        // cannot offload and must not fail the build.
        let spec = SPEC.replace(
            "  transform.named_sequence @cast_and_call_a",
            "  transform.named_sequence @match_unused(%arg: !transform.any_op) {\n    \
             transform.yield %arg : !transform.any_op\n  }\n  \
             transform.named_sequence @cast_and_call_a",
        );
        let out = neutralize(&spec).expect("an uninvoked matcher is not a problem");
        assert_eq!(out.matchers, 4);
    }

    #[test]
    fn an_invoked_matcher_without_bounds_is_refused() {
        // The failure this guards: a matcher added to the loop that constrains
        // no dimension would still fire, and the "CPU-only" baseline would
        // quietly offload.
        let spec = SPEC.replace(
            "        @match_admitted_int8 -> @cast_and_call_a\n",
            "        @match_admitted_int8 -> @cast_and_call_a,\n        \
             @match_c -> @cast_and_call_a\n",
        );
        let spec = spec.replace(
            "  transform.named_sequence @cast_and_call_a",
            "  transform.named_sequence @match_c(%arg: !transform.any_op) {\n    \
             transform.yield %arg : !transform.any_op\n  }\n  \
             transform.named_sequence @cast_and_call_a",
        );
        let err = neutralize(&spec).expect_err("an unconstrained matcher must be refused");
        let message = err.to_string();
        assert!(message.contains("@match_c"), "{message}");
        assert!(!message.contains("@match_a"), "{message}");
    }

    #[test]
    fn every_foreach_match_list_is_read() {
        // The shipped spec runs a second match loop for the requantized int8
        // path. A matcher there offloads exactly as much as one in the first
        // loop, so it has to be checked for bounds too -- reading only the
        // first list is how a "CPU-only" baseline silently offloads.
        let spec = SPEC.replace(
            "      : (!transform.any_op) -> (!transform.any_op)\n  }",
            "      : (!transform.any_op) -> (!transform.any_op)\n    \
             transform.foreach_match in %requant_func\n        \
             @match_c -> @cast_and_call_a\n      \
             : (!transform.any_op) -> (!transform.any_op)\n  }",
        );
        let unconstrained = spec.replace(
            "  transform.named_sequence @cast_and_call_a",
            "  transform.named_sequence @match_c(%arg: !transform.any_op) {\n    \
             transform.yield %arg : !transform.any_op\n  }\n  \
             transform.named_sequence @cast_and_call_a",
        );
        let err = neutralize(&unconstrained)
            .expect_err("a second loop's unconstrained matcher must be refused");
        assert!(err.to_string().contains("@match_c"), "{err}");

        let constrained = spec.replace(
            "  transform.named_sequence @cast_and_call_a",
            "  transform.named_sequence @match_c(%arg: !transform.any_op) {\n    \
             transform.iree.match.dim_bounds %input_value[3], umin = 1, umax = 512 \
             : !transform.any_value\n    \
             transform.yield %arg : !transform.any_op\n  }\n  \
             transform.named_sequence @cast_and_call_a",
        );
        let out = neutralize(&constrained).expect("a bounded matcher in either loop is fine");
        assert_eq!(out.matchers, 5);
    }

    #[test]
    fn marked_entries_are_uncommented_and_nothing_else_changes() {
        // The marker is indented here on purpose: `enable_elementwise`
        // trims leading whitespace before looking for it, so the rule does
        // not depend on the spec's own formatting.
        let spec = [
            "        transform.foreach_match in %func",
            "    //@ROCKET_ELEMENTWISE@            @match_a -> @call_a,",
            "//@ROCKET_ELEMENTWISE@            @match_b -> @call_b,",
            "            // an ordinary comment",
            "            @match_c -> @call_c",
            "",
        ]
        .join("\n");
        let spec = spec.as_str();
        let out = enable_elementwise(spec).expect("the marked entries must be enabled");
        assert_eq!(out.enabled, 2);
        assert!(out.text.contains("            @match_a -> @call_a,"));
        assert!(out.text.contains("            @match_b -> @call_b,"));
        assert!(!out.text.contains(ELEMENTWISE_MARKER));
        // Untouched lines stay byte-identical, including comments that are
        // not the marker.
        assert!(out.text.contains("// an ordinary comment"));
        assert!(out.text.contains("            @match_c -> @call_c"));
        assert!(out.text.ends_with('\n'));
    }

    /// Without this the flag would silently do nothing on a spec whose marker
    /// was renamed, and a measurement of "zero element-wise sites" would be
    /// indistinguishable from a real one.
    #[test]
    fn a_spec_with_no_marked_entries_is_refused() {
        let err =
            enable_elementwise(SPEC).expect_err("a spec with no marked entries must be refused");
        assert!(err.to_string().contains("--elementwise"));
    }

    /// The two flags compose in one direction only. Enabling first and
    /// neutralizing second means the element-wise matchers are checked for
    /// `dim_bounds` and defeated along with everything else, so
    /// `--no-offload --elementwise` is a true baseline for `--elementwise`.
    #[test]
    fn enabling_then_neutralizing_defeats_the_elementwise_matchers_too() {
        let spec = "\
transform.named_sequence @match_ew(%r: !transform.any_op) -> !transform.any_op {\n\
  transform.iree.match.dim_bounds %v[2], umin = 1, umax = 3072 : !transform.any_value\n\
}\n\
transform.named_sequence @match_conv(%r: !transform.any_op) -> !transform.any_op {\n\
  transform.iree.match.dim_bounds %v[3], umin = 1, umax = 512 : !transform.any_value\n\
}\n\
    transform.foreach_match in %func\n//@ROCKET_ELEMENTWISE@      @match_ew -> @call_ew,\n      @match_conv -> @call_conv\n";

        let enabled = enable_elementwise(spec).expect("marked entry is enabled");
        assert_eq!(enabled.enabled, 1);
        let neutralized = neutralize(&enabled.text).expect("both matchers are bounded");
        assert_eq!(
            neutralized.matchers, 2,
            "the element-wise matcher must be seen"
        );
        assert_eq!(neutralized.rewritten, 2);
        assert!(!neutralized.text.contains("umax = 3072"));
    }

    /// The shipped spec's own marked entries, checked against the file rather
    /// than a fixture: a renamed matcher or a dropped entry shows up here.
    #[test]
    fn the_shipped_spec_enables_its_elementwise_matchers() {
        let spec = std::fs::read_to_string(crate::default_transform_spec_path())
            .expect("the shipped spec must be readable");
        let enabled =
            enable_elementwise(&spec).expect("the shipped spec must carry marked entries");
        assert_eq!(
            enabled.enabled, 3,
            "expected the add/sub/mul element-wise entries"
        );

        // Enabled, they must survive `neutralize` -- which is what proves
        // each one carries a `dim_bounds`, since neutralize refuses a
        // `foreach_match` matcher that does not.
        let matchers = foreach_match_matchers(&enabled.text);
        for name in [
            "match_elementwise_add_f32",
            "match_elementwise_sub_f32",
            "match_elementwise_mul_f32",
        ] {
            assert!(
                matchers.contains(name),
                "{name} must join the foreach_match list once enabled"
            );
        }
        neutralize(&enabled.text).expect("the enabled spec must still yield a no-offload spec");
    }

    #[test]
    fn the_shipped_spec_can_be_neutralized() {
        let spec = std::fs::read_to_string(crate::default_transform_spec_path())
            .expect("the shipped spec must be readable");
        let out = neutralize(&spec).expect("the shipped spec must yield a no-offload spec");
        // Counted over the ops' own lines rather than the whole text: the
        // spec's prose names both ops -- its header explains where the
        // channel ceilings went -- and a comment decides no offload.
        let op_lines = |text: &str, op: &str| {
            text.lines()
                .filter(|line| !line.trim_start().starts_with("//"))
                .filter(|line| line.contains(op))
                .count()
        };
        assert_eq!(
            out.rewritten,
            op_lines(&spec, DIM_BOUNDS_OP) + op_lines(&spec, ADMITTED_OP)
        );
        assert!(out.matchers >= 20, "{} matchers", out.matchers);
        // The convolution and matmul matchers are defeated through the
        // admission op, the pooling and element-wise ones through their
        // bounds; both must be live in the shipped spec, or this test would
        // pass while only half the loop was disarmed.
        assert!(op_lines(&spec, ADMITTED_OP) >= 60);
        assert!(op_lines(&spec, DIM_BOUNDS_OP) >= 40);
        for line in out
            .text
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
        {
            if line.contains(DIM_BOUNDS_OP) {
                assert!(
                    line.contains("umin = 999999, umax = 999999"),
                    "left a live bound: {line}"
                );
            }
            if line.contains(ADMITTED_OP) {
                assert!(
                    line.contains(NO_OFFLOAD_ATTR),
                    "left a live admission check: {line}"
                );
            }
        }
    }
}
