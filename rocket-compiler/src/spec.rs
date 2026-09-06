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
//! match loop defeated. Every matcher constrains at least one dimension with
//! `transform.iree.match.dim_bounds`, so rewriting every bound to
//! `umin = umax = 999999` -- larger than any dimension a real model has --
//! makes all of them decline, while leaving the passes around the loop, the
//! device topology and the placement pin exactly as the offload arm sees
//! them.

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

/// Result of neutering a spec, kept together so the caller can report what it
/// did rather than trusting it silently.
#[derive(Debug)]
pub struct NeutralizedSpec {
    pub text: String,
    /// How many `dim_bounds` bounds pairs were rewritten.
    pub rewritten: usize,
    /// The matcher names taken from the `foreach_match` list.
    pub matchers: usize,
}

/// Returns `spec` with every `dim_bounds` interval replaced by the sentinel.
///
/// Fails if any matcher in the `foreach_match` list has no `dim_bounds` at
/// all: that matcher would still fire, and the caller would get a "CPU-only"
/// baseline that quietly offloads part of the model -- exactly the class of
/// error this whole path exists to prevent. It is checked rather than assumed
/// because the spec grows matchers over time and nothing else would notice.
pub fn neutralize(spec: &str) -> Result<NeutralizedSpec, Box<dyn Error>> {
    let matchers = foreach_match_matchers(spec);
    if matchers.is_empty() {
        return Err(format!(
            "found no `{DIM_BOUNDS_OP}`-constrained matchers in the transform spec: its \
             `transform.foreach_match` list could not be read, so a no-offload spec cannot \
             be derived from it"
        )
        .into());
    }

    let unconstrained: Vec<&str> = matchers
        .iter()
        .copied()
        .filter(|name| !sequence_has_dim_bounds(spec, name))
        .collect();
    if !unconstrained.is_empty() {
        return Err(format!(
            "cannot build a no-offload spec: matcher(s) {} constrain no dimension with \
             `{DIM_BOUNDS_OP}`, so rewriting the bounds would not stop them from claiming \
             convolutions. The baseline would silently offload. Give each one a dim_bounds \
             (every other matcher has at least one), or defeat it another way.",
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
        return Err(format!("transform spec contains no `{DIM_BOUNDS_OP}` to rewrite").into());
    }

    Ok(NeutralizedSpec {
        text,
        rewritten,
        matchers: matchers.len(),
    })
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

/// Whether `@name`'s `transform.named_sequence` body contains a `dim_bounds`.
///
/// Delimited by the next `transform.named_sequence` declaration rather than by
/// brace matching: the spec's sequences are top-level and consecutive, and
/// brace counting would have to understand MLIR's string and attribute
/// literals to be correct.
fn sequence_has_dim_bounds(spec: &str, name: &str) -> bool {
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
            return spec[end..body_end].contains(DIM_BOUNDS_OP);
        }
        search = start;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

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
  transform.named_sequence @cast_and_call_a(%arg: !transform.any_op) {
    transform.yield
  }
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    transform.foreach_match in %func
        @match_a -> @cast_and_call_a,
        @match_b -> @cast_and_call_a
      : (!transform.any_op) -> (!transform.any_op)
  }
"#;

    #[test]
    fn every_bound_becomes_the_sentinel() {
        let out = neutralize(SPEC).expect("spec is well formed");
        assert_eq!(out.rewritten, 3);
        assert_eq!(out.matchers, 2);
        assert!(!out.text.contains("umax = 512"), "{}", out.text);
        assert!(!out.text.contains("umin = 2"), "{}", out.text);
        assert_eq!(out.text.matches("umin = 999999, umax = 999999").count(), 3);
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
        assert_eq!(out.matchers, 2);
    }

    #[test]
    fn an_invoked_matcher_without_bounds_is_refused() {
        // The failure this guards: a matcher added to the loop that constrains
        // no dimension would still fire, and the "CPU-only" baseline would
        // quietly offload.
        let spec = SPEC.replace(
            "        @match_b -> @cast_and_call_a\n",
            "        @match_b -> @cast_and_call_a,\n        @match_c -> @cast_and_call_a\n",
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
        assert_eq!(out.matchers, 3);
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
        assert_eq!(out.rewritten, spec.matches(DIM_BOUNDS_OP).count());
        assert!(out.matchers >= 20, "{} matchers", out.matchers);
        // Every surviving bound is the sentinel. Checked over the op's own
        // lines rather than the whole text: the spec's prose mentions bounds
        // too, and comments are not what decides an offload.
        for line in out.text.lines().filter(|l| l.contains(DIM_BOUNDS_OP)) {
            assert!(
                line.contains("umin = 999999, umax = 999999"),
                "left a live bound: {line}"
            );
        }
    }
}
