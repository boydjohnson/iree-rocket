//! Selection-time decisions reconciled with final placement.
//!
//! COMPILER_ROADMAP.md section 3. Two records are taken from one compile,
//! at the two phases where each is true:
//!
//!   * the **decision record** (`decisions.rs`) at the end of preprocessing,
//!     where every convolution and matmul candidate is still an op and the
//!     shared planner has just been asked about it;
//!   * the **placement report** (`report.rs`) at `executable-targets`, where
//!     the dispatches those candidates actually became exist.
//!
//! Neither alone answers "why is this operation not on the NPU". The
//! decision record knows the planner's and the admission envelope's reasons
//! but not whether a matcher went on to claim the op; the placement report
//! knows what ran where but nothing about why. Reconciling them is what
//! separates the three cases a reader cares about: the planner refused, the
//! envelope has no evidence, or both said yes and a *matcher* still declined
//! -- which is a semantic or fusion gap in the transform spec, and the only
//! one of the three that this repository can close without new hardware
//! measurements.
//!
//! What the reconciliation deliberately does not do is claim a per-op
//! identity between the two records. A Rocket dispatch has no per-op name --
//! the transform spec splices a handful of fixed executables across every
//! matched shape -- and a CPU dispatch's name encodes its own loop ranges,
//! not the candidate's logical shape. So the join is by count and by op
//! *kind*, and every statement below is one those two support.

use std::fmt;

use crate::{decisions::DecisionRecord, report::PlacementReport};

/// How the two records line up.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Reconciliation {
    pub candidates: usize,
    pub accepted: usize,
    pub hardware_jobs: i64,
    pub rocket_dispatches: usize,
    pub cpu_dispatches: usize,
    pub contraction_cpu_dispatches: usize,
}

impl Reconciliation {
    /// Accepted candidates with no Rocket dispatch site to account for them.
    /// A floor, not a total: it assumes every Rocket site belongs to a
    /// candidate, which over-counts sites only when a pooling or element-wise
    /// matcher also claimed something.
    pub fn unclaimed(&self) -> usize {
        self.accepted.saturating_sub(self.rocket_dispatches)
    }

    /// Rocket dispatch sites beyond the accepted convolutions and matmuls:
    /// the pooling and element-wise matchers, which the decision record does
    /// not cover.
    pub fn beyond_candidates(&self) -> usize {
        self.rocket_dispatches.saturating_sub(self.accepted)
    }
}

pub struct PlacementAudit {
    pub decisions: DecisionRecord,
    pub placement: PlacementReport,
}

impl PlacementAudit {
    pub fn new(decisions: DecisionRecord, placement: PlacementReport) -> Self {
        PlacementAudit {
            decisions,
            placement,
        }
    }

    pub fn reconciliation(&self) -> Reconciliation {
        Reconciliation {
            candidates: self.decisions.decisions.len(),
            accepted: self.decisions.accepted(),
            hardware_jobs: self.decisions.hardware_jobs(),
            rocket_dispatches: self.placement.rocket_dispatches(),
            cpu_dispatches: self.placement.cpu_dispatches(),
            contraction_cpu_dispatches: self.placement.contraction_cpu_dispatches(),
        }
    }

    /// The strict-offload verdict: `None` when every convolution and matmul
    /// candidate reached the NPU, otherwise the reason it did not.
    ///
    /// Documented scope. This fails on (a) any candidate the planner, the
    /// admission envelope or the op's own form ruled out, (b) any accepted
    /// candidate with no Rocket dispatch site to account for it, and (c) any
    /// CPU dispatch whose export name says it is running a convolution or a
    /// matmul. It cannot see an operation IREE fused into a larger
    /// element-wise dispatch, which leaves no named evidence anywhere in the
    /// module -- so a pass here means "nothing left that this compile can
    /// observe on the CPU", not a proof.
    ///
    /// An empty decision record is itself a failure: with nothing to check
    /// against, a silent pass would be the worst possible answer.
    pub fn strict_offload_failure(&self) -> Option<String> {
        if self.decisions.is_empty() {
            return Some(
                "--strict-offload: no rocket.plan_decisions record was found, so no candidate \
                 could be checked. Either the module holds no convolution or matmul, or \
                 rocket-plan-candidates did not run."
                    .to_string(),
            );
        }
        let mut problems = Vec::new();
        for decision in self.decisions.not_accepted() {
            problems.push(format!(
                "  {} at {} -> {} [{}]: {}",
                decision.kind,
                decision.location,
                decision.decision,
                decision.status,
                decision.detail
            ));
        }
        let reconciliation = self.reconciliation();
        if reconciliation.unclaimed() > 0 {
            problems.push(format!(
                "  {} accepted candidate(s) reached no Rocket dispatch site; no matcher in the \
                 transform spec claimed them",
                reconciliation.unclaimed()
            ));
        }
        for executable in self.placement.contraction_cpu_executables() {
            problems.push(format!(
                "  {} ({} CPU dispatch site(s)) is running {}",
                executable.name,
                executable.dispatches,
                executable.kinds().join(", ")
            ));
        }
        if problems.is_empty() {
            return None;
        }
        // Findings, not candidates: a candidate that was refused and the CPU
        // dispatch that ended up running it are two lines about one
        // operation, and calling that "two candidates" would be wrong.
        Some(format!(
            "--strict-offload: offload is incomplete, {} finding(s):\n{}",
            problems.len(),
            problems.join("\n")
        ))
    }

    /// The machine-readable form. Hand-rolled rather than pulling in a
    /// serializer: the shapes are flat and the escaping is the whole of the
    /// work.
    pub fn to_json(&self) -> String {
        let mut out = String::from("{\n  \"candidates\": [\n");
        for (i, decision) in self.decisions.decisions.iter().enumerate() {
            let comma = if i + 1 == self.decisions.decisions.len() {
                ""
            } else {
                ","
            };
            out.push_str(&format!(
                "    {{\"function\": {}, \"kind\": {}, \"layout\": {}, \"shape\": {}, \
                 \"precision\": {}, \"decision\": {}, \"status\": {}, \"limit\": {}, \
                 \"detail\": {}, \"hardware_jobs\": {}, \"columns\": {}, \"location\": {}}}{comma}\n",
                json_string(&decision.function),
                json_string(&decision.kind),
                json_string(&decision.layout),
                json_string(&decision.shape),
                json_string(&decision.precision),
                json_string(&decision.decision),
                json_string(&decision.status),
                json_string(&decision.limit),
                json_string(&decision.detail),
                decision.jobs,
                decision.columns,
                json_string(&decision.location),
            ));
        }
        let r = self.reconciliation();
        out.push_str("  ],\n  \"summary\": {");
        out.push_str(&format!(
            "\"candidates\": {}, \"direct\": {}, \"tiled\": {}, \"cpu\": {}, \"deferred\": {}, \
             \"hardware_jobs\": {}",
            r.candidates,
            self.decisions.count_of("direct"),
            self.decisions.count_of("tiled"),
            self.decisions.count_of("cpu"),
            self.decisions.count_of("deferred"),
            r.hardware_jobs,
        ));
        out.push_str("},\n  \"placement\": {\n");
        out.push_str(&format!(
            "    \"rocket\": {{\"executables\": {}, \"dispatch_sites\": {}}},\n",
            self.placement.rocket_executables.len(),
            r.rocket_dispatches
        ));
        out.push_str(&format!(
            "    \"cpu\": {{\"executables\": {}, \"dispatch_sites\": {}, \
             \"contraction_dispatch_sites\": {}}}\n",
            self.placement.cpu_executables.len(),
            r.cpu_dispatches,
            r.contraction_cpu_dispatches
        ));
        out.push_str("  },\n  \"reconciliation\": {");
        out.push_str(&format!(
            "\"accepted\": {}, \"rocket_dispatch_sites\": {}, \"unclaimed\": {}, \
             \"rocket_dispatch_sites_beyond_candidates\": {}, \
             \"contraction_cpu_dispatch_sites\": {}",
            r.accepted,
            r.rocket_dispatches,
            r.unclaimed(),
            r.beyond_candidates(),
            r.contraction_cpu_dispatches,
        ));
        out.push_str("}\n}\n");
        out
    }
}

fn json_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

impl fmt::Display for PlacementAudit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let r = self.reconciliation();
        if self.decisions.is_empty() {
            writeln!(
                f,
                "Placement decisions: none recorded (no convolution or matmul candidate, or \
                 rocket-plan-candidates did not run)."
            )?;
        } else {
            writeln!(
                f,
                "Placement decisions ({} candidate(s) at selection time):",
                r.candidates
            )?;
            for decision in &self.decisions.decisions {
                write!(f, "{decision}")?;
            }
            writeln!(
                f,
                "  {} direct, {} tiled, {} cpu, {} deferred; {} hardware job(s) planned",
                self.decisions.count_of("direct"),
                self.decisions.count_of("tiled"),
                self.decisions.count_of("cpu"),
                self.decisions.count_of("deferred"),
                r.hardware_jobs,
            )?;
        }
        writeln!(f)?;
        write!(f, "{}", self.placement)?;
        writeln!(f)?;
        writeln!(f, "Reconciliation:")?;
        writeln!(
            f,
            "  {} accepted candidate(s) -> {} Rocket dispatch site(s) running {} hardware job(s)",
            r.accepted, r.rocket_dispatches, r.hardware_jobs
        )?;
        if r.unclaimed() > 0 {
            writeln!(
                f,
                "  at least {} accepted candidate(s) reached no Rocket dispatch: the planner and \
                 the admission envelope both said yes, so a matcher in the transform spec \
                 declined them",
                r.unclaimed()
            )?;
        }
        if r.beyond_candidates() > 0 {
            writeln!(
                f,
                "  {} Rocket dispatch site(s) beyond the convolution and matmul candidates: the \
                 pooling and element-wise matchers, which this record does not cover",
                r.beyond_candidates()
            )?;
        }
        writeln!(
            f,
            "  {} CPU dispatch site(s), {} of them convolution- or matmul-shaped",
            r.cpu_dispatches, r.contraction_cpu_dispatches
        )?;
        for executable in self.placement.contraction_cpu_executables() {
            writeln!(
                f,
                "    - {} ({} site(s)): {}",
                executable.name,
                executable.dispatches,
                executable.kinds().join(", ")
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A preprocessing-phase function carrying three decisions: one tiled
    /// and accepted, one the planner refused, one the admission envelope
    /// declined.
    const DECISIONS_IR: &str = r#"
  util.func public @main(%arg0: tensor<1xf16>) -> tensor<1xf16> attributes {rocket.plan_decisions = [{columns = 1 : i64, decision = "tiled", detail = "cbuf 11/1, tiles 6, columns 1", jobs = 6 : i64, kind = "dense_conv2d", layout = "nhwc", limit = "none", loc = loc("m.mlir":10:3), precision = "fp16", shape = "226x226 Cin 3 Cout 32 k3x3 s2", status = "ok"}, {columns = 0 : i64, decision = "cpu", detail = "output height disagrees with the input", jobs = 0 : i64, kind = "dense_conv2d", layout = "nhwc", limit = "shape", loc = loc("m.mlir":20:3), precision = "fp16", shape = "225x225 Cin 32 Cout 32 k3x3 s1", status = "invalid_shape"}, {columns = 0 : i64, decision = "cpu", detail = "input channels must be 1..=512, not 960", jobs = 0 : i64, kind = "depthwise_conv2d", layout = "nchw", limit = "validation", loc = loc("m.mlir":30:3), precision = "fp16", shape = "9x9 Cin 960 Cout 960 k3x3 s1", status = "unvalidated_configuration"}]} {
    util.return %arg0 : tensor<1xf16>
  }
"#;

    /// The matching `executable-targets` module: one Rocket dispatch site
    /// and one CPU convolution.
    const PLACEMENT_IR: &str = r#"
  hal.executable private @rocket_dynamic_executable attributes {rocket.final = "rocket"} {
    hal.executable.variant public @rocket_dynamic_conv2d_v1 target(<"rocket", "rocket-flatbuffer-v1">) {
      hal.executable.export public @rocket_dynamic_conv2d ordinal(0) layout(#layout) {
      }
    }
  }
  hal.executable private @main$async_dispatch_9 attributes {rocket.final = "cpu"} {
    hal.executable.variant public @embedded_elf_arm_64 target(<"llvm-cpu", "embedded-elf-arm_64">) {
      hal.executable.export public @main$async_dispatch_9_conv_9x9x960x3x3_f16xf16xf32 ordinal(0) layout(#layout) {
      }
    }
  }
  util.func public @main$async() {
    %0 = stream.cmd.execute with() {
      stream.cmd.dispatch @rocket_dynamic_executable::@rocket_dynamic_conv2d_v1::@rocket_dynamic_conv2d() {
      }
    } => !stream.timepoint
    %1 = stream.cmd.execute with() {
      stream.cmd.dispatch @main$async_dispatch_9::@embedded_elf_arm_64::@main$async_dispatch_9_conv_9x9x960x3x3_f16xf16xf32() {
      }
    } => !stream.timepoint
    util.return
  }
"#;

    fn audit() -> PlacementAudit {
        PlacementAudit::new(
            DecisionRecord::scan(DECISIONS_IR),
            PlacementReport::scan(PLACEMENT_IR),
        )
    }

    #[test]
    fn the_three_counts_stay_separate() {
        let r = audit().reconciliation();
        assert_eq!(r.candidates, 3);
        assert_eq!(r.accepted, 1);
        // One dispatch site, six hardware jobs behind it: the distinction
        // section 3 asks the report to keep.
        assert_eq!(r.rocket_dispatches, 1);
        assert_eq!(r.hardware_jobs, 6);
        assert_eq!(r.unclaimed(), 0);
        assert_eq!(r.beyond_candidates(), 0);
        assert_eq!(r.contraction_cpu_dispatches, 1);
    }

    #[test]
    fn each_refusal_names_its_limit_class() {
        let text = audit().to_string();
        assert!(text.contains("-> CPU [invalid_shape] [fp16]"), "{text}");
        assert!(
            text.contains("the operation is malformed on its own terms"),
            "{text}"
        );
        assert!(
            text.contains("validation policy: register-representable, not measured"),
            "{text}"
        );
        assert!(
            text.contains("dense_conv2d at m.mlir:10:3 -> Rocket, tiled into 6 hardware job(s)"),
            "{text}"
        );
    }

    #[test]
    fn a_cpu_convolution_is_named_in_the_reconciliation() {
        let text = audit().to_string();
        assert!(
            text.contains("1 CPU dispatch site(s), 1 of them convolution- or matmul-shaped"),
            "{text}"
        );
        assert!(
            text.contains("main$async_dispatch_9 (1 site(s)): conv"),
            "{text}"
        );
    }

    /// The case the reconciliation exists for: the planner and the envelope
    /// both accepted, and no Rocket dispatch site came of it. Only the two
    /// records together can say that.
    #[test]
    fn an_accepted_candidate_with_no_dispatch_is_reported_as_a_matcher_gap() {
        let audit = PlacementAudit::new(
            DecisionRecord::scan(DECISIONS_IR),
            PlacementReport::scan(""),
        );
        assert_eq!(audit.reconciliation().unclaimed(), 1);
        let text = audit.to_string();
        assert!(
            text.contains("at least 1 accepted candidate(s) reached no Rocket dispatch"),
            "{text}"
        );
        assert!(
            text.contains("a matcher in the transform spec declined them"),
            "{text}"
        );
    }

    #[test]
    fn strict_offload_names_every_candidate_that_stayed_behind() {
        let failure = audit()
            .strict_offload_failure()
            .expect("two refusals and a CPU convolution must fail strict offload");
        assert!(
            failure.contains("offload is incomplete, 3 finding(s)"),
            "{failure}"
        );
        assert!(
            failure.contains("m.mlir:20:3 -> cpu [invalid_shape]"),
            "{failure}"
        );
        assert!(
            failure.contains("m.mlir:30:3 -> cpu [unvalidated_configuration]"),
            "{failure}"
        );
        assert!(failure.contains("main$async_dispatch_9"), "{failure}");
    }

    /// Nothing to check against is a failure, not a pass.
    #[test]
    fn strict_offload_refuses_an_empty_record() {
        let audit = PlacementAudit::new(DecisionRecord::default(), PlacementReport::scan(""));
        let failure = audit.strict_offload_failure().expect("must not pass");
        assert!(
            failure.contains("no rocket.plan_decisions record"),
            "{failure}"
        );
    }

    #[test]
    fn a_fully_offloaded_module_passes_strict_offload() {
        const ACCEPTED: &str = r#"
  util.func public @main() attributes {rocket.plan_decisions = [{columns = 1 : i64, decision = "direct", detail = "cbuf 6/6", jobs = 1 : i64, kind = "matmul", layout = "row_major", limit = "none", loc = loc("m.mlir":4:3), precision = "fp16", shape = "197x768 x 768x768", status = "ok"}]} {
    util.return
  }
  hal.executable private @rocket_matmul_executable attributes {rocket.final = "rocket"} {
    hal.executable.variant public @v target(<"rocket", "rocket-flatbuffer-v1">) {
      hal.executable.export public @rocket_matmul ordinal(0) layout(#layout) {
      }
    }
  }
  util.func public @main$async() {
    %0 = stream.cmd.execute with() {
      stream.cmd.dispatch @rocket_matmul_executable::@v::@rocket_matmul() {
      }
    } => !stream.timepoint
    util.return
  }
"#;
        let audit = PlacementAudit::new(
            DecisionRecord::scan(ACCEPTED),
            PlacementReport::scan(ACCEPTED),
        );
        assert_eq!(audit.strict_offload_failure(), None);
    }

    #[test]
    fn the_json_carries_every_field_and_the_reconciliation() {
        let json = audit().to_json();
        assert!(json.contains("\"kind\": \"depthwise_conv2d\""), "{json}");
        assert!(json.contains("\"limit\": \"validation\""), "{json}");
        assert!(json.contains("\"precision\": \"fp16\""), "{json}");
        assert!(json.contains("\"hardware_jobs\": 6"), "{json}");
        assert!(json.contains("\"location\": \"m.mlir:30:3\""), "{json}");
        assert!(
            json.contains("\"contraction_cpu_dispatch_sites\": 1"),
            "{json}"
        );
        // Every `"` in a detail string has to survive the round trip.
        assert!(!json.contains("\n\"detail\""), "{json}");
    }
}
