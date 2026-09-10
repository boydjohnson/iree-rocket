//! The layout decision record, read back out of the IR.
//!
//! `rocket-assign-layout` (the compiler plugin) decides, at the flow phase,
//! which of each Rocket dispatch's inputs read their producer's NC1HWC2
//! cube in place and how many Rocket dispatches read its result that way,
//! writes the decision into the dispatch's trailing layout push constant,
//! and records every edge and every reader count in a
//! `rocket.layout_decisions` array attribute on the function. This module
//! reads that back -- the layout half of the placement report
//! (COMPILER_ROADMAP.md 6.2), captured at the phase that wrote it for the
//! same reason `decisions.rs` captures its record at preprocessing.
//!
//! The scanner is `decisions.rs`'s: an array of dictionaries whose values
//! are strings, `i64`s and one location.

use std::{collections::BTreeMap, fmt};

use crate::decisions::{
    balanced, format_location, leading_integer, location_aliases, symbol_name, top_level_groups,
    top_level_split, unquote,
};

const LAYOUT_ATTR: &str = "rocket.layout_decisions = ";

/// One input binding of a Rocket dispatch, and where its bytes come from.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LayoutEdge {
    pub function: String,
    /// `executable::export` of the consumer.
    pub site: String,
    /// The consumer's kernel kind.
    pub kind: String,
    pub binding: i64,
    /// The producer's `executable::export` when it is a Rocket dispatch,
    /// else `argument`, `cpu`, `tied` or `other`.
    pub producer: String,
    /// `packed` (reads the producer's cube in place) or `dense` (a pack).
    pub verdict: String,
    /// The identity's failing condition, or why the producer offers no cube.
    pub reason: String,
    pub location: String,
}

impl LayoutEdge {
    /// Whether both ends are Rocket dispatches: the edges the compiler can
    /// decide and `--strict-layout` insists on.
    pub fn npu_to_npu(&self) -> bool {
        self.producer.contains("::")
    }

    pub fn packed(&self) -> bool {
        self.verdict == "packed"
    }
}

/// One Rocket dispatch's own result: whether it publishes a cube and how
/// many readers were declared to take it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LayoutSite {
    pub function: String,
    pub site: String,
    pub kind: String,
    /// `cube` or `dense`.
    pub verdict: String,
    pub reason: String,
    pub packed_readers: i64,
    pub location: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LayoutRecord {
    pub edges: Vec<LayoutEdge>,
    pub sites: Vec<LayoutSite>,
}

impl LayoutRecord {
    /// Reads every `rocket.layout_decisions` attribute in `ir_text`.
    pub fn scan(ir_text: &str) -> Self {
        let aliases = location_aliases(ir_text);
        let mut record = LayoutRecord::default();
        for line in ir_text.lines() {
            let Some(marker) = line.find(LAYOUT_ATTR) else {
                continue;
            };
            let function = symbol_name(&line[..marker]).unwrap_or_default();
            let Some(array) = balanced(&line[marker + LAYOUT_ATTR.len()..], '[', ']') else {
                continue;
            };
            for entry in top_level_groups(array, '{', '}') {
                let mut fields = BTreeMap::new();
                for field in top_level_split(entry, ',') {
                    if let Some((key, value)) = field.split_once(" = ") {
                        fields.insert(key.trim(), value.trim());
                    }
                }
                let text = |key: &str| fields.get(key).map(|v| unquote(v)).unwrap_or_default();
                let location = fields
                    .get("loc")
                    .map(|value| {
                        if value.starts_with('#') {
                            aliases
                                .get(*value)
                                .cloned()
                                .unwrap_or_else(|| value.to_string())
                        } else {
                            format_location(value)
                        }
                    })
                    .unwrap_or_default();
                match fields.get("edge").copied() {
                    Some("\"input\"") => record.edges.push(LayoutEdge {
                        function: function.clone(),
                        site: text("site"),
                        kind: text("kind"),
                        binding: fields
                            .get("binding")
                            .map(|v| leading_integer(v))
                            .unwrap_or(0),
                        producer: text("producer"),
                        verdict: text("verdict"),
                        reason: text("reason"),
                        location,
                    }),
                    Some("\"output\"") => record.sites.push(LayoutSite {
                        function: function.clone(),
                        site: text("site"),
                        kind: text("kind"),
                        verdict: text("verdict"),
                        reason: text("reason"),
                        packed_readers: fields
                            .get("packed_readers")
                            .map(|v| leading_integer(v))
                            .unwrap_or(0),
                        location,
                    }),
                    _ => {}
                }
            }
        }
        record
    }

    pub fn is_empty(&self) -> bool {
        self.edges.is_empty() && self.sites.is_empty()
    }

    pub fn npu_edges(&self) -> impl Iterator<Item = &LayoutEdge> {
        self.edges.iter().filter(|edge| edge.npu_to_npu())
    }

    pub fn packed_npu_edges(&self) -> usize {
        self.npu_edges().filter(|edge| edge.packed()).count()
    }

    pub fn dense_npu_edges(&self) -> impl Iterator<Item = &LayoutEdge> {
        self.npu_edges().filter(|edge| !edge.packed())
    }

    /// Dispatches declared to be read only through their cube: the ones
    /// whose dense output write the runtime skips when every declared reader
    /// chains on the same command buffer.
    pub fn cube_only_sites(&self) -> usize {
        self.sites
            .iter()
            .filter(|site| site.packed_readers > 0)
            .count()
    }
}

impl fmt::Display for LayoutEdge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "  {} -> {}[{}] ({}): {}",
            self.producer, self.site, self.binding, self.kind, self.verdict
        )?;
        if !self.reason.is_empty() && !self.packed() {
            write!(f, ": {}", self.reason)?;
        }
        writeln!(f, " @ {}", self.location)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const IR: &str = r#"
  util.func public @main(%arg0: tensor<1xf16>) -> tensor<1xf16> attributes {rocket.layout_decisions = [{edge = "output", kind = "conv2d", loc = loc("m.mlir":10:3), packed_readers = 1 : i64, reason = "", site = "conv_exec::conv", verdict = "cube"}, {binding = 0 : i64, edge = "input", kind = "conv2d", loc = loc("m.mlir":10:3), producer = "argument", reason = "the value is a function argument", site = "conv_exec::conv", verdict = "dense"}, {binding = 0 : i64, edge = "input", kind = "conv2d", loc = loc("m.mlir":12:3), producer = "conv_exec::conv", reason = "geometries identical", site = "conv_exec::conv", verdict = "packed"}, {binding = 0 : i64, edge = "input", kind = "conv2d", loc = loc("m.mlir":14:3), producer = "pool_exec::pool", reason = "producer surfaces 52 pixels apart, consumer packs at 49", site = "conv_exec::conv", verdict = "dense"}]} {
    util.return %arg0 : tensor<1xf16>
  }
"#;

    #[test]
    fn the_record_reads_back_edges_and_sites() {
        let record = LayoutRecord::scan(IR);
        assert_eq!(record.edges.len(), 3);
        assert_eq!(record.sites.len(), 1);
        assert_eq!(record.sites[0].packed_readers, 1);
        assert_eq!(record.sites[0].verdict, "cube");
        assert_eq!(record.npu_edges().count(), 2);
        assert_eq!(record.packed_npu_edges(), 1);
        let dense: Vec<_> = record.dense_npu_edges().collect();
        assert_eq!(dense.len(), 1);
        assert_eq!(dense[0].producer, "pool_exec::pool");
        assert_eq!(
            dense[0].reason,
            "producer surfaces 52 pixels apart, consumer packs at 49"
        );
        assert_eq!(dense[0].location, "m.mlir:14:3");
        assert_eq!(record.cube_only_sites(), 1);
        assert_eq!(
            format!("{}", dense[0]),
            "  pool_exec::pool -> conv_exec::conv[0] (conv2d): dense: producer surfaces 52 pixels apart, consumer packs at 49 @ m.mlir:14:3\n"
        );
    }
}
