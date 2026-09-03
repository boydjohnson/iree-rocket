//! Differential check of `ConvPlan`'s **depthwise** plans against vendor
//! RKNN captures.
//!
//! Every other fixture in this suite is dense, which was a real gap:
//! depthwise has its own channel granule, its own coefficient footprint, and
//! -- the reason this corpus was built -- its own row tiling, and int8
//! depthwise is currently wrong on hardware whenever the planner splits
//! rows. Nothing had ever compared a depthwise *plan* against the vendor.
//!
//! Five extents (34, 56, 70, 112, 130) at `Cin` 48..384 in both precisions,
//! chosen so the row-tile count crosses from one to many within each
//! precision -- at 130x130 fp16 `Cin` 384 the vendor itself emits 129 row
//! tiles.
//!
//! Captured with:
//!
//! ```text
//! uv run --with torch,onnxscript,numpy build_vendor_fixtures.py \
//!   --out-dir DIR --fixture-out DIR/fixtures.json --channel-grid-only \
//!   --depthwise --channel-grid-max 384 --channel-grid-step 48 \
//!   --channel-grid-extent <E> --channel-grid-kernel 3 [--quant i8]
//! ```
//!
//! # Reading a depthwise capture
//!
//! Two things differ from the dense corpora and both will mislead you:
//!
//!   * **`plan_index` 0 is a two-program prefix, not a plan.** The complete
//!     alternatives are the higher indices. Selecting plan 0 the way the
//!     dense tests can makes every case look like a two-tile plan with a
//!     `data_entries` that does not depend on `Cin`, which is nonsense.
//!   * **`CNA_WEIGHT_SIZE2.WEIGHT_KERNELS` is 1**, not `Cout`. Depthwise
//!     carries one filter per channel, so there is no output-channel group
//!     axis and `output_channel_groups` -- which asserts the per-group
//!     counts sum to `Cout` -- does not apply here.

#[path = "support/vendor_fixture.rs"]
mod vendor_fixture;

use std::collections::BTreeMap;

use iree_rocket_hal::rocket::conv::{ConvPlan, Multiplier, Precision, Quantization, Shape};
use vendor_fixture::{Case, FixtureFile, Program, complete_row_plan};

const FP16: &str = include_str!("fixtures/conv_vendor_fixtures_depthwise.json");
const INT8: &str = include_str!("fixtures/conv_vendor_fixtures_depthwise_i8.json");

fn int8() -> Precision {
    Precision::Int8(Quantization {
        input_zero_point: 0,
        output_zero_point: 0,
        weight_zero_point: 0,
        input_scale: 1.0,
        weights_scale: 1.0,
        multiplier: Multiplier::from_ratio(1.0),
    })
}

/// One vendor plan's CBUF split, or `None` when no program in it reprograms
/// `CNA_CBUF_CON0` (a continuation program leaves the field at zero) or when
/// the programs disagree.
fn plan_bank_split(programs: &[&Program]) -> Option<(u32, u32)> {
    let mut split = None;
    for program in programs {
        let candidate = (program.cbuf_data_banks, program.cbuf_weight_banks);
        if candidate.0 + candidate.1 != 12 {
            continue;
        }
        if split.is_some_and(|existing| existing != candidate) {
            return None;
        }
        split = Some(candidate);
    }
    split
}

fn plan_for(case: &Case, precision: Precision) -> ConvPlan {
    let s = &case.shape;
    let shape = Shape::depthwise_with_precision(s.width, s.height, s.stride, s.cin, precision)
        .with_padding([s.pad_h as usize, s.pad_w as usize]);
    ConvPlan::new(shape, [s.kernel_h as usize, s.kernel_w as usize])
}

fn check(fixtures: &str, precision: Precision, label: &str) -> (usize, usize, usize) {
    let fixtures: FixtureFile = serde_json::from_str(fixtures).expect("valid fixture JSON");
    assert_eq!(fixtures.schema, 1);
    let (mut scored, mut bank_agree, mut slivered) = (0, 0, 0);
    let mut bank_bad = Vec::new();
    let mut over_tiled = Vec::new();
    eprintln!("\n=== {label} ===");
    eprintln!(
        "  {:>7}{:>6}  {:>7}  {:>9} {:>14}  {}",
        "ext", "Cin", "banks", "our tiles", "vendor alts", "our out_rows"
    );
    for case in &fixtures.cases {
        let s = &case.shape;
        assert!(s.depthwise, "{} is not a depthwise fixture", case.model);
        assert_eq!(s.cin, s.cout, "depthwise fixture must have Cout == Cin");
        let plan = plan_for(case, precision);

        let mut by_plan: BTreeMap<u32, Vec<&Program>> = BTreeMap::new();
        for program in &case.vendor_programs {
            by_plan.entry(program.plan_index).or_default().push(program);
        }
        // Only complete alternatives are comparable; see the module docs on
        // why plan 0 is not one.
        let complete: Vec<&Vec<&Program>> = by_plan
            .values()
            .filter(|programs| complete_row_plan(programs, s))
            .collect();
        if complete.is_empty() {
            eprintln!(
                "  {}x{} Cin={:<4} no complete vendor row plan",
                s.width, s.height, s.cin
            );
            continue;
        }
        scored += 1;

        let splits: Vec<(u32, u32)> = complete
            .iter()
            .filter_map(|programs| plan_bank_split(programs))
            .collect();
        let ours = (plan.data_banks(), plan.weight_banks());
        if splits.contains(&ours) {
            bank_agree += 1;
        } else {
            bank_bad.push((case.model.clone(), ours, splits.clone()));
        }

        let alternatives: Vec<usize> = complete.iter().map(|programs| programs.len()).collect();
        let fewest = *alternatives.iter().min().expect("a complete alternative");
        if plan.tiles().len() > fewest {
            over_tiled.push((case.model.clone(), plan.tiles().len(), fewest));
        }

        // A trailing tile far shorter than its siblings is the greedy
        // remainder the vendor spreads instead. Counted for visibility only;
        // see the note in the test below for why it is not a defect.
        let out_rows: Vec<u32> = plan.tiles().iter().map(|tile| tile.rows.out_rows).collect();
        let sliver = out_rows.len() > 1 && *out_rows.last().expect("a tile") * 2 <= out_rows[0];
        if sliver {
            slivered += 1;
        }
        eprintln!(
            "  {:>7}{:>6}  {:>7}  {:>9} {:>14}  {}{}",
            format!("{}x{}", s.width, s.height),
            s.cin,
            format!("{}/{}", ours.0, ours.1),
            plan.tiles().len(),
            format!("{alternatives:?}"),
            if out_rows.len() > 8 {
                format!("{:?}..", &out_rows[..6])
            } else {
                format!("{out_rows:?}")
            },
            if sliver { "  <- sliver" } else { "" },
        );
    }
    eprintln!("  bank split matches a complete vendor plan: {bank_agree}/{scored}");
    eprintln!("  row split differs in shape from the vendor: {slivered}/{scored}");
    for (model, ours, splits) in &bank_bad {
        eprintln!("    bank split differs: {model} ours={ours:?} vendor={splits:?}");
    }
    for (model, ours, fewest) in &over_tiled {
        eprintln!("    more tiles than the vendor: {model} ours={ours} vendor={fewest}");
    }
    assert!(
        bank_bad.is_empty(),
        "{} depthwise case(s) disagree with every complete vendor plan on the CBUF split",
        bank_bad.len()
    );
    assert!(
        over_tiled.is_empty(),
        "{} depthwise case(s) need more dispatches than the vendor's own plan",
        over_tiled.len()
    );
    (bank_agree, scored, slivered)
}

#[test]
fn depthwise_plans_match_vendor_captures() {
    let (fp16_agree, fp16_scored, fp16_slivers) = check(FP16, Precision::Fp16, "fp16 depthwise");
    let (int8_agree, int8_scored, int8_slivers) = check(INT8, int8(), "int8 depthwise");
    assert!(
        fp16_scored >= 35 && int8_scored >= 35,
        "depthwise corpus unexpectedly small: {fp16_scored} fp16, {int8_scored} int8"
    );
    assert_eq!((fp16_agree, int8_agree), (fp16_scored, int8_scored));

    // Our row split is greedy -- fill each tile to capacity, remainder last
    // -- where the vendor spreads the remainder over the same number of
    // tiles. At 56x56 Cin 384 fp16 both plan 10 tiles at 11/1 with an
    // 8-input-row capacity, and the vendor's output rows are
    // [6,6,6,6,6,6,5,5,5,5] where ours are [7,6,6,6,6,6,6,6,6,1].
    //
    // **This is counted, not failed on, because it costs nothing.** Measured
    // over these shapes, greedy and balanced are identical on both metrics
    // that matter: the dispatch count is the same by construction (balancing
    // holds N fixed), and the total input rows read is the same too -- the
    // halo is `N * (kernel_height - 1)` however the rows are distributed, and
    // `sum(out_rows)` is the output height either way. 880 input rows read
    // for both, across six representative shapes.
    //
    // What does differ is the tile-size spread: greedy's largest tile is one
    // row bigger and its smallest can be a single row. That would matter if
    // the makespan were set by the largest tile, but with tens of tiles over
    // three cores it is set by the total, which is equal. So this is
    // recorded as a difference from the vendor, not as a defect -- and
    // notably `fp16_depthwise_corpus_shapes_are_exact` passes 40/40 on
    // hardware, slivered plans included, so it is not a correctness issue
    // either.
    eprintln!(
        "\nplans whose row split differs in shape from the vendor's (no cost difference): \
         fp16 {fp16_slivers}/{fp16_scored}, int8 {int8_slivers}/{int8_scored}"
    );
}
