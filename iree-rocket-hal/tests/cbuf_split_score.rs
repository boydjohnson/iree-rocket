//! Scores ConvPlan's CBUF bank split against the vendor fixture corpus.
//!
//! This exists to evaluate a candidate split rule, so unlike
//! `conv_vendor_fixture_channels_768` it deliberately scores the *whole* 12x12
//! channel grid -- including the `Cin > MAX_INPUT_CHANNELS` cases that the gate
//! test skips as exploratory, which are exactly the cases the split is known to
//! disagree on. It asserts nothing about the diverging region; it prints an
//! agreement count so two rules can be compared.
//!
//! Run with:
//!   cargo test -p iree-rocket-hal --test cbuf_split_score -- --ignored --nocapture

#[path = "support/vendor_fixture.rs"]
mod vendor_fixture;

use std::collections::{BTreeMap, BTreeSet};

use iree_rocket_hal::rocket::conv::{
    ConvPlan, Kernels, Multiplier, Precision, Quantization, Shape,
};
use vendor_fixture::{FixtureFile, Program};

const FP16_FIXTURES: &str = include_str!("fixtures/conv_vendor_fixtures_channels_768.json");
const INT8_FIXTURES: &str = include_str!("fixtures/conv_vendor_fixtures_channels_768_i8.json");
/// The base corpus: k in {1,3,5,7} and small Cin, so unlike the 12x12 channel
/// grid it actually produces partially-filled trailing CBUF entries.
const BASE_FP16_FIXTURES: &str = include_str!("fixtures/conv_vendor_fixtures.json");
const BASE_INT8_FIXTURES: &str = include_str!("fixtures/conv_vendor_fixtures_i8.json");
/// Six spatial extents (7, 14, 28, 56, 112, 224) at Cout 256, k=3, Cin 64..768.
///
/// The channel grids above are 28x28 **only**, which is how a split rule that
/// scored 143/144 there still computed wrong values at 56x56 on hardware. This
/// is the axis that gap was hiding in.
const SPATIAL_FP16_FIXTURES: &str = include_str!("fixtures/conv_vendor_fixtures_spatial.json");
const SPATIAL_INT8_FIXTURES: &str = include_str!("fixtures/conv_vendor_fixtures_spatial_i8.json");

fn plan_bank_split(programs: &[&Program]) -> Option<(u32, u32)> {
    let splits = programs
        .iter()
        .filter_map(|program| {
            let split = (program.cbuf_data_banks, program.cbuf_weight_banks);
            (split.0 + split.1 == 12).then_some(split)
        })
        .collect::<BTreeSet<_>>();
    (splits.len() == 1).then(|| *splits.first().unwrap())
}

fn score(fixtures: &str, precision: Precision, label: &str) {
    // The corpus runs to Cin 768; planning those is the entire point here.
    // Safe because nothing built by this harness is ever dispatched.
    unsafe { std::env::set_var("ROCKET_ALLOW_UNBACKED_CHANNELS", "1") };
    let fixtures: FixtureFile = serde_json::from_str(fixtures).expect("valid fixture JSON");
    // The planner asserts on purpose for unbacked shapes; keep the output readable.
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));

    let mut scored = 0;
    let mut agree = 0;
    let mut unusable = 0;
    let mut unplannable = 0;
    let mut capacity_sum: u64 = 0;
    // (cin, kernel) -> (agree, total), so a kernel-dependent rule is visible.
    let mut by_cin_kernel = BTreeMap::<(u32, u32), (u32, u32)>::new();
    let mut disagreements = Vec::new();

    for case in &fixtures.cases {
        let s = &case.shape;
        let mut by_plan = BTreeMap::<u32, Vec<&Program>>::new();
        for program in &case.vendor_programs {
            by_plan.entry(program.plan_index).or_default().push(program);
        }
        let Some(plan_zero) = by_plan.get(&0) else {
            unusable += 1;
            continue;
        };
        let Some(vendor) = plan_bank_split(plan_zero) else {
            unusable += 1;
            continue;
        };

        // Above the capture-backed channel cap the planner asserts rather
        // than guessing a split (a saturating coefficient working set has no
        // grantable partition). Count those separately instead of aborting:
        // "cannot plan" is a distinct outcome from "planned differently".
        let (pad_h, pad_w) = (s.pad_h as usize, s.pad_w as usize);
        let (width, height, stride, cin, cout) = (s.width, s.height, s.stride, s.cin, s.cout);
        let kernels: Kernels = [s.kernel_h as usize, s.kernel_w as usize];
        let planned = std::panic::catch_unwind(|| {
            let shape = Shape::with_precision(width, height, stride, cin, cout, precision)
                .with_padding([pad_h, pad_w]);
            let plan = ConvPlan::new(shape, kernels);
            let rows = shape.max_tile_input_rows_for_data_banks(plan.data_banks());
            (plan.data_banks(), plan.weight_banks(), rows)
        });
        let Ok((d, w, rows)) = planned else {
            unplannable += 1;
            by_cin_kernel.entry((s.width, s.cin)).or_insert((0, 0)).1 += 1;
            continue;
        };

        capacity_sum += u64::from(rows);
        let ours = (d, w);
        scored += 1;
        let entry = by_cin_kernel.entry((s.width, s.cin)).or_insert((0, 0));
        entry.1 += 1;
        if ours == vendor {
            agree += 1;
            entry.0 += 1;
        } else {
            disagreements.push((s.cin, s.cout, s.kernel_h, ours, vendor));
        }
    }

    std::panic::set_hook(previous_hook);
    println!("\n=== {label} ===");
    println!(
        "  agreement: {agree}/{scored}   (unplannable: {unplannable}, unusable fixtures: {unusable})\n  sum(max_tile_input_rows) = {capacity_sum}"
    );
    println!("  by (extent, Cin):");
    for ((ext, cin), (a, t)) in &by_cin_kernel {
        let mark = if a == t { " " } else { "*" };
        println!("   {mark} {ext}^2 Cin={cin:<4}  {a}/{t}");
    }
    if !disagreements.is_empty() {
        println!("  disagreements (Cin, Cout, k, ours d/w, vendor d/w):");
        for (cin, cout, k, ours, vendor) in &disagreements {
            println!(
                "      Cin={cin:<4} Cout={cout:<4} k={k}  ours={}/{}  vendor={}/{}",
                ours.0, ours.1, vendor.0, vendor.1
            );
        }
    }
}

#[test]
#[ignore = "scoring harness, not a gate; run with --ignored --nocapture"]
fn score_cbuf_split_against_corpus() {
    score(FP16_FIXTURES, Precision::Fp16, "fp16 12x12 channel grid");
    score(BASE_FP16_FIXTURES, Precision::Fp16, "fp16 base corpus (k=1,3,5,7)");
    score(SPATIAL_FP16_FIXTURES, Precision::Fp16, "fp16 spatial sweep (6 extents)");
    // The quantization parameters do not affect the bank split; these mirror
    // the ones the gate test uses so the two are comparing like with like.
    let int8 = Precision::Int8(Quantization {
        input_zero_point: 0,
        output_zero_point: -3,
        weight_zero_point: 0,
        input_scale: 1.0,
        weights_scale: 1.0,
        multiplier: Multiplier {
            scale: 19636,
            shift: 24,
        },
    });
    score(INT8_FIXTURES, int8, "int8 12x12 channel grid");
    score(BASE_INT8_FIXTURES, int8, "int8 base corpus (k=1,3,5,7)");
    score(SPATIAL_INT8_FIXTURES, int8, "int8 spatial sweep (6 extents)");
}
