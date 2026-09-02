//! Field-by-field diff of ConvPlan's CBUF residency against vendor captures,
//! for dense 1x1 across the `Cin` range where the device is known wrong.
//!
//! Dense int8 1x1 is exact to `Cin` 384 and wrong from 400 up, with exactly one
//! output row correct -- the signature of a residency miscount. Every previous
//! attempt on it used raw hardware probes, because there were **no vendor
//! captures at k=1 above `Cin` 128** to compare against. This supplies them and
//! reports which field first disagrees, rather than a pass/fail.
//!
//!   cargo test -p iree-rocket-hal --test k1_residency_diff -- --ignored --nocapture

#[path = "support/vendor_fixture.rs"]
mod vendor_fixture;

use std::collections::BTreeMap;

use iree_rocket_hal::rocket::builders::RegCmd;
use iree_rocket_hal::rocket::builders::cna::{CnaCbufCon1, CnaConvCon2};
use iree_rocket_hal::rocket::conv::{
    ConvPlan, Kernels, Multiplier, Precision, Quantization, Shape,
};
use vendor_fixture::{FixtureFile, Program, register_value};

const K1_FP16: &str = include_str!("fixtures/conv_vendor_fixtures_k1.json");
const K1_INT8: &str = include_str!("fixtures/conv_vendor_fixtures_k1_i8.json");

/// `feature_grains` occupies `CNA_CONV_CON2` from bit 4.
fn grains(commands: &[RegCmd]) -> Option<u32> {
    register_value::<CnaConvCon2>(commands).map(|raw| (raw >> 4) & 0x3ff)
}

fn diff(fixtures: &str, precision: Precision, label: &str) {
    let fixtures: FixtureFile = serde_json::from_str(fixtures).expect("valid fixture JSON");
    println!("\n=== {label} ===");
    println!(
        "  {:<6} {:<11} {:<13} {:<15} {:<13} {}",
        "Cin", "tiles v/o", "banks v/o", "in_rows v/o", "entries v/o", "grains v/o"
    );

    for case in &fixtures.cases {
        let s = &case.shape;
        let shape = Shape::with_precision(s.width, s.height, s.stride, s.cin, s.cout, precision)
            .with_padding([s.pad_h as usize, s.pad_w as usize]);
        let kernels: Kernels = [s.kernel_h as usize, s.kernel_w as usize];

        let plan = match std::panic::catch_unwind(|| ConvPlan::new(shape, kernels)) {
            Ok(plan) => plan,
            Err(_) => {
                println!("  {:<6} <planner refused>", s.cin);
                continue;
            }
        };

        let mut by_plan: BTreeMap<u32, Vec<&Program>> = BTreeMap::new();
        for program in &case.vendor_programs {
            by_plan.entry(program.plan_index).or_default().push(program);
        }
        let Some(plan_zero) = by_plan.get(&0) else {
            println!("  {:<6} <no plan-0 programs>", s.cin);
            continue;
        };
        let mut vendor = plan_zero.clone();
        vendor.sort_by_key(|program| program.out_offset);

        let generated = plan.programs();
        let tiles = plan.tiles();
        let first = vendor.first().expect("plan-0 is non-empty");
        let our_rows = tiles.first().map(|t| t.rows.in_rows).unwrap_or(0);
        let our_entries = generated
            .first()
            .and_then(|c| register_value::<CnaCbufCon1>(c))
            .unwrap_or(0);
        let our_grains = generated.first().and_then(|c| grains(c)).unwrap_or(0);

        let mark = |a: u32, b: u32| if a == b { ' ' } else { '*' };
        println!(
            "  {:<6} {}{:<10} {}{:<12} {}{:<14} {}{:<12} {}{}",
            s.cin,
            mark(vendor.len() as u32, tiles.len() as u32),
            format!("{}/{}", vendor.len(), tiles.len()),
            mark(first.cbuf_weight_banks, plan.weight_banks()),
            format!(
                "{}/{}/{}/{}",
                first.cbuf_data_banks,
                first.cbuf_weight_banks,
                plan.data_banks(),
                plan.weight_banks()
            ),
            mark(first.in_rows, our_rows),
            format!("{}/{}", first.in_rows, our_rows),
            mark(first.cbuf_data_entries, our_entries),
            format!("{}/{}", first.cbuf_data_entries, our_entries),
            mark(first.feature_grains, our_grains),
            format!("{}/{}", first.feature_grains, our_grains),
        );
    }
}

#[test]
#[ignore = "diagnostic, not a gate; run with --ignored --nocapture"]
fn k1_residency_fields_match_vendor() {
    diff(K1_FP16, Precision::Fp16, "fp16 dense 1x1, 32x32 Cout 64");
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
    diff(K1_INT8, int8, "int8 dense 1x1, 32x32 Cout 64");
}
