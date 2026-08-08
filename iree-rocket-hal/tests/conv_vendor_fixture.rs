#[path = "support/vendor_fixture.rs"]
mod vendor_fixture;

use std::collections::BTreeMap;

use iree_rocket_hal::rocket::conv::{ConvPlan, Kernels, Shape};
use vendor_fixture::{
    Case, FixtureFile, Program, complete_row_plan, decoded_prefix_matches, exact_plan_match,
    normalize_row_tiles, output_channel_groups,
};

const FIXTURES: &str = include_str!("fixtures/conv_vendor_fixtures.json");
const ELEM_BYTES: u32 = 2;

fn print_difference(
    case: &Case,
    plan: &ConvPlan,
    by_plan: &BTreeMap<u32, Vec<&Program>>,
    normalized_by_plan: &BTreeMap<u32, Vec<&Program>>,
) {
    let s = &case.shape;
    eprintln!(
        "\n=== vendor/ConvPlan divergence: {} ===\n  shape={}x{} Cin={} Cout={} K={}x{} pad={}x{} stride={}\n  ConvPlan: banks d/w={}/{}, tiles={}",
        case.model,
        s.width,
        s.height,
        s.cin,
        s.cout,
        s.kernel_h,
        s.kernel_w,
        s.pad_h,
        s.pad_w,
        s.stride,
        plan.data_banks(),
        plan.weight_banks(),
        plan.tiles().len(),
    );
    for (index, tile) in plan.tiles().iter().enumerate() {
        eprintln!(
            "    tile {index}: rows out={}+{} in={}+{}; cols out={}+{} in={}+{}",
            tile.rows.out_first,
            tile.rows.out_rows,
            tile.rows.in_first,
            tile.rows.in_rows,
            tile.columns.out_first,
            tile.columns.out_cols,
            tile.columns.in_first,
            tile.columns.in_cols,
        );
    }
    eprintln!("  Vendor alternatives:");
    for (index, programs) in by_plan {
        let normalized = &normalized_by_plan[index];
        let banks = programs
            .first()
            .map(|program| format!("{}/{}", program.cbuf_data_banks, program.cbuf_weight_banks))
            .unwrap_or_else(|| "?".to_string());
        eprintln!(
            "    plan {index}: raw_programs={} normalized_row_tiles={} complete_row_plan={} banks d/w={banks}",
            programs.len(),
            normalized.len(),
            complete_row_plan(normalized, &case.shape),
        );
        for (program_index, program) in normalized.iter().enumerate() {
            eprintln!(
                "      program {program_index}: in={}x{} @0x{:x}, out={}x{} @0x{:x}, grains={}, entries={}, atomics={}",
                program.in_width,
                program.in_rows,
                program.in_offset,
                program.out_width,
                program.out_rows,
                program.out_offset,
                program.feature_grains,
                program.cbuf_data_entries,
                program.out_atomics,
            );
        }
    }
}

#[test]
fn vendor_fixture_plans_cover_convplan_shapes() {
    let fixtures: FixtureFile = serde_json::from_str(FIXTURES).expect("valid fixture JSON");
    assert_eq!(fixtures.schema, 1);
    assert!(
        fixtures.cases.len() >= 100,
        "expanded vendor corpus unexpectedly contains only {} cases",
        fixtures.cases.len()
    );
    let mut matching_cases = 0;
    let mut multi_group_cases = 0;
    let mut incomplete_cases = Vec::new();
    let mut layout_cases = Vec::new();
    let mut divergences = Vec::new();

    for case in fixtures.cases {
        let s = &case.shape;
        let shape = Shape::with_precision(
            s.width,
            s.height,
            s.stride,
            s.cin,
            s.cout,
            iree_rocket_hal::rocket::conv::Precision::Fp16,
        )
        .with_padding([s.pad_h as usize, s.pad_w as usize]);
        let kernels: Kernels = [s.kernel_h as usize, s.kernel_w as usize];
        let plan = ConvPlan::new(shape, kernels);

        let mut by_plan: BTreeMap<u32, Vec<&Program>> = BTreeMap::new();
        for program in &case.vendor_programs {
            by_plan.entry(program.plan_index).or_default().push(program);
        }
        let normalized_by_plan: BTreeMap<u32, Vec<&Program>> = by_plan
            .iter()
            .map(|(index, programs)| (*index, normalize_row_tiles(programs)))
            .collect();

        let complete = normalized_by_plan
            .iter()
            .filter(|(_, programs)| complete_row_plan(programs, s))
            .collect::<Vec<_>>();
        if complete.is_empty() {
            incomplete_cases.push(case.model.clone());
            continue;
        }
        let matching = complete
            .iter()
            .copied()
            .filter(|(_, programs)| exact_plan_match(programs, s, &plan))
            .collect::<Vec<_>>();
        if matching.is_empty() {
            if normalized_by_plan.get(&0).is_some_and(|programs| {
                !complete_row_plan(programs, s) && decoded_prefix_matches(programs, s, &plan)
            }) {
                incomplete_cases.push(case.model.clone());
                continue;
            }
            // RKNN physically pads dense rows to an 8-pixel fp16 pitch,
            // while ConvPlan's hardware-tested host ABI is compact NHWC.
            // CBUF charge is still comparable, but exact safe boundaries are
            // not until Shape carries an explicit physical input stride.
            if s.cin <= 4 && !s.width.is_multiple_of(8) {
                layout_cases.push(case.model.clone());
                continue;
            }
            print_difference(&case, &plan, &by_plan, &normalized_by_plan);
            eprintln!(
                "{}: no complete vendor row plan exactly matches ConvPlan; raw/normalized plans are {:?}",
                case.model,
                by_plan
                    .iter()
                    .map(|(index, programs)| {
                        (*index, (programs.len(), normalized_by_plan[index].len()))
                    })
                    .collect::<Vec<_>>()
            );
            divergences.push(case.model.clone());
            continue;
        }

        // A spatial match against ConvPlan only checks the row-tile
        // representative that `normalize_row_tiles` kept. ConvPlan does not
        // model output-channel grouping yet, so this is the only place the
        // repeats it discarded get checked at all: every row tile in the
        // winning plan must repeat the same output-channel groups, and
        // those groups must sum to Cout with the right destination stride.
        let expected_kernels = shape.programmed_kernels();
        let group_checked = matching
            .iter()
            .filter_map(|&(index, _)| {
                let raw = &by_plan[index];
                output_channel_groups(raw, s, expected_kernels, ELEM_BYTES)
                    .ok()
                    .map(|groups| (*index, groups))
            })
            .collect::<Vec<_>>();
        let Some((_, groups)) = group_checked.first() else {
            let (index, _) = matching[0];
            let error = output_channel_groups(&by_plan[index], s, expected_kernels, ELEM_BYTES)
                .unwrap_err();
            eprintln!(
                "{}: plan {index} matches ConvPlan spatially, but its output-channel groups do not check out: {error:?}",
                case.model,
            );
            divergences.push(case.model.clone());
            continue;
        };
        matching_cases += 1;
        if groups.len() > 1 {
            multi_group_cases += 1;
        }

        if case.model == "conv-w32-h400-k1-s1-ci1-co1" {
            let (_, programs) = matching[0];
            assert_eq!(plan.data_banks(), 4);
            assert_eq!(plan.weight_banks(), 8);
            assert_eq!(programs[0].cbuf_data_banks, 4);
            assert_eq!(programs[0].cbuf_weight_banks, 8);
        }
    }
    assert!(
        matching_cases > 0,
        "no fixture had a comparable vendor plan"
    );
    eprintln!(
        "fixture coverage: {matching_cases} exact ({multi_group_cases} with validated multi-output-channel-group plans), {} matched decoded plan-0 prefixes, {} dense physical-pitch comparisons deferred",
        incomplete_cases.len(),
        layout_cases.len(),
    );
    if !incomplete_cases.is_empty() {
        eprintln!("  decoded plan-0 prefixes: {}", incomplete_cases.join(", "));
    }
    if !layout_cases.is_empty() {
        eprintln!("  deferred physical pitches: {}", layout_cases.join(", "));
    }
    assert!(
        divergences.is_empty(),
        "{} complete vendor fixture(s) diverged from ConvPlan: {}",
        divergences.len(),
        divergences.join(", ")
    );
}
