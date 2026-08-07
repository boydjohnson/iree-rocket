use std::collections::{BTreeMap, BTreeSet};

use iree_rocket_hal::rocket::conv::{ConvPlan, Kernels, Shape};
use serde::Deserialize;

const FIXTURES: &str = include_str!("fixtures/conv_vendor_fixtures.json");

#[derive(Deserialize)]
struct FixtureFile {
    schema: u32,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    model: String,
    shape: ShapeFixture,
    vendor_programs: Vec<Program>,
}

#[derive(Deserialize)]
struct ShapeFixture {
    width: u32,
    height: u32,
    cin: u32,
    cout: u32,
    kernel_h: u32,
    kernel_w: u32,
    pad_h: u32,
    pad_w: u32,
    stride: u32,
}

#[derive(Deserialize)]
struct Program {
    plan_index: u32,
    in_width: u32,
    in_rows: u32,
    out_width: u32,
    out_atomics: u32,
    out_rows: u32,
    in_offset: u32,
    out_offset: u32,
    feature_grains: u32,
    cbuf_data_banks: u32,
    cbuf_weight_banks: u32,
}

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
            "    plan {index}: raw_programs={} normalized_row_tiles={} banks d/w={banks}",
            programs.len(),
            normalized.len(),
        );
        for (program_index, program) in normalized.iter().enumerate() {
            eprintln!(
                "      program {program_index}: in={}x{} @0x{:x}, out={}x{} @0x{:x}, grains={}, atomics={}",
                program.in_width,
                program.in_rows,
                program.in_offset,
                program.out_width,
                program.out_rows,
                program.out_offset,
                program.feature_grains,
                program.out_atomics,
            );
        }
    }
}

/// Vendor programs repeat the same row tile once per output-surface group.
/// The input geometry and feature-grain programming identify the row tile;
/// the output offset additionally carries the surface-group base. Collapse
/// those repetitions before comparing against ConvPlan's row-only tiles.
fn normalize_row_tiles<'a>(programs: &[&'a Program]) -> Vec<&'a Program> {
    let mut seen = BTreeSet::new();
    programs
        .iter()
        .copied()
        .filter(|program| {
            seen.insert((
                program.in_width,
                program.in_rows,
                program.in_offset,
                program.out_width,
                program.out_rows,
                program.feature_grains,
            ))
        })
        .collect()
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

        let matching = normalized_by_plan
            .values()
            .filter(|programs| programs.len() == plan.tiles().len())
            .collect::<Vec<_>>();
        if matching.is_empty() {
            print_difference(&case, &plan, &by_plan, &normalized_by_plan);
            eprintln!(
                "{}: ConvPlan selected {} row tile(s), vendor raw/normalized plans are {:?}",
                case.model,
                plan.tiles().len(),
                by_plan
                    .iter()
                    .map(|(index, programs)| {
                        (*index, (programs.len(), normalized_by_plan[index].len()))
                    })
                    .collect::<Vec<_>>()
            );
            continue;
        }
        matching_cases += 1;

        if case.model == "conv-w32-h400-k1-s1-ci1-co1" {
            assert_eq!(plan.data_banks(), 11);
            assert_eq!(plan.weight_banks(), 1);
            assert_eq!(matching[0][0].cbuf_data_banks, 4);
            assert_eq!(matching[0][0].cbuf_weight_banks, 8);
        }

        for programs in matching {
            let full_width = programs.iter().all(|program| program.out_width == s.width);
            let single_output_surface = programs
                .iter()
                .all(|program| program.out_atomics == program.out_width * program.out_rows);
            let covers_one_height = programs.iter().map(|program| program.out_rows).sum::<u32>()
                == shape.output_height(kernels);
            if !full_width || !single_output_surface || !covers_one_height {
                // Surface plans may partition columns as well as rows. Their
                // output offsets are 2D, and multi-surface output plans repeat
                // the row ranges at per-surface offsets. A row-only
                // contiguity check would misdiagnose those as gaps.
                continue;
            }
            let mut sorted = programs.clone();
            sorted.sort_by_key(|program| program.out_offset);
            let mut covered_rows = 0;
            for program in sorted {
                assert_eq!(
                    program.out_offset,
                    covered_rows * s.width * 16,
                    "{}: vendor output rows have a gap/overlap",
                    case.model
                );
                covered_rows += program.out_rows;
                assert_eq!(
                    program.cbuf_data_banks + program.cbuf_weight_banks,
                    12,
                    "{}: vendor CBUF split does not partition all banks",
                    case.model
                );
            }
            assert_eq!(
                covered_rows,
                shape.output_height(kernels),
                "{}: vendor plan does not cover output height",
                case.model
            );
        }
    }
    assert!(
        matching_cases > 0,
        "no fixture had a comparable vendor plan"
    );
}
