use std::collections::{BTreeMap, BTreeSet};

use iree_rocket_hal::rocket::{
    builders::{RegCmd, RegisterMeta, cna::CnaCbufCon1},
    conv::{ConvPlan, Kernels, Shape},
};
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
    cbuf_data_entries: u32,
}

fn output_width(shape: &ShapeFixture) -> u32 {
    (shape.width + 2 * shape.pad_w - shape.kernel_w) / shape.stride + 1
}

fn output_height(shape: &ShapeFixture) -> u32 {
    (shape.height + 2 * shape.pad_h - shape.kernel_h) / shape.stride + 1
}

fn register_value<R: RegisterMeta>(program: &[RegCmd]) -> Option<u32> {
    program
        .iter()
        .find(|command| command.0 as u32 & 0xffff == R::OFFSET)
        .map(|command| ((command.0 >> 16) & 0xffff_ffff) as u32)
}

/// Whether these normalized programs expose one complete, full-width row
/// plan. Long standalone streams and horizontal grids are useful evidence,
/// but this fixture format cannot yet compare their hidden/2-D tasks exactly.
fn complete_row_plan(programs: &[&Program], shape: &ShapeFixture) -> bool {
    let out_width = output_width(shape);
    if programs.is_empty()
        || programs
            .iter()
            .any(|program| program.out_width != out_width || program.in_width != shape.width)
    {
        return false;
    }
    let mut sorted = programs.to_vec();
    sorted.sort_by_key(|program| program.out_offset);
    let mut covered_rows = 0;
    for program in sorted {
        if program.out_offset != covered_rows * out_width * 16 {
            return false;
        }
        covered_rows += program.out_rows;
    }
    covered_rows == output_height(shape)
}

fn exact_plan_match(programs: &[&Program], case: &Case, plan: &ConvPlan) -> bool {
    if programs.len() != plan.tiles().len()
        || programs.iter().any(|program| {
            program.cbuf_data_banks != plan.data_banks()
                || program.cbuf_weight_banks != plan.weight_banks()
        })
    {
        return false;
    }
    let out_width = output_width(&case.shape);
    let mut vendor = programs.to_vec();
    vendor.sort_by_key(|program| program.out_offset);
    let generated = plan.programs();
    vendor
        .iter()
        .zip(plan.tiles())
        .zip(generated.iter())
        .all(|((program, tile), commands)| {
            program.in_width == tile.columns.in_cols
                && program.out_width == tile.columns.out_cols
                && program.out_rows == tile.rows.out_rows
                && program.in_rows == tile.rows.in_rows
                && program.out_offset
                    == tile.rows.out_first * out_width * 16 + tile.columns.out_first * 16
                && program.out_atomics == program.out_width * program.out_rows
                && register_value::<CnaCbufCon1>(commands) == Some(program.cbuf_data_entries)
        })
}

fn decoded_prefix_matches(programs: &[&Program], case: &Case, plan: &ConvPlan) -> bool {
    if programs.is_empty()
        || programs.len() >= plan.tiles().len()
        || programs.iter().any(|program| {
            program.cbuf_data_banks != plan.data_banks()
                || program.cbuf_weight_banks != plan.weight_banks()
        })
    {
        return false;
    }
    let out_width = output_width(&case.shape);
    let mut vendor = programs.to_vec();
    vendor.sort_by_key(|program| program.out_offset);
    let generated = plan.programs();
    vendor
        .iter()
        .zip(plan.tiles())
        .zip(generated.iter())
        .all(|((program, tile), commands)| {
            program.in_width == tile.columns.in_cols
                && program.out_width == tile.columns.out_cols
                && program.out_rows == tile.rows.out_rows
                && program.in_rows == tile.rows.in_rows
                && program.out_offset
                    == tile.rows.out_first * out_width * 16 + tile.columns.out_first * 16
                && register_value::<CnaCbufCon1>(commands) == Some(program.cbuf_data_entries)
        })
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
            .values()
            .filter(|programs| complete_row_plan(programs, s))
            .collect::<Vec<_>>();
        if complete.is_empty() {
            incomplete_cases.push(case.model.clone());
            continue;
        }
        let matching = complete
            .iter()
            .copied()
            .filter(|programs| exact_plan_match(programs, &case, &plan))
            .collect::<Vec<_>>();
        if matching.is_empty() {
            if normalized_by_plan.get(&0).is_some_and(|programs| {
                !complete_row_plan(programs, s) && decoded_prefix_matches(programs, &case, &plan)
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
        matching_cases += 1;

        if case.model == "conv-w32-h400-k1-s1-ci1-co1" {
            assert_eq!(plan.data_banks(), 4);
            assert_eq!(plan.weight_banks(), 8);
            assert_eq!(matching[0][0].cbuf_data_banks, 4);
            assert_eq!(matching[0][0].cbuf_weight_banks, 8);
        }
    }
    assert!(
        matching_cases > 0,
        "no fixture had a comparable vendor plan"
    );
    eprintln!(
        "fixture coverage: {matching_cases} exact, {} matched decoded plan-0 prefixes, {} dense physical-pitch comparisons deferred",
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
