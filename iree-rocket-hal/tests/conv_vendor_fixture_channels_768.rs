use std::collections::{BTreeMap, BTreeSet};

use iree_rocket_hal::rocket::{
    builders::{RegCmd, RegisterMeta, cna::CnaCbufCon1},
    conv::{
        ConvPlan, Kernels, MAX_INPUT_CHANNELS, MAX_INT8_INPUT_CHANNELS, MAX_OUTPUT_CHANNELS,
        Multiplier, Precision, Quantization, Shape,
    },
};
use serde::Deserialize;

const FP16_FIXTURES: &str = include_str!("fixtures/conv_vendor_fixtures_channels_768.json");
const INT8_FIXTURES: &str = include_str!("fixtures/conv_vendor_fixtures_channels_768_i8.json");

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

#[derive(Clone, Copy)]
enum FixturePrecision {
    Fp16,
    Int8,
}

impl FixturePrecision {
    fn name(self) -> &'static str {
        match self {
            Self::Fp16 => "fp16",
            Self::Int8 => "int8",
        }
    }

    fn precision(self) -> Precision {
        match self {
            Self::Fp16 => Precision::Fp16,
            Self::Int8 => Precision::Int8(Quantization {
                input_zero_point: 0,
                output_zero_point: -3,
                multiplier: Multiplier {
                    scale: 19636,
                    shift: 24,
                },
            }),
        }
    }

    fn max_input_channels(self) -> u32 {
        match self {
            Self::Fp16 => MAX_INPUT_CHANNELS,
            Self::Int8 => MAX_INT8_INPUT_CHANNELS,
        }
    }
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

/// Collapses repeated row geometry emitted once per output-kernel group.
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

/// Continuation tasks may encode CBUF_CON0 as 0/0 to mean reuse. Recover the
/// one actual allocation programmed by another task in the same vendor plan.
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
    if programs.len() != plan.tiles().len() {
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

fn print_grouped_bank_differences(
    precision: FixturePrecision,
    differences: &BTreeMap<(u32, u32, u32, u32, u32), Vec<u32>>,
) {
    if differences.is_empty() {
        return;
    }
    eprintln!("\n{} grouped CBUF divergences:", precision.name());
    for ((cin, plan_data, plan_weight, vendor_data, vendor_weight), couts) in differences {
        eprintln!(
            "  Cin={cin:3}: ConvPlan={plan_data}/{plan_weight}, vendor={vendor_data}/{vendor_weight}, Cout={couts:?}"
        );
    }
}

fn run_channel_grid(fixtures: &str, precision: FixturePrecision) {
    let fixtures: FixtureFile = serde_json::from_str(fixtures).expect("valid fixture JSON");
    assert_eq!(fixtures.schema, 1);
    assert_eq!(fixtures.cases.len(), 144, "expected the 12x12 channel grid");

    let mut supported = 0;
    let mut exploratory = Vec::new();
    let mut bank_matches = 0;
    let mut exact_matches = 0;
    let mut incomplete_row_plans = Vec::new();
    let mut tile_differences = Vec::new();
    let mut printed_row_routes = BTreeSet::new();
    let mut missing_plan_zero_split = Vec::new();
    let mut grouped_bank_differences = BTreeMap::<(u32, u32, u32, u32, u32), Vec<u32>>::new();

    for case in fixtures.cases {
        let s = &case.shape;
        if s.cin > precision.max_input_channels() || s.cout > MAX_OUTPUT_CHANNELS {
            exploratory.push((s.cin, s.cout));
            continue;
        }
        supported += 1;

        let shape = Shape::with_precision(
            s.width,
            s.height,
            s.stride,
            s.cin,
            s.cout,
            precision.precision(),
        )
        .with_padding([s.pad_h as usize, s.pad_w as usize]);
        let kernels: Kernels = [s.kernel_h as usize, s.kernel_w as usize];
        let plan = ConvPlan::new(shape, kernels);

        let mut by_plan = BTreeMap::<u32, Vec<&Program>>::new();
        for program in &case.vendor_programs {
            by_plan.entry(program.plan_index).or_default().push(program);
        }

        let Some(plan_zero) = by_plan.get(&0) else {
            missing_plan_zero_split.push(case.model.clone());
            continue;
        };
        let Some((vendor_data, vendor_weight)) = plan_bank_split(plan_zero) else {
            missing_plan_zero_split.push(case.model.clone());
            continue;
        };

        let generated_split = (plan.data_banks(), plan.weight_banks());
        let banks_match = generated_split == (vendor_data, vendor_weight);
        if banks_match {
            bank_matches += 1;
        } else {
            grouped_bank_differences
                .entry((
                    s.cin,
                    generated_split.0,
                    generated_split.1,
                    vendor_data,
                    vendor_weight,
                ))
                .or_default()
                .push(s.cout);
        }

        let normalized_by_plan = by_plan
            .iter()
            .map(|(plan_index, programs)| (*plan_index, normalize_row_tiles(programs)))
            .collect::<Vec<_>>();
        let complete = normalized_by_plan
            .iter()
            .filter(|(_, programs)| complete_row_plan(programs, s))
            .collect::<Vec<_>>();
        if complete.is_empty() {
            incomplete_row_plans.push(case.model.clone());
        } else if !banks_match {
            // The bank mismatch is already grouped above. Do not count the
            // same case again as a row-tiling mismatch.
        } else if complete.iter().any(|(_, programs)| {
            plan_bank_split(programs) == Some(generated_split)
                && exact_plan_match(programs, &case, &plan)
        }) {
            exact_matches += 1;
        } else {
            // Cout does not affect row tiling once the bank split is fixed.
            // Print one representative per precision/Cin/split so a large
            // channel grid remains readable while retaining every field the
            // exact matcher compares.
            if printed_row_routes.insert((s.cin, generated_split.0, generated_split.1)) {
                let generated_programs = plan.programs();
                let generated_route = plan
                    .tiles()
                    .iter()
                    .zip(&generated_programs)
                    .map(|(tile, commands)| {
                        (
                            tile.rows.out_first,
                            tile.rows.out_rows,
                            tile.rows.in_first,
                            tile.rows.in_rows,
                            tile.columns.out_first,
                            tile.columns.out_cols,
                            tile.columns.in_first,
                            tile.columns.in_cols,
                            register_value::<CnaCbufCon1>(commands),
                        )
                    })
                    .collect::<Vec<_>>();
                eprintln!(
                    "\n{} representative row-route divergence: {} generated={generated_route:?}",
                    precision.name(),
                    case.model,
                );
                for (plan_index, programs) in &complete {
                    let mut vendor = programs.to_vec();
                    vendor.sort_by_key(|program| program.out_offset);
                    let vendor_route = vendor
                        .iter()
                        .map(|program| {
                            (
                                program.out_offset,
                                program.out_rows,
                                program.in_offset,
                                program.in_rows,
                                program.out_width,
                                program.in_width,
                                program.cbuf_data_entries,
                            )
                        })
                        .collect::<Vec<_>>();
                    eprintln!("  vendor plan {plan_index}: {vendor_route:?}");
                }
            }
            tile_differences.push(case.model.clone());
        }
    }

    for couts in grouped_bank_differences.values_mut() {
        couts.sort_unstable();
    }
    print_grouped_bank_differences(precision, &grouped_bank_differences);
    if !tile_differences.is_empty() {
        eprintln!(
            "\n{} row-plan divergences after bank splits match: {}",
            precision.name(),
            tile_differences.join(", "),
        );
    }
    let exploratory_cin = exploratory
        .iter()
        .filter_map(|(cin, _)| (*cin > precision.max_input_channels()).then_some(*cin))
        .collect::<BTreeSet<_>>();
    let exploratory_cout = exploratory
        .iter()
        .filter_map(|(_, cout)| (*cout > MAX_OUTPUT_CHANNELS).then_some(*cout))
        .collect::<BTreeSet<_>>();
    eprintln!(
        "\n{} expanded fixture coverage: supported={supported}, exploratory={}, bank_matches={bank_matches}, exact_row_plans={exact_matches}, incomplete_row_plans={}, tile_differences={}, missing_plan_zero_split={}",
        precision.name(),
        exploratory.len(),
        incomplete_row_plans.len(),
        tile_differences.len(),
        missing_plan_zero_split.len(),
    );
    eprintln!(
        "  exploratory values: Cin={exploratory_cin:?}, Cout={exploratory_cout:?} (any pair containing one is outside current ConvPlan limits)"
    );

    assert_eq!(supported, 64);
    assert_eq!(exploratory.len(), 80);
    assert!(
        missing_plan_zero_split.is_empty(),
        "vendor plan 0 had no unique nonzero CBUF split: {missing_plan_zero_split:?}"
    );
    assert!(
        grouped_bank_differences.is_empty(),
        "{} supported channel cases have grouped ConvPlan/vendor CBUF divergences above",
        supported - bank_matches,
    );
    assert!(
        tile_differences.is_empty(),
        "{} complete vendor row plans diverged from ConvPlan: {}",
        tile_differences.len(),
        tile_differences.join(", "),
    );
}

#[test]
fn expanded_fp16_channel_grid_matches_vendor_plans() {
    run_channel_grid(FP16_FIXTURES, FixturePrecision::Fp16);
}

#[test]
fn expanded_int8_channel_grid_matches_vendor_plans() {
    run_channel_grid(INT8_FIXTURES, FixturePrecision::Int8);
}
