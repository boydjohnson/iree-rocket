#[path = "support/vendor_fixture.rs"]
mod vendor_fixture;

use std::collections::{BTreeMap, BTreeSet};

use iree_rocket_hal::rocket::conv::{
    ConvPlan, Kernels, MAX_INPUT_CHANNELS, MAX_INT8_INPUT_CHANNELS, MAX_OUTPUT_CHANNELS,
    Multiplier, Precision, Quantization, Shape,
};
use vendor_fixture::{
    FixtureFile, Program, complete_row_plan, exact_plan_match, normalize_row_tiles,
    output_channel_groups, register_value,
};

use iree_rocket_hal::rocket::builders::cna::CnaCbufCon1;

const FP16_FIXTURES: &str = include_str!("fixtures/conv_vendor_fixtures_channels_768.json");
const INT8_FIXTURES: &str = include_str!("fixtures/conv_vendor_fixtures_channels_768_i8.json");

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
                weight_zero_point: 0,
                input_scale: 1.0,
                weights_scale: 1.0,
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

    fn elem_bytes(self) -> u32 {
        match self {
            Self::Fp16 => 2,
            Self::Int8 => 1,
        }
    }
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
    let mut multi_group_matches = 0;
    let mut incomplete_row_plans = Vec::new();
    let mut tile_differences = Vec::new();
    let mut group_differences = Vec::new();
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
        if std::env::var("DUMP_SPLITS").is_ok() {
            println!(
                "SPLIT cin={} cout={} vendor={}/{} plan={}/{}",
                s.cin, s.cout, vendor_data, vendor_weight, generated_split.0, generated_split.1
            );
        }
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
        } else if let Some(&(plan_index, _)) = complete.iter().find(|(plan_index, programs)| {
            plan_bank_split(by_plan[plan_index].as_slice()) == Some(generated_split)
                && exact_plan_match(programs, s, &plan)
        }) {
            // The row-tile representative matches ConvPlan spatially. That
            // representative is only one program out of however many
            // `normalize_row_tiles` collapsed away -- one per
            // output-channel group. ConvPlan does not model that axis yet,
            // so this is the only place the repeats get checked at all:
            // every row tile in the winning plan must repeat the same
            // output-channel groups, summing to the programmed kernel
            // count, with the right destination stride between groups.
            let raw = &by_plan[&plan_index];
            match output_channel_groups(raw, s, shape.programmed_kernels(), precision.elem_bytes())
            {
                Ok(groups) => {
                    exact_matches += 1;
                    if groups.len() > 1 {
                        multi_group_matches += 1;
                    }
                }
                Err(error) => {
                    eprintln!(
                        "{}: plan {plan_index} matches ConvPlan spatially and on CBUF banks, but its output-channel groups do not check out: {error:?}",
                        case.model,
                    );
                    group_differences.push(case.model.clone());
                }
            }
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
    if !group_differences.is_empty() {
        eprintln!(
            "\n{} output-channel-group divergences after row plans match: {}",
            precision.name(),
            group_differences.join(", "),
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
        "\n{} expanded fixture coverage: supported={supported}, exploratory={}, bank_matches={bank_matches}, exact_row_plans={exact_matches} ({multi_group_matches} with validated multi-output-channel-group plans), incomplete_row_plans={}, tile_differences={}, group_differences={}, missing_plan_zero_split={}",
        precision.name(),
        exploratory.len(),
        incomplete_row_plans.len(),
        tile_differences.len(),
        group_differences.len(),
        missing_plan_zero_split.len(),
    );
    eprintln!(
        "  exploratory values: Cin={exploratory_cin:?}, Cout={exploratory_cout:?} (any pair containing one is outside current ConvPlan limits)"
    );

    // Per precision, because the two no longer share a channel ceiling.
    //
    // **int8 is 144/0**: `MAX_INT8_INPUT_CHANNELS` went 512 -> 1344 and
    // `MAX_INT8_OUTPUT_CHANNELS` was split out at 1792 (2026-09-03), so the
    // whole Cin/Cout 64..768 corpus is in range. **fp16 stays 96/48**: its
    // ceilings are untouched, because the hardware evidence behind the raise
    // is int8 only.
    //
    // The old comment here predicted "raising MAX_INPUT_CHANNELS to 768 makes
    // these 144/0 and the bank comparison then fails on ~10 cases". It fails
    // on exactly **one** Cin, listed below: that prediction predates the
    // group-division fix (`MAX_UNDIVIDED_WEIGHT_BANKS`), after which ConvPlan
    // reproduces the vendor at 576, 640, 704-at-large-Cout and 768.
    let (expected_supported, expected_exploratory) = match precision {
        FixturePrecision::Fp16 => (96, 48),
        FixturePrecision::Int8 => (144, 0),
    };
    assert_eq!(supported, expected_supported);
    assert_eq!(exploratory.len(), expected_exploratory);
    assert!(
        missing_plan_zero_split.is_empty(),
        "vendor plan 0 had no unique nonzero CBUF split: {missing_plan_zero_split:?}"
    );

    // One documented divergence, and it is **hardware-validated as correct**.
    //
    // At Cin 704, k=3, 28x28, ConvPlan splits 4/8 where the vendor splits 5/7,
    // and only at small Cout (64 and 128; at Cout >= 192 the two agree). The
    // difference grants the *weights* one bank more than the vendor rather
    // than fewer, which is the safe direction -- and it is not left as
    // reasoning: `accumulator_size_e_probe` ran 28x28 Cin 704 at Cout 64, 128
    // and 256 on RK3588 and every one is 0 mismatches against the oracle.
    //
    // Kept as an allowlist rather than a relaxed assertion so a *new*
    // divergence still fails, and so this one cannot quietly widen: the exact
    // splits and the exact Cout list are pinned.
    let known_hardware_validated: &[((u32, u32, u32, u32, u32), &[u32])] =
        &[((704, 4, 8, 5, 7), &[64, 128])];
    let unexpected: Vec<_> = grouped_bank_differences
        .iter()
        .filter(|(key, couts)| {
            !known_hardware_validated
                .iter()
                .any(|(k, c)| k == *key && c == &couts.as_slice())
        })
        .collect();
    assert!(
        unexpected.is_empty(),
        "{} supported channel cases have undocumented ConvPlan/vendor CBUF divergences: {unexpected:?}",
        supported - bank_matches,
    );
    assert!(
        tile_differences.is_empty(),
        "{} complete vendor row plans diverged from ConvPlan: {}",
        tile_differences.len(),
        tile_differences.join(", "),
    );
    assert!(
        group_differences.is_empty(),
        "{} complete vendor row plans had inconsistent output-channel groups: {}",
        group_differences.len(),
        group_differences.join(", "),
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
