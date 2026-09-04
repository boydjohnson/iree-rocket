//! ConvPlan vs the vendor above the old 512/768 channel ceilings, and the
//! first committed depthwise corpus.
//!
//! Built with `build_vendor_fixtures.py` (spike repo) on 2026-09-03 to justify
//! raising `MAX_INT8_INPUT_CHANNELS` 512 -> 1344 and splitting
//! `MAX_INT8_OUTPUT_CHANNELS` out at 1792. `conv_vendor_fixture_channels_768`
//! stops at Cin/Cout 768 and is k=3 / 28x28 / stride 1 only, so nothing
//! covered the range MobileNetV2 actually needs.
//!
//! **What is here.** Dense: a Cin sweep to 1792 at Cout 64, 448 and 1792; a
//! coarse Cin x Cout grid to 1792; and MobileNetV2's own widest dense 1x1
//! convolutions at their real extents (14x14 and 7x7, k=1). Depthwise: C
//! 64..1344 at extents 7, 14 and 28, k=3.
//!
//! **What this checks and what it does not.** It compares the CBUF bank split
//! only -- the quantity the high-channel divergence was ever about -- not row
//! plans or output-channel groups; `conv_vendor_fixture_channels_768` does
//! those on its own range. The corpus is fp16-generated, and the split model
//! under test (`streamed_weight_bank_preference`) is precision-independent,
//! but the int8 *data* demand is not covered here; the int8 evidence for the
//! raise is hardware, recorded on `MAX_INT8_INPUT_CHANNELS`.

#[path = "support/vendor_fixture.rs"]
mod vendor_fixture;

use iree_rocket_hal::rocket::conv::{ConvPlan, Precision, Shape};
use vendor_fixture::FixtureFile;

const WIDE: &str = include_str!("fixtures/conv_vendor_fixtures_wide.json");
const DEPTHWISE: &str = include_str!("fixtures/conv_vendor_fixtures_depthwise.json");

/// The vendor's split for a case: the first program, by plan index, that
/// carries a nonzero one. Plan 0 does not always populate it.
fn vendor_split(case: &vendor_fixture::Case) -> Option<(u32, u32)> {
    let mut programs: Vec<_> = case.vendor_programs.iter().collect();
    programs.sort_by_key(|p| p.plan_index);
    programs
        .iter()
        .find(|p| p.cbuf_data_banks != 0 || p.cbuf_weight_banks != 0)
        .map(|p| (p.cbuf_data_banks, p.cbuf_weight_banks))
}

struct Scores {
    agree: usize,
    refused: usize,
    differences: Vec<(u32, u32, u32, usize, (u32, u32), (u32, u32))>,
}

fn score(fixtures: &str, depthwise: bool) -> Scores {
    // The corpus deliberately reaches past the shipped channel ceilings --
    // that is the point of it -- so shape construction runs with the
    // characterization hatch. The ceilings bound the *channel padding* rules;
    // what is under test here is the CBUF split.
    unsafe { std::env::set_var("ROCKET_ALLOW_UNBACKED_CHANNELS", "1") };
    // Refusals are an expected, counted outcome here (k=3 above the
    // coefficient working set), and each one is a caught panic. Silence the
    // default hook for the duration so the report is readable.
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let file: FixtureFile = serde_json::from_str(fixtures).expect("fixture json");
    let mut scores = Scores {
        agree: 0,
        refused: 0,
        differences: Vec::new(),
    };
    for case in &file.cases {
        let Some(vendor) = vendor_split(case) else {
            continue;
        };
        let s = &case.shape;
        let kernels = [s.kernel_h as usize, s.kernel_w as usize];
        let planned = std::panic::catch_unwind(|| {
            let mut shape =
                Shape::with_precision(s.width, s.height, 1, s.cin, s.cout, Precision::Fp16);
            if depthwise {
                shape = shape.with_depthwise();
            }
            let plan = ConvPlan::new(shape, kernels);
            (plan.data_banks(), plan.weight_banks())
        });
        match planned {
            Ok(split) if split == vendor => scores.agree += 1,
            Ok(split) => scores
                .differences
                .push((s.cin, s.cout, s.width, kernels[0], split, vendor)),
            Err(_) => scores.refused += 1,
        }
    }
    std::panic::set_hook(previous_hook);
    scores
}

/// Depthwise agrees with the vendor everywhere, at every extent and channel
/// count the corpus covers, and refuses nothing.
///
/// This is the gate on `Shape::streamed_contraction_channels`. Before it, the
/// streamed coefficient working set used the *dense* product
/// `kh * kw * Cin * 64`, which scales with C: depthwise C=1344 at k=3 asked for
/// 13 of the eleven grantable CBUF banks and was refused outright, which is
/// what kept MobileNetV2's 528/816/1344 depthwise stages on the CPU. A
/// depthwise output channel accumulates over one input channel, so the
/// contraction depth is 1 and the working set does not scale with C at all.
///
/// The fix is additive: a 128-case sweep (k=3 and k=5, extents 7/14/28/56,
/// C 32..512) produced byte-identical plans before and after.
#[test]
fn depthwise_channel_grid_matches_vendor_plans() {
    let scores = score(DEPTHWISE, true);
    println!(
        "depthwise: agree={} refused={} differ={}",
        scores.agree,
        scores.refused,
        scores.differences.len()
    );
    for (cin, cout, extent, k, plan, vendor) in &scores.differences {
        println!(
            "  {extent}^2 C {cin}/{cout} k{k}: ConvPlan {}/{} vendor {}/{}",
            plan.0, plan.1, vendor.0, vendor.1
        );
    }
    assert_eq!(
        scores.differences.len(),
        0,
        "depthwise CBUF split divergences"
    );
    assert_eq!(scores.refused, 0, "depthwise cases ConvPlan refused");
    assert_eq!(scores.agree, 63);
}

/// Dense above the old ceilings: two divergences, both hardware-validated as
/// correct, and a refusal band that the transform spec's own caps keep off the
/// compiled path.
///
/// * **28x28 Cin 704 Cout 64 k3** -- ConvPlan 4/8, vendor 5/7. The residual
///   `conv_vendor_fixture_channels_768` also carries. Board: 0 mismatches at
///   Cout 64, 128 and 256.
/// * **14x14 Cin 816 Cout 136 k1** -- ConvPlan 5/7, vendor 10/2, and one of
///   MobileNetV2's own convolutions. Board: 0 mismatches.
///
/// Both give the *weights* at least as many banks as the vendor, which is the
/// safe direction; a split that starves the weights is what produces wrong
/// values. That is the reasoning, but the assertions rest on the board runs.
///
/// The refusals are k=3 at Cin 1216, where the coefficient working set
/// first exceeds the eleven grantable banks. `ConvPlan` refuses rather than
/// mis-planning, and `@match_dynamic_conv2d_3x3_int8` caps Cin at 1152 so the
/// refusal is unreachable from a compiled model. The corpus keeps the edge
/// itself and drops k=3 Cin > 1216: that band only ever produces refusals and
/// was three quarters of the file's bytes, and keeps one Cout at the edge
/// rather than three (the Cin 1216 / Cout 1792 case alone carries 5378 vendor
/// programs and 2 MB).
#[test]
fn wide_dense_channel_grid_matches_vendor_plans() {
    let scores = score(WIDE, false);
    println!(
        "dense wide: agree={} refused={} differ={}",
        scores.agree,
        scores.refused,
        scores.differences.len()
    );
    let mut seen: Vec<(u32, u32, usize, (u32, u32), (u32, u32))> = scores
        .differences
        .iter()
        .map(|(cin, cout, _, k, plan, vendor)| (*cin, *cout, *k, *plan, *vendor))
        .collect();
    seen.sort_unstable();
    assert_eq!(
        seen,
        vec![(704, 64, 3, (4, 8), (5, 7)), (816, 136, 1, (5, 7), (10, 2)),],
        "undocumented dense CBUF split divergences above the old ceilings"
    );
    assert_eq!(scores.agree, 83);
    assert_eq!(scores.refused, 1);
}
