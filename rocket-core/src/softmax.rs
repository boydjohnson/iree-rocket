//! Softmax over the width axis: which stages the op decomposes into, and
//! which shapes the lowering will take.
//!
//! [SOFTMAX.md](../../../SOFTMAX.md) is the design and the evidence; this
//! module is its §7 as a plan. Everything here follows from a measurement
//! recorded there, and the shape refusals below are the places where the
//! measurement stops rather than where the hardware does.
//!
//! The decomposition, and why each stage is its own task:
//!
//! 1. **The row maximum**, a tree of PPU max-pool levels over the width
//!    axis. There is no convolution path for a maximum, and the pooling
//!    window is at most 8 wide, so wide rows take several levels (§2, S2).
//! 2. **The maximum into an operand plane**, on the host: a width-1,
//!    `rows`-wide vector negated, scaled by the `exp` task's own BN
//!    multiplier and written as `i32` per channel. Microseconds, like the
//!    reciprocal (§4, S5b).
//! 3. **`exp`**, the DPU LUT, int8 in and the table's Q15 integer out as
//!    `int16`, **with the per-row maximum subtracted inside it** from that
//!    plane. There is no separate subtract stage: §7 always put the
//!    subtraction here, and S5b measured it exact. Its tail needs the tuned
//!    underflow slope or a wide softmax is corrupted outright (§3.3, §3.4).
//! 5. **Two bridge tasks**, because an integer input makes the DPU's
//!    datapath integer: the first reads `int16` to fp16 exactly and cannot
//!    scale, the second applies the `2^-15` that turns a Q15 count into a
//!    probability (§4.2).
//! 6. **The sum**, a tree of depthwise all-ones windows. The window is 3x3
//!    at stride 3 because the planner refuses every non-square and even
//!    square kernel at a stride above 1, and an fp16 average pool cannot
//!    reduce one axis at all (§2.1, §2.2, S2).
//! 7. **The reciprocal, on the host**: `M` f32 divides, 0.004 ms for 64
//!    rows, against a 200:1 range no int8 LUT input can take (§4, S4).
//! 8. **Normalise**, one per-channel multiply on the BN MUL plane.
//!
//! This module names the stages and their extents. Column tiling stays with
//! the planners that own it -- `PoolingPlan` for a max level, `ConvPlan` for
//! a sum level -- exactly as [`crate::fc`] delegates to [`crate::conv`].

use crate::error::{PlanError, PlanErrorCode, refuse_unless};

/// The `exp` table's domain edge in real units, `16384 / EXP_LUT_BN_SCALE_K`.
/// Measured exactly (§3.3).
pub const EXP_DOMAIN_EDGE: f32 = 5.0;

/// The only `exp` input scale with an end-to-end measurement behind it:
/// §3.4 chose it against 64 real ViT-B/16 attention rows and the board
/// returned 1.12% total variation from an exact softmax.
pub const MEASURED_EXP_INPUT_SCALE: f32 = 1.0 / 20.0;

/// Widest PPU pooling window the direct path accepts.
const MAX_POOL_KERNEL: u32 = 8;

/// The sum tree's window: the one strided square kernel the corpus backs
/// (§2.2).
pub const SUM_KERNEL: u32 = 3;

/// Rows whose per-channel planes have an end-to-end measurement behind
/// them. §4's end-to-end gate ran 64; S1 checked the plane itself at 32.
const MEASURED_MAX_ROWS: u32 = 64;

/// Keys measured through the whole reduction, by S2's max tree.
const MEASURED_MAX_KEYS: u32 = 512;

/// Logical softmax shape, in §1.1's layout.
///
/// The reduced axis is **width**, because every reduction this hardware has
/// runs over space and none over channels. For attention that means the
/// score matrix arrives transposed -- keys on width, queries on channels --
/// which costs nothing, since it is only a question of which operand of
/// `Q K^T` goes down the weight path (§1.1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shape {
    /// The reduced extent: keys, on the width axis.
    pub keys: u32,
    /// The independent extent: queries or rows, on the channel axis. Each
    /// gets its own maximum and its own denominator, and each is one lane
    /// of the per-channel operand planes.
    pub rows: u32,
    /// Multiplier applied before the `exp` table is indexed, which trades
    /// its resolution against its domain (§3.2). Only
    /// [`MEASURED_EXP_INPUT_SCALE`] is admitted.
    pub exp_input_scale: f32,
}

impl Shape {
    /// §3.4's shape for `keys` keys and `rows` rows.
    pub fn new(keys: u32, rows: u32) -> Shape {
        Shape {
            keys,
            rows,
            exp_input_scale: MEASURED_EXP_INPUT_SCALE,
        }
    }

    /// The most negative input the int8 code space reaches at this scale,
    /// which is what the underflow slope has to decay to zero over (§3.3).
    pub fn exp_reach(&self) -> f32 {
        -128.0 * self.exp_input_scale
    }
}

/// One task of the lowering, with the extents its builder needs.
///
/// A stage is one task because the hardware makes it one, not for tidiness:
/// the LUT never fuses into a producing convolution (`activation.rs`), a
/// maximum has no convolution path, and the two bridge tasks exist because
/// an integer input and a fractional multiply cannot share a datapath
/// (§4.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// One level of the row-maximum tree: a `kernel x 1` PPU max pool at
    /// stride `kernel`, with trailing padding made up by the window's own
    /// clamp-to-image (§2.1). `PoolingPlan` splits it into column tiles.
    MaxLevel {
        input_width: u32,
        kernel: u32,
        output_width: u32,
        pad_right: u32,
    },
    /// The width-1 maximum into the `exp` task's BN ALU plane, on the
    /// host: negate it (the plane *adds*), scale it by the task's own
    /// `lut_bn_mul_operand` (the operand is pre-scaled because the add
    /// lands after the multiply), and write it as `i32` per channel -- the
    /// element format the int8 datapath wants, where the fp16 datapath
    /// wants `f32` (S1, S5b).
    ///
    /// On the host because it is a width-1, `rows`-wide vector, which §4
    /// established is the host's side of the line. S2a's device gather
    /// remains an optimisation.
    HostMaxToPlane,
    /// The `exp` LUT: int8 scores in, the table's Q15 integer out as
    /// `int16`, with the per-row maximum subtracted from the plane above.
    Exp,
    /// `int16` to fp16, exactly, which is all an integer read can do
    /// (§4.2).
    BridgeRead,
    /// The `2^-15` that turns a Q15 count into a probability, on the fp16
    /// path's BN MUL plane.
    BridgeScale,
    /// One level of the sum tree: a depthwise all-ones `3x3` window at
    /// stride 3 on a height-1 cube, with `pad_left` chosen so the windows
    /// tile the row exactly (§2.2). `ConvPlan` plans it.
    SumLevel {
        input_width: u32,
        pad_left: u32,
        output_width: u32,
    },
    /// `1/d` on the host: `rows` f32 divides (§4).
    HostReciprocal,
    /// The normalising per-channel multiply on the BN MUL plane. Its
    /// operand is `1/d` in *real* units, never `1/d_q15`, which is a
    /// subnormal worth a couple of bits (§4.2).
    Normalize,
}

/// The ordered stages of one softmax, and what they need.
#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    shape: Shape,
    stages: Vec<Stage>,
}

impl Plan {
    /// Plans `shape`, or says why not.
    pub fn try_new(shape: Shape) -> Result<Plan, PlanError> {
        refuse_unless!(
            shape.keys > 0 && shape.rows > 0,
            PlanErrorCode::InvalidShape,
            "softmax extents must be nonzero; got {} keys and {} rows",
            shape.keys,
            shape.rows
        );
        refuse_unless!(
            shape.exp_input_scale == MEASURED_EXP_INPUT_SCALE,
            PlanErrorCode::UnvalidatedConfiguration,
            "the exp input scale is {}, and only {} has an end-to-end measurement behind it: \
             SOFTMAX.md section 3.4 chose it against real attention rows and the board \
             returned 1.12% total variation. The optimum is shallow, so a nearby scale is \
             very likely fine -- but nobody has checked, and the tail slope has to be \
             retuned with it",
            shape.exp_input_scale,
            MEASURED_EXP_INPUT_SCALE
        );
        refuse_unless!(
            shape.rows <= MEASURED_MAX_ROWS,
            PlanErrorCode::UnvalidatedConfiguration,
            "{} rows is past the {MEASURED_MAX_ROWS} measured end to end (SOFTMAX.md section \
             4). Each row is one lane of three per-channel operand planes, and the widest \
             plane any probe has driven is 64 lanes",
            shape.rows
        );
        refuse_unless!(
            shape.keys <= MEASURED_MAX_KEYS,
            PlanErrorCode::UnvalidatedConfiguration,
            "{} keys is past the {MEASURED_MAX_KEYS} the reduction is measured over \
             (SOFTMAX.md section 2.1's max tree and section 2.2's sum tree)",
            shape.keys
        );

        let mut stages = Vec::new();
        for level in max_levels(shape.keys) {
            stages.push(level);
        }
        stages.push(Stage::HostMaxToPlane);
        stages.push(Stage::Exp);
        stages.push(Stage::BridgeRead);
        stages.push(Stage::BridgeScale);
        for level in sum_levels(shape.keys) {
            stages.push(level);
        }
        stages.push(Stage::HostReciprocal);
        stages.push(Stage::Normalize);

        Ok(Plan { shape, stages })
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }

    pub fn stages(&self) -> &[Stage] {
        &self.stages
    }

    /// Device tasks, which is every stage but the host's reciprocal.
    pub fn device_tasks(&self) -> usize {
        self.stages
            .iter()
            .filter(|stage| !matches!(stage, Stage::HostReciprocal | Stage::HostMaxToPlane))
            .count()
    }

    /// Where the host has to intervene, splitting the job into
    /// submissions: two, each measured at 0.037 to 0.059 ms (§4). One
    /// before `exp` for the maximum's plane and one after the sum for the
    /// reciprocal, both width-1 vectors.
    pub fn host_boundaries(&self) -> usize {
        self.stages
            .iter()
            .filter(|stage| matches!(stage, Stage::HostReciprocal | Stage::HostMaxToPlane))
            .count()
    }
}

/// The max tree's levels, widest window first.
///
/// Widest-first is deliberate: every stage boundary costs a completion IRQ
/// (§5.3), so a shallow tree of wide windows beats a deep one of narrow
/// ones. A partial last window needs nothing from the caller, because a
/// padded max tap is clamp-to-image (§2.1).
fn max_levels(keys: u32) -> Vec<Stage> {
    let mut levels = Vec::new();
    let mut width = keys;
    while width > 1 {
        let kernel = MAX_POOL_KERNEL.min(width);
        let output_width = width.div_ceil(kernel);
        levels.push(Stage::MaxLevel {
            input_width: width,
            kernel,
            output_width,
            pad_right: output_width * kernel - width,
        });
        width = output_width;
    }
    levels
}

/// The sum tree's levels.
///
/// A convolution has leading padding registers and no trailing ones, and
/// its output extent is floor-mode, so a row whose length is not a multiple
/// of 3 drops its tail unless the *leading* pad makes up the difference.
/// `p = (3 - width % 3) % 3` tiles `[-p, width - 1]` exactly, and `p <= 2`
/// stays under the kernel (§2.2).
fn sum_levels(keys: u32) -> Vec<Stage> {
    let mut levels = Vec::new();
    let mut width = keys;
    while width > 1 {
        let pad_left = (SUM_KERNEL - width % SUM_KERNEL) % SUM_KERNEL;
        let output_width = (width + pad_left) / SUM_KERNEL;
        levels.push(Stage::SumLevel {
            input_width: width,
            pad_left,
            output_width,
        });
        width = output_width;
    }
    levels
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_stage_order_is_the_design_s_own() {
        let plan = Plan::try_new(Shape::new(197, 64)).unwrap();
        let tail: Vec<Stage> = plan
            .stages()
            .iter()
            .copied()
            .filter(|stage| !matches!(stage, Stage::MaxLevel { .. } | Stage::SumLevel { .. }))
            .collect();
        assert_eq!(
            tail,
            vec![
                Stage::HostMaxToPlane,
                Stage::Exp,
                Stage::BridgeRead,
                Stage::BridgeScale,
                Stage::HostReciprocal,
                Stage::Normalize,
            ]
        );
        // The sum levels follow the bridge, because they sum the
        // probabilities rather than the scores.
        let exp_at = plan
            .stages()
            .iter()
            .position(|stage| *stage == Stage::BridgeScale)
            .unwrap();
        let first_sum = plan
            .stages()
            .iter()
            .position(|stage| matches!(stage, Stage::SumLevel { .. }))
            .unwrap();
        assert!(first_sum > exp_at);
    }

    /// Both trees have to cover every key exactly once, and the two do it
    /// differently: the max tree pads at the *end* of a level and leans on
    /// clamp-to-image, while the sum tree pads at the *start* because a
    /// convolution has no trailing pad register.
    #[test]
    fn both_trees_tile_the_row() {
        for keys in [1u32, 2, 3, 8, 9, 27, 64, 128, 197, 384, 512] {
            let plan = Plan::try_new(Shape::new(keys, 8)).unwrap();
            let mut width = keys;
            for stage in plan.stages() {
                if let Stage::MaxLevel {
                    input_width,
                    kernel,
                    output_width,
                    pad_right,
                } = *stage
                {
                    assert_eq!(input_width, width, "keys {keys}");
                    assert!(kernel <= MAX_POOL_KERNEL);
                    assert_eq!(output_width * kernel, input_width + pad_right);
                    assert!(pad_right < kernel, "a window entirely in the padding hangs");
                    width = output_width;
                }
            }
            assert_eq!(width, 1, "the max tree did not reduce keys {keys} to one");

            let mut width = keys;
            for stage in plan.stages() {
                if let Stage::SumLevel {
                    input_width,
                    pad_left,
                    output_width,
                } = *stage
                {
                    assert_eq!(input_width, width, "keys {keys}");
                    assert!(pad_left < SUM_KERNEL);
                    assert_eq!(output_width * SUM_KERNEL, input_width + pad_left);
                    width = output_width;
                }
            }
            assert_eq!(width, 1, "the sum tree did not reduce keys {keys} to one");
        }
    }

    /// The costs §5 and §4 argue about, as numbers the planner can report
    /// before anything runs.
    #[test]
    fn the_task_count_is_what_the_design_predicts() {
        // 197 keys: 3 max levels (197 -> 25 -> 4 -> 1) and 5 sum levels
        // (197 -> 66 -> 22 -> 8 -> 3 -> 1), plus the six fixed device
        // stages.
        let plan = Plan::try_new(Shape::new(197, 64)).unwrap();
        // Three max levels, five sum levels, and four fixed device stages:
        // exp, the two bridge tasks, and the normalize.
        assert_eq!(plan.device_tasks(), 3 + 5 + 4);
        assert_eq!(plan.host_boundaries(), 2);
    }

    #[test]
    fn a_single_key_still_plans() {
        let plan = Plan::try_new(Shape::new(1, 1)).unwrap();
        // Nothing to reduce, so no tree levels at all -- and the softmax of
        // one element is 1.0, which the remaining stages still compute.
        assert!(
            !plan
                .stages()
                .iter()
                .any(|stage| matches!(stage, Stage::MaxLevel { .. } | Stage::SumLevel { .. }))
        );
        assert_eq!(plan.device_tasks(), 4);
    }

    #[test]
    fn a_zero_extent_is_refused_as_malformed() {
        let error = Plan::try_new(Shape::new(0, 8)).unwrap_err();
        assert_eq!(error.code(), PlanErrorCode::InvalidShape);
    }

    /// The refusals that mark where the *measurement* stops, not the
    /// hardware. Each one names what would have to be measured.
    #[test]
    fn unmeasured_shapes_are_refused_as_unvalidated() {
        for shape in [
            Shape::new(513, 8),
            Shape::new(197, 65),
            Shape {
                keys: 197,
                rows: 8,
                exp_input_scale: 1.0 / 16.0,
            },
        ] {
            let error = Plan::try_new(shape).unwrap_err();
            assert_eq!(
                error.code(),
                PlanErrorCode::UnvalidatedConfiguration,
                "{shape:?} should refuse as unvalidated, not as something else"
            );
        }
    }

    #[test]
    fn the_exp_reach_is_the_slope_s_own_distance() {
        let shape = Shape::new(197, 8);
        assert_eq!(shape.exp_reach(), -6.4);
        assert!(
            -shape.exp_reach() > EXP_DOMAIN_EDGE,
            "the tuned slope only exists when the input reaches past the table's domain"
        );
    }
}
