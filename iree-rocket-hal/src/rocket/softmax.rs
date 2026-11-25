//! Softmax over the width axis, as register programs.
//!
//! [`rocket_core::softmax`] is the plan -- which stages, in what order,
//! over what extents, and which shapes the lowering refuses. This module
//! emits them, and owns the two things a plan cannot: the scratch layout
//! and the submission boundaries.
//!
//! **Three submissions, not one**, because two stages are the host's and
//! both sit in the middle (§4, §5):
//!
//! ```text
//!   submit 1   the row-maximum tree                     (PPU, int8)
//!     host     negate the maximum, scale it by the exp task's own BN
//!              multiplier, write it as i32 per channel
//!   submit 2   exp -> Q15, the two bridge tasks, the sum tree
//!     host     1/d per row
//!   submit 3   the normalising multiply
//! ```
//!
//! Each boundary is one `SUBMIT` plus one `PREP_BO`, measured at 0.037 to
//! 0.059 ms, against ISSUES.md P8's 1.6 to 7.4 ms of flat *per-dispatch*
//! host cost -- so the split is not a second dispatch and does not cost
//! like one. Both host steps work on a width-1, `rows`-wide vector; every
//! pass over the `[keys, rows]` cube is on the device.
//!
//! Every mechanism here was measured on an RK3588 before it was assembled;
//! SOFTMAX.md §8 is the record, probe by probe. What this module adds is
//! the arithmetic that has to agree between them -- three different cube
//! layouts, in particular, and getting one wrong is how an assembly like
//! this fails:
//!
//! * a **PPU** cube rounds both surface strides up to four pixels, so a
//!   width-1 maximum puts channel `c` at `(c / lanes) * 4 * 16 + ...`;
//! * a **DPU** cube's surface stride is `width * height` atoms, with no
//!   rounding;
//! * an int8 atom holds 16 channels and an fp16 or int16 atom 8.

use rocket_core::softmax::{Plan, Shape, Stage};

use crate::rocket::{
    activation::{LutBuffers, LutShape, LutTable, build_lut_q15_regcmd_with_plane},
    bn_plane::{
        BnPlaneBuffers, BnPlaneOp, BnPlaneShape, Q15BridgeBuffers, Q15BridgeShape,
        build_bn_plane_regcmd, build_q15_bridge_regcmd,
    },
    builders::RegCmd,
    conv::{self, Buffers as ConvBuffers, ConvPlan, PlanError, Precision},
    pooling::{PoolingBuffers, PoolingMethod, PoolingPlan, PoolingPrecision, PoolingShape},
};

/// Bytes in one feature atom, every layout.
const ATOM_BYTES: usize = 16;
/// The `exp` table's Q15 scale, which cancels in softmax's own ratio but
/// has to be divided out for a standalone op (§4.2).
const Q15_SCALE: f32 = 32768.0;

/// Channels one atom holds at a given element width.
fn lanes(element_bytes: usize) -> usize {
    ATOM_BYTES / element_bytes
}

/// Bytes a DPU cube occupies: surfaces of `pixels` atoms, no rounding.
fn dpu_cube_bytes(channels: u32, pixels: usize, element_bytes: usize) -> usize {
    (channels as usize).div_ceil(lanes(element_bytes)) * pixels * ATOM_BYTES
}

/// Byte offset of one element of a DPU cube.
pub fn dpu_cube_offset(channel: usize, pixel: usize, pixels: usize, element_bytes: usize) -> usize {
    let lanes = lanes(element_bytes);
    (channel / lanes) * pixels * ATOM_BYTES + pixel * ATOM_BYTES + (channel % lanes) * element_bytes
}

/// Bytes a PPU cube occupies: surfaces rounded up to four pixels.
fn ppu_cube_bytes(channels: u32, pixels: usize, element_bytes: usize) -> usize {
    (channels as usize).div_ceil(lanes(element_bytes)) * pixels.next_multiple_of(4) * ATOM_BYTES
}

/// Byte offset of one element of a PPU cube.
pub fn ppu_cube_offset(channel: usize, pixel: usize, pixels: usize, element_bytes: usize) -> usize {
    let lanes = lanes(element_bytes);
    (channel / lanes) * pixels.next_multiple_of(4) * ATOM_BYTES
        + pixel * ATOM_BYTES
        + (channel % lanes) * element_bytes
}

/// Where each intermediate lives inside one scratch allocation.
///
/// One buffer rather than nine, because every one of these is touched only
/// by tasks of this op and the driver's scratch pool keys on size
/// (MULTICORE.md §12). The two the host reads -- the final maximum and the
/// final sum -- are given their own accessors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Scratch {
    max_levels: Vec<usize>,
    q15: usize,
    counts: usize,
    probabilities: usize,
    sum_levels: Vec<usize>,
    bytes: usize,
}

impl Scratch {
    /// Total bytes the scratch allocation needs.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Offset of the width-1 maximum the host reads, in a **PPU** cube at
    /// int8. Zero-length when there was nothing to reduce.
    pub fn max_offset(&self) -> Option<usize> {
        self.max_levels.last().copied()
    }

    /// Offset of the width-1 sum the host reads, in a **DPU** cube at fp16.
    pub fn sum_offset(&self) -> Option<usize> {
        self.sum_levels.last().copied()
    }

    /// Offset of the probabilities the normalising multiply reads, which is
    /// also what it writes over when the caller passes the same address.
    pub fn probabilities_offset(&self) -> usize {
        self.probabilities
    }

    /// Offset of the `exp` task's Q15 `int16` output. For a caller that
    /// wants to read an intermediate when an assembly goes wrong -- which
    /// is what these are for, since deriving one from another is its own
    /// way to be wrong.
    pub fn q15_offset(&self) -> usize {
        self.q15
    }

    /// Offset of the bridge's first task's fp16 output.
    pub fn counts_offset(&self) -> usize {
        self.counts
    }

    /// Offset of each sum level's output, outermost first.
    pub fn sum_level_offsets(&self) -> &[usize] {
        &self.sum_levels
    }

    /// Offset of each max level's output, outermost first.
    pub fn max_level_offsets(&self) -> &[usize] {
        &self.max_levels
    }
}

/// The buffers one softmax needs. Addresses, not handles: this module emits
/// relocated programs and leaves allocation to the caller.
pub struct SoftmaxBuffers {
    /// The op's input: an int8 `[keys, rows]` DPU cube of scores, already
    /// at the `exp` table's own input scale. In §1.1's layout that is what
    /// the score matmul's own requantisation produces.
    pub scores: u32,
    /// One allocation of [`Scratch::bytes`].
    pub scratch: u32,
    /// The `exp` task's BN ALU plane: `i32` per channel, written by the
    /// host between submissions 1 and 2 (§4.2, S5b).
    pub alu_plane: u32,
    /// A constant fp16 `2^-15` for every channel, for the bridge's second
    /// task. Compile-time constant; [`write_constant_plane`] fills it.
    pub const_plane: u32,
    /// `1/d` per row as fp16, written by the host between submissions 2
    /// and 3.
    pub reciprocal_plane: u32,
    /// The all-ones depthwise coefficients the sum tree shares, packed
    /// once. [`pack_sum_weights`] fills it.
    pub sum_weights: u32,
    /// A zeroed bias for the sum tree's convolutions.
    pub sum_bias: u32,
    /// The op's output: an fp16 `[keys, rows]` DPU cube of probabilities.
    pub output: u32,
}

/// A planned softmax: the core plan, the per-level sub-plans the planners
/// that own them produced, and the scratch layout.
#[derive(Clone, Debug)]
pub struct SoftmaxPlan {
    plan: Plan,
    max_levels: Vec<PoolingPlan>,
    sum_levels: Vec<ConvPlan>,
    scratch: Scratch,
}

impl SoftmaxPlan {
    /// Plans `shape`, or says why not.
    ///
    /// Refusals come from [`rocket_core::softmax`] for the op's own shape
    /// and from `PoolingPlan`/`ConvPlan` for a level they cannot tile.
    pub fn try_new(shape: Shape) -> Result<SoftmaxPlan, PlanError> {
        let plan = Plan::try_new(shape)?;
        // Every cube is the *physical* width: the only one at which the
        // PPU's four-pixel surface rounding and the DPU's plain
        // `width * height` agree, and this op's score buffer is read by
        // both (`rocket_core::softmax::Shape::physical_keys`).
        let keys = shape.physical_keys() as usize;
        let rows = shape.rows;

        let mut max_levels = Vec::new();
        let mut sum_levels = Vec::new();
        let mut offsets_max = Vec::new();
        let mut offsets_sum = Vec::new();
        let mut cursor = 0usize;

        for stage in plan.stages() {
            match *stage {
                Stage::MaxLevel {
                    input_width,
                    kernel,
                    output_width,
                    pad_right,
                } => {
                    max_levels.push(PoolingPlan::new(PoolingShape {
                        input_width,
                        input_height: 1,
                        input_channels: rows,
                        output_width,
                        output_height: 1,
                        output_channels: rows,
                        precision: PoolingPrecision::Int8,
                        kernel_width: kernel,
                        kernel_height: 1,
                        stride_x: kernel,
                        stride_y: 1,
                        method: PoolingMethod::Max,
                        pad_left: 0,
                        pad_top: 0,
                        pad_right,
                        pad_bottom: 0,
                        pad_value: PoolingMethod::Max.pad_fill_value(PoolingPrecision::Int8),
                    }));
                    offsets_max.push(cursor);
                    cursor += ppu_cube_bytes(rows, output_width as usize, 1);
                }
                Stage::SumLevel {
                    input_width,
                    pad_left,
                    output_width,
                } => {
                    let conv_shape = conv::Shape {
                        width: input_width,
                        height: 1,
                        stride: rocket_core::softmax::SUM_KERNEL,
                        in_channels: rows,
                        out_channels: rows,
                        precision: Precision::Fp16,
                        // [pad_top, pad_left]: the vertical pad is what
                        // lets a 3-row window sit on a 1-row image at all,
                        // and its taps are zeros, which is what a sum
                        // wants (§2.2).
                        padding: Some([1, pad_left as usize]),
                        activation: conv::Activation::None,
                        depthwise: true,
                    };
                    let level = ConvPlan::try_new(conv_shape, SUM_KERNELS)?;
                    debug_assert_eq!(conv_shape.output_width(SUM_KERNELS), output_width);
                    sum_levels.push(level);
                    offsets_sum.push(cursor);
                    cursor += dpu_cube_bytes(rows, output_width as usize, 2);
                }
                _ => {}
            }
        }

        let q15 = cursor;
        cursor += dpu_cube_bytes(rows, keys, 2);
        let counts = cursor;
        cursor += dpu_cube_bytes(rows, keys, 2);
        let probabilities = cursor;
        cursor += dpu_cube_bytes(rows, keys, 2);

        Ok(SoftmaxPlan {
            plan,
            max_levels,
            sum_levels,
            scratch: Scratch {
                max_levels: offsets_max,
                q15,
                counts,
                probabilities,
                sum_levels: offsets_sum,
                bytes: cursor,
            },
        })
    }

    pub fn shape(&self) -> Shape {
        self.plan.shape()
    }

    pub fn scratch(&self) -> &Scratch {
        &self.scratch
    }

    /// The `BN_MUL` operand the `exp` task programs, which the host needs
    /// to scale the maximum into [`SoftmaxBuffers::alu_plane`] (S5b).
    pub fn exp_bn_mul_operand(&self) -> u32 {
        crate::rocket::activation::lut_bn_mul_operand(
            self.plan.shape().exp_input_scale,
            self.exp_table(),
        )
    }

    /// The `exp` table this plan uses: §3.3's tuned underflow slope at the
    /// shape's own input scale, without which a wide softmax is corrupted
    /// outright rather than merely inaccurate.
    pub fn exp_table(&self) -> LutTable {
        LutTable::exp_with_decaying_tail(self.plan.shape().exp_input_scale)
    }

    /// Submission 1: the row-maximum tree.
    pub fn programs_max(&self, bufs: &SoftmaxBuffers) -> Vec<Vec<RegCmd>> {
        let mut programs = Vec::new();
        for (index, level) in self.max_levels.iter().enumerate() {
            let input = if index == 0 {
                bufs.scores
            } else {
                bufs.scratch + self.scratch.max_levels[index - 1] as u32
            };
            programs.extend(level.programs_with_buffers(&PoolingBuffers {
                input_addr: input,
                output_addr: bufs.scratch + self.scratch.max_levels[index] as u32,
            }));
        }
        programs
    }

    /// Submission 2: `exp` with the host's plane, the two bridge tasks, and
    /// the sum tree.
    pub fn programs_exp_and_sum(&self, bufs: &SoftmaxBuffers) -> Vec<Vec<RegCmd>> {
        let shape = self.plan.shape();
        let mut programs = Vec::new();

        programs.push(build_lut_q15_regcmd_with_plane(
            &LutShape {
                width: shape.physical_keys(),
                height: 1,
                channels: shape.rows,
                input_zero_point: 0x80,
                output_zero_point: 0x80,
                input_scale: shape.exp_input_scale,
                // Ignored by the Q15 builder: its converter is at identity.
                output_scale: 1.0,
            },
            &LutBuffers {
                input_addr: bufs.scores,
                output_addr: bufs.scratch + self.scratch.q15 as u32,
            },
            self.exp_table(),
            bufs.alu_plane,
        ));

        // Bridge task 1: int16 to fp16, exactly, which is all an integer
        // read can do -- with that input the datapath is integer and a
        // fractional result truncates (§4.2).
        programs.push(build_q15_bridge_regcmd(
            &Q15BridgeShape {
                width: shape.physical_keys(),
                height: 1,
                channels: shape.rows,
                input_int16: true,
                constant_shift: 0,
                per_channel: false,
                per_channel_add: false,
                out_scale: 0,
            },
            &Q15BridgeBuffers {
                input_addr: bufs.scratch + self.scratch.q15 as u32,
                plane_addr: 0,
                output_addr: bufs.scratch + self.scratch.counts as u32,
            },
        ));

        // Bridge task 2: the 2^-15 that makes them probabilities, on the
        // fp16 path, from the one stage in the DPU that multiplies by a
        // true fraction.
        programs.push(build_q15_bridge_regcmd(
            &Q15BridgeShape {
                width: shape.physical_keys(),
                height: 1,
                channels: shape.rows,
                input_int16: false,
                constant_shift: 0,
                per_channel: true,
                per_channel_add: false,
                out_scale: 0,
            },
            &Q15BridgeBuffers {
                input_addr: bufs.scratch + self.scratch.counts as u32,
                plane_addr: bufs.const_plane,
                output_addr: bufs.scratch + self.scratch.probabilities as u32,
            },
        ));

        for (index, level) in self.sum_levels.iter().enumerate() {
            let input = if index == 0 {
                bufs.scratch + self.scratch.probabilities as u32
            } else {
                bufs.scratch + self.scratch.sum_levels[index - 1] as u32
            };
            programs.extend(level.programs_with_buffers(ConvBuffers {
                input,
                weights: bufs.sum_weights,
                bias: bufs.sum_bias,
                output: bufs.scratch + self.scratch.sum_levels[index] as u32,
            }));
        }
        programs
    }

    /// Submission 3: the normalising per-channel multiply.
    pub fn programs_normalize(&self, bufs: &SoftmaxBuffers) -> Vec<Vec<RegCmd>> {
        let shape = self.plan.shape();
        vec![build_bn_plane_regcmd(
            &BnPlaneShape {
                width: shape.physical_keys(),
                height: 1,
                channels: shape.rows,
                op: BnPlaneOp::Mul,
                mul_shift: 0,
            },
            &BnPlaneBuffers {
                input_addr: bufs.scratch + self.scratch.probabilities as u32,
                plane_addr: bufs.reciprocal_plane,
                output_addr: bufs.output,
            },
        )]
    }

    /// Every device task, for a caller that wants the count without
    /// building the programs.
    pub fn device_tasks(&self) -> usize {
        self.max_levels
            .iter()
            .map(|level| level.tiles().len())
            .sum::<usize>()
            + 3
            + self
                .sum_levels
                .iter()
                .map(|level| level.tiles().len())
                .sum::<usize>()
            + 1
    }

    /// The host's own two steps, in the order they happen.
    ///
    /// Reading these back is how the host knows what to write into the two
    /// planes, and the offsets are into [`SoftmaxBuffers::scratch`].
    pub fn host_steps(&self) -> Vec<Stage> {
        self.plan
            .stages()
            .iter()
            .copied()
            .filter(|stage| matches!(stage, Stage::HostMaxToPlane | Stage::HostReciprocal))
            .collect()
    }
}

/// The sum tree's kernel, as `ConvPlan` wants it.
pub const SUM_KERNELS: conv::Kernels = [
    rocket_core::softmax::SUM_KERNEL as usize,
    rocket_core::softmax::SUM_KERNEL as usize,
];

/// Bytes the shared all-ones coefficients need for `rows` channels.
pub fn sum_weight_bytes(rows: u32) -> Result<u32, PlanError> {
    let shape = conv::Shape {
        width: rocket_core::softmax::SUM_KERNEL,
        height: 1,
        stride: rocket_core::softmax::SUM_KERNEL,
        in_channels: rows,
        out_channels: rows,
        precision: Precision::Fp16,
        padding: Some([1, 0]),
        activation: conv::Activation::None,
        depthwise: true,
    };
    Ok(shape.weight_bytes(SUM_KERNELS))
}

/// Packs the all-ones depthwise coefficients every sum level shares.
///
/// One buffer serves the whole tree, because neither the channel count nor
/// the kernel changes between levels (§2.2).
pub fn pack_sum_weights(rows: u32, packed: &mut [u8]) -> Result<usize, &'static str> {
    let kernel = rocket_core::softmax::SUM_KERNEL as usize;
    let bytes = sum_weight_bytes(rows).map_err(|_| "the sum tree's shape has no weight size")?;
    let padded_channels = bytes as usize / (kernel * kernel * 2);
    let one = fp16_one();
    let mut dense = vec![0u8; rows as usize * kernel * kernel * 2];
    for slot in dense.chunks_mut(2) {
        slot.copy_from_slice(&one);
    }
    rocket_core::weights::pack_depthwise_to_rocket_weights(
        &dense,
        kernel,
        kernel,
        rows as usize,
        padded_channels,
        2,
        packed,
    )
}

/// Fills the bridge's constant plane with fp16 `2^-15` for every channel.
///
/// Exactly representable, since fp16 subnormals are `k * 2^-24` and this is
/// `512 * 2^-24` -- and necessarily a separate multiply from `1/d`, whose
/// product with it would be a non-power-of-two subnormal worth a couple of
/// bits (§4.2).
pub fn write_constant_plane(rows: u32, plane: &mut [u8]) -> Result<usize, &'static str> {
    let channels = rows.max(16).next_multiple_of(16);
    let values = vec![1.0f32 / Q15_SCALE; rows as usize];
    crate::rocket::bn_plane::write_bn_mul_plane_fp16(plane, channels, &values)
}

/// The fp16 bit pattern of 1.0.
fn fp16_one() -> [u8; 2] {
    0x3C00u16.to_le_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_task_count_matches_the_core_plan() {
        let plan = SoftmaxPlan::try_new(Shape::new(197, 64)).unwrap();
        // Every level may tile, so the HAL's count is at least the plan's.
        assert!(plan.device_tasks() >= 3 + 5 + 4);
        assert_eq!(plan.host_steps().len(), 2);
        assert_eq!(
            plan.host_steps(),
            vec![Stage::HostMaxToPlane, Stage::HostReciprocal]
        );
    }

    /// The three submissions have to cover every device task exactly once,
    /// and the scratch has to hold every intermediate they name.
    #[test]
    fn the_submissions_cover_every_task_and_fit_the_scratch() {
        for keys in [8u32, 64, 197, 384] {
            let plan = SoftmaxPlan::try_new(Shape::new(keys, 32)).unwrap();
            let bufs = SoftmaxBuffers {
                scores: 0x1_0000,
                scratch: 0x10_0000,
                alu_plane: 0x2_0000,
                const_plane: 0x3_0000,
                reciprocal_plane: 0x4_0000,
                sum_weights: 0x5_0000,
                sum_bias: 0x6_0000,
                output: 0x7_0000,
            };
            let total = plan.programs_max(&bufs).len()
                + plan.programs_exp_and_sum(&bufs).len()
                + plan.programs_normalize(&bufs).len();
            assert_eq!(total, plan.device_tasks(), "keys {keys}");
            assert!(plan.scratch().bytes() > 0);
            assert!(plan.scratch().max_offset().is_some(), "keys {keys}");
            assert!(plan.scratch().sum_offset().is_some(), "keys {keys}");
        }
    }

    /// The two layouts this module has to keep apart. A PPU cube rounds its
    /// surface stride up to four pixels and a DPU cube does not, so at
    /// width 1 they differ by exactly the thing that would silently corrupt
    /// every channel past the first.
    #[test]
    fn the_two_cube_layouts_differ_where_they_should() {
        // Width 1, int8, 16 lanes per atom: channel 16 is the second
        // surface.
        assert_eq!(ppu_cube_offset(16, 0, 1, 1), 4 * ATOM_BYTES);
        assert_eq!(dpu_cube_offset(16, 0, 1, 1), ATOM_BYTES);
        // Inside one atom they agree.
        assert_eq!(ppu_cube_offset(3, 0, 1, 1), dpu_cube_offset(3, 0, 1, 1));
        // And at a width that is already a multiple of four they agree
        // everywhere.
        for channel in 0..32 {
            assert_eq!(
                ppu_cube_offset(channel, 2, 8, 2),
                dpu_cube_offset(channel, 2, 8, 2)
            );
        }
    }

    /// The physical width is the one at which this op's two readers of the
    /// score buffer agree, and the assembly is wrong without it -- quietly,
    /// since a row whose maximum sits in the first atom still comes out
    /// right.
    #[test]
    fn every_cube_uses_the_physical_width() {
        let shape = Shape::new(197, 64);
        assert_eq!(shape.physical_keys(), 200);
        let pixels = shape.physical_keys() as usize;
        for channel in 0..shape.rows as usize {
            assert_eq!(
                ppu_cube_offset(channel, 0, pixels, 1),
                dpu_cube_offset(channel, 0, pixels, 1),
                "channel {channel} of the score cube"
            );
        }
        // And at the logical width they do not, which is the bug this
        // guards against.
        assert_ne!(
            ppu_cube_offset(16, 0, shape.keys as usize, 1),
            dpu_cube_offset(16, 0, shape.keys as usize, 1)
        );
    }

    #[test]
    fn a_shape_the_core_refuses_is_refused_here() {
        assert!(SoftmaxPlan::try_new(Shape::new(0, 8)).is_err());
        assert!(SoftmaxPlan::try_new(Shape::new(197, 65)).is_err());
    }

    #[test]
    fn the_shared_coefficients_are_all_ones_and_sized_for_every_level() {
        let rows = 32;
        let bytes = sum_weight_bytes(rows).unwrap() as usize;
        let mut packed = vec![0xA5u8; bytes];
        let written = pack_sum_weights(rows, &mut packed).unwrap();
        assert_eq!(written, bytes);
        // Every level of the tree shares it, so its size cannot depend on
        // the level's width.
        let plan = SoftmaxPlan::try_new(Shape::new(197, rows)).unwrap();
        for level in &plan.sum_levels {
            assert_eq!(level.shape().weight_bytes(SUM_KERNELS) as usize, bytes);
        }
    }

    #[test]
    fn the_constant_plane_is_the_exact_subnormal() {
        let mut plane = vec![0u8; 4 * 16];
        write_constant_plane(8, &mut plane).unwrap();
        // 2^-15 is 512 * 2^-24, so 0x0200.
        assert_eq!(u16::from_le_bytes([plane[0], plane[1]]), 0x0200);
        assert_eq!(u16::from_le_bytes([plane[14], plane[15]]), 0x0200);
    }
}
