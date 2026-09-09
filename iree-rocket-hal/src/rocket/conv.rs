//! Vendor-derived convolution register program.
//!
//! **The planner is not here any more.** `Shape`, `Precision`, the channel
//! and CBUF limits, `Tile`/`ColumnTile`/`Tile2D` and the pure `ConvPlan`
//! live in `rocket-core` (COMPILER_ROADMAP.md section 1) and are re-exported
//! below, so every `conv::` path still resolves; this file keeps the
//! register emission -- the capture-derived program builders, the
//! experiment overrides that steer them, relocation and BS packing -- plus
//! the `ConvPlan` wrapper that puts `programs()` back on the plan. The
//! evidence comments on the planning rules moved with the rules.
//!
//! This began as a bit-exact reproduction of group 1 (the complete
//! single-core alternative) from the vendor-compiled `32x32x3 -> 32x32x8`
//! fp16 convolution captures, which differ only in kernel geometry: 1x1 with
//! no padding versus 3x3 with SAME padding. [`conv_2d`] still reproduces that
//! program byte for byte, and the hash tests below pin it.
//!
//! It now generalises to arbitrary [`Shape`] and to output-row [`Tile`]s.
//! Every shape- and tile-dependent register formula is derived from vendor
//! captures rather than assumed: the tile formulas from a cross-group diff of
//! the six captured plans, and the width and height formulas from a sweep of
//! 35 captures (212 convolution programs) spanning widths 32..256 and heights
//! 32..256. Registers that vary in no capture stay literal constants.
//!
//! The ordinary tile builder supports fp16 `Cin` 1..=80, int8 `Cin`
//! 1..=128, `Cout` 1..=512, strides 1..4, and 1x1 or 3x3 kernels.
//! [`ConvPlan`] additionally plans kernel extents from 1 through 11,
//! including even and non-square kernels and horizontal tiling where a
//! full-width row cannot fit. [`Shape::with_padding`] makes the leading
//! padding independent of the kernel extent; without it each axis defaults
//! to `extent / 2`, preserving the historical odd-kernel API.
//!
//! Kernels need not be square. A sweep of 53 non-square captures shows the
//! two extents govern their own axes throughout -- `weight_width` and
//! `pad_left` follow the kernel's width, `weight_height`, `pad_top` and
//! `feature_grains` follow its height, and the coefficient footprint is
//! `kh * kw * pad(Cin) * element_bytes` -- so the direct geometry needed no
//! new rule, only the removal of the assumption that one `k` served both.
//! What does *not* carry over is the CBUF split: at equal coefficient
//! demand, mirrored shapes split differently, so [`ConvPlan`] plans
//! non-square kernels only up to the demand where the captures still agree
//! and otherwise requires an explicit split.
//!
//! # Datatypes
//!
//! The datatype is a **3-bit precision field** set per pipeline stage, not
//! a separate datapath, so a rung is "the right field value plus the right
//! layout for the element width". Every layout rule below keys off the
//! element *width* rather than the numeric interpretation, which is what
//! makes the table short:
//!
//! | rung | field | element | feature atom | coeff. N/K group | result |
//! |---|---:|---:|---:|---:|---|
//! | [`Precision::Int4`] | 6 | 4 bit | 32 ch | 64 / 32 | int16 |
//! | [`Precision::Int8`] | 0 | 1 B | 16 ch | 32 / 32 | int8 (requantized) |
//! | [`Precision::Int8Accumulator`] | 0 | 1 B | 16 ch | 32 / 32 | int32 |
//! | [`Precision::Int16`] | 1 | 2 B | 8 ch | 16 / 32 | int16 |
//! | [`Precision::Fp16`] | 2 | 2 B | 8 ch | 16 / 32 | fp16 |
//! | [`Precision::Fp16Accumulator`] | 2 | 2 B | 8 ch | 16 / 32 | fp32 |
//! | [`Precision::Bf16`] | 3 | 2 B | 8 ch | 16 / 32 | bf16 |
//! | [`Precision::Tf32`] | 7 | 4 B | 4 ch | 16 / **16** | fp32 |
//!
//! Three things in that table do not follow from the width and are the
//! places a new rung goes wrong:
//!
//! - **tf32's field is front-of-pipe only.** The DPU's precision enum has
//!   no tf32 code and writes nothing if given one, so its stages run at
//!   fp32 instead. Every other rung programs one value everywhere.
//! - **tf32's coefficient tile.** The tile is a constant 1024 bytes at
//!   every width; below four bytes the K-group is pinned at 32 and the
//!   N-group absorbs the width, but at four bytes the N-group stays 16 and
//!   the K-group halves. A uniform-coefficient test cannot see this at all.
//! - **fp16's narrowing is a bit, not a precision.** An fp16 convolution
//!   already accumulates in fp32; `fp32tofp16_en` is what narrows the
//!   result on the way out. Keeping the accumulator is that bit cleared
//!   plus the float writer's 4-byte geometry, and moves no register on the
//!   input side at all.
//! - **int4's write-out.** Its int16 result is written by the integer path,
//!   which strides as if each element were eight bytes (`size_e = 7`) with
//!   an 8x surface multiplier, not the float path's natural 1 and 2x. With
//!   the natural values the DPU writes 512 bytes and stops.
//!
//! Each rung is validated against `tests/conv2d_oracle_hw.rs`'s oracle on
//! an RK3588 board, with the addressing-sensitive `Selectors` and `Dense`
//! patterns rather than `Counting` alone, and -- for the rungs that have a
//! narrower neighbour sharing their layout -- with `WideOperands` cases
//! whose values a narrower datatype could not hold.
//!
//! All of that holds in both precisions. A matching int8 sweep of 60
//! captures reproduces every kernel-geometry formula unchanged, and the
//! paired fp16/int8 diff moves only the fields precision already moved --
//! the precision selectors, the doubled channel counts, and the
//! requantization path. `weight_width`, `weight_height`, `pad_left` and
//! `pad_top` are byte-identical across precision at every rectangular
//! geometry.
//!
//! Input channel count picks the memory layout. While a pixel fits in half a
//! feature atom -- `Cin` up to 4 -- the vendor keeps it dense NHWC and the
//! CNA pads internally. From `Cin` 5 the map becomes NC1HWC2 surfaces, and
//! the row strides, the CBUF bank split and `data_entries` all change with
//! it; `data_entries` in particular stops depending on the tile height
//! entirely. `FeatureLayout` names the two regimes.
//!
//! Channel padding is a table rather than arithmetic. It is `atoms * 8`
//! except at three atoms, where `datain_channel` stays 24 but coefficients
//! use 32, and at seven, where `datain_channel` stays 56 but coefficients
//! use 64. Atom counts 5, 6, 9 and 10 are unpadded, so no arithmetic rule
//! fits and none is invented.
//!
//! Output channels remain one streamed kernel set rather than splitting into
//! fp16 kernel groups. The capture corpus covers this through `Cout = 512`,
//! with hardware validation through `Cout = 128`.
//!
//! Keeping this separate from `rocket::regcmd` keeps a capture-derived path
//! distinct from that module's Mesa-derived one.
//!
//! # Output-row tiles
//!
//! [`conv_2d_tile`] emits the same single-core program over an arbitrary
//! range of output rows. Every tile-dependent register value is derived
//! from a cross-group diff of the six captured programs, which cover three
//! alternative height splits (32 / 16+16 / 11+11+10) at two kernel sizes --
//! twelve independent observations per register. Sixteen registers vary
//! across those captures and take derived values here; the other 109 are
//! literal constants reproduced from the capture. The derived values are
//! checked against every observation by the tile register test below.
//!
//! A tile program is **not** byte-identical to captured groups 2-6. Those
//! are the vendor's own multi-core plans and carry PPU/PPU_RDMA blocks whose
//! role has not been derived, plus a plan index in two documented-reserved
//! fields. A tile emitted here is a standalone single-core program covering
//! a row range, intended for submission as one of several independent jobs.
//! Its geometry registers match the captures exactly; its command sequence
//! is group 1's.

use crate::rocket::builders::{
    Bits, RegCmd, Register, RegisterMeta,
    cna::{
        CnaCbufCon0, CnaCbufCon1, CnaConvCon1, CnaConvCon2, CnaConvCon3, CnaCvtCon0, CnaCvtCon1,
        CnaCvtCon2, CnaCvtCon3, CnaCvtCon4, CnaCvtCon5, CnaDataSize0, CnaDataSize1, CnaDataSize2,
        CnaDataSize3, CnaDcompAddr0, CnaDcompAmount0, CnaDcompAmount1, CnaDcompAmount2,
        CnaDcompAmount3, CnaDcompAmount4, CnaDcompAmount5, CnaDcompAmount6, CnaDcompAmount7,
        CnaDcompAmount8, CnaDcompAmount9, CnaDcompAmount10, CnaDcompAmount11, CnaDcompAmount12,
        CnaDcompAmount13, CnaDcompAmount14, CnaDcompAmount15, CnaDcompCtrl, CnaDcompRegnum,
        CnaDmaCon0, CnaDmaCon1, CnaDmaCon2, CnaFcCon0, CnaFcCon1, CnaFcCon2, CnaFcDataSize0,
        CnaFcDataSize1, CnaFeatureDataAddr, CnaPadCon0, CnaPadCon1, CnaWeightSize0, CnaWeightSize1,
        CnaWeightSize2,
    },
    core::{CoreClipTruncate, CoreDataoutSize0, CoreDataoutSize1, CoreMiscCfg, CoreReserved3030},
    dpu::{
        DpuBnAluCfg, DpuBnCfg, DpuBnMulCfg, DpuBnReluxCmpValue, DpuBsAluCfg, DpuBsCfg, DpuBsMulCfg,
        DpuBsOwCfg, DpuBsOwOp, DpuBsReluxCmpValue, DpuDataCubeChannel, DpuDataCubeHeight,
        DpuDataCubeNotchAddr, DpuDataCubeWidth, DpuDataFormat, DpuDstBaseAddr, DpuDstSurfStride,
        DpuEwCfg, DpuEwCvtOffsetValue, DpuEwCvtScaleValue, DpuEwOpValue0, DpuEwOpValue1,
        DpuEwOpValue2, DpuEwOpValue3, DpuEwOpValue4, DpuEwOpValue5, DpuEwOpValue6, DpuEwOpValue7,
        DpuEwReluxCmpValue, DpuFeatureModeCfg, DpuLutAccessCfg, DpuLutAccessData, DpuLutCfg,
        DpuLutInfo, DpuLutLeEnd, DpuLutLeSlopeScale, DpuLutLeSlopeShift, DpuLutLeStart,
        DpuLutLoEnd, DpuLutLoSlopeScale, DpuLutLoSlopeShift, DpuLutLoStart, DpuOffsetPend,
        DpuOutCvtOffset, DpuOutCvtScale, DpuOutCvtShift, DpuReserved40c4, DpuSPointer,
        DpuSurfaceAdd, DpuWdmaSize0, DpuWdmaSize1,
    },
    dpu_rdma::{
        DpuRdmaBnBaseAddr, DpuRdmaBrdmaCfg, DpuRdmaBsBaseAddr, DpuRdmaDataCubeChannel,
        DpuRdmaDataCubeHeight, DpuRdmaDataCubeWidth, DpuRdmaErdmaCfg, DpuRdmaEwBaseAddr,
        DpuRdmaEwSurfNotch, DpuRdmaEwSurfStride, DpuRdmaFeatureModeCfg, DpuRdmaNrdmaCfg,
        DpuRdmaPadCfg, DpuRdmaSPointer, DpuRdmaSrcBaseAddr, DpuRdmaSrcDmaCfg, DpuRdmaSurfNotch,
        DpuRdmaWeight,
    },
    pc::{PCOperationMask, PCRegisterAmounts, PCTrailer},
    values::{ArgbInputMode, BurstLength, DataPrecision, DpuOutputMode, OutputPrecision},
};

// The planner itself -- descriptors, limits, layout geometry, CBUF
// partitioning, tiles and `ConvPlan` -- lives in `rocket-core` now
// (COMPILER_ROADMAP.md section 1) and is re-exported wholesale so every
// existing caller and test keeps its `conv::` path. What remains in this
// file is register emission: the capture-derived program builders, the
// experiment overrides that steer them, relocation, and BS packing.
pub use rocket_core::{
    conv::*,
    error::{PlanError, PlanErrorCode},
};

/// The 3-bit precision field this datatype programs into the CNA input,
/// CORE processing and DPU stages.
fn data_precision(precision: Precision) -> DataPrecision {
    match precision {
        Precision::Fp16 | Precision::Fp16Accumulator => DataPrecision::Fp16,
        Precision::Bf16 => DataPrecision::Bf16,
        Precision::Int16 => DataPrecision::Int16,
        Precision::Int4 => DataPrecision::Int4,
        Precision::Tf32 => DataPrecision::Tf32,
        Precision::Int8(_) | Precision::Int8Accumulator(_) => DataPrecision::Int8,
    }
}

/// The precision field the DPU's *input* and *processing* stages take.
///
/// The same value as [`Precision::data_precision`] for every rung but
/// tf32, whose 7 is a front-of-pipe code: the DPU enum reaches only 6,
/// and setting its stages to 7 makes it write nothing at all. tf32
/// therefore runs the DPU at fp32, which is what its accumulator is.
fn dpu_data_precision(precision: Precision) -> OutputPrecision {
    match precision {
        Precision::Tf32 => OutputPrecision::Fp32,
        Precision::Fp16 | Precision::Fp16Accumulator => OutputPrecision::Fp16,
        Precision::Bf16 => OutputPrecision::Bf16,
        Precision::Int16 => OutputPrecision::Int16,
        Precision::Int4 => OutputPrecision::Int4,
        Precision::Int8(_) | Precision::Int8Accumulator(_) => OutputPrecision::Int8,
    }
}

/// The precision field the DPU output stage programs.
///
/// Not simply [`Precision::data_precision`]: two rungs write a result
/// wider than their operands. int8 accumulator output writes int32, and
/// int4 writes the int16 its MACs accumulate into -- the
/// `int4 -> int16` / `int8 -> int32` / `fp16 -> fp32` pairing from
/// `../rockchip-npu-notes/datatypes.md`, with fp16's fp32 accumulator
/// converted down on the way out where int4's int16 one is not.
///
/// The DPU's enum is also not the CNA/CORE one; see [`OutputPrecision`].
fn output_data_precision(precision: Precision) -> OutputPrecision {
    match precision {
        Precision::Fp16 => OutputPrecision::Fp16,
        Precision::Bf16 => OutputPrecision::Bf16,
        Precision::Int16 => OutputPrecision::Int16,
        Precision::Int4 => OutputPrecision::Int16,
        // The DPU has no tf32 code at all; the stage runs at fp32.
        // fp16-with-accumulator asks for the same writer from the other
        // direction: its stages stay fp16 and only the result widens.
        Precision::Tf32 | Precision::Fp16Accumulator => OutputPrecision::Fp32,
        Precision::Int8(_) => OutputPrecision::Int8,
        Precision::Int8Accumulator(_) => OutputPrecision::Int32,
    }
}

/// `(bn_bypass, bn_relu_bypass, bn_relux_en, cmp)`, the four fields the
/// activation sweep found moving.
fn bn_programming(activation: Activation) -> (u32, u32, u32, u32) {
    match activation {
        Activation::None => (1, 1, 0, 0),
        Activation::Relu => (0, 0, 0, 0),
        Activation::Clamped { cmp } => (0, 0, 1, cmp),
    }
}

/// `CNA_CONV_CON1`, `DPU_FEATURE_MODE_CFG` and
/// `DPU_RDMA_RDMA_FEATURE_MODE_CFG` all carry the same mode, 3 for
/// depthwise against 0 for a dense convolution.
fn conv_mode(shape: Shape) -> u32 {
    if shape.depthwise { 3 } else { 0 }
}

/// `DPU_BS_OW_CFG.SIZE_E_0/1/2`: 3 for depthwise, **7 for dense
/// accumulator output**, 1 for everything else.
///
/// The 7 is the integer-output stride quirk
/// (`rockchip-npu-notes/encodings/size-e-quirk.md`,
/// `rocket-userspace`'s `gen_matmul_int8`): an integer conv output strides
/// as if each element were 8 bytes, regardless of its actual width. It
/// looks wrong against the float rule (`size_e = bytes - 1`, so 3 for a
/// 4-byte int32) and it is not; do not "fix" it to 3.
///
/// It is only meaningful together with `mc_surf_out = 0` and
/// `surf_add = dataout_w * dataout_h * 8` -- the three are one geometry,
/// and moving any one alone reads as inert. See
/// [`DENSE_ACCUMULATOR_SURF_MULT`] and [`bs_ow_size_e_override`].
///
/// **On the requantized path this value is load-bearing in a sharper
/// way**: it is one of the few registers where a wrong value *hangs the
/// NPU* rather than returning wrong data. Measured at 32x32 Cin=384
/// Cout=64 k1 [HW sweep, planck 2026-09-03, `accumulator_size_e_probe`]:
/// requantized int8 (which leaves `OD_BYPASS` clear) is bit-exact at
/// `size_e = 1` in ~30 ms, and at 3 or 7 writes 1024 of 65536 bytes and
/// takes ~525 ms per tile -- the watchdog killing the job, with `PREP_BO`
/// still returning success.
fn bs_ow_size_e(shape: Shape) -> u32 {
    if let Some(size_e) = bs_ow_size_e_override(shape.in_channels) {
        return size_e;
    }
    if let Some(size_e) = int4_override(shape.precision, "SIZE_E") {
        return size_e;
    }
    if shape.depthwise {
        3
    } else if shape.precision.writes_fp32_result() {
        // The float rule, `size_e = output bytes - 1`: a 4-byte fp32
        // result is 3, which is what the notes' fp32-out writer uses.
        3
    } else if shape.precision.writes_accumulators() || shape.precision == Precision::Int4 {
        // int4 is the integer-output quirk
        // `../rockchip-npu-notes/encodings/size-e-quirk.md` describes:
        // its result is a 2-byte int16, whose natural `size_e` would be
        // 1, but the integer write path strides as if each element were
        // eight bytes. Measured directly here [HW sweep, planck
        // 2026-09-03, `int4_output_write_map_probe`]: at 8x8 Cin 64
        // Cout 64, `size_e` 0/1/2/3 write 0/2/3/4 of the eight output
        // surfaces and stop, and only 7 writes all 8192 bytes.
        7
    } else {
        1
    }
}

/// A planned convolution with its register programs: the pure
/// [`rocket_core::conv::ConvPlan`] plus emission.
///
/// Every planning query (`tiles`, `shape`, `data_banks`, ...) derefs to the
/// core plan; this type adds only what needs the register builders. It is
/// the adapter COMPILER_ROADMAP.md section 1 asks for, so that no caller
/// of `ConvPlan::new(shape, kernels).programs()` had to change when the
/// planner moved.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConvPlan(rocket_core::conv::ConvPlan);

impl std::ops::Deref for ConvPlan {
    type Target = rocket_core::conv::ConvPlan;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<rocket_core::conv::ConvPlan> for ConvPlan {
    fn from(plan: rocket_core::conv::ConvPlan) -> ConvPlan {
        ConvPlan(plan)
    }
}

impl ConvPlan {
    /// Plans `shape` under the automatic CBUF partition, panicking on a
    /// refusal with the planner's message.
    pub fn new(shape: Shape, kernels: Kernels) -> ConvPlan {
        ConvPlan(rocket_core::conv::ConvPlan::new(shape, kernels))
    }

    /// [`ConvPlan::new`], returning the planner's refusal instead.
    pub fn try_new(shape: Shape, kernels: Kernels) -> Result<ConvPlan, PlanError> {
        rocket_core::conv::ConvPlan::try_new(shape, kernels).map(ConvPlan)
    }

    /// Plans `shape` under an explicit CBUF partition; see the core
    /// [`rocket_core::conv::ConvPlan::with_cbuf_banks`].
    pub fn with_cbuf_banks(
        shape: Shape,
        kernels: Kernels,
        data_banks: u32,
        weight_banks: u32,
    ) -> ConvPlan {
        ConvPlan(rocket_core::conv::ConvPlan::with_cbuf_banks(
            shape,
            kernels,
            data_banks,
            weight_banks,
        ))
    }

    /// The pure plan, for callers that want to hand it across a boundary
    /// that does not know about register programs.
    pub fn plan(&self) -> &rocket_core::conv::ConvPlan {
        &self.0
    }

    /// Emits one relocatable register program per planned tile.
    ///
    /// The programs still carry tile offsets rather than addresses; use
    /// [`ConvPlan::programs_with_buffers`] to get submission-ready ones.
    pub fn programs(&self) -> Vec<Vec<RegCmd>> {
        self.tiles()
            .iter()
            .map(|tile| {
                conv_2d_tile_program(
                    self.shape(),
                    self.kernels(),
                    tile,
                    feature_grains_planned(self.shape(), self.kernels(), &tile.rows),
                    self.data_banks(),
                    self.weight_banks(),
                    OutputPlacement::SharedImage,
                )
            })
            .collect()
    }

    /// Emits one submission-ready register program per planned tile, bound to
    /// `buffers`.
    ///
    /// All tiles share the same four buffers: each program's own tile offsets
    /// are what select its slice of them, so there is no per-tile address
    /// arithmetic for the caller to do. Submit each program as its own job and
    /// wait for its fence before the next -- tiles reload their own weights,
    /// so no CBUF state has to survive between them.
    pub fn programs_with_buffers(&self, buffers: Buffers) -> Vec<Vec<RegCmd>> {
        self.programs()
            .into_iter()
            .map(|mut commands| {
                relocate(&mut commands, buffers);
                commands
            })
            .collect()
    }

    /// Emits accumulator programs whose outputs occupy independent,
    /// contiguous ranges of one private scratch buffer.
    ///
    /// A normal tile program addresses a sub-rectangle of a shared full-image
    /// surface: its destination surface stride and row notch therefore retain
    /// the full output geometry. Merely replacing its destination base is not
    /// enough to stage it independently. This entry point changes all three
    /// pieces together -- base, surface stride, and notch -- and returns the
    /// same tile layout the caller must use when compacting scratch into its
    /// logical output tensor.
    pub fn programs_with_staged_accumulator_output(
        &self,
        buffers: Buffers,
    ) -> StagedAccumulatorOutput {
        let mut staged = self.staged_accumulator_programs();
        for (program, tile) in staged.programs.iter_mut().zip(&staged.tiles) {
            relocate_staged_accumulator(program, buffers, tile);
        }
        staged
    }

    /// [`ConvPlan::programs_with_staged_accumulator_output`] before
    /// relocation: the programs still carry their input/weight/bias tile
    /// offsets and an output offset of zero, and `tiles` says where each
    /// one's output lands. Bind each with [`relocate_staged_accumulator`]
    /// -- once per program, against whichever buffers that tile should use,
    /// which is how one dispatch's tiles can be spread over several NPU
    /// contexts without planning the dispatch once per context.
    pub fn staged_accumulator_programs(&self) -> StagedAccumulatorOutput {
        let layout = self
            .accumulator_output_layout()
            .unwrap_or_else(|error| panic!("{error}"));
        let programs = self
            .tiles()
            .iter()
            .map(|tile| {
                conv_2d_tile_program(
                    self.shape(),
                    self.kernels(),
                    tile,
                    feature_grains_planned(self.shape(), self.kernels(), &tile.rows),
                    self.data_banks(),
                    self.weight_banks(),
                    OutputPlacement::ContiguousTile,
                )
            })
            .collect();
        StagedAccumulatorOutput {
            programs,
            tiles: layout.tiles,
            scratch_bytes: layout.scratch_bytes,
        }
    }
}

static ACCUMULATOR_OD_ENGAGE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

static ACCUMULATOR_BS_ENGAGE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// `DPU_BS_MUL_CFG.bs_mul_shift_value`, and its negated twin
/// `DPU_DATA_FORMAT.bs_mul_shift_value_neg`, in every quantized capture.
const BS_MUL_SHIFT_VALUE: u32 = 14;

/// `DPU_RDMA_RDMA_BRDMA_CFG.brdma_data_use` when nothing consumes BRDMA, which
/// is every path that bypasses the BS plane. See ISSUES.md C8: a fetch left
/// enabled with no consumer poisons the core.
const BRDMA_DATA_USE_NONE: u32 = 0;

/// `DPU_RDMA_RDMA_BRDMA_CFG.brdma_data_use` when BRDMA supplies bias only.
const BRDMA_DATA_USE_BIAS: u32 = 1;

/// The same field once requantization is active and BRDMA also supplies the
/// scale and shift operands.
const BRDMA_DATA_USE_QUANTIZED: u32 = 7;

/// ARGB input mode for a dense feature map.
///
/// Only reachable for `Cin` 1..=4, since wider pixels use surfaces. The
/// captures confirm 3 and 4 directly; 1 and 2 follow the enum's own
/// definition, which came from the vendor register description.
fn argb_input_mode(in_channels: u32) -> ArgbInputMode {
    match in_channels {
        1 => ArgbInputMode::OneChannel,
        2 => ArgbInputMode::TwoChannels,
        3 => ArgbInputMode::ThreeChannels,
        4 => ArgbInputMode::FourChannels,
        _ => unreachable!("dense layout is only used up to four input channels"),
    }
}

/// Submission-ready accumulator programs and the exact scratch layout they
/// write. `buffers.output` passed to
/// [`ConvPlan::programs_with_staged_accumulator_output`] must address at
/// least `scratch_bytes` bytes.
pub struct StagedAccumulatorOutput {
    pub programs: Vec<Vec<RegCmd>>,
    pub tiles: Vec<AccumulatorOutputTile>,
    pub scratch_bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OutputPlacement {
    SharedImage,
    ContiguousTile,
}

#[inline]
fn zero<R: RegisterMeta>() -> RegCmd {
    Register::<R>::new().build()
}

/// One output channel's entry in the BS (bias/scale) buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BsEntry {
    /// `round(bias / (input_scale * weight_scale[c]))`.
    pub bias: i32,
    /// Addend applied to each raw coefficient for this output channel.
    /// Quantized affine weights use `-weight_zero_point[c]` so the hardware
    /// dot product sees `raw_weight - weight_zero_point[c]`.
    pub constant: i16,
    /// `round(BS_UNIT_MULTIPLIER * weight_scale[c] / max(weight_scale))`.
    pub multiplier: i16,
}

impl Default for BsEntry {
    /// A zero bias at unit multiplier -- what a convolution with uniform
    /// weight scales and no bias needs. Pair it with
    /// [`Multiplier::for_unit_bs`], which cancels the gain this carries.
    fn default() -> BsEntry {
        BsEntry {
            bias: 0,
            constant: BS_CONSTANT,
            multiplier: BS_UNIT_MULTIPLIER,
        }
    }
}

/// Writes the BS buffer `DPU_RDMA_RDMA_BS_BASE_ADDR` points at.
///
/// Required for int8: `brdma_data_use` is 7 there rather than the fp16 1, so
/// BRDMA fetches a multiplier operand alongside the bias, and `bs_mul_src`
/// makes the BS stage use it. A zeroed buffer supplies a zero multiplier and
/// produces a zero output, which is why the fp16 tests' habit of zeroing the
/// bias buffer does not carry over.
///
/// The layout is planar within a block of eight output channels and repeats
/// per block, which is not what a flat array of per-channel structs would
/// look like -- it was read off three converted models whose biases and
/// per-channel weight magnitudes were varied independently.
pub fn write_bs_buffer(buffer: &mut [u8], entries: &[BsEntry]) {
    let needed = bs_buffer_bytes(entries.len() as u32);
    assert!(
        buffer.len() >= needed,
        "BS buffer is {} bytes, needs {needed} for {} channels",
        buffer.len(),
        entries.len()
    );
    buffer[..needed].fill(0);
    for (index, entry) in entries.iter().enumerate() {
        let block = index / BS_CHANNELS_PER_BLOCK;
        let lane = index % BS_CHANNELS_PER_BLOCK;
        let base = block * BS_BLOCK_BYTES;
        let bias = base + lane * 4;
        buffer[bias..bias + 4].copy_from_slice(&entry.bias.to_le_bytes());
        let constant = base + 32 + lane * 2;
        buffer[constant..constant + 2].copy_from_slice(&entry.constant.to_le_bytes());
        let multiplier = base + 48 + lane * 2;
        buffer[multiplier..multiplier + 2].copy_from_slice(&entry.multiplier.to_le_bytes());
    }
}

/// Converts a logical quantized bias vector into Rocket's physical BS buffer.
///
/// The logical ABI is little-endian `i32` bias values in accumulator units.
/// Rocket's BS bias plane is normalized by the input and weight scales, while
/// its constant plane supplies the affine weight correction. Weight scales
/// are currently per-tensor in the executable ABI; per-channel scales can be
/// added without changing the physical writer.
pub fn pack_int8_bias_to_bs(
    dense: &[u8],
    output_channels: usize,
    padded_output_channels: usize,
    input_scale: f32,
    weights_scale: f32,
    weight_zero_point: i8,
    packed: &mut [u8],
) -> Result<usize, &'static str> {
    if dense.len() < output_channels.saturating_mul(4) {
        return Err("int8 bias is smaller than its declared shape");
    }
    if padded_output_channels < output_channels {
        return Err("padded int8 bias channels are smaller than logical channels");
    }
    let scale = f64::from(input_scale) * f64::from(weights_scale);
    if !scale.is_finite() || scale <= 0.0 {
        return Err("int8 bias scales must be finite and positive");
    }
    let needed = bs_buffer_bytes(padded_output_channels as u32);
    if packed.len() < needed {
        return Err("Rocket int8 BS destination is smaller than its declared shape");
    }
    let mut entries = vec![BsEntry::default(); padded_output_channels];
    for (channel, entry) in entries.iter_mut().take(output_channels).enumerate() {
        let offset = channel * 4;
        let bias = i32::from_le_bytes(dense[offset..offset + 4].try_into().unwrap());
        let normalized = (f64::from(bias) / scale).round();
        if normalized < f64::from(i32::MIN) || normalized > f64::from(i32::MAX) {
            return Err("int8 bias normalization overflows i32");
        }
        entry.bias = normalized as i32;
        entry.constant = -(i16::from(weight_zero_point));
    }
    write_bs_buffer(&mut packed[..needed], &entries);
    Ok(needed)
}

/// The four DMA base addresses a conv program reads and writes through.
///
/// Programs come out of this module carrying tile *offsets* in their address
/// registers, not addresses -- see [`conv_2d_tile`]. [`relocate`] binds them
/// to real memory, which is the last step before submission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Buffers {
    /// Feature data, `CNA_FEATURE_DATA_ADDR`.
    pub input: u32,
    /// Packed weights, `CNA_DCOMP_ADDR0`. See
    /// [`crate::rocket::tensor_layout::pack_hwcf_to_rocket_weights`].
    pub weights: u32,
    /// Bias, and for int8 the per-channel multiplier alongside it,
    /// `DPU_RDMA_RDMA_BS_BASE_ADDR`. See [`write_bs_buffer`].
    pub bias: u32,
    /// Output feature data, `DPU_DST_BASE_ADDR`.
    pub output: u32,
}

fn decode_identity(command: &RegCmd) -> (u32, u32) {
    ((command.0 >> 48) as u32, command.0 as u32 & 0xffff)
}

/// Binds one address register to `address`, keeping whatever tile offset the
/// program already put there.
///
/// Matching by typed register identity rather than a hardcoded command index
/// is what makes this fail loudly if a program is ever reordered or gains a
/// second write to the same address register, instead of quietly relocating
/// the wrong word.
fn relocate_one<R: RegisterMeta>(commands: &mut [RegCmd], address: u32, keep_tile_offset: bool) {
    assert_eq!(
        address & 0xf,
        0,
        "NPU DMA address for register {:#x}:{:#x} is not 16-byte aligned",
        R::DOMAIN,
        R::OFFSET
    );

    let matches: Vec<_> = commands
        .iter()
        .enumerate()
        .filter_map(|(index, command)| {
            (decode_identity(command) == (R::DOMAIN, R::OFFSET)).then_some(index)
        })
        .collect();
    assert_eq!(
        matches.len(),
        1,
        "expected exactly one {:#x}:{:#x} relocation, found {matches:?}",
        R::DOMAIN,
        R::OFFSET
    );

    // Normally add rather than overwrite: a tile program already carries its
    // own byte offset from the tensor base in these registers, exactly as the
    // vendor's own height-split programs do. Independently staged output
    // tiles are the exception and bind DPU_DST_BASE_ADDR directly.
    let tile_offset = if keep_tile_offset {
        (commands[matches[0]].0 >> 16) as u32
    } else {
        0
    };
    commands[matches[0]] = RegCmd::new(R::DOMAIN, R::OFFSET, address + tile_offset);
}

/// Binds a program's four address registers to real memory, in place.
///
/// Every address must be 16-byte aligned -- the DPU addresses output in
/// 16-byte feature atoms, and the fetch side reads them the same way, so an
/// unaligned base silently shears every surface. Panics rather than
/// truncating.
///
/// A program carries exactly one write of each of the four registers, so
/// relocating twice would double the offsets; this is a one-shot step on a
/// freshly built program, not something to reapply.
pub fn relocate(commands: &mut [RegCmd], buffers: Buffers) {
    relocate_one::<CnaFeatureDataAddr>(commands, buffers.input, true);
    relocate_one::<CnaDcompAddr0>(commands, buffers.weights, true);
    relocate_one::<DpuRdmaBsBaseAddr>(commands, buffers.bias, true);
    relocate_one::<DpuDstBaseAddr>(commands, buffers.output, true);
}

/// Binds one program from [`ConvPlan::staged_accumulator_programs`]: input,
/// weights and bias keep their tile offsets, and the output is placed at
/// `buffers.output + tile.scratch_offset`, the tile's own contiguous range.
pub fn relocate_staged_accumulator(
    commands: &mut [RegCmd],
    buffers: Buffers,
    tile: &AccumulatorOutputTile,
) {
    let local_output = buffers
        .output
        .checked_add(
            u32::try_from(tile.scratch_offset)
                .expect("accumulator tile scratch offset exceeds u32"),
        )
        .expect("accumulator tile DMA address overflow");
    relocate_with_exact_output(
        commands,
        Buffers {
            output: local_output,
            ..buffers
        },
    );
}

/// Binds a program while replacing its output tile offset with an exact DMA
/// address.
///
/// This is for callers that stage each output tile in a separate contiguous
/// scratch range. Input, weight, and bias addresses retain the tile offsets
/// carried by the program; only `buffers.output` is used verbatim.
pub fn relocate_with_exact_output(commands: &mut [RegCmd], buffers: Buffers) {
    relocate_one::<CnaFeatureDataAddr>(commands, buffers.input, true);
    relocate_one::<CnaDcompAddr0>(commands, buffers.weights, true);
    relocate_one::<DpuRdmaBsBaseAddr>(commands, buffers.bias, true);
    relocate_one::<DpuDstBaseAddr>(commands, buffers.output, false);
}

/// 2-3 and 4-6 are alternative two- and three-core height-split programs,
/// not continuations of this command stream.
pub fn conv_2d(kernels: Kernels) -> Vec<RegCmd> {
    let shape = Shape::CAPTURED;
    conv_2d_tile(shape, kernels, &Tile::whole(shape, kernels))
}

/// Builds the single-core regcmd program for one output-row `tile`.
///
/// `conv_2d(k)` is `conv_2d_tile(k, &Tile::whole(k))` and reproduces captured
/// group 1 bit for bit. For a partial tile, the sixteen tile-dependent
/// registers take the values derived from the cross-group capture diff; see
/// the module documentation for what a tile program does and does not match.
///
/// The feature and destination address registers carry this tile's *offset*
/// from the tensor base, exactly as the vendor's own split programs do. A
/// caller relocating these must add the buffer's DMA address to the existing
/// value rather than overwrite it.
pub fn conv_2d_tile(shape: Shape, kernels: Kernels, tile: &Tile) -> Vec<RegCmd> {
    conv_2d_tile_with_grains(shape, kernels, tile, feature_grains(kernels, tile))
}

/// [`feature_grains`], with the characterization override applied.
///
/// Only `ConvPlan` uses this. The override is gated on `Cin` so a probe can
/// change the shapes under study without disturbing the health canary the
/// hardware harness runs first -- forcing a global value breaks known-good
/// shapes, which is itself evidence that the prefetch is not a free parameter.
fn feature_grains_planned(shape: Shape, kernels: Kernels, tile: &Tile) -> u32 {
    let programmed = feature_grains(kernels, tile);
    if shape.in_channels < grains_override_min_channels() {
        return programmed;
    }
    match grains_override() {
        Some(GrainsOverride::Exact(value)) => value,
        Some(GrainsOverride::Cap(cap)) => programmed.min(cap),
        None => programmed,
    }
}

/// Lowest `Cin` the grains override applies to (`ROCKET_FEATURE_GRAINS_MIN_CIN`,
/// default 0).
fn grains_override_min_channels() -> u32 {
    std::env::var("ROCKET_FEATURE_GRAINS_MIN_CIN")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}

/// Test-only override of the programmed prefetch. See [`grains_override`].
enum GrainsOverride {
    Exact(u32),
    Cap(u32),
}

/// Lets a hardware probe drive `feature_grains` through the *whole* ConvPlan
/// path rather than hand-building one tile, which is what
/// `conv_2d_tile_with_grains` already allows for a single tile.
///
/// This exists because our value and the vendor's disagree systematically --
/// ours is rows-driven and stays near 33, while the vendor drives it down with
/// channel pressure (33, 16, 9, ... 6) -- and no gate test compares the field.
/// `ROCKET_FEATURE_GRAINS=<n>` pins it; `ROCKET_FEATURE_GRAINS_MAX=<n>` clamps
/// it. Nothing on the compiled path sets either.
fn grains_override() -> Option<GrainsOverride> {
    if let Ok(value) = std::env::var("ROCKET_FEATURE_GRAINS")
        && let Ok(parsed) = value.parse()
    {
        return Some(GrainsOverride::Exact(parsed));
    }
    if let Ok(value) = std::env::var("ROCKET_FEATURE_GRAINS_MAX")
        && let Ok(parsed) = value.parse()
    {
        return Some(GrainsOverride::Cap(parsed));
    }
    None
}

/// Characterization override for the accumulator `DPU_SURFACE_ADD.surf_add`.
///
/// Exceeding the per-channel coefficient limit raises a DMA **read** error and
/// stalls the rk_iommu, and the accumulator output registers are the only part
/// of the program with no vendor capture behind them. Nothing on the compiled
/// path sets this.
///
/// Gated by `ROCKET_ACC_SURF_ADD_MIN_CIN` because the hardware harness runs a
/// low-`Cin` accumulator canary first: overriding globally breaks that canary
/// and the run aborts as "device sick" rather than measuring anything. That
/// the canary breaks at all is itself the finding that 16 is load-bearing.
fn accumulator_surf_add_override(in_channels: u32) -> Option<u32> {
    let min_channels: u32 = std::env::var("ROCKET_ACC_SURF_ADD_MIN_CIN")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    if in_channels < min_channels {
        return None;
    }
    std::env::var("ROCKET_ACC_SURF_ADD")
        .ok()
        .and_then(|value| value.parse().ok())
}

/// Characterization overrides scoped to [`Precision::Int4`].
///
/// int4's compute is exact on hardware but its write-out stops after two
/// 256-byte atoms, which is a writer-geometry question -- the same class the
/// int8 accumulator's `mc_surf_out` / `size_e` / `surf_add` triple turned
/// out to be. Scoping these to int4 keeps every other precision, including
/// the hardware harness's health canary, on the shipped program, so a sweep
/// measures int4 rather than breaking the instrument.
///
/// `ROCKET_INT4_{OUT_PRECISION,MC_SURF_OUT,SIZE_E,SURF_ADD}`. Nothing on the
/// compiled path sets any of them.
fn int4_override(precision: Precision, name: &str) -> Option<u32> {
    if precision != Precision::Int4 {
        return None;
    }
    std::env::var(format!("ROCKET_INT4_{name}"))
        .ok()
        .and_then(|value| value.parse().ok())
}

/// Characterization override for `DPU_BS_OW_CFG.SIZE_E_0/1/2`
/// (`ROCKET_ACC_SIZE_E`, gated by `ROCKET_ACC_SIZE_E_MIN_CIN`).
///
/// Kept, like [`accumulator_surf_add_override`], because it carries a settled
/// negative result rather than an open question. Nothing on the compiled path
/// sets it. The `_MIN_CIN` gate exists so the harness's low-`Cin` accumulator
/// canary stays on the shipped program; without it a sweep aborts as "device
/// sick" instead of measuring anything.
///
/// **What it settled** [HW sweep, planck 2026-09-03]. `size_e` is a BS/OW-stage
/// field and the int32-accumulator path bypasses that stage, so the override is
/// inert there: 0, 1, 3 and 7 all produce a byte-identical, bit-exact result at
/// 32x32 Cin=384, and all four produce the identical 6144+512-byte truncation at
/// Cin=385. It is emphatically *not* inert on the requantized int8 path, which
/// leaves `OD_BYPASS` clear -- 3 and 7 there write 1024 of 65536 bytes and hang
/// the job. See [`Shape::bs_ow_size_e`] for the table.
///
/// So `rockchip-npu-notes/encodings/size-e-quirk.md`'s "integer outputs stride
/// as `size_e = 7` regardless of byte width" is a fact about a path that keeps
/// the OW stage engaged, and does not carry to this one.
///
/// **Refined 2026-09-05** by [`od_engage`], which engages that stage on any
/// path that bypasses it. Clearing `od_bypass` always makes `size_e` live;
/// `bs_bypass` has nothing to do with it (with the BS plane clocked and
/// `od_bypass` still 1, `size_e = 7` is still inert, bit-exact at 32x32 Cin 64
/// Cout 128 k1). The converse does **not** hold -- `od_bypass = 1` does not
/// imply inert, see [`Precision::Int4`] below, which is load-bearing there.
///
/// With the stage live, one value works and every other one stalls the writer
/// into a ~530 ms watchdog kill. Swept 0..7 at 32x32 Cin 128 k1
/// (`accumulator_size_e_probe`, `ROCKET_ACC_OD_ENGAGE=1`, canary healthy
/// throughout, `past_end` 0 everywhere) plus a depthwise arm through
/// `conv_depthwise_hw` [HW sweep, planck 2026-09-05]:
///
/// | output | bytes | required `size_e` |
/// |---|---|---|
/// | dense fp16 | 2 | 1 |
/// | dense fp32 | 4 | 3 |
/// | dense int8, requantized | 1 | 1 |
/// | dense int32, accumulator | 4 | **7** |
/// | int16 from int4 operands | 2 | **7** |
/// | depthwise fp16 | 2 | 3 |
///
/// **Every value this function returns is confirmed correct**, including the
/// two that looked like exceptions. There is no single arithmetic rule; the
/// value is a property of the writer, in four groups:
///
/// * **Float output**: `bytes - 1`, the natural stride.
/// * **The raw integer accumulator writer**: a fixed **7** whatever the output
///   width -- 4-byte int32 and 2-byte int16 both. This is exactly the notes'
///   quirk and its mental model (the DPU casts its wide accumulator and writes
///   it with one fixed integer geometry), now confirmed with the stage live
///   rather than inferred.
/// * **Requantized int8**: 1, which is neither the float rule (0) nor the
///   accumulator's 7.
/// * **Depthwise**: 3 at fp16, where dense fp16 is 1 -- its own geometry.
///
/// What the sweep *does* refute is the vendor register doc's "number of
/// 8-channel groups in a row, minus 1" reading: the value is Cout-independent.
/// Cout 16 still requires 7 on the accumulator path, where `Cout/8 - 1` would
/// be 1 and 1 writes 512 of 131072 bytes. Every earlier measurement used Cout
/// 64, where `Cout/8 - 1` is also 7, so the two readings had never been
/// separated. The accumulator
/// truncation in [`MAX_ACCUMULATOR_COEFFICIENT_BYTES_PER_CHANNEL`] is still
/// unexplained, and this is one more register eliminated: with `size_e` inert
/// and `surf_add` swept, the output-side register archaeology is exhausted.
fn bs_ow_size_e_override(in_channels: u32) -> Option<u32> {
    let min_channels: u32 = std::env::var("ROCKET_ACC_SIZE_E_MIN_CIN")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    if in_channels < min_channels {
        return None;
    }
    std::env::var("ROCKET_ACC_SIZE_E")
        .ok()
        .and_then(|value| value.parse().ok())
}

/// Characterization override for `DPU_BS_CFG.BS_BYPASS` on the int32
/// accumulator path (`ROCKET_ACC_BS_ENGAGE=1`, or
/// [`set_accumulator_bs_engage`] for in-process arms).
///
/// **What it is for.** ISSUES.md C8: an `Int8Accumulator` job poisons the core
/// it ran on and a following wide fp16 job hangs, while the same shapes as
/// requantized [`Precision::Int8`] or [`Precision::Fp16`] poison nothing. A
/// register-program diff of one shape at all three precisions (32x32 Cin 64
/// Cout 128 1x1, `examples/dump_conv_plan_regcmd.rs`) found the three
/// precisions write the *identical* register set -- so nothing is inherited
/// stale -- and exactly four registers carry a value unique to the poisoner:
/// `DATA_FORMAT.out_precision` (4 = int32), this `BS_CFG.bs_bypass`,
/// `BS_OW_CFG` (`size_e`, already shown inert here by
/// [`bs_ow_size_e_override`], and `od_bypass`) and `SURFACE_ADD`.
///
/// `out_precision` and `bs_bypass` are confounded, because accumulator mode
/// sets both. This separates them: with the override on, the BS plane is
/// *engaged* but both of its arithmetic sub-stages are bypassed, so the stage
/// is clocked and drained while the result is bit-for-bit what it was. A run
/// that still poisons indicts the int32 writer itself; one that stops
/// poisoning indicts the bypassed plane, and makes the fix a one-bit change
/// instead of a compiler path.
///
/// Nothing on the compiled path sets it.
fn accumulator_bs_engage() -> bool {
    ACCUMULATOR_BS_ENGAGE.load(std::sync::atomic::Ordering::Relaxed)
        || std::env::var("ROCKET_ACC_BS_ENGAGE").is_ok_and(|value| value != "0")
}

/// Characterization override for `DPU_BS_OW_CFG.OD_BYPASS`: engages the output
/// converter on any path that would bypass it -- the int32 accumulator, and
/// fp16/fp32, whose `size_e` is otherwise untestable
/// (`ROCKET_ACC_OD_ENGAGE=1`, or [`set_od_engage`]).
///
/// The companion to [`accumulator_bs_engage`], and the one combination C8 had
/// not reached: `out_precision = 4` with the **OW stage engaged**. The
/// accumulator path is the only poisoner and it runs `od_bypass = 1`, but so
/// do two clean paths (fp16, fp16-f32out), so the field is already eliminated
/// as a *discriminator*; this asks the different question of whether the int32
/// writer still poisons when the output converter is in the path.
///
/// **Hazard.** `od_bypass = 0` is what makes `size_e` live -- on the
/// requantized path a `size_e` of 3 or 7 there writes 1024 of 65536 bytes and
/// hangs the job. The accumulator path's own `size_e` is 7, so pair this with
/// `ROCKET_ACC_SIZE_E` and always run it behind `ROCKET_PAD_OUTPUT`: a wider
/// stride can push the write past the allocation, fault, stall the rk_iommu
/// and wedge the board until a reboot.
///
/// Nothing on the compiled path sets it.
fn od_engage() -> bool {
    ACCUMULATOR_OD_ENGAGE.load(std::sync::atomic::Ordering::Relaxed)
        || std::env::var("ROCKET_ACC_OD_ENGAGE").is_ok_and(|value| value != "0")
}

/// Characterization override for `DPU_BS_OW_CFG.OW_SRC` (`ROCKET_OW_SRC=0|1`).
///
/// The field selects where the CPEND stage's operand comes from: 0 = the
/// `DPU_BS_OW_OP` configuration register, 1 = "from outside". This crate sets
/// it to 1 whenever a quantization is present, but neither `rocket-userspace`
/// nor the Mesa program it was diffed against ever sets bit 0 -- both emit
/// `BS_OW_CFG` as `tp_org_en | size_e_2 | size_e_1 | size_e_0 | od_bypass`
/// with no `ow_src` term.
///
/// **Checked on hardware 2026-09-05, and our 1 is required.** `conv_int8_hw`
/// passes 4/4 at the shipped value and fails **3 of 4** at `ROCKET_OW_SRC=0`.
/// The two stacks feed the CPEND operand from different places and each is
/// self-consistent: `rocket-userspace` leaves `ow_src` at 0 and supplies the
/// operand in the `DPU_BS_OW_OP` configuration register (`0x80 - weight_zp`
/// on its depthwise branch), while this crate writes `DPU_BS_OW_OP = 0` and
/// takes it from BRDMA, which the requantized path already loads with the
/// bias/scale/shift triple (`BRDMA_DATA_USE_QUANTIZED`). So the divergence is
/// a design choice, not a defect, and the two settings are not
/// interchangeable.
///
/// On the accumulator path CPEND is bypassed and the field is inert (0
/// mismatches either way at 32x32 Cin 128 Cout 64 k1), so the `1` it gets
/// there is merely unused.
///
/// Nothing on the compiled path sets it.
fn ow_src_override() -> Option<u32> {
    if acc_vendor_part("owsrc") {
        return Some(0);
    }
    if cpend_from_configuration() {
        return Some(0);
    }
    std::env::var("ROCKET_OW_SRC")
        .ok()
        .and_then(|value| value.parse().ok())
}

/// `ROCKET_CPEND=mesa` adopts the vendor stack's CPEND wiring wholesale:
/// `ow_src = 0` plus the operand in `DPU_BS_OW_OP` as `0x80 - weight_zero_point`.
///
/// The two halves have to move together -- see [`ow_src_override`] -- so this
/// exists rather than making a caller set both knobs consistently. Note that
/// Mesa only ever engages CPEND on its **depthwise** path; its direct-conv
/// programs all set `od_bypass = 1`, so on a dense shape this is an adoption
/// experiment with no vendor ground truth behind it.
///
/// **Tried on hardware 2026-09-05 and rejected.** Moving both fields together
/// does work where moving `ow_src` alone did not -- `conv_int8_hw` 4/4,
/// `conv_int8_map_hw` 6/6, `conv_depthwise_int8_exact_hw` 3/3, and depthwise
/// fp16 with or without `od_bypass` cleared. The operand is genuinely
/// load-bearing (sweeping `ROCKET_BS_OW_OP` with `ow_src = 0`: 127 and 128
/// pass 4/4, 129 passes 3/4, and 0, 64, 192, 255 pass 1/4).
///
/// It fails on **affine int8**, and structurally rather than by a value:
/// `conv_int8_vendor_affine_hw` cycles a per-output-channel weight zero point
/// of `[-127, -43, 0, 42, 125]`, so `0x80 - weight_zp` is `[255, 171, 128, 86,
/// 3]` -- five different operands where `DPU_BS_OW_OP` is one 16-bit scalar.
/// No value of it passes (swept 0, 3, 85, 128, 171, 253, 255: 0/1 every time).
/// BRDMA can carry a per-channel operand and a configuration register cannot,
/// so this crate's wiring is strictly the more general of the two and Mesa's
/// is the uniform-zero-point special case. Keep `ow_src = 1`.
fn cpend_from_configuration() -> bool {
    std::env::var("ROCKET_CPEND").is_ok_and(|value| value == "mesa")
}

/// Characterization override for `DPU_BS_OW_OP.ow_op`, the CPEND operand when
/// `ow_src = 0` (`ROCKET_BS_OW_OP=<value>`; `ROCKET_CPEND=mesa` supplies the
/// vendor's `0x80 - weight_zero_point` instead). This crate ships 0 because it
/// feeds the operand from BRDMA.
fn bs_ow_op_value(quantization: Option<&Quantization>) -> u32 {
    if let Some(value) = std::env::var("ROCKET_BS_OW_OP")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
    {
        return value & 0xffff;
    }
    if cpend_from_configuration() {
        let weight_zero_point = quantization.map_or(0, |q| q.weight_zero_point);
        return (0x80i64 - i64::from(weight_zero_point)) as u32 & 0xffff;
    }
    0
}

/// `ROCKET_ACC_VENDOR=<parts>` makes the int32-accumulator program match
/// `rocket-userspace`'s `gen_conv2d_int8` field by field, for ISSUES.md C8.
///
/// A register diff of the two emitters at 32x32 Cin 64 Cout 128 k1 found the
/// same 126 registers and only ten differing values, three of them DMA
/// addresses. Of the rest, this crate turns the **requantization datapath on**
/// while writing the raw int32 accumulator -- `qd_en = 1`, BRDMA fetching the
/// bias/scale/shift triple, a BS MUL shift of 14 and the BS sub-stages left
/// un-bypassed -- and the vendor never emits that combination. Its own header
/// states the rule: "int8_out=0 (default) keeps the validated int32-raw
/// datapath (qd_en=0, size_e=7/surf*8, int32 output, host requant); int8_out=1
/// switches to Mesa's int8-output writer: QD_EN=1 ...".
///
/// Parts are comma-separated so the difference can be bisected: `qd`, `brdma`,
/// `bs`, `owsrc`, or `all`. Nothing on the compiled path sets it.
/// `ROCKET_ACC_BRDMA=1` puts the accumulator path's `brdma_data_use` back to
/// the pre-C8-fix value, so the hardware test can still reproduce the hang.
fn acc_brdma_restore() -> bool {
    std::env::var("ROCKET_ACC_BRDMA").is_ok_and(|value| value == "1")
}

fn acc_vendor_part(part: &str) -> bool {
    static PARTS: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    let parts = PARTS.get_or_init(|| {
        std::env::var("ROCKET_ACC_VENDOR")
            .unwrap_or_default()
            .split(',')
            .map(|value| value.trim().to_ascii_lowercase())
            .filter(|value| !value.is_empty())
            .collect()
    });
    parts.iter().any(|value| value == part || value == "all")
}

/// Turns [`od_engage`] on or off for this process; see
/// [`set_accumulator_bs_engage`].
pub fn set_od_engage(engaged: bool) {
    ACCUMULATOR_OD_ENGAGE.store(engaged, std::sync::atomic::Ordering::Relaxed);
}

/// Turns [`accumulator_bs_engage`] on or off for this process, for a test that
/// needs to alternate it between arms (edition 2024 makes `set_var` unsafe, and
/// the hardware arms run in one process on purpose).
pub fn set_accumulator_bs_engage(engaged: bool) {
    ACCUMULATOR_BS_ENGAGE.store(engaged, std::sync::atomic::Ordering::Relaxed);
}

/// Characterization override selecting rocket-userspace's `surf_add` *rule*
/// rather than a constant (`ROCKET_ACC_SURF_MULT`, gated by
/// `ROCKET_ACC_SIZE_E_MIN_CIN`).
///
/// `gen_matmul_int8` sets `surf_add = dst_surf_stride * 8` with
/// `dst_surf_stride = dataout_height * dataout_width` **of the task**. On a
/// height-tiled plan every tile has a different `out_rows`, so no single
/// `ROCKET_ACC_SURF_ADD` constant can reproduce it -- which is why the constant
/// sweep recorded in `accumulator-per-channel-coefficient-limit` could not have
/// found this even in principle. This applies the rule per tile.
fn accumulator_surf_mult_override(in_channels: u32) -> Option<u32> {
    let min_channels: u32 = std::env::var("ROCKET_ACC_SIZE_E_MIN_CIN")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    if in_channels < min_channels {
        return None;
    }
    std::env::var("ROCKET_ACC_SURF_MULT")
        .ok()
        .and_then(|value| value.parse().ok())
}

/// Surface multiplier for dense int32-accumulator output: `surf_add =
/// dataout_width * dataout_height * 8`, per task.
///
/// The 8 is the integer-output stride quirk, the same one behind
/// [`Shape::bs_ow_size_e`]'s 7: the writer strides as if each output element
/// were 8 bytes even though an int32 is 4. HW-validated here at 32x32 Cin 384
/// [planck 2026-09-03] -- mult 8 writes 100% of the buffer bit-exactly, and
/// 4 / 2 / 1 write 75% / 62.5% / 56.2%, leaving the rest at the poison
/// sentinel, exactly as `rocket-userspace`'s `gen_matmul_int8` header warns
/// ("halves the surface stride, leaving every output column past the first few
/// surfaces as the `0xAA` sentinel").
const DENSE_ACCUMULATOR_SURF_MULT: u32 = 8;

/// Characterization override for `DPU_DATA_FORMAT.mc_surf_out`
/// (`ROCKET_ACC_MC_SURF_OUT`, gated by `ROCKET_ACC_SIZE_E_MIN_CIN`).
///
/// The third knob of the accumulator output writer, and the one that makes the
/// other two readable. `rocket-userspace/include/npu_dpu.h` documents the field
/// as `0 = 16B/pixel one surface, 1 = 2/4 surf serial`, and its HW-validated
/// int8 -> int32 matmul (`gen_matmul_int8`) leaves it **0** while using
/// `size_e = 7` and `surf_add = dst_surf_stride * 8`. This crate's accumulator
/// mode instead sets it to **1** with `size_e = 1` and `surf_add = 16`, which is
/// a different writer, not a variant of the same one.
///
/// That is why sweeping `surf_add` alone and `size_e` alone both read as "no
/// effect / no value helps": in the serial writer there is no surface stride for
/// either field to describe. The three move together or not at all.
fn accumulator_mc_surf_out_override(in_channels: u32) -> Option<u32> {
    let min_channels: u32 = std::env::var("ROCKET_ACC_SIZE_E_MIN_CIN")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    if in_channels < min_channels {
        return None;
    }
    std::env::var("ROCKET_ACC_MC_SURF_OUT")
        .ok()
        .and_then(|value| value.parse().ok())
}

/// Builds a tile program with an explicit `feature_grains`, for probing which
/// values the hardware accepts. Prefer [`conv_2d_tile`].
pub fn conv_2d_tile_with_grains(
    shape: Shape,
    kernels: Kernels,
    tile: &Tile,
    feature_grains: u32,
) -> Vec<RegCmd> {
    assert_default_cbuf_kernel(kernels);
    let tile = Tile2D {
        rows: *tile,
        columns: ColumnTile::whole(shape, kernels),
    };
    conv_2d_tile_program(
        shape,
        kernels,
        &tile,
        feature_grains,
        shape.data_banks(kernels),
        shape.weight_banks(kernels),
        OutputPlacement::SharedImage,
    )
}

/// Builds a large-kernel program with an explicit CBUF partition.
///
/// The focused kernel sweep shows that 7x7 switches away from demand once
/// coefficient demand exceeds seven banks, while 9x9 and 11x11 use their own
/// streaming schedules. This entry point keeps unresolved policy out of
/// [`conv_2d_tile`]. Prefer [`ConvPlan`] when its capture-backed policy covers
/// the operation; this entry point is the low-level override. The two bank
/// counts must be nonzero and sum to the RK3588's twelve CBUF banks.
pub fn conv_2d_tile_with_cbuf_banks(
    shape: Shape,
    kernels: Kernels,
    tile: &Tile,
    data_banks: u32,
    weight_banks: u32,
) -> Vec<RegCmd> {
    assert!(
        data_banks > 0 && weight_banks > 0 && data_banks + weight_banks == CBUF_BANKS,
        "explicit CBUF partition must have nonzero data and weight banks summing to {CBUF_BANKS}; \
         got data={data_banks}, weights={weight_banks}"
    );
    let tile = Tile2D {
        rows: *tile,
        columns: ColumnTile::whole(shape, kernels),
    };
    assert!(
        tile.rows.in_rows <= shape.max_tile_input_rows_for_data_banks(data_banks),
        "tile reads {} input rows, but {data_banks} data banks fit at most {} for this shape",
        tile.rows.in_rows,
        shape.max_tile_input_rows_for_data_banks(data_banks),
    );
    conv_2d_tile_program(
        shape,
        kernels,
        &tile,
        feature_grains(kernels, &tile.rows),
        data_banks,
        weight_banks,
        OutputPlacement::SharedImage,
    )
}

/// Builds one rectangular large-kernel tile with an explicit CBUF split.
///
/// Horizontal tiles use the capture-derived grouped-line DMA mode and retain
/// the full tensor strides. Prefer [`conv_2d_tile_with_cbuf_banks`] for a
/// full-width row tile.
pub fn conv_2d_tile_2d_with_cbuf_banks(
    shape: Shape,
    kernels: Kernels,
    tile: &Tile2D,
    data_banks: u32,
    weight_banks: u32,
) -> Vec<RegCmd> {
    assert_eq!(
        shape.layout(),
        FeatureLayout::Surfaces,
        "horizontal tiling currently has capture backing only for NC1HWC2 surfaces"
    );
    assert_eq!(
        shape.stride, 1,
        "horizontal tiling currently has capture backing only at stride 1"
    );
    assert!(
        data_banks > 0 && weight_banks > 0 && data_banks + weight_banks == CBUF_BANKS,
        "explicit CBUF partition must have nonzero data and weight banks summing to {CBUF_BANKS}; \
         got data={data_banks}, weights={weight_banks}"
    );
    let max_rows =
        shape.max_tile_input_rows_for_width_and_data_banks(tile.columns.in_cols, data_banks);
    assert!(
        tile.rows.in_rows <= max_rows,
        "tile reads {}x{} input pixels, but {data_banks} data banks fit at most \
         {max_rows} rows at this width",
        tile.columns.in_cols,
        tile.rows.in_rows,
    );
    conv_2d_tile_program(
        shape,
        kernels,
        tile,
        feature_grains(kernels, &tile.rows),
        data_banks,
        weight_banks,
        OutputPlacement::SharedImage,
    )
}

fn conv_2d_tile_program(
    shape: Shape,
    kernels: Kernels,
    tile: &Tile2D,
    feature_grains: u32,
    data_banks: u32,
    weight_banks: u32,
    output_placement: OutputPlacement,
) -> Vec<RegCmd> {
    let padded_channels = shape.padded_channels();
    let weight_channels = shape.weight_channels();
    // The DPU counts output channels in whole granules while the CNA counts
    // the real kernels. Both appear below, and they differ at every Cout
    // that is not already a multiple of the granule.
    let padded_out_channels = shape.padded_out_channels();
    let (bn_bypass, bn_relu_bypass, bn_relux_en, bn_relux_cmp) = bn_programming(shape.activation);

    // Precision reaches the program in three ways: an enum replicated across
    // eight fields in four blocks, a set of bypasses that the quantized path
    // clears, and the requantization constants themselves.
    // The CNA and CORE stages take the front-of-pipe code; the DPU stages
    // take their own enum, which differs from it for tf32 alone.
    let precision = data_precision(shape.precision);
    let dpu_precision: Bits<3> = dpu_data_precision(shape.precision).into();
    let quantization = shape.precision.quantization();
    let accumulator_output = shape.precision.writes_accumulators();
    let output_precision: Bits<3> = match int4_override(shape.precision, "OUT_PRECISION") {
        Some(value) => Bits::new(value),
        None => output_data_precision(shape.precision).into(),
    };
    // See [`accumulator_bs_engage`]: the C8 probe's BS pass-through.
    let bs_passthrough = accumulator_output && accumulator_bs_engage();
    let bs_vendor = accumulator_output && acc_vendor_part("bs");
    // `BS_MUL_SHIFT_VALUE` and its negated twin in `DPU_DATA_FORMAT` are a
    // constant 14 in every int8 capture and 0 in every fp16 one. Nothing in
    // the corpus varies it, so it is not derived from anything.
    //
    // The pass-through has to zero it. `bs_mul_bypass` bypasses the *multiply*
    // and not the shift that follows it, so a BS plane engaged with the
    // shipped 14 right-shifts every accumulator by 14 -- measured, and it is
    // what made the first pass-through arm return an all-zero buffer.
    let bs_mul_shift = if quantization.is_some()
        && !bs_passthrough
        && !(accumulator_output && acc_vendor_part("bs"))
    {
        BS_MUL_SHIFT_VALUE
    } else {
        0
    };
    // BRDMA carries bias alone at fp16 and the full bias/scale/shift triple
    // once requantization is active.
    // **ISSUES.md C8's root cause.** The int32-accumulator path bypasses the BS
    // plane, so the bias/scale/shift triple BRDMA would fetch has no consumer --
    // and leaving the fetch enabled anyway is what poisons the NPU core, making
    // a following wide fp16 job hang until the power domain cycles. Bisected
    // against `rocket-userspace`'s `gen_conv2d_int8`, which leaves
    // `DPU_RDMA_BRDMA_CFG` at 0 here and does not poison: of the seven
    // non-address fields the two emitters disagree on, this one alone accounts
    // for it. `ROCKET_ACC_BRDMA=1` restores the old value, which is how the
    // hardware test still reproduces the hang on demand.
    let brdma_data_use = if accumulator_output && !acc_brdma_restore() {
        BRDMA_DATA_USE_NONE
    } else if quantization.is_some() {
        BRDMA_DATA_USE_QUANTIZED
    } else {
        BRDMA_DATA_USE_BIAS
    };
    // Bits, not bytes: a half-byte element cannot express a per-kernel
    // footprint as a byte count times a channel count.
    let element_bits = shape.precision.element_bits();

    let rows = &tile.rows;
    let columns = &tile.columns;
    let full_width = shape.width;
    let height = shape.height;
    let full_out_width = shape.output_width(kernels);
    let out_height = shape.output_height(kernels);
    let input_width = columns.in_cols;
    let out_width = columns.out_cols;
    assert!(
        input_width <= shape.max_tile_input_width(),
        "tile reads {input_width} input columns at {} feature atoms per pixel; the last entry \
         slab would begin at entry {}, past the {MAX_ENTRY_SLAB_BASE} the CBUF can address, \
         and the hardware reads it from the front of the line instead (ISSUES.md C10)",
        shape.cbuf_atoms(),
        (shape.cbuf_atoms().div_ceil(CBUF_ATOMS_PER_ENTRY) - 1) * input_width,
    );
    let horizontally_tiled = columns.out_first != 0 || out_width != full_out_width;
    let (output_base_offset, output_surface_pixels, output_notch) = match output_placement {
        OutputPlacement::SharedImage => (
            tile.output_offset(shape, kernels),
            full_out_width * out_height,
            full_out_width - out_width,
        ),
        OutputPlacement::ContiguousTile => {
            assert!(
                accumulator_output,
                "contiguous tile output is only validated for Int8Accumulator precision"
            );
            (0, out_width * rows.out_rows, 0)
        }
    };

    assert!(
        feature_grains <= MAX_FEATURE_GRAINS,
        "tile requires {feature_grains} feature grains; CNA_CONV_CON2.feature_grains encodes at most {MAX_FEATURE_GRAINS}"
    );

    assert!(
        rows.out_rows > 0 && rows.out_first + rows.out_rows <= out_height,
        "tile output rows {}..{} fall outside the {out_height}-row output",
        rows.out_first,
        rows.out_first + rows.out_rows
    );
    assert!(
        columns.out_cols > 0 && columns.out_first + columns.out_cols <= full_out_width,
        "tile output columns {}..{} fall outside the {full_out_width}-column output",
        columns.out_first,
        columns.out_first + columns.out_cols
    );
    let charged_input_width = shape.cbuf_input_width(input_width);
    assert!(
        rows.in_rows * charged_input_width <= shape.max_data_entries(),
        "tile reads {}x{} charged pixels; CNA_CBUF_CON1.data_entries holds at most {} \
         at {:?}",
        charged_input_width,
        rows.in_rows,
        shape.max_data_entries(),
        shape.precision,
    );
    assert!(
        rows.in_rows > 0 && rows.in_first + rows.in_rows <= height,
        "tile input rows {}..{} fall outside the {height}-row image",
        rows.in_first,
        rows.in_first + rows.in_rows
    );
    assert!(
        columns.in_cols > 0 && columns.in_first + columns.in_cols <= full_width,
        "tile input columns {}..{} fall outside the {full_width}-column image",
        columns.in_first,
        columns.in_first + columns.in_cols
    );

    let kernel = shape.kernel_programming(kernels);

    // Layout-dependent programming. Dense rows are counted in pixels and the
    // whole tile is resident, so `data_entries` scales with the tile height.
    // Surfaces are counted in atoms and `data_entries` does not depend on the
    // tile at all -- the same field carries different quantities in the two
    // regimes, which is why they are computed apart rather than parameterised.
    //
    // The surface `data_entries` charge packs 4 atoms per entry and rounds
    // *up*: vendor captures at width 13/29/30/31 (Cin=8, one atom/pixel)
    // program 4/8/8/8 respectively, not the 3/7/7/7 floor division gives.
    // Every capture before these was at a width a multiple of 4, where floor
    // and ceiling agree, which is how a real compiled model first exposed
    // this as scattered-pixel corruption on hardware.
    let (line_stride, surf_stride, data_entries) = match (shape.layout(), horizontally_tiled) {
        (FeatureLayout::Dense, _) => (
            full_width,
            full_width * (height - 1),
            charged_input_width * rows.in_rows,
        ),
        (FeatureLayout::Surfaces, false) => (
            full_width * 4,
            // This field is a 28-bit signed/bias-style encoding, despite the
            // register definition exposing it as unsigned. The original
            // image corpus never went below four rows, so ordinary unsigned
            // subtraction happened to reproduce it. A 160-model FC sweep
            // maps M to width and uses height=1; RKNN then writes
            // `M * (1 - 4)` modulo 2^28 (M=4 is 0x0fff_fff4).
            //
            // Keep the arithmetic wrapping here and let the typed register
            // field mask it to 28 bits below. This is the vendor encoding,
            // not an attempt to use a negative byte stride in host memory.
            full_width.wrapping_mul(height.wrapping_sub(4)) & 0x0fff_ffff,
            (input_width * shape.cbuf_atoms()).div_ceil(CBUF_ATOMS_PER_ENTRY),
        ),
        // Every captured width-partitioned task enables grouped-line mode
        // below and switches to these strides. `surf_stride` is the full
        // surface area less the local input width: exact for 125/139 at 9x9,
        // 124/142 at 11x11/Cin48, and 40/64 at 11x11/Cin64.
        (FeatureLayout::Surfaces, true) => (
            full_width,
            full_width * height - input_width,
            (input_width * shape.cbuf_atoms()).div_ceil(CBUF_ATOMS_PER_ENTRY),
        ),
    };

    let weight_bytes_per_kernel = kernel.height * kernel.width * weight_channels * element_bits / 8;
    let weight_bytes = shape.weight_bytes(kernels);
    let mut commands = Vec::with_capacity(136);

    // CNA preamble, followed by the DPU/DPU_RDMA ping-pong pointers.
    let mut cbuf_con0 = Register::<CnaCbufCon0>::new();
    cbuf_con0
        .weight_bank(Bits::new(weight_banks))
        .data_bank(Bits::new(data_banks));
    commands.push(cbuf_con0.build());
    commands.push(zero::<CnaDcompRegnum>());
    commands.push(zero::<CnaDcompCtrl>());

    // The dense regime is the CNA's ARGB image-input path: `argb_in` names
    // the channel count (OneChannel = 8 through FourChannels = 11) and both
    // `nonalign_dma` and `group_line_off` are set. The full-width surface
    // regime clears all three. Horizontal surface tiles set `group_line_off`
    // while leaving the other two clear, exactly as every width-partitioned
    // task in the focused kernel captures does.
    //
    // Leaving these at the captured C3 values made the hardware read three
    // channels per pixel at every channel count, which is what
    // `conv_multichannel_hw` caught.
    let mut conv_con1 = Register::<CnaConvCon1>::new();
    match shape.layout() {
        FeatureLayout::Dense => {
            conv_con1
                .nonalign_dma(Bits::new(1))
                .group_line_off(Bits::new(1))
                .argb_in(argb_input_mode(shape.in_channels).into());
        }
        FeatureLayout::Surfaces => {
            conv_con1
                .nonalign_dma(Bits::new(0))
                .group_line_off(Bits::new(u32::from(horizontally_tiled)))
                .argb_in(Bits::new(0));
        }
    }
    conv_con1
        .proc_precision(precision.into())
        .in_precision(precision.into())
        .conv_mode(Bits::new(conv_mode(shape)));
    commands.push(conv_con1.build());
    commands.push(
        Register::<DpuSPointer>::new()
            .pointer_pp_mode(Bits::new(1))
            .executer_pp_en(Bits::new(1))
            .pointer_pp_en(Bits::new(1))
            .build(),
    );
    commands.push(
        Register::<DpuRdmaSPointer>::new()
            .pointer_pp_mode(Bits::new(1))
            .executer_pp_en(Bits::new(1))
            .pointer_pp_en(Bits::new(1))
            .build(),
    );

    // CNA convolution and DMA programming.
    commands.push(conv_con1.build());
    commands.push(
        Register::<CnaConvCon2>::new()
            .feature_grains(Bits::new(feature_grains))
            .build(),
    );
    commands.push(
        Register::<CnaConvCon3>::new()
            .conv_x_stride(Bits::new(shape.stride))
            .conv_y_stride(Bits::new(shape.stride))
            .build(),
    );
    commands.push(
        Register::<CnaDataSize0>::new()
            .datain_width(Bits::new(input_width))
            .datain_height(Bits::new(rows.in_rows))
            .build(),
    );
    commands.push(
        Register::<CnaDataSize1>::new()
            .datain_channel_real(Bits::new((shape.in_channels - 1) % CHANNEL_REAL_MODULUS))
            .datain_channel(Bits::new(padded_channels))
            .build(),
    );
    commands.push(
        Register::<CnaDataSize2>::new()
            .dataout_width(Bits::new(out_width))
            .build(),
    );
    commands.push(
        Register::<CnaDataSize3>::new()
            .dataout_atomics(Bits::new(out_width * rows.out_rows))
            .build(),
    );
    commands.push(
        Register::<CnaWeightSize0>::new()
            .weight_bytes(Bits::new(weight_bytes))
            .build(),
    );
    commands.push(
        Register::<CnaWeightSize1>::new()
            .weight_bytes_per_kernel(Bits::new(weight_bytes_per_kernel))
            .build(),
    );
    commands.push(
        Register::<CnaWeightSize2>::new()
            .weight_width(Bits::new(kernel.width))
            .weight_height(Bits::new(kernel.height))
            .weight_kernels(Bits::new(shape.programmed_kernels()))
            .build(),
    );
    commands.push(cbuf_con0.build());
    commands.push(
        Register::<CnaCbufCon1>::new()
            .data_entries(Bits::new(data_entries))
            .build(),
    );
    commands.push(
        Register::<CnaCvtCon0>::new()
            .data_sign(Bits::new(1))
            .cvt_type(Bits::new(1))
            .cvt_bypass(Bits::new(1))
            .build(),
    );
    commands.push(
        Register::<CnaCvtCon1>::new()
            .cvt_scale0(Bits::new(1))
            .build(),
    );
    commands.push(
        Register::<CnaCvtCon2>::new()
            .cvt_scale1(Bits::new(1))
            .build(),
    );
    commands.push(
        Register::<CnaCvtCon3>::new()
            .cvt_scale2(Bits::new(1))
            .build(),
    );
    commands.push(
        Register::<CnaCvtCon4>::new()
            .cvt_scale3(Bits::new(1))
            .build(),
    );
    commands.push(zero::<CnaFcCon0>());
    commands.push(zero::<CnaFcCon1>());
    commands.push(
        // Both axes carry only the padding still visible at the tile's first
        // output coordinate. Interior horizontal tiles clear `pad_left`.
        Register::<CnaPadCon0>::new()
            .pad_top(Bits::new(rows.pad_top))
            .pad_left(Bits::new(columns.pad_left))
            .build(),
    );
    commands.push(
        Register::<CnaFeatureDataAddr>::new()
            .feature_base_addr(Bits::new(tile.input_offset(shape)))
            .build(),
    );
    commands.push(zero::<CnaFcCon2>());
    commands.push(
        Register::<CnaDmaCon0>::new()
            .data_burst_len(BurstLength::Sixteen.into())
            .weight_burst_len(BurstLength::Sixteen.into())
            .build(),
    );
    commands.push(
        Register::<CnaDmaCon1>::new()
            .line_stride(Bits::new(line_stride))
            .build(),
    );
    commands.push(
        Register::<CnaDmaCon2>::new()
            .surf_stride(Bits::new(surf_stride))
            .build(),
    );
    commands.push(
        Register::<CnaFcDataSize0>::new()
            .dma_width(Bits::new(input_width))
            .dma_height(Bits::new(rows.in_rows))
            .build(),
    );
    commands.push(
        Register::<CnaFcDataSize1>::new()
            .dma_channel(Bits::new(padded_channels))
            .build(),
    );
    commands.push(zero::<CnaDcompCtrl>());
    commands.push(zero::<CnaDcompRegnum>());
    commands.push(zero::<CnaDcompAddr0>());
    commands.push(zero::<CnaDcompAmount0>());
    commands.push(zero::<CnaDcompAmount1>());
    commands.push(zero::<CnaDcompAmount2>());
    commands.push(zero::<CnaDcompAmount3>());
    commands.push(zero::<CnaDcompAmount4>());
    commands.push(zero::<CnaDcompAmount5>());
    commands.push(zero::<CnaDcompAmount6>());
    commands.push(zero::<CnaDcompAmount7>());
    commands.push(zero::<CnaDcompAmount8>());
    commands.push(zero::<CnaDcompAmount9>());
    commands.push(zero::<CnaDcompAmount10>());
    commands.push(zero::<CnaDcompAmount11>());
    commands.push(zero::<CnaDcompAmount12>());
    commands.push(zero::<CnaDcompAmount13>());
    commands.push(zero::<CnaDcompAmount14>());
    commands.push(zero::<CnaDcompAmount15>());
    commands.push(zero::<CnaCvtCon5>());
    // Out-of-image taps contribute the quantized encoding of 0.0, which is
    // the input zero point and not zero. fp16 pads with a literal 0 in every
    // capture; int8 pads with the zero point in every capture.
    commands.push(
        Register::<CnaPadCon1>::new()
            .pad_value(Bits::new(
                quantization.map_or(0, |q| q.input_zero_point as u32),
            ))
            .build(),
    );

    // CORE.
    commands.push(
        Register::<CoreMiscCfg>::new()
            .proc_precision(precision.into())
            .qd_en(Bits::new(u32::from(
                quantization.is_some() && !(accumulator_output && acc_vendor_part("qd")),
            )))
            .dw_en(Bits::new(u32::from(shape.depthwise)))
            .build(),
    );
    commands.push(
        Register::<CoreDataoutSize0>::new()
            .dataout_width(Bits::new(out_width - 1))
            .dataout_height(Bits::new(rows.out_rows - 1))
            .build(),
    );
    commands.push(
        Register::<CoreDataoutSize1>::new()
            .dataout_channel(Bits::new(padded_out_channels - 1))
            .build(),
    );
    commands.push(zero::<CoreClipTruncate>());
    commands.push(zero::<CoreReserved3030>());

    // DPU output, conversion, and disabled LUT programming.
    commands.push(
        Register::<DpuFeatureModeCfg>::new()
            .burst_len(BurstLength::Sixteen.into())
            .output_mode(DpuOutputMode::ExternalMemory.into())
            .conv_mode(Bits::new(conv_mode(shape)))
            .build(),
    );
    commands.push(
        Register::<DpuDataFormat>::new()
            .in_precision(dpu_precision)
            .out_precision(output_precision)
            .proc_precision(dpu_precision)
            // 0 selects the "16 B/pixel, one surface" writer, 1 the "2/4
            // surface serial" one. Dense accumulator output uses 0, matching
            // `rocket-userspace`'s validated int8 -> int32 program; only
            // depthwise accumulator output is still on the serial writer,
            // which is the configuration its 256-byte write atom was measured
            // under. Every non-accumulator path has always used 0.
            .mc_surf_out(Bits::new(
                int4_override(shape.precision, "MC_SURF_OUT")
                    .or_else(|| accumulator_mc_surf_out_override(shape.in_channels))
                    .unwrap_or(u32::from(accumulator_output && shape.depthwise)),
            ))
            .bs_mul_shift_value_neg(Bits::new(bs_mul_shift))
            .build(),
    );
    commands.push(zero::<DpuOffsetPend>());
    commands.push(
        Register::<DpuDstBaseAddr>::new()
            .dst_base_addr(Bits::new(output_base_offset))
            .build(),
    );
    commands.push(
        Register::<DpuDstSurfStride>::new()
            .dst_surf_stride(Bits::new(output_surface_pixels))
            .build(),
    );
    commands.push(
        Register::<DpuDataCubeWidth>::new()
            .width(Bits::new(out_width - 1))
            .build(),
    );
    commands.push(
        Register::<DpuDataCubeHeight>::new()
            .height(Bits::new(rows.out_rows - 1))
            .build(),
    );
    commands.push(
        Register::<DpuDataCubeNotchAddr>::new()
            .notch_addr_0(Bits::new(output_notch))
            .notch_addr_1(Bits::new(output_notch))
            .build(),
    );
    commands.push(
        Register::<DpuDataCubeChannel>::new()
            .orig_channel(Bits::new(shape.out_channels - 1))
            .channel(Bits::new(padded_out_channels - 1))
            .build(),
    );
    // With the override on, the accumulator path engages the BS plane as a
    // pass-through -- the stage is clocked, both arithmetic sub-stages are
    // bypassed and the shift above is zeroed, so the output is unchanged --
    // which separates C8's `bs_bypass` from its `out_precision`.
    commands.push(
        Register::<DpuBsCfg>::new()
            .bs_bypass(Bits::new(u32::from(accumulator_output && !bs_passthrough)))
            // The vendor's int32-raw word is 0x53: every sub-stage bypassed and
            // no ALU algo/source, against this crate's 0x20141.
            .bs_alu_algo(Bits::new(if bs_vendor { 0 } else { 2 }))
            .bs_alu_src(Bits::new(if bs_vendor { 0 } else { 1 }))
            .bs_relu_bypass(Bits::new(1))
            .bs_alu_bypass(Bits::new(u32::from(bs_passthrough || bs_vendor)))
            .bs_mul_bypass(Bits::new(u32::from(
                quantization.is_none() || bs_passthrough || bs_vendor,
            )))
            .build(),
    );
    commands.push(zero::<DpuBsAluCfg>());
    commands.push(
        Register::<DpuBsMulCfg>::new()
            .bs_mul_shift_value(Bits::new(bs_mul_shift))
            .bs_mul_src(Bits::new(u32::from(quantization.is_some() && !bs_vendor)))
            .build(),
    );
    commands.push(zero::<DpuBsReluxCmpValue>());
    commands.push(
        Register::<DpuBsOwCfg>::new()
            // 3 for depthwise against 1 for dense, at every captured channel
            // count and in both precisions.
            .size_e_0(Bits::new(bs_ow_size_e(shape)))
            .size_e_1(Bits::new(bs_ow_size_e(shape)))
            .size_e_2(Bits::new(bs_ow_size_e(shape)))
            .od_bypass(Bits::new(u32::from(
                (quantization.is_none() || accumulator_output) && !od_engage(),
            )))
            .ow_src(Bits::new(
                ow_src_override().unwrap_or(u32::from(quantization.is_some())),
            ))
            .build(),
    );
    commands.push(
        Register::<DpuBsOwOp>::new()
            .ow_op(Bits::new(bs_ow_op_value(quantization.as_ref())))
            .build(),
    );
    commands.push(
        Register::<DpuWdmaSize0>::new()
            .channel_wdma(Bits::new(padded_out_channels - 1))
            .build(),
    );
    commands.push(
        Register::<DpuWdmaSize1>::new()
            .height_wdma(Bits::new(rows.out_rows - 1))
            .width_wdma(Bits::new(out_width - 1))
            .build(),
    );
    commands.push(
        Register::<DpuBnCfg>::new()
            .bn_relu_bypass(Bits::new(bn_relu_bypass))
            .bn_relux_en(Bits::new(bn_relux_en))
            // The ALU and MUL halves of the BN stage stay bypassed whatever
            // the activation; only the relu half is ever used.
            .bn_mul_bypass(Bits::new(1))
            .bn_alu_bypass(Bits::new(1))
            .bn_bypass(Bits::new(bn_bypass))
            .build(),
    );
    // Zero in every capture, activated or not: enabling the relu costs no
    // operand buffer and no DMA (`DPU_RDMA_RDMA_BN_BASE_ADDR` stays zero too).
    commands.push(zero::<DpuBnAluCfg>());
    commands.push(zero::<DpuBnMulCfg>());
    commands.push(
        Register::<DpuBnReluxCmpValue>::new()
            .bn_relux_cmp_dat(Bits::new(bn_relux_cmp))
            .build(),
    );
    commands.push(
        Register::<DpuEwCfg>::new()
            .ew_relu_bypass(Bits::new(1))
            .ew_op_cvt_bypass(Bits::new(1))
            .ew_lut_bypass(Bits::new(1))
            .ew_op_bypass(Bits::new(1))
            .ew_bypass(Bits::new(1))
            .build(),
    );
    commands.push(zero::<DpuEwCvtOffsetValue>());
    commands.push(
        Register::<DpuEwCvtScaleValue>::new()
            .ew_op_cvt_scale(Bits::new(1))
            .build(),
    );
    commands.push(zero::<DpuEwReluxCmpValue>());
    // Output conversion. fp16 lets `fp32tofp16_en` do the narrowing; int8
    // programs the multiplier as a normalized mantissa/shift pair and the
    // output zero point as an offset. Exact accumulator output bypasses BS
    // and CPEND and leaves this final converter at identity.
    let output_offset = if accumulator_output {
        0
    } else {
        quantization.map_or(0, |q| q.output_zero_point as u32)
    };
    let output_scale = if accumulator_output {
        1
    } else {
        quantization.map_or(1, |q| q.multiplier.scale)
    };
    let output_shift = if accumulator_output {
        0
    } else {
        quantization.map_or(0, |q| q.multiplier.shift)
    };
    commands.push(
        Register::<DpuOutCvtOffset>::new()
            .out_cvt_offset(Bits::new(output_offset))
            .build(),
    );
    commands.push(
        Register::<DpuOutCvtScale>::new()
            // The bit that does the narrowing on the ordinary fp16 path.
            // A rung whose result *is* the fp32 accumulator clears it; the
            // quantized paths never set it.
            .fp32tofp16_en(Bits::new(u32::from(
                quantization.is_none() && !shape.precision.writes_fp32_result(),
            )))
            .out_cvt_scale(Bits::new(output_scale))
            .build(),
    );
    commands.push(
        Register::<DpuOutCvtShift>::new()
            .out_cvt_shift(Bits::new(output_shift))
            .build(),
    );
    commands.push(zero::<DpuEwOpValue0>());
    commands.push(zero::<DpuEwOpValue1>());
    commands.push(zero::<DpuEwOpValue2>());
    commands.push(zero::<DpuEwOpValue3>());
    commands.push(zero::<DpuEwOpValue4>());
    commands.push(zero::<DpuEwOpValue5>());
    commands.push(zero::<DpuEwOpValue6>());
    commands.push(zero::<DpuEwOpValue7>());
    commands.push(
        // Accumulator output is hardware-validated with the fixed logical
        // value 16 together with DPU_DATA_FORMAT.mc_surf_out=1. CNA's
        // DATA_SIZE3.surf_mode deliberately remains zero: enabling it
        // changes the convolution results for NC1HWC2 input. Requantized
        // output uses half an output atom per pixel, and is otherwise not
        // precision-dependent:
        // this field is byte-identical across every fp16/int8 capture pair,
        // unlike `weight_bytes_per_kernel` right above, which halves.
        // Depthwise doubles this. Confirmed as a factor rather than a
        // constant by the stride-2 capture, whose 16x16 output takes 1024
        // against the dense 512.
        Register::<DpuSurfaceAdd>::new()
            .surf_add(Bits::new(if accumulator_output {
                let mult = accumulator_surf_mult_override(shape.in_channels)
                    .unwrap_or(DENSE_ACCUMULATOR_SURF_MULT);
                if shape.depthwise {
                    // Depthwise accumulator output stays on the serial writer,
                    // which serializes its 32-lane blocks with the fixed
                    // hardware value 16. `ROCKET_ACC_SURF_ADD` overrides it.
                    accumulator_surf_add_override(shape.in_channels).unwrap_or(16)
                } else {
                    // `rocket-userspace`'s rule: `dst_surf_stride * 8`, where
                    // `dst_surf_stride` is the *task's* `dataout_height *
                    // dataout_width`, not the whole image's. Per-task is
                    // load-bearing and is why a constant `ROCKET_ACC_SURF_ADD`
                    // could never express this on a tiled plan -- every tile
                    // has its own `out_rows`. Legal here because
                    // `programs_with_staged_accumulator_output` gives each tile
                    // its own contiguous scratch range, so a tile really is a
                    // standalone image; the non-accumulator branch below uses
                    // whole-image dims precisely because its tiles share one.
                    accumulator_surf_add_override(shape.in_channels)
                        .unwrap_or(out_width * rows.out_rows * mult)
                }
            } else if let Some(surf_add) = int4_override(shape.precision, "SURF_ADD") {
                surf_add
            } else if shape.precision.writes_fp32_result() {
                // The float path's surface multiplier is the output element
                // width: 2 for a 2-byte result, 4 for an fp32 one.
                full_out_width * out_height * 4
            } else if shape.precision == Precision::Int4 {
                // The integer write path's surface multiplier is 8, not the
                // float path's 2 -- the other half of the `size_e` quirk.
                // At Cout 128 the 2x value stops after 10 of 16 surfaces
                // and the 8x one writes all 16384 bytes.
                full_out_width * out_height * DENSE_ACCUMULATOR_SURF_MULT
            } else {
                full_out_width * out_height * 2 * if shape.depthwise { 2 } else { 1 }
            }))
            .build(),
    );
    commands.push(zero::<DpuReserved40c4>());
    commands.push(zero::<DpuLutAccessCfg>());
    commands.push(zero::<DpuLutAccessData>());
    commands.push(zero::<DpuLutCfg>());
    commands.push(zero::<DpuLutInfo>());
    commands.push(zero::<DpuLutLeStart>());
    commands.push(zero::<DpuLutLeEnd>());
    commands.push(zero::<DpuLutLoStart>());
    commands.push(zero::<DpuLutLoEnd>());
    commands.push(zero::<DpuLutLeSlopeScale>());
    commands.push(zero::<DpuLutLeSlopeShift>());
    commands.push(zero::<DpuLutLoSlopeScale>());
    commands.push(zero::<DpuLutLoSlopeShift>());

    // DPU_RDMA. The main feature path is disabled because CNA/CORE feed
    // DPU directly; BRDMA supplies the bias data.
    commands.push(
        Register::<DpuRdmaDataCubeWidth>::new()
            .width(Bits::new(out_width - 1))
            .build(),
    );
    commands.push(
        Register::<DpuRdmaDataCubeHeight>::new()
            .height(Bits::new(rows.out_rows - 1))
            .build(),
    );
    commands.push(
        Register::<DpuRdmaDataCubeChannel>::new()
            .channel(Bits::new(padded_out_channels - 1))
            .build(),
    );
    commands.push(zero::<DpuRdmaSrcBaseAddr>());
    commands.push(
        Register::<DpuRdmaBrdmaCfg>::new()
            .brdma_data_use(Bits::new(brdma_data_use))
            .build(),
    );
    commands.push(zero::<DpuRdmaBsBaseAddr>());
    commands.push(zero::<DpuRdmaNrdmaCfg>());
    commands.push(zero::<DpuRdmaBnBaseAddr>());
    commands.push(
        Register::<DpuRdmaErdmaCfg>::new()
            .erdma_disable(Bits::new(1))
            .build(),
    );
    commands.push(zero::<DpuRdmaEwBaseAddr>());
    commands.push(zero::<DpuRdmaEwSurfStride>());
    commands.push(
        Register::<DpuRdmaFeatureModeCfg>::new()
            .burst_len(BurstLength::Sixteen.into())
            .mrdma_disable(Bits::new(1))
            .in_precision(dpu_precision)
            .proc_precision(dpu_precision)
            .conv_mode(Bits::new(conv_mode(shape)))
            .build(),
    );
    commands.push(zero::<DpuRdmaSrcDmaCfg>());
    commands.push(zero::<DpuRdmaSurfNotch>());
    commands.push(zero::<DpuRdmaPadCfg>());
    commands.push(
        Register::<DpuRdmaWeight>::new()
            .e_weight(Bits::new(1))
            .n_weight(Bits::new(1))
            .b_weight(Bits::new(1))
            .m_weight(Bits::new(1))
            .build(),
    );
    commands.push(zero::<DpuRdmaEwSurfNotch>());

    // Vendor PC trailer: placeholder, zero register count, required marker,
    // combined operation-enable mask, and six words of alignment padding.
    commands.push(PCTrailer::single_task_placeholder());
    commands.push(zero::<PCRegisterAmounts>());
    commands.push(PCTrailer::required_marker());
    commands.push(PCTrailer::operation_enable(PCOperationMask::CONVOLUTION));
    commands.extend((0..6).map(|_| PCTrailer::alignment_padding()));

    debug_assert_eq!(commands.len(), 136);
    commands
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode(command: &RegCmd) -> (u32, u32, u32) {
        (
            (command.0 >> 48) as u32,
            command.0 as u32 & 0xffff,
            (command.0 >> 16) as u32,
        )
    }

    fn fnv1a(commands: &[RegCmd]) -> u64 {
        commands
            .iter()
            .flat_map(|command| command.0.to_le_bytes())
            .fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
                (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
            })
    }

    /// Registers that carry a precision field, and nothing else that a
    /// datatype change is allowed to move.
    fn precision_register_identities() -> Vec<(u32, u32)> {
        vec![
            (
                <CnaConvCon1 as RegisterMeta>::DOMAIN,
                <CnaConvCon1 as RegisterMeta>::OFFSET,
            ),
            (
                <CoreMiscCfg as RegisterMeta>::DOMAIN,
                <CoreMiscCfg as RegisterMeta>::OFFSET,
            ),
            (
                <DpuDataFormat as RegisterMeta>::DOMAIN,
                <DpuDataFormat as RegisterMeta>::OFFSET,
            ),
            (
                <DpuRdmaFeatureModeCfg as RegisterMeta>::DOMAIN,
                <DpuRdmaFeatureModeCfg as RegisterMeta>::OFFSET,
            ),
        ]
    }

    /// The 2-byte rungs are the fp16 program with the precision field
    /// changed, and nothing else.
    ///
    /// This is the same diff `rockchip-npu-notes` records for the matmul
    /// path -- `gen_matmul_bf16` == `gen_matmul_fp16` with only the
    /// precision words moved -- asserted here for the convolution program,
    /// which has four such registers rather than three (the DPU_RDMA stage
    /// carries a pair as well). It is what makes "bf16 is fp16 with a
    /// different field value" a checked claim rather than a hope: any future
    /// geometry rule that keys off `Precision::Fp16` by name instead of by
    /// element width will fail here.
    #[test]
    fn two_byte_precisions_differ_from_fp16_only_in_the_precision_registers() {
        let kernels: Kernels = [3, 3];
        for (precision, field) in [
            (Precision::Bf16, DataPrecision::Bf16 as u32),
            (Precision::Int16, DataPrecision::Int16 as u32),
        ] {
            let fp16 = Shape::with_precision(32, 32, 1, 64, 32, Precision::Fp16);
            let other = Shape::with_precision(32, 32, 1, 64, 32, precision);
            assert_eq!(
                fp16.weight_bytes(kernels),
                other.weight_bytes(kernels),
                "{precision:?} coefficient footprint must match fp16"
            );
            assert_eq!(
                fp16.output_scratch_bytes(kernels),
                other.output_scratch_bytes(kernels),
                "{precision:?} output allocation must match fp16"
            );

            let tile = Tile::whole(fp16, kernels);
            let baseline = conv_2d_tile(fp16, kernels, &tile);
            let candidate = conv_2d_tile(other, kernels, &tile);
            assert_eq!(baseline.len(), candidate.len());

            let expected = precision_register_identities();
            let mut moved = Vec::new();
            for (before, after) in baseline.iter().zip(&candidate) {
                let (domain, offset, before_value) = decode(before);
                let (after_domain, after_offset, after_value) = decode(after);
                assert_eq!((domain, offset), (after_domain, after_offset));
                if before_value != after_value {
                    moved.push((domain, offset, before_value, after_value));
                }
            }
            // `CNA_CONV_CON1` is written twice by the program, so compare
            // the distinct identities rather than the raw sequence.
            let mut identities: Vec<(u32, u32)> = moved
                .iter()
                .map(|(domain, offset, _, _)| (*domain, *offset))
                .collect();
            identities.dedup();
            assert_eq!(
                identities, expected,
                "{precision:?} moved unexpected registers: {moved:x?}"
            );
            // Every moved word must differ only where a 3-bit precision
            // field sits: xor the two and the result has to be a set of
            // 3-bit-aligned nibbles, never a stray geometry bit.
            for (domain, offset, before_value, after_value) in moved {
                let fp16_field = DataPrecision::Fp16 as u32;
                let mut rebuilt = before_value;
                for shift in 0..30 {
                    if (before_value >> shift) & 0x7 == fp16_field
                        && (after_value >> shift) & 0x7 == field
                    {
                        rebuilt = (rebuilt & !(0x7 << shift)) | (field << shift);
                    }
                }
                assert_eq!(
                    rebuilt, after_value,
                    "{precision:?} changed {domain:#x}:{offset:#x} outside its \
                     precision fields: {before_value:#010x} -> {after_value:#010x}"
                );
            }
        }
    }

    /// int4's geometry falls out of the shared atom widths.
    ///
    /// Every number here is `atom bytes * 8 / element bits` rather than a
    /// table entry, which is the claim worth pinning: the 32-channel feature
    /// atom, the 64-kernel coefficient atom and the halved coefficient
    /// footprint are all consequences of the half-byte element, and they
    /// match what `../rockchip-npu-notes/encodings/tile-layouts.md` records
    /// for int4.
    /// Keeping the fp16 accumulator moves the output writer and nothing
    /// else.
    ///
    /// fp16 already accumulates in fp32 and only loses precision on the way
    /// out, so this rung must not disturb the input side at all: same
    /// coefficient buffer, same feature geometry, same CNA and CORE
    /// programming. What it does move is exactly the four registers that
    /// describe the result -- and `fp32tofp16_en` in particular, which is
    /// what performs the narrowing that this rung exists to avoid.
    #[test]
    fn fp16_accumulator_moves_only_the_output_writer() {
        let kernels: Kernels = [3, 3];
        let narrowed = Shape::with_precision(8, 8, 1, 64, 64, Precision::Fp16);
        let kept = Shape::with_precision(8, 8, 1, 64, 64, Precision::Fp16Accumulator);

        assert_eq!(
            narrowed.weight_bytes(kernels),
            kept.weight_bytes(kernels),
            "the coefficient buffer must be untouched"
        );
        assert_eq!(narrowed.padded_channels(), kept.padded_channels());
        assert_eq!(narrowed.padded_out_channels(), kept.padded_out_channels());
        assert_eq!(
            narrowed.output_scratch_bytes(kernels) * 2,
            kept.output_scratch_bytes(kernels),
            "a 4-byte result doubles the allocation"
        );
        // The 4-byte result writes a 4-lane output cube, where the fp16 one
        // writes 8 -- `C2 = 16 bytes / out element`, the same cube int8's
        // int32 accumulator and tf32's fp32 result use.
        assert_eq!(
            kept.output_atom_bytes() / kept.precision.output_element_bytes(),
            4
        );
        assert_eq!(
            narrowed.output_atom_bytes() / narrowed.precision.output_element_bytes(),
            8
        );

        let moved: Vec<(u32, u32)> =
            conv_2d_tile(narrowed, kernels, &Tile::whole(narrowed, kernels))
                .iter()
                .zip(&conv_2d_tile(kept, kernels, &Tile::whole(kept, kernels)))
                .filter_map(|(before, after)| {
                    let (domain, offset, before_value) = decode(before);
                    let (_, _, after_value) = decode(after);
                    (before_value != after_value).then_some((domain, offset))
                })
                .collect();
        assert_eq!(
            moved,
            vec![
                (
                    <DpuDataFormat as RegisterMeta>::DOMAIN,
                    <DpuDataFormat as RegisterMeta>::OFFSET
                ),
                (
                    <DpuBsOwCfg as RegisterMeta>::DOMAIN,
                    <DpuBsOwCfg as RegisterMeta>::OFFSET
                ),
                (
                    <DpuOutCvtScale as RegisterMeta>::DOMAIN,
                    <DpuOutCvtScale as RegisterMeta>::OFFSET
                ),
                (
                    <DpuSurfaceAdd as RegisterMeta>::DOMAIN,
                    <DpuSurfaceAdd as RegisterMeta>::OFFSET
                ),
            ],
            "only the DPU output writer may move"
        );
    }

    #[test]
    fn int4_geometry_follows_the_half_byte_element() {
        let kernels: Kernels = [3, 3];
        let int4 = Shape::with_precision(16, 16, 1, 64, 64, Precision::Int4);
        assert_eq!(int4.precision.element_bits(), 4);
        assert_eq!(int4.precision.channels_per_atom(), 32);
        assert_eq!(int4.precision.out_channel_granule(), 64);
        assert_eq!(int4.feature_atoms(), 2);
        assert_eq!(int4.padded_channels(), 64);
        // int4 accumulates to int16, so the result is four times as wide as
        // an operand.
        assert_eq!(int4.precision.output_element_bytes(), 2);

        // Half of int8's coefficient footprint at the same shape, and a
        // quarter of fp16's -- the whole point of the rung.
        let int8 = Shape::with_precision(
            16,
            16,
            1,
            64,
            64,
            Precision::Int8(Quantization {
                input_zero_point: 0,
                output_zero_point: 0,
                weight_zero_point: 0,
                input_scale: 1.0,
                weights_scale: 1.0,
                multiplier: Multiplier::from_ratio(1.0),
            }),
        );
        let fp16 = Shape::with_precision(16, 16, 1, 64, 64, Precision::Fp16);
        assert_eq!(int4.weight_bytes(kernels) * 2, int8.weight_bytes(kernels));
        assert_eq!(int4.weight_bytes(kernels) * 4, fp16.weight_bytes(kernels));

        // The precision field reaches the program as 6, and the DPU writes
        // an int16 result (field 1) rather than an int4 one.
        let tile = Tile::whole(int4, kernels);
        let program = conv_2d_tile(int4, kernels, &tile);
        let data_format = program
            .iter()
            .map(decode)
            .find(|(domain, offset, _)| {
                (*domain, *offset)
                    == (
                        <DpuDataFormat as RegisterMeta>::DOMAIN,
                        <DpuDataFormat as RegisterMeta>::OFFSET,
                    )
            })
            .expect("program must configure DPU_DATA_FORMAT")
            .2;
        // `DPU_DATA_FORMAT` is out[31:29], in[28:26], proc[2:0].
        assert_eq!((data_format >> 29) & 0x7, OutputPrecision::Int16 as u32);
        assert_eq!((data_format >> 26) & 0x7, DataPrecision::Int4 as u32);
        assert_eq!(data_format & 0x7, DataPrecision::Int4 as u32);
    }

    /// Depthwise is opt-in per width, not inherited by every new rung.
    #[test]
    fn depthwise_accepts_only_the_measured_element_widths() {
        for precision in [Precision::Fp16, Precision::Bf16, Precision::Int16] {
            let _ = Shape::with_precision(34, 34, 1, 32, 32, precision).with_depthwise();
        }
        let quantization = Quantization {
            input_zero_point: 0,
            output_zero_point: 0,
            weight_zero_point: 0,
            input_scale: 1.0,
            weights_scale: 1.0,
            multiplier: Multiplier::from_ratio(1.0),
        };
        let _ = Shape::with_precision(34, 34, 1, 32, 32, Precision::Int8(quantization))
            .with_depthwise();

        for precision in [Precision::Int4, Precision::Tf32, Precision::Fp16Accumulator] {
            let shape = Shape::with_precision(34, 34, 1, 32, 32, precision);
            assert!(
                std::panic::catch_unwind(move || shape.with_depthwise()).is_err(),
                "{precision:?} depthwise must be refused until it is measured"
            );
        }
    }

    #[test]
    #[should_panic(expected = "whole 32-channel feature atom")]
    fn int4_refuses_a_partial_feature_atom() {
        let _ = Shape::with_precision(16, 16, 1, 48, 64, Precision::Int4);
    }

    #[test]
    fn vendor_reference_program_has_expected_layout() {
        let commands = conv_2d([1, 1]);
        assert_eq!(commands.len(), 136);

        assert_eq!(decode(&commands[0]), (0x0201, 0x1040, 0x0000_00b1));
        assert_eq!(decode(&commands[4]), (0x1001, 0x4004, 0x0000_000e));
        assert_eq!(decode(&commands[5]), (0x2001, 0x5004, 0x0000_000e));
        assert_eq!(decode(&commands[54]), (0x0801, 0x3010, 0x0000_0200));
        assert_eq!(decode(&commands[59]), (0x1001, 0x400c, 0x0000_01e4));
        assert_eq!(decode(&commands[109]), (0x2001, 0x500c, 0x0000_001f));

        assert_eq!(commands[126].0, 0);
        assert_eq!(commands[127].0, 0x0101_0000_0000_0014);
        assert_eq!(commands[128].0, 0x0041_0000_0000_0000);
        assert_eq!(commands[129].0, 0x0081_0000_001d_0008);
        assert!(commands[130..].iter().all(|command| command.0 == 0));
    }

    #[test]
    fn programs_match_captured_vendor_group_one_bit_for_bit() {
        assert_eq!(fnv1a(&conv_2d([1, 1])), 0x2577_26d7_f13a_1636);
        assert_eq!(fnv1a(&conv_2d([3, 3])), 0x8da7_c9ed_d561_7ccf);
    }

    #[test]
    fn kernel_geometry_changes_exactly_the_five_vendor_words() {
        let one_by_one = conv_2d([1, 1]);
        let three_by_three = conv_2d([3, 3]);
        let changed: Vec<_> = one_by_one
            .iter()
            .zip(&three_by_three)
            .enumerate()
            .filter_map(|(index, (left, right))| (left.0 != right.0).then_some(index))
            .collect();

        assert_eq!(changed, [7, 13, 14, 15, 25]);
        assert_eq!(decode(&one_by_one[7]).2, 0x0000_0210);
        assert_eq!(decode(&three_by_three[7]).2, 0x0000_0240);
        assert_eq!(decode(&one_by_one[13]).2, 0x0000_0080);
        assert_eq!(decode(&three_by_three[13]).2, 0x0000_0480);
        assert_eq!(decode(&one_by_one[14]).2, 0x0000_0010);
        assert_eq!(decode(&three_by_three[14]).2, 0x0000_0090);
        assert_eq!(decode(&one_by_one[15]).2, 0x0101_0008);
        assert_eq!(decode(&three_by_three[15]).2, 0x0303_0008);
        assert_eq!(decode(&one_by_one[25]).2, 0);
        assert_eq!(decode(&three_by_three[25]).2, 0x0000_0011);
    }

    #[test]
    fn even_kernels_program_verbatim_with_independent_padding() {
        let shape = Shape::CAPTURED.with_padding([0, 1]);
        let kernels = [4, 6];
        let plan = ConvPlan::new(shape, kernels);

        assert_eq!(
            (shape.output_width(kernels), shape.output_height(kernels)),
            (29, 29)
        );
        assert_eq!((plan.data_banks(), plan.weight_banks()), (1, 11));
        assert_eq!(plan.tiles(), &[Tile2D::whole(shape, kernels)]);

        let program = &plan.programs()[0];
        assert_eq!(value_of::<CnaWeightSize2>(program), 0x0604_0008);
        assert_eq!(value_of::<CnaPadCon0>(program), 0x10);
        assert_eq!(value_of::<CnaConvCon2>(program), 0x240);
        assert_eq!(value_of::<CnaWeightSize0>(program), 0xc00);
        assert_eq!(value_of::<CnaWeightSize1>(program), 0x180);

        let int8_shape =
            Shape::with_precision(32, 32, 1, 3, 8, captured_int8()).with_padding([0, 1]);
        let int8_programs = ConvPlan::new(int8_shape, kernels).programs();
        let int8_program = &int8_programs[0];
        assert_eq!(value_of::<CnaWeightSize2>(int8_program), 0x0604_0008);
        assert_eq!(value_of::<CnaPadCon0>(int8_program), 0x10);
        assert_eq!(value_of::<CnaConvCon2>(int8_program), 0x240);
    }

    #[test]
    fn even_kernel_default_padding_is_half_the_extent() {
        let shape = Shape::CAPTURED;
        for extent in [2usize, 4, 6, 8, 10] {
            let kernels = [extent, extent];
            assert_eq!(
                (shape.output_width(kernels), shape.output_height(kernels)),
                (33, 33),
                "k{extent}"
            );
            let plan = ConvPlan::new(shape, kernels);
            assert!(!plan.tiles().is_empty(), "k{extent}");
            assert!(plan.programs().iter().all(|program| program.len() == 136));
        }
    }

    #[test]
    fn even_kernel_tiles_keep_the_full_tap_halo() {
        let shape = Shape::new(8, 8).with_padding([0, 0]);
        let tiles = Tile::split(shape, [4, 4], 2);

        assert_eq!(
            tiles,
            [
                Tile {
                    out_first: 0,
                    out_rows: 3,
                    in_first: 0,
                    in_rows: 6,
                    pad_top: 0,
                },
                Tile {
                    out_first: 3,
                    out_rows: 2,
                    in_first: 3,
                    in_rows: 5,
                    pad_top: 0,
                },
            ]
        );
    }

    #[test]
    fn even_non_square_plans_take_the_captured_split_at_every_measured_demand() {
        // The even sweep's pressure row: 256x32, Cin 32, Cout 64 fp16, where
        // coefficient demand is `ceil(kh * kw / 8)`. Through six banks every
        // mirrored even pair agrees with its twin and with the demand-based
        // allocator.
        for (kernels, banks) in [
            ([4usize, 8usize], (8u32, 4u32)),
            ([8, 4], (8, 4)),
            ([4, 10], (7, 5)),
            ([10, 4], (7, 5)),
            ([6, 8], (6, 6)),
            ([8, 6], (6, 6)),
        ] {
            let shape = Shape::with_out_channels(256, 32, 1, 32, 64);
            let plan = ConvPlan::new(shape, kernels);
            assert_eq!(
                (plan.data_banks(), plan.weight_banks()),
                banks,
                "{kernels:?} CBUF split"
            );
        }
    }

    #[test]
    fn even_square_plans_follow_demand_to_the_measured_ceiling() {
        // 10x10 was refused outright before the fill-in row, on the strength
        // of a single Cout 64 capture at demand 13. Walking Cout down shows
        // the extent was never the problem: at demands 5 and 7 a 10x10 takes
        // exactly the demand-based split the captures show.
        for (cout, banks) in [(24u32, (7u32, 5u32)), (32, (5, 7))] {
            let plan = ConvPlan::new(Shape::with_out_channels(256, 32, 1, 32, cout), [10, 10]);
            assert_eq!(
                (plan.data_banks(), plan.weight_banks()),
                banks,
                "10x10 Cout {cout} CBUF split"
            );
        }
        // 8x8 at Cout 64 is the last demand the ladder confirms, at eight.
        let plan = ConvPlan::new(Shape::with_out_channels(256, 32, 1, 32, 64), [8, 8]);
        assert_eq!((plan.data_banks(), plan.weight_banks()), (4, 8));
    }

    #[test]
    #[should_panic(expected = "even square kernel [8, 8] needs 9")]
    fn automatic_plan_refuses_the_even_square_ladder_above_eight_banks() {
        // Cout 72 puts 8x8 at demand 9, the first rung the ladder shows
        // leaving the demand rule: the capture is 8/4 where demand asks 3/9.
        let _ = ConvPlan::new(Shape::with_out_channels(256, 32, 1, 32, 72), [8, 8]);
    }

    #[test]
    #[should_panic(expected = "even kernel [6, 10] needs 8")]
    fn automatic_plan_refuses_the_even_rectangle_that_disagrees_with_its_mirror() {
        // 6x10 and 10x6 both ask for eight banks and are captured 4/8 and
        // 8/4. Demand cannot choose between them, so neither is planned --
        // even though 6x10's own capture happens to match the demand rule.
        let _ = ConvPlan::new(Shape::with_out_channels(256, 32, 1, 32, 64), [6, 10]);
    }

    #[test]
    #[should_panic(expected = "even kernel [6, 8] needs 6")]
    fn automatic_plan_refuses_int8_even_rectangles_fp16_still_plans() {
        // The one place the two precisions need different bounds. At Cin 32,
        // Cout 128 int8 a 6x8 is captured 8/4 where demand asks 6/6, while
        // the matching fp16 shape at demand 6 takes 6/6 exactly.
        let shape = Shape::with_precision(256, 32, 1, 32, 128, captured_int8());
        let _ = ConvPlan::new(shape, [6, 8]);
    }

    #[test]
    #[should_panic(expected = "even kernels currently have capture backing only at stride 1")]
    fn automatic_plan_refuses_even_kernels_above_stride_one() {
        // Every point in the even grid is stride 1. Nothing says the pad or
        // grains formulas survive a stride an even kernel has never been
        // captured at.
        let _ = ConvPlan::new(Shape::with_stride(32, 32, 2), [4, 4]);
    }

    #[test]
    #[should_panic(expected = "where the captured split follows coefficient demand")]
    fn automatic_plan_refuses_unsettled_even_kernel_cbuf_pressure() {
        let shape = Shape::with_out_channels(256, 32, 1, 32, 64);
        let _ = ConvPlan::new(shape, [10, 10]);
    }

    #[test]
    #[should_panic(expected = "automatic CBUF allocation")]
    fn default_builder_rejects_large_kernel_without_explicit_banks() {
        let _ = conv_2d([5, 5]);
    }

    #[test]
    fn explicit_cbuf_builder_accepts_focused_large_kernels() {
        let shape = Shape::with_out_channels(256, 32, 1, 32, 64);
        for (kernel, data_banks, weight_banks, max_rows, min_tiles) in [
            (7usize, 8u32, 4u32, 16u32, 4u32),
            (9, 6, 6, 12, 8),
            (11, 7, 5, 14, 8),
        ] {
            let kernels = [kernel, kernel];
            assert_eq!(
                shape.max_tile_input_rows_for_data_banks(data_banks),
                max_rows
            );
            assert_eq!(
                shape.min_tiles_for_data_banks(kernels, data_banks),
                min_tiles
            );
            for tile in Tile::split(shape, kernels, min_tiles) {
                assert_eq!(
                    conv_2d_tile_with_cbuf_banks(shape, kernels, &tile, data_banks, weight_banks,)
                        .len(),
                    136
                );
            }
        }
    }

    #[test]
    fn conv_plan_preserves_the_captured_single_program() {
        let plan = ConvPlan::new(Shape::CAPTURED, [3, 3]);
        assert_eq!((plan.data_banks(), plan.weight_banks()), (1, 11));
        assert_eq!(plan.output_column_widths(), &[32]);
        assert_eq!(plan.tiles(), &[Tile2D::whole(Shape::CAPTURED, [3, 3])]);
        let programs = plan.programs();
        assert_eq!(programs.len(), 1);
        assert_eq!(fnv1a(&programs[0]), fnv1a(&conv_2d([3, 3])));
    }

    #[test]
    fn conv_plan_reconciles_the_expanded_vendor_fixture_routes() {
        // Granting the whole-map data demand avoids an otherwise unnecessary
        // split and still leaves four coefficient banks, above the measured
        // floor of three.
        let single = Shape::with_out_channels(32, 32, 1, 128, 64);
        let plan = ConvPlan::new(single, [3, 3]);
        assert_eq!((plan.data_banks(), plan.weight_banks()), (8, 4));
        assert_eq!(plan.tiles().len(), 1);

        // Bit 14 is not int8-specific: expanded fp16 Cin-1 captures use it
        // too, and keep these shapes in one task.
        let fp16_wide_entries = Shape::with_out_channels(128, 128, 1, 1, 1);
        let plan = ConvPlan::new(fp16_wide_entries, [1, 1]);
        assert_eq!(plan.tiles().len(), 1);
        assert_eq!(value_of::<CnaCbufCon1>(&plan.programs()[0]), 16_384);

        // Standalone vendor plans fill each tile to capacity rather than
        // balancing the output height over the minimum tile count.
        let tall = Shape::with_out_channels(64, 128, 1, 128, 8);
        let plan = ConvPlan::new(tall, [3, 3]);
        assert_eq!((plan.data_banks(), plan.weight_banks()), (11, 1));
        assert_eq!(
            plan.tiles()
                .iter()
                .map(|tile| tile.rows.out_rows)
                .collect::<Vec<_>>(),
            vec![21, 20, 20, 20, 20, 20, 7]
        );

        // Int8 dense CBUF rows charge width 226 as 240, but the host tensor
        // itself has a compact 226*3-byte row pitch. RKNN's padded-pitch
        // 92+91+43 route starts compact tiles at unsafe offsets, so retain
        // three tiles while moving both interior feature bases to 16-byte
        // boundaries.
        let int8 = Shape::with_precision(226, 226, 1, 3, 64, Precision::Int8(quantization()));
        let plan = ConvPlan::new(int8, [3, 3]);
        assert_eq!((plan.data_banks(), plan.weight_banks()), (11, 1));
        assert_eq!(
            plan.tiles()
                .iter()
                .map(|tile| (tile.rows.out_rows, tile.rows.in_rows))
                .collect::<Vec<_>>(),
            vec![(73, 74), (80, 82), (73, 74)]
        );
        assert_eq!(
            plan.programs()
                .iter()
                .map(|program| value_of::<CnaCbufCon1>(program))
                .collect::<Vec<_>>(),
            vec![17_760, 19_680, 17_760]
        );

        // CBUF/data_entries permits 1023 rows here, but feature_grains adds
        // one for a 1x1 kernel. Keep each task at 1022 rows so the 10-bit
        // field never receives the unencodable value 1024.
        let feature_grain_limited = Shape::with_out_channels(32, 1200, 1, 1, 1);
        let plan = ConvPlan::new(feature_grain_limited, [1, 1]);
        assert_eq!((plan.data_banks(), plan.weight_banks()), (10, 2));
        assert_eq!(
            plan.tiles()
                .iter()
                .map(|tile| tile.rows.in_rows)
                .collect::<Vec<_>>(),
            vec![1022, 178]
        );
        assert!(
            plan.tiles()
                .iter()
                .all(|tile| feature_grains([1, 1], &tile.rows) <= MAX_FEATURE_GRAINS)
        );
        assert_eq!(plan.programs().len(), 2);

        // At the exact surface-capacity boundary, standalone RKNN plans do
        // not let bottom-edge clipping enlarge the final grain. Both
        // precisions have 24 input atoms here and therefore take the same
        // 5/7 split and 14+13+1 route.
        for shape in [
            Shape::with_out_channels(28, 28, 1, 192, 64).with_padding([1, 1]),
            Shape::with_precision(28, 28, 1, 384, 64, captured_int8()).with_padding([1, 1]),
        ] {
            let plan = ConvPlan::new(shape, [3, 3]);
            assert_eq!((plan.data_banks(), plan.weight_banks()), (5, 7));
            assert_eq!(
                plan.tiles()
                    .iter()
                    .map(|tile| tile.rows.out_rows)
                    .collect::<Vec<_>>(),
                vec![14, 13, 1]
            );
        }

        // The same policy remains visible at a one-row continuation grain:
        // first produce two rows, then 26 single rows through the bottom.
        let tiny_grain = Shape::with_out_channels(28, 28, 1, 512, 64).with_padding([1, 1]);
        let plan = ConvPlan::new(tiny_grain, [3, 3]);
        assert_eq!((plan.data_banks(), plan.weight_banks()), (3, 9));
        let mut expected = vec![2];
        expected.extend(vec![1; 26]);
        assert_eq!(
            plan.tiles()
                .iter()
                .map(|tile| tile.rows.out_rows)
                .collect::<Vec<_>>(),
            expected
        );
    }

    /// The focused above-3x3 policies at the sweep's centre shape.
    ///
    /// k=11 takes `(4, 8)` here, not the capture's `(7, 5)`:
    /// `unstarved_large_kernel_partition` raises the grant to the streamed
    /// coefficient preference, which is what removed the `Cin` cliff. The
    /// extra coefficient bank costs data banks, so the row no longer fits one
    /// column tile -- hence the `[128, 128]` split and 12 tiles rather than 7.
    /// All four rows are board-validated by
    /// `conv_kernel_size_hw::large_kernel_cbuf_partitions_run_on_npu`.
    #[test]
    fn conv_plan_selects_the_focused_large_kernel_policies() {
        let shape = Shape::with_out_channels(256, 32, 1, 32, 64);
        for (kernel, banks, columns, tiles) in [
            (5usize, (8u32, 4u32), &[256u32][..], 3usize),
            (7, (5, 7), &[256][..], 8),
            (9, (6, 6), &[256][..], 7),
            (11, (4, 8), &[128, 128][..], 12),
        ] {
            let plan = ConvPlan::new(shape, [kernel, kernel]);
            assert_eq!(
                (plan.data_banks(), plan.weight_banks()),
                banks,
                "k{kernel} banks"
            );
            assert_eq!(plan.output_column_widths(), columns, "k{kernel} columns");
            assert_eq!(plan.tiles().len(), tiles, "k{kernel} tiles");
            assert!(plan.programs().iter().all(|program| program.len() == 136));
        }
    }

    /// The streamed floor only ever raises a capture-derived grant.
    ///
    /// This is what makes `unstarved_large_kernel_partition` safe to apply to
    /// shapes the captures already validate: every such shape keeps a
    /// coefficient grant at least as large as the one it was validated with,
    /// so the correction can cost tiles but cannot starve a stream that was
    /// previously fed.
    #[test]
    fn unstarved_partition_never_lowers_the_captured_coefficient_grant() {
        for kernel in [7usize, 9, 11] {
            for cin in [16u32, 24, 32, 48, 64, 96, 128, 192] {
                for captured in [(8u32, 4u32), (7, 5), (6, 6), (5, 7), (3, 9)] {
                    let shape = Shape::with_out_channels(256, 32, 1, cin, 64);
                    let (data, weights) =
                        unstarved_large_kernel_partition(shape, [kernel, kernel], captured);
                    assert!(
                        weights >= captured.1,
                        "k{kernel} Cin {cin} lowered {} to {weights}",
                        captured.1,
                    );
                    assert_eq!(data + weights, CBUF_BANKS, "k{kernel} Cin {cin} sums");
                    assert!(data >= 1, "k{kernel} Cin {cin} left no data bank");
                }
            }
        }
    }

    /// The `11/1` silent-zero finding, kept as a planner invariant.
    ///
    /// An early planner granted `Cin` = `Cout` = 256, k=3 a single
    /// coefficient bank at every extent from 26x26 to 48x48, and the NPU
    /// completed those jobs with all-zero output -- deterministically, every
    /// element, with the pass/fail boundary on the split flips (ISSUES.md
    /// C12; LIMITS.md "Hazards inside the limits"). The same working set at
    /// the 7/5 split the planner grants today is exact at every extent from
    /// 20 to 58 (`planck`, 2026-09-08, fp16 and int8, `selectors` and
    /// `dense`), and forcing the old 11/1 back with `ROCKET_CBUF_SPLIT` is a
    /// watchdog kill with the output unwritten (10/2 too; 9/3 and 8/4 are
    /// exact), so the "all-zero" was a killed job read as a result before C3.
    /// The fault was the grant, not the shape -- the same class as C9's
    /// large-kernel cliff -- and what prevents it is
    /// [`streamed_weight_bank_preference`]: a streamed coefficient working
    /// set needs its banks whatever the feature map asks for.
    ///
    /// Pin that a k=3 plan never grants fewer coefficient banks than the
    /// streamed working set needs, at every extent across the split flips,
    /// so the correction in `demand_based_cbuf_partition` cannot be lost
    /// without this failing.
    #[test]
    fn dense_k3_plan_never_starves_the_streamed_coefficient_working_set() {
        let kernels = [3usize, 3];
        for precision in [Precision::Fp16, captured_int8()] {
            for cin in [64u32, 128, 192, 256, 320, 384, 448, 512] {
                for cout in [64u32, 256, 512] {
                    for extent in (8..=64).step_by(2) {
                        let shape = Shape::with_precision(extent, extent, 1, cin, cout, precision);
                        let plan = ConvPlan::new(shape, kernels);
                        let streamed = streamed_weight_bank_preference(
                            shape.streamed_contraction_channels(),
                            kernels,
                            precision.element_bits(),
                        );
                        // A footprint that fits outright needs only what it
                        // occupies; anything larger is streamed and needs the
                        // working set resident.
                        let needed = streamed.min(shape.weight_bank_demand(kernels));
                        assert!(
                            plan.weight_banks() >= needed,
                            "{precision:?} {extent}x{extent} Cin {cin} Cout {cout} k3: granted {} \
                             coefficient banks, the streamed working set needs {needed}",
                            plan.weight_banks()
                        );
                    }
                }
            }
        }
        // The shape the finding was recorded on, at the extents it tabulated:
        // one split on both sides of the old flips, and never a single bank.
        for extent in [20u32, 24, 26, 30, 36, 48, 50, 58] {
            let shape = Shape::with_precision(extent, extent, 1, 256, 256, Precision::Fp16);
            let plan = ConvPlan::new(shape, kernels);
            assert_eq!(
                (plan.data_banks(), plan.weight_banks()),
                (7, 5),
                "Cin 256 Cout 256 k3 at {extent}x{extent}"
            );
        }
    }

    #[test]
    fn conv_plan_reproduces_the_three_hardware_proven_rectangular_grids() {
        for (kernel, in_channels, banks, columns, tiles) in [
            (9usize, 64u32, (6u32, 6u32), &[135u32, 121][..], 19usize),
            (11, 48, (5, 7), &[137, 119][..], 27),
            (11, 64, (3, 9), &[59, 54, 54, 54, 35][..], 68),
        ] {
            let shape = Shape::with_out_channels(256, 32, 1, in_channels, 64);
            let plan = ConvPlan::new(shape, [kernel, kernel]);
            assert_eq!(
                (plan.data_banks(), plan.weight_banks()),
                banks,
                "k{kernel} Cin {in_channels} banks"
            );
            assert_eq!(
                plan.output_column_widths(),
                columns,
                "k{kernel} Cin {in_channels} columns"
            );
            assert_eq!(
                plan.tiles().len(),
                tiles,
                "k{kernel} Cin {in_channels} tiles"
            );
        }
    }

    /// Captured plan-0 words from the rectangular-kernel sweep, one row per
    /// capture in `rknn-files/sweep-kshape`: `conv-w32-h32-k3x7-s1`, its
    /// mirror `conv-w32-h32-k7x3-s1`, and `conv-w32-h32-k1x11-s1`.
    ///
    /// The mirrored pair is the whole point. `CnaWeightSize2` and
    /// `CnaPadCon0` swap their halves with the kernel, and `CnaConvCon2`
    /// moves with the kernel's height alone -- 36 grains at 3x7 against 42 at
    /// 7x3 -- while the coefficient footprint, which depends on the area,
    /// stays put across the swap.
    const CAPTURED_NON_SQUARE: [(Kernels, u32, u32, u32, u32, u32); 3] = [
        //  kernel     WeightSize2  PadCon0  ConvCon2  WeightSize0  WeightSize1
        ([3, 7], 0x0703_0008, 0x31, 0x240, 0xa80, 0x150),
        ([7, 3], 0x0307_0008, 0x13, 0x2a0, 0xa80, 0x150),
        ([1, 11], 0x0b01_0008, 0x50, 0x210, 0x580, 0x0b0),
    ];

    #[test]
    fn non_square_kernels_program_each_extent_on_its_own_axis() {
        for (kernels, weight_size2, pad_con0, conv_con2, weight_bytes, per_kernel) in
            CAPTURED_NON_SQUARE
        {
            let shape = Shape::CAPTURED;
            let plan = ConvPlan::new(shape, kernels);
            assert_eq!(
                (plan.data_banks(), plan.weight_banks()),
                (1, 11),
                "{kernels:?} CBUF split"
            );
            assert_eq!(plan.tiles().len(), 1, "{kernels:?} tiles");

            let program = &plan.programs()[0];
            assert_eq!(
                value_of::<CnaWeightSize2>(program),
                weight_size2,
                "{kernels:?} weight_width/height"
            );
            assert_eq!(
                value_of::<CnaPadCon0>(program),
                pad_con0,
                "{kernels:?} pad_left/top"
            );
            assert_eq!(
                value_of::<CnaConvCon2>(program),
                conv_con2,
                "{kernels:?} feature_grains"
            );
            assert_eq!(
                value_of::<CnaWeightSize0>(program),
                weight_bytes,
                "{kernels:?} weight_bytes"
            );
            assert_eq!(
                value_of::<CnaWeightSize1>(program),
                per_kernel,
                "{kernels:?} weight_bytes_per_kernel"
            );
            // SAME padding on both axes, so a non-square kernel leaves the
            // output extent alone.
            assert_eq!(
                (shape.output_width(kernels), shape.output_height(kernels)),
                (32, 32),
                "{kernels:?} output extent"
            );
        }
    }

    #[test]
    fn non_square_plans_take_the_captured_split_where_demand_still_decides() {
        // 256x32, Cin 32, Cout 64: the five captures in this shape whose
        // coefficient demand is four or five banks, each matching its
        // capture's plan-0 CBUF split.
        for (kernels, banks) in [
            ([3usize, 9usize], (8u32, 4u32)),
            ([9, 3], (8, 4)),
            ([3, 11], (7, 5)),
            ([11, 3], (7, 5)),
            ([5, 7], (7, 5)),
        ] {
            let shape = Shape::with_out_channels(256, 32, 1, 32, 64);
            let plan = ConvPlan::new(shape, kernels);
            assert_eq!(
                (plan.data_banks(), plan.weight_banks()),
                banks,
                "{kernels:?} CBUF split"
            );
        }
    }

    #[test]
    fn int8_non_square_kernels_program_the_same_geometry_words_as_fp16() {
        // `conv-w32-h32-k3x7-s1-i8` and `conv-w32-h32-k7x3-s1-i8` in
        // rknn-files/sweep-kshape-i8. Every geometry word matches the fp16
        // capture of the same shape exactly -- including the coefficient
        // footprint, where an int8 atom's doubled channel padding and halved
        // element size cancel at these channel counts.
        for (kernels, weight_size2, pad_con0, conv_con2, weight_bytes, per_kernel) in
            CAPTURED_NON_SQUARE
        {
            if kernels == [1, 11] {
                continue; // no int8 capture at this shape
            }
            let shape = Shape::with_precision(32, 32, 1, 3, 8, captured_int8());
            let plan = ConvPlan::new(shape, kernels);
            let program = &plan.programs()[0];
            assert_eq!(
                value_of::<CnaWeightSize2>(program),
                weight_size2,
                "{kernels:?}"
            );
            assert_eq!(value_of::<CnaPadCon0>(program), pad_con0, "{kernels:?}");
            assert_eq!(value_of::<CnaConvCon2>(program), conv_con2, "{kernels:?}");
            assert_eq!(
                value_of::<CnaWeightSize0>(program),
                weight_bytes,
                "{kernels:?}"
            );
            assert_eq!(
                value_of::<CnaWeightSize1>(program),
                per_kernel,
                "{kernels:?}"
            );
        }
    }

    #[test]
    fn int8_non_square_plans_take_the_captured_split() {
        // 256x32, Cin 32, Cout 64 int8: an int8 coefficient is one byte, so
        // these ask for half the fp16 demand and every one stays
        // demand-based -- including 9x7, whose fp16 twin does not.
        for (kernels, banks) in [
            ([3usize, 9usize], (8u32, 4u32)),
            ([9, 3], (8, 4)),
            ([5, 7], (8, 4)),
            ([7, 5], (8, 4)),
        ] {
            let shape = Shape::with_precision(256, 32, 1, 32, 64, captured_int8());
            let plan = ConvPlan::new(shape, kernels);
            assert_eq!(
                (plan.data_banks(), plan.weight_banks()),
                banks,
                "{kernels:?} CBUF split"
            );
        }
    }

    #[test]
    #[should_panic(expected = "stops following coefficient demand")]
    fn conv_plan_refuses_the_non_square_splits_the_captures_do_not_settle() {
        // 11x5 and its mirror 5x11 have the same seven-bank coefficient
        // demand and split differently -- 8/4 against 5/7 -- so demand alone
        // cannot choose and the planner declines to guess.
        let _ = ConvPlan::new(Shape::with_out_channels(256, 32, 1, 32, 64), [11, 5]);
    }

    #[test]
    fn explicit_banks_plan_a_non_square_kernel_the_allocator_refuses() {
        let shape = Shape::with_out_channels(256, 32, 1, 32, 64);
        let plan = ConvPlan::with_cbuf_banks(shape, [11, 5], 8, 4);
        assert_eq!((plan.data_banks(), plan.weight_banks()), (8, 4));
        assert!(!plan.tiles().is_empty());
        assert!(plan.programs().iter().all(|program| program.len() == 136));
    }

    /// The six captured groups, in the order they appear in the regcmd blob:
    /// one 1-core plan, then a 2-core plan, then a 3-core plan.
    const CAPTURED_PLANS: [(u32, u32); 6] = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)];

    fn captured_tile(kernels: Kernels, group: usize) -> Tile {
        let (tiles, index) = CAPTURED_PLANS[group];
        Tile::split(Shape::CAPTURED, kernels, tiles)[index as usize]
    }

    /// First write to `R`. A few registers are written twice per program --
    /// CnaConvCon1 and CnaCbufCon0 among them -- always with the same value.
    fn first_value_of<R: RegisterMeta>(commands: &[RegCmd]) -> u32 {
        commands
            .iter()
            .filter(|command| {
                (command.0 >> 48) as u32 == R::DOMAIN && (command.0 as u32 & 0xffff) == R::OFFSET
            })
            .map(|command| (command.0 >> 16) as u32)
            .next()
            .expect("register is never written")
    }

    fn value_of<R: RegisterMeta>(commands: &[RegCmd]) -> u32 {
        let matches: Vec<u32> = commands
            .iter()
            .filter(|command| {
                (command.0 >> 48) as u32 == R::DOMAIN && (command.0 as u32 & 0xffff) == R::OFFSET
            })
            .map(|command| (command.0 >> 16) as u32)
            .collect();
        assert_eq!(
            matches.len(),
            1,
            "expected exactly one write to the register"
        );
        matches[0]
    }

    #[test]
    fn tile_split_reproduces_vendor_row_ranges() {
        // REG_DPU_DATA_CUBE_HEIGHT decodes to 32, 16/16, and 11/11/10 rows
        // across the three captured plans.
        let rows: Vec<Vec<u32>> = [1, 2, 3]
            .iter()
            .map(|&n| {
                Tile::split(Shape::CAPTURED, [3, 3], n)
                    .iter()
                    .map(|t| t.out_rows)
                    .collect()
            })
            .collect();
        assert_eq!(rows, vec![vec![32], vec![16, 16], vec![11, 11, 10]]);

        // Every plan covers the image exactly once, with no gap or overlap.
        for tiles in 1..=IMAGE_HEIGHT {
            let split = Tile::split(Shape::CAPTURED, [3, 3], tiles);
            assert_eq!(split.iter().map(|t| t.out_rows).sum::<u32>(), IMAGE_HEIGHT);
            for pair in split.windows(2) {
                assert_eq!(pair[0].out_first + pair[0].out_rows, pair[1].out_first);
            }
        }
    }

    #[test]
    fn tile_halo_matches_captured_feature_offsets() {
        // DESIGN_NOTES: the 3x3 programs begin each continuation tile one
        // input row earlier than the 1x1 programs do.
        let three = Tile::split(Shape::CAPTURED, [3, 3], 3);
        let one = Tile::split(Shape::CAPTURED, [1, 1], 3);
        assert_eq!(
            three
                .iter()
                .map(|t| t.input_offset(Shape::CAPTURED))
                .collect::<Vec<_>>(),
            [0x0000, 0x0780, 0x0fc0]
        );
        assert_eq!(
            one.iter()
                .map(|t| t.input_offset(Shape::CAPTURED))
                .collect::<Vec<_>>(),
            [0x0000, 0x0840, 0x1080]
        );
        assert_eq!(
            three
                .iter()
                .map(|t| t.output_offset(Shape::CAPTURED, [3, 3]))
                .collect::<Vec<_>>(),
            [0x0000, 0x1600, 0x2c00]
        );

        // Rows actually read, including halo and excluding padded rows.
        assert_eq!(
            three.iter().map(|t| t.in_rows).collect::<Vec<_>>(),
            [12, 13, 11]
        );
        assert_eq!(
            three.iter().map(|t| t.pad_top).collect::<Vec<_>>(),
            [1, 0, 0]
        );
    }

    #[test]
    fn short_large_kernel_tiles_retain_remaining_top_padding() {
        // Hardware exposed this at 7x7/Cin64: two output rows fit per tile,
        // so the second tile starts at output row 2 while one of the three
        // top-padding rows is still in force. Treating every continuation
        // tile as unpadded shifted that tile's convolution down by one row.
        let shape = Shape::with_out_channels(256, 32, 1, 64, 64);
        let split = Tile::split(shape, [7, 7], 16);
        assert_eq!(
            split
                .iter()
                .take(3)
                .map(|tile| tile.out_first)
                .collect::<Vec<_>>(),
            [0, 2, 4]
        );
        assert_eq!(
            split
                .iter()
                .take(3)
                .map(|tile| tile.pad_top)
                .collect::<Vec<_>>(),
            [3, 1, 0]
        );

        // The one-row 11x11/Cin64 plan keeps decreasing the padding until
        // the sixth output row finally has a complete real-input footprint.
        let split = Tile::split(shape, [11, 11], 32);
        assert_eq!(
            split
                .iter()
                .take(7)
                .map(|tile| tile.pad_top)
                .collect::<Vec<_>>(),
            [5, 4, 3, 2, 1, 0, 0]
        );
    }

    #[test]
    fn column_tiles_reproduce_large_kernel_capture_boundaries() {
        let k9 = Shape::with_out_channels(256, 32, 1, 64, 64);
        assert_eq!(
            ColumnTile::split(k9, [9, 9], &[135, 121]),
            [
                ColumnTile {
                    out_first: 0,
                    out_cols: 135,
                    in_first: 0,
                    in_cols: 139,
                    pad_left: 4,
                },
                ColumnTile {
                    out_first: 135,
                    out_cols: 121,
                    in_first: 131,
                    in_cols: 125,
                    pad_left: 0,
                },
            ]
        );

        let k11_c48 = Shape::with_out_channels(256, 32, 1, 48, 64);
        assert_eq!(
            ColumnTile::split(k11_c48, [11, 11], &[137, 119]),
            [
                ColumnTile {
                    out_first: 0,
                    out_cols: 137,
                    in_first: 0,
                    in_cols: 142,
                    pad_left: 5,
                },
                ColumnTile {
                    out_first: 137,
                    out_cols: 119,
                    in_first: 132,
                    in_cols: 124,
                    pad_left: 0,
                },
            ]
        );

        let k11_c64 = Shape::with_out_channels(256, 32, 1, 64, 64);
        let columns = ColumnTile::split(k11_c64, [11, 11], &[59, 54, 54, 54, 35]);
        assert_eq!(
            columns
                .iter()
                .map(|tile| (tile.out_first, tile.in_first, tile.in_cols, tile.pad_left))
                .collect::<Vec<_>>(),
            [
                (0, 0, 64, 5),
                (59, 54, 64, 0),
                (113, 108, 64, 0),
                (167, 162, 64, 0),
                (221, 216, 40, 0),
            ]
        );
    }

    #[test]
    fn horizontal_programming_matches_captured_11x11_tiles() {
        let shape = Shape::with_out_channels(256, 32, 1, 64, 64);
        let columns = ColumnTile::split(shape, [11, 11], &[59, 54, 54, 54, 35]);
        let rows = Tile {
            out_first: 0,
            out_rows: 7,
            in_first: 0,
            in_rows: 12,
            pad_top: 5,
        };

        let left = conv_2d_tile_2d_with_cbuf_banks(
            shape,
            [11, 11],
            &Tile2D {
                rows,
                columns: columns[0],
            },
            3,
            9,
        );
        assert_eq!(first_value_of::<CnaConvCon1>(&left), 0x2000_0120);
        assert_eq!(first_value_of::<CnaCbufCon0>(&left), 0x93);
        assert_eq!(value_of::<CnaDataSize0>(&left), 0x0040_000c);
        assert_eq!(value_of::<CnaDataSize2>(&left), 59);
        assert_eq!(value_of::<CnaDataSize3>(&left), 59 * 7);
        assert_eq!(value_of::<CnaCbufCon1>(&left), 128);
        assert_eq!(value_of::<CnaPadCon0>(&left), 0x55);
        assert_eq!(value_of::<CnaDmaCon1>(&left), 256);
        assert_eq!(value_of::<CnaDmaCon2>(&left), 8192 - 64);
        assert_eq!(value_of::<CnaFcDataSize0>(&left), 0x0040_000c);
        assert_eq!(value_of::<CoreDataoutSize0>(&left), 0x0006_003a);
        assert_eq!(value_of::<DpuDataCubeNotchAddr>(&left), 0x00c5_00c5);
        assert_eq!(value_of::<DpuDstSurfStride>(&left), 256 * 32 * 16);
        assert_eq!(value_of::<DpuSurfaceAdd>(&left), 256 * 32 * 32);

        let middle = conv_2d_tile_2d_with_cbuf_banks(
            shape,
            [11, 11],
            &Tile2D {
                rows,
                columns: columns[1],
            },
            3,
            9,
        );
        assert_eq!(value_of::<CnaDataSize2>(&middle), 54);
        assert_eq!(value_of::<CnaPadCon0>(&middle), 0x05);
        assert_eq!(value_of::<CnaFeatureDataAddr>(&middle), 54 * 16);
        assert_eq!(value_of::<DpuDstBaseAddr>(&middle), 59 * 16);
        assert_eq!(value_of::<DpuDataCubeNotchAddr>(&middle), 0x00ca_00ca);
    }

    #[test]
    fn tile_registers_match_all_six_captured_groups() {
        // Observed values, group 1 through group 6, from the bitbiter reports
        // in the design-spike repo (conv-group-N.md).
        const FEATURE_GRAINS: [[u32; 6]; 2] = [
            [36, 21, 20, 16, 16, 14], // 3x3
            [33, 17, 17, 12, 12, 11], // 1x1
        ];
        const DATA_SIZE0: [[u32; 6]; 2] = [
            [
                0x20_0020, 0x20_0011, 0x20_0011, 0x20_000c, 0x20_000d, 0x20_000b,
            ],
            [
                0x20_0020, 0x20_0010, 0x20_0010, 0x20_000b, 0x20_000b, 0x20_000a,
            ],
        ];
        const CBUF_CON1: [[u32; 6]; 2] = [
            [0x400, 0x220, 0x220, 0x180, 0x1a0, 0x160],
            [0x400, 0x200, 0x200, 0x160, 0x160, 0x140],
        ];
        const PAD_CON0: [[u32; 6]; 2] = [
            [0x11, 0x11, 0x10, 0x11, 0x10, 0x10],
            [0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
        ];
        const FEATURE_ADDR: [[u32; 6]; 2] = [
            [0, 0, 0xb40, 0, 0x780, 0xfc0],
            [0, 0, 0xc00, 0, 0x840, 0x1080],
        ];
        // Output geometry is identical for both kernels: same output tile.
        const DATA_SIZE3: [u32; 6] = [0x400, 0x200, 0x200, 0x160, 0x160, 0x140];
        const DST_BASE: [u32; 6] = [0, 0, 0x2000, 0, 0x1600, 0x2c00];
        const CUBE_HEIGHT: [u32; 6] = [0x1f, 0x0f, 0x0f, 0x0a, 0x0a, 0x09];
        const DATAOUT_SIZE0: [u32; 6] = [
            0x1f_001f, 0x0f_001f, 0x0f_001f, 0x0a_001f, 0x0a_001f, 0x09_001f,
        ];

        for (row, kernels) in [(0usize, [3usize, 3]), (1, [1, 1])] {
            for group in 0..6 {
                let tile = captured_tile(kernels, group);
                let program = conv_2d_tile(Shape::CAPTURED, kernels, &tile);
                let at = |what: &str, got: u32, want: u32| {
                    assert_eq!(
                        got,
                        want,
                        "{what} mismatch, {kernels:?} group {}",
                        group + 1
                    );
                };

                // FEATURE_GRAINS is a field, not a whole register: the vendor's
                // multi-core programs also carry a plan index in bits 31:28,
                // which a standalone tile program does not reproduce.
                let grains = (value_of::<CnaConvCon2>(&program) & 0x0000_3ff0) >> 4;
                at("feature_grains", grains, FEATURE_GRAINS[row][group]);

                at(
                    "data_size0",
                    value_of::<CnaDataSize0>(&program),
                    DATA_SIZE0[row][group],
                );
                at(
                    "fc_data_size0",
                    value_of::<CnaFcDataSize0>(&program),
                    DATA_SIZE0[row][group],
                );
                at(
                    "cbuf_con1",
                    value_of::<CnaCbufCon1>(&program),
                    CBUF_CON1[row][group],
                );
                at(
                    "pad_con0",
                    value_of::<CnaPadCon0>(&program),
                    PAD_CON0[row][group],
                );
                at(
                    "feature_data_addr",
                    value_of::<CnaFeatureDataAddr>(&program),
                    FEATURE_ADDR[row][group],
                );
                at(
                    "data_size3",
                    value_of::<CnaDataSize3>(&program),
                    DATA_SIZE3[group],
                );
                at(
                    "dst_base_addr",
                    value_of::<DpuDstBaseAddr>(&program),
                    DST_BASE[group],
                );
                at(
                    "data_cube_height",
                    value_of::<DpuDataCubeHeight>(&program),
                    CUBE_HEIGHT[group],
                );
                at(
                    "rdma_data_cube_height",
                    value_of::<DpuRdmaDataCubeHeight>(&program),
                    CUBE_HEIGHT[group],
                );
                at(
                    "core_dataout_size0",
                    value_of::<CoreDataoutSize0>(&program),
                    DATAOUT_SIZE0[group],
                );
                at(
                    "wdma_size1",
                    value_of::<DpuWdmaSize1>(&program),
                    DATAOUT_SIZE0[group],
                );
            }
        }
    }

    #[test]
    fn cbuf_bank_split_matches_every_swept_shape() {
        // (width, height) -> data_bank, read out of CNA_CBUF_CON0 across the
        // 134 C3/Cout8/stride-1 programs in the shape-sweep corpus. Every one
        // also satisfies data_bank + weight_bank == 12.
        const OBSERVED: [(u32, u32, u32); 11] = [
            (32, 32, 1),
            (32, 64, 2),
            (32, 128, 4),
            (32, 256, 8),
            (64, 32, 2),
            (64, 64, 4),
            (96, 32, 3),
            (128, 32, 4),
            (128, 128, 11),
            (192, 32, 6),
            (256, 32, 8),
        ];
        for (width, height, data_bank) in OBSERVED {
            let shape = Shape::new(width, height);
            assert_eq!(
                shape.data_banks([3, 3]),
                data_bank,
                "data_bank for {width}x{height}"
            );
            assert_eq!(
                shape.data_banks([3, 3]) + shape.weight_banks([3, 3]),
                12,
                "bank split for {width}x{height} must cover all 12 CBUF banks"
            );
        }
    }

    #[test]
    fn wider_shapes_scale_the_geometry_registers() {
        // Formulas validated against 212 C3 stride-1 programs from 35 captures.
        // 256 wide caps a tile at 44 input rows by the vendor's own CBUF
        // rule (11 data banks x 1024 pixels / 256), so 64 rows need two.
        let shape = Shape::new(256, 64);
        assert_eq!(shape.max_tile_input_rows([3, 3]), 44);
        assert_eq!(shape.min_tiles([3, 3]), 2);

        let split = Tile::split(shape, [3, 3], 2);
        let tile = split[0];
        let program = conv_2d_tile(shape, [3, 3], &tile);

        assert_eq!(
            value_of::<CnaDataSize0>(&program),
            (256 << 16) | tile.in_rows
        );
        assert_eq!(value_of::<CnaCbufCon1>(&program), 256 * tile.in_rows);
        assert_eq!(value_of::<CnaDataSize3>(&program), 256 * tile.out_rows);
        assert_eq!(
            value_of::<CoreDataoutSize0>(&program),
            ((tile.out_rows - 1) << 16) | 255
        );
        assert_eq!(value_of::<DpuDataCubeHeight>(&program), tile.out_rows - 1);

        // Row strides follow the dense NHWC input and C8 fp16 output.
        assert_eq!(shape.input_row_stride(), 256 * 3 * 2);
        assert_eq!(shape.output_row_stride([3, 3]), 256 * 8 * 2);

        // A three-way split of 64 rows, with the 3x3 halo on continuations.
        let three = Tile::split(shape, [3, 3], 3);
        assert_eq!(
            three.iter().map(|t| t.out_rows).collect::<Vec<_>>(),
            [22, 21, 21]
        );
        assert_eq!(three[1].in_first, 21);
        assert_eq!(three[1].output_offset(shape, [3, 3]), 22 * 256 * 8 * 2);
    }

    #[test]
    fn vendor_capacity_rule_matches_the_width_sweep() {
        // Largest tile the vendor emits at each width, from the width sweep:
        // data_banks * 1024 / width, exact at every measured point.
        for (width, rows) in [(256u32, 32u32), (512, 22), (768, 14), (1024, 11), (1536, 7)] {
            let shape = Shape::new(width, 32);
            assert_eq!(
                shape.max_tile_input_rows([3, 3]),
                rows,
                "max tile rows at {width} wide"
            );
        }
    }

    #[test]
    fn stride_scales_output_geometry_and_halo() {
        // Formulas confirmed on 150 stride-2, -3 and -4 programs.
        let shape = Shape::with_stride(128, 64, 2);
        assert_eq!(shape.output_width([3, 3]), 64);
        assert_eq!(shape.output_height([3, 3]), 32);
        assert_eq!(shape.output_row_stride([3, 3]), 64 * 8 * 2);

        // A two-way split of the 32 output rows. The continuation tile
        // projects back through the stride: 16 * 2 - 1 = 31.
        let split = Tile::split(shape, [3, 3], 2);
        assert_eq!(
            split.iter().map(|t| t.out_rows).collect::<Vec<_>>(),
            [16, 16]
        );
        assert_eq!(split[0].in_first, 0);
        assert_eq!(split[0].in_rows, 32);
        assert_eq!(split[0].pad_top, 1);
        assert_eq!(split[1].in_first, 31);
        assert_eq!(split[1].in_rows, 33);
        assert_eq!(split[1].pad_top, 0);

        // The stride reaches CNA_CONV_CON3, and output-side registers carry
        // output geometry rather than input.
        let program = conv_2d_tile(shape, [3, 3], &split[0]);
        assert_eq!(value_of::<CnaConvCon3>(&program), (2 << 3) | 2);
        assert_eq!(value_of::<CnaDataSize2>(&program), 64);
        // Raw word: the DST_SURF_STRIDE field is shifted 4, so the encoded
        // word is sixteen times the 64 * 32 field value.
        assert_eq!(value_of::<DpuDstSurfStride>(&program), 64 * 32 * 16);

        // Stride 1 is unchanged.
        let flat = Shape::new(128, 64);
        assert_eq!(flat.output_height([3, 3]), 64);
        assert_eq!(Tile::split(flat, [3, 3], 2)[1].in_first, 31);
    }

    #[test]
    fn channel_layout_boundary_is_half_an_atom() {
        // Cin 4 is the last dense case (8 bytes); Cin 5 is already surfaces.
        // Measured directly: line_stride/width is 1.00 up to Cin 4 and 4.00
        // from Cin 5 onward.
        for cin in 1..=4 {
            assert_eq!(
                Shape::with_channels(32, 32, 1, cin).layout(),
                FeatureLayout::Dense,
                "Cin {cin}"
            );
        }
        for cin in [5, 6, 7, 8, 16, 80] {
            assert_eq!(
                Shape::with_channels(32, 32, 1, cin).layout(),
                FeatureLayout::Surfaces,
                "Cin {cin}"
            );
        }
    }

    #[test]
    fn channel_padding_matches_the_fill_in_sweep() {
        // (Cin, datain_channel, weight_channels) read out of the captures.
        // The two exceptions are 3 atoms, where the fields disagree, and
        // 7 atoms, where both round up.
        const OBSERVED: [(u32, u32, u32); 16] = [
            (3, 8, 8),
            (4, 8, 8),
            (5, 8, 8),
            (8, 8, 8),
            (9, 16, 16),
            (12, 16, 16),
            (16, 16, 16),
            (20, 24, 32),
            (24, 24, 32),
            (28, 32, 32),
            (32, 32, 32),
            (36, 40, 40),
            (40, 40, 40),
            (48, 48, 48),
            // Seven atoms pads the coefficients to 64 but leaves
            // datain_channel at 56, the same split three atoms makes. This
            // row read (56, 64, 64) until a field-by-field comparison
            // against the whole corpus showed the capture programs 56.
            (56, 56, 64),
            (64, 64, 64),
        ];
        for (cin, padded, weights) in OBSERVED {
            let shape = Shape::with_channels(32, 32, 1, cin);
            assert_eq!(shape.padded_channels(), padded, "datain_channel Cin {cin}");
            assert_eq!(
                shape.weight_channels(),
                weights,
                "weight channels Cin {cin}"
            );
        }
        // 72 and 80 are unpadded, confirming this is not a power-of-two rule.
        assert_eq!(Shape::with_channels(32, 32, 1, 72).weight_channels(), 72);
        assert_eq!(Shape::with_channels(32, 32, 1, 80).weight_channels(), 80);
    }

    #[test]
    fn fp16_channel_padding_follows_the_quad_atom_rule_to_512() {
        // (Cin, datain_channel, weight_channels) from the large-Cin sweep.
        // Read against the old two-entry table, 24 and 56 look like a
        // `2**n - 1` rule; 88, 120, 152 and the rest show it is not. Every
        // row where the two counts disagree has an atom count of 3 mod 4,
        // and the rows on either side of each are here to keep a rule
        // separable from a table.
        const OBSERVED: [(u32, u32, u32); 24] = [
            (81, 88, 96),
            (88, 88, 96),
            (89, 96, 96),
            (96, 96, 96),
            (104, 104, 104),
            (112, 112, 112),
            (113, 120, 128),
            (120, 120, 128),
            (121, 128, 128),
            (128, 128, 128),
            (136, 136, 136),
            (144, 144, 144),
            (152, 152, 160),
            (160, 160, 160),
            (184, 184, 192),
            (185, 192, 192),
            (216, 216, 224),
            (224, 224, 224),
            (248, 248, 256),
            (256, 256, 256),
            (280, 280, 288),
            (344, 344, 352),
            (440, 440, 448),
            (504, 504, 512),
        ];
        for (cin, padded, weights) in OBSERVED {
            let shape = Shape::with_channels(32, 32, 1, cin);
            assert_eq!(shape.padded_channels(), padded, "datain_channel Cin {cin}");
            assert_eq!(
                shape.weight_channels(),
                weights,
                "weight channels Cin {cin}"
            );
        }
    }

    #[test]
    fn int8_channel_padding_stays_exact_where_fp16_bumps() {
        // The same atom counts that bump at fp16 -- 11 and 15, here 176 and
        // 240 -- pass through unchanged at int8, out to 512. The coefficient
        // padding never leaves `padded_channels`.
        for cin in [
            129u32, 144, 175, 176, 177, 225, 240, 241, 304, 368, 432, 496, 512,
        ] {
            let shape = Shape::with_precision(32, 32, 1, cin, 8, Precision::Int8(quantization()));
            let whole_atoms = cin.div_ceil(16) * 16;
            assert_eq!(
                shape.padded_channels(),
                whole_atoms,
                "int8 datain Cin {cin}"
            );
            assert_eq!(
                shape.weight_channels(),
                whole_atoms,
                "int8 weights Cin {cin}"
            );
        }
    }

    #[test]
    fn cbuf_atom_charge_rounds_to_whole_groups_in_both_precisions() {
        // Recovered from `CNA_CBUF_CON1.data_entries`, which carries
        // `input_width * cbuf_atoms / 4` in the surface regime. The two-entry
        // version this replaced was right only below the old ceilings; every
        // value here past them is one it charged an atom short.
        for (cin, charged) in [
            (32u32, 4u32),
            (48, 6),
            (56, 8),
            (88, 12),
            (96, 12),
            (104, 13),
            (112, 14),
            (120, 16),
            (128, 16),
            (152, 20),
            (184, 24),
            (216, 28),
            (248, 32),
            (280, 36),
            (344, 44),
            (440, 56),
            (504, 64),
        ] {
            assert_eq!(
                Shape::with_channels(32, 32, 1, cin).cbuf_atoms(),
                charged,
                "fp16 CBUF atom charge at Cin {cin}"
            );
        }

        // int8 charges the same way even though its padding does not bump.
        for (cin, charged) in [
            (48u32, 4u32),
            (112, 8),
            (176, 12),
            (192, 12),
            (240, 16),
            (304, 20),
            (368, 24),
            (432, 28),
            (496, 32),
        ] {
            let shape = Shape::with_precision(32, 32, 1, cin, 8, Precision::Int8(quantization()));
            assert_eq!(
                shape.cbuf_atoms(),
                charged,
                "int8 CBUF atom charge at Cin {cin}"
            );
        }
    }

    #[test]
    fn argb_input_mode_follows_the_layout() {
        // CNA_CONV_CON1 across the channel sweep: dense programs the ARGB
        // image path with nonalign_dma and group_line_off set, surfaces
        // clear all three. Cin 3 -> 10/1/1, Cin 4 -> 11/1/1, Cin >= 5 ->
        // 0/0/0. Leaving these at the C3 values made every channel count
        // read as three channels.
        const ARGB_IN: u32 = 0x0000_f000;
        const NONALIGN_DMA: u32 = 0x4000_0000;
        const GROUP_LINE_OFF: u32 = 0x2000_0000;
        let field = |program: &[RegCmd], mask: u32| {
            (first_value_of::<CnaConvCon1>(program) & mask) >> (mask.trailing_zeros())
        };

        for (cin, argb) in [(3u32, 10u32), (4, 11)] {
            let shape = Shape::with_channels(32, 32, 1, cin);
            let program = conv_2d_tile(shape, [3, 3], &Tile::whole(shape, [3, 3]));
            assert_eq!(field(&program, ARGB_IN), argb, "argb_in at Cin {cin}");
            assert_eq!(field(&program, NONALIGN_DMA), 1, "nonalign at Cin {cin}");
            assert_eq!(
                field(&program, GROUP_LINE_OFF),
                1,
                "group_line at Cin {cin}"
            );
        }
        for cin in [5u32, 8, 16, 24, 64] {
            let shape = Shape::with_channels(32, 32, 1, cin);
            let program = conv_2d_tile(shape, [3, 3], &Tile::whole(shape, [3, 3]));
            assert_eq!(field(&program, ARGB_IN), 0, "argb_in at Cin {cin}");
            assert_eq!(field(&program, NONALIGN_DMA), 0, "nonalign at Cin {cin}");
            assert_eq!(
                field(&program, GROUP_LINE_OFF),
                0,
                "group_line at Cin {cin}"
            );
        }
    }

    #[test]
    fn channel_bank_split_matches_the_sweep() {
        // At 32x32 the surface rule reduces to ceil(weight_atoms / 2).
        for (cin, banks) in [
            (8u32, 1u32),
            (16, 1),
            (20, 2),
            (32, 2),
            (40, 3),
            (48, 3),
            (56, 4),
            (64, 4),
            (72, 5),
            (80, 5),
        ] {
            assert_eq!(
                Shape::with_channels(32, 32, 1, cin).data_banks([3, 3]),
                banks,
                "data_bank at Cin {cin}"
            );
        }
        // Dense stays on the pixel rule it was derived with.
        assert_eq!(Shape::with_channels(128, 32, 1, 3).data_banks([3, 3]), 4);
        assert_eq!(Shape::with_channels(128, 32, 1, 4).data_banks([3, 3]), 4);
        // ...and the surface rule diverges immediately at the boundary.
        assert_eq!(Shape::with_channels(128, 32, 1, 8).data_banks([3, 3]), 2);
    }

    #[test]
    fn multi_channel_registers_match_the_captures() {
        // conv-w128-h32-k3-s1-ci16-co8: line_stride 512, surf_stride 3584,
        // data_entries 64, datain_channel 16, data_bank 4.
        let shape = Shape::with_channels(128, 32, 1, 16);
        let program = conv_2d_tile(shape, [3, 3], &Tile::whole(shape, [3, 3]));
        assert_eq!(value_of::<CnaDmaCon1>(&program), 512);
        assert_eq!(value_of::<CnaDmaCon2>(&program), 3584);
        assert_eq!(value_of::<CnaCbufCon1>(&program), 64);
        assert_eq!(value_of::<CnaDataSize1>(&program) & 0xffff, 16);
        assert_eq!(shape.data_banks([3, 3]), 4);
        // Weight footprint follows the weight padding, not datain_channel.
        assert_eq!(value_of::<CnaWeightSize1>(&program), 9 * 16 * 2);
        assert_eq!(value_of::<CnaWeightSize0>(&program), 9 * 16 * 2 * 8);

        // Cin 24 is the case where the two paddings disagree: datain_channel
        // stays 24 while the coefficients occupy 32 channels.
        let odd = Shape::with_channels(32, 32, 1, 24);
        let odd_program = conv_2d_tile(odd, [3, 3], &Tile::whole(odd, [3, 3]));
        assert_eq!(value_of::<CnaDataSize1>(&odd_program) & 0xffff, 24);
        assert_eq!(value_of::<CnaWeightSize1>(&odd_program), 9 * 32 * 2);
        assert_eq!(value_of::<CnaCbufCon1>(&odd_program), 32);
    }

    #[test]
    fn surface_data_entries_rounds_up_at_widths_not_a_multiple_of_four() {
        // Every surface `data_entries` capture before this one used a width
        // divisible by 4, where floor and ceiling division agree. Vendor
        // captures at Cin=8 (one atom/pixel) expose the real rule: width
        // 13/29/30/31 program data_entries 4/8/8/8, not the 3/7/7/7 floor
        // division used to compute. The 30-wide, Cout=16 case is the shape
        // a real compiled model hit as scattered-pixel corruption on
        // hardware.
        for (width, data_entries) in [(13u32, 4u32), (29, 8), (30, 8), (31, 8)] {
            let shape = Shape::with_out_channels(width, 16, 1, 8, 16);
            let program = conv_2d_tile(shape, [1, 1], &Tile::whole(shape, [1, 1]));
            assert_eq!(
                value_of::<CnaCbufCon1>(&program),
                data_entries,
                "data_entries at width {width}"
            );
        }
    }

    #[test]
    fn entry_slab_base_bounds_the_input_width() {
        // Every point is a measured pass/fail pair on `planck` 2026-09-05:
        // the last width that is exact and the first that is not, at 1x1.
        let unpadded = |width, cin, precision| {
            Shape::with_precision(width, 1, 1, cin, 64, precision).with_padding([0, 0])
        };
        for (cin, precision, last_exact) in [
            (1792u32, Precision::Fp16, 37u32),
            (1024, Precision::Fp16, 66),
            (768, Precision::Fp16, 89),
            (512, Precision::Fp16, 136),
            (256, Precision::Fp16, 292),
            (128, Precision::Fp16, 682),
            (96, Precision::Fp16, 1023),
            // Nine atoms is three slabs, the third of them partial.
            (72, Precision::Fp16, 1023),
            (768, Precision::Bf16, 89),
            (768, Precision::Int16, 89),
            // Half the channels per atom, so K 384 is the same 24 slabs.
            (384, Precision::Tf32, 89),
        ] {
            let wide = unpadded(2047, cin, precision);
            assert_eq!(
                wide.max_tile_input_width(),
                last_exact,
                "Cin {cin} {precision:?}"
            );
            let bounded = unpadded(last_exact, cin, precision);
            assert_eq!(bounded.max_tile_input_width(), last_exact);
        }
        // One slab (Cin <= 32 at fp16) and dense rows have no base to
        // overflow: 2000 wide at K 32 is measured exact.
        assert_eq!(
            unpadded(2000, 32, Precision::Fp16).max_tile_input_width(),
            2000
        );
        assert_eq!(
            unpadded(2000, 40, Precision::Fp16).max_tile_input_width(),
            2000
        );
        assert_eq!(
            unpadded(2000, 3, Precision::Fp16).max_tile_input_width(),
            2000
        );
    }

    #[test]
    fn wide_lines_plan_column_tiles_inside_the_slab_bound() {
        // 89x1 at K 768 is the last full-width line the hardware reads
        // correctly; 90 has to split, and the split has to keep every
        // column inside the bound. Rows do not buy anything: 90x4 splits
        // the same way.
        let shape = |width, height| {
            Shape::with_precision(width, height, 1, 768, 64, Precision::Fp16).with_padding([0, 0])
        };
        let whole = ConvPlan::new(shape(89, 1), [1, 1]);
        assert_eq!(whole.output_column_widths(), &[89]);

        for (width, height, columns) in [(90u32, 1u32, 2usize), (90, 4, 2), (197, 1, 3)] {
            let plan = ConvPlan::new(shape(width, height), [1, 1]);
            assert_eq!(
                plan.output_column_widths().len(),
                columns,
                "{width}x{height} column count"
            );
            assert!(
                plan.tiles().iter().all(|tile| tile.columns.in_cols <= 89),
                "{width}x{height} column widths {:?}",
                plan.output_column_widths()
            );
            // Every program emits, which is where the emitter's own guard
            // would otherwise fire.
            assert_eq!(plan.programs().len(), plan.tiles().len());
        }
    }

    #[test]
    fn a_line_that_does_not_fit_its_data_banks_is_not_forced_through() {
        // Measured pairs on `planck` 2026-09-05, 3x3 pad 1 at height one:
        // the coefficient floor leaves 4 data banks at Cin 768 and 3 at
        // Cin 512/1024, and the single line either fits them or does not.
        let shape = |width, cin| Shape::with_precision(width, 1, 1, cin, 64, Precision::Fp16);
        for (width, cin, banks, fits) in [
            (85u32, 768u32, 4u32, true),
            (86, 768, 4, false),
            (89, 512, 3, true),
            (100, 512, 3, false),
            (128, 512, 3, false),
            (62, 1024, 3, false),
        ] {
            let capacity = shape(width, cin).max_tile_input_rows_for_data_banks(banks);
            assert_eq!(
                capacity >= 1,
                fits,
                "{width}x1 Cin {cin} at {banks} data banks"
            );
            let plan = ConvPlan::new(shape(width, cin), [3, 3]);
            assert_eq!(plan.data_banks(), banks, "{width}x1 Cin {cin} split");
            assert_eq!(
                plan.output_column_widths().len() == 1,
                fits,
                "{width}x1 Cin {cin} columns {:?}",
                plan.output_column_widths()
            );
            assert_eq!(plan.programs().len(), plan.tiles().len());
        }
    }

    #[test]
    #[should_panic(expected = "past the 2047 the CBUF can address")]
    fn emitter_refuses_a_line_past_the_slab_bound() {
        let shape = Shape::with_precision(90, 1, 1, 768, 64, Precision::Fp16).with_padding([0, 0]);
        let _ = conv_2d_tile(shape, [1, 1], &Tile::whole(shape, [1, 1]));
    }

    #[test]
    #[should_panic(expected = "input channels must be")]
    fn rejects_channels_beyond_the_validated_range() {
        // 96 was beyond it until the large-Cin sweep; 513 is the first value
        // past what either precision now measures.
        let _ = Shape::with_channels(32, 32, 1, MAX_INPUT_CHANNELS + 1);
    }

    #[test]
    fn output_channels_pad_to_whole_granules() {
        // Every value the corpus covers. Unlike the input padding this is a
        // rule rather than a table: no exceptions at 20, 24, 40, 56 or 72,
        // the values where the input padding needed them.
        for (out_channels, padded) in [
            (1u32, 16u32),
            (2, 16),
            (8, 16),
            (14, 16),
            (16, 16),
            (20, 32),
            (24, 32),
            (28, 32),
            (32, 32),
            (40, 48),
            (48, 48),
            (56, 64),
            (64, 64),
            (72, 80),
            (80, 80),
            (96, 96),
            (128, 128),
            (256, 256),
            (512, 512),
        ] {
            let shape = Shape::with_out_channels(32, 32, 1, 3, out_channels);
            assert_eq!(
                shape.padded_out_channels(),
                padded,
                "padded output channels at Cout {out_channels}"
            );
        }
    }

    #[test]
    fn output_channel_registers_match_the_captures() {
        // conv-w32-h32-k3-s1-ci3-co40, single-core plan: weight_bytes 5760,
        // weight_kernels 40, orig_channel 39, and the four padded-count
        // registers 47 -- Cout 40 rounds to 48. Cout 40 is chosen because
        // every one of those five numbers is distinct there.
        let shape = Shape::with_out_channels(32, 32, 1, 3, 40);
        let program = conv_2d_tile(shape, [3, 3], &Tile::whole(shape, [3, 3]));
        assert_eq!(value_of::<CnaWeightSize0>(&program), 5760);
        assert_eq!(value_of::<CnaWeightSize1>(&program), 144);
        assert_eq!(value_of::<CnaWeightSize2>(&program), 50_528_296);
        assert_eq!(value_of::<DpuDataCubeChannel>(&program), 2_555_951);
        assert_eq!(value_of::<DpuWdmaSize0>(&program), 47);
        assert_eq!(value_of::<CoreDataoutSize1>(&program), 47);

        // The output is NC1HWC2 regardless of Cout: the channel count sets
        // how many 8-channel surfaces there are, not how wide a row is, so
        // both output strides are the same as the Cout 8 capture's.
        assert_eq!(value_of::<DpuDstSurfStride>(&program), 32 * 32 * 16);

        // Cout 9 is the smallest value where the true and padded counts
        // differ by more than the granule rounding of a multiple of 8.
        let odd = Shape::with_out_channels(32, 32, 1, 3, 9);
        let odd_program = conv_2d_tile(odd, [3, 3], &Tile::whole(odd, [3, 3]));
        assert_eq!(value_of::<CnaWeightSize0>(&odd_program), 1296);
        assert_eq!(value_of::<DpuDataCubeChannel>(&odd_program), 524_303);
        assert_eq!(value_of::<DpuWdmaSize0>(&odd_program), 15);
    }

    #[test]
    fn output_channels_reach_the_bank_split_only_through_the_weight_footprint() {
        // The one place in the corpus where Cout moves the CBUF allocation.
        // At 256x32 with Cin 32 the feature data wants 16 banks and cannot
        // have them, so whatever the weights take comes straight off it.
        let narrow = Shape::with_out_channels(256, 32, 1, 32, 16);
        assert_eq!(narrow.weight_bytes([3, 3]), 9216); // one bank
        assert_eq!(narrow.data_banks([3, 3]), 11);
        assert_eq!(narrow.weight_banks([3, 3]), 1);

        let wide = Shape::with_out_channels(256, 32, 1, 32, 64);
        assert_eq!(wide.weight_bytes([3, 3]), 36864); // two banks
        assert_eq!(wide.data_banks([3, 3]), 10);
        assert_eq!(wide.weight_banks([3, 3]), 2);

        // Where the data is *not* over budget, a bigger kernel set changes
        // nothing: the weights are taking slack that was already theirs.
        for out_channels in [8u32, 16, 64, 128] {
            let shape = Shape::with_out_channels(32, 32, 1, 3, out_channels);
            assert_eq!(
                shape.data_banks([3, 3]),
                1,
                "32x32 Cin 3 keeps one data bank at Cout {out_channels}"
            );
        }

        // And when the weights are the larger claim they are the ones cut
        // short: 589824 bytes want 18 banks, more than the CBUF holds, but
        // the feature data still gets the 4 it asked for.
        let huge = Shape::with_out_channels(32, 32, 1, 64, 512);
        assert_eq!(huge.weight_bytes([3, 3]), 589_824);
        assert_eq!(huge.data_banks([3, 3]), 4);
        assert_eq!(huge.weight_banks([3, 3]), 8);
    }

    /// The output-parity rule and both accumulator refusals are gone, and
    /// this pins *why* rather than merely asserting the absence.
    ///
    /// The rule was: a dense accumulator tile is correct only when
    /// `tile_pixels * blocks_per_pixel` is even, because the DPU commits
    /// output in whole 256-byte units. That was a property of the **serial**
    /// writer (`mc_surf_out = 1`) the dense accumulator used to drive, whose
    /// blocks are 128 bytes -- so an odd block count left a trailing half-unit
    /// unwritten.
    ///
    /// The writer is now `mc_surf_out = 0` / `size_e = 7` /
    /// `surf_add = dataout * 8`, whose cube is 16-byte atoms of C2 = 4 int32
    /// lanes. `blocks_per_pixel` is then `padded_out_channels / 4`, and
    /// `padded_out_channels` is always a multiple of the 32-channel granule,
    /// so the block count is always a multiple of 8 -- **even by
    /// construction, at every shape**. The rule cannot fire, which is the
    /// arithmetic reason the padding is unreachable rather than merely
    /// untriggered by the cases tried.
    #[test]
    fn accumulator_block_count_is_even_by_construction_so_parity_cannot_bind() {
        for cout in [1u32, 8, 31, 32, 33, 64, 96, 136, 256, 353, 768] {
            let shape = Shape::with_precision(
                9,
                7,
                1,
                8,
                cout,
                Precision::Int8Accumulator(Quantization {
                    input_zero_point: 0,
                    output_zero_point: 0,
                    weight_zero_point: 0,
                    ..quantization()
                }),
            );
            let blocks = shape.output_blocks_per_pixel();
            assert_eq!(
                blocks,
                shape.padded_out_channels() / 4,
                "Cout={cout}: the accumulator cube is C2=4"
            );
            assert!(
                blocks.is_multiple_of(2),
                "Cout={cout}: blocks={blocks} must be even by construction"
            );
            // And the hook is now the identity at every one of them, including
            // the 3x3-output/3x3-kernel case that used to be refused outright.
            assert_eq!(
                shape.parity_padded_shape([1, 1]).unwrap().out_channels,
                cout
            );
        }

        let refused = Shape::with_precision(
            3,
            3,
            1,
            8,
            32,
            Precision::Int8Accumulator(Quantization {
                input_zero_point: 0,
                output_zero_point: 0,
                weight_zero_point: 0,
                ..quantization()
            }),
        );
        assert!(
            refused.parity_padded_shape([3, 3]).is_ok(),
            "the 3x3-output/3x3-kernel refusal is gone with the serial writer"
        );
    }

    /// The saturating coefficient working set is refused, not silently
    /// clamped into a one-data-bank split.
    ///
    /// A 5x5 kernel at `Cin` 512 is constructible today and wants 25 banks,
    /// far past the eleven grantable. Before the guard it came back 1/11 --
    /// about one input row per tile -- from the same clamp interaction the
    /// expanded corpus exposes at `Cin` >= 576. (7x7 does not reach the
    /// allocator: above coefficient demand seven it takes its own
    /// capture-derived 8/4 schedule.)
    #[test]
    #[should_panic(expected = "not capture-backed")]
    fn saturating_coefficient_working_set_is_refused() {
        let shape = Shape::with_precision(32, 32, 1, 512, 64, Precision::Fp16);
        let _ = ConvPlan::new(shape, [5, 5]);
    }

    /// A 5x5 kernel at `Cin` 192 plans 2/10, the split the vendor uses.
    ///
    /// This is the shape that catches a `Cin`-curve fit being extrapolated
    /// across kernel sizes: a two-pass rule derived from k=3 (which fits all
    /// 13 k=3 points from `Cin` 384 to 768) plans 6/6 here. Nothing else in
    /// the suite reaches it -- the base corpus has k=5 only at `Cin` 3, where
    /// the coefficient preference is 1.
    #[test]
    fn large_kernel_high_channel_split_matches_vendor() {
        let shape =
            Shape::with_precision(28, 28, 1, 192, 256, Precision::Fp16).with_padding([2, 2]);
        let plan = ConvPlan::new(shape, [5, 5]);
        assert_eq!((plan.data_banks(), plan.weight_banks()), (2, 10));
    }

    /// The guard does not disturb the range that is capture-backed: a 3x3
    /// kernel at the same `Cin` wants nine banks and still plans 3/9, the
    /// split the expanded corpus records.
    #[test]
    fn largest_capture_backed_working_set_still_plans() {
        let shape =
            Shape::with_precision(28, 28, 1, 512, 256, Precision::Fp16).with_padding([1, 1]);
        let plan = ConvPlan::new(shape, [3, 3]);
        assert_eq!((plan.data_banks(), plan.weight_banks()), (3, 9));
    }

    #[test]
    fn weight_banks_floor_matches_all_five_hardware_points() {
        // Cin=Cout=N, 3x3, 30x30, swept explicitly via ConvPlan::with_cbuf_banks
        // in iree-rocket-hal/tests/conv_cbuf_split_sweep_hw.rs's
        // weight_bank_floor_probe* family -- each is the exact hardware
        // boundary (floor-1 fails 0/5, floor passes 5/5). Not a flat
        // constant: it rises +1 per 64 Cin from 256, then plateaus at 5
        // from 384 on -- a first guess of `Cin/128 + 1` fit 256 and 512
        // but predicted 4, not 5, at 384.
        for (cin, floor) in [(256u32, 3u32), (320, 4), (384, 5), (448, 5), (512, 5)] {
            assert_eq!(
                weight_banks_floor(cin),
                floor,
                "Cin={cin} weight_banks floor"
            );
        }

        // The real VGG-19 shapes this was chasing: features.19 (Cin=256,
        // Cout=512) and features.21 (Cin=512, Cout=512), both hardware-proven
        // all-zero at the pre-fix automatic 11/1, and both hardware-confirmed
        // fixed post-fix -- features.19 at floor=3
        // (fixed_formula_resolves_features_19_and_21, 5/5) and features.21 at
        // floor=5, *not* 3 (weight_bank_floor_probe_at_cin_512 found
        // weight_banks=3 and 4 both still 0/5 at Cin=512; only 5 and up
        // pass). Cout does not move the floor -- only Cin does, matching the
        // asymmetry `DESIGN_NOTES.md`'s vendor-formula cross product found.
        assert_eq!(weight_banks_floor(256), 3);
        assert_eq!(weight_banks_floor(512), 5);

        // This measured floor is a safety lower bound, not the vendor's
        // preferred allocation. The expanded corpus independently shows a
        // larger streamed working-set grant at K3; keep the two rules
        // explicit so a policy change cannot masquerade as a new hardware
        // minimum.
        assert_eq!(streamed_weight_bank_preference(256, [3, 3], 16), 5);
        assert_eq!(streamed_weight_bank_preference(512, [3, 3], 16), 9);
        // int8 and int4 share the fp16 calibration; only tf32's four bytes
        // move the working set.
        assert_eq!(streamed_weight_bank_preference(512, [3, 3], 8), 9);
        assert_eq!(streamed_weight_bank_preference(512, [3, 3], 4), 9);
        assert_eq!(streamed_weight_bank_preference(576, [3, 3], 16), 6);
        assert_eq!(streamed_weight_bank_preference(576, [3, 3], 32), 11);

        // features.0's shape: weight_banks=1 here is not a starved
        // footprint, it is the footprint's *entire* real demand (1,728
        // bytes fits in a fraction of one bank) -- hardware-confirmed
        // correct at 11/1 by the original five-shape sweep, and the floor
        // must leave it alone rather than take banks a fully-resident
        // footprint never needed. Below Cin=256 the floor itself is
        // unvalidated (see weight_banks_floor's doc comment); this only
        // confirms the starved branch doesn't fire here at all.
        let features_0 = Shape::with_out_channels(226, 226, 1, 3, 64).with_padding([0, 0]);
        assert_eq!(features_0.data_banks([3, 3]), 11);
        assert_eq!(features_0.weight_banks([3, 3]), 1);

        // Same for the 256x32/Cin32 pair above: weight demand of 1 and 2
        // banks respectively are each already fully satisfied by what they
        // are granted, not clamped down from something larger.
        let narrow = Shape::with_out_channels(256, 32, 1, 32, 16);
        assert_eq!(narrow.weight_banks([3, 3]), 1);
        let wide = Shape::with_out_channels(256, 32, 1, 32, 64);
        assert_eq!(wide.weight_banks([3, 3]), 2);
    }

    #[test]
    #[should_panic(expected = "output channels must be")]
    fn rejects_output_channels_beyond_the_validated_range() {
        let _ = Shape::with_out_channels(32, 32, 1, 3, MAX_OUTPUT_CHANNELS + 1);
    }

    /// Depthwise stopped sharing the dense ceiling on 2026-09-06.
    ///
    /// Before that a depthwise `Shape` was bounded by `MAX_INPUT_CHANNELS`
    /// alone, so raising the dense constant to 3584 would have doubled the
    /// depthwise range without a single depthwise measurement. This is the
    /// test that says the two moved apart on purpose: the same channel count
    /// builds a dense shape and is refused a depthwise one.
    #[test]
    fn depthwise_stops_below_the_dense_channel_ceiling() {
        let channels = MAX_DEPTHWISE_CHANNELS + 64;
        assert!(
            channels <= MAX_INPUT_CHANNELS,
            "the dense ceiling admits it"
        );
        let _dense = Shape::with_out_channels(32, 32, 1, channels, channels);

        let shape = Shape::with_out_channels(32, 32, 1, channels, channels);
        let refused = std::panic::catch_unwind(move || shape.with_depthwise());
        assert!(
            refused.is_err(),
            "depthwise must refuse {channels} channels"
        );

        // And the ceiling itself still builds, so this is a boundary rather
        // than a blanket refusal.
        let _at_the_ceiling =
            Shape::with_out_channels(32, 32, 1, MAX_DEPTHWISE_CHANNELS, MAX_DEPTHWISE_CHANNELS)
                .with_depthwise();
    }

    #[test]
    fn dense_feature_offset_safe_requires_full_alignment() {
        // The older width/channel sweeps used uniform values and therefore
        // could only observe whole-pixel leading-column loss. The exact
        // features.0 regression uses non-uniform data and shows offset 4 is
        // already unsafe at Cin=3. Conservatively require offset 0 at every
        // fp16 dense width until a data-rich sweep proves otherwise.
        let cin1 = Shape::with_out_channels(225, 60, 1, 1, 8).with_padding([0, 0]); // stride mod16=2
        for in_first in 0..8 {
            assert_eq!(
                cin1.dense_feature_offset_safe(in_first),
                in_first == 0,
                "Cin=1 in_first={in_first}"
            );
        }
        let cin2 = Shape::with_out_channels(225, 60, 1, 2, 8).with_padding([0, 0]); // stride mod16=4
        for in_first in 0..8 {
            assert_eq!(
                cin2.dense_feature_offset_safe(in_first),
                matches!(in_first, 0 | 4),
                "Cin=2 in_first={in_first}"
            );
        }
        let cin3 = Shape::with_out_channels(227, 60, 1, 3, 256).with_padding([0, 0]); // stride mod16=2
        for in_first in 0..8 {
            assert_eq!(
                cin3.dense_feature_offset_safe(in_first),
                in_first == 0,
                "Cin=3 in_first={in_first}"
            );
        }
        let cin4 = Shape::with_out_channels(225, 60, 1, 4, 256).with_padding([0, 0]); // stride mod16=8
        for in_first in 0..8 {
            assert_eq!(
                cin4.dense_feature_offset_safe(in_first),
                in_first % 2 == 0,
                "Cin=4 in_first={in_first}"
            );
        }
        // Cout plays no role -- the address formula never references it.
        let small_cout = Shape::with_out_channels(227, 60, 1, 3, 8).with_padding([0, 0]);
        assert!(small_cout.dense_feature_offset_safe(0));
        assert!(!small_cout.dense_feature_offset_safe(2));

        // The affine-int8 oracle measured the same failure at offset 2 for
        // this compact VGG input pitch: 226 * 3 = 678 bytes, or 6 mod 16.
        let int8 = Shape::with_precision(226, 226, 1, 3, 64, captured_int8()).with_padding([0, 0]);
        assert_eq!(int8.input_row_stride(), 678);
        for in_first in 0..16 {
            assert_eq!(
                int8.dense_feature_offset_safe(in_first),
                in_first % 8 == 0,
                "int8 Cin=3 in_first={in_first}",
            );
        }

        // Surfaces (Cin > 4) are a different addressing path this defect
        // has not been shown to reach; always reported safe.
        let surfaces = Shape::with_out_channels(227, 60, 1, 5, 8);
        for in_first in 0..8 {
            assert!(surfaces.dense_feature_offset_safe(in_first));
        }
    }

    #[test]
    fn conv_plan_moves_run1_tile_boundary_off_the_hardware_confirmed_break() {
        // rocket_conv_harness.py's run1 (iree-rocket-design-spike): Cin=3
        // dense, Cout=256, 3x3, 228x228 physically-padded input. Before this
        // fix, ConvPlan::new's automatic split put tile 5's out_first/
        // in_first at 189 (odd -- an 8-byte-misaligned feature base at this
        // shape's stride), hardware-confirmed to corrupt one leading pixel
        // of every one of that tile's 37 output rows
        // (conv_dense_shared_buffer_dispatch_hw.rs).
        let kernels = [3, 3];
        let shape = Shape::with_out_channels(228, 228, 1, 3, 256).with_padding([0, 0]);
        let plan = ConvPlan::new(shape, kernels);
        assert_eq!(plan.data_banks(), 10);
        assert_eq!(plan.weight_banks(), 2);
        assert_eq!(plan.tiles().len(), 6);

        let mut covered = 0u32;
        for tile in plan.tiles() {
            assert_eq!(
                tile.rows.out_first, covered,
                "row coverage must stay contiguous with no gap or overlap"
            );
            covered += tile.rows.out_rows;
            assert!(
                shape.dense_feature_offset_safe(tile.rows.in_first),
                "tile at in_first={} is not alignment-safe",
                tile.rows.in_first
            );
        }
        assert_eq!(covered, shape.output_height(kernels));

        // Greedy capacity filling makes every full tile 42 output rows and
        // leaves a short 16-row tail. All boundaries are even and therefore
        // safe at this row pitch; in particular the old 189 boundary is no
        // longer present.
        assert_eq!(plan.tiles()[4].rows.out_rows, 42);
        assert_eq!(plan.tiles()[5].rows.out_first, 210);
        assert_eq!(plan.tiles()[5].rows.in_first, 210);
        assert_eq!(plan.tiles()[5].rows.out_rows, 16);
    }

    #[test]
    fn conv_plan_aligns_the_compact_int8_vgg_boundary() {
        let kernels = [3, 3];
        let shape = Shape::with_precision(226, 226, 1, 3, 64, captured_int8()).with_padding([0, 0]);
        let plan = ConvPlan::new(shape, kernels);

        assert_eq!((plan.data_banks(), plan.weight_banks()), (11, 1));
        assert_eq!(plan.tiles().len(), 3);
        assert_eq!(plan.tiles()[0].rows.in_first, 0);
        assert_ne!(plan.tiles()[1].rows.in_first, 91);
        let mut covered = 0;
        for tile in plan.tiles() {
            assert_eq!(tile.rows.out_first, covered);
            covered += tile.rows.out_rows;
            assert!(
                shape.dense_feature_offset_safe(tile.rows.in_first),
                "unsafe int8 tile at in_first={} offset={}",
                tile.rows.in_first,
                tile.rows.input_offset(shape) % FEATURE_ATOM_BYTES,
            );
        }
        assert_eq!(covered, shape.output_height(kernels));
    }

    #[test]
    fn dense_row_tiling_stays_gap_free_and_alignment_safe_across_a_shape_sweep() {
        // Host-side sweep, no hardware needed: dense_feature_offset_safe is
        // a pure function of the plan ConvPlan::new already produces, so
        // this checks the fix holds broadly rather than just at run1's one
        // shape. 4 Cin values x 4 Cout values x 7 widths x 4 heights = 448
        // shapes. Width 226 is the newly-found VGG-19 features.0 regression
        // point and must remain in the ordinary suite.
        let kernels = [3, 3];
        for cin in [1u32, 2, 3, 4] {
            for cout in [1u32, 8, 64, 256] {
                for width in [30u32, 61, 97, 225, 226, 227, 300] {
                    for height in [30u32, 61, 226, 300] {
                        let shape = Shape::with_out_channels(width, height, 1, cin, cout)
                            .with_padding([0, 0]);
                        let plan = ConvPlan::new(shape, kernels);
                        let mut covered = 0u32;
                        for tile in plan.tiles() {
                            assert_eq!(
                                tile.rows.out_first, covered,
                                "cin={cin} cout={cout} w={width} h={height}: gap/overlap"
                            );
                            covered += tile.rows.out_rows;
                            assert!(
                                shape.dense_feature_offset_safe(tile.rows.in_first),
                                "cin={cin} cout={cout} w={width} h={height}: unsafe tile at \
                                 in_first={}",
                                tile.rows.in_first
                            );
                        }
                        assert_eq!(
                            covered,
                            shape.output_height(kernels),
                            "cin={cin} cout={cout} w={width} h={height}: coverage mismatch"
                        );
                    }
                }
            }
        }
    }

    /// The quantization of `conv-w32-h32-k3-s1-i8`, read off the capture.
    fn captured_int8() -> Precision {
        Precision::Int8(Quantization {
            input_zero_point: 0,
            output_zero_point: -3,
            weight_zero_point: 0,
            input_scale: 1.0,
            weights_scale: 1.0,
            multiplier: Multiplier {
                scale: 19636,
                shift: 24,
            },
        })
    }

    #[test]
    fn int8_rounds_the_programmed_kernel_count_up_to_even() {
        // Vendor captures at 32x32, Cin 3, 3x3: the programmed kernel count
        // and the coefficient footprint, against the true Cout.
        //
        // `conv-w32-h32-k3-s1-ci3-co{1,2,3,4,5,12}-i8` and their fp16 twins.
        // The int8 corpus had nothing below Cout 8 until hardware failed at
        // Cout 1, which is why this went unnoticed: every int8 Cout ever
        // captured was already even.
        const BYTES_PER_KERNEL: u32 = 144; // 3 * 3 * pad(Cin 3) * 1 byte

        for (cout, kernels) in [(1u32, 2u32), (2, 2), (3, 4), (4, 4), (5, 6), (12, 12)] {
            let shape = Shape::with_precision(32, 32, 1, 3, cout, Precision::Int8(quantization()));
            assert_eq!(shape.programmed_kernels(), kernels, "int8 Cout {cout}");
            assert_eq!(
                shape.weight_bytes([3, 3]),
                kernels * BYTES_PER_KERNEL,
                "int8 weight_bytes at Cout {cout}"
            );
        }

        // fp16 programs the true count at every value, odd ones included.
        for cout in [1u32, 2, 6, 9, 14] {
            let shape = Shape::with_out_channels(32, 32, 1, 3, cout);
            assert_eq!(shape.programmed_kernels(), cout, "fp16 Cout {cout}");
            // 16 padded input channels at fp16, two bytes each.
            assert_eq!(shape.weight_bytes([3, 3]), cout * 9 * 8 * 2);
        }
    }

    #[test]
    fn int8_channel_padding_is_a_rule_not_a_table() {
        // An int8 atom carries 16 channels, so both paddings double their
        // granule. Measured at 15 Cin values and 10 Cout values with no
        // deviation -- including three atoms (33..48) and seven atoms (112),
        // where the fp16 table needs exceptions. Neither recurs.
        for (in_channels, padded) in [
            (3u32, 16u32),
            (4, 16),
            (16, 16),
            (17, 32),
            (24, 32),
            (32, 32),
            (33, 48),
            (40, 48),
            (48, 48),
            (64, 64),
            (80, 80),
            (112, 112),
            (128, 128),
        ] {
            let shape =
                Shape::with_precision(32, 32, 1, in_channels, 8, Precision::Int8(quantization()));
            assert_eq!(
                shape.padded_channels(),
                padded,
                "int8 datain_channel at Cin {in_channels}"
            );
            // Unlike fp16, the two padded counts never disagree.
            assert_eq!(shape.weight_channels(), padded);
        }

        for (out_channels, padded) in [
            (8u32, 32u32),
            (16, 32),
            (20, 32),
            (32, 32),
            (40, 64),
            (48, 64),
            (64, 64),
            (96, 96),
            (112, 128),
        ] {
            let shape =
                Shape::with_precision(32, 32, 1, 3, out_channels, Precision::Int8(quantization()));
            assert_eq!(
                shape.padded_out_channels(),
                padded,
                "int8 padded Cout at {out_channels}"
            );
        }
    }

    fn quantization() -> Quantization {
        match captured_int8() {
            Precision::Int8(quantization) => quantization,
            _ => unreachable!(),
        }
    }

    #[test]
    fn dense_layout_boundary_is_channels_not_bytes() {
        // The boundary is four channels in *both* precisions. Written as a
        // byte-width test -- "narrower than half a feature atom" -- it comes
        // out right at fp16 and wrong at int8, where it would allow eight.
        // The captures put Cin 4 on the ARGB path and Cin 8 on surfaces at
        // both precisions.
        for precision in [Precision::Fp16, Precision::Int8(quantization())] {
            for (in_channels, layout) in [
                (1u32, FeatureLayout::Dense),
                (4, FeatureLayout::Dense),
                (5, FeatureLayout::Surfaces),
                (8, FeatureLayout::Surfaces),
            ] {
                let shape = Shape::with_precision(32, 32, 1, in_channels, 8, precision);
                assert_eq!(
                    shape.layout(),
                    layout,
                    "layout at Cin {in_channels}, {precision:?}"
                );
            }
        }
    }

    #[test]
    fn int8_registers_match_the_captures() {
        // conv-w32-h32-k3-s1-i8, single-core plan. Every field this builder
        // writes matches the capture; the values below are the ones that
        // move with precision.
        let shape = Shape::with_precision(32, 32, 1, 3, 8, captured_int8());
        let program = conv_2d_tile(shape, [3, 3], &Tile::whole(shape, [3, 3]));

        // Coefficients are one byte per element, so the footprint halves
        // even though the padded channel count doubles.
        assert_eq!(value_of::<CnaWeightSize0>(&program), 16 * 9 * 8);
        assert_eq!(value_of::<CnaWeightSize1>(&program), 16 * 9);

        // Padding contributes the input zero point, not zero.
        assert_eq!(value_of::<CnaPadCon1>(&program), 0);
        let offset = Shape::with_precision(
            32,
            32,
            1,
            3,
            8,
            Precision::Int8(Quantization {
                input_zero_point: -1,
                ..quantization()
            }),
        );
        let offset_program = conv_2d_tile(offset, [3, 3], &Tile::whole(offset, [3, 3]));
        assert_eq!(value_of::<CnaPadCon1>(&offset_program), u32::MAX);

        // Requantization: the multiplier, its shift, and the output zero
        // point, none of which the fp16 path programs at all.
        assert_eq!(value_of::<DpuOutCvtScale>(&program), 19636);
        assert_eq!(value_of::<DpuOutCvtShift>(&program), 24);
        assert_eq!(value_of::<DpuOutCvtOffset>(&program), (-3i32) as u32);
    }

    #[test]
    fn int8_accumulator_output_uses_the_hardware_validated_bypasses() {
        let quantization = Quantization {
            input_zero_point: 0,
            output_zero_point: 0,
            weight_zero_point: 0,
            ..quantization()
        };
        let shape = Shape::with_precision(4, 4, 1, 1, 8, Precision::Int8Accumulator(quantization));
        let program = conv_2d_tile(shape, [1, 1], &Tile::whole(shape, [1, 1]));

        let data_format = value_of::<DpuDataFormat>(&program);
        assert_eq!(data_format & 0b111, 0, "DPU processing stays int8");
        assert_eq!((data_format >> 26) & 0b111, 0, "DPU input stays int8");
        assert_eq!((data_format >> 29) & 0b111, 4, "DPU output is int32");
        assert_eq!(
            (data_format >> 3) & 1,
            0,
            "dense accumulator output uses the one-surface writer, not the serial one"
        );
        assert_eq!(
            (value_of::<CnaDataSize3>(&program) >> 22) & 0b11,
            0,
            "CNA surface serial mode must remain disabled"
        );
        assert_eq!(value_of::<DpuSurfaceAdd>(&program), (4 * 4 * 8) << 4);
        assert_eq!(value_of::<DpuBsCfg>(&program) & 1, 1, "BS is bypassed");
        assert_eq!(
            (value_of::<DpuBsOwCfg>(&program) >> 1) & 1,
            1,
            "CPEND is bypassed"
        );
        assert_eq!(value_of::<DpuOutCvtOffset>(&program), 0);
        assert_eq!(value_of::<DpuOutCvtScale>(&program), 1);
        assert_eq!(value_of::<DpuOutCvtShift>(&program), 0);

        assert_eq!(shape.precision.element_bytes(), 1);
        assert_eq!(shape.precision.output_element_bytes(), 4);
        assert_eq!(shape.output_channel_block_bytes(), FEATURE_ATOM_BYTES);
        let requantized = Shape::with_precision(4, 4, 1, 1, 8, Precision::Int8(quantization));
        assert_eq!(
            shape.output_scratch_bytes([1, 1]),
            4 * requantized.output_scratch_bytes([1, 1])
        );
    }

    #[test]
    fn int8_accumulator_uses_the_c2_4_output_cube_without_forced_tiling() {
        let shape = Shape::with_precision(
            32,
            32,
            1,
            64,
            128,
            Precision::Int8Accumulator(Quantization {
                input_zero_point: 0,
                output_zero_point: 0,
                weight_zero_point: 0,
                ..quantization()
            }),
        );
        let plan = ConvPlan::new(shape, [1, 1]);

        assert_eq!(plan.output_column_widths(), &[32]);
        assert_eq!(plan.tiles().len(), 1);
        assert_eq!(
            shape.output_channel_block_bytes(),
            FEATURE_ATOM_BYTES,
            "the dense accumulator cube is 16-byte atoms of C2=4 int32 lanes"
        );
        assert_eq!(shape.output_row_stride([1, 1]), 32 * FEATURE_ATOM_BYTES);
    }

    #[test]
    fn staged_accumulator_plan_partitions_scratch_and_programs_local_surfaces() {
        let shape = Shape::with_precision(
            32,
            32,
            1,
            353,
            64,
            Precision::Int8Accumulator(Quantization {
                input_zero_point: 0,
                output_zero_point: 0,
                weight_zero_point: 0,
                ..quantization()
            }),
        );
        let plan = ConvPlan::new(shape, [1, 1]);
        assert_eq!(plan.tiles().len(), 2, "this is the first failing Cin plan");

        let staged = plan.programs_with_staged_accumulator_output(RELOCATION);
        assert_eq!(staged.programs.len(), plan.tiles().len());
        assert_eq!(staged.tiles.len(), plan.tiles().len());
        assert_eq!(staged.scratch_bytes, shape.output_scratch_bytes([1, 1]));

        let mut next_offset = 0;
        for (index, ((tile, output), program)) in plan
            .tiles()
            .iter()
            .zip(&staged.tiles)
            .zip(&staged.programs)
            .enumerate()
        {
            let tile_pixels = tile.rows.out_rows as usize * tile.columns.out_cols as usize;
            assert_eq!(output.scratch_offset, next_offset, "tile {index}");
            assert_eq!(output.scratch_bytes, tile_pixels * 2 * 128, "tile {index}");
            assert_eq!(output.output_row, tile.rows.out_first as usize);
            assert_eq!(output.output_column, tile.columns.out_first as usize);
            assert_eq!(output.output_rows, tile.rows.out_rows as usize);
            assert_eq!(output.output_columns, tile.columns.out_cols as usize);
            assert_eq!(
                value_of::<DpuDstBaseAddr>(program),
                RELOCATION.output + output.scratch_offset as u32,
                "tile {index} destination"
            );
            assert_eq!(
                value_of::<DpuDstSurfStride>(program),
                tile_pixels as u32 * FEATURE_ATOM_BYTES,
                "tile {index} surface stride"
            );
            assert_eq!(
                value_of::<DpuDataCubeNotchAddr>(program),
                0,
                "tile {index} notch"
            );
            next_offset += output.scratch_bytes;
        }
        assert_eq!(next_offset, staged.scratch_bytes);
    }

    #[test]
    fn contiguous_accumulator_column_tile_drops_shared_image_notch() {
        let shape = Shape::with_precision(
            32,
            32,
            1,
            64,
            64,
            Precision::Int8Accumulator(Quantization {
                input_zero_point: 0,
                output_zero_point: 0,
                weight_zero_point: 0,
                ..quantization()
            }),
        );
        let tile = Tile2D {
            rows: Tile::whole(shape, [1, 1]),
            columns: ColumnTile::from_output_range(shape, [1, 1], 4, 12),
        };
        let data_banks = shape.data_banks([1, 1]);
        let weight_banks = shape.weight_banks([1, 1]);
        let shared = conv_2d_tile_program(
            shape,
            [1, 1],
            &tile,
            feature_grains([1, 1], &tile.rows),
            data_banks,
            weight_banks,
            OutputPlacement::SharedImage,
        );
        let contiguous = conv_2d_tile_program(
            shape,
            [1, 1],
            &tile,
            feature_grains([1, 1], &tile.rows),
            data_banks,
            weight_banks,
            OutputPlacement::ContiguousTile,
        );

        assert_eq!(value_of::<DpuDstBaseAddr>(&shared), 4 * 16);
        assert_eq!(value_of::<DpuDstSurfStride>(&shared), 32 * 32 * 16);
        assert_eq!(value_of::<DpuDataCubeNotchAddr>(&shared), 0x14_0014);
        assert_eq!(value_of::<DpuDstBaseAddr>(&contiguous), 0);
        assert_eq!(value_of::<DpuDstSurfStride>(&contiguous), 12 * 32 * 16);
        assert_eq!(value_of::<DpuDataCubeNotchAddr>(&contiguous), 0);
    }

    #[test]
    #[should_panic(expected = "currently requires zero input, weight, and output zero-points")]
    fn int8_accumulator_output_rejects_unvalidated_affine_zero_points() {
        let mut quantization = quantization();
        quantization.input_zero_point = 1;
        let _ = Shape::with_precision(4, 4, 1, 1, 8, Precision::Int8Accumulator(quantization));
    }

    #[test]
    fn fp16_and_int8_differ_only_where_the_corpus_says_they_do() {
        // A paired diff, the same comparison the corpus was built to allow.
        // The fp16 side is already hardware-validated, so this pins the int8
        // side against it rather than against a second unknown.
        let fp16 = Shape::new(32, 32);
        let int8 = Shape::with_precision(32, 32, 1, 3, 8, captured_int8());
        let a = conv_2d_tile(fp16, [3, 3], &Tile::whole(fp16, [3, 3]));
        let b = conv_2d_tile(int8, [3, 3], &Tile::whole(int8, [3, 3]));
        assert_eq!(a.len(), b.len(), "precision must not change program length");

        let differing = a
            .iter()
            .zip(&b)
            .filter(|(left, right)| left.0 != right.0)
            .count();
        // Seventeen distinct registers, and `CNA_CONV_CON1` is written
        // twice per program, so eighteen words. That is fewer than the 33
        // fields the sweep reports across all geometries for two reasons:
        // several fields share a register, and several move only where the
        // channel padding or the bank split reacts, neither of which does
        // at Cin 3 Cout 8.
        assert_eq!(
            differing, 18,
            "unexpected number of registers differing between precisions"
        );
    }

    #[test]
    fn requantization_multiplier_normalizes_its_mantissa() {
        // Every OUT_CVT_SCALE in the corpus lands in [2^14, 2^15), with the
        // shift chosen to put it there. These are real (scale, shift) pairs
        // from int8 captures; re-encoding their ratio must reproduce them.
        for (scale, shift) in [
            (19636u32, 24u32),
            (27245, 23),
            (29533, 26),
            (32573, 24),
            (16625, 24),
            (23916, 25),
        ] {
            let encoded = Multiplier::from_ratio(Multiplier { scale, shift }.ratio());
            assert_eq!(
                (encoded.scale, encoded.shift),
                (scale, shift),
                "round trip of {scale}/2^{shift}"
            );
        }

        // The normalization itself, over a wide range of multipliers.
        for exponent in -30i32..4 {
            for step in 1..8 {
                let ratio = 2f64.powi(exponent) * (1.0 + f64::from(step) / 8.0);
                let encoded = Multiplier::from_ratio(ratio);
                assert!(
                    (MANTISSA_FLOOR..2 * MANTISSA_FLOOR).contains(&encoded.scale),
                    "mantissa {} out of range for ratio {ratio}",
                    encoded.scale
                );
                // Within half a mantissa step of the real value.
                let error = (encoded.ratio() - ratio).abs() / ratio;
                assert!(error < 1.0 / f64::from(MANTISSA_FLOOR), "ratio {ratio}");
            }
        }
    }

    #[test]
    fn unit_bs_multiplier_cancels_out_of_the_requantisation() {
        // The composite gain is `(bs_mul >> 7) * (scale / 2^shift)`, so a
        // unit plane entry has to be divided back out. Measured on hardware:
        // at cvt_shift 14 a BS multiplier of 128 gives unit gain, and the
        // output doubles with each doubling of it from there.
        let bs_gain = f64::from(BS_UNIT_MULTIPLIER >> BS_MULTIPLIER_SHIFT);
        assert_eq!(bs_gain, 128.0);
        for exponent in 0..8u32 {
            let wanted = 1.0 / f64::from(1u32 << exponent);
            let multiplier = Multiplier::for_unit_bs(wanted);
            let composite = bs_gain * multiplier.ratio();
            assert!(
                (composite - wanted).abs() < wanted / 1024.0,
                "composite gain {composite} for a requested {wanted}"
            );
        }
        // The probe's own crossing point, restated: unit total gain leaves
        // the per-tensor stage at 2^-7 of unity, which normalises to a
        // mantissa of 2^14 at shift 21.
        let unit = Multiplier::for_unit_bs(1.0);
        assert_eq!((unit.scale, unit.shift), (1 << 14, 21));
    }

    #[test]
    fn int8_bias_packing_normalizes_and_pads_bs_entries() {
        let logical = [200i32, -100i32]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        let mut packed = vec![0xa5; bs_buffer_bytes(8)];
        let written = pack_int8_bias_to_bs(&logical, 2, 8, 0.5, 0.25, 7, &mut packed).unwrap();
        assert_eq!(written, bs_buffer_bytes(8));
        assert_eq!(i32::from_le_bytes(packed[0..4].try_into().unwrap()), 1600);
        assert_eq!(i32::from_le_bytes(packed[4..8].try_into().unwrap()), -800);
        assert_eq!(i16::from_le_bytes(packed[32..34].try_into().unwrap()), -7);
        assert_eq!(i16::from_le_bytes(packed[34..36].try_into().unwrap()), -7);
        assert!(packed[8..32].iter().all(|&byte| byte == 0));
    }

    #[test]
    fn bs_buffer_matches_the_converted_models() {
        // Byte-for-byte against `co16`, a Cout=16 model with per-channel
        // weight magnitudes 0.01*(c+1) and biases 0.05*(c+1). Two blocks,
        // and the second block is what shows the layout repeats rather than
        // running as one flat plane.
        let entries: Vec<BsEntry> = (0..16)
            .map(|c| BsEntry {
                bias: 162675,
                constant: BS_CONSTANT,
                multiplier: 1024 * (c + 1),
            })
            .collect();
        let mut buffer = vec![0u8; bs_buffer_bytes(16)];
        assert_eq!(buffer.len(), 128);
        write_bs_buffer(&mut buffer, &entries);

        // First block: biases at 0, the constant plane at 32, multipliers
        // at 48.
        assert_eq!(&buffer[0..4], &162675i32.to_le_bytes());
        assert_eq!(&buffer[32..34], &128i16.to_le_bytes());
        assert_eq!(&buffer[48..50], &1024i16.to_le_bytes());
        assert_eq!(&buffer[62..64], &8192i16.to_le_bytes());
        // Second block repeats the same three planes for channels 8..15.
        assert_eq!(&buffer[64..68], &162675i32.to_le_bytes());
        assert_eq!(&buffer[96..98], &128i16.to_le_bytes());
        assert_eq!(&buffer[112..114], &9216i16.to_le_bytes());
        assert_eq!(&buffer[126..128], &16384i16.to_le_bytes());

        // A partial block is padded, not packed: Cout 4 still occupies 64
        // bytes with the unused lanes left zero.
        let mut partial = vec![0xffu8; bs_buffer_bytes(4)];
        assert_eq!(partial.len(), 64);
        write_bs_buffer(
            &mut partial,
            &[
                BsEntry {
                    bias: 162675,
                    constant: BS_CONSTANT,
                    multiplier: 4096,
                },
                BsEntry {
                    bias: 162675,
                    constant: BS_CONSTANT,
                    multiplier: 8192,
                },
                BsEntry {
                    bias: 162675,
                    constant: BS_CONSTANT,
                    multiplier: 12288,
                },
                BsEntry {
                    bias: 162675,
                    constant: BS_CONSTANT,
                    multiplier: 16384,
                },
            ],
        );
        assert_eq!(
            &partial[48..56],
            &[0x00, 0x10, 0x00, 0x20, 0x00, 0x30, 0x00, 0x40]
        );
        assert!(partial[16..32].iter().all(|&b| b == 0), "unused bias lanes");
        assert!(partial[56..64].iter().all(|&b| b == 0), "unused mul lanes");

        // The default entry is what a uniform-scale, zero-bias convolution
        // needs, and it is not a zeroed buffer.
        assert_eq!(BsEntry::default().multiplier, BS_UNIT_MULTIPLIER);
        assert_ne!(BsEntry::default().multiplier, 0);
    }

    #[test]
    fn tile_capacity_charges_surfaces_per_atom() {
        // A surface pixel costs `weight_atoms` times what a dense one does,
        // so the same bank allocation carries proportionally fewer rows.
        // Charging one atom per pixel here let a 256x32 Cin 32 tile claim 44
        // rows against a real capacity of 22, and the hardware quietly
        // dropped the rows past the end -- the taps below the cut came back
        // as if they were off the bottom of the image.
        let narrow = Shape::with_out_channels(256, 32, 1, 32, 16);
        assert_eq!(narrow.data_banks([3, 3]), 11);
        assert_eq!(narrow.max_tile_input_rows([3, 3]), 22);
        assert_eq!(narrow.min_tiles([3, 3]), 2);

        // The same shape with a kernel set big enough to take a second bank
        // for coefficients loses two rows of capacity with it. Both numbers
        // are the vendor's own split: 22/12 rows here, 20/14 below.
        let wide = Shape::with_out_channels(256, 32, 1, 32, 64);
        assert_eq!(wide.data_banks([3, 3]), 10);
        assert_eq!(wide.max_tile_input_rows([3, 3]), 20);
        assert_eq!(wide.min_tiles([3, 3]), 2);

        // The dense regime is unchanged: one atom per pixel is correct there,
        // which is why the Cin 3 width sweep never saw this.
        for (width, rows) in [(256u32, 32u32), (512, 22), (768, 14), (1024, 11), (1536, 7)] {
            assert_eq!(
                Shape::new(width, 32).max_tile_input_rows([3, 3]),
                rows,
                "dense capacity at {width} wide"
            );
        }
    }

    #[test]
    fn tiles_of_a_plan_write_disjoint_output_ranges() {
        for tiles in 1..=3 {
            let split = Tile::split(Shape::CAPTURED, [3, 3], tiles);
            let mut covered = vec![0u32; IMAGE_HEIGHT as usize];
            for tile in &split {
                for row in tile.out_first..tile.out_first + tile.out_rows {
                    covered[row as usize] += 1;
                }
            }
            assert!(
                covered.iter().all(|&n| n == 1),
                "{tiles}-tile plan does not partition the output exactly"
            );
        }
    }

    const RELOCATION: Buffers = Buffers {
        input: 0x1_0000,
        weights: 0x2_0000,
        bias: 0x3_0000,
        output: 0x4_0000,
    };

    #[test]
    fn relocation_assigns_addresses_to_a_whole_image_program() {
        let mut commands = conv_2d([3, 3]);
        // A whole-image program starts at both tensors' base, so there is no
        // offset for the relocation to preserve and it reads as assignment.
        assert_eq!(value_of::<CnaFeatureDataAddr>(&commands), 0);
        assert_eq!(value_of::<DpuDstBaseAddr>(&commands), 0);

        relocate(&mut commands, RELOCATION);

        assert_eq!(value_of::<CnaFeatureDataAddr>(&commands), 0x1_0000);
        assert_eq!(value_of::<CnaDcompAddr0>(&commands), 0x2_0000);
        assert_eq!(value_of::<DpuRdmaBsBaseAddr>(&commands), 0x3_0000);
        assert_eq!(value_of::<DpuDstBaseAddr>(&commands), 0x4_0000);
    }

    #[test]
    fn relocation_adds_the_tile_offset_rather_than_overwriting_it() {
        let shape = Shape::CAPTURED;
        let kernels = [3, 3];
        let split = Tile::split(shape, kernels, 2);
        let mut commands = conv_2d_tile(shape, kernels, &split[1]);
        // The second tile of a two-way split starts partway into the feature
        // map and partway into the output -- the captured values the
        // six-group tile test pins independently.
        assert_eq!(value_of::<CnaFeatureDataAddr>(&commands), 0xb40);
        assert_eq!(value_of::<DpuDstBaseAddr>(&commands), 0x2000);

        relocate(&mut commands, RELOCATION);

        assert_eq!(value_of::<CnaFeatureDataAddr>(&commands), 0x1_0000 + 0xb40);
        assert_eq!(value_of::<DpuDstBaseAddr>(&commands), 0x4_0000 + 0x2000);
        // Weights and bias are whole-tensor: every tile reads all of them, so
        // these two carry no offset to add.
        assert_eq!(value_of::<CnaDcompAddr0>(&commands), 0x2_0000);
        assert_eq!(value_of::<DpuRdmaBsBaseAddr>(&commands), 0x3_0000);
    }

    #[test]
    fn exact_output_relocation_preserves_only_the_input_tile_offset() {
        let shape = Shape::CAPTURED;
        let kernels = [3, 3];
        let split = Tile::split(shape, kernels, 2);
        let mut commands = conv_2d_tile(shape, kernels, &split[1]);

        relocate_with_exact_output(&mut commands, RELOCATION);

        assert_eq!(value_of::<CnaFeatureDataAddr>(&commands), 0x1_0000 + 0xb40);
        assert_eq!(value_of::<CnaDcompAddr0>(&commands), 0x2_0000);
        assert_eq!(value_of::<DpuRdmaBsBaseAddr>(&commands), 0x3_0000);
        assert_eq!(value_of::<DpuDstBaseAddr>(&commands), 0x4_0000);
    }

    #[test]
    #[should_panic(expected = "not 16-byte aligned")]
    fn relocation_rejects_a_misaligned_address() {
        let mut commands = conv_2d([1, 1]);
        relocate(
            &mut commands,
            Buffers {
                input: 0x1_0008,
                ..RELOCATION
            },
        );
    }

    #[test]
    fn plan_programs_with_buffers_relocates_every_tile() {
        let shape = Shape::with_out_channels(256, 32, 1, 32, 64);
        let plan = ConvPlan::new(shape, [5, 5]);
        assert_eq!(plan.tiles().len(), 3);

        let bare = plan.programs();
        let bound = plan.programs_with_buffers(RELOCATION);
        assert_eq!(bound.len(), bare.len());

        for (tile, (bare, bound)) in bare.iter().zip(&bound).enumerate() {
            assert_eq!(
                value_of::<CnaFeatureDataAddr>(bound),
                0x1_0000 + value_of::<CnaFeatureDataAddr>(bare),
                "tile {tile} input"
            );
            assert_eq!(
                value_of::<DpuDstBaseAddr>(bound),
                0x4_0000 + value_of::<DpuDstBaseAddr>(bare),
                "tile {tile} output"
            );
            // Relocation touches the four address registers and nothing else.
            let changed = bare
                .iter()
                .zip(bound)
                .filter(|(left, right)| left.0 != right.0)
                .count();
            assert_eq!(changed, 4, "tile {tile} changed words");
        }

        // Distinct tiles really do land at distinct output addresses -- a
        // relocation that overwrote instead of adding would collapse these.
        let outputs: Vec<_> = bound
            .iter()
            .map(|program| value_of::<DpuDstBaseAddr>(program))
            .collect();
        assert_eq!(outputs[0], 0x4_0000);
        assert!(
            outputs[0] < outputs[1] && outputs[1] < outputs[2],
            "tile outputs are not strictly increasing: {outputs:x?}"
        );
    }

    /// Whole-register values read off the fp16 activation sweep, at
    /// `conv-w32-h32-k3-s1-ci32-co32` and its four activated siblings. The
    /// same three `DPU_BN_CFG` values appear in all fourteen fp16
    /// comparison groups, across both feature layouts and both kernels.
    #[test]
    fn activation_programs_the_captured_bn_registers() {
        let base = Shape::with_out_channels(32, 32, 1, 32, 32);
        for (activation, bn_cfg, cmp) in [
            (Activation::None, 0x53, 0),
            (Activation::Relu, 0x12, 0),
            (Activation::clamped_fp16(6.0), 0x92, 0x40C0_0000),
            (Activation::clamped_fp16(2.0), 0x92, 0x4000_0000),
            (Activation::clamped_fp16(1.0), 0x92, 0x3F80_0000),
        ] {
            let program = conv_2d_tile(
                base.with_activation(activation),
                [3, 3],
                &Tile::whole(base, [3, 3]),
            );
            assert_eq!(value_of::<DpuBnCfg>(&program), bn_cfg, "{activation:?}");
            assert_eq!(
                value_of::<DpuBnReluxCmpValue>(&program),
                cmp,
                "{activation:?} cmp"
            );
            // The BN stage needs no operand: these are zero in every
            // capture, activated or not.
            assert_eq!(value_of::<DpuBnAluCfg>(&program), 0);
            assert_eq!(value_of::<DpuBnMulCfg>(&program), 0);
            assert_eq!(value_of::<DpuRdmaBnBaseAddr>(&program), 0);
            // The vendor leaves BS alone; only BN moves. The retired
            // Mesa-derived builder fused activation here instead, which is
            // what this pins against.
            assert_eq!(
                value_of::<DpuBsCfg>(&program),
                0x2_0150,
                "{activation:?} BS"
            );
            assert_eq!(value_of::<DpuBsReluxCmpValue>(&program), 0);
        }
    }

    #[test]
    fn activation_changes_only_the_two_bn_words() {
        let shape = Shape::with_out_channels(32, 32, 1, 32, 32);
        let tile = Tile::whole(shape, [3, 3]);
        let plain = conv_2d_tile(shape, [3, 3], &tile);
        for (activation, expected) in [
            // Relu leaves the cmp value at its unactivated zero.
            (Activation::Relu, 1),
            (Activation::clamped_fp16(6.0), 2),
            (Activation::clamped_int8(6.0, 0.02, 0.003), 2),
        ] {
            let activated = conv_2d_tile(shape.with_activation(activation), [3, 3], &tile);
            let changed = plain
                .iter()
                .zip(&activated)
                .filter(|(left, right)| left.0 != right.0)
                .count();
            assert_eq!(changed, expected, "{activation:?} changed words");
        }
    }

    #[test]
    fn int8_clamp_is_the_ceiling_in_the_post_bs_domain() {
        // The capture-derived accumulator-unit ceiling is multiplied by the
        // effective gain of the default BS plane before BN sees it.
        let (input, weights) = (0.02, 0.003);
        let cmp = |ceiling| match Activation::clamped_int8(ceiling, input, weights) {
            Activation::Clamped { cmp } => cmp,
            other => panic!("expected a clamp, got {other:?}"),
        };
        let bs_gain = u32::from(
            u16::try_from(BS_UNIT_MULTIPLIER >> BS_MULTIPLIER_SHIFT)
                .expect("the unit BS gain must be positive"),
        );
        assert_eq!(
            cmp(1.0),
            (1.0 / (f64::from(input) * f64::from(weights)) * f64::from(bs_gain)).round() as u32
        );
        // Linear in the ceiling, but only to within the rounding -- which is
        // exactly what the captures show: 6 x 86815 is 520890 against a
        // captured 520891, and 2 x 276086 is 552172 against 552171.
        for multiple in [2u32, 6] {
            let scaled = cmp(multiple as f32);
            let exact = multiple * cmp(1.0);
            assert!(
                scaled.abs_diff(exact) <= 2,
                "x{multiple}: {scaled} is not within rounding of {exact}"
            );
        }
        // Not the fp16 encoding, which is what the two precisions differ on.
        assert_ne!(cmp(6.0), 6.0f32.to_bits());
    }

    /// The depthwise channel ladder, both precisions, read off the nine
    /// captures. The retired Mesa-derived channel rule would say 64, 64 and
    /// 128 for the fp16 8, 32 and 96 rows.
    #[test]
    fn depthwise_pads_channels_to_the_captured_granule() {
        for (channels, fp16, int8) in [
            (8u32, 32u32, 64u32),
            (16, 32, 64),
            (32, 32, 64),
            (48, 64, 64),
            (64, 64, 64),
            (96, 96, 128),
            (128, 128, 128),
        ] {
            let shape = Shape::with_out_channels(32, 32, 1, channels, channels).with_depthwise();
            assert_eq!(shape.padded_out_channels(), fp16, "fp16 c{channels}");
            let quantized = Shape::with_precision(32, 32, 1, channels, channels, captured_int8())
                .with_depthwise();
            assert_eq!(quantized.padded_out_channels(), int8, "int8 c{channels}");
        }
    }

    /// `CNA_WEIGHT_SIZE0.weight_bytes` across the same ladder: one filter
    /// per input channel, with the channel count padded to a whole CBUF
    /// atom group. The int8 48-channel row is the one that separates this
    /// from the dense weight padding, which would read 432 rather than 576.
    #[test]
    fn depthwise_weight_bytes_match_the_captures() {
        for (channels, fp16, int8) in [
            (8u32, 144u32, 144u32),
            (16, 288, 144),
            (32, 576, 288),
            (48, 864, 576),
            (64, 1152, 576),
            (96, 1728, 864),
            (128, 2304, 1152),
        ] {
            let shape = Shape::with_out_channels(32, 32, 1, channels, channels).with_depthwise();
            assert_eq!(shape.weight_bytes([3, 3]), fp16, "fp16 c{channels}");
            let quantized = Shape::with_precision(32, 32, 1, channels, channels, captured_int8())
                .with_depthwise();
            assert_eq!(quantized.weight_bytes([3, 3]), int8, "int8 c{channels}");
        }
        // The 5x5 point, which says the rule is not 3x3-specific.
        let five = Shape::with_out_channels(32, 32, 1, 32, 32).with_depthwise();
        assert_eq!(five.weight_bytes([5, 5]), 1600);
        assert_eq!(
            Shape::with_precision(32, 32, 1, 32, 32, captured_int8())
                .with_depthwise()
                .weight_bytes([5, 5]),
            800
        );
    }

    /// The ten fields the depthwise diff found moving, at the geometry the
    /// dense control shares.
    #[test]
    fn depthwise_programs_the_captured_registers() {
        let shape = Shape::with_out_channels(32, 32, 1, 32, 32);
        let tile = Tile::whole(shape, [3, 3]);
        let dense = conv_2d_tile(shape, [3, 3], &tile);
        let depthwise = conv_2d_tile(shape.with_depthwise(), [3, 3], &tile);

        // weight_kernels drops to 1 whatever the channel count.
        assert_eq!(value_of::<CnaWeightSize2>(&dense) & 0x3fff, 32);
        assert_eq!(value_of::<CnaWeightSize2>(&depthwise) & 0x3fff, 1);
        assert_eq!(value_of::<CnaWeightSize0>(&dense), 18432);
        assert_eq!(value_of::<CnaWeightSize0>(&depthwise), 576);
        // SURF_ADD doubles. The field sits in register bits 31:4, so the
        // raw word is sixteen times the logical value the captures report.
        assert_eq!(value_of::<DpuSurfaceAdd>(&dense) >> 4, 2048);
        assert_eq!(value_of::<DpuSurfaceAdd>(&depthwise) >> 4, 4096);
        // DW_EN, and the three conv_mode copies.
        assert_ne!(
            value_of::<CoreMiscCfg>(&dense),
            value_of::<CoreMiscCfg>(&depthwise)
        );
        for (name, dense_word, dw_word) in [
            // Written twice per program, always with the same value.
            (
                "cna",
                first_value_of::<CnaConvCon1>(&dense),
                first_value_of::<CnaConvCon1>(&depthwise),
            ),
            (
                "dpu",
                value_of::<DpuFeatureModeCfg>(&dense),
                value_of::<DpuFeatureModeCfg>(&depthwise),
            ),
            (
                "dpu_rdma",
                value_of::<DpuRdmaFeatureModeCfg>(&dense),
                value_of::<DpuRdmaFeatureModeCfg>(&depthwise),
            ),
        ] {
            assert_ne!(dense_word, dw_word, "{name} conv_mode");
        }
    }

    #[test]
    fn depthwise_at_stride_two_still_doubles_the_surface() {
        // The capture whose output geometry differs, which is what makes
        // the doubling a factor rather than a constant 4096.
        let shape = Shape::with_out_channels(32, 32, 2, 32, 32);
        let tile = Tile::whole(shape, [3, 3]);
        assert_eq!(
            value_of::<DpuSurfaceAdd>(&conv_2d_tile(shape, [3, 3], &tile)) >> 4,
            512
        );
        assert_eq!(
            value_of::<DpuSurfaceAdd>(&conv_2d_tile(shape.with_depthwise(), [3, 3], &tile)) >> 4,
            1024
        );
    }

    #[test]
    #[should_panic(expected = "channel multiplier of one")]
    fn depthwise_refuses_a_channel_multiplier_above_one() {
        let _ = Shape::with_out_channels(32, 32, 1, 32, 64).with_depthwise();
    }
}
