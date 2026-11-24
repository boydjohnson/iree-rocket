//! The DPU's BN per-channel operand plane, fed by NRDMA.
//!
//! The DPU carries two per-channel operand planes between the accumulator
//! and the output converter, each with an add (ALU) and a multiply (MUL)
//! sub-stage:
//!
//! * **BS**, fed by BRDMA from `RDMA_BS_BASE_ADDR`. This crate drives it:
//!   it is where a convolution's bias lives
//!   ([`crate::rocket::tensor_layout::pack_fp16_bias_to_rocket`] at fp16,
//!   [`crate::rocket::conv::write_bs_buffer`]'s bias/constant/multiplier
//!   blocks at int8).
//! * **BN**, fed by NRDMA from `RDMA_BN_BASE_ADDR`. Nothing in this crate
//!   has ever driven it. Every builder here writes `RDMA_NRDMA_CFG` and
//!   `RDMA_BN_BASE_ADDR` as zero, and the two `*_src` fields that would
//!   select NRDMA over a config register (`BN_CFG.bn_alu_src`,
//!   `BN_MUL_CFG.bn_mul_src`) are 0 everywhere.
//!
//! This module is the first thing to drive it, and it exists for
//! [SOFTMAX.md](../../../../SOFTMAX.md)'s probe S1. A per-channel plane is
//! what broadcasts a softmax row maximum back across its row and what
//! applies the row's reciprocal: in the width-mapped layout that document
//! argues for, a row statistic *is* a per-channel value, and at physical
//! height 1 a per-channel plane is constant over the entire remaining
//! axis.
//!
//! **Status, after the first hardware round (RK3588, 2026-11-24,
//! `tests/bn_plane_probe_hw.rs`): the plane works, and both operand
//! encodings are measured.** The ALU operand is a flat IEEE `f32` at
//! `channel * 4` ([`bn_plane_bytes`]); the MUL operand is a flat IEEE
//! `fp16` at `channel * 2` with the shift field at 0
//! ([`bn_mul_plane_bytes`]); each is constant across every spatial
//! position of the cube and indexed by logical channel past the first
//! atom; and the stage order is `x * m + a` -- the addend is not scaled by
//! the multiplier. A plain job submitted straight afterwards still runs
//! correctly, so enabling NRDMA does not reproduce ISSUES.md C8's
//! poisoning the way leaving BRDMA enabled did.
//!
//! One thing is **open**: the layout when *both* sub-stages are engaged at
//! once. Each stream's own stride is known, but with both enabled the MUL
//! stream is fetched two bytes further on than `Mul` alone reads it, so
//! the two are not simply both based at `BN_BASE_ADDR`. SOFTMAX.md's
//! two-pass form engages one sub-stage per task and does not need it.
//!
//! What is not a hypothesis is the task this builder wraps the plane in.
//! It is [`crate::rocket::elementwise::build_unary_regcmd`]'s program with
//! the element-wise stage switched off instead of on, and the two
//! configuration words that produces are the vendor's own: `BS_CFG=0x53`
//! and `EW_CFG=0x383`, the pair that appears on every plain pass-through
//! DPU copy in a complete capture of the vendor's compiled softmax
//! (`rknpu-spelunking/NOTES.md` Follow-up 5 -- those copies feed each
//! stage of its max-reduction tree). `bn_plane_bypass_words_match_the_vendor_passthrough`
//! asserts both.
//!
//! **ISSUES.md C8 is the hazard to respect here.** Leaving a DMA operand
//! fetch enabled with no consumer is what poisoned the NPU core and hung
//! the *following* job until the power domain cycled -- it was BRDMA, the
//! sibling of the DMA this module enables, and `brdma_data_use` left
//! nonzero while the BS plane was bypassed was the single field that
//! accounted for it. So [`build_bn_plane_regcmd`] derives
//! `nrdma_data_use` from the same [`BnPlaneOp`] that engages the BN
//! sub-stages, rather than taking it from the caller: the fetch and its
//! consumer cannot disagree.

use crate::rocket::{
    builders::{Bits, RegCmd, Register, dpu::*, dpu_rdma::*},
    regcmd::{KICK_DPU, KICK_DPU_RDMA, push_kick, zero},
};

/// Physical width of one feature atom, and the channel granularity the DPU
/// task header is programmed at.
const FEATURE_ATOMIC_SIZE: u32 = 16;

/// Which BN sub-stages read their operand from NRDMA.
///
/// Each variant fixes both the `BN_CFG`/`BN_MUL_CFG` bypass and `*_src`
/// bits and the `RDMA_NRDMA_CFG.nrdma_data_use` bitmask, so a fetch is
/// never enabled without a consumer (see the module doc comment on C8).
/// `nrdma_data_use` is documented as `bit[0]` ALU, `bit[1]` CPEND (tied to
/// 0 -- BN has no CPEND stage), `bit[2]` MUL, `bit[3]` TRT.
///
/// SOFTMAX.md's two-pass form uses [`BnPlaneOp::Add`] and
/// [`BnPlaneOp::Mul`] in *separate* tasks -- the max subtraction rides the
/// `exp` task's ALU, the reciprocal rides a later task's MUL -- so
/// [`BnPlaneOp::AddThenMul`] is not on that critical path. It is here
/// because it is the variant that pins down the stage *order*, which no
/// register description in this tree states.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BnPlaneOp {
    /// Per-channel add only: `BN_CFG.bn_alu_algo=2` with `bn_alu_src=1`,
    /// MUL bypassed, `nrdma_data_use=0b0001`.
    Add,
    /// Per-channel multiply only: `BN_MUL_CFG.bn_mul_src=1`, ALU
    /// bypassed, `nrdma_data_use=0b0100`.
    Mul,
    /// Both engaged, `nrdma_data_use=0b0101`. Which order the hardware
    /// applies them in -- `(x + a) * m` or `x * m + a` -- is exactly what
    /// the probe uses this variant to settle.
    AddThenMul,
}

impl BnPlaneOp {
    /// `RDMA_NRDMA_CFG.nrdma_data_use` for this variant.
    pub fn nrdma_data_use(self) -> u32 {
        match self {
            BnPlaneOp::Add => 0b0001,
            BnPlaneOp::Mul => 0b0100,
            BnPlaneOp::AddThenMul => 0b0101,
        }
    }

    fn alu_engaged(self) -> bool {
        matches!(self, BnPlaneOp::Add | BnPlaneOp::AddThenMul)
    }

    fn mul_engaged(self) -> bool {
        matches!(self, BnPlaneOp::Mul | BnPlaneOp::AddThenMul)
    }
}

/// Logical shape for the pass-through-plus-BN-plane task: one input cube,
/// one per-channel operand plane, one output cube of the same shape as the
/// input. fp16 only -- the int8 path would need a zero-point/scale recipe
/// no capture in this tree shows, which is the same line
/// [`crate::rocket::elementwise::EwUnaryShape`] draws for the same reason.
#[derive(Clone, Copy, Debug)]
pub struct BnPlaneShape {
    pub width: u32,
    pub height: u32,
    /// Real (unpadded) channel count. The task header is programmed at
    /// [`FEATURE_ATOMIC_SIZE`] granularity, and the operand plane has to
    /// be sized for *that* -- see [`bn_plane_bytes`].
    pub channels: u32,
    pub op: BnPlaneOp,
    /// `BN_MUL_CFG.bn_mul_shift_value`, and its negated twin
    /// `DATA_FORMAT.bn_mul_shift_value_neg`, which
    /// [`crate::rocket::activation::build_lut_regcmd`] programs as a pair.
    ///
    /// Meaningful only when [`BnPlaneOp::mul_engaged`]. A 6-bit shift
    /// field alongside a 16-bit `bn_mul_operand` register is what a
    /// *fixed-point* multiplier looks like, and the BS plane's int8 form
    /// is exactly that (an `i16` multiplier at shift 14,
    /// `conv::BS_MUL_SHIFT_VALUE`). Whether the NRDMA-sourced multiplier
    /// is also fixed-point at fp16, or an IEEE `f32` the way the fp16 BS
    /// *bias* is, is the probe's second question. Pass `0` for the float
    /// hypothesis and `14` for the fixed-point one.
    pub mul_shift: u32,
}

impl BnPlaneShape {
    /// Channel count the task header is programmed for -- the logical
    /// count rounded up to [`FEATURE_ATOMIC_SIZE`], matching
    /// `build_unary_regcmd`'s own `task_channels`.
    pub fn programmed_channels(&self) -> u32 {
        self.channels
            .max(FEATURE_ATOMIC_SIZE)
            .next_multiple_of(FEATURE_ATOMIC_SIZE)
    }
}

pub struct BnPlaneBuffers {
    pub input_addr: u32,
    /// Base of the per-channel operand plane, `RDMA_BN_BASE_ADDR`. Must
    /// address at least [`bn_plane_bytes`] bytes: the DPU is programmed
    /// for [`BnPlaneShape::programmed_channels`], so a plane sized to the
    /// *logical* count can have NRDMA reach into whatever follows it --
    /// the same trap [`crate::rocket::tensor_layout::rocket_fp16_bias_storage_size`]
    /// exists to close on the BS side.
    pub plane_addr: u32,
    pub output_addr: u32,
}

/// Bytes an ALU operand plane occupies for a task programmed at
/// `programmed_channels`: four per channel.
///
/// **Hardware-confirmed** on an RK3588 2026-11-24 by
/// `tests/bn_plane_probe_hw.rs`: the ALU operand is a flat IEEE `f32` at
/// `channel * 4`, which is the same form the one previously-confirmed
/// plane in this tree takes (the fp16 BS ALU operand is "a 32-bit float at
/// the RK3588 BRDMA boundary" -- `rocket_fp16_bias_storage_size`,
/// `pack_fp16_bias_to_rocket`). The fixed-point `i32` form BS takes at
/// int8 was ruled out in the same run.
///
/// The MUL operand is **not** the same width: see
/// [`bn_mul_plane_bytes`]. The two register widths predicted this --
/// `BN_ALU_CFG` is a single `Bits<32>` field, `BN_MUL_CFG.bn_mul_operand`
/// is 16 bits -- and the hardware agrees with both.
pub fn bn_plane_bytes(programmed_channels: u32) -> usize {
    programmed_channels as usize * 4
}

/// Bytes a MUL operand plane occupies: **two** per channel, not four.
///
/// **Hardware-confirmed** in the same run as [`bn_plane_bytes`], and found
/// the hard way. The probe's first sweep loaded the multiplier as an `f32`
/// at a 4-byte pitch and as fixed point at shifts 0 and 14, and every
/// candidate failed -- but failed in a way that named the answer: every
/// *even* channel returned exactly zero while every odd one returned the
/// input scaled by the fp16 interpretation of the *upper* half of the
/// preceding `f32`. A 4-byte-pitch plane read 2 bytes at a time is exactly
/// that pattern. So the multiplier is a 16-bit IEEE **fp16** at
/// `channel * 2`, with `BN_MUL_CFG.bn_mul_shift_value` at 0.
///
/// That the operand is a float rather than fixed point matters beyond
/// tidiness: a 6-bit shift beside a 16-bit integer would have bounded the
/// multiplier's dynamic range, and SOFTMAX.md section 4 pushes a row
/// reciprocal `1/d` through this stage for `d` anywhere in `[1, N]`. As
/// fp16 that range is nothing.
pub fn bn_mul_plane_bytes(programmed_channels: u32) -> usize {
    programmed_channels as usize * 2
}

/// Writes an ALU operand plane as flat IEEE `f32`, one per channel, and
/// zero-fills the padded tail so NRDMA's fetch of
/// [`BnPlaneShape::programmed_channels`] stays inside `buffer`.
///
/// `values` is indexed by logical channel. See [`bn_plane_bytes`] for why
/// `f32` is the hypothesis rather than a measurement.
pub fn write_bn_alu_plane_fp32(
    buffer: &mut [u8],
    programmed_channels: u32,
    values: &[f32],
) -> Result<usize, &'static str> {
    let needed = bn_plane_bytes(programmed_channels);
    if values.len() > programmed_channels as usize {
        return Err("more BN plane values than the task's programmed channel count");
    }
    if buffer.len() < needed {
        return Err("BN plane buffer is smaller than the programmed channel count needs");
    }
    buffer[..needed].fill(0);
    for (channel, &value) in values.iter().enumerate() {
        let at = channel * 4;
        buffer[at..at + 4].copy_from_slice(&value.to_bits().to_le_bytes());
    }
    Ok(needed)
}

/// Writes a MUL operand plane as flat IEEE `fp16`, one per channel, and
/// zero-fills the padded tail so NRDMA's fetch of
/// [`BnPlaneShape::programmed_channels`] stays inside `buffer`.
///
/// `values` is indexed by logical channel. A value that is not finite, or
/// whose magnitude is outside fp16's normal range, is refused rather than
/// silently flushed to zero or infinity -- a multiplier that quietly
/// became zero would make a whole channel's output vanish, which is the
/// failure mode that is hardest to attribute later. See
/// [`bn_mul_plane_bytes`] for the hardware evidence behind the format.
pub fn write_bn_mul_plane_fp16(
    buffer: &mut [u8],
    programmed_channels: u32,
    values: &[f32],
) -> Result<usize, &'static str> {
    let needed = bn_mul_plane_bytes(programmed_channels);
    if values.len() > programmed_channels as usize {
        return Err("more BN plane values than the task's programmed channel count");
    }
    if buffer.len() < needed {
        return Err("BN plane buffer is smaller than the programmed channel count needs");
    }
    buffer[..needed].fill(0);
    for (channel, &value) in values.iter().enumerate() {
        let bits = fp16_bits(value)?;
        let at = channel * 2;
        buffer[at..at + 2].copy_from_slice(&bits.to_le_bytes());
    }
    Ok(needed)
}

/// The fp16 bit pattern nearest `value`, round-to-nearest-even, refusing
/// anything outside the normal range. A sibling of `pooling.rs`'s own
/// `fp16_bits`, which panics instead because its only caller asks for
/// `65536 / k`; here the value is caller data.
fn fp16_bits(value: f32) -> Result<u16, &'static str> {
    if !value.is_finite() {
        return Err("a BN multiplier must be finite");
    }
    if value == 0.0 {
        return Ok((value.to_bits() >> 16) as u16 & 0x8000);
    }
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    if !(1..=30).contains(&exponent) {
        return Err("a BN multiplier must be inside fp16's normal range");
    }
    let mantissa = bits & 0x7f_ffff;
    let mut half = sign | ((exponent as u16) << 10) | ((mantissa >> 13) as u16);
    let remainder = mantissa & 0x1fff;
    if remainder > 0x1000 || (remainder == 0x1000 && half & 1 == 1) {
        half += 1;
    }
    Ok(half)
}

/// Builds the standalone pass-through task with the BN plane engaged:
/// DPU flying mode, `DPU_RDMA`'s MRDMA fetches the input cube, NRDMA
/// fetches the per-channel operand(s), BS and EW are bypassed, output
/// written to real memory.
///
/// Structurally this is
/// [`crate::rocket::elementwise::build_unary_regcmd`] with the
/// element-wise stage turned off and the BN stage turned on, so the
/// datapath is the same one that function's hardware tests exercise:
/// MRDMA upconverts fp16 to fp32 (`mrdma_fp16tofp32_en`), the arithmetic
/// happens in fp32, and `OUT_CVT`'s `fp32tofp16_en` narrows on the way
/// out. Three differences, all of them the point of the task:
///
/// * `BN_CFG`/`BN_MUL_CFG` engage per [`BnPlaneShape::op`] with
///   `bn_alu_src`/`bn_mul_src` selecting NRDMA instead of the config
///   register. `BN_ALU_CFG.bn_alu_operand` and
///   `BN_MUL_CFG.bn_mul_operand` are left at zero deliberately: the
///   register description says each is "ignored when `*_src=1` (operand
///   instead comes from DPU_RDMA's NRDMA)", so a nonzero value there
///   would make a passing probe ambiguous about which source was read.
/// * `RDMA_NRDMA_CFG` and `RDMA_BN_BASE_ADDR` carry a real fetch rather
///   than zero, derived from `op` (module doc comment, C8).
/// * `EW_CFG` is a full bypass, which is the vendor's `0x383`.
///
/// `ERDMA` stays explicitly disabled -- there is no second tensor here,
/// and the same C8 reasoning applies to it.
// Straight-line hardware register program; hand-transcribing it into a
// `vec![]` literal risks a copy/reorder mistake silently miscoding a register.
#[allow(clippy::vec_init_then_push)]
pub fn build_bn_plane_regcmd(shape: &BnPlaneShape, bufs: &BnPlaneBuffers) -> Vec<RegCmd> {
    assert!(
        shape.width > 0 && shape.height > 0 && shape.channels > 0,
        "build_bn_plane_regcmd: width, height, and channels must be nonzero"
    );
    assert!(
        shape.mul_shift == 0 || shape.op.mul_engaged(),
        "build_bn_plane_regcmd: mul_shift is only meaningful when the MUL stage is engaged, \
         got op={:?} mul_shift={}",
        shape.op,
        shape.mul_shift
    );
    assert!(
        bufs.plane_addr != 0,
        "build_bn_plane_regcmd: the whole point of this task is a real NRDMA fetch; a zero \
         plane address with nrdma_data_use set is the ISSUES.md C8 shape"
    );

    let task_channels = shape.programmed_channels();
    let output_area = shape.width * shape.height;
    let alu = shape.op.alu_engaged();
    let mul = shape.op.mul_engaged();

    let mut cmds: Vec<RegCmd> = Vec::new();

    cmds.push(
        Register::<DpuSPointer>::new()
            .pointer_pp_mode(Bits::new(1))
            .executer_pp_en(Bits::new(1))
            .pointer_pp_en(Bits::new(1))
            .build(),
    );
    cmds.push(
        Register::<DpuRdmaSPointer>::new()
            .pointer_pp_mode(Bits::new(1))
            .executer_pp_en(Bits::new(1))
            .pointer_pp_en(Bits::new(1))
            .build(),
    );

    cmds.push(
        Register::<DpuFeatureModeCfg>::new()
            .flying_mode(Bits::new(1))
            .output_mode(Bits::new(2))
            .burst_len(Bits::new(15))
            .build(),
    );

    cmds.push(
        Register::<DpuDataFormat>::new()
            .in_precision(Bits::new(2))
            .out_precision(Bits::new(2))
            .proc_precision(Bits::new(2))
            .bn_mul_shift_value_neg(Bits::new(shape.mul_shift))
            .build(),
    );

    cmds.push(zero::<DpuOffsetPend>());
    cmds.push(
        Register::<DpuDstBaseAddr>::new()
            .dst_base_addr(Bits::new(bufs.output_addr))
            .build(),
    );
    cmds.push(
        Register::<DpuDstSurfStride>::new()
            .dst_surf_stride(Bits::new(output_area))
            .build(),
    );
    cmds.push(
        Register::<DpuDataCubeWidth>::new()
            .width(Bits::new(shape.width - 1))
            .build(),
    );
    cmds.push(
        Register::<DpuDataCubeHeight>::new()
            .height(Bits::new(shape.height - 1))
            .build(),
    );
    cmds.push(zero::<DpuDataCubeNotchAddr>());
    cmds.push(
        Register::<DpuDataCubeChannel>::new()
            .orig_channel(Bits::new(shape.channels - 1))
            .channel(Bits::new(task_channels - 1))
            .build(),
    );

    // BS plane fully bypassed: the vendor's pass-through `BS_CFG=0x53`.
    cmds.push(
        Register::<DpuBsCfg>::new()
            .bs_bypass(Bits::new(1))
            .bs_alu_bypass(Bits::new(1))
            .bs_mul_bypass(Bits::new(1))
            .bs_relu_bypass(Bits::new(1))
            .build(),
    );
    cmds.push(zero::<DpuBsAluCfg>());
    cmds.push(zero::<DpuBsMulCfg>());
    cmds.push(zero::<DpuBsReluxCmpValue>());
    cmds.push(
        Register::<DpuBsOwCfg>::new()
            .od_bypass(Bits::new(1))
            .build(),
    );
    cmds.push(zero::<DpuBsOwOp>());
    cmds.push(
        Register::<DpuWdmaSize0>::new()
            .channel_wdma(Bits::new(task_channels - 1))
            .build(),
    );
    cmds.push(
        Register::<DpuWdmaSize1>::new()
            .height_wdma(Bits::new(shape.height - 1))
            .width_wdma(Bits::new(shape.width - 1))
            .build(),
    );

    // The BN plane: the one stage this task does not bypass. `bn_alu_algo`
    // is 2 (Add) -- the same opcode `build_lut_regcmd` programs for its own
    // scalar input-offset use of this stage, and the same numbering the EW
    // ALU's `ew_alu_algo` uses.
    cmds.push(
        Register::<DpuBnCfg>::new()
            .bn_bypass(Bits::new(0))
            .bn_alu_bypass(Bits::new(u32::from(!alu)))
            .bn_mul_bypass(Bits::new(u32::from(!mul)))
            .bn_relu_bypass(Bits::new(1))
            .bn_alu_algo(Bits::new(if alu { 2 } else { 0 }))
            .bn_alu_src(Bits::new(u32::from(alu)))
            .build(),
    );
    // Operands deliberately zero: ignored when the matching `*_src` is 1,
    // so a nonzero one would leave a passing probe ambiguous about which
    // source the hardware read (see this function's doc comment).
    cmds.push(zero::<DpuBnAluCfg>());
    cmds.push(
        Register::<DpuBnMulCfg>::new()
            .bn_mul_src(Bits::new(u32::from(mul)))
            .bn_truncate_src(Bits::new(0))
            .bn_mul_shift_value(Bits::new(shape.mul_shift))
            .bn_mul_operand(Bits::new(0))
            .build(),
    );
    cmds.push(zero::<DpuBnReluxCmpValue>());

    // EW fully bypassed: the vendor's pass-through `EW_CFG=0x383`.
    cmds.push(
        Register::<DpuEwCfg>::new()
            .ew_bypass(Bits::new(1))
            .ew_op_bypass(Bits::new(1))
            .ew_op_cvt_bypass(Bits::new(1))
            .ew_lut_bypass(Bits::new(1))
            .ew_relu_bypass(Bits::new(1))
            .build(),
    );
    cmds.push(zero::<DpuEwCvtOffsetValue>());
    cmds.push(
        Register::<DpuEwCvtScaleValue>::new()
            .ew_op_cvt_scale(Bits::new(1))
            .build(),
    );
    cmds.push(zero::<DpuEwReluxCmpValue>());

    cmds.push(zero::<DpuOutCvtOffset>());
    cmds.push(
        Register::<DpuOutCvtScale>::new()
            .fp32tofp16_en(Bits::new(1))
            .out_cvt_scale(Bits::new(1))
            .build(),
    );
    cmds.push(zero::<DpuOutCvtShift>());

    cmds.push(zero::<DpuEwOpValue0>());
    cmds.push(zero::<DpuEwOpValue1>());
    cmds.push(zero::<DpuEwOpValue2>());
    cmds.push(zero::<DpuEwOpValue3>());
    cmds.push(zero::<DpuEwOpValue4>());
    cmds.push(zero::<DpuEwOpValue5>());
    cmds.push(zero::<DpuEwOpValue6>());
    cmds.push(zero::<DpuEwOpValue7>());
    cmds.push(
        Register::<DpuSurfaceAdd>::new()
            .surf_add(Bits::new(output_area))
            .build(),
    );
    cmds.push(zero::<DpuReserved40c4>());

    // LUT block unused by this op.
    cmds.push(zero::<DpuLutAccessCfg>());
    cmds.push(zero::<DpuLutAccessData>());
    cmds.push(zero::<DpuLutCfg>());
    cmds.push(zero::<DpuLutInfo>());
    cmds.push(zero::<DpuLutLeStart>());
    cmds.push(zero::<DpuLutLeEnd>());
    cmds.push(zero::<DpuLutLoStart>());
    cmds.push(zero::<DpuLutLoEnd>());
    cmds.push(zero::<DpuLutLeSlopeScale>());
    cmds.push(zero::<DpuLutLeSlopeShift>());
    cmds.push(zero::<DpuLutLoSlopeScale>());
    cmds.push(zero::<DpuLutLoSlopeShift>());

    cmds.push(
        Register::<DpuRdmaDataCubeWidth>::new()
            .width(Bits::new(shape.width - 1))
            .build(),
    );
    cmds.push(
        Register::<DpuRdmaDataCubeHeight>::new()
            .height(Bits::new(shape.height - 1))
            .build(),
    );
    cmds.push(
        Register::<DpuRdmaDataCubeChannel>::new()
            .channel(Bits::new(task_channels - 1))
            .build(),
    );
    cmds.push(
        Register::<DpuRdmaSrcBaseAddr>::new()
            .src_base_addr(Bits::new(bufs.input_addr))
            .build(),
    );
    // BS is bypassed, so BRDMA stays off -- C8's own field.
    cmds.push(zero::<DpuRdmaBrdmaCfg>());
    cmds.push(zero::<DpuRdmaBsBaseAddr>());
    cmds.push(
        Register::<DpuRdmaNrdmaCfg>::new()
            .nrdma_data_use(Bits::new(shape.op.nrdma_data_use()))
            .build(),
    );
    cmds.push(
        Register::<DpuRdmaBnBaseAddr>::new()
            .bn_base_addr(Bits::new(bufs.plane_addr))
            .build(),
    );
    cmds.push(
        Register::<DpuRdmaErdmaCfg>::new()
            .erdma_disable(Bits::new(1))
            .build(),
    );
    cmds.push(zero::<DpuRdmaEwBaseAddr>());
    cmds.push(zero::<DpuRdmaEwSurfStride>());

    cmds.push(
        Register::<DpuRdmaFeatureModeCfg>::new()
            .flying_mode(Bits::new(1))
            .burst_len(Bits::new(15))
            .in_precision(Bits::new(2))
            .proc_precision(Bits::new(2))
            .mrdma_fp16tofp32_en(Bits::new(1))
            .build(),
    );
    cmds.push(zero::<DpuRdmaSrcDmaCfg>());
    cmds.push(zero::<DpuRdmaSurfNotch>());
    cmds.push(zero::<DpuRdmaPadCfg>());
    cmds.push(
        Register::<DpuRdmaWeight>::new()
            .e_weight(Bits::new(1))
            .n_weight(Bits::new(1))
            .b_weight(Bits::new(1))
            .m_weight(Bits::new(1))
            .build(),
    );
    cmds.push(zero::<DpuRdmaEwSurfNotch>());

    push_kick(&mut cmds, KICK_DPU | KICK_DPU_RDMA);
    cmds
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rocket::builders::RegisterMeta;

    fn program(op: BnPlaneOp, mul_shift: u32) -> Vec<RegCmd> {
        build_bn_plane_regcmd(
            &BnPlaneShape {
                width: 4,
                height: 4,
                channels: 8,
                op,
                mul_shift,
            },
            &BnPlaneBuffers {
                input_addr: 0x1000,
                plane_addr: 0x2000,
                output_addr: 0x3000,
            },
        )
    }

    /// Same `[Domain:8][Value:32][Offset:16]` decode as `conv.rs`'s own
    /// test helper.
    fn value_of<R: RegisterMeta>(cmds: &[RegCmd]) -> u32 {
        let matches: Vec<u32> = cmds
            .iter()
            .filter(|cmd| (cmd.0 >> 48) as u32 == R::DOMAIN && (cmd.0 as u32 & 0xffff) == R::OFFSET)
            .map(|cmd| (cmd.0 >> 16) as u32)
            .collect();
        assert_eq!(
            matches.len(),
            1,
            "expected exactly one write to the register"
        );
        matches[0]
    }

    /// The two bypass words this task produces are not this module's
    /// invention: they are the pair on every plain pass-through DPU copy
    /// in the vendor's own compiled softmax (`rknpu-spelunking/NOTES.md`
    /// Follow-up 5, which scanned a complete 36864-byte self-consistent
    /// capture and found `BS_CFG=0x53` with `EW_CFG=0x383` on each copy
    /// feeding its max-reduction tree). If a later edit changes a bypass
    /// bit, this is what should notice.
    #[test]
    fn bn_plane_bypass_words_match_the_vendor_passthrough() {
        let cmds = program(BnPlaneOp::Add, 0);
        assert_eq!(value_of::<DpuBsCfg>(&cmds), 0x53);
        assert_eq!(value_of::<DpuEwCfg>(&cmds), 0x383);
    }

    /// The fetch and its consumer are derived from one field, so they
    /// cannot disagree -- ISSUES.md C8 is what happens when they do.
    #[test]
    fn nrdma_fetch_matches_the_engaged_stages() {
        for (op, data_use, alu_src, mul_src, alu_bypass, mul_bypass) in [
            (BnPlaneOp::Add, 0b0001, 1, 0, 0, 1),
            (BnPlaneOp::Mul, 0b0100, 0, 1, 1, 0),
            (BnPlaneOp::AddThenMul, 0b0101, 1, 1, 0, 0),
        ] {
            let cmds = program(op, 0);
            assert_eq!(
                value_of::<DpuRdmaNrdmaCfg>(&cmds),
                data_use << 1,
                "{op:?}: nrdma_data_use is bits 4:1"
            );
            assert_ne!(value_of::<DpuRdmaBnBaseAddr>(&cmds), 0, "{op:?}");
            let bn_cfg = value_of::<DpuBnCfg>(&cmds);
            assert_eq!(bn_cfg & 1, 0, "{op:?}: bn_bypass must be clear");
            assert_eq!((bn_cfg >> 1) & 1, alu_bypass, "{op:?}: bn_alu_bypass");
            assert_eq!((bn_cfg >> 4) & 1, mul_bypass, "{op:?}: bn_mul_bypass");
            assert_eq!((bn_cfg >> 8) & 1, alu_src, "{op:?}: bn_alu_src");
            assert_eq!(
                value_of::<DpuBnMulCfg>(&cmds) & 1,
                mul_src,
                "{op:?}: bn_mul_src"
            );
            // BRDMA, the DMA C8 was actually about, stays off throughout.
            assert_eq!(value_of::<DpuRdmaBrdmaCfg>(&cmds), 0, "{op:?}");
        }
    }

    /// `Add` engages the ALU as an add, and leaves the register-sourced
    /// operand at zero so a hardware result can only have come from the
    /// plane.
    #[test]
    fn add_programs_the_alu_opcode_and_no_register_operand() {
        let cmds = program(BnPlaneOp::Add, 0);
        assert_eq!((value_of::<DpuBnCfg>(&cmds) >> 16) & 0xf, 2);
        assert_eq!(value_of::<DpuBnAluCfg>(&cmds), 0);
        assert_eq!(value_of::<DpuBnMulCfg>(&cmds) >> 16, 0, "bn_mul_operand");
    }

    /// The shift goes to both halves of the pair `build_lut_regcmd`
    /// programs together.
    #[test]
    fn mul_shift_reaches_both_registers() {
        let cmds = program(BnPlaneOp::Mul, 14);
        assert_eq!((value_of::<DpuBnMulCfg>(&cmds) >> 8) & 0x3f, 14);
        // `DATA_FORMAT.bn_mul_shift_value_neg` is bits 15:10, not the 21:16
        // `ew_truncate_neg` above it.
        assert_eq!((value_of::<DpuDataFormat>(&cmds) >> 10) & 0x3f, 14);
    }

    #[test]
    #[should_panic(expected = "mul_shift is only meaningful")]
    fn add_only_refuses_a_shift() {
        program(BnPlaneOp::Add, 14);
    }

    #[test]
    #[should_panic(expected = "C8")]
    fn a_zero_plane_address_is_refused() {
        build_bn_plane_regcmd(
            &BnPlaneShape {
                width: 1,
                height: 1,
                channels: 8,
                op: BnPlaneOp::Add,
                mul_shift: 0,
            },
            &BnPlaneBuffers {
                input_addr: 0x1000,
                plane_addr: 0,
                output_addr: 0x3000,
            },
        );
    }

    #[test]
    fn the_plane_is_sized_for_the_programmed_channels_not_the_logical_ones() {
        let shape = BnPlaneShape {
            width: 1,
            height: 1,
            channels: 3,
            op: BnPlaneOp::Add,
            mul_shift: 0,
        };
        assert_eq!(shape.programmed_channels(), 16);
        assert_eq!(bn_plane_bytes(shape.programmed_channels()), 64);

        let mut buffer = vec![0xA5u8; 64];
        let written =
            write_bn_alu_plane_fp32(&mut buffer, shape.programmed_channels(), &[1.0, -2.0, 0.5])
                .unwrap();
        assert_eq!(written, 64);
        assert_eq!(&buffer[0..4], &1.0f32.to_bits().to_le_bytes());
        assert_eq!(&buffer[4..8], &(-2.0f32).to_bits().to_le_bytes());
        assert_eq!(&buffer[8..12], &0.5f32.to_bits().to_le_bytes());
        assert!(
            buffer[12..].iter().all(|&b| b == 0),
            "padded tail must be zeroed: NRDMA fetches all 16 programmed channels"
        );
    }

    #[test]
    fn the_plane_writer_rejects_a_short_buffer_and_too_many_values() {
        assert!(write_bn_alu_plane_fp32(&mut [0; 63], 16, &[0.0]).is_err());
        assert!(write_bn_alu_plane_fp32(&mut [0; 64], 16, &[0.0; 17]).is_err());
    }
}
