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
        // One exception, and it is the one SOFTMAX.md needs: an exact
        // power of two below fp16's normal range is still exact as a
        // subnormal, since subnormals are `k * 2^-24`. `2^-15` is `512 *
        // 2^-24`, and it is the constant that turns a Q15 count into a
        // probability -- the only stage in the DPU that multiplies by a
        // true fraction is this plane, because every other one is an
        // integer multiply-then-right-shift (measured: BS MUL at shift 15
        // and `OUT_CVT` both return 0 for a Q15 count).
        //
        // A *non*-power-of-two subnormal is still refused. `1/d_q15` is
        // one of those: at the median `d` of a real attention row it has
        // about 2% of relative precision and at the worst 24%, which is
        // why the constant and the per-row reciprocal have to be separate
        // multiplies rather than folded into one plane.
        // `exponent` here is already fp16-biased-relative: it is the
        // unbiased f32 exponent plus 15, so `2^-15` arrives as 0. An fp16
        // subnormal encodes `k * 2^-24`, so `k = 2^(exponent + 9)` -- 512
        // for `2^-15`, down to 1 for `2^-24`.
        let mantissa = bits & 0x7f_ffff;
        if mantissa == 0 && (-9..=0).contains(&exponent) {
            return Ok(sign | (1u16 << (exponent + 9)));
        }
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

/// Element type [`build_plane_gather_regcmd`] writes.
///
/// The ALU plane wants `f32` and the MUL plane `fp16`
/// ([`bn_plane_bytes`], [`bn_mul_plane_bytes`]), so the gather has to be
/// able to produce either.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlaneElement {
    /// `out_precision = 2`, narrowed by `fp32tofp16_en`. Two bytes per
    /// channel, four channels short of an atom's worth.
    Fp16,
    /// `out_precision = 5`, with `fp32tofp16_en` cleared and the 4-byte
    /// writer's own `(mc_surf_out, size_e, surf_add)` triple.
    ///
    /// **Measured** (`the_fp32_writer_triple_probe`, RK3588 2026-11-24):
    /// `(0, 1, dst_surf_stride * 2)`. Reading
    /// `Precision::Fp16Accumulator`'s recipe off `conv.rs` gave `size_e =
    /// 3` -- its "float rule, size_e = output bytes - 1" -- and that hangs
    /// the job at the watchdog, as does 7; `surf_add` at twice or half the
    /// right value writes every other surface or stops after the first.
    ///
    /// `conv.rs` is explicit about why one field at a time reads as
    /// "nothing helps": the three are one writer configuration and "move
    /// together or not at all". It records two known-good int32 triples --
    /// `(0, 7, dst_surf_stride * 8)` and `(1, 1, 16)` -- and notes that the
    /// 8 is the integer-output quirk, "not the float path's natural 1 and
    /// 2x". This is that 2x, confirmed: the multiplier is 2 and `size_e` is
    /// 1, with the OW stage bypassed.
    Fp32,
}

/// Turns a *strided* width-1 cube into a dense per-channel plane operand.
///
/// SOFTMAX.md section 7's bridge, probe S2a. The row maximum is produced by
/// the PPU, whose writer rounds both cube strides up to four pixels, so a
/// width-1 pooling output puts channel `c` at `(c / 8) * 64 + (c % 8) * 2`
/// -- 48 bytes of nothing between consecutive surfaces
/// (`reduce_width_hw.rs`'s `the_reduced_vector_is_four_pixel_strided_not_dense`).
/// The BN ALU plane wants a dense `f32` at `channel * 4`. This task is one
/// DPU pass-through that closes both gaps at once: it reads the strided
/// cube and writes a dense one, converting the element type on the way.
///
/// The two mechanisms it leans on, both **unconfirmed here** before S2a:
///
/// * `RDMA_SURF_NOTCH.surf_notch_addr`, "number of pixels from the end of
///   this process's feature map to the end of the underlying shape's
///   feature map" -- `input_surface_pixels - 1` of slack per surface. The
///   register is zero in every other builder in this crate.
/// * `PlaneElement::Fp32`'s writer, above.
///
/// Why a width-1 *output* cube is dense, which is the half that needs no
/// probe: `DST_SURF_STRIDE` and `SURFACE_ADD` count 16-byte atoms, so at
/// width 1 each surface is exactly one atom and surfaces are consecutive.
/// Channel `c` then lands at `(c / lanes) * 16 + (c % lanes) * bytes`,
/// which is `c * bytes` for both lane counts -- 8 lanes at fp16, 4 at fp32.
// Straight-line hardware register program; hand-transcribing it into a
// `vec![]` literal risks a copy/reorder mistake silently miscoding a register.
#[allow(clippy::vec_init_then_push)]
pub fn build_plane_gather_regcmd(
    channels: u32,
    input_surface_pixels: u32,
    element: PlaneElement,
    input_addr: u32,
    output_addr: u32,
) -> Vec<RegCmd> {
    assert!(
        channels > 0,
        "build_plane_gather_regcmd: channels must be nonzero"
    );
    assert!(
        input_surface_pixels > 0,
        "build_plane_gather_regcmd: a surface holds at least the one pixel being read"
    );
    let task_channels = channels
        .max(FEATURE_ATOMIC_SIZE)
        .next_multiple_of(FEATURE_ATOMIC_SIZE);
    let fp32 = element == PlaneElement::Fp32;
    // The 4-byte writer's triple, measured rather than derived -- see
    // `PlaneElement::Fp32`. At width 1 the surface stride is one atom, so
    // the fp32 multiplier of 2 makes `surf_add` 2. The fp16 pass-throughs
    // in this crate leave both at their one-atom values.
    let size_e = if fp32 { 1 } else { 0 };
    let dst_surf_stride = 1;
    let surf_add = if fp32 {
        dst_surf_stride * 2
    } else {
        dst_surf_stride
    };

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
            .out_precision(Bits::new(if fp32 { 5 } else { 2 }))
            .proc_precision(Bits::new(2))
            // The parallel writer, not the serial one: `(1, 1, 16)` is a
            // different writer and hangs here.
            .mc_surf_out(Bits::new(0))
            .build(),
    );

    cmds.push(zero::<DpuOffsetPend>());
    cmds.push(
        Register::<DpuDstBaseAddr>::new()
            .dst_base_addr(Bits::new(output_addr))
            .build(),
    );
    // Width 1, height 1: one atom per surface, surfaces consecutive.
    cmds.push(
        Register::<DpuDstSurfStride>::new()
            .dst_surf_stride(Bits::new(dst_surf_stride))
            .build(),
    );
    cmds.push(zero::<DpuDataCubeWidth>());
    cmds.push(zero::<DpuDataCubeHeight>());
    cmds.push(zero::<DpuDataCubeNotchAddr>());
    cmds.push(
        Register::<DpuDataCubeChannel>::new()
            .orig_channel(Bits::new(channels - 1))
            .channel(Bits::new(task_channels - 1))
            .build(),
    );

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
            .size_e_0(Bits::new(size_e))
            .size_e_1(Bits::new(size_e))
            .size_e_2(Bits::new(size_e))
            .od_bypass(Bits::new(1))
            .build(),
    );
    cmds.push(zero::<DpuBsOwOp>());
    cmds.push(
        Register::<DpuWdmaSize0>::new()
            .channel_wdma(Bits::new(task_channels - 1))
            .build(),
    );
    cmds.push(zero::<DpuWdmaSize1>());

    cmds.push(
        Register::<DpuBnCfg>::new()
            .bn_bypass(Bits::new(1))
            .bn_alu_bypass(Bits::new(1))
            .bn_mul_bypass(Bits::new(1))
            .bn_relu_bypass(Bits::new(1))
            .build(),
    );
    cmds.push(zero::<DpuBnAluCfg>());
    cmds.push(zero::<DpuBnMulCfg>());
    cmds.push(zero::<DpuBnReluxCmpValue>());

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
            .fp32tofp16_en(Bits::new(u32::from(!fp32)))
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
            .surf_add(Bits::new(surf_add))
            .build(),
    );
    cmds.push(zero::<DpuReserved40c4>());

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

    cmds.push(zero::<DpuRdmaDataCubeWidth>());
    cmds.push(zero::<DpuRdmaDataCubeHeight>());
    cmds.push(
        Register::<DpuRdmaDataCubeChannel>::new()
            .channel(Bits::new(task_channels - 1))
            .build(),
    );
    cmds.push(
        Register::<DpuRdmaSrcBaseAddr>::new()
            .src_base_addr(Bits::new(input_addr))
            .build(),
    );
    cmds.push(zero::<DpuRdmaBrdmaCfg>());
    cmds.push(zero::<DpuRdmaBsBaseAddr>());
    cmds.push(zero::<DpuRdmaNrdmaCfg>());
    cmds.push(zero::<DpuRdmaBnBaseAddr>());
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
    // The slack this task exists for: the pixels between the one being
    // read and the end of the underlying surface.
    cmds.push(
        Register::<DpuRdmaSurfNotch>::new()
            .surf_notch_addr(Bits::new(input_surface_pixels - 1))
            .build(),
    );
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

/// Shape for the Q15-to-fp16 bridge, SOFTMAX.md §9's last host step.
///
/// `exp` emits the table's Q15 integer as `int16`
/// ([`crate::rocket::activation::build_lut_q15_regcmd`]), and a standalone
/// softmax has to hand the rest of the model fp16 probabilities. That needs
/// an integer-to-float read, an exact `2^-15`, and the per-row `1/d` -- and
/// the last two are separate stages, so the one task can do all of it:
///
/// * **`2^-15`** on the **BS** MUL stage, operand 1 at shift 15, from the
///   config register rather than BRDMA so that DMA stays off (ISSUES C8).
///   `conv.rs` notes that `bs_mul_bypass` bypasses the multiply "and not
///   the shift that follows it", which is why the operand and the shift are
///   programmed together here.
/// * **`1/d`** on the **BN** MUL plane from NRDMA, per channel, the operand
///   S1 measured as a flat fp16 at `channel * 2`. It has to be `1/d` in
///   *real* units: `1/d_q15` is 1.5e-7 to 3e-5, below fp16's normal range,
///   which is why the `2^-15` is a separate stage at all.
///
/// `input_int16` and `constant_shift` are separable so a probe can tell an
/// int16 *read* failing from the scaling failing: with `input_int16` false
/// and `constant_shift` 0 this is a plain fp16 pass-through, the control
/// arm.
///
/// **Status: unmeasured.** Whether MRDMA can feed an integer cube into a
/// float datapath at all is the open question -- it has an
/// `mrdma_fp16tofp32_en` and no integer equivalent.
#[derive(Clone, Copy, Debug)]
pub struct Q15BridgeShape {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    /// `in_precision = 1` ("Integer 16bit") rather than 2, and no
    /// `mrdma_fp16tofp32_en`.
    pub input_int16: bool,
    /// BS MUL shift, so a multiply by `2^-constant_shift`. Zero bypasses
    /// the stage.
    pub constant_shift: u32,
    /// Engage the BN MUL plane from NRDMA.
    pub per_channel: bool,
    /// Engage the BN **ALU** plane from NRDMA as well, which adds a
    /// per-channel value before the multiply. SOFTMAX.md §7's
    /// subtract-and-quantize stage carries `-m` here.
    pub per_channel_add: bool,
    /// Narrow the output to int8 with `OUT_CVT` scaling by this integer --
    /// the `exp` table's input scale reciprocal, 20 for §3.4's `1/20`.
    /// Zero leaves the fp16 output alone.
    ///
    /// `OUT_CVT`'s *shift* is an integer right shift (S3), but whether its
    /// 16-bit *multiply* is applied to a float datapath value as a float is
    /// what `the_subtract_and_quantize_probe` asks.
    pub out_scale: u32,
}

pub struct Q15BridgeBuffers {
    pub input_addr: u32,
    /// The per-channel `1/d` plane; zero when `per_channel` is false.
    pub plane_addr: u32,
    pub output_addr: u32,
}

// Straight-line hardware register program; hand-transcribing it into a
// `vec![]` literal risks a copy/reorder mistake silently miscoding a register.
#[allow(clippy::vec_init_then_push)]
pub fn build_q15_bridge_regcmd(shape: &Q15BridgeShape, bufs: &Q15BridgeBuffers) -> Vec<RegCmd> {
    assert!(
        shape.width > 0 && shape.height > 0 && shape.channels > 0,
        "build_q15_bridge_regcmd: width, height and channels must be nonzero"
    );
    assert!(
        (!shape.per_channel && !shape.per_channel_add) || bufs.plane_addr != 0,
        "build_q15_bridge_regcmd: the per-channel stage needs a real plane address, or this \
         is ISSUES.md C8's shape"
    );
    let task_channels = shape
        .channels
        .max(FEATURE_ATOMIC_SIZE)
        .next_multiple_of(FEATURE_ATOMIC_SIZE);
    let surface_stride = shape.width * shape.height;
    let constant = shape.constant_shift > 0;
    let input_addr = bufs.input_addr;
    let output_addr = bufs.output_addr;
    let narrow = shape.out_scale > 0;

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
            .in_precision(Bits::new(if shape.input_int16 { 1 } else { 2 }))
            .out_precision(Bits::new(if narrow { 0 } else { 2 }))
            .proc_precision(Bits::new(2))
            .mc_surf_out(Bits::new(0))
            .bs_mul_shift_value_neg(Bits::new(shape.constant_shift))
            .build(),
    );

    cmds.push(zero::<DpuOffsetPend>());
    cmds.push(
        Register::<DpuDstBaseAddr>::new()
            .dst_base_addr(Bits::new(output_addr))
            .build(),
    );
    // Width 1, height 1: one atom per surface, surfaces consecutive.
    cmds.push(
        Register::<DpuDstSurfStride>::new()
            .dst_surf_stride(Bits::new(surface_stride))
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

    cmds.push(
        Register::<DpuBsCfg>::new()
            .bs_bypass(Bits::new(u32::from(!constant)))
            .bs_alu_bypass(Bits::new(1))
            .bs_mul_bypass(Bits::new(u32::from(!constant)))
            .bs_relu_bypass(Bits::new(1))
            .build(),
    );
    cmds.push(zero::<DpuBsAluCfg>());
    cmds.push(
        Register::<DpuBsMulCfg>::new()
            // Operand 1 at shift `s` is a multiply by `2^-s`, taken from
            // the config register so BRDMA stays off (C8).
            .bs_mul_operand(Bits::new(u32::from(constant)))
            .bs_mul_shift_value(Bits::new(shape.constant_shift))
            .bs_mul_src(Bits::new(0))
            .bs_truncate_src(Bits::new(0))
            .build(),
    );
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

    cmds.push(
        Register::<DpuBnCfg>::new()
            .bn_bypass(Bits::new(u32::from(
                !shape.per_channel && !shape.per_channel_add,
            )))
            .bn_alu_bypass(Bits::new(u32::from(!shape.per_channel_add)))
            .bn_mul_bypass(Bits::new(u32::from(!shape.per_channel)))
            .bn_relu_bypass(Bits::new(1))
            .bn_alu_algo(Bits::new(if shape.per_channel_add { 2 } else { 0 }))
            .bn_alu_src(Bits::new(u32::from(shape.per_channel_add)))
            .build(),
    );
    cmds.push(zero::<DpuBnAluCfg>());
    cmds.push(
        Register::<DpuBnMulCfg>::new()
            .bn_mul_src(Bits::new(u32::from(shape.per_channel)))
            .bn_truncate_src(Bits::new(0))
            .build(),
    );
    cmds.push(zero::<DpuBnReluxCmpValue>());

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
            // The narrowing bit is the fp32-to-fp16 one, so an int8 output
            // clears it and lets `out_cvt_scale` carry the scale instead.
            .fp32tofp16_en(Bits::new(u32::from(!narrow)))
            .out_cvt_scale(Bits::new(if narrow { shape.out_scale } else { 1 }))
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
            // The narrowing bit is set, so the one-atom multiplier.
            .surf_add(Bits::new(surface_stride))
            .build(),
    );
    cmds.push(zero::<DpuReserved40c4>());

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
            .src_base_addr(Bits::new(input_addr))
            .build(),
    );
    cmds.push(zero::<DpuRdmaBrdmaCfg>());
    cmds.push(zero::<DpuRdmaBsBaseAddr>());
    cmds.push(
        Register::<DpuRdmaNrdmaCfg>::new()
            .nrdma_data_use(Bits::new(
                u32::from(shape.per_channel_add) | (u32::from(shape.per_channel) << 2),
            ))
            .build(),
    );
    cmds.push(
        Register::<DpuRdmaBnBaseAddr>::new()
            .bn_base_addr(Bits::new(if shape.per_channel || shape.per_channel_add {
                bufs.plane_addr
            } else {
                0
            }))
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
            .in_precision(Bits::new(if shape.input_int16 { 1 } else { 2 }))
            .proc_precision(Bits::new(2))
            .mrdma_fp16tofp32_en(Bits::new(u32::from(!shape.input_int16)))
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

    /// The 4-byte writer's measured triple, pinned so a regression shows
    /// up here rather than as a watchdog kill on the board. Three of the
    /// four neighbouring values hang the job and the fourth writes every
    /// other surface, so these are not free choices -- see
    /// `PlaneElement::Fp32`.
    #[test]
    fn the_fp32_gather_programs_the_measured_writer_triple() {
        let cmds = build_plane_gather_regcmd(32, 4, PlaneElement::Fp32, 0x1000, 0x2000);
        // DATA_FORMAT: out_precision 5 at bits 31:29, mc_surf_out 0 at bit 3.
        let format = value_of::<DpuDataFormat>(&cmds);
        assert_eq!(format >> 29, 5, "out_precision");
        assert_eq!((format >> 3) & 1, 0, "mc_surf_out: the parallel writer");
        // BS_OW_CFG: size_e_0 sits at bits 4:2.
        let ow = value_of::<DpuBsOwCfg>(&cmds);
        assert_eq!((ow >> 2) & 0x7, 1, "size_e_0");
        // SURFACE_ADD is twice DST_SURF_STRIDE on the float path. Both
        // fields sit at bit 4 of their registers.
        assert_eq!(value_of::<DpuSurfaceAdd>(&cmds) >> 4, 2);
        assert_eq!(value_of::<DpuDstSurfStride>(&cmds) >> 4, 1);
        // And the narrowing bit is what distinguishes it from the fp16 arm.
        assert_eq!(
            value_of::<DpuOutCvtScale>(&cmds) >> 16 & 1,
            0,
            "fp32tofp16_en"
        );

        let fp16 = build_plane_gather_regcmd(32, 4, PlaneElement::Fp16, 0x1000, 0x2000);
        assert_eq!(value_of::<DpuDataFormat>(&fp16) >> 29, 2);
        assert_eq!((value_of::<DpuBsOwCfg>(&fp16) >> 2) & 0x7, 0);
        assert_eq!(value_of::<DpuSurfaceAdd>(&fp16) >> 4, 1);
    }

    /// The slack the gather exists to skip.
    #[test]
    fn the_gather_programs_the_input_surface_slack() {
        for pixels in [1u32, 4] {
            let cmds = build_plane_gather_regcmd(8, pixels, PlaneElement::Fp32, 0x1000, 0x2000);
            assert_eq!(
                value_of::<DpuRdmaSurfNotch>(&cmds) >> 4,
                pixels - 1,
                "surf_notch_addr is bits 31:4, in pixels"
            );
        }
    }

    /// The subnormal encoding, pinned: `2^-15` is `512 * 2^-24`, so its
    /// fp16 word is `0x0200`. Getting this wrong produced a factor of four
    /// on the board before it produced a unit-test failure, which is the
    /// argument for the test.
    #[test]
    fn exact_power_of_two_subnormals_encode() {
        for (value, bits) in [
            (2.0f32.powi(-15), 0x0200u16),
            (2.0f32.powi(-16), 0x0100),
            (2.0f32.powi(-24), 0x0001),
            (2.0f32.powi(-14), 0x0400), // the smallest normal
        ] {
            let mut buffer = vec![0u8; 32];
            write_bn_mul_plane_fp16(&mut buffer, 16, &[value]).unwrap();
            assert_eq!(
                u16::from_le_bytes([buffer[0], buffer[1]]),
                bits,
                "2^{} encodes wrong",
                value.log2()
            );
        }
        // Not a power of two and below the normal range: refused, because
        // its precision there is a few bits (SOFTMAX.md section 4).
        assert!(write_bn_mul_plane_fp16(&mut vec![0u8; 32], 16, &[2.83e-6]).is_err());
        assert!(write_bn_mul_plane_fp16(&mut vec![0u8; 32], 16, &[2.0f32.powi(-25)]).is_err());
    }

    #[test]
    fn the_plane_writer_rejects_a_short_buffer_and_too_many_values() {
        assert!(write_bn_alu_plane_fp32(&mut [0; 63], 16, &[0.0]).is_err());
        assert!(write_bn_alu_plane_fp32(&mut [0; 64], 16, &[0.0; 17]).is_err());
    }
}
