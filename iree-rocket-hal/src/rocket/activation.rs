//! Activations: the fused [`Activation`] modes and the standalone DPU LUT.
//!
//! Two genuinely different mechanisms live here because they are two
//! answers to the same question -- "what post-processing runs on a
//! convolution's accumulator output":
//!
//! - [`Activation::Relu`]/[`Activation::Relux`] fuse into the producing
//!   op's own DPU pass. They are just a field on a conv/FC/pooling shape;
//!   no separate task, no extra memory round-trip.
//! - Sigmoid/tanh/exp go through a DPU LUT ([`LutTable`]), which live
//!   hardware tracing of the vendor runtime showed *never* fuses -- it
//!   always runs as its own task reading the producer's output back from
//!   memory. [`build_lut_regcmd`] is that standalone task (DPU/MRDMA
//!   "flying mode": DPU's main input comes from DPU_RDMA rather than the
//!   convolution pipeline), and [`build_conv_then_lut_regcmd`] chains a
//!   conv into it.
//!
//! The fused [`Activation`] enum is capture-derived, from
//! [`crate::rocket::conv`]. [`LutTable`]/[`build_lut_regcmd`] are
//! Mesa-independent (there is no Mesa/Teflon reference for sigmoid/tanh at
//! all); [`build_conv_then_lut_regcmd`] chains a [`crate::rocket::conv`]
//! task into the standalone LUT task.

use crate::rocket::{
    builders::{Bits, DOMAIN_DPU, RegCmd, Register, dpu::*, dpu_rdma::*},
    conv::{self, ConvPlan, Kernels},
    regcmd::{KICK_DPU, KICK_DPU_RDMA, push_kick, zero},
    registers::REG_DPU_LUT_ACCESS_DATA,
};

/// Fused post-processing applied to a conv/FC/pooling op's own accumulator
/// before write-back, mirroring how a real compiler would fuse a
/// `linalg.generic` elementwise op into its producer rather than emit a
/// standalone op -- this hardware has no dedicated activation unit, only a
/// bypassable post-processing stage on the op that already runs (DPU's BS
/// core, for conv/FC; only reachable on pooling paths that have a real DPU
/// stage ahead of PPU).
///
/// Hardware-validated on real RK3588 (via a since-retired mesa-lineage-only
/// test -- `conv.rs`'s own BN-stage fused activation is the production path
/// now, hardware-validated separately by `conv_phase1_validation_hw.rs`):
/// BS's ALU-then-RELU-then-MUL bit ordering means this applies as
/// `relu(bias_add(x))`, the standard fusion order -- confirmed by `Relu`
/// never decreasing output vs. `None` across a fill sweep, and `Relux`'s
/// independent upper clamp confirmed by `cmp: 0` forcing constant output
/// regardless of input (domain-independent, since 0 clamps in any scale).
/// Still open: what real-world numeric domain a non-zero `cmp` needs to be
/// in for a given quantization scale -- `cmp` values of 1000/3000 showed no
/// visible effect across accumulator magnitudes reached by uniform-fill
/// inputs 100..140 at this test's placeholder scale=1.0, so a useful `cmp`
/// for a real (calibrated) shape needs deriving from that shape's own
/// scale/shift math, not assumed to transfer from this test's constants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    None,
    Relu,
    /// Clamps to `cmp` in addition to the zero floor (`DPU_BS_RELUX_CMP_VALUE`).
    Relux {
        cmp: u32,
    },
}

/// A DPU LUT (piecewise lookup table) configuration, as used by
/// `build_lut_regcmd`. Confirmed via `rknpu-spelunking/NOTES.md`'s
/// "Decoding the DPU LUT block: sigmoid/tanh regcmd capture" section --
/// exported standalone `nn.Sigmoid()`/`nn.Tanh()` ONNX models through
/// `rknn-convert` (real `rknn-toolkit2` under the hood) and decoded the
/// compiled `.rknn`'s regcmd blob directly, cross-validating the recipe
/// across two independent activation functions.
///
/// The domain/indexing recipe is compiler-generic, confirmed byte-
/// identical between the sigmoid and tanh captures, and is therefore
/// hardcoded inside the builder rather than exposed here (same
/// risk-avoidance choice as `FC_PHYSICAL_HEIGHT` in the FC section below --
/// narrow the API to only what's actually been observed): the LUT domain
/// is always split at `x=0` into `LE = [-16384, 0)` and `LO = [0, 16384]`,
/// each table addressed as `index = (x - region_start) >> 5`, giving
/// exactly 513 entries per table (both known-good captures agree). Only
/// the table contents, EW op bypass state, and the LE-side underflow
/// extrapolation slope vary
/// per function -- `lut_tables::SIGMOID_LE`/`SIGMOID_LO`/`TANH_LE`/
/// `TANH_LO` are the exact captured vendor data, not reimplemented math.
#[derive(Clone, Copy, Debug)]
pub struct LutTable {
    /// 513 raw 16-bit table entries for the `x < 0` half.
    pub le_entries: &'static [u16; 513],
    /// 513 raw 16-bit table entries for the `x >= 0` half.
    pub lo_entries: &'static [u16; 513],
    /// `DPU_EW_CFG.EW_OP_BYPASS` captured for this specific function.
    /// Sigmoid uses `EW_CFG=0x300`; tanh uses `EW_CFG=0x302`.
    pub ew_op_bypass: u8,
    /// Extrapolation slope for `x` below `-16384` (`LUT_LE_SLOPE_SCALE`/
    /// `_SHIFT`'s `UFLOW` sub-fields). Zero in the tanh capture (flat
    /// clamp); nonzero in sigmoid's (a small residual slope, consistent
    /// with sigmoid's tail not being fully flat even far from zero).
    pub le_slope_uflow_scale: u16,
    pub le_slope_uflow_shift: u8,
}

impl LutTable {
    pub fn sigmoid() -> Self {
        LutTable {
            le_entries: &crate::rocket::lut_tables::SIGMOID_LE,
            lo_entries: &crate::rocket::lut_tables::SIGMOID_LO,
            ew_op_bypass: 0,
            le_slope_uflow_scale: 23107,
            le_slope_uflow_shift: 22,
        }
    }

    pub fn tanh() -> Self {
        LutTable {
            le_entries: &crate::rocket::lut_tables::TANH_LE,
            lo_entries: &crate::rocket::lut_tables::TANH_LO,
            ew_op_bypass: 1,
            le_slope_uflow_scale: 0,
            le_slope_uflow_shift: 0,
        }
    }

    /// Softmax's exp(x) step (Phase 5 of the ukernel roadmap), routed
    /// through this exact same standalone DPU LUT path -- confirmed via
    /// live hardware trace of the vendor runtime, byte-identical recipe to
    /// `sigmoid()`/`tanh()` above (`rknpu-spelunking/NOTES.md`'s softmax
    /// Phase 5 "Follow-up 4" section): same domain split, same 513-entry
    /// indexing, same `EW_CFG` field layout. Two things are specific to
    /// exp and confirmed only from this one real capture (not fit across
    /// multiple scales/zero-points the way `sigmoid()`/`tanh()`'s tables
    /// were cross-validated against each other):
    /// - `le_slope_uflow_scale`/`_shift` = 0 (flat clamp below `x=-16384`,
    ///   same choice as `tanh()`, confirmed by direct register capture,
    ///   not inferred by analogy).
    /// - `EXP_LO` (the `x >= 0` half) is a placeholder, not real captured
    ///   data -- see its own doc comment in `lut_tables.rs`. Softmax's
    ///   max-subtraction guarantees this half is never legitimately read,
    ///   but a caller building a *different* op around this table (not
    ///   softmax's own max-subtract-then-exp pattern) would get an
    ///   unvalidated constant `1.0` for any `x >= 0` input.
    ///
    /// One piece of good, independent supporting evidence this recipe
    /// generalizes correctly: the real captured `BN_MUL_CFG=0x6a660000`
    /// for this op backs out to `input_scale ~= 10.49` via the *existing*
    /// `lut_bn_mul()`/`LUT_BN_SCALE_K=2596.513` formula (fit from
    /// sigmoid/tanh captures, not exp) -- i.e. `build_lut_regcmd`'s
    /// generic `input_scale`/`input_zero_point` handling doesn't need any
    /// exp-specific change, only this table.
    pub fn exp() -> Self {
        LutTable {
            le_entries: &crate::rocket::lut_tables::EXP_LE,
            lo_entries: &crate::rocket::lut_tables::EXP_LO,
            ew_op_bypass: 1,
            le_slope_uflow_scale: 0,
            le_slope_uflow_shift: 0,
        }
    }
}

/// Writes one LUT table bank via the auto-incrementing `LUT_ACCESS_DATA`
/// address port: write `LUT_ACCESS_CFG` exactly once (write mode,
/// `table_id`, addr 0), then one `LUT_ACCESS_DATA` write per entry -- no
/// separate address register write between entries, confirmed by the
/// vendor capture never interleaving one (see `LutTable`'s doc comment).
fn push_lut_table(cmds: &mut Vec<RegCmd>, table_id: u32, entries: &[u16; 513]) {
    cmds.push(
        Register::<DpuLutAccessCfg>::new()
            .lut_access_type(Bits::new(1))
            .lut_table_id(Bits::new(table_id))
            .lut_addr(Bits::new(0))
            .build(),
    );
    for &entry in entries.iter() {
        let data = entry as i16 as i32 as u32;
        cmds.push(RegCmd::new(DOMAIN_DPU, REG_DPU_LUT_ACCESS_DATA, data));
    }
}

fn push_lut_tables_and_config(cmds: &mut Vec<RegCmd>, table: LutTable) {
    push_lut_table(cmds, 0, table.le_entries);
    push_lut_table(cmds, 1, table.lo_entries);
    cmds.push(
        Register::<DpuLutCfg>::new()
            .lut_hybrid_priority(Bits::new(1))
            .lut_oflow_priority(Bits::new(1))
            .lut_lo_le_mux(Bits::new(2))
            .build(),
    );
    cmds.push(
        Register::<DpuLutInfo>::new()
            .lut_le_index_select(Bits::new(5))
            .lut_lo_index_select(Bits::new(5))
            .build(),
    );
    cmds.push(
        Register::<DpuLutLeStart>::new()
            .lut_le_start(Bits::new((-16384i32) as u32))
            .build(),
    );
    cmds.push(
        Register::<DpuLutLeEnd>::new()
            .lut_le_end(Bits::new(0))
            .build(),
    );
    cmds.push(
        Register::<DpuLutLoStart>::new()
            .lut_lo_start(Bits::new(0))
            .build(),
    );
    cmds.push(
        Register::<DpuLutLoEnd>::new()
            .lut_lo_end(Bits::new(16384))
            .build(),
    );
    cmds.push(
        Register::<DpuLutLeSlopeScale>::new()
            .lut_le_slope_uflow_scale(Bits::new(table.le_slope_uflow_scale as u32))
            .build(),
    );
    cmds.push(
        Register::<DpuLutLeSlopeShift>::new()
            .lut_le_slope_uflow_shift(Bits::new(table.le_slope_uflow_shift as u32))
            .build(),
    );
    cmds.push(zero::<DpuLutLoSlopeScale>());
    cmds.push(zero::<DpuLutLoSlopeShift>());
}

/// Empirically-fit constant relating a shape's `input_scale` to DPU BN's
/// multiply stage in `build_lut_regcmd` -- reverse-engineered from
/// 5 independent int8-quantized `rknn-toolkit2` sigmoid captures at
/// different calibration scales (see `lut_bn_mul`'s doc comment and
/// `rknpu-spelunking/NOTES.md`). Not a documented hardware constant.
const LUT_BN_SCALE_K: f32 = 2596.513;

/// `BN_MUL_OPERAND`/`BN_MUL_SHIFT` for `build_lut_regcmd`:
/// `multiplier = input_scale * LUT_BN_SCALE_K`, normalized (standard
/// mantissa/exponent split) so `operand = round(multiplier * 2^shift)`
/// lands in `[16384, 32768)` -- reconstructs the captured register values
/// exactly across all 5 known data points (see the call site's doc
/// comment for which zero points were used and the caveat on
/// `lut_bn_alu`).
fn lut_bn_mul(input_scale: f32) -> (u32, u32) {
    let multiplier = input_scale * LUT_BN_SCALE_K;
    let e = multiplier.log2().floor() as i32;
    let shift = 14 - e;
    let operand = (multiplier * 2f32.powi(shift)).round() as u32;
    (operand, shift as u32)
}

/// `BN_ALU_OPERAND` for `build_lut_regcmd`: `-real_zero_point *
/// bn_mul_operand`. Exact for 3 of 5 known data points; an unresolved ~1/16
/// discrepancy showed up for the other 2 (both `real_zero_point == 42`) --
/// see the call site's doc comment. Always exactly right when
/// `real_zero_point == 0`.
fn lut_bn_alu(real_zero_point: i32, bn_mul_operand: u32) -> u32 {
    let alu = -(real_zero_point as i64) * (bn_mul_operand as i64);
    alu as i32 as u32
}

fn lut_bn_alu_supports_zero_point(real_zero_point: i32) -> bool {
    // Exact against the current captured data set for -128, -2, and 127.
    // Zero is exact by construction because the ALU operand is 0 regardless
    // of the multiply operand. Known bad captures exist for 42.
    matches!(real_zero_point, -128 | -2 | 0 | 127)
}

/// `OUT_CVT_SCALE`/`OUT_CVT_SHIFT` for `build_lut_regcmd`.
/// `build_conv_cna_core_dpu_dpu_rdma`'s normal `out_scale`/`out_shift`
/// computation targets a raw conv accumulator's magnitude (via
/// `conv_scale = input_scale*weights_scale/output_scale`) -- the wrong
/// domain entirely for a LUT table's own already-normalized output
/// (`lut_tables.rs`'s entries represent `real_value * 32768`, a fixed
/// Q15-like encoding independent of this op's quantization params).
/// Reverse-engineered from the same 4 int8-quantized captures used for
/// `lut_bn_mul`: `multiplier = 1.0 / (32768.0 * output_scale)`, normalized
/// the same way (`shift = 14 - floor(log2(multiplier))`, `operand =
/// round(multiplier * 2^shift)`) -- reconstructs all 4 captured
/// `OUT_CVT_SCALE` values closely (all `OUT_CVT_SHIFT=21` in this
/// output_scale range, itself a consequence of the shared normalization
/// formula rather than a separately-confirmed constant).
fn lut_out_cvt(output_scale: f32) -> (u32, u32) {
    let multiplier = 1.0 / (32768.0 * output_scale);
    let e = multiplier.log2().floor() as i32;
    let shift = 14 - e;
    let operand = (multiplier * 2f32.powi(shift)).round() as u32;
    (operand, shift as u32)
}

/// Logical shape for a standalone DPU LUT pass -- the only LUT-activation
/// path this crate builds, matching the vendor compiler's decoded
/// sigmoid/tanh routing (confirmed via live hardware trace of the vendor
/// runtime itself, see `rknpu-spelunking/NOTES.md`): DPU runs in flying
/// mode, DPU_RDMA/MRDMA supplies the input tensor directly from memory,
/// and CNA/CORE are not kicked. Chain a real conv task (with
/// `Activation::None`) into this via `build_conv_then_lut_regcmd` for a
/// conv-then-sigmoid/tanh pipeline -- LUT activations are never fused
/// into the conv's own DPU pass, unlike `Relu`/`Relux`.
#[derive(Clone, Copy)]
pub struct LutShape {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub input_zero_point: u32,
    pub output_zero_point: u32,
    /// Quantization scale for the bytes read by DPU_RDMA/MRDMA.
    pub input_scale: f32,
    /// Quantization scale for the bytes written by DPU.
    pub output_scale: f32,
}

pub struct LutBuffers {
    pub input_addr: u32,
    pub output_addr: u32,
}

pub fn build_lut_regcmd(shape: &LutShape, bufs: &LutBuffers, table: LutTable) -> Vec<RegCmd> {
    assert!(
        shape.width > 0 && shape.height > 0 && shape.channels > 0,
        "build_lut_regcmd: width, height, and channels must be nonzero"
    );

    const FEATURE_ATOMIC_SIZE: u32 = 16;

    let task_channels = shape
        .channels
        .max(FEATURE_ATOMIC_SIZE)
        .next_multiple_of(FEATURE_ATOMIC_SIZE);
    let surface_stride = shape.width * shape.height * task_channels;
    let out_offset = shape.output_zero_point.wrapping_sub(0x80);
    let (bn_mul_operand, bn_mul_shift) = lut_bn_mul(shape.input_scale);
    let real_zero_point = shape.input_zero_point.wrapping_sub(0x80) as i8 as i32;
    assert!(
        lut_bn_alu_supports_zero_point(real_zero_point),
        "build_lut_regcmd: only input_zero_point values whose decoded signed zero point is one of \
         -128, -2, 0, or 127 are supported for now; got raw input_zero_point={} (decoded {})",
        shape.input_zero_point,
        real_zero_point
    );
    let bn_alu_operand = lut_bn_alu(real_zero_point, bn_mul_operand);
    let (out_cvt_scale, out_cvt_shift) = lut_out_cvt(shape.output_scale);

    let mut cmds = Vec::new();

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
            .bn_mul_shift_value_neg(Bits::new(bn_mul_shift))
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

    cmds.push(
        Register::<DpuBnCfg>::new()
            .bn_alu_algo(Bits::new(2))
            .bn_alu_src(Bits::new(0))
            .bn_relu_bypass(Bits::new(1))
            .bn_mul_bypass(Bits::new(0))
            .bn_alu_bypass(Bits::new(0))
            .bn_bypass(Bits::new(0))
            .build(),
    );
    cmds.push(
        Register::<DpuBnAluCfg>::new()
            .bn_alu_operand(Bits::new(bn_alu_operand))
            .build(),
    );
    cmds.push(
        Register::<DpuBnMulCfg>::new()
            .bn_mul_operand(Bits::new(bn_mul_operand))
            .bn_mul_shift_value(Bits::new(bn_mul_shift))
            .bn_mul_src(Bits::new(0))
            .bn_truncate_src(Bits::new(0))
            .build(),
    );
    cmds.push(zero::<DpuBnReluxCmpValue>());

    cmds.push(
        Register::<DpuEwCfg>::new()
            .ew_relu_bypass(Bits::new(1))
            .ew_op_cvt_bypass(Bits::new(1))
            .ew_lut_bypass(Bits::new(0))
            .ew_op_bypass(Bits::new(table.ew_op_bypass as u32))
            .ew_bypass(Bits::new(0))
            .build(),
    );
    cmds.push(zero::<DpuEwCvtOffsetValue>());
    cmds.push(
        Register::<DpuEwCvtScaleValue>::new()
            .ew_op_cvt_scale(Bits::new(1))
            .build(),
    );
    cmds.push(zero::<DpuEwReluxCmpValue>());

    cmds.push(
        Register::<DpuOutCvtOffset>::new()
            .out_cvt_offset(Bits::new(out_offset))
            .build(),
    );
    cmds.push(
        Register::<DpuOutCvtScale>::new()
            .out_cvt_scale(Bits::new(out_cvt_scale))
            .build(),
    );
    cmds.push(
        Register::<DpuOutCvtShift>::new()
            .out_cvt_shift(Bits::new(out_cvt_shift))
            .build(),
    );
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
            .surf_add(Bits::new(surface_stride))
            .build(),
    );
    cmds.push(zero::<DpuReserved40c4>());

    push_lut_tables_and_config(&mut cmds, table);

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

/// DMA addresses for a conv-then-LUT-activation pipeline. `Relu`/`Relux`
/// fuse into the conv's own DPU pass but sigmoid/tanh never do -- confirmed
/// by live hardware tracing of the vendor runtime itself
/// (`rknpu-spelunking/NOTES.md`), which showed a real separate task
/// reading the conv's output back from memory, not on-chip pipelining, and
/// reproduced across a real 47-point geometry/precision sweep of static
/// compiles (`iree-rocket-design-spike/DESIGN_NOTES.md`'s "Conv+LUT fusion
/// sweep" section) -- every capture split into two tasks, none fused.
/// `intermediate_addr` is that real round-trip buffer: pure inter-task DMA
/// memory the CPU never touches, so it doesn't need `prep_bo()`/`fini_bo()`
/// the way a job-boundary buffer would.
pub struct ConvThenLutBuffers {
    pub input_addr: u32,
    pub weights_addr: u32,
    pub bias_addr: u32,
    pub intermediate_addr: u32,
    pub output_addr: u32,
}

/// Builds a conv task (forced `Activation::None` -- LUT activations are
/// never fused into the conv's own DPU pass, see `ConvThenLutBuffers`'s
/// doc comment) chained into a standalone LUT task, matching the vendor
/// runtime's own real routing for conv->sigmoid/tanh. Returns
/// `(conv_cmds, lut_cmds)` -- submit both as one multi-task job via
/// `device::submit_tasks(fd, &[(conv_cmd_addr, conv_cmds.len() as u32), \
/// (lut_cmd_addr, lut_cmds.len() as u32)], ...)`, not as two separate
/// jobs (the kernel dispatches task 2 only after task 1's own hardware
/// completion IRQ fires, so no CPU-side wait on `intermediate_addr` is
/// needed in between -- see `submit_tasks`'s doc comment).
pub fn build_conv_then_lut_regcmd(
    conv_shape: &conv::Shape,
    kernels: Kernels,
    lut_shape: &LutShape,
    table: LutTable,
    bufs: &ConvThenLutBuffers,
) -> (Vec<RegCmd>, Vec<RegCmd>) {
    assert!(
        matches!(conv_shape.activation, conv::Activation::None),
        "build_conv_then_lut_regcmd: conv_shape.activation must be Activation::None -- \
         LUT activations always run as a separate task, never fused into the conv's own DPU pass"
    );
    let mut conv_tasks = ConvPlan::new(*conv_shape, kernels).programs_with_buffers(conv::Buffers {
        input: bufs.input_addr,
        weights: bufs.weights_addr,
        bias: bufs.bias_addr,
        output: bufs.intermediate_addr,
    });
    assert_eq!(
        conv_tasks.len(),
        1,
        "build_conv_then_lut_regcmd: conv requires {} CBUF height splits; not supported by \
         this single-task LUT pipeline",
        conv_tasks.len()
    );
    let conv_cmds = conv_tasks.remove(0);
    // NOT a push_kick() call here, deliberately: ConvPlan::programs_with_buffers
    // already ends this task with its own PC trailer
    // (PCTrailer::operation_enable(PCOperationMask::CONVOLUTION), exactly
    // CNA|CORE|DPU|DPU_RDMA -- see conv.rs's tile builder). An earlier
    // version of this function pushed a second, redundant kick here, carried
    // over unexamined from `build_pooling_via_dpu_bypass_regcmd`'s
    // pre-migration Mesa-derived bypass stage (that builder never self-kicked,
    // so a caller-supplied kick
    // there was the ONLY kick, not a second one -- see that function's own
    // corrected doc comment for the real hardware evidence).
    // PC_OPERATION_ENABLE is edge-triggered, not passive state: writing it
    // twice re-kicks the same blocks immediately after the first, which is
    // what a real RK3588 run of this exact double-kick shape showed --
    // job completes, no hang, but the result comes back zeroed.
    let lut_cmds = build_lut_regcmd(
        lut_shape,
        &LutBuffers {
            input_addr: bufs.intermediate_addr,
            output_addr: bufs.output_addr,
        },
        table,
    );
    (conv_cmds, lut_cmds)
}
