//! Element-wise (EW) tensor-tensor ops on the DPU's EW/ERDMA block.
//!
//! Currently one shape: add/subtract of a conv's output with a second
//! tensor ([`EwAddShape`], [`build_add_regcmd`]/[`build_conv_then_add_regcmd`]).
//! The EW block is otherwise fully bypassed by every other builder in this
//! crate.
//!
//! `mesa_conv`-free -- capture-derived, like [`crate::rocket::activation`]'s
//! `build_conv_then_lut_regcmd`. A dedicated 47-model `Conv2d(x) +/- w`
//! sweep (`iree-rocket-design-spike`'s `sweep_convadd_generate.py`/
//! `sweep_convadd_diff.py`, see `DESIGN_NOTES.md`'s "Conv+add fusion sweep"
//! section) found the vendor never fuses the add into the producing conv's
//! own task -- 0 of 257 recovered task pairs combine them -- always two
//! separate tasks, the same two-task shape [`build_conv_then_lut_regcmd`]
//! already uses for sigmoid/tanh. This module's builders implement that
//! confirmed shape; see [`build_add_regcmd`]'s own doc comment for exactly
//! which register values are capture-confirmed vs. inferred.
//!
//! A standalone element-wise op (EW without a producing conv) would reuse
//! [`build_add_regcmd`] directly, the same way [`build_lut_regcmd`] serves
//! both the standalone and conv-then-LUT cases.
//!
//! [`build_lut_regcmd`]: crate::rocket::activation::build_lut_regcmd
//! [`build_conv_then_lut_regcmd`]: crate::rocket::activation::build_conv_then_lut_regcmd

use crate::rocket::{
    builders::{Bits, RegCmd, Register, dpu::*, dpu_rdma::*},
    conv::{self, ConvPlan, Kernels, Multiplier},
    regcmd::{KICK_CNA, KICK_CORE, KICK_DPU, KICK_DPU_RDMA, push_kick, zero},
};

/// Element precision for [`EwAddShape`]. A separate enum from
/// [`crate::rocket::conv::Precision`] -- this task carries no quantization
/// payload of its own (its int8 fields are plain ratios/zero-points, not a
/// [`conv::Quantization`]), so there is nothing for a shared type to add.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EwPrecision {
    Fp16,
    Int8,
}

/// Logical shape for the standalone EW-add/subtract task -- the second half
/// of the two-task `Conv2d(x) +/- w` pipeline
/// ([`build_conv_then_add_regcmd`]), or usable on its own
/// ([`build_add_regcmd`]) the same way [`crate::rocket::activation::LutShape`]
/// is. `width`/`height`/`channels` are the shape of BOTH tensors this task
/// combines (the producing conv's own output shape, when chained) -- this
/// op has no reduction, so input and output geometry are identical.
#[derive(Clone, Copy, Debug)]
pub struct EwAddShape {
    pub width: u32,
    pub height: u32,
    /// Real (unpadded) channel count.
    pub channels: u32,
    pub precision: EwPrecision,
    /// Raw `ew_alu_algo` opcode (bits 19:16 of `EW_CFG`). Only `2` (Add) and
    /// `4` (Minus) are hardware-confirmed for this task shape, both
    /// precisions, by the conv+add sweep -- the TRM's full documented set
    /// also includes `0=Max, 1=Min, 3=Div, 5=Abs, 6=Neg, 7=Floor, 8=Ceil`,
    /// untested here.
    ///
    /// The sweep also found this is NOT purely a caller preference: real
    /// `rknn-toolkit2` compiles route subtraction differently by precision
    /// -- fp16 always used the real `algo=4` (Minus) opcode directly (with
    /// an unnegated `EW_CVT_SCALE_VALUE`); every int8 subtraction instead
    /// reused `algo=2` (Add) with a negated scale (see
    /// `iree-rocket-hal/src/rocket/mesa_conv.rs`'s superseded `AddTensor`
    /// doc comment for the sign-negation mechanics, still accurate for the
    /// int8 case). Pick `algo` accordingly; this type does not choose for
    /// the caller.
    pub algo: u32,
    /// int8 only, ignored for fp16: this task's own real, decoded output
    /// zero point. Also used verbatim (negated) to re-center the primary
    /// operand's own zero point in `BS_ALU_CFG` -- confirmed exact, 10/10
    /// real int8 captures in the conv+add sweep, by cross-referencing each
    /// capture's EW task `BS_ALU_CFG` against its own producing conv task's
    /// `DPU_OUT_CVT_OFFSET` in the same file (always an exact negation).
    /// When chaining after a real conv via [`build_conv_then_add_regcmd`],
    /// this is the SAME value as `conv_shape`'s own
    /// `Quantization.output_zero_point` -- both the intermediate's real
    /// zero point (what `BS_ALU_CFG` undoes) and this task's own final
    /// output zero point happen to be the same field on this type since
    /// `OUT_CVT_OFFSET` reuses it too; a caller wanting genuinely different
    /// intermediate/final zero points is not supported by this shape (no
    /// real capture has shown they differ).
    pub output_zero_point: i32,
    /// int8 only, ignored for fp16: raw `EW_CVT_OFFSET_VALUE`, passed
    /// through verbatim -- same ambiguity the superseded `AddTensor.
    /// cvt_offset` already had (Mesa's own C used
    /// `operation->addition_offset` as-is with no zero-point-derived
    /// formula shown deriving it; this doesn't invent one either).
    pub w_cvt_offset: u32,
    /// int8 only, ignored for fp16: the second tensor's real scale divided
    /// by the intermediate's real (producing conv's own output) scale.
    /// Encoded into `EW_CVT_SCALE_VALUE`/`_SHIFT` via
    /// [`Multiplier::from_ratio`] -- INFERRED, not independently
    /// hardware-confirmed: the register pair is the same 16-bit-operand/
    /// 6-bit-shift shape `Multiplier` already encodes for `DPU_OUT_CVT_
    /// SCALE` elsewhere in this crate, but the real ratio rknn-toolkit2
    /// picked for the sweep's own captures isn't recoverable from decoded
    /// registers alone, so this specific mapping has not been checked
    /// against a known value the way the rest of this shape's fields have.
    pub w_scale_ratio: f64,
    /// int8 only, ignored for fp16: the intermediate's real (producing
    /// conv's own output) scale divided by this task's own final output
    /// scale. Encoded into `OUT_CVT_SCALE`/`_SHIFT` the same way
    /// `w_scale_ratio` is, with the same INFERRED caveat.
    pub output_scale_ratio: f64,
}

pub struct EwAddBuffers {
    /// The primary operand -- when chained via
    /// [`build_conv_then_add_regcmd`], the producing conv's own real output
    /// buffer, fetched here via `DPU_RDMA`'s ordinary main-fetch path
    /// (confirmed: this task's `DPU_RDMA_RDMA_FEATURE_MODE_CFG.MRDMA_
    /// DISABLE` is `0`, i.e. active, in every real capture -- the same
    /// default [`build_lut_regcmd`] already uses).
    ///
    /// [`build_lut_regcmd`]: crate::rocket::activation::build_lut_regcmd
    pub intermediate_addr: u32,
    /// The second tensor's real address, fetched via `ERDMA`
    /// (`DPU_RDMA_RDMA_ERDMA_CFG.ERDMA_DISABLE=0`, `EW_BASE_ADDR`/
    /// `EW_SURF_STRIDE` real -- confirmed in every real capture, both
    /// precisions).
    pub w_addr: u32,
    pub output_addr: u32,
}

/// Builds the standalone EW-add/subtract task: DPU flying mode, `DPU_RDMA`
/// fetches the primary operand, `ERDMA` fetches the second tensor, `EW`
/// combines them, output written to real memory. Confirmed bit-exact
/// against the conv+add sweep's real captures for BOTH precisions (see the
/// module doc comment): task header (`DPU_FEATURE_MODE_CFG`: flying_mode=1,
/// output_mode=2, burst_len=15 -- byte-identical to
/// [`build_lut_regcmd`]'s own), kick (`KICK_DPU|KICK_DPU_RDMA` only),
/// `DST_SURF_STRIDE`/`EW_SURF_STRIDE`/`SURFACE_ADD` (`width*height`, no
/// channel factor -- unlike `build_lut_regcmd`'s own `width*height*
/// task_channels`), channel padding (reuses `build_lut_regcmd`'s exact
/// `channels.max(16).next_multiple_of(16)` formula), `EW_CFG`'s shared bits
/// and its precision-dependent `edata_size`/`ew_cvt_type`, `DPU_DATA_FORMAT`/
/// `DPU_RDMA_RDMA_FEATURE_MODE_CFG`'s fp16-only precision fields, and `BS`/
/// `BN` handling (`BN` always fully bypassed; `BS` fully bypassed for fp16,
/// or for int8 a fixed-constant re-centering of the primary operand's own
/// zero point -- see [`EwAddShape::output_zero_point`]'s doc comment for the
/// exact confirmed formula). `EW_CVT_SCALE`/`OUT_CVT_SCALE`'s exact ratio
/// semantics for int8 are the one INFERRED (not independently
/// hardware-confirmed) piece -- see [`EwAddShape::w_scale_ratio`]/
/// [`EwAddShape::output_scale_ratio`]'s own doc comments.
///
/// NOT YET RUN ON REAL HARDWARE through this exact function -- structurally
/// confirmed via static capture evidence only, same status
/// `build_conv_then_lut_regcmd` had before its own hardware round. See
/// `tests/conv_with_add_hw.rs` for the gate this needs before trusting it.
///
/// [`build_lut_regcmd`]: crate::rocket::activation::build_lut_regcmd
pub fn build_add_regcmd(shape: &EwAddShape, bufs: &EwAddBuffers) -> Vec<RegCmd> {
    assert!(
        shape.width > 0 && shape.height > 0 && shape.channels > 0,
        "build_add_regcmd: width, height, and channels must be nonzero"
    );
    assert!(
        matches!(shape.algo, 2 | 4),
        "build_add_regcmd: only ew_alu_algo 2 (Add) and 4 (Minus) are hardware-confirmed for \
         this task shape, got {}",
        shape.algo
    );

    const FEATURE_ATOMIC_SIZE: u32 = 16;
    let task_channels = shape
        .channels
        .max(FEATURE_ATOMIC_SIZE)
        .next_multiple_of(FEATURE_ATOMIC_SIZE);
    let output_area = shape.width * shape.height;
    let is_fp16 = matches!(shape.precision, EwPrecision::Fp16);

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

    let mut data_format_builder = Register::<DpuDataFormat>::new();
    if is_fp16 {
        data_format_builder
            .in_precision(Bits::new(2))
            .out_precision(Bits::new(2))
            .proc_precision(Bits::new(2));
    }
    cmds.push(data_format_builder.build());

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

    if is_fp16 {
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
    } else {
        cmds.push(
            Register::<DpuBsCfg>::new()
                .bs_alu_algo(Bits::new(2))
                .bs_relu_bypass(Bits::new(1))
                .build(),
        );
        cmds.push(
            Register::<DpuBsAluCfg>::new()
                .bs_alu_operand(Bits::new(shape.output_zero_point.wrapping_neg() as u32))
                .build(),
        );
        // Fixed constant, confirmed byte-identical across all 10 real int8
        // captures regardless of geometry/channels/kernel/stride -- an
        // identity multiply, not a real per-model requantization (that work
        // is EW_CVT's/OUT_CVT's, not BS's, in this task shape).
        cmds.push(
            Register::<DpuBsMulCfg>::new()
                .bs_mul_operand(Bits::new(0x4000))
                .build(),
        );
    }
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

    // BN always fully bypassed, both precisions -- confirmed identical
    // (BN_CFG=0x53) in every real capture. Unlike build_lut_regcmd, this op
    // has no need for BN's domain-shift trick: int8's zero-point
    // re-centering is BS's job here (above), and fp16 needs neither.
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

    // ew_cvt_type/edata_size are the one EW_CFG-bit precision split: int8's
    // values are bit-identical to mesa_conv.rs's superseded fused-path
    // constants; fp16's are new (that path never emitted fp16).
    let (ew_cvt_type, edata_size) = if is_fp16 { (0, 2) } else { (1, 1) };
    cmds.push(
        Register::<DpuEwCfg>::new()
            .ew_cvt_type(Bits::new(ew_cvt_type))
            .ew_data_mode(Bits::new(1))
            .edata_size(Bits::new(edata_size))
            .ew_alu_algo(Bits::new(shape.algo))
            .ew_relu_bypass(Bits::new(1))
            .ew_lut_bypass(Bits::new(1))
            .ew_op_src(Bits::new(1)) // operand from outside (the second tensor)
            .build(),
    );

    if is_fp16 {
        cmds.push(zero::<DpuEwCvtOffsetValue>());
        cmds.push(
            Register::<DpuEwCvtScaleValue>::new()
                .ew_op_cvt_scale(Bits::new(1))
                .build(),
        );
    } else {
        cmds.push(
            Register::<DpuEwCvtOffsetValue>::new()
                .ew_op_cvt_offset(Bits::new(shape.w_cvt_offset))
                .build(),
        );
        let m = Multiplier::from_ratio(shape.w_scale_ratio);
        cmds.push(
            Register::<DpuEwCvtScaleValue>::new()
                .ew_op_cvt_scale(Bits::new(m.scale))
                .ew_op_cvt_shift(Bits::new(m.shift))
                .build(),
        );
    }
    cmds.push(zero::<DpuEwReluxCmpValue>());

    if is_fp16 {
        // Ground truth (both fp16 branches in this crate agree, see
        // build_conv_regcmd's own Fp16 arm): offset and shift zeroed, scale
        // is the fp32->fp16 conversion enable bit plus an identity scale.
        cmds.push(zero::<DpuOutCvtOffset>());
        cmds.push(
            Register::<DpuOutCvtScale>::new()
                .fp32tofp16_en(Bits::new(1))
                .out_cvt_scale(Bits::new(1))
                .build(),
        );
        cmds.push(zero::<DpuOutCvtShift>());
    } else {
        cmds.push(
            Register::<DpuOutCvtOffset>::new()
                .out_cvt_offset(Bits::new(shape.output_zero_point as u32))
                .build(),
        );
        let m = Multiplier::from_ratio(shape.output_scale_ratio);
        cmds.push(
            Register::<DpuOutCvtScale>::new()
                .out_cvt_scale(Bits::new(m.scale))
                .build(),
        );
        cmds.push(
            Register::<DpuOutCvtShift>::new()
                .out_cvt_shift(Bits::new(m.shift))
                .build(),
        );
    }

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

    // LUT block unused by this op -- all zero, confirmed in every real
    // capture (this task's DPU_LUT_ACCESS_CFG etc. never carry a real
    // table).
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
            .src_base_addr(Bits::new(bufs.intermediate_addr))
            .build(),
    );
    cmds.push(zero::<DpuRdmaBrdmaCfg>());
    cmds.push(zero::<DpuRdmaBsBaseAddr>());
    cmds.push(zero::<DpuRdmaNrdmaCfg>());
    cmds.push(zero::<DpuRdmaBnBaseAddr>());
    cmds.push(
        Register::<DpuRdmaErdmaCfg>::new()
            .erdma_data_mode(Bits::new(1))
            .erdma_data_size(Bits::new(edata_size))
            .build(),
    );
    cmds.push(
        Register::<DpuRdmaEwBaseAddr>::new()
            .ew_base_addr(Bits::new(bufs.w_addr))
            .build(),
    );
    cmds.push(
        Register::<DpuRdmaEwSurfStride>::new()
            .ew_surf_stride(Bits::new(output_area.max(12)))
            .build(),
    );

    let mut rdma_feat_mode_builder = Register::<DpuRdmaFeatureModeCfg>::new();
    rdma_feat_mode_builder
        .flying_mode(Bits::new(1))
        .burst_len(Bits::new(15));
    if is_fp16 {
        rdma_feat_mode_builder
            .in_precision(Bits::new(2))
            .proc_precision(Bits::new(2))
            .mrdma_fp16tofp32_en(Bits::new(1));
    }
    cmds.push(rdma_feat_mode_builder.build());
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

/// DMA addresses for a conv-then-EW-add/subtract pipeline. `intermediate_addr`
/// is a real inter-task round-trip buffer, same convention as
/// `ConvThenLutBuffers::intermediate_addr` -- pure inter-task DMA memory the
/// CPU never touches.
pub struct ConvThenAddBuffers {
    pub input_addr: u32,
    pub weights_addr: u32,
    pub bias_addr: u32,
    pub intermediate_addr: u32,
    pub w_addr: u32,
    pub output_addr: u32,
}

/// Builds a real conv task chained into [`build_add_regcmd`], matching the
/// vendor's own real routing for `Conv2d(x) +/- w` -- confirmed by the
/// conv+add sweep to be the ONLY shape the vendor ever emits for this
/// pattern (see the module doc comment). Returns `(conv_cmds, add_cmds)`;
/// submit both as one multi-task job via `device::submit_tasks`, same
/// discipline `build_conv_then_lut_regcmd` documents.
///
/// `add.width`/`add.height`/`add.channels` must match `conv_shape`'s own
/// output geometry (`conv_shape.output_width(kernels)`/`output_height(
/// kernels)`/`out_channels`) -- this function does not derive them
/// automatically, matching `build_conv_then_lut_regcmd`'s own convention of
/// taking the downstream shape as an explicit, independently-constructed
/// argument.
pub fn build_conv_then_add_regcmd(
    conv_shape: &conv::Shape,
    kernels: Kernels,
    add: &EwAddShape,
    bufs: &ConvThenAddBuffers,
) -> (Vec<RegCmd>, Vec<RegCmd>) {
    let mut conv_tasks = ConvPlan::new(*conv_shape, kernels).programs_with_buffers(conv::Buffers {
        input: bufs.input_addr,
        weights: bufs.weights_addr,
        bias: bufs.bias_addr,
        output: bufs.intermediate_addr,
    });
    assert_eq!(
        conv_tasks.len(),
        1,
        "build_conv_then_add_regcmd: conv requires {} CBUF height splits; not supported by \
         this single-task EW-add pipeline",
        conv_tasks.len()
    );
    let mut conv_cmds = conv_tasks.remove(0);
    // KICK_DPU_RDMA forced on for the same reason build_conv_then_lut_regcmd's
    // conv stage and build_pooling_via_dpu_bypass_regcmd's bypass stage
    // already force it: this conv writes a real memory round-trip buffer for
    // a downstream task to read, and every other memory-writing conv task in
    // this crate kicks DPU_RDMA unconditionally.
    push_kick(
        &mut conv_cmds,
        KICK_CNA | KICK_CORE | KICK_DPU | KICK_DPU_RDMA,
    );

    let add_cmds = build_add_regcmd(
        add,
        &EwAddBuffers {
            intermediate_addr: bufs.intermediate_addr,
            w_addr: bufs.w_addr,
            output_addr: bufs.output_addr,
        },
    );
    (conv_cmds, add_cmds)
}
