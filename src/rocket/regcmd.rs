//! Shared, Mesa-faithful regcmd construction for a single conv operation.
//!
//! Ported field-for-field from `mesa-rocket-userspace/rkt_regcmd.c`'s
//! `fill_first_regcmd()` and `rkt_task.c`'s `fill_task()`/
//! `rkt_split_tasks()` (see rknpu-spelunking/NOTES.md for the full
//! derivation history and hardware validation -- this is what turned
//! `rkt-basic.rs` from a permanently-hanging/wrong-output test into one
//! that dispatches, completes, and produces real, input-tracking output).
//! Originally written directly into `rkt-basic.rs` for that file's one
//! fixed shape; generalized here so `rkt-job.rs`/`rkt-simple-job.rs` can
//! share it instead of each carrying their own independently incomplete
//! regcmd builders.
//!
//! Scope, deliberately not general beyond what any of this repo's test
//! clients actually need:
//! - Single-task operations only (`rkt_split_tasks()`'s "full weights,
//!   full input" branch) -- covers every shape small enough not to need
//!   multi-task CBUF splitting, which is all three clients in this repo.
//! - No `addition_input`/`add_tensor` support -- none of the three bins
//!   use a second input tensor, so the DPU_RDMA/EW "add_tensor != -1"
//!   branches in Mesa's source are simply not implemented here.
//! - The `input_channels_real == 1 && output_channels_real > 1` "wide
//!   atomic" branch in `fill_task()` isn't implemented either -- no
//!   current client's shape reaches it, and porting an untested branch
//!   with no way to validate it would just be a latent bug. `build()`
//!   asserts this combination isn't requested rather than silently
//!   producing wrong output for it.

use crate::rocket::{
    builders::{
        Bits, DOMAIN_CORE, DOMAIN_DPU, DOMAIN_PC, RegCmd, Register, RegisterMeta, cna::*, core::*,
        dpu::*, dpu_rdma::*, ppu::*, ppu_rdma::*,
    },
    registers::{
        PC_OPERATION_ENABLE_OP_EN, PC_OPERATION_ENABLE_RESERVED_0, REG_PC_OPERATION_ENABLE,
        REG_PC_REGISTER_AMOUNTS,
    },
};

fn zero<R: RegisterMeta>() -> RegCmd {
    Register::<R>::new().build()
}

/// Logical shape of a single conv operation (`operation->*` in Mesa).
pub struct ConvShape {
    pub input_width: u32,
    pub input_height: u32,
    pub input_channels: u32,
    pub output_width: u32,
    pub output_height: u32,
    pub output_channels: u32,
    pub weights_width: u32,
    pub weights_height: u32,
    pub stride: u32,
    pub depthwise: bool,
    pub input_zero_point: u32,
    pub output_zero_point: u32,
    pub weights_zero_point: u32,
    /// Quantization scale factors. None of this repo's clients have real
    /// calibrated tensors -- 1.0 (no rescaling) for all three -- but a
    /// real caller with an actual quantized model would need real values
    /// here, so these are plumbed through rather than hardcoded.
    pub input_scale: f32,
    pub weights_scale: f32,
    pub output_scale: f32,
    pub truncate_bits: u32,
}

/// DMA addresses for the four buffers a single-task conv op needs.
pub struct ConvBuffers {
    pub input_addr: u32,
    pub weights_addr: u32,
    /// DPU's BS (bias-subtract) block is never actually bypassed by Mesa
    /// -- it always runs its ALU against a real biases buffer, even for
    /// operations with no logical bias (in which case this should just be
    /// zero-filled). See DPU_BS_CFG below.
    pub bias_addr: u32,
    pub output_addr: u32,
}

pub fn build_conv_regcmd(shape: &ConvShape, bufs: &ConvBuffers) -> Vec<RegCmd> {
    let input_channels_real_is_one = shape.input_channels == 1;
    assert!(
        !(input_channels_real_is_one && shape.output_channels > 1),
        "build_conv_regcmd: input_channels==1 && output_channels>1 needs \
         fill_task()'s \"wide atomic\" branch, not implemented here (no \
         current client's shape exercises it)"
    );

    // rkt_ml.h geometry constants.
    const CBUF_ENTRY_SIZE: u32 = 128; // CBUF_BANK_SIZE(32768) / CBUF_ENTRIES_PER_BANK(256)
    const CBUF_ENTRIES_PER_BANK: u32 = 256;
    const CBUF_BANKS: u32 = 12;
    const FEATURE_ATOMIC_SIZE: u32 = 16;
    const ATOMIC_K_SIZE: u32 = 16;

    // calc_entries_per_slice() (bpe=1, int8).
    let atomics_per_entry = CBUF_ENTRY_SIZE / FEATURE_ATOMIC_SIZE;
    let total_c_atomics = shape.input_channels.div_ceil(FEATURE_ATOMIC_SIZE);
    let last_c_atomics = total_c_atomics % atomics_per_entry;
    let int_c_entries = (total_c_atomics / atomics_per_entry) * shape.input_width;
    let frac_c_entries = if last_c_atomics == 3 {
        shape.input_width
    } else {
        (last_c_atomics * shape.input_width).div_ceil(atomics_per_entry)
    };
    let entries_per_slice = int_c_entries + frac_c_entries;

    // calc_input_banks().
    let input_banks = (entries_per_slice * shape.input_height).div_ceil(CBUF_ENTRIES_PER_BANK);
    assert!(
        input_banks < CBUF_BANKS,
        "build_conv_regcmd: shape needs {input_banks} input CBUF banks (only \
         {CBUF_BANKS} total) -- too big for the single-task path this \
         function implements (rkt_split_tasks()'s general multi-task \
         splitting branch isn't ported); shrink the operation or add that \
         support"
    );

    // rkt_split_tasks(): the single-task branch sets weight_banks to
    // CBUF_BANKS - input_banks directly -- NOT calc_weights_banks()'s own
    // result (that function's output only decides the reuse/no-reuse
    // branch elsewhere, never reaches CBUF_CON0 here).
    let weight_banks = CBUF_BANKS - input_banks;

    // fill_task(): channel/kernel counts round up to the hardware's
    // atomic granularity, independent of the op's logical shape.
    let task_input_channels = shape
        .input_channels
        .max(FEATURE_ATOMIC_SIZE)
        .next_multiple_of(FEATURE_ATOMIC_SIZE);
    let mut task_output_channels = shape.output_channels.max(32).next_multiple_of(32);
    if shape.depthwise {
        if shape.output_channels <= 32 {
            task_output_channels *= 2;
        }
        task_output_channels = task_output_channels.next_multiple_of(64);
    }
    let weights_kernels = if shape.depthwise {
        1
    } else {
        shape.output_channels.next_multiple_of(2)
    };

    // fill_task(): input_channels_real==1 && (output_channels_real>1 ||
    // addition) selects a "wide atomic" branch -- asserted unreachable
    // above, so this is always the plain else branch for every shape this
    // function actually gets called with.
    let line_stride = |w: u32| w * ATOMIC_K_SIZE;
    let input_line_stride = line_stride(shape.input_width) / 4;
    let input_surface_stride = input_line_stride * (shape.input_height / 4 - 1);
    let output_surface_stride =
        (line_stride(shape.output_width) * shape.output_height) / FEATURE_ATOMIC_SIZE;
    let surfaces_per_row =
        shape.output_width * shape.output_height * 2 * if shape.depthwise { 2 } else { 1 };

    // fill_task(): input_data_entries, three-way branch.
    let input_data_entries = if input_channels_real_is_one {
        shape.input_width * shape.input_height
    } else if shape.input_width == 40 && shape.input_channels == 40 {
        40
    } else {
        (shape.input_width * 2 * shape.input_channels.div_ceil(FEATURE_ATOMIC_SIZE)).div_ceil(8)
    };

    // rkt_regcmd.c: DPU_OUT_CVT_OFFSET/SCALE/SHIFT -- float-bits-based
    // fixed-point conversion (f32::to_bits() is bit-exact for C's fui()).
    let out_offset = shape.output_zero_point.wrapping_sub(0x80);
    let conv_scale = (shape.input_scale * shape.weights_scale) / shape.output_scale;
    let scale_bits = conv_scale.to_bits();
    let mut out_shift = 127 + 31 - 32 - (scale_bits >> 23) + 16;
    if shape.truncate_bits > 0 {
        out_shift -= 1;
    }
    let mut out_scale = ((scale_bits >> 9) & 0x7fff) + 1;
    if out_scale < (1 << 14) {
        out_scale |= 1 << 14;
    }

    let mut cmds: Vec<RegCmd> = Vec::new();

    // ========================================================================
    // CNA
    // ========================================================================

    let mut cbuf_con0_builder = Register::<CnaCbufCon0>::new();
    cbuf_con0_builder
        .weight_bank(Bits::new(weight_banks))
        .data_bank(Bits::new(input_banks));
    cmds.push(cbuf_con0_builder.build()); // written again after WEIGHT_SIZE2 below -- Mesa emits it twice

    cmds.push(zero::<CnaDcompRegnum>());
    cmds.push(zero::<CnaDcompCtrl>());

    let mut conv_con1_builder = Register::<CnaConvCon1>::new();
    if input_channels_real_is_one {
        conv_con1_builder
            .nonalign_dma(Bits::new(1))
            .group_line_off(Bits::new(1))
            .argb_in(Bits::new(8));
    }
    if shape.depthwise {
        conv_con1_builder.conv_mode(Bits::new(3));
    }
    cmds.push(conv_con1_builder.build());

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
    cmds.push(conv_con1_builder.build()); // second write, same value

    cmds.push(
        Register::<CnaConvCon2>::new()
            .feature_grains(Bits::new(50 + shape.stride + 1)) // Mesa's own comment: "Magic"
            .build(),
    );
    cmds.push(
        Register::<CnaConvCon3>::new()
            .conv_x_stride(Bits::new(shape.stride))
            .conv_y_stride(Bits::new(shape.stride))
            .build(),
    );

    cmds.push(
        Register::<CnaDataSize0>::new()
            .datain_width(Bits::new(shape.input_width))
            .datain_height(Bits::new(shape.input_height))
            .build(),
    );
    cmds.push(
        Register::<CnaDataSize1>::new()
            .datain_channel_real(Bits::new(shape.input_channels - 1))
            .datain_channel(Bits::new(task_input_channels))
            .build(),
    );
    cmds.push(
        Register::<CnaDataSize2>::new()
            .dataout_width(Bits::new(shape.output_width))
            .build(),
    );
    cmds.push(
        Register::<CnaDataSize3>::new()
            .dataout_atomics(Bits::new(shape.output_width * shape.output_height))
            .build(),
    );

    cmds.push(
        Register::<CnaWeightSize0>::new()
            .weight_bytes(Bits::new(
                shape.weights_width * shape.weights_height * task_input_channels * weights_kernels,
            ))
            .build(),
    );
    cmds.push(
        Register::<CnaWeightSize1>::new()
            .weight_bytes_per_kernel(Bits::new(
                shape.weights_width * shape.weights_height * task_input_channels,
            ))
            .build(),
    );
    cmds.push(
        Register::<CnaWeightSize2>::new()
            .weight_width(Bits::new(shape.weights_width))
            .weight_height(Bits::new(shape.weights_height))
            .weight_kernels(Bits::new(weights_kernels))
            .build(),
    );

    cmds.push(cbuf_con0_builder.build());

    cmds.push(
        Register::<CnaCbufCon1>::new()
            .data_entries(Bits::new(input_data_entries))
            .build(),
    );

    if input_channels_real_is_one {
        const CVT_TRUNCATE: u32 = 14;
        const CVT_SCALE: u32 = 16384;
        const CVT_OFFSET: u32 = 65408;
        cmds.push(
            Register::<CnaCvtCon0>::new()
                .cvt_truncate_0(Bits::new(CVT_TRUNCATE))
                .cvt_truncate_1(Bits::new(CVT_TRUNCATE))
                .cvt_truncate_2(Bits::new(CVT_TRUNCATE))
                .cvt_truncate_3(Bits::new(CVT_TRUNCATE))
                .build(),
        );
        cmds.push(
            Register::<CnaCvtCon1>::new()
                .cvt_scale0(Bits::new(CVT_SCALE))
                .cvt_offset0(Bits::new(CVT_OFFSET))
                .build(),
        );
        cmds.push(
            Register::<CnaCvtCon2>::new()
                .cvt_scale1(Bits::new(CVT_SCALE))
                .cvt_offset1(Bits::new(CVT_OFFSET))
                .build(),
        );
        cmds.push(
            Register::<CnaCvtCon3>::new()
                .cvt_scale2(Bits::new(CVT_SCALE))
                .cvt_offset2(Bits::new(CVT_OFFSET))
                .build(),
        );
        cmds.push(
            Register::<CnaCvtCon4>::new()
                .cvt_scale3(Bits::new(CVT_SCALE))
                .cvt_offset3(Bits::new(CVT_OFFSET))
                .build(),
        );
    } else {
        cmds.push(
            Register::<CnaCvtCon0>::new()
                .data_sign(Bits::new(1))
                .cvt_type(Bits::new(1))
                .cvt_bypass(Bits::new(1))
                .build(),
        );
        cmds.push(
            Register::<CnaCvtCon1>::new()
                .cvt_scale0(Bits::new(1))
                .build(),
        );
        cmds.push(
            Register::<CnaCvtCon2>::new()
                .cvt_scale1(Bits::new(1))
                .build(),
        );
        cmds.push(
            Register::<CnaCvtCon3>::new()
                .cvt_scale2(Bits::new(1))
                .build(),
        );
        cmds.push(
            Register::<CnaCvtCon4>::new()
                .cvt_scale3(Bits::new(1))
                .build(),
        );
    }

    cmds.push(zero::<CnaFcCon0>());
    cmds.push(zero::<CnaFcCon1>());
    cmds.push(
        Register::<CnaPadCon0>::new()
            .pad_top(Bits::new(0))
            .pad_left(Bits::new(0))
            .build(),
    );
    cmds.push(
        Register::<CnaFeatureDataAddr>::new()
            .feature_base_addr(Bits::new(bufs.input_addr))
            .build(),
    );
    cmds.push(zero::<CnaFcCon2>());
    cmds.push(
        Register::<CnaDmaCon0>::new()
            .data_burst_len(Bits::new(15))
            .weight_burst_len(Bits::new(15))
            .build(),
    );
    cmds.push(
        Register::<CnaDmaCon1>::new()
            .line_stride(Bits::new(input_line_stride))
            .build(),
    );
    cmds.push(
        Register::<CnaDmaCon2>::new()
            .surf_stride(Bits::new(input_surface_stride))
            .build(),
    );

    cmds.push(
        Register::<CnaFcDataSize0>::new()
            .dma_width(Bits::new(shape.input_width))
            .dma_height(Bits::new(shape.input_height))
            .build(),
    );
    cmds.push(
        Register::<CnaFcDataSize1>::new()
            .dma_channel(Bits::new(task_input_channels))
            .build(),
    );

    cmds.push(zero::<CnaDcompCtrl>());
    cmds.push(zero::<CnaDcompRegnum>());
    cmds.push(
        Register::<CnaDcompAddr0>::new()
            .decompress_addr0(Bits::new(bufs.weights_addr))
            .build(),
    );
    cmds.push(zero::<CnaDcompAmount0>());
    cmds.push(zero::<CnaDcompAmount1>());
    cmds.push(zero::<CnaDcompAmount2>());
    cmds.push(zero::<CnaDcompAmount3>());
    cmds.push(zero::<CnaDcompAmount4>());
    cmds.push(zero::<CnaDcompAmount5>());
    cmds.push(zero::<CnaDcompAmount6>());
    cmds.push(zero::<CnaDcompAmount7>());
    cmds.push(zero::<CnaDcompAmount8>());
    cmds.push(zero::<CnaDcompAmount9>());
    cmds.push(zero::<CnaDcompAmount10>());
    cmds.push(zero::<CnaDcompAmount11>());
    cmds.push(zero::<CnaDcompAmount12>());
    cmds.push(zero::<CnaDcompAmount13>());
    cmds.push(zero::<CnaDcompAmount14>());
    cmds.push(zero::<CnaDcompAmount15>());

    cmds.push(if input_channels_real_is_one {
        Register::<CnaCvtCon5>::new()
            .per_channel_cvt_en(Bits::new(65535))
            .build()
    } else {
        zero::<CnaCvtCon5>()
    });

    let mut pad_con1 = if shape.weights_width >= 3 && shape.input_zero_point == 0 {
        0xffff8080u32
    } else {
        shape.input_zero_point.wrapping_sub(0x80)
    };
    if shape.depthwise && shape.input_zero_point == 0x8b {
        pad_con1 = 0x0b0b;
    }
    cmds.push(
        Register::<CnaPadCon1>::new()
            .pad_value(Bits::new(pad_con1))
            .build(),
    );

    // ========================================================================
    // CORE
    // ========================================================================

    let mut misc_cfg_builder = Register::<CoreMiscCfg>::new();
    misc_cfg_builder.qd_en(Bits::new(1));
    if shape.depthwise {
        misc_cfg_builder.dw_en(Bits::new(1));
    }
    cmds.push(misc_cfg_builder.build());

    cmds.push(
        Register::<CoreDataoutSize0>::new()
            .dataout_width(Bits::new(shape.output_width - 1))
            .dataout_height(Bits::new(shape.output_height - 1))
            .build(),
    );
    cmds.push(
        Register::<CoreDataoutSize1>::new()
            .dataout_channel(Bits::new(task_output_channels - 1))
            .build(),
    );
    cmds.push(
        Register::<CoreClipTruncate>::new()
            .clip_truncate(Bits::new(shape.truncate_bits))
            .build(),
    );
    cmds.push(RegCmd::new(DOMAIN_CORE, 0x3030, 0)); // TRM-mandated reserved write, no REG_CORE_* name

    // ========================================================================
    // DPU
    // ========================================================================

    let mut feat_mode_builder = Register::<DpuFeatureModeCfg>::new();
    feat_mode_builder
        .burst_len(Bits::new(15))
        .output_mode(Bits::new(2));
    if shape.depthwise {
        feat_mode_builder.conv_mode(Bits::new(3));
    }
    cmds.push(feat_mode_builder.build());

    cmds.push(zero::<DpuDataFormat>());
    cmds.push(zero::<DpuOffsetPend>());
    cmds.push(
        Register::<DpuDstBaseAddr>::new()
            .dst_base_addr(Bits::new(bufs.output_addr))
            .build(),
    );
    cmds.push(
        Register::<DpuDstSurfStride>::new()
            .dst_surf_stride(Bits::new(output_surface_stride))
            .build(),
    );
    cmds.push(
        Register::<DpuDataCubeWidth>::new()
            .width(Bits::new(shape.output_width - 1))
            .build(),
    );
    cmds.push(
        Register::<DpuDataCubeHeight>::new()
            .height(Bits::new(shape.output_height - 1))
            .build(),
    );
    cmds.push(zero::<DpuDataCubeNotchAddr>());
    cmds.push(
        Register::<DpuDataCubeChannel>::new()
            .orig_channel(Bits::new(shape.output_channels - 1))
            .channel(Bits::new(task_output_channels - 1))
            .build(),
    );

    // BS (bias-subtract): Mesa never bypasses this outright -- always runs
    // the ALU (ALGO(2)|SRC(1)), only RELU/MUL bypassed. Needs a real (if
    // zero-filled) biases buffer, wired in via DPU_RDMA_RDMA_BS_BASE_ADDR
    // below.
    cmds.push(
        Register::<DpuBsCfg>::new()
            .bs_alu_algo(Bits::new(2))
            .bs_alu_src(Bits::new(1))
            .bs_relu_bypass(Bits::new(1))
            .bs_mul_bypass(Bits::new(1))
            .build(),
    );
    cmds.push(zero::<DpuBsAluCfg>());
    cmds.push(zero::<DpuBsMulCfg>());
    cmds.push(zero::<DpuBsReluxCmpValue>());
    cmds.push(if shape.depthwise {
        Register::<DpuBsOwCfg>::new()
            .size_e_0(Bits::new(3))
            .size_e_1(Bits::new(3))
            .size_e_2(Bits::new(3))
            .build()
    } else {
        Register::<DpuBsOwCfg>::new()
            .size_e_0(Bits::new(1))
            .size_e_1(Bits::new(1))
            .size_e_2(Bits::new(1))
            .build()
    });
    cmds.push(
        Register::<DpuBsOwOp>::new()
            .ow_op(Bits::new(0x80 - shape.weights_zero_point))
            .build(),
    );
    cmds.push(
        Register::<DpuWdmaSize0>::new()
            .channel_wdma(Bits::new(task_output_channels - 1))
            .build(),
    );
    cmds.push(
        Register::<DpuWdmaSize1>::new()
            .height_wdma(Bits::new(shape.output_height - 1))
            .width_wdma(Bits::new(shape.output_width - 1))
            .build(),
    );

    // BN (batchnorm): genuinely fully bypassed, all four bits.
    cmds.push(
        Register::<DpuBnCfg>::new()
            .bn_relu_bypass(Bits::new(1))
            .bn_mul_bypass(Bits::new(1))
            .bn_alu_bypass(Bits::new(1))
            .bn_bypass(Bits::new(1))
            .build(),
    );
    cmds.push(zero::<DpuBnAluCfg>());
    cmds.push(zero::<DpuBnMulCfg>());
    cmds.push(zero::<DpuBnReluxCmpValue>());

    // EW (elementwise): add_tensor == -1 branch (unconditional -- no
    // add_tensor support in this function), fully bypassed, all five bits.
    cmds.push(
        Register::<DpuEwCfg>::new()
            .ew_relu_bypass(Bits::new(1))
            .ew_op_cvt_bypass(Bits::new(1))
            .ew_lut_bypass(Bits::new(1))
            .ew_op_bypass(Bits::new(1))
            .ew_bypass(Bits::new(1))
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
            .out_cvt_scale(Bits::new(out_scale))
            .build(),
    );
    cmds.push(
        Register::<DpuOutCvtShift>::new()
            .out_cvt_shift(Bits::new(out_shift - 1))
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
            .surf_add(Bits::new(surfaces_per_row))
            .build(),
    );
    cmds.push(RegCmd::new(DOMAIN_DPU, 0x40c4, 0)); // TRM-mandated reserved write, no REG_DPU_* name

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

    // ========================================================================
    // DPU_RDMA
    // ========================================================================

    cmds.push(
        Register::<DpuRdmaDataCubeWidth>::new()
            .width(Bits::new(shape.output_width - 1))
            .build(),
    );
    cmds.push(
        Register::<DpuRdmaDataCubeHeight>::new()
            .height(Bits::new(shape.output_height - 1))
            .build(),
    );
    cmds.push(
        Register::<DpuRdmaDataCubeChannel>::new()
            .channel(Bits::new(task_output_channels - 1))
            .build(),
    );

    cmds.push(zero::<DpuRdmaSrcBaseAddr>()); // add_tensor == -1
    cmds.push(
        Register::<DpuRdmaBrdmaCfg>::new()
            .brdma_data_use(Bits::new(1))
            .build(),
    );
    cmds.push(
        Register::<DpuRdmaBsBaseAddr>::new()
            .bs_base_addr(Bits::new(bufs.bias_addr))
            .build(),
    );
    cmds.push(zero::<DpuRdmaNrdmaCfg>());
    cmds.push(zero::<DpuRdmaBnBaseAddr>());

    cmds.push(
        Register::<DpuRdmaErdmaCfg>::new()
            .erdma_disable(Bits::new(1))
            .build(),
    ); // add_tensor == -1
    cmds.push(zero::<DpuRdmaEwBaseAddr>());
    cmds.push(zero::<DpuRdmaEwSurfStride>());

    let mut rdma_feat_mode_builder = Register::<DpuRdmaFeatureModeCfg>::new();
    rdma_feat_mode_builder
        .burst_len(Bits::new(15))
        .mrdma_disable(Bits::new(1));
    if shape.depthwise {
        rdma_feat_mode_builder.conv_mode(Bits::new(3));
    }
    cmds.push(rdma_feat_mode_builder.build());
    cmds.push(zero::<DpuRdmaSrcDmaCfg>());
    cmds.push(zero::<DpuRdmaSurfNotch>()); // add_tensor == -1
    cmds.push(zero::<DpuRdmaPadCfg>());
    cmds.push(
        Register::<DpuRdmaWeight>::new()
            .e_weight(Bits::new(1))
            .n_weight(Bits::new(1))
            .b_weight(Bits::new(1))
            .m_weight(Bits::new(1))
            .build(),
    );
    cmds.push(zero::<DpuRdmaEwSurfNotch>()); // add_tensor == -1

    // ========================================================================
    // Kick sequence -- ported verbatim from Mesa's fill_first_regcmd() tail,
    // num_tasks == 1 branch. See rkt-basic.rs's top-of-file doc comment /
    // NOTES.md for why each of these four entries is shaped this way.
    // ========================================================================

    cmds.push(RegCmd::new_raw(0x0)); // PC_BASE_ADDRESS placeholder (single task)
    cmds.push(RegCmd::new(DOMAIN_PC, REG_PC_REGISTER_AMOUNTS, 0));
    cmds.push(RegCmd::new_raw(0x0041000000000000)); // TRM: required immediately before op_en
    cmds.push(RegCmd::new(0x81, REG_PC_OPERATION_ENABLE, unsafe {
        PC_OPERATION_ENABLE_RESERVED_0(14) | PC_OPERATION_ENABLE_OP_EN(1)
    }));

    if cmds.len() % 2 != 0 {
        cmds.push(RegCmd::new_raw(0x0));
    }

    cmds
}

//===========================================================================
// Pooling (standalone PPU, "flying mode" -- TRM Ch.36 Fig 36-6): PPU_RDMA
// reads the input straight from memory and feeds PPU directly, bypassing
// CNA/CORE/DPU entirely. There is NO Mesa/Teflon reference for this path
// (`rkt_ml.c` only ever implements convolution) -- every field below is
// derived from the PPU/PPU_RDMA register layout in `builders/ppu.rs` /
// `builders/ppu_rdma.rs` (bindgen'd from Mesa's own `registers.xml`, see
// builders.rs's DOMAIN_* comment) plus the TRM Ch.36 §4.6/§4.7 prose and
// `build_conv_regcmd`'s established conventions (N-1 encoding on every
// *_RDMA/CORE/DPU cube dimension, the same four-entry PC kick tail). NONE
// of the following has been hardware-validated yet -- see the UNCONFIRMED
// markers below and iree-rocket-hal/tests/pooling_hw.rs's doc comment for
// the sweep tests needed before trusting this in production:
//
// - `PoolingMethod`'s bit encoding (0/1/2 for max/min/avg) -- the TRM
//   prose lists "avg/max/min" but that's not necessarily encoding order.
// - PPU_RDMA's `src_line_stride`/`src_surf_stride` and PPU's
//   `dst_surf_stride`/`misc_ctrl.surf_len` formulas -- derived by analogy
//   to CNA's input-side and DPU's output-side stride math in
//   `build_conv_regcmd`, not independently confirmed for PPU/PPU_RDMA.
// - Whether firing the same all-blocks op_en kick as `build_conv_regcmd`
//   (which also asserts CNA/CORE/DPU/DPU_RDMA's op_en, not just PPU/
//   PPU_RDMA's) is actually safe when those blocks have no configured
//   task -- vs. needing a pooling-only op_en combination instead.
//===========================================================================

/// UNCONFIRMED bit encoding -- see module doc comment above. Best guess
/// only; must be swept against real hardware (feed a kernel window with a
/// known max/min/mean-distinguishing pattern and see which value of
/// `PPU_OPERATION_MODE_CFG_POOLING_METHOD` actually produces which result)
/// before trusting this mapping.
#[derive(Clone, Copy)]
pub enum PoolingMethod {
    Max,
    Min,
    Avg,
}

impl PoolingMethod {
    fn bits(self) -> u32 {
        match self {
            PoolingMethod::Max => 0,
            PoolingMethod::Min => 1,
            PoolingMethod::Avg => 2,
        }
    }
}

/// Logical shape of a single standalone pooling operation. Single-task
/// only, no CBUF-budget splitting to worry about (PPU has no CBUF -- that
/// concern is CNA/CORE-specific), no `index_en` output wiring yet (that
/// needs a second output buffer for argmax/argmin positions, not plumbed
/// here).
pub struct PoolingShape {
    pub input_width: u32,
    pub input_height: u32,
    pub input_channels: u32,
    pub output_width: u32,
    pub output_height: u32,
    pub output_channels: u32,
    pub kernel_width: u32,
    pub kernel_height: u32,
    pub stride_x: u32,
    pub stride_y: u32,
    pub method: PoolingMethod,
    pub pad_left: u32,
    pub pad_top: u32,
    pub pad_right: u32,
    pub pad_bottom: u32,
    /// Fill value for padded taps (e.g. -inf-ish for max, 0 for avg --
    /// caller's responsibility to pick something sane for `method`).
    pub pad_value: u32,
}

/// DMA addresses for the two buffers a standalone pooling op needs.
pub struct PoolingBuffers {
    pub input_addr: u32,
    /// NOTE: `PPU_DST_BASE_ADDR` is a genuinely 28-bit hardware field
    /// (unlike every other block's 32-bit `*_BASE_ADDR` registers -- see
    /// `PpuDstBaseAddr::dst_base_addr`'s `Bits<28>` parameter, confirmed
    /// against the bindgen'd header, not a guess). A `CREATE_BO`
    /// allocation whose `dma_address` lands >= 0x1000_0000 (256MiB) into
    /// DMA-visible space cannot be addressed by PPU's output stage as-is.
    /// `build_pooling_regcmd` asserts this explicitly (a clear panic
    /// message) rather than letting `Bits::<28>::new` fail with its
    /// generic "value exceeds designated bit width" message. Whether the
    /// real fix is "PPU output must live in the first 256MiB" or "this
    /// field is actually a shifted/word address with more real range"
    /// is unconfirmed -- needs checking once a test allocates enough
    /// buffers to push an output address past the boundary.
    pub output_addr: u32,
}

pub fn build_pooling_regcmd(shape: &PoolingShape, bufs: &PoolingBuffers) -> Vec<RegCmd> {
    const PPU_DST_BASE_ADDR_BITS: u32 = 28;
    assert!(
        bufs.output_addr < (1u32 << PPU_DST_BASE_ADDR_BITS),
        "build_pooling_regcmd: output_addr {:#x} does not fit PPU_DST_BASE_ADDR's \
         {PPU_DST_BASE_ADDR_BITS}-bit field (see PoolingBuffers::output_addr's doc \
         comment) -- this buffer's physical/DMA address lands past the 256MiB PPU \
         can currently address for output",
        bufs.output_addr
    );

    const ATOMIC_K_SIZE: u32 = 16;
    const FEATURE_ATOMIC_SIZE: u32 = 16;

    // UNCONFIRMED, derived by analogy to CNA's input-side stride math in
    // build_conv_regcmd -- see module doc comment.
    let src_line_stride = shape.input_width * ATOMIC_K_SIZE;
    let src_surf_stride = src_line_stride * shape.input_height;

    // UNCONFIRMED, derived by analogy to DPU's output-side stride math in
    // build_conv_regcmd -- see module doc comment.
    let dst_surf_stride =
        (shape.output_width * ATOMIC_K_SIZE * shape.output_height) / FEATURE_ATOMIC_SIZE;
    let surf_len = shape.output_width * shape.output_height;

    // Average-pooling's divide-as-multiply trick (TRM §4.6): precomputed
    // reciprocal of the kernel dimension, x2^16. Always computed (not just
    // under Avg) -- build_conv_regcmd's own precedent is to always fill
    // every register regardless of which branch is logically active.
    let recip_kernel_width = ((1u64 << 16) / shape.kernel_width as u64) as u32;
    let recip_kernel_height = ((1u64 << 16) / shape.kernel_height as u64) as u32;

    let mut cmds: Vec<RegCmd> = Vec::new();

    // ========================================================================
    // Ping-pong pointers -- same pattern/values as build_conv_regcmd's
    // DPU/DPU_RDMA pair, applied to the two blocks this op actually uses.
    // ========================================================================

    cmds.push(
        Register::<PpuSPointer>::new()
            .pointer_pp_mode(Bits::new(1))
            .executer_pp_en(Bits::new(1))
            .pointer_pp_en(Bits::new(1))
            .build(),
    );
    cmds.push(
        Register::<PpuRdmaSPointer>::new()
            .pointer_pp_mode(Bits::new(1))
            .executer_pp_en(Bits::new(1))
            .pointer_pp_en(Bits::new(1))
            .build(),
    );

    // ========================================================================
    // PPU_RDMA -- standalone read side, feeds PPU directly from memory.
    // ========================================================================

    cmds.push(
        Register::<PpuRdmaCubeInWidth>::new()
            .cube_in_width(Bits::new(shape.input_width - 1))
            .build(),
    );
    cmds.push(
        Register::<PpuRdmaCubeInHeight>::new()
            .cube_in_height(Bits::new(shape.input_height - 1))
            .build(),
    );
    cmds.push(
        Register::<PpuRdmaCubeInChannel>::new()
            .cube_in_channel(Bits::new(shape.input_channels - 1))
            .build(),
    );
    cmds.push(
        Register::<PpuRdmaSrcBaseAddr>::new()
            .src_base_addr(Bits::new(bufs.input_addr))
            .build(),
    );
    cmds.push(
        Register::<PpuRdmaSrcLineStride>::new()
            .src_line_stride(Bits::new(src_line_stride))
            .build(),
    );
    cmds.push(
        Register::<PpuRdmaSrcSurfStride>::new()
            .src_surf_stride(Bits::new(src_surf_stride))
            .build(),
    );
    cmds.push(zero::<PpuRdmaDataFormat>()); // in_precision = 0 (int8)

    // ========================================================================
    // PPU
    // ========================================================================

    cmds.push(
        Register::<PpuDataCubeInWidth>::new()
            .cube_in_width(Bits::new(shape.input_width - 1))
            .build(),
    );
    cmds.push(
        Register::<PpuDataCubeInHeight>::new()
            .cube_in_height(Bits::new(shape.input_height - 1))
            .build(),
    );
    cmds.push(
        Register::<PpuDataCubeInChannel>::new()
            .cube_in_channel(Bits::new(shape.input_channels - 1))
            .build(),
    );
    cmds.push(
        Register::<PpuDataCubeOutWidth>::new()
            .cube_out_width(Bits::new(shape.output_width - 1))
            .build(),
    );
    cmds.push(
        Register::<PpuDataCubeOutHeight>::new()
            .cube_out_height(Bits::new(shape.output_height - 1))
            .build(),
    );
    cmds.push(
        Register::<PpuDataCubeOutChannel>::new()
            .cube_out_channel(Bits::new(shape.output_channels - 1))
            .build(),
    );

    cmds.push(
        Register::<PpuOperationModeCfg>::new()
            .pooling_method(Bits::new(shape.method.bits()))
            .flying_mode(Bits::new(1)) // standalone via PPU_RDMA, not pipelined after DPU
            .index_en(Bits::new(0)) // no argmax/argmin output wiring yet
            .use_cnt(Bits::new(0))
            .notch_addr(Bits::new(0))
            .build(),
    );
    cmds.push(
        Register::<PpuPoolingKernelCfg>::new()
            .kernel_width(Bits::new(shape.kernel_width - 1))
            .kernel_height(Bits::new(shape.kernel_height - 1))
            .kernel_stride_width(Bits::new(shape.stride_x - 1))
            .kernel_stride_height(Bits::new(shape.stride_y - 1))
            .build(),
    );
    cmds.push(
        Register::<PpuRecipKernelWidth>::new()
            .recip_kernel_width(Bits::new(recip_kernel_width))
            .build(),
    );
    cmds.push(
        Register::<PpuRecipKernelHeight>::new()
            .recip_kernel_height(Bits::new(recip_kernel_height))
            .build(),
    );
    cmds.push(
        Register::<PpuPoolingPaddingCfg>::new()
            .pad_left(Bits::new(shape.pad_left))
            .pad_top(Bits::new(shape.pad_top))
            .pad_right(Bits::new(shape.pad_right))
            .pad_bottom(Bits::new(shape.pad_bottom))
            .build(),
    );
    cmds.push(
        Register::<PpuPaddingValue1Cfg>::new()
            .pad_value_0(Bits::new(shape.pad_value))
            .build(),
    );
    cmds.push(zero::<PpuPaddingValue2Cfg>());

    cmds.push(
        Register::<PpuDstBaseAddr>::new()
            .dst_base_addr(Bits::new(bufs.output_addr))
            .build(),
    );
    cmds.push(
        Register::<PpuDstSurfStride>::new()
            .dst_surf_stride(Bits::new(dst_surf_stride))
            .build(),
    );
    cmds.push(zero::<PpuDataFormat>()); // proc_precision=0 (int8), dpu_flyin=0 (standalone)
    cmds.push(
        Register::<PpuMiscCtrl>::new()
            .burst_len(Bits::new(15))
            .nonalign(Bits::new(0))
            .mc_surf_out(Bits::new(0))
            .surf_len(Bits::new(surf_len))
            .build(),
    );

    // ========================================================================
    // Kick sequence -- identical tail to build_conv_regcmd's (single task,
    // all-blocks op_en broadcast). See that function's own comment; whether
    // this is safe to fire unconditionally for CNA/CORE/DPU/DPU_RDMA when
    // this op never configured them is one of the open questions flagged
    // in this section's module doc comment.
    // ========================================================================

    cmds.push(RegCmd::new_raw(0x0));
    cmds.push(RegCmd::new(DOMAIN_PC, REG_PC_REGISTER_AMOUNTS, 0));
    cmds.push(RegCmd::new_raw(0x0041000000000000));
    cmds.push(RegCmd::new(0x81, REG_PC_OPERATION_ENABLE, unsafe {
        PC_OPERATION_ENABLE_RESERVED_0(14) | PC_OPERATION_ENABLE_OP_EN(1)
    }));

    if cmds.len() % 2 != 0 {
        cmds.push(RegCmd::new_raw(0x0));
    }

    cmds
}
