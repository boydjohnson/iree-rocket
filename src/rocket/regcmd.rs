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
    registers::{REG_PC_OPERATION_ENABLE, REG_PC_REGISTER_AMOUNTS},
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

/// Shared CNA->CORE->DPU->DPU_RDMA emission, factored out of
/// `build_conv_regcmd` so `build_conv_then_pooling_regcmd` can reuse the
/// exact same hardware-validated sequence instead of duplicating it. Pure
/// extraction -- no behavior change versus the original single function
/// (verified by inspection: every `cmds.push` below is unchanged from
/// before the split, in the same order).
///
/// `dpu_output_mode` is the one new degree of freedom the pooling caller
/// needs: `DPU_FEATURE_MODE_CFG.output_mode` is bit0->PPU (on-chip,
/// pipelined), bit1->outside/memory (can be both). `build_conv_regcmd`
/// always passes `2` (outside only, its original hardcoded behavior,
/// unchanged). When bit1 is clear, `DPU_DST_BASE_ADDR`/`DPU_DST_SURF_
/// STRIDE` are written as 0 rather than `bufs.output_addr`-derived values
/// -- UNCONFIRMED whether the hardware truly ignores them in that case or
/// just doesn't care what's there, but 0 matches this codebase's existing
/// "safe default when unused" convention.
fn build_conv_cna_core_dpu_dpu_rdma(
    shape: &ConvShape,
    bufs: &ConvBuffers,
    dpu_output_mode: u32,
) -> Vec<RegCmd> {
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
        .output_mode(Bits::new(dpu_output_mode));
    if shape.depthwise {
        feat_mode_builder.conv_mode(Bits::new(3));
    }
    cmds.push(feat_mode_builder.build());

    cmds.push(zero::<DpuDataFormat>());
    cmds.push(zero::<DpuOffsetPend>());
    // bit1 clear (not writing "outside"/memory) -> 0, see this function's
    // own doc comment.
    let dpu_writes_to_memory = dpu_output_mode & 0b10 != 0;
    cmds.push(
        Register::<DpuDstBaseAddr>::new()
            .dst_base_addr(Bits::new(if dpu_writes_to_memory {
                bufs.output_addr
            } else {
                0
            }))
            .build(),
    );
    cmds.push(
        Register::<DpuDstSurfStride>::new()
            .dst_surf_stride(Bits::new(if dpu_writes_to_memory {
                output_surface_stride
            } else {
                0
            }))
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

    cmds
}

// PC_OPERATION_ENABLE's bitmask, one bit per engine block, confirmed
// against real rknn-toolkit2-compiled regcmd programs (both conv.rknn and
// the pooling.rknn capture in rknpu-spelunking/NOTES.md's "Decoding a real
// regcmd program for a pooling-only op" section): bit = log2(mesa_target)
// - 9, where mesa_target is Mesa registers.xml's `target` enum value for
// that block (PC=0x100 CNA=0x200 CORE=0x800 DPU=0x1000 DPU_RDMA=0x2000
// PPU=0x4000 PPU_RDMA=0x8000). Bit 1 has no corresponding block (the gap
// between CNA=0x200 and CORE=0x800) and is never set in any real capture.
//
// conv.rknn's single kick is always `KICK_CNA | KICK_CORE | KICK_DPU |
// KICK_DPU_RDMA` (0x1d). The pooling.rknn capture fires *two* kicks per
// tile: a bypass CNA/CORE/DPU stage (`0x0d`, no DPU_RDMA) followed
// separately by `KICK_PPU | KICK_PPU_RDMA` (0x60) -- notably with bit 0
// (`PC_OPERATION_ENABLE_OP_EN` in rkt_registers.h's naming) clear, which
// is what revealed that bit isn't a generic mandatory "go" flag -- it's
// just CNA's own enable bit, like every other bit here.
const KICK_CNA: u32 = 1 << 0;
const KICK_CORE: u32 = 1 << 2;
const KICK_DPU: u32 = 1 << 3;
const KICK_DPU_RDMA: u32 = 1 << 4;
const KICK_PPU: u32 = 1 << 5;
const KICK_PPU_RDMA: u32 = 1 << 6;

/// Appends the standard single-task "kick" tail (ported verbatim from
/// Mesa's fill_first_regcmd(), num_tasks == 1 branch -- see rkt-basic.rs's
/// top-of-file doc comment / NOTES.md) and pads to an even length. Shared
/// by every regcmd builder in this module -- every one of them ends a
/// single task the same way, differing only in which blocks that task's
/// kick should actually enable (see `KICK_*` above -- pass exactly the
/// bits for the blocks this task configured, not a fixed value).
fn push_kick(cmds: &mut Vec<RegCmd>, enable_mask: u32) {
    cmds.push(RegCmd::new_raw(0x0)); // PC_BASE_ADDRESS placeholder (single task)
    cmds.push(RegCmd::new(DOMAIN_PC, REG_PC_REGISTER_AMOUNTS, 0));
    cmds.push(RegCmd::new_raw(0x0041000000000000)); // TRM: required immediately before op_en
    cmds.push(RegCmd::new(0x81, REG_PC_OPERATION_ENABLE, enable_mask));

    if cmds.len() % 2 != 0 {
        cmds.push(RegCmd::new_raw(0x0));
    }
}

pub fn build_conv_regcmd(shape: &ConvShape, bufs: &ConvBuffers) -> Vec<RegCmd> {
    let mut cmds = build_conv_cna_core_dpu_dpu_rdma(shape, bufs, 2); // 2 = outside/memory only, unchanged original behavior
    push_kick(&mut cmds, KICK_CNA | KICK_CORE | KICK_DPU | KICK_DPU_RDMA);
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
// - RESOLVED (was open when this task hung real hardware): the kick used
//   to fire `build_conv_regcmd`'s fixed CNA/CORE/DPU/DPU_RDMA bitmask
//   unconditionally, which never actually enabled PPU/PPU_RDMA for this
//   task -- a likely root cause of the hang, independent of the stride
//   formulas above. Now kicks `KICK_PPU | KICK_PPU_RDMA` only, confirmed
//   against a real rknn-toolkit2-compiled pooling.rknn capture (see
//   NOTES.md in rknpu-spelunking). Not yet re-validated on hardware with
//   this fix.
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
    /// generic "value exceeds designated bit width" message.
    ///
    /// UPDATE (post first hardware run): this was wrong -- writing the raw
    /// byte address here produced a job that completed (no hang, PREP_BO
    /// returned) but left the *entire* output buffer at zero (see
    /// pooling_hw.rs's `pooling_dump_full_output_buffer`, added
    /// specifically to check this). The 28-bit width is the same
    /// bits[31:4] convention TRM Ch.36 documents explicitly for
    /// `pc_base_address` ("bits 31:4 regcmd DMA address") -- a 32-bit byte
    /// address shifted right by 4 (assuming 16-byte alignment) always fits
    /// exactly in 28 bits with zero waste, unlike a genuine "only 256MiB
    /// addressable" hardware limit (which would be an odd, wasteful design
    /// choice and doesn't match anything else in this chapter). `DPU_DST_
    /// BASE_ADDR` being 32-bit and used unshifted in the hardware-
    /// validated `build_conv_regcmd` shows the shift isn't universal --
    /// only registers too narrow to hold a full byte address seem to use
    /// it, and PPU_DST_BASE_ADDR is the only such *_BASE_ADDR field found
    /// so far. `build_pooling_regcmd` now asserts 16-byte alignment
    /// (trivially satisfied by any page-aligned CREATE_BO allocation) and
    /// shifts right by 4 instead. NOT YET RE-CONFIRMED ON HARDWARE.
    pub output_addr: u32,
}

pub fn build_pooling_regcmd(shape: &PoolingShape, bufs: &PoolingBuffers) -> Vec<RegCmd> {
    assert!(
        bufs.output_addr % 16 == 0,
        "build_pooling_regcmd: output_addr {:#x} is not 16-byte aligned -- \
         PPU_DST_BASE_ADDR is written as address >> 4 (see PoolingBuffers::output_addr's \
         doc comment), which silently drops any non-zero low 4 bits instead of failing \
         loudly, so this must be checked explicitly",
        bufs.output_addr
    );
    let dst_base_addr_shifted = bufs.output_addr >> 4;

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
            .dst_base_addr(Bits::new(dst_base_addr_shifted))
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
    // Kick sequence. Previously reused build_conv_regcmd's fixed CNA/CORE/
    // DPU/DPU_RDMA kick value verbatim even though this task configures
    // only PPU/PPU_RDMA -- that kick never set PPU's or PPU_RDMA's enable
    // bit at all while claiming to enable four completely unconfigured
    // blocks, which is what hung real hardware (job timeout + IOMMU
    // stall-request timeout). Confirmed via rknpu-spelunking's real
    // rknn-toolkit2-compiled pooling.rknn capture (see NOTES.md "Decoding a
    // real regcmd program for a pooling-only op"): a standalone-flying PPU
    // task there is kicked with exactly `KICK_PPU | KICK_PPU_RDMA` (0x60),
    // its own separate task from any CNA/CORE/DPU work. NOT YET
    // RE-VALIDATED ON HARDWARE with this fix.
    // ========================================================================

    push_kick(&mut cmds, KICK_PPU | KICK_PPU_RDMA);

    cmds
}

//===========================================================================
// Pooling, pipelined directly after a real DPU stage (TRM Ch.36 Fig 36-4/
// 36-5's "do some pipeline surface ops" step, NOT the standalone/flying
// Fig 36-6 path `build_pooling_regcmd` implements above).
//
// Chosen after the standalone path hung real hardware (NPU job timeout +
// IOMMU "Enable stall request timed out"). Re-reading TRM Fig 36-6 itself
// ruled out one leading hypothesis for that hang -- the official standalone
// flow never configures CNA/CORE/DPU/DPU_RDMA at all, so leaving them
// zeroed (as `build_pooling_regcmd` already does) is not the bug. That left
// PPU_RDMA's own fetch-stride math (also unconfirmed) or the borrowed
// all-blocks "kick" as the remaining suspects -- the kick one is now
// confirmed and fixed (see `build_pooling_regcmd`'s kick-sequence comment
// and the `KICK_*` constants above `push_kick`); PPU_RDMA's stride math
// remains unconfirmed since this pipelined path doesn't use PPU_RDMA at
// all (see below).
//
// This path sidesteps the still-open stride-math question rather than
// resolving it: it runs a real, hardware-validated CNA->CORE->DPU
// convolution (the exact same sequence `build_conv_regcmd` already proves
// on hardware, reused via `build_conv_cna_core_dpu_dpu_rdma`) and routes
// DPU's output on-chip straight into PPU (`DPU_FEATURE_MODE_CFG.output_mode`
// bit0, `PPU_DATA_FORMAT.dpu_flyin`) instead of through PPU_RDMA. No
// PPU_RDMA fetch is used at all, so the kick is `KICK_CNA | KICK_CORE |
// KICK_DPU | KICK_DPU_RDMA | KICK_PPU` -- every block this task actually
// configures, PPU_RDMA correctly left both unconfigured and unkicked.
//
// Still NOT hardware-validated as of writing -- see tests/ for the
// hardware-in-the-loop test this needs before trusting it.
//===========================================================================

/// Pooling-specific parameters for the pipelined path. `conv_shape`'s own
/// `output_width`/`output_height`/`output_channels` become PPU's INPUT
/// shape (DPU's output, routed on-chip) -- this struct only carries
/// pooling's own output shape and kernel/pad/method parameters.
pub struct PipelinedPoolingShape {
    pub kernel_width: u32,
    pub kernel_height: u32,
    pub stride_x: u32,
    pub stride_y: u32,
    pub method: PoolingMethod,
    pub pad_left: u32,
    pub pad_top: u32,
    pub pad_right: u32,
    pub pad_bottom: u32,
    pub pad_value: u32,
    pub output_width: u32,
    pub output_height: u32,
    pub output_channels: u32,
}

/// DMA addresses for a conv-then-pipelined-pooling op. Unlike
/// `ConvBuffers`, there is no separate "conv output" address: DPU's output
/// is routed on-chip directly into PPU (`output_mode` bit0 only, bit1/
/// "outside" left clear), so it never needs a memory round-trip. Only the
/// final pooled result needs a real buffer.
pub struct ConvThenPoolingBuffers {
    pub input_addr: u32,
    pub weights_addr: u32,
    pub bias_addr: u32,
    pub output_addr: u32,
}

pub fn build_conv_then_pooling_regcmd(
    conv_shape: &ConvShape,
    pooling: &PipelinedPoolingShape,
    bufs: &ConvThenPoolingBuffers,
) -> Vec<RegCmd> {
    assert!(
        bufs.output_addr % 16 == 0,
        "build_conv_then_pooling_regcmd: output_addr {:#x} is not 16-byte aligned -- \
         PPU_DST_BASE_ADDR is written as address >> 4 (see build_pooling_regcmd's \
         PoolingBuffers::output_addr doc comment for why)",
        bufs.output_addr
    );

    const FEATURE_ATOMIC_SIZE: u32 = 16;
    const ATOMIC_K_SIZE: u32 = 16;

    // dpu_output_mode = 1: bit0 only ("to PPU", on-chip) -- bit1
    // ("outside"/memory) deliberately left clear, see module doc comment.
    // The conv-stage buffers (input/weights/bias) are real; output_addr
    // there is irrelevant (never written) since bit1 is clear.
    let mut cmds = build_conv_cna_core_dpu_dpu_rdma(
        conv_shape,
        &ConvBuffers {
            input_addr: bufs.input_addr,
            weights_addr: bufs.weights_addr,
            bias_addr: bufs.bias_addr,
            output_addr: 0,
        },
        1,
    );

    // ========================================================================
    // PPU -- pipelined directly after DPU (dpu_flyin=1), no PPU_RDMA
    // involved. TRM Fig 36-4/36-5's "do some pipeline surface ops" step.
    // ========================================================================

    cmds.push(
        Register::<PpuSPointer>::new()
            .pointer_pp_mode(Bits::new(1))
            .executer_pp_en(Bits::new(1))
            .pointer_pp_en(Bits::new(1))
            .build(),
    );

    cmds.push(
        Register::<PpuDataCubeInWidth>::new()
            .cube_in_width(Bits::new(conv_shape.output_width - 1))
            .build(),
    );
    cmds.push(
        Register::<PpuDataCubeInHeight>::new()
            .cube_in_height(Bits::new(conv_shape.output_height - 1))
            .build(),
    );
    cmds.push(
        Register::<PpuDataCubeInChannel>::new()
            .cube_in_channel(Bits::new(conv_shape.output_channels - 1))
            .build(),
    );
    cmds.push(
        Register::<PpuDataCubeOutWidth>::new()
            .cube_out_width(Bits::new(pooling.output_width - 1))
            .build(),
    );
    cmds.push(
        Register::<PpuDataCubeOutHeight>::new()
            .cube_out_height(Bits::new(pooling.output_height - 1))
            .build(),
    );
    cmds.push(
        Register::<PpuDataCubeOutChannel>::new()
            .cube_out_channel(Bits::new(pooling.output_channels - 1))
            .build(),
    );

    cmds.push(
        Register::<PpuOperationModeCfg>::new()
            .pooling_method(Bits::new(pooling.method.bits()))
            .flying_mode(Bits::new(0)) // pipelined after DPU, not standalone
            .index_en(Bits::new(0))
            .use_cnt(Bits::new(0))
            .notch_addr(Bits::new(0))
            .build(),
    );
    cmds.push(
        Register::<PpuPoolingKernelCfg>::new()
            .kernel_width(Bits::new(pooling.kernel_width - 1))
            .kernel_height(Bits::new(pooling.kernel_height - 1))
            .kernel_stride_width(Bits::new(pooling.stride_x - 1))
            .kernel_stride_height(Bits::new(pooling.stride_y - 1))
            .build(),
    );

    let recip_kernel_width = ((1u64 << 16) / pooling.kernel_width as u64) as u32;
    let recip_kernel_height = ((1u64 << 16) / pooling.kernel_height as u64) as u32;
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
            .pad_left(Bits::new(pooling.pad_left))
            .pad_top(Bits::new(pooling.pad_top))
            .pad_right(Bits::new(pooling.pad_right))
            .pad_bottom(Bits::new(pooling.pad_bottom))
            .build(),
    );
    cmds.push(
        Register::<PpuPaddingValue1Cfg>::new()
            .pad_value_0(Bits::new(pooling.pad_value))
            .build(),
    );
    cmds.push(zero::<PpuPaddingValue2Cfg>());

    // Tested both this (>>4, by analogy to PC_BASE_ADDRESS's documented
    // bits[31:4] convention) and the raw address directly against real
    // hardware with a small (0x3000) buffer address -- both produced
    // identical zero-effect results, ruling out address encoding as the
    // actual blocker. Keeping the shifted form since it's still the more
    // principled default absent evidence either way.
    cmds.push(
        Register::<PpuDstBaseAddr>::new()
            .dst_base_addr(Bits::new(bufs.output_addr >> 4))
            .build(),
    );
    let ppu_dst_surf_stride =
        (pooling.output_width * ATOMIC_K_SIZE * pooling.output_height) / FEATURE_ATOMIC_SIZE;
    cmds.push(
        Register::<PpuDstSurfStride>::new()
            .dst_surf_stride(Bits::new(ppu_dst_surf_stride))
            .build(),
    );
    cmds.push(
        Register::<PpuDataFormat>::new()
            .dpu_flyin(Bits::new(1)) // source is DPU's on-chip output, not PPU_RDMA
            .build(),
    );
    let surf_len = pooling.output_width * pooling.output_height;
    cmds.push(
        Register::<PpuMiscCtrl>::new()
            .burst_len(Bits::new(15))
            .nonalign(Bits::new(0))
            .mc_surf_out(Bits::new(0))
            .surf_len(Bits::new(surf_len))
            .build(),
    );

    // Previously added an explicit PPU_OPERATION_ENABLE write here (PPU's
    // own DOMAIN_PPU-tagged register) because the shared kick's fixed
    // CNA/CORE/DPU/DPU_RDMA bitmask never set PPU's enable bit, and there
    // was no evidence either mechanism actually enabled PPU. There's now
    // real evidence: a real rknn-toolkit2-compiled pooling.rknn capture
    // (see NOTES.md in rknpu-spelunking, "Decoding a real regcmd program
    // for a pooling-only op") never writes any per-block *_OPERATION_ENABLE
    // register for *any* block, including PPU -- only the broadcast PC
    // kick, with a bitmask scoped to exactly the blocks that task uses.
    // Dropped the per-block write and folded KICK_PPU into the kick below
    // instead, matching that real pattern. NOT YET HARDWARE-VALIDATED.
    push_kick(
        &mut cmds,
        KICK_CNA | KICK_CORE | KICK_DPU | KICK_DPU_RDMA | KICK_PPU,
    );

    cmds
}
