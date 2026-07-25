//! Vendor-derived convolution register program.
//!
//! This is deliberately a narrow reference implementation: it reproduces
//! group 1 (the complete single-core alternative) from the vendor-compiled
//! `32x32x3 -> 32x32x8` fp16 convolution captures. The two captures differ
//! only in kernel geometry: 1x1/no padding versus 3x3/SAME padding.
//!
//! Keeping this separate from `rocket::regcmd` gives us a bit-exact baseline
//! while the general convolution builder grows the missing padding and
//! multi-core-plan inputs.

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
    values::{ArgbInputMode, BurstLength, DataPrecision, DpuOutputMode},
};

/// `[kernel_height, kernel_width]`.
pub type Kernels = [usize; 2];

#[derive(Clone, Copy)]
struct KernelProgramming {
    // The builder accepts the 10-bit field value. The encoded register
    // words are therefore 0x210/0x240 because the field starts at bit 4.
    feature_grains: u32,
    size: u32,
    padding: u32,
}

fn kernel_programming(kernels: Kernels) -> KernelProgramming {
    match kernels {
        [1, 1] => KernelProgramming {
            feature_grains: 0x21,
            size: 1,
            padding: 0,
        },
        [3, 3] => KernelProgramming {
            feature_grains: 0x24,
            size: 3,
            padding: 1,
        },
        _ => panic!("conv_2d only has vendor reference data for 1x1 and 3x3 square kernels"),
    }
}

#[inline]
fn zero<R: RegisterMeta>() -> RegCmd {
    Register::<R>::new().build()
}

/// Builds the vendor-matching single-core regcmd program for the captured
/// `32x32x3 -> 32x32x8` fp16 convolution.
///
/// `kernels` must be `[1, 1]` or `[3, 3]`. Buffer-address fields are zero,
/// as they are in the captured RKNN regcmd blob; a caller that submits this
/// program must relocate the feature, weight, bias, and destination
/// addresses first.
///
/// The returned 136-word sequence is group 1 of the vendor blob. Groups
/// 2-3 and 4-6 are alternative two- and three-core height-split programs,
/// not continuations of this command stream.
pub fn conv_2d(kernels: Kernels) -> Vec<RegCmd> {
    const WIDTH: u32 = 32;
    const HEIGHT: u32 = 32;
    const INPUT_CHANNELS: u32 = 3;
    const TASK_INPUT_CHANNELS: u32 = 8;
    const OUTPUT_CHANNELS: u32 = 8;
    const TASK_OUTPUT_CHANNELS: u32 = 16;
    const FP16_BYTES: u32 = 2;
    const WEIGHT_BANKS: u32 = 11;
    const DATA_BANKS: u32 = 1;

    let kernel = kernel_programming(kernels);
    let weight_bytes_per_kernel = kernel.size * kernel.size * TASK_INPUT_CHANNELS * FP16_BYTES;
    let weight_bytes = weight_bytes_per_kernel * OUTPUT_CHANNELS;
    let mut commands = Vec::with_capacity(136);

    // CNA preamble, followed by the DPU/DPU_RDMA ping-pong pointers.
    let mut cbuf_con0 = Register::<CnaCbufCon0>::new();
    cbuf_con0
        .weight_bank(Bits::new(WEIGHT_BANKS))
        .data_bank(Bits::new(DATA_BANKS));
    commands.push(cbuf_con0.build());
    commands.push(zero::<CnaDcompRegnum>());
    commands.push(zero::<CnaDcompCtrl>());

    let mut conv_con1 = Register::<CnaConvCon1>::new();
    conv_con1
        .nonalign_dma(Bits::new(1))
        .group_line_off(Bits::new(1))
        .argb_in(ArgbInputMode::ThreeChannels.into())
        .proc_precision(DataPrecision::Fp16.into())
        .in_precision(DataPrecision::Fp16.into());
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
            .feature_grains(Bits::new(kernel.feature_grains))
            .build(),
    );
    commands.push(
        Register::<CnaConvCon3>::new()
            .conv_x_stride(Bits::new(1))
            .conv_y_stride(Bits::new(1))
            .build(),
    );
    commands.push(
        Register::<CnaDataSize0>::new()
            .datain_width(Bits::new(WIDTH))
            .datain_height(Bits::new(HEIGHT))
            .build(),
    );
    commands.push(
        Register::<CnaDataSize1>::new()
            .datain_channel_real(Bits::new(INPUT_CHANNELS - 1))
            .datain_channel(Bits::new(TASK_INPUT_CHANNELS))
            .build(),
    );
    commands.push(
        Register::<CnaDataSize2>::new()
            .dataout_width(Bits::new(WIDTH))
            .build(),
    );
    commands.push(
        Register::<CnaDataSize3>::new()
            .dataout_atomics(Bits::new(WIDTH * HEIGHT))
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
            .weight_width(Bits::new(kernel.size))
            .weight_height(Bits::new(kernel.size))
            .weight_kernels(Bits::new(OUTPUT_CHANNELS))
            .build(),
    );
    commands.push(cbuf_con0.build());
    commands.push(
        Register::<CnaCbufCon1>::new()
            .data_entries(Bits::new(WIDTH * HEIGHT))
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
        Register::<CnaPadCon0>::new()
            .pad_top(Bits::new(kernel.padding))
            .pad_left(Bits::new(kernel.padding))
            .build(),
    );
    commands.push(zero::<CnaFeatureDataAddr>());
    commands.push(zero::<CnaFcCon2>());
    commands.push(
        Register::<CnaDmaCon0>::new()
            .data_burst_len(BurstLength::Sixteen.into())
            .weight_burst_len(BurstLength::Sixteen.into())
            .build(),
    );
    commands.push(
        Register::<CnaDmaCon1>::new()
            .line_stride(Bits::new(WIDTH))
            .build(),
    );
    commands.push(
        Register::<CnaDmaCon2>::new()
            .surf_stride(Bits::new(WIDTH * (HEIGHT - 1)))
            .build(),
    );
    commands.push(
        Register::<CnaFcDataSize0>::new()
            .dma_width(Bits::new(WIDTH))
            .dma_height(Bits::new(HEIGHT))
            .build(),
    );
    commands.push(
        Register::<CnaFcDataSize1>::new()
            .dma_channel(Bits::new(TASK_INPUT_CHANNELS))
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
    commands.push(zero::<CnaPadCon1>());

    // CORE.
    commands.push(
        Register::<CoreMiscCfg>::new()
            .proc_precision(DataPrecision::Fp16.into())
            .build(),
    );
    commands.push(
        Register::<CoreDataoutSize0>::new()
            .dataout_width(Bits::new(WIDTH - 1))
            .dataout_height(Bits::new(HEIGHT - 1))
            .build(),
    );
    commands.push(
        Register::<CoreDataoutSize1>::new()
            .dataout_channel(Bits::new(TASK_OUTPUT_CHANNELS - 1))
            .build(),
    );
    commands.push(zero::<CoreClipTruncate>());
    commands.push(zero::<CoreReserved3030>());

    // DPU output, conversion, and disabled LUT programming.
    commands.push(
        Register::<DpuFeatureModeCfg>::new()
            .burst_len(BurstLength::Sixteen.into())
            .output_mode(DpuOutputMode::ExternalMemory.into())
            .build(),
    );
    commands.push(
        Register::<DpuDataFormat>::new()
            .in_precision(DataPrecision::Fp16.into())
            .out_precision(DataPrecision::Fp16.into())
            .proc_precision(DataPrecision::Fp16.into())
            .build(),
    );
    commands.push(zero::<DpuOffsetPend>());
    commands.push(zero::<DpuDstBaseAddr>());
    commands.push(
        Register::<DpuDstSurfStride>::new()
            .dst_surf_stride(Bits::new(WIDTH * HEIGHT))
            .build(),
    );
    commands.push(
        Register::<DpuDataCubeWidth>::new()
            .width(Bits::new(WIDTH - 1))
            .build(),
    );
    commands.push(
        Register::<DpuDataCubeHeight>::new()
            .height(Bits::new(HEIGHT - 1))
            .build(),
    );
    commands.push(zero::<DpuDataCubeNotchAddr>());
    commands.push(
        Register::<DpuDataCubeChannel>::new()
            .orig_channel(Bits::new(OUTPUT_CHANNELS - 1))
            .channel(Bits::new(TASK_OUTPUT_CHANNELS - 1))
            .build(),
    );
    commands.push(
        Register::<DpuBsCfg>::new()
            .bs_alu_algo(Bits::new(2))
            .bs_alu_src(Bits::new(1))
            .bs_relu_bypass(Bits::new(1))
            .bs_mul_bypass(Bits::new(1))
            .build(),
    );
    commands.push(zero::<DpuBsAluCfg>());
    commands.push(zero::<DpuBsMulCfg>());
    commands.push(zero::<DpuBsReluxCmpValue>());
    commands.push(
        Register::<DpuBsOwCfg>::new()
            .size_e_0(Bits::new(1))
            .size_e_1(Bits::new(1))
            .size_e_2(Bits::new(1))
            .od_bypass(Bits::new(1))
            .build(),
    );
    commands.push(zero::<DpuBsOwOp>());
    commands.push(
        Register::<DpuWdmaSize0>::new()
            .channel_wdma(Bits::new(TASK_OUTPUT_CHANNELS - 1))
            .build(),
    );
    commands.push(
        Register::<DpuWdmaSize1>::new()
            .height_wdma(Bits::new(HEIGHT - 1))
            .width_wdma(Bits::new(WIDTH - 1))
            .build(),
    );
    commands.push(
        Register::<DpuBnCfg>::new()
            .bn_relu_bypass(Bits::new(1))
            .bn_mul_bypass(Bits::new(1))
            .bn_alu_bypass(Bits::new(1))
            .bn_bypass(Bits::new(1))
            .build(),
    );
    commands.push(zero::<DpuBnAluCfg>());
    commands.push(zero::<DpuBnMulCfg>());
    commands.push(zero::<DpuBnReluxCmpValue>());
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
    commands.push(zero::<DpuOutCvtOffset>());
    commands.push(
        Register::<DpuOutCvtScale>::new()
            .fp32tofp16_en(Bits::new(1))
            .out_cvt_scale(Bits::new(1))
            .build(),
    );
    commands.push(zero::<DpuOutCvtShift>());
    commands.push(zero::<DpuEwOpValue0>());
    commands.push(zero::<DpuEwOpValue1>());
    commands.push(zero::<DpuEwOpValue2>());
    commands.push(zero::<DpuEwOpValue3>());
    commands.push(zero::<DpuEwOpValue4>());
    commands.push(zero::<DpuEwOpValue5>());
    commands.push(zero::<DpuEwOpValue6>());
    commands.push(zero::<DpuEwOpValue7>());
    commands.push(
        Register::<DpuSurfaceAdd>::new()
            .surf_add(Bits::new(WIDTH * HEIGHT * FP16_BYTES))
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
            .width(Bits::new(WIDTH - 1))
            .build(),
    );
    commands.push(
        Register::<DpuRdmaDataCubeHeight>::new()
            .height(Bits::new(HEIGHT - 1))
            .build(),
    );
    commands.push(
        Register::<DpuRdmaDataCubeChannel>::new()
            .channel(Bits::new(TASK_OUTPUT_CHANNELS - 1))
            .build(),
    );
    commands.push(zero::<DpuRdmaSrcBaseAddr>());
    commands.push(
        Register::<DpuRdmaBrdmaCfg>::new()
            .brdma_data_use(Bits::new(1))
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
            .in_precision(DataPrecision::Fp16.into())
            .proc_precision(DataPrecision::Fp16.into())
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
    #[should_panic(expected = "only has vendor reference data")]
    fn rejects_uncaptured_kernel_geometry() {
        let _ = conv_2d([5, 5]);
    }
}
