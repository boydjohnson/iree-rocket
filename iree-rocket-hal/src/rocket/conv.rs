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
    DOMAIN_CORE, DOMAIN_DPU, RegCmd, Register, RegisterMeta,
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
    core::{CoreClipTruncate, CoreDataoutSize0, CoreDataoutSize1, CoreMiscCfg},
    dpu::{
        DpuBnAluCfg, DpuBnCfg, DpuBnMulCfg, DpuBnReluxCmpValue, DpuBsAluCfg, DpuBsCfg, DpuBsMulCfg,
        DpuBsOwCfg, DpuBsOwOp, DpuBsReluxCmpValue, DpuDataCubeChannel, DpuDataCubeHeight,
        DpuDataCubeNotchAddr, DpuDataCubeWidth, DpuDataFormat, DpuDstBaseAddr, DpuDstSurfStride,
        DpuEwCfg, DpuEwCvtOffsetValue, DpuEwCvtScaleValue, DpuEwOpValue0, DpuEwOpValue1,
        DpuEwOpValue2, DpuEwOpValue3, DpuEwOpValue4, DpuEwOpValue5, DpuEwOpValue6, DpuEwOpValue7,
        DpuEwReluxCmpValue, DpuFeatureModeCfg, DpuLutAccessCfg, DpuLutAccessData, DpuLutCfg,
        DpuLutInfo, DpuLutLeEnd, DpuLutLeSlopeScale, DpuLutLeSlopeShift, DpuLutLeStart,
        DpuLutLoEnd, DpuLutLoSlopeScale, DpuLutLoSlopeShift, DpuLutLoStart, DpuOffsetPend,
        DpuOutCvtOffset, DpuOutCvtScale, DpuOutCvtShift, DpuSPointer, DpuSurfaceAdd, DpuWdmaSize0,
        DpuWdmaSize1,
    },
    dpu_rdma::{
        DpuRdmaBnBaseAddr, DpuRdmaBrdmaCfg, DpuRdmaBsBaseAddr, DpuRdmaDataCubeChannel,
        DpuRdmaDataCubeHeight, DpuRdmaDataCubeWidth, DpuRdmaErdmaCfg, DpuRdmaEwBaseAddr,
        DpuRdmaEwSurfNotch, DpuRdmaEwSurfStride, DpuRdmaFeatureModeCfg, DpuRdmaNrdmaCfg,
        DpuRdmaPadCfg, DpuRdmaSPointer, DpuRdmaSrcBaseAddr, DpuRdmaSrcDmaCfg, DpuRdmaSurfNotch,
        DpuRdmaWeight,
    },
    pc::PCRegisterAmounts,
};

/// `[kernel_height, kernel_width]`.
pub type Kernels = [usize; 2];

#[derive(Clone, Copy)]
struct KernelProgramming {
    feature_grains: u32,
    weight_bytes: u32,
    weight_bytes_per_kernel: u32,
    weight_size: u32,
    padding: u32,
}

fn kernel_programming(kernels: Kernels) -> KernelProgramming {
    match kernels {
        [1, 1] => KernelProgramming {
            feature_grains: 0x0000_0210,
            weight_bytes: 0x0000_0080,
            weight_bytes_per_kernel: 0x0000_0010,
            weight_size: 0x0101_0008,
            padding: 0,
        },
        [3, 3] => KernelProgramming {
            feature_grains: 0x0000_0240,
            weight_bytes: 0x0000_0480,
            weight_bytes_per_kernel: 0x0000_0090,
            weight_size: 0x0303_0008,
            padding: 0x0000_0011,
        },
        _ => panic!("conv_2d only has vendor reference data for 1x1 and 3x3 square kernels"),
    }
}

#[inline]
fn register<R: RegisterMeta>(value: u32) -> RegCmd {
    Register::<R>::from_val(value).build()
}

macro_rules! push_registers {
    ($commands:expr; $($register:ty => $value:expr),+ $(,)?) => {
        $(
            $commands.push(register::<$register>($value));
        )+
    };
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
    let kernel = kernel_programming(kernels);
    let mut commands = Vec::with_capacity(136);

    // CNA preamble, followed by the DPU/DPU_RDMA ping-pong pointers.
    push_registers!(commands;
        CnaCbufCon0 => 0x0000_00b1,
        CnaDcompRegnum => 0,
        CnaDcompCtrl => 0,
        CnaConvCon1 => 0x6000_a120,
        DpuSPointer => 0x0000_000e,
        DpuRdmaSPointer => 0x0000_000e,
    );

    // CNA convolution and DMA programming.
    push_registers!(commands;
        CnaConvCon1 => 0x6000_a120,
        CnaConvCon2 => kernel.feature_grains,
        CnaConvCon3 => 0x0000_0009,
        CnaDataSize0 => 0x0020_0020,
        CnaDataSize1 => 0x0002_0008,
        CnaDataSize2 => 0x0000_0020,
        CnaDataSize3 => 0x0000_0400,
        CnaWeightSize0 => kernel.weight_bytes,
        CnaWeightSize1 => kernel.weight_bytes_per_kernel,
        CnaWeightSize2 => kernel.weight_size,
        CnaCbufCon0 => 0x0000_00b1,
        CnaCbufCon1 => 0x0000_0400,
        CnaCvtCon0 => 0x0000_000b,
        CnaCvtCon1 => 0x0001_0000,
        CnaCvtCon2 => 0x0001_0000,
        CnaCvtCon3 => 0x0001_0000,
        CnaCvtCon4 => 0x0001_0000,
        CnaFcCon0 => 0,
        CnaFcCon1 => 0,
        CnaPadCon0 => kernel.padding,
        CnaFeatureDataAddr => 0,
        CnaFcCon2 => 0,
        CnaDmaCon0 => 0x000f_000f,
        CnaDmaCon1 => 0x0000_0020,
        CnaDmaCon2 => 0x0000_03e0,
        CnaFcDataSize0 => 0x0020_0020,
        CnaFcDataSize1 => 0x0000_0008,
        CnaDcompCtrl => 0,
        CnaDcompRegnum => 0,
        CnaDcompAddr0 => 0,
        CnaDcompAmount0 => 0,
        CnaDcompAmount1 => 0,
        CnaDcompAmount2 => 0,
        CnaDcompAmount3 => 0,
        CnaDcompAmount4 => 0,
        CnaDcompAmount5 => 0,
        CnaDcompAmount6 => 0,
        CnaDcompAmount7 => 0,
        CnaDcompAmount8 => 0,
        CnaDcompAmount9 => 0,
        CnaDcompAmount10 => 0,
        CnaDcompAmount11 => 0,
        CnaDcompAmount12 => 0,
        CnaDcompAmount13 => 0,
        CnaDcompAmount14 => 0,
        CnaDcompAmount15 => 0,
        CnaCvtCon5 => 0,
        CnaPadCon1 => 0,
    );

    // CORE. Offset 0x3030 is present in the vendor stream but is absent
    // from registers.xml/rkt_registers.h, so it cannot use a typed builder.
    push_registers!(commands;
        CoreMiscCfg => 0x0000_0200,
        CoreDataoutSize0 => 0x001f_001f,
        CoreDataoutSize1 => 0x0000_000f,
        CoreClipTruncate => 0,
    );
    commands.push(RegCmd::new(DOMAIN_CORE, 0x3030, 0));

    // DPU output, conversion, and disabled LUT programming.
    push_registers!(commands;
        DpuFeatureModeCfg => 0x0000_01e4,
        DpuDataFormat => 0x4800_0002,
        DpuOffsetPend => 0,
        DpuDstBaseAddr => 0,
        DpuDstSurfStride => 0x0000_4000,
        DpuDataCubeWidth => 0x0000_001f,
        DpuDataCubeHeight => 0x0000_001f,
        DpuDataCubeNotchAddr => 0,
        DpuDataCubeChannel => 0x0007_000f,
        DpuBsCfg => 0x0002_0150,
        DpuBsAluCfg => 0,
        DpuBsMulCfg => 0,
        DpuBsReluxCmpValue => 0,
        DpuBsOwCfg => 0x0000_0126,
        DpuBsOwOp => 0,
        DpuWdmaSize0 => 0x0000_000f,
        DpuWdmaSize1 => 0x001f_001f,
        DpuBnCfg => 0x0000_0053,
        DpuBnAluCfg => 0,
        DpuBnMulCfg => 0,
        DpuBnReluxCmpValue => 0,
        DpuEwCfg => 0x0000_0383,
        DpuEwCvtOffsetValue => 0,
        DpuEwCvtScaleValue => 1,
        DpuEwReluxCmpValue => 0,
        DpuOutCvtOffset => 0,
        DpuOutCvtScale => 0x0001_0001,
        DpuOutCvtShift => 0,
        DpuEwOpValue0 => 0,
        DpuEwOpValue1 => 0,
        DpuEwOpValue2 => 0,
        DpuEwOpValue3 => 0,
        DpuEwOpValue4 => 0,
        DpuEwOpValue5 => 0,
        DpuEwOpValue6 => 0,
        DpuEwOpValue7 => 0,
        DpuSurfaceAdd => 0x0000_8000,
    );
    // Like CORE 0x3030, DPU 0x40c4 has no generated register definition.
    commands.push(RegCmd::new(DOMAIN_DPU, 0x40c4, 0));
    push_registers!(commands;
        DpuLutAccessCfg => 0,
        DpuLutAccessData => 0,
        DpuLutCfg => 0,
        DpuLutInfo => 0,
        DpuLutLeStart => 0,
        DpuLutLeEnd => 0,
        DpuLutLoStart => 0,
        DpuLutLoEnd => 0,
        DpuLutLeSlopeScale => 0,
        DpuLutLeSlopeShift => 0,
        DpuLutLoSlopeScale => 0,
        DpuLutLoSlopeShift => 0,
    );

    // DPU_RDMA. The main feature path is disabled because CNA/CORE feed
    // DPU directly; BRDMA supplies the bias data.
    push_registers!(commands;
        DpuRdmaDataCubeWidth => 0x0000_001f,
        DpuRdmaDataCubeHeight => 0x0000_001f,
        DpuRdmaDataCubeChannel => 0x0000_000f,
        DpuRdmaSrcBaseAddr => 0,
        DpuRdmaBrdmaCfg => 0x0000_0002,
        DpuRdmaBsBaseAddr => 0,
        DpuRdmaNrdmaCfg => 0,
        DpuRdmaBnBaseAddr => 0,
        DpuRdmaErdmaCfg => 1,
        DpuRdmaEwBaseAddr => 0,
        DpuRdmaEwSurfStride => 0,
        DpuRdmaFeatureModeCfg => 0x0001_7850,
        DpuRdmaSrcDmaCfg => 0,
        DpuRdmaSurfNotch => 0,
        DpuRdmaPadCfg => 0,
        DpuRdmaWeight => 0x0101_0101,
        DpuRdmaEwSurfNotch => 0,
    );

    // Vendor PC trailer: placeholder, zero register count, required marker,
    // combined operation-enable mask, and six words of alignment padding.
    commands.push(RegCmd::new_raw(0));
    commands.push(register::<PCRegisterAmounts>(0));
    commands.push(RegCmd::new_raw(0x0041_0000_0000_0000));
    commands.push(RegCmd::new_raw(0x0081_0000_001d_0008));
    commands.extend((0..6).map(|_| RegCmd::new_raw(0)));

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
