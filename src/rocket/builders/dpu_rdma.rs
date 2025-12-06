use crate::rocket::{
    builders::{Bits, Register, RegisterMeta},
    registers::*,
};

// ========================================================================
// RDMA_S_STATUS (0x5000)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuRdmaSStatus;

impl RegisterMeta for DpuRdmaSStatus {
    const DOMAIN: u32 = target_DPU_RDMA;
    const OFFSET: u32 = REG_DPU_RDMA_RDMA_S_STATUS;
}

impl Register<DpuRdmaSStatus> {
    pub fn status_0(&mut self, status_0: Bits<2>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_S_STATUS_STATUS_0__MASK, unsafe {
            DPU_RDMA_RDMA_S_STATUS_STATUS_0(status_0.val())
        })
    }

    pub fn status_1(&mut self, status_1: Bits<2>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_S_STATUS_STATUS_1__MASK, unsafe {
            DPU_RDMA_RDMA_S_STATUS_STATUS_1(status_1.val())
        })
    }
}

// ========================================================================
// RDMA_S_POINTER (0x5004)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuRdmaSPointer;

impl RegisterMeta for DpuRdmaSPointer {
    const DOMAIN: u32 = target_DPU_RDMA;
    const OFFSET: u32 = REG_DPU_RDMA_RDMA_S_POINTER;
}

impl Register<DpuRdmaSPointer> {
    pub fn pointer(&mut self, pointer: Bits<1>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_S_POINTER_POINTER__MASK, unsafe {
            DPU_RDMA_RDMA_S_POINTER_POINTER(pointer.val())
        })
    }

    pub fn pointer_pp_en(&mut self, pointer_pp_en: Bits<1>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN__MASK, unsafe {
            DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN(pointer_pp_en.val())
        })
    }

    pub fn executer_pp_en(&mut self, executer_pp_en: Bits<1>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN__MASK, unsafe {
            DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN(executer_pp_en.val())
        })
    }

    pub fn pointer_pp_mode(&mut self, pointer_pp_mode: Bits<1>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE__MASK, unsafe {
            DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE(pointer_pp_mode.val())
        })
    }

    pub fn pointer_pp_clear(&mut self, pointer_pp_clear: Bits<1>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_S_POINTER_POINTER_PP_CLEAR__MASK, unsafe {
            DPU_RDMA_RDMA_S_POINTER_POINTER_PP_CLEAR(pointer_pp_clear.val())
        })
    }

    pub fn executer_pp_clear(&mut self, executer_pp_clear: Bits<1>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_CLEAR__MASK, unsafe {
            DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_CLEAR(executer_pp_clear.val())
        })
    }

    pub fn executer(&mut self, executer: Bits<1>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_S_POINTER_EXECUTER__MASK, unsafe {
            DPU_RDMA_RDMA_S_POINTER_EXECUTER(executer.val())
        })
    }
}

// ========================================================================
// RDMA_OPERATION_ENABLE (0x5008)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuRdmaOperationEnable;

impl RegisterMeta for DpuRdmaOperationEnable {
    const DOMAIN: u32 = target_DPU_RDMA;
    const OFFSET: u32 = REG_DPU_RDMA_RDMA_OPERATION_ENABLE;
}

impl Register<DpuRdmaOperationEnable> {
    pub fn op_en(&mut self, op_en: Bits<1>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_OPERATION_ENABLE_OP_EN__MASK, unsafe {
            DPU_RDMA_RDMA_OPERATION_ENABLE_OP_EN(op_en.val())
        })
    }
}

// ========================================================================
// RDMA_DATA_CUBE_WIDTH (0x500C)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuRdmaDataCubeWidth;

impl RegisterMeta for DpuRdmaDataCubeWidth {
    const DOMAIN: u32 = target_DPU_RDMA;
    const OFFSET: u32 = REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH;
}

impl Register<DpuRdmaDataCubeWidth> {
    pub fn width(&mut self, width: Bits<13>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH__MASK, unsafe {
            DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH(width.val())
        })
    }
}

// ========================================================================
// RDMA_DATA_CUBE_HEIGHT (0x5010)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuRdmaDataCubeHeight;

impl RegisterMeta for DpuRdmaDataCubeHeight {
    const DOMAIN: u32 = target_DPU_RDMA;
    const OFFSET: u32 = REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT;
}

impl Register<DpuRdmaDataCubeHeight> {
    pub fn height(&mut self, height: Bits<13>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT__MASK, unsafe {
            DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT(height.val())
        })
    }

    pub fn ew_line_notch_addr(&mut self, ew_line_notch_addr: Bits<13>) -> &mut Self {
        self.set_field(
            DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_EW_LINE_NOTCH_ADDR__MASK,
            unsafe { DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_EW_LINE_NOTCH_ADDR(ew_line_notch_addr.val()) },
        )
    }
}

// ========================================================================
// RDMA_DATA_CUBE_CHANNEL (0x5014)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuRdmaDataCubeChannel;

impl RegisterMeta for DpuRdmaDataCubeChannel {
    const DOMAIN: u32 = target_DPU_RDMA;
    const OFFSET: u32 = REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL;
}

impl Register<DpuRdmaDataCubeChannel> {
    pub fn channel(&mut self, channel: Bits<13>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL__MASK, unsafe {
            DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL(channel.val())
        })
    }
}

// ========================================================================
// RDMA_SRC_BASE_ADDR (0x5018)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuRdmaSrcBaseAddr;

impl RegisterMeta for DpuRdmaSrcBaseAddr {
    const DOMAIN: u32 = target_DPU_RDMA;
    const OFFSET: u32 = REG_DPU_RDMA_RDMA_SRC_BASE_ADDR;
}

impl Register<DpuRdmaSrcBaseAddr> {
    pub fn src_base_addr(&mut self, src_base_addr: Bits<32>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__MASK, unsafe {
            DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR(src_base_addr.val())
        })
    }
}

// ========================================================================
// RDMA_BRDMA_CFG (0x501C)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuRdmaBrdmaCfg;

impl RegisterMeta for DpuRdmaBrdmaCfg {
    const DOMAIN: u32 = target_DPU_RDMA;
    const OFFSET: u32 = REG_DPU_RDMA_RDMA_BRDMA_CFG;
}

impl Register<DpuRdmaBrdmaCfg> {
    pub fn brdma_data_use(&mut self, brdma_data_use: Bits<4>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_BRDMA_CFG_BRDMA_DATA_USE__MASK, unsafe {
            DPU_RDMA_RDMA_BRDMA_CFG_BRDMA_DATA_USE(brdma_data_use.val())
        })
    }
}

// ========================================================================
// RDMA_BS_BASE_ADDR (0x5020)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuRdmaBsBaseAddr;

impl RegisterMeta for DpuRdmaBsBaseAddr {
    const DOMAIN: u32 = target_DPU_RDMA;
    const OFFSET: u32 = REG_DPU_RDMA_RDMA_BS_BASE_ADDR;
}

impl Register<DpuRdmaBsBaseAddr> {
    pub fn bs_base_addr(&mut self, bs_base_addr: Bits<32>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_BS_BASE_ADDR_BS_BASE_ADDR__MASK, unsafe {
            DPU_RDMA_RDMA_BS_BASE_ADDR_BS_BASE_ADDR(bs_base_addr.val())
        })
    }
}

// ========================================================================
// RDMA_NRDMA_CFG (0x5028)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuRdmaNrdmaCfg;

impl RegisterMeta for DpuRdmaNrdmaCfg {
    const DOMAIN: u32 = target_DPU_RDMA;
    const OFFSET: u32 = REG_DPU_RDMA_RDMA_NRDMA_CFG;
}

impl Register<DpuRdmaNrdmaCfg> {
    pub fn nrdma_data_use(&mut self, nrdma_data_use: Bits<4>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_NRDMA_CFG_NRDMA_DATA_USE__MASK, unsafe {
            DPU_RDMA_RDMA_NRDMA_CFG_NRDMA_DATA_USE(nrdma_data_use.val())
        })
    }
}

// ========================================================================
// RDMA_BN_BASE_ADDR (0x502C)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuRdmaBnBaseAddr;

impl RegisterMeta for DpuRdmaBnBaseAddr {
    const DOMAIN: u32 = target_DPU_RDMA;
    const OFFSET: u32 = REG_DPU_RDMA_RDMA_BN_BASE_ADDR;
}

impl Register<DpuRdmaBnBaseAddr> {
    pub fn bn_base_addr(&mut self, bn_base_addr: Bits<32>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_BN_BASE_ADDR_BN_BASE_ADDR__MASK, unsafe {
            DPU_RDMA_RDMA_BN_BASE_ADDR_BN_BASE_ADDR(bn_base_addr.val())
        })
    }
}

// ========================================================================
// RDMA_ERDMA_CFG (0x5034)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuRdmaErdmaCfg;

impl RegisterMeta for DpuRdmaErdmaCfg {
    const DOMAIN: u32 = target_DPU_RDMA;
    const OFFSET: u32 = REG_DPU_RDMA_RDMA_ERDMA_CFG;
}

impl Register<DpuRdmaErdmaCfg> {
    pub fn erdma_disable(&mut self, erdma_disable: Bits<1>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DISABLE__MASK, unsafe {
            DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DISABLE(erdma_disable.val())
        })
    }

    pub fn ov4k_bypass(&mut self, ov4k_bypass: Bits<1>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_ERDMA_CFG_OV4K_BYPASS__MASK, unsafe {
            DPU_RDMA_RDMA_ERDMA_CFG_OV4K_BYPASS(ov4k_bypass.val())
        })
    }

    pub fn erdma_data_size(&mut self, erdma_data_size: Bits<2>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE__MASK, unsafe {
            DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE(erdma_data_size.val())
        })
    }

    pub fn erdma_nonalign(&mut self, erdma_nonalign: Bits<1>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_NONALIGN__MASK, unsafe {
            DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_NONALIGN(erdma_nonalign.val())
        })
    }

    pub fn erdma_surf_mode(&mut self, erdma_surf_mode: Bits<1>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_SURF_MODE__MASK, unsafe {
            DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_SURF_MODE(erdma_surf_mode.val())
        })
    }

    pub fn erdma_data_mode(&mut self, erdma_data_mode: Bits<2>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE__MASK, unsafe {
            DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE(erdma_data_mode.val())
        })
    }
}

// ========================================================================
// RDMA_EW_BASE_ADDR (0x5038)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuRdmaEwBaseAddr;

impl RegisterMeta for DpuRdmaEwBaseAddr {
    const DOMAIN: u32 = target_DPU_RDMA;
    const OFFSET: u32 = REG_DPU_RDMA_RDMA_EW_BASE_ADDR;
}

impl Register<DpuRdmaEwBaseAddr> {
    pub fn ew_base_addr(&mut self, ew_base_addr: Bits<32>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR__MASK, unsafe {
            DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR(ew_base_addr.val())
        })
    }
}

// ========================================================================
// RDMA_EW_SURF_STRIDE (0x5040)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuRdmaEwSurfStride;

impl RegisterMeta for DpuRdmaEwSurfStride {
    const DOMAIN: u32 = target_DPU_RDMA;
    const OFFSET: u32 = REG_DPU_RDMA_RDMA_EW_SURF_STRIDE;
}

impl Register<DpuRdmaEwSurfStride> {
    pub fn ew_surf_stride(&mut self, ew_surf_stride: Bits<28>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_EW_SURF_STRIDE_EW_SURF_STRIDE__MASK, unsafe {
            DPU_RDMA_RDMA_EW_SURF_STRIDE_EW_SURF_STRIDE(ew_surf_stride.val())
        })
    }
}

// ========================================================================
// RDMA_FEATURE_MODE_CFG (0x5044)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuRdmaFeatureModeCfg;

impl RegisterMeta for DpuRdmaFeatureModeCfg {
    const DOMAIN: u32 = target_DPU_RDMA;
    const OFFSET: u32 = REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG;
}

impl Register<DpuRdmaFeatureModeCfg> {
    pub fn flying_mode(&mut self, flying_mode: Bits<1>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE__MASK, unsafe {
            DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE(flying_mode.val())
        })
    }

    pub fn conv_mode(&mut self, conv_mode: Bits<2>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_FEATURE_MODE_CFG_CONV_MODE__MASK, unsafe {
            DPU_RDMA_RDMA_FEATURE_MODE_CFG_CONV_MODE(conv_mode.val())
        })
    }

    pub fn mrdma_fp16tofp32_en(&mut self, mrdma_fp16tofp32_en: Bits<1>) -> &mut Self {
        self.set_field(
            DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN__MASK,
            unsafe {
                DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN(mrdma_fp16tofp32_en.val())
            },
        )
    }

    pub fn mrdma_disable(&mut self, mrdma_disable: Bits<1>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_DISABLE__MASK, unsafe {
            DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_DISABLE(mrdma_disable.val())
        })
    }

    pub fn proc_precision(&mut self, proc_precision: Bits<3>) -> &mut Self {
        self.set_field(
            DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION__MASK,
            unsafe { DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION(proc_precision.val()) },
        )
    }

    pub fn comb_use(&mut self, comb_use: Bits<3>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_FEATURE_MODE_CFG_COMB_USE__MASK, unsafe {
            DPU_RDMA_RDMA_FEATURE_MODE_CFG_COMB_USE(comb_use.val())
        })
    }

    pub fn burst_len(&mut self, burst_len: Bits<4>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN__MASK, unsafe {
            DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN(burst_len.val())
        })
    }

    pub fn in_precision(&mut self, in_precision: Bits<3>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION__MASK, unsafe {
            DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION(in_precision.val())
        })
    }
}

// ========================================================================
// RDMA_SRC_DMA_CFG (0x5048)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuRdmaSrcDmaCfg;

impl RegisterMeta for DpuRdmaSrcDmaCfg {
    const DOMAIN: u32 = target_DPU_RDMA;
    const OFFSET: u32 = REG_DPU_RDMA_RDMA_SRC_DMA_CFG;
}

impl Register<DpuRdmaSrcDmaCfg> {
    pub fn kernel_width(&mut self, kernel_width: Bits<3>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_SRC_DMA_CFG_KERNEL_WIDTH__MASK, unsafe {
            DPU_RDMA_RDMA_SRC_DMA_CFG_KERNEL_WIDTH(kernel_width.val())
        })
    }

    pub fn kernel_height(&mut self, kernel_height: Bits<3>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_SRC_DMA_CFG_KERNEL_HEIGHT__MASK, unsafe {
            DPU_RDMA_RDMA_SRC_DMA_CFG_KERNEL_HEIGHT(kernel_height.val())
        })
    }

    pub fn kernel_stride_width(&mut self, kernel_stride_width: Bits<3>) -> &mut Self {
        self.set_field(
            DPU_RDMA_RDMA_SRC_DMA_CFG_KERNEL_STRIDE_WIDTH__MASK,
            unsafe { DPU_RDMA_RDMA_SRC_DMA_CFG_KERNEL_STRIDE_WIDTH(kernel_stride_width.val()) },
        )
    }

    pub fn kernel_stride_height(&mut self, kernel_stride_height: Bits<3>) -> &mut Self {
        self.set_field(
            DPU_RDMA_RDMA_SRC_DMA_CFG_KERNEL_STRIDE_HEIGHT__MASK,
            unsafe { DPU_RDMA_RDMA_SRC_DMA_CFG_KERNEL_STRIDE_HEIGHT(kernel_stride_height.val()) },
        )
    }

    pub fn unpooling_en(&mut self, unpooling_en: Bits<1>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_SRC_DMA_CFG_UNPOOLING_EN__MASK, unsafe {
            DPU_RDMA_RDMA_SRC_DMA_CFG_UNPOOLING_EN(unpooling_en.val())
        })
    }

    pub fn pooling_method(&mut self, pooling_method: Bits<1>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_SRC_DMA_CFG_POOLING_METHOD__MASK, unsafe {
            DPU_RDMA_RDMA_SRC_DMA_CFG_POOLING_METHOD(pooling_method.val())
        })
    }

    pub fn line_notch_addr(&mut self, line_notch_addr: Bits<13>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_SRC_DMA_CFG_LINE_NOTCH_ADDR__MASK, unsafe {
            DPU_RDMA_RDMA_SRC_DMA_CFG_LINE_NOTCH_ADDR(line_notch_addr.val())
        })
    }
}

// ========================================================================
// RDMA_SURF_NOTCH (0x504C)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuRdmaSurfNotch;

impl RegisterMeta for DpuRdmaSurfNotch {
    const DOMAIN: u32 = target_DPU_RDMA;
    const OFFSET: u32 = REG_DPU_RDMA_RDMA_SURF_NOTCH;
}

impl Register<DpuRdmaSurfNotch> {
    pub fn surf_notch_addr(&mut self, surf_notch_addr: Bits<28>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_SURF_NOTCH_SURF_NOTCH_ADDR__MASK, unsafe {
            DPU_RDMA_RDMA_SURF_NOTCH_SURF_NOTCH_ADDR(surf_notch_addr.val())
        })
    }
}

// ========================================================================
// RDMA_PAD_CFG (0x5064)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuRdmaPadCfg;

impl RegisterMeta for DpuRdmaPadCfg {
    const DOMAIN: u32 = target_DPU_RDMA;
    const OFFSET: u32 = REG_DPU_RDMA_RDMA_PAD_CFG;
}

impl Register<DpuRdmaPadCfg> {
    pub fn pad_left(&mut self, pad_left: Bits<3>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_PAD_CFG_PAD_LEFT__MASK, unsafe {
            DPU_RDMA_RDMA_PAD_CFG_PAD_LEFT(pad_left.val())
        })
    }

    pub fn pad_top(&mut self, pad_top: Bits<3>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_PAD_CFG_PAD_TOP__MASK, unsafe {
            DPU_RDMA_RDMA_PAD_CFG_PAD_TOP(pad_top.val())
        })
    }

    pub fn pad_value(&mut self, pad_value: Bits<16>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_PAD_CFG_PAD_VALUE__MASK, unsafe {
            DPU_RDMA_RDMA_PAD_CFG_PAD_VALUE(pad_value.val())
        })
    }
}

// ========================================================================
// RDMA_WEIGHT (0x5068)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuRdmaWeight;

impl RegisterMeta for DpuRdmaWeight {
    const DOMAIN: u32 = target_DPU_RDMA;
    const OFFSET: u32 = REG_DPU_RDMA_RDMA_WEIGHT;
}

impl Register<DpuRdmaWeight> {
    pub fn m_weight(&mut self, m_weight: Bits<8>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_WEIGHT_M_WEIGHT__MASK, unsafe {
            DPU_RDMA_RDMA_WEIGHT_M_WEIGHT(m_weight.val())
        })
    }

    pub fn b_weight(&mut self, b_weight: Bits<8>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_WEIGHT_B_WEIGHT__MASK, unsafe {
            DPU_RDMA_RDMA_WEIGHT_B_WEIGHT(b_weight.val())
        })
    }

    pub fn n_weight(&mut self, n_weight: Bits<8>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_WEIGHT_N_WEIGHT__MASK, unsafe {
            DPU_RDMA_RDMA_WEIGHT_N_WEIGHT(n_weight.val())
        })
    }

    pub fn e_weight(&mut self, e_weight: Bits<8>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_WEIGHT_E_WEIGHT__MASK, unsafe {
            DPU_RDMA_RDMA_WEIGHT_E_WEIGHT(e_weight.val())
        })
    }
}

// ========================================================================
// RDMA_EW_SURF_NOTCH (0x506C)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuRdmaEwSurfNotch;

impl RegisterMeta for DpuRdmaEwSurfNotch {
    const DOMAIN: u32 = target_DPU_RDMA;
    const OFFSET: u32 = REG_DPU_RDMA_RDMA_EW_SURF_NOTCH;
}

impl Register<DpuRdmaEwSurfNotch> {
    pub fn ew_surf_notch(&mut self, ew_surf_notch: Bits<28>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_EW_SURF_NOTCH_EW_SURF_NOTCH__MASK, unsafe {
            DPU_RDMA_RDMA_EW_SURF_NOTCH_EW_SURF_NOTCH(ew_surf_notch.val())
        })
    }
}
