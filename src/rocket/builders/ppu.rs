use crate::rocket::{
    builders::{Bits, Register, RegisterMeta},
    registers::*,
};

// ========================================================================
// S_STATUS (0x6000)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct PpuSStatus;

impl RegisterMeta for PpuSStatus {
    const DOMAIN: u32 = target_PPU;
    const OFFSET: u32 = REG_PPU_S_STATUS;
}

impl Register<PpuSStatus> {
    pub fn status_0(&mut self, status_0: Bits<2>) -> &mut Self {
        self.set_field(PPU_S_STATUS_STATUS_0__MASK, unsafe {
            PPU_S_STATUS_STATUS_0(status_0.val())
        })
    }

    pub fn status_1(&mut self, status_1: Bits<2>) -> &mut Self {
        self.set_field(PPU_S_STATUS_STATUS_1__MASK, unsafe {
            PPU_S_STATUS_STATUS_1(status_1.val())
        })
    }
}

// ========================================================================
// S_POINTER (0x6004)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct PpuSPointer;

impl RegisterMeta for PpuSPointer {
    const DOMAIN: u32 = target_PPU;
    const OFFSET: u32 = REG_PPU_S_POINTER;
}

impl Register<PpuSPointer> {
    pub fn pointer(&mut self, pointer: Bits<1>) -> &mut Self {
        self.set_field(PPU_S_POINTER_POINTER__MASK, unsafe {
            PPU_S_POINTER_POINTER(pointer.val())
        })
    }

    pub fn pointer_pp_en(&mut self, pointer_pp_en: Bits<1>) -> &mut Self {
        self.set_field(PPU_S_POINTER_POINTER_PP_EN__MASK, unsafe {
            PPU_S_POINTER_POINTER_PP_EN(pointer_pp_en.val())
        })
    }

    pub fn executer_pp_en(&mut self, executer_pp_en: Bits<1>) -> &mut Self {
        self.set_field(PPU_S_POINTER_EXECUTER_PP_EN__MASK, unsafe {
            PPU_S_POINTER_EXECUTER_PP_EN(executer_pp_en.val())
        })
    }

    pub fn pointer_pp_mode(&mut self, pointer_pp_mode: Bits<1>) -> &mut Self {
        self.set_field(PPU_S_POINTER_POINTER_PP_MODE__MASK, unsafe {
            PPU_S_POINTER_POINTER_PP_MODE(pointer_pp_mode.val())
        })
    }

    pub fn pointer_pp_clear(&mut self, pointer_pp_clear: Bits<1>) -> &mut Self {
        self.set_field(PPU_S_POINTER_POINTER_PP_CLEAR__MASK, unsafe {
            PPU_S_POINTER_POINTER_PP_CLEAR(pointer_pp_clear.val())
        })
    }

    pub fn executer_pp_clear(&mut self, executer_pp_clear: Bits<1>) -> &mut Self {
        self.set_field(PPU_S_POINTER_EXECUTER_PP_CLEAR__MASK, unsafe {
            PPU_S_POINTER_EXECUTER_PP_CLEAR(executer_pp_clear.val())
        })
    }

    pub fn executer(&mut self, executer: Bits<1>) -> &mut Self {
        self.set_field(PPU_S_POINTER_EXECUTER__MASK, unsafe {
            PPU_S_POINTER_EXECUTER(executer.val())
        })
    }
}

// ========================================================================
// OPERATION_ENABLE (0x6008)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct PpuOperationEnable;

impl RegisterMeta for PpuOperationEnable {
    const DOMAIN: u32 = target_PPU;
    const OFFSET: u32 = REG_PPU_OPERATION_ENABLE;
}

impl Register<PpuOperationEnable> {
    pub fn op_en(&mut self, op_en: Bits<1>) -> &mut Self {
        self.set_field(PPU_OPERATION_ENABLE_OP_EN__MASK, unsafe {
            PPU_OPERATION_ENABLE_OP_EN(op_en.val())
        })
    }
}

// ========================================================================
// DATA_CUBE_IN_WIDTH (0x600C)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct PpuDataCubeInWidth;

impl RegisterMeta for PpuDataCubeInWidth {
    const DOMAIN: u32 = target_PPU;
    const OFFSET: u32 = REG_PPU_DATA_CUBE_IN_WIDTH;
}

impl Register<PpuDataCubeInWidth> {
    pub fn cube_in_width(&mut self, cube_in_width: Bits<13>) -> &mut Self {
        self.set_field(PPU_DATA_CUBE_IN_WIDTH_CUBE_IN_WIDTH__MASK, unsafe {
            PPU_DATA_CUBE_IN_WIDTH_CUBE_IN_WIDTH(cube_in_width.val())
        })
    }
}

// ========================================================================
// DATA_CUBE_IN_HEIGHT (0x6010)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct PpuDataCubeInHeight;

impl RegisterMeta for PpuDataCubeInHeight {
    const DOMAIN: u32 = target_PPU;
    const OFFSET: u32 = REG_PPU_DATA_CUBE_IN_HEIGHT;
}

impl Register<PpuDataCubeInHeight> {
    pub fn cube_in_height(&mut self, cube_in_height: Bits<13>) -> &mut Self {
        self.set_field(PPU_DATA_CUBE_IN_HEIGHT_CUBE_IN_HEIGHT__MASK, unsafe {
            PPU_DATA_CUBE_IN_HEIGHT_CUBE_IN_HEIGHT(cube_in_height.val())
        })
    }
}

// ========================================================================
// DATA_CUBE_IN_CHANNEL (0x6014)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct PpuDataCubeInChannel;

impl RegisterMeta for PpuDataCubeInChannel {
    const DOMAIN: u32 = target_PPU;
    const OFFSET: u32 = REG_PPU_DATA_CUBE_IN_CHANNEL;
}

impl Register<PpuDataCubeInChannel> {
    pub fn cube_in_channel(&mut self, cube_in_channel: Bits<13>) -> &mut Self {
        self.set_field(PPU_DATA_CUBE_IN_CHANNEL_CUBE_IN_CHANNEL__MASK, unsafe {
            PPU_DATA_CUBE_IN_CHANNEL_CUBE_IN_CHANNEL(cube_in_channel.val())
        })
    }
}

// ========================================================================
// DATA_CUBE_OUT_WIDTH (0x6018)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct PpuDataCubeOutWidth;

impl RegisterMeta for PpuDataCubeOutWidth {
    const DOMAIN: u32 = target_PPU;
    const OFFSET: u32 = REG_PPU_DATA_CUBE_OUT_WIDTH;
}

impl Register<PpuDataCubeOutWidth> {
    pub fn cube_out_width(&mut self, cube_out_width: Bits<13>) -> &mut Self {
        self.set_field(PPU_DATA_CUBE_OUT_WIDTH_CUBE_OUT_WIDTH__MASK, unsafe {
            PPU_DATA_CUBE_OUT_WIDTH_CUBE_OUT_WIDTH(cube_out_width.val())
        })
    }
}

// ========================================================================
// DATA_CUBE_OUT_HEIGHT (0x601C)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct PpuDataCubeOutHeight;

impl RegisterMeta for PpuDataCubeOutHeight {
    const DOMAIN: u32 = target_PPU;
    const OFFSET: u32 = REG_PPU_DATA_CUBE_OUT_HEIGHT;
}

impl Register<PpuDataCubeOutHeight> {
    pub fn cube_out_height(&mut self, cube_out_height: Bits<13>) -> &mut Self {
        self.set_field(PPU_DATA_CUBE_OUT_HEIGHT_CUBE_OUT_HEIGHT__MASK, unsafe {
            PPU_DATA_CUBE_OUT_HEIGHT_CUBE_OUT_HEIGHT(cube_out_height.val())
        })
    }
}

// ========================================================================
// DATA_CUBE_OUT_CHANNEL (0x6020)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct PpuDataCubeOutChannel;

impl RegisterMeta for PpuDataCubeOutChannel {
    const DOMAIN: u32 = target_PPU;
    const OFFSET: u32 = REG_PPU_DATA_CUBE_OUT_CHANNEL;
}

impl Register<PpuDataCubeOutChannel> {
    pub fn cube_out_channel(&mut self, cube_out_channel: Bits<13>) -> &mut Self {
        self.set_field(PPU_DATA_CUBE_OUT_CHANNEL_CUBE_OUT_CHANNEL__MASK, unsafe {
            PPU_DATA_CUBE_OUT_CHANNEL_CUBE_OUT_CHANNEL(cube_out_channel.val())
        })
    }
}

// ========================================================================
// OPERATION_MODE_CFG (0x6024)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct PpuOperationModeCfg;

impl RegisterMeta for PpuOperationModeCfg {
    const DOMAIN: u32 = target_PPU;
    const OFFSET: u32 = REG_PPU_OPERATION_MODE_CFG;
}

impl Register<PpuOperationModeCfg> {
    pub fn pooling_method(&mut self, pooling_method: Bits<2>) -> &mut Self {
        self.set_field(PPU_OPERATION_MODE_CFG_POOLING_METHOD__MASK, unsafe {
            PPU_OPERATION_MODE_CFG_POOLING_METHOD(pooling_method.val())
        })
    }

    pub fn flying_mode(&mut self, flying_mode: Bits<1>) -> &mut Self {
        self.set_field(PPU_OPERATION_MODE_CFG_FLYING_MODE__MASK, unsafe {
            PPU_OPERATION_MODE_CFG_FLYING_MODE(flying_mode.val())
        })
    }

    pub fn use_cnt(&mut self, use_cnt: Bits<3>) -> &mut Self {
        self.set_field(PPU_OPERATION_MODE_CFG_USE_CNT__MASK, unsafe {
            PPU_OPERATION_MODE_CFG_USE_CNT(use_cnt.val())
        })
    }

    pub fn notch_addr(&mut self, notch_addr: Bits<13>) -> &mut Self {
        self.set_field(PPU_OPERATION_MODE_CFG_NOTCH_ADDR__MASK, unsafe {
            PPU_OPERATION_MODE_CFG_NOTCH_ADDR(notch_addr.val())
        })
    }

    pub fn index_en(&mut self, index_en: Bits<1>) -> &mut Self {
        self.set_field(PPU_OPERATION_MODE_CFG_INDEX_EN__MASK, unsafe {
            PPU_OPERATION_MODE_CFG_INDEX_EN(index_en.val())
        })
    }
}

// ========================================================================
// POOLING_KERNEL_CFG (0x6034)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct PpuPoolingKernelCfg;

impl RegisterMeta for PpuPoolingKernelCfg {
    const DOMAIN: u32 = target_PPU;
    const OFFSET: u32 = REG_PPU_POOLING_KERNEL_CFG;
}

impl Register<PpuPoolingKernelCfg> {
    pub fn kernel_width(&mut self, kernel_width: Bits<4>) -> &mut Self {
        self.set_field(PPU_POOLING_KERNEL_CFG_KERNEL_WIDTH__MASK, unsafe {
            PPU_POOLING_KERNEL_CFG_KERNEL_WIDTH(kernel_width.val())
        })
    }

    pub fn kernel_height(&mut self, kernel_height: Bits<4>) -> &mut Self {
        self.set_field(PPU_POOLING_KERNEL_CFG_KERNEL_HEIGHT__MASK, unsafe {
            PPU_POOLING_KERNEL_CFG_KERNEL_HEIGHT(kernel_height.val())
        })
    }

    pub fn kernel_stride_width(&mut self, kernel_stride_width: Bits<4>) -> &mut Self {
        self.set_field(PPU_POOLING_KERNEL_CFG_KERNEL_STRIDE_WIDTH__MASK, unsafe {
            PPU_POOLING_KERNEL_CFG_KERNEL_STRIDE_WIDTH(kernel_stride_width.val())
        })
    }

    pub fn kernel_stride_height(&mut self, kernel_stride_height: Bits<4>) -> &mut Self {
        self.set_field(PPU_POOLING_KERNEL_CFG_KERNEL_STRIDE_HEIGHT__MASK, unsafe {
            PPU_POOLING_KERNEL_CFG_KERNEL_STRIDE_HEIGHT(kernel_stride_height.val())
        })
    }
}

// ========================================================================
// RECIP_KERNEL_WIDTH (0x6038)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct PpuRecipKernelWidth;

impl RegisterMeta for PpuRecipKernelWidth {
    const DOMAIN: u32 = target_PPU;
    const OFFSET: u32 = REG_PPU_RECIP_KERNEL_WIDTH;
}

impl Register<PpuRecipKernelWidth> {
    pub fn recip_kernel_width(&mut self, recip_kernel_width: Bits<17>) -> &mut Self {
        self.set_field(PPU_RECIP_KERNEL_WIDTH_RECIP_KERNEL_WIDTH__MASK, unsafe {
            PPU_RECIP_KERNEL_WIDTH_RECIP_KERNEL_WIDTH(recip_kernel_width.val())
        })
    }
}

// ========================================================================
// RECIP_KERNEL_HEIGHT (0x603C)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct PpuRecipKernelHeight;

impl RegisterMeta for PpuRecipKernelHeight {
    const DOMAIN: u32 = target_PPU;
    const OFFSET: u32 = REG_PPU_RECIP_KERNEL_HEIGHT;
}

impl Register<PpuRecipKernelHeight> {
    pub fn recip_kernel_height(&mut self, recip_kernel_height: Bits<17>) -> &mut Self {
        self.set_field(PPU_RECIP_KERNEL_HEIGHT_RECIP_KERNEL_HEIGHT__MASK, unsafe {
            PPU_RECIP_KERNEL_HEIGHT_RECIP_KERNEL_HEIGHT(recip_kernel_height.val())
        })
    }
}

// ========================================================================
// POOLING_PADDING_CFG (0x6040)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct PpuPoolingPaddingCfg;

impl RegisterMeta for PpuPoolingPaddingCfg {
    const DOMAIN: u32 = target_PPU;
    const OFFSET: u32 = REG_PPU_POOLING_PADDING_CFG;
}

impl Register<PpuPoolingPaddingCfg> {
    pub fn pad_left(&mut self, pad_left: Bits<3>) -> &mut Self {
        self.set_field(PPU_POOLING_PADDING_CFG_PAD_LEFT__MASK, unsafe {
            PPU_POOLING_PADDING_CFG_PAD_LEFT(pad_left.val())
        })
    }

    pub fn pad_top(&mut self, pad_top: Bits<3>) -> &mut Self {
        self.set_field(PPU_POOLING_PADDING_CFG_PAD_TOP__MASK, unsafe {
            PPU_POOLING_PADDING_CFG_PAD_TOP(pad_top.val())
        })
    }

    pub fn pad_right(&mut self, pad_right: Bits<3>) -> &mut Self {
        self.set_field(PPU_POOLING_PADDING_CFG_PAD_RIGHT__MASK, unsafe {
            PPU_POOLING_PADDING_CFG_PAD_RIGHT(pad_right.val())
        })
    }

    pub fn pad_bottom(&mut self, pad_bottom: Bits<3>) -> &mut Self {
        self.set_field(PPU_POOLING_PADDING_CFG_PAD_BOTTOM__MASK, unsafe {
            PPU_POOLING_PADDING_CFG_PAD_BOTTOM(pad_bottom.val())
        })
    }
}

// ========================================================================
// PADDING_VALUE_1_CFG (0x6044)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct PpuPaddingValue1Cfg;

impl RegisterMeta for PpuPaddingValue1Cfg {
    const DOMAIN: u32 = target_PPU;
    const OFFSET: u32 = REG_PPU_PADDING_VALUE_1_CFG;
}

impl Register<PpuPaddingValue1Cfg> {
    pub fn pad_value_0(&mut self, pad_value_0: Bits<32>) -> &mut Self {
        self.set_field(PPU_PADDING_VALUE_1_CFG_PAD_VALUE_0__MASK, unsafe {
            PPU_PADDING_VALUE_1_CFG_PAD_VALUE_0(pad_value_0.val())
        })
    }
}

// ========================================================================
// PADDING_VALUE_2_CFG (0x6048)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct PpuPaddingValue2Cfg;

impl RegisterMeta for PpuPaddingValue2Cfg {
    const DOMAIN: u32 = target_PPU;
    const OFFSET: u32 = REG_PPU_PADDING_VALUE_2_CFG;
}

impl Register<PpuPaddingValue2Cfg> {
    pub fn pad_value_1(&mut self, pad_value_1: Bits<3>) -> &mut Self {
        self.set_field(PPU_PADDING_VALUE_2_CFG_PAD_VALUE_1__MASK, unsafe {
            PPU_PADDING_VALUE_2_CFG_PAD_VALUE_1(pad_value_1.val())
        })
    }
}

// ========================================================================
// DST_BASE_ADDR (0x6070)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct PpuDstBaseAddr;

impl RegisterMeta for PpuDstBaseAddr {
    const DOMAIN: u32 = target_PPU;
    const OFFSET: u32 = REG_PPU_DST_BASE_ADDR;
}

impl Register<PpuDstBaseAddr> {
    pub fn dst_base_addr(&mut self, dst_base_addr: Bits<28>) -> &mut Self {
        self.set_field(PPU_DST_BASE_ADDR_DST_BASE_ADDR__MASK, unsafe {
            PPU_DST_BASE_ADDR_DST_BASE_ADDR(dst_base_addr.val())
        })
    }
}

// ========================================================================
// DST_SURF_STRIDE (0x607C)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct PpuDstSurfStride;

impl RegisterMeta for PpuDstSurfStride {
    const DOMAIN: u32 = target_PPU;
    const OFFSET: u32 = REG_PPU_DST_SURF_STRIDE;
}

impl Register<PpuDstSurfStride> {
    pub fn dst_surf_stride(&mut self, dst_surf_stride: Bits<28>) -> &mut Self {
        self.set_field(PPU_DST_SURF_STRIDE_DST_SURF_STRIDE__MASK, unsafe {
            PPU_DST_SURF_STRIDE_DST_SURF_STRIDE(dst_surf_stride.val())
        })
    }
}

// ========================================================================
// DATA_FORMAT (0x6084)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct PpuDataFormat;

impl RegisterMeta for PpuDataFormat {
    const DOMAIN: u32 = target_PPU;
    const OFFSET: u32 = REG_PPU_DATA_FORMAT;
}

impl Register<PpuDataFormat> {
    pub fn proc_precision(&mut self, proc_precision: Bits<3>) -> &mut Self {
        self.set_field(PPU_DATA_FORMAT_PROC_PRECISION__MASK, unsafe {
            PPU_DATA_FORMAT_PROC_PRECISION(proc_precision.val())
        })
    }

    pub fn dpu_flyin(&mut self, dpu_flyin: Bits<1>) -> &mut Self {
        self.set_field(PPU_DATA_FORMAT_DPU_FLYIN__MASK, unsafe {
            PPU_DATA_FORMAT_DPU_FLYIN(dpu_flyin.val())
        })
    }

    pub fn index_add(&mut self, index_add: Bits<28>) -> &mut Self {
        self.set_field(PPU_DATA_FORMAT_INDEX_ADD__MASK, unsafe {
            PPU_DATA_FORMAT_INDEX_ADD(index_add.val())
        })
    }
}

// ========================================================================
// MISC_CTRL (0x60DC)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct PpuMiscCtrl;

impl RegisterMeta for PpuMiscCtrl {
    const DOMAIN: u32 = target_PPU;
    const OFFSET: u32 = REG_PPU_MISC_CTRL;
}

impl Register<PpuMiscCtrl> {
    pub fn burst_len(&mut self, burst_len: Bits<4>) -> &mut Self {
        self.set_field(PPU_MISC_CTRL_BURST_LEN__MASK, unsafe {
            PPU_MISC_CTRL_BURST_LEN(burst_len.val())
        })
    }

    pub fn nonalign(&mut self, nonalign: Bits<1>) -> &mut Self {
        self.set_field(PPU_MISC_CTRL_NONALIGN__MASK, unsafe {
            PPU_MISC_CTRL_NONALIGN(nonalign.val())
        })
    }

    pub fn mc_surf_out(&mut self, mc_surf_out: Bits<1>) -> &mut Self {
        self.set_field(PPU_MISC_CTRL_MC_SURF_OUT__MASK, unsafe {
            PPU_MISC_CTRL_MC_SURF_OUT(mc_surf_out.val())
        })
    }

    pub fn surf_len(&mut self, surf_len: Bits<16>) -> &mut Self {
        self.set_field(PPU_MISC_CTRL_SURF_LEN__MASK, unsafe {
            PPU_MISC_CTRL_SURF_LEN(surf_len.val())
        })
    }
}
