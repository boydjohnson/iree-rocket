use crate::rocket::{
    builders::{Bits, Register, RegisterMeta},
    registers::*,
};

// ========================================================================
// S_STATUS (0x4000)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuSStatus;

impl RegisterMeta for DpuSStatus {
    const DOMAIN: u32 = target_DPU;
    const OFFSET: u32 = REG_DPU_S_STATUS;
}

impl Register<DpuSStatus> {
    pub fn status_0(&mut self, status_0: Bits<2>) -> &mut Self {
        self.set_field(DPU_S_STATUS_STATUS_0__MASK, unsafe {
            DPU_S_STATUS_STATUS_0(status_0.val())
        })
    }

    pub fn status_1(&mut self, status_1: Bits<2>) -> &mut Self {
        self.set_field(DPU_S_STATUS_STATUS_1__MASK, unsafe {
            DPU_S_STATUS_STATUS_1(status_1.val())
        })
    }
}

// ========================================================================
// S_POINTER (0x4004)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuSPointer;

impl RegisterMeta for DpuSPointer {
    const DOMAIN: u32 = target_DPU;
    const OFFSET: u32 = REG_DPU_S_POINTER;
}

impl Register<DpuSPointer> {
    pub fn pointer(&mut self, pointer: Bits<1>) -> &mut Self {
        self.set_field(DPU_S_POINTER_POINTER__MASK, unsafe {
            DPU_S_POINTER_POINTER(pointer.val())
        })
    }

    pub fn pointer_pp_en(&mut self, pointer_pp_en: Bits<1>) -> &mut Self {
        self.set_field(DPU_S_POINTER_POINTER_PP_EN__MASK, unsafe {
            DPU_S_POINTER_POINTER_PP_EN(pointer_pp_en.val())
        })
    }

    pub fn executer_pp_en(&mut self, executer_pp_en: Bits<1>) -> &mut Self {
        self.set_field(DPU_S_POINTER_EXECUTER_PP_EN__MASK, unsafe {
            DPU_S_POINTER_EXECUTER_PP_EN(executer_pp_en.val())
        })
    }

    pub fn pointer_pp_mode(&mut self, pointer_pp_mode: Bits<1>) -> &mut Self {
        self.set_field(DPU_S_POINTER_POINTER_PP_MODE__MASK, unsafe {
            DPU_S_POINTER_POINTER_PP_MODE(pointer_pp_mode.val())
        })
    }

    pub fn pointer_pp_clear(&mut self, pointer_pp_clear: Bits<1>) -> &mut Self {
        self.set_field(DPU_S_POINTER_POINTER_PP_CLEAR__MASK, unsafe {
            DPU_S_POINTER_POINTER_PP_CLEAR(pointer_pp_clear.val())
        })
    }

    pub fn executer_pp_clear(&mut self, executer_pp_clear: Bits<1>) -> &mut Self {
        self.set_field(DPU_S_POINTER_EXECUTER_PP_CLEAR__MASK, unsafe {
            DPU_S_POINTER_EXECUTER_PP_CLEAR(executer_pp_clear.val())
        })
    }

    pub fn executer(&mut self, executer: Bits<1>) -> &mut Self {
        self.set_field(DPU_S_POINTER_EXECUTER__MASK, unsafe {
            DPU_S_POINTER_EXECUTER(executer.val())
        })
    }
}

// ========================================================================
// OPERATION_ENABLE (0x4008)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuOperationEnable;

impl RegisterMeta for DpuOperationEnable {
    const DOMAIN: u32 = target_DPU;
    const OFFSET: u32 = REG_DPU_OPERATION_ENABLE;
}

impl Register<DpuOperationEnable> {
    pub fn op_en(&mut self, op_en: Bits<1>) -> &mut Self {
        self.set_field(DPU_OPERATION_ENABLE_OP_EN__MASK, unsafe {
            DPU_OPERATION_ENABLE_OP_EN(op_en.val())
        })
    }
}

// ========================================================================
// FEATURE_MODE_CFG (0x400C)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuFeatureModeCfg;

impl RegisterMeta for DpuFeatureModeCfg {
    const DOMAIN: u32 = target_DPU;
    const OFFSET: u32 = REG_DPU_FEATURE_MODE_CFG;
}

impl Register<DpuFeatureModeCfg> {
    pub fn flying_mode(&mut self, flying_mode: Bits<1>) -> &mut Self {
        self.set_field(DPU_FEATURE_MODE_CFG_FLYING_MODE__MASK, unsafe {
            DPU_FEATURE_MODE_CFG_FLYING_MODE(flying_mode.val())
        })
    }

    pub fn output_mode(&mut self, output_mode: Bits<2>) -> &mut Self {
        self.set_field(DPU_FEATURE_MODE_CFG_OUTPUT_MODE__MASK, unsafe {
            DPU_FEATURE_MODE_CFG_OUTPUT_MODE(output_mode.val())
        })
    }

    pub fn conv_mode(&mut self, conv_mode: Bits<2>) -> &mut Self {
        self.set_field(DPU_FEATURE_MODE_CFG_CONV_MODE__MASK, unsafe {
            DPU_FEATURE_MODE_CFG_CONV_MODE(conv_mode.val())
        })
    }

    pub fn burst_len(&mut self, burst_len: Bits<4>) -> &mut Self {
        self.set_field(DPU_FEATURE_MODE_CFG_BURST_LEN__MASK, unsafe {
            DPU_FEATURE_MODE_CFG_BURST_LEN(burst_len.val())
        })
    }

    pub fn surf_len(&mut self, surf_len: Bits<16>) -> &mut Self {
        self.set_field(DPU_FEATURE_MODE_CFG_SURF_LEN__MASK, unsafe {
            DPU_FEATURE_MODE_CFG_SURF_LEN(surf_len.val())
        })
    }

    pub fn nonalign(&mut self, nonalign: Bits<1>) -> &mut Self {
        self.set_field(DPU_FEATURE_MODE_CFG_NONALIGN__MASK, unsafe {
            DPU_FEATURE_MODE_CFG_NONALIGN(nonalign.val())
        })
    }

    pub fn rgp_type(&mut self, rgp_type: Bits<4>) -> &mut Self {
        self.set_field(DPU_FEATURE_MODE_CFG_RGP_TYPE__MASK, unsafe {
            DPU_FEATURE_MODE_CFG_RGP_TYPE(rgp_type.val())
        })
    }

    pub fn tp_en(&mut self, tp_en: Bits<1>) -> &mut Self {
        self.set_field(DPU_FEATURE_MODE_CFG_TP_EN__MASK, unsafe {
            DPU_FEATURE_MODE_CFG_TP_EN(tp_en.val())
        })
    }

    pub fn comb_use(&mut self, comb_use: Bits<1>) -> &mut Self {
        self.set_field(DPU_FEATURE_MODE_CFG_COMB_USE__MASK, unsafe {
            DPU_FEATURE_MODE_CFG_COMB_USE(comb_use.val())
        })
    }
}

// ========================================================================
// DATA_FORMAT (0x4010)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuDataFormat;

impl RegisterMeta for DpuDataFormat {
    const DOMAIN: u32 = target_DPU;
    const OFFSET: u32 = REG_DPU_DATA_FORMAT;
}

impl Register<DpuDataFormat> {
    pub fn proc_precision(&mut self, proc_precision: Bits<3>) -> &mut Self {
        self.set_field(DPU_DATA_FORMAT_PROC_PRECISION__MASK, unsafe {
            DPU_DATA_FORMAT_PROC_PRECISION(proc_precision.val())
        })
    }

    pub fn mc_surf_out(&mut self, mc_surf_out: Bits<1>) -> &mut Self {
        self.set_field(DPU_DATA_FORMAT_MC_SURF_OUT__MASK, unsafe {
            DPU_DATA_FORMAT_MC_SURF_OUT(mc_surf_out.val())
        })
    }

    pub fn bs_mul_shift_value_neg(&mut self, bs_mul_shift_value_neg: Bits<6>) -> &mut Self {
        self.set_field(DPU_DATA_FORMAT_BS_MUL_SHIFT_VALUE_NEG__MASK, unsafe {
            DPU_DATA_FORMAT_BS_MUL_SHIFT_VALUE_NEG(bs_mul_shift_value_neg.val())
        })
    }

    pub fn bn_mul_shift_value_neg(&mut self, bn_mul_shift_value_neg: Bits<6>) -> &mut Self {
        self.set_field(DPU_DATA_FORMAT_BN_MUL_SHIFT_VALUE_NEG__MASK, unsafe {
            DPU_DATA_FORMAT_BN_MUL_SHIFT_VALUE_NEG(bn_mul_shift_value_neg.val())
        })
    }

    pub fn ew_truncate_neg(&mut self, ew_truncate_neg: Bits<10>) -> &mut Self {
        self.set_field(DPU_DATA_FORMAT_EW_TRUNCATE_NEG__MASK, unsafe {
            DPU_DATA_FORMAT_EW_TRUNCATE_NEG(ew_truncate_neg.val())
        })
    }

    pub fn in_precision(&mut self, in_precision: Bits<3>) -> &mut Self {
        self.set_field(DPU_DATA_FORMAT_IN_PRECISION__MASK, unsafe {
            DPU_DATA_FORMAT_IN_PRECISION(in_precision.val())
        })
    }

    pub fn out_precision(&mut self, out_precision: Bits<3>) -> &mut Self {
        self.set_field(DPU_DATA_FORMAT_OUT_PRECISION__MASK, unsafe {
            DPU_DATA_FORMAT_OUT_PRECISION(out_precision.val())
        })
    }
}

// ========================================================================
// OFFSET_PEND (0x4014)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuOffsetPend;

impl RegisterMeta for DpuOffsetPend {
    const DOMAIN: u32 = target_DPU;
    const OFFSET: u32 = REG_DPU_OFFSET_PEND;
}

impl Register<DpuOffsetPend> {
    pub fn offset_pend(&mut self, offset_pend: Bits<16>) -> &mut Self {
        self.set_field(DPU_OFFSET_PEND_OFFSET_PEND__MASK, unsafe {
            DPU_OFFSET_PEND_OFFSET_PEND(offset_pend.val())
        })
    }
}

// ========================================================================
// DST_BASE_ADDR (0x4020)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuDstBaseAddr;

impl RegisterMeta for DpuDstBaseAddr {
    const DOMAIN: u32 = target_DPU;
    const OFFSET: u32 = REG_DPU_DST_BASE_ADDR;
}

impl Register<DpuDstBaseAddr> {
    pub fn dst_base_addr(&mut self, dst_base_addr: Bits<32>) -> &mut Self {
        self.set_field(DPU_DST_BASE_ADDR_DST_BASE_ADDR__MASK, unsafe {
            DPU_DST_BASE_ADDR_DST_BASE_ADDR(dst_base_addr.val())
        })
    }
}

// ========================================================================
// DST_SURF_STRIDE (0x4024)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuDstSurfStride;

impl RegisterMeta for DpuDstSurfStride {
    const DOMAIN: u32 = target_DPU;
    const OFFSET: u32 = REG_DPU_DST_SURF_STRIDE;
}

impl Register<DpuDstSurfStride> {
    pub fn dst_surf_stride(&mut self, dst_surf_stride: Bits<28>) -> &mut Self {
        self.set_field(DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__MASK, unsafe {
            DPU_DST_SURF_STRIDE_DST_SURF_STRIDE(dst_surf_stride.val())
        })
    }
}

// ========================================================================
// DATA_CUBE_WIDTH (0x4030)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuDataCubeWidth;

impl RegisterMeta for DpuDataCubeWidth {
    const DOMAIN: u32 = target_DPU;
    const OFFSET: u32 = REG_DPU_DATA_CUBE_WIDTH;
}

impl Register<DpuDataCubeWidth> {
    pub fn width(&mut self, width: Bits<13>) -> &mut Self {
        self.set_field(DPU_DATA_CUBE_WIDTH_WIDTH__MASK, unsafe {
            DPU_DATA_CUBE_WIDTH_WIDTH(width.val())
        })
    }
}

// ========================================================================
// DATA_CUBE_HEIGHT (0x4034)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuDataCubeHeight;

impl RegisterMeta for DpuDataCubeHeight {
    const DOMAIN: u32 = target_DPU;
    const OFFSET: u32 = REG_DPU_DATA_CUBE_HEIGHT;
}

impl Register<DpuDataCubeHeight> {
    pub fn height(&mut self, height: Bits<13>) -> &mut Self {
        self.set_field(DPU_DATA_CUBE_HEIGHT_HEIGHT__MASK, unsafe {
            DPU_DATA_CUBE_HEIGHT_HEIGHT(height.val())
        })
    }

    pub fn minmax_ctl(&mut self, minmax_ctl: Bits<3>) -> &mut Self {
        self.set_field(DPU_DATA_CUBE_HEIGHT_MINMAX_CTL__MASK, unsafe {
            DPU_DATA_CUBE_HEIGHT_MINMAX_CTL(minmax_ctl.val())
        })
    }
}

// ========================================================================
// DATA_CUBE_NOTCH_ADDR (0x4038)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuDataCubeNotchAddr;

impl RegisterMeta for DpuDataCubeNotchAddr {
    const DOMAIN: u32 = target_DPU;
    const OFFSET: u32 = REG_DPU_DATA_CUBE_NOTCH_ADDR;
}

impl Register<DpuDataCubeNotchAddr> {
    pub fn notch_addr_0(&mut self, notch_addr_0: Bits<13>) -> &mut Self {
        self.set_field(DPU_DATA_CUBE_NOTCH_ADDR_NOTCH_ADDR_0__MASK, unsafe {
            DPU_DATA_CUBE_NOTCH_ADDR_NOTCH_ADDR_0(notch_addr_0.val())
        })
    }

    pub fn notch_addr_1(&mut self, notch_addr_1: Bits<13>) -> &mut Self {
        self.set_field(DPU_DATA_CUBE_NOTCH_ADDR_NOTCH_ADDR_1__MASK, unsafe {
            DPU_DATA_CUBE_NOTCH_ADDR_NOTCH_ADDR_1(notch_addr_1.val())
        })
    }
}

// ========================================================================
// DATA_CUBE_CHANNEL (0x403C)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuDataCubeChannel;

impl RegisterMeta for DpuDataCubeChannel {
    const DOMAIN: u32 = target_DPU;
    const OFFSET: u32 = REG_DPU_DATA_CUBE_CHANNEL;
}

impl Register<DpuDataCubeChannel> {
    pub fn channel(&mut self, channel: Bits<13>) -> &mut Self {
        self.set_field(DPU_DATA_CUBE_CHANNEL_CHANNEL__MASK, unsafe {
            DPU_DATA_CUBE_CHANNEL_CHANNEL(channel.val())
        })
    }

    pub fn orig_channel(&mut self, orig_channel: Bits<13>) -> &mut Self {
        self.set_field(DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL__MASK, unsafe {
            DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL(orig_channel.val())
        })
    }
}

// ========================================================================
// BS_CFG (0x4040)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuBsCfg;

impl RegisterMeta for DpuBsCfg {
    const DOMAIN: u32 = target_DPU;
    const OFFSET: u32 = REG_DPU_BS_CFG;
}

impl Register<DpuBsCfg> {
    pub fn bs_bypass(&mut self, bs_bypass: Bits<1>) -> &mut Self {
        self.set_field(DPU_BS_CFG_BS_BYPASS__MASK, unsafe {
            DPU_BS_CFG_BS_BYPASS(bs_bypass.val())
        })
    }

    pub fn bs_alu_bypass(&mut self, bs_alu_bypass: Bits<1>) -> &mut Self {
        self.set_field(DPU_BS_CFG_BS_ALU_BYPASS__MASK, unsafe {
            DPU_BS_CFG_BS_ALU_BYPASS(bs_alu_bypass.val())
        })
    }

    pub fn bs_mul_bypass(&mut self, bs_mul_bypass: Bits<1>) -> &mut Self {
        self.set_field(DPU_BS_CFG_BS_MUL_BYPASS__MASK, unsafe {
            DPU_BS_CFG_BS_MUL_BYPASS(bs_mul_bypass.val())
        })
    }

    pub fn bs_mul_prelu(&mut self, bs_mul_prelu: Bits<1>) -> &mut Self {
        self.set_field(DPU_BS_CFG_BS_MUL_PRELU__MASK, unsafe {
            DPU_BS_CFG_BS_MUL_PRELU(bs_mul_prelu.val())
        })
    }

    pub fn bs_relu_bypass(&mut self, bs_relu_bypass: Bits<1>) -> &mut Self {
        self.set_field(DPU_BS_CFG_BS_RELU_BYPASS__MASK, unsafe {
            DPU_BS_CFG_BS_RELU_BYPASS(bs_relu_bypass.val())
        })
    }

    pub fn bs_relux_en(&mut self, bs_relux_en: Bits<1>) -> &mut Self {
        self.set_field(DPU_BS_CFG_BS_RELUX_EN__MASK, unsafe {
            DPU_BS_CFG_BS_RELUX_EN(bs_relux_en.val())
        })
    }

    pub fn bs_alu_src(&mut self, bs_alu_src: Bits<1>) -> &mut Self {
        self.set_field(DPU_BS_CFG_BS_ALU_SRC__MASK, unsafe {
            DPU_BS_CFG_BS_ALU_SRC(bs_alu_src.val())
        })
    }

    pub fn bs_alu_algo(&mut self, bs_alu_algo: Bits<4>) -> &mut Self {
        self.set_field(DPU_BS_CFG_BS_ALU_ALGO__MASK, unsafe {
            DPU_BS_CFG_BS_ALU_ALGO(bs_alu_algo.val())
        })
    }
}

// ========================================================================
// BS_ALU_CFG (0x4044)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuBsAluCfg;

impl RegisterMeta for DpuBsAluCfg {
    const DOMAIN: u32 = target_DPU;
    const OFFSET: u32 = REG_DPU_BS_ALU_CFG;
}

impl Register<DpuBsAluCfg> {
    pub fn bs_alu_operand(&mut self, bs_alu_operand: Bits<32>) -> &mut Self {
        self.set_field(DPU_BS_ALU_CFG_BS_ALU_OPERAND__MASK, unsafe {
            DPU_BS_ALU_CFG_BS_ALU_OPERAND(bs_alu_operand.val())
        })
    }
}

// ========================================================================
// BS_MUL_CFG (0x4048)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuBsMulCfg;

impl RegisterMeta for DpuBsMulCfg {
    const DOMAIN: u32 = target_DPU;
    const OFFSET: u32 = REG_DPU_BS_MUL_CFG;
}

impl Register<DpuBsMulCfg> {
    pub fn bs_mul_src(&mut self, bs_mul_src: Bits<1>) -> &mut Self {
        self.set_field(DPU_BS_MUL_CFG_BS_MUL_SRC__MASK, unsafe {
            DPU_BS_MUL_CFG_BS_MUL_SRC(bs_mul_src.val())
        })
    }

    pub fn bs_truncate_src(&mut self, bs_truncate_src: Bits<1>) -> &mut Self {
        self.set_field(DPU_BS_MUL_CFG_BS_TRUNCATE_SRC__MASK, unsafe {
            DPU_BS_MUL_CFG_BS_TRUNCATE_SRC(bs_truncate_src.val())
        })
    }

    pub fn bs_mul_shift_value(&mut self, bs_mul_shift_value: Bits<6>) -> &mut Self {
        self.set_field(DPU_BS_MUL_CFG_BS_MUL_SHIFT_VALUE__MASK, unsafe {
            DPU_BS_MUL_CFG_BS_MUL_SHIFT_VALUE(bs_mul_shift_value.val())
        })
    }

    pub fn bs_mul_operand(&mut self, bs_mul_operand: Bits<16>) -> &mut Self {
        self.set_field(DPU_BS_MUL_CFG_BS_MUL_OPERAND__MASK, unsafe {
            DPU_BS_MUL_CFG_BS_MUL_OPERAND(bs_mul_operand.val())
        })
    }
}

// ========================================================================
// BS_RELUX_CMP_VALUE (0x404C)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuBsReluxCmpValue;

impl RegisterMeta for DpuBsReluxCmpValue {
    const DOMAIN: u32 = target_DPU;
    const OFFSET: u32 = REG_DPU_BS_RELUX_CMP_VALUE;
}

impl Register<DpuBsReluxCmpValue> {
    pub fn bs_relux_cmp_dat(&mut self, bs_relux_cmp_dat: Bits<32>) -> &mut Self {
        self.set_field(DPU_BS_RELUX_CMP_VALUE_BS_RELUX_CMP_DAT__MASK, unsafe {
            DPU_BS_RELUX_CMP_VALUE_BS_RELUX_CMP_DAT(bs_relux_cmp_dat.val())
        })
    }
}

// ========================================================================
// BS_OW_CFG (0x4050)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuBsOwCfg;

impl RegisterMeta for DpuBsOwCfg {
    const DOMAIN: u32 = target_DPU;
    const OFFSET: u32 = REG_DPU_BS_OW_CFG;
}

impl Register<DpuBsOwCfg> {
    pub fn ow_src(&mut self, ow_src: Bits<1>) -> &mut Self {
        self.set_field(DPU_BS_OW_CFG_OW_SRC__MASK, unsafe {
            DPU_BS_OW_CFG_OW_SRC(ow_src.val())
        })
    }

    pub fn od_bypass(&mut self, od_bypass: Bits<1>) -> &mut Self {
        self.set_field(DPU_BS_OW_CFG_OD_BYPASS__MASK, unsafe {
            DPU_BS_OW_CFG_OD_BYPASS(od_bypass.val())
        })
    }

    pub fn size_e_0(&mut self, size_e_0: Bits<3>) -> &mut Self {
        self.set_field(DPU_BS_OW_CFG_SIZE_E_0__MASK, unsafe {
            DPU_BS_OW_CFG_SIZE_E_0(size_e_0.val())
        })
    }

    pub fn size_e_1(&mut self, size_e_1: Bits<3>) -> &mut Self {
        self.set_field(DPU_BS_OW_CFG_SIZE_E_1__MASK, unsafe {
            DPU_BS_OW_CFG_SIZE_E_1(size_e_1.val())
        })
    }

    pub fn size_e_2(&mut self, size_e_2: Bits<3>) -> &mut Self {
        self.set_field(DPU_BS_OW_CFG_SIZE_E_2__MASK, unsafe {
            DPU_BS_OW_CFG_SIZE_E_2(size_e_2.val())
        })
    }

    pub fn tp_org_en(&mut self, tp_org_en: Bits<1>) -> &mut Self {
        self.set_field(DPU_BS_OW_CFG_TP_ORG_EN__MASK, unsafe {
            DPU_BS_OW_CFG_TP_ORG_EN(tp_org_en.val())
        })
    }

    pub fn rgp_cnter(&mut self, rgp_cnter: Bits<4>) -> &mut Self {
        self.set_field(DPU_BS_OW_CFG_RGP_CNTER__MASK, unsafe {
            DPU_BS_OW_CFG_RGP_CNTER(rgp_cnter.val())
        })
    }
}

// ========================================================================
// BS_OW_OP (0x4054)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuBsOwOp;

impl RegisterMeta for DpuBsOwOp {
    const DOMAIN: u32 = target_DPU;
    const OFFSET: u32 = REG_DPU_BS_OW_OP;
}

impl Register<DpuBsOwOp> {
    pub fn ow_op(&mut self, ow_op: Bits<16>) -> &mut Self {
        self.set_field(DPU_BS_OW_OP_OW_OP__MASK, unsafe {
            DPU_BS_OW_OP_OW_OP(ow_op.val())
        })
    }
}

// ========================================================================
// WDMA_SIZE_0 (0x4058)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuWdmaSize0;

impl RegisterMeta for DpuWdmaSize0 {
    const DOMAIN: u32 = target_DPU;
    const OFFSET: u32 = REG_DPU_WDMA_SIZE_0;
}

impl Register<DpuWdmaSize0> {
    pub fn channel_wdma(&mut self, channel_wdma: Bits<13>) -> &mut Self {
        self.set_field(DPU_WDMA_SIZE_0_CHANNEL_WDMA__MASK, unsafe {
            DPU_WDMA_SIZE_0_CHANNEL_WDMA(channel_wdma.val())
        })
    }

    pub fn size_c_wdma(&mut self, size_c_wdma: Bits<11>) -> &mut Self {
        self.set_field(DPU_WDMA_SIZE_0_SIZE_C_WDMA__MASK, unsafe {
            DPU_WDMA_SIZE_0_SIZE_C_WDMA(size_c_wdma.val())
        })
    }

    pub fn tp_precision(&mut self, tp_precision: Bits<1>) -> &mut Self {
        self.set_field(DPU_WDMA_SIZE_0_TP_PRECISION__MASK, unsafe {
            DPU_WDMA_SIZE_0_TP_PRECISION(tp_precision.val())
        })
    }
}

// ========================================================================
// WDMA_SIZE_1 (0x405C)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuWdmaSize1;

impl RegisterMeta for DpuWdmaSize1 {
    const DOMAIN: u32 = target_DPU;
    const OFFSET: u32 = REG_DPU_WDMA_SIZE_1;
}

impl Register<DpuWdmaSize1> {
    pub fn width_wdma(&mut self, width_wdma: Bits<13>) -> &mut Self {
        self.set_field(DPU_WDMA_SIZE_1_WIDTH_WDMA__MASK, unsafe {
            DPU_WDMA_SIZE_1_WIDTH_WDMA(width_wdma.val())
        })
    }

    pub fn height_wdma(&mut self, height_wdma: Bits<13>) -> &mut Self {
        self.set_field(DPU_WDMA_SIZE_1_HEIGHT_WDMA__MASK, unsafe {
            DPU_WDMA_SIZE_1_HEIGHT_WDMA(height_wdma.val())
        })
    }
}

// ========================================================================
// BN_CFG (0x4060)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuBnCfg;

impl RegisterMeta for DpuBnCfg {
    const DOMAIN: u32 = target_DPU;
    const OFFSET: u32 = REG_DPU_BN_CFG;
}

impl Register<DpuBnCfg> {
    pub fn bn_bypass(&mut self, bn_bypass: Bits<1>) -> &mut Self {
        self.set_field(DPU_BN_CFG_BN_BYPASS__MASK, unsafe {
            DPU_BN_CFG_BN_BYPASS(bn_bypass.val())
        })
    }

    pub fn bn_alu_bypass(&mut self, bn_alu_bypass: Bits<1>) -> &mut Self {
        self.set_field(DPU_BN_CFG_BN_ALU_BYPASS__MASK, unsafe {
            DPU_BN_CFG_BN_ALU_BYPASS(bn_alu_bypass.val())
        })
    }

    pub fn bn_mul_bypass(&mut self, bn_mul_bypass: Bits<1>) -> &mut Self {
        self.set_field(DPU_BN_CFG_BN_MUL_BYPASS__MASK, unsafe {
            DPU_BN_CFG_BN_MUL_BYPASS(bn_mul_bypass.val())
        })
    }

    pub fn bn_mul_prelu(&mut self, bn_mul_prelu: Bits<1>) -> &mut Self {
        self.set_field(DPU_BN_CFG_BN_MUL_PRELU__MASK, unsafe {
            DPU_BN_CFG_BN_MUL_PRELU(bn_mul_prelu.val())
        })
    }

    pub fn bn_relu_bypass(&mut self, bn_relu_bypass: Bits<1>) -> &mut Self {
        self.set_field(DPU_BN_CFG_BN_RELU_BYPASS__MASK, unsafe {
            DPU_BN_CFG_BN_RELU_BYPASS(bn_relu_bypass.val())
        })
    }

    pub fn bn_relux_en(&mut self, bn_relux_en: Bits<1>) -> &mut Self {
        self.set_field(DPU_BN_CFG_BN_RELUX_EN__MASK, unsafe {
            DPU_BN_CFG_BN_RELUX_EN(bn_relux_en.val())
        })
    }

    pub fn bn_alu_src(&mut self, bn_alu_src: Bits<1>) -> &mut Self {
        self.set_field(DPU_BN_CFG_BN_ALU_SRC__MASK, unsafe {
            DPU_BN_CFG_BN_ALU_SRC(bn_alu_src.val())
        })
    }

    pub fn bn_alu_algo(&mut self, bn_alu_algo: Bits<4>) -> &mut Self {
        self.set_field(DPU_BN_CFG_BN_ALU_ALGO__MASK, unsafe {
            DPU_BN_CFG_BN_ALU_ALGO(bn_alu_algo.val())
        })
    }
}

// ========================================================================
// BN_ALU_CFG (0x4064)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuBnAluCfg;

impl RegisterMeta for DpuBnAluCfg {
    const DOMAIN: u32 = target_DPU;
    const OFFSET: u32 = REG_DPU_BN_ALU_CFG;
}

impl Register<DpuBnAluCfg> {
    pub fn bn_alu_operand(&mut self, bn_alu_operand: Bits<32>) -> &mut Self {
        self.set_field(DPU_BN_ALU_CFG_BN_ALU_OPERAND__MASK, unsafe {
            DPU_BN_ALU_CFG_BN_ALU_OPERAND(bn_alu_operand.val())
        })
    }
}

// ========================================================================
// BN_MUL_CFG (0x4068)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuBnMulCfg;

impl RegisterMeta for DpuBnMulCfg {
    const DOMAIN: u32 = target_DPU;
    const OFFSET: u32 = REG_DPU_BN_MUL_CFG;
}

impl Register<DpuBnMulCfg> {
    pub fn bn_mul_src(&mut self, bn_mul_src: Bits<1>) -> &mut Self {
        self.set_field(DPU_BN_MUL_CFG_BN_MUL_SRC__MASK, unsafe {
            DPU_BN_MUL_CFG_BN_MUL_SRC(bn_mul_src.val())
        })
    }

    pub fn bn_truncate_src(&mut self, bn_truncate_src: Bits<1>) -> &mut Self {
        self.set_field(DPU_BN_MUL_CFG_BN_TRUNCATE_SRC__MASK, unsafe {
            DPU_BN_MUL_CFG_BN_TRUNCATE_SRC(bn_truncate_src.val())
        })
    }

    pub fn bn_mul_shift_value(&mut self, bn_mul_shift_value: Bits<6>) -> &mut Self {
        self.set_field(DPU_BN_MUL_CFG_BN_MUL_SHIFT_VALUE__MASK, unsafe {
            DPU_BN_MUL_CFG_BN_MUL_SHIFT_VALUE(bn_mul_shift_value.val())
        })
    }

    pub fn bn_mul_operand(&mut self, bn_mul_operand: Bits<16>) -> &mut Self {
        self.set_field(DPU_BN_MUL_CFG_BN_MUL_OPERAND__MASK, unsafe {
            DPU_BN_MUL_CFG_BN_MUL_OPERAND(bn_mul_operand.val())
        })
    }
}

// ========================================================================
// BN_RELUX_CMP_VALUE (0x406C)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuBnReluxCmpValue;

impl RegisterMeta for DpuBnReluxCmpValue {
    const DOMAIN: u32 = target_DPU;
    const OFFSET: u32 = REG_DPU_BN_RELUX_CMP_VALUE;
}

impl Register<DpuBnReluxCmpValue> {
    pub fn bn_relux_cmp_dat(&mut self, bn_relux_cmp_dat: Bits<32>) -> &mut Self {
        self.set_field(DPU_BN_RELUX_CMP_VALUE_BN_RELUX_CMP_DAT__MASK, unsafe {
            DPU_BN_RELUX_CMP_VALUE_BN_RELUX_CMP_DAT(bn_relux_cmp_dat.val())
        })
    }
}

// ========================================================================
// EW_CFG (0x4070)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuEwCfg;

impl RegisterMeta for DpuEwCfg {
    const DOMAIN: u32 = target_DPU;
    const OFFSET: u32 = REG_DPU_EW_CFG;
}

impl Register<DpuEwCfg> {
    pub fn ew_bypass(&mut self, ew_bypass: Bits<1>) -> &mut Self {
        self.set_field(DPU_EW_CFG_EW_BYPASS__MASK, unsafe {
            DPU_EW_CFG_EW_BYPASS(ew_bypass.val())
        })
    }

    pub fn ew_op_bypass(&mut self, ew_op_bypass: Bits<1>) -> &mut Self {
        self.set_field(DPU_EW_CFG_EW_OP_BYPASS__MASK, unsafe {
            DPU_EW_CFG_EW_OP_BYPASS(ew_op_bypass.val())
        })
    }

    pub fn ew_op_type(&mut self, ew_op_type: Bits<1>) -> &mut Self {
        self.set_field(DPU_EW_CFG_EW_OP_TYPE__MASK, unsafe {
            DPU_EW_CFG_EW_OP_TYPE(ew_op_type.val())
        })
    }

    pub fn ew_mul_prelu(&mut self, ew_mul_prelu: Bits<1>) -> &mut Self {
        self.set_field(DPU_EW_CFG_EW_MUL_PRELU__MASK, unsafe {
            DPU_EW_CFG_EW_MUL_PRELU(ew_mul_prelu.val())
        })
    }

    pub fn ew_op_src(&mut self, ew_op_src: Bits<1>) -> &mut Self {
        self.set_field(DPU_EW_CFG_EW_OP_SRC__MASK, unsafe {
            DPU_EW_CFG_EW_OP_SRC(ew_op_src.val())
        })
    }

    pub fn ew_lut_bypass(&mut self, ew_lut_bypass: Bits<1>) -> &mut Self {
        self.set_field(DPU_EW_CFG_EW_LUT_BYPASS__MASK, unsafe {
            DPU_EW_CFG_EW_LUT_BYPASS(ew_lut_bypass.val())
        })
    }

    pub fn ew_op_cvt_bypass(&mut self, ew_op_cvt_bypass: Bits<1>) -> &mut Self {
        self.set_field(DPU_EW_CFG_EW_OP_CVT_BYPASS__MASK, unsafe {
            DPU_EW_CFG_EW_OP_CVT_BYPASS(ew_op_cvt_bypass.val())
        })
    }

    pub fn ew_relu_bypass(&mut self, ew_relu_bypass: Bits<1>) -> &mut Self {
        self.set_field(DPU_EW_CFG_EW_RELU_BYPASS__MASK, unsafe {
            DPU_EW_CFG_EW_RELU_BYPASS(ew_relu_bypass.val())
        })
    }

    pub fn ew_relux_en(&mut self, ew_relux_en: Bits<1>) -> &mut Self {
        self.set_field(DPU_EW_CFG_EW_RELUX_EN__MASK, unsafe {
            DPU_EW_CFG_EW_RELUX_EN(ew_relux_en.val())
        })
    }

    pub fn ew_alu_algo(&mut self, ew_alu_algo: Bits<4>) -> &mut Self {
        self.set_field(DPU_EW_CFG_EW_ALU_ALGO__MASK, unsafe {
            DPU_EW_CFG_EW_ALU_ALGO(ew_alu_algo.val())
        })
    }

    pub fn ew_binary_en(&mut self, ew_binary_en: Bits<1>) -> &mut Self {
        self.set_field(DPU_EW_CFG_EW_BINARY_EN__MASK, unsafe {
            DPU_EW_CFG_EW_BINARY_EN(ew_binary_en.val())
        })
    }

    pub fn ew_equal_en(&mut self, ew_equal_en: Bits<1>) -> &mut Self {
        self.set_field(DPU_EW_CFG_EW_EQUAL_EN__MASK, unsafe {
            DPU_EW_CFG_EW_EQUAL_EN(ew_equal_en.val())
        })
    }

    pub fn edata_size(&mut self, edata_size: Bits<2>) -> &mut Self {
        self.set_field(DPU_EW_CFG_EDATA_SIZE__MASK, unsafe {
            DPU_EW_CFG_EDATA_SIZE(edata_size.val())
        })
    }

    pub fn ew_data_mode(&mut self, ew_data_mode: Bits<2>) -> &mut Self {
        self.set_field(DPU_EW_CFG_EW_DATA_MODE__MASK, unsafe {
            DPU_EW_CFG_EW_DATA_MODE(ew_data_mode.val())
        })
    }

    pub fn ew_cvt_round(&mut self, ew_cvt_round: Bits<1>) -> &mut Self {
        self.set_field(DPU_EW_CFG_EW_CVT_ROUND__MASK, unsafe {
            DPU_EW_CFG_EW_CVT_ROUND(ew_cvt_round.val())
        })
    }

    pub fn ew_cvt_type(&mut self, ew_cvt_type: Bits<1>) -> &mut Self {
        self.set_field(DPU_EW_CFG_EW_CVT_TYPE__MASK, unsafe {
            DPU_EW_CFG_EW_CVT_TYPE(ew_cvt_type.val())
        })
    }
}

// ========================================================================
// EW_CVT_OFFSET_VALUE (0x4074)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuEwCvtOffsetValue;

impl RegisterMeta for DpuEwCvtOffsetValue {
    const DOMAIN: u32 = target_DPU;
    const OFFSET: u32 = REG_DPU_EW_CVT_OFFSET_VALUE;
}

impl Register<DpuEwCvtOffsetValue> {
    pub fn ew_op_cvt_offset(&mut self, ew_op_cvt_offset: Bits<32>) -> &mut Self {
        self.set_field(DPU_EW_CVT_OFFSET_VALUE_EW_OP_CVT_OFFSET__MASK, unsafe {
            DPU_EW_CVT_OFFSET_VALUE_EW_OP_CVT_OFFSET(ew_op_cvt_offset.val())
        })
    }
}

// ========================================================================
// EW_CVT_SCALE_VALUE (0x4078)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuEwCvtScaleValue;

impl RegisterMeta for DpuEwCvtScaleValue {
    const DOMAIN: u32 = target_DPU;
    const OFFSET: u32 = REG_DPU_EW_CVT_SCALE_VALUE;
}

impl Register<DpuEwCvtScaleValue> {
    pub fn ew_op_cvt_scale(&mut self, ew_op_cvt_scale: Bits<16>) -> &mut Self {
        self.set_field(DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE__MASK, unsafe {
            DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE(ew_op_cvt_scale.val())
        })
    }

    pub fn ew_op_cvt_shift(&mut self, ew_op_cvt_shift: Bits<6>) -> &mut Self {
        self.set_field(DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SHIFT__MASK, unsafe {
            DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SHIFT(ew_op_cvt_shift.val())
        })
    }

    pub fn ew_truncate(&mut self, ew_truncate: Bits<10>) -> &mut Self {
        self.set_field(DPU_EW_CVT_SCALE_VALUE_EW_TRUNCATE__MASK, unsafe {
            DPU_EW_CVT_SCALE_VALUE_EW_TRUNCATE(ew_truncate.val())
        })
    }
}

// ========================================================================
// EW_RELUX_CMP_VALUE (0x407C)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuEwReluxCmpValue;

impl RegisterMeta for DpuEwReluxCmpValue {
    const DOMAIN: u32 = target_DPU;
    const OFFSET: u32 = REG_DPU_EW_RELUX_CMP_VALUE;
}

impl Register<DpuEwReluxCmpValue> {
    pub fn ew_relux_cmp_dat(&mut self, ew_relux_cmp_dat: Bits<32>) -> &mut Self {
        self.set_field(DPU_EW_RELUX_CMP_VALUE_EW_RELUX_CMP_DAT__MASK, unsafe {
            DPU_EW_RELUX_CMP_VALUE_EW_RELUX_CMP_DAT(ew_relux_cmp_dat.val())
        })
    }
}

// ========================================================================
// OUT_CVT_OFFSET (0x4080)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuOutCvtOffset;

impl RegisterMeta for DpuOutCvtOffset {
    const DOMAIN: u32 = target_DPU;
    const OFFSET: u32 = REG_DPU_OUT_CVT_OFFSET;
}

impl Register<DpuOutCvtOffset> {
    pub fn out_cvt_offset(&mut self, out_cvt_offset: Bits<32>) -> &mut Self {
        self.set_field(DPU_OUT_CVT_OFFSET_OUT_CVT_OFFSET__MASK, unsafe {
            DPU_OUT_CVT_OFFSET_OUT_CVT_OFFSET(out_cvt_offset.val())
        })
    }
}

// ========================================================================
// OUT_CVT_SCALE (0x4084)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuOutCvtScale;

impl RegisterMeta for DpuOutCvtScale {
    const DOMAIN: u32 = target_DPU;
    const OFFSET: u32 = REG_DPU_OUT_CVT_SCALE;
}

impl Register<DpuOutCvtScale> {
    pub fn out_cvt_scale(&mut self, out_cvt_scale: Bits<16>) -> &mut Self {
        self.set_field(DPU_OUT_CVT_SCALE_OUT_CVT_SCALE__MASK, unsafe {
            DPU_OUT_CVT_SCALE_OUT_CVT_SCALE(out_cvt_scale.val())
        })
    }

    pub fn fp32tofp16_en(&mut self, fp32tofp16_en: Bits<1>) -> &mut Self {
        self.set_field(DPU_OUT_CVT_SCALE_FP32TOFP16_EN__MASK, unsafe {
            DPU_OUT_CVT_SCALE_FP32TOFP16_EN(fp32tofp16_en.val())
        })
    }
}

// ========================================================================
// OUT_CVT_SHIFT (0x4088)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuOutCvtShift;

impl RegisterMeta for DpuOutCvtShift {
    const DOMAIN: u32 = target_DPU;
    const OFFSET: u32 = REG_DPU_OUT_CVT_SHIFT;
}

impl Register<DpuOutCvtShift> {
    pub fn out_cvt_shift(&mut self, out_cvt_shift: Bits<12>) -> &mut Self {
        self.set_field(DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT__MASK, unsafe {
            DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT(out_cvt_shift.val())
        })
    }

    pub fn minus_exp(&mut self, minus_exp: Bits<8>) -> &mut Self {
        self.set_field(DPU_OUT_CVT_SHIFT_MINUS_EXP__MASK, unsafe {
            DPU_OUT_CVT_SHIFT_MINUS_EXP(minus_exp.val())
        })
    }

    pub fn cvt_round(&mut self, cvt_round: Bits<1>) -> &mut Self {
        self.set_field(DPU_OUT_CVT_SHIFT_CVT_ROUND__MASK, unsafe {
            DPU_OUT_CVT_SHIFT_CVT_ROUND(cvt_round.val())
        })
    }

    pub fn cvt_type(&mut self, cvt_type: Bits<1>) -> &mut Self {
        self.set_field(DPU_OUT_CVT_SHIFT_CVT_TYPE__MASK, unsafe {
            DPU_OUT_CVT_SHIFT_CVT_TYPE(cvt_type.val())
        })
    }
}

// ========================================================================
// EW_OP_VALUE (0x4090 - 0x40AC)
// ========================================================================
macro_rules! define_dpu_ew_op_value {
    ($name:ident, $offset:expr, $field_mask:ident, $field_macro:ident) => {
        #[derive(Debug, Clone, Copy)]
        pub struct $name;

        impl RegisterMeta for $name {
            const DOMAIN: u32 = target_DPU;
            const OFFSET: u32 = $offset;
        }

        impl Register<$name> {
            pub fn ew_operand(&mut self, ew_operand: Bits<32>) -> &mut Self {
                self.set_field($field_mask, unsafe { $field_macro(ew_operand.val()) })
            }
        }
    };
}

define_dpu_ew_op_value!(
    DpuEwOpValue0,
    REG_DPU_EW_OP_VALUE_0,
    DPU_EW_OP_VALUE_0_EW_OPERAND_0__MASK,
    DPU_EW_OP_VALUE_0_EW_OPERAND_0
);
define_dpu_ew_op_value!(
    DpuEwOpValue1,
    REG_DPU_EW_OP_VALUE_1,
    DPU_EW_OP_VALUE_1_EW_OPERAND_1__MASK,
    DPU_EW_OP_VALUE_1_EW_OPERAND_1
);
define_dpu_ew_op_value!(
    DpuEwOpValue2,
    REG_DPU_EW_OP_VALUE_2,
    DPU_EW_OP_VALUE_2_EW_OPERAND_2__MASK,
    DPU_EW_OP_VALUE_2_EW_OPERAND_2
);
define_dpu_ew_op_value!(
    DpuEwOpValue3,
    REG_DPU_EW_OP_VALUE_3,
    DPU_EW_OP_VALUE_3_EW_OPERAND_3__MASK,
    DPU_EW_OP_VALUE_3_EW_OPERAND_3
);
define_dpu_ew_op_value!(
    DpuEwOpValue4,
    REG_DPU_EW_OP_VALUE_4,
    DPU_EW_OP_VALUE_4_EW_OPERAND_4__MASK,
    DPU_EW_OP_VALUE_4_EW_OPERAND_4
);
define_dpu_ew_op_value!(
    DpuEwOpValue5,
    REG_DPU_EW_OP_VALUE_5,
    DPU_EW_OP_VALUE_5_EW_OPERAND_5__MASK,
    DPU_EW_OP_VALUE_5_EW_OPERAND_5
);
define_dpu_ew_op_value!(
    DpuEwOpValue6,
    REG_DPU_EW_OP_VALUE_6,
    DPU_EW_OP_VALUE_6_EW_OPERAND_6__MASK,
    DPU_EW_OP_VALUE_6_EW_OPERAND_6
);
define_dpu_ew_op_value!(
    DpuEwOpValue7,
    REG_DPU_EW_OP_VALUE_7,
    DPU_EW_OP_VALUE_7_EW_OPERAND_7__MASK,
    DPU_EW_OP_VALUE_7_EW_OPERAND_7
);

// ========================================================================
// SURFACE_ADD (0x40C0)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuSurfaceAdd;

impl RegisterMeta for DpuSurfaceAdd {
    const DOMAIN: u32 = target_DPU;
    const OFFSET: u32 = REG_DPU_SURFACE_ADD;
}

impl Register<DpuSurfaceAdd> {
    pub fn surf_add(&mut self, surf_add: Bits<28>) -> &mut Self {
        self.set_field(DPU_SURFACE_ADD_SURF_ADD__MASK, unsafe {
            DPU_SURFACE_ADD_SURF_ADD(surf_add.val())
        })
    }
}

// ========================================================================
// LUT_ACCESS_CFG (0x4100)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuLutAccessCfg;

impl RegisterMeta for DpuLutAccessCfg {
    const DOMAIN: u32 = target_DPU;
    const OFFSET: u32 = REG_DPU_LUT_ACCESS_CFG;
}

impl Register<DpuLutAccessCfg> {
    pub fn lut_addr(&mut self, lut_addr: Bits<10>) -> &mut Self {
        self.set_field(DPU_LUT_ACCESS_CFG_LUT_ADDR__MASK, unsafe {
            DPU_LUT_ACCESS_CFG_LUT_ADDR(lut_addr.val())
        })
    }

    pub fn lut_table_id(&mut self, lut_table_id: Bits<1>) -> &mut Self {
        self.set_field(DPU_LUT_ACCESS_CFG_LUT_TABLE_ID__MASK, unsafe {
            DPU_LUT_ACCESS_CFG_LUT_TABLE_ID(lut_table_id.val())
        })
    }

    pub fn lut_access_type(&mut self, lut_access_type: Bits<1>) -> &mut Self {
        self.set_field(DPU_LUT_ACCESS_CFG_LUT_ACCESS_TYPE__MASK, unsafe {
            DPU_LUT_ACCESS_CFG_LUT_ACCESS_TYPE(lut_access_type.val())
        })
    }
}

// ========================================================================
// LUT_ACCESS_DATA (0x4104)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuLutAccessData;

impl RegisterMeta for DpuLutAccessData {
    const DOMAIN: u32 = target_DPU;
    const OFFSET: u32 = REG_DPU_LUT_ACCESS_DATA;
}

impl Register<DpuLutAccessData> {
    pub fn lut_access_data(&mut self, lut_access_data: Bits<16>) -> &mut Self {
        self.set_field(DPU_LUT_ACCESS_DATA_LUT_ACCESS_DATA__MASK, unsafe {
            DPU_LUT_ACCESS_DATA_LUT_ACCESS_DATA(lut_access_data.val())
        })
    }
}

// ========================================================================
// LUT_CFG (0x4108)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuLutCfg;

impl RegisterMeta for DpuLutCfg {
    const DOMAIN: u32 = target_DPU;
    const OFFSET: u32 = REG_DPU_LUT_CFG;
}

impl Register<DpuLutCfg> {
    pub fn lut_road_sel(&mut self, lut_road_sel: Bits<1>) -> &mut Self {
        self.set_field(DPU_LUT_CFG_LUT_ROAD_SEL__MASK, unsafe {
            DPU_LUT_CFG_LUT_ROAD_SEL(lut_road_sel.val())
        })
    }

    pub fn lut_expand_en(&mut self, lut_expand_en: Bits<1>) -> &mut Self {
        self.set_field(DPU_LUT_CFG_LUT_EXPAND_EN__MASK, unsafe {
            DPU_LUT_CFG_LUT_EXPAND_EN(lut_expand_en.val())
        })
    }

    pub fn lut_lo_le_mux(&mut self, lut_lo_le_mux: Bits<2>) -> &mut Self {
        self.set_field(DPU_LUT_CFG_LUT_LO_LE_MUX__MASK, unsafe {
            DPU_LUT_CFG_LUT_LO_LE_MUX(lut_lo_le_mux.val())
        })
    }

    pub fn lut_uflow_priority(&mut self, lut_uflow_priority: Bits<1>) -> &mut Self {
        self.set_field(DPU_LUT_CFG_LUT_UFLOW_PRIORITY__MASK, unsafe {
            DPU_LUT_CFG_LUT_UFLOW_PRIORITY(lut_uflow_priority.val())
        })
    }

    pub fn lut_oflow_priority(&mut self, lut_oflow_priority: Bits<1>) -> &mut Self {
        self.set_field(DPU_LUT_CFG_LUT_OFLOW_PRIORITY__MASK, unsafe {
            DPU_LUT_CFG_LUT_OFLOW_PRIORITY(lut_oflow_priority.val())
        })
    }

    pub fn lut_hybrid_priority(&mut self, lut_hybrid_priority: Bits<1>) -> &mut Self {
        self.set_field(DPU_LUT_CFG_LUT_HYBRID_PRIORITY__MASK, unsafe {
            DPU_LUT_CFG_LUT_HYBRID_PRIORITY(lut_hybrid_priority.val())
        })
    }

    pub fn lut_cal_sel(&mut self, lut_cal_sel: Bits<1>) -> &mut Self {
        self.set_field(DPU_LUT_CFG_LUT_CAL_SEL__MASK, unsafe {
            DPU_LUT_CFG_LUT_CAL_SEL(lut_cal_sel.val())
        })
    }
}

// ========================================================================
// LUT_INFO (0x410C)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuLutInfo;

impl RegisterMeta for DpuLutInfo {
    const DOMAIN: u32 = target_DPU;
    const OFFSET: u32 = REG_DPU_LUT_INFO;
}

impl Register<DpuLutInfo> {
    pub fn lut_le_index_select(&mut self, lut_le_index_select: Bits<8>) -> &mut Self {
        self.set_field(DPU_LUT_INFO_LUT_LE_INDEX_SELECT__MASK, unsafe {
            DPU_LUT_INFO_LUT_LE_INDEX_SELECT(lut_le_index_select.val())
        })
    }

    pub fn lut_lo_index_select(&mut self, lut_lo_index_select: Bits<8>) -> &mut Self {
        self.set_field(DPU_LUT_INFO_LUT_LO_INDEX_SELECT__MASK, unsafe {
            DPU_LUT_INFO_LUT_LO_INDEX_SELECT(lut_lo_index_select.val())
        })
    }
}

// ========================================================================
// LUT_LE_START (0x4110)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuLutLeStart;

impl RegisterMeta for DpuLutLeStart {
    const DOMAIN: u32 = target_DPU;
    const OFFSET: u32 = REG_DPU_LUT_LE_START;
}

impl Register<DpuLutLeStart> {
    pub fn lut_le_start(&mut self, lut_le_start: Bits<32>) -> &mut Self {
        self.set_field(DPU_LUT_LE_START_LUT_LE_START__MASK, unsafe {
            DPU_LUT_LE_START_LUT_LE_START(lut_le_start.val())
        })
    }
}

// ========================================================================
// LUT_LE_END (0x4114)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuLutLeEnd;

impl RegisterMeta for DpuLutLeEnd {
    const DOMAIN: u32 = target_DPU;
    const OFFSET: u32 = REG_DPU_LUT_LE_END;
}

impl Register<DpuLutLeEnd> {
    pub fn lut_le_end(&mut self, lut_le_end: Bits<32>) -> &mut Self {
        self.set_field(DPU_LUT_LE_END_LUT_LE_END__MASK, unsafe {
            DPU_LUT_LE_END_LUT_LE_END(lut_le_end.val())
        })
    }
}

// ========================================================================
// LUT_LO_START (0x4118)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuLutLoStart;

impl RegisterMeta for DpuLutLoStart {
    const DOMAIN: u32 = target_DPU;
    const OFFSET: u32 = REG_DPU_LUT_LO_START;
}

impl Register<DpuLutLoStart> {
    pub fn lut_lo_start(&mut self, lut_lo_start: Bits<32>) -> &mut Self {
        self.set_field(DPU_LUT_LO_START_LUT_LO_START__MASK, unsafe {
            DPU_LUT_LO_START_LUT_LO_START(lut_lo_start.val())
        })
    }
}

// ========================================================================
// LUT_LO_END (0x411C)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuLutLoEnd;

impl RegisterMeta for DpuLutLoEnd {
    const DOMAIN: u32 = target_DPU;
    const OFFSET: u32 = REG_DPU_LUT_LO_END;
}

impl Register<DpuLutLoEnd> {
    pub fn lut_lo_end(&mut self, lut_lo_end: Bits<32>) -> &mut Self {
        self.set_field(DPU_LUT_LO_END_LUT_LO_END__MASK, unsafe {
            DPU_LUT_LO_END_LUT_LO_END(lut_lo_end.val())
        })
    }
}

// ========================================================================
// LUT_LE_SLOPE_SCALE (0x4120)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuLutLeSlopeScale;

impl RegisterMeta for DpuLutLeSlopeScale {
    const DOMAIN: u32 = target_DPU;
    const OFFSET: u32 = REG_DPU_LUT_LE_SLOPE_SCALE;
}

impl Register<DpuLutLeSlopeScale> {
    pub fn lut_le_slope_uflow_scale(&mut self, lut_le_slope_uflow_scale: Bits<16>) -> &mut Self {
        self.set_field(
            DPU_LUT_LE_SLOPE_SCALE_LUT_LE_SLOPE_UFLOW_SCALE__MASK,
            unsafe {
                DPU_LUT_LE_SLOPE_SCALE_LUT_LE_SLOPE_UFLOW_SCALE(lut_le_slope_uflow_scale.val())
            },
        )
    }

    pub fn lut_le_slope_oflow_scale(&mut self, lut_le_slope_oflow_scale: Bits<16>) -> &mut Self {
        self.set_field(
            DPU_LUT_LE_SLOPE_SCALE_LUT_LE_SLOPE_OFLOW_SCALE__MASK,
            unsafe {
                DPU_LUT_LE_SLOPE_SCALE_LUT_LE_SLOPE_OFLOW_SCALE(lut_le_slope_oflow_scale.val())
            },
        )
    }
}

// ========================================================================
// LUT_LE_SLOPE_SHIFT (0x4124)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuLutLeSlopeShift;

impl RegisterMeta for DpuLutLeSlopeShift {
    const DOMAIN: u32 = target_DPU;
    const OFFSET: u32 = REG_DPU_LUT_LE_SLOPE_SHIFT;
}

impl Register<DpuLutLeSlopeShift> {
    pub fn lut_le_slope_uflow_shift(&mut self, lut_le_slope_uflow_shift: Bits<5>) -> &mut Self {
        self.set_field(
            DPU_LUT_LE_SLOPE_SHIFT_LUT_LE_SLOPE_UFLOW_SHIFT__MASK,
            unsafe {
                DPU_LUT_LE_SLOPE_SHIFT_LUT_LE_SLOPE_UFLOW_SHIFT(lut_le_slope_uflow_shift.val())
            },
        )
    }

    pub fn lut_le_slope_oflow_shift(&mut self, lut_le_slope_oflow_shift: Bits<5>) -> &mut Self {
        self.set_field(
            DPU_LUT_LE_SLOPE_SHIFT_LUT_LE_SLOPE_OFLOW_SHIFT__MASK,
            unsafe {
                DPU_LUT_LE_SLOPE_SHIFT_LUT_LE_SLOPE_OFLOW_SHIFT(lut_le_slope_oflow_shift.val())
            },
        )
    }
}

// ========================================================================
// LUT_LO_SLOPE_SCALE (0x4128)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuLutLoSlopeScale;

impl RegisterMeta for DpuLutLoSlopeScale {
    const DOMAIN: u32 = target_DPU;
    const OFFSET: u32 = REG_DPU_LUT_LO_SLOPE_SCALE;
}

impl Register<DpuLutLoSlopeScale> {
    pub fn lut_lo_slope_uflow_scale(&mut self, lut_lo_slope_uflow_scale: Bits<16>) -> &mut Self {
        self.set_field(
            DPU_LUT_LO_SLOPE_SCALE_LUT_LO_SLOPE_UFLOW_SCALE__MASK,
            unsafe {
                DPU_LUT_LO_SLOPE_SCALE_LUT_LO_SLOPE_UFLOW_SCALE(lut_lo_slope_uflow_scale.val())
            },
        )
    }

    pub fn lut_lo_slope_oflow_scale(&mut self, lut_lo_slope_oflow_scale: Bits<16>) -> &mut Self {
        self.set_field(
            DPU_LUT_LO_SLOPE_SCALE_LUT_LO_SLOPE_OFLOW_SCALE__MASK,
            unsafe {
                DPU_LUT_LO_SLOPE_SCALE_LUT_LO_SLOPE_OFLOW_SCALE(lut_lo_slope_oflow_scale.val())
            },
        )
    }
}

// ========================================================================
// LUT_LO_SLOPE_SHIFT (0x412C)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuLutLoSlopeShift;

impl RegisterMeta for DpuLutLoSlopeShift {
    const DOMAIN: u32 = target_DPU;
    const OFFSET: u32 = REG_DPU_LUT_LO_SLOPE_SHIFT;
}

impl Register<DpuLutLoSlopeShift> {
    pub fn lut_lo_slope_uflow_shift(&mut self, lut_lo_slope_uflow_shift: Bits<5>) -> &mut Self {
        self.set_field(
            DPU_LUT_LO_SLOPE_SHIFT_LUT_LO_SLOPE_UFLOW_SHIFT__MASK,
            unsafe {
                DPU_LUT_LO_SLOPE_SHIFT_LUT_LO_SLOPE_UFLOW_SHIFT(lut_lo_slope_uflow_shift.val())
            },
        )
    }

    pub fn lut_lo_slope_oflow_shift(&mut self, lut_lo_slope_oflow_shift: Bits<5>) -> &mut Self {
        self.set_field(
            DPU_LUT_LO_SLOPE_SHIFT_LUT_LO_SLOPE_OFLOW_SHIFT__MASK,
            unsafe {
                DPU_LUT_LO_SLOPE_SHIFT_LUT_LO_SLOPE_OFLOW_SHIFT(lut_lo_slope_oflow_shift.val())
            },
        )
    }
}
