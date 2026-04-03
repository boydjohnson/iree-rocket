use crate::rocket::{
    builders::{Bits, Register, RegisterMeta},
    registers::*,
};

#[derive(Debug, Clone, Copy)]
pub struct CnaSPointer;

impl RegisterMeta for CnaSPointer {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_CNA;
    const OFFSET: u32 = REG_CNA_S_POINTER;
}

impl Register<CnaSPointer> {
    pub fn pointer(&mut self, pointer: Bits<1>) -> &mut Self {
        self.set_field(CNA_S_POINTER_POINTER__MASK, unsafe {
            CNA_S_POINTER_POINTER(pointer.val())
        })
    }

    pub fn pointer_pp_en(&mut self, pp_en: Bits<1>) -> &mut Self {
        self.set_field(CNA_S_POINTER_POINTER_PP_EN__MASK, unsafe {
            CNA_S_POINTER_POINTER_PP_EN(pp_en.val())
        })
    }

    pub fn executor_pp_en(&mut self, pp_en: Bits<1>) -> &mut Self {
        self.set_field(CNA_S_POINTER_EXECUTER_PP_EN__MASK, unsafe {
            CNA_S_POINTER_EXECUTER_PP_EN(pp_en.val())
        })
    }

    pub fn pointer_pp_mode(&mut self, pp_mode: Bits<1>) -> &mut Self {
        self.set_field(CNA_S_POINTER_POINTER_PP_MODE__MASK, unsafe {
            CNA_S_POINTER_POINTER_PP_MODE(pp_mode.val())
        })
    }

    pub fn pointer_pp_clear(&mut self, pp_clear: Bits<1>) -> &mut Self {
        self.set_field(CNA_S_POINTER_POINTER_PP_CLEAR__MASK, unsafe {
            CNA_S_POINTER_POINTER_PP_CLEAR(pp_clear.val())
        })
    }

    pub fn executer_pp_clear(&mut self, pp_clear: Bits<1>) -> &mut Self {
        self.set_field(CNA_S_POINTER_EXECUTER_PP_CLEAR__MASK, unsafe {
            CNA_S_POINTER_EXECUTER_PP_CLEAR(pp_clear.val())
        })
    }

    pub fn executer(&mut self, executer: Bits<1>) -> &mut Self {
        self.set_field(CNA_S_POINTER_EXECUTER__MASK, unsafe {
            CNA_S_POINTER_EXECUTER(executer.val())
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CnaOperationEnable;

impl RegisterMeta for CnaOperationEnable {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_CNA;
    const OFFSET: u32 = REG_CNA_OPERATION_ENABLE;
}

impl Register<CnaOperationEnable> {
    pub fn op_en(&mut self, op_en: Bits<1>) -> &mut Self {
        self.set_field(CNA_OPERATION_ENABLE_OP_EN__MASK, unsafe {
            CNA_OPERATION_ENABLE_OP_EN(op_en.val())
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CnaConvCon1;

impl RegisterMeta for CnaConvCon1 {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_CNA;
    const OFFSET: u32 = REG_CNA_CONV_CON1;
}

impl Register<CnaConvCon1> {
    pub fn conv_mode(&mut self, conv_mode: Bits<4>) -> &mut Self {
        self.set_field(CNA_CONV_CON1_CONV_MODE__MASK, unsafe {
            CNA_CONV_CON1_CONV_MODE(conv_mode.val())
        })
    }

    pub fn in_precision(&mut self, in_precision: Bits<3>) -> &mut Self {
        self.set_field(CNA_CONV_CON1_IN_PRECISION__MASK, unsafe {
            CNA_CONV_CON1_IN_PRECISION(in_precision.val())
        })
    }

    pub fn proc_precision(&mut self, proc_precision: Bits<3>) -> &mut Self {
        self.set_field(CNA_CONV_CON1_PROC_PRECISION__MASK, unsafe {
            CNA_CONV_CON1_PROC_PRECISION(proc_precision.val())
        })
    }

    pub fn argb_in(&mut self, argb_in: Bits<4>) -> &mut Self {
        self.set_field(CNA_CONV_CON1_ARGB_IN__MASK, unsafe {
            CNA_CONV_CON1_ARGB_IN(argb_in.val())
        })
    }

    pub fn deconv(&mut self, deconv: Bits<1>) -> &mut Self {
        self.set_field(CNA_CONV_CON1_DECONV__MASK, unsafe {
            CNA_CONV_CON1_DECONV(deconv.val())
        })
    }

    pub fn group_line_off(&mut self, group_line_off: Bits<1>) -> &mut Self {
        self.set_field(CNA_CONV_CON1_GROUP_LINE_OFF__MASK, unsafe {
            CNA_CONV_CON1_GROUP_LINE_OFF(group_line_off.val())
        })
    }

    pub fn nonalign_dma(&mut self, nonalign_dma: Bits<1>) -> &mut Self {
        self.set_field(CNA_CONV_CON1_NONALIGN_DMA__MASK, unsafe {
            CNA_CONV_CON1_NONALIGN_DMA(nonalign_dma.val())
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CnaConvCon2;

impl RegisterMeta for CnaConvCon2 {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_CNA;
    const OFFSET: u32 = REG_CNA_CONV_CON2;
}

impl Register<CnaConvCon2> {
    pub fn cmd_fifo_srst(&mut self, cmd_fifo_srst: Bits<1>) -> &mut Self {
        self.set_field(CNA_CONV_CON2_CMD_FIFO_SRST__MASK, unsafe {
            CNA_CONV_CON2_CMD_FIFO_SRST(cmd_fifo_srst.val())
        })
    }

    pub fn csc_do_en(&mut self, csc_do_en: Bits<1>) -> &mut Self {
        self.set_field(CNA_CONV_CON2_CSC_DO_EN__MASK, unsafe {
            CNA_CONV_CON2_CSC_DO_EN(csc_do_en.val())
        })
    }

    pub fn csc_wo_en(&mut self, csc_wo_en: Bits<1>) -> &mut Self {
        self.set_field(CNA_CONV_CON2_CSC_WO_EN__MASK, unsafe {
            CNA_CONV_CON2_CSC_WO_EN(csc_wo_en.val())
        })
    }

    pub fn feature_grains(&mut self, feature_grain: Bits<10>) -> &mut Self {
        self.set_field(CNA_CONV_CON2_FEATURE_GRAINS__MASK, unsafe {
            CNA_CONV_CON2_FEATURE_GRAINS(feature_grain.val())
        })
    }

    pub fn kernel_group(&mut self, kernel_group: Bits<8>) -> &mut Self {
        self.set_field(CNA_CONV_CON2_KERNEL_GROUP__MASK, unsafe {
            CNA_CONV_CON2_KERNEL_GROUP(kernel_group.val())
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CnaConvCon3;

impl RegisterMeta for CnaConvCon3 {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_CNA;
    const OFFSET: u32 = REG_CNA_CONV_CON3;
}

impl Register<CnaConvCon3> {
    pub fn conv_x_stride(&mut self, conv_x_stride: Bits<3>) -> &mut Self {
        self.set_field(CNA_CONV_CON3_CONV_X_STRIDE__MASK, unsafe {
            CNA_CONV_CON3_CONV_X_STRIDE(conv_x_stride.val())
        })
    }

    pub fn conv_y_stride(&mut self, conv_y_stride: Bits<3>) -> &mut Self {
        self.set_field(CNA_CONV_CON3_CONV_Y_STRIDE__MASK, unsafe {
            CNA_CONV_CON3_CONV_Y_STRIDE(conv_y_stride.val())
        })
    }

    pub fn deconv_x_stride(&mut self, deconv_x_stride: Bits<3>) -> &mut Self {
        self.set_field(CNA_CONV_CON3_DECONV_X_STRIDE__MASK, unsafe {
            CNA_CONV_CON3_DECONV_X_STRIDE(deconv_x_stride.val())
        })
    }

    pub fn deconv_y_stride(&mut self, deconv_y_stride: Bits<3>) -> &mut Self {
        self.set_field(CNA_CONV_CON3_DECONV_Y_STRIDE__MASK, unsafe {
            CNA_CONV_CON3_DECONV_Y_STRIDE(deconv_y_stride.val())
        })
    }

    pub fn atrous_x_dilation(&mut self, atrous_x_dilation: Bits<5>) -> &mut Self {
        self.set_field(CNA_CONV_CON3_ATROUS_X_DILATION__MASK, unsafe {
            CNA_CONV_CON3_ATROUS_X_DILATION(atrous_x_dilation.val())
        })
    }

    pub fn atrous_y_dilation(&mut self, atrous_y_dilation: Bits<5>) -> &mut Self {
        self.set_field(CNA_CONV_CON3_ATROUS_Y_DILATION__MASK, unsafe {
            CNA_CONV_CON3_ATROUS_Y_DILATION(atrous_y_dilation.val())
        })
    }

    pub fn nn_mode(&mut self, nn_mode: Bits<3>) -> &mut Self {
        self.set_field(CNA_CONV_CON3_NN_MODE__MASK, unsafe {
            CNA_CONV_CON3_NN_MODE(nn_mode.val())
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CnaDataSize0;

impl RegisterMeta for CnaDataSize0 {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_CNA;
    const OFFSET: u32 = REG_CNA_DATA_SIZE0;
}

impl Register<CnaDataSize0> {
    pub fn datain_width(&mut self, datain_width: Bits<11>) -> &mut Self {
        self.set_field(CNA_DATA_SIZE0_DATAIN_WIDTH__MASK, unsafe {
            CNA_DATA_SIZE0_DATAIN_WIDTH(datain_width.val())
        })
    }

    pub fn datain_height(&mut self, datain_height: Bits<11>) -> &mut Self {
        self.set_field(CNA_DATA_SIZE0_DATAIN_HEIGHT__MASK, unsafe {
            CNA_DATA_SIZE0_DATAIN_HEIGHT(datain_height.val())
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CnaDataSize1;

impl RegisterMeta for CnaDataSize1 {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_CNA;
    const OFFSET: u32 = REG_CNA_DATA_SIZE1;
}

impl Register<CnaDataSize1> {
    pub fn datain_channel(&mut self, datain_channel: Bits<16>) -> &mut Self {
        self.set_field(CNA_DATA_SIZE1_DATAIN_CHANNEL__MASK, unsafe {
            CNA_DATA_SIZE1_DATAIN_CHANNEL(datain_channel.val())
        })
    }

    pub fn datain_channel_real(&mut self, datain_channel_real: Bits<14>) -> &mut Self {
        self.set_field(CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL__MASK, unsafe {
            CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL(datain_channel_real.val())
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CnaDataSize2;

impl RegisterMeta for CnaDataSize2 {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_CNA;
    const OFFSET: u32 = REG_CNA_DATA_SIZE2;
}

impl Register<CnaDataSize2> {
    pub fn dataout_width(&mut self, dataout_width: Bits<11>) -> &mut Self {
        self.set_field(CNA_DATA_SIZE2_DATAOUT_WIDTH__MASK, unsafe {
            CNA_DATA_SIZE2_DATAOUT_WIDTH(dataout_width.val())
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CnaDataSize3;

impl RegisterMeta for CnaDataSize3 {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_CNA;
    const OFFSET: u32 = REG_CNA_DATA_SIZE3;
}

impl Register<CnaDataSize3> {
    pub fn dataout_atomics(&mut self, dataout_atomics: Bits<22>) -> &mut Self {
        self.set_field(CNA_DATA_SIZE3_DATAOUT_ATOMICS__MASK, unsafe {
            CNA_DATA_SIZE3_DATAOUT_ATOMICS(dataout_atomics.val())
        })
    }

    pub fn surf_mode(&mut self, surf_mode: Bits<2>) -> &mut Self {
        self.set_field(CNA_DATA_SIZE3_SURF_MODE__MASK, unsafe {
            CNA_DATA_SIZE3_SURF_MODE(surf_mode.val())
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CnaWeightSize0;

impl RegisterMeta for CnaWeightSize0 {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_CNA;
    const OFFSET: u32 = REG_CNA_WEIGHT_SIZE0;
}

impl Register<CnaWeightSize0> {
    pub fn weight_bytes(&mut self, weight_bytes: Bits<32>) -> &mut Self {
        self.set_field(CNA_WEIGHT_SIZE0_WEIGHT_BYTES__MASK, unsafe {
            CNA_WEIGHT_SIZE0_WEIGHT_BYTES(weight_bytes.val())
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CnaWeightSize1;

impl RegisterMeta for CnaWeightSize1 {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_CNA;
    const OFFSET: u32 = REG_CNA_WEIGHT_SIZE1;
}

impl Register<CnaWeightSize1> {
    pub fn weight_bytes_per_kernel(&mut self, weight_bytes_per_kernel: Bits<19>) -> &mut Self {
        self.set_field(CNA_WEIGHT_SIZE1_WEIGHT_BYTES_PER_KERNEL__MASK, unsafe {
            CNA_WEIGHT_SIZE1_WEIGHT_BYTES_PER_KERNEL(weight_bytes_per_kernel.val())
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CnaWeightSize2;

impl RegisterMeta for CnaWeightSize2 {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_CNA;
    const OFFSET: u32 = REG_CNA_WEIGHT_SIZE2;
}

impl Register<CnaWeightSize2> {
    pub fn weight_width(&mut self, weight_width: Bits<5>) -> &mut Self {
        self.set_field(CNA_WEIGHT_SIZE2_WEIGHT_WIDTH__MASK, unsafe {
            CNA_WEIGHT_SIZE2_WEIGHT_WIDTH(weight_width.val())
        })
    }

    pub fn weight_height(&mut self, weight_height: Bits<5>) -> &mut Self {
        self.set_field(CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT__MASK, unsafe {
            CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT(weight_height.val())
        })
    }

    pub fn weight_kernels(&mut self, weight_kernels: Bits<14>) -> &mut Self {
        self.set_field(CNA_WEIGHT_SIZE2_WEIGHT_KERNELS__MASK, unsafe {
            CNA_WEIGHT_SIZE2_WEIGHT_KERNELS(weight_kernels.val())
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CnaCbufCon0;

impl RegisterMeta for CnaCbufCon0 {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_CNA;
    const OFFSET: u32 = REG_CNA_CBUF_CON0;
}

impl Register<CnaCbufCon0> {
    pub fn weight_reuse(&mut self, weight_reuse: Bits<1>) -> &mut Self {
        self.set_field(CNA_CBUF_CON0_WEIGHT_REUSE__MASK, unsafe {
            CNA_CBUF_CON0_WEIGHT_REUSE(weight_reuse.val())
        })
    }

    pub fn data_reuse(&mut self, data_reuse: Bits<1>) -> &mut Self {
        self.set_field(CNA_CBUF_CON0_DATA_REUSE__MASK, unsafe {
            CNA_CBUF_CON0_DATA_REUSE(data_reuse.val())
        })
    }

    pub fn fc_data_bank(&mut self, fc_data_bank: Bits<3>) -> &mut Self {
        self.set_field(CNA_CBUF_CON0_FC_DATA_BANK__MASK, unsafe {
            CNA_CBUF_CON0_FC_DATA_BANK(fc_data_bank.val())
        })
    }

    pub fn weight_bank(&mut self, weight_bank: Bits<4>) -> &mut Self {
        self.set_field(CNA_CBUF_CON0_WEIGHT_BANK__MASK, unsafe {
            CNA_CBUF_CON0_WEIGHT_BANK(weight_bank.val())
        })
    }

    pub fn data_bank(&mut self, data_bank: Bits<4>) -> &mut Self {
        self.set_field(CNA_CBUF_CON0_DATA_BANK__MASK, unsafe {
            CNA_CBUF_CON0_DATA_BANK(data_bank.val())
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CnaCbufCon1;

impl RegisterMeta for CnaCbufCon1 {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_CNA;
    const OFFSET: u32 = REG_CNA_CBUF_CON1;
}

impl Register<CnaCbufCon1> {
    pub fn data_entries(&mut self, data_entries: Bits<14>) -> &mut Self {
        self.set_field(CNA_CBUF_CON1_DATA_ENTRIES__MASK, unsafe {
            CNA_CBUF_CON1_DATA_ENTRIES(data_entries.val())
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CnaCvtCon0;

impl RegisterMeta for CnaCvtCon0 {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_CNA;
    const OFFSET: u32 = REG_CNA_CVT_CON0;
}

impl Register<CnaCvtCon0> {
    pub fn cvt_bypass(&mut self, cvt_bypass: Bits<1>) -> &mut Self {
        self.set_field(CNA_CVT_CON0_CVT_BYPASS__MASK, unsafe {
            CNA_CVT_CON0_CVT_BYPASS(cvt_bypass.val())
        })
    }

    pub fn cvt_type(&mut self, cvt_type: Bits<1>) -> &mut Self {
        self.set_field(CNA_CVT_CON0_CVT_TYPE__MASK, unsafe {
            CNA_CVT_CON0_CVT_TYPE(cvt_type.val())
        })
    }

    pub fn round_type(&mut self, round_type: Bits<1>) -> &mut Self {
        self.set_field(CNA_CVT_CON0_ROUND_TYPE__MASK, unsafe {
            CNA_CVT_CON0_ROUND_TYPE(round_type.val())
        })
    }

    pub fn data_sign(&mut self, data_sign: Bits<1>) -> &mut Self {
        self.set_field(CNA_CVT_CON0_DATA_SIGN__MASK, unsafe {
            CNA_CVT_CON0_DATA_SIGN(data_sign.val())
        })
    }

    pub fn cvt_truncate_0(&mut self, cvt_truncate_0: Bits<6>) -> &mut Self {
        self.set_field(CNA_CVT_CON0_CVT_TRUNCATE_0__MASK, unsafe {
            CNA_CVT_CON0_CVT_TRUNCATE_0(cvt_truncate_0.val())
        })
    }

    pub fn cvt_truncate_1(&mut self, cvt_truncate_1: Bits<6>) -> &mut Self {
        self.set_field(CNA_CVT_CON0_CVT_TRUNCATE_1__MASK, unsafe {
            CNA_CVT_CON0_CVT_TRUNCATE_1(cvt_truncate_1.val())
        })
    }

    pub fn cvt_truncate_2(&mut self, cvt_truncate_2: Bits<6>) -> &mut Self {
        self.set_field(CNA_CVT_CON0_CVT_TRUNCATE_2__MASK, unsafe {
            CNA_CVT_CON0_CVT_TRUNCATE_2(cvt_truncate_2.val())
        })
    }

    pub fn cvt_truncate_3(&mut self, cvt_truncate_3: Bits<6>) -> &mut Self {
        self.set_field(CNA_CVT_CON0_CVT_TRUNCATE_3__MASK, unsafe {
            CNA_CVT_CON0_CVT_TRUNCATE_3(cvt_truncate_3.val())
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CnaCvtCon1;

impl RegisterMeta for CnaCvtCon1 {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_CNA;
    const OFFSET: u32 = REG_CNA_CVT_CON1;
}

impl Register<CnaCvtCon1> {
    pub fn cvt_offset0(&mut self, cvt_offset0: Bits<16>) -> &mut Self {
        self.set_field(CNA_CVT_CON1_CVT_OFFSET0__MASK, unsafe {
            CNA_CVT_CON1_CVT_OFFSET0(cvt_offset0.val())
        })
    }

    pub fn cvt_scale0(&mut self, cvt_scale0: Bits<16>) -> &mut Self {
        self.set_field(CNA_CVT_CON1_CVT_SCALE0__MASK, unsafe {
            CNA_CVT_CON1_CVT_SCALE0(cvt_scale0.val())
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CnaCvtCon2;

impl RegisterMeta for CnaCvtCon2 {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_CNA;
    const OFFSET: u32 = REG_CNA_CVT_CON2;
}

impl Register<CnaCvtCon2> {
    pub fn cvt_offset1(&mut self, cvt_offset1: Bits<16>) -> &mut Self {
        self.set_field(CNA_CVT_CON2_CVT_OFFSET1__MASK, unsafe {
            CNA_CVT_CON2_CVT_OFFSET1(cvt_offset1.val())
        })
    }

    pub fn cvt_scale1(&mut self, cvt_scale1: Bits<16>) -> &mut Self {
        self.set_field(CNA_CVT_CON2_CVT_SCALE1__MASK, unsafe {
            CNA_CVT_CON2_CVT_SCALE1(cvt_scale1.val())
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CnaCvtCon3;

impl RegisterMeta for CnaCvtCon3 {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_CNA;
    const OFFSET: u32 = REG_CNA_CVT_CON3;
}

impl Register<CnaCvtCon3> {
    pub fn cvt_offset2(&mut self, cvt_offset2: Bits<16>) -> &mut Self {
        self.set_field(CNA_CVT_CON3_CVT_OFFSET2__MASK, unsafe {
            CNA_CVT_CON3_CVT_OFFSET2(cvt_offset2.val())
        })
    }

    pub fn cvt_scale2(&mut self, cvt_scale2: Bits<16>) -> &mut Self {
        self.set_field(CNA_CVT_CON3_CVT_SCALE2__MASK, unsafe {
            CNA_CVT_CON3_CVT_SCALE2(cvt_scale2.val())
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CnaCvtCon4;

impl RegisterMeta for CnaCvtCon4 {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_CNA;
    const OFFSET: u32 = REG_CNA_CVT_CON4;
}

impl Register<CnaCvtCon4> {
    pub fn cvt_offset3(&mut self, cvt_offset3: Bits<16>) -> &mut Self {
        self.set_field(CNA_CVT_CON4_CVT_OFFSET3__MASK, unsafe {
            CNA_CVT_CON4_CVT_OFFSET3(cvt_offset3.val())
        })
    }

    pub fn cvt_scale3(&mut self, cvt_scale3: Bits<16>) -> &mut Self {
        self.set_field(CNA_CVT_CON4_CVT_SCALE3__MASK, unsafe {
            CNA_CVT_CON4_CVT_SCALE3(cvt_scale3.val())
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CnaFcCon0;

impl RegisterMeta for CnaFcCon0 {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_CNA;
    const OFFSET: u32 = REG_CNA_FC_CON0;
}

impl Register<CnaFcCon0> {
    pub fn fc_skip_en(&mut self, fc_skip_en: Bits<1>) -> &mut Self {
        self.set_field(CNA_FC_CON0_FC_SKIP_EN__MASK, unsafe {
            CNA_FC_CON0_FC_SKIP_EN(fc_skip_en.val())
        })
    }

    pub fn fc_skip_data(&mut self, fc_skip_data: Bits<16>) -> &mut Self {
        self.set_field(CNA_FC_CON0_FC_SKIP_DATA__MASK, unsafe {
            CNA_FC_CON0_FC_SKIP_DATA(fc_skip_data.val())
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CnaFcCon1;

impl RegisterMeta for CnaFcCon1 {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_CNA;
    const OFFSET: u32 = REG_CNA_FC_CON1;
}

impl Register<CnaFcCon1> {
    pub fn data_offset(&mut self, data_offset: Bits<17>) -> &mut Self {
        self.set_field(CNA_FC_CON1_DATA_OFFSET__MASK, unsafe {
            CNA_FC_CON1_DATA_OFFSET(data_offset.val())
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CnaPadCon0;

impl RegisterMeta for CnaPadCon0 {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_CNA;
    const OFFSET: u32 = REG_CNA_PAD_CON0;
}

impl Register<CnaPadCon0> {
    pub fn pad_top(&mut self, pad_top: Bits<4>) -> &mut Self {
        self.set_field(CNA_PAD_CON0_PAD_TOP__MASK, unsafe {
            CNA_PAD_CON0_PAD_TOP(pad_top.val())
        })
    }

    pub fn pad_left(&mut self, pad_left: Bits<4>) -> &mut Self {
        self.set_field(CNA_PAD_CON0_PAD_LEFT__MASK, unsafe {
            CNA_PAD_CON0_PAD_LEFT(pad_left.val())
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CnaFeatureDataAddr;

impl RegisterMeta for CnaFeatureDataAddr {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_CNA;
    const OFFSET: u32 = REG_CNA_FEATURE_DATA_ADDR;
}

impl Register<CnaFeatureDataAddr> {
    pub fn feature_base_addr(&mut self, feature_base_addr: Bits<32>) -> &mut Self {
        self.set_field(CNA_FEATURE_DATA_ADDR_FEATURE_BASE_ADDR__MASK, unsafe {
            CNA_FEATURE_DATA_ADDR_FEATURE_BASE_ADDR(feature_base_addr.val())
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CnaFcCon2;

impl RegisterMeta for CnaFcCon2 {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_CNA;
    const OFFSET: u32 = REG_CNA_FC_CON2;
}

impl Register<CnaFcCon2> {
    pub fn weight_offset(&mut self, weight_offset: Bits<17>) -> &mut Self {
        self.set_field(CNA_FC_CON2_WEIGHT_OFFSET__MASK, unsafe {
            CNA_FC_CON2_WEIGHT_OFFSET(weight_offset.val())
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CnaDmaCon0;

impl RegisterMeta for CnaDmaCon0 {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_CNA;
    const OFFSET: u32 = REG_CNA_DMA_CON0;
}

impl Register<CnaDmaCon0> {
    pub fn data_burst_len(&mut self, data_burst_len: Bits<4>) -> &mut Self {
        self.set_field(CNA_DMA_CON0_DATA_BURST_LEN__MASK, unsafe {
            CNA_DMA_CON0_DATA_BURST_LEN(data_burst_len.val())
        })
    }

    pub fn weight_burst_len(&mut self, weight_burst_len: Bits<4>) -> &mut Self {
        self.set_field(CNA_DMA_CON0_WEIGHT_BURST_LEN__MASK, unsafe {
            CNA_DMA_CON0_WEIGHT_BURST_LEN(weight_burst_len.val())
        })
    }

    pub fn ov4k_bypass(&mut self, ov4k_bypass: Bits<1>) -> &mut Self {
        self.set_field(CNA_DMA_CON0_OV4K_BYPASS__MASK, unsafe {
            CNA_DMA_CON0_OV4K_BYPASS(ov4k_bypass.val())
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CnaDmaCon1;

impl RegisterMeta for CnaDmaCon1 {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_CNA;
    const OFFSET: u32 = REG_CNA_DMA_CON1;
}

impl Register<CnaDmaCon1> {
    pub fn line_stride(&mut self, line_stride: Bits<28>) -> &mut Self {
        self.set_field(CNA_DMA_CON1_LINE_STRIDE__MASK, unsafe {
            CNA_DMA_CON1_LINE_STRIDE(line_stride.val())
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CnaDmaCon2;

impl RegisterMeta for CnaDmaCon2 {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_CNA;
    const OFFSET: u32 = REG_CNA_DMA_CON2;
}

impl Register<CnaDmaCon2> {
    pub fn surf_stride(&mut self, surf_stride: Bits<28>) -> &mut Self {
        self.set_field(CNA_DMA_CON2_SURF_STRIDE__MASK, unsafe {
            CNA_DMA_CON2_SURF_STRIDE(surf_stride.val())
        })
    }
}

// ========================================================================
// FC_DATA_SIZE0 (0x1084)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct CnaFcDataSize0;

impl RegisterMeta for CnaFcDataSize0 {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_CNA;
    const OFFSET: u32 = REG_CNA_FC_DATA_SIZE0;
}

impl Register<CnaFcDataSize0> {
    pub fn dma_height(&mut self, dma_height: Bits<11>) -> &mut Self {
        self.set_field(CNA_FC_DATA_SIZE0_DMA_HEIGHT__MASK, unsafe {
            CNA_FC_DATA_SIZE0_DMA_HEIGHT(dma_height.val())
        })
    }

    pub fn dma_width(&mut self, dma_width: Bits<14>) -> &mut Self {
        self.set_field(CNA_FC_DATA_SIZE0_DMA_WIDTH__MASK, unsafe {
            CNA_FC_DATA_SIZE0_DMA_WIDTH(dma_width.val())
        })
    }
}

// ========================================================================
// FC_DATA_SIZE1 (0x1088)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct CnaFcDataSize1;

impl RegisterMeta for CnaFcDataSize1 {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_CNA;
    const OFFSET: u32 = REG_CNA_FC_DATA_SIZE1;
}

impl Register<CnaFcDataSize1> {
    pub fn dma_channel(&mut self, dma_channel: Bits<16>) -> &mut Self {
        self.set_field(CNA_FC_DATA_SIZE1_DMA_CHANNEL__MASK, unsafe {
            CNA_FC_DATA_SIZE1_DMA_CHANNEL(dma_channel.val())
        })
    }
}

// ========================================================================
// CLK_GATE (0x1090)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct CnaClkGate;

impl RegisterMeta for CnaClkGate {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_CNA;
    const OFFSET: u32 = REG_CNA_CLK_GATE;
}

impl Register<CnaClkGate> {
    pub fn cna_feature_disable_clkgate(
        &mut self,
        cna_feature_disable_clkgate: Bits<1>,
    ) -> &mut Self {
        self.set_field(CNA_CLK_GATE_CNA_FEATURE_DISABLE_CLKGATE__MASK, unsafe {
            CNA_CLK_GATE_CNA_FEATURE_DISABLE_CLKGATE(cna_feature_disable_clkgate.val())
        })
    }

    pub fn cna_weight_disable_clkgate(&mut self, cna_weight_disable_clkgate: Bits<1>) -> &mut Self {
        self.set_field(CNA_CLK_GATE_CNA_WEIGHT_DISABLE_CLKGATE__MASK, unsafe {
            CNA_CLK_GATE_CNA_WEIGHT_DISABLE_CLKGATE(cna_weight_disable_clkgate.val())
        })
    }

    pub fn csc_disable_clkgate(&mut self, csc_disable_clkgate: Bits<1>) -> &mut Self {
        self.set_field(CNA_CLK_GATE_CSC_DISABLE_CLKGATE__MASK, unsafe {
            CNA_CLK_GATE_CSC_DISABLE_CLKGATE(csc_disable_clkgate.val())
        })
    }

    pub fn cbuf_cs_disable_clkgate(&mut self, cbuf_cs_disable_clkgate: Bits<1>) -> &mut Self {
        self.set_field(CNA_CLK_GATE_CBUF_CS_DISABLE_CLKGATE__MASK, unsafe {
            CNA_CLK_GATE_CBUF_CS_DISABLE_CLKGATE(cbuf_cs_disable_clkgate.val())
        })
    }
}

// ========================================================================
// DCOMP_CTRL (0x1100)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct CnaDcompCtrl;

impl RegisterMeta for CnaDcompCtrl {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_CNA;
    const OFFSET: u32 = REG_CNA_DCOMP_CTRL;
}

impl Register<CnaDcompCtrl> {
    pub fn decomp_control(&mut self, decomp_control: Bits<3>) -> &mut Self {
        self.set_field(CNA_DCOMP_CTRL_DECOMP_CONTROL__MASK, unsafe {
            CNA_DCOMP_CTRL_DECOMP_CONTROL(decomp_control.val())
        })
    }

    pub fn wt_dec_bypass(&mut self, wt_dec_bypass: Bits<1>) -> &mut Self {
        self.set_field(CNA_DCOMP_CTRL_WT_DEC_BYPASS__MASK, unsafe {
            CNA_DCOMP_CTRL_WT_DEC_BYPASS(wt_dec_bypass.val())
        })
    }
}

// ========================================================================
// DCOMP_REGNUM (0x1104)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct CnaDcompRegnum;

impl RegisterMeta for CnaDcompRegnum {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_CNA;
    const OFFSET: u32 = REG_CNA_DCOMP_REGNUM;
}

impl Register<CnaDcompRegnum> {
    pub fn dcomp_regnum(&mut self, dcomp_regnum: Bits<32>) -> &mut Self {
        self.set_field(CNA_DCOMP_REGNUM_DCOMP_REGNUM__MASK, unsafe {
            CNA_DCOMP_REGNUM_DCOMP_REGNUM(dcomp_regnum.val())
        })
    }
}

// ========================================================================
// DCOMP_ADDR0 (0x1110)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct CnaDcompAddr0;

impl RegisterMeta for CnaDcompAddr0 {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_CNA;
    const OFFSET: u32 = REG_CNA_DCOMP_ADDR0;
}

impl Register<CnaDcompAddr0> {
    pub fn decompress_addr0(&mut self, decompress_addr0: Bits<32>) -> &mut Self {
        self.set_field(CNA_DCOMP_ADDR0_DECOMPRESS_ADDR0__MASK, unsafe {
            CNA_DCOMP_ADDR0_DECOMPRESS_ADDR0(decompress_addr0.val())
        })
    }
}

macro_rules! define_dcomp_amount {
    ($name:ident, $offset:expr, $field_mask:ident, $field_macro:ident) => {
        #[derive(Debug, Clone, Copy)]
        pub struct $name;

        impl RegisterMeta for $name {
            const DOMAIN: u32 = crate::rocket::builders::DOMAIN_CNA;
            const OFFSET: u32 = $offset;
        }

        impl Register<$name> {
            pub fn dcomp_amount(&mut self, dcomp_amount: Bits<32>) -> &mut Self {
                self.set_field($field_mask, unsafe { $field_macro(dcomp_amount.val()) })
            }
        }
    };
}

define_dcomp_amount!(
    CnaDcompAmount0,
    REG_CNA_DCOMP_AMOUNT0,
    CNA_DCOMP_AMOUNT0_DCOMP_AMOUNT0__MASK,
    CNA_DCOMP_AMOUNT0_DCOMP_AMOUNT0
);
define_dcomp_amount!(
    CnaDcompAmount1,
    REG_CNA_DCOMP_AMOUNT1,
    CNA_DCOMP_AMOUNT1_DCOMP_AMOUNT1__MASK,
    CNA_DCOMP_AMOUNT1_DCOMP_AMOUNT1
);
define_dcomp_amount!(
    CnaDcompAmount2,
    REG_CNA_DCOMP_AMOUNT2,
    CNA_DCOMP_AMOUNT2_DCOMP_AMOUNT2__MASK,
    CNA_DCOMP_AMOUNT2_DCOMP_AMOUNT2
);
define_dcomp_amount!(
    CnaDcompAmount3,
    REG_CNA_DCOMP_AMOUNT3,
    CNA_DCOMP_AMOUNT3_DCOMP_AMOUNT3__MASK,
    CNA_DCOMP_AMOUNT3_DCOMP_AMOUNT3
);
define_dcomp_amount!(
    CnaDcompAmount4,
    REG_CNA_DCOMP_AMOUNT4,
    CNA_DCOMP_AMOUNT4_DCOMP_AMOUNT4__MASK,
    CNA_DCOMP_AMOUNT4_DCOMP_AMOUNT4
);
define_dcomp_amount!(
    CnaDcompAmount5,
    REG_CNA_DCOMP_AMOUNT5,
    CNA_DCOMP_AMOUNT5_DCOMP_AMOUNT5__MASK,
    CNA_DCOMP_AMOUNT5_DCOMP_AMOUNT5
);
define_dcomp_amount!(
    CnaDcompAmount6,
    REG_CNA_DCOMP_AMOUNT6,
    CNA_DCOMP_AMOUNT6_DCOMP_AMOUNT6__MASK,
    CNA_DCOMP_AMOUNT6_DCOMP_AMOUNT6
);
define_dcomp_amount!(
    CnaDcompAmount7,
    REG_CNA_DCOMP_AMOUNT7,
    CNA_DCOMP_AMOUNT7_DCOMP_AMOUNT7__MASK,
    CNA_DCOMP_AMOUNT7_DCOMP_AMOUNT7
);
define_dcomp_amount!(
    CnaDcompAmount8,
    REG_CNA_DCOMP_AMOUNT8,
    CNA_DCOMP_AMOUNT8_DCOMP_AMOUNT8__MASK,
    CNA_DCOMP_AMOUNT8_DCOMP_AMOUNT8
);
define_dcomp_amount!(
    CnaDcompAmount9,
    REG_CNA_DCOMP_AMOUNT9,
    CNA_DCOMP_AMOUNT9_DCOMP_AMOUNT9__MASK,
    CNA_DCOMP_AMOUNT9_DCOMP_AMOUNT9
);
define_dcomp_amount!(
    CnaDcompAmount10,
    REG_CNA_DCOMP_AMOUNT10,
    CNA_DCOMP_AMOUNT10_DCOMP_AMOUNT10__MASK,
    CNA_DCOMP_AMOUNT10_DCOMP_AMOUNT10
);
define_dcomp_amount!(
    CnaDcompAmount11,
    REG_CNA_DCOMP_AMOUNT11,
    CNA_DCOMP_AMOUNT11_DCOMP_AMOUNT11__MASK,
    CNA_DCOMP_AMOUNT11_DCOMP_AMOUNT11
);
define_dcomp_amount!(
    CnaDcompAmount12,
    REG_CNA_DCOMP_AMOUNT12,
    CNA_DCOMP_AMOUNT12_DCOMP_AMOUNT12__MASK,
    CNA_DCOMP_AMOUNT12_DCOMP_AMOUNT12
);
define_dcomp_amount!(
    CnaDcompAmount13,
    REG_CNA_DCOMP_AMOUNT13,
    CNA_DCOMP_AMOUNT13_DCOMP_AMOUNT13__MASK,
    CNA_DCOMP_AMOUNT13_DCOMP_AMOUNT13
);
define_dcomp_amount!(
    CnaDcompAmount14,
    REG_CNA_DCOMP_AMOUNT14,
    CNA_DCOMP_AMOUNT14_DCOMP_AMOUNT14__MASK,
    CNA_DCOMP_AMOUNT14_DCOMP_AMOUNT14
);
define_dcomp_amount!(
    CnaDcompAmount15,
    REG_CNA_DCOMP_AMOUNT15,
    CNA_DCOMP_AMOUNT15_DCOMP_AMOUNT15__MASK,
    CNA_DCOMP_AMOUNT15_DCOMP_AMOUNT15
);

#[derive(Debug, Clone, Copy)]
pub struct CnaCvtCon5;

impl RegisterMeta for CnaCvtCon5 {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_CNA;
    const OFFSET: u32 = REG_CNA_CVT_CON5;
}

impl Register<CnaCvtCon5> {
    pub fn per_channel_cvt_en(&mut self, per_channel_cvt_en: Bits<32>) -> &mut Self {
        self.set_field(CNA_CVT_CON5_PER_CHANNEL_CVT_EN__MASK, unsafe {
            CNA_CVT_CON5_PER_CHANNEL_CVT_EN(per_channel_cvt_en.val())
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CnaPadCon1;

impl RegisterMeta for CnaPadCon1 {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_CNA;
    const OFFSET: u32 = REG_CNA_PAD_CON1;
}

impl Register<CnaPadCon1> {
    pub fn pad_value(&mut self, pad_value: Bits<32>) -> &mut Self {
        self.set_field(CNA_PAD_CON1_PAD_VALUE__MASK, unsafe {
            CNA_PAD_CON1_PAD_VALUE(pad_value.val())
        })
    }
}
