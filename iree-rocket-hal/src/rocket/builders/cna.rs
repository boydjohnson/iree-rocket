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
    /// Description: Selects which of the two shadow register groups is ready to be applied.
    ///
    /// Bit width: 1
    /// Range of values: 1'd0: Register group 0; 1'd1: Register group 1.
    /// Known limitations: None documented.
    /// Related registers: Works together with `pointer_pp_en`/`pointer_pp_mode` in this
    /// register, and with `cna_operation_enable.op_en` which is shadowed for ping-pong
    /// along with every register after it.
    pub fn pointer(&mut self, pointer: Bits<1>) -> &mut Self {
        self.set_field(CNA_S_POINTER_POINTER__MASK, unsafe {
            CNA_S_POINTER_POINTER(pointer.val())
        })
    }

    /// Description: Enables ping-pong toggling of the register group pointer.
    ///
    /// Bit width: 1
    /// Range of values: 1'd0: Disable; 1'd1: Enable.
    /// Known limitations: The toggle rule itself is chosen by `pointer_pp_mode`.
    /// Related registers: `pointer`, `pointer_pp_mode`, `pointer_pp_clear`; this is the
    /// generic ping-pong pattern reused identically by CORE/DPU/DPU_RDMA/PPU/PPU_RDMA
    /// (TRM §4.2/§5.1).
    pub fn pointer_pp_en(&mut self, pp_en: Bits<1>) -> &mut Self {
        self.set_field(CNA_S_POINTER_POINTER_PP_EN__MASK, unsafe {
            CNA_S_POINTER_POINTER_PP_EN(pp_en.val())
        })
    }

    /// Description: Enables ping-pong toggling of the executer group.
    ///
    /// Bit width: 1
    /// Range of values: 1'd0: Disable; 1'd1: Enable.
    /// Known limitations: None documented.
    /// Related registers: `executer`, `executer_pp_clear`; mirrors `pointer_pp_en` but for
    /// the executer group rather than the register group.
    pub fn executor_pp_en(&mut self, pp_en: Bits<1>) -> &mut Self {
        self.set_field(CNA_S_POINTER_EXECUTER_PP_EN__MASK, unsafe {
            CNA_S_POINTER_EXECUTER_PP_EN(pp_en.val())
        })
    }

    /// Description: Selects the ping-pong toggle rule for the register group pointer.
    ///
    /// Bit width: 1
    /// Range of values: 1'd0: Pointer ping-pong by executer (e.g. if current executer is
    /// 0, next pointer will toggle to 1); 1'd1: Pointer ping-pong by pointer (e.g. if
    /// current pointer is 0, next pointer will toggle to 1).
    /// Known limitations: Only meaningful when `pointer_pp_en` is set.
    /// Related registers: `pointer`, `pointer_pp_en`.
    pub fn pointer_pp_mode(&mut self, pp_mode: Bits<1>) -> &mut Self {
        self.set_field(CNA_S_POINTER_POINTER_PP_MODE__MASK, unsafe {
            CNA_S_POINTER_POINTER_PP_MODE(pp_mode.val())
        })
    }

    /// Description: Clears the register group pointer back to 0.
    ///
    /// Bit width: 1
    /// Range of values: Write 1 to clear pointer to 0; W1C (write-1-to-clear).
    /// Known limitations: None documented.
    /// Related registers: `pointer`, `executer_pp_clear`.
    pub fn pointer_pp_clear(&mut self, pp_clear: Bits<1>) -> &mut Self {
        self.set_field(CNA_S_POINTER_POINTER_PP_CLEAR__MASK, unsafe {
            CNA_S_POINTER_POINTER_PP_CLEAR(pp_clear.val())
        })
    }

    /// Description: Clears the executer group pointer back to 0.
    ///
    /// Bit width: 1
    /// Range of values: Write 1 to clear pointer to 0; W1C (write-1-to-clear).
    /// Known limitations: None documented.
    /// Related registers: `executer`, `pointer_pp_clear`.
    pub fn executer_pp_clear(&mut self, pp_clear: Bits<1>) -> &mut Self {
        self.set_field(CNA_S_POINTER_EXECUTER_PP_CLEAR__MASK, unsafe {
            CNA_S_POINTER_EXECUTER_PP_CLEAR(pp_clear.val())
        })
    }

    /// Description: Selects which executer register group is currently used.
    ///
    /// Bit width: 1
    /// Range of values: 1'd0: Executer group 0; 1'd1: Executer group 1.
    /// Known limitations: None documented.
    /// Related registers: `executor_pp_en`, `executer_pp_clear`; readable back via
    /// `cna_s_status`'s `status_0`/`status_1` idle/operating/pending fields.
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
    /// Description: Triggers the CNA block to begin operating.
    ///
    /// Bit width: 1
    /// Range of values: 1'd0: Disable; 1'd1: Enable.
    /// Known limitations: This register and every register after it in the block are
    /// shadowed for ping-pong operation.
    /// Related registers: `cna_s_pointer` (selects which shadow group this write lands
    /// in); analogous to `core_operation_enable`, `dpu_operation_enable`,
    /// `ppu_operation_enable`, and the block enable bits in `global_operation_enable`.
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
    /// Description: Selects the convolution mode.
    ///
    /// Bit width: 4
    /// Range of values: 2'd0: Direct convolution; 2'd1/2'd2: Reserved; 2'd3: Depthwise
    /// convolution. (TRM enumerates this as a 2-bit field over bits 3:0; see Known
    /// limitations.)
    /// Known limitations: TRM lists this field as bits 3:0 but describes only a 2-bit
    /// enum (conv_mode 2'd0..2'd3); code models it as `Bits<4>` matching the bit range.
    /// Related registers: `dpu_feature_mode_cfg.conv_mode` must track this for
    /// zero-skipping mode (TRM §4.2/Fig 36-5: `conv_mode` must be 3 when zero-skipping is
    /// enabled).
    pub fn conv_mode(&mut self, conv_mode: Bits<4>) -> &mut Self {
        self.set_field(CNA_CONV_CON1_CONV_MODE__MASK, unsafe {
            CNA_CONV_CON1_CONV_MODE(conv_mode.val())
        })
    }

    /// Description: Selects the input data precision.
    ///
    /// Bit width: 3
    /// Range of values: 3'd0: int8; 3'd1: int16; 3'd2: float16; 3'd3: bfloat16; 3'd4/3'd5:
    /// Reserved; 3'd6: int4; 3'd7: tf32.
    /// Known limitations: None documented.
    /// Related registers: `proc_precision` (same enum, controls processing precision
    /// rather than input precision); `core_misc_cfg.proc_precision` and
    /// `dpu_data_format.in_precision`/`proc_precision` use related but larger (6-value)
    /// enums.
    pub fn in_precision(&mut self, in_precision: Bits<3>) -> &mut Self {
        self.set_field(CNA_CONV_CON1_IN_PRECISION__MASK, unsafe {
            CNA_CONV_CON1_IN_PRECISION(in_precision.val())
        })
    }

    /// Description: Selects the internal processing precision.
    ///
    /// Bit width: 3
    /// Range of values: 3'd0: int8; 3'd1: int16; 3'd2: float16; 3'd3: bfloat16; 3'd4/3'd5:
    /// Reserved; 3'd6: int4; 3'd7: tf32.
    /// Known limitations: None documented.
    /// Related registers: `in_precision`; `core_misc_cfg.proc_precision`.
    pub fn proc_precision(&mut self, proc_precision: Bits<3>) -> &mut Self {
        self.set_field(CNA_CONV_CON1_PROC_PRECISION__MASK, unsafe {
            CNA_CONV_CON1_PROC_PRECISION(proc_precision.val())
        })
    }

    /// Description: Non-align channel layer control for ARGB-style input.
    ///
    /// Bit width: 4
    /// Range of values: 4'd8: 1 channel input mode; 4'd9: 2 channel input mode; 4'd10: 3
    /// channel input mode; 4'd11: 4 channel input mode.
    /// Known limitations: None documented.
    /// Related registers: `nonalign_dma` should be enabled alongside ARGB mode per the
    /// TRM's note on that field.
    pub fn argb_in(&mut self, argb_in: Bits<4>) -> &mut Self {
        self.set_field(CNA_CONV_CON1_ARGB_IN__MASK, unsafe {
            CNA_CONV_CON1_ARGB_IN(argb_in.val())
        })
    }

    /// Description: Enables the deconvolution function.
    ///
    /// Bit width: 1
    /// Range of values: 1'd0: Disable; 1'd1: Enable.
    /// Known limitations: None documented.
    /// Related registers: `cna_conv_con3.deconv_x_stride`/`deconv_y_stride` configure the
    /// deconvolution stride when this is enabled.
    pub fn deconv(&mut self, deconv: Bits<1>) -> &mut Self {
        self.set_field(CNA_CONV_CON1_DECONV__MASK, unsafe {
            CNA_CONV_CON1_DECONV(deconv.val())
        })
    }

    /// Description: Disables group line fetch, affecting only line-fetch efficiency.
    ///
    /// Bit width: 1
    /// Range of values: 1'd0: Enable group line fetch; 1'd1: Disable.
    /// Known limitations: This setting only influences line fetch efficiency, not
    /// correctness.
    /// Related registers: None.
    pub fn group_line_off(&mut self, group_line_off: Bits<1>) -> &mut Self {
        self.set_field(CNA_CONV_CON1_GROUP_LINE_OFF__MASK, unsafe {
            CNA_CONV_CON1_GROUP_LINE_OFF(group_line_off.val())
        })
    }

    /// Description: Enables CNA DMA non-align mode, letting DMA fetch feature data
    /// continuously.
    ///
    /// Bit width: 1
    /// Range of values: 1'd0: Disable; 1'd1: Enable (should be enabled under ARGB mode).
    /// Known limitations: TRM recommends enabling this bit under ARGB mode (`argb_in`
    /// set).
    /// Related registers: `argb_in`.
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
    /// Description: Soft-resets the command FIFO.
    ///
    /// Bit width: 1
    /// Range of values: Boolean; reserved for debug purposes.
    /// Known limitations: Reserved for debug purpose per the TRM; not part of normal
    /// operation.
    /// Related registers: None.
    pub fn cmd_fifo_srst(&mut self, cmd_fifo_srst: Bits<1>) -> &mut Self {
        self.set_field(CNA_CONV_CON2_CMD_FIFO_SRST__MASK, unsafe {
            CNA_CONV_CON2_CMD_FIFO_SRST(cmd_fifo_srst.val())
        })
    }

    /// Description: Controls whether the sequence scanner outputs feature data to CORE.
    ///
    /// Bit width: 1
    /// Range of values: 1'd0: Enable csc output feature data to core; 1'd1: Disable.
    /// Known limitations: None documented.
    /// Related registers: `csc_wo_en` (equivalent field for weight data);
    /// `cna_clk_gate.csc_disable_clkgate` gates the same sequence-scan block's clock.
    pub fn csc_do_en(&mut self, csc_do_en: Bits<1>) -> &mut Self {
        self.set_field(CNA_CONV_CON2_CSC_DO_EN__MASK, unsafe {
            CNA_CONV_CON2_CSC_DO_EN(csc_do_en.val())
        })
    }

    /// Description: Controls whether the sequence scanner outputs weight data to CORE.
    ///
    /// Bit width: 1
    /// Range of values: 1'd0: Enable csc output weight data to core; 1'd1: Disable.
    /// Known limitations: None documented.
    /// Related registers: `csc_do_en`; `cna_clk_gate.csc_disable_clkgate`.
    pub fn csc_wo_en(&mut self, csc_wo_en: Bits<1>) -> &mut Self {
        self.set_field(CNA_CONV_CON2_CSC_WO_EN__MASK, unsafe {
            CNA_CONV_CON2_CSC_WO_EN(csc_wo_en.val())
        })
    }

    /// Description: Number of feature data rows that must be buffered before convolution
    /// starts.
    ///
    /// Bit width: 10
    /// Range of values: 0x000-0x3FF; TRM suggests setting this to
    /// `y_stride + weight_height + 1`.
    /// Known limitations: Pass the logical 10-bit field value, not an encoded register
    /// word. The hardware field begins at bit 4, so `feature_grains(Bits::new(0x21))`
    /// produces register value `0x210`.
    /// Related registers: `cna_conv_con3.conv_y_stride`, `cna_weight_size2.weight_height`
    /// (used to derive the suggested value).
    pub fn feature_grains(&mut self, feature_grain: Bits<10>) -> &mut Self {
        self.set_field(CNA_CONV_CON2_FEATURE_GRAINS__MASK, unsafe {
            CNA_CONV_CON2_FEATURE_GRAINS(feature_grain.val())
        })
    }

    /// Description: Number of kernel groups, minus one, to process.
    ///
    /// Bit width: 8
    /// Range of values: 0x00-0xFF. In int8, 32 kernels form 1 group; in int16 or fp16, 16
    /// kernels form 1 group. E.g. for 256 kernels in int8, set to 256/32 - 1 = 15.
    /// Known limitations: The grouping size (32 vs 16 kernels) depends on
    /// `in_precision`/`proc_precision`.
    /// Related registers: `cna_weight_size2.weight_kernels`, `cna_conv_con1.in_precision`.
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
    /// Description: Convolution stride value in the x direction.
    ///
    /// Bit width: 3
    /// Range of values: 0x0-0x7.
    /// Known limitations: None documented.
    /// Related registers: `conv_y_stride`; `feature_grains` suggested value depends on
    /// `conv_y_stride`.
    pub fn conv_x_stride(&mut self, conv_x_stride: Bits<3>) -> &mut Self {
        self.set_field(CNA_CONV_CON3_CONV_X_STRIDE__MASK, unsafe {
            CNA_CONV_CON3_CONV_X_STRIDE(conv_x_stride.val())
        })
    }

    /// Description: Convolution stride value in the y direction.
    ///
    /// Bit width: 3
    /// Range of values: 0x0-0x7.
    /// Known limitations: None documented.
    /// Related registers: `conv_x_stride`; `cna_conv_con2.feature_grains` suggested value
    /// depends on this.
    pub fn conv_y_stride(&mut self, conv_y_stride: Bits<3>) -> &mut Self {
        self.set_field(CNA_CONV_CON3_CONV_Y_STRIDE__MASK, unsafe {
            CNA_CONV_CON3_CONV_Y_STRIDE(conv_y_stride.val())
        })
    }

    /// Description: Deconvolution stride in the x direction, expressed as pad numbers
    /// inserted in the feature map row between 2 pixels.
    ///
    /// Bit width: 3
    /// Range of values: 0x0-0x7.
    /// Known limitations: Only meaningful when `cna_conv_con1.deconv` is enabled.
    /// Related registers: `cna_conv_con1.deconv`, `deconv_y_stride`.
    pub fn deconv_x_stride(&mut self, deconv_x_stride: Bits<3>) -> &mut Self {
        self.set_field(CNA_CONV_CON3_DECONV_X_STRIDE__MASK, unsafe {
            CNA_CONV_CON3_DECONV_X_STRIDE(deconv_x_stride.val())
        })
    }

    /// Description: Deconvolution stride in the y direction, expressed as pad numbers
    /// inserted in the feature map column between 2 pixels.
    ///
    /// Bit width: 3
    /// Range of values: 0x0-0x7.
    /// Known limitations: Only meaningful when `cna_conv_con1.deconv` is enabled.
    /// Related registers: `cna_conv_con1.deconv`, `deconv_x_stride`.
    pub fn deconv_y_stride(&mut self, deconv_y_stride: Bits<3>) -> &mut Self {
        self.set_field(CNA_CONV_CON3_DECONV_Y_STRIDE__MASK, unsafe {
            CNA_CONV_CON3_DECONV_Y_STRIDE(deconv_y_stride.val())
        })
    }

    /// Description: Atrous (dilated) convolution dilation amount in the x direction,
    /// expressed as pad numbers inserted in the feature map row between 2 pixels.
    ///
    /// Bit width: 5
    /// Range of values: 0x00-0x1F. Setting this register value > 0 enables atrous
    /// convolution.
    /// Known limitations: None documented.
    /// Related registers: `atrous_y_dilation`.
    pub fn atrous_x_dilation(&mut self, atrous_x_dilation: Bits<5>) -> &mut Self {
        self.set_field(CNA_CONV_CON3_ATROUS_X_DILATION__MASK, unsafe {
            CNA_CONV_CON3_ATROUS_X_DILATION(atrous_x_dilation.val())
        })
    }

    /// Description: Atrous (dilated) convolution dilation amount in the y direction,
    /// expressed as pad numbers inserted in the feature map column between 2 pixels.
    ///
    /// Bit width: 5
    /// Range of values: 0x00-0x1F. Setting this register value > 0 enables atrous
    /// convolution.
    /// Known limitations: None documented.
    /// Related registers: `atrous_x_dilation`.
    pub fn atrous_y_dilation(&mut self, atrous_y_dilation: Bits<5>) -> &mut Self {
        self.set_field(CNA_CONV_CON3_ATROUS_Y_DILATION__MASK, unsafe {
            CNA_CONV_CON3_ATROUS_Y_DILATION(atrous_y_dilation.val())
        })
    }

    /// Description: Selects the multicore MAC array co-work mode.
    ///
    /// Bit width: 3
    /// Range of values: 3'd0: Int8 mac array 32x32 mode; 3'd1: 64x32 mode; 3'd2: 96x32
    /// mode; 3'd3: Reserved; 3'd4: 32x64 mode; 3'd5: 32x96 mode; 3'd6/3'd7: Reserved.
    /// Known limitations: This register targets multicore mode; keep it at 3'd0 for
    /// single-core mode.
    /// Related registers: None.
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
    /// Description: Input feature data width.
    ///
    /// Bit width: 11
    /// Range of values: 0x000-0x7FF.
    /// Known limitations: None documented.
    /// Related registers: `datain_height`, `dataout_width` (post-convolution width in
    /// `cna_data_size2`), `cna_dma_con1.line_stride`.
    pub fn datain_width(&mut self, datain_width: Bits<11>) -> &mut Self {
        self.set_field(CNA_DATA_SIZE0_DATAIN_WIDTH__MASK, unsafe {
            CNA_DATA_SIZE0_DATAIN_WIDTH(datain_width.val())
        })
    }

    /// Description: Input feature data height.
    ///
    /// Bit width: 11
    /// Range of values: 0x000-0x7FF.
    /// Known limitations: None documented.
    /// Related registers: `datain_width`.
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
    /// Description: Input feature data channel number.
    ///
    /// Bit width: 16
    /// Range of values: 0x0000-0xFFFF. In int8 mode this should be an integer multiple of
    /// 8; in int16/float16 mode, an integer multiple of 4.
    /// Known limitations: When the true channel count is not aligned to that multiple,
    /// use `datain_channel_real` to record the true count instead.
    /// Related registers: `datain_channel_real`.
    pub fn datain_channel(&mut self, datain_channel: Bits<16>) -> &mut Self {
        self.set_field(CNA_DATA_SIZE1_DATAIN_CHANNEL__MASK, unsafe {
            CNA_DATA_SIZE1_DATAIN_CHANNEL(datain_channel.val())
        })
    }

    /// Description: Real (unpadded) input channel count, used when the input channel
    /// count is not an integer multiple of 8 (int8) or 4 (int16/float16).
    ///
    /// Bit width: 14
    /// Range of values: 0x0000-0x3FFF.
    /// Known limitations: Only needed when `datain_channel` is padded up to the required
    /// alignment.
    /// Related registers: `datain_channel`.
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
    /// Description: Data width after convolution.
    ///
    /// Bit width: 11
    /// Range of values: 0x000-0x7FF.
    /// Known limitations: None documented.
    /// Related registers: `cna_data_size0.datain_width`, `cna_data_size3.dataout_atomics`.
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
    /// Description: Total output pixel count (data atomics) after convolution.
    ///
    /// Bit width: 22
    /// Range of values: 0x000000-0x3FFFFF.
    /// Known limitations: None documented.
    /// Related registers: `dataout_width`, `surf_mode`.
    pub fn dataout_atomics(&mut self, dataout_atomics: Bits<22>) -> &mut Self {
        self.set_field(CNA_DATA_SIZE3_DATAOUT_ATOMICS__MASK, unsafe {
            CNA_DATA_SIZE3_DATAOUT_ATOMICS(dataout_atomics.val())
        })
    }

    /// Description: Surface serial mode for output data.
    ///
    /// Bit width: 2
    /// Range of values: 2'd0: 1surf series; 2'd1: 1surf series; 2'd2: 2 surf series;
    /// 2'd3: 4 surf series.
    /// Known limitations: TRM lists both 2'd0 and 2'd1 as "1surf series" verbatim.
    /// Related registers: `dataout_atomics`; `dpu_data_format.mc_surf_out`,
    /// `ppu_misc_ctrl.mc_surf_out` use analogous surface-series settings downstream.
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
    /// Description: Total weight bytes for this convolution.
    ///
    /// Bit width: 32
    /// Range of values: 0x00000000-0xFFFFFFFF.
    /// Known limitations: None documented.
    /// Related registers: `cna_weight_size1.weight_bytes_per_kernel`,
    /// `cna_weight_size2.weight_kernels`.
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
    /// Description: Weight bytes for one kernel.
    ///
    /// Bit width: 19
    /// Range of values: 0x00000-0x7FFFF.
    /// Known limitations: None documented.
    /// Related registers: `cna_weight_size0.weight_bytes`.
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
    /// Description: Kernel width.
    ///
    /// Bit width: 5
    /// Range of values: 0x00-0x1F.
    /// Known limitations: None documented.
    /// Related registers: `weight_height`, `weight_kernels`.
    pub fn weight_width(&mut self, weight_width: Bits<5>) -> &mut Self {
        self.set_field(CNA_WEIGHT_SIZE2_WEIGHT_WIDTH__MASK, unsafe {
            CNA_WEIGHT_SIZE2_WEIGHT_WIDTH(weight_width.val())
        })
    }

    /// Description: Kernel height.
    ///
    /// Bit width: 5
    /// Range of values: 0x00-0x1F.
    /// Known limitations: None documented.
    /// Related registers: `weight_width`; `cna_conv_con2.feature_grains` suggested value
    /// is derived in part from kernel height.
    pub fn weight_height(&mut self, weight_height: Bits<5>) -> &mut Self {
        self.set_field(CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT__MASK, unsafe {
            CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT(weight_height.val())
        })
    }

    /// Description: Number of weight kernels.
    ///
    /// Bit width: 14
    /// Range of values: 0x0000-0x3FFF.
    /// Known limitations: None documented.
    /// Related registers: `cna_conv_con2.kernel_group` (kernels are grouped 32-at-a-time
    /// in int8, 16-at-a-time in int16/fp16).
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
    /// Description: Enables weight data reuse, fetching weight directly from the
    /// internal CBUF buffer instead of re-DMAing from system memory.
    ///
    /// Bit width: 1
    /// Range of values: 1'd0: Disable; 1'd1: Enable data reuse.
    /// Known limitations: None documented.
    /// Related registers: `data_reuse` (equivalent for feature data); `weight_bank`
    /// (which CBUF banks hold the reused weight data).
    pub fn weight_reuse(&mut self, weight_reuse: Bits<1>) -> &mut Self {
        self.set_field(CNA_CBUF_CON0_WEIGHT_REUSE__MASK, unsafe {
            CNA_CBUF_CON0_WEIGHT_REUSE(weight_reuse.val())
        })
    }

    /// Description: Enables feature data reuse, fetching data directly from the internal
    /// CBUF buffer instead of re-DMAing from system memory.
    ///
    /// Bit width: 1
    /// Range of values: 1'd0: Disable; 1'd1: Enable data reuse.
    /// Known limitations: None documented.
    /// Related registers: `weight_reuse`; `data_bank` (which CBUF banks hold the reused
    /// feature data); `data_entries` in `cna_cbuf_con1`.
    pub fn data_reuse(&mut self, data_reuse: Bits<1>) -> &mut Self {
        self.set_field(CNA_CBUF_CON0_DATA_REUSE__MASK, unsafe {
            CNA_CBUF_CON0_DATA_REUSE(data_reuse.val())
        })
    }

    /// Description: Number of CBUF banks reserved for FC zero-skipping feature data.
    ///
    /// Bit width: 3
    /// Range of values: Set to 1 in FC zero-skipping mode; otherwise must be set to 0.
    /// Known limitations: Only meaningful when `cna_fc_con0.fc_skip_en` is enabled.
    /// Related registers: `cna_fc_con0.fc_skip_en`, `data_bank`, `weight_bank`.
    pub fn fc_data_bank(&mut self, fc_data_bank: Bits<3>) -> &mut Self {
        self.set_field(CNA_CBUF_CON0_FC_DATA_BANK__MASK, unsafe {
            CNA_CBUF_CON0_FC_DATA_BANK(fc_data_bank.val())
        })
    }

    /// Description: Number of CBUF banks occupied by weight data, counted downward from
    /// the top bank.
    ///
    /// Bit width: 4
    /// Range of values: 0-15, matching the complete 4-bit hardware field. The older TRM
    /// description only enumerates 1-7, but a working RK3588 vendor convolution capture
    /// programs 11.
    /// Known limitations: Must be allocated so it doesn't overlap the banks claimed by
    /// `data_bank`. The exact bank-count/topology interpretation for values above 7 is
    /// not documented by the older TRM.
    /// Related registers: `data_bank`, `fc_data_bank`, `weight_reuse`.
    pub fn weight_bank(&mut self, weight_bank: Bits<4>) -> &mut Self {
        self.set_field(CNA_CBUF_CON0_WEIGHT_BANK__MASK, unsafe {
            CNA_CBUF_CON0_WEIGHT_BANK(weight_bank.val())
        })
    }

    /// Description: Number of CBUF banks occupied by feature data, counted upward from
    /// bank 0.
    ///
    /// Bit width: 4
    /// Range of values: 4'd0: Bank 0 occupied by feature data; 4'd1: Bank 0 and bank 1
    /// occupied by feature data; 4'd2: Bank 0/1/2 occupied by feature data; ... 4'd6:
    /// Bank 0-6 occupied by feature data.
    /// Known limitations: Must be allocated so it doesn't overlap the banks claimed by
    /// `weight_bank`.
    /// Related registers: `weight_bank`, `fc_data_bank`, `data_reuse`,
    /// `cna_cbuf_con1.data_entries`.
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
    /// Description: Number of CBUF bank-spaces needed to store one feature map row.
    ///
    /// Bit width: 13
    /// Range of values: 0x0000-0x1FFF, per the TRM's bit table (bits 12:0; bits 31:13
    /// reserved).
    /// Known limitations: The compiled register mask (`CNA_CBUF_CON1_DATA_ENTRIES__MASK`,
    /// from Mesa's `rkt_registers.h`) is actually 0x3fff (14 bits) — one bit wider than
    /// the TRM's own reserved-bit boundary. `Bits<13>` here is a deliberately tighter
    /// caller-side assertion matching the TRM table; it can never write a value that the
    /// wider hardware mask wouldn't also have accepted, so this only forbids the one
    /// TRM-reserved bit (12 vs 13) without changing what's actually written to hardware.
    /// Related registers: `cna_cbuf_con0.data_bank`.
    pub fn data_entries(&mut self, data_entries: Bits<13>) -> &mut Self {
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
    /// Description: Bypasses the input convert (CVT) function entirely.
    ///
    /// Bit width: 1
    /// Range of values: 1'd0: Enable CVT function; 1'd1: Disable CVT function.
    /// Known limitations: None documented.
    /// Related registers: `cvt_type`, `round_type`, `data_sign`, the `cvt_truncate_*` and
    /// `cvt_scale*`/`cvt_offset*` fields, `cna_cvt_con5.per_channel_cvt_en`.
    pub fn cvt_bypass(&mut self, cvt_bypass: Bits<1>) -> &mut Self {
        self.set_field(CNA_CVT_CON0_CVT_BYPASS__MASK, unsafe {
            CNA_CVT_CON0_CVT_BYPASS(cvt_bypass.val())
        })
    }

    /// Description: Selects the calculation order of the input convert function.
    ///
    /// Bit width: 1
    /// Range of values: 1'd0: Multiply first, then add; 1'd1: CVT function will do add
    /// first, then multiply.
    /// Known limitations: Only meaningful when `cvt_bypass` is not set.
    /// Related registers: `cvt_bypass`, `cvt_scale*`/`cvt_offset*` (the multiply/add
    /// operands).
    pub fn cvt_type(&mut self, cvt_type: Bits<1>) -> &mut Self {
        self.set_field(CNA_CVT_CON0_CVT_TYPE__MASK, unsafe {
            CNA_CVT_CON0_CVT_TYPE(cvt_type.val())
        })
    }

    /// Description: Selects the rounding rule used by the input convert.
    ///
    /// Bit width: 1
    /// Range of values: 1'd0: Odd in, even not; 1'd1: Round-up 0.5 to 1.
    /// Known limitations: None documented.
    /// Related registers: `core_clip_truncate.round_type`,
    /// `dpu_out_cvt_offset/scale/shift.cvt_round` use an analogous rounding choice
    /// downstream.
    pub fn round_type(&mut self, round_type: Bits<1>) -> &mut Self {
        self.set_field(CNA_CVT_CON0_ROUND_TYPE__MASK, unsafe {
            CNA_CVT_CON0_ROUND_TYPE(round_type.val())
        })
    }

    /// Description: Selects whether feature data is treated as signed or unsigned.
    ///
    /// Bit width: 1
    /// Range of values: 1'd0: Unsigned; 1'd1: Signed.
    /// Known limitations: Project history notes an attempted hardware fix flipping this
    /// bit for an int8 identity-weight precision bug made results worse and was reverted
    /// — the correct interaction of this bit with int8 data is still open.
    /// Related registers: `cvt_bypass`; `cna_conv_con1.in_precision`.
    pub fn data_sign(&mut self, data_sign: Bits<1>) -> &mut Self {
        self.set_field(CNA_CVT_CON0_DATA_SIGN__MASK, unsafe {
            CNA_CVT_CON0_DATA_SIGN(data_sign.val())
        })
    }

    /// Description: CVT truncate value for the 1st channel.
    ///
    /// Bit width: 6
    /// Range of values: 0x00-0x3F.
    /// Known limitations: None documented.
    /// Related registers: `cvt_truncate_1`, `cvt_truncate_2`, `cvt_truncate_3`;
    /// `cna_cvt_con1.cvt_scale0`/`cvt_offset0`.
    pub fn cvt_truncate_0(&mut self, cvt_truncate_0: Bits<6>) -> &mut Self {
        self.set_field(CNA_CVT_CON0_CVT_TRUNCATE_0__MASK, unsafe {
            CNA_CVT_CON0_CVT_TRUNCATE_0(cvt_truncate_0.val())
        })
    }

    /// Description: CVT truncate value for the 2nd channel.
    ///
    /// Bit width: 6
    /// Range of values: 0x00-0x3F.
    /// Known limitations: None documented.
    /// Related registers: `cvt_truncate_0`, `cvt_truncate_2`, `cvt_truncate_3`;
    /// `cna_cvt_con2.cvt_scale1`/`cvt_offset1`.
    pub fn cvt_truncate_1(&mut self, cvt_truncate_1: Bits<6>) -> &mut Self {
        self.set_field(CNA_CVT_CON0_CVT_TRUNCATE_1__MASK, unsafe {
            CNA_CVT_CON0_CVT_TRUNCATE_1(cvt_truncate_1.val())
        })
    }

    /// Description: CVT truncate value for the 3rd channel.
    ///
    /// Bit width: 6
    /// Range of values: 0x00-0x3F.
    /// Known limitations: None documented.
    /// Related registers: `cvt_truncate_0`, `cvt_truncate_1`, `cvt_truncate_3`;
    /// `cna_cvt_con3.cvt_scale2`/`cvt_offset2`.
    pub fn cvt_truncate_2(&mut self, cvt_truncate_2: Bits<6>) -> &mut Self {
        self.set_field(CNA_CVT_CON0_CVT_TRUNCATE_2__MASK, unsafe {
            CNA_CVT_CON0_CVT_TRUNCATE_2(cvt_truncate_2.val())
        })
    }

    /// Description: CVT truncate value for the 4th channel.
    ///
    /// Bit width: 6
    /// Range of values: 0x00-0x3F.
    /// Known limitations: None documented.
    /// Related registers: `cvt_truncate_0`, `cvt_truncate_1`, `cvt_truncate_2`;
    /// `cna_cvt_con4.cvt_scale3`/`cvt_offset3`.
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
    /// Description: CVT adder operand for the 1st channel.
    ///
    /// Bit width: 16
    /// Range of values: 0x0000-0xFFFF.
    /// Known limitations: None documented.
    /// Related registers: `cvt_scale0`; `cna_cvt_con0.cvt_truncate_0`.
    pub fn cvt_offset0(&mut self, cvt_offset0: Bits<16>) -> &mut Self {
        self.set_field(CNA_CVT_CON1_CVT_OFFSET0__MASK, unsafe {
            CNA_CVT_CON1_CVT_OFFSET0(cvt_offset0.val())
        })
    }

    /// Description: CVT multiplier operand for the 1st channel.
    ///
    /// Bit width: 16
    /// Range of values: 0x0000-0xFFFF.
    /// Known limitations: None documented.
    /// Related registers: `cvt_offset0`; `cna_cvt_con0.cvt_type` (order of multiply vs
    /// add).
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
    /// Description: CVT adder operand for the 2nd channel.
    ///
    /// Bit width: 16
    /// Range of values: 0x0000-0xFFFF.
    /// Known limitations: None documented.
    /// Related registers: `cvt_scale1`; `cna_cvt_con0.cvt_truncate_1`.
    pub fn cvt_offset1(&mut self, cvt_offset1: Bits<16>) -> &mut Self {
        self.set_field(CNA_CVT_CON2_CVT_OFFSET1__MASK, unsafe {
            CNA_CVT_CON2_CVT_OFFSET1(cvt_offset1.val())
        })
    }

    /// Description: CVT multiplier operand for the 2nd channel.
    ///
    /// Bit width: 16
    /// Range of values: 0x0000-0xFFFF.
    /// Known limitations: None documented.
    /// Related registers: `cvt_offset1`; `cna_cvt_con0.cvt_type`.
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
    /// Description: CVT adder operand for the 3rd channel.
    ///
    /// Bit width: 16
    /// Range of values: 0x0000-0xFFFF.
    /// Known limitations: None documented.
    /// Related registers: `cvt_scale2`; `cna_cvt_con0.cvt_truncate_2`.
    pub fn cvt_offset2(&mut self, cvt_offset2: Bits<16>) -> &mut Self {
        self.set_field(CNA_CVT_CON3_CVT_OFFSET2__MASK, unsafe {
            CNA_CVT_CON3_CVT_OFFSET2(cvt_offset2.val())
        })
    }

    /// Description: CVT multiplier operand for the 3rd channel.
    ///
    /// Bit width: 16
    /// Range of values: 0x0000-0xFFFF.
    /// Known limitations: None documented.
    /// Related registers: `cvt_offset2`; `cna_cvt_con0.cvt_type`.
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
    /// Description: CVT adder operand for the 4th channel.
    ///
    /// Bit width: 16
    /// Range of values: 0x0000-0xFFFF.
    /// Known limitations: None documented.
    /// Related registers: `cvt_scale3`; `cna_cvt_con0.cvt_truncate_3`.
    pub fn cvt_offset3(&mut self, cvt_offset3: Bits<16>) -> &mut Self {
        self.set_field(CNA_CVT_CON4_CVT_OFFSET3__MASK, unsafe {
            CNA_CVT_CON4_CVT_OFFSET3(cvt_offset3.val())
        })
    }

    /// Description: CVT multiplier operand for the 4th channel.
    ///
    /// Bit width: 16
    /// Range of values: 0x0000-0xFFFF.
    /// Known limitations: None documented.
    /// Related registers: `cvt_offset3`; `cna_cvt_con0.cvt_type`.
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
    /// Description: Enables FC (fully-connected) zero-skipping mode.
    ///
    /// Bit width: 1
    /// Range of values: 1'd0: Disable; 1'd1: Enable — skip some feature data value,
    /// normally skip zero. When one pixel's feature data equals `fc_skip_data`, the
    /// corresponding weight data is not fetched from system memory.
    /// Known limitations: Per TRM Fig 36-5, when zero-skipping is enabled DPU's
    /// `conv_mode` must be 3, BS_CORE must be bypassed, `alu_src=0`, `mul_src=1`,
    /// `alu_algo=3` — convolution accumulation moves to BN_CORE and any extra operators
    /// must run on EW_CORE fed via NRDMA (`ew_src=1`).
    /// Related registers: `fc_skip_data`, `cna_fc_con1.data_offset`,
    /// `cna_fc_con2.weight_offset`, `cna_cbuf_con0.fc_data_bank`.
    pub fn fc_skip_en(&mut self, fc_skip_en: Bits<1>) -> &mut Self {
        self.set_field(CNA_FC_CON0_FC_SKIP_EN__MASK, unsafe {
            CNA_FC_CON0_FC_SKIP_EN(fc_skip_en.val())
        })
    }

    /// Description: The feature data value treated as "skippable" in FC zero-skipping
    /// mode.
    ///
    /// Bit width: 16
    /// Range of values: 0x0000-0xFFFF; normally set to 0.
    /// Known limitations: Only meaningful when `fc_skip_en` is set.
    /// Related registers: `fc_skip_en`.
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
    /// Description: Feature data offset used in FC zero-skipping mode.
    ///
    /// Bit width: 17
    /// Range of values: 0x00000-0x1FFFF.
    /// Known limitations: Only meaningful when `cna_fc_con0.fc_skip_en` is set.
    /// Related registers: `cna_fc_con0.fc_skip_en`, `cna_fc_con2.weight_offset`.
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
    /// Description: Number of pad rows added to the top of the feature map.
    ///
    /// Bit width: 4
    /// Range of values: 0x0-0xF.
    /// Known limitations: None documented.
    /// Related registers: `pad_left`; `cna_pad_con1.pad_value` (the value used to fill
    /// padded cells).
    pub fn pad_top(&mut self, pad_top: Bits<4>) -> &mut Self {
        self.set_field(CNA_PAD_CON0_PAD_TOP__MASK, unsafe {
            CNA_PAD_CON0_PAD_TOP(pad_top.val())
        })
    }

    /// Description: Number of pad columns added to the left of the feature map.
    ///
    /// Bit width: 4
    /// Range of values: 0x0-0xF.
    /// Known limitations: None documented.
    /// Related registers: `pad_top`; `cna_pad_con1.pad_value`.
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
    /// Description: Base address of the input feature data in system memory.
    ///
    /// Bit width: 32
    /// Range of values: 0x00000000-0xFFFFFFFF.
    /// Known limitations: None documented.
    /// Related registers: `cna_dma_con1.line_stride`, `cna_dma_con2.surf_stride`,
    /// `cna_fc_con2.weight_offset` (the weight-address counterpart).
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
    /// Description: Weight data address used in FC zero-skipping mode.
    ///
    /// Bit width: 17
    /// Range of values: 0x00000-0x1FFFF.
    /// Known limitations: Only meaningful when `cna_fc_con0.fc_skip_en` is set.
    /// Related registers: `cna_fc_con0.fc_skip_en`, `cna_fc_con1.data_offset`.
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
    /// Description: AXI burst length for feature data DMA.
    ///
    /// Bit width: 4
    /// Range of values: 4'd3: Burst length is 4; 4'd7: Burst length is 8; 4'd15: Burst
    /// length is 16.
    /// Known limitations: None documented.
    /// Related registers: `weight_burst_len`, `ov4k_bypass`.
    pub fn data_burst_len(&mut self, data_burst_len: Bits<4>) -> &mut Self {
        self.set_field(CNA_DMA_CON0_DATA_BURST_LEN__MASK, unsafe {
            CNA_DMA_CON0_DATA_BURST_LEN(data_burst_len.val())
        })
    }

    /// Description: AXI burst length for weight data DMA.
    ///
    /// Bit width: 4
    /// Range of values: 4'd3: Burst length is 4; 4'd7: Burst length is 8; 4'd15: Burst
    /// length is 16.
    /// Known limitations: None documented.
    /// Related registers: `data_burst_len`, `ov4k_bypass`.
    pub fn weight_burst_len(&mut self, weight_burst_len: Bits<4>) -> &mut Self {
        self.set_field(CNA_DMA_CON0_WEIGHT_BURST_LEN__MASK, unsafe {
            CNA_DMA_CON0_WEIGHT_BURST_LEN(weight_burst_len.val())
        })
    }

    /// Description: Controls whether bursts over 4K are split into 2 independent burst
    /// commands.
    ///
    /// Bit width: 1
    /// Range of values: 1'd0: Enable this feature (split over-4K bursts); 1'd1: Bypass
    /// this feature.
    /// Known limitations: None documented.
    /// Related registers: `data_burst_len`, `weight_burst_len`.
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
    /// Description: Line stride — feature width including the virtual box (padding
    /// beyond the logical shape).
    ///
    /// Bit width: 28
    /// Range of values: 0x0000000-0xFFFFFFF.
    /// Known limitations: None documented.
    /// Related registers: `surf_stride` (in `cna_dma_con2`); `cna_data_size0.datain_width`;
    /// analogous to `ppu_rdma`'s `src_line_stride` which is documented the same way
    /// ("including Virtual box").
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
    /// Description: Surface stride — the feature map's actual surface area.
    ///
    /// Bit width: 28
    /// Range of values: 0x0000000-0xFFFFFFF.
    /// Known limitations: None documented.
    /// Related registers: `cna_dma_con1.line_stride`; analogous to `ppu_rdma`'s
    /// `src_surf_stride`.
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
    /// Description: Feature input height for the AXI DMA used by FC zero-skip fetch.
    ///
    /// Bit width: 11
    /// Range of values: 0x000-0x7FF.
    /// Known limitations: Applies to the DMA-side fetch path for FC zero-skipping; see
    /// `cna_fc_con0.fc_skip_en`.
    /// Related registers: `dma_width`, `cna_fc_data_size1.dma_channel`.
    pub fn dma_height(&mut self, dma_height: Bits<11>) -> &mut Self {
        self.set_field(CNA_FC_DATA_SIZE0_DMA_HEIGHT__MASK, unsafe {
            CNA_FC_DATA_SIZE0_DMA_HEIGHT(dma_height.val())
        })
    }

    /// Description: Feature input width for the AXI DMA used by FC zero-skip fetch.
    ///
    /// Bit width: 14
    /// Range of values: 0x0000-0x3FFF.
    /// Known limitations: Applies to the DMA-side fetch path for FC zero-skipping; see
    /// `cna_fc_con0.fc_skip_en`.
    /// Related registers: `dma_height`, `cna_fc_data_size1.dma_channel`.
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
    /// Description: Feature input channel count for the AXI DMA used by FC zero-skip
    /// fetch.
    ///
    /// Bit width: 16
    /// Range of values: 0x0000-0xFFFF.
    /// Known limitations: Applies to the DMA-side fetch path for FC zero-skipping; see
    /// `cna_fc_con0.fc_skip_en`.
    /// Related registers: `cna_fc_data_size0.dma_width`/`dma_height`.
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
    /// Description: Disables automatic clock gating for the feature-fetch sub-block.
    ///
    /// Bit width: 1
    /// Range of values: 1'd0: Auto clock gate is enabled; 1'd1: Disable feature block
    /// clock gate.
    /// Known limitations: Automatic localized clock gating normally applies near every
    /// flip-flop plus at the block level (TRM §5.2); this bit only disables the
    /// block-level gate for the feature-fetch sub-block.
    /// Related registers: `cna_weight_disable_clkgate`, `csc_disable_clkgate`,
    /// `cbuf_cs_disable_clkgate`.
    pub fn cna_feature_disable_clkgate(
        &mut self,
        cna_feature_disable_clkgate: Bits<1>,
    ) -> &mut Self {
        self.set_field(CNA_CLK_GATE_CNA_FEATURE_DISABLE_CLKGATE__MASK, unsafe {
            CNA_CLK_GATE_CNA_FEATURE_DISABLE_CLKGATE(cna_feature_disable_clkgate.val())
        })
    }

    /// Description: Disables automatic clock gating for the weight-fetch sub-block.
    ///
    /// Bit width: 1
    /// Range of values: 1'd0: Auto clock gate is enabled; 1'd1: Disable weight block
    /// clock gate.
    /// Known limitations: See TRM §5.2 on automatic clock gating.
    /// Related registers: `cna_feature_disable_clkgate`, `csc_disable_clkgate`,
    /// `cbuf_cs_disable_clkgate`.
    pub fn cna_weight_disable_clkgate(&mut self, cna_weight_disable_clkgate: Bits<1>) -> &mut Self {
        self.set_field(CNA_CLK_GATE_CNA_WEIGHT_DISABLE_CLKGATE__MASK, unsafe {
            CNA_CLK_GATE_CNA_WEIGHT_DISABLE_CLKGATE(cna_weight_disable_clkgate.val())
        })
    }

    /// Description: Disables automatic clock gating for the sequence-scan (CSC) block.
    ///
    /// Bit width: 1
    /// Range of values: 1'd0: Auto clock gate is enabled; 1'd1: Disable csc block clock
    /// gate.
    /// Known limitations: See TRM §5.2 on automatic clock gating.
    /// Related registers: `cna_conv_con2.csc_do_en`/`csc_wo_en`;
    /// `cna_feature_disable_clkgate`, `cna_weight_disable_clkgate`,
    /// `cbuf_cs_disable_clkgate`.
    pub fn csc_disable_clkgate(&mut self, csc_disable_clkgate: Bits<1>) -> &mut Self {
        self.set_field(CNA_CLK_GATE_CSC_DISABLE_CLKGATE__MASK, unsafe {
            CNA_CLK_GATE_CSC_DISABLE_CLKGATE(csc_disable_clkgate.val())
        })
    }

    /// Description: Disables automatic clock gating for the CBUF cache.
    ///
    /// Bit width: 1
    /// Range of values: 1'd0: Auto clock gate is enabled; 1'd1: Disable CBUF clock auto
    /// gate.
    /// Known limitations: See TRM §5.2 on automatic clock gating.
    /// Related registers: `cna_feature_disable_clkgate`, `cna_weight_disable_clkgate`,
    /// `csc_disable_clkgate`; `cna_cbuf_con0`/`cna_cbuf_con1` (the CBUF bank allocation
    /// this clock serves).
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
    /// Description: Control register for the weight decompress engine.
    ///
    /// Bit width: 3
    /// Range of values: 0x0-0x7 (TRM gives no further enum breakdown for this field).
    /// Known limitations: TRM does not document the individual meanings of the 3-bit
    /// value beyond "control register for weight decompress".
    /// Related registers: `wt_dec_bypass`, `cna_dcomp_regnum.dcomp_regnum`,
    /// `cna_dcomp_addr0.decompress_addr0`, the 16 `dcomp_amountN` registers.
    pub fn decomp_control(&mut self, decomp_control: Bits<3>) -> &mut Self {
        self.set_field(CNA_DCOMP_CTRL_DECOMP_CONTROL__MASK, unsafe {
            CNA_DCOMP_CTRL_DECOMP_CONTROL(decomp_control.val())
        })
    }

    /// Description: Bypasses the weight decompress function.
    ///
    /// Bit width: 1
    /// Range of values: Boolean bypass flag (TRM gives no explicit 0/1 enum text beyond
    /// "Bypass weight decompress").
    /// Known limitations: When bypassed, the `dcomp_regnum`/`dcomp_addr0`/`dcomp_amountN`
    /// registers presumably have no effect, though the TRM does not state this
    /// explicitly.
    /// Related registers: `decomp_control`, `cna_dcomp_regnum.dcomp_regnum`,
    /// `cna_dcomp_addr0.decompress_addr0`, the 16 `dcomp_amountN` registers.
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
    /// Description: Weight decompress register count.
    ///
    /// Bit width: 32
    /// Range of values: 0x00000000-0xFFFFFFFF.
    /// Known limitations: None documented.
    /// Related registers: `cna_dcomp_ctrl.decomp_control`/`wt_dec_bypass`,
    /// `cna_dcomp_addr0.decompress_addr0`, the 16 `dcomp_amountN` registers.
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
    /// Description: Base address of the weight for decompression.
    ///
    /// Bit width: 32
    /// Range of values: 0x00000000-0xFFFFFFFF (per code's `Bits<32>`; see Known
    /// limitations).
    /// Known limitations: TRM's bit table lists this field as bits 31:4 (28 bits) with
    /// bits 3:0 reserved/read-only (implying 16-byte address alignment, same convention
    /// as `PpuDstBaseAddr::dst_base_addr` and `pc_base_address.pc_src_address`). However,
    /// unlike those two registers, the generated macro for this field
    /// (`CNA_DCOMP_ADDR0_DECOMPRESS_ADDR0`, from `rkt_registers.h`) has shift=0 and
    /// mask=0xffffffff — i.e. Mesa's own reverse-engineered header does *not* encode the
    /// bits[31:4] convention here, unlike the PC/PPU registers where its shift=4 agrees
    /// with the TRM table. Since weight decompression is bypassed on every path this
    /// crate currently builds (`cna_dcomp_ctrl`/`cna_dcomp_regnum` always zeroed), this
    /// field's value is inert on every test that can currently be run, so the
    /// TRM-vs-Mesa disagreement is unconfirmed either way. Kept as `Bits<32>` (matching
    /// what the compiled macro actually does) rather than introducing an address shift
    /// that the macro wouldn't apply back — shifting here would silently drop the top 4
    /// address bits without the fix the PPU/PC precedent relies on.
    /// Related registers: `cna_dcomp_ctrl`, `cna_dcomp_regnum.dcomp_regnum`, the 16
    /// `dcomp_amountN` registers.
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
            /// Description: Amount (size) of the weight data to decompress for this
            /// decompress region.
            ///
            /// Bit width: 32
            /// Range of values: 0x00000000-0xFFFFFFFF.
            /// Known limitations: This same doc comment covers all 16 instantiations of
            /// this macro — `CnaDcompAmount0` through `CnaDcompAmount15`, backing
            /// registers DCOMP_AMOUNT0-15 (offsets 0x1140-0x117C) — each an independently
            /// sized decompress-region-size register with an identical bit layout.
            /// Related registers: `cna_dcomp_ctrl.decomp_control`/`wt_dec_bypass`,
            /// `cna_dcomp_regnum.dcomp_regnum`, `cna_dcomp_addr0.decompress_addr0`, and the
            /// other 15 `dcomp_amountN` registers.
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
    /// Description: Per-channel enable mask for the input convert (CVT) function.
    ///
    /// Bit width: 32
    /// Range of values: 0x00000000-0xFFFFFFFF; one bit per channel. Int4 has 32 channels
    /// total for 128 bits; int8 has 16 channels (per the TRM's partial note on channel
    /// packing for this field).
    /// Known limitations: TRM's description of the int8 case is truncated ("Int 8 16
    /// channel...") in the source table.
    /// Related registers: `cna_cvt_con0.cvt_bypass` and the per-channel
    /// `cvt_scaleN`/`cvt_offsetN`/`cvt_truncateN` fields this mask enables/disables.
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
    /// Description: The value used to fill padded feature-map cells.
    ///
    /// Bit width: 32
    /// Range of values: 0x00000000-0xFFFFFFFF.
    /// Known limitations: None documented.
    /// Related registers: `cna_pad_con0.pad_top`/`pad_left` (pad counts); analogous to
    /// `ppu_padding_value_1/2_cfg`, which splits an even wider pad value across two
    /// registers for PPU.
    pub fn pad_value(&mut self, pad_value: Bits<32>) -> &mut Self {
        self.set_field(CNA_PAD_CON1_PAD_VALUE__MASK, unsafe {
            CNA_PAD_CON1_PAD_VALUE(pad_value.val())
        })
    }
}
