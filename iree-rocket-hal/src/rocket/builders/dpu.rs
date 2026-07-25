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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU;
    const OFFSET: u32 = REG_DPU_S_STATUS;
}

impl Register<DpuSStatus> {
    /// Description: Executer 0 ping-pong status (idle / operating / operating cascaded).
    ///
    /// Bit width: 2
    /// Range of values: 0=idle, 1=operating, 2=operating with executer 1 waiting to operate, 3=reserved.
    /// Known limitations: TRM marks this field read-only (status output only) — writing it through this builder has no defined hardware effect.
    /// Related registers: status_1 (same register); s_pointer (executer/pointer selection).
    pub fn status_0(&mut self, status_0: Bits<2>) -> &mut Self {
        self.set_field(DPU_S_STATUS_STATUS_0__MASK, unsafe {
            DPU_S_STATUS_STATUS_0(status_0.val())
        })
    }

    /// Description: Executer 1 ping-pong status (idle / operating / operating cascaded).
    ///
    /// Bit width: 2
    /// Range of values: 0=idle, 1=operating, 2=operating, executer 1 waiting to operate, 3=reserved.
    /// Known limitations: TRM marks this field read-only (status output only) — writing it through this builder has no defined hardware effect.
    /// Related registers: status_0 (same register); s_pointer (executer/pointer selection).
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU;
    const OFFSET: u32 = REG_DPU_S_POINTER;
}

impl Register<DpuSPointer> {
    /// Description: Selects which of the two shadow register groups is ready to be set/used next.
    ///
    /// Bit width: 1
    /// Range of values: 0=register group 0, 1=register group 1.
    /// Known limitations: Only meaningful when ping-pong is in use; ignored in non-ping-pong single-buffer usage.
    /// Related registers: pointer_pp_en, pointer_pp_mode, executer.
    pub fn pointer(&mut self, pointer: Bits<1>) -> &mut Self {
        self.set_field(DPU_S_POINTER_POINTER__MASK, unsafe {
            DPU_S_POINTER_POINTER(pointer.val())
        })
    }

    /// Description: Enables ping-pong toggling of the register group pointer.
    ///
    /// Bit width: 1
    /// Range of values: 0=disable, 1=enable.
    /// Known limitations: Works together with pointer_pp_mode which decides the toggle rule.
    /// Related registers: pointer, pointer_pp_mode, pointer_pp_clear.
    pub fn pointer_pp_en(&mut self, pointer_pp_en: Bits<1>) -> &mut Self {
        self.set_field(DPU_S_POINTER_POINTER_PP_EN__MASK, unsafe {
            DPU_S_POINTER_POINTER_PP_EN(pointer_pp_en.val())
        })
    }

    /// Description: Enables ping-pong toggling of the executer group.
    ///
    /// Bit width: 1
    /// Range of values: 0=disable, 1=enable.
    /// Known limitations: None documented.
    /// Related registers: executer_pp_clear, pointer_pp_mode, executer.
    pub fn executer_pp_en(&mut self, executer_pp_en: Bits<1>) -> &mut Self {
        self.set_field(DPU_S_POINTER_EXECUTER_PP_EN__MASK, unsafe {
            DPU_S_POINTER_EXECUTER_PP_EN(executer_pp_en.val())
        })
    }

    /// Description: Selects the ping-pong toggle rule for the register group pointer.
    ///
    /// Bit width: 1
    /// Range of values: 0=toggle by executer (e.g. executer 0 active -> next pointer toggles to 1), 1=toggle by pointer (e.g. pointer 0 active -> next pointer toggles to 1).
    /// Known limitations: Only takes effect when pointer_pp_en is enabled.
    /// Related registers: pointer_pp_en, pointer.
    pub fn pointer_pp_mode(&mut self, pointer_pp_mode: Bits<1>) -> &mut Self {
        self.set_field(DPU_S_POINTER_POINTER_PP_MODE__MASK, unsafe {
            DPU_S_POINTER_POINTER_PP_MODE(pointer_pp_mode.val())
        })
    }

    /// Description: Clears the register group pointer back to 0.
    ///
    /// Bit width: 1
    /// Range of values: write 1 to clear (W1C); reads back 0.
    /// Known limitations: None documented.
    /// Related registers: pointer, pointer_pp_en.
    pub fn pointer_pp_clear(&mut self, pointer_pp_clear: Bits<1>) -> &mut Self {
        self.set_field(DPU_S_POINTER_POINTER_PP_CLEAR__MASK, unsafe {
            DPU_S_POINTER_POINTER_PP_CLEAR(pointer_pp_clear.val())
        })
    }

    /// Description: Clears the executer group pointer back to 0.
    ///
    /// Bit width: 1
    /// Range of values: write 1 to clear (W1C); reads back 0.
    /// Known limitations: None documented.
    /// Related registers: executer, executer_pp_en.
    pub fn executer_pp_clear(&mut self, executer_pp_clear: Bits<1>) -> &mut Self {
        self.set_field(DPU_S_POINTER_EXECUTER_PP_CLEAR__MASK, unsafe {
            DPU_S_POINTER_EXECUTER_PP_CLEAR(executer_pp_clear.val())
        })
    }

    /// Description: Reports/selects which of the two executer register groups is in use.
    ///
    /// Bit width: 1
    /// Range of values: 0=executer group 0, 1=executer group 1.
    /// Known limitations: TRM marks this field read-only (status); writing it through this builder has no defined hardware effect.
    /// Related registers: pointer, s_status.
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU;
    const OFFSET: u32 = REG_DPU_OPERATION_ENABLE;
}

impl Register<DpuOperationEnable> {
    /// Description: Triggers the DPU block to start operating on the currently configured register group.
    ///
    /// Bit width: 1
    /// Range of values: 0=disable, 1=enable/trigger.
    /// Known limitations: This register and every register after it in the block are shadowed for ping-pong operation (per TRM).
    /// Related registers: s_pointer (selects which shadow group op_en applies to).
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU;
    const OFFSET: u32 = REG_DPU_FEATURE_MODE_CFG;
}

impl Register<DpuFeatureModeCfg> {
    /// Description: Selects whether DPU's main input data comes from the convolution pipeline or from DPU_RDMA.
    ///
    /// Bit width: 1
    /// Range of values: 0=DPU core main data is from convolution output, 1=DPU core main data is from MRDMA (DPU running standalone).
    /// Known limitations: When enabled (Fig 36-7 "DPU flying mode"), DPU_RDMA's own feature_mode_cfg.flying_mode and MRDMA fetch config must also be set up; DPU then bypasses the convolution pipeline entirely.
    /// Related registers: DPU_RDMA dpu_rdma_feature_mode_cfg.flying_mode/mrdma_disable, output_mode.
    pub fn flying_mode(&mut self, flying_mode: Bits<1>) -> &mut Self {
        self.set_field(DPU_FEATURE_MODE_CFG_FLYING_MODE__MASK, unsafe {
            DPU_FEATURE_MODE_CFG_FLYING_MODE(flying_mode.val())
        })
    }

    /// Description: Routes the DPU core output to PPU and/or straight to external memory.
    ///
    /// Bit width: 2
    /// Range of values: bit0=output goes to PPU (active high), bit1=output goes to outside/external memory (active high); both bits may be set simultaneously.
    /// Known limitations: None documented.
    /// Related registers: dst_base_addr/dst_surf_stride (external-memory destination), PPU's flying_mode (consumes DPU's PPU-routed output).
    pub fn output_mode(&mut self, output_mode: Bits<2>) -> &mut Self {
        self.set_field(DPU_FEATURE_MODE_CFG_OUTPUT_MODE__MASK, unsafe {
            DPU_FEATURE_MODE_CFG_OUTPUT_MODE(output_mode.val())
        })
    }

    /// Description: Selects normal vs depthwise convolution mode for how DPU consumes CORE's output.
    ///
    /// Bit width: 2
    /// Range of values: 0=normal convolution mode, 1=reserved, 2=reserved, 3=depthwise convolution mode.
    /// Known limitations: Zero-skipping/fully-connected mode (Fig 36-5) requires conv_mode=3 together with BS_CORE bypassed, bs_alu_src=0, bs_mul_src=1, bs_alu_algo=3 — the convolution accumulation itself is then done by BN_CORE instead of BS_CORE.
    /// Related registers: bs_cfg.bs_bypass, bs_cfg.bs_alu_src, bs_mul_cfg.bs_mul_src, bs_cfg.bs_alu_algo.
    pub fn conv_mode(&mut self, conv_mode: Bits<2>) -> &mut Self {
        self.set_field(DPU_FEATURE_MODE_CFG_CONV_MODE__MASK, unsafe {
            DPU_FEATURE_MODE_CFG_CONV_MODE(conv_mode.val())
        })
    }

    /// Description: AXI burst length used by DPU's data fetch/write path.
    ///
    /// Bit width: 4
    /// Range of values: 3=Burst4, 7=Burst8, 15=Burst16 (other values not documented).
    /// Known limitations: None documented.
    /// Related registers: nonalign, surf_len.
    pub fn burst_len(&mut self, burst_len: Bits<4>) -> &mut Self {
        self.set_field(DPU_FEATURE_MODE_CFG_BURST_LEN__MASK, unsafe {
            DPU_FEATURE_MODE_CFG_BURST_LEN(burst_len.val())
        })
    }

    /// Description: In non-align output mode, how many 8-byte units to store.
    ///
    /// Bit width: 16
    /// Range of values: 0-65535.
    /// Known limitations: Only meaningful when nonalign=1.
    /// Related registers: nonalign, data_format.mc_surf_out.
    pub fn surf_len(&mut self, surf_len: Bits<16>) -> &mut Self {
        self.set_field(DPU_FEATURE_MODE_CFG_SURF_LEN__MASK, unsafe {
            DPU_FEATURE_MODE_CFG_SURF_LEN(surf_len.val())
        })
    }

    /// Description: Enables non-align output mode, used when the output data flow shape is the same as the input data flow shape.
    ///
    /// Bit width: 1
    /// Range of values: 0=disable, 1=enable.
    /// Known limitations: When enabled, surf_len must be configured to describe the 8-byte store count.
    /// Related registers: surf_len.
    pub fn nonalign(&mut self, nonalign: Bits<1>) -> &mut Self {
        self.set_field(DPU_FEATURE_MODE_CFG_NONALIGN__MASK, unsafe {
            DPU_FEATURE_MODE_CFG_NONALIGN(nonalign.val())
        })
    }

    /// Description: Selects the regroup cut width applied to input data.
    ///
    /// Bit width: 4
    /// Range of values: 0=cut all input (128bit), 1=cut 4bit, 2=cut 8bit, 3=cut 16bit, 4=cut 32bit, 5=cut 64bit.
    /// Known limitations: None documented.
    /// Related registers: tp_en, bs_ow_cfg (rgp_cnter, size_e_0/1/2).
    pub fn rgp_type(&mut self, rgp_type: Bits<4>) -> &mut Self {
        self.set_field(DPU_FEATURE_MODE_CFG_RGP_TYPE__MASK, unsafe {
            DPU_FEATURE_MODE_CFG_RGP_TYPE(rgp_type.val())
        })
    }

    /// Description: Enables transpose of the DPU output.
    ///
    /// Bit width: 1
    /// Range of values: 0=disable, 1=enable.
    /// Known limitations: None documented.
    /// Related registers: bs_ow_cfg.tp_org_en, wdma_size_0.tp_precision.
    pub fn tp_en(&mut self, tp_en: Bits<1>) -> &mut Self {
        self.set_field(DPU_FEATURE_MODE_CFG_TP_EN__MASK, unsafe {
            DPU_FEATURE_MODE_CFG_TP_EN(tp_en.val())
        })
    }

    /// Description: Combine-use flag mirroring DPU_RDMA's comb_use bit 0.
    ///
    /// Bit width: 1
    /// Range of values: 0/1 — see DPU_RDMA's dpu_rdma_feature_mode_cfg.comb_use[0].
    /// Known limitations: Must be kept consistent with DPU_RDMA's comb_use[0] setting per TRM note.
    /// Related registers: DPU_RDMA dpu_rdma_feature_mode_cfg.comb_use.
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU;
    const OFFSET: u32 = REG_DPU_DATA_FORMAT;
}

impl Register<DpuDataFormat> {
    /// Description: Sets the internal precision DPU computes in.
    ///
    /// Bit width: 3
    /// Range of values: 0=Integer 8bit, 1=Integer 16bit, 2=Float point 16bit, 3=Bfloat 16bit, 4=Integer 32bit, 5=Float point 32bit, 6=Integer 4bit.
    /// Known limitations: None documented.
    /// Related registers: in_precision, out_precision.
    pub fn proc_precision(&mut self, proc_precision: Bits<3>) -> &mut Self {
        self.set_field(DPU_DATA_FORMAT_PROC_PRECISION__MASK, unsafe {
            DPU_DATA_FORMAT_PROC_PRECISION(proc_precision.val())
        })
    }

    /// Description: Selects how many surfaces the DPU output is serialized across.
    ///
    /// Bit width: 1
    /// Range of values: 0=output feature obeys the rule of 16-byte align for one pixel, 1=output feature can output 2 surface serial or 4 surf serials.
    /// Known limitations: None documented.
    /// Related registers: feature_mode_cfg.surf_len, surface_add.
    pub fn mc_surf_out(&mut self, mc_surf_out: Bits<1>) -> &mut Self {
        self.set_field(DPU_DATA_FORMAT_MC_SURF_OUT__MASK, unsafe {
            DPU_DATA_FORMAT_MC_SURF_OUT(mc_surf_out.val())
        })
    }

    /// Description: Shift amount used by the BS core MUL stage when the data being shifted is negative.
    ///
    /// Bit width: 6
    /// Range of values: 0-63.
    /// Known limitations: Only applies to negative-valued data; the positive-data shift amount is configured separately in bs_mul_cfg.bs_mul_shift_value.
    /// Related registers: bs_mul_cfg.bs_mul_shift_value, bs_mul_cfg.bs_truncate_src.
    pub fn bs_mul_shift_value_neg(&mut self, bs_mul_shift_value_neg: Bits<6>) -> &mut Self {
        self.set_field(DPU_DATA_FORMAT_BS_MUL_SHIFT_VALUE_NEG__MASK, unsafe {
            DPU_DATA_FORMAT_BS_MUL_SHIFT_VALUE_NEG(bs_mul_shift_value_neg.val())
        })
    }

    /// Description: Shift amount used by the BN core MUL stage when the data being shifted is negative.
    ///
    /// Bit width: 6
    /// Range of values: 0-63.
    /// Known limitations: Only applies to negative-valued data; the positive-data shift amount is configured separately in bn_mul_cfg.bn_mul_shift_value.
    /// Related registers: bn_mul_cfg.bn_mul_shift_value, bn_mul_cfg.bn_truncate_src.
    pub fn bn_mul_shift_value_neg(&mut self, bn_mul_shift_value_neg: Bits<6>) -> &mut Self {
        self.set_field(DPU_DATA_FORMAT_BN_MUL_SHIFT_VALUE_NEG__MASK, unsafe {
            DPU_DATA_FORMAT_BN_MUL_SHIFT_VALUE_NEG(bn_mul_shift_value_neg.val())
        })
    }

    /// Description: Shift amount used by the EW core when the data being shifted is negative.
    ///
    /// Bit width: 10
    /// Range of values: 0-1023.
    /// Known limitations: Only applies to negative-valued data; the positive-data shift value is configured separately in ew_cvt_scale_value.ew_truncate.
    /// Related registers: ew_cvt_scale_value.ew_truncate.
    pub fn ew_truncate_neg(&mut self, ew_truncate_neg: Bits<10>) -> &mut Self {
        self.set_field(DPU_DATA_FORMAT_EW_TRUNCATE_NEG__MASK, unsafe {
            DPU_DATA_FORMAT_EW_TRUNCATE_NEG(ew_truncate_neg.val())
        })
    }

    /// Description: Sets DPU's input precision.
    ///
    /// Bit width: 3
    /// Range of values: 0=Integer 8bit, 1=Integer 16bit, 2=Float point 16bit, 3=Bfloat 16bit, 4=Integer 32bit, 5=Float point 32bit, 6=Integer 4bit.
    /// Known limitations: TRM states this must match DPU_RDMA's input precision ("same with DPU_RDMA").
    /// Related registers: DPU_RDMA dpu_rdma_feature_mode_cfg.in_precision, proc_precision, out_precision.
    pub fn in_precision(&mut self, in_precision: Bits<3>) -> &mut Self {
        self.set_field(DPU_DATA_FORMAT_IN_PRECISION__MASK, unsafe {
            DPU_DATA_FORMAT_IN_PRECISION(in_precision.val())
        })
    }

    /// Description: Sets DPU's output precision.
    ///
    /// Bit width: 3
    /// Range of values: 0=Integer 8bit, 1=Integer 16bit, 2=Float point 16bit, 3=Bfloat 16bit, 4=Integer 32bit, 5=Float point 32bit, 6=Integer 4bit.
    /// Known limitations: None documented.
    /// Related registers: out_cvt_scale.fp32tofp16_en, proc_precision.
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU;
    const OFFSET: u32 = REG_DPU_OFFSET_PEND;
}

impl Register<DpuOffsetPend> {
    /// Description: Value assigned to the extra (padding) channel.
    ///
    /// Bit width: 16
    /// Range of values: 0-65535.
    /// Known limitations: None documented.
    /// Related registers: data_cube_channel (channel/orig_channel).
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU;
    const OFFSET: u32 = REG_DPU_DST_BASE_ADDR;
}

impl Register<DpuDstBaseAddr> {
    /// Description: Base address DPU writes its output to in external memory.
    ///
    /// Bit width: 32
    /// Range of values: any address; TRM lists bits 31:4 as the writable field and bits 3:0 as hardware-reserved, i.e. the address is effectively 16-byte aligned.
    /// Known limitations: Only relevant when feature_mode_cfg.output_mode routes output to "outside".
    /// Related registers: dst_surf_stride, feature_mode_cfg.output_mode.
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU;
    const OFFSET: u32 = REG_DPU_DST_SURF_STRIDE;
}

impl Register<DpuDstSurfStride> {
    /// Description: Output shape surface stride in external memory.
    ///
    /// Bit width: 28
    /// Range of values: 0 to 2^28-1, in 16-byte units. Pass `stride_bytes / 16`;
    /// the builder shifts that logical field value into register bits 31:4.
    /// Known limitations: The byte stride must be 16-byte aligned. Passing the already
    /// encoded register word would shift it a second time.
    /// Related registers: dst_base_addr.
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU;
    const OFFSET: u32 = REG_DPU_DATA_CUBE_WIDTH;
}

impl Register<DpuDataCubeWidth> {
    /// Description: Width of the input cube processed by DPU.
    ///
    /// Bit width: 13
    /// Range of values: 0-8191.
    /// Known limitations: None documented.
    /// Related registers: data_cube_height.height, data_cube_channel.channel, wdma_size_1.width_wdma.
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU;
    const OFFSET: u32 = REG_DPU_DATA_CUBE_HEIGHT;
}

impl Register<DpuDataCubeHeight> {
    /// Description: Height of the input cube processed by DPU.
    ///
    /// Bit width: 13
    /// Range of values: 0-8191.
    /// Known limitations: None documented.
    /// Related registers: data_cube_width.width, data_cube_channel.channel, wdma_size_1.height_wdma.
    pub fn height(&mut self, height: Bits<13>) -> &mut Self {
        self.set_field(DPU_DATA_CUBE_HEIGHT_HEIGHT__MASK, unsafe {
            DPU_DATA_CUBE_HEIGHT_HEIGHT(height.val())
        })
    }

    /// Description: Configures the min/max reduction operator.
    ///
    /// Bit width: 3
    /// Range of values: bit0=enable minmax op, bit1=minmax type, bit2=probability only.
    /// Known limitations: None documented.
    /// Related registers: None.
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU;
    const OFFSET: u32 = REG_DPU_DATA_CUBE_NOTCH_ADDR;
}

impl Register<DpuDataCubeNotchAddr> {
    /// Description: Number of pixels from the end of width to the end of the shape line end (first tracked boundary).
    ///
    /// Bit width: 13
    /// Range of values: 0-8191.
    /// Known limitations: None documented.
    /// Related registers: notch_addr_1, data_cube_width.width.
    pub fn notch_addr_0(&mut self, notch_addr_0: Bits<13>) -> &mut Self {
        self.set_field(DPU_DATA_CUBE_NOTCH_ADDR_NOTCH_ADDR_0__MASK, unsafe {
            DPU_DATA_CUBE_NOTCH_ADDR_NOTCH_ADDR_0(notch_addr_0.val())
        })
    }

    /// Description: Number of pixels from the end of width to the end of the shape line end (second tracked boundary).
    ///
    /// Bit width: 13
    /// Range of values: 0-8191.
    /// Known limitations: None documented.
    /// Related registers: notch_addr_0, data_cube_width.width.
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU;
    const OFFSET: u32 = REG_DPU_DATA_CUBE_CHANNEL;
}

impl Register<DpuDataCubeChannel> {
    /// Description: Output cube channel count DPU processes.
    ///
    /// Bit width: 13
    /// Range of values: 0-8191.
    /// Known limitations: None documented.
    /// Related registers: orig_channel, data_cube_width.width, data_cube_height.height.
    pub fn channel(&mut self, channel: Bits<13>) -> &mut Self {
        self.set_field(DPU_DATA_CUBE_CHANNEL_CHANNEL__MASK, unsafe {
            DPU_DATA_CUBE_CHANNEL_CHANNEL(channel.val())
        })
    }

    /// Description: Original (pre-padding) output channel count.
    ///
    /// Bit width: 13
    /// Range of values: 0-8191.
    /// Known limitations: None documented.
    /// Related registers: channel.
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU;
    const OFFSET: u32 = REG_DPU_BS_CFG;
}

impl Register<DpuBsCfg> {
    /// Description: Bypasses the entire BS (first cascaded ALU) core.
    ///
    /// Bit width: 1
    /// Range of values: 0=do not bypass BS core, 1=bypass BS core.
    /// Known limitations: Zero-skipping/fully-connected mode (Fig 36-5) requires BS_CORE to be bypassed (this bit=1) so that the convolution accumulation is instead performed by BN_CORE.
    /// Related registers: bs_alu_bypass, bs_mul_bypass, bs_relu_bypass, feature_mode_cfg.conv_mode, bn_cfg (accumulation stage when BS is bypassed).
    pub fn bs_bypass(&mut self, bs_bypass: Bits<1>) -> &mut Self {
        self.set_field(DPU_BS_CFG_BS_BYPASS__MASK, unsafe {
            DPU_BS_CFG_BS_BYPASS(bs_bypass.val())
        })
    }

    /// Description: Bypasses the BS core's ALU (add/minus) stage.
    ///
    /// Bit width: 1
    /// Range of values: 0=do not bypass, 1=bypass.
    /// Known limitations: Has no additional effect once bs_bypass (whole-stage bypass) is set.
    /// Related registers: bs_bypass, bs_alu_algo, bs_alu_src, bs_alu_cfg.bs_alu_operand.
    pub fn bs_alu_bypass(&mut self, bs_alu_bypass: Bits<1>) -> &mut Self {
        self.set_field(DPU_BS_CFG_BS_ALU_BYPASS__MASK, unsafe {
            DPU_BS_CFG_BS_ALU_BYPASS(bs_alu_bypass.val())
        })
    }

    /// Description: Bypasses the BS core's MUL stage.
    ///
    /// Bit width: 1
    /// Range of values: 0=do not bypass, 1=bypass.
    /// Known limitations: Has no additional effect once bs_bypass is set.
    /// Related registers: bs_bypass, bs_mul_prelu, bs_mul_cfg.
    pub fn bs_mul_bypass(&mut self, bs_mul_bypass: Bits<1>) -> &mut Self {
        self.set_field(DPU_BS_CFG_BS_MUL_BYPASS__MASK, unsafe {
            DPU_BS_CFG_BS_MUL_BYPASS(bs_mul_bypass.val())
        })
    }

    /// Description: Enables PReLU-style signed-multiply mode in the BS core MUL stage.
    ///
    /// Bit width: 1
    /// Range of values: 0=disable, 1=enable.
    /// Known limitations: Only meaningful when bs_mul_bypass=0.
    /// Related registers: bs_mul_bypass, bs_mul_cfg.bs_mul_operand.
    pub fn bs_mul_prelu(&mut self, bs_mul_prelu: Bits<1>) -> &mut Self {
        self.set_field(DPU_BS_CFG_BS_MUL_PRELU__MASK, unsafe {
            DPU_BS_CFG_BS_MUL_PRELU(bs_mul_prelu.val())
        })
    }

    /// Description: Bypasses the BS core's RELU op.
    ///
    /// Bit width: 1
    /// Range of values: 0=do not bypass, 1=bypass.
    /// Known limitations: None documented.
    /// Related registers: bs_relux_en, bs_relux_cmp_value.bs_relux_cmp_dat.
    pub fn bs_relu_bypass(&mut self, bs_relu_bypass: Bits<1>) -> &mut Self {
        self.set_field(DPU_BS_CFG_BS_RELU_BYPASS__MASK, unsafe {
            DPU_BS_CFG_BS_RELU_BYPASS(bs_relu_bypass.val())
        })
    }

    /// Description: Enables RELUX (clamped ReLU) in the BS core.
    ///
    /// Bit width: 1
    /// Range of values: 0=disable, 1=enable.
    /// Known limitations: Only takes effect when bs_relu_bypass=0.
    /// Related registers: bs_relu_bypass, bs_relux_cmp_value.bs_relux_cmp_dat.
    pub fn bs_relux_en(&mut self, bs_relux_en: Bits<1>) -> &mut Self {
        self.set_field(DPU_BS_CFG_BS_RELUX_EN__MASK, unsafe {
            DPU_BS_CFG_BS_RELUX_EN(bs_relux_en.val())
        })
    }

    /// Description: Selects where the BS core ALU operand comes from.
    ///
    /// Bit width: 1
    /// Range of values: 0=from configuration register (bs_alu_cfg.bs_alu_operand), 1=from outside (DPU_RDMA's BRDMA feed).
    /// Known limitations: Zero-skipping mode (Fig 36-5) requires bs_alu_src=0.
    /// Related registers: bs_alu_cfg.bs_alu_operand, DPU_RDMA brdma_cfg.
    pub fn bs_alu_src(&mut self, bs_alu_src: Bits<1>) -> &mut Self {
        self.set_field(DPU_BS_CFG_BS_ALU_SRC__MASK, unsafe {
            DPU_BS_CFG_BS_ALU_SRC(bs_alu_src.val())
        })
    }

    /// Description: Selects the BS core ALU operation.
    ///
    /// Bit width: 4
    /// Range of values: 2=Add, 4=Minus; 0,1,3,5,6,7,8 reserved.
    /// Known limitations: Zero-skipping mode (Fig 36-5) requires bs_alu_algo=3, one of the TRM-listed "reserved" values for this path — its behavior there is documented only by that special-case flow, not by the field's own enum.
    /// Related registers: bs_alu_bypass, bs_alu_src.
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU;
    const OFFSET: u32 = REG_DPU_BS_ALU_CFG;
}

impl Register<DpuBsAluCfg> {
    /// Description: Operand value used by the BS core ALU stage.
    ///
    /// Bit width: 32
    /// Range of values: full 32-bit value (signed/unsigned interpretation depends on proc_precision).
    /// Known limitations: Ignored when bs_cfg.bs_alu_src=1 (operand instead comes from DPU_RDMA's BRDMA) or when bs_alu_bypass/bs_bypass is set.
    /// Related registers: bs_cfg.bs_alu_src, bs_cfg.bs_alu_bypass, bs_cfg.bs_alu_algo.
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU;
    const OFFSET: u32 = REG_DPU_BS_MUL_CFG;
}

impl Register<DpuBsMulCfg> {
    /// Description: Selects where the BS core MUL operand comes from.
    ///
    /// Bit width: 1
    /// Range of values: 0=from configuration register (bs_mul_operand), 1=from outside (DPU_RDMA's BRDMA feed).
    /// Known limitations: None documented.
    /// Related registers: bs_mul_operand, bs_cfg.bs_mul_bypass.
    pub fn bs_mul_src(&mut self, bs_mul_src: Bits<1>) -> &mut Self {
        self.set_field(DPU_BS_MUL_CFG_BS_MUL_SRC__MASK, unsafe {
            DPU_BS_MUL_CFG_BS_MUL_SRC(bs_mul_src.val())
        })
    }

    /// Description: Selects where the BS core MUL stage's shift value comes from.
    ///
    /// Bit width: 1
    /// Range of values: 0=from configuration register (bs_mul_shift_value), 1=from outside (DPU_RDMA's BRDMA feed).
    /// Known limitations: None documented.
    /// Related registers: bs_mul_shift_value, data_format.bs_mul_shift_value_neg.
    pub fn bs_truncate_src(&mut self, bs_truncate_src: Bits<1>) -> &mut Self {
        self.set_field(DPU_BS_MUL_CFG_BS_TRUNCATE_SRC__MASK, unsafe {
            DPU_BS_MUL_CFG_BS_TRUNCATE_SRC(bs_truncate_src.val())
        })
    }

    /// Description: Shift amount used by the BS core MUL stage for positive data.
    ///
    /// Bit width: 6
    /// Range of values: 0-63.
    /// Known limitations: The negative-data counterpart is data_format.bs_mul_shift_value_neg; only used when bs_truncate_src=0.
    /// Related registers: data_format.bs_mul_shift_value_neg, bs_truncate_src.
    pub fn bs_mul_shift_value(&mut self, bs_mul_shift_value: Bits<6>) -> &mut Self {
        self.set_field(DPU_BS_MUL_CFG_BS_MUL_SHIFT_VALUE__MASK, unsafe {
            DPU_BS_MUL_CFG_BS_MUL_SHIFT_VALUE(bs_mul_shift_value.val())
        })
    }

    /// Description: Operand value used by the BS core MUL stage.
    ///
    /// Bit width: 16
    /// Range of values: full 16-bit value.
    /// Known limitations: Ignored when bs_mul_src=1 (operand instead comes from BRDMA).
    /// Related registers: bs_mul_src, bs_cfg.bs_mul_bypass, bs_cfg.bs_mul_prelu.
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU;
    const OFFSET: u32 = REG_DPU_BS_RELUX_CMP_VALUE;
}

impl Register<DpuBsReluxCmpValue> {
    /// Description: Comparison value RELUX clamps against in the BS core.
    ///
    /// Bit width: 32
    /// Range of values: full 32-bit value.
    /// Known limitations: Only used when bs_cfg.bs_relux_en=1 and bs_cfg.bs_relu_bypass=0.
    /// Related registers: bs_cfg.bs_relux_en, bs_cfg.bs_relu_bypass.
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU;
    const OFFSET: u32 = REG_DPU_BS_OW_CFG;
}

impl Register<DpuBsOwCfg> {
    /// Description: Selects where the CPEND (output-width regroup) operand comes from.
    ///
    /// Bit width: 1
    /// Range of values: 0=from configuration register (bs_ow_op.ow_op), 1=from outside.
    /// Known limitations: None documented.
    /// Related registers: bs_ow_op.ow_op, od_bypass.
    pub fn ow_src(&mut self, ow_src: Bits<1>) -> &mut Self {
        self.set_field(DPU_BS_OW_CFG_OW_SRC__MASK, unsafe {
            DPU_BS_OW_CFG_OW_SRC(ow_src.val())
        })
    }

    /// Description: Bypasses the CPEND stage.
    ///
    /// Bit width: 1
    /// Range of values: 0=do not bypass, 1=bypass.
    /// Known limitations: None documented.
    /// Related registers: ow_src, bs_ow_op.ow_op.
    pub fn od_bypass(&mut self, od_bypass: Bits<1>) -> &mut Self {
        self.set_field(DPU_BS_OW_CFG_OD_BYPASS__MASK, unsafe {
            DPU_BS_OW_CFG_OD_BYPASS(od_bypass.val())
        })
    }

    /// Description: Number of 8-channel groups in a row for the first output line, minus 1.
    ///
    /// Bit width: 3
    /// Range of values: 0-7 (encodes 1-8 groups of 8 channels).
    /// Known limitations: None documented.
    /// Related registers: size_e_1, size_e_2, feature_mode_cfg.rgp_type, rgp_cnter.
    pub fn size_e_0(&mut self, size_e_0: Bits<3>) -> &mut Self {
        self.set_field(DPU_BS_OW_CFG_SIZE_E_0__MASK, unsafe {
            DPU_BS_OW_CFG_SIZE_E_0(size_e_0.val())
        })
    }

    /// Description: Number of 8-channel groups in a row for the middle output line, minus 1.
    ///
    /// Bit width: 3
    /// Range of values: 0-7 (encodes 1-8 groups of 8 channels).
    /// Known limitations: None documented.
    /// Related registers: size_e_0, size_e_2, rgp_cnter.
    pub fn size_e_1(&mut self, size_e_1: Bits<3>) -> &mut Self {
        self.set_field(DPU_BS_OW_CFG_SIZE_E_1__MASK, unsafe {
            DPU_BS_OW_CFG_SIZE_E_1(size_e_1.val())
        })
    }

    /// Description: Number of 8-channel groups in a row for the last output line, minus 1.
    ///
    /// Bit width: 3
    /// Range of values: 0-7 (encodes 1-8 groups of 8 channels).
    /// Known limitations: None documented.
    /// Related registers: size_e_0, size_e_1, rgp_cnter.
    pub fn size_e_2(&mut self, size_e_2: Bits<3>) -> &mut Self {
        self.set_field(DPU_BS_OW_CFG_SIZE_E_2__MASK, unsafe {
            DPU_BS_OW_CFG_SIZE_E_2(size_e_2.val())
        })
    }

    /// Description: Enables original (pre-regroup) transpose ordering.
    ///
    /// Bit width: 1
    /// Range of values: 0=disable, 1=enable.
    /// Known limitations: None documented.
    /// Related registers: feature_mode_cfg.tp_en.
    pub fn tp_org_en(&mut self, tp_org_en: Bits<1>) -> &mut Self {
        self.set_field(DPU_BS_OW_CFG_TP_ORG_EN__MASK, unsafe {
            DPU_BS_OW_CFG_TP_ORG_EN(tp_org_en.val())
        })
    }

    /// Description: Regroup decimation counter — selects 1-of-N data elements to keep.
    ///
    /// Bit width: 4
    /// Range of values: 0=select all data, 1=select 1 from every 2, 2=select 1 from every 4, 3=select 1 from every 8; other values reserved.
    /// Known limitations: None documented.
    /// Related registers: feature_mode_cfg.rgp_type.
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU;
    const OFFSET: u32 = REG_DPU_BS_OW_OP;
}

impl Register<DpuBsOwOp> {
    /// Description: CPEND operand value.
    ///
    /// Bit width: 16
    /// Range of values: full 16-bit value.
    /// Known limitations: Ignored when bs_ow_cfg.ow_src=1 (operand instead comes from outside).
    /// Related registers: bs_ow_cfg.ow_src, bs_ow_cfg.od_bypass.
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU;
    const OFFSET: u32 = REG_DPU_WDMA_SIZE_0;
}

impl Register<DpuWdmaSize0> {
    /// Description: Channel count for DPU's own write-DMA output.
    ///
    /// Bit width: 13
    /// Range of values: 0-8191.
    /// Known limitations: None documented.
    /// Related registers: size_c_wdma, wdma_size_1.width_wdma/height_wdma.
    pub fn channel_wdma(&mut self, channel_wdma: Bits<13>) -> &mut Self {
        self.set_field(DPU_WDMA_SIZE_0_CHANNEL_WDMA__MASK, unsafe {
            DPU_WDMA_SIZE_0_CHANNEL_WDMA(channel_wdma.val())
        })
    }

    /// Description: Size_c parameter for DPU's write-DMA (channel-dimension DMA sizing).
    ///
    /// Bit width: 11
    /// Range of values: 0-2047.
    /// Known limitations: None documented.
    /// Related registers: channel_wdma.
    pub fn size_c_wdma(&mut self, size_c_wdma: Bits<11>) -> &mut Self {
        self.set_field(DPU_WDMA_SIZE_0_SIZE_C_WDMA__MASK, unsafe {
            DPU_WDMA_SIZE_0_SIZE_C_WDMA(size_c_wdma.val())
        })
    }

    /// Description: Selects transpose precision (bit width) for DPU's write-DMA path.
    ///
    /// Bit width: 1
    /// Range of values: 0=8bit, 1=16bit.
    /// Known limitations: Only relevant when feature_mode_cfg.tp_en is enabled.
    /// Related registers: feature_mode_cfg.tp_en.
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU;
    const OFFSET: u32 = REG_DPU_WDMA_SIZE_1;
}

impl Register<DpuWdmaSize1> {
    /// Description: Width for DPU's write-DMA output.
    ///
    /// Bit width: 13
    /// Range of values: 0-8191.
    /// Known limitations: None documented.
    /// Related registers: height_wdma, wdma_size_0.channel_wdma.
    pub fn width_wdma(&mut self, width_wdma: Bits<13>) -> &mut Self {
        self.set_field(DPU_WDMA_SIZE_1_WIDTH_WDMA__MASK, unsafe {
            DPU_WDMA_SIZE_1_WIDTH_WDMA(width_wdma.val())
        })
    }

    /// Description: Height for DPU's write-DMA output.
    ///
    /// Bit width: 13
    /// Range of values: 0-8191.
    /// Known limitations: None documented.
    /// Related registers: width_wdma, wdma_size_0.channel_wdma.
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU;
    const OFFSET: u32 = REG_DPU_BN_CFG;
}

impl Register<DpuBnCfg> {
    /// Description: Bypasses the entire BN (second cascaded ALU) core.
    ///
    /// Bit width: 1
    /// Range of values: 0=do not bypass BN core, 1=bypass BN core.
    /// Known limitations: Zero-skipping/fully-connected mode (Fig 36-5) requires BN_CORE to perform the convolution accumulation itself (i.e. not bypassed) when BS_CORE is bypassed.
    /// Related registers: bs_cfg.bs_bypass, bn_alu_bypass, bn_mul_bypass, bn_relu_bypass.
    pub fn bn_bypass(&mut self, bn_bypass: Bits<1>) -> &mut Self {
        self.set_field(DPU_BN_CFG_BN_BYPASS__MASK, unsafe {
            DPU_BN_CFG_BN_BYPASS(bn_bypass.val())
        })
    }

    /// Description: Bypasses the BN core's ALU (add/minus) stage.
    ///
    /// Bit width: 1
    /// Range of values: 0=do not bypass, 1=bypass.
    /// Known limitations: Has no additional effect once bn_bypass is set.
    /// Related registers: bn_bypass, bn_alu_algo, bn_alu_src, bn_alu_cfg.bn_alu_operand.
    pub fn bn_alu_bypass(&mut self, bn_alu_bypass: Bits<1>) -> &mut Self {
        self.set_field(DPU_BN_CFG_BN_ALU_BYPASS__MASK, unsafe {
            DPU_BN_CFG_BN_ALU_BYPASS(bn_alu_bypass.val())
        })
    }

    /// Description: Bypasses the BN core's MUL stage.
    ///
    /// Bit width: 1
    /// Range of values: 0=do not bypass, 1=bypass.
    /// Known limitations: Has no additional effect once bn_bypass is set.
    /// Related registers: bn_bypass, bn_mul_prelu, bn_mul_cfg.
    pub fn bn_mul_bypass(&mut self, bn_mul_bypass: Bits<1>) -> &mut Self {
        self.set_field(DPU_BN_CFG_BN_MUL_BYPASS__MASK, unsafe {
            DPU_BN_CFG_BN_MUL_BYPASS(bn_mul_bypass.val())
        })
    }

    /// Description: Enables PReLU-style signed-multiply mode in the BN core MUL stage.
    ///
    /// Bit width: 1
    /// Range of values: 0=disable, 1=enable.
    /// Known limitations: Only meaningful when bn_mul_bypass=0.
    /// Related registers: bn_mul_bypass, bn_mul_cfg.bn_mul_operand.
    pub fn bn_mul_prelu(&mut self, bn_mul_prelu: Bits<1>) -> &mut Self {
        self.set_field(DPU_BN_CFG_BN_MUL_PRELU__MASK, unsafe {
            DPU_BN_CFG_BN_MUL_PRELU(bn_mul_prelu.val())
        })
    }

    /// Description: Bypasses the BN core's RELU op.
    ///
    /// Bit width: 1
    /// Range of values: 0=do not bypass, 1=bypass.
    /// Known limitations: None documented.
    /// Related registers: bn_relux_en, bn_relux_cmp_value.bn_relux_cmp_dat.
    pub fn bn_relu_bypass(&mut self, bn_relu_bypass: Bits<1>) -> &mut Self {
        self.set_field(DPU_BN_CFG_BN_RELU_BYPASS__MASK, unsafe {
            DPU_BN_CFG_BN_RELU_BYPASS(bn_relu_bypass.val())
        })
    }

    /// Description: Enables RELUX (clamped ReLU) in the BN core.
    ///
    /// Bit width: 1
    /// Range of values: 0=disable, 1=enable.
    /// Known limitations: Only takes effect when bn_relu_bypass=0.
    /// Related registers: bn_relu_bypass, bn_relux_cmp_value.bn_relux_cmp_dat.
    pub fn bn_relux_en(&mut self, bn_relux_en: Bits<1>) -> &mut Self {
        self.set_field(DPU_BN_CFG_BN_RELUX_EN__MASK, unsafe {
            DPU_BN_CFG_BN_RELUX_EN(bn_relux_en.val())
        })
    }

    /// Description: Selects where the BN core ALU operand comes from.
    ///
    /// Bit width: 1
    /// Range of values: 0=from configuration register (bn_alu_cfg.bn_alu_operand), 1=from outside (DPU_RDMA's NRDMA feed).
    /// Known limitations: None documented.
    /// Related registers: bn_alu_cfg.bn_alu_operand, DPU_RDMA nrdma_cfg.
    pub fn bn_alu_src(&mut self, bn_alu_src: Bits<1>) -> &mut Self {
        self.set_field(DPU_BN_CFG_BN_ALU_SRC__MASK, unsafe {
            DPU_BN_CFG_BN_ALU_SRC(bn_alu_src.val())
        })
    }

    /// Description: Selects the BN core ALU operation.
    ///
    /// Bit width: 4
    /// Range of values: 2=Add, 4=Minus; 0,1,3,5,6,7,8 reserved.
    /// Known limitations: In zero-skipping mode, BN_CORE performs the convolution accumulation itself instead of BS_CORE (Fig 36-5) — see bs_alu_algo's note on the analogous alu_algo=3 requirement for that flow.
    /// Related registers: bn_alu_bypass, bn_alu_src.
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU;
    const OFFSET: u32 = REG_DPU_BN_ALU_CFG;
}

impl Register<DpuBnAluCfg> {
    /// Description: Operand value used by the BN core ALU stage.
    ///
    /// Bit width: 32
    /// Range of values: full 32-bit value.
    /// Known limitations: Ignored when bn_cfg.bn_alu_src=1 (operand instead comes from DPU_RDMA's NRDMA).
    /// Related registers: bn_cfg.bn_alu_src, bn_cfg.bn_alu_bypass, bn_cfg.bn_alu_algo.
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU;
    const OFFSET: u32 = REG_DPU_BN_MUL_CFG;
}

impl Register<DpuBnMulCfg> {
    /// Description: Selects where the BN core MUL operand comes from.
    ///
    /// Bit width: 1
    /// Range of values: 0=from configuration register (bn_mul_operand), 1=from outside (NRDMA).
    /// Known limitations: None documented.
    /// Related registers: bn_mul_operand, bn_cfg.bn_mul_bypass.
    pub fn bn_mul_src(&mut self, bn_mul_src: Bits<1>) -> &mut Self {
        self.set_field(DPU_BN_MUL_CFG_BN_MUL_SRC__MASK, unsafe {
            DPU_BN_MUL_CFG_BN_MUL_SRC(bn_mul_src.val())
        })
    }

    /// Description: Selects where the BN core MUL stage's shift value comes from.
    ///
    /// Bit width: 1
    /// Range of values: 0=from configuration register (bn_mul_shift_value), 1=from outside (NRDMA).
    /// Known limitations: None documented.
    /// Related registers: bn_mul_shift_value, data_format.bn_mul_shift_value_neg.
    pub fn bn_truncate_src(&mut self, bn_truncate_src: Bits<1>) -> &mut Self {
        self.set_field(DPU_BN_MUL_CFG_BN_TRUNCATE_SRC__MASK, unsafe {
            DPU_BN_MUL_CFG_BN_TRUNCATE_SRC(bn_truncate_src.val())
        })
    }

    /// Description: Shift amount used by the BN core MUL stage for positive data.
    ///
    /// Bit width: 6
    /// Range of values: 0-63.
    /// Known limitations: The negative-data counterpart is data_format.bn_mul_shift_value_neg; only used when bn_truncate_src=0.
    /// Related registers: data_format.bn_mul_shift_value_neg, bn_truncate_src.
    pub fn bn_mul_shift_value(&mut self, bn_mul_shift_value: Bits<6>) -> &mut Self {
        self.set_field(DPU_BN_MUL_CFG_BN_MUL_SHIFT_VALUE__MASK, unsafe {
            DPU_BN_MUL_CFG_BN_MUL_SHIFT_VALUE(bn_mul_shift_value.val())
        })
    }

    /// Description: Operand value used by the BN core MUL stage.
    ///
    /// Bit width: 16
    /// Range of values: full 16-bit value.
    /// Known limitations: Ignored when bn_mul_src=1 (operand instead comes from NRDMA).
    /// Related registers: bn_mul_src, bn_cfg.bn_mul_bypass, bn_cfg.bn_mul_prelu.
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU;
    const OFFSET: u32 = REG_DPU_BN_RELUX_CMP_VALUE;
}

impl Register<DpuBnReluxCmpValue> {
    /// Description: Comparison value RELUX clamps against in the BN core.
    ///
    /// Bit width: 32
    /// Range of values: full 32-bit value.
    /// Known limitations: Only used when bn_cfg.bn_relux_en=1 and bn_cfg.bn_relu_bypass=0.
    /// Related registers: bn_cfg.bn_relux_en, bn_cfg.bn_relu_bypass.
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU;
    const OFFSET: u32 = REG_DPU_EW_CFG;
}

impl Register<DpuEwCfg> {
    /// Description: Bypasses the entire EW (third cascaded, element-wise) core.
    ///
    /// Bit width: 1
    /// Range of values: 0=do not bypass EW core, 1=bypass EW core.
    /// Known limitations: Zero-skipping mode (Fig 36-5) requires any extra per-element operators to be performed by EW_CORE (i.e. not bypassed), fed via NRDMA with ew_op_src=1.
    /// Related registers: ew_op_bypass, ew_relu_bypass, ew_lut_bypass, ew_op_src.
    pub fn ew_bypass(&mut self, ew_bypass: Bits<1>) -> &mut Self {
        self.set_field(DPU_EW_CFG_EW_BYPASS__MASK, unsafe {
            DPU_EW_CFG_EW_BYPASS(ew_bypass.val())
        })
    }

    /// Description: Bypasses the EW core's ALU and MUL op stage.
    ///
    /// Bit width: 1
    /// Range of values: 0=do not bypass, 1=bypass.
    /// Known limitations: Has no additional effect once ew_bypass (whole-stage bypass) is set.
    /// Related registers: ew_bypass, ew_op_type, ew_alu_algo.
    pub fn ew_op_bypass(&mut self, ew_op_bypass: Bits<1>) -> &mut Self {
        self.set_field(DPU_EW_CFG_EW_OP_BYPASS__MASK, unsafe {
            DPU_EW_CFG_EW_OP_BYPASS(ew_op_bypass.val())
        })
    }

    /// Description: Selects whether the EW core's operator stage performs an ALU op or a MUL op.
    ///
    /// Bit width: 1
    /// Range of values: 0=ALU, 1=MUL.
    /// Known limitations: None documented.
    /// Related registers: ew_alu_algo, ew_mul_prelu, ew_op_bypass.
    pub fn ew_op_type(&mut self, ew_op_type: Bits<1>) -> &mut Self {
        self.set_field(DPU_EW_CFG_EW_OP_TYPE__MASK, unsafe {
            DPU_EW_CFG_EW_OP_TYPE(ew_op_type.val())
        })
    }

    /// Description: Enables PReLU-style signed-multiply mode in the EW core MUL stage.
    ///
    /// Bit width: 1
    /// Range of values: 0=disable, 1=enable.
    /// Known limitations: Only meaningful when ew_op_type selects MUL.
    /// Related registers: ew_op_type, EW_OP_VALUE_0..7 (ew_operand).
    pub fn ew_mul_prelu(&mut self, ew_mul_prelu: Bits<1>) -> &mut Self {
        self.set_field(DPU_EW_CFG_EW_MUL_PRELU__MASK, unsafe {
            DPU_EW_CFG_EW_MUL_PRELU(ew_mul_prelu.val())
        })
    }

    /// Description: Selects where the EW core operator's operand comes from.
    ///
    /// Bit width: 1
    /// Range of values: 0=from configuration register (EW_OP_VALUE_0-7), 1=from outside (DPU_RDMA's ERDMA/NRDMA feed).
    /// Known limitations: Zero-skipping mode (Fig 36-5) requires this to select "from outside" so extra operators are fed via NRDMA.
    /// Related registers: EW_OP_VALUE_0..7, DPU_RDMA erdma_cfg/nrdma_cfg.
    pub fn ew_op_src(&mut self, ew_op_src: Bits<1>) -> &mut Self {
        self.set_field(DPU_EW_CFG_EW_OP_SRC__MASK, unsafe {
            DPU_EW_CFG_EW_OP_SRC(ew_op_src.val())
        })
    }

    /// Description: Bypasses the activation LUT engine for the EW core's data path.
    ///
    /// Bit width: 1
    /// Range of values: 0=do not bypass, 1=bypass.
    /// Known limitations: None documented.
    /// Related registers: lut_cfg, lut_le_start/end, lut_lo_start/end.
    pub fn ew_lut_bypass(&mut self, ew_lut_bypass: Bits<1>) -> &mut Self {
        self.set_field(DPU_EW_CFG_EW_LUT_BYPASS__MASK, unsafe {
            DPU_EW_CFG_EW_LUT_BYPASS(ew_lut_bypass.val())
        })
    }

    /// Description: Bypasses the EW core's input converter (scale/offset/shift).
    ///
    /// Bit width: 1
    /// Range of values: 0=do not bypass, 1=bypass.
    /// Known limitations: None documented.
    /// Related registers: ew_cvt_offset_value.ew_op_cvt_offset, ew_cvt_scale_value.
    pub fn ew_op_cvt_bypass(&mut self, ew_op_cvt_bypass: Bits<1>) -> &mut Self {
        self.set_field(DPU_EW_CFG_EW_OP_CVT_BYPASS__MASK, unsafe {
            DPU_EW_CFG_EW_OP_CVT_BYPASS(ew_op_cvt_bypass.val())
        })
    }

    /// Description: Bypasses the EW core's RELU op.
    ///
    /// Bit width: 1
    /// Range of values: 0=do not bypass, 1=bypass.
    /// Known limitations: None documented.
    /// Related registers: ew_relux_en, ew_relux_cmp_value.ew_relux_cmp_dat.
    pub fn ew_relu_bypass(&mut self, ew_relu_bypass: Bits<1>) -> &mut Self {
        self.set_field(DPU_EW_CFG_EW_RELU_BYPASS__MASK, unsafe {
            DPU_EW_CFG_EW_RELU_BYPASS(ew_relu_bypass.val())
        })
    }

    /// Description: Enables RELUX (clamped ReLU) in the EW core.
    ///
    /// Bit width: 1
    /// Range of values: 0=disable, 1=enable.
    /// Known limitations: Only takes effect when ew_relu_bypass=0.
    /// Related registers: ew_relu_bypass, ew_relux_cmp_value.ew_relux_cmp_dat.
    pub fn ew_relux_en(&mut self, ew_relux_en: Bits<1>) -> &mut Self {
        self.set_field(DPU_EW_CFG_EW_RELUX_EN__MASK, unsafe {
            DPU_EW_CFG_EW_RELUX_EN(ew_relux_en.val())
        })
    }

    /// Description: Selects the EW core ALU operation.
    ///
    /// Bit width: 4
    /// Range of values: 0=Max, 1=Min, 2=Add, 3=Div, 4=Minus, 5=Abs, 6=Neg, 7=Floor, 8=Ceil.
    /// Known limitations: None documented.
    /// Related registers: ew_op_type, ew_binary_en, ew_equal_en.
    pub fn ew_alu_algo(&mut self, ew_alu_algo: Bits<4>) -> &mut Self {
        self.set_field(DPU_EW_CFG_EW_ALU_ALGO__MASK, unsafe {
            DPU_EW_CFG_EW_ALU_ALGO(ew_alu_algo.val())
        })
    }

    /// Description: Enables the binary variant of the min/max operation.
    ///
    /// Bit width: 1
    /// Range of values: 0=disable, 1=enable.
    /// Known limitations: Only meaningful when ew_alu_algo selects Max or Min.
    /// Related registers: ew_alu_algo, ew_equal_en.
    pub fn ew_binary_en(&mut self, ew_binary_en: Bits<1>) -> &mut Self {
        self.set_field(DPU_EW_CFG_EW_BINARY_EN__MASK, unsafe {
            DPU_EW_CFG_EW_BINARY_EN(ew_binary_en.val())
        })
    }

    /// Description: Enables the equal variant of the min/max operation.
    ///
    /// Bit width: 1
    /// Range of values: 0=disable, 1=enable.
    /// Known limitations: Only meaningful when ew_alu_algo selects Max or Min.
    /// Related registers: ew_alu_algo, ew_binary_en.
    pub fn ew_equal_en(&mut self, ew_equal_en: Bits<1>) -> &mut Self {
        self.set_field(DPU_EW_CFG_EW_EQUAL_EN__MASK, unsafe {
            DPU_EW_CFG_EW_EQUAL_EN(ew_equal_en.val())
        })
    }

    /// Description: Data element size of the cube fetched from ERDMA.
    ///
    /// Bit width: 2
    /// Range of values: 0=4bit, 1=8bit, 2=16bit, 3=32bit.
    /// Known limitations: Must match ERDMA's actual fetch data size configuration.
    /// Related registers: DPU_RDMA erdma_cfg.erdma_data_size, ew_data_mode.
    pub fn edata_size(&mut self, edata_size: Bits<2>) -> &mut Self {
        self.set_field(DPU_EW_CFG_EDATA_SIZE__MASK, unsafe {
            DPU_EW_CFG_EDATA_SIZE(edata_size.val())
        })
    }

    /// Description: Data layout mode of the data fetched from ERDMA.
    ///
    /// Bit width: 2
    /// Range of values: TRM's bit table gives no enumerated values for this field beyond its bit position (29:28); cross-reference DPU_RDMA's erdma_data_mode (per-channel/per-pixel/per-channel-by-pixel) for the intended semantics.
    /// Known limitations: TRM's own bit table gives no enumerated description for this field.
    /// Related registers: DPU_RDMA erdma_cfg.erdma_data_mode, edata_size.
    pub fn ew_data_mode(&mut self, ew_data_mode: Bits<2>) -> &mut Self {
        self.set_field(DPU_EW_CFG_EW_DATA_MODE__MASK, unsafe {
            DPU_EW_CFG_EW_DATA_MODE(ew_data_mode.val())
        })
    }

    /// Description: Rounding rule used by the EW input converter when the fractional part is exactly 0.5.
    ///
    /// Bit width: 1
    /// Range of values: 0=if the integer is odd, carry 1; 1=carry 1 no matter what the integer is.
    /// Known limitations: None documented.
    /// Related registers: ew_cvt_type, ew_cvt_scale_value, ew_cvt_offset_value.
    pub fn ew_cvt_round(&mut self, ew_cvt_round: Bits<1>) -> &mut Self {
        self.set_field(DPU_EW_CFG_EW_CVT_ROUND__MASK, unsafe {
            DPU_EW_CFG_EW_CVT_ROUND(ew_cvt_round.val())
        })
    }

    /// Description: Order of operations for the EW input converter.
    ///
    /// Bit width: 1
    /// Range of values: 0=mul first, 1=add first.
    /// Known limitations: None documented.
    /// Related registers: ew_cvt_round, ew_cvt_offset_value, ew_cvt_scale_value.
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU;
    const OFFSET: u32 = REG_DPU_EW_CVT_OFFSET_VALUE;
}

impl Register<DpuEwCvtOffsetValue> {
    /// Description: Offset added by the EW core's input converter.
    ///
    /// Bit width: 32
    /// Range of values: full 32-bit value.
    /// Known limitations: Ignored when ew_cfg.ew_op_cvt_bypass=1.
    /// Related registers: ew_cfg.ew_op_cvt_bypass, ew_cvt_scale_value.
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU;
    const OFFSET: u32 = REG_DPU_EW_CVT_SCALE_VALUE;
}

impl Register<DpuEwCvtScaleValue> {
    /// Description: Scale factor applied by the EW core's input converter.
    ///
    /// Bit width: 16
    /// Range of values: full 16-bit value.
    /// Known limitations: Ignored when ew_cfg.ew_op_cvt_bypass=1.
    /// Related registers: ew_cfg.ew_op_cvt_bypass, ew_op_cvt_shift, ew_cvt_offset_value.
    pub fn ew_op_cvt_scale(&mut self, ew_op_cvt_scale: Bits<16>) -> &mut Self {
        self.set_field(DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE__MASK, unsafe {
            DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE(ew_op_cvt_scale.val())
        })
    }

    /// Description: Shift amount applied by the EW core's input converter.
    ///
    /// Bit width: 6
    /// Range of values: 0-63.
    /// Known limitations: Ignored when ew_cfg.ew_op_cvt_bypass=1.
    /// Related registers: ew_op_cvt_scale, ew_cfg.ew_op_cvt_bypass.
    pub fn ew_op_cvt_shift(&mut self, ew_op_cvt_shift: Bits<6>) -> &mut Self {
        self.set_field(DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SHIFT__MASK, unsafe {
            DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SHIFT(ew_op_cvt_shift.val())
        })
    }

    /// Description: Shift value applied by the EW core's output-side truncation.
    ///
    /// Bit width: 10
    /// Range of values: 0-1023.
    /// Known limitations: The negative-data counterpart is data_format.ew_truncate_neg.
    /// Related registers: data_format.ew_truncate_neg.
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU;
    const OFFSET: u32 = REG_DPU_EW_RELUX_CMP_VALUE;
}

impl Register<DpuEwReluxCmpValue> {
    /// Description: Comparison value RELUX clamps against in the EW core.
    ///
    /// Bit width: 32
    /// Range of values: full 32-bit value.
    /// Known limitations: Only used when ew_cfg.ew_relux_en=1 and ew_cfg.ew_relu_bypass=0.
    /// Related registers: ew_cfg.ew_relux_en, ew_cfg.ew_relu_bypass.
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU;
    const OFFSET: u32 = REG_DPU_OUT_CVT_OFFSET;
}

impl Register<DpuOutCvtOffset> {
    /// Description: Offset applied by DPU's final output converter.
    ///
    /// Bit width: 32
    /// Range of values: full 32-bit value.
    /// Known limitations: None documented.
    /// Related registers: out_cvt_scale, out_cvt_shift.
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU;
    const OFFSET: u32 = REG_DPU_OUT_CVT_SCALE;
}

impl Register<DpuOutCvtScale> {
    /// Description: Scale factor applied by DPU's final output converter.
    ///
    /// Bit width: 16
    /// Range of values: full 16-bit value.
    /// Known limitations: None documented.
    /// Related registers: out_cvt_offset, out_cvt_shift, fp32tofp16_en.
    pub fn out_cvt_scale(&mut self, out_cvt_scale: Bits<16>) -> &mut Self {
        self.set_field(DPU_OUT_CVT_SCALE_OUT_CVT_SCALE__MASK, unsafe {
            DPU_OUT_CVT_SCALE_OUT_CVT_SCALE(out_cvt_scale.val())
        })
    }

    /// Description: Enables conversion of the output from fp32 to fp16.
    ///
    /// Bit width: 1
    /// Range of values: 0=disable, 1=enable.
    /// Known limitations: Only meaningful when data_format.out_precision is configured for a floating-point output.
    /// Related registers: data_format.out_precision, out_cvt_scale.
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU;
    const OFFSET: u32 = REG_DPU_OUT_CVT_SHIFT;
}

impl Register<DpuOutCvtShift> {
    /// Description: Shift amount applied by DPU's final output converter.
    ///
    /// Bit width: 12
    /// Range of values: 0-4095.
    /// Known limitations: None documented.
    /// Related registers: minus_exp, out_cvt_scale, out_cvt_offset.
    pub fn out_cvt_shift(&mut self, out_cvt_shift: Bits<12>) -> &mut Self {
        self.set_field(DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT__MASK, unsafe {
            DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT(out_cvt_shift.val())
        })
    }

    /// Description: Exponent subtracted by the output converter (used for floating-point output scaling).
    ///
    /// Bit width: 8
    /// Range of values: 0-255.
    /// Known limitations: None documented.
    /// Related registers: out_cvt_shift, data_format.out_precision.
    pub fn minus_exp(&mut self, minus_exp: Bits<8>) -> &mut Self {
        self.set_field(DPU_OUT_CVT_SHIFT_MINUS_EXP__MASK, unsafe {
            DPU_OUT_CVT_SHIFT_MINUS_EXP(minus_exp.val())
        })
    }

    /// Description: Rounding rule used by the output converter when the fractional part is exactly 0.5.
    ///
    /// Bit width: 1
    /// Range of values: 0=if the integer is odd, carry 1; 1=carry 1 no matter what the integer is.
    /// Known limitations: None documented.
    /// Related registers: cvt_type.
    pub fn cvt_round(&mut self, cvt_round: Bits<1>) -> &mut Self {
        self.set_field(DPU_OUT_CVT_SHIFT_CVT_ROUND__MASK, unsafe {
            DPU_OUT_CVT_SHIFT_CVT_ROUND(cvt_round.val())
        })
    }

    /// Description: Order of operations for the output converter when the fractional part is exactly 0.5.
    ///
    /// Bit width: 1
    /// Range of values: 0=MUL first, 1=ALU first.
    /// Known limitations: None documented.
    /// Related registers: cvt_round, out_cvt_scale, out_cvt_offset.
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
            const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU;
            const OFFSET: u32 = $offset;
        }

        impl Register<$name> {
            /// Description: General-purpose operand register fed to the EW core's ALU/MUL stage. This
            /// macro is instantiated eight times — `DpuEwOpValue0` through `DpuEwOpValue7`, backing
            /// `EW_OP_VALUE_0` through `EW_OP_VALUE_7` (0x4090-0x40AC) — one independent operand slot
            /// per instance (the TRM calls these "the 1st"/"2nd"/.../"8th EW operand for EW core op").
            ///
            /// Bit width: 32
            /// Range of values: full 32-bit value.
            /// Known limitations: Ignored when ew_cfg.ew_op_src=1 (operand instead comes from DPU_RDMA's ERDMA/NRDMA feed).
            /// Related registers: ew_cfg.ew_op_src, ew_cfg.ew_op_type, ew_cfg.ew_mul_prelu.
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU;
    const OFFSET: u32 = REG_DPU_SURFACE_ADD;
}

impl Register<DpuSurfaceAdd> {
    /// Description: Number of surfaces in a row for DPU's output addressing.
    ///
    /// Bit width: 28
    /// Range of values: 0 to 2^28-1, in 16-byte units. Pass the logical field value;
    /// the builder shifts it into register bits 31:4.
    /// Known limitations: Passing an already encoded register word would shift it a
    /// second time.
    /// Related registers: dst_surf_stride, data_format.mc_surf_out.
    pub fn surf_add(&mut self, surf_add: Bits<28>) -> &mut Self {
        self.set_field(DPU_SURFACE_ADD_SURF_ADD__MASK, unsafe {
            DPU_SURFACE_ADD_SURF_ADD(surf_add.val())
        })
    }
}

// ========================================================================
// RESERVED_40C4 (0x40C4)
// ========================================================================

/// Mandatory zero write found in vendor and Mesa DPU programs.
///
/// Offset `0x40c4` is absent from `registers.xml` and `rkt_registers.h`, so
/// its hardware purpose and field layout remain unknown. Modeling the known
/// zero-valued write as a register type keeps call sites typed without
/// pretending that arbitrary values or fields are understood.
#[derive(Debug, Clone, Copy)]
pub struct DpuReserved40c4;

impl RegisterMeta for DpuReserved40c4 {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU;
    const OFFSET: u32 = 0x40c4;
}

// ========================================================================
// LUT_ACCESS_CFG (0x4100)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DpuLutAccessCfg;

impl RegisterMeta for DpuLutAccessCfg {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU;
    const OFFSET: u32 = REG_DPU_LUT_ACCESS_CFG;
}

impl Register<DpuLutAccessCfg> {
    /// Description: Table entry address for the LUT read/write access port.
    ///
    /// Bit width: 10
    /// Range of values: 0-1023.
    /// Known limitations: None documented.
    /// Related registers: lut_table_id, lut_access_type, lut_access_data.
    pub fn lut_addr(&mut self, lut_addr: Bits<10>) -> &mut Self {
        self.set_field(DPU_LUT_ACCESS_CFG_LUT_ADDR__MASK, unsafe {
            DPU_LUT_ACCESS_CFG_LUT_ADDR(lut_addr.val())
        })
    }

    /// Description: Selects which piecewise LUT (LE or LO) the access port targets.
    ///
    /// Bit width: 1
    /// Range of values: 0=LE LUT, 1=LO LUT.
    /// Known limitations: None documented.
    /// Related registers: lut_addr, lut_access_type, lut_le_start/end, lut_lo_start/end.
    pub fn lut_table_id(&mut self, lut_table_id: Bits<1>) -> &mut Self {
        self.set_field(DPU_LUT_ACCESS_CFG_LUT_TABLE_ID__MASK, unsafe {
            DPU_LUT_ACCESS_CFG_LUT_TABLE_ID(lut_table_id.val())
        })
    }

    /// Description: Selects read vs write for the LUT access port.
    ///
    /// Bit width: 1
    /// Range of values: 0=Read, 1=Write.
    /// Known limitations: None documented.
    /// Related registers: lut_addr, lut_table_id, lut_access_data.
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU;
    const OFFSET: u32 = REG_DPU_LUT_ACCESS_DATA;
}

impl Register<DpuLutAccessData> {
    /// Description: Data value read from or written to the LUT table entry selected by lut_access_cfg.
    ///
    /// Bit width: 16
    /// Range of values: full 16-bit value.
    /// Known limitations: None documented.
    /// Related registers: lut_access_cfg (lut_addr, lut_table_id, lut_access_type).
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU;
    const OFFSET: u32 = REG_DPU_LUT_CFG;
}

impl Register<DpuLutCfg> {
    /// Description: Selects between the LUT engine's first or second data road/pipeline.
    ///
    /// Bit width: 1
    /// Range of values: 0=1st, 1=2nd.
    /// Known limitations: None documented.
    /// Related registers: lut_lo_le_mux, lut_expand_en.
    pub fn lut_road_sel(&mut self, lut_road_sel: Bits<1>) -> &mut Self {
        self.set_field(DPU_LUT_CFG_LUT_ROAD_SEL__MASK, unsafe {
            DPU_LUT_CFG_LUT_ROAD_SEL(lut_road_sel.val())
        })
    }

    /// Description: Combines the LE and LO tables into one larger expanded table.
    ///
    /// Bit width: 1
    /// Range of values: 0=disable, 1=enable.
    /// Known limitations: lut_cal_sel is only meaningful when this bit is set.
    /// Related registers: lut_cal_sel, lut_lo_le_mux.
    pub fn lut_expand_en(&mut self, lut_expand_en: Bits<1>) -> &mut Self {
        self.set_field(DPU_LUT_CFG_LUT_EXPAND_EN__MASK, unsafe {
            DPU_LUT_CFG_LUT_EXPAND_EN(lut_expand_en.val())
        })
    }

    /// Description: Multiplexes between the LO LUT and LE LUT outputs.
    ///
    /// Bit width: 2
    /// Range of values: implementation-defined selection value; TRM's own bit table gives no enumerated meanings for this field beyond its name.
    /// Known limitations: TRM's own bit table gives no enumerated description for this field's values.
    /// Related registers: lut_hybrid_priority, lut_expand_en.
    pub fn lut_lo_le_mux(&mut self, lut_lo_le_mux: Bits<2>) -> &mut Self {
        self.set_field(DPU_LUT_CFG_LUT_LO_LE_MUX__MASK, unsafe {
            DPU_LUT_CFG_LUT_LO_LE_MUX(lut_lo_le_mux.val())
        })
    }

    /// Description: Selects which table (LE or LO) wins when both tables underflow simultaneously.
    ///
    /// Bit width: 1
    /// Range of values: 0=LE LUT, 1=LO LUT.
    /// Known limitations: None documented.
    /// Related registers: lut_oflow_priority, lut_hybrid_priority, lut_le_slope_scale/shift (uflow fields), lut_lo_slope_scale/shift (uflow fields).
    pub fn lut_uflow_priority(&mut self, lut_uflow_priority: Bits<1>) -> &mut Self {
        self.set_field(DPU_LUT_CFG_LUT_UFLOW_PRIORITY__MASK, unsafe {
            DPU_LUT_CFG_LUT_UFLOW_PRIORITY(lut_uflow_priority.val())
        })
    }

    /// Description: Selects which table (LE or LO) wins when both tables overflow simultaneously.
    ///
    /// Bit width: 1
    /// Range of values: 0=LE LUT, 1=LO LUT.
    /// Known limitations: None documented.
    /// Related registers: lut_uflow_priority, lut_hybrid_priority, lut_le_slope_scale/shift (oflow fields), lut_lo_slope_scale/shift (oflow fields).
    pub fn lut_oflow_priority(&mut self, lut_oflow_priority: Bits<1>) -> &mut Self {
        self.set_field(DPU_LUT_CFG_LUT_OFLOW_PRIORITY__MASK, unsafe {
            DPU_LUT_CFG_LUT_OFLOW_PRIORITY(lut_oflow_priority.val())
        })
    }

    /// Description: Selects which table (LE or LO) wins when both tables overlap in-range ("hybrid" flow).
    ///
    /// Bit width: 1
    /// Range of values: 0=LE LUT, 1=LO LUT.
    /// Known limitations: None documented.
    /// Related registers: lut_uflow_priority, lut_oflow_priority.
    pub fn lut_hybrid_priority(&mut self, lut_hybrid_priority: Bits<1>) -> &mut Self {
        self.set_field(DPU_LUT_CFG_LUT_HYBRID_PRIORITY__MASK, unsafe {
            DPU_LUT_CFG_LUT_HYBRID_PRIORITY(lut_hybrid_priority.val())
        })
    }

    /// Description: LUT calculate select, used only in expanded-table mode.
    ///
    /// Bit width: 1
    /// Range of values: implementation-defined selection value; TRM gives no enumerated meanings.
    /// Known limitations: Only useful when lut_expand_en=1 (per TRM).
    /// Related registers: lut_expand_en.
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU;
    const OFFSET: u32 = REG_DPU_LUT_INFO;
}

impl Register<DpuLutInfo> {
    /// Description: Selects which input bits feed the LE table's index generator.
    ///
    /// Bit width: 8
    /// Range of values: bit-select value used by the index generator to choose which input bits form the LE table index.
    /// Known limitations: None documented.
    /// Related registers: lut_lo_index_select, lut_le_start, lut_le_end.
    pub fn lut_le_index_select(&mut self, lut_le_index_select: Bits<8>) -> &mut Self {
        self.set_field(DPU_LUT_INFO_LUT_LE_INDEX_SELECT__MASK, unsafe {
            DPU_LUT_INFO_LUT_LE_INDEX_SELECT(lut_le_index_select.val())
        })
    }

    /// Description: Selects which input bits feed the LO table's index generator.
    ///
    /// Bit width: 8
    /// Range of values: bit-select value used by the index generator to choose which input bits form the LO table index.
    /// Known limitations: None documented.
    /// Related registers: lut_le_index_select, lut_lo_start, lut_lo_end.
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU;
    const OFFSET: u32 = REG_DPU_LUT_LE_START;
}

impl Register<DpuLutLeStart> {
    /// Description: Start point of the LE piecewise LUT's domain.
    ///
    /// Bit width: 32
    /// Range of values: full 32-bit value (interpreted per proc_precision).
    /// Known limitations: None documented.
    /// Related registers: lut_le_end, lut_le_slope_scale/shift (extrapolation beyond this domain).
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU;
    const OFFSET: u32 = REG_DPU_LUT_LE_END;
}

impl Register<DpuLutLeEnd> {
    /// Description: End point of the LE piecewise LUT's domain.
    ///
    /// Bit width: 32
    /// Range of values: full 32-bit value.
    /// Known limitations: None documented.
    /// Related registers: lut_le_start, lut_le_slope_scale/shift.
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU;
    const OFFSET: u32 = REG_DPU_LUT_LO_START;
}

impl Register<DpuLutLoStart> {
    /// Description: Start point of the LO piecewise LUT's domain.
    ///
    /// Bit width: 32
    /// Range of values: full 32-bit value.
    /// Known limitations: None documented.
    /// Related registers: lut_lo_end, lut_lo_slope_scale/shift.
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU;
    const OFFSET: u32 = REG_DPU_LUT_LO_END;
}

impl Register<DpuLutLoEnd> {
    /// Description: End point of the LO piecewise LUT's domain.
    ///
    /// Bit width: 32
    /// Range of values: full 32-bit value.
    /// Known limitations: None documented.
    /// Related registers: lut_lo_start, lut_lo_slope_scale/shift.
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU;
    const OFFSET: u32 = REG_DPU_LUT_LE_SLOPE_SCALE;
}

impl Register<DpuLutLeSlopeScale> {
    /// Description: Linear-extrapolation slope scale used by the LE table below its start point (underflow).
    ///
    /// Bit width: 16
    /// Range of values: full 16-bit value.
    /// Known limitations: Only applies when the LUT input falls below lut_le_start.
    /// Related registers: lut_le_start, lut_le_slope_uflow_shift, lut_uflow_priority.
    pub fn lut_le_slope_uflow_scale(&mut self, lut_le_slope_uflow_scale: Bits<16>) -> &mut Self {
        self.set_field(
            DPU_LUT_LE_SLOPE_SCALE_LUT_LE_SLOPE_UFLOW_SCALE__MASK,
            unsafe {
                DPU_LUT_LE_SLOPE_SCALE_LUT_LE_SLOPE_UFLOW_SCALE(lut_le_slope_uflow_scale.val())
            },
        )
    }

    /// Description: Linear-extrapolation slope scale used by the LE table beyond its end point (overflow).
    ///
    /// Bit width: 16
    /// Range of values: full 16-bit value.
    /// Known limitations: Only applies when the LUT input exceeds lut_le_end.
    /// Related registers: lut_le_end, lut_le_slope_oflow_shift, lut_oflow_priority.
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU;
    const OFFSET: u32 = REG_DPU_LUT_LE_SLOPE_SHIFT;
}

impl Register<DpuLutLeSlopeShift> {
    /// Description: Shift amount paired with lut_le_slope_uflow_scale for LE-table underflow extrapolation.
    ///
    /// Bit width: 5
    /// Range of values: 0-31.
    /// Known limitations: Only applies when the LUT input falls below lut_le_start.
    /// Related registers: lut_le_slope_uflow_scale, lut_le_start.
    pub fn lut_le_slope_uflow_shift(&mut self, lut_le_slope_uflow_shift: Bits<5>) -> &mut Self {
        self.set_field(
            DPU_LUT_LE_SLOPE_SHIFT_LUT_LE_SLOPE_UFLOW_SHIFT__MASK,
            unsafe {
                DPU_LUT_LE_SLOPE_SHIFT_LUT_LE_SLOPE_UFLOW_SHIFT(lut_le_slope_uflow_shift.val())
            },
        )
    }

    /// Description: Shift amount paired with lut_le_slope_oflow_scale for LE-table overflow extrapolation.
    ///
    /// Bit width: 5
    /// Range of values: 0-31.
    /// Known limitations: Only applies when the LUT input exceeds lut_le_end.
    /// Related registers: lut_le_slope_oflow_scale, lut_le_end.
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU;
    const OFFSET: u32 = REG_DPU_LUT_LO_SLOPE_SCALE;
}

impl Register<DpuLutLoSlopeScale> {
    /// Description: Linear-extrapolation slope scale used by the LO table below its start point (underflow).
    ///
    /// Bit width: 16
    /// Range of values: full 16-bit value.
    /// Known limitations: Only applies when the LUT input falls below lut_lo_start.
    /// Related registers: lut_lo_start, lut_lo_slope_uflow_shift, lut_uflow_priority.
    pub fn lut_lo_slope_uflow_scale(&mut self, lut_lo_slope_uflow_scale: Bits<16>) -> &mut Self {
        self.set_field(
            DPU_LUT_LO_SLOPE_SCALE_LUT_LO_SLOPE_UFLOW_SCALE__MASK,
            unsafe {
                DPU_LUT_LO_SLOPE_SCALE_LUT_LO_SLOPE_UFLOW_SCALE(lut_lo_slope_uflow_scale.val())
            },
        )
    }

    /// Description: Linear-extrapolation slope scale used by the LO table beyond its end point (overflow).
    ///
    /// Bit width: 16
    /// Range of values: full 16-bit value.
    /// Known limitations: Only applies when the LUT input exceeds lut_lo_end.
    /// Related registers: lut_lo_end, lut_lo_slope_oflow_shift, lut_oflow_priority.
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU;
    const OFFSET: u32 = REG_DPU_LUT_LO_SLOPE_SHIFT;
}

impl Register<DpuLutLoSlopeShift> {
    /// Description: Shift amount paired with lut_lo_slope_uflow_scale for LO-table underflow extrapolation.
    ///
    /// Bit width: 5
    /// Range of values: 0-31.
    /// Known limitations: Only applies when the LUT input falls below lut_lo_start.
    /// Related registers: lut_lo_slope_uflow_scale, lut_lo_start.
    pub fn lut_lo_slope_uflow_shift(&mut self, lut_lo_slope_uflow_shift: Bits<5>) -> &mut Self {
        self.set_field(
            DPU_LUT_LO_SLOPE_SHIFT_LUT_LO_SLOPE_UFLOW_SHIFT__MASK,
            unsafe {
                DPU_LUT_LO_SLOPE_SHIFT_LUT_LO_SLOPE_UFLOW_SHIFT(lut_lo_slope_uflow_shift.val())
            },
        )
    }

    /// Description: Shift amount paired with lut_lo_slope_oflow_scale for LO-table overflow extrapolation.
    ///
    /// Bit width: 5
    /// Range of values: 0-31.
    /// Known limitations: Only applies when the LUT input exceeds lut_lo_end.
    /// Related registers: lut_lo_slope_oflow_scale, lut_lo_end.
    pub fn lut_lo_slope_oflow_shift(&mut self, lut_lo_slope_oflow_shift: Bits<5>) -> &mut Self {
        self.set_field(
            DPU_LUT_LO_SLOPE_SHIFT_LUT_LO_SLOPE_OFLOW_SHIFT__MASK,
            unsafe {
                DPU_LUT_LO_SLOPE_SHIFT_LUT_LO_SLOPE_OFLOW_SHIFT(lut_lo_slope_oflow_shift.val())
            },
        )
    }
}
