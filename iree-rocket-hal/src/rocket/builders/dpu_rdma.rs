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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU_RDMA;
    const OFFSET: u32 = REG_DPU_RDMA_RDMA_S_STATUS;
}

impl Register<DpuRdmaSStatus> {
    /// Description: Executer 0's ping-pong operating state.
    ///
    /// Bit width: 2 (bits 1:0 in the TRM)
    /// Range of values: 2'd0 executer 0 idle; 2'd1 executer 0 operating; 2'd2 executer 0 operating and executer 1 waiting to operate; 2'd3 reserved.
    /// Known limitations: Read-only in hardware (RO); reported here as a setter only because this crate models every field uniformly as a builder method for constructing register images.
    /// Related registers: Companion to `status_1` (executer 1) in the same register; mirrors the generic ping-pong status pattern shared by CNA/CORE/DPU/PPU's own `S_STATUS` registers.
    pub fn status_0(&mut self, status_0: Bits<2>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_S_STATUS_STATUS_0__MASK, unsafe {
            DPU_RDMA_RDMA_S_STATUS_STATUS_0(status_0.val())
        })
    }

    /// Description: Executer 1's ping-pong operating state.
    ///
    /// Bit width: 2 (bits 17:16 in the TRM)
    /// Range of values: 2'd0 executer 1 idle; 2'd1 executer 1 operating; 2'd2 executer 1 operating, waiting to operate; 2'd3 reserved.
    /// Known limitations: Read-only in hardware (RO); reported here as a setter only because this crate models every field uniformly as a builder method for constructing register images.
    /// Related registers: Companion to `status_0` (executer 0) in the same register.
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU_RDMA;
    const OFFSET: u32 = REG_DPU_RDMA_RDMA_S_POINTER;
}

impl Register<DpuRdmaSPointer> {
    /// Description: Selects which of the two shadow register groups is ready to be applied.
    ///
    /// Bit width: 1 (bit 0)
    /// Range of values: 1'd0 register group 0; 1'd1 register group 1.
    /// Known limitations: Meaningful only when ping-pong is in use; ignored if `pointer_pp_en` toggling supersedes manual selection.
    /// Related registers: Works together with `pointer_pp_en`, `pointer_pp_mode`, `pointer_pp_clear`; same generic ping-pong pattern reused by every block's own `S_POINTER` register (CNA/CORE/DPU/PPU).
    pub fn pointer(&mut self, pointer: Bits<1>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_S_POINTER_POINTER__MASK, unsafe {
            DPU_RDMA_RDMA_S_POINTER_POINTER(pointer.val())
        })
    }

    /// Description: Enables ping-pong toggling of the register group.
    ///
    /// Bit width: 1 (bit 1)
    /// Range of values: 1'd0 disable; 1'd1 enable.
    /// Known limitations: None documented.
    /// Related registers: `pointer`, `pointer_pp_mode`, `pointer_pp_clear`.
    pub fn pointer_pp_en(&mut self, pointer_pp_en: Bits<1>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN__MASK, unsafe {
            DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN(pointer_pp_en.val())
        })
    }

    /// Description: Enables ping-pong toggling of the executer group.
    ///
    /// Bit width: 1 (bit 2)
    /// Range of values: 1'd0 disable; 1'd1 enable.
    /// Known limitations: None documented.
    /// Related registers: `executer`, `executer_pp_clear`.
    pub fn executer_pp_en(&mut self, executer_pp_en: Bits<1>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN__MASK, unsafe {
            DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN(executer_pp_en.val())
        })
    }

    /// Description: Selects the rule by which the register-group pointer toggles.
    ///
    /// Bit width: 1 (bit 3)
    /// Range of values: 1'd0 toggle by executer (e.g. current executer 0 -> next pointer toggles to 1); 1'd1 toggle by pointer (e.g. current pointer 0 -> next pointer toggles to 1).
    /// Known limitations: Only relevant when `pointer_pp_en` is enabled.
    /// Related registers: `pointer`, `pointer_pp_en`.
    pub fn pointer_pp_mode(&mut self, pointer_pp_mode: Bits<1>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE__MASK, unsafe {
            DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE(pointer_pp_mode.val())
        })
    }

    /// Description: Clears the register-group ping-pong pointer back to 0.
    ///
    /// Bit width: 1 (bit 4)
    /// Range of values: Write 1 to clear pointer to 0; self-clearing (W1C).
    /// Known limitations: None documented.
    /// Related registers: `pointer`, `pointer_pp_mode`.
    pub fn pointer_pp_clear(&mut self, pointer_pp_clear: Bits<1>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_S_POINTER_POINTER_PP_CLEAR__MASK, unsafe {
            DPU_RDMA_RDMA_S_POINTER_POINTER_PP_CLEAR(pointer_pp_clear.val())
        })
    }

    /// Description: Clears the executer-group ping-pong pointer back to 0.
    ///
    /// Bit width: 1 (bit 5)
    /// Range of values: Write 1 to clear pointer to 0; self-clearing (W1C).
    /// Known limitations: None documented.
    /// Related registers: `executer`, `executer_pp_en`.
    pub fn executer_pp_clear(&mut self, executer_pp_clear: Bits<1>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_CLEAR__MASK, unsafe {
            DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_CLEAR(executer_pp_clear.val())
        })
    }

    /// Description: Selects which executer register group is used.
    ///
    /// Bit width: 1 (bit 16)
    /// Range of values: 1'd0 executer group 0; 1'd1 executer group 1.
    /// Known limitations: TRM marks this field RO; exposed as a builder setter here only for uniformity with the rest of the crate's register-image construction.
    /// Related registers: `executer_pp_en`, `executer_pp_clear`.
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU_RDMA;
    const OFFSET: u32 = REG_DPU_RDMA_RDMA_OPERATION_ENABLE;
}

impl Register<DpuRdmaOperationEnable> {
    /// Description: Triggers the DPU_RDMA block to begin operating on the currently-configured register group.
    ///
    /// Bit width: 1 (bit 0)
    /// Range of values: 1'd0 disable; 1'd1 enable.
    /// Known limitations: This register and every register after it in the block are shadowed for ping-pong operation; per the TRM's regcmd op_en ordering rule (chapter 36 section 4.1), op_en entries should be written last in a batch, immediately after all other block registers are staged.
    /// Related registers: All other DPU_RDMA registers in this file (they are latched/shadowed relative to this bit); mirrors `global_operation_enable.dpu_rdma_op_en`.
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU_RDMA;
    const OFFSET: u32 = REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH;
}

impl Register<DpuRdmaDataCubeWidth> {
    /// Description: Input feature map width fed into DPU_RDMA.
    ///
    /// Bit width: 13 (bits 12:0)
    /// Range of values: 0 to 8191, N-1 encoded (actual width minus 1).
    /// Known limitations: None documented.
    /// Related registers: `height` (RDMA_DATA_CUBE_HEIGHT), `channel` (RDMA_DATA_CUBE_CHANNEL) describe the same input cube.
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU_RDMA;
    const OFFSET: u32 = REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT;
}

impl Register<DpuRdmaDataCubeHeight> {
    /// Description: Input feature map height fed into DPU_RDMA.
    ///
    /// Bit width: 13 (bits 12:0)
    /// Range of values: 0 to 8191, N-1 encoded (actual height minus 1).
    /// Known limitations: None documented.
    /// Related registers: `width` (RDMA_DATA_CUBE_WIDTH), `channel` (RDMA_DATA_CUBE_CHANNEL) describe the same input cube.
    pub fn height(&mut self, height: Bits<13>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT__MASK, unsafe {
            DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT(height.val())
        })
    }

    /// Description: Line notch address for the EW (element-wise) cube, used for end-of-line bookkeeping.
    ///
    /// Bit width: 13 (bits 28:16)
    /// Range of values: 0 to 8191 (pixel/line offset).
    /// Known limitations: Only meaningful when ERDMA is feeding EW operand data.
    /// Related registers: `erdma_data_mode`/`ew_surf_stride` (RDMA_ERDMA_CFG, RDMA_EW_SURF_STRIDE), `ew_surf_notch` (RDMA_EW_SURF_NOTCH).
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU_RDMA;
    const OFFSET: u32 = REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL;
}

impl Register<DpuRdmaDataCubeChannel> {
    /// Description: Input feature map channel count fed into DPU_RDMA.
    ///
    /// Bit width: 13 (bits 12:0)
    /// Range of values: 0 to 8191, N-1 encoded (actual channel count minus 1).
    /// Known limitations: None documented.
    /// Related registers: `width` (RDMA_DATA_CUBE_WIDTH), `height` (RDMA_DATA_CUBE_HEIGHT) describe the same input cube.
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU_RDMA;
    const OFFSET: u32 = REG_DPU_RDMA_RDMA_SRC_BASE_ADDR;
}

impl Register<DpuRdmaSrcBaseAddr> {
    /// Description: Base address of DPU's main input data when running in flying mode (i.e. MRDMA's source address).
    ///
    /// Bit width: 32 (bits 31:0)
    /// Range of values: Any full 32-bit system memory address.
    /// Known limitations: Only consumed when `flying_mode` is enabled in RDMA_FEATURE_MODE_CFG and `mrdma_disable` is clear; otherwise DPU's main data comes from the convolution pipeline instead.
    /// Related registers: `flying_mode`, `mrdma_disable`, `mrdma_fp16tofp32_en` (RDMA_FEATURE_MODE_CFG), `m_weight` (RDMA_WEIGHT, MRDMA's arbiter weight).
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU_RDMA;
    const OFFSET: u32 = REG_DPU_RDMA_RDMA_BRDMA_CFG;
}

impl Register<DpuRdmaBrdmaCfg> {
    /// Description: Selects which BS-stage operand types BRDMA fetches from memory.
    ///
    /// Bit width: 4 (bits 4:1)
    /// Range of values: Bitmask: bit[0] read ALU operand; bit[1] read CPEND operand; bit[2] read MUL operand; bit[3] read TRT operand. Set the corresponding bit to 1 to enable.
    /// Known limitations: Feeds DPU's BS core only; has no effect unless BS's `*_src` fields are configured to pull from BRDMA rather than a static config register.
    /// Related registers: `bs_base_addr` (RDMA_BS_BASE_ADDR, BRDMA's fetch address); DPU's `bs_alu_src`/`bs_mul_src`/`ow_src` (dpu_bs_cfg family, DPU block).
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU_RDMA;
    const OFFSET: u32 = REG_DPU_RDMA_RDMA_BS_BASE_ADDR;
}

impl Register<DpuRdmaBsBaseAddr> {
    /// Description: Base address BRDMA reads from for BS ALU, BS CPEND, and BS MUL operands.
    ///
    /// Bit width: 32 (bits 31:0)
    /// Range of values: Any full 32-bit system memory address.
    /// Known limitations: Only used for the operand types enabled in `brdma_data_use`.
    /// Related registers: `brdma_data_use` (RDMA_BRDMA_CFG), DPU's BS core registers (`dpu_bs_cfg`, `bs_alu_cfg`, `bs_mul_cfg`).
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU_RDMA;
    const OFFSET: u32 = REG_DPU_RDMA_RDMA_NRDMA_CFG;
}

impl Register<DpuRdmaNrdmaCfg> {
    /// Description: Selects which BN-stage operand types NRDMA fetches from memory.
    ///
    /// Bit width: 4 (bits 4:1)
    /// Range of values: Bitmask: bit[0] read ALU operand; bit[1] read CPEND operand (tied to 0 — BN has no CPEND stage); bit[2] read MUL operand; bit[3] read TRT operand. Set the corresponding bit to 1 to enable.
    /// Known limitations: Bit[1] (CPEND) is tied to 0 in hardware since BN has no OW/CPEND stage, unlike BS. Feeds DPU's BN core only; has no effect unless BN's `*_src` fields select NRDMA. Also used as the EW feed path when zero-skipping FC mode routes extra operators through EW via NRDMA (`ew_src=1`, per TRM Fig 36-5).
    /// Related registers: `bn_base_addr` (RDMA_BN_BASE_ADDR, NRDMA's fetch address); DPU's `bn_alu_src`/`bn_mul_src` (dpu_bn_cfg family, DPU block).
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU_RDMA;
    const OFFSET: u32 = REG_DPU_RDMA_RDMA_BN_BASE_ADDR;
}

impl Register<DpuRdmaBnBaseAddr> {
    /// Description: Base address NRDMA reads from for BN ALU and BN MUL operands.
    ///
    /// Bit width: 32 (bits 31:0)
    /// Range of values: Any full 32-bit system memory address.
    /// Known limitations: Only used for the operand types enabled in `nrdma_data_use`.
    /// Related registers: `nrdma_data_use` (RDMA_NRDMA_CFG), DPU's BN core registers (`dpu_bn_cfg`, `bn_alu_cfg`, `bn_mul_cfg`).
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU_RDMA;
    const OFFSET: u32 = REG_DPU_RDMA_RDMA_ERDMA_CFG;
}

impl Register<DpuRdmaErdmaCfg> {
    /// Description: Disables the ERDMA sub-DMA entirely.
    ///
    /// Bit width: 1 (bit 0)
    /// Range of values: 1'd0 do not disable ERDMA; 1'd1 disable ERDMA.
    /// Known limitations: When disabled, EW cannot receive an external operand feed via ERDMA regardless of `erdma_data_mode`/`erdma_data_size` settings.
    /// Related registers: `ew_base_addr` (RDMA_EW_BASE_ADDR), `comb_use` (RDMA_FEATURE_MODE_CFG) which can also route MRDMA's read to ERDMA.
    pub fn erdma_disable(&mut self, erdma_disable: Bits<1>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DISABLE__MASK, unsafe {
            DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DISABLE(erdma_disable.val())
        })
    }

    /// Description: Controls whether bursts larger than 4K are split into two independent burst commands.
    ///
    /// Bit width: 1 (bit 1)
    /// Range of values: 1'd0 enable this feature (split >4K bursts); 1'd1 bypass this feature.
    /// Known limitations: None documented.
    /// Related registers: `burst_len` (RDMA_FEATURE_MODE_CFG); analogous to CNA's own `ov4k_bypass` (`cna_dma_con0`).
    pub fn ov4k_bypass(&mut self, ov4k_bypass: Bits<1>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_ERDMA_CFG_OV4K_BYPASS__MASK, unsafe {
            DPU_RDMA_RDMA_ERDMA_CFG_OV4K_BYPASS(ov4k_bypass.val())
        })
    }

    /// Description: Precision of the cube data that ERDMA reads.
    ///
    /// Bit width: 2 (bits 3:2)
    /// Range of values: 2'd0 4-bit; 2'd1 8-bit; 2'd2 16-bit; 2'd3 32-bit.
    /// Known limitations: None documented.
    /// Related registers: `in_precision`/`proc_precision` (RDMA_FEATURE_MODE_CFG) describe the main-path precision separately from ERDMA's own.
    pub fn erdma_data_size(&mut self, erdma_data_size: Bits<2>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE__MASK, unsafe {
            DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE(erdma_data_size.val())
        })
    }

    /// Description: Enables non-align read mode for the EW cube.
    ///
    /// Bit width: 1 (bit 28)
    /// Range of values: 1'd0 do not use non-align mode; 1'd1 use non-align mode.
    /// Known limitations: None documented.
    /// Related registers: `erdma_surf_mode`, `ew_surf_stride` (RDMA_EW_SURF_STRIDE).
    pub fn erdma_nonalign(&mut self, erdma_nonalign: Bits<1>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_NONALIGN__MASK, unsafe {
            DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_NONALIGN(erdma_nonalign.val())
        })
    }

    /// Description: Surface mode of the EW cube.
    ///
    /// Bit width: 1 (bit 29)
    /// Range of values: 1'd0 1 surface series; 1'd1 2 surface series.
    /// Known limitations: None documented.
    /// Related registers: `ew_surf_stride` (RDMA_EW_SURF_STRIDE), `ew_surf_notch` (RDMA_EW_SURF_NOTCH).
    pub fn erdma_surf_mode(&mut self, erdma_surf_mode: Bits<1>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_SURF_MODE__MASK, unsafe {
            DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_SURF_MODE(erdma_surf_mode.val())
        })
    }

    /// Description: Whether the data ERDMA reads is organized per channel, per pixel, or per channel-by-pixel.
    ///
    /// Bit width: 2 (bits 31:30)
    /// Range of values: 2'd0 per channel; 2'd1 per pixel; 2'd2 per channel by pixel; 2'd3 reserved.
    /// Known limitations: When set to per-channel, `ew_surf_stride` must be set to 1.
    /// Related registers: `ew_surf_stride` (RDMA_EW_SURF_STRIDE).
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU_RDMA;
    const OFFSET: u32 = REG_DPU_RDMA_RDMA_EW_BASE_ADDR;
}

impl Register<DpuRdmaEwBaseAddr> {
    /// Description: Base address ERDMA reads from for the EW operand.
    ///
    /// Bit width: 32 (bits 31:0)
    /// Range of values: Any full 32-bit system memory address.
    /// Known limitations: Only used when ERDMA is enabled (`erdma_disable` clear).
    /// Related registers: `erdma_disable`/`erdma_data_mode`/`erdma_data_size` (RDMA_ERDMA_CFG), `ew_surf_stride` (RDMA_EW_SURF_STRIDE), DPU's EW core (`dpu_ew_cfg`, `ew_alu_cfg`).
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU_RDMA;
    const OFFSET: u32 = REG_DPU_RDMA_RDMA_EW_SURF_STRIDE;
}

impl Register<DpuRdmaEwSurfStride> {
    /// Description: Surface stride of the element-wise (EW) feature map read by ERDMA.
    ///
    /// Bit width: 28 (bits 31:4)
    /// Range of values: 0 to 2^28-1, in 16-byte units. Pass `stride_bytes / 16`;
    /// the builder shifts that logical field value into register bits 31:4. Must be set
    /// to 1 if `erdma_data_mode` is per-channel.
    /// Known limitations: The byte stride must be 16-byte aligned. Passing the already
    /// encoded register word would shift it a second time.
    /// Related registers: `erdma_data_mode`/`erdma_surf_mode` (RDMA_ERDMA_CFG), `ew_surf_notch` (RDMA_EW_SURF_NOTCH).
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU_RDMA;
    const OFFSET: u32 = REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG;
}

impl Register<DpuRdmaFeatureModeCfg> {
    /// Description: Enables flying mode, where DPU's main input data comes from DPU_RDMA (MRDMA) instead of the convolution pipeline.
    ///
    /// Bit width: 1 (bit 0)
    /// Range of values: 1'd0 DPU core main data is from convolution output; 1'd1 DPU core main data is from MRDMA.
    /// Known limitations: This is the defining bit of "flying mode" (TRM Fig 36-7): when enabled, DPU runs standalone against arbitrary memory via `src_base_addr`, bypassing CNA/CORE entirely; `mrdma_disable` must be clear for this to take effect.
    /// Related registers: `src_base_addr` (RDMA_SRC_BASE_ADDR), `mrdma_disable`, `mrdma_fp16tofp32_en`; mirrors DPU's own `dpu_feature_mode_cfg.flying_mode`.
    pub fn flying_mode(&mut self, flying_mode: Bits<1>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE__MASK, unsafe {
            DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE(flying_mode.val())
        })
    }

    /// Description: Selects the convolution mode the fed data corresponds to.
    ///
    /// Bit width: 2 (bits 2:1)
    /// Range of values: 2'd0 Dc; 2'd1 reserved; 2'd2 reserved; 2'd3 depthwise.
    /// Known limitations: Only the Dc (0) and depthwise (3) encodings are defined; 1 and 2 are reserved.
    /// Related registers: Mirrors CNA's `cna_conv_con1.conv_mode` and DPU's `dpu_feature_mode_cfg.conv_mode`.
    pub fn conv_mode(&mut self, conv_mode: Bits<2>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_FEATURE_MODE_CFG_CONV_MODE__MASK, unsafe {
            DPU_RDMA_RDMA_FEATURE_MODE_CFG_CONV_MODE(conv_mode.val())
        })
    }

    /// Description: Enables conversion of DPU's input data from fp16 to fp32.
    ///
    /// Bit width: 1 (bit 3)
    /// Range of values: 1'd0 disable; 1'd1 enable.
    /// Known limitations: Only meaningful when MRDMA is supplying DPU's main input (flying mode) and the source data is fp16.
    /// Related registers: `flying_mode`, `mrdma_disable`, `in_precision`/`proc_precision`.
    pub fn mrdma_fp16tofp32_en(&mut self, mrdma_fp16tofp32_en: Bits<1>) -> &mut Self {
        self.set_field(
            DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN__MASK,
            unsafe {
                DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN(mrdma_fp16tofp32_en.val())
            },
        )
    }

    /// Description: Disables the MRDMA sub-DMA entirely.
    ///
    /// Bit width: 1 (bit 4)
    /// Range of values: 1'd0 do not disable MRDMA; 1'd1 disable MRDMA.
    /// Known limitations: If disabled while `flying_mode` is set, DPU has no main data source; typically only disabled when DPU_RDMA is used purely to feed BS/BN/EW operands (BRDMA/NRDMA/ERDMA) without a flying-mode main input.
    /// Related registers: `flying_mode`, `src_base_addr` (RDMA_SRC_BASE_ADDR), `m_weight` (RDMA_WEIGHT).
    pub fn mrdma_disable(&mut self, mrdma_disable: Bits<1>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_DISABLE__MASK, unsafe {
            DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_DISABLE(mrdma_disable.val())
        })
    }

    /// Description: Process precision used for the fed data.
    ///
    /// Bit width: 3 (bits 7:5)
    /// Range of values: 3'd0 int8; 3'd1 int16; 3'd2 fp16; 3'd3 bf16; 3'd4 int32; 3'd5 fp32; 3'd6 int4.
    /// Known limitations: None documented.
    /// Related registers: `in_precision` (same register, describes input rather than process precision); mirrors CNA/CORE/DPU's own `proc_precision` fields.
    pub fn proc_precision(&mut self, proc_precision: Bits<3>) -> &mut Self {
        self.set_field(
            DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION__MASK,
            unsafe { DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION(proc_precision.val()) },
        )
    }

    /// Description: Controls whether MRDMA and ERDMA share a single read and which of them the fetched data is routed to.
    ///
    /// Bit width: 3 (bits 10:8)
    /// Range of values: Bitmask: bit[0] enable MRDMA and ERDMA to read the same data; bit[1] route the read data to MRDMA; bit[2] route the read data to ERDMA.
    /// Known limitations: Only meaningful when both MRDMA and ERDMA are in use simultaneously (i.e. neither `mrdma_disable` nor `erdma_disable` is set).
    /// Related registers: `mrdma_disable` (this register), `erdma_disable` (RDMA_ERDMA_CFG); mirrors DPU's own `dpu_feature_mode_cfg.comb_use[0]`.
    pub fn comb_use(&mut self, comb_use: Bits<3>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_FEATURE_MODE_CFG_COMB_USE__MASK, unsafe {
            DPU_RDMA_RDMA_FEATURE_MODE_CFG_COMB_USE(comb_use.val())
        })
    }

    /// Description: AXI burst length used by DPU_RDMA's reads.
    ///
    /// Bit width: 4 (bits 14:11)
    /// Range of values: 4'd3 Burst4; 4'd7 Burst8; 4'd15 Burst16.
    /// Known limitations: Only the three enumerated encodings (3, 7, 15) are valid; other values are undefined.
    /// Related registers: `ov4k_bypass` (RDMA_ERDMA_CFG) also affects burst splitting behavior.
    pub fn burst_len(&mut self, burst_len: Bits<4>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN__MASK, unsafe {
            DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN(burst_len.val())
        })
    }

    /// Description: Input data precision fed into DPU_RDMA.
    ///
    /// Bit width: 3 (bits 17:15)
    /// Range of values: 3'd0 Integer 8bit; 3'd1 Integer 16bit; 3'd2 Float point 16bit; 3'd3 Bfloat 16bit; 3'd4 Integer 32bit; 3'd5 Float point 32bit; 3'd6 Integer 4bit.
    /// Known limitations: None documented.
    /// Related registers: `proc_precision` (same register); `erdma_data_size` (RDMA_ERDMA_CFG) is ERDMA's own, separate precision selector.
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU_RDMA;
    const OFFSET: u32 = REG_DPU_RDMA_RDMA_SRC_DMA_CFG;
}

impl Register<DpuRdmaSrcDmaCfg> {
    /// Description: Un-pooling kernel width used by the un-pooling/upsample data path.
    ///
    /// Bit width: 3 (bits 2:0)
    /// Range of values: 0 to 7, N-1 encoded (actual kernel width minus 1).
    /// Known limitations: Only meaningful when `unpooling_en` is set.
    /// Related registers: `kernel_height`, `kernel_stride_width`, `kernel_stride_height`, `unpooling_en`, `pooling_method` (same register).
    pub fn kernel_width(&mut self, kernel_width: Bits<3>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_SRC_DMA_CFG_KERNEL_WIDTH__MASK, unsafe {
            DPU_RDMA_RDMA_SRC_DMA_CFG_KERNEL_WIDTH(kernel_width.val())
        })
    }

    /// Description: Un-pooling kernel height used by the un-pooling/upsample data path.
    ///
    /// Bit width: 3 (bits 5:3)
    /// Range of values: 0 to 7, N-1 encoded (actual kernel height minus 1).
    /// Known limitations: Only meaningful when `unpooling_en` is set.
    /// Related registers: `kernel_width`, `kernel_stride_width`, `kernel_stride_height`, `unpooling_en`, `pooling_method` (same register).
    pub fn kernel_height(&mut self, kernel_height: Bits<3>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_SRC_DMA_CFG_KERNEL_HEIGHT__MASK, unsafe {
            DPU_RDMA_RDMA_SRC_DMA_CFG_KERNEL_HEIGHT(kernel_height.val())
        })
    }

    /// Description: Un-pooling kernel stride width used by the un-pooling/upsample data path.
    ///
    /// Bit width: 3 (bits 8:6)
    /// Range of values: 0 to 7, N-1 encoded (actual stride width minus 1).
    /// Known limitations: Only meaningful when `unpooling_en` is set.
    /// Related registers: `kernel_width`, `kernel_height`, `kernel_stride_height`, `unpooling_en` (same register).
    pub fn kernel_stride_width(&mut self, kernel_stride_width: Bits<3>) -> &mut Self {
        self.set_field(
            DPU_RDMA_RDMA_SRC_DMA_CFG_KERNEL_STRIDE_WIDTH__MASK,
            unsafe { DPU_RDMA_RDMA_SRC_DMA_CFG_KERNEL_STRIDE_WIDTH(kernel_stride_width.val()) },
        )
    }

    /// Description: Un-pooling kernel stride height used by the un-pooling/upsample data path.
    ///
    /// Bit width: 3 (bits 11:9)
    /// Range of values: 0 to 7, N-1 encoded (actual stride height minus 1).
    /// Known limitations: Only meaningful when `unpooling_en` is set.
    /// Related registers: `kernel_width`, `kernel_height`, `kernel_stride_width`, `unpooling_en` (same register).
    pub fn kernel_stride_height(&mut self, kernel_stride_height: Bits<3>) -> &mut Self {
        self.set_field(
            DPU_RDMA_RDMA_SRC_DMA_CFG_KERNEL_STRIDE_HEIGHT__MASK,
            unsafe { DPU_RDMA_RDMA_SRC_DMA_CFG_KERNEL_STRIDE_HEIGHT(kernel_stride_height.val()) },
        )
    }

    /// Description: Enables the un-pooling (upsample) data path for this DPU_RDMA transfer.
    ///
    /// Bit width: 1 (bit 12)
    /// Range of values: 1'd0 disable; 1'd1 enable.
    /// Known limitations: When enabled, the kernel/stride fields and `pad_cfg` become meaningful; this doubles up the same register block used for the plain flying-mode data feed.
    /// Related registers: `kernel_width`/`kernel_height`/`kernel_stride_width`/`kernel_stride_height`/`pooling_method` (same register), `pad_left`/`pad_top`/`pad_value` (RDMA_PAD_CFG).
    pub fn unpooling_en(&mut self, unpooling_en: Bits<1>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_SRC_DMA_CFG_UNPOOLING_EN__MASK, unsafe {
            DPU_RDMA_RDMA_SRC_DMA_CFG_UNPOOLING_EN(unpooling_en.val())
        })
    }

    /// Description: Selects the pooling method the un-pooling path should invert/mirror.
    ///
    /// Bit width: 1 (bit 13)
    /// Range of values: 1'd0 average pooling (up-sampling can use this mode); 1'd1 min or max pooling.
    /// Known limitations: Only meaningful when `unpooling_en` is set.
    /// Related registers: `unpooling_en` (same register); PPU's own `ppu_operation_mode_cfg.pooling_method` for the forward pooling path.
    pub fn pooling_method(&mut self, pooling_method: Bits<1>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_SRC_DMA_CFG_POOLING_METHOD__MASK, unsafe {
            DPU_RDMA_RDMA_SRC_DMA_CFG_POOLING_METHOD(pooling_method.val())
        })
    }

    /// Description: Number of pixels from the end of the transfer's width to the end of the underlying shape's feature line.
    ///
    /// Bit width: 13 (bits 31:19)
    /// Range of values: 0 to 8191 (pixel count).
    /// Known limitations: None documented.
    /// Related registers: `surf_notch_addr` (RDMA_SURF_NOTCH) is the analogous end-of-surface bookkeeping field; `ew_line_notch_addr` (RDMA_DATA_CUBE_HEIGHT) is the EW-path equivalent.
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU_RDMA;
    const OFFSET: u32 = REG_DPU_RDMA_RDMA_SURF_NOTCH;
}

impl Register<DpuRdmaSurfNotch> {
    /// Description: Number of pixels from the end of this process's feature map to the end of the underlying shape's feature map.
    ///
    /// Bit width: 28 (bits 31:4)
    /// Range of values: 0 to 2^28-1 (pixel count).
    /// Known limitations: None documented.
    /// Related registers: `line_notch_addr` (RDMA_SRC_DMA_CFG), `ew_surf_notch` (RDMA_EW_SURF_NOTCH) are the analogous notch fields for the width-line and EW paths.
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU_RDMA;
    const OFFSET: u32 = REG_DPU_RDMA_RDMA_PAD_CFG;
}

impl Register<DpuRdmaPadCfg> {
    /// Description: Un-pooling left pad amount.
    ///
    /// Bit width: 3 (bits 2:0)
    /// Range of values: 0 to 7 (pixel count).
    /// Known limitations: Only meaningful when `unpooling_en` (RDMA_SRC_DMA_CFG) is set.
    /// Related registers: `pad_top`, `pad_value` (same register); `unpooling_en` (RDMA_SRC_DMA_CFG).
    pub fn pad_left(&mut self, pad_left: Bits<3>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_PAD_CFG_PAD_LEFT__MASK, unsafe {
            DPU_RDMA_RDMA_PAD_CFG_PAD_LEFT(pad_left.val())
        })
    }

    /// Description: Un-pooling top pad amount.
    ///
    /// Bit width: 3 (bits 6:4)
    /// Range of values: 0 to 7 (pixel count).
    /// Known limitations: Only meaningful when `unpooling_en` (RDMA_SRC_DMA_CFG) is set.
    /// Related registers: `pad_left`, `pad_value` (same register); `unpooling_en` (RDMA_SRC_DMA_CFG).
    pub fn pad_top(&mut self, pad_top: Bits<3>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_PAD_CFG_PAD_TOP__MASK, unsafe {
            DPU_RDMA_RDMA_PAD_CFG_PAD_TOP(pad_top.val())
        })
    }

    /// Description: Value used to fill padded pixels in the un-pooling data path.
    ///
    /// Bit width: 16 (bits 31:16)
    /// Range of values: 0 to 65535, interpreted per the configured input precision.
    /// Known limitations: Only meaningful when `unpooling_en` (RDMA_SRC_DMA_CFG) is set.
    /// Related registers: `pad_left`, `pad_top` (same register); `unpooling_en` (RDMA_SRC_DMA_CFG); compare CNA's own `cna_pad_con1.pad_value`.
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU_RDMA;
    const OFFSET: u32 = REG_DPU_RDMA_RDMA_WEIGHT;
}

impl Register<DpuRdmaWeight> {
    /// Description: AXI arbiter weight for MRDMA.
    ///
    /// Bit width: 8 (bits 7:0)
    /// Range of values: 0 to 255 (relative arbitration weight).
    /// Known limitations: Only relevant when MRDMA contends with BRDMA/NRDMA/ERDMA for AXI bandwidth; has no effect if `mrdma_disable` is set.
    /// Related registers: `b_weight`, `n_weight`, `e_weight` (same register); `mrdma_disable` (RDMA_FEATURE_MODE_CFG).
    pub fn m_weight(&mut self, m_weight: Bits<8>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_WEIGHT_M_WEIGHT__MASK, unsafe {
            DPU_RDMA_RDMA_WEIGHT_M_WEIGHT(m_weight.val())
        })
    }

    /// Description: AXI arbiter weight for BRDMA.
    ///
    /// Bit width: 8 (bits 15:8)
    /// Range of values: 0 to 255 (relative arbitration weight).
    /// Known limitations: Only relevant when BRDMA contends with MRDMA/NRDMA/ERDMA for AXI bandwidth.
    /// Related registers: `m_weight`, `n_weight`, `e_weight` (same register); `brdma_data_use` (RDMA_BRDMA_CFG).
    pub fn b_weight(&mut self, b_weight: Bits<8>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_WEIGHT_B_WEIGHT__MASK, unsafe {
            DPU_RDMA_RDMA_WEIGHT_B_WEIGHT(b_weight.val())
        })
    }

    /// Description: AXI arbiter weight for NRDMA.
    ///
    /// Bit width: 8 (bits 23:16)
    /// Range of values: 0 to 255 (relative arbitration weight).
    /// Known limitations: Only relevant when NRDMA contends with MRDMA/BRDMA/ERDMA for AXI bandwidth.
    /// Related registers: `m_weight`, `b_weight`, `e_weight` (same register); `nrdma_data_use` (RDMA_NRDMA_CFG).
    pub fn n_weight(&mut self, n_weight: Bits<8>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_WEIGHT_N_WEIGHT__MASK, unsafe {
            DPU_RDMA_RDMA_WEIGHT_N_WEIGHT(n_weight.val())
        })
    }

    /// Description: AXI arbiter weight for ERDMA.
    ///
    /// Bit width: 8 (bits 31:24)
    /// Range of values: 0 to 255 (relative arbitration weight).
    /// Known limitations: Only relevant when ERDMA contends with MRDMA/BRDMA/NRDMA for AXI bandwidth; has no effect if `erdma_disable` is set.
    /// Related registers: `m_weight`, `b_weight`, `n_weight` (same register); `erdma_disable` (RDMA_ERDMA_CFG).
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
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_DPU_RDMA;
    const OFFSET: u32 = REG_DPU_RDMA_RDMA_EW_SURF_NOTCH;
}

impl Register<DpuRdmaEwSurfNotch> {
    /// Description: Surface notch of the EW (element-wise) cube — end-of-surface bookkeeping analogous to `surf_notch_addr` but specific to the EW/ERDMA path.
    ///
    /// Bit width: 28 (bits 31:4)
    /// Range of values: 0 to 2^28-1 (pixel count).
    /// Known limitations: Only meaningful when ERDMA is enabled and feeding the EW stage.
    /// Related registers: `surf_notch_addr` (RDMA_SURF_NOTCH), `ew_surf_stride` (RDMA_EW_SURF_STRIDE), `ew_line_notch_addr` (RDMA_DATA_CUBE_HEIGHT).
    pub fn ew_surf_notch(&mut self, ew_surf_notch: Bits<28>) -> &mut Self {
        self.set_field(DPU_RDMA_RDMA_EW_SURF_NOTCH_EW_SURF_NOTCH__MASK, unsafe {
            DPU_RDMA_RDMA_EW_SURF_NOTCH_EW_SURF_NOTCH(ew_surf_notch.val())
        })
    }
}
