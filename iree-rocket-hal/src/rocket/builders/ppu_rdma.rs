use crate::rocket::{
    builders::{Bits, Register, RegisterMeta},
    registers::*,
};

// ========================================================================
// RDMA_S_STATUS (0x7000)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct PpuRdmaSStatus;

impl RegisterMeta for PpuRdmaSStatus {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_PPU_RDMA;
    const OFFSET: u32 = REG_PPU_RDMA_RDMA_S_STATUS;
}

impl Register<PpuRdmaSStatus> {
    /// Description: Reports executer 0's current operating state within PPU_RDMA's executer/pointer ping-pong pair.
    ///
    /// Bit width: 2
    /// Range of values: 2'd0 = executer 0 is in idle state; 2'd1 = executer 0 is operating; 2'd2 = executer 0 is operating, executer 1 is waiting to operate; 2'd3 = reserved.
    /// Known limitations: Read-only status bit (bits 1:0 per the TRM); this builder's field is a setter over the whole register value, but hardware treats the bit as RO.
    /// Related registers: status_1 (executer 1's equivalent status); PpuRdmaSPointer::executer (selects the active executer group).
    pub fn status_0(&mut self, status_0: Bits<2>) -> &mut Self {
        self.set_field(PPU_RDMA_RDMA_S_STATUS_STATUS_0__MASK, unsafe {
            PPU_RDMA_RDMA_S_STATUS_STATUS_0(status_0.val())
        })
    }

    /// Description: Reports executer 1's current operating state within PPU_RDMA's executer/pointer ping-pong pair.
    ///
    /// Bit width: 2
    /// Range of values: 2'd0 = executer 1 is in idle state; 2'd1 = executer 1 is operating; 2'd2 = executer 1 is operating, executer 1 is waiting to operate (as given verbatim in the TRM); 2'd3 = reserved.
    /// Known limitations: Read-only status bit (bits 17:16 per the TRM).
    /// Related registers: status_0 (executer 0's equivalent status); PpuRdmaSPointer::executer (selects the active executer group).
    pub fn status_1(&mut self, status_1: Bits<2>) -> &mut Self {
        self.set_field(PPU_RDMA_RDMA_S_STATUS_STATUS_1__MASK, unsafe {
            PPU_RDMA_RDMA_S_STATUS_STATUS_1(status_1.val())
        })
    }
}

// ========================================================================
// RDMA_S_POINTER (0x7004)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct PpuRdmaSPointer;

impl RegisterMeta for PpuRdmaSPointer {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_PPU_RDMA;
    const OFFSET: u32 = REG_PPU_RDMA_RDMA_S_POINTER;
}

impl Register<PpuRdmaSPointer> {
    /// Description: Selects which of the two shadow register groups (0 or 1) is ready to be applied on the next operation.
    ///
    /// Bit width: 1
    /// Range of values: 1'd0 = register group 0; 1'd1 = register group 1.
    /// Known limitations: Meaningful mainly when ping-pong is not fully automatic; when pointer_pp_en is set the hardware toggles this value itself per pointer_pp_mode.
    /// Related registers: pointer_pp_en, pointer_pp_mode, pointer_pp_clear.
    pub fn pointer(&mut self, pointer: Bits<1>) -> &mut Self {
        self.set_field(PPU_RDMA_RDMA_S_POINTER_POINTER__MASK, unsafe {
            PPU_RDMA_RDMA_S_POINTER_POINTER(pointer.val())
        })
    }

    /// Description: Enables ping-pong (double-buffered) operation of the register group pointer.
    ///
    /// Bit width: 1
    /// Range of values: 1'd0 = disable; 1'd1 = enable.
    /// Known limitations: None documented.
    /// Related registers: pointer, pointer_pp_mode, pointer_pp_clear.
    pub fn pointer_pp_en(&mut self, pointer_pp_en: Bits<1>) -> &mut Self {
        self.set_field(PPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN__MASK, unsafe {
            PPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN(pointer_pp_en.val())
        })
    }

    /// Description: Enables ping-pong (double-buffered) operation of the executer group.
    ///
    /// Bit width: 1
    /// Range of values: 1'd0 = disable; 1'd1 = enable.
    /// Known limitations: None documented.
    /// Related registers: executer, executer_pp_clear.
    pub fn executer_pp_en(&mut self, executer_pp_en: Bits<1>) -> &mut Self {
        self.set_field(PPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN__MASK, unsafe {
            PPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN(executer_pp_en.val())
        })
    }

    /// Description: Selects the toggle rule used for register-group ping-pong.
    ///
    /// Bit width: 1
    /// Range of values: 1'd0 = pointer toggles by executer (e.g. if current executer is 0, next pointer toggles to 1); 1'd1 = pointer toggles by pointer itself (e.g. if current pointer is 0, next pointer toggles to 1).
    /// Known limitations: Only has an effect when pointer_pp_en is enabled.
    /// Related registers: pointer, pointer_pp_en.
    pub fn pointer_pp_mode(&mut self, pointer_pp_mode: Bits<1>) -> &mut Self {
        self.set_field(PPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE__MASK, unsafe {
            PPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE(pointer_pp_mode.val())
        })
    }

    /// Description: Write 1 to reset the register-group pointer back to 0.
    ///
    /// Bit width: 1
    /// Range of values: W1C — write 1'd1 to clear the pointer to 0; reads back as 0.
    /// Known limitations: Write-1-to-clear; writing 0 has no effect.
    /// Related registers: pointer, pointer_pp_en.
    pub fn pointer_pp_clear(&mut self, pointer_pp_clear: Bits<1>) -> &mut Self {
        self.set_field(PPU_RDMA_RDMA_S_POINTER_POINTER_PP_CLEAR__MASK, unsafe {
            PPU_RDMA_RDMA_S_POINTER_POINTER_PP_CLEAR(pointer_pp_clear.val())
        })
    }

    /// Description: Write 1 to reset the executer-group pointer back to 0.
    ///
    /// Bit width: 1
    /// Range of values: W1C — write 1'd1 to clear the pointer to 0; reads back as 0.
    /// Known limitations: Write-1-to-clear; writing 0 has no effect.
    /// Related registers: executer, executer_pp_en.
    pub fn executer_pp_clear(&mut self, executer_pp_clear: Bits<1>) -> &mut Self {
        self.set_field(PPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_CLEAR__MASK, unsafe {
            PPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_CLEAR(executer_pp_clear.val())
        })
    }

    /// Description: Selects which executer register group is currently in use.
    ///
    /// Bit width: 1
    /// Range of values: 1'd0 = executer group 0; 1'd1 = executer group 1.
    /// Known limitations: The TRM marks this bit (16) as read-only status; this builder exposes it as a settable field regardless, presumably for pre-loading the shadow register before hardware takes ownership of ping-pong toggling.
    /// Related registers: executer_pp_en, executer_pp_clear, status_0/status_1 on PpuRdmaSStatus.
    pub fn executer(&mut self, executer: Bits<1>) -> &mut Self {
        self.set_field(PPU_RDMA_RDMA_S_POINTER_EXECUTER__MASK, unsafe {
            PPU_RDMA_RDMA_S_POINTER_EXECUTER(executer.val())
        })
    }
}

// ========================================================================
// RDMA_OPERATION_ENABLE (0x7008)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct PpuRdmaOperationEnable;

impl RegisterMeta for PpuRdmaOperationEnable {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_PPU_RDMA;
    const OFFSET: u32 = REG_PPU_RDMA_RDMA_OPERATION_ENABLE;
}

impl Register<PpuRdmaOperationEnable> {
    /// Description: Triggers the PPU_RDMA block to begin operating using the currently latched register group.
    ///
    /// Bit width: 1
    /// Range of values: 1'd0 = disable; 1'd1 = enable.
    /// Known limitations: This register and every register after it in the block are shadowed for ping-pong operation per the TRM; only takes effect once PPU's own flying_mode is set to fetch from PPU_RDMA.
    /// Related registers: PpuRdmaSPointer fields (ping-pong control); ppu_operation_mode_cfg.flying_mode (in the PPU block, selects whether PPU sources data from PPU_RDMA); GLOBAL.global_operation_enable ppu_rdma_op_en bit.
    pub fn op_en(&mut self, op_en: Bits<1>) -> &mut Self {
        self.set_field(PPU_RDMA_RDMA_OPERATION_ENABLE_OP_EN__MASK, unsafe {
            PPU_RDMA_RDMA_OPERATION_ENABLE_OP_EN(op_en.val())
        })
    }
}

// ========================================================================
// RDMA_CUBE_IN_WIDTH (0x700C)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct PpuRdmaCubeInWidth;

impl RegisterMeta for PpuRdmaCubeInWidth {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_PPU_RDMA;
    const OFFSET: u32 = REG_PPU_RDMA_RDMA_CUBE_IN_WIDTH;
}

impl Register<PpuRdmaCubeInWidth> {
    /// Description: Sets the width of the pooling cube fetched by PPU_RDMA, encoded as (width - 1).
    ///
    /// Bit width: 13
    /// Range of values: 0x0000-0x1FFF, representing an actual width of 1 to 8192 (value = width - 1).
    /// Known limitations: Only used when PPU is in flying mode fed by PPU_RDMA.
    /// Related registers: cube_in_height, cube_in_channel, src_line_stride/src_surf_stride (must be consistent with this shape plus any virtual-box padding).
    pub fn cube_in_width(&mut self, cube_in_width: Bits<13>) -> &mut Self {
        self.set_field(PPU_RDMA_RDMA_CUBE_IN_WIDTH_CUBE_IN_WIDTH__MASK, unsafe {
            PPU_RDMA_RDMA_CUBE_IN_WIDTH_CUBE_IN_WIDTH(cube_in_width.val())
        })
    }
}

// ========================================================================
// RDMA_CUBE_IN_HEIGHT (0x7010)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct PpuRdmaCubeInHeight;

impl RegisterMeta for PpuRdmaCubeInHeight {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_PPU_RDMA;
    const OFFSET: u32 = REG_PPU_RDMA_RDMA_CUBE_IN_HEIGHT;
}

impl Register<PpuRdmaCubeInHeight> {
    /// Description: Sets the height of the pooling cube fetched by PPU_RDMA, encoded as (height - 1).
    ///
    /// Bit width: 13
    /// Range of values: 0x0000-0x1FFF, representing an actual height of 1 to 8192 (value = height - 1).
    /// Known limitations: Only used when PPU is in flying mode fed by PPU_RDMA.
    /// Related registers: cube_in_width, cube_in_channel, src_line_stride/src_surf_stride (must be consistent with this shape plus any virtual-box padding).
    pub fn cube_in_height(&mut self, cube_in_height: Bits<13>) -> &mut Self {
        self.set_field(PPU_RDMA_RDMA_CUBE_IN_HEIGHT_CUBE_IN_HEIGHT__MASK, unsafe {
            PPU_RDMA_RDMA_CUBE_IN_HEIGHT_CUBE_IN_HEIGHT(cube_in_height.val())
        })
    }
}

// ========================================================================
// RDMA_CUBE_IN_CHANNEL (0x7014)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct PpuRdmaCubeInChannel;

impl RegisterMeta for PpuRdmaCubeInChannel {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_PPU_RDMA;
    const OFFSET: u32 = REG_PPU_RDMA_RDMA_CUBE_IN_CHANNEL;
}

impl Register<PpuRdmaCubeInChannel> {
    /// Description: Sets the channel count of the pooling cube fetched by PPU_RDMA, encoded as (channel - 1).
    ///
    /// Bit width: 13
    /// Range of values: 0x0000-0x1FFF, representing an actual channel count of 1 to 8192 (value = channel - 1).
    /// Known limitations: Only used when PPU is in flying mode fed by PPU_RDMA.
    /// Related registers: cube_in_width, cube_in_height, src_surf_stride (must be consistent with this shape plus any virtual-box padding).
    pub fn cube_in_channel(&mut self, cube_in_channel: Bits<13>) -> &mut Self {
        self.set_field(
            PPU_RDMA_RDMA_CUBE_IN_CHANNEL_CUBE_IN_CHANNEL__MASK,
            unsafe { PPU_RDMA_RDMA_CUBE_IN_CHANNEL_CUBE_IN_CHANNEL(cube_in_channel.val()) },
        )
    }
}

// ========================================================================
// RDMA_SRC_BASE_ADDR (0x701C)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct PpuRdmaSrcBaseAddr;

impl RegisterMeta for PpuRdmaSrcBaseAddr {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_PPU_RDMA;
    const OFFSET: u32 = REG_PPU_RDMA_RDMA_SRC_BASE_ADDR;
}

impl Register<PpuRdmaSrcBaseAddr> {
    /// Description: Base address in system memory of the pooling cube data that PPU_RDMA fetches.
    ///
    /// Bit width: 32
    /// Range of values: Any 32-bit byte address.
    /// Known limitations: Only used when PPU is in flying mode fed by PPU_RDMA.
    /// Related registers: cube_in_width/height/channel (shape read from this address); src_line_stride, src_surf_stride (row/surface layout at this address).
    pub fn src_base_addr(&mut self, src_base_addr: Bits<32>) -> &mut Self {
        self.set_field(PPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__MASK, unsafe {
            PPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR(src_base_addr.val())
        })
    }
}

// ========================================================================
// RDMA_SRC_LINE_STRIDE (0x7024)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct PpuRdmaSrcLineStride;

impl RegisterMeta for PpuRdmaSrcLineStride {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_PPU_RDMA;
    const OFFSET: u32 = REG_PPU_RDMA_RDMA_SRC_LINE_STRIDE;
}

impl Register<PpuRdmaSrcLineStride> {
    /// Description: Stride (in bytes) between consecutive rows of the source pooling cube, including any "virtual box" padding beyond the logical cube width.
    ///
    /// Bit width: 28
    /// Range of values: 0 to 2^28-1, in 16-byte units. Pass `stride_bytes / 16`;
    /// the builder shifts that logical field value into register bits 31:4.
    /// Known limitations: The byte stride must be 16-byte aligned and must account for
    /// padding beyond the actual cube shape ("including Virtual box" per the TRM notes).
    /// Passing the already encoded register word would shift it a second time.
    /// Related registers: cube_in_width, src_surf_stride, src_base_addr.
    pub fn src_line_stride(&mut self, src_line_stride: Bits<28>) -> &mut Self {
        self.set_field(
            PPU_RDMA_RDMA_SRC_LINE_STRIDE_SRC_LINE_STRIDE__MASK,
            unsafe { PPU_RDMA_RDMA_SRC_LINE_STRIDE_SRC_LINE_STRIDE(src_line_stride.val()) },
        )
    }
}

// ========================================================================
// RDMA_SRC_SURF_STRIDE (0x7028)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct PpuRdmaSrcSurfStride;

impl RegisterMeta for PpuRdmaSrcSurfStride {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_PPU_RDMA;
    const OFFSET: u32 = REG_PPU_RDMA_RDMA_SRC_SURF_STRIDE;
}

impl Register<PpuRdmaSrcSurfStride> {
    /// Description: Stride (in bytes) between consecutive surfaces/planes of the source pooling cube, including any "virtual box" padding beyond the logical cube area.
    ///
    /// Bit width: 28
    /// Range of values: 0 to 2^28-1, in 16-byte units. Pass `stride_bytes / 16`;
    /// the builder shifts that logical field value into register bits 31:4.
    /// Known limitations: The byte stride must be 16-byte aligned and must account for
    /// padding beyond the actual cube shape ("including Virtual box" per the TRM notes).
    /// Passing the already encoded register word would shift it a second time.
    /// Related registers: cube_in_channel, src_line_stride, src_base_addr.
    pub fn src_surf_stride(&mut self, src_surf_stride: Bits<28>) -> &mut Self {
        self.set_field(
            PPU_RDMA_RDMA_SRC_SURF_STRIDE_SRC_SURF_STRIDE__MASK,
            unsafe { PPU_RDMA_RDMA_SRC_SURF_STRIDE_SRC_SURF_STRIDE(src_surf_stride.val()) },
        )
    }
}

// ========================================================================
// RDMA_DATA_FORMAT (0x7030)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct PpuRdmaDataFormat;

impl RegisterMeta for PpuRdmaDataFormat {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_PPU_RDMA;
    const OFFSET: u32 = REG_PPU_RDMA_RDMA_DATA_FORMAT;
}

impl Register<PpuRdmaDataFormat> {
    /// Description: Selects the input data precision of the source pooling cube fetched by PPU_RDMA.
    ///
    /// Bit width: 2
    /// Range of values: 2'd0 = 4bit; 2'd1 = 8bit; 2'd2 = 16bit; 2'd3 = 32bit.
    /// Known limitations: Only used when PPU is in flying mode fed by PPU_RDMA.
    /// Related registers: proc_precision on the main PPU block's ppu_data_format register (must be kept consistent with this field); src_base_addr/src_line_stride/src_surf_stride (byte layout depends on this precision).
    pub fn in_precision(&mut self, in_precision: Bits<2>) -> &mut Self {
        self.set_field(PPU_RDMA_RDMA_DATA_FORMAT_IN_PRECISION__MASK, unsafe {
            PPU_RDMA_RDMA_DATA_FORMAT_IN_PRECISION(in_precision.val())
        })
    }
}
