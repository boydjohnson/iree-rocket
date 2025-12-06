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
    const DOMAIN: u32 = target_PPU_RDMA;
    const OFFSET: u32 = REG_PPU_RDMA_RDMA_S_STATUS;
}

impl Register<PpuRdmaSStatus> {
    pub fn status_0(&mut self, status_0: Bits<2>) -> &mut Self {
        self.set_field(PPU_RDMA_RDMA_S_STATUS_STATUS_0__MASK, unsafe {
            PPU_RDMA_RDMA_S_STATUS_STATUS_0(status_0.val())
        })
    }

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
    const DOMAIN: u32 = target_PPU_RDMA;
    const OFFSET: u32 = REG_PPU_RDMA_RDMA_S_POINTER;
}

impl Register<PpuRdmaSPointer> {
    pub fn pointer(&mut self, pointer: Bits<1>) -> &mut Self {
        self.set_field(PPU_RDMA_RDMA_S_POINTER_POINTER__MASK, unsafe {
            PPU_RDMA_RDMA_S_POINTER_POINTER(pointer.val())
        })
    }

    pub fn pointer_pp_en(&mut self, pointer_pp_en: Bits<1>) -> &mut Self {
        self.set_field(PPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN__MASK, unsafe {
            PPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN(pointer_pp_en.val())
        })
    }

    pub fn executer_pp_en(&mut self, executer_pp_en: Bits<1>) -> &mut Self {
        self.set_field(PPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN__MASK, unsafe {
            PPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN(executer_pp_en.val())
        })
    }

    pub fn pointer_pp_mode(&mut self, pointer_pp_mode: Bits<1>) -> &mut Self {
        self.set_field(PPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE__MASK, unsafe {
            PPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE(pointer_pp_mode.val())
        })
    }

    pub fn pointer_pp_clear(&mut self, pointer_pp_clear: Bits<1>) -> &mut Self {
        self.set_field(PPU_RDMA_RDMA_S_POINTER_POINTER_PP_CLEAR__MASK, unsafe {
            PPU_RDMA_RDMA_S_POINTER_POINTER_PP_CLEAR(pointer_pp_clear.val())
        })
    }

    pub fn executer_pp_clear(&mut self, executer_pp_clear: Bits<1>) -> &mut Self {
        self.set_field(PPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_CLEAR__MASK, unsafe {
            PPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_CLEAR(executer_pp_clear.val())
        })
    }

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
    const DOMAIN: u32 = target_PPU_RDMA;
    const OFFSET: u32 = REG_PPU_RDMA_RDMA_OPERATION_ENABLE;
}

impl Register<PpuRdmaOperationEnable> {
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
    const DOMAIN: u32 = target_PPU_RDMA;
    const OFFSET: u32 = REG_PPU_RDMA_RDMA_CUBE_IN_WIDTH;
}

impl Register<PpuRdmaCubeInWidth> {
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
    const DOMAIN: u32 = target_PPU_RDMA;
    const OFFSET: u32 = REG_PPU_RDMA_RDMA_CUBE_IN_HEIGHT;
}

impl Register<PpuRdmaCubeInHeight> {
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
    const DOMAIN: u32 = target_PPU_RDMA;
    const OFFSET: u32 = REG_PPU_RDMA_RDMA_CUBE_IN_CHANNEL;
}

impl Register<PpuRdmaCubeInChannel> {
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
    const DOMAIN: u32 = target_PPU_RDMA;
    const OFFSET: u32 = REG_PPU_RDMA_RDMA_SRC_BASE_ADDR;
}

impl Register<PpuRdmaSrcBaseAddr> {
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
    const DOMAIN: u32 = target_PPU_RDMA;
    const OFFSET: u32 = REG_PPU_RDMA_RDMA_SRC_LINE_STRIDE;
}

impl Register<PpuRdmaSrcLineStride> {
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
    const DOMAIN: u32 = target_PPU_RDMA;
    const OFFSET: u32 = REG_PPU_RDMA_RDMA_SRC_SURF_STRIDE;
}

impl Register<PpuRdmaSrcSurfStride> {
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
    const DOMAIN: u32 = target_PPU_RDMA;
    const OFFSET: u32 = REG_PPU_RDMA_RDMA_DATA_FORMAT;
}

impl Register<PpuRdmaDataFormat> {
    pub fn in_precision(&mut self, in_precision: Bits<2>) -> &mut Self {
        self.set_field(PPU_RDMA_RDMA_DATA_FORMAT_IN_PRECISION__MASK, unsafe {
            PPU_RDMA_RDMA_DATA_FORMAT_IN_PRECISION(in_precision.val())
        })
    }
}
