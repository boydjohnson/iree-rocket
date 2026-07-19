use crate::rocket::{
    builders::{Bits, Register, RegisterMeta},
    registers::*,
};

// target_DDMA (0x10000) doesn't fit in the 16-bit domain field the
// RegCmd packing scheme uses (bits 48-63) -- same problem as
// DOMAIN_GLOBAL, see builders.rs. Unfixed here since nothing currently
// uses this module; flagged so it doesn't look silently correct if that
// changes.

// ========================================================================
// CFG_OUTSTANDING (0x8000)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DdmaCfgOutstanding;

impl RegisterMeta for DdmaCfgOutstanding {
    const DOMAIN: u32 = target_DDMA;
    const OFFSET: u32 = REG_DDMA_CFG_OUTSTANDING;
}

impl Register<DdmaCfgOutstanding> {
    pub fn rd_os_cnt(&mut self, rd_os_cnt: Bits<8>) -> &mut Self {
        self.set_field(DDMA_CFG_OUTSTANDING_RD_OS_CNT__MASK, unsafe {
            DDMA_CFG_OUTSTANDING_RD_OS_CNT(rd_os_cnt.val())
        })
    }

    pub fn wr_os_cnt(&mut self, wr_os_cnt: Bits<8>) -> &mut Self {
        self.set_field(DDMA_CFG_OUTSTANDING_WR_OS_CNT__MASK, unsafe {
            DDMA_CFG_OUTSTANDING_WR_OS_CNT(wr_os_cnt.val())
        })
    }
}

// ========================================================================
// RD_WEIGHT_0 (0x8004)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DdmaRdWeight0;

impl RegisterMeta for DdmaRdWeight0 {
    const DOMAIN: u32 = target_DDMA;
    const OFFSET: u32 = REG_DDMA_RD_WEIGHT_0;
}

impl Register<DdmaRdWeight0> {
    pub fn rd_weight_feature(&mut self, rd_weight_feature: Bits<8>) -> &mut Self {
        self.set_field(DDMA_RD_WEIGHT_0_RD_WEIGHT_FEATURE__MASK, unsafe {
            DDMA_RD_WEIGHT_0_RD_WEIGHT_FEATURE(rd_weight_feature.val())
        })
    }

    pub fn rd_weight_kernel(&mut self, rd_weight_kernel: Bits<8>) -> &mut Self {
        self.set_field(DDMA_RD_WEIGHT_0_RD_WEIGHT_KERNEL__MASK, unsafe {
            DDMA_RD_WEIGHT_0_RD_WEIGHT_KERNEL(rd_weight_kernel.val())
        })
    }

    pub fn rd_weight_dpu(&mut self, rd_weight_dpu: Bits<8>) -> &mut Self {
        self.set_field(DDMA_RD_WEIGHT_0_RD_WEIGHT_DPU__MASK, unsafe {
            DDMA_RD_WEIGHT_0_RD_WEIGHT_DPU(rd_weight_dpu.val())
        })
    }

    pub fn rd_weight_pdp(&mut self, rd_weight_pdp: Bits<8>) -> &mut Self {
        self.set_field(DDMA_RD_WEIGHT_0_RD_WEIGHT_PDP__MASK, unsafe {
            DDMA_RD_WEIGHT_0_RD_WEIGHT_PDP(rd_weight_pdp.val())
        })
    }
}

// ========================================================================
// WR_WEIGHT_0 (0x8008)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DdmaWrWeight0;

impl RegisterMeta for DdmaWrWeight0 {
    const DOMAIN: u32 = target_DDMA;
    const OFFSET: u32 = REG_DDMA_WR_WEIGHT_0;
}

impl Register<DdmaWrWeight0> {
    pub fn wr_weight_dpu(&mut self, wr_weight_dpu: Bits<8>) -> &mut Self {
        self.set_field(DDMA_WR_WEIGHT_0_WR_WEIGHT_DPU__MASK, unsafe {
            DDMA_WR_WEIGHT_0_WR_WEIGHT_DPU(wr_weight_dpu.val())
        })
    }

    pub fn wr_weight_pdp(&mut self, wr_weight_pdp: Bits<8>) -> &mut Self {
        self.set_field(DDMA_WR_WEIGHT_0_WR_WEIGHT_PDP__MASK, unsafe {
            DDMA_WR_WEIGHT_0_WR_WEIGHT_PDP(wr_weight_pdp.val())
        })
    }
}

// ========================================================================
// CFG_ID_ERROR (0x800C)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DdmaCfgIdError;

impl RegisterMeta for DdmaCfgIdError {
    const DOMAIN: u32 = target_DDMA;
    const OFFSET: u32 = REG_DDMA_CFG_ID_ERROR;
}

impl Register<DdmaCfgIdError> {
    pub fn rd_resp_id(&mut self, rd_resp_id: Bits<5>) -> &mut Self {
        self.set_field(DDMA_CFG_ID_ERROR_RD_RESP_ID__MASK, unsafe {
            DDMA_CFG_ID_ERROR_RD_RESP_ID(rd_resp_id.val())
        })
    }

    pub fn wr_resp_id(&mut self, wr_resp_id: Bits<4>) -> &mut Self {
        self.set_field(DDMA_CFG_ID_ERROR_WR_RESP_ID__MASK, unsafe {
            DDMA_CFG_ID_ERROR_WR_RESP_ID(wr_resp_id.val())
        })
    }
}

// ========================================================================
// RD_WEIGHT_1 (0x8010)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DdmaRdWeight1;

impl RegisterMeta for DdmaRdWeight1 {
    const DOMAIN: u32 = target_DDMA;
    const OFFSET: u32 = REG_DDMA_RD_WEIGHT_1;
}

impl Register<DdmaRdWeight1> {
    pub fn rd_weight_pc(&mut self, rd_weight_pc: Bits<8>) -> &mut Self {
        self.set_field(DDMA_RD_WEIGHT_1_RD_WEIGHT_PC__MASK, unsafe {
            DDMA_RD_WEIGHT_1_RD_WEIGHT_PC(rd_weight_pc.val())
        })
    }
}

// ========================================================================
// CFG_DMA_FIFO_CLR (0x8014)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DdmaCfgDmaFifoClr;

impl RegisterMeta for DdmaCfgDmaFifoClr {
    const DOMAIN: u32 = target_DDMA;
    const OFFSET: u32 = REG_DDMA_CFG_DMA_FIFO_CLR;
}

impl Register<DdmaCfgDmaFifoClr> {
    pub fn dma_fifo_clr(&mut self, dma_fifo_clr: Bits<1>) -> &mut Self {
        self.set_field(DDMA_CFG_DMA_FIFO_CLR_DMA_FIFO_CLR__MASK, unsafe {
            DDMA_CFG_DMA_FIFO_CLR_DMA_FIFO_CLR(dma_fifo_clr.val())
        })
    }
}

// ========================================================================
// CFG_DMA_ARB (0x8018)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DdmaCfgDmaArb;

impl RegisterMeta for DdmaCfgDmaArb {
    const DOMAIN: u32 = target_DDMA;
    const OFFSET: u32 = REG_DDMA_CFG_DMA_ARB;
}

impl Register<DdmaCfgDmaArb> {
    pub fn rd_fix_arb(&mut self, rd_fix_arb: Bits<3>) -> &mut Self {
        self.set_field(DDMA_CFG_DMA_ARB_RD_FIX_ARB__MASK, unsafe {
            DDMA_CFG_DMA_ARB_RD_FIX_ARB(rd_fix_arb.val())
        })
    }

    pub fn wr_fix_arb(&mut self, wr_fix_arb: Bits<3>) -> &mut Self {
        self.set_field(DDMA_CFG_DMA_ARB_WR_FIX_ARB__MASK, unsafe {
            DDMA_CFG_DMA_ARB_WR_FIX_ARB(wr_fix_arb.val())
        })
    }

    pub fn rd_arbit_model(&mut self, rd_arbit_model: Bits<1>) -> &mut Self {
        self.set_field(DDMA_CFG_DMA_ARB_RD_ARBIT_MODEL__MASK, unsafe {
            DDMA_CFG_DMA_ARB_RD_ARBIT_MODEL(rd_arbit_model.val())
        })
    }

    pub fn wr_arbit_model(&mut self, wr_arbit_model: Bits<1>) -> &mut Self {
        self.set_field(DDMA_CFG_DMA_ARB_WR_ARBIT_MODEL__MASK, unsafe {
            DDMA_CFG_DMA_ARB_WR_ARBIT_MODEL(wr_arbit_model.val())
        })
    }
}

// ========================================================================
// CFG_DMA_RD_QOS (0x8020)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DdmaCfgDmaRdQos;

impl RegisterMeta for DdmaCfgDmaRdQos {
    const DOMAIN: u32 = target_DDMA;
    const OFFSET: u32 = REG_DDMA_CFG_DMA_RD_QOS;
}

impl Register<DdmaCfgDmaRdQos> {
    pub fn rd_feature_qos(&mut self, rd_feature_qos: Bits<2>) -> &mut Self {
        self.set_field(DDMA_CFG_DMA_RD_QOS_RD_FEATURE_QOS__MASK, unsafe {
            DDMA_CFG_DMA_RD_QOS_RD_FEATURE_QOS(rd_feature_qos.val())
        })
    }

    pub fn rd_kernel_qos(&mut self, rd_kernel_qos: Bits<2>) -> &mut Self {
        self.set_field(DDMA_CFG_DMA_RD_QOS_RD_KERNEL_QOS__MASK, unsafe {
            DDMA_CFG_DMA_RD_QOS_RD_KERNEL_QOS(rd_kernel_qos.val())
        })
    }

    pub fn rd_dpu_qos(&mut self, rd_dpu_qos: Bits<2>) -> &mut Self {
        self.set_field(DDMA_CFG_DMA_RD_QOS_RD_DPU_QOS__MASK, unsafe {
            DDMA_CFG_DMA_RD_QOS_RD_DPU_QOS(rd_dpu_qos.val())
        })
    }

    pub fn rd_ppu_qos(&mut self, rd_ppu_qos: Bits<2>) -> &mut Self {
        self.set_field(DDMA_CFG_DMA_RD_QOS_RD_PPU_QOS__MASK, unsafe {
            DDMA_CFG_DMA_RD_QOS_RD_PPU_QOS(rd_ppu_qos.val())
        })
    }

    pub fn rd_pc_qos(&mut self, rd_pc_qos: Bits<2>) -> &mut Self {
        self.set_field(DDMA_CFG_DMA_RD_QOS_RD_PC_QOS__MASK, unsafe {
            DDMA_CFG_DMA_RD_QOS_RD_PC_QOS(rd_pc_qos.val())
        })
    }
}

// ========================================================================
// CFG_DMA_RD_CFG (0x8024)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DdmaCfgDmaRdCfg;

impl RegisterMeta for DdmaCfgDmaRdCfg {
    const DOMAIN: u32 = target_DDMA;
    const OFFSET: u32 = REG_DDMA_CFG_DMA_RD_CFG;
}

impl Register<DdmaCfgDmaRdCfg> {
    pub fn rd_arsize(&mut self, rd_arsize: Bits<3>) -> &mut Self {
        self.set_field(DDMA_CFG_DMA_RD_CFG_RD_ARSIZE__MASK, unsafe {
            DDMA_CFG_DMA_RD_CFG_RD_ARSIZE(rd_arsize.val())
        })
    }

    pub fn rd_arburst(&mut self, rd_arburst: Bits<2>) -> &mut Self {
        self.set_field(DDMA_CFG_DMA_RD_CFG_RD_ARBURST__MASK, unsafe {
            DDMA_CFG_DMA_RD_CFG_RD_ARBURST(rd_arburst.val())
        })
    }

    pub fn rd_arprot(&mut self, rd_arprot: Bits<3>) -> &mut Self {
        self.set_field(DDMA_CFG_DMA_RD_CFG_RD_ARPROT__MASK, unsafe {
            DDMA_CFG_DMA_RD_CFG_RD_ARPROT(rd_arprot.val())
        })
    }

    pub fn rd_arcache(&mut self, rd_arcache: Bits<4>) -> &mut Self {
        self.set_field(DDMA_CFG_DMA_RD_CFG_RD_ARCACHE__MASK, unsafe {
            DDMA_CFG_DMA_RD_CFG_RD_ARCACHE(rd_arcache.val())
        })
    }

    pub fn rd_arlock(&mut self, rd_arlock: Bits<1>) -> &mut Self {
        self.set_field(DDMA_CFG_DMA_RD_CFG_RD_ARLOCK__MASK, unsafe {
            DDMA_CFG_DMA_RD_CFG_RD_ARLOCK(rd_arlock.val())
        })
    }
}

// ========================================================================
// CFG_DMA_WR_CFG (0x8028)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DdmaCfgDmaWrCfg;

impl RegisterMeta for DdmaCfgDmaWrCfg {
    const DOMAIN: u32 = target_DDMA;
    const OFFSET: u32 = REG_DDMA_CFG_DMA_WR_CFG;
}

impl Register<DdmaCfgDmaWrCfg> {
    pub fn wr_awsize(&mut self, wr_awsize: Bits<3>) -> &mut Self {
        self.set_field(DDMA_CFG_DMA_WR_CFG_WR_AWSIZE__MASK, unsafe {
            DDMA_CFG_DMA_WR_CFG_WR_AWSIZE(wr_awsize.val())
        })
    }

    pub fn wr_awburst(&mut self, wr_awburst: Bits<2>) -> &mut Self {
        self.set_field(DDMA_CFG_DMA_WR_CFG_WR_AWBURST__MASK, unsafe {
            DDMA_CFG_DMA_WR_CFG_WR_AWBURST(wr_awburst.val())
        })
    }

    pub fn wr_awprot(&mut self, wr_awprot: Bits<3>) -> &mut Self {
        self.set_field(DDMA_CFG_DMA_WR_CFG_WR_AWPROT__MASK, unsafe {
            DDMA_CFG_DMA_WR_CFG_WR_AWPROT(wr_awprot.val())
        })
    }

    pub fn wr_awcache(&mut self, wr_awcache: Bits<4>) -> &mut Self {
        self.set_field(DDMA_CFG_DMA_WR_CFG_WR_AWCACHE__MASK, unsafe {
            DDMA_CFG_DMA_WR_CFG_WR_AWCACHE(wr_awcache.val())
        })
    }

    pub fn wr_awlock(&mut self, wr_awlock: Bits<1>) -> &mut Self {
        self.set_field(DDMA_CFG_DMA_WR_CFG_WR_AWLOCK__MASK, unsafe {
            DDMA_CFG_DMA_WR_CFG_WR_AWLOCK(wr_awlock.val())
        })
    }
}

// ========================================================================
// CFG_DMA_WSTRB (0x802C)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DdmaCfgDmaWstrb;

impl RegisterMeta for DdmaCfgDmaWstrb {
    const DOMAIN: u32 = target_DDMA;
    const OFFSET: u32 = REG_DDMA_CFG_DMA_WSTRB;
}

impl Register<DdmaCfgDmaWstrb> {
    pub fn wr_wstrb(&mut self, wr_wstrb: Bits<32>) -> &mut Self {
        self.set_field(DDMA_CFG_DMA_WSTRB_WR_WSTRB__MASK, unsafe {
            DDMA_CFG_DMA_WSTRB_WR_WSTRB(wr_wstrb.val())
        })
    }
}

// ========================================================================
// CFG_STATUS (0x8030)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct DdmaCfgStatus;

impl RegisterMeta for DdmaCfgStatus {
    const DOMAIN: u32 = target_DDMA;
    const OFFSET: u32 = REG_DDMA_CFG_STATUS;
}

impl Register<DdmaCfgStatus> {
    pub fn idel(&mut self, idel: Bits<1>) -> &mut Self {
        self.set_field(DDMA_CFG_STATUS_IDEL__MASK, unsafe {
            DDMA_CFG_STATUS_IDEL(idel.val())
        })
    }
}
