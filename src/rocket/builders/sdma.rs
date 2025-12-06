use crate::rocket::{
    builders::{Bits, Register, RegisterMeta},
    registers::*,
};

// ========================================================================
// CFG_OUTSTANDING (0x9000)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct SdmaCfgOutstanding;

impl RegisterMeta for SdmaCfgOutstanding {
    const DOMAIN: u32 = target_SDMA;
    const OFFSET: u32 = REG_SDMA_CFG_OUTSTANDING;
}

impl Register<SdmaCfgOutstanding> {
    pub fn rd_os_cnt(&mut self, rd_os_cnt: Bits<8>) -> &mut Self {
        self.set_field(SDMA_CFG_OUTSTANDING_RD_OS_CNT__MASK, unsafe {
            SDMA_CFG_OUTSTANDING_RD_OS_CNT(rd_os_cnt.val())
        })
    }

    pub fn wr_os_cnt(&mut self, wr_os_cnt: Bits<8>) -> &mut Self {
        self.set_field(SDMA_CFG_OUTSTANDING_WR_OS_CNT__MASK, unsafe {
            SDMA_CFG_OUTSTANDING_WR_OS_CNT(wr_os_cnt.val())
        })
    }
}

// ========================================================================
// RD_WEIGHT_0 (0x9004)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct SdmaRdWeight0;

impl RegisterMeta for SdmaRdWeight0 {
    const DOMAIN: u32 = target_SDMA;
    const OFFSET: u32 = REG_SDMA_RD_WEIGHT_0;
}

impl Register<SdmaRdWeight0> {
    pub fn rd_weight_feature(&mut self, rd_weight_feature: Bits<8>) -> &mut Self {
        self.set_field(SDMA_RD_WEIGHT_0_RD_WEIGHT_FEATURE__MASK, unsafe {
            SDMA_RD_WEIGHT_0_RD_WEIGHT_FEATURE(rd_weight_feature.val())
        })
    }

    pub fn rd_weight_kernel(&mut self, rd_weight_kernel: Bits<8>) -> &mut Self {
        self.set_field(SDMA_RD_WEIGHT_0_RD_WEIGHT_KERNEL__MASK, unsafe {
            SDMA_RD_WEIGHT_0_RD_WEIGHT_KERNEL(rd_weight_kernel.val())
        })
    }

    pub fn rd_weight_dpu(&mut self, rd_weight_dpu: Bits<8>) -> &mut Self {
        self.set_field(SDMA_RD_WEIGHT_0_RD_WEIGHT_DPU__MASK, unsafe {
            SDMA_RD_WEIGHT_0_RD_WEIGHT_DPU(rd_weight_dpu.val())
        })
    }

    pub fn rd_weight_pdp(&mut self, rd_weight_pdp: Bits<8>) -> &mut Self {
        self.set_field(SDMA_RD_WEIGHT_0_RD_WEIGHT_PDP__MASK, unsafe {
            SDMA_RD_WEIGHT_0_RD_WEIGHT_PDP(rd_weight_pdp.val())
        })
    }
}

// ========================================================================
// WR_WEIGHT_0 (0x9008)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct SdmaWrWeight0;

impl RegisterMeta for SdmaWrWeight0 {
    const DOMAIN: u32 = target_SDMA;
    const OFFSET: u32 = REG_SDMA_WR_WEIGHT_0;
}

impl Register<SdmaWrWeight0> {
    pub fn wr_weight_dpu(&mut self, wr_weight_dpu: Bits<8>) -> &mut Self {
        self.set_field(SDMA_WR_WEIGHT_0_WR_WEIGHT_DPU__MASK, unsafe {
            SDMA_WR_WEIGHT_0_WR_WEIGHT_DPU(wr_weight_dpu.val())
        })
    }

    pub fn wr_weight_pdp(&mut self, wr_weight_pdp: Bits<8>) -> &mut Self {
        self.set_field(SDMA_WR_WEIGHT_0_WR_WEIGHT_PDP__MASK, unsafe {
            SDMA_WR_WEIGHT_0_WR_WEIGHT_PDP(wr_weight_pdp.val())
        })
    }
}

// ========================================================================
// CFG_ID_ERROR (0x900C)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct SdmaCfgIdError;

impl RegisterMeta for SdmaCfgIdError {
    const DOMAIN: u32 = target_SDMA;
    const OFFSET: u32 = REG_SDMA_CFG_ID_ERROR;
}

impl Register<SdmaCfgIdError> {
    pub fn rd_resp_id(&mut self, rd_resp_id: Bits<5>) -> &mut Self {
        self.set_field(SDMA_CFG_ID_ERROR_RD_RESP_ID__MASK, unsafe {
            SDMA_CFG_ID_ERROR_RD_RESP_ID(rd_resp_id.val())
        })
    }

    pub fn wr_resp_id(&mut self, wr_resp_id: Bits<4>) -> &mut Self {
        self.set_field(SDMA_CFG_ID_ERROR_WR_RESP_ID__MASK, unsafe {
            SDMA_CFG_ID_ERROR_WR_RESP_ID(wr_resp_id.val())
        })
    }
}

// ========================================================================
// RD_WEIGHT_1 (0x9010)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct SdmaRdWeight1;

impl RegisterMeta for SdmaRdWeight1 {
    const DOMAIN: u32 = target_SDMA;
    const OFFSET: u32 = REG_SDMA_RD_WEIGHT_1;
}

impl Register<SdmaRdWeight1> {
    pub fn rd_weight_pc(&mut self, rd_weight_pc: Bits<8>) -> &mut Self {
        self.set_field(SDMA_RD_WEIGHT_1_RD_WEIGHT_PC__MASK, unsafe {
            SDMA_RD_WEIGHT_1_RD_WEIGHT_PC(rd_weight_pc.val())
        })
    }
}

// ========================================================================
// CFG_DMA_FIFO_CLR (0x9014)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct SdmaCfgDmaFifoClr;

impl RegisterMeta for SdmaCfgDmaFifoClr {
    const DOMAIN: u32 = target_SDMA;
    const OFFSET: u32 = REG_SDMA_CFG_DMA_FIFO_CLR;
}

impl Register<SdmaCfgDmaFifoClr> {
    pub fn dma_fifo_clr(&mut self, dma_fifo_clr: Bits<1>) -> &mut Self {
        self.set_field(SDMA_CFG_DMA_FIFO_CLR_DMA_FIFO_CLR__MASK, unsafe {
            SDMA_CFG_DMA_FIFO_CLR_DMA_FIFO_CLR(dma_fifo_clr.val())
        })
    }
}

// ========================================================================
// CFG_DMA_ARB (0x9018)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct SdmaCfgDmaArb;

impl RegisterMeta for SdmaCfgDmaArb {
    const DOMAIN: u32 = target_SDMA;
    const OFFSET: u32 = REG_SDMA_CFG_DMA_ARB;
}

impl Register<SdmaCfgDmaArb> {
    pub fn rd_fix_arb(&mut self, rd_fix_arb: Bits<3>) -> &mut Self {
        self.set_field(SDMA_CFG_DMA_ARB_RD_FIX_ARB__MASK, unsafe {
            SDMA_CFG_DMA_ARB_RD_FIX_ARB(rd_fix_arb.val())
        })
    }

    pub fn wr_fix_arb(&mut self, wr_fix_arb: Bits<3>) -> &mut Self {
        self.set_field(SDMA_CFG_DMA_ARB_WR_FIX_ARB__MASK, unsafe {
            SDMA_CFG_DMA_ARB_WR_FIX_ARB(wr_fix_arb.val())
        })
    }

    pub fn rd_arbit_model(&mut self, rd_arbit_model: Bits<1>) -> &mut Self {
        self.set_field(SDMA_CFG_DMA_ARB_RD_ARBIT_MODEL__MASK, unsafe {
            SDMA_CFG_DMA_ARB_RD_ARBIT_MODEL(rd_arbit_model.val())
        })
    }

    pub fn wr_arbit_model(&mut self, wr_arbit_model: Bits<1>) -> &mut Self {
        self.set_field(SDMA_CFG_DMA_ARB_WR_ARBIT_MODEL__MASK, unsafe {
            SDMA_CFG_DMA_ARB_WR_ARBIT_MODEL(wr_arbit_model.val())
        })
    }
}

// ========================================================================
// CFG_DMA_RD_QOS (0x9020)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct SdmaCfgDmaRdQos;

impl RegisterMeta for SdmaCfgDmaRdQos {
    const DOMAIN: u32 = target_SDMA;
    const OFFSET: u32 = REG_SDMA_CFG_DMA_RD_QOS;
}

impl Register<SdmaCfgDmaRdQos> {
    pub fn rd_feature_qos(&mut self, rd_feature_qos: Bits<2>) -> &mut Self {
        self.set_field(SDMA_CFG_DMA_RD_QOS_RD_FEATURE_QOS__MASK, unsafe {
            SDMA_CFG_DMA_RD_QOS_RD_FEATURE_QOS(rd_feature_qos.val())
        })
    }

    pub fn rd_kernel_qos(&mut self, rd_kernel_qos: Bits<2>) -> &mut Self {
        self.set_field(SDMA_CFG_DMA_RD_QOS_RD_KERNEL_QOS__MASK, unsafe {
            SDMA_CFG_DMA_RD_QOS_RD_KERNEL_QOS(rd_kernel_qos.val())
        })
    }

    pub fn rd_dpu_qos(&mut self, rd_dpu_qos: Bits<2>) -> &mut Self {
        self.set_field(SDMA_CFG_DMA_RD_QOS_RD_DPU_QOS__MASK, unsafe {
            SDMA_CFG_DMA_RD_QOS_RD_DPU_QOS(rd_dpu_qos.val())
        })
    }

    pub fn rd_ppu_qos(&mut self, rd_ppu_qos: Bits<2>) -> &mut Self {
        self.set_field(SDMA_CFG_DMA_RD_QOS_RD_PPU_QOS__MASK, unsafe {
            SDMA_CFG_DMA_RD_QOS_RD_PPU_QOS(rd_ppu_qos.val())
        })
    }

    pub fn rd_pc_qos(&mut self, rd_pc_qos: Bits<2>) -> &mut Self {
        self.set_field(SDMA_CFG_DMA_RD_QOS_RD_PC_QOS__MASK, unsafe {
            SDMA_CFG_DMA_RD_QOS_RD_PC_QOS(rd_pc_qos.val())
        })
    }
}

// ========================================================================
// CFG_DMA_RD_CFG (0x9024)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct SdmaCfgDmaRdCfg;

impl RegisterMeta for SdmaCfgDmaRdCfg {
    const DOMAIN: u32 = target_SDMA;
    const OFFSET: u32 = REG_SDMA_CFG_DMA_RD_CFG;
}

impl Register<SdmaCfgDmaRdCfg> {
    pub fn rd_arsize(&mut self, rd_arsize: Bits<3>) -> &mut Self {
        self.set_field(SDMA_CFG_DMA_RD_CFG_RD_ARSIZE__MASK, unsafe {
            SDMA_CFG_DMA_RD_CFG_RD_ARSIZE(rd_arsize.val())
        })
    }

    pub fn rd_arburst(&mut self, rd_arburst: Bits<2>) -> &mut Self {
        self.set_field(SDMA_CFG_DMA_RD_CFG_RD_ARBURST__MASK, unsafe {
            SDMA_CFG_DMA_RD_CFG_RD_ARBURST(rd_arburst.val())
        })
    }

    pub fn rd_arprot(&mut self, rd_arprot: Bits<3>) -> &mut Self {
        self.set_field(SDMA_CFG_DMA_RD_CFG_RD_ARPROT__MASK, unsafe {
            SDMA_CFG_DMA_RD_CFG_RD_ARPROT(rd_arprot.val())
        })
    }

    pub fn rd_arcache(&mut self, rd_arcache: Bits<4>) -> &mut Self {
        self.set_field(SDMA_CFG_DMA_RD_CFG_RD_ARCACHE__MASK, unsafe {
            SDMA_CFG_DMA_RD_CFG_RD_ARCACHE(rd_arcache.val())
        })
    }

    pub fn rd_arlock(&mut self, rd_arlock: Bits<1>) -> &mut Self {
        self.set_field(SDMA_CFG_DMA_RD_CFG_RD_ARLOCK__MASK, unsafe {
            SDMA_CFG_DMA_RD_CFG_RD_ARLOCK(rd_arlock.val())
        })
    }
}

// ========================================================================
// CFG_DMA_WR_CFG (0x9028)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct SdmaCfgDmaWrCfg;

impl RegisterMeta for SdmaCfgDmaWrCfg {
    const DOMAIN: u32 = target_SDMA;
    const OFFSET: u32 = REG_SDMA_CFG_DMA_WR_CFG;
}

impl Register<SdmaCfgDmaWrCfg> {
    pub fn wr_awsize(&mut self, wr_awsize: Bits<3>) -> &mut Self {
        self.set_field(SDMA_CFG_DMA_WR_CFG_WR_AWSIZE__MASK, unsafe {
            SDMA_CFG_DMA_WR_CFG_WR_AWSIZE(wr_awsize.val())
        })
    }

    pub fn wr_awburst(&mut self, wr_awburst: Bits<2>) -> &mut Self {
        self.set_field(SDMA_CFG_DMA_WR_CFG_WR_AWBURST__MASK, unsafe {
            SDMA_CFG_DMA_WR_CFG_WR_AWBURST(wr_awburst.val())
        })
    }

    pub fn wr_awprot(&mut self, wr_awprot: Bits<3>) -> &mut Self {
        self.set_field(SDMA_CFG_DMA_WR_CFG_WR_AWPROT__MASK, unsafe {
            SDMA_CFG_DMA_WR_CFG_WR_AWPROT(wr_awprot.val())
        })
    }

    pub fn wr_awcache(&mut self, wr_awcache: Bits<4>) -> &mut Self {
        self.set_field(SDMA_CFG_DMA_WR_CFG_WR_AWCACHE__MASK, unsafe {
            SDMA_CFG_DMA_WR_CFG_WR_AWCACHE(wr_awcache.val())
        })
    }

    pub fn wr_awlock(&mut self, wr_awlock: Bits<1>) -> &mut Self {
        self.set_field(SDMA_CFG_DMA_WR_CFG_WR_AWLOCK__MASK, unsafe {
            SDMA_CFG_DMA_WR_CFG_WR_AWLOCK(wr_awlock.val())
        })
    }
}

// ========================================================================
// CFG_DMA_WSTRB (0x902C)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct SdmaCfgDmaWstrb;

impl RegisterMeta for SdmaCfgDmaWstrb {
    const DOMAIN: u32 = target_SDMA;
    const OFFSET: u32 = REG_SDMA_CFG_DMA_WSTRB;
}

impl Register<SdmaCfgDmaWstrb> {
    pub fn wr_wstrb(&mut self, wr_wstrb: Bits<32>) -> &mut Self {
        self.set_field(SDMA_CFG_DMA_WSTRB_WR_WSTRB__MASK, unsafe {
            SDMA_CFG_DMA_WSTRB_WR_WSTRB(wr_wstrb.val())
        })
    }
}

// ========================================================================
// CFG_STATUS (0x9030)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct SdmaCfgStatus;

impl RegisterMeta for SdmaCfgStatus {
    const DOMAIN: u32 = target_SDMA;
    const OFFSET: u32 = REG_SDMA_CFG_STATUS;
}

impl Register<SdmaCfgStatus> {
    pub fn idel(&mut self, idel: Bits<1>) -> &mut Self {
        self.set_field(SDMA_CFG_STATUS_IDEL__MASK, unsafe {
            SDMA_CFG_STATUS_IDEL(idel.val())
        })
    }
}
