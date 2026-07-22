use crate::rocket::{
    builders::{Bits, Register, RegisterMeta},
    registers::*,
};

// target_SDMA (0x20000) doesn't fit in the 16-bit domain field the
// RegCmd packing scheme uses (bits 48-63) -- same problem as
// DOMAIN_GLOBAL, see builders.rs. Unfixed here since nothing currently
// uses this module; flagged so it doesn't look silently correct if that
// changes.

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
    /// Description: Maximum number of outstanding (in-flight) read transactions SDMA's AXI master may issue at once.
    ///
    /// Bit width: 8 (bits 7:0 of the register; reset 0x00)
    /// Range of values: 0-255, raw outstanding-count limit.
    /// Known limitations: None documented.
    /// Related registers: Paired with `wr_os_cnt` in the same register; analogous to DDMA's `cfg_outstanding.rd_os_cnt`.
    pub fn rd_os_cnt(&mut self, rd_os_cnt: Bits<8>) -> &mut Self {
        self.set_field(SDMA_CFG_OUTSTANDING_RD_OS_CNT__MASK, unsafe {
            SDMA_CFG_OUTSTANDING_RD_OS_CNT(rd_os_cnt.val())
        })
    }

    /// Description: Maximum number of outstanding (in-flight) write transactions SDMA's AXI master may issue at once.
    ///
    /// Bit width: 8 (bits 15:8 of the register; reset 0x00)
    /// Range of values: 0-255, raw outstanding-count limit.
    /// Known limitations: None documented.
    /// Related registers: Paired with `rd_os_cnt` in the same register; analogous to DDMA's `cfg_outstanding.wr_os_cnt`.
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
    /// Description: Arbiter weight given to the "feature" client's read bursts when SDMA's read arbitration is in weighted (non-fixed) mode.
    ///
    /// Bit width: 8 (bits 7:0 of the register; reset 0x00)
    /// Range of values: 0-255, raw weight value; higher weight wins more arbitration cycles relative to the other clients.
    /// Known limitations: None documented.
    /// Related registers: Sibling fields `rd_weight_kernel`/`rd_weight_dpu`/`rd_weight_pdp` in this register, `rd_weight_pc` in `RD_WEIGHT_1`; arbitration mode selected by `CFG_DMA_ARB.rd_arbit_model`.
    pub fn rd_weight_feature(&mut self, rd_weight_feature: Bits<8>) -> &mut Self {
        self.set_field(SDMA_RD_WEIGHT_0_RD_WEIGHT_FEATURE__MASK, unsafe {
            SDMA_RD_WEIGHT_0_RD_WEIGHT_FEATURE(rd_weight_feature.val())
        })
    }

    /// Description: Arbiter weight given to the "kernel" (weight-data) client's read bursts when SDMA's read arbitration is in weighted (non-fixed) mode.
    ///
    /// Bit width: 8 (bits 15:8 of the register; reset 0x00)
    /// Range of values: 0-255, raw weight value; higher weight wins more arbitration cycles relative to the other clients.
    /// Known limitations: None documented.
    /// Related registers: Sibling fields `rd_weight_feature`/`rd_weight_dpu`/`rd_weight_pdp` in this register, `rd_weight_pc` in `RD_WEIGHT_1`; arbitration mode selected by `CFG_DMA_ARB.rd_arbit_model`.
    pub fn rd_weight_kernel(&mut self, rd_weight_kernel: Bits<8>) -> &mut Self {
        self.set_field(SDMA_RD_WEIGHT_0_RD_WEIGHT_KERNEL__MASK, unsafe {
            SDMA_RD_WEIGHT_0_RD_WEIGHT_KERNEL(rd_weight_kernel.val())
        })
    }

    /// Description: Arbiter weight given to the DPU client's read bursts when SDMA's read arbitration is in weighted (non-fixed) mode.
    ///
    /// Bit width: 8 (bits 23:16 of the register; reset 0x00)
    /// Range of values: 0-255, raw weight value; higher weight wins more arbitration cycles relative to the other clients.
    /// Known limitations: None documented.
    /// Related registers: Sibling fields `rd_weight_feature`/`rd_weight_kernel`/`rd_weight_pdp` in this register, `rd_weight_pc` in `RD_WEIGHT_1`; arbitration mode selected by `CFG_DMA_ARB.rd_arbit_model`.
    pub fn rd_weight_dpu(&mut self, rd_weight_dpu: Bits<8>) -> &mut Self {
        self.set_field(SDMA_RD_WEIGHT_0_RD_WEIGHT_DPU__MASK, unsafe {
            SDMA_RD_WEIGHT_0_RD_WEIGHT_DPU(rd_weight_dpu.val())
        })
    }

    /// Description: Arbiter weight given to the PPU client's read bursts when SDMA's read arbitration is in weighted (non-fixed) mode.
    ///
    /// Bit width: 8 (bits 31:24 of the register; reset 0x00)
    /// Range of values: 0-255, raw weight value; higher weight wins more arbitration cycles relative to the other clients.
    /// Known limitations: TRM names the field `rd_weight_pdp` but its prose description reads "Weight of PPU read burst" — the field controls the PPU client despite the "pdp" spelling; carried through as-is from the TRM/generated bindings.
    /// Related registers: Sibling fields `rd_weight_feature`/`rd_weight_kernel`/`rd_weight_dpu` in this register, `rd_weight_pc` in `RD_WEIGHT_1`; arbitration mode selected by `CFG_DMA_ARB.rd_arbit_model`.
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
    /// Description: Arbiter weight given to the DPU client's write bursts when SDMA's write arbitration is in weighted (non-fixed) mode.
    ///
    /// Bit width: 8 (bits 7:0 of the register; reset 0x00)
    /// Range of values: 0-255, raw weight value; higher weight wins more arbitration cycles relative to the other clients.
    /// Known limitations: None documented.
    /// Related registers: Sibling field `wr_weight_pdp` in this register; arbitration mode selected by `CFG_DMA_ARB.wr_arbit_model`.
    pub fn wr_weight_dpu(&mut self, wr_weight_dpu: Bits<8>) -> &mut Self {
        self.set_field(SDMA_WR_WEIGHT_0_WR_WEIGHT_DPU__MASK, unsafe {
            SDMA_WR_WEIGHT_0_WR_WEIGHT_DPU(wr_weight_dpu.val())
        })
    }

    /// Description: Arbiter weight given to the PPU client's write bursts when SDMA's write arbitration is in weighted (non-fixed) mode.
    ///
    /// Bit width: 8 (bits 15:8 of the register; reset 0x00)
    /// Range of values: 0-255, raw weight value; higher weight wins more arbitration cycles relative to the other clients.
    /// Known limitations: TRM names the field `wr_weight_pdp` but its prose description reads "Write_weight_ppu" — the field controls the PPU client despite the "pdp" spelling; carried through as-is from the TRM/generated bindings.
    /// Related registers: Sibling field `wr_weight_dpu` in this register; arbitration mode selected by `CFG_DMA_ARB.wr_arbit_model`.
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
    /// Description: Captures the AXI ID of the most recent read transaction that received an error response.
    ///
    /// Bit width: 5 (bits 4:0 of the register; reset 0x00)
    /// Range of values: 0-31, raw AXI read-ID value.
    /// Known limitations: TRM marks this field RW rather than RO/W1C, but per the RKNN_TRM_Ch36.md architectural summary it is a status-capture field (the last error response's id); no explicit clear mechanism is documented for it.
    /// Related registers: `wr_resp_id` in the same register (write-side equivalent); PC's `pc_interrupt_status` bit 12 ("DMA read error") signals when this capture is meaningful.
    pub fn rd_resp_id(&mut self, rd_resp_id: Bits<5>) -> &mut Self {
        self.set_field(SDMA_CFG_ID_ERROR_RD_RESP_ID__MASK, unsafe {
            SDMA_CFG_ID_ERROR_RD_RESP_ID(rd_resp_id.val())
        })
    }

    /// Description: Captures the AXI ID of the most recent write transaction that received an error response.
    ///
    /// Bit width: 4 (bits 9:6 of the register; reset 0x0)
    /// Range of values: 0-15, raw AXI write-ID value.
    /// Known limitations: TRM marks this field RW rather than RO/W1C, but per the RKNN_TRM_Ch36.md architectural summary it is a status-capture field (the last error response's id); no explicit clear mechanism is documented for it.
    /// Related registers: `rd_resp_id` in the same register (read-side equivalent); PC's `pc_interrupt_status` bit 13 ("DMA write error") signals when this capture is meaningful.
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
    /// Description: Arbiter weight given to the PC (task-sequencer) client's read bursts when SDMA's read arbitration is in weighted (non-fixed) mode.
    ///
    /// Bit width: 8 (bits 7:0 of the register; reset 0x00)
    /// Range of values: 0-255, raw weight value; higher weight wins more arbitration cycles relative to the other clients.
    /// Known limitations: None documented.
    /// Related registers: `rd_weight_feature`/`rd_weight_kernel`/`rd_weight_dpu`/`rd_weight_pdp` in `RD_WEIGHT_0`; arbitration mode selected by `CFG_DMA_ARB.rd_arbit_model`.
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
    /// Description: Clears SDMA's internal DMA FIFO.
    ///
    /// Bit width: 1 (bit 0 of the register; reset 0x0)
    /// Range of values: 0 = no effect / normal operation, 1 = clear the FIFO.
    /// Known limitations: TRM lists this as plain RW rather than a self-clearing pulse (W1C/W1S); whether hardware auto-clears the bit after the clear takes effect or software must write it back to 0 is not stated.
    /// Related registers: None.
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
    /// Description: Selects which client wins SDMA's read arbiter when it is running in fixed-priority mode.
    ///
    /// Bit width: 3 (bits 2:0 of the register; reset 0x0)
    /// Range of values: 0-7, raw client-select index (exact per-client encoding not enumerated in the TRM's raw text — see the per-client weight registers `rd_weight_0`/`rd_weight_1` for the named clients this arbiter chooses among: feature/kernel/dpu/ppu/pc).
    /// Known limitations: Only meaningful when `rd_arbit_model` selects fixed-priority arbitration; ignored under round-robin/weighted arbitration.
    /// Related registers: `rd_arbit_model` (mode select) in this register; `rd_weight_feature`/`rd_weight_kernel`/`rd_weight_dpu`/`rd_weight_pdp` (`RD_WEIGHT_0`) and `rd_weight_pc` (`RD_WEIGHT_1`).
    pub fn rd_fix_arb(&mut self, rd_fix_arb: Bits<3>) -> &mut Self {
        self.set_field(SDMA_CFG_DMA_ARB_RD_FIX_ARB__MASK, unsafe {
            SDMA_CFG_DMA_ARB_RD_FIX_ARB(rd_fix_arb.val())
        })
    }

    /// Description: Selects which client wins SDMA's write arbiter when it is running in fixed-priority mode.
    ///
    /// Bit width: 3 (bits 6:4 of the register; reset 0x0)
    /// Range of values: 0-7, raw client-select index (exact per-client encoding not enumerated in the TRM's raw text — see `wr_weight_0` for the named write clients: dpu/ppu).
    /// Known limitations: Only meaningful when `wr_arbit_model` selects fixed-priority arbitration; ignored under round-robin/weighted arbitration.
    /// Related registers: `wr_arbit_model` (mode select) in this register; `wr_weight_dpu`/`wr_weight_pdp` (`WR_WEIGHT_0`).
    pub fn wr_fix_arb(&mut self, wr_fix_arb: Bits<3>) -> &mut Self {
        self.set_field(SDMA_CFG_DMA_ARB_WR_FIX_ARB__MASK, unsafe {
            SDMA_CFG_DMA_ARB_WR_FIX_ARB(wr_fix_arb.val())
        })
    }

    /// Description: Selects SDMA's read-channel arbitration scheme.
    ///
    /// Bit width: 1 (bit 8 of the register; reset 0x0)
    /// Range of values: Per RKNN_TRM_Ch36.md's architectural summary this is a fixed-vs-round-robin mode select; the raw TRM text gives only the field name ("Read_arbit_model") without enumerating which of 0/1 is which mode.
    /// Known limitations: Exact 0/1-to-mode mapping not stated in either source; when in fixed mode, `rd_fix_arb` selects the winning client, otherwise the per-client `rd_weight_*` registers apply.
    /// Related registers: `rd_fix_arb` in this register; `rd_weight_feature`/`rd_weight_kernel`/`rd_weight_dpu`/`rd_weight_pdp`/`rd_weight_pc`.
    pub fn rd_arbit_model(&mut self, rd_arbit_model: Bits<1>) -> &mut Self {
        self.set_field(SDMA_CFG_DMA_ARB_RD_ARBIT_MODEL__MASK, unsafe {
            SDMA_CFG_DMA_ARB_RD_ARBIT_MODEL(rd_arbit_model.val())
        })
    }

    /// Description: Selects SDMA's write-channel arbitration scheme.
    ///
    /// Bit width: 1 (bit 9 of the register; reset 0x0)
    /// Range of values: Per RKNN_TRM_Ch36.md's architectural summary this is a fixed-vs-round-robin mode select; the raw TRM text gives only the field name ("Write_arbit_model") without enumerating which of 0/1 is which mode.
    /// Known limitations: Exact 0/1-to-mode mapping not stated in either source; when in fixed mode, `wr_fix_arb` selects the winning client, otherwise the per-client `wr_weight_*` registers apply.
    /// Related registers: `wr_fix_arb` in this register; `wr_weight_dpu`/`wr_weight_pdp`.
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
    /// Description: AXI QoS level SDMA attaches to the "feature" client's read transactions (drives the ARQOS signal).
    ///
    /// Bit width: 2 (bits 1:0 of the register; reset 0x0)
    /// Range of values: 0-3, raw AXI QoS level; higher value requests higher priority/quality-of-service from the downstream interconnect.
    /// Known limitations: None documented.
    /// Related registers: Sibling QoS fields `rd_kernel_qos`/`rd_dpu_qos`/`rd_ppu_qos`/`rd_pc_qos` in this register; distinct from the `rd_weight_*` arbiter-weight registers, which affect SDMA's internal arbitration rather than the downstream AXI fabric's QoS handling.
    pub fn rd_feature_qos(&mut self, rd_feature_qos: Bits<2>) -> &mut Self {
        self.set_field(SDMA_CFG_DMA_RD_QOS_RD_FEATURE_QOS__MASK, unsafe {
            SDMA_CFG_DMA_RD_QOS_RD_FEATURE_QOS(rd_feature_qos.val())
        })
    }

    /// Description: AXI QoS level SDMA attaches to the "kernel" (weight-data) client's read transactions (drives the ARQOS signal).
    ///
    /// Bit width: 2 (bits 3:2 of the register; reset 0x0)
    /// Range of values: 0-3, raw AXI QoS level; higher value requests higher priority/quality-of-service from the downstream interconnect.
    /// Known limitations: None documented.
    /// Related registers: Sibling QoS fields `rd_feature_qos`/`rd_dpu_qos`/`rd_ppu_qos`/`rd_pc_qos` in this register.
    pub fn rd_kernel_qos(&mut self, rd_kernel_qos: Bits<2>) -> &mut Self {
        self.set_field(SDMA_CFG_DMA_RD_QOS_RD_KERNEL_QOS__MASK, unsafe {
            SDMA_CFG_DMA_RD_QOS_RD_KERNEL_QOS(rd_kernel_qos.val())
        })
    }

    /// Description: AXI QoS level SDMA attaches to the DPU client's read transactions (drives the ARQOS signal).
    ///
    /// Bit width: 2 (bits 5:4 of the register; reset 0x0)
    /// Range of values: 0-3, raw AXI QoS level; higher value requests higher priority/quality-of-service from the downstream interconnect.
    /// Known limitations: None documented.
    /// Related registers: Sibling QoS fields `rd_feature_qos`/`rd_kernel_qos`/`rd_ppu_qos`/`rd_pc_qos` in this register.
    pub fn rd_dpu_qos(&mut self, rd_dpu_qos: Bits<2>) -> &mut Self {
        self.set_field(SDMA_CFG_DMA_RD_QOS_RD_DPU_QOS__MASK, unsafe {
            SDMA_CFG_DMA_RD_QOS_RD_DPU_QOS(rd_dpu_qos.val())
        })
    }

    /// Description: AXI QoS level SDMA attaches to the PPU client's read transactions (drives the ARQOS signal).
    ///
    /// Bit width: 2 (bits 7:6 of the register; reset 0x0)
    /// Range of values: 0-3, raw AXI QoS level; higher value requests higher priority/quality-of-service from the downstream interconnect.
    /// Known limitations: None documented.
    /// Related registers: Sibling QoS fields `rd_feature_qos`/`rd_kernel_qos`/`rd_dpu_qos`/`rd_pc_qos` in this register.
    pub fn rd_ppu_qos(&mut self, rd_ppu_qos: Bits<2>) -> &mut Self {
        self.set_field(SDMA_CFG_DMA_RD_QOS_RD_PPU_QOS__MASK, unsafe {
            SDMA_CFG_DMA_RD_QOS_RD_PPU_QOS(rd_ppu_qos.val())
        })
    }

    /// Description: AXI QoS level SDMA attaches to the PC (task-sequencer) client's read transactions (drives the ARQOS signal).
    ///
    /// Bit width: 2 (bits 9:8 of the register; reset 0x0)
    /// Range of values: 0-3, raw AXI QoS level; higher value requests higher priority/quality-of-service from the downstream interconnect.
    /// Known limitations: None documented.
    /// Related registers: Sibling QoS fields `rd_feature_qos`/`rd_kernel_qos`/`rd_dpu_qos`/`rd_ppu_qos` in this register.
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
    /// Description: Raw AXI ARSIZE value SDMA drives on its read channel — the number of bytes transferred per beat.
    ///
    /// Bit width: 3 (bits 2:0 of the register; reset 0x0)
    /// Range of values: Standard AXI ARSIZE encoding: 0=1 byte, 1=2, 2=4, 3=8, 4=16, 5=32, 6=64, 7=128 bytes/beat.
    /// Known limitations: None documented.
    /// Related registers: Sibling passthrough fields `rd_arburst`/`rd_arprot`/`rd_arcache`/`rd_arlock` in this register; write-side equivalent `wr_awsize` in `CFG_DMA_WR_CFG`.
    pub fn rd_arsize(&mut self, rd_arsize: Bits<3>) -> &mut Self {
        self.set_field(SDMA_CFG_DMA_RD_CFG_RD_ARSIZE__MASK, unsafe {
            SDMA_CFG_DMA_RD_CFG_RD_ARSIZE(rd_arsize.val())
        })
    }

    /// Description: Raw AXI ARBURST value SDMA drives on its read channel — the burst addressing type.
    ///
    /// Bit width: 2 (bits 4:3 of the register; reset 0x0)
    /// Range of values: Standard AXI ARBURST encoding: 0=FIXED, 1=INCR, 2=WRAP, 3=reserved.
    /// Known limitations: None documented.
    /// Related registers: Sibling passthrough fields `rd_arsize`/`rd_arprot`/`rd_arcache`/`rd_arlock` in this register; write-side equivalent `wr_awburst` in `CFG_DMA_WR_CFG`.
    pub fn rd_arburst(&mut self, rd_arburst: Bits<2>) -> &mut Self {
        self.set_field(SDMA_CFG_DMA_RD_CFG_RD_ARBURST__MASK, unsafe {
            SDMA_CFG_DMA_RD_CFG_RD_ARBURST(rd_arburst.val())
        })
    }

    /// Description: Raw AXI ARPROT value SDMA drives on its read channel — the protection/access-type attributes.
    ///
    /// Bit width: 3 (bits 7:5 of the register; reset 0x0)
    /// Range of values: Standard AXI ARPROT encoding: bit0 unprivileged(0)/privileged(1), bit1 secure(0)/non-secure(1), bit2 data(0)/instruction(1) access.
    /// Known limitations: None documented.
    /// Related registers: Sibling passthrough fields `rd_arsize`/`rd_arburst`/`rd_arcache`/`rd_arlock` in this register; write-side equivalent `wr_awprot` in `CFG_DMA_WR_CFG`.
    pub fn rd_arprot(&mut self, rd_arprot: Bits<3>) -> &mut Self {
        self.set_field(SDMA_CFG_DMA_RD_CFG_RD_ARPROT__MASK, unsafe {
            SDMA_CFG_DMA_RD_CFG_RD_ARPROT(rd_arprot.val())
        })
    }

    /// Description: Raw AXI ARCACHE value SDMA drives on its read channel — the bufferable/cacheable/allocate memory-type attributes.
    ///
    /// Bit width: 4 (bits 11:8 of the register; reset 0x0)
    /// Range of values: Standard AXI ARCACHE 4-bit encoding (bufferable, modifiable, allocate hints); 0x0-0xF.
    /// Known limitations: None documented.
    /// Related registers: Sibling passthrough fields `rd_arsize`/`rd_arburst`/`rd_arprot`/`rd_arlock` in this register; write-side equivalent `wr_awcache` in `CFG_DMA_WR_CFG`.
    pub fn rd_arcache(&mut self, rd_arcache: Bits<4>) -> &mut Self {
        self.set_field(SDMA_CFG_DMA_RD_CFG_RD_ARCACHE__MASK, unsafe {
            SDMA_CFG_DMA_RD_CFG_RD_ARCACHE(rd_arcache.val())
        })
    }

    /// Description: Raw AXI ARLOCK value SDMA drives on its read channel — requests an exclusive/locked access.
    ///
    /// Bit width: 1 (bit 12 of the register; reset 0x0)
    /// Range of values: 0 = normal access, 1 = exclusive (locked) access, per the AXI4 single-bit ARLOCK encoding.
    /// Known limitations: None documented.
    /// Related registers: Sibling passthrough fields `rd_arsize`/`rd_arburst`/`rd_arprot`/`rd_arcache` in this register; write-side equivalent `wr_awlock` in `CFG_DMA_WR_CFG`.
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
    /// Description: Raw AXI AWSIZE value SDMA drives on its write channel — the number of bytes transferred per beat.
    ///
    /// Bit width: 3 (bits 2:0 of the register; reset 0x0)
    /// Range of values: Standard AXI AWSIZE encoding: 0=1 byte, 1=2, 2=4, 3=8, 4=16, 5=32, 6=64, 7=128 bytes/beat.
    /// Known limitations: None documented.
    /// Related registers: Sibling passthrough fields `wr_awburst`/`wr_awprot`/`wr_awcache`/`wr_awlock` in this register; read-side equivalent `rd_arsize` in `CFG_DMA_RD_CFG`; `cfg_dma_wstrb` provides the per-byte write-strobe mask for these writes.
    pub fn wr_awsize(&mut self, wr_awsize: Bits<3>) -> &mut Self {
        self.set_field(SDMA_CFG_DMA_WR_CFG_WR_AWSIZE__MASK, unsafe {
            SDMA_CFG_DMA_WR_CFG_WR_AWSIZE(wr_awsize.val())
        })
    }

    /// Description: Raw AXI AWBURST value SDMA drives on its write channel — the burst addressing type.
    ///
    /// Bit width: 2 (bits 4:3 of the register; reset 0x0)
    /// Range of values: Standard AXI AWBURST encoding: 0=FIXED, 1=INCR, 2=WRAP, 3=reserved.
    /// Known limitations: None documented.
    /// Related registers: Sibling passthrough fields `wr_awsize`/`wr_awprot`/`wr_awcache`/`wr_awlock` in this register; read-side equivalent `rd_arburst` in `CFG_DMA_RD_CFG`.
    pub fn wr_awburst(&mut self, wr_awburst: Bits<2>) -> &mut Self {
        self.set_field(SDMA_CFG_DMA_WR_CFG_WR_AWBURST__MASK, unsafe {
            SDMA_CFG_DMA_WR_CFG_WR_AWBURST(wr_awburst.val())
        })
    }

    /// Description: Raw AXI AWPROT value SDMA drives on its write channel — the protection/access-type attributes.
    ///
    /// Bit width: 3 (bits 7:5 of the register; reset 0x0)
    /// Range of values: Standard AXI AWPROT encoding: bit0 unprivileged(0)/privileged(1), bit1 secure(0)/non-secure(1), bit2 data(0)/instruction(1) access.
    /// Known limitations: None documented.
    /// Related registers: Sibling passthrough fields `wr_awsize`/`wr_awburst`/`wr_awcache`/`wr_awlock` in this register; read-side equivalent `rd_arprot` in `CFG_DMA_RD_CFG`.
    pub fn wr_awprot(&mut self, wr_awprot: Bits<3>) -> &mut Self {
        self.set_field(SDMA_CFG_DMA_WR_CFG_WR_AWPROT__MASK, unsafe {
            SDMA_CFG_DMA_WR_CFG_WR_AWPROT(wr_awprot.val())
        })
    }

    /// Description: Raw AXI AWCACHE value SDMA drives on its write channel — the bufferable/cacheable/allocate memory-type attributes.
    ///
    /// Bit width: 4 (bits 11:8 of the register; reset 0x0)
    /// Range of values: Standard AXI AWCACHE 4-bit encoding (bufferable, modifiable, allocate hints); 0x0-0xF.
    /// Known limitations: None documented.
    /// Related registers: Sibling passthrough fields `wr_awsize`/`wr_awburst`/`wr_awprot`/`wr_awlock` in this register; read-side equivalent `rd_arcache` in `CFG_DMA_RD_CFG`.
    pub fn wr_awcache(&mut self, wr_awcache: Bits<4>) -> &mut Self {
        self.set_field(SDMA_CFG_DMA_WR_CFG_WR_AWCACHE__MASK, unsafe {
            SDMA_CFG_DMA_WR_CFG_WR_AWCACHE(wr_awcache.val())
        })
    }

    /// Description: Raw AXI AWLOCK value SDMA drives on its write channel — requests an exclusive/locked access.
    ///
    /// Bit width: 1 (bit 12 of the register; reset 0x0)
    /// Range of values: 0 = normal access, 1 = exclusive (locked) access, per the AXI4 single-bit AWLOCK encoding.
    /// Known limitations: None documented.
    /// Related registers: Sibling passthrough fields `wr_awsize`/`wr_awburst`/`wr_awprot`/`wr_awcache` in this register; read-side equivalent `rd_arlock` in `CFG_DMA_RD_CFG`.
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
    /// Description: Raw AXI WSTRB value SDMA drives on its write channel — per-byte write-strobe mask for the data bus.
    ///
    /// Bit width: 32 (bits 31:0 of the register; reset 0x00000000)
    /// Range of values: One bit per byte lane; 1 = that byte is written, 0 = that byte lane is masked off, per standard AXI WSTRB semantics.
    /// Known limitations: None documented.
    /// Related registers: Used together with `wr_awsize`/`wr_awburst`/`wr_awprot`/`wr_awcache`/`wr_awlock` (`CFG_DMA_WR_CFG`) to fully specify SDMA's outgoing write transactions.
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
    /// Description: Reports whether SDMA's DMA engine is idle (no outstanding read/write activity).
    ///
    /// Bit width: 1 (bit 8 of the register; reset 0x0)
    /// Range of values: 0 = busy, 1 = idle (per the field name; TRM's raw text spells this "Idel" and lists no explicit enum, only the bit position and RW attribute).
    /// Known limitations: TRM marks this field RW even though it is architecturally a status/read bit (RKNN_TRM_Ch36.md describes `cfg_status` as an idle-bit status register); no software-write semantics are documented for it.
    /// Related registers: `cfg_dma_fifo_clr.dma_fifo_clr` (should typically be issued only while idle); PC's task-completion tracking (`pc_task_status`) for the overall pipeline.
    pub fn idel(&mut self, idel: Bits<1>) -> &mut Self {
        self.set_field(SDMA_CFG_STATUS_IDEL__MASK, unsafe {
            SDMA_CFG_STATUS_IDEL(idel.val())
        })
    }
}
