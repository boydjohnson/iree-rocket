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
    /// Description: Maximum number of outstanding (in-flight, not yet responded) AXI read
    /// transactions DDMA is allowed to issue.
    ///
    /// Bit width: 8 (TRM bits 7:0)
    /// Range of values: 0x00-0xFF, a raw outstanding-transaction count. Reset value 0x00.
    /// Known limitations: TRM gives no guidance on a safe/effective value; interacts with
    /// downstream AXI interconnect and DRAM controller outstanding-transaction limits, which
    /// are not documented in this chapter.
    /// Related registers: wr_os_cnt (same register, write-side counterpart).
    pub fn rd_os_cnt(&mut self, rd_os_cnt: Bits<8>) -> &mut Self {
        self.set_field(DDMA_CFG_OUTSTANDING_RD_OS_CNT__MASK, unsafe {
            DDMA_CFG_OUTSTANDING_RD_OS_CNT(rd_os_cnt.val())
        })
    }

    /// Description: Maximum number of outstanding (in-flight, not yet responded) AXI write
    /// transactions DDMA is allowed to issue.
    ///
    /// Bit width: 8 (TRM bits 15:8)
    /// Range of values: 0x00-0xFF, a raw outstanding-transaction count. Reset value 0x00.
    /// Known limitations: TRM gives no guidance on a safe/effective value; interacts with
    /// downstream AXI interconnect and DRAM controller outstanding-transaction limits, which
    /// are not documented in this chapter.
    /// Related registers: rd_os_cnt (same register, read-side counterpart).
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
    /// Description: Per-client AXI arbiter weight for the "feature" client's DDMA read
    /// requests (relative priority when multiple clients contend for DDMA read bandwidth).
    ///
    /// Bit width: 8 (TRM bits 7:0)
    /// Range of values: 0x00-0xFF, a raw arbitration weight (larger = more bandwidth share).
    /// Reset value 0x00.
    /// Known limitations: TRM does not document the arbitration algorithm's exact weight
    /// semantics (e.g. whether weight is a credit count, a round-robin slot count, or a
    /// priority level) or interaction with `cfg_dma_arb`'s fixed-vs-round-robin mode select.
    /// Related registers: rd_weight_kernel, rd_weight_dpu, rd_weight_pdp (same register, other
    /// clients); rd_weight_pc (DdmaRdWeight1); wr_weight_dpu/wr_weight_pdp (DdmaWrWeight0,
    /// write-side counterparts); cfg_dma_arb (arbitration mode select).
    pub fn rd_weight_feature(&mut self, rd_weight_feature: Bits<8>) -> &mut Self {
        self.set_field(DDMA_RD_WEIGHT_0_RD_WEIGHT_FEATURE__MASK, unsafe {
            DDMA_RD_WEIGHT_0_RD_WEIGHT_FEATURE(rd_weight_feature.val())
        })
    }

    /// Description: Per-client AXI arbiter weight for the "kernel" (weight-data) client's
    /// DDMA read requests (relative priority when multiple clients contend for DDMA read
    /// bandwidth).
    ///
    /// Bit width: 8 (TRM bits 15:8)
    /// Range of values: 0x00-0xFF, a raw arbitration weight (larger = more bandwidth share).
    /// Reset value 0x00.
    /// Known limitations: TRM's own prose for this field reads "Weight of read weight burst"
    /// (i.e. weight-data burst) despite the field being named `rd_weight_kernel` — "kernel"
    /// here refers to the convolution weight/kernel data client, not a CPU/OS kernel. Exact
    /// arbitration algorithm semantics undocumented (see rd_weight_feature).
    /// Related registers: rd_weight_feature, rd_weight_dpu, rd_weight_pdp (same register,
    /// other clients); cfg_dma_arb (arbitration mode select).
    pub fn rd_weight_kernel(&mut self, rd_weight_kernel: Bits<8>) -> &mut Self {
        self.set_field(DDMA_RD_WEIGHT_0_RD_WEIGHT_KERNEL__MASK, unsafe {
            DDMA_RD_WEIGHT_0_RD_WEIGHT_KERNEL(rd_weight_kernel.val())
        })
    }

    /// Description: Per-client AXI arbiter weight for the DPU client's DDMA read requests
    /// (relative priority when multiple clients contend for DDMA read bandwidth).
    ///
    /// Bit width: 8 (TRM bits 23:16)
    /// Range of values: 0x00-0xFF, a raw arbitration weight (larger = more bandwidth share).
    /// Reset value 0x00.
    /// Known limitations: Exact arbitration algorithm semantics undocumented (see
    /// rd_weight_feature).
    /// Related registers: rd_weight_feature, rd_weight_kernel, rd_weight_pdp (same register,
    /// other clients); wr_weight_dpu (DdmaWrWeight0, write-side counterpart); cfg_dma_arb
    /// (arbitration mode select).
    pub fn rd_weight_dpu(&mut self, rd_weight_dpu: Bits<8>) -> &mut Self {
        self.set_field(DDMA_RD_WEIGHT_0_RD_WEIGHT_DPU__MASK, unsafe {
            DDMA_RD_WEIGHT_0_RD_WEIGHT_DPU(rd_weight_dpu.val())
        })
    }

    /// Description: Per-client AXI arbiter weight for the PPU client's DDMA read requests
    /// (relative priority when multiple clients contend for DDMA read bandwidth).
    ///
    /// Bit width: 8 (TRM bits 31:24)
    /// Range of values: 0x00-0xFF, a raw arbitration weight (larger = more bandwidth share).
    /// Reset value 0x00.
    /// Known limitations: TRM names both the register field and its prose description
    /// "rd_weight_pdp" / "Weight of PPU read burst" — "pdp" appears to be the TRM's own
    /// abbreviation for the PPU client (possibly a typo for "ppu"), not a distinct block.
    /// Exact arbitration algorithm semantics otherwise undocumented (see rd_weight_feature).
    /// Related registers: rd_weight_feature, rd_weight_kernel, rd_weight_dpu (same register,
    /// other clients); wr_weight_pdp (DdmaWrWeight0, write-side counterpart); rd_ppu_qos
    /// (DdmaCfgDmaRdQos); cfg_dma_arb (arbitration mode select).
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
    /// Description: Per-client AXI arbiter weight for the DPU client's DDMA write requests
    /// (relative priority when multiple clients contend for DDMA write bandwidth).
    ///
    /// Bit width: 8 (TRM bits 7:0)
    /// Range of values: 0x00-0xFF, a raw arbitration weight (larger = more bandwidth share).
    /// Reset value 0x00.
    /// Known limitations: TRM prose ("Write_weight_dpu") gives no further detail on the
    /// arbitration algorithm; same undocumented weight semantics as the read-side weights.
    /// Related registers: wr_weight_pdp (same register, other client); rd_weight_dpu
    /// (DdmaRdWeight0, read-side counterpart); cfg_dma_arb (arbitration mode select).
    pub fn wr_weight_dpu(&mut self, wr_weight_dpu: Bits<8>) -> &mut Self {
        self.set_field(DDMA_WR_WEIGHT_0_WR_WEIGHT_DPU__MASK, unsafe {
            DDMA_WR_WEIGHT_0_WR_WEIGHT_DPU(wr_weight_dpu.val())
        })
    }

    /// Description: Per-client AXI arbiter weight for the PPU client's DDMA write requests
    /// (relative priority when multiple clients contend for DDMA write bandwidth).
    ///
    /// Bit width: 8 (TRM bits 15:8)
    /// Range of values: 0x00-0xFF, a raw arbitration weight (larger = more bandwidth share).
    /// Reset value 0x00.
    /// Known limitations: TRM prose ("Write_weight_ppu") uses "ppu" here despite the field
    /// being named `wr_weight_pdp`, mirroring the same pdp/ppu naming inconsistency seen in
    /// DdmaRdWeight0::rd_weight_pdp. Arbitration algorithm semantics otherwise undocumented.
    /// Related registers: wr_weight_dpu (same register, other client); rd_weight_pdp
    /// (DdmaRdWeight0, read-side counterpart); cfg_dma_arb (arbitration mode select).
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
    /// Description: Captures the AXI ID of the transaction that most recently received an
    /// erroneous read response, for error diagnosis.
    ///
    /// Bit width: 5 (TRM bits 4:0)
    /// Range of values: 0x00-0x1F, a raw AXI ID value. Reset value 0x00.
    /// Known limitations: TRM lists this field as RW (not RO), which is unusual for a
    /// status/capture field — no documented mechanism for how/when hardware latches a new
    /// value into it, nor whether software writes to it have any effect versus being purely
    /// informational; not confirmed against hardware behavior in this crate.
    /// Related registers: wr_resp_id (same register, write-side counterpart); cfg_status
    /// (idle bit, no direct documented link to error capture).
    pub fn rd_resp_id(&mut self, rd_resp_id: Bits<5>) -> &mut Self {
        self.set_field(DDMA_CFG_ID_ERROR_RD_RESP_ID__MASK, unsafe {
            DDMA_CFG_ID_ERROR_RD_RESP_ID(rd_resp_id.val())
        })
    }

    /// Description: Captures the AXI ID of the transaction that most recently received an
    /// erroneous write response, for error diagnosis.
    ///
    /// Bit width: 4 (TRM bits 9:6)
    /// Range of values: 0x0-0xF, a raw AXI ID value. Reset value 0x0.
    /// Known limitations: TRM lists this field as RW (not RO), same caveat as rd_resp_id
    /// regarding capture semantics being undocumented.
    /// Related registers: rd_resp_id (same register, read-side counterpart); cfg_status
    /// (idle bit, no direct documented link to error capture).
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
    /// Description: Per-client AXI arbiter weight for the PC (task sequencer / regcmd fetch)
    /// client's DDMA read requests (relative priority when multiple clients contend for DDMA
    /// read bandwidth).
    ///
    /// Bit width: 8 (TRM bits 7:0)
    /// Range of values: 0x00-0xFF, a raw arbitration weight (larger = more bandwidth share).
    /// Reset value 0x00.
    /// Known limitations: This is the sole documented field of a register that otherwise
    /// spans a 32-bit word (bits 31:8 reserved/RO); exact arbitration algorithm semantics
    /// undocumented (see DdmaRdWeight0::rd_weight_feature).
    /// Related registers: rd_weight_feature/rd_weight_kernel/rd_weight_dpu/rd_weight_pdp
    /// (DdmaRdWeight0, other clients' read weights); rd_pc_qos (DdmaCfgDmaRdQos); cfg_dma_arb
    /// (arbitration mode select).
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
    /// Description: Clears DDMA's internal data FIFO, for recovering from an error/stall
    /// condition without a full block reset.
    ///
    /// Bit width: 1 (TRM bit 0)
    /// Range of values: 0 = no effect, 1 = clear DMA FIFO. Reset value 0.
    /// Known limitations: TRM lists the attribute as RW rather than W1C or a self-clearing
    /// strobe, so it is unconfirmed whether hardware auto-clears this bit after the FIFO
    /// clear completes or whether software must explicitly write it back to 0; not yet
    /// validated against real hardware in this crate.
    /// Related registers: cfg_status (idel — DDMA should be idle before clearing its FIFO).
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
    /// Description: Selects which read client is always granted priority when
    /// `rd_arbit_model` selects fixed-priority (as opposed to round-robin/weighted)
    /// arbitration among DDMA's read clients.
    ///
    /// Bit width: 3 (TRM bits 2:0)
    /// Range of values: 0-7, a client index. TRM does not enumerate which numeric value maps
    /// to which named client (pc/ppu/dpu/kernel/feature); condensed notes (RKNN_TRM_Ch36.md
    /// §4.8) only describe the field's purpose, not its encoding.
    /// Known limitations: Meaning is only consulted when rd_arbit_model selects fixed mode;
    /// otherwise the rd_weight_* registers govern arbitration instead.
    /// Related registers: rd_arbit_model (mode select this field depends on); rd_weight_pc/
    /// rd_weight_feature/rd_weight_kernel/rd_weight_dpu/rd_weight_pdp (weighted-mode
    /// alternative).
    pub fn rd_fix_arb(&mut self, rd_fix_arb: Bits<3>) -> &mut Self {
        self.set_field(DDMA_CFG_DMA_ARB_RD_FIX_ARB__MASK, unsafe {
            DDMA_CFG_DMA_ARB_RD_FIX_ARB(rd_fix_arb.val())
        })
    }

    /// Description: Selects which write client is always granted priority when
    /// `wr_arbit_model` selects fixed-priority (as opposed to round-robin/weighted)
    /// arbitration among DDMA's write clients.
    ///
    /// Bit width: 3 (TRM bits 6:4)
    /// Range of values: 0-7, a client index. TRM does not enumerate which numeric value maps
    /// to which named client.
    /// Known limitations: Meaning is only consulted when wr_arbit_model selects fixed mode;
    /// otherwise the wr_weight_* registers govern arbitration instead.
    /// Related registers: wr_arbit_model (mode select this field depends on); wr_weight_dpu/
    /// wr_weight_pdp (weighted-mode alternative).
    pub fn wr_fix_arb(&mut self, wr_fix_arb: Bits<3>) -> &mut Self {
        self.set_field(DDMA_CFG_DMA_ARB_WR_FIX_ARB__MASK, unsafe {
            DDMA_CFG_DMA_ARB_WR_FIX_ARB(wr_fix_arb.val())
        })
    }

    /// Description: Selects DDMA's read-client arbitration model — fixed priority vs.
    /// weighted/round-robin.
    ///
    /// Bit width: 1 (TRM bit 8)
    /// Range of values: TRM prose ("Read_arbit_model") does not spell out which of 0/1 is
    /// fixed vs. round-robin; RKNN_TRM_Ch36.md §4.8 confirms the field selects between "fixed
    /// vs round-robin arbitration mode, per-direction" without giving the bit encoding.
    /// Known limitations: Encoding (0 vs 1 = fixed) not documented in the source material.
    /// Related registers: rd_fix_arb (used when this selects fixed mode); rd_weight_pc/
    /// rd_weight_feature/rd_weight_kernel/rd_weight_dpu/rd_weight_pdp (used when this
    /// selects weighted/round-robin mode).
    pub fn rd_arbit_model(&mut self, rd_arbit_model: Bits<1>) -> &mut Self {
        self.set_field(DDMA_CFG_DMA_ARB_RD_ARBIT_MODEL__MASK, unsafe {
            DDMA_CFG_DMA_ARB_RD_ARBIT_MODEL(rd_arbit_model.val())
        })
    }

    /// Description: Selects DDMA's write-client arbitration model — fixed priority vs.
    /// weighted/round-robin.
    ///
    /// Bit width: 1 (TRM bit 9)
    /// Range of values: TRM prose ("Write_arbit_model") does not spell out which of 0/1 is
    /// fixed vs. round-robin; see rd_arbit_model for the analogous read-side field.
    /// Known limitations: Encoding (0 vs 1 = fixed) not documented in the source material.
    /// Related registers: wr_fix_arb (used when this selects fixed mode); wr_weight_dpu/
    /// wr_weight_pdp (used when this selects weighted/round-robin mode).
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
    /// Description: AXI QoS (quality-of-service) level attached to the "feature" client's
    /// DDMA read transactions, presented on the AXI `arqos` signal to the interconnect.
    ///
    /// Bit width: 2 (TRM bits 1:0)
    /// Range of values: 0-3 per the AMBA AXI `arqos` convention (0 = lowest priority/default,
    /// 3 = highest); TRM prose ("Read feature_qos") does not itself define the 4 levels
    /// beyond naming the field. Reset value 0x0.
    /// Known limitations: Actual QoS-to-latency/bandwidth mapping is determined by the SoC's
    /// AXI interconnect/DRAM controller QoS policy, which is outside this chapter's scope.
    /// Related registers: rd_kernel_qos/rd_dpu_qos/rd_ppu_qos/rd_pc_qos (same register, other
    /// clients); rd_weight_feature (DdmaRdWeight0, arbitration weight for the same client).
    pub fn rd_feature_qos(&mut self, rd_feature_qos: Bits<2>) -> &mut Self {
        self.set_field(DDMA_CFG_DMA_RD_QOS_RD_FEATURE_QOS__MASK, unsafe {
            DDMA_CFG_DMA_RD_QOS_RD_FEATURE_QOS(rd_feature_qos.val())
        })
    }

    /// Description: AXI QoS (quality-of-service) level attached to the "kernel" (weight-data)
    /// client's DDMA read transactions, presented on the AXI `arqos` signal to the
    /// interconnect.
    ///
    /// Bit width: 2 (TRM bits 3:2)
    /// Range of values: 0-3 per the AMBA AXI `arqos` convention (0 = lowest priority/default,
    /// 3 = highest). Reset value 0x0.
    /// Known limitations: Actual QoS-to-latency/bandwidth mapping is determined by the SoC's
    /// AXI interconnect/DRAM controller QoS policy, which is outside this chapter's scope.
    /// Related registers: rd_feature_qos/rd_dpu_qos/rd_ppu_qos/rd_pc_qos (same register,
    /// other clients); rd_weight_kernel (DdmaRdWeight0, arbitration weight for the same
    /// client).
    pub fn rd_kernel_qos(&mut self, rd_kernel_qos: Bits<2>) -> &mut Self {
        self.set_field(DDMA_CFG_DMA_RD_QOS_RD_KERNEL_QOS__MASK, unsafe {
            DDMA_CFG_DMA_RD_QOS_RD_KERNEL_QOS(rd_kernel_qos.val())
        })
    }

    /// Description: AXI QoS (quality-of-service) level attached to the DPU client's DDMA read
    /// transactions, presented on the AXI `arqos` signal to the interconnect.
    ///
    /// Bit width: 2 (TRM bits 5:4)
    /// Range of values: 0-3 per the AMBA AXI `arqos` convention (0 = lowest priority/default,
    /// 3 = highest). Reset value 0x0.
    /// Known limitations: Actual QoS-to-latency/bandwidth mapping is determined by the SoC's
    /// AXI interconnect/DRAM controller QoS policy, which is outside this chapter's scope.
    /// Related registers: rd_feature_qos/rd_kernel_qos/rd_ppu_qos/rd_pc_qos (same register,
    /// other clients); rd_weight_dpu (DdmaRdWeight0, arbitration weight for the same client).
    pub fn rd_dpu_qos(&mut self, rd_dpu_qos: Bits<2>) -> &mut Self {
        self.set_field(DDMA_CFG_DMA_RD_QOS_RD_DPU_QOS__MASK, unsafe {
            DDMA_CFG_DMA_RD_QOS_RD_DPU_QOS(rd_dpu_qos.val())
        })
    }

    /// Description: AXI QoS (quality-of-service) level attached to the PPU client's DDMA read
    /// transactions, presented on the AXI `arqos` signal to the interconnect.
    ///
    /// Bit width: 2 (TRM bits 7:6)
    /// Range of values: 0-3 per the AMBA AXI `arqos` convention (0 = lowest priority/default,
    /// 3 = highest). Reset value 0x0.
    /// Known limitations: Actual QoS-to-latency/bandwidth mapping is determined by the SoC's
    /// AXI interconnect/DRAM controller QoS policy, which is outside this chapter's scope.
    /// Related registers: rd_feature_qos/rd_kernel_qos/rd_dpu_qos/rd_pc_qos (same register,
    /// other clients); rd_weight_pdp (DdmaRdWeight0, arbitration weight for the same client).
    pub fn rd_ppu_qos(&mut self, rd_ppu_qos: Bits<2>) -> &mut Self {
        self.set_field(DDMA_CFG_DMA_RD_QOS_RD_PPU_QOS__MASK, unsafe {
            DDMA_CFG_DMA_RD_QOS_RD_PPU_QOS(rd_ppu_qos.val())
        })
    }

    /// Description: AXI QoS (quality-of-service) level attached to the PC (task sequencer /
    /// regcmd fetch) client's DDMA read transactions, presented on the AXI `arqos` signal to
    /// the interconnect.
    ///
    /// Bit width: 2 (TRM bits 9:8)
    /// Range of values: 0-3 per the AMBA AXI `arqos` convention (0 = lowest priority/default,
    /// 3 = highest). Reset value 0x0.
    /// Known limitations: Actual QoS-to-latency/bandwidth mapping is determined by the SoC's
    /// AXI interconnect/DRAM controller QoS policy, which is outside this chapter's scope.
    /// Related registers: rd_feature_qos/rd_kernel_qos/rd_dpu_qos/rd_ppu_qos (same register,
    /// other clients); rd_weight_pc (DdmaRdWeight1, arbitration weight for the same client).
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
    /// Description: Raw AXI `arsize` value (transfer size per beat) DDMA drives on every read
    /// transaction it issues.
    ///
    /// Bit width: 3 (TRM bits 2:0)
    /// Range of values: Standard AMBA AXI `arsize` encoding: 0=1 byte, 1=2 bytes, 2=4 bytes,
    /// 3=8 bytes, 4=16 bytes, 5=32 bytes, 6=64 bytes, 7=128 bytes per beat. TRM prose
    /// ("Read_arsize") does not restate the AXI encoding itself.
    /// Known limitations: This is a raw passthrough field — the crate/caller is responsible
    /// for choosing a value consistent with the actual transfer width and any alignment
    /// requirements; no validation is performed here.
    /// Related registers: rd_arburst/rd_arprot/rd_arcache/rd_arlock (same register, other AXI
    /// read attributes); wr_awsize (DdmaCfgDmaWrCfg, write-side counterpart).
    pub fn rd_arsize(&mut self, rd_arsize: Bits<3>) -> &mut Self {
        self.set_field(DDMA_CFG_DMA_RD_CFG_RD_ARSIZE__MASK, unsafe {
            DDMA_CFG_DMA_RD_CFG_RD_ARSIZE(rd_arsize.val())
        })
    }

    /// Description: Raw AXI `arburst` value (burst type) DDMA drives on every read
    /// transaction it issues.
    ///
    /// Bit width: 2 (TRM bits 4:3)
    /// Range of values: Standard AMBA AXI `arburst` encoding: 0=FIXED, 1=INCR, 2=WRAP,
    /// 3=reserved. TRM prose ("Read_arburst") does not restate the AXI encoding itself.
    /// Known limitations: Raw passthrough field; caller is responsible for choosing a value
    /// appropriate to the access pattern.
    /// Related registers: rd_arsize/rd_arprot/rd_arcache/rd_arlock (same register, other AXI
    /// read attributes); wr_awburst (DdmaCfgDmaWrCfg, write-side counterpart).
    pub fn rd_arburst(&mut self, rd_arburst: Bits<2>) -> &mut Self {
        self.set_field(DDMA_CFG_DMA_RD_CFG_RD_ARBURST__MASK, unsafe {
            DDMA_CFG_DMA_RD_CFG_RD_ARBURST(rd_arburst.val())
        })
    }

    /// Description: Raw AXI `arprot` value (protection/privilege attributes) DDMA drives on
    /// every read transaction it issues.
    ///
    /// Bit width: 3 (TRM bits 7:5)
    /// Range of values: Standard AMBA AXI `arprot` encoding, one bit each: bit0 = unprivileged
    /// (0) vs privileged (1) access, bit1 = secure (0) vs non-secure (1), bit2 = data (0) vs
    /// instruction (1) access. TRM prose ("Read_arprot") does not restate the AXI encoding
    /// itself.
    /// Known limitations: Raw passthrough field; caller is responsible for choosing a value
    /// appropriate to the system's security/privilege configuration.
    /// Related registers: rd_arsize/rd_arburst/rd_arcache/rd_arlock (same register, other AXI
    /// read attributes); wr_awprot (DdmaCfgDmaWrCfg, write-side counterpart).
    pub fn rd_arprot(&mut self, rd_arprot: Bits<3>) -> &mut Self {
        self.set_field(DDMA_CFG_DMA_RD_CFG_RD_ARPROT__MASK, unsafe {
            DDMA_CFG_DMA_RD_CFG_RD_ARPROT(rd_arprot.val())
        })
    }

    /// Description: Raw AXI `arcache` value (memory/cacheability attributes) DDMA drives on
    /// every read transaction it issues.
    ///
    /// Bit width: 4 (TRM bits 11:8)
    /// Range of values: Standard AMBA AXI `arcache` encoding (bufferable/cacheable/
    /// allocate bits per the AXI protocol spec). TRM prose ("Read_arcache") does not restate
    /// the AXI encoding itself.
    /// Known limitations: Raw passthrough field; caller is responsible for choosing a value
    /// consistent with the target memory region's cacheability/shareability.
    /// Related registers: rd_arsize/rd_arburst/rd_arprot/rd_arlock (same register, other AXI
    /// read attributes); wr_awcache (DdmaCfgDmaWrCfg, write-side counterpart).
    pub fn rd_arcache(&mut self, rd_arcache: Bits<4>) -> &mut Self {
        self.set_field(DDMA_CFG_DMA_RD_CFG_RD_ARCACHE__MASK, unsafe {
            DDMA_CFG_DMA_RD_CFG_RD_ARCACHE(rd_arcache.val())
        })
    }

    /// Description: Raw AXI `arlock` value (exclusive/locked access indicator) DDMA drives on
    /// every read transaction it issues.
    ///
    /// Bit width: 1 (TRM bit 12)
    /// Range of values: 0 = normal access, 1 = exclusive access (per AMBA AXI `arlock`
    /// convention). TRM prose ("Read_arlock") does not restate the AXI encoding itself.
    /// Known limitations: Raw passthrough field; DDMA being a background/general-purpose DMA
    /// engine, exclusive-access semantics for it specifically are not elaborated in this
    /// chapter.
    /// Related registers: rd_arsize/rd_arburst/rd_arprot/rd_arcache (same register, other AXI
    /// read attributes); wr_awlock (DdmaCfgDmaWrCfg, write-side counterpart).
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
    /// Description: Raw AXI `awsize` value (transfer size per beat) DDMA drives on every
    /// write transaction it issues.
    ///
    /// Bit width: 3 (TRM bits 2:0)
    /// Range of values: Standard AMBA AXI `awsize` encoding: 0=1 byte, 1=2 bytes, 2=4 bytes,
    /// 3=8 bytes, 4=16 bytes, 5=32 bytes, 6=64 bytes, 7=128 bytes per beat. TRM prose
    /// ("Write_awsize") does not restate the AXI encoding itself.
    /// Known limitations: Raw passthrough field; caller is responsible for choosing a value
    /// consistent with the actual transfer width and any alignment requirements.
    /// Related registers: wr_awburst/wr_awprot/wr_awcache/wr_awlock (same register, other AXI
    /// write attributes); rd_arsize (DdmaCfgDmaRdCfg, read-side counterpart).
    pub fn wr_awsize(&mut self, wr_awsize: Bits<3>) -> &mut Self {
        self.set_field(DDMA_CFG_DMA_WR_CFG_WR_AWSIZE__MASK, unsafe {
            DDMA_CFG_DMA_WR_CFG_WR_AWSIZE(wr_awsize.val())
        })
    }

    /// Description: Raw AXI `awburst` value (burst type) DDMA drives on every write
    /// transaction it issues.
    ///
    /// Bit width: 2 (TRM bits 4:3)
    /// Range of values: Standard AMBA AXI `awburst` encoding: 0=FIXED, 1=INCR, 2=WRAP,
    /// 3=reserved. TRM prose ("Write_awburst") does not restate the AXI encoding itself.
    /// Known limitations: Raw passthrough field; caller is responsible for choosing a value
    /// appropriate to the access pattern.
    /// Related registers: wr_awsize/wr_awprot/wr_awcache/wr_awlock (same register, other AXI
    /// write attributes); rd_arburst (DdmaCfgDmaRdCfg, read-side counterpart).
    pub fn wr_awburst(&mut self, wr_awburst: Bits<2>) -> &mut Self {
        self.set_field(DDMA_CFG_DMA_WR_CFG_WR_AWBURST__MASK, unsafe {
            DDMA_CFG_DMA_WR_CFG_WR_AWBURST(wr_awburst.val())
        })
    }

    /// Description: Raw AXI `awprot` value (protection/privilege attributes) DDMA drives on
    /// every write transaction it issues.
    ///
    /// Bit width: 3 (TRM bits 7:5)
    /// Range of values: Standard AMBA AXI `awprot` encoding, one bit each: bit0 = unprivileged
    /// (0) vs privileged (1) access, bit1 = secure (0) vs non-secure (1), bit2 = data (0) vs
    /// instruction (1) access. TRM prose ("Write_awprot") does not restate the AXI encoding
    /// itself.
    /// Known limitations: Raw passthrough field; caller is responsible for choosing a value
    /// appropriate to the system's security/privilege configuration.
    /// Related registers: wr_awsize/wr_awburst/wr_awcache/wr_awlock (same register, other AXI
    /// write attributes); rd_arprot (DdmaCfgDmaRdCfg, read-side counterpart).
    pub fn wr_awprot(&mut self, wr_awprot: Bits<3>) -> &mut Self {
        self.set_field(DDMA_CFG_DMA_WR_CFG_WR_AWPROT__MASK, unsafe {
            DDMA_CFG_DMA_WR_CFG_WR_AWPROT(wr_awprot.val())
        })
    }

    /// Description: Raw AXI `awcache` value (memory/cacheability attributes) DDMA drives on
    /// every write transaction it issues.
    ///
    /// Bit width: 4 (TRM bits 11:8)
    /// Range of values: Standard AMBA AXI `awcache` encoding (bufferable/cacheable/allocate
    /// bits per the AXI protocol spec). TRM prose ("Write awcache") does not restate the AXI
    /// encoding itself.
    /// Known limitations: Raw passthrough field; caller is responsible for choosing a value
    /// consistent with the target memory region's cacheability/shareability.
    /// Related registers: wr_awsize/wr_awburst/wr_awprot/wr_awlock (same register, other AXI
    /// write attributes); rd_arcache (DdmaCfgDmaRdCfg, read-side counterpart).
    pub fn wr_awcache(&mut self, wr_awcache: Bits<4>) -> &mut Self {
        self.set_field(DDMA_CFG_DMA_WR_CFG_WR_AWCACHE__MASK, unsafe {
            DDMA_CFG_DMA_WR_CFG_WR_AWCACHE(wr_awcache.val())
        })
    }

    /// Description: Raw AXI `awlock` value (exclusive/locked access indicator) DDMA drives on
    /// every write transaction it issues.
    ///
    /// Bit width: 1 (TRM bit 12)
    /// Range of values: 0 = normal access, 1 = exclusive access (per AMBA AXI `awlock`
    /// convention). TRM prose ("Write_awlock") does not restate the AXI encoding itself.
    /// Known limitations: Raw passthrough field; DDMA being a background/general-purpose DMA
    /// engine, exclusive-access semantics for it specifically are not elaborated in this
    /// chapter.
    /// Related registers: wr_awsize/wr_awburst/wr_awprot/wr_awcache (same register, other AXI
    /// write attributes); rd_arlock (DdmaCfgDmaRdCfg, read-side counterpart).
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
    /// Description: Raw AXI `wstrb` (write strobe / per-byte-lane write-enable) value DDMA
    /// drives on its write transactions, selecting which bytes of each write beat are
    /// actually written to memory.
    ///
    /// Bit width: 32 (TRM bits 31:0)
    /// Range of values: 0x00000000-0xFFFFFFFF, one bit per byte lane (1 = byte lane is
    /// written, 0 = byte lane is masked off), per the AMBA AXI `wstrb` convention. Reset
    /// value 0x00000000.
    /// Known limitations: TRM prose ("Write_wstrb") does not clarify whether this is a
    /// static/global strobe applied to every beat of every write transaction, or how it
    /// interacts with transfers narrower than the full bus width; not confirmed against
    /// hardware behavior in this crate.
    /// Related registers: wr_awsize (DdmaCfgDmaWrCfg — transfer size, which normally
    /// determines which byte lanes are active).
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
    /// Description: Reports whether DDMA is currently idle (no outstanding read/write
    /// activity) — field name and spelling ("idel") reproduced verbatim from the TRM, which
    /// appears to be a typo for "idle".
    ///
    /// Bit width: 1 (TRM bit 8)
    /// Range of values: 0 = not idle (busy), 1 = idle (per TRM prose "Idel"; exact 0/1
    /// polarity assumed from the conventional meaning of an idle-status bit, not spelled out
    /// further in the TRM prose itself). Reset value 0.
    /// Known limitations: TRM lists this field's attribute as RW rather than RO, which is
    /// unusual for a hardware status bit — it is unconfirmed whether this is a TRM
    /// transcription error, whether software writes to it are actually ignored by hardware,
    /// or whether it has some other documented-nowhere write behavior; not validated against
    /// real hardware in this crate. Exposing it via a setter method here should be treated
    /// with that caveat in mind.
    /// Related registers: cfg_dma_fifo_clr (DDMA should typically be idle before clearing its
    /// FIFO).
    pub fn idel(&mut self, idel: Bits<1>) -> &mut Self {
        self.set_field(DDMA_CFG_STATUS_IDEL__MASK, unsafe {
            DDMA_CFG_STATUS_IDEL(idel.val())
        })
    }
}
