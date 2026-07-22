use crate::rocket::{
    builders::{Bits, Register, RegisterMeta},
    registers::*,
};

// ========================================================================
// OPERATION_ENABLE (0xF008)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct GlobalOperationEnable;

impl RegisterMeta for GlobalOperationEnable {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_GLOBAL;
    const OFFSET: u32 = REG_GLOBAL_OPERATION_ENABLE;
}

impl Register<GlobalOperationEnable> {
    /// Description: Chip-wide enable signal that triggers the CNA block to operate, independent of CNA's own per-block enable.
    ///
    /// Bit width: 1
    /// Range of values: 0 = disable; 1 = enable
    /// Known limitations: This is the GLOBAL block's own address-mapped register (0xF008), separate from CNA's own `CnaOperationEnable::op_en`; the two are not automatically kept in sync by hardware.
    /// Related registers: CnaOperationEnable::op_en, core_op_en, dpu_op_en, dpu_rdma_op_en, ppu_op_en, ppu_rdma_op_en
    pub fn cna_op_en(&mut self, cna_op_en: Bits<1>) -> &mut Self {
        self.set_field(GLOBAL_OPERATION_ENABLE_CNA_OP_EN__MASK, unsafe {
            GLOBAL_OPERATION_ENABLE_CNA_OP_EN(cna_op_en.val())
        })
    }

    /// Description: Chip-wide enable signal that triggers the CORE block to operate, independent of CORE's own per-block enable.
    ///
    /// Bit width: 1
    /// Range of values: 0 = disable; 1 = enable
    /// Known limitations: Bit 1 of this register (between cna_op_en and this field) is reserved.
    /// Related registers: CoreOperationEnable::op_en, cna_op_en, dpu_op_en
    pub fn core_op_en(&mut self, core_op_en: Bits<1>) -> &mut Self {
        self.set_field(GLOBAL_OPERATION_ENABLE_CORE_OP_EN__MASK, unsafe {
            GLOBAL_OPERATION_ENABLE_CORE_OP_EN(core_op_en.val())
        })
    }

    /// Description: Chip-wide enable signal that triggers the DPU block to operate, independent of DPU's own per-block enable.
    ///
    /// Bit width: 1
    /// Range of values: 0 = disable; 1 = enable
    /// Known limitations: None documented.
    /// Related registers: DpuOperationEnable::op_en, core_op_en, dpu_rdma_op_en
    pub fn dpu_op_en(&mut self, dpu_op_en: Bits<1>) -> &mut Self {
        self.set_field(GLOBAL_OPERATION_ENABLE_DPU_OP_EN__MASK, unsafe {
            GLOBAL_OPERATION_ENABLE_DPU_OP_EN(dpu_op_en.val())
        })
    }

    /// Description: Chip-wide enable signal that triggers the DPU_RDMA block to operate, independent of DPU_RDMA's own per-block enable.
    ///
    /// Bit width: 1
    /// Range of values: 0 = disable; 1 = enable
    /// Known limitations: Only relevant when DPU is running in flying mode (main data sourced from DPU_RDMA/MRDMA rather than the convolution pipeline).
    /// Related registers: dpu_op_en, ppu_op_en
    pub fn dpu_rdma_op_en(&mut self, dpu_rdma_op_en: Bits<1>) -> &mut Self {
        self.set_field(GLOBAL_OPERATION_ENABLE_DPU_RDMA_OP_EN__MASK, unsafe {
            GLOBAL_OPERATION_ENABLE_DPU_RDMA_OP_EN(dpu_rdma_op_en.val())
        })
    }

    /// Description: Chip-wide enable signal that triggers the PPU block to operate, independent of PPU's own per-block enable.
    ///
    /// Bit width: 1
    /// Range of values: 0 = disable; 1 = enable
    /// Known limitations: None documented.
    /// Related registers: PpuOperationEnable::op_en, dpu_rdma_op_en, ppu_rdma_op_en
    pub fn ppu_op_en(&mut self, ppu_op_en: Bits<1>) -> &mut Self {
        self.set_field(GLOBAL_OPERATION_ENABLE_PPU_OP_EN__MASK, unsafe {
            GLOBAL_OPERATION_ENABLE_PPU_OP_EN(ppu_op_en.val())
        })
    }

    /// Description: Chip-wide enable signal that triggers the PPU_RDMA block to operate, independent of PPU_RDMA's own per-block enable.
    ///
    /// Bit width: 1
    /// Range of values: 0 = disable; 1 = enable
    /// Known limitations: Only relevant when PPU is running in flying mode (standalone pooling fed by PPU_RDMA rather than pipelined after DPU).
    /// Related registers: ppu_op_en
    pub fn ppu_rdma_op_en(&mut self, ppu_rdma_op_en: Bits<1>) -> &mut Self {
        self.set_field(GLOBAL_OPERATION_ENABLE_PPU_RDMA_OP_EN__MASK, unsafe {
            GLOBAL_OPERATION_ENABLE_PPU_RDMA_OP_EN(ppu_rdma_op_en.val())
        })
    }
}
