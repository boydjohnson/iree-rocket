use crate::rocket::{
    builders::{Bits, Register, RegisterMeta},
    registers::*,
};

// ========================================================================
// OPERATION_ENABLE (0xF008)
// ========================================================================
#[derive(Clone, Copy)]
pub struct GlobalOperationEnable;

impl RegisterMeta for GlobalOperationEnable {
    const DOMAIN: u32 = target_GLOBAL;
    const OFFSET: u32 = REG_GLOBAL_OPERATION_ENABLE;
}

impl Register<GlobalOperationEnable> {
    pub fn cna_op_en(&mut self, cna_op_en: Bits<1>) -> &mut Self {
        self.set_field(GLOBAL_OPERATION_ENABLE_CNA_OP_EN__MASK, unsafe {
            GLOBAL_OPERATION_ENABLE_CNA_OP_EN(cna_op_en.val())
        })
    }

    pub fn core_op_en(&mut self, core_op_en: Bits<1>) -> &mut Self {
        self.set_field(GLOBAL_OPERATION_ENABLE_CORE_OP_EN__MASK, unsafe {
            GLOBAL_OPERATION_ENABLE_CORE_OP_EN(core_op_en.val())
        })
    }

    pub fn dpu_op_en(&mut self, dpu_op_en: Bits<1>) -> &mut Self {
        self.set_field(GLOBAL_OPERATION_ENABLE_DPU_OP_EN__MASK, unsafe {
            GLOBAL_OPERATION_ENABLE_DPU_OP_EN(dpu_op_en.val())
        })
    }

    pub fn dpu_rdma_op_en(&mut self, dpu_rdma_op_en: Bits<1>) -> &mut Self {
        self.set_field(GLOBAL_OPERATION_ENABLE_DPU_RDMA_OP_EN__MASK, unsafe {
            GLOBAL_OPERATION_ENABLE_DPU_RDMA_OP_EN(dpu_rdma_op_en.val())
        })
    }

    pub fn ppu_op_en(&mut self, ppu_op_en: Bits<1>) -> &mut Self {
        self.set_field(GLOBAL_OPERATION_ENABLE_PPU_OP_EN__MASK, unsafe {
            GLOBAL_OPERATION_ENABLE_PPU_OP_EN(ppu_op_en.val())
        })
    }

    pub fn ppu_rdma_op_en(&mut self, ppu_rdma_op_en: Bits<1>) -> &mut Self {
        self.set_field(GLOBAL_OPERATION_ENABLE_PPU_RDMA_OP_EN__MASK, unsafe {
            GLOBAL_OPERATION_ENABLE_PPU_RDMA_OP_EN(ppu_rdma_op_en.val())
        })
    }
}
