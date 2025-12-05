use crate::rocket::{
    builders::{Bits, Register, RegisterMeta},
    registers::*,
};

pub struct PCOperationEnable;

impl RegisterMeta for PCOperationEnable {
    const DOMAIN: u32 = target_PC;
    const OFFSET: u32 = REG_PC_OPERATION_ENABLE;
}

impl Register<PCOperationEnable> {
    pub fn op_enable(&mut self, enable: bool) -> &mut Self {
        self.set_flag(unsafe { PC_OPERATION_ENABLE_OP_EN(1) }, enable)
    }
}

pub struct PCBaseAddress;

impl RegisterMeta for PCBaseAddress {
    const DOMAIN: u32 = target_PC;
    const OFFSET: u32 = REG_PC_BASE_ADDRESS;
}

impl Register<PCBaseAddress> {
    pub fn pc_sel(&mut self, enable: bool) -> &mut Self {
        self.set_flag(unsafe { PC_BASE_ADDRESS_PC_SEL(1) }, enable)
    }

    pub fn pc_src_address(&mut self, val: Bits<28>) -> &mut Self {
        self.set_field(PC_BASE_ADDRESS_PC_SOURCE_ADDR__MASK, unsafe {
            PC_BASE_ADDRESS_PC_SOURCE_ADDR(val.val())
        })
    }
}

pub struct PCRegisterAmounts;

impl RegisterMeta for PCRegisterAmounts {
    const DOMAIN: u32 = target_PC;
    const OFFSET: u32 = REG_PC_REGISTER_AMOUNTS;
}

impl Register<PCRegisterAmounts> {
    pub fn pc_data_amount(&mut self, val: Bits<16>) -> &mut Self {
        self.set_field(PC_REGISTER_AMOUNTS_PC_DATA_AMOUNT__MASK, unsafe {
            PC_REGISTER_AMOUNTS_PC_DATA_AMOUNT(val.val())
        })
    }
}

pub struct PCInterruptMask;

impl RegisterMeta for PCInterruptMask {
    const DOMAIN: u32 = target_PC;
    const OFFSET: u32 = REG_PC_INTERRUPT_MASK;
}

impl Register<PCInterruptMask> {
    pub fn cna_feature_0(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_MASK_CNA_FEATURE_0, enable)
    }

    pub fn cna_feature_1(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_MASK_CNA_FEATURE_1, enable)
    }

    pub fn cna_weight_0(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_MASK_CNA_WEIGHT_0, enable)
    }

    pub fn cna_weight_1(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_MASK_CNA_WEIGHT_1, enable)
    }

    pub fn cna_csc_0(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_MASK_CNA_CSC_0, enable)
    }

    pub fn cna_csc_1(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_MASK_CNA_CSC_1, enable)
    }

    pub fn core_0(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_MASK_CORE_0, enable)
    }

    pub fn core_1(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_MASK_CORE_1, enable)
    }

    pub fn dpu_0(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_MASK_DPU_0, enable)
    }

    pub fn dpu_1(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_MASK_DPU_1, enable)
    }

    pub fn ppu_0(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_MASK_PPU_0, enable)
    }

    pub fn ppu_1(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_MASK_PPU_1, enable)
    }

    pub fn dma_read_error(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_MASK_DMA_READ_ERROR, enable)
    }

    pub fn dma_write_error(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_MASK_DMA_WRITE_ERROR, enable)
    }
}

pub struct PCInterruptClear;

impl RegisterMeta for PCInterruptClear {
    const DOMAIN: u32 = target_PC;
    const OFFSET: u32 = REG_PC_INTERRUPT_CLEAR;
}

impl Register<PCInterruptClear> {
    pub fn cna_feature_0(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_CLEAR_CNA_FEATURE_0, enable)
    }

    pub fn cna_feature_1(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_CLEAR_CNA_FEATURE_1, enable)
    }

    pub fn cna_weight_0(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_CLEAR_CNA_WEIGHT_0, enable)
    }

    pub fn cna_weight_1(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_CLEAR_CNA_WEIGHT_1, enable)
    }

    pub fn cna_csc_0(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_CLEAR_CNA_CSC_0, enable)
    }

    pub fn cna_csc_1(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_CLEAR_CNA_CSC_1, enable)
    }

    pub fn core_0(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_CLEAR_CORE_0, enable)
    }

    pub fn core_1(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_CLEAR_CORE_1, enable)
    }

    pub fn dpu_0(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_CLEAR_DPU_0, enable)
    }

    pub fn dpu_1(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_CLEAR_DPU_1, enable)
    }

    pub fn ppu_0(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_CLEAR_PPU_0, enable)
    }

    pub fn ppu_1(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_CLEAR_PPU_1, enable)
    }

    pub fn dma_read_error(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_CLEAR_DMA_READ_ERROR, enable)
    }

    pub fn dma_write_error(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_CLEAR_DMA_WRITE_ERROR, enable)
    }
}

pub struct PCTaskCon;

impl RegisterMeta for PCTaskCon {
    const DOMAIN: u32 = target_PC;
    const OFFSET: u32 = REG_PC_TASK_CON;
}

impl Register<PCTaskCon> {
    pub fn task_number(&mut self, task_number: Bits<12>) -> &mut Self {
        self.set_field(PC_TASK_CON_TASK_NUMBER__MASK, unsafe {
            PC_TASK_CON_TASK_NUMBER(task_number.val())
        })
    }

    pub fn task_pp_enable(&mut self, enable: bool) -> &mut Self {
        self.set_field(PC_TASK_CON_TASK_PP_EN__MASK, unsafe {
            PC_TASK_CON_TASK_PP_EN(enable as u32)
        })
    }

    pub fn count_clear(&mut self, enable: bool) -> &mut Self {
        self.set_field(PC_TASK_CON_TASK_COUNT_CLEAR__MASK, unsafe {
            PC_TASK_CON_TASK_COUNT_CLEAR(enable as u32)
        })
    }
}

pub struct PCTaskDMABaseAddr;

impl RegisterMeta for PCTaskDMABaseAddr {
    const DOMAIN: u32 = target_PC;
    const OFFSET: u32 = REG_PC_TASK_DMA_BASE_ADDR;
}

impl Register<PCTaskDMABaseAddr> {
    pub fn base_address(&mut self, address: Bits<28>) -> &mut Self {
        self.set_field(PC_TASK_DMA_BASE_ADDR_DMA_BASE_ADDR__MASK, unsafe {
            PC_TASK_DMA_BASE_ADDR_DMA_BASE_ADDR(address.val())
        })
    }
}
