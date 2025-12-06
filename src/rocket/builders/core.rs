use crate::rocket::{
    builders::{Bits, Register, RegisterMeta},
    registers::*,
};

// ========================================================================
// S_STATUS (0x3000)
// ========================================================================
pub struct CoreSStatus;

impl RegisterMeta for CoreSStatus {
    const DOMAIN: u32 = target_CORE;
    const OFFSET: u32 = REG_CORE_S_STATUS;
}

impl Register<CoreSStatus> {
    pub fn status_0(&mut self, status_0: Bits<2>) -> &mut Self {
        self.set_field(CORE_S_STATUS_STATUS_0__MASK, unsafe {
            CORE_S_STATUS_STATUS_0(status_0.val())
        })
    }

    pub fn status_1(&mut self, status_1: Bits<2>) -> &mut Self {
        self.set_field(CORE_S_STATUS_STATUS_1__MASK, unsafe {
            CORE_S_STATUS_STATUS_1(status_1.val())
        })
    }
}

// ========================================================================
// S_POINTER (0x3004)
// ========================================================================
pub struct CoreSPointer;

impl RegisterMeta for CoreSPointer {
    const DOMAIN: u32 = target_CORE;
    const OFFSET: u32 = REG_CORE_S_POINTER;
}

impl Register<CoreSPointer> {
    pub fn pointer(&mut self, pointer: Bits<1>) -> &mut Self {
        self.set_field(CORE_S_POINTER_POINTER__MASK, unsafe {
            CORE_S_POINTER_POINTER(pointer.val())
        })
    }

    pub fn pointer_pp_en(&mut self, pointer_pp_en: Bits<1>) -> &mut Self {
        self.set_field(CORE_S_POINTER_POINTER_PP_EN__MASK, unsafe {
            CORE_S_POINTER_POINTER_PP_EN(pointer_pp_en.val())
        })
    }

    pub fn executer_pp_en(&mut self, executer_pp_en: Bits<1>) -> &mut Self {
        self.set_field(CORE_S_POINTER_EXECUTER_PP_EN__MASK, unsafe {
            CORE_S_POINTER_EXECUTER_PP_EN(executer_pp_en.val())
        })
    }

    pub fn pointer_pp_mode(&mut self, pointer_pp_mode: Bits<1>) -> &mut Self {
        self.set_field(CORE_S_POINTER_POINTER_PP_MODE__MASK, unsafe {
            CORE_S_POINTER_POINTER_PP_MODE(pointer_pp_mode.val())
        })
    }

    pub fn pointer_pp_clear(&mut self, pointer_pp_clear: Bits<1>) -> &mut Self {
        self.set_field(CORE_S_POINTER_POINTER_PP_CLEAR__MASK, unsafe {
            CORE_S_POINTER_POINTER_PP_CLEAR(pointer_pp_clear.val())
        })
    }

    pub fn executer_pp_clear(&mut self, executer_pp_clear: Bits<1>) -> &mut Self {
        self.set_field(CORE_S_POINTER_EXECUTER_PP_CLEAR__MASK, unsafe {
            CORE_S_POINTER_EXECUTER_PP_CLEAR(executer_pp_clear.val())
        })
    }

    pub fn executer(&mut self, executer: Bits<1>) -> &mut Self {
        self.set_field(CORE_S_POINTER_EXECUTER__MASK, unsafe {
            CORE_S_POINTER_EXECUTER(executer.val())
        })
    }
}

// ========================================================================
// OPERATION_ENABLE (0x3008)
// ========================================================================
pub struct CoreOperationEnable;

impl RegisterMeta for CoreOperationEnable {
    const DOMAIN: u32 = target_CORE;
    const OFFSET: u32 = REG_CORE_OPERATION_ENABLE;
}

impl Register<CoreOperationEnable> {
    pub fn op_en(&mut self, op_en: Bits<1>) -> &mut Self {
        self.set_field(CORE_OPERATION_ENABLE_OP_EN__MASK, unsafe {
            CORE_OPERATION_ENABLE_OP_EN(op_en.val())
        })
    }
}

// ========================================================================
// MAC_GATING (0x300C)
// ========================================================================
pub struct CoreMacGating;

impl RegisterMeta for CoreMacGating {
    const DOMAIN: u32 = target_CORE;
    const OFFSET: u32 = REG_CORE_MAC_GATING;
}

impl Register<CoreMacGating> {
    pub fn slcg_op_en(&mut self, slcg_op_en: Bits<27>) -> &mut Self {
        self.set_field(CORE_MAC_GATING_SLCG_OP_EN__MASK, unsafe {
            CORE_MAC_GATING_SLCG_OP_EN(slcg_op_en.val())
        })
    }
}

// ========================================================================
// MISC_CFG (0x3010)
// ========================================================================
pub struct CoreMiscCfg;

impl RegisterMeta for CoreMiscCfg {
    const DOMAIN: u32 = target_CORE;
    const OFFSET: u32 = REG_CORE_MISC_CFG;
}

impl Register<CoreMiscCfg> {
    pub fn qd_en(&mut self, qd_en: Bits<1>) -> &mut Self {
        self.set_field(CORE_MISC_CFG_QD_EN__MASK, unsafe {
            CORE_MISC_CFG_QD_EN(qd_en.val())
        })
    }

    pub fn dw_en(&mut self, dw_en: Bits<1>) -> &mut Self {
        self.set_field(CORE_MISC_CFG_DW_EN__MASK, unsafe {
            CORE_MISC_CFG_DW_EN(dw_en.val())
        })
    }

    pub fn proc_precision(&mut self, proc_precision: Bits<3>) -> &mut Self {
        self.set_field(CORE_MISC_CFG_PROC_PRECISION__MASK, unsafe {
            CORE_MISC_CFG_PROC_PRECISION(proc_precision.val())
        })
    }

    pub fn soft_gating(&mut self, soft_gating: Bits<6>) -> &mut Self {
        self.set_field(CORE_MISC_CFG_SOFT_GATING__MASK, unsafe {
            CORE_MISC_CFG_SOFT_GATING(soft_gating.val())
        })
    }
}

// ========================================================================
// DATAOUT_SIZE_0 (0x3014)
// ========================================================================
pub struct CoreDataoutSize0;

impl RegisterMeta for CoreDataoutSize0 {
    const DOMAIN: u32 = target_CORE;
    const OFFSET: u32 = REG_CORE_DATAOUT_SIZE_0;
}

impl Register<CoreDataoutSize0> {
    pub fn dataout_width(&mut self, dataout_width: Bits<16>) -> &mut Self {
        self.set_field(CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH__MASK, unsafe {
            CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH(dataout_width.val())
        })
    }

    pub fn dataout_height(&mut self, dataout_height: Bits<16>) -> &mut Self {
        self.set_field(CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT__MASK, unsafe {
            CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT(dataout_height.val())
        })
    }
}

// ========================================================================
// DATAOUT_SIZE_1 (0x3018)
// ========================================================================
pub struct CoreDataoutSize1;

impl RegisterMeta for CoreDataoutSize1 {
    const DOMAIN: u32 = target_CORE;
    const OFFSET: u32 = REG_CORE_DATAOUT_SIZE_1;
}

impl Register<CoreDataoutSize1> {
    pub fn dataout_channel(&mut self, dataout_channel: Bits<16>) -> &mut Self {
        self.set_field(CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL__MASK, unsafe {
            CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL(dataout_channel.val())
        })
    }
}

// ========================================================================
// CLIP_TRUNCATE (0x301C)
// ========================================================================
pub struct CoreClipTruncate;

impl RegisterMeta for CoreClipTruncate {
    const DOMAIN: u32 = target_CORE;
    const OFFSET: u32 = REG_CORE_CLIP_TRUNCATE;
}

impl Register<CoreClipTruncate> {
    pub fn clip_truncate(&mut self, clip_truncate: Bits<5>) -> &mut Self {
        self.set_field(CORE_CLIP_TRUNCATE_CLIP_TRUNCATE__MASK, unsafe {
            CORE_CLIP_TRUNCATE_CLIP_TRUNCATE(clip_truncate.val())
        })
    }

    pub fn round_type(&mut self, round_type: Bits<1>) -> &mut Self {
        self.set_field(CORE_CLIP_TRUNCATE_ROUND_TYPE__MASK, unsafe {
            CORE_CLIP_TRUNCATE_ROUND_TYPE(round_type.val())
        })
    }
}
