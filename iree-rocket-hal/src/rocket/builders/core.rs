use crate::rocket::{
    builders::{Bits, Register, RegisterMeta},
    registers::*,
};

// ========================================================================
// S_STATUS (0x3000)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct CoreSStatus;

impl RegisterMeta for CoreSStatus {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_CORE;
    const OFFSET: u32 = REG_CORE_S_STATUS;
}

impl Register<CoreSStatus> {
    /// Description: Reports executer 0's current run state within the S_STATUS ping-pong pair.
    ///
    /// Bit width: 2
    /// Range of values: 0 = idle; 1 = operating; 2 = operating, executer 1 waiting to operate; 3 = reserved
    /// Known limitations: Read-only hardware status; writing it has no effect on the block's actual state.
    /// Related registers: status_1 (executer 1's equivalent field); CoreSPointer::executer (selects the live executer group)
    pub fn status_0(&mut self, status_0: Bits<2>) -> &mut Self {
        self.set_field(CORE_S_STATUS_STATUS_0__MASK, unsafe {
            CORE_S_STATUS_STATUS_0(status_0.val())
        })
    }

    /// Description: Reports executer 1's current run state within the S_STATUS ping-pong pair.
    ///
    /// Bit width: 2
    /// Range of values: 0 = idle; 1 = operating; 2 = operating, executer 1 waiting to operate; 3 = reserved
    /// Known limitations: Read-only hardware status; writing it has no effect on the block's actual state.
    /// Related registers: status_0 (executer 0's equivalent field); CoreSPointer::executer (selects the live executer group)
    pub fn status_1(&mut self, status_1: Bits<2>) -> &mut Self {
        self.set_field(CORE_S_STATUS_STATUS_1__MASK, unsafe {
            CORE_S_STATUS_STATUS_1(status_1.val())
        })
    }
}

// ========================================================================
// S_POINTER (0x3004)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct CoreSPointer;

impl RegisterMeta for CoreSPointer {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_CORE;
    const OFFSET: u32 = REG_CORE_S_POINTER;
}

impl Register<CoreSPointer> {
    /// Description: Selects which of the 2 shadow register groups is ready to be configured next.
    ///
    /// Bit width: 1
    /// Range of values: 0 = register group 0; 1 = register group 1
    /// Known limitations: Only meaningful in combination with pointer_pp_en; when ping-pong is disabled this simply selects the single active group.
    /// Related registers: pointer_pp_en, pointer_pp_mode, pointer_pp_clear, executer
    pub fn pointer(&mut self, pointer: Bits<1>) -> &mut Self {
        self.set_field(CORE_S_POINTER_POINTER__MASK, unsafe {
            CORE_S_POINTER_POINTER(pointer.val())
        })
    }

    /// Description: Enables ping-pong toggling of the register group pointer between the two shadow register groups.
    ///
    /// Bit width: 1
    /// Range of values: 0 = disable; 1 = enable
    /// Known limitations: The toggle rule (by executer vs. by pointer) is chosen separately via pointer_pp_mode.
    /// Related registers: pointer, pointer_pp_mode, executer_pp_en
    pub fn pointer_pp_en(&mut self, pointer_pp_en: Bits<1>) -> &mut Self {
        self.set_field(CORE_S_POINTER_POINTER_PP_EN__MASK, unsafe {
            CORE_S_POINTER_POINTER_PP_EN(pointer_pp_en.val())
        })
    }

    /// Description: Enables ping-pong toggling of the executer group between the two hardware executers.
    ///
    /// Bit width: 1
    /// Range of values: 0 = disable; 1 = enable
    /// Known limitations: Independent of, but typically paired with, pointer_pp_en.
    /// Related registers: executer, pointer_pp_en
    pub fn executer_pp_en(&mut self, executer_pp_en: Bits<1>) -> &mut Self {
        self.set_field(CORE_S_POINTER_EXECUTER_PP_EN__MASK, unsafe {
            CORE_S_POINTER_EXECUTER_PP_EN(executer_pp_en.val())
        })
    }

    /// Description: Selects the rule used to toggle the register group pointer when ping-pong is enabled.
    ///
    /// Bit width: 1
    /// Range of values: 0 = pointer toggles by executer (e.g. executer 0 active -> next pointer toggles to 1); 1 = pointer toggles by pointer (e.g. pointer 0 active -> next pointer toggles to 1)
    /// Known limitations: Only has an effect when pointer_pp_en is set.
    /// Related registers: pointer_pp_en, pointer, executer
    pub fn pointer_pp_mode(&mut self, pointer_pp_mode: Bits<1>) -> &mut Self {
        self.set_field(CORE_S_POINTER_POINTER_PP_MODE__MASK, unsafe {
            CORE_S_POINTER_POINTER_PP_MODE(pointer_pp_mode.val())
        })
    }

    /// Description: Write-1-to-clear: resets the register group pointer back to 0.
    ///
    /// Bit width: 1
    /// Range of values: 0 = no effect; 1 = clear pointer to 0
    /// Known limitations: Self-clearing (W1C) — reads back as 0 after being applied.
    /// Related registers: pointer, executer_pp_clear
    pub fn pointer_pp_clear(&mut self, pointer_pp_clear: Bits<1>) -> &mut Self {
        self.set_field(CORE_S_POINTER_POINTER_PP_CLEAR__MASK, unsafe {
            CORE_S_POINTER_POINTER_PP_CLEAR(pointer_pp_clear.val())
        })
    }

    /// Description: Write-1-to-clear: resets the executer group pointer back to 0.
    ///
    /// Bit width: 1
    /// Range of values: 0 = no effect; 1 = clear pointer to 0
    /// Known limitations: Self-clearing (W1C) — reads back as 0 after being applied.
    /// Related registers: executer, pointer_pp_clear
    pub fn executer_pp_clear(&mut self, executer_pp_clear: Bits<1>) -> &mut Self {
        self.set_field(CORE_S_POINTER_EXECUTER_PP_CLEAR__MASK, unsafe {
            CORE_S_POINTER_EXECUTER_PP_CLEAR(executer_pp_clear.val())
        })
    }

    /// Description: Selects which of the 2 hardware executers is currently designated to run.
    ///
    /// Bit width: 1
    /// Range of values: 0 = executer group 0; 1 = executer group 1
    /// Known limitations: The TRM lists this bit as read-only status, but the driver exposes it as a settable field alongside the rest of the ping-pong configuration.
    /// Related registers: executer_pp_en, pointer, CoreSStatus::status_0/status_1
    pub fn executer(&mut self, executer: Bits<1>) -> &mut Self {
        self.set_field(CORE_S_POINTER_EXECUTER__MASK, unsafe {
            CORE_S_POINTER_EXECUTER(executer.val())
        })
    }
}

// ========================================================================
// OPERATION_ENABLE (0x3008)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct CoreOperationEnable;

impl RegisterMeta for CoreOperationEnable {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_CORE;
    const OFFSET: u32 = REG_CORE_OPERATION_ENABLE;
}

impl Register<CoreOperationEnable> {
    /// Description: Triggers the CORE block to begin operating on the currently configured register group.
    ///
    /// Bit width: 1
    /// Range of values: 0 = disable; 1 = enable, fetch/run the block
    /// Known limitations: This register and every register after it in the CORE block are shadowed for ping-pong operation; per the PC regcmd convention this op_en entry should be written last, after all other CORE registers for the task.
    /// Related registers: CoreSPointer (selects the shadow group this enable applies to), GLOBAL::core_op_en (the chip-wide equivalent)
    pub fn op_en(&mut self, op_en: Bits<1>) -> &mut Self {
        self.set_field(CORE_OPERATION_ENABLE_OP_EN__MASK, unsafe {
            CORE_OPERATION_ENABLE_OP_EN(op_en.val())
        })
    }
}

// ========================================================================
// MAC_GATING (0x300C)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct CoreMacGating;

impl RegisterMeta for CoreMacGating {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_CORE;
    const OFFSET: u32 = REG_CORE_MAC_GATING;
}

impl Register<CoreMacGating> {
    /// Description: Per-subblock soft clock-gating enable bitmask for the MAC array's automatic clock gating.
    ///
    /// Bit width: 27
    /// Range of values: 0x0000000-0x7FFFFFF; reset value 0x07800800 (default gating pattern enabled by the hardware)
    /// Known limitations: Only relevant when tracking down clock-gating-related timing/power issues; the reset value already enables the recommended default gating and normally does not need to be touched.
    /// Related registers: CoreMiscCfg::soft_gating (accumulator gating, separate from this MAC gating field)
    pub fn slcg_op_en(&mut self, slcg_op_en: Bits<27>) -> &mut Self {
        self.set_field(CORE_MAC_GATING_SLCG_OP_EN__MASK, unsafe {
            CORE_MAC_GATING_SLCG_OP_EN(slcg_op_en.val())
        })
    }
}

// ========================================================================
// MISC_CFG (0x3010)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct CoreMiscCfg;

impl RegisterMeta for CoreMiscCfg {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_CORE;
    const OFFSET: u32 = REG_CORE_MISC_CFG;
}

impl Register<CoreMiscCfg> {
    /// Description: Enables quantized ("quantify") feature-data calculation in the accumulator.
    ///
    /// Bit width: 1
    /// Range of values: 0 = disable; 1 = enable
    /// Known limitations: None documented beyond matching the quantization mode configured upstream in CNA's convert stage.
    /// Related registers: dw_en, proc_precision, CnaCvtCon0 (input quantization convert config)
    pub fn qd_en(&mut self, qd_en: Bits<1>) -> &mut Self {
        self.set_field(CORE_MISC_CFG_QD_EN__MASK, unsafe {
            CORE_MISC_CFG_QD_EN(qd_en.val())
        })
    }

    /// Description: Enables depthwise-convolution mode in the MAC array/accumulator.
    ///
    /// Bit width: 1
    /// Range of values: 0 = disable; 1 = depthwise mode enable
    /// Known limitations: Must be kept consistent with CNA's own depthwise conv_mode setting for the same task.
    /// Related registers: CnaConvCon1::conv_mode, qd_en
    pub fn dw_en(&mut self, dw_en: Bits<1>) -> &mut Self {
        self.set_field(CORE_MISC_CFG_DW_EN__MASK, unsafe {
            CORE_MISC_CFG_DW_EN(dw_en.val())
        })
    }

    /// Description: Selects the numeric precision used for the CORE block's processing.
    ///
    /// Bit width: 3
    /// Range of values: 0 = int8; 1 = int16; 2 = fp16; 3 = bf16; 4-5 = reserved; 6 = int4; 7 = tf32
    /// Known limitations: Must match the precision configured for CNA (CnaConvCon1::proc_precision) for the same task; values 4 and 5 are reserved/undefined.
    /// Related registers: CnaConvCon1::proc_precision, CnaConvCon1::in_precision
    pub fn proc_precision(&mut self, proc_precision: Bits<3>) -> &mut Self {
        self.set_field(CORE_MISC_CFG_PROC_PRECISION__MASK, unsafe {
            CORE_MISC_CFG_PROC_PRECISION(proc_precision.val())
        })
    }

    /// Description: Accumulator soft clock-gating control signal.
    ///
    /// Bit width: 6
    /// Range of values: 0x00-0x3F
    /// Known limitations: Power/timing tuning knob; interacts with the automatic localized clock gating described for the whole RKNN block, not required for basic functional operation.
    /// Related registers: CoreMacGating::slcg_op_en
    pub fn soft_gating(&mut self, soft_gating: Bits<6>) -> &mut Self {
        self.set_field(CORE_MISC_CFG_SOFT_GATING__MASK, unsafe {
            CORE_MISC_CFG_SOFT_GATING(soft_gating.val())
        })
    }
}

// ========================================================================
// DATAOUT_SIZE_0 (0x3014)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct CoreDataoutSize0;

impl RegisterMeta for CoreDataoutSize0 {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_CORE;
    const OFFSET: u32 = REG_CORE_DATAOUT_SIZE_0;
}

impl Register<CoreDataoutSize0> {
    /// Description: Width of the data output by CORE after activation.
    ///
    /// Bit width: 16
    /// Range of values: 0x0000-0xFFFF
    /// Known limitations: Describes CORE's own output shape, which feeds DPU next in the pipeline; keep consistent with CNA's dataout_width and any downstream DPU/PPU shape registers for the same task.
    /// Related registers: dataout_height, CoreDataoutSize1::dataout_channel, CnaDataSize2::dataout_width
    pub fn dataout_width(&mut self, dataout_width: Bits<16>) -> &mut Self {
        self.set_field(CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH__MASK, unsafe {
            CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH(dataout_width.val())
        })
    }

    /// Description: Height of the data output by CORE after activation.
    ///
    /// Bit width: 16
    /// Range of values: 0x0000-0xFFFF
    /// Known limitations: Describes CORE's own output shape, which feeds DPU next in the pipeline.
    /// Related registers: dataout_width, CoreDataoutSize1::dataout_channel
    pub fn dataout_height(&mut self, dataout_height: Bits<16>) -> &mut Self {
        self.set_field(CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT__MASK, unsafe {
            CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT(dataout_height.val())
        })
    }
}

// ========================================================================
// DATAOUT_SIZE_1 (0x3018)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct CoreDataoutSize1;

impl RegisterMeta for CoreDataoutSize1 {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_CORE;
    const OFFSET: u32 = REG_CORE_DATAOUT_SIZE_1;
}

impl Register<CoreDataoutSize1> {
    /// Description: Number of channels in the data output by CORE after activation.
    ///
    /// Bit width: 16
    /// Range of values: 0x0000-0xFFFF
    /// Known limitations: Describes CORE's own output shape, which feeds DPU next in the pipeline.
    /// Related registers: CoreDataoutSize0::dataout_width, CoreDataoutSize0::dataout_height
    pub fn dataout_channel(&mut self, dataout_channel: Bits<16>) -> &mut Self {
        self.set_field(CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL__MASK, unsafe {
            CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL(dataout_channel.val())
        })
    }
}

// ========================================================================
// CLIP_TRUNCATE (0x301C)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct CoreClipTruncate;

impl RegisterMeta for CoreClipTruncate {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_CORE;
    const OFFSET: u32 = REG_CORE_CLIP_TRUNCATE;
}

impl Register<CoreClipTruncate> {
    /// Description: Number of bits the accumulator's output is truncated/shifted by before leaving CORE.
    ///
    /// Bit width: 5
    /// Range of values: 0x00-0x1F
    /// Known limitations: Bit 5 of this register is reserved and sits between this field and round_type; the exact shift needed depends on the quantization scale chosen upstream in CNA's convert stage.
    /// Related registers: round_type, CnaCvtCon0::cvt_truncate_0..3
    pub fn clip_truncate(&mut self, clip_truncate: Bits<5>) -> &mut Self {
        self.set_field(CORE_CLIP_TRUNCATE_CLIP_TRUNCATE__MASK, unsafe {
            CORE_CLIP_TRUNCATE_CLIP_TRUNCATE(clip_truncate.val())
        })
    }

    /// Description: Selects the rounding rule applied when truncating the accumulator's output.
    ///
    /// Bit width: 1
    /// Range of values: 0 = odd-in-even-not (round-half-to-even); 1 = round-half-up (0.5 rounds up to 1)
    /// Known limitations: None documented.
    /// Related registers: clip_truncate
    pub fn round_type(&mut self, round_type: Bits<1>) -> &mut Self {
        self.set_field(CORE_CLIP_TRUNCATE_ROUND_TYPE__MASK, unsafe {
            CORE_CLIP_TRUNCATE_ROUND_TYPE(round_type.val())
        })
    }
}
