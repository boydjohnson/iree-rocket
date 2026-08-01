use std::ops::{BitOr, BitOrAssign};

use crate::rocket::{
    builders::{Bits, RegCmd, Register, RegisterMeta},
    registers::*,
};

#[derive(Debug, Clone, Copy)]
pub struct PCOperationEnable;

impl RegisterMeta for PCOperationEnable {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_PC;
    const OFFSET: u32 = REG_PC_OPERATION_ENABLE;
}

impl Register<PCOperationEnable> {
    /// Description: Enables the PC module to fetch and apply the register program for each task.
    ///
    /// Bit width: 1
    /// Range of values: false (1'd0) = disable PC module; true (1'd1) = enable PC module to fetch registers for each task.
    /// Known limitations: This is the AHB-side "kick" bit; a regcmd buffer applied via PC mode additionally carries its own per-block op_en entries (bit 55 of each 64-bit regcmd word), which are strongly recommended to be the last entries in the list, immediately preceded by a `0x0041_xxxx_xxxx_xxxx`-style entry (see pc_register_amounts). Whether this bit drives slave-mode AHB writes or autonomous PC-mode DMA depends on pc_base_address.pc_sel.
    /// Related registers: pc_base_address (pc_sel mode select, pc_src_address fetch address), pc_register_amounts (pc_data_amount), pc_task_con (task_number).
    pub fn op_enable(&mut self, enable: bool) -> &mut Self {
        self.set_flag(unsafe { PC_OPERATION_ENABLE_OP_EN(1) }, enable)
    }
}

/// Per-block operation-enable mask carried by the final PC regcmd word.
///
/// This mask is distinct from the AHB-side [`PCOperationEnable`] register
/// builder above. Each bit starts one engine block configured by the
/// preceding register program.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PCOperationMask(u32);

impl PCOperationMask {
    pub const CNA: Self = Self(1 << 0);
    // Bit 1 has no corresponding engine target.
    pub const CORE: Self = Self(1 << 2);
    pub const DPU: Self = Self(1 << 3);
    pub const DPU_RDMA: Self = Self(1 << 4);
    pub const PPU: Self = Self(1 << 5);
    pub const PPU_RDMA: Self = Self(1 << 6);

    pub const CONVOLUTION: Self = Self(Self::CNA.0 | Self::CORE.0 | Self::DPU.0 | Self::DPU_RDMA.0);

    pub const fn bits(self) -> u32 {
        self.0
    }
}

impl BitOr for PCOperationMask {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for PCOperationMask {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

/// Typed constructors for the non-register instruction words in a PC
/// regcmd trailer.
///
/// These words are part of the PC program protocol rather than ordinary
/// register writes, so they cannot implement [`RegisterMeta`].
#[derive(Debug, Clone, Copy)]
pub struct PCTrailer;

impl PCTrailer {
    /// Placeholder emitted before the trailer of a single-task program.
    pub fn single_task_placeholder() -> RegCmd {
        RegCmd::new_raw(0)
    }

    /// Marker required immediately before the operation-enable instruction.
    pub fn required_marker() -> RegCmd {
        RegCmd::new_raw(0x0041_0000_0000_0000)
    }

    /// Starts exactly the engine blocks selected by `mask`.
    pub fn operation_enable(mask: PCOperationMask) -> RegCmd {
        // Domain 0x81 sets the regcmd operation-enable flag while offset
        // 0x0008 selects PC_OPERATION_ENABLE.
        RegCmd::new(0x81, REG_PC_OPERATION_ENABLE, mask.bits())
    }

    /// Zero word used to align a PC register program.
    pub fn alignment_padding() -> RegCmd {
        RegCmd::new_raw(0)
    }
}

#[cfg(test)]
mod trailer_tests {
    use super::*;

    #[test]
    fn convolution_trailer_matches_vendor_encoding() {
        assert_eq!(PCTrailer::single_task_placeholder().0, 0);
        assert_eq!(PCTrailer::required_marker().0, 0x0041_0000_0000_0000);
        assert_eq!(
            PCTrailer::operation_enable(PCOperationMask::CONVOLUTION).0,
            0x0081_0000_001d_0008
        );
        assert_eq!(PCTrailer::alignment_padding().0, 0);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PCBaseAddress;

impl RegisterMeta for PCBaseAddress {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_PC;
    const OFFSET: u32 = REG_PC_BASE_ADDRESS;
}

impl Register<PCBaseAddress> {
    /// Description: Selects whether register configuration arrives via autonomous PC-mode DMA or direct AHB slave writes.
    ///
    /// Bit width: 1
    /// Range of values: false (1'd0) = PC mode, use AXI DMA to fetch register config; true (1'd1) = Slave mode, use AHB to set registers directly.
    /// Known limitations: Only meaningful together with pc_src_address, which is ignored in slave mode since no autonomous fetch occurs. Reserved bits 3:1 of this register sit between pc_sel and pc_src_address.
    /// Related registers: pc_src_address (same register, regcmd fetch address), pc_operation_enable.op_en (starts fetch/apply), pc_task_dma_base_addr (per-task data DMA base, distinct from this regcmd fetch address).
    pub fn pc_sel(&mut self, enable: bool) -> &mut Self {
        self.set_flag(unsafe { PC_BASE_ADDRESS_PC_SEL(1) }, enable)
    }

    /// Description: Base address in system memory of the regcmd register-program DMA'd and applied autonomously in PC mode.
    ///
    /// Bit width: 28
    /// Range of values: Address bits [31:4] of the DMA source location (16-byte aligned); bits [3:1] are reserved/RO in the underlying register.
    /// Known limitations: Only used when pc_sel selects PC mode; ignored in slave mode. The pointed-to buffer must follow the 64-bit regcmd word format (block-select bits [63:48], value bits [47:16], offset bits [15:0] per entry), with per-block op_en entries placed last.
    /// Related registers: pc_sel (mode select), pc_register_amounts.pc_data_amount (number of regcmd words to fetch for the task), pc_operation_enable.op_en (triggers the fetch).
    pub fn pc_src_address(&mut self, val: Bits<28>) -> &mut Self {
        self.set_field(PC_BASE_ADDRESS_PC_SOURCE_ADDR__MASK, unsafe {
            PC_BASE_ADDRESS_PC_SOURCE_ADDR(val.val())
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PCRegisterAmounts;

impl RegisterMeta for PCRegisterAmounts {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_PC;
    const OFFSET: u32 = REG_PC_REGISTER_AMOUNTS;
}

impl Register<PCRegisterAmounts> {
    /// Description: Number of 64-bit regcmd register-program words PC must fetch and apply for one task.
    ///
    /// Bit width: 16
    /// Range of values: 0-65535 registers. Each register entry is 64 bits wide, formatted as bit[63:48] = target block select (bit[56]=PC, bit[57]=CNA, bit[59]=CORE, bit[60]=DPU, bit[61]=DPU_RDMA, bit[62]=PPU, bit[63]=PPU_RDMA), bit[55]=1 to set that block's own op_en, bit[47:16] = the register's value, bit[15:0] = the register's offset address within its block.
    /// Known limitations: op_en-setting entries (bit[55]=1, e.g. `0x0081_0000_007f_0008` to set every block's op_en at once) are strongly recommended to be last in the list; a `0x0041_xxxx_xxxx_xxxx`-style entry must be written immediately before the final op_en entries.
    /// Related registers: pc_base_address.pc_src_address (fetch source address), pc_task_con.task_number (number of tasks), pc_operation_enable.op_en (starts fetch).
    pub fn pc_data_amount(&mut self, val: Bits<16>) -> &mut Self {
        self.set_field(PC_REGISTER_AMOUNTS_PC_DATA_AMOUNT__MASK, unsafe {
            PC_REGISTER_AMOUNTS_PC_DATA_AMOUNT(val.val())
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PCInterruptMask;

impl RegisterMeta for PCInterruptMask {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_PC;
    const OFFSET: u32 = REG_PC_INTERRUPT_MASK;
}

impl Register<PCInterruptMask> {
    /// Description: Enables propagation of the CNA feature group 0 completion interrupt (int_mask bit 0) to the PC's output interrupt lines.
    ///
    /// Bit width: 1
    /// Range of values: false = masked (event won't propagate); true = enabled (set 1 to enable interrupt).
    /// Known limitations: In PC mode, this mask register sets the masking that applies to the last task in the running group (per TRM note on pc_interrupt_mask). Reset value of the full 17-bit field is 0x1ffff (all events enabled by default).
    /// Related registers: pc_interrupt_clear.cna_feature_0 (W1C for the same event), pc_interrupt_status/pc_interrupt_raw_status bit 0 (not modeled in this crate).
    pub fn cna_feature_0(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_MASK_CNA_FEATURE_0, enable)
    }

    /// Description: Enables propagation of the CNA feature group 1 completion interrupt (int_mask bit 1) to the PC's output interrupt lines.
    ///
    /// Bit width: 1
    /// Range of values: false = masked; true = enabled (set 1 to enable interrupt).
    /// Known limitations: In PC mode, this mask register sets the masking that applies to the last task in the running group. Reset value of the full 17-bit field is 0x1ffff.
    /// Related registers: pc_interrupt_clear.cna_feature_1, pc_interrupt_status/pc_interrupt_raw_status bit 1 (not modeled in this crate).
    pub fn cna_feature_1(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_MASK_CNA_FEATURE_1, enable)
    }

    /// Description: Enables propagation of the CNA weight group 0 completion interrupt (int_mask bit 2) to the PC's output interrupt lines.
    ///
    /// Bit width: 1
    /// Range of values: false = masked; true = enabled (set 1 to enable interrupt).
    /// Known limitations: In PC mode, this mask register sets the masking that applies to the last task in the running group. Reset value of the full 17-bit field is 0x1ffff.
    /// Related registers: pc_interrupt_clear.cna_weight_0, pc_interrupt_status/pc_interrupt_raw_status bit 2 (not modeled in this crate).
    pub fn cna_weight_0(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_MASK_CNA_WEIGHT_0, enable)
    }

    /// Description: Enables propagation of the CNA weight group 1 completion interrupt (int_mask bit 3) to the PC's output interrupt lines.
    ///
    /// Bit width: 1
    /// Range of values: false = masked; true = enabled (set 1 to enable interrupt).
    /// Known limitations: In PC mode, this mask register sets the masking that applies to the last task in the running group. Reset value of the full 17-bit field is 0x1ffff.
    /// Related registers: pc_interrupt_clear.cna_weight_1, pc_interrupt_status/pc_interrupt_raw_status bit 3 (not modeled in this crate).
    pub fn cna_weight_1(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_MASK_CNA_WEIGHT_1, enable)
    }

    /// Description: Enables propagation of the CNA csc (convert) group 0 completion interrupt (int_mask bit 4) to the PC's output interrupt lines.
    ///
    /// Bit width: 1
    /// Range of values: false = masked; true = enabled (set 1 to enable interrupt).
    /// Known limitations: In PC mode, this mask register sets the masking that applies to the last task in the running group. Reset value of the full 17-bit field is 0x1ffff.
    /// Related registers: pc_interrupt_clear.cna_csc_0, pc_interrupt_status/pc_interrupt_raw_status bit 4 (not modeled in this crate).
    pub fn cna_csc_0(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_MASK_CNA_CSC_0, enable)
    }

    /// Description: Enables propagation of the CNA csc (convert) group 1 completion interrupt (int_mask bit 5) to the PC's output interrupt lines.
    ///
    /// Bit width: 1
    /// Range of values: false = masked; true = enabled (set 1 to enable interrupt).
    /// Known limitations: In PC mode, this mask register sets the masking that applies to the last task in the running group. Reset value of the full 17-bit field is 0x1ffff.
    /// Related registers: pc_interrupt_clear.cna_csc_1, pc_interrupt_status/pc_interrupt_raw_status bit 5 (not modeled in this crate).
    pub fn cna_csc_1(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_MASK_CNA_CSC_1, enable)
    }

    /// Description: Enables propagation of the CORE group 0 completion interrupt (int_mask bit 6) to the PC's output interrupt lines.
    ///
    /// Bit width: 1
    /// Range of values: false = masked; true = enabled (set 1 to enable interrupt).
    /// Known limitations: In PC mode, this mask register sets the masking that applies to the last task in the running group. Reset value of the full 17-bit field is 0x1ffff.
    /// Related registers: pc_interrupt_clear.core_0, pc_interrupt_status/pc_interrupt_raw_status bit 6 (not modeled in this crate).
    pub fn core_0(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_MASK_CORE_0, enable)
    }

    /// Description: Enables propagation of the CORE group 1 completion interrupt (int_mask bit 7) to the PC's output interrupt lines.
    ///
    /// Bit width: 1
    /// Range of values: false = masked; true = enabled (set 1 to enable interrupt).
    /// Known limitations: In PC mode, this mask register sets the masking that applies to the last task in the running group. Reset value of the full 17-bit field is 0x1ffff.
    /// Related registers: pc_interrupt_clear.core_1, pc_interrupt_status/pc_interrupt_raw_status bit 7 (not modeled in this crate).
    pub fn core_1(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_MASK_CORE_1, enable)
    }

    /// Description: Enables propagation of the DPU group 0 completion interrupt (int_mask bit 8) to the PC's output interrupt lines.
    ///
    /// Bit width: 1
    /// Range of values: false = masked; true = enabled (set 1 to enable interrupt).
    /// Known limitations: In PC mode, this mask register sets the masking that applies to the last task in the running group. Reset value of the full 17-bit field is 0x1ffff.
    /// Related registers: pc_interrupt_clear.dpu_0, pc_interrupt_status/pc_interrupt_raw_status bit 8 (not modeled in this crate).
    pub fn dpu_0(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_MASK_DPU_0, enable)
    }

    /// Description: Enables propagation of the DPU group 1 completion interrupt (int_mask bit 9) to the PC's output interrupt lines.
    ///
    /// Bit width: 1
    /// Range of values: false = masked; true = enabled (set 1 to enable interrupt).
    /// Known limitations: In PC mode, this mask register sets the masking that applies to the last task in the running group. Reset value of the full 17-bit field is 0x1ffff.
    /// Related registers: pc_interrupt_clear.dpu_1, pc_interrupt_status/pc_interrupt_raw_status bit 9 (not modeled in this crate).
    pub fn dpu_1(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_MASK_DPU_1, enable)
    }

    /// Description: Enables propagation of the PPU group 0 completion interrupt (int_mask bit 10) to the PC's output interrupt lines.
    ///
    /// Bit width: 1
    /// Range of values: false = masked; true = enabled (set 1 to enable interrupt).
    /// Known limitations: In PC mode, this mask register sets the masking that applies to the last task in the running group. Reset value of the full 17-bit field is 0x1ffff.
    /// Related registers: pc_interrupt_clear.ppu_0, pc_interrupt_status/pc_interrupt_raw_status bit 10 (not modeled in this crate).
    pub fn ppu_0(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_MASK_PPU_0, enable)
    }

    /// Description: Enables propagation of the PPU group 1 completion interrupt (int_mask bit 11) to the PC's output interrupt lines.
    ///
    /// Bit width: 1
    /// Range of values: false = masked; true = enabled (set 1 to enable interrupt).
    /// Known limitations: In PC mode, this mask register sets the masking that applies to the last task in the running group. Reset value of the full 17-bit field is 0x1ffff.
    /// Related registers: pc_interrupt_clear.ppu_1, pc_interrupt_status/pc_interrupt_raw_status bit 11 (not modeled in this crate).
    pub fn ppu_1(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_MASK_PPU_1, enable)
    }

    /// Description: Enables propagation of the DMA read error interrupt (int_mask bit 12) to the PC's output interrupt lines.
    ///
    /// Bit width: 1
    /// Range of values: false = masked; true = enabled (set 1 to enable interrupt).
    /// Known limitations: In PC mode, this mask register sets the masking that applies to the last task in the running group. Reset value of the full 17-bit field is 0x1ffff.
    /// Related registers: pc_interrupt_clear.dma_read_error, pc_interrupt_status/pc_interrupt_raw_status bit 12 (not modeled in this crate).
    pub fn dma_read_error(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_MASK_DMA_READ_ERROR, enable)
    }

    /// Description: Enables propagation of the DMA write error interrupt (int_mask bit 13) to the PC's output interrupt lines.
    ///
    /// Bit width: 1
    /// Range of values: false = masked; true = enabled (set 1 to enable interrupt).
    /// Known limitations: In PC mode, this mask register sets the masking that applies to the last task in the running group. Reset value of the full 17-bit field is 0x1ffff.
    /// Related registers: pc_interrupt_clear.dma_write_error, pc_interrupt_status/pc_interrupt_raw_status bit 13 (not modeled in this crate).
    pub fn dma_write_error(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_MASK_DMA_WRITE_ERROR, enable)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PCInterruptClear;

impl RegisterMeta for PCInterruptClear {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_PC;
    const OFFSET: u32 = REG_PC_INTERRUPT_CLEAR;
}

impl Register<PCInterruptClear> {
    /// Description: Clears (acknowledges) the latched CNA feature group 0 interrupt (done_clr bit 0).
    ///
    /// Bit width: 1
    /// Range of values: true = clear the pending interrupt for this event; false = no effect.
    /// Known limitations: Write-1-to-clear (W1C); the bit self-clears after the pulse and does not persist as a stored value. Asserted interrupts stay asserted until explicitly cleared here.
    /// Related registers: pc_interrupt_mask.cna_feature_0 (enables/disables propagation of this same event), pc_interrupt_status/pc_interrupt_raw_status bit 0 (not modeled in this crate).
    pub fn cna_feature_0(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_CLEAR_CNA_FEATURE_0, enable)
    }

    /// Description: Clears (acknowledges) the latched CNA feature group 1 interrupt (done_clr bit 1).
    ///
    /// Bit width: 1
    /// Range of values: true = clear the pending interrupt for this event; false = no effect.
    /// Known limitations: Write-1-to-clear (W1C); self-clears after the pulse.
    /// Related registers: pc_interrupt_mask.cna_feature_1, pc_interrupt_status/pc_interrupt_raw_status bit 1 (not modeled in this crate).
    pub fn cna_feature_1(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_CLEAR_CNA_FEATURE_1, enable)
    }

    /// Description: Clears (acknowledges) the latched CNA weight group 0 interrupt (done_clr bit 2).
    ///
    /// Bit width: 1
    /// Range of values: true = clear the pending interrupt for this event; false = no effect.
    /// Known limitations: Write-1-to-clear (W1C); self-clears after the pulse.
    /// Related registers: pc_interrupt_mask.cna_weight_0, pc_interrupt_status/pc_interrupt_raw_status bit 2 (not modeled in this crate).
    pub fn cna_weight_0(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_CLEAR_CNA_WEIGHT_0, enable)
    }

    /// Description: Clears (acknowledges) the latched CNA weight group 1 interrupt (done_clr bit 3).
    ///
    /// Bit width: 1
    /// Range of values: true = clear the pending interrupt for this event; false = no effect.
    /// Known limitations: Write-1-to-clear (W1C); self-clears after the pulse.
    /// Related registers: pc_interrupt_mask.cna_weight_1, pc_interrupt_status/pc_interrupt_raw_status bit 3 (not modeled in this crate).
    pub fn cna_weight_1(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_CLEAR_CNA_WEIGHT_1, enable)
    }

    /// Description: Clears (acknowledges) the latched CNA csc (convert) group 0 interrupt (done_clr bit 4).
    ///
    /// Bit width: 1
    /// Range of values: true = clear the pending interrupt for this event; false = no effect.
    /// Known limitations: Write-1-to-clear (W1C); self-clears after the pulse.
    /// Related registers: pc_interrupt_mask.cna_csc_0, pc_interrupt_status/pc_interrupt_raw_status bit 4 (not modeled in this crate).
    pub fn cna_csc_0(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_CLEAR_CNA_CSC_0, enable)
    }

    /// Description: Clears (acknowledges) the latched CNA csc (convert) group 1 interrupt (done_clr bit 5).
    ///
    /// Bit width: 1
    /// Range of values: true = clear the pending interrupt for this event; false = no effect.
    /// Known limitations: Write-1-to-clear (W1C); self-clears after the pulse.
    /// Related registers: pc_interrupt_mask.cna_csc_1, pc_interrupt_status/pc_interrupt_raw_status bit 5 (not modeled in this crate).
    pub fn cna_csc_1(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_CLEAR_CNA_CSC_1, enable)
    }

    /// Description: Clears (acknowledges) the latched CORE group 0 interrupt (done_clr bit 6).
    ///
    /// Bit width: 1
    /// Range of values: true = clear the pending interrupt for this event; false = no effect.
    /// Known limitations: Write-1-to-clear (W1C); self-clears after the pulse.
    /// Related registers: pc_interrupt_mask.core_0, pc_interrupt_status/pc_interrupt_raw_status bit 6 (not modeled in this crate).
    pub fn core_0(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_CLEAR_CORE_0, enable)
    }

    /// Description: Clears (acknowledges) the latched CORE group 1 interrupt (done_clr bit 7).
    ///
    /// Bit width: 1
    /// Range of values: true = clear the pending interrupt for this event; false = no effect.
    /// Known limitations: Write-1-to-clear (W1C); self-clears after the pulse.
    /// Related registers: pc_interrupt_mask.core_1, pc_interrupt_status/pc_interrupt_raw_status bit 7 (not modeled in this crate).
    pub fn core_1(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_CLEAR_CORE_1, enable)
    }

    /// Description: Clears (acknowledges) the latched DPU group 0 interrupt (done_clr bit 8).
    ///
    /// Bit width: 1
    /// Range of values: true = clear the pending interrupt for this event; false = no effect.
    /// Known limitations: Write-1-to-clear (W1C); self-clears after the pulse.
    /// Related registers: pc_interrupt_mask.dpu_0, pc_interrupt_status/pc_interrupt_raw_status bit 8 (not modeled in this crate).
    pub fn dpu_0(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_CLEAR_DPU_0, enable)
    }

    /// Description: Clears (acknowledges) the latched DPU group 1 interrupt (done_clr bit 9).
    ///
    /// Bit width: 1
    /// Range of values: true = clear the pending interrupt for this event; false = no effect.
    /// Known limitations: Write-1-to-clear (W1C); self-clears after the pulse.
    /// Related registers: pc_interrupt_mask.dpu_1, pc_interrupt_status/pc_interrupt_raw_status bit 9 (not modeled in this crate).
    pub fn dpu_1(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_CLEAR_DPU_1, enable)
    }

    /// Description: Clears (acknowledges) the latched PPU group 0 interrupt (done_clr bit 10).
    ///
    /// Bit width: 1
    /// Range of values: true = clear the pending interrupt for this event; false = no effect.
    /// Known limitations: Write-1-to-clear (W1C); self-clears after the pulse.
    /// Related registers: pc_interrupt_mask.ppu_0, pc_interrupt_status/pc_interrupt_raw_status bit 10 (not modeled in this crate).
    pub fn ppu_0(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_CLEAR_PPU_0, enable)
    }

    /// Description: Clears (acknowledges) the latched PPU group 1 interrupt (done_clr bit 11).
    ///
    /// Bit width: 1
    /// Range of values: true = clear the pending interrupt for this event; false = no effect.
    /// Known limitations: Write-1-to-clear (W1C); self-clears after the pulse.
    /// Related registers: pc_interrupt_mask.ppu_1, pc_interrupt_status/pc_interrupt_raw_status bit 11 (not modeled in this crate).
    pub fn ppu_1(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_CLEAR_PPU_1, enable)
    }

    /// Description: Clears (acknowledges) the latched DMA read error interrupt (done_clr bit 12).
    ///
    /// Bit width: 1
    /// Range of values: true = clear the pending interrupt for this event; false = no effect.
    /// Known limitations: Write-1-to-clear (W1C); self-clears after the pulse.
    /// Related registers: pc_interrupt_mask.dma_read_error, pc_interrupt_status/pc_interrupt_raw_status bit 12 (not modeled in this crate).
    pub fn dma_read_error(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_CLEAR_DMA_READ_ERROR, enable)
    }

    /// Description: Clears (acknowledges) the latched DMA write error interrupt (done_clr bit 13).
    ///
    /// Bit width: 1
    /// Range of values: true = clear the pending interrupt for this event; false = no effect.
    /// Known limitations: Write-1-to-clear (W1C); self-clears after the pulse.
    /// Related registers: pc_interrupt_mask.dma_write_error, pc_interrupt_status/pc_interrupt_raw_status bit 13 (not modeled in this crate).
    pub fn dma_write_error(&mut self, enable: bool) -> &mut Self {
        self.set_flag(PC_INTERRUPT_CLEAR_DMA_WRITE_ERROR, enable)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PCTaskCon;

impl RegisterMeta for PCTaskCon {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_PC;
    const OFFSET: u32 = REG_PC_TASK_CON;
}

impl Register<PCTaskCon> {
    /// Description: Sets the total number of tasks to be executed by this PC operation.
    ///
    /// Bit width: 12
    /// Range of values: 0-4095 (PC task number).
    /// Known limitations: None documented beyond the general task-fetch flow (this value gates how many task-register groups PC will step through before stopping).
    /// Related registers: pc_task_con.task_pp_enable (whether task register groups ping-pong), pc_task_dma_base_addr (per-task DMA base), pc_operation_enable.op_en (starts the run), pc_task_status (reports current task counter).
    pub fn task_number(&mut self, task_number: Bits<12>) -> &mut Self {
        self.set_field(PC_TASK_CON_TASK_NUMBER__MASK, unsafe {
            PC_TASK_CON_TASK_NUMBER(task_number.val())
        })
    }

    /// Description: Enables ping-pong mode for fetching successive tasks' register groups.
    ///
    /// Bit width: 1
    /// Range of values: false (1'd0) = ping-pong off, the second group's register setting is fetched only after the first group's task operation has finished; true (1'd1) = tasks' registers are fetched in ping-pong mode, the second group's register setting is fetched immediately after the first group's register fetching (not execution) is finished.
    /// Known limitations: This only hides register-fetch latency between tasks; it is independent of, but analogous to, the per-block S_POINTER ping-pong mechanism used within CNA/CORE/DPU/PPU (see pointer_pp_en / executer_pp_en in those blocks).
    /// Related registers: task_number, pc_task_dma_base_addr.
    pub fn task_pp_enable(&mut self, enable: bool) -> &mut Self {
        self.set_field(PC_TASK_CON_TASK_PP_EN__MASK, unsafe {
            PC_TASK_CON_TASK_PP_EN(enable as u32)
        })
    }

    /// Description: Clears the counter that tracks the currently-executing task index.
    ///
    /// Bit width: 1
    /// Range of values: true = clear the task counter; false = no effect.
    /// Known limitations: Write-1-to-clear (W1C) style field; the TRM suggests clearing this before a task run is started so the counter begins from a known state.
    /// Related registers: task_number, pc_task_status (holds the live task counter value this clears).
    pub fn count_clear(&mut self, enable: bool) -> &mut Self {
        self.set_field(PC_TASK_CON_TASK_COUNT_CLEAR__MASK, unsafe {
            PC_TASK_CON_TASK_COUNT_CLEAR(enable as u32)
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PCTaskDMABaseAddr;

impl RegisterMeta for PCTaskDMABaseAddr {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_PC;
    const OFFSET: u32 = REG_PC_TASK_DMA_BASE_ADDR;
}

impl Register<PCTaskDMABaseAddr> {
    /// Description: Sets the per-task base address added to each downstream engine's DMA offset to form the final AXI address.
    ///
    /// Bit width: 28
    /// Range of values: Address bits [31:4] (bits [3:0] are reserved/RO in the underlying register).
    /// Known limitations: Distinct from pc_base_address.pc_src_address, which points at the regcmd program itself, not task data. For feature DMA, weight DMA, DPU DMA, and PPU DMA, the address appearing on the AXI bus is this base address plus each engine's own offset address carried in its regcmd entry.
    /// Related registers: pc_base_address.pc_src_address (regcmd fetch address, separate from this), pc_task_con.task_number.
    pub fn base_address(&mut self, address: Bits<28>) -> &mut Self {
        self.set_field(PC_TASK_DMA_BASE_ADDR_DMA_BASE_ADDR__MASK, unsafe {
            PC_TASK_DMA_BASE_ADDR_DMA_BASE_ADDR(address.val())
        })
    }
}
