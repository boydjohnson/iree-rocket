//! Primitives shared by the register-command builders in this crate.
//!
//! This module used to hold every builder as well. Those moved out to one
//! module per hardware op -- [`crate::rocket::conv`],
//! [`crate::rocket::pooling`], [`crate::rocket::activation`], and
//! [`crate::rocket::elementwise`] -- leaving behind only what multiple
//! builders genuinely share: the zero-register helper and the
//! `PC_OPERATION_ENABLE` kick tail.
//!
//! Nothing here is op-specific. New shared helpers belong here only if more
//! than one op module needs them; anything used by a single op belongs in
//! that op's module instead.

use crate::rocket::builders::{
    RegCmd, Register, RegisterMeta,
    pc::{PCOperationMask, PCRegisterAmounts, PCTrailer},
};

pub(crate) fn zero<R: RegisterMeta>() -> RegCmd {
    Register::<R>::new().build()
}

// PC_OPERATION_ENABLE's bitmask, one bit per engine block, confirmed
// against real rknn-toolkit2-compiled regcmd programs (both conv.rknn and
// the pooling.rknn capture in rknpu-spelunking/NOTES.md's "Decoding a real
// regcmd program for a pooling-only op" section): bit = log2(mesa_target)
// - 9, where mesa_target is Mesa registers.xml's `target` enum value for
// that block (PC=0x100 CNA=0x200 CORE=0x800 DPU=0x1000 DPU_RDMA=0x2000
// PPU=0x4000 PPU_RDMA=0x8000). Bit 1 has no corresponding block (the gap
// between CNA=0x200 and CORE=0x800) and is never set in any real capture.
//
// The pooling.rknn capture fires a PPU-only kick after its separately
// dispatched bypass-conv task. That kick is `KICK_PPU | KICK_PPU_RDMA`
// (0x60), notably with bit 0 clear. This is what revealed that bit 0 is not
// a generic mandatory "go" flag: it enables CNA, just as each other bit
// enables its corresponding block.
pub(crate) const KICK_DPU: PCOperationMask = PCOperationMask::DPU;
pub(crate) const KICK_DPU_RDMA: PCOperationMask = PCOperationMask::DPU_RDMA;
pub(crate) const KICK_PPU: PCOperationMask = PCOperationMask::PPU;
pub(crate) const KICK_PPU_RDMA: PCOperationMask = PCOperationMask::PPU_RDMA;

/// Appends the standard single-task `PC_OPERATION_ENABLE` trailer and pads
/// the command buffer to an even length. Pass exactly the bits for the
/// blocks configured by this task.
pub(crate) fn push_kick(cmds: &mut Vec<RegCmd>, enable_mask: PCOperationMask) {
    cmds.push(PCTrailer::single_task_placeholder());
    cmds.push(zero::<PCRegisterAmounts>());
    cmds.push(PCTrailer::required_marker());
    cmds.push(PCTrailer::operation_enable(enable_mask));

    if !cmds.len().is_multiple_of(2) {
        cmds.push(PCTrailer::alignment_padding());
    }
}
