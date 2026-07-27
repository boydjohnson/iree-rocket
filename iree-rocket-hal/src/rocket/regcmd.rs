//! Primitives shared by every Mesa-derived regcmd builder in this crate.
//!
//! This module used to hold every builder as well. Those moved out to one
//! module per hardware op -- [`crate::rocket::mesa_conv`],
//! [`crate::rocket::pooling`], [`crate::rocket::activation`], and
//! [`crate::rocket::elementwise`] -- leaving behind only what all of them
//! genuinely share: the zero-register helper, the [`Precision`] domain, the
//! `PC_OPERATION_ENABLE` kick tail, and the multi-task PC link patching.
//!
//! Nothing here is op-specific. New shared helpers belong here only if more
//! than one op module needs them; anything used by a single op belongs in
//! that op's module instead.

use crate::rocket::{
    builders::{
        DOMAIN_PC, RegCmd, Register, RegisterMeta,
        pc::{PCOperationMask, PCRegisterAmounts, PCTrailer},
    },
    registers::{REG_PC_BASE_ADDRESS, REG_PC_OPERATION_ENABLE, REG_PC_REGISTER_AMOUNTS},
};

pub(crate) fn zero<R: RegisterMeta>() -> RegCmd {
    Register::<R>::new().build()
}

/// Which numeric domain a conv op computes in. Hardware-validated:
/// `Int8` by essentially every hw test in this repo; `Fp16` by
/// `rkt-fp16.rs`'s round-7 recipe (rknpu-spelunking/NOTES.md's
/// "Attempted real fp16 dispatch on hardware" section through its
/// resolution) -- a real, non-uniform weight/input pair (`0.25 * 10.5`)
/// computing a bit-exact correct product (`2.625`) on real hardware.
///
/// `Fp16` is only wired up for the `input_channels == 1` path in
/// [`crate::rocket::mesa_conv`] -- every fp16 hardware test used that
/// shape, and the multi-channel branch's CVT convention (already its own
/// distinct code path for int8) was never independently re-derived or
/// validated for fp16. `mesa_conv::build_conv_cna_core_dpu_dpu_rdma`
/// asserts this rather than silently emitting untested output for it,
/// matching that module's existing convention for other untested shape
/// combinations (see its module doc comment's "wide atomic" note).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    Int8,
    Fp16,
}

impl Precision {
    /// Bytes per element -- scales the byte-footprint-only fields
    /// (`CNA_WEIGHT_SIZE0/1`) that Mesa's original `fill_task()` formula
    /// assumes are always 1 byte/element. Flagged as an open question in
    /// early fp16 rounds ("not patched or ruled out"), later confirmed
    /// load-bearing once a real, non-uniform weight/input pair exposed
    /// it (a uniformly-filled weight buffer can't distinguish a wrong
    /// byte count from a correct one).
    pub fn bytes_per_element(self) -> u32 {
        match self {
            Precision::Int8 => 1,
            Precision::Fp16 => 2,
        }
    }

    /// The shared 3-bit precision enum `CNA_CONV_CON1`/`CORE_MISC_CFG`
    /// use, which also matches `DPU_DATA_FORMAT`/`DPU_RDMA_FEATURE_
    /// MODE_CFG`'s own wider 6-value enum for every value in use here
    /// (int8=0, fp16=2 in both) -- per TRM chapter36.txt.
    pub(crate) fn enum_value(self) -> u32 {
        match self {
            Precision::Int8 => 0,
            Precision::Fp16 => 2,
        }
    }
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
// conv.rknn's single kick is always `KICK_CNA | KICK_CORE | KICK_DPU |
// KICK_DPU_RDMA` (0x1d). The pooling.rknn capture fires *two* kicks per
// tile: a bypass CNA/CORE/DPU stage (`0x0d`, no DPU_RDMA) followed
// separately by `KICK_PPU | KICK_PPU_RDMA` (0x60) -- notably with bit 0
// (`PC_OPERATION_ENABLE_OP_EN` in rkt_registers.h's naming) clear, which
// is what revealed that bit isn't a generic mandatory "go" flag -- it's
// just CNA's own enable bit, like every other bit here.
pub(crate) const KICK_CNA: PCOperationMask = PCOperationMask::CNA;
pub(crate) const KICK_CORE: PCOperationMask = PCOperationMask::CORE;
pub(crate) const KICK_DPU: PCOperationMask = PCOperationMask::DPU;
pub(crate) const KICK_DPU_RDMA: PCOperationMask = PCOperationMask::DPU_RDMA;
pub(crate) const KICK_PPU: PCOperationMask = PCOperationMask::PPU;
pub(crate) const KICK_PPU_RDMA: PCOperationMask = PCOperationMask::PPU_RDMA;

/// Appends the standard single-task "kick" tail (ported verbatim from
/// Mesa's fill_first_regcmd(), num_tasks == 1 branch -- see rkt-basic.rs's
/// top-of-file doc comment / NOTES.md) and pads to an even length. Shared
/// by every regcmd builder in this crate -- every one of them ends a
/// single task the same way, differing only in which blocks that task's
/// kick should actually enable (see `KICK_*` above -- pass exactly the
/// bits for the blocks this task configured, not a fixed value).
pub(crate) fn push_kick_for_task_count(
    cmds: &mut Vec<RegCmd>,
    enable_mask: PCOperationMask,
    task_count: usize,
) {
    if task_count == 1 {
        cmds.push(PCTrailer::single_task_placeholder());
    } else {
        cmds.push(RegCmd::new(DOMAIN_PC, REG_PC_BASE_ADDRESS, 0));
    }
    cmds.push(zero::<PCRegisterAmounts>());
    cmds.push(PCTrailer::required_marker());
    cmds.push(PCTrailer::operation_enable(enable_mask));

    if !cmds.len().is_multiple_of(2) {
        cmds.push(PCTrailer::alignment_padding());
    }
}

pub(crate) fn push_kick(cmds: &mut Vec<RegCmd>, enable_mask: PCOperationMask) {
    push_kick_for_task_count(cmds, enable_mask, 1);
}

pub(crate) fn task_link_trailer_index(commands: &[RegCmd]) -> Result<usize, &'static str> {
    let is_register = |command: &RegCmd, domain: u32, offset: u32| {
        ((command.0 >> 48) & 0xff) == u64::from(domain & 0xff)
            && (command.0 & 0xffff) == u64::from(offset)
    };
    if commands.len() < 4 {
        return Err("regcmd task is too short to contain a PC link trailer");
    }

    for index in (0..=commands.len() - 4).rev() {
        if is_register(&commands[index], DOMAIN_PC, REG_PC_BASE_ADDRESS)
            && is_register(&commands[index + 1], DOMAIN_PC, REG_PC_REGISTER_AMOUNTS)
            && commands[index + 2].0 == 0x0041_0000_0000_0000
            && is_register(&commands[index + 3], 0x81, REG_PC_OPERATION_ENABLE)
        {
            return Ok(index);
        }
    }
    Err("regcmd task has no PC link trailer")
}

/// Patches Mesa's embedded next-task PC links after command-buffer DMA
/// addresses are known.
///
/// Multi-split operations use both the kernel's ordered task array and the
/// trailing `PC_BASE_ADDRESS`/`PC_REGISTER_AMOUNTS` pair in each regcmd.
/// Mesa patches those registers to the next task before submission so the
/// alternate ping-pong register group is prepared while the current task
/// runs. The final task retains its zero-valued link.
///
/// This path remains experimental: real RK3588 testing with the mainline
/// driver produced only task 0's output when linked tasks were submitted in
/// one DRM job. Production callers should submit and fence tasks separately.
pub fn link_regcmd_tasks(
    tasks: &mut [Vec<RegCmd>],
    task_dma_addresses: &[u32],
) -> Result<(), &'static str> {
    if tasks.len() != task_dma_addresses.len() {
        return Err("regcmd task and DMA-address counts differ");
    }
    if tasks.len() <= 1 {
        return Ok(());
    }

    let trailer_indices = tasks
        .iter()
        .map(|commands| task_link_trailer_index(commands))
        .collect::<Result<Vec<_>, _>>()?;
    for index in 0..tasks.len() - 1 {
        let next_payload_commands = trailer_indices[index + 1];
        let next_register_amount = (next_payload_commands / 2).next_multiple_of(2);
        let next_register_amount = u32::try_from(next_register_amount)
            .map_err(|_| "next regcmd register amount exceeds u32")?;
        let trailer = trailer_indices[index];
        tasks[index][trailer] = RegCmd::new(
            DOMAIN_PC,
            REG_PC_BASE_ADDRESS,
            task_dma_addresses[index + 1],
        );
        tasks[index][trailer + 1] =
            RegCmd::new(DOMAIN_PC, REG_PC_REGISTER_AMOUNTS, next_register_amount);
    }
    Ok(())
}
