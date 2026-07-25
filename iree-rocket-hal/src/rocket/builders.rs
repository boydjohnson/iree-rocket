use std::marker::PhantomData;

pub mod cna;
pub mod core;
pub mod ddma;
pub mod dpu;
pub mod dpu_rdma;
pub mod global;
pub mod pc;
pub mod ppu;
pub mod ppu_rdma;
pub mod sdma;

// Domain IDs for the 64-bit RegCmd packing (bits 48-63).
//
// CNA/PC/CORE/DPU used to be hand-rolled (0x01/0x02/0x11/0x41/0x81) and
// did NOT match Mesa's own registers.xml `target` enum (PC=0x100,
// CNA=0x200, CORE=0x800, DPU=0x1000, DPU_RDMA=0x2000, PPU=0x4000,
// PPU_RDMA=0x8000, DDMA=0x10000, SDMA=0x20000, GLOBAL=0x40000 -- see
// gitlab.freedesktop.org/mesa/mesa, src/gallium/drivers/rocket/registers.xml).
//
// Every domain's `RegisterMeta::DOMAIN` ultimately needs `(target value |
// 1)`, not the bare target value -- confirmed by decoding a real compiled
// regcmd program out of conv.rknn (see rknpu-spelunking/NOTES.md):
// CNA/PC/CORE/DPU/DPU_RDMA/PPU/PPU_RDMA all showed up tagged as
// `target | 1`, never the bare value. Mesa's target enum values are
// conspicuously all even, consistent with bit 0 being a separate flag
// outside the domain selector proper; unconfirmed whether mainline's
// kernel driver actually requires it or just ignores it, but matching
// known-working vendor output is the lower-risk choice either way.
//
// dpu_rdma.rs/ppu.rs/ppu_rdma.rs already referenced the bindgen-generated
// `target_*` constants directly (via `rkt_registers.h` ->
// `registers.rs`'s `include!`), which looked more correct than
// cna.rs/core.rs/dpu.rs/pc.rs's old hand-rolled constants at a glance --
// but they were missing the same `| 1` this file now adds, so they were
// ALSO wrong, just less obviously. `rkt-job.rs` exercises the DPU_RDMA
// path (DpuRdmaSrcBaseAddr etc.) and would have been tagging those writes
// with domain 0x2000 instead of the correct 0x2001.
//
// DOMAIN_GLOBAL is NOT fixable this way -- target_GLOBAL is 0x40000,
// which doesn't fit in a 16-bit domain field at all (`<< 48` on a value
// that big overflows/truncates in a u64 RegCmd). Same problem applies to
// DDMA (0x10000) and SDMA (0x20000) in ddma.rs/sdma.rs, left unchanged
// there too -- neither is used by any of the job binaries currently.
// GLOBAL_OPERATION_ENABLE's real wire encoding remains an open question
// (see NOTES.md); the old 0x81 here is unconfirmed either way, just kept
// since there's nothing better to replace it with yet.
pub const DOMAIN_CNA: u32 = crate::rocket::registers::target_CNA | 1;
pub const DOMAIN_PC: u32 = crate::rocket::registers::target_PC | 1;
pub const DOMAIN_CORE: u32 = crate::rocket::registers::target_CORE | 1;
pub const DOMAIN_DPU: u32 = crate::rocket::registers::target_DPU | 1;
pub const DOMAIN_DPU_RDMA: u32 = crate::rocket::registers::target_DPU_RDMA | 1;
pub const DOMAIN_PPU: u32 = crate::rocket::registers::target_PPU | 1;
pub const DOMAIN_PPU_RDMA: u32 = crate::rocket::registers::target_PPU_RDMA | 1;
pub const DOMAIN_GLOBAL: u32 = 0x81; // UNCONFIRMED -- see comment above

#[derive(Clone, Copy)]
pub struct Register<R> {
    val: u32,
    _marker: PhantomData<R>,
}

pub trait RegisterMeta {
    const DOMAIN: u32;
    const OFFSET: u32;
}

impl<R: RegisterMeta> Default for Register<R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: RegisterMeta> Register<R> {
    pub fn new() -> Self {
        Self {
            val: 0,
            _marker: PhantomData,
        }
    }

    /// Seeds the builder from an already-encoded register value instead of
    /// starting at 0 -- lets a caller round-trip a register a builder
    /// didn't originally construct (e.g. patching one field of an
    /// existing `RegCmd` from `build_conv_regcmd`'s output) through the
    /// same typed setters as a fresh `new()`, preserving every other
    /// field already present in `val`.
    pub fn from_val(val: u32) -> Self {
        Self {
            val,
            _marker: PhantomData,
        }
    }

    pub fn build(self) -> RegCmd {
        RegCmd::new(R::DOMAIN, R::OFFSET, self.val)
    }

    #[inline]
    pub fn set_flag(&mut self, mask: u32, enable: bool) -> &mut Self {
        if enable {
            self.val |= mask;
        } else {
            self.val &= !mask;
        }
        self
    }

    #[inline]
    pub fn set_field(&mut self, mask: u32, encoded: u32) -> &mut Self {
        self.val = (self.val & !mask) | (encoded & mask);
        self
    }
}

pub struct RegCmd(pub u64);

impl RegCmd {
    pub fn new(domain: u32, reg_offset: u32, val: u32) -> Self {
        // Packing format: [Domain:8][Value:32][Offset:16]
        RegCmd(((domain as u64) << 48) | ((val as u64) << 16) | (reg_offset as u64))
    }

    pub fn new_raw(val: u64) -> Self {
        RegCmd(val)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Bits<const N: usize>(u32);

impl<const N: usize> Bits<N> {
    pub const fn new(val: u32) -> Self {
        // Use u64 for check to avoid overflow if N=32
        if N < 32 {
            assert!(
                (val as u64) < (1u64 << N),
                "Value exceeds designated bit width!"
            );
        }
        Bits(val)
    }

    pub const fn val(&self) -> u32 {
        self.0
    }
}
