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
// These used to be hand-rolled (0x01/0x02/0x11/0x41/0x81) and did NOT
// match Mesa's own registers.xml `target` enum (PC=0x100, CNA=0x200,
// CORE=0x800, DPU=0x1000, DPU_RDMA=0x2000, PPU=0x4000, PPU_RDMA=0x8000,
// DDMA=0x10000, SDMA=0x20000, GLOBAL=0x40000 -- see
// gitlab.freedesktop.org/mesa/mesa, src/gallium/drivers/rocket/registers.xml).
// dpu_rdma.rs/ppu.rs/ppu_rdma.rs/ddma.rs/sdma.rs already reference the
// bindgen-generated `target_*` constants from that same enum (via
// `rkt_registers.h` -> `registers.rs`'s `include!`) directly and were
// therefore already correct; cna.rs/core.rs/dpu.rs/pc.rs/global.rs used
// these wrong constants instead -- an inconsistency introduced when the
// later files switched to using the generated constants but the earlier
// ones (which is everything rkt-basic.rs actually exercises) were never
// updated to match.
//
// The `| 1` matches what a real compiled regcmd program (decoded out of
// conv.rknn -- see rknpu-spelunking/NOTES.md) actually puts in this field
// for every domain that was checkable: CNA/PC/CORE/DPU/DPU_RDMA/PPU/
// PPU_RDMA all showed up as (target value | 1), never the bare target
// value. Mesa's target enum values are conspicuously all even, which is
// consistent with bit 0 being a separate flag outside the domain
// selector proper; unconfirmed whether mainline's kernel driver actually
// requires it or just ignores it, but matching known-working vendor
// output is the lower-risk choice either way.
//
// DOMAIN_GLOBAL is NOT fixed by this -- target_GLOBAL is 0x40000, which
// doesn't fit in a 16-bit domain field at all (`<< 48` on a value that
// big overflows/truncates in a u64 RegCmd). The old 0x81 was already
// wrong under the corrected scheme too, and we don't have a confirmed
// replacement -- GLOBAL_OPERATION_ENABLE's real wire encoding is still
// an open question (see NOTES.md). Left unchanged, still flagged wrong.
pub const DOMAIN_CNA: u32 = crate::rocket::registers::target_CNA | 1;
pub const DOMAIN_PC: u32 = crate::rocket::registers::target_PC | 1;
pub const DOMAIN_CORE: u32 = crate::rocket::registers::target_CORE | 1;
pub const DOMAIN_DPU: u32 = crate::rocket::registers::target_DPU | 1;
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

impl<R: RegisterMeta> Register<R> {
    pub fn new() -> Self {
        Self {
            val: 0,
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
