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

// Domain IDs for the 64-bit RegCmd packing (Bits 48-55)
// These are specific to the Rocket NPU hardware implementation.
pub const DOMAIN_CNA: u32 = 0x01;
pub const DOMAIN_PC: u32 = 0x02;
pub const DOMAIN_CORE: u32 = 0x11;
pub const DOMAIN_DPU: u32 = 0x41;
pub const DOMAIN_GLOBAL: u32 = 0x81;

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

impl<const N: usize> Bits<const N: usize> {
    pub const fn new(val: u32) -> Self {
        // Use u64 for check to avoid overflow if N=32
        if N < 32 {
            assert!((val as u64) < (1u64 << N), "Value exceeds designated bit width!");
        }
        Bits(val)
    }

    pub const fn val(&self) -> u32 {
        self.0
    }
}
