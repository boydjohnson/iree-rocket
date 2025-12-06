use std::marker::PhantomData;

pub mod cna;
pub mod pc;

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

    /// Set/clear a single-bit (or multi-bit) flag given a mask.
    #[inline]
    pub fn set_flag(&mut self, mask: u32, enable: bool) -> &mut Self {
        if enable {
            self.val |= mask;
        } else {
            self.val &= !mask;
        }
        self
    }

    /// Set a field: clear `mask`, then OR in the already-encoded value.
    ///
    /// `encoded` should already be shifted & masked (e.g. FIELD(val)).
    #[inline]
    pub fn set_field(&mut self, mask: u32, encoded: u32) -> &mut Self {
        self.val = (self.val & !mask) | (encoded & mask);
        self
    }
}

pub struct RegCmd(pub(crate) u64);

impl RegCmd {
    // Helper to pack (Domain ID | Value | Offset)
    pub fn new(domain: u32, reg_offset: u32, val: u32) -> Self {
        RegCmd(((domain as u64) << 48) | ((val as u64) << 16) | (reg_offset as u64))
    }

    pub fn new_raw(val: u64) -> Self {
        RegCmd(val)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Bits<const N: usize>(u32);

impl<const N: usize> Bits<N> {
    /// Create a new value.
    /// This asserts bounds at COMPILE TIME for constants, and RUNTIME for variables.
    pub const fn new(val: u32) -> Self {
        // Use u64 for check to avoid overflow if N=32
        assert!(
            (val as u64) < (1u64 << N),
            "Value exceeds designated bit width!"
        );
        Bits(val)
    }

    /// Extract the raw u32
    pub const fn val(&self) -> u32 {
        self.0
    }
}
