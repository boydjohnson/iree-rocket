//! Named values shared by multiple register-field builders.
//!
//! The register setters still accept [`Bits`], so callers can
//! use an undocumented or newly discovered encoding when necessary. These
//! enums cover the stable, repeatedly used encodings and convert directly
//! to the matching field width.

use super::Bits;

/// Numeric precision encodings shared by CNA, CORE, DPU, and DPU_RDMA.
///
/// Only encodings that have the same meaning across all four blocks are
/// included. Individual blocks expose additional values, such as TF32 in
/// CNA/CORE or FP32 in DPU, through their underlying `Bits<3>` setters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum DataPrecision {
    Int8 = 0,
    Int16 = 1,
    Fp16 = 2,
    Bf16 = 3,
    Int4 = 6,
}

impl From<DataPrecision> for Bits<3> {
    fn from(precision: DataPrecision) -> Self {
        Self::new(precision as u32)
    }
}

/// AXI burst-length encodings shared by CNA, DPU, and DPU_RDMA.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum BurstLength {
    Four = 3,
    Eight = 7,
    Sixteen = 15,
}

impl From<BurstLength> for Bits<4> {
    fn from(length: BurstLength) -> Self {
        Self::new(length as u32)
    }
}

/// CNA non-aligned ARGB input mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ArgbInputMode {
    OneChannel = 8,
    TwoChannels = 9,
    ThreeChannels = 10,
    FourChannels = 11,
}

impl From<ArgbInputMode> for Bits<4> {
    fn from(mode: ArgbInputMode) -> Self {
        Self::new(mode as u32)
    }
}

/// Destinations selected by `DpuFeatureModeCfg::output_mode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum DpuOutputMode {
    Disabled = 0,
    Ppu = 1,
    ExternalMemory = 2,
    PpuAndExternalMemory = 3,
}

impl From<DpuOutputMode> for Bits<2> {
    fn from(mode: DpuOutputMode) -> Self {
        Self::new(mode as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn named_values_match_wire_encodings() {
        assert_eq!(Bits::<3>::from(DataPrecision::Fp16).val(), 2);
        assert_eq!(Bits::<4>::from(BurstLength::Sixteen).val(), 15);
        assert_eq!(Bits::<4>::from(ArgbInputMode::ThreeChannels).val(), 10);
        assert_eq!(Bits::<2>::from(DpuOutputMode::ExternalMemory).val(), 0b10);
    }
}
