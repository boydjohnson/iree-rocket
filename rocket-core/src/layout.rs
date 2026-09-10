//! The feature-cube layout contract: the geometry of an `NC1HWC2` cube as a
//! pure function of what the compiler knows, and the identity under which
//! one dispatch may read another's cube in place.
//!
//! COMPILER_ROADMAP.md section 6.1. Until 2026-09-10 this was computed in
//! five places in `rocket-hal-driver/src/command_buffer.rs` -- once per
//! dispatch kind, on the way in and on the way out -- and compared in a
//! sixth (`chainable_cube`). None of it depended on anything the runtime
//! knows that the compiler does not, so it moves here, where the plugin can
//! ask the same question through `rocket-plan-ffi` and the driver checks
//! rather than rediscovers. Every number below is the driver's, unchanged;
//! the tests pin them and the driver's own `chain_identity_tests` pin the
//! verdicts against the bytes.
//!
//! # The cube
//!
//! A feature map lives in memory as `C1` surfaces of `H x W` pixels, each
//! pixel one 16-byte atom ([`FEATURE_ATOM_BYTES`]) holding `C2` channels,
//! `C2 = 16 / element bytes`. Surfaces are `surface_pixel_count * 16` bytes
//! apart. Everything a producer and a consumer must agree on is therefore
//! four numbers, [`CubeGeometry`], and the two rules that are not obvious
//! from the element width are:
//!
//! - **Channel padding is a property of the reader, and it differs by
//!   unit.** The CNA (convolution, matmul, element-wise) pads the channel
//!   count to 16 *lanes* whatever the element width, so at fp16 that is two
//!   atoms; the PPU (pooling) programs whole atoms only, 8 lanes at fp16
//!   and 16 at int8. [`packed_channels`] states both.
//! - **The PPU strides its surfaces by the pixel count rounded up to
//!   four.** The vendor's `7x5` controls program 36 for an area of 35, and a
//!   `7x7` pool output lives at stride 52 while a convolution of the same 49
//!   pixels repacks at 49. [`cube_geometry`] applies it to
//!   [`CubeKind::Pooling`] alone.
//!
//! # The identity
//!
//! A consumer reads a producer's cube in place when the repack it would
//! otherwise perform reproduces that cube byte for byte:
//! `pack(compact(cube)) == cube`. [`chain_identity`] is that claim, and it
//! refuses in exactly the order `chainable_cube` did: a consumer that pads
//! its channels or whose pixel is a partial atom can never chain (the repack
//! zeroes those lanes and the producer left its own padding channels
//! there), and otherwise the pixel counts, surface strides and logical pixel
//! widths must be equal. A producer's *padding* surfaces past the logical
//! width do not enter into it: they lie beyond everything the consumer
//! reads.
//!
//! What this module does not decide is whether a producer *offers* a cube
//! at all -- a fanned-out dispatch has no single scratch, an int8
//! accumulator writes 128-byte blocks -- or whether a consumer's input is
//! in the surface layout to begin with (a `Cin <= 4` convolution reads
//! dense ARGB). Those are [`crate::conv::Shape`] facts and the driver's,
//! and they stay where they are.

use crate::{
    conv::FEATURE_ATOM_BYTES,
    error::{PlanError, PlanErrorCode},
};

/// Which unit reads or writes the cube, which is what picks the channel
/// padding and the surface stride.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CubeKind {
    /// A convolution's input feature map or output.
    Conv,
    /// A matmul's `[M, K]` operand or `[M, N]` result: width `M`, height 1.
    Matmul,
    /// A pooling input or output, through the PPU.
    Pooling,
    /// An element-wise or LUT operand or result.
    Elementwise,
}

impl CubeKind {
    fn is_ppu(self) -> bool {
        matches!(self, CubeKind::Pooling)
    }
}

/// The four numbers a producer and a consumer must agree on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CubeGeometry {
    /// Logical pixels, `width * height`.
    pub pixel_count: usize,
    /// Pixels one surface is strided by: [`CubeGeometry::pixel_count`]
    /// rounded up to four for the PPU, equal to it everywhere else.
    pub surface_pixel_count: usize,
    /// Bytes one logical pixel occupies: channels times element bytes.
    pub bytes_per_pixel: usize,
    /// Bytes the reader would pad the pixel to: [`packed_channels`] times
    /// element bytes. Equal to [`CubeGeometry::bytes_per_pixel`] only when
    /// the channel count already fills the reader's padding unit.
    pub packed_bytes_per_pixel: usize,
}

impl CubeGeometry {
    /// Whether the logical pixel is a whole number of atoms, so that no
    /// repack would zero a partial trailing atom.
    pub fn is_whole_atom(&self) -> bool {
        self.bytes_per_pixel != 0
            && self
                .bytes_per_pixel
                .is_multiple_of(FEATURE_ATOM_BYTES as usize)
    }

    /// Whether the cube is exactly the reader's own geometry: whole atoms
    /// and no padding surfaces beyond the logical channels. This is the
    /// consumer-side precondition of [`chain_identity`], and what a PPU or
    /// element-wise producer requires before offering its cube.
    pub fn is_exact(&self) -> bool {
        self.is_whole_atom() && self.packed_bytes_per_pixel == self.bytes_per_pixel
    }

    /// Surfaces the reader packs, one per padded atom.
    pub fn packed_surfaces(&self) -> usize {
        self.packed_bytes_per_pixel
            .div_ceil(FEATURE_ATOM_BYTES as usize)
    }

    /// Bytes the packed cube occupies: every padded surface at the surface
    /// stride. The driver's `nc1hwc2_storage_size(surface_pixel_count,
    /// packed_bytes_per_pixel)`.
    pub fn storage_bytes(&self) -> Result<usize, PlanError> {
        self.surface_pixel_count
            .checked_mul(self.packed_surfaces())
            .and_then(|value| value.checked_mul(FEATURE_ATOM_BYTES as usize))
            .ok_or_else(|| {
                PlanError::new(
                    PlanErrorCode::HardwareLimit,
                    "feature cube storage size overflows usize",
                )
            })
    }

    /// Bytes the dense NHWC tensor occupies: logical pixels at the logical
    /// width, which is what a consumer's binding must hold.
    pub fn dense_bytes(&self) -> Result<usize, PlanError> {
        self.pixel_count
            .checked_mul(self.bytes_per_pixel)
            .ok_or_else(|| {
                PlanError::new(
                    PlanErrorCode::HardwareLimit,
                    "dense feature map size overflows usize",
                )
            })
    }

    /// The consumer-side half of [`chain_identity`]: whether this geometry
    /// could read *any* producer's cube in place. Checked before a producer
    /// is looked for, so a decline names the consumer rather than a
    /// producer it would have refused anyway.
    pub fn can_chain(&self) -> Result<(), ChainRefusal> {
        if !self.is_whole_atom() {
            return Err(ChainRefusal::PartialAtom {
                bytes_per_pixel: self.bytes_per_pixel,
            });
        }
        if self.packed_bytes_per_pixel != self.bytes_per_pixel {
            return Err(ChainRefusal::ConsumerPadsChannels {
                bytes_per_pixel: self.bytes_per_pixel,
                packed_bytes_per_pixel: self.packed_bytes_per_pixel,
            });
        }
        Ok(())
    }
}

/// Why a consumer cannot read a producer's cube in place. Each variant is
/// one way `pack(compact(cube)) == cube` stops holding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChainRefusal {
    /// The consumer's pixel is not a whole number of atoms: its repack
    /// zeroes the partial trailing atom, where the producer left its
    /// padding channels.
    PartialAtom { bytes_per_pixel: usize },
    /// The consumer pads its channels past the logical width: the padding
    /// surfaces are zero after a repack and producer bytes here.
    ConsumerPadsChannels {
        bytes_per_pixel: usize,
        packed_bytes_per_pixel: usize,
    },
    /// Surfaces are `pixel_count * 16` bytes apart, so unequal counts put
    /// surface 1 somewhere the consumer does not look.
    PixelCount { producer: usize, consumer: usize },
    /// The PPU's four-rounded stride against everyone else's exact one.
    SurfaceStride { producer: usize, consumer: usize },
    /// The producer wrote a different logical channel width.
    PixelWidth { producer: usize, consumer: usize },
}

impl std::fmt::Display for ChainRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChainRefusal::PartialAtom { bytes_per_pixel } => write!(
                f,
                "consumer width {bytes_per_pixel} is not a whole number of 16-byte atoms"
            ),
            ChainRefusal::ConsumerPadsChannels {
                bytes_per_pixel,
                packed_bytes_per_pixel,
            } => write!(
                f,
                "consumer width {bytes_per_pixel} packs to {packed_bytes_per_pixel}, \
                 not a whole-atom identity"
            ),
            ChainRefusal::PixelCount { producer, consumer } => write!(
                f,
                "producer wrote {producer} pixels, consumer reads {consumer}"
            ),
            ChainRefusal::SurfaceStride { producer, consumer } => write!(
                f,
                "producer surfaces {producer} pixels apart, consumer packs at {consumer}"
            ),
            ChainRefusal::PixelWidth { producer, consumer } => write!(
                f,
                "producer pixel is {producer} bytes, consumer pixel is {consumer}"
            ),
        }
    }
}

impl std::error::Error for ChainRefusal {}

/// Channels the reader pads a pixel to.
///
/// The CNA-fed kinds pad to 16 lanes whatever the element width -- so an
/// 8-channel fp16 pixel packs to two atoms -- while the PPU programs whole
/// atoms only: `16 / element_bytes` lanes, 8 at fp16 and 16 at int8. Both
/// rules were previously spelled in the driver (`shape.in_channels.max(16)
/// .next_multiple_of(16)` at every CNA site, `PoolingShape::
/// programmed_channels` for the PPU); the driver now reads them from here.
///
/// `None` when the padded count does not fit `u32`, which no real tensor
/// reaches; a caller that cannot carry the `Option` may treat it as a
/// malformed shape.
pub fn packed_channels(kind: CubeKind, element_bytes: u32, channels: u32) -> Option<u32> {
    if kind.is_ppu() {
        channels.checked_next_multiple_of(lanes_per_atom(element_bytes))
    } else {
        channels.max(16).checked_next_multiple_of(16)
    }
}

fn lanes_per_atom(element_bytes: u32) -> u32 {
    FEATURE_ATOM_BYTES / element_bytes.max(1)
}

/// The geometry of a `width x height x channels` feature map at
/// `element_bytes` per element, as `kind` reads or writes it.
///
/// `element_bytes` rather than a precision rung because the rung's only
/// contribution is the element width, and it differs by side: a
/// convolution's input is [`crate::conv::Precision::element_bytes`] wide
/// and its output [`crate::conv::Precision::output_element_bytes`], which
/// is what makes an fp32-result rung a 4-lane cube on the way out and an
/// 8-lane one on the way in. Sub-byte elements (int4) have no cube here;
/// the caller must not ask.
///
/// Refuses a zero extent or channel count, an element width that does not
/// divide the atom, and any product that overflows.
pub fn cube_geometry(
    kind: CubeKind,
    element_bytes: u32,
    width: u32,
    height: u32,
    channels: u32,
) -> Result<CubeGeometry, PlanError> {
    if element_bytes == 0 || !(FEATURE_ATOM_BYTES).is_multiple_of(element_bytes) {
        return Err(PlanError::new(
            PlanErrorCode::InvalidShape,
            format!("element width {element_bytes} bytes does not divide a 16-byte feature atom"),
        ));
    }
    if width == 0 || height == 0 || channels == 0 {
        return Err(PlanError::new(
            PlanErrorCode::InvalidShape,
            format!("feature map {width}x{height}x{channels} has a zero extent"),
        ));
    }
    let overflow = || {
        PlanError::new(
            PlanErrorCode::HardwareLimit,
            format!("feature map {width}x{height}x{channels} overflows usize"),
        )
    };
    let pixel_count = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(overflow)?;
    let surface_pixel_count = if kind.is_ppu() {
        pixel_count
            .checked_next_multiple_of(4)
            .ok_or_else(overflow)?
    } else {
        pixel_count
    };
    let bytes_per_pixel = (channels as usize)
        .checked_mul(element_bytes as usize)
        .ok_or_else(overflow)?;
    let packed_bytes_per_pixel = packed_channels(kind, element_bytes, channels)
        .and_then(|lanes| (lanes as usize).checked_mul(element_bytes as usize))
        .ok_or_else(overflow)?;
    Ok(CubeGeometry {
        pixel_count,
        surface_pixel_count,
        bytes_per_pixel,
        packed_bytes_per_pixel,
    })
}

/// Whether `consumer` may read `producer`'s cube in place of repacking its
/// own: `pack(compact(producer)) == producer` under the consumer's
/// geometry. See the module comment for what each refusal means.
///
/// The producer's `packed_bytes_per_pixel` is not consulted: its padding
/// surfaces, if any, lie past the `pixel_count * bytes_per_pixel` region
/// the consumer reads.
pub fn chain_identity(
    producer: &CubeGeometry,
    consumer: &CubeGeometry,
) -> Result<(), ChainRefusal> {
    consumer.can_chain()?;
    if producer.pixel_count != consumer.pixel_count {
        return Err(ChainRefusal::PixelCount {
            producer: producer.pixel_count,
            consumer: consumer.pixel_count,
        });
    }
    if producer.surface_pixel_count != consumer.surface_pixel_count {
        return Err(ChainRefusal::SurfaceStride {
            producer: producer.surface_pixel_count,
            consumer: consumer.surface_pixel_count,
        });
    }
    if producer.bytes_per_pixel != consumer.bytes_per_pixel {
        return Err(ChainRefusal::PixelWidth {
            producer: producer.bytes_per_pixel,
            consumer: consumer.bytes_per_pixel,
        });
    }
    Ok(())
}

/// What the compiler declared about one dispatch's layout, as the trailing
/// `u32` push constant carries it (`Conv2DDef.runtime_layout` and its
/// twins). COMPILER_ROADMAP.md 6.2, first form.
///
/// The word is `packed_inputs | (packed_readers << 16)`, so zero -- what
/// every shim passes as a literal, and what an executable built without
/// `rocket-assign-layout` sees -- is "every input dense, always write the
/// dense output": exactly the runtime's behaviour before any layout was
/// declared. `packed_inputs` has bit `i` set when input binding `i` reads
/// its producer's cube in place ([`chain_identity`] held at compile time);
/// `packed_readers` is how many Rocket dispatches read this dispatch's
/// result that way, or zero when any other reader exists. The driver
/// attempts a chain only on a declared-packed input, and skips the dense
/// output write only when exactly `packed_readers` consumers chained on the
/// same command buffer -- a reader on a later command buffer leaves the
/// tally short and the write happens, which is what "first form" means.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DispatchLayout {
    pub packed_inputs: u16,
    pub packed_readers: u16,
}

impl DispatchLayout {
    pub const READERS_SHIFT: u32 = 16;

    pub fn from_word(word: u32) -> DispatchLayout {
        DispatchLayout {
            packed_inputs: (word & 0xFFFF) as u16,
            packed_readers: (word >> Self::READERS_SHIFT) as u16,
        }
    }

    pub fn to_word(self) -> u32 {
        u32::from(self.packed_inputs) | (u32::from(self.packed_readers) << Self::READERS_SHIFT)
    }

    /// Whether input binding `index` was declared to read its producer's
    /// cube in place.
    pub fn input_packed(self, index: u32) -> bool {
        index < 16 && self.packed_inputs & (1 << index) != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_layout_word_round_trips_and_zero_is_all_dense() {
        let zero = DispatchLayout::from_word(0);
        assert_eq!(zero, DispatchLayout::default());
        assert_eq!(zero.packed_readers, 0);
        assert!(!zero.input_packed(0));
        let layout = DispatchLayout {
            packed_inputs: 0b1001,
            packed_readers: 3,
        };
        assert_eq!(layout.to_word(), 0x0003_0009);
        assert_eq!(DispatchLayout::from_word(0x0003_0009), layout);
        assert!(layout.input_packed(0));
        assert!(!layout.input_packed(1));
        assert!(layout.input_packed(3));
        assert!(!layout.input_packed(16));
        // An older executable's plain reader count decodes as a count with
        // no packed input, which is what it meant.
        assert_eq!(
            DispatchLayout::from_word(2 << DispatchLayout::READERS_SHIFT),
            DispatchLayout {
                packed_inputs: 0,
                packed_readers: 2,
            }
        );
    }

    fn conv(elem: u32, w: u32, h: u32, c: u32) -> CubeGeometry {
        cube_geometry(CubeKind::Conv, elem, w, h, c).unwrap()
    }

    fn pool(elem: u32, w: u32, h: u32, c: u32) -> CubeGeometry {
        cube_geometry(CubeKind::Pooling, elem, w, h, c).unwrap()
    }

    #[test]
    fn the_cna_pads_to_sixteen_lanes_whatever_the_width() {
        // The driver's `in_channels.max(16).next_multiple_of(16)`.
        for kind in [CubeKind::Conv, CubeKind::Matmul, CubeKind::Elementwise] {
            assert_eq!(packed_channels(kind, 2, 1), Some(16));
            assert_eq!(packed_channels(kind, 2, 8), Some(16));
            assert_eq!(packed_channels(kind, 2, 16), Some(16));
            assert_eq!(packed_channels(kind, 2, 17), Some(32));
            assert_eq!(packed_channels(kind, 1, 20), Some(32));
            assert_eq!(packed_channels(kind, 4, 4), Some(16));
        }
        // So an 8-channel fp16 pixel is one whole atom logically and two
        // packed -- the case `a_consumer_that_pads_its_channels_breaks_the
        // _identity` pins at the byte level.
        let g = conv(2, 4, 4, 8);
        assert_eq!(g.bytes_per_pixel, 16);
        assert_eq!(g.packed_bytes_per_pixel, 32);
        assert!(g.is_whole_atom());
        assert!(!g.is_exact());
    }

    #[test]
    fn the_ppu_pads_to_one_atom() {
        // `PoolingShape::programmed_channels`: fp16 C1..C8 are programmed
        // C8, int8 C12 is programmed C16.
        assert_eq!(packed_channels(CubeKind::Pooling, 2, 1), Some(8));
        assert_eq!(packed_channels(CubeKind::Pooling, 2, 8), Some(8));
        assert_eq!(packed_channels(CubeKind::Pooling, 2, 9), Some(16));
        assert_eq!(packed_channels(CubeKind::Pooling, 1, 12), Some(16));
        assert_eq!(packed_channels(CubeKind::Pooling, 1, 16), Some(16));
        assert_eq!(packed_channels(CubeKind::Pooling, 1, 17), Some(32));
        assert!(pool(2, 4, 4, 8).is_exact());
        assert!(pool(2, 4, 4, 24).is_exact());
        // The same 24 channels are two atoms at the CNA, which pads them to
        // 32: a conv output at Cout 24 can feed a pool in place but not a
        // conv -- MobileNetV2's declined edges.
        assert!(!conv(2, 4, 4, 24).is_exact());
        assert!(chain_identity(&conv(2, 4, 4, 24), &pool(2, 4, 4, 24)).is_ok());
        assert_eq!(
            chain_identity(&conv(2, 4, 4, 24), &conv(2, 4, 4, 24)),
            Err(ChainRefusal::ConsumerPadsChannels {
                bytes_per_pixel: 48,
                packed_bytes_per_pixel: 64,
            })
        );
    }

    #[test]
    fn the_ppu_strides_surfaces_by_the_count_rounded_to_four() {
        assert_eq!(pool(2, 7, 7, 64).surface_pixel_count, 52);
        assert_eq!(pool(2, 7, 5, 64).surface_pixel_count, 36);
        assert_eq!(pool(2, 8, 8, 64).surface_pixel_count, 64);
        assert_eq!(conv(2, 7, 7, 64).surface_pixel_count, 49);
        // A 7x7 pool output does not feed a 49-pixel conv in place; an 8x8
        // one does.
        assert_eq!(
            chain_identity(&pool(2, 7, 7, 64), &conv(2, 7, 7, 64)),
            Err(ChainRefusal::SurfaceStride {
                producer: 52,
                consumer: 49,
            })
        );
        assert!(chain_identity(&pool(2, 8, 8, 64), &conv(2, 8, 8, 64)).is_ok());
        // And the stride, not the pixel count, is what the storage is
        // sized by: the padded surfaces exist in memory.
        assert_eq!(pool(2, 7, 7, 64).storage_bytes().unwrap(), 52 * 8 * 16);
        assert_eq!(conv(2, 7, 7, 64).storage_bytes().unwrap(), 49 * 8 * 16);
    }

    #[test]
    fn a_whole_atom_cube_chains_and_each_mismatch_is_named() {
        // ResNet50's conv1 -> conv2 edge.
        let g = conv(2, 56, 56, 64);
        assert!(chain_identity(&g, &g).is_ok());
        assert_eq!(
            chain_identity(&conv(2, 56, 56, 64), &conv(2, 56, 56, 128)),
            Err(ChainRefusal::PixelWidth {
                producer: 128,
                consumer: 256,
            })
        );
        assert_eq!(
            chain_identity(&conv(2, 8, 1, 32), &conv(2, 4, 1, 32)),
            Err(ChainRefusal::PixelCount {
                producer: 8,
                consumer: 4,
            })
        );
        // Cin 20 fp16 is 40 bytes: two atoms and half of a third.
        assert_eq!(
            chain_identity(&conv(2, 4, 4, 20), &conv(2, 4, 4, 20)),
            Err(ChainRefusal::PartialAtom {
                bytes_per_pixel: 40
            })
        );
    }

    #[test]
    fn the_producers_padding_is_not_consulted() {
        // A producer that itself pads (its own kind's rule) still chains
        // into an exact consumer of its logical width: the padding surfaces
        // lie past what the consumer reads.
        let producer = CubeGeometry {
            packed_bytes_per_pixel: 64 + 2 * 16,
            ..conv(2, 8, 8, 32)
        };
        assert!(chain_identity(&producer, &conv(2, 8, 8, 32)).is_ok());
    }

    #[test]
    fn a_matmul_operand_is_a_width_m_height_one_cube() {
        let g = cube_geometry(CubeKind::Matmul, 2, 197, 1, 768).unwrap();
        assert_eq!(g.pixel_count, 197);
        assert_eq!(g.surface_pixel_count, 197);
        assert_eq!(g.bytes_per_pixel, 1536);
        assert!(g.is_exact());
        assert!(chain_identity(&conv(2, 197, 1, 768), &g).is_ok());
    }

    #[test]
    fn the_output_side_of_an_fp32_result_rung_is_a_four_lane_cube() {
        // fp16 in, fp32 out: 8 lanes per atom in, 4 out, same pixel count.
        let out = conv(4, 8, 8, 4);
        assert_eq!(out.bytes_per_pixel, 16);
        assert_eq!(out.packed_bytes_per_pixel, 64);
        assert!(out.is_whole_atom());
    }

    #[test]
    fn refusals() {
        let err = |kind, e, w, h, c| cube_geometry(kind, e, w, h, c).unwrap_err().code();
        assert_eq!(err(CubeKind::Conv, 0, 1, 1, 1), PlanErrorCode::InvalidShape);
        assert_eq!(err(CubeKind::Conv, 3, 1, 1, 1), PlanErrorCode::InvalidShape);
        assert_eq!(err(CubeKind::Conv, 2, 0, 1, 1), PlanErrorCode::InvalidShape);
        assert_eq!(err(CubeKind::Conv, 2, 1, 1, 0), PlanErrorCode::InvalidShape);
        // The padded channel count is the first thing to leave u32.
        assert_eq!(packed_channels(CubeKind::Conv, 2, u32::MAX), None);
        assert_eq!(
            err(CubeKind::Conv, 2, 1, 1, u32::MAX),
            PlanErrorCode::HardwareLimit
        );
        // Extents alone fit usize; the storage product does not.
        let huge = cube_geometry(CubeKind::Conv, 2, u32::MAX, u32::MAX, 4096).unwrap();
        assert_eq!(
            huge.storage_bytes().unwrap_err().code(),
            PlanErrorCode::HardwareLimit
        );
        assert_eq!(
            format!(
                "{}",
                ChainRefusal::PartialAtom {
                    bytes_per_pixel: 40
                }
            ),
            "consumer width 40 is not a whole number of 16-byte atoms"
        );
    }
}
