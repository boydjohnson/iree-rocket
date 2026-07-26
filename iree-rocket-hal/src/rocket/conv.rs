//! Vendor-derived convolution register program.
//!
//! This began as a bit-exact reproduction of group 1 (the complete
//! single-core alternative) from the vendor-compiled `32x32x3 -> 32x32x8`
//! fp16 convolution captures, which differ only in kernel geometry: 1x1 with
//! no padding versus 3x3 with SAME padding. [`conv_2d`] still reproduces that
//! program byte for byte, and the hash tests below pin it.
//!
//! It now generalises to arbitrary [`Shape`] and to output-row [`Tile`]s.
//! Every shape- and tile-dependent register formula is derived from vendor
//! captures rather than assumed: the tile formulas from a cross-group diff of
//! the six captured plans, and the width and height formulas from a sweep of
//! 35 captures (212 convolution programs) spanning widths 32..256 and heights
//! 32..256. Registers that vary in no capture stay literal constants.
//!
//! Supported case: `Cin` 1..=80, `Cout=8`, strides 1..4, 1x1 or 3x3 kernels.
//!
//! Input channel count picks the memory layout. While a pixel fits in half a
//! feature atom -- `Cin` up to 4 -- the vendor keeps it dense NHWC and the
//! CNA pads internally. From `Cin` 5 the map becomes NC1HWC2 surfaces, and
//! the row strides, the CBUF bank split and `data_entries` all change with
//! it; `data_entries` in particular stops depending on the tile height
//! entirely. `FeatureLayout` names the two regimes.
//!
//! Channel padding is a table rather than arithmetic. It is `atoms * 8`
//! except at three atoms, where `datain_channel` stays 24 but coefficients
//! use 32, and at seven, where both round to 64. Atom counts 5, 6, 9 and 10
//! are unpadded, so no arithmetic rule fits and none is invented.
//!
//! `Cout > 8` is a separate axis: it needs the fp16 kernel-group split,
//! which no formula here covers.
//!
//! Keeping this separate from `rocket::regcmd` keeps a capture-derived path
//! distinct from that module's Mesa-derived one.
//!
//! # Output-row tiles
//!
//! [`conv_2d_tile`] emits the same single-core program over an arbitrary
//! range of output rows. Every tile-dependent register value is derived
//! from a cross-group diff of the six captured programs, which cover three
//! alternative height splits (32 / 16+16 / 11+11+10) at two kernel sizes --
//! twelve independent observations per register. Sixteen registers vary
//! across those captures and take derived values here; the other 109 are
//! literal constants reproduced from the capture. The derived values are
//! checked against every observation by the tile register test below.
//!
//! A tile program is **not** byte-identical to captured groups 2-6. Those
//! are the vendor's own multi-core plans and carry PPU/PPU_RDMA blocks whose
//! role has not been derived, plus a plan index in two documented-reserved
//! fields. A tile emitted here is a standalone single-core program covering
//! a row range, intended for submission as one of several independent jobs.
//! Its geometry registers match the captures exactly; its command sequence
//! is group 1's.

use crate::rocket::builders::{
    Bits, RegCmd, Register, RegisterMeta,
    cna::{
        CnaCbufCon0, CnaCbufCon1, CnaConvCon1, CnaConvCon2, CnaConvCon3, CnaCvtCon0, CnaCvtCon1,
        CnaCvtCon2, CnaCvtCon3, CnaCvtCon4, CnaCvtCon5, CnaDataSize0, CnaDataSize1, CnaDataSize2,
        CnaDataSize3, CnaDcompAddr0, CnaDcompAmount0, CnaDcompAmount1, CnaDcompAmount2,
        CnaDcompAmount3, CnaDcompAmount4, CnaDcompAmount5, CnaDcompAmount6, CnaDcompAmount7,
        CnaDcompAmount8, CnaDcompAmount9, CnaDcompAmount10, CnaDcompAmount11, CnaDcompAmount12,
        CnaDcompAmount13, CnaDcompAmount14, CnaDcompAmount15, CnaDcompCtrl, CnaDcompRegnum,
        CnaDmaCon0, CnaDmaCon1, CnaDmaCon2, CnaFcCon0, CnaFcCon1, CnaFcCon2, CnaFcDataSize0,
        CnaFcDataSize1, CnaFeatureDataAddr, CnaPadCon0, CnaPadCon1, CnaWeightSize0, CnaWeightSize1,
        CnaWeightSize2,
    },
    core::{CoreClipTruncate, CoreDataoutSize0, CoreDataoutSize1, CoreMiscCfg, CoreReserved3030},
    dpu::{
        DpuBnAluCfg, DpuBnCfg, DpuBnMulCfg, DpuBnReluxCmpValue, DpuBsAluCfg, DpuBsCfg, DpuBsMulCfg,
        DpuBsOwCfg, DpuBsOwOp, DpuBsReluxCmpValue, DpuDataCubeChannel, DpuDataCubeHeight,
        DpuDataCubeNotchAddr, DpuDataCubeWidth, DpuDataFormat, DpuDstBaseAddr, DpuDstSurfStride,
        DpuEwCfg, DpuEwCvtOffsetValue, DpuEwCvtScaleValue, DpuEwOpValue0, DpuEwOpValue1,
        DpuEwOpValue2, DpuEwOpValue3, DpuEwOpValue4, DpuEwOpValue5, DpuEwOpValue6, DpuEwOpValue7,
        DpuEwReluxCmpValue, DpuFeatureModeCfg, DpuLutAccessCfg, DpuLutAccessData, DpuLutCfg,
        DpuLutInfo, DpuLutLeEnd, DpuLutLeSlopeScale, DpuLutLeSlopeShift, DpuLutLeStart,
        DpuLutLoEnd, DpuLutLoSlopeScale, DpuLutLoSlopeShift, DpuLutLoStart, DpuOffsetPend,
        DpuOutCvtOffset, DpuOutCvtScale, DpuOutCvtShift, DpuReserved40c4, DpuSPointer,
        DpuSurfaceAdd, DpuWdmaSize0, DpuWdmaSize1,
    },
    dpu_rdma::{
        DpuRdmaBnBaseAddr, DpuRdmaBrdmaCfg, DpuRdmaBsBaseAddr, DpuRdmaDataCubeChannel,
        DpuRdmaDataCubeHeight, DpuRdmaDataCubeWidth, DpuRdmaErdmaCfg, DpuRdmaEwBaseAddr,
        DpuRdmaEwSurfNotch, DpuRdmaEwSurfStride, DpuRdmaFeatureModeCfg, DpuRdmaNrdmaCfg,
        DpuRdmaPadCfg, DpuRdmaSPointer, DpuRdmaSrcBaseAddr, DpuRdmaSrcDmaCfg, DpuRdmaSurfNotch,
        DpuRdmaWeight,
    },
    pc::{PCOperationMask, PCRegisterAmounts, PCTrailer},
    values::{ArgbInputMode, BurstLength, DataPrecision, DpuOutputMode},
};

/// `[kernel_height, kernel_width]`.
pub type Kernels = [usize; 2];

/// Height of the originally captured image, in rows.
pub const IMAGE_HEIGHT: u32 = 32;

/// Width of the originally captured image, in pixels.
pub const IMAGE_WIDTH: u32 = 32;

/// Default input channels: the C3 dense NHWC case of the original captures.
pub const INPUT_CHANNELS: u32 = 3;

/// Largest input channel count with capture backing.
pub const MAX_INPUT_CHANNELS: u32 = 80;

/// Widest input pixel the vendor keeps in dense NHWC, in bytes.
///
/// A C4 fp16 pixel is 8 bytes and stays dense; a C5 pixel is 10 bytes and
/// switches to NC1HWC2 surfaces. The boundary is half a 16-byte feature
/// atom, not a whole one -- `Cin` 5, 6 and 7 are already surfaces.
const DENSE_PIXEL_BYTES: u32 = 8;

/// Padded channel counts by feature-atom count, `(datain_channel, weights)`.
///
/// Indexed by `ceil(Cin / 8) - 1`. Padding is `atoms * 8` everywhere except
/// two measured exceptions, so this is a table rather than arithmetic:
///
/// - 3 atoms (`Cin` 17..24): `datain_channel` stays 24 but weights use 32
/// - 7 atoms (`Cin` 49..56): both round up to 64
///
/// Atom counts 5, 6, 9 and 10 are unpadded, so this is neither "round to a
/// power of two" nor "round to even", and no rule is claimed. Encoding the
/// measurements directly avoids inventing one.
const CHANNEL_PADDING: [(u32, u32); 10] = [
    (8, 8),
    (16, 16),
    (24, 32),
    (32, 32),
    (40, 40),
    (48, 48),
    (64, 64),
    (64, 64),
    (72, 72),
    (80, 80),
];

/// Output channels of the captured reference convolution.
pub const OUTPUT_CHANNELS: u32 = 8;

/// Largest output-channel count this builder will program.
///
/// `CNA_WEIGHT_SIZE2.weight_kernels` is 14 bits, so 16383 is the encodable
/// ceiling. The corpus reaches 512 in a single unsplit program, and nothing
/// in it suggests a limit below the field width -- the vendor never splits
/// the kernel set for capacity at any point measured. The cap is set at the
/// measured extent rather than the encodable one, on the same principle as
/// `MAX_INPUT_CHANNELS`.
pub const MAX_OUTPUT_CHANNELS: u32 = 512;

/// Granularity the DPU's output-channel count is rounded up to.
///
/// Four registers -- `CORE_DATAOUT_SIZE_1.dataout_channel`,
/// `DPU_DATA_CUBE_CHANNEL.channel`, `DPU_RDMA_RDMA_DATA_CUBE_CHANNEL.channel`
/// and `DPU_WDMA_SIZE_0.channel_wdma` -- carry this padded count while
/// `weight_kernels` and `orig_channel` carry the true one. Unlike the input
/// channel padding, this is a clean rule with no table and no exceptions:
/// verified at every Cout in the corpus, including the awkward 20, 24, 28,
/// 40, 56 and 72 where the input padding needed special cases.
const OUTPUT_CHANNEL_GRANULE: u32 = 16;

/// Physical width of one feature atom.
const FEATURE_ATOM_BYTES: u32 = 16;

/// Total CBUF banks the CNA partitions between feature data and weights.
const CBUF_BANKS: u32 = 12;

/// Bytes one CBUF bank holds: 256 entries of 128 bytes.
const CBUF_BANK_BYTES: u32 = 256 * 128;

/// Largest value `CNA_CBUF_CON1.data_entries` can encode. The field is 14
/// bits and holds `tile_input_rows * width`, so it is a hard bound on how
/// much of a feature map one program may cover -- 63 input rows at 256 wide,
/// 127 at 128 wide. This is the hardware reason the vendor splits tall
/// convolutions for capacity: its own `256x256` capture uses 44-row tiles,
/// comfortably under the 63-row ceiling.
const MAX_DATA_ENTRIES: u32 = 0x3fff;

/// Feature pixels one CBUF bank holds: 256 entries of 128 bytes, at one
/// 16-byte feature atom per pixel, is 2048 -- but the vendor allocates in
/// half-bank steps, which is the 1024 below.
const PIXELS_PER_BANK_STEP: u32 = 1024;

/// Logical geometry of the whole feature map a program operates on.
///
/// Every register formula below is validated against a sweep of 35 vendor
/// captures (212 convolution programs) spanning widths 32..256 and heights
/// 32..256, restricted to the supported case: `Cin=3`, `Cout=8`, stride 1,
/// and 1x1 or 3x3 kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub width: u32,
    pub height: u32,
    /// Equal in both axes; `CNA_CONV_CON3` programs it directly, confirmed
    /// across 150 stride-2, -3 and -4 programs.
    pub stride: u32,
    /// Real input channels, before any padding.
    pub in_channels: u32,
    /// Real output channels, before any padding. Programmed directly into
    /// `CNA_WEIGHT_SIZE2.weight_kernels` and
    /// `DPU_DATA_CUBE_CHANNEL.orig_channel` with no rounding at all: the
    /// corpus confirms 23 distinct values from 1 to 512, including 9, 14,
    /// 20, 28, 40, 56 and 72.
    pub out_channels: u32,
}

/// How the feature map is laid out in memory, which the channel count picks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeatureLayout {
    /// Dense NHWC: the pixel is narrower than half a feature atom and the
    /// CNA pads it internally. This is the C3 case of the original captures.
    Dense,
    /// NC1HWC2: one 16-byte atom per pixel per surface.
    Surfaces,
}

impl Shape {
    /// The `32x32` stride-1 geometry of the original vendor captures.
    pub const CAPTURED: Shape = Shape {
        width: IMAGE_WIDTH,
        height: IMAGE_HEIGHT,
        stride: 1,
        in_channels: INPUT_CHANNELS,
        out_channels: OUTPUT_CHANNELS,
    };

    pub fn new(width: u32, height: u32) -> Shape {
        Shape::with_stride(width, height, 1)
    }

    pub fn with_stride(width: u32, height: u32, stride: u32) -> Shape {
        Shape::with_channels(width, height, stride, INPUT_CHANNELS)
    }

    pub fn with_channels(width: u32, height: u32, stride: u32, in_channels: u32) -> Shape {
        Shape::with_out_channels(width, height, stride, in_channels, OUTPUT_CHANNELS)
    }

    pub fn with_out_channels(
        width: u32,
        height: u32,
        stride: u32,
        in_channels: u32,
        out_channels: u32,
    ) -> Shape {
        assert!(
            width > 0 && height > 0,
            "convolution extents must be nonzero"
        );
        assert!(stride > 0, "convolution stride must be nonzero");
        assert!(
            (1..=MAX_INPUT_CHANNELS).contains(&in_channels),
            "input channels must be 1..={MAX_INPUT_CHANNELS}; beyond that the \
             channel padding has no capture backing"
        );
        assert!(
            (1..=MAX_OUTPUT_CHANNELS).contains(&out_channels),
            "output channels must be 1..={MAX_OUTPUT_CHANNELS}, the range the \
             capture corpus covers and the 14-bit weight_kernels field encodes"
        );
        Shape {
            width,
            height,
            stride,
            in_channels,
            out_channels,
        }
    }

    /// Feature atoms one pixel occupies once padded.
    pub fn feature_atoms(&self) -> u32 {
        self.in_channels.div_ceil(8)
    }

    /// Channel count programmed into `CNA_DATA_SIZE1.datain_channel`.
    pub fn padded_channels(&self) -> u32 {
        CHANNEL_PADDING[self.feature_atoms() as usize - 1].0
    }

    /// Channel count the coefficient footprint is computed from.
    pub fn weight_channels(&self) -> u32 {
        CHANNEL_PADDING[self.feature_atoms() as usize - 1].1
    }

    /// Output channel count the DPU is programmed with, rounded up to a
    /// whole [`OUTPUT_CHANNEL_GRANULE`] and never below one.
    ///
    /// The floor is what makes Cout 8 and Cout 16 program the same value,
    /// which is why the shape-only corpus -- fixed at Cout 8 -- could not
    /// distinguish this from the true count.
    pub fn padded_out_channels(&self) -> u32 {
        self.out_channels
            .next_multiple_of(OUTPUT_CHANNEL_GRANULE)
            .max(OUTPUT_CHANNEL_GRANULE)
    }

    /// Bytes of fp16 coefficients the whole kernel set occupies.
    ///
    /// `weight_channels * k * k * Cout * 2`, which reproduces
    /// `CNA_WEIGHT_SIZE0.weight_bytes` in all 829 programs of the corpus.
    /// Note the *padded* input channel count, and specifically the weight
    /// padding rather than the data padding -- at three atoms the two differ,
    /// and it is the weight one that this follows.
    pub fn weight_bytes(&self, kernels: Kernels) -> u32 {
        let kernel = kernel_programming(kernels);
        kernel.size * kernel.size * self.weight_channels() * self.out_channels * 2
    }

    /// Atoms per pixel implied by the weight padding, which is what the CBUF
    /// accounting follows -- not `feature_atoms`. At 3 atoms the two differ.
    fn weight_atoms(&self) -> u32 {
        self.weight_channels() / 8
    }

    /// Whether the feature map is dense NHWC or NC1HWC2 surfaces.
    pub fn layout(&self) -> FeatureLayout {
        if self.in_channels * 2 <= DENSE_PIXEL_BYTES {
            FeatureLayout::Dense
        } else {
            FeatureLayout::Surfaces
        }
    }

    /// Output width, `floor((w + 2 * pad - k) / stride) + 1`. Matches all 150
    /// stride-2, -3 and -4 programs in the sweep corpus.
    pub fn output_width(&self, kernels: Kernels) -> u32 {
        let kernel = kernel_programming(kernels);
        (self.width + 2 * kernel.padding - kernel.size) / self.stride + 1
    }

    /// Output height, by the same rule.
    pub fn output_height(&self, kernels: Kernels) -> u32 {
        let kernel = kernel_programming(kernels);
        (self.height + 2 * kernel.padding - kernel.size) / self.stride + 1
    }

    /// Byte stride of one input row.
    ///
    /// Dense rows are exactly `Cin` fp16 values wide. Surface rows carry one
    /// 16-byte atom per pixel, and the surfaces themselves sit
    /// `width * height * 16` bytes apart.
    pub fn input_row_stride(&self) -> u32 {
        match self.layout() {
            FeatureLayout::Dense => self.width * self.in_channels * 2,
            FeatureLayout::Surfaces => self.width * FEATURE_ATOM_BYTES,
        }
    }

    /// Byte distance between consecutive NC1HWC2 input surfaces.
    pub fn input_surface_stride(&self) -> u32 {
        self.width * self.height * FEATURE_ATOM_BYTES
    }

    /// Byte stride of one output row.
    ///
    /// The output is always NC1HWC2 with one 16-byte atom per pixel per
    /// surface, so this does not depend on `Cout` -- the channel count sets
    /// how many surfaces there are, not how wide a row is. Output geometry,
    /// not input: at stride greater than one the two differ, and the corpus
    /// programs the output dimensions in every one of 150 such programs.
    pub fn output_row_stride(&self, kernels: Kernels) -> u32 {
        self.output_width(kernels) * FEATURE_ATOM_BYTES
    }

    /// CBUF banks the feature data would take if nothing competed for them.
    ///
    /// Derived from 134 captured programs across 11 distinct `(width,
    /// height)` shapes. Deliberately uncapped: a demand above the 12 banks
    /// that exist is meaningful, because it is what makes the weights the
    /// smaller claim in [`data_banks`].
    fn data_bank_demand(&self) -> u32 {
        let pixels = self.width * self.height;
        match self.layout() {
            FeatureLayout::Dense => pixels.div_ceil(PIXELS_PER_BANK_STEP),
            // Surfaces charge per atom, at twice the pixels per step. Fits
            // every measured point: at 32x32 this is `ceil(atoms / 2)`,
            // giving 1,1,2,2,3,3,4,4,5,5 across atom counts 1 through 10.
            FeatureLayout::Surfaces => {
                (pixels * self.weight_atoms()).div_ceil(2 * PIXELS_PER_BANK_STEP)
            }
        }
        .max(1)
    }

    /// CBUF banks the weights would take if nothing competed for them.
    ///
    /// Weights are streamed rather than held resident -- a 512-kernel
    /// program with 589824 bytes of coefficients runs with 8 banks, which
    /// hold 262144 -- so exceeding the CBUF is not an error. Like the data
    /// demand, this stays uncapped so the comparison below can see it.
    fn weight_bank_demand(&self, kernels: Kernels) -> u32 {
        self.weight_bytes(kernels).div_ceil(CBUF_BANK_BYTES).max(1)
    }

    /// CBUF banks the vendor assigns to feature data.
    ///
    /// The two claimants split all 12 banks: every capture satisfies
    /// `data_banks + weight_banks == 12`. When they fit together each takes
    /// what it asked for; when they do not, the *smaller* claim is honoured
    /// in full and the larger takes the remainder. Both halves of that are
    /// measured -- 256x32 Cin 32 Cout 64 wants 16 data banks and 2 weight
    /// banks and is programmed 10/2, while 32x32 Cin 64 Cout 512 wants 4 and
    /// 18 and is programmed 4/8.
    ///
    /// This is the only path by which `Cout` reaches the bank split, and it
    /// opens only once the feature data is already over budget. The corpus
    /// shows it exactly once: at 256x32 with Cin 32, Cout 16 needs 9216
    /// bytes of coefficients and takes one bank, Cout 64 needs 36864 and
    /// takes two, pushing the data allocation from 11 banks down to 10.
    pub fn data_banks(&self, kernels: Kernels) -> u32 {
        let data = self.data_bank_demand();
        let weights = self.weight_bank_demand(kernels);
        let granted = if data <= weights {
            data
        } else {
            data.min(CBUF_BANKS.saturating_sub(weights))
        };
        granted.clamp(1, CBUF_BANKS - 1)
    }

    /// CBUF banks the vendor assigns to weights: everything left over.
    pub fn weight_banks(&self, kernels: Kernels) -> u32 {
        CBUF_BANKS - self.data_banks(kernels)
    }

    /// Most input rows one program may read.
    ///
    /// Two bounds apply and the CBUF one is the tighter. The hard limit is
    /// the 14-bit `CNA_CBUF_CON1.data_entries` field, which holds
    /// `rows * width`; the vendor never approaches it.
    ///
    /// The CBUF bound is the inverse of [`data_bank_demand`]: a bank holds
    /// 1024 dense pixels or 2048 pixel-atoms, so the rows that fit are
    /// whatever the banks granted can carry *at this shape's cost per
    /// pixel*. Charging one atom per pixel unconditionally -- which is what
    /// this did while every capture backing it had `Cin = 3` -- is right in
    /// the dense regime and over-optimistic by a factor of `weight_atoms` in
    /// the surface one. `conv_outchannel_hw` caught it at 256x32 with
    /// `Cin = 32`, where the old rule allowed 44 rows against a real
    /// capacity of 22 and the tile silently lost its last input rows.
    ///
    /// Predicts the vendor's own largest single-core tile in all 17 corpus
    /// captures that split, against 12 for the atom-blind version: 32 rows
    /// at 256 wide with `Cin` 3, 22 at 512, 14 at 768, 11 at 1024, 7 at
    /// 1536, and 22 at 256 wide with `Cin` 32 -- dropping to 20 when a
    /// larger `Cout` takes a second bank for coefficients.
    ///
    /// Hardware tolerates the looser bound in the dense regime
    /// (`conv_wide_shape_hw` passes at ~33 rows on a 256-wide map, above the
    /// vendor's 32), so there it is conservatism rather than a correctness
    /// requirement -- the same pattern as `feature_grains`. In the surface
    /// regime it is a correctness requirement.
    pub fn max_tile_input_rows(&self, kernels: Kernels) -> u32 {
        let banks = self.data_banks(kernels);
        let capacity = match self.layout() {
            FeatureLayout::Dense => banks * PIXELS_PER_BANK_STEP / self.width,
            FeatureLayout::Surfaces => {
                2 * banks * PIXELS_PER_BANK_STEP / (self.width * self.weight_atoms())
            }
        };
        capacity.min(MAX_DATA_ENTRIES / self.width).max(1)
    }

    /// Fewest tiles this shape must be split into to stay encodable.
    ///
    /// A tile producing `r` output rows reads up to `r + 2 * padding` input
    /// rows once its halo is counted, so the padding is charged here rather
    /// than discovered as an overflow inside the builder.
    pub fn min_tiles(&self, kernels: Kernels) -> u32 {
        let halo = 2 * kernel_programming(kernels).padding;
        let rows = self
            .max_tile_input_rows(kernels)
            .saturating_sub(halo)
            .max(1);
        // A tile of `r` output rows reads about `r * stride` input rows.
        let output_rows = rows.div_ceil(self.stride).max(1);
        self.output_height(kernels).div_ceil(output_rows)
    }
}

#[derive(Clone, Copy)]
struct KernelProgramming {
    size: u32,
    padding: u32,
}

fn kernel_programming(kernels: Kernels) -> KernelProgramming {
    // Deliberately not `(k - 1) / 2`: only these two geometries have capture
    // backing, and guessing is what this module exists to avoid.
    match kernels {
        [1, 1] => KernelProgramming {
            size: 1,
            padding: 0,
        },
        [3, 3] => KernelProgramming {
            size: 3,
            padding: 1,
        },
        _ => panic!("conv_2d only has vendor reference data for 1x1 and 3x3 square kernels"),
    }
}

/// ARGB input mode for a dense feature map.
///
/// Only reachable for `Cin` 1..=4, since wider pixels use surfaces. The
/// captures confirm 3 and 4 directly; 1 and 2 follow the enum's own
/// definition, which came from the vendor register description.
fn argb_input_mode(in_channels: u32) -> ArgbInputMode {
    match in_channels {
        1 => ArgbInputMode::OneChannel,
        2 => ArgbInputMode::TwoChannels,
        3 => ArgbInputMode::ThreeChannels,
        4 => ArgbInputMode::FourChannels,
        _ => unreachable!("dense layout is only used up to four input channels"),
    }
}

/// One contiguous range of output rows, with the input rows it reads.
///
/// `in_rows` counts input rows actually fetched from memory: it includes the
/// halo row a continuation tile reads from its neighbour, and excludes rows
/// supplied by zero padding. `pad_top` is 1 only for a tile whose first
/// output row sits at the top of the image and therefore has no real input
/// row above it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tile {
    pub out_first: u32,
    pub out_rows: u32,
    pub in_first: u32,
    pub in_rows: u32,
    pub pad_top: u32,
}

impl Tile {
    /// Splits `shape` into `tiles` output-row ranges.
    ///
    /// At the captured `32x32` geometry this reproduces the vendor's own
    /// splits: `tiles = 1` gives 32 rows, `tiles = 2` gives 16+16, and
    /// `tiles = 3` gives 11+11+10, matching captured groups 1, 2-3, and 4-6.
    pub fn split(shape: Shape, kernels: Kernels, tiles: u32) -> Vec<Tile> {
        let output_height = shape.output_height(kernels);
        assert!(
            (1..=output_height).contains(&tiles),
            "tile count must be between 1 and the {output_height} output rows"
        );
        let padding = kernel_programming(kernels).padding;
        let stride = shape.stride;
        let base = output_height / tiles;
        let remainder = output_height % tiles;

        let mut out = Vec::with_capacity(tiles as usize);
        let mut out_first: u32 = 0;
        for index in 0..tiles {
            let out_rows = base + u32::from(index < remainder);

            // Halo: the first input row a tile touches is its first output
            // row projected back through the stride, less the padding it
            // would otherwise read above the image. Matches all 150
            // stride-2, -3 and -4 programs in the corpus.
            let in_first = (out_first * stride).saturating_sub(padding);
            let last_tap = (out_first + out_rows - 1) * stride + padding;
            let exact = last_tap.min(shape.height - 1) - in_first + 1;

            // The vendor reads at least a full stride block per output row,
            // which exceeds the exact tap span at stride > 1. Taking the
            // larger of the two is safe by construction: it is never below
            // `exact`, so every tap the tile needs is resident. Where the
            // corpus disagrees it reads more still, which costs DMA rather
            // than correctness.
            let in_rows = exact.max(out_rows * stride).min(shape.height - in_first);

            out.push(Tile {
                out_first,
                out_rows,
                in_first,
                in_rows,
                pad_top: if out_first == 0 { padding } else { 0 },
            });
            out_first += out_rows;
        }
        out
    }

    /// The single tile covering the whole image.
    pub fn whole(shape: Shape, kernels: Kernels) -> Tile {
        Tile::split(shape, kernels, 1)[0]
    }

    /// Byte offset of this tile's first input row from the tensor base.
    pub fn input_offset(&self, shape: Shape) -> u32 {
        self.in_first * shape.input_row_stride()
    }

    /// Byte offset of this tile's first output row from the tensor base.
    pub fn output_offset(&self, shape: Shape, kernels: Kernels) -> u32 {
        self.out_first * shape.output_row_stride(kernels)
    }
}

#[inline]
fn zero<R: RegisterMeta>() -> RegCmd {
    Register::<R>::new().build()
}

/// Builds the vendor-matching single-core regcmd program for the captured
/// `32x32x3 -> 32x32x8` fp16 convolution.
///
/// `kernels` must be `[1, 1]` or `[3, 3]`. Buffer-address fields are zero,
/// as they are in the captured RKNN regcmd blob; a caller that submits this
/// program must relocate the feature, weight, bias, and destination
/// addresses first.
///
/// The returned 136-word sequence is group 1 of the vendor blob. Groups
/// 2-3 and 4-6 are alternative two- and three-core height-split programs,
/// not continuations of this command stream.
pub fn conv_2d(kernels: Kernels) -> Vec<RegCmd> {
    let shape = Shape::CAPTURED;
    conv_2d_tile(shape, kernels, &Tile::whole(shape, kernels))
}

/// Builds the single-core regcmd program for one output-row `tile`.
///
/// `conv_2d(k)` is `conv_2d_tile(k, &Tile::whole(k))` and reproduces captured
/// group 1 bit for bit. For a partial tile, the sixteen tile-dependent
/// registers take the values derived from the cross-group capture diff; see
/// the module documentation for what a tile program does and does not match.
///
/// The feature and destination address registers carry this tile's *offset*
/// from the tensor base, exactly as the vendor's own split programs do. A
/// caller relocating these must add the buffer's DMA address to the existing
/// value rather than overwrite it.
pub fn conv_2d_tile(shape: Shape, kernels: Kernels, tile: &Tile) -> Vec<RegCmd> {
    conv_2d_tile_with_grains(shape, kernels, tile, feature_grains(kernels, tile))
}

/// The `CNA_CONV_CON2.feature_grains` value this module programs.
///
/// Feature rows the CNA buffers before the convolution starts. A shape sweep
/// of 49 vendor captures shows the vendor does not use one formula: across
/// 297 programs it matches this prefetch value 63% of the time, uses exactly
/// `in_rows` 28% of the time, and goes *below* `in_rows` in 6% -- and no
/// register field in the corpus separates those cases, so the choice appears
/// to come from compiler state that never reaches the register program.
///
/// The TRM calls its own formula "suggested", which implies a range of valid
/// settings rather than one correct value, and this larger-than-vendor value
/// is the one the passing hardware tests use. `conv_grains_probe_hw` measures
/// the range that actually works.
pub fn feature_grains(kernels: Kernels, tile: &Tile) -> u32 {
    tile.in_rows + kernel_programming(kernels).size + tile.pad_top
}

/// Builds a tile program with an explicit `feature_grains`, for probing which
/// values the hardware accepts. Prefer [`conv_2d_tile`].
pub fn conv_2d_tile_with_grains(
    shape: Shape,
    kernels: Kernels,
    tile: &Tile,
    feature_grains: u32,
) -> Vec<RegCmd> {
    let padded_channels = shape.padded_channels();
    let weight_channels = shape.weight_channels();
    // The DPU counts output channels in whole granules while the CNA counts
    // the real kernels. Both appear below, and they differ at every Cout
    // that is not already a multiple of 16.
    let padded_out_channels = shape.padded_out_channels();
    const FP16_BYTES: u32 = 2;

    let width = shape.width;
    let height = shape.height;
    let out_width = shape.output_width(kernels);
    let out_height = shape.output_height(kernels);
    let weight_banks = shape.weight_banks(kernels);
    let data_banks = shape.data_banks(kernels);

    assert!(
        tile.out_rows > 0 && tile.out_first + tile.out_rows <= out_height,
        "tile output rows {}..{} fall outside the {out_height}-row output",
        tile.out_first,
        tile.out_first + tile.out_rows
    );
    assert!(
        tile.in_rows * width <= MAX_DATA_ENTRIES,
        "tile reads {} rows of {width} pixels; CNA_CBUF_CON1.data_entries is \
         14 bits and holds at most {MAX_DATA_ENTRIES}. Split into at least {} \
         tiles (Shape::min_tiles)",
        tile.in_rows,
        shape.min_tiles(kernels),
    );
    assert!(
        tile.in_rows > 0 && tile.in_first + tile.in_rows <= height,
        "tile input rows {}..{} fall outside the {height}-row image",
        tile.in_first,
        tile.in_first + tile.in_rows
    );

    let kernel = kernel_programming(kernels);

    // Layout-dependent programming. Dense rows are counted in pixels and the
    // whole tile is resident, so `data_entries` scales with the tile height.
    // Surfaces are counted in atoms and `data_entries` does not depend on the
    // tile at all -- the same field carries different quantities in the two
    // regimes, which is why they are computed apart rather than parameterised.
    let (line_stride, surf_stride, data_entries) = match shape.layout() {
        FeatureLayout::Dense => (width, width * (height - 1), width * tile.in_rows),
        FeatureLayout::Surfaces => (
            width * 4,
            width * (height - 4),
            width * shape.weight_atoms() / 4,
        ),
    };

    let weight_bytes_per_kernel = kernel.size * kernel.size * weight_channels * FP16_BYTES;
    let weight_bytes = shape.weight_bytes(kernels);
    let mut commands = Vec::with_capacity(136);

    // CNA preamble, followed by the DPU/DPU_RDMA ping-pong pointers.
    let mut cbuf_con0 = Register::<CnaCbufCon0>::new();
    cbuf_con0
        .weight_bank(Bits::new(weight_banks))
        .data_bank(Bits::new(data_banks));
    commands.push(cbuf_con0.build());
    commands.push(zero::<CnaDcompRegnum>());
    commands.push(zero::<CnaDcompCtrl>());

    // The dense regime is the CNA's ARGB image-input path: `argb_in` names
    // the channel count (OneChannel = 8 through FourChannels = 11) and both
    // `nonalign_dma` and `group_line_off` are set. The surface regime clears
    // all three. Measured across the channel sweep: Cin 3 programs 10/1/1,
    // Cin 4 programs 11/1/1, and every Cin from 5 up programs 0/0/0.
    //
    // Leaving these at the captured C3 values made the hardware read three
    // channels per pixel at every channel count, which is what
    // `conv_multichannel_hw` caught.
    let mut conv_con1 = Register::<CnaConvCon1>::new();
    match shape.layout() {
        FeatureLayout::Dense => {
            conv_con1
                .nonalign_dma(Bits::new(1))
                .group_line_off(Bits::new(1))
                .argb_in(argb_input_mode(shape.in_channels).into());
        }
        FeatureLayout::Surfaces => {
            conv_con1
                .nonalign_dma(Bits::new(0))
                .group_line_off(Bits::new(0))
                .argb_in(Bits::new(0));
        }
    }
    conv_con1
        .proc_precision(DataPrecision::Fp16.into())
        .in_precision(DataPrecision::Fp16.into());
    commands.push(conv_con1.build());
    commands.push(
        Register::<DpuSPointer>::new()
            .pointer_pp_mode(Bits::new(1))
            .executer_pp_en(Bits::new(1))
            .pointer_pp_en(Bits::new(1))
            .build(),
    );
    commands.push(
        Register::<DpuRdmaSPointer>::new()
            .pointer_pp_mode(Bits::new(1))
            .executer_pp_en(Bits::new(1))
            .pointer_pp_en(Bits::new(1))
            .build(),
    );

    // CNA convolution and DMA programming.
    commands.push(conv_con1.build());
    commands.push(
        Register::<CnaConvCon2>::new()
            .feature_grains(Bits::new(feature_grains))
            .build(),
    );
    commands.push(
        Register::<CnaConvCon3>::new()
            .conv_x_stride(Bits::new(shape.stride))
            .conv_y_stride(Bits::new(shape.stride))
            .build(),
    );
    commands.push(
        Register::<CnaDataSize0>::new()
            .datain_width(Bits::new(width))
            .datain_height(Bits::new(tile.in_rows))
            .build(),
    );
    commands.push(
        Register::<CnaDataSize1>::new()
            .datain_channel_real(Bits::new(shape.in_channels - 1))
            .datain_channel(Bits::new(padded_channels))
            .build(),
    );
    commands.push(
        Register::<CnaDataSize2>::new()
            .dataout_width(Bits::new(out_width))
            .build(),
    );
    commands.push(
        Register::<CnaDataSize3>::new()
            .dataout_atomics(Bits::new(out_width * tile.out_rows))
            .build(),
    );
    commands.push(
        Register::<CnaWeightSize0>::new()
            .weight_bytes(Bits::new(weight_bytes))
            .build(),
    );
    commands.push(
        Register::<CnaWeightSize1>::new()
            .weight_bytes_per_kernel(Bits::new(weight_bytes_per_kernel))
            .build(),
    );
    commands.push(
        Register::<CnaWeightSize2>::new()
            .weight_width(Bits::new(kernel.size))
            .weight_height(Bits::new(kernel.size))
            .weight_kernels(Bits::new(shape.out_channels))
            .build(),
    );
    commands.push(cbuf_con0.build());
    commands.push(
        Register::<CnaCbufCon1>::new()
            .data_entries(Bits::new(data_entries))
            .build(),
    );
    commands.push(
        Register::<CnaCvtCon0>::new()
            .data_sign(Bits::new(1))
            .cvt_type(Bits::new(1))
            .cvt_bypass(Bits::new(1))
            .build(),
    );
    commands.push(
        Register::<CnaCvtCon1>::new()
            .cvt_scale0(Bits::new(1))
            .build(),
    );
    commands.push(
        Register::<CnaCvtCon2>::new()
            .cvt_scale1(Bits::new(1))
            .build(),
    );
    commands.push(
        Register::<CnaCvtCon3>::new()
            .cvt_scale2(Bits::new(1))
            .build(),
    );
    commands.push(
        Register::<CnaCvtCon4>::new()
            .cvt_scale3(Bits::new(1))
            .build(),
    );
    commands.push(zero::<CnaFcCon0>());
    commands.push(zero::<CnaFcCon1>());
    commands.push(
        // pad_left is a kernel property (width is never tiled); pad_top is a
        // tile property. The untiled captures set both to 1 and cannot
        // distinguish them -- groups 3, 5, and 6 program 0x10, separating the
        // two nibbles.
        Register::<CnaPadCon0>::new()
            .pad_top(Bits::new(tile.pad_top))
            .pad_left(Bits::new(kernel.padding))
            .build(),
    );
    commands.push(
        Register::<CnaFeatureDataAddr>::new()
            .feature_base_addr(Bits::new(tile.input_offset(shape)))
            .build(),
    );
    commands.push(zero::<CnaFcCon2>());
    commands.push(
        Register::<CnaDmaCon0>::new()
            .data_burst_len(BurstLength::Sixteen.into())
            .weight_burst_len(BurstLength::Sixteen.into())
            .build(),
    );
    commands.push(
        Register::<CnaDmaCon1>::new()
            .line_stride(Bits::new(line_stride))
            .build(),
    );
    commands.push(
        Register::<CnaDmaCon2>::new()
            .surf_stride(Bits::new(surf_stride))
            .build(),
    );
    commands.push(
        Register::<CnaFcDataSize0>::new()
            .dma_width(Bits::new(width))
            .dma_height(Bits::new(tile.in_rows))
            .build(),
    );
    commands.push(
        Register::<CnaFcDataSize1>::new()
            .dma_channel(Bits::new(padded_channels))
            .build(),
    );
    commands.push(zero::<CnaDcompCtrl>());
    commands.push(zero::<CnaDcompRegnum>());
    commands.push(zero::<CnaDcompAddr0>());
    commands.push(zero::<CnaDcompAmount0>());
    commands.push(zero::<CnaDcompAmount1>());
    commands.push(zero::<CnaDcompAmount2>());
    commands.push(zero::<CnaDcompAmount3>());
    commands.push(zero::<CnaDcompAmount4>());
    commands.push(zero::<CnaDcompAmount5>());
    commands.push(zero::<CnaDcompAmount6>());
    commands.push(zero::<CnaDcompAmount7>());
    commands.push(zero::<CnaDcompAmount8>());
    commands.push(zero::<CnaDcompAmount9>());
    commands.push(zero::<CnaDcompAmount10>());
    commands.push(zero::<CnaDcompAmount11>());
    commands.push(zero::<CnaDcompAmount12>());
    commands.push(zero::<CnaDcompAmount13>());
    commands.push(zero::<CnaDcompAmount14>());
    commands.push(zero::<CnaDcompAmount15>());
    commands.push(zero::<CnaCvtCon5>());
    commands.push(zero::<CnaPadCon1>());

    // CORE.
    commands.push(
        Register::<CoreMiscCfg>::new()
            .proc_precision(DataPrecision::Fp16.into())
            .build(),
    );
    commands.push(
        Register::<CoreDataoutSize0>::new()
            .dataout_width(Bits::new(out_width - 1))
            .dataout_height(Bits::new(tile.out_rows - 1))
            .build(),
    );
    commands.push(
        Register::<CoreDataoutSize1>::new()
            .dataout_channel(Bits::new(padded_out_channels - 1))
            .build(),
    );
    commands.push(zero::<CoreClipTruncate>());
    commands.push(zero::<CoreReserved3030>());

    // DPU output, conversion, and disabled LUT programming.
    commands.push(
        Register::<DpuFeatureModeCfg>::new()
            .burst_len(BurstLength::Sixteen.into())
            .output_mode(DpuOutputMode::ExternalMemory.into())
            .build(),
    );
    commands.push(
        Register::<DpuDataFormat>::new()
            .in_precision(DataPrecision::Fp16.into())
            .out_precision(DataPrecision::Fp16.into())
            .proc_precision(DataPrecision::Fp16.into())
            .build(),
    );
    commands.push(zero::<DpuOffsetPend>());
    commands.push(
        Register::<DpuDstBaseAddr>::new()
            .dst_base_addr(Bits::new(tile.output_offset(shape, kernels)))
            .build(),
    );
    commands.push(
        Register::<DpuDstSurfStride>::new()
            .dst_surf_stride(Bits::new(out_width * out_height))
            .build(),
    );
    commands.push(
        Register::<DpuDataCubeWidth>::new()
            .width(Bits::new(out_width - 1))
            .build(),
    );
    commands.push(
        Register::<DpuDataCubeHeight>::new()
            .height(Bits::new(tile.out_rows - 1))
            .build(),
    );
    commands.push(zero::<DpuDataCubeNotchAddr>());
    commands.push(
        Register::<DpuDataCubeChannel>::new()
            .orig_channel(Bits::new(shape.out_channels - 1))
            .channel(Bits::new(padded_out_channels - 1))
            .build(),
    );
    commands.push(
        Register::<DpuBsCfg>::new()
            .bs_alu_algo(Bits::new(2))
            .bs_alu_src(Bits::new(1))
            .bs_relu_bypass(Bits::new(1))
            .bs_mul_bypass(Bits::new(1))
            .build(),
    );
    commands.push(zero::<DpuBsAluCfg>());
    commands.push(zero::<DpuBsMulCfg>());
    commands.push(zero::<DpuBsReluxCmpValue>());
    commands.push(
        Register::<DpuBsOwCfg>::new()
            .size_e_0(Bits::new(1))
            .size_e_1(Bits::new(1))
            .size_e_2(Bits::new(1))
            .od_bypass(Bits::new(1))
            .build(),
    );
    commands.push(zero::<DpuBsOwOp>());
    commands.push(
        Register::<DpuWdmaSize0>::new()
            .channel_wdma(Bits::new(padded_out_channels - 1))
            .build(),
    );
    commands.push(
        Register::<DpuWdmaSize1>::new()
            .height_wdma(Bits::new(tile.out_rows - 1))
            .width_wdma(Bits::new(out_width - 1))
            .build(),
    );
    commands.push(
        Register::<DpuBnCfg>::new()
            .bn_relu_bypass(Bits::new(1))
            .bn_mul_bypass(Bits::new(1))
            .bn_alu_bypass(Bits::new(1))
            .bn_bypass(Bits::new(1))
            .build(),
    );
    commands.push(zero::<DpuBnAluCfg>());
    commands.push(zero::<DpuBnMulCfg>());
    commands.push(zero::<DpuBnReluxCmpValue>());
    commands.push(
        Register::<DpuEwCfg>::new()
            .ew_relu_bypass(Bits::new(1))
            .ew_op_cvt_bypass(Bits::new(1))
            .ew_lut_bypass(Bits::new(1))
            .ew_op_bypass(Bits::new(1))
            .ew_bypass(Bits::new(1))
            .build(),
    );
    commands.push(zero::<DpuEwCvtOffsetValue>());
    commands.push(
        Register::<DpuEwCvtScaleValue>::new()
            .ew_op_cvt_scale(Bits::new(1))
            .build(),
    );
    commands.push(zero::<DpuEwReluxCmpValue>());
    commands.push(zero::<DpuOutCvtOffset>());
    commands.push(
        Register::<DpuOutCvtScale>::new()
            .fp32tofp16_en(Bits::new(1))
            .out_cvt_scale(Bits::new(1))
            .build(),
    );
    commands.push(zero::<DpuOutCvtShift>());
    commands.push(zero::<DpuEwOpValue0>());
    commands.push(zero::<DpuEwOpValue1>());
    commands.push(zero::<DpuEwOpValue2>());
    commands.push(zero::<DpuEwOpValue3>());
    commands.push(zero::<DpuEwOpValue4>());
    commands.push(zero::<DpuEwOpValue5>());
    commands.push(zero::<DpuEwOpValue6>());
    commands.push(zero::<DpuEwOpValue7>());
    commands.push(
        Register::<DpuSurfaceAdd>::new()
            .surf_add(Bits::new(out_width * out_height * FP16_BYTES))
            .build(),
    );
    commands.push(zero::<DpuReserved40c4>());
    commands.push(zero::<DpuLutAccessCfg>());
    commands.push(zero::<DpuLutAccessData>());
    commands.push(zero::<DpuLutCfg>());
    commands.push(zero::<DpuLutInfo>());
    commands.push(zero::<DpuLutLeStart>());
    commands.push(zero::<DpuLutLeEnd>());
    commands.push(zero::<DpuLutLoStart>());
    commands.push(zero::<DpuLutLoEnd>());
    commands.push(zero::<DpuLutLeSlopeScale>());
    commands.push(zero::<DpuLutLeSlopeShift>());
    commands.push(zero::<DpuLutLoSlopeScale>());
    commands.push(zero::<DpuLutLoSlopeShift>());

    // DPU_RDMA. The main feature path is disabled because CNA/CORE feed
    // DPU directly; BRDMA supplies the bias data.
    commands.push(
        Register::<DpuRdmaDataCubeWidth>::new()
            .width(Bits::new(out_width - 1))
            .build(),
    );
    commands.push(
        Register::<DpuRdmaDataCubeHeight>::new()
            .height(Bits::new(tile.out_rows - 1))
            .build(),
    );
    commands.push(
        Register::<DpuRdmaDataCubeChannel>::new()
            .channel(Bits::new(padded_out_channels - 1))
            .build(),
    );
    commands.push(zero::<DpuRdmaSrcBaseAddr>());
    commands.push(
        Register::<DpuRdmaBrdmaCfg>::new()
            .brdma_data_use(Bits::new(1))
            .build(),
    );
    commands.push(zero::<DpuRdmaBsBaseAddr>());
    commands.push(zero::<DpuRdmaNrdmaCfg>());
    commands.push(zero::<DpuRdmaBnBaseAddr>());
    commands.push(
        Register::<DpuRdmaErdmaCfg>::new()
            .erdma_disable(Bits::new(1))
            .build(),
    );
    commands.push(zero::<DpuRdmaEwBaseAddr>());
    commands.push(zero::<DpuRdmaEwSurfStride>());
    commands.push(
        Register::<DpuRdmaFeatureModeCfg>::new()
            .burst_len(BurstLength::Sixteen.into())
            .mrdma_disable(Bits::new(1))
            .in_precision(DataPrecision::Fp16.into())
            .proc_precision(DataPrecision::Fp16.into())
            .build(),
    );
    commands.push(zero::<DpuRdmaSrcDmaCfg>());
    commands.push(zero::<DpuRdmaSurfNotch>());
    commands.push(zero::<DpuRdmaPadCfg>());
    commands.push(
        Register::<DpuRdmaWeight>::new()
            .e_weight(Bits::new(1))
            .n_weight(Bits::new(1))
            .b_weight(Bits::new(1))
            .m_weight(Bits::new(1))
            .build(),
    );
    commands.push(zero::<DpuRdmaEwSurfNotch>());

    // Vendor PC trailer: placeholder, zero register count, required marker,
    // combined operation-enable mask, and six words of alignment padding.
    commands.push(PCTrailer::single_task_placeholder());
    commands.push(zero::<PCRegisterAmounts>());
    commands.push(PCTrailer::required_marker());
    commands.push(PCTrailer::operation_enable(PCOperationMask::CONVOLUTION));
    commands.extend((0..6).map(|_| PCTrailer::alignment_padding()));

    debug_assert_eq!(commands.len(), 136);
    commands
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode(command: &RegCmd) -> (u32, u32, u32) {
        (
            (command.0 >> 48) as u32,
            command.0 as u32 & 0xffff,
            (command.0 >> 16) as u32,
        )
    }

    fn fnv1a(commands: &[RegCmd]) -> u64 {
        commands
            .iter()
            .flat_map(|command| command.0.to_le_bytes())
            .fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
                (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
            })
    }

    #[test]
    fn vendor_reference_program_has_expected_layout() {
        let commands = conv_2d([1, 1]);
        assert_eq!(commands.len(), 136);

        assert_eq!(decode(&commands[0]), (0x0201, 0x1040, 0x0000_00b1));
        assert_eq!(decode(&commands[4]), (0x1001, 0x4004, 0x0000_000e));
        assert_eq!(decode(&commands[5]), (0x2001, 0x5004, 0x0000_000e));
        assert_eq!(decode(&commands[54]), (0x0801, 0x3010, 0x0000_0200));
        assert_eq!(decode(&commands[59]), (0x1001, 0x400c, 0x0000_01e4));
        assert_eq!(decode(&commands[109]), (0x2001, 0x500c, 0x0000_001f));

        assert_eq!(commands[126].0, 0);
        assert_eq!(commands[127].0, 0x0101_0000_0000_0014);
        assert_eq!(commands[128].0, 0x0041_0000_0000_0000);
        assert_eq!(commands[129].0, 0x0081_0000_001d_0008);
        assert!(commands[130..].iter().all(|command| command.0 == 0));
    }

    #[test]
    fn programs_match_captured_vendor_group_one_bit_for_bit() {
        assert_eq!(fnv1a(&conv_2d([1, 1])), 0x2577_26d7_f13a_1636);
        assert_eq!(fnv1a(&conv_2d([3, 3])), 0x8da7_c9ed_d561_7ccf);
    }

    #[test]
    fn kernel_geometry_changes_exactly_the_five_vendor_words() {
        let one_by_one = conv_2d([1, 1]);
        let three_by_three = conv_2d([3, 3]);
        let changed: Vec<_> = one_by_one
            .iter()
            .zip(&three_by_three)
            .enumerate()
            .filter_map(|(index, (left, right))| (left.0 != right.0).then_some(index))
            .collect();

        assert_eq!(changed, [7, 13, 14, 15, 25]);
        assert_eq!(decode(&one_by_one[7]).2, 0x0000_0210);
        assert_eq!(decode(&three_by_three[7]).2, 0x0000_0240);
        assert_eq!(decode(&one_by_one[13]).2, 0x0000_0080);
        assert_eq!(decode(&three_by_three[13]).2, 0x0000_0480);
        assert_eq!(decode(&one_by_one[14]).2, 0x0000_0010);
        assert_eq!(decode(&three_by_three[14]).2, 0x0000_0090);
        assert_eq!(decode(&one_by_one[15]).2, 0x0101_0008);
        assert_eq!(decode(&three_by_three[15]).2, 0x0303_0008);
        assert_eq!(decode(&one_by_one[25]).2, 0);
        assert_eq!(decode(&three_by_three[25]).2, 0x0000_0011);
    }

    #[test]
    #[should_panic(expected = "only has vendor reference data")]
    fn rejects_uncaptured_kernel_geometry() {
        let _ = conv_2d([5, 5]);
    }

    /// The six captured groups, in the order they appear in the regcmd blob:
    /// one 1-core plan, then a 2-core plan, then a 3-core plan.
    const CAPTURED_PLANS: [(u32, u32); 6] = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)];

    fn captured_tile(kernels: Kernels, group: usize) -> Tile {
        let (tiles, index) = CAPTURED_PLANS[group];
        Tile::split(Shape::CAPTURED, kernels, tiles)[index as usize]
    }

    /// First write to `R`. A few registers are written twice per program --
    /// CnaConvCon1 and CnaCbufCon0 among them -- always with the same value.
    fn first_value_of<R: RegisterMeta>(commands: &[RegCmd]) -> u32 {
        commands
            .iter()
            .filter(|command| {
                (command.0 >> 48) as u32 == R::DOMAIN && (command.0 as u32 & 0xffff) == R::OFFSET
            })
            .map(|command| (command.0 >> 16) as u32)
            .next()
            .expect("register is never written")
    }

    fn value_of<R: RegisterMeta>(commands: &[RegCmd]) -> u32 {
        let matches: Vec<u32> = commands
            .iter()
            .filter(|command| {
                (command.0 >> 48) as u32 == R::DOMAIN && (command.0 as u32 & 0xffff) == R::OFFSET
            })
            .map(|command| (command.0 >> 16) as u32)
            .collect();
        assert_eq!(
            matches.len(),
            1,
            "expected exactly one write to the register"
        );
        matches[0]
    }

    #[test]
    fn tile_split_reproduces_vendor_row_ranges() {
        // REG_DPU_DATA_CUBE_HEIGHT decodes to 32, 16/16, and 11/11/10 rows
        // across the three captured plans.
        let rows: Vec<Vec<u32>> = [1, 2, 3]
            .iter()
            .map(|&n| {
                Tile::split(Shape::CAPTURED, [3, 3], n)
                    .iter()
                    .map(|t| t.out_rows)
                    .collect()
            })
            .collect();
        assert_eq!(rows, vec![vec![32], vec![16, 16], vec![11, 11, 10]]);

        // Every plan covers the image exactly once, with no gap or overlap.
        for tiles in 1..=IMAGE_HEIGHT {
            let split = Tile::split(Shape::CAPTURED, [3, 3], tiles);
            assert_eq!(split.iter().map(|t| t.out_rows).sum::<u32>(), IMAGE_HEIGHT);
            for pair in split.windows(2) {
                assert_eq!(pair[0].out_first + pair[0].out_rows, pair[1].out_first);
            }
        }
    }

    #[test]
    fn tile_halo_matches_captured_feature_offsets() {
        // DESIGN_NOTES: the 3x3 programs begin each continuation tile one
        // input row earlier than the 1x1 programs do.
        let three = Tile::split(Shape::CAPTURED, [3, 3], 3);
        let one = Tile::split(Shape::CAPTURED, [1, 1], 3);
        assert_eq!(
            three
                .iter()
                .map(|t| t.input_offset(Shape::CAPTURED))
                .collect::<Vec<_>>(),
            [0x0000, 0x0780, 0x0fc0]
        );
        assert_eq!(
            one.iter()
                .map(|t| t.input_offset(Shape::CAPTURED))
                .collect::<Vec<_>>(),
            [0x0000, 0x0840, 0x1080]
        );
        assert_eq!(
            three
                .iter()
                .map(|t| t.output_offset(Shape::CAPTURED, [3, 3]))
                .collect::<Vec<_>>(),
            [0x0000, 0x1600, 0x2c00]
        );

        // Rows actually read, including halo and excluding padded rows.
        assert_eq!(
            three.iter().map(|t| t.in_rows).collect::<Vec<_>>(),
            [12, 13, 11]
        );
        assert_eq!(
            three.iter().map(|t| t.pad_top).collect::<Vec<_>>(),
            [1, 0, 0]
        );
    }

    #[test]
    fn tile_registers_match_all_six_captured_groups() {
        // Observed values, group 1 through group 6, from the bitbiter reports
        // in the design-spike repo (conv-group-N.md).
        const FEATURE_GRAINS: [[u32; 6]; 2] = [
            [36, 21, 20, 16, 16, 14], // 3x3
            [33, 17, 17, 12, 12, 11], // 1x1
        ];
        const DATA_SIZE0: [[u32; 6]; 2] = [
            [
                0x20_0020, 0x20_0011, 0x20_0011, 0x20_000c, 0x20_000d, 0x20_000b,
            ],
            [
                0x20_0020, 0x20_0010, 0x20_0010, 0x20_000b, 0x20_000b, 0x20_000a,
            ],
        ];
        const CBUF_CON1: [[u32; 6]; 2] = [
            [0x400, 0x220, 0x220, 0x180, 0x1a0, 0x160],
            [0x400, 0x200, 0x200, 0x160, 0x160, 0x140],
        ];
        const PAD_CON0: [[u32; 6]; 2] = [
            [0x11, 0x11, 0x10, 0x11, 0x10, 0x10],
            [0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
        ];
        const FEATURE_ADDR: [[u32; 6]; 2] = [
            [0, 0, 0xb40, 0, 0x780, 0xfc0],
            [0, 0, 0xc00, 0, 0x840, 0x1080],
        ];
        // Output geometry is identical for both kernels: same output tile.
        const DATA_SIZE3: [u32; 6] = [0x400, 0x200, 0x200, 0x160, 0x160, 0x140];
        const DST_BASE: [u32; 6] = [0, 0, 0x2000, 0, 0x1600, 0x2c00];
        const CUBE_HEIGHT: [u32; 6] = [0x1f, 0x0f, 0x0f, 0x0a, 0x0a, 0x09];
        const DATAOUT_SIZE0: [u32; 6] = [
            0x1f_001f, 0x0f_001f, 0x0f_001f, 0x0a_001f, 0x0a_001f, 0x09_001f,
        ];

        for (row, kernels) in [(0usize, [3usize, 3]), (1, [1, 1])] {
            for group in 0..6 {
                let tile = captured_tile(kernels, group);
                let program = conv_2d_tile(Shape::CAPTURED, kernels, &tile);
                let at = |what: &str, got: u32, want: u32| {
                    assert_eq!(
                        got,
                        want,
                        "{what} mismatch, {kernels:?} group {}",
                        group + 1
                    );
                };

                // FEATURE_GRAINS is a field, not a whole register: the vendor's
                // multi-core programs also carry a plan index in bits 31:28,
                // which a standalone tile program does not reproduce.
                let grains = (value_of::<CnaConvCon2>(&program) & 0x0000_3ff0) >> 4;
                at("feature_grains", grains, FEATURE_GRAINS[row][group]);

                at(
                    "data_size0",
                    value_of::<CnaDataSize0>(&program),
                    DATA_SIZE0[row][group],
                );
                at(
                    "fc_data_size0",
                    value_of::<CnaFcDataSize0>(&program),
                    DATA_SIZE0[row][group],
                );
                at(
                    "cbuf_con1",
                    value_of::<CnaCbufCon1>(&program),
                    CBUF_CON1[row][group],
                );
                at(
                    "pad_con0",
                    value_of::<CnaPadCon0>(&program),
                    PAD_CON0[row][group],
                );
                at(
                    "feature_data_addr",
                    value_of::<CnaFeatureDataAddr>(&program),
                    FEATURE_ADDR[row][group],
                );
                at(
                    "data_size3",
                    value_of::<CnaDataSize3>(&program),
                    DATA_SIZE3[group],
                );
                at(
                    "dst_base_addr",
                    value_of::<DpuDstBaseAddr>(&program),
                    DST_BASE[group],
                );
                at(
                    "data_cube_height",
                    value_of::<DpuDataCubeHeight>(&program),
                    CUBE_HEIGHT[group],
                );
                at(
                    "rdma_data_cube_height",
                    value_of::<DpuRdmaDataCubeHeight>(&program),
                    CUBE_HEIGHT[group],
                );
                at(
                    "core_dataout_size0",
                    value_of::<CoreDataoutSize0>(&program),
                    DATAOUT_SIZE0[group],
                );
                at(
                    "wdma_size1",
                    value_of::<DpuWdmaSize1>(&program),
                    DATAOUT_SIZE0[group],
                );
            }
        }
    }

    #[test]
    fn cbuf_bank_split_matches_every_swept_shape() {
        // (width, height) -> data_bank, read out of CNA_CBUF_CON0 across the
        // 134 C3/Cout8/stride-1 programs in the shape-sweep corpus. Every one
        // also satisfies data_bank + weight_bank == 12.
        const OBSERVED: [(u32, u32, u32); 11] = [
            (32, 32, 1),
            (32, 64, 2),
            (32, 128, 4),
            (32, 256, 8),
            (64, 32, 2),
            (64, 64, 4),
            (96, 32, 3),
            (128, 32, 4),
            (128, 128, 11),
            (192, 32, 6),
            (256, 32, 8),
        ];
        for (width, height, data_bank) in OBSERVED {
            let shape = Shape::new(width, height);
            assert_eq!(
                shape.data_banks([3, 3]),
                data_bank,
                "data_bank for {width}x{height}"
            );
            assert_eq!(
                shape.data_banks([3, 3]) + shape.weight_banks([3, 3]),
                12,
                "bank split for {width}x{height} must cover all 12 CBUF banks"
            );
        }
    }

    #[test]
    fn wider_shapes_scale_the_geometry_registers() {
        // Formulas validated against 212 C3 stride-1 programs from 35 captures.
        // 256 wide caps a tile at 44 input rows by the vendor's own CBUF
        // rule (11 data banks x 1024 pixels / 256), so 64 rows need two.
        let shape = Shape::new(256, 64);
        assert_eq!(shape.max_tile_input_rows([3, 3]), 44);
        assert_eq!(shape.min_tiles([3, 3]), 2);

        let split = Tile::split(shape, [3, 3], 2);
        let tile = split[0];
        let program = conv_2d_tile(shape, [3, 3], &tile);

        assert_eq!(
            value_of::<CnaDataSize0>(&program),
            (256 << 16) | tile.in_rows
        );
        assert_eq!(value_of::<CnaCbufCon1>(&program), 256 * tile.in_rows);
        assert_eq!(value_of::<CnaDataSize3>(&program), 256 * tile.out_rows);
        assert_eq!(
            value_of::<CoreDataoutSize0>(&program),
            ((tile.out_rows - 1) << 16) | 255
        );
        assert_eq!(value_of::<DpuDataCubeHeight>(&program), tile.out_rows - 1);

        // Row strides follow the dense NHWC input and C8 fp16 output.
        assert_eq!(shape.input_row_stride(), 256 * 3 * 2);
        assert_eq!(shape.output_row_stride([3, 3]), 256 * 8 * 2);

        // A three-way split of 64 rows, with the 3x3 halo on continuations.
        let three = Tile::split(shape, [3, 3], 3);
        assert_eq!(
            three.iter().map(|t| t.out_rows).collect::<Vec<_>>(),
            [22, 21, 21]
        );
        assert_eq!(three[1].in_first, 21);
        assert_eq!(three[1].output_offset(shape, [3, 3]), 22 * 256 * 8 * 2);
    }

    #[test]
    fn vendor_capacity_rule_matches_the_width_sweep() {
        // Largest tile the vendor emits at each width, from the width sweep:
        // data_banks * 1024 / width, exact at every measured point.
        for (width, rows) in [(256u32, 32u32), (512, 22), (768, 14), (1024, 11), (1536, 7)] {
            let shape = Shape::new(width, 32);
            assert_eq!(
                shape.max_tile_input_rows([3, 3]),
                rows,
                "max tile rows at {width} wide"
            );
        }
    }

    #[test]
    fn stride_scales_output_geometry_and_halo() {
        // Formulas confirmed on 150 stride-2, -3 and -4 programs.
        let shape = Shape::with_stride(128, 64, 2);
        assert_eq!(shape.output_width([3, 3]), 64);
        assert_eq!(shape.output_height([3, 3]), 32);
        assert_eq!(shape.output_row_stride([3, 3]), 64 * 8 * 2);

        // A two-way split of the 32 output rows. The continuation tile
        // projects back through the stride: 16 * 2 - 1 = 31.
        let split = Tile::split(shape, [3, 3], 2);
        assert_eq!(
            split.iter().map(|t| t.out_rows).collect::<Vec<_>>(),
            [16, 16]
        );
        assert_eq!(split[0].in_first, 0);
        assert_eq!(split[0].in_rows, 32);
        assert_eq!(split[0].pad_top, 1);
        assert_eq!(split[1].in_first, 31);
        assert_eq!(split[1].in_rows, 33);
        assert_eq!(split[1].pad_top, 0);

        // The stride reaches CNA_CONV_CON3, and output-side registers carry
        // output geometry rather than input.
        let program = conv_2d_tile(shape, [3, 3], &split[0]);
        assert_eq!(value_of::<CnaConvCon3>(&program), (2 << 3) | 2);
        assert_eq!(value_of::<CnaDataSize2>(&program), 64);
        // Raw word: the DST_SURF_STRIDE field is shifted 4, so the encoded
        // word is sixteen times the 64 * 32 field value.
        assert_eq!(value_of::<DpuDstSurfStride>(&program), 64 * 32 * 16);

        // Stride 1 is unchanged.
        let flat = Shape::new(128, 64);
        assert_eq!(flat.output_height([3, 3]), 64);
        assert_eq!(Tile::split(flat, [3, 3], 2)[1].in_first, 31);
    }

    #[test]
    fn channel_layout_boundary_is_half_an_atom() {
        // Cin 4 is the last dense case (8 bytes); Cin 5 is already surfaces.
        // Measured directly: line_stride/width is 1.00 up to Cin 4 and 4.00
        // from Cin 5 onward.
        for cin in 1..=4 {
            assert_eq!(
                Shape::with_channels(32, 32, 1, cin).layout(),
                FeatureLayout::Dense,
                "Cin {cin}"
            );
        }
        for cin in [5, 6, 7, 8, 16, 80] {
            assert_eq!(
                Shape::with_channels(32, 32, 1, cin).layout(),
                FeatureLayout::Surfaces,
                "Cin {cin}"
            );
        }
    }

    #[test]
    fn channel_padding_matches_the_fill_in_sweep() {
        // (Cin, datain_channel, weight_channels) read out of the captures.
        // The two exceptions are 3 atoms, where the fields disagree, and
        // 7 atoms, where both round up.
        const OBSERVED: [(u32, u32, u32); 16] = [
            (3, 8, 8),
            (4, 8, 8),
            (5, 8, 8),
            (8, 8, 8),
            (9, 16, 16),
            (12, 16, 16),
            (16, 16, 16),
            (20, 24, 32),
            (24, 24, 32),
            (28, 32, 32),
            (32, 32, 32),
            (36, 40, 40),
            (40, 40, 40),
            (48, 48, 48),
            (56, 64, 64),
            (64, 64, 64),
        ];
        for (cin, padded, weights) in OBSERVED {
            let shape = Shape::with_channels(32, 32, 1, cin);
            assert_eq!(shape.padded_channels(), padded, "datain_channel Cin {cin}");
            assert_eq!(
                shape.weight_channels(),
                weights,
                "weight channels Cin {cin}"
            );
        }
        // 72 and 80 are unpadded, confirming this is not a power-of-two rule.
        assert_eq!(Shape::with_channels(32, 32, 1, 72).weight_channels(), 72);
        assert_eq!(Shape::with_channels(32, 32, 1, 80).weight_channels(), 80);
    }

    #[test]
    fn argb_input_mode_follows_the_layout() {
        // CNA_CONV_CON1 across the channel sweep: dense programs the ARGB
        // image path with nonalign_dma and group_line_off set, surfaces
        // clear all three. Cin 3 -> 10/1/1, Cin 4 -> 11/1/1, Cin >= 5 ->
        // 0/0/0. Leaving these at the C3 values made every channel count
        // read as three channels.
        const ARGB_IN: u32 = 0x0000_f000;
        const NONALIGN_DMA: u32 = 0x4000_0000;
        const GROUP_LINE_OFF: u32 = 0x2000_0000;
        let field = |program: &[RegCmd], mask: u32| {
            (first_value_of::<CnaConvCon1>(program) & mask) >> (mask.trailing_zeros())
        };

        for (cin, argb) in [(3u32, 10u32), (4, 11)] {
            let shape = Shape::with_channels(32, 32, 1, cin);
            let program = conv_2d_tile(shape, [3, 3], &Tile::whole(shape, [3, 3]));
            assert_eq!(field(&program, ARGB_IN), argb, "argb_in at Cin {cin}");
            assert_eq!(field(&program, NONALIGN_DMA), 1, "nonalign at Cin {cin}");
            assert_eq!(
                field(&program, GROUP_LINE_OFF),
                1,
                "group_line at Cin {cin}"
            );
        }
        for cin in [5u32, 8, 16, 24, 64] {
            let shape = Shape::with_channels(32, 32, 1, cin);
            let program = conv_2d_tile(shape, [3, 3], &Tile::whole(shape, [3, 3]));
            assert_eq!(field(&program, ARGB_IN), 0, "argb_in at Cin {cin}");
            assert_eq!(field(&program, NONALIGN_DMA), 0, "nonalign at Cin {cin}");
            assert_eq!(
                field(&program, GROUP_LINE_OFF),
                0,
                "group_line at Cin {cin}"
            );
        }
    }

    #[test]
    fn channel_bank_split_matches_the_sweep() {
        // At 32x32 the surface rule reduces to ceil(weight_atoms / 2).
        for (cin, banks) in [
            (8u32, 1u32),
            (16, 1),
            (20, 2),
            (32, 2),
            (40, 3),
            (48, 3),
            (56, 4),
            (64, 4),
            (72, 5),
            (80, 5),
        ] {
            assert_eq!(
                Shape::with_channels(32, 32, 1, cin).data_banks([3, 3]),
                banks,
                "data_bank at Cin {cin}"
            );
        }
        // Dense stays on the pixel rule it was derived with.
        assert_eq!(Shape::with_channels(128, 32, 1, 3).data_banks([3, 3]), 4);
        assert_eq!(Shape::with_channels(128, 32, 1, 4).data_banks([3, 3]), 4);
        // ...and the surface rule diverges immediately at the boundary.
        assert_eq!(Shape::with_channels(128, 32, 1, 8).data_banks([3, 3]), 2);
    }

    #[test]
    fn multi_channel_registers_match_the_captures() {
        // conv-w128-h32-k3-s1-ci16-co8: line_stride 512, surf_stride 3584,
        // data_entries 64, datain_channel 16, data_bank 4.
        let shape = Shape::with_channels(128, 32, 1, 16);
        let program = conv_2d_tile(shape, [3, 3], &Tile::whole(shape, [3, 3]));
        assert_eq!(value_of::<CnaDmaCon1>(&program), 512);
        assert_eq!(value_of::<CnaDmaCon2>(&program), 3584);
        assert_eq!(value_of::<CnaCbufCon1>(&program), 64);
        assert_eq!(value_of::<CnaDataSize1>(&program) & 0xffff, 16);
        assert_eq!(shape.data_banks([3, 3]), 4);
        // Weight footprint follows the weight padding, not datain_channel.
        assert_eq!(value_of::<CnaWeightSize1>(&program), 9 * 16 * 2);
        assert_eq!(value_of::<CnaWeightSize0>(&program), 9 * 16 * 2 * 8);

        // Cin 24 is the case where the two paddings disagree: datain_channel
        // stays 24 while the coefficients occupy 32 channels.
        let odd = Shape::with_channels(32, 32, 1, 24);
        let odd_program = conv_2d_tile(odd, [3, 3], &Tile::whole(odd, [3, 3]));
        assert_eq!(value_of::<CnaDataSize1>(&odd_program) & 0xffff, 24);
        assert_eq!(value_of::<CnaWeightSize1>(&odd_program), 9 * 32 * 2);
        assert_eq!(value_of::<CnaCbufCon1>(&odd_program), 32);
    }

    #[test]
    #[should_panic(expected = "input channels must be")]
    fn rejects_channels_beyond_the_validated_range() {
        let _ = Shape::with_channels(32, 32, 1, 96);
    }

    #[test]
    fn output_channels_pad_to_whole_granules() {
        // Every value the corpus covers. Unlike the input padding this is a
        // rule rather than a table: no exceptions at 20, 24, 40, 56 or 72,
        // the values where the input padding needed them.
        for (out_channels, padded) in [
            (1u32, 16u32),
            (2, 16),
            (8, 16),
            (14, 16),
            (16, 16),
            (20, 32),
            (24, 32),
            (28, 32),
            (32, 32),
            (40, 48),
            (48, 48),
            (56, 64),
            (64, 64),
            (72, 80),
            (80, 80),
            (96, 96),
            (128, 128),
            (256, 256),
            (512, 512),
        ] {
            let shape = Shape::with_out_channels(32, 32, 1, 3, out_channels);
            assert_eq!(
                shape.padded_out_channels(),
                padded,
                "padded output channels at Cout {out_channels}"
            );
        }
    }

    #[test]
    fn output_channel_registers_match_the_captures() {
        // conv-w32-h32-k3-s1-ci3-co40, single-core plan: weight_bytes 5760,
        // weight_kernels 40, orig_channel 39, and the four padded-count
        // registers 47 -- Cout 40 rounds to 48. Cout 40 is chosen because
        // every one of those five numbers is distinct there.
        let shape = Shape::with_out_channels(32, 32, 1, 3, 40);
        let program = conv_2d_tile(shape, [3, 3], &Tile::whole(shape, [3, 3]));
        assert_eq!(value_of::<CnaWeightSize0>(&program), 5760);
        assert_eq!(value_of::<CnaWeightSize1>(&program), 144);
        assert_eq!(value_of::<CnaWeightSize2>(&program), 50_528_296);
        assert_eq!(value_of::<DpuDataCubeChannel>(&program), 2_555_951);
        assert_eq!(value_of::<DpuWdmaSize0>(&program), 47);
        assert_eq!(value_of::<CoreDataoutSize1>(&program), 47);

        // The output is NC1HWC2 regardless of Cout: the channel count sets
        // how many 8-channel surfaces there are, not how wide a row is, so
        // both output strides are the same as the Cout 8 capture's.
        assert_eq!(value_of::<DpuDstSurfStride>(&program), 32 * 32 * 16);

        // Cout 9 is the smallest value where the true and padded counts
        // differ by more than the granule rounding of a multiple of 8.
        let odd = Shape::with_out_channels(32, 32, 1, 3, 9);
        let odd_program = conv_2d_tile(odd, [3, 3], &Tile::whole(odd, [3, 3]));
        assert_eq!(value_of::<CnaWeightSize0>(&odd_program), 1296);
        assert_eq!(value_of::<DpuDataCubeChannel>(&odd_program), 524_303);
        assert_eq!(value_of::<DpuWdmaSize0>(&odd_program), 15);
    }

    #[test]
    fn output_channels_reach_the_bank_split_only_through_the_weight_footprint() {
        // The one place in the corpus where Cout moves the CBUF allocation.
        // At 256x32 with Cin 32 the feature data wants 16 banks and cannot
        // have them, so whatever the weights take comes straight off it.
        let narrow = Shape::with_out_channels(256, 32, 1, 32, 16);
        assert_eq!(narrow.weight_bytes([3, 3]), 9216); // one bank
        assert_eq!(narrow.data_banks([3, 3]), 11);
        assert_eq!(narrow.weight_banks([3, 3]), 1);

        let wide = Shape::with_out_channels(256, 32, 1, 32, 64);
        assert_eq!(wide.weight_bytes([3, 3]), 36864); // two banks
        assert_eq!(wide.data_banks([3, 3]), 10);
        assert_eq!(wide.weight_banks([3, 3]), 2);

        // Where the data is *not* over budget, a bigger kernel set changes
        // nothing: the weights are taking slack that was already theirs.
        for out_channels in [8u32, 16, 64, 128] {
            let shape = Shape::with_out_channels(32, 32, 1, 3, out_channels);
            assert_eq!(
                shape.data_banks([3, 3]),
                1,
                "32x32 Cin 3 keeps one data bank at Cout {out_channels}"
            );
        }

        // And when the weights are the larger claim they are the ones cut
        // short: 589824 bytes want 18 banks, more than the CBUF holds, but
        // the feature data still gets the 4 it asked for.
        let huge = Shape::with_out_channels(32, 32, 1, 64, 512);
        assert_eq!(huge.weight_bytes([3, 3]), 589_824);
        assert_eq!(huge.data_banks([3, 3]), 4);
        assert_eq!(huge.weight_banks([3, 3]), 8);
    }

    #[test]
    #[should_panic(expected = "output channels must be")]
    fn rejects_output_channels_beyond_the_validated_range() {
        let _ = Shape::with_out_channels(32, 32, 1, 3, 513);
    }

    #[test]
    fn tile_capacity_charges_surfaces_per_atom() {
        // A surface pixel costs `weight_atoms` times what a dense one does,
        // so the same bank allocation carries proportionally fewer rows.
        // Charging one atom per pixel here let a 256x32 Cin 32 tile claim 44
        // rows against a real capacity of 22, and the hardware quietly
        // dropped the rows past the end -- the taps below the cut came back
        // as if they were off the bottom of the image.
        let narrow = Shape::with_out_channels(256, 32, 1, 32, 16);
        assert_eq!(narrow.data_banks([3, 3]), 11);
        assert_eq!(narrow.max_tile_input_rows([3, 3]), 22);
        assert_eq!(narrow.min_tiles([3, 3]), 2);

        // The same shape with a kernel set big enough to take a second bank
        // for coefficients loses two rows of capacity with it. Both numbers
        // are the vendor's own split: 22/12 rows here, 20/14 below.
        let wide = Shape::with_out_channels(256, 32, 1, 32, 64);
        assert_eq!(wide.data_banks([3, 3]), 10);
        assert_eq!(wide.max_tile_input_rows([3, 3]), 20);
        assert_eq!(wide.min_tiles([3, 3]), 2);

        // The dense regime is unchanged: one atom per pixel is correct there,
        // which is why the Cin 3 width sweep never saw this.
        for (width, rows) in [(256u32, 32u32), (512, 22), (768, 14), (1024, 11), (1536, 7)] {
            assert_eq!(
                Shape::new(width, 32).max_tile_input_rows([3, 3]),
                rows,
                "dense capacity at {width} wide"
            );
        }
    }

    #[test]
    fn tiles_of_a_plan_write_disjoint_output_ranges() {
        for tiles in 1..=3 {
            let split = Tile::split(Shape::CAPTURED, [3, 3], tiles);
            let mut covered = vec![0u32; IMAGE_HEIGHT as usize];
            for tile in &split {
                for row in tile.out_first..tile.out_first + tile.out_rows {
                    covered[row as usize] += 1;
                }
            }
            assert!(
                covered.iter().all(|&n| n == 1),
                "{tiles}-tile plan does not partition the output exactly"
            );
        }
    }
}
