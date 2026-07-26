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
//! Supported case: `Cin=3` dense NHWC, `Cout=8`, stride 1, 1x1 or 3x3
//! kernels. Wider channel counts move the feature map onto multiple NC1HWC2
//! surfaces and change the row strides; nothing here covers that.
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

/// Input channels this builder supports. A C3 fp16 pixel is 6 bytes and
/// occupies one 16-byte feature atom once padded to the CNA's C8 task width,
/// which is what keeps the row strides below single-surface.
pub const INPUT_CHANNELS: u32 = 3;

/// Output channels this builder supports: one fp16 kernel group.
pub const OUTPUT_CHANNELS: u32 = 8;

/// Total CBUF banks the CNA partitions between feature data and weights.
const CBUF_BANKS: u32 = 12;

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
}

impl Shape {
    /// The `32x32` geometry of the original vendor captures.
    pub const CAPTURED: Shape = Shape {
        width: IMAGE_WIDTH,
        height: IMAGE_HEIGHT,
    };

    pub fn new(width: u32, height: u32) -> Shape {
        assert!(
            width > 0 && height > 0,
            "convolution extents must be nonzero"
        );
        Shape { width, height }
    }

    /// Byte stride of one dense NHWC input row.
    pub fn input_row_stride(&self) -> u32 {
        self.width * INPUT_CHANNELS * 2
    }

    /// Byte stride of one output row.
    pub fn output_row_stride(&self) -> u32 {
        self.width * OUTPUT_CHANNELS * 2
    }

    /// CBUF banks the vendor assigns to feature data.
    ///
    /// Derived from 134 captured programs across 11 distinct `(width,
    /// height)` shapes, every one of which also satisfies
    /// `data_banks + weight_banks == 12`. The cap at 11 leaves one bank for
    /// weights, which no capture goes below.
    pub fn data_banks(&self) -> u32 {
        (self.width * self.height)
            .div_ceil(PIXELS_PER_BANK_STEP)
            .clamp(1, CBUF_BANKS - 1)
    }

    /// CBUF banks the vendor assigns to weights.
    pub fn weight_banks(&self) -> u32 {
        CBUF_BANKS - self.data_banks()
    }

    /// Most input rows one program may read, bounded by the 14-bit
    /// `CNA_CBUF_CON1.data_entries` field.
    pub fn max_tile_input_rows(&self) -> u32 {
        MAX_DATA_ENTRIES / self.width
    }

    /// Fewest tiles this shape must be split into to stay encodable.
    ///
    /// A tile producing `r` output rows reads up to `r + 2 * padding` input
    /// rows once its halo is counted, so the padding is charged here rather
    /// than discovered as an overflow inside the builder.
    pub fn min_tiles(&self, kernels: Kernels) -> u32 {
        let halo = 2 * kernel_programming(kernels).padding;
        let budget = self.max_tile_input_rows().saturating_sub(halo).max(1);
        self.height.div_ceil(budget)
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
        assert!(
            (1..=shape.height).contains(&tiles),
            "tile count must be between 1 and the {} image rows",
            shape.height
        );
        let padding = kernel_programming(kernels).padding;
        let base = shape.height / tiles;
        let remainder = shape.height % tiles;

        let mut out = Vec::with_capacity(tiles as usize);
        let mut out_first: u32 = 0;
        for index in 0..tiles {
            let out_rows = base + u32::from(index < remainder);
            let in_first = out_first.saturating_sub(padding);
            let in_last = (out_first + out_rows - 1 + padding).min(shape.height - 1);
            out.push(Tile {
                out_first,
                out_rows,
                in_first,
                in_rows: in_last - in_first + 1,
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
    pub fn output_offset(&self, shape: Shape) -> u32 {
        self.out_first * shape.output_row_stride()
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
    const TASK_INPUT_CHANNELS: u32 = 8;
    const TASK_OUTPUT_CHANNELS: u32 = 16;
    const FP16_BYTES: u32 = 2;

    let width = shape.width;
    let height = shape.height;
    let weight_banks = shape.weight_banks();
    let data_banks = shape.data_banks();

    assert!(
        tile.out_rows > 0 && tile.out_first + tile.out_rows <= height,
        "tile output rows {}..{} fall outside the {height}-row image",
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
    let weight_bytes_per_kernel = kernel.size * kernel.size * TASK_INPUT_CHANNELS * FP16_BYTES;
    let weight_bytes = weight_bytes_per_kernel * OUTPUT_CHANNELS;
    let mut commands = Vec::with_capacity(136);

    // CNA preamble, followed by the DPU/DPU_RDMA ping-pong pointers.
    let mut cbuf_con0 = Register::<CnaCbufCon0>::new();
    cbuf_con0
        .weight_bank(Bits::new(weight_banks))
        .data_bank(Bits::new(data_banks));
    commands.push(cbuf_con0.build());
    commands.push(zero::<CnaDcompRegnum>());
    commands.push(zero::<CnaDcompCtrl>());

    let mut conv_con1 = Register::<CnaConvCon1>::new();
    conv_con1
        .nonalign_dma(Bits::new(1))
        .group_line_off(Bits::new(1))
        .argb_in(ArgbInputMode::ThreeChannels.into())
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
            .conv_x_stride(Bits::new(1))
            .conv_y_stride(Bits::new(1))
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
            .datain_channel_real(Bits::new(INPUT_CHANNELS - 1))
            .datain_channel(Bits::new(TASK_INPUT_CHANNELS))
            .build(),
    );
    commands.push(
        Register::<CnaDataSize2>::new()
            .dataout_width(Bits::new(width))
            .build(),
    );
    commands.push(
        Register::<CnaDataSize3>::new()
            .dataout_atomics(Bits::new(width * tile.out_rows))
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
            .weight_kernels(Bits::new(OUTPUT_CHANNELS))
            .build(),
    );
    commands.push(cbuf_con0.build());
    commands.push(
        Register::<CnaCbufCon1>::new()
            .data_entries(Bits::new(width * tile.in_rows))
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
            .line_stride(Bits::new(width))
            .build(),
    );
    commands.push(
        Register::<CnaDmaCon2>::new()
            .surf_stride(Bits::new(width * (height - 1)))
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
            .dma_channel(Bits::new(TASK_INPUT_CHANNELS))
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
            .dataout_width(Bits::new(width - 1))
            .dataout_height(Bits::new(tile.out_rows - 1))
            .build(),
    );
    commands.push(
        Register::<CoreDataoutSize1>::new()
            .dataout_channel(Bits::new(TASK_OUTPUT_CHANNELS - 1))
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
            .dst_base_addr(Bits::new(tile.output_offset(shape)))
            .build(),
    );
    commands.push(
        Register::<DpuDstSurfStride>::new()
            .dst_surf_stride(Bits::new(width * height))
            .build(),
    );
    commands.push(
        Register::<DpuDataCubeWidth>::new()
            .width(Bits::new(width - 1))
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
            .orig_channel(Bits::new(OUTPUT_CHANNELS - 1))
            .channel(Bits::new(TASK_OUTPUT_CHANNELS - 1))
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
            .channel_wdma(Bits::new(TASK_OUTPUT_CHANNELS - 1))
            .build(),
    );
    commands.push(
        Register::<DpuWdmaSize1>::new()
            .height_wdma(Bits::new(tile.out_rows - 1))
            .width_wdma(Bits::new(width - 1))
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
            .surf_add(Bits::new(width * height * FP16_BYTES))
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
            .width(Bits::new(width - 1))
            .build(),
    );
    commands.push(
        Register::<DpuRdmaDataCubeHeight>::new()
            .height(Bits::new(tile.out_rows - 1))
            .build(),
    );
    commands.push(
        Register::<DpuRdmaDataCubeChannel>::new()
            .channel(Bits::new(TASK_OUTPUT_CHANNELS - 1))
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
                .map(|t| t.output_offset(Shape::CAPTURED))
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
                shape.data_banks(),
                data_bank,
                "data_bank for {width}x{height}"
            );
            assert_eq!(
                shape.data_banks() + shape.weight_banks(),
                12,
                "bank split for {width}x{height} must cover all 12 CBUF banks"
            );
        }
    }

    #[test]
    fn wider_shapes_scale_the_geometry_registers() {
        // Formulas validated against 212 C3 stride-1 programs from 35 captures.
        // 256 wide caps a tile at 63 input rows, so a 64-row image needs two.
        let shape = Shape::new(256, 64);
        assert_eq!(shape.max_tile_input_rows(), 63);
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
        assert_eq!(shape.output_row_stride(), 256 * 8 * 2);

        // A three-way split of 64 rows, with the 3x3 halo on continuations.
        let three = Tile::split(shape, [3, 3], 3);
        assert_eq!(
            three.iter().map(|t| t.out_rows).collect::<Vec<_>>(),
            [22, 21, 21]
        );
        assert_eq!(three[1].in_first, 21);
        assert_eq!(three[1].output_offset(shape), 22 * 256 * 8 * 2);
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
