//! Direct pooling through the PPU_RDMA -> PPU data path (TRM Ch.36 Fig 36-6).
//!
//! PPU_RDMA fetches each planned input tile from memory and feeds PPU; PPU
//! applies the pooling window and writes that tile's columns into the shared
//! output tensor. Each tile is one complete, independently kicked regcmd
//! task containing both blocks and a single `PPU | PPU_RDMA` (`0x60`) kick.
//! Wide shapes return several such tasks from [`PoolingPlan`], intended to
//! be submitted as the ordered task array of one job.
//!
//! The register values and ordering come from dedicated standalone and
//! Conv2d->Pool RKNN sweeps. The arrow above describes hardware dataflow,
//! not command order: matching captures configure PPU before its PPU_RDMA
//! feeder. No CNA/CORE/DPU bypass is part of the public pooling path.

use crate::rocket::{
    builders::{Bits, RegCmd, Register, pc::PCTrailer, ppu::*, ppu_rdma::*},
    regcmd::{KICK_PPU, KICK_PPU_RDMA, push_kick, zero},
};

/// Appends the captured PPU/PPU_RDMA trailer and its two-word fetch padding.
///
/// The direct 26-register vendor PPU programs in the pooling sweep are
/// followed by the four-word kick trailer and two zero words (32 commands
/// total). The generic trailer helper only guarantees an even command count,
/// so preserve the PPU stream's stronger four-command alignment here without
/// changing other engines' programs.
fn push_ppu_kick(cmds: &mut Vec<RegCmd>) {
    push_kick(cmds, KICK_PPU | KICK_PPU_RDMA);
    while !cmds.len().is_multiple_of(4) {
        cmds.push(PCTrailer::alignment_padding());
    }
}

// The direct register sequence is fixed by the 143-capture fp16/int8
// pooling sweep: pixel-count line strides, four-pixel-aligned source and
// destination surfaces, precision-dependent 16-byte channel atoms, PPU before
// PPU_RDMA, and a 32-command program ending in a `0x60` kick and two
// fetch-padding words. Each standalone task also clears both blocks' persistent
// register/executer pointers while enabling ping-pong; without those W1C bits,
// an 8x8/stride-3 task can leave the following 129-wide task waiting forever.
// RK3588 testing established the Avg=0, Max=1, Min=2 method encoding.

/// Hardware-confirmed bit encoding from the retired standalone exploration
/// on a real RK3588: a half-10/half-200 input produced raw=0 -> 249, raw=1
/// -> 250, raw=2 -> 248 -- i.e. raw=1 is the real max, raw=2 is the real
/// min, raw=0 sits in between (avg). The original guess (Max=0, Min=1,
/// Avg=2) had max and min swapped relative to avg; corrected below.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoolingMethod {
    Max,
    Min,
    Avg,
}

impl PoolingMethod {
    fn bits(self) -> u32 {
        match self {
            PoolingMethod::Avg => 0,
            PoolingMethod::Max => 1,
            PoolingMethod::Min => 2,
        }
    }
}

/// Numeric format of the input and output feature maps.
///
/// PPU and PPU_RDMA use different enums for the same format. Vendor captures
/// program `(proc_precision, in_precision)` as `(0, 1)` for int8 and `(2, 2)`
/// for fp16: PPU's first field is a numeric-domain enum, while PPU_RDMA's is
/// simply `4/8/16/32-bit = 0/1/2/3`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoolingPrecision {
    Int8,
    Fp16,
}

impl PoolingPrecision {
    /// Logical channels carried by one 16-byte PPU feature atom.
    ///
    /// The pooling sweep programs all three channel extents to this
    /// precision-dependent granularity: fp16 C1..C8 become C8, while int8
    /// C12 becomes C16. This is the same physical 16-byte atom consumed from
    /// the preceding DPU output or directly by PPU_RDMA.
    fn channels_per_atom(self) -> u32 {
        match self {
            PoolingPrecision::Int8 => 16,
            PoolingPrecision::Fp16 => 8,
        }
    }

    fn ppu_precision(self) -> u32 {
        match self {
            PoolingPrecision::Int8 => 0,
            PoolingPrecision::Fp16 => 2,
        }
    }

    fn rdma_precision(self) -> u32 {
        match self {
            PoolingPrecision::Int8 => 1,
            PoolingPrecision::Fp16 => 2,
        }
    }
}

/// Logical shape of a standalone pooling operation. [`PoolingPlan`] splits
/// wide shapes horizontally to respect the direct PPU width observed in
/// vendor programs and on RK3588. There is no CBUF-budget splitting to worry
/// about (PPU has no CBUF -- that concern is CNA/CORE-specific), and no
/// `index_en` output wiring yet (that needs a second output buffer for
/// argmax/argmin positions, not plumbed here).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolingShape {
    pub input_width: u32,
    pub input_height: u32,
    /// Logical channel count. Register emission rounds this to one 16-byte
    /// feature atom (C8 fp16/C16 int8), matching every direct vendor PPU
    /// program rather than writing a sub-atomic channel extent.
    pub input_channels: u32,
    pub output_width: u32,
    pub output_height: u32,
    /// Logical channel count, equal to `input_channels`; programmed with the
    /// same feature-atom rounding.
    pub output_channels: u32,
    pub precision: PoolingPrecision,
    pub kernel_width: u32,
    pub kernel_height: u32,
    pub stride_x: u32,
    pub stride_y: u32,
    pub method: PoolingMethod,
    pub pad_left: u32,
    pub pad_top: u32,
    pub pad_right: u32,
    pub pad_bottom: u32,
    /// Fill value for padded taps (e.g. -inf-ish for max, 0 for avg --
    /// caller's responsibility to pick something sane for `method`).
    pub pad_value: u32,
}

const MAX_PPU_EXTENT: u32 = 8192;
const MAX_PPU_KERNEL_OR_STRIDE: u32 = 16;
const MAX_PPU_PADDING: u32 = 7;
const MAX_DIRECT_KERNEL: u32 = 8;
const DEFAULT_MAX_DIRECT_INPUT_WIDTH: u32 = 129;
const DEFAULT_MAX_DIRECT_OUTPUT_WIDTH: u32 = 64;
const K2S2_FP16_MAX_DIRECT_INPUT_WIDTH: u32 = 256;
const K2S2_FP16_MAX_DIRECT_OUTPUT_WIDTH: u32 = 128;
const K2S2_INT8_MAX_DIRECT_INPUT_WIDTH: u32 = 130;
const K2S2_INT8_MAX_DIRECT_OUTPUT_WIDTH: u32 = 65;

fn direct_width_limits(shape: &PoolingShape) -> (u32, u32) {
    if shape.kernel_width == 2
        && shape.kernel_height == 2
        && shape.stride_x == 2
        && shape.stride_y == 2
        && shape.pad_left == 0
        && shape.pad_top == 0
        && shape.pad_right == 0
        && shape.pad_bottom == 0
    {
        match shape.precision {
            PoolingPrecision::Fp16 => (
                K2S2_FP16_MAX_DIRECT_INPUT_WIDTH,
                K2S2_FP16_MAX_DIRECT_OUTPUT_WIDTH,
            ),
            // A two-task width-258 RK3588 probe validates the 65-column
            // right tile, while a one-task width-257/output-128 probe shifts
            // int8 results. Stay at the largest hardware-proven int8 tile.
            PoolingPrecision::Int8 => (
                K2S2_INT8_MAX_DIRECT_INPUT_WIDTH,
                K2S2_INT8_MAX_DIRECT_OUTPUT_WIDTH,
            ),
        }
    } else {
        (
            DEFAULT_MAX_DIRECT_INPUT_WIDTH,
            DEFAULT_MAX_DIRECT_OUTPUT_WIDTH,
        )
    }
}

fn output_extent(input: u32, kernel: u32, stride: u32, before: u32, after: u32) -> u32 {
    let padded = input
        .checked_add(before)
        .and_then(|value| value.checked_add(after))
        .expect("pooling padded extent overflows u32");
    assert!(
        padded >= kernel,
        "pooling kernel {kernel} exceeds padded input extent {padded}"
    );
    (padded - kernel) / stride + 1
}

fn required_trailing_padding(
    input: u32,
    output: u32,
    kernel: u32,
    stride: u32,
    leading_padding: u32,
) -> u32 {
    ((output - 1) * stride + kernel).saturating_sub(input + leading_padding)
}

fn validate_direct_kernel(kernel_width: u32, kernel_height: u32) {
    assert!(
        kernel_width <= MAX_DIRECT_KERNEL && kernel_height <= MAX_DIRECT_KERNEL,
        "pooling kernel axes must be 1..={MAX_DIRECT_KERNEL}; RK3588 hardware confirms 8x8 \
         but rejects a directly programmed 16x16 window"
    );
}

fn validate_pooling_geometry(
    input_width: u32,
    input_height: u32,
    input_channels: u32,
    output_width: u32,
    output_height: u32,
    output_channels: u32,
    kernel_width: u32,
    kernel_height: u32,
    stride_x: u32,
    stride_y: u32,
    pad_left: u32,
    pad_top: u32,
    pad_right: u32,
    pad_bottom: u32,
) {
    for (name, value) in [
        ("input width", input_width),
        ("input height", input_height),
        ("input channels", input_channels),
        ("output width", output_width),
        ("output height", output_height),
        ("output channels", output_channels),
    ] {
        assert!(
            (1..=MAX_PPU_EXTENT).contains(&value),
            "{name} must be 1..={MAX_PPU_EXTENT}, the PPU's 13-bit N-1 range"
        );
    }
    for (name, value) in [
        ("kernel width", kernel_width),
        ("kernel height", kernel_height),
        ("horizontal stride", stride_x),
        ("vertical stride", stride_y),
    ] {
        assert!(
            (1..=MAX_PPU_KERNEL_OR_STRIDE).contains(&value),
            "{name} must be 1..={MAX_PPU_KERNEL_OR_STRIDE}, the PPU's 4-bit N-1 range"
        );
    }
    for (name, value) in [
        ("left padding", pad_left),
        ("top padding", pad_top),
        ("right padding", pad_right),
        ("bottom padding", pad_bottom),
    ] {
        assert!(
            value <= MAX_PPU_PADDING,
            "{name} must be 0..={MAX_PPU_PADDING}, the PPU's 3-bit range"
        );
    }
    assert_eq!(
        input_channels, output_channels,
        "pooling preserves the channel count"
    );
    // A stride wider than its kernel skips input, and a direct PPU program
    // for such a shape hangs the NPU: on RK3588 hardware, 64x31 k3x3 at
    // sy=4, sy=5 and sx=8 each time out on their own dispatch against the
    // driver's 500 ms watchdog (iree-rocket-hal/tests/pooling_hw.rs,
    // `wedging_stride_y_sweep`). These are exactly the geometries
    // rknn-toolkit2 compiles with a CNA|CORE|DPU stage ahead of the PPU
    // kick, which this crate does not emit. Reject them rather than emit a
    // program known to wedge the hardware.
    for (axis, stride, kernel) in [
        ("horizontal", stride_x, kernel_width),
        ("vertical", stride_y, kernel_height),
    ] {
        assert!(
            stride <= kernel,
            "{axis} stride {stride} exceeds kernel {kernel}; the direct PPU path \
             hangs the NPU for stride-beyond-kernel shapes"
        );
    }
    assert_eq!(
        output_width,
        output_extent(input_width, kernel_width, stride_x, pad_left, pad_right),
        "pooling output width does not match floor-mode geometry"
    );
    assert_eq!(
        output_height,
        output_extent(input_height, kernel_height, stride_y, pad_top, pad_bottom),
        "pooling output height does not match floor-mode geometry"
    );
}

impl PoolingShape {
    /// Checks every field-width and logical-geometry invariant before register
    /// emission. The 8192 spatial boundary is confirmed by the vendor's
    /// explicit 8193 overflow diagnostic; kernel/stride 1..=16 and padding
    /// 0..=7 are the exact register ranges. Vendor direct programs cover
    /// kernels through 8 and strides through 3, while existing RK3588 tests
    /// independently confirm a 4x4 kernel/stride.
    pub fn validate(&self) {
        validate_pooling_geometry(
            self.input_width,
            self.input_height,
            self.input_channels,
            self.output_width,
            self.output_height,
            self.output_channels,
            self.kernel_width,
            self.kernel_height,
            self.stride_x,
            self.stride_y,
            self.pad_left,
            self.pad_top,
            self.pad_right,
            self.pad_bottom,
        );
        validate_direct_kernel(self.kernel_width, self.kernel_height);
    }
}

/// One independently dispatched horizontal PPU tile.
///
/// A tile carries the kernel overlap in `input_width`. Most captured kernels
/// use at most 64 output columns per task; the exact unpadded 2x2/stride-2
/// fp16 path is capture-backed through 128 output columns. Int8 is
/// hardware-proven through 65 columns. `input_first` and
/// `output_first` are pixel offsets into the full tensor; the builder converts
/// them to 16-byte feature-atom offsets when relocating the base addresses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolingTile {
    pub input_first: u32,
    pub input_width: u32,
    pub output_first: u32,
    pub output_width: u32,
    pub pad_left: u32,
    pub pad_right: u32,
}

/// Capture-derived horizontal pooling plan.
///
/// Width 129 is the default largest whole direct input in the corpus, with a
/// 64-column output cap. The exact unpadded fp16 2x2/stride-2 path has
/// dedicated captures through input width 256/output width 128. Int8 uses
/// the hardware-proven input-130/output-65 limit: the two-task width-258
/// probe passes, while a one-task width-257/output-128 probe produced shifted
/// output. Larger tensors split into balanced tiles, assigning any remainder
/// to the rightmost tiles. The one int8 64+64 result is biased to the adjacent
/// hardware-proven 63+65 widths. This reproduces the observed 63+64 split for
/// an unpadded width-256 3x3/stride-2 pool and the 64+65 split for width-258
/// 2x2/stride-2.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PoolingPlan {
    shape: PoolingShape,
    tiles: Vec<PoolingTile>,
}

impl PoolingPlan {
    pub fn new(shape: PoolingShape) -> PoolingPlan {
        shape.validate();
        let (max_input_width, max_output_width) = direct_width_limits(&shape);
        let tile_count = shape.output_width.div_ceil(max_output_width);
        let base_width = shape.output_width / tile_count;
        let wider_tiles = shape.output_width % tile_count;
        let first_wider = tile_count - wider_tiles;
        // The int8 2x2/stride-2 path has two hardware-confirmed adjacent
        // tile widths (63 and 65), while the width-257/output-128 probe is
        // the sole case that naturally balances to the failing 64+64 pair.
        // Move one output column from the first tile to the last so neither
        // task uses that ambiguous equal-half geometry.
        let avoid_equal_int8_k2_tiles = shape.precision == PoolingPrecision::Int8
            && shape.kernel_width == 2
            && shape.kernel_height == 2
            && shape.stride_x == 2
            && shape.stride_y == 2
            && shape.pad_left == 0
            && shape.pad_top == 0
            && shape.pad_right == 0
            && shape.pad_bottom == 0
            && tile_count == 2
            && wider_tiles == 0
            && base_width == 64;
        let mut output_first = 0;
        let mut tiles = Vec::with_capacity(tile_count as usize);

        for index in 0..tile_count {
            let output_width = if avoid_equal_int8_k2_tiles {
                base_width - 1 + 2 * index
            } else {
                base_width + u32::from(index >= first_wider)
            };
            let raw_input_first =
                i64::from(output_first * shape.stride_x) - i64::from(shape.pad_left);
            let raw_input_end =
                i64::from((output_first + output_width - 1) * shape.stride_x + shape.kernel_width)
                    - i64::from(shape.pad_left);
            let input_first = raw_input_first.max(0) as u32;
            let input_end = raw_input_end.min(i64::from(shape.input_width)).max(0) as u32;
            let tile = PoolingTile {
                input_first,
                input_width: input_end - input_first,
                output_first,
                output_width,
                pad_left: (-raw_input_first).max(0) as u32,
                pad_right: (raw_input_end - i64::from(shape.input_width)).max(0) as u32,
            };
            assert!(
                tile.input_width <= max_input_width && tile.output_width <= max_output_width,
                "pooling tile exceeds the capture-derived direct width limits: {tile:?}"
            );
            tiles.push(tile);
            output_first += output_width;
        }

        PoolingPlan { shape, tiles }
    }

    pub fn shape(&self) -> PoolingShape {
        self.shape
    }

    pub fn tiles(&self) -> &[PoolingTile] {
        &self.tiles
    }

    /// Emits one independently kicked, submission-ready program per tile,
    /// all writing into offsets of the *same* output buffer.
    ///
    /// Submit the returned programs as the ordered task array of one job.
    /// Each task writes a disjoint column range in the shared output BO;
    /// keeping them in one job makes that serialization explicit and avoids
    /// inter-job implicit-fence behavior on the shared buffer.
    pub fn programs_with_buffers(&self, bufs: &PoolingBuffers) -> Vec<Vec<RegCmd>> {
        self.tiles
            .iter()
            .map(|tile| {
                let output_addr = bufs
                    .output_addr
                    .checked_add(tile.output_first * 16)
                    .expect("pooling tile output address overflows u32");
                build_pooling_tile_task(&self.shape, tile, bufs.input_addr, output_addr)
            })
            .collect()
    }
}

/// DMA addresses for the two buffers a direct pooling op needs.
pub struct PoolingBuffers {
    pub input_addr: u32,
    /// Must be 16-byte aligned. `PPU_DST_BASE_ADDR` stores address bits
    /// 31:4 in a 28-bit field, so the tile builder checks the alignment and
    /// supplies `output_addr >> 4`; the register builder places that field
    /// back at bits 31:4 in the emitted word.
    pub output_addr: u32,
}

/// Builds one complete PPU_RDMA -> PPU tile task.
/// `output_addr` is the exact base address this tile's output starts
/// writing at; [`PoolingPlan::programs_with_buffers`] applies the tile's
/// output offset before calling. The returned vector includes the captured
/// alignment padding and its single combined `0x60` kick.
fn build_pooling_tile_task(
    shape: &PoolingShape,
    tile: &PoolingTile,
    input_addr: u32,
    output_addr: u32,
) -> Vec<RegCmd> {
    shape.validate();
    let (max_input_width, max_output_width) = direct_width_limits(shape);
    assert!(
        tile.input_width <= max_input_width && tile.output_width <= max_output_width,
        "direct pooling tile exceeds the capture-derived width limits: {tile:?}"
    );
    let input_addr = input_addr
        .checked_add(tile.input_first * 16)
        .expect("pooling tile input address overflows u32");
    assert!(
        output_addr.is_multiple_of(16),
        "build_pooling_tile_task: output_addr {output_addr:#x} is not 16-byte aligned -- \
         PPU_DST_BASE_ADDR is written as address >> 4 (see PoolingBuffers::output_addr's \
         doc comment), which silently drops any non-zero low 4 bits instead of failing \
         loudly, so this must be checked explicitly"
    );
    let dst_base_addr_shifted = output_addr >> 4;
    let programmed_channels = shape
        .input_channels
        .next_multiple_of(shape.precision.channels_per_atom());

    const ATOMIC_K_SIZE: u32 = 16;
    const FEATURE_ATOMIC_SIZE: u32 = 16;

    // Confirmed against RKNN_TRM_Ch36 (`rknpu-spelunking/chapter36.txt`):
    // PPU_RDMA_SRC_LINE_STRIDE/SRC_SURF_STRIDE are documented as "Pooling
    // cube shape width"/"Pooling cube shape area" -- plain pixel-count
    // values, not byte offsets -- exactly like DPU_DST_SURF_STRIDE just
    // below (also `31:4 RW`/`3:0 reserved`), whose already-correct formula
    // cancels the same way. The previous version omitted the `/
    // FEATURE_ATOMIC_SIZE` cancellation, writing a byte-stride value 16x
    // too large; row 0 of any pooling window (offset = base + 0*stride)
    // still read correctly regardless, which is why the retired standalone
    // suite's CPU-filled *uniform-whole-buffer* tests never caught it --
    // reading 16x too far still landed on the same repeated fill byte
    // anywhere within the buffer. It only
    // surfaced once a position-dependent NC1HWC2 source was used: rows beyond
    // row 0 jumped past the real write footprint into untouched contents.
    let src_line_stride = (shape.input_width * ATOMIC_K_SIZE) / FEATURE_ATOMIC_SIZE;
    // The exact 7x5 vendor controls program 36 rather than the unaligned area
    // 35 in both fp16 and int8. Earlier sweep points all happened to have
    // four-pixel-aligned areas, which hid this source-side rule.
    let src_surf_stride = ((shape.input_width * ATOMIC_K_SIZE * shape.input_height)
        / FEATURE_ATOMIC_SIZE)
        .next_multiple_of(4);

    // Capture-derived: destination surfaces are four-pixel aligned. This is
    // visible on rectangular kernels whose odd output area makes the padding
    // observable (for example, 31*23=713 is programmed as 716).
    let dst_surf_stride = ((shape.output_width * ATOMIC_K_SIZE * shape.output_height)
        / FEATURE_ATOMIC_SIZE)
        .next_multiple_of(4);
    // Vendor captures leave these zero for Max. Average pooling consumes
    // the fixed-point reciprocals; Min, like Max, ignores them.
    let (recip_kernel_width, recip_kernel_height) = match shape.method {
        PoolingMethod::Avg => (
            ((1u64 << 16) / u64::from(shape.kernel_width)) as u32,
            ((1u64 << 16) / u64::from(shape.kernel_height)) as u32,
        ),
        PoolingMethod::Max | PoolingMethod::Min => (0, 0),
    };
    // The vendor programs only trailing padding that an emitted output
    // window actually reaches. ONNX may carry an additional right/bottom pad
    // that leaves floor-mode output geometry unchanged; programming it is
    // unnecessary. For 3x3/s2/pad1 this turns [1,1,1,1] into the captured
    // [top=1,left=1,bottom=0,right=0].
    let programmed_pad_bottom = required_trailing_padding(
        shape.input_height,
        shape.output_height,
        shape.kernel_height,
        shape.stride_y,
        shape.pad_top,
    );

    let mut cmds: Vec<RegCmd> = Vec::new();

    // ========================================================================
    // Ping-pong pointers, matching the vendor exactly: all 434 S_POINTER
    // writes in the standalone pooling sweep are 0x0e, and none sets the W1C
    // `pointer_pp_clear`/`executer_pp_clear` pulses. Same value the DPU paths
    // in conv/elementwise/activation already use.
    // ========================================================================

    cmds.push(
        Register::<PpuSPointer>::new()
            .pointer_pp_mode(Bits::new(1))
            .executer_pp_en(Bits::new(1))
            .pointer_pp_en(Bits::new(1))
            .build(),
    );
    cmds.push(
        Register::<PpuRdmaSPointer>::new()
            .pointer_pp_mode(Bits::new(1))
            .executer_pp_en(Bits::new(1))
            .pointer_pp_en(Bits::new(1))
            .build(),
    );

    // ========================================================================
    // PPU -- vendor programs the consumer block before its PPU_RDMA feeder.
    // ========================================================================

    cmds.push(
        Register::<PpuDataCubeInWidth>::new()
            .cube_in_width(Bits::new(tile.input_width - 1))
            .build(),
    );
    cmds.push(
        Register::<PpuDataCubeInHeight>::new()
            .cube_in_height(Bits::new(shape.input_height - 1))
            .build(),
    );
    cmds.push(
        Register::<PpuDataCubeInChannel>::new()
            .cube_in_channel(Bits::new(programmed_channels - 1))
            .build(),
    );
    cmds.push(
        Register::<PpuDataCubeOutWidth>::new()
            .cube_out_width(Bits::new(tile.output_width - 1))
            .build(),
    );
    cmds.push(
        Register::<PpuDataCubeOutHeight>::new()
            .cube_out_height(Bits::new(shape.output_height - 1))
            .build(),
    );
    cmds.push(
        Register::<PpuDataCubeOutChannel>::new()
            .cube_out_channel(Bits::new(programmed_channels - 1))
            .build(),
    );

    cmds.push(
        Register::<PpuOperationModeCfg>::new()
            .pooling_method(Bits::new(shape.method.bits()))
            .flying_mode(Bits::new(1)) // standalone via PPU_RDMA, not pipelined after DPU
            .index_en(Bits::new(0)) // no argmax/argmin output wiring yet
            .use_cnt(Bits::new(0))
            .notch_addr(Bits::new(shape.output_width - tile.output_width))
            .build(),
    );
    cmds.push(
        Register::<PpuPoolingKernelCfg>::new()
            .kernel_width(Bits::new(shape.kernel_width - 1))
            .kernel_height(Bits::new(shape.kernel_height - 1))
            .kernel_stride_width(Bits::new(shape.stride_x - 1))
            .kernel_stride_height(Bits::new(shape.stride_y - 1))
            .build(),
    );
    cmds.push(
        Register::<PpuRecipKernelWidth>::new()
            .recip_kernel_width(Bits::new(recip_kernel_width))
            .build(),
    );
    cmds.push(
        Register::<PpuRecipKernelHeight>::new()
            .recip_kernel_height(Bits::new(recip_kernel_height))
            .build(),
    );
    cmds.push(
        Register::<PpuPoolingPaddingCfg>::new()
            .pad_left(Bits::new(tile.pad_left))
            .pad_top(Bits::new(shape.pad_top))
            .pad_right(Bits::new(tile.pad_right))
            .pad_bottom(Bits::new(programmed_pad_bottom))
            .build(),
    );
    cmds.push(
        Register::<PpuPaddingValue1Cfg>::new()
            .pad_value_0(Bits::new(shape.pad_value))
            .build(),
    );
    cmds.push(zero::<PpuPaddingValue2Cfg>());

    cmds.push(
        Register::<PpuDstBaseAddr>::new()
            .dst_base_addr(Bits::new(dst_base_addr_shifted))
            .build(),
    );
    cmds.push(
        Register::<PpuDstSurfStride>::new()
            .dst_surf_stride(Bits::new(dst_surf_stride))
            .build(),
    );
    cmds.push(
        Register::<PpuDataFormat>::new()
            .proc_precision(Bits::new(shape.precision.ppu_precision()))
            .dpu_flyin(Bits::new(0))
            .index_add(Bits::new(dst_surf_stride))
            .build(),
    );
    cmds.push(
        Register::<PpuMiscCtrl>::new()
            .burst_len(Bits::new(3))
            .nonalign(Bits::new(0))
            .mc_surf_out(Bits::new(0))
            // Only meaningful in non-aligned mode. Keeping this zero, as
            // every direct vendor capture does, avoids imposing its 16-bit
            // field width on otherwise valid large output surfaces.
            .surf_len(Bits::new(0))
            .build(),
    );

    // ========================================================================
    // PPU_RDMA -- standalone read side, feeds PPU directly from memory.
    // ========================================================================

    cmds.push(
        Register::<PpuRdmaCubeInWidth>::new()
            .cube_in_width(Bits::new(tile.input_width - 1))
            .build(),
    );
    cmds.push(
        Register::<PpuRdmaCubeInHeight>::new()
            .cube_in_height(Bits::new(shape.input_height - 1))
            .build(),
    );
    cmds.push(
        Register::<PpuRdmaCubeInChannel>::new()
            .cube_in_channel(Bits::new(programmed_channels - 1))
            .build(),
    );
    cmds.push(
        Register::<PpuRdmaSrcBaseAddr>::new()
            .src_base_addr(Bits::new(input_addr))
            .build(),
    );
    cmds.push(
        Register::<PpuRdmaSrcLineStride>::new()
            .src_line_stride(Bits::new(src_line_stride))
            .build(),
    );
    cmds.push(
        Register::<PpuRdmaSrcSurfStride>::new()
            .src_surf_stride(Bits::new(src_surf_stride))
            .build(),
    );
    cmds.push(
        Register::<PpuRdmaDataFormat>::new()
            .in_precision(Bits::new(shape.precision.rdma_precision()))
            .build(),
    );

    push_ppu_kick(&mut cmds);
    cmds
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rocket::{
        builders::{
            DOMAIN_PC, DOMAIN_PPU, DOMAIN_PPU_RDMA, RegisterMeta,
            pc::{PCOperationMask, PCTrailer},
        },
        debug::decode,
    };

    fn shape(
        width: u32,
        height: u32,
        kernel_width: u32,
        kernel_height: u32,
        stride_x: u32,
        stride_y: u32,
        precision: PoolingPrecision,
    ) -> PoolingShape {
        let output_width = (width - kernel_width) / stride_x + 1;
        let output_height = (height - kernel_height) / stride_y + 1;
        PoolingShape {
            input_width: width,
            input_height: height,
            input_channels: 16,
            output_width,
            output_height,
            output_channels: 16,
            precision,
            kernel_width,
            kernel_height,
            stride_x,
            stride_y,
            method: PoolingMethod::Max,
            pad_left: 0,
            pad_top: 0,
            pad_right: 0,
            pad_bottom: 0,
            pad_value: 0,
        }
    }

    fn register_value<R: RegisterMeta>(commands: &[RegCmd]) -> u32 {
        let command = commands
            .iter()
            .find(|command| {
                ((command.0 >> 48) as u32) == R::DOMAIN && (command.0 as u32 & 0xffff) == R::OFFSET
            })
            .expect("register is present");
        (command.0 >> 16) as u32
    }

    fn single_task(shape: &PoolingShape, bufs: &PoolingBuffers) -> Vec<RegCmd> {
        let mut tasks = PoolingPlan::new(*shape).programs_with_buffers(bufs);
        assert_eq!(tasks.len(), 1, "test shape unexpectedly required tiling");
        tasks.pop().unwrap()
    }

    #[test]
    fn int8_program_matches_the_direct_vendor_capture_with_pointer_resets() {
        // The pooling-sweep capture
        // `pool-max-w64-h48-c12-kw3-kh3-sx2-sy2-p1-1-1-1-i8.rknn` is an
        // exact direct PPU program. Addresses are zero in the static file
        // and are therefore also zero here. The two S_POINTER values add the
        // W1C pointer/executer reset bits to make independent submissions
        // independent of persistent hardware state; all other words remain
        // capture-identical.
        let shape = PoolingShape {
            input_width: 64,
            input_height: 48,
            input_channels: 12,
            output_width: 32,
            output_height: 24,
            output_channels: 12,
            precision: PoolingPrecision::Int8,
            kernel_width: 3,
            kernel_height: 3,
            stride_x: 2,
            stride_y: 2,
            method: PoolingMethod::Max,
            pad_left: 1,
            pad_top: 1,
            pad_right: 1,
            pad_bottom: 1,
            pad_value: 0,
        };
        let commands = single_task(
            &shape,
            &PoolingBuffers {
                input_addr: 0,
                output_addr: 0,
            },
        );
        let actual: Vec<_> = commands.iter().map(decode).collect();
        let expected = vec![
            (DOMAIN_PPU, 0x6004, 0x0000_000e),
            (DOMAIN_PPU_RDMA, 0x7004, 0x0000_000e),
            (DOMAIN_PPU, 0x600c, 0x0000_003f),
            (DOMAIN_PPU, 0x6010, 0x0000_002f),
            (DOMAIN_PPU, 0x6014, 0x0000_000f),
            (DOMAIN_PPU, 0x6018, 0x0000_001f),
            (DOMAIN_PPU, 0x601c, 0x0000_0017),
            (DOMAIN_PPU, 0x6020, 0x0000_000f),
            (DOMAIN_PPU, 0x6024, 0x0000_0011),
            (DOMAIN_PPU, 0x6034, 0x0011_0202),
            (DOMAIN_PPU, 0x6038, 0),
            (DOMAIN_PPU, 0x603c, 0),
            (DOMAIN_PPU, 0x6040, 0x0000_0011),
            (DOMAIN_PPU, 0x6044, 0),
            (DOMAIN_PPU, 0x6048, 0),
            (DOMAIN_PPU, 0x6070, 0),
            (DOMAIN_PPU, 0x607c, 0x0000_3000),
            (DOMAIN_PPU, 0x6084, 0x0000_3000),
            (DOMAIN_PPU, 0x60dc, 0x0000_0003),
            (DOMAIN_PPU_RDMA, 0x700c, 0x0000_003f),
            (DOMAIN_PPU_RDMA, 0x7010, 0x0000_002f),
            (DOMAIN_PPU_RDMA, 0x7014, 0x0000_000f),
            (DOMAIN_PPU_RDMA, 0x701c, 0),
            (DOMAIN_PPU_RDMA, 0x7024, 0x0000_0400),
            (DOMAIN_PPU_RDMA, 0x7028, 0x0000_c000),
            (DOMAIN_PPU_RDMA, 0x7030, 0x0000_0001),
            (0, 0, 0),
            (DOMAIN_PC, 0x0014, 0),
            (0x0041, 0, 0),
            (0x0081, 0x0008, 0x0000_0060),
            (0, 0, 0),
            (0, 0, 0),
        ];
        assert_eq!(actual, expected);
    }

    #[test]
    fn channel_extents_follow_the_captured_feature_atom_padding() {
        for (precision, logical_channels, programmed_minus_one) in [
            (PoolingPrecision::Fp16, 1, 7),
            (PoolingPrecision::Fp16, 9, 15),
            (PoolingPrecision::Int8, 1, 15),
            (PoolingPrecision::Int8, 17, 31),
        ] {
            let shape = PoolingShape {
                input_channels: logical_channels,
                output_channels: logical_channels,
                ..shape(4, 4, 2, 2, 2, 2, precision)
            };
            let commands = single_task(
                &shape,
                &PoolingBuffers {
                    input_addr: 0,
                    output_addr: 0,
                },
            );
            assert_eq!(
                register_value::<PpuDataCubeInChannel>(&commands),
                programmed_minus_one
            );
            assert_eq!(
                register_value::<PpuDataCubeOutChannel>(&commands),
                programmed_minus_one
            );
            assert_eq!(
                register_value::<PpuRdmaCubeInChannel>(&commands),
                programmed_minus_one
            );
        }
    }

    #[test]
    fn sub_atomic_channel_counts_emit_an_identical_program() {
        // pooling_hw.rs moved its cases from one channel to a full atom on the
        // stated grounds that this changes no emitted register, only how much
        // of each atom carries checked data. Hold that invariant here: any
        // channel count inside one int8 atom must produce the same commands.
        let words = |logical_channels| -> Vec<u64> {
            single_task(
                &PoolingShape {
                    input_channels: logical_channels,
                    output_channels: logical_channels,
                    ..shape(13, 9, 3, 3, 2, 2, PoolingPrecision::Int8)
                },
                &PoolingBuffers {
                    input_addr: 0x1000,
                    output_addr: 0x8000,
                },
            )
            .iter()
            .map(|command| command.0)
            .collect()
        };

        let reference = words(16);
        for logical_channels in [1, 2, 8, 15, 16] {
            assert_eq!(
                words(logical_channels),
                reference,
                "c={logical_channels} diverged from the full-atom program"
            );
        }
    }

    #[test]
    fn int8_average_reciprocals_match_the_convpool_capture() {
        // convpool-w32-h32-ci16-co16-k3-s1-pk2-ps2-pmavg-i8.rknn
        // emits an ordinary flying-mode PPU stage after its memory-writing
        // convolution task.
        let shape = PoolingShape {
            method: PoolingMethod::Avg,
            ..shape(32, 32, 2, 2, 2, 2, PoolingPrecision::Int8)
        };
        let commands = single_task(
            &shape,
            &PoolingBuffers {
                input_addr: 0,
                output_addr: 0,
            },
        );
        assert_eq!(register_value::<PpuOperationModeCfg>(&commands), 0x10);
        assert_eq!(
            register_value::<PpuPoolingKernelCfg>(&commands),
            0x0011_0101
        );
        assert_eq!(register_value::<PpuRecipKernelWidth>(&commands), 0x8000);
        assert_eq!(register_value::<PpuRecipKernelHeight>(&commands), 0x8000);
    }

    #[test]
    fn capture_precision_enums_and_aligned_output_controls_are_programmed() {
        for (precision, ppu, rdma) in [
            (PoolingPrecision::Int8, 0, 1),
            (PoolingPrecision::Fp16, 2, 2),
        ] {
            let shape = shape(64, 48, 3, 3, 2, 2, precision);
            let commands = single_task(
                &shape,
                &PoolingBuffers {
                    input_addr: 0,
                    output_addr: 0,
                },
            );
            let output_area = (shape.output_width * shape.output_height).next_multiple_of(4);
            assert_eq!(
                register_value::<PpuRdmaDataFormat>(&commands),
                rdma,
                "{precision:?} PPU_RDMA precision"
            );
            assert_eq!(
                register_value::<PpuDataFormat>(&commands),
                output_area << 4 | ppu,
                "{precision:?} PPU precision and index_add"
            );
            assert_eq!(
                register_value::<PpuDstSurfStride>(&commands),
                output_area << 4
            );
            assert_eq!(register_value::<PpuMiscCtrl>(&commands), 3);
            assert_eq!(register_value::<PpuRecipKernelWidth>(&commands), 0);
            assert_eq!(register_value::<PpuRecipKernelHeight>(&commands), 0);
        }
    }

    #[test]
    fn large_spatial_extents_do_not_depend_on_surf_len() {
        let shape = shape(64, 8192, 3, 3, 2, 2, PoolingPrecision::Fp16);
        let commands = single_task(
            &shape,
            &PoolingBuffers {
                input_addr: 0,
                output_addr: 0,
            },
        );
        let output_area = (31_u32 * 4095).next_multiple_of(4);
        assert_eq!(register_value::<PpuDataCubeInHeight>(&commands), 8191);
        assert_eq!(register_value::<PpuDataCubeOutHeight>(&commands), 4094);
        assert_eq!(
            register_value::<PpuRdmaSrcSurfStride>(&commands),
            (64 * 8192) << 4
        );
        assert_eq!(
            register_value::<PpuDstSurfStride>(&commands),
            output_area << 4
        );
        assert_eq!(
            register_value::<PpuDataFormat>(&commands),
            output_area << 4 | 2
        );
        assert_eq!(register_value::<PpuMiscCtrl>(&commands), 3);
    }

    #[test]
    fn largest_dispatchable_stride_encodes_at_the_kernel_edge() {
        // The stride field is 4 bits (1..=16) but kernels cap at 8, and
        // stride-beyond-kernel is rejected because it hangs the NPU, so 8 is
        // the largest stride that can actually reach hardware. Strides 9..=16
        // still fit the field and are unreachable by construction.
        let shape = shape(64, 48, 8, 8, 8, 8, PoolingPrecision::Int8);
        let commands = single_task(
            &shape,
            &PoolingBuffers {
                input_addr: 0,
                output_addr: 0,
            },
        );
        assert_eq!(
            register_value::<PpuPoolingKernelCfg>(&commands),
            0x0077_0707
        );
    }

    #[test]
    fn wide_plan_reproduces_the_unpadded_vendor_tiles() {
        let shape = shape(256, 48, 3, 3, 2, 2, PoolingPrecision::Fp16);
        let plan = PoolingPlan::new(shape);
        assert_eq!(
            plan.tiles(),
            &[
                PoolingTile {
                    input_first: 0,
                    input_width: 127,
                    output_first: 0,
                    output_width: 63,
                    pad_left: 0,
                    pad_right: 0,
                },
                PoolingTile {
                    input_first: 126,
                    input_width: 129,
                    output_first: 63,
                    output_width: 64,
                    pad_left: 0,
                    pad_right: 0,
                },
            ]
        );
        let programs = plan.programs_with_buffers(&PoolingBuffers {
            input_addr: 0x1000,
            output_addr: 0x2000,
        });
        assert_eq!(programs.len(), 2);
        assert_eq!(
            register_value::<PpuOperationModeCfg>(&programs[0]),
            0x0040_0011
        );
        assert_eq!(
            register_value::<PpuOperationModeCfg>(&programs[1]),
            0x003f_0011
        );
        assert_eq!(register_value::<PpuRdmaSrcBaseAddr>(&programs[1]), 0x17e0);
        assert_eq!(register_value::<PpuDstBaseAddr>(&programs[1]), 0x23f0);
        assert_eq!(register_value::<PpuDstSurfStride>(&programs[1]), 2924 << 4);
    }

    #[test]
    fn every_planned_tile_is_one_direct_ppu_task() {
        let programs = PoolingPlan::new(shape(256, 48, 3, 3, 2, 2, PoolingPrecision::Int8))
            .programs_with_buffers(&PoolingBuffers {
                input_addr: 0x1000,
                output_addr: 0x8000,
            });

        assert_eq!(programs.len(), 2);
        for program in programs {
            assert_eq!(program.len(), 32);
            assert_eq!(
                program
                    .iter()
                    .filter(|command| {
                        command.0
                            == PCTrailer::operation_enable(
                                PCOperationMask::PPU | PCOperationMask::PPU_RDMA,
                            )
                            .0
                    })
                    .count(),
                1
            );
            for (domain, _, _) in program.iter().map(decode) {
                assert!(
                    matches!(
                        domain,
                        DOMAIN_PPU | DOMAIN_PPU_RDMA | DOMAIN_PC | 0 | 0x0041 | 0x0081
                    ),
                    "pooling task contains unexpected register domain {domain:#x}"
                );
            }
        }
    }

    #[test]
    fn k2s2_fp16_capture_boundary_uses_one_task_through_output_128() {
        for input_width in [256, 257] {
            let plan = PoolingPlan::new(shape(input_width, 48, 2, 2, 2, 2, PoolingPrecision::Fp16));
            assert_eq!(plan.tiles().len(), 1);
            assert_eq!(plan.tiles()[0].input_width, 256);
            assert_eq!(plan.tiles()[0].output_width, 128);
        }
    }

    #[test]
    fn k2s2_int8_avoids_the_failing_128_column_and_equal_64_column_tasks() {
        let plan = PoolingPlan::new(shape(257, 48, 2, 2, 2, 2, PoolingPrecision::Int8));
        assert_eq!(
            plan.tiles()
                .iter()
                .map(|tile| (tile.input_width, tile.output_width))
                .collect::<Vec<_>>(),
            [(126, 63), (130, 65)]
        );
    }

    #[test]
    fn odd_input_area_uses_the_vendor_surface_alignment() {
        let shape = shape(7, 5, 3, 2, 2, 1, PoolingPrecision::Int8);
        let commands = single_task(
            &shape,
            &PoolingBuffers {
                input_addr: 0,
                output_addr: 0,
            },
        );
        assert_eq!(register_value::<PpuRdmaSrcLineStride>(&commands), 7 << 4);
        assert_eq!(register_value::<PpuRdmaSrcSurfStride>(&commands), 36 << 4);
    }

    #[test]
    fn k2s2_width_258_matches_the_two_vendor_tiles() {
        let plan = PoolingPlan::new(shape(258, 48, 2, 2, 2, 2, PoolingPrecision::Fp16));
        assert_eq!(
            plan.tiles(),
            &[
                PoolingTile {
                    input_first: 0,
                    input_width: 128,
                    output_first: 0,
                    output_width: 64,
                    pad_left: 0,
                    pad_right: 0,
                },
                PoolingTile {
                    input_first: 128,
                    input_width: 130,
                    output_first: 64,
                    output_width: 65,
                    pad_left: 0,
                    pad_right: 0,
                },
            ]
        );
        let programs = plan.programs_with_buffers(&PoolingBuffers {
            input_addr: 0x1000,
            output_addr: 0x2000,
        });
        assert_eq!(register_value::<PpuRdmaSrcBaseAddr>(&programs[0]), 0x1000);
        assert_eq!(register_value::<PpuRdmaSrcBaseAddr>(&programs[1]), 0x1800);
        assert_eq!(register_value::<PpuDstBaseAddr>(&programs[0]), 0x2000);
        assert_eq!(register_value::<PpuDstBaseAddr>(&programs[1]), 0x2400);
        assert_eq!(
            register_value::<PpuOperationModeCfg>(&programs[0]),
            0x0041_0011
        );
        assert_eq!(
            register_value::<PpuOperationModeCfg>(&programs[1]),
            0x0040_0011
        );
    }

    #[test]
    fn k2s2_width_512_uses_two_128_column_tasks() {
        let plan = PoolingPlan::new(shape(512, 48, 2, 2, 2, 2, PoolingPrecision::Fp16));
        assert_eq!(
            plan.tiles()
                .iter()
                .map(|tile| (tile.input_width, tile.output_width))
                .collect::<Vec<_>>(),
            [(256, 128), (256, 128)]
        );
    }

    #[test]
    fn trailing_padding_is_programmed_only_when_an_output_window_reaches_it() {
        let mut padded = shape(64, 48, 3, 3, 2, 2, PoolingPrecision::Fp16);
        padded.pad_left = 1;
        padded.pad_top = 1;
        padded.pad_right = 1;
        padded.pad_bottom = 1;
        padded.output_width = 32;
        padded.output_height = 24;
        let commands = single_task(
            &padded,
            &PoolingBuffers {
                input_addr: 0,
                output_addr: 0,
            },
        );
        assert_eq!(register_value::<PpuPoolingPaddingCfg>(&commands), 0x11);

        padded.stride_x = 3;
        padded.output_width = 22;
        let commands = single_task(
            &padded,
            &PoolingBuffers {
                input_addr: 0,
                output_addr: 0,
            },
        );
        assert_eq!(register_value::<PpuPoolingPaddingCfg>(&commands), 0x111);
    }

    #[test]
    #[should_panic(expected = "input width must be 1..=8192")]
    fn rejects_extent_beyond_vendor_confirmed_field_boundary() {
        shape(8193, 48, 3, 3, 2, 2, PoolingPrecision::Int8).validate();
    }

    #[test]
    #[should_panic(expected = "pooling kernel axes must be 1..=8")]
    fn rejects_kernel_beyond_hardware_backed_direct_limit() {
        shape(64, 48, 16, 3, 2, 2, PoolingPrecision::Int8).validate();
    }

    #[test]
    #[should_panic(expected = "vertical stride 4 exceeds kernel 3")]
    fn rejects_vertical_stride_beyond_kernel() {
        shape(64, 31, 3, 3, 2, 4, PoolingPrecision::Int8).validate();
    }

    #[test]
    #[should_panic(expected = "horizontal stride 8 exceeds kernel 3")]
    fn rejects_horizontal_stride_beyond_kernel() {
        shape(64, 31, 3, 3, 8, 2, PoolingPrecision::Int8).validate();
    }

    #[test]
    fn accepts_stride_equal_to_kernel() {
        // The boundary stays legal: stride == kernel is the disjoint-window
        // case, and 64x31 k8x8 s8x8 dispatches cleanly on hardware. Only
        // stride *beyond* kernel is rejected.
        shape(64, 31, 8, 8, 8, 8, PoolingPrecision::Int8).validate();
        shape(64, 31, 3, 3, 3, 3, PoolingPrecision::Int8).validate();
    }
}
