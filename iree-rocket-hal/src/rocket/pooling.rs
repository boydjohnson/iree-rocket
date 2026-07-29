//! Pooling on the PPU, in both shapes this crate now builds.
//!
//! - **Standalone flying** ([`build_pooling_regcmd`], TRM Ch.36 Fig 36-6):
//!   PPU_RDMA reads the input straight from memory and feeds PPU directly,
//!   bypassing CNA/CORE/DPU entirely. Kept for reference -- it is not what
//!   the vendor compiler emits, and its PPU_RDMA stride math remains
//!   unconfirmed.
//! - **Via a DPU bypass stage** ([`build_pooling_via_dpu_bypass_tasks`]):
//!   one job containing separately-kicked tasks: a near-identity conv
//!   followed by one PPU pass per horizontal tile. This is the shape a real
//!   rknn-toolkit2-compiled model actually emits. The single-tile dataflow
//!   has hardware coverage; the exact one-job multi-task topology and
//!   multi-tile case are covered below and await a board run.
//!
//! A third shape -- pooling pipelined on-chip directly after a real conv via
//! `dpu_flyin`, no PPU_RDMA fetch at all -- was implemented and then
//! retired rather than migrated off the Mesa-derived convolution builder. A dedicated 53-model
//! `Conv2d -> Pool` sweep (`iree-rocket-design-spike`'s
//! `sweep_convpool_generate.py`/`sweep_convpool_diff.py`, see
//! `DESIGN_NOTES.md`'s "Conv+pool fusion sweep" section) found zero capture
//! evidence for it across every swept geometry and precision -- the real
//! vendor toolchain always emits the two-kick bypass shape above instead.
//! Its own doc comment had already hedged that it was "chosen after the
//! standalone path hung real hardware," i.e. a fallback tried when the
//! flying-mode path failed, not something derived from a real compiler
//! output.
//!
//! [`build_max_reduction_tree_regcmd`] chains the bypass shape into the
//! max-reduction half of a softmax; see its section comment for the two
//! gaps that keep this from being a whole softmax op.
//!
//! Mesa-derived in structure, though there is no Mesa/Teflon reference for
//! pooling itself (`rkt_ml.c` only ever implements convolution) -- the
//! fields come from the PPU/PPU_RDMA register layout plus TRM Ch.36 prose.
//! Both builders use [`crate::rocket::conv`]'s capture-derived `ConvPlan` for
//! their conv-coupled stage.

use crate::rocket::{
    activation::Activation,
    builders::{Bits, RegCmd, Register, pc::PCTrailer, ppu::*, ppu_rdma::*},
    conv::{self, Activation as ConvActivation, ConvPlan, Kernels},
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

//===========================================================================
// Pooling (standalone PPU, "flying mode" -- TRM Ch.36 Fig 36-6): PPU_RDMA
// reads the input straight from memory and feeds PPU directly, bypassing
// CNA/CORE/DPU entirely. There is NO Mesa/Teflon reference for this path
// (`rkt_ml.c` only ever implements convolution) -- every field below is
// derived from the PPU/PPU_RDMA register layout in `builders/ppu.rs` /
// `builders/ppu_rdma.rs` (bindgen'd from Mesa's own `registers.xml`, see
// builders.rs's DOMAIN_* comment) plus the TRM Ch.36 §4.6/§4.7 prose and
// `build_conv_regcmd`'s established conventions (N-1 encoding on every
// *_RDMA/CORE/DPU cube dimension). This
// standalone shape is retained as a register-level reference, not as the
// production path. Its hardware suite was retired after the large-window and
// tiled-width cases failed; the vendor-observed two-stage path below now
// carries the numerical hardware matrix.
//
// - RESOLVED (hardware-confirmed by the retired standalone exploration on a
//   real RK3588): `PoolingMethod`'s bit encoding is Avg=0, Max=1, Min=2 --
//   see `PoolingMethod::bits()`'s doc comment.
// - RESOLVED by the dedicated 143-capture fp16/int8 pooling sweep:
//   PPU_RDMA line/surface strides are input width/area; PPU destination
//   surface stride and `index_add` are output area rounded up to four
//   pixels; aligned mode leaves `surf_len=0`; all three channel extents pad
//   to a 16-byte feature atom (C8 fp16/C16 int8); PPU registers precede the
//   PPU_RDMA registers; and the kick tail carries two final zero words for
//   a 32-command direct program. The same sweep confirms spatial extents
//   through 8192 and both precision enums.
// - RESOLVED (was open when this task hung real hardware): the kick
//   used to fire `build_conv_regcmd`'s fixed CNA/CORE/DPU/DPU_RDMA bitmask
//   unconditionally, which never actually enabled PPU/PPU_RDMA for this
//   task -- the root cause of the original hang. Now kicks `KICK_PPU |
//   KICK_PPU_RDMA` only, confirmed against a real rknn-toolkit2-compiled
//   pooling.rknn capture (see NOTES.md in rknpu-spelunking) and against
//   real hardware.
// - SUPERSEDED: this standalone-only shape is not actually what the real
//   vendor compiler emits at all -- the same NOTES.md capture shows a
//   *third* shape (a real CNA/CORE/DPU bypass task, memory round-trip,
//   *then* a separately-kicked standalone-PPU task) that matches neither
//   this path nor the pipelined `dpu_flyin` path below. See
//   `build_pooling_via_dpu_bypass_tasks` further down, which is the
//   recommended default going forward; this function is kept for hardware
//   comparison/reference.
//===========================================================================

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
    /// Fused post-processing for this pooling op, per [[Activation]]'s own
    /// doc comment. PPU itself has no ALU/activation capability at all
    /// (confirmed by enumerating every register in `builders/ppu.rs`) --
    /// this field is only meaningful on a pooling path that runs a real DPU
    /// stage ahead of PPU. `build_pooling_via_dpu_bypass_tasks` applies it
    /// to that DPU stage's BS core (the same fusion point Phase 1 validated
    /// for conv); `build_pooling_regcmd`'s pure standalone-flying path (no
    /// DPU at all) cannot honor this and asserts it's `Activation::None`
    /// rather than silently ignoring a non-`None` value.
    pub activation: Activation,
}

const MAX_PPU_EXTENT: u32 = 8192;
const MAX_PPU_KERNEL_OR_STRIDE: u32 = 16;
const MAX_PPU_PADDING: u32 = 7;
const MAX_DIRECT_KERNEL: u32 = 8;
const MAX_DIRECT_INPUT_WIDTH: u32 = 129;
const MAX_DIRECT_OUTPUT_WIDTH: u32 = 64;

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
/// The vendor caps a direct task at 64 output columns and carries the kernel
/// overlap in `input_width`. `input_first` and `output_first` are pixel
/// offsets into the full tensor; the builder converts them to 16-byte feature
/// atom offsets when relocating the two base-address registers.
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
/// Width 129 is the largest whole direct input in the corpus. Larger tensors
/// split into balanced tiles of at most 64 output columns, assigning any
/// remainder to the rightmost tiles. This reproduces the observed 63+64
/// split for an unpadded width-256 3x3/stride-2 pool and 64+64 for its padded
/// counterpart.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PoolingPlan {
    shape: PoolingShape,
    tiles: Vec<PoolingTile>,
}

impl PoolingPlan {
    pub fn new(shape: PoolingShape) -> PoolingPlan {
        shape.validate();
        let tile_count = shape.output_width.div_ceil(MAX_DIRECT_OUTPUT_WIDTH);
        let base_width = shape.output_width / tile_count;
        let wider_tiles = shape.output_width % tile_count;
        let first_wider = tile_count - wider_tiles;
        let mut output_first = 0;
        let mut tiles = Vec::with_capacity(tile_count as usize);

        for index in 0..tile_count {
            let output_width = base_width + u32::from(index >= first_wider);
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
                tile.input_width <= MAX_DIRECT_INPUT_WIDTH
                    && tile.output_width <= MAX_DIRECT_OUTPUT_WIDTH,
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
    /// Multiple tiles submitted as separate DRM jobs against one shared
    /// output BO get an implicit write-after-write dependency from the
    /// kernel's GEM fence tracking (`drm_sched_job_add_implicit_dependencies`,
    /// per-buffer-object, not per-byte-range) -- later tiles cannot even
    /// dispatch to hardware until earlier ones signal completion, silently
    /// serializing what looks like independent parallel work. Confirmed on
    /// real hardware: `wide_pooling_plan_runs_on_npu` intermittently came
    /// back with one whole tile's region still at its pre-fill sentinel,
    /// because that tile's job was gated behind another job's fence rather
    /// than actually running independently. Use
    /// `programs_with_separate_output_buffers` instead for genuinely
    /// independent tiles; this method is kept for callers that specifically
    /// want one contiguous output buffer. The preferred vendor-style
    /// bypass path puts all programs in one job as ordered tasks, where
    /// serialization is explicit and no inter-job fence gates a later tile.
    pub fn programs_with_buffers(&self, bufs: &PoolingBuffers) -> Vec<Vec<RegCmd>> {
        self.tiles
            .iter()
            .map(|tile| {
                let output_addr = bufs
                    .output_addr
                    .checked_add(tile.output_first * 16)
                    .expect("pooling tile output address overflows u32");
                let mut commands =
                    build_ppu_standalone_flying(&self.shape, tile, bufs.input_addr, output_addr);
                push_ppu_kick(&mut commands);
                commands
            })
            .collect()
    }

    /// Emits one independently kicked, submission-ready program per tile,
    /// each writing to its own dedicated output buffer rather than an
    /// offset within one shared buffer -- so tiles submitted as separate
    /// DRM jobs have no shared-BO write dependency and can genuinely run
    /// independently (see `programs_with_buffers`'s doc comment for why
    /// that matters).
    ///
    /// Each tile's `dst_surf_stride`/`notch_addr` are still computed from
    /// the *whole* shape's `output_width` (a horizontally-tiled row is not
    /// contiguous in memory -- consecutive rows are `shape.output_width`
    /// columns apart, not `tile.output_width`), so this does NOT give tiles
    /// a compact, tile-width-only buffer. Each address in
    /// `tile_output_base_addrs` must point at a buffer with the same full
    /// footprint as the combined image (same size/layout `programs_with_
    /// buffers`'s single shared buffer would need) -- just a physically
    /// distinct GEM object per tile, so no two tiles share one buffer's
    /// `dma_resv` and no implicit write-write dependency serializes them.
    /// Only that tile's own column range within its buffer ever gets
    /// written; callers reassemble the final image from each tile's slice
    /// afterward. Must have exactly one entry per `self.tiles()`, in order.
    pub fn programs_with_separate_output_buffers(
        &self,
        input_addr: u32,
        tile_output_base_addrs: &[u32],
    ) -> Vec<Vec<RegCmd>> {
        assert_eq!(
            tile_output_base_addrs.len(),
            self.tiles.len(),
            "programs_with_separate_output_buffers: need exactly one output address per tile"
        );
        self.tiles
            .iter()
            .zip(tile_output_base_addrs)
            .map(|(tile, &base_addr)| {
                let output_addr = base_addr
                    .checked_add(tile.output_first * 16)
                    .expect("pooling tile output address overflows u32");
                let mut commands =
                    build_ppu_standalone_flying(&self.shape, tile, input_addr, output_addr);
                push_ppu_kick(&mut commands);
                commands
            })
            .collect()
    }
}

/// DMA addresses for the two buffers a standalone pooling op needs.
pub struct PoolingBuffers {
    pub input_addr: u32,
    /// NOTE: `PPU_DST_BASE_ADDR` is a genuinely 28-bit hardware field
    /// (unlike every other block's 32-bit `*_BASE_ADDR` registers -- see
    /// `PpuDstBaseAddr::dst_base_addr`'s `Bits<28>` parameter, confirmed
    /// against the bindgen'd header, not a guess). A `CREATE_BO`
    /// allocation whose `dma_address` lands >= 0x1000_0000 (256MiB) into
    /// DMA-visible space cannot be addressed by PPU's output stage as-is.
    /// `build_pooling_regcmd` asserts this explicitly (a clear panic
    /// message) rather than letting `Bits::<28>::new` fail with its
    /// generic "value exceeds designated bit width" message.
    ///
    /// UPDATE (post first hardware run): this was wrong -- writing the raw
    /// byte address here produced a job that completed (no hang, PREP_BO
    /// returned) but left the *entire* output buffer at zero (confirmed by
    /// the retired standalone suite's sentinel-filled diagnostic). The
    /// 28-bit width is the same
    /// bits[31:4] convention TRM Ch.36 documents explicitly for
    /// `pc_base_address` ("bits 31:4 regcmd DMA address") -- a 32-bit byte
    /// address shifted right by 4 (assuming 16-byte alignment) always fits
    /// exactly in 28 bits with zero waste, unlike a genuine "only 256MiB
    /// addressable" hardware limit (which would be an odd, wasteful design
    /// choice and doesn't match anything else in this chapter). `DPU_DST_
    /// BASE_ADDR` being 32-bit and used unshifted in the hardware-
    /// validated `build_conv_regcmd` shows the shift isn't universal --
    /// only registers too narrow to hold a full byte address seem to use
    /// it, and PPU_DST_BASE_ADDR is the only such *_BASE_ADDR field found
    /// so far. `build_pooling_regcmd` now asserts 16-byte alignment
    /// (trivially satisfied by any page-aligned CREATE_BO allocation) and
    /// shifts right by 4 instead. NOT YET RE-CONFIRMED ON HARDWARE.
    pub output_addr: u32,
}

/// Standalone-flying PPU_RDMA+PPU register emission, factored out of
/// `build_pooling_regcmd` so `build_pooling_via_dpu_bypass_tasks` (the real
/// two-kick vendor shape, see its own doc comment) can reuse the exact same
/// PPU-stage sequence, differing only in where PPU_RDMA fetches from (a
/// caller-supplied real input buffer for the standalone path; a bypass
/// conv's memory-written output for the two-kick path) and in which kick
/// mask the caller appends afterwards. Pure extraction -- no behavior
/// change versus the original single function.
/// `output_addr` is the exact base address this tile's output starts
/// writing at -- callers sharing one output buffer across tiles must add
/// their own `tile.output_first * 16` offset before calling (see
/// `PoolingPlan::programs_with_buffers`); callers giving each tile its own
/// dedicated buffer (see `PoolingPlan::programs_with_separate_output_buffers`)
/// pass that buffer's address unmodified.
fn build_ppu_standalone_flying(
    shape: &PoolingShape,
    tile: &PoolingTile,
    input_addr: u32,
    output_addr: u32,
) -> Vec<RegCmd> {
    shape.validate();
    assert!(
        tile.input_width <= MAX_DIRECT_INPUT_WIDTH && tile.output_width <= MAX_DIRECT_OUTPUT_WIDTH,
        "direct pooling tile exceeds the capture-derived width limits: {tile:?}"
    );
    let input_addr = input_addr
        .checked_add(tile.input_first * 16)
        .expect("pooling tile input address overflows u32");
    assert!(
        output_addr.is_multiple_of(16),
        "build_ppu_standalone_flying: output_addr {output_addr:#x} is not 16-byte aligned -- \
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
    // surfaced once a real DPU-written (not uniformly-filled) source buffer
    // was read in `pooling_via_dpu_bypass_hw.rs`, where rows beyond row 0
    // jumped past the DPU's actual (small) write footprint into untouched
    // buffer contents.
    let src_line_stride = (shape.input_width * ATOMIC_K_SIZE) / FEATURE_ATOMIC_SIZE;
    let src_surf_stride =
        (shape.input_width * ATOMIC_K_SIZE * shape.input_height) / FEATURE_ATOMIC_SIZE;

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
    // Ping-pong pointers -- same pattern/values as build_conv_regcmd's
    // DPU/DPU_RDMA pair, applied to the two blocks this op actually uses.
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

    cmds
}

/// Standalone-flying PPU pooling (TRM Ch.36 Fig 36-6): PPU_RDMA fetches
/// straight from `bufs.input_addr`, CNA/CORE/DPU untouched. Kicked with just
/// `KICK_PPU | KICK_PPU_RDMA`, confirmed against a real rknn-toolkit2-
/// compiled pooling.rknn capture's standalone-flying task (see NOTES.md
/// "Decoding a real regcmd program for a pooling-only op") -- previously
/// reused `build_conv_regcmd`'s fixed CNA/CORE/DPU/DPU_RDMA kick verbatim,
/// which never set PPU's/PPU_RDMA's enable bit at all and is what hung real
/// hardware. NOT YET RE-VALIDATED ON HARDWARE with this fix -- and per the
/// real capture, this standalone-only shape isn't actually what the vendor
/// compiler emits at all; see `build_pooling_via_dpu_bypass_tasks` below
/// for that real two-kick shape. Kept for hardware comparison/reference,
/// not as the recommended default. This convenience entry point accepts
/// only a one-tile shape; use [`PoolingPlan::programs_with_buffers`] for a
/// width that requires horizontal splitting.
pub fn build_pooling_regcmd(shape: &PoolingShape, bufs: &PoolingBuffers) -> Vec<RegCmd> {
    assert!(
        matches!(shape.activation, Activation::None),
        "build_pooling_regcmd: fused activation ({:?}) requested but this standalone-flying \
         path has no DPU stage at all -- PPU has no ALU/activation capability of its own (see \
         PoolingShape::activation's doc comment). Use build_pooling_via_dpu_bypass_tasks \
         instead if fused activation is needed.",
        shape.activation
    );
    let plan = PoolingPlan::new(*shape);
    assert_eq!(
        plan.tiles.len(),
        1,
        "build_pooling_regcmd accepts one direct PPU task; width {} needs {} horizontal \
         tiles, so use PoolingPlan::programs_with_buffers",
        shape.input_width,
        plan.tiles.len()
    );
    let mut cmds =
        build_ppu_standalone_flying(shape, &plan.tiles[0], bufs.input_addr, bufs.output_addr);
    push_ppu_kick(&mut cmds);
    cmds
}

//===========================================================================
// Pooling via a real DPU bypass stage, one bypass task followed by one or
// more separately-kicked PPU tasks -- the
// shape a real rknn-toolkit2-compiled model actually emits (see NOTES.md's
// "Decoding a real regcmd program for a pooling-only op" and its follow-up
// "Checked against iree-rocket-hal/src/" section), matching NEITHER of the
// two paths above:
//
// 1. A real (but trivial/near-identity) CNA->CORE->DPU task, built via
//    `conv::ConvPlan` (`dpu_output_mode=2` i.e. outside/memory, matching
//    the real capture's `DPU_FEATURE_MODE_CFG.output_mode=2`), writing its
//    output to a real intermediate buffer (`bufs.bypass_output_addr`).
//    Self-kicks via `ConvPlan`'s own PC trailer
//    (`PCOperationMask::CONVOLUTION` = `CNA|CORE|DPU|DPU_RDMA`) -- no
//    caller-supplied kick needed or wanted here; see this function's own
//    body comment for the real hardware history of getting that right
//    (`PC_OPERATION_ENABLE` is edge-triggered, so a redundant second kick
//    actively corrupts the result, not just wastes a write).
//    `CNA_WEIGHT_SIZE0/1/2`'s tiny values in the real capture, vs.
//    conv.rknn's real-kernel values, are consistent with this stage doing
//    a near-identity passthrough rather than real conv math -- reflected
//    here by `bypass_shape` being caller-supplied rather than hardcoded,
//    so a 1x1, zero-point=0, scale=1.0 identity-ish shape can be passed in
//    without this function assuming what "trivial" means numerically.
// 2. A second, separately-kicked task: PPU_RDMA fetches from
//    `bufs.bypass_output_addr` (the first stage's real memory output, not
//    an external caller-supplied input) and PPU pools it exactly like
//    `build_pooling_regcmd`'s standalone path -- reuses
//    `build_ppu_standalone_flying` verbatim. Kicked `KICK_PPU |
//    KICK_PPU_RDMA`, same as the real capture's `0x60` second kick.
//
// RESOLVED (real RK3588, first hardware round): the two engine kicks cannot
// be concatenated into one `rocket_task` regcmd buffer. Writing the
// convolution kick and then the PPU kick in one PC program left the bypass
// intermediate completely untouched: the second kick replaced the first
// before the convolution ran. That experiment established a task boundary,
// not a job boundary. Rocket's UAPI explicitly represents one job as an
// ordered array of complete task programs, and its IRQ handler advances to
// the next task on the same core. The vendor-style representation is
// therefore one `rocket_job` containing the bypass task followed by one PPU
// task per horizontal tile. `build_pooling_via_dpu_bypass_tasks` emits that
// ordered task list.
//===========================================================================

/// DMA addresses for the two-kick bypass-then-pool shape. `bypass_output_addr`
/// is a real intermediate buffer (unlike the pipelined path's on-chip
/// `dpu_flyin`, this shape genuinely round-trips through memory between the
/// two kicks, matching the real vendor capture) -- must be 16-byte aligned
/// for the same `PPU_RDMA` fetch-side reason `output_addr` must be (see
/// `PoolingBuffers::output_addr`'s doc comment), though PPU_RDMA's own
/// `src_base_addr` field is a full 32-bit address (unshifted), unlike
/// `PPU_DST_BASE_ADDR` -- 16-byte alignment is still required so the
/// bypass stage's own DPU output-side stride math stays consistent with
/// every other buffer in this module, not because PPU_RDMA's register width
/// demands it.
pub struct PoolingViaBypassBuffers {
    pub input_addr: u32,
    pub weights_addr: u32,
    pub bias_addr: u32,
    pub bypass_output_addr: u32,
    pub output_addr: u32,
}

/// Builds the vendor-style task sequence for a pooling operation:
///
/// 1. one complete CNA/CORE/DPU bypass program writing the intermediate;
/// 2. one complete PPU/PPU_RDMA program per horizontal pooling tile.
///
/// The returned programs are intended to be the ordered `rocket_task` array
/// of one `rocket_job`. Every entry has its own PC trailer and engine kick;
/// they must not be concatenated into one regcmd buffer. The kernel advances
/// between entries on completion IRQs and signals the job fence only after
/// the last tile.
///
/// This deliberately exposes drivers that do not route PPU completion IRQs:
/// such a driver can advance from the DPU bypass to PPU tile 0, but will
/// watchdog there instead of advancing to PPU tile 1. A successful wide
/// numerical test therefore validates both the emitted task chain and the
/// driver's PPU task-completion path.
pub fn build_pooling_via_dpu_bypass_tasks(
    bypass_shape: &conv::Shape,
    bypass_kernels: Kernels,
    pooling_shape: &PoolingShape,
    bufs: &PoolingViaBypassBuffers,
) -> Vec<Vec<RegCmd>> {
    pooling_shape.validate();
    let bypass_precision = match bypass_shape.precision {
        conv::Precision::Fp16 => PoolingPrecision::Fp16,
        conv::Precision::Int8(_) => PoolingPrecision::Int8,
    };
    assert_eq!(
        pooling_shape.precision, bypass_precision,
        "pooling precision must match the DPU bypass output precision"
    );
    assert_eq!(
        bypass_shape.output_width(bypass_kernels),
        pooling_shape.input_width,
        "DPU bypass output width must match pooling input width"
    );
    assert_eq!(
        bypass_shape.output_height(bypass_kernels),
        pooling_shape.input_height,
        "DPU bypass output height must match pooling input height"
    );
    assert_eq!(
        bypass_shape.out_channels, pooling_shape.input_channels,
        "DPU bypass output channels must match pooling input channels"
    );
    assert!(
        bufs.bypass_output_addr.is_multiple_of(16),
        "build_pooling_via_dpu_bypass_tasks: bypass_output_addr {:#x} is not \
         16-byte aligned",
        bufs.bypass_output_addr
    );
    assert!(
        matches!(bypass_shape.activation, ConvActivation::None),
        "build_pooling_via_dpu_bypass_tasks: bypass_shape.activation must be None -- \
         fused activation for this pooling path is expressed on `pooling_shape.activation` \
         (the op's own logical shape), not on the internal near-identity bypass conv shape, \
         so there's one canonical place a caller/HAL layer needs to set it. This function \
         applies `pooling_shape.activation` to the bypass stage's own fused-activation stage."
    );
    // The bypass stage is the only real DPU instance in this pooling path,
    // so fused activation rides on it, same as Phase 1's conv activation.
    // Only the `activation` field is overridden here; every other
    // geometry/quant field comes from the caller-supplied `bypass_shape`
    // unchanged. `conv::Activation` fuses through the BN stage (see that
    // type's own doc comment) rather than the retired Mesa builder's BS stage -- a
    // different port of the same hardware, not yet independently
    // hardware-validated in this specific bypass-then-pool composition
    // (only `cmp: 0`, which clamps to a constant zero regardless of which
    // stage or numeric domain applies it, has real hardware behind it here
    // -- see `pooling_via_bypass_relux_cmp_zero_forces_constant_output`).
    let bypass_activation = match pooling_shape.activation {
        Activation::None => ConvActivation::None,
        Activation::Relu => ConvActivation::Relu,
        Activation::Relux { cmp } => ConvActivation::Clamped { cmp },
    };
    let bypass_shape_with_activation = conv::Shape {
        activation: bypass_activation,
        ..*bypass_shape
    };

    // Stage 1: real (near-identity) CNA->CORE->DPU task, output to memory.
    // conv.rs's tile builder always programs DPU_FEATURE_MODE_CFG.output_mode
    // as external-memory (never on-chip dpu_flyin), which matches this
    // stage's own `2` (outside/memory) exactly -- and, per the module doc
    // comment's retirement note, the real vendor toolchain never emits the
    // on-chip-routed shape for a genuine conv+pool graph anyway.
    let mut bypass_tasks = ConvPlan::new(bypass_shape_with_activation, bypass_kernels)
        .programs_with_buffers(conv::Buffers {
            input: bufs.input_addr,
            weights: bufs.weights_addr,
            bias: bufs.bias_addr,
            output: bufs.bypass_output_addr,
        });
    assert_eq!(
        bypass_tasks.len(),
        1,
        "build_pooling_via_dpu_bypass_tasks: bypass conv requires {} CBUF height splits; \
         the vendor-style pooling path requires exactly one bypass task",
        bypass_tasks.len()
    );
    let bypass_cmds = bypass_tasks.remove(0);
    // NOT a push_kick() call here, deliberately: ConvPlan::programs_with_buffers
    // already ends this task with its own PC trailer
    // (PCTrailer::operation_enable(PCOperationMask::CONVOLUTION), which is
    // exactly CNA|CORE|DPU|DPU_RDMA -- see conv.rs's tile builder). An
    // earlier version of this function pushed a second, redundant kick here,
    // carried over unexamined from the pre-migration Mesa-derived stage 1
    // (that builder never self-kicked, so a
    // caller-supplied kick there was the ONLY kick, not a second one).
    // PC_OPERATION_ENABLE is edge-triggered, not passive state: a real RK3588
    // run showed the second write re-kicks the same blocks immediately after
    // the first, corrupting the result to an all-zero buf_mid (sentinel-fill
    // diagnostic caught it -- job completes, no hang, but the data is wrong)
    // instead of the harmless "last write wins" this was wrongly assumed to
    // be. The real vendor capture's own kick reading 0x0d (missing the
    // DPU_RDMA bit) is a fact about the retired builder's kick construction,
    // not evidence this stage's ConvPlan-emitted kick is incomplete.

    // Remaining tasks: standalone-flying PPU tiles, each fetching its
    // overlapping input range from stage 1's full memory output and writing
    // its disjoint output-column range into the shared final output.
    let plan = PoolingPlan::new(*pooling_shape);
    let mut tasks = Vec::with_capacity(1 + plan.tiles.len());
    tasks.push(bypass_cmds);
    tasks.extend(plan.tiles.iter().map(|tile| {
        let output_addr = bufs
            .output_addr
            .checked_add(tile.output_first * 16)
            .expect("pooling tile output address overflows u32");
        let mut commands =
            build_ppu_standalone_flying(pooling_shape, tile, bufs.bypass_output_addr, output_addr);
        push_ppu_kick(&mut commands);
        commands
    }));
    tasks
}

/// Compatibility wrapper for callers that require the original one-tile
/// tuple. Multi-tile pooling must use
/// [`build_pooling_via_dpu_bypass_tasks`] and submit all returned programs
/// as tasks of one job.
pub fn build_pooling_via_dpu_bypass_regcmd(
    bypass_shape: &conv::Shape,
    bypass_kernels: Kernels,
    pooling_shape: &PoolingShape,
    bufs: &PoolingViaBypassBuffers,
) -> (Vec<RegCmd>, Vec<RegCmd>) {
    let mut tasks =
        build_pooling_via_dpu_bypass_tasks(bypass_shape, bypass_kernels, pooling_shape, bufs);
    assert_eq!(
        tasks.len(),
        2,
        "build_pooling_via_dpu_bypass_regcmd accepts one PPU tile; width {} produces {} tiles, \
         so use build_pooling_via_dpu_bypass_tasks",
        pooling_shape.input_width,
        tasks.len() - 1
    );
    let pooling_cmds = tasks.pop().unwrap();
    let bypass_cmds = tasks.pop().unwrap();
    (bypass_cmds, pooling_cmds)
}

//===========================================================================
// Softmax, Phase 5 of the ukernel roadmap. Research spike, NOT a complete
// hardware-validated op yet -- see `rknpu-spelunking/NOTES.md`'s softmax
// Phase 5 section (all 5 "Follow-up"s) and
// `project_softmax_phase5_status.md` in this repo's memory for the full
// derivation. Two of three pieces are confirmed:
//
// 1. exp(x) itself is the standalone DPU LUT path above, unchanged --
//    `LutTable::exp()` plus `build_conv_then_lut_regcmd`/`build_lut_regcmd`
//    already handle it, no new code needed.
// 2. Max-reduction is a confirmed **N-stage tree** of (bypass-conv,
//    PPU-max-pool) pairs -- structurally identical to
//    `build_pooling_via_dpu_bypass_regcmd`'s single-stage shape, just
//    repeated with each stage's output feeding the next stage's input.
//    Live, self-consistent hardware capture found exactly 5 such stages
//    for one specific (32-channel) test shape, exactly matching the
//    static `.rknn` file's original prediction of 5
//    `PPU_OPERATION_MODE_CFG=0x11` blocks. `build_max_reduction_tree_regcmd`
//    below generalizes the PAIRED-STAGE SHAPE (not a specific stage count
//    or kernel-size schedule -- that part is caller-supplied, like every
//    other shape struct in this file) to arbitrary `N`.
//
// What's still missing, deliberately NOT implemented here (would be
// fabricating unconfirmed behavior rather than porting a proven recipe):
// - How the just-computed max actually gets subtracted from each element
//   before the exp LUT reads it. The confirmed exp dispatch's own
//   `BN_BASE_ADDR` was captured as 0, and its `SRC_BASE_ADDR` reads
//   directly from the graph's real conv output, not from the max-tree's
//   output buffer -- so the subtraction is NOT wired through DPU_RDMA's
//   per-position bias-buffer mechanism the way a first guess might
//   assume. Real data-flow between the max tree and the exp step is
//   still unmapped (see NOTES.md's Follow-up 5 "working hypothesis"
//   section).
// - The reciprocal/normalize (divide-by-sum) step. Exhaustively searched
//   for as a second DPU LUT/BN dispatch or a Sum/Avg-mode PPU block
//   across a complete, self-consistent capture -- found nowhere. Current
//   best guess is it isn't a distinct on-chip step in this graph at all
//   (folded into output requant, or done off-chip) -- nothing to port
//   until that's settled.
//
// There is deliberately no `build_softmax_regcmd` top-level entry point
// yet -- assembling one now would silently paper over the two gaps
// above with guesses. What's below is the reusable, structurally-
// confirmed piece (the max-reduction tree) plus a pointer to the
// already-existing exp LUT piece; wire them into a real op once the
// missing data-flow is confirmed on hardware.
//===========================================================================

/// One stage of a max-reduction tree: a near-identity bypass conv (same
/// role as `build_pooling_via_dpu_bypass_regcmd`'s `bypass_shape`) paired
/// with a PPU pooling stage that actually does the reduction. Caller
/// supplies both shapes explicitly per stage -- this module does not
/// guess a general "how many stages / what kernel size" schedule from an
/// element count; the one real hardware capture behind this code only
/// confirms the PAIRED-STAGE SHAPE generalizes, not any particular
/// tiling algorithm (see this section's module doc comment).
pub struct MaxReductionStage {
    pub bypass_shape: conv::Shape,
    pub bypass_kernels: Kernels,
    pub pooling_shape: PoolingShape,
}

/// DMA addresses for one stage of the tree. `output_addr` is this stage's
/// real memory output -- either the next stage's `bypass_input_addr`, or
/// (for the last stage) the tree's final reduced-max output.
pub struct MaxReductionStageBuffers {
    pub bypass_input_addr: u32,
    pub bypass_weights_addr: u32,
    pub bypass_bias_addr: u32,
    pub bypass_output_addr: u32,
    pub output_addr: u32,
}

/// Builds an `N`-stage max-reduction tree using the compatibility one-tile
/// wrapper. Returns one `(bypass_cmds, pooling_cmds)` pair per stage, in
/// order. Each vector is a complete task program; callers may flatten the
/// pairs in order into one `submit_tasks()` job, or use one ordered job per
/// pair and wait before submitting the dependent next stage. The specific
/// multi-stage chain is not yet independently hardware-validated
/// end-to-end; only the underlying single-stage dataflow is.
pub fn build_max_reduction_tree_regcmd(
    stages: &[MaxReductionStage],
    bufs: &[MaxReductionStageBuffers],
) -> Vec<(Vec<RegCmd>, Vec<RegCmd>)> {
    assert!(
        !stages.is_empty(),
        "build_max_reduction_tree_regcmd: at least one stage is required"
    );
    assert_eq!(
        stages.len(),
        bufs.len(),
        "build_max_reduction_tree_regcmd: one MaxReductionStageBuffers entry is required per stage"
    );
    stages
        .iter()
        .zip(bufs.iter())
        .map(|(stage, buf)| {
            build_pooling_via_dpu_bypass_regcmd(
                &stage.bypass_shape,
                stage.bypass_kernels,
                &stage.pooling_shape,
                &PoolingViaBypassBuffers {
                    input_addr: buf.bypass_input_addr,
                    weights_addr: buf.bypass_weights_addr,
                    bias_addr: buf.bypass_bias_addr,
                    bypass_output_addr: buf.bypass_output_addr,
                    output_addr: buf.output_addr,
                },
            )
        })
        .collect()
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
            activation: Activation::None,
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

    #[test]
    fn int8_program_matches_the_direct_vendor_capture_word_for_word() {
        // The pooling-sweep capture
        // `pool-max-w64-h48-c12-kw3-kh3-sx2-sy2-p1-1-1-1-i8.rknn` is an
        // exact direct PPU program. Addresses are zero in the static file
        // and are therefore also zero here.
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
            activation: Activation::None,
        };
        let commands = build_pooling_regcmd(
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
            let commands = build_pooling_regcmd(
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
    fn int8_average_reciprocals_match_the_convpool_capture() {
        // convpool-w32-h32-ci16-co16-k3-s1-pk2-ps2-pmavg-i8.rknn
        // emits an ordinary flying-mode PPU stage after its memory-writing
        // convolution task.
        let shape = PoolingShape {
            method: PoolingMethod::Avg,
            ..shape(32, 32, 2, 2, 2, 2, PoolingPrecision::Int8)
        };
        let commands = build_pooling_regcmd(
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
            let commands = build_pooling_regcmd(
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
        let commands = build_pooling_regcmd(
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
    fn supported_kernel_edge_keeps_the_full_stride_field_available() {
        let shape = shape(64, 48, 8, 8, 16, 16, PoolingPrecision::Int8);
        let commands = build_pooling_regcmd(
            &shape,
            &PoolingBuffers {
                input_addr: 0,
                output_addr: 0,
            },
        );
        assert_eq!(
            register_value::<PpuPoolingKernelCfg>(&commands),
            0x00ff_0707
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
    fn vendor_bypass_path_is_one_job_worth_of_three_complete_tasks() {
        let pooling_shape = shape(256, 9, 3, 3, 2, 2, PoolingPrecision::Int8);
        let bypass_shape = conv::Shape {
            width: pooling_shape.input_width,
            height: pooling_shape.input_height,
            stride: 1,
            in_channels: pooling_shape.input_channels,
            out_channels: pooling_shape.output_channels,
            precision: conv::Precision::Int8(conv::Quantization {
                input_zero_point: 0,
                output_zero_point: 0,
                multiplier: conv::Multiplier::from_ratio(1.0),
            }),
            padding: Some([0, 0]),
            activation: ConvActivation::None,
            depthwise: false,
        };
        let programs = build_pooling_via_dpu_bypass_tasks(
            &bypass_shape,
            [1, 1],
            &pooling_shape,
            &PoolingViaBypassBuffers {
                input_addr: 0x1000,
                weights_addr: 0x2000,
                bias_addr: 0x3000,
                bypass_output_addr: 0x4000,
                output_addr: 0x8000,
            },
        );

        assert_eq!(programs.len(), 3);
        assert!(programs[0].iter().any(|command| {
            command.0 == PCTrailer::operation_enable(PCOperationMask::CONVOLUTION).0
        }));
        for program in &programs[1..] {
            assert_eq!(program.len(), 32);
            assert!(program.iter().any(|command| {
                command.0
                    == PCTrailer::operation_enable(PCOperationMask::PPU | PCOperationMask::PPU_RDMA)
                        .0
            }));
        }

        assert_eq!(
            register_value::<PpuOperationModeCfg>(&programs[1]),
            0x0040_0011
        );
        assert_eq!(
            register_value::<PpuOperationModeCfg>(&programs[2]),
            0x003f_0011
        );
        assert_eq!(register_value::<PpuRdmaSrcBaseAddr>(&programs[1]), 0x4000);
        assert_eq!(register_value::<PpuRdmaSrcBaseAddr>(&programs[2]), 0x47e0);
        assert_eq!(register_value::<PpuDstBaseAddr>(&programs[1]), 0x8000);
        assert_eq!(register_value::<PpuDstBaseAddr>(&programs[2]), 0x83f0);
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
        let commands = build_pooling_regcmd(
            &padded,
            &PoolingBuffers {
                input_addr: 0,
                output_addr: 0,
            },
        );
        assert_eq!(register_value::<PpuPoolingPaddingCfg>(&commands), 0x11);

        padded.stride_x = 3;
        padded.output_width = 22;
        let commands = build_pooling_regcmd(
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
}
