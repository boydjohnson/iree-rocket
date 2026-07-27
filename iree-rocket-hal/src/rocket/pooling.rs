//! Pooling on the PPU, in all three shapes this hardware supports.
//!
//! - **Standalone flying** ([`build_pooling_regcmd`], TRM Ch.36 Fig 36-6):
//!   PPU_RDMA reads the input straight from memory and feeds PPU directly,
//!   bypassing CNA/CORE/DPU entirely. Kept for reference -- it is not what
//!   the vendor compiler emits, and its PPU_RDMA stride math remains
//!   unconfirmed.
//! - **Via a DPU bypass stage** ([`build_pooling_via_dpu_bypass_regcmd`]):
//!   two separately-kicked tasks, a near-identity conv followed by a PPU
//!   pass. This is the shape a real rknn-toolkit2-compiled model actually
//!   emits, and the one with hardware behind it.
//! - **Pipelined after a real conv** ([`build_conv_then_pooling_regcmd`]):
//!   DPU's output routed on-chip into PPU via `dpu_flyin`, no PPU_RDMA
//!   fetch at all.
//!
//! [`build_max_reduction_tree_regcmd`] chains the bypass shape into the
//! max-reduction half of a softmax; see its section comment for the two
//! gaps that keep this from being a whole softmax op.
//!
//! Mesa-derived in structure, though there is no Mesa/Teflon reference for
//! pooling itself (`rkt_ml.c` only ever implements convolution) -- the
//! fields come from the PPU/PPU_RDMA register layout plus TRM Ch.36 prose.
//! The eventual capture-derived builder for this op belongs in this same
//! module.

use crate::rocket::{
    activation::Activation,
    builders::{Bits, RegCmd, Register, ppu::*, ppu_rdma::*},
    mesa_conv::{
        ConvBuffers, ConvShape, build_conv_cna_core_dpu_dpu_rdma, require_single_conv_task,
    },
    regcmd::{
        KICK_CNA, KICK_CORE, KICK_DPU, KICK_DPU_RDMA, KICK_PPU, KICK_PPU_RDMA, push_kick, zero,
    },
};

//===========================================================================
// Pooling (standalone PPU, "flying mode" -- TRM Ch.36 Fig 36-6): PPU_RDMA
// reads the input straight from memory and feeds PPU directly, bypassing
// CNA/CORE/DPU entirely. There is NO Mesa/Teflon reference for this path
// (`rkt_ml.c` only ever implements convolution) -- every field below is
// derived from the PPU/PPU_RDMA register layout in `builders/ppu.rs` /
// `builders/ppu_rdma.rs` (bindgen'd from Mesa's own `registers.xml`, see
// builders.rs's DOMAIN_* comment) plus the TRM Ch.36 §4.6/§4.7 prose and
// `build_conv_regcmd`'s established conventions (N-1 encoding on every
// *_RDMA/CORE/DPU cube dimension, the same four-entry PC kick tail). NONE
// of the following has been hardware-validated yet -- see the UNCONFIRMED
// markers below and iree-rocket-hal/tests/pooling_hw.rs's doc comment for
// the sweep tests needed before trusting this in production:
//
// - RESOLVED (hardware-confirmed, real RK3588, via
//   `pooling_method_encoding_discovery`): `PoolingMethod`'s bit encoding is
//   Avg=0, Max=1, Min=2 -- see `PoolingMethod::bits()`'s doc comment.
// - PPU_RDMA's `src_line_stride`/`src_surf_stride` and PPU's
//   `dst_surf_stride`/`misc_ctrl.surf_len` formulas -- derived by analogy
//   to CNA's input-side and DPU's output-side stride math in
//   `build_conv_regcmd`; still not independently confirmed against a
//   non-uniform/asymmetric shape, but exercised without issue by every
//   `completes_and_output_tracks_input` test using this path.
// - RESOLVED (was open when this task hung real hardware, now hardware
//   re-confirmed working -- all of `pooling_hw.rs`'s tests pass): the kick
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
//   `build_pooling_via_dpu_bypass_regcmd` further down, which is the
//   recommended default going forward; this function is kept for hardware
//   comparison/reference.
//===========================================================================

/// Hardware-confirmed bit encoding (`iree-rocket-hal/tests/pooling_hw.rs`'s
/// `pooling_method_encoding_discovery`, real RK3588, via
/// `build_pooling_regcmd`'s standalone-flying path): a half-10/half-200
/// input produced raw=0 -> 249, raw=1 -> 250, raw=2 -> 248 -- i.e. raw=1 is
/// the real max, raw=2 is the real min, raw=0 sits in between (avg). The
/// original guess (Max=0, Min=1, Avg=2) had max and min swapped relative to
/// avg; corrected below.
#[derive(Clone, Copy)]
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

/// Logical shape of a single standalone pooling operation. Single-task
/// only, no CBUF-budget splitting to worry about (PPU has no CBUF -- that
/// concern is CNA/CORE-specific), no `index_en` output wiring yet (that
/// needs a second output buffer for argmax/argmin positions, not plumbed
/// here).
pub struct PoolingShape {
    pub input_width: u32,
    pub input_height: u32,
    pub input_channels: u32,
    pub output_width: u32,
    pub output_height: u32,
    pub output_channels: u32,
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
    /// stage ahead of PPU. `build_pooling_via_dpu_bypass_regcmd` applies it
    /// to that DPU stage's BS core (the same fusion point Phase 1 validated
    /// for conv); `build_pooling_regcmd`'s pure standalone-flying path (no
    /// DPU at all) cannot honor this and asserts it's `Activation::None`
    /// rather than silently ignoring a non-`None` value.
    pub activation: Activation,
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
    /// returned) but left the *entire* output buffer at zero (see
    /// pooling_hw.rs's `pooling_dump_full_output_buffer`, added
    /// specifically to check this). The 28-bit width is the same
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
/// `build_pooling_regcmd` so `build_pooling_via_dpu_bypass_regcmd` (the real
/// two-kick vendor shape, see its own doc comment) can reuse the exact same
/// PPU-stage sequence, differing only in where PPU_RDMA fetches from (a
/// caller-supplied real input buffer for the standalone path; a bypass
/// conv's memory-written output for the two-kick path) and in which kick
/// mask the caller appends afterwards. Pure extraction -- no behavior
/// change versus the original single function.
fn build_ppu_standalone_flying(
    shape: &PoolingShape,
    input_addr: u32,
    output_addr: u32,
) -> Vec<RegCmd> {
    assert!(
        output_addr.is_multiple_of(16),
        "build_ppu_standalone_flying: output_addr {output_addr:#x} is not 16-byte aligned -- \
         PPU_DST_BASE_ADDR is written as address >> 4 (see PoolingBuffers::output_addr's \
         doc comment), which silently drops any non-zero low 4 bits instead of failing \
         loudly, so this must be checked explicitly"
    );
    let dst_base_addr_shifted = output_addr >> 4;

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
    // still read correctly regardless, which is why every hardware test
    // using this function against a CPU-filled *uniform-whole-buffer* input
    // (pooling_hw.rs) never caught it -- reading 16x too far still landed on
    // the same repeated fill byte anywhere within the buffer. It only
    // surfaced once a real DPU-written (not uniformly-filled) source buffer
    // was read in `pooling_via_dpu_bypass_hw.rs`, where rows beyond row 0
    // jumped past the DPU's actual (small) write footprint into untouched
    // buffer contents.
    let src_line_stride = (shape.input_width * ATOMIC_K_SIZE) / FEATURE_ATOMIC_SIZE;
    let src_surf_stride =
        (shape.input_width * ATOMIC_K_SIZE * shape.input_height) / FEATURE_ATOMIC_SIZE;

    // UNCONFIRMED, derived by analogy to DPU's output-side stride math in
    // build_conv_regcmd -- see module doc comment.
    let dst_surf_stride =
        (shape.output_width * ATOMIC_K_SIZE * shape.output_height) / FEATURE_ATOMIC_SIZE;
    let surf_len = shape.output_width * shape.output_height;

    // Average-pooling's divide-as-multiply trick (TRM §4.6): precomputed
    // reciprocal of the kernel dimension, x2^16. Always computed (not just
    // under Avg) -- build_conv_regcmd's own precedent is to always fill
    // every register regardless of which branch is logically active.
    let recip_kernel_width = ((1u64 << 16) / shape.kernel_width as u64) as u32;
    let recip_kernel_height = ((1u64 << 16) / shape.kernel_height as u64) as u32;

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
    // PPU_RDMA -- standalone read side, feeds PPU directly from memory.
    // ========================================================================

    cmds.push(
        Register::<PpuRdmaCubeInWidth>::new()
            .cube_in_width(Bits::new(shape.input_width - 1))
            .build(),
    );
    cmds.push(
        Register::<PpuRdmaCubeInHeight>::new()
            .cube_in_height(Bits::new(shape.input_height - 1))
            .build(),
    );
    cmds.push(
        Register::<PpuRdmaCubeInChannel>::new()
            .cube_in_channel(Bits::new(shape.input_channels - 1))
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
    cmds.push(zero::<PpuRdmaDataFormat>()); // in_precision = 0 (int8)

    // ========================================================================
    // PPU
    // ========================================================================

    cmds.push(
        Register::<PpuDataCubeInWidth>::new()
            .cube_in_width(Bits::new(shape.input_width - 1))
            .build(),
    );
    cmds.push(
        Register::<PpuDataCubeInHeight>::new()
            .cube_in_height(Bits::new(shape.input_height - 1))
            .build(),
    );
    cmds.push(
        Register::<PpuDataCubeInChannel>::new()
            .cube_in_channel(Bits::new(shape.input_channels - 1))
            .build(),
    );
    cmds.push(
        Register::<PpuDataCubeOutWidth>::new()
            .cube_out_width(Bits::new(shape.output_width - 1))
            .build(),
    );
    cmds.push(
        Register::<PpuDataCubeOutHeight>::new()
            .cube_out_height(Bits::new(shape.output_height - 1))
            .build(),
    );
    cmds.push(
        Register::<PpuDataCubeOutChannel>::new()
            .cube_out_channel(Bits::new(shape.output_channels - 1))
            .build(),
    );

    cmds.push(
        Register::<PpuOperationModeCfg>::new()
            .pooling_method(Bits::new(shape.method.bits()))
            .flying_mode(Bits::new(1)) // standalone via PPU_RDMA, not pipelined after DPU
            .index_en(Bits::new(0)) // no argmax/argmin output wiring yet
            .use_cnt(Bits::new(0))
            .notch_addr(Bits::new(0))
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
            .pad_left(Bits::new(shape.pad_left))
            .pad_top(Bits::new(shape.pad_top))
            .pad_right(Bits::new(shape.pad_right))
            .pad_bottom(Bits::new(shape.pad_bottom))
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
    cmds.push(zero::<PpuDataFormat>()); // proc_precision=0 (int8), dpu_flyin=0 (standalone)
    cmds.push(
        Register::<PpuMiscCtrl>::new()
            .burst_len(Bits::new(15))
            .nonalign(Bits::new(0))
            .mc_surf_out(Bits::new(0))
            .surf_len(Bits::new(surf_len))
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
/// compiler emits at all; see `build_pooling_via_dpu_bypass_regcmd` below
/// for that real two-kick shape. Kept for hardware comparison/reference,
/// not as the recommended default.
pub fn build_pooling_regcmd(shape: &PoolingShape, bufs: &PoolingBuffers) -> Vec<RegCmd> {
    assert!(
        matches!(shape.activation, Activation::None),
        "build_pooling_regcmd: fused activation ({:?}) requested but this standalone-flying \
         path has no DPU stage at all -- PPU has no ALU/activation capability of its own (see \
         PoolingShape::activation's doc comment). Use build_pooling_via_dpu_bypass_regcmd \
         instead if fused activation is needed.",
        shape.activation
    );
    let mut cmds = build_ppu_standalone_flying(shape, bufs.input_addr, bufs.output_addr);
    push_kick(&mut cmds, KICK_PPU | KICK_PPU_RDMA);
    cmds
}

//===========================================================================
// Pooling via a real DPU bypass stage, two separately-kicked tasks -- the
// shape a real rknn-toolkit2-compiled model actually emits (see NOTES.md's
// "Decoding a real regcmd program for a pooling-only op" and its follow-up
// "Checked against iree-rocket-hal/src/" section), matching NEITHER of the
// two paths above:
//
// 1. A real (but trivial/near-identity) CNA->CORE->DPU task, same sequence
//    `build_conv_regcmd` already proves on hardware
//    (`build_conv_cna_core_dpu_dpu_rdma`, `dpu_output_mode=2` i.e. outside/
//    memory, matching the real capture's `DPU_FEATURE_MODE_CFG.output_mode
//    =2`), writing its output to a real intermediate buffer
//    (`bufs.bypass_output_addr`). Kicked `KICK_CNA | KICK_CORE | KICK_DPU |
//    KICK_DPU_RDMA` -- despite the real capture's kick reading `0x0d` (no
//    DPU_RDMA bit), real hardware evidence overrides that reading: even
//    with genuinely separate submit()/prep_bo() calls per task, omitting
//    DPU_RDMA left the intermediate buffer completely unwritten (see the
//    kick-fix note further down). `CNA_WEIGHT_SIZE0/1/2`'s tiny values in
//    the real capture, vs. conv.rknn's real-kernel values, are consistent
//    with this stage doing a near-identity passthrough rather than real
//    conv math -- reflected here by `bypass_shape` being caller-supplied
//    rather than hardcoded, so a 1x1, zero-point=0, scale=1.0 identity-ish
//    shape can be passed in without this function assuming what "trivial"
//    means numerically.
// 2. A second, separately-kicked task: PPU_RDMA fetches from
//    `bufs.bypass_output_addr` (the first stage's real memory output, not
//    an external caller-supplied input) and PPU pools it exactly like
//    `build_pooling_regcmd`'s standalone path -- reuses
//    `build_ppu_standalone_flying` verbatim. Kicked `KICK_PPU |
//    KICK_PPU_RDMA`, same as the real capture's `0x60` second kick.
//
// RESOLVED (real RK3588, first hardware round): the roadmap plan's Phase 0
// "one or two submit() calls?" question was originally guessed at "one" --
// reasoning that the real vendor capture's two kicks live inside one
// contiguous regcmd dump, and `drm_rocket_job`/`drm_rocket_task` (device.rs)
// carry a single opaque `regcmd`/`regcmd_count` per task, so nothing
// *structurally* prevented one combined blob. That guess was wrong: a
// single-`submit()` version of this (writing `PC_OPERATION_ENABLE=0x0d`
// then immediately `0x60`, no wait in between) left the bypass stage's
// intermediate buffer completely untouched (sentinel-fill diagnostic,
// `tests/pooling_via_dpu_bypass_hw.rs`'s
// `pooling_via_bypass_dump_intermediate_and_output_buffers`, real
// hardware) -- i.e. the CNA/CORE/DPU stage never got to actually run
// before the second write replaced which engines were enabled. The real
// vendor capture being one contiguous *file dump* doesn't mean the
// original driver issued it as one *submission* -- that inference doesn't
// hold. This function now returns two separate task command lists; the
// caller must `submit()` the first, `prep_bo()`-wait on
// `bufs.bypass_output_addr`'s buffer, *then* `submit()` the second --
// exactly like two independent dispatches, not one.
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

/// Returns `(stage_1_bypass_cmds, stage_2_pooling_cmds)` -- two independent
/// tasks, NOT one combined blob (see the module doc comment above for why:
/// a single-`submit()` version of this left the bypass stage's output
/// completely unwritten on real hardware). The caller must `submit()` and
/// fully `prep_bo()`-wait on stage 1 (specifically on
/// `bufs.bypass_output_addr`'s buffer) before `submit()`ing stage 2 --
/// exactly like two ordinary, separately-fenced dispatches.
pub fn build_pooling_via_dpu_bypass_regcmd(
    bypass_shape: &ConvShape,
    pooling_shape: &PoolingShape,
    bufs: &PoolingViaBypassBuffers,
) -> (Vec<RegCmd>, Vec<RegCmd>) {
    assert!(
        bufs.bypass_output_addr.is_multiple_of(16),
        "build_pooling_via_dpu_bypass_regcmd: bypass_output_addr {:#x} is not \
         16-byte aligned",
        bufs.bypass_output_addr
    );
    assert!(
        matches!(bypass_shape.activation, Activation::None),
        "build_pooling_via_dpu_bypass_regcmd: bypass_shape.activation must be None -- \
         fused activation for this pooling path is expressed on `pooling_shape.activation` \
         (the op's own logical shape), not on the internal near-identity bypass conv shape, \
         so there's one canonical place a caller/HAL layer needs to set it. This function \
         applies `pooling_shape.activation` to the bypass stage's BS core itself."
    );
    // Phase 3: the bypass stage is the only real DPU (BS-core) instance in
    // this pooling path, so fused activation rides on it exactly like
    // Phase 1's conv activation -- same enum, same BS-core fusion point,
    // already hardware-validated for conv. Only the `activation` field is
    // overridden here; every other geometry/quant field comes from the
    // caller-supplied `bypass_shape` unchanged.
    let bypass_shape_with_activation = ConvShape {
        activation: pooling_shape.activation,
        ..*bypass_shape
    };

    // Stage 1: real (near-identity) CNA->CORE->DPU task, output to memory.
    let bypass_task = require_single_conv_task(&bypass_shape_with_activation);
    let mut bypass_cmds = build_conv_cna_core_dpu_dpu_rdma(
        &bypass_shape_with_activation,
        &ConvBuffers {
            input_addr: bufs.input_addr,
            weights_addr: bufs.weights_addr,
            bias_addr: bufs.bias_addr,
            output_addr: bufs.bypass_output_addr,
        },
        &bypass_task,
        2, // outside/memory, matching the real capture
        None,
    );
    // KICK_DPU_RDMA included despite the real vendor capture's kick reading
    // 0x0d (no DPU_RDMA bit) -- hardware evidence from the two-submit fix
    // above overrides that reading: even with genuinely separate
    // submit()/prep_bo() calls, a 0x0d-kicked stage 1 left buf_mid
    // completely unwritten (sentinel-fill diagnostic, real hardware).
    // Every other memory-writing conv task in this crate
    // (`mesa_conv::build_conv_regcmd`, and even this module's own
    // build_conv_then_pooling_regcmd on-chip-routed stage) kicks
    // DPU_RDMA unconditionally -- DPU_RDMA's enable bit is apparently
    // required for the memory write-back to actually commit, regardless of
    // dpu_output_mode, contradicting the vendor capture's literal decoded
    // value (which may reflect something else about how the vendor
    // compiler's tasks share DPU_RDMA state across the larger multi-tile
    // program it came from, not a standalone single-task requirement).
    push_kick(
        &mut bypass_cmds,
        KICK_CNA | KICK_CORE | KICK_DPU | KICK_DPU_RDMA,
    );

    // Stage 2: standalone-flying PPU, fetching from stage 1's real output.
    let mut pooling_cmds =
        build_ppu_standalone_flying(pooling_shape, bufs.bypass_output_addr, bufs.output_addr);
    push_kick(&mut pooling_cmds, KICK_PPU | KICK_PPU_RDMA);

    (bypass_cmds, pooling_cmds)
}

//===========================================================================
// Pooling, pipelined directly after a real DPU stage (TRM Ch.36 Fig 36-4/
// 36-5's "do some pipeline surface ops" step, NOT the standalone/flying
// Fig 36-6 path `build_pooling_regcmd` implements above).
//
// Chosen after the standalone path hung real hardware (NPU job timeout +
// IOMMU "Enable stall request timed out"). Re-reading TRM Fig 36-6 itself
// ruled out one leading hypothesis for that hang -- the official standalone
// flow never configures CNA/CORE/DPU/DPU_RDMA at all, so leaving them
// zeroed (as `build_pooling_regcmd` already does) is not the bug. That left
// PPU_RDMA's own fetch-stride math (also unconfirmed) or the borrowed
// all-blocks "kick" as the remaining suspects -- the kick one is now
// confirmed and fixed (see `build_pooling_regcmd`'s kick-sequence comment
// and the `KICK_*` constants above `push_kick`); PPU_RDMA's stride math
// remains unconfirmed since this pipelined path doesn't use PPU_RDMA at
// all (see below).
//
// This path sidesteps the still-open stride-math question rather than
// resolving it: it runs a real, hardware-validated CNA->CORE->DPU
// convolution (the exact same sequence `build_conv_regcmd` already proves
// on hardware, reused via `build_conv_cna_core_dpu_dpu_rdma`) and routes
// DPU's output on-chip straight into PPU (`DPU_FEATURE_MODE_CFG.output_mode`
// bit0, `PPU_DATA_FORMAT.dpu_flyin`) instead of through PPU_RDMA. No
// PPU_RDMA fetch is used at all, so the kick is `KICK_CNA | KICK_CORE |
// KICK_DPU | KICK_DPU_RDMA | KICK_PPU` -- every block this task actually
// configures, PPU_RDMA correctly left both unconfigured and unkicked.
//
// Still NOT hardware-validated as of writing -- see tests/ for the
// hardware-in-the-loop test this needs before trusting it.
//===========================================================================

/// Pooling-specific parameters for the pipelined path. `conv_shape`'s own
/// `output_width`/`output_height`/`output_channels` become PPU's INPUT
/// shape (DPU's output, routed on-chip) -- this struct only carries
/// pooling's own output shape and kernel/pad/method parameters.
pub struct PipelinedPoolingShape {
    pub kernel_width: u32,
    pub kernel_height: u32,
    pub stride_x: u32,
    pub stride_y: u32,
    pub method: PoolingMethod,
    pub pad_left: u32,
    pub pad_top: u32,
    pub pad_right: u32,
    pub pad_bottom: u32,
    pub pad_value: u32,
    pub output_width: u32,
    pub output_height: u32,
    pub output_channels: u32,
}

/// DMA addresses for a conv-then-pipelined-pooling op. Unlike
/// `ConvBuffers`, there is no separate "conv output" address: DPU's output
/// is routed on-chip directly into PPU (`output_mode` bit0 only, bit1/
/// "outside" left clear), so it never needs a memory round-trip. Only the
/// final pooled result needs a real buffer.
pub struct ConvThenPoolingBuffers {
    pub input_addr: u32,
    pub weights_addr: u32,
    pub bias_addr: u32,
    pub output_addr: u32,
}

pub fn build_conv_then_pooling_regcmd(
    conv_shape: &ConvShape,
    pooling: &PipelinedPoolingShape,
    bufs: &ConvThenPoolingBuffers,
) -> Vec<RegCmd> {
    assert!(
        bufs.output_addr.is_multiple_of(16),
        "build_conv_then_pooling_regcmd: output_addr {:#x} is not 16-byte aligned -- \
         PPU_DST_BASE_ADDR is written as address >> 4 (see build_pooling_regcmd's \
         PoolingBuffers::output_addr doc comment for why)",
        bufs.output_addr
    );

    const FEATURE_ATOMIC_SIZE: u32 = 16;
    const ATOMIC_K_SIZE: u32 = 16;

    // dpu_output_mode = 1: bit0 only ("to PPU", on-chip) -- bit1
    // ("outside"/memory) deliberately left clear, see module doc comment.
    // The conv-stage buffers (input/weights/bias) are real; output_addr
    // there is irrelevant (never written) since bit1 is clear.
    let conv_task = require_single_conv_task(conv_shape);
    let mut cmds = build_conv_cna_core_dpu_dpu_rdma(
        conv_shape,
        &ConvBuffers {
            input_addr: bufs.input_addr,
            weights_addr: bufs.weights_addr,
            bias_addr: bufs.bias_addr,
            output_addr: 0,
        },
        &conv_task,
        1,
        None,
    );

    // ========================================================================
    // PPU -- pipelined directly after DPU (dpu_flyin=1), no PPU_RDMA
    // involved. TRM Fig 36-4/36-5's "do some pipeline surface ops" step.
    // ========================================================================

    cmds.push(
        Register::<PpuSPointer>::new()
            .pointer_pp_mode(Bits::new(1))
            .executer_pp_en(Bits::new(1))
            .pointer_pp_en(Bits::new(1))
            .build(),
    );

    cmds.push(
        Register::<PpuDataCubeInWidth>::new()
            .cube_in_width(Bits::new(conv_shape.output_width - 1))
            .build(),
    );
    cmds.push(
        Register::<PpuDataCubeInHeight>::new()
            .cube_in_height(Bits::new(conv_shape.output_height - 1))
            .build(),
    );
    cmds.push(
        Register::<PpuDataCubeInChannel>::new()
            .cube_in_channel(Bits::new(conv_shape.output_channels - 1))
            .build(),
    );
    cmds.push(
        Register::<PpuDataCubeOutWidth>::new()
            .cube_out_width(Bits::new(pooling.output_width - 1))
            .build(),
    );
    cmds.push(
        Register::<PpuDataCubeOutHeight>::new()
            .cube_out_height(Bits::new(pooling.output_height - 1))
            .build(),
    );
    cmds.push(
        Register::<PpuDataCubeOutChannel>::new()
            .cube_out_channel(Bits::new(pooling.output_channels - 1))
            .build(),
    );

    cmds.push(
        Register::<PpuOperationModeCfg>::new()
            .pooling_method(Bits::new(pooling.method.bits()))
            .flying_mode(Bits::new(0)) // pipelined after DPU, not standalone
            .index_en(Bits::new(0))
            .use_cnt(Bits::new(0))
            .notch_addr(Bits::new(0))
            .build(),
    );
    cmds.push(
        Register::<PpuPoolingKernelCfg>::new()
            .kernel_width(Bits::new(pooling.kernel_width - 1))
            .kernel_height(Bits::new(pooling.kernel_height - 1))
            .kernel_stride_width(Bits::new(pooling.stride_x - 1))
            .kernel_stride_height(Bits::new(pooling.stride_y - 1))
            .build(),
    );

    let recip_kernel_width = ((1u64 << 16) / pooling.kernel_width as u64) as u32;
    let recip_kernel_height = ((1u64 << 16) / pooling.kernel_height as u64) as u32;
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
            .pad_left(Bits::new(pooling.pad_left))
            .pad_top(Bits::new(pooling.pad_top))
            .pad_right(Bits::new(pooling.pad_right))
            .pad_bottom(Bits::new(pooling.pad_bottom))
            .build(),
    );
    cmds.push(
        Register::<PpuPaddingValue1Cfg>::new()
            .pad_value_0(Bits::new(pooling.pad_value))
            .build(),
    );
    cmds.push(zero::<PpuPaddingValue2Cfg>());

    // Tested both this (>>4, by analogy to PC_BASE_ADDRESS's documented
    // bits[31:4] convention) and the raw address directly against real
    // hardware with a small (0x3000) buffer address -- both produced
    // identical zero-effect results, ruling out address encoding as the
    // actual blocker. Keeping the shifted form since it's still the more
    // principled default absent evidence either way.
    cmds.push(
        Register::<PpuDstBaseAddr>::new()
            .dst_base_addr(Bits::new(bufs.output_addr >> 4))
            .build(),
    );
    let ppu_dst_surf_stride =
        (pooling.output_width * ATOMIC_K_SIZE * pooling.output_height) / FEATURE_ATOMIC_SIZE;
    cmds.push(
        Register::<PpuDstSurfStride>::new()
            .dst_surf_stride(Bits::new(ppu_dst_surf_stride))
            .build(),
    );
    cmds.push(
        Register::<PpuDataFormat>::new()
            .dpu_flyin(Bits::new(1)) // source is DPU's on-chip output, not PPU_RDMA
            .build(),
    );
    let surf_len = pooling.output_width * pooling.output_height;
    cmds.push(
        Register::<PpuMiscCtrl>::new()
            .burst_len(Bits::new(15))
            .nonalign(Bits::new(0))
            .mc_surf_out(Bits::new(0))
            .surf_len(Bits::new(surf_len))
            .build(),
    );

    // Previously added an explicit PPU_OPERATION_ENABLE write here (PPU's
    // own DOMAIN_PPU-tagged register) because the shared kick's fixed
    // CNA/CORE/DPU/DPU_RDMA bitmask never set PPU's enable bit, and there
    // was no evidence either mechanism actually enabled PPU. There's now
    // real evidence: a real rknn-toolkit2-compiled pooling.rknn capture
    // (see NOTES.md in rknpu-spelunking, "Decoding a real regcmd program
    // for a pooling-only op") never writes any per-block *_OPERATION_ENABLE
    // register for *any* block, including PPU -- only the broadcast PC
    // kick, with a bitmask scoped to exactly the blocks that task uses.
    // Dropped the per-block write and folded KICK_PPU into the kick below
    // instead, matching that real pattern. NOT YET HARDWARE-VALIDATED.
    push_kick(
        &mut cmds,
        KICK_CNA | KICK_CORE | KICK_DPU | KICK_DPU_RDMA | KICK_PPU,
    );

    cmds
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
    pub bypass_shape: ConvShape,
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

/// Builds an `N`-stage max-reduction tree, each stage reusing
/// `build_pooling_via_dpu_bypass_regcmd` unchanged. Returns one
/// `(bypass_cmds, pooling_cmds)` pair per stage, in order -- the caller
/// must `submit()`+`prep_bo()`-wait through EVERY pair sequentially
/// (bypass stage, then pooling stage, then the next stage's bypass,
/// ...), exactly like `build_pooling_via_dpu_bypass_regcmd`'s own
/// single-stage discipline (see its doc comment for why: a combined
/// single-`submit_tasks()` job silently left one real hardware round
/// unwritten). Chaining `N>1` stages this way has the same *mechanism*
/// as the proven `N=1` case, but the specific 5-stage chain itself is
/// NOT YET independently hardware-validated end-to-end through this
/// exact function -- only the underlying single-stage building block is.
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
