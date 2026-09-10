//! `iree_hal_command_buffer_vtable_t`. `dispatch` is the real regcmd
//! integration point: turns the executable's `UkernelShape`
//! (`executable::shape`) plus this dispatch's buffer bindings into a
//! regcmd task list via the matching `iree_rocket_hal::rocket::regcmd::
//! build_*_regcmd` function and stashes it (regcmd tasks + the real GEM
//! handles it touches, see `RecordedOp::Dispatch`) on the command buffer,
//! to be submitted as one job by `device::queue_execute`.
//!
//! Binding convention is per-ukernel-kind: `Conv2d` is binding 0 = input,
//! 1 = weights, 2 = bias, 3 = output; `Pooling` is binding 0 = input,
//! 1 = output (no weights/bias). This started as an arbitrary placeholder
//! of our own choosing (no real compiler emitted a binding layout), but as
//! of `executable_cache.rs`'s tag `3` (a real, versioned wire format for
//! `ConvShape`, see that module's doc comment and
//! `iree_rocket_hal::rocket::executable_format`), it's a **frozen
//! cross-repo ABI contract**: a real IREE compiler `TargetBackend` (not
//! part of this crate, not yet written) would need to emit dispatches
//! respecting this exact binding order, since nothing here validates
//! binding *semantics* -- only binding *count* is checked (see the
//! `bindings.count < 4`/`< 2` guards below).
//!
//! `Conv2d`'s `build_conv_regcmd_tasks` call is wrapped in `catch_unwind` (see
//! below) as a backstop: `executable_cache.rs`'s `validate_conv_shape` is
//! deliberately not proven exhaustive of every panic reachable from
//! `build_conv_regcmd` (some of its internal `assert!`s, and every
//! register-field `Bits::<N>::new` bit-width check, are gated on formulas
//! derived from the shape rather than direct fields) -- without this,
//! any validation gap would abort the whole host process (an unwind
//! crossing a plain `extern "C"` boundary calls `process::abort()` since
//! Rust 1.71, uncatchable as an IREE status) instead of failing this one
//! dispatch gracefully.
//!
//! Supports recording multiple `dispatch` calls per command buffer --
//! `apply_ops` returns every recorded dispatch's regcmd program, in call
//! order, and `device::queue_execute` submits each as its own individually
//! fenced hardware job (the same "submit, then `prep_bo`-wait before the
//! next" sequencing already used for one dispatch's own CBUF-height-split
//! task list, since the mainline driver's inter-task IRQ transition isn't
//! reliable on RK3588 -- see that function's comment). Originally this
//! rejected a second recorded dispatch outright: real compiled programs
//! (a ResNet50 bottleneck block's shortcut projection and its first reduce
//! conv, both independent 1x1 convs reading the same upstream activation)
//! route two independent Rocket dispatches into one stream partition/
//! command buffer whenever nothing forces them apart, so "one only" broke
//! on the first real model exercising that shape, not just a hypothetical.
//! `collective` (multi-device reduce/broadcast/etc.) isn't applicable to a
//! single discrete NPU and stays UNIMPLEMENTED indefinitely, not just for
//! now.
//!
//! `fill_buffer`/`update_buffer`/`copy_buffer` all operate on our own
//! permanently-host-mapped `RocketBuffer`s (see buffer.rs), so they're
//! implemented purely host-side via IREE's own generic
//! `iree_hal_buffer_map_fill`/`_write`/`_copy` helpers (the same primitives
//! `iree/hal/local/inline_command_buffer.c` uses) -- no ioctl/hardware
//! involvement needed. Like `dispatch`, they're *recorded* here (as
//! `RecordedOp` entries, in call order) and only actually applied by
//! `device::queue_execute`'s `apply_ops` call, after the wait-semaphore
//! gate -- executing them immediately at record time (the way IREE's own
//! `ALLOW_INLINE_EXECUTION` mode does) would violate the wait-before-
//! execute contract every other command-buffer category here already
//! relies on. `update_buffer`'s source bytes are copied into the recorded
//! op immediately (the caller's source buffer isn't guaranteed to outlive
//! the recording call), matching `iree_hal_deferred_command_buffer_t`'s
//! identical requirement.

use std::{
    os::fd::{AsRawFd, BorrowedFd, RawFd},
    sync::Arc,
};

use crate::{
    bindings::{
        iree_const_byte_span_t, iree_device_size_t, iree_hal_buffer_barrier_t,
        iree_hal_buffer_ref_list_t, iree_hal_buffer_ref_t, iree_hal_buffer_t, iree_hal_channel_t,
        iree_hal_collective_op_t, iree_hal_command_buffer_mode_t, iree_hal_command_buffer_t,
        iree_hal_command_buffer_vtable_t, iree_hal_command_category_t, iree_hal_copy_flags_t,
        iree_hal_dispatch_config_t, iree_hal_dispatch_flags_t, iree_hal_event_t,
        iree_hal_executable_function_t, iree_hal_executable_t, iree_hal_execution_barrier_flags_t,
        iree_hal_execution_stage_t, iree_hal_fill_flags_t, iree_hal_label_color_t,
        iree_hal_label_location_t, iree_hal_memory_advise_flags_t, iree_hal_memory_barrier_t,
        iree_hal_queue_affinity_t, iree_hal_update_flags_t, iree_host_size_t, iree_status_t,
        iree_string_view_t,
    },
    buffer::RocketBuffer,
    executable::UkernelShape,
    profile,
    scratch_pool::ScratchBuffer as RocketOwnedBuffer,
    status, weight_cache,
};
use iree_rocket_hal::rocket::{
    activation::{LutBuffers, build_lut_regcmd},
    builders::RegCmd,
    conv::{
        AccumulatorOutputTile, Activation, Buffers, ConvPlan, FeatureLayout, Precision, relocate,
        relocate_staged_accumulator,
    },
    device::{OwnedBuffer as RocketGemBuffer, fini_bo},
    elementwise::{
        EwAddBuffers, EwAddShape, EwBinaryOp, EwPrecision, EwUnaryBuffers, build_add_regcmd,
        build_add_regcmd_with_relu, build_unary_regcmd,
    },
    fc,
    layout::{ChainRefusal, CubeGeometry, CubeKind, chain_identity, cube_geometry},
    pooling::{PoolingBuffers, PoolingPlan},
    tensor_layout::{
        nc1hwc2_storage_size, pack_depthwise_to_rocket_weights, pack_fp16_bias_to_rocket,
        pack_hwcf_to_rocket_weights, pack_hwcf_to_rocket_weights_affine_i8,
        pack_hwcf_to_rocket_weights_padded, pack_nhwc_to_nc1hwc2_padded,
        rocket_fp16_bias_storage_size, rocket_weight_storage_size,
    },
};

#[derive(Clone, Copy)]
pub enum InputPackingLayout {
    Dense,
    Nc1hwc2,
}

/// Defers dense-NHWC to NC1HWC2 packing until command-buffer execution.
///
/// The scratch allocation and its DMA address are fixed while recording so
/// the regcmd can be built immediately. The copy itself must happen later,
/// Where a [`StagedCopy`] reads from.
pub enum CopySource {
    /// An IREE binding the hardware would have read directly on context 0.
    Binding {
        buffer: *mut iree_hal_buffer_t,
        offset: usize,
    },
    /// Another context's cached coefficient packing, immutable once
    /// published. `weight_buffer` is the binding it was packed from, whose
    /// generation the copy is published under.
    Shared {
        source: Arc<weight_cache::SharedBuffer>,
        weight_buffer: *mut iree_hal_buffer_t,
    },
}

/// A byte copy into a non-zero context's own scratch, applied by
/// `apply_ops` before the job that reads it is submitted.
///
/// MULTICORE.md §3/§5.2: a job on context `k` may only name BOs created on
/// context `k`'s file. Almost every operand already goes through
/// driver-private scratch (packing, compaction); the few that the hardware
/// reads straight out of an IREE buffer on context 0 -- the ARGB feature
/// input for `Cin <= 4`, coefficients and bias that need no packing -- are
/// copied here instead, and so is a cached packing that lives on another
/// context. On context 0 nothing is staged and no byte moves that did not
/// move before.
pub struct StagedCopy {
    pub source: CopySource,
    pub length: usize,
    pub scratch_ptr: *mut u8,
    pub scratch_handle: u32,
    /// For a `Shared` source: the packing to publish under this context's
    /// key once the copy is flushed, so the next command buffer here hits.
    pub publish: Option<(WeightPublish, Arc<weight_cache::SharedBuffer>)>,
}

/// Where a replica buffer's bytes come from when a dispatch fans out.
#[derive(Clone, Copy)]
pub enum ReplicaSource {
    /// A host pointer valid for the command buffer's life: the home
    /// context's packed scratch, filled by `apply_ops` before the replica
    /// copies run.
    Host(*const u8),
    /// An IREE binding the home context hands to the hardware directly.
    Binding {
        buffer: *mut iree_hal_buffer_t,
        offset: usize,
    },
}

/// How an input scratch is laid out, so a replica can copy just the rows
/// its tiles read: `surfaces` planes of `surface_stride` bytes, each holding
/// `block_bytes` per pixel of a `width` x `height` image (NC1HWC2), or one
/// dense plane with `block_bytes` per pixel and a stride of zero.
#[derive(Clone, Copy)]
pub struct BandGeometry {
    pub width: usize,
    pub height: usize,
    pub surfaces: usize,
    pub surface_stride: usize,
    pub block_bytes: usize,
}

/// One tile's input rectangle, in input pixels.
#[derive(Clone, Copy, Debug)]
pub struct InputBand {
    pub row: usize,
    pub rows: usize,
    pub column: usize,
    pub columns: usize,
}

/// One operand copied onto a sibling context for the tiles that run there.
/// With `bands` set only those rectangles of the source are copied (the
/// tiles' input rows plus halo, MULTICORE.md §5.4 level 2); otherwise the
/// whole `length`.
pub struct ReplicaCopy {
    pub buffer: RocketOwnedBuffer,
    pub source: ReplicaSource,
    pub length: usize,
    pub geometry: Option<BandGeometry>,
    pub bands: Vec<InputBand>,
}

/// A sibling context's coefficients: a copy of an unpacked binding, or a
/// packing shared through `weight_cache` (a hit on that context, or a
/// copy of the home packing published there once it lands).
pub enum ReplicaWeights {
    Direct(ReplicaCopy),
    Packed {
        buffer: Arc<weight_cache::SharedBuffer>,
        copy: Option<(ReplicaSource, usize, Option<WeightPublish>)>,
    },
}

impl ReplicaWeights {
    fn dma_address(&self) -> u32 {
        match self {
            ReplicaWeights::Direct(copy) => copy.buffer.dma_address,
            ReplicaWeights::Packed { buffer, .. } => buffer.dma_address,
        }
    }

    fn handle(&self) -> u32 {
        match self {
            ReplicaWeights::Direct(copy) => copy.buffer.handle,
            ReplicaWeights::Packed { buffer, .. } => buffer.handle,
        }
    }
}

/// A sibling context's full copy of one dispatch's operands, so the tiles
/// placed there (MULTICORE.md §5.4 level 2) can name buffers on their own
/// file. Each tile writes its own rows of `output`, and the gather
/// compaction reads them from here.
///
/// A replica copies the whole input rather than the tile's band: at
/// ~10 GB/s the copy is a few percent of the tile's hardware time, and
/// it keeps the tile programs identical to the single-context ones --
/// only their four base addresses change.
pub struct Replica {
    pub context: Arc<crate::pool::NpuContext>,
    pub input: ReplicaCopy,
    pub weights: ReplicaWeights,
    pub bias: ReplicaCopy,
    pub output: RocketOwnedBuffer,
}

impl Replica {
    fn buffers(&self) -> Buffers {
        Buffers {
            input: self.input.buffer.dma_address,
            weights: self.weights.dma_address(),
            bias: self.bias.buffer.dma_address,
            output: self.output.dma_address,
        }
    }
}

/// The output rectangle one tile writes, in output pixels, plus its index
/// into the plan's tile list (which is also its index into a staged
/// accumulator layout).
#[derive(Clone, Copy, Debug)]
pub struct TileRect {
    pub index: usize,
    pub row: usize,
    pub rows: usize,
    pub column: usize,
    pub columns: usize,
}

/// Which file one task of a dispatch is submitted on and the BOs it names
/// there, for `device::queue_execute`.
pub struct TaskTarget {
    pub fd: RawFd,
    pub context: usize,
    pub in_bo_handles: Vec<u32>,
    pub out_bo_handles: Vec<u32>,
}

/// What a fanned-out replica needs to know about the home context's
/// coefficients: where to copy them from and what to publish the copy as.
#[derive(Clone, Copy)]
struct WeightFanoutSource {
    key: Option<weight_cache::Key>,
    generation: u64,
    /// False when an earlier op on this command buffer writes the weight
    /// binding, in which case no packing of it may be published.
    publishable: bool,
    length: usize,
}

/// Whether multi-tile dispatches spread their tiles over sibling contexts
/// (`ROCKET_FANOUT=0` keeps every tile on the command buffer's own
/// context, which is M1's behaviour).
fn fanout_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ROCKET_FANOUT").map_or(true, |value| value != "0"))
}

/// Whether a dispatch may read a preceding dispatch's output cube in place
/// instead of repacking the dense buffer that cube was compacted into
/// (`ROCKET_CHAIN=0` restores the unconditional repack). `ROCKET_CHAIN=debug`
/// additionally names every edge it takes and every one it declines, which is
/// the only way to tell "the shape does not qualify" from "no producer was
/// found" -- see [`chainable_cube`]. ISSUES.md P2 step 2.
fn chain_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ROCKET_CHAIN").map_or(true, |value| value != "0"))
}

fn chain_debug() -> bool {
    static DEBUG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DEBUG.get_or_init(|| std::env::var("ROCKET_CHAIN").is_ok_and(|value| value == "debug"))
}

/// Whether a dispatch may skip writing its dense output buffer when the
/// compiler counted its readers and every one of them chained to its output
/// cube on this command buffer (`ROCKET_LAZY_COMPACT=0` restores the
/// unconditional compaction, `=debug` names every decision). The count comes
/// down as the last push constant (`Conv2DDef.runtime_dense_readers`); see
/// [`compaction_elidable`] for the rule and ISSUES.md P2 for why the signal
/// has to come from the compiler.
fn lazy_compact_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        chain_enabled() && std::env::var("ROCKET_LAZY_COMPACT").map_or(true, |value| value != "0")
    })
}

fn lazy_compact_debug() -> bool {
    static DEBUG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DEBUG.get_or_init(|| std::env::var("ROCKET_LAZY_COMPACT").is_ok_and(|value| value == "debug"))
}

/// The one rule that decides whether a dispatch's dense output write can be
/// skipped, kept pure so it can be pinned by a test.
///
/// `dense_readers` is the compiler's count of Rocket dispatches that read the
/// result in the final program, or 0 if it saw any other reader or did not
/// count. `chained_readers` is how many consumers on this command buffer took
/// the output cube in place; `dense_read_seen` is whether anything recorded
/// on this command buffer read the dense bytes instead (a consumer that
/// declined to chain, a dispatch kind that cannot chain, a copy).
///
/// Every failure mode keeps the write: a reader on a later command buffer
/// leaves `chained_readers` short of the count, a same-buffer reader that
/// did not chain sets `dense_read_seen`, and an executable compiled without
/// the count says 0. Over-counting on the compiler's side can only ever
/// keep a write that could have been skipped, never skip one that was
/// needed, since a consumer that chained cannot also need the dense bytes.
/// Dense output writes skipped so far in this process; `profile::report`
/// prints it next to the `compact` phase.
pub static ELIDED_COMPACTIONS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

pub fn compaction_elidable(
    dense_readers: u32,
    chained_readers: u32,
    dense_read_seen: bool,
) -> bool {
    dense_readers > 0 && chained_readers == dense_readers && !dense_read_seen
}

/// Records that `binding`'s bytes are about to be read from the dense buffer
/// by the dispatch or copy being recorded, so no earlier dispatch that wrote
/// them may skip its compaction. Every read that does not go through
/// [`chainable_cube`] must pass here; a missed call site is a silent
/// all-zero read, which is why the conservative direction is the cheap one.
///
/// # Safety
///
/// `binding.buffer` and every recorded write target must be live
/// `RocketBuffer`s -- the same contract as [`chainable_cube`].
unsafe fn note_dense_read(cb: &mut RocketCommandBuffer, binding: &iree_hal_buffer_ref_t) {
    if binding.buffer.is_null() {
        return;
    }
    let Some(read) = (unsafe { dma_range(binding.buffer, binding.offset, binding.length) }) else {
        return;
    };
    for op in cb.ops.iter_mut() {
        let overlaps = recorded_write_extent(op)
            .and_then(|(buffer, offset, length)| unsafe { dma_range(buffer, offset, length) })
            .is_some_and(|wrote| wrote.0 < read.1 && read.0 < wrote.1);
        if let (
            true,
            RecordedOp::Dispatch {
                dense_read_seen, ..
            },
        ) = (overlaps, op)
        {
            *dense_read_seen = true;
        }
    }
}

/// after preceding recorded update/fill/copy operations have populated the
/// real IREE input buffer.
#[derive(Clone, Copy)]
pub struct InputPacking {
    pub input_buffer: *mut iree_hal_buffer_t,
    pub input_offset: iree_device_size_t,
    pub input_length: iree_device_size_t,
    pub scratch_ptr: *mut u8,
    pub scratch_length: usize,
    pub scratch_handle: u32,
    pub source_pixel_count: usize,
    pub packed_pixel_count: usize,
    pub bytes_per_pixel: usize,
    pub packed_bytes_per_pixel: usize,
    pub padding_byte: u8,
    pub layout: InputPackingLayout,
}

/// Defers logical HWCF-to-Rocket coefficient packing until execution.
///
/// The original binding can be populated by earlier recorded operations, so
/// the copy cannot happen while recording the dispatch. The regcmd points at
/// `scratch_handle`; `apply_ops` fills and flushes it immediately before the
/// hardware submission.
#[derive(Clone, Copy)]
pub struct WeightPacking {
    pub weight_buffer: *mut iree_hal_buffer_t,
    pub weight_offset: iree_device_size_t,
    pub weight_length: iree_device_size_t,
    pub scratch_ptr: *mut u8,
    pub scratch_length: usize,
    pub scratch_handle: u32,
    pub filter_height: usize,
    pub filter_width: usize,
    pub input_channels: usize,
    /// Logical output channels present in the source binding.
    pub output_channels: usize,
    /// Physical output channels encoded in the regcmd and packed weights.
    pub programmed_output_channels: usize,
    pub element_size: usize,
    /// One filter per input channel (no `(input, output)` pairing) packed
    /// tap-major via [`pack_depthwise_to_rocket_weights`] instead of
    /// [`pack_hwcf_to_rocket_weights`]'s blocked dense order. `output_channels`
    /// is unused in this mode -- Cout is always Cin, per
    /// `iree-rocket-hal`'s `Shape::with_depthwise` -- and `padded_channels`
    /// is read instead.
    pub depthwise: bool,
    /// Tap-major stride, only meaningful when `depthwise` is set. See
    /// `iree-rocket-hal`'s `Shape::depthwise_padded_channels`.
    pub padded_channels: usize,
    pub weight_zero_point: Option<i8>,
    /// Set only under `ROCKET_WEIGHT_CACHE=verify`: the cached buffer the
    /// regcmd actually points at. The packing above then runs into a private
    /// probe buffer instead, and `apply_ops` compares the two -- so a stale
    /// entry, a missed generation bump or a key that fails to separate two
    /// different packings fails loudly instead of quietly changing results.
    pub verify_against: Option<*const u8>,
}

/// What `apply_ops` needs to make a freshly packed buffer reusable.
///
/// The generation is deliberately not carried from record time: it is read
/// again immediately *before* packing, so a concurrent write to the weight
/// binding lands on a higher generation than the entry is published under
/// and can never be mistaken for the bytes that were packed.
#[derive(Clone, Copy)]
pub struct WeightPublish {
    pub key: weight_cache::Key,
    pub bytes: usize,
}

/// Where a dispatch's packed coefficients came from, and what has to happen
/// to them -- see `weight_cache`.
struct StagedWeights {
    /// DMA address the regcmd's coefficient read is pointed at.
    addr: u32,
    /// GEM handle for the job's `in_bo_handles`.
    handle: u32,
    /// `None` on a cache hit: the coefficients are already packed.
    packing: Option<WeightPacking>,
    /// The packed buffer, held so it outlives this command buffer's
    /// execution whether the cache or this dispatch created it.
    scratch: Option<Arc<weight_cache::SharedBuffer>>,
    /// Set when this dispatch packed its own: publish it once that succeeds.
    publish: Option<WeightPublish>,
    /// Verify mode only: the private buffer the re-pack lands in.
    probe: Option<RocketOwnedBuffer>,
    /// What a replica on another context copies, see `build_replicas`.
    fanout: WeightFanoutSource,
}

impl StagedWeights {
    /// Coefficients the hardware reads straight out of the IREE binding,
    /// with no packing and nothing to cache (the depthwise-free int8 path).
    fn direct(addr: u32, handle: u32, length: usize) -> StagedWeights {
        StagedWeights {
            addr,
            handle,
            packing: None,
            scratch: None,
            publish: None,
            probe: None,
            fanout: WeightFanoutSource {
                key: None,
                generation: 0,
                publishable: false,
                length,
            },
        }
    }
}

/// The buffer a recorded operation writes, if it writes one.
///
/// `weight_cache` refuses a hit when an operation already recorded on this
/// command buffer targets the weight binding: that write has not been
/// applied yet, so the generation counter cannot see it, and reusing a
/// buffer packed from the pre-write bytes would silently use stale weights.
fn recorded_write_target(op: &RecordedOp) -> Option<*mut iree_hal_buffer_t> {
    recorded_write_extent(op).map(|(buffer, _, _)| buffer)
}

/// [`recorded_write_target`] with the byte range the operation actually
/// writes, as `(buffer, offset, length)`.
///
/// IREE hands out one `iree_hal_buffer_t` per allocation and packs several
/// transient tensors into it at different offsets, so the buffer alone does
/// not say whether two operations touch the same bytes -- which is all
/// `recorded_write_target`'s caller needs and not nearly enough for
/// [`chainable_cube`], where treating a neighbouring tensor's write as a
/// blocker declined every residual skip in ResNet50.
///
/// A dispatch's length is what its compaction writes, which `queue_execute`
/// checks against the bytes it actually produced, not the binding's declared
/// length.
fn recorded_write_extent(
    op: &RecordedOp,
) -> Option<(*mut iree_hal_buffer_t, iree_device_size_t, usize)> {
    match op {
        RecordedOp::Fill { target, .. } | RecordedOp::Update { target, .. } => {
            Some((target.buffer, target.offset, target.length))
        }
        RecordedOp::Copy { target, .. } => Some((target.buffer, target.offset, target.length)),
        RecordedOp::Dispatch {
            output_compaction, ..
        } => output_compaction.as_ref().map(|oc| {
            (
                oc.output_buffer,
                oc.output_offset,
                oc.output_pixel_count * oc.bytes_per_pixel,
            )
        }),
    }
}

/// The device-visible byte range a binding covers, which is what decides
/// whether two recorded operations touch the same memory.
///
/// Comparing `iree_hal_buffer_t` pointers would be enough for the buffers
/// IREE hands this driver today, but the DMA address is the identity the
/// hardware itself uses: two buffer objects that alias one allocation share
/// it, and a stale chain served from a producer's scratch is silent.
///
/// # Safety
///
/// `buffer` must be a live `RocketBuffer`, which every direct binding
/// recorded on this command buffer is (`stage_direct` casts the same way at
/// the same point).
unsafe fn dma_range(
    buffer: *mut iree_hal_buffer_t,
    offset: iree_device_size_t,
    length: usize,
) -> Option<(u64, u64)> {
    if buffer.is_null() {
        return None;
    }
    let base = unsafe { &*(buffer as *const RocketBuffer) }.dma_address as u64 + offset as u64;
    Some((base, base + length as u64))
}

/// The output cube a dispatch may read in place of repacking `binding`.
///
/// Walks the ops recorded so far backwards to the most recent write whose
/// device-visible bytes overlap the ones this dispatch reads. That write is
/// the only one whose bytes the consumer can see, so if it is not a dispatch
/// offering a compatible cube -- a fill, a copy, an update, a fanned-out
/// dispatch, an accumulator dispatch, a partial overlap -- there is nothing
/// to chain and the caller repacks as before. Writes that miss the range are
/// skipped: IREE packs neighbouring transients into one allocation, and
/// treating those as blockers declined every residual skip in ResNet50.
///
/// The compatibility test is exactly the condition under which
/// `pack_nhwc_to_nc1hwc2_padded(compact_atomic_output(cube))` reproduces
/// `cube`:
///
/// - the identical device byte range, so the consumer really is reading what
///   the producer wrote and not a tensor that merely shares its allocation;
/// - equal pixel counts, since the surface stride is `pixels * 16` on both
///   sides and a producer with physical height padding (`fc.rs`'s padded row
///   count, a reduced output extent) strides differently from what the
///   consumer's own geometry would pack;
/// - equal surface strides (the consumer's packed pixel count), which the
///   PPU alone rounds up to four pixels;
/// - equal logical pixel widths, and the consumer asking for no channel
///   padding beyond them (`packed == logical`), since padding surfaces are
///   zero after a repack but hold the producer's padding channels here;
/// - a whole number of 16-byte atoms per pixel, since a partial trailing atom
///   is zeroed by the repack and holds the producer's padding channels here.
///
/// `consumer` is the cube this dispatch would pack its input to; the
/// identity itself is `rocket_core::layout::chain_identity`, and this
/// function adds only what the runtime alone knows -- which recorded write
/// produced the bytes, and whether it published a cube.
///
/// # Safety
///
/// `binding` and every recorded write target must be live `RocketBuffer`s,
/// which every direct binding on this command buffer is -- indirect ones are
/// rejected in `dispatch()` and `apply_ops_until_dispatch`.
fn chainable_cube(
    cb: &mut RocketCommandBuffer,
    binding: &iree_hal_buffer_ref_t,
    consumer: CubeGeometry,
    what: &str,
) -> Option<OutputCube> {
    if !chain_enabled() || binding.buffer.is_null() {
        return None;
    }
    // A consumer that pads its channels, or whose pixel is a partial atom,
    // cannot alias a producer cube however well the producer matches.
    if let Err(refusal) = consumer.can_chain() {
        if chain_debug() {
            eprintln!("rocket: chain declined ({what}): {refusal}");
        }
        return None;
    }
    // The most recent write that overlaps the bytes this dispatch reads.
    // Writes to other regions of the same allocation are neighbouring
    // tensors, not blockers, so they are skipped rather than declined on.
    let want = unsafe {
        dma_range(
            binding.buffer,
            binding.offset,
            consumer.pixel_count * consumer.bytes_per_pixel,
        )
    }?;
    let Some((producer_index, producer)) = cb.ops.iter().enumerate().rev().find(|(_, op)| {
        recorded_write_extent(op)
            .and_then(|(buffer, offset, length)| unsafe { dma_range(buffer, offset, length) })
            .is_some_and(|wrote| wrote.0 < want.1 && want.0 < wrote.1)
    }) else {
        // Nothing on this command buffer wrote these bytes: they came from
        // outside -- an upload, a CPU dispatch, an earlier submission. The
        // commonest reason not to chain, and worth telling apart from a
        // producer whose cube does not qualify.
        if chain_debug() {
            eprintln!("rocket: chain declined ({what}): no producer on this command buffer");
        }
        return None;
    };
    let RecordedOp::Dispatch {
        output_cube: Some(cube),
        ..
    } = producer
    else {
        if chain_debug() {
            eprintln!("rocket: chain declined ({what}): producer offers no cube");
        }
        return None;
    };
    // Overlapping is not enough: the producer must have written exactly the
    // region this dispatch reads, in the geometry it would have packed --
    // the identity `rocket_core::layout::chain_identity` states and
    // `tensor_layout.rs`'s `chain_identity_tests` pin to the bytes.
    let same_bytes = unsafe {
        dma_range(
            cube.dense_buffer,
            cube.dense_offset,
            cube.geometry.pixel_count * cube.geometry.bytes_per_pixel,
        )
    } == Some(want);
    let verdict: Result<(), Option<ChainRefusal>> = if same_bytes {
        chain_identity(&cube.geometry, &consumer).map_err(Some)
    } else {
        Err(None)
    };
    if let Err(refusal) = verdict {
        if chain_debug() {
            let why = refusal.map_or_else(
                || "producer wrote a different byte range".to_string(),
                |refusal| refusal.to_string(),
            );
            eprintln!(
                "rocket: chain declined ({what}): {why}; producer cube {}x{} (surfaces {} apart) at +{} vs consumer {}x{} (surfaces {} apart) at +{}",
                cube.geometry.pixel_count,
                cube.geometry.bytes_per_pixel,
                cube.geometry.surface_pixel_count,
                cube.dense_offset,
                consumer.pixel_count,
                consumer.bytes_per_pixel,
                consumer.surface_pixel_count,
                binding.offset
            );
        }
        return None;
    }
    if chain_debug() {
        eprintln!(
            "rocket: chain taken ({what}): {} pixels x {} bytes, skipping the repack",
            consumer.pixel_count, consumer.bytes_per_pixel
        );
    }
    let cube = *cube;
    // The producer now has one reader that will never touch its dense
    // output; `apply_ops` weighs this against the compiler's count.
    if let RecordedOp::Dispatch {
        chained_readers, ..
    } = &mut cb.ops[producer_index]
    {
        *chained_readers += 1;
    }
    Some(cube)
}

/// Allocates one [`Replica`] per sibling context in `contexts`, each holding
/// this dispatch's four operands on that context's file.
///
/// The input, bias and (unpacked) weights are copied from wherever the home
/// context reads them; packed coefficients come through `weight_cache`,
/// either a hit on the sibling or a copy of the home packing that is
/// published there once `apply_ops` has flushed it.
///
/// # Safety
///
/// Every context's file must stay open for the command buffer's life, and
/// `Binding` sources must be live `RocketBuffer`s retained by it.
// Several of these are already tuple-grouped; a struct wrapper would just
// move the same fields into a constructor.
#[allow(clippy::too_many_arguments)]
unsafe fn build_replicas(
    contexts: &[Arc<crate::pool::NpuContext>],
    input: (ReplicaSource, usize),
    input_geometry: Option<BandGeometry>,
    tile_bands: &[InputBand],
    tile_context: &[usize],
    weights: (
        Option<&Arc<weight_cache::SharedBuffer>>,
        ReplicaSource,
        WeightFanoutSource,
    ),
    bias: (ReplicaSource, usize),
    output_bytes: usize,
) -> Vec<Replica> {
    let alloc = |context: &Arc<crate::pool::NpuContext>, bytes: usize| unsafe {
        let fd = context.file.as_raw_fd();
        RocketOwnedBuffer::new(fd, bytes.max(1), BorrowedFd::borrow_raw(fd))
    };
    contexts
        .iter()
        .enumerate()
        .map(|(replica_index, context)| {
            let (home_packed, weight_source, fanout) = weights;
            let weights = match (home_packed, fanout.key) {
                (Some(home), Some(key)) => {
                    let key = weight_cache::Key {
                        context: context.id,
                        ..key
                    };
                    let hit = fanout
                        .publishable
                        .then(|| weight_cache::lookup(&key, fanout.generation))
                        .flatten();
                    match hit {
                        Some(buffer) => ReplicaWeights::Packed { buffer, copy: None },
                        None => ReplicaWeights::Packed {
                            buffer: weight_cache::SharedBuffer::new(unsafe {
                                let fd = context.file.as_raw_fd();
                                RocketGemBuffer::new(
                                    fd,
                                    fanout.length.max(1),
                                    BorrowedFd::borrow_raw(fd),
                                )
                            }),
                            copy: Some((
                                ReplicaSource::Host(home.host_ptr as *const u8),
                                fanout.length,
                                fanout.publishable.then_some(WeightPublish {
                                    key,
                                    bytes: fanout.length,
                                }),
                            )),
                        },
                    }
                }
                _ => ReplicaWeights::Direct(ReplicaCopy {
                    buffer: alloc(context, fanout.length),
                    source: weight_source,
                    length: fanout.length,
                    geometry: None,
                    bands: Vec::new(),
                }),
            };
            Replica {
                context: Arc::clone(context),
                input: ReplicaCopy {
                    buffer: alloc(context, input.1),
                    source: input.0,
                    length: input.1,
                    geometry: input_geometry,
                    bands: tile_bands
                        .iter()
                        .zip(tile_context)
                        .filter(|(_, owner)| **owner == replica_index + 1)
                        .map(|(band, _)| *band)
                        .collect(),
                },
                weights,
                bias: ReplicaCopy {
                    buffer: alloc(context, bias.1),
                    source: bias.0,
                    length: bias.1,
                    geometry: None,
                    bands: Vec::new(),
                },
                output: alloc(context, output_bytes),
            }
        })
        .collect()
}

/// Tile `t` of a dispatch with `replicas` siblings runs on context index
/// `t % (replicas + 1)`: tiles in order alternate over every context, so a
/// dispatch with more tiles than contexts keeps every core busy and one
/// with fewer uses as many as it has tiles.
fn tile_contexts(tiles: usize, replicas: usize) -> Vec<usize> {
    (0..tiles).map(|tile| tile % (replicas + 1)).collect()
}

/// Copies the replica's bytes into place on the worker, after the home
/// context's own packing has produced them.
unsafe fn apply_replica_copy(fd: RawFd, copy: &ReplicaCopy) -> Result<(), iree_status_t> {
    if let (Some(geometry), false) = (copy.geometry, copy.bands.is_empty()) {
        let source_base = match copy.source {
            ReplicaSource::Host(ptr) => ptr,
            ReplicaSource::Binding { buffer, offset } => {
                let source = unsafe { &*(buffer as *const RocketBuffer) };
                unsafe { source.host_ptr.add(offset) as *const u8 }
            }
        };
        let row_bytes = geometry.width * geometry.block_bytes;
        for band in &copy.bands {
            let row = band.row.min(geometry.height);
            let rows = band.rows.min(geometry.height - row);
            let column = band.column.min(geometry.width);
            let columns = band.columns.min(geometry.width - column);
            for surface in 0..geometry.surfaces {
                let plane = surface * geometry.surface_stride;
                if column == 0 && columns == geometry.width {
                    // Full-width rows are one contiguous run per plane.
                    let offset = plane + row * row_bytes;
                    let bytes = rows * row_bytes;
                    if offset + bytes > copy.length {
                        return Err(status::from_code(
                            crate::bindings::iree_status_code_e_IREE_STATUS_INTERNAL,
                        ));
                    }
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            source_base.add(offset),
                            copy.buffer.host_ptr.add(offset),
                            bytes,
                        )
                    };
                } else {
                    for y in row..row + rows {
                        let offset = plane + (y * geometry.width + column) * geometry.block_bytes;
                        let bytes = columns * geometry.block_bytes;
                        if offset + bytes > copy.length {
                            return Err(status::from_code(
                                crate::bindings::iree_status_code_e_IREE_STATUS_INTERNAL,
                            ));
                        }
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                source_base.add(offset),
                                copy.buffer.host_ptr.add(offset),
                                bytes,
                            )
                        };
                    }
                }
            }
        }
        if unsafe { fini_bo(fd, copy.buffer.handle) }.is_err() {
            return Err(status::from_code(
                crate::bindings::iree_status_code_e_IREE_STATUS_INTERNAL,
            ));
        }
        return Ok(());
    }
    unsafe {
        apply_replica_bytes(
            fd,
            copy.source,
            copy.length,
            copy.buffer.host_ptr,
            copy.buffer.handle,
        )
    }
}

unsafe fn apply_replica_bytes(
    fd: RawFd,
    source: ReplicaSource,
    length: usize,
    destination: *mut u8,
    handle: u32,
) -> Result<(), iree_status_t> {
    let source_ptr = match source {
        ReplicaSource::Host(ptr) => ptr,
        ReplicaSource::Binding { buffer, offset } => {
            let source = unsafe { &*(buffer as *const RocketBuffer) };
            unsafe { source.host_ptr.add(offset) as *const u8 }
        }
    };
    unsafe { std::ptr::copy_nonoverlapping(source_ptr, destination, length) };
    if unsafe { fini_bo(fd, handle) }.is_err() {
        return Err(status::from_code(
            crate::bindings::iree_status_code_e_IREE_STATUS_INTERNAL,
        ));
    }
    Ok(())
}

/// The DMA address and GEM handle a job on `cb`'s context may use for a
/// binding the hardware reads directly: the binding itself on context 0,
/// a [`StagedCopy`] of it into this context's own scratch anywhere else.
///
/// # Safety
///
/// `binding.buffer` must be a live `RocketBuffer` and `cb.fd` a live Rocket
/// DRM file description.
unsafe fn stage_direct(
    cb: &RocketCommandBuffer,
    binding: &iree_hal_buffer_ref_t,
    scratch_buffers: &mut Vec<RocketOwnedBuffer>,
    staged_copies: &mut Vec<StagedCopy>,
) -> (u32, u32) {
    let rocket_buffer = unsafe { &*(binding.buffer as *const RocketBuffer) };
    if cb.context.id == 0 {
        return (
            rocket_buffer.dma_address + binding.offset as u32,
            rocket_buffer.handle,
        );
    }
    let length = binding.length;
    let scratch =
        unsafe { RocketOwnedBuffer::new(cb.fd, length.max(1), BorrowedFd::borrow_raw(cb.fd)) };
    let staged = (scratch.dma_address, scratch.handle);
    staged_copies.push(StagedCopy {
        source: CopySource::Binding {
            buffer: binding.buffer,
            offset: binding.offset,
        },
        length,
        scratch_ptr: scratch.host_ptr,
        scratch_handle: scratch.handle,
        publish: None,
    });
    scratch_buffers.push(scratch);
    staged
}

/// Points a dispatch at its packed coefficients, reusing a cached packing
/// when one is valid for this binding at this geometry.
///
/// On a non-zero context a packing cached on another context cannot be
/// named by this context's job; it is copied into a fresh buffer here
/// (`staged_copies`) instead of packed again, and published under this
/// context's key once the copy has landed.
///
/// # Safety
///
/// `weight_ref.buffer` must be a live `RocketBuffer` and `cb.fd` a live
/// Rocket DRM file description.
unsafe fn stage_weights(
    cb: &RocketCommandBuffer,
    weight_ref: &iree_hal_buffer_ref_t,
    geometry: weight_cache::Geometry,
    staged_copies: &mut Vec<StagedCopy>,
) -> StagedWeights {
    let key = weight_cache::Key {
        buffer: weight_ref.buffer as usize,
        offset: weight_ref.offset as u64,
        length: weight_ref.length as u64,
        geometry,
        context: cb.context.id,
    };
    let generation = unsafe { crate::buffer::generation(weight_ref.buffer) };
    let pending_writer = cb
        .ops
        .iter()
        .filter_map(recorded_write_target)
        .any(|target| target == weight_ref.buffer);
    if pending_writer {
        weight_cache::note_recorded_writer();
    }
    let packing = |scratch_ptr: *mut u8, scratch_handle: u32, verify_against| WeightPacking {
        weight_buffer: weight_ref.buffer,
        weight_offset: weight_ref.offset,
        weight_length: weight_ref.length,
        scratch_ptr,
        scratch_length: geometry.scratch_length,
        scratch_handle,
        filter_height: geometry.filter_height,
        filter_width: geometry.filter_width,
        input_channels: geometry.input_channels,
        output_channels: geometry.output_channels,
        programmed_output_channels: geometry.programmed_output_channels,
        element_size: geometry.element_size,
        depthwise: geometry.depthwise,
        padded_channels: geometry.padded_channels,
        weight_zero_point: geometry.weight_zero_point,
        verify_against,
    };

    let fanout = WeightFanoutSource {
        key: Some(key),
        generation,
        publishable: !pending_writer,
        length: geometry.scratch_length,
    };
    let cached = if pending_writer {
        None
    } else {
        weight_cache::lookup(&key, generation)
    };
    if let Some(cached) = cached {
        // Verify mode re-packs into a throwaway buffer and hands `apply_ops`
        // the cached bytes to check it against; the regcmd still reads the
        // cached buffer, so the comparison covers what the hardware sees.
        let probe = weight_cache::verifying().then(|| unsafe {
            RocketOwnedBuffer::new(
                cb.fd,
                geometry.scratch_length.max(1),
                BorrowedFd::borrow_raw(cb.fd),
            )
        });
        let verify_packing = probe
            .as_ref()
            .map(|probe| packing(probe.host_ptr, probe.handle, Some(cached.host_ptr)));
        return StagedWeights {
            addr: cached.dma_address,
            handle: cached.handle,
            packing: verify_packing,
            scratch: Some(cached),
            publish: None,
            probe,
            fanout,
        };
    }

    let scratch = weight_cache::SharedBuffer::new(unsafe {
        RocketGemBuffer::new(
            cb.fd,
            geometry.scratch_length.max(1),
            BorrowedFd::borrow_raw(cb.fd),
        )
    });
    // Another context already packed these bytes: a copy is cheaper than a
    // pack, and it is published under this context's key when it lands.
    if !pending_writer && let Some(other) = weight_cache::lookup_other_context(&key, generation) {
        staged_copies.push(StagedCopy {
            source: CopySource::Shared {
                source: other,
                weight_buffer: weight_ref.buffer,
            },
            length: geometry.scratch_length,
            scratch_ptr: scratch.host_ptr,
            scratch_handle: scratch.handle,
            publish: Some((
                WeightPublish {
                    key,
                    bytes: geometry.scratch_length,
                },
                Arc::clone(&scratch),
            )),
        });
        return StagedWeights {
            addr: scratch.dma_address,
            handle: scratch.handle,
            packing: None,
            scratch: Some(scratch),
            publish: None,
            probe: None,
            fanout,
        };
    }
    StagedWeights {
        addr: scratch.dma_address,
        handle: scratch.handle,
        packing: Some(packing(scratch.host_ptr, scratch.handle, None)),
        publish: Some(WeightPublish {
            key,
            bytes: geometry.scratch_length,
        }),
        scratch: Some(scratch),
        probe: None,
        fanout,
    }
}

/// Defers logical dense FP16 bias widening and padding until execution.
///
/// FP16 BRDMA consumes widened 32-bit ALU operands and is programmed for the
/// atomically padded output-channel count. A private, zero-padded buffer both
/// performs the FP16-to-FP32 bridge and prevents that physical read width from
/// escaping an exact-sized binding or observing neighboring suballocations.
#[derive(Clone, Copy)]
pub struct BiasPacking {
    pub bias_buffer: *mut iree_hal_buffer_t,
    pub bias_offset: iree_device_size_t,
    pub bias_length: iree_device_size_t,
    pub scratch_ptr: *mut u8,
    pub scratch_length: usize,
    pub scratch_handle: u32,
    pub output_channels: usize,
    pub padded_output_channels: usize,
    pub int8: bool,
    pub input_scale: f32,
    pub weights_scale: f32,
    pub weight_zero_point: i8,
}

/// Bridges the RK3588 DPU's atomic-slot output write-back (16-byte-aligned
/// slots regardless of dtype, `FEATURE_ATOMIC_SIZE=16`) to IREE's densely-
/// packed ABI output buffer -- see `iree-rocket-hal/src/rocket/conv.rs`'s
/// `Shape::output_scratch_bytes` doc comment and the "Conv2d output
/// compaction" investigation this fixes. `dispatch()` points the regcmd at
/// a driver-private scratch buffer instead of the real output buffer;
/// `queue_execute`, after its existing post-dispatch `prep_bo` wait
/// confirms the hardware write is complete, interleaves the hardware channel
/// blocks into `output_buffer` (the real, dense IREE buffer, retained with
/// every other direct dispatch binding so it survives until `queue_execute`
/// runs). Ordinary output uses 16-byte blocks; accumulator output uses the
/// CORE-native block of 32 i32 lanes (128 bytes).
#[derive(Clone)]
pub struct OutputCompaction {
    pub output_buffer: *mut iree_hal_buffer_t,
    pub output_offset: iree_device_size_t,
    pub output_length: iree_device_size_t,
    pub scratch_ptr: *mut u8,
    pub scratch_length: usize,
    pub source_pixel_count: usize,
    pub output_pixel_count: usize,
    pub output_width: usize,
    pub bytes_per_pixel: usize,
    pub source_block_bytes: usize,
    pub source_tiles: Option<Arc<[AccumulatorOutputTile]>>,
    /// Fan-out gather: each replica's output scratch as `(host pointer,
    /// length)`, the context index each tile ran on (`0` = home) and each
    /// tile's output rectangle. All empty for a dispatch that ran at home.
    pub replica_scratch: Vec<(usize, usize)>,
    pub tile_context: Vec<usize>,
    pub tile_rects: Vec<TileRect>,
}

/// A dispatch's NC1HWC2 output cube, offered to whichever later dispatch in
/// this command buffer reads the dense buffer it compacts into.
///
/// The producer writes feature-atomic surfaces into a driver-private scratch
/// BO and [`OutputCompaction`] interleaves them into the dense IREE buffer;
/// the consumer then reads that dense buffer straight back into its own
/// scratch through `pack_nhwc_to_nc1hwc2_padded`. Under the conditions
/// [`chainable_cube`] checks, `pack(compact(cube)) == cube` byte for byte, so
/// the consumer may simply point its regcmd at the producer's scratch and
/// skip the repack entirely -- bit-identical output, one full pass over the
/// tensor saved. This is the aliasing `encodings/cross-op-chaining.md` proved
/// legal and ISSUES.md P2 has carried since: the input feature cube and the
/// fp16-narrowed output cube are the same layout, both `feat_idx` with a
/// 16-byte channel atom.
///
/// The compaction still runs by default: nothing on a command buffer can
/// prove the dense buffer has no other reader -- a later CPU dispatch, a
/// later command buffer, the model's own output. The proof comes from the
/// compiler as the dispatch's trailing push constant, its count of Rocket
/// readers (`Conv2DDef.runtime_dense_readers`, `rocket-mark-dense-readers`),
/// and `apply_ops` skips the dense write when exactly that many consumers
/// chained here and nothing else read the bytes -- see
/// [`compaction_elidable`]. ISSUES.md P2, the compaction half.
#[derive(Clone, Copy)]
pub struct OutputCube {
    /// The dense IREE buffer and offset this cube is compacted into, which
    /// is what a consumer's input binding is matched against.
    pub dense_buffer: *mut iree_hal_buffer_t,
    pub dense_offset: iree_device_size_t,
    /// The scratch BO the hardware wrote, on this command buffer's own
    /// context -- a job may only name BOs created on its own file, and every
    /// dispatch recorded here shares the command buffer's context.
    pub dma_address: u32,
    pub handle: u32,
    pub host_ptr: *mut u8,
    pub length: usize,
    /// The cube's geometry -- `pixel_count` logical pixels, surfaces
    /// `surface_pixel_count * 16` bytes apart, `bytes_per_pixel` a whole
    /// number of atoms wherever a cube is offered at all -- computed through
    /// `rocket_core::layout` so this driver and the compiler cannot arrive at
    /// two geometries for one shape (COMPILER_ROADMAP.md 6.1).
    pub geometry: CubeGeometry,
}

/// One recorded command-buffer operation, in call order -- see module doc
/// comment for why these are recorded rather than applied immediately.
// `Dispatch` dwarfs the other variants, but boxing its dozen-odd
// `Option<...>` fields would touch every construction and field-access
// site across this file for no behavioral change; not worth it here.
#[allow(clippy::large_enum_variant)]
pub enum RecordedOp {
    Fill {
        target: iree_hal_buffer_ref_t,
        /// Patterns are always <= 8 bytes (`pattern_length` further below
        /// bounds it) -- matches `iree_hal_deferred_command_buffer_t`'s
        /// identical fixed-size inline pattern storage.
        pattern: [u8; 8],
        pattern_length: u8,
    },
    Update {
        target: iree_hal_buffer_ref_t,
        /// Copied out of the caller's `source_buffer` at record time --
        /// that pointer is only guaranteed valid for the duration of the
        /// `update_buffer` call itself.
        source: Vec<u8>,
    },
    Copy {
        source: iree_hal_buffer_ref_t,
        target: iree_hal_buffer_ref_t,
    },
    Dispatch {
        regcmd_tasks: Vec<Vec<RegCmd>>,
        /// The DPU execution mode this dispatch programs. `None` denotes a
        /// non-DPU dispatch such as pooling. Kept with the recorded work so
        /// `device::queue_execute` can apply hardware transition rules in
        /// actual submission order.
        dpu_mode: Option<DpuMode>,
        /// Diagnostic only: the precision this dispatch programs, so
        /// `ROCKET_DISPATCH_TIMES` can name what hung rather than just when.
        precision_tag: Option<Precision>,
        /// Every direct binding supplied to the dispatch, retained exactly
        /// once at record time as required by the IREE HAL command-buffer
        /// contract. The command buffer releases them from `destroy()`.
        retained_bindings: Vec<*mut iree_hal_buffer_t>,
        /// Driver-private GEM allocations whose DMA addresses and host
        /// mappings are referenced by the packing/compaction descriptors and
        /// baked into `regcmd_tasks`. Owning them here keeps them alive until
        /// the command buffer has finished executing and closes/unmaps them
        /// on every destruction path.
        scratch_buffers: Vec<RocketOwnedBuffer>,
        /// Copies into this context's scratch that must land before the job
        /// runs -- see [`StagedCopy`]. Empty on context 0.
        staged_copies: Vec<StagedCopy>,
        /// Sibling contexts this dispatch's tiles are spread over, and per
        /// task which one runs it (`0` is this command buffer's own
        /// context, `i + 1` is `replicas[i]`). Empty when every task runs
        /// at home.
        replicas: Vec<Replica>,
        tile_context: Vec<usize>,
        /// GEM handles of every buffer this dispatch reads (bindings other
        /// than the output) -- must be listed in `drm_rocket_job.in_bo_handles`
        /// (device.rs's `queue_execute`) so the kernel driver's implicit
        /// fencing/dependency tracking actually knows the job touches them.
        /// Previously only the regcmd program's own GEM buffer was listed
        /// there, which happened to let SUBMIT/PREP_BO round-trip (proving
        /// the ioctl plumbing itself worked) but never told the kernel
        /// about the real input/weight/bias/output BOs at all -- see any
        /// hand-rolled hardware test in iree-rocket-hal's `tests/`
        /// directory (e.g. `conv_phase1_validation_hw.rs`) for the ioctl
        /// call shape that always did this correctly.
        in_bo_handles: Vec<u32>,
        /// GEM handles of every buffer this dispatch writes.
        out_bo_handles: Vec<u32>,
        /// Set for multi-channel Conv2d dispatches whose dense IREE input
        /// must be packed into NC1HWC2 before the NPU reads it.
        input_packing: Option<InputPacking>,
        /// The second tensor of a two-tensor element-wise op, packed the
        /// same way as `input_packing`. None for every other kind.
        operand_packing: Option<InputPacking>,
        /// Set for regular fp16 Conv2d dispatches whose logical HWCF filter
        /// must be packed into the CNA's blocked coefficient order.
        weight_packing: Option<WeightPacking>,
        /// Set for FP16 Conv2d and FullyConnected dispatches. Logical FP16
        /// values are widened to BRDMA's FP32 operands and the physical tail
        /// is zero-padded to the DPU's programmed output-channel count.
        bias_packing: Option<BiasPacking>,
        /// Set only for `Conv2d` -- see `OutputCompaction`'s own doc comment.
        /// `None` for `Pooling` (unaffected today, flagged as a follow-up
        /// risk -- same DPU write-back stage almost certainly has the same
        /// atomic-slot mismatch, just not fixed here).
        output_compaction: Option<OutputCompaction>,
        /// Human-readable shape of this dispatch, the key `ROCKET_PROFILE`
        /// groups its per-op timings under. Empty unless profiling is on --
        /// building it costs a `format!` per recorded dispatch, which is
        /// exactly the kind of thing that has no business in the inference
        /// loop by default.
        profile_label: String,
        /// The packed coefficients the regcmd reads, shared with
        /// `weight_cache`. Held here so the allocation outlives this command
        /// buffer's execution whether it was reused or built by this
        /// dispatch; `scratch_buffers` cannot own it because the cache may
        /// still be handing it to later command buffers.
        weight_scratch: Option<Arc<weight_cache::SharedBuffer>>,
        /// Set when this dispatch writes its result as a plain 16-byte-atom
        /// NC1HWC2 cube: the offer a later dispatch reading the same dense
        /// buffer may take instead of repacking it. See [`OutputCube`].
        output_cube: Option<OutputCube>,
        /// Set when this dispatch packed its own coefficients: `apply_ops`
        /// publishes them once the packing has actually succeeded, never at
        /// record time, so no other command buffer can reach a buffer that
        /// has not been filled yet.
        weight_publish: Option<WeightPublish>,
        /// The compiler's count of Rocket dispatches that read this
        /// dispatch's result (`Conv2DDef.runtime_dense_readers`), 0 when it
        /// did not count or saw another reader. Non-conv kinds record 0.
        dense_readers: u32,
        /// How many later dispatches on this command buffer took
        /// `output_cube` in place of the dense buffer. Bumped at record time
        /// by [`chainable_cube`].
        chained_readers: u32,
        /// Whether anything recorded after this dispatch read its dense
        /// output bytes without chaining. Set by [`note_dense_read`].
        dense_read_seen: bool,
    },
}

/// DPU state programmed by a dispatch.
///
/// Depthwise and dense convolution use distinct DPU write-back modes. The
/// device queue uses this to quiesce the hardware at the one empirically
/// unsafe transition, after a depthwise completion and before a dense submit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DpuMode {
    Dense,
    Depthwise,
}

/// What `apply_ops` hands back for each recorded `dispatch`, in call order --
/// the regcmd program plus the real BO handles it touches, so
/// `device::queue_execute` can build a correct `drm_rocket_job` instead of
/// submitting with only the regcmd buffer's own handle listed.
pub struct DispatchJob {
    pub regcmd_tasks: &'static [Vec<RegCmd>],
    pub dpu_mode: Option<DpuMode>,
    pub precision_tag: Option<Precision>,
    pub in_bo_handles: &'static [u32],
    pub out_bo_handles: &'static [u32],
    pub output_compaction: Option<OutputCompaction>,
    /// See `RecordedOp::Dispatch::profile_label`. Borrowed from the recorded
    /// op, which outlives the job (the command buffer is retained across
    /// `queue_execute`).
    pub profile_label: &'static str,
    /// One per task in `regcmd_tasks`: the file it is submitted on and the
    /// BOs it names there.
    pub task_targets: Vec<TaskTarget>,
}

/// What every `iree_hal_command_buffer_t*` this driver hands out actually
/// points to. `base` (the real, fully-defined `iree_hal_command_buffer_t`,
/// filled via `iree_hal_command_buffer_initialize`) must be the first
/// field, matching `buffer::RocketBuffer`'s convention.
#[repr(C)]
pub struct RocketCommandBuffer {
    pub base: iree_hal_command_buffer_t,
    /// Every fill/update/copy/dispatch recorded so far, in call order.
    /// `device::queue_execute` replays these via `apply_ops`.
    pub ops: Vec<RecordedOp>,
    /// Backing storage for `base.validation_state`. IREE's generic
    /// command_buffer.c enables validation by default
    /// (`IREE_HAL_COMMAND_BUFFER_VALIDATION_ENABLE=1`) for any mode without
    /// `IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED` and unconditionally derefs
    /// `validation_state` in that case -- passing NULL (as this used to)
    /// segfaults the instant a validated command buffer is created (CTS's
    /// `EventTest.SignalAndReset` hit this first, once `create_event` started
    /// returning success and the test could reach `iree_hal_command_buffer_
    /// create`). A `Vec`'s heap buffer address is stable across the
    /// `RocketCommandBuffer` itself moving/being boxed, so storing it as a
    /// plain field (rather than mimicking the null driver reference's
    /// single-trailing-allocation trick) is fine.
    validation_state: Vec<u8>,
    /// The NPU context this command buffer was placed on at creation
    /// (`device::create_command_buffer`): the DRM file every scratch GEM
    /// buffer below is allocated on, and the worker `queue_execute` sends
    /// it to. Context 0 is a dup of the allocator's file, so IREE buffers
    /// may be handed to the hardware directly there; on any other context
    /// a job may only name BOs created on that context's file, which is
    /// what `stage_direct` and the per-context `weight_cache` key ensure.
    context: Arc<crate::pool::NpuContext>,
    /// `context.file`'s raw fd, for the many `RocketOwnedBuffer::new` and
    /// `fini_bo` calls below. Valid as long as `context` is, which the
    /// `Arc` guarantees.
    fd: RawFd,
    /// The other contexts, in the order a multi-tile dispatch takes them
    /// for its replicas (`device::WorkerPool::siblings`).
    siblings: Vec<Arc<crate::pool::NpuContext>>,
}

impl RocketCommandBuffer {
    /// The sibling contexts a `tiles`-task dispatch spreads over: at most
    /// `tiles - 1`, so no context is given nothing to do.
    fn fanout_contexts(&self, tiles: usize) -> &[Arc<crate::pool::NpuContext>] {
        if !fanout_enabled() || tiles < 2 {
            return &[];
        }
        &self.siblings[..self.siblings.len().min(tiles - 1)]
    }
}

/// The context `command_buffer` was recorded against, for `queue_execute`.
///
/// # Safety
///
/// `command_buffer` must be a valid, non-null pointer to a
/// `RocketCommandBuffer` created by [`create`] and still live.
pub unsafe fn context_id(command_buffer: *mut iree_hal_command_buffer_t) -> usize {
    unsafe { (&*cast(command_buffer)).context.id }
}

unsafe fn cast(command_buffer: *mut iree_hal_command_buffer_t) -> *mut RocketCommandBuffer {
    command_buffer as *mut RocketCommandBuffer
}

/// Retains every direct dispatch binding once, including bindings that do
/// not need a packing bridge. IREE permits the caller to release its own
/// references as soon as `dispatch()` returns, so raw buffer pointers and GEM
/// handles recorded in the command buffer are only valid if the command
/// buffer owns corresponding references.
unsafe fn retain_direct_bindings(refs: &[iree_hal_buffer_ref_t]) -> Vec<*mut iree_hal_buffer_t> {
    refs.iter()
        .map(|binding| {
            unsafe { crate::bindings::iree_hal_buffer_retain(binding.buffer) };
            binding.buffer
        })
        .collect()
}

/// Not part of the vtable -- `device::queue_execute` calls this directly
/// after its wait-semaphore gate, once per recorded dispatch. Applies the
/// recorded ops from `*cursor` **in call order**: every fill/update/copy
/// immediately (host-side, via IREE's generic `iree_hal_buffer_map_*`
/// helpers -- `buffer::map_range`/`unmap_range` already back those
/// correctly), and the first `dispatch` it reaches has its operands packed
/// and is returned as the job to submit, with `*cursor` left just past it.
/// `None` once the ops are exhausted.
///
/// One dispatch at a time is the whole point. This used to walk the entire
/// command buffer and hand back every job at once, which packed every
/// dispatch's input *before any dispatch had run* -- correct only while no
/// dispatch in a command buffer read what an earlier one in the same buffer
/// wrote. IREE puts two dependent Rocket dispatches in one command buffer
/// with an execution barrier between them whenever nothing on the CPU sits
/// between them, and the second then packed the transient before the first
/// had compacted into it: an all-zero input, silently, for any chained
/// pair (ISSUES.md C13; the requantized MobileNetV2 `Cout` 24 and `Cin`
/// 1344 "anomalies"). The caller runs the returned job to completion --
/// submit, wait, compact -- before asking for the next, which is exactly
/// the ordering the barrier IREE recorded between them requires and the
/// only one `execution_barrier` (a no-op here) could ever have meant.
///
/// # Safety
///
/// `command_buffer` must be a valid, non-null pointer to a
/// `RocketCommandBuffer` created by [`create`] and still live, and
/// `*cursor` must be a valid index into its recorded ops (0 on the first
/// call, then whatever this function last wrote back).
pub unsafe fn apply_ops_until_dispatch(
    command_buffer: *mut iree_hal_command_buffer_t,
    cursor: &mut usize,
) -> Result<Option<DispatchJob>, iree_status_t> {
    let cb = unsafe { &*cast(command_buffer) };
    while let Some(op) = cb.ops.get(*cursor) {
        *cursor += 1;
        // Indirect bindings (buffer == NULL, real buffer resolved from
        // binding_table.buffer_slot -- see command_buffer.h's own doc
        // comment on iree_hal_buffer_ref_t) aren't resolved anywhere in
        // this file today. Found the hard way, as a real segfault on real
        // hardware (iree_hal_buffer_map_copy called with a garbage/null
        // buffer pointer) the first time an actual compiled `.vmfb`
        // reached this code -- this project's own hand-driven CTS tests
        // only ever construct direct bindings. Reject cleanly instead of
        // dereferencing a null buffer; real indirect-binding support would
        // need resolving `binding_table` here (queue_execute's caller
        // does have it available, per device.rs, but nothing plumbs it
        // into apply_ops today) -- out of scope for this fix.
        let indirect_ref = match op {
            RecordedOp::Fill { target, .. } => target.buffer.is_null(),
            RecordedOp::Update { target, .. } => target.buffer.is_null(),
            RecordedOp::Copy { source, target } => {
                source.buffer.is_null() || target.buffer.is_null()
            }
            RecordedOp::Dispatch { .. } => false, // rejected earlier, in dispatch() itself.
        };
        if indirect_ref {
            return Err(status::from_code(
                crate::bindings::iree_status_code_e_IREE_STATUS_UNIMPLEMENTED,
            ));
        }

        match op {
            RecordedOp::Fill {
                target,
                pattern,
                pattern_length,
            } => {
                let st = unsafe {
                    crate::bindings::iree_hal_buffer_map_fill(
                        target.buffer,
                        target.offset,
                        target.length,
                        pattern.as_ptr() as *const std::ffi::c_void,
                        *pattern_length as iree_host_size_t,
                    )
                };
                if !st.is_null() {
                    return Err(st);
                }
            }
            RecordedOp::Update { target, source } => {
                let st = unsafe {
                    crate::bindings::iree_hal_buffer_map_write(
                        target.buffer,
                        target.offset,
                        source.as_ptr() as *const std::ffi::c_void,
                        source.len() as iree_device_size_t,
                    )
                };
                if !st.is_null() {
                    return Err(st);
                }
            }
            RecordedOp::Copy { source, target } => {
                let st = unsafe {
                    crate::bindings::iree_hal_buffer_map_copy(
                        source.buffer,
                        source.offset,
                        target.buffer,
                        target.offset,
                        target.length,
                    )
                };
                if !st.is_null() {
                    return Err(st);
                }
            }
            RecordedOp::Dispatch {
                regcmd_tasks,
                dpu_mode,
                precision_tag,
                in_bo_handles,
                out_bo_handles,
                input_packing,
                operand_packing,
                weight_packing,
                bias_packing,
                output_compaction,
                profile_label,
                weight_scratch,
                weight_publish,
                staged_copies,
                replicas,
                tile_context,
                scratch_buffers: _,
                retained_bindings: _,
                // Record-time only: a later dispatch reads it while the
                // command buffer is still being built, never here.
                output_cube: _,
                dense_readers,
                chained_readers,
                dense_read_seen,
            } => {
                if let Some(packing) = input_packing {
                    apply_input_packing(cb.fd, packing, profile_label)?;
                }
                // Bindings the hardware would read directly, and packings
                // cached on another context: byte copies into this
                // context's own scratch. Empty on context 0.
                for copy in staged_copies {
                    let timer = profile::start();
                    let (source_ptr, generation) = match &copy.source {
                        CopySource::Binding { buffer, offset } => {
                            let source = unsafe { &*(*buffer as *const RocketBuffer) };
                            (unsafe { source.host_ptr.add(*offset) as *const u8 }, 0)
                        }
                        CopySource::Shared {
                            source,
                            weight_buffer,
                        } => (source.host_ptr as *const u8, unsafe {
                            crate::buffer::generation(*weight_buffer)
                        }),
                    };
                    unsafe {
                        std::ptr::copy_nonoverlapping(source_ptr, copy.scratch_ptr, copy.length)
                    };
                    if unsafe { fini_bo(cb.fd, copy.scratch_handle) }.is_err() {
                        return Err(status::from_code(
                            crate::bindings::iree_status_code_e_IREE_STATUS_INTERNAL,
                        ));
                    }
                    if let Some((publish, scratch)) = &copy.publish {
                        weight_cache::publish(
                            publish.key,
                            generation,
                            Arc::clone(scratch),
                            publish.bytes,
                        );
                    }
                    profile::stop(timer, profile::Phase::Stage, profile_label, copy.length);
                }
                // The second tensor of a two-tensor element-wise op. None
                // for every other dispatch kind.
                if let Some(packing) = operand_packing {
                    apply_input_packing(cb.fd, packing, profile_label)?;
                }
                if let Some(packing) = weight_packing {
                    let timer = profile::start();
                    // Read before packing, not after: a concurrent write to
                    // the binding then lands on a higher generation than the
                    // entry is published under, so it can never be mistaken
                    // for the bytes this pack actually read.
                    let generation = unsafe { crate::buffer::generation(packing.weight_buffer) };
                    // Depthwise's dense_len has no Cout factor -- one filter
                    // per input channel, not a kernel set per output channel
                    // (packing.output_channels is unused in this mode; see
                    // WeightPacking's doc comment).
                    let dense_len = if packing.depthwise {
                        packing
                            .filter_height
                            .checked_mul(packing.filter_width)
                            .and_then(|value| value.checked_mul(packing.input_channels))
                            .and_then(|value| value.checked_mul(packing.element_size))
                    } else {
                        packing
                            .filter_height
                            .checked_mul(packing.filter_width)
                            .and_then(|value| value.checked_mul(packing.input_channels))
                            .and_then(|value| value.checked_mul(packing.output_channels))
                            .and_then(|value| value.checked_mul(packing.element_size))
                    }
                    .ok_or_else(|| {
                        status::from_code(crate::bindings::iree_status_code_e_IREE_STATUS_INTERNAL)
                    })?;
                    if dense_len as u64 > packing.weight_length as u64 {
                        return Err(status::from_code(
                            crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                        ));
                    }
                    let weights = unsafe { &*(packing.weight_buffer as *const RocketBuffer) };
                    let dense = unsafe {
                        std::slice::from_raw_parts(
                            weights.host_ptr.add(packing.weight_offset),
                            dense_len,
                        )
                    };
                    let scratch = unsafe {
                        std::slice::from_raw_parts_mut(packing.scratch_ptr, packing.scratch_length)
                    };
                    let pack_result = if packing.depthwise {
                        pack_depthwise_to_rocket_weights(
                            dense,
                            packing.filter_height,
                            packing.filter_width,
                            packing.input_channels,
                            packing.padded_channels,
                            packing.element_size,
                            scratch,
                        )
                    } else if let Some(zero_point) = packing.weight_zero_point {
                        if packing.programmed_output_channels > packing.output_channels {
                            if zero_point != 0 {
                                Err(
                                    "programmed Cout padding requires symmetric accumulator weights",
                                )
                            } else {
                                pack_hwcf_to_rocket_weights_padded(
                                    dense,
                                    packing.filter_height,
                                    packing.filter_width,
                                    packing.input_channels,
                                    packing.output_channels,
                                    packing.programmed_output_channels,
                                    packing.element_size,
                                    scratch,
                                )
                            }
                        } else {
                            let zero_points = vec![zero_point; packing.output_channels];
                            pack_hwcf_to_rocket_weights_affine_i8(
                                dense,
                                packing.filter_height,
                                packing.filter_width,
                                packing.input_channels,
                                packing.output_channels,
                                &zero_points,
                                scratch,
                            )
                        }
                    } else {
                        pack_hwcf_to_rocket_weights(
                            dense,
                            packing.filter_height,
                            packing.filter_width,
                            packing.input_channels,
                            packing.output_channels,
                            packing.element_size,
                            scratch,
                        )
                    };
                    if pack_result.is_err() {
                        return Err(status::from_code(
                            crate::bindings::iree_status_code_e_IREE_STATUS_INTERNAL,
                        ));
                    }
                    if unsafe {
                        iree_rocket_hal::rocket::device::fini_bo(cb.fd, packing.scratch_handle)
                    }
                    .is_err()
                    {
                        return Err(status::from_code(
                            crate::bindings::iree_status_code_e_IREE_STATUS_INTERNAL,
                        ));
                    }
                    profile::stop(
                        timer,
                        profile::Phase::PackWeights,
                        profile_label,
                        packing.scratch_length,
                    );
                    // Verify mode: the regcmd reads the cached buffer, so
                    // comparing it against what a fresh pack produces checks
                    // exactly what the hardware will see. A mismatch means a
                    // write the generation counter missed, a key that fails
                    // to separate two different packings, or a stale entry --
                    // all of which would otherwise be silent wrong numbers.
                    if let Some(expected) = packing.verify_against {
                        let packed = unsafe {
                            std::slice::from_raw_parts(packing.scratch_ptr, packing.scratch_length)
                        };
                        let cached =
                            unsafe { std::slice::from_raw_parts(expected, packing.scratch_length) };
                        if let Some(index) = packed.iter().zip(cached).position(|(a, b)| a != b) {
                            eprintln!(
                                "rocket: ROCKET_WEIGHT_CACHE=verify mismatch at byte {index} of                                  {} for `{profile_label}`: cached {:#04x}, freshly packed {:#04x}.                                  The cached coefficients do not match this dispatch's weights;                                  run with ROCKET_WEIGHT_CACHE=0 to confirm, then look for a write                                  to the weight binding that does not reach `buffer::note_write`.",
                                packing.scratch_length, cached[index], packed[index],
                            );
                            return Err(status::from_code(
                                crate::bindings::iree_status_code_e_IREE_STATUS_DATA_LOSS,
                            ));
                        }
                    }
                    // Only now, with the buffer actually filled and flushed,
                    // is it safe for another command buffer to point at it.
                    if let (Some(publish), Some(scratch)) = (weight_publish, weight_scratch) {
                        weight_cache::publish(
                            publish.key,
                            generation,
                            Arc::clone(scratch),
                            publish.bytes,
                        );
                    }
                }
                if let Some(packing) = bias_packing {
                    let timer = profile::start();
                    let dense_len = packing
                        .output_channels
                        .checked_mul(if packing.int8 { 4 } else { 2 })
                        .ok_or_else(|| {
                            status::from_code(
                                crate::bindings::iree_status_code_e_IREE_STATUS_INTERNAL,
                            )
                        })?;
                    if dense_len as u64 > packing.bias_length as u64 {
                        return Err(status::from_code(
                            crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                        ));
                    }
                    let bias = unsafe { &*(packing.bias_buffer as *const RocketBuffer) };
                    let dense = unsafe {
                        std::slice::from_raw_parts(
                            bias.host_ptr.add(packing.bias_offset),
                            dense_len,
                        )
                    };
                    let scratch = unsafe {
                        std::slice::from_raw_parts_mut(packing.scratch_ptr, packing.scratch_length)
                    };
                    if packing.int8 {
                        if iree_rocket_hal::rocket::conv::pack_int8_bias_to_bs(
                            dense,
                            packing.output_channels,
                            packing.padded_output_channels,
                            packing.input_scale,
                            packing.weights_scale,
                            packing.weight_zero_point,
                            scratch,
                        )
                        .is_err()
                        {
                            return Err(status::from_code(
                                crate::bindings::iree_status_code_e_IREE_STATUS_INTERNAL,
                            ));
                        }
                    } else if pack_fp16_bias_to_rocket(
                        dense,
                        packing.output_channels,
                        packing.padded_output_channels,
                        scratch,
                    )
                    .is_err()
                    {
                        return Err(status::from_code(
                            crate::bindings::iree_status_code_e_IREE_STATUS_INTERNAL,
                        ));
                    }
                    if unsafe {
                        iree_rocket_hal::rocket::device::fini_bo(cb.fd, packing.scratch_handle)
                    }
                    .is_err()
                    {
                        return Err(status::from_code(
                            crate::bindings::iree_status_code_e_IREE_STATUS_INTERNAL,
                        ));
                    }
                    profile::stop(
                        timer,
                        profile::Phase::PackBias,
                        profile_label,
                        packing.scratch_length,
                    );
                }
                // Fan-out: the sibling contexts' copies of every operand,
                // taken now that the home packing has produced them. A
                // packed coefficient copy is published under the sibling's
                // key so the next command buffer there hits.
                for replica in replicas {
                    let timer = profile::start();
                    let fd = replica.context.file.as_raw_fd();
                    let mut bytes = replica.input.length + replica.bias.length;
                    unsafe { apply_replica_copy(fd, &replica.input)? };
                    unsafe { apply_replica_copy(fd, &replica.bias)? };
                    match &replica.weights {
                        ReplicaWeights::Direct(copy) => {
                            bytes += copy.length;
                            unsafe { apply_replica_copy(fd, copy)? };
                        }
                        ReplicaWeights::Packed {
                            buffer,
                            copy: Some((source, length, publish)),
                        } => {
                            bytes += length;
                            unsafe {
                                apply_replica_bytes(
                                    fd,
                                    *source,
                                    *length,
                                    buffer.host_ptr,
                                    buffer.handle,
                                )?
                            };
                            if let (Some(publish), Some(packing)) = (publish, weight_packing) {
                                let generation =
                                    unsafe { crate::buffer::generation(packing.weight_buffer) };
                                weight_cache::publish(
                                    publish.key,
                                    generation,
                                    Arc::clone(buffer),
                                    publish.bytes,
                                );
                            } else if let Some(publish) = publish {
                                // The home hit the cache, so the binding's
                                // current generation is the one its entry
                                // matched; a write since would have made the
                                // home miss instead.
                                let generation = unsafe {
                                    crate::buffer::generation(publish.key.buffer as *mut _)
                                };
                                weight_cache::publish(
                                    publish.key,
                                    generation,
                                    Arc::clone(buffer),
                                    publish.bytes,
                                );
                            }
                        }
                        ReplicaWeights::Packed { copy: None, .. } => {}
                    }
                    profile::stop(timer, profile::Phase::Stage, profile_label, bytes);
                }

                // Sync every buffer the NPU is about to read for device access.
                //
                // The packing paths above each `fini_bo` the scratch they
                // just wrote, so anything staged through them was already
                // covered. A buffer handed to the hardware *directly* was
                // not, and IREE's own buffers reach us that way: the CPU
                // writes them (an upload, or a preceding CPU dispatch) and
                // the NPU then DMAs stale memory.
                //
                // Only convolution's dense feature path (`Cin <= 4`, the
                // ARGB modes) consumes an IREE buffer directly today --
                // every other input is staged -- which is why this showed up
                // as "sub-atom Cin is wrong" and looked like a packing or
                // alignment defect. It is neither: hardware-measured, the
                // bases involved are already 16-byte aligned, and staging
                // the same bytes through scratch fixed it only because
                // staging carries this sync along with it.
                //
                // Syncing every input handle rather than just the
                // un-staged ones keeps the rule simple and removes the same
                // hazard for any buffer passed directly in future. A
                // redundant sync on an already-synced scratch BO is a cache
                // operation with no effect on correctness.
                let timer = profile::start();
                for &input_handle in in_bo_handles.iter() {
                    if unsafe { fini_bo(cb.fd, input_handle) }.is_err() {
                        return Err(status::from_code(
                            crate::bindings::iree_status_code_e_IREE_STATUS_INTERNAL,
                        ));
                    }
                }
                profile::stop(timer, profile::Phase::SyncInputs, profile_label, 0);
                let task_targets = (0..regcmd_tasks.len())
                    .map(|task| match tile_context.get(task).copied().unwrap_or(0) {
                        0 => TaskTarget {
                            fd: cb.fd,
                            context: cb.context.id,
                            in_bo_handles: in_bo_handles.clone(),
                            out_bo_handles: out_bo_handles.clone(),
                        },
                        index => {
                            let replica = &replicas[index - 1];
                            TaskTarget {
                                fd: replica.context.file.as_raw_fd(),
                                context: replica.context.id,
                                in_bo_handles: vec![
                                    replica.input.buffer.handle,
                                    replica.weights.handle(),
                                    replica.bias.buffer.handle,
                                ],
                                out_bo_handles: vec![replica.output.handle],
                            }
                        }
                    })
                    .collect();
                // Every consumer of this output has been recorded by now, so
                // this is where the dense write can be judged unnecessary:
                // the compiler said how many Rocket readers there are, and
                // the command buffer saw whether each one chained.
                let elide = lazy_compact_enabled()
                    && output_compaction.is_some()
                    && compaction_elidable(*dense_readers, *chained_readers, *dense_read_seen);
                if lazy_compact_debug() && output_compaction.is_some() {
                    eprintln!(
                        "rocket: compaction {} ({}): {} reader(s) counted, {} chained, dense read {}",
                        if elide { "skipped" } else { "kept" },
                        profile_label,
                        dense_readers,
                        chained_readers,
                        if *dense_read_seen { "seen" } else { "not seen" }
                    );
                }
                if elide {
                    ELIDED_COMPACTIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                return Ok(Some(DispatchJob {
                    regcmd_tasks: regcmd_tasks.as_slice(),
                    dpu_mode: *dpu_mode,
                    precision_tag: *precision_tag,
                    in_bo_handles: in_bo_handles.as_slice(),
                    out_bo_handles: out_bo_handles.as_slice(),
                    output_compaction: if elide {
                        None
                    } else {
                        output_compaction.clone()
                    },
                    profile_label: profile_label.as_str(),
                    task_targets,
                }));
            }
        }
    }
    Ok(None)
}

/// # Safety
///
/// `device_allocator` must be a valid, non-null pointer to a live
/// `iree_hal_allocator_t` that outlives the returned command buffer.
pub unsafe fn create(
    device_allocator: *mut crate::bindings::iree_hal_allocator_t,
    context: Arc<crate::pool::NpuContext>,
    siblings: Vec<Arc<crate::pool::NpuContext>>,
    mode: iree_hal_command_buffer_mode_t,
    command_categories: iree_hal_command_category_t,
    queue_affinity: iree_hal_queue_affinity_t,
    binding_capacity: iree_host_size_t,
) -> *mut iree_hal_command_buffer_t {
    let validation_state_size = unsafe {
        crate::bindings::iree_hal_command_buffer_validation_state_size(mode, binding_capacity)
    };
    let fd = context.file.as_raw_fd();
    let cb = Box::new(RocketCommandBuffer {
        base: unsafe { std::mem::zeroed() }, // filled by iree_hal_command_buffer_initialize below
        ops: Vec::new(),
        validation_state: vec![0u8; validation_state_size],
        context,
        fd,
        siblings,
    });
    let cb_ptr = Box::into_raw(cb);
    unsafe {
        crate::bindings::iree_hal_command_buffer_initialize(
            device_allocator,
            mode,
            command_categories,
            queue_affinity,
            binding_capacity,
            (*cb_ptr).validation_state.as_mut_ptr() as *mut std::ffi::c_void,
            &VTABLE,
            &mut (*cb_ptr).base,
        );
    }
    cb_ptr as *mut iree_hal_command_buffer_t
}

unsafe extern "C" fn destroy(command_buffer: *mut iree_hal_command_buffer_t) {
    unsafe {
        let cb = Box::from_raw(cast(command_buffer));
        // Release exactly what fill_buffer/update_buffer/copy_buffer/
        // dispatch retained at record time -- see those functions' own
        // comments. Dispatch retains every direct binding, including buffers
        // used without a packing bridge, because IREE permits callers to
        // release their references immediately after recording. Dropping `cb`
        // after this loop also drops every driver-private `OwnedBuffer`, which
        // unmaps its VMA and closes its GEM handle.
        for op in &cb.ops {
            match op {
                RecordedOp::Fill { target, .. } => {
                    crate::bindings::iree_hal_buffer_release(target.buffer);
                }
                RecordedOp::Update { target, .. } => {
                    crate::bindings::iree_hal_buffer_release(target.buffer);
                }
                RecordedOp::Copy { source, target } => {
                    crate::bindings::iree_hal_buffer_release(source.buffer);
                    crate::bindings::iree_hal_buffer_release(target.buffer);
                }
                RecordedOp::Dispatch {
                    retained_bindings, ..
                } => {
                    for &buffer in retained_bindings {
                        crate::bindings::iree_hal_buffer_release(buffer);
                    }
                }
            }
        }
        drop(cb);
    }
}

// Real no-ops (not status_stub -- that returns UNIMPLEMENTED, which would
// break every real caller that expects begin/end to just work).
#[allow(unused_variables)]
unsafe extern "C" fn begin(command_buffer: *mut iree_hal_command_buffer_t) -> iree_status_t {
    status::ok()
}
#[allow(unused_variables)]
unsafe extern "C" fn end(command_buffer: *mut iree_hal_command_buffer_t) -> iree_status_t {
    status::ok()
}

status_stub!(begin_debug_group(
    command_buffer: *mut iree_hal_command_buffer_t,
    label: iree_string_view_t,
    label_color: iree_hal_label_color_t,
    location: *const iree_hal_label_location_t,
) -> iree_status_t);

status_stub!(end_debug_group(command_buffer: *mut iree_hal_command_buffer_t) -> iree_status_t);

// Real no-op, not status_stub -- CTS's TransientBufferTest.
// FillThenCopyInSingleCommandBuffer records fill/barrier/copy into one
// command buffer and expects it to succeed. `apply_ops` (device.rs)
// already replays every recorded op strictly in push order, so ordering
// between ops recorded before/after a barrier is already guaranteed by
// this driver's own execution model -- there's nothing left for a real
// barrier to enforce, same reasoning as signal_event/reset_event/
// wait_events below.
#[allow(unused_variables)]
unsafe extern "C" fn execution_barrier(
    command_buffer: *mut iree_hal_command_buffer_t,
    source_stage_mask: iree_hal_execution_stage_t,
    target_stage_mask: iree_hal_execution_stage_t,
    flags: iree_hal_execution_barrier_flags_t,
    memory_barrier_count: iree_host_size_t,
    memory_barriers: *const iree_hal_memory_barrier_t,
    buffer_barrier_count: iree_host_size_t,
    buffer_barriers: *const iree_hal_buffer_barrier_t,
) -> iree_status_t {
    status::ok()
}

// Real no-ops, not status_stub -- see event.rs's module doc comment: this
// driver's only real synchronization granularity is per-command-buffer
// (device::queue_execute's blocking SUBMIT/PREP_BO), so there's no finer
// in-command-buffer schedule for these to actually mark/enforce. CTS's
// EventTest exercises these through a real command buffer and expects
// success (an execution barrier is a correct, conservative treatment of
// wait_events, per iree_hal_null_command_buffer_wait_events's own comment).
#[allow(unused_variables)]
unsafe extern "C" fn signal_event(
    command_buffer: *mut iree_hal_command_buffer_t,
    event: *mut iree_hal_event_t,
    source_stage_mask: iree_hal_execution_stage_t,
) -> iree_status_t {
    status::ok()
}

#[allow(unused_variables)]
unsafe extern "C" fn reset_event(
    command_buffer: *mut iree_hal_command_buffer_t,
    event: *mut iree_hal_event_t,
    source_stage_mask: iree_hal_execution_stage_t,
) -> iree_status_t {
    status::ok()
}

#[allow(unused_variables)]
unsafe extern "C" fn wait_events(
    command_buffer: *mut iree_hal_command_buffer_t,
    event_count: iree_host_size_t,
    events: *mut *const iree_hal_event_t,
    source_stage_mask: iree_hal_execution_stage_t,
    target_stage_mask: iree_hal_execution_stage_t,
    memory_barrier_count: iree_host_size_t,
    memory_barriers: *const iree_hal_memory_barrier_t,
    buffer_barrier_count: iree_host_size_t,
    buffer_barriers: *const iree_hal_buffer_barrier_t,
) -> iree_status_t {
    status::ok()
}

status_stub!(advise_buffer(
    command_buffer: *mut iree_hal_command_buffer_t,
    buffer_ref: iree_hal_buffer_ref_t,
    flags: iree_hal_memory_advise_flags_t,
    arg0: u64,
    arg1: u64,
) -> iree_status_t);

#[allow(unused_variables)]
unsafe extern "C" fn fill_buffer(
    command_buffer: *mut iree_hal_command_buffer_t,
    target_ref: iree_hal_buffer_ref_t,
    pattern: *const std::ffi::c_void,
    pattern_length: iree_host_size_t,
    flags: iree_hal_fill_flags_t,
) -> iree_status_t {
    if pattern_length > 8 {
        return status::from_code(crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT);
    }
    let mut pattern_buf = [0u8; 8];
    unsafe {
        std::ptr::copy_nonoverlapping(
            pattern as *const u8,
            pattern_buf.as_mut_ptr(),
            pattern_length,
        );
    }
    // See RecordedOp's own doc comment: recorded ops hold onto their
    // buffer_ref's raw `buffer` pointer for later use in `apply_ops`,
    // which runs at queue_execute time -- potentially well after this
    // recording call returns and the caller drops its own reference.
    // Without retaining here, that's a real use-after-free (found the
    // hard way, as a real segfault on real hardware).
    unsafe {
        crate::bindings::iree_hal_buffer_retain(target_ref.buffer);
    }
    let cb = unsafe { &mut *cast(command_buffer) };
    cb.ops.push(RecordedOp::Fill {
        target: target_ref,
        pattern: pattern_buf,
        pattern_length: pattern_length as u8,
    });
    status::ok()
}

#[allow(unused_variables)]
unsafe extern "C" fn update_buffer(
    command_buffer: *mut iree_hal_command_buffer_t,
    source_buffer: *const std::ffi::c_void,
    source_offset: iree_host_size_t,
    target_ref: iree_hal_buffer_ref_t,
    flags: iree_hal_update_flags_t,
) -> iree_status_t {
    let len = target_ref.length;
    let mut source = vec![0u8; len];
    unsafe {
        let src = (source_buffer as *const u8).add(source_offset);
        std::ptr::copy_nonoverlapping(src, source.as_mut_ptr(), len);
    }
    // See fill_buffer's comment on why this retain is needed.
    unsafe {
        crate::bindings::iree_hal_buffer_retain(target_ref.buffer);
    }
    let cb = unsafe { &mut *cast(command_buffer) };
    cb.ops.push(RecordedOp::Update {
        target: target_ref,
        source,
    });
    status::ok()
}

#[allow(unused_variables)]
unsafe extern "C" fn copy_buffer(
    command_buffer: *mut iree_hal_command_buffer_t,
    source_ref: iree_hal_buffer_ref_t,
    target_ref: iree_hal_buffer_ref_t,
    flags: iree_hal_copy_flags_t,
) -> iree_status_t {
    // See fill_buffer's comment on why these retains are needed.
    unsafe {
        crate::bindings::iree_hal_buffer_retain(source_ref.buffer);
        crate::bindings::iree_hal_buffer_retain(target_ref.buffer);
    }
    let cb = unsafe { &mut *cast(command_buffer) };
    unsafe { note_dense_read(cb, &source_ref) };
    cb.ops.push(RecordedOp::Copy {
        source: source_ref,
        target: target_ref,
    });
    status::ok()
}

status_stub!(collective(
    command_buffer: *mut iree_hal_command_buffer_t,
    channel: *mut iree_hal_channel_t,
    op: iree_hal_collective_op_t,
    param: u32,
    send_ref: iree_hal_buffer_ref_t,
    recv_ref: iree_hal_buffer_ref_t,
    element_count: iree_device_size_t,
) -> iree_status_t);

#[allow(unused_variables)]
/// Times `dispatch_impl` as `ROCKET_PROFILE`'s record phase.
///
/// Everything a dispatch costs before the hardware ever sees it -- planning
/// the convolution, allocating scratch, emitting the regcmd -- happens here,
/// at record time, not at submit time, and none of it shows up in a
/// per-job timer. The label is read back off the op that was just recorded
/// rather than threaded out of `dispatch_impl`, which keeps that function's
/// forty-odd early returns untouched.
unsafe extern "C" fn dispatch(
    command_buffer: *mut iree_hal_command_buffer_t,
    executable: *mut iree_hal_executable_t,
    function: iree_hal_executable_function_t,
    config: iree_hal_dispatch_config_t,
    constants: iree_const_byte_span_t,
    bindings: iree_hal_buffer_ref_list_t,
    flags: iree_hal_dispatch_flags_t,
) -> iree_status_t {
    let timer = profile::start();
    let status = unsafe {
        dispatch_impl(
            command_buffer,
            executable,
            function,
            config,
            constants,
            bindings,
            flags,
        )
    };
    if timer.is_some() {
        let cb = unsafe { &*cast(command_buffer) };
        let label = match cb.ops.last() {
            Some(RecordedOp::Dispatch { profile_label, .. }) => profile_label.as_str(),
            _ => profile::NO_OP,
        };
        profile::stop(timer, profile::Phase::Record, label, 0);
    }
    status
}

unsafe extern "C" fn dispatch_impl(
    command_buffer: *mut iree_hal_command_buffer_t,
    executable: *mut iree_hal_executable_t,
    _function: iree_hal_executable_function_t,
    _config: iree_hal_dispatch_config_t,
    constants: iree_const_byte_span_t,
    bindings: iree_hal_buffer_ref_list_t,
    _flags: iree_hal_dispatch_flags_t,
) -> iree_status_t {
    let cb = unsafe { &mut *cast(command_buffer) };
    let shape = unsafe { &*crate::executable::shape(executable) };
    let constants = if constants.data_length == 0 {
        &[]
    } else {
        if constants.data.is_null() {
            return status::from_code(
                crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
            );
        }
        unsafe { std::slice::from_raw_parts(constants.data, constants.data_length) }
    };
    let refs = unsafe { std::slice::from_raw_parts(bindings.values, bindings.count) };

    // Indirect bindings (iree_hal_buffer_ref_t.buffer == NULL, real buffer
    // resolved from a binding_table.buffer_slot at queue_execute time, not
    // known yet at this record-time call -- see command_buffer.h's own doc
    // comment on iree_hal_buffer_ref_t) are the default IREE compiles
    // programs with (`--iree-hal-indirect-command-buffers` defaults to
    // true). This driver's whole design builds the real regcmd (with
    // concrete DMA addresses) immediately here, at record time, which is
    // fundamentally incompatible with a binding whose concrete buffer isn't
    // known yet -- found the hard way, as a real segfault on real hardware,
    // the first time an actual compiled `.vmfb` (as opposed to this
    // project's own hand-driven CTS tests, which only ever construct direct
    // bindings via iree_hal_make_buffer_ref) reached this function. Rather
    // than dereference a null `RocketBuffer*` (undefined behavior), reject
    // indirect bindings with a clear, real error -- true support would need
    // deferring regcmd construction to queue_execute time, a larger
    // redesign out of scope for this fix.
    if let Some(r) = refs.iter().find(|r| r.buffer.is_null()) {
        let _ = r;
        return status::from_code(crate::bindings::iree_status_code_e_IREE_STATUS_UNIMPLEMENTED);
    }

    // `r.offset` is the byte offset of this binding WITHIN its underlying
    // buffer -- always 0 in every hand-driven CTS test so far (each of
    // those constructs its own dedicated buffer per binding via
    // iree_hal_make_buffer_ref(buf, 0, size)), but a REAL compiled IREE
    // program routinely sub-allocates multiple tensor arguments out of one
    // combined transient buffer at nonzero offsets (confirmed on real
    // hardware: input/weights bindings shared one buffer, weights at
    // offset=64). Forgetting to add it here silently pointed the regcmd's
    // weight-read register at byte 0 of that shared buffer (the INPUT
    // tensor's own data) instead of the real weight value 64 bytes in --
    // found via a hardware diagnostic dump, not by inspection.
    // (Every direct use of a binding now goes through `stage_direct`, which
    // computes that address itself.)

    // Binding convention: per-ukernel-kind, since each kind's regcmd
    // builder needs a different set of buffers -- see module doc comment
    // for why this is now a frozen cross-repo ABI contract, not just a
    // placeholder.
    match shape {
        UkernelShape::Conv2d(executable) => {
            let (resolved_shape, kernels) = match executable.resolve_shape(constants) {
                Ok(resolved) => resolved,
                Err(_) => {
                    return status::from_code(
                        crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                    );
                }
            };
            let shape = &resolved_shape;
            let programmed_shape = match shape.parity_padded_shape(kernels) {
                Ok(programmed_shape) => programmed_shape,
                Err(_) => {
                    return status::from_code(
                        crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                    );
                }
            };
            // With a residual epilogue the bindings are input, weights, bias,
            // residual, output; otherwise input, weights, bias, output.
            let output_index = if executable.epilogue_add { 4 } else { 3 };
            if bindings.count < output_index + 1 {
                return status::from_code(
                    crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                );
            }
            if executable.epilogue_add
                && (shape.precision != Precision::Fp16 || shape.out_channels % 16 != 0)
            {
                // The EW task reads the conv's cube as an fp16 feature cube of
                // whole 16-byte atoms; nothing else is validated.
                return status::from_code(
                    crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                );
            }
            // The input's cube geometry, through the layout contract the
            // compiler reads (COMPILER_ROADMAP.md 6.1). Computed for the
            // dense ARGB layouts too: the byte widths are the same, only
            // the packing below is skipped for them.
            let Ok(input_geometry) = cube_geometry(
                CubeKind::Conv,
                shape.precision.element_bytes(),
                shape.width,
                shape.height,
                shape.in_channels,
            ) else {
                return status::from_code(
                    crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                );
            };
            let pixel_count = input_geometry.pixel_count;
            let input_bytes_per_pixel = input_geometry.bytes_per_pixel;
            let packed_input_bytes_per_pixel = input_geometry.packed_bytes_per_pixel;
            if !matches!(
                input_geometry.dense_bytes(),
                Ok(value) if value as u64 <= refs[0].length as u64
            ) {
                return status::from_code(
                    crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                );
            }
            let mut scratch_buffers = Vec::with_capacity(4);
            let mut staged_copies = Vec::new();

            // CNA's surface-layout path consumes 16-byte feature-atomic
            // NC1HWC2 surfaces. Shapes with 1..=4 channels use the hardware's
            // dense ARGB modes and must remain dense; packing Cin 2..=4 into
            // 16-byte slots makes those modes read padding as later pixels.
            // A preceding dispatch on this command buffer may already have
            // this input in the cube layout the CNA wants, in which case the
            // repack below is a full pass over the tensor that reproduces
            // bytes the hardware already wrote. Read them in place instead.
            // ISSUES.md P2 step 2; see `chainable_cube` for when that is
            // byte-identical.
            let chained_input = if shape.layout() == FeatureLayout::Surfaces {
                chainable_cube(cb, &refs[0], input_geometry, "conv input")
            } else {
                None
            };
            // Whatever is not read through a cube is read from the dense
            // buffer, and its producer must keep writing it.
            if chained_input.is_none() {
                unsafe { note_dense_read(cb, &refs[0]) };
            }
            unsafe {
                note_dense_read(cb, &refs[1]);
                note_dense_read(cb, &refs[2]);
            }
            let (input_addr, input_handle, input_packing) = if let Some(cube) = chained_input {
                (cube.dma_address, cube.handle, None)
            } else if shape.layout() == FeatureLayout::Surfaces {
                let scratch_bytes =
                    match nc1hwc2_storage_size(pixel_count, packed_input_bytes_per_pixel) {
                        Ok(value) => value,
                        Err(_) => {
                            return status::from_code(
                                crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                            );
                        }
                    };
                let scratch = unsafe {
                    RocketOwnedBuffer::new(
                        cb.fd,
                        scratch_bytes.max(1),
                        BorrowedFd::borrow_raw(cb.fd),
                    )
                };
                let packed = (
                    scratch.dma_address,
                    scratch.handle,
                    Some(InputPacking {
                        input_buffer: refs[0].buffer,
                        input_offset: refs[0].offset,
                        input_length: refs[0].length,
                        scratch_ptr: scratch.host_ptr,
                        scratch_length: scratch_bytes,
                        scratch_handle: scratch.handle,
                        source_pixel_count: pixel_count,
                        packed_pixel_count: pixel_count,
                        bytes_per_pixel: input_bytes_per_pixel,
                        packed_bytes_per_pixel: packed_input_bytes_per_pixel,
                        padding_byte: 0,
                        layout: InputPackingLayout::Nc1hwc2,
                    }),
                );
                scratch_buffers.push(scratch);
                packed
            } else {
                let (addr, handle) =
                    unsafe { stage_direct(cb, &refs[0], &mut scratch_buffers, &mut staged_copies) };
                (addr, handle, None)
            };
            // IREE's conv ABI supplies a logical HWCF filter. Regular fp16
            // convolution consumes a blocked coefficient stream instead:
            // output-block, input-group, X, Y, output-lane, input-lane.
            // This is independently deferred for the same reason as input
            // packing: an earlier recorded operation may populate weights.
            let element_size = shape.precision.element_bytes() as usize;
            let staged_weights = if !shape.depthwise {
                if !matches!(
                    kernels[0]
                    .checked_mul(kernels[1])
                    .and_then(|value| value.checked_mul(shape.in_channels as usize))
                    .and_then(|value| value.checked_mul(shape.out_channels as usize))
                    .and_then(|value| value.checked_mul(element_size)),
                    Some(value) if value as u64 <= refs[1].length as u64
                ) {
                    return status::from_code(
                        crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                    );
                }
                let scratch_bytes = match rocket_weight_storage_size(
                    kernels[0],
                    kernels[1],
                    shape.in_channels as usize,
                    programmed_shape.out_channels as usize,
                    element_size,
                ) {
                    Ok(value) => value,
                    Err(_) => {
                        return status::from_code(
                            crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                        );
                    }
                };
                unsafe {
                    stage_weights(
                        cb,
                        &refs[1],
                        weight_cache::Geometry {
                            filter_height: kernels[0],
                            filter_width: kernels[1],
                            input_channels: shape.in_channels as usize,
                            output_channels: shape.out_channels as usize,
                            programmed_output_channels: programmed_shape.out_channels as usize,
                            element_size,
                            depthwise: false,
                            padded_channels: 0,
                            weight_zero_point: shape
                                .precision
                                .quantization()
                                .map(|q| q.weight_zero_point as i8),
                            scratch_length: scratch_bytes,
                        },
                        &mut staged_copies,
                    )
                }
            } else if shape.depthwise {
                // One filter per input channel -- no Cout factor, unlike
                // the dense branch above. The compiler-emitted dispatch
                // (transform.0.mlir's depthwise matcher) supplies the
                // logical [Cin][kh][kw] filter, which must be packed into
                // the tap-major, grouped-CNA order for both FP16 and int8.
                // Int8 used to fall through to the un-packed buffer here,
                // producing hardware-only corruption while FP16 passed.
                if !matches!(
                    kernels[0]
                    .checked_mul(kernels[1])
                    .and_then(|value| value.checked_mul(shape.in_channels as usize))
                    .and_then(|value| value.checked_mul(element_size)),
                    Some(value) if value as u64 <= refs[1].length as u64
                ) {
                    return status::from_code(
                        crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                    );
                }
                let scratch_bytes = shape.weight_bytes(kernels) as usize;
                unsafe {
                    stage_weights(
                        cb,
                        &refs[1],
                        weight_cache::Geometry {
                            filter_height: kernels[0],
                            filter_width: kernels[1],
                            input_channels: shape.in_channels as usize,
                            output_channels: shape.out_channels as usize,
                            programmed_output_channels: shape.out_channels as usize,
                            element_size,
                            depthwise: true,
                            padded_channels: shape.depthwise_padded_channels() as usize,
                            weight_zero_point: shape
                                .precision
                                .quantization()
                                .map(|q| q.weight_zero_point as i8),
                            scratch_length: scratch_bytes,
                        },
                        &mut staged_copies,
                    )
                }
            } else {
                let (addr, handle) =
                    unsafe { stage_direct(cb, &refs[1], &mut scratch_buffers, &mut staged_copies) };
                StagedWeights::direct(addr, handle, refs[1].length as usize)
            };
            let StagedWeights {
                addr: weights_addr,
                handle: weights_handle,
                packing: weight_packing,
                scratch: weight_scratch,
                publish: weight_publish,
                probe: weight_probe,
                fanout: weight_fanout,
            } = staged_weights;
            if let Some(probe) = weight_probe {
                scratch_buffers.push(probe);
            }
            let (bias_addr, bias_handle, bias_packing) = if shape.precision == Precision::Fp16 {
                let output_channels = shape.out_channels as usize;
                let padded_output_channels = programmed_shape.padded_out_channels() as usize;
                if !matches!(
                    output_channels.checked_mul(element_size),
                    Some(value) if value as u64 <= refs[2].length as u64
                ) {
                    return status::from_code(
                        crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                    );
                }
                let scratch_bytes = match rocket_fp16_bias_storage_size(padded_output_channels) {
                    Ok(value) => value,
                    Err(_) => {
                        return status::from_code(
                            crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                        );
                    }
                };
                let scratch = unsafe {
                    RocketOwnedBuffer::new(
                        cb.fd,
                        scratch_bytes.max(1),
                        BorrowedFd::borrow_raw(cb.fd),
                    )
                };
                let packed = (
                    scratch.dma_address,
                    scratch.handle,
                    Some(BiasPacking {
                        bias_buffer: refs[2].buffer,
                        bias_offset: refs[2].offset,
                        bias_length: refs[2].length,
                        scratch_ptr: scratch.host_ptr,
                        scratch_length: scratch_bytes,
                        scratch_handle: scratch.handle,
                        output_channels,
                        padded_output_channels,
                        int8: false,
                        input_scale: 1.0,
                        weights_scale: 1.0,
                        weight_zero_point: 0,
                    }),
                );
                scratch_buffers.push(scratch);
                packed
            } else if let Precision::Int8(q) | Precision::Int8Accumulator(q) = shape.precision {
                let output_channels = shape.out_channels as usize;
                let padded_output_channels = programmed_shape.padded_out_channels() as usize;
                if output_channels
                    .checked_mul(4)
                    .is_none_or(|value| value as u64 > refs[2].length as u64)
                {
                    return status::from_code(
                        crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                    );
                }
                let scratch_bytes = programmed_shape.bs_buffer_bytes();
                let scratch = unsafe {
                    RocketOwnedBuffer::new(
                        cb.fd,
                        scratch_bytes.max(1),
                        BorrowedFd::borrow_raw(cb.fd),
                    )
                };
                let packed = (
                    scratch.dma_address,
                    scratch.handle,
                    Some(BiasPacking {
                        bias_buffer: refs[2].buffer,
                        bias_offset: refs[2].offset,
                        bias_length: refs[2].length,
                        scratch_ptr: scratch.host_ptr,
                        scratch_length: scratch_bytes,
                        scratch_handle: scratch.handle,
                        output_channels,
                        padded_output_channels,
                        int8: true,
                        input_scale: q.input_scale,
                        weights_scale: q.weights_scale,
                        weight_zero_point: q.weight_zero_point as i8,
                    }),
                );
                scratch_buffers.push(scratch);
                packed
            } else {
                let (addr, handle) =
                    unsafe { stage_direct(cb, &refs[2], &mut scratch_buffers, &mut staged_copies) };
                (addr, handle, None)
            };
            // DPU output write-back (16-byte slots normally, 128-byte native
            // accumulator blocks for bypassed int32) doesn't match IREE's
            // densely-packed ABI output buffer -- see `OutputCompaction`'s
            // own doc comment.
            // Bridge it with a driver-private scratch buffer: the regcmd
            // writes there instead of the real output buffer, and
            // `queue_execute` compacts the real values into the real
            // buffer after the hardware write completes.
            let scratch_bytes = programmed_shape.output_scratch_bytes(kernels).max(1);
            let scratch = unsafe {
                RocketOwnedBuffer::new(cb.fd, scratch_bytes, BorrowedFd::borrow_raw(cb.fd))
            };
            let bufs = Buffers {
                input: input_addr,
                weights: weights_addr,
                bias: bias_addr,
                output: scratch.dma_address,
            };
            // catch_unwind backstop -- see module doc comment for exactly
            // why. `resolve_shape` already trial-plans this exact shape via
            // `validate_conv_shape` (which shares this same `ConvPlan::new`
            // call), so a panic here indicates a genuine internal
            // inconsistency rather than an ordinary user error. The builder
            // only returns fresh local vectors, so a panic mid-build leaves
            // no shared state half-mutated.
            // Fan-out sources: what a sibling context copies for each operand.
            let input_source = match (&chained_input, &input_packing) {
                // A chained input is already a cube on this context's file;
                // a sibling copies it exactly as it copies a fresh packing.
                (Some(cube), _) => (ReplicaSource::Host(cube.host_ptr as *const u8), cube.length),
                (None, Some(packing)) => (
                    ReplicaSource::Host(packing.scratch_ptr as *const u8),
                    packing.scratch_length,
                ),
                (None, None) => (
                    ReplicaSource::Binding {
                        buffer: refs[0].buffer,
                        offset: refs[0].offset as usize,
                    },
                    refs[0].length as usize,
                ),
            };
            let bias_source = match &bias_packing {
                Some(packing) => (
                    ReplicaSource::Host(packing.scratch_ptr as *const u8),
                    packing.scratch_length,
                ),
                None => (
                    ReplicaSource::Binding {
                        buffer: refs[2].buffer,
                        offset: refs[2].offset as usize,
                    },
                    refs[2].length as usize,
                ),
            };
            let weight_binding_source = ReplicaSource::Binding {
                buffer: refs[1].buffer,
                offset: refs[1].offset as usize,
            };
            let input_geometry = Some(match (&chained_input, &input_packing) {
                // The producer's cube may carry padding surfaces past the
                // logical channels; a band copy only needs the ones this
                // dispatch reads, at the producer's own surface stride.
                (Some(cube), _) => BandGeometry {
                    width: shape.width as usize,
                    height: shape.height as usize,
                    surfaces: cube.geometry.bytes_per_pixel / 16,
                    surface_stride: cube.geometry.surface_pixel_count * 16,
                    block_bytes: 16,
                },
                (None, Some(packing)) if matches!(packing.layout, InputPackingLayout::Nc1hwc2) => {
                    BandGeometry {
                        width: shape.width as usize,
                        height: shape.height as usize,
                        surfaces: packing.packed_bytes_per_pixel / 16,
                        surface_stride: packing.packed_pixel_count * 16,
                        block_bytes: 16,
                    }
                }
                (None, Some(packing)) => BandGeometry {
                    width: shape.width as usize,
                    height: shape.height as usize,
                    surfaces: 1,
                    surface_stride: 0,
                    block_bytes: packing.packed_bytes_per_pixel,
                },
                (None, None) => BandGeometry {
                    width: shape.width as usize,
                    height: shape.height as usize,
                    surfaces: 1,
                    surface_stride: 0,
                    block_bytes: input_bytes_per_pixel,
                },
            });
            let planned = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let plan = ConvPlan::new(programmed_shape, kernels);
                let tiles = plan.tiles().len();
                // A residual epilogue's EW task must run after every tile
                // and reads the whole cube, so its dispatch stays on one
                // in-order queue: no fan-out.
                let contexts = if executable.epilogue_add {
                    &[][..]
                } else {
                    cb.fanout_contexts(tiles)
                };
                let tile_context = tile_contexts(tiles, contexts.len());
                let tile_bands: Vec<InputBand> = plan
                    .tiles()
                    .iter()
                    .map(|tile| InputBand {
                        row: tile.rows.in_first as usize,
                        rows: tile.rows.in_rows as usize,
                        column: tile.columns.in_first as usize,
                        columns: tile.columns.in_cols as usize,
                    })
                    .collect();
                let replicas = unsafe {
                    build_replicas(
                        contexts,
                        input_source,
                        input_geometry,
                        &tile_bands,
                        &tile_context,
                        (
                            weight_scratch.as_ref(),
                            weight_binding_source,
                            weight_fanout,
                        ),
                        bias_source,
                        scratch_bytes,
                    )
                };
                let buffers_for = |tile: usize| match tile_context[tile] {
                    0 => bufs,
                    index => replicas[index - 1].buffers(),
                };
                let (programs, source_tiles) = if programmed_shape.precision.writes_accumulators() {
                    let staged = plan.staged_accumulator_programs();
                    assert_eq!(staged.scratch_bytes, scratch_bytes);
                    let programs: Vec<Vec<RegCmd>> = staged
                        .programs
                        .into_iter()
                        .zip(&staged.tiles)
                        .enumerate()
                        .map(|(tile, (mut program, layout))| {
                            relocate_staged_accumulator(&mut program, buffers_for(tile), layout);
                            program
                        })
                        .collect();
                    (
                        programs,
                        Some(Arc::<[AccumulatorOutputTile]>::from(staged.tiles)),
                    )
                } else {
                    let programs: Vec<Vec<RegCmd>> = plan
                        .programs()
                        .into_iter()
                        .enumerate()
                        .map(|(tile, mut program)| {
                            relocate(&mut program, buffers_for(tile));
                            program
                        })
                        .collect();
                    (programs, None)
                };
                let tile_rects = if replicas.is_empty() {
                    Vec::new()
                } else {
                    plan.tiles()
                        .iter()
                        .enumerate()
                        .map(|(index, tile)| TileRect {
                            index,
                            row: tile.rows.out_first as usize,
                            rows: tile.rows.out_rows as usize,
                            column: tile.columns.out_first as usize,
                            columns: tile.columns.out_cols as usize,
                        })
                        .collect()
                };
                (programs, source_tiles, replicas, tile_context, tile_rects)
            })) {
                Ok(planned) => planned,
                Err(_) => {
                    return status::from_code(
                        crate::bindings::iree_status_code_e_IREE_STATUS_INTERNAL,
                    );
                }
            };
            let (mut regcmd_tasks, source_tiles, replicas, tile_context, tile_rects) = planned;
            let replica_scratch: Vec<(usize, usize)> = replicas
                .iter()
                .map(|replica| (replica.output.host_ptr as usize, replica.output.size))
                .collect();
            let tile_context = if replicas.is_empty() {
                Vec::new()
            } else {
                tile_context
            };
            // The output's geometry through the same contract, at the
            // *output* element width -- an fp32-result rung is a 4-lane cube
            // here and an 8-lane one on the input side.
            let Ok(output_geometry) = cube_geometry(
                CubeKind::Conv,
                shape.precision.output_element_bytes(),
                shape.output_width(kernels),
                shape.output_height(kernels),
                shape.out_channels,
            ) else {
                return status::from_code(
                    crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                );
            };
            let output_pixel_count = output_geometry.pixel_count;
            let output_bytes_per_pixel = output_geometry.bytes_per_pixel;
            let tile_rects_empty = tile_rects.is_empty();
            let source_tiles_none = source_tiles.is_none();
            let mut output_compaction = Some(OutputCompaction {
                output_buffer: refs[output_index].buffer,
                output_offset: refs[output_index].offset,
                output_length: refs[output_index].length,
                scratch_ptr: scratch.host_ptr,
                scratch_length: scratch_bytes,
                source_pixel_count: output_pixel_count,
                output_pixel_count,
                output_width: shape.output_width(kernels) as usize,
                bytes_per_pixel: output_bytes_per_pixel,
                source_block_bytes: programmed_shape.output_atom_bytes() as usize,
                source_tiles,
                replica_scratch,
                tile_context: tile_context.clone(),
                tile_rects,
            });
            let output_handle = scratch.handle;
            let conv_output_addr = scratch.dma_address;
            // The cube a later dispatch may read in place. A fanned-out
            // dispatch has no single one -- each context wrote its own tiles
            // into its own copy, and only the compaction gathers them -- and
            // an accumulator dispatch's 128-byte blocks are not the CNA's
            // input layout at all. Both fall back to the repack.
            let plain_atoms = programmed_shape.output_atom_bytes() as usize == 16;
            // `Shape::output_cube_geometry` is the compiler's view of the same
            // offer; the two must agree on whether there is a cube at all.
            debug_assert_eq!(shape.output_cube_geometry(kernels).is_some(), plain_atoms);
            let mut output_cube = (chain_enabled()
                && plain_atoms
                && tile_rects_empty
                && source_tiles_none
                && output_geometry.is_whole_atom())
            .then_some(OutputCube {
                dense_buffer: refs[output_index].buffer,
                dense_offset: refs[output_index].offset,
                dma_address: scratch.dma_address,
                handle: scratch.handle,
                host_ptr: scratch.host_ptr,
                length: scratch_bytes,
                geometry: output_geometry,
            });
            scratch_buffers.push(scratch);
            // The residual epilogue: pack the fourth binding as a feature cube
            // of the output geometry, then one EW task after the tiles adds
            // it to the conv's cube and writes a second scratch, which is
            // what gets compacted. The conv's own cube is never touched by
            // the host. Validated on hardware by `conv_residual_add_hw`.
            let mut in_bo_handles = vec![input_handle, weights_handle, bias_handle];
            let mut out_bo_handles = vec![output_handle];
            let mut operand_packing = None;
            if executable.epilogue_add {
                let Some(cube) = ElementwiseCube::new(
                    shape.output_width(kernels),
                    shape.output_height(kernels),
                    shape.out_channels,
                    2,
                ) else {
                    return status::from_code(
                        crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                    );
                };
                if cube.scratch_bytes != scratch_bytes {
                    // The EW task's cube must be the conv's cube, byte for byte.
                    return status::from_code(
                        crate::bindings::iree_status_code_e_IREE_STATUS_INTERNAL,
                    );
                }
                // The skip is a block input, so on a residual network it is
                // another dispatch's output as often as the feature input
                // is -- and it is the wide tensor. Chain it the same way.
                let chained_skip =
                    chainable_cube(cb, &refs[3], cube.geometry, "conv residual skip");
                if chained_skip.is_none() {
                    unsafe { note_dense_read(cb, &refs[3]) };
                }
                let residual = match chained_skip {
                    Some(skip) => Some((None, skip.dma_address, skip.handle)),
                    None => {
                        pack_elementwise_input(cb, &refs[3], &cube).map(|(scratch, packing)| {
                            let addr = scratch.dma_address;
                            let handle = scratch.handle;
                            (Some((scratch, packing)), addr, handle)
                        })
                    }
                };
                let (
                    Some((residual_staged, residual_addr, residual_handle)),
                    Some((sum_scratch, sum_compaction)),
                ) = (
                    residual,
                    compact_elementwise_output(cb, &refs[output_index], &cube),
                )
                else {
                    return status::from_code(
                        crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                    );
                };
                let add = EwAddShape {
                    width: shape.output_width(kernels),
                    height: shape.output_height(kernels),
                    channels: shape.out_channels,
                    precision: EwPrecision::Fp16,
                    op: EwBinaryOp::Add,
                    output_zero_point: 0,
                    w_cvt_offset: 0,
                    w_scale_ratio: 1.0,
                    output_scale_ratio: 1.0,
                };
                regcmd_tasks.push(build_add_regcmd_with_relu(
                    &add,
                    &EwAddBuffers {
                        intermediate_addr: conv_output_addr,
                        w_addr: residual_addr,
                        output_addr: sum_scratch.dma_address,
                    },
                    executable.epilogue_activation == Activation::Relu,
                ));
                in_bo_handles.push(residual_handle);
                out_bo_handles.push(sum_scratch.handle);
                if let Some((residual_scratch, residual_packing)) = residual_staged {
                    operand_packing = Some(residual_packing);
                    scratch_buffers.push(residual_scratch);
                }
                // The EW task's sum, not the convolution's own cube, is what
                // this dispatch publishes: it is what the compaction reads
                // and so what a later dispatch would otherwise repack.
                output_cube = output_cube.map(|_| OutputCube {
                    dense_buffer: refs[output_index].buffer,
                    dense_offset: refs[output_index].offset,
                    dma_address: sum_scratch.dma_address,
                    handle: sum_scratch.handle,
                    host_ptr: sum_scratch.host_ptr,
                    length: cube.scratch_bytes,
                    geometry: cube.geometry,
                });
                output_compaction = Some(sum_compaction);
                scratch_buffers.push(sum_scratch);
            }
            let retained_bindings = unsafe { retain_direct_bindings(refs) };
            let profile_label = profile::label(|| {
                format!(
                    "conv {} {}x{}x{}->{} k{}x{} s{}{}",
                    profile::precision_name(shape.precision),
                    shape.height,
                    shape.width,
                    shape.in_channels,
                    shape.out_channels,
                    kernels[0],
                    kernels[1],
                    shape.stride,
                    if shape.depthwise { " dw" } else { "" },
                )
            });
            cb.ops.push(RecordedOp::Dispatch {
                regcmd_tasks,
                dpu_mode: Some(if shape.depthwise {
                    DpuMode::Depthwise
                } else {
                    DpuMode::Dense
                }),
                precision_tag: Some(shape.precision),
                retained_bindings,
                scratch_buffers,
                staged_copies,
                replicas,
                tile_context,
                in_bo_handles,
                out_bo_handles,
                input_packing,
                operand_packing,
                weight_packing,
                bias_packing,
                output_compaction,
                profile_label,
                weight_scratch,
                output_cube,
                weight_publish,
                dense_readers: executable.dense_readers(constants),
                chained_readers: 0,
                dense_read_seen: false,
            });
        }
        UkernelShape::Matmul(executable) => {
            let resolved = match executable.resolve_shape(constants) {
                Ok(resolved) => resolved,
                Err(_) => {
                    return status::from_code(
                        crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                    );
                }
            };
            let shape = &resolved;
            if bindings.count < 4 {
                return status::from_code(
                    crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                );
            }

            let m = shape.m as usize;
            let k = shape.k as usize;
            let n = shape.n as usize;
            let element_size = shape.precision.element_bytes() as usize;
            let input_zero_point = shape
                .precision
                .quantization()
                .map_or(0, |quantization| quantization.input_zero_point);
            // fc.rs's real vendor-confirmed lowering has physical height
            // exactly one -- no `FC_PHYSICAL_HEIGHT` padding, so the
            // packed pixel count is just the logical row count `m`.
            let physical_pixel_count = m;
            // Both operands' cube geometries through the layout contract
            // (COMPILER_ROADMAP.md 6.1): width `m`, height 1, at `k` and `n`
            // channels.
            let (Ok(input_geometry), Ok(output_geometry)) = (
                cube_geometry(CubeKind::Matmul, element_size as u32, shape.m, 1, shape.k),
                cube_geometry(CubeKind::Matmul, element_size as u32, shape.m, 1, shape.n),
            ) else {
                return status::from_code(
                    crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                );
            };
            let input_bytes_per_pixel = input_geometry.bytes_per_pixel;
            let output_bytes_per_pixel = output_geometry.bytes_per_pixel;
            let input_len = m.checked_mul(input_bytes_per_pixel);
            let weights_len = k
                .checked_mul(n)
                .and_then(|value| value.checked_mul(element_size));
            let bias_len = n.checked_mul(element_size);
            let output_len = m.checked_mul(output_bytes_per_pixel);
            if !matches!(input_len, Some(value) if value as u64 <= refs[0].length as u64)
                || !matches!(weights_len, Some(value) if value as u64 <= refs[1].length as u64)
                || !matches!(bias_len, Some(value) if value as u64 <= refs[2].length as u64)
                || !matches!(output_len, Some(value) if value as u64 <= refs[3].length as u64)
                || !(0..=u8::MAX as i32).contains(&input_zero_point)
            {
                return status::from_code(
                    crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                );
            }

            // FC is lowered by fc::Plan to a height-one 1x1 convolution
            // (see fc.rs's module doc comment) -- the public input already
            // is exactly `m` physical rows, so this only needs the same
            // NC1HWC2 channel blocking used by convolution, no row padding.
            let packed_input_bytes_per_pixel = input_geometry.packed_bytes_per_pixel;
            let (input_scratch_bytes, input_layout) = if k > 1 {
                match nc1hwc2_storage_size(physical_pixel_count, packed_input_bytes_per_pixel) {
                    Ok(value) => (value, InputPackingLayout::Nc1hwc2),
                    Err(_) => {
                        return status::from_code(
                            crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                        );
                    }
                }
            } else {
                match physical_pixel_count.checked_mul(input_bytes_per_pixel) {
                    Some(value) => (value, InputPackingLayout::Dense),
                    None => {
                        return status::from_code(
                            crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                        );
                    }
                }
            };
            if input_scratch_bytes > u32::MAX as usize {
                return status::from_code(
                    crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                );
            }
            // The [M,K] operand is a width-M, height-1 feature cube, so a
            // producer's cube of M pixels at K channels is exactly what the
            // repack would build (`chainable_cube`). The [K,N] operand is a
            // coefficient stream and never chains.
            let chained_input = if matches!(input_layout, InputPackingLayout::Nc1hwc2) {
                chainable_cube(cb, &refs[0], input_geometry, "matmul input")
            } else {
                None
            };
            if chained_input.is_none() {
                unsafe { note_dense_read(cb, &refs[0]) };
            }
            let input_scratch = if chained_input.is_some() {
                None
            } else {
                Some(unsafe {
                    RocketOwnedBuffer::new(
                        cb.fd,
                        input_scratch_bytes.max(1),
                        BorrowedFd::borrow_raw(cb.fd),
                    )
                })
            };
            let input_packing = input_scratch.as_ref().map(|input_scratch| InputPacking {
                input_buffer: refs[0].buffer,
                input_offset: refs[0].offset,
                input_length: refs[0].length,
                scratch_ptr: input_scratch.host_ptr,
                scratch_length: input_scratch_bytes,
                scratch_handle: input_scratch.handle,
                source_pixel_count: m,
                packed_pixel_count: physical_pixel_count,
                bytes_per_pixel: input_bytes_per_pixel,
                packed_bytes_per_pixel: packed_input_bytes_per_pixel,
                padding_byte: input_zero_point as u8,
                layout: input_layout,
            });
            let (input_addr, input_handle) = match (&chained_input, &input_scratch) {
                (Some(cube), _) => (cube.dma_address, cube.handle),
                (None, Some(scratch)) => (scratch.dma_address, scratch.handle),
                (None, None) => unreachable!("matmul input is either chained or packed"),
            };

            // FC weights arrive as a logical row-major [K,N] matrix, which
            // is exactly a 1x1 HWCF filter and therefore always needs the
            // CNA coefficient transform (for both int8 and fp16).
            let weight_scratch_bytes = match rocket_weight_storage_size(1, 1, k, n, element_size) {
                Ok(value) => value,
                Err(_) => {
                    return status::from_code(
                        crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                    );
                }
            };
            if weight_scratch_bytes > u32::MAX as usize {
                return status::from_code(
                    crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                );
            }
            let mut staged_copies = Vec::new();
            let mut staged_scratch = Vec::new();
            let StagedWeights {
                addr: weights_addr,
                handle: weights_handle,
                packing: weight_packing,
                scratch: weight_scratch,
                publish: weight_publish,
                probe: weight_probe,
                fanout: weight_fanout,
            } = unsafe {
                stage_weights(
                    cb,
                    &refs[1],
                    weight_cache::Geometry {
                        filter_height: 1,
                        filter_width: 1,
                        input_channels: k,
                        output_channels: n,
                        programmed_output_channels: n,
                        element_size,
                        depthwise: false,
                        padded_channels: 0,
                        weight_zero_point: None,
                        scratch_length: weight_scratch_bytes,
                    },
                    &mut staged_copies,
                )
            };

            let (bias_addr, bias_handle, bias_packing, bias_scratch) = if shape.precision
                == Precision::Fp16
            {
                let padded_output_channels = shape.as_conv_shape().padded_out_channels() as usize;
                let bias_scratch_bytes = match rocket_fp16_bias_storage_size(padded_output_channels)
                {
                    Ok(value) => value,
                    Err(_) => {
                        return status::from_code(
                            crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                        );
                    }
                };
                let scratch = unsafe {
                    RocketOwnedBuffer::new(
                        cb.fd,
                        bias_scratch_bytes.max(1),
                        BorrowedFd::borrow_raw(cb.fd),
                    )
                };
                let packing = BiasPacking {
                    bias_buffer: refs[2].buffer,
                    bias_offset: refs[2].offset,
                    bias_length: refs[2].length,
                    scratch_ptr: scratch.host_ptr,
                    scratch_length: bias_scratch_bytes,
                    scratch_handle: scratch.handle,
                    output_channels: n,
                    padded_output_channels,
                    int8: false,
                    input_scale: 1.0,
                    weights_scale: 1.0,
                    weight_zero_point: 0,
                };
                (
                    scratch.dma_address,
                    scratch.handle,
                    Some(packing),
                    Some(scratch),
                )
            } else {
                let (addr, handle) =
                    unsafe { stage_direct(cb, &refs[2], &mut staged_scratch, &mut staged_copies) };
                (addr, handle, None, None)
            };

            let output_scratch_bytes =
                match nc1hwc2_storage_size(physical_pixel_count, output_bytes_per_pixel) {
                    Ok(value) => value,
                    Err(_) => {
                        return status::from_code(
                            crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                        );
                    }
                };
            if output_scratch_bytes > u32::MAX as usize {
                return status::from_code(
                    crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                );
            }
            let output_scratch = unsafe {
                RocketOwnedBuffer::new(
                    cb.fd,
                    output_scratch_bytes.max(1),
                    BorrowedFd::borrow_raw(cb.fd),
                )
            };
            let bufs = Buffers {
                input: input_addr,
                weights: weights_addr,
                bias: bias_addr,
                output: output_scratch.dma_address,
            };
            let input_source = match (&chained_input, &input_packing) {
                // A chained input is already a cube on this context's file;
                // a sibling copies it exactly as it copies a fresh packing.
                (Some(cube), _) => (ReplicaSource::Host(cube.host_ptr as *const u8), cube.length),
                (None, Some(packing)) => (
                    ReplicaSource::Host(packing.scratch_ptr as *const u8),
                    packing.scratch_length,
                ),
                (None, None) => (
                    ReplicaSource::Binding {
                        buffer: refs[0].buffer,
                        offset: refs[0].offset as usize,
                    },
                    refs[0].length as usize,
                ),
            };
            let bias_source = match &bias_packing {
                Some(packing) => (
                    ReplicaSource::Host(packing.scratch_ptr as *const u8),
                    packing.scratch_length,
                ),
                None => (
                    ReplicaSource::Binding {
                        buffer: refs[2].buffer,
                        offset: refs[2].offset as usize,
                    },
                    refs[2].length as usize,
                ),
            };
            let weight_binding_source = ReplicaSource::Binding {
                buffer: refs[1].buffer,
                offset: refs[1].offset as usize,
            };
            let input_geometry = match (&chained_input, &input_packing) {
                (Some(cube), _) => Some(BandGeometry {
                    width: m,
                    height: 1,
                    surfaces: cube.geometry.bytes_per_pixel / 16,
                    surface_stride: cube.geometry.surface_pixel_count * 16,
                    block_bytes: 16,
                }),
                (None, Some(packing)) => Some(match packing.layout {
                    InputPackingLayout::Nc1hwc2 => BandGeometry {
                        width: m,
                        height: 1,
                        surfaces: packing.packed_bytes_per_pixel / 16,
                        surface_stride: packing.packed_pixel_count * 16,
                        block_bytes: 16,
                    },
                    InputPackingLayout::Dense => BandGeometry {
                        width: m,
                        height: 1,
                        surfaces: 1,
                        surface_stride: 0,
                        block_bytes: packing.packed_bytes_per_pixel,
                    },
                }),
                (None, None) => None,
            };
            let planned = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let plan = fc::Plan::new(*shape);
                let tiles = plan.conv_plan().tiles().len();
                let contexts = cb.fanout_contexts(tiles);
                let tile_context = tile_contexts(tiles, contexts.len());
                let tile_bands: Vec<InputBand> = plan
                    .conv_plan()
                    .tiles()
                    .iter()
                    .map(|tile| InputBand {
                        row: tile.rows.in_first as usize,
                        rows: tile.rows.in_rows as usize,
                        column: tile.columns.in_first as usize,
                        columns: tile.columns.in_cols as usize,
                    })
                    .collect();
                let replicas = unsafe {
                    build_replicas(
                        contexts,
                        input_source,
                        input_geometry,
                        &tile_bands,
                        &tile_context,
                        (
                            weight_scratch.as_ref(),
                            weight_binding_source,
                            weight_fanout,
                        ),
                        bias_source,
                        output_scratch_bytes,
                    )
                };
                let programs: Vec<Vec<RegCmd>> = plan
                    .programs()
                    .into_iter()
                    .enumerate()
                    .map(|(tile, mut program)| {
                        let buffers = match tile_context[tile] {
                            0 => bufs,
                            index => replicas[index - 1].buffers(),
                        };
                        relocate(&mut program, buffers);
                        program
                    })
                    .collect();
                let tile_rects = if replicas.is_empty() {
                    Vec::new()
                } else {
                    plan.conv_plan()
                        .tiles()
                        .iter()
                        .enumerate()
                        .map(|(index, tile)| TileRect {
                            index,
                            row: tile.rows.out_first as usize,
                            rows: tile.rows.out_rows as usize,
                            column: tile.columns.out_first as usize,
                            columns: tile.columns.out_cols as usize,
                        })
                        .collect()
                };
                (programs, replicas, tile_context, tile_rects)
            })) {
                Ok(planned) => planned,
                Err(_) => {
                    return status::from_code(
                        crate::bindings::iree_status_code_e_IREE_STATUS_INTERNAL,
                    );
                }
            };
            let (regcmd_tasks, replicas, tile_context, tile_rects) = planned;
            let replica_scratch: Vec<(usize, usize)> = replicas
                .iter()
                .map(|replica| (replica.output.host_ptr as usize, replica.output.size))
                .collect();
            let tile_context = if replicas.is_empty() {
                Vec::new()
            } else {
                tile_context
            };

            let weight_scratch_handle = weights_handle;
            let output_scratch_handle = output_scratch.handle;
            // The cube a later dispatch may read in place: M pixels at N
            // channels, one 16-byte atom per surface, no row padding. Not
            // offered when the tiles fanned out (each context holds its own
            // rows) or the write-out is the accumulator's 128-byte block.
            let output_block_bytes = shape.as_conv_shape().output_channel_block_bytes() as usize;
            let output_cube = (chain_enabled()
                && tile_rects.is_empty()
                && output_block_bytes == 16
                && output_geometry.is_whole_atom())
            .then_some(OutputCube {
                dense_buffer: refs[3].buffer,
                dense_offset: refs[3].offset,
                dma_address: output_scratch.dma_address,
                handle: output_scratch.handle,
                host_ptr: output_scratch.host_ptr,
                length: output_scratch_bytes,
                geometry: output_geometry,
            });
            let output_compaction = Some(OutputCompaction {
                output_buffer: refs[3].buffer,
                output_offset: refs[3].offset,
                output_length: refs[3].length,
                scratch_ptr: output_scratch.host_ptr,
                scratch_length: output_scratch_bytes,
                // No row padding to discard (see physical_pixel_count's
                // own comment) -- source and output pixel counts are the
                // same real `m`.
                source_pixel_count: m,
                output_pixel_count: m,
                output_width: m,
                bytes_per_pixel: output_bytes_per_pixel,
                source_block_bytes: shape.as_conv_shape().output_channel_block_bytes() as usize,
                source_tiles: None,
                replica_scratch,
                tile_context: tile_context.clone(),
                tile_rects,
            });
            let retained_bindings = unsafe { retain_direct_bindings(refs) };
            let profile_label = profile::label(|| {
                format!(
                    "matmul {} {}x{}x{}",
                    profile::precision_name(shape.precision),
                    m,
                    k,
                    n,
                )
            });
            let mut scratch_buffers = vec![output_scratch];
            scratch_buffers.extend(input_scratch);
            scratch_buffers.extend(staged_scratch);
            if let Some(probe) = weight_probe {
                scratch_buffers.push(probe);
            }
            if let Some(scratch) = bias_scratch {
                scratch_buffers.push(scratch);
            }
            // Weights and bias are read from their dense buffers whatever
            // happened to the input.
            unsafe {
                note_dense_read(cb, &refs[1]);
                note_dense_read(cb, &refs[2]);
            }
            cb.ops.push(RecordedOp::Dispatch {
                regcmd_tasks,
                dpu_mode: Some(DpuMode::Dense),
                precision_tag: None,
                retained_bindings,
                scratch_buffers,
                staged_copies,
                replicas,
                tile_context,
                in_bo_handles: vec![input_handle, weight_scratch_handle, bias_handle],
                out_bo_handles: vec![output_scratch_handle],
                input_packing,
                operand_packing: None,
                weight_packing,
                bias_packing,
                output_compaction,
                profile_label,
                weight_scratch,
                output_cube,
                dense_readers: executable.dense_readers(constants),
                chained_readers: 0,
                dense_read_seen: false,
                weight_publish,
            });
        }
        UkernelShape::Pooling(executable) => {
            let shape = match executable.resolve_shape(constants) {
                Ok(shape) => shape,
                Err(_) => {
                    return status::from_code(
                        crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                    );
                }
            };
            // 0=input, 1=output. Every horizontal tile is one direct
            // PPU/PPU_RDMA task; all tasks belong to this one dispatch/job.
            if bindings.count < 2 {
                return status::from_code(
                    crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                );
            }

            // The PPU reads and writes NC1HWC2 cubes; IREE's bindings are
            // dense NHWC. Both ends are repacked on the host, exactly as the
            // convolution arm does, and for the same reason -- nothing in
            // the compiler produces the hardware layout. A pool that
            // consumed its producer's already-packed output would need
            // neither repack, which is the cross-op chaining ISSUES.md P2
            // describes and this is deliberately not that.
            let element_bytes = shape.precision.element_bytes() as usize;
            // Both cubes through the layout contract (COMPILER_ROADMAP.md
            // 6.1), which carries the PPU's two rules: channels padded to one
            // atom, and surfaces strided by the pixel count rounded up to
            // four -- `build_pooling_tile_task` programs both strides that
            // way (the vendor's 7x5 controls program 36 for an area of 35),
            // so the packed cubes must be strided the same or every surface
            // past the first is read, and written, at the wrong offset.
            let (Ok(input_geometry), Ok(output_geometry)) = (
                cube_geometry(
                    CubeKind::Pooling,
                    shape.precision.element_bytes(),
                    shape.input_width,
                    shape.input_height,
                    shape.input_channels,
                ),
                cube_geometry(
                    CubeKind::Pooling,
                    shape.precision.element_bytes(),
                    shape.output_width,
                    shape.output_height,
                    shape.input_channels,
                ),
            ) else {
                return status::from_code(
                    crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                );
            };
            debug_assert_eq!(
                input_geometry.packed_bytes_per_pixel,
                shape.packed_bytes_per_pixel() as usize
            );
            let logical_bytes_per_pixel = input_geometry.bytes_per_pixel;
            let packed_bytes_per_pixel = input_geometry.packed_bytes_per_pixel;
            let input_pixels = input_geometry.pixel_count;
            let output_pixels = output_geometry.pixel_count;
            let packed_input_pixels = input_geometry.surface_pixel_count;
            let packed_output_pixels = output_geometry.surface_pixel_count;
            let dense_input_bytes = input_pixels.checked_mul(logical_bytes_per_pixel);
            let dense_output_bytes = output_pixels.checked_mul(logical_bytes_per_pixel);
            if !matches!(dense_input_bytes, Some(value) if value as u64 <= refs[0].length as u64)
                || !matches!(dense_output_bytes, Some(value) if value as u64 <= refs[1].length as u64)
                || element_bytes == 0
            {
                return status::from_code(
                    crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                );
            }

            let input_scratch_bytes =
                match nc1hwc2_storage_size(packed_input_pixels, packed_bytes_per_pixel) {
                    Ok(value) => value,
                    Err(_) => {
                        return status::from_code(
                            crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                        );
                    }
                };
            let output_scratch_bytes =
                match nc1hwc2_storage_size(packed_output_pixels, packed_bytes_per_pixel) {
                    Ok(value) => value,
                    Err(_) => {
                        return status::from_code(
                            crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                        );
                    }
                };
            if input_scratch_bytes > u32::MAX as usize || output_scratch_bytes > u32::MAX as usize {
                return status::from_code(
                    crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                );
            }

            // A producer's cube qualifies when its surfaces are already the
            // PPU's stride -- the pixel count rounded up to four, so only an
            // image whose count is a multiple of four -- and its channels
            // are whole atoms (`chainable_cube`).
            let chained_input = chainable_cube(cb, &refs[0], input_geometry, "pool input");
            if chained_input.is_none() {
                unsafe { note_dense_read(cb, &refs[0]) };
            }
            let input_scratch = if chained_input.is_some() {
                None
            } else {
                Some(unsafe {
                    RocketOwnedBuffer::new(
                        cb.fd,
                        input_scratch_bytes.max(1),
                        BorrowedFd::borrow_raw(cb.fd),
                    )
                })
            };
            let output_scratch = unsafe {
                RocketOwnedBuffer::new(
                    cb.fd,
                    output_scratch_bytes.max(1),
                    BorrowedFd::borrow_raw(cb.fd),
                )
            };
            let (input_addr, input_handle) = match (&chained_input, &input_scratch) {
                (Some(cube), _) => (cube.dma_address, cube.handle),
                (None, Some(scratch)) => (scratch.dma_address, scratch.handle),
                (None, None) => unreachable!("pool input is either chained or packed"),
            };
            let input_packing = input_scratch.as_ref().map(|input_scratch| InputPacking {
                input_buffer: refs[0].buffer,
                input_offset: refs[0].offset,
                input_length: refs[0].length,
                scratch_ptr: input_scratch.host_ptr,
                scratch_length: input_scratch_bytes,
                scratch_handle: input_scratch.handle,
                source_pixel_count: input_pixels,
                packed_pixel_count: packed_input_pixels,
                bytes_per_pixel: logical_bytes_per_pixel,
                packed_bytes_per_pixel,
                // The pixels between the image and the four-pixel surface
                // boundary are never addressed by the PPU -- the line stride
                // walks rows of `input_width` -- so their contents cannot
                // reach a result and zero is as good as anything. This is
                // *not* the pooling pad fill, which is a register
                // (`PoolingMethod::pad_fill_value`) rather than buffer
                // contents.
                padding_byte: 0,
                layout: InputPackingLayout::Nc1hwc2,
            });
            let output_compaction = Some(OutputCompaction {
                output_buffer: refs[1].buffer,
                output_offset: refs[1].offset,
                output_length: refs[1].length,
                scratch_ptr: output_scratch.host_ptr,
                scratch_length: output_scratch_bytes,
                // Surfaces are strided by the padded count and only the real
                // pixels are copied back out, which is the same split the
                // convolution arm uses to discard its row padding.
                source_pixel_count: packed_output_pixels,
                output_pixel_count: output_pixels,
                output_width: shape.output_width as usize,
                bytes_per_pixel: logical_bytes_per_pixel,
                // One 16-byte feature atom per pixel per surface, which is
                // what the PPU's cube strides are counted in.
                source_block_bytes: 16,
                source_tiles: None,
                replica_scratch: Vec::new(),
                tile_context: Vec::new(),
                tile_rects: Vec::new(),
            });

            let bufs = PoolingBuffers {
                input_addr,
                output_addr: output_scratch.dma_address,
            };
            let regcmd_tasks = PoolingPlan::new(shape).programs_with_buffers(&bufs);
            let in_bo_handles = vec![input_handle];
            let out_bo_handles = vec![output_scratch.handle];
            // The pool's own cube: real pixels at the PPU's four-rounded
            // surface stride. Offered only when its channels are whole atoms
            // with no padding, since a consumer would read the padding lanes.
            let output_cube =
                (chain_enabled() && output_geometry.is_exact()).then_some(OutputCube {
                    dense_buffer: refs[1].buffer,
                    dense_offset: refs[1].offset,
                    dma_address: output_scratch.dma_address,
                    handle: output_scratch.handle,
                    host_ptr: output_scratch.host_ptr,
                    length: output_scratch_bytes,
                    geometry: output_geometry,
                });
            let retained_bindings = unsafe { retain_direct_bindings(refs) };
            let profile_label = profile::label(|| {
                format!(
                    "pool {:?} {}x{}x{} k{}x{} s{}x{}",
                    shape.method,
                    shape.input_height,
                    shape.input_width,
                    shape.input_channels,
                    shape.kernel_height,
                    shape.kernel_width,
                    shape.stride_y,
                    shape.stride_x,
                )
            });
            let mut scratch_buffers = vec![output_scratch];
            scratch_buffers.extend(input_scratch);
            cb.ops.push(RecordedOp::Dispatch {
                regcmd_tasks,
                dpu_mode: None,
                precision_tag: None,
                retained_bindings,
                scratch_buffers,
                staged_copies: Vec::new(),
                replicas: Vec::new(),
                tile_context: Vec::new(),
                in_bo_handles,
                out_bo_handles,
                input_packing,
                operand_packing: None,
                weight_packing: None,
                bias_packing: None,
                output_compaction,
                profile_label,
                weight_scratch: None,
                output_cube,
                dense_readers: executable.dense_readers(constants),
                chained_readers: 0,
                dense_read_seen: false,
                weight_publish: None,
            });
        }
        UkernelShape::ElementwiseUnary(executable) => {
            let shape = match executable.resolve_shape(constants) {
                Ok(shape) => shape,
                Err(_) => {
                    return status::from_code(
                        crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                    );
                }
            };
            // 0=input, 1=output. fp16 is all `EwUnaryShape` supports.
            let Some(cube) = ElementwiseCube::new(shape.width, shape.height, shape.channels, 2)
            else {
                return invalid_argument();
            };
            if bindings.count < 2 {
                return invalid_argument();
            }
            let Some(input) = elementwise_operand(cb, &refs[0], &cube, "ew input") else {
                return invalid_argument();
            };
            let Some((output_scratch, output_compaction)) =
                compact_elementwise_output(cb, &refs[1], &cube)
            else {
                return invalid_argument();
            };
            let output_cube = elementwise_output_cube(&refs[1], &output_scratch, &cube);
            let regcmd_tasks = vec![build_unary_regcmd(
                &shape,
                &EwUnaryBuffers {
                    input_addr: input.addr,
                    output_addr: output_scratch.dma_address,
                },
            )];
            let profile_label = profile::label(|| {
                format!(
                    "ew {:?} {}x{}x{}",
                    shape.algo, shape.height, shape.width, shape.channels
                )
            });
            // These kinds record no output cube and take none, so every
            // operand is a dense read.
            cb.ops.push(RecordedOp::Dispatch {
                regcmd_tasks,
                // Neither field describes an element-wise task. `dpu_mode`
                // exists for the depthwise-to-dense conv quiescence
                // workaround, and `conv::Precision`'s int8 arm carries a
                // convolution's `Quantization`. Same reasoning as the
                // pooling arm: not a conv, so it neither triggers the
                // workaround nor claims a state it does not have. The
                // profile label carries the op and its geometry instead.
                dpu_mode: None,
                precision_tag: None,
                retained_bindings: unsafe { retain_direct_bindings(refs) },
                in_bo_handles: vec![input.handle],
                out_bo_handles: vec![output_scratch.handle],
                scratch_buffers: input.scratch.into_iter().chain([output_scratch]).collect(),
                staged_copies: Vec::new(),
                replicas: Vec::new(),
                tile_context: Vec::new(),
                input_packing: input.packing,
                operand_packing: None,
                weight_packing: None,
                bias_packing: None,
                output_compaction: Some(output_compaction),
                profile_label,
                weight_scratch: None,
                output_cube,
                dense_readers: executable.dense_readers(constants),
                chained_readers: 0,
                dense_read_seen: false,
                weight_publish: None,
            });
        }
        UkernelShape::ElementwiseBinary(executable) => {
            let shape = match executable.resolve_shape(constants) {
                Ok(shape) => shape,
                Err(_) => {
                    return status::from_code(
                        crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                    );
                }
            };
            // 0=primary operand, 1=second tensor, 2=output. Unlike every
            // other kind here, two bindings are packed on the way in:
            // DPU_RDMA fetches the first through its main feed and ERDMA
            // fetches the second, and neither is a layout the compiler
            // produces. Both cubes have the identical geometry -- this op
            // does not broadcast.
            let Some(cube) = ElementwiseCube::new(shape.width, shape.height, shape.channels, 2)
            else {
                return invalid_argument();
            };
            if bindings.count < 3 {
                return invalid_argument();
            }
            let Some(input) = elementwise_operand(cb, &refs[0], &cube, "ew input") else {
                return invalid_argument();
            };
            let Some(operand) = elementwise_operand(cb, &refs[1], &cube, "ew operand") else {
                return invalid_argument();
            };
            let Some((output_scratch, output_compaction)) =
                compact_elementwise_output(cb, &refs[2], &cube)
            else {
                return invalid_argument();
            };
            let output_cube = elementwise_output_cube(&refs[2], &output_scratch, &cube);
            let regcmd_tasks = vec![build_add_regcmd(
                &shape,
                &EwAddBuffers {
                    intermediate_addr: input.addr,
                    w_addr: operand.addr,
                    output_addr: output_scratch.dma_address,
                },
            )];
            let profile_label = profile::label(|| {
                format!(
                    "ew {:?} {}x{}x{}",
                    shape.op, shape.height, shape.width, shape.channels
                )
            });
            // These kinds record no output cube and take none, so every
            // operand is a dense read.
            cb.ops.push(RecordedOp::Dispatch {
                regcmd_tasks,
                // See the unary arm.
                dpu_mode: None,
                precision_tag: None,
                retained_bindings: unsafe { retain_direct_bindings(refs) },
                in_bo_handles: vec![input.handle, operand.handle],
                out_bo_handles: vec![output_scratch.handle],
                scratch_buffers: input
                    .scratch
                    .into_iter()
                    .chain(operand.scratch)
                    .chain([output_scratch])
                    .collect(),
                staged_copies: Vec::new(),
                replicas: Vec::new(),
                tile_context: Vec::new(),
                input_packing: input.packing,
                operand_packing: operand.packing,
                weight_packing: None,
                bias_packing: None,
                output_compaction: Some(output_compaction),
                profile_label,
                weight_scratch: None,
                output_cube,
                dense_readers: executable.dense_readers(constants),
                chained_readers: 0,
                dense_read_seen: false,
                weight_publish: None,
            });
        }
        UkernelShape::ElementwiseLut(executable) => {
            let shape = match executable.resolve_shape(constants) {
                Ok(shape) => shape,
                Err(_) => {
                    return status::from_code(
                        crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
                    );
                }
            };
            // 0=input, 1=output. int8 is all `LutShape` supports.
            let Some(cube) = ElementwiseCube::new(shape.width, shape.height, shape.channels, 1)
            else {
                return invalid_argument();
            };
            if bindings.count < 2 {
                return invalid_argument();
            }
            let Some(input) = elementwise_operand(cb, &refs[0], &cube, "ew input") else {
                return invalid_argument();
            };
            let Some((output_scratch, output_compaction)) =
                compact_elementwise_output(cb, &refs[1], &cube)
            else {
                return invalid_argument();
            };
            let output_cube = elementwise_output_cube(&refs[1], &output_scratch, &cube);
            let regcmd_tasks = vec![build_lut_regcmd(
                &shape,
                &LutBuffers {
                    input_addr: input.addr,
                    output_addr: output_scratch.dma_address,
                },
                executable.function.table(),
            )];
            let profile_label = profile::label(|| {
                format!(
                    "lut {:?} {}x{}x{}",
                    executable.function, shape.height, shape.width, shape.channels
                )
            });
            // These kinds record no output cube and take none, so every
            // operand is a dense read.
            cb.ops.push(RecordedOp::Dispatch {
                regcmd_tasks,
                // See the unary arm.
                dpu_mode: None,
                precision_tag: None,
                retained_bindings: unsafe { retain_direct_bindings(refs) },
                in_bo_handles: vec![input.handle],
                out_bo_handles: vec![output_scratch.handle],
                scratch_buffers: input.scratch.into_iter().chain([output_scratch]).collect(),
                staged_copies: Vec::new(),
                replicas: Vec::new(),
                tile_context: Vec::new(),
                input_packing: input.packing,
                operand_packing: None,
                weight_packing: None,
                bias_packing: None,
                output_compaction: Some(output_compaction),
                profile_label,
                weight_scratch: None,
                output_cube,
                dense_readers: executable.dense_readers(constants),
                chained_readers: 0,
                dense_read_seen: false,
                weight_publish: None,
            });
        }
    }
    status::ok()
}

/// The early return every element-wise binding check shares.
fn invalid_argument() -> iree_status_t {
    status::from_code(crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT)
}

/// The cube arithmetic both element-wise kinds share.
///
/// `packed_channels` is the builders' own
/// `channels.max(16).next_multiple_of(16)`, which is a channel count rounded
/// to 16 **regardless of precision** -- not `PoolingShape::programmed_channels`,
/// which rounds to the precision's channels-per-atom. The two differ exactly
/// where it matters: at fp16 with 24 channels, pooling would program three
/// surfaces and these builders program four.
/// `ew_unary_multi_surface_hw.rs` is the hardware statement of that.
#[derive(Clone, Copy)]
struct ElementwiseCube {
    /// Through the layout contract (COMPILER_ROADMAP.md 6.1); the fields
    /// below are its members, kept spelled out for the packing code.
    geometry: CubeGeometry,
    pixels: usize,
    logical_bytes_per_pixel: usize,
    packed_bytes_per_pixel: usize,
    scratch_bytes: usize,
    width: usize,
}

impl ElementwiseCube {
    fn new(width: u32, height: u32, channels: u32, element_bytes: usize) -> Option<Self> {
        let geometry = cube_geometry(
            CubeKind::Elementwise,
            element_bytes as u32,
            width,
            height,
            channels,
        )
        .ok()?;
        let scratch_bytes = geometry.storage_bytes().ok()?;
        if scratch_bytes > u32::MAX as usize {
            return None;
        }
        Some(Self {
            geometry,
            pixels: geometry.pixel_count,
            logical_bytes_per_pixel: geometry.bytes_per_pixel,
            packed_bytes_per_pixel: geometry.packed_bytes_per_pixel,
            scratch_bytes,
            width: width as usize,
        })
    }

    /// Bytes the caller's own dense NHWC binding must hold for this cube.
    fn dense_bytes(&self) -> Option<usize> {
        self.pixels.checked_mul(self.logical_bytes_per_pixel)
    }
}

/// Allocates a scratch cube for one dense NHWC input binding and describes
/// the repack that fills it.
fn pack_elementwise_input(
    cb: &mut RocketCommandBuffer,
    binding: &iree_hal_buffer_ref_t,
    cube: &ElementwiseCube,
) -> Option<(RocketOwnedBuffer, InputPacking)> {
    if cube.dense_bytes()? as u64 > binding.length as u64 {
        return None;
    }
    let scratch = unsafe {
        RocketOwnedBuffer::new(
            cb.fd,
            cube.scratch_bytes.max(1),
            BorrowedFd::borrow_raw(cb.fd),
        )
    };
    let packing = InputPacking {
        input_buffer: binding.buffer,
        input_offset: binding.offset,
        input_length: binding.length,
        scratch_ptr: scratch.host_ptr,
        scratch_length: cube.scratch_bytes,
        scratch_handle: scratch.handle,
        source_pixel_count: cube.pixels,
        // No four-pixel surface rounding, unlike pooling: the EW and LUT
        // builders program `DPU_DST_SURF_STRIDE`/`DPU_SURFACE_ADD` as exactly
        // `width * height` atoms, where `build_pooling_tile_task` rounds up.
        packed_pixel_count: cube.pixels,
        bytes_per_pixel: cube.logical_bytes_per_pixel,
        packed_bytes_per_pixel: cube.packed_bytes_per_pixel,
        // Channels past the real count are cube padding the op writes and no
        // caller reads back. Zero is the only value that is also a valid
        // input for every opcode and curve here, so a hardware fault that
        // reached them would not be disguised as a plausible result.
        padding_byte: 0,
        layout: InputPackingLayout::Nc1hwc2,
    };
    Some((scratch, packing))
}

/// One element-wise operand, resolved: a producer's cube read in place when
/// one qualifies (`chainable_cube`), else a fresh packing plus the dense
/// read that goes with it.
struct ElementwiseOperand {
    addr: u32,
    handle: u32,
    scratch: Option<RocketOwnedBuffer>,
    packing: Option<InputPacking>,
}

fn elementwise_operand(
    cb: &mut RocketCommandBuffer,
    binding: &iree_hal_buffer_ref_t,
    cube: &ElementwiseCube,
    what: &str,
) -> Option<ElementwiseOperand> {
    // The EW and LUT cubes are exact in pixels and pad channels to 16 like a
    // convolution's input, so a conv, matmul or EW producer's cube is the
    // same bytes whenever the channel count is whole atoms.
    if let Some(producer) = chainable_cube(cb, binding, cube.geometry, what) {
        return Some(ElementwiseOperand {
            addr: producer.dma_address,
            handle: producer.handle,
            scratch: None,
            packing: None,
        });
    }
    unsafe { note_dense_read(cb, binding) };
    let (scratch, packing) = pack_elementwise_input(cb, binding, cube)?;
    Some(ElementwiseOperand {
        addr: scratch.dma_address,
        handle: scratch.handle,
        scratch: Some(scratch),
        packing: Some(packing),
    })
}

/// The op's own output cube, offered to later dispatches when its channels
/// are whole atoms with no padding lanes a consumer would otherwise read.
fn elementwise_output_cube(
    binding: &iree_hal_buffer_ref_t,
    scratch: &RocketOwnedBuffer,
    cube: &ElementwiseCube,
) -> Option<OutputCube> {
    (chain_enabled() && cube.geometry.is_exact()).then_some(OutputCube {
        dense_buffer: binding.buffer,
        dense_offset: binding.offset,
        dma_address: scratch.dma_address,
        handle: scratch.handle,
        host_ptr: scratch.host_ptr,
        length: cube.scratch_bytes,
        geometry: cube.geometry,
    })
}

/// The output half: a scratch cube plus the compaction back to dense NHWC.
fn compact_elementwise_output(
    cb: &mut RocketCommandBuffer,
    binding: &iree_hal_buffer_ref_t,
    cube: &ElementwiseCube,
) -> Option<(RocketOwnedBuffer, OutputCompaction)> {
    if cube.dense_bytes()? as u64 > binding.length as u64 {
        return None;
    }
    let scratch = unsafe {
        RocketOwnedBuffer::new(
            cb.fd,
            cube.scratch_bytes.max(1),
            BorrowedFd::borrow_raw(cb.fd),
        )
    };
    let compaction = OutputCompaction {
        output_buffer: binding.buffer,
        output_offset: binding.offset,
        output_length: binding.length,
        scratch_ptr: scratch.host_ptr,
        scratch_length: cube.scratch_bytes,
        // No reduction: the output cube is the input cube's geometry, so
        // these are equal and there is no surface padding to skip.
        source_pixel_count: cube.pixels,
        output_pixel_count: cube.pixels,
        output_width: cube.width,
        bytes_per_pixel: cube.logical_bytes_per_pixel,
        // One 16-byte feature atom per pixel per surface, which is what the
        // cube strides are counted in.
        source_block_bytes: 16,
        source_tiles: None,
        replica_scratch: Vec::new(),
        tile_context: Vec::new(),
        tile_rects: Vec::new(),
    };
    Some((scratch, compaction))
}

/// Reads one dense NHWC binding, repacks it into its NC1HWC2 scratch buffer
/// and publishes it to the device.
///
/// Extracted from `apply_ops` so a two-tensor element-wise dispatch can run
/// it for both operands. Every other dispatch kind has exactly one input
/// tensor, which is why this used to be inline.
fn apply_input_packing(
    fd: i32,
    packing: &InputPacking,
    profile_label: &str,
) -> Result<(), iree_status_t> {
    let timer = profile::start();
    let dense_len = packing
        .source_pixel_count
        .checked_mul(packing.bytes_per_pixel)
        .ok_or_else(|| {
            status::from_code(crate::bindings::iree_status_code_e_IREE_STATUS_INTERNAL)
        })?;
    if dense_len as u64 > packing.input_length as u64 {
        return Err(status::from_code(
            crate::bindings::iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
        ));
    }
    let input = unsafe { &*(packing.input_buffer as *const RocketBuffer) };
    let dense =
        unsafe { std::slice::from_raw_parts(input.host_ptr.add(packing.input_offset), dense_len) };
    let padded_dense = if packing.source_pixel_count != packing.packed_pixel_count {
        let padded_len = packing
            .packed_pixel_count
            .checked_mul(packing.bytes_per_pixel)
            .ok_or_else(|| {
                status::from_code(crate::bindings::iree_status_code_e_IREE_STATUS_INTERNAL)
            })?;
        let mut padded = vec![packing.padding_byte; padded_len];
        padded[..dense_len].copy_from_slice(dense);
        Some(padded)
    } else {
        None
    };
    let dense = padded_dense.as_deref().unwrap_or(dense);
    let scratch =
        unsafe { std::slice::from_raw_parts_mut(packing.scratch_ptr, packing.scratch_length) };
    let packing_result = match packing.layout {
        InputPackingLayout::Dense => {
            if dense.len() > scratch.len() {
                Err("dense padded input exceeds its scratch buffer")
            } else {
                scratch[..dense.len()].copy_from_slice(dense);
                Ok(dense.len())
            }
        }
        InputPackingLayout::Nc1hwc2 => pack_nhwc_to_nc1hwc2_padded(
            dense,
            packing.packed_pixel_count,
            packing.bytes_per_pixel,
            packing.packed_bytes_per_pixel,
            scratch,
        ),
    };
    if packing_result.is_err() {
        return Err(status::from_code(
            crate::bindings::iree_status_code_e_IREE_STATUS_INTERNAL,
        ));
    }
    if unsafe { iree_rocket_hal::rocket::device::fini_bo(fd, packing.scratch_handle) }.is_err() {
        return Err(status::from_code(
            crate::bindings::iree_status_code_e_IREE_STATUS_INTERNAL,
        ));
    }
    profile::stop(
        timer,
        profile::Phase::PackInput,
        profile_label,
        packing.scratch_length,
    );
    Ok(())
}

pub static VTABLE: iree_hal_command_buffer_vtable_t = iree_hal_command_buffer_vtable_t {
    destroy: Some(destroy),
    begin: Some(begin),
    end: Some(end),
    begin_debug_group: Some(begin_debug_group),
    end_debug_group: Some(end_debug_group),
    execution_barrier: Some(execution_barrier),
    signal_event: Some(signal_event),
    reset_event: Some(reset_event),
    wait_events: Some(wait_events),
    advise_buffer: Some(advise_buffer),
    fill_buffer: Some(fill_buffer),
    update_buffer: Some(update_buffer),
    copy_buffer: Some(copy_buffer),
    collective: Some(collective),
    dispatch: Some(dispatch),
};

#[cfg(test)]
mod lazy_compaction_tests {
    use super::compaction_elidable;

    // The rule behind skipping a dense output write. Each case is one way the
    // write must be kept; only the last skips it.
    #[test]
    fn dense_write_is_kept_unless_every_counted_reader_chained() {
        // No count from the compiler: never skip, however many chained.
        assert!(!compaction_elidable(0, 0, false));
        assert!(!compaction_elidable(0, 3, false));
        // A reader the command buffer never saw -- it is on a later one.
        assert!(!compaction_elidable(2, 1, false));
        // A same-buffer reader that read the dense bytes instead of chaining.
        assert!(!compaction_elidable(1, 1, true));
        // More chained than counted cannot happen, but if it did the count is
        // not to be trusted.
        assert!(!compaction_elidable(1, 2, false));
        assert!(compaction_elidable(1, 1, false));
        assert!(compaction_elidable(2, 2, false));
    }
}
