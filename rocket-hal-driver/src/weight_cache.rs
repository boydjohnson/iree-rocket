//! Cross-inference cache for packed convolution coefficients.
//!
//! IREE's conv ABI hands this driver a logical HWCF filter; the CNA reads a
//! blocked coefficient stream (output-block, input-group, X, Y, output-lane,
//! input-lane). `command_buffer::apply_ops` bridges the two with
//! `pack_hwcf_to_rocket_weights` and friends -- per dispatch, per inference.
//! Profiling MobileNetV2 fp16 (`ROCKET_PROFILE=1`, ISSUES.md P6) put that at
//! 99 ms of a 321 ms run, a third of all host time, for data that never
//! changes: a `.vmfb`'s filters are constants written once at module load.
//!
//! This caches the packed GEM buffer so the transform runs once per
//! (weight binding, geometry) rather than once per dispatch. There is no
//! within-inference reuse to be had -- MobileNetV2's 36 weight-bearing
//! dispatches all have different filters -- so the whole win is from the
//! second inference onward, which is what a benchmark loop or a served model
//! actually does.
//!
//! # Why a hit is safe
//!
//! A packed buffer is only reusable if the bytes it was packed from have not
//! changed. Three things establish that, and all three must hold:
//!
//! 1. **The source buffer is the same object.** The key carries the
//!    `iree_hal_buffer_t*`, and [`forget`] drops every entry for a buffer
//!    from `buffer::destroy`, so a recycled allocation at the same address
//!    cannot inherit the previous occupant's entry.
//! 2. **Nothing has written it since.** `buffer::RocketBuffer` carries a
//!    generation counter bumped on every write this driver can observe, and
//!    an entry is only a hit at the generation it was packed at. Every host
//!    write to an IREE buffer funnels through `buffer::unmap_range` with
//!    write access -- `iree_hal_buffer_map_fill`/`_write`/`_copy` are what
//!    the command-buffer ops, the `queue_*` ops and `iree_hal_file_read` all
//!    reduce to -- and the one device-side write that does not (a dispatch's
//!    output compaction, which writes `host_ptr` directly) bumps it
//!    explicitly. Those two sites are the complete set; adding a third way to
//!    write a buffer means adding a bump with it.
//! 3. **Nothing is about to write it before this dispatch runs.** A hit is
//!    refused if any operation already recorded on the same command buffer
//!    targets the weight binding -- exactly the case deferred packing exists
//!    for (see `WeightPacking`'s doc comment). The generation check cannot
//!    catch this one on its own, because the recorded write has not been
//!    applied yet at the point the dispatch is recorded.
//!
//! What that does *not* cover is another thread writing the weight binding
//! between this dispatch being recorded and being submitted, with no HAL
//! barrier between them. That program is already racy against the deferred
//! packing this replaces (it would pack a torn mixture); the cache turns it
//! into reading the older bytes instead.
//!
//! ```text
//! ROCKET_WEIGHT_CACHE=0        disable; pack every dispatch, as before
//! ROCKET_WEIGHT_CACHE=verify   pack anyway on a hit and compare, loudly
//! ROCKET_WEIGHT_CACHE_MB=N     cache byte budget, default 256 MiB
//! ```

use std::{
    collections::HashMap,
    ops::Deref,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

use iree_rocket_hal::rocket::device::OwnedBuffer;

use crate::bindings::iree_hal_buffer_t;

/// A packed-coefficient GEM allocation shared between the cache and every
/// command buffer that points a regcmd at it.
///
/// `OwnedBuffer` holds a raw host mapping and so is neither `Send` nor
/// `Sync`. Sharing one is sound here because it is written exactly once --
/// by the `apply_ops` that packed it, before [`publish`] makes it reachable
/// -- and is read-only for the rest of its life, by other threads' regcmds
/// and by the NPU. The allocation itself outlives every reference: the
/// `Arc` is what closes the GEM handle.
pub struct SharedBuffer(OwnedBuffer);

// SAFETY: see the type's doc comment -- immutable after publication, and the
// underlying GEM handle/mapping is valid for as long as any `Arc` is.
unsafe impl Send for SharedBuffer {}
unsafe impl Sync for SharedBuffer {}

impl SharedBuffer {
    pub fn new(buffer: OwnedBuffer) -> Arc<SharedBuffer> {
        Arc::new(SharedBuffer(buffer))
    }
}

impl Deref for SharedBuffer {
    type Target = OwnedBuffer;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Everything about a dispatch that changes the packed bytes.
///
/// Two dispatches sharing a weight binding but differing in any of these
/// pack differently, so they are distinct entries rather than a collision.
/// `scratch_length` is derived from the rest, and is carried so a mismatch
/// shows up as a miss rather than as a buffer of the wrong size.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Geometry {
    pub filter_height: usize,
    pub filter_width: usize,
    pub input_channels: usize,
    pub output_channels: usize,
    pub programmed_output_channels: usize,
    pub element_size: usize,
    pub depthwise: bool,
    pub padded_channels: usize,
    pub weight_zero_point: Option<i8>,
    pub scratch_length: usize,
}

/// Identifies the packed form of one weight binding.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Key {
    /// The source `iree_hal_buffer_t*`, as an address. Never dereferenced
    /// through the key -- `forget` keeps it from outliving the buffer.
    pub buffer: usize,
    pub offset: u64,
    pub length: u64,
    pub geometry: Geometry,
}

struct Entry {
    generation: u64,
    buffer: Arc<SharedBuffer>,
    bytes: usize,
}

#[derive(Default)]
struct Cache {
    entries: HashMap<Key, Entry>,
    bytes: usize,
    /// High-water mark. `bytes` is near zero by the time the report runs --
    /// every entry is keyed on an IREE buffer, and those are destroyed (and
    /// so forgotten) before the device is -- so the live figure says nothing
    /// about how much the cache actually held.
    peak_bytes: usize,
    hits: u64,
    /// No entry for this key at all -- a first sight, or an entry `forget`
    /// dropped because the source buffer was destroyed.
    misses_absent: u64,
    /// An entry existed but was packed at an older generation, so something
    /// wrote the weight binding since. Separated from `misses_absent`
    /// because they mean opposite things: a binding that is stale every
    /// inference is being rewritten every inference, which is a cost above
    /// this driver, not a cache that is failing to do its job.
    misses_stale: u64,
    refused_recorded_writer: u64,
    refused_budget: u64,
}

fn cache() -> &'static Mutex<Cache> {
    static CACHE: std::sync::OnceLock<Mutex<Cache>> = std::sync::OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(Cache::default()))
}

fn lock() -> std::sync::MutexGuard<'static, Cache> {
    cache()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn setting() -> &'static str {
    static SETTING: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    SETTING.get_or_init(|| std::env::var("ROCKET_WEIGHT_CACHE").unwrap_or_default())
}

/// Whether packed coefficients are reused at all (`ROCKET_WEIGHT_CACHE=0`
/// restores the pack-every-dispatch behaviour, for A/B measurement and as an
/// escape hatch).
pub fn enabled() -> bool {
    setting() != "0"
}

/// Whether a hit should still pack, into a private buffer, and compare
/// (`ROCKET_WEIGHT_CACHE=verify`).
///
/// The comparison is against the buffer the regcmd actually points at, so it
/// checks the thing that matters: a missed generation bump, a key that does
/// not separate two genuinely different packings, or a stale entry all show
/// up here as a byte mismatch instead of as quietly wrong logits.
pub fn verifying() -> bool {
    setting() == "verify"
}

fn budget_bytes() -> usize {
    static BUDGET: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *BUDGET.get_or_init(|| {
        std::env::var("ROCKET_WEIGHT_CACHE_MB")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(256)
            .saturating_mul(1024 * 1024)
    })
}

/// The packed buffer for `key`, if one was packed at `generation`.
pub fn lookup(key: &Key, generation: u64) -> Option<Arc<SharedBuffer>> {
    if !enabled() {
        return None;
    }
    let mut cache = lock();
    match cache.entries.get(key) {
        Some(entry) if entry.generation == generation => {
            let buffer = Arc::clone(&entry.buffer);
            cache.hits += 1;
            Some(buffer)
        }
        Some(_) => {
            // The source has been rewritten since. Drop the entry rather
            // than leave a permanently-missing one holding its bytes.
            if let Some(entry) = cache.entries.remove(key) {
                cache.bytes = cache.bytes.saturating_sub(entry.bytes);
            }
            cache.misses_stale += 1;
            None
        }
        None => {
            cache.misses_absent += 1;
            None
        }
    }
}

/// Records a dispatch whose own command buffer already has a recorded write
/// to its weight binding, so no cached packing could be used for it.
pub fn note_recorded_writer() {
    if !enabled() {
        return;
    }
    lock().refused_recorded_writer += 1;
}

/// Makes a freshly packed buffer reusable. Called after the packing that
/// filled it has completed, never before.
pub fn publish(key: Key, generation: u64, buffer: Arc<SharedBuffer>, bytes: usize) {
    if !enabled() {
        return;
    }
    let mut cache = lock();
    let previous = cache.entries.remove(&key);
    if let Some(previous) = previous {
        cache.bytes = cache.bytes.saturating_sub(previous.bytes);
    }
    if cache.bytes.saturating_add(bytes) > budget_bytes() {
        cache.refused_budget += 1;
        return;
    }
    cache.bytes += bytes;
    cache.peak_bytes = cache.peak_bytes.max(cache.bytes);
    cache.entries.insert(
        key,
        Entry {
            generation,
            buffer,
            bytes,
        },
    );
}

/// Drops every entry keyed on `buffer`, from `buffer::destroy`.
///
/// Without this a later allocation landing on the same address would inherit
/// the dead buffer's entries -- the classic ABA, and the reason the key can
/// be a bare address at all.
pub fn forget(buffer: *mut iree_hal_buffer_t) {
    if !enabled() {
        return;
    }
    let address = buffer as usize;
    let mut cache = lock();
    let mut freed = 0;
    cache.entries.retain(|key, entry| {
        if key.buffer == address {
            freed += entry.bytes;
            false
        } else {
            true
        }
    });
    cache.bytes = cache.bytes.saturating_sub(freed);
}

/// Drops every entry, from `device::destroy`.
///
/// A cached allocation holds a `RawFd` copy of the device's DRM file, so it
/// must not outlive the device. In practice `forget` has already emptied the
/// cache by then -- every entry is keyed on an IREE buffer, and those are
/// released before the device is -- but nothing in the HAL contract promises
/// that ordering to this module, so do not rely on it.
pub fn clear() {
    let mut cache = lock();
    cache.entries.clear();
    cache.bytes = 0;
}

/// Hits, misses, refusals and peak resident bytes, for the `ROCKET_PROFILE`
/// report.
pub fn stats() -> Stats {
    let cache = lock();
    Stats {
        hits: cache.hits,
        misses_absent: cache.misses_absent,
        misses_stale: cache.misses_stale,
        recorded_writers: cache.refused_recorded_writer,
        over_budget: cache.refused_budget,
        peak_bytes: cache.peak_bytes,
    }
}

/// What [`stats`] reports.
pub struct Stats {
    pub hits: u64,
    pub misses_absent: u64,
    pub misses_stale: u64,
    pub recorded_writers: u64,
    pub over_budget: u64,
    pub peak_bytes: usize,
}

/// Monotonic write counter for one IREE buffer.
///
/// Lives here rather than on `RocketBuffer` alone so the ordering rules stay
/// with the reasoning that depends on them: [`bump`] is release, [`current`]
/// is acquire, so a reader that sees generation N also sees every byte
/// written before the bump that produced it.
#[derive(Default)]
pub struct Generation(AtomicU64);

impl Generation {
    pub fn current(&self) -> u64 {
        self.0.load(Ordering::Acquire)
    }

    pub fn bump(&self) {
        self.0.fetch_add(1, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry() -> Geometry {
        Geometry {
            filter_height: 1,
            filter_width: 1,
            input_channels: 32,
            output_channels: 64,
            programmed_output_channels: 64,
            element_size: 2,
            depthwise: false,
            padded_channels: 0,
            weight_zero_point: None,
            scratch_length: 4096,
        }
    }

    #[test]
    fn geometry_separates_otherwise_identical_bindings() {
        let base = Key {
            buffer: 0x1000,
            offset: 0,
            length: 4096,
            geometry: geometry(),
        };
        let mut depthwise = base;
        depthwise.geometry.depthwise = true;
        assert_ne!(base, depthwise);

        let mut padded = base;
        padded.geometry.programmed_output_channels = 96;
        assert_ne!(base, padded);

        let mut quantized = base;
        quantized.geometry.weight_zero_point = Some(0);
        assert_ne!(base, quantized);
    }

    #[test]
    fn a_write_invalidates_the_generation() {
        let generation = Generation::default();
        assert_eq!(generation.current(), 0);
        generation.bump();
        assert_eq!(generation.current(), 1);
    }
}
