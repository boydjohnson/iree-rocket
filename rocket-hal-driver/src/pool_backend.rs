//! Explicit `iree_hal_pool_t` support: what
//! `iree_hal_device_query_queue_pool_backend` hands out, and the state of a
//! buffer `queue_alloca` reserves from such a pool.
//!
//! IREE's pools (passthrough, TLSF, fixed-block) are host-side bookkeeping
//! over *slabs*, which a slab provider acquires from the driver. Ours makes
//! every slab one ordinary allocator buffer -- a CREATE_BO, host-mapped and
//! NPU-visible -- and wraps a reserved range of it as a `RocketBuffer` whose
//! `handle` is the slab's and whose `dma_address`/`host_ptr` are offset into
//! it. That keeps a pooled buffer bindable by a dispatch, which casts its
//! bindings straight to `RocketBuffer` (`buffer::Backing`).
//!
//! The device exposes exactly one queue (bit 0 of the affinity mask), so
//! there is one backend; other bits are `INVALID_ARGUMENT`, which is how a
//! caller learns the queue does not exist. Every queue operation here is
//! complete by the time it signals, so reservations are released with no
//! death frontier: a freed block is immediately reusable, and a pool never
//! has to hand out a block that still needs a wait.

use std::sync::{
    Mutex,
    atomic::{AtomicPtr, AtomicU64, Ordering},
};

use crate::{
    bindings::{
        iree_async_notification_t, iree_async_proactor_t, iree_device_size_t, iree_hal_allocator_t,
        iree_hal_buffer_params_t, iree_hal_buffer_release_callback_t, iree_hal_buffer_t,
        iree_hal_buffer_usage_t, iree_hal_memory_type_t, iree_hal_pool_reservation_t,
        iree_hal_pool_t, iree_hal_slab_provider_stats_t, iree_hal_slab_provider_t,
        iree_hal_slab_provider_trim_flags_t, iree_hal_slab_provider_visited_set_t,
        iree_hal_slab_provider_vtable_t, iree_hal_slab_t, iree_status_t,
    },
    buffer::{Backing, RocketBuffer},
    status,
};

/// The slab provider and pool notification every explicit pool on this
/// device is built from. Created on the first backend query; the device
/// holds one reference to each and drops it in `destroy` (pools retain their
/// own).
pub struct PoolBackend {
    pub slab_provider: *mut iree_hal_slab_provider_t,
    pub notification: *mut iree_async_notification_t,
}

unsafe impl Send for PoolBackend {}

impl PoolBackend {
    /// # Safety
    ///
    /// `allocator` must be this driver's live allocator and `proactor` the
    /// device's proactor.
    pub unsafe fn create(
        allocator: *mut iree_hal_allocator_t,
        proactor: *mut iree_async_proactor_t,
    ) -> Result<Self, iree_status_t> {
        let mut notification = std::ptr::null_mut();
        let st = unsafe {
            crate::bindings::iree_async_notification_create(
                proactor,
                crate::bindings::iree_async_notification_flag_bits_e_IREE_ASYNC_NOTIFICATION_FLAG_NONE,
                &mut notification,
            )
        };
        if !st.is_null() {
            return Err(st);
        }
        unsafe { crate::bindings::iree_hal_allocator_retain(allocator) };
        let provider = Box::new(RocketSlabProvider {
            base: iree_hal_slab_provider_t {
                ref_count: 1,
                vtable: &SLAB_PROVIDER_VTABLE,
            },
            allocator,
            acquired: AtomicU64::new(0),
            released: AtomicU64::new(0),
        });
        Ok(Self {
            slab_provider: Box::into_raw(provider) as *mut iree_hal_slab_provider_t,
            notification,
        })
    }
}

impl Drop for PoolBackend {
    fn drop(&mut self) {
        unsafe {
            release_slab_provider(self.slab_provider);
            crate::bindings::iree_async_notification_release(self.notification);
        }
    }
}

/// `iree_hal_slab_provider_release`, which lives in iree::hal::memory -- a
/// library a binary linking this driver need not link. Same protocol: the
/// count is the provider's first field, and the last reference destroys it.
unsafe fn release_slab_provider(provider: *mut iree_hal_slab_provider_t) {
    let count = unsafe { std::sync::atomic::AtomicI32::from_ptr(&raw mut (*provider).ref_count) };
    if count.fetch_sub(1, Ordering::AcqRel) == 1 {
        unsafe { slab_provider_destroy(provider) };
    }
}

#[repr(C)]
struct RocketSlabProvider {
    base: iree_hal_slab_provider_t,
    /// Retained: slabs are allocated through it, and a pool can outlive the
    /// device that handed out this provider.
    allocator: *mut iree_hal_allocator_t,
    acquired: AtomicU64,
    released: AtomicU64,
}

unsafe fn provider_cast(provider: *const iree_hal_slab_provider_t) -> *mut RocketSlabProvider {
    provider as *mut RocketSlabProvider
}

unsafe extern "C" fn slab_provider_destroy(provider: *mut iree_hal_slab_provider_t) {
    let provider = unsafe { Box::from_raw(provider_cast(provider)) };
    unsafe { crate::bindings::iree_hal_allocator_release(provider.allocator) };
}

/// A slab is one allocator buffer; `provider_handle` is that buffer, retained
/// until `release_slab`.
unsafe extern "C" fn acquire_slab(
    provider: *mut iree_hal_slab_provider_t,
    min_length: iree_device_size_t,
    out_slab: *mut iree_hal_slab_t,
) -> iree_status_t {
    let provider = unsafe { &*provider_cast(provider) };
    let params = iree_hal_buffer_params_t {
        usage: crate::bindings::iree_hal_buffer_usage_bits_t_IREE_HAL_BUFFER_USAGE_TRANSFER
            | crate::bindings::iree_hal_buffer_usage_bits_t_IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE,
        access: crate::bindings::iree_hal_memory_access_bits_t_IREE_HAL_MEMORY_ACCESS_ALL as u16,
        type_: crate::bindings::iree_hal_memory_type_bits_t_IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        queue_affinity: 1,
        min_alignment: 0,
    };
    let mut buffer = std::ptr::null_mut();
    let st = unsafe {
        crate::bindings::iree_hal_allocator_allocate_buffer(
            provider.allocator,
            params,
            min_length,
            &mut buffer,
        )
    };
    if !st.is_null() {
        return st;
    }
    provider.acquired.fetch_add(1, Ordering::Relaxed);
    let rb = unsafe { &*(buffer as *const RocketBuffer) };
    unsafe {
        *out_slab = iree_hal_slab_t {
            base_ptr: rb.host_ptr,
            length: min_length,
            provider_handle: buffer as u64,
        };
    }
    status::ok()
}

unsafe extern "C" fn release_slab(
    provider: *mut iree_hal_slab_provider_t,
    slab: *const iree_hal_slab_t,
) {
    let provider = unsafe { &*provider_cast(provider) };
    provider.released.fetch_add(1, Ordering::Relaxed);
    unsafe {
        crate::bindings::iree_hal_buffer_release((*slab).provider_handle as *mut iree_hal_buffer_t)
    };
}

/// `[slab_offset, slab_offset + allocation_size)` of the slab's BO as a
/// `RocketBuffer` view. The view retains the slab's buffer, so the BO
/// outlives it even if the pool releases the slab first.
unsafe extern "C" fn wrap_buffer(
    provider: *mut iree_hal_slab_provider_t,
    slab: *const iree_hal_slab_t,
    slab_offset: iree_device_size_t,
    allocation_size: iree_device_size_t,
    params: iree_hal_buffer_params_t,
    release_callback: iree_hal_buffer_release_callback_t,
    out_buffer: *mut *mut iree_hal_buffer_t,
) -> iree_status_t {
    let _ = provider;
    let slab = unsafe { &*slab };
    let in_range = slab_offset
        .checked_add(allocation_size)
        .is_some_and(|end| end <= slab.length);
    if !in_range {
        return status::from_code(crate::bindings::iree_status_code_e_IREE_STATUS_OUT_OF_RANGE);
    }
    let parent = slab.provider_handle as *mut iree_hal_buffer_t;
    let slab_buffer = unsafe { &*(parent as *const RocketBuffer) };
    let Ok(dma_address) = u32::try_from(slab_buffer.dma_address as u64 + slab_offset as u64) else {
        return status::from_code(crate::bindings::iree_status_code_e_IREE_STATUS_OUT_OF_RANGE);
    };
    unsafe { crate::bindings::iree_hal_buffer_retain(parent) };
    let view = Box::new(RocketBuffer {
        base: unsafe { std::mem::zeroed() },
        handle: slab_buffer.handle,
        dma_address,
        host_ptr: unsafe { slab_buffer.host_ptr.add(slab_offset) },
        fd: slab_buffer.fd,
        deallocated: std::sync::atomic::AtomicBool::new(false),
        generation: crate::weight_cache::Generation::default(),
        synced_generation: AtomicU64::new(u64::MAX),
        mapped_bytes: slab_buffer.mapped_bytes.saturating_sub(slab_offset),
        cache: std::sync::Arc::clone(&slab_buffer.cache),
        backing: Backing::SlabView {
            parent,
            release: release_callback,
        },
    });
    let view = Box::into_raw(view);
    unsafe {
        initialize_base(
            view,
            crate::bindings::iree_hal_buffer_allocation_placement(parent),
            params,
            allocation_size,
        );
        *out_buffer = view as *mut iree_hal_buffer_t;
    }
    status::ok()
}

/// Fills `buffer.base` the way `allocator::allocate_buffer` does.
unsafe fn initialize_base(
    buffer: *mut RocketBuffer,
    placement: crate::bindings::iree_hal_buffer_placement_t,
    params: iree_hal_buffer_params_t,
    size: iree_device_size_t,
) {
    unsafe {
        crate::bindings::iree_hal_buffer_initialize(
            placement,
            &mut (*buffer).base,
            size,
            0,
            size,
            crate::allocator::host_mapped_memory_type(params.type_),
            params.access,
            crate::allocator::host_mapped_usage(params.usage),
            &crate::buffer::VTABLE,
            &mut (*buffer).base,
        );
    }
}

unsafe extern "C" fn prefault(provider: *mut iree_hal_slab_provider_t, slab: *mut iree_hal_slab_t) {
    // CREATE_BO commits every page at allocation.
    let _ = (provider, slab);
}

unsafe extern "C" fn trim(
    provider: *mut iree_hal_slab_provider_t,
    flags: iree_hal_slab_provider_trim_flags_t,
) {
    // Nothing is cached here; freed slabs go straight back to the
    // allocator, whose own buffer cache does the keeping.
    let _ = (provider, flags);
}

unsafe extern "C" fn query_stats(
    provider: *const iree_hal_slab_provider_t,
    visited: *mut iree_hal_slab_provider_visited_set_t,
    out_stats: *mut iree_hal_slab_provider_stats_t,
) {
    // `iree_hal_slab_provider_visited`, inlined for the same reason as
    // `release_slab_provider`.
    let visited = unsafe { &mut *visited };
    let seen = &visited.providers[..visited.count.min(visited.providers.len())];
    if seen.contains(&provider) {
        return;
    }
    if visited.count >= visited.providers.len() {
        return;
    }
    visited.providers[visited.count] = provider;
    visited.count += 1;
    let p = unsafe { &*provider_cast(provider) };
    unsafe {
        (*out_stats).total_acquired += p.acquired.load(Ordering::Relaxed);
        (*out_stats).total_released += p.released.load(Ordering::Relaxed);
    }
}

unsafe extern "C" fn query_properties(
    provider: *const iree_hal_slab_provider_t,
    out_memory_type: *mut iree_hal_memory_type_t,
    out_supported_usage: *mut iree_hal_buffer_usage_t,
) {
    let _ = provider;
    unsafe {
        *out_memory_type = crate::allocator::host_mapped_memory_type(
            crate::bindings::iree_hal_memory_type_bits_t_IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        );
        *out_supported_usage = crate::allocator::host_mapped_usage(
            crate::bindings::iree_hal_buffer_usage_bits_t_IREE_HAL_BUFFER_USAGE_TRANSFER
                | crate::bindings::iree_hal_buffer_usage_bits_t_IREE_HAL_BUFFER_USAGE_DISPATCH
                | crate::bindings::iree_hal_buffer_usage_bits_t_IREE_HAL_BUFFER_USAGE_SHARING_IMMUTABLE,
        );
    }
}

static SLAB_PROVIDER_VTABLE: iree_hal_slab_provider_vtable_t = iree_hal_slab_provider_vtable_t {
    destroy: Some(slab_provider_destroy),
    acquire_slab: Some(acquire_slab),
    release_slab: Some(release_slab),
    wrap_buffer: Some(wrap_buffer),
    prefault: Some(prefault),
    trim: Some(trim),
    query_stats: Some(query_stats),
    query_properties: Some(query_properties),
};

/// What a `RocketBuffer` from `queue_alloca(pool)` owns: the pool, the
/// reservation, and the slab view the reservation was materialized as.
///
/// The buffer exists before its storage does when the pool is exhausted at
/// `queue_alloca` time: IREE requires the call to return the buffer at once
/// and signal only when the bytes are usable. Until [`Pooled::commit`] the
/// buffer's address fields are zero and `buffer::is_pending` is true, which
/// mapping and dispatch recording refuse.
pub struct Pooled {
    /// Retained.
    pool: *mut iree_hal_pool_t,
    /// `Some` from commit until `queue_dealloca` (or destroy) hands it back.
    reservation: Mutex<Option<iree_hal_pool_reservation_t>>,
    /// The slab view, retained; null until committed.
    view: AtomicPtr<iree_hal_buffer_t>,
}

unsafe impl Send for Pooled {}
unsafe impl Sync for Pooled {}

impl Pooled {
    pub fn is_committed(&self) -> bool {
        !self.view.load(Ordering::Acquire).is_null()
    }

    /// Returns the reservation to the pool, which wakes anything parked on
    /// its notification. Idempotent. The view stays: the bytes are still
    /// mapped, the buffer is merely marked deallocated.
    pub fn release_reservation(&self) {
        let reservation = self.reservation.lock().unwrap().take();
        if let Some(reservation) = reservation {
            unsafe {
                crate::bindings::iree_hal_pool_release_reservation(
                    self.pool,
                    &reservation,
                    std::ptr::null(),
                )
            };
        }
    }

    /// # Safety
    ///
    /// Called once, from `buffer::destroy`.
    pub unsafe fn destroy(self) {
        self.release_reservation();
        let view = self.view.swap(std::ptr::null_mut(), Ordering::AcqRel);
        unsafe {
            if !view.is_null() {
                crate::bindings::iree_hal_buffer_release(view);
            }
            crate::bindings::iree_hal_pool_release(self.pool);
        }
    }
}

/// A reservation attempt. `Ok(None)` when the pool has nothing to give now
/// (exhausted, over budget) -- retry after its notification.
unsafe fn try_reserve(
    pool: *mut iree_hal_pool_t,
    size: iree_device_size_t,
    alignment: iree_device_size_t,
) -> Result<Option<iree_hal_pool_reservation_t>, iree_status_t> {
    let mut reservation: iree_hal_pool_reservation_t = unsafe { std::mem::zeroed() };
    let mut info: crate::bindings::iree_hal_pool_acquire_info_t = unsafe { std::mem::zeroed() };
    let mut result =
        crate::bindings::iree_hal_pool_acquire_result_e_IREE_HAL_POOL_ACQUIRE_EXHAUSTED;
    // No ALLOW_WAIT_FRONTIER: every release here carries a null frontier, so
    // a pool has no block that needs one.
    let st = unsafe {
        crate::bindings::iree_hal_pool_acquire_reservation(
            pool,
            size,
            alignment,
            std::ptr::null(),
            crate::bindings::iree_hal_pool_reserve_flag_bits_e_IREE_HAL_POOL_RESERVE_FLAG_NONE,
            &mut reservation,
            &mut info,
            &mut result,
        )
    };
    if !st.is_null() {
        return Err(st);
    }
    match result {
        crate::bindings::iree_hal_pool_acquire_result_e_IREE_HAL_POOL_ACQUIRE_OK
        | crate::bindings::iree_hal_pool_acquire_result_e_IREE_HAL_POOL_ACQUIRE_OK_FRESH => {
            Ok(Some(reservation))
        }
        crate::bindings::iree_hal_pool_acquire_result_e_IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT => {
            // Not asked for, so a well-behaved pool never answers this. Give
            // the block back with its dependency intact and wait like an
            // exhaustion.
            unsafe {
                crate::bindings::iree_hal_pool_release_reservation(
                    pool,
                    &reservation,
                    info.wait_frontier,
                )
            };
            Ok(None)
        }
        _ => Ok(None),
    }
}

/// Materializes `reservation` and publishes its address into `buffer`.
///
/// # Safety
///
/// `buffer` must be a live pooled `RocketBuffer` that is not yet committed,
/// and nothing may read its address fields until this returns -- the
/// queue_alloca contract: the caller waits for the alloca signal, which is
/// raised only after this.
unsafe fn commit(
    buffer: *mut RocketBuffer,
    params: iree_hal_buffer_params_t,
    reservation: iree_hal_pool_reservation_t,
) -> iree_status_t {
    let Backing::Pooled(pooled) = (unsafe { &(*buffer).backing }) else {
        return status::from_code(crate::bindings::iree_status_code_e_IREE_STATUS_INTERNAL);
    };
    let mut view = std::ptr::null_mut();
    let st = unsafe {
        crate::bindings::iree_hal_pool_materialize_reservation(
            pooled.pool,
            params,
            &reservation,
            crate::bindings::iree_hal_pool_materialize_flag_bits_e_IREE_HAL_POOL_MATERIALIZE_FLAG_NONE,
            &mut view,
        )
    };
    if !st.is_null() {
        unsafe {
            crate::bindings::iree_hal_pool_release_reservation(
                pooled.pool,
                &reservation,
                std::ptr::null(),
            )
        };
        return st;
    }
    // The view is our own wrap_buffer's `RocketBuffer`.
    let v = unsafe { &*(view as *const RocketBuffer) };
    unsafe {
        (&raw mut (*buffer).handle).write(v.handle);
        (&raw mut (*buffer).dma_address).write(v.dma_address);
        (&raw mut (*buffer).host_ptr).write(v.host_ptr);
        (&raw mut (*buffer).fd).write(v.fd);
        (&raw mut (*buffer).mapped_bytes).write(v.mapped_bytes);
    }
    *pooled.reservation.lock().unwrap() = Some(reservation);
    pooled.view.store(view, Ordering::Release);
    status::ok()
}

/// The pool-backed half of `device::queue_alloca`: returns the buffer at
/// once, and calls `on_committed` (which owns the wait/signal ordering) only
/// once the storage is in place -- synchronously when the pool has room now,
/// from a parked thread after the pool's notification otherwise. On the
/// parked path a failure reaches `on_committed` as its status, which it must
/// route to the signal list; on the synchronous path an error it returns is
/// the call's own.
///
/// # Safety
///
/// `device` and `allocator` must be the live device and its allocator,
/// `pool` a live pool, `out_buffer` writable.
#[allow(clippy::too_many_arguments)]
pub unsafe fn alloca(
    device: *mut crate::bindings::iree_hal_device_t,
    allocator: *mut iree_hal_allocator_t,
    pool: *mut iree_hal_pool_t,
    params: iree_hal_buffer_params_t,
    allocation_size: iree_device_size_t,
    out_buffer: *mut *mut iree_hal_buffer_t,
    on_committed: impl FnOnce(*mut iree_hal_buffer_t, iree_status_t) -> iree_status_t + Send + 'static,
) -> iree_status_t {
    let alloc = unsafe { &*(allocator as *const crate::allocator::RocketAllocator) };
    unsafe { crate::bindings::iree_hal_pool_retain(pool) };
    let buffer = Box::into_raw(Box::new(RocketBuffer {
        base: unsafe { std::mem::zeroed() },
        handle: 0,
        dma_address: 0,
        host_ptr: std::ptr::null_mut(),
        fd: std::os::fd::AsRawFd::as_raw_fd(&alloc.file),
        deallocated: std::sync::atomic::AtomicBool::new(false),
        generation: crate::weight_cache::Generation::default(),
        synced_generation: AtomicU64::new(u64::MAX),
        mapped_bytes: 0,
        cache: std::sync::Arc::clone(&alloc.cache),
        backing: Backing::Pooled(Box::new(Pooled {
            pool,
            reservation: Mutex::new(None),
            view: AtomicPtr::new(std::ptr::null_mut()),
        })),
    }));
    let placement = crate::bindings::iree_hal_buffer_placement_t {
        device,
        // See allocator::allocate_buffer: one queue, and ASYNCHRONOUS so
        // iree_hal_device_queue_dealloca reaches our queue_dealloca.
        queue_affinity: 1,
        flags:
            crate::bindings::iree_hal_buffer_placement_flag_bits_t_IREE_HAL_BUFFER_PLACEMENT_FLAG_ASYNCHRONOUS,
        reserved: 0,
    };
    unsafe { initialize_base(buffer, placement, params, allocation_size) };
    let buffer_ptr = buffer as *mut iree_hal_buffer_t;

    let alignment = params.min_alignment.max(1);
    let reserved = match unsafe { try_reserve(pool, allocation_size, alignment) } {
        Ok(reserved) => reserved,
        Err(st) => {
            unsafe { crate::bindings::iree_hal_buffer_release(buffer_ptr) };
            return st;
        }
    };
    if let Some(reservation) = reserved {
        let st = unsafe { commit(buffer, params, reservation) };
        if !st.is_null() {
            unsafe { crate::bindings::iree_hal_buffer_release(buffer_ptr) };
            return st;
        }
        let st = on_committed(buffer_ptr, status::ok());
        if !st.is_null() {
            unsafe { crate::bindings::iree_hal_buffer_release(buffer_ptr) };
            return st;
        }
        unsafe { *out_buffer = buffer_ptr };
        return status::ok();
    }

    let notification = unsafe { crate::bindings::iree_hal_pool_notification(pool) };
    if notification.is_null() {
        unsafe { crate::bindings::iree_hal_buffer_release(buffer_ptr) };
        return status::from_code(
            crate::bindings::iree_status_code_e_IREE_STATUS_RESOURCE_EXHAUSTED,
        );
    }
    // The parked thread holds its own reference, so the caller may drop
    // theirs before the storage arrives.
    unsafe { crate::bindings::iree_hal_buffer_retain(buffer_ptr) };
    let buffer = SendPtr(buffer);
    let notification = SendPtr(notification);
    let pool = SendPtr(pool);
    std::thread::spawn(move || {
        let buffer = buffer.get();
        let notification = notification.get();
        let pool = pool.get();
        let infinite = crate::bindings::iree_timeout_t {
            type_: crate::bindings::iree_timeout_type_e_IREE_TIMEOUT_ABSOLUTE,
            nanos: i64::MAX,
        };
        // Observe, re-check, wait: a release between the failed attempt and
        // the wait advances the epoch, so it cannot be missed.
        let outcome = loop {
            let token =
                unsafe { crate::bindings::iree_async_notification_begin_observe(notification) };
            let attempt = unsafe { try_reserve(pool, allocation_size, alignment) };
            let reserved = match attempt {
                Ok(Some(reservation)) => Some(Ok(reservation)),
                Ok(None) => {
                    unsafe {
                        crate::bindings::iree_async_notification_wait_for_token(
                            notification,
                            token,
                            infinite,
                        )
                    };
                    None
                }
                Err(st) => Some(Err(st)),
            };
            unsafe { crate::bindings::iree_async_notification_end_observe(notification) };
            if let Some(reserved) = reserved {
                break reserved;
            }
        };
        let st = match outcome {
            Ok(reservation) => unsafe { commit(buffer, params, reservation) },
            Err(st) => st,
        };
        let buffer_ptr = buffer as *mut iree_hal_buffer_t;
        let _ = on_committed(buffer_ptr, st);
        unsafe { crate::bindings::iree_hal_buffer_release(buffer_ptr) };
    });
    unsafe { *out_buffer = buffer_ptr };
    status::ok()
}

/// Returns `buffer`'s reservation to its pool if it came from [`alloca`];
/// a no-op for any other buffer. `queue_dealloca` calls this once its waits
/// are satisfied.
///
/// # Safety
///
/// `buffer` must be a live `RocketBuffer`.
pub unsafe fn release_reservation(buffer: *mut iree_hal_buffer_t) {
    if let Backing::Pooled(pooled) = unsafe { &(*(buffer as *const RocketBuffer)).backing } {
        pooled.release_reservation();
    }
}

/// Moves raw pointers into the parked thread. Sound because each one is a
/// refcounted IREE object retained for the thread's lifetime (the buffer, and
/// through it the pool and its notification).
struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}

impl<T> SendPtr<T> {
    /// A method, not `.0`: closures capture a bare field projection, which
    /// would capture the raw pointer and lose the `Send` impl.
    fn get(self) -> *mut T {
        self.0
    }
}
