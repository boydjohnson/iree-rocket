//! `iree_hal_device_vtable_t`. `create` opens `/dev/accel/accel0` and
//! wires up the sub-objects (allocator, proactor-from-pool); `queue_execute`
//! is the real dispatch path: wait on `wait_semaphore_list`, pull the
//! regcmd program `command_buffer::dispatch` recorded, write it to a GEM
//! buffer, `SUBMIT`, blocking `PREP_BO`, then signal
//! `signal_semaphore_list` -- the synchronous pattern from
//! `local_sync/sync_device.c` that this crate's research phase identified
//! as the right model for a driver whose only completion signal is a
//! blocking ioctl rather than a native timeline/fence primitive.

use std::os::fd::AsRawFd;

use crate::bindings::{
    iree_allocator_t, iree_const_byte_span_t, iree_device_size_t, iree_hal_alloca_flags_t,
    iree_hal_allocator_t, iree_hal_buffer_binding_table_t, iree_hal_buffer_params_t,
    iree_hal_buffer_ref_list_t, iree_hal_buffer_t, iree_hal_channel_params_t,
    iree_hal_channel_provider_t, iree_hal_channel_t, iree_hal_command_buffer_mode_t,
    iree_hal_command_buffer_t, iree_hal_command_category_t, iree_hal_copy_flags_t,
    iree_hal_dealloca_flags_t, iree_hal_device_capabilities_t, iree_hal_device_create_params_t,
    iree_hal_device_external_capture_options_t, iree_hal_device_profiling_options_t,
    iree_hal_device_t, iree_hal_device_topology_info_t, iree_hal_device_vtable_t,
    iree_hal_dispatch_config_t, iree_hal_dispatch_flags_t, iree_hal_event_flags_t,
    iree_hal_event_t, iree_hal_executable_cache_t, iree_hal_executable_function_t,
    iree_hal_executable_t, iree_hal_execute_flags_t, iree_hal_external_file_flags_t,
    iree_hal_file_t, iree_hal_fill_flags_t, iree_hal_host_call_context_t,
    iree_hal_host_call_flag_bits_e_IREE_HAL_HOST_CALL_FLAG_NON_BLOCKING,
    iree_hal_host_call_flags_t, iree_hal_host_call_t, iree_hal_memory_access_t, iree_hal_pool_t,
    iree_hal_queue_affinity_t, iree_hal_queue_pool_backend_t, iree_hal_read_flags_t,
    iree_hal_resource_t,
    iree_hal_semaphore_compatibility_bits_t_IREE_HAL_SEMAPHORE_COMPATIBILITY_ALL,
    iree_hal_semaphore_compatibility_t, iree_hal_semaphore_flags_t, iree_hal_semaphore_list_t,
    iree_hal_semaphore_t, iree_hal_topology_edge_t, iree_hal_update_flags_t,
    iree_hal_write_flags_t, iree_host_size_t, iree_io_file_handle_t,
    iree_status_code_e_IREE_STATUS_INTERNAL, iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT,
    iree_status_code_e_IREE_STATUS_UNAVAILABLE, iree_status_t, iree_string_view_t, iree_timeout_t,
    iree_timeout_type_e_IREE_TIMEOUT_ABSOLUTE,
};
use crate::buffer::RocketBuffer;
use crate::status;
use iree_rocket_hal::rocket::api::{DRM_IOCTL_BASE, drm_version};
use iree_rocket_hal::rocket::device as rocket_device;

const DEVICE_PATH: &str = "/dev/accel/accel0";

// nr=0x00 is generic DRM_IOCTL_VERSION, not rocket-specific -- see
// rkt-test.rs's identical use of this pattern.
nix::ioctl_readwrite!(drm_get_version, DRM_IOCTL_BASE, 0x00, drm_version);

const EXPECTED_DRIVER_NAME: &[u8] = b"rocket";

/// `/dev/accel/accel0` is a generic DRM accel-subsystem path -- nothing
/// guarantees the node at that path is actually bound to the mainline
/// `rocket` driver rather than some other accelerator's DRM node (e.g. an
/// Intel NPU's `intel_vpu` driver on a dev laptop, which registers under
/// the exact same `/dev/accel/accel0` path). Opening it always succeeds
/// either way, so `std::fs::File::open` alone can't tell them apart --
/// only a DRM_IOCTL_VERSION round-trip (same identification strategy
/// librknnrt.so itself uses to find "rknpu" among `/dev/dri/cardN`, see
/// NOTES.md) can. Sending our rocket-specific ioctls to an unrelated
/// driver's ioctl handler is exactly what caused CTS's buffer/
/// command_buffer suites to abort the whole test process on such hosts
/// (`iree_rocket_hal::rocket::device::Buffer::new`'s `.expect(...)` on an
/// unexpected ioctl failure) instead of the intended, gracefully-skipped
/// UNAVAILABLE outcome.
fn is_rocket_device(file: &std::fs::File) -> bool {
    let fd = file.as_raw_fd();
    let mut version = drm_version {
        version_major: 0,
        version_minor: 0,
        version_patchlevel: 0,
        name_len: 0,
        name: std::ptr::null_mut(),
        date_len: 0,
        date: std::ptr::null_mut(),
        desc_len: 0,
        desc: std::ptr::null_mut(),
    };
    if unsafe { drm_get_version(fd, &mut version) }.is_err() {
        return false;
    }
    let mut name_buf = vec![0u8; version.name_len as usize];
    version.name = name_buf.as_mut_ptr() as *mut std::os::raw::c_char;
    if unsafe { drm_get_version(fd, &mut version) }.is_err() {
        return false;
    }
    name_buf.truncate(version.name_len as usize);
    name_buf == EXPECTED_DRIVER_NAME
}

/// What every `iree_hal_device_t*` this driver hands out actually points
/// to. Opaque base type, `resource` at offset 0.
#[repr(C)]
pub struct RocketDevice {
    pub resource: iree_hal_resource_t,
    pub host_allocator: iree_allocator_t,
    pub identifier: Vec<u8>,
    pub file: std::fs::File,
    pub device_allocator: *mut iree_hal_allocator_t,
    pub proactor_pool: *mut crate::bindings::iree_async_proactor_pool_t,
    pub proactor: *mut crate::bindings::iree_async_proactor_t,
    /// Zeroed (not part of a topology) until `assign_topology_info` is
    /// called by `iree_hal_device_group_builder_finalize` -- mirrors
    /// iree-null-driver-reference/device.c's `topology_info` field. Does
    /// NOT retain/register with `topology_info.frontier.tracker` (unlike
    /// the null driver) -- frontier_tracker.h isn't in this crate's
    /// bindgen allowlist (see build.rs) and nothing exercised by this
    /// driver today needs cross-device causal tracking. Revisit if/when
    /// multi-device topologies or real frontier-based sync matter.
    pub topology_info: crate::bindings::iree_hal_device_topology_info_t,
}

unsafe fn cast(device: *mut iree_hal_device_t) -> *mut RocketDevice {
    device as *mut RocketDevice
}

/// Every `queue_*` op below is synchronous (this driver's only completion
/// signals are blocking ioctls, see module doc comment), so "queue order"
/// is enforced simply: block on every wait semaphore before doing any
/// work, then signal (or, on failure, fail) the signal list after. Shared
/// by `queue_execute` and every `queue_alloca`/`queue_dealloca`/
/// `queue_fill`/`queue_update`/`queue_copy`/`queue_host_call` below.
unsafe fn wait_all(list: iree_hal_semaphore_list_t) -> iree_status_t {
    let infinite = iree_timeout_t {
        type_: iree_timeout_type_e_IREE_TIMEOUT_ABSOLUTE,
        nanos: i64::MAX, // IREE_TIME_INFINITE_FUTURE
    };
    unsafe {
        for i in 0..list.count as isize {
            let sem = *list.semaphores.offset(i);
            let value = *list.payload_values.offset(i);
            let st = crate::bindings::iree_hal_semaphore_wait(sem, value, infinite, 0);
            if !st.is_null() {
                return st;
            }
        }
    }
    status::ok()
}

unsafe fn signal_all(list: iree_hal_semaphore_list_t) -> iree_status_t {
    unsafe { crate::bindings::iree_hal_semaphore_list_signal(list, std::ptr::null()) }
}

unsafe fn fail_all(list: iree_hal_semaphore_list_t, failure: iree_status_t) {
    unsafe { crate::bindings::iree_hal_semaphore_list_fail(list, failure) }
}

/// Wraps a `!Send` value (almost always a raw pointer into IREE's C
/// object graph) so it can move into a thread spawned by
/// `run_after_wait`. Sound here because every value this wraps is either
/// a refcounted HAL object retained before crossing the thread boundary
/// (see the `iree_hal_*_retain` calls at each `run_after_wait` call site)
/// or plain data with no aliasing concerns (byte buffers, small
/// `Copy` structs) that has already been deep-copied out of caller-owned
/// memory.
struct AssertSend<T>(T);
unsafe impl<T> Send for AssertSend<T> {}

impl<T> AssertSend<T> {
    /// Deliberately a method, not a `.0` field access: Rust's 2021+
    /// disjoint-closure-capture rules see straight through a bare `.0`
    /// projection and capture just the inner (non-`Send`) field instead of
    /// the whole `AssertSend<T>` wrapper, silently defeating the `unsafe
    /// impl Send` above. A method call is opaque to that analysis and
    /// forces capture of the whole wrapper.
    fn into_inner(self) -> T {
        self.0
    }
}

/// Deep, retained copy of an `iree_hal_semaphore_list_t`. The original's
/// `semaphores`/`payload_values` arrays are typically stack-allocated in
/// the caller's own frame (e.g. CTS's `SemaphoreList` test helper) and
/// don't outlive the call that handed them to a `queue_*` vtable function
/// -- fine for `run_after_wait`'s synchronous fast path (same call, same
/// frame), fatal for its deferred path, whose background thread runs
/// after the call -- and the caller's frame -- may already have returned.
/// Retains every semaphore on construction and releases them on drop, so
/// the semaphore objects themselves also survive independent of whatever
/// the caller does with its own references in the meantime.
struct OwnedSemaphoreList {
    semaphores: Vec<*mut iree_hal_semaphore_t>,
    payload_values: Vec<u64>,
}

unsafe impl Send for OwnedSemaphoreList {}

impl OwnedSemaphoreList {
    unsafe fn new(list: iree_hal_semaphore_list_t) -> Self {
        // std::slice::from_raw_parts requires a non-null, aligned pointer
        // even for a zero-length slice -- an empty semaphore list (a
        // legitimate, common case: not every queue_* call needs to signal
        // or wait on anything) is typically {count: 0, semaphores: NULL,
        // payload_values: NULL}, which would be UB to hand to it directly.
        if list.count == 0 {
            return Self {
                semaphores: Vec::new(),
                payload_values: Vec::new(),
            };
        }
        let semaphores: Vec<*mut iree_hal_semaphore_t> =
            unsafe { std::slice::from_raw_parts(list.semaphores, list.count as usize).to_vec() };
        let payload_values: Vec<u64> = unsafe {
            std::slice::from_raw_parts(list.payload_values, list.count as usize).to_vec()
        };
        for &sem in &semaphores {
            unsafe { crate::bindings::iree_hal_semaphore_retain(sem) };
        }
        Self {
            semaphores,
            payload_values,
        }
    }

    fn as_list(&mut self) -> iree_hal_semaphore_list_t {
        iree_hal_semaphore_list_t {
            count: self.semaphores.len() as iree_host_size_t,
            semaphores: self.semaphores.as_mut_ptr(),
            payload_values: self.payload_values.as_mut_ptr(),
        }
    }
}

impl Drop for OwnedSemaphoreList {
    fn drop(&mut self) {
        for &sem in &self.semaphores {
            unsafe { crate::bindings::iree_hal_semaphore_release(sem) };
        }
    }
}

/// Real IREE queue semantics: a `queue_*` call enqueues work against the
/// device queue and returns immediately -- it must not block the calling
/// thread on its own `wait_list`. CTS's `QueueAllocaTest.
/// AllocaWithWaitSemaphores` is a direct test of this: it signals its
/// wait semaphore from the SAME thread only *after* the `queue_alloca`
/// call returns, so a driver that blocks inside the call deadlocks. This
/// driver has no real async executor, so "enqueue and return" is
/// approximated with a detached `std::thread` per deferred call when the
/// wait isn't already satisfied -- not efficient, but correct, and every
/// `queue_*` op here is cheap enough (a host memcpy or a blocking ioctl)
/// that thread-per-call overhead doesn't matter for a test-focused
/// driver.
///
/// When `wait_list` is already satisfied, runs `work` synchronously right
/// now instead of spawning anything -- the common case (empty/already-
/// signaled waits), and on failure returns the error directly rather than
/// routing it through `signal_list`, preserving the exact behavior every
/// already-passing test depends on. `work` receives the signal list to
/// use (either the original, in the synchronous case, or an owned/
/// retained copy that outlives the deferred call) for callers that need
/// to hand it to something else (e.g. `queue_host_call`'s callback
/// context) rather than just signaling it themselves.
unsafe fn run_after_wait(
    wait_list: iree_hal_semaphore_list_t,
    signal_list: iree_hal_semaphore_list_t,
    work: impl FnOnce(iree_hal_semaphore_list_t) -> iree_status_t + Send + 'static,
) -> iree_status_t {
    if unsafe { crate::bindings::iree_hal_semaphore_list_poll(wait_list) } {
        let st = work(signal_list);
        if !st.is_null() {
            return st;
        }
        return unsafe { signal_all(signal_list) };
    }

    let mut owned_wait = unsafe { OwnedSemaphoreList::new(wait_list) };
    let mut owned_signal = unsafe { OwnedSemaphoreList::new(signal_list) };
    std::thread::spawn(move || {
        let st = unsafe { wait_all(owned_wait.as_list()) };
        if !st.is_null() {
            unsafe { fail_all(owned_signal.as_list(), st) };
            return;
        }
        let st = work(owned_signal.as_list());
        if !st.is_null() {
            unsafe { fail_all(owned_signal.as_list(), st) };
            return;
        }
        unsafe { signal_all(owned_signal.as_list()) };
    });
    status::ok()
}

/// Mirrors `iree_hal_null_device_create()`. `identifier` is borrowed only
/// for the duration of this call (copied into the device's own storage).
pub unsafe fn create(
    identifier: &[u8],
    create_params: *const iree_hal_device_create_params_t,
    host_allocator: iree_allocator_t,
    out_device: *mut *mut iree_hal_device_t,
) -> iree_status_t {
    if create_params.is_null() {
        return status::from_code(iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT as u32);
    }
    let proactor_pool = unsafe { (*create_params).proactor_pool };
    if proactor_pool.is_null() {
        // Real requirement, not a relaxation this driver invented -- see
        // module doc comment / iree-null-driver-reference/device.c's own
        // IREE_ASSERT_ARGUMENT(create_params->proactor_pool).
        return status::from_code(iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT as u32);
    }

    let file = match std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
    {
        Ok(f) => f,
        Err(_) => return status::from_code(iree_status_code_e_IREE_STATUS_UNAVAILABLE as u32),
    };
    if !is_rocket_device(&file) {
        return status::from_code(iree_status_code_e_IREE_STATUS_UNAVAILABLE as u32);
    }
    let allocator_file = match file.try_clone() {
        Ok(f) => f,
        Err(_) => return status::from_code(iree_status_code_e_IREE_STATUS_UNAVAILABLE as u32),
    };

    unsafe {
        crate::bindings::iree_async_proactor_pool_retain(proactor_pool);
    }
    let mut proactor: *mut crate::bindings::iree_async_proactor_t = std::ptr::null_mut();
    let proactor_status =
        unsafe { crate::bindings::iree_async_proactor_pool_get(proactor_pool, 0, &mut proactor) };
    if !proactor_status.is_null() {
        unsafe {
            crate::bindings::iree_async_proactor_pool_release(proactor_pool);
        }
        return proactor_status;
    }

    let device_allocator = crate::allocator::create(allocator_file, host_allocator);

    let device = Box::new(RocketDevice {
        resource: iree_hal_resource_t {
            ref_count: 1,
            vtable: &VTABLE as *const _ as *const std::ffi::c_void,
        },
        host_allocator,
        identifier: identifier.to_vec(),
        file,
        device_allocator,
        proactor_pool,
        proactor,
        topology_info: unsafe { std::mem::zeroed() },
    });
    let device_ptr = Box::into_raw(device) as *mut iree_hal_device_t;
    // Chicken-and-egg: the allocator needs to know its owning device (see
    // RocketAllocator::device's doc comment) but is itself one of the
    // fields used to construct RocketDevice above, so the device's final
    // pointer doesn't exist until just now.
    unsafe { crate::allocator::set_device(device_allocator, device_ptr) };
    unsafe {
        *out_device = device_ptr;
    }
    status::ok()
}

unsafe extern "C" fn destroy(device: *mut iree_hal_device_t) {
    unsafe {
        let d = &*cast(device);
        crate::bindings::iree_hal_allocator_release(d.device_allocator);
        crate::bindings::iree_async_proactor_pool_release(d.proactor_pool);
        drop(Box::from_raw(cast(device)));
    }
}

unsafe extern "C" fn id(device: *mut iree_hal_device_t) -> iree_string_view_t {
    let d = unsafe { &*cast(device) };
    iree_string_view_t {
        data: d.identifier.as_ptr() as *const std::os::raw::c_char,
        size: d.identifier.len(),
    }
}

unsafe extern "C" fn host_allocator(device: *mut iree_hal_device_t) -> iree_allocator_t {
    unsafe { (*cast(device)).host_allocator }
}

unsafe extern "C" fn device_allocator(device: *mut iree_hal_device_t) -> *mut iree_hal_allocator_t {
    // Not retained -- matches iree-null-driver-reference/device.c's own
    // iree_hal_null_device_allocator (caller borrows it).
    unsafe { (*cast(device)).device_allocator }
}

void_stub!(replace_device_allocator(
    device: *mut iree_hal_device_t,
    new_allocator: *mut iree_hal_allocator_t,
));

void_stub!(replace_channel_provider(
    device: *mut iree_hal_device_t,
    new_provider: *mut iree_hal_channel_provider_t,
));

status_stub!(trim(device: *mut iree_hal_device_t) -> iree_status_t);

/// Real device-configuration query support, unlike most of this file's
/// other `status_stub!`s -- this one is load-bearing at compiled-module
/// load time, not just an optional capability probe. A compiled `.vmfb`
/// targeting `#hal.device.target<"rocket", [#hal.executable.target<
/// "rocket", "rocket-flatbuffer-v1">]>` embeds a compiler-generated device-
/// selection initializer (`TargetDevice::buildDeviceTargetMatch`'s
/// default implementation, `DeviceTargetAttr::buildDeviceIDAndExecutable
/// FormatsMatch`, compiler/src/iree/compiler/Dialect/HAL/IR/HALAttrs.cpp)
/// that queries EXACTLY two categories on whatever device `--device=`
/// already created, before ever executing anything: `"hal.device.id"`
/// (must glob-match this device's own identifier, "rocket") and
/// `"hal.executable.format"` (must match at least one of the compiled
/// module's executable target format strings). Leaving this as the
/// original always-`UNIMPLEMENTED` stub meant EVERY compiled module ever
/// silently failed device selection with a generic "HAL device not found
/// or unavailable" error, no matter how correct the rest of the pipeline
/// was -- found the hard way, on real hardware, compiling the first real
/// program through the new `rocket` IREE compiler plugin (a separate
/// sibling repo). None of this project's own hand-driven CTS tests
/// (`cts/conv_dispatch_test.cc`/`pooling_dispatch_test.cc`) ever exercise
/// this path -- they call `iree_hal_executable_cache_create`/
/// `iree_hal_command_buffer_dispatch` directly, bypassing the compiled-
/// module device-selection initializer entirely.
///
/// Modeled on `iree-null-driver-reference/device.c`'s
/// `iree_hal_null_device_query_i64` (same category names, same "return
/// NOT_FOUND for anything unrecognized" convention) -- this driver's
/// identifier is always the fixed literal `b"rocket"` (`driver.rs`'s
/// `DRIVER_NAME`). The FlatBuffer format is the compiler/runtime contract;
/// the manual format remains advertised temporarily for legacy CTS inputs.
unsafe extern "C" fn query_i64(
    device: *mut iree_hal_device_t,
    category: iree_string_view_t,
    key: iree_string_view_t,
    out_value: *mut i64,
) -> iree_status_t {
    let d = unsafe { &*cast(device) };
    let category =
        unsafe { std::slice::from_raw_parts(category.data as *const u8, category.size as usize) };
    let key = unsafe { std::slice::from_raw_parts(key.data as *const u8, key.size as usize) };

    unsafe {
        *out_value = 0;
    }

    if category == b"hal.device.id" {
        unsafe {
            *out_value = if key == d.identifier.as_slice() { 1 } else { 0 };
        }
        return status::ok();
    }
    if category == b"hal.executable.format" {
        unsafe {
            *out_value = if key == b"rocket-flatbuffer-v1" || key == b"rocket-conv2d-v1" {
                1
            } else {
                0
            };
        }
        return status::ok();
    }

    status::from_code(crate::bindings::iree_status_code_e_IREE_STATUS_NOT_FOUND as u32)
}

#[allow(unused_variables)]
unsafe extern "C" fn query_capabilities(
    device: *mut iree_hal_device_t,
    out_capabilities: *mut iree_hal_device_capabilities_t,
) -> iree_status_t {
    // Zeroed = no special hardware capabilities declared yet -- mirrors
    // iree-null-driver-reference/device.c's query_capabilities exactly.
    // Enough for iree_hal_device_group_builder_finalize() to build a
    // single-device topology (numa_node=0, no import/export types) without
    // UNIMPLEMENTED bubbling up and leaving the CTS's cached device_group
    // half-built.
    unsafe {
        *out_capabilities = std::mem::zeroed();
    }
    status::ok()
}

unsafe extern "C" fn topology_info(
    device: *mut iree_hal_device_t,
) -> *const iree_hal_device_topology_info_t {
    let d = unsafe { &*cast(device) };
    &d.topology_info
}

#[allow(unused_variables)]
unsafe extern "C" fn refine_topology_edge(
    src_device: *mut iree_hal_device_t,
    dst_device: *mut iree_hal_device_t,
    edge: *mut iree_hal_topology_edge_t,
) -> iree_status_t {
    // Only called for same-driver device pairs; rocket has no multi-device
    // interconnect knowledge to contribute yet, so the capability-derived
    // edge stands as-is -- matches iree-null-driver-reference exactly.
    status::ok()
}

unsafe extern "C" fn assign_topology_info(
    device: *mut iree_hal_device_t,
    topology_info: *const iree_hal_device_topology_info_t,
) -> iree_status_t {
    let d = unsafe { &mut *cast(device) };
    d.topology_info = if topology_info.is_null() {
        unsafe { std::mem::zeroed() }
    } else {
        unsafe { std::ptr::read(topology_info) }
    };
    status::ok()
}

status_stub!(create_channel(
    device: *mut iree_hal_device_t,
    queue_affinity: iree_hal_queue_affinity_t,
    params: iree_hal_channel_params_t,
    out_channel: *mut *mut iree_hal_channel_t,
) -> iree_status_t);

#[allow(unused_variables)]
unsafe extern "C" fn create_command_buffer(
    device: *mut iree_hal_device_t,
    mode: iree_hal_command_buffer_mode_t,
    command_categories: iree_hal_command_category_t,
    queue_affinity: iree_hal_queue_affinity_t,
    binding_capacity: iree_host_size_t,
    out_command_buffer: *mut *mut iree_hal_command_buffer_t,
) -> iree_status_t {
    let d = unsafe { &*cast(device) };
    unsafe {
        *out_command_buffer = crate::command_buffer::create(
            d.device_allocator,
            mode,
            command_categories,
            queue_affinity,
            binding_capacity,
        );
    }
    status::ok()
}

#[allow(unused_variables)]
unsafe extern "C" fn create_event(
    device: *mut iree_hal_device_t,
    queue_affinity: iree_hal_queue_affinity_t,
    flags: iree_hal_event_flags_t,
    out_event: *mut *mut iree_hal_event_t,
) -> iree_status_t {
    let d = unsafe { &*cast(device) };
    unsafe { crate::event::create(queue_affinity, flags, d.host_allocator, out_event) }
}

#[allow(unused_variables)]
unsafe extern "C" fn create_executable_cache(
    device: *mut iree_hal_device_t,
    identifier: iree_string_view_t,
    out_executable_cache: *mut *mut iree_hal_executable_cache_t,
) -> iree_status_t {
    unsafe {
        *out_executable_cache = crate::executable_cache::create();
    }
    status::ok()
}

#[allow(unused_variables)]
unsafe extern "C" fn import_file(
    device: *mut iree_hal_device_t,
    queue_affinity: iree_hal_queue_affinity_t,
    access: iree_hal_memory_access_t,
    handle: *mut iree_io_file_handle_t,
    flags: iree_hal_external_file_flags_t,
    out_file: *mut *mut iree_hal_file_t,
) -> iree_status_t {
    let d = unsafe { &*cast(device) };
    unsafe { crate::file::import(access, handle, d.host_allocator, out_file) }
}

#[allow(unused_variables)]
unsafe extern "C" fn create_semaphore(
    device: *mut iree_hal_device_t,
    queue_affinity: iree_hal_queue_affinity_t,
    initial_value: u64,
    flags: iree_hal_semaphore_flags_t,
    out_semaphore: *mut *mut iree_hal_semaphore_t,
) -> iree_status_t {
    let d = unsafe { &*cast(device) };
    unsafe {
        *out_semaphore = crate::semaphore::create(d.proactor, initial_value, d.host_allocator);
    }
    status::ok()
}

#[allow(unused_variables)]
unsafe extern "C" fn query_semaphore_compatibility(
    device: *mut iree_hal_device_t,
    semaphore: *mut iree_hal_semaphore_t,
) -> iree_hal_semaphore_compatibility_t {
    // We don't yet distinguish semaphores created by this device from
    // ones created elsewhere (no cross-driver import support) -- assume
    // full compatibility, matching every semaphore this driver could
    // plausibly be handed today (it's the only HAL driver in the process
    // in any realistic near-term usage).
    iree_hal_semaphore_compatibility_bits_t_IREE_HAL_SEMAPHORE_COMPATIBILITY_ALL
}

status_stub!(query_queue_pool_backend(
    device: *mut iree_hal_device_t,
    queue_affinity: iree_hal_queue_affinity_t,
    out_backend: *mut iree_hal_queue_pool_backend_t,
) -> iree_status_t);

#[allow(unused_variables)]
unsafe extern "C" fn queue_alloca(
    device: *mut iree_hal_device_t,
    queue_affinity: iree_hal_queue_affinity_t,
    wait_semaphore_list: iree_hal_semaphore_list_t,
    signal_semaphore_list: iree_hal_semaphore_list_t,
    pool: *mut iree_hal_pool_t,
    params: iree_hal_buffer_params_t,
    allocation_size: iree_device_size_t,
    flags: iree_hal_alloca_flags_t,
    out_buffer: *mut *mut iree_hal_buffer_t,
) -> iree_status_t {
    if !pool.is_null() {
        // Real pooling (TLSF/FixedBlock/Passthrough pool backends, cross-
        // queue wait-frontier semantics, ...) isn't implemented -- every
        // direct (pool=NULL) allocation below is. See device.rs's task
        // list.
        return status::unimplemented();
    }
    let d = unsafe { &*cast(device) };
    // The real allocation has no data dependency on wait_semaphore_list
    // (nothing has read/written through the new buffer yet), so unlike
    // every other queue_* op here it's safe -- and necessary -- to do
    // eagerly: *out_buffer must be valid the instant this call returns
    // (CTS's QueueAllocaTest.AllocaWithWaitSemaphores checks this
    // immediately, before it ever signals the wait), matching real IREE
    // queue-submission semantics where the call enqueues and returns
    // without blocking on its own wait list. Only the signal is deferred.
    let st = unsafe {
        crate::bindings::iree_hal_allocator_allocate_buffer(
            d.device_allocator,
            params,
            allocation_size,
            out_buffer,
        )
    };
    if !st.is_null() {
        return st;
    }
    unsafe {
        run_after_wait(wait_semaphore_list, signal_semaphore_list, |_sig| {
            status::ok()
        })
    }
}

#[allow(unused_variables)]
unsafe extern "C" fn queue_dealloca(
    device: *mut iree_hal_device_t,
    queue_affinity: iree_hal_queue_affinity_t,
    wait_semaphore_list: iree_hal_semaphore_list_t,
    signal_semaphore_list: iree_hal_semaphore_list_t,
    buffer: *mut iree_hal_buffer_t,
    flags: iree_hal_dealloca_flags_t,
) -> iree_status_t {
    unsafe { crate::bindings::iree_hal_buffer_retain(buffer) };
    let buffer = AssertSend(buffer);
    unsafe {
        run_after_wait(wait_semaphore_list, signal_semaphore_list, move |_sig| {
            let buffer = buffer.into_inner();
            // Logical only -- see RocketBuffer::deallocated's doc comment.
            // The caller retains its own reference and is still
            // responsible for eventually releasing `buffer`; this only
            // makes further queue ops against it correctly fail.
            unsafe {
                crate::buffer::mark_deallocated(buffer);
                crate::bindings::iree_hal_buffer_release(buffer);
            }
            status::ok()
        })
    }
}

// queue_fill/queue_update/queue_copy: the device-level (non-command-buffer)
// counterparts of command_buffer.rs's fill_buffer/update_buffer/copy_buffer.
// Real IREE callers use these directly for standalone transfers outside a
// recorded command buffer -- e.g. the CTS test harness's own
// CreateDeviceBufferWithData helper uses queue_update, not the command-
// buffer op, for uploads above the command-buffer inline-payload size.
// Since these are also host-mapped-memory operations (see buffer.rs), wait
// (via run_after_wait, deferred if not already satisfied) then apply then
// signal, same as every other queue_* op here.
#[allow(unused_variables)]
unsafe extern "C" fn queue_fill(
    device: *mut iree_hal_device_t,
    queue_affinity: iree_hal_queue_affinity_t,
    wait_semaphore_list: iree_hal_semaphore_list_t,
    signal_semaphore_list: iree_hal_semaphore_list_t,
    target_buffer: *mut iree_hal_buffer_t,
    target_offset: iree_device_size_t,
    length: iree_device_size_t,
    pattern: *const std::ffi::c_void,
    pattern_length: iree_host_size_t,
    flags: iree_hal_fill_flags_t,
) -> iree_status_t {
    if pattern_length as usize > 8 {
        return status::from_code(iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT as u32);
    }
    // pattern may point to the caller's stack (a local variable, in every
    // CTS test that hits this) -- copy it now since a deferred call (see
    // run_after_wait) runs after the caller's frame may already be gone.
    let mut pattern_buf = [0u8; 8];
    unsafe {
        std::ptr::copy_nonoverlapping(
            pattern as *const u8,
            pattern_buf.as_mut_ptr(),
            pattern_length as usize,
        );
    }
    unsafe { crate::bindings::iree_hal_buffer_retain(target_buffer) };
    let target_buffer = AssertSend(target_buffer);
    unsafe {
        run_after_wait(wait_semaphore_list, signal_semaphore_list, move |_sig| {
            let target_buffer = target_buffer.into_inner();
            let st = unsafe {
                if crate::buffer::is_deallocated(target_buffer) {
                    status::from_code(iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT as u32)
                } else {
                    crate::bindings::iree_hal_buffer_map_fill(
                        target_buffer,
                        target_offset,
                        length,
                        pattern_buf.as_ptr() as *const std::ffi::c_void,
                        pattern_length,
                    )
                }
            };
            unsafe { crate::bindings::iree_hal_buffer_release(target_buffer) };
            st
        })
    }
}

#[allow(unused_variables)]
unsafe extern "C" fn queue_update(
    device: *mut iree_hal_device_t,
    queue_affinity: iree_hal_queue_affinity_t,
    wait_semaphore_list: iree_hal_semaphore_list_t,
    signal_semaphore_list: iree_hal_semaphore_list_t,
    source_buffer: *const std::ffi::c_void,
    source_offset: iree_host_size_t,
    target_buffer: *mut iree_hal_buffer_t,
    target_offset: iree_device_size_t,
    length: iree_device_size_t,
    flags: iree_hal_update_flags_t,
) -> iree_status_t {
    // source_buffer may point to the caller's stack/heap memory that
    // won't survive past this call if deferred -- copy it now, same
    // reasoning as queue_fill's pattern copy.
    let mut source = vec![0u8; length as usize];
    unsafe {
        let src = (source_buffer as *const u8).add(source_offset as usize);
        std::ptr::copy_nonoverlapping(src, source.as_mut_ptr(), length as usize);
    }
    unsafe { crate::bindings::iree_hal_buffer_retain(target_buffer) };
    let target_buffer = AssertSend(target_buffer);
    unsafe {
        run_after_wait(wait_semaphore_list, signal_semaphore_list, move |_sig| {
            let target_buffer = target_buffer.into_inner();
            let st = unsafe {
                if crate::buffer::is_deallocated(target_buffer) {
                    status::from_code(iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT as u32)
                } else {
                    crate::bindings::iree_hal_buffer_map_write(
                        target_buffer,
                        target_offset,
                        source.as_ptr() as *const std::ffi::c_void,
                        length,
                    )
                }
            };
            unsafe { crate::bindings::iree_hal_buffer_release(target_buffer) };
            st
        })
    }
}

#[allow(unused_variables)]
unsafe extern "C" fn queue_copy(
    device: *mut iree_hal_device_t,
    queue_affinity: iree_hal_queue_affinity_t,
    wait_semaphore_list: iree_hal_semaphore_list_t,
    signal_semaphore_list: iree_hal_semaphore_list_t,
    source_buffer: *mut iree_hal_buffer_t,
    source_offset: iree_device_size_t,
    target_buffer: *mut iree_hal_buffer_t,
    target_offset: iree_device_size_t,
    length: iree_device_size_t,
    flags: iree_hal_copy_flags_t,
) -> iree_status_t {
    unsafe {
        crate::bindings::iree_hal_buffer_retain(source_buffer);
        crate::bindings::iree_hal_buffer_retain(target_buffer);
    }
    let source_buffer = AssertSend(source_buffer);
    let target_buffer = AssertSend(target_buffer);
    unsafe {
        run_after_wait(wait_semaphore_list, signal_semaphore_list, move |_sig| {
            let source_buffer = source_buffer.into_inner();
            let target_buffer = target_buffer.into_inner();
            let st = unsafe {
                if crate::buffer::is_deallocated(source_buffer)
                    || crate::buffer::is_deallocated(target_buffer)
                {
                    status::from_code(iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT as u32)
                } else {
                    crate::bindings::iree_hal_buffer_map_copy(
                        source_buffer,
                        source_offset,
                        target_buffer,
                        target_offset,
                        length,
                    )
                }
            };
            unsafe {
                crate::bindings::iree_hal_buffer_release(source_buffer);
                crate::bindings::iree_hal_buffer_release(target_buffer);
            }
            st
        })
    }
}

// Delegate to IREE's own iree_hal_file_read/write -- generic wrappers that
// dispatch to file.rs's read/write vtable slots, which do the real
// host-memcpy work via iree_hal_buffer_map_write/_read. Same deferred
// wait-then-work-then-signal pattern (via run_after_wait) as every other
// queue_* op here; both the file and the buffer are refcounted HAL
// objects (not raw caller memory), so retaining them for the deferred
// call's duration is enough -- no byte-level copy needed like queue_fill/
// queue_update's pattern/source_buffer.
#[allow(unused_variables)]
unsafe extern "C" fn queue_read(
    device: *mut iree_hal_device_t,
    queue_affinity: iree_hal_queue_affinity_t,
    wait_semaphore_list: iree_hal_semaphore_list_t,
    signal_semaphore_list: iree_hal_semaphore_list_t,
    source_file: *mut iree_hal_file_t,
    source_offset: u64,
    target_buffer: *mut iree_hal_buffer_t,
    target_offset: iree_device_size_t,
    length: iree_device_size_t,
    flags: iree_hal_read_flags_t,
) -> iree_status_t {
    unsafe {
        crate::bindings::iree_hal_file_retain(source_file);
        crate::bindings::iree_hal_buffer_retain(target_buffer);
    }
    let source_file = AssertSend(source_file);
    let target_buffer = AssertSend(target_buffer);
    unsafe {
        run_after_wait(wait_semaphore_list, signal_semaphore_list, move |_sig| {
            let source_file = source_file.into_inner();
            let target_buffer = target_buffer.into_inner();
            let st = unsafe {
                crate::bindings::iree_hal_file_read(
                    source_file,
                    source_offset,
                    target_buffer,
                    target_offset,
                    length,
                )
            };
            unsafe {
                crate::bindings::iree_hal_file_release(source_file);
                crate::bindings::iree_hal_buffer_release(target_buffer);
            }
            st
        })
    }
}

#[allow(unused_variables)]
unsafe extern "C" fn queue_write(
    device: *mut iree_hal_device_t,
    queue_affinity: iree_hal_queue_affinity_t,
    wait_semaphore_list: iree_hal_semaphore_list_t,
    signal_semaphore_list: iree_hal_semaphore_list_t,
    source_buffer: *mut iree_hal_buffer_t,
    source_offset: iree_device_size_t,
    target_file: *mut iree_hal_file_t,
    target_offset: u64,
    length: iree_device_size_t,
    flags: iree_hal_write_flags_t,
) -> iree_status_t {
    unsafe {
        crate::bindings::iree_hal_buffer_retain(source_buffer);
        crate::bindings::iree_hal_file_retain(target_file);
    }
    let source_buffer = AssertSend(source_buffer);
    let target_file = AssertSend(target_file);
    unsafe {
        run_after_wait(wait_semaphore_list, signal_semaphore_list, move |_sig| {
            let source_buffer = source_buffer.into_inner();
            let target_file = target_file.into_inner();
            let st = unsafe {
                crate::bindings::iree_hal_file_write(
                    target_file,
                    target_offset,
                    source_buffer,
                    source_offset,
                    length,
                )
            };
            unsafe {
                crate::bindings::iree_hal_buffer_release(source_buffer);
                crate::bindings::iree_hal_file_release(target_file);
            }
            st
        })
    }
}

// Blocking mode only (IREE_HAL_HOST_CALL_FLAG_NONE) -- deferred/async
// callbacks (a callback returning IREE_STATUS_DEFERRED and signaling the
// cloned signal_semaphore_list itself later) aren't implemented. Neither
// mode may block on wait_semaphore_list itself (see run_after_wait's doc
// comment) -- deferred here too, when the wait isn't already satisfied.
#[allow(unused_variables)]
unsafe extern "C" fn queue_host_call(
    device: *mut iree_hal_device_t,
    queue_affinity: iree_hal_queue_affinity_t,
    wait_semaphore_list: iree_hal_semaphore_list_t,
    signal_semaphore_list: iree_hal_semaphore_list_t,
    call: iree_hal_host_call_t,
    args: *const u64,
    flags: iree_hal_host_call_flags_t,
) -> iree_status_t {
    // args always points to exactly 4 values (iree_hal_device_queue_
    // host_call's own public signature is `const uint64_t args[4]`) and,
    // like every per-call payload here, may point to the caller's stack --
    // copy it now since a deferred call (see run_after_wait) runs after
    // the caller's frame may already be gone.
    let mut args_buf = [0u64; 4];
    unsafe { std::ptr::copy_nonoverlapping(args, args_buf.as_mut_ptr(), 4) };
    let device = AssertSend(device);
    let user_data = AssertSend(call.user_data);

    let non_blocking =
        flags & (iree_hal_host_call_flag_bits_e_IREE_HAL_HOST_CALL_FLAG_NON_BLOCKING as u64) != 0;
    if non_blocking {
        // "The signal semaphores ... will be signaled immediately after the
        // queue has issued the call ... any error returned is ignored."
        // Also omitted from the callback's own context (per
        // iree_hal_host_call_context_t::signal_semaphore_list's doc
        // comment) -- an empty list, not the real one, so the callback
        // can't (successfully) signal them itself and double up with the
        // signal below. This is "signal, then call" (not run_after_wait's
        // "work, then signal on success" ordering), so it's handled
        // directly rather than through that helper.
        let f = call.fn_.map(AssertSend);
        let signal_then_call = move |sig: iree_hal_semaphore_list_t| {
            let sig_status = unsafe { signal_all(sig) };
            if let Some(f) = f {
                let mut context = iree_hal_host_call_context_t {
                    device: device.into_inner(),
                    queue_affinity,
                    signal_semaphore_list: iree_hal_semaphore_list_t {
                        count: 0,
                        semaphores: std::ptr::null_mut(),
                        payload_values: std::ptr::null_mut(),
                    },
                };
                let _ = unsafe {
                    (f.into_inner())(user_data.into_inner(), args_buf.as_ptr(), &mut context)
                };
            }
            sig_status
        };
        if unsafe { crate::bindings::iree_hal_semaphore_list_poll(wait_semaphore_list) } {
            return signal_then_call(signal_semaphore_list);
        }
        let mut owned_wait = unsafe { OwnedSemaphoreList::new(wait_semaphore_list) };
        let mut owned_signal = unsafe { OwnedSemaphoreList::new(signal_semaphore_list) };
        std::thread::spawn(move || {
            let st = unsafe { wait_all(owned_wait.as_list()) };
            if !st.is_null() {
                unsafe { fail_all(owned_signal.as_list(), st) };
                return;
            }
            signal_then_call(owned_signal.as_list());
        });
        return status::ok();
    }

    let Some(f) = call.fn_ else {
        return status::from_code(iree_status_code_e_IREE_STATUS_INVALID_ARGUMENT as u32);
    };
    let f = AssertSend(f);
    // Not routed through run_after_wait: its fast path returns a failed
    // `work()` directly as the caller's own return status, which is right
    // for e.g. queue_fill (a validation failure is a synchronous
    // rejection) but wrong here -- queue_host_call's own documented
    // contract is "if the call succeeds and returns OK the semaphores
    // will be signaled and otherwise they will be failed": the callback's
    // result is ALWAYS routed through fail_all/signal_all, and this
    // function ALWAYS returns OK once past the `call.fn_` precondition
    // check above, on both the fast and deferred paths alike. Getting
    // this wrong (returning the callback's error directly on the fast
    // path) surfaced as CTS's QueueHostCallTest.CallbackReturnsError
    // seeing the raw PERMISSION_DENIED as queue_host_call's own return
    // value, then hanging forever waiting on a signal_semaphore_list that
    // never got signaled or failed.
    let invoke_and_report = move |sig: iree_hal_semaphore_list_t| -> iree_status_t {
        let mut context = iree_hal_host_call_context_t {
            device: device.into_inner(),
            queue_affinity,
            signal_semaphore_list: sig,
        };
        let result =
            unsafe { (f.into_inner())(user_data.into_inner(), args_buf.as_ptr(), &mut context) };
        if result.is_null() {
            unsafe { signal_all(sig) }
        } else {
            unsafe { fail_all(sig, result) };
            status::ok()
        }
    };
    if unsafe { crate::bindings::iree_hal_semaphore_list_poll(wait_semaphore_list) } {
        return invoke_and_report(signal_semaphore_list);
    }
    let mut owned_wait = unsafe { OwnedSemaphoreList::new(wait_semaphore_list) };
    let mut owned_signal = unsafe { OwnedSemaphoreList::new(signal_semaphore_list) };
    std::thread::spawn(move || {
        let st = unsafe { wait_all(owned_wait.as_list()) };
        if !st.is_null() {
            unsafe { fail_all(owned_signal.as_list(), st) };
            return;
        }
        invoke_and_report(owned_signal.as_list());
    });
    status::ok()
}

status_stub!(queue_dispatch(
    device: *mut iree_hal_device_t,
    queue_affinity: iree_hal_queue_affinity_t,
    wait_semaphore_list: iree_hal_semaphore_list_t,
    signal_semaphore_list: iree_hal_semaphore_list_t,
    executable: *mut iree_hal_executable_t,
    function: iree_hal_executable_function_t,
    config: iree_hal_dispatch_config_t,
    constants: iree_const_byte_span_t,
    bindings: iree_hal_buffer_ref_list_t,
    flags: iree_hal_dispatch_flags_t,
) -> iree_status_t);

#[allow(unused_variables)]
/// Interleaves DPU feature-atomic output surfaces into dense NHWC pixels.
///
/// Each 16-byte surface stores one channel group for every spatial pixel;
/// `DPU_DST_SURF_STRIDE` advances between those full spatial surfaces.
/// Dense NHWC instead stores every channel group for one pixel contiguously,
/// so copy one surface chunk at a time into each destination pixel. The final
/// logical surface may use fewer than 16 bytes.
fn compact_atomic_output(
    scratch: &[u8],
    source_pixel_count: usize,
    output_pixel_count: usize,
    bytes_per_pixel: usize,
    dst: &mut [u8],
) -> usize {
    const ATOMIC_STRIDE: usize = 16;
    let mut written = 0;
    for pixel in 0..output_pixel_count {
        let mut pixel_written = 0;
        while pixel_written < bytes_per_pixel {
            let surface = pixel_written / ATOMIC_STRIDE;
            let chunk_len = (bytes_per_pixel - pixel_written).min(ATOMIC_STRIDE);
            let src_off = surface * source_pixel_count * ATOMIC_STRIDE + pixel * ATOMIC_STRIDE;
            let dst_off = pixel * bytes_per_pixel + pixel_written;
            if src_off + chunk_len > scratch.len() || dst_off + chunk_len > dst.len() {
                return written;
            }
            dst[dst_off..dst_off + chunk_len]
                .copy_from_slice(&scratch[src_off..src_off + chunk_len]);
            written += chunk_len;
            pixel_written += chunk_len;
        }
    }
    written
}

#[cfg(test)]
mod compaction_tests {
    use super::compact_atomic_output;

    #[test]
    fn compacts_16_byte_slots_into_dense_output() {
        const PIXELS: usize = 4;
        const BPP: usize = 2;
        let mut scratch = vec![0xAAu8; PIXELS * 16];
        for i in 0..PIXELS {
            scratch[i * 16] = i as u8;
            scratch[i * 16 + 1] = 0x10 + i as u8;
        }
        let mut dst = vec![0u8; PIXELS * BPP];
        let written = compact_atomic_output(&scratch, PIXELS, PIXELS, BPP, &mut dst);
        assert_eq!(written, PIXELS * BPP);
        assert_eq!(dst, vec![0x00, 0x10, 0x01, 0x11, 0x02, 0x12, 0x03, 0x13]);
    }

    #[test]
    fn interleaves_multiple_channel_surfaces() {
        const PIXELS: usize = 2;
        const BPP: usize = 20;
        let mut scratch = vec![0xAAu8; PIXELS * 2 * 16];
        scratch[0..16].copy_from_slice(&[0; 16]);
        scratch[16..32].copy_from_slice(&[1; 16]);
        scratch[32..48].copy_from_slice(&[2; 16]);
        scratch[48..64].copy_from_slice(&[3; 16]);

        let mut dst = vec![0u8; PIXELS * BPP];
        let written = compact_atomic_output(&scratch, PIXELS, PIXELS, BPP, &mut dst);
        assert_eq!(written, PIXELS * BPP);
        assert_eq!(&dst[0..16], &[0; 16]);
        assert_eq!(&dst[16..20], &[2; 4]);
        assert_eq!(&dst[20..36], &[1; 16]);
        assert_eq!(&dst[36..40], &[3; 4]);
    }

    #[test]
    fn stops_short_when_dst_too_small() {
        let scratch = vec![0u8; 64];
        let mut dst = vec![0u8; 2];
        let written = compact_atomic_output(&scratch, 4, 4, 2, &mut dst);
        assert_eq!(written, 2);
    }

    #[test]
    fn stops_short_when_scratch_too_small() {
        let scratch = vec![0u8; 16];
        let mut dst = vec![0u8; 8];
        let written = compact_atomic_output(&scratch, 4, 4, 2, &mut dst);
        assert_eq!(written, 2);
    }

    #[test]
    fn compacts_fc_row_zero_with_physical_height_padding() {
        const LOGICAL_PIXELS: usize = 2;
        const PHYSICAL_PIXELS: usize = LOGICAL_PIXELS * 4;
        const BPP: usize = 20;
        let mut scratch = vec![0xAAu8; PHYSICAL_PIXELS * 2 * 16];
        scratch[0..16].copy_from_slice(&[1; 16]);
        scratch[16..32].copy_from_slice(&[2; 16]);
        let second_surface = PHYSICAL_PIXELS * 16;
        scratch[second_surface..second_surface + 16].copy_from_slice(&[3; 16]);
        scratch[second_surface + 16..second_surface + 32].copy_from_slice(&[4; 16]);

        let mut dst = vec![0u8; LOGICAL_PIXELS * BPP];
        let written =
            compact_atomic_output(&scratch, PHYSICAL_PIXELS, LOGICAL_PIXELS, BPP, &mut dst);
        assert_eq!(written, LOGICAL_PIXELS * BPP);
        assert_eq!(&dst[0..16], &[1; 16]);
        assert_eq!(&dst[16..20], &[3; 4]);
        assert_eq!(&dst[20..36], &[2; 16]);
        assert_eq!(&dst[36..40], &[4; 4]);
    }
}

unsafe extern "C" fn queue_execute(
    device: *mut iree_hal_device_t,
    queue_affinity: iree_hal_queue_affinity_t,
    wait_semaphore_list: iree_hal_semaphore_list_t,
    signal_semaphore_list: iree_hal_semaphore_list_t,
    command_buffer: *mut iree_hal_command_buffer_t,
    binding_table: iree_hal_buffer_binding_table_t,
    flags: iree_hal_execute_flags_t,
) -> iree_status_t {
    // `command_buffer` may legitimately be NULL: `iree_hal_device_queue_
    // barrier()` (device.c) calls `iree_hal_device_queue_execute(..., NULL,
    // ...)` directly as a pure sync point with no actual work -- e.g.
    // `iree_hal_device_queue_dealloca()` takes this path for any buffer
    // whose placement isn't flagged ASYNCHRONOUS (true for every buffer
    // this driver hands out, since allocator::allocate_buffer never sets
    // that flag), which is how CTS's `TransientBufferTest.
    // DeallocateTransient` ends up calling straight into `queue_execute`
    // with no command buffer at all -- segfaulted here (null-deref inside
    // apply_ops) before this check existed.
    if !command_buffer.is_null() {
        unsafe { crate::bindings::iree_hal_command_buffer_retain(command_buffer) };
    }
    let device = AssertSend(device);
    let command_buffer = AssertSend(command_buffer);
    unsafe {
        run_after_wait(wait_semaphore_list, signal_semaphore_list, move |_sig| {
            let device = device.into_inner();
            let command_buffer = command_buffer.into_inner();
            let d = unsafe { &*cast(device) };

            // Replay every recorded fill/update/copy (host-side, no
            // hardware involved) and pull out the one recorded
            // `dispatch`'s regcmd, if any. A non-null command buffer with
            // no recorded `dispatch` (e.g. CTS's EventTest -- just
            // signal_event/reset_event, nothing to actually run on the
            // NPU) is likewise a legitimate empty job, not an error: skip
            // straight to signaling. Matches the coarse per-command-buffer
            // sync granularity documented in event.rs -- there's
            // genuinely nothing to submit to hardware.
            let cmds = if command_buffer.is_null() {
                None
            } else {
                match unsafe { crate::command_buffer::apply_ops(command_buffer) } {
                    Ok(cmds) => cmds,
                    Err(st) => {
                        unsafe { crate::bindings::iree_hal_command_buffer_release(command_buffer) };
                        return st;
                    }
                }
            };
            let result = 'result: {
                if let Some(job) = cmds.filter(|j| !j.regcmd_tasks.is_empty()) {
                    let fd = d.file.as_raw_fd();
                    let regcmd_tasks = job.regcmd_tasks;
                    if regcmd_tasks.iter().any(Vec::is_empty) {
                        break 'result status::from_code(
                            iree_status_code_e_IREE_STATUS_INTERNAL as u32,
                        );
                    }

                    // Allocate every split before submission so all command
                    // buffers remain alive until the dispatch is complete.
                    let mut cmd_bufs = Vec::with_capacity(regcmd_tasks.len());
                    for regcmd in regcmd_tasks {
                        let cmd_bytes = regcmd.len() * std::mem::size_of::<u64>();
                        let cmd_len = cmd_bytes.next_multiple_of(4096);
                        cmd_bufs.push(unsafe { rocket_device::Buffer::new(fd, cmd_len, &d.file) });
                    }

                    let mut task_descriptors = Vec::with_capacity(regcmd_tasks.len());
                    for (regcmd, cmd_buf) in regcmd_tasks.iter().zip(&cmd_bufs) {
                        unsafe {
                            let cmd_slice = std::slice::from_raw_parts_mut(
                                cmd_buf.host_ptr as *mut u64,
                                regcmd.len(),
                            );
                            for (i, c) in regcmd.iter().enumerate() {
                                cmd_slice[i] = c.0;
                            }
                        }
                        if unsafe { rocket_device::fini_bo(fd, cmd_buf.handle) }.is_err() {
                            break 'result status::from_code(
                                iree_status_code_e_IREE_STATUS_UNAVAILABLE as u32,
                            );
                        }
                        task_descriptors.push((cmd_buf.dma_address, regcmd.len() as u32));
                    }

                    // Real input/output GEM handles from the command
                    // buffer's recorded dispatch (command_buffer.rs's
                    // `RecordedOp::Dispatch`), same convention
                    // conv_hw.rs's hand-rolled ioctl call already validates
                    // against real hardware: the regcmd buffer's own
                    // handle plus every buffer the dispatch reads go in
                    // in_bo_handles, everything it writes goes in
                    // out_bo_handles. Without these the kernel driver's
                    // implicit fencing has no way to know the job touches
                    // those buffers at all -- SUBMIT/PREP_BO would still
                    // round-trip (proving the ioctl plumbing works) but
                    // give no real completion/dependency guarantee for the
                    // tensors themselves.
                    // Deduplicated: a real compiled program can pack
                    // multiple bindings (e.g. input+weights) into one
                    // combined transient buffer at different offsets, so
                    // the same GEM handle can legitimately appear more than
                    // once in job.in_bo_handles -- untested against the
                    // kernel driver's fence/dependency tracking before now
                    // (every prior hand-driven test used one dedicated
                    // buffer per binding), so dedupe defensively rather
                    // than assume DRM_ROCKET_SUBMIT tolerates duplicates.
                    let mut in_handles =
                        Vec::with_capacity(cmd_bufs.len() + job.in_bo_handles.len());
                    for cmd_buf in &cmd_bufs {
                        in_handles.push(cmd_buf.handle);
                    }
                    for &h in job.in_bo_handles {
                        if !in_handles.contains(&h) {
                            in_handles.push(h);
                        }
                    }
                    // The mainline driver's IRQ-mediated transition between
                    // tasks in one drm_rocket_job is not reliable on RK3588:
                    // task 0 completes correctly, but every later split leaves
                    // its output rows untouched. Submitting the same regcmds
                    // as individually fenced jobs is hardware-validated. Each
                    // split reloads its weights, so no CBUF state must survive
                    // between jobs.
                    for &(regcmd_addr, regcmd_count) in &task_descriptors {
                        if unsafe {
                            rocket_device::submit(
                                fd,
                                regcmd_addr,
                                regcmd_count,
                                &in_handles,
                                job.out_bo_handles,
                            )
                        }
                        .is_err()
                        {
                            break 'result status::from_code(
                                iree_status_code_e_IREE_STATUS_UNAVAILABLE as u32,
                            );
                        }

                        // PREP_BO waits on DMA_RESV_USAGE_WRITE fences for the
                        // specific output handle. Besides making results
                        // host-visible, waiting here prevents the next split
                        // from entering the kernel until this one has completed.
                        for &out_handle in job.out_bo_handles {
                            if unsafe { rocket_device::prep_bo(fd, out_handle, 2_000_000_000) }
                                .is_err()
                            {
                                break 'result status::from_code(
                                    iree_status_code_e_IREE_STATUS_UNAVAILABLE as u32,
                                );
                            }
                        }
                    }

                    // Conv2d only (see `command_buffer::OutputCompaction`'s
                    // doc comment): the hardware write above landed in a
                    // driver-private scratch buffer, atomic-slot-strided,
                    // not the real IREE-visible output buffer. Compact it
                    // into the real buffer now that prep_bo above confirmed
                    // the write is complete and host-visible.
                    if let Some(oc) = job.output_compaction {
                        let expected_bytes = oc.output_pixel_count * oc.bytes_per_pixel;
                        if expected_bytes as u64 > oc.output_length as u64 {
                            break 'result status::from_code(
                                iree_status_code_e_IREE_STATUS_INTERNAL as u32,
                            );
                        }
                        let scratch = unsafe {
                            std::slice::from_raw_parts(oc.scratch_ptr, oc.scratch_length)
                        };
                        let out_rb = unsafe { &*(oc.output_buffer as *const RocketBuffer) };
                        let dst = unsafe {
                            std::slice::from_raw_parts_mut(
                                out_rb.host_ptr.add(oc.output_offset as usize),
                                expected_bytes,
                            )
                        };
                        compact_atomic_output(
                            scratch,
                            oc.source_pixel_count,
                            oc.output_pixel_count,
                            oc.bytes_per_pixel,
                            dst,
                        );
                    }
                }
                status::ok()
            };
            if !command_buffer.is_null() {
                unsafe { crate::bindings::iree_hal_command_buffer_release(command_buffer) };
            }
            result
        })
    }
}

status_stub!(queue_flush(
    device: *mut iree_hal_device_t,
    queue_affinity: iree_hal_queue_affinity_t,
) -> iree_status_t);

status_stub!(profiling_begin(
    device: *mut iree_hal_device_t,
    options: *const iree_hal_device_profiling_options_t,
) -> iree_status_t);

status_stub!(profiling_flush(device: *mut iree_hal_device_t) -> iree_status_t);
status_stub!(profiling_end(device: *mut iree_hal_device_t) -> iree_status_t);

status_stub!(external_capture_begin(
    device: *mut iree_hal_device_t,
    options: *const iree_hal_device_external_capture_options_t,
) -> iree_status_t);

status_stub!(external_capture_end(device: *mut iree_hal_device_t) -> iree_status_t);

pub static VTABLE: iree_hal_device_vtable_t = iree_hal_device_vtable_t {
    destroy: Some(destroy),
    id: Some(id),
    host_allocator: Some(host_allocator),
    device_allocator: Some(device_allocator),
    replace_device_allocator: Some(replace_device_allocator),
    replace_channel_provider: Some(replace_channel_provider),
    trim: Some(trim),
    query_i64: Some(query_i64),
    query_capabilities: Some(query_capabilities),
    topology_info: Some(topology_info),
    refine_topology_edge: Some(refine_topology_edge),
    assign_topology_info: Some(assign_topology_info),
    create_channel: Some(create_channel),
    create_command_buffer: Some(create_command_buffer),
    create_event: Some(create_event),
    create_executable_cache: Some(create_executable_cache),
    import_file: Some(import_file),
    create_semaphore: Some(create_semaphore),
    query_semaphore_compatibility: Some(query_semaphore_compatibility),
    query_queue_pool_backend: Some(query_queue_pool_backend),
    queue_alloca: Some(queue_alloca),
    queue_dealloca: Some(queue_dealloca),
    queue_fill: Some(queue_fill),
    queue_update: Some(queue_update),
    queue_copy: Some(queue_copy),
    queue_read: Some(queue_read),
    queue_write: Some(queue_write),
    queue_host_call: Some(queue_host_call),
    queue_dispatch: Some(queue_dispatch),
    queue_execute: Some(queue_execute),
    queue_flush: Some(queue_flush),
    profiling_begin: Some(profiling_begin),
    profiling_flush: Some(profiling_flush),
    profiling_end: Some(profiling_end),
    external_capture_begin: Some(external_capture_begin),
    external_capture_end: Some(external_capture_end),
};
