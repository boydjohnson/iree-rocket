//! `iree_hal_driver_vtable_t` + `iree_hal_driver_factory_t`. Modeled on
//! `iree-null-driver-reference/driver.c`: the actual "not implemented yet"
//! boundary lives in `create()` below (driver construction itself fails),
//! matching that skeleton's pattern of keeping every vtable slot a real,
//! correctly-typed function while placing the single deliberate
//! UNIMPLEMENTED at the outermost, first-reached point.

use crate::bindings::{
    iree_allocator_t, iree_hal_device_create_params_t, iree_hal_device_id_t,
    iree_hal_device_info_t, iree_hal_device_t, iree_hal_driver_factory_t, iree_hal_driver_info_t,
    iree_hal_driver_registry_t, iree_hal_driver_t, iree_hal_driver_vtable_t, iree_host_size_t,
    iree_status_t, iree_string_builder_t, iree_string_pair_t, iree_string_view_t,
};
use crate::status;

void_stub!(destroy(driver: *mut iree_hal_driver_t));

status_stub!(query_available_devices(
    driver: *mut iree_hal_driver_t,
    host_allocator: iree_allocator_t,
    out_device_info_count: *mut iree_host_size_t,
    out_device_infos: *mut *mut iree_hal_device_info_t,
) -> iree_status_t);

status_stub!(dump_device_info(
    driver: *mut iree_hal_driver_t,
    device_id: iree_hal_device_id_t,
    builder: *mut iree_string_builder_t,
) -> iree_status_t);

status_stub!(create_device_by_id(
    driver: *mut iree_hal_driver_t,
    device_id: iree_hal_device_id_t,
    param_count: iree_host_size_t,
    params: *const iree_string_pair_t,
    create_params: *const iree_hal_device_create_params_t,
    host_allocator: iree_allocator_t,
    out_device: *mut *mut iree_hal_device_t,
) -> iree_status_t);

status_stub!(create_device_by_path(
    driver: *mut iree_hal_driver_t,
    driver_name: iree_string_view_t,
    device_path: iree_string_view_t,
    param_count: iree_host_size_t,
    params: *const iree_string_pair_t,
    create_params: *const iree_hal_device_create_params_t,
    host_allocator: iree_allocator_t,
    out_device: *mut *mut iree_hal_device_t,
) -> iree_status_t);

pub static VTABLE: iree_hal_driver_vtable_t = iree_hal_driver_vtable_t {
    destroy: Some(destroy),
    query_available_devices: Some(query_available_devices),
    dump_device_info: Some(dump_device_info),
    create_device_by_id: Some(create_device_by_id),
    create_device_by_path: Some(create_device_by_path),
};

// ============================================================================
// iree_hal_driver_factory_t -- what actually gets registered with IREE's
// driver registry (`iree_hal_driver_registry_register_factory`).
// ============================================================================

unsafe extern "C" fn factory_enumerate(
    _self_: *mut std::ffi::c_void,
    _out_driver_info_count: *mut iree_host_size_t,
    _out_driver_infos: *mut *const iree_hal_driver_info_t,
) -> iree_status_t {
    // TODO: report a real iree_hal_driver_info_t for "rocket" once driver
    // creation (below) actually succeeds -- see driver.c's
    // iree_hal_null_driver_factory_enumerate for the shape.
    status::unimplemented()
}

unsafe extern "C" fn factory_try_create(
    _self_: *mut std::ffi::c_void,
    _driver_name: iree_string_view_t,
    _host_allocator: iree_allocator_t,
    _out_driver: *mut *mut iree_hal_driver_t,
) -> iree_status_t {
    // TODO: this is the real "not implemented yet" boundary -- allocate an
    // iree_hal_resource_t-based driver object (mirroring
    // iree_hal_null_driver_create), probe /dev/accel/accel0 is openable,
    // and only then return it. Every other stub above already has the
    // right shape to be filled in once there's a real driver instance to
    // operate on.
    status::unimplemented()
}

// iree_hal_driver_factory_t's `self_: *mut c_void` field makes it non-Sync
// by default (bindgen has no way to know we always leave it null and never
// alias through it), which a `static` requires. Safe here specifically
// because `self_` is always `null_mut()` -- there's nothing to race on.
struct SyncFactory(iree_hal_driver_factory_t);
unsafe impl Sync for SyncFactory {}

static FACTORY: SyncFactory = SyncFactory(iree_hal_driver_factory_t {
    self_: std::ptr::null_mut(),
    enumerate: Some(factory_enumerate),
    try_create: Some(factory_try_create),
});

/// Mirrors `iree_hal_null_driver_module_register()` -- the entry point a
/// host application calls to make "rocket" available through
/// `iree_hal_driver_registry_t`.
pub unsafe fn register(registry: *mut iree_hal_driver_registry_t) -> iree_status_t {
    unsafe { crate::bindings::iree_hal_driver_registry_register_factory(registry, &FACTORY.0) }
}
