//! `iree_hal_driver_vtable_t` + `iree_hal_driver_factory_t`. Modeled on
//! `iree-null-driver-reference/driver.c`. `factory_try_create` is where
//! `/dev/accel/accel0` actually gets probed; `create_device_by_id`/
//! `create_device_by_path` open a fresh handle and hand off to
//! `device::create`.

use crate::bindings::{
    iree_allocator_t, iree_hal_device_create_params_t, iree_hal_device_id_t,
    iree_hal_device_info_t, iree_hal_device_t, iree_hal_driver_factory_t, iree_hal_driver_info_t,
    iree_hal_driver_registry_t, iree_hal_driver_t, iree_hal_driver_vtable_t, iree_hal_resource_t,
    iree_host_size_t, iree_status_code_e_IREE_STATUS_UNAVAILABLE, iree_status_t,
    iree_string_builder_t, iree_string_pair_t, iree_string_view_t,
};
use crate::status;

const DRIVER_NAME: &[u8] = b"rocket";
const DEVICE_PATH: &str = "/dev/accel/accel0";

/// What every `iree_hal_driver_t*` this driver hands out actually points
/// to. Opaque base type, `resource` at offset 0.
#[repr(C)]
pub struct RocketDriver {
    pub resource: iree_hal_resource_t,
    pub host_allocator: iree_allocator_t,
    /// Owned "rocket" bytes -- `iree_string_view_t`s returned from `id()`/
    /// device info point into this rather than a 'static literal so the
    /// pattern matches real drivers that build the identifier at runtime.
    pub identifier: Vec<u8>,
}

unsafe fn cast(driver: *mut iree_hal_driver_t) -> *mut RocketDriver {
    driver as *mut RocketDriver
}

unsafe extern "C" fn destroy(driver: *mut iree_hal_driver_t) {
    unsafe { drop(Box::from_raw(cast(driver))) }
}

status_stub!(dump_device_info(
    driver: *mut iree_hal_driver_t,
    device_id: iree_hal_device_id_t,
    builder: *mut iree_string_builder_t,
) -> iree_status_t);

#[allow(unused_variables)]
unsafe extern "C" fn query_available_devices(
    driver: *mut iree_hal_driver_t,
    host_allocator: iree_allocator_t,
    out_device_info_count: *mut iree_host_size_t,
    out_device_infos: *mut *mut iree_hal_device_info_t,
) -> iree_status_t {
    // Exactly one device -- this hardware doesn't support enumerating
    // multiple accel nodes today (only /dev/accel/accel0 is probed).
    let info = iree_hal_device_info_t {
        device_id: 0,
        path: iree_string_view_t {
            data: std::ptr::null(),
            size: 0,
        },
        name: iree_string_view_t {
            data: c"default".as_ptr(),
            size: 7,
        },
    };
    unsafe {
        let status = crate::bindings::iree_allocator_clone(
            host_allocator,
            crate::bindings::iree_const_byte_span_t {
                data: &info as *const _ as *const u8,
                data_length: std::mem::size_of::<iree_hal_device_info_t>(),
            },
            out_device_infos as *mut *mut std::ffi::c_void,
        );
        *out_device_info_count = 1;
        status
    }
}

unsafe extern "C" fn create_device_by_id(
    driver: *mut iree_hal_driver_t,
    device_id: iree_hal_device_id_t,
    param_count: iree_host_size_t,
    params: *const iree_string_pair_t,
    create_params: *const iree_hal_device_create_params_t,
    host_allocator: iree_allocator_t,
    out_device: *mut *mut iree_hal_device_t,
) -> iree_status_t {
    let _ = (device_id, param_count, params); // only one device, no path/param parsing yet
    let drv = unsafe { &*cast(driver) };
    unsafe { crate::device::create(&drv.identifier, create_params, host_allocator, out_device) }
}

unsafe extern "C" fn create_device_by_path(
    driver: *mut iree_hal_driver_t,
    driver_name: iree_string_view_t,
    device_path: iree_string_view_t,
    param_count: iree_host_size_t,
    params: *const iree_string_pair_t,
    create_params: *const iree_hal_device_create_params_t,
    host_allocator: iree_allocator_t,
    out_device: *mut *mut iree_hal_device_t,
) -> iree_status_t {
    let _ = (driver_name, device_path, param_count, params); // single fixed device path today
    let drv = unsafe { &*cast(driver) };
    unsafe { crate::device::create(&drv.identifier, create_params, host_allocator, out_device) }
}

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

// *const c_char fields make iree_hal_driver_info_t non-Sync by default --
// safe here since it's entirely immutable 'static data (same reasoning as
// SyncFactory below).
struct SyncDriverInfo(iree_hal_driver_info_t);
unsafe impl Sync for SyncDriverInfo {}

static INFO: SyncDriverInfo = SyncDriverInfo(iree_hal_driver_info_t {
    driver_name: iree_string_view_t {
        data: DRIVER_NAME.as_ptr() as *const std::os::raw::c_char,
        size: DRIVER_NAME.len(),
    },
    full_name: iree_string_view_t {
        data: c"Rockchip RK3588 NPU (accel/rocket)".as_ptr(),
        size: 35,
    },
});

unsafe extern "C" fn factory_enumerate(
    _self_: *mut std::ffi::c_void,
    out_driver_info_count: *mut iree_host_size_t,
    out_driver_infos: *mut *const iree_hal_driver_info_t,
) -> iree_status_t {
    unsafe {
        *out_driver_info_count = 1;
        *out_driver_infos = &INFO.0;
    }
    status::ok()
}

unsafe extern "C" fn factory_try_create(
    _self_: *mut std::ffi::c_void,
    driver_name: iree_string_view_t,
    host_allocator: iree_allocator_t,
    out_driver: *mut *mut iree_hal_driver_t,
) -> iree_status_t {
    let requested =
        unsafe { std::slice::from_raw_parts(driver_name.data as *const u8, driver_name.size) };
    if requested != DRIVER_NAME {
        return status::from_code(iree_status_code_e_IREE_STATUS_UNAVAILABLE);
    }

    // Probe -- confirms the kernel driver is actually present rather than
    // deferring that discovery to the first create_device_by_id call.
    if std::fs::metadata(DEVICE_PATH).is_err() {
        return status::from_code(iree_status_code_e_IREE_STATUS_UNAVAILABLE);
    }

    let driver = Box::new(RocketDriver {
        resource: iree_hal_resource_t {
            ref_count: 1,
            vtable: &VTABLE as *const _ as *const std::ffi::c_void,
        },
        host_allocator,
        identifier: DRIVER_NAME.to_vec(),
    });
    unsafe {
        *out_driver = Box::into_raw(driver) as *mut iree_hal_driver_t;
    }
    status::ok()
}

// self_ = null_mut() below makes iree_hal_driver_factory_t non-Sync by
// default (bindgen has no way to know we never alias through it) -- safe
// here specifically because self_ is always null.
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
