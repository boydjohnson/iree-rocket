//! `iree_hal_file_vtable_t`. Only backs
//! `IREE_IO_FILE_HANDLE_TYPE_HOST_ALLOCATION` handles -- the only kind any
//! current caller ever imports (CTS's `test_base.h` `ReadBufferData` helper
//! wraps a plain host `std::vector` via
//! `iree_io_file_handle_wrap_host_allocation` when it needs to read back a
//! buffer that isn't host-mappable/`MAPPING_SCOPED`, e.g.
//! `TransientBufferTest`'s buffers). Real disk-file (FD-backed) import
//! stays `UNIMPLEMENTED`.
//!
//! `read`/`write` are plain host memcpy via `iree_hal_buffer_map_write`/
//! `_read` (buffer.rs's `map_range`/`unmap_range` already back those) --
//! `storage_buffer`/`async_handle` stay unset since this file type has
//! neither a real HAL buffer nor an async-proactor-backed handle behind it,
//! just a raw host span.

use crate::bindings::{
    iree_allocator_t, iree_async_file_t, iree_device_size_t, iree_hal_buffer_t, iree_hal_file_t,
    iree_hal_file_vtable_t, iree_hal_memory_access_t, iree_hal_resource_t, iree_io_file_handle_t,
    iree_io_file_handle_type_e_IREE_IO_FILE_HANDLE_TYPE_HOST_ALLOCATION, iree_status_t,
};
use crate::status;

pub struct RocketFile {
    pub resource: iree_hal_resource_t,
    pub host_allocator: iree_allocator_t,
    /// Retained for this file's lifetime, released in `destroy`.
    pub handle: *mut iree_io_file_handle_t,
    pub access: iree_hal_memory_access_t,
    pub data: *mut u8,
    pub data_length: usize,
}

unsafe fn cast(file: *mut iree_hal_file_t) -> *mut RocketFile {
    file as *mut RocketFile
}

pub unsafe fn import(
    access: iree_hal_memory_access_t,
    handle: *mut iree_io_file_handle_t,
    host_allocator: iree_allocator_t,
    out_file: *mut *mut iree_hal_file_t,
) -> iree_status_t {
    let primitive = unsafe { crate::bindings::iree_io_file_handle_primitive(handle) };
    if primitive.type_ != iree_io_file_handle_type_e_IREE_IO_FILE_HANDLE_TYPE_HOST_ALLOCATION {
        return status::unimplemented();
    }
    let span = unsafe { primitive.value.host_allocation };

    unsafe {
        crate::bindings::iree_io_file_handle_retain(handle);
    }
    let file = Box::new(RocketFile {
        resource: iree_hal_resource_t {
            ref_count: 1,
            vtable: &VTABLE as *const _ as *const std::ffi::c_void,
        },
        host_allocator,
        handle,
        access,
        data: span.data,
        data_length: span.data_length as usize,
    });
    unsafe {
        *out_file = Box::into_raw(file) as *mut iree_hal_file_t;
    }
    status::ok()
}

unsafe extern "C" fn destroy(file: *mut iree_hal_file_t) {
    unsafe {
        let f = Box::from_raw(cast(file));
        crate::bindings::iree_io_file_handle_release(f.handle);
    }
}

unsafe extern "C" fn allowed_access(file: *mut iree_hal_file_t) -> iree_hal_memory_access_t {
    unsafe { (*cast(file)).access }
}

unsafe extern "C" fn length(file: *mut iree_hal_file_t) -> u64 {
    unsafe { (*cast(file)).data_length as u64 }
}

#[allow(unused_variables)]
unsafe extern "C" fn storage_buffer(file: *mut iree_hal_file_t) -> *mut iree_hal_buffer_t {
    std::ptr::null_mut()
}

#[allow(unused_variables)]
unsafe extern "C" fn async_handle(file: *mut iree_hal_file_t) -> *mut iree_async_file_t {
    std::ptr::null_mut()
}

#[allow(unused_variables)]
unsafe extern "C" fn supports_synchronous_io(file: *mut iree_hal_file_t) -> bool {
    true
}

#[allow(unused_variables)]
unsafe extern "C" fn read(
    file: *mut iree_hal_file_t,
    file_offset: u64,
    buffer: *mut iree_hal_buffer_t,
    buffer_offset: iree_device_size_t,
    length: iree_device_size_t,
) -> iree_status_t {
    let f = unsafe { &*cast(file) };
    let src = unsafe { f.data.add(file_offset as usize) };
    unsafe {
        crate::bindings::iree_hal_buffer_map_write(
            buffer,
            buffer_offset,
            src as *const std::ffi::c_void,
            length,
        )
    }
}

#[allow(unused_variables)]
unsafe extern "C" fn write(
    file: *mut iree_hal_file_t,
    file_offset: u64,
    buffer: *mut iree_hal_buffer_t,
    buffer_offset: iree_device_size_t,
    length: iree_device_size_t,
) -> iree_status_t {
    let f = unsafe { &*cast(file) };
    let dst = unsafe { f.data.add(file_offset as usize) };
    unsafe {
        crate::bindings::iree_hal_buffer_map_read(
            buffer,
            buffer_offset,
            dst as *mut std::ffi::c_void,
            length,
        )
    }
}

pub static VTABLE: iree_hal_file_vtable_t = iree_hal_file_vtable_t {
    destroy: Some(destroy),
    allowed_access: Some(allowed_access),
    length: Some(length),
    storage_buffer: Some(storage_buffer),
    async_handle: Some(async_handle),
    supports_synchronous_io: Some(supports_synchronous_io),
    read: Some(read),
    write: Some(write),
};
