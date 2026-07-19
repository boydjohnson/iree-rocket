//! `iree_hal_buffer_vtable_t`. Small vtable, maps directly onto our
//! existing mmap'd GEM BO wrapper (`iree-rocket-hal`'s `Buffer`):
//! `map_range`/`unmap_range` are just handing out the already-mmap'd
//! `host_ptr`, and `flush_range`/`invalidate_range` are where
//! `DRM_ROCKET_PREP_BO`/`FINI_BO`-style CPU/GPU cache sync would go if
//! this hardware needs explicit sync (TODO: check -- rkt-basic.rs's
//! `Buffer` never needed to for raw GEM BOs, but confirm before assuming
//! these can stay no-ops).

use crate::bindings::{
    iree_device_size_t, iree_hal_buffer_mapping_t, iree_hal_buffer_t, iree_hal_buffer_vtable_t,
    iree_hal_mapping_mode_t, iree_hal_memory_access_t,
};

void_stub!(recycle(buffer: *mut iree_hal_buffer_t));
void_stub!(destroy(buffer: *mut iree_hal_buffer_t));

// TODO: the real one -- point mapping->contents at the Buffer's existing
// host_ptr (already mmap'd by CREATE_BO; no separate map step needed).
status_stub!(map_range(
    buffer: *mut iree_hal_buffer_t,
    mapping_mode: iree_hal_mapping_mode_t,
    memory_access: iree_hal_memory_access_t,
    local_byte_offset: iree_device_size_t,
    local_byte_length: iree_device_size_t,
    mapping: *mut iree_hal_buffer_mapping_t,
) -> iree_status_t);

status_stub!(unmap_range(
    buffer: *mut iree_hal_buffer_t,
    local_byte_offset: iree_device_size_t,
    local_byte_length: iree_device_size_t,
    mapping: *mut iree_hal_buffer_mapping_t,
) -> iree_status_t);

status_stub!(invalidate_range(
    buffer: *mut iree_hal_buffer_t,
    local_byte_offset: iree_device_size_t,
    local_byte_length: iree_device_size_t,
) -> iree_status_t);

status_stub!(flush_range(
    buffer: *mut iree_hal_buffer_t,
    local_byte_offset: iree_device_size_t,
    local_byte_length: iree_device_size_t,
) -> iree_status_t);

pub static VTABLE: iree_hal_buffer_vtable_t = iree_hal_buffer_vtable_t {
    recycle: Some(recycle),
    destroy: Some(destroy),
    map_range: Some(map_range),
    unmap_range: Some(unmap_range),
    invalidate_range: Some(invalidate_range),
    flush_range: Some(flush_range),
};
