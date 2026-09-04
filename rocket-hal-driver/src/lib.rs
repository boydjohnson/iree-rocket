//! Out-of-tree IREE HAL driver for the Rockchip RK3588 NPU (`accel/rocket`
//! mainline kernel driver, `/dev/accel/accel0`). Skeleton scaffolding
//! stage: every vtable slot listed in `iree/hal/api.h` is a real,
//! correctly-typed function (never `None`), modeled directly on IREE's own
//! `iree/hal/drivers/null/` skeleton driver (mirrored for reference at
//! rknpu-spelunking/iree-null-driver-reference/) -- but nearly all of them
//! return `IREE_STATUS_UNIMPLEMENTED` for now. See each module's doc
//! comment for what's actually planned to replace its stubs, and
//! rknpu-spelunking/NOTES.md for the research this crate started from.
//!
//! `[lib] crate-type = ["staticlib", "rlib"]` in Cargo.toml -- confirmed
//! from IREE's own source that `iree_register_external_hal_driver` (CMake)
//! links a static library target directly, no dlopen involved (see
//! Cargo.toml's comment for the exact files checked). The `rlib` half is
//! for `cargo check`/internal use without the CMake side built.
//!
//! `iree_hal_rocket_driver_module_register` below is this crate's real,
//! `#[no_mangle]` C ABI entry point -- what
//! `IREE_EXTERNAL_ROCKET_HAL_DRIVER_REGISTER` will name once the CMake
//! wiring exists.

#![allow(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    dead_code
)]

pub mod bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

#[macro_use]
mod stub_macros;

pub mod status;

pub mod allocator;
pub mod buffer;
pub mod command_buffer;
pub mod cpu_affinity;
pub mod device;
pub mod driver;
pub mod event;
pub mod executable;
pub mod executable_cache;
pub mod file;
pub mod profile;
pub mod semaphore;
pub mod weight_cache;

/// Mirrors `iree_hal_null_driver_module_register()`'s C ABI signature
/// exactly -- the name a CMake `IREE_EXTERNAL_ROCKET_HAL_DRIVER_REGISTER`
/// property will reference once the CMake side (see rocket-hal-driver's
/// task list / rknpu-spelunking/NOTES.md) is wired up.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iree_hal_rocket_driver_module_register(
    registry: *mut bindings::iree_hal_driver_registry_t,
) -> bindings::iree_status_t {
    unsafe { driver::register(registry) }
}
