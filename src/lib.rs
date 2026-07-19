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
//! `[lib] crate-type = ["rlib"]` in Cargo.toml, not yet `cdylib` -- turning
//! this into an actually loadable HAL driver plugin needs IREE's own
//! compiled runtime libraries (libiree_hal, libiree_base, ...) to exist
//! and be linkable, which requires building IREE's runtime from source
//! (CMake/ninja) -- a separate, substantial milestone not done yet. Until
//! then this only needs to type-check against the vendored headers
//! (`vendor/iree-headers/`, pinned to the commit in
//! `vendor/IREE_COMMIT.txt`).

#![allow(non_upper_case_globals, non_camel_case_types, non_snake_case, dead_code)]

pub mod bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

#[macro_use]
mod stub_macros;

pub mod status;

pub mod allocator;
pub mod buffer;
pub mod command_buffer;
pub mod device;
pub mod driver;
pub mod executable;
pub mod executable_cache;
pub mod semaphore;
