//! Run rocket-compiler modules on the RK3588 NPU from Rust.
//!
//! A safe wrapper over `libiree_rocket_runtime.so` (`rocket/runtime.h`): a
//! [`Session`] holds the NPU and the CPU as one IREE device group, which every
//! module rocket-compiler produces needs, and IREE's own single-device
//! session API cannot provide.
//!
//! ```no_run
//! use rocket_runtime::{BufferView, ElementType, Instance, Session, SessionOptions};
//!
//! # fn main() -> rocket_runtime::Result<()> {
//! let instance = Instance::new()?;
//! let mut session = Session::new(&instance, &SessionOptions::default())?;
//! session.load_module_file("mobilenet_v2.fp16.npu.vmfb")?;
//!
//! // f16 without the `half` feature: raw little-endian bytes (0x2E66 = 0.1).
//! let pixels: Vec<u8> = [0x66, 0x2E].repeat(3 * 224 * 224);
//! let input = BufferView::from_bytes(&session, &[1, 3, 224, 224], ElementType::Float16, &pixels)?;
//!
//! let outputs = session.call("module.main_graph", &[&input])?;
//! println!("{:?}", outputs[0]);
//! # Ok(())
//! # }
//! ```
//!
//! The library is linked dynamically and found at build time through
//! `ROCKET_RUNTIME_DIR` or pkg-config (see the `rocket-runtime-sys` crate).
//! A module must come from the rocket-compiler, and so the IREE version, the
//! library was built with.

// IREE's width typedefs (iree_device_size_t, iree_hal_dim_t, its enum types)
// depend on the target and on IREE's build configuration; the casts that are
// no-ops on 64-bit Linux keep the code correct where they are not.
#![allow(clippy::unnecessary_cast)]

mod buffer;
mod error;
mod session;

pub use buffer::{BufferView, Element, ElementType};
pub use error::{Error, Result, StatusCode};
pub use session::{CpuPlacement, Devices, Function, Instance, Session, SessionOptions};

/// The raw bindings, for anything this crate does not wrap.
pub use rocket_runtime_sys as sys;

/// `ROCKET_RUNTIME_API_VERSION` as the library was compiled, which should
/// equal the version these bindings were generated from
/// ([`sys::ROCKET_RUNTIME_API_VERSION`]).
pub fn api_version() -> u32 {
    // SAFETY: no arguments, no state.
    unsafe { sys::rocket_runtime_api_version() }
}
