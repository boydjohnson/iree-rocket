//! Pure planning for the RK3588 NPU: what the compiler and the runtime both
//! need to know about a convolution or matmul before either emits anything.
//!
//! This crate deliberately depends on nothing. It builds and tests on the
//! host without a device, DRM, IREE, MLIR or the register builders, which
//! is what lets the compiler plugin ask it the same question the HAL asks
//! at dispatch time (COMPILER_ROADMAP.md section 1). DMA addresses,
//! relocation, register programs, buffer allocation and submission stay in
//! `iree-rocket-hal`, which consumes the plans made here.

pub mod admission;
pub mod conv;
pub mod error;
pub mod fc;
pub mod layout;
pub mod policy;
pub mod weights;
