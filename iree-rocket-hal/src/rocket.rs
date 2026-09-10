#[allow(
    non_upper_case_globals,
    non_camel_case_types,
    unsafe_op_in_unsafe_fn,
    clippy::all,
    clippy::pedantic
)]
pub mod api;
pub mod builders;
pub mod conv;
pub mod debug;
pub mod device;
pub mod executable_format;
pub mod fc;
pub mod lut_tables;
pub mod regcmd;

// One module per hardware op, split out of what used to be a single
// `regcmd` grab-bag. `regcmd` itself now holds only what all of them
// share (see its module doc comment).
pub mod activation;
pub mod elementwise;
pub mod pooling;
#[allow(
    non_upper_case_globals,
    non_camel_case_types,
    unsafe_op_in_unsafe_fn,
    clippy::all,
    clippy::pedantic
)]
pub mod registers;
pub mod tensor_layout;

/// The feature-cube layout contract (COMPILER_ROADMAP.md 6.1), re-exported
/// from `rocket-core` so the driver reads it from one place.
pub use rocket_core::layout;

/// The coefficient layout and its packers (COMPILER_ROADMAP.md 6.3),
/// re-exported from `rocket-core`; `tensor_layout` re-exports the packers
/// under their old names, this is the `WeightPlan` that selects them.
pub use rocket_core::weights;
