//! Generated bindings for the Rocket executable FlatBuffers schema.

#![allow(clippy::all)]
#![allow(dead_code)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(unused_imports)]
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(mismatched_lifetime_syntaxes)]

mod generated {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/generated/rust/rocket_executable_def_generated.rs"
    ));
}

pub use generated::iree::hal::rocket;
