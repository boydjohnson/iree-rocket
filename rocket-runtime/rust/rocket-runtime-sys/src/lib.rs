//! Raw bindings to `libiree_rocket_runtime.so`: `rocket/runtime.h` and the
//! IREE runtime C API it is built on, generated from the installed package's
//! headers. Use the `rocket-runtime` crate unless you need something it does
//! not wrap.
//!
//! A few IREE helpers are `static inline` or macros in C and so have no
//! symbol to bind; the ones a caller needs are reimplemented at the bottom of
//! this file.

#![allow(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    dead_code,
    // bindgen's generated array helpers predate the 2024 edition's lint.
    unsafe_op_in_unsafe_fn,
    clippy::all
)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
include!(concat!(env!("OUT_DIR"), "/allocator_system.rs"));

/// The low bits of an `iree_status_t` that hold its code (`IREE_STATUS_CODE_MASK`).
pub const IREE_STATUS_CODE_MASK_BITS: usize = 0x1F;

/// `iree_status_code()`: the code of a status, `IREE_STATUS_OK` (0) for a null one.
#[inline]
pub fn iree_status_code(status: iree_status_t) -> iree_status_code_t {
    ((status as usize) & IREE_STATUS_CODE_MASK_BITS) as iree_status_code_t
}

/// `iree_status_is_ok()`.
#[inline]
pub fn iree_status_is_ok(status: iree_status_t) -> bool {
    status.is_null()
}

/// `iree_status_from_code()`: a status carrying only a code, no message or
/// storage, as an allocator control function returns.
#[inline]
pub fn iree_status_from_code(code: iree_status_code_t) -> iree_status_t {
    ((code as usize) & IREE_STATUS_CODE_MASK_BITS) as iree_status_t
}

/// `iree_make_string_view()` over a Rust string. Borrowed: valid only while
/// `value` is.
#[inline]
pub fn iree_make_string_view(value: &str) -> iree_string_view_t {
    iree_string_view_t {
        data: value.as_ptr().cast(),
        size: value.len() as _,
    }
}

/// `iree_make_const_byte_span()` over a Rust slice. Borrowed: valid only while
/// `bytes` is.
#[inline]
pub fn iree_make_const_byte_span(bytes: &[u8]) -> iree_const_byte_span_t {
    iree_const_byte_span_t {
        data: bytes.as_ptr(),
        data_length: bytes.len() as _,
    }
}
