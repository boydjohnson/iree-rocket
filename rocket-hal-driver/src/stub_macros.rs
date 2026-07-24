//! Generates vtable-slot stub functions from a signature, matching the
//! `null` HAL driver's philosophy (`iree-null-driver-reference/README.md`
//! in rknpu-spelunking): every vtable slot is a real, correctly-typed
//! function -- never `None` -- but the body is a placeholder until wired
//! up to real logic. `status_stub!` covers the vast majority of vtable
//! slots (`-> iree_status_t`); `void_stub!`/`bool_stub!` cover the handful
//! that return something else.

#[macro_export]
macro_rules! status_stub {
    ($name:ident($($arg:ident: $ty:ty),* $(,)?) -> iree_status_t) => {
        #[allow(unused_variables)]
        pub unsafe extern "C" fn $name($($arg: $ty),*) -> $crate::bindings::iree_status_t {
            $crate::status::unimplemented()
        }
    };
}

#[macro_export]
macro_rules! void_stub {
    ($name:ident($($arg:ident: $ty:ty),* $(,)?)) => {
        #[allow(unused_variables)]
        pub unsafe extern "C" fn $name($($arg: $ty),*) {}
    };
}

#[macro_export]
macro_rules! bool_stub {
    ($name:ident($($arg:ident: $ty:ty),* $(,)?) -> bool, $val:expr) => {
        #[allow(unused_variables)]
        pub unsafe extern "C" fn $name($($arg: $ty),*) -> bool {
            $val
        }
    };
}
