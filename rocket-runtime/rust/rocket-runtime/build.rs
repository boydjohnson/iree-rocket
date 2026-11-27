//! Lets this crate's own tests and examples find libiree_rocket_runtime.so
//! without LD_LIBRARY_PATH: an rpath to the directory rocket-runtime-sys
//! linked it from. Programs that depend on this crate are not affected; they
//! ship the .so as they see fit (beside the binary with an $ORIGIN rpath, or
//! on the loader path).
fn main() {
    if let Some(libdir) = std::env::var_os("DEP_IREE_ROCKET_RUNTIME_LIBDIR") {
        let libdir = libdir.to_string_lossy();
        println!("cargo:rustc-link-arg-tests=-Wl,-rpath,{libdir}");
        println!("cargo:rustc-link-arg-examples=-Wl,-rpath,{libdir}");
    }
}
