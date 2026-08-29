use std::{env, path::PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let include_dir = manifest_dir.join("vendor/iree-headers");
    let header = include_dir.join("iree/compiler/embedding_api.h");

    println!("cargo:rerun-if-changed={}", header.display());

    // Dynamic loading (not link-time linking): libIREECompiler.so's path is only
    // known at runtime (this repo has multiple build dirs -- host, host-aarch64,
    // build -- with no single correct default), so bindgen emits a struct that
    // dlopens the library and resolves every symbol in its ::new() constructor
    // instead of generating `extern "C"` declarations for the linker to resolve.
    let bindings = bindgen::Builder::default()
        .header(header.to_str().unwrap())
        .clang_arg(format!("-I{}", include_dir.display()))
        .dynamic_library_name("IREECompilerApi")
        .dynamic_link_require_all(true)
        .allowlist_function("ireeCompiler.*")
        .allowlist_type("iree_compiler_.*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("failed to generate IREE compiler embedding API bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("failed to write bindings.rs");
}
