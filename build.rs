use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=vendor/iree-headers");

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let iree_headers = manifest_dir.join("vendor/iree-headers");
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());

    // iree/hal/api.h is IREE's umbrella header -- pulls in every public HAL
    // type (driver/device/allocator/buffer/command_buffer/semaphore/
    // executable/...) plus iree/base/api.h transitively. Confirmed to
    // preprocess standalone against just this vendored subset (base, hal,
    // async, io, schemas) with no CMake-generated config headers involved
    // -- see vendor/IREE_COMMIT.txt for the pinned upstream commit.
    let bindings = bindgen::Builder::default()
        .header(iree_headers.join("iree/hal/api.h").to_str().unwrap())
        .clang_arg(format!("-I{}", iree_headers.display()))
        .allowlist_file(".*/iree/(hal|base|async|io|schemas)/.*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
