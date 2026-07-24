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
    // iree/async/util/proactor_pool.h isn't pulled in transitively by
    // iree/hal/api.h (device_create_params_t only needs the opaque
    // iree_async_proactor_pool_t forward-decl for that), but
    // device::create needs its real functions (pool_retain/get/release)
    // to obtain a proactor for semaphore creation -- see semaphore.rs's
    // module doc comment for why that's needed at all.
    // bindgen defaults to the HOST clang target, which is wrong when cross-
    // compiling (e.g. `cargo build --target aarch64-unknown-linux-gnu` from
    // this x86_64 dev machine, for the actual RK3588 board) -- struct
    // layouts/padding/alignment must be computed for the TARGET, not the
    // build host. Cargo always sets TARGET to the real `--target` triple
    // (host or cross) during build.rs, so pass it straight through as
    // clang's `--target`.
    let target = env::var("TARGET").unwrap();

    let bindings = bindgen::Builder::default()
        .header(iree_headers.join("iree/hal/api.h").to_str().unwrap())
        .header(
            iree_headers
                .join("iree/async/util/proactor_pool.h")
                .to_str()
                .unwrap(),
        )
        .clang_arg(format!("-I{}", iree_headers.display()))
        .clang_arg(format!("--target={target}"))
        .allowlist_file(".*/iree/(hal|base|async|io|schemas)/.*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
