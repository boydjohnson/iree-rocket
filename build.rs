use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=rkt_registers.hpp");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());

    // 2. Set up the bindgen builder
    let bindings = bindgen::Builder::default()
        .header("rkt_registers.h")
        .wrap_static_fns(true)
        .wrap_static_fns_path(out_path.join("wrap_static_fns"))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    // 3. Write the bindings to the $OUT_DIR/bindings.rs file.
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");

    cc::Build::new()
        .file(out_path.join("wrap_static_fns.c"))
        .includes([env!("CARGO_MANIFEST_DIR")])
        .compile("wrap_static_fns");

    println!("cargo:link-lib=static=wrap_static_fns");
}
