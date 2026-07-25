use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=rkt_registers.h");

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

    generate_rocket_accel_bindings(&out_path);
}

// Bindings for the kernel's DRM `rocket` accel driver uapi (GEM BO create/
// prep/fini + job submit ioctls against /dev/accel/accel0). Vendored from
// linux-stable at vendor/LINUX_COMMIT.txt's pinned commit -- see that file
// for provenance. rocket_accel.h unconditionally #includes drm.h, which
// unconditionally #includes drm_mode.h, so both are vendored alongside it
// even though the allowlist below only lets the rocket-specific items
// (plus the one generic drm_gem_close ioctl device.rs also issues) through;
// none of drm_mode.h's KMS/display types are relevant to an NPU-only accel
// driver and letting them through was the fault of the previous manually
// dumped `api.rs`, which had no filtering at all.
fn generate_rocket_accel_bindings(out_path: &PathBuf) {
    println!("cargo:rerun-if-changed=vendor/linux-headers");

    let vendor_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("vendor/linux-headers");
    let target = env::var("TARGET").unwrap();

    let bindings = bindgen::Builder::default()
        .header(
            vendor_dir
                .join("drm/rocket_accel.h")
                .to_str()
                .unwrap()
                .to_string(),
        )
        .clang_arg(format!("-I{}", vendor_dir.display()))
        .clang_arg(format!("--target={target}"))
        // __user is a sparse-only annotation (see <linux/compiler_types.h>
        // in the kernel tree); plain userspace clang doesn't define it, and
        // linux-libc-dev's installed <linux/types.h> doesn't either.
        .clang_arg("-D__user=")
        .allowlist_type("drm_rocket_.*|drm_gem_close|drm_version")
        .allowlist_var("DRM_ROCKET_.*|DRM_COMMAND_BASE|DRM_IOCTL_BASE")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate rocket_accel bindings");

    bindings
        .write_to_file(out_path.join("rocket_accel_bindings.rs"))
        .expect("Couldn't write rocket_accel bindings!");
}
