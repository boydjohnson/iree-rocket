//! Finds the installed IreeRocketRuntime package, generates bindings from its
//! headers and links its shared library.
//!
//! The package is found, in order, at:
//!
//! 1. `ROCKET_RUNTIME_DIR` -- the prefix of an unpacked package (the directory
//!    holding `include/` and `lib/`). The simplest choice when
//!    cross-compiling: no pkg-config configuration needed.
//! 2. pkg-config's `iree-rocket-runtime`. When cross-compiling, the
//!    `pkg-config` crate needs `PKG_CONFIG_ALLOW_CROSS=1` (or a
//!    target-specific `PKG_CONFIG_PATH_<target>`).
//!
//! Bindings are generated against the package's own headers rather than
//! checked in, because IREE has no stable C ABI across releases: the headers
//! that came with the library are the only ones that describe it.

use std::{
    env, fs,
    path::{Path, PathBuf},
};

struct Package {
    include_dirs: Vec<PathBuf>,
    lib_dir: PathBuf,
    /// `NAME=VALUE` compile definitions the headers need from their consumer.
    defines: Vec<(String, Option<String>)>,
}

fn main() {
    println!("cargo:rerun-if-env-changed=ROCKET_RUNTIME_DIR");
    println!("cargo:rerun-if-changed=wrapper.h");

    let package = match env::var_os("ROCKET_RUNTIME_DIR") {
        Some(dir) => from_prefix(Path::new(&dir)),
        None => from_pkg_config(),
    };

    // Dynamic only. The package's static archive carries the Rocket driver's
    // own copy of the Rust standard library, which would collide with this
    // program's; the .so hides it behind its version script.
    println!(
        "cargo:rustc-link-search=native={}",
        package.lib_dir.display()
    );
    println!("cargo:rustc-link-lib=dylib=iree_rocket_runtime");
    // Read by dependents' build scripts as DEP_IREE_ROCKET_RUNTIME_LIBDIR, so
    // their tests and examples can find the .so without LD_LIBRARY_PATH.
    println!("cargo:libdir={}", package.lib_dir.display());

    let mut builder = bindgen::Builder::default()
        .header("wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .allowlist_function("(iree|rocket_runtime)_.*")
        .allowlist_type("(iree|rocket_runtime)_.*")
        .allowlist_var("(IREE|ROCKET_RUNTIME)_.*")
        .derive_default(true)
        .generate_comments(false)
        .layout_tests(false);
    for dir in &package.include_dirs {
        builder = builder.clang_arg(format!("-I{}", dir.display()));
    }
    for (name, value) in &package.defines {
        builder = builder.clang_arg(match value {
            Some(value) => format!("-D{name}={value}"),
            None => format!("-D{name}"),
        });
    }
    let bindings = builder
        .generate()
        .expect("generating bindings from rocket/runtime.h");
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("writing bindings.rs");

    // iree_allocator_system() is `static inline` in C, so bindgen cannot bind
    // it. It is two fields, one of which names the allocator the library was
    // built with -- the same IREE_ALLOCATOR_SYSTEM_CTL definition the headers
    // need -- so it is generated here from that definition.
    let ctl = package
        .defines
        .iter()
        .find(|(name, _)| name == "IREE_ALLOCATOR_SYSTEM_CTL")
        .and_then(|(_, value)| value.clone())
        .expect(
            "the package defines no IREE_ALLOCATOR_SYSTEM_CTL; its pkg-config \
             Cflags must carry -DIREE_ALLOCATOR_SYSTEM_CTL=<allocator ctl>",
        );
    fs::write(
        out_dir.join("allocator_system.rs"),
        format!(
            "/// The allocator IREE was built with (`iree_allocator_system()` in C).\n\
             pub fn iree_allocator_system() -> iree_allocator_t {{\n    \
                 iree_allocator_t {{ self_: ::core::ptr::null_mut(), ctl: Some({ctl}) }}\n\
             }}\n"
        ),
    )
    .expect("writing allocator_system.rs");
}

/// An unpacked package: `include/` and `lib/` under one prefix. The
/// compile definitions come from the package's own .pc file, read directly so
/// no pkg-config binary or configuration is needed.
fn from_prefix(prefix: &Path) -> Package {
    let include = prefix.join("include");
    let lib_dir = prefix.join("lib");
    let pc = lib_dir.join("pkgconfig/iree-rocket-runtime.pc");
    assert!(
        include.join("rocket/runtime.h").is_file(),
        "ROCKET_RUNTIME_DIR={} has no include/rocket/runtime.h; point it at an \
         unpacked IreeRocketRuntime package (the directory holding include/ and lib/)",
        prefix.display()
    );
    println!("cargo:rerun-if-changed={}", pc.display());
    let pc_text =
        fs::read_to_string(&pc).unwrap_or_else(|e| panic!("reading {}: {e}", pc.display()));
    let cflags = pc_text
        .lines()
        .find_map(|line| line.strip_prefix("Cflags:"))
        .unwrap_or("");
    Package {
        include_dirs: vec![include],
        lib_dir,
        defines: cflags
            .split_whitespace()
            .filter_map(|flag| flag.strip_prefix("-D"))
            .map(split_define)
            .collect(),
    }
}

fn from_pkg_config() -> Package {
    let library = pkg_config::Config::new()
        // Link directives are emitted above, dynamic only; this is only asked
        // for paths and definitions.
        .cargo_metadata(false)
        .probe("iree-rocket-runtime")
        .unwrap_or_else(|e| {
            panic!(
                "the IreeRocketRuntime package was not found. Set ROCKET_RUNTIME_DIR \
                 to an unpacked package, or put its lib/pkgconfig on PKG_CONFIG_PATH \
                 (with PKG_CONFIG_ALLOW_CROSS=1 when cross-compiling).\n{e}"
            )
        });
    Package {
        include_dirs: library.include_paths,
        lib_dir: library
            .link_paths
            .into_iter()
            .next()
            .expect("iree-rocket-runtime.pc names no library directory"),
        defines: library.defines.into_iter().collect(),
    }
}

fn split_define(define: &str) -> (String, Option<String>) {
    match define.split_once('=') {
        Some((name, value)) => (name.to_owned(), Some(value.to_owned())),
        None => (define.to_owned(), None),
    }
}
