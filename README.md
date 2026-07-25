# iree-rocket

Mono-repo for building [IREE](https://github.com/iree-org/iree) with support
for the Rocket NPU backend (RK3588). This repository produces:

- `iree-compile` — with the Rocket compiler target registered.
- `iree-run-module` / `iree-benchmark-module` — with the Rocket HAL driver
  statically linked in.

## Layout

| Path | Role |
|---|---|
| [`rocket-schema`](rocket-schema) | Canonical FlatBuffers schema for Rocket executables; shared by the compiler plugin (C++) and the runtime crates (Rust). |
| [`iree-rocket-hal`](iree-rocket-hal) | Low-level Rust crate: ioctl/mmap access to the RK3588 NPU and register command building. |
| [`rocket-hal-driver`](rocket-hal-driver) | Rust `staticlib` implementing IREE's HAL driver interface, statically linked into IREE via `iree_register_external_hal_driver()`. Depends on `iree-rocket-hal` and `rocket-schema`. Includes HAL CTS wiring under `cts/`. |
| [`rocket-compiler-plugin`](rocket-compiler-plugin) | C++ IREE compiler target plugin ("Rocket"), loaded via `IREE_CMAKE_PLUGIN_PATHS`. Serializes executables using `rocket-schema`'s FlatBuffer format. |
| `iree-build/iree-src` | `iree-org/iree` as a pinned git submodule. |
| `iree-build` | CMake configuration used to build IREE with the Rocket driver/plugin. |

## Building

Each of the three artifacts has its own build directory under `iree-build/`,
since each configures a different CMake source root (the vendored `iree-src`
directly, vs. one of two small wrapper projects that register the Rocket HAL
driver before IREE's own `add_subdirectory` runs):

```sh
# iree-compile, with the Rocket compiler target registered
./iree-build/configure-compiler-host.sh
cmake --build iree-build/build

# iree-run-module / iree-benchmark-module, host build, with HAL CTS
(cd iree-build/host && cmake --preset runtime-host && cmake --build --preset runtime-host)

# iree-run-module / iree-benchmark-module, cross-compiled for the RK3588 board.
# Build compiler-host first: this preset points IREE_HOST_BIN_DIR at
# iree-build/build/tools for codegen tools (e.g. iree-c-embed-data) that
# IREE's build runs on the host even when cross-compiling the runtime.
(cd iree-build/host-aarch64 && cmake --preset runtime-aarch64 && cmake --build --preset runtime-aarch64)
```

`iree-build/host/CMakePresets.json` and `iree-build/host-aarch64/CMakePresets.json`
each live next to the wrapper `CMakeLists.txt` they configure (CMake presets
are always rooted at the directory containing the `CMakeLists.txt`, so a
single repo-root `CMakePresets.json` can't span these plus the vendored
`iree-src` tree). `configure-compiler-host.sh` covers the third case, which
has no wrapper project at all.

The Rust crates (`rocket-schema`, `iree-rocket-hal`, `rocket-hal-driver`) form
a single Cargo workspace and can be built/checked independently of the CMake
builds above:

```sh
cargo build --workspace
cargo test --workspace
```

## Submodules

After cloning, initialize `iree-src` (and its own third-party submodules):

```sh
git submodule update --init --recursive
```
