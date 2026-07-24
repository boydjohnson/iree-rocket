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
| `iree-src` | `iree-org/iree` as a pinned git submodule. |
| `iree-build` | CMake wrapper projects used to configure IREE with the Rocket driver/plugin. |

## Building

Three [CMake presets](CMakePresets.json) cover the three artifacts:

```sh
# iree-compile, with the Rocket compiler target registered
cmake --preset compiler-host && cmake --build --preset compiler-host

# iree-run-module / iree-benchmark-module, host build, with HAL CTS
cmake --preset runtime-host && cmake --build --preset runtime-host

# iree-run-module / iree-benchmark-module, cross-compiled for the RK3588 board
cmake --preset runtime-aarch64 && cmake --build --preset runtime-aarch64
```

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
