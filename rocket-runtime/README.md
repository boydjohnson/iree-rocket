# rocket-runtime

The iree-rocket runtime as a library, for C, C++ and Rust programs that run
rocket-compiler modules themselves rather than through `iree-run-module`. The
Rust bindings are in [`rust/`](rust).

## Why it exists

Every module `rocket-compiler` produces targets two devices: the RK3588 NPU
(`rocket`) for the operations it claims and the CPU (`local`) for the rest.
IREE's own convenience API, `iree_runtime_session_create_with_device()`, takes
one device, so it cannot load these modules. Without this library, an embedder
would have to rebuild by hand what `iree-run-module --device=rocket
--device=local-task` sets up: both drivers, a device group, the HAL module and
a VM context.

[`include/rocket/runtime.h`](include/rocket/runtime.h) is that setup as one
call, `rocket_runtime_session_create()`. Its functions mirror IREE's
`iree/runtime/session.h` and `call.h`, and everything else stays plain IREE
(buffer views, VM lists, statuses). It works against stock IREE 3.14 and needs
no patch.

Three things it does that `iree-run-module` does through flags, which a library
cannot count on anyone having parsed:

- **CPU worker placement:** the CPU device's workers go on the CPUs with the
  highest `cpu_capacity`, which are cpu4-7 (the A76s) on the RK3588. IREE's
  default placement measured 105 ms against 60-63 ms on MobileNetV2 fp16.
  `ROCKET_RUNTIME_CPU_POLICY_LIST` takes an explicit list, and `..._IREE_DEFAULT`
  restores IREE's own placement.
- **Lead-device allocation:** buffers more than one device touches are
  allocated from the lead device (the NPU), as `--device_lead_allocator=0`
  does. The HAL module's default policy fails inside a module's `__init` with
  an `INTERNAL` from `hal.device.queue.alloca`.
- **CPU-only mode:** `ROCKET_RUNTIME_DEVICES_CPU_ONLY` runs a module compiled
  with `--no-offload` on a machine without the NPU.

## The package

The nightly's `iree-rocket-runtime-aarch64` artifact, or locally:

```sh
cmake --build iree-build/host-aarch64/build --target rocket_runtime_package
cmake --install iree-build/host-aarch64/build --component IreeRocketRuntime \
  --prefix /some/dir --strip
```

| Path | |
|---|---|
| `include/iree-rocket-runtime/rocket/runtime.h`, `include/iree-rocket-runtime/iree/...` | this API and all of IREE's runtime headers, one directory down so they never collide with another IREE's headers |
| `lib/libiree_rocket_runtime.so` | IREE's runtime, the Rocket driver and this API in one file; exports only `iree_*` and `rocket_runtime_*`. The SONAME follows IREE (`.so.314` for IREE 3.14), since every IREE bump changes the `iree_*` ABI |
| `lib/libiree_rocket_runtime.a` | the same contents as one static archive (C and C++ only, see below) |
| `lib/pkgconfig/iree-rocket-runtime{,-static}.pc` | `pkg-config --cflags --libs iree-rocket-runtime` |
| `lib/cmake/IreeRocketRuntime/` | `find_package(IreeRocketRuntime)` with `IreeRocket::runtime` and `IreeRocket::runtime_static` |
| `share/rocket-runtime/samples/` | `embed.cc` (minimal C++) and `run_module.c` (a small `iree-run-module`) |

The package is relocatable. pkg-config and CMake both resolve paths relative to
where it was unpacked, at any `CMAKE_INSTALL_LIBDIR` depth (`lib` here,
`lib/aarch64-linux-gnu` in the Debian packages under [`debian/`](../debian)).

```sh
c++ -std=c++17 embed.cc $(pkg-config --cflags --libs iree-rocket-runtime)
./a.out model.vmfb module.main_graph
```

Prefer the shared library. The static archive contains the Rocket driver's own
copy of the Rust standard library, so a Rust program that links it gets
duplicate std symbols. The `.so`'s version script ([`src/exports.map`](src/exports.map))
hides it, which is also why the [Rust bindings](rust) link the `.so` and never
the archive.

## Compatibility

A module must come from the `rocket-compiler`, and therefore the IREE version,
this library was built with. IREE has no stable HAL or bytecode ABI across
releases; a mismatched module fails to load. `ROCKET_RUNTIME_API_VERSION` and
`rocket_runtime_api_version()` detect a mismatch between this header and the
library.

## Build notes

- **Visibility:** the wrapper projects (`iree-build/host*/CMakeLists.txt`) set
  `IREE_VISIBILITY_HIDDEN=OFF`. IREE otherwise builds `-fvisibility=hidden`,
  and with `IREE_API_EXPORT` empty in C, a shared library would have nothing
  to export.
- **Public define:** consumers need `-DIREE_ALLOCATOR_SYSTEM_CTL=...`, or
  `iree_allocator_system()` does not exist. The `.pc` and CMake files carry it.
- **Unified isn't self-contained:** IREE's `iree_runtime_unified` target still
  links every per-library archive through its CMake interface, and the archive
  itself lacks flatcc's runtime. The package links these files by path
  instead.

## Verified

On the board with ViT-B/16 fp16 (73 NPU sites), `iree-run-module`,
`rocket-run-module` (static and shared) and `embed.cc` built from the unpacked
package (pkg-config shared and static, CMake shared and static) print
byte-identical logits. The Rust bindings' example matches `iree-run-module`
byte for byte on MobileNetV2 and ViT-B/16 with a random input (see
[`rust/`](rust)).
