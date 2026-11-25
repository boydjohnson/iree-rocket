# iree-rocket

[![CI](https://github.com/boydjohnson/iree-rocket/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/boydjohnson/iree-rocket/actions/workflows/ci.yml)
[![Nightly artifacts](https://github.com/boydjohnson/iree-rocket/actions/workflows/nightly.yml/badge.svg?branch=main)](https://github.com/boydjohnson/iree-rocket/actions/workflows/nightly.yml)

Mono-repo for building [IREE](https://github.com/iree-org/iree) with support
for the Rocket NPU backend (RK3588). This repository produces:

- `iree-compile` — with the Rocket compiler target registered.
- `iree-run-module` / `iree-benchmark-module` — with the Rocket HAL driver
  statically linked in.

## Layout

| Path | Role |
|---|---|
| [`rocket-schema`](rocket-schema) | Canonical FlatBuffers schema for Rocket executables; shared by the compiler plugin (C++) and the runtime crates (Rust). |
| [`rocket-plan-ffi`](rocket-plan-ffi) | Versioned C ABI (`include/rocket_plan.h`) over `rocket-core`'s planner, layout contract and weight packers, built as a host staticlib by the compiler plugin's CMake and linked into `libIREECompiler.so`; how the compiler asks the same planner -- and packs the same coefficient bytes -- the runtime does. |
| [`rocket-core`](rocket-core) | Pure, dependency-free Rust crate: convolution/matmul descriptors, hardware limits, layout geometry, CBUF partitioning and tile planning, with fallible `try_*` APIs that return a `PlanError` code instead of panicking. Shared by the runtime and, eventually, the compiler. |
| [`iree-rocket-hal`](iree-rocket-hal) | Low-level Rust crate: ioctl/mmap access to the RK3588 NPU and register command building. Consumes `rocket-core`'s plans and re-exports its planner under `rocket::conv`. |
| [`rocket-hal-driver`](rocket-hal-driver) | Rust `staticlib` implementing IREE's HAL driver interface, statically linked into IREE via `iree_register_external_hal_driver()`. Depends on `iree-rocket-hal` and `rocket-schema`. Includes HAL CTS wiring under `cts/`. |
| [`rocket-compiler-plugin`](rocket-compiler-plugin) | C++ IREE compiler target plugin ("Rocket"), loaded via `IREE_CMAKE_PLUGIN_PATHS`. Serializes executables using `rocket-schema`'s FlatBuffer format. |
| [`rocket-compiler`](rocket-compiler) | Rust driver over `libIREECompiler.so`. Applies the Rocket transform spec and the device flags it expects, and can audit why each convolution and matmul is where it is. |
| [`tools`](tools) | Python harnesses: ONNX import/export, the board regression gates, the whole-model survey, the precision-transition probe. |
| `iree-build/iree-src` | `iree-org/iree` as a pinned git submodule. |
| `iree-build` | CMake configuration used to build IREE with the Rocket driver/plugin. |

## Documents

| Document | What it holds |
|---|---|
| [`agent-docs/developer-tools.md`](agent-docs/developer-tools.md) | **Start here to work on the repo.** How to build the three configurations, compile a model with `rocket-compiler`, read an `audit`, every flag that changes placement, `ROCKET_PROFILE` and the runtime's environment knobs, the board regression gates, the model survey and the precision probe. |
| [`LIMITS.md`](LIMITS.md) | What the stack is *measured* to do: the channel, kernel, stride and precision bounds, and which of them each layer enforces. |
| [`ISSUES.md`](ISSUES.md) | What is still open, with the measurements behind each entry. |
| [`ROADMAP.md`](ROADMAP.md) | Which MLIR operations the hardware could run but the compiler cannot yet reach. |
| [`COMPILER_ROADMAP.md`](COMPILER_ROADMAP.md) | The compiler-side plan; the section numbers other documents cite. |
| [`DYNAMIC_SHAPES.md`](DYNAMIC_SHAPES.md), [`SOFTMAX.md`](SOFTMAX.md) | Single-topic design notes. |
| [`rocket-hal-driver/MULTICORE.md`](rocket-hal-driver/MULTICORE.md) | The multi-context worker design and its measurements. |
| [`rocket-schema/README.md`](rocket-schema/README.md) | The executable format and its compatibility rules. |
| [`agent-docs/`](agent-docs) | Working notes: open questions, branch audits, dated survey results. |

## Getting started

```sh
git submodule update --init --recursive   # iree-src and its own submodules
cargo test --workspace                    # the Rust side, no IREE build needed
```

The CMake builds, the compiler driver and the board workflow are all in
[`agent-docs/developer-tools.md`](agent-docs/developer-tools.md). To compile a
model without building IREE at all, take the `iree-compile-x86_64` artifact
from the latest [nightly run](https://github.com/boydjohnson/iree-rocket/actions/workflows/nightly.yml)
-- it unpacks anywhere and needs no environment setup
([details](agent-docs/developer-tools.md#prebuilt-compiler-nightly)).

## Continuous integration

Three workflows, under [`.github/workflows/`](.github/workflows). The badges
are live -- each shows its workflow's latest run on `main`; the reusable one
has none of its own, for the reason in the table.

| Workflow | Status | Trigger | Jobs | Typical runtime |
|---|---|---|---|---|
| [`ci.yml`](.github/workflows/ci.yml) -- **CI** | [![CI](https://github.com/boydjohnson/iree-rocket/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/boydjohnson/iree-rocket/actions/workflows/ci.yml) | every push to `main`, every pull request | `check, build, test`; `cargo fmt --check` | ~1.5 min |
| [`nightly.yml`](.github/workflows/nightly.yml) -- **Nightly artifacts** | [![Nightly artifacts](https://github.com/boydjohnson/iree-rocket/actions/workflows/nightly.yml/badge.svg?branch=main)](https://github.com/boydjohnson/iree-rocket/actions/workflows/nightly.yml) | `cron 0 7 * * *` (07:00 UTC), `workflow_dispatch` | `compiler-host`, then `runtime-aarch64` | ~20 min warm ccache, up to ~4 h cold |
| [`iree-build.yml`](.github/workflows/iree-build.yml) -- **IREE build (reusable)** | no badge: `workflow_call` only, so its runs are attributed to the caller | called by `nightly.yml` | one `build` job per `target` input | see above |

### CI -- `ci.yml`

The gate on every pull request. Rust only: it never configures CMake and never
builds IREE, so it cannot catch a C++ plugin or CMake break -- that is the
nightly's job.

- **`check, build, test`** (`ubuntu-latest`) -- `cargo check --workspace
  --all-targets`, then `cargo build --workspace`, then `cargo test
  --workspace` over the whole Cargo workspace (`rocket-core`,
  `rocket-plan-ffi`, `rocket-schema`, `iree-rocket-hal`, `rocket-hal-driver`,
  `rocket-compiler`). Installs `libclang-dev` first, because
  `rocket-hal-driver` and `iree-rocket-hal` both run bindgen over C headers
  (`vendor/iree-headers`, `rkt_registers.h`) from `build.rs`. Stable Rust,
  with `Swatinem/rust-cache`. The hardware tests -- the ones that open
  `/dev/accel/accel0` -- are `#[ignore]`d, so this covers the host-safe unit
  tests only; the real NPU gates are run by hand against a board
  ([developer tools](agent-docs/developer-tools.md#board-convolution-regression-gate)).
- **`cargo fmt --check`** (`ubuntu-latest`) -- `cargo fmt --all -- --check` on
  **nightly** rustfmt, not stable: `rustfmt.toml` sets `imports_granularity`,
  which is still unstable, and stable rustfmt would silently ignore it and
  disagree with how the code is actually formatted.

There is no clippy job and no lit-test job.

### Nightly artifacts -- `nightly.yml`

Builds the three things this repo exists to produce and attaches them to the
run. Both jobs delegate to `iree-build.yml`; `runtime-aarch64` `needs:`
`compiler-host`, because the aarch64 configuration points `IREE_HOST_BIN_DIR`
at the host build's `tools/` for the codegen tools IREE runs on the host even
when cross-compiling. Runs overlapping an earlier nightly cancel it
(`concurrency: cancel-in-progress`) rather than fight it for the shared
ccache. `workflow_dispatch` takes a `retention-days` input, default 14.

- **`compiler-host`** -- configures the vendored `iree-src` with the Rocket
  compiler target plugin (`iree-build/configure-compiler-host.sh`, Release,
  `IREE_BUILD_TESTS=OFF`), builds it, then builds `rocket-compiler`
  separately with cargo (it `dlopen`s `libIREECompiler.so` rather than linking
  it). Packaging rewrites each tool's RUNPATH to `$ORIGIN/../lib` with
  `patchelf`, so the tarball works unpacked anywhere, and then **smoke-tests
  the packaged binary the way a user gets it**: untarred in a scratch
  directory with `env -i`, compiling a model both with and without
  `--no-offload`, with the checkout's own copy of the transform spec moved
  aside so a broken packaged lookup cannot hide behind it. Uploads
  `iree-compile-x86_64` (`bin/rocket-compiler`, `bin/iree-compile`,
  `bin/iree-lld`, `lib/libIREECompiler.so`, `share/rocket/<spec>`) and
  `iree-host-tools` (the full host `tools/` dir, for the next job). Default
  timeout 330 min -- deliberately under GitHub's 6 h ceiling, so a build that
  runs long still reaches its ccache save step.
- **`runtime-aarch64`** -- unpacks `iree-host-tools` back into
  `iree-build/build/`, installs the `gcc`/`g++-aarch64-linux-gnu` cross
  toolchain, configures with `iree-build/configure-runtime-aarch64.sh`
  (Release, `IREE_ENABLE_WERROR_FLAG=OFF`, tests **on** -- that is what builds
  the CTS) and builds `iree-run-module` / `iree-benchmark-module` with the
  Rocket HAL driver statically linked. Then `cargo test --workspace --release
  --target aarch64-unknown-linux-gnu --no-run` for the crates' own libtest
  binaries, which the CMake build never produces. Uploads
  `iree-runtime-aarch64`, `rocket-cts-aarch64` (the non-executable HAL CTS
  categories) and `rocket-rust-tests-aarch64`, the last two with a `RUN.md`
  saying how to run them on the board. Timeout 120 min: no LLVM here.

A nightly that fails with *"the hosted runner lost communication with the
server"* and no failed step lost its runner mid-build (the LLVM compile is the
usual place); it is not a repo break. Re-run it -- the retry restores the last
successful run's ccache, so it is warm.

Every artifact is a single `.tar.gz` -- `actions/upload-artifact` zips its
input and the zip round-trip drops the executable bit. Each carries a
`VERSION.txt` naming the repo SHA, the `iree-src` pin and the run URL.

### IREE build (reusable) -- `iree-build.yml`

One `build` job, parameterized by `target` (`compiler-host` or
`runtime-aarch64`); an unknown value fails the job up front, since every build
step is gated on it and a typo would otherwise produce a green job that built
nothing. Inputs worth knowing about:

| Input | Default | Why |
|---|---|---|
| `runner` | `ubuntu-24.04` | Pinned, never `ubuntu-latest`: the image's glibc becomes the floor of every binary built on it, and `ubuntu-latest` moved to 26.04 (glibc 2.43) in November 2026, after which no artifact ran on either consumer. |
| `max-glibc` | `2.39` | Ceiling on the newest `GLIBC_x.y` symbol a shipped binary may require. `.github/scripts/check-glibc-ceiling.sh` fails the job above it, before upload, rather than letting it fail in a user's shell. At or under both consumers: the developer host (Ubuntu 24.04, 2.39) and the board (Debian 13, 2.41). |
| `timeout-minutes` | 330 | Under GitHub's hard 6 h ceiling so the ccache save still runs. |
| `retention-days` | 14 | Artifact retention. |

ccache does the heavy lifting: the key is keyed on the **`iree-src` submodule
SHA** (not `github.sha`, which would throw away the whole LLVM build on every
unrelated Rust commit) plus the runner image and the configure scripts, with
restore-keys that deliberately fall back across IREE bumps. Restore and save
are separate steps rather than `actions/cache`, so a build that fails or times
out still banks the objects it did compile. The two targets get separate caches
(3 GB host, 1 GB aarch64) to stay under the 10 GB per-repo budget -- otherwise
the two nightly jobs evict each other's LLVM objects every run.
