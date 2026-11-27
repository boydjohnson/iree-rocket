# CI, in detail

Every workflow this repository runs, what each job does and why it is
configured the way it is. `README.md` carries the outside-facing summary --
which workflow gates what, and the live badges; this is the version for
someone changing a workflow or debugging a red run.

Three workflows, under [`.github/workflows/`](../.github/workflows).

| Workflow | Trigger | Jobs | Typical runtime |
|---|---|---|---|
| `ci.yml` -- CI | every push to `main`, every pull request | `check, build, test`; `cargo fmt --check` | ~1.5 min |
| `nightly.yml` -- Nightly artifacts | `cron 0 7 * * *` (07:00 UTC), `workflow_dispatch` | `compiler-host`, then `runtime-aarch64`; `compiler-aarch64` alongside | ~20 min warm ccache, up to ~4 h cold |
| `iree-build.yml` -- IREE build (reusable) | `workflow_call` only, so its runs are attributed to the caller and it has no badge of its own | one `build` job per `target` input | see above |

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
  ([developer tools](developer-tools.md#board-convolution-regression-gate)).
- **`cargo fmt --check`** (`ubuntu-latest`) -- `cargo fmt --all -- --check` on
  **nightly** rustfmt, not stable: `rustfmt.toml` sets `imports_granularity`,
  which is still unstable, and stable rustfmt would silently ignore it and
  disagree with how the code is actually formatted. A second step checks the
  Rust bindings in `rocket-runtime/rust`, their own workspace (building them
  needs the installed runtime package, which CI does not have; formatting
  does not).

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
- **`compiler-aarch64`** -- the `compiler-host` target again, on
  `ubuntu-24.04-arm`: the same build, packaging and smoke test, natively for
  aarch64. The package is named after `uname -m`, so this one uploads
  `iree-compile-aarch64`, a compiler that runs on the board itself. It skips
  `iree-host-tools`: `runtime-aarch64` runs on x86_64 and could not execute
  them. Its ccache is separate (the key carries the runner) and as large as
  the x86_64 one, so its first run is a cold LLVM build and may hit the
  330 min timeout; the next run picks up the ccache it banked.
- **`runtime-aarch64`** -- unpacks `iree-host-tools` back into
  `iree-build/build/`, installs the `gcc`/`g++-aarch64-linux-gnu` cross
  toolchain, configures with `iree-build/configure-runtime-aarch64.sh`
  (Release, `IREE_ENABLE_WERROR_FLAG=OFF`, tests **on** -- that is what builds
  the CTS) and builds `iree-run-module` / `iree-benchmark-module` with the
  Rocket HAL driver statically linked. Then `cargo test --workspace --release
  --target aarch64-unknown-linux-gnu --no-run` for the crates' own libtest
  binaries, which the CMake build never produces. Installs the
  `IreeRocketRuntime` component (rocket-runtime/: `libiree_rocket_runtime.{so,a}`,
  headers, pkg-config and CMake files), link-tests its `embed.cc` sample
  through both `.pc` files and builds the Rust bindings (`rocket-runtime/rust`,
  every target) against the unpacked package -- link only, the runner cannot
  run aarch64. Uploads
  `iree-runtime-aarch64`, `iree-rocket-runtime-aarch64`, `rocket-cts-aarch64`
  (the non-executable HAL CTS categories) and `rocket-rust-tests-aarch64`, the
  last two with a `RUN.md` saying how to run them on the board. Timeout
  120 min: no LLVM here.

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
nothing. So does `runtime-aarch64` on an arm runner. `compiler-host` builds
for whatever the `runner` is. Inputs worth knowing about:

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
