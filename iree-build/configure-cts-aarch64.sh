#!/usr/bin/env bash
# Configures the rocket-hal-driver CTS (cts/, gtest binaries including
# conv_schema_harness_test) for aarch64 Linux (the RK3588 board). Unlike
# configure-runtime-aarch64.sh, which configures host-aarch64/ (no cts/
# subdirectory), this cross-compiles the host/ project -- the one that adds
# cts/ as a sibling of iree-src -- with the same aarch64 toolchain and host
# tools used for the runtime build.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cmake -S "${ROOT}/host" \
  -B "${ROOT}/host/build-aarch64" \
  -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="${ROOT}/aarch64_linux_gnu.cmake" \
  -DIREE_HOST_BIN_DIR="${ROOT}/build/tools" \
  "$@"
