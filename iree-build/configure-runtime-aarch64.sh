#!/usr/bin/env bash
# Configures the IREE runtime for aarch64 Linux (the RK3588 board) with the
# Rocket HAL driver statically linked. The host compiler build supplies the
# code-generation tools needed while cross-compiling, but no compiler code is
# included in the target build.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cmake -S "${ROOT}/host-aarch64" \
  -B "${ROOT}/host-aarch64/build" \
  -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="${ROOT}/aarch64_linux_gnu.cmake" \
  -DIREE_HOST_BIN_DIR="${ROOT}/build/tools" \
  -DIREE_BUILD_COMPILER=OFF \
  "$@"
