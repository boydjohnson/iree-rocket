#!/usr/bin/env bash
# Configures iree-src directly to build iree-compile with the Rocket compiler
# target plugin registered. No wrapper CMakeLists.txt is needed for this one
# (unlike host/ and host-aarch64/): the external-HAL-driver mechanism needs a
# host project to call iree_register_external_hal_driver() before
# add_subdirectory(iree) runs, but IREE_CMAKE_PLUGIN_PATHS is a plain cache
# variable IREE's own CMakeLists.txt reads directly, so no wrapping is
# required here.
#
# iree-src is a vendored third-party submodule (vanilla iree-org/iree), so
# this can't be expressed as a CMakePresets.json living inside it -- presets
# are always rooted at the directory containing the CMakeLists.txt they
# configure. See ../host/CMakePresets.json and ../host-aarch64/CMakePresets.json
# for the two builds that do have their own wrapper project.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cmake -S "${ROOT}/iree-src" -B "${ROOT}/build" -G Ninja \
  -DIREE_BUILD_COMPILER=ON \
  -DIREE_CMAKE_PLUGIN_PATHS="${ROOT}/../rocket-compiler-plugin" \
  -DIREE_HAL_DRIVER_AMDGPU=OFF \
  -DIREE_HAL_DRIVER_CUDA=OFF \
  -DIREE_HAL_DRIVER_HIP=OFF \
  -DIREE_HAL_DRIVER_METAL=OFF \
  -DIREE_HAL_DRIVER_NULL=OFF \
  -DIREE_HAL_DRIVER_VULKAN=OFF \
  -DIREE_TARGET_BACKEND_CUDA=OFF \
  -DIREE_TARGET_BACKEND_METAL_SPIRV=OFF \
  -DIREE_TARGET_BACKEND_ROCM=OFF \
  -DIREE_TARGET_BACKEND_VULKAN_SPIRV=OFF \
  -DIREE_TARGET_BACKEND_WEBGPU_SPIRV=OFF \
  "$@"
