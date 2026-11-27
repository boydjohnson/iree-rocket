#!/usr/bin/env bash
# Configures a SECOND host compiler that carries the `vulkan-spirv` target
# backend alongside `rocket`, so one `iree-compile` can place part of a module
# on the RK3588's NPU and the rest on its Mali-G610 through panthor/panvk.
# agent-docs/vulkan-panvk-spike.md is the whole story; sections 5-7 are what
# this build exists for.
#
# It is deliberately NOT ../build. Two reasons:
#
#   * `build/` is the compiler everything else in this repository depends on
#     (rocket-compiler loads its libIREECompiler.so, the aarch64 runtime
#     build points IREE_HOST_BIN_DIR at its tools). Turning a target backend
#     on there triggers a ~7400-step rebuild of all of it to gain a backend
#     only this spike uses.
#   * There is no shortcut through dynamic plugins: the Rocket plugin is
#     registered statically and IREE's dynamically-loaded-plugin mechanism is
#     experimental and off in our build, so the stock wheel cannot be taught
#     the `rocket` target. A separate full build is the route.
#
# ~52 minutes on 16 cores with no ccache, 8144 steps. Afterwards:
#
#   export IREE_COMPILER_LIB=iree-build/build-vulkan/lib/libIREECompiler.so
#   cargo run -p rocket-compiler -- compile --input m.mlir --output m.vmfb \
#     --cpu-device-name=gpu_device --cpu-device-backend=vulkan \
#     --iree-flag=--iree-vulkan-target=valhall3
#
# `iree-lld` is not built here, so an llvm-cpu arm compiled with this library
# needs --iree-flag=--iree-llvmcpu-embedded-linker-path=<repo>/iree-build/build/tools/iree-lld.
#
# The matching board runtime is host-aarch64 configured with
# -DROCKET_WITH_VULKAN=ON, and its Vulkan HAL must carry
# patches/0001-vulkan-flush-invalidate-noncoherent-mappings.patch or it
# returns wrong data without saying so.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cmake -S "${ROOT}/iree-src" -B "${ROOT}/build-vulkan" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DIREE_BUILD_COMPILER=ON \
  -DIREE_BUILD_SAMPLES=OFF \
  -DIREE_BUILD_TESTS=OFF \
  -DIREE_CMAKE_PLUGIN_PATHS="${ROOT}/../rocket-compiler-plugin" \
  -DIREE_HAL_DRIVER_AMDGPU=OFF \
  -DIREE_HAL_DRIVER_CUDA=OFF \
  -DIREE_HAL_DRIVER_HIP=OFF \
  -DIREE_HAL_DRIVER_METAL=OFF \
  -DIREE_HAL_DRIVER_NULL=OFF \
  -DIREE_TARGET_BACKEND_CUDA=OFF \
  -DIREE_TARGET_BACKEND_METAL_SPIRV=OFF \
  -DIREE_TARGET_BACKEND_ROCM=OFF \
  -DIREE_TARGET_BACKEND_VULKAN_SPIRV=ON \
  -DIREE_TARGET_BACKEND_WEBGPU_SPIRV=OFF \
  -DIREE_INPUT_TORCH=ON \
  -DIREE_INPUT_STABLEHLO=OFF \
  -DIREE_ENABLE_WERROR_FLAG=OFF \
  "$@"
