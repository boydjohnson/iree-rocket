# Minimal host project proving rocket-hal-driver actually links into IREE's
# runtime via the real (CMake-static-link, not dlopen -- see
# rocket-hal-driver/Cargo.toml) external-HAL-driver mechanism. Registers
# "rocket" with iree_register_external_hal_driver() BEFORE add_subdirectory
# (iree) processes runtime/src/iree/hal/drivers/CMakeLists.txt, matching
# the null driver README's documented pattern.

cmake_minimum_required(VERSION 3.16)
project(rocket_hal_driver_host C CXX ASM)

# iree-src enables testing within its own subtree, but that doesn't propagate
# to rocket-cts below -- it's added as a *sibling* subdirectory of this
# project, not nested inside iree-src, and CMake's enable_testing() only
# cascades from a directory to its own descendants. Without this, rocket-cts's
# add_test() calls are silently inert: the binaries build, but `ctest -N`
# finds nothing (no rocket-cts/CTestTestfile.cmake gets generated at all).
enable_testing()

set(IREE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../iree-src" CACHE PATH "")
set(ROCKET_HAL_DRIVER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../rocket-hal-driver" CACHE PATH "")

list(APPEND CMAKE_MODULE_PATH "${IREE_SOURCE_DIR}/build_tools/cmake")
include(iree_plugin_register)

iree_register_external_hal_driver(
  NAME rocket
  SOURCE_DIR "${ROCKET_HAL_DRIVER_DIR}"
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/rocket-hal-driver"
  DRIVER_TARGET rocket_hal_driver_static
  REGISTER_FN iree_hal_rocket_driver_module_register
)

set(IREE_EXTERNAL_HAL_DRIVERS "rocket" CACHE STRING "" FORCE)
set(IREE_BUILD_COMPILER OFF CACHE BOOL "" FORCE)
# ON so the HAL CTS (runtime/src/iree/hal/cts) gets built -- see
# rocket-hal-driver/cts/ for the "rocket" backend registration linked into
# it below. Pulls in the rest of IREE's own unit test suite too; that's an
# accepted cost of using IREE_BUILD_TESTS as the CTS gate. OFF builds only the
# runtime, its tools and rocket-runtime, and then needs no host iree-compile
# (IREE_HOST_BIN_DIR) for test modules: debian/rules builds that way.
option(ROCKET_BUILD_TESTS "Build IREE's runtime tests and the rocket HAL CTS" ON)
set(IREE_BUILD_TESTS ${ROCKET_BUILD_TESTS} CACHE BOOL "" FORCE)
set(IREE_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_DOCS OFF CACHE BOOL "" FORCE)
# Opt-in Vulkan HAL driver, for the Mali-G610 through Mesa panvk. It is OFF by
# default because the host iree-compile is built without the vulkan-spirv
# target, so the Vulkan executable tests cannot be generated. Build only the
# tool targets (iree-run-module, iree-benchmark-module) with it ON.
option(ROCKET_WITH_VULKAN "Also build IREE's Vulkan HAL driver" OFF)
set(IREE_HAL_DRIVER_VULKAN ${ROCKET_WITH_VULKAN} CACHE BOOL "" FORCE)
# iree_check_test_suite() normally skips generating a check-test module for
# backends whose HAL driver is disabled (see the driver-availability check in
# build_tools/cmake/iree_check_test.cmake), but this flag (IREE default: ON)
# forces every default backend's module to build regardless. That includes
# vulkan-spirv, which the host iree-compile this project's IREE_HOST_BIN_DIR
# points at (built with IREE_TARGET_BACKEND_VULKAN_SPIRV=OFF, see
# ../configure-compiler-host.sh) can't target -- `ninja` then fails on that
# check-test module's generation. OFF makes the module generation actually
# respect IREE_HAL_DRIVER_VULKAN=OFF above instead of overriding it.
set(IREE_BUILD_ALL_CHECK_TEST_MODULES OFF CACHE BOOL "" FORCE)

# Default symbol visibility, so rocket-runtime's libiree_rocket_runtime.so can
# export IREE's C API. IREE builds -fvisibility=hidden by default and its
# IREE_API_EXPORT is empty in C, which leaves a shared library nothing to
# export; this is the option IREE provides for re-exporting. The .so's version
# script (rocket-runtime/src/exports.map) decides what is actually public.
set(IREE_VISIBILITY_HIDDEN OFF CACHE BOOL "" FORCE)

add_subdirectory("${IREE_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/iree")

# HAL CTS backend registration for "rocket" -- runs the non-executable CTS
# categories (buffer/command_buffer/core/file/queue) against the real
# driver. See rocket-hal-driver/cts/backends.cc for why no executable
# categories are built (no compiler side needed).
if(ROCKET_BUILD_TESTS)
  add_subdirectory("${ROCKET_HAL_DRIVER_DIR}/cts" "${CMAKE_CURRENT_BINARY_DIR}/rocket-cts")
endif()

# The embeddable runtime: rocket/runtime.h, libiree_rocket_runtime.so and the
# IreeRocketRuntime install component (see rocket-runtime/CMakeLists.txt).
add_subdirectory("${ROCKET_HAL_DRIVER_DIR}/../rocket-runtime" "${CMAKE_CURRENT_BINARY_DIR}/rocket-runtime")
