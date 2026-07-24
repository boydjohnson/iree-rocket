# Minimal host project proving rocket-hal-driver actually links into IREE's
# runtime via the real (CMake-static-link, not dlopen -- see
# rocket-hal-driver/Cargo.toml) external-HAL-driver mechanism. Registers
# "rocket" with iree_register_external_hal_driver() BEFORE add_subdirectory
# (iree) processes runtime/src/iree/hal/drivers/CMakeLists.txt, matching
# the null driver README's documented pattern.

cmake_minimum_required(VERSION 3.16)
project(rocket_hal_driver_host C CXX ASM)

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
# accepted cost of using IREE_BUILD_TESTS as the CTS gate.
set(IREE_BUILD_TESTS ON CACHE BOOL "" FORCE)
set(IREE_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(IREE_HAL_DRIVER_VULKAN OFF CACHE BOOL "" FORCE)

add_subdirectory("${IREE_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/iree")

# HAL CTS backend registration for "rocket" -- runs the non-executable CTS
# categories (buffer/command_buffer/core/file/queue) against the real
# driver. See rocket-hal-driver/cts/backends.cc for why no executable
# categories are built (no compiler side needed).
add_subdirectory("${ROCKET_HAL_DRIVER_DIR}/cts" "${CMAKE_CURRENT_BINARY_DIR}/rocket-cts")
