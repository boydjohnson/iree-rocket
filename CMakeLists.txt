# Builds rocket-hal-driver's Rust staticlib via cargo and exposes it as a
# CMake IMPORTED STATIC target, so IREE's iree_register_external_hal_driver()
# mechanism (see the host project that add_subdirectory()s this file) can
# link it directly. See src/lib.rs for why this is a staticlib and not a
# dlopen'd cdylib -- confirmed from IREE's own source that the external HAL
# driver mechanism is link-time only.

cmake_minimum_required(VERSION 3.16)

set(ROCKET_TARGET_DIR "${CMAKE_CURRENT_BINARY_DIR}/cargo-target")
set(ROCKET_LIB_PATH "${ROCKET_TARGET_DIR}/release/librocket_hal_driver.a")

file(GLOB_RECURSE ROCKET_RUST_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.rs"
)

add_custom_command(
  OUTPUT "${ROCKET_LIB_PATH}"
  COMMAND cargo build --release
          --manifest-path "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml"
          --target-dir "${ROCKET_TARGET_DIR}"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  DEPENDS ${ROCKET_RUST_SOURCES} "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml"
  COMMENT "Building rocket-hal-driver (cargo build --release)"
  VERBATIM
)
add_custom_target(rocket_hal_driver_cargo DEPENDS "${ROCKET_LIB_PATH}")

add_library(rocket_hal_driver_static STATIC IMPORTED GLOBAL)
set_target_properties(rocket_hal_driver_static PROPERTIES
  IMPORTED_LOCATION "${ROCKET_LIB_PATH}"
)
add_dependencies(rocket_hal_driver_static rocket_hal_driver_cargo)
