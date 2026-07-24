# Builds rocket-hal-driver's Rust staticlib via cargo and exposes it as a
# CMake IMPORTED STATIC target, so IREE's iree_register_external_hal_driver()
# mechanism (see the host project that add_subdirectory()s this file) can
# link it directly. See src/lib.rs for why this is a staticlib and not a
# dlopen'd cdylib -- confirmed from IREE's own source that the external HAL
# driver mechanism is link-time only.

cmake_minimum_required(VERSION 3.16)

set(ROCKET_TARGET_DIR "${CMAKE_CURRENT_BINARY_DIR}/cargo-target")

# When cross-compiling (CMAKE_CROSSCOMPILING set by a toolchain file, e.g.
# aarch64_linux_gnu.cmake for the actual RK3588 board target) cargo needs
# an explicit --target, its own cross-linker set via the
# CARGO_TARGET_<TRIPLE>_LINKER env var convention, and its output lands
# under target-dir/<triple>/release/ instead of target-dir/release/ --
# matches exactly how iree-rocket-hal's own aarch64 cross-builds have
# worked all session (CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=
# aarch64-linux-gnu-gcc).
set(ROCKET_CARGO_ENV)
set(ROCKET_CARGO_TARGET_ARGS)
if(CMAKE_CROSSCOMPILING AND CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
  set(ROCKET_RUST_TARGET "aarch64-unknown-linux-gnu")
  set(ROCKET_CARGO_TARGET_ARGS --target "${ROCKET_RUST_TARGET}")
  set(ROCKET_CARGO_ENV
    "CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=${CMAKE_C_COMPILER}"
  )
  set(ROCKET_LIB_PATH "${ROCKET_TARGET_DIR}/${ROCKET_RUST_TARGET}/release/librocket_hal_driver.a")
else()
  set(ROCKET_LIB_PATH "${ROCKET_TARGET_DIR}/release/librocket_hal_driver.a")
endif()

file(GLOB_RECURSE ROCKET_RUST_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.rs"
)
file(GLOB_RECURSE ROCKET_SCHEMA_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/../rocket-schema/src/*.rs"
  "${CMAKE_CURRENT_SOURCE_DIR}/../rocket-schema/generated/rust/*.rs"
  "${CMAKE_CURRENT_SOURCE_DIR}/../rocket-schema/schema/*.fbs"
)

add_custom_command(
  OUTPUT "${ROCKET_LIB_PATH}"
  COMMAND ${CMAKE_COMMAND} -E env ${ROCKET_CARGO_ENV}
          cargo build --release
          --manifest-path "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml"
          --target-dir "${ROCKET_TARGET_DIR}"
          ${ROCKET_CARGO_TARGET_ARGS}
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  DEPENDS
    ${ROCKET_RUST_SOURCES}
    ${ROCKET_SCHEMA_SOURCES}
    "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml"
    "${CMAKE_CURRENT_SOURCE_DIR}/../rocket-schema/Cargo.toml"
  COMMENT "Building rocket-hal-driver (cargo build --release${ROCKET_CARGO_TARGET_ARGS})"
  VERBATIM
)
add_custom_target(rocket_hal_driver_cargo DEPENDS "${ROCKET_LIB_PATH}")

add_library(rocket_hal_driver_static STATIC IMPORTED GLOBAL)
set_target_properties(rocket_hal_driver_static PROPERTIES
  IMPORTED_LOCATION "${ROCKET_LIB_PATH}"
  # CMake can't introspect an externally-built (cargo) .a's symbol table,
  # so without this it has no way to know rocket_hal_driver_static itself
  # needs iree::hal/iree::async symbols (iree_hal_file_*,
  # iree_async_proactor_pool_* -- device.rs's queue_read/queue_write/
  # create/destroy) resolved *after* it on the final link line. Omitting
  # this worked by accident for consumers whose own link graph happened to
  # place iree::hal/iree::async after this target already (e.g. the
  # iree_hal_cts_test_suite()-generated binaries, which pull rocket-cts's
  # `backends` OBJECT library's dependencies in independently) but broke
  # for cts/pooling_dispatch_test.cc's hand-written iree_cc_test(), whose
  # DEPS list happened to put iree::hal before rocket_hal_driver_static --
  # ld's single left-to-right archive pass never revisits an already-
  # scanned .a for symbols a later archive turns out to need. Setting this
  # makes CMake place iree::hal/iree::async after rocket_hal_driver_static
  # on every consumer's link line automatically, regardless of that
  # consumer's own DEPS order.
  INTERFACE_LINK_LIBRARIES "iree::hal;iree::async;iree::async::util::proactor_pool"
)
add_dependencies(rocket_hal_driver_static rocket_hal_driver_cargo)
