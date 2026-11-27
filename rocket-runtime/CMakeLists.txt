# The embeddable iree-rocket runtime: rocket/runtime.h (a session over the
# [rocket, local-task] device group every rocket-compiler module needs) and the
# library package C, C++ and Rust programs link against.
#
# add_subdirectory() this AFTER IREE itself, from a host project that has
# registered the rocket external HAL driver (iree-build/host*/CMakeLists.txt).
# It defines:
#
#   rocket_runtime          static, in-tree: links IREE's ordinary per-library
#                           targets, the way iree-run-module does. For samples
#                           and tests built inside this tree.
#   iree_rocket_runtime     libiree_rocket_runtime.so: IREE's runtime, the
#                           rocket driver and rocket/runtime.h in one file,
#                           exporting only iree_* and rocket_runtime_* (see
#                           src/exports.map). The recommended way to consume it.
#   iree_rocket_runtime_archive
#                           libiree_rocket_runtime.a: the same contents as one
#                           static archive, for C and C++ programs that link
#                           statically.
#   rocket_runtime_package  builds everything the install step below ships.
#
# and an install component, IreeRocketRuntime:
#
#   cmake --install <build> --component IreeRocketRuntime --prefix <dir> --strip
#
#   <dir>/include/iree-rocket-runtime/{rocket/runtime.h,iree/...}  headers
#   <dir>/lib/libiree_rocket_runtime.so{,.314,.314.0.1.0}         shared
#   <dir>/lib/libiree_rocket_runtime.a                            static
#   <dir>/lib/pkgconfig/iree-rocket-runtime{,-static}.pc
#   <dir>/lib/cmake/IreeRocketRuntime/IreeRocketRuntimeConfig.cmake
#
# (`lib` is CMAKE_INSTALL_LIBDIR: lib/<triplet> in a Debian build. The .pc and
# CMake files find the prefix relative to themselves either way.)
#
# The static archive is for C and C++ programs only. It contains the Rocket HAL
# driver's copy of the Rust standard library, so a Rust program that links it
# gets duplicate std symbols; Rust links the .so, whose version script hides it.

cmake_minimum_required(VERSION 3.24)

set(ROCKET_RUNTIME_VERSION "0.1.0")

# The SONAME follows IREE: libiree_rocket_runtime.so.<major><minor, 2 digits>,
# so .so.314 for IREE 3.14. The library exports IREE's whole C API (iree_*),
# and IREE makes no ABI promise between releases, so an IREE bump is an ABI
# break whatever this file's own version says. Deriving it means the bump
# cannot be forgotten. Changes to rocket/runtime.h between IREE bumps must
# stay additive: they ship under the same SONAME.
file(READ "${IREE_SOURCE_DIR}/runtime/version.json" _IREE_VERSION_JSON)
string(JSON _IREE_VERSION GET "${_IREE_VERSION_JSON}" "package-version")
if(NOT _IREE_VERSION MATCHES "^([0-9]+)\\.([0-9]+)")
  message(FATAL_ERROR "cannot read an IREE version from '${_IREE_VERSION}'")
endif()
math(EXPR _IREE_ABI "${CMAKE_MATCH_1} * 100 + ${CMAKE_MATCH_2}")
set(ROCKET_RUNTIME_SOVERSION "${_IREE_ABI}")

# Headers go one directory down (include/iree-rocket-runtime/{rocket,iree}),
# because they include all of IREE's runtime headers: installed straight
# into /usr/include they would collide file for file with any other IREE.
set(ROCKET_RUNTIME_INCLUDE_SUBDIR "iree-rocket-runtime")

include(GNUInstallDirs)
find_package(Threads REQUIRED)

if(NOT TARGET iree_runtime_unified OR NOT TARGET rocket_hal_driver_static
   OR NOT TARGET flatcc_parsing)
  message(FATAL_ERROR
    "rocket-runtime must be added after IREE, from a project that registers "
    "the rocket external HAL driver (see iree-build/host-aarch64).")
endif()

# The one source file, compiled once and position-independent so the same
# objects serve every library below.
add_library(rocket_runtime_objects OBJECT src/runtime.c)
target_include_directories(rocket_runtime_objects PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
)
# Usage requirements only (include directories, defines): an OBJECT library
# links nothing.
target_link_libraries(rocket_runtime_objects PUBLIC iree_runtime_unified)
set_target_properties(rocket_runtime_objects PROPERTIES
  C_STANDARD 11
  POSITION_INDEPENDENT_CODE ON
)

#-------------------------------------------------------------------------------
# In-tree static library
#-------------------------------------------------------------------------------

add_library(rocket_runtime STATIC $<TARGET_OBJECTS:rocket_runtime_objects>)
target_include_directories(rocket_runtime PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
)
# iree::runtime pulls in iree::hal::drivers, whose generated init_external.c
# registers the rocket driver; the rest is what src/runtime.c calls directly.
target_link_libraries(rocket_runtime PUBLIC
  iree::runtime
  iree::async
  iree::async::util::proactor_pool
  iree::base::threading
  iree::hal::drivers::local_task::task_driver
  iree::hal::local::loaders::registration
  iree::hal::local::plugins::registration
  iree::io::file_handle
  iree::modules::hal
  iree::task::api
  iree::vm::bytecode::module
)

#-------------------------------------------------------------------------------
# What the package's libraries are made of
#-------------------------------------------------------------------------------

# Exactly these archives, by file. Not through their targets: unified's CMake
# interface still names every per-library IREE archive it was merged from, and
# the driver's names iree::hal and iree::async. The unified archive holds every
# iree_cc_library object; flatcc's runtime (the bytecode verifier calls it) is
# the one external piece it leaves out.
get_target_property(_ROCKET_DRIVER_ARCHIVE rocket_hal_driver_static
                    IMPORTED_LOCATION)
set(_PACKAGE_ARCHIVES
  "$<TARGET_FILE:iree_runtime_unified>"
  "$<TARGET_FILE:flatcc_parsing>"
  "${_ROCKET_DRIVER_ARCHIVE}"
)
set(_PACKAGE_SYSTEM_LIBS Threads::Threads ${CMAKE_DL_LIBS} m rt)

#-------------------------------------------------------------------------------
# Shared library
#-------------------------------------------------------------------------------

set(_ROCKET_EXPORTS_MAP "${CMAKE_CURRENT_SOURCE_DIR}/src/exports.map")

add_library(iree_rocket_runtime SHARED $<TARGET_OBJECTS:rocket_runtime_objects>)
add_dependencies(iree_rocket_runtime
  iree_runtime_unified flatcc_parsing rocket_hal_driver_cargo)
target_include_directories(iree_rocket_runtime PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
)
target_link_libraries(iree_rocket_runtime PRIVATE
  # Whole-archive so every IREE API function is in the library, not only the
  # ones rocket/runtime.h happens to call: callers use iree_hal_buffer_view_*,
  # iree_vm_list_* and the rest directly.
  "-Wl,--whole-archive"
  "$<TARGET_FILE:iree_runtime_unified>"
  "-Wl,--no-whole-archive"
  "$<TARGET_FILE:flatcc_parsing>"
  "${_ROCKET_DRIVER_ARCHIVE}"
  ${_PACKAGE_SYSTEM_LIBS}
)
target_link_options(iree_rocket_runtime PRIVATE
  "LINKER:--version-script=${_ROCKET_EXPORTS_MAP}"
  # Calls between IREE functions bind inside the library instead of through
  # the PLT; nothing is meant to interpose on them.
  "LINKER:-Bsymbolic"
  "LINKER:--no-undefined"
)
set_target_properties(iree_rocket_runtime PROPERTIES
  VERSION "${ROCKET_RUNTIME_SOVERSION}.${ROCKET_RUNTIME_VERSION}"
  SOVERSION "${ROCKET_RUNTIME_SOVERSION}"
  LINK_DEPENDS "${_ROCKET_EXPORTS_MAP};${_ROCKET_DRIVER_ARCHIVE}"
)

#-------------------------------------------------------------------------------
# Static archive
#-------------------------------------------------------------------------------

# One archive rather than several: ld rescans a single archive until nothing
# new resolves, so a consumer never has to get the order right. (With separate
# archives it does: the driver needs IREE symbols, and unified's
# init_external.o needs the driver.) Merged with an ar MRI script; the Rust
# standard library's debug info is most of the driver's 50 MB and is dropped.
set(_ARCHIVE "${CMAKE_CURRENT_BINARY_DIR}/libiree_rocket_runtime.a")
set(_MRI "${CMAKE_CURRENT_BINARY_DIR}/libiree_rocket_runtime.mri")
file(GENERATE OUTPUT "${_MRI}" CONTENT
"CREATE ${_ARCHIVE}
ADDMOD $<JOIN:$<TARGET_OBJECTS:rocket_runtime_objects>, >
ADDLIB $<TARGET_FILE:iree_runtime_unified>
ADDLIB $<TARGET_FILE:flatcc_parsing>
ADDLIB ${_ROCKET_DRIVER_ARCHIVE}
SAVE
END
")
add_custom_command(
  OUTPUT "${_ARCHIVE}"
  COMMAND "${CMAKE_COMMAND}" -E rm -f "${_ARCHIVE}"
  COMMAND "${CMAKE_COMMAND}"
          -DAR=${CMAKE_AR} -DMRI=${_MRI}
          -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/run_ar_mri.cmake"
  COMMAND "${CMAKE_STRIP}" --strip-debug "${_ARCHIVE}"
  COMMAND "${CMAKE_RANLIB}" "${_ARCHIVE}"
  DEPENDS
    rocket_runtime_objects
    $<TARGET_OBJECTS:rocket_runtime_objects>
    iree_runtime_unified flatcc_parsing rocket_hal_driver_cargo
    "$<TARGET_FILE:iree_runtime_unified>"
    "$<TARGET_FILE:flatcc_parsing>"
    "${_ROCKET_DRIVER_ARCHIVE}"
    "${_MRI}"
  COMMENT "Merging libiree_rocket_runtime.a"
  VERBATIM
)
# ALL, so the nightly's plain `cmake --build` leaves it ready to install. The
# merge itself takes about a second.
add_custom_target(iree_rocket_runtime_archive ALL DEPENDS "${_ARCHIVE}")

add_custom_target(rocket_runtime_package
  DEPENDS iree_rocket_runtime iree_rocket_runtime_archive
)

#-------------------------------------------------------------------------------
# Install component IreeRocketRuntime
#-------------------------------------------------------------------------------

set(_COMPONENT IreeRocketRuntime)

install(TARGETS iree_rocket_runtime
  COMPONENT ${_COMPONENT}
  EXCLUDE_FROM_ALL
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)
install(FILES "${_ARCHIVE}"
  COMPONENT ${_COMPONENT}
  EXCLUDE_FROM_ALL
  DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)

set(_INCLUDE_DEST "${CMAKE_INSTALL_INCLUDEDIR}/${ROCKET_RUNTIME_INCLUDE_SUBDIR}")
install(DIRECTORY include/
  COMPONENT ${_COMPONENT}
  EXCLUDE_FROM_ALL
  DESTINATION "${_INCLUDE_DEST}"
)
# All of IREE's runtime headers, not only the ones rocket/runtime.h reaches:
# callers include iree/io, iree/hal/utils and so on directly. Tests and test
# utilities stay behind. The generated schema headers live in the build tree,
# found from the unified library's own binary directory (IREE_BINARY_DIR is
# scoped to IREE's subdirectory and not visible here).
get_target_property(_UNIFIED_BINARY_DIR iree_runtime_unified BINARY_DIR)
cmake_path(GET _UNIFIED_BINARY_DIR PARENT_PATH _IREE_RUNTIME_BINARY_DIR)
foreach(_root
    "${IREE_SOURCE_DIR}/runtime/src/iree"
    "${_IREE_RUNTIME_BINARY_DIR}")
  install(DIRECTORY "${_root}/"
    COMPONENT ${_COMPONENT}
    EXCLUDE_FROM_ALL
    DESTINATION "${_INCLUDE_DEST}/iree"
    FILES_MATCHING
      PATTERN "*.h"
      # Textually included by headers (iree/hal/profile_metrics.h).
      PATTERN "*.inc"
      PATTERN "*_test.h" EXCLUDE
      PATTERN "testing" EXCLUDE
      PATTERN "test" EXCLUDE
      PATTERN "CMakeFiles" EXCLUDE
  )
endforeach()

# The one compile definition IREE's public headers need from their consumer:
# iree_allocator_system() exists only when IREE_ALLOCATOR_SYSTEM_CTL names the
# allocator IREE was built with. Derived as iree/base/CMakeLists.txt derives it.
# (The IREE_HAVE_* definitions IREE's targets also carry are read only inside
# IREE's own sources.)
if(IREE_ALLOCATOR_SYSTEM_SELF)
  message(FATAL_ERROR "rocket-runtime does not package an allocator with "
                      "IREE_ALLOCATOR_SYSTEM_SELF state")
endif()
if(IREE_ALLOCATOR_SYSTEM_CTL)
  set(ROCKET_RUNTIME_PUBLIC_DEFINES
      "IREE_ALLOCATOR_SYSTEM_CTL=${IREE_ALLOCATOR_SYSTEM_CTL}")
else()
  string(REPLACE "-" "_" _ALLOCATOR "${IREE_ALLOCATOR_SYSTEM}")
  set(ROCKET_RUNTIME_PUBLIC_DEFINES
      "IREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_${_ALLOCATOR}_ctl")
endif()
set(_PUBLIC_CFLAGS "-D${ROCKET_RUNTIME_PUBLIC_DEFINES}")
# What a static consumer links besides the archive. libgcc_s supplies the
# unwinder the Rust standard library calls.
set(_STATIC_SYSTEM_LIBS "-lpthread -ldl -lm -lrt -lgcc_s")

# pkg-config and CMake package files. Both resolve paths relative to where the
# package was unpacked, so the tarball is relocatable. The way back up to the
# prefix depends on how deep CMAKE_INSTALL_LIBDIR is: `lib` for the tarball,
# `lib/aarch64-linux-gnu` for Debian. A fixed ../.. would put a Debian
# install's prefix at /usr/lib.
if(IS_ABSOLUTE "${CMAKE_INSTALL_LIBDIR}" OR IS_ABSOLUTE "${CMAKE_INSTALL_INCLUDEDIR}")
  message(FATAL_ERROR "rocket-runtime needs a relative CMAKE_INSTALL_LIBDIR and "
                      "CMAKE_INSTALL_INCLUDEDIR, so its package files stay relocatable")
endif()
file(RELATIVE_PATH _PC_TO_PREFIX
  "/prefix/${CMAKE_INSTALL_LIBDIR}/pkgconfig" "/prefix")
file(RELATIVE_PATH _CMAKE_TO_PREFIX
  "/prefix/${CMAKE_INSTALL_LIBDIR}/cmake/IreeRocketRuntime" "/prefix")
# file(RELATIVE_PATH) can end in a slash; the templates add their own.
string(REGEX REPLACE "/$" "" _PC_TO_PREFIX "${_PC_TO_PREFIX}")
string(REGEX REPLACE "/$" "" _CMAKE_TO_PREFIX "${_CMAKE_TO_PREFIX}")
set(ROCKET_RUNTIME_INSTALL_INCLUDEDIR "${_INCLUDE_DEST}")
foreach(_pc iree-rocket-runtime iree-rocket-runtime-static)
  configure_file("cmake/${_pc}.pc.in" "${CMAKE_CURRENT_BINARY_DIR}/${_pc}.pc" @ONLY)
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${_pc}.pc"
    COMPONENT ${_COMPONENT}
    EXCLUDE_FROM_ALL
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig"
  )
endforeach()
configure_file(cmake/IreeRocketRuntimeConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/IreeRocketRuntimeConfig.cmake" @ONLY)
configure_file(cmake/IreeRocketRuntimeConfigVersion.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/IreeRocketRuntimeConfigVersion.cmake" @ONLY)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/IreeRocketRuntimeConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/IreeRocketRuntimeConfigVersion.cmake"
  COMPONENT ${_COMPONENT}
  EXCLUDE_FROM_ALL
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/IreeRocketRuntime"
)

#-------------------------------------------------------------------------------
# Samples
#-------------------------------------------------------------------------------

add_subdirectory(samples)
