# Entry point included by IREE's build via -DIREE_CMAKE_PLUGIN_PATHS=<this dir>
# (see iree-src/CMakeLists.txt's own documentation of that mechanism, lines
# ~202-238). This file itself lives OUTSIDE iree-src's own source tree (a
# sibling repo to rocket-hal-driver/iree-rocket-hal, not a fork of IREE), so
# iree_package_ns()/iree_package_name() would otherwise fall back to their
# out-of-tree, PROJECT_SOURCE_DIR-relative path -- iree_setup_c_src_root()
# establishes this directory as its own package root instead, giving clean
# "rocket_compiler_plugin::target::Rocket"-style target names. Same pattern
# rocket-hal-driver/cts/CMakeLists.txt already uses for the identical reason.
iree_setup_c_src_root(
  PACKAGE_ROOT_PREFIX "rocket_compiler_plugin"
)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/target/Rocket target/Rocket)

if(IREE_BUILD_TESTS)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/test test)
endif()
