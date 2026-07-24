# CTS wiring for the "rocket" HAL driver. add_subdirectory()'d directly
# from host/CMakeLists.txt and host-aarch64/CMakeLists.txt, after
# add_subdirectory(iree), so the iree_cc_library/iree_hal_cts_test_suite
# macros are already defined. Lives outside iree-src (like the rest of
# rocket-hal-driver) so iree_package_ns() would otherwise fall back to its
# out-of-tree, PROJECT_SOURCE_DIR-relative path -- iree_setup_c_src_root()
# below establishes this directory as its own package root instead, giving
# clean "rocket_cts::backends"-style target names.
#
# No TESTDATA_LIBS passed to iree_hal_cts_test_suite(): only the
# non-executable test binaries (buffer/command_buffer/core/file/queue) get
# built. See backends.cc for why that's the point, not a limitation to
# work around.

iree_setup_c_src_root(
  PACKAGE_ROOT_PREFIX "rocket_cts"
)

# iree_hal_cts_test_suite() hardcodes its test_main.cc path as
# "${PROJECT_SOURCE_DIR}/runtime/src/iree/hal/cts/util/test_main.cc",
# correct only when PROJECT_SOURCE_DIR is IREE's own root -- true inside
# iree-src's own add_subdirectory tree, but this directory is
# add_subdirectory()'d from the *host* project (see host/CMakeLists.txt),
# where PROJECT_SOURCE_DIR reverts to the host project's root once control
# returns from add_subdirectory(iree-src). Overriding it here (directory-
# scoped set(), not CACHE/PARENT_SCOPE -- doesn't leak back to the caller)
# points the macro at the real IREE root.
set(PROJECT_SOURCE_DIR "${IREE_SOURCE_DIR}")

iree_cc_library(
  NAME
    backends
  SRCS
    "backends.cc"
  DEPS
    iree::hal
    iree::hal::cts::util::registry
    rocket_hal_driver_static
  TESTONLY
  ALWAYSLINK
  PUBLIC
)

iree_hal_cts_test_suite(
  BACKENDS_LIB
    ::backends
  NAME
    rocket
)

# Hand-written (not iree_hal_cts_test_suite's generic dispatch tests --
# structurally skipped for "rocket", see backends.cc) end-to-end test for
# the pooling ukernel, driving the real public IREE HAL API directly. See
# pooling_dispatch_test.cc's own module doc comment for why this exists
# alongside iree-rocket-hal/tests/pooling_hw.rs rather than instead of it.
#
# UNVERIFIED against a real build (no IREE source tree available while
# writing this): the exact gtest DEPS target name(s) below are a guess by
# analogy to other iree_cc_test() call sites elsewhere in IREE's own tree
# (commonly "gtest" + "gtest_main", sometimes bundled as a single
# "iree::testing::gtest"-style target) -- confirm/fix against whichever
# convention this project's actual IREE checkout uses before trusting this
# builds.
iree_cc_test(
  NAME
    pooling_dispatch_test
  SRCS
    "pooling_dispatch_test.cc"
  DEPS
    iree::hal
    rocket_hal_driver_static
    gtest
    gtest_main
)

# Same hand-written end-to-end pattern as pooling_dispatch_test.cc above,
# covering Conv2d instead -- see conv_dispatch_test.cc's own module doc
# comment. Notably exercises executable_cache.rs's tag=2 (Precision::Fp16),
# the newest addition to the tag convention.
iree_cc_test(
  NAME
    conv_dispatch_test
  SRCS
    "conv_dispatch_test.cc"
  DEPS
    iree::hal
    rocket_hal_driver_static
    gtest
    gtest_main
)
