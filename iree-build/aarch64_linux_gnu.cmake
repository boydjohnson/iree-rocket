# CMake toolchain file for cross-compiling IREE's runtime for aarch64 Linux
# (glibc) -- the RK3588 board's actual architecture. Modeled on IREE's own
# build_tools/cmake/linux_riscv64.cmake, but much simpler: Ubuntu's
# gcc-aarch64-linux-gnu/g++-aarch64-linux-gnu packages are a complete
# cross-toolchain with their own built-in sysroot search paths (same ones
# iree-rocket-hal's own aarch64 cross-builds already rely on all session --
# no separate --sysroot/RISCV_TOOLCHAIN_ROOT-style config needed).

cmake_minimum_required(VERSION 3.16)

if(AARCH64_LINUX_GNU_TOOLCHAIN_INCLUDED)
  return()
endif()
set(AARCH64_LINUX_GNU_TOOLCHAIN_INCLUDED true)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER   aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
set(CMAKE_AR           aarch64-linux-gnu-ar)
set(CMAKE_RANLIB       aarch64-linux-gnu-ranlib)
set(CMAKE_STRIP        aarch64-linux-gnu-strip)

# Don't try to run target-architecture binaries on this (x86_64) build host
# during CMake's own compiler/feature checks.
set(CMAKE_CROSSCOMPILING TRUE)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
