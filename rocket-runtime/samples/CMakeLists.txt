# rocket-run-module (C) is built twice, against the in-tree static library and
# against the shared library, so both link paths are exercised on every build.
# embed (C++) is the minimal embedding the package ships as source; it is built
# here against the shared library the way an outside consumer would use it.

add_executable(rocket-run-module run_module.c)
target_link_libraries(rocket-run-module PRIVATE rocket_runtime)
set_target_properties(rocket-run-module PROPERTIES C_STANDARD 11)

# The shared-library consumers get headers only, as the package provides them:
# IREE's include directories and the one public definition, without linking
# IREE a second time.
foreach(_target rocket-run-module-shared embed)
  if(_target STREQUAL "embed")
    add_executable(${_target} embed.cc)
    set_target_properties(${_target} PROPERTIES CXX_STANDARD 17)
  else()
    add_executable(${_target} run_module.c)
    set_target_properties(${_target} PROPERTIES C_STANDARD 11)
  endif()
  target_link_libraries(${_target} PRIVATE iree_rocket_runtime)
  target_include_directories(${_target} PRIVATE
    $<TARGET_PROPERTY:iree_runtime_unified,INTERFACE_INCLUDE_DIRECTORIES>)
  target_compile_definitions(${_target} PRIVATE ${ROCKET_RUNTIME_PUBLIC_DEFINES})
  set_target_properties(${_target} PROPERTIES
    # Finds libiree_rocket_runtime.so next to it once both are copied over.
    BUILD_RPATH "$ORIGIN"
  )
endforeach()

add_dependencies(rocket_runtime_package
  rocket-run-module rocket-run-module-shared embed)

install(FILES run_module.c embed.cc
  COMPONENT IreeRocketRuntime
  EXCLUDE_FROM_ALL
  DESTINATION "${CMAKE_INSTALL_DATADIR}/rocket-runtime/samples"
)
