set(
  ROCKET_SCHEMA_ROOT
  "${CMAKE_CURRENT_LIST_DIR}/../../../rocket-schema"
  CACHE PATH
  "Path to the rocket-schema repository"
)
set(
  ROCKET_EXECUTABLE_SCHEMA
  "${ROCKET_SCHEMA_ROOT}/schema/rocket_executable_def.fbs"
)
if(NOT EXISTS "${ROCKET_EXECUTABLE_SCHEMA}")
  message(FATAL_ERROR
    "Rocket executable schema not found at ${ROCKET_EXECUTABLE_SCHEMA}; "
    "set ROCKET_SCHEMA_ROOT to the rocket-schema checkout")
endif()

flatbuffer_c_library(
  NAME
    RocketExecutableDef
  SRCS
    "${ROCKET_EXECUTABLE_SCHEMA}"
  FLATCC_ARGS
    "--reader"
    "--builder"
    "--verifier"
  PUBLIC
)

iree_cc_library(
  NAME
    Rocket
  SRCS
    "RocketAnnotateFinalPlacementPass.cpp"
    "RocketAnnotateOriginalPlacementPass.cpp"
    "RocketTarget.cpp"
  DEPS
    LLVMSupport
    MLIRIR
    MLIRLinalgDialect
    MLIRPass
    MLIRSupport
    iree::compiler::Dialect::HAL::IR
    iree::compiler::Dialect::HAL::Target
    iree::compiler::PluginAPI
    iree::compiler::Utils
    ::RocketExecutableDef
  ALWAYSLINK
  PUBLIC
)

iree_compiler_register_plugin(
  PLUGIN_ID
    hal_target_rocket
  TARGET
    ::Rocket
)
