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

set(_ROCKET_SRCS
  "RocketAnnotateFinalPlacementPass.cpp"
  "RocketAnnotateOriginalPlacementPass.cpp"
  "RocketDemoteConvInputsPass.cpp"
  "RocketPinUnclaimedDispatchesPass.cpp"
  "RocketTarget.cpp"
  "RocketTransposeQuantizedConvPass.cpp"
  "RocketVerifyConvShapesPass.cpp"
)
set(_ROCKET_DEPS
  LLVMSupport
  MLIRIR
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRLinalgUtils
  MLIRPass
  MLIRSupport
  MLIRTensorDialect
  MLIRTransforms
  iree::compiler::Dialect::Flow::IR
  iree::compiler::Dialect::HAL::IR
  iree::compiler::Dialect::HAL::Target
  iree::compiler::Dialect::Stream::IR
  iree::compiler::PluginAPI
  iree::compiler::Utils
  ::RocketExecutableDef
)
set(_ROCKET_DEFINES)

# onnx.ConvInteger has no torch-mlir lowering, so the Rocket plugin supplies
# its own expansion (RocketExpandOnnxConvIntegerPass.cpp) -- but only when the
# in-tree Torch input plugin is being built, since that is what provides the
# torch dialect the pass rewrites. torch-iree_compiler_defs is the Torch
# plugin's own INTERFACE target, and carries the include root for the
# tablegen-generated torch op headers alongside torch-mlir's include/ tree.
if(IREE_INPUT_TORCH)
  list(APPEND _ROCKET_SRCS "RocketExpandOnnxConvIntegerPass.cpp")
  list(APPEND _ROCKET_DEPS
    torch-iree_compiler_defs
    iree::compiler::plugins::input::Torch::torch-mlir::TorchDialectIR
  )
  list(APPEND _ROCKET_DEFINES "ROCKET_ENABLE_ONNX_INPUT")
endif()

iree_cc_library(
  NAME
    Rocket
  HDRS
    "RocketPasses.h"
  SRCS
    ${_ROCKET_SRCS}
  DEPS
    ${_ROCKET_DEPS}
  DEFINES
    ${_ROCKET_DEFINES}
  ALWAYSLINK
  PUBLIC
)

iree_compiler_register_plugin(
  PLUGIN_ID
    hal_target_rocket
  TARGET
    ::Rocket
)
