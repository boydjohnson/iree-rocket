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

# The shared planner. rocket-core's Rust planner is exposed to this C++
# plugin through rocket-plan-ffi's versioned C ABI (rocket_plan.h), built
# here as a host static library by cargo and linked into libIREECompiler.so
# like any other dependency. COMPILER_ROADMAP.md section 2: the compiler asks
# the same planner the runtime asks, so the two cannot drift.
#
# The cargo build uses its own target directory under the CMake build tree
# so it never races a developer's `cargo build` in the workspace's target/,
# and the DEPENDS glob makes an edit to either Rust crate rebuild the
# library on the next ninja run (cargo itself decides what to recompile).
find_program(ROCKET_CARGO cargo REQUIRED
  DOC "cargo, needed to build rocket-plan-ffi for the Rocket compiler plugin")
set(ROCKET_WORKSPACE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." CACHE PATH
  "Path to the iree-rocket Cargo workspace root")
set(_ROCKET_CARGO_TARGET_DIR "${CMAKE_CURRENT_BINARY_DIR}/cargo")
set(_ROCKET_PLAN_FFI_LIB "${_ROCKET_CARGO_TARGET_DIR}/release/librocket_plan_ffi.a")
file(GLOB_RECURSE _ROCKET_PLAN_FFI_SOURCES CONFIGURE_DEPENDS
  "${ROCKET_WORKSPACE_ROOT}/rocket-core/src/*.rs"
  "${ROCKET_WORKSPACE_ROOT}/rocket-core/Cargo.toml"
  "${ROCKET_WORKSPACE_ROOT}/rocket-plan-ffi/src/*.rs"
  "${ROCKET_WORKSPACE_ROOT}/rocket-plan-ffi/Cargo.toml"
  "${ROCKET_WORKSPACE_ROOT}/rocket-plan-ffi/include/*.h"
)
add_custom_command(
  OUTPUT "${_ROCKET_PLAN_FFI_LIB}"
  COMMAND ${CMAKE_COMMAND} -E env "CARGO_TARGET_DIR=${_ROCKET_CARGO_TARGET_DIR}"
          "${ROCKET_CARGO}" build --release --quiet -p rocket-plan-ffi
          --manifest-path "${ROCKET_WORKSPACE_ROOT}/Cargo.toml"
  DEPENDS ${_ROCKET_PLAN_FFI_SOURCES}
  COMMENT "cargo build rocket-plan-ffi (the shared planner's C ABI)"
  VERBATIM
)
add_custom_target(rocket_plan_ffi_build DEPENDS "${_ROCKET_PLAN_FFI_LIB}")
add_library(rocket_plan_ffi STATIC IMPORTED GLOBAL)
set_target_properties(rocket_plan_ffi PROPERTIES
  IMPORTED_LOCATION "${_ROCKET_PLAN_FFI_LIB}"
  INTERFACE_INCLUDE_DIRECTORIES "${ROCKET_WORKSPACE_ROOT}/rocket-plan-ffi/include"
  # What a Rust staticlib's std needs from the host.
  INTERFACE_LINK_LIBRARIES "pthread;dl;m"
)
add_dependencies(rocket_plan_ffi rocket_plan_ffi_build)

# The plugin's own transform-dialect matchers (transform.rocket.match.*).
iree_tablegen_library(
  NAME
    RocketTransformOpsGen
  TD_FILE
    "RocketTransformOps.td"
  OUTS
    --gen-op-decls RocketTransformOps.h.inc
    --gen-op-defs RocketTransformOps.cpp.inc
)

set(_ROCKET_SRCS
  "RocketAnnotateFinalPlacementPass.cpp"
  "RocketAnnotateOriginalPlacementPass.cpp"
  "RocketDemoteConvInputsPass.cpp"
  "RocketExpandGemvToMatmulPass.cpp"
  "RocketUnbatchMatmulPass.cpp"
  "RocketFoldConvPadPass.cpp"
  "RocketFoldNeutralInitPass.cpp"
  "RocketFuseConvRelu6Pass.cpp"
  "RocketFuseInt8RequantEpiloguePass.cpp"
  "RocketAssignLayoutPass.cpp"
  "RocketPackWeightsPass.cpp"
  "RocketPinUnclaimedDispatchesPass.cpp"
  "RocketPlanCandidatesPass.cpp"
  "RocketPlanQuery.cpp"
  "RocketPromoteUnclaimedConvInputsPass.cpp"
  "RocketTarget.cpp"
  "RocketTransformExtension.cpp"
  "RocketTransposeQuantizedConvPass.cpp"
  "RocketVerifyConvShapesPass.cpp"
)
set(_ROCKET_DEPS
  ::RocketTransformOpsGen
  LLVMSupport
  MLIRIR
  MLIRTransformDialect
  MLIRTransformDialectInterfaces
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRLinalgUtils
  MLIRMathDialect
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
  rocket_plan_ffi
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
    "RocketPlanQuery.h"
    "RocketTransformExtension.h"
    "RocketTransformOps.h.inc"
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
