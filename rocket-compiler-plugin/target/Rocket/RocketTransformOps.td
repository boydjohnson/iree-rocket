// Copyright 2026
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// The Rocket plugin's own transform-dialect matchers, registered as an
// extension of the transform dialect (RocketTransformExtension.cpp) so the
// transform spec can use them alongside `transform.iree.match.*`.

#ifndef ROCKET_COMPILER_PLUGIN_TARGET_ROCKET_ROCKETTRANSFORMOPS
#define ROCKET_COMPILER_PLUGIN_TARGET_ROCKET_ROCKETTRANSFORMOPS

include "mlir/Dialect/Transform/Interfaces/MatchInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def Rocket_MatchAdmittedOp : Op<Transform_Dialect, "rocket.match.admitted",
    [IsolatedFromAbove,
     MatchOpInterface,
     SingleOpMatcher,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = [{Succeeds if the shared planner and the admission envelope
                  both accept the operation.}];
  let description = [{
    The one check every convolution and matmul matcher in
    `rocket_conv2d_transform_spec.mlir` carries. It asks `rocket-core` two
    separate questions, and declines if either says no.

    **Did the planner refuse these extents?** `rocket-plan-candidates` runs
    before the match loop, plans every candidate through `rocket-plan-ffi`
    and tags the refused ones `rocket.plan_refused`; this op declines a
    tagged op. An op that pass never saw, or one whose spatial extents are
    dynamic (which it defers to the runtime), carries no tag and passes this
    half.

    **Is the shape class characterized?** `rocket_admit_conv` /
    `rocket_admit_matmul` answer from `rocket_core::admission`, which is
    indexed by precision, kernel, stride and channel counts alone -- so it
    decides even when the spatial extents are still symbolic. These are the
    ceilings that used to be `transform.iree.match.dim_bounds` lines in this
    spec, two or three per matcher; keeping them in Rust is what lets the
    compiler and the HAL be checked against one table.

    `precision` names the rung the *lowering* will program, for the case
    where the operand types do not imply it: a requantized int8 convolution
    and an accumulator one are both `i8 x i8 -> i32` in the IR and have
    separate envelopes. Accepted spellings are `fp16`, `int8_accumulator`
    and `int8_requant`. Omitted, the rung is read from the operand types,
    which is right for every fp16 matcher.

    `no_offload` makes the op decline unconditionally. That is how
    `rocket-compiler --no-offload` defeats the match loop now that the
    matchers carry no dimension bounds to rewrite; see
    `rocket-compiler/src/spec.rs`.

    #### Return modes

    Succeeds if the payload op is admitted. Produces a silenceable failure
    naming the refusal otherwise, including when the handle's payload is not
    a convolution or matmul this dialect can read. Produces a definite
    failure if the handle is not associated with a single payload op.
  }];

  let arguments = (ins TransformHandleTypeInterface:$operand_handle,
                       OptionalAttr<StrAttr>:$precision,
                       UnitAttr:$no_offload);
  let assemblyFormat = "$operand_handle attr-dict `:` type($operand_handle)";
  let extraClassDeclaration = SingleOpMatcher.extraDeclaration;

  let cppNamespace = "mlir::iree_compiler::IREE::HAL::rocket_transform";
}

#endif // ROCKET_COMPILER_PLUGIN_TARGET_ROCKET_ROCKETTRANSFORMOPS
