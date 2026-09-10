// Copyright 2026
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Reading a Rocket dispatch at the flow phase: which executable it runs on,
// what its target config says, and the value of each dimension the shim
// passes as a push constant. Shared by rocket-pack-weights and
// rocket-assign-layout, which both need the dispatch's geometry from the
// same two places -- the config's statics and the `arith.constant` operands
// the target's `runtime_dimensions` list names.

#ifndef IREE_COMPILER_PLUGINS_TARGET_ROCKET_ROCKETDISPATCHQUERY_H_
#define IREE_COMPILER_PLUGINS_TARGET_ROCKET_ROCKETDISPATCHQUERY_H_

#include <cstdint>
#include <optional>

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/SymbolTable.h"
#include "rocket_plan.h"

namespace mlir::iree_compiler::IREE::HAL::rocket_query {

constexpr StringLiteral kRocketBackend = "rocket";

// The Rocket executable a dispatch runs on, or nothing when it is not a
// Rocket dispatch or names more than one entry point.
struct DispatchTarget {
  IREE::HAL::ExecutableOp executableOp;
  IREE::HAL::ExecutableVariantOp variantOp;
  IREE::HAL::ExecutableExportOp exportOp;
  SymbolRefAttr entryPoint;
  IREE::HAL::ExecutableTargetAttr target;
};

inline std::optional<DispatchTarget> rocketTarget(IREE::Flow::DispatchOp dispatchOp) {
  auto entryPoints = dispatchOp.getEntryPointsAttr();
  if (!entryPoints || entryPoints.size() != 1) {
    return std::nullopt;
  }
  auto entryPoint = dyn_cast<SymbolRefAttr>(entryPoints[0]);
  if (!entryPoint) {
    return std::nullopt;
  }
  Operation *symbol = SymbolTable::lookupNearestSymbolFrom(dispatchOp, entryPoint);
  auto exportOp = dyn_cast_or_null<IREE::HAL::ExecutableExportOp>(symbol);
  if (!exportOp) {
    return std::nullopt;
  }
  auto variantOp = exportOp->getParentOfType<IREE::HAL::ExecutableVariantOp>();
  auto executableOp = variantOp ? variantOp->getParentOfType<IREE::HAL::ExecutableOp>()
                                : IREE::HAL::ExecutableOp();
  if (!variantOp || !executableOp) {
    return std::nullopt;
  }
  IREE::HAL::ExecutableTargetAttr target = variantOp.getTarget();
  if (!target || target.getBackend().getValue() != kRocketBackend) {
    return std::nullopt;
  }
  return DispatchTarget{executableOp, variantOp, exportOp, entryPoint, target};
}

inline size_t arrayLength(DictionaryAttr config, StringRef key) {
  auto array = dyn_cast_or_null<ArrayAttr>(config.get(key));
  return array ? array.size() : 0;
}

inline bool boolFlag(DictionaryAttr config, StringRef key) {
  auto attr = dyn_cast_or_null<BoolAttr>(config.get(key));
  return attr && attr.getValue();
}

// Push constants before the bindings: the runtime dimensions, the runtime
// quantization parameters, and the trailing layout word (or the retired
// reader count) when the target declares one -- the order RocketTarget.cpp
// checks and the driver consumes.
inline size_t constantCount(DictionaryAttr config) {
  size_t constants = arrayLength(config, "runtime_dimensions") +
                     arrayLength(config, "runtime_quantization");
  if (boolFlag(config, "runtime_dense_readers") || boolFlag(config, "runtime_layout")) {
    ++constants;
  }
  return constants;
}

// One dimension of the dispatch: the push constant the target lists it as,
// which must be an `arith.constant` here, or the config's static value.
inline std::optional<int64_t> dimension(DictionaryAttr config, OperandRange arguments,
                                        StringRef name) {
  if (auto runtimeDims = dyn_cast_or_null<ArrayAttr>(config.get("runtime_dimensions"))) {
    for (auto [index, attr] : llvm::enumerate(runtimeDims)) {
      auto listed = dyn_cast<StringAttr>(attr);
      if (!listed || listed.getValue() != name) {
        continue;
      }
      if (index >= arguments.size()) {
        return std::nullopt;
      }
      auto constantOp = arguments[index].getDefiningOp<arith::ConstantOp>();
      if (!constantOp) {
        return std::nullopt;
      }
      auto value = dyn_cast<IntegerAttr>(constantOp.getValue());
      if (!value) {
        return std::nullopt;
      }
      return value.getInt();
    }
  }
  auto value = dyn_cast_or_null<IntegerAttr>(config.get(name));
  if (!value) {
    return std::nullopt;
  }
  return value.getInt();
}

// The wire's precision spellings, to the ABI's codes: the three rungs the
// transform spec programs.
inline std::optional<uint32_t> precisionCode(DictionaryAttr config) {
  auto precision = dyn_cast_or_null<StringAttr>(config.get("precision"));
  if (!precision) {
    return std::nullopt;
  }
  StringRef name = precision.getValue();
  if (name == "fp16") {
    return ROCKET_PLAN_PRECISION_FP16;
  }
  if (name == "int8") {
    return ROCKET_PLAN_PRECISION_INT8;
  }
  if (name == "int8_accumulator") {
    return ROCKET_PLAN_PRECISION_INT8_ACCUMULATOR;
  }
  return std::nullopt;
}

// Bytes one input element occupies on the rung, and one output element:
// the fp32 accumulator is the one rung where the two differ.
inline size_t inputElementBytes(uint32_t precision) {
  return precision == ROCKET_PLAN_PRECISION_FP16 ? 2 : 1;
}
inline size_t outputElementBytes(uint32_t precision) {
  switch (precision) {
  case ROCKET_PLAN_PRECISION_FP16:
    return 2;
  case ROCKET_PLAN_PRECISION_INT8_ACCUMULATOR:
    return 4;
  default:
    return 1;
  }
}

// Follows the metadata-only ops dispatch formation leaves between a producer
// and its consumer.
inline Value throughReshapes(Value value) {
  while (true) {
    if (auto reshape = value.getDefiningOp<IREE::Flow::TensorReshapeOp>()) {
      value = reshape.getSource();
    } else if (auto bitcast = value.getDefiningOp<IREE::Flow::TensorBitCastOp>()) {
      value = bitcast.getSource();
    } else {
      return value;
    }
  }
}

} // namespace mlir::iree_compiler::IREE::HAL::rocket_query

#endif // IREE_COMPILER_PLUGINS_TARGET_ROCKET_ROCKETDISPATCHQUERY_H_
