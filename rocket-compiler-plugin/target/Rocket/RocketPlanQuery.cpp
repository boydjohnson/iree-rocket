// Copyright 2026
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "RocketPlanQuery.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include "rocket_plan.h"

namespace mlir::iree_compiler::IREE::HAL {
namespace {

int64_t dim(ShapedType type, unsigned index, bool &dynamic) {
  int64_t extent = type.getDimSize(index);
  if (ShapedType::isDynamic(extent)) {
    dynamic = true;
  }
  return extent;
}

void readSteps(Operation *op, StringRef name, int64_t &h, int64_t &w) {
  auto attr = op->getAttrOfType<DenseIntElementsAttr>(name);
  if (!attr || attr.getNumElements() != 2) {
    return;
  }
  auto values = attr.getValues<int64_t>();
  h = values[0];
  w = values[1];
}

} // namespace

std::optional<RocketCandidate> readRocketCandidate(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  // Every candidate kind takes exactly one input, one filter and one init;
  // a fill or a cast generic has fewer and must not be indexed.
  if (!linalgOp || linalgOp.getNumDpsInputs() != 2 ||
      linalgOp.getNumDpsInits() != 1) {
    return std::nullopt;
  }
  auto input = dyn_cast<RankedTensorType>(linalgOp.getDpsInputs()[0].getType());
  auto filter =
      dyn_cast<RankedTensorType>(linalgOp.getDpsInputs()[1].getType());
  auto output =
      dyn_cast<RankedTensorType>(linalgOp.getDpsInits()[0].getType());
  if (!input || !filter || !output) {
    return std::nullopt;
  }
  RocketCandidate c;
  c.inputElement = input.getElementType();
  c.filterElement = filter.getElementType();
  c.outputElement = output.getElementType();
  bool &dyn = c.dynamic;
  // The channel counts and the kernel are read through this instead, so a
  // convolution with dynamic spatial extents still has a usable admission
  // query. `dyn` stays the planner's stricter flag: it is set by everything
  // including these.
  bool &chan = c.channelsDynamic;
  auto both = [&](ShapedType type, unsigned index) {
    int64_t extent = dim(type, index, dyn);
    if (ShapedType::isDynamic(extent)) {
      chan = true;
    }
    return extent;
  };
  bool recognized =
      llvm::TypeSwitch<Operation *, bool>(op)
          .Case<linalg::Conv2DNhwcHwcfOp>([&](auto) {
            c.kind = "dense_conv2d";
            c.layout = "nhwc";
            c.batch = dim(input, 0, dyn);
            c.height = dim(input, 1, dyn);
            c.width = dim(input, 2, dyn);
            c.inChannels = both(input, 3);
            c.kernelHeight = both(filter, 0);
            c.kernelWidth = both(filter, 1);
            c.outChannels = both(filter, 3);
            c.outHeight = dim(output, 1, dyn);
            c.outWidth = dim(output, 2, dyn);
            return true;
          })
          .Case<linalg::Conv2DNchwFchwOp>([&](auto) {
            c.kind = "dense_conv2d";
            c.layout = "nchw";
            c.batch = dim(input, 0, dyn);
            c.inChannels = both(input, 1);
            c.height = dim(input, 2, dyn);
            c.width = dim(input, 3, dyn);
            c.outChannels = both(filter, 0);
            c.kernelHeight = both(filter, 2);
            c.kernelWidth = both(filter, 3);
            c.outHeight = dim(output, 2, dyn);
            c.outWidth = dim(output, 3, dyn);
            return true;
          })
          .Case<linalg::DepthwiseConv2DNhwcHwcOp>([&](auto) {
            c.kind = "depthwise_conv2d";
            c.layout = "nhwc";
            c.depthwise = true;
            c.batch = dim(input, 0, dyn);
            c.height = dim(input, 1, dyn);
            c.width = dim(input, 2, dyn);
            c.inChannels = both(input, 3);
            c.kernelHeight = both(filter, 0);
            c.kernelWidth = both(filter, 1);
            c.outChannels = c.inChannels;
            c.outHeight = dim(output, 1, dyn);
            c.outWidth = dim(output, 2, dyn);
            return true;
          })
          .Case<linalg::DepthwiseConv2DNchwChwOp>([&](auto) {
            c.kind = "depthwise_conv2d";
            c.layout = "nchw";
            c.depthwise = true;
            c.batch = dim(input, 0, dyn);
            c.inChannels = both(input, 1);
            c.height = dim(input, 2, dyn);
            c.width = dim(input, 3, dyn);
            c.kernelHeight = both(filter, 1);
            c.kernelWidth = both(filter, 2);
            c.outChannels = c.inChannels;
            c.outHeight = dim(output, 2, dyn);
            c.outWidth = dim(output, 3, dyn);
            return true;
          })
          .Case<linalg::MatmulOp>([&](linalg::MatmulOp matmul) {
            c.kind = "matmul";
            c.layout = "row_major";
            c.matmul = true;
            // [M,K] x [K,N]: M is the convolution width, K/N the channels.
            c.width = dim(input, 0, dyn);
            c.height = 1;
            c.inChannels = both(input, 1);
            c.outChannels = both(filter, 1);
            c.outWidth = dim(output, 0, dyn);
            c.outHeight = 1;
            // M is a spatial extent for the planner and an admission axis
            // for the envelope, so it counts for both.
            if (ShapedType::isDynamic(c.width)) {
              chan = true;
            }
            if (matmul.hasUserDefinedMaps()) {
              c.formProblem = "user-defined indexing maps (a transposed or "
                              "broadcast operand); the FC lowering takes "
                              "row-major [M,K] x [K,N] only";
            }
            return true;
          })
          .Default([](Operation *) { return false; });
  if (!recognized) {
    return std::nullopt;
  }
  if (!c.matmul) {
    readSteps(op, "strides", c.strideH, c.strideW);
    readSteps(op, "dilations", c.dilationH, c.dilationW);
  }
  return c;
}

std::optional<uint32_t> rocketPrecisionFor(const RocketCandidate &c,
                                           std::string &why) {
  Type in = c.inputElement, f = c.filterElement, out = c.outputElement;
  if (in.isF16() && f.isF16() && (out.isF32() || out.isF16())) {
    return ROCKET_PLAN_PRECISION_FP16;
  }
  if (in.isBF16() && f.isBF16() && (out.isF32() || out.isBF16())) {
    return ROCKET_PLAN_PRECISION_BF16;
  }
  if (in.isSignlessInteger(8) && f.isSignlessInteger(8) &&
      out.isSignlessInteger(32)) {
    return ROCKET_PLAN_PRECISION_INT8_ACCUMULATOR;
  }
  if (in.isF32() || f.isF32()) {
    why = "f32 operands: the demotion left this op at f32, so no Rocket "
          "matcher's f16/f16/f32 typing applies";
    return std::nullopt;
  }
  std::string types;
  llvm::raw_string_ostream os(types);
  os << in << " x " << f << " -> " << out;
  why = "operand types " + types + " have no Rocket lowering";
  return std::nullopt;
}

std::string rocketShapeSummary(const RocketCandidate &c) {
  std::string s;
  llvm::raw_string_ostream os(s);
  if (c.matmul) {
    os << c.width << "x" << c.inChannels << " x " << c.inChannels << "x"
       << c.outChannels;
    return s;
  }
  os << c.width << "x" << c.height << " Cin " << c.inChannels << " Cout "
     << c.outChannels << " k" << c.kernelHeight << "x" << c.kernelWidth << " s"
     << c.strideH;
  if (c.strideW != c.strideH) {
    os << "x" << c.strideW;
  }
  return s;
}

} // namespace mlir::iree_compiler::IREE::HAL
