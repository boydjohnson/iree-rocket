// Copyright 2026
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Decides, per Rocket dispatch, which of its inputs read their producer's
// NC1HWC2 cube in place and how many Rocket dispatches read its own result
// that way, and writes the decision into the dispatch's trailing layout push
// constant (`rocket_core::layout::DispatchLayout`,
// `Conv2DDef.runtime_layout` and its twins). COMPILER_ROADMAP.md 6.2, the
// first form.
//
// The driver used to make both decisions itself at record time: it matched
// a consumer's input against the byte range of every earlier dispatch's
// output and compared cube geometries, and it skipped a dense output write
// when the number of consumers that chained equalled a reader count the
// compiler passed down. The geometry half of that is a pure function of
// shapes the compiler knows (`rocket_core::layout`, the same one the driver
// calls), so it is decided here, once, and the driver checks the
// declaration: it attempts a chain only on an input declared packed, fails
// `INTERNAL` if the producer it finds does not match, and still falls back
// to the dense write when a declared reader turns out to be on a later
// command buffer -- which is why the count survives the first form.
//
// The walk is the reader count's: after rocket-pin-unclaimed-dispatches,
// every reader of a dispatch result is a formed `flow.dispatch` and the set
// is final. Each Rocket dispatch's binding geometries come from its target
// config and constant push constants through rocket-plan-ffi; an edge
// between two Rocket dispatches is packed when the producer publishes a
// cube and `rocket_plan_chain_identity` holds, dense otherwise, with the
// failing condition recorded. Every such edge, packed or dense, and every
// dispatch's reader count go into a `rocket.layout_decisions` array on the
// function, which `rocket-compiler audit` prints -- the layout half of the
// placement report.

#include <optional>
#include <string>

#include "RocketDispatchQuery.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "rocket_plan.h"

namespace mlir::iree_compiler::IREE::HAL {
namespace {

using namespace rocket_query;

constexpr StringLiteral kLayoutDecisionsAttrName = "rocket.layout_decisions";
constexpr uint32_t kReadersShift = 16;

// One side of an edge, as rocket_plan_cube_geometry describes it.
struct Cube {
  rocket_plan_cube_desc_t desc{};
  bool wholeAtom = false;
  bool exact = false;
};

std::optional<Cube> cube(uint32_t kind, size_t elementBytes, int64_t width, int64_t height,
                         int64_t channels) {
  Cube result;
  result.desc.struct_size = sizeof(result.desc);
  result.desc.kind = kind;
  result.desc.element_bytes = static_cast<uint32_t>(elementBytes);
  result.desc.width = static_cast<uint64_t>(width);
  result.desc.height = static_cast<uint64_t>(height);
  result.desc.channels = static_cast<uint64_t>(channels);
  rocket_plan_cube_geometry_t geometry{};
  geometry.struct_size = sizeof(geometry);
  if (rocket_plan_cube_geometry(&result.desc, &geometry, nullptr, 0) != ROCKET_PLAN_OK) {
    return std::nullopt;
  }
  result.wholeAtom = geometry.whole_atom != 0;
  result.exact = geometry.exact != 0;
  return result;
}

// What the pass knows about one Rocket dispatch.
struct Site {
  IREE::Flow::DispatchOp op;
  std::string name;
  DictionaryAttr config;
  StringRef kernel;
  // Push constants before the bindings, and whether the last of them is a
  // layout word this pass may write.
  unsigned constants = 0;
  bool declaresLayout = false;
  // Per input binding: the cube this dispatch would pack that input to, when
  // the binding can read a cube at all.
  SmallVector<std::optional<Cube>> inputs;
  // The cube this dispatch publishes, when it publishes one, else why not.
  std::optional<Cube> output;
  std::string noOutputReason;
  // Decisions.
  uint32_t packedInputs = 0;
  uint32_t packedReaders = 0;
};

std::string siteName(DispatchTarget target) {
  return (target.executableOp.getSymName() + "::" + target.exportOp.getSymName()).str();
}

// Fills the geometry half of `site` from its config and constant operands.
// Returns false when the dispatch is not one this pass lays out (a kind
// with no cube on either side, or a dimension that is not a constant).
bool describe(Site &site) {
  DictionaryAttr config = site.config;
  OperandRange arguments = site.op.getArguments();
  std::optional<uint32_t> precision = precisionCode(config);
  auto dim = [&](StringRef name) { return dimension(config, arguments, name); };
  if (site.kernel == "conv2d") {
    if (!precision) {
      return false;
    }
    auto width = dim("input_width"), height = dim("input_height"),
         cin = dim("input_channels"), cout = dim("output_channels"),
         kw = dim("weights_width"), kh = dim("weights_height"), stride = dim("stride");
    if (!width || !height || !cin || !cout || !kw || !kh || !stride) {
      return false;
    }
    bool depthwise = boolFlag(config, "depthwise");
    bool epilogueAdd = boolFlag(config, "epilogue_add");
    // The output extents the runtime derives: ask the planner, which is what
    // the driver's `Shape::output_width` is, rather than restate the
    // formula. Padding is the def's explicit pair, defaulting to zero.
    rocket_plan_conv_desc_t desc{};
    desc.struct_size = sizeof(desc);
    desc.precision = *precision;
    desc.width = *width;
    desc.height = *height;
    desc.in_channels = *cin;
    desc.out_channels = *cout;
    desc.stride = *stride;
    desc.kernel_height = *kh;
    desc.kernel_width = *kw;
    desc.pad_top = dim("pad_top").value_or(0);
    desc.pad_left = dim("pad_left").value_or(0);
    desc.activation = ROCKET_PLAN_ACTIVATION_NONE;
    desc.depthwise = depthwise ? 1 : 0;
    desc.quantization.input_scale = 1.0f;
    desc.quantization.weights_scale = 1.0f;
    desc.quantization.output_scale = 1.0f;
    rocket_plan_conv_plan_t plan{};
    plan.struct_size = sizeof(plan);
    char message[256] = {0};
    if (rocket_plan_conv(&desc, nullptr, &plan, message, sizeof(message)) != ROCKET_PLAN_OK) {
      site.noOutputReason = std::string("planner refused: ") + message;
      return true;
    }
    size_t inElem = inputElementBytes(*precision);
    size_t outElem = outputElementBytes(*precision);
    site.inputs.assign(epilogueAdd ? 4 : 3, std::nullopt);
    if (*cin > ROCKET_PLAN_MAX_DENSE_CHANNELS) {
      site.inputs[0] = cube(ROCKET_PLAN_CUBE_CONV, inElem, *width, *height, *cin);
    }
    if (epilogueAdd) {
      // The residual epilogue: the skip is packed as an fp16 feature cube of
      // the output's own geometry, and what the dispatch publishes is the
      // EW task's sum, in that same geometry.
      site.inputs[3] =
          cube(ROCKET_PLAN_CUBE_ELEMENTWISE, 2, plan.output_width, plan.output_height, *cout);
      std::optional<Cube> sum =
          cube(ROCKET_PLAN_CUBE_ELEMENTWISE, 2, plan.output_width, plan.output_height, *cout);
      if (sum && sum->exact) {
        site.output = sum;
      } else {
        site.noOutputReason = "residual sum cube pads its channels";
      }
      return true;
    }
    if (depthwise && *precision == ROCKET_PLAN_PRECISION_INT8_ACCUMULATOR) {
      // The depthwise accumulator writer's atom is not the feature atom
      // (`Shape::output_atom_bytes`): no cube.
      site.noOutputReason = "depthwise accumulator output atom";
      return true;
    }
    std::optional<Cube> out =
        cube(ROCKET_PLAN_CUBE_CONV, outElem, plan.output_width, plan.output_height, *cout);
    if (out && out->wholeAtom) {
      site.output = out;
    } else {
      site.noOutputReason = "output pixel is not a whole number of atoms";
    }
    return true;
  }
  if (site.kernel == "matmul") {
    if (!precision) {
      return false;
    }
    auto m = dim("m"), k = dim("k"), n = dim("n");
    if (!m || !k || !n) {
      return false;
    }
    size_t inElem = inputElementBytes(*precision);
    size_t outElem = outputElementBytes(*precision);
    site.inputs.assign(3, std::nullopt);
    // A K of one is read dense (fc.rs's `InputPackingLayout::Dense`).
    if (*k > 1) {
      site.inputs[0] = cube(ROCKET_PLAN_CUBE_MATMUL, inElem, *m, 1, *k);
    }
    std::optional<Cube> out = cube(ROCKET_PLAN_CUBE_MATMUL, outElem, *m, 1, *n);
    if (out && out->wholeAtom) {
      site.output = out;
    } else {
      site.noOutputReason = "output pixel is not a whole number of atoms";
    }
    return true;
  }
  if (site.kernel == "pooling") {
    if (!precision) {
      return false;
    }
    auto width = dim("input_width"), height = dim("input_height"), channels = dim("channels"),
         kw = dim("kernel_width"), kh = dim("kernel_height"), sx = dim("stride_x"),
         sy = dim("stride_y");
    if (!width || !height || !channels || !kw || !kh || !sx || !sy || *kw <= 0 || *kh <= 0 ||
        *sx <= 0 || *sy <= 0) {
      return false;
    }
    // The driver's `floor_output_extent`: the padded extent less the kernel,
    // floored over the stride, plus one.
    int64_t paddedW = *width + dim("pad_left").value_or(0) + dim("pad_right").value_or(0);
    int64_t paddedH = *height + dim("pad_top").value_or(0) + dim("pad_bottom").value_or(0);
    if (paddedW < *kw || paddedH < *kh) {
      return false;
    }
    int64_t ow = (paddedW - *kw) / *sx + 1;
    int64_t oh = (paddedH - *kh) / *sy + 1;
    size_t elem = inputElementBytes(*precision);
    site.inputs.assign(2, std::nullopt);
    site.inputs[0] = cube(ROCKET_PLAN_CUBE_POOLING, elem, *width, *height, *channels);
    std::optional<Cube> out = cube(ROCKET_PLAN_CUBE_POOLING, elem, ow, oh, *channels);
    if (out && out->exact) {
      site.output = out;
    } else {
      site.noOutputReason = "pool output pads its channels";
    }
    return true;
  }
  if (site.kernel == "elementwise_binary") {
    auto width = dim("width"), height = dim("height"), channels = dim("channels");
    if (!width || !height || !channels) {
      return false;
    }
    site.inputs.assign(3, std::nullopt);
    site.inputs[0] = cube(ROCKET_PLAN_CUBE_ELEMENTWISE, 2, *width, *height, *channels);
    site.inputs[1] = site.inputs[0];
    std::optional<Cube> out = cube(ROCKET_PLAN_CUBE_ELEMENTWISE, 2, *width, *height, *channels);
    if (out && out->exact) {
      site.output = out;
    } else {
      site.noOutputReason = "element-wise cube pads its channels";
    }
    return true;
  }
  return false;
}

// The verdict on one edge.
struct Edge {
  Site *consumer;
  unsigned binding;
  // The producer site, or null when the value comes from elsewhere.
  Site *producer;
  // "argument", "cpu", "other" when `producer` is null.
  std::string producerKind;
  bool packed = false;
  std::string reason;
};

struct RocketAssignLayoutPass
    : public PassWrapper<RocketAssignLayoutPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RocketAssignLayoutPass)

  StringRef getArgument() const final { return "rocket-assign-layout"; }
  StringRef getDescription() const final {
    return "Declares, per Rocket dispatch, which inputs read their producer's "
           "NC1HWC2 cube in place and how many Rocket dispatches read its "
           "result that way, in the trailing layout push constant; records "
           "every edge on the function as rocket.layout_decisions.";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect>();
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    if (rocket_plan_abi_version() != ROCKET_PLAN_ABI_VERSION) {
      module.emitError() << "rocket-assign-layout: rocket-plan-ffi reports ABI version "
                         << rocket_plan_abi_version() << " but this plugin was built against "
                         << ROCKET_PLAN_ABI_VERSION;
      return signalPassFailure();
    }

    // Every Rocket dispatch, described.
    std::vector<std::unique_ptr<Site>> sites;
    llvm::DenseMap<Operation *, Site *> siteOf;
    module.walk([&](IREE::Flow::DispatchOp dispatchOp) {
      std::optional<DispatchTarget> target = rocketTarget(dispatchOp);
      if (!target) {
        return;
      }
      DictionaryAttr config = target->target.getConfiguration();
      auto kernel = config ? dyn_cast_or_null<StringAttr>(config.get("kernel")) : StringAttr();
      if (!kernel) {
        return;
      }
      auto site = std::make_unique<Site>();
      site->op = dispatchOp;
      site->name = siteName(*target);
      site->config = config;
      site->kernel = kernel.getValue();
      site->constants = static_cast<unsigned>(constantCount(config));
      site->declaresLayout = boolFlag(config, "runtime_layout");
      if (dispatchOp.getNumResults() != 1 || dispatchOp.getArguments().size() < site->constants) {
        return;
      }
      if (!describe(*site)) {
        site->inputs.clear();
        site->output.reset();
        if (site->noOutputReason.empty()) {
          site->noOutputReason = "not a laid-out kind, or a dimension is not a constant";
        }
      }
      siteOf[dispatchOp.getOperation()] = site.get();
      sites.push_back(std::move(site));
    });

    // Input edges: each cube-capable binding, back to its producer.
    std::vector<Edge> edges;
    for (auto &site : sites) {
      OperandRange arguments = site->op.getArguments();
      for (unsigned binding = 0; binding < site->inputs.size(); ++binding) {
        if (!site->inputs[binding]) {
          continue;
        }
        unsigned index = site->constants + binding;
        if (index >= arguments.size()) {
          continue;
        }
        Edge edge{site.get(), binding, nullptr, "", false, ""};
        if (site->op.isOperandTied(site->op.getWorkload().size() + index)) {
          edge.producerKind = "tied";
          edge.reason = "the binding is a tied operand";
          edges.push_back(edge);
          continue;
        }
        Value source = throughReshapes(arguments[index]);
        Operation *definer = source.getDefiningOp();
        if (!definer) {
          edge.producerKind = "argument";
          edge.reason = "the value is a function argument";
        } else if (auto producerOp = dyn_cast<IREE::Flow::DispatchOp>(definer)) {
          if (Site *producer = siteOf.lookup(producerOp)) {
            edge.producer = producer;
            if (cast<OpResult>(source).getResultNumber() != 0) {
              edge.reason = "the producer's result is not its first";
            } else if (!producer->output) {
              edge.reason = producer->noOutputReason;
            } else {
              char message[256] = {0};
              uint32_t status = rocket_plan_chain_identity(&producer->output->desc,
                                                           &site->inputs[binding]->desc, message,
                                                           sizeof(message));
              if (status == ROCKET_PLAN_OK) {
                edge.packed = true;
                edge.reason = "geometries identical";
              } else {
                edge.reason = message;
              }
            }
          } else {
            edge.producerKind = "cpu";
            edge.reason = "the producer is a CPU dispatch";
          }
        } else {
          edge.producerKind = "other";
          edge.reason = std::string("the producer is ") + definer->getName().getStringRef().str();
        }
        if (edge.packed) {
          site->packedInputs |= 1u << binding;
        }
        edges.push_back(edge);
      }
    }

    // Reader counts: every reader must be a Rocket dispatch that takes the
    // cube, through the same metadata ops, or the count is zero.
    for (auto &site : sites) {
      if (!site->output) {
        continue;
      }
      unsigned count = 0;
      bool allPacked = true;
      std::function<void(Value)> walk = [&](Value value) {
        for (OpOperand &use : value.getUses()) {
          if (!allPacked) {
            return;
          }
          Operation *owner = use.getOwner();
          if (isa<IREE::Flow::TensorReshapeOp, IREE::Flow::TensorBitCastOp>(owner)) {
            if (use.getOperandNumber() != 0 || owner->getNumResults() != 1) {
              allPacked = false;
              return;
            }
            walk(owner->getResult(0));
            continue;
          }
          auto reader = dyn_cast<IREE::Flow::DispatchOp>(owner);
          Site *consumer = reader ? siteOf.lookup(reader) : nullptr;
          if (!consumer) {
            allPacked = false;
            return;
          }
          unsigned index = use.getOperandNumber();
          unsigned first = reader.getWorkload().size();
          unsigned last = first + reader.getArguments().size();
          if (index < first || index >= last || reader.isOperandTied(index) ||
              index - first < consumer->constants) {
            allPacked = false;
            return;
          }
          unsigned binding = index - first - consumer->constants;
          if (!(consumer->packedInputs & (1u << binding))) {
            allPacked = false;
            return;
          }
          ++count;
        }
      };
      walk(site->op.getResult(0));
      site->packedReaders = allPacked ? count : 0;
    }

    // Write the word, and the record.
    unsigned declared = 0, packedEdges = 0, denseNpuEdges = 0, elidable = 0;
    llvm::DenseMap<Operation *, SmallVector<Attribute>> recordsByFunction;
    for (auto &site : sites) {
      OpBuilder builder(site->op);
      if (site->declaresLayout && site->constants > 0) {
        unsigned position = site->constants - 1;
        OperandRange arguments = site->op.getArguments();
        if (!arguments[position].getType().isInteger(32)) {
          site->op.emitOpError() << "rocket-assign-layout: target declares runtime_layout but "
                                    "argument "
                                 << position << " is not an i32 push constant";
          return signalPassFailure();
        }
        uint32_t word = (site->packedInputs & 0xFFFFu) |
                        (std::min<uint32_t>(site->packedReaders, 0xFFFFu) << kReadersShift);
        Value constant = arith::ConstantOp::create(
            builder, site->op.getLoc(), builder.getI32IntegerAttr(static_cast<int32_t>(word)));
        site->op->setOperand(arguments.getBeginOperandIndex() + position, constant);
        ++declared;
        if (site->packedReaders > 0) {
          ++elidable;
        }
      }
      Operation *function = site->op->getParentOp();
      while (function && !isa<FunctionOpInterface>(function)) {
        function = function->getParentOp();
      }
      if (!function) {
        continue;
      }
      recordsByFunction[function].push_back(builder.getDictionaryAttr({
          builder.getNamedAttr("edge", builder.getStringAttr("output")),
          builder.getNamedAttr("site", builder.getStringAttr(site->name)),
          builder.getNamedAttr("kind", builder.getStringAttr(site->kernel)),
          builder.getNamedAttr("verdict", builder.getStringAttr(
                                              site->output ? "cube" : "dense")),
          builder.getNamedAttr("reason", builder.getStringAttr(
                                             site->output ? "" : site->noOutputReason)),
          builder.getNamedAttr("packed_readers",
                               builder.getI64IntegerAttr(site->packedReaders)),
          builder.getNamedAttr("loc", site->op.getLoc()),
      }));
    }
    for (Edge &edge : edges) {
      OpBuilder builder(edge.consumer->op);
      Operation *function = edge.consumer->op->getParentOp();
      while (function && !isa<FunctionOpInterface>(function)) {
        function = function->getParentOp();
      }
      if (!function) {
        continue;
      }
      if (edge.producer) {
        if (edge.packed) {
          ++packedEdges;
        } else {
          ++denseNpuEdges;
        }
      }
      recordsByFunction[function].push_back(builder.getDictionaryAttr({
          builder.getNamedAttr("edge", builder.getStringAttr("input")),
          builder.getNamedAttr("site", builder.getStringAttr(edge.consumer->name)),
          builder.getNamedAttr("kind", builder.getStringAttr(edge.consumer->kernel)),
          builder.getNamedAttr("binding", builder.getI64IntegerAttr(edge.binding)),
          builder.getNamedAttr("producer",
                               builder.getStringAttr(edge.producer ? edge.producer->name
                                                                   : edge.producerKind)),
          builder.getNamedAttr("verdict", builder.getStringAttr(edge.packed ? "packed" : "dense")),
          builder.getNamedAttr("reason", builder.getStringAttr(edge.reason)),
          builder.getNamedAttr("loc", edge.consumer->op.getLoc()),
      }));
    }
    for (auto &[function, records] : recordsByFunction) {
      OpBuilder builder(function);
      function->setAttr(kLayoutDecisionsAttrName, builder.getArrayAttr(records));
    }
    if (!sites.empty()) {
      module.emitRemark() << "rocket-assign-layout: " << declared << " dispatch(es) declared, "
                          << packedEdges << " NPU->NPU edge(s) packed, " << denseNpuEdges
                          << " left dense, " << elidable
                          << " dispatch(es) read only through their cube";
    }
  }
};

static PassRegistration<RocketAssignLayoutPass> reg;

} // namespace
} // namespace mlir::iree_compiler::IREE::HAL
