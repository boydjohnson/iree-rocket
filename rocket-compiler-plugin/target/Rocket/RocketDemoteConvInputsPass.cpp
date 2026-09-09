// Copyright 2026
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Demotes the input operands of an all-f32 named 2-D convolution or matmul to
// f16, leaving the accumulator at f32 -- Rocket's ABI is f16-in/f32-accumulate
// (see call_rocket_dynamic_conv2d in the transform spec), while models
// commonly arrive as plain f32 from ONNX/torch import.
//
// The registered name still says "conv" because it is what the spec, the
// tests and the docs all reference; the matmul case was added later and the
// mechanism is identical.
//
// This replaces `iree-global-opt-demote-contraction-inputs
// {type=f16 operation=conv}`, which this spec used to call and which is
// silently wrong for any strided or dilated convolution. That pass rebuilds
// the named op with `linalg::getPrunedAttributeList(namedOp)`
// (DemoteContractionInputs.cpp), and that helper elides
// `op.getAttributeNames()` -- which for a named convolution *includes*
// `strides` and `dilations`. The rebuilt op carries neither, so both silently
// fall back to 1.
//
// On MobileNetV2 that turned the stride-2 stem conv (1x225x225x3 ->
// 1x112x112x48) into a nominal stride-1 conv reading only a 114x114 corner of
// its input. It is a correctness bug on its own -- the IR is wrong before any
// Rocket matcher runs, so the op is miscomputed even when it stays on the CPU
// -- and it also let @match_dynamic_conv2d_3x3 (which requires strides == 1)
// claim a convolution it was never meant to, dispatching it to the NPU with
// the wrong stride. Upstream's test only covers `strides = dense<1>`, where
// the loss is invisible.
//
// The convolutions handled here mirror the replaced pass's `operation=conv`
// list exactly, so that part is a behaviour-preserving swap apart from the
// attributes. Depthwise convs are still not demoted (see below).
//
// `linalg.matmul` was added 2026-09-04, and not for matching -- the spec's
// matcher already claimed f32 matmuls, because @call_rocket_matmul narrowed
// both operands to f16 itself. That narrowing is the problem: it lives inside
// a `util.func` that is never inlined (every dispatch it forms is named
// `call_rocket_matmul_dispatch_N`), so the constant weights are invisible to
// const-expr hoisting and the truncf runs as a CPU dispatch on **every
// inference** -- 1.79M elements for MobileNetV2's classifier, into a fresh
// transient buffer, which then defeats the runtime's packed-coefficient cache
// as well (all misses, all `miss (new)`; see ISSUES.md P6). Demoting here
// instead puts the truncf in the caller, next to the constant, where
// hoist-into-globals and const-eval fold it into an initializer exactly as
// they already do for every convolution's weights.
//
// `indexing_maps` and `cast` join `strides`/`dilations` in the carried-over
// list for the same reason those two are there: `getPrunedAttributeList`
// elides every inherent attribute name, and for `linalg.matmul` the indexing
// maps are precisely what distinguishes a plain matmul from a transposed or
// broadcasting one. Dropping them would rebuild a transposed matmul as an
// untransposed one -- the same silent miscompile the strides bug was.
//
// Depthwise was tried and reverted 2026-09-01, and the reason has since been
// narrowed twice. Demoting it does let three of MobileNetV2 **static-int8**'s
// stride-2 depthwise convolutions match (18 -> 21 offloaded dispatch sites),
// and that model is then *wrong*: max|err| 3.5 on the logits with top-1
// incorrect on every input measured, against 0.36 for the same f16 demotion
// run entirely on the CPU. It is not the convolutions -- each of the three is
// exact to f16 epsilon in isolation, with and without a tensor.pad producer,
// and two of them sharing one dynamic executable is exact too. Bisecting by
// channel bound, offloading the 144-channel one alone is fine and adding the
// 192-channel one breaks it. That is a command buffer mixing fp16 depthwise
// with int8 dispatches, which is ISSUES.md C8, not a property of depthwise.
//
// On the plain fp16 model it is correct: 44 sites against 37, max|err| 0.0500
// vs 0.0192 on a CPU f32 reference, top-1 and top-5 stable, byte-identical
// over five consecutive runs (measured 2026-09-04). It is simply *slower* --
// 186 ms against 148 -- because a depthwise convolution is the cheapest op in
// the model per byte moved and loses to the per-dispatch layout round trip.
// ISSUES.md P7 has the full accounting and what would change it. So this is
// still the right default, but for a performance reason on one model and a
// correctness reason on the other; do not read it as "depthwise is broken".
//
// **Re-measured 2026-09-07, and the gap is a quarter of what it was.** The
// 186-vs-148 above was taken before M2's scratch pool and before ReLU6
// fusion. On the current baseline, adding the two lines below gives 44 sites
// against 37 and:
//
//   133.0 ms   37 sites
//   142.5 ms   44 sites, depthwise clamps on the CPU        1.071x
//   140.0 ms   44 sites, depthwise ReLU6 fused into BN      1.053x
//
// The fused arm is the interesting one: P7 suspected the recorded `outside`
// rise was partly the 17 depthwise ReLU6 clamps that offloading un-fuses,
// and it is -- but only 2.5 ms of the 9.5 ms gap. That machinery is in the
// tree and hardware-validated (`rocket-fuse-conv-relu6` handles
// `DepthwiseConv2DNchwChwOp`, `#rocket_dynamic_depthwise_relu6_target` and
// its stride-2 twin, `conv_fp16_bias_activation_hw`'s depthwise arms), so
// re-testing this costs exactly the two lines:
//
//   DemoteInputsToF16<linalg::DepthwiseConv2DNhwcHwcOp>,
//   DemoteInputsToF16<linalg::DepthwiseConv2DNchwChwOp>,
//
// plus their PromoteInputsToF32 counterparts. What is left of the gap is
// P7's items 2-4: the explicit pad IREE materializes as its own dispatch,
// the DEPTHWISE_TO_DENSE_QUIESCENCE dwell, and the Cin 512 matcher cap.
// Accuracy at 44 sites is max|diff| 0.0320 against a --no-offload CPU arm
// (0.0184 at 37), top-1 and top-5 stable.
//
// **Re-measured 2026-09-08 with the driver chain (P2 step 2) on, and it
// changes nothing**: 130.5 vs 137.0 ms at taskset -c 4-7 (1.05x), 108 vs
// 127.5 with --task_topology_cpu_ids=4,5,6,7 (1.18x), chain on and off
// within a millisecond of each other. The chain takes the same single edge
// in both builds: every offloaded depthwise reads its input through the
// explicit tensor.pad, which is a CPU dispatch, and the seven convolutions
// cost 60% of the CPU time they replace in NPU time alone at 200 MHz. So the
// lever P7 ranked first is worth ~0 here until the pad folds into the
// dispatch and the residual add leaves the CPU; ISSUES.md P7 has the edge
// census and the phase deltas. The demote stays off.
//
// **Re-measured 2026-09-09 at 600 MHz, after ISSUES.md M2 was resolved.**
// M2 was the clock this file's 2026-09-08 verdict named as the binding term,
// so it was the one input that had actually changed. Governor `performance`,
// four interleaved passes, medians, against a --no-offload arm built by the
// same pipeline:
//
//                        base(37)   dw(44)   nooff    dw vs base
//   default topo 200 MHz    99.4     99.8    108.0    1.004x slower
//   default topo 600 MHz    87.5     81.0    108.0    1.080x FASTER
//   four workers 200 MHz    83.1     93.2     54.5    1.122x slower
//   four workers 600 MHz    73.1     74.7     54.5    1.022x slower
//
// The clock is worth 8-10 points to the depthwise arm in *both* allocations,
// which is what P7 predicted, and it is enough to flip the sign at the
// default allocation. max|diff| vs --no-offload 0.0051 (dw) / 0.0050 (base),
// top-1 stable, zero faults in ~60 runs.
//
// **The demote still stays off, but the reason has moved.** It is no longer
// "depthwise loses to the round trip": at four workers --no-offload is
// 54.5 ms against a best NPU arm of 73.1, so the CPU is 1.34x faster and the
// whole fp16 offload on this model is underwater. The CPU baseline nearly
// doubles from four workers (108.0 -> 54.5) while every NPU arm gains 8-14%.
// Whether this demote is on is a detail inside a losing trade, and the 81.0
// vs 108.0 row that appears to beat the CPU only does so at an allocation
// that starves the CPU of workers -- taskset does NOT set IREE's worker
// count. Turn this on when the offload itself is competitive at four
// workers, not before.
//
// **Two corrections to the paragraph above this one.** The "same single
// edge" census is stale: compaction now skips 67 dense output writes in the
// 37-site build and 464 in the 44-site one, because the residual-add-on-NPU
// and lazy-compaction work landed the same day that census was taken.
// And do not measure any of this under governor `ondemand`: it read the
// four-worker control at 1.078x against the documented 1.18x and gave the
// default-topology control the wrong *sign*. Under `performance` both
// controls reproduce.
//
// Anything left alone is safe: an op that stays f32 fails the matchers' f16
// typing and goes to the CPU, and RocketPromoteUnclaimedConvInputsPass gives
// f32 back to anything demoted that the match loop then declines.

#include <cstdlib>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

#include <array>

namespace mlir::iree_compiler::IREE::HAL {
namespace {

// The named-op attributes `getPrunedAttributeList` drops on the floor. Each is
// optional on the op, so a missing one means "already the default" and is
// simply not carried over -- which is also why naming an attribute an op does
// not have (`strides` on a matmul, `indexing_maps` on a convolution) costs
// nothing.
constexpr std::array<StringRef, 4> kShapeDefiningAttrNames = {
    "strides", "dilations", "indexing_maps", "cast"};

// Marks what this pass rewrote, so RocketPromoteUnclaimedConvInputsPass can
// undo exactly its own work on the convolutions the matchers then decline --
// and nothing else. A convolution a model authored in f16 itself, or any
// truncf a model wrote by hand, carries no tag and is left alone.
constexpr StringLiteral kDemotedAttrName = "rocket.f16_demoted";

// Elementwise truncf of a whole tensor, as a linalg.generic -- the same shape
// of rewrite the upstream pass emits, so dispatch formation folds it into the
// producer exactly as before.
Value truncateToF16(PatternRewriter &rewriter, Location loc, Value input) {
  auto inputType = cast<RankedTensorType>(input.getType());
  Type f16 = rewriter.getF16Type();
  auto resultType =
      RankedTensorType::get(inputType.getShape(), f16, inputType.getEncoding());
  SmallVector<AffineMap> maps(
      2, rewriter.getMultiDimIdentityMap(inputType.getRank()));
  SmallVector<utils::IteratorType> iteratorTypes(inputType.getRank(),
                                                 utils::IteratorType::parallel);
  Value empty = tensor::EmptyOp::create(
      rewriter, loc, tensor::getMixedSizes(rewriter, loc, input), f16);
  return linalg::GenericOp::create(
             rewriter, loc, TypeRange{resultType}, ValueRange{input},
             ValueRange{empty}, maps, iteratorTypes,
             [&](OpBuilder &b, Location loc, ValueRange args) {
               Value truncated = arith::TruncFOp::create(b, loc, f16, args[0]);
               linalg::YieldOp::create(b, loc, truncated);
             })
      ->getResult(0);
}

// Tags an op this pass created or rewrote. See kDemotedAttrName.
void markDemoted(Operation *op, PatternRewriter &rewriter) {
  op->setAttr(kDemotedAttrName, rewriter.getUnitAttr());
}

// The f32 zero a static, zero-filled `tensor.pad` fills with, or null for
// any other pad (dynamic amounts, a nonzero or non-constant fill).
bool isStaticZeroPad(tensor::PadOp pad) {
  if (!pad.getLow().empty() || !pad.getHigh().empty()) {
    return false;
  }
  Value fill = pad.getConstantPaddingValue();
  auto constant = fill ? fill.getDefiningOp<arith::ConstantOp>() : nullptr;
  auto value = constant ? dyn_cast<FloatAttr>(constant.getValue()) : nullptr;
  return value && value.getValue().isZero();
}

// Demotes `value`, the convolution's operand, to f16.
//
// Ordinarily that is one truncf generic over the operand. A zero pad in
// front of the operand -- the explicit "same" padding every imported 3x3
// convolution arrives with, possibly under the channels-last collapse -- is
// demoted *through* instead: the truncf goes on the pad's source and the pad
// is rebuilt in f16. Truncating and zero-padding commute exactly, and the
// order matters for two later passes. rocket-fold-conv-pad and the pad-1
// matchers want `pad -> collapse -> conv` with nothing between, which is the
// f16 import's spelling and lets the CNA pad instead of a CPU copy; and the
// truncf then sits directly on the producer's result, where the producer's
// own f16 -> f32 widen cancels it and the two Rocket dispatches become
// adjacent (ISSUES.md P2). Left as `truncf(pad(x))`, neither happens: the
// pad is a CPU dispatch between every pair of convolutions.
//
// The rebuilt pad carries no tag (the DAG matchers compare whole attribute
// dictionaries); RocketPromoteUnclaimedConvInputsPass recognises it by the
// tagged truncf underneath.
Value demoteInput(PatternRewriter &rewriter, Location loc, Value value) {
  auto collapse = value.getDefiningOp<tensor::CollapseShapeOp>();
  Value padded = collapse ? collapse.getSrc() : value;
  if (auto pad = padded.getDefiningOp<tensor::PadOp>()) {
    auto sourceType = cast<RankedTensorType>(pad.getSource().getType());
    if (isStaticZeroPad(pad) && sourceType.getElementType().isF32()) {
      Type f16 = rewriter.getF16Type();
      Value source = truncateToF16(rewriter, loc, pad.getSource());
      markDemoted(source.getDefiningOp(), rewriter);
      Value zero = arith::ConstantOp::create(rewriter, loc, rewriter.getF16FloatAttr(0.0f));
      Value newPad = tensor::PadOp::create(
          rewriter, loc, cast<RankedTensorType>(pad.getType()).clone(f16), source,
          pad.getMixedLowPad(), pad.getMixedHighPad(), zero);
      if (!collapse) {
        return newPad;
      }
      return tensor::CollapseShapeOp::create(
          rewriter, loc, cast<RankedTensorType>(collapse.getType()).clone(f16), newPad,
          collapse.getReassociationIndices());
    }
  }
  Value demoted = truncateToF16(rewriter, loc, value);
  markDemoted(demoted.getDefiningOp(), rewriter);
  return demoted;
}

// P7 experiment gate. The depthwise demote is off by default -- the scope
// comment above carries the accounting and the 2026-09-09 re-measurement at
// 600 MHz. `ROCKET_DEMOTE_DEPTHWISE=1` builds the 44-site arm from the same
// compiler binary, so both arms come out of one build and the A/B costs a
// recompile of the model rather than of the compiler.
//
// Verify the gate by dispatch-site count, not by trusting the env var: 37
// sites off, 44 on, the delta being rocket_dynamic_depthwise_relu6_executable
// (4) and its _s2 twin (3). `rocket-compiler audit` prints them. And build
// board arms with --llvmcpu-target-triple aarch64-linux-gnu, or the vmfb is
// x86 and every run dies with "HAL device `cpu_device` not found".
//
// Read once: a pass runs many times per compile.
static bool rocketDemoteDepthwiseEnabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("ROCKET_DEMOTE_DEPTHWISE");
    return value && llvm::StringRef(value) != "0";
  }();
  return enabled;
}

template <typename ContractionOpTy>
struct DemoteInputsToF16 : OpRewritePattern<ContractionOpTy> {
  using OpRewritePattern<ContractionOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(ContractionOpTy convOp,
                                PatternRewriter &rewriter) const override {
    // Only all-f32 operand sets, matching the pass this replaces: an op
    // already authored in f16 (or any mixed-precision one) is left alone.
    if (convOp->hasAttr(kDemotedAttrName)) {
      return failure();
    }
    Type f32 = rewriter.getF32Type();
    if (!llvm::all_of(convOp->getOperands(), [&](Value operand) {
          auto type = dyn_cast<RankedTensorType>(operand.getType());
          return type && type.getElementType() == f32;
        })) {
      return failure();
    }

    // Read the shape-defining attributes off the original op before it is
    // replaced. Named linalg ops keep these as inherent attributes, so this
    // reaches them whether or not they are stored as properties.
    SmallVector<NamedAttribute> attributes =
        linalg::getPrunedAttributeList(convOp);
    for (StringRef name : kShapeDefiningAttrNames) {
      if (Attribute attr = convOp->getAttr(name)) {
        attributes.emplace_back(rewriter.getStringAttr(name), attr);
      }
    }

    Location loc = convOp.getLoc();
    SmallVector<Value> demotedInputs;
    for (OpOperand *inputOperand : convOp.getDpsInputOperands()) {
      demotedInputs.push_back(demoteInput(rewriter, loc, inputOperand->get()));
    }
    auto demotedOp = rewriter.replaceOpWithNewOp<ContractionOpTy>(
        convOp, demotedInputs, convOp.getDpsInits(), attributes);
    markDemoted(demotedOp, rewriter);
    return success();
  }
};

struct RocketDemoteConvInputsPass
    : public PassWrapper<RocketDemoteConvInputsPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RocketDemoteConvInputsPass)

  StringRef getArgument() const final {
    return "rocket-demote-conv-inputs-to-f16";
  }
  StringRef getDescription() const final {
    return "Demotes all-f32 named 2-D convolution and matmul inputs to f16, "
           "keeping the f32 accumulator and preserving the shape-defining "
           "attributes getPrunedAttributeList elides.";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() final {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<DemoteInputsToF16<linalg::Conv2DOp>,
                 DemoteInputsToF16<linalg::Conv2DNchwFchwOp>,
                 DemoteInputsToF16<linalg::Conv2DNhwcHwcfOp>,
                 DemoteInputsToF16<linalg::Conv2DNhwcFhwcOp>,
                 DemoteInputsToF16<linalg::Conv2DNgchwFgchwOp>,
                 DemoteInputsToF16<linalg::Conv2DNgchwGfchwOp>,
                 DemoteInputsToF16<linalg::MatmulOp>>(context);
    if (rocketDemoteDepthwiseEnabled()) {
      patterns.add<DemoteInputsToF16<linalg::DepthwiseConv2DNhwcHwcOp>,
                   DemoteInputsToF16<linalg::DepthwiseConv2DNchwChwOp>>(context);
    }
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

static PassRegistration<RocketDemoteConvInputsPass> reg;

} // namespace
} // namespace mlir::iree_compiler::IREE::HAL
