// Copyright 2026 Boyd Johnson
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROCKET_HAL_DRIVER_CTS_CONV_TEST_HARNESS_H_
#define ROCKET_HAL_DRIVER_CTS_CONV_TEST_HARNESS_H_

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "iree/hal/api.h"

namespace rocket::testing {

// Bit-exact fp16 <-> fp32 conversions, exposed (not just used internally by
// RunFp16Conv2d) so other callers -- e.g. a fixture loader interpreting raw
// .npy fp16 data -- can reuse a single tested implementation instead of a
// second copy.
uint16_t F32ToF16(float value);
float F16ToF32(uint16_t bits);

// Logical convolution geometry accepted by the RKT1 Conv2D ukernel.
//
// Input and output tensors use dense NHWC order. Weights use dense HWCF
// order. Padding is symmetric: pad_top is also the bottom padding and
// pad_left is also the right padding, matching Rocket's current ConvShape.
struct Conv2dProblem {
  uint32_t input_height = 1;
  uint32_t input_width = 1;
  uint32_t input_channels = 1;
  uint32_t output_channels = 1;
  uint32_t kernel_height = 1;
  uint32_t kernel_width = 1;
  uint32_t stride = 1;
  uint32_t pad_top = 0;
  uint32_t pad_left = 0;

  uint32_t output_height() const;
  uint32_t output_width() const;
  size_t input_element_count() const;
  size_t weight_element_count() const;
  size_t output_element_count() const;
};

struct Conv2dResult {
  // Dense NHWC values decoded from the driver's output buffer.
  std::vector<float> actual;
  // Independently evaluated dense NHWC convolution of the f16-rounded
  // operands. Each result is rounded to f16 before being returned.
  std::vector<float> expected;
};

// Builds the header-prefixed `rocket-flatbuffer-v1` executable consumed by
// the driver. Exposed separately so schema construction can be regression
// tested on hosts without a Rocket device.
std::vector<uint8_t> BuildFp16Conv2dExecutable(const Conv2dProblem &problem);

// Runs a regular FP16 Conv2D from an RKT1 executable through the public IREE
// HAL API. `input` is dense NHWC and `weights` is dense HWCF. The driver,
// rather than this harness, is responsible for Rocket's NC1HWC2 input
// packing, blocked coefficient packing, and atomic-slot output compaction.
//
// Throws std::invalid_argument for an invalid fixture and std::runtime_error
// for a HAL/build failure.
Conv2dResult RunFp16Conv2d(iree_hal_device_t *device,
                           const Conv2dProblem &problem,
                           const std::vector<float> &input,
                           const std::vector<float> &weights);

// Allocates a device buffer of `byte_length` (device-optimal, mappable,
// dispatch-storage-capable -- the same params RunFp16Conv2d's own buffers
// use), with no initial content. Exposed so a caller can hold one buffer
// alive across multiple RunFp16Conv2dBound calls -- see that function's doc
// comment for why that matters. Caller owns the returned buffer and must
// release it with iree_hal_buffer_release.
iree_hal_buffer_t *AllocateBuffer(iree_hal_device_t *device,
                                  size_t byte_length);

// Byte offsets for one dispatch's four bindings (input, weights, bias,
// output; see command_buffer.rs's documented binding order) within a
// caller-owned shared buffer. No lengths are stored here -- they are always
// derivable from the Conv2dProblem at the point of use (element counts x
// sizeof(uint16_t) for fp16 storage, plus RunFp16Conv2d's fixed neutral bias
// size), so there is nothing to keep in sync with a second copy.
struct Conv2dBufferLayout {
  iree_device_size_t input_offset = 0;
  iree_device_size_t weight_offset = 0;
  iree_device_size_t bias_offset = 0;
  iree_device_size_t output_offset = 0;
};

// Computes a simple sequential, page-aligned layout for `problem`'s four
// bindings starting at `base_offset` within a shared buffer. Returns the
// layout and the offset immediately after its last region -- pass that back
// in as the next problem's `base_offset` to pack more than one problem into
// the same shared buffer without overlap.
std::pair<Conv2dBufferLayout, iree_device_size_t>
PackConv2dBufferLayout(const Conv2dProblem &problem,
                       iree_device_size_t base_offset);

// Like RunFp16Conv2d, but binds to a caller-owned `shared_buffer` at
// `layout`'s offsets instead of allocating and releasing fresh buffers of
// its own. Writes `input`/`weights` (quantized to f16) and a zeroed bias
// region into `shared_buffer` at `layout`'s offsets, dispatches, and reads
// the result back from the same buffer -- `shared_buffer` is never
// allocated or released by this function, so its other regions (e.g.
// another dispatch's operands, written by an earlier RunFp16Conv2dBound
// call against the same buffer) stay resident and untouched.
//
// This exists to reproduce the real compiled model's binding pattern for
// isolated regression testing: nonzero-offset slices of one long-lived
// hoisted constant/transient allocation, never freed between dispatches --
// unlike RunFp16Conv2d's convenience of a fresh, independently-scoped
// allocation every call. See DESIGN_NOTES.md (iree-rocket-design-spike),
// "Planck result: real weights and activations also fail to reproduce the
// poison" for why this distinction matters: every synthetic and real-data
// reproduction attempt using fresh per-dispatch buffers (RunFp16Conv2d) has
// failed to reproduce a persistent-CBUF corruption bug that the real
// compiled model (persistent, shared buffer allocation) hits reliably.
//
// Throws std::invalid_argument for an invalid fixture and std::runtime_error
// for a HAL/build failure, same as RunFp16Conv2d.
Conv2dResult RunFp16Conv2dBound(iree_hal_device_t *device,
                                iree_hal_buffer_t *shared_buffer,
                                const Conv2dProblem &problem,
                                const Conv2dBufferLayout &layout,
                                const std::vector<float> &input,
                                const std::vector<float> &weights);

// Like calling RunFp16Conv2dBound once per (problem, layout, input, weights)
// entry against the same `shared_buffer` -- except every dispatch is
// recorded into ONE command buffer (one begin/end, one queue_execute, one
// semaphore wait for the whole batch) instead of one command buffer per
// dispatch. This is what IREE's own stream partitioner does automatically
// for independent same-affinity dispatches with nothing forcing them apart
// (`stream.cmd.concurrent` grouped into one `stream.cmd.execute`), and it
// exercises rocket-hal-driver's multi-dispatch-per-command-buffer code path
// -- added in `0325b14`, "Modify runtime to allow concurrent dispatch" --
// that RunFp16Conv2d/RunFp16Conv2dBound never reach on their own. See
// DESIGN_NOTES.md, "Planck result: a minimal two-conv iree-compile module
// reproduces the poison": the real compiled reproduction grouped its two
// dispatches exactly this way, and no prior CTS harness attempt (all
// single-dispatch-per-command-buffer) had exercised this path at all.
//
// `problems`, `layouts`, `inputs`, and `weights` must all be the same size,
// index-aligned (entry i's problem/layout/input/weights belong together).
// Results are returned in the same order. Throws std::invalid_argument for
// an invalid fixture and std::runtime_error for a HAL/build failure, same
// as RunFp16Conv2d.
std::vector<Conv2dResult>
RunFp16Conv2dBatch(iree_hal_device_t *device, iree_hal_buffer_t *shared_buffer,
                   const std::vector<Conv2dProblem> &problems,
                   const std::vector<Conv2dBufferLayout> &layouts,
                   const std::vector<std::vector<float>> &inputs,
                   const std::vector<std::vector<float>> &weights);

// Reproduces the real compiled model's exact buffer layout for a two-
// dispatch poison/victim pair, rather than RunFp16Conv2dBound/
// RunFp16Conv2dBatch's per-problem-packed convention. `first`'s and
// `second`'s input/weight operands are each their own independent buffer
// (matching the real compiled Stream IR, where every operand tensor imports
// as its own `!stream.resource<external>`), but their BS/bias and raw NPU
// output regions are packed *tightly* adjacent in two shared buffers --
// first's bytes immediately followed by second's, no padding or alignment
// gap -- exactly matching the real module's single shared output/BS
// `!stream.resource<transient>` allocations. Bias sizing also matches the
// real unmodified transform.0.mlir's compiled convention: two bytes per
// output channel, not the other RunFp16Conv2d* functions' oversized 4096-
// byte neutral placeholder.
//
// This is the next reproduction attempt after RunFp16Conv2dBound,
// RunFp16Conv2dBatch, and the register-level oracle harness all failed to
// reproduce a persistent-CBUF corruption bug that
// poison_victim_isolation/conv.mlir (a real iree-compile'd module,
// iree-rocket-design-spike root) reproduces reliably. Every prior attempt
// packed each dispatch's *entire* input+weight+bias+output block
// separately, so the victim's output never sat anywhere near the poison's
// own output; this is the first to test output-buffer adjacency
// specifically. See DESIGN_NOTES.md, "Planck result: the 0325b14
// multi-dispatch code path is not the deciding factor either, but its
// investigation found the real lead."
//
// Both dispatches are recorded into one command buffer (like
// RunFp16Conv2dBatch), matching the real model's stream.cmd.concurrent
// grouping too. Returns {first_result, second_result}. Throws
// std::invalid_argument for an invalid fixture and std::runtime_error for a
// HAL/build failure, same as RunFp16Conv2d.
std::pair<Conv2dResult, Conv2dResult> RunFp16Conv2dPairWithAdjacentOutputs(
    iree_hal_device_t *device, const Conv2dProblem &first,
    const std::vector<float> &first_input,
    const std::vector<float> &first_weights, const Conv2dProblem &second,
    const std::vector<float> &second_input,
    const std::vector<float> &second_weights);

} // namespace rocket::testing

#endif // ROCKET_HAL_DRIVER_CTS_CONV_TEST_HARNESS_H_
