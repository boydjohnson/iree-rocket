// Copyright 2026 Boyd Johnson
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROCKET_HAL_DRIVER_CTS_CONV_TEST_HARNESS_H_
#define ROCKET_HAL_DRIVER_CTS_CONV_TEST_HARNESS_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "iree/hal/api.h"

namespace rocket::testing {

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

} // namespace rocket::testing

#endif // ROCKET_HAL_DRIVER_CTS_CONV_TEST_HARNESS_H_
