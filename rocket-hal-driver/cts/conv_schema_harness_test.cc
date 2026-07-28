// Copyright 2026 Boyd Johnson
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "conv_test_harness.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "iree/async/util/proactor_pool.h"
#include "iree/base/threading/numa.h"
#include "gtest/gtest.h"

extern "C" iree_status_t
iree_hal_rocket_driver_module_register(iree_hal_driver_registry_t *registry);

namespace rocket::testing {
namespace {

bool CreateDevice(iree_hal_driver_t **out_driver,
                  iree_hal_device_t **out_device, std::string *out_error) {
  iree_status_t status = iree_hal_rocket_driver_module_register(
      iree_hal_driver_registry_default());
  if (iree_status_is_already_exists(status)) {
    iree_status_free(status);
    status = iree_ok_status();
  }

  iree_hal_driver_t *driver = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_driver_registry_try_create(
        iree_hal_driver_registry_default(), iree_make_cstring_view("rocket"),
        iree_allocator_system(), &driver);
  }

  iree_async_proactor_pool_t *proactor_pool = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_async_proactor_pool_create(
        iree_numa_node_count(), /*node_ids=*/nullptr,
        iree_async_proactor_pool_options_default(), iree_allocator_system(),
        &proactor_pool);
  }

  iree_hal_device_t *device = nullptr;
  if (iree_status_is_ok(status)) {
    iree_hal_device_create_params_t params =
        iree_hal_device_create_params_default();
    params.proactor_pool = proactor_pool;
    status = iree_hal_driver_create_default_device(
        driver, &params, iree_allocator_system(), &device);
  }
  if (proactor_pool)
    iree_async_proactor_pool_release(proactor_pool);

  if (!iree_status_is_ok(status)) {
    iree_allocator_t allocator = iree_allocator_system();
    char *message = nullptr;
    iree_host_size_t length = 0;
    iree_status_to_string(status, &allocator, &message, &length);
    *out_error =
        message ? std::string(message, length) : std::string("(no message)");
    if (message)
      iree_allocator_free(allocator, message);
    iree_status_free(status);
    if (driver)
      iree_hal_driver_release(driver);
    return false;
  }
  *out_driver = driver;
  *out_device = device;
  return true;
}

std::vector<float> MakeInput(const Conv2dProblem &problem) {
  std::vector<float> values(problem.input_element_count());
  for (size_t i = 0; i < values.size(); ++i) {
    // Exactly representable binary fractions keep the CPU route deterministic.
    values[i] = static_cast<float>(static_cast<int>(i % 9) - 4) * 0.125f;
  }
  return values;
}

std::vector<float> MakeWeights(const Conv2dProblem &problem) {
  std::vector<float> values(problem.weight_element_count());
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<float>(static_cast<int>(i % 5) - 2) * 0.25f;
  }
  return values;
}

class ConvSchemaHarnessTest : public ::testing::TestWithParam<Conv2dProblem> {};

TEST_P(ConvSchemaHarnessTest, MatchesIndependentDenseReference) {
  iree_hal_driver_t *driver = nullptr;
  iree_hal_device_t *device = nullptr;
  std::string error;
  if (!CreateDevice(&driver, &device, &error)) {
    GTEST_SKIP() << "Rocket device unavailable: " << error;
  }

  const Conv2dProblem &problem = GetParam();
  Conv2dResult result;
  try {
    result = RunFp16Conv2d(device, problem, MakeInput(problem),
                           MakeWeights(problem));
  } catch (...) {
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    throw;
  }
  iree_hal_device_release(device);
  iree_hal_driver_release(driver);

  ASSERT_EQ(result.actual.size(), result.expected.size());
  for (size_t i = 0; i < result.actual.size(); ++i) {
    // Operands and the CPU result are explicitly rounded to f16. A small
    // tolerance still permits hardware's internal accumulation order to
    // differ without hiding packing/indexing failures.
    const float tolerance =
        std::max(0.002f, std::abs(result.expected[i]) * 0.002f);
    EXPECT_NEAR(result.actual[i], result.expected[i], tolerance)
        << "dense NHWC output element " << i;
  }
}

INSTANTIATE_TEST_SUITE_P(
    KernelStrideAndPadding, ConvSchemaHarnessTest,
    ::testing::Values(
        Conv2dProblem{/*input_height=*/4, /*input_width=*/5,
                      /*input_channels=*/1, /*output_channels=*/1,
                      /*kernel_height=*/1, /*kernel_width=*/1,
                      /*stride=*/1, /*pad_top=*/0, /*pad_left=*/0},
        Conv2dProblem{/*input_height=*/6, /*input_width=*/7,
                      /*input_channels=*/3, /*output_channels=*/2,
                      /*kernel_height=*/3, /*kernel_width=*/3,
                      /*stride=*/1, /*pad_top=*/1, /*pad_left=*/1},
        Conv2dProblem{/*input_height=*/8, /*input_width=*/9,
                      /*input_channels=*/3, /*output_channels=*/3,
                      /*kernel_height=*/3, /*kernel_width=*/5,
                      /*stride=*/2, /*pad_top=*/1, /*pad_left=*/2},
        Conv2dProblem{/*input_height=*/7, /*input_width=*/8,
                      /*input_channels=*/2, /*output_channels=*/2,
                      /*kernel_height=*/4, /*kernel_width=*/2,
                      /*stride=*/2, /*pad_top=*/0, /*pad_left=*/1}));

} // namespace
} // namespace rocket::testing
