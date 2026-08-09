// Copyright 2026 Boyd Johnson
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "conv_test_harness.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
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

// Minimal ".npy" reader: fp16 ('<f2') or fp32 ('<f4'), C order only. Just
// enough to load fixtures this project's own extraction scripts produce
// (extract_poison_predecessor_capture.py, prepare_resnet128_context_probes.py
// in iree-rocket-design-spike) -- not a general numpy-format reader.
std::vector<float> LoadNpyFloat(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("failed to open " + path);
  }
  char magic[6] = {};
  file.read(magic, sizeof(magic));
  if (!file || std::memcmp(magic, "\x93NUMPY", sizeof(magic)) != 0) {
    throw std::runtime_error(path + " is not a .npy file");
  }
  uint8_t version[2] = {};
  file.read(reinterpret_cast<char *>(version), sizeof(version));
  uint32_t header_len = 0;
  if (version[0] == 1) {
    uint16_t header_len16 = 0;
    file.read(reinterpret_cast<char *>(&header_len16), sizeof(header_len16));
    header_len = header_len16;
  } else {
    file.read(reinterpret_cast<char *>(&header_len), sizeof(header_len));
  }
  std::string header(header_len, '\0');
  file.read(header.data(), header_len);
  if (!file) {
    throw std::runtime_error(path + " has a truncated .npy header");
  }
  if (header.find("'fortran_order': True") != std::string::npos) {
    throw std::runtime_error(path + " is fortran-order, unsupported");
  }

  const std::string descr_key = "'descr': '";
  const size_t descr_start = header.find(descr_key);
  if (descr_start == std::string::npos) {
    throw std::runtime_error(path + " header has no 'descr'");
  }
  const size_t descr_value_start = descr_start + descr_key.size();
  const size_t descr_end = header.find('\'', descr_value_start);
  const std::string descr =
      header.substr(descr_value_start, descr_end - descr_value_start);

  const std::string shape_key = "'shape': (";
  const size_t shape_start = header.find(shape_key);
  if (shape_start == std::string::npos) {
    throw std::runtime_error(path + " header has no 'shape'");
  }
  const size_t shape_value_start = shape_start + shape_key.size();
  const size_t shape_end = header.find(')', shape_value_start);
  std::istringstream shape_stream(
      header.substr(shape_value_start, shape_end - shape_value_start));
  size_t element_count = 1;
  std::string dimension;
  while (std::getline(shape_stream, dimension, ',')) {
    const size_t first = dimension.find_first_not_of(" \t");
    if (first == std::string::npos)
      continue;
    element_count *= std::stoul(dimension.substr(first));
  }

  std::vector<float> result(element_count);
  if (descr == "<f4") {
    file.read(reinterpret_cast<char *>(result.data()),
              element_count * sizeof(float));
  } else if (descr == "<f2") {
    std::vector<uint16_t> raw(element_count);
    file.read(reinterpret_cast<char *>(raw.data()),
              element_count * sizeof(uint16_t));
    for (size_t i = 0; i < element_count; ++i) {
      result[i] = F16ToF32(raw[i]);
    }
  } else {
    throw std::runtime_error(path + " has unsupported dtype '" + descr + "'");
  }
  if (!file) {
    throw std::runtime_error(path + " has truncated .npy data");
  }
  return result;
}

// Fixtures are deployed by copying a directory next to wherever the test
// binary is actually run from (this tree is scp'd to `planck` by hand, not
// installed/run through ctest, so there is no fixed relative-to-source-tree
// path that would resolve on the board). Override with
// ROCKET_POISON_FIXTURE_DIR; the default assumes `poison_predecessor_fixtures/`
// sits in the current working directory alongside the binary.
std::string PoisonFixtureDir() {
  if (const char *env = std::getenv("ROCKET_POISON_FIXTURE_DIR")) {
    return env;
  }
  return "poison_predecessor_fixtures";
}

class ConvSchemaHarnessTest : public ::testing::TestWithParam<Conv2dProblem> {};

std::string
ConvProblemName(const ::testing::TestParamInfo<Conv2dProblem> &info) {
  const Conv2dProblem &p = info.param;
  std::ostringstream name;
  name << "H" << p.input_height << "W" << p.input_width << "C"
       << p.input_channels << "F" << p.output_channels << "K" << p.kernel_height
       << "x" << p.kernel_width << "S" << p.stride << "P" << p.pad_top << "x"
       << p.pad_left;
  return name.str();
}

TEST(ConvSchemaHarnessHostTest, BuildsHeaderPrefixedRkt1Executable) {
  const Conv2dProblem problem{
      /*input_height=*/6,   /*input_width=*/7,
      /*input_channels=*/3, /*output_channels=*/2,
      /*kernel_height=*/3,  /*kernel_width=*/5,
      /*stride=*/2,         /*pad_top=*/1,
      /*pad_left=*/2};
  const std::vector<uint8_t> executable = BuildFp16Conv2dExecutable(problem);
  ASSERT_GT(executable.size(), 64u);
  EXPECT_EQ(std::string(executable.begin(), executable.begin() + 4), "RKT1");
  EXPECT_EQ(std::string(executable.begin() + 68, executable.begin() + 72),
            "RKT1");
  uint64_t content_size = 0;
  for (int i = 0; i < 8; ++i) {
    content_size |= static_cast<uint64_t>(executable[8 + i]) << (8 * i);
  }
  EXPECT_EQ(content_size, executable.size() - 64);
}

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
  size_t mismatch_count = 0;
  float maximum_error = 0.0f;
  std::ostringstream samples;
  for (size_t i = 0; i < result.actual.size(); ++i) {
    // Operands and the CPU result are explicitly rounded to f16. A small
    // tolerance still permits hardware's internal accumulation order to
    // differ without hiding packing/indexing failures.
    const float tolerance =
        std::max(0.002f, std::abs(result.expected[i]) * 0.002f);
    const float error = std::abs(result.actual[i] - result.expected[i]);
    if (error > tolerance) {
      maximum_error = std::max(maximum_error, error);
      if (mismatch_count < 8) {
        samples << "\n  [" << i << "] actual " << result.actual[i]
                << ", expected " << result.expected[i] << ", tolerance "
                << tolerance;
      }
      ++mismatch_count;
    }
  }
  EXPECT_EQ(mismatch_count, 0u)
      << mismatch_count << "/" << result.actual.size()
      << " dense NHWC elements differed; maximum error " << maximum_error
      << "; first mismatches:" << samples.str();
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
                      /*stride=*/1, /*pad_top=*/1, /*pad_left=*/2},
        Conv2dProblem{/*input_height=*/7, /*input_width=*/8,
                      /*input_channels=*/2, /*output_channels=*/2,
                      /*kernel_height=*/4, /*kernel_width=*/2,
                      /*stride=*/1, /*pad_top=*/0, /*pad_left=*/1},
        Conv2dProblem{/*input_height=*/7, /*input_width=*/8,
                      /*input_channels=*/3, /*output_channels=*/2,
                      /*kernel_height=*/3, /*kernel_width=*/3,
                      /*stride=*/2, /*pad_top=*/1, /*pad_left=*/1}),
    ConvProblemName);

// The exact hardware-job shapes and order DESIGN_NOTES.md (in
// iree-rocket-design-spike) isolated as a persistent-CBUF poison during
// ResNet-50 bring-up ("Planck result: the single Cout=64 1x1 job is the
// poison"): a Cin=Cout=64 1x1 convolution plans to the extreme CBUF split
// data_banks=11/weight_banks=1, and running it immediately before a
// Cin=Cout=128 3x3 convolution (banks 3/9) corrupts the second dispatch's
// upper output-channel half (Cout 64-127) in the real compiled model --
// confirmed there via `rocket-max128.vmfb`, which returns all-NaN logits,
// versus the known-good `rocket-max128-no64-1x1.vmfb` that keeps the 1x1 on
// CPU instead.
const Conv2dProblem kPoisonPredecessorProblem{
    /*input_height=*/128,   /*input_width=*/128,
    /*input_channels=*/64,  /*output_channels=*/64,
    /*kernel_height=*/1,    /*kernel_width=*/1,
    /*stride=*/1,           /*pad_top=*/0, /*pad_left=*/0};
const Conv2dProblem kPoisonVictimProblem{
    /*input_height=*/66,    /*input_width=*/66,
    /*input_channels=*/128, /*output_channels=*/128,
    /*kernel_height=*/3,    /*kernel_width=*/3,
    /*stride=*/1,           /*pad_top=*/0, /*pad_left=*/0};

// Checks the victim result against the documented poison signature: the
// lower half (Cout 0-63) stays correct, the upper half (Cout 64-127)
// diverges. Shared between the synthetic-data and real-weight variants below
// so both report the same lower/upper breakdown on failure.
void CheckPoisonVictimSignature(const Conv2dProblem &victim,
                                const Conv2dResult &victim_result) {
  ASSERT_EQ(victim_result.actual.size(), victim_result.expected.size());
  const uint32_t out_h = victim.output_height();
  const uint32_t out_w = victim.output_width();
  size_t lower_mismatches = 0;
  size_t upper_mismatches = 0;
  float upper_max_error = 0.0f;
  std::ostringstream upper_samples;
  for (uint32_t oy = 0; oy < out_h; ++oy) {
    for (uint32_t ox = 0; ox < out_w; ++ox) {
      for (uint32_t oc = 0; oc < victim.output_channels; ++oc) {
        const size_t index =
            ((static_cast<size_t>(oy) * out_w + ox) * victim.output_channels) +
            oc;
        const float tolerance = std::max(
            0.002f, std::abs(victim_result.expected[index]) * 0.002f);
        const float error =
            std::abs(victim_result.actual[index] - victim_result.expected[index]);
        if (error <= tolerance)
          continue;
        if (oc < 64) {
          ++lower_mismatches;
          continue;
        }
        ++upper_mismatches;
        upper_max_error = std::max(upper_max_error, error);
        if (upper_samples.tellp() < 512) {
          upper_samples << "\n  [y=" << oy << ", x=" << ox << ", c=" << oc
                        << "] actual " << victim_result.actual[index]
                        << ", expected " << victim_result.expected[index];
        }
      }
    }
  }

  EXPECT_EQ(lower_mismatches, 0u)
      << lower_mismatches
      << " Cout 0-63 elements diverged; unexpected -- this is not the "
         "documented poison signature, which leaves the lower half correct";
  EXPECT_EQ(upper_mismatches, 0u)
      << upper_mismatches << "/" << (static_cast<size_t>(out_h) * out_w * 64)
      << " Cout 64-127 elements diverged with max error " << upper_max_error
      << "; this is the documented poison-predecessor signature "
         "(DESIGN_NOTES.md, \"the single Cout=64 1x1 job is the poison\"); "
         "first mismatches:"
      << upper_samples.str();
}

// Two prior attempts to reproduce the poison in `iree-rocket-hal`'s
// register-level oracle harness (conv2d_oracle_hw.rs's
// poison_predecessor_sequential_regression_runs_every_case_before_failing,
// which calls submit_jobs/prep_bo directly) both passed -- first with
// uniform `Counting` data, then with diverse `Dense` data -- neither
// reproducing the corruption the full compiled model hits every time. This
// test instead goes through this driver's real dispatch path
// (executable_cache.rs's executable preparation, command_buffer.rs's
// dispatch recording, device.rs's queue_execute/semaphore signaling), the
// one register-level harness bypasses entirely. Each RunFp16Conv2d call
// below is its own iree_hal_command_buffer_t / queue_execute / semaphore
// round trip, matching the real failing model's structure: DESIGN_NOTES.md's
// max128 capture confirmed "seven Rocket command buffers and exactly one
// dispatch in every buffer," not one command buffer recording both.
//
// This also passed, with synthetic-but-diverse data (MakeInput/MakeWeights)
// -- see PoisonPredecessorCorruptsFollowingDispatchWithRealWeights below,
// which loads real ResNet weights/activations instead.
TEST(ConvSchemaHarnessRegressionTest, PoisonPredecessorCorruptsFollowingDispatch) {
  iree_hal_driver_t *driver = nullptr;
  iree_hal_device_t *device = nullptr;
  std::string error;
  if (!CreateDevice(&driver, &device, &error)) {
    GTEST_SKIP() << "Rocket device unavailable: " << error;
  }

  const Conv2dProblem &poison = kPoisonPredecessorProblem;
  const Conv2dProblem &victim = kPoisonVictimProblem;
  Conv2dResult victim_result;
  try {
    // The poison dispatch's own result is not checked -- DESIGN_NOTES.md
    // already established it is not itself corrupted; only its effect on
    // the dispatch immediately following it is under test.
    RunFp16Conv2d(device, poison, MakeInput(poison), MakeWeights(poison));
    victim_result =
        RunFp16Conv2d(device, victim, MakeInput(victim), MakeWeights(victim));
  } catch (...) {
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    throw;
  }
  iree_hal_device_release(device);
  iree_hal_driver_release(driver);

  CheckPoisonVictimSignature(victim, victim_result);
}

// Same sequence as PoisonPredecessorCorruptsFollowingDispatch, but with real
// ResNet-50 weights and real (forward-pass-computed, not procedurally
// generated) activations for both the poison and victim dispatches, loaded
// from fixture files instead of MakeInput/MakeWeights. Three synthetic-data
// reproduction attempts (this file's sibling test above, plus two register-
// level oracle patterns in iree-rocket-hal) have all failed to reproduce the
// corruption every real-weight compiled-model capture hits without
// exception -- see DESIGN_NOTES.md, "Planck result: the real driver dispatch
// path also fails to reproduce the poison." Real bit patterns are the one
// remaining untested variable.
//
// Fixtures: iree-rocket-design-spike's extract_poison_predecessor_capture.py
// (poison side) and prepare_resnet128_context_probes.py (victim side, the
// existing conv-64x64-c128-k3-resnet/layer-1 capture) produce
// {poison,victim}_{input,kernel}.npy; see PoisonFixtureDir's doc comment for
// where this test looks for them. Skips (not fails) if they are missing,
// like the device-unavailable case above -- this is diagnostic, not a
// standing CI regression, since it depends on a sibling repo's captured
// data rather than anything checked into this repo.
TEST(ConvSchemaHarnessRegressionTest,
    PoisonPredecessorCorruptsFollowingDispatchWithRealWeights) {
  iree_hal_driver_t *driver = nullptr;
  iree_hal_device_t *device = nullptr;
  std::string error;
  if (!CreateDevice(&driver, &device, &error)) {
    GTEST_SKIP() << "Rocket device unavailable: " << error;
  }

  const Conv2dProblem &poison = kPoisonPredecessorProblem;
  const Conv2dProblem &victim = kPoisonVictimProblem;
  const std::string fixture_dir = PoisonFixtureDir();
  std::vector<float> poison_input, poison_weights, victim_input,
      victim_weights;
  try {
    poison_input = LoadNpyFloat(fixture_dir + "/poison_input.npy");
    poison_weights = LoadNpyFloat(fixture_dir + "/poison_kernel.npy");
    victim_input = LoadNpyFloat(fixture_dir + "/victim_input.npy");
    victim_weights = LoadNpyFloat(fixture_dir + "/victim_kernel.npy");
  } catch (const std::exception &e) {
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    GTEST_SKIP() << "real-weight fixtures unavailable in \"" << fixture_dir
                 << "\": " << e.what()
                 << " -- set ROCKET_POISON_FIXTURE_DIR to point at them, or "
                    "run extract_poison_predecessor_capture.py and copy its "
                    "output next to this binary";
  }
  ASSERT_EQ(poison_input.size(), poison.input_element_count());
  ASSERT_EQ(poison_weights.size(), poison.weight_element_count());
  ASSERT_EQ(victim_input.size(), victim.input_element_count());
  ASSERT_EQ(victim_weights.size(), victim.weight_element_count());

  Conv2dResult victim_result;
  try {
    RunFp16Conv2d(device, poison, poison_input, poison_weights);
    victim_result = RunFp16Conv2d(device, victim, victim_input, victim_weights);
  } catch (...) {
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    throw;
  }
  iree_hal_device_release(device);
  iree_hal_driver_release(driver);

  CheckPoisonVictimSignature(victim, victim_result);
}

// Same real-weight sequence as
// PoisonPredecessorCorruptsFollowingDispatchWithRealWeights, but through
// RunFp16Conv2dBound instead of RunFp16Conv2d: both dispatches bind
// nonzero-offset regions of one shared iree_hal_buffer_t, allocated once and
// never released between them, instead of each getting a fresh,
// independently-scoped allocation that's fully released before the next
// begins. This is the one structural difference from the real compiled
// model that no earlier attempt in this file or in iree-rocket-hal's
// register-level oracle harness has controlled for: the real model binds
// poison/victim weights as offsets into one long-lived hoisted constant
// allocation, with activations flowing through the stream allocator's
// transient/scratch reuse -- never fresh per-dispatch allocations. See
// DESIGN_NOTES.md, "Planck result: real weights and activations also fail
// to reproduce the poison."
TEST(ConvSchemaHarnessRegressionTest,
    PoisonPredecessorCorruptsFollowingDispatchWithSharedBuffer) {
  iree_hal_driver_t *driver = nullptr;
  iree_hal_device_t *device = nullptr;
  std::string error;
  if (!CreateDevice(&driver, &device, &error)) {
    GTEST_SKIP() << "Rocket device unavailable: " << error;
  }

  const Conv2dProblem &poison = kPoisonPredecessorProblem;
  const Conv2dProblem &victim = kPoisonVictimProblem;
  const std::string fixture_dir = PoisonFixtureDir();
  std::vector<float> poison_input, poison_weights, victim_input,
      victim_weights;
  try {
    poison_input = LoadNpyFloat(fixture_dir + "/poison_input.npy");
    poison_weights = LoadNpyFloat(fixture_dir + "/poison_kernel.npy");
    victim_input = LoadNpyFloat(fixture_dir + "/victim_input.npy");
    victim_weights = LoadNpyFloat(fixture_dir + "/victim_kernel.npy");
  } catch (const std::exception &e) {
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    GTEST_SKIP() << "real-weight fixtures unavailable in \"" << fixture_dir
                 << "\": " << e.what()
                 << " -- set ROCKET_POISON_FIXTURE_DIR to point at them, or "
                    "run extract_poison_predecessor_capture.py and copy its "
                    "output next to this binary";
  }
  ASSERT_EQ(poison_input.size(), poison.input_element_count());
  ASSERT_EQ(poison_weights.size(), poison.weight_element_count());
  ASSERT_EQ(victim_input.size(), victim.input_element_count());
  ASSERT_EQ(victim_weights.size(), victim.weight_element_count());

  const auto poison_packed = PackConv2dBufferLayout(poison, 0);
  const Conv2dBufferLayout &poison_layout = poison_packed.first;
  const auto victim_packed =
      PackConv2dBufferLayout(victim, poison_packed.second);
  const Conv2dBufferLayout &victim_layout = victim_packed.first;
  const iree_device_size_t shared_buffer_bytes = victim_packed.second;

  iree_hal_buffer_t *shared_buffer = nullptr;
  Conv2dResult victim_result;
  try {
    shared_buffer = AllocateBuffer(device, shared_buffer_bytes);
    // Neither call releases shared_buffer between them -- both the poison
    // dispatch's own operands and the victim's stay resident in it exactly
    // like the real model's hoisted constant allocation, for as long as
    // shared_buffer itself is alive.
    RunFp16Conv2dBound(device, shared_buffer, poison, poison_layout,
                       poison_input, poison_weights);
    victim_result = RunFp16Conv2dBound(
        device, shared_buffer, victim, victim_layout, victim_input,
        victim_weights);
  } catch (...) {
    if (shared_buffer)
      iree_hal_buffer_release(shared_buffer);
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    throw;
  }
  iree_hal_buffer_release(shared_buffer);
  iree_hal_device_release(device);
  iree_hal_driver_release(driver);

  CheckPoisonVictimSignature(victim, victim_result);
}

// Same real weights/activations and shared persistent buffer as
// PoisonPredecessorCorruptsFollowingDispatchWithSharedBuffer, but through
// RunFp16Conv2dBatch instead of two RunFp16Conv2dBound calls: both
// dispatches are now recorded into ONE command buffer (one queue_execute,
// one semaphore wait for the pair) instead of each getting its own command
// buffer/queue_execute/semaphore round trip. This is the one remaining
// structural difference between every prior attempt in this file (all
// single-dispatch-per-command-buffer, even the shared-buffer variant above)
// and the real minimal iree-compile'd reproduction in
// poison_victim_isolation/, whose Stream IR groups both dispatches into one
// stream.cmd.concurrent region -- see DESIGN_NOTES.md, "Planck result: a
// minimal two-conv iree-compile module reproduces the poison." It exercises
// rocket-hal-driver's multi-dispatch-per-command-buffer code path added in
// `0325b14` ("Modify runtime to allow concurrent dispatch"), which no
// earlier test in this file had reached at all.
TEST(ConvSchemaHarnessRegressionTest,
    PoisonPredecessorCorruptsFollowingDispatchAsOneCommandBufferBatch) {
  iree_hal_driver_t *driver = nullptr;
  iree_hal_device_t *device = nullptr;
  std::string error;
  if (!CreateDevice(&driver, &device, &error)) {
    GTEST_SKIP() << "Rocket device unavailable: " << error;
  }

  const Conv2dProblem &poison = kPoisonPredecessorProblem;
  const Conv2dProblem &victim = kPoisonVictimProblem;
  const std::string fixture_dir = PoisonFixtureDir();
  std::vector<float> poison_input, poison_weights, victim_input,
      victim_weights;
  try {
    poison_input = LoadNpyFloat(fixture_dir + "/poison_input.npy");
    poison_weights = LoadNpyFloat(fixture_dir + "/poison_kernel.npy");
    victim_input = LoadNpyFloat(fixture_dir + "/victim_input.npy");
    victim_weights = LoadNpyFloat(fixture_dir + "/victim_kernel.npy");
  } catch (const std::exception &e) {
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    GTEST_SKIP() << "real-weight fixtures unavailable in \"" << fixture_dir
                 << "\": " << e.what()
                 << " -- set ROCKET_POISON_FIXTURE_DIR to point at them, or "
                    "run extract_poison_predecessor_capture.py and copy its "
                    "output next to this binary";
  }
  ASSERT_EQ(poison_input.size(), poison.input_element_count());
  ASSERT_EQ(poison_weights.size(), poison.weight_element_count());
  ASSERT_EQ(victim_input.size(), victim.input_element_count());
  ASSERT_EQ(victim_weights.size(), victim.weight_element_count());

  const auto poison_packed = PackConv2dBufferLayout(poison, 0);
  const Conv2dBufferLayout &poison_layout = poison_packed.first;
  const auto victim_packed =
      PackConv2dBufferLayout(victim, poison_packed.second);
  const Conv2dBufferLayout &victim_layout = victim_packed.first;
  const iree_device_size_t shared_buffer_bytes = victim_packed.second;

  iree_hal_buffer_t *shared_buffer = nullptr;
  std::vector<Conv2dResult> results;
  try {
    shared_buffer = AllocateBuffer(device, shared_buffer_bytes);
    results = RunFp16Conv2dBatch(
        device, shared_buffer, {poison, victim},
        {poison_layout, victim_layout}, {poison_input, victim_input},
        {poison_weights, victim_weights});
  } catch (...) {
    if (shared_buffer)
      iree_hal_buffer_release(shared_buffer);
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    throw;
  }
  iree_hal_buffer_release(shared_buffer);
  iree_hal_device_release(device);
  iree_hal_driver_release(driver);

  ASSERT_EQ(results.size(), 2u);
  CheckPoisonVictimSignature(victim, results[1]);
}

// Same real weights/activations as every other real-weight test above, but
// through RunFp16Conv2dPairWithAdjacentOutputs: poison's and victim's raw
// NPU output (and BS) land tightly adjacent in one shared buffer -- no
// input/weight bytes between them -- exactly matching the real compiled
// module's Stream IR layout, found by reading poison_victim_isolation/'s
// generated stream.mlir directly. This is the one buffer-layout property no
// earlier CTS attempt has matched: RunFp16Conv2dBound/RunFp16Conv2dBatch
// pack each dispatch's *entire* four-binding block together, so victim's
// output never sat near poison's own output in any of them. See
// DESIGN_NOTES.md, "Planck result: the 0325b14 multi-dispatch code path is
// not the deciding factor either, but its investigation found the real
// lead."
TEST(ConvSchemaHarnessRegressionTest,
    PoisonPredecessorCorruptsFollowingDispatchWithAdjacentOutputs) {
  iree_hal_driver_t *driver = nullptr;
  iree_hal_device_t *device = nullptr;
  std::string error;
  if (!CreateDevice(&driver, &device, &error)) {
    GTEST_SKIP() << "Rocket device unavailable: " << error;
  }

  const Conv2dProblem &poison = kPoisonPredecessorProblem;
  const Conv2dProblem &victim = kPoisonVictimProblem;
  const std::string fixture_dir = PoisonFixtureDir();
  std::vector<float> poison_input, poison_weights, victim_input,
      victim_weights;
  try {
    poison_input = LoadNpyFloat(fixture_dir + "/poison_input.npy");
    poison_weights = LoadNpyFloat(fixture_dir + "/poison_kernel.npy");
    victim_input = LoadNpyFloat(fixture_dir + "/victim_input.npy");
    victim_weights = LoadNpyFloat(fixture_dir + "/victim_kernel.npy");
  } catch (const std::exception &e) {
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    GTEST_SKIP() << "real-weight fixtures unavailable in \"" << fixture_dir
                 << "\": " << e.what()
                 << " -- set ROCKET_POISON_FIXTURE_DIR to point at them, or "
                    "run extract_poison_predecessor_capture.py and copy its "
                    "output next to this binary";
  }
  ASSERT_EQ(poison_input.size(), poison.input_element_count());
  ASSERT_EQ(poison_weights.size(), poison.weight_element_count());
  ASSERT_EQ(victim_input.size(), victim.input_element_count());
  ASSERT_EQ(victim_weights.size(), victim.weight_element_count());

  Conv2dResult victim_result;
  try {
    const auto results = RunFp16Conv2dPairWithAdjacentOutputs(
        device, poison, poison_input, poison_weights, victim, victim_input,
        victim_weights);
    victim_result = results.second;
  } catch (...) {
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
    throw;
  }
  iree_hal_device_release(device);
  iree_hal_driver_release(driver);

  CheckPoisonVictimSignature(victim, victim_result);
}

} // namespace
} // namespace rocket::testing
