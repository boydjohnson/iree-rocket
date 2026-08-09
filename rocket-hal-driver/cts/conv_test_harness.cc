// Copyright 2026 Boyd Johnson
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "conv_test_harness.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "iree/base/api.h"
#include "rocket_executable_def_builder.h"

namespace rocket::testing {

uint16_t F32ToF16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000u;
  int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
  uint32_t fraction = bits & 0x7fffffu;

  if ((bits & 0x7fffffffu) == 0)
    return static_cast<uint16_t>(sign);
  if (((bits >> 23) & 0xffu) == 0xffu) {
    return static_cast<uint16_t>(sign | 0x7c00u |
                                 (fraction == 0 ? 0 : 0x0200u));
  }
  if (exponent <= 0) {
    if (exponent < -10)
      return static_cast<uint16_t>(sign);
    fraction |= 0x800000u;
    const uint32_t shift = static_cast<uint32_t>(14 - exponent);
    uint32_t rounded = fraction >> shift;
    const uint32_t remainder = fraction & ((1u << shift) - 1);
    const uint32_t halfway = 1u << (shift - 1);
    if (remainder > halfway || (remainder == halfway && (rounded & 1))) {
      ++rounded;
    }
    return static_cast<uint16_t>(sign | rounded);
  }
  if (exponent >= 31)
    return static_cast<uint16_t>(sign | 0x7c00u);

  uint32_t rounded_fraction = fraction >> 13;
  const uint32_t remainder = fraction & 0x1fffu;
  if (remainder > 0x1000u || (remainder == 0x1000u && (rounded_fraction & 1))) {
    ++rounded_fraction;
    if (rounded_fraction == 0x400u) {
      rounded_fraction = 0;
      ++exponent;
      if (exponent == 31)
        return static_cast<uint16_t>(sign | 0x7c00u);
    }
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) |
                               rounded_fraction);
}

float F16ToF32(uint16_t bits) {
  const uint32_t sign = (bits >> 15) & 1;
  const uint32_t exponent = (bits >> 10) & 0x1f;
  const uint32_t fraction = bits & 0x3ff;
  if (exponent == 0 && fraction != 0) {
    const float subnormal =
        static_cast<float>(fraction) * std::ldexp(1.0f, -24);
    return sign ? -subnormal : subnormal;
  }
  uint32_t f32_bits = 0;
  if (exponent == 0) {
    f32_bits = sign << 31;
  } else if (exponent == 0x1f) {
    f32_bits = (sign << 31) | (0xffu << 23) | (fraction << 13);
  } else {
    f32_bits = (sign << 31) | ((exponent + 127 - 15) << 23) | (fraction << 13);
  }
  float value;
  std::memcpy(&value, &f32_bits, sizeof(value));
  return value;
}

iree_hal_buffer_t *AllocateBuffer(iree_hal_device_t *device,
                                  size_t byte_length) {
  iree_hal_buffer_params_t params;
  std::memset(&params, 0, sizeof(params));
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                 IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                 IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL;

  iree_hal_buffer_t *buffer = nullptr;
  iree_status_t status = iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(device), params,
      std::max<size_t>(byte_length, 1), &buffer);
  if (!iree_status_is_ok(status)) {
    char message[2048] = {};
    iree_host_size_t length = 0;
    iree_status_format(status, sizeof(message), message, &length);
    std::string text = "iree_hal_allocator_allocate_buffer failed: ";
    text.append(length == 0
                    ? std::string("(no message)")
                    : std::string(message,
                                  std::min<size_t>(length, sizeof(message) - 1)));
    iree_status_free(status);
    throw std::runtime_error(text);
  }
  return buffer;
}

namespace {

constexpr size_t kIreeFlatbufferHeaderSize = 64;
// The current FP16 BS/bias binding is hardware-layout storage rather than a
// logical dense tensor (see RunFp16Conv2d's own comment on this). Both the
// fresh-buffer-per-dispatch path and the shared-buffer path below reserve
// this same generously-sized neutral region rather than sizing it exactly.
constexpr iree_device_size_t kNeutralBiasBytes = 4096;

void ThrowStatus(iree_status_t status, const char *operation) {
  if (iree_status_is_ok(status))
    return;
  char message[2048] = {};
  iree_host_size_t length = 0;
  iree_status_format(status, sizeof(message), message, &length);
  std::string text = operation;
  text.append(" failed: ");
  text.append(length == 0
                  ? std::string("(no message)")
                  : std::string(message,
                                std::min<size_t>(length, sizeof(message) - 1)));
  iree_status_free(status);
  throw std::runtime_error(text);
}

std::vector<uint16_t> QuantizeToF16(const std::vector<float> &values) {
  std::vector<uint16_t> result;
  result.reserve(values.size());
  for (float value : values)
    result.push_back(F32ToF16(value));
  return result;
}

void StoreLe32(uint8_t *destination, uint32_t value) {
  for (int i = 0; i < 4; ++i) {
    destination[i] = static_cast<uint8_t>((value >> (i * 8)) & 0xff);
  }
}

void StoreLe64(uint8_t *destination, uint64_t value) {
  for (int i = 0; i < 8; ++i) {
    destination[i] = static_cast<uint8_t>((value >> (i * 8)) & 0xff);
  }
}

std::vector<uint8_t> BuildRkt1Executable(const Conv2dProblem &problem) {
  flatbuffers_builder_t builder;
  if (flatcc_builder_init(&builder) != 0) {
    throw std::runtime_error("failed to initialize RKT1 FlatBuffer builder");
  }

  std::vector<uint8_t> result;
  void *flatbuffer = nullptr;
  try {
    // FlatCC builds tables bottom-up. Child refs are created before the root
    // buffer is started by ExecutableDef_create_as_root below.
    const auto name = flatbuffers_string_create_str(&builder, "conv_test");
    if (!name) {
      throw std::runtime_error("failed to build RKT1 export name");
    }
    // Do not use Conv2DDef_create with a zero runtime_dimensions ref:
    // FlatCC's generated convenience function unconditionally tries to add
    // every vector argument, and adding a null optional-vector ref fails.
    // The field-wise API lets a fully static executable omit that vector.
    if (iree_hal_rocket_Conv2DDef_start(&builder) ||
        iree_hal_rocket_Conv2DDef_input_width_add(&builder,
                                                  problem.input_width) ||
        iree_hal_rocket_Conv2DDef_input_height_add(&builder,
                                                   problem.input_height) ||
        iree_hal_rocket_Conv2DDef_input_channels_add(&builder,
                                                     problem.input_channels) ||
        iree_hal_rocket_Conv2DDef_output_width_add(&builder,
                                                   problem.output_width()) ||
        iree_hal_rocket_Conv2DDef_output_height_add(&builder,
                                                    problem.output_height()) ||
        iree_hal_rocket_Conv2DDef_output_channels_add(
            &builder, problem.output_channels) ||
        iree_hal_rocket_Conv2DDef_weights_width_add(&builder,
                                                    problem.kernel_width) ||
        iree_hal_rocket_Conv2DDef_weights_height_add(&builder,
                                                     problem.kernel_height) ||
        iree_hal_rocket_Conv2DDef_stride_add(&builder, problem.stride) ||
        iree_hal_rocket_Conv2DDef_precision_add(
            &builder, iree_hal_rocket_Precision_FP16) ||
        iree_hal_rocket_Conv2DDef_pad_top_add(&builder, problem.pad_top) ||
        iree_hal_rocket_Conv2DDef_pad_left_add(&builder, problem.pad_left)) {
      throw std::runtime_error("failed to populate RKT1 Conv2DDef");
    }
    const auto conv = iree_hal_rocket_Conv2DDef_end(&builder);
    if (!conv) {
      throw std::runtime_error("failed to build RKT1 Conv2DDef");
    }
    const auto export_def = iree_hal_rocket_ExportDef_create(
        &builder, name, iree_hal_rocket_KernelDef_as_Conv2DDef(conv));
    if (!export_def) {
      throw std::runtime_error("failed to build RKT1 ExportDef");
    }
    const auto exports =
        iree_hal_rocket_ExportDef_vec_create(&builder, &export_def, 1);
    if (!exports) {
      throw std::runtime_error("failed to build RKT1 exports vector");
    }
    if (!iree_hal_rocket_ExecutableDef_create_as_root(&builder, exports)) {
      throw std::runtime_error("failed to build RKT1 executable root");
    }

    size_t flatbuffer_size = 0;
    flatbuffer = flatcc_builder_finalize_buffer(&builder, &flatbuffer_size);
    if (!flatbuffer) {
      throw std::runtime_error(
          "failed to finalize RKT1 convolution executable");
    }
    result.assign(kIreeFlatbufferHeaderSize + flatbuffer_size, 0);
    std::memcpy(result.data(), "RKT1", 4);
    StoreLe32(result.data() + 4, 0);
    StoreLe64(result.data() + 8, flatbuffer_size);
    std::memcpy(result.data() + kIreeFlatbufferHeaderSize, flatbuffer,
                flatbuffer_size);
  } catch (...) {
    if (flatbuffer)
      flatcc_builder_free(flatbuffer);
    flatcc_builder_clear(&builder);
    throw;
  }
  flatcc_builder_free(flatbuffer);
  flatcc_builder_clear(&builder);
  return result;
}

iree_hal_buffer_t *AllocateAndWrite(iree_hal_device_t *device,
                                    const void *source, size_t byte_length) {
  iree_hal_buffer_t *buffer = AllocateBuffer(device, byte_length);
  try {
    if (byte_length != 0) {
      ThrowStatus(iree_hal_buffer_map_write(buffer, 0, source, byte_length),
                  "iree_hal_buffer_map_write");
    }
  } catch (...) {
    iree_hal_buffer_release(buffer);
    throw;
  }
  return buffer;
}

// Maps, invalidates, copies out, and unmaps `byte_length` bytes of f16
// storage starting at `offset` in `buffer`. Shared by every function below
// that reads a dispatch's raw output back off the device.
std::vector<uint16_t> ReadBackF16(iree_hal_buffer_t *buffer,
                                  iree_device_size_t offset,
                                  iree_device_size_t byte_length) {
  std::vector<uint16_t> result(byte_length / sizeof(uint16_t));
  iree_hal_buffer_mapping_t mapping;
  ThrowStatus(iree_hal_buffer_map_range(buffer, IREE_HAL_MAPPING_MODE_SCOPED,
                                        IREE_HAL_MEMORY_ACCESS_READ, offset,
                                        byte_length, &mapping),
              "map output for readback");
  try {
    ThrowStatus(iree_hal_buffer_mapping_invalidate_range(
                    &mapping, 0, IREE_HAL_WHOLE_BUFFER),
                "invalidate output for readback");
    std::memcpy(result.data(), mapping.contents.data, byte_length);
  } catch (...) {
    iree_hal_buffer_unmap_range(&mapping);
    throw;
  }
  ThrowStatus(iree_hal_buffer_unmap_range(&mapping),
              "unmap output for readback");
  return result;
}

std::vector<float> DecodeF16(const std::vector<uint16_t> &values) {
  std::vector<float> result;
  result.reserve(values.size());
  for (uint16_t value : values)
    result.push_back(F16ToF32(value));
  return result;
}

std::vector<float> ReferenceConv2d(const Conv2dProblem &problem,
                                   const std::vector<uint16_t> &input,
                                   const std::vector<uint16_t> &weights) {
  std::vector<float> output(problem.output_element_count());
  const int32_t input_height = static_cast<int32_t>(problem.input_height);
  const int32_t input_width = static_cast<int32_t>(problem.input_width);
  for (uint32_t oy = 0; oy < problem.output_height(); ++oy) {
    for (uint32_t ox = 0; ox < problem.output_width(); ++ox) {
      for (uint32_t oc = 0; oc < problem.output_channels; ++oc) {
        float sum = 0.0f;
        for (uint32_t ky = 0; ky < problem.kernel_height; ++ky) {
          const int32_t iy = static_cast<int32_t>(oy * problem.stride + ky) -
                             static_cast<int32_t>(problem.pad_top);
          if (iy < 0 || iy >= input_height)
            continue;
          for (uint32_t kx = 0; kx < problem.kernel_width; ++kx) {
            const int32_t ix = static_cast<int32_t>(ox * problem.stride + kx) -
                               static_cast<int32_t>(problem.pad_left);
            if (ix < 0 || ix >= input_width)
              continue;
            for (uint32_t ic = 0; ic < problem.input_channels; ++ic) {
              const size_t input_index =
                  ((static_cast<size_t>(iy) * problem.input_width + ix) *
                       problem.input_channels +
                   ic);
              const size_t weight_index =
                  (((static_cast<size_t>(ky) * problem.kernel_width + kx) *
                        problem.input_channels +
                    ic) *
                       problem.output_channels +
                   oc);
              sum += F16ToF32(input[input_index]) *
                     F16ToF32(weights[weight_index]);
            }
          }
        }
        const size_t output_index =
            ((static_cast<size_t>(oy) * problem.output_width() + ox) *
                 problem.output_channels +
             oc);
        output[output_index] = F16ToF32(F32ToF16(sum));
      }
    }
  }
  return output;
}

// Records and submits one dispatch against four already-allocated,
// already-populated buffer regions, waiting for it to complete before
// returning. Shared by RunFp16Conv2d (fresh buffers, released right after)
// and RunFp16Conv2dBound (a caller-owned buffer, still alive after this
// returns) -- the two differ only in where their four regions come from and
// how long they live, not in how the dispatch itself is built and run.
void DispatchFp16Conv2d(iree_hal_device_t *device, const Conv2dProblem &problem,
                        iree_hal_buffer_t *input_buffer,
                        iree_device_size_t input_offset,
                        iree_device_size_t input_bytes,
                        iree_hal_buffer_t *weight_buffer,
                        iree_device_size_t weight_offset,
                        iree_device_size_t weight_bytes,
                        iree_hal_buffer_t *bias_buffer,
                        iree_device_size_t bias_offset,
                        iree_device_size_t bias_bytes,
                        iree_hal_buffer_t *output_buffer,
                        iree_device_size_t output_offset,
                        iree_device_size_t output_bytes) {
  const std::vector<uint8_t> executable_data = BuildRkt1Executable(problem);

  iree_hal_executable_cache_t *cache = nullptr;
  iree_hal_executable_t *executable = nullptr;
  iree_hal_command_buffer_t *command_buffer = nullptr;
  iree_hal_semaphore_t *semaphore = nullptr;
  try {
    ThrowStatus(iree_hal_executable_cache_create(
                    device, iree_make_cstring_view("conv-test"), &cache),
                "iree_hal_executable_cache_create");
    iree_hal_executable_params_t params;
    std::memset(&params, 0, sizeof(params));
    params.executable_format = iree_make_cstring_view("rocket-flatbuffer-v1");
    params.executable_data = iree_make_const_byte_span(executable_data.data(),
                                                       executable_data.size());
    ThrowStatus(iree_hal_executable_cache_prepare_executable(cache, &params,
                                                             &executable),
                "iree_hal_executable_cache_prepare_executable");

    ThrowStatus(iree_hal_command_buffer_create(
                    device, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
                    IREE_HAL_COMMAND_CATEGORY_DISPATCH,
                    IREE_HAL_QUEUE_AFFINITY_ANY, /*binding_capacity=*/0,
                    &command_buffer),
                "iree_hal_command_buffer_create");
    ThrowStatus(iree_hal_command_buffer_begin(command_buffer),
                "iree_hal_command_buffer_begin");

    iree_hal_buffer_ref_t refs[4] = {
        iree_hal_make_buffer_ref(input_buffer, input_offset, input_bytes),
        iree_hal_make_buffer_ref(weight_buffer, weight_offset, weight_bytes),
        iree_hal_make_buffer_ref(bias_buffer, bias_offset, bias_bytes),
        iree_hal_make_buffer_ref(output_buffer, output_offset, output_bytes),
    };
    iree_hal_buffer_ref_list_t bindings = {4, refs};
    iree_hal_dispatch_config_t config = {};
    config.workgroup_count[0] = 1;
    config.workgroup_count[1] = 1;
    config.workgroup_count[2] = 1;
    iree_hal_executable_function_t function = {};
    function.value = 0;
    ThrowStatus(
        iree_hal_command_buffer_dispatch(command_buffer, executable, function,
                                         config, iree_const_byte_span_empty(),
                                         bindings, IREE_HAL_DISPATCH_FLAG_NONE),
        "iree_hal_command_buffer_dispatch");
    ThrowStatus(iree_hal_command_buffer_end(command_buffer),
                "iree_hal_command_buffer_end");

    ThrowStatus(iree_hal_semaphore_create(device, IREE_HAL_QUEUE_AFFINITY_ANY,
                                          0, IREE_HAL_SEMAPHORE_FLAG_NONE,
                                          &semaphore),
                "iree_hal_semaphore_create");
    uint64_t signal_value = 1;
    iree_hal_semaphore_list_t signal_list = {1, &semaphore, &signal_value};
    ThrowStatus(iree_hal_device_queue_execute(
                    device, IREE_HAL_QUEUE_AFFINITY_ANY,
                    iree_hal_semaphore_list_empty(), signal_list,
                    command_buffer, iree_hal_buffer_binding_table_empty(),
                    IREE_HAL_EXECUTE_FLAG_NONE),
                "iree_hal_device_queue_execute");
    ThrowStatus(iree_hal_semaphore_wait(semaphore, signal_value,
                                        iree_infinite_timeout(),
                                        IREE_ASYNC_WAIT_FLAG_NONE),
                "iree_hal_semaphore_wait");
  } catch (...) {
    if (semaphore)
      iree_hal_semaphore_release(semaphore);
    if (command_buffer)
      iree_hal_command_buffer_release(command_buffer);
    if (executable)
      iree_hal_executable_release(executable);
    if (cache)
      iree_hal_executable_cache_release(cache);
    throw;
  }
  iree_hal_semaphore_release(semaphore);
  iree_hal_command_buffer_release(command_buffer);
  iree_hal_executable_release(executable);
  iree_hal_executable_cache_release(cache);
}

iree_device_size_t AlignUp(iree_device_size_t value,
                           iree_device_size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

// One dispatch's problem plus its four already-populated buffer regions,
// for DispatchFp16Conv2dBatch below.
struct BatchDispatchEntry {
  Conv2dProblem problem;
  iree_hal_buffer_t *input_buffer;
  iree_device_size_t input_offset;
  iree_device_size_t input_bytes;
  iree_hal_buffer_t *weight_buffer;
  iree_device_size_t weight_offset;
  iree_device_size_t weight_bytes;
  iree_hal_buffer_t *bias_buffer;
  iree_device_size_t bias_offset;
  iree_device_size_t bias_bytes;
  iree_hal_buffer_t *output_buffer;
  iree_device_size_t output_offset;
  iree_device_size_t output_bytes;
};

// Records every entry's dispatch into ONE command buffer -- one
// iree_hal_command_buffer_begin/end, one queue_execute, one semaphore wait
// for the whole batch -- instead of DispatchFp16Conv2d's one command buffer
// per dispatch. This is what IREE's own stream partitioner does for
// independent same-affinity dispatches with nothing forcing them apart
// (stream.cmd.concurrent grouped into one stream.cmd.execute), and it
// exercises rocket-hal-driver's multi-dispatch-per-command-buffer code path
// (`command_buffer::apply_ops` returning a `Vec<DispatchJob>`,
// `device::queue_execute` looping over all of them -- added in `0325b14`,
// "Modify runtime to allow concurrent dispatch") that DispatchFp16Conv2d
// never reaches, since it always records exactly one dispatch per command
// buffer. See DESIGN_NOTES.md (iree-rocket-design-spike), "Planck result: a
// minimal two-conv iree-compile module reproduces the poison" -- the real
// compiled reproduction grouped both dispatches this way, and no prior CTS
// harness attempt (all single-dispatch-per-command-buffer) had exercised
// this path at all.
void DispatchFp16Conv2dBatch(iree_hal_device_t *device,
                             const std::vector<BatchDispatchEntry> &entries) {
  iree_hal_executable_cache_t *cache = nullptr;
  iree_hal_command_buffer_t *command_buffer = nullptr;
  iree_hal_semaphore_t *semaphore = nullptr;
  std::vector<iree_hal_executable_t *> executables(entries.size(), nullptr);
  try {
    ThrowStatus(iree_hal_executable_cache_create(
                    device, iree_make_cstring_view("conv-test-batch"), &cache),
                "iree_hal_executable_cache_create");
    for (size_t i = 0; i < entries.size(); ++i) {
      const std::vector<uint8_t> executable_data =
          BuildRkt1Executable(entries[i].problem);
      iree_hal_executable_params_t params;
      std::memset(&params, 0, sizeof(params));
      params.executable_format = iree_make_cstring_view("rocket-flatbuffer-v1");
      params.executable_data = iree_make_const_byte_span(
          executable_data.data(), executable_data.size());
      ThrowStatus(iree_hal_executable_cache_prepare_executable(
                      cache, &params, &executables[i]),
                  "iree_hal_executable_cache_prepare_executable");
    }

    ThrowStatus(iree_hal_command_buffer_create(
                    device, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
                    IREE_HAL_COMMAND_CATEGORY_DISPATCH,
                    IREE_HAL_QUEUE_AFFINITY_ANY, /*binding_capacity=*/0,
                    &command_buffer),
                "iree_hal_command_buffer_create");
    ThrowStatus(iree_hal_command_buffer_begin(command_buffer),
                "iree_hal_command_buffer_begin");

    for (size_t i = 0; i < entries.size(); ++i) {
      const BatchDispatchEntry &e = entries[i];
      iree_hal_buffer_ref_t refs[4] = {
          iree_hal_make_buffer_ref(e.input_buffer, e.input_offset,
                                   e.input_bytes),
          iree_hal_make_buffer_ref(e.weight_buffer, e.weight_offset,
                                   e.weight_bytes),
          iree_hal_make_buffer_ref(e.bias_buffer, e.bias_offset,
                                   e.bias_bytes),
          iree_hal_make_buffer_ref(e.output_buffer, e.output_offset,
                                   e.output_bytes),
      };
      iree_hal_buffer_ref_list_t bindings = {4, refs};
      iree_hal_dispatch_config_t config = {};
      config.workgroup_count[0] = 1;
      config.workgroup_count[1] = 1;
      config.workgroup_count[2] = 1;
      iree_hal_executable_function_t function = {};
      function.value = 0;
      ThrowStatus(
          iree_hal_command_buffer_dispatch(
              command_buffer, executables[i], function, config,
              iree_const_byte_span_empty(), bindings,
              IREE_HAL_DISPATCH_FLAG_NONE),
          "iree_hal_command_buffer_dispatch");
    }
    ThrowStatus(iree_hal_command_buffer_end(command_buffer),
                "iree_hal_command_buffer_end");

    ThrowStatus(iree_hal_semaphore_create(device, IREE_HAL_QUEUE_AFFINITY_ANY,
                                          0, IREE_HAL_SEMAPHORE_FLAG_NONE,
                                          &semaphore),
                "iree_hal_semaphore_create");
    uint64_t signal_value = 1;
    iree_hal_semaphore_list_t signal_list = {1, &semaphore, &signal_value};
    ThrowStatus(iree_hal_device_queue_execute(
                    device, IREE_HAL_QUEUE_AFFINITY_ANY,
                    iree_hal_semaphore_list_empty(), signal_list,
                    command_buffer, iree_hal_buffer_binding_table_empty(),
                    IREE_HAL_EXECUTE_FLAG_NONE),
                "iree_hal_device_queue_execute");
    ThrowStatus(iree_hal_semaphore_wait(semaphore, signal_value,
                                        iree_infinite_timeout(),
                                        IREE_ASYNC_WAIT_FLAG_NONE),
                "iree_hal_semaphore_wait");
  } catch (...) {
    if (semaphore)
      iree_hal_semaphore_release(semaphore);
    if (command_buffer)
      iree_hal_command_buffer_release(command_buffer);
    for (iree_hal_executable_t *executable : executables) {
      if (executable)
        iree_hal_executable_release(executable);
    }
    if (cache)
      iree_hal_executable_cache_release(cache);
    throw;
  }
  iree_hal_semaphore_release(semaphore);
  iree_hal_command_buffer_release(command_buffer);
  for (iree_hal_executable_t *executable : executables) {
    iree_hal_executable_release(executable);
  }
  iree_hal_executable_cache_release(cache);
}

void ValidateConv2dOperands(const Conv2dProblem &problem,
                            const std::vector<float> &input,
                            const std::vector<float> &weights) {
  if (problem.input_channels == 0 || problem.output_channels == 0) {
    throw std::invalid_argument("Conv2D channels must be nonzero");
  }
  if (problem.pad_top >= problem.kernel_height ||
      problem.pad_left >= problem.kernel_width) {
    throw std::invalid_argument(
        "Conv2D padding must be smaller than its kernel");
  }
  if (input.size() != problem.input_element_count()) {
    throw std::invalid_argument(
        "input element count does not match Conv2D problem");
  }
  if (weights.size() != problem.weight_element_count()) {
    throw std::invalid_argument(
        "weight element count does not match Conv2D problem");
  }
}

} // namespace

std::vector<uint8_t> BuildFp16Conv2dExecutable(const Conv2dProblem &problem) {
  return BuildRkt1Executable(problem);
}

uint32_t Conv2dProblem::output_height() const {
  const uint64_t padded = static_cast<uint64_t>(input_height) + 2ull * pad_top;
  if (stride == 0 || kernel_height == 0 || padded < kernel_height) {
    throw std::invalid_argument("invalid Conv2D height/kernel/stride");
  }
  return static_cast<uint32_t>((padded - kernel_height) / stride + 1);
}

uint32_t Conv2dProblem::output_width() const {
  const uint64_t padded = static_cast<uint64_t>(input_width) + 2ull * pad_left;
  if (stride == 0 || kernel_width == 0 || padded < kernel_width) {
    throw std::invalid_argument("invalid Conv2D width/kernel/stride");
  }
  return static_cast<uint32_t>((padded - kernel_width) / stride + 1);
}

size_t Conv2dProblem::input_element_count() const {
  return static_cast<size_t>(input_height) * input_width * input_channels;
}

size_t Conv2dProblem::weight_element_count() const {
  return static_cast<size_t>(kernel_height) * kernel_width * input_channels *
         output_channels;
}

size_t Conv2dProblem::output_element_count() const {
  return static_cast<size_t>(output_height()) * output_width() *
         output_channels;
}

Conv2dResult RunFp16Conv2d(iree_hal_device_t *device,
                           const Conv2dProblem &problem,
                           const std::vector<float> &input,
                           const std::vector<float> &weights) {
  if (!device)
    throw std::invalid_argument("device must not be null");
  ValidateConv2dOperands(problem, input, weights);

  const std::vector<uint16_t> input_f16 = QuantizeToF16(input);
  const std::vector<uint16_t> weights_f16 = QuantizeToF16(weights);
  // The current FP16 BS/bias binding is hardware-layout storage rather than
  // a logical dense tensor. This harness deliberately verifies input and
  // coefficient bindings, so keep bias neutral and generously sized until
  // the driver grows a logical-bias packing bridge analogous to weights.
  std::vector<uint8_t> bias_bytes(kNeutralBiasBytes, 0);
  std::vector<uint16_t> output_f16(problem.output_element_count(), 0);

  iree_hal_buffer_t *input_buffer = nullptr;
  iree_hal_buffer_t *weight_buffer = nullptr;
  iree_hal_buffer_t *bias_buffer = nullptr;
  iree_hal_buffer_t *output_buffer = nullptr;
  try {
    input_buffer = AllocateAndWrite(device, input_f16.data(),
                                    input_f16.size() * sizeof(uint16_t));
    weight_buffer = AllocateAndWrite(device, weights_f16.data(),
                                     weights_f16.size() * sizeof(uint16_t));
    bias_buffer =
        AllocateAndWrite(device, bias_bytes.data(), bias_bytes.size());
    output_buffer = AllocateAndWrite(device, output_f16.data(),
                                     output_f16.size() * sizeof(uint16_t));

    const iree_device_size_t input_bytes = input_f16.size() * sizeof(uint16_t);
    const iree_device_size_t weight_bytes =
        weights_f16.size() * sizeof(uint16_t);
    const iree_device_size_t output_bytes =
        output_f16.size() * sizeof(uint16_t);
    DispatchFp16Conv2d(device, problem, input_buffer, 0, input_bytes,
                       weight_buffer, 0, weight_bytes, bias_buffer, 0,
                       bias_bytes.size(), output_buffer, 0, output_bytes);
    output_f16 = ReadBackF16(output_buffer, 0, output_bytes);
  } catch (...) {
    if (output_buffer)
      iree_hal_buffer_release(output_buffer);
    if (bias_buffer)
      iree_hal_buffer_release(bias_buffer);
    if (weight_buffer)
      iree_hal_buffer_release(weight_buffer);
    if (input_buffer)
      iree_hal_buffer_release(input_buffer);
    throw;
  }

  iree_hal_buffer_release(output_buffer);
  iree_hal_buffer_release(bias_buffer);
  iree_hal_buffer_release(weight_buffer);
  iree_hal_buffer_release(input_buffer);

  Conv2dResult result;
  result.expected = ReferenceConv2d(problem, input_f16, weights_f16);
  result.actual = DecodeF16(output_f16);
  return result;
}

std::pair<Conv2dBufferLayout, iree_device_size_t>
PackConv2dBufferLayout(const Conv2dProblem &problem,
                       iree_device_size_t base_offset) {
  constexpr iree_device_size_t kAlignment = 4096;
  Conv2dBufferLayout layout;
  layout.input_offset = AlignUp(base_offset, kAlignment);
  layout.weight_offset = AlignUp(
      layout.input_offset + problem.input_element_count() * sizeof(uint16_t),
      kAlignment);
  layout.bias_offset = AlignUp(
      layout.weight_offset + problem.weight_element_count() * sizeof(uint16_t),
      kAlignment);
  layout.output_offset =
      AlignUp(layout.bias_offset + kNeutralBiasBytes, kAlignment);
  const iree_device_size_t next_offset = AlignUp(
      layout.output_offset + problem.output_element_count() * sizeof(uint16_t),
      kAlignment);
  return {layout, next_offset};
}

Conv2dResult RunFp16Conv2dBound(iree_hal_device_t *device,
                                iree_hal_buffer_t *shared_buffer,
                                const Conv2dProblem &problem,
                                const Conv2dBufferLayout &layout,
                                const std::vector<float> &input,
                                const std::vector<float> &weights) {
  if (!device)
    throw std::invalid_argument("device must not be null");
  if (!shared_buffer)
    throw std::invalid_argument("shared_buffer must not be null");
  ValidateConv2dOperands(problem, input, weights);

  const std::vector<uint16_t> input_f16 = QuantizeToF16(input);
  const std::vector<uint16_t> weights_f16 = QuantizeToF16(weights);
  const iree_device_size_t input_bytes = input_f16.size() * sizeof(uint16_t);
  const iree_device_size_t weight_bytes =
      weights_f16.size() * sizeof(uint16_t);
  const iree_device_size_t output_bytes =
      problem.output_element_count() * sizeof(uint16_t);
  std::vector<uint8_t> bias_bytes(kNeutralBiasBytes, 0);
  std::vector<uint16_t> zero_output(problem.output_element_count(), 0);

  // Unlike RunFp16Conv2d's AllocateAndWrite, every write below lands in a
  // region of a buffer the caller allocated and will keep alive across
  // other dispatches -- the point of this function.
  ThrowStatus(iree_hal_buffer_map_write(shared_buffer, layout.input_offset,
                                        input_f16.data(), input_bytes),
              "write bound input");
  ThrowStatus(iree_hal_buffer_map_write(shared_buffer, layout.weight_offset,
                                        weights_f16.data(), weight_bytes),
              "write bound weights");
  ThrowStatus(iree_hal_buffer_map_write(shared_buffer, layout.bias_offset,
                                        bias_bytes.data(), bias_bytes.size()),
              "write bound bias");
  ThrowStatus(iree_hal_buffer_map_write(shared_buffer, layout.output_offset,
                                        zero_output.data(), output_bytes),
              "zero bound output");

  DispatchFp16Conv2d(device, problem, shared_buffer, layout.input_offset,
                     input_bytes, shared_buffer, layout.weight_offset,
                     weight_bytes, shared_buffer, layout.bias_offset,
                     bias_bytes.size(), shared_buffer, layout.output_offset,
                     output_bytes);

  const std::vector<uint16_t> output_f16 =
      ReadBackF16(shared_buffer, layout.output_offset, output_bytes);

  Conv2dResult result;
  result.expected = ReferenceConv2d(problem, input_f16, weights_f16);
  result.actual = DecodeF16(output_f16);
  return result;
}

std::vector<Conv2dResult>
RunFp16Conv2dBatch(iree_hal_device_t *device, iree_hal_buffer_t *shared_buffer,
                   const std::vector<Conv2dProblem> &problems,
                   const std::vector<Conv2dBufferLayout> &layouts,
                   const std::vector<std::vector<float>> &inputs,
                   const std::vector<std::vector<float>> &weights) {
  if (!device)
    throw std::invalid_argument("device must not be null");
  if (!shared_buffer)
    throw std::invalid_argument("shared_buffer must not be null");
  if (problems.size() != layouts.size() || problems.size() != inputs.size() ||
      problems.size() != weights.size()) {
    throw std::invalid_argument(
        "problems/layouts/inputs/weights must be the same size");
  }

  std::vector<std::vector<uint16_t>> input_f16_all(problems.size());
  std::vector<std::vector<uint16_t>> weights_f16_all(problems.size());
  std::vector<BatchDispatchEntry> entries;
  entries.reserve(problems.size());
  for (size_t i = 0; i < problems.size(); ++i) {
    ValidateConv2dOperands(problems[i], inputs[i], weights[i]);
    input_f16_all[i] = QuantizeToF16(inputs[i]);
    weights_f16_all[i] = QuantizeToF16(weights[i]);
    const iree_device_size_t input_bytes =
        input_f16_all[i].size() * sizeof(uint16_t);
    const iree_device_size_t weight_bytes =
        weights_f16_all[i].size() * sizeof(uint16_t);
    const iree_device_size_t output_bytes =
        problems[i].output_element_count() * sizeof(uint16_t);
    std::vector<uint8_t> bias_bytes(kNeutralBiasBytes, 0);
    std::vector<uint16_t> zero_output(problems[i].output_element_count(), 0);

    const Conv2dBufferLayout &layout = layouts[i];
    ThrowStatus(iree_hal_buffer_map_write(shared_buffer, layout.input_offset,
                                          input_f16_all[i].data(), input_bytes),
                "write batch input");
    ThrowStatus(iree_hal_buffer_map_write(shared_buffer, layout.weight_offset,
                                          weights_f16_all[i].data(),
                                          weight_bytes),
                "write batch weights");
    ThrowStatus(iree_hal_buffer_map_write(shared_buffer, layout.bias_offset,
                                          bias_bytes.data(), bias_bytes.size()),
                "write batch bias");
    ThrowStatus(iree_hal_buffer_map_write(shared_buffer, layout.output_offset,
                                          zero_output.data(), output_bytes),
                "zero batch output");

    entries.push_back(BatchDispatchEntry{
        problems[i], shared_buffer, layout.input_offset, input_bytes,
        shared_buffer, layout.weight_offset, weight_bytes, shared_buffer,
        layout.bias_offset, bias_bytes.size(), shared_buffer,
        layout.output_offset, output_bytes});
  }

  DispatchFp16Conv2dBatch(device, entries);

  std::vector<Conv2dResult> results;
  results.reserve(problems.size());
  for (size_t i = 0; i < problems.size(); ++i) {
    const iree_device_size_t output_bytes =
        problems[i].output_element_count() * sizeof(uint16_t);
    const std::vector<uint16_t> output_f16 =
        ReadBackF16(shared_buffer, layouts[i].output_offset, output_bytes);

    Conv2dResult result;
    result.expected =
        ReferenceConv2d(problems[i], input_f16_all[i], weights_f16_all[i]);
    result.actual = DecodeF16(output_f16);
    results.push_back(std::move(result));
  }
  return results;
}

// Reproduces the real compiled model's exact buffer layout for a two-
// dispatch poison/victim pair (see DESIGN_NOTES.md, "Planck result: the
// 0325b14 multi-dispatch code path is not the deciding factor either, but
// its investigation found the real lead"): `first` and `second`'s
// input/weight operands are each their own independent buffer -- like
// RunFp16Conv2d, unlike RunFp16Conv2dBound/RunFp16Conv2dBatch's packed
// shared buffer -- but their BS/bias and raw NPU output regions are packed
// *tightly* adjacent in two shared buffers, first's bytes immediately
// followed by second's with no padding or alignment gap between them. This
// exactly matches the real compiled Stream IR: one transient resource with
// poison's raw output at [0, poison_bytes) and victim's at [poison_bytes,
// poison_bytes + victim_bytes), no input/weight bytes anywhere in between,
// and one 384-byte transient resource for both BS regions the same way.
// Bias sizing matches the real unmodified transform.0.mlir's compiled
// convention too: two bytes per output channel, not
// RunFp16Conv2d/RunFp16Conv2dBound/RunFp16Conv2dBatch's generously-oversized
// 4096-byte neutral placeholder (DESIGN_NOTES.md's earlier "full-BS"
// experiment already found exact BS sizing numerically irrelevant to this
// bug, so this only matters here for matching the real layout precisely).
// Both dispatches are recorded into one command buffer, matching the real
// model's stream.cmd.concurrent grouping too.
std::pair<Conv2dResult, Conv2dResult> RunFp16Conv2dPairWithAdjacentOutputs(
    iree_hal_device_t *device, const Conv2dProblem &first,
    const std::vector<float> &first_input,
    const std::vector<float> &first_weights, const Conv2dProblem &second,
    const std::vector<float> &second_input,
    const std::vector<float> &second_weights) {
  if (!device)
    throw std::invalid_argument("device must not be null");
  ValidateConv2dOperands(first, first_input, first_weights);
  ValidateConv2dOperands(second, second_input, second_weights);

  const std::vector<uint16_t> first_input_f16 = QuantizeToF16(first_input);
  const std::vector<uint16_t> first_weights_f16 = QuantizeToF16(first_weights);
  const std::vector<uint16_t> second_input_f16 = QuantizeToF16(second_input);
  const std::vector<uint16_t> second_weights_f16 =
      QuantizeToF16(second_weights);
  const iree_device_size_t first_input_bytes =
      first_input_f16.size() * sizeof(uint16_t);
  const iree_device_size_t first_weight_bytes =
      first_weights_f16.size() * sizeof(uint16_t);
  const iree_device_size_t second_input_bytes =
      second_input_f16.size() * sizeof(uint16_t);
  const iree_device_size_t second_weight_bytes =
      second_weights_f16.size() * sizeof(uint16_t);
  const iree_device_size_t first_bias_bytes = first.output_channels * 2;
  const iree_device_size_t second_bias_bytes = second.output_channels * 2;
  const iree_device_size_t first_output_bytes =
      first.output_element_count() * sizeof(uint16_t);
  const iree_device_size_t second_output_bytes =
      second.output_element_count() * sizeof(uint16_t);

  iree_hal_buffer_t *first_input_buffer = nullptr;
  iree_hal_buffer_t *first_weight_buffer = nullptr;
  iree_hal_buffer_t *second_input_buffer = nullptr;
  iree_hal_buffer_t *second_weight_buffer = nullptr;
  iree_hal_buffer_t *bias_buffer = nullptr;
  iree_hal_buffer_t *output_buffer = nullptr;
  std::vector<uint16_t> first_output_f16;
  std::vector<uint16_t> second_output_f16;
  try {
    first_input_buffer =
        AllocateAndWrite(device, first_input_f16.data(), first_input_bytes);
    first_weight_buffer = AllocateAndWrite(device, first_weights_f16.data(),
                                           first_weight_bytes);
    second_input_buffer =
        AllocateAndWrite(device, second_input_f16.data(), second_input_bytes);
    second_weight_buffer = AllocateAndWrite(device, second_weights_f16.data(),
                                            second_weight_bytes);

    bias_buffer = AllocateBuffer(device, first_bias_bytes + second_bias_bytes);
    const std::vector<uint8_t> zero_bias(first_bias_bytes + second_bias_bytes,
                                         0);
    ThrowStatus(
        iree_hal_buffer_map_write(bias_buffer, 0, zero_bias.data(),
                                  zero_bias.size()),
        "zero adjacent bias");

    output_buffer =
        AllocateBuffer(device, first_output_bytes + second_output_bytes);
    const std::vector<uint8_t> zero_output(
        first_output_bytes + second_output_bytes, 0);
    ThrowStatus(
        iree_hal_buffer_map_write(output_buffer, 0, zero_output.data(),
                                  zero_output.size()),
        "zero adjacent output");

    std::vector<BatchDispatchEntry> entries;
    entries.push_back(BatchDispatchEntry{
        first, first_input_buffer, 0, first_input_bytes, first_weight_buffer,
        0, first_weight_bytes, bias_buffer, 0, first_bias_bytes,
        output_buffer, 0, first_output_bytes});
    entries.push_back(BatchDispatchEntry{
        second, second_input_buffer, 0, second_input_bytes,
        second_weight_buffer, 0, second_weight_bytes, bias_buffer,
        first_bias_bytes, second_bias_bytes, output_buffer, first_output_bytes,
        second_output_bytes});

    DispatchFp16Conv2dBatch(device, entries);

    first_output_f16 = ReadBackF16(output_buffer, 0, first_output_bytes);
    second_output_f16 =
        ReadBackF16(output_buffer, first_output_bytes, second_output_bytes);
  } catch (...) {
    if (output_buffer)
      iree_hal_buffer_release(output_buffer);
    if (bias_buffer)
      iree_hal_buffer_release(bias_buffer);
    if (second_weight_buffer)
      iree_hal_buffer_release(second_weight_buffer);
    if (second_input_buffer)
      iree_hal_buffer_release(second_input_buffer);
    if (first_weight_buffer)
      iree_hal_buffer_release(first_weight_buffer);
    if (first_input_buffer)
      iree_hal_buffer_release(first_input_buffer);
    throw;
  }
  iree_hal_buffer_release(output_buffer);
  iree_hal_buffer_release(bias_buffer);
  iree_hal_buffer_release(second_weight_buffer);
  iree_hal_buffer_release(second_input_buffer);
  iree_hal_buffer_release(first_weight_buffer);
  iree_hal_buffer_release(first_input_buffer);

  Conv2dResult first_result;
  first_result.expected =
      ReferenceConv2d(first, first_input_f16, first_weights_f16);
  first_result.actual = DecodeF16(first_output_f16);

  Conv2dResult second_result;
  second_result.expected =
      ReferenceConv2d(second, second_input_f16, second_weights_f16);
  second_result.actual = DecodeF16(second_output_f16);

  return {first_result, second_result};
}

} // namespace rocket::testing
