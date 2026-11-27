// Copyright 2026 The iree-rocket Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The smallest C++ program that runs a rocket-compiler module, using nothing
// but the installed package. Its input comes from host memory, the way an
// application holds its data, rather than from a command-line string.
//
//   embed model.vmfb module.main_graph
//
// runs `main_graph` on one 1x3x224x224 f16 image whose every element is 0.1,
// and prints the first result. Build it against the package with either
//
//   c++ -std=c++17 embed.cc $(pkg-config --cflags --libs iree-rocket-runtime)
//   c++ -std=c++17 embed.cc $(pkg-config --cflags --libs iree-rocket-runtime-static)
//
// or, from CMake, find_package(IreeRocketRuntime) and link IreeRocket::runtime.

#include <cstdint>
#include <cstdio>
#include <vector>

#include "rocket/runtime.h"

namespace {

// Prints a failed status and frees it; true when |status| was OK.
bool ok(iree_status_t status) {
  if (iree_status_is_ok(status)) return true;
  iree_status_fprint(stderr, status);
  iree_status_free(status);
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s MODULE.vmfb FUNCTION\n", argv[0]);
    return 2;
  }
  iree_allocator_t host = iree_allocator_system();

  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(&instance_options);
  iree_runtime_instance_options_use_all_available_drivers(&instance_options);
  iree_runtime_instance_t* instance = nullptr;
  if (!ok(iree_runtime_instance_create(&instance_options, host, &instance))) {
    return 1;
  }

  // The defaults: the NPU plus the CPU, CPU workers on the fastest cores.
  rocket_runtime_session_options_t options;
  rocket_runtime_session_options_initialize(&options);
  rocket_runtime_session_t* session = nullptr;
  bool good =
      ok(rocket_runtime_session_create(instance, &options, host, &session)) &&
      ok(rocket_runtime_session_append_bytecode_module_from_file(session,
                                                                 argv[1]));

  // 0x2E66 is 0.1 in IEEE half precision.
  std::vector<uint16_t> pixels(1 * 3 * 224 * 224, 0x2E66);
  const iree_hal_dim_t shape[] = {1, 3, 224, 224};
  iree_hal_buffer_params_t params = {};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage = IREE_HAL_BUFFER_USAGE_DEFAULT;
  iree_hal_buffer_view_t* input = nullptr;
  good = good &&
         ok(iree_hal_buffer_view_allocate_buffer_copy(
             rocket_runtime_session_device(session, 0),
             rocket_runtime_session_device_allocator(session),
             IREE_ARRAYSIZE(shape), shape, IREE_HAL_ELEMENT_TYPE_FLOAT_16,
             IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, params,
             iree_make_const_byte_span(pixels.data(),
                                       pixels.size() * sizeof(pixels[0])),
             &input));

  rocket_runtime_call_t call = {};
  iree_hal_buffer_view_t* output = nullptr;
  good = good &&
         ok(rocket_runtime_call_initialize_by_name(
             session, iree_make_cstring_view(argv[2]), &call)) &&
         ok(rocket_runtime_call_inputs_push_back_buffer_view(&call, input)) &&
         ok(rocket_runtime_call_invoke(&call, /*flags=*/0)) &&
         ok(rocket_runtime_call_outputs_pop_front_buffer_view(&call, &output));

  if (good) {
    std::printf("api %u, %zu device(s)\n", rocket_runtime_api_version(),
                static_cast<size_t>(rocket_runtime_session_device_count(session)));
    good = ok(iree_hal_buffer_view_fprint(stdout, output,
                                          /*max_element_count=*/1024, host));
    std::printf("\n");
  }

  iree_hal_buffer_view_release(output);
  iree_hal_buffer_view_release(input);
  if (call.session) rocket_runtime_call_deinitialize(&call);
  rocket_runtime_session_release(session);
  iree_runtime_instance_release(instance);
  return good ? 0 : 1;
}
