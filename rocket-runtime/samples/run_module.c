// Copyright 2026 The iree-rocket Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Runs one function of a rocket-compiler module through rocket/runtime.h:
// the smallest complete embedding, and the check that the library produces
// what iree-run-module does.
//
//   rocket-run-module [--cpu-only] [--cpu-ids=4-7] [--iterations=N]
//       model.vmfb module.main_graph 1x3x224x224xf32=0.1 [more inputs...]
//
// Inputs use IREE's text form (iree_hal_buffer_view_parse). Outputs print the
// way iree-run-module prints them, so the two can be diffed.

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocket/runtime.h"

static void usage(const char* argv0) {
  fprintf(stderr,
          "usage: %s [--cpu-only] [--cpu-ids=LIST | --iree-cpus] "
          "[--iterations=N] MODULE.vmfb FUNCTION [INPUT...]\n"
          "  --cpu-ids=4-7  pin CPU workers to these CPUs (default: the "
          "fastest cluster)\n"
          "  --iree-cpus    use IREE's own worker placement instead\n"
          "  INPUT is IREE's text form, e.g. 1x3x224x224xf32=0.1\n",
          argv0);
}

static iree_status_t run(int argc, char** argv) {
  rocket_runtime_session_options_t options;
  rocket_runtime_session_options_initialize(&options);
  long iterations = 1;

  int arg = 1;
  for (; arg < argc && strncmp(argv[arg], "--", 2) == 0; ++arg) {
    if (strcmp(argv[arg], "--cpu-only") == 0) {
      options.devices = ROCKET_RUNTIME_DEVICES_CPU_ONLY;
    } else if (strncmp(argv[arg], "--cpu-ids=", 10) == 0) {
      options.cpu_policy = ROCKET_RUNTIME_CPU_POLICY_LIST;
      options.cpu_ids = argv[arg] + 10;
    } else if (strcmp(argv[arg], "--iree-cpus") == 0) {
      options.cpu_policy = ROCKET_RUNTIME_CPU_POLICY_IREE_DEFAULT;
    } else if (strncmp(argv[arg], "--iterations=", 13) == 0) {
      iterations = strtol(argv[arg] + 13, NULL, 10);
      if (iterations < 1) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "--iterations must be at least 1");
      }
    } else {
      usage(argv[0]);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "unknown flag %s",
                              argv[arg]);
    }
  }
  if (argc - arg < 2) {
    usage(argv[0]);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "need a module and a function");
  }
  const char* module_path = argv[arg++];
  const char* function_name = argv[arg++];

  if (rocket_runtime_api_version() != ROCKET_RUNTIME_API_VERSION) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "rocket/runtime.h is API %d but the library is %u",
                            ROCKET_RUNTIME_API_VERSION,
                            rocket_runtime_api_version());
  }

  iree_allocator_t host_allocator = iree_allocator_system();
  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(&instance_options);
  iree_runtime_instance_options_use_all_available_drivers(&instance_options);
  iree_runtime_instance_t* instance = NULL;
  IREE_RETURN_IF_ERROR(iree_runtime_instance_create(
      &instance_options, host_allocator, &instance));

  rocket_runtime_session_t* session = NULL;
  iree_status_t status = rocket_runtime_session_create(
      instance, &options, host_allocator, &session);
  if (iree_status_is_ok(status)) {
    fprintf(stderr, "session: %" PRIhsz " device(s)\n",
            rocket_runtime_session_device_count(session));
    status = rocket_runtime_session_append_bytecode_module_from_file(
        session, module_path);
  }

  rocket_runtime_call_t call;
  memset(&call, 0, sizeof(call));
  if (iree_status_is_ok(status)) {
    status = rocket_runtime_call_initialize_by_name(
        session, iree_make_cstring_view(function_name), &call);
  }

  // Parse the inputs once; each iteration pushes the same buffer views.
  iree_host_size_t input_count = (iree_host_size_t)(argc - arg);
  iree_hal_buffer_view_t** inputs = NULL;
  if (iree_status_is_ok(status) && input_count > 0) {
    status = iree_allocator_malloc_array(host_allocator, input_count,
                                         sizeof(*inputs), (void**)&inputs);
    if (iree_status_is_ok(status)) {
      memset(inputs, 0, input_count * sizeof(*inputs));
    }
  }
  for (iree_host_size_t i = 0; i < input_count && iree_status_is_ok(status);
       ++i) {
    status = iree_hal_buffer_view_parse(
        iree_make_cstring_view(argv[arg + i]),
        rocket_runtime_session_device(session, 0),
        rocket_runtime_session_device_allocator(session), &inputs[i]);
    if (!iree_status_is_ok(status)) {
      status = iree_status_annotate_f(status, "parsing input %" PRIhsz " '%s'",
                                      i, argv[arg + i]);
    }
  }

  iree_time_t first_ns = 0, rest_ns = 0;
  for (long it = 0; it < iterations && iree_status_is_ok(status); ++it) {
    rocket_runtime_call_reset(&call);
    for (iree_host_size_t i = 0; i < input_count && iree_status_is_ok(status);
         ++i) {
      status = rocket_runtime_call_inputs_push_back_buffer_view(&call,
                                                                inputs[i]);
    }
    iree_time_t start = iree_time_now();
    if (iree_status_is_ok(status)) {
      status = rocket_runtime_call_invoke(&call, /*flags=*/0);
    }
    iree_time_t elapsed = iree_time_now() - start;
    if (it == 0) {
      first_ns = elapsed;
    } else {
      rest_ns += elapsed;
    }
  }

  // Print the last iteration's results the way iree-run-module does.
  if (iree_status_is_ok(status)) {
    iree_vm_list_t* outputs = rocket_runtime_call_outputs(&call);
    iree_host_size_t output_count = iree_vm_list_size(outputs);
    for (iree_host_size_t i = 0; i < output_count && iree_status_is_ok(status);
         ++i) {
      iree_hal_buffer_view_t* output = NULL;
      status = rocket_runtime_call_outputs_pop_front_buffer_view(&call,
                                                                 &output);
      if (iree_status_is_ok(status)) {
        fprintf(stdout, "result[%" PRIhsz "]: hal.buffer_view\n", i);
        status = iree_hal_buffer_view_fprint(stdout, output,
                                             /*max_element_count=*/1024,
                                             host_allocator);
        fprintf(stdout, "\n");
      }
      iree_hal_buffer_view_release(output);
    }
  }
  if (iree_status_is_ok(status)) {
    fprintf(stderr, "first call %.3f ms", first_ns / 1e6);
    if (iterations > 1) {
      fprintf(stderr, ", then %.3f ms mean over %ld",
              rest_ns / 1e6 / (double)(iterations - 1), iterations - 1);
    }
    fprintf(stderr, "\n");
  }

  for (iree_host_size_t i = 0; inputs && i < input_count; ++i) {
    iree_hal_buffer_view_release(inputs[i]);
  }
  iree_allocator_free(host_allocator, inputs);
  if (call.session) rocket_runtime_call_deinitialize(&call);
  rocket_runtime_session_release(session);
  iree_runtime_instance_release(instance);
  return status;
}

int main(int argc, char** argv) {
  iree_status_t status = run(argc, argv);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return 1;
  }
  return 0;
}
