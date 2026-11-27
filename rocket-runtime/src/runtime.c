// Copyright 2026 The iree-rocket Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The session is iree/runtime/session.c with one change: the HAL module is
// bound to a device group of [rocket, local-task] instead of a single device.
// Device creation follows iree/tooling/context_util.c (what iree-run-module
// runs), except that the local-task driver is built here with an explicit
// worker topology rather than from IREE's command-line flags, which a library
// cannot rely on anyone having parsed.

#include "rocket/runtime.h"

#include <stdio.h>
#include <string.h>

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/threading/numa.h"
#include "iree/hal/drivers/local_task/task_device.h"
#include "iree/hal/drivers/local_task/task_driver.h"
#include "iree/hal/local/loaders/registration/init.h"
#include "iree/hal/local/plugins/registration/init.h"
#include "iree/io/file_contents.h"
#include "iree/modules/hal/module.h"
#include "iree/task/api.h"
#include "iree/task/topology.h"
#include "iree/vm/bytecode/module.h"

IREE_API_EXPORT uint32_t rocket_runtime_api_version(void) {
  return ROCKET_RUNTIME_API_VERSION;
}

//===----------------------------------------------------------------------===//
// CPU selection
//===----------------------------------------------------------------------===//

// Covers every RK3588 (8 CPUs) with room to spare; a larger list is an error
// rather than a silent truncation.
#define ROCKET_RUNTIME_MAX_CPUS 256

typedef struct rocket_runtime_cpu_set_t {
  iree_host_size_t count;
  uint32_t ids[ROCKET_RUNTIME_MAX_CPUS];
} rocket_runtime_cpu_set_t;

static iree_status_t rocket_runtime_cpu_set_add(rocket_runtime_cpu_set_t* set,
                                                uint32_t cpu) {
  for (iree_host_size_t i = 0; i < set->count; ++i) {
    if (set->ids[i] == cpu) return iree_ok_status();
  }
  if (set->count == ROCKET_RUNTIME_MAX_CPUS) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "more than %d CPUs requested",
                            ROCKET_RUNTIME_MAX_CPUS);
  }
  set->ids[set->count++] = cpu;
  return iree_ok_status();
}

// Parses the `taskset -c` spelling ("4-7", "0-3,6"), the same one the driver's
// ROCKET_HOST_CPUS takes.
static iree_status_t rocket_runtime_parse_cpu_list(
    iree_string_view_t list, rocket_runtime_cpu_set_t* out_set) {
  out_set->count = 0;
  while (!iree_string_view_is_empty(list)) {
    iree_string_view_t part;
    iree_string_view_split(list, ',', &part, &list);
    part = iree_string_view_trim(part);
    if (iree_string_view_is_empty(part)) continue;
    iree_string_view_t first_text, last_text;
    intptr_t dash = iree_string_view_split(part, '-', &first_text, &last_text);
    uint32_t first = 0, last = 0;
    if (!iree_string_view_atoi_uint32(iree_string_view_trim(first_text),
                                      &first)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "bad CPU id '%.*s' in cpu_ids", (int)part.size,
                              part.data);
    }
    last = first;
    if (dash != -1 && !iree_string_view_atoi_uint32(
                          iree_string_view_trim(last_text), &last)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "bad CPU range '%.*s' in cpu_ids",
                              (int)part.size, part.data);
    }
    if (last < first) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "descending CPU range '%.*s' in cpu_ids",
                              (int)part.size, part.data);
    }
    for (uint32_t cpu = first; cpu <= last; ++cpu) {
      IREE_RETURN_IF_ERROR(rocket_runtime_cpu_set_add(out_set, cpu));
    }
  }
  if (out_set->count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "cpu_ids names no CPUs");
  }
  return iree_ok_status();
}

// Every CPU sharing the highest cpu_capacity, the same rule as the driver's
// rocket-hal-driver/src/cpu_affinity.rs. Leaves |out_set| empty when there is
// nothing to prefer: capacities absent (uniform machines and kernels without
// the energy model do not publish them) or all equal.
static void rocket_runtime_fastest_cpus(rocket_runtime_cpu_set_t* out_set) {
  out_set->count = 0;
  uint32_t capacities[ROCKET_RUNTIME_MAX_CPUS];
  iree_host_size_t cpu_count = 0;
  for (; cpu_count < ROCKET_RUNTIME_MAX_CPUS; ++cpu_count) {
    char path[64];
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%u/cpu_capacity",
             (unsigned)cpu_count);
    FILE* file = fopen(path, "r");
    // CPUs are numbered contiguously; the first miss ends the list.
    if (!file) break;
    unsigned capacity = 0;
    int matched = fscanf(file, "%u", &capacity);
    fclose(file);
    if (matched != 1) return;
    capacities[cpu_count] = capacity;
  }
  if (cpu_count == 0) return;
  uint32_t highest = 0;
  bool uniform = true;
  for (iree_host_size_t i = 0; i < cpu_count; ++i) {
    if (capacities[i] > highest) highest = capacities[i];
    if (capacities[i] != capacities[0]) uniform = false;
  }
  if (uniform) return;
  for (iree_host_size_t i = 0; i < cpu_count; ++i) {
    if (capacities[i] == highest) out_set->ids[out_set->count++] = (uint32_t)i;
  }
}

//===----------------------------------------------------------------------===//
// CPU device
//===----------------------------------------------------------------------===//

// Creates the executors for the CPU device: one pinned to |cpus|, or IREE's
// own flag-driven selection when |cpus| is empty.
static iree_status_t rocket_runtime_create_executors(
    const rocket_runtime_cpu_set_t* cpus, iree_allocator_t host_allocator,
    iree_host_size_t executor_capacity, iree_task_executor_t** executors,
    iree_host_size_t* out_executor_count) {
  *out_executor_count = 0;
  if (cpus->count == 0) {
    return iree_task_executors_create_from_flags(
        host_allocator, executor_capacity, executors, out_executor_count);
  }
  iree_task_topology_t topology;
  IREE_RETURN_IF_ERROR(iree_task_topology_initialize_from_logical_cpu_set(
      cpus->count, cpus->ids, &topology));
  iree_task_executor_options_t options;
  iree_task_executor_options_initialize(&options);
  iree_status_t status = iree_task_executor_create(options, &topology,
                                                   host_allocator, &executors[0]);
  iree_task_topology_deinitialize(&topology);
  if (iree_status_is_ok(status)) *out_executor_count = 1;
  return status;
}

// The local-task driver, assembled the way its registration factory
// (iree/hal/drivers/local_task/registration/driver_module.c) does it but with
// our executors.
static iree_status_t rocket_runtime_create_cpu_driver(
    const rocket_runtime_cpu_set_t* cpus, iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver) {
  iree_hal_task_device_params_t params;
  iree_hal_task_device_params_initialize(&params);

  iree_task_executor_t* executors[16] = {NULL};
  iree_host_size_t executor_count = 0;
  IREE_RETURN_IF_ERROR(rocket_runtime_create_executors(
      cpus, host_allocator, IREE_ARRAYSIZE(executors), executors,
      &executor_count));

  iree_hal_executable_plugin_manager_t* plugin_manager = NULL;
  iree_status_t status = iree_hal_executable_plugin_manager_create_from_flags(
      host_allocator, &plugin_manager);

  iree_hal_executable_loader_t* loaders[8] = {NULL};
  iree_host_size_t loader_count = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_create_all_available_executable_loaders(
        plugin_manager, IREE_ARRAYSIZE(loaders), &loader_count, loaders,
        host_allocator);
  }

  iree_hal_allocator_t* device_allocator = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_allocator_create_heap(iree_make_cstring_view("local"),
                                            host_allocator, host_allocator,
                                            &device_allocator);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_task_driver_create(
        IREE_SV("local-task"), &params, executor_count, executors,
        loader_count, loaders, device_allocator, host_allocator, out_driver);
  }

  for (iree_host_size_t i = 0; i < executor_count; ++i) {
    iree_task_executor_release(executors[i]);
  }
  for (iree_host_size_t i = 0; i < loader_count; ++i) {
    iree_hal_executable_loader_release(loaders[i]);
  }
  iree_hal_executable_plugin_manager_release(plugin_manager);
  iree_hal_allocator_release(device_allocator);
  return status;
}

static iree_status_t rocket_runtime_create_cpu_device(
    const rocket_runtime_session_options_t* options,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  rocket_runtime_cpu_set_t cpus;
  cpus.count = 0;
  switch (options->cpu_policy) {
    case ROCKET_RUNTIME_CPU_POLICY_FASTEST:
      rocket_runtime_fastest_cpus(&cpus);
      break;
    case ROCKET_RUNTIME_CPU_POLICY_LIST:
      if (!options->cpu_ids) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "ROCKET_RUNTIME_CPU_POLICY_LIST needs cpu_ids");
      }
      IREE_RETURN_IF_ERROR(rocket_runtime_parse_cpu_list(
          iree_make_cstring_view(options->cpu_ids), &cpus));
      break;
    case ROCKET_RUNTIME_CPU_POLICY_IREE_DEFAULT:
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown cpu_policy %d",
                              (int)options->cpu_policy);
  }

  iree_hal_driver_t* driver = NULL;
  IREE_RETURN_IF_ERROR(
      rocket_runtime_create_cpu_driver(&cpus, host_allocator, &driver));
  iree_status_t status = iree_hal_driver_create_default_device(
      driver, create_params, host_allocator, out_device);
  iree_hal_driver_release(driver);
  return status;
}

//===----------------------------------------------------------------------===//
// Device group
//===----------------------------------------------------------------------===//

static iree_status_t rocket_runtime_create_device_group(
    iree_runtime_instance_t* instance,
    const rocket_runtime_session_options_t* options,
    iree_allocator_t host_allocator, iree_hal_device_group_t** out_group) {
  *out_group = NULL;

  // Both drivers do async file I/O through the proactor pool; the rocket
  // device refuses to start without one.
  iree_async_proactor_pool_t* proactor_pool = NULL;
  IREE_RETURN_IF_ERROR(iree_async_proactor_pool_create(
      iree_numa_node_count(), /*node_ids=*/NULL,
      iree_async_proactor_pool_options_default(), host_allocator,
      &proactor_pool));
  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  create_params.proactor_pool = proactor_pool;

  iree_hal_device_t* devices[2] = {NULL, NULL};
  iree_host_size_t device_count = 0;
  iree_status_t status = iree_ok_status();

  // The NPU first, so it is device 0: the lead device whose allocator inputs
  // come from, matching `iree-run-module --device=rocket --device=local-task`.
  if (options->devices == ROCKET_RUNTIME_DEVICES_NPU_AND_CPU) {
    iree_hal_driver_registry_t* registry =
        iree_runtime_instance_driver_registry(instance);
    if (!registry) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "the instance has no driver registry; create it with "
          "iree_runtime_instance_options_use_all_available_drivers");
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_create_device(registry, IREE_SV("rocket"),
                                      &create_params, host_allocator,
                                      &devices[device_count]);
      if (iree_status_is_ok(status)) {
        ++device_count;
      } else {
        status = iree_status_annotate(
            status, IREE_SV("creating the rocket (NPU) device; use "
                            "ROCKET_RUNTIME_DEVICES_CPU_ONLY on a machine "
                            "without one"));
      }
    }
  } else if (options->devices != ROCKET_RUNTIME_DEVICES_CPU_ONLY) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown devices %d", (int)options->devices);
  }

  if (iree_status_is_ok(status)) {
    status = rocket_runtime_create_cpu_device(
        options, &create_params, host_allocator, &devices[device_count]);
    if (iree_status_is_ok(status)) ++device_count;
  }
  iree_async_proactor_pool_release(proactor_pool);

  iree_async_frontier_tracker_t* frontier_tracker = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_async_frontier_tracker_create(
        iree_async_frontier_tracker_options_default(), host_allocator,
        &frontier_tracker);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_device_group_builder_t builder;
    iree_hal_device_group_builder_initialize(&builder, frontier_tracker);
    for (iree_host_size_t i = 0; i < device_count && iree_status_is_ok(status);
         ++i) {
      status = iree_hal_device_group_builder_add_device(&builder, devices[i]);
    }
    if (iree_status_is_ok(status)) {
      // Retains the devices and zeroes the builder, success or failure.
      status = iree_hal_device_group_builder_finalize(&builder, host_allocator,
                                                      out_group);
    } else {
      iree_hal_device_group_builder_deinitialize(&builder);
    }
  }
  iree_async_frontier_tracker_release(frontier_tracker);
  for (iree_host_size_t i = 0; i < device_count; ++i) {
    iree_hal_device_release(devices[i]);
  }
  return status;
}

// Allocator selection for buffers more than one device may touch: always the
// lead device's when it is among the candidates, as iree-run-module's
// --device_lead_allocator=0 does (iree/tooling/context_util.c). The HAL
// module's default policy takes whichever candidate a query happens to list
// first instead, and with [rocket, local-task] that fails inside the module's
// own __init (an INTERNAL from hal.device.queue.alloca, measured on the board).
static iree_status_t rocket_runtime_select_lead_allocator(
    void* user_data, iree_host_size_t device_count,
    const iree_hal_device_queue_affinity_pair_t* devices,
    iree_hal_memory_type_t memory_types, iree_hal_buffer_usage_t buffer_usage,
    iree_hal_module_device_allocator_select_flags_t flags,
    iree_host_size_t* out_selection) {
  (void)memory_types;
  (void)buffer_usage;
  (void)flags;
  *out_selection = 0;
  for (iree_host_size_t i = 0; i < device_count; ++i) {
    if (devices[i].device == (iree_hal_device_t*)user_data) {
      *out_selection = i;
      break;
    }
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// rocket_runtime_session_t
//===----------------------------------------------------------------------===//

struct rocket_runtime_session_t {
  iree_atomic_ref_count_t ref_count;
  iree_allocator_t host_allocator;
  // Retained: devices and pools live on the instance's registry.
  iree_runtime_instance_t* instance;
  // Retained; the group must outlive its devices, and the HAL module holds
  // the devices.
  iree_hal_device_group_t* device_group;
  iree_vm_context_t* context;
  // Owned by |context|; cached for device lookups.
  iree_vm_module_state_t* hal_module_state;
};

IREE_API_EXPORT void rocket_runtime_session_options_initialize(
    rocket_runtime_session_options_t* out_options) {
  memset(out_options, 0, sizeof(*out_options));
  out_options->devices = ROCKET_RUNTIME_DEVICES_NPU_AND_CPU;
  out_options->cpu_policy = ROCKET_RUNTIME_CPU_POLICY_FASTEST;
  out_options->cpu_ids = NULL;
  out_options->context_flags = IREE_VM_CONTEXT_FLAG_NONE;
}

static void rocket_runtime_session_destroy(rocket_runtime_session_t* session) {
  // The context holds the HAL module and with it the devices; release it
  // before the group the devices belong to.
  iree_vm_context_release(session->context);
  iree_hal_device_group_release(session->device_group);
  iree_runtime_instance_release(session->instance);
  iree_allocator_free(session->host_allocator, session);
}

IREE_API_EXPORT iree_status_t rocket_runtime_session_create(
    iree_runtime_instance_t* instance,
    const rocket_runtime_session_options_t* options,
    iree_allocator_t host_allocator, rocket_runtime_session_t** out_session) {
  IREE_ASSERT_ARGUMENT(instance);
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_session);
  *out_session = NULL;

  rocket_runtime_session_t* session = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*session),
                                             (void**)&session));
  memset(session, 0, sizeof(*session));
  iree_atomic_ref_count_init(&session->ref_count);
  session->host_allocator = host_allocator;
  session->instance = instance;
  iree_runtime_instance_retain(instance);

  iree_status_t status = rocket_runtime_create_device_group(
      instance, options, host_allocator, &session->device_group);

  iree_vm_instance_t* vm_instance = iree_runtime_instance_vm_instance(instance);
  if (iree_status_is_ok(status)) {
    status = iree_vm_context_create(vm_instance, options->context_flags,
                                    host_allocator, &session->context);
  }

  iree_vm_module_t* hal_module = NULL;
  if (iree_status_is_ok(status)) {
    iree_hal_module_device_policy_t device_policy =
        iree_hal_module_device_policy_default();
    device_policy.allocator_select.fn = rocket_runtime_select_lead_allocator;
    device_policy.allocator_select.user_data =
        iree_hal_device_group_device_at(session->device_group, 0);
    status = iree_hal_module_create(
        vm_instance, device_policy, session->device_group,
        IREE_HAL_MODULE_FLAG_NONE,
        iree_hal_module_debug_sink_stdio(stderr), host_allocator, &hal_module);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_context_register_modules(session->context, 1, &hal_module);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_context_resolve_module_state(session->context, hal_module,
                                                  &session->hal_module_state);
  }
  iree_vm_module_release(hal_module);

  if (iree_status_is_ok(status)) {
    *out_session = session;
  } else {
    rocket_runtime_session_destroy(session);
  }
  return status;
}

IREE_API_EXPORT void rocket_runtime_session_retain(
    rocket_runtime_session_t* session) {
  if (session) iree_atomic_ref_count_inc(&session->ref_count);
}

IREE_API_EXPORT void rocket_runtime_session_release(
    rocket_runtime_session_t* session) {
  if (session && iree_atomic_ref_count_dec(&session->ref_count) == 1) {
    rocket_runtime_session_destroy(session);
  }
}

IREE_API_EXPORT iree_allocator_t
rocket_runtime_session_host_allocator(const rocket_runtime_session_t* session) {
  IREE_ASSERT_ARGUMENT(session);
  return session->host_allocator;
}

IREE_API_EXPORT iree_runtime_instance_t* rocket_runtime_session_instance(
    const rocket_runtime_session_t* session) {
  IREE_ASSERT_ARGUMENT(session);
  return session->instance;
}

IREE_API_EXPORT iree_vm_context_t* rocket_runtime_session_context(
    const rocket_runtime_session_t* session) {
  IREE_ASSERT_ARGUMENT(session);
  return session->context;
}

IREE_API_EXPORT iree_hal_device_group_t* rocket_runtime_session_device_group(
    const rocket_runtime_session_t* session) {
  IREE_ASSERT_ARGUMENT(session);
  return session->device_group;
}

IREE_API_EXPORT iree_host_size_t
rocket_runtime_session_device_count(const rocket_runtime_session_t* session) {
  IREE_ASSERT_ARGUMENT(session);
  return iree_hal_device_group_device_count(session->device_group);
}

IREE_API_EXPORT iree_hal_device_t* rocket_runtime_session_device(
    const rocket_runtime_session_t* session, iree_host_size_t index) {
  IREE_ASSERT_ARGUMENT(session);
  if (index >= iree_hal_device_group_device_count(session->device_group)) {
    return NULL;
  }
  return iree_hal_device_group_device_at(session->device_group, index);
}

IREE_API_EXPORT iree_hal_allocator_t* rocket_runtime_session_device_allocator(
    const rocket_runtime_session_t* session) {
  iree_hal_device_t* device = rocket_runtime_session_device(session, 0);
  return device ? iree_hal_device_allocator(device) : NULL;
}

IREE_API_EXPORT iree_status_t rocket_runtime_session_append_module(
    rocket_runtime_session_t* session, iree_vm_module_t* module) {
  IREE_ASSERT_ARGUMENT(session);
  IREE_ASSERT_ARGUMENT(module);
  return iree_vm_context_register_modules(session->context, 1, &module);
}

IREE_API_EXPORT iree_status_t
rocket_runtime_session_append_bytecode_module_from_memory(
    rocket_runtime_session_t* session, iree_const_byte_span_t flatbuffer_data,
    iree_allocator_t flatbuffer_allocator) {
  IREE_ASSERT_ARGUMENT(session);
  iree_vm_module_t* module = NULL;
  // Consumes |flatbuffer_data| on failure too.
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_module_create(
      iree_runtime_instance_vm_instance(session->instance),
      IREE_VM_BYTECODE_MODULE_FLAG_NONE, flatbuffer_data, flatbuffer_allocator,
      session->host_allocator, &module));
  iree_status_t status = rocket_runtime_session_append_module(session, module);
  iree_vm_module_release(module);
  return status;
}

IREE_API_EXPORT iree_status_t
rocket_runtime_session_append_bytecode_module_from_file(
    rocket_runtime_session_t* session, const char* file_path) {
  IREE_ASSERT_ARGUMENT(session);
  IREE_ASSERT_ARGUMENT(file_path);
  iree_io_file_contents_t* contents = NULL;
  IREE_RETURN_IF_ERROR(
      iree_io_file_contents_read(iree_make_cstring_view(file_path),
                                 session->host_allocator, &contents),
      "reading module '%s'", file_path);
  return rocket_runtime_session_append_bytecode_module_from_memory(
      session, contents->const_buffer,
      iree_io_file_contents_deallocator(contents));
}

IREE_API_EXPORT iree_status_t rocket_runtime_session_lookup_function(
    const rocket_runtime_session_t* session, iree_string_view_t full_name,
    iree_vm_function_t* out_function) {
  IREE_ASSERT_ARGUMENT(session);
  IREE_ASSERT_ARGUMENT(out_function);
  return iree_vm_context_resolve_function(session->context, full_name,
                                          out_function);
}

IREE_API_EXPORT iree_status_t rocket_runtime_session_call(
    rocket_runtime_session_t* session, const iree_vm_function_t* function,
    iree_vm_list_t* input_list, iree_vm_list_t* output_list) {
  IREE_ASSERT_ARGUMENT(session);
  IREE_ASSERT_ARGUMENT(function);
  return iree_vm_invoke(session->context, *function,
                        IREE_VM_INVOCATION_FLAG_NONE, /*policy=*/NULL,
                        input_list, output_list, session->host_allocator);
}

IREE_API_EXPORT iree_status_t rocket_runtime_session_call_by_name(
    rocket_runtime_session_t* session, iree_string_view_t full_name,
    iree_vm_list_t* input_list, iree_vm_list_t* output_list) {
  iree_vm_function_t function;
  IREE_RETURN_IF_ERROR(
      rocket_runtime_session_lookup_function(session, full_name, &function));
  return rocket_runtime_session_call(session, &function, input_list,
                                     output_list);
}

IREE_API_EXPORT iree_status_t
rocket_runtime_session_trim(rocket_runtime_session_t* session) {
  IREE_ASSERT_ARGUMENT(session);
  return iree_vm_context_notify(session->context, IREE_VM_SIGNAL_LOW_MEMORY);
}

//===----------------------------------------------------------------------===//
// rocket_runtime_call_t
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t rocket_runtime_call_initialize(
    rocket_runtime_session_t* session, iree_vm_function_t function,
    rocket_runtime_call_t* out_call) {
  IREE_ASSERT_ARGUMENT(session);
  IREE_ASSERT_ARGUMENT(out_call);
  memset(out_call, 0, sizeof(*out_call));

  // Size the lists from the calling convention, as iree_runtime_call does.
  iree_vm_function_signature_t signature =
      iree_vm_function_signature(&function);
  iree_string_view_t arguments, results;
  IREE_RETURN_IF_ERROR(iree_vm_function_call_get_cconv_fragments(
      &signature, &arguments, &results));

  out_call->session = session;
  rocket_runtime_session_retain(session);
  out_call->function = function;
  iree_status_t status =
      iree_vm_list_create(iree_vm_make_undefined_type_def(), arguments.size,
                          session->host_allocator, &out_call->inputs);
  if (iree_status_is_ok(status)) {
    status =
        iree_vm_list_create(iree_vm_make_undefined_type_def(), results.size,
                            session->host_allocator, &out_call->outputs);
  }
  if (!iree_status_is_ok(status)) rocket_runtime_call_deinitialize(out_call);
  return status;
}

IREE_API_EXPORT iree_status_t rocket_runtime_call_initialize_by_name(
    rocket_runtime_session_t* session, iree_string_view_t full_name,
    rocket_runtime_call_t* out_call) {
  iree_vm_function_t function;
  IREE_RETURN_IF_ERROR(
      rocket_runtime_session_lookup_function(session, full_name, &function));
  return rocket_runtime_call_initialize(session, function, out_call);
}

IREE_API_EXPORT void rocket_runtime_call_deinitialize(
    rocket_runtime_call_t* call) {
  IREE_ASSERT_ARGUMENT(call);
  iree_vm_list_release(call->inputs);
  iree_vm_list_release(call->outputs);
  rocket_runtime_session_release(call->session);
  memset(call, 0, sizeof(*call));
}

IREE_API_EXPORT void rocket_runtime_call_reset(rocket_runtime_call_t* call) {
  IREE_ASSERT_ARGUMENT(call);
  iree_status_ignore(iree_vm_list_resize(call->inputs, 0));
  iree_status_ignore(iree_vm_list_resize(call->outputs, 0));
}

IREE_API_EXPORT iree_vm_list_t* rocket_runtime_call_inputs(
    const rocket_runtime_call_t* call) {
  IREE_ASSERT_ARGUMENT(call);
  return call->inputs;
}

IREE_API_EXPORT iree_vm_list_t* rocket_runtime_call_outputs(
    const rocket_runtime_call_t* call) {
  IREE_ASSERT_ARGUMENT(call);
  return call->outputs;
}

IREE_API_EXPORT iree_status_t
rocket_runtime_call_invoke(rocket_runtime_call_t* call, uint32_t flags) {
  IREE_ASSERT_ARGUMENT(call);
  if (flags != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "rocket_runtime_call_invoke flags are reserved");
  }
  return rocket_runtime_session_call(call->session, &call->function,
                                     call->inputs, call->outputs);
}

IREE_API_EXPORT iree_status_t rocket_runtime_call_inputs_push_back_buffer_view(
    rocket_runtime_call_t* call, iree_hal_buffer_view_t* buffer_view) {
  IREE_ASSERT_ARGUMENT(call);
  IREE_ASSERT_ARGUMENT(buffer_view);
  iree_vm_ref_t value = {0};
  IREE_RETURN_IF_ERROR(iree_vm_ref_wrap_assign(
      buffer_view, iree_hal_buffer_view_type(), &value));
  return iree_vm_list_push_ref_retain(call->inputs, &value);
}

IREE_API_EXPORT iree_status_t rocket_runtime_call_outputs_pop_front_buffer_view(
    rocket_runtime_call_t* call, iree_hal_buffer_view_t** out_buffer_view) {
  IREE_ASSERT_ARGUMENT(call);
  IREE_ASSERT_ARGUMENT(out_buffer_view);
  *out_buffer_view = NULL;
  iree_vm_ref_t value = {0};
  IREE_RETURN_IF_ERROR(iree_vm_list_pop_front_ref_move(call->outputs, &value));
  return iree_hal_buffer_view_check_deref(value, out_buffer_view);
}
