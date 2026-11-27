// Copyright 2026 The iree-rocket Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Embedding API for running rocket-compiler modules from C and C++.
//
// Every module rocket-compiler produces targets TWO devices: the RK3588 NPU
// (`rocket`) for the operations it claims, and the CPU (`local`) for the rest.
// IREE's own convenience API, iree_runtime_session_create_with_device(), takes
// exactly one device, so it cannot load these modules. This header is the
// replacement: a session that owns both devices as one IREE device group, the
// same group `iree-run-module --device=rocket --device=local-task` builds.
//
// Everything else is plain IREE: the caller creates an iree_runtime_instance_t,
// allocates inputs with iree_hal_buffer_view_* against
// rocket_runtime_session_device_allocator(), and reads outputs back the usual
// way. The functions here mirror iree/runtime/session.h and call.h so that code
// written against those ports by renaming.
//
// Minimal use:
//
//   iree_runtime_instance_options_t instance_options;
//   iree_runtime_instance_options_initialize(&instance_options);
//   iree_runtime_instance_options_use_all_available_drivers(&instance_options);
//   iree_runtime_instance_t* instance = NULL;
//   IREE_CHECK_OK(iree_runtime_instance_create(
//       &instance_options, iree_allocator_system(), &instance));
//
//   rocket_runtime_session_options_t options;
//   rocket_runtime_session_options_initialize(&options);
//   rocket_runtime_session_t* session = NULL;
//   IREE_CHECK_OK(rocket_runtime_session_create(
//       instance, &options, iree_allocator_system(), &session));
//   IREE_CHECK_OK(rocket_runtime_session_append_bytecode_module_from_file(
//       session, "model.vmfb"));
//
//   rocket_runtime_call_t call;
//   IREE_CHECK_OK(rocket_runtime_call_initialize_by_name(
//       session, iree_make_cstring_view("module.main_graph"), &call));
//   IREE_CHECK_OK(rocket_runtime_call_inputs_push_back_buffer_view(&call, in));
//   IREE_CHECK_OK(rocket_runtime_call_invoke(&call, /*flags=*/0));
//   IREE_CHECK_OK(rocket_runtime_call_outputs_pop_front_buffer_view(&call, &out));
//   rocket_runtime_call_deinitialize(&call);
//
//   rocket_runtime_session_release(session);
//   iree_runtime_instance_release(instance);
//
// The module must come from the rocket-compiler (and therefore the IREE
// version) this library was built with. IREE has no stable HAL or bytecode ABI
// across releases; a mismatched module fails to load rather than misbehaving,
// but it does fail.

#ifndef ROCKET_RUNTIME_H_
#define ROCKET_RUNTIME_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/runtime/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Version of this header's API, bumped on any incompatible change. Compare
// against rocket_runtime_api_version() to detect a header/library mismatch.
#define ROCKET_RUNTIME_API_VERSION 1

// Returns ROCKET_RUNTIME_API_VERSION as the library was compiled.
IREE_API_EXPORT uint32_t rocket_runtime_api_version(void);

//===----------------------------------------------------------------------===//
// rocket_runtime_session_options_t
//===----------------------------------------------------------------------===//

// Which devices the session creates.
typedef enum rocket_runtime_devices_e {
  // The NPU (device 0, the lead) and the CPU (device 1). What every
  // rocket-compiler module that offloads anything needs. Fails if the NPU
  // (/dev/accel/accel*) cannot be opened.
  ROCKET_RUNTIME_DEVICES_NPU_AND_CPU = 0,
  // The CPU alone. Enough for a module compiled with `--no-offload`, and the
  // only option on a machine without the NPU; a module that offloads fails to
  // load with this set.
  ROCKET_RUNTIME_DEVICES_CPU_ONLY = 1,
} rocket_runtime_devices_t;

// Which CPUs the CPU device's worker threads run on.
//
// This matters more than it looks on the RK3588 (4x Cortex-A55 + 4x A76):
// IREE's default topology places workers on whichever cluster it likes, and
// MobileNetV2 fp16 measured 105 ms that way against 60-63 ms pinned to the big
// cores. The default here pins.
typedef enum rocket_runtime_cpu_policy_e {
  // One worker per CPU sharing the machine's highest
  // /sys/devices/system/cpu/cpuN/cpu_capacity (cpu4-7 on the RK3588). On a
  // machine with uniform or unpublished capacities this is
  // ROCKET_RUNTIME_CPU_POLICY_IREE_DEFAULT.
  ROCKET_RUNTIME_CPU_POLICY_FASTEST = 0,
  // One worker per CPU in |cpu_ids|.
  ROCKET_RUNTIME_CPU_POLICY_LIST = 1,
  // IREE's own selection. It honours the --task_topology_* flags when the
  // application has parsed IREE flags (iree_flags_parse), and is IREE's
  // physical-core default otherwise.
  ROCKET_RUNTIME_CPU_POLICY_IREE_DEFAULT = 2,
} rocket_runtime_cpu_policy_t;

typedef struct rocket_runtime_session_options_t {
  rocket_runtime_devices_t devices;
  rocket_runtime_cpu_policy_t cpu_policy;
  // CPUs for ROCKET_RUNTIME_CPU_POLICY_LIST in the `taskset -c` spelling, e.g.
  // "4-7" or "0-3,6". Ignored under the other policies. Only read during
  // rocket_runtime_session_create().
  const char* cpu_ids;
  // Flags for the VM context the session's modules are loaded into.
  iree_vm_context_flags_t context_flags;
} rocket_runtime_session_options_t;

// Initializes |out_options| to the defaults: NPU and CPU, workers on the
// fastest CPUs.
IREE_API_EXPORT void rocket_runtime_session_options_initialize(
    rocket_runtime_session_options_t* out_options);

//===----------------------------------------------------------------------===//
// rocket_runtime_session_t
//===----------------------------------------------------------------------===//

// A VM context holding the HAL module, bound to the session's device group,
// plus whatever user modules are appended to it. Thread-compatible, like
// iree_runtime_session_t: calls into one session must be externally
// synchronized. Separate sessions may run concurrently.
typedef struct rocket_runtime_session_t rocket_runtime_session_t;

// Creates a session on |instance| with the devices |options| asks for.
//
// The NPU device comes from |instance|'s driver registry, so the instance must
// have been created with iree_runtime_instance_options_use_all_available_drivers
// (or otherwise have the "rocket" driver registered). The CPU device is built
// directly with the worker placement |options| selects.
//
// |out_session| must be released by the caller.
IREE_API_EXPORT iree_status_t rocket_runtime_session_create(
    iree_runtime_instance_t* instance,
    const rocket_runtime_session_options_t* options,
    iree_allocator_t host_allocator, rocket_runtime_session_t** out_session);

IREE_API_EXPORT void rocket_runtime_session_retain(
    rocket_runtime_session_t* session);
IREE_API_EXPORT void rocket_runtime_session_release(
    rocket_runtime_session_t* session);

IREE_API_EXPORT iree_allocator_t
rocket_runtime_session_host_allocator(const rocket_runtime_session_t* session);

IREE_API_EXPORT iree_runtime_instance_t* rocket_runtime_session_instance(
    const rocket_runtime_session_t* session);

// The VM context, for callers that want iree_vm_* directly.
IREE_API_EXPORT iree_vm_context_t* rocket_runtime_session_context(
    const rocket_runtime_session_t* session);

// The session's device group. Valid for the lifetime of the session.
IREE_API_EXPORT iree_hal_device_group_t* rocket_runtime_session_device_group(
    const rocket_runtime_session_t* session);

// Number of devices in the session: 2 for ROCKET_RUNTIME_DEVICES_NPU_AND_CPU,
// 1 for ROCKET_RUNTIME_DEVICES_CPU_ONLY.
IREE_API_EXPORT iree_host_size_t
rocket_runtime_session_device_count(const rocket_runtime_session_t* session);

// The device at |index|, or NULL if out of range. Index 0 is the lead device:
// the NPU when present, otherwise the CPU. Not retained for the caller.
IREE_API_EXPORT iree_hal_device_t* rocket_runtime_session_device(
    const rocket_runtime_session_t* session, iree_host_size_t index);

// The lead device's allocator: the one to allocate input buffers with, as
// iree-run-module does. Not retained for the caller.
IREE_API_EXPORT iree_hal_allocator_t* rocket_runtime_session_device_allocator(
    const rocket_runtime_session_t* session);

// Appends |module| to the session's context. Modules are resolved in the order
// appended, so a module must come after the modules it imports.
IREE_API_EXPORT iree_status_t rocket_runtime_session_append_module(
    rocket_runtime_session_t* session, iree_vm_module_t* module);

// Loads a .vmfb from |flatbuffer_data| and appends it. |flatbuffer_allocator|
// frees the data once the module no longer needs it, and is called even when
// loading fails; pass iree_allocator_null() for data the caller keeps alive
// for the session's lifetime.
IREE_API_EXPORT iree_status_t
rocket_runtime_session_append_bytecode_module_from_memory(
    rocket_runtime_session_t* session, iree_const_byte_span_t flatbuffer_data,
    iree_allocator_t flatbuffer_allocator);

// Reads a .vmfb from |file_path| and appends it.
IREE_API_EXPORT iree_status_t
rocket_runtime_session_append_bytecode_module_from_file(
    rocket_runtime_session_t* session, const char* file_path);

// Resolves a function such as "module.main_graph".
IREE_API_EXPORT iree_status_t rocket_runtime_session_lookup_function(
    const rocket_runtime_session_t* session, iree_string_view_t full_name,
    iree_vm_function_t* out_function);

// Synchronously invokes |function| with |input_list|, appending its results to
// |output_list|.
IREE_API_EXPORT iree_status_t rocket_runtime_session_call(
    rocket_runtime_session_t* session, const iree_vm_function_t* function,
    iree_vm_list_t* input_list, iree_vm_list_t* output_list);

IREE_API_EXPORT iree_status_t rocket_runtime_session_call_by_name(
    rocket_runtime_session_t* session, iree_string_view_t full_name,
    iree_vm_list_t* input_list, iree_vm_list_t* output_list);

// Asks every module to drop caches it can rebuild (IREE_VM_SIGNAL_LOW_MEMORY).
IREE_API_EXPORT iree_status_t
rocket_runtime_session_trim(rocket_runtime_session_t* session);

//===----------------------------------------------------------------------===//
// rocket_runtime_call_t
//===----------------------------------------------------------------------===//

// One function plus reusable argument and result lists; the counterpart of
// iree_runtime_call_t. Initialize once, then per invocation: push inputs,
// invoke, pop outputs, and rocket_runtime_call_reset() before the next.
typedef struct rocket_runtime_call_t {
  rocket_runtime_session_t* session;
  iree_vm_function_t function;
  iree_vm_list_t* inputs;
  iree_vm_list_t* outputs;
} rocket_runtime_call_t;

IREE_API_EXPORT iree_status_t rocket_runtime_call_initialize(
    rocket_runtime_session_t* session, iree_vm_function_t function,
    rocket_runtime_call_t* out_call);

IREE_API_EXPORT iree_status_t rocket_runtime_call_initialize_by_name(
    rocket_runtime_session_t* session, iree_string_view_t full_name,
    rocket_runtime_call_t* out_call);

IREE_API_EXPORT void rocket_runtime_call_deinitialize(
    rocket_runtime_call_t* call);

// Clears the input and output lists for another invocation.
IREE_API_EXPORT void rocket_runtime_call_reset(rocket_runtime_call_t* call);

IREE_API_EXPORT iree_vm_list_t* rocket_runtime_call_inputs(
    const rocket_runtime_call_t* call);
IREE_API_EXPORT iree_vm_list_t* rocket_runtime_call_outputs(
    const rocket_runtime_call_t* call);

// |flags| is reserved and must be 0.
IREE_API_EXPORT iree_status_t
rocket_runtime_call_invoke(rocket_runtime_call_t* call, uint32_t flags);

// Appends |buffer_view| to the inputs, retaining it.
IREE_API_EXPORT iree_status_t rocket_runtime_call_inputs_push_back_buffer_view(
    rocket_runtime_call_t* call, iree_hal_buffer_view_t* buffer_view);

// Moves the first output out as a buffer view; the caller must release it.
IREE_API_EXPORT iree_status_t rocket_runtime_call_outputs_pop_front_buffer_view(
    rocket_runtime_call_t* call, iree_hal_buffer_view_t** out_buffer_view);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // ROCKET_RUNTIME_H_
