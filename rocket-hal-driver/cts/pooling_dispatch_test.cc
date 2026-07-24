// Hand-written (not IREE's generic CTS harness) end-to-end test for the
// standalone-PPU pooling ukernel, exercised through the REAL public IREE
// HAL API (iree_hal_command_buffer_dispatch / iree_hal_device_queue_execute)
// rather than iree-rocket-hal's raw regcmd+ioctl test
// (iree-rocket-hal/tests/pooling_hw.rs). This is what actually proves
// executable_cache.rs's tag-byte convention, command_buffer.rs's Pooling
// binding convention (0=input, 1=output), and device.rs's queue_execute
// real-GEM-handle fix all wire together correctly end to end --
// conv_hw.rs/pooling_hw.rs only prove the regcmd+ioctl layer in isolation,
// bypassing this driver's HAL vtables entirely.
//
// Deliberately NOT using IREE's generic CTS harness (CtsRegistry /
// CommandBufferDispatchTest) -- see backends.cc's module doc comment for
// why: this driver has no real compiler/executable-format, so
// executable_format is registered as nullptr there and CTS's own
// dispatch-test suite is structurally skipped. This file drives the same
// real API calls by hand instead.
//
// Built by the same CMake path as backends.cc (cts/CMakeLists.txt) --
// links against a real, compiled IREE runtime. Unlike a plain `cargo
// test` (which can only reach this driver's own vtable functions
// directly), the generic iree_hal_* trampolines this file calls
// (iree_hal_command_buffer_create, iree_hal_buffer_map_write,
// iree_hal_device_queue_execute, ...) have no linkable definition outside
// a real IREE build -- see rocket-hal-driver's own build.rs, which only
// runs bindgen against IREE's headers and never links IREE's compiled
// runtime itself.

#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/threading/numa.h"
#include "iree/hal/api.h"

extern "C" iree_status_t iree_hal_rocket_driver_module_register(
    iree_hal_driver_registry_t* registry);

namespace {

void CheckOk(iree_status_t status, const char* what) {
  if (!iree_status_is_ok(status)) {
    iree_allocator_t allocator = iree_allocator_system();
    char* message = nullptr;
    iree_host_size_t length = 0;
    iree_status_to_string(status, &allocator, &message, &length);
    ADD_FAILURE() << what << " failed: " << (message ? message : "?");
    if (message) iree_allocator_free(allocator, message);
    iree_status_free(status);
    FAIL();
  }
}

// Same device-creation steps as backends.cc's CreateRocketDevice --
// duplicated (not shared -- that one has file-local `static` linkage) to
// keep this file self-contained. Returns false (does not fail the test)
// if the rocket device is unavailable, matching CtsRegistry's own
// UNAVAILABLE-means-skip convention. On failure, `out_error` gets the real
// iree_status_t message (caller's responsibility to free) -- previously
// this discarded the status entirely and the caller's GTEST_SKIP() printed
// a fixed, guessed reason ("no /dev/accel/accel0?") regardless of what
// actually failed, which was actively misleading once the device file
// turned out to exist and the kernel module was loaded.
bool CreateDevice(iree_hal_driver_t** out_driver, iree_hal_device_t** out_device,
                   std::string* out_error) {
  iree_status_t status =
      iree_hal_rocket_driver_module_register(iree_hal_driver_registry_default());
  if (iree_status_is_already_exists(status)) {
    iree_status_free(status);
    status = iree_ok_status();
  }

  iree_hal_driver_t* driver = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_driver_registry_try_create(
        iree_hal_driver_registry_default(), iree_make_cstring_view("rocket"),
        iree_allocator_system(), &driver);
  }

  // device.rs's create() (device.rs:294-299) hard-requires a non-null
  // create_params->proactor_pool -- IREE_ASSERT_ARGUMENT-equivalent, per
  // its own comment, matching iree-null-driver-reference/device.c. A
  // zeroed iree_hal_device_create_params_t (this function's previous
  // behavior) leaves it null, which is exactly what surfaced as a bare,
  // message-less INVALID_ARGUMENT status once CreateDevice's real error
  // got surfaced instead of being discarded. Build a real pool per
  // proactor_pool.h's own documented "Typical usage" recipe.
  iree_async_proactor_pool_t* proactor_pool = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_async_proactor_pool_create(
        iree_numa_node_count(), /*node_ids=*/nullptr,
        iree_async_proactor_pool_options_default(), iree_allocator_system(),
        &proactor_pool);
  }

  iree_hal_device_t* device = nullptr;
  if (iree_status_is_ok(status)) {
    iree_hal_device_create_params_t params = iree_hal_device_create_params_default();
    params.proactor_pool = proactor_pool;
    status = iree_hal_driver_create_default_device(driver, &params,
                                                    iree_allocator_system(), &device);
  }
  // The device retains the pool on success (proactor_pool.h: "Device
  // retains the pool — caller can release immediately"); release our
  // reference either way, matching the header's documented usage.
  if (proactor_pool) {
    iree_async_proactor_pool_release(proactor_pool);
  }

  if (!iree_status_is_ok(status)) {
    iree_allocator_t allocator = iree_allocator_system();
    char* message = nullptr;
    iree_host_size_t length = 0;
    iree_status_to_string(status, &allocator, &message, &length);
    *out_error = message ? std::string(message, length) : std::string("(no message)");
    if (message) iree_allocator_free(allocator, message);
    iree_status_free(status);
    if (driver) iree_hal_driver_release(driver);
    return false;
  }
  *out_driver = driver;
  *out_device = device;
  return true;
}

iree_hal_buffer_t* AllocateAndFill(iree_hal_device_t* device, iree_device_size_t size,
                                    uint8_t fill) {
  iree_hal_buffer_params_t params;
  memset(&params, 0, sizeof(params));
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                 IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL;

  iree_hal_buffer_t* buffer = nullptr;
  CheckOk(iree_hal_allocator_allocate_buffer(iree_hal_device_allocator(device), params, size,
                                              &buffer),
          "iree_hal_allocator_allocate_buffer");

  std::vector<uint8_t> data(static_cast<size_t>(size), fill);
  CheckOk(iree_hal_buffer_map_write(buffer, 0, data.data(), size), "iree_hal_buffer_map_write");
  return buffer;
}

// Runs the Pooling ukernel (executable_cache.rs's tag-byte convention,
// tag=1 -- see that module's doc comment) end to end through the real
// command-buffer/dispatch/queue_execute path and returns the first output
// byte.
uint8_t RunPooling(iree_hal_device_t* device, uint8_t input_fill) {
  constexpr iree_device_size_t kSize = 4096;
  iree_hal_buffer_t* input = AllocateAndFill(device, kSize, input_fill);
  iree_hal_buffer_t* output = AllocateAndFill(device, kSize, 0);

  iree_hal_executable_cache_t* cache = nullptr;
  CheckOk(iree_hal_executable_cache_create(device, iree_make_cstring_view("test"), &cache),
          "iree_hal_executable_cache_create");

  uint8_t tag = 1;  // Pooling.
  iree_hal_executable_params_t exec_params;
  memset(&exec_params, 0, sizeof(exec_params));
  exec_params.executable_format = iree_make_cstring_view("");
  exec_params.executable_data = iree_make_const_byte_span(&tag, 1);
  iree_hal_executable_t* executable = nullptr;
  CheckOk(iree_hal_executable_cache_prepare_executable(cache, &exec_params, &executable),
          "iree_hal_executable_cache_prepare_executable");

  iree_hal_command_buffer_t* cb = nullptr;
  CheckOk(iree_hal_command_buffer_create(device, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
                                          IREE_HAL_COMMAND_CATEGORY_DISPATCH,
                                          IREE_HAL_QUEUE_AFFINITY_ANY,
                                          /*binding_capacity=*/0, &cb),
          "iree_hal_command_buffer_create");
  CheckOk(iree_hal_command_buffer_begin(cb), "iree_hal_command_buffer_begin");

  iree_hal_buffer_ref_t refs[2] = {
      iree_hal_make_buffer_ref(input, 0, kSize),
      iree_hal_make_buffer_ref(output, 0, kSize),
  };
  iree_hal_buffer_ref_list_t bindings;
  bindings.count = 2;
  bindings.values = refs;
  iree_hal_dispatch_config_t config;
  memset(&config, 0, sizeof(config));
  // iree_hal_command_buffer_dispatch (command_buffer.c) treats an all-zero
  // static workgroup_count as an intentional no-op dispatch and returns
  // iree_ok_status() WITHOUT ever calling this driver's vtable dispatch --
  // "no (intentional) side-effects" is the exact comment there. This
  // driver's dispatch (command_buffer.rs) ignores workgroup_count entirely
  // (each executable is a single fixed-shape ukernel, not a grid), so any
  // non-zero value here is fine -- it only needs to not read as "no work".
  config.workgroup_count[0] = 1;
  config.workgroup_count[1] = 1;
  config.workgroup_count[2] = 1;
  iree_hal_executable_function_t function;
  function.value = 0;
  CheckOk(iree_hal_command_buffer_dispatch(cb, executable, function, config,
                                            iree_const_byte_span_empty(), bindings,
                                            IREE_HAL_DISPATCH_FLAG_NONE),
          "iree_hal_command_buffer_dispatch");
  CheckOk(iree_hal_command_buffer_end(cb), "iree_hal_command_buffer_end");

  iree_hal_semaphore_t* sem = nullptr;
  CheckOk(iree_hal_semaphore_create(device, IREE_HAL_QUEUE_AFFINITY_ANY, 0,
                                     IREE_HAL_SEMAPHORE_FLAG_NONE, &sem),
          "iree_hal_semaphore_create");
  uint64_t signal_value = 1;
  iree_hal_semaphore_list_t signal_list;
  signal_list.count = 1;
  signal_list.semaphores = &sem;
  signal_list.payload_values = &signal_value;
  CheckOk(iree_hal_device_queue_execute(device, IREE_HAL_QUEUE_AFFINITY_ANY,
                                         iree_hal_semaphore_list_empty(), signal_list, cb,
                                         iree_hal_buffer_binding_table_empty(),
                                         IREE_HAL_EXECUTE_FLAG_NONE),
          "iree_hal_device_queue_execute");
  CheckOk(iree_hal_semaphore_wait(sem, 1, iree_infinite_timeout(),
                                   IREE_ASYNC_WAIT_FLAG_NONE),
          "iree_hal_semaphore_wait");

  // Not iree_hal_buffer_map_read(): that convenience wrapper never calls
  // iree_hal_buffer_mapping_invalidate_range (confirmed by inspection --
  // no caller anywhere in iree/hal/buffer.c) regardless of memory type, so
  // it never triggers buffer.rs's invalidate_range (-> device::prep_bo)
  // that actually makes the PPU's DMA write visible to this CPU read on
  // rocket's genuinely non-coherent memory. Map manually and invalidate
  // before reading instead -- same fix class as the missing fini_bo(buf_out)
  // bug already found and fixed in iree-rocket-hal/tests/{pooling,
  // conv_then_pooling}_hw.rs, one layer up the stack.
  uint8_t result = 0;
  {
    iree_hal_buffer_mapping_t mapping;
    CheckOk(iree_hal_buffer_map_range(output, IREE_HAL_MAPPING_MODE_SCOPED,
                                       IREE_HAL_MEMORY_ACCESS_READ, 0, 1, &mapping),
            "iree_hal_buffer_map_range");
    CheckOk(iree_hal_buffer_mapping_invalidate_range(&mapping, 0, IREE_HAL_WHOLE_BUFFER),
            "iree_hal_buffer_mapping_invalidate_range");
    result = mapping.contents.data[0];
    CheckOk(iree_hal_buffer_unmap_range(&mapping), "iree_hal_buffer_unmap_range");
  }

  iree_hal_semaphore_release(sem);
  iree_hal_command_buffer_release(cb);
  iree_hal_executable_release(executable);
  iree_hal_executable_cache_release(cache);
  iree_hal_buffer_release(input);
  iree_hal_buffer_release(output);
  return result;
}

TEST(RocketPoolingDispatch, OutputTracksInput) {
  iree_hal_driver_t* driver = nullptr;
  iree_hal_device_t* device = nullptr;
  std::string error;
  if (!CreateDevice(&driver, &device, &error)) {
    GTEST_SKIP() << "rocket device unavailable: " << error;
  }

  uint8_t low = RunPooling(device, 10);
  uint8_t high = RunPooling(device, 200);
  EXPECT_NE(low, high) << "pooling output didn't change between input_fill=10 (" << int(low)
                        << ") and input_fill=200 (" << int(high)
                        << ") -- suggests the dispatch isn't really reading the input";

  iree_hal_device_release(device);
  iree_hal_driver_release(driver);
}

}  // namespace
