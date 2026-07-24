// CTS backend registration for the "rocket" HAL driver (Rockchip RK3588
// NPU, /dev/accel/accel0 -- see ../src/driver.rs and ../src/device.rs).
//
// Registers "rocket" with no executable_formats. iree_hal_cts_test_suite()
// (build_tools/cmake/iree_hal_cts_test_suite.cmake) only creates the
// executable/dispatch-dependent test binaries when TESTDATA_LIBS is given,
// and CTS_REGISTER_EXECUTABLE_TEST_SUITE's accumulator pushes nothing for a
// backend with an empty executable_formats list -- so those categories are
// skipped structurally rather than failing. Every other category (driver,
// semaphore, event, allocator, buffer, command_buffer) exercises the real
// driver, real ioctls, real hardware, with zero dependency on iree-compile
// or the rocket-hal-driver executable-cache/dispatch path (still stubbed --
// see executable_cache.rs, command_buffer.rs's dispatch, device.rs's
// queue_dispatch).

#include "iree/hal/api.h"
#include "iree/hal/cts/util/registry.h"

extern "C" iree_status_t iree_hal_rocket_driver_module_register(
    iree_hal_driver_registry_t* registry);

namespace iree::hal::cts {

static iree_status_t CreateRocketDevice(
    const iree_hal_device_create_params_t* create_params,
    iree_hal_driver_t** out_driver, iree_hal_device_t** out_device) {
  iree_status_t status = iree_hal_rocket_driver_module_register(
      iree_hal_driver_registry_default());
  if (iree_status_is_already_exists(status)) {
    iree_status_free(status);
    status = iree_ok_status();
  }

  // rocket's factory_try_create (driver.rs) returns UNAVAILABLE when
  // /dev/accel/accel0 doesn't exist -- e.g. this suite running on an x86_64
  // dev machine rather than the RK3588 board. CtsRegistry treats
  // UNAVAILABLE from the factory as "backend not present" and skips.
  iree_hal_driver_t* driver = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_driver_registry_try_create(
        iree_hal_driver_registry_default(),
        iree_make_cstring_view("rocket"), iree_allocator_system(), &driver);
  }

  iree_hal_device_t* device = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_driver_create_default_device(
        driver, create_params, iree_allocator_system(), &device);
  }

  if (iree_status_is_ok(status)) {
    *out_driver = driver;
    *out_device = device;
  } else {
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
  }
  return status;
}

static bool rocket_registered_ =
    (CtsRegistry::RegisterBackend({
         "rocket",
         {"rocket", CreateRocketDevice,
          /*executable_format=*/nullptr,
          /*executable_data=*/nullptr, RecordingMode::kDirect,
          /*unsupported_tests=*/{},
          /*expected_failures=*/{}},
         /*tags=*/{},
     }),
     true);

}  // namespace iree::hal::cts
