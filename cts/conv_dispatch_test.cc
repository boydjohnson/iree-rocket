// Hand-written (not IREE's generic CTS harness -- see backends.cc's module
// doc comment for why) end-to-end test for the Conv2d ukernel, exercised
// through the REAL public IREE HAL API, mirroring
// pooling_dispatch_test.cc's structure. Two things this proves that
// iree-rocket-hal/tests/conv_hw.rs and tests/conv_activation_hw.rs can't:
// executable_cache.rs's tag-byte convention actually selects the right
// ConvShape (including the newly-added tag=2 for Precision::Fp16, see that
// module's doc comment), and command_buffer.rs's Conv2d binding convention
// (0=input, 1=weights, 2=bias, 3=output) wires correctly through the real
// command-buffer/dispatch/queue_execute path -- iree-rocket-hal's own tests
// only prove the regcmd+ioctl layer directly, bypassing this driver's HAL
// vtables entirely.
//
// The fp16 exact-product check mirrors iree-rocket-hal's
// tests/conv_hw.rs::conv_fp16_tracks_exact_product exactly (same
// hardware-confirmed shape and operand pairs -- see that test and
// rknpu-spelunking's project_conv_dtype_coverage memory for the underlying
// hardware investigation) -- the point here is confirming tag=2 threads
// Precision::Fp16 correctly through this driver's own dispatch path, not
// re-deriving the recipe.
//
// Built by the same CMake path as pooling_dispatch_test.cc
// (cts/CMakeLists.txt) -- links against a real, compiled IREE runtime.

#include <cmath>
#include <cstdint>
#include <cstring>
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

// Duplicated from pooling_dispatch_test.cc (file-local statics there, kept
// self-contained here rather than shared -- same call as that file).
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

iree_hal_buffer_t* AllocateAndFillBytes(iree_hal_device_t* device, iree_device_size_t size,
                                         const std::vector<uint8_t>& data) {
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
  CheckOk(iree_hal_buffer_map_write(buffer, 0, data.data(), size), "iree_hal_buffer_map_write");
  return buffer;
}

iree_hal_buffer_t* AllocateAndFill(iree_hal_device_t* device, iree_device_size_t size,
                                    uint8_t fill) {
  std::vector<uint8_t> data(static_cast<size_t>(size), fill);
  return AllocateAndFillBytes(device, size, data);
}

// Uniform u16-element fill -- the same "uniform fill sidesteps not knowing
// the real per-pixel packing stride" trick iree-rocket-hal's
// tests/conv_hw.rs::fill_u16 uses: every 2-byte slot (at whatever the real
// pixel stride turns out to be) reads back this exact bit pattern, so it
// doesn't matter whether the hardware's real fp16 layout is dense or
// atomic-padded.
iree_hal_buffer_t* AllocateAndFillU16(iree_hal_device_t* device, iree_device_size_t size,
                                       uint16_t fill) {
  std::vector<uint8_t> data(static_cast<size_t>(size));
  for (size_t i = 0; i + 1 < data.size(); i += 2) {
    data[i] = static_cast<uint8_t>(fill & 0xff);
    data[i + 1] = static_cast<uint8_t>((fill >> 8) & 0xff);
  }
  return AllocateAndFillBytes(device, size, data);
}

// Minimal IEEE-754 binary16 <-> f32 conversion, duplicated from
// iree-rocket-hal's tests/conv_hw.rs rather than shared (same file-local
// duplication convention that file itself already uses relative to
// conv_activation_hw.rs).
float F16ToF32(uint16_t bits) {
  uint32_t sign = (bits >> 15) & 0x1;
  uint32_t exp = (bits >> 10) & 0x1f;
  uint32_t frac = bits & 0x3ff;
  uint32_t f32_bits;
  if (exp == 0) {
    if (frac == 0) {
      f32_bits = sign << 31;
    } else {
      float subnormal = static_cast<float>(frac) * std::ldexp(1.0f, -24);
      return sign == 1 ? -subnormal : subnormal;
    }
  } else if (exp == 0x1f) {
    f32_bits = (sign << 31) | (0xffu << 23) | (frac << 13);
  } else {
    uint32_t new_exp = exp + (127 - 15);
    f32_bits = (sign << 31) | (new_exp << 23) | (frac << 13);
  }
  float result;
  memcpy(&result, &f32_bits, sizeof(result));
  return result;
}

uint16_t F32ToF16Bits(float v) {
  uint32_t bits;
  memcpy(&bits, &v, sizeof(bits));
  uint32_t sign = (bits >> 31) & 0x1;
  int32_t exp = static_cast<int32_t>((bits >> 23) & 0xff);
  uint32_t frac = bits & 0x7fffff;
  int32_t new_exp = exp - 127 + 15;
  EXPECT_GE(new_exp, 1) << "value out of easy f16 range";
  EXPECT_LT(new_exp, 31) << "value out of easy f16 range";
  return static_cast<uint16_t>((sign << 15) | (static_cast<uint32_t>(new_exp) << 10) |
                               (frac >> 13));
}

// Runs the Conv2d ukernel (executable_cache.rs's tag-byte convention) end
// to end through the real command-buffer/dispatch/queue_execute path.
// `tag` selects which hardcoded ConvShape (0=int8, 2=fp16 -- see that
// module's doc comment); `input`/`weights` are the raw byte contents this
// test wants in those two buffers, uniformly filled by the caller so the
// real per-pixel packing stride doesn't need to be known.
std::vector<uint8_t> RunConv(iree_hal_device_t* device, uint8_t tag,
                             iree_hal_buffer_t* input, iree_hal_buffer_t* weights) {
  constexpr iree_device_size_t kSize = 4096;
  iree_hal_buffer_t* bias = AllocateAndFill(device, kSize, 0);
  iree_hal_buffer_t* output = AllocateAndFill(device, kSize, 0);

  iree_hal_executable_cache_t* cache = nullptr;
  CheckOk(iree_hal_executable_cache_create(device, iree_make_cstring_view("test"), &cache),
          "iree_hal_executable_cache_create");

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

  // 0=input, 1=weights, 2=bias, 3=output -- command_buffer.rs's Conv2d
  // binding convention.
  iree_hal_buffer_ref_t refs[4] = {
      iree_hal_make_buffer_ref(input, 0, kSize),
      iree_hal_make_buffer_ref(weights, 0, kSize),
      iree_hal_make_buffer_ref(bias, 0, kSize),
      iree_hal_make_buffer_ref(output, 0, kSize),
  };
  iree_hal_buffer_ref_list_t bindings;
  bindings.count = 4;
  bindings.values = refs;
  iree_hal_dispatch_config_t config;
  memset(&config, 0, sizeof(config));
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

  // Not iree_hal_buffer_map_read() -- see pooling_dispatch_test.cc's
  // comment on the same call for why manual map+invalidate is required on
  // rocket's non-coherent memory.
  std::vector<uint8_t> result(4, 0);
  {
    iree_hal_buffer_mapping_t mapping;
    CheckOk(iree_hal_buffer_map_range(output, IREE_HAL_MAPPING_MODE_SCOPED,
                                       IREE_HAL_MEMORY_ACCESS_READ, 0, 4, &mapping),
            "iree_hal_buffer_map_range");
    CheckOk(iree_hal_buffer_mapping_invalidate_range(&mapping, 0, IREE_HAL_WHOLE_BUFFER),
            "iree_hal_buffer_mapping_invalidate_range");
    memcpy(result.data(), mapping.contents.data, 4);
    CheckOk(iree_hal_buffer_unmap_range(&mapping), "iree_hal_buffer_unmap_range");
  }

  iree_hal_semaphore_release(sem);
  iree_hal_command_buffer_release(cb);
  iree_hal_executable_release(executable);
  iree_hal_executable_cache_release(cache);
  iree_hal_buffer_release(bias);
  iree_hal_buffer_release(output);
  return result;
}

float RunConvFp16(iree_hal_device_t* device, float input_fill, float weight_fill) {
  constexpr iree_device_size_t kSize = 4096;
  iree_hal_buffer_t* input = AllocateAndFillU16(device, kSize, F32ToF16Bits(input_fill));
  iree_hal_buffer_t* weights = AllocateAndFillU16(device, kSize, F32ToF16Bits(weight_fill));
  std::vector<uint8_t> raw = RunConv(device, /*tag=*/2, input, weights);
  iree_hal_buffer_release(input);
  iree_hal_buffer_release(weights);
  uint16_t bits = static_cast<uint16_t>(raw[0]) | (static_cast<uint16_t>(raw[1]) << 8);
  return F16ToF32(bits);
}

TEST(RocketConvDispatch, Fp16TracksExactProduct) {
  iree_hal_driver_t* driver = nullptr;
  iree_hal_device_t* device = nullptr;
  std::string error;
  if (!CreateDevice(&driver, &device, &error)) {
    GTEST_SKIP() << "rocket device unavailable: " << error;
  }

  // Same operand pairs as iree-rocket-hal's
  // tests/conv_hw.rs::conv_fp16_tracks_exact_product -- chosen so both
  // operands and their product are exactly fp16-representable, so this
  // asserts exact equality, not a tolerance.
  struct Case {
    float input_fill;
    float weight_fill;
    float expected;
  };
  for (const Case& c : {Case{10.5f, 0.25f, 2.625f}, Case{3.0f, 2.0f, 6.0f},
                        Case{4.0f, 4.0f, 16.0f}}) {
    float result = RunConvFp16(device, c.input_fill, c.weight_fill);
    EXPECT_EQ(result, c.expected)
        << "input_fill=" << c.input_fill << ", weight_fill=" << c.weight_fill;
  }

  iree_hal_device_release(device);
  iree_hal_driver_release(driver);
}

TEST(RocketConvDispatch, Int8DefaultShapeCompletes) {
  iree_hal_driver_t* driver = nullptr;
  iree_hal_device_t* device = nullptr;
  std::string error;
  if (!CreateDevice(&driver, &device, &error)) {
    GTEST_SKIP() << "rocket device unavailable: " << error;
  }

  // Deliberately not asserting a specific output value here: tag=0's
  // Precision::Int8 default shape has a known, still-open numeric bug for
  // nonzero real weights (see rknpu-spelunking's
  // project_conv_dtype_coverage memory -- the identity-weight
  // investigation), and this shape's zero_point=0 raw-encoding convention
  // hasn't itself been independently confirmed against real hardware the
  // way conv_with_add_hw.rs's 0x80-convention shape has. This test's only
  // job is confirming tag=0 still dispatches successfully through the
  // real HAL path after the Precision field was added to ConvShape --
  // regcmd.rs's own hardware tests (conv_hw.rs, conv_precision_hw
  // diagnostics) are where the numeric question itself is tracked.
  constexpr iree_device_size_t kSize = 4096;
  iree_hal_buffer_t* input = AllocateAndFill(device, kSize, 200);
  iree_hal_buffer_t* weights = AllocateAndFill(device, kSize, 2);
  std::vector<uint8_t> raw = RunConv(device, /*tag=*/0, input, weights);
  iree_hal_buffer_release(input);
  iree_hal_buffer_release(weights);
  (void)raw;  // Dispatch completing without CheckOk aborting is the assertion.

  iree_hal_device_release(device);
  iree_hal_driver_release(driver);
}

}  // namespace
