// Standalone test, zero IREE/MLIR dependency by design (matches
// RocketConv2dEncoding.h's own header-only, dependency-free scope) --
// compile and run directly with any C++17 compiler, no CMake/IREE build
// needed:
//   c++ -std=c++17 RocketConv2dEncoding_test.cc -o /tmp/rocket_encoding_test
//   && /tmp/rocket_encoding_test
//
// Golden bytes hand-derived from iree-rocket-hal's own known_good_shape()
// test fixture (executable_format.rs) -- the same 4x4x1->4x4x1, 1x1
// kernel, int8, all-placeholder-quantization shape used throughout that
// crate's own test suite. Verifies this C++ encoder produces byte-for-byte
// identical output to the committed Rust encode_conv_shape_v1.
#include <cstdio>
#include <cstring>

#include "RocketConv2dEncoding.h"

namespace {

int failures = 0;

void expectEq(uint8_t actual, uint8_t expected, size_t index) {
  if (actual != expected) {
    std::fprintf(stderr, "byte[%zu]: expected 0x%02x, got 0x%02x\n", index,
                 expected, actual);
    ++failures;
  }
}

}  // namespace

int main() {
  rocket::RocketConv2dShapeV1 shape;
  shape.inputWidth = 4;
  shape.inputHeight = 4;
  shape.inputChannels = 1;
  shape.outputWidth = 4;
  shape.outputHeight = 4;
  shape.outputChannels = 1;
  shape.weightsWidth = 1;
  shape.weightsHeight = 1;
  shape.stride = 1;
  shape.depthwise = false;
  shape.inputZeroPoint = 0;
  shape.outputZeroPoint = 0;
  shape.weightsZeroPoint = 0;
  shape.inputScale = 1.0f;
  shape.weightsScale = 1.0f;
  shape.outputScale = 1.0f;
  shape.truncateBits = 0;
  shape.activation = rocket::Activation::None;
  shape.activationCmp = 0;
  shape.precision = rocket::Precision::Int8;

  auto encoded = rocket::encodeConv2dShapeV1(shape);

  // clang-format off
  const uint8_t golden[rocket::kConv2dV1TotalLen] = {
      0x03,                    // tag
      0x01, 0x00, 0x00, 0x00,  // format_version = 1
      0x04, 0x00, 0x00, 0x00,  // input_width = 4
      0x04, 0x00, 0x00, 0x00,  // input_height = 4
      0x01, 0x00, 0x00, 0x00,  // input_channels = 1
      0x04, 0x00, 0x00, 0x00,  // output_width = 4
      0x04, 0x00, 0x00, 0x00,  // output_height = 4
      0x01, 0x00, 0x00, 0x00,  // output_channels = 1
      0x01, 0x00, 0x00, 0x00,  // weights_width = 1
      0x01, 0x00, 0x00, 0x00,  // weights_height = 1
      0x01, 0x00, 0x00, 0x00,  // stride = 1
      0x00, 0x00, 0x00, 0x00,  // depthwise = false
      0x00, 0x00, 0x00, 0x00,  // input_zero_point = 0
      0x00, 0x00, 0x00, 0x00,  // output_zero_point = 0
      0x00, 0x00, 0x00, 0x00,  // weights_zero_point = 0
      0x00, 0x00, 0x80, 0x3f,  // input_scale = 1.0f (0x3F800000 LE)
      0x00, 0x00, 0x80, 0x3f,  // weights_scale = 1.0f
      0x00, 0x00, 0x80, 0x3f,  // output_scale = 1.0f
      0x00, 0x00, 0x00, 0x00,  // truncate_bits = 0
      0x00, 0x00, 0x00, 0x00,  // activation_tag = 0 (None)
      0x00, 0x00, 0x00, 0x00,  // activation_cmp = 0
      0x00, 0x00, 0x00, 0x00,  // precision_tag = 0 (Int8)
  };
  // clang-format on

  static_assert(sizeof(golden) == rocket::kConv2dV1TotalLen);
  for (size_t i = 0; i < rocket::kConv2dV1TotalLen; ++i) {
    expectEq(encoded[i], golden[i], i);
  }

  // Also sanity-check a Relux + Fp16 variant to exercise the non-zero
  // activation_cmp / precision_tag=1 paths (not just the all-zeros default
  // case, which can't distinguish "field written" from "field never
  // touched").
  rocket::RocketConv2dShapeV1 shape2 = shape;
  shape2.activation = rocket::Activation::Relux;
  shape2.activationCmp = 42;
  shape2.precision = rocket::Precision::Fp16;
  auto encoded2 = rocket::encodeConv2dShapeV1(shape2);
  // activation_tag at byte offset 1(tag)+4(version)+17*4(plain fields)=73.
  expectEq(encoded2[73], 2, 73);   // Relux
  expectEq(encoded2[77], 42, 77);  // activation_cmp
  expectEq(encoded2[81], 1, 81);   // Fp16 precision_tag

  if (failures == 0) {
    std::printf("OK: %zu bytes match golden encoding\n",
                rocket::kConv2dV1TotalLen);
    return 0;
  }
  std::fprintf(stderr, "FAILED: %d byte mismatches\n", failures);
  return 1;
}
