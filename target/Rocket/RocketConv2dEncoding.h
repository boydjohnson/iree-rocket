// Header-only, zero MLIR/LLVM dependency by design -- independently
// unit-testable without linking against any of IREE's compiler libraries.
//
// Mirrors iree-rocket-hal/src/rocket/executable_format.rs's
// `encode_conv_shape_v1` byte-for-byte (verified field-by-field against
// that file): tag byte 0x03 (CONV2D_V1_TAG) followed by 84 little-endian
// bytes -- format_version, then ConvShape's 19 fields in declared struct
// order, Activation/Precision as their OWN wire-format tags (not any
// internal hardware-register encoding). See that Rust file's own module
// doc comment for the authoritative byte-layout table this must stay in
// sync with.
#ifndef ROCKET_CONV2D_ENCODING_H_
#define ROCKET_CONV2D_ENCODING_H_

#include <array>
#include <cstdint>
#include <cstring>

namespace rocket {

// Wire-format-owned tags -- deliberately distinct from any internal
// hardware-register encoding those concepts might also have on the Rust
// side (e.g. Precision::enum_value()'s CNA/CORE/DPU register bits).
enum class Activation : uint32_t { None = 0, Relu = 1, Relux = 2 };
enum class Precision : uint32_t { Int8 = 0, Fp16 = 1 };

// Mirrors iree_rocket_hal::rocket::regcmd::ConvShape's field list and
// declared order exactly.
struct RocketConv2dShapeV1 {
  uint32_t inputWidth = 0;
  uint32_t inputHeight = 0;
  uint32_t inputChannels = 0;
  uint32_t outputWidth = 0;
  uint32_t outputHeight = 0;
  uint32_t outputChannels = 0;
  uint32_t weightsWidth = 0;
  uint32_t weightsHeight = 0;
  uint32_t stride = 0;
  bool depthwise = false;
  uint32_t inputZeroPoint = 0;
  uint32_t outputZeroPoint = 0;
  uint32_t weightsZeroPoint = 0;
  float inputScale = 1.0f;
  float weightsScale = 1.0f;
  float outputScale = 1.0f;
  uint32_t truncateBits = 0;
  Activation activation = Activation::None;
  uint32_t activationCmp = 0;  // only meaningful if activation == Relux.
  Precision precision = Precision::Int8;
};

inline constexpr uint8_t kConv2dV1Tag = 3;
inline constexpr uint32_t kConv2dV1FormatVersion = 1;
inline constexpr size_t kConv2dV1PayloadLen = 84;  // 4 (version) + 20*4 (fields).
inline constexpr size_t kConv2dV1TotalLen = 1 + kConv2dV1PayloadLen;  // + tag byte.

// Encodes `shape` as [tag byte][84-byte payload], matching
// decode_conv_shape_v1's expected input byte-for-byte. Host is assumed
// little-endian (aarch64/x86_64, the only realistic build/board hosts for
// this project) -- not byte-swapped for big-endian hosts, which don't
// exist in this project's scope.
inline std::array<uint8_t, kConv2dV1TotalLen> encodeConv2dShapeV1(
    const RocketConv2dShapeV1& shape) {
  std::array<uint8_t, kConv2dV1TotalLen> out{};
  size_t pos = 0;
  out[pos++] = kConv2dV1Tag;

  auto putU32 = [&](uint32_t v) {
    std::memcpy(out.data() + pos, &v, sizeof(v));
    pos += sizeof(v);
  };
  auto putF32Bits = [&](float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    putU32(bits);
  };

  putU32(kConv2dV1FormatVersion);
  putU32(shape.inputWidth);
  putU32(shape.inputHeight);
  putU32(shape.inputChannels);
  putU32(shape.outputWidth);
  putU32(shape.outputHeight);
  putU32(shape.outputChannels);
  putU32(shape.weightsWidth);
  putU32(shape.weightsHeight);
  putU32(shape.stride);
  putU32(shape.depthwise ? 1u : 0u);
  putU32(shape.inputZeroPoint);
  putU32(shape.outputZeroPoint);
  putU32(shape.weightsZeroPoint);
  putF32Bits(shape.inputScale);
  putF32Bits(shape.weightsScale);
  putF32Bits(shape.outputScale);
  putU32(shape.truncateBits);
  putU32(static_cast<uint32_t>(shape.activation));
  putU32(shape.activationCmp);
  putU32(static_cast<uint32_t>(shape.precision));

  return out;
}

}  // namespace rocket

#endif  // ROCKET_CONV2D_ENCODING_H_
