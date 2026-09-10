/* Copyright 2026
 *
 * Licensed under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * C ABI over rocket-core's convolution/matmul planner, for callers that are
 * not Rust: today the IREE compiler plugin (COMPILER_ROADMAP.md section 2).
 *
 * Contract:
 *   - Every struct starts with `struct_size`; the caller sets it to
 *     sizeof(the struct it compiled against). A mismatch is
 *     ROCKET_PLAN_INVALID_ARGUMENT, never a misread.
 *   - Dimensions are 64-bit on the way in. The planner narrows them with
 *     checked conversions; an MLIR extent that does not fit the hardware's
 *     32-bit fields is a refusal, not a truncation.
 *   - Nothing allocated in Rust crosses the boundary. Messages are written
 *     into a caller-owned buffer and NUL-terminated (truncated to fit).
 *   - No panic unwinds across the boundary; one that is caught reports
 *     ROCKET_PLAN_INTERNAL with the panic message.
 *   - The ABI is versioned by ROCKET_PLAN_ABI_VERSION; a caller compares it
 *     with rocket_plan_abi_version() before trusting any layout below.
 */
#ifndef ROCKET_PLAN_H_
#define ROCKET_PLAN_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ROCKET_PLAN_ABI_VERSION 5u

/* Mirrors rocket_core::conv::Precision. Values are part of the ABI. */
typedef enum rocket_plan_precision_e {
  ROCKET_PLAN_PRECISION_FP16 = 0,
  ROCKET_PLAN_PRECISION_FP16_ACCUMULATOR = 1,
  ROCKET_PLAN_PRECISION_BF16 = 2,
  ROCKET_PLAN_PRECISION_INT16 = 3,
  ROCKET_PLAN_PRECISION_TF32 = 4,
  ROCKET_PLAN_PRECISION_INT4 = 5,
  ROCKET_PLAN_PRECISION_INT8 = 6,
  ROCKET_PLAN_PRECISION_INT8_ACCUMULATOR = 7,
} rocket_plan_precision_e;

/* Mirrors rocket_core::conv::Activation. */
typedef enum rocket_plan_activation_e {
  ROCKET_PLAN_ACTIVATION_NONE = 0,
  ROCKET_PLAN_ACTIVATION_RELU = 1,
  /* clamp(x, 0, ceiling); `activation_ceiling` carries the ceiling. */
  ROCKET_PLAN_ACTIVATION_CLAMPED = 2,
} rocket_plan_activation_e;

/* Mirrors rocket_core::error::PlanErrorCode, plus the three the boundary
 * itself produces. */
typedef enum rocket_plan_status_e {
  ROCKET_PLAN_OK = 0,
  ROCKET_PLAN_INVALID_SHAPE = 1,
  ROCKET_PLAN_UNSUPPORTED_SEMANTICS = 2,
  ROCKET_PLAN_HARDWARE_LIMIT = 3,
  ROCKET_PLAN_CAPACITY_EXCEEDED = 4,
  ROCKET_PLAN_UNVALIDATED_CONFIGURATION = 5,
  ROCKET_PLAN_INTERNAL = 6,
  /* A malformed call: null pointer, wrong struct_size, enum out of range. */
  ROCKET_PLAN_INVALID_ARGUMENT = 7,
  /* rocket_plan_chain_identity: the consumer cannot read the producer's
   * cube in place; the message says which of the identity's conditions
   * fails. Added in ABI version 4. */
  ROCKET_PLAN_LAYOUT_MISMATCH = 8,
} rocket_plan_status_e;

/* A convolution whose input has this many channels or fewer reads dense
 * ARGB rather than a feature cube (rocket_core::conv::MAX_DENSE_CHANNELS,
 * FeatureLayout::Dense): such an input can never chain. Added in ABI
 * version 5; the value is part of the ABI. */
#define ROCKET_PLAN_MAX_DENSE_CHANNELS 4u

/* Mirrors rocket_core::layout::CubeKind: which unit reads or writes a
 * feature cube, which picks its channel padding and surface stride. Added
 * in ABI version 4. */
typedef enum rocket_plan_cube_kind_e {
  ROCKET_PLAN_CUBE_CONV = 0,
  ROCKET_PLAN_CUBE_MATMUL = 1,
  ROCKET_PLAN_CUBE_POOLING = 2,
  ROCKET_PLAN_CUBE_ELEMENTWISE = 3,
} rocket_plan_cube_kind_e;

/* int8 requantization parameters; read only for the two INT8 precisions. */
typedef struct rocket_plan_quantization_t {
  int32_t input_zero_point;
  int32_t output_zero_point;
  int32_t weight_zero_point;
  float input_scale;
  float weights_scale;
  float output_scale;
} rocket_plan_quantization_t;

/* A whole 2-D convolution, in the planner's logical NHWC terms. */
typedef struct rocket_plan_conv_desc_t {
  uint32_t struct_size;
  uint32_t precision;  /* rocket_plan_precision_e */
  uint64_t width;
  uint64_t height;
  uint64_t in_channels;
  uint64_t out_channels;
  uint64_t stride;      /* same on both axes */
  uint64_t kernel_height;
  uint64_t kernel_width;
  /* Leading zero padding the CNA applies. -1 on both selects the planner's
   * default, `kernel / 2` per axis. */
  int64_t pad_top;
  int64_t pad_left;
  uint32_t activation;  /* rocket_plan_activation_e */
  float activation_ceiling;
  uint8_t depthwise;
  uint8_t reserved_[7];
  rocket_plan_quantization_t quantization;
} rocket_plan_conv_desc_t;

/* `[m, k] x [k, n]`, planned through the captured height-one 1x1 lowering. */
typedef struct rocket_plan_matmul_desc_t {
  uint32_t struct_size;
  uint32_t precision;  /* rocket_plan_precision_e */
  uint64_t m;
  uint64_t k;
  uint64_t n;
  uint32_t activation;  /* rocket_plan_activation_e */
  float activation_ceiling;
  rocket_plan_quantization_t quantization;
} rocket_plan_matmul_desc_t;

/* A shape class the compiler is asking the admission envelope about
 * (rocket_core::admission). No spatial extents and no quantization numbers:
 * the envelope is indexed by neither, which is what lets it answer for a
 * convolution whose height and width are still dynamic.
 *
 * `precision` is the precision the *lowering programs*, not one implied by
 * the operand types: a requantized int8 convolution and an accumulator one
 * are both `i8 x i8 -> i32` in the IR and have separate envelopes. */
typedef struct rocket_plan_admission_desc_t {
  uint32_t struct_size;
  uint32_t precision;  /* rocket_plan_precision_e */
  uint64_t kernel_height;
  uint64_t kernel_width;
  uint64_t stride;      /* same on both axes */
  uint64_t in_channels;
  uint64_t out_channels;
  uint8_t depthwise;
  uint8_t reserved_[7];
} rocket_plan_admission_desc_t;

typedef struct rocket_plan_matmul_admission_desc_t {
  uint32_t struct_size;
  uint32_t precision;  /* rocket_plan_precision_e */
  uint64_t m;
  uint64_t k;
  uint64_t n;
} rocket_plan_matmul_admission_desc_t;

/* Mirrors rocket_core::policy::PlanningPolicy. NULL means "no overrides",
 * which is also what the runtime plans under unless its environment says
 * otherwise; a compiler that passes NULL therefore agrees with a runtime
 * whose environment is clean. */
typedef struct rocket_plan_policy_t {
  uint32_t struct_size;
  uint8_t allow_unbacked_channels;
  uint8_t allow_large_kernel_probing;
  uint8_t reserved_[2];
} rocket_plan_policy_t;

/* What the planner decided, for a call that returned ROCKET_PLAN_OK. */
typedef struct rocket_plan_conv_plan_t {
  uint32_t struct_size;
  uint32_t data_banks;
  uint32_t weight_banks;
  /* Standalone hardware jobs the dispatch runs; 1 is a direct plan. */
  uint32_t tile_count;
  /* Column tiles across the output row; 1 means no horizontal split. */
  uint32_t column_count;
  uint32_t output_width;
  uint32_t output_height;
  uint32_t reserved_;
  uint64_t weight_bytes;
  uint64_t output_scratch_bytes;
} rocket_plan_conv_plan_t;

/* One feature map as `kind` reads or writes it (COMPILER_ROADMAP.md 6.1).
 * `element_bytes` is the element width of *this side* -- a convolution's
 * input is its input element, its output the output element, so an
 * fp32-result rung is 2 here on the way in and 4 on the way out. A matmul
 * operand is width m, height 1. Sub-byte elements have no cube; ask with
 * 1, 2 or 4 only. */
typedef struct rocket_plan_cube_desc_t {
  uint32_t struct_size;
  uint32_t kind;          /* rocket_plan_cube_kind_e */
  uint32_t element_bytes; /* 1, 2 or 4 */
  uint32_t reserved_;
  uint64_t width;
  uint64_t height;
  uint64_t channels;
} rocket_plan_cube_desc_t;

/* What rocket_plan_cube_geometry computed: the four numbers a producer and
 * a consumer must agree on, the packed storage, and the two predicates the
 * chain identity is built from. */
typedef struct rocket_plan_cube_geometry_t {
  uint32_t struct_size;
  /* The logical pixel is a whole number of 16-byte atoms. */
  uint8_t whole_atom;
  /* whole_atom, and the reader pads no channels past the logical ones:
   * the consumer-side precondition of the chain identity. */
  uint8_t exact;
  uint8_t reserved_[2];
  uint64_t pixel_count;
  /* Pixels one surface is strided by: pixel_count rounded up to four for
   * POOLING, equal to it for every other kind. */
  uint64_t surface_pixel_count;
  /* channels * element_bytes. */
  uint64_t bytes_per_pixel;
  /* The width the reader pads a pixel to: 16 lanes for the CNA-fed kinds
   * whatever the element width, one atom for POOLING. */
  uint64_t packed_bytes_per_pixel;
  /* Bytes the packed cube occupies. */
  uint64_t storage_bytes;
} rocket_plan_cube_geometry_t;

/* The ABI version this library was built for. */
uint32_t rocket_plan_abi_version(void);

/* A fixed name for a status, e.g. "unvalidated_configuration". */
const char* rocket_plan_status_name(uint32_t status);

/* A fixed name for a precision rung, in the spelling the transform spec's
 * `precision` attribute uses: "fp16", "int8_requant", "int8_accumulator",
 * ... Out-of-range values name "unknown" rather than reading past the
 * table. Added in ABI version 3. */
const char* rocket_plan_precision_name(uint32_t precision);

/* Plans `desc` under `policy` (NULL for none). On ROCKET_PLAN_OK fills
 * `out_plan` (may be NULL if the caller only wants the verdict). On any
 * other status writes the refusal message into `message` (may be NULL or
 * `message_capacity` 0 to skip). Never panics across the boundary. */
uint32_t rocket_plan_conv(const rocket_plan_conv_desc_t* desc,
                          const rocket_plan_policy_t* policy,
                          rocket_plan_conv_plan_t* out_plan, char* message,
                          size_t message_capacity);

uint32_t rocket_plan_matmul(const rocket_plan_matmul_desc_t* desc,
                            const rocket_plan_policy_t* policy,
                            rocket_plan_conv_plan_t* out_plan, char* message,
                            size_t message_capacity);

/* Whether the compiler has evidence for this shape class: the ceilings that
 * used to be `transform.iree.match.dim_bounds` lines in the transform spec.
 * ROCKET_PLAN_OK to claim it, otherwise a status (usually
 * ROCKET_PLAN_UNVALIDATED_CONFIGURATION) and a message naming the actual
 * value and the ceiling. Never panics across the boundary.
 *
 * This is a *different question* from rocket_plan_conv, and both have to be
 * asked: admission says the class has been characterized, planning says the
 * concrete extents can be programmed. */
uint32_t rocket_admit_conv(const rocket_plan_admission_desc_t* desc,
                           char* message, size_t message_capacity);

uint32_t rocket_admit_matmul(const rocket_plan_matmul_admission_desc_t* desc,
                             char* message, size_t message_capacity);

/* The geometry of one feature cube, the same function the runtime builds
 * its `OutputCube` from. On ROCKET_PLAN_OK fills `out_geometry` (may be
 * NULL for the verdict alone). Never panics across the boundary. Added in
 * ABI version 4. */
uint32_t rocket_plan_cube_geometry(const rocket_plan_cube_desc_t* desc,
                                   rocket_plan_cube_geometry_t* out_geometry,
                                   char* message, size_t message_capacity);

/* Whether `consumer` may read `producer`'s cube in place of repacking it:
 * pack(compact(cube)) == cube under the consumer's geometry, the identity
 * the runtime's cross-dispatch chaining rests on. ROCKET_PLAN_OK when it
 * holds, ROCKET_PLAN_LAYOUT_MISMATCH with the failing condition in
 * `message` when it does not, and a malformed descriptor's own status
 * otherwise. Says nothing about whether the producer offers a cube at all
 * (a fanned-out or accumulator dispatch does not) or whether the consumer
 * reads one (a Cin <= 4 convolution reads dense ARGB); those stay the
 * runtime's, see rocket_core::layout. Added in ABI version 4. */
uint32_t rocket_plan_chain_identity(const rocket_plan_cube_desc_t* producer,
                                    const rocket_plan_cube_desc_t* consumer,
                                    char* message, size_t message_capacity);

/* Packs a convolution's logical HWCF filter into the CNA's blocked
 * coefficient stream -- the bytes the runtime would otherwise produce at
 * dispatch time, through the same rocket_core::weights::WeightPlan, so a
 * compile-time packed filter is byte-identical to a runtime packed one
 * (COMPILER_ROADMAP.md 6.3).
 *
 * `desc` needs the shape and precision; activation and scales do not
 * affect the bytes (the int8 rungs' weights_zero_point does, and is read).
 * `dense` holds exactly kh * kw * Cin * Cout elements (kh * kw * Cin for
 * depthwise) at the rung's element width, in HWCF order as IREE's conv ABI
 * hands them over. On ROCKET_PLAN_OK the packed length is written to
 * `out_packed_length` (may be NULL) and, when `packed` is non-NULL, the
 * bytes to `packed`, which must hold at least that many. Pass `packed` as
 * NULL to query the size without packing; `dense` is then not read. A
 * length mismatch or a short buffer is ROCKET_PLAN_INVALID_ARGUMENT with
 * nothing written. Never panics across the boundary. Added in ABI version
 * 5. */
uint32_t rocket_pack_conv_weights(const rocket_plan_conv_desc_t* desc,
                                  const uint8_t* dense, size_t dense_length,
                                  uint8_t* packed, size_t packed_capacity,
                                  size_t* out_packed_length, char* message,
                                  size_t message_capacity);

/* The [K, N] operand of a matmul, packed for the height-one 1x1 lowering
 * the runtime plans it as: symmetric on every rung, K input channels, N
 * output channels. Same contract as rocket_pack_conv_weights. Added in ABI
 * version 5. */
uint32_t rocket_pack_matmul_weights(const rocket_plan_matmul_desc_t* desc,
                                    const uint8_t* dense, size_t dense_length,
                                    uint8_t* packed, size_t packed_capacity,
                                    size_t* out_packed_length, char* message,
                                    size_t message_capacity);

#ifdef __cplusplus
}
#endif

#endif /* ROCKET_PLAN_H_ */
