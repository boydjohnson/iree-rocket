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

#define ROCKET_PLAN_ABI_VERSION 2u

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

/* Mirrors rocket_core::error::PlanErrorCode, plus the two the boundary
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
} rocket_plan_status_e;

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

/* The ABI version this library was built for. */
uint32_t rocket_plan_abi_version(void);

/* A fixed name for a status, e.g. "unvalidated_configuration". */
const char* rocket_plan_status_name(uint32_t status);

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

#ifdef __cplusplus
}
#endif

#endif /* ROCKET_PLAN_H_ */
