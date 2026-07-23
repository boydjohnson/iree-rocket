// Copyright 2026
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Hand-authored Transform Dialect script referenced by RocketTarget.cpp's top
// doc comment (modeled on
// samples/custom_dispatch/cpu/embedded/example_transform_spec.mlir in the
// IREE tree). Matches `linalg.conv_2d_nhwc_hwcf` (this exact static shape --
// v1 is deliberately not shape-generic, see RocketTarget.cpp) and splices in
// a `flow.dispatch` to a hand-authored `hal.executable` already targeting
// "rocket", with the matched op's static shape/dtype facts stamped directly
// onto that executable's `#hal.executable.target` config dict.
//
// Usage: `--iree-preprocessing-transform-spec-filename=<this file>`.
//
// MULTI-DEVICE USAGE (the load-bearing part of this file beyond the basic
// splice): if the containing program has ANY other ops besides the matched
// conv2d (a real model, not a conv-only toy graph), do NOT compile with
// `--iree-hal-target-backends=rocket,llvm-cpu` (or any two-backend list on
// one implicit device) -- IREE's plain multi-backend flag means "compile
// every dispatch for BOTH listed backends" (portability/fat-binary
// semantics), not "route each dispatch to whichever backend can handle it".
// Concretely confirmed on a real MobileNetV2 graph: RocketTargetBackend's
// translation pipeline is intentionally empty (it only knows how to
// serialize the specific pre-spliced `rocket_executable` this file builds),
// so IREE tries to compile every OTHER dispatch (e.g. the classifier's
// `linalg.matmul`) for "rocket" too, and `serializeExecutable` has nothing to
// emit -- a hard compile error, before conv-matching is even reached.
//
// The real fix, proven to compile clean end-to-end (rocket conv2d +
// unrelated llvm-cpu matmul in one module, both correctly and
// independently resolved -- see the `stream.affinity` attribute on the
// `flow.dispatch` below): declare TWO NAMED devices and give the
// rocket-targeted dispatch an explicit affinity to the rocket one, letting
// everything else default to the other:
//
//   iree-compile \
//     --iree-hal-target-device=rocket_device=rocket \
//     --iree-hal-target-device=cpu_device=local \
//     --iree-hal-local-target-device-backends=llvm-cpu \
//     --iree-hal-default-device=cpu_device \
//     --iree-hal-indirect-command-buffers=false \
//     --iree-preprocessing-transform-spec-filename=rocket_conv2d_transform_spec.mlir \
//     model.mlir -o model.vmfb
//
// `--iree-hal-indirect-command-buffers=false` is required (not new to this
// file, but easy to forget when copying this recipe): rocket-hal-driver
// does not implement indirect-binding support (deferred regcmd construction
// at execute time -- see project memory). Confirmed via `iree-dump-module`
// that this flag switches the emitted VM import from
// `hal.device.queue.execute.indirect` to plain `hal.device.queue.execute`
// (no `.indirect` import remains anywhere in the compiled module). Note this
// does NOT by itself fix the `MAPPING_PERSISTENT` `PERMISSION_DENIED` a
// mixed rocket+llvm-cpu module hits at runtime -- that recurs on the plain
// (direct) `hal.device.queue.execute` path too and needed a separate real
// fix in `rocket-hal-driver`'s own allocator (see below).
//
// This is real, non-experimental, wired-up IREE machinery -- not a hack:
// `DeviceAnalysis::gatherRequiredExecutableTargets`
// (compiler/.../HAL/Analysis/DeviceAnalysis.cpp) resolves each dispatch's
// affinity to its device's own `#hal.executable.target` list and requires
// ONLY those targets for that dispatch's executable; ops with no affinity
// fall back to `stream.affinity.default` (set by `--iree-hal-default-device`)
// and never require the rocket backend at all.
//
// IMPORTANT gotcha found empirically and worth preserving here: use
// `#hal.device.affinity<@rocket_device>`, NOT `#hal.device.promise<@rocket_device>`,
// even though HALAttrs.td's own doc comment for `#hal.device.promise`
// describes it as the mechanism for "input programs" to reference a device
// "prior to [it] being declared" (exactly this preprocessing-stage use
// case, on paper). In practice `#hal.device.promise`'s `$device` parameter
// is a plain `StringAttr`, not a real `SymbolRefAttr` -- so generic MLIR
// symbol-use analysis does NOT see it as a live reference to the
// CLI-materialized `util.global private @rocket_device`, and that global
// gets silently DCE'd as dead before `ResolveDevicePromisesPass` ever runs,
// producing `error: ... references a promised device that was not
// declared`. `#hal.device.affinity`'s `$device` IS a real `SymbolRefAttr`,
// which generic symbol-use tracking DOES see (keeping the global alive
// regardless of DCE passes running in between), and nothing in the real
// HAL pass pipeline requires the referenced device to already exist at the
// point the attribute is written -- only when `DeviceAnalysis` actually
// resolves it, which is well after device declaration. Confirmed via
// `iree-dump-module` on the resulting .vmfb: both `@cpu_device`
// (`llvm-cpu`/`embedded-elf-x86_64`) and `@rocket_device`
// (`rocket`/`rocket-conv2d-v1`) appear as independently-queried HAL devices,
// and the unrelated matmul's executable only ever lists
// `embedded-elf-x86_64` as an available format.
//
// TWO REAL RUNTIME BUGS FOUND AND FIXED IN `rocket-hal-driver` GETTING THIS
// EXACT rocket+llvm-cpu MULTI-DEVICE MODULE WORKING END TO END ON REAL
// HARDWARE (both hardware-confirmed; correct conv=6.0 and matmul=4.0
// simultaneously, from one compiled module, on the actual RK3588 board):
//
// 1. `RocketAllocator`'s `allocate_buffer`/`query_buffer_compatibility`
//    passed the caller's requested `usage` bits straight through, unlike
//    IREE's generic heap allocator (`allocator_heap.c`, used by
//    `local-sync`/`local-task`) which always opportunistically grants
//    `MAPPING_SCOPED|MAPPING_PERSISTENT|MAPPING_ACCESS_RANDOM`. Since
//    `iree-run-module`'s tooling allocates every `--input=` buffer via ONE
//    lead device's allocator (`rocket`, ordinal 0 here) regardless of which
//    device actually consumes it, buffers destined for the cpu_device's
//    matmul lacked the mapping bits its inline CPU dispatch execution
//    genuinely needs, and `hal.device.queue.execute` failed with
//    `PERMISSION_DENIED`: "requested usage was not specified when the
//    buffer was allocated ... operation requires MAPPING_PERSISTENT".
//    Fixed by having `RocketAllocator` unconditionally grant those same
//    mapping bits too -- truthful given every rocket buffer genuinely is
//    permanently host-mmap'd, unified-memory hardware (see `buffer.rs`'s own
//    doc comment) -- rather than only when a caller happened to ask.
//
// 2. Separately (and this is the one that produced silently WRONG NUMBERS,
//    not a crash): `RocketBuffer::unmap_range` was a total no-op (matching
//    the driver's "stays mmap'd forever" design), correct for callers that
//    explicitly flush themselves (`iree_hal_buffer_map_write`, used by every
//    hand-driven CTS test, checks `HOST_COHERENT` and calls flush when
//    absent) -- but IREE's OWN `iree_hal_buffer_view_generate_buffer_in_situ`
//    (`buffer_view_util.c`, the exact code path `iree-run-module
//    --input=<shape>=<fill>` splat values go through) does NOT flush
//    explicitly; it relies entirely on unmap to make the write visible,
//    which a true no-op unmap never provides. Concretely: input/weight
//    buffers populated this way read back correctly on the CPU (same-address
//    same-core load-after-store is always coherent, mapping or not) but the
//    DPU read stale/pre-existing physical memory instead of the real values
//    -- a deterministic, INPUT-INDEPENDENT garbage result (confirmed by
//    feeding two different input values and getting byte-identical hardware
//    output both times). Fixed by having `unmap_range` flush (`FINI_BO`)
//    whenever the mapping allowed write access, matching what a real
//    hardware driver's unmap would naturally provide.
//
// Diagnostic method that found both: temporary `eprintln!` instrumentation
// in `command_buffer.rs`'s `dispatch()` and `device.rs`'s `queue_execute()`
// printing real buffer addresses/handles/host-side bytes at record time, and
// the full scratch/output bytes before and after compaction -- same
// technique (and same files) as the two GEM-handle bugs found earlier this
// project. Removed once both fixes were confirmed on hardware.

#rocket_target = #hal.executable.target<"rocket", "rocket-conv2d-v1", {
  input_width = 4 : i32, input_height = 4 : i32, input_channels = 1 : i32,
  output_width = 4 : i32, output_height = 4 : i32, output_channels = 1 : i32,
  weights_width = 1 : i32, weights_height = 1 : i32, stride = 1 : i32,
  depthwise = false,
  input_zero_point = 0 : i32, output_zero_point = 0 : i32, weights_zero_point = 0 : i32,
  input_scale = 1.0 : f32, weights_scale = 1.0 : f32, output_scale = 1.0 : f32,
  truncate_bits = 0 : i32,
  activation = "none", activation_cmp = 0 : i32,
  precision = "fp16"
}>

#pipeline_layout = #hal.pipeline.layout<constants = 0, bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>

module attributes {transform.with_named_sequence} {

  hal.executable private @rocket_executable {
    hal.executable.variant public @rocket_conv2d_v1 target(#rocket_target) {
      hal.executable.export public @rocket_conv2d ordinal(0) layout(#pipeline_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @rocket_conv2d() {
          return
        }
      }
    }
  }

  // 0=input, 1=weights, 2=bias, 3=output.
  //
  // `stream.affinity` here is the multi-device fix documented at the top of
  // this file -- pins this specific dispatch to `@rocket_device` so it
  // never gets bundled into a generic multi-backend requirement set with
  // whatever the rest of the program targets. `@rocket_device` itself does
  // not need to exist yet when this file is applied (preprocessing runs
  // before `--iree-hal-target-device` is materialized into real
  // `util.global`s) -- only by the time `DeviceAnalysis`/
  // `MaterializeInterfaces` actually resolve it.
  util.func private @call_rocket_conv2d(%input: tensor<1x4x4x1xf16>, %filter: tensor<1x1x1x1xf16>) -> tensor<1x4x4x1xf16> {
    %cst = arith.constant 0.000000e+00 : f16
    %bias_empty = tensor.empty() : tensor<1x1x1x1xf16>
    %bias = linalg.fill ins(%cst : f16) outs(%bias_empty : tensor<1x1x1x1xf16>) -> tensor<1x1x1x1xf16>
    %0 = flow.dispatch @rocket_executable::@rocket_conv2d_v1::@rocket_conv2d(%input, %filter, %bias)
        {stream.affinity = #hal.device.affinity<@rocket_device>}
        : (tensor<1x4x4x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>) -> tensor<1x4x4x1xf16>
    util.return %0 : tensor<1x4x4x1xf16>
  }

  transform.named_sequence @match_conv2d(%root: !transform.any_op {transform.readonly}) -> (!transform.any_value, !transform.any_value) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
      ^bb0(%input: tensor<1x4x4x1xf16>, %filter: tensor<1x1x1x1xf16>):
        %cst = arith.constant 0.000000e+00 : f16
        %empty = tensor.empty() {"match.operation_name_only"} : tensor<1x4x4x1xf16>
        %filled = linalg.fill ins(%cst : f16) outs(%empty : tensor<1x4x4x1xf16>) -> tensor<1x4x4x1xf16>
        %result = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
            ins(%input, %filter : tensor<1x4x4x1xf16>, tensor<1x1x1x1xf16>)
            outs(%filled : tensor<1x4x4x1xf16>) -> tensor<1x4x4x1xf16>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    transform.yield %ins, %outs : !transform.any_value, !transform.any_value
  }

  transform.named_sequence @cast_and_call_conv2d(%ins: !transform.any_value {transform.readonly},
                                                  %out: !transform.any_value {transform.readonly}) {
    %root = transform.get_defining_op %out : (!transform.any_value) -> !transform.any_op
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %executable = transform.util.import_symbol @rocket_executable into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_rocket_conv2d into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
          transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %funcs = transform.structured.match ops{["util.func"]} in %module : (!transform.any_op) -> !transform.any_op
    transform.foreach %funcs : !transform.any_op {
      ^bb1(%func: !transform.any_op):
        transform.foreach_match in %func
            @match_conv2d -> @cast_and_call_conv2d
          : (!transform.any_op) -> (!transform.any_op)
    }
    transform.apply_dce to %module : !transform.any_op
    transform.yield
  }
}
