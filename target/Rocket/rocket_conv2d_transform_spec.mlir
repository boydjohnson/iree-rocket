// v3 separates generic convolution recognition from the static executable
// specializations required by the current Rocket backend. The matchers below
// use ConvolutionOpInterface to inspect element types, dimensions, stride, and
// dilation without spelling complete tensor types in a DAG.
//
// RocketTarget.cpp still requires literal shape fields in every
// #hal.executable.target config. Consequently, each executable keeps an exact
// dimension guard. An unsupported shape fails these matchers and remains on
// the default CPU device instead of being sent to Rocket with a wrong config.

#rocket_target_0 = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "conv2d",
  input_width = 112 : i32, input_height = 112 : i32, input_channels = 32 : i32,
  output_width = 112 : i32, output_height = 112 : i32, output_channels = 16 : i32,
  weights_width = 1 : i32, weights_height = 1 : i32, stride = 1 : i32,
  depthwise = false,
  input_zero_point = 0 : i32, output_zero_point = 0 : i32, weights_zero_point = 0 : i32,
  input_scale = 1.0 : f32, weights_scale = 1.0 : f32, output_scale = 1.0 : f32,
  truncate_bits = 0 : i32,
  activation = "none", activation_cmp = 0 : i32,
  precision = "fp16"
}>

#rocket_target_1 = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "conv2d",
  input_width = 56 : i32, input_height = 56 : i32, input_channels = 96 : i32,
  output_width = 56 : i32, output_height = 56 : i32, output_channels = 24 : i32,
  weights_width = 1 : i32, weights_height = 1 : i32, stride = 1 : i32,
  depthwise = false,
  input_zero_point = 0 : i32, output_zero_point = 0 : i32, weights_zero_point = 0 : i32,
  input_scale = 1.0 : f32, weights_scale = 1.0 : f32, output_scale = 1.0 : f32,
  truncate_bits = 0 : i32,
  activation = "none", activation_cmp = 0 : i32,
  precision = "fp16"
}>

#rocket_target_2 = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "conv2d",
  input_width = 56 : i32, input_height = 56 : i32, input_channels = 24 : i32,
  output_width = 56 : i32, output_height = 56 : i32, output_channels = 144 : i32,
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

// IMPORTANT: unlike the CollapseDimensionsPass fix above, `stream.topology`
// must live on the TARGET module being compiled (standalone_repro.mlir /
// mnv2_fp16_clean.mlir), not on THIS transform-spec file's own module --
// they are two separate MLIR modules (the transform interpreter loads this
// file as a library and applies it to the real target module; top-level
// attributes on this file never get copied over). It is injected below via
// `transform.annotate` + `transform.param.constant` inside
// @cast_and_call_conv2d_0, targeting `%module`
// (transform.util.get_nearest_symbol_table's result, i.e. the real target
// module), each time our matcher fires. `transform.annotate` just does a
// plain `setAttr`, so repeated firings (multiple matched convs) simply
// overwrite the same value idempotently -- harmless.
//
// SECOND real bug found (after the CollapseDimensionsPass attr-drop above):
// once the rocket dispatch's f16 output is consumed DIRECTLY by another
// dispatch on a DIFFERENT device within the same computation (a genuine
// cross-device data dependency, not just "two independent unrelated
// dispatches on different devices" like the doc-comment example in
// rocket_conv2d_transform_spec.mlir), Stream correctly forms a
// `!stream.resource<transient>` allocated `on(#hal.device.optimal<[cpu,
// rocket]>)` (see --compile-to=stream output) with explicit
// `stream.cmd.flush` sync between the two device partitions -- this is
// real, intentional IREE multi-device machinery, not a bug on its own.
// But lowering it through HAL requires
// `ResolveTopologyQueriesPass::resolveMemoryPropertiesOp`
// (Dialect/HAL/Transforms/ResolveTopologyQueries.cpp) to prove the
// cross-device buffer can actually be shared -- it does this via
// `tryAddSharedUsageBits`, which needs the module's `stream.topology`
// attribute (`#hal.device.topology<links = [...]>`) to declare
// `transparent_access`/`unified_memory` between rocket_device and
// cpu_device. With NO topology attribute at all (the default, and what
// design_b_fix_dispatch_region.mlir implicitly had), this silently fails
// (the pass ignores resolveMemoryPropertiesOp's LogicalResult) and leaves
// a `hal.allocator.resolve_memory_properties` op with no lowering, which
// then hard-fails much later at VM conversion with "failed to legalize
// operation 'hal.allocator.resolve_memory_properties'" -- a confusing
// error far from its real cause. Declaring the topology below (truthfully
// -- RK3588's rocket NPU really is unified, host-mmap'd memory, see
// project memory's RocketAllocator/RocketBuffer notes) fixes it cleanly.
module attributes {transform.with_named_sequence} {

  hal.executable private @rocket_executable_0 {
    hal.executable.variant public @rocket_conv2d_v1_0 target(#rocket_target_0) {
      hal.executable.export public @rocket_conv2d_0 ordinal(0) layout(#pipeline_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @rocket_conv2d_0() {
          return
        }
      }
    }
  }

  // Only the bare conv op is matched/replaced. filter-transpose, bias
  // extf/broadcast, and everything downstream (expand_shape, truncf,
  // relu6 clamp) stay exactly as they were in the real graph, untouched --
  // this dispatch does a ZERO-bias raw conv on the NPU, upcasts to f32,
  // and adds the real (already-broadcast) bias back in on CPU, matching
  // the bare conv op's exact original semantics (outs + conv(ins)).
  util.func private @call_rocket_conv2d_0(%input: tensor<1x112x112x32xf16>, %filter: tensor<1x1x32x16xf16>, %bias_bcast: tensor<1x112x112x16xf32>) -> tensor<1x112x112x16xf32> {
    %zero_bias_empty = tensor.empty() : tensor<16xf16>
    %zero_f16 = arith.constant 0.0 : f16
    %zero_bias = linalg.fill ins(%zero_f16 : f16) outs(%zero_bias_empty : tensor<16xf16>) -> tensor<16xf16>
    %raw_f16 = flow.dispatch @rocket_executable_0::@rocket_conv2d_v1_0::@rocket_conv2d_0(%input, %filter, %zero_bias)
        {stream.affinity = #hal.device.affinity<@rocket_device>}
        : (tensor<1x112x112x32xf16>, tensor<1x1x32x16xf16>, tensor<16xf16>) -> tensor<1x112x112x16xf16>
    // ROOT CAUSE (confirmed by reading IREE source + bisecting IR dumps
    // stage-by-stage with --mlir-print-ir-after-all, not guessed):
    //
    // A hand-authored `flow.dispatch.region ... attributes {stream.affinity
    // = ...}` (the design_b_fix_dispatch_region.mlir attempt) is NOT
    // sufficient. It survives Preprocessing and GlobalOptimization intact
    // (confirmed via --compile-to=global-optimization), but
    // DispatchCreation's `CollapseDimensionsPass` unconditionally destroys
    // it. That pass's `hoistTensorReshapesOutOfDispatchRegion()`
    // (DispatchCreation/CollapseDimensions.cpp) is called via
    // `funcOp->walk<DispatchRegionOp>` on EVERY `flow.dispatch.region` in
    // the function -- unconditionally, regardless of whether any reshape
    // ops are actually present to hoist -- and unconditionally rebuilds the
    // op via a bare `IREE::Flow::DispatchRegionOp::create(rewriter, loc,
    // newReturnTypes, newDynamicDims, dispatchOp.getWorkload())` at line
    // 913, which does NOT call `setDialectAttrs()` (unlike
    // ConvertRegionToWorkgroups.cpp:167, which does). So our
    // `stream.affinity` attribute is silently dropped before
    // ConvertDispatchRegionsToWorkgroupsPass even runs. Confirmed by
    // diffing --compile-to=dispatch-creation output before/after this exact
    // pass: attribute present right before CollapseDimensionsPass, gone
    // right after, on the exact same op. This is a genuine IREE upstream
    // bug (attribute-preservation oversight in CollapseDimensionsPass), not
    // a conceptual misunderstanding about affinity/util.call boundaries.
    //
    // ALSO confirmed dead-end: setting `stream.affinity` directly as a
    // linalg.generic's own `attrs = {...}` clause (the very first thing
    // tried, before any dispatch.region) has no effect for a different
    // reason -- `wrapOpInDispatchRegion`/`makeEmptyDispatchRegion`
    // (Flow/Transforms/RegionOpUtils.cpp), which
    // CloneProducersIntoDispatchRegionsPass uses to auto-wrap any bare
    // linalg.generic left outside a dispatch, creates a brand new empty
    // region and never consults the wrapped op's own attributes at all.
    //
    // WORKAROUND (this file): skip `flow.dispatch.region` entirely and
    // hand-author the ALREADY-OUTLINED `flow.dispatch.workgroups` form
    // directly (with real iree_tensor_ext.dispatch.tensor.load/store
    // boilerplate, mechanically copied from what
    // ConvertDispatchRegionsToWorkgroupsPass itself produces for an
    // equivalent region -- see the dispatch-creation IR dump captured
    // while root-causing this). CollapseDimensionsPass's walk only matches
    // `IREE::Flow::DispatchRegionOp`; a `flow.dispatch.workgroups` op is a
    // different op type (already "isolated from above", already
    // dispatch-shaped) and is never visited by it, so the attribute we set
    // here is never at risk of being dropped by that pass. It is also never
    // touched by FormDispatchRegions/ElementwiseOpFusion/
    // CloneProducersIntoDispatchRegions (those only cluster ops NOT already
    // inside a dispatch). It only gets touched once more, by
    // OutlineDispatchRegions at the `flow` stage, which DOES correctly
    // preserve dialect attrs (confirmed via source read,
    // OutlineDispatchRegions.cpp:81,144) -- and empirically confirmed here:
    // `stream.affinity` is present on the resulting flow.dispatch call
    // through dispatch-creation, flow, and stream stages, and the module
    // compiles clean end-to-end (see design_b_fix_v2_workgroups compile log
    // in project memory / agent report).
    %final = flow.dispatch.workgroups(%raw_f16, %bias_bcast) : (tensor<1x112x112x16xf16>, tensor<1x112x112x16xf32>) -> tensor<1x112x112x16xf32>
        attributes { stream.affinity = #hal.device.affinity<@cpu_device> } =
        (%arg3: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x112x112x16xf16>>, %arg4: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x112x112x16xf32>>, %arg5: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x112x112x16xf32>>) {
      %raw_f16_ld = iree_tensor_ext.dispatch.tensor.load %arg3, offsets = [0, 0, 0, 0], sizes = [1, 112, 112, 16], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x112x112x16xf16>> -> tensor<1x112x112x16xf16>
      %bias_ld = iree_tensor_ext.dispatch.tensor.load %arg4, offsets = [0, 0, 0, 0], sizes = [1, 112, 112, 16], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x112x112x16xf32>> -> tensor<1x112x112x16xf32>
      %final_empty = tensor.empty() : tensor<1x112x112x16xf32>
      %final_inner = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%raw_f16_ld, %bias_ld : tensor<1x112x112x16xf16>, tensor<1x112x112x16xf32>) outs(%final_empty : tensor<1x112x112x16xf32>) {
      ^bb0(%a: f16, %b: f32, %out: f32):
        %ae = arith.extf %a : f16 to f32
        %s = arith.addf %ae, %b : f32
        linalg.yield %s : f32
      } -> tensor<1x112x112x16xf32>
      iree_tensor_ext.dispatch.tensor.store %final_inner, %arg5, offsets = [0, 0, 0, 0], sizes = [1, 112, 112, 16], strides = [1, 1, 1, 1] : tensor<1x112x112x16xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x112x112x16xf32>>
      flow.return
    } count() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    util.return %final : tensor<1x112x112x16xf32>
  }

  hal.executable private @rocket_executable_1 {
    hal.executable.variant public @rocket_conv2d_v1_1 target(#rocket_target_1) {
      hal.executable.export public @rocket_conv2d_1 ordinal(0) layout(#pipeline_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @rocket_conv2d_1() {
          return
        }
      }
    }
  }

  hal.executable private @rocket_executable_2 {
    hal.executable.variant public @rocket_conv2d_v1_2 target(#rocket_target_2) {
      hal.executable.export public @rocket_conv2d_2 ordinal(0) layout(#pipeline_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @rocket_conv2d_2() {
          return
        }
      }
    }
  }

  // Shape 1: 1x56x56x96 -> 1x56x56x24 (filter 24x96x1x1). Mechanically
  // identical structure to call_rocket_conv2d_0, just a different static
  // shape -- proves the fix is shape-generic, not a one-off.
  util.func private @call_rocket_conv2d_1(%input: tensor<1x56x56x96xf16>, %filter: tensor<1x1x96x24xf16>, %bias_bcast: tensor<1x56x56x24xf32>) -> tensor<1x56x56x24xf32> {
    %zero_bias_empty = tensor.empty() : tensor<24xf16>
    %zero_f16 = arith.constant 0.0 : f16
    %zero_bias = linalg.fill ins(%zero_f16 : f16) outs(%zero_bias_empty : tensor<24xf16>) -> tensor<24xf16>
    %raw_f16 = flow.dispatch @rocket_executable_1::@rocket_conv2d_v1_1::@rocket_conv2d_1(%input, %filter, %zero_bias)
        {stream.affinity = #hal.device.affinity<@rocket_device>}
        : (tensor<1x56x56x96xf16>, tensor<1x1x96x24xf16>, tensor<24xf16>) -> tensor<1x56x56x24xf16>
    %final = flow.dispatch.workgroups(%raw_f16, %bias_bcast) : (tensor<1x56x56x24xf16>, tensor<1x56x56x24xf32>) -> tensor<1x56x56x24xf32>
        attributes { stream.affinity = #hal.device.affinity<@cpu_device> } =
        (%arg3: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x56x56x24xf16>>, %arg4: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x56x56x24xf32>>, %arg5: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x56x56x24xf32>>) {
      %raw_f16_ld = iree_tensor_ext.dispatch.tensor.load %arg3, offsets = [0, 0, 0, 0], sizes = [1, 56, 56, 24], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x56x56x24xf16>> -> tensor<1x56x56x24xf16>
      %bias_ld = iree_tensor_ext.dispatch.tensor.load %arg4, offsets = [0, 0, 0, 0], sizes = [1, 56, 56, 24], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x56x56x24xf32>> -> tensor<1x56x56x24xf32>
      %final_empty = tensor.empty() : tensor<1x56x56x24xf32>
      %final_inner = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%raw_f16_ld, %bias_ld : tensor<1x56x56x24xf16>, tensor<1x56x56x24xf32>) outs(%final_empty : tensor<1x56x56x24xf32>) {
      ^bb0(%a: f16, %b: f32, %out: f32):
        %ae = arith.extf %a : f16 to f32
        %s = arith.addf %ae, %b : f32
        linalg.yield %s : f32
      } -> tensor<1x56x56x24xf32>
      iree_tensor_ext.dispatch.tensor.store %final_inner, %arg5, offsets = [0, 0, 0, 0], sizes = [1, 56, 56, 24], strides = [1, 1, 1, 1] : tensor<1x56x56x24xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x56x56x24xf32>>
      flow.return
    } count() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    util.return %final : tensor<1x56x56x24xf32>
  }

  // Shape 2: 1x56x56x24 -> 1x56x56x144 (filter 144x24x1x1).
  util.func private @call_rocket_conv2d_2(%input: tensor<1x56x56x24xf16>, %filter: tensor<1x1x24x144xf16>, %bias_bcast: tensor<1x56x56x144xf32>) -> tensor<1x56x56x144xf32> {
    %zero_bias_empty = tensor.empty() : tensor<144xf16>
    %zero_f16 = arith.constant 0.0 : f16
    %zero_bias = linalg.fill ins(%zero_f16 : f16) outs(%zero_bias_empty : tensor<144xf16>) -> tensor<144xf16>
    %raw_f16 = flow.dispatch @rocket_executable_2::@rocket_conv2d_v1_2::@rocket_conv2d_2(%input, %filter, %zero_bias)
        {stream.affinity = #hal.device.affinity<@rocket_device>}
        : (tensor<1x56x56x24xf16>, tensor<1x1x24x144xf16>, tensor<144xf16>) -> tensor<1x56x56x144xf16>
    %final = flow.dispatch.workgroups(%raw_f16, %bias_bcast) : (tensor<1x56x56x144xf16>, tensor<1x56x56x144xf32>) -> tensor<1x56x56x144xf32>
        attributes { stream.affinity = #hal.device.affinity<@cpu_device> } =
        (%arg3: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x56x56x144xf16>>, %arg4: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x56x56x144xf32>>, %arg5: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x56x56x144xf32>>) {
      %raw_f16_ld = iree_tensor_ext.dispatch.tensor.load %arg3, offsets = [0, 0, 0, 0], sizes = [1, 56, 56, 144], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x56x56x144xf16>> -> tensor<1x56x56x144xf16>
      %bias_ld = iree_tensor_ext.dispatch.tensor.load %arg4, offsets = [0, 0, 0, 0], sizes = [1, 56, 56, 144], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x56x56x144xf32>> -> tensor<1x56x56x144xf32>
      %final_empty = tensor.empty() : tensor<1x56x56x144xf32>
      %final_inner = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%raw_f16_ld, %bias_ld : tensor<1x56x56x144xf16>, tensor<1x56x56x144xf32>) outs(%final_empty : tensor<1x56x56x144xf32>) {
      ^bb0(%a: f16, %b: f32, %out: f32):
        %ae = arith.extf %a : f16 to f32
        %s = arith.addf %ae, %b : f32
        linalg.yield %s : f32
      } -> tensor<1x56x56x144xf32>
      iree_tensor_ext.dispatch.tensor.store %final_inner, %arg5, offsets = [0, 0, 0, 0], sizes = [1, 56, 56, 144], strides = [1, 1, 1, 1] : tensor<1x56x56x144xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x56x56x144xf32>>
      flow.return
    } count() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    util.return %final : tensor<1x56x56x144xf32>
  }

  transform.named_sequence @match_conv2d_0(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.conv_2d_nhwc_hwcf"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = f16, rhs_type = f16, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [112, 112] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [16] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [32] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>
    transform.yield %root : !transform.any_op
  }

  transform.named_sequence @cast_and_call_conv2d_0(%root: !transform.any_op {transform.readonly}) {
    %ins = transform.get_operand %root[all] : (!transform.any_op) -> !transform.any_value
    %out = transform.get_result %root[all] : (!transform.any_op) -> !transform.any_value
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    // Declare truthfully that rocket_device and cpu_device share unified,
    // transparently-accessible memory (real RK3588 hardware fact -- see
    // project memory's RocketAllocator/RocketBuffer notes) so
    // ResolveTopologyQueriesPass can resolve the `#hal.device.optimal<[cpu,
    // rocket]>`-affinitized cross-device buffer Stream forms for values
    // flowing directly from our rocket dispatch into CPU-side compute. Must
    // be set on the REAL target module (%module), not this transform-spec
    // file's own module -- see file-level comment above. Idempotent: fires
    // once per matched conv, each time just overwriting the same value.
    %topology_attr = transform.param.constant #hal.device.topology<links = [
        (@rocket_device -> @cpu_device = {transparent_access = true, unified_memory = true}),
        (@cpu_device -> @rocket_device = {transparent_access = true, unified_memory = true})
      ]> -> !transform.any_param
    transform.annotate %module "stream.topology" = %topology_attr : !transform.any_op, !transform.any_param
    %executable = transform.util.import_symbol @rocket_executable_0 into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_rocket_conv2d_0 into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
          transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @match_conv2d_1(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.conv_2d_nhwc_hwcf"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = f16, rhs_type = f16, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [56, 56] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [24] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [96] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>
    transform.yield %root : !transform.any_op
  }

  transform.named_sequence @cast_and_call_conv2d_1(%root: !transform.any_op {transform.readonly}) {
    %ins = transform.get_operand %root[all] : (!transform.any_op) -> !transform.any_value
    %out = transform.get_result %root[all] : (!transform.any_op) -> !transform.any_value
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %topology_attr = transform.param.constant #hal.device.topology<links = [
        (@rocket_device -> @cpu_device = {transparent_access = true, unified_memory = true}),
        (@cpu_device -> @rocket_device = {transparent_access = true, unified_memory = true})
      ]> -> !transform.any_param
    transform.annotate %module "stream.topology" = %topology_attr : !transform.any_op, !transform.any_param
    %executable = transform.util.import_symbol @rocket_executable_1 into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_rocket_conv2d_1 into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
          transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @match_conv2d_2(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.conv_2d_nhwc_hwcf"] : !transform.any_op
    %batch, %out_img, %out_ch, %filter, %in_ch, %depth, %strides, %dilations =
        transform.iree.match.convolution %root,
          lhs_type = f16, rhs_type = f16, output_type = f32
          : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [56, 56] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [144] : !transform.param<i64>
    transform.iree.match.dims_equal %filter, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [24] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>
    transform.yield %root : !transform.any_op
  }

  transform.named_sequence @cast_and_call_conv2d_2(%root: !transform.any_op {transform.readonly}) {
    %ins = transform.get_operand %root[all] : (!transform.any_op) -> !transform.any_value
    %out = transform.get_result %root[all] : (!transform.any_op) -> !transform.any_value
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %topology_attr = transform.param.constant #hal.device.topology<links = [
        (@rocket_device -> @cpu_device = {transparent_access = true, unified_memory = true}),
        (@cpu_device -> @rocket_device = {transparent_access = true, unified_memory = true})
      ]> -> !transform.any_param
    transform.annotate %module "stream.topology" = %topology_attr : !transform.any_op, !transform.any_param
    %executable = transform.util.import_symbol @rocket_executable_2 into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_rocket_conv2d_2 into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
          transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %funcs = transform.structured.match ops{["util.func"]} in %module : (!transform.any_op) -> !transform.any_op
    // ONNX commonly imports Conv as NCHW/FCHW, while Rocket's logical
    // convolution ABI and the matchers above use NHWC/HWCF. Run IREE's
    // semantics-preserving layout conversion before attempting any Rocket
    // specialization. It inserts the required input/filter/init/output
    // transposes for regular convolutions. Later canonicalization can fold
    // adjacent transposes across the graph. Depthwise NCHW conversion remains
    // separate work; the current Rocket transform has no depthwise matcher.
    %channels_last_funcs = transform.apply_registered_pass
        "iree-preprocessing-convert-conv-to-channels-last" to %funcs
        : (!transform.any_op) -> !transform.any_op
    %canonical_funcs = transform.apply_registered_pass "canonicalize"
        to %channels_last_funcs : (!transform.any_op) -> !transform.any_op
    transform.foreach %canonical_funcs : !transform.any_op {
      ^bb1(%func: !transform.any_op):
        transform.foreach_match in %func
            @match_conv2d_0 -> @cast_and_call_conv2d_0,
            @match_conv2d_1 -> @cast_and_call_conv2d_1,
            @match_conv2d_2 -> @cast_and_call_conv2d_2
          : (!transform.any_op) -> (!transform.any_op)
    }
    transform.apply_dce to %module : !transform.any_op
    transform.yield
  }
}
