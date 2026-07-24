// Replaces the hardware-confirmed fp16 [4,32] x [32,16] linalg.matmul
// specialization with a Rocket fully-connected dispatch.
//
// linalg.matmul computes `outs + lhs * rhs`, while Rocket FC exposes a
// vector bias. To preserve arbitrary matmul init tensors, the NPU dispatch
// uses a zero bias and a CPU-affinitized workgroup extends the f16 result to
// f32 and adds the original init tensor.

#rocket_fc_target_4x32x16 = #hal.executable.target<"rocket", "rocket-flatbuffer-v1", {
  kernel = "fully_connected",
  m = 4 : i32, k = 32 : i32, n = 16 : i32,
  input_zero_point = 0 : i32, output_zero_point = 0 : i32, weights_zero_point = 0 : i32,
  input_scale = 1.0 : f32, weights_scale = 1.0 : f32, output_scale = 1.0 : f32,
  truncate_bits = 0 : i32,
  activation = "none", activation_cmp = 0 : i32,
  precision = "fp16"
}>

#rocket_fc_pipeline_layout = #hal.pipeline.layout<constants = 0, bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>

module attributes {transform.with_named_sequence} {

  hal.executable private @rocket_fc_executable_4x32x16 {
    hal.executable.variant public @rocket_fc_v1_4x32x16 target(#rocket_fc_target_4x32x16) {
      hal.executable.export public @rocket_fc_4x32x16 ordinal(0) layout(#rocket_fc_pipeline_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @rocket_fc_4x32x16() {
          return
        }
      }
    }
  }

  util.func private @call_rocket_fc_4x32x16(
      %lhs: tensor<4x32xf16>,
      %rhs: tensor<32x16xf16>,
      %init: tensor<4x16xf32>) -> tensor<4x16xf32> {
    %zero_bias_empty = tensor.empty() : tensor<16xf16>
    %zero_f16 = arith.constant 0.0 : f16
    %zero_bias = linalg.fill ins(%zero_f16 : f16)
        outs(%zero_bias_empty : tensor<16xf16>) -> tensor<16xf16>
    %raw_f16 = flow.dispatch
        @rocket_fc_executable_4x32x16::@rocket_fc_v1_4x32x16::@rocket_fc_4x32x16(
          %lhs, %rhs, %zero_bias)
        {stream.affinity = #hal.device.affinity<@rocket_device>}
        : (tensor<4x32xf16>, tensor<32x16xf16>, tensor<16xf16>)
          -> tensor<4x16xf16>

    %final = flow.dispatch.workgroups(%raw_f16, %init)
        : (tensor<4x16xf16>, tensor<4x16xf32>) -> tensor<4x16xf32>
        attributes {stream.affinity = #hal.device.affinity<@cpu_device>} =
        (%raw_arg: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x16xf16>>,
         %init_arg: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x16xf32>>,
         %out_arg: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x16xf32>>) {
      %raw = iree_tensor_ext.dispatch.tensor.load %raw_arg,
          offsets = [0, 0], sizes = [4, 16], strides = [1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x16xf16>>
          -> tensor<4x16xf16>
      %init_value = iree_tensor_ext.dispatch.tensor.load %init_arg,
          offsets = [0, 0], sizes = [4, 16], strides = [1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x16xf32>>
          -> tensor<4x16xf32>
      %empty = tensor.empty() : tensor<4x16xf32>
      %sum = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]
        } ins(%raw, %init_value : tensor<4x16xf16>, tensor<4x16xf32>)
          outs(%empty : tensor<4x16xf32>) {
      ^bb0(%a: f16, %b: f32, %out: f32):
        %a_f32 = arith.extf %a : f16 to f32
        %value = arith.addf %a_f32, %b : f32
        linalg.yield %value : f32
      } -> tensor<4x16xf32>
      iree_tensor_ext.dispatch.tensor.store %sum, %out_arg,
          offsets = [0, 0], sizes = [4, 16], strides = [1, 1]
          : tensor<4x16xf32>
          -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x16xf32>>
      flow.return
    } count() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    util.return %final : tensor<4x16xf32>
  }

  transform.named_sequence @match_fc_4x32x16(
      %root: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %root ["linalg.matmul"] : !transform.any_op
    %batch, %m, %n, %k = transform.iree.match.contraction %root,
        lhs_type = f16, rhs_type = f16, output_type = f32
        : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %m, [4] : !transform.param<i64>
    transform.iree.match.dims_equal %n, [16] : !transform.param<i64>
    transform.iree.match.dims_equal %k, [32] : !transform.param<i64>
    transform.yield %root : !transform.any_op
  }

  transform.named_sequence @cast_and_call_fc_4x32x16(
      %root: !transform.any_op {transform.readonly}) {
    %ins = transform.get_operand %root[all]
        : (!transform.any_op) -> !transform.any_value
    %out = transform.get_result %root[all]
        : (!transform.any_op) -> !transform.any_value
    %module = transform.util.get_nearest_symbol_table %root
        : (!transform.any_op) -> !transform.any_op
    %topology_attr = transform.param.constant #hal.device.topology<links = [
        (@rocket_device -> @cpu_device = {
          transparent_access = true, unified_memory = true}),
        (@cpu_device -> @rocket_device = {
          transparent_access = true, unified_memory = true})
      ]> -> !transform.any_param
    transform.annotate %module "stream.topology" = %topology_attr
        : !transform.any_op, !transform.any_param
    %executable = transform.util.import_symbol
        @rocket_fc_executable_4x32x16 into %module if undefined
        : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol
        @call_rocket_fc_4x32x16 into %module if undefined
        : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
      transform.type_conversion.tensor.cast_shape_dynamic_dims
    } : (!transform.any_op, !transform.any_value, !transform.any_value,
         !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %funcs = transform.structured.match ops{["util.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    transform.foreach %funcs : !transform.any_op {
    ^bb1(%func: !transform.any_op):
      transform.foreach_match in %func
          @match_fc_4x32x16 -> @cast_and_call_fc_4x32x16
          : (!transform.any_op) -> (!transform.any_op)
    }
    transform.apply_dce to %module : !transform.any_op
    transform.yield
  }
}
