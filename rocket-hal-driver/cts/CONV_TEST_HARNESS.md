# RKT1 convolution test harness

`conv_test_harness.{h,cc}` exercises a convolution at the public driver
boundary. It is intentionally not another register-command fixture.

```text
Conv2dProblem + dense tensors
             |
             v
RKT1 FlatBuffer executable
             |
             v
IREE HAL: prepare -> dispatch -> queue_execute
             |
             v
Rocket driver packing + NPU + output compaction
             |
             v
dense NHWC result ---- compare ---- independent CPU convolution
```

The harness accepts a regular FP16 convolution described by:

- input height, width, and channel count;
- output channel count;
- independent height/width kernel extents;
- equal-axis stride; and
- symmetric height/width padding (`pad_top` is also bottom padding and
  `pad_left` is also right padding).

Input data is logical dense NHWC and weights are logical dense HWCF. The
harness does not call Rocket tensor-packing helpers. That is deliberate: the
test covers the driver's deferred input packing, coefficient packing, actual
buffer binding order, FP16-to-FP32 BRDMA bias widening, NPU submission, and
dense output compaction. The reference path indexes the logical tensors
directly and has no dependency on the Rocket layouts or register-command
builder.

The harness releases its input, weight, and bias buffer references after
recording and before queue submission. Correct execution therefore also checks
the IREE HAL requirement that a command buffer retain every direct dispatch
binding for its full lifetime.

`conv_schema_harness_test.cc` shows the extension pattern. Add a
`Conv2dProblem` to `INSTANTIATE_TEST_SUITE_P`; the deterministic tensor
generators and the CPU route adapt to its geometry. Current cases cover 1x1,
3x3 SAME padding, a non-square 3x5 kernel, an even 4x2 kernel with asymmetric
axis padding, and a stride-2 3x3 kernel. Non-square/even cases use stride 1
because that is the extent of their current capture-backed CBUF policy.

The checked comparison rounds operands and expected results to FP16 and uses a
small absolute/relative tolerance for accumulation-order differences. Large
layout, binding, or padding errors remain readily visible.

## Deliberate current boundaries

- The helper covers regular FP16 convolution. Int8 needs a reference
  requantization policy matching `Multiplier` and a logical-to-hardware int8
  coefficient bridge in the driver.
- Depthwise needs the driver to accept a documented logical depthwise weight
  layout and invoke its existing depthwise packer.
- Bias is a logical dense FP16 tensor. Dedicated exact-sized bindings and a
  nonzero-offset binding populated by a preceding command-buffer update
  between poisoned neighboring suballocations verify the driver's deferred
  widening and zero-padding bridge to BRDMA storage.
- The host build verifies compilation and skips cleanly without
  `/dev/accel/accel0`. The parameterized cases must be run on an RK3588 board
  for NPU result validation.

These boundaries are adapter points in `RunFp16Conv2d`, not reasons to fork
the dispatch or comparison machinery for each new ukernel or datatype.
