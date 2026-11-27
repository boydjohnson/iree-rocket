# Bring-your-own quantization, measured (2026-11-27)

The question: if a user quantizes a model themselves with stock ONNX Runtime
settings and hands it to `rocket-compiler`, what do they get, compared with
the same fp32 model run through `tools/quantize_onnx_int8.py`?

## Setup

- Sources: torchvision MobileNetV2 and ResNet50, fp32, opset 17
  (`survey/mobilenet_v2/mobilenet_v2.f32.onnx`,
  `survey/2026-11/resnet50/resnet50.f32.onnx`).
- Every variant gets the same batch pin, `quant_pre_process`, and the same 32
  synthetic calibration batches (`synthetic_batches`, mean 0.45, std 0.225).
  Only the `quantize_static` arguments differ.
- Variants:
  - `ours` is `tools/quantize_onnx_int8.py` with its defaults: QOperator,
    per-tensor, U8S8, `Conv Relu`.
  - `ours_skip` adds `--quantize-residual-skip` (ResNet50 only).
  - `ort_qdq` is `quantize_static` with every default: QDQ, per-tensor,
    S8S8, all op types.
  - `ort_qdq_pc` is the same with `per_channel=True`.
  - `ort_qop` is QOperator with the other defaults.
  - `ort_qop_pc` is QOperator with `per_channel=True`.
- Pipeline: `tools/import_onnx.py`, then `rocket-compiler compile` (main at
  `ebeb967`, IREE 3.14) for an NPU arm and a `--no-offload` arm.
- Board: orangepi on 7.3.6 with the patched `rocket.ko`, NPU at 600 MHz,
  `--task_topology_cpu_ids=4,5,6,7`.
  - **The CPU governor was `ondemand`**, not the `performance` the surveys
    used. Changing it needs sudo. That inflates every NPU arm (see
    `planck-measurement-environment`), so compare arms within this table
    only, not against earlier surveys.
  - Timing: two interleaved passes, median of 6 repetitions each.
- Artifacts: `survey/quant-byo-2026-11-27/` in the main checkout.

## Results

| model | variant | compiles | NPU sites / CPU sites | conv precision on NPU | NPU ms (p1/p2) | own `--no-offload` ms |
|---|---|---|---|---|---|---|
| MobileNetV2 | ours | yes | 54 / 91 | int8 | 50.6 / 47.7 | 50.3 / 51.7 |
| MobileNetV2 | ort_qdq | yes | 54 / 127 | **fp16** | 81.5 / 79.3 | 53.9 / 55.1 |
| MobileNetV2 | ort_qdq_pc | yes | 54 / 127 | **fp16** | 82.3 / 79.8 | 56.4 / 55.6 |
| MobileNetV2 | ort_qop, ort_qop_pc | **no** (both arms) | | | | |
| MobileNetV2 | fp16 build, for scale | | | fp16 | 16.2 | |
| ResNet50 | ours | yes | 56 / 31 | int8 | 65.2 / 65.6 | 481 / 488 |
| ResNet50 | ours_skip | yes | 56 / 23 | int8 | 69.6 / 69.4 | 493 / 480 |
| ResNet50 | ort_qdq | yes | 56 / 120 | **fp16** | 239 / 250 | 1565 / 1561 |
| ResNet50 | ort_qdq_pc | yes | 56 / 120 | **fp16** | 249 / 244 | 1586 / 1572 |
| ResNet50 | ort_qop, ort_qop_pc | **no** (both arms) | | | | |
| ResNet50 | fp16 build, for scale | | | fp16 | 77.3 | |

Every module that compiled ran, and its top-1 agreed with onnxruntime on the
quantized model. Synthetic calibration and a synthetic input mean that says
nothing about accuracy.

## What it shows

1. **Stock QOperator does not compile.** ORT's QOperator quantizer emits
   `com.microsoft` `QLinearAdd`, `QLinearGlobalAveragePool` and `QGemm` for
   the non-conv ops. The `QLinearAdd` (on si8 activations, ORT's default
   QInt8) fails torch-mlir legalization on both arms. QOperator is the form
   the Rocket matchers want, and our tool only reaches it by quantizing
   nothing but `Conv` and `Relu`.

2. **Stock QDQ, ORT's default, compiles and offloads every conv, at fp16.**
   IREE has no pass that turns `DQ -> conv -> Q` into an integer convolution.
   So the weights dequantize at compile time and the spec claims an f16
   conv. Every activation edge keeps its fake quantization on the CPU:
   - each conv's fp16 output is quantized to i8 (`..._f16xi8`) and
     dequantized straight back (`..._i8xf16`);
   - that adds 110 elementwise/transpose dispatches on ResNet50 and breaks
     every NPU-to-NPU edge.

   The cost:
   - MobileNetV2's NPU arm is 1.45x **slower than its own CPU build** and
     4.9x slower than the plain fp16 build.
   - ResNet50 is 3.7x slower than `ours`, and 3.1x slower than not
     quantizing at all.

3. **Per-channel changed nothing**, because on the QDQ path the weights are
   dequantized to f16 anyway. The torch-mlir per-channel fallback that
   `quantize_onnx_int8.py` guards against only bites on the QOperator path,
   which does not compile here for an earlier reason.

4. **Our tool is what makes int8 pay.** ResNet50 `ours` runs 7.4x faster than
   its own CPU arm. MobileNetV2 `ours` is only level with its CPU arm, and
   3x slower than fp16. The tool leaves ReLU6 (`Clip`) in float, so every
   MobileNetV2 edge carries a CPU requant. Stock ORT folds `Clip` by itself.

5. **Regression found along the way: the int8 residual add no longer reaches
   the NPU.** `ours_skip` gains nothing over `ours`. Recompiling the very
   model that measured 50.6 -> 27.5 ms on 2026-11-24
   (`survey/relufuse/resnet50-224.int8s.mlir`) with today's compiler gives
   the same module. There is no `rocket_elementwise_add_int8_relu`
   executable, and the add runs as a CPU
   `elementwise_transpose_256x3136_i8` over two i8 inputs.
   - **Cause.** The IREE 3.14 bump (#100) changed how DequantizeLinear and
     QuantizeLinear lower: they now arrive as `iree_linalg_ext.dequantize_affine`
     / `quantize_affine`. `rocket-fuse-int8-residual-add` and
     `rocket-fuse-int8-max-pool` only knew the old `linalg.generic` form.
     The bump's int8 check compared the board set, which has no int8 skip.
     It is break 6 in `agent-docs/iree-3.14-bump.md`.
   - **Fixed** in the same change as this document: both passes accept the
     affine ops, and the lit tests gain cases in the 3.14 form.
   - **Re-measured on the board**, governor `performance` this time, so not
     comparable with the table above:
     - `ours_skip`: 41.1 / 43.8 ms -> **17.6 / 17.2 ms**, with 15 of 16
       residual adds on the NPU and CPU dispatches 23 -> 5. Top-5 equals
       onnxruntime's; max|err| against it is 0.115 (was 0.132).
     - `ours`: 40.3 / 40.9 -> 36.0 / 36.1 ms, from the int8 stem pool.
       Its output is bit-identical to the unfixed build.
     - The Nov-24 model recompiled: 18.9 / 17.8 ms.
     - The `--no-offload` arm is unchanged (1e-7). 0 NPU timeouts.

## Implications for scope

- **"Bring your own pre-quantized ONNX" does not work today with either
  format ORT offers.** QOperator fails to build. QDQ builds, then runs
  slower than fp16, and on MobileNetV2 slower than the CPU.
- **The highest-value compiler item is a QDQ -> requantized-conv
  rewrite**: `Q(conv(DQ(x), DQ(w)) + b)` into the canonical form
  `rocket-fuse-int8-requant-epilogue` already produces. QDQ is ORT's default
  and the common interchange format. Per-channel weight scales come right
  behind: the BS plane can carry them (`iree-rocket-hal/src/rocket/conv.rs`,
  `pack_int8_bias_to_bs`), but the executable ABI is per-tensor.
- **The quantizer is part of the product either way.**
  - On both models, its choices (which ops to fold, which scales to share,
    the skip quantization) decide between 65 ms and 245 ms.
  - Folding `Clip` is a missing default there.
