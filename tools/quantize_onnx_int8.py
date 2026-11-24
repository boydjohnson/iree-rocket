#!/usr/bin/env python3
"""Statically quantize an fp32 ONNX model into the int8 form Rocket can offload.

Run with a Python that has onnxruntime, e.g.

    ~/.local/share/pipx/venvs/optimum-onnx/bin/python tools/quantize_onnx_int8.py ...

Why static rather than ORT's `quantize_dynamic`: dynamic quantization emits
DynamicQuantizeLinear, which recomputes each activation's scale at runtime by
scanning the whole tensor in f32. On MobileNetV2 that is ~19M f32 elements of
reduction traffic per inference -- far more work than the convolutions it is
trying to speed up. Static quantization bakes the scales in from a calibration
pass, so all of that disappears.

Three choices here are not stylistic; the Rocket path breaks without them, and
`verify_quantized_model` below re-checks each one on the result rather than
trusting the flags:

  * QuantFormat.QOperator, so convolutions become `QLinearConv`. That is the
    op torch-mlir lowers to `linalg.conv_2d_nchw_fchw_q`, which the Rocket
    transform spec then dequantizes and claims.

  * per_channel=False. torch-mlir's QLinearConv pattern *silently* falls back
    to dequantizing both operands and running a full f32 convolution when the
    weight scales are per-channel -- no error, just a much slower model than
    the one you started with.

  * A symmetric (zero-point 0) weight quantization, which QuantType.QInt8
    gives by default. Rocket's `int8_accumulator` precision refuses a non-zero
    weight zero point, because only the zero-zero-point hardware bypass is
    validated. Activation zero points are free to be non-zero: the compiler
    folds those into a CPU-side correction.

One choice is about speed rather than correctness: Relu is quantized
alongside Conv (`--quantize-op-types Conv Relu`, the default). For Relu,
ORT's QOperator quantizer does not emit a quantized op -- it *removes* the
activation and gives the convolution in front of it the activation's output
scale and zero point, so the clamp at zero point 0 is the ReLU. (Clip, the
ReLU6 MobileNetV2 uses, is handled the same way by ORT but has not been
measured through this stack, so it is not in the default.) Left
in float, every Conv -> Relu -> Conv edge became QLinearConv -> Dequantize ->
Relu -> Quantize -> QLinearConv: a CPU requantization pass between every
pair of offloaded convolutions (33 of them on ResNet50) and no NPU->NPU edge
anywhere. It also changes the model's numbers (one rounding instead of two),
so outputs are not comparable with a model quantized without it. A Relu
whose input is not quantized -- after a residual Add, which stays float --
is left as it is.

The batch dimension is pinned to 1 first. Every matcher in the Rocket
transform spec requires batch 1, so a symbolic-batch model compiles perfectly
cleanly and offloads nothing at all.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper, shape_inference

try:
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process
except ImportError as err:  # pragma: no cover - environment guard
    raise SystemExit(
        "onnxruntime is required. Try "
        "~/.local/share/pipx/venvs/optimum-onnx/bin/python"
    ) from err


def pin_batch(model: onnx.ModelProto) -> onnx.ModelProto:
    """Replace every symbolic dim on the graph boundary with 1 and re-infer."""

    def pin(values) -> None:
        for value in values:
            for dim in value.type.tensor_type.shape.dim:
                if dim.HasField("dim_param"):
                    dim.ClearField("dim_param")
                    dim.dim_value = 1

    pin(model.graph.input)
    pin(model.graph.output)
    # Stale value_info carries the old symbolic dims through inference and
    # produces a model that disagrees with its own inputs.
    del model.graph.value_info[:]
    return shape_inference.infer_shapes(model)


class ArrayCalibrationReader(CalibrationDataReader):
    """Feeds a fixed stack of already-preprocessed NCHW batches to the calibrator."""

    def __init__(self, input_name: str, batches: np.ndarray) -> None:
        self.input_name = input_name
        self.batches = batches
        self.index = 0

    def get_next(self):
        if self.index >= len(self.batches):
            return None
        batch = self.batches[self.index : self.index + 1]
        self.index += 1
        return {self.input_name: batch}

    def rewind(self) -> None:
        self.index = 0


def load_images(directory: Path, count: int, mean: float, std: float) -> np.ndarray:
    from PIL import Image

    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = sorted(p for p in directory.rglob("*") if p.suffix.lower() in suffixes)
    if not paths:
        raise SystemExit(f"no images found under {directory}")
    paths = paths[:count]
    print(f"calibrating on {len(paths)} image(s) from {directory}")

    batches = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        # Resize-shortest-side-to-256 then center crop 224, the standard
        # ImageNet eval transform this model was trained against.
        width, height = image.size
        scale = 256 / min(width, height)
        image = image.resize(
            (round(width * scale), round(height * scale)), Image.BILINEAR
        )
        width, height = image.size
        left, top = (width - 224) // 2, (height - 224) // 2
        image = image.crop((left, top, left + 224, top + 224))
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - mean) / std
        batches.append(array.transpose(2, 0, 1))
    return np.stack(batches).astype(np.float32)


def synthetic_batches(count: int, mean: float, std: float) -> np.ndarray:
    """Random calibration input.

    This produces a *functional* model with *bad* accuracy: calibration is
    what picks each activation's range, and noise has neither the dynamic
    range nor the per-channel structure of real images. Use it to bring up
    and benchmark the compile/offload path, never to judge accuracy.
    """
    rng = np.random.default_rng(20260830)
    # Uniform over the pixel range, then the model's own normalization, so
    # the ranges are at least the right order of magnitude.
    pixels = rng.uniform(0.0, 1.0, size=(count, 224, 224, 3)).astype(np.float32)
    normalized = (pixels - mean) / std
    return normalized.transpose(0, 3, 1, 2).copy()


def asymmetric_pad_convs(path: Path) -> list[str]:
    """Conv nodes whose ONNX `pads` are asymmetric.

    torch-mlir's QLinearConv lowering *segfaults* on these. Its padding helper
    picks a pad value with `isa<IntegerType>` / `isa<FloatType>` on the input
    dtype, and a quantized dtype is neither, so the value stays null and the
    aten.pad it builds gets a null operand. Nothing warns; iree-compile dumps
    core. (The ConvInteger path avoids this because
    RocketExpandOnnxConvIntegerPass does its own padding.)

    Leaving these few nodes in float is a real fix rather than a dodge: the
    Rocket transform spec demotes leftover f32 convolutions to f16 and offloads
    them through the fp16 path, so they still run on the NPU.
    """
    model = onnx.load(path)
    names = []
    for node in model.graph.node:
        if node.op_type != "Conv":
            continue
        for attr in node.attribute:
            if attr.name == "pads":
                pads = list(attr.ints)
                half = len(pads) // 2
                if pads[:half] != pads[half:]:
                    names.append(node.name)
    return names


def share_maxpool_quantization(path: Path) -> int:
    """Give each MaxPool's output its input's quantization.

    QOperator format leaves MaxPool in float: `DequantizeLinear(x) ->
    MaxPool -> QuantizeLinear` with an output scale calibrated on its own.
    Max commutes with a monotone dequantization, so with equal parameters
    the pair is exactly a max pool on the int8 codes -- no rounding at all --
    and that is what lets the NPU run it at int8. With different parameters
    it is a max pool followed by a requantization, which only adds rounding:
    every value a max pool outputs is already on its input's grid. ONNX
    Runtime's QDQ quantizer shares the parameters the same way.

    Rewrites the output scale and zero-point initializers in place, so every
    consumer (the QuantizeLinear and the convolutions that read its scale)
    stays consistent. Returns the number of MaxPools changed.
    """
    model = onnx.load(path)
    graph = model.graph
    producers = {out: node for node in graph.node for out in node.output}
    initializers = {init.name: init for init in graph.initializer}
    changed = 0
    for node in graph.node:
        if node.op_type != "MaxPool":
            continue
        dequant = producers.get(node.input[0])
        if dequant is None or dequant.op_type != "DequantizeLinear":
            continue
        quants = [n for n in graph.node if n.op_type == "QuantizeLinear" and n.input[0] == node.output[0]]
        if len(quants) != 1:
            continue
        quant = quants[0]
        if len(dequant.input) < 3 or len(quant.input) < 3:
            continue
        pairs = [(dequant.input[1], quant.input[1]), (dequant.input[2], quant.input[2])]
        if not all(src in initializers and dst in initializers for src, dst in pairs):
            continue
        for src, dst in pairs:
            initializers[dst].CopyFrom(
                numpy_helper.from_array(numpy_helper.to_array(initializers[src]), dst)
            )
        changed += 1
    onnx.save(model, path)
    return changed


def quantize_residual_skips(path: Path) -> int:
    """Make every residual Add read its float operand back through int8.

    After QOperator quantization a residual block's tail is
    `Add(DequantizeLinear(conv_out), skip)` with `skip` a *float* tensor --
    the previous block's ReLU output, kept in float because Add is not a
    quantized op here. That skip is also quantized (a QuantizeLinear feeds
    the next convolutions from it), so this points the Add at
    `DequantizeLinear(that QuantizeLinear)` instead: both operands int8, the
    form a fully quantized ResNet deploys, and one the NPU's int8 EW add can
    run. It is lossy -- the skip loses everything below its 8-bit
    quantization step -- and it is a model change, so compare the result
    with its own `--no-offload` build, not with the model without it.
    Returns the number of Add operands rewritten.
    """
    model = onnx.load(path)
    graph = model.graph
    producers = {out: node for node in graph.node for out in node.output}
    quantizers: dict[str, onnx.NodeProto] = {}
    for node in graph.node:
        if node.op_type == "QuantizeLinear":
            quantizers.setdefault(node.input[0], node)

    new_nodes: list[onnx.NodeProto] = []
    rewritten = 0
    dequantized: dict[str, str] = {}
    for node in graph.node:
        if node.op_type == "Add":
            for slot, name in enumerate(node.input):
                producer = producers.get(name)
                if producer is not None and producer.op_type == "DequantizeLinear":
                    continue
                quantizer = quantizers.get(name)
                if quantizer is None:
                    continue
                if name not in dequantized:
                    out = f"{name}_requantized_skip"
                    new_nodes.append(
                        onnx.helper.make_node(
                            "DequantizeLinear",
                            [quantizer.output[0], quantizer.input[1], quantizer.input[2]],
                            [out],
                            name=f"{name}_skip_DequantizeLinear",
                        )
                    )
                    dequantized[name] = out
                node.input[slot] = dequantized[name]
                rewritten += 1
        new_nodes.append(node)
    del graph.node[:]
    graph.node.extend(new_nodes)
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return rewritten


def verify_quantized_model(path: Path) -> int:
    """Re-check on the output the three properties the Rocket path depends on.

    Returns the number of offloadable QLinearConv nodes. Raises if any node
    would take a path that silently defeats the point of quantizing.
    """
    model = onnx.load(path)
    initializers = {init.name: init for init in model.graph.initializer}

    def const(name: str):
        init = initializers.get(name)
        return None if init is None else numpy_helper.to_array(init)

    qlinear = [n for n in model.graph.node if n.op_type == "QLinearConv"]
    leftover = [n for n in model.graph.node if n.op_type == "Conv"]
    problems: list[str] = []

    # QOperator format spells anything but Conv/MatMul in Microsoft's own
    # domain (QLinearAdd, QLinearGlobalAveragePool, QGemm, ...). torch-mlir
    # has no lowering for those, so they do not merely stay on the CPU -- they
    # fail input conversion and the whole compile dies.
    microsoft = sorted({n.op_type for n in model.graph.node if n.domain == "com.microsoft"})
    if microsoft:
        problems.append(
            "com.microsoft ops present: "
            + ", ".join(microsoft)
            + ". torch-mlir cannot lower these; restrict --quantize-op-types."
        )

    for node in qlinear:
        # QLinearConv inputs: x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp
        weight_scale = const(node.input[4])
        weight_zp = const(node.input[5])
        if weight_scale is not None and weight_scale.size > 1:
            problems.append(
                f"{node.name}: per-channel weight scale ({weight_scale.size} values). "
                "torch-mlir dequantizes this to a full f32 convolution."
            )
        if weight_zp is not None and np.any(weight_zp != 0):
            problems.append(
                f"{node.name}: non-zero weight zero point. Rocket's "
                "int8_accumulator precision rejects this."
            )

    print(f"  QLinearConv nodes (offloadable): {len(qlinear)}")
    print(f"  Conv nodes left in float (fp16 path): {len(leftover)}")
    if problems:
        print("\nverification failed:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        raise SystemExit("quantized model would not offload as intended")
    print("  weight scales per-tensor, weight zero points zero: OK")
    return len(qlinear)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="fp32 .onnx model")
    parser.add_argument("--output", type=Path, required=True, help="quantized .onnx path")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--calibration-dir", type=Path, help="directory of calibration images"
    )
    source.add_argument(
        "--calibration-npy",
        type=Path,
        help="pre-preprocessed float32 array, shape [N, 3, 224, 224]",
    )
    source.add_argument(
        "--synthetic-calibration",
        type=int,
        metavar="N",
        help="calibrate on N random inputs. Functional but inaccurate -- for "
        "bringing up the offload path, not for judging accuracy.",
    )

    parser.add_argument("--calibration-count", type=int, default=64)
    # google/mobilenet_v2_1.0_224 preprocessing: pixels scaled to [-1, 1].
    # Wrong values here do not break the model, they just calibrate it
    # against ranges it will never see.
    parser.add_argument("--mean", type=float, default=0.5)
    parser.add_argument("--std", type=float, default=0.5)
    parser.add_argument(
        "--exclude-node",
        action="append",
        default=[],
        metavar="NAME",
        help="leave this node in float; it takes the Rocket fp16 path instead. "
        "Repeatable.",
    )
    parser.add_argument(
        "--quantize-op-types",
        nargs="+",
        default=["Conv", "Relu"],
        metavar="OP",
        help="ONNX op types to quantize (default: Conv Relu). Relu is not a "
        "quantized op here: ORT folds it into the convolution in front of it, "
        "which is what lets consecutive offloaded convolutions chain on the "
        "NPU (see the module docstring). Widening further is usually wrong: "
        "QOperator format spells quantized Add/Gemm/pooling as com.microsoft "
        "ops that torch-mlir cannot lower at all, and Conv is the only thing "
        "Rocket offloads anyway. Everything left in float is demoted to f16 "
        "and can still take the Rocket fp16 path.",
    )
    parser.add_argument(
        "--quantize-residual-skip",
        action="store_true",
        help="read every residual Add's float skip back through its own int8 "
        "quantization, so both Add operands are int8 and the NPU can run the "
        "add. Lossy and a model change; off by default.",
    )
    parser.add_argument(
        "--quantize-asymmetric-pads",
        action="store_true",
        help="quantize convolutions with asymmetric padding too. They are "
        "excluded by default because torch-mlir's QLinearConv lowering "
        "segfaults on them; pass this only once that is fixed.",
    )
    args = parser.parse_args()

    work = args.output.parent / f".{args.output.stem}.pinned.onnx"
    prepared = args.output.parent / f".{args.output.stem}.prepared.onnx"

    print(f"pinning batch to 1 in {args.input}")
    model = pin_batch(onnx.load(args.input))
    onnx.save(model, work)

    input_name = model.graph.input[0].name
    print(f"input tensor: {input_name}")

    # ORT strongly recommends this before static quantization: it runs symbolic
    # shape inference and folds constants, without which the calibrator sees
    # far fewer quantizable tensors.
    print("running quant_pre_process")
    quant_pre_process(str(work), str(prepared), skip_symbolic_shape=False)

    if args.calibration_dir:
        batches = load_images(
            args.calibration_dir, args.calibration_count, args.mean, args.std
        )
    elif args.calibration_npy:
        batches = np.load(args.calibration_npy).astype(np.float32)
        print(f"calibrating on {len(batches)} batch(es) from {args.calibration_npy}")
    else:
        batches = synthetic_batches(args.synthetic_calibration, args.mean, args.std)
        print(
            f"WARNING: calibrating on {len(batches)} RANDOM inputs. The model will "
            "run, but its accuracy is meaningless. Use --calibration-dir with real "
            "images before trusting any output.",
            file=sys.stderr,
        )

    excluded = list(args.exclude_node)
    if not args.quantize_asymmetric_pads:
        asymmetric = asymmetric_pad_convs(prepared)
        if asymmetric:
            print(
                f"leaving {len(asymmetric)} asymmetrically-padded conv(s) in float "
                "(torch-mlir's QLinearConv lowering crashes on them); they will "
                "take the Rocket fp16 path:"
            )
            for name in asymmetric:
                print(f"    {name}")
            excluded.extend(asymmetric)

    print(f"quantizing -> {args.output}")
    quantize_static(
        str(prepared),
        str(args.output),
        ArrayCalibrationReader(input_name, batches),
        quant_format=QuantFormat.QOperator,
        per_channel=False,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=args.quantize_op_types,
        nodes_to_exclude=excluded,
    )

    shared = share_maxpool_quantization(args.output)
    if shared:
        print(f"gave {shared} MaxPool output(s) their input's quantization")

    if args.quantize_residual_skip:
        count = quantize_residual_skips(args.output)
        print(f"read {count} residual skip operand(s) back through int8")

    print("verifying:")
    verify_quantized_model(args.output)

    for scratch in (work, prepared):
        scratch.unlink(missing_ok=True)
    print(f"\nwrote {args.output}")
    print("next: iree-import-onnx, then rocket-compiler audit")


if __name__ == "__main__":
    main()
