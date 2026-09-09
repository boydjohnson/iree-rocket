#!/usr/bin/env python3
"""Export a torchvision model to ONNX in the form this stack wants to import.

A registry rather than a one-off script, because the survey in ISSUES.md is a
comparison *between* models and an export difference between two of them is
indistinguishable from a compiler difference. Everything here exports the same
way: `.eval()`, static batch 1, opset 17, no `dynamic_axes`.

Three choices are load-bearing, each of which has cost this repository time:

**fp16 comes from `model.half()`, never from a post-hoc converter.**
`onnxconverter_common.float16.convert_float_to_float16` halves an initializer
without halving the input edge it feeds, and the result is a graph ONNX Runtime
refuses to load -- on CLIP, as `Type parameter (T) of Optype (Conv) bound to
different types (tensor(float) and tensor(float16))`. ResNet50 survived the
same conversion, so a model that converts cleanly proves nothing about the
next one.

**No `dynamic_axes`, which pins batch 1 statically.** The Rocket ABI fixes
batch at one and every matcher in the transform spec requires it, so a model
whose batch is a symbolic `dim_param` compiles cleanly and offloads *nothing*.
Exporting without `dynamic_axes` sidesteps the pinning problem instead of
fixing it afterwards -- `tools/import_onnx.py --dim` is for models published by
someone else.

**`graph.value_info` is cleared and re-inferred.** The exporter writes types
that shape inference can reproduce, and stale ones survive graph edits made
later.

An f32 export is worth having next to the fp16 one and both are offered here.
The transform spec demotes convolutions and matmuls to f16 itself and restores
f32 on whatever the match loop leaves behind, so an f32 import chooses
precision per *operation*; the fp16 import chooses it for the whole graph and
is the stronger CPU baseline. They are different measurements, not two spellings
of one.
"""

from __future__ import annotations

import argparse
from pathlib import Path

def torchvision_recipe(attribute: str, weights_name: str, shape: tuple[int, ...]):
    """A single-image classifier: pretrained weights, one float input.

    The spatial extent comes from the model rather than a uniform 224: a
    survey that silently resizes a model is measuring a shape its author
    never shipped.
    """

    def build(dtype):
        import torch
        import torchvision.models as models

        weights = getattr(models, weights_name).DEFAULT
        model = getattr(models, attribute)(weights=weights).eval()
        if dtype is torch.float16:
            model = model.half()
        return (
            model,
            (torch.randn(*shape, dtype=dtype),),
            ["input"],
            ["output"],
        )

    return build


def blip_captioning_recipe(checkpoint: str, resolution: int, tokens: int):
    """BLIP image captioning, wrapped down to one forward pass.

    Two things this wrapper decides, both of which change what gets measured:

    **One forward pass, not `generate()`.** The Rocket ABI has no way to
    express an autoregressive loop and the survey compares one invocation
    against one invocation, so the export is a single teacher-forced step:
    pixels and a fixed-length token prefix in, decoder logits out.

    **Logits, not the loss.** `BlipForConditionalGeneration` returns a scalar
    loss alongside its logits when labels are implied, and a scalar is a
    useless correctness signal -- CLIP's `logits_per_image` taught this
    repository that once already, at 1x1. The `tokens x vocab` logit block is
    wide enough that an argmax over it means something.
    """

    def build(dtype):
        import torch
        from transformers import BlipForConditionalGeneration

        model = BlipForConditionalGeneration.from_pretrained(checkpoint).eval()
        if dtype is torch.float16:
            model = model.half()

        class OneStep(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, pixel_values, input_ids):
                return self.inner(
                    pixel_values=pixel_values, input_ids=input_ids
                ).logits

        # Token ids are int64 whatever the activations are; only the pixels
        # follow the export precision.
        example = (
            torch.randn(1, 3, resolution, resolution, dtype=dtype),
            torch.randint(0, 30000, (1, tokens), dtype=torch.int64),
        )
        return OneStep(model).eval(), example, ["pixel_values", "input_ids"], ["logits"]

    return build


# Every model the survey can export, by name. A recipe is a callable taking the
# torch dtype and returning (module, example inputs, input names, output names),
# so a model that needs a wrapper is not a special case in `export` below.
REGISTRY = {
    # The set the repository already measures, here so a re-export is
    # reproducible rather than remembered.
    "mobilenet_v2": torchvision_recipe(
        "mobilenet_v2", "MobileNet_V2_Weights", (1, 3, 224, 224)
    ),
    "resnet50": torchvision_recipe("resnet50", "ResNet50_Weights", (1, 3, 224, 224)),
    "vgg19": torchvision_recipe("vgg19", "VGG19_Weights", (1, 3, 224, 224)),
    # The two models that carry the 2026-09-09 ceiling raise on a real graph.
    # `wide_resnet50_2` doubles ResNet50's bottleneck widths (to Cout 2048 at
    # 1x1) and `vit_l_16` has the 1024x4096 MLP that LIMITS.md names as the
    # reason `MAX_OUTPUT_CHANNELS` moved from 3584 to 4096.
    "wide_resnet50_2": torchvision_recipe(
        "wide_resnet50_2", "Wide_ResNet50_2_Weights", (1, 3, 224, 224)
    ),
    "vit_l_16": torchvision_recipe("vit_l_16", "ViT_L_16_Weights", (1, 3, 224, 224)),
    "vit_b_16": torchvision_recipe("vit_b_16", "ViT_B_16_Weights", (1, 3, 224, 224)),
    "densenet121": torchvision_recipe(
        "densenet121", "DenseNet121_Weights", (1, 3, 224, 224)
    ),
    "efficientnet_b0": torchvision_recipe(
        "efficientnet_b0", "EfficientNet_B0_Weights", (1, 3, 224, 224)
    ),
    "inception_v3": torchvision_recipe(
        "inception_v3", "Inception_V3_Weights", (1, 3, 299, 299)
    ),
    "resnext50_32x4d": torchvision_recipe(
        "resnext50_32x4d", "ResNeXt50_32X4D_Weights", (1, 3, 224, 224)
    ),
    "convnext_tiny": torchvision_recipe(
        "convnext_tiny", "ConvNeXt_Tiny_Weights", (1, 3, 224, 224)
    ),
    # Vision-language. Two inputs, one of them integer, and a vision tower
    # whose 16x16 patch embedding no matcher claims -- the same shape ViT-L
    # leaves on the CPU.
    "blip": blip_captioning_recipe(
        "Salesforce/blip-image-captioning-base", 384, 16
    ),
}


def ort_constant_fold(path: Path) -> Path:
    """Folds the exporter's runtime shape arithmetic away, via ONNX Runtime.

    `torch.onnx.export` writes `nn.MultiheadAttention`'s reshapes with their
    target shape *computed* -- Shape, Slice, Concat -- rather than as a
    constant, even though every extent is static once the batch is pinned.
    ONNX shape inference cannot see through that, so the Reshape's result type
    is unknown, the importer emits `!torch.vtensor<[],f16>` for it, and
    torch-mlir then fails to legalize the rank-5 Transpose that follows:

        error: failed to legalize operation 'torch.operator' that was
        explicitly marked illegal: "onnx.Transpose" ... perm = [3, 1, 2, 0, 4]
        (!torch.vtensor<[],f16>) -> !torch.vtensor<[],f16>

    ONNX Runtime's BASIC optimization level constant-folds those chains,
    which restores a static shape on every intermediate. BASIC deliberately,
    not EXTENDED: the extended level fuses attention into `com.microsoft`
    contrib ops, which import through an entirely different path with five
    rewrites of its own (see `tools/import_onnx.py`), and this is an export
    we control -- there is no reason to take that path when the plain ops
    import fine.
    """
    import onnxruntime as ort

    folded = path.with_suffix(".folded.onnx")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    options.optimized_model_filepath = str(folded)
    ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    folded.replace(path)
    print(f"    ONNX Runtime constant folding -> {path.name}")
    return path


def export(
    name: str, precision: str, out_dir: Path, opset: int, simplify: bool
) -> Path:
    import torch
    import onnx
    from onnx import shape_inference

    dtype = torch.float16 if precision == "fp16" else torch.float32
    model, example, input_names, output_names = REGISTRY[name](dtype)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.{precision}.onnx"
    print(f"[1/2] torch.onnx.export -> {path}")
    torch.onnx.export(
        model,
        example,
        str(path),
        opset_version=opset,
        input_names=input_names,
        output_names=output_names,
        do_constant_folding=True,
        dynamo=False,
    )

    if simplify:
        path = ort_constant_fold(path)

    print("[2/2] clearing value_info and re-inferring shapes")
    graph = onnx.load(str(path))
    del graph.graph.value_info[:]
    graph = shape_inference.infer_shapes(graph, strict_mode=True)
    onnx.checker.check_model(graph)
    onnx.save(graph, str(path))

    inputs = ", ".join(
        f"{i.name}{[d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]}"
        for i in graph.graph.input
    )
    print(f"    inputs: {inputs}")
    print(f"    {path.stat().st_size / 1e6:.1f} MB")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=sorted(REGISTRY))
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    parser.add_argument("--precision", choices=("fp16", "f32"), default="fp16")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="skip the ONNX Runtime constant-folding pass",
    )
    args = parser.parse_args()
    export(
        args.model,
        args.precision,
        args.out_dir,
        args.opset,
        not args.no_simplify,
    )


if __name__ == "__main__":
    main()
