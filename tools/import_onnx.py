#!/usr/bin/env python3
"""Pin an ONNX model's symbolic dims, build an ONNX Runtime oracle, import to MLIR.

Steps that all have to happen, in this order, each of which has bitten this
repository at least once.

**Pin the symbolic dims.** `iree-import-onnx` will happily import a graph whose
batch is a `dim_param`, but the Rocket ABI fixes batch at one and every matcher
in the transform spec requires it, so a dynamic model compiles cleanly and
offloads *nothing*. Some models leave more than batch symbolic --
`vit-base-patch16-224-ONNX` leaves all four input dims that way, so channels
and the spatial extents need pinning too. `--dim NAME=VALUE` sets each one; a
`dim_param` with no value given is an error rather than a guess.

**Build the oracle before importing, not after.** Without a reference from the
model's own runtime, a difference measured later cannot be attributed to the
NPU rather than to the import. ISSUES.md C14 is exactly that case: a
float16-converted ViT that ONNX Runtime runs correctly and that IREE
mis-imports, found only because the f32 arm of the same model was exact
against its oracle while the fp16 arm was not -- on the host CPU, with no NPU
in the picture.

**Write the input as raw `.bin` as well as `.npy`.** `iree-run-module` reads
`--input=@file` as raw bytes, so handing it a `.npy` silently shifts the tensor
by the header. Outputs *are* real npy files.

## ONNX Runtime *optimized* exports (com.microsoft contrib ops)

`onnx-community/Qwen3-0.6B-ONNX` and models like it are ORT-optimized exports
carrying `GroupQueryAttention`, `RotaryEmbedding`,
`SimplifiedLayerNormalization` and `SkipSimplifiedLayerNormalization`; there is
often no plain-op variant published. torch-mlir supports all four and registers
them into the *default* domain pattern set, so a node written with an empty
domain still matches -- but four extra rewrites are needed, and this script
applies them automatically when it sees such a node:

1. **`value_info` is pinned, not cleared.** The opposite of what the plain path
   above does, and the opposite of what README says. ONNX has no schema for
   ORT's fused ops, so `infer_shapes` cannot recompute their result types;
   clearing drops them, the importer emits `!torch.none` for the result, and
   legalization then refuses the op. ORT also writes *expression* dims into
   `value_info` (`sequence_length * num_attention_heads`), which never appear
   on the graph boundary and are therefore easy to miss -- `--dim` values are
   substituted into products of known names to resolve them.

2. **Trailing empty optional node inputs are dropped.**
   `OpBinder::tensorOperandsList` counts operands including the `!torch.none`
   the importer materializes, and several com.microsoft patterns dispatch on
   that count. ORT writes `GroupQueryAttention` with 9 inputs whose last two
   are empty when `do_rotary=0`; the pattern wants 7 and silently declines.

3. **`SkipSimplifiedLayerNormalization` is split** into `Add` +
   `SimplifiedLayerNormalization`. torch-mlir's pattern declines any node with
   more than two results; every one of these has four, and the residual sum is
   genuinely used. Both halves are separately supported.

4. **Rank-3 `RotaryEmbedding` is routed through the rank-4 entry point.**
   torch-mlir's rank-3 path reshapes where it must transpose, so every token
   but a handful gets another position's rotation and the logits come out
   uncorrelated. `wrap_rank3_rotary_embedding` has the formula and the
   evidence. ONNX Runtime computes the wrapped graph bit-identically, which
   is what makes it a workaround and not a change of model.

5. **`--large-model` is passed to `iree-import-onnx`**, because `onnx.checker`
   rejects `SimplifiedLayerNormalization` in the default domain and that flag
   is what skips the checker.

    tools/import_onnx.py model.onnx --out-dir vit \\
        --dim batch_size=1 --dim num_channels=3 --dim height=224 --dim width=224

    tools/import_onnx.py model.onnx --out-dir qwen3 \\
        --dim batch_size=1 --dim sequence_length=128 \\
        --dim past_sequence_length=0 --dim total_sequence_length=128 \\
        --dim num_attention_heads=16 --dim num_key_value_heads=8

Needs `onnx`, `onnxruntime` and `numpy` (a `uv venv` is enough) and
`iree-import-onnx` on PATH.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import zlib
from pathlib import Path

CONTRIB_OPS = frozenset(
    {
        "GroupQueryAttention",
        "RotaryEmbedding",
        "SimplifiedLayerNormalization",
        "SkipSimplifiedLayerNormalization",
        "MultiHeadAttention",
        "Attention",
    }
)


def resolve_dim(name: str, shape: dict[str, int]) -> int | None:
    """Value for one `dim_param`, resolving `a * b` products of known names.

    ORT writes expression dims into `value_info`, and they are not optional:
    an unpinned one leaves a symbolic extent inside a fused op's result type.
    """
    if name in shape:
        return shape[name]
    factors = [part.strip() for part in name.split("*")]
    if len(factors) > 1:
        values = []
        for factor in factors:
            if factor in shape:
                values.append(shape[factor])
            elif factor.isdigit():
                values.append(int(factor))
            else:
                return None
        product = 1
        for value in values:
            product *= value
        return product
    return None


def pin_value(value, shape: dict[str, int], unresolved: set[str]) -> None:
    for dim in value.type.tensor_type.shape.dim:
        if not dim.HasField("dim_param"):
            continue
        resolved = resolve_dim(dim.dim_param, shape)
        if resolved is None:
            unresolved.add(dim.dim_param)
            continue
        dim.Clear()
        dim.dim_value = resolved


def has_contrib_ops(model) -> bool:
    return any(
        node.domain == "com.microsoft" or node.op_type in CONTRIB_OPS
        for node in model.graph.node
    )


def drop_trailing_empty_inputs(model) -> int:
    """Removes explicitly-unset trailing optional inputs. See step 2."""
    dropped = 0
    for node in model.graph.node:
        while len(node.input) > 0 and node.input[-1] == "":
            del node.input[-1]
            dropped += 1
    return dropped


def split_skip_layernorm(model, onnx) -> int:
    """`SkipSimplifiedLayerNormalization` -> `Add` + `SimplifiedLayerNormalization`.

    ORT's node takes (input, skip, gamma) and yields (output, mean, inv_std,
    input_plus_skip). torch-mlir declines anything with more than two results,
    so the sum is computed by an explicit `Add` -- which is also the value the
    fourth result carries, so consumers of it bind to the `Add` directly.
    """
    from onnx import helper

    nodes = list(model.graph.node)
    rewritten = []
    count = 0
    for node in nodes:
        if node.op_type != "SkipSimplifiedLayerNormalization":
            rewritten.append(node)
            continue
        count += 1
        data, skip = node.input[0], node.input[1]
        gamma = node.input[2] if len(node.input) > 2 else None
        # The fourth result is the residual sum when the producer named it;
        # otherwise the Add needs a fresh name for its own result.
        summed = node.output[3] if len(node.output) > 3 and node.output[3] else (
            f"{node.name or node.output[0]}_skip_sum"
        )
        rewritten.append(
            helper.make_node("Add", [data, skip], [summed], name=f"{summed}_add")
        )
        norm_inputs = [summed] + ([gamma] if gamma else [])
        norm = helper.make_node(
            "SimplifiedLayerNormalization",
            norm_inputs,
            [node.output[0]],
            name=f"{node.output[0]}_norm",
        )
        for attribute in node.attribute:
            if attribute.name in ("epsilon", "axis"):
                norm.attribute.append(attribute)
        rewritten.append(norm)
    if count:
        del model.graph.node[:]
        model.graph.node.extend(rewritten)
    return count


def sample_for(value, shape: dict[str, int], overrides: dict[str, str], np, onnx):
    """Oracle input for one graph input.

    Heuristics, all overridable with `--input NAME=KIND`: a name mentioning
    `mask` gets ones (a zero mask can make attention's softmax degenerate),
    `position` gets an arange, another integer input gets small token ids, and
    a float input gets uniform noise -- or nothing at all when an extent is
    zero, which is what an empty KV cache looks like on a prefill.
    """
    tensor = value.type.tensor_type
    dims = [d.dim_value for d in tensor.shape.dim]
    is_int = tensor.elem_type in (
        onnx.TensorProto.INT64,
        onnx.TensorProto.INT32,
    )
    dtype = np.int64 if tensor.elem_type == onnx.TensorProto.INT64 else (
        np.int32 if tensor.elem_type == onnx.TensorProto.INT32 else np.float32
    )
    kind = overrides.get(value.name)
    if kind is None:
        lowered = value.name.lower()
        if 0 in dims:
            kind = "zeros"
        elif "mask" in lowered:
            kind = "ones"
        elif "position" in lowered:
            kind = "arange"
        elif is_int:
            kind = "ids"
        else:
            kind = "uniform"

    # crc32, not hash(): Python randomizes string hashing per process, so
    # hash() here would generate different inputs on every run and the
    # oracle would not be comparable with anything produced by another.
    rng = np.random.default_rng(zlib.crc32(value.name.encode()))
    if kind == "zeros":
        return np.zeros(dims, dtype=dtype)
    if kind == "ones":
        return np.ones(dims, dtype=dtype)
    if kind == "arange":
        return np.arange(int(np.prod(dims)), dtype=dtype).reshape(dims)
    if kind == "ids":
        return rng.integers(0, 1000, size=dims).astype(dtype)
    return rng.uniform(-1.0, 1.0, size=dims).astype(dtype)


def wrap_rank3_rotary_embedding(model, onnx, shapes) -> int:
    """Routes rank-3 `RotaryEmbedding` through the rank-4 entry point.

    torch-mlir's `ConvertOnnxVariantRotaryEmbeddingOp` accepts rank-3 or
    rank-4 input and **its rank-3 path is wrong**: it goes from
    `[batch, seq, heads*head_size]` to `[batch, heads, seq, head_size]` with a
    `tensor.reshape`, which reinterprets the row-major buffer instead of
    transposing the seq and head axes. The element at true (seq `s`, head `n`)
    is then visited at reinterpreted sequence index `(heads*s + n) mod seq`
    and gets that position's rotation; only pairs where
    `(heads*s + n) mod seq == s` survive, which on Qwen3-0.6B is 8 of 1024.
    The symptom is logits uncorrelated with the reference while every
    non-attention output looks fine.

    The rank-4 entry point skips that reshape and is correct, so each node is
    wrapped: `Reshape [B,S,H] -> [B,S,n,hs]`, `Transpose perm=[0,2,1,3]`,
    RotaryEmbedding, `Transpose` back, `Reshape` back. `head_size` is
    `2 * cos_cache.shape[1]`. ONNX Runtime computes the wrapped graph
    bit-identically to the unwrapped one, which is what makes this a
    workaround rather than a change of model.

    Upstream bug, worked around here, not fixed: a real fix is a transpose
    instead of the reshape in that pattern, and a rebuilt libIREECompiler.so.
    """
    from onnx import helper, numpy_helper
    import numpy as np

    nodes = list(model.graph.node)
    rewritten = []
    count = 0
    for node in nodes:
        if node.op_type != "RotaryEmbedding":
            rewritten.append(node)
            continue
        source = shapes.get(node.input[0])
        cos = shapes.get(node.input[2]) if len(node.input) > 2 else None
        if not source or len(source) != 3 or not cos or len(cos) != 2:
            # Already rank 4, or a shape this cannot reason about: leave it
            # alone rather than wrap something that is not the broken form.
            rewritten.append(node)
            continue
        batch, seq, hidden = source
        head_size = 2 * cos[1]
        if head_size <= 0 or hidden % head_size:
            rewritten.append(node)
            continue
        heads = hidden // head_size
        count += 1

        tag = node.output[0]
        split_shape = f"{tag}_rope_split_shape"
        merge_shape = f"{tag}_rope_merge_shape"
        model.graph.initializer.append(
            numpy_helper.from_array(
                np.array([batch, seq, heads, head_size], dtype=np.int64), split_shape
            )
        )
        model.graph.initializer.append(
            numpy_helper.from_array(
                np.array([batch, seq, hidden], dtype=np.int64), merge_shape
            )
        )

        split, permuted, rotated, unpermuted = (
            f"{tag}_rope_split",
            f"{tag}_rope_bnsh",
            f"{tag}_rope_rotated",
            f"{tag}_rope_bsnh",
        )
        rewritten.append(
            helper.make_node("Reshape", [node.input[0], split_shape], [split])
        )
        rewritten.append(
            helper.make_node("Transpose", [split], [permuted], perm=[0, 2, 1, 3])
        )
        rank4 = helper.make_node(
            "RotaryEmbedding",
            [permuted] + list(node.input[1:]),
            [rotated],
            name=f"{tag}_rope_rank4",
        )
        rank4.attribute.extend(node.attribute)
        rank4.domain = node.domain
        rewritten.append(rank4)
        rewritten.append(
            helper.make_node("Transpose", [rotated], [unpermuted], perm=[0, 2, 1, 3])
        )
        rewritten.append(
            helper.make_node("Reshape", [unpermuted, merge_shape], [node.output[0]])
        )

        # RotaryEmbedding has no ONNX schema, so inference cannot type any of
        # these; declaring them by hand is the same requirement as step 2.
        for name, dims in (
            (split, [batch, seq, heads, head_size]),
            (permuted, [batch, heads, seq, head_size]),
            (rotated, [batch, heads, seq, head_size]),
            (unpermuted, [batch, seq, heads, head_size]),
        ):
            model.graph.value_info.append(
                helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, dims)
            )
    if count:
        del model.graph.node[:]
        model.graph.node.extend(rewritten)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--dim",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="value for one dim_param; repeatable, and every one must be given",
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="NAME=KIND",
        help="override an oracle input: zeros, ones, arange, ids or uniform",
    )
    parser.add_argument("--skip-oracle", action="store_true")
    args = parser.parse_args()

    import numpy as np
    import onnx
    from onnx import shape_inference

    shape = {}
    for entry in args.dim:
        name, _, value = entry.partition("=")
        if not value.lstrip("-").isdigit():
            raise SystemExit(f"error: --dim wants NAME=VALUE, got {entry!r}")
        shape[name] = int(value)
    overrides = dict(entry.split("=", 1) for entry in args.input)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.model.stem

    print(f"[1/3] pinning {args.model}")
    model = onnx.load(str(args.model))
    contrib = has_contrib_ops(model)
    print(f"    com.microsoft contrib ops: {'yes' if contrib else 'no'}")

    unresolved: set[str] = set()
    for value in list(model.graph.input) + list(model.graph.output):
        pin_value(value, shape, unresolved)

    if contrib:
        # Pin, do not clear: infer_shapes cannot recompute a fused op's result
        # type, and a dropped one becomes !torch.none at the importer.
        for value in model.graph.value_info:
            pin_value(value, shape, unresolved)
        dropped = drop_trailing_empty_inputs(model)
        split = split_skip_layernorm(model, onnx)
        shapes = {}
        for value in (
            list(model.graph.input)
            + list(model.graph.output)
            + list(model.graph.value_info)
        ):
            dims = value.type.tensor_type.shape.dim
            if all(not d.HasField("dim_param") for d in dims):
                shapes[value.name] = [d.dim_value for d in dims]
        for initializer in model.graph.initializer:
            shapes[initializer.name] = list(initializer.dims)
        wrapped = wrap_rank3_rotary_embedding(model, onnx, shapes)
        print(f"    dropped {dropped} empty optional input(s)")
        print(f"    split {split} SkipSimplifiedLayerNormalization node(s)")
        print(f"    wrapped {wrapped} rank-3 RotaryEmbedding node(s)")
    else:
        del model.graph.value_info[:]
        model = shape_inference.infer_shapes(model, strict_mode=True)
        onnx.checker.check_model(model)

    if unresolved:
        raise SystemExit(
            "error: unpinned dim_param(s): "
            + ", ".join(sorted(unresolved))
            + "\n       pass each as --dim NAME=VALUE. A product of pinned "
            "names resolves on its own; anything else does not, and guessing "
            "one is worse than stopping."
        )

    pinned = args.out_dir / f"{stem}.pinned.onnx"
    onnx.save(
        model,
        str(pinned),
        save_as_external_data=contrib,
        all_tensors_to_one_file=True,
        location=f"{stem}.pinned.onnx_data" if contrib else None,
    )

    samples = {
        value.name: sample_for(value, shape, overrides, np, onnx)
        for value in model.graph.input
    }
    for name, sample in samples.items():
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
        np.save(args.out_dir / f"{stem}.{safe}.npy", sample)
        # Raw bytes, because that is what --input=@file reads.
        sample.tofile(args.out_dir / f"{stem}.{safe}.bin")
    print(f"    {len(samples)} input(s) written")

    if args.skip_oracle:
        print("[2/3] oracle skipped")
    else:
        print("[2/3] ONNX Runtime oracle")
        import onnxruntime as ort

        session = ort.InferenceSession(
            str(pinned), providers=["CPUExecutionProvider"]
        )
        wanted = {i.name for i in session.get_inputs()}
        outputs = session.run(
            None, {k: v for k, v in samples.items() if k in wanted}
        )
        primary = np.asarray(outputs[0], dtype=np.float32)
        np.save(args.out_dir / f"{stem}.oracle.npy", primary)
        flat = primary.reshape(-1)
        print(
            f"    {primary.shape} argmax={int(flat.argmax())} "
            f"min={flat.min():.4f} max={flat.max():.4f}"
        )

    mlir = args.out_dir / f"{stem}.mlir"
    print(f"[3/3] iree-import-onnx -> {mlir}")
    command = ["iree-import-onnx", str(pinned), "-o", str(mlir)]
    if contrib:
        # onnx.checker rejects SimplifiedLayerNormalization in the default
        # domain; --large-model is what skips it.
        command.append("--large-model")
    subprocess.run(command, check=True)
    print(f"    {mlir.stat().st_size / 1e6:.0f} MB")
    print(
        f"\nnext: rocket-compiler audit --input {mlir}\n"
        f"      and compare any run against {stem}.oracle.npy, not against zero."
    )


if __name__ == "__main__":
    sys.exit(main())
