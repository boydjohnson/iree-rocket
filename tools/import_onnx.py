#!/usr/bin/env python3
"""Pin an ONNX model's symbolic dims, build an ONNX Runtime oracle, import to MLIR.

Three steps that all have to happen, in this order, and each of which has bitten
this repository at least once.

**Pin the symbolic dims.** `iree-import-onnx` will happily import a graph whose
batch is a `dim_param`, but the Rocket ABI fixes batch at one and every matcher
in the transform spec requires it, so a dynamic model compiles cleanly and
offloads *nothing*. Some models leave more than batch symbolic --
`vit-base-patch16-224-ONNX` leaves all four input dims that way, so channels and
the spatial extents need pinning too or the patch-embed convolution stays
symbolic as well. `--dim name=value` sets each one; a `dim_param` with no
value given is an error rather than a guess.

**Build the oracle before importing, not after.** Without a reference from the
model's own runtime, a difference measured later cannot be attributed to the
NPU rather than to the import. ISSUES.md C14 is exactly that case: a
float16-converted ViT that ONNX Runtime runs correctly and that IREE
mis-imports, found only because the f32 arm of the same model was exact
against its oracle while the fp16 arm was not -- on the host CPU, with no NPU
in the picture.

**Write the input as raw `.bin` as well as `.npy`.** `iree-run-module` reads
`--input=@file` as raw bytes, so handing it a `.npy` silently shifts the
tensor by the header. The `.npy` is for numpy; the `.bin` is what
`--input=1x3x224x224xf32=@input.bin` wants. Outputs *are* real npy files.

    tools/import_onnx.py model.onnx --out-dir vit \\
        --dim batch_size=1 --dim num_channels=3 --dim height=224 --dim width=224

Needs `onnx`, `onnxruntime` and `numpy` (a `uv venv` is enough) and
`iree-import-onnx` on PATH.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def pin_dims(model, shape: dict[str, int]) -> None:
    """Replaces every `dim_param` on the graph inputs and outputs."""
    for values in (model.graph.input, model.graph.output):
        for value in values:
            for dim in value.type.tensor_type.shape.dim:
                if not dim.HasField("dim_param"):
                    continue
                name = dim.dim_param
                if name not in shape:
                    raise SystemExit(
                        f"error: {value.name} has symbolic dim {name!r} and no "
                        f"--dim {name}=N was given. Pinning it wrong is worse "
                        f"than not pinning it, so this does not guess."
                    )
                dim.Clear()
                dim.dim_value = shape[name]


def describe(model) -> None:
    for kind, values in (("in ", model.graph.input), ("out", model.graph.output)):
        for value in values:
            dims = [
                d.dim_param if d.HasField("dim_param") else d.dim_value
                for d in value.type.tensor_type.shape.dim
            ]
            print(f"    {kind} {value.name}: {dims}")


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
        "--seed", type=int, default=20260909, help="RNG seed for the oracle input"
    )
    parser.add_argument(
        "--skip-oracle",
        action="store_true",
        help="skip the ONNX Runtime reference. Only for a model whose oracle "
        "already exists: see this module's docstring for why it is the default.",
    )
    args = parser.parse_args()

    import numpy as np
    import onnx
    from onnx import shape_inference

    shape = {}
    for entry in args.dim:
        name, _, value = entry.partition("=")
        if not value.isdigit():
            raise SystemExit(f"error: --dim wants NAME=VALUE, got {entry!r}")
        shape[name] = int(value)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.model.stem

    print(f"[1/3] pinning {args.model}")
    model = onnx.load(str(args.model))
    pin_dims(model, shape)
    # Stale value_info keeps the old symbolic shapes and wins over what
    # inference would derive, so it goes before inference runs.
    del model.graph.value_info[:]
    model = shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model)
    pinned = args.out_dir / f"{stem}.pinned.onnx"
    onnx.save(model, str(pinned))
    describe(model)

    entry = model.graph.input[0]
    dims = [d.dim_value for d in entry.type.tensor_type.shape.dim]
    rng = np.random.default_rng(args.seed)
    sample = rng.uniform(-1.0, 1.0, size=tuple(dims)).astype(np.float32)
    np.save(args.out_dir / f"{stem}.input.npy", sample)
    # Raw bytes, because that is what --input=@file reads.
    sample.tofile(args.out_dir / f"{stem}.input.bin")

    if args.skip_oracle:
        print("[2/3] oracle skipped")
    else:
        print("[2/3] ONNX Runtime oracle")
        import onnxruntime as ort

        session = ort.InferenceSession(
            str(pinned), providers=["CPUExecutionProvider"]
        )
        logits = session.run(None, {entry.name: sample})[0].astype(np.float32)
        np.save(args.out_dir / f"{stem}.oracle.npy", logits)
        flat = logits.reshape(-1)
        print(
            f"    {logits.shape} argmax={int(flat.argmax())} "
            f"min={flat.min():.4f} max={flat.max():.4f}"
        )

    mlir = args.out_dir / f"{stem}.mlir"
    print(f"[3/3] iree-import-onnx -> {mlir}")
    subprocess.run(
        ["iree-import-onnx", str(pinned), "-o", str(mlir)],
        check=True,
    )
    print(f"    {mlir.stat().st_size / 1e6:.0f} MB")
    print(
        f"\nnext: rocket-compiler audit --input {mlir}\n"
        f"      and compare any run against {stem}.oracle.npy, not against zero."
    )


if __name__ == "__main__":
    sys.exit(main())
