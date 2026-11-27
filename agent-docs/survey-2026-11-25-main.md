# Model survey, main 3d04335 (2026-11-25)

> **Superseded: this is the last run on the IREE 3.12 toolchain**
> (`a869dc3c`), kept as the "before" for the bump. `main` is on IREE 3.14 since
> b37a044; the current numbers for the same ten models, and why they moved
> (mostly faster `--no-offload` CPU baselines, not slower NPU arms), are in
> [`iree-3.14-bump.md`](iree-3.14-bump.md#10-model-survey-vs-survey2026-11-25-main-orangepi-600-mhz-performance).

Board: orangepi (RK3588), Armbian 7.3.5, rocket.ko PPU + 600 MHz clock + 088, governor `performance`.
Built from a clean worktree at 3d04335 (the main checkout's staged edits excluded). Inputs/oracles
hardlinked from the 2026-11-24 run; compile, run, and bench were all redone. Artifacts, logs and
the scripts (`survey_all.sh`, `count_sites.py`) are in the gitignored `survey/2026-11-25-main/` on laplace. 0 NPU timeouts. All 20 arms argmax == f32 oracle.

| model | NPU sites | CPU sites | NPU 4xA76 ms | CPU 4xA76 ms | x | NPU 8c | CPU 8c | x | vs 11-24 (NPU 4c) |
|---|---|---|---|---|---|---|---|---|---|
| mobilenet_v2 fp16 | 54 | 5 | 9.6 | 54.3 | 5.64 | 9.8 | 57.7 | 5.91 | 10.9 -> 9.6 |
| resnet50 fp16 | 56 | 21 | 52.0 | 894 | 17.2 | 52.6 | 1011 | 19.2 | = |
| resnext50_32x4d fp16 | 56 | 21 | 55.6 | 502 | 9.03 | 57.3 | 685 | 12.0 | = |
| convnext_tiny fp16 | 60 | 216 | 94.1 | 252 | 2.68 | 98.1 | 252 | 2.57 | = |
| wide_resnet50_2 fp16 | 55 | 9 | 87.9 | 1904 | 21.7 | 87.6 | 3012 | 34.4 | = |
| vit_b_16 fp16 | 50 | 249 | 202 | 1664 | 8.24 | 192 | 1327 | 6.91 | = |
| vit_l_16 fp16 | 98 | 489 | 671 | 11430 | 17.0 | 686 | 13871 | 20.2 | = |
| blip fp16 | 170 | 621 | 817 | 4818 | 5.90 | 740 | 4005 | 5.41 | = (CPU arm 9% faster) |
| bert_base f32 import | 72 | 199 | 106 | 1241 | 11.7 | 105 | 1356 | 12.9 | = |
| bert_base_384 f32 import | 72 | 199 | 412 | 4846 | 11.8 | 458 | 3875 | 8.46 | = |

## CPU dispatch sites by op kind (NPU arm), from `count_sites.py` (totals match the placement report)

| model | on NPU | left on CPU |
|---|---|---|
| mobilenet_v2 | conv2d 35, depthwise 17, avgpool 1, matmul 1 | elementwise 4, transpose 1 |
| resnet50 / resnext50 | conv2d 53, maxpool 1, avgpool 1, matmul 1 | elementwise 18, elementwise_transpose 2, memcpy 1 |
| wide_resnet50_2 | conv2d 53, avgpool 1, matmul 1 | elementwise 5, transpose 2, memcpy 1, conv 1 |
| convnext_tiny | matmul 41, depthwise 18, avgpool 1 | elementwise 120, reduction 46, rsqrt 23, memcpy 17, transpose 10 |
| vit_b_16 | matmul 50 | elementwise 88, reduction 50, broadcast 37, rsqrt 25, batch_matmul 24, softmax 12, transpose 13 |
| vit_l_16 | matmul 98 | elementwise 172, reduction 98, broadcast 73, rsqrt 49, batch_matmul 48, softmax 24, transpose 25 |
| blip | matmul 170 | elementwise 214, reduction 126, transpose 73, batch_matmul 72, rsqrt 63, softmax 36, broadcast 36, matmul 1 (LM head, Cout 30524 > 4096) |
| bert_base(_384) | matmul 72 | elementwise 87, reduction 50, rsqrt 25, batch_matmul 24, softmax 12, broadcast 1 |

The only declined candidate is BLIP's 768x30524 LM head (`unvalidated_configuration`, Cout cap 4096).
