# IREE submodule bump: a869dc3c (3.12.0rc20260717+1) -> v3.14.0 (2026-11-25)

`iree-build/iree-src` moved from `a869dc3c` (2026-07-17) to the v3.14.0 tag
`5332c682` (2026-11-16, 303 commits ahead, a direct descendant). The gap is
3.13 plus 3.14: 25 LLVM integrates, 3 torch-mlir bumps and Tracy 0.15.

**Verdict: cheap to land, about 500 lines plus a re-vendor. One break was not
on the known-exposure list, and it is a 3.5x performance cliff that compiles
and runs correctly**, so a bump that only checked "tests pass" would have
shipped it. Every model is correct, and every site count is unchanged except
where 3.14 itself removed CPU glue. int8 is bit-identical to the old compiler.

## What broke, and the fix for each

| # | Break | Where it surfaced | Fix |
|---|---|---|---|
| 1 | HAL `iree_hal_command_buffer_vtable_t` gained `flush_buffer` / `invalidate_buffer` after `advise_buffer` | would have been a silent slot shift; here it was a **compile error**, because `VTABLE` is a named-field struct literal over bindgen output | Re-vendored all 20 changed headers (3.14 adds none to base/hal/async/io/schemas). Both slots are implemented as recorded ops applied in order, mirroring `iree_hal_inline_command_buffer_*`: map scoped, then flush (FINI_BO) or invalidate (PREP_BO). A flush counts as a dense read, so lazy compaction cannot elide the write it publishes. The IREE compiler never emits these ops today (only the HAL VM module exports them), and the 3.14 CTS does not test them. |
| 2 | Strict properties (PR 24983, plus upstream's transform dialect) | `RocketTransformOps.td` failed TableGen: "requires prop-dict ... missing attribute 'precision'" | `prop-dict attr-dict` in our op's format, so `match.admitted %x <{precision = "..."}>`. Rewrote the 16 spec lines, `rocket-compiler`'s `no_offload` injector (`spec.rs`) and its tests. Upstream's `transform.get_parent_op {deduplicate, op_name}` also went strict: 1 spec line. One test's `util.global {inlining_policy}` became `<inlining_policy = ...>`. |
| 3 | **LLVM removed every named elementwise linalg op** (`linalg.add/sub/mul/exp/...`, llvm #220905/#220912/#220916) | 3 C++ compile errors (`linalg::AddOp`); 3 spec matchers that keyed on `operation_name ["linalg.add"]` would have silently matched nothing | `linalg-specialize-generic-ops` now emits `linalg.elementwise <kind>`, and it also raises broadcasts, hoisted scalars and ternary `select`. New `transform.rocket.match.elementwise %x <{kind = "add"}>` (identity maps only, which is what the named ops implied). The C++ residual-add matchers use a helper that accepts only an identity-map `ElementwiseOp` of kind add. |
| 4 | **arith no longer folds `truncf(extf(x))` for f16 -> f32 -> f16.** APFloat's `isLosslesslyConvertibleTo` now refuses any IEEE -> IEEE conversion, because converting a signaling NaN quiets it. The fold survives only under `nnan`. | **Not a compile error.** MobileNetV2 fp16 compiled, ran and was correct, but CPU dispatch sites went 5 -> 55 and the NPU arm went 9.6 -> 33.7 ms. Every Rocket shim widens its f16 result with an `extf`, and the design relied on that cancelling against the consumer's `truncf` so that two Rocket dispatches end up adjacent and chain (ISSUES.md P2). | `rocket-fold-neutral-init` now does the cancellation itself (`CancelWidenNarrow`): a narrowing generic over a widening generic, optionally through one `expand_shape`/`collapse_shape`, back to the source type. The only value that changes is a signaling NaN, which keeps its payload, as every IREE before 3.14 did. 3 lit cases added. |
| 5 | Upstream `iree-global-opt-demote-contraction-inputs` was rewritten and **keeps `strides`** now | `rocket_verify_conv_shapes_upstream_demote.mlir`, a tripwire expecting the old stride-dropping bug, failed | Test flipped to assert stride 2 survives and the verifier accepts it. Our own `RocketDemoteConvInputsPass` workaround may now be redundant (follow-up). |

Also fixed in passing: 6 lit files in `rocket-compiler-plugin/test/` were
never listed in `test/CMakeLists.txt`, so ctest skipped them. Two of them
exercise the residual-add fusion that break 3 touched, so they are
registered now. `rocket-hal-driver/CMakeLists.txt` now also depends on the
vendored headers, so a re-vendor alone rebuilds the driver.

Not broken: `rocket-compiler/vendor/.../embedding_api.h` is byte-identical.
The plugin's IREE C++ surface (9 headers) only gained `explicit operator
bool` on `TargetRegistryRef`, which we do not use. The removed
`iree-global-opt-convert-batch-matmul-to-matmul` was never named by us. The
3.14 QDQ/im2col global-opt change does not reach claimed convs (int8 below).

## Gates

- **Lit:** 55/55, via ctest after registering the six; also `llvm-lit` over the whole directory.
- **Rust:** `cargo nextest run --workspace` 508/508 on host; `rocket-compiler` 51/51.
- **Board gate** (`cargo nextest run -p iree-rocket-hal --release --run-ignored all`, built natively on orangepi): 514/514. That crate links no IREE, so it is unaffected by construction. The kernel logged 8 "NPU job timed out" in the window of the `c8_precision_transition_hw` / `c16_pool_sequences` probes, which reproduce core poisoning on purpose.
- **HAL CTS on the board** (the real ABI check for the driver): core 28, buffer 20, command_buffer 37 and the conv schema harness 10 all pass. queue fails the same **8** it always has (seven explicit-pool `QueueAllocaTest` cases plus `QueueHostCallTest.AsyncCallback`; see `rocket-hal-driver/MULTICORE.md`). 0 timeouts.
- **Board runtime:** `configure-runtime-aarch64.sh` plus `iree-run-module`/`iree-benchmark-module` built clean (452 targets), with `IREE_HOST_BIN_DIR` pointed at the 3.14 host tools.

## 10-model survey vs `survey/2026-11-25-main` (orangepi, 600 MHz, performance)

Same harness, same inputs, MLIR and oracles (hardlinked), compile/run/bench
redone. 0 NPU timeouts. All 20 arms argmax == f32 oracle, and top-5
agreement is identical to 11-25 on every arm. The candidate, accepted and
NPU-site counts are **identical on all 10 models**.

| model | NPU 4xA76 ms | CPU 4xA76 ms | x | NPU 8c | CPU 8c | x | CPU sites (NPU arm) |
|---|---|---|---|---|---|---|---|
| mobilenet_v2 fp16 | 9.6 -> 10.3 | 54.3 -> 41.8 | 5.64 -> 4.06 | 9.8 -> 9.8 | 57.7 -> 43.7 | 5.91 -> 4.45 | 5 -> 4 |
| resnet50 fp16 | 52.0 -> 51.6 | 894 -> 818 | 17.2 -> 15.9 | 52.6 -> 53.6 | 1011 -> 852 | 19.2 -> 15.9 | 21 -> 21 |
| resnext50_32x4d fp16 | 55.6 -> 59.7 | 502 -> 455 | 9.03 -> 7.62 | 57.3 -> 58.1 | 685 -> 644 | 12.0 -> 11.1 | 21 -> 21 |
| convnext_tiny fp16 | 94.1 -> 90.2 | 252 -> 247 | 2.68 -> 2.74 | 98.1 -> 88.3 | 252 -> 244 | 2.57 -> 2.76 | 216 -> 137 |
| wide_resnet50_2 fp16 | 87.9 -> 86.1 | 1904 -> 2083 | 21.7 -> 24.2 | 87.6 -> 85.8 | 3012 -> 3566 | 34.4 -> 41.6 | 9 -> 9 |
| vit_b_16 fp16 | 202 -> 189 | 1664 -> 1667 | 8.24 -> 8.82 | 192 -> 178 | 1327 -> 1276 | 6.91 -> 7.17 | 249 -> 175 |
| vit_l_16 fp16 | 671 -> 641 | 11430 -> 11385 | 17.0 -> 17.8 | 686 -> 662 | 13871 -> 13576 | 20.2 -> 20.5 | 489 -> 343 |
| blip fp16 | 817 -> 779 | 4818 -> 4980 | 5.90 -> 6.39 | 740 -> 693 | 4005 -> 4187 | 5.41 -> 6.04 | 621 -> 445 |
| bert_base f32 import | 106 -> 103 | 1241 -> 1397 | 11.7 -> 13.6 | 105 -> 100 | 1356 -> 1432 | 12.9 -> 14.3 | 199 -> 125 |
| bert_base_384 f32 import | 412 -> 403 | 4846 -> 5248 | 11.8 -> 13.0 | 458 -> 444 | 3875 -> 4060 | 8.46 -> 9.14 | 199 -> 125 |

What moved, and why:

- **NPU arms: equal or a little faster everywhere** (transformers -3% to -7%), except resnext50 at 4xA76 (+7%, a wide [57-65] spread, one run; the 8-core arm is +1%).
- **Transformer CPU residue fell 30-37%.** In the NPU arm, 3.14 fuses LayerNorm's `rsqrt` and its elementwise tail into the reduction dispatches. Per kind on vit_b: elementwise -37, rsqrt -25, elementwise_transpose -11. This is IREE's own improvement.
- **The `--no-offload` baselines moved on their own**, which is most of the change in the ratios. CNN CPU arms got 9-23% faster even though mnv2/resnet gained ~33 standalone elementwise sites (their ReLU/ReLU6 clamps no longer fuse into the CPU conv dispatches). bert, bert_384, blip and wide_resnet's CPU arms got 3-13% slower. So mnv2's 5.6x -> 4.1x is a faster CPU, not a slower NPU.

## int8 (not in the survey): board set, old vs new compiler

`survey/boardset/{mnv2,resnet50-224}.int8.mlir`, each compiled with the
pinned compiler (11-25 toolchain) and with 3.14, then run on orangepi with
the matching runtime (4xA76, `iree-benchmark-module` median of 6).

| model | arm | NPU sites | CPU sites | output vs pinned compiler | ms pinned -> 3.14 |
|---|---|---|---|---|---|
| mnv2 int8 | npu | 54 = 54 | 90 -> 91 | **bit-identical** | 35.6 -> 30.5 |
| mnv2 int8 | cpu | 0 | 72 = 72 | bit-identical | 75.1 -> 51.0 |
| resnet50 int8 | npu | 56 = 56 | 59 -> 63 (+4 transposes) | **bit-identical** | 59.2 -> 48.9 |
| resnet50 int8 | cpu | 0 | 63 -> 66 | bit-identical | 462 -> 403 |

## Not covered

- **The importer.** The survey reused the 11-25 `.mlir` imports, so the 3 torch-mlir bumps were exercised for parsing and lowering, not for `onnx -> torch` import.
- **The CPU-arm ReLU/ReLU6 fusion loss** (the ~33 extra standalone clamp dispatches in the `--no-offload` CNN arms) is not diagnosed. It doesn't touch the NPU arms.
- **Tracy 0.15** was not built (tracing is off in these builds).

## Cost to land

- **Code:** about 500 lines (above), plus 20 re-vendored headers. Already done on this branch.
- **Build:** a cold LLVM/IREE compiler build is about 48 min at -j14 on laplace (16 cores). The board runtime is about 90 s.
- **Nightly / CI:** nothing else pins the old IREE. The nightly's `compiler-host` cache key follows the submodule, so the first nightly after merge is a cold build.
- **Follow-ups:** check whether `RocketDemoteConvInputsPass` is still needed now that upstream keeps strides; diagnose the CPU-arm clamp fusion loss; re-import one model end to end with the 3.14 importer.
