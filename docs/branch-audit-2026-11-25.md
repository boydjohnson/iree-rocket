# Local branch audit, 2026-11-25

Against `main` at `3e97ce8967`. 66 local branches stood against 7 live remote
ones; this records what each one held before 52 of them were deleted.

## Why the obvious tests cannot be used here

Every PR in this repository lands on `main` as a squash commit, so a
branch's commits are never ancestors of `main` and its blobs never appear
under its own hashes. Three tests therefore give confidently wrong answers,
and the first cleanup (2026-09-07) recorded all three:

* `git branch --merged` and `git merge-base --is-ancestor` miss every
  squashed branch.
* `git diff origin/main <branch>` counts the branch's *stale* version of
  every line `main` has rewritten since the squash as new content, because
  the diff runs from `main` to the branch. A fully merged branch that forked
  a few months ago therefore shows thousands of "unique" lines -- in this
  audit's first pass `c12-cbuf-split-hazard`, merged in September, scored
  9,058 of them, and `kernel-spike`, which really is unmerged, scored 848.
  The metric does not even rank correctly.
* `git merge-tree` conflicts on nearly everything, because `main` kept
  editing the same files after the squash landed.

**What works** is matching commit subjects, because a squash body lists the
original subjects one per `* ` line, and a single-commit PR carries its
subject in the squash subject. Two normalisations are required and both
matter: strip the leading `* `, and strip trailing ` (#N)` *repeatedly* --
`main` carries at least one subject re-merged as `... (#42) (#44)`, which a
single strip leaves unmatched and reports as unmerged work.

On this tree that method settled 52 of 55 candidates. The three it did not
settle are the three the September audit had already set aside, which is the
first independent reproduction of that result.

## Deleted: merged in substance, every commit subject present on `main`

Tips recorded here because the reflog expires. Any of these can be restored
with `git branch <name> <sha>` while the objects live, or read from the
squash commit on `main` afterwards.

| branch | tip | last commit |
|---|---|---|
| `c12-cbuf-split-hazard` | `c9c9ee7215` | 2026-09-08 |
| `c2-requant-tie-rule` | `a0dafc6e80` | 2026-09-07 |
| `c9-int8-large-kernel-retraction` | `43775bffd0` | 2026-09-07 |
| `claude/adoring-chaum-cae335` | `733bf30b02` | 2026-11-23 |
| `claude/orangepi-kernel-upgrade-d4a479` | `d495f6f026` | 2026-09-10 |
| `claude/roadmap-sdpa` | `5631a56805` | 2026-11-24 |
| `claude/softmax-lever-1` | `69de6cb26f` | 2026-11-24 |
| `claude/softmax-lever-2` | `aaaa787cfb` | 2026-11-24 |
| `claude/softmax-lever-3` | `16471346de` | 2026-11-24 |
| `claude/softmax-threaded-arm` | `2c51f88a0f` | 2026-11-24 |
| `claude/softmax-wall-time` | `442820914f` | 2026-11-24 |
| `clippy-hygiene` | `60a2104aba` | 2026-09-08 |
| `compiler-roadmap-4-5-compile-time-tiling` | `5a868883d3` | 2026-09-09 |
| `compiler-roadmap-section-6-layout` | `2dac323012` | 2026-09-09 |
| `conv-pad-folding` | `ca151d0780` | 2026-09-07 |
| `declared-layout-6-2` | `463a241257` | 2026-09-10 |
| `dynamic-shapes-doc` | `2c5b7734e9` | 2026-09-07 |
| `feature/int8-downsample` | `cef252e564` | 2026-11-23 |
| `feature/int8-stride2` | `e87b370131` | 2026-11-23 |
| `feature/open-questions` | `c53cd4862b` | 2026-11-23 |
| `feature/p3-input-sync` | `d3ba6445f3` | 2026-11-23 |
| `feature/p4-cbuf-reuse` | `91a98bc14a` | 2026-11-23 |
| `feature/stem-7x7-s2` | `9273e2a598` | 2026-11-23 |
| `int8-pooling-hang-issue` | `3b3d5b9d35` | 2026-09-10 |
| `layout-contract-6-1` | `7afcd24932` | 2026-09-10 |
| `model-survey-harness` | `3e1d9f1333` | 2026-09-09 |
| `p7-chain-remeasure` | `25e2d301b3` | 2026-09-08 |
| `p7-depthwise-clock-retest` | `15899b63c8` | 2026-09-09 |
| `perf/convnext-depthwise` | `abce0dc4a5` | 2026-11-24 |
| `perf/depthwise-pad-fold` | `70911a0d55` | 2026-11-24 |
| `perf/fp16-stem-7x7-s2` | `573bb7cf82` | 2026-11-24 |
| `perf/grouped-conv-block-diagonal` | `5573254d40` | 2026-11-24 |
| `perf/grouped-conv-hal` | `4e012b7b50` | 2026-11-24 |
| `perf/int8-cpu-gaps` | `83d8f9d897` | 2026-11-23 |
| `perf/int8-maxpool` | `ff45469fd4` | 2026-11-24 |
| `perf/int8-residual-add` | `d1e01cb240` | 2026-11-24 |
| `perf/int8-stage4-requant` | `b15ea5d126` | 2026-11-24 |
| `perf/int8-stem-requant` | `e594567b22` | 2026-11-24 |
| `perf/iree-buffer-cache` | `b4a4bd00fa` | 2026-11-23 |
| `perf/matmul-dense-input` | `b3a6e5de03` | 2026-11-24 |
| `perf/matmul-dense-output` | `43e60f2d40` | 2026-11-24 |
| `perf/mnv2-depthwise` | `14ea6a20ed` | 2026-11-24 |
| `perf/mnv2-residual-add` | `41b532456d` | 2026-11-24 |
| `perf/quiesce-dwell` | `e2a821a4fb` | 2026-11-24 |
| `perf/stem-pad1-relu6` | `14a152ea93` | 2026-11-24 |
| `planner-admission-gate` | `2001345bfb` | 2026-09-09 |
| `relu6-and-p7` | `04caa12c54` | 2026-09-07 |
| `relu6-f16-import-fusion` | `ae7b8dc552` | 2026-09-10 |
| `residual-add-fusion` | `d6ee309962` | 2026-09-08 |
| `roadmap-section4-record-correction` | `665d0f1545` | 2026-09-10 |
| `rocket-core-extraction` | `66b0062a47` | 2026-09-09 |
| `weights-packed-6-3` | `057dbb5af4` | 2026-09-10 |

## Kept: content `main` does not have

All three are also pushed to `origin` now, as `archive/kernel-spike`,
`archive/more-mobilenetv2-convs` and `archive/resnet-fixes`. Two of them had
never been pushed at all and one had its remote deleted, so until today the
only copy of this work was in one working tree on one machine.

### `feature/more-mobilenetv2-convs` (`55befbae7b`, tip literally "WIP")

The compiler pass on it was superseded -- `RocketFuseQLinearConvPass.cpp`
became `main`'s `RocketFuseInt8RequantEpiloguePass.cpp` -- and its
`conv_depthwise_fp16_exact_hw.rs` was broadened into
`conv_depthwise_two_byte_exact_hw.rs`. What is unique is the **vendor
fixture corpora**, and they are the reason this branch is worth keeping:

| file | size | what `main` has |
|---|---|---|
| `tests/fixtures/conv_vendor_fixtures_depthwise.json` | 729,662 B, 40 cases | same path, 293,288 B, 63 *different* cases |
| `tests/fixtures/conv_vendor_fixtures_depthwise_i8.json` | 338,789 B | absent |
| `tests/fixtures/conv_vendor_fixtures_spatial_mid.json` | 45,465 B | absent |
| `tests/fixtures/conv_vendor_fixtures_spatial_mid_i8.json` | 46,387 B | absent |
| `tests/conv_vendor_fixture_depthwise.rs` | | absent (`main` reads the depthwise corpus from `conv_vendor_fixture_wide.rs`) |
| `tools/npu_hang_survey.sh` | | absent |

The depthwise corpus is the significant one and the name collision hides it:
the branch's 40 cases are 48 to 384 channels at 34, 56, 70, 112 and 130
spatial, with **6 to 134 vendor programs each** -- the multi-tile CBUF-split
captures. `main`'s 63 cases are different shapes (14x14 at 64 channels and
the like) with 6 programs each. Checked across all 10 fixture files in
`main`: not one of the branch's 40 shapes appears anywhere, under any name.
These are the 40 fixtures the CBUF-split work was validated against.

### `kernel-spike` (`5f86818cf0`)

Genuinely unmerged, and **two of the September note's claims for it do not
hold and should not be repeated**:

* `rocket_fc_transform_spec.mlir` is *not* this branch's work. It exists at
  the branch's merge-base `0f28c621`, so `main` deleted it later; it is
  recoverable from `main`'s own history and is not a reason to keep anything.
* The `ROCKET_PP_*` environment variables do not exist on this branch at
  all, nor on `main`. Nothing matches `ROCKET_PP` on either side.

What is really unique:

* `tests/pooling_hw.rs` -- absent from `main`, which has `pooling_oracle_hw.rs`
  instead.
* `examples/dump_ppu_pointer.rs` -- absent.
* the `vendor-bypass-shapes` cargo feature (`iree-rocket-hal/Cargo.toml`,
  used by `pooling_hw.rs`).
* 59 non-trivial lines of `src/rocket/pooling.rs` that `main` lacks --
  stride-beyond-kernel rejection with its three tests
  (`rejects_horizontal_stride_beyond_kernel`,
  `rejects_vertical_stride_beyond_kernel`,
  `accepts_stride_equal_to_kernel`), a sub-atomic channel-count equivalence
  test, and the reciprocal average-pooling divisor
  `((1u64 << 16) / kernel_height)`. Note the direction: `main`'s
  `pooling.rs` is the *larger* file (1,663 lines against 1,258), so this is
  not a stale branch holding an old version -- it is a guard that never
  landed. The September note's "~266 lines" is an overcount.

### `feature/resnet-fixes` (`ef857ddd23`)

Merged except one file, `iree-build/configure-cts-aarch64.sh`, still absent
from `main`. Unchanged from the September finding.

## Not touched

`main`; the branches tracking live remotes (`claude/c15-remeasure-packer`,
checked out in another worktree and in use, `claude/iree-rocket-online-softmax-feb1ed`,
`claude/softmax-perf-gap`, `fix/int8-stem-hang`, `perf/int8-maxpool-pad`,
`perf/patchify-conv-as-matmul`); and `softmax-followup`,
`feature/cpu-profiling` and `section2-planner-gate`, which track
`origin/main` rather than a deleted branch and so were never in scope.

## The one thing worth doing next

Nothing here was salvaged *into* `main`, deliberately. The code on the
keepers does not stand alone -- `pooling_hw.rs` needs the cargo feature and
the `pooling.rs` guard, and the fixture harness needs its corpora -- so
copying files onto a fresh branch would have produced a tree that does not
build and a second copy of 1.1 MB of fixtures. Pushing the tips preserves
all of it exactly.

The item with real value not in `main` is the 40-shape depthwise corpus and
its `_i8` counterpart. Wiring those into `main`'s vendor-fixture test is a
piece of work, not a cleanup, and it needs the name collision resolved
first.
