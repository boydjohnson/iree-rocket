//! Pure convolution planning: descriptors, hardware limits, layout geometry,
//! CBUF partitioning and tile planning, with no register emission.
//!
//! Everything here was extracted from `iree-rocket-hal`'s `conv.rs` (see
//! COMPILER_ROADMAP.md section 1) so that the compiler and the runtime ask
//! the same planner the same question. The evidence comments travelled with
//! the code: every limit and every formula below is capture- or
//! board-derived, and each says which. The register program that consumes a
//! [`ConvPlan`] still lives in the HAL, which re-exports this module.
//!
//! The fallible entry points are [`Shape::try_with_precision`] and its
//! `try_with_*` builders, [`ConvPlan::try_new`] and
//! [`ConvPlan::try_with_cbuf_banks`]; each returns a [`PlanError`] whose
//! [`PlanErrorCode`] says whether the shape is malformed, the semantics are
//! unsupported, the hardware cannot hold it, or it merely has no capture
//! backing. The panicking constructors that the HAL and its tests grew up
//! with are kept as thin wrappers over those, with the same messages.

use crate::{
    error::{PlanError, PlanErrorCode, refuse_unless},
    policy,
};

/// `[kernel_height, kernel_width]`.
pub type Kernels = [usize; 2];

/// `[pad_top, pad_left]`.
///
/// The CNA has no trailing-padding registers. The output extent determines
/// the implied bottom and right padding.
pub type Padding = [usize; 2];

/// Height of the originally captured image, in rows.
pub const IMAGE_HEIGHT: u32 = 32;

/// Width of the originally captured image, in pixels.
pub const IMAGE_WIDTH: u32 = 32;

/// Default input channels: the C3 dense NHWC case of the original captures.
pub const INPUT_CHANNELS: u32 = 3;

/// Largest input channel count with capture backing, fp16.
///
/// Raised 512 -> 1344 (2026-09-03), on the same evidence and at the same time
/// as [`MAX_INT8_INPUT_CHANNELS`]. Board, `accumulator_size_e_probe` with
/// `ROCKET_ACC_PROBE_PRECISION=fp16`, `Dense` pattern, one shape per process,
/// **0 mismatches at every point**:
///
/// * k=1, 14x14 Cout 64: `Cin` 256, 512, 576, 640, 704, 768, 896, 960, 1024,
///   1152, 1280, 1344, 1536, **1792**, across one to five tiles.
/// * k=3, 28x28 Cout 64: `Cin` to **1152**, including the 1/11 split at 1152.
/// * Cout, 7x7 `Cin` 448: 528, 640, 768, 1024, 1344, 1792, **2048**, with the
///   split flat at 2d/10w throughout.
///
/// Vendor agreement above the old ceiling is
/// `tests/conv_vendor_fixture_wide.rs`, whose corpus is fp16-generated: 83
/// agree, 2 documented and hardware-validated divergences, one refusal edge.
///
/// **The earlier 960 attempt (2026-08-28) failed for a reason that no longer
/// holds.** It was reverted because `conv_vendor_fixture_channels_768.rs`
/// caught real ConvPlan/vendor divergence for dense shapes at `Cin`
/// 576/640/704/768 -- ConvPlan predicted 1/11 against the vendor's 6/6, 5/7,
/// 4/8, 4/8. The 2026-09-02 group-division fix
/// ([`MAX_UNDIVIDED_WEIGHT_BANKS`]) reproduces all four exactly; the only
/// residual in that corpus is `Cin` 704 at small `Cout`, which is
/// hardware-exact.
///
/// This bounds the *channel* rules only. Whether a given `(Cin, Cout,
/// kernel)` fits the twelve CBUF banks is a separate question, and one
/// [`ConvPlan`] answers on its own -- at k=3 it is the binding one well
/// before this, refusing `Cin >= 1216` outright.
///
/// **Raised 1344 -> 1792 on 2026-09-04, for the matmul lowering.** A matmul
/// reaches this hardware as a convolution of height *one* with `K` as `Cin`
/// (`fc::Shape`), and MobileNetV2's classifier is `[1,1792] x [1792,1001]`
/// -- so `K = 1792` was over the old ceiling, and the geometry that carries
/// it, a 1x1 spatial "image", is not one any conv sweep had run. Measured on
/// `planck` with `dtype_boundary_probe`, 23 points, **0 mismatches and 0
/// device timeouts**:
///
///   K at M=1, N=64: 512, 1024, 1344, 1792, 2048 -- under `Selectors` and
///     again under `Counting`, which makes every input lane contribute
///   N at M=1, K=1792: 64, 512, 1001, 1792, 2048
///   M at K=1792, N=64: 1, 2, 7, 16, 32 (widths at height one; the CBUF
///     split moves 7/5 -> 2/10 -> 4/8 across that range and every one is
///     exact)
///   the classifier itself, `M=1 K=1792 N=1001`, under both patterns and
///     again with the fp32 accumulator kept (`Fp16Accumulator`)
///
/// 1792 rather than the 2048 that was also measured, on this file's usual
/// principle: 1792 is what a real model needs and what the earlier 14x14
/// sweep already reached. `fc_matmul_geometry_matches_oracle` is the
/// regression, and `fc_matmul_ladder_matches_the_fc_lowering` is what keeps
/// its cases identical to what `fc::Shape::as_conv_shape` actually builds --
/// a ladder that only *resembled* the production lowering would be measuring
/// its own geometry.
///
/// **The other 2-byte rungs now have their own evidence at this value**
/// (2026-09-04), rather than only inheriting it: `bf16_regression_matrix`
/// (58/58) and `int16_regression_matrix` (37/37) both run `Cin` 512, 1024
/// and 1344 at k=1 and 512/1024 at k=3, `Cout` to 1792, the ragged 33/65/129
/// and 40/72/129, 56x56 and 112x112 multi-tile, 5x5 and 7x7, and stride 2.
/// `fp16_accumulator_matrix` (52/52) runs the same battery on the
/// fp32-result writer. So the sharing is measured at all four widths of the
/// family, not argued from the element width alone.
///
/// Shared between dense and depthwise `Shape` construction. That sharing is
/// what made the 960 attempt unsafe; it is not a problem here, because the
/// depthwise half now has its own corpus
/// (`conv_vendor_fixtures_depthwise.json`, 63/63 agreement to C=1344) and
/// because fp16 depthwise cannot reach a compiled dispatch at all -- the
/// demote pass deliberately excludes it (see
/// `RocketDemoteConvInputsPass.cpp`, reverted 2026-09-01).
///
/// **Raised 1792 -> 3584 on 2026-09-06**, for the transformer matmuls. A
/// ViT-B/16 MLP is `K = N = 3072` and its QKV projection is `N = 2304`;
/// both sat over the old ceiling, and Qwen3's MLP is 3072 as well. Measured
/// on `planck` from a quiet board with `dtype_boundary_probe`, one sweep
/// per process, **0 mismatches and 0 device timeouts** at every point:
///
/// * k=1, 14x14 Cout 64, `Cin` 1792, 2048, 2304, 2560, 2816, 3072, 3328,
///   3584 -- and on past the value taken here: 3840, 4096, 4608, 5120, 6144,
///   **8192**. Under `Selectors` and again under `Counting`, which makes
///   every one of those lanes contribute.
/// * ragged `Cin` 1793, 2049, 2313, 3073, 3585, 4095 under `Selectors`, and
///   the same six under `Counting` with the fp32 output container
///   (`fp16acc`) -- see the note below on why the fp16 one cannot carry
///   them.
/// * `Cout` at 7x7 `Cin` 448: 1792, 2048, 2304, 2560, 3072, 3584, 4096, and
///   ragged 1793, 2049, 2313, 3073, 3585, 4095. The CBUF split is flat at
///   2d/10w across all of it, as it was over the previous raise's range.
/// * the `onehot` read map at `Cout == Cin`, which is the instrument that
///   says *where* a value was read from rather than only whether the sum is
///   right: exact at 2313, 3072, 3584 and the ragged 3585 (14x14), at 2048
///   (56x56, 56 tiles), and at 3072/3584 on the height-one matmul geometry
///   `M = 197` where the row column-tiles.
/// * the ViT shapes themselves, `197x1`: `K` 3072 `N` 768, and `K` 768 `N`
///   2304 and 3072.
/// * multi-tile 56x56 at `Cin` 2048, 3072, 3584 (112 to 224 tiles).
/// * stride 2 at `Cin` 1792..4096 and `Cout` 2304..3584, which is also the
///   first measurement of a *strided column partition* -- see
///   `ConvPlan::new_with_cbuf_partition`.
///
/// **`Counting` cannot be read at fp16 above `Cin` 2048.** Its expected
/// output is the input-channel count itself, and fp16 spaces integers by
/// two from 2048 and by four from 4096, so `Cin` 2049 comes back 2048 --
/// `max|diff| = 1`, every pixel, which reads exactly like a real
/// channel-padding fault. The counts taken above are either representable
/// (every aligned one here is) or run on `fp16acc`, whose fp32 output
/// container holds them exactly. This is the same instrument trap as the
/// bf16 ladder's nine-significant-bit ceiling.
///
/// 3584 rather than the 4096 (and, at k=1 `Cin`, 8192) that also measured
/// clean, on this file's usual principle: 3584 covers every transformer
/// shape in the corpus with a rung of headroom. A ViT-L/16 MLP at 4096
/// would need the constant moved, not another measurement.
///
/// **Raised 3584 -> 4096 on 2026-09-09.** No new corpus was needed: the
/// 2026-09-06 sweep that set 3584 had already measured 4096 clean at every
/// rung, and LIMITS.md said so in as many words -- the constant sat lower
/// only on this repository's principle that a limit is what a real model
/// needs and the corpus reaches. A ViT-L/16 MLP at 4096 is that model.
/// Re-confirmed on `planck` from a quiet board before moving it: fp16 `Cin`
/// 3584 and 4096 at 14x14 `Cout` 64, fp16 `Cout` 3584 and 4096 at 7x7 `Cin`
/// 448, the same two points at int8 under `SelectorsAffine`, and fp16
/// `Cin` = `Cout` = 4096 at 14x14. 0 mismatches, 0 device timeouts.
pub const MAX_INPUT_CHANNELS: u32 = 4096;

/// `CNA_DATA_SIZE1.datain_channel_real` counts `Cin - 1` modulo this, even
/// though the field is 14 bits wide and could hold far more.
///
/// Confirmed in both precisions and only visible above 64 channels, which no
/// hardware test had reached: fp16 `Cin` 72 programs 7 and 80 programs 15;
/// int8 `Cin` 112 programs 47 and 128 programs 63.
pub const CHANNEL_REAL_MODULUS: u32 = 64;

/// Largest input-channel count the int8 sweep measures.
///
/// Raised 512 -> 1344 (2026-09-03) on hardware evidence, after the DPU output
/// writer fix removed the failure the old 512 was containing. Measured on
/// RK3588 with `accumulator_size_e_probe`, `Dense` pattern, one shape per
/// process, **every point 0 mismatches**:
///
/// * **k=1**, 14x14 Cout 64: `Cin` 512, 576, 640, 704, 768, 896, 1024, 1152,
///   1280, 1344, 1408, 1536, 1792, **2048** -- exact throughout, single- and
///   multi-tile.
/// * **k=3**, 28x28 Cout 64/448: `Cin` up to **1152**, including the 1/11
///   splits at 1088 and 1152. `ConvPlan` refuses `Cin >= 1216` at k=3 outright
///   (the coefficient working set exceeds the eleven grantable banks), so that
///   range is loud rather than silent.
/// * MobileNetV2's own widest dense 1x1 convolutions, at their real extents:
///   14x14 `Cin` 528->88/136 and 816->136; 7x7 816->224, 1344->224, 1344->448,
///   and 448->**1792**.
///
/// 1344 rather than 2048 because 1344 is what MobileNetV2 needs and what the
/// vendor corpus reaches; the k=1 points above it are measured but not
/// corpus-backed. This bounds the *channel padding* rules only -- whether a
/// given `(Cin, Cout, kernel)` fits the twelve CBUF banks stays `ConvPlan`'s
/// separate question, and at k=3 it is the binding one well before this.
///
/// **Raised 1344 -> 3584 on 2026-09-06**, with the rest of the rungs; see
/// [`MAX_INPUT_CHANNELS`] for the sweep. int8's own points: k=1 14x14 Cout
/// 64 at `Cin` 1792, 2304, 3072, 3584, 4096 under `SelectorsAffine` and
/// again under `Counting` (whose int8 output shift makes the count
/// readable), the `onehot` read map at `Cout == Cin` 3584, `Cout`
/// 2304..4096 at 7x7 `Cin` 448, and stride 2 at `Cin` 2304..4096.
///
/// **Raised 3584 -> 4096 on 2026-09-09.** No new corpus was needed: the
/// 2026-09-06 sweep that set 3584 had already measured 4096 clean at every
/// rung, and LIMITS.md said so in as many words -- the constant sat lower
/// only on this repository's principle that a limit is what a real model
/// needs and the corpus reaches. A ViT-L/16 MLP at 4096 is that model.
/// Re-confirmed on `planck` from a quiet board before moving it: fp16 `Cin`
/// 3584 and 4096 at 14x14 `Cout` 64, fp16 `Cout` 3584 and 4096 at 7x7 `Cin`
/// 448, the same two points at int8 under `SelectorsAffine`, and fp16
/// `Cin` = `Cout` = 4096 at 14x14. 0 mismatches, 0 device timeouts.
pub const MAX_INT8_INPUT_CHANNELS: u32 = 4096;

/// Largest output-channel count the int8 sweep measures.
///
/// Split from [`MAX_OUTPUT_CHANNELS`] (2026-09-03) rather than raising the
/// shared constant, mirroring [`MAX_INT8_INPUT_CHANNELS`] against
/// [`MAX_INPUT_CHANNELS`]: the hardware evidence below is int8 only, and fp16
/// has none above 768.
///
/// Measured exact at 7x7 `Cin` 448 with `Cout` 768, 1024, 1280, 1536, 1792 and
/// **2048** -- the CBUF split does not move across that range (7d/5w
/// throughout), which is consistent with `MAX_OUTPUT_CHANNELS`' own note that
/// the high-channel divergence is indexed by `Cin`, not `Cout`. Set at 1792,
/// MobileNetV2's widest, rather than the 2048 that was also measured.
///
/// **Raised 1792 -> 3584 on 2026-09-06**: exact at 7x7 `Cin` 448 for `Cout`
/// 2304, 3072, 3584 and 4096, split flat at 7d/5w throughout, and at
/// `Cout == Cin` 3584 under the `onehot` read map. Same sweep as
/// [`MAX_INPUT_CHANNELS`].
///
/// **Raised 3584 -> 4096 on 2026-09-09.** No new corpus was needed: the
/// 2026-09-06 sweep that set 3584 had already measured 4096 clean at every
/// rung, and LIMITS.md said so in as many words -- the constant sat lower
/// only on this repository's principle that a limit is what a real model
/// needs and the corpus reaches. A ViT-L/16 MLP at 4096 is that model.
/// Re-confirmed on `planck` from a quiet board before moving it: fp16 `Cin`
/// 3584 and 4096 at 14x14 `Cout` 64, fp16 `Cout` 3584 and 4096 at 7x7 `Cin`
/// 448, the same two points at int8 under `SelectorsAffine`, and fp16
/// `Cin` = `Cout` = 4096 at 14x14. 0 mismatches, 0 device timeouts.
pub const MAX_INT8_OUTPUT_CHANNELS: u32 = 4096;

/// Channel ceilings for int4, set to what the hardware ladder measures
/// rather than to what the arithmetic would allow.
///
/// int4 packs four times as densely as fp16, so nothing in the CBUF model
/// stops these from being much higher; they were low because the evidence
/// stopped there. Raise them with the measurement, not ahead of it.
///
/// Raised 512 -> 1344 / 512 -> 1792 on 2026-09-04, to where the 2-byte
/// rungs sit. `int4_regression_matrix_matches_oracle` is 51/51 on `planck`,
/// with these among them: `Cin` 512/1024/1344 at k=1 and 512/1024 at k=3
/// (`ConvPlan` refuses k=3 past 1152), `Cout` 512/1024/1792, the ragged 96
/// and 160 against int4's 64-channel output granule, 56x56 and 112x112
/// multi-tile, 5x5 and 7x7, and stride 2 at both k=1 and k=3.
///
/// `Cin` is always a whole 32-channel feature atom here -- `with_precision`
/// refuses a partial one -- so unlike the 2-byte rungs there is no ragged
/// input-channel case to bound.
///
/// **Raised 1344 -> 3584 / 1792 -> 3584 on 2026-09-06**, with the rest of
/// the rungs (see [`MAX_INPUT_CHANNELS`]): `Cin` 1792, 2304, 3072, 3584 and
/// 4096 at k=1 under both `Selectors` and `Counting`, and `Cout`
/// 2304..4096 at 7x7 `Cin` 448. 3584 is 112 whole feature atoms, so the
/// whole-atom rule below still admits the ceiling itself.
///
/// The `onehot` read map is the one instrument this rung cannot run: it
/// encodes each input's own NHWC index, which does not fit a nibble, and it
/// fails at `Cin` 64 exactly as it does at 3584. So int4's addressing
/// evidence up here is `Selectors` alone, one rung weaker than every other
/// datatype's.
///
/// **Raised 3584 -> 4096 on 2026-09-09.** No new corpus was needed: the
/// 2026-09-06 sweep that set 3584 had already measured 4096 clean at every
/// rung, and LIMITS.md said so in as many words -- the constant sat lower
/// only on this repository's principle that a limit is what a real model
/// needs and the corpus reaches. A ViT-L/16 MLP at 4096 is that model.
/// Re-confirmed on `planck` from a quiet board before moving it: fp16 `Cin`
/// 3584 and 4096 at 14x14 `Cout` 64, fp16 `Cout` 3584 and 4096 at 7x7 `Cin`
/// 448, the same two points at int8 under `SelectorsAffine`, and fp16
/// `Cin` = `Cout` = 4096 at 14x14. 0 mismatches, 0 device timeouts.
pub const MAX_INT4_INPUT_CHANNELS: u32 = 4096;

///
/// **Raised 3584 -> 4096 on 2026-09-09.** No new corpus was needed: the
/// 2026-09-06 sweep that set 3584 had already measured 4096 clean at every
/// rung, and LIMITS.md said so in as many words -- the constant sat lower
/// only on this repository's principle that a limit is what a real model
/// needs and the corpus reaches. A ViT-L/16 MLP at 4096 is that model.
/// Re-confirmed on `planck` from a quiet board before moving it: fp16 `Cin`
/// 3584 and 4096 at 14x14 `Cout` 64, fp16 `Cout` 3584 and 4096 at 7x7 `Cin`
/// 448, the same two points at int8 under `SelectorsAffine`, and fp16
/// `Cin` = `Cout` = 4096 at 14x14. 0 mismatches, 0 device timeouts.
pub const MAX_INT4_OUTPUT_CHANNELS: u32 = 4096;

/// Channel ceilings for tf32, again the extent of the measurement rather
/// than of the arithmetic. A 4-byte element charges four times fp16's CBUF
/// residency per channel, so these will always sit lower than the 2-byte
/// rungs' on the `Cin` side, where residency is what binds.
///
/// Raised 256 -> 1024 / 512 -> 1792 on 2026-09-04.
/// `tf32_regression_matrix_matches_oracle` is 50/50 on `planck`: `Cin` 512
/// and 1024 at k=1, 384/512/576 at k=3 (straddling the split change at 544,
/// where the plan goes 3/9 two tiles -> 1/11 fourteen), `Cout` 512/1024/1792,
/// the ragged 34/66/130 and 12/20/68, 56x56 and 112x112 multi-tile, 5x5 at
/// `Cin` 64 and 7x7 at 32, and stride 2.
///
/// `Cout` does not charge feature residency, which is why it reaches the
/// same 1792 as the 2-byte rungs while `Cin` stops at 1024 -- past that
/// `ConvPlan` refuses at k=3, and it is the coefficient working set rather
/// than this constant that binds there.
///
/// Two hardware faults were found and fixed getting here, both tf32-only
/// and both of them hangs rather than wrong data:
/// [`Precision::out_channel_granule`] (a padded `Cout` at 8 modulo 16) and
/// `streamed_weight_bank_preference_for_group`'s coefficient working set,
/// which was calibrated at two bytes and starved the 4-byte stream.
///
/// **Raised 1024 -> 3584 / 1792 -> 3584 on 2026-09-06.** The paragraph
/// above predicted this rung would stay lower than the 2-byte ones because
/// a 4-byte element charges four times the CBUF residency per channel --
/// that prediction is now measured wrong at k=1, where the residency is not
/// what binds: `Cin` 1792, 2304, 3072, 3584 and 4096 are exact at 14x14
/// Cout 64 under `Selectors` and `Counting`, the `onehot` read map is exact
/// at `Cout == Cin` 3584, and `Cout` 2304..4096 is exact at 7x7 `Cin` 448.
/// The tile count is roughly double the 2-byte rungs' at the same shape
/// (28 against 14 at `Cin` 3584), which is the residency showing up as
/// geometry rather than as a refusal. At k=3 it still binds, and there
/// `ConvPlan`'s own refusal is what governs -- see [`MAX_INPUT_CHANNELS`].
///
/// **Raised 3584 -> 4096 on 2026-09-09.** No new corpus was needed: the
/// 2026-09-06 sweep that set 3584 had already measured 4096 clean at every
/// rung, and LIMITS.md said so in as many words -- the constant sat lower
/// only on this repository's principle that a limit is what a real model
/// needs and the corpus reaches. A ViT-L/16 MLP at 4096 is that model.
/// Re-confirmed on `planck` from a quiet board before moving it: fp16 `Cin`
/// 3584 and 4096 at 14x14 `Cout` 64, fp16 `Cout` 3584 and 4096 at 7x7 `Cin`
/// 448, the same two points at int8 under `SelectorsAffine`, and fp16
/// `Cin` = `Cout` = 4096 at 14x14. 0 mismatches, 0 device timeouts.
pub const MAX_TF32_INPUT_CHANNELS: u32 = 4096;

///
/// **Raised 3584 -> 4096 on 2026-09-09.** No new corpus was needed: the
/// 2026-09-06 sweep that set 3584 had already measured 4096 clean at every
/// rung, and LIMITS.md said so in as many words -- the constant sat lower
/// only on this repository's principle that a limit is what a real model
/// needs and the corpus reaches. A ViT-L/16 MLP at 4096 is that model.
/// Re-confirmed on `planck` from a quiet board before moving it: fp16 `Cin`
/// 3584 and 4096 at 14x14 `Cout` 64, fp16 `Cout` 3584 and 4096 at 7x7 `Cin`
/// 448, the same two points at int8 under `SelectorsAffine`, and fp16
/// `Cin` = `Cout` = 4096 at 14x14. 0 mismatches, 0 device timeouts.
pub const MAX_TF32_OUTPUT_CHANNELS: u32 = 4096;

/// Widest input pixel the vendor keeps in dense NHWC, in bytes.
///
/// A C4 fp16 pixel is 8 bytes and stays dense; a C5 pixel is 10 bytes and
/// switches to NC1HWC2 surfaces. The boundary is half a 16-byte feature
/// atom, not a whole one -- `Cin` 5, 6 and 7 are already surfaces.
/// Most input channels the dense ARGB path carries.
///
/// This is a channel-count boundary, not a byte-width one. The fp16 rule
/// used to be written as "a pixel narrower than half a feature atom", which
/// comes out at four channels and is right -- but for the wrong reason: at
/// int8 the same byte test would put the boundary at eight, and it does not
/// move. `Cin` 4 programs the ARGB path in both precisions and `Cin` 8
/// programs surfaces in both. The real constraint is `ArgbInputMode`, which
/// enumerates one to four channels because it is an image-input path.
pub const MAX_DENSE_CHANNELS: u32 = 4;

/// Rounds a feature-atom count up to a whole group of four.
///
/// A count one short of a multiple of four takes the next multiple; every
/// other count passes through. So 3 becomes 4, 7 becomes 8, 11 becomes 12,
/// and 5, 6, 9, 10 are left alone.
///
/// This was a two-entry table for as long as the corpus stopped at `Cin` 80,
/// where only the 3- and 7-atom exceptions were reachable. Those two suggest
/// `2**n - 1`, which is wrong: the large-`Cin` sweep finds exceptions at 11,
/// 15, 19, 23, 27, 31, 35, 43, 55 and 63 atoms as well, and every one of
/// them -- like 3 and 7 -- is one short of a multiple of four.
///
/// Two different quantities round this way, and they are not the same
/// quantity:
///
/// - the fp16 *weight* channel padding ([`Shape::weight_channels`]), which
///   int8 does not share;
/// - the CBUF atom charge ([`Shape::cbuf_atoms`]), which both precisions do.
///
/// Fits all 66 fp16 and all 44 int8 channel counts measured from 3 to 512.
fn quad_atoms(atoms: u32) -> u32 {
    if atoms % 4 == 3 { atoms + 1 } else { atoms }
}

/// Output channels of the captured reference convolution.
pub const OUTPUT_CHANNELS: u32 = 8;

/// Largest output-channel count this builder will program, fp16.
///
/// `CNA_WEIGHT_SIZE2.weight_kernels` is 14 bits, so 16383 is the encodable
/// ceiling. This is set at the *measured* extent instead, on the same
/// principle as [`MAX_INPUT_CHANNELS`].
///
/// Raised 512 -> 768 (2026-09-01) on the expanded vendor corpus, then
/// 768 -> 1792 (2026-09-03) on hardware: 7x7 `Cin` 448 is exact at `Cout`
/// 528, 640, 768, 1024, 1344, 1792 and 2048, with the CBUF split flat at
/// 2d/10w across the whole range. The high-channel divergence this constant's
/// previous note worried about is indexed by `Cin`, not `Cout`, and that note
/// is superseded: it described the pre-2026-09-02 split model.
///
/// Depthwise constructs with `out_channels == in_channels`, so
/// [`MAX_INPUT_CHANNELS`] binds it rather than this.
///
/// **Raised 1792 -> 3584 on 2026-09-06** on the same sweep as
/// [`MAX_INPUT_CHANNELS`], whose doc comment carries the evidence. The
/// `Cout` half of it is the 7x7 `Cin` 448 ladder (1792..4096, aligned and
/// ragged), the `onehot` read map at `Cout == Cin` to 3585, and stride 2 at
/// `Cout` 2304..3584.
///
/// Depthwise does *not* follow it up: it constructs with
/// `out_channels == in_channels`, and [`MAX_DEPTHWISE_CHANNELS`] now holds
/// that path at the extent its own corpus reaches.
///
/// **Raised 3584 -> 4096 on 2026-09-09.** No new corpus was needed: the
/// 2026-09-06 sweep that set 3584 had already measured 4096 clean at every
/// rung, and LIMITS.md said so in as many words -- the constant sat lower
/// only on this repository's principle that a limit is what a real model
/// needs and the corpus reaches. A ViT-L/16 MLP at 4096 is that model.
/// Re-confirmed on `planck` from a quiet board before moving it: fp16 `Cin`
/// 3584 and 4096 at 14x14 `Cout` 64, fp16 `Cout` 3584 and 4096 at 7x7 `Cin`
/// 448, the same two points at int8 under `SelectorsAffine`, and fp16
/// `Cin` = `Cout` = 4096 at 14x14. 0 mismatches, 0 device timeouts.
pub const MAX_OUTPUT_CHANNELS: u32 = 4096;

/// Most channels a *depthwise* convolution will program, at any precision.
///
/// Split out on 2026-09-06, when the dense ceilings went to 3584. Until
/// then depthwise rode [`MAX_INPUT_CHANNELS`] -- it constructs with
/// `out_channels == in_channels`, so that constant bound it -- and letting
/// it keep riding would have extended the depthwise path by a factor of two
/// on no depthwise evidence at all. The dense sweep does not carry over:
/// depthwise has its own coefficient grouping (a 64-*byte* run), its own
/// 256-byte output write atom, and its own vendor corpus, and each of those
/// has been wrong at a shape the dense path was right at.
///
/// 1792 is where the shared constant already stood, so this is a freeze
/// rather than a claim. The depthwise evidence behind it stops earlier
/// still: the vendor corpus reaches C=1344, the hardware exactness tests
/// reach 1536, and the transform spec's depthwise matchers stop at 1344.
/// Raise it the same way as any other limit here -- a depthwise measurement
/// first, on `conv_depthwise_two_byte_exact_hw` or a probe of its own.
pub const MAX_DEPTHWISE_CHANNELS: u32 = 1792;

/// Physical width of one feature atom.
pub const FEATURE_ATOM_BYTES: u32 = 16;

/// Total CBUF banks the CNA partitions between feature data and weights.
pub const CBUF_BANKS: u32 = 12;

/// Bytes one CBUF bank holds: 256 entries of 128 bytes.
pub const CBUF_BANK_BYTES: u32 = 256 * 128;

/// Feature atoms the CBUF charges per `data_entries` entry.
///
/// The surface feature charge is counted in whole entries of four atoms, and
/// it rounds *up*: a row whose atom count is not a multiple of four still
/// occupies the whole final entry. `CNA_CBUF_CON1.data_entries` has always
/// been programmed that way (see its `div_ceil` below); the residency bound
/// in [`Shape::max_tile_input_rows_for_width_and_data_banks`] has to charge
/// the same way or it over-commits the CBUF.
pub const CBUF_ATOMS_PER_ENTRY: u32 = 4;

/// Minimum safe `weight_banks` once a coefficient footprint is being
/// starved (granted fewer banks than its own uncapped demand -- see
/// `demand_based_cbuf_partition`), as a function of `weight_channels`
/// (padded `Cin`). Five hardware points, all via
/// `iree-rocket-hal/tests/conv_cbuf_split_sweep_hw.rs::weight_bank_floor_probe*`,
/// each the exact boundary (one value below fails 0/5, the value at or
/// above passes 5/5):
///
///   Cin  256  320  384  448  512
///   min    3    4    5    5    5
///
/// A first guess (`floor = Cin/128 + 1`) fit the 256/512 endpoints exactly
/// and predicted 4 at 384; the real answer there is 5. The corrected
/// picture, checked against all five points: **linear at +1 per 64 `Cin`
/// from 256, plateauing at 5 from 384 on**. Nothing below Cin=256 has been
/// probed with an explicit low override -- every validated shape down there
/// (`features.0`'s Cin=3 included) has a real weight demand small enough
/// that the starved branch never fires for it in the first place, so `3` is
/// used unconditionally below 256 on the strength of the trend (floor rises
/// with `Cin`, never falls) rather than a direct measurement. See
/// DESIGN_NOTES.md "The floor is a slope, then a plateau" in
/// iree-rocket-design-spike.
pub fn weight_banks_floor(weight_channels: u32) -> u32 {
    if weight_channels <= 256 {
        3
    } else {
        (3 + (weight_channels - 256) / 64).min(5)
    }
}

/// Vendor-preferred coefficient grant for one streamed output group.
///
/// The expanded 28x28/K3 channel grid isolates this from spatial demand:
/// once the total coefficient tensor is too large to reside, both fp16 and
/// int8 reserve one 64-byte coefficient group per `(kernel tap, Cin)`.
/// Dividing that working set by a 32-KiB CBUF bank predicts every observed
/// high-channel split without depending on `Cout`:
///
/// `Cin=192,256,320,384,448,512 -> weight banks=4,5,6,7,8,9`.
///
/// This is a preferred allocation, not a new hardware-safety minimum; the
/// independently measured [`weight_banks_floor`] remains in force below it.
pub fn streamed_weight_bank_preference(
    weight_channels: u32,
    kernels: Kernels,
    element_bits: u32,
) -> u32 {
    let undivided =
        streamed_weight_bank_preference_for_group(weight_channels, kernels, 1, element_bits);
    if undivided <= MAX_UNDIVIDED_WEIGHT_BANKS {
        return undivided;
    }
    // Deliberately unclamped: a working set that still saturates after the
    // division is refused by `demand_based_cbuf_partition`, not quietly turned
    // into a one-data-bank split. 5x5 `Cin` 512 wants 13 banks even divided,
    // and no capture covers it.
    streamed_weight_bank_preference_for_group(weight_channels, kernels, 2, element_bits)
}

/// Largest coefficient grant the vendor will take without dividing the
/// streamed output-channel group -- i.e. it always leaves at least two banks
/// for feature data.
///
/// Measured, not chosen: the spatial corpus shows the group divided at every
/// `Cin` whose undivided grant reaches eleven (k=3 `Cin` >= 576) and undivided
/// at ten (5x5 `Cin` 192 keeps the vendor's 2/10).
pub const MAX_UNDIVIDED_WEIGHT_BANKS: u32 = CBUF_BANKS - 2;

/// Bytes one CBUF entry holds, and entries one bank holds. The multi-pass
/// correction below depends on the remainder within a bank, so the two have to
/// be visible separately even though their product is [`CBUF_BANK_BYTES`].
pub const CBUF_ENTRY_BYTES: u32 = 128;

pub const CBUF_ENTRIES_PER_BANK: u32 = CBUF_BANK_BYTES / CBUF_ENTRY_BYTES;

/// Coefficient grant when the streamed output-channel group is divided by
/// `group_divisor`.
///
/// `group_divisor == 1` reproduces the single-pass formula exactly: the
/// vendor's working set is `Cin * kh * g * kw_q` bytes for a group of `g`
/// output channels, which at the 64-bytes-per-tap calibration point is the
/// same product as `kh * kw * Cin * 64`.
///
/// Two details come from the vendor's routine (`librknnc.so`, file offset
/// 0x190ce70) rather than from fitting: a divided group is streamed in more
/// than one pass, which costs **one extra bank** unless the coefficient tail
/// divides a bank evenly, and never fewer than two.
///
/// The corpus never shows a divisor beyond two. An earlier attempt let the
/// search keep halving until the *feature map* fit, which drove 56x56 `Cin`
/// 640 to five banks against the vendor's seven and computed wrong values on
/// hardware; the division threshold is a property of the coefficient working
/// set alone, not of the spatial extent.
pub fn streamed_weight_bank_preference_for_group(
    weight_channels: u32,
    kernels: Kernels,
    group_divisor: u32,
    element_bits: u32,
) -> u32 {
    const STREAMED_BYTES_PER_INPUT_TAP: u32 = 64;
    // 64 bytes is what one streamed output-channel group costs per (tap,
    // Cin) at 16 bits and below -- the corpus measures the *same* 64 at
    // fp16 and int8, so the group is a fixed number of channels and 64
    // bytes is what they occupy at two bytes or fewer.
    //
    // At four bytes those same channels occupy twice as much, and taking
    // the calibrated 64 there does not merely mis-predict a split: it
    // grants a starved coefficient stream and **hangs the NPU**. Measured
    // on `planck` 2026-09-04, tf32 8x8 k=3 `Cout` 64: `Cin` 512 plans 3/9
    // and is exact, 576 through 896 plan 5/7 or 4/8 and every one is a
    // watchdog kill at ~500 ms. fp16 at the identical *coefficient
    // footprint* -- `Cin` 1152, the same 1327104 bytes -- plans 1/11 and is
    // exact, which is what says the fault is the grant rather than the
    // size. With the scaling below tf32 `Cin` 576 asks for 11 banks like
    // its fp16 twin.
    //
    // Deliberately `max(1, ..)` rather than a ratio: the corpus pins 1- and
    // 2-byte widths at 64 and this must not move them.
    let group_bytes_per_tap = STREAMED_BYTES_PER_INPUT_TAP * (element_bits / 16).max(1);
    let working_set = kernels[0] as u32 * kernels[1] as u32 * weight_channels * group_bytes_per_tap
        / group_divisor;
    let entries = working_set.div_ceil(CBUF_ENTRY_BYTES);
    let banks = entries.div_ceil(CBUF_ENTRIES_PER_BANK).max(1);
    if group_divisor == 1 {
        return banks;
    }
    if banks < 2 {
        return 2;
    }
    let remainder = entries % CBUF_ENTRIES_PER_BANK;
    if remainder != 0 && !CBUF_ENTRIES_PER_BANK.is_multiple_of(remainder) {
        banks + 1
    } else {
        banks
    }
}

/// Largest `CNA_CBUF_CON1.data_entries` value the expanded corpus shows the
/// hardware field can encode.
///
/// The generated register header exposes only bits 13:0, but both precisions
/// use bit 14: 128x128/Cin1 fp16 programs 16,384, and 64x400 reaches 25,600
/// in both fp16 and int8. Bit 15 remains unobserved.
pub const MAX_DATA_ENTRIES: u32 = 0x7fff;

/// Largest CBUF entry offset an input line's entry slab may begin at.
///
/// A surface-layout line is held in the CBUF slab-major: every pixel's
/// first entry (its first four feature atoms), then every pixel's second,
/// and so on, so slab `s` begins `s * in_cols` entries into the line. That
/// base is an 11-bit quantity. Past 2047 it wraps, and the slab is read
/// from the front of the line instead -- slab 0's pixels from `base - 2048`
/// on, then slab 1's -- which presents as the last 32 fp16 channels (16
/// tf32, 64 int8) of *every* pixel being wrong while everything below them
/// is exact, and as more slabs the further past the bound the line goes.
///
/// Measured on `planck` 2026-09-05 with `dtype_boundary_probe`, one shape
/// per process, and its `onehot` read map. The boundary sits at exactly
/// `(slabs - 1) * in_cols == 2048` at every depth tried on a 1x1 kernel --
/// fp16 K 96, 128, 256, 512, 768, 1024 and 1792; K 72 and 40, where a
/// partial slab counts as one; bf16 and int16 at K 768; tf32 at K 384,
/// where the same atom count is half the channels; the int8 accumulator
/// path at K 512 -- and K 32, a single slab, is exact at width 2000. The
/// wrap decodes exactly under the `onehot` map at K 256, 768 and 1024. It
/// is a property of the line, not the tile: 90x2 and 90x4 at K 768 fail
/// like 90x1, while 89x4 is exact. And it is not capacity: 90x1 K 768
/// fails at 9/3 and 6/6 as it does at 5/7, while 88x1 loses rows at 4/8.
/// The vendor FC corpus never reaches it because its widest shape is 32.
///
/// A 3x3 kernel at height one fails *earlier* than this bound (86x1 at K 768
/// against 89x1 for 1x1), but that is a different fault with the same
/// signature: the coefficient floor leaves it 4 data banks, its one line
/// does not fit them, and the capacity formula used to force a row through
/// anyway. That is fixed alongside this; see
/// [`Shape::max_tile_input_rows_for_width_and_data_banks`]. With both in
/// place every measured shape is exact: 1x1 at M 90, 128, 197 and 296, and
/// 3x3 at 86..89x1 K 768, 100..134x1 K 512, 62..65x1 K 1024, all as column
/// tiles. See ISSUES.md C10.
pub const MAX_ENTRY_SLAB_BASE: u32 = 0x7ff;

/// Largest logical value encodable by the 10-bit
/// `CNA_CONV_CON2.feature_grains` field.
pub const MAX_FEATURE_GRAINS: u32 = 0x03ff;

/// Feature pixels one CBUF bank holds: 256 entries of 128 bytes, at one
/// 16-byte feature atom per pixel, is 2048 -- but the vendor allocates in
/// half-bank steps, which is the 1024 below.
pub const PIXELS_PER_BANK_STEP: u32 = 1024;

/// Element precision of a convolution.
///
/// The int8 side is derived from a corpus deliberately built to mirror the
/// fp16 geometries, so every int8 capture is one half of a two-point diff in
/// which precision is the only thing that moved. Across 21 such pairs
/// exactly 33 register fields differ.
///
/// The datatype menu beyond fp16/int8 is a **3-bit precision field**, set
/// independently for the CNA/CORE input and processing stages and for the
/// DPU output stage, rather than a separate datapath. Adding a rung is
/// therefore "pick the field value, then get the element width's layout
/// right"; every layout rule in this module keys off the element *width*,
/// not the numeric interpretation, which is why the 2-byte rungs share the
/// fp16 geometry exactly. The field values are
/// `int8 = 0, int16 = 1, fp16 = 2, bf16 = 3, int32 = 4, fp32 = 5, int4 = 6`
/// (`tf32 = 7`, CNA/CORE only), each established by hardware sweep; see
/// `../rockchip-npu-notes/encodings/precision-field.md`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    Fp16,
    /// bfloat16: the same 2-byte operand and MAC rate as fp16 with fp32
    /// dynamic range, at the cost of three mantissa bits. Byte-for-byte the
    /// fp16 geometry -- feature atom of 8 channels, 16-kernel weight group,
    /// the same coefficient padding -- so the only thing that moves against
    /// [`Precision::Fp16`] is the precision field, 2 -> 3, in the CNA, CORE
    /// and DPU stages.
    Bf16,
    /// Signed 16-bit integer inputs and coefficients. Also a 2-byte element
    /// on the fp16 geometry, precision field 1.
    ///
    /// The compute side is sound, but the notes' matmul work found no
    /// full-iteration integer *output* writer for int16
    /// (`encodings/output-transpose-int16.md`), so this variant exists to be
    /// characterized on hardware before anything depends on it.
    Int16,
    /// fp16 inputs and coefficients with the **fp32 accumulator** written
    /// to memory instead of narrowed back to fp16.
    ///
    /// Byte-for-byte the [`Precision::Fp16`] program on the input side --
    /// same 2-byte element, same feature atom, same coefficient padding --
    /// with the DPU's output stage switched from the fp16 narrowing writer
    /// to the fp32 one: `out_precision = 5`, `size_e = 3`, a 4x surface
    /// multiplier, and `fp32tofp16_en` cleared. That last bit is what
    /// actually does the narrowing on the ordinary fp16 path, so leaving it
    /// set with an fp32 output precision would be self-contradictory.
    ///
    /// What it buys is the output rounding, not the arithmetic: an fp16
    /// convolution already accumulates in fp32 and only loses precision on
    /// the way out. It costs twice the output bandwidth and twice the
    /// result allocation.
    ///
    /// The 4-byte result means a 4-lane output cube (`C2 = 16 bytes / 4`),
    /// the same cube int8's int32 accumulator and tf32's fp32 result write;
    /// the *input* cube is unchanged at 8 fp16 channels.
    Fp16Accumulator,
    /// tf32: an fp32 container holding a 10-bit mantissa with fp32 range,
    /// accumulating into fp32.
    ///
    /// The only 4-byte input rung, and the only one whose precision field
    /// is not the same at every stage: 7 reaches the CNA and CORE, while
    /// the DPU has no tf32 code and runs at fp32 (5). Its geometry follows
    /// the width like every other rung -- a 16-byte feature atom holds four
    /// tf32 channels -- with one exception that does not, the coefficient
    /// input group, which halves to 16 channels
    /// (`iree_rocket_hal::rocket::tensor_layout::TF32_WEIGHT_INPUT_GROUP_CHANNELS`).
    ///
    /// Half the MAC rate of fp16/bf16, so this buys precision-with-range
    /// rather than speed: fp16's mantissa with fp32's exponent.
    ///
    Tf32,
    /// int4 inputs and coefficients with an int16 result.
    ///
    /// The half-byte element is the reason this module measures element
    /// *bits*: at 4 bits the 16-byte feature atom holds 32 channels and the
    /// 32-byte coefficient atom holds 64 kernels, both of which fall out of
    /// the shared atom widths rather than needing a table.
    ///
    /// Products of two int4 values reach 64 and accumulate in int16, which
    /// is the pairing `../rockchip-npu-notes/datatypes.md` records
    /// (`int4 -> int16`, as against `int8 -> int32` and `fp16 -> fp32`).
    /// There is no requantization on this path: the DPU writes the
    /// accumulator, so a shape whose accumulator can leave int16 is the
    /// caller's problem.
    Int4,
    /// Quantized int8, carrying the parameters that are not derivable from
    /// the shape and must come from the compiler.
    Int8(Quantization),
    /// Signed int8 inputs and coefficients with the exact signed int32 MAC
    /// accumulator written to memory. This keeps the validated int8 compute
    /// configuration but bypasses the DPU's requantization stages.
    Int8Accumulator(Quantization),
}

impl Precision {
    /// Bytes one input-feature or coefficient element occupies.
    ///
    /// Accumulator output does not change the int8 input/weight packing; use
    /// [`Precision::output_element_bytes`] when sizing the result tensor.
    /// Bytes one input-feature or coefficient element occupies.
    ///
    /// Accumulator output does not change the int8 input/weight packing; use
    /// [`Precision::output_element_bytes`] when sizing the result tensor.
    pub fn element_bytes(&self) -> u32 {
        let bits = self.element_bits();
        assert!(
            bits.is_multiple_of(8),
            "{self:?} elements are sub-byte; use element_bits or bytes_for"
        );
        bits / 8
    }

    /// Bits one input-feature or coefficient element occupies.
    ///
    /// int4 is why the width is measured in bits: every atom, padding and
    /// footprint rule in this module is `atom bytes * 8 / element bits`, and
    /// stating it in bytes cannot express a half.
    /// Bits one input-feature or coefficient element occupies.
    ///
    /// int4 is why the width is measured in bits: every atom, padding and
    /// footprint rule in this module is `atom bytes * 8 / element bits`, and
    /// stating it in bytes cannot express a half.
    pub fn element_bits(&self) -> u32 {
        match self {
            Precision::Tf32 => 32,
            Precision::Fp16 | Precision::Fp16Accumulator | Precision::Bf16 | Precision::Int16 => 16,
            Precision::Int8(_) | Precision::Int8Accumulator(_) => 8,
            Precision::Int4 => 4,
        }
    }

    /// Bytes `elements` input-feature or coefficient elements occupy.
    ///
    /// Panics on a count that would end mid-byte. Every quantity this is
    /// asked for is a padded whole atom, so a half byte here means a padding
    /// rule went wrong rather than that a caller wanted a ragged buffer.
    /// Bytes `elements` input-feature or coefficient elements occupy.
    ///
    /// Panics on a count that would end mid-byte. Every quantity this is
    /// asked for is a padded whole atom, so a half byte here means a padding
    /// rule went wrong rather than that a caller wanted a ragged buffer.
    pub fn bytes_for(&self, elements: u32) -> u32 {
        let bits = elements * self.element_bits();
        assert!(
            bits.is_multiple_of(8),
            "{elements} {self:?} elements are {bits} bits, not a whole number of bytes"
        );
        bits / 8
    }

    /// Whether this precision shares the fp16 *layout* family.
    ///
    /// Every coefficient-padding, weight-group and CBUF rule in this module
    /// follows the element width rather than the numeric interpretation, so
    /// bf16 and int16 inherit the fp16 geometry exactly rather than needing
    /// their own corpus. What they do not inherit is *evidence*: only fp16
    /// has vendor captures, so anything gated on capture backing says so in
    /// its own comment.
    /// Whether this precision shares the fp16 *layout* family.
    ///
    /// Every coefficient-padding, weight-group and CBUF rule in this module
    /// follows the element width rather than the numeric interpretation, so
    /// bf16 and int16 inherit the fp16 geometry exactly rather than needing
    /// their own corpus. What they do not inherit is *evidence*: only fp16
    /// has vendor captures, so anything gated on capture backing says so in
    /// its own comment.
    pub fn shares_fp16_layout(&self) -> bool {
        self.element_bits() == 16
    }

    /// Bytes one logical output element occupies.
    /// Bytes one logical output element occupies.
    pub fn output_element_bytes(&self) -> u32 {
        match self {
            // int4 accumulates into int16, so its result is wider than its
            // operands -- the one rung where the two differ by more than a
            // requantization.
            Precision::Fp16 | Precision::Bf16 | Precision::Int16 | Precision::Int4 => 2,
            Precision::Int8(_) => 1,
            Precision::Int8Accumulator(_) | Precision::Tf32 | Precision::Fp16Accumulator => 4,
        }
    }

    /// Whether this mode writes the exact int32 convolution accumulator.
    ///
    /// Deliberately int8-only: it gates the *integer* accumulator writer --
    /// `size_e = 7`, the 8x surface multiplier, the staged contiguous-tile
    /// output placement and the BS/CPEND bypasses. The fp32-result rungs
    /// keep the float writer and are asked about with
    /// [`Precision::writes_fp32_result`] instead.
    /// Whether this mode writes the exact int32 convolution accumulator.
    ///
    /// Deliberately int8-only: it gates the *integer* accumulator writer --
    /// `size_e = 7`, the 8x surface multiplier, the staged contiguous-tile
    /// output placement and the BS/CPEND bypasses. The fp32-result rungs
    /// keep the float writer and are asked about with
    /// [`Precision::writes_fp32_result`] instead.
    pub fn writes_accumulators(&self) -> bool {
        matches!(self, Precision::Int8Accumulator(_))
    }

    /// Whether the DPU writes a 4-byte fp32 result.
    ///
    /// Two rungs do: tf32, whose accumulator has no narrower container, and
    /// fp16 with its accumulator kept. Both take the float writer's natural
    /// geometry for a 4-byte element -- `size_e = 3` and a 4x surface
    /// multiplier -- which is what the notes' proven fp32-out matmul writer
    /// uses, and not the integer path's `size_e = 7` / 8x.
    /// Whether the DPU writes a 4-byte fp32 result.
    ///
    /// Two rungs do: tf32, whose accumulator has no narrower container, and
    /// fp16 with its accumulator kept. Both take the float writer's natural
    /// geometry for a 4-byte element -- `size_e = 3` and a 4x surface
    /// multiplier -- which is what the notes' proven fp32-out matmul writer
    /// uses, and not the integer path's `size_e = 7` / 8x.
    pub fn writes_fp32_result(&self) -> bool {
        matches!(self, Precision::Tf32 | Precision::Fp16Accumulator)
    }

    /// Channels one 16-byte feature atom carries.
    /// Channels one 16-byte feature atom carries.
    pub fn channels_per_atom(&self) -> u32 {
        FEATURE_ATOM_BYTES * 8 / self.element_bits()
    }

    /// Granularity the DPU's output-channel count rounds up to.
    ///
    /// Four registers -- `CORE_DATAOUT_SIZE_1.dataout_channel`,
    /// `DPU_DATA_CUBE_CHANNEL.channel`,
    /// `DPU_RDMA_RDMA_DATA_CUBE_CHANNEL.channel` and
    /// `DPU_WDMA_SIZE_0.channel_wdma` -- carry the padded count while
    /// `weight_kernels` and `orig_channel` carry the true one.
    ///
    /// Twice the atom width in both captured precisions: 16 for fp16 and 32
    /// for int8. A clean rule with no table and no exceptions in either --
    /// verified at every fp16 Cout in the corpus, including the awkward 20,
    /// 24, 28, 40, 56 and 72 where the *input* padding needed special cases,
    /// and at 10 int8 values from 8 to 112.
    ///
    /// **tf32 is the one rung where twice the atom width is not enough, and
    /// getting it wrong hangs the NPU rather than returning wrong data.** A
    /// 4-byte element makes `channels_per_atom` 4, so the rule above gives
    /// 8 -- the only value in the menu that is not a multiple of 16. Every
    /// tf32 shape whose padded `Cout` lands at 8 modulo 16 wedges the core
    /// until the watchdog kills the job at ~500 ms; every one at 0 modulo 16
    /// is exact. Measured on `planck` 2026-09-04 with `dtype_boundary_probe`,
    /// three geometries and 23 points:
    ///
    ///   16x16 k3: hangs at Cout 8, 20, 24, 36, 40; exact at 12, 16, 28,
    ///     32, 44
    ///   7x7 k1: hangs at 8, 20, 24, 40; exact at 16, 32
    ///   8x8 k1: hangs at 8, 20, 24, 40; exact at 48 -- three runs, identical
    ///
    /// Padded, those hanging counts are 8, 24, 24, 40, 40 and the exact ones
    /// 16, 16, 32, 32, 48: every hang is `8 (mod 16)`, every pass `0`.
    ///
    /// Padding those same shapes to 16 instead clears every one of them --
    /// all 15 re-measured cases pass. The mechanism is not established
    /// beyond the modulus, and no other rung can reach the condition,
    /// because every other granule in the menu -- 16 at
    /// fp16/bf16/int16/fp16-f32out, 32 at int8, 64 at int4 -- is already a
    /// multiple of 16.
    /// Granularity the DPU's output-channel count rounds up to.
    ///
    /// Four registers -- `CORE_DATAOUT_SIZE_1.dataout_channel`,
    /// `DPU_DATA_CUBE_CHANNEL.channel`,
    /// `DPU_RDMA_RDMA_DATA_CUBE_CHANNEL.channel` and
    /// `DPU_WDMA_SIZE_0.channel_wdma` -- carry the padded count while
    /// `weight_kernels` and `orig_channel` carry the true one.
    ///
    /// Twice the atom width in both captured precisions: 16 for fp16 and 32
    /// for int8. A clean rule with no table and no exceptions in either --
    /// verified at every fp16 Cout in the corpus, including the awkward 20,
    /// 24, 28, 40, 56 and 72 where the *input* padding needed special cases,
    /// and at 10 int8 values from 8 to 112.
    ///
    /// **tf32 is the one rung where twice the atom width is not enough, and
    /// getting it wrong hangs the NPU rather than returning wrong data.** A
    /// 4-byte element makes `channels_per_atom` 4, so the rule above gives
    /// 8 -- the only value in the menu that is not a multiple of 16. Every
    /// tf32 shape whose padded `Cout` lands at 8 modulo 16 wedges the core
    /// until the watchdog kills the job at ~500 ms; every one at 0 modulo 16
    /// is exact. Measured on `planck` 2026-09-04 with `dtype_boundary_probe`,
    /// three geometries and 23 points:
    ///
    ///   16x16 k3: hangs at Cout 8, 20, 24, 36, 40; exact at 12, 16, 28,
    ///     32, 44
    ///   7x7 k1: hangs at 8, 20, 24, 40; exact at 16, 32
    ///   8x8 k1: hangs at 8, 20, 24, 40; exact at 48 -- three runs, identical
    ///
    /// Padded, those hanging counts are 8, 24, 24, 40, 40 and the exact ones
    /// 16, 16, 32, 32, 48: every hang is `8 (mod 16)`, every pass `0`.
    ///
    /// Padding those same shapes to 16 instead clears every one of them --
    /// all 15 re-measured cases pass. The mechanism is not established
    /// beyond the modulus, and no other rung can reach the condition,
    /// because every other granule in the menu -- 16 at
    /// fp16/bf16/int16/fp16-f32out, 32 at int8, 64 at int4 -- is already a
    /// multiple of 16.
    pub fn out_channel_granule(&self) -> u32 {
        match self {
            Precision::Tf32 => 16,
            _ => 2 * self.channels_per_atom(),
        }
    }

    /// Bytes in the widest, four-channel dense ARGB storage class: 8 at fp16
    /// and 4 at int8. CBUF planning uses the shape-specific 1/2/4/4-channel
    /// charge in [`Shape::dense_cbuf_pixel_bytes`] instead.
    /// Bytes in the widest, four-channel dense ARGB storage class: 8 at fp16
    /// and 4 at int8. CBUF planning uses the shape-specific 1/2/4/4-channel
    /// charge in [`Shape::dense_cbuf_pixel_bytes`] instead.
    pub fn dense_pixel_bytes(&self) -> u32 {
        4 * self.element_bytes()
    }

    /// Most input channels this builder will program.
    ///
    /// fp16 stops at 80, where `CHANNEL_PADDING` runs out of measured rows.
    /// int8 reaches 128, where its sweep stops -- the padding there is a
    /// rule rather than a table, so the limit is the extent of the evidence
    /// rather than the extent of the arithmetic.
    /// Most input channels this builder will program.
    ///
    /// fp16 stops at 80, where `CHANNEL_PADDING` runs out of measured rows.
    /// int8 reaches 128, where its sweep stops -- the padding there is a
    /// rule rather than a table, so the limit is the extent of the evidence
    /// rather than the extent of the arithmetic.
    pub fn max_in_channels(&self) -> u32 {
        match self {
            Precision::Fp16 | Precision::Fp16Accumulator | Precision::Bf16 | Precision::Int16 => {
                MAX_INPUT_CHANNELS
            }
            Precision::Int8(_) | Precision::Int8Accumulator(_) => MAX_INT8_INPUT_CHANNELS,
            Precision::Int4 => MAX_INT4_INPUT_CHANNELS,
            Precision::Tf32 => MAX_TF32_INPUT_CHANNELS,
        }
    }

    /// Largest output-channel count this precision has capture or hardware
    /// backing for. The int8 side reaches further; see
    /// [`MAX_INT8_OUTPUT_CHANNELS`].
    /// Largest output-channel count this precision has capture or hardware
    /// backing for. The int8 side reaches further; see
    /// [`MAX_INT8_OUTPUT_CHANNELS`].
    pub fn max_out_channels(&self) -> u32 {
        match self {
            Precision::Fp16 | Precision::Fp16Accumulator | Precision::Bf16 | Precision::Int16 => {
                MAX_OUTPUT_CHANNELS
            }
            Precision::Int8(_) | Precision::Int8Accumulator(_) => MAX_INT8_OUTPUT_CHANNELS,
            Precision::Int4 => MAX_INT4_OUTPUT_CHANNELS,
            Precision::Tf32 => MAX_TF32_OUTPUT_CHANNELS,
        }
    }

    /// Calibration parameters, for the precisions that requantize. `None`
    /// is also the "unquantized rung" predicate the register program keys
    /// its BS/CVT bypasses off.
    /// Calibration parameters, for the precisions that requantize. `None`
    /// is also the "unquantized rung" predicate the register program keys
    /// its BS/CVT bypasses off.
    pub fn quantization(&self) -> Option<Quantization> {
        match self {
            Precision::Fp16
            | Precision::Fp16Accumulator
            | Precision::Bf16
            | Precision::Int16
            | Precision::Int4
            | Precision::Tf32 => None,
            Precision::Int8(quantization) | Precision::Int8Accumulator(quantization) => {
                Some(*quantization)
            }
        }
    }
}

/// Quantization parameters for an int8 convolution.
///
/// None of these are derivable from the shape -- they come from calibration,
/// so the compiler supplies them. What *is* derivable is how they are
/// encoded, which is what [`Multiplier`] captures.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quantization {
    /// Quantized encoding of 0.0 on the input, programmed into
    /// `CNA_PAD_CON1.pad_value`.
    ///
    /// This is the value an out-of-image tap contributes, and it is *not*
    /// zero. Every fp16 capture pads with 0 and every int8 capture pads with
    /// the input zero point; carrying the fp16 constant across would leave
    /// interior pixels correct and every pixel touching an image edge wrong
    /// by a constant.
    pub input_zero_point: i32,
    /// Quantized encoding of 0.0 on the output, programmed into
    /// `DPU_OUT_CVT_OFFSET`.
    pub output_zero_point: i32,
    pub weight_zero_point: i32,
    /// Real-valued input and weight calibration scales used to normalize bias.
    pub input_scale: f32,
    pub weights_scale: f32,
    /// Requantization multiplier, `input_scale * weight_scale / output_scale`.
    pub multiplier: Multiplier,
}

// Calibration data is validated as finite before it reaches the hardware
// path. Keep the historical `Eq` bound on `Precision`/`Shape` while storing
// the schema's native f32 values here.
impl Eq for Quantization {}

/// A requantization multiplier in the hardware's normalized fixed-point form.
///
/// `DPU_OUT_CVT_SCALE` holds a mantissa and `DPU_OUT_CVT_SHIFT` its negative
/// exponent, so the real multiplier is `scale / 2^shift`. Every one of the 21
/// int8 captures has its mantissa inside `[2^14, 2^15)` -- the shift is
/// chosen to normalize it there, which is what makes the pair recoverable
/// from a single real number.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Multiplier {
    pub scale: u32,
    pub shift: u32,
}

/// Lowest mantissa the normalized form uses; `2 * MANTISSA_FLOOR` is the
/// exclusive upper bound.
pub const MANTISSA_FLOOR: u32 = 1 << 14;

/// Largest shift `DPU_OUT_CVT_SHIFT` encodes. The field is 6 bits; the
/// corpus only reaches 26.
pub const MAX_CVT_SHIFT: u32 = 63;

impl Multiplier {
    /// Encodes a real multiplier, normalizing the mantissa into
    /// `[2^14, 2^15)`.
    ///
    /// Panics rather than saturating on a multiplier the form cannot carry:
    /// a silently clamped requantization scale is a whole-tensor error that
    /// would be very hard to attribute later. Callers that receive a ratio
    /// from outside the process -- a compiler-produced executable, a
    /// dispatch's push constants -- want [`Multiplier::try_from_ratio`]
    /// instead, so a bad one becomes a rejected dispatch rather than an
    /// unwind across an `extern "C"` boundary.
    pub fn from_ratio(ratio: f64) -> Multiplier {
        match Multiplier::try_from_ratio(ratio) {
            Ok(multiplier) => multiplier,
            Err(reason) => panic!("{reason}: {ratio}"),
        }
    }

    /// [`Multiplier::from_ratio`]'s fallible form.
    pub fn try_from_ratio(ratio: f64) -> Result<Multiplier, &'static str> {
        if !ratio.is_finite() || ratio <= 0.0 {
            return Err("requantization multiplier must be finite and positive");
        }
        // `scaled` is `ratio * 2^shift` throughout, driven into the mantissa
        // range. The exponent is signed while it is being searched for: a
        // multiplier above 1 normalizes to a shift below 14, and only the
        // final value has to be a nonnegative field.
        let mut shift: i32 = 14;
        let mut scaled = ratio * f64::from(MANTISSA_FLOOR);
        while scaled < f64::from(MANTISSA_FLOOR) {
            scaled *= 2.0;
            shift += 1;
            if shift > MAX_CVT_SHIFT as i32 {
                return Err("requantization multiplier is too small to encode; \
                     DPU_OUT_CVT_SHIFT tops out at 63");
            }
        }
        while scaled >= f64::from(2 * MANTISSA_FLOOR) {
            scaled /= 2.0;
            shift -= 1;
        }
        // Rounding can carry the mantissa back out of range at the top.
        let mut scale = scaled.round() as u32;
        if scale >= 2 * MANTISSA_FLOOR {
            scale /= 2;
            shift -= 1;
        }
        if shift < 0 {
            return Err("requantization multiplier is too large to encode; \
                 DPU_OUT_CVT_SHIFT cannot be negative");
        }
        Ok(Multiplier {
            scale,
            shift: shift as u32,
        })
    }

    /// Encodes the per-tensor half of a requantisation whose per-channel BS
    /// multipliers are normalised to [`BS_UNIT_MULTIPLIER`].
    ///
    /// The hardware applies `(accumulator * bs_multiplier) >>
    /// BS_MULTIPLIER_SHIFT` before this stage sees it, so a plane at unit
    /// contributes a gain of `2^(14 - 7)` that has to come back out here.
    /// Measured on hardware; see [`BS_MULTIPLIER_SHIFT`].
    pub fn for_unit_bs(total_ratio: f64) -> Multiplier {
        match Multiplier::try_for_unit_bs(total_ratio) {
            Ok(multiplier) => multiplier,
            Err(reason) => panic!("{reason}: {total_ratio}"),
        }
    }

    /// [`Multiplier::for_unit_bs`]'s fallible form, for a ratio that arrives
    /// from outside the process.
    pub fn try_for_unit_bs(total_ratio: f64) -> Result<Multiplier, &'static str> {
        let bs_gain = f64::from(BS_UNIT_MULTIPLIER >> BS_MULTIPLIER_SHIFT);
        Multiplier::try_from_ratio(total_ratio / bs_gain)
    }

    /// The real multiplier this pair encodes.
    pub fn ratio(&self) -> f64 {
        f64::from(self.scale) / 2f64.powi(self.shift as i32)
    }
}

/// Activation fused into the convolution's own DPU pass.
///
/// The vendor runs this in the **BN** stage, not BS: across a 30-capture
/// activation sweep `DPU_BS_CFG` is byte-identical at `0x20150` for every
/// activation while `DPU_BN_CFG` moves, and `DPU_BN_ALU_CFG`,
/// `DPU_BN_MUL_CFG` and `DPU_RDMA_RDMA_BN_BASE_ADDR` stay zero throughout.
/// Turning the stage on costs no operand buffer and no DMA.
///
/// The retired Mesa-derived convolution builder fused activation into the BS
/// stage instead. Nothing ran that path against a real activated model, so
/// this is not evidence it computed the wrong thing -- only that it was not
/// what the vendor emits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    /// `DPU_BN_CFG` = `0x53`. The whole BN stage is bypassed.
    None,
    /// Unbounded ReLU, `DPU_BN_CFG` = `0x12`.
    Relu,
    /// ReLU clamped at a ceiling (relu6 and friends), `DPU_BN_CFG` = `0x92`.
    ///
    /// `cmp` is the ceiling in the *accumulator's* own units, which is where
    /// BN sits -- before `DPU_OUT_CVT` requantizes. Build it with
    /// [`Activation::clamped_fp16`] or [`Activation::clamped_int8`] rather
    /// than by hand; the two precisions encode it completely differently.
    Clamped { cmp: u32 },
}

impl Activation {
    /// A clamped ReLU for an fp16 convolution.
    ///
    /// The fp16 accumulator is float, so the ceiling goes in as its raw
    /// IEEE-754 **binary32** bit pattern -- not fp16, despite the
    /// surrounding pipeline. Confirmed at three ceilings: 1.0 is
    /// `0x3F80_0000`, 2.0 `0x4000_0000`, 6.0 `0x40C0_0000`.
    /// A clamped ReLU for an fp16 convolution.
    ///
    /// The fp16 accumulator is float, so the ceiling goes in as its raw
    /// IEEE-754 **binary32** bit pattern -- not fp16, despite the
    /// surrounding pipeline. Confirmed at three ceilings: 1.0 is
    /// `0x3F80_0000`, 2.0 `0x4000_0000`, 6.0 `0x40C0_0000`.
    pub fn clamped_fp16(ceiling: f32) -> Activation {
        Activation::try_clamped_fp16(ceiling).unwrap_or_else(|error| panic!("{error}"))
    }

    /// [`Activation::clamped_fp16`], returning the refusal instead of panicking.
    pub fn try_clamped_fp16(ceiling: f32) -> Result<Activation, PlanError> {
        refuse_unless!(
            ceiling.is_finite() && ceiling > 0.0,
            PlanErrorCode::InvalidShape,
            "activation ceiling must be finite and positive, got {ceiling}"
        );
        Ok(Activation::Clamped {
            cmp: ceiling.to_bits(),
        })
    }

    /// A clamped ReLU for an int8 convolution.
    ///
    /// The int8 accumulator is a scaled integer, so the ceiling is divided
    /// by the accumulator's unit -- `input_scale * weights_scale` -- then
    /// expressed in the BN stage's post-BS domain. `BsEntry::default` (in the HAL)
    /// multiplies by [`BS_UNIT_MULTIPLIER`] and the hardware applies the
    /// effective shift [`BS_MULTIPLIER_SHIFT`], giving a gain of 128 before
    /// BN sees the value. [`Multiplier::for_unit_bs`] removes that gain
    /// later, in `OUT_CVT`, after the clamp has already happened.
    ///
    /// This is why the two scales are taken separately rather than as the
    /// [`Multiplier`] the output conversion uses: that one has the output
    /// scale divided into it already and cannot be undone.
    ///
    /// Derived by observing that `cmp / ceiling` is constant per model
    /// across all three swept ceilings, then multiplying it by the capture's
    /// own `conv_scale` and landing on exactly 255.0 for the clip-to-1.0
    /// models. The additional BS gain was then measured on hardware:
    /// programming the capture-derived value clamps the final output to
    /// zero, while multiplying it by 128 clamps at the requested value.
    ///
    /// This constructor is paired with `BsEntry::default` (in the HAL) and
    /// [`Multiplier::for_unit_bs`]. Callers deliberately using a different
    /// BS multiplier must construct [`Activation::Clamped`] in that custom
    /// post-BS domain.
    /// A clamped ReLU for an int8 convolution.
    ///
    /// The int8 accumulator is a scaled integer, so the ceiling is divided
    /// by the accumulator's unit -- `input_scale * weights_scale` -- then
    /// expressed in the BN stage's post-BS domain. `BsEntry::default` (in the HAL)
    /// multiplies by [`BS_UNIT_MULTIPLIER`] and the hardware applies the
    /// effective shift [`BS_MULTIPLIER_SHIFT`], giving a gain of 128 before
    /// BN sees the value. [`Multiplier::for_unit_bs`] removes that gain
    /// later, in `OUT_CVT`, after the clamp has already happened.
    ///
    /// This is why the two scales are taken separately rather than as the
    /// [`Multiplier`] the output conversion uses: that one has the output
    /// scale divided into it already and cannot be undone.
    ///
    /// Derived by observing that `cmp / ceiling` is constant per model
    /// across all three swept ceilings, then multiplying it by the capture's
    /// own `conv_scale` and landing on exactly 255.0 for the clip-to-1.0
    /// models. The additional BS gain was then measured on hardware:
    /// programming the capture-derived value clamps the final output to
    /// zero, while multiplying it by 128 clamps at the requested value.
    ///
    /// This constructor is paired with `BsEntry::default` (in the HAL) and
    /// [`Multiplier::for_unit_bs`]. Callers deliberately using a different
    /// BS multiplier must construct [`Activation::Clamped`] in that custom
    /// post-BS domain.
    pub fn clamped_int8(ceiling: f32, input_scale: f32, weights_scale: f32) -> Activation {
        Activation::try_clamped_int8(ceiling, input_scale, weights_scale)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    /// [`Activation::clamped_int8`], returning the refusal instead of panicking.
    pub fn try_clamped_int8(
        ceiling: f32,
        input_scale: f32,
        weights_scale: f32,
    ) -> Result<Activation, PlanError> {
        refuse_unless!(
            ceiling.is_finite() && ceiling > 0.0,
            PlanErrorCode::InvalidShape,
            "activation ceiling must be finite and positive, got {ceiling}"
        );
        let unit = f64::from(input_scale) * f64::from(weights_scale);
        refuse_unless!(
            unit.is_finite() && unit > 0.0,
            PlanErrorCode::InvalidShape,
            "input_scale * weights_scale must be finite and positive, got {unit}"
        );
        let bs_gain = f64::from(BS_UNIT_MULTIPLIER >> BS_MULTIPLIER_SHIFT);
        let cmp = (f64::from(ceiling) / unit * bs_gain).round();
        refuse_unless!(
            (0.0..=f64::from(u32::MAX)).contains(&cmp),
            PlanErrorCode::HardwareLimit,
            "activation ceiling {ceiling} is {cmp} post-BS units, outside \
             the 32-bit BN_RELUX_CMP_VALUE field"
        );
        Ok(Activation::Clamped { cmp: cmp as u32 })
    }
}

/// Logical geometry of the whole feature map a program operates on.
///
/// Every register formula below is validated against a sweep of 35 vendor
/// captures (212 convolution programs) spanning widths 32..256 and heights
/// 32..256, initially at `Cin=3`, `Cout=8`, stride 1, and 1x1 or 3x3
/// kernels. The later channel, stride, rectangular, and even-kernel sweeps
/// extend the individual rules documented on the fields and methods below.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub width: u32,
    pub height: u32,
    /// Equal in both axes; `CNA_CONV_CON3` programs it directly, confirmed
    /// across 150 stride-2, -3 and -4 programs.
    pub stride: u32,
    /// Real input channels, before any padding.
    pub in_channels: u32,
    /// Real output channels, before any padding. Normally programmed directly
    /// into `CNA_WEIGHT_SIZE2.weight_kernels` and
    /// `DPU_DATA_CUBE_CHANNEL.orig_channel` with no rounding at all: the
    /// corpus confirms 23 distinct values from 1 to 512, including 9, 14,
    /// 20, 28, 40, 56 and 72. The driver plans a copied, physically widened
    /// shape for the accumulator parity workaround; this logical shape and
    /// its ABI buffer sizes do not change.
    pub out_channels: u32,
    /// Element precision, and for int8 the quantization parameters with it.
    pub precision: Precision,
    /// Explicit `[pad_top, pad_left]`, or `None` to use `kernel / 2` on each
    /// axis. Keeping the default implicit preserves the original constructors
    /// while allowing padding to vary independently for even kernels.
    pub padding: Option<Padding>,
    /// Activation fused into this convolution's own DPU pass.
    pub activation: Activation,
    /// One filter per input channel rather than one per (input, output)
    /// pair, `CORE_MISC_CFG.DW_EN`.
    ///
    /// The capture corpus covers only a channel multiplier of one, so
    /// `out_channels` must equal `in_channels`; the builder asserts it.
    pub depthwise: bool,
}

/// How the feature map is laid out in memory, which the channel count picks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeatureLayout {
    /// Dense NHWC: the pixel is narrower than half a feature atom and the
    /// CNA pads it internally. This is the C3 case of the original captures.
    Dense,
    /// NC1HWC2: one 16-byte atom per pixel per surface.
    Surfaces,
}

impl Shape {
    /// The `32x32` stride-1 geometry of the original vendor captures.
    /// The `32x32` stride-1 geometry of the original vendor captures.
    pub const CAPTURED: Shape = Shape {
        width: IMAGE_WIDTH,
        height: IMAGE_HEIGHT,
        stride: 1,
        in_channels: INPUT_CHANNELS,
        out_channels: OUTPUT_CHANNELS,
        precision: Precision::Fp16,
        padding: None,
        activation: Activation::None,
        depthwise: false,
    };

    pub fn new(width: u32, height: u32) -> Shape {
        Shape::with_stride(width, height, 1)
    }

    pub fn with_stride(width: u32, height: u32, stride: u32) -> Shape {
        Shape::with_channels(width, height, stride, INPUT_CHANNELS)
    }

    pub fn with_channels(width: u32, height: u32, stride: u32, in_channels: u32) -> Shape {
        Shape::with_out_channels(width, height, stride, in_channels, OUTPUT_CHANNELS)
    }

    pub fn with_out_channels(
        width: u32,
        height: u32,
        stride: u32,
        in_channels: u32,
        out_channels: u32,
    ) -> Shape {
        Shape::with_precision(
            width,
            height,
            stride,
            in_channels,
            out_channels,
            Precision::Fp16,
        )
    }

    pub fn with_precision(
        width: u32,
        height: u32,
        stride: u32,
        in_channels: u32,
        out_channels: u32,
        precision: Precision,
    ) -> Shape {
        Shape::try_with_precision(width, height, stride, in_channels, out_channels, precision)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    /// [`Shape::with_precision`], returning the planner's refusal instead of
    /// panicking. This is the fallible root every other constructor and
    /// `try_with_*` builder goes through.
    pub fn try_with_precision(
        width: u32,
        height: u32,
        stride: u32,
        in_channels: u32,
        out_channels: u32,
        precision: Precision,
    ) -> Result<Shape, PlanError> {
        refuse_unless!(
            width > 0 && height > 0,
            PlanErrorCode::InvalidShape,
            "convolution extents must be nonzero"
        );
        refuse_unless!(
            stride > 0,
            PlanErrorCode::InvalidShape,
            "convolution stride must be nonzero"
        );
        refuse_unless!(
            in_channels > 0 && out_channels > 0,
            PlanErrorCode::InvalidShape,
            "convolution channel counts must be nonzero"
        );
        refuse_unless!(
            (1..=precision.max_in_channels()).contains(&in_channels)
                || policy::unbacked_channels_allowed(),
            PlanErrorCode::UnvalidatedConfiguration,
            "input channels must be 1..={}; beyond that the channel padding \
             has no capture backing at this precision",
            precision.max_in_channels()
        );
        refuse_unless!(
            (1..=precision.max_out_channels()).contains(&out_channels)
                || policy::unbacked_channels_allowed(),
            PlanErrorCode::UnvalidatedConfiguration,
            "output channels must be 1..={}; beyond that the capture corpus \
             does not reach and the measurement has not been made",
            precision.max_out_channels()
        );
        if precision == Precision::Int4 {
            // A partial int4 feature atom has no measurement behind it, and
            // the ARGB dense path below `MAX_DENSE_CHANNELS` cannot address
            // a nibble at all. Whole atoms are also what every useful int4
            // shape has, so this refuses rather than guesses.
            refuse_unless!(
                in_channels.is_multiple_of(FEATURE_ATOM_BYTES * 2),
                PlanErrorCode::UnvalidatedConfiguration,
                "int4 input channels must be a whole {}-channel feature atom; \
                 partial int4 atoms are unmeasured",
                FEATURE_ATOM_BYTES * 2
            );
        }
        if let Precision::Int8Accumulator(quantization) = precision {
            refuse_unless!(
                quantization.input_zero_point == 0
                    && quantization.output_zero_point == 0
                    && quantization.weight_zero_point == 0,
                PlanErrorCode::UnsupportedSemantics,
                "int32 accumulator output currently requires zero input, weight, and output zero-points"
            );
        }
        // Every input-side byte count the planner and the DMA engines form
        // from these extents is bounded by the NC1HWC2 surface stride,
        // `width * height * 16`; refusing it here is what lets the rest of
        // the planner use plain u32 arithmetic.
        refuse_unless!(
            width
                .checked_mul(height)
                .and_then(|pixels| pixels.checked_mul(FEATURE_ATOM_BYTES))
                .is_some(),
            PlanErrorCode::HardwareLimit,
            "input surface {width}x{height} exceeds the 32-bit address space the DMA \
             engines can name"
        );
        Ok(Shape {
            width,
            height,
            stride,
            in_channels,
            out_channels,
            precision,
            padding: None,
            activation: Activation::None,
            depthwise: false,
        })
    }

    /// Sets the model's leading padding independently on each axis.
    ///
    /// The capture corpus covers padding no larger than its kernel extent;
    /// the exact per-kernel bound is checked when the kernel is supplied.
    /// Sets the model's leading padding independently on each axis.
    ///
    /// The capture corpus covers padding no larger than its kernel extent;
    /// the exact per-kernel bound is checked when the kernel is supplied.
    pub fn with_padding(self, padding: Padding) -> Shape {
        self.try_with_padding(padding)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    /// [`Shape::with_padding`], returning the refusal instead of panicking.
    pub fn try_with_padding(mut self, padding: Padding) -> Result<Shape, PlanError> {
        refuse_unless!(
            padding.into_iter().all(|pad| pad <= 15),
            PlanErrorCode::HardwareLimit,
            "convolution padding must fit the CNA's 4-bit pad fields"
        );
        self.padding = Some(padding);
        Ok(self)
    }

    /// Fuses `activation` into this convolution's own DPU pass.
    /// Fuses `activation` into this convolution's own DPU pass.
    pub fn with_activation(self, activation: Activation) -> Shape {
        self.try_with_activation(activation)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    /// [`Shape::with_activation`], returning the refusal instead of panicking.
    pub fn try_with_activation(mut self, activation: Activation) -> Result<Shape, PlanError> {
        refuse_unless!(
            !self.precision.writes_accumulators() || activation == Activation::None,
            PlanErrorCode::UnsupportedSemantics,
            "int32 accumulator output must not fuse an activation"
        );
        self.activation = activation;
        Ok(self)
    }

    /// Makes this a depthwise convolution, one filter per input channel.
    ///
    /// Only a channel multiplier of one is captured, so the output channel
    /// count must already equal the input one.
    ///
    /// # Packing the weight buffer
    ///
    /// Use `iree_rocket_hal::rocket::tensor_layout::pack_depthwise_to_rocket_weights`,
    /// not `pack_hwcf_to_rocket_weights`. A depthwise filter is
    /// `[Cin][kh][kw]` and the hardware wants it tap-major,
    /// `(ky * kw + kx) * padded_channels + channel`, which is the transpose
    /// of how torch and ONNX store it.
    ///
    /// The capture sweep could not have shown this -- a capture carries the
    /// register program, never the buffer it points at. It came from one-hot
    /// probing every slot of a real weight buffer on hardware
    /// (`tests/conv_depthwise_probe_hw.rs`).
    /// Makes this a depthwise convolution, one filter per input channel.
    ///
    /// Only a channel multiplier of one is captured, so the output channel
    /// count must already equal the input one.
    ///
    /// # Packing the weight buffer
    ///
    /// Use `iree_rocket_hal::rocket::tensor_layout::pack_depthwise_to_rocket_weights`,
    /// not `pack_hwcf_to_rocket_weights`. A depthwise filter is
    /// `[Cin][kh][kw]` and the hardware wants it tap-major,
    /// `(ky * kw + kx) * padded_channels + channel`, which is the transpose
    /// of how torch and ONNX store it.
    ///
    /// The capture sweep could not have shown this -- a capture carries the
    /// register program, never the buffer it points at. It came from one-hot
    /// probing every slot of a real weight buffer on hardware
    /// (`tests/conv_depthwise_probe_hw.rs`).
    pub fn with_depthwise(self) -> Shape {
        self.try_with_depthwise()
            .unwrap_or_else(|error| panic!("{error}"))
    }

    /// [`Shape::with_depthwise`], returning the refusal instead of panicking.
    pub fn try_with_depthwise(mut self) -> Result<Shape, PlanError> {
        refuse_unless!(
            self.in_channels == self.out_channels,
            PlanErrorCode::UnsupportedSemantics,
            "depthwise capture backing covers a channel multiplier of one only"
        );
        refuse_unless!(
            self.in_channels <= MAX_DEPTHWISE_CHANNELS || policy::unbacked_channels_allowed(),
            PlanErrorCode::UnvalidatedConfiguration,
            "depthwise channels must be 1..={MAX_DEPTHWISE_CHANNELS}; the dense ceilings \
             reach further on dense evidence that does not carry over -- see \
             MAX_DEPTHWISE_CHANNELS"
        );
        // Depthwise is *not* automatically inherited by a new datatype the
        // way the dense path is. Two things stop it:
        //
        //   * the coefficient grouping is a 64-byte run
        //     (`tensor_layout::pack_depthwise_to_rocket_weights`), so it is
        //     a channel count only once an element width is fixed -- 32 at
        //     two bytes, 64 at one, and *128* at int4's half byte, which
        //     that function's byte-valued `element_size` cannot express;
        //   * depthwise keeps its own output writer, with a 256-byte write
        //     atom on the serial path, which no fp32-result measurement
        //     covers.
        //
        // So the widths with hardware behind them are the 1- and 2-byte
        // ones, and everything else is refused rather than guessed at. The
        // int8 grouping bug this grouping was written to fix was invisible
        // to a uniform-weight probe, which is exactly what a speculative
        // depthwise rung would get tested with first.
        refuse_unless!(
            matches!(self.precision.element_bits(), 8 | 16) && !self.precision.writes_fp32_result(),
            PlanErrorCode::UnvalidatedConfiguration,
            "{:?} depthwise is unmeasured; see Shape::with_depthwise",
            self.precision
        );
        self.depthwise = true;
        Ok(self)
    }

    /// Channel granule the programmed channel count rounds up to.
    ///
    /// Depthwise doubles it -- fp16 rounds to 32 where dense rounds to 16,
    /// int8 to 64 where dense rounds to 32 -- which the nine-point channel
    /// ladder pins in both precisions.
    ///
    /// Mesa's own depthwise path instead doubles the count when it is at
    /// most 32 and then rounds to a multiple of 64. That disagrees with the
    /// captures at three of the seven fp16 points (Cout 8 and 32 are
    /// programmed 32 where Mesa says 64, and 96 is programmed 96 where Mesa
    /// says 128), agreeing only where the two rules coincide.
    /// Channel granule the programmed channel count rounds up to.
    ///
    /// Depthwise doubles it -- fp16 rounds to 32 where dense rounds to 16,
    /// int8 to 64 where dense rounds to 32 -- which the nine-point channel
    /// ladder pins in both precisions.
    ///
    /// Mesa's own depthwise path instead doubles the count when it is at
    /// most 32 and then rounds to a multiple of 64. That disagrees with the
    /// captures at three of the seven fp16 points (Cout 8 and 32 are
    /// programmed 32 where Mesa says 64, and 96 is programmed 96 where Mesa
    /// says 128), agreeing only where the two rules coincide.
    pub fn out_channel_granule(&self) -> u32 {
        let dense = self.precision.out_channel_granule();
        if self.depthwise { 2 * dense } else { dense }
    }

    pub fn kernel_programming(&self, kernels: Kernels) -> KernelProgramming {
        kernel_programming(kernels, self.padding)
    }

    /// The relation [`Shape::output_width`] and [`Shape::output_height`]
    /// assert: each kernel extent fits its padded input extent. Checked up
    /// front by [`ConvPlan::try_new`] so the planner never reaches those
    /// asserts on a malformed descriptor.
    pub fn check_extents_against(&self, kernel: KernelProgramming) -> Result<(), PlanError> {
        let padded_width = self.width + 2 * kernel.pad_left;
        refuse_unless!(
            kernel.width <= padded_width,
            PlanErrorCode::InvalidShape,
            "kernel width {} exceeds the padded input width {padded_width}",
            kernel.width
        );
        let padded_height = self.height + 2 * kernel.pad_top;
        refuse_unless!(
            kernel.height <= padded_height,
            PlanErrorCode::InvalidShape,
            "kernel height {} exceeds the padded input height {padded_height}",
            kernel.height
        );
        Ok(())
    }

    /// [`Shape::weight_bytes`] with checked arithmetic: the coefficient
    /// tensor has to fit the 32-bit byte counts the CNA is programmed with,
    /// and at the channel ceilings times an 11x11 kernel it does not.
    pub fn try_weight_bytes(&self, kernels: Kernels) -> Result<u32, PlanError> {
        let kernel = self.try_kernel_programming(kernels)?;
        let elements = if self.depthwise {
            u64::from(self.cbuf_atoms()) * u64::from(self.precision.channels_per_atom())
        } else {
            u64::from(self.weight_channels()) * u64::from(self.programmed_kernels())
        };
        let bytes = elements * u64::from(self.precision.element_bits()) / 8
            * u64::from(kernel.height)
            * u64::from(kernel.width);
        u32::try_from(bytes).map_err(|_| {
            PlanError::new(
                PlanErrorCode::HardwareLimit,
                format!(
                    "coefficient tensor of {bytes} bytes ({:?} kernel {kernels:?}) exceeds \
                     the 32-bit byte counts the CNA is programmed with",
                    self.precision
                ),
            )
        })
    }

    /// [`Shape::kernel_programming`], returning the refusal instead of
    /// panicking.
    pub fn try_kernel_programming(&self, kernels: Kernels) -> Result<KernelProgramming, PlanError> {
        try_kernel_programming(kernels, self.padding)
    }

    /// Feature atoms one pixel occupies once padded.
    /// Feature atoms one pixel occupies once padded.
    pub fn feature_atoms(&self) -> u32 {
        self.in_channels
            .div_ceil(self.precision.channels_per_atom())
            .max(1)
    }

    /// Channel count programmed into `CNA_DATA_SIZE1.datain_channel`.
    ///
    /// Whole atoms in both precisions, with no exception anywhere: `Cin`
    /// rounded up to 8 at fp16 and to 16 at int8. The fp16 side used to be a
    /// table, but only because the same table carried the weight padding,
    /// which is where the exceptions actually live -- the feature side never
    /// had any. Measured at 66 fp16 and 44 int8 channel counts to 512.
    /// Channel count programmed into `CNA_DATA_SIZE1.datain_channel`.
    ///
    /// Whole atoms in both precisions, with no exception anywhere: `Cin`
    /// rounded up to 8 at fp16 and to 16 at int8. The fp16 side used to be a
    /// table, but only because the same table carried the weight padding,
    /// which is where the exceptions actually live -- the feature side never
    /// had any. Measured at 66 fp16 and 44 int8 channel counts to 512.
    pub fn padded_channels(&self) -> u32 {
        self.feature_atoms() * self.precision.channels_per_atom()
    }

    /// Channel count the coefficient footprint is computed from.
    ///
    /// At fp16 the atom count rounds up to a whole group of four, so `Cin`
    /// 17..24 pads to 32 while `datain_channel` stays 24, and the same bump
    /// lands on 88, 120, 152, 184, 216, 248, 280, 344, 440 and 504 -- every
    /// count where `ceil(Cin / 8)` is `3 mod 4`, out to the 512 measured.
    /// At int8 it does not: the coefficient padding is exactly
    /// `padded_channels` at all 44 measured counts, including int8's own 3-,
    /// 7-, 11- and 15-atom points where the fp16 rule would have bumped it.
    ///
    /// The asymmetry is the fp16 weight layout: the TRM has fp16 loading 16
    /// kernels per group against int8's 32.
    /// bf16 and int16 take the fp16 branch: the quad-atom bump is a property
    /// of the 16-kernel weight group a 2-byte element loads with, which
    /// `../rockchip-npu-notes/encodings/tile-layouts.md` records as shared
    /// (`weight_int16` == `weight_fp16`, and bf16 reuses the same tile).
    /// Channel count the coefficient footprint is computed from.
    ///
    /// At fp16 the atom count rounds up to a whole group of four, so `Cin`
    /// 17..24 pads to 32 while `datain_channel` stays 24, and the same bump
    /// lands on 88, 120, 152, 184, 216, 248, 280, 344, 440 and 504 -- every
    /// count where `ceil(Cin / 8)` is `3 mod 4`, out to the 512 measured.
    /// At int8 it does not: the coefficient padding is exactly
    /// `padded_channels` at all 44 measured counts, including int8's own 3-,
    /// 7-, 11- and 15-atom points where the fp16 rule would have bumped it.
    ///
    /// The asymmetry is the fp16 weight layout: the TRM has fp16 loading 16
    /// kernels per group against int8's 32.
    /// bf16 and int16 take the fp16 branch: the quad-atom bump is a property
    /// of the 16-kernel weight group a 2-byte element loads with, which
    /// `../rockchip-npu-notes/encodings/tile-layouts.md` records as shared
    /// (`weight_int16` == `weight_fp16`, and bf16 reuses the same tile).
    pub fn weight_channels(&self) -> u32 {
        if self.precision.shares_fp16_layout() {
            quad_atoms(self.feature_atoms()) * self.precision.channels_per_atom()
        } else {
            self.padded_channels()
        }
    }

    /// Kernel count the coefficient side is programmed with.
    ///
    /// At int8 this is `Cout` rounded up to an even number; at fp16 it is
    /// `Cout` itself. `CNA_WEIGHT_SIZE2.weight_kernels` and the coefficient
    /// footprint both follow it, while `orig_channel` keeps the true count.
    ///
    /// Measured against vendor captures at 32x32, `Cin` 3, 3x3: int8 `Cout`
    /// 1 programs 2 kernels and 2 x `bytes_per_kernel`, 3 programs 4, 5
    /// programs 6, and even counts pass through. fp16 programs the true
    /// count at every value including 1, 9 and 14.
    ///
    /// This is what int8 `Cout` 1 was failing on. Programming a single
    /// kernel put output channel 0 wrong by about two LSB on hardware, with
    /// a result that alternated between consecutive jobs, while fp16 `Cout`
    /// 1 was exact. The int8 corpus had no capture below `Cout` 8, so
    /// nothing until now said the vendor never programs an odd kernel count
    /// there.
    /// A depthwise convolution programs a single kernel whatever its channel
    /// count: there is one filter per input channel rather than a kernel set
    /// per output channel, and the channel dimension is carried by the cube
    /// registers instead. All nine depthwise captures program 1 here, in
    /// both precisions, which is also why `sweep_axis.py` cannot group them
    /// -- it matches a program to its model partly by `weight_kernels`.
    /// Kernel count the coefficient side is programmed with.
    ///
    /// At int8 this is `Cout` rounded up to an even number; at fp16 it is
    /// `Cout` itself. `CNA_WEIGHT_SIZE2.weight_kernels` and the coefficient
    /// footprint both follow it, while `orig_channel` keeps the true count.
    ///
    /// Measured against vendor captures at 32x32, `Cin` 3, 3x3: int8 `Cout`
    /// 1 programs 2 kernels and 2 x `bytes_per_kernel`, 3 programs 4, 5
    /// programs 6, and even counts pass through. fp16 programs the true
    /// count at every value including 1, 9 and 14.
    ///
    /// This is what int8 `Cout` 1 was failing on. Programming a single
    /// kernel put output channel 0 wrong by about two LSB on hardware, with
    /// a result that alternated between consecutive jobs, while fp16 `Cout`
    /// 1 was exact. The int8 corpus had no capture below `Cout` 8, so
    /// nothing until now said the vendor never programs an odd kernel count
    /// there.
    /// A depthwise convolution programs a single kernel whatever its channel
    /// count: there is one filter per input channel rather than a kernel set
    /// per output channel, and the channel dimension is carried by the cube
    /// registers instead. All nine depthwise captures program 1 here, in
    /// both precisions, which is also why `sweep_axis.py` cannot group them
    /// -- it matches a program to its model partly by `weight_kernels`.
    pub fn programmed_kernels(&self) -> u32 {
        if self.depthwise {
            return 1;
        }
        // The even-kernel rounding belongs to the sub-16-bit widths, whose
        // coefficient atom holds 32 or 64 kernels. It is measured at int8
        // and assumed at int4; the 2- and 4-byte widths program the true
        // count, which is also what `WeightLayout` sizes their buffers to.
        if self.precision.element_bits() >= 16 {
            self.out_channels
        } else {
            self.out_channels.next_multiple_of(2)
        }
    }

    /// Bytes to allocate and populate for this convolution's BS buffer.
    ///
    /// Sized from the *padded* output channel count, not the true one, which
    /// matters only when the two differ enough to cross a BS block.
    ///
    /// This is defensive, not a fix for anything currently known to be
    /// broken. It came out of chasing the int8 `Cout` 1 defect, where
    /// poisoning the bytes after this buffer moved the result -- but that
    /// turned out to be a symptom: the job was already wrong because the
    /// programmed kernel count was odd, and poisoning adjacent memory only
    /// perturbed an already-broken job. [`Shape::programmed_kernels`] is the
    /// actual fix, and `Cout` 1 is exact on hardware with it.
    ///
    /// Populating the padded count is kept because it costs a few hundred
    /// bytes and leaves no undefined region for a DMA to reach into.
    /// `int8_bs_read_extent_probe` re-run against the corrected kernel count
    /// would say whether even that is necessary.
    /// Bytes to allocate and populate for this convolution's BS buffer.
    ///
    /// Sized from the *padded* output channel count, not the true one, which
    /// matters only when the two differ enough to cross a BS block.
    ///
    /// This is defensive, not a fix for anything currently known to be
    /// broken. It came out of chasing the int8 `Cout` 1 defect, where
    /// poisoning the bytes after this buffer moved the result -- but that
    /// turned out to be a symptom: the job was already wrong because the
    /// programmed kernel count was odd, and poisoning adjacent memory only
    /// perturbed an already-broken job. [`Shape::programmed_kernels`] is the
    /// actual fix, and `Cout` 1 is exact on hardware with it.
    ///
    /// Populating the padded count is kept because it costs a few hundred
    /// bytes and leaves no undefined region for a DMA to reach into.
    /// `int8_bs_read_extent_probe` re-run against the corrected kernel count
    /// would say whether even that is necessary.
    pub fn bs_buffer_bytes(&self) -> usize {
        bs_buffer_bytes(self.padded_out_channels())
    }

    /// Output channel count the DPU is programmed with, rounded up to a
    /// whole [`OUTPUT_CHANNEL_GRANULE`] and never below one.
    ///
    /// The floor is what makes Cout 8 and Cout 16 program the same value,
    /// which is why the shape-only corpus -- fixed at Cout 8 -- could not
    /// distinguish this from the true count.
    /// Output channel count the DPU is programmed with, rounded up to a
    /// whole [`OUTPUT_CHANNEL_GRANULE`] and never below one.
    ///
    /// The floor is what makes Cout 8 and Cout 16 program the same value,
    /// which is why the shape-only corpus -- fixed at Cout 8 -- could not
    /// distinguish this from the true count.
    pub fn padded_out_channels(&self) -> u32 {
        let granule = self.out_channel_granule();
        self.out_channels.next_multiple_of(granule).max(granule)
    }

    /// Output blocks per pixel the DPU commits, the quantity the output
    /// parity rule counts.
    /// Output blocks per pixel the DPU commits, the quantity the output
    /// parity rule counts.
    pub fn output_blocks_per_pixel(&self) -> u32 {
        (self.padded_out_channels() * self.precision.output_element_bytes())
            .div_ceil(self.output_atom_bytes())
    }

    /// Returns the physical shape that should be handed to the planner.
    ///
    /// `self` remains the logical ABI shape; this is where a physical/logical
    /// divergence would live. **There is currently none** -- it is the
    /// identity -- and it is kept as the hook because the driver, the
    /// executable format and the oracle harness are all already routed
    /// through it.
    ///
    /// It used to widen `Cout` to satisfy an "even committed block count"
    /// rule, and to refuse two families of shape outright: a 3x3 accumulator
    /// output extent with a 3x3 kernel, and anything past 384 coefficient
    /// bytes per output channel. All three were consequences of the dense
    /// accumulator driving the DPU's *serial* writer (`mc_surf_out = 1`),
    /// which stops emitting once it runs out of surfaces. With the writer
    /// corrected to `mc_surf_out = 0` / `size_e = 7` /
    /// `surf_add = dataout * 8`, and the readback to the C2=4 cube that writer
    /// produces, none of the three has anything left to describe: coefficient
    /// footprints of 1024 bytes/channel at 1x1 and 2304 at 3x3 are bit-exact,
    /// single- and multi-tile [HW sweep, planck 2026-09-03; see
    /// `Shape::output_channel_block_bytes`].
    /// Returns the physical shape that should be handed to the planner.
    ///
    /// `self` remains the logical ABI shape; this is where a physical/logical
    /// divergence would live. **There is currently none** -- it is the
    /// identity -- and it is kept as the hook because the driver, the
    /// executable format and the oracle harness are all already routed
    /// through it.
    ///
    /// It used to widen `Cout` to satisfy an "even committed block count"
    /// rule, and to refuse two families of shape outright: a 3x3 accumulator
    /// output extent with a 3x3 kernel, and anything past 384 coefficient
    /// bytes per output channel. All three were consequences of the dense
    /// accumulator driving the DPU's *serial* writer (`mc_surf_out = 1`),
    /// which stops emitting once it runs out of surfaces. With the writer
    /// corrected to `mc_surf_out = 0` / `size_e = 7` /
    /// `surf_add = dataout * 8`, and the readback to the C2=4 cube that writer
    /// produces, none of the three has anything left to describe: coefficient
    /// footprints of 1024 bytes/channel at 1x1 and 2304 at 3x3 are bit-exact,
    /// single- and multi-tile [HW sweep, planck 2026-09-03; see
    /// `Shape::output_channel_block_bytes`].
    pub fn parity_padded_shape(&self, _kernels: Kernels) -> Result<Shape, &'static str> {
        Ok(*self)
    }

    /// Conservative physical output allocation for this convolution, in
    /// bytes.
    ///
    /// Requantized output uses 16-byte feature-atomic surfaces. Bypassed i32
    /// output instead retains CORE's native 32-channel accumulator blocks,
    /// which occupy 128 bytes. The DPU programs the block-rounded
    /// [`Shape::padded_out_channels`] count rather than the logical one, so
    /// this allocates enough complete blocks for that padded count. This is
    /// the capture-derived counterpart of the retired Mesa builder's
    /// output-allocation formula. The total is tiling-agnostic: normal tile
    /// programs address sub-ranges of one shared image, while
    /// `programs_with_staged_accumulator_output` (on the HAL's `ConvPlan`) partitions the
    /// same byte count into contiguous tile-local ranges.
    /// Conservative physical output allocation for this convolution, in
    /// bytes.
    ///
    /// Requantized output uses 16-byte feature-atomic surfaces. Bypassed i32
    /// output instead retains CORE's native 32-channel accumulator blocks,
    /// which occupy 128 bytes. The DPU programs the block-rounded
    /// [`Shape::padded_out_channels`] count rather than the logical one, so
    /// this allocates enough complete blocks for that padded count. This is
    /// the capture-derived counterpart of the retired Mesa builder's
    /// output-allocation formula. The total is tiling-agnostic: normal tile
    /// programs address sub-ranges of one shared image, while
    /// `programs_with_staged_accumulator_output` (on the HAL's `ConvPlan`) partitions the
    /// same byte count into contiguous tile-local ranges.
    pub fn output_scratch_bytes(&self, kernels: Kernels) -> usize {
        let channel_bytes =
            self.padded_out_channels() as usize * self.precision.output_element_bytes() as usize;
        // The *write* atom, not the programmed block -- depthwise accumulator
        // output writes 256-byte atoms, so sizing this at 128 left the DPU
        // writing twice what was allocated. See `output_atom_bytes`.
        let block_bytes = self.output_atom_bytes() as usize;
        let block_count = channel_bytes.div_ceil(block_bytes);
        self.output_width(kernels) as usize
            * self.output_height(kernels) as usize
            * block_count
            * block_bytes
    }

    /// Bytes of fp16 coefficients the whole kernel set occupies.
    ///
    /// `weight_channels * kh * kw * Cout * 2`, which reproduces
    /// `CNA_WEIGHT_SIZE0.weight_bytes` in all 829 programs of the corpus, and
    /// in all 633 of the rectangular-kernel sweep once the two kernel extents
    /// are taken apart. Note the *padded* input channel count, and
    /// specifically the weight padding rather than the data padding -- at
    /// three atoms the two differ, and it is the weight one that this follows.
    /// A depthwise convolution drops the `Cout` factor entirely -- one
    /// filter per input channel -- and pads the channel count to a whole
    /// CBUF atom group rather than to the weight padding the dense path
    /// uses. The two differ only at int8: 48 channels is charged as 64 there
    /// (three atoms round to four), which is what the captured 576 bytes at
    /// 3x3 says and what the dense `weight_channels` would have read as 432.
    /// 3x3 at 128 channels costs 2304 bytes fp16 where dense costs 73728.
    /// Bytes of fp16 coefficients the whole kernel set occupies.
    ///
    /// `weight_channels * kh * kw * Cout * 2`, which reproduces
    /// `CNA_WEIGHT_SIZE0.weight_bytes` in all 829 programs of the corpus, and
    /// in all 633 of the rectangular-kernel sweep once the two kernel extents
    /// are taken apart. Note the *padded* input channel count, and
    /// specifically the weight padding rather than the data padding -- at
    /// three atoms the two differ, and it is the weight one that this follows.
    /// A depthwise convolution drops the `Cout` factor entirely -- one
    /// filter per input channel -- and pads the channel count to a whole
    /// CBUF atom group rather than to the weight padding the dense path
    /// uses. The two differ only at int8: 48 channels is charged as 64 there
    /// (three atoms round to four), which is what the captured 576 bytes at
    /// 3x3 says and what the dense `weight_channels` would have read as 432.
    /// 3x3 at 128 channels costs 2304 bytes fp16 where dense costs 73728.
    pub fn weight_bytes(&self, kernels: Kernels) -> u32 {
        let kernel = self.kernel_programming(kernels);
        if self.depthwise {
            return kernel.height
                * kernel.width
                * self
                    .precision
                    .bytes_for(self.cbuf_atoms() * self.precision.channels_per_atom());
        }
        kernel.height
            * kernel.width
            * self
                .precision
                .bytes_for(self.weight_channels() * self.programmed_kernels())
    }

    /// Padded channel count `iree_rocket_hal::rocket::tensor_layout::pack_depthwise_to_rocket_weights`'s
    /// tap-major stride uses -- a whole CBUF atom *group*, not just a whole
    /// atom (see [`Shape::weight_channels`]'s doc comment for why that
    /// differs from [`Shape::padded_channels`] at fp16's 3-mod-4 atom
    /// counts). [`Shape::weight_bytes`]'s depthwise branch reads the same
    /// value via [`Shape::cbuf_atoms`]; this just exposes it for callers
    /// packing the weight buffer instead of only sizing it.
    ///
    /// Only meaningful when [`Shape::depthwise`] is set -- callers packing a
    /// dense filter want [`Shape::weight_channels`] instead.
    /// Padded channel count `iree_rocket_hal::rocket::tensor_layout::pack_depthwise_to_rocket_weights`'s
    /// tap-major stride uses -- a whole CBUF atom *group*, not just a whole
    /// atom (see [`Shape::weight_channels`]'s doc comment for why that
    /// differs from [`Shape::padded_channels`] at fp16's 3-mod-4 atom
    /// counts). [`Shape::weight_bytes`]'s depthwise branch reads the same
    /// value via [`Shape::cbuf_atoms`]; this just exposes it for callers
    /// packing the weight buffer instead of only sizing it.
    ///
    /// Only meaningful when [`Shape::depthwise`] is set -- callers packing a
    /// dense filter want [`Shape::weight_channels`] instead.
    pub fn depthwise_padded_channels(&self) -> u32 {
        self.cbuf_atoms() * self.precision.channels_per_atom()
    }

    /// Atoms per pixel the CBUF charges for, which is neither
    /// `feature_atoms` nor the count implied by `weight_channels`.
    ///
    /// The atom count rounds up to a whole group of four in *both*
    /// precisions -- three charged as four, seven as eight, eleven as
    /// twelve -- while five, six, nine and ten are charged as themselves.
    ///
    /// At fp16 below `Cin` 80 this is invisible, because the weight padding
    /// bumps the same counts and `weight_channels` arrives pre-rounded. At int8
    /// the padding is exact and the rounding has to be applied here or
    /// `data_entries` comes out short: `Cin` 33..48 programs 4 atoms' worth
    /// against a padded 48, and 97..112 programs 8 against a padded 112.
    ///
    /// Above `Cin` 80 it stops being invisible at fp16 too. This was a
    /// two-entry match while the corpus ended there; the large-`Cin` sweep
    /// reads the charge back out of `data_entries` at 47 fp16 and 27 int8
    /// channel counts, and the two-entry version is wrong at 20 of the fp16
    /// and 10 of the int8 -- every one of them a `3 mod 4` atom count above
    /// the old ceiling. Charging one atom short there would silently drop a
    /// tile's last input rows, which is how the same class of bug surfaced
    /// in `conv_outchannel_hw` at 256x32.
    ///
    /// `data_entries` is not the only consumer: `data_bank_demand` and
    /// `max_tile_input_rows_for_width_and_data_banks` must bill the same
    /// rounded count. Both used the exact one until 2026-08-31 and lost a
    /// tile's last output rows at int8 for precisely the `3 mod 4` counts
    /// above -- the same failure this comment already predicted, one layer
    /// out.
    /// Atoms per pixel the CBUF charges for, which is neither
    /// `feature_atoms` nor the count implied by `weight_channels`.
    ///
    /// The atom count rounds up to a whole group of four in *both*
    /// precisions -- three charged as four, seven as eight, eleven as
    /// twelve -- while five, six, nine and ten are charged as themselves.
    ///
    /// At fp16 below `Cin` 80 this is invisible, because the weight padding
    /// bumps the same counts and `weight_channels` arrives pre-rounded. At int8
    /// the padding is exact and the rounding has to be applied here or
    /// `data_entries` comes out short: `Cin` 33..48 programs 4 atoms' worth
    /// against a padded 48, and 97..112 programs 8 against a padded 112.
    ///
    /// Above `Cin` 80 it stops being invisible at fp16 too. This was a
    /// two-entry match while the corpus ended there; the large-`Cin` sweep
    /// reads the charge back out of `data_entries` at 47 fp16 and 27 int8
    /// channel counts, and the two-entry version is wrong at 20 of the fp16
    /// and 10 of the int8 -- every one of them a `3 mod 4` atom count above
    /// the old ceiling. Charging one atom short there would silently drop a
    /// tile's last input rows, which is how the same class of bug surfaced
    /// in `conv_outchannel_hw` at 256x32.
    ///
    /// `data_entries` is not the only consumer: `data_bank_demand` and
    /// `max_tile_input_rows_for_width_and_data_banks` must bill the same
    /// rounded count. Both used the exact one until 2026-08-31 and lost a
    /// tile's last output rows at int8 for precisely the `3 mod 4` counts
    /// above -- the same failure this comment already predicted, one layer
    /// out.
    pub fn cbuf_atoms(&self) -> u32 {
        quad_atoms(self.feature_atoms())
    }

    /// Whether the feature map is dense NHWC or NC1HWC2 surfaces.
    /// Whether the feature map is dense NHWC or NC1HWC2 surfaces.
    pub fn layout(&self) -> FeatureLayout {
        if self.in_channels <= MAX_DENSE_CHANNELS {
            FeatureLayout::Dense
        } else {
            FeatureLayout::Surfaces
        }
    }

    /// Output width, `floor((w + 2 * pad_left - kw) / stride) + 1`. Matches
    /// all 150 stride-2, -3 and -4 programs in the sweep corpus. Each extent
    /// governs its own axis, so a 3x9 and a 9x3 differ here.
    /// Output width, `floor((w + 2 * pad_left - kw) / stride) + 1`. Matches
    /// all 150 stride-2, -3 and -4 programs in the sweep corpus. Each extent
    /// governs its own axis, so a 3x9 and a 9x3 differ here.
    pub fn output_width(&self, kernels: Kernels) -> u32 {
        let kernel = self.kernel_programming(kernels);
        let padded = self.width + 2 * kernel.pad_left;
        assert!(
            kernel.width <= padded,
            "kernel width {} exceeds the padded input width {padded}",
            kernel.width
        );
        (padded - kernel.width) / self.stride + 1
    }

    /// Output height, by the same rule on the kernel's height.
    /// Output height, by the same rule on the kernel's height.
    pub fn output_height(&self, kernels: Kernels) -> u32 {
        let kernel = self.kernel_programming(kernels);
        let padded = self.height + 2 * kernel.pad_top;
        assert!(
            kernel.height <= padded,
            "kernel height {} exceeds the padded input height {padded}",
            kernel.height
        );
        (padded - kernel.height) / self.stride + 1
    }

    /// Byte stride of one input row.
    ///
    /// Dense rows are exactly `Cin` fp16 values wide. Surface rows carry one
    /// 16-byte atom per pixel, and the surfaces themselves sit
    /// `width * height * 16` bytes apart.
    /// Byte stride of one input row.
    ///
    /// Dense rows are exactly `Cin` fp16 values wide. Surface rows carry one
    /// 16-byte atom per pixel, and the surfaces themselves sit
    /// `width * height * 16` bytes apart.
    pub fn input_row_stride(&self) -> u32 {
        match self.layout() {
            FeatureLayout::Dense => self.width * self.in_channels * self.precision.element_bytes(),
            FeatureLayout::Surfaces => self.width * FEATURE_ATOM_BYTES,
        }
    }

    /// Byte distance between consecutive NC1HWC2 input surfaces.
    /// Byte distance between consecutive NC1HWC2 input surfaces.
    pub fn input_surface_stride(&self) -> u32 {
        self.width * self.height * FEATURE_ATOM_BYTES
    }

    /// Width charged to dense CBUF storage.
    ///
    /// Vendor dense tensors keep the logical width in `CNA_DMA_CON1`, but
    /// round the resident row to one precision-sized feature atom: width 226
    /// is charged as 232 in fp16 and 240 in int8. `CNA_CBUF_CON1` and the
    /// captured continuation-tile offsets both expose this padding.
    /// Width charged to dense CBUF storage.
    ///
    /// Vendor dense tensors keep the logical width in `CNA_DMA_CON1`, but
    /// round the resident row to one precision-sized feature atom: width 226
    /// is charged as 232 in fp16 and 240 in int8. `CNA_CBUF_CON1` and the
    /// captured continuation-tile offsets both expose this padding.
    pub fn cbuf_input_width(&self, input_width: u32) -> u32 {
        match self.layout() {
            FeatureLayout::Dense => {
                input_width.next_multiple_of(self.precision.channels_per_atom())
            }
            FeatureLayout::Surfaces => input_width,
        }
    }

    /// Bytes charged per dense CBUF pixel.
    ///
    /// The ARGB modes are 1-, 2- and 4-channel storage classes: Cin 1 and 2
    /// retain their true widths, while Cin 3 rounds to the same class as Cin
    /// 4. This is visible in the expanded corpus at tall Cin-1 shapes, where
    /// charging four channels over-allocates data banks and invents tiles the
    /// vendor does not need.
    /// Bytes charged per dense CBUF pixel.
    ///
    /// The ARGB modes are 1-, 2- and 4-channel storage classes: Cin 1 and 2
    /// retain their true widths, while Cin 3 rounds to the same class as Cin
    /// 4. This is visible in the expanded corpus at tall Cin-1 shapes, where
    /// charging four channels over-allocates data banks and invents tiles the
    /// vendor does not need.
    pub fn dense_cbuf_pixel_bytes(&self) -> u32 {
        self.in_channels.next_power_of_two().min(4) * self.precision.element_bytes()
    }

    pub fn max_data_entries(&self) -> u32 {
        MAX_DATA_ENTRIES
    }

    /// Whether a dense-layout tile whose feature fetch starts at input row
    /// `in_first` is safe for general, non-uniform tensor data.
    ///
    /// Measured on real RK3588 hardware, not derived from documentation
    /// (`iree-rocket-design-spike`'s `conv_dense_shared_buffer_dispatch_hw.rs`,
    /// `conv_dense_odd_in_first_probe_hw.rs`,
    /// `conv_dense_alignment_width_sweep_hw.rs`, and
    /// `conv_dense_alignment_channel_sweep_hw.rs`/
    /// `conv_dense_alignment_in_first_growth_hw.rs` -- see DESIGN_NOTES.md
    /// there, "The dense (Cin<=4) ARGB path silently corrupts multi-row
    /// dispatches" and its follow-ups, for the full characterization).
    ///
    /// `CNA_FEATURE_DATA_ADDR` is `in_first * input_row_stride`, and dense
    /// mode's `input_row_stride` is not always a multiple of 16. Earlier
    /// hardware probes filled every x position and input channel alike and
    /// concluded that `nonalign_dma` safely compensates for offsets up to
    /// one dense pixel wide. That conclusion was an artifact of the data:
    /// a sub-pixel/channel displacement is invisible when all displaced
    /// values are equal.
    ///
    /// `conv_features0_exact_hw.rs` uses x-, y-, and channel-varying data
    /// at VGG-19 `features.0`'s exact lowered shape. Every tile at offset 0
    /// passed exactly, while all three tiles at offset 4 were about 94%
    /// wrong across their complete output ranges, deterministically in all
    /// three repetitions. A subsequent data-rich hardware sweep tested every
    /// even byte offset: all 14 offset-0 cases passed, while every case at
    /// offsets 2, 4, 6, 8, 10, 12, and 14 failed. The affine-int8 Cartesian
    /// oracle subsequently exposed the same defect at offset 2 for VGG-19's
    /// lowered 226x226/Cin=3 shape: tile 0 passed exactly and corruption began
    /// at tile 1's first output row. Dense tiles therefore require a fully
    /// 16-byte-aligned feature base at both precisions.
    ///
    /// Always `true` outside dense layout: surfaces (`Cin > 4`) use a
    /// different addressing path this defect has not been shown to reach.
    /// RKNN's dense int8 tensors use a padded physical row pitch, making its
    /// captured boundaries aligned. The host ABI is compact NHWC, so it must
    /// instead move boundaries according to the compact stride here until
    /// [`Shape`] can represent an explicit physical input pitch.
    /// Whether a dense-layout tile whose feature fetch starts at input row
    /// `in_first` is safe for general, non-uniform tensor data.
    ///
    /// Measured on real RK3588 hardware, not derived from documentation
    /// (`iree-rocket-design-spike`'s `conv_dense_shared_buffer_dispatch_hw.rs`,
    /// `conv_dense_odd_in_first_probe_hw.rs`,
    /// `conv_dense_alignment_width_sweep_hw.rs`, and
    /// `conv_dense_alignment_channel_sweep_hw.rs`/
    /// `conv_dense_alignment_in_first_growth_hw.rs` -- see DESIGN_NOTES.md
    /// there, "The dense (Cin<=4) ARGB path silently corrupts multi-row
    /// dispatches" and its follow-ups, for the full characterization).
    ///
    /// `CNA_FEATURE_DATA_ADDR` is `in_first * input_row_stride`, and dense
    /// mode's `input_row_stride` is not always a multiple of 16. Earlier
    /// hardware probes filled every x position and input channel alike and
    /// concluded that `nonalign_dma` safely compensates for offsets up to
    /// one dense pixel wide. That conclusion was an artifact of the data:
    /// a sub-pixel/channel displacement is invisible when all displaced
    /// values are equal.
    ///
    /// `conv_features0_exact_hw.rs` uses x-, y-, and channel-varying data
    /// at VGG-19 `features.0`'s exact lowered shape. Every tile at offset 0
    /// passed exactly, while all three tiles at offset 4 were about 94%
    /// wrong across their complete output ranges, deterministically in all
    /// three repetitions. A subsequent data-rich hardware sweep tested every
    /// even byte offset: all 14 offset-0 cases passed, while every case at
    /// offsets 2, 4, 6, 8, 10, 12, and 14 failed. The affine-int8 Cartesian
    /// oracle subsequently exposed the same defect at offset 2 for VGG-19's
    /// lowered 226x226/Cin=3 shape: tile 0 passed exactly and corruption began
    /// at tile 1's first output row. Dense tiles therefore require a fully
    /// 16-byte-aligned feature base at both precisions.
    ///
    /// Always `true` outside dense layout: surfaces (`Cin > 4`) use a
    /// different addressing path this defect has not been shown to reach.
    /// RKNN's dense int8 tensors use a padded physical row pitch, making its
    /// captured boundaries aligned. The host ABI is compact NHWC, so it must
    /// instead move boundaries according to the compact stride here until
    /// [`Shape`] can represent an explicit physical input pitch.
    pub fn dense_feature_offset_safe(&self, in_first: u32) -> bool {
        if self.layout() != FeatureLayout::Dense {
            return true;
        }
        let offset = (in_first * self.input_row_stride()) % FEATURE_ATOM_BYTES;
        offset == 0
    }

    /// The input row a tile's feature fetch starts at, for an output range
    /// beginning at `out_first` -- the half of [`Tile::from_bounds`]'s
    /// formula that determines [`Shape::dense_feature_offset_safe`], broken
    /// out so a boundary search can probe it without building a whole
    /// [`Tile`].
    /// The input row a tile's feature fetch starts at, for an output range
    /// beginning at `out_first` -- the half of [`Tile::from_bounds`]'s
    /// formula that determines [`Shape::dense_feature_offset_safe`], broken
    /// out so a boundary search can probe it without building a whole
    /// [`Tile`].
    pub fn tile_in_first(&self, kernels: Kernels, out_first: u32) -> u32 {
        let padding = self.kernel_programming(kernels).pad_top;
        (out_first * self.stride).saturating_sub(padding)
    }

    /// Byte stride of one output row.
    ///
    /// Output geometry, not input: at stride greater than one the two differ.
    /// Every output cube here is 16-byte NC1HWC2 atoms except depthwise
    /// accumulator output; see [`Shape::output_channel_block_bytes`].
    /// Byte stride of one output row.
    ///
    /// Output geometry, not input: at stride greater than one the two differ.
    /// Every output cube here is 16-byte NC1HWC2 atoms except depthwise
    /// accumulator output; see [`Shape::output_channel_block_bytes`].
    pub fn output_row_stride(&self, kernels: Kernels) -> u32 {
        self.output_width(kernels) * self.output_channel_block_bytes()
    }

    /// Bytes occupied by one pixel in one hardware output-channel block.
    ///
    /// **16 bytes for dense accumulator output, i.e. an ordinary NC1HWC2
    /// atom holding C2 = 4 int32 lanes** -- the same cube
    /// `rockchip-npu-notes/encodings/tile-layouts.md` documents for an int32
    /// output (`C2 = 16 bytes / out-element bytes`), and the cube
    /// `rocket-userspace`'s `gen_matmul_int8` writes.
    ///
    /// This was 128 (CORE's 32-channel accumulator block) for as long as the
    /// dense accumulator drove the DPU's *serial* writer, `mc_surf_out = 1`.
    /// That writer is the one that truncates past ~384 coefficient bytes per
    /// output channel, and the 128-byte block was the readback model that made
    /// its output decodable. Both are gone together: the writer is now
    /// `mc_surf_out = 0` / `size_e = 7` / `surf_add = dataout * 8`, and this is
    /// the cube it produces [HW sweep, planck 2026-09-03,
    /// `accumulator_size_e_probe` with `ROCKET_ACC_LAYOUT_SCAN=1`, which scores
    /// C2=4 surface-major at 100.0% of lanes and every other candidate at
    /// 32-38%].
    ///
    /// Depthwise accumulator output keeps the serial writer and its 128-byte
    /// programmed block (256-byte write atom, see
    /// [`Shape::output_atom_bytes`]): the change above is measured on dense
    /// shapes only.
    /// Bytes occupied by one pixel in one hardware output-channel block.
    ///
    /// **16 bytes for dense accumulator output, i.e. an ordinary NC1HWC2
    /// atom holding C2 = 4 int32 lanes** -- the same cube
    /// `rockchip-npu-notes/encodings/tile-layouts.md` documents for an int32
    /// output (`C2 = 16 bytes / out-element bytes`), and the cube
    /// `rocket-userspace`'s `gen_matmul_int8` writes.
    ///
    /// This was 128 (CORE's 32-channel accumulator block) for as long as the
    /// dense accumulator drove the DPU's *serial* writer, `mc_surf_out = 1`.
    /// That writer is the one that truncates past ~384 coefficient bytes per
    /// output channel, and the 128-byte block was the readback model that made
    /// its output decodable. Both are gone together: the writer is now
    /// `mc_surf_out = 0` / `size_e = 7` / `surf_add = dataout * 8`, and this is
    /// the cube it produces [HW sweep, planck 2026-09-03,
    /// `accumulator_size_e_probe` with `ROCKET_ACC_LAYOUT_SCAN=1`, which scores
    /// C2=4 surface-major at 100.0% of lanes and every other candidate at
    /// 32-38%].
    ///
    /// Depthwise accumulator output keeps the serial writer and its 128-byte
    /// programmed block (256-byte write atom, see
    /// [`Shape::output_atom_bytes`]): the change above is measured on dense
    /// shapes only.
    pub fn output_channel_block_bytes(&self) -> u32 {
        if self.precision.writes_accumulators() && self.depthwise {
            self.precision.out_channel_granule() * self.precision.output_element_bytes()
        } else {
            FEATURE_ATOM_BYTES
        }
    }

    /// Bytes one pixel occupies in one hardware output atom, as the host must
    /// read the result back.
    ///
    /// Deliberately distinct from [`Shape::output_channel_block_bytes`],
    /// which is what the DPU is *programmed* with. The two agree everywhere
    /// except depthwise accumulator output, where the write atom is twice the
    /// programmed block: the depthwise DPU processes 64 int8 channels per
    /// pass -- the same 64-channel coefficient group
    /// `pack_depthwise_to_rocket_weights` writes -- and emits all 64 i32
    /// lanes of a pixel contiguously, so its surface stride advances every
    /// 256 bytes rather than every 128.
    ///
    /// Established on RK3588 with an identity depthwise filter, whose output
    /// must equal its input: the returned block permutation is reproduced
    /// exactly, at 32x32x64, 32x32x128 and 17x13x64, by "surface-major over
    /// 256-byte atoms" and by nothing else. The dense accumulator path was
    /// measured the same way at the identical shape (32x32, Cin = Cout = 64,
    /// 1x1, so the *only* difference is `depthwise`) and is exact with the
    /// 128-byte atom, which is why this is conditioned on `depthwise` rather
    /// than widened for all accumulator output.
    /// Bytes one pixel occupies in one hardware output atom, as the host must
    /// read the result back.
    ///
    /// Deliberately distinct from [`Shape::output_channel_block_bytes`],
    /// which is what the DPU is *programmed* with. The two agree everywhere
    /// except depthwise accumulator output, where the write atom is twice the
    /// programmed block: the depthwise DPU processes 64 int8 channels per
    /// pass -- the same 64-channel coefficient group
    /// `pack_depthwise_to_rocket_weights` writes -- and emits all 64 i32
    /// lanes of a pixel contiguously, so its surface stride advances every
    /// 256 bytes rather than every 128.
    ///
    /// Established on RK3588 with an identity depthwise filter, whose output
    /// must equal its input: the returned block permutation is reproduced
    /// exactly, at 32x32x64, 32x32x128 and 17x13x64, by "surface-major over
    /// 256-byte atoms" and by nothing else. The dense accumulator path was
    /// measured the same way at the identical shape (32x32, Cin = Cout = 64,
    /// 1x1, so the *only* difference is `depthwise`) and is exact with the
    /// 128-byte atom, which is why this is conditioned on `depthwise` rather
    /// than widened for all accumulator output.
    pub fn output_atom_bytes(&self) -> u32 {
        if self.depthwise && self.precision.writes_accumulators() {
            return 2
                * self.precision.out_channel_granule()
                * self.precision.output_element_bytes();
        }
        self.output_channel_block_bytes()
    }

    /// Contraction depth one output channel accumulates over, which is what
    /// the streamed coefficient working set scales with.
    ///
    /// `Cin` for a dense convolution: every output channel's filter spans all
    /// input channels, so a streamed group of output channels reserves one
    /// 64-byte coefficient group per `(kernel tap, Cin)` -- the model in
    /// [`streamed_weight_bank_preference_for_group`].
    ///
    /// **One for depthwise**, and that is the whole of the difference. A
    /// depthwise output channel accumulates over exactly one input channel, so
    /// its filter is `kh * kw` values rather than `kh * kw * Cin`, and the
    /// whole weight tensor is `C * kh * kw` bytes -- at most a bank, which is
    /// what `rockchip-npu-notes/encodings/cbuf-bank-slack.md` means by "its
    /// weight is one per-channel `KH*KW*G`-byte cube (<= 1 bank)". Feeding the
    /// dense product here instead made the working set scale with `C` and
    /// refused wide depthwise outright: at C=1344, k=3 it asked for 13 of the
    /// eleven grantable banks, which is what kept MobileNetV2's 528/816/1344
    /// depthwise stages on the CPU.
    /// Contraction depth one output channel accumulates over, which is what
    /// the streamed coefficient working set scales with.
    ///
    /// `Cin` for a dense convolution: every output channel's filter spans all
    /// input channels, so a streamed group of output channels reserves one
    /// 64-byte coefficient group per `(kernel tap, Cin)` -- the model in
    /// [`streamed_weight_bank_preference_for_group`].
    ///
    /// **One for depthwise**, and that is the whole of the difference. A
    /// depthwise output channel accumulates over exactly one input channel, so
    /// its filter is `kh * kw` values rather than `kh * kw * Cin`, and the
    /// whole weight tensor is `C * kh * kw` bytes -- at most a bank, which is
    /// what `rockchip-npu-notes/encodings/cbuf-bank-slack.md` means by "its
    /// weight is one per-channel `KH*KW*G`-byte cube (<= 1 bank)". Feeding the
    /// dense product here instead made the working set scale with `C` and
    /// refused wide depthwise outright: at C=1344, k=3 it asked for 13 of the
    /// eleven grantable banks, which is what kept MobileNetV2's 528/816/1344
    /// depthwise stages on the CPU.
    pub fn streamed_contraction_channels(&self) -> u32 {
        if self.depthwise {
            1
        } else {
            self.weight_channels()
        }
    }

    /// CBUF banks the feature data would take if nothing competed for them.
    ///
    /// Derived from 134 captured programs across 11 distinct `(width,
    /// height)` shapes. Deliberately uncapped: a demand above the 12 banks
    /// that exist is meaningful, because it is what makes the weights the
    /// smaller claim in [`data_banks`].
    /// CBUF banks the feature data would take if nothing competed for them.
    ///
    /// Derived from 134 captured programs across 11 distinct `(width,
    /// height)` shapes. Deliberately uncapped: a demand above the 12 banks
    /// that exist is meaningful, because it is what makes the weights the
    /// smaller claim in [`data_banks`].
    pub fn data_bank_demand(&self) -> u32 {
        match self.layout() {
            // Dense CBUF rows round the spatial width to one feature atom.
            // Their pixel charge follows the ARGB storage class: 1, 2, 4,
            // or 4 channels.
            FeatureLayout::Dense => {
                (self.cbuf_input_width(self.width) * self.height * self.dense_cbuf_pixel_bytes())
                    .div_ceil(8 * PIXELS_PER_BANK_STEP)
            }
            // Surfaces charge per atom, at twice the pixels per step. Fits
            // every measured point: at 32x32 fp16 this is `ceil(atoms / 2)`,
            // giving 1,1,2,2,3,3,4,4,5,5 across atom counts 1 through 10.
            //
            // The atom count is `cbuf_atoms`, the count the CBUF actually
            // bills, not `weight_atoms`. The two agree at fp16 -- there
            // `weight_channels` is already quad-rounded, so `weight_atoms`
            // *is* `cbuf_atoms`, which is why using either reproduced every
            // fp16 capture -- but they part company at int8, whose
            // `weight_channels` is exact. Billing the exact count there
            // under-grants a data bank at `Cin` 33..48, 97..112, 225..240
            // and every 64 thereafter, and the tile then reads past the
            // resident window: the last input rows are silently dropped and
            // the corresponding output rows come back wrong. Measured on
            // RK3588 at Cin 48, 112 and 240, where the byte shortfall
            // predicts 3.88, 3.88 and 0.12 rows and the hardware loses
            // exactly 4, 4 and 1.
            FeatureLayout::Surfaces => {
                (self.width * self.height * self.cbuf_atoms()).div_ceil(2 * PIXELS_PER_BANK_STEP)
            }
        }
        .max(1)
    }

    /// CBUF banks the weights would take if nothing competed for them.
    ///
    /// Weights are streamed rather than held resident -- a 512-kernel
    /// program with 589824 bytes of coefficients runs with 8 banks, which
    /// hold 262144 -- so exceeding the CBUF is not an error. Like the data
    /// demand, this stays uncapped so the comparison below can see it.
    /// CBUF banks the weights would take if nothing competed for them.
    ///
    /// Weights are streamed rather than held resident -- a 512-kernel
    /// program with 589824 bytes of coefficients runs with 8 banks, which
    /// hold 262144 -- so exceeding the CBUF is not an error. Like the data
    /// demand, this stays uncapped so the comparison below can see it.
    pub fn weight_bank_demand(&self, kernels: Kernels) -> u32 {
        self.weight_bytes(kernels).div_ceil(CBUF_BANK_BYTES).max(1)
    }

    pub fn demand_based_cbuf_partition(&self, kernels: Kernels) -> (u32, u32) {
        self.try_demand_based_cbuf_partition(kernels)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    /// [`Shape::demand_based_cbuf_partition`], returning the refusal instead
    /// of panicking.
    pub fn try_demand_based_cbuf_partition(
        &self,
        kernels: Kernels,
    ) -> Result<(u32, u32), PlanError> {
        let data = self.data_bank_demand();
        let weights = self.weight_bank_demand(kernels);
        let granted = if data <= weights {
            data
        } else {
            data.min(CBUF_BANKS.saturating_sub(weights))
        };
        let data_banks = granted.clamp(1, CBUF_BANKS - 1);
        let weight_banks = CBUF_BANKS - data_banks;
        // Not clamped to the grantable count. A clamp makes the correction
        // below test `weight_banks > streamed_preference` as `11 > 11`, which
        // never fires, and the split degenerates to a single data bank --
        // roughly one input row per tile. Refusing is the honest answer for a
        // region no capture covers.
        //
        // A two-pass reading of the k=3 curve (`streamed / 2 + 1` above the
        // grant that still leaves three data banks) fits all 13 `Cin` points
        // from 384 to 768 and was tried here. It is wrong: a k=5 sweep over
        // the same `Cin` range shows a multi-pass sawtooth that resets twice
        // (weight banks run 10,8,7,7,8,10 then 7,7,8,9,10,11 then
        // 6,6,7,7,8,8,9,9,10,10,10,11), and the vendor uses 1/11 and 2/10
        // freely there -- so a single-data-bank split is not inherently wrong
        // and the "leaves three data banks" threshold does not survive a
        // second kernel size. The k=3 fit would have mis-planned k=5 `Cin` 192
        // as 6/6 against the vendor's 2/10.
        let streamed_preference = streamed_weight_bank_preference(
            self.streamed_contraction_channels(),
            kernels,
            self.precision.element_bits(),
        );
        refuse_unless!(
            streamed_preference < CBUF_BANKS,
            PlanErrorCode::UnvalidatedConfiguration,
            "coefficient working set wants {streamed_preference} CBUF banks for \
             {}x{} kernel and {} weight channels, more than the {} grantable; the \
             CBUF partition above that point is not capture-backed (see \
             MAX_INPUT_CHANNELS' doc comment)",
            kernels[0],
            kernels[1],
            self.weight_channels(),
            CBUF_BANKS - 1
        );
        let floor = weight_banks_floor(self.weight_channels()).max(streamed_preference);

        // Total coefficient size can otherwise consume eleven banks and
        // leave only one for feature data, even though CNA streams weights
        // in the bounded working set above. Once data has actually been
        // starved to that single-bank minimum, cap coefficients at the
        // streamed grant and return the unused banks to data.
        //
        // The grant is `floor`, not `streamed_preference`, even though the
        // trigger above compares against `streamed_preference`: hardware
        // confirmed (`bank_partition_flip_boundary_probe_runs_every_case_before_failing`)
        // that granting exactly `streamed_preference` here reads back zero
        // past whatever channel count fit in that many banks -- e.g. Cin 3/4/5
        // K3 fp16 has `streamed_preference = 1` but needs the same `floor = 3`
        // every other capture in this Cin range is validated against.
        // `streamed_preference` staying the trigger is deliberate: below it,
        // the earlier `granted = data` computation already grants at least
        // `floor` banks on its own, so there is nothing to correct.
        if data_banks == 1 && weight_banks > streamed_preference && weights > streamed_preference {
            let weight_banks = floor.min(CBUF_BANKS - 1);
            return Ok((CBUF_BANKS - weight_banks, weight_banks));
        }

        // A coefficient footprint that would take weight_banks_floor's banks
        // or fewer on its own (`weights <= weight_banks`) is not being
        // starved by this grant -- it fits, same as any other capture, and
        // is left alone. One that has been clamped down below the floor
        // despite wanting more (`weights > weight_banks`) is the case
        // weight_banks_floor's doc comment covers: raise it to the floor
        // and let feature data give up the difference, trading tile count
        // for correctness. `weights` is already known unbounded here, so
        // the floor itself, not `weights`, is always what gets granted.
        if weight_banks < floor && weights > weight_banks {
            let weight_banks = floor.min(CBUF_BANKS - 1);
            return Ok((CBUF_BANKS - weight_banks, weight_banks));
        }

        // The 32x32/Cin128/Cout64/3x3 capture is the one measured case where
        // honoring coefficient demand would force a spatial split, while
        // starving it by one bank keeps the whole image resident. The vendor
        // chooses 8/4 instead of the demand-only 7/5. Express that as the
        // observable policy: preserve a single spatial tile when granting the
        // whole-map data demand can do so without crossing the validated
        // coefficient-bank floor.
        let whole_data_banks = data.min(CBUF_BANKS - 1);
        let whole_weight_banks = CBUF_BANKS - whole_data_banks;
        let whole_rows = Tile::whole(*self, kernels).in_rows;
        let baseline_fits = whole_rows <= self.max_tile_input_rows_for_data_banks(data_banks);
        let whole_data_fits =
            whole_rows <= self.max_tile_input_rows_for_data_banks(whole_data_banks);
        if whole_data_banks > data_banks
            && whole_weight_banks >= floor
            && !baseline_fits
            && whole_data_fits
        {
            Ok((whole_data_banks, whole_weight_banks))
        } else {
            Ok((data_banks, weight_banks))
        }
    }

    /// CBUF banks the vendor assigns to feature data.
    ///
    /// The two claimants split all 12 banks: every capture satisfies
    /// `data_banks + weight_banks == 12`. When they fit together each takes
    /// what it asked for; when they do not, the *smaller* claim is honoured
    /// in full and the larger takes the remainder. Both halves of that are
    /// measured -- 256x32 Cin 32 Cout 64 wants 16 data banks and 2 weight
    /// banks and is programmed 10/2, while 32x32 Cin 64 Cout 512 wants 4 and
    /// 18 and is programmed 4/8.
    ///
    /// This is the only path by which `Cout` reaches the bank split, and it
    /// opens only once the feature data is already over budget. The corpus
    /// shows it exactly once: at 256x32 with Cin 32, Cout 16 needs 9216
    /// bytes of coefficients and takes one bank, Cout 64 needs 36864 and
    /// takes two, pushing the data allocation from 11 banks down to 10.
    /// CBUF banks the vendor assigns to feature data.
    ///
    /// The two claimants split all 12 banks: every capture satisfies
    /// `data_banks + weight_banks == 12`. When they fit together each takes
    /// what it asked for; when they do not, the *smaller* claim is honoured
    /// in full and the larger takes the remainder. Both halves of that are
    /// measured -- 256x32 Cin 32 Cout 64 wants 16 data banks and 2 weight
    /// banks and is programmed 10/2, while 32x32 Cin 64 Cout 512 wants 4 and
    /// 18 and is programmed 4/8.
    ///
    /// This is the only path by which `Cout` reaches the bank split, and it
    /// opens only once the feature data is already over budget. The corpus
    /// shows it exactly once: at 256x32 with Cin 32, Cout 16 needs 9216
    /// bytes of coefficients and takes one bank, Cout 64 needs 36864 and
    /// takes two, pushing the data allocation from 11 banks down to 10.
    pub fn data_banks(&self, kernels: Kernels) -> u32 {
        assert_default_cbuf_kernel(kernels);
        self.demand_based_cbuf_partition(kernels).0
    }

    /// CBUF banks the vendor assigns to weights: everything left over.
    /// CBUF banks the vendor assigns to weights: everything left over.
    pub fn weight_banks(&self, kernels: Kernels) -> u32 {
        CBUF_BANKS - self.data_banks(kernels)
    }

    /// Most input rows that fit in an explicit feature-data bank allocation.
    ///
    /// This is the capacity half of [`max_tile_input_rows`], exposed for the
    /// large-kernel hardware probe whose CBUF partition comes from the
    /// focused `(Cin, Cout, k)` capture sweep rather than the 1x1/3x3
    /// automatic allocator.
    /// Most input rows that fit in an explicit feature-data bank allocation.
    ///
    /// This is the capacity half of [`max_tile_input_rows`], exposed for the
    /// large-kernel hardware probe whose CBUF partition comes from the
    /// focused `(Cin, Cout, k)` capture sweep rather than the 1x1/3x3
    /// automatic allocator.
    pub fn max_tile_input_rows_for_data_banks(&self, data_banks: u32) -> u32 {
        self.max_tile_input_rows_for_width_and_data_banks(self.width, data_banks)
    }

    /// Most input rows that fit when a task reads only `input_width`.
    ///
    /// Horizontal tiling changes the resident CBUF footprint without
    /// changing the tensor's memory strides. The three large-kernel captures
    /// that require it sit exactly on this product bound:
    /// `input_width * input_rows * atoms * 16 <= data_banks * 32768`.
    ///
    /// **Zero when not even one row fits.** This used to end in `.max(1)`,
    /// which forced a single row through whatever the arithmetic said, and
    /// the hardware then read the line's tail from the wrong place: a 3x3 at
    /// 86x1 `Cin` 768 fp16 is granted 4 data banks by the coefficient floor,
    /// its one line is 2064 entries against the 2048 they hold, and it
    /// computed wrong values on `planck` 2026-09-05 while 85x1 (2040
    /// entries) was exact and the same 86x1 at 6/6 was exact. Returning zero
    /// lets `plan_grid` decline the width, which is what sends the plan to
    /// column tiles. See ISSUES.md C10.
    /// Most input rows that fit when a task reads only `input_width`.
    ///
    /// Horizontal tiling changes the resident CBUF footprint without
    /// changing the tensor's memory strides. The three large-kernel captures
    /// that require it sit exactly on this product bound:
    /// `input_width * input_rows * atoms * 16 <= data_banks * 32768`.
    ///
    /// **Zero when not even one row fits.** This used to end in `.max(1)`,
    /// which forced a single row through whatever the arithmetic said, and
    /// the hardware then read the line's tail from the wrong place: a 3x3 at
    /// 86x1 `Cin` 768 fp16 is granted 4 data banks by the coefficient floor,
    /// its one line is 2064 entries against the 2048 they hold, and it
    /// computed wrong values on `planck` 2026-09-05 while 85x1 (2040
    /// entries) was exact and the same 86x1 at 6/6 was exact. Returning zero
    /// lets `plan_grid` decline the width, which is what sends the plan to
    /// column tiles. See ISSUES.md C10.
    pub fn max_tile_input_rows_for_width_and_data_banks(
        &self,
        input_width: u32,
        data_banks: u32,
    ) -> u32 {
        assert!(
            (1..CBUF_BANKS).contains(&data_banks),
            "data banks must be between 1 and {}, leaving at least one weight bank",
            CBUF_BANKS - 1
        );
        assert!(
            (1..=self.width).contains(&input_width),
            "tile input width must be between 1 and the tensor width {}",
            self.width
        );
        let charged_width = self.cbuf_input_width(input_width);
        let capacity = match self.layout() {
            FeatureLayout::Dense => {
                data_banks * (CBUF_BANK_BYTES / 4) / (charged_width * self.dense_cbuf_pixel_bytes())
            }
            // `cbuf_atoms`, not `weight_atoms`, for the reason spelled out
            // in `data_bank_demand`: the two differ only at int8, and only
            // where the exact count is one short of a multiple of four.
            FeatureLayout::Surfaces => {
                // Whole entries, not bare atoms. Charging the unrounded atom
                // count lets a tile claim more rows than actually fit
                // whenever a row's atoms are not a multiple of four, and the
                // overflow lands on the tail of the tile's last input row --
                // which shows up as the last output row of the tile being
                // wrong from some column onwards, with every earlier row
                // exact. Hardware-confirmed on 13 geometries either side of
                // the boundary, including the two tightest (Cin=16 fp16,
                // 115x113 fits at 5626 entries and passes, 113x113 wanted
                // 5643 against the 5632 available and fails).
                let entries_per_row =
                    (input_width * self.cbuf_atoms()).div_ceil(CBUF_ATOMS_PER_ENTRY);
                data_banks * CBUF_BANK_BYTES
                    / (entries_per_row * CBUF_ATOMS_PER_ENTRY * FEATURE_ATOM_BYTES)
            }
        };
        capacity.min(self.max_data_entries() / charged_width)
    }

    /// Widest input row one task may read at this depth.
    ///
    /// The last entry slab's base, `(slabs - 1) * in_cols`, has to fit
    /// [`MAX_ENTRY_SLAB_BASE`]. Dense rows and a depth of one slab are
    /// unbounded here and take the tensor width; the row count does not
    /// enter, because the base is an offset within a line.
    /// Widest input row one task may read at this depth.
    ///
    /// The last entry slab's base, `(slabs - 1) * in_cols`, has to fit
    /// [`MAX_ENTRY_SLAB_BASE`]. Dense rows and a depth of one slab are
    /// unbounded here and take the tensor width; the row count does not
    /// enter, because the base is an offset within a line.
    pub fn max_tile_input_width(&self) -> u32 {
        let extra_slabs = match self.layout() {
            FeatureLayout::Dense => 0,
            FeatureLayout::Surfaces => self.cbuf_atoms().div_ceil(CBUF_ATOMS_PER_ENTRY) - 1,
        };
        MAX_ENTRY_SLAB_BASE
            .checked_div(extra_slabs)
            .map_or(self.width, |bound| bound.min(self.width))
    }

    /// Conservative input-row limit imposed by `feature_grains`.
    ///
    /// The first tile carries the full top padding and therefore has the
    /// largest value: `in_rows + kernel_height + pad_top`. Continuation
    /// tiles can fit at least as many rows, so using the first-tile bound for
    /// every tile keeps the planner simple and guarantees encodability.
    /// Conservative input-row limit imposed by `feature_grains`.
    ///
    /// The first tile carries the full top padding and therefore has the
    /// largest value: `in_rows + kernel_height + pad_top`. Continuation
    /// tiles can fit at least as many rows, so using the first-tile bound for
    /// every tile keeps the planner simple and guarantees encodability.
    pub fn max_feature_grain_input_rows(&self, kernels: Kernels) -> u32 {
        let kernel = self.kernel_programming(kernels);
        MAX_FEATURE_GRAINS
            .saturating_sub(kernel.height + kernel.pad_top)
            .max(1)
    }

    /// Most input rows one program may read.
    ///
    /// Two bounds apply and the CBUF one is usually tighter. The hard limit
    /// is the observed 15-bit `CNA_CBUF_CON1.data_entries` field. Dense rows
    /// charge `rows * atom_aligned_width`; surface rows use their atom count.
    ///
    /// The CBUF bound is the inverse of [`data_bank_demand`]: a bank holds
    /// 1024 dense pixels or 2048 pixel-atoms, so the rows that fit are
    /// whatever the banks granted can carry *at this shape's cost per
    /// pixel*. Charging one atom per pixel unconditionally -- which is what
    /// this did while every capture backing it had `Cin = 3` -- is right in
    /// the dense regime and over-optimistic by the surface atom count in
    /// the surface one. `conv_outchannel_hw` caught it at 256x32 with
    /// `Cin = 32`, where the old rule allowed 44 rows against a real
    /// capacity of 22 and the tile silently lost its last input rows.
    ///
    /// Predicts the vendor's own largest single-core tile in all 17 corpus
    /// captures that split, against 12 for the atom-blind version: 32 rows
    /// at 256 wide with `Cin` 3, 22 at 512, 14 at 768, 11 at 1024, 7 at
    /// 1536, and 22 at 256 wide with `Cin` 32 -- dropping to 20 when a
    /// larger `Cout` takes a second bank for coefficients.
    ///
    /// Hardware tolerates the looser bound in the dense regime
    /// (`conv_wide_shape_hw` passes at ~33 rows on a 256-wide map, above the
    /// vendor's 32), so there it is conservatism rather than a correctness
    /// requirement -- the same pattern as `feature_grains`. In the surface
    /// regime it is a correctness requirement.
    /// Most input rows one program may read.
    ///
    /// Two bounds apply and the CBUF one is usually tighter. The hard limit
    /// is the observed 15-bit `CNA_CBUF_CON1.data_entries` field. Dense rows
    /// charge `rows * atom_aligned_width`; surface rows use their atom count.
    ///
    /// The CBUF bound is the inverse of [`data_bank_demand`]: a bank holds
    /// 1024 dense pixels or 2048 pixel-atoms, so the rows that fit are
    /// whatever the banks granted can carry *at this shape's cost per
    /// pixel*. Charging one atom per pixel unconditionally -- which is what
    /// this did while every capture backing it had `Cin = 3` -- is right in
    /// the dense regime and over-optimistic by the surface atom count in
    /// the surface one. `conv_outchannel_hw` caught it at 256x32 with
    /// `Cin = 32`, where the old rule allowed 44 rows against a real
    /// capacity of 22 and the tile silently lost its last input rows.
    ///
    /// Predicts the vendor's own largest single-core tile in all 17 corpus
    /// captures that split, against 12 for the atom-blind version: 32 rows
    /// at 256 wide with `Cin` 3, 22 at 512, 14 at 768, 11 at 1024, 7 at
    /// 1536, and 22 at 256 wide with `Cin` 32 -- dropping to 20 when a
    /// larger `Cout` takes a second bank for coefficients.
    ///
    /// Hardware tolerates the looser bound in the dense regime
    /// (`conv_wide_shape_hw` passes at ~33 rows on a 256-wide map, above the
    /// vendor's 32), so there it is conservatism rather than a correctness
    /// requirement -- the same pattern as `feature_grains`. In the surface
    /// regime it is a correctness requirement.
    pub fn max_tile_input_rows(&self, kernels: Kernels) -> u32 {
        self.max_tile_input_rows_for_data_banks(self.data_banks(kernels))
            .min(self.max_feature_grain_input_rows(kernels))
    }

    /// Fewest tiles this shape must be split into to stay encodable.
    ///
    /// A stride-1 tile producing `r` output rows reads up to
    /// `r + kernel_height - 1` input rows once its halo is counted, so the
    /// tap span is charged here rather than discovered as an overflow inside
    /// the builder.
    /// Fewest tiles this shape must be split into to stay encodable.
    ///
    /// A stride-1 tile producing `r` output rows reads up to
    /// `r + kernel_height - 1` input rows once its halo is counted, so the
    /// tap span is charged here rather than discovered as an overflow inside
    /// the builder.
    pub fn min_tiles(&self, kernels: Kernels) -> u32 {
        self.min_tiles_for_data_banks(kernels, self.data_banks(kernels))
    }

    /// Fewest output-row tiles needed with an explicit feature-data split.
    /// Fewest output-row tiles needed with an explicit feature-data split.
    pub fn min_tiles_for_data_banks(&self, kernels: Kernels, data_banks: u32) -> u32 {
        self.min_tiles_for_width_and_data_banks(kernels, self.width, data_banks)
    }

    /// Fewest output-row tiles needed at an explicit input-tile width.
    /// Fewest output-row tiles needed at an explicit input-tile width.
    pub fn min_tiles_for_width_and_data_banks(
        &self,
        kernels: Kernels,
        input_width: u32,
        data_banks: u32,
    ) -> u32 {
        // A tile's un-clipped tap span is `(out_rows - 1) * stride + kh`.
        // For the old odd SAME-padded case `kh - 1 == 2 * pad_top`; using
        // the kernel extent is the general form and remains conservative at
        // image edges for even and explicit-padding kernels.
        let halo = self.kernel_programming(kernels).height - 1;
        let rows = self
            .max_tile_input_rows_for_width_and_data_banks(input_width, data_banks)
            .min(self.max_feature_grain_input_rows(kernels))
            .saturating_sub(halo)
            .max(1);
        // A tile of `r` output rows reads about `r * stride` input rows.
        let output_rows = rows.div_ceil(self.stride).max(1);
        self.output_height(kernels).div_ceil(output_rows)
    }
}

/// The kernel's two extents and the leading padding supplied by the model.
///
/// The extents are kept apart because a non-square kernel moves them apart.
/// Every kernel capture before the rectangular sweep was square, so the two
/// were never observed differing and a single `size` sufficed. A sweep of 53
/// non-square captures (633 convolution programs, 28 rectangular shapes)
/// separates them: `weight_height` and `weight_width` carry the kernel's own
/// height and width with no swap, `pad_left` follows the width alone,
/// `pad_top` and `feature_grains` follow the height alone, and the
/// coefficient footprint is `kh * kw * pad(Cin) * element_bytes`. All 228
/// programs outside the high-pressure regime satisfy every one of those.
///
/// The even-kernel sweep separates the padding from the extent: both extents
/// are programmed verbatim, while `pad_top` and `pad_left` independently
/// carry the model's values. They therefore enter this structure rather than
/// being reconstructed from `kernel / 2`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KernelProgramming {
    pub height: u32,
    pub width: u32,
    /// Zero-padded rows above the first output row, `kh / 2`.
    pub pad_top: u32,
    /// Zero-padded columns left of the first output column, `kw / 2`.
    pub pad_left: u32,
}

pub fn kernel_programming(kernels: Kernels, padding: Option<Padding>) -> KernelProgramming {
    try_kernel_programming(kernels, padding).unwrap_or_else(|error| panic!("{error}"))
}

/// [`kernel_programming`], returning the refusal instead of panicking.
pub fn try_kernel_programming(
    kernels: Kernels,
    padding: Option<Padding>,
) -> Result<KernelProgramming, PlanError> {
    let [height, width] = kernels;
    let backed = |extent: usize| (1..=11).contains(&extent);
    refuse_unless!(
        backed(height) && backed(width),
        PlanErrorCode::UnvalidatedConfiguration,
        "conv_2d only has vendor reference data for kernel extents from 1 through 11, \
         got {height}x{width}"
    );
    let [pad_top, pad_left] = padding.unwrap_or([height / 2, width / 2]);
    refuse_unless!(
        pad_top < height && pad_left < width,
        PlanErrorCode::InvalidShape,
        "padding must be smaller than its kernel extent; got padding \
         {pad_top}x{pad_left} for kernel {height}x{width}"
    );
    Ok(KernelProgramming {
        height: height as u32,
        width: width as u32,
        pad_top: pad_top as u32,
        pad_left: pad_left as u32,
    })
}

pub fn assert_default_cbuf_kernel(kernels: Kernels) {
    assert!(
        matches!(kernels, [1, 1] | [3, 3]),
        "automatic CBUF allocation only has runtime backing for 1x1 and 3x3; \
         use ConvPlan, or conv_2d_tile_with_cbuf_banks for an explicit override"
    );
}

/// One contiguous range of output rows, with the input rows it reads.
///
/// `in_rows` counts input rows actually fetched from memory: it includes the
/// halo row a continuation tile reads from its neighbour, and excludes rows
/// supplied by zero padding. `pad_top` is the part of the kernel's top
/// padding still visible at this output range. It is usually nonzero only
/// for the first tile, but a large kernel split into very short tiles can
/// leave the second or later tile inside the image's top-padding region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tile {
    pub out_first: u32,
    pub out_rows: u32,
    pub in_first: u32,
    pub in_rows: u32,
    pub pad_top: u32,
}

impl Tile {
    /// Splits `shape` into `tiles` output-row ranges.
    ///
    /// At the captured `32x32` geometry this reproduces the vendor's own
    /// splits: `tiles = 1` gives 32 rows, `tiles = 2` gives 16+16, and
    /// `tiles = 3` gives 11+11+10, matching captured groups 1, 2-3, and 4-6.
    pub fn split(shape: Shape, kernels: Kernels, tiles: u32) -> Vec<Tile> {
        let output_height = shape.output_height(kernels);
        assert!(
            (1..=output_height).contains(&tiles),
            "tile count must be between 1 and the {output_height} output rows"
        );
        let base = output_height / tiles;
        let remainder = output_height % tiles;

        let mut out = Vec::with_capacity(tiles as usize);
        let mut out_first: u32 = 0;
        for index in 0..tiles {
            let out_rows = base + u32::from(index < remainder);
            out.push(Tile::from_bounds(shape, kernels, out_first, out_rows));
            out_first += out_rows;
        }
        out
    }

    /// Greedily fills each row tile to `max_input_rows`, leaving any short
    /// remainder in the last tile.
    ///
    /// This is the vendor's standalone-plan policy. For example, a 226-row
    /// fp16 shape with room for 48 input rows becomes output rows
    /// `47+46+46+46+41`, while an int8 capacity of 93 produces
    /// `92+91+43`. [`Tile::split`] remains the balanced primitive used when
    /// an explicit number of parallel-core partitions is requested.
    pub fn split_greedy_to_capacity(
        shape: Shape,
        kernels: Kernels,
        max_input_rows: u32,
    ) -> Option<Vec<Tile>> {
        let output_height = shape.output_height(kernels);
        let whole = Tile::from_bounds(shape, kernels, 0, output_height);
        if whole.in_rows <= max_input_rows {
            return Some(vec![whole]);
        }

        let kernel = shape.kernel_programming(kernels);
        let mut tiles = Vec::new();
        let mut out_first = 0;
        while out_first < output_height {
            let remaining = output_height - out_first;
            let tile = (1..=remaining).rev().find_map(|out_rows| {
                let tile = Tile::from_bounds(shape, kernels, out_first, out_rows);

                // Once a map needs more than one task, RKNN fixes each
                // task's output grain from the full, unclipped kernel span.
                // In particular it does not use bottom-edge clipping to
                // merge the final grain into its predecessor: at K3/P1 a
                // 15-row capacity is 14+13+1, not 14+14, and a 3-row
                // capacity is 2+1+...+1, not 2+1+...+2. Top padding still
                // reduces the first grain's fetched span.
                let in_first = shape.tile_in_first(kernels, out_first);
                let last_tap = (out_first + out_rows - 1) * shape.stride + kernel.height - 1;
                let unclipped_in_last = last_tap.saturating_sub(kernel.pad_top);
                let capacity_rows = (unclipped_in_last - in_first + 1).max(out_rows * shape.stride);
                (capacity_rows <= max_input_rows).then_some(tile)
            })?;
            out_first += tile.out_rows;
            tiles.push(tile);
        }
        Some(tiles)
    }

    /// Builds one tile from an explicit output-row range, applying the same
    /// halo/padding formula every tile in a [`Tile::split`] plan uses.
    ///
    /// Broken out of `split` so `realign_dense_row_tiles` can rebuild an
    /// individual tile after moving its boundary, without duplicating this
    /// arithmetic -- the two must stay in exact agreement, since a
    /// realigned tile has to be bit-identical to what `split` itself would
    /// have produced had it picked that boundary in the first place.
    pub fn from_bounds(shape: Shape, kernels: Kernels, out_first: u32, out_rows: u32) -> Tile {
        let kernel = shape.kernel_programming(kernels);
        let padding = kernel.pad_top;
        let stride = shape.stride;

        // Halo: the first input row a tile touches is its first output row
        // projected back through the stride, less the padding it would
        // otherwise read above the image. Matches all 150 stride-2, -3 and
        // -4 programs in the corpus.
        let in_first = shape.tile_in_first(kernels, out_first);
        let last_tap = (out_first + out_rows - 1) * stride + kernel.height - 1;
        let in_last = last_tap.saturating_sub(padding).min(shape.height - 1);
        let exact = in_last - in_first + 1;

        // The vendor reads at least a full stride block per output row,
        // which exceeds the exact tap span at stride > 1. Taking the larger
        // of the two is safe by construction: it is never below `exact`, so
        // every tap the tile needs is resident. Where the corpus disagrees
        // it reads more still, which costs DMA rather than correctness.
        let in_rows = exact.max(out_rows * stride).min(shape.height - in_first);

        let projected_first = out_first * stride;
        Tile {
            out_first,
            out_rows,
            in_first,
            in_rows,
            pad_top: padding.saturating_sub(projected_first),
        }
    }

    /// The single tile covering the whole image.
    pub fn whole(shape: Shape, kernels: Kernels) -> Tile {
        Tile::split(shape, kernels, 1)[0]
    }

    /// Byte offset of this tile's first input row from the tensor base.
    pub fn input_offset(&self, shape: Shape) -> u32 {
        self.in_first * shape.input_row_stride()
    }

    /// Byte offset of this tile's first output row from the tensor base.
    pub fn output_offset(&self, shape: Shape, kernels: Kernels) -> u32 {
        self.out_first * shape.output_row_stride(kernels)
    }
}

/// One contiguous range of output columns and the input columns it reads.
///
/// This is the horizontal analogue of [`Tile`]. `in_cols` includes the
/// overlap with neighbouring column tiles and excludes columns supplied by
/// zero padding. The tensor's row and surface strides remain those of the
/// full [`Shape`]; only the task-local geometry and base offset change.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColumnTile {
    pub out_first: u32,
    pub out_cols: u32,
    pub in_first: u32,
    pub in_cols: u32,
    pub pad_left: u32,
}

impl ColumnTile {
    /// Derives one horizontal tile from an output-column range.
    pub fn from_output_range(
        shape: Shape,
        kernels: Kernels,
        out_first: u32,
        out_cols: u32,
    ) -> ColumnTile {
        let output_width = shape.output_width(kernels);
        assert!(
            out_cols > 0 && out_first + out_cols <= output_width,
            "tile output columns {out_first}..{} fall outside the {output_width}-column output",
            out_first + out_cols
        );
        // The horizontal analogue of `Tile::split`, so the horizontal padding
        // is the one that applies.
        let kernel = shape.kernel_programming(kernels);
        let projected_first = out_first * shape.stride;
        let in_first = projected_first.saturating_sub(kernel.pad_left);
        let last_tap = (out_first + out_cols - 1) * shape.stride + kernel.width - 1;
        let in_last = last_tap
            .saturating_sub(kernel.pad_left)
            .min(shape.width - 1);
        let exact = in_last - in_first + 1;
        // A tile that is the whole row takes the whole row, even when the
        // taps do not reach the end of it.
        //
        // `in_cols` is programmed as `CNA_DATA_SIZE0.datain_width`, and in
        // dense layout the CNA advances rows by that value rather than by
        // the separately programmed `line_stride`. At stride > 1 an extent
        // where `(width - kernel) % stride != 0` leaves a partial trailing
        // window no output tap consumes, so `exact` (and the vendor's
        // `out_cols * stride` floor) both land *short* of the real row
        // pitch -- and every row after the first is then read at the wrong
        // offset, which corrupts the entire output rather than just its
        // edge. Hardware-confirmed across 16 stride-2/3/4 geometries in
        // `dense_geometry_regression_cases`: the failures are exactly the
        // ones where this used to come out below `shape.width`.
        //
        // Only the un-partitioned case is widened. A genuine horizontal
        // partition programs grouped-line mode and a `surf_stride` that
        // already accounts for the local width, so its tiles keep the
        // narrower span they are supposed to have.
        let spans_full_row = out_first == 0 && out_first + out_cols == output_width;
        let in_cols = if spans_full_row {
            shape.width - in_first
        } else {
            exact
                .max(out_cols * shape.stride)
                .min(shape.width - in_first)
        };

        ColumnTile {
            out_first,
            out_cols,
            in_first,
            in_cols,
            pad_left: kernel.pad_left.saturating_sub(projected_first),
        }
    }

    /// Splits the output into explicitly sized column ranges.
    ///
    /// Explicit widths keep the capture-derived partition boundaries visible:
    /// 9x9/Cin64 uses 135+121, 11x11/Cin48 uses 137+119, and
    /// 11x11/Cin64 uses 59+54+54+54+35.
    pub fn split(shape: Shape, kernels: Kernels, output_widths: &[u32]) -> Vec<ColumnTile> {
        assert!(
            !output_widths.is_empty(),
            "at least one column tile is required"
        );
        assert_eq!(
            output_widths.iter().sum::<u32>(),
            shape.output_width(kernels),
            "column-tile widths must cover the output exactly"
        );
        let mut out_first = 0;
        output_widths
            .iter()
            .map(|&out_cols| {
                let tile = ColumnTile::from_output_range(shape, kernels, out_first, out_cols);
                out_first += out_cols;
                tile
            })
            .collect()
    }

    pub fn whole(shape: Shape, kernels: Kernels) -> ColumnTile {
        ColumnTile::from_output_range(shape, kernels, 0, shape.output_width(kernels))
    }

    pub fn input_offset(&self, shape: Shape) -> u32 {
        match shape.layout() {
            FeatureLayout::Dense => {
                self.in_first * shape.in_channels * shape.precision.element_bytes()
            }
            FeatureLayout::Surfaces => self.in_first * FEATURE_ATOM_BYTES,
        }
    }

    pub fn output_offset(&self, shape: Shape) -> u32 {
        self.out_first * shape.output_channel_block_bytes()
    }
}

/// A rectangular output tile with both vertical and horizontal input halos.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tile2D {
    pub rows: Tile,
    pub columns: ColumnTile,
}

impl Tile2D {
    pub fn whole(shape: Shape, kernels: Kernels) -> Tile2D {
        Tile2D {
            rows: Tile::whole(shape, kernels),
            columns: ColumnTile::whole(shape, kernels),
        }
    }

    /// Builds a rectangular grid using explicit output-column widths and the
    /// conservative row capacity for each resulting input width.
    pub fn grid(
        shape: Shape,
        kernels: Kernels,
        output_widths: &[u32],
        data_banks: u32,
    ) -> Vec<Tile2D> {
        let columns = ColumnTile::split(shape, kernels, output_widths);
        let mut tiles = Vec::new();
        for columns in columns {
            let row_tiles =
                shape.min_tiles_for_width_and_data_banks(kernels, columns.in_cols, data_banks);
            tiles.extend(
                Tile::split(shape, kernels, row_tiles)
                    .into_iter()
                    .map(|rows| Tile2D { rows, columns }),
            );
        }
        tiles
    }

    pub fn input_offset(&self, shape: Shape) -> u32 {
        self.rows.input_offset(shape) + self.columns.input_offset(shape)
    }

    pub fn output_offset(&self, shape: Shape, kernels: Kernels) -> u32 {
        self.rows.output_offset(shape, kernels) + self.columns.output_offset(shape)
    }
}

/// A complete standalone-job plan for one convolution.
///
/// The plan owns the policy that the low-level tile builders intentionally
/// leave with their caller: the CBUF split, horizontal partition (if any),
/// and the row split for each column. Programs returned by `programs()` (on the HAL's `ConvPlan`)
/// retain tile-relative buffer offsets and still need normal DMA relocation.
///
/// For 1x1 and 3x3 this is the existing demand-based allocator. Even kernels
/// use that allocator too, at stride 1, through the demands the even sweep
/// measures: eight banks square in both precisions, and with two even extents
/// six in fp16 and four in int8. The fp16
/// stride-1 5x5 through 11x11 policies are the conservative partitions from
/// the focused capture and hardware sweep. The three shapes that require
/// horizontal tiling retain their hardware-proven captured boundaries; other
/// surface-layout shapes fall back to the fewest balanced columns that
/// satisfy the same two-dimensional CBUF capacity bound. Those fallback
/// partitions are derived rather than hardware-validated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConvPlan {
    shape: Shape,
    kernels: Kernels,
    data_banks: u32,
    weight_banks: u32,
    output_column_widths: Vec<u32>,
    tiles: Vec<Tile2D>,
}

/// Logical destination and private-scratch range for one independently
/// staged accumulator-output tile.
///
/// The matching register program writes every channel surface contiguously
/// inside this range. Callers can therefore compact it into a dense NHWC
/// tensor without re-deriving the DPU's tile-local surface geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AccumulatorOutputTile {
    pub scratch_offset: usize,
    pub scratch_bytes: usize,
    pub output_row: usize,
    pub output_column: usize,
    pub output_rows: usize,
    pub output_columns: usize,
}

/// [`ConvPlan::accumulator_output_layout`]'s answer: one entry per tile and
/// the scratch total they partition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AccumulatorOutputLayout {
    pub tiles: Vec<AccumulatorOutputTile>,
    pub scratch_bytes: usize,
}

impl ConvPlan {
    /// Plans all standalone jobs needed to cover `shape` exactly once.
    ///
    /// Panics when the requested operation lies outside the supported policy.
    /// In particular, the capture-specific odd-square policies above 3x3
    /// require fp16 and stride 1, every even kernel requires stride 1, and
    /// NC1HWC2 input is required if horizontal tiling is necessary.
    /// Plans all standalone jobs needed to cover `shape` exactly once.
    ///
    /// Panics when the requested operation lies outside the supported policy.
    /// In particular, the capture-specific odd-square policies above 3x3
    /// require fp16 and stride 1, every even kernel requires stride 1, and
    /// NC1HWC2 input is required if horizontal tiling is necessary.
    pub fn new(shape: Shape, kernels: Kernels) -> ConvPlan {
        ConvPlan::try_new(shape, kernels).unwrap_or_else(|error| panic!("{error}"))
    }

    /// [`ConvPlan::new`], returning the planner's refusal instead of
    /// panicking. Validates the kernel against the padded extents and the
    /// coefficient footprint against the address space first, so that
    /// nothing below it can trip an arithmetic invariant on a malformed
    /// descriptor.
    pub fn try_new(shape: Shape, kernels: Kernels) -> Result<ConvPlan, PlanError> {
        let kernel = shape.try_kernel_programming(kernels)?;
        shape.check_extents_against(kernel)?;
        shape.try_weight_bytes(kernels)?;
        if kernel.height != kernel.width {
            return ConvPlan::new_with_cbuf_partition(
                shape,
                kernels,
                non_square_cbuf_partition(shape, kernels)?,
            );
        }
        let (data_banks, weight_banks) = match kernel.height {
            1 | 3 => shape.try_demand_based_cbuf_partition(kernels)?,
            2 | 4 | 6 | 8 | 10 => even_square_cbuf_partition(shape, kernels)?,
            5 => {
                check_large_kernel_plan_case(shape, 5, true)?;
                shape.try_demand_based_cbuf_partition(kernels)?
            }
            7 => {
                check_large_kernel_plan_case(shape, 7, true)?;
                // The focused sweep follows coefficient demand through seven
                // banks (1/11, 2/10, 8/4, 7/5 and 5/7 are all observed), then
                // switches to the streamed 8/4 schedule at demand ten.
                if shape.weight_bank_demand(kernels) <= 7 {
                    shape.try_demand_based_cbuf_partition(kernels)?
                } else {
                    unstarved_large_kernel_partition(shape, kernels, (8, 4))
                }
            }
            9 => {
                check_large_kernel_plan_case(shape, 9, false)?;
                let captured = if (33..=48).contains(&shape.in_channels) {
                    (7, 5)
                } else {
                    (6, 6)
                };
                unstarved_large_kernel_partition(shape, kernels, captured)
            }
            11 => {
                check_large_kernel_plan_case(shape, 11, false)?;
                let captured = match shape.in_channels {
                    1..=32 => (7, 5),
                    33..=48 => (5, 7),
                    _ => (3, 9),
                };
                unstarved_large_kernel_partition(shape, kernels, captured)
            }
            _ => {
                return Err(PlanError::new(
                    PlanErrorCode::Internal,
                    format!("kernel_programming accepted an unsupported kernel {kernels:?}"),
                ));
            }
        };
        ConvPlan::new_with_cbuf_partition(shape, kernels, (data_banks, weight_banks))
    }

    /// Plans `shape` against an explicit CBUF split.
    ///
    /// The split is the one piece of policy the capture corpus does not
    /// settle uniformly -- `non_square_cbuf_partition` documents where it
    /// stops following coefficient demand -- so this is the escape hatch for
    /// the shapes [`ConvPlan::new`] refuses. Everything downstream, row
    /// splitting and horizontal partitioning both, follows from the split.
    /// The two bank counts must be nonzero and sum to the RK3588's twelve
    /// CBUF banks.
    /// Plans `shape` against an explicit CBUF split.
    ///
    /// The split is the one piece of policy the capture corpus does not
    /// settle uniformly -- `non_square_cbuf_partition` documents where it
    /// stops following coefficient demand -- so this is the escape hatch for
    /// the shapes [`ConvPlan::new`] refuses. Everything downstream, row
    /// splitting and horizontal partitioning both, follows from the split.
    /// The two bank counts must be nonzero and sum to the RK3588's twelve
    /// CBUF banks.
    pub fn with_cbuf_banks(
        shape: Shape,
        kernels: Kernels,
        data_banks: u32,
        weight_banks: u32,
    ) -> ConvPlan {
        ConvPlan::try_with_cbuf_banks(shape, kernels, data_banks, weight_banks)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    /// [`ConvPlan::with_cbuf_banks`], returning the refusal instead of
    /// panicking.
    pub fn try_with_cbuf_banks(
        shape: Shape,
        kernels: Kernels,
        data_banks: u32,
        weight_banks: u32,
    ) -> Result<ConvPlan, PlanError> {
        refuse_unless!(
            data_banks > 0 && weight_banks > 0 && data_banks + weight_banks == CBUF_BANKS,
            PlanErrorCode::InvalidShape,
            "explicit CBUF partition must have nonzero data and weight banks summing to \
             {CBUF_BANKS}; got data={data_banks}, weights={weight_banks}"
        );
        let kernel = shape.try_kernel_programming(kernels)?;
        shape.check_extents_against(kernel)?;
        shape.try_weight_bytes(kernels)?;
        ConvPlan::new_with_cbuf_partition(shape, kernels, (data_banks, weight_banks))
    }

    fn new_with_cbuf_partition(
        shape: Shape,
        kernels: Kernels,
        (data_banks, weight_banks): (u32, u32),
    ) -> Result<ConvPlan, PlanError> {
        shape
            .parity_padded_shape(kernels)
            .map_err(|reason| PlanError::new(PlanErrorCode::UnsupportedSemantics, reason))?;
        let full_width = vec![shape.output_width(kernels)];
        if let Some(tiles) = plan_grid(shape, kernels, &full_width, data_banks) {
            return Ok(ConvPlan {
                shape,
                kernels,
                data_banks,
                weight_banks,
                output_column_widths: full_width,
                tiles,
            });
        }

        refuse_unless!(
            shape.layout() == FeatureLayout::Surfaces,
            PlanErrorCode::UnvalidatedConfiguration,
            "convolution needs horizontal tiling, which is only capture-backed for NC1HWC2 surfaces"
        );
        // A column partition at stride > 1 used to be refused here, on the
        // grounds that no vendor capture had one. It is measured now
        // (`planck`, 2026-09-06, `dtype_boundary_probe` with the stride
        // field): fp16 32x32 k=1 s=2 at `Cin` 1792/2304/3072/3584/4096 (32
        // to 64 tiles), int8 at 3584 and 4096, the `onehot` read map at
        // `Cout == Cin` 2048, and -- the case that actually exercises the
        // halo -- k=3 s=2 with padding at 160x8 `Cin` 512/768 fp16, 200x8
        // `Cin` 512 int8, plus the read map at 160x8 `Cin` = `Cout` = 512.
        // Every one exact.
        //
        // Lifting it matters more after the 3584 channel ceilings than
        // before them: `Shape::max_tile_input_width` falls as `Cin` rises
        // (18 pixels at fp16 `Cin` 3584 against 37 at 1792), so the widths
        // that need a column partition come down into the range real
        // strided convolutions occupy, and this refusal is a panic rather
        // than a fallback.

        if let Some(output_column_widths) = captured_column_partition(shape, kernels) {
            let tiles = Tile2D::grid(shape, kernels, &output_column_widths, data_banks);
            refuse_unless!(
                grid_fits(shape, kernels, &tiles, data_banks),
                PlanErrorCode::Internal,
                "captured column partition exceeds its measured CBUF capacity"
            );
            return Ok(ConvPlan {
                shape,
                kernels,
                data_banks,
                weight_banks,
                output_column_widths,
                tiles,
            });
        }

        let output_width = shape.output_width(kernels);
        for column_count in 2..=output_width {
            let output_column_widths = balanced_column_widths(output_width, column_count);
            if let Some(tiles) = plan_grid(shape, kernels, &output_column_widths, data_banks) {
                return Ok(ConvPlan {
                    shape,
                    kernels,
                    data_banks,
                    weight_banks,
                    output_column_widths,
                    tiles,
                });
            }
        }

        Err(PlanError::new(
            PlanErrorCode::CapacityExceeded,
            format!(
                "no standalone tile fits {shape:?} {kernels:?} with CBUF split \
                 {data_banks}/{weight_banks}"
            ),
        ))
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }

    pub fn kernels(&self) -> Kernels {
        self.kernels
    }

    pub fn data_banks(&self) -> u32 {
        self.data_banks
    }

    pub fn weight_banks(&self) -> u32 {
        self.weight_banks
    }

    pub fn output_column_widths(&self) -> &[u32] {
        &self.output_column_widths
    }

    pub fn tiles(&self) -> &[Tile2D] {
        &self.tiles
    }

    /// Where each tile's int32 accumulator output lands in one private
    /// scratch buffer: contiguous, disjoint ranges in tile order, summing
    /// to [`Shape::output_scratch_bytes`]. The HAL's
    /// `staged_accumulator_programs` emits one contiguous-tile program per
    /// entry; the compaction that follows reads this layout back.
    pub fn accumulator_output_layout(&self) -> Result<AccumulatorOutputLayout, PlanError> {
        refuse_unless!(
            self.shape.precision.writes_accumulators(),
            PlanErrorCode::UnsupportedSemantics,
            "staged accumulator output requires Int8Accumulator precision"
        );
        let overflow = |what: &str| {
            PlanError::new(
                PlanErrorCode::InvalidShape,
                format!(
                    "accumulator {what} overflow for {:?} {:?}",
                    self.shape, self.kernels
                ),
            )
        };

        let source_block_bytes = self.shape.output_atom_bytes() as usize;
        let padded_bytes_per_pixel = self.shape.padded_out_channels() as usize
            * self.shape.precision.output_element_bytes() as usize;
        let blocks_per_pixel = padded_bytes_per_pixel.div_ceil(source_block_bytes);
        let mut scratch_offset = 0usize;
        let mut tiles = Vec::with_capacity(self.tiles.len());
        for tile in &self.tiles {
            let tile_pixels = tile.rows.out_rows as usize * tile.columns.out_cols as usize;
            let scratch_bytes = tile_pixels
                .checked_mul(blocks_per_pixel)
                .and_then(|value| value.checked_mul(source_block_bytes))
                .ok_or_else(|| overflow("tile scratch size"))?;
            tiles.push(AccumulatorOutputTile {
                scratch_offset,
                scratch_bytes,
                output_row: tile.rows.out_first as usize,
                output_column: tile.columns.out_first as usize,
                output_rows: tile.rows.out_rows as usize,
                output_columns: tile.columns.out_cols as usize,
            });
            scratch_offset = scratch_offset
                .checked_add(scratch_bytes)
                .ok_or_else(|| overflow("scratch partition"))?;
        }
        refuse_unless!(
            scratch_offset == self.shape.output_scratch_bytes(self.kernels),
            PlanErrorCode::Internal,
            "staged accumulator tiles must partition the full scratch allocation: \
             {scratch_offset} bytes planned against {} allocated",
            self.shape.output_scratch_bytes(self.kernels)
        );
        Ok(AccumulatorOutputLayout {
            tiles,
            scratch_bytes: scratch_offset,
        })
    }
}

/// Guards the above-3x3 plan policies, which came from an fp16 capture
/// sweep.
///
/// `precision_neutral` says whether the policy this kernel takes is stated
/// in *bytes* or in *channels*, which is what decides whether the fp16
/// sweep's CBUF split carries over to the other rungs:
///
///   * 5x5 takes [`Shape::demand_based_cbuf_partition`] -- the same
///     function 1x1 and 3x3 use at every precision, whose demand comes from
///     [`Shape::weight_bytes`] and [`Shape::data_bank_demand`]. There is no
///     fp16-specific number anywhere in that path, so refusing the other
///     rungs was a gate on the sweep's precision rather than on anything
///     the policy does. Same for 7x7, whose one threshold is a
///     `weight_bank_demand` in banks of bytes.
///   * 9x9 and 11x11 instead key on `in_channels`, a channel *count* whose
///     byte footprint is four times larger at tf32 than at int4. Those
///     tables cannot be reinterpreted at another width without measuring,
///     so they stay fp16.
///
/// Stride stays 1 everywhere above 3x3: the sweep has no strided capture at
/// any kernel size, and that gap is precision-independent.
///
/// The `Cin` ceiling is [`large_kernel_max_in_channels`], which is where
/// the hardware measurement lives.
fn check_large_kernel_plan_case(
    shape: Shape,
    kernel: usize,
    precision_neutral: bool,
) -> Result<(), PlanError> {
    // Characterization needs to reach exactly the shapes this refuses: every
    // refusal below is a record of what hardware does, and the only way to
    // improve that record is to build the refused shape and read what comes
    // back. This lifts all three -- precision, stride and `Cin` -- because
    // each of them was written from a measurement that stopped where the
    // instrument stopped. See `large_kernel_probing_allowed`.
    if policy::large_kernel_probing_allowed() {
        return Ok(());
    }
    refuse_unless!(
        precision_neutral || matches!(shape.precision, Precision::Fp16),
        PlanErrorCode::UnvalidatedConfiguration,
        "automatic planning at this kernel size currently has capture backing \
         only for fp16"
    );
    refuse_unless!(
        shape.stride == 1,
        PlanErrorCode::UnvalidatedConfiguration,
        "automatic planning above 3x3 currently has capture backing only at stride 1"
    );
    let ceiling = large_kernel_max_in_channels(shape.precision, kernel);
    refuse_unless!(
        shape.in_channels <= ceiling,
        PlanErrorCode::UnvalidatedConfiguration,
        "{kernel}x{kernel} {:?} is measured correct only to Cin {ceiling}, not \
         {}; above it the NPU hangs and the watchdog kills the job",
        shape.precision,
        shape.in_channels
    );
    Ok(())
}

/// A capture-derived split for a kernel above 3x3, raised if it would starve
/// the coefficient stream.
///
/// The splits above 3x3 are read straight off the fp16 capture sweep -- 7x7's
/// `(8, 4)` fallback, 9x9's `Cin`-keyed table, 11x11's -- and unlike
/// [`Shape::demand_based_cbuf_partition`] they never consulted
/// [`streamed_weight_bank_preference`]. That is what the `Cin` cliff was: at
/// 7x7 fp16 `Cin` 72 the hardcoded `(8, 4)` grants four coefficient banks
/// where the streamed working set needs seven, the stream starves, and the
/// watchdog kills the job at ~500 ms. It is the same fault the tf32 k=3 hang
/// turned out to be, and the same fix -- a grant, not a size, is what breaks.
///
/// Measured on `planck` 2026-09-07 by forcing every partition at the first
/// hanging shape with `ROCKET_CBUF_SPLIT`: 7x7 fp16 `Cin` 72 is **exact at
/// 1/11 through 7/5 and hangs at 8/4, 9/3, 10/2 and 11/1**. Five coefficient
/// banks are enough there and four are not, so the cliff was never a hardware
/// ceiling on `Cin` -- it was this policy handing out four banks regardless of
/// what the stream asked for. C9 recorded it as "not a CBUF-split artifact"
/// on the strength of *which* split the planner chose at each kernel size,
/// which is not the same question as what happens when the split is forced.
///
/// The capture's own weight grant is kept as the lower bound: the preference
/// is conservative rather than tight (`Cin` 64 wants seven banks and is exact
/// with four), so this only ever raises, never lowers, and every shape the
/// captures already validate keeps a grant at least as large as the one it
/// was validated with.
pub fn unstarved_large_kernel_partition(
    shape: Shape,
    kernels: Kernels,
    captured: (u32, u32),
) -> (u32, u32) {
    let preference = streamed_weight_bank_preference(
        shape.streamed_contraction_channels(),
        kernels,
        shape.precision.element_bits(),
    );
    let weight_banks = captured.1.max(preference).min(CBUF_BANKS - 1);
    (CBUF_BANKS - weight_banks, weight_banks)
}

/// Largest `Cin` a kernel above 3x3 is measured correct at, per precision.
///
/// Measured on `planck` with `dtype_boundary_probe`, one shape per case (the
/// sweep contaminates itself) with a canary between runs. The boundary moves
/// with neither extent nor `Cout`: 7x7 fp16 `Cin` 208 is exact at extents 8x8,
/// 16x16 and 32x32 and at `Cout` 64, 128 and 256, and 7x7 int8 `Cin` 208 is
/// exact at `Cout` 256.
///
///   7x7   fp16/bf16/int16/fp16-acc/int8/int8-acc  208 exact, 224 wrong or hangs
///   7x7   int4                                    224 exact, 256 hangs
///   7x7   tf32                                     96 exact, 128 wrong
///   9x9   fp16                                    128 exact, 144 wrong
///   11x11 fp16                                     64 exact,  96 hangs
///   5x5   fp16/bf16/int8 to `Cin` 320, tf32 to 192: all exact
///
/// **These are ~3x the ceilings this table carried until 2026-09-07, and the
/// difference is a planner fix, not new hardware.** The old numbers -- 64 for
/// the 2-byte family, 32 for tf32, 128 for int4 -- were the point at which a
/// starved coefficient grant hung the NPU. The splits above 3x3 are read off
/// the fp16 capture sweep and, unlike `demand_based_cbuf_partition`, never
/// consulted `streamed_weight_bank_preference`; 7x7's hardcoded `(8, 4)` gave
/// four coefficient banks no matter what the stream asked for.
/// `unstarved_large_kernel_partition` raises the grant to the streamed
/// preference and every one of those hangs became exact. What is left is a
/// genuine boundary: at 7x7 `Cin` 224 **no** CBUF partition works -- 1/11
/// computes wrong values and the other ten hang -- so unlike the old cliff it
/// does not move when the split is forced.
///
/// The old "the ceilings fit `Cin * element_bytes` at 128 or 64 bytes" reading
/// is **withdrawn**. It was a fit to the starvation boundary, which was a
/// property of our grant; the real ceilings sit at one `Cin` for six
/// precisions of four different widths, which is not a byte budget at all.
///
/// 5x5 is deliberately absent: nothing at 5x5 has failed yet, at any width or
/// precision. 9x9 and 11x11 carry fp16 only -- `check_large_kernel_plan_case`
/// admits nothing else there -- so their rows say nothing about the rest.
///
/// **int8 used to be refused above 3x3 outright, and that refusal was wrong.**
/// It rested on 5x5 and 7x7 returning the same value in every output channel
/// of a pixel at `Cin` 16, 32 and 64 alike -- read at the time as coefficients
/// not reaching their channels. The cause was the instrument: on 2026-09-04
/// `dtype_boundary_probe` had no `SelectorsAffine` branch, so every int8 case
/// fell through to `Selectors`, a signed coefficient form that is not the int8
/// ABI and says nothing about the device. Re-measured, the same binary fails
/// that way at 1x1 and 3x3 too, where int8 is known exact to `Cin` 512 -- the
/// fault never had a kernel-size dependence, and only looked like one because
/// 1x1 and 3x3 were gated by the ladders, which pass the affine pattern, and
/// never went through the probe.
pub fn large_kernel_max_in_channels(precision: Precision, kernel: usize) -> u32 {
    match precision {
        // 5x5 has no measured ceiling at any width; the CBUF planner's own
        // refusal is what bounds it.
        _ if kernel <= 5 => u32::MAX,
        // 9x9 and 11x11 admit fp16 only, so one number each covers them.
        _ if kernel == 9 => 128,
        _ if kernel == 11 => 64,
        Precision::Tf32 => 96,
        Precision::Int4 => 224,
        Precision::Fp16
        | Precision::Fp16Accumulator
        | Precision::Bf16
        | Precision::Int16
        | Precision::Int8(_)
        | Precision::Int8Accumulator(_) => 208,
    }
}

/// Largest coefficient demand at which a non-square kernel's CBUF split is
/// still the demand-based one.
///
/// Every non-square capture at or below this, in *both* precisions, takes
/// exactly the partition [`Shape::demand_based_cbuf_partition`] computes: all
/// 28 rectangular shapes at `Cin` 3, and the 256x32 `Cin` 32 captures whose
/// demand is one to five banks. The first disagreement is at seven.
const MAX_NON_SQUARE_DEMAND_BASED_WEIGHT_BANKS: u32 = 5;

/// Largest coefficient demand for which the even square captures follow the
/// demand-based CBUF split.
///
/// The even sweep reaches eight banks at 8x8 and agrees exactly. Eight is
/// also where it stops, which the fill-in row measures directly rather than
/// leaving to the gap between 8x8 and 10x10: holding the kernel at 8x8 and
/// walking `Cout` through 72, 80, 88, 96 and 104 steps the demand through
/// 9..=13, and every one of those five is captured 8/4 where the demand rule
/// asks for 3/9, 2/10 and 1/11.
///
/// This also settles what the lone 10x10 capture meant. It was read as an
/// extent the vendor treats differently; it is not. A 10x10 at `Cout` 24 and
/// 32 -- demands 5 and 7 -- takes the demand-based split exactly, so 10x10
/// plans unaided below the ceiling like any other even square, and the 5/7 at
/// `Cout` 64 is the demand-13 behaviour rather than a property of the kernel.
///
/// Holds in both precisions: every int8 even square at or below eight matches
/// too.
const MAX_EVEN_SQUARE_DEMAND_BASED_WEIGHT_BANKS: u32 = 8;

/// Largest coefficient demand for which a kernel with two even extents
/// follows the demand-based CBUF split, in fp16.
///
/// Separate from [`MAX_NON_SQUARE_DEMAND_BASED_WEIGHT_BANKS`] because the odd
/// corpus and the even one disagree about where demand stops deciding, and
/// each measures its own parity. The even pressure row runs 4x8/8x4 at four
/// banks, 4x10/10x4 at five and 6x8/8x6 at six, and every one of those takes
/// the demand-based split.
///
/// Six is where they stop, and the fill-in row measures the stop rather than
/// assuming it: at eight banks the mirrored pair 6x10 / 10x6 splits 4/8
/// against 8/4. So the orientation asymmetry the odd rectangles show at seven
/// and eight is *not* absent from even extents -- it starts one demand step
/// later. In both parities the member that departs from demand is the taller
/// of the pair, and in both the corpus offers no rule for which way it goes.
const MAX_EVEN_NON_SQUARE_FP16_DEMAND_BASED_WEIGHT_BANKS: u32 = 6;

/// The same bound in int8, where it is lower.
///
/// int8 leaves the demand rule earlier and less tidily. At `Cin` 32, `Cout`
/// 128 the mirrored pairs 4x8/8x4 (demand 4) match, 6x8/8x6 (six) are
/// captured 8/4 where demand asks 6/6, 6x10/10x6 (eight) match again, and
/// 8x10/10x8 (ten) split 2/10 against 7/5. Agreement is not monotone in
/// demand, so the last demand below the first disagreement is the only bound
/// the data supports, and that is four.
///
/// That a lower bound is needed at all was found by measurement, not
/// prediction: the fp16 fill-in row was what prompted running the same
/// comparison in int8, and the int8 disagreement at six sits below the fp16
/// bound of six.
const MAX_EVEN_NON_SQUARE_INT8_DEMAND_BASED_WEIGHT_BANKS: u32 = 4;

fn even_square_cbuf_partition(shape: Shape, kernels: Kernels) -> Result<(u32, u32), PlanError> {
    refuse_unless!(
        shape.stride == 1,
        PlanErrorCode::UnvalidatedConfiguration,
        "even kernels currently have capture backing only at stride 1"
    );
    let demand = shape.weight_bank_demand(kernels);
    refuse_unless!(
        demand <= MAX_EVEN_SQUARE_DEMAND_BASED_WEIGHT_BANKS,
        PlanErrorCode::UnvalidatedConfiguration,
        "even square kernel {kernels:?} needs {demand} coefficient banks, above the \
         {MAX_EVEN_SQUARE_DEMAND_BASED_WEIGHT_BANKS} where the captured split follows \
         coefficient demand; use an explicit CBUF split"
    );
    shape.try_demand_based_cbuf_partition(kernels)
}

/// CBUF split for a non-square kernel.
///
/// The rectangular sweep shows the vendor's split is not a function of
/// coefficient demand alone. At 256x32, `Cin` 32, `Cout` 64 fp16 the mirrored
/// pairs 5x11/11x5, 7x9/9x7 and 9x11/11x9 each share a demand and each split
/// differently, with the taller kernel of every pair landing on 8/4 while the
/// wider one keeps its coefficient claim.
///
/// The int8 sweep separates demand from precision. An int8 coefficient is one
/// byte, so the same geometries ask for half the banks and stay demand-based;
/// doubling `Cout` to 128 restores the fp16 demands exactly, and there the
/// split leaves the demand rule too. So the break follows coefficient demand
/// rather than precision -- but *how* it breaks does not: at matched demand
/// all three int8 mirrored pairs split symmetrically (8/4, 8/4, 5/7) where no
/// fp16 pair does. Whatever carries kernel height into the fp16 policy does
/// not survive quantization, and no capture in either corpus isolates it.
///
/// Below the disagreement the question does not arise, and there the
/// 1x1/3x3 allocator is exact in both precisions. Above it this refuses
/// rather than guesses; `conv_2d_tile_with_cbuf_banks` and
/// [`ConvPlan::with_cbuf_banks`] take an explicit split.
///
/// A kernel with two even extents is bounded by its own corpus instead, and
/// per precision. The odd disagreement is not evidence about shapes the even
/// sweep measured directly, and refusing a captured 6x8 because an uncaptured
/// 5x11 misbehaves would be reading one parity's policy off the other's. The
/// even bounds are lower than the even square path's eight because a mirrored
/// pair can disagree where a square has no mirror to disagree with.
fn non_square_cbuf_partition(shape: Shape, kernels: Kernels) -> Result<(u32, u32), PlanError> {
    refuse_unless!(
        shape.stride == 1,
        PlanErrorCode::UnvalidatedConfiguration,
        "non-square kernels currently have capture backing only at stride 1"
    );
    let kernel = shape.try_kernel_programming(kernels)?;
    let both_even = kernel.height.is_multiple_of(2) && kernel.width.is_multiple_of(2);
    let (limit, parity) = match (both_even, shape.precision) {
        (true, Precision::Fp16) => (MAX_EVEN_NON_SQUARE_FP16_DEMAND_BASED_WEIGHT_BANKS, "even"),
        (true, _) => (MAX_EVEN_NON_SQUARE_INT8_DEMAND_BASED_WEIGHT_BANKS, "even"),
        (false, _) => (MAX_NON_SQUARE_DEMAND_BASED_WEIGHT_BANKS, "non-square"),
    };
    let demand = shape.weight_bank_demand(kernels);
    refuse_unless!(
        demand <= limit,
        PlanErrorCode::UnvalidatedConfiguration,
        "{parity} kernel {kernels:?} needs {demand} coefficient banks, above the \
         {limit} where the captured split stops following coefficient demand; \
         use an explicit CBUF split"
    );
    shape.try_demand_based_cbuf_partition(kernels)
}

fn captured_column_partition(shape: Shape, kernels: Kernels) -> Option<Vec<u32>> {
    let focused_shape = shape.width == 256
        && shape.height == 32
        && shape.stride == 1
        && shape.out_channels == 64
        && matches!(shape.precision, Precision::Fp16)
        && shape.padding.is_none();
    if !focused_shape {
        return None;
    }
    // Keyed on the whole kernel, not just its height: these boundaries were
    // captured at 9x9 and 11x11, and a 9x3 shares neither their coefficient
    // footprint nor their halo.
    match (kernels, shape.in_channels) {
        ([9, 9], 64) => Some(vec![135, 121]),
        ([11, 11], 48) => Some(vec![137, 119]),
        ([11, 11], 64) => Some(vec![59, 54, 54, 54, 35]),
        _ => None,
    }
}

pub fn balanced_column_widths(output_width: u32, columns: u32) -> Vec<u32> {
    let base = output_width / columns;
    let remainder = output_width % columns;
    (0..columns)
        .map(|index| base + u32::from(index < remainder))
        .collect()
}

fn plan_grid(
    shape: Shape,
    kernels: Kernels,
    output_widths: &[u32],
    data_banks: u32,
) -> Option<Vec<Tile2D>> {
    let columns = ColumnTile::split(shape, kernels, output_widths);
    let mut tiles = Vec::new();
    for columns in columns {
        // A column too wide for its slab base has no row split that saves
        // it; declining here is what sends the caller to a finer partition.
        if columns.in_cols > shape.max_tile_input_width() {
            return None;
        }
        let max_rows = shape
            .max_tile_input_rows_for_width_and_data_banks(columns.in_cols, data_banks)
            .min(shape.max_feature_grain_input_rows(kernels));
        let greedy = Tile::split_greedy_to_capacity(shape, kernels, max_rows)?;
        let row_tiles =
            realign_dense_row_tiles(shape, kernels, &greedy, max_rows).or_else(|| {
                // Compact fp16 dense rows can force a boundary away from the
                // vendor's physically padded position. Preserve the existing
                // safe fallback there if the greedy boundary cannot be moved
                // without overflowing either neighbour.
                (1..=shape.output_height(kernels)).find_map(|count| {
                    let rows = Tile::split(shape, kernels, count);
                    if !rows.iter().all(|tile| tile.in_rows <= max_rows) {
                        return None;
                    }
                    realign_dense_row_tiles(shape, kernels, &rows, max_rows)
                })
            })?;
        tiles.extend(row_tiles.into_iter().map(|rows| Tile2D { rows, columns }));
    }
    Some(tiles)
}

/// For dense-layout convolutions, nudges `tiles`' interior row boundaries so
/// every tile's `in_first` is safe against `nonalign_dma`'s leading-pixel
/// defect ([`Shape::dense_feature_offset_safe`]), re-deriving each affected
/// tile from its shifted boundary via [`Tile::from_bounds`]. A no-op --
/// `Some(tiles.to_vec())` -- outside dense layout and for a single-tile
/// plan (whose only boundary, row 0, is always safe: `in_first` there is
/// always 0 regardless of padding or stride, and offset 0 trivially passes
/// [`Shape::dense_feature_offset_safe`]).
///
/// Boundaries are searched outward from their original position, closest
/// first, bounded by how much room the two neighbouring tiles have to give
/// up (each must keep at least one output row). `max_rows` re-gates
/// capacity after a shift, since moving a boundary changes both
/// neighbours' `in_rows`, not just the moved one's.
///
/// `None` if some interior boundary has no safe, capacity-respecting
/// position within that room -- the caller's existing
/// retry-with-more-tiles loop (`plan_grid`) is what handles that, exactly
/// as it already does for a plain capacity miss; more tiles means less
/// room per boundary here, not more, so this is not expected to resolve on
/// a later retry in general, and a shape that never finds a fit will
/// surface as [`ConvPlan::new`]'s existing "needs horizontal tiling" panic
/// for dense layout -- refusing outright rather than emitting a plan with
/// a known-unsafe tile, matching this file's existing policy for the
/// `weight_banks < 3` and `Cin <= 4` bugs above it.
///
/// RKNN's int8 corpus pads dense rows to a precision-sized spatial atom
/// (226 -> 240), but the host ABI is compact NHWC. The data-rich int8 oracle
/// confirmed that a compact row at a nonzero offset corrupts exactly like
/// fp16, so int8 is intentionally realigned here as well. This can differ
/// from a vendor boundary that is safe only under RKNN's padded row pitch.
fn realign_dense_row_tiles(
    shape: Shape,
    kernels: Kernels,
    tiles: &[Tile],
    max_rows: u32,
) -> Option<Vec<Tile>> {
    if shape.layout() != FeatureLayout::Dense || tiles.len() <= 1 {
        return Some(tiles.to_vec());
    }

    let output_height = shape.output_height(kernels);
    let mut boundaries: Vec<u32> = tiles.iter().map(|tile| tile.out_first).collect();
    boundaries.push(output_height);

    for i in 1..boundaries.len() - 1 {
        let lower = boundaries[i - 1] + 1;
        let upper = boundaries[i + 1] - 1;
        if lower > upper {
            return None;
        }
        let current = boundaries[i];
        if shape.dense_feature_offset_safe(shape.tile_in_first(kernels, current)) {
            continue;
        }

        let max_distance = (current - lower).max(upper - current);
        let mut best = None;
        for distance in 1..=max_distance {
            let candidates = [current.checked_sub(distance), current.checked_add(distance)];
            for candidate in candidates.into_iter().flatten() {
                if candidate < lower || candidate > upper {
                    continue;
                }
                if shape.dense_feature_offset_safe(shape.tile_in_first(kernels, candidate)) {
                    best = Some(candidate);
                    break;
                }
            }
            if best.is_some() {
                break;
            }
        }
        boundaries[i] = best?;
    }

    let realigned: Vec<Tile> = boundaries
        .windows(2)
        .map(|window| Tile::from_bounds(shape, kernels, window[0], window[1] - window[0]))
        .collect();
    realigned
        .iter()
        .all(|tile| tile.in_rows <= max_rows)
        .then_some(realigned)
}

fn grid_fits(shape: Shape, kernels: Kernels, tiles: &[Tile2D], data_banks: u32) -> bool {
    tiles.iter().all(|tile| {
        tile.columns.in_cols <= shape.max_tile_input_width()
            && tile.rows.in_rows
                <= shape
                    .max_tile_input_rows_for_width_and_data_banks(tile.columns.in_cols, data_banks)
            && feature_grains(kernels, &tile.rows) <= MAX_FEATURE_GRAINS
    })
}

/// Builds the vendor-matching single-core regcmd program for the captured
/// `32x32x3 -> 32x32x8` fp16 convolution.
///
/// `kernels` must be `[1, 1]` or `[3, 3]`. Buffer-address fields are zero,
/// as they are in the captured RKNN regcmd blob; a caller that submits this
/// program must relocate the feature, weight, bias, and destination
/// addresses first.
///
/// The returned 136-word sequence is group 1 of the vendor blob. Groups
/// Output channels one BS block covers.
pub const BS_CHANNELS_PER_BLOCK: usize = 8;

/// Bytes one BS block occupies: eight `i32` biases, eight `i16` of a
/// constant, and eight `i16` multipliers, each plane padded to eight
/// channels whether or not they are all used.
pub const BS_BLOCK_BYTES: usize = 64;

/// The `i16` plane between the biases and the multipliers.
///
/// Constant at 128 in every model measured -- across `Cout` 4, 8 and 16,
/// uniform and per-channel weight magnitudes, zero and nonzero biases.
/// Nothing has been found that moves it, so no meaning is claimed for it
/// beyond "the value the vendor writes".
pub const BS_CONSTANT: i16 = 128;

/// Multiplier the channel carrying the largest weight scale is given.
///
/// The per-channel multipliers are `round(BS_UNIT_MULTIPLIER * scale[c] /
/// max(scale))`, so the widest channel gets exactly this and the rest scale
/// down from it. Reproduces every measured table exactly.
pub const BS_UNIT_MULTIPLIER: i16 = 1 << 14;

/// Right shift the BS stage applies to `accumulator * bs_multiplier`.
///
/// **Measured, not read off a register.** `DPU_BS_MUL_CFG.bs_mul_shift_value`
/// is 14 in every capture and the multiplier plane normalises to `2^14`,
/// which made 14 the obvious reading -- and it is wrong by a factor of 128.
/// `conv_int8_probe_hw` pins it two independent ways: holding the
/// accumulator at 1 and sweeping `out_cvt_shift`, the output reaches 1 at
/// 21, and `28 - 21 = 7`; holding the shift at 14 and sweeping the BS
/// multiplier, the output is 1 at 128 and doubles from there.
///
/// Nothing in the corpus could have shown this. The vendor never programs a
/// case where a wrong shift is observable, because whatever it does is
/// absorbed into `OUT_CVT` -- so the constant was unobservable in captures
/// and had to come from hardware.
///
/// One thing this does not explain: with the vendor's own values, a peak
/// plane entry of `2^14` and an `OUT_CVT` multiplier equal to the textbook
/// `input_scale * weight_scale / output_scale`, the composite comes out 128
/// times too large. Either a further divisor exists that has not been
/// found, or the vendor's plane feeds a stage this probe did not move. The
/// law below is validated where it was measured and is not a claim about
/// what the vendor's configuration means.
pub const BS_MULTIPLIER_SHIFT: u32 = 7;

/// Bytes the BS buffer occupies for `out_channels` output channels.
///
/// Prefer [`Shape::bs_buffer_bytes`], which passes the padded count. BRDMA
/// has been observed reading past what the true count declares.
pub fn bs_buffer_bytes(out_channels: u32) -> usize {
    (out_channels as usize).div_ceil(BS_CHANNELS_PER_BLOCK) * BS_BLOCK_BYTES
}

/// The `CNA_CONV_CON2.feature_grains` value this module programs.
///
/// Feature rows the CNA buffers before the convolution starts. A shape sweep
/// of 49 vendor captures shows the vendor does not use one formula: across
/// 297 programs it matches this prefetch value 63% of the time, uses exactly
/// `in_rows` 28% of the time, and goes *below* `in_rows` in 6% -- and no
/// register field in the corpus separates those cases, so the choice appears
/// to come from compiler state that never reaches the register program.
///
/// The TRM calls its own formula "suggested", which implies a range of valid
/// settings rather than one correct value, and this larger-than-vendor value
/// is the one the passing hardware tests use. `conv_grains_probe_hw` measures
/// the range that actually works.
/// The kernel term is its *height*, which the square corpus could not show:
/// across the rectangular sweep's 228 low-pressure programs the vendor's
/// value tracks `kernel_height` and is unchanged by `kernel_width`.
pub fn feature_grains(kernels: Kernels, tile: &Tile) -> u32 {
    tile.in_rows + kernel_programming(kernels, None).height + tile.pad_top
}

#[cfg(test)]
mod tests {
    use super::*;

    fn int8_accumulator(zero_point: i32) -> Precision {
        Precision::Int8Accumulator(Quantization {
            input_zero_point: zero_point,
            output_zero_point: 0,
            weight_zero_point: 0,
            input_scale: 1.0,
            weights_scale: 1.0,
            multiplier: Multiplier::for_unit_bs(1.0),
        })
    }

    fn code_of<T: std::fmt::Debug>(result: Result<T, PlanError>) -> PlanErrorCode {
        result.expect_err("expected a refusal").code()
    }

    // --- Shape construction refusals, one per code.

    #[test]
    fn zero_extents_and_stride_are_invalid_shapes() {
        assert_eq!(
            code_of(Shape::try_with_precision(0, 8, 1, 3, 8, Precision::Fp16)),
            PlanErrorCode::InvalidShape
        );
        assert_eq!(
            code_of(Shape::try_with_precision(8, 0, 1, 3, 8, Precision::Fp16)),
            PlanErrorCode::InvalidShape
        );
        assert_eq!(
            code_of(Shape::try_with_precision(8, 8, 0, 3, 8, Precision::Fp16)),
            PlanErrorCode::InvalidShape
        );
    }

    #[test]
    fn channels_past_the_corpus_are_unvalidated_not_invalid() {
        let error = Shape::try_with_precision(8, 8, 1, MAX_INPUT_CHANNELS + 1, 8, Precision::Fp16)
            .expect_err("above the ceiling");
        assert_eq!(error.code(), PlanErrorCode::UnvalidatedConfiguration);
        // The message is what the panicking constructor has always printed,
        // and what `#[should_panic(expected = ...)]` tests in the HAL match.
        assert!(
            error.message().contains("input channels must be 1..="),
            "{error}"
        );
        assert_eq!(format!("{error}"), error.message());
        assert_eq!(
            code_of(Shape::try_with_precision(
                8,
                8,
                1,
                3,
                MAX_OUTPUT_CHANNELS + 1,
                Precision::Fp16
            )),
            PlanErrorCode::UnvalidatedConfiguration
        );
        assert_eq!(
            code_of(Shape::try_with_precision(8, 8, 1, 40, 8, Precision::Int4)),
            PlanErrorCode::UnvalidatedConfiguration
        );
    }

    #[test]
    fn accumulator_zero_points_and_activations_are_unsupported_semantics() {
        assert_eq!(
            code_of(Shape::try_with_precision(
                8,
                8,
                1,
                16,
                16,
                int8_accumulator(1)
            )),
            PlanErrorCode::UnsupportedSemantics
        );
        let shape = Shape::try_with_precision(8, 8, 1, 16, 16, int8_accumulator(0)).unwrap();
        assert_eq!(
            code_of(shape.try_with_activation(Activation::Relu)),
            PlanErrorCode::UnsupportedSemantics
        );
        let shape = Shape::try_with_precision(8, 8, 1, 16, 32, Precision::Fp16).unwrap();
        assert_eq!(
            code_of(shape.try_with_depthwise()),
            PlanErrorCode::UnsupportedSemantics
        );
    }

    #[test]
    fn padding_past_the_register_field_is_a_hardware_limit() {
        let shape = Shape::try_with_precision(8, 8, 1, 3, 8, Precision::Fp16).unwrap();
        assert_eq!(
            code_of(shape.try_with_padding([16, 0])),
            PlanErrorCode::HardwareLimit
        );
        assert!(shape.try_with_padding([15, 15]).is_ok());
    }

    #[test]
    fn overflowing_surfaces_are_refused_without_panicking() {
        // 2^20 x 2^20 pixels x 16 bytes does not fit a u32 surface stride;
        // before the fallible constructor this overflowed in
        // `input_surface_stride` (a panic in debug, a wrap in release).
        let error = Shape::try_with_precision(1 << 20, 1 << 20, 1, 64, 64, Precision::Fp16)
            .expect_err("surface overflow");
        assert_eq!(error.code(), PlanErrorCode::HardwareLimit);
        assert!(error.message().contains("32-bit address space"), "{error}");
        // Just under it is accepted at construction.
        assert!(Shape::try_with_precision(1 << 14, 1 << 13, 1, 64, 64, Precision::Fp16).is_ok());
    }

    #[test]
    fn overflowing_coefficient_tensors_are_refused_without_panicking() {
        // 3584 x 3584 x 4 bytes x 11 x 11 = 6.2 GB of coefficients.
        let shape = Shape::try_with_precision(32, 32, 1, 3584, 3584, Precision::Tf32).unwrap();
        let error = shape
            .try_weight_bytes([11, 11])
            .expect_err("coefficient overflow");
        assert_eq!(error.code(), PlanErrorCode::HardwareLimit);
        assert_eq!(
            code_of(ConvPlan::try_new(shape, [11, 11])),
            PlanErrorCode::HardwareLimit
        );
        // The same shape at 1x1 is 51 MB and fine.
        assert_eq!(
            shape.try_weight_bytes([1, 1]).unwrap(),
            shape.weight_bytes([1, 1])
        );
    }

    // --- Kernel and plan refusals.

    #[test]
    fn kernel_refusals_name_the_defect() {
        let shape = Shape::try_with_precision(32, 32, 1, 3, 8, Precision::Fp16).unwrap();
        assert_eq!(
            code_of(ConvPlan::try_new(shape, [13, 13])),
            PlanErrorCode::UnvalidatedConfiguration
        );
        let padded = shape.try_with_padding([3, 3]).unwrap();
        assert_eq!(
            code_of(ConvPlan::try_new(padded, [3, 3])),
            PlanErrorCode::InvalidShape
        );
        // A kernel wider than its zero-padded input: 3 columns, no padding.
        let narrow = Shape::try_with_precision(2, 32, 1, 3, 8, Precision::Fp16)
            .unwrap()
            .try_with_padding([0, 0])
            .unwrap();
        let error = ConvPlan::try_new(narrow, [3, 3]).expect_err("kernel exceeds input");
        assert_eq!(error.code(), PlanErrorCode::InvalidShape);
        assert!(
            error.message().contains("exceeds the padded input width"),
            "{error}"
        );
    }

    #[test]
    fn even_kernels_at_stride_two_are_unvalidated() {
        let shape = Shape::try_with_precision(32, 32, 2, 3, 8, Precision::Fp16).unwrap();
        let error = ConvPlan::try_new(shape, [2, 2]).expect_err("even kernel, stride 2");
        assert_eq!(error.code(), PlanErrorCode::UnvalidatedConfiguration);
        assert!(
            error
                .message()
                .contains("even kernels currently have capture backing only at stride 1")
        );
    }

    #[test]
    fn explicit_cbuf_partitions_must_sum_to_the_bank_count() {
        let shape = Shape::try_with_precision(32, 32, 1, 3, 8, Precision::Fp16).unwrap();
        assert_eq!(
            code_of(ConvPlan::try_with_cbuf_banks(shape, [3, 3], 4, 4)),
            PlanErrorCode::InvalidShape
        );
        assert_eq!(
            code_of(ConvPlan::try_with_cbuf_banks(shape, [3, 3], 0, CBUF_BANKS)),
            PlanErrorCode::InvalidShape
        );
    }

    #[test]
    fn a_dense_row_that_needs_horizontal_tiling_is_unvalidated() {
        // One data bank holds 8192 dense pixels; a 3000-wide dense row at
        // 8 bytes a pixel does not fit, so the plan would need column tiles,
        // which only NC1HWC2 surfaces have capture backing for.
        let shape = Shape::try_with_precision(3000, 1, 1, 3, 8, Precision::Fp16)
            .unwrap()
            .try_with_padding([0, 0])
            .unwrap();
        let error = ConvPlan::try_with_cbuf_banks(shape, [1, 1], 1, CBUF_BANKS - 1)
            .expect_err("dense horizontal tiling");
        assert_eq!(error.code(), PlanErrorCode::UnvalidatedConfiguration);
        assert!(
            error
                .message()
                .contains("only capture-backed for NC1HWC2 surfaces"),
            "{error}"
        );
    }

    #[test]
    fn a_surface_no_column_split_can_hold_is_capacity_exceeded() {
        // Cin 3584 is 448 atoms, 112 CBUF entries per single-pixel row; one
        // data bank then holds four input rows, and an 11x11 kernel over an
        // 11x11 input needs all eleven at once whatever the column split.
        let shape = Shape::try_with_precision(11, 11, 1, 3584, 8, Precision::Fp16)
            .unwrap()
            .try_with_padding([0, 0])
            .unwrap();
        let error = ConvPlan::try_with_cbuf_banks(shape, [11, 11], 1, CBUF_BANKS - 1)
            .expect_err("no tile fits");
        assert_eq!(error.code(), PlanErrorCode::CapacityExceeded);
        assert!(
            error.message().contains("no standalone tile fits"),
            "{error}"
        );
    }

    // --- The fallible and panicking paths agree.

    #[test]
    fn try_new_and_new_plan_identically() {
        let cases = [
            (Shape::with_out_channels(224, 224, 2, 3, 32), [3, 3]),
            (Shape::with_out_channels(56, 56, 1, 64, 64), [3, 3]),
            (Shape::with_out_channels(14, 14, 1, 256, 256), [1, 1]),
            (
                Shape::with_out_channels(7, 1, 1, 1792, 1001).with_padding([0, 0]),
                [1, 1],
            ),
            (
                Shape::with_precision(32, 32, 1, 64, 64, Precision::Fp16).with_depthwise(),
                [3, 3],
            ),
        ];
        for (shape, kernels) in cases {
            let planned = ConvPlan::try_new(shape, kernels).unwrap();
            assert_eq!(
                planned,
                ConvPlan::new(shape, kernels),
                "{shape:?} {kernels:?}"
            );
            assert_eq!(
                planned.data_banks() + planned.weight_banks(),
                CBUF_BANKS,
                "{shape:?} {kernels:?}"
            );
            assert!(!planned.tiles().is_empty());
        }
    }

    #[test]
    fn panicking_constructors_carry_the_refusal_message() {
        let caught = std::panic::catch_unwind(|| {
            Shape::with_precision(8, 8, 1, MAX_INPUT_CHANNELS + 1, 8, Precision::Fp16)
        })
        .expect_err("should panic");
        let message = caught
            .downcast_ref::<String>()
            .cloned()
            .expect("panic payload is the formatted refusal");
        assert!(message.contains("input channels must be 1..="), "{message}");
    }

    // --- Accumulator staging layout.

    #[test]
    fn accumulator_layout_partitions_the_scratch_exactly() {
        let shape = Shape::with_precision(32, 32, 1, 64, 64, int8_accumulator(0));
        let plan = ConvPlan::try_new(shape, [3, 3]).unwrap();
        let layout = plan.accumulator_output_layout().unwrap();
        assert_eq!(layout.tiles.len(), plan.tiles().len());
        assert_eq!(layout.scratch_bytes, shape.output_scratch_bytes([3, 3]));
        let mut expected_offset = 0;
        for tile in &layout.tiles {
            assert_eq!(tile.scratch_offset, expected_offset);
            expected_offset += tile.scratch_bytes;
        }
        assert_eq!(expected_offset, layout.scratch_bytes);
    }

    #[test]
    fn accumulator_layout_refuses_non_accumulator_precisions() {
        let plan = ConvPlan::try_new(Shape::with_out_channels(32, 32, 1, 64, 64), [3, 3]).unwrap();
        assert_eq!(
            code_of(plan.accumulator_output_layout()),
            PlanErrorCode::UnsupportedSemantics
        );
    }

    // --- Activation ceilings.

    #[test]
    fn activation_ceilings_are_checked() {
        assert_eq!(
            code_of(Activation::try_clamped_fp16(-1.0)),
            PlanErrorCode::InvalidShape
        );
        assert_eq!(
            code_of(Activation::try_clamped_fp16(f32::NAN)),
            PlanErrorCode::InvalidShape
        );
        assert_eq!(
            code_of(Activation::try_clamped_int8(6.0, 0.0, 1.0)),
            PlanErrorCode::InvalidShape
        );
        assert_eq!(
            code_of(Activation::try_clamped_int8(1e30, 1e-30, 1e-30)),
            PlanErrorCode::HardwareLimit
        );
        assert_eq!(
            Activation::try_clamped_fp16(6.0).unwrap(),
            Activation::clamped_fp16(6.0)
        );
    }
}
