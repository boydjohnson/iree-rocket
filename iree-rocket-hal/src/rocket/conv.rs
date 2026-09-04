//! Vendor-derived convolution register program.
//!
//! This began as a bit-exact reproduction of group 1 (the complete
//! single-core alternative) from the vendor-compiled `32x32x3 -> 32x32x8`
//! fp16 convolution captures, which differ only in kernel geometry: 1x1 with
//! no padding versus 3x3 with SAME padding. [`conv_2d`] still reproduces that
//! program byte for byte, and the hash tests below pin it.
//!
//! It now generalises to arbitrary [`Shape`] and to output-row [`Tile`]s.
//! Every shape- and tile-dependent register formula is derived from vendor
//! captures rather than assumed: the tile formulas from a cross-group diff of
//! the six captured plans, and the width and height formulas from a sweep of
//! 35 captures (212 convolution programs) spanning widths 32..256 and heights
//! 32..256. Registers that vary in no capture stay literal constants.
//!
//! The ordinary tile builder supports fp16 `Cin` 1..=80, int8 `Cin`
//! 1..=128, `Cout` 1..=512, strides 1..4, and 1x1 or 3x3 kernels.
//! [`ConvPlan`] additionally plans kernel extents from 1 through 11,
//! including even and non-square kernels and horizontal tiling where a
//! full-width row cannot fit. [`Shape::with_padding`] makes the leading
//! padding independent of the kernel extent; without it each axis defaults
//! to `extent / 2`, preserving the historical odd-kernel API.
//!
//! Kernels need not be square. A sweep of 53 non-square captures shows the
//! two extents govern their own axes throughout -- `weight_width` and
//! `pad_left` follow the kernel's width, `weight_height`, `pad_top` and
//! `feature_grains` follow its height, and the coefficient footprint is
//! `kh * kw * pad(Cin) * element_bytes` -- so the direct geometry needed no
//! new rule, only the removal of the assumption that one `k` served both.
//! What does *not* carry over is the CBUF split: at equal coefficient
//! demand, mirrored shapes split differently, so [`ConvPlan`] plans
//! non-square kernels only up to the demand where the captures still agree
//! and otherwise requires an explicit split.
//!
//! All of that holds in both precisions. A matching int8 sweep of 60
//! captures reproduces every kernel-geometry formula unchanged, and the
//! paired fp16/int8 diff moves only the fields precision already moved --
//! the precision selectors, the doubled channel counts, and the
//! requantization path. `weight_width`, `weight_height`, `pad_left` and
//! `pad_top` are byte-identical across precision at every rectangular
//! geometry.
//!
//! Input channel count picks the memory layout. While a pixel fits in half a
//! feature atom -- `Cin` up to 4 -- the vendor keeps it dense NHWC and the
//! CNA pads internally. From `Cin` 5 the map becomes NC1HWC2 surfaces, and
//! the row strides, the CBUF bank split and `data_entries` all change with
//! it; `data_entries` in particular stops depending on the tile height
//! entirely. `FeatureLayout` names the two regimes.
//!
//! Channel padding is a table rather than arithmetic. It is `atoms * 8`
//! except at three atoms, where `datain_channel` stays 24 but coefficients
//! use 32, and at seven, where `datain_channel` stays 56 but coefficients
//! use 64. Atom counts 5, 6, 9 and 10 are unpadded, so no arithmetic rule
//! fits and none is invented.
//!
//! Output channels remain one streamed kernel set rather than splitting into
//! fp16 kernel groups. The capture corpus covers this through `Cout = 512`,
//! with hardware validation through `Cout = 128`.
//!
//! Keeping this separate from `rocket::regcmd` keeps a capture-derived path
//! distinct from that module's Mesa-derived one.
//!
//! # Output-row tiles
//!
//! [`conv_2d_tile`] emits the same single-core program over an arbitrary
//! range of output rows. Every tile-dependent register value is derived
//! from a cross-group diff of the six captured programs, which cover three
//! alternative height splits (32 / 16+16 / 11+11+10) at two kernel sizes --
//! twelve independent observations per register. Sixteen registers vary
//! across those captures and take derived values here; the other 109 are
//! literal constants reproduced from the capture. The derived values are
//! checked against every observation by the tile register test below.
//!
//! A tile program is **not** byte-identical to captured groups 2-6. Those
//! are the vendor's own multi-core plans and carry PPU/PPU_RDMA blocks whose
//! role has not been derived, plus a plan index in two documented-reserved
//! fields. A tile emitted here is a standalone single-core program covering
//! a row range, intended for submission as one of several independent jobs.
//! Its geometry registers match the captures exactly; its command sequence
//! is group 1's.

use crate::rocket::builders::{
    Bits, RegCmd, Register, RegisterMeta,
    cna::{
        CnaCbufCon0, CnaCbufCon1, CnaConvCon1, CnaConvCon2, CnaConvCon3, CnaCvtCon0, CnaCvtCon1,
        CnaCvtCon2, CnaCvtCon3, CnaCvtCon4, CnaCvtCon5, CnaDataSize0, CnaDataSize1, CnaDataSize2,
        CnaDataSize3, CnaDcompAddr0, CnaDcompAmount0, CnaDcompAmount1, CnaDcompAmount2,
        CnaDcompAmount3, CnaDcompAmount4, CnaDcompAmount5, CnaDcompAmount6, CnaDcompAmount7,
        CnaDcompAmount8, CnaDcompAmount9, CnaDcompAmount10, CnaDcompAmount11, CnaDcompAmount12,
        CnaDcompAmount13, CnaDcompAmount14, CnaDcompAmount15, CnaDcompCtrl, CnaDcompRegnum,
        CnaDmaCon0, CnaDmaCon1, CnaDmaCon2, CnaFcCon0, CnaFcCon1, CnaFcCon2, CnaFcDataSize0,
        CnaFcDataSize1, CnaFeatureDataAddr, CnaPadCon0, CnaPadCon1, CnaWeightSize0, CnaWeightSize1,
        CnaWeightSize2,
    },
    core::{CoreClipTruncate, CoreDataoutSize0, CoreDataoutSize1, CoreMiscCfg, CoreReserved3030},
    dpu::{
        DpuBnAluCfg, DpuBnCfg, DpuBnMulCfg, DpuBnReluxCmpValue, DpuBsAluCfg, DpuBsCfg, DpuBsMulCfg,
        DpuBsOwCfg, DpuBsOwOp, DpuBsReluxCmpValue, DpuDataCubeChannel, DpuDataCubeHeight,
        DpuDataCubeNotchAddr, DpuDataCubeWidth, DpuDataFormat, DpuDstBaseAddr, DpuDstSurfStride,
        DpuEwCfg, DpuEwCvtOffsetValue, DpuEwCvtScaleValue, DpuEwOpValue0, DpuEwOpValue1,
        DpuEwOpValue2, DpuEwOpValue3, DpuEwOpValue4, DpuEwOpValue5, DpuEwOpValue6, DpuEwOpValue7,
        DpuEwReluxCmpValue, DpuFeatureModeCfg, DpuLutAccessCfg, DpuLutAccessData, DpuLutCfg,
        DpuLutInfo, DpuLutLeEnd, DpuLutLeSlopeScale, DpuLutLeSlopeShift, DpuLutLeStart,
        DpuLutLoEnd, DpuLutLoSlopeScale, DpuLutLoSlopeShift, DpuLutLoStart, DpuOffsetPend,
        DpuOutCvtOffset, DpuOutCvtScale, DpuOutCvtShift, DpuReserved40c4, DpuSPointer,
        DpuSurfaceAdd, DpuWdmaSize0, DpuWdmaSize1,
    },
    dpu_rdma::{
        DpuRdmaBnBaseAddr, DpuRdmaBrdmaCfg, DpuRdmaBsBaseAddr, DpuRdmaDataCubeChannel,
        DpuRdmaDataCubeHeight, DpuRdmaDataCubeWidth, DpuRdmaErdmaCfg, DpuRdmaEwBaseAddr,
        DpuRdmaEwSurfNotch, DpuRdmaEwSurfStride, DpuRdmaFeatureModeCfg, DpuRdmaNrdmaCfg,
        DpuRdmaPadCfg, DpuRdmaSPointer, DpuRdmaSrcBaseAddr, DpuRdmaSrcDmaCfg, DpuRdmaSurfNotch,
        DpuRdmaWeight,
    },
    pc::{PCOperationMask, PCRegisterAmounts, PCTrailer},
    values::{ArgbInputMode, BurstLength, DataPrecision, DpuOutputMode, OutputPrecision},
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
/// Shared between dense and depthwise `Shape` construction. That sharing is
/// what made the 960 attempt unsafe; it is not a problem here, because the
/// depthwise half now has its own corpus
/// (`conv_vendor_fixtures_depthwise.json`, 63/63 agreement to C=1344) and
/// because fp16 depthwise cannot reach a compiled dispatch at all -- the
/// demote pass deliberately excludes it (see
/// `RocketDemoteConvInputsPass.cpp`, reverted 2026-09-01).
pub const MAX_INPUT_CHANNELS: u32 = 1344;

/// `CNA_DATA_SIZE1.datain_channel_real` counts `Cin - 1` modulo this, even
/// though the field is 14 bits wide and could hold far more.
///
/// Confirmed in both precisions and only visible above 64 channels, which no
/// hardware test had reached: fp16 `Cin` 72 programs 7 and 80 programs 15;
/// int8 `Cin` 112 programs 47 and 128 programs 63.
const CHANNEL_REAL_MODULUS: u32 = 64;

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
pub const MAX_INT8_INPUT_CHANNELS: u32 = 1344;

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
pub const MAX_INT8_OUTPUT_CHANNELS: u32 = 1792;

/// Channel ceilings for int4, set to what the hardware ladder measures
/// rather than to what the arithmetic would allow.
///
/// int4 packs four times as densely as fp16, so nothing in the CBUF model
/// stops these from being much higher; they are low because the evidence
/// stops here. Raise them with the measurement, not ahead of it.
pub const MAX_INT4_INPUT_CHANNELS: u32 = 512;
pub const MAX_INT4_OUTPUT_CHANNELS: u32 = 512;

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
const MAX_DENSE_CHANNELS: u32 = 4;

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
pub const MAX_OUTPUT_CHANNELS: u32 = 1792;

/// `DPU_BS_MUL_CFG.bs_mul_shift_value`, and its negated twin
/// `DPU_DATA_FORMAT.bs_mul_shift_value_neg`, in every quantized capture.
const BS_MUL_SHIFT_VALUE: u32 = 14;

/// `DPU_RDMA_RDMA_BRDMA_CFG.brdma_data_use` when BRDMA supplies bias only.
const BRDMA_DATA_USE_BIAS: u32 = 1;

/// The same field once requantization is active and BRDMA also supplies the
/// scale and shift operands.
const BRDMA_DATA_USE_QUANTIZED: u32 = 7;

/// Physical width of one feature atom.
const FEATURE_ATOM_BYTES: u32 = 16;

/// Total CBUF banks the CNA partitions between feature data and weights.
const CBUF_BANKS: u32 = 12;

/// Bytes one CBUF bank holds: 256 entries of 128 bytes.
const CBUF_BANK_BYTES: u32 = 256 * 128;

/// Feature atoms the CBUF charges per `data_entries` entry.
///
/// The surface feature charge is counted in whole entries of four atoms, and
/// it rounds *up*: a row whose atom count is not a multiple of four still
/// occupies the whole final entry. `CNA_CBUF_CON1.data_entries` has always
/// been programmed that way (see its `div_ceil` below); the residency bound
/// in [`Shape::max_tile_input_rows_for_width_and_data_banks`] has to charge
/// the same way or it over-commits the CBUF.
/// Whether `Shape` may be built with more input channels than the capture
/// corpus backs.
///
/// This exists so the CBUF-split scoring harness
/// (`tests/cbuf_split_score.rs`) and the high-channel hardware probes can
/// reach past the cap; without it `Shape` refuses to build and the most
/// interesting part of the vendor corpus is invisible. Nothing on the compiled
/// path sets this.
///
/// It lifts **both** channel ceilings. It used to lift only the input one,
/// which made a whole class of shape unreachable for characterization:
/// MobileNetV2's widest dense 1x1 is `Cin` 448 -> `Cout` 1792, and no probe
/// could construct it to find out whether the `Cout` ceiling was a real limit
/// or just the extent of the measurement. It was the latter.
fn unbacked_channels_allowed() -> bool {
    std::env::var_os("ROCKET_ALLOW_UNBACKED_CHANNELS").is_some()
}

const CBUF_ATOMS_PER_ENTRY: u32 = 4;

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
fn weight_banks_floor(weight_channels: u32) -> u32 {
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
fn streamed_weight_bank_preference(weight_channels: u32, kernels: Kernels) -> u32 {
    let undivided = streamed_weight_bank_preference_for_group(weight_channels, kernels, 1);
    if undivided <= MAX_UNDIVIDED_WEIGHT_BANKS {
        return undivided;
    }
    // Deliberately unclamped: a working set that still saturates after the
    // division is refused by `demand_based_cbuf_partition`, not quietly turned
    // into a one-data-bank split. 5x5 `Cin` 512 wants 13 banks even divided,
    // and no capture covers it.
    streamed_weight_bank_preference_for_group(weight_channels, kernels, 2)
}

/// Largest coefficient grant the vendor will take without dividing the
/// streamed output-channel group -- i.e. it always leaves at least two banks
/// for feature data.
///
/// Measured, not chosen: the spatial corpus shows the group divided at every
/// `Cin` whose undivided grant reaches eleven (k=3 `Cin` >= 576) and undivided
/// at ten (5x5 `Cin` 192 keeps the vendor's 2/10).
const MAX_UNDIVIDED_WEIGHT_BANKS: u32 = CBUF_BANKS - 2;

/// Bytes one CBUF entry holds, and entries one bank holds. The multi-pass
/// correction below depends on the remainder within a bank, so the two have to
/// be visible separately even though their product is [`CBUF_BANK_BYTES`].
const CBUF_ENTRY_BYTES: u32 = 128;
const CBUF_ENTRIES_PER_BANK: u32 = CBUF_BANK_BYTES / CBUF_ENTRY_BYTES;

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
fn streamed_weight_bank_preference_for_group(
    weight_channels: u32,
    kernels: Kernels,
    group_divisor: u32,
) -> u32 {
    const STREAMED_BYTES_PER_INPUT_TAP: u32 = 64;
    let working_set =
        kernels[0] as u32 * kernels[1] as u32 * weight_channels * STREAMED_BYTES_PER_INPUT_TAP
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
const MAX_DATA_ENTRIES: u32 = 0x7fff;

/// Largest logical value encodable by the 10-bit
/// `CNA_CONV_CON2.feature_grains` field.
const MAX_FEATURE_GRAINS: u32 = 0x03ff;

/// Feature pixels one CBUF bank holds: 256 entries of 128 bytes, at one
/// 16-byte feature atom per pixel, is 2048 -- but the vendor allocates in
/// half-bank steps, which is the 1024 below.
const PIXELS_PER_BANK_STEP: u32 = 1024;

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
    pub fn element_bits(&self) -> u32 {
        match self {
            Precision::Fp16 | Precision::Bf16 | Precision::Int16 => 16,
            Precision::Int8(_) | Precision::Int8Accumulator(_) => 8,
            Precision::Int4 => 4,
        }
    }

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
    pub fn shares_fp16_layout(&self) -> bool {
        self.element_bits() == 16
    }

    /// Bytes one logical output element occupies.
    pub fn output_element_bytes(&self) -> u32 {
        match self {
            // int4 accumulates into int16, so its result is wider than its
            // operands -- the one rung where the two differ by more than a
            // requantization.
            Precision::Fp16 | Precision::Bf16 | Precision::Int16 | Precision::Int4 => 2,
            Precision::Int8(_) => 1,
            Precision::Int8Accumulator(_) => 4,
        }
    }

    /// Whether this mode writes the exact int32 convolution accumulator.
    pub fn writes_accumulators(&self) -> bool {
        matches!(self, Precision::Int8Accumulator(_))
    }

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
    /// Twice the atom width in both precisions: 16 for fp16 and 32 for int8.
    /// A clean rule with no table and no exceptions in either -- verified at
    /// every fp16 Cout in the corpus, including the awkward 20, 24, 28, 40,
    /// 56 and 72 where the *input* padding needed special cases, and at 10
    /// int8 values from 8 to 112.
    pub fn out_channel_granule(&self) -> u32 {
        2 * self.channels_per_atom()
    }

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
    pub fn max_in_channels(&self) -> u32 {
        match self {
            Precision::Fp16 | Precision::Bf16 | Precision::Int16 => MAX_INPUT_CHANNELS,
            Precision::Int8(_) | Precision::Int8Accumulator(_) => MAX_INT8_INPUT_CHANNELS,
            Precision::Int4 => MAX_INT4_INPUT_CHANNELS,
        }
    }

    /// Largest output-channel count this precision has capture or hardware
    /// backing for. The int8 side reaches further; see
    /// [`MAX_INT8_OUTPUT_CHANNELS`].
    pub fn max_out_channels(&self) -> u32 {
        match self {
            Precision::Fp16 | Precision::Bf16 | Precision::Int16 => MAX_OUTPUT_CHANNELS,
            Precision::Int8(_) | Precision::Int8Accumulator(_) => MAX_INT8_OUTPUT_CHANNELS,
            Precision::Int4 => MAX_INT4_OUTPUT_CHANNELS,
        }
    }

    /// The 3-bit precision field this datatype programs into the CNA input,
    /// CORE processing and DPU stages.
    fn data_precision(&self) -> DataPrecision {
        match self {
            Precision::Fp16 => DataPrecision::Fp16,
            Precision::Bf16 => DataPrecision::Bf16,
            Precision::Int16 => DataPrecision::Int16,
            Precision::Int4 => DataPrecision::Int4,
            Precision::Int8(_) | Precision::Int8Accumulator(_) => DataPrecision::Int8,
        }
    }

    /// The precision field the DPU output stage programs.
    ///
    /// Not simply [`Precision::data_precision`]: two rungs write a result
    /// wider than their operands. int8 accumulator output writes int32, and
    /// int4 writes the int16 its MACs accumulate into -- the
    /// `int4 -> int16` / `int8 -> int32` / `fp16 -> fp32` pairing from
    /// `../rockchip-npu-notes/datatypes.md`, with fp16's fp32 accumulator
    /// converted down on the way out where int4's int16 one is not.
    ///
    /// The DPU's enum is also not the CNA/CORE one; see [`OutputPrecision`].
    fn output_data_precision(&self) -> OutputPrecision {
        match self {
            Precision::Fp16 => OutputPrecision::Fp16,
            Precision::Bf16 => OutputPrecision::Bf16,
            Precision::Int16 => OutputPrecision::Int16,
            Precision::Int4 => OutputPrecision::Int16,
            Precision::Int8(_) => OutputPrecision::Int8,
            Precision::Int8Accumulator(_) => OutputPrecision::Int32,
        }
    }

    /// Calibration parameters, for the precisions that requantize. `None`
    /// is also the "unquantized rung" predicate the register program keys
    /// its BS/CVT bypasses off.
    pub fn quantization(&self) -> Option<Quantization> {
        match self {
            Precision::Fp16 | Precision::Bf16 | Precision::Int16 | Precision::Int4 => None,
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
const MANTISSA_FLOOR: u32 = 1 << 14;

/// Largest shift `DPU_OUT_CVT_SHIFT` encodes. The field is 6 bits; the
/// corpus only reaches 26.
const MAX_CVT_SHIFT: u32 = 63;

impl Multiplier {
    /// Encodes a real multiplier, normalizing the mantissa into
    /// `[2^14, 2^15)`.
    ///
    /// Panics rather than saturating on a multiplier the form cannot carry:
    /// a silently clamped requantization scale is a whole-tensor error that
    /// would be very hard to attribute later.
    pub fn from_ratio(ratio: f64) -> Multiplier {
        assert!(
            ratio.is_finite() && ratio > 0.0,
            "requantization multiplier must be finite and positive, got {ratio}"
        );
        // `scaled` is `ratio * 2^shift` throughout, driven into the mantissa
        // range. The exponent is signed while it is being searched for: a
        // multiplier above 1 normalizes to a shift below 14, and only the
        // final value has to be a nonnegative field.
        let mut shift: i32 = 14;
        let mut scaled = ratio * f64::from(MANTISSA_FLOOR);
        while scaled < f64::from(MANTISSA_FLOOR) {
            scaled *= 2.0;
            shift += 1;
            assert!(
                shift <= MAX_CVT_SHIFT as i32,
                "requantization multiplier {ratio} is too small to encode; \
                 DPU_OUT_CVT_SHIFT tops out at {MAX_CVT_SHIFT}"
            );
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
        assert!(
            shift >= 0,
            "requantization multiplier {ratio} is too large to encode; \
             DPU_OUT_CVT_SHIFT cannot be negative"
        );
        Multiplier {
            scale,
            shift: shift as u32,
        }
    }

    /// Encodes the per-tensor half of a requantisation whose per-channel BS
    /// multipliers are normalised to [`BS_UNIT_MULTIPLIER`].
    ///
    /// The hardware applies `(accumulator * bs_multiplier) >>
    /// BS_MULTIPLIER_SHIFT` before this stage sees it, so a plane at unit
    /// contributes a gain of `2^(14 - 7)` that has to come back out here.
    /// Measured on hardware; see [`BS_MULTIPLIER_SHIFT`].
    pub fn for_unit_bs(total_ratio: f64) -> Multiplier {
        let bs_gain = f64::from(BS_UNIT_MULTIPLIER >> BS_MULTIPLIER_SHIFT);
        Multiplier::from_ratio(total_ratio / bs_gain)
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
    pub fn clamped_fp16(ceiling: f32) -> Activation {
        assert!(
            ceiling.is_finite() && ceiling > 0.0,
            "activation ceiling must be finite and positive, got {ceiling}"
        );
        Activation::Clamped {
            cmp: ceiling.to_bits(),
        }
    }

    /// A clamped ReLU for an int8 convolution.
    ///
    /// The int8 accumulator is a scaled integer, so the ceiling is divided
    /// by the accumulator's unit -- `input_scale * weights_scale` -- then
    /// expressed in the BN stage's post-BS domain. [`BsEntry::default`]
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
    /// This constructor is paired with [`BsEntry::default`] and
    /// [`Multiplier::for_unit_bs`]. Callers deliberately using a different
    /// BS multiplier must construct [`Activation::Clamped`] in that custom
    /// post-BS domain.
    pub fn clamped_int8(ceiling: f32, input_scale: f32, weights_scale: f32) -> Activation {
        assert!(
            ceiling.is_finite() && ceiling > 0.0,
            "activation ceiling must be finite and positive, got {ceiling}"
        );
        let unit = f64::from(input_scale) * f64::from(weights_scale);
        assert!(
            unit.is_finite() && unit > 0.0,
            "input_scale * weights_scale must be finite and positive, got {unit}"
        );
        let bs_gain = f64::from(BS_UNIT_MULTIPLIER >> BS_MULTIPLIER_SHIFT);
        let cmp = (f64::from(ceiling) / unit * bs_gain).round();
        assert!(
            (0.0..=f64::from(u32::MAX)).contains(&cmp),
            "activation ceiling {ceiling} is {cmp} post-BS units, outside \
             the 32-bit BN_RELUX_CMP_VALUE field"
        );
        Activation::Clamped { cmp: cmp as u32 }
    }

    /// `(bn_bypass, bn_relu_bypass, bn_relux_en, cmp)`, the four fields the
    /// activation sweep found moving.
    fn bn_programming(self) -> (u32, u32, u32, u32) {
        match self {
            Activation::None => (1, 1, 0, 0),
            Activation::Relu => (0, 0, 0, 0),
            Activation::Clamped { cmp } => (0, 0, 1, cmp),
        }
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
        assert!(
            width > 0 && height > 0,
            "convolution extents must be nonzero"
        );
        assert!(stride > 0, "convolution stride must be nonzero");
        assert!(
            (1..=precision.max_in_channels()).contains(&in_channels) || unbacked_channels_allowed(),
            "input channels must be 1..={}; beyond that the channel padding \
             has no capture backing at this precision",
            precision.max_in_channels()
        );
        assert!(
            (1..=precision.max_out_channels()).contains(&out_channels)
                || unbacked_channels_allowed(),
            "output channels must be 1..={}; beyond that the capture corpus \
             does not reach and the measurement has not been made",
            precision.max_out_channels()
        );
        if precision == Precision::Int4 {
            // A partial int4 feature atom has no measurement behind it, and
            // the ARGB dense path below `MAX_DENSE_CHANNELS` cannot address
            // a nibble at all. Whole atoms are also what every useful int4
            // shape has, so this refuses rather than guesses.
            assert!(
                in_channels.is_multiple_of(FEATURE_ATOM_BYTES * 2),
                "int4 input channels must be a whole {}-channel feature atom; \
                 partial int4 atoms are unmeasured",
                FEATURE_ATOM_BYTES * 2,
            );
        }
        if let Precision::Int8Accumulator(quantization) = precision {
            assert!(
                quantization.input_zero_point == 0
                    && quantization.output_zero_point == 0
                    && quantization.weight_zero_point == 0,
                "int32 accumulator output currently requires zero input, weight, and output zero-points"
            );
        }
        Shape {
            width,
            height,
            stride,
            in_channels,
            out_channels,
            precision,
            padding: None,
            activation: Activation::None,
            depthwise: false,
        }
    }

    /// Sets the model's leading padding independently on each axis.
    ///
    /// The capture corpus covers padding no larger than its kernel extent;
    /// the exact per-kernel bound is checked when the kernel is supplied.
    pub fn with_padding(mut self, padding: Padding) -> Shape {
        assert!(
            padding.into_iter().all(|pad| pad <= 15),
            "convolution padding must fit the CNA's 4-bit pad fields"
        );
        self.padding = Some(padding);
        self
    }

    /// Fuses `activation` into this convolution's own DPU pass.
    pub fn with_activation(mut self, activation: Activation) -> Shape {
        assert!(
            !self.precision.writes_accumulators() || activation == Activation::None,
            "int32 accumulator output must not fuse an activation"
        );
        self.activation = activation;
        self
    }

    /// Makes this a depthwise convolution, one filter per input channel.
    ///
    /// Only a channel multiplier of one is captured, so the output channel
    /// count must already equal the input one.
    ///
    /// # Packing the weight buffer
    ///
    /// Use [`crate::rocket::tensor_layout::pack_depthwise_to_rocket_weights`],
    /// not `pack_hwcf_to_rocket_weights`. A depthwise filter is
    /// `[Cin][kh][kw]` and the hardware wants it tap-major,
    /// `(ky * kw + kx) * padded_channels + channel`, which is the transpose
    /// of how torch and ONNX store it.
    ///
    /// The capture sweep could not have shown this -- a capture carries the
    /// register program, never the buffer it points at. It came from one-hot
    /// probing every slot of a real weight buffer on hardware
    /// (`tests/conv_depthwise_probe_hw.rs`).
    pub fn with_depthwise(mut self) -> Shape {
        assert_eq!(
            self.in_channels, self.out_channels,
            "depthwise capture backing covers a channel multiplier of one only"
        );
        self.depthwise = true;
        self
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
    fn out_channel_granule(&self) -> u32 {
        let dense = self.precision.out_channel_granule();
        if self.depthwise { 2 * dense } else { dense }
    }

    /// `CNA_CONV_CON1`, `DPU_FEATURE_MODE_CFG` and
    /// `DPU_RDMA_RDMA_FEATURE_MODE_CFG` all carry the same mode, 3 for
    /// depthwise against 0 for a dense convolution.
    fn conv_mode(&self) -> u32 {
        if self.depthwise { 3 } else { 0 }
    }

    /// `DPU_BS_OW_CFG.SIZE_E_0/1/2`: 3 for depthwise, **7 for dense
    /// accumulator output**, 1 for everything else.
    ///
    /// The 7 is the integer-output stride quirk
    /// (`rockchip-npu-notes/encodings/size-e-quirk.md`,
    /// `rocket-userspace`'s `gen_matmul_int8`): an integer conv output strides
    /// as if each element were 8 bytes, regardless of its actual width. It
    /// looks wrong against the float rule (`size_e = bytes - 1`, so 3 for a
    /// 4-byte int32) and it is not; do not "fix" it to 3.
    ///
    /// It is only meaningful together with `mc_surf_out = 0` and
    /// `surf_add = dataout_w * dataout_h * 8` -- the three are one geometry,
    /// and moving any one alone reads as inert. See
    /// [`DENSE_ACCUMULATOR_SURF_MULT`] and [`bs_ow_size_e_override`].
    ///
    /// **On the requantized path this value is load-bearing in a sharper
    /// way**: it is one of the few registers where a wrong value *hangs the
    /// NPU* rather than returning wrong data. Measured at 32x32 Cin=384
    /// Cout=64 k1 [HW sweep, planck 2026-09-03, `accumulator_size_e_probe`]:
    /// requantized int8 (which leaves `OD_BYPASS` clear) is bit-exact at
    /// `size_e = 1` in ~30 ms, and at 3 or 7 writes 1024 of 65536 bytes and
    /// takes ~525 ms per tile -- the watchdog killing the job, with `PREP_BO`
    /// still returning success.
    fn bs_ow_size_e(&self) -> u32 {
        if let Some(size_e) = bs_ow_size_e_override(self.in_channels) {
            return size_e;
        }
        if let Some(size_e) = int4_override(self.precision, "SIZE_E") {
            return size_e;
        }
        if self.depthwise {
            3
        } else if self.precision.writes_accumulators() || self.precision == Precision::Int4 {
            // int4 is the integer-output quirk
            // `../rockchip-npu-notes/encodings/size-e-quirk.md` describes:
            // its result is a 2-byte int16, whose natural `size_e` would be
            // 1, but the integer write path strides as if each element were
            // eight bytes. Measured directly here [HW sweep, planck
            // 2026-09-03, `int4_output_write_map_probe`]: at 8x8 Cin 64
            // Cout 64, `size_e` 0/1/2/3 write 0/2/3/4 of the eight output
            // surfaces and stop, and only 7 writes all 8192 bytes.
            7
        } else {
            1
        }
    }

    fn kernel_programming(&self, kernels: Kernels) -> KernelProgramming {
        kernel_programming(kernels, self.padding)
    }

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
    pub fn programmed_kernels(&self) -> u32 {
        if self.depthwise {
            return 1;
        }
        if self.precision.shares_fp16_layout() {
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
    pub fn bs_buffer_bytes(&self) -> usize {
        bs_buffer_bytes(self.padded_out_channels())
    }

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
    /// [`ConvPlan::programs_with_staged_accumulator_output`] partitions the
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

    /// Padded channel count [`crate::rocket::tensor_layout::pack_depthwise_to_rocket_weights`]'s
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
    fn cbuf_atoms(&self) -> u32 {
        quad_atoms(self.feature_atoms())
    }

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
    pub fn input_row_stride(&self) -> u32 {
        match self.layout() {
            FeatureLayout::Dense => self.width * self.in_channels * self.precision.element_bytes(),
            FeatureLayout::Surfaces => self.width * FEATURE_ATOM_BYTES,
        }
    }

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
    fn cbuf_input_width(&self, input_width: u32) -> u32 {
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
    fn dense_cbuf_pixel_bytes(&self) -> u32 {
        self.in_channels.next_power_of_two().min(4) * self.precision.element_bytes()
    }

    fn max_data_entries(&self) -> u32 {
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
    fn tile_in_first(&self, kernels: Kernels, out_first: u32) -> u32 {
        let padding = self.kernel_programming(kernels).pad_top;
        (out_first * self.stride).saturating_sub(padding)
    }

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
    fn streamed_contraction_channels(&self) -> u32 {
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
    fn data_bank_demand(&self) -> u32 {
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
    fn weight_bank_demand(&self, kernels: Kernels) -> u32 {
        self.weight_bytes(kernels).div_ceil(CBUF_BANK_BYTES).max(1)
    }

    fn demand_based_cbuf_partition(&self, kernels: Kernels) -> (u32, u32) {
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
        let streamed_preference =
            streamed_weight_bank_preference(self.streamed_contraction_channels(), kernels);
        assert!(
            streamed_preference <= CBUF_BANKS - 1,
            "coefficient working set wants {streamed_preference} CBUF banks for \
             {}x{} kernel and {} weight channels, more than the {} grantable; the \
             CBUF partition above that point is not capture-backed (see \
             MAX_INPUT_CHANNELS' doc comment)",
            kernels[0],
            kernels[1],
            self.weight_channels(),
            CBUF_BANKS - 1,
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
            return (CBUF_BANKS - weight_banks, weight_banks);
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
            return (CBUF_BANKS - weight_banks, weight_banks);
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
            (whole_data_banks, whole_weight_banks)
        } else {
            (data_banks, weight_banks)
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
    pub fn data_banks(&self, kernels: Kernels) -> u32 {
        assert_default_cbuf_kernel(kernels);
        self.demand_based_cbuf_partition(kernels).0
    }

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
    pub fn max_tile_input_rows_for_data_banks(&self, data_banks: u32) -> u32 {
        self.max_tile_input_rows_for_width_and_data_banks(self.width, data_banks)
    }

    /// Most input rows that fit when a task reads only `input_width`.
    ///
    /// Horizontal tiling changes the resident CBUF footprint without
    /// changing the tensor's memory strides. The three large-kernel captures
    /// that require it sit exactly on this product bound:
    /// `input_width * input_rows * atoms * 16 <= data_banks * 32768`.
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
        capacity.min(self.max_data_entries() / charged_width).max(1)
    }

    /// Conservative input-row limit imposed by `feature_grains`.
    ///
    /// The first tile carries the full top padding and therefore has the
    /// largest value: `in_rows + kernel_height + pad_top`. Continuation
    /// tiles can fit at least as many rows, so using the first-tile bound for
    /// every tile keeps the planner simple and guarantees encodability.
    fn max_feature_grain_input_rows(&self, kernels: Kernels) -> u32 {
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
    pub fn min_tiles(&self, kernels: Kernels) -> u32 {
        self.min_tiles_for_data_banks(kernels, self.data_banks(kernels))
    }

    /// Fewest output-row tiles needed with an explicit feature-data split.
    pub fn min_tiles_for_data_banks(&self, kernels: Kernels, data_banks: u32) -> u32 {
        self.min_tiles_for_width_and_data_banks(kernels, self.width, data_banks)
    }

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
#[derive(Clone, Copy)]
struct KernelProgramming {
    height: u32,
    width: u32,
    /// Zero-padded rows above the first output row, `kh / 2`.
    pad_top: u32,
    /// Zero-padded columns left of the first output column, `kw / 2`.
    pad_left: u32,
}

fn kernel_programming(kernels: Kernels, padding: Option<Padding>) -> KernelProgramming {
    let [height, width] = kernels;
    let backed = |extent: usize| (1..=11).contains(&extent);
    assert!(
        backed(height) && backed(width),
        "conv_2d only has vendor reference data for kernel extents from 1 through 11, \
         got {height}x{width}"
    );
    let [pad_top, pad_left] = padding.unwrap_or([height / 2, width / 2]);
    assert!(
        pad_top < height && pad_left < width,
        "padding must be smaller than its kernel extent; got padding \
         {pad_top}x{pad_left} for kernel {height}x{width}"
    );
    KernelProgramming {
        height: height as u32,
        width: width as u32,
        pad_top: pad_top as u32,
        pad_left: pad_left as u32,
    }
}

fn assert_default_cbuf_kernel(kernels: Kernels) {
    assert!(
        matches!(kernels, [1, 1] | [3, 3]),
        "automatic CBUF allocation only has runtime backing for 1x1 and 3x3; \
         use ConvPlan, or conv_2d_tile_with_cbuf_banks for an explicit override"
    );
}

/// ARGB input mode for a dense feature map.
///
/// Only reachable for `Cin` 1..=4, since wider pixels use surfaces. The
/// captures confirm 3 and 4 directly; 1 and 2 follow the enum's own
/// definition, which came from the vendor register description.
fn argb_input_mode(in_channels: u32) -> ArgbInputMode {
    match in_channels {
        1 => ArgbInputMode::OneChannel,
        2 => ArgbInputMode::TwoChannels,
        3 => ArgbInputMode::ThreeChannels,
        4 => ArgbInputMode::FourChannels,
        _ => unreachable!("dense layout is only used up to four input channels"),
    }
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
    fn split_greedy_to_capacity(
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
    /// Broken out of `split` so [`realign_dense_row_tiles`] can rebuild an
    /// individual tile after moving its boundary, without duplicating this
    /// arithmetic -- the two must stay in exact agreement, since a
    /// realigned tile has to be bit-identical to what `split` itself would
    /// have produced had it picked that boundary in the first place.
    fn from_bounds(shape: Shape, kernels: Kernels, out_first: u32, out_rows: u32) -> Tile {
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

    fn input_offset(&self, shape: Shape) -> u32 {
        match shape.layout() {
            FeatureLayout::Dense => {
                self.in_first * shape.in_channels * shape.precision.element_bytes()
            }
            FeatureLayout::Surfaces => self.in_first * FEATURE_ATOM_BYTES,
        }
    }

    fn output_offset(&self, shape: Shape) -> u32 {
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

    fn input_offset(&self, shape: Shape) -> u32 {
        self.rows.input_offset(shape) + self.columns.input_offset(shape)
    }

    fn output_offset(&self, shape: Shape, kernels: Kernels) -> u32 {
        self.rows.output_offset(shape, kernels) + self.columns.output_offset(shape)
    }
}

/// A complete standalone-job plan for one convolution.
///
/// The plan owns the policy that the low-level tile builders intentionally
/// leave with their caller: the CBUF split, horizontal partition (if any),
/// and the row split for each column. Programs returned by [`ConvPlan::programs`]
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

/// Submission-ready accumulator programs and the exact scratch layout they
/// write. `buffers.output` passed to
/// [`ConvPlan::programs_with_staged_accumulator_output`] must address at
/// least `scratch_bytes` bytes.
pub struct StagedAccumulatorOutput {
    pub programs: Vec<Vec<RegCmd>>,
    pub tiles: Vec<AccumulatorOutputTile>,
    pub scratch_bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OutputPlacement {
    SharedImage,
    ContiguousTile,
}

impl ConvPlan {
    /// Plans all standalone jobs needed to cover `shape` exactly once.
    ///
    /// Panics when the requested operation lies outside the supported policy.
    /// In particular, the capture-specific odd-square policies above 3x3
    /// require fp16 and stride 1, every even kernel requires stride 1, and
    /// NC1HWC2 input is required if horizontal tiling is necessary.
    pub fn new(shape: Shape, kernels: Kernels) -> ConvPlan {
        let kernel = shape.kernel_programming(kernels);
        if kernel.height != kernel.width {
            return ConvPlan::new_with_cbuf_partition(
                shape,
                kernels,
                non_square_cbuf_partition(shape, kernels),
            );
        }
        let (data_banks, weight_banks) = match kernel.height {
            1 | 3 => shape.demand_based_cbuf_partition(kernels),
            2 | 4 | 6 | 8 | 10 => even_square_cbuf_partition(shape, kernels),
            5 => {
                assert_large_kernel_plan_case(shape);
                shape.demand_based_cbuf_partition(kernels)
            }
            7 => {
                assert_large_kernel_plan_case(shape);
                // The focused sweep follows coefficient demand through seven
                // banks (1/11, 2/10, 8/4, 7/5 and 5/7 are all observed), then
                // switches to the streamed 8/4 schedule at demand ten.
                if shape.weight_bank_demand(kernels) <= 7 {
                    shape.demand_based_cbuf_partition(kernels)
                } else {
                    (8, 4)
                }
            }
            9 => {
                assert_large_kernel_plan_case(shape);
                if (33..=48).contains(&shape.in_channels) {
                    (7, 5)
                } else {
                    (6, 6)
                }
            }
            11 => {
                assert_large_kernel_plan_case(shape);
                match shape.in_channels {
                    1..=32 => (7, 5),
                    33..=48 => (5, 7),
                    _ => (3, 9),
                }
            }
            _ => unreachable!("kernel_programming accepted an unsupported kernel"),
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
    pub fn with_cbuf_banks(
        shape: Shape,
        kernels: Kernels,
        data_banks: u32,
        weight_banks: u32,
    ) -> ConvPlan {
        assert!(
            data_banks > 0 && weight_banks > 0 && data_banks + weight_banks == CBUF_BANKS,
            "explicit CBUF partition must have nonzero data and weight banks summing to \
             {CBUF_BANKS}; got data={data_banks}, weights={weight_banks}"
        );
        ConvPlan::new_with_cbuf_partition(shape, kernels, (data_banks, weight_banks))
    }

    fn new_with_cbuf_partition(
        shape: Shape,
        kernels: Kernels,
        (data_banks, weight_banks): (u32, u32),
    ) -> ConvPlan {
        shape
            .parity_padded_shape(kernels)
            .map(|_| ())
            .expect("unsupported accumulator output geometry");
        let full_width = vec![shape.output_width(kernels)];
        if let Some(tiles) = plan_grid(shape, kernels, &full_width, data_banks) {
            return ConvPlan {
                shape,
                kernels,
                data_banks,
                weight_banks,
                output_column_widths: full_width,
                tiles,
            };
        }

        assert_eq!(
            shape.layout(),
            FeatureLayout::Surfaces,
            "convolution needs horizontal tiling, which is only capture-backed for NC1HWC2 surfaces"
        );
        assert_eq!(
            shape.stride, 1,
            "convolution needs horizontal tiling, which is only capture-backed at stride 1"
        );

        if let Some(output_column_widths) = captured_column_partition(shape, kernels) {
            let tiles = Tile2D::grid(shape, kernels, &output_column_widths, data_banks);
            assert!(
                grid_fits(shape, kernels, &tiles, data_banks),
                "captured column partition exceeds its measured CBUF capacity"
            );
            return ConvPlan {
                shape,
                kernels,
                data_banks,
                weight_banks,
                output_column_widths,
                tiles,
            };
        }

        let output_width = shape.output_width(kernels);
        for column_count in 2..=output_width {
            let output_column_widths = balanced_column_widths(output_width, column_count);
            if let Some(tiles) = plan_grid(shape, kernels, &output_column_widths, data_banks) {
                return ConvPlan {
                    shape,
                    kernels,
                    data_banks,
                    weight_banks,
                    output_column_widths,
                    tiles,
                };
            }
        }

        panic!(
            "no standalone tile fits {shape:?} {kernels:?} with CBUF split \
             {data_banks}/{weight_banks}"
        );
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

    /// Emits one relocatable register program per planned tile.
    ///
    /// The programs still carry tile offsets rather than addresses; use
    /// [`ConvPlan::programs_with_buffers`] to get submission-ready ones.
    pub fn programs(&self) -> Vec<Vec<RegCmd>> {
        self.tiles
            .iter()
            .map(|tile| {
                conv_2d_tile_program(
                    self.shape,
                    self.kernels,
                    tile,
                    feature_grains_planned(self.shape, self.kernels, &tile.rows),
                    self.data_banks,
                    self.weight_banks,
                    OutputPlacement::SharedImage,
                )
            })
            .collect()
    }

    /// Emits one submission-ready register program per planned tile, bound to
    /// `buffers`.
    ///
    /// All tiles share the same four buffers: each program's own tile offsets
    /// are what select its slice of them, so there is no per-tile address
    /// arithmetic for the caller to do. Submit each program as its own job and
    /// wait for its fence before the next -- tiles reload their own weights,
    /// so no CBUF state has to survive between them.
    pub fn programs_with_buffers(&self, buffers: Buffers) -> Vec<Vec<RegCmd>> {
        self.programs()
            .into_iter()
            .map(|mut commands| {
                relocate(&mut commands, buffers);
                commands
            })
            .collect()
    }

    /// Emits accumulator programs whose outputs occupy independent,
    /// contiguous ranges of one private scratch buffer.
    ///
    /// A normal tile program addresses a sub-rectangle of a shared full-image
    /// surface: its destination surface stride and row notch therefore retain
    /// the full output geometry. Merely replacing its destination base is not
    /// enough to stage it independently. This entry point changes all three
    /// pieces together -- base, surface stride, and notch -- and returns the
    /// same tile layout the caller must use when compacting scratch into its
    /// logical output tensor.
    pub fn programs_with_staged_accumulator_output(
        &self,
        buffers: Buffers,
    ) -> StagedAccumulatorOutput {
        assert!(
            self.shape.precision.writes_accumulators(),
            "staged accumulator output requires Int8Accumulator precision"
        );

        let source_block_bytes = self.shape.output_atom_bytes() as usize;
        let padded_bytes_per_pixel = self.shape.padded_out_channels() as usize
            * self.shape.precision.output_element_bytes() as usize;
        let blocks_per_pixel = padded_bytes_per_pixel.div_ceil(source_block_bytes);
        let mut scratch_offset = 0usize;
        let mut programs = Vec::with_capacity(self.tiles.len());
        let mut output_tiles = Vec::with_capacity(self.tiles.len());

        for tile in &self.tiles {
            let tile_pixels = tile.rows.out_rows as usize * tile.columns.out_cols as usize;
            let scratch_bytes = tile_pixels
                .checked_mul(blocks_per_pixel)
                .and_then(|value| value.checked_mul(source_block_bytes))
                .expect("accumulator tile scratch size overflow");
            let local_output = buffers
                .output
                .checked_add(
                    u32::try_from(scratch_offset)
                        .expect("accumulator tile scratch offset exceeds u32"),
                )
                .expect("accumulator tile DMA address overflow");
            let mut program = conv_2d_tile_program(
                self.shape,
                self.kernels,
                tile,
                feature_grains_planned(self.shape, self.kernels, &tile.rows),
                self.data_banks,
                self.weight_banks,
                OutputPlacement::ContiguousTile,
            );
            relocate_with_exact_output(
                &mut program,
                Buffers {
                    output: local_output,
                    ..buffers
                },
            );
            programs.push(program);
            output_tiles.push(AccumulatorOutputTile {
                scratch_offset,
                scratch_bytes,
                output_row: tile.rows.out_first as usize,
                output_column: tile.columns.out_first as usize,
                output_rows: tile.rows.out_rows as usize,
                output_columns: tile.columns.out_cols as usize,
            });
            scratch_offset = scratch_offset
                .checked_add(scratch_bytes)
                .expect("accumulator scratch partition overflow");
        }

        assert_eq!(
            scratch_offset,
            self.shape.output_scratch_bytes(self.kernels),
            "staged accumulator tiles must partition the full scratch allocation"
        );
        StagedAccumulatorOutput {
            programs,
            tiles: output_tiles,
            scratch_bytes: scratch_offset,
        }
    }
}

fn assert_large_kernel_plan_case(shape: Shape) {
    assert!(
        matches!(shape.precision, Precision::Fp16),
        "automatic planning above 3x3 currently has capture backing only for fp16"
    );
    assert_eq!(
        shape.stride, 1,
        "automatic planning above 3x3 currently has capture backing only at stride 1"
    );
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

fn even_square_cbuf_partition(shape: Shape, kernels: Kernels) -> (u32, u32) {
    assert_eq!(
        shape.stride, 1,
        "even kernels currently have capture backing only at stride 1"
    );
    let demand = shape.weight_bank_demand(kernels);
    assert!(
        demand <= MAX_EVEN_SQUARE_DEMAND_BASED_WEIGHT_BANKS,
        "even square kernel {kernels:?} needs {demand} coefficient banks, above the \
         {MAX_EVEN_SQUARE_DEMAND_BASED_WEIGHT_BANKS} where the captured split follows \
         coefficient demand; use an explicit CBUF split"
    );
    shape.demand_based_cbuf_partition(kernels)
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
fn non_square_cbuf_partition(shape: Shape, kernels: Kernels) -> (u32, u32) {
    assert_eq!(
        shape.stride, 1,
        "non-square kernels currently have capture backing only at stride 1"
    );
    let kernel = shape.kernel_programming(kernels);
    let both_even = kernel.height.is_multiple_of(2) && kernel.width.is_multiple_of(2);
    let (limit, parity) = match (both_even, shape.precision) {
        (true, Precision::Fp16) => (MAX_EVEN_NON_SQUARE_FP16_DEMAND_BASED_WEIGHT_BANKS, "even"),
        (true, _) => (MAX_EVEN_NON_SQUARE_INT8_DEMAND_BASED_WEIGHT_BANKS, "even"),
        (false, _) => (MAX_NON_SQUARE_DEMAND_BASED_WEIGHT_BANKS, "non-square"),
    };
    let demand = shape.weight_bank_demand(kernels);
    assert!(
        demand <= limit,
        "{parity} kernel {kernels:?} needs {demand} coefficient banks, above the \
         {limit} where the captured split stops following coefficient demand; \
         use an explicit CBUF split"
    );
    shape.demand_based_cbuf_partition(kernels)
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

fn balanced_column_widths(output_width: u32, columns: u32) -> Vec<u32> {
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
        tile.rows.in_rows
            <= shape.max_tile_input_rows_for_width_and_data_banks(tile.columns.in_cols, data_banks)
            && feature_grains(kernels, &tile.rows) <= MAX_FEATURE_GRAINS
    })
}

#[inline]
fn zero<R: RegisterMeta>() -> RegCmd {
    Register::<R>::new().build()
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

/// One output channel's entry in the BS (bias/scale) buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BsEntry {
    /// `round(bias / (input_scale * weight_scale[c]))`.
    pub bias: i32,
    /// Addend applied to each raw coefficient for this output channel.
    /// Quantized affine weights use `-weight_zero_point[c]` so the hardware
    /// dot product sees `raw_weight - weight_zero_point[c]`.
    pub constant: i16,
    /// `round(BS_UNIT_MULTIPLIER * weight_scale[c] / max(weight_scale))`.
    pub multiplier: i16,
}

impl Default for BsEntry {
    /// A zero bias at unit multiplier -- what a convolution with uniform
    /// weight scales and no bias needs. Pair it with
    /// [`Multiplier::for_unit_bs`], which cancels the gain this carries.
    fn default() -> BsEntry {
        BsEntry {
            bias: 0,
            constant: BS_CONSTANT,
            multiplier: BS_UNIT_MULTIPLIER,
        }
    }
}

/// Bytes the BS buffer occupies for `out_channels` output channels.
///
/// Prefer [`Shape::bs_buffer_bytes`], which passes the padded count. BRDMA
/// has been observed reading past what the true count declares.
pub fn bs_buffer_bytes(out_channels: u32) -> usize {
    (out_channels as usize).div_ceil(BS_CHANNELS_PER_BLOCK) * BS_BLOCK_BYTES
}

/// Writes the BS buffer `DPU_RDMA_RDMA_BS_BASE_ADDR` points at.
///
/// Required for int8: `brdma_data_use` is 7 there rather than the fp16 1, so
/// BRDMA fetches a multiplier operand alongside the bias, and `bs_mul_src`
/// makes the BS stage use it. A zeroed buffer supplies a zero multiplier and
/// produces a zero output, which is why the fp16 tests' habit of zeroing the
/// bias buffer does not carry over.
///
/// The layout is planar within a block of eight output channels and repeats
/// per block, which is not what a flat array of per-channel structs would
/// look like -- it was read off three converted models whose biases and
/// per-channel weight magnitudes were varied independently.
pub fn write_bs_buffer(buffer: &mut [u8], entries: &[BsEntry]) {
    let needed = bs_buffer_bytes(entries.len() as u32);
    assert!(
        buffer.len() >= needed,
        "BS buffer is {} bytes, needs {needed} for {} channels",
        buffer.len(),
        entries.len()
    );
    buffer[..needed].fill(0);
    for (index, entry) in entries.iter().enumerate() {
        let block = index / BS_CHANNELS_PER_BLOCK;
        let lane = index % BS_CHANNELS_PER_BLOCK;
        let base = block * BS_BLOCK_BYTES;
        let bias = base + lane * 4;
        buffer[bias..bias + 4].copy_from_slice(&entry.bias.to_le_bytes());
        let constant = base + 32 + lane * 2;
        buffer[constant..constant + 2].copy_from_slice(&entry.constant.to_le_bytes());
        let multiplier = base + 48 + lane * 2;
        buffer[multiplier..multiplier + 2].copy_from_slice(&entry.multiplier.to_le_bytes());
    }
}

/// Converts a logical quantized bias vector into Rocket's physical BS buffer.
///
/// The logical ABI is little-endian `i32` bias values in accumulator units.
/// Rocket's BS bias plane is normalized by the input and weight scales, while
/// its constant plane supplies the affine weight correction. Weight scales
/// are currently per-tensor in the executable ABI; per-channel scales can be
/// added without changing the physical writer.
pub fn pack_int8_bias_to_bs(
    dense: &[u8],
    output_channels: usize,
    padded_output_channels: usize,
    input_scale: f32,
    weights_scale: f32,
    weight_zero_point: i8,
    packed: &mut [u8],
) -> Result<usize, &'static str> {
    if dense.len() < output_channels.saturating_mul(4) {
        return Err("int8 bias is smaller than its declared shape");
    }
    if padded_output_channels < output_channels {
        return Err("padded int8 bias channels are smaller than logical channels");
    }
    let scale = f64::from(input_scale) * f64::from(weights_scale);
    if !scale.is_finite() || scale <= 0.0 {
        return Err("int8 bias scales must be finite and positive");
    }
    let needed = bs_buffer_bytes(padded_output_channels as u32);
    if packed.len() < needed {
        return Err("Rocket int8 BS destination is smaller than its declared shape");
    }
    let mut entries = vec![BsEntry::default(); padded_output_channels];
    for (channel, entry) in entries.iter_mut().take(output_channels).enumerate() {
        let offset = channel * 4;
        let bias = i32::from_le_bytes(dense[offset..offset + 4].try_into().unwrap());
        let normalized = (f64::from(bias) / scale).round();
        if normalized < f64::from(i32::MIN) || normalized > f64::from(i32::MAX) {
            return Err("int8 bias normalization overflows i32");
        }
        entry.bias = normalized as i32;
        entry.constant = -(i16::from(weight_zero_point));
    }
    write_bs_buffer(&mut packed[..needed], &entries);
    Ok(needed)
}

/// The four DMA base addresses a conv program reads and writes through.
///
/// Programs come out of this module carrying tile *offsets* in their address
/// registers, not addresses -- see [`conv_2d_tile`]. [`relocate`] binds them
/// to real memory, which is the last step before submission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Buffers {
    /// Feature data, `CNA_FEATURE_DATA_ADDR`.
    pub input: u32,
    /// Packed weights, `CNA_DCOMP_ADDR0`. See
    /// [`crate::rocket::tensor_layout::pack_hwcf_to_rocket_weights`].
    pub weights: u32,
    /// Bias, and for int8 the per-channel multiplier alongside it,
    /// `DPU_RDMA_RDMA_BS_BASE_ADDR`. See [`write_bs_buffer`].
    pub bias: u32,
    /// Output feature data, `DPU_DST_BASE_ADDR`.
    pub output: u32,
}

fn decode_identity(command: &RegCmd) -> (u32, u32) {
    ((command.0 >> 48) as u32, command.0 as u32 & 0xffff)
}

/// Binds one address register to `address`, keeping whatever tile offset the
/// program already put there.
///
/// Matching by typed register identity rather than a hardcoded command index
/// is what makes this fail loudly if a program is ever reordered or gains a
/// second write to the same address register, instead of quietly relocating
/// the wrong word.
fn relocate_one<R: RegisterMeta>(commands: &mut [RegCmd], address: u32, keep_tile_offset: bool) {
    assert_eq!(
        address & 0xf,
        0,
        "NPU DMA address for register {:#x}:{:#x} is not 16-byte aligned",
        R::DOMAIN,
        R::OFFSET
    );

    let matches: Vec<_> = commands
        .iter()
        .enumerate()
        .filter_map(|(index, command)| {
            (decode_identity(command) == (R::DOMAIN, R::OFFSET)).then_some(index)
        })
        .collect();
    assert_eq!(
        matches.len(),
        1,
        "expected exactly one {:#x}:{:#x} relocation, found {matches:?}",
        R::DOMAIN,
        R::OFFSET
    );

    // Normally add rather than overwrite: a tile program already carries its
    // own byte offset from the tensor base in these registers, exactly as the
    // vendor's own height-split programs do. Independently staged output
    // tiles are the exception and bind DPU_DST_BASE_ADDR directly.
    let tile_offset = keep_tile_offset
        .then(|| (commands[matches[0]].0 >> 16) as u32)
        .unwrap_or(0);
    commands[matches[0]] = RegCmd::new(R::DOMAIN, R::OFFSET, address + tile_offset);
}

/// Binds a program's four address registers to real memory, in place.
///
/// Every address must be 16-byte aligned -- the DPU addresses output in
/// 16-byte feature atoms, and the fetch side reads them the same way, so an
/// unaligned base silently shears every surface. Panics rather than
/// truncating.
///
/// A program carries exactly one write of each of the four registers, so
/// relocating twice would double the offsets; this is a one-shot step on a
/// freshly built program, not something to reapply.
pub fn relocate(commands: &mut [RegCmd], buffers: Buffers) {
    relocate_one::<CnaFeatureDataAddr>(commands, buffers.input, true);
    relocate_one::<CnaDcompAddr0>(commands, buffers.weights, true);
    relocate_one::<DpuRdmaBsBaseAddr>(commands, buffers.bias, true);
    relocate_one::<DpuDstBaseAddr>(commands, buffers.output, true);
}

/// Binds a program while replacing its output tile offset with an exact DMA
/// address.
///
/// This is for callers that stage each output tile in a separate contiguous
/// scratch range. Input, weight, and bias addresses retain the tile offsets
/// carried by the program; only `buffers.output` is used verbatim.
pub fn relocate_with_exact_output(commands: &mut [RegCmd], buffers: Buffers) {
    relocate_one::<CnaFeatureDataAddr>(commands, buffers.input, true);
    relocate_one::<CnaDcompAddr0>(commands, buffers.weights, true);
    relocate_one::<DpuRdmaBsBaseAddr>(commands, buffers.bias, true);
    relocate_one::<DpuDstBaseAddr>(commands, buffers.output, false);
}

/// 2-3 and 4-6 are alternative two- and three-core height-split programs,
/// not continuations of this command stream.
pub fn conv_2d(kernels: Kernels) -> Vec<RegCmd> {
    let shape = Shape::CAPTURED;
    conv_2d_tile(shape, kernels, &Tile::whole(shape, kernels))
}

/// Builds the single-core regcmd program for one output-row `tile`.
///
/// `conv_2d(k)` is `conv_2d_tile(k, &Tile::whole(k))` and reproduces captured
/// group 1 bit for bit. For a partial tile, the sixteen tile-dependent
/// registers take the values derived from the cross-group capture diff; see
/// the module documentation for what a tile program does and does not match.
///
/// The feature and destination address registers carry this tile's *offset*
/// from the tensor base, exactly as the vendor's own split programs do. A
/// caller relocating these must add the buffer's DMA address to the existing
/// value rather than overwrite it.
pub fn conv_2d_tile(shape: Shape, kernels: Kernels, tile: &Tile) -> Vec<RegCmd> {
    conv_2d_tile_with_grains(shape, kernels, tile, feature_grains(kernels, tile))
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

/// [`feature_grains`], with the characterization override applied.
///
/// Only `ConvPlan` uses this. The override is gated on `Cin` so a probe can
/// change the shapes under study without disturbing the health canary the
/// hardware harness runs first -- forcing a global value breaks known-good
/// shapes, which is itself evidence that the prefetch is not a free parameter.
fn feature_grains_planned(shape: Shape, kernels: Kernels, tile: &Tile) -> u32 {
    let programmed = feature_grains(kernels, tile);
    if shape.in_channels < grains_override_min_channels() {
        return programmed;
    }
    match grains_override() {
        Some(GrainsOverride::Exact(value)) => value,
        Some(GrainsOverride::Cap(cap)) => programmed.min(cap),
        None => programmed,
    }
}

/// Lowest `Cin` the grains override applies to (`ROCKET_FEATURE_GRAINS_MIN_CIN`,
/// default 0).
fn grains_override_min_channels() -> u32 {
    std::env::var("ROCKET_FEATURE_GRAINS_MIN_CIN")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}

/// Test-only override of the programmed prefetch. See [`grains_override`].
enum GrainsOverride {
    Exact(u32),
    Cap(u32),
}

/// Lets a hardware probe drive `feature_grains` through the *whole* ConvPlan
/// path rather than hand-building one tile, which is what
/// `conv_2d_tile_with_grains` already allows for a single tile.
///
/// This exists because our value and the vendor's disagree systematically --
/// ours is rows-driven and stays near 33, while the vendor drives it down with
/// channel pressure (33, 16, 9, ... 6) -- and no gate test compares the field.
/// `ROCKET_FEATURE_GRAINS=<n>` pins it; `ROCKET_FEATURE_GRAINS_MAX=<n>` clamps
/// it. Nothing on the compiled path sets either.
fn grains_override() -> Option<GrainsOverride> {
    if let Ok(value) = std::env::var("ROCKET_FEATURE_GRAINS") {
        if let Ok(parsed) = value.parse() {
            return Some(GrainsOverride::Exact(parsed));
        }
    }
    if let Ok(value) = std::env::var("ROCKET_FEATURE_GRAINS_MAX") {
        if let Ok(parsed) = value.parse() {
            return Some(GrainsOverride::Cap(parsed));
        }
    }
    None
}

/// Characterization override for the accumulator `DPU_SURFACE_ADD.surf_add`.
///
/// Exceeding the per-channel coefficient limit raises a DMA **read** error and
/// stalls the rk_iommu, and the accumulator output registers are the only part
/// of the program with no vendor capture behind them. Nothing on the compiled
/// path sets this.
///
/// Gated by `ROCKET_ACC_SURF_ADD_MIN_CIN` because the hardware harness runs a
/// low-`Cin` accumulator canary first: overriding globally breaks that canary
/// and the run aborts as "device sick" rather than measuring anything. That
/// the canary breaks at all is itself the finding that 16 is load-bearing.
fn accumulator_surf_add_override(in_channels: u32) -> Option<u32> {
    let min_channels: u32 = std::env::var("ROCKET_ACC_SURF_ADD_MIN_CIN")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    if in_channels < min_channels {
        return None;
    }
    std::env::var("ROCKET_ACC_SURF_ADD")
        .ok()
        .and_then(|value| value.parse().ok())
}

/// Characterization overrides scoped to [`Precision::Int4`].
///
/// int4's compute is exact on hardware but its write-out stops after two
/// 256-byte atoms, which is a writer-geometry question -- the same class the
/// int8 accumulator's `mc_surf_out` / `size_e` / `surf_add` triple turned
/// out to be. Scoping these to int4 keeps every other precision, including
/// the hardware harness's health canary, on the shipped program, so a sweep
/// measures int4 rather than breaking the instrument.
///
/// `ROCKET_INT4_{OUT_PRECISION,MC_SURF_OUT,SIZE_E,SURF_ADD}`. Nothing on the
/// compiled path sets any of them.
fn int4_override(precision: Precision, name: &str) -> Option<u32> {
    if precision != Precision::Int4 {
        return None;
    }
    std::env::var(format!("ROCKET_INT4_{name}"))
        .ok()
        .and_then(|value| value.parse().ok())
}

/// Characterization override for `DPU_BS_OW_CFG.SIZE_E_0/1/2`
/// (`ROCKET_ACC_SIZE_E`, gated by `ROCKET_ACC_SIZE_E_MIN_CIN`).
///
/// Kept, like [`accumulator_surf_add_override`], because it carries a settled
/// negative result rather than an open question. Nothing on the compiled path
/// sets it. The `_MIN_CIN` gate exists so the harness's low-`Cin` accumulator
/// canary stays on the shipped program; without it a sweep aborts as "device
/// sick" instead of measuring anything.
///
/// **What it settled** [HW sweep, planck 2026-09-03]. `size_e` is a BS/OW-stage
/// field and the int32-accumulator path bypasses that stage, so the override is
/// inert there: 0, 1, 3 and 7 all produce a byte-identical, bit-exact result at
/// 32x32 Cin=384, and all four produce the identical 6144+512-byte truncation at
/// Cin=385. It is emphatically *not* inert on the requantized int8 path, which
/// leaves `OD_BYPASS` clear -- 3 and 7 there write 1024 of 65536 bytes and hang
/// the job. See [`Shape::bs_ow_size_e`] for the table.
///
/// So `rockchip-npu-notes/encodings/size-e-quirk.md`'s "integer outputs stride
/// as `size_e = 7` regardless of byte width" is a fact about a path that keeps
/// the OW stage engaged, and does not carry to this one. The accumulator
/// truncation in [`MAX_ACCUMULATOR_COEFFICIENT_BYTES_PER_CHANNEL`] is still
/// unexplained, and this is one more register eliminated: with `size_e` inert
/// and `surf_add` swept, the output-side register archaeology is exhausted.
fn bs_ow_size_e_override(in_channels: u32) -> Option<u32> {
    let min_channels: u32 = std::env::var("ROCKET_ACC_SIZE_E_MIN_CIN")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    if in_channels < min_channels {
        return None;
    }
    std::env::var("ROCKET_ACC_SIZE_E")
        .ok()
        .and_then(|value| value.parse().ok())
}

/// Characterization override selecting rocket-userspace's `surf_add` *rule*
/// rather than a constant (`ROCKET_ACC_SURF_MULT`, gated by
/// `ROCKET_ACC_SIZE_E_MIN_CIN`).
///
/// `gen_matmul_int8` sets `surf_add = dst_surf_stride * 8` with
/// `dst_surf_stride = dataout_height * dataout_width` **of the task**. On a
/// height-tiled plan every tile has a different `out_rows`, so no single
/// `ROCKET_ACC_SURF_ADD` constant can reproduce it -- which is why the constant
/// sweep recorded in `accumulator-per-channel-coefficient-limit` could not have
/// found this even in principle. This applies the rule per tile.
fn accumulator_surf_mult_override(in_channels: u32) -> Option<u32> {
    let min_channels: u32 = std::env::var("ROCKET_ACC_SIZE_E_MIN_CIN")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    if in_channels < min_channels {
        return None;
    }
    std::env::var("ROCKET_ACC_SURF_MULT")
        .ok()
        .and_then(|value| value.parse().ok())
}

/// Surface multiplier for dense int32-accumulator output: `surf_add =
/// dataout_width * dataout_height * 8`, per task.
///
/// The 8 is the integer-output stride quirk, the same one behind
/// [`Shape::bs_ow_size_e`]'s 7: the writer strides as if each output element
/// were 8 bytes even though an int32 is 4. HW-validated here at 32x32 Cin 384
/// [planck 2026-09-03] -- mult 8 writes 100% of the buffer bit-exactly, and
/// 4 / 2 / 1 write 75% / 62.5% / 56.2%, leaving the rest at the poison
/// sentinel, exactly as `rocket-userspace`'s `gen_matmul_int8` header warns
/// ("halves the surface stride, leaving every output column past the first few
/// surfaces as the `0xAA` sentinel").
const DENSE_ACCUMULATOR_SURF_MULT: u32 = 8;

/// Characterization override for `DPU_DATA_FORMAT.mc_surf_out`
/// (`ROCKET_ACC_MC_SURF_OUT`, gated by `ROCKET_ACC_SIZE_E_MIN_CIN`).
///
/// The third knob of the accumulator output writer, and the one that makes the
/// other two readable. `rocket-userspace/include/npu_dpu.h` documents the field
/// as `0 = 16B/pixel one surface, 1 = 2/4 surf serial`, and its HW-validated
/// int8 -> int32 matmul (`gen_matmul_int8`) leaves it **0** while using
/// `size_e = 7` and `surf_add = dst_surf_stride * 8`. This crate's accumulator
/// mode instead sets it to **1** with `size_e = 1` and `surf_add = 16`, which is
/// a different writer, not a variant of the same one.
///
/// That is why sweeping `surf_add` alone and `size_e` alone both read as "no
/// effect / no value helps": in the serial writer there is no surface stride for
/// either field to describe. The three move together or not at all.
fn accumulator_mc_surf_out_override(in_channels: u32) -> Option<u32> {
    let min_channels: u32 = std::env::var("ROCKET_ACC_SIZE_E_MIN_CIN")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    if in_channels < min_channels {
        return None;
    }
    std::env::var("ROCKET_ACC_MC_SURF_OUT")
        .ok()
        .and_then(|value| value.parse().ok())
}

/// Builds a tile program with an explicit `feature_grains`, for probing which
/// values the hardware accepts. Prefer [`conv_2d_tile`].
pub fn conv_2d_tile_with_grains(
    shape: Shape,
    kernels: Kernels,
    tile: &Tile,
    feature_grains: u32,
) -> Vec<RegCmd> {
    assert_default_cbuf_kernel(kernels);
    let tile = Tile2D {
        rows: *tile,
        columns: ColumnTile::whole(shape, kernels),
    };
    conv_2d_tile_program(
        shape,
        kernels,
        &tile,
        feature_grains,
        shape.data_banks(kernels),
        shape.weight_banks(kernels),
        OutputPlacement::SharedImage,
    )
}

/// Builds a large-kernel program with an explicit CBUF partition.
///
/// The focused kernel sweep shows that 7x7 switches away from demand once
/// coefficient demand exceeds seven banks, while 9x9 and 11x11 use their own
/// streaming schedules. This entry point keeps unresolved policy out of
/// [`conv_2d_tile`]. Prefer [`ConvPlan`] when its capture-backed policy covers
/// the operation; this entry point is the low-level override. The two bank
/// counts must be nonzero and sum to the RK3588's twelve CBUF banks.
pub fn conv_2d_tile_with_cbuf_banks(
    shape: Shape,
    kernels: Kernels,
    tile: &Tile,
    data_banks: u32,
    weight_banks: u32,
) -> Vec<RegCmd> {
    assert!(
        data_banks > 0 && weight_banks > 0 && data_banks + weight_banks == CBUF_BANKS,
        "explicit CBUF partition must have nonzero data and weight banks summing to {CBUF_BANKS}; \
         got data={data_banks}, weights={weight_banks}"
    );
    let tile = Tile2D {
        rows: *tile,
        columns: ColumnTile::whole(shape, kernels),
    };
    assert!(
        tile.rows.in_rows <= shape.max_tile_input_rows_for_data_banks(data_banks),
        "tile reads {} input rows, but {data_banks} data banks fit at most {} for this shape",
        tile.rows.in_rows,
        shape.max_tile_input_rows_for_data_banks(data_banks),
    );
    conv_2d_tile_program(
        shape,
        kernels,
        &tile,
        feature_grains(kernels, &tile.rows),
        data_banks,
        weight_banks,
        OutputPlacement::SharedImage,
    )
}

/// Builds one rectangular large-kernel tile with an explicit CBUF split.
///
/// Horizontal tiles use the capture-derived grouped-line DMA mode and retain
/// the full tensor strides. Prefer [`conv_2d_tile_with_cbuf_banks`] for a
/// full-width row tile.
pub fn conv_2d_tile_2d_with_cbuf_banks(
    shape: Shape,
    kernels: Kernels,
    tile: &Tile2D,
    data_banks: u32,
    weight_banks: u32,
) -> Vec<RegCmd> {
    assert_eq!(
        shape.layout(),
        FeatureLayout::Surfaces,
        "horizontal tiling currently has capture backing only for NC1HWC2 surfaces"
    );
    assert_eq!(
        shape.stride, 1,
        "horizontal tiling currently has capture backing only at stride 1"
    );
    assert!(
        data_banks > 0 && weight_banks > 0 && data_banks + weight_banks == CBUF_BANKS,
        "explicit CBUF partition must have nonzero data and weight banks summing to {CBUF_BANKS}; \
         got data={data_banks}, weights={weight_banks}"
    );
    let max_rows =
        shape.max_tile_input_rows_for_width_and_data_banks(tile.columns.in_cols, data_banks);
    assert!(
        tile.rows.in_rows <= max_rows,
        "tile reads {}x{} input pixels, but {data_banks} data banks fit at most \
         {max_rows} rows at this width",
        tile.columns.in_cols,
        tile.rows.in_rows,
    );
    conv_2d_tile_program(
        shape,
        kernels,
        tile,
        feature_grains(kernels, &tile.rows),
        data_banks,
        weight_banks,
        OutputPlacement::SharedImage,
    )
}

fn conv_2d_tile_program(
    shape: Shape,
    kernels: Kernels,
    tile: &Tile2D,
    feature_grains: u32,
    data_banks: u32,
    weight_banks: u32,
    output_placement: OutputPlacement,
) -> Vec<RegCmd> {
    let padded_channels = shape.padded_channels();
    let weight_channels = shape.weight_channels();
    // The DPU counts output channels in whole granules while the CNA counts
    // the real kernels. Both appear below, and they differ at every Cout
    // that is not already a multiple of the granule.
    let padded_out_channels = shape.padded_out_channels();
    let (bn_bypass, bn_relu_bypass, bn_relux_en, bn_relux_cmp) = shape.activation.bn_programming();

    // Precision reaches the program in three ways: an enum replicated across
    // eight fields in four blocks, a set of bypasses that the quantized path
    // clears, and the requantization constants themselves.
    let precision = shape.precision.data_precision();
    let quantization = shape.precision.quantization();
    let accumulator_output = shape.precision.writes_accumulators();
    let output_precision: Bits<3> = match int4_override(shape.precision, "OUT_PRECISION") {
        Some(value) => Bits::new(value),
        None => shape.precision.output_data_precision().into(),
    };
    // `BS_MUL_SHIFT_VALUE` and its negated twin in `DPU_DATA_FORMAT` are a
    // constant 14 in every int8 capture and 0 in every fp16 one. Nothing in
    // the corpus varies it, so it is not derived from anything.
    let bs_mul_shift = if quantization.is_some() {
        BS_MUL_SHIFT_VALUE
    } else {
        0
    };
    // BRDMA carries bias alone at fp16 and the full bias/scale/shift triple
    // once requantization is active.
    let brdma_data_use = if quantization.is_some() {
        BRDMA_DATA_USE_QUANTIZED
    } else {
        BRDMA_DATA_USE_BIAS
    };
    // Bits, not bytes: a half-byte element cannot express a per-kernel
    // footprint as a byte count times a channel count.
    let element_bits = shape.precision.element_bits();

    let rows = &tile.rows;
    let columns = &tile.columns;
    let full_width = shape.width;
    let height = shape.height;
    let full_out_width = shape.output_width(kernels);
    let out_height = shape.output_height(kernels);
    let input_width = columns.in_cols;
    let out_width = columns.out_cols;
    let horizontally_tiled = columns.out_first != 0 || out_width != full_out_width;
    let (output_base_offset, output_surface_pixels, output_notch) = match output_placement {
        OutputPlacement::SharedImage => (
            tile.output_offset(shape, kernels),
            full_out_width * out_height,
            full_out_width - out_width,
        ),
        OutputPlacement::ContiguousTile => {
            assert!(
                accumulator_output,
                "contiguous tile output is only validated for Int8Accumulator precision"
            );
            (0, out_width * rows.out_rows, 0)
        }
    };

    assert!(
        feature_grains <= MAX_FEATURE_GRAINS,
        "tile requires {feature_grains} feature grains; CNA_CONV_CON2.feature_grains encodes at most {MAX_FEATURE_GRAINS}"
    );

    assert!(
        rows.out_rows > 0 && rows.out_first + rows.out_rows <= out_height,
        "tile output rows {}..{} fall outside the {out_height}-row output",
        rows.out_first,
        rows.out_first + rows.out_rows
    );
    assert!(
        columns.out_cols > 0 && columns.out_first + columns.out_cols <= full_out_width,
        "tile output columns {}..{} fall outside the {full_out_width}-column output",
        columns.out_first,
        columns.out_first + columns.out_cols
    );
    let charged_input_width = shape.cbuf_input_width(input_width);
    assert!(
        rows.in_rows * charged_input_width <= shape.max_data_entries(),
        "tile reads {}x{} charged pixels; CNA_CBUF_CON1.data_entries holds at most {} \
         at {:?}",
        charged_input_width,
        rows.in_rows,
        shape.max_data_entries(),
        shape.precision,
    );
    assert!(
        rows.in_rows > 0 && rows.in_first + rows.in_rows <= height,
        "tile input rows {}..{} fall outside the {height}-row image",
        rows.in_first,
        rows.in_first + rows.in_rows
    );
    assert!(
        columns.in_cols > 0 && columns.in_first + columns.in_cols <= full_width,
        "tile input columns {}..{} fall outside the {full_width}-column image",
        columns.in_first,
        columns.in_first + columns.in_cols
    );

    let kernel = shape.kernel_programming(kernels);

    // Layout-dependent programming. Dense rows are counted in pixels and the
    // whole tile is resident, so `data_entries` scales with the tile height.
    // Surfaces are counted in atoms and `data_entries` does not depend on the
    // tile at all -- the same field carries different quantities in the two
    // regimes, which is why they are computed apart rather than parameterised.
    //
    // The surface `data_entries` charge packs 4 atoms per entry and rounds
    // *up*: vendor captures at width 13/29/30/31 (Cin=8, one atom/pixel)
    // program 4/8/8/8 respectively, not the 3/7/7/7 floor division gives.
    // Every capture before these was at a width a multiple of 4, where floor
    // and ceiling agree, which is how a real compiled model first exposed
    // this as scattered-pixel corruption on hardware.
    let (line_stride, surf_stride, data_entries) = match (shape.layout(), horizontally_tiled) {
        (FeatureLayout::Dense, _) => (
            full_width,
            full_width * (height - 1),
            charged_input_width * rows.in_rows,
        ),
        (FeatureLayout::Surfaces, false) => (
            full_width * 4,
            // This field is a 28-bit signed/bias-style encoding, despite the
            // register definition exposing it as unsigned. The original
            // image corpus never went below four rows, so ordinary unsigned
            // subtraction happened to reproduce it. A 160-model FC sweep
            // maps M to width and uses height=1; RKNN then writes
            // `M * (1 - 4)` modulo 2^28 (M=4 is 0x0fff_fff4).
            //
            // Keep the arithmetic wrapping here and let the typed register
            // field mask it to 28 bits below. This is the vendor encoding,
            // not an attempt to use a negative byte stride in host memory.
            full_width.wrapping_mul(height.wrapping_sub(4)) & 0x0fff_ffff,
            (input_width * shape.cbuf_atoms()).div_ceil(CBUF_ATOMS_PER_ENTRY),
        ),
        // Every captured width-partitioned task enables grouped-line mode
        // below and switches to these strides. `surf_stride` is the full
        // surface area less the local input width: exact for 125/139 at 9x9,
        // 124/142 at 11x11/Cin48, and 40/64 at 11x11/Cin64.
        (FeatureLayout::Surfaces, true) => (
            full_width,
            full_width * height - input_width,
            (input_width * shape.cbuf_atoms()).div_ceil(CBUF_ATOMS_PER_ENTRY),
        ),
    };

    let weight_bytes_per_kernel = kernel.height * kernel.width * weight_channels * element_bits / 8;
    let weight_bytes = shape.weight_bytes(kernels);
    let mut commands = Vec::with_capacity(136);

    // CNA preamble, followed by the DPU/DPU_RDMA ping-pong pointers.
    let mut cbuf_con0 = Register::<CnaCbufCon0>::new();
    cbuf_con0
        .weight_bank(Bits::new(weight_banks))
        .data_bank(Bits::new(data_banks));
    commands.push(cbuf_con0.build());
    commands.push(zero::<CnaDcompRegnum>());
    commands.push(zero::<CnaDcompCtrl>());

    // The dense regime is the CNA's ARGB image-input path: `argb_in` names
    // the channel count (OneChannel = 8 through FourChannels = 11) and both
    // `nonalign_dma` and `group_line_off` are set. The full-width surface
    // regime clears all three. Horizontal surface tiles set `group_line_off`
    // while leaving the other two clear, exactly as every width-partitioned
    // task in the focused kernel captures does.
    //
    // Leaving these at the captured C3 values made the hardware read three
    // channels per pixel at every channel count, which is what
    // `conv_multichannel_hw` caught.
    let mut conv_con1 = Register::<CnaConvCon1>::new();
    match shape.layout() {
        FeatureLayout::Dense => {
            conv_con1
                .nonalign_dma(Bits::new(1))
                .group_line_off(Bits::new(1))
                .argb_in(argb_input_mode(shape.in_channels).into());
        }
        FeatureLayout::Surfaces => {
            conv_con1
                .nonalign_dma(Bits::new(0))
                .group_line_off(Bits::new(u32::from(horizontally_tiled)))
                .argb_in(Bits::new(0));
        }
    }
    conv_con1
        .proc_precision(precision.into())
        .in_precision(precision.into())
        .conv_mode(Bits::new(shape.conv_mode()));
    commands.push(conv_con1.build());
    commands.push(
        Register::<DpuSPointer>::new()
            .pointer_pp_mode(Bits::new(1))
            .executer_pp_en(Bits::new(1))
            .pointer_pp_en(Bits::new(1))
            .build(),
    );
    commands.push(
        Register::<DpuRdmaSPointer>::new()
            .pointer_pp_mode(Bits::new(1))
            .executer_pp_en(Bits::new(1))
            .pointer_pp_en(Bits::new(1))
            .build(),
    );

    // CNA convolution and DMA programming.
    commands.push(conv_con1.build());
    commands.push(
        Register::<CnaConvCon2>::new()
            .feature_grains(Bits::new(feature_grains))
            .build(),
    );
    commands.push(
        Register::<CnaConvCon3>::new()
            .conv_x_stride(Bits::new(shape.stride))
            .conv_y_stride(Bits::new(shape.stride))
            .build(),
    );
    commands.push(
        Register::<CnaDataSize0>::new()
            .datain_width(Bits::new(input_width))
            .datain_height(Bits::new(rows.in_rows))
            .build(),
    );
    commands.push(
        Register::<CnaDataSize1>::new()
            .datain_channel_real(Bits::new((shape.in_channels - 1) % CHANNEL_REAL_MODULUS))
            .datain_channel(Bits::new(padded_channels))
            .build(),
    );
    commands.push(
        Register::<CnaDataSize2>::new()
            .dataout_width(Bits::new(out_width))
            .build(),
    );
    commands.push(
        Register::<CnaDataSize3>::new()
            .dataout_atomics(Bits::new(out_width * rows.out_rows))
            .build(),
    );
    commands.push(
        Register::<CnaWeightSize0>::new()
            .weight_bytes(Bits::new(weight_bytes))
            .build(),
    );
    commands.push(
        Register::<CnaWeightSize1>::new()
            .weight_bytes_per_kernel(Bits::new(weight_bytes_per_kernel))
            .build(),
    );
    commands.push(
        Register::<CnaWeightSize2>::new()
            .weight_width(Bits::new(kernel.width))
            .weight_height(Bits::new(kernel.height))
            .weight_kernels(Bits::new(shape.programmed_kernels()))
            .build(),
    );
    commands.push(cbuf_con0.build());
    commands.push(
        Register::<CnaCbufCon1>::new()
            .data_entries(Bits::new(data_entries))
            .build(),
    );
    commands.push(
        Register::<CnaCvtCon0>::new()
            .data_sign(Bits::new(1))
            .cvt_type(Bits::new(1))
            .cvt_bypass(Bits::new(1))
            .build(),
    );
    commands.push(
        Register::<CnaCvtCon1>::new()
            .cvt_scale0(Bits::new(1))
            .build(),
    );
    commands.push(
        Register::<CnaCvtCon2>::new()
            .cvt_scale1(Bits::new(1))
            .build(),
    );
    commands.push(
        Register::<CnaCvtCon3>::new()
            .cvt_scale2(Bits::new(1))
            .build(),
    );
    commands.push(
        Register::<CnaCvtCon4>::new()
            .cvt_scale3(Bits::new(1))
            .build(),
    );
    commands.push(zero::<CnaFcCon0>());
    commands.push(zero::<CnaFcCon1>());
    commands.push(
        // Both axes carry only the padding still visible at the tile's first
        // output coordinate. Interior horizontal tiles clear `pad_left`.
        Register::<CnaPadCon0>::new()
            .pad_top(Bits::new(rows.pad_top))
            .pad_left(Bits::new(columns.pad_left))
            .build(),
    );
    commands.push(
        Register::<CnaFeatureDataAddr>::new()
            .feature_base_addr(Bits::new(tile.input_offset(shape)))
            .build(),
    );
    commands.push(zero::<CnaFcCon2>());
    commands.push(
        Register::<CnaDmaCon0>::new()
            .data_burst_len(BurstLength::Sixteen.into())
            .weight_burst_len(BurstLength::Sixteen.into())
            .build(),
    );
    commands.push(
        Register::<CnaDmaCon1>::new()
            .line_stride(Bits::new(line_stride))
            .build(),
    );
    commands.push(
        Register::<CnaDmaCon2>::new()
            .surf_stride(Bits::new(surf_stride))
            .build(),
    );
    commands.push(
        Register::<CnaFcDataSize0>::new()
            .dma_width(Bits::new(input_width))
            .dma_height(Bits::new(rows.in_rows))
            .build(),
    );
    commands.push(
        Register::<CnaFcDataSize1>::new()
            .dma_channel(Bits::new(padded_channels))
            .build(),
    );
    commands.push(zero::<CnaDcompCtrl>());
    commands.push(zero::<CnaDcompRegnum>());
    commands.push(zero::<CnaDcompAddr0>());
    commands.push(zero::<CnaDcompAmount0>());
    commands.push(zero::<CnaDcompAmount1>());
    commands.push(zero::<CnaDcompAmount2>());
    commands.push(zero::<CnaDcompAmount3>());
    commands.push(zero::<CnaDcompAmount4>());
    commands.push(zero::<CnaDcompAmount5>());
    commands.push(zero::<CnaDcompAmount6>());
    commands.push(zero::<CnaDcompAmount7>());
    commands.push(zero::<CnaDcompAmount8>());
    commands.push(zero::<CnaDcompAmount9>());
    commands.push(zero::<CnaDcompAmount10>());
    commands.push(zero::<CnaDcompAmount11>());
    commands.push(zero::<CnaDcompAmount12>());
    commands.push(zero::<CnaDcompAmount13>());
    commands.push(zero::<CnaDcompAmount14>());
    commands.push(zero::<CnaDcompAmount15>());
    commands.push(zero::<CnaCvtCon5>());
    // Out-of-image taps contribute the quantized encoding of 0.0, which is
    // the input zero point and not zero. fp16 pads with a literal 0 in every
    // capture; int8 pads with the zero point in every capture.
    commands.push(
        Register::<CnaPadCon1>::new()
            .pad_value(Bits::new(
                quantization.map_or(0, |q| q.input_zero_point as u32),
            ))
            .build(),
    );

    // CORE.
    commands.push(
        Register::<CoreMiscCfg>::new()
            .proc_precision(precision.into())
            .qd_en(Bits::new(u32::from(quantization.is_some())))
            .dw_en(Bits::new(u32::from(shape.depthwise)))
            .build(),
    );
    commands.push(
        Register::<CoreDataoutSize0>::new()
            .dataout_width(Bits::new(out_width - 1))
            .dataout_height(Bits::new(rows.out_rows - 1))
            .build(),
    );
    commands.push(
        Register::<CoreDataoutSize1>::new()
            .dataout_channel(Bits::new(padded_out_channels - 1))
            .build(),
    );
    commands.push(zero::<CoreClipTruncate>());
    commands.push(zero::<CoreReserved3030>());

    // DPU output, conversion, and disabled LUT programming.
    commands.push(
        Register::<DpuFeatureModeCfg>::new()
            .burst_len(BurstLength::Sixteen.into())
            .output_mode(DpuOutputMode::ExternalMemory.into())
            .conv_mode(Bits::new(shape.conv_mode()))
            .build(),
    );
    commands.push(
        Register::<DpuDataFormat>::new()
            .in_precision(precision.into())
            .out_precision(output_precision)
            .proc_precision(precision.into())
            // 0 selects the "16 B/pixel, one surface" writer, 1 the "2/4
            // surface serial" one. Dense accumulator output uses 0, matching
            // `rocket-userspace`'s validated int8 -> int32 program; only
            // depthwise accumulator output is still on the serial writer,
            // which is the configuration its 256-byte write atom was measured
            // under. Every non-accumulator path has always used 0.
            .mc_surf_out(Bits::new(
                int4_override(shape.precision, "MC_SURF_OUT")
                    .or_else(|| accumulator_mc_surf_out_override(shape.in_channels))
                    .unwrap_or(u32::from(accumulator_output && shape.depthwise)),
            ))
            .bs_mul_shift_value_neg(Bits::new(bs_mul_shift))
            .build(),
    );
    commands.push(zero::<DpuOffsetPend>());
    commands.push(
        Register::<DpuDstBaseAddr>::new()
            .dst_base_addr(Bits::new(output_base_offset))
            .build(),
    );
    commands.push(
        Register::<DpuDstSurfStride>::new()
            .dst_surf_stride(Bits::new(output_surface_pixels))
            .build(),
    );
    commands.push(
        Register::<DpuDataCubeWidth>::new()
            .width(Bits::new(out_width - 1))
            .build(),
    );
    commands.push(
        Register::<DpuDataCubeHeight>::new()
            .height(Bits::new(rows.out_rows - 1))
            .build(),
    );
    commands.push(
        Register::<DpuDataCubeNotchAddr>::new()
            .notch_addr_0(Bits::new(output_notch))
            .notch_addr_1(Bits::new(output_notch))
            .build(),
    );
    commands.push(
        Register::<DpuDataCubeChannel>::new()
            .orig_channel(Bits::new(shape.out_channels - 1))
            .channel(Bits::new(padded_out_channels - 1))
            .build(),
    );
    commands.push(
        Register::<DpuBsCfg>::new()
            .bs_bypass(Bits::new(u32::from(accumulator_output)))
            .bs_alu_algo(Bits::new(2))
            .bs_alu_src(Bits::new(1))
            .bs_relu_bypass(Bits::new(1))
            .bs_mul_bypass(Bits::new(u32::from(quantization.is_none())))
            .build(),
    );
    commands.push(zero::<DpuBsAluCfg>());
    commands.push(
        Register::<DpuBsMulCfg>::new()
            .bs_mul_shift_value(Bits::new(bs_mul_shift))
            .bs_mul_src(Bits::new(u32::from(quantization.is_some())))
            .build(),
    );
    commands.push(zero::<DpuBsReluxCmpValue>());
    commands.push(
        Register::<DpuBsOwCfg>::new()
            // 3 for depthwise against 1 for dense, at every captured channel
            // count and in both precisions.
            .size_e_0(Bits::new(shape.bs_ow_size_e()))
            .size_e_1(Bits::new(shape.bs_ow_size_e()))
            .size_e_2(Bits::new(shape.bs_ow_size_e()))
            .od_bypass(Bits::new(u32::from(
                quantization.is_none() || accumulator_output,
            )))
            .ow_src(Bits::new(u32::from(quantization.is_some())))
            .build(),
    );
    commands.push(zero::<DpuBsOwOp>());
    commands.push(
        Register::<DpuWdmaSize0>::new()
            .channel_wdma(Bits::new(padded_out_channels - 1))
            .build(),
    );
    commands.push(
        Register::<DpuWdmaSize1>::new()
            .height_wdma(Bits::new(rows.out_rows - 1))
            .width_wdma(Bits::new(out_width - 1))
            .build(),
    );
    commands.push(
        Register::<DpuBnCfg>::new()
            .bn_relu_bypass(Bits::new(bn_relu_bypass))
            .bn_relux_en(Bits::new(bn_relux_en))
            // The ALU and MUL halves of the BN stage stay bypassed whatever
            // the activation; only the relu half is ever used.
            .bn_mul_bypass(Bits::new(1))
            .bn_alu_bypass(Bits::new(1))
            .bn_bypass(Bits::new(bn_bypass))
            .build(),
    );
    // Zero in every capture, activated or not: enabling the relu costs no
    // operand buffer and no DMA (`DPU_RDMA_RDMA_BN_BASE_ADDR` stays zero too).
    commands.push(zero::<DpuBnAluCfg>());
    commands.push(zero::<DpuBnMulCfg>());
    commands.push(
        Register::<DpuBnReluxCmpValue>::new()
            .bn_relux_cmp_dat(Bits::new(bn_relux_cmp))
            .build(),
    );
    commands.push(
        Register::<DpuEwCfg>::new()
            .ew_relu_bypass(Bits::new(1))
            .ew_op_cvt_bypass(Bits::new(1))
            .ew_lut_bypass(Bits::new(1))
            .ew_op_bypass(Bits::new(1))
            .ew_bypass(Bits::new(1))
            .build(),
    );
    commands.push(zero::<DpuEwCvtOffsetValue>());
    commands.push(
        Register::<DpuEwCvtScaleValue>::new()
            .ew_op_cvt_scale(Bits::new(1))
            .build(),
    );
    commands.push(zero::<DpuEwReluxCmpValue>());
    // Output conversion. fp16 lets `fp32tofp16_en` do the narrowing; int8
    // programs the multiplier as a normalized mantissa/shift pair and the
    // output zero point as an offset. Exact accumulator output bypasses BS
    // and CPEND and leaves this final converter at identity.
    let output_offset = if accumulator_output {
        0
    } else {
        quantization.map_or(0, |q| q.output_zero_point as u32)
    };
    let output_scale = if accumulator_output {
        1
    } else {
        quantization.map_or(1, |q| q.multiplier.scale)
    };
    let output_shift = if accumulator_output {
        0
    } else {
        quantization.map_or(0, |q| q.multiplier.shift)
    };
    commands.push(
        Register::<DpuOutCvtOffset>::new()
            .out_cvt_offset(Bits::new(output_offset))
            .build(),
    );
    commands.push(
        Register::<DpuOutCvtScale>::new()
            .fp32tofp16_en(Bits::new(u32::from(quantization.is_none())))
            .out_cvt_scale(Bits::new(output_scale))
            .build(),
    );
    commands.push(
        Register::<DpuOutCvtShift>::new()
            .out_cvt_shift(Bits::new(output_shift))
            .build(),
    );
    commands.push(zero::<DpuEwOpValue0>());
    commands.push(zero::<DpuEwOpValue1>());
    commands.push(zero::<DpuEwOpValue2>());
    commands.push(zero::<DpuEwOpValue3>());
    commands.push(zero::<DpuEwOpValue4>());
    commands.push(zero::<DpuEwOpValue5>());
    commands.push(zero::<DpuEwOpValue6>());
    commands.push(zero::<DpuEwOpValue7>());
    commands.push(
        // Accumulator output is hardware-validated with the fixed logical
        // value 16 together with DPU_DATA_FORMAT.mc_surf_out=1. CNA's
        // DATA_SIZE3.surf_mode deliberately remains zero: enabling it
        // changes the convolution results for NC1HWC2 input. Requantized
        // output uses half an output atom per pixel, and is otherwise not
        // precision-dependent:
        // this field is byte-identical across every fp16/int8 capture pair,
        // unlike `weight_bytes_per_kernel` right above, which halves.
        // Depthwise doubles this. Confirmed as a factor rather than a
        // constant by the stride-2 capture, whose 16x16 output takes 1024
        // against the dense 512.
        Register::<DpuSurfaceAdd>::new()
            .surf_add(Bits::new(if accumulator_output {
                let mult = accumulator_surf_mult_override(shape.in_channels)
                    .unwrap_or(DENSE_ACCUMULATOR_SURF_MULT);
                if shape.depthwise {
                    // Depthwise accumulator output stays on the serial writer,
                    // which serializes its 32-lane blocks with the fixed
                    // hardware value 16. `ROCKET_ACC_SURF_ADD` overrides it.
                    accumulator_surf_add_override(shape.in_channels).unwrap_or(16)
                } else {
                    // `rocket-userspace`'s rule: `dst_surf_stride * 8`, where
                    // `dst_surf_stride` is the *task's* `dataout_height *
                    // dataout_width`, not the whole image's. Per-task is
                    // load-bearing and is why a constant `ROCKET_ACC_SURF_ADD`
                    // could never express this on a tiled plan -- every tile
                    // has its own `out_rows`. Legal here because
                    // `programs_with_staged_accumulator_output` gives each tile
                    // its own contiguous scratch range, so a tile really is a
                    // standalone image; the non-accumulator branch below uses
                    // whole-image dims precisely because its tiles share one.
                    accumulator_surf_add_override(shape.in_channels)
                        .unwrap_or(out_width * rows.out_rows * mult)
                }
            } else if let Some(surf_add) = int4_override(shape.precision, "SURF_ADD") {
                surf_add
            } else if shape.precision == Precision::Int4 {
                // The integer write path's surface multiplier is 8, not the
                // float path's 2 -- the other half of the `size_e` quirk.
                // At Cout 128 the 2x value stops after 10 of 16 surfaces
                // and the 8x one writes all 16384 bytes.
                full_out_width * out_height * DENSE_ACCUMULATOR_SURF_MULT
            } else {
                full_out_width * out_height * 2 * if shape.depthwise { 2 } else { 1 }
            }))
            .build(),
    );
    commands.push(zero::<DpuReserved40c4>());
    commands.push(zero::<DpuLutAccessCfg>());
    commands.push(zero::<DpuLutAccessData>());
    commands.push(zero::<DpuLutCfg>());
    commands.push(zero::<DpuLutInfo>());
    commands.push(zero::<DpuLutLeStart>());
    commands.push(zero::<DpuLutLeEnd>());
    commands.push(zero::<DpuLutLoStart>());
    commands.push(zero::<DpuLutLoEnd>());
    commands.push(zero::<DpuLutLeSlopeScale>());
    commands.push(zero::<DpuLutLeSlopeShift>());
    commands.push(zero::<DpuLutLoSlopeScale>());
    commands.push(zero::<DpuLutLoSlopeShift>());

    // DPU_RDMA. The main feature path is disabled because CNA/CORE feed
    // DPU directly; BRDMA supplies the bias data.
    commands.push(
        Register::<DpuRdmaDataCubeWidth>::new()
            .width(Bits::new(out_width - 1))
            .build(),
    );
    commands.push(
        Register::<DpuRdmaDataCubeHeight>::new()
            .height(Bits::new(rows.out_rows - 1))
            .build(),
    );
    commands.push(
        Register::<DpuRdmaDataCubeChannel>::new()
            .channel(Bits::new(padded_out_channels - 1))
            .build(),
    );
    commands.push(zero::<DpuRdmaSrcBaseAddr>());
    commands.push(
        Register::<DpuRdmaBrdmaCfg>::new()
            .brdma_data_use(Bits::new(brdma_data_use))
            .build(),
    );
    commands.push(zero::<DpuRdmaBsBaseAddr>());
    commands.push(zero::<DpuRdmaNrdmaCfg>());
    commands.push(zero::<DpuRdmaBnBaseAddr>());
    commands.push(
        Register::<DpuRdmaErdmaCfg>::new()
            .erdma_disable(Bits::new(1))
            .build(),
    );
    commands.push(zero::<DpuRdmaEwBaseAddr>());
    commands.push(zero::<DpuRdmaEwSurfStride>());
    commands.push(
        Register::<DpuRdmaFeatureModeCfg>::new()
            .burst_len(BurstLength::Sixteen.into())
            .mrdma_disable(Bits::new(1))
            .in_precision(precision.into())
            .proc_precision(precision.into())
            .conv_mode(Bits::new(shape.conv_mode()))
            .build(),
    );
    commands.push(zero::<DpuRdmaSrcDmaCfg>());
    commands.push(zero::<DpuRdmaSurfNotch>());
    commands.push(zero::<DpuRdmaPadCfg>());
    commands.push(
        Register::<DpuRdmaWeight>::new()
            .e_weight(Bits::new(1))
            .n_weight(Bits::new(1))
            .b_weight(Bits::new(1))
            .m_weight(Bits::new(1))
            .build(),
    );
    commands.push(zero::<DpuRdmaEwSurfNotch>());

    // Vendor PC trailer: placeholder, zero register count, required marker,
    // combined operation-enable mask, and six words of alignment padding.
    commands.push(PCTrailer::single_task_placeholder());
    commands.push(zero::<PCRegisterAmounts>());
    commands.push(PCTrailer::required_marker());
    commands.push(PCTrailer::operation_enable(PCOperationMask::CONVOLUTION));
    commands.extend((0..6).map(|_| PCTrailer::alignment_padding()));

    debug_assert_eq!(commands.len(), 136);
    commands
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode(command: &RegCmd) -> (u32, u32, u32) {
        (
            (command.0 >> 48) as u32,
            command.0 as u32 & 0xffff,
            (command.0 >> 16) as u32,
        )
    }

    fn fnv1a(commands: &[RegCmd]) -> u64 {
        commands
            .iter()
            .flat_map(|command| command.0.to_le_bytes())
            .fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
                (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
            })
    }

    /// Registers that carry a precision field, and nothing else that a
    /// datatype change is allowed to move.
    fn precision_register_identities() -> Vec<(u32, u32)> {
        vec![
            (
                <CnaConvCon1 as RegisterMeta>::DOMAIN,
                <CnaConvCon1 as RegisterMeta>::OFFSET,
            ),
            (
                <CoreMiscCfg as RegisterMeta>::DOMAIN,
                <CoreMiscCfg as RegisterMeta>::OFFSET,
            ),
            (
                <DpuDataFormat as RegisterMeta>::DOMAIN,
                <DpuDataFormat as RegisterMeta>::OFFSET,
            ),
            (
                <DpuRdmaFeatureModeCfg as RegisterMeta>::DOMAIN,
                <DpuRdmaFeatureModeCfg as RegisterMeta>::OFFSET,
            ),
        ]
    }

    /// The 2-byte rungs are the fp16 program with the precision field
    /// changed, and nothing else.
    ///
    /// This is the same diff `rockchip-npu-notes` records for the matmul
    /// path -- `gen_matmul_bf16` == `gen_matmul_fp16` with only the
    /// precision words moved -- asserted here for the convolution program,
    /// which has four such registers rather than three (the DPU_RDMA stage
    /// carries a pair as well). It is what makes "bf16 is fp16 with a
    /// different field value" a checked claim rather than a hope: any future
    /// geometry rule that keys off `Precision::Fp16` by name instead of by
    /// element width will fail here.
    #[test]
    fn two_byte_precisions_differ_from_fp16_only_in_the_precision_registers() {
        let kernels: Kernels = [3, 3];
        for (precision, field) in [
            (Precision::Bf16, u32::from(DataPrecision::Bf16 as u32)),
            (Precision::Int16, u32::from(DataPrecision::Int16 as u32)),
        ] {
            let fp16 = Shape::with_precision(32, 32, 1, 64, 32, Precision::Fp16);
            let other = Shape::with_precision(32, 32, 1, 64, 32, precision);
            assert_eq!(
                fp16.weight_bytes(kernels),
                other.weight_bytes(kernels),
                "{precision:?} coefficient footprint must match fp16"
            );
            assert_eq!(
                fp16.output_scratch_bytes(kernels),
                other.output_scratch_bytes(kernels),
                "{precision:?} output allocation must match fp16"
            );

            let tile = Tile::whole(fp16, kernels);
            let baseline = conv_2d_tile(fp16, kernels, &tile);
            let candidate = conv_2d_tile(other, kernels, &tile);
            assert_eq!(baseline.len(), candidate.len());

            let expected = precision_register_identities();
            let mut moved = Vec::new();
            for (before, after) in baseline.iter().zip(&candidate) {
                let (domain, offset, before_value) = decode(before);
                let (after_domain, after_offset, after_value) = decode(after);
                assert_eq!((domain, offset), (after_domain, after_offset));
                if before_value != after_value {
                    moved.push((domain, offset, before_value, after_value));
                }
            }
            // `CNA_CONV_CON1` is written twice by the program, so compare
            // the distinct identities rather than the raw sequence.
            let mut identities: Vec<(u32, u32)> = moved
                .iter()
                .map(|(domain, offset, _, _)| (*domain, *offset))
                .collect();
            identities.dedup();
            assert_eq!(
                identities, expected,
                "{precision:?} moved unexpected registers: {moved:x?}"
            );
            // Every moved word must differ only where a 3-bit precision
            // field sits: xor the two and the result has to be a set of
            // 3-bit-aligned nibbles, never a stray geometry bit.
            for (domain, offset, before_value, after_value) in moved {
                let fp16_field = DataPrecision::Fp16 as u32;
                let mut rebuilt = before_value;
                for shift in 0..30 {
                    if (before_value >> shift) & 0x7 == fp16_field
                        && (after_value >> shift) & 0x7 == field
                    {
                        rebuilt = (rebuilt & !(0x7 << shift)) | (field << shift);
                    }
                }
                assert_eq!(
                    rebuilt, after_value,
                    "{precision:?} changed {domain:#x}:{offset:#x} outside its \
                     precision fields: {before_value:#010x} -> {after_value:#010x}"
                );
            }
        }
    }

    /// int4's geometry falls out of the shared atom widths.
    ///
    /// Every number here is `atom bytes * 8 / element bits` rather than a
    /// table entry, which is the claim worth pinning: the 32-channel feature
    /// atom, the 64-kernel coefficient atom and the halved coefficient
    /// footprint are all consequences of the half-byte element, and they
    /// match what `../rockchip-npu-notes/encodings/tile-layouts.md` records
    /// for int4.
    #[test]
    fn int4_geometry_follows_the_half_byte_element() {
        let kernels: Kernels = [3, 3];
        let int4 = Shape::with_precision(16, 16, 1, 64, 64, Precision::Int4);
        assert_eq!(int4.precision.element_bits(), 4);
        assert_eq!(int4.precision.channels_per_atom(), 32);
        assert_eq!(int4.precision.out_channel_granule(), 64);
        assert_eq!(int4.feature_atoms(), 2);
        assert_eq!(int4.padded_channels(), 64);
        // int4 accumulates to int16, so the result is four times as wide as
        // an operand.
        assert_eq!(int4.precision.output_element_bytes(), 2);

        // Half of int8's coefficient footprint at the same shape, and a
        // quarter of fp16's -- the whole point of the rung.
        let int8 = Shape::with_precision(
            16,
            16,
            1,
            64,
            64,
            Precision::Int8(Quantization {
                input_zero_point: 0,
                output_zero_point: 0,
                weight_zero_point: 0,
                input_scale: 1.0,
                weights_scale: 1.0,
                multiplier: Multiplier::from_ratio(1.0),
            }),
        );
        let fp16 = Shape::with_precision(16, 16, 1, 64, 64, Precision::Fp16);
        assert_eq!(int4.weight_bytes(kernels) * 2, int8.weight_bytes(kernels));
        assert_eq!(int4.weight_bytes(kernels) * 4, fp16.weight_bytes(kernels));

        // The precision field reaches the program as 6, and the DPU writes
        // an int16 result (field 1) rather than an int4 one.
        let tile = Tile::whole(int4, kernels);
        let program = conv_2d_tile(int4, kernels, &tile);
        let data_format = program
            .iter()
            .map(decode)
            .find(|(domain, offset, _)| {
                (*domain, *offset)
                    == (
                        <DpuDataFormat as RegisterMeta>::DOMAIN,
                        <DpuDataFormat as RegisterMeta>::OFFSET,
                    )
            })
            .expect("program must configure DPU_DATA_FORMAT")
            .2;
        // `DPU_DATA_FORMAT` is out[31:29], in[28:26], proc[2:0].
        assert_eq!((data_format >> 29) & 0x7, OutputPrecision::Int16 as u32);
        assert_eq!((data_format >> 26) & 0x7, DataPrecision::Int4 as u32);
        assert_eq!(data_format & 0x7, DataPrecision::Int4 as u32);
    }

    #[test]
    #[should_panic(expected = "whole 32-channel feature atom")]
    fn int4_refuses_a_partial_feature_atom() {
        let _ = Shape::with_precision(16, 16, 1, 48, 64, Precision::Int4);
    }

    #[test]
    fn vendor_reference_program_has_expected_layout() {
        let commands = conv_2d([1, 1]);
        assert_eq!(commands.len(), 136);

        assert_eq!(decode(&commands[0]), (0x0201, 0x1040, 0x0000_00b1));
        assert_eq!(decode(&commands[4]), (0x1001, 0x4004, 0x0000_000e));
        assert_eq!(decode(&commands[5]), (0x2001, 0x5004, 0x0000_000e));
        assert_eq!(decode(&commands[54]), (0x0801, 0x3010, 0x0000_0200));
        assert_eq!(decode(&commands[59]), (0x1001, 0x400c, 0x0000_01e4));
        assert_eq!(decode(&commands[109]), (0x2001, 0x500c, 0x0000_001f));

        assert_eq!(commands[126].0, 0);
        assert_eq!(commands[127].0, 0x0101_0000_0000_0014);
        assert_eq!(commands[128].0, 0x0041_0000_0000_0000);
        assert_eq!(commands[129].0, 0x0081_0000_001d_0008);
        assert!(commands[130..].iter().all(|command| command.0 == 0));
    }

    #[test]
    fn programs_match_captured_vendor_group_one_bit_for_bit() {
        assert_eq!(fnv1a(&conv_2d([1, 1])), 0x2577_26d7_f13a_1636);
        assert_eq!(fnv1a(&conv_2d([3, 3])), 0x8da7_c9ed_d561_7ccf);
    }

    #[test]
    fn kernel_geometry_changes_exactly_the_five_vendor_words() {
        let one_by_one = conv_2d([1, 1]);
        let three_by_three = conv_2d([3, 3]);
        let changed: Vec<_> = one_by_one
            .iter()
            .zip(&three_by_three)
            .enumerate()
            .filter_map(|(index, (left, right))| (left.0 != right.0).then_some(index))
            .collect();

        assert_eq!(changed, [7, 13, 14, 15, 25]);
        assert_eq!(decode(&one_by_one[7]).2, 0x0000_0210);
        assert_eq!(decode(&three_by_three[7]).2, 0x0000_0240);
        assert_eq!(decode(&one_by_one[13]).2, 0x0000_0080);
        assert_eq!(decode(&three_by_three[13]).2, 0x0000_0480);
        assert_eq!(decode(&one_by_one[14]).2, 0x0000_0010);
        assert_eq!(decode(&three_by_three[14]).2, 0x0000_0090);
        assert_eq!(decode(&one_by_one[15]).2, 0x0101_0008);
        assert_eq!(decode(&three_by_three[15]).2, 0x0303_0008);
        assert_eq!(decode(&one_by_one[25]).2, 0);
        assert_eq!(decode(&three_by_three[25]).2, 0x0000_0011);
    }

    #[test]
    fn even_kernels_program_verbatim_with_independent_padding() {
        let shape = Shape::CAPTURED.with_padding([0, 1]);
        let kernels = [4, 6];
        let plan = ConvPlan::new(shape, kernels);

        assert_eq!(
            (shape.output_width(kernels), shape.output_height(kernels)),
            (29, 29)
        );
        assert_eq!((plan.data_banks(), plan.weight_banks()), (1, 11));
        assert_eq!(plan.tiles(), &[Tile2D::whole(shape, kernels)]);

        let program = &plan.programs()[0];
        assert_eq!(value_of::<CnaWeightSize2>(program), 0x0604_0008);
        assert_eq!(value_of::<CnaPadCon0>(program), 0x10);
        assert_eq!(value_of::<CnaConvCon2>(program), 0x240);
        assert_eq!(value_of::<CnaWeightSize0>(program), 0xc00);
        assert_eq!(value_of::<CnaWeightSize1>(program), 0x180);

        let int8_shape =
            Shape::with_precision(32, 32, 1, 3, 8, captured_int8()).with_padding([0, 1]);
        let int8_programs = ConvPlan::new(int8_shape, kernels).programs();
        let int8_program = &int8_programs[0];
        assert_eq!(value_of::<CnaWeightSize2>(int8_program), 0x0604_0008);
        assert_eq!(value_of::<CnaPadCon0>(int8_program), 0x10);
        assert_eq!(value_of::<CnaConvCon2>(int8_program), 0x240);
    }

    #[test]
    fn even_kernel_default_padding_is_half_the_extent() {
        let shape = Shape::CAPTURED;
        for extent in [2usize, 4, 6, 8, 10] {
            let kernels = [extent, extent];
            assert_eq!(
                (shape.output_width(kernels), shape.output_height(kernels)),
                (33, 33),
                "k{extent}"
            );
            let plan = ConvPlan::new(shape, kernels);
            assert!(!plan.tiles().is_empty(), "k{extent}");
            assert!(plan.programs().iter().all(|program| program.len() == 136));
        }
    }

    #[test]
    fn even_kernel_tiles_keep_the_full_tap_halo() {
        let shape = Shape::new(8, 8).with_padding([0, 0]);
        let tiles = Tile::split(shape, [4, 4], 2);

        assert_eq!(
            tiles,
            [
                Tile {
                    out_first: 0,
                    out_rows: 3,
                    in_first: 0,
                    in_rows: 6,
                    pad_top: 0,
                },
                Tile {
                    out_first: 3,
                    out_rows: 2,
                    in_first: 3,
                    in_rows: 5,
                    pad_top: 0,
                },
            ]
        );
    }

    #[test]
    fn even_non_square_plans_take_the_captured_split_at_every_measured_demand() {
        // The even sweep's pressure row: 256x32, Cin 32, Cout 64 fp16, where
        // coefficient demand is `ceil(kh * kw / 8)`. Through six banks every
        // mirrored even pair agrees with its twin and with the demand-based
        // allocator.
        for (kernels, banks) in [
            ([4usize, 8usize], (8u32, 4u32)),
            ([8, 4], (8, 4)),
            ([4, 10], (7, 5)),
            ([10, 4], (7, 5)),
            ([6, 8], (6, 6)),
            ([8, 6], (6, 6)),
        ] {
            let shape = Shape::with_out_channels(256, 32, 1, 32, 64);
            let plan = ConvPlan::new(shape, kernels);
            assert_eq!(
                (plan.data_banks(), plan.weight_banks()),
                banks,
                "{kernels:?} CBUF split"
            );
        }
    }

    #[test]
    fn even_square_plans_follow_demand_to_the_measured_ceiling() {
        // 10x10 was refused outright before the fill-in row, on the strength
        // of a single Cout 64 capture at demand 13. Walking Cout down shows
        // the extent was never the problem: at demands 5 and 7 a 10x10 takes
        // exactly the demand-based split the captures show.
        for (cout, banks) in [(24u32, (7u32, 5u32)), (32, (5, 7))] {
            let plan = ConvPlan::new(Shape::with_out_channels(256, 32, 1, 32, cout), [10, 10]);
            assert_eq!(
                (plan.data_banks(), plan.weight_banks()),
                banks,
                "10x10 Cout {cout} CBUF split"
            );
        }
        // 8x8 at Cout 64 is the last demand the ladder confirms, at eight.
        let plan = ConvPlan::new(Shape::with_out_channels(256, 32, 1, 32, 64), [8, 8]);
        assert_eq!((plan.data_banks(), plan.weight_banks()), (4, 8));
    }

    #[test]
    #[should_panic(expected = "even square kernel [8, 8] needs 9")]
    fn automatic_plan_refuses_the_even_square_ladder_above_eight_banks() {
        // Cout 72 puts 8x8 at demand 9, the first rung the ladder shows
        // leaving the demand rule: the capture is 8/4 where demand asks 3/9.
        let _ = ConvPlan::new(Shape::with_out_channels(256, 32, 1, 32, 72), [8, 8]);
    }

    #[test]
    #[should_panic(expected = "even kernel [6, 10] needs 8")]
    fn automatic_plan_refuses_the_even_rectangle_that_disagrees_with_its_mirror() {
        // 6x10 and 10x6 both ask for eight banks and are captured 4/8 and
        // 8/4. Demand cannot choose between them, so neither is planned --
        // even though 6x10's own capture happens to match the demand rule.
        let _ = ConvPlan::new(Shape::with_out_channels(256, 32, 1, 32, 64), [6, 10]);
    }

    #[test]
    #[should_panic(expected = "even kernel [6, 8] needs 6")]
    fn automatic_plan_refuses_int8_even_rectangles_fp16_still_plans() {
        // The one place the two precisions need different bounds. At Cin 32,
        // Cout 128 int8 a 6x8 is captured 8/4 where demand asks 6/6, while
        // the matching fp16 shape at demand 6 takes 6/6 exactly.
        let shape = Shape::with_precision(256, 32, 1, 32, 128, captured_int8());
        let _ = ConvPlan::new(shape, [6, 8]);
    }

    #[test]
    #[should_panic(expected = "even kernels currently have capture backing only at stride 1")]
    fn automatic_plan_refuses_even_kernels_above_stride_one() {
        // Every point in the even grid is stride 1. Nothing says the pad or
        // grains formulas survive a stride an even kernel has never been
        // captured at.
        let _ = ConvPlan::new(Shape::with_stride(32, 32, 2), [4, 4]);
    }

    #[test]
    #[should_panic(expected = "where the captured split follows coefficient demand")]
    fn automatic_plan_refuses_unsettled_even_kernel_cbuf_pressure() {
        let shape = Shape::with_out_channels(256, 32, 1, 32, 64);
        let _ = ConvPlan::new(shape, [10, 10]);
    }

    #[test]
    #[should_panic(expected = "automatic CBUF allocation")]
    fn default_builder_rejects_large_kernel_without_explicit_banks() {
        let _ = conv_2d([5, 5]);
    }

    #[test]
    fn explicit_cbuf_builder_accepts_focused_large_kernels() {
        let shape = Shape::with_out_channels(256, 32, 1, 32, 64);
        for (kernel, data_banks, weight_banks, max_rows, min_tiles) in [
            (7usize, 8u32, 4u32, 16u32, 4u32),
            (9, 6, 6, 12, 8),
            (11, 7, 5, 14, 8),
        ] {
            let kernels = [kernel, kernel];
            assert_eq!(
                shape.max_tile_input_rows_for_data_banks(data_banks),
                max_rows
            );
            assert_eq!(
                shape.min_tiles_for_data_banks(kernels, data_banks),
                min_tiles
            );
            for tile in Tile::split(shape, kernels, min_tiles) {
                assert_eq!(
                    conv_2d_tile_with_cbuf_banks(shape, kernels, &tile, data_banks, weight_banks,)
                        .len(),
                    136
                );
            }
        }
    }

    #[test]
    fn conv_plan_preserves_the_captured_single_program() {
        let plan = ConvPlan::new(Shape::CAPTURED, [3, 3]);
        assert_eq!((plan.data_banks(), plan.weight_banks()), (1, 11));
        assert_eq!(plan.output_column_widths(), &[32]);
        assert_eq!(plan.tiles(), &[Tile2D::whole(Shape::CAPTURED, [3, 3])]);
        let programs = plan.programs();
        assert_eq!(programs.len(), 1);
        assert_eq!(fnv1a(&programs[0]), fnv1a(&conv_2d([3, 3])));
    }

    #[test]
    fn conv_plan_reconciles_the_expanded_vendor_fixture_routes() {
        // Granting the whole-map data demand avoids an otherwise unnecessary
        // split and still leaves four coefficient banks, above the measured
        // floor of three.
        let single = Shape::with_out_channels(32, 32, 1, 128, 64);
        let plan = ConvPlan::new(single, [3, 3]);
        assert_eq!((plan.data_banks(), plan.weight_banks()), (8, 4));
        assert_eq!(plan.tiles().len(), 1);

        // Bit 14 is not int8-specific: expanded fp16 Cin-1 captures use it
        // too, and keep these shapes in one task.
        let fp16_wide_entries = Shape::with_out_channels(128, 128, 1, 1, 1);
        let plan = ConvPlan::new(fp16_wide_entries, [1, 1]);
        assert_eq!(plan.tiles().len(), 1);
        assert_eq!(value_of::<CnaCbufCon1>(&plan.programs()[0]), 16_384);

        // Standalone vendor plans fill each tile to capacity rather than
        // balancing the output height over the minimum tile count.
        let tall = Shape::with_out_channels(64, 128, 1, 128, 8);
        let plan = ConvPlan::new(tall, [3, 3]);
        assert_eq!((plan.data_banks(), plan.weight_banks()), (11, 1));
        assert_eq!(
            plan.tiles()
                .iter()
                .map(|tile| tile.rows.out_rows)
                .collect::<Vec<_>>(),
            vec![21, 20, 20, 20, 20, 20, 7]
        );

        // Int8 dense CBUF rows charge width 226 as 240, but the host tensor
        // itself has a compact 226*3-byte row pitch. RKNN's padded-pitch
        // 92+91+43 route starts compact tiles at unsafe offsets, so retain
        // three tiles while moving both interior feature bases to 16-byte
        // boundaries.
        let int8 = Shape::with_precision(226, 226, 1, 3, 64, Precision::Int8(quantization()));
        let plan = ConvPlan::new(int8, [3, 3]);
        assert_eq!((plan.data_banks(), plan.weight_banks()), (11, 1));
        assert_eq!(
            plan.tiles()
                .iter()
                .map(|tile| (tile.rows.out_rows, tile.rows.in_rows))
                .collect::<Vec<_>>(),
            vec![(73, 74), (80, 82), (73, 74)]
        );
        assert_eq!(
            plan.programs()
                .iter()
                .map(|program| value_of::<CnaCbufCon1>(program))
                .collect::<Vec<_>>(),
            vec![17_760, 19_680, 17_760]
        );

        // CBUF/data_entries permits 1023 rows here, but feature_grains adds
        // one for a 1x1 kernel. Keep each task at 1022 rows so the 10-bit
        // field never receives the unencodable value 1024.
        let feature_grain_limited = Shape::with_out_channels(32, 1200, 1, 1, 1);
        let plan = ConvPlan::new(feature_grain_limited, [1, 1]);
        assert_eq!((plan.data_banks(), plan.weight_banks()), (10, 2));
        assert_eq!(
            plan.tiles()
                .iter()
                .map(|tile| tile.rows.in_rows)
                .collect::<Vec<_>>(),
            vec![1022, 178]
        );
        assert!(
            plan.tiles()
                .iter()
                .all(|tile| feature_grains([1, 1], &tile.rows) <= MAX_FEATURE_GRAINS)
        );
        assert_eq!(plan.programs().len(), 2);

        // At the exact surface-capacity boundary, standalone RKNN plans do
        // not let bottom-edge clipping enlarge the final grain. Both
        // precisions have 24 input atoms here and therefore take the same
        // 5/7 split and 14+13+1 route.
        for shape in [
            Shape::with_out_channels(28, 28, 1, 192, 64).with_padding([1, 1]),
            Shape::with_precision(28, 28, 1, 384, 64, captured_int8()).with_padding([1, 1]),
        ] {
            let plan = ConvPlan::new(shape, [3, 3]);
            assert_eq!((plan.data_banks(), plan.weight_banks()), (5, 7));
            assert_eq!(
                plan.tiles()
                    .iter()
                    .map(|tile| tile.rows.out_rows)
                    .collect::<Vec<_>>(),
                vec![14, 13, 1]
            );
        }

        // The same policy remains visible at a one-row continuation grain:
        // first produce two rows, then 26 single rows through the bottom.
        let tiny_grain = Shape::with_out_channels(28, 28, 1, 512, 64).with_padding([1, 1]);
        let plan = ConvPlan::new(tiny_grain, [3, 3]);
        assert_eq!((plan.data_banks(), plan.weight_banks()), (3, 9));
        let mut expected = vec![2];
        expected.extend(vec![1; 26]);
        assert_eq!(
            plan.tiles()
                .iter()
                .map(|tile| tile.rows.out_rows)
                .collect::<Vec<_>>(),
            expected
        );
    }

    #[test]
    fn conv_plan_selects_the_focused_large_kernel_policies() {
        let shape = Shape::with_out_channels(256, 32, 1, 32, 64);
        for (kernel, banks, tiles) in [
            (5usize, (8u32, 4u32), 3usize),
            (7, (5, 7), 8),
            (9, (6, 6), 7),
            (11, (7, 5), 7),
        ] {
            let plan = ConvPlan::new(shape, [kernel, kernel]);
            assert_eq!(
                (plan.data_banks(), plan.weight_banks()),
                banks,
                "k{kernel} banks"
            );
            assert_eq!(plan.output_column_widths(), &[256], "k{kernel} columns");
            assert_eq!(plan.tiles().len(), tiles, "k{kernel} tiles");
            assert!(plan.programs().iter().all(|program| program.len() == 136));
        }
    }

    #[test]
    fn conv_plan_reproduces_the_three_hardware_proven_rectangular_grids() {
        for (kernel, in_channels, banks, columns, tiles) in [
            (9usize, 64u32, (6u32, 6u32), &[135u32, 121][..], 19usize),
            (11, 48, (5, 7), &[137, 119][..], 27),
            (11, 64, (3, 9), &[59, 54, 54, 54, 35][..], 68),
        ] {
            let shape = Shape::with_out_channels(256, 32, 1, in_channels, 64);
            let plan = ConvPlan::new(shape, [kernel, kernel]);
            assert_eq!(
                (plan.data_banks(), plan.weight_banks()),
                banks,
                "k{kernel} Cin {in_channels} banks"
            );
            assert_eq!(
                plan.output_column_widths(),
                columns,
                "k{kernel} Cin {in_channels} columns"
            );
            assert_eq!(
                plan.tiles().len(),
                tiles,
                "k{kernel} Cin {in_channels} tiles"
            );
        }
    }

    /// Captured plan-0 words from the rectangular-kernel sweep, one row per
    /// capture in `rknn-files/sweep-kshape`: `conv-w32-h32-k3x7-s1`, its
    /// mirror `conv-w32-h32-k7x3-s1`, and `conv-w32-h32-k1x11-s1`.
    ///
    /// The mirrored pair is the whole point. `CnaWeightSize2` and
    /// `CnaPadCon0` swap their halves with the kernel, and `CnaConvCon2`
    /// moves with the kernel's height alone -- 36 grains at 3x7 against 42 at
    /// 7x3 -- while the coefficient footprint, which depends on the area,
    /// stays put across the swap.
    const CAPTURED_NON_SQUARE: [(Kernels, u32, u32, u32, u32, u32); 3] = [
        //  kernel     WeightSize2  PadCon0  ConvCon2  WeightSize0  WeightSize1
        ([3, 7], 0x0703_0008, 0x31, 0x240, 0xa80, 0x150),
        ([7, 3], 0x0307_0008, 0x13, 0x2a0, 0xa80, 0x150),
        ([1, 11], 0x0b01_0008, 0x50, 0x210, 0x580, 0x0b0),
    ];

    #[test]
    fn non_square_kernels_program_each_extent_on_its_own_axis() {
        for (kernels, weight_size2, pad_con0, conv_con2, weight_bytes, per_kernel) in
            CAPTURED_NON_SQUARE
        {
            let shape = Shape::CAPTURED;
            let plan = ConvPlan::new(shape, kernels);
            assert_eq!(
                (plan.data_banks(), plan.weight_banks()),
                (1, 11),
                "{kernels:?} CBUF split"
            );
            assert_eq!(plan.tiles().len(), 1, "{kernels:?} tiles");

            let program = &plan.programs()[0];
            assert_eq!(
                value_of::<CnaWeightSize2>(program),
                weight_size2,
                "{kernels:?} weight_width/height"
            );
            assert_eq!(
                value_of::<CnaPadCon0>(program),
                pad_con0,
                "{kernels:?} pad_left/top"
            );
            assert_eq!(
                value_of::<CnaConvCon2>(program),
                conv_con2,
                "{kernels:?} feature_grains"
            );
            assert_eq!(
                value_of::<CnaWeightSize0>(program),
                weight_bytes,
                "{kernels:?} weight_bytes"
            );
            assert_eq!(
                value_of::<CnaWeightSize1>(program),
                per_kernel,
                "{kernels:?} weight_bytes_per_kernel"
            );
            // SAME padding on both axes, so a non-square kernel leaves the
            // output extent alone.
            assert_eq!(
                (shape.output_width(kernels), shape.output_height(kernels)),
                (32, 32),
                "{kernels:?} output extent"
            );
        }
    }

    #[test]
    fn non_square_plans_take_the_captured_split_where_demand_still_decides() {
        // 256x32, Cin 32, Cout 64: the five captures in this shape whose
        // coefficient demand is four or five banks, each matching its
        // capture's plan-0 CBUF split.
        for (kernels, banks) in [
            ([3usize, 9usize], (8u32, 4u32)),
            ([9, 3], (8, 4)),
            ([3, 11], (7, 5)),
            ([11, 3], (7, 5)),
            ([5, 7], (7, 5)),
        ] {
            let shape = Shape::with_out_channels(256, 32, 1, 32, 64);
            let plan = ConvPlan::new(shape, kernels);
            assert_eq!(
                (plan.data_banks(), plan.weight_banks()),
                banks,
                "{kernels:?} CBUF split"
            );
        }
    }

    #[test]
    fn int8_non_square_kernels_program_the_same_geometry_words_as_fp16() {
        // `conv-w32-h32-k3x7-s1-i8` and `conv-w32-h32-k7x3-s1-i8` in
        // rknn-files/sweep-kshape-i8. Every geometry word matches the fp16
        // capture of the same shape exactly -- including the coefficient
        // footprint, where an int8 atom's doubled channel padding and halved
        // element size cancel at these channel counts.
        for (kernels, weight_size2, pad_con0, conv_con2, weight_bytes, per_kernel) in
            CAPTURED_NON_SQUARE
        {
            if kernels == [1, 11] {
                continue; // no int8 capture at this shape
            }
            let shape = Shape::with_precision(32, 32, 1, 3, 8, captured_int8());
            let plan = ConvPlan::new(shape, kernels);
            let program = &plan.programs()[0];
            assert_eq!(
                value_of::<CnaWeightSize2>(program),
                weight_size2,
                "{kernels:?}"
            );
            assert_eq!(value_of::<CnaPadCon0>(program), pad_con0, "{kernels:?}");
            assert_eq!(value_of::<CnaConvCon2>(program), conv_con2, "{kernels:?}");
            assert_eq!(
                value_of::<CnaWeightSize0>(program),
                weight_bytes,
                "{kernels:?}"
            );
            assert_eq!(
                value_of::<CnaWeightSize1>(program),
                per_kernel,
                "{kernels:?}"
            );
        }
    }

    #[test]
    fn int8_non_square_plans_take_the_captured_split() {
        // 256x32, Cin 32, Cout 64 int8: an int8 coefficient is one byte, so
        // these ask for half the fp16 demand and every one stays
        // demand-based -- including 9x7, whose fp16 twin does not.
        for (kernels, banks) in [
            ([3usize, 9usize], (8u32, 4u32)),
            ([9, 3], (8, 4)),
            ([5, 7], (8, 4)),
            ([7, 5], (8, 4)),
        ] {
            let shape = Shape::with_precision(256, 32, 1, 32, 64, captured_int8());
            let plan = ConvPlan::new(shape, kernels);
            assert_eq!(
                (plan.data_banks(), plan.weight_banks()),
                banks,
                "{kernels:?} CBUF split"
            );
        }
    }

    #[test]
    #[should_panic(expected = "stops following coefficient demand")]
    fn conv_plan_refuses_the_non_square_splits_the_captures_do_not_settle() {
        // 11x5 and its mirror 5x11 have the same seven-bank coefficient
        // demand and split differently -- 8/4 against 5/7 -- so demand alone
        // cannot choose and the planner declines to guess.
        let _ = ConvPlan::new(Shape::with_out_channels(256, 32, 1, 32, 64), [11, 5]);
    }

    #[test]
    fn explicit_banks_plan_a_non_square_kernel_the_allocator_refuses() {
        let shape = Shape::with_out_channels(256, 32, 1, 32, 64);
        let plan = ConvPlan::with_cbuf_banks(shape, [11, 5], 8, 4);
        assert_eq!((plan.data_banks(), plan.weight_banks()), (8, 4));
        assert!(!plan.tiles().is_empty());
        assert!(plan.programs().iter().all(|program| program.len() == 136));
    }

    /// The six captured groups, in the order they appear in the regcmd blob:
    /// one 1-core plan, then a 2-core plan, then a 3-core plan.
    const CAPTURED_PLANS: [(u32, u32); 6] = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)];

    fn captured_tile(kernels: Kernels, group: usize) -> Tile {
        let (tiles, index) = CAPTURED_PLANS[group];
        Tile::split(Shape::CAPTURED, kernels, tiles)[index as usize]
    }

    /// First write to `R`. A few registers are written twice per program --
    /// CnaConvCon1 and CnaCbufCon0 among them -- always with the same value.
    fn first_value_of<R: RegisterMeta>(commands: &[RegCmd]) -> u32 {
        commands
            .iter()
            .filter(|command| {
                (command.0 >> 48) as u32 == R::DOMAIN && (command.0 as u32 & 0xffff) == R::OFFSET
            })
            .map(|command| (command.0 >> 16) as u32)
            .next()
            .expect("register is never written")
    }

    fn value_of<R: RegisterMeta>(commands: &[RegCmd]) -> u32 {
        let matches: Vec<u32> = commands
            .iter()
            .filter(|command| {
                (command.0 >> 48) as u32 == R::DOMAIN && (command.0 as u32 & 0xffff) == R::OFFSET
            })
            .map(|command| (command.0 >> 16) as u32)
            .collect();
        assert_eq!(
            matches.len(),
            1,
            "expected exactly one write to the register"
        );
        matches[0]
    }

    #[test]
    fn tile_split_reproduces_vendor_row_ranges() {
        // REG_DPU_DATA_CUBE_HEIGHT decodes to 32, 16/16, and 11/11/10 rows
        // across the three captured plans.
        let rows: Vec<Vec<u32>> = [1, 2, 3]
            .iter()
            .map(|&n| {
                Tile::split(Shape::CAPTURED, [3, 3], n)
                    .iter()
                    .map(|t| t.out_rows)
                    .collect()
            })
            .collect();
        assert_eq!(rows, vec![vec![32], vec![16, 16], vec![11, 11, 10]]);

        // Every plan covers the image exactly once, with no gap or overlap.
        for tiles in 1..=IMAGE_HEIGHT {
            let split = Tile::split(Shape::CAPTURED, [3, 3], tiles);
            assert_eq!(split.iter().map(|t| t.out_rows).sum::<u32>(), IMAGE_HEIGHT);
            for pair in split.windows(2) {
                assert_eq!(pair[0].out_first + pair[0].out_rows, pair[1].out_first);
            }
        }
    }

    #[test]
    fn tile_halo_matches_captured_feature_offsets() {
        // DESIGN_NOTES: the 3x3 programs begin each continuation tile one
        // input row earlier than the 1x1 programs do.
        let three = Tile::split(Shape::CAPTURED, [3, 3], 3);
        let one = Tile::split(Shape::CAPTURED, [1, 1], 3);
        assert_eq!(
            three
                .iter()
                .map(|t| t.input_offset(Shape::CAPTURED))
                .collect::<Vec<_>>(),
            [0x0000, 0x0780, 0x0fc0]
        );
        assert_eq!(
            one.iter()
                .map(|t| t.input_offset(Shape::CAPTURED))
                .collect::<Vec<_>>(),
            [0x0000, 0x0840, 0x1080]
        );
        assert_eq!(
            three
                .iter()
                .map(|t| t.output_offset(Shape::CAPTURED, [3, 3]))
                .collect::<Vec<_>>(),
            [0x0000, 0x1600, 0x2c00]
        );

        // Rows actually read, including halo and excluding padded rows.
        assert_eq!(
            three.iter().map(|t| t.in_rows).collect::<Vec<_>>(),
            [12, 13, 11]
        );
        assert_eq!(
            three.iter().map(|t| t.pad_top).collect::<Vec<_>>(),
            [1, 0, 0]
        );
    }

    #[test]
    fn short_large_kernel_tiles_retain_remaining_top_padding() {
        // Hardware exposed this at 7x7/Cin64: two output rows fit per tile,
        // so the second tile starts at output row 2 while one of the three
        // top-padding rows is still in force. Treating every continuation
        // tile as unpadded shifted that tile's convolution down by one row.
        let shape = Shape::with_out_channels(256, 32, 1, 64, 64);
        let split = Tile::split(shape, [7, 7], 16);
        assert_eq!(
            split
                .iter()
                .take(3)
                .map(|tile| tile.out_first)
                .collect::<Vec<_>>(),
            [0, 2, 4]
        );
        assert_eq!(
            split
                .iter()
                .take(3)
                .map(|tile| tile.pad_top)
                .collect::<Vec<_>>(),
            [3, 1, 0]
        );

        // The one-row 11x11/Cin64 plan keeps decreasing the padding until
        // the sixth output row finally has a complete real-input footprint.
        let split = Tile::split(shape, [11, 11], 32);
        assert_eq!(
            split
                .iter()
                .take(7)
                .map(|tile| tile.pad_top)
                .collect::<Vec<_>>(),
            [5, 4, 3, 2, 1, 0, 0]
        );
    }

    #[test]
    fn column_tiles_reproduce_large_kernel_capture_boundaries() {
        let k9 = Shape::with_out_channels(256, 32, 1, 64, 64);
        assert_eq!(
            ColumnTile::split(k9, [9, 9], &[135, 121]),
            [
                ColumnTile {
                    out_first: 0,
                    out_cols: 135,
                    in_first: 0,
                    in_cols: 139,
                    pad_left: 4,
                },
                ColumnTile {
                    out_first: 135,
                    out_cols: 121,
                    in_first: 131,
                    in_cols: 125,
                    pad_left: 0,
                },
            ]
        );

        let k11_c48 = Shape::with_out_channels(256, 32, 1, 48, 64);
        assert_eq!(
            ColumnTile::split(k11_c48, [11, 11], &[137, 119]),
            [
                ColumnTile {
                    out_first: 0,
                    out_cols: 137,
                    in_first: 0,
                    in_cols: 142,
                    pad_left: 5,
                },
                ColumnTile {
                    out_first: 137,
                    out_cols: 119,
                    in_first: 132,
                    in_cols: 124,
                    pad_left: 0,
                },
            ]
        );

        let k11_c64 = Shape::with_out_channels(256, 32, 1, 64, 64);
        let columns = ColumnTile::split(k11_c64, [11, 11], &[59, 54, 54, 54, 35]);
        assert_eq!(
            columns
                .iter()
                .map(|tile| (tile.out_first, tile.in_first, tile.in_cols, tile.pad_left))
                .collect::<Vec<_>>(),
            [
                (0, 0, 64, 5),
                (59, 54, 64, 0),
                (113, 108, 64, 0),
                (167, 162, 64, 0),
                (221, 216, 40, 0),
            ]
        );
    }

    #[test]
    fn horizontal_programming_matches_captured_11x11_tiles() {
        let shape = Shape::with_out_channels(256, 32, 1, 64, 64);
        let columns = ColumnTile::split(shape, [11, 11], &[59, 54, 54, 54, 35]);
        let rows = Tile {
            out_first: 0,
            out_rows: 7,
            in_first: 0,
            in_rows: 12,
            pad_top: 5,
        };

        let left = conv_2d_tile_2d_with_cbuf_banks(
            shape,
            [11, 11],
            &Tile2D {
                rows,
                columns: columns[0],
            },
            3,
            9,
        );
        assert_eq!(first_value_of::<CnaConvCon1>(&left), 0x2000_0120);
        assert_eq!(first_value_of::<CnaCbufCon0>(&left), 0x93);
        assert_eq!(value_of::<CnaDataSize0>(&left), 0x0040_000c);
        assert_eq!(value_of::<CnaDataSize2>(&left), 59);
        assert_eq!(value_of::<CnaDataSize3>(&left), 59 * 7);
        assert_eq!(value_of::<CnaCbufCon1>(&left), 128);
        assert_eq!(value_of::<CnaPadCon0>(&left), 0x55);
        assert_eq!(value_of::<CnaDmaCon1>(&left), 256);
        assert_eq!(value_of::<CnaDmaCon2>(&left), 8192 - 64);
        assert_eq!(value_of::<CnaFcDataSize0>(&left), 0x0040_000c);
        assert_eq!(value_of::<CoreDataoutSize0>(&left), 0x0006_003a);
        assert_eq!(value_of::<DpuDataCubeNotchAddr>(&left), 0x00c5_00c5);
        assert_eq!(value_of::<DpuDstSurfStride>(&left), 256 * 32 * 16);
        assert_eq!(value_of::<DpuSurfaceAdd>(&left), 256 * 32 * 32);

        let middle = conv_2d_tile_2d_with_cbuf_banks(
            shape,
            [11, 11],
            &Tile2D {
                rows,
                columns: columns[1],
            },
            3,
            9,
        );
        assert_eq!(value_of::<CnaDataSize2>(&middle), 54);
        assert_eq!(value_of::<CnaPadCon0>(&middle), 0x05);
        assert_eq!(value_of::<CnaFeatureDataAddr>(&middle), 54 * 16);
        assert_eq!(value_of::<DpuDstBaseAddr>(&middle), 59 * 16);
        assert_eq!(value_of::<DpuDataCubeNotchAddr>(&middle), 0x00ca_00ca);
    }

    #[test]
    fn tile_registers_match_all_six_captured_groups() {
        // Observed values, group 1 through group 6, from the bitbiter reports
        // in the design-spike repo (conv-group-N.md).
        const FEATURE_GRAINS: [[u32; 6]; 2] = [
            [36, 21, 20, 16, 16, 14], // 3x3
            [33, 17, 17, 12, 12, 11], // 1x1
        ];
        const DATA_SIZE0: [[u32; 6]; 2] = [
            [
                0x20_0020, 0x20_0011, 0x20_0011, 0x20_000c, 0x20_000d, 0x20_000b,
            ],
            [
                0x20_0020, 0x20_0010, 0x20_0010, 0x20_000b, 0x20_000b, 0x20_000a,
            ],
        ];
        const CBUF_CON1: [[u32; 6]; 2] = [
            [0x400, 0x220, 0x220, 0x180, 0x1a0, 0x160],
            [0x400, 0x200, 0x200, 0x160, 0x160, 0x140],
        ];
        const PAD_CON0: [[u32; 6]; 2] = [
            [0x11, 0x11, 0x10, 0x11, 0x10, 0x10],
            [0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
        ];
        const FEATURE_ADDR: [[u32; 6]; 2] = [
            [0, 0, 0xb40, 0, 0x780, 0xfc0],
            [0, 0, 0xc00, 0, 0x840, 0x1080],
        ];
        // Output geometry is identical for both kernels: same output tile.
        const DATA_SIZE3: [u32; 6] = [0x400, 0x200, 0x200, 0x160, 0x160, 0x140];
        const DST_BASE: [u32; 6] = [0, 0, 0x2000, 0, 0x1600, 0x2c00];
        const CUBE_HEIGHT: [u32; 6] = [0x1f, 0x0f, 0x0f, 0x0a, 0x0a, 0x09];
        const DATAOUT_SIZE0: [u32; 6] = [
            0x1f_001f, 0x0f_001f, 0x0f_001f, 0x0a_001f, 0x0a_001f, 0x09_001f,
        ];

        for (row, kernels) in [(0usize, [3usize, 3]), (1, [1, 1])] {
            for group in 0..6 {
                let tile = captured_tile(kernels, group);
                let program = conv_2d_tile(Shape::CAPTURED, kernels, &tile);
                let at = |what: &str, got: u32, want: u32| {
                    assert_eq!(
                        got,
                        want,
                        "{what} mismatch, {kernels:?} group {}",
                        group + 1
                    );
                };

                // FEATURE_GRAINS is a field, not a whole register: the vendor's
                // multi-core programs also carry a plan index in bits 31:28,
                // which a standalone tile program does not reproduce.
                let grains = (value_of::<CnaConvCon2>(&program) & 0x0000_3ff0) >> 4;
                at("feature_grains", grains, FEATURE_GRAINS[row][group]);

                at(
                    "data_size0",
                    value_of::<CnaDataSize0>(&program),
                    DATA_SIZE0[row][group],
                );
                at(
                    "fc_data_size0",
                    value_of::<CnaFcDataSize0>(&program),
                    DATA_SIZE0[row][group],
                );
                at(
                    "cbuf_con1",
                    value_of::<CnaCbufCon1>(&program),
                    CBUF_CON1[row][group],
                );
                at(
                    "pad_con0",
                    value_of::<CnaPadCon0>(&program),
                    PAD_CON0[row][group],
                );
                at(
                    "feature_data_addr",
                    value_of::<CnaFeatureDataAddr>(&program),
                    FEATURE_ADDR[row][group],
                );
                at(
                    "data_size3",
                    value_of::<CnaDataSize3>(&program),
                    DATA_SIZE3[group],
                );
                at(
                    "dst_base_addr",
                    value_of::<DpuDstBaseAddr>(&program),
                    DST_BASE[group],
                );
                at(
                    "data_cube_height",
                    value_of::<DpuDataCubeHeight>(&program),
                    CUBE_HEIGHT[group],
                );
                at(
                    "rdma_data_cube_height",
                    value_of::<DpuRdmaDataCubeHeight>(&program),
                    CUBE_HEIGHT[group],
                );
                at(
                    "core_dataout_size0",
                    value_of::<CoreDataoutSize0>(&program),
                    DATAOUT_SIZE0[group],
                );
                at(
                    "wdma_size1",
                    value_of::<DpuWdmaSize1>(&program),
                    DATAOUT_SIZE0[group],
                );
            }
        }
    }

    #[test]
    fn cbuf_bank_split_matches_every_swept_shape() {
        // (width, height) -> data_bank, read out of CNA_CBUF_CON0 across the
        // 134 C3/Cout8/stride-1 programs in the shape-sweep corpus. Every one
        // also satisfies data_bank + weight_bank == 12.
        const OBSERVED: [(u32, u32, u32); 11] = [
            (32, 32, 1),
            (32, 64, 2),
            (32, 128, 4),
            (32, 256, 8),
            (64, 32, 2),
            (64, 64, 4),
            (96, 32, 3),
            (128, 32, 4),
            (128, 128, 11),
            (192, 32, 6),
            (256, 32, 8),
        ];
        for (width, height, data_bank) in OBSERVED {
            let shape = Shape::new(width, height);
            assert_eq!(
                shape.data_banks([3, 3]),
                data_bank,
                "data_bank for {width}x{height}"
            );
            assert_eq!(
                shape.data_banks([3, 3]) + shape.weight_banks([3, 3]),
                12,
                "bank split for {width}x{height} must cover all 12 CBUF banks"
            );
        }
    }

    #[test]
    fn wider_shapes_scale_the_geometry_registers() {
        // Formulas validated against 212 C3 stride-1 programs from 35 captures.
        // 256 wide caps a tile at 44 input rows by the vendor's own CBUF
        // rule (11 data banks x 1024 pixels / 256), so 64 rows need two.
        let shape = Shape::new(256, 64);
        assert_eq!(shape.max_tile_input_rows([3, 3]), 44);
        assert_eq!(shape.min_tiles([3, 3]), 2);

        let split = Tile::split(shape, [3, 3], 2);
        let tile = split[0];
        let program = conv_2d_tile(shape, [3, 3], &tile);

        assert_eq!(
            value_of::<CnaDataSize0>(&program),
            (256 << 16) | tile.in_rows
        );
        assert_eq!(value_of::<CnaCbufCon1>(&program), 256 * tile.in_rows);
        assert_eq!(value_of::<CnaDataSize3>(&program), 256 * tile.out_rows);
        assert_eq!(
            value_of::<CoreDataoutSize0>(&program),
            ((tile.out_rows - 1) << 16) | 255
        );
        assert_eq!(value_of::<DpuDataCubeHeight>(&program), tile.out_rows - 1);

        // Row strides follow the dense NHWC input and C8 fp16 output.
        assert_eq!(shape.input_row_stride(), 256 * 3 * 2);
        assert_eq!(shape.output_row_stride([3, 3]), 256 * 8 * 2);

        // A three-way split of 64 rows, with the 3x3 halo on continuations.
        let three = Tile::split(shape, [3, 3], 3);
        assert_eq!(
            three.iter().map(|t| t.out_rows).collect::<Vec<_>>(),
            [22, 21, 21]
        );
        assert_eq!(three[1].in_first, 21);
        assert_eq!(three[1].output_offset(shape, [3, 3]), 22 * 256 * 8 * 2);
    }

    #[test]
    fn vendor_capacity_rule_matches_the_width_sweep() {
        // Largest tile the vendor emits at each width, from the width sweep:
        // data_banks * 1024 / width, exact at every measured point.
        for (width, rows) in [(256u32, 32u32), (512, 22), (768, 14), (1024, 11), (1536, 7)] {
            let shape = Shape::new(width, 32);
            assert_eq!(
                shape.max_tile_input_rows([3, 3]),
                rows,
                "max tile rows at {width} wide"
            );
        }
    }

    #[test]
    fn stride_scales_output_geometry_and_halo() {
        // Formulas confirmed on 150 stride-2, -3 and -4 programs.
        let shape = Shape::with_stride(128, 64, 2);
        assert_eq!(shape.output_width([3, 3]), 64);
        assert_eq!(shape.output_height([3, 3]), 32);
        assert_eq!(shape.output_row_stride([3, 3]), 64 * 8 * 2);

        // A two-way split of the 32 output rows. The continuation tile
        // projects back through the stride: 16 * 2 - 1 = 31.
        let split = Tile::split(shape, [3, 3], 2);
        assert_eq!(
            split.iter().map(|t| t.out_rows).collect::<Vec<_>>(),
            [16, 16]
        );
        assert_eq!(split[0].in_first, 0);
        assert_eq!(split[0].in_rows, 32);
        assert_eq!(split[0].pad_top, 1);
        assert_eq!(split[1].in_first, 31);
        assert_eq!(split[1].in_rows, 33);
        assert_eq!(split[1].pad_top, 0);

        // The stride reaches CNA_CONV_CON3, and output-side registers carry
        // output geometry rather than input.
        let program = conv_2d_tile(shape, [3, 3], &split[0]);
        assert_eq!(value_of::<CnaConvCon3>(&program), (2 << 3) | 2);
        assert_eq!(value_of::<CnaDataSize2>(&program), 64);
        // Raw word: the DST_SURF_STRIDE field is shifted 4, so the encoded
        // word is sixteen times the 64 * 32 field value.
        assert_eq!(value_of::<DpuDstSurfStride>(&program), 64 * 32 * 16);

        // Stride 1 is unchanged.
        let flat = Shape::new(128, 64);
        assert_eq!(flat.output_height([3, 3]), 64);
        assert_eq!(Tile::split(flat, [3, 3], 2)[1].in_first, 31);
    }

    #[test]
    fn channel_layout_boundary_is_half_an_atom() {
        // Cin 4 is the last dense case (8 bytes); Cin 5 is already surfaces.
        // Measured directly: line_stride/width is 1.00 up to Cin 4 and 4.00
        // from Cin 5 onward.
        for cin in 1..=4 {
            assert_eq!(
                Shape::with_channels(32, 32, 1, cin).layout(),
                FeatureLayout::Dense,
                "Cin {cin}"
            );
        }
        for cin in [5, 6, 7, 8, 16, 80] {
            assert_eq!(
                Shape::with_channels(32, 32, 1, cin).layout(),
                FeatureLayout::Surfaces,
                "Cin {cin}"
            );
        }
    }

    #[test]
    fn channel_padding_matches_the_fill_in_sweep() {
        // (Cin, datain_channel, weight_channels) read out of the captures.
        // The two exceptions are 3 atoms, where the fields disagree, and
        // 7 atoms, where both round up.
        const OBSERVED: [(u32, u32, u32); 16] = [
            (3, 8, 8),
            (4, 8, 8),
            (5, 8, 8),
            (8, 8, 8),
            (9, 16, 16),
            (12, 16, 16),
            (16, 16, 16),
            (20, 24, 32),
            (24, 24, 32),
            (28, 32, 32),
            (32, 32, 32),
            (36, 40, 40),
            (40, 40, 40),
            (48, 48, 48),
            // Seven atoms pads the coefficients to 64 but leaves
            // datain_channel at 56, the same split three atoms makes. This
            // row read (56, 64, 64) until a field-by-field comparison
            // against the whole corpus showed the capture programs 56.
            (56, 56, 64),
            (64, 64, 64),
        ];
        for (cin, padded, weights) in OBSERVED {
            let shape = Shape::with_channels(32, 32, 1, cin);
            assert_eq!(shape.padded_channels(), padded, "datain_channel Cin {cin}");
            assert_eq!(
                shape.weight_channels(),
                weights,
                "weight channels Cin {cin}"
            );
        }
        // 72 and 80 are unpadded, confirming this is not a power-of-two rule.
        assert_eq!(Shape::with_channels(32, 32, 1, 72).weight_channels(), 72);
        assert_eq!(Shape::with_channels(32, 32, 1, 80).weight_channels(), 80);
    }

    #[test]
    fn fp16_channel_padding_follows_the_quad_atom_rule_to_512() {
        // (Cin, datain_channel, weight_channels) from the large-Cin sweep.
        // Read against the old two-entry table, 24 and 56 look like a
        // `2**n - 1` rule; 88, 120, 152 and the rest show it is not. Every
        // row where the two counts disagree has an atom count of 3 mod 4,
        // and the rows on either side of each are here to keep a rule
        // separable from a table.
        const OBSERVED: [(u32, u32, u32); 24] = [
            (81, 88, 96),
            (88, 88, 96),
            (89, 96, 96),
            (96, 96, 96),
            (104, 104, 104),
            (112, 112, 112),
            (113, 120, 128),
            (120, 120, 128),
            (121, 128, 128),
            (128, 128, 128),
            (136, 136, 136),
            (144, 144, 144),
            (152, 152, 160),
            (160, 160, 160),
            (184, 184, 192),
            (185, 192, 192),
            (216, 216, 224),
            (224, 224, 224),
            (248, 248, 256),
            (256, 256, 256),
            (280, 280, 288),
            (344, 344, 352),
            (440, 440, 448),
            (504, 504, 512),
        ];
        for (cin, padded, weights) in OBSERVED {
            let shape = Shape::with_channels(32, 32, 1, cin);
            assert_eq!(shape.padded_channels(), padded, "datain_channel Cin {cin}");
            assert_eq!(
                shape.weight_channels(),
                weights,
                "weight channels Cin {cin}"
            );
        }
    }

    #[test]
    fn int8_channel_padding_stays_exact_where_fp16_bumps() {
        // The same atom counts that bump at fp16 -- 11 and 15, here 176 and
        // 240 -- pass through unchanged at int8, out to 512. The coefficient
        // padding never leaves `padded_channels`.
        for cin in [
            129u32, 144, 175, 176, 177, 225, 240, 241, 304, 368, 432, 496, 512,
        ] {
            let shape = Shape::with_precision(32, 32, 1, cin, 8, Precision::Int8(quantization()));
            let whole_atoms = cin.div_ceil(16) * 16;
            assert_eq!(
                shape.padded_channels(),
                whole_atoms,
                "int8 datain Cin {cin}"
            );
            assert_eq!(
                shape.weight_channels(),
                whole_atoms,
                "int8 weights Cin {cin}"
            );
        }
    }

    #[test]
    fn cbuf_atom_charge_rounds_to_whole_groups_in_both_precisions() {
        // Recovered from `CNA_CBUF_CON1.data_entries`, which carries
        // `input_width * cbuf_atoms / 4` in the surface regime. The two-entry
        // version this replaced was right only below the old ceilings; every
        // value here past them is one it charged an atom short.
        for (cin, charged) in [
            (32u32, 4u32),
            (48, 6),
            (56, 8),
            (88, 12),
            (96, 12),
            (104, 13),
            (112, 14),
            (120, 16),
            (128, 16),
            (152, 20),
            (184, 24),
            (216, 28),
            (248, 32),
            (280, 36),
            (344, 44),
            (440, 56),
            (504, 64),
        ] {
            assert_eq!(
                Shape::with_channels(32, 32, 1, cin).cbuf_atoms(),
                charged,
                "fp16 CBUF atom charge at Cin {cin}"
            );
        }

        // int8 charges the same way even though its padding does not bump.
        for (cin, charged) in [
            (48u32, 4u32),
            (112, 8),
            (176, 12),
            (192, 12),
            (240, 16),
            (304, 20),
            (368, 24),
            (432, 28),
            (496, 32),
        ] {
            let shape = Shape::with_precision(32, 32, 1, cin, 8, Precision::Int8(quantization()));
            assert_eq!(
                shape.cbuf_atoms(),
                charged,
                "int8 CBUF atom charge at Cin {cin}"
            );
        }
    }

    #[test]
    fn argb_input_mode_follows_the_layout() {
        // CNA_CONV_CON1 across the channel sweep: dense programs the ARGB
        // image path with nonalign_dma and group_line_off set, surfaces
        // clear all three. Cin 3 -> 10/1/1, Cin 4 -> 11/1/1, Cin >= 5 ->
        // 0/0/0. Leaving these at the C3 values made every channel count
        // read as three channels.
        const ARGB_IN: u32 = 0x0000_f000;
        const NONALIGN_DMA: u32 = 0x4000_0000;
        const GROUP_LINE_OFF: u32 = 0x2000_0000;
        let field = |program: &[RegCmd], mask: u32| {
            (first_value_of::<CnaConvCon1>(program) & mask) >> (mask.trailing_zeros())
        };

        for (cin, argb) in [(3u32, 10u32), (4, 11)] {
            let shape = Shape::with_channels(32, 32, 1, cin);
            let program = conv_2d_tile(shape, [3, 3], &Tile::whole(shape, [3, 3]));
            assert_eq!(field(&program, ARGB_IN), argb, "argb_in at Cin {cin}");
            assert_eq!(field(&program, NONALIGN_DMA), 1, "nonalign at Cin {cin}");
            assert_eq!(
                field(&program, GROUP_LINE_OFF),
                1,
                "group_line at Cin {cin}"
            );
        }
        for cin in [5u32, 8, 16, 24, 64] {
            let shape = Shape::with_channels(32, 32, 1, cin);
            let program = conv_2d_tile(shape, [3, 3], &Tile::whole(shape, [3, 3]));
            assert_eq!(field(&program, ARGB_IN), 0, "argb_in at Cin {cin}");
            assert_eq!(field(&program, NONALIGN_DMA), 0, "nonalign at Cin {cin}");
            assert_eq!(
                field(&program, GROUP_LINE_OFF),
                0,
                "group_line at Cin {cin}"
            );
        }
    }

    #[test]
    fn channel_bank_split_matches_the_sweep() {
        // At 32x32 the surface rule reduces to ceil(weight_atoms / 2).
        for (cin, banks) in [
            (8u32, 1u32),
            (16, 1),
            (20, 2),
            (32, 2),
            (40, 3),
            (48, 3),
            (56, 4),
            (64, 4),
            (72, 5),
            (80, 5),
        ] {
            assert_eq!(
                Shape::with_channels(32, 32, 1, cin).data_banks([3, 3]),
                banks,
                "data_bank at Cin {cin}"
            );
        }
        // Dense stays on the pixel rule it was derived with.
        assert_eq!(Shape::with_channels(128, 32, 1, 3).data_banks([3, 3]), 4);
        assert_eq!(Shape::with_channels(128, 32, 1, 4).data_banks([3, 3]), 4);
        // ...and the surface rule diverges immediately at the boundary.
        assert_eq!(Shape::with_channels(128, 32, 1, 8).data_banks([3, 3]), 2);
    }

    #[test]
    fn multi_channel_registers_match_the_captures() {
        // conv-w128-h32-k3-s1-ci16-co8: line_stride 512, surf_stride 3584,
        // data_entries 64, datain_channel 16, data_bank 4.
        let shape = Shape::with_channels(128, 32, 1, 16);
        let program = conv_2d_tile(shape, [3, 3], &Tile::whole(shape, [3, 3]));
        assert_eq!(value_of::<CnaDmaCon1>(&program), 512);
        assert_eq!(value_of::<CnaDmaCon2>(&program), 3584);
        assert_eq!(value_of::<CnaCbufCon1>(&program), 64);
        assert_eq!(value_of::<CnaDataSize1>(&program) & 0xffff, 16);
        assert_eq!(shape.data_banks([3, 3]), 4);
        // Weight footprint follows the weight padding, not datain_channel.
        assert_eq!(value_of::<CnaWeightSize1>(&program), 9 * 16 * 2);
        assert_eq!(value_of::<CnaWeightSize0>(&program), 9 * 16 * 2 * 8);

        // Cin 24 is the case where the two paddings disagree: datain_channel
        // stays 24 while the coefficients occupy 32 channels.
        let odd = Shape::with_channels(32, 32, 1, 24);
        let odd_program = conv_2d_tile(odd, [3, 3], &Tile::whole(odd, [3, 3]));
        assert_eq!(value_of::<CnaDataSize1>(&odd_program) & 0xffff, 24);
        assert_eq!(value_of::<CnaWeightSize1>(&odd_program), 9 * 32 * 2);
        assert_eq!(value_of::<CnaCbufCon1>(&odd_program), 32);
    }

    #[test]
    fn surface_data_entries_rounds_up_at_widths_not_a_multiple_of_four() {
        // Every surface `data_entries` capture before this one used a width
        // divisible by 4, where floor and ceiling division agree. Vendor
        // captures at Cin=8 (one atom/pixel) expose the real rule: width
        // 13/29/30/31 program data_entries 4/8/8/8, not the 3/7/7/7 floor
        // division used to compute. The 30-wide, Cout=16 case is the shape
        // a real compiled model hit as scattered-pixel corruption on
        // hardware.
        for (width, data_entries) in [(13u32, 4u32), (29, 8), (30, 8), (31, 8)] {
            let shape = Shape::with_out_channels(width, 16, 1, 8, 16);
            let program = conv_2d_tile(shape, [1, 1], &Tile::whole(shape, [1, 1]));
            assert_eq!(
                value_of::<CnaCbufCon1>(&program),
                data_entries,
                "data_entries at width {width}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "input channels must be")]
    fn rejects_channels_beyond_the_validated_range() {
        // 96 was beyond it until the large-Cin sweep; 513 is the first value
        // past what either precision now measures.
        let _ = Shape::with_channels(32, 32, 1, MAX_INPUT_CHANNELS + 1);
    }

    #[test]
    fn output_channels_pad_to_whole_granules() {
        // Every value the corpus covers. Unlike the input padding this is a
        // rule rather than a table: no exceptions at 20, 24, 40, 56 or 72,
        // the values where the input padding needed them.
        for (out_channels, padded) in [
            (1u32, 16u32),
            (2, 16),
            (8, 16),
            (14, 16),
            (16, 16),
            (20, 32),
            (24, 32),
            (28, 32),
            (32, 32),
            (40, 48),
            (48, 48),
            (56, 64),
            (64, 64),
            (72, 80),
            (80, 80),
            (96, 96),
            (128, 128),
            (256, 256),
            (512, 512),
        ] {
            let shape = Shape::with_out_channels(32, 32, 1, 3, out_channels);
            assert_eq!(
                shape.padded_out_channels(),
                padded,
                "padded output channels at Cout {out_channels}"
            );
        }
    }

    #[test]
    fn output_channel_registers_match_the_captures() {
        // conv-w32-h32-k3-s1-ci3-co40, single-core plan: weight_bytes 5760,
        // weight_kernels 40, orig_channel 39, and the four padded-count
        // registers 47 -- Cout 40 rounds to 48. Cout 40 is chosen because
        // every one of those five numbers is distinct there.
        let shape = Shape::with_out_channels(32, 32, 1, 3, 40);
        let program = conv_2d_tile(shape, [3, 3], &Tile::whole(shape, [3, 3]));
        assert_eq!(value_of::<CnaWeightSize0>(&program), 5760);
        assert_eq!(value_of::<CnaWeightSize1>(&program), 144);
        assert_eq!(value_of::<CnaWeightSize2>(&program), 50_528_296);
        assert_eq!(value_of::<DpuDataCubeChannel>(&program), 2_555_951);
        assert_eq!(value_of::<DpuWdmaSize0>(&program), 47);
        assert_eq!(value_of::<CoreDataoutSize1>(&program), 47);

        // The output is NC1HWC2 regardless of Cout: the channel count sets
        // how many 8-channel surfaces there are, not how wide a row is, so
        // both output strides are the same as the Cout 8 capture's.
        assert_eq!(value_of::<DpuDstSurfStride>(&program), 32 * 32 * 16);

        // Cout 9 is the smallest value where the true and padded counts
        // differ by more than the granule rounding of a multiple of 8.
        let odd = Shape::with_out_channels(32, 32, 1, 3, 9);
        let odd_program = conv_2d_tile(odd, [3, 3], &Tile::whole(odd, [3, 3]));
        assert_eq!(value_of::<CnaWeightSize0>(&odd_program), 1296);
        assert_eq!(value_of::<DpuDataCubeChannel>(&odd_program), 524_303);
        assert_eq!(value_of::<DpuWdmaSize0>(&odd_program), 15);
    }

    #[test]
    fn output_channels_reach_the_bank_split_only_through_the_weight_footprint() {
        // The one place in the corpus where Cout moves the CBUF allocation.
        // At 256x32 with Cin 32 the feature data wants 16 banks and cannot
        // have them, so whatever the weights take comes straight off it.
        let narrow = Shape::with_out_channels(256, 32, 1, 32, 16);
        assert_eq!(narrow.weight_bytes([3, 3]), 9216); // one bank
        assert_eq!(narrow.data_banks([3, 3]), 11);
        assert_eq!(narrow.weight_banks([3, 3]), 1);

        let wide = Shape::with_out_channels(256, 32, 1, 32, 64);
        assert_eq!(wide.weight_bytes([3, 3]), 36864); // two banks
        assert_eq!(wide.data_banks([3, 3]), 10);
        assert_eq!(wide.weight_banks([3, 3]), 2);

        // Where the data is *not* over budget, a bigger kernel set changes
        // nothing: the weights are taking slack that was already theirs.
        for out_channels in [8u32, 16, 64, 128] {
            let shape = Shape::with_out_channels(32, 32, 1, 3, out_channels);
            assert_eq!(
                shape.data_banks([3, 3]),
                1,
                "32x32 Cin 3 keeps one data bank at Cout {out_channels}"
            );
        }

        // And when the weights are the larger claim they are the ones cut
        // short: 589824 bytes want 18 banks, more than the CBUF holds, but
        // the feature data still gets the 4 it asked for.
        let huge = Shape::with_out_channels(32, 32, 1, 64, 512);
        assert_eq!(huge.weight_bytes([3, 3]), 589_824);
        assert_eq!(huge.data_banks([3, 3]), 4);
        assert_eq!(huge.weight_banks([3, 3]), 8);
    }

    /// The output-parity rule and both accumulator refusals are gone, and
    /// this pins *why* rather than merely asserting the absence.
    ///
    /// The rule was: a dense accumulator tile is correct only when
    /// `tile_pixels * blocks_per_pixel` is even, because the DPU commits
    /// output in whole 256-byte units. That was a property of the **serial**
    /// writer (`mc_surf_out = 1`) the dense accumulator used to drive, whose
    /// blocks are 128 bytes -- so an odd block count left a trailing half-unit
    /// unwritten.
    ///
    /// The writer is now `mc_surf_out = 0` / `size_e = 7` /
    /// `surf_add = dataout * 8`, whose cube is 16-byte atoms of C2 = 4 int32
    /// lanes. `blocks_per_pixel` is then `padded_out_channels / 4`, and
    /// `padded_out_channels` is always a multiple of the 32-channel granule,
    /// so the block count is always a multiple of 8 -- **even by
    /// construction, at every shape**. The rule cannot fire, which is the
    /// arithmetic reason the padding is unreachable rather than merely
    /// untriggered by the cases tried.
    #[test]
    fn accumulator_block_count_is_even_by_construction_so_parity_cannot_bind() {
        for cout in [1u32, 8, 31, 32, 33, 64, 96, 136, 256, 353, 768] {
            let shape = Shape::with_precision(
                9,
                7,
                1,
                8,
                cout,
                Precision::Int8Accumulator(Quantization {
                    input_zero_point: 0,
                    output_zero_point: 0,
                    weight_zero_point: 0,
                    ..quantization()
                }),
            );
            let blocks = shape.output_blocks_per_pixel();
            assert_eq!(
                blocks,
                shape.padded_out_channels() / 4,
                "Cout={cout}: the accumulator cube is C2=4"
            );
            assert!(
                blocks.is_multiple_of(2),
                "Cout={cout}: blocks={blocks} must be even by construction"
            );
            // And the hook is now the identity at every one of them, including
            // the 3x3-output/3x3-kernel case that used to be refused outright.
            assert_eq!(
                shape.parity_padded_shape([1, 1]).unwrap().out_channels,
                cout
            );
        }

        let refused = Shape::with_precision(
            3,
            3,
            1,
            8,
            32,
            Precision::Int8Accumulator(Quantization {
                input_zero_point: 0,
                output_zero_point: 0,
                weight_zero_point: 0,
                ..quantization()
            }),
        );
        assert!(
            refused.parity_padded_shape([3, 3]).is_ok(),
            "the 3x3-output/3x3-kernel refusal is gone with the serial writer"
        );
    }

    /// The saturating coefficient working set is refused, not silently
    /// clamped into a one-data-bank split.
    ///
    /// A 5x5 kernel at `Cin` 512 is constructible today and wants 25 banks,
    /// far past the eleven grantable. Before the guard it came back 1/11 --
    /// about one input row per tile -- from the same clamp interaction the
    /// expanded corpus exposes at `Cin` >= 576. (7x7 does not reach the
    /// allocator: above coefficient demand seven it takes its own
    /// capture-derived 8/4 schedule.)
    #[test]
    #[should_panic(expected = "not capture-backed")]
    fn saturating_coefficient_working_set_is_refused() {
        let shape = Shape::with_precision(32, 32, 1, 512, 64, Precision::Fp16);
        let _ = ConvPlan::new(shape, [5, 5]);
    }

    /// A 5x5 kernel at `Cin` 192 plans 2/10, the split the vendor uses.
    ///
    /// This is the shape that catches a `Cin`-curve fit being extrapolated
    /// across kernel sizes: a two-pass rule derived from k=3 (which fits all
    /// 13 k=3 points from `Cin` 384 to 768) plans 6/6 here. Nothing else in
    /// the suite reaches it -- the base corpus has k=5 only at `Cin` 3, where
    /// the coefficient preference is 1.
    #[test]
    fn large_kernel_high_channel_split_matches_vendor() {
        let shape =
            Shape::with_precision(28, 28, 1, 192, 256, Precision::Fp16).with_padding([2, 2]);
        let plan = ConvPlan::new(shape, [5, 5]);
        assert_eq!((plan.data_banks(), plan.weight_banks()), (2, 10));
    }

    /// The guard does not disturb the range that is capture-backed: a 3x3
    /// kernel at the same `Cin` wants nine banks and still plans 3/9, the
    /// split the expanded corpus records.
    #[test]
    fn largest_capture_backed_working_set_still_plans() {
        let shape =
            Shape::with_precision(28, 28, 1, 512, 256, Precision::Fp16).with_padding([1, 1]);
        let plan = ConvPlan::new(shape, [3, 3]);
        assert_eq!((plan.data_banks(), plan.weight_banks()), (3, 9));
    }

    #[test]
    fn weight_banks_floor_matches_all_five_hardware_points() {
        // Cin=Cout=N, 3x3, 30x30, swept explicitly via ConvPlan::with_cbuf_banks
        // in iree-rocket-hal/tests/conv_cbuf_split_sweep_hw.rs's
        // weight_bank_floor_probe* family -- each is the exact hardware
        // boundary (floor-1 fails 0/5, floor passes 5/5). Not a flat
        // constant: it rises +1 per 64 Cin from 256, then plateaus at 5
        // from 384 on -- a first guess of `Cin/128 + 1` fit 256 and 512
        // but predicted 4, not 5, at 384.
        for (cin, floor) in [(256u32, 3u32), (320, 4), (384, 5), (448, 5), (512, 5)] {
            assert_eq!(
                weight_banks_floor(cin),
                floor,
                "Cin={cin} weight_banks floor"
            );
        }

        // The real VGG-19 shapes this was chasing: features.19 (Cin=256,
        // Cout=512) and features.21 (Cin=512, Cout=512), both hardware-proven
        // all-zero at the pre-fix automatic 11/1, and both hardware-confirmed
        // fixed post-fix -- features.19 at floor=3
        // (fixed_formula_resolves_features_19_and_21, 5/5) and features.21 at
        // floor=5, *not* 3 (weight_bank_floor_probe_at_cin_512 found
        // weight_banks=3 and 4 both still 0/5 at Cin=512; only 5 and up
        // pass). Cout does not move the floor -- only Cin does, matching the
        // asymmetry `DESIGN_NOTES.md`'s vendor-formula cross product found.
        assert_eq!(weight_banks_floor(256), 3);
        assert_eq!(weight_banks_floor(512), 5);

        // This measured floor is a safety lower bound, not the vendor's
        // preferred allocation. The expanded corpus independently shows a
        // larger streamed working-set grant at K3; keep the two rules
        // explicit so a policy change cannot masquerade as a new hardware
        // minimum.
        assert_eq!(streamed_weight_bank_preference(256, [3, 3]), 5);
        assert_eq!(streamed_weight_bank_preference(512, [3, 3]), 9);

        // features.0's shape: weight_banks=1 here is not a starved
        // footprint, it is the footprint's *entire* real demand (1,728
        // bytes fits in a fraction of one bank) -- hardware-confirmed
        // correct at 11/1 by the original five-shape sweep, and the floor
        // must leave it alone rather than take banks a fully-resident
        // footprint never needed. Below Cin=256 the floor itself is
        // unvalidated (see weight_banks_floor's doc comment); this only
        // confirms the starved branch doesn't fire here at all.
        let features_0 = Shape::with_out_channels(226, 226, 1, 3, 64).with_padding([0, 0]);
        assert_eq!(features_0.data_banks([3, 3]), 11);
        assert_eq!(features_0.weight_banks([3, 3]), 1);

        // Same for the 256x32/Cin32 pair above: weight demand of 1 and 2
        // banks respectively are each already fully satisfied by what they
        // are granted, not clamped down from something larger.
        let narrow = Shape::with_out_channels(256, 32, 1, 32, 16);
        assert_eq!(narrow.weight_banks([3, 3]), 1);
        let wide = Shape::with_out_channels(256, 32, 1, 32, 64);
        assert_eq!(wide.weight_banks([3, 3]), 2);
    }

    #[test]
    #[should_panic(expected = "output channels must be")]
    fn rejects_output_channels_beyond_the_validated_range() {
        let _ = Shape::with_out_channels(32, 32, 1, 3, MAX_OUTPUT_CHANNELS + 1);
    }

    #[test]
    fn dense_feature_offset_safe_requires_full_alignment() {
        // The older width/channel sweeps used uniform values and therefore
        // could only observe whole-pixel leading-column loss. The exact
        // features.0 regression uses non-uniform data and shows offset 4 is
        // already unsafe at Cin=3. Conservatively require offset 0 at every
        // fp16 dense width until a data-rich sweep proves otherwise.
        let cin1 = Shape::with_out_channels(225, 60, 1, 1, 8).with_padding([0, 0]); // stride mod16=2
        for in_first in 0..8 {
            assert_eq!(
                cin1.dense_feature_offset_safe(in_first),
                in_first == 0,
                "Cin=1 in_first={in_first}"
            );
        }
        let cin2 = Shape::with_out_channels(225, 60, 1, 2, 8).with_padding([0, 0]); // stride mod16=4
        for in_first in 0..8 {
            assert_eq!(
                cin2.dense_feature_offset_safe(in_first),
                matches!(in_first, 0 | 4),
                "Cin=2 in_first={in_first}"
            );
        }
        let cin3 = Shape::with_out_channels(227, 60, 1, 3, 256).with_padding([0, 0]); // stride mod16=2
        for in_first in 0..8 {
            assert_eq!(
                cin3.dense_feature_offset_safe(in_first),
                in_first == 0,
                "Cin=3 in_first={in_first}"
            );
        }
        let cin4 = Shape::with_out_channels(225, 60, 1, 4, 256).with_padding([0, 0]); // stride mod16=8
        for in_first in 0..8 {
            assert_eq!(
                cin4.dense_feature_offset_safe(in_first),
                in_first % 2 == 0,
                "Cin=4 in_first={in_first}"
            );
        }
        // Cout plays no role -- the address formula never references it.
        let small_cout = Shape::with_out_channels(227, 60, 1, 3, 8).with_padding([0, 0]);
        assert!(small_cout.dense_feature_offset_safe(0));
        assert!(!small_cout.dense_feature_offset_safe(2));

        // The affine-int8 oracle measured the same failure at offset 2 for
        // this compact VGG input pitch: 226 * 3 = 678 bytes, or 6 mod 16.
        let int8 = Shape::with_precision(226, 226, 1, 3, 64, captured_int8()).with_padding([0, 0]);
        assert_eq!(int8.input_row_stride(), 678);
        for in_first in 0..16 {
            assert_eq!(
                int8.dense_feature_offset_safe(in_first),
                in_first % 8 == 0,
                "int8 Cin=3 in_first={in_first}",
            );
        }

        // Surfaces (Cin > 4) are a different addressing path this defect
        // has not been shown to reach; always reported safe.
        let surfaces = Shape::with_out_channels(227, 60, 1, 5, 8);
        for in_first in 0..8 {
            assert!(surfaces.dense_feature_offset_safe(in_first));
        }
    }

    #[test]
    fn conv_plan_moves_run1_tile_boundary_off_the_hardware_confirmed_break() {
        // rocket_conv_harness.py's run1 (iree-rocket-design-spike): Cin=3
        // dense, Cout=256, 3x3, 228x228 physically-padded input. Before this
        // fix, ConvPlan::new's automatic split put tile 5's out_first/
        // in_first at 189 (odd -- an 8-byte-misaligned feature base at this
        // shape's stride), hardware-confirmed to corrupt one leading pixel
        // of every one of that tile's 37 output rows
        // (conv_dense_shared_buffer_dispatch_hw.rs).
        let kernels = [3, 3];
        let shape = Shape::with_out_channels(228, 228, 1, 3, 256).with_padding([0, 0]);
        let plan = ConvPlan::new(shape, kernels);
        assert_eq!(plan.data_banks(), 10);
        assert_eq!(plan.weight_banks(), 2);
        assert_eq!(plan.tiles().len(), 6);

        let mut covered = 0u32;
        for tile in plan.tiles() {
            assert_eq!(
                tile.rows.out_first, covered,
                "row coverage must stay contiguous with no gap or overlap"
            );
            covered += tile.rows.out_rows;
            assert!(
                shape.dense_feature_offset_safe(tile.rows.in_first),
                "tile at in_first={} is not alignment-safe",
                tile.rows.in_first
            );
        }
        assert_eq!(covered, shape.output_height(kernels));

        // Greedy capacity filling makes every full tile 42 output rows and
        // leaves a short 16-row tail. All boundaries are even and therefore
        // safe at this row pitch; in particular the old 189 boundary is no
        // longer present.
        assert_eq!(plan.tiles()[4].rows.out_rows, 42);
        assert_eq!(plan.tiles()[5].rows.out_first, 210);
        assert_eq!(plan.tiles()[5].rows.in_first, 210);
        assert_eq!(plan.tiles()[5].rows.out_rows, 16);
    }

    #[test]
    fn conv_plan_aligns_the_compact_int8_vgg_boundary() {
        let kernels = [3, 3];
        let shape = Shape::with_precision(226, 226, 1, 3, 64, captured_int8()).with_padding([0, 0]);
        let plan = ConvPlan::new(shape, kernels);

        assert_eq!((plan.data_banks(), plan.weight_banks()), (11, 1));
        assert_eq!(plan.tiles().len(), 3);
        assert_eq!(plan.tiles()[0].rows.in_first, 0);
        assert_ne!(plan.tiles()[1].rows.in_first, 91);
        let mut covered = 0;
        for tile in plan.tiles() {
            assert_eq!(tile.rows.out_first, covered);
            covered += tile.rows.out_rows;
            assert!(
                shape.dense_feature_offset_safe(tile.rows.in_first),
                "unsafe int8 tile at in_first={} offset={}",
                tile.rows.in_first,
                tile.rows.input_offset(shape) % FEATURE_ATOM_BYTES,
            );
        }
        assert_eq!(covered, shape.output_height(kernels));
    }

    #[test]
    fn dense_row_tiling_stays_gap_free_and_alignment_safe_across_a_shape_sweep() {
        // Host-side sweep, no hardware needed: dense_feature_offset_safe is
        // a pure function of the plan ConvPlan::new already produces, so
        // this checks the fix holds broadly rather than just at run1's one
        // shape. 4 Cin values x 4 Cout values x 7 widths x 4 heights = 448
        // shapes. Width 226 is the newly-found VGG-19 features.0 regression
        // point and must remain in the ordinary suite.
        let kernels = [3, 3];
        for cin in [1u32, 2, 3, 4] {
            for cout in [1u32, 8, 64, 256] {
                for width in [30u32, 61, 97, 225, 226, 227, 300] {
                    for height in [30u32, 61, 226, 300] {
                        let shape = Shape::with_out_channels(width, height, 1, cin, cout)
                            .with_padding([0, 0]);
                        let plan = ConvPlan::new(shape, kernels);
                        let mut covered = 0u32;
                        for tile in plan.tiles() {
                            assert_eq!(
                                tile.rows.out_first, covered,
                                "cin={cin} cout={cout} w={width} h={height}: gap/overlap"
                            );
                            covered += tile.rows.out_rows;
                            assert!(
                                shape.dense_feature_offset_safe(tile.rows.in_first),
                                "cin={cin} cout={cout} w={width} h={height}: unsafe tile at \
                                 in_first={}",
                                tile.rows.in_first
                            );
                        }
                        assert_eq!(
                            covered,
                            shape.output_height(kernels),
                            "cin={cin} cout={cout} w={width} h={height}: coverage mismatch"
                        );
                    }
                }
            }
        }
    }

    /// The quantization of `conv-w32-h32-k3-s1-i8`, read off the capture.
    fn captured_int8() -> Precision {
        Precision::Int8(Quantization {
            input_zero_point: 0,
            output_zero_point: -3,
            weight_zero_point: 0,
            input_scale: 1.0,
            weights_scale: 1.0,
            multiplier: Multiplier {
                scale: 19636,
                shift: 24,
            },
        })
    }

    #[test]
    fn int8_rounds_the_programmed_kernel_count_up_to_even() {
        // Vendor captures at 32x32, Cin 3, 3x3: the programmed kernel count
        // and the coefficient footprint, against the true Cout.
        //
        // `conv-w32-h32-k3-s1-ci3-co{1,2,3,4,5,12}-i8` and their fp16 twins.
        // The int8 corpus had nothing below Cout 8 until hardware failed at
        // Cout 1, which is why this went unnoticed: every int8 Cout ever
        // captured was already even.
        const BYTES_PER_KERNEL: u32 = 144; // 3 * 3 * pad(Cin 3) * 1 byte

        for (cout, kernels) in [(1u32, 2u32), (2, 2), (3, 4), (4, 4), (5, 6), (12, 12)] {
            let shape = Shape::with_precision(32, 32, 1, 3, cout, Precision::Int8(quantization()));
            assert_eq!(shape.programmed_kernels(), kernels, "int8 Cout {cout}");
            assert_eq!(
                shape.weight_bytes([3, 3]),
                kernels * BYTES_PER_KERNEL,
                "int8 weight_bytes at Cout {cout}"
            );
        }

        // fp16 programs the true count at every value, odd ones included.
        for cout in [1u32, 2, 6, 9, 14] {
            let shape = Shape::with_out_channels(32, 32, 1, 3, cout);
            assert_eq!(shape.programmed_kernels(), cout, "fp16 Cout {cout}");
            // 16 padded input channels at fp16, two bytes each.
            assert_eq!(shape.weight_bytes([3, 3]), cout * 9 * 8 * 2);
        }
    }

    #[test]
    fn int8_channel_padding_is_a_rule_not_a_table() {
        // An int8 atom carries 16 channels, so both paddings double their
        // granule. Measured at 15 Cin values and 10 Cout values with no
        // deviation -- including three atoms (33..48) and seven atoms (112),
        // where the fp16 table needs exceptions. Neither recurs.
        for (in_channels, padded) in [
            (3u32, 16u32),
            (4, 16),
            (16, 16),
            (17, 32),
            (24, 32),
            (32, 32),
            (33, 48),
            (40, 48),
            (48, 48),
            (64, 64),
            (80, 80),
            (112, 112),
            (128, 128),
        ] {
            let shape =
                Shape::with_precision(32, 32, 1, in_channels, 8, Precision::Int8(quantization()));
            assert_eq!(
                shape.padded_channels(),
                padded,
                "int8 datain_channel at Cin {in_channels}"
            );
            // Unlike fp16, the two padded counts never disagree.
            assert_eq!(shape.weight_channels(), padded);
        }

        for (out_channels, padded) in [
            (8u32, 32u32),
            (16, 32),
            (20, 32),
            (32, 32),
            (40, 64),
            (48, 64),
            (64, 64),
            (96, 96),
            (112, 128),
        ] {
            let shape =
                Shape::with_precision(32, 32, 1, 3, out_channels, Precision::Int8(quantization()));
            assert_eq!(
                shape.padded_out_channels(),
                padded,
                "int8 padded Cout at {out_channels}"
            );
        }
    }

    fn quantization() -> Quantization {
        match captured_int8() {
            Precision::Int8(quantization) => quantization,
            _ => unreachable!(),
        }
    }

    #[test]
    fn dense_layout_boundary_is_channels_not_bytes() {
        // The boundary is four channels in *both* precisions. Written as a
        // byte-width test -- "narrower than half a feature atom" -- it comes
        // out right at fp16 and wrong at int8, where it would allow eight.
        // The captures put Cin 4 on the ARGB path and Cin 8 on surfaces at
        // both precisions.
        for precision in [Precision::Fp16, Precision::Int8(quantization())] {
            for (in_channels, layout) in [
                (1u32, FeatureLayout::Dense),
                (4, FeatureLayout::Dense),
                (5, FeatureLayout::Surfaces),
                (8, FeatureLayout::Surfaces),
            ] {
                let shape = Shape::with_precision(32, 32, 1, in_channels, 8, precision);
                assert_eq!(
                    shape.layout(),
                    layout,
                    "layout at Cin {in_channels}, {precision:?}"
                );
            }
        }
    }

    #[test]
    fn int8_registers_match_the_captures() {
        // conv-w32-h32-k3-s1-i8, single-core plan. Every field this builder
        // writes matches the capture; the values below are the ones that
        // move with precision.
        let shape = Shape::with_precision(32, 32, 1, 3, 8, captured_int8());
        let program = conv_2d_tile(shape, [3, 3], &Tile::whole(shape, [3, 3]));

        // Coefficients are one byte per element, so the footprint halves
        // even though the padded channel count doubles.
        assert_eq!(value_of::<CnaWeightSize0>(&program), 16 * 9 * 8);
        assert_eq!(value_of::<CnaWeightSize1>(&program), 16 * 9);

        // Padding contributes the input zero point, not zero.
        assert_eq!(value_of::<CnaPadCon1>(&program), 0);
        let offset = Shape::with_precision(
            32,
            32,
            1,
            3,
            8,
            Precision::Int8(Quantization {
                input_zero_point: -1,
                ..quantization()
            }),
        );
        let offset_program = conv_2d_tile(offset, [3, 3], &Tile::whole(offset, [3, 3]));
        assert_eq!(value_of::<CnaPadCon1>(&offset_program), u32::MAX);

        // Requantization: the multiplier, its shift, and the output zero
        // point, none of which the fp16 path programs at all.
        assert_eq!(value_of::<DpuOutCvtScale>(&program), 19636);
        assert_eq!(value_of::<DpuOutCvtShift>(&program), 24);
        assert_eq!(value_of::<DpuOutCvtOffset>(&program), (-3i32) as u32);
    }

    #[test]
    fn int8_accumulator_output_uses_the_hardware_validated_bypasses() {
        let quantization = Quantization {
            input_zero_point: 0,
            output_zero_point: 0,
            weight_zero_point: 0,
            ..quantization()
        };
        let shape = Shape::with_precision(4, 4, 1, 1, 8, Precision::Int8Accumulator(quantization));
        let program = conv_2d_tile(shape, [1, 1], &Tile::whole(shape, [1, 1]));

        let data_format = value_of::<DpuDataFormat>(&program);
        assert_eq!(data_format & 0b111, 0, "DPU processing stays int8");
        assert_eq!((data_format >> 26) & 0b111, 0, "DPU input stays int8");
        assert_eq!((data_format >> 29) & 0b111, 4, "DPU output is int32");
        assert_eq!(
            (data_format >> 3) & 1,
            0,
            "dense accumulator output uses the one-surface writer, not the serial one"
        );
        assert_eq!(
            (value_of::<CnaDataSize3>(&program) >> 22) & 0b11,
            0,
            "CNA surface serial mode must remain disabled"
        );
        assert_eq!(value_of::<DpuSurfaceAdd>(&program), (4 * 4 * 8) << 4);
        assert_eq!(value_of::<DpuBsCfg>(&program) & 1, 1, "BS is bypassed");
        assert_eq!(
            (value_of::<DpuBsOwCfg>(&program) >> 1) & 1,
            1,
            "CPEND is bypassed"
        );
        assert_eq!(value_of::<DpuOutCvtOffset>(&program), 0);
        assert_eq!(value_of::<DpuOutCvtScale>(&program), 1);
        assert_eq!(value_of::<DpuOutCvtShift>(&program), 0);

        assert_eq!(shape.precision.element_bytes(), 1);
        assert_eq!(shape.precision.output_element_bytes(), 4);
        assert_eq!(shape.output_channel_block_bytes(), FEATURE_ATOM_BYTES);
        let requantized = Shape::with_precision(4, 4, 1, 1, 8, Precision::Int8(quantization));
        assert_eq!(
            shape.output_scratch_bytes([1, 1]),
            4 * requantized.output_scratch_bytes([1, 1])
        );
    }

    #[test]
    fn int8_accumulator_uses_the_c2_4_output_cube_without_forced_tiling() {
        let shape = Shape::with_precision(
            32,
            32,
            1,
            64,
            128,
            Precision::Int8Accumulator(Quantization {
                input_zero_point: 0,
                output_zero_point: 0,
                weight_zero_point: 0,
                ..quantization()
            }),
        );
        let plan = ConvPlan::new(shape, [1, 1]);

        assert_eq!(plan.output_column_widths(), &[32]);
        assert_eq!(plan.tiles().len(), 1);
        assert_eq!(
            shape.output_channel_block_bytes(),
            FEATURE_ATOM_BYTES,
            "the dense accumulator cube is 16-byte atoms of C2=4 int32 lanes"
        );
        assert_eq!(shape.output_row_stride([1, 1]), 32 * FEATURE_ATOM_BYTES);
    }

    #[test]
    fn staged_accumulator_plan_partitions_scratch_and_programs_local_surfaces() {
        let shape = Shape::with_precision(
            32,
            32,
            1,
            353,
            64,
            Precision::Int8Accumulator(Quantization {
                input_zero_point: 0,
                output_zero_point: 0,
                weight_zero_point: 0,
                ..quantization()
            }),
        );
        let plan = ConvPlan::new(shape, [1, 1]);
        assert_eq!(plan.tiles().len(), 2, "this is the first failing Cin plan");

        let staged = plan.programs_with_staged_accumulator_output(RELOCATION);
        assert_eq!(staged.programs.len(), plan.tiles().len());
        assert_eq!(staged.tiles.len(), plan.tiles().len());
        assert_eq!(staged.scratch_bytes, shape.output_scratch_bytes([1, 1]));

        let mut next_offset = 0;
        for (index, ((tile, output), program)) in plan
            .tiles()
            .iter()
            .zip(&staged.tiles)
            .zip(&staged.programs)
            .enumerate()
        {
            let tile_pixels = tile.rows.out_rows as usize * tile.columns.out_cols as usize;
            assert_eq!(output.scratch_offset, next_offset, "tile {index}");
            assert_eq!(output.scratch_bytes, tile_pixels * 2 * 128, "tile {index}");
            assert_eq!(output.output_row, tile.rows.out_first as usize);
            assert_eq!(output.output_column, tile.columns.out_first as usize);
            assert_eq!(output.output_rows, tile.rows.out_rows as usize);
            assert_eq!(output.output_columns, tile.columns.out_cols as usize);
            assert_eq!(
                value_of::<DpuDstBaseAddr>(program),
                RELOCATION.output + output.scratch_offset as u32,
                "tile {index} destination"
            );
            assert_eq!(
                value_of::<DpuDstSurfStride>(program),
                tile_pixels as u32 * FEATURE_ATOM_BYTES,
                "tile {index} surface stride"
            );
            assert_eq!(
                value_of::<DpuDataCubeNotchAddr>(program),
                0,
                "tile {index} notch"
            );
            next_offset += output.scratch_bytes;
        }
        assert_eq!(next_offset, staged.scratch_bytes);
    }

    #[test]
    fn contiguous_accumulator_column_tile_drops_shared_image_notch() {
        let shape = Shape::with_precision(
            32,
            32,
            1,
            64,
            64,
            Precision::Int8Accumulator(Quantization {
                input_zero_point: 0,
                output_zero_point: 0,
                weight_zero_point: 0,
                ..quantization()
            }),
        );
        let tile = Tile2D {
            rows: Tile::whole(shape, [1, 1]),
            columns: ColumnTile::from_output_range(shape, [1, 1], 4, 12),
        };
        let data_banks = shape.data_banks([1, 1]);
        let weight_banks = shape.weight_banks([1, 1]);
        let shared = conv_2d_tile_program(
            shape,
            [1, 1],
            &tile,
            feature_grains([1, 1], &tile.rows),
            data_banks,
            weight_banks,
            OutputPlacement::SharedImage,
        );
        let contiguous = conv_2d_tile_program(
            shape,
            [1, 1],
            &tile,
            feature_grains([1, 1], &tile.rows),
            data_banks,
            weight_banks,
            OutputPlacement::ContiguousTile,
        );

        assert_eq!(value_of::<DpuDstBaseAddr>(&shared), 4 * 16);
        assert_eq!(value_of::<DpuDstSurfStride>(&shared), 32 * 32 * 16);
        assert_eq!(value_of::<DpuDataCubeNotchAddr>(&shared), 0x14_0014);
        assert_eq!(value_of::<DpuDstBaseAddr>(&contiguous), 0);
        assert_eq!(value_of::<DpuDstSurfStride>(&contiguous), 12 * 32 * 16);
        assert_eq!(value_of::<DpuDataCubeNotchAddr>(&contiguous), 0);
    }

    #[test]
    #[should_panic(expected = "currently requires zero input, weight, and output zero-points")]
    fn int8_accumulator_output_rejects_unvalidated_affine_zero_points() {
        let mut quantization = quantization();
        quantization.input_zero_point = 1;
        let _ = Shape::with_precision(4, 4, 1, 1, 8, Precision::Int8Accumulator(quantization));
    }

    #[test]
    fn fp16_and_int8_differ_only_where_the_corpus_says_they_do() {
        // A paired diff, the same comparison the corpus was built to allow.
        // The fp16 side is already hardware-validated, so this pins the int8
        // side against it rather than against a second unknown.
        let fp16 = Shape::new(32, 32);
        let int8 = Shape::with_precision(32, 32, 1, 3, 8, captured_int8());
        let a = conv_2d_tile(fp16, [3, 3], &Tile::whole(fp16, [3, 3]));
        let b = conv_2d_tile(int8, [3, 3], &Tile::whole(int8, [3, 3]));
        assert_eq!(a.len(), b.len(), "precision must not change program length");

        let differing = a
            .iter()
            .zip(&b)
            .filter(|(left, right)| left.0 != right.0)
            .count();
        // Seventeen distinct registers, and `CNA_CONV_CON1` is written
        // twice per program, so eighteen words. That is fewer than the 33
        // fields the sweep reports across all geometries for two reasons:
        // several fields share a register, and several move only where the
        // channel padding or the bank split reacts, neither of which does
        // at Cin 3 Cout 8.
        assert_eq!(
            differing, 18,
            "unexpected number of registers differing between precisions"
        );
    }

    #[test]
    fn requantization_multiplier_normalizes_its_mantissa() {
        // Every OUT_CVT_SCALE in the corpus lands in [2^14, 2^15), with the
        // shift chosen to put it there. These are real (scale, shift) pairs
        // from int8 captures; re-encoding their ratio must reproduce them.
        for (scale, shift) in [
            (19636u32, 24u32),
            (27245, 23),
            (29533, 26),
            (32573, 24),
            (16625, 24),
            (23916, 25),
        ] {
            let encoded = Multiplier::from_ratio(Multiplier { scale, shift }.ratio());
            assert_eq!(
                (encoded.scale, encoded.shift),
                (scale, shift),
                "round trip of {scale}/2^{shift}"
            );
        }

        // The normalization itself, over a wide range of multipliers.
        for exponent in -30i32..4 {
            for step in 1..8 {
                let ratio = 2f64.powi(exponent) * (1.0 + f64::from(step) / 8.0);
                let encoded = Multiplier::from_ratio(ratio);
                assert!(
                    (MANTISSA_FLOOR..2 * MANTISSA_FLOOR).contains(&encoded.scale),
                    "mantissa {} out of range for ratio {ratio}",
                    encoded.scale
                );
                // Within half a mantissa step of the real value.
                let error = (encoded.ratio() - ratio).abs() / ratio;
                assert!(error < 1.0 / f64::from(MANTISSA_FLOOR), "ratio {ratio}");
            }
        }
    }

    #[test]
    fn unit_bs_multiplier_cancels_out_of_the_requantisation() {
        // The composite gain is `(bs_mul >> 7) * (scale / 2^shift)`, so a
        // unit plane entry has to be divided back out. Measured on hardware:
        // at cvt_shift 14 a BS multiplier of 128 gives unit gain, and the
        // output doubles with each doubling of it from there.
        let bs_gain = f64::from(BS_UNIT_MULTIPLIER >> BS_MULTIPLIER_SHIFT);
        assert_eq!(bs_gain, 128.0);
        for exponent in 0..8u32 {
            let wanted = 1.0 / f64::from(1u32 << exponent);
            let multiplier = Multiplier::for_unit_bs(wanted);
            let composite = bs_gain * multiplier.ratio();
            assert!(
                (composite - wanted).abs() < wanted / 1024.0,
                "composite gain {composite} for a requested {wanted}"
            );
        }
        // The probe's own crossing point, restated: unit total gain leaves
        // the per-tensor stage at 2^-7 of unity, which normalises to a
        // mantissa of 2^14 at shift 21.
        let unit = Multiplier::for_unit_bs(1.0);
        assert_eq!((unit.scale, unit.shift), (1 << 14, 21));
    }

    #[test]
    fn int8_bias_packing_normalizes_and_pads_bs_entries() {
        let logical = [200i32, -100i32]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        let mut packed = vec![0xa5; bs_buffer_bytes(8)];
        let written = pack_int8_bias_to_bs(&logical, 2, 8, 0.5, 0.25, 7, &mut packed).unwrap();
        assert_eq!(written, bs_buffer_bytes(8));
        assert_eq!(i32::from_le_bytes(packed[0..4].try_into().unwrap()), 1600);
        assert_eq!(i32::from_le_bytes(packed[4..8].try_into().unwrap()), -800);
        assert_eq!(i16::from_le_bytes(packed[32..34].try_into().unwrap()), -7);
        assert_eq!(i16::from_le_bytes(packed[34..36].try_into().unwrap()), -7);
        assert!(packed[8..32].iter().all(|&byte| byte == 0));
    }

    #[test]
    fn bs_buffer_matches_the_converted_models() {
        // Byte-for-byte against `co16`, a Cout=16 model with per-channel
        // weight magnitudes 0.01*(c+1) and biases 0.05*(c+1). Two blocks,
        // and the second block is what shows the layout repeats rather than
        // running as one flat plane.
        let entries: Vec<BsEntry> = (0..16)
            .map(|c| BsEntry {
                bias: 162675,
                constant: BS_CONSTANT,
                multiplier: 1024 * (c + 1),
            })
            .collect();
        let mut buffer = vec![0u8; bs_buffer_bytes(16)];
        assert_eq!(buffer.len(), 128);
        write_bs_buffer(&mut buffer, &entries);

        // First block: biases at 0, the constant plane at 32, multipliers
        // at 48.
        assert_eq!(&buffer[0..4], &162675i32.to_le_bytes());
        assert_eq!(&buffer[32..34], &128i16.to_le_bytes());
        assert_eq!(&buffer[48..50], &1024i16.to_le_bytes());
        assert_eq!(&buffer[62..64], &8192i16.to_le_bytes());
        // Second block repeats the same three planes for channels 8..15.
        assert_eq!(&buffer[64..68], &162675i32.to_le_bytes());
        assert_eq!(&buffer[96..98], &128i16.to_le_bytes());
        assert_eq!(&buffer[112..114], &9216i16.to_le_bytes());
        assert_eq!(&buffer[126..128], &16384i16.to_le_bytes());

        // A partial block is padded, not packed: Cout 4 still occupies 64
        // bytes with the unused lanes left zero.
        let mut partial = vec![0xffu8; bs_buffer_bytes(4)];
        assert_eq!(partial.len(), 64);
        write_bs_buffer(
            &mut partial,
            &[
                BsEntry {
                    bias: 162675,
                    constant: BS_CONSTANT,
                    multiplier: 4096,
                },
                BsEntry {
                    bias: 162675,
                    constant: BS_CONSTANT,
                    multiplier: 8192,
                },
                BsEntry {
                    bias: 162675,
                    constant: BS_CONSTANT,
                    multiplier: 12288,
                },
                BsEntry {
                    bias: 162675,
                    constant: BS_CONSTANT,
                    multiplier: 16384,
                },
            ],
        );
        assert_eq!(
            &partial[48..56],
            &[0x00, 0x10, 0x00, 0x20, 0x00, 0x30, 0x00, 0x40]
        );
        assert!(partial[16..32].iter().all(|&b| b == 0), "unused bias lanes");
        assert!(partial[56..64].iter().all(|&b| b == 0), "unused mul lanes");

        // The default entry is what a uniform-scale, zero-bias convolution
        // needs, and it is not a zeroed buffer.
        assert_eq!(BsEntry::default().multiplier, BS_UNIT_MULTIPLIER);
        assert_ne!(BsEntry::default().multiplier, 0);
    }

    #[test]
    fn tile_capacity_charges_surfaces_per_atom() {
        // A surface pixel costs `weight_atoms` times what a dense one does,
        // so the same bank allocation carries proportionally fewer rows.
        // Charging one atom per pixel here let a 256x32 Cin 32 tile claim 44
        // rows against a real capacity of 22, and the hardware quietly
        // dropped the rows past the end -- the taps below the cut came back
        // as if they were off the bottom of the image.
        let narrow = Shape::with_out_channels(256, 32, 1, 32, 16);
        assert_eq!(narrow.data_banks([3, 3]), 11);
        assert_eq!(narrow.max_tile_input_rows([3, 3]), 22);
        assert_eq!(narrow.min_tiles([3, 3]), 2);

        // The same shape with a kernel set big enough to take a second bank
        // for coefficients loses two rows of capacity with it. Both numbers
        // are the vendor's own split: 22/12 rows here, 20/14 below.
        let wide = Shape::with_out_channels(256, 32, 1, 32, 64);
        assert_eq!(wide.data_banks([3, 3]), 10);
        assert_eq!(wide.max_tile_input_rows([3, 3]), 20);
        assert_eq!(wide.min_tiles([3, 3]), 2);

        // The dense regime is unchanged: one atom per pixel is correct there,
        // which is why the Cin 3 width sweep never saw this.
        for (width, rows) in [(256u32, 32u32), (512, 22), (768, 14), (1024, 11), (1536, 7)] {
            assert_eq!(
                Shape::new(width, 32).max_tile_input_rows([3, 3]),
                rows,
                "dense capacity at {width} wide"
            );
        }
    }

    #[test]
    fn tiles_of_a_plan_write_disjoint_output_ranges() {
        for tiles in 1..=3 {
            let split = Tile::split(Shape::CAPTURED, [3, 3], tiles);
            let mut covered = vec![0u32; IMAGE_HEIGHT as usize];
            for tile in &split {
                for row in tile.out_first..tile.out_first + tile.out_rows {
                    covered[row as usize] += 1;
                }
            }
            assert!(
                covered.iter().all(|&n| n == 1),
                "{tiles}-tile plan does not partition the output exactly"
            );
        }
    }

    const RELOCATION: Buffers = Buffers {
        input: 0x1_0000,
        weights: 0x2_0000,
        bias: 0x3_0000,
        output: 0x4_0000,
    };

    #[test]
    fn relocation_assigns_addresses_to_a_whole_image_program() {
        let mut commands = conv_2d([3, 3]);
        // A whole-image program starts at both tensors' base, so there is no
        // offset for the relocation to preserve and it reads as assignment.
        assert_eq!(value_of::<CnaFeatureDataAddr>(&commands), 0);
        assert_eq!(value_of::<DpuDstBaseAddr>(&commands), 0);

        relocate(&mut commands, RELOCATION);

        assert_eq!(value_of::<CnaFeatureDataAddr>(&commands), 0x1_0000);
        assert_eq!(value_of::<CnaDcompAddr0>(&commands), 0x2_0000);
        assert_eq!(value_of::<DpuRdmaBsBaseAddr>(&commands), 0x3_0000);
        assert_eq!(value_of::<DpuDstBaseAddr>(&commands), 0x4_0000);
    }

    #[test]
    fn relocation_adds_the_tile_offset_rather_than_overwriting_it() {
        let shape = Shape::CAPTURED;
        let kernels = [3, 3];
        let split = Tile::split(shape, kernels, 2);
        let mut commands = conv_2d_tile(shape, kernels, &split[1]);
        // The second tile of a two-way split starts partway into the feature
        // map and partway into the output -- the captured values the
        // six-group tile test pins independently.
        assert_eq!(value_of::<CnaFeatureDataAddr>(&commands), 0xb40);
        assert_eq!(value_of::<DpuDstBaseAddr>(&commands), 0x2000);

        relocate(&mut commands, RELOCATION);

        assert_eq!(value_of::<CnaFeatureDataAddr>(&commands), 0x1_0000 + 0xb40);
        assert_eq!(value_of::<DpuDstBaseAddr>(&commands), 0x4_0000 + 0x2000);
        // Weights and bias are whole-tensor: every tile reads all of them, so
        // these two carry no offset to add.
        assert_eq!(value_of::<CnaDcompAddr0>(&commands), 0x2_0000);
        assert_eq!(value_of::<DpuRdmaBsBaseAddr>(&commands), 0x3_0000);
    }

    #[test]
    fn exact_output_relocation_preserves_only_the_input_tile_offset() {
        let shape = Shape::CAPTURED;
        let kernels = [3, 3];
        let split = Tile::split(shape, kernels, 2);
        let mut commands = conv_2d_tile(shape, kernels, &split[1]);

        relocate_with_exact_output(&mut commands, RELOCATION);

        assert_eq!(value_of::<CnaFeatureDataAddr>(&commands), 0x1_0000 + 0xb40);
        assert_eq!(value_of::<CnaDcompAddr0>(&commands), 0x2_0000);
        assert_eq!(value_of::<DpuRdmaBsBaseAddr>(&commands), 0x3_0000);
        assert_eq!(value_of::<DpuDstBaseAddr>(&commands), 0x4_0000);
    }

    #[test]
    #[should_panic(expected = "not 16-byte aligned")]
    fn relocation_rejects_a_misaligned_address() {
        let mut commands = conv_2d([1, 1]);
        relocate(
            &mut commands,
            Buffers {
                input: 0x1_0008,
                ..RELOCATION
            },
        );
    }

    #[test]
    fn plan_programs_with_buffers_relocates_every_tile() {
        let shape = Shape::with_out_channels(256, 32, 1, 32, 64);
        let plan = ConvPlan::new(shape, [5, 5]);
        assert_eq!(plan.tiles().len(), 3);

        let bare = plan.programs();
        let bound = plan.programs_with_buffers(RELOCATION);
        assert_eq!(bound.len(), bare.len());

        for (tile, (bare, bound)) in bare.iter().zip(&bound).enumerate() {
            assert_eq!(
                value_of::<CnaFeatureDataAddr>(bound),
                0x1_0000 + value_of::<CnaFeatureDataAddr>(bare),
                "tile {tile} input"
            );
            assert_eq!(
                value_of::<DpuDstBaseAddr>(bound),
                0x4_0000 + value_of::<DpuDstBaseAddr>(bare),
                "tile {tile} output"
            );
            // Relocation touches the four address registers and nothing else.
            let changed = bare
                .iter()
                .zip(bound)
                .filter(|(left, right)| left.0 != right.0)
                .count();
            assert_eq!(changed, 4, "tile {tile} changed words");
        }

        // Distinct tiles really do land at distinct output addresses -- a
        // relocation that overwrote instead of adding would collapse these.
        let outputs: Vec<_> = bound
            .iter()
            .map(|program| value_of::<DpuDstBaseAddr>(program))
            .collect();
        assert_eq!(outputs[0], 0x4_0000);
        assert!(
            outputs[0] < outputs[1] && outputs[1] < outputs[2],
            "tile outputs are not strictly increasing: {outputs:x?}"
        );
    }

    /// Whole-register values read off the fp16 activation sweep, at
    /// `conv-w32-h32-k3-s1-ci32-co32` and its four activated siblings. The
    /// same three `DPU_BN_CFG` values appear in all fourteen fp16
    /// comparison groups, across both feature layouts and both kernels.
    #[test]
    fn activation_programs_the_captured_bn_registers() {
        let base = Shape::with_out_channels(32, 32, 1, 32, 32);
        for (activation, bn_cfg, cmp) in [
            (Activation::None, 0x53, 0),
            (Activation::Relu, 0x12, 0),
            (Activation::clamped_fp16(6.0), 0x92, 0x40C0_0000),
            (Activation::clamped_fp16(2.0), 0x92, 0x4000_0000),
            (Activation::clamped_fp16(1.0), 0x92, 0x3F80_0000),
        ] {
            let program = conv_2d_tile(
                base.with_activation(activation),
                [3, 3],
                &Tile::whole(base, [3, 3]),
            );
            assert_eq!(value_of::<DpuBnCfg>(&program), bn_cfg, "{activation:?}");
            assert_eq!(
                value_of::<DpuBnReluxCmpValue>(&program),
                cmp,
                "{activation:?} cmp"
            );
            // The BN stage needs no operand: these are zero in every
            // capture, activated or not.
            assert_eq!(value_of::<DpuBnAluCfg>(&program), 0);
            assert_eq!(value_of::<DpuBnMulCfg>(&program), 0);
            assert_eq!(value_of::<DpuRdmaBnBaseAddr>(&program), 0);
            // The vendor leaves BS alone; only BN moves. The retired
            // Mesa-derived builder fused activation here instead, which is
            // what this pins against.
            assert_eq!(
                value_of::<DpuBsCfg>(&program),
                0x2_0150,
                "{activation:?} BS"
            );
            assert_eq!(value_of::<DpuBsReluxCmpValue>(&program), 0);
        }
    }

    #[test]
    fn activation_changes_only_the_two_bn_words() {
        let shape = Shape::with_out_channels(32, 32, 1, 32, 32);
        let tile = Tile::whole(shape, [3, 3]);
        let plain = conv_2d_tile(shape, [3, 3], &tile);
        for (activation, expected) in [
            // Relu leaves the cmp value at its unactivated zero.
            (Activation::Relu, 1),
            (Activation::clamped_fp16(6.0), 2),
            (Activation::clamped_int8(6.0, 0.02, 0.003), 2),
        ] {
            let activated = conv_2d_tile(shape.with_activation(activation), [3, 3], &tile);
            let changed = plain
                .iter()
                .zip(&activated)
                .filter(|(left, right)| left.0 != right.0)
                .count();
            assert_eq!(changed, expected, "{activation:?} changed words");
        }
    }

    #[test]
    fn int8_clamp_is_the_ceiling_in_the_post_bs_domain() {
        // The capture-derived accumulator-unit ceiling is multiplied by the
        // effective gain of the default BS plane before BN sees it.
        let (input, weights) = (0.02, 0.003);
        let cmp = |ceiling| match Activation::clamped_int8(ceiling, input, weights) {
            Activation::Clamped { cmp } => cmp,
            other => panic!("expected a clamp, got {other:?}"),
        };
        let bs_gain = u32::from(
            u16::try_from(BS_UNIT_MULTIPLIER >> BS_MULTIPLIER_SHIFT)
                .expect("the unit BS gain must be positive"),
        );
        assert_eq!(
            cmp(1.0),
            (1.0 / (f64::from(input) * f64::from(weights)) * f64::from(bs_gain)).round() as u32
        );
        // Linear in the ceiling, but only to within the rounding -- which is
        // exactly what the captures show: 6 x 86815 is 520890 against a
        // captured 520891, and 2 x 276086 is 552172 against 552171.
        for multiple in [2u32, 6] {
            let scaled = cmp(multiple as f32);
            let exact = multiple * cmp(1.0);
            assert!(
                scaled.abs_diff(exact) <= 2,
                "x{multiple}: {scaled} is not within rounding of {exact}"
            );
        }
        // Not the fp16 encoding, which is what the two precisions differ on.
        assert_ne!(cmp(6.0), 6.0f32.to_bits());
    }

    /// The depthwise channel ladder, both precisions, read off the nine
    /// captures. The retired Mesa-derived channel rule would say 64, 64 and
    /// 128 for the fp16 8, 32 and 96 rows.
    #[test]
    fn depthwise_pads_channels_to_the_captured_granule() {
        for (channels, fp16, int8) in [
            (8u32, 32u32, 64u32),
            (16, 32, 64),
            (32, 32, 64),
            (48, 64, 64),
            (64, 64, 64),
            (96, 96, 128),
            (128, 128, 128),
        ] {
            let shape = Shape::with_out_channels(32, 32, 1, channels, channels).with_depthwise();
            assert_eq!(shape.padded_out_channels(), fp16, "fp16 c{channels}");
            let quantized = Shape::with_precision(32, 32, 1, channels, channels, captured_int8())
                .with_depthwise();
            assert_eq!(quantized.padded_out_channels(), int8, "int8 c{channels}");
        }
    }

    /// `CNA_WEIGHT_SIZE0.weight_bytes` across the same ladder: one filter
    /// per input channel, with the channel count padded to a whole CBUF
    /// atom group. The int8 48-channel row is the one that separates this
    /// from the dense weight padding, which would read 432 rather than 576.
    #[test]
    fn depthwise_weight_bytes_match_the_captures() {
        for (channels, fp16, int8) in [
            (8u32, 144u32, 144u32),
            (16, 288, 144),
            (32, 576, 288),
            (48, 864, 576),
            (64, 1152, 576),
            (96, 1728, 864),
            (128, 2304, 1152),
        ] {
            let shape = Shape::with_out_channels(32, 32, 1, channels, channels).with_depthwise();
            assert_eq!(shape.weight_bytes([3, 3]), fp16, "fp16 c{channels}");
            let quantized = Shape::with_precision(32, 32, 1, channels, channels, captured_int8())
                .with_depthwise();
            assert_eq!(quantized.weight_bytes([3, 3]), int8, "int8 c{channels}");
        }
        // The 5x5 point, which says the rule is not 3x3-specific.
        let five = Shape::with_out_channels(32, 32, 1, 32, 32).with_depthwise();
        assert_eq!(five.weight_bytes([5, 5]), 1600);
        assert_eq!(
            Shape::with_precision(32, 32, 1, 32, 32, captured_int8())
                .with_depthwise()
                .weight_bytes([5, 5]),
            800
        );
    }

    /// The ten fields the depthwise diff found moving, at the geometry the
    /// dense control shares.
    #[test]
    fn depthwise_programs_the_captured_registers() {
        let shape = Shape::with_out_channels(32, 32, 1, 32, 32);
        let tile = Tile::whole(shape, [3, 3]);
        let dense = conv_2d_tile(shape, [3, 3], &tile);
        let depthwise = conv_2d_tile(shape.with_depthwise(), [3, 3], &tile);

        // weight_kernels drops to 1 whatever the channel count.
        assert_eq!(value_of::<CnaWeightSize2>(&dense) & 0x3fff, 32);
        assert_eq!(value_of::<CnaWeightSize2>(&depthwise) & 0x3fff, 1);
        assert_eq!(value_of::<CnaWeightSize0>(&dense), 18432);
        assert_eq!(value_of::<CnaWeightSize0>(&depthwise), 576);
        // SURF_ADD doubles. The field sits in register bits 31:4, so the
        // raw word is sixteen times the logical value the captures report.
        assert_eq!(value_of::<DpuSurfaceAdd>(&dense) >> 4, 2048);
        assert_eq!(value_of::<DpuSurfaceAdd>(&depthwise) >> 4, 4096);
        // DW_EN, and the three conv_mode copies.
        assert_ne!(
            value_of::<CoreMiscCfg>(&dense),
            value_of::<CoreMiscCfg>(&depthwise)
        );
        for (name, dense_word, dw_word) in [
            // Written twice per program, always with the same value.
            (
                "cna",
                first_value_of::<CnaConvCon1>(&dense),
                first_value_of::<CnaConvCon1>(&depthwise),
            ),
            (
                "dpu",
                value_of::<DpuFeatureModeCfg>(&dense),
                value_of::<DpuFeatureModeCfg>(&depthwise),
            ),
            (
                "dpu_rdma",
                value_of::<DpuRdmaFeatureModeCfg>(&dense),
                value_of::<DpuRdmaFeatureModeCfg>(&depthwise),
            ),
        ] {
            assert_ne!(dense_word, dw_word, "{name} conv_mode");
        }
    }

    #[test]
    fn depthwise_at_stride_two_still_doubles_the_surface() {
        // The capture whose output geometry differs, which is what makes
        // the doubling a factor rather than a constant 4096.
        let shape = Shape::with_out_channels(32, 32, 2, 32, 32);
        let tile = Tile::whole(shape, [3, 3]);
        assert_eq!(
            value_of::<DpuSurfaceAdd>(&conv_2d_tile(shape, [3, 3], &tile)) >> 4,
            512
        );
        assert_eq!(
            value_of::<DpuSurfaceAdd>(&conv_2d_tile(shape.with_depthwise(), [3, 3], &tile)) >> 4,
            1024
        );
    }

    #[test]
    #[should_panic(expected = "channel multiplier of one")]
    fn depthwise_refuses_a_channel_multiplier_above_one() {
        let _ = Shape::with_out_channels(32, 32, 1, 32, 64).with_depthwise();
    }
}
