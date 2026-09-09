//! What the *compiler* is willing to claim, as opposed to what the hardware
//! can be programmed to do.
//!
//! [`crate::conv::ConvPlan`] answers "can this be programmed, and how" --
//! register fields, CBUF residency, tile geometry. That is hardware
//! legality, and it needs the spatial extents. This module answers a
//! narrower and different question: "is this shape class inside the corpus
//! and the board measurements the matchers were written from" -- which
//! depends only on the precision, the kernel, the stride and the channel
//! counts, and so is answerable on a convolution whose height and width are
//! still dynamic.
//!
//! Both are required. A shape can be inside the envelope here and still be
//! refused by `ConvPlan` at its real extents (a dense row too wide for one
//! CBUF bank), and it can plan cleanly while sitting outside every
//! measurement anyone has taken.
//!
//! # Why this exists
//!
//! Until 2026-09-09 these ceilings lived in
//! `rocket_conv2d_transform_spec.mlir` as `transform.iree.match.dim_bounds`
//! lines, two or three per matcher, across sixty-odd matchers. That had two
//! defects. The numbers drifted apart between matchers that describe the
//! *same* hardware path -- a stride-2 1x1 convolution was admitted to `Cin`
//! 3584 when a ReLU followed it and to 512 when nothing did -- and they
//! were unreachable from the runtime, so nothing could check that the
//! compiler and the HAL agreed about what was measured. Both are fixed by
//! having one table, here, that the plugin reads through
//! `rocket-plan-ffi`'s `rocket_admit_conv`.
//!
//! # The table the matchers had
//!
//! Recorded so the consolidation below is reviewable rather than asserted.
//! `Cin`/`Cout`, fp16 dense, by kernel and stride:
//!
//! | matcher family | k1 s1 | k1 s2 | k1 s3/s4 | k3 s1 | k3 s2 |
//! |---|---|---|---|---|---|
//! | plain | 3584/3584 | 512/512 | 512/512 | 1152/1792 | 512/512 |
//! | `+relu` | 3584/3584 | 3584/3584 | -- | 1152/1792 | 1152/1792 |
//! | `+bias` | 3584/3584 | 3584/3584 | -- | 1152/1792 | 1152/1792 |
//! | `pad1` | -- | -- | -- | 1152/3584 | 512..1152/3584 |
//!
//! The rows differ only in the epilogue fused after the convolution, which
//! the CNA's channel path does not see at all: the bias rides the BS plane
//! and the activation the BN plane, both downstream of the MAC array. So
//! the spread is matcher accretion, not five different measurements, and
//! each class below takes the largest of its row -- every one of which was
//! already live in a shipping matcher against the same hardware path.
//!
//! int8 is keyed on the precision the lowering *programs*, which is the one
//! distinction in the old table that was real: the accumulator path
//! (`i8 x i8 -> i32`, [`Precision::Int8Accumulator`]) and the requantized
//! path ([`Precision::Int8`]) have separate corpora, and the requantized
//! one stops earlier.

use crate::{
    conv::{self, Precision},
    error::{PlanError, PlanErrorCode},
};

/// The channel envelope for one shape class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChannelCeilings {
    pub in_channels: u32,
    pub out_channels: u32,
    /// Smallest output-channel count with evidence. One on every class but
    /// the requantized int8 ones, where it is a whole 16-channel output
    /// atom; see [`conv_ceilings`].
    pub min_out_channels: u32,
}

/// Largest stride any matcher describes. The captured programs stop here
/// and so does every board sweep; `CNA_CONV_CON1.conv_x_stride` would hold
/// more.
pub const MAX_ADMITTED_STRIDE: u64 = 4;

/// A convolution the compiler is considering claiming, in the terms the
/// admission envelope is indexed by. Spatial extents are deliberately
/// absent: they are `ConvPlan`'s question, and they are frequently still
/// dynamic when this one has to be answered.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConvAdmission {
    /// The precision the lowering will *program*, which for int8 is not
    /// implied by the operand types: a requantized convolution and an
    /// accumulator convolution are both `i8 x i8 -> i32` in the IR.
    pub precision: Precision,
    pub depthwise: bool,
    pub kernel_height: u64,
    pub kernel_width: u64,
    pub stride: u64,
    pub in_channels: u64,
    pub out_channels: u64,
}

/// A matmul the compiler is considering claiming. `[m, k] x [k, n]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MatmulAdmission {
    pub precision: Precision,
    pub m: u64,
    pub k: u64,
    pub n: u64,
}

/// How the precision table is keyed. Two int8 rungs, because the two int8
/// lowerings have separate evidence; everything else that has a matcher is
/// the fp16 rung.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PrecisionClass {
    /// fp16 operands with an fp32 accumulator: every dense and depthwise
    /// floating-point matcher.
    Fp16,
    /// `i8 x i8 -> i32` written out as int32 -- the accumulator path.
    Int8Accumulator,
    /// int8 in, int8 out, requantized on the DPU's own output stage.
    Int8Requant,
}

impl PrecisionClass {
    fn of(precision: Precision) -> Option<PrecisionClass> {
        match precision {
            Precision::Fp16 => Some(PrecisionClass::Fp16),
            Precision::Int8Accumulator(_) => Some(PrecisionClass::Int8Accumulator),
            Precision::Int8(_) => Some(PrecisionClass::Int8Requant),
            // bf16, int16, int4, tf32 and the fp32-accumulator fp16 rung
            // are all implemented in the HAL and exercised by the datatype
            // matrix on hardware, but no matcher in the transform spec
            // claims one, so the compiler has no envelope for them.
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            PrecisionClass::Fp16 => "fp16",
            PrecisionClass::Int8Accumulator => "int8 (accumulator)",
            PrecisionClass::Int8Requant => "int8 (requantized)",
        }
    }
}

/// The channel envelope for a class, or `None` when no matcher describes
/// the class at all.
///
/// `kernel` is the square kernel extent; non-square kernels have no matcher
/// and no entry. `stride` matters only on the fp16 dense rows, where 3 and
/// 4 were never taken past 512 in either direction.
pub fn conv_ceilings(
    precision: Precision,
    depthwise: bool,
    kernel: u64,
    stride: u64,
) -> Option<ChannelCeilings> {
    let class = PrecisionClass::of(precision)?;
    if stride == 0 || stride > MAX_ADMITTED_STRIDE {
        return None;
    }
    let ceilings = |in_channels, out_channels| {
        Some(ChannelCeilings {
            in_channels,
            out_channels,
            min_out_channels: 1,
        })
    };
    // Every entry below carries the evidence that set it. These notes came
    // from the matcher comments in `rocket_conv2d_transform_spec.mlir`,
    // where they sat next to the `dim_bounds` lines this table replaced.
    match (class, depthwise, kernel) {
        // **fp16 depthwise**, raised 512 -> 1536 on 2026-09-09.
        //
        // 512 was where the fp16 depthwise matchers had stood since they
        // were written, and nothing had gone back to it: the int8 depthwise
        // rung went to 1344 in September, [`conv::MAX_DEPTHWISE_CHANNELS`]
        // is 1792, and `ConvPlan` plans fp16 depthwise cleanly to that
        // ceiling. It bit: MobileNetV2's own fp16 depthwise convolutions at
        // C=576 (14x14 and 7x7) and C=960 (7x7) sat *above* it, so six
        // dispatch sites in a model this repo measures every week were on
        // the CPU for want of a number.
        //
        // 1536, not 1792, because 1536 is what the compiled end-to-end gate
        // proves. `tools/e2e_conv_regression.py` compiles a Rocket and a CPU
        // module from the same MLIR and compares them on `planck`, and its
        // `depthwise_fp16_c576`, `_c960`, `_c1536` and `_c1536_s2` cases
        // cover the two model widths, the ceiling itself and a strided
        // multi-tile plan at it. Raise it the way every other limit here
        // moves: measure the next rung first.
        (PrecisionClass::Fp16, true, 1 | 3) => ceilings(1536, 1536),
        // **fp16 dense 1x1**, both channel axes at the dense ceilings.
        //
        // `Cin`: [`conv::MAX_INPUT_CHANNELS`], 512 -> 1344 (2026-09-03) on
        // hardware -- k=1 exact at 14x14 Cout 64 for Cin 256..1792 across
        // one to five tiles, with the fp16 vendor corpus agreeing
        // (`conv_vendor_fixture_wide.rs`) -- then 1344 -> 3584
        // (2026-09-06): quiet board, `Selectors` for addressing and
        // `Counting` for lane coverage at every point, the `onehot` read
        // map at Cout == Cin, Cin 1792 through 3584 in 256-channel steps
        // and on to 8192, ragged 1793..4095, 56x56 multi-tile to 3584.
        //
        // `Cout`: [`conv::MAX_OUTPUT_CHANNELS`], 528 -> 1792 -> 3584.
        // Exact at 7x7 Cin 448 for Cout 528, 640, 768, 1024, 1344, 1792,
        // 2048, then 2304, 2560, 3072, 3584 and 4096, CBUF split flat at
        // 2d/10w over the whole range -- the high-channel divergence is
        // indexed by `Cin`, not `Cout`. Ragged Cout 1793, 2049, 2313, 3073,
        // 3585 and 4095 are exact too.
        //
        // Stride 2 rides the same numbers: the sweep covers stride 2 at Cin
        // 1792..4096 and Cout 2304..3584, and the epilogue a matcher fuses
        // (bias on the BS plane, activation on the BN plane) is downstream
        // of the MAC array and does not touch the channel path.
        (PrecisionClass::Fp16, false, 1) if stride <= 2 => {
            ceilings(conv::MAX_INPUT_CHANNELS, conv::MAX_OUTPUT_CHANNELS)
        }
        // **fp16 dense 3x3**. `Cin` stops at 1152, not at
        // [`conv::MAX_INPUT_CHANNELS`]: at a 3x3 kernel the coefficient
        // working set binds first, and `ConvPlan` refuses Cin >= 1216
        // outright because it exceeds the eleven grantable CBUF banks. 1152
        // is hardware-exact at 28x28 Cout 64 for Cin 512..1152, including
        // the 1/11 split at 1152. `Cout` does not charge feature residency
        // and rides the dense ceiling, which the `pad1` matchers already
        // admitted.
        (PrecisionClass::Fp16, false, 3) if stride <= 2 => {
            ceilings(1152, conv::MAX_OUTPUT_CHANNELS)
        }
        // **fp16 dense at stride 3 and 4.** Only ever admitted to 512, and
        // no sweep has been taken at either stride.
        (PrecisionClass::Fp16, false, 1 | 3) => ceilings(512, 512),
        // **int8 accumulator, dense 1x1.**
        // [`conv::MAX_INT8_INPUT_CHANNELS`] 512 -> 1344 -> 3584: 14x14 Cout
        // 64 at Cin 1792, 2304, 3072, 3584 and 4096 under `SelectorsAffine`
        // and again under `Counting`, the `onehot` read map at Cout == Cin
        // 3584, and stride 2 at Cin 2304..4096.
        // [`conv::MAX_INT8_OUTPUT_CHANNELS`] 1792 -> 3584: exact at 7x7 Cin
        // 448 for Cout 768 through 4096, CBUF split flat at 7d/5w.
        (PrecisionClass::Int8Accumulator, false, 1) => ceilings(
            conv::MAX_INT8_INPUT_CHANNELS,
            conv::MAX_INT8_OUTPUT_CHANNELS,
        ),
        // **int8 accumulator, dense 3x3.** `Cin` 1152 for the same reason
        // as fp16's: `ConvPlan` refuses Cin >= 1216, and 1152 is exact at
        // Cout 64 and 448 including the 1/11 splits at 1088 and 1152.
        // `Cout` stays 512 -- the corpus backing above it was established
        // against the 1x1 kernel, not this one.
        (PrecisionClass::Int8Accumulator, false, 3) => ceilings(1152, 512),
        // **int8 accumulator, depthwise.** Raised 512 -> 1344 (2026-09-03)
        // with the depthwise coefficient-model fix: the streamed working
        // set had been using the *dense* product `kh*kw*Cin*64`, which
        // scales with C and asked for 13 of eleven grantable banks at
        // C=1344. A depthwise output channel accumulates over one input
        // channel, so the contraction depth is 1
        // (`Shape::streamed_contraction_channels`).
        (PrecisionClass::Int8Accumulator, true, 1 | 3) => ceilings(1344, 1344),
        // **int8 requantized, dense 1x1.** 1344 is the widest `Cin` a
        // measured model asks for, and it is the model that says so:
        // MobileNetV2-static-int8's 7x7 Cin 1344 -> Cout 448 projection on
        // this path is max|diff| 0.35 against the CPU arm, same argmax and
        // top-5 (`planck`, 2026-09-08). It sat at 816 for two days because
        // admitting 1344 moved the logits to 5.01 while every isolated
        // instrument said the shape was exact -- and it was: that
        // convolution is one of two in the model whose output feeds another
        // convolution with nothing on the CPU between, and the driver
        // packed the chained dispatch's input before the producer had
        // written it (ISSUES.md C13). The HAL sweep is exact to Cin 1792
        // and the compiled differential to 1344; raise this on a model, not
        // on a fixture. The accumulator path's caps do not apply here and
        // never did -- they come from a 384-coefficient-bytes-per-output-
        // channel limit this path does not have.
        //
        // `min_out_channels` is one 16-channel output atom. It was 32 for
        // two days on the strength of MobileNetV2's `112x112 Cin 48 -> Cout
        // 24` projection, which moved the logits to max|diff| 4.71 when
        // admitted; that was the *other* chained edge and the same C13
        // fault, not a width. It is exact now in the same model-level
        // measurement. Below 16 is untested.
        (PrecisionClass::Int8Requant, false, 1) => Some(ChannelCeilings {
            in_channels: 1344,
            out_channels: 1792,
            min_out_channels: 16,
        }),
        // **int8 requantized, dense 3x3.** Raised 512 -> 768, gated by
        // `requant_int8_3x3_cin768`. Lower than the 1x1 ceiling because
        // that is where this kernel's compiled differential stops, not
        // because 3x3 is known to fail above it -- the HAL sweep is exact
        // at 3x3 Cin 1024. MobileNetV2's 3x3 convolutions are all
        // depthwise, so nothing in the measured models needs more.
        (PrecisionClass::Int8Requant, false, 3) => Some(ChannelCeilings {
            in_channels: 768,
            out_channels: 768,
            min_out_channels: 16,
        }),
        // **int8 requantized, depthwise.** The accumulator path's 1344,
        // for the same coefficient-model reason; depthwise `Cout` is always
        // `Cin`, so the output-atom floor does not apply.
        (PrecisionClass::Int8Requant, true, 1 | 3) => ceilings(1344, 1344),
        _ => None,
    }
}

/// Accepts `admission` if the compiler has evidence for its shape class.
///
/// A refusal is [`PlanErrorCode::UnvalidatedConfiguration`]: the hardware
/// may well run the operation, and `ConvPlan` may well plan it. Nobody has
/// measured it, so the compiler leaves it on the CPU.
pub fn admit_conv(admission: &ConvAdmission) -> Result<(), PlanError> {
    let ConvAdmission {
        precision,
        depthwise,
        kernel_height,
        kernel_width,
        stride,
        in_channels,
        out_channels,
    } = *admission;

    let refuse = |message: String| {
        Err(PlanError::new(
            PlanErrorCode::UnvalidatedConfiguration,
            message,
        ))
    };

    let Some(class) = PrecisionClass::of(precision) else {
        return refuse(format!(
            "no Rocket matcher claims a {precision:?} convolution, so the compiler has no \
             admission envelope for one"
        ));
    };
    if kernel_height != kernel_width {
        return refuse(format!(
            "non-square kernel {kernel_height}x{kernel_width}: the matchers describe square \
             1x1 and 3x3 kernels only"
        ));
    }
    if stride == 0 || stride > MAX_ADMITTED_STRIDE {
        return refuse(format!(
            "stride {stride}: the matchers describe strides 1 through {MAX_ADMITTED_STRIDE}"
        ));
    }
    let Some(ceilings) = conv_ceilings(precision, depthwise, kernel_height, stride) else {
        return refuse(format!(
            "no {} {} {kernel_height}x{kernel_width} stride-{stride} convolution has been \
             characterized, so no matcher describes one",
            class.name(),
            if depthwise { "depthwise" } else { "dense" },
        ));
    };
    if in_channels == 0 || out_channels == 0 {
        return Err(PlanError::new(
            PlanErrorCode::InvalidShape,
            format!("channel counts must be positive, got Cin {in_channels} Cout {out_channels}"),
        ));
    }
    if in_channels > u64::from(ceilings.in_channels) {
        return refuse(format!(
            "Cin {in_channels} is past the {} {} {kernel_height}x{kernel_width} stride-{stride} \
             input-channel ceiling of {}",
            class.name(),
            if depthwise { "depthwise" } else { "dense" },
            ceilings.in_channels,
        ));
    }
    if out_channels < u64::from(ceilings.min_out_channels) {
        return refuse(format!(
            "Cout {out_channels} is below the {} {} {kernel_height}x{kernel_width} \
             stride-{stride} output-channel floor of {}",
            class.name(),
            if depthwise { "depthwise" } else { "dense" },
            ceilings.min_out_channels,
        ));
    }
    if out_channels > u64::from(ceilings.out_channels) {
        return refuse(format!(
            "Cout {out_channels} is past the {} {} {kernel_height}x{kernel_width} stride-{stride} \
             output-channel ceiling of {}",
            class.name(),
            if depthwise { "depthwise" } else { "dense" },
            ceilings.out_channels,
        ));
    }
    // Depthwise programs `Cout == Cin`; a channel multiplier is a different
    // operation and `Shape::try_with_depthwise` refuses it downstream.
    if depthwise && in_channels != out_channels {
        return Err(PlanError::new(
            PlanErrorCode::UnsupportedSemantics,
            format!(
                "depthwise convolution with Cin {in_channels} and Cout {out_channels}: the \
                 lowering programs one output channel per input channel"
            ),
        ));
    }
    Ok(())
}

/// `M` ceiling for a matmul.
///
/// `CNA_DATA_SIZE0.datain_width` is 11 bits, so 2047 is the widest row the
/// field holds. Unlike the channel ceilings this one is a hardware limit on
/// a *single tile*, and `ConvPlan` splits a wider `M` into column tiles
/// (ISSUES.md C10, and the sweep at M 90/128/197/296/1035/2000).
///
/// **Raised 2047 -> 4096 on 2026-09-09**, and this one did need a new
/// measurement: the note here previously read "no board measurement has
/// been taken above it, so the compiler stops here", which is why a
/// transformer prefill longer than 2047 tokens was refused by policy while
/// the planner behind it was planning the shape happily. It plans `M`
/// 65536 into 737 column tiles; nothing structural was ever in the way.
///
/// Measured on `planck` from a quiet board, one shape per process, with
/// the **`onehot` read map** rather than the probe's default. That choice
/// is the measurement: on a height-one image -- which is what the FC
/// lowering makes a matmul -- `Selectors` and `Dense` cannot see a pixel
/// shift at all, because their `x*7` term vanishes modulo 7 at `y = 0`, so
/// every pixel of a channel carries the same value. `onehot` encodes each
/// input's own linear index, so a wrong result says *where* the read came
/// from. That is exactly the instrument C10's wide-row fault needed.
///
/// fp16 at `Cin` = `Cout` = 512: `M` 2047 (control), 2048, 2304, 3072,
/// 4096, 6144 and 8192, plus ragged 2049 and 4095 -- 16 to 61 column tiles,
/// 0 mismatches. `M` 4096 again at `Cin` = `Cout` 1024 and 2048, which
/// narrows each tile by widening the slab count (63 and 128 tiles). And `M`
/// 4096 at every other rung: int8, int8-accumulator, bf16, int16, tf32 and
/// fp16-with-fp32-output, all exact.
///
/// 4096 rather than the 8192 the fp16 ladder reached, because this constant
/// is precision-independent and 4096 is the widest `M` measured at *every*
/// rung. Raising it further wants the other rungs measured there first.
pub const MAX_ADMITTED_MATMUL_M: u64 = 4096;

/// Accepts `admission` if the compiler has evidence for the matmul's class.
pub fn admit_matmul(admission: &MatmulAdmission) -> Result<(), PlanError> {
    let MatmulAdmission { precision, m, k, n } = *admission;
    let refuse = |message: String| {
        Err(PlanError::new(
            PlanErrorCode::UnvalidatedConfiguration,
            message,
        ))
    };
    let Some(class) = PrecisionClass::of(precision) else {
        return refuse(format!(
            "no Rocket matcher claims a {precision:?} matmul, so the compiler has no admission \
             envelope for one"
        ));
    };
    if class != PrecisionClass::Fp16 {
        return refuse(format!(
            "only the fp16 matmul matcher exists; {} has no matmul evidence",
            class.name()
        ));
    }
    if m == 0 || k == 0 || n == 0 {
        return Err(PlanError::new(
            PlanErrorCode::InvalidShape,
            format!("matmul extents must be positive, got {m}x{k} x {k}x{n}"),
        ));
    }
    if m > MAX_ADMITTED_MATMUL_M {
        return refuse(format!(
            "M {m} is past the matmul row ceiling of {MAX_ADMITTED_MATMUL_M} \
             (CNA_DATA_SIZE0.datain_width is 11 bits)"
        ));
    }
    // K and N are the convolution's channel counts, so they ride the dense
    // 1x1 ceilings the FC lowering maps onto.
    if k > u64::from(conv::MAX_INPUT_CHANNELS) {
        return refuse(format!(
            "K {k} is past the fp16 input-channel ceiling of {}",
            conv::MAX_INPUT_CHANNELS
        ));
    }
    if n > u64::from(conv::MAX_OUTPUT_CHANNELS) {
        return refuse(format!(
            "N {n} is past the fp16 output-channel ceiling of {}",
            conv::MAX_OUTPUT_CHANNELS
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conv::{Multiplier, Quantization};

    fn int8(quantized: bool) -> Precision {
        let q = Quantization {
            input_zero_point: 0,
            output_zero_point: 0,
            weight_zero_point: 0,
            input_scale: 1.0,
            weights_scale: 1.0,
            multiplier: Multiplier::for_unit_bs(1.0),
        };
        if quantized {
            Precision::Int8(q)
        } else {
            Precision::Int8Accumulator(q)
        }
    }

    fn conv(
        precision: Precision,
        depthwise: bool,
        kernel: u64,
        stride: u64,
        cin: u64,
        cout: u64,
    ) -> ConvAdmission {
        ConvAdmission {
            precision,
            depthwise,
            kernel_height: kernel,
            kernel_width: kernel,
            stride,
            in_channels: cin,
            out_channels: cout,
        }
    }

    #[test]
    fn fp16_dense_1x1_rides_the_dense_ceilings() {
        assert!(admit_conv(&conv(Precision::Fp16, false, 1, 1, 4096, 4096)).is_ok());
        let error = admit_conv(&conv(Precision::Fp16, false, 1, 1, 4097, 64))
            .expect_err("one channel past the ceiling");
        assert_eq!(error.code(), PlanErrorCode::UnvalidatedConfiguration);
        assert!(error.message().contains("4097"), "{error}");
        assert!(error.message().contains("4096"), "{error}");
        assert!(admit_conv(&conv(Precision::Fp16, false, 1, 1, 64, 4097)).is_err());
        // The extent the 2026-09-06 corpus stopped at is still inside.
        assert!(admit_conv(&conv(Precision::Fp16, false, 1, 1, 3584, 3584)).is_ok());
    }

    #[test]
    fn stride_two_matches_stride_one_and_three_does_not() {
        // The consolidation this module exists for: the epilogue a matcher
        // fuses does not move the channel path, so stride 2 gets the same
        // envelope as stride 1. Stride 3 and 4 keep the 512 they had.
        assert!(admit_conv(&conv(Precision::Fp16, false, 1, 2, 3584, 3584)).is_ok());
        assert!(admit_conv(&conv(Precision::Fp16, false, 1, 3, 512, 512)).is_ok());
        assert!(admit_conv(&conv(Precision::Fp16, false, 1, 3, 513, 64)).is_err());
        assert!(admit_conv(&conv(Precision::Fp16, false, 1, 5, 64, 64)).is_err());
    }

    #[test]
    fn fp16_3x3_stops_at_the_coefficient_working_set() {
        assert!(admit_conv(&conv(Precision::Fp16, false, 3, 1, 1152, 3584)).is_ok());
        assert!(admit_conv(&conv(Precision::Fp16, false, 3, 1, 1153, 64)).is_err());
        assert!(admit_conv(&conv(Precision::Fp16, false, 3, 2, 1152, 3584)).is_ok());
    }

    #[test]
    fn the_two_int8_lowerings_have_separate_envelopes() {
        // Same operand types in the IR, different programmed precision.
        assert!(admit_conv(&conv(int8(false), false, 1, 1, 3584, 3584)).is_ok());
        assert!(admit_conv(&conv(int8(true), false, 1, 1, 3584, 64)).is_err());
        assert!(admit_conv(&conv(int8(true), false, 1, 1, 1344, 1792)).is_ok());
        assert!(admit_conv(&conv(int8(true), false, 3, 1, 768, 768)).is_ok());
        assert!(admit_conv(&conv(int8(true), false, 3, 1, 769, 64)).is_err());
        assert!(admit_conv(&conv(int8(false), false, 3, 1, 1152, 512)).is_ok());
        assert!(admit_conv(&conv(int8(false), false, 3, 1, 64, 513)).is_err());
    }

    #[test]
    fn depthwise_has_its_own_ceilings_per_precision() {
        // MobileNetV2's two widest fp16 depthwise convolutions, which sat
        // above the old 512 and were on the CPU because of it.
        assert!(admit_conv(&conv(Precision::Fp16, true, 3, 1, 576, 576)).is_ok());
        assert!(admit_conv(&conv(Precision::Fp16, true, 3, 1, 960, 960)).is_ok());
        assert!(admit_conv(&conv(Precision::Fp16, true, 3, 1, 1536, 1536)).is_ok());
        assert!(admit_conv(&conv(Precision::Fp16, true, 3, 1, 1537, 1537)).is_err());
        // Still under the planner's own depthwise ceiling, deliberately:
        // 1536 is what the compiled gate measures, 1792 is what `ConvPlan`
        // would program.
        assert!(
            u64::from(conv::MAX_DEPTHWISE_CHANNELS)
                > u64::from(
                    conv_ceilings(Precision::Fp16, true, 3, 1)
                        .unwrap()
                        .in_channels
                )
        );
        assert!(admit_conv(&conv(int8(false), true, 3, 2, 1344, 1344)).is_ok());
        assert!(admit_conv(&conv(int8(true), true, 1, 1, 1344, 1344)).is_ok());
        assert!(admit_conv(&conv(int8(true), true, 1, 1, 1345, 1345)).is_err());
    }

    #[test]
    fn the_requantized_path_has_an_output_atom_floor() {
        // The one *lower* bound in the table: below one 16-channel output
        // atom the requantized path is untested, and MobileNetV2's Cout 24
        // projection is what put a floor here in the first place.
        assert!(admit_conv(&conv(int8(true), false, 1, 1, 64, 16)).is_ok());
        let error =
            admit_conv(&conv(int8(true), false, 1, 1, 64, 15)).expect_err("below one output atom");
        assert_eq!(error.code(), PlanErrorCode::UnvalidatedConfiguration);
        assert!(error.message().contains("floor"), "{error}");
        // Neither the accumulator path nor depthwise has the floor.
        assert!(admit_conv(&conv(int8(false), false, 1, 1, 64, 8)).is_ok());
        assert!(admit_conv(&conv(int8(true), true, 3, 1, 8, 8)).is_ok());
    }

    #[test]
    fn a_depthwise_channel_multiplier_is_unsupported_not_unvalidated() {
        let error = admit_conv(&conv(Precision::Fp16, true, 3, 1, 64, 128))
            .expect_err("Cout != Cin on a depthwise op");
        assert_eq!(error.code(), PlanErrorCode::UnsupportedSemantics);
    }

    #[test]
    fn kernels_and_precisions_without_a_matcher_are_refused() {
        assert!(admit_conv(&conv(Precision::Fp16, false, 5, 1, 64, 64)).is_err());
        assert!(admit_conv(&conv(Precision::Bf16, false, 1, 1, 64, 64)).is_err());
        assert!(admit_conv(&conv(Precision::Tf32, false, 1, 1, 64, 64)).is_err());
        let mut nonsquare = conv(Precision::Fp16, false, 3, 1, 64, 64);
        nonsquare.kernel_width = 1;
        assert!(admit_conv(&nonsquare).is_err());
    }

    #[test]
    fn zero_channels_are_invalid_rather_than_unvalidated() {
        let error = admit_conv(&conv(Precision::Fp16, false, 1, 1, 0, 64))
            .expect_err("a zero channel count");
        assert_eq!(error.code(), PlanErrorCode::InvalidShape);
    }

    #[test]
    fn matmul_bounds_m_by_the_row_field_and_k_n_by_the_channel_ceilings() {
        let matmul = |m, k, n| {
            admit_matmul(&MatmulAdmission {
                precision: Precision::Fp16,
                m,
                k,
                n,
            })
        };
        // M at the raised ceiling, with K and N at theirs.
        assert!(matmul(4096, 4096, 4096).is_ok());
        // The old ceiling is still inside, and a transformer prefill just
        // past it -- the shape the raise was for -- is now admitted.
        assert!(matmul(2047, 3584, 3584).is_ok());
        assert!(matmul(2048, 64, 64).is_ok());
        assert!(matmul(4097, 64, 64).is_err());
        assert!(matmul(64, 4097, 64).is_err());
        assert!(matmul(64, 64, 4097).is_err());
        assert_eq!(
            matmul(0, 64, 64).expect_err("zero M").code(),
            PlanErrorCode::InvalidShape
        );
        assert!(
            admit_matmul(&MatmulAdmission {
                precision: int8(false),
                m: 64,
                k: 64,
                n: 64
            })
            .is_err()
        );
    }

    #[test]
    fn every_admitted_class_has_ceilings_and_the_reverse() {
        // A table-shaped guard: if a class is admitted by `admit_conv` it
        // must have an entry, and an entry must be reachable.
        for precision in [Precision::Fp16, int8(false), int8(true)] {
            for depthwise in [false, true] {
                for kernel in [1u64, 3] {
                    for stride in 1..=MAX_ADMITTED_STRIDE {
                        let ceilings = conv_ceilings(precision, depthwise, kernel, stride)
                            .expect("every (precision, depthwise, kernel, stride) has an entry");
                        assert!(ceilings.in_channels > 0 && ceilings.out_channels > 0);
                        assert!(ceilings.min_out_channels <= ceilings.out_channels);
                        let cin = u64::from(ceilings.in_channels);
                        let cout = if depthwise {
                            cin
                        } else {
                            u64::from(ceilings.out_channels)
                        };
                        assert!(
                            admit_conv(&conv(precision, depthwise, kernel, stride, cin, cout))
                                .is_ok(),
                            "{precision:?} depthwise={depthwise} k{kernel} s{stride}"
                        );
                    }
                }
            }
        }
    }
}
