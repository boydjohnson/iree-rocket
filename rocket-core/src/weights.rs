//! The CNA's blocked coefficient layout, and the packers that produce it
//! from IREE's logical HWCF filter.
//!
//! Moved here from `iree-rocket-hal/src/rocket/tensor_layout.rs` on
//! 2026-09-10 (COMPILER_ROADMAP.md 6.3) so the compiler plugin can pack a
//! constant filter at compile time through `rocket-plan-ffi` with the *same*
//! function the driver packs with at dispatch time -- "move, do not copy",
//! because a second spelling of this layout is what produced the depthwise
//! tap-major bug. The HAL re-exports every item under its old path; the
//! byte-level tests stay beside the HAL's activation packers.
//!
//! [`WeightPlan`] is the one new thing: the packer *selection* the driver's
//! `apply_ops` used to make inline -- tap-major for depthwise, the affine
//! int8 packer when the rung carries a zero point, the plain packer
//! otherwise -- as a value both sides construct from the shape, so the
//! compiler cannot pick a different packer than the runtime would have.

use crate::{
    conv::{FEATURE_ATOM_BYTES, Kernels, Shape},
    error::{PlanError, PlanErrorCode},
    fc,
};

/// Physical byte width of one NC1HWC2 inner-channel block, as the packers
/// count it.
const FEATURE_ATOMIC_BYTES: usize = FEATURE_ATOM_BYTES as usize;

/// Physical byte width of one output-kernel coefficient atom.
///
/// The convolution kernel group is 32 lanes for int8 and 16 lanes for
/// fp16, so both occupy 32 bytes.
pub const WEIGHT_ATOMIC_BYTES: usize = 32;

/// Number of input channels in one coefficient group.
///
/// Unlike the output-kernel group, this remains 32 channels for fp16.
/// A C32-to-C16 fp16 hardware probe distinguished this from a 16-channel
/// interpretation exactly: the latter split each logical output across two
/// hardware output kernels.
pub const WEIGHT_INPUT_GROUP_CHANNELS: usize = 32;

/// Input channels in one coefficient group for a **4-byte** element.
///
/// The group is 32 channels at every width from int4 to int16, and halves
/// at four bytes -- the one exception, recorded for the matmul weight tile
/// in `../rockchip-npu-notes/encodings/tile-layouts.md` (`tf32` is
/// `(N/16, K/16, 16, 16)` against fp16's `(N/16, K/32, 16, 32)`), which
/// keeps the group a constant 1024 bytes rather than a constant channel
/// count.
///
/// This is exactly the trap the notes warn about twice: at a shape with one
/// input group the two groupings produce identical bytes, so a `Cin` at or
/// below the candidate group cannot test it. tf32 coefficient tests need
/// `Cin >= 32`.
pub const TF32_WEIGHT_INPUT_GROUP_CHANNELS: usize = 16;

/// Output kernels in one coefficient block for a **4-byte** element.
///
/// The 32-byte coefficient atom holds 16 kernels at fp16, 32 at int8 and 64
/// at int4 -- the element width, straight through. At four bytes it does
/// *not* halve again to 8: the N-group stays 16 and the K-group halves
/// instead, keeping the 1024-byte tile
/// (`../rockchip-npu-notes/encodings/tile-layouts.md`).
pub const TF32_WEIGHT_OUTPUT_BLOCK_CHANNELS: usize = 16;

/// Physical byte width of one depthwise coefficient group.
///
/// The depthwise DPU serializes coefficients tap-major within a fixed 64-byte
/// group, so the group holds a precision-dependent number of *channels*: 32
/// at fp16 and 64 at int8. This is deliberately a byte width rather than a
/// channel count -- an int8-specific channel constant read naturally but
/// invited the wrong value, and a wrong one is close to invisible (a
/// uniform-weight probe cannot see a tap permutation at all).
///
/// Confirmed on RK3588 by delta-function probe: with a single nonzero tap,
/// output channels 0..15, 16..31, 32..47 and 48..63 came back shifted by taps
/// 0, 2, 4 and 6 respectively under the previous 16-channel rule, which
/// solves to an address function of `tap * 64 + channel` -- one contiguous
/// run of all 64 int8 channels per tap.
const DEPTHWISE_GROUP_BYTES: usize = 64;

/// Returns the storage needed for an uncompressed Rocket convolution filter.
///
/// The padding matches [`crate::conv::Shape::weight_channels`] and
/// [`crate::conv::Shape::programmed_kernels`]. FP16 pads input
/// channels to 8-channel atoms, with an atom count one short of a multiple of
/// four rounded up; its output kernel count remains exact. Int8 pads input
/// channels to 16-channel atoms and output kernels to an even count.
pub fn rocket_weight_storage_size(
    filter_height: usize,
    filter_width: usize,
    input_channels: usize,
    output_channels: usize,
    element_size: usize,
) -> Result<usize, &'static str> {
    if !matches!(element_size, 1 | 2 | 4) {
        return Err("invalid Rocket convolution filter shape");
    }
    rocket_weight_storage_size_bits(
        filter_height,
        filter_width,
        input_channels,
        output_channels,
        element_size * 8,
    )
}

/// [`rocket_weight_storage_size`] in element *bits*, which is what int4
/// needs: half a byte per coefficient, two per byte.
///
/// Every rule here follows the element width. The quad-atom input bump is
/// the 2-byte family's 16-kernel weight group (fp16, bf16, int16); the
/// even-kernel rounding is what the narrower groups take.
pub fn rocket_weight_storage_size_bits(
    filter_height: usize,
    filter_width: usize,
    input_channels: usize,
    output_channels: usize,
    element_bits: usize,
) -> Result<usize, &'static str> {
    let layout = WeightLayout::new(input_channels, output_channels, element_bits)?;
    if filter_height == 0 || filter_width == 0 {
        return Err("invalid Rocket convolution filter shape");
    }
    filter_height
        .checked_mul(filter_width)
        .and_then(|value| value.checked_mul(layout.padded_input_channels))
        .and_then(|value| value.checked_mul(layout.programmed_output_channels))
        .and_then(|value| value.checked_mul(element_bits))
        .map(|bits| bits / 8)
        .ok_or("Rocket convolution filter storage size overflows usize")
}

/// The blocked coefficient order's dimensions, shared by every element
/// width.
///
/// Splitting this out is what lets the nibble packer reuse the byte
/// packer's loop nest instead of restating it: the *order* is identical
/// across widths and only the store differs.
struct WeightLayout {
    output_block_channels: usize,
    input_group_channels: usize,
    padded_input_channels: usize,
    programmed_output_channels: usize,
    input_groups: usize,
    output_blocks: usize,
}

impl WeightLayout {
    fn new(
        input_channels: usize,
        output_channels: usize,
        element_bits: usize,
    ) -> Result<WeightLayout, &'static str> {
        if input_channels == 0 || output_channels == 0 || !matches!(element_bits, 4 | 8 | 16 | 32) {
            return Err("invalid Rocket convolution filter shape");
        }
        let channels_per_atom = FEATURE_ATOMIC_BYTES * 8 / element_bits;
        let input_atoms = input_channels.div_ceil(channels_per_atom);
        // The 3-mod-4 atom bump belongs to the 2-byte family alone; see
        // `conv::Shape::weight_channels`.
        let padded_input_atoms = if element_bits == 16 && input_atoms % 4 == 3 {
            input_atoms + 1
        } else {
            input_atoms
        };
        let padded_input_channels = padded_input_atoms * channels_per_atom;
        let programmed_output_channels = if element_bits < 16 {
            output_channels.next_multiple_of(2)
        } else {
            output_channels
        };
        // The coefficient *tile* is a constant 1024 bytes at every width --
        // `(N-group) * (K-group) * element bytes` -- which is what fixes
        // both groups. Below four bytes the K-group is pinned at 32
        // channels and the N-group absorbs the width, so the N-group is
        // `WEIGHT_ATOMIC_BYTES * 8 / element_bits`: 16 kernels at fp16, 32
        // at int8, 64 at int4. At four bytes that formula would give 8, and
        // the hardware instead keeps the N-group at 16 and halves the
        // K-group to 16. Both halves are from
        // `../rockchip-npu-notes/encodings/tile-layouts.md` and both are
        // needed: with only the K-group halved, tf32 computes correct
        // values under a uniform-coefficient pattern and wrong ones under
        // any pattern that varies with the output channel.
        let (output_block_channels, input_group_channels) = if element_bits == 32 {
            (
                TF32_WEIGHT_OUTPUT_BLOCK_CHANNELS,
                TF32_WEIGHT_INPUT_GROUP_CHANNELS,
            )
        } else {
            (
                WEIGHT_ATOMIC_BYTES * 8 / element_bits,
                WEIGHT_INPUT_GROUP_CHANNELS,
            )
        };
        Ok(WeightLayout {
            output_block_channels,
            input_group_channels,
            padded_input_channels,
            programmed_output_channels,
            input_groups: padded_input_channels.div_ceil(input_group_channels),
            output_blocks: programmed_output_channels.div_ceil(output_block_channels),
        })
    }

    /// Visits every destination coefficient slot in physical order, with the
    /// logical HWCF element it carries.
    ///
    /// `source` is `None` for a lane the filter does not reach -- input
    /// channel padding, or an output kernel past the logical count.
    fn visit_slots(
        &self,
        filter_height: usize,
        filter_width: usize,
        input_channels: usize,
        output_channels: usize,
        mut visit: impl FnMut(usize, Option<usize>, usize),
    ) {
        let mut slot = 0;
        for output_block in 0..self.output_blocks {
            for input_group in 0..self.input_groups {
                for filter_y in 0..filter_height {
                    for filter_x in 0..filter_width {
                        for output_lane in 0..self.output_block_channels {
                            let output_channel =
                                output_block * self.output_block_channels + output_lane;
                            if output_channel >= self.programmed_output_channels {
                                continue;
                            }
                            for input_lane in 0..self.input_group_channels {
                                let input_channel =
                                    input_group * self.input_group_channels + input_lane;
                                if input_channel >= self.padded_input_channels {
                                    continue;
                                }
                                let source = (output_channel < output_channels
                                    && input_channel < input_channels)
                                    .then(|| {
                                        (((filter_y * filter_width + filter_x) * input_channels
                                            + input_channel)
                                            * output_channels)
                                            + output_channel
                                    });
                                visit(slot, source, output_channel);
                                slot += 1;
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Packs a logical HWCF filter into the RK3588 CNA coefficient order.
///
/// The physical nesting is:
///
/// `output_block -> input_group -> filter_y -> filter_x ->
/// output_lane -> input_lane`
///
/// An output block is one 32-byte weight atom: 32 kernels for int8 or 16
/// kernels for fp16. Input groups remain 32 channels for both precisions,
/// with a partial final group when the register-programmed input channel
/// count is not divisible by 32. Input channels and output kernels use the
/// precision-dependent padding documented by
/// [`rocket_weight_storage_size`].
pub fn pack_hwcf_to_rocket_weights(
    dense: &[u8],
    filter_height: usize,
    filter_width: usize,
    input_channels: usize,
    output_channels: usize,
    element_size: usize,
    packed: &mut [u8],
) -> Result<usize, &'static str> {
    pack_hwcf_to_rocket_weights_impl(
        dense,
        filter_height,
        filter_width,
        input_channels,
        output_channels,
        output_channels,
        element_size,
        None,
        false,
        packed,
    )
}

/// [`pack_hwcf_to_rocket_weights`] forced down the per-coefficient
/// reference path, whatever the element width.
///
/// Same bytes, about 7x slower: this is the arm that drives
/// [`WeightLayout::visit_slots`] one coefficient at a time. It exists so the
/// two arms can be measured against each other inside one binary -- the
/// board's rate harness
/// (`iree-rocket-hal/tests/weight_pack_rate_hw.rs`) quotes both -- and so
/// the equivalence sweep has a reference that is the real fallback code
/// rather than a restatement of the layout. Production callers want
/// [`pack_hwcf_to_rocket_weights`], which routes.
pub fn pack_hwcf_to_rocket_weights_reference(
    dense: &[u8],
    filter_height: usize,
    filter_width: usize,
    input_channels: usize,
    output_channels: usize,
    element_size: usize,
    packed: &mut [u8],
) -> Result<usize, &'static str> {
    pack_hwcf_to_rocket_weights_impl(
        dense,
        filter_height,
        filter_width,
        input_channels,
        output_channels,
        output_channels,
        element_size,
        None,
        true,
        packed,
    )
}

/// Packs HWCF coefficients for a *wider* programmed output-channel count
/// than the filter logically has, zero-filling the surplus channels.
///
/// This is the coefficient half of
/// [`crate::conv::Shape::parity_padded_out_channels`]: the RK3588
/// DPU only commits accumulator output in whole 256-byte units, so a shape
/// whose output width and block count are both odd has to be programmed with
/// more output channels than it needs. The surplus channels must compute
/// zero, and their results are then discarded on the way back out.
///
/// `padded_output_channels` must be at least `output_channels`. Passing them
/// equal is exactly [`pack_hwcf_to_rocket_weights`].
// Each argument is an independent tensor-shape dimension; a struct wrapper
// would just move the same fields into a constructor.
#[allow(clippy::too_many_arguments)]
pub fn pack_hwcf_to_rocket_weights_padded(
    dense: &[u8],
    filter_height: usize,
    filter_width: usize,
    input_channels: usize,
    output_channels: usize,
    padded_output_channels: usize,
    element_size: usize,
    packed: &mut [u8],
) -> Result<usize, &'static str> {
    pack_hwcf_to_rocket_weights_impl(
        dense,
        filter_height,
        filter_width,
        input_channels,
        output_channels,
        padded_output_channels,
        element_size,
        None,
        false,
        packed,
    )
}

/// Packs an int4 HWCF filter into the CNA coefficient order, two
/// coefficients to a byte.
///
/// `dense` carries one logical coefficient per `i8`, each in `-8..=7`, in
/// the same HWCF order [`pack_hwcf_to_rocket_weights`] takes. The physical
/// order is identical to every other width -- the same
/// output_block/input_group/tap/lane nesting -- so only the store changes:
/// consecutive coefficient slots share a byte, the even slot in the low
/// nibble.
///
/// The nibble order is *not* a free choice and is not swapped:
/// `../rockchip-npu-notes/encodings/tile-layouts.md` records int4 as packed
/// low-nibble-first with `HILO = 0`.
///
/// The N-group falls out of the shared 32-byte coefficient atom rather than
/// being a special case: at half a byte per element it holds **64** kernels
/// against int8's 32 and fp16's 16. That is the int4 trap the notes call
/// out -- an int4 filter packed with int8's 32-kernel group coincides with
/// the correct one at a single input group and diverges past it -- so a
/// meaningful test needs `Cin` above 32.
pub fn pack_hwcf_to_rocket_weights_int4(
    dense: &[i8],
    filter_height: usize,
    filter_width: usize,
    input_channels: usize,
    output_channels: usize,
    packed: &mut [u8],
) -> Result<usize, &'static str> {
    pack_hwcf_to_rocket_weights_int4_padded(
        dense,
        filter_height,
        filter_width,
        input_channels,
        output_channels,
        output_channels,
        packed,
    )
}

/// [`pack_hwcf_to_rocket_weights_int4`] with a wider programmed output
/// channel count than the filter logically has, zero-filling the surplus.
pub fn pack_hwcf_to_rocket_weights_int4_padded(
    dense: &[i8],
    filter_height: usize,
    filter_width: usize,
    input_channels: usize,
    output_channels: usize,
    padded_output_channels: usize,
    packed: &mut [u8],
) -> Result<usize, &'static str> {
    if padded_output_channels < output_channels {
        return Err("padded output channel count is smaller than the logical one");
    }
    let dense_len = filter_height
        .checked_mul(filter_width)
        .and_then(|value| value.checked_mul(input_channels))
        .and_then(|value| value.checked_mul(output_channels))
        .ok_or("dense HWCF storage size overflows usize")?;
    if dense.len() < dense_len {
        return Err("dense HWCF filter is smaller than its declared shape");
    }
    if dense[..dense_len]
        .iter()
        .any(|&value| !(-8..=7).contains(&value))
    {
        return Err("int4 coefficient is outside -8..=7");
    }
    let packed_len = rocket_weight_storage_size_bits(
        filter_height,
        filter_width,
        input_channels,
        padded_output_channels,
        4,
    )?;
    if packed.len() < packed_len {
        return Err("Rocket weight destination is smaller than its declared shape");
    }
    packed[..packed_len].fill(0);

    let layout = WeightLayout::new(input_channels, padded_output_channels, 4)?;
    layout.visit_slots(
        filter_height,
        filter_width,
        input_channels,
        output_channels,
        |slot, source, _| {
            let Some(source) = source else { return };
            let nibble = (dense[source] as u8) & 0xf;
            if slot.is_multiple_of(2) {
                packed[slot / 2] = (packed[slot / 2] & 0xf0) | nibble;
            } else {
                packed[slot / 2] = (packed[slot / 2] & 0x0f) | (nibble << 4);
            }
        },
    );
    Ok(packed_len)
}

/// Packs a dense NHWC int4 feature map into NC1HWC2, two channels a byte.
///
/// `dense` carries one logical value per `i8`, each in `-8..=7`, in NHWC
/// order. The 16-byte feature atom holds **32** int4 channels, so a pixel
/// occupies `ceil(Cin / 32)` atoms exactly as it does at every other width.
///
/// `input_channels` must be a whole multiple of two: a surface boundary in
/// the middle of a byte has no meaning, and every int4 channel count the
/// convolution builder programs is a whole atom anyway.
pub fn pack_nhwc_to_nc1hwc2_int4(
    dense: &[i8],
    pixel_count: usize,
    input_channels: usize,
    packed: &mut [u8],
) -> Result<usize, &'static str> {
    const CHANNELS_PER_ATOM: usize = FEATURE_ATOMIC_BYTES * 2;
    if input_channels == 0 || !input_channels.is_multiple_of(2) {
        return Err("int4 channel count must be a nonzero multiple of two");
    }
    if dense.len() < pixel_count * input_channels {
        return Err("dense NHWC feature map is smaller than its declared shape");
    }
    if dense[..pixel_count * input_channels]
        .iter()
        .any(|&value| !(-8..=7).contains(&value))
    {
        return Err("int4 feature value is outside -8..=7");
    }
    let surfaces = input_channels.div_ceil(CHANNELS_PER_ATOM);
    let written = surfaces * pixel_count * FEATURE_ATOMIC_BYTES;
    if packed.len() < written {
        return Err("NC1HWC2 destination is smaller than its declared shape");
    }
    packed[..written].fill(0);
    for pixel in 0..pixel_count {
        for channel in 0..input_channels {
            let surface = channel / CHANNELS_PER_ATOM;
            let lane = channel % CHANNELS_PER_ATOM;
            let offset = (surface * pixel_count + pixel) * FEATURE_ATOMIC_BYTES + lane / 2;
            let nibble = (dense[pixel * input_channels + channel] as u8) & 0xf;
            if lane.is_multiple_of(2) {
                packed[offset] = (packed[offset] & 0xf0) | nibble;
            } else {
                packed[offset] = (packed[offset] & 0x0f) | (nibble << 4);
            }
        }
    }
    Ok(written)
}

/// Packs a quantized int8 HWCF filter and fills physical input-channel
/// padding with each output channel's weight zero point.
///
/// The live bytes in `dense` are raw quantized coefficients and are copied
/// unchanged. A padded input lane participates in the hardware dot product,
/// so its neutral value is `weight_zero_points[output_channel]`, not
/// necessarily zero. The matching BS constant is `-weight_zero_point`; see
/// [`crate::conv::BsEntry::constant`].
pub fn pack_hwcf_to_rocket_weights_affine_i8(
    dense: &[u8],
    filter_height: usize,
    filter_width: usize,
    input_channels: usize,
    output_channels: usize,
    weight_zero_points: &[i8],
    packed: &mut [u8],
) -> Result<usize, &'static str> {
    if weight_zero_points.len() != output_channels {
        return Err("int8 weight zero-point count does not match output channels");
    }
    pack_hwcf_to_rocket_weights_impl(
        dense,
        filter_height,
        filter_width,
        input_channels,
        output_channels,
        output_channels,
        1,
        Some(weight_zero_points),
        false,
        packed,
    )
}

#[allow(clippy::too_many_arguments)]
fn pack_hwcf_to_rocket_weights_impl(
    dense: &[u8],
    filter_height: usize,
    filter_width: usize,
    input_channels: usize,
    output_channels: usize,
    padded_output_channels: usize,
    element_size: usize,
    weight_zero_points: Option<&[i8]>,
    force_reference: bool,
    packed: &mut [u8],
) -> Result<usize, &'static str> {
    if padded_output_channels < output_channels {
        return Err("padded output channel count is smaller than the logical one");
    }
    let dense_len = filter_height
        .checked_mul(filter_width)
        .and_then(|value| value.checked_mul(input_channels))
        .and_then(|value| value.checked_mul(output_channels))
        .and_then(|value| value.checked_mul(element_size))
        .ok_or("dense HWCF storage size overflows usize")?;
    if dense.len() < dense_len {
        return Err("dense HWCF filter is smaller than its declared shape");
    }

    let packed_len = rocket_weight_storage_size(
        filter_height,
        filter_width,
        input_channels,
        padded_output_channels,
        element_size,
    )?;
    if packed.len() < packed_len {
        return Err("Rocket weight destination is smaller than its declared shape");
    }

    let layout = WeightLayout::new(input_channels, padded_output_channels, element_size * 8)?;

    // The strided fast path produces the same bytes as `visit_slots` and is
    // about 7x quicker, but only for a *symmetric* 1- or 2-byte element: a
    // nonzero weight zero point writes a per-output-channel value into the
    // padding lanes, which is not a permutation of the dense filter at all,
    // and the 4-byte group geometry it would need is untested. All-zero
    // zero points are symmetric by definition -- the neutral value and the
    // zero fill coincide -- so the int8 convolution rungs that carry a zero
    // point of 0 take the fast path too.
    let symmetric = weight_zero_points.is_none_or(|points| points.iter().all(|&point| point == 0));
    if symmetric && !force_reference && matches!(element_size, 1 | 2) {
        let fill = layout.padded_input_channels != input_channels
            || layout.programmed_output_channels != output_channels;
        if fill {
            packed[..packed_len].fill(0);
        }
        if element_size == 1 {
            pack_dense_striped::<1>(
                dense,
                filter_height * filter_width,
                input_channels,
                output_channels,
                &layout,
                packed,
            );
        } else {
            pack_dense_striped::<2>(
                dense,
                filter_height * filter_width,
                input_channels,
                output_channels,
                &layout,
                packed,
            );
        }
        return Ok(packed_len);
    }

    pack_dense_visited(
        dense,
        filter_height,
        filter_width,
        input_channels,
        output_channels,
        element_size,
        weight_zero_points,
        &layout,
        &mut packed[..packed_len],
    );

    Ok(packed_len)
}

/// The original per-coefficient packer: one closure call per destination
/// slot, driving [`WeightLayout::visit_slots`].
///
/// This is the fallback and the reference. It is what every hardware
/// validation in the tree was taken against, it is the only arm that can
/// write a nonzero weight zero point into a padding lane, and it is the only
/// arm that knows tf32's different group geometry -- so it stays, and
/// `mod tests`'s sweep compares the fast path against it rather than against
/// a restatement of the layout.
#[allow(clippy::too_many_arguments)]
fn pack_dense_visited(
    dense: &[u8],
    filter_height: usize,
    filter_width: usize,
    input_channels: usize,
    output_channels: usize,
    element_size: usize,
    weight_zero_points: Option<&[i8]>,
    layout: &WeightLayout,
    packed: &mut [u8],
) {
    packed.fill(0);
    layout.visit_slots(
        filter_height,
        filter_width,
        input_channels,
        output_channels,
        |slot, source, output_channel| {
            let dst_offset = slot * element_size;
            match source {
                Some(source) => {
                    let src_offset = source * element_size;
                    packed[dst_offset..dst_offset + element_size]
                        .copy_from_slice(&dense[src_offset..src_offset + element_size]);
                }
                None => {
                    if let Some(zero_points) = weight_zero_points
                        && output_channel < output_channels
                    {
                        packed[dst_offset] = zero_points[output_channel] as u8;
                    }
                }
            }
        },
    );
}

/// [`WeightLayout::visit_slots`]'s coefficient order, written as strided
/// runs instead of per-element visits.
///
/// Why it is faster is the whole point, so it is worth stating: for a fixed
/// `(output block, input group, filter position, output lane)` the
/// destination coefficients are **contiguous** -- one run of
/// `input_group_channels` slots -- and the dense elements that fill them sit
/// a **constant stride** of `output_channels` apart, because HWCF puts the
/// output channel last. So the innermost loop becomes one zipped pair of
/// fixed-size chunk iterators over a pair of slices whose bounds were
/// checked once, rather than a closure invoked once per coefficient that
/// recomputes two multiplies and bounds-checks a one-element
/// `copy_from_slice`. On the board that is 214 MB/s against about 1.6 GB/s
/// (SOFTMAX.md section 11.3); using `as_chunks` rather than `chunks_exact`
/// is itself worth 1.6x of it, which is a fair measure of how much of the
/// original cost was bounds checking.
///
/// Callers must have zero-filled `packed` unless the shape has no padding
/// lane at all: lanes past `input_channels`, and output kernels past
/// `output_channels` up to the programmed count, are not written here.
///
/// `E` is the element size in bytes. Only 1 and 2 are instantiated, which
/// is int8 and the whole 2-byte family; see
/// [`pack_hwcf_to_rocket_weights_impl`] for why the others stay on
/// `visit_slots`.
fn pack_dense_striped<const E: usize>(
    dense: &[u8],
    taps: usize,
    input_channels: usize,
    output_channels: usize,
    layout: &WeightLayout,
    packed: &mut [u8],
) {
    let mut slot = 0usize;
    for output_block in 0..layout.output_blocks {
        for input_group in 0..layout.input_groups {
            let group_base = input_group * layout.input_group_channels;
            // Slots this group occupies per output lane, and how many of
            // them the filter actually reaches.
            let lanes = layout
                .input_group_channels
                .min(layout.padded_input_channels - group_base);
            let live = lanes.min(input_channels.saturating_sub(group_base));
            for tap in 0..taps {
                for output_lane in 0..layout.output_block_channels {
                    let output_channel = output_block * layout.output_block_channels + output_lane;
                    if output_channel >= layout.programmed_output_channels {
                        // Past the programmed count: no slot at all, which
                        // is why `slot` does not advance here.
                        continue;
                    }
                    if output_channel < output_channels && live > 0 {
                        let source_base = ((tap * input_channels + group_base) * output_channels
                            + output_channel)
                            * E;
                        let destination = &mut packed[slot * E..(slot + live) * E];
                        let source = &dense[source_base..];
                        for (out, value) in destination
                            .as_chunks_mut::<E>()
                            .0
                            .iter_mut()
                            .zip(source.as_chunks::<E>().0.iter().step_by(output_channels))
                        {
                            *out = *value;
                        }
                    }
                    slot += lanes;
                }
            }
        }
    }
}

/// Packs a depthwise filter into the RK3588 CNA coefficient order.
///
/// A depthwise filter is `[channels][filter_height][filter_width]` -- one
/// `kh x kw` kernel per input channel, with no `(input, output)` pairing at
/// all. The hardware wants it **tap-major within a channel group**: channels
/// FP16 groups are 32 channels; int8 groups are 16 channels. Every group's
/// own taps sit contiguously before the next group starts. This differs from
/// the dense coefficient grouping, which is 32 channels for both precisions.
/// before the next group starts:
///
/// ```text
/// slot = group_base(channel)
///      + (ky * filter_width + kx) * group_width(channel)
///      + (channel % group_width)
/// ```
///
/// where `group_base` is the running element offset of that channel's group,
/// and `group_width` is the precision-specific group width except a final,
/// shorter one when `padded_channels` isn't a whole multiple of it.
/// This is the transpose of how torch and ONNX store a depthwise filter, and
/// nothing like `pack_hwcf_to_rocket_weights`'s own blocked dense order.
///
/// **This grouping was missed the first time.** The original hardware probe
/// (`tests/conv_depthwise_probe_hw.rs`, one-hot slot-by-slot at Cin 8 and
/// 12) never exceeded one 32-channel group, so a single global stride --
/// `slot = (ky*kw+kx)*padded_channels+channel` -- looked equivalent and
/// shipped instead; every subsequent depthwise validation
/// (`tests/conv_phase1_validation_hw.rs`, the nine-point channel-count
/// ladder in DESIGN_NOTES.md) also stayed at or below 128 channels without
/// ever probing the packed buffer's own internal layout, only the register
/// program's declared total byte count, which the group boundary doesn't
/// change. It was found by routing a real depthwise dispatch through the
/// actual compiled MLIR/driver path for the first time (transform.0.mlir's
/// `@match_dynamic_depthwise_conv2d`) at Cin 128, and pinned down exactly by
/// three follow-up probes -- distinct known values on every tap of one
/// channel, summed on real hardware -- at Cin 128 (4 exact groups), 256 (8
/// exact groups), and 144 (4 full groups plus a genuine 16-wide tail group).
/// All three matched this formula bit-for-bit and none matched the old flat
/// one. Only fp16 has been checked this way; int8 reuses
/// The int8 width was confirmed by the raw accumulator probe: packing it as
/// fp16's 32-channel groups makes channel 0 sum taps 1,1,2,2,...,5 instead of
/// 1..9. The dense path's grouping remains unchanged.
///
/// `padded_channels` is the count the register program's
/// `CNA_WEIGHT_SIZE0.weight_bytes` is sized from -- [`Shape::weight_bytes`]
/// divided by `kh * kw * element_size` -- not the real channel count. The
/// two differ whenever the channel count is not a whole CBUF atom group;
/// padding slots are left zero and contribute nothing. Group boundaries are
/// computed from this padded count, not the raw channel count, so a
/// trailing padding slot (if any) stays inside the last real group instead
/// of opening an all-padding group of its own.
///
/// [`Shape::weight_bytes`]: crate::conv::Shape::weight_bytes
pub fn pack_depthwise_to_rocket_weights(
    dense: &[u8],
    filter_height: usize,
    filter_width: usize,
    channels: usize,
    padded_channels: usize,
    element_size: usize,
    packed: &mut [u8],
) -> Result<usize, &'static str> {
    if filter_height == 0
        || filter_width == 0
        || channels == 0
        || padded_channels < channels
        // The group is a byte width divided by the element size, so a
        // sub-byte element cannot be expressed here at all: int4 would want
        // 128 channels per group and this signature can only say 64. A
        // 4-byte element is expressible but unmeasured. Both are refused
        // rather than silently mis-grouped -- the int8 bug this grouping was
        // written to fix stayed invisible under a uniform-weight probe.
        || !matches!(element_size, 1 | 2)
    {
        return Err("invalid Rocket depthwise filter shape");
    }

    let dense_len = filter_height
        .checked_mul(filter_width)
        .and_then(|value| value.checked_mul(channels))
        .and_then(|value| value.checked_mul(element_size))
        .ok_or("depthwise filter storage size overflows usize")?;
    if dense.len() < dense_len {
        return Err("dense depthwise filter is smaller than its declared shape");
    }

    let packed_len = filter_height
        .checked_mul(filter_width)
        .and_then(|value| value.checked_mul(padded_channels))
        .and_then(|value| value.checked_mul(element_size))
        .ok_or("packed depthwise filter storage size overflows usize")?;
    if packed.len() < packed_len {
        return Err("packed depthwise filter buffer is too small");
    }

    packed[..packed_len].fill(0);

    let group_channels = DEPTHWISE_GROUP_BYTES / element_size;
    let full_groups = padded_channels / group_channels;
    let tail_width = padded_channels - full_groups * group_channels;

    for channel in 0..channels {
        let group = channel / group_channels;
        let channel_in_group = channel % group_channels;
        let group_width = if group < full_groups {
            group_channels
        } else {
            tail_width
        };
        let group_base = group * filter_height * filter_width * group_channels;
        for ky in 0..filter_height {
            for kx in 0..filter_width {
                let from = ((channel * filter_height + ky) * filter_width + kx) * element_size;
                let to = (group_base + (ky * filter_width + kx) * group_width + channel_in_group)
                    * element_size;
                packed[to..to + element_size].copy_from_slice(&dense[from..from + element_size]);
            }
        }
    }
    Ok(packed_len)
}

/// One coefficient tensor as a dispatch programs it, and the packer it
/// takes. Built the same way by the driver (at dispatch) and the compiler
/// (at `rocket-pack-weights`), which is what makes a compile-time packed
/// filter byte-identical to what the driver would have produced.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeightPlan {
    pub shape: Shape,
    pub kernels: Kernels,
    /// The filter's zero point on the int8 rungs of a convolution, applied
    /// to every output channel (the wire carries one scalar); `None` packs
    /// symmetrically, which is what a matmul's `[K, N]` operand has always
    /// had and what the fp16 rungs need.
    pub weight_zero_point: Option<i8>,
}

impl WeightPlan {
    /// A convolution's filter: the zero point comes from the rung.
    pub fn for_conv(shape: Shape, kernels: Kernels) -> WeightPlan {
        WeightPlan {
            shape,
            kernels,
            weight_zero_point: shape
                .precision
                .quantization()
                .map(|q| q.weight_zero_point as i8),
        }
    }

    /// A matmul's `[K, N]` operand, through the 1x1 lowering `fc.rs` plans
    /// it as. Symmetric on every rung.
    pub fn for_matmul(shape: fc::Shape) -> Result<WeightPlan, PlanError> {
        Ok(WeightPlan {
            shape: shape.try_as_conv_shape()?,
            kernels: fc::KERNELS,
            weight_zero_point: None,
        })
    }

    /// Bytes one coefficient occupies. Sub-byte rungs have no packer here.
    pub fn element_bytes(&self) -> Result<usize, PlanError> {
        let bits = self.shape.precision.element_bits();
        if !bits.is_multiple_of(8) {
            return Err(PlanError::new(
                PlanErrorCode::UnsupportedSemantics,
                format!("{:?} coefficients are sub-byte", self.shape.precision),
            ));
        }
        Ok(bits as usize / 8)
    }

    /// Bytes the logical filter occupies in IREE's binding: `kh * kw * Cin *
    /// Cout` elements, or `kh * kw * Cin` for depthwise (one filter per
    /// input channel, no `Cout` factor).
    pub fn dense_bytes(&self) -> Result<usize, PlanError> {
        let [kh, kw] = self.kernels;
        let overflow = || {
            PlanError::new(
                PlanErrorCode::HardwareLimit,
                "dense coefficient byte count overflows usize",
            )
        };
        let elements = kh
            .checked_mul(kw)
            .and_then(|v| v.checked_mul(self.shape.in_channels as usize))
            .and_then(|v| {
                if self.shape.depthwise {
                    Some(v)
                } else {
                    v.checked_mul(self.shape.out_channels as usize)
                }
            })
            .ok_or_else(overflow)?;
        elements
            .checked_mul(self.element_bytes()?)
            .ok_or_else(overflow)
    }

    /// Output channels the register program carries, which the dense packer
    /// zero-fills up to.
    fn programmed_out_channels(&self) -> Result<usize, PlanError> {
        self.shape
            .parity_padded_shape(self.kernels)
            .map(|shape| shape.out_channels as usize)
            .map_err(|reason| PlanError::new(PlanErrorCode::InvalidShape, reason))
    }

    /// Bytes the packed coefficient stream occupies: what the driver sizes
    /// its scratch to, and what the compiler sizes the packed global to.
    pub fn packed_bytes(&self) -> Result<usize, PlanError> {
        if self.shape.depthwise {
            return Ok(self.shape.try_weight_bytes(self.kernels)? as usize);
        }
        let [kh, kw] = self.kernels;
        rocket_weight_storage_size(
            kh,
            kw,
            self.shape.in_channels as usize,
            self.programmed_out_channels()?,
            self.element_bytes()?,
        )
        .map_err(|reason| PlanError::new(PlanErrorCode::InvalidShape, reason))
    }

    /// Packs `dense` (exactly [`WeightPlan::dense_bytes`] long) into
    /// `packed` (at least [`WeightPlan::packed_bytes`] long), returning the
    /// bytes written. The selection is the driver's `apply_ops` arm, moved.
    pub fn pack(&self, dense: &[u8], packed: &mut [u8]) -> Result<usize, PlanError> {
        let dense_len = self.dense_bytes()?;
        let packed_len = self.packed_bytes()?;
        if dense.len() != dense_len {
            return Err(PlanError::new(
                PlanErrorCode::InvalidShape,
                format!(
                    "dense filter is {} bytes, the shape needs {dense_len}",
                    dense.len()
                ),
            ));
        }
        if packed.len() < packed_len {
            return Err(PlanError::new(
                PlanErrorCode::InvalidShape,
                format!(
                    "packed buffer is {} bytes, the layout needs {packed_len}",
                    packed.len()
                ),
            ));
        }
        let [kh, kw] = self.kernels;
        let cin = self.shape.in_channels as usize;
        let cout = self.shape.out_channels as usize;
        let element_bytes = self.element_bytes()?;
        let result = if self.shape.depthwise {
            pack_depthwise_to_rocket_weights(
                dense,
                kh,
                kw,
                cin,
                self.shape.depthwise_padded_channels() as usize,
                element_bytes,
                packed,
            )
        } else if let Some(zero_point) = self.weight_zero_point {
            let programmed = self.programmed_out_channels()?;
            if programmed > cout {
                if zero_point != 0 {
                    Err("programmed Cout padding requires symmetric accumulator weights")
                } else {
                    pack_hwcf_to_rocket_weights_padded(
                        dense,
                        kh,
                        kw,
                        cin,
                        cout,
                        programmed,
                        element_bytes,
                        packed,
                    )
                }
            } else {
                let zero_points = vec![zero_point; cout];
                pack_hwcf_to_rocket_weights_affine_i8(
                    dense,
                    kh,
                    kw,
                    cin,
                    cout,
                    &zero_points,
                    packed,
                )
            }
        } else {
            pack_hwcf_to_rocket_weights(dense, kh, kw, cin, cout, element_bytes, packed)
        };
        result.map_err(|reason| PlanError::new(PlanErrorCode::InvalidShape, reason))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conv::{self, Multiplier, Precision, Quantization};

    fn int8() -> Precision {
        Precision::Int8(Quantization {
            input_zero_point: 0,
            output_zero_point: 0,
            weight_zero_point: 3,
            input_scale: 1.0,
            weights_scale: 1.0,
            multiplier: Multiplier::try_from_ratio(1.0).unwrap(),
        })
    }

    #[test]
    fn the_plan_sizes_agree_with_the_shape_and_the_packer() {
        // Dense: the packer's own size; depthwise: the register-programmed
        // one. Both are what the driver sized its scratch to.
        for (cin, cout, k, precision) in [
            (64, 64, 1, Precision::Fp16),
            (24, 88, 3, Precision::Fp16),
            (3, 32, 3, Precision::Fp16),
            (64, 128, 1, int8()),
            (20, 40, 3, int8()),
        ] {
            let shape = conv::Shape::try_with_precision(16, 16, 1, cin, cout, precision).unwrap();
            let plan = WeightPlan::for_conv(shape, [k, k]);
            let elem = precision.element_bytes() as usize;
            assert_eq!(
                plan.packed_bytes().unwrap(),
                rocket_weight_storage_size(k, k, cin as usize, cout as usize, elem).unwrap()
            );
            assert_eq!(
                plan.dense_bytes().unwrap(),
                k * k * cin as usize * cout as usize * elem
            );
        }
        let dw = conv::Shape::try_with_precision(16, 16, 1, 96, 96, Precision::Fp16)
            .unwrap()
            .try_with_depthwise()
            .unwrap();
        let plan = WeightPlan::for_conv(dw, [3, 3]);
        assert_eq!(
            plan.packed_bytes().unwrap(),
            dw.weight_bytes([3, 3]) as usize
        );
        assert_eq!(plan.dense_bytes().unwrap(), 3 * 3 * 96 * 2);
    }

    #[test]
    fn the_plan_picks_the_packer_the_driver_did() {
        // fp16 dense: the plain packer, byte for byte.
        let shape = conv::Shape::try_with_precision(8, 8, 1, 24, 40, Precision::Fp16).unwrap();
        let plan = WeightPlan::for_conv(shape, [3, 3]);
        let dense: Vec<u8> = (0..plan.dense_bytes().unwrap())
            .map(|i| (i % 253 + 1) as u8)
            .collect();
        let mut packed = vec![0u8; plan.packed_bytes().unwrap()];
        let mut expected = vec![0u8; packed.len()];
        assert_eq!(plan.pack(&dense, &mut packed).unwrap(), packed.len());
        pack_hwcf_to_rocket_weights(&dense, 3, 3, 24, 40, 2, &mut expected).unwrap();
        assert_eq!(packed, expected);

        // int8 with a zero point: the affine packer, with the scalar
        // broadcast over every output channel, which fills the padding lanes
        // with 3 rather than 0.
        let shape = conv::Shape::try_with_precision(8, 8, 1, 20, 40, int8()).unwrap();
        let plan = WeightPlan::for_conv(shape, [1, 1]);
        assert_eq!(plan.weight_zero_point, Some(3));
        let dense: Vec<u8> = (0..plan.dense_bytes().unwrap())
            .map(|i| (i % 251 + 1) as u8)
            .collect();
        let mut packed = vec![0u8; plan.packed_bytes().unwrap()];
        let mut expected = vec![0u8; packed.len()];
        plan.pack(&dense, &mut packed).unwrap();
        pack_hwcf_to_rocket_weights_affine_i8(&dense, 1, 1, 20, 40, &[3; 40], &mut expected)
            .unwrap();
        assert_eq!(packed, expected);
        assert!(packed.contains(&3));

        // A matmul is symmetric whatever the rung.
        let plan = WeightPlan::for_matmul(fc::Shape::new(16, 64, 64, int8())).unwrap();
        assert_eq!(plan.weight_zero_point, None);
        assert_eq!(plan.kernels, fc::KERNELS);

        // Depthwise: tap-major.
        let dw = conv::Shape::try_with_precision(8, 8, 1, 32, 32, Precision::Fp16)
            .unwrap()
            .try_with_depthwise()
            .unwrap();
        let plan = WeightPlan::for_conv(dw, [3, 3]);
        let dense: Vec<u8> = (0..plan.dense_bytes().unwrap())
            .map(|i| (i % 250 + 1) as u8)
            .collect();
        let mut packed = vec![0u8; plan.packed_bytes().unwrap()];
        let mut expected = vec![0u8; packed.len()];
        plan.pack(&dense, &mut packed).unwrap();
        pack_depthwise_to_rocket_weights(
            &dense,
            3,
            3,
            32,
            dw.depthwise_padded_channels() as usize,
            2,
            &mut expected,
        )
        .unwrap();
        assert_eq!(packed, expected);
    }

    #[test]
    fn a_wrong_length_is_refused_not_misread() {
        let shape = conv::Shape::try_with_precision(8, 8, 1, 16, 16, Precision::Fp16).unwrap();
        let plan = WeightPlan::for_conv(shape, [1, 1]);
        let mut packed = vec![0u8; plan.packed_bytes().unwrap()];
        assert_eq!(
            plan.pack(&[0u8; 10], &mut packed).unwrap_err().code(),
            PlanErrorCode::InvalidShape
        );
        let dense = vec![0u8; plan.dense_bytes().unwrap()];
        assert_eq!(
            plan.pack(&dense, &mut packed[..8]).unwrap_err().code(),
            PlanErrorCode::InvalidShape
        );
    }

    /// The strided fast path and the per-coefficient reference must produce
    /// the same bytes at every shape the fast path claims.
    ///
    /// The sweep is not a sample: it is written to hit each edge the layout
    /// has, and it *asserts* that it hit them, because every one of them is
    /// an edge a shape sweep can silently miss (the depthwise grouping bug
    /// and the int4 N-group trap in this file's own doc comments were both
    /// missed exactly that way). The four, with the shapes that reach them:
    ///
    /// * a partial final input group -- `Cin` not a multiple of 32;
    /// * the 3-mod-4 input-atom bump, which belongs to the 2-byte family
    ///   alone -- `Cin` in 33..=48 or 97..=112 at two bytes;
    /// * a partial final output block -- `Cout` not a multiple of the
    ///   block's 16 (fp16) or 32 (int8) kernels;
    /// * `padded_output_channels` strictly greater than `output_channels`,
    ///   the parity-padded accumulator arm.
    ///
    /// Both element widths and kernels 1, 3, 5 and 7 run against all of it.
    /// Every case here has symmetric coefficients and a 1- or 2-byte
    /// element, which is exactly the routing condition, so each one does
    /// drive the fast path.
    #[test]
    fn the_striped_packer_is_byte_identical_to_the_visited_one() {
        let mut saw_partial_input_group = false;
        let mut saw_input_atom_bump = false;
        let mut saw_partial_output_block = false;
        let mut saw_output_padding = false;

        for element_size in [1usize, 2] {
            for [kh, kw] in [[1usize, 1], [3, 3], [5, 5], [7, 7], [1, 3], [7, 1]] {
                for cin in [1usize, 3, 8, 17, 20, 32, 33, 40, 48, 64, 97, 112, 129, 256] {
                    for cout in [1usize, 2, 15, 16, 17, 24, 32, 40, 64, 88] {
                        for extra in [0usize, 1, 2] {
                            let padded_cout = cout + extra;
                            let dense_len = kh * kw * cin * cout * element_size;
                            let dense: Vec<u8> = (0..dense_len)
                                .map(|i| ((i * 37 + i / 251) % 251 + 1) as u8)
                                .collect();
                            let packed_len =
                                rocket_weight_storage_size(kh, kw, cin, padded_cout, element_size)
                                    .unwrap();

                            let layout =
                                WeightLayout::new(cin, padded_cout, element_size * 8).unwrap();
                            saw_partial_input_group |= !layout
                                .padded_input_channels
                                .is_multiple_of(layout.input_group_channels);
                            saw_input_atom_bump |= layout.padded_input_channels
                                > cin.next_multiple_of(
                                    FEATURE_ATOMIC_BYTES * 8 / (element_size * 8),
                                );
                            saw_partial_output_block |= !layout
                                .programmed_output_channels
                                .is_multiple_of(layout.output_block_channels);
                            saw_output_padding |= layout.programmed_output_channels > cout;

                            let mut reference = vec![0xABu8; packed_len];
                            pack_dense_visited(
                                &dense,
                                kh,
                                kw,
                                cin,
                                cout,
                                element_size,
                                None,
                                &layout,
                                &mut reference,
                            );

                            let mut fast = vec![0xCDu8; packed_len];
                            assert_eq!(
                                pack_hwcf_to_rocket_weights_padded(
                                    &dense,
                                    kh,
                                    kw,
                                    cin,
                                    cout,
                                    padded_cout,
                                    element_size,
                                    &mut fast,
                                )
                                .unwrap(),
                                packed_len
                            );

                            assert!(
                                fast == reference,
                                "fast path differs at {kh}x{kw} Cin {cin} Cout {cout} \
                                 (padded {padded_cout}) element_size {element_size}: \
                                 first byte {:?}",
                                fast.iter().zip(reference.iter()).position(|(a, b)| a != b)
                            );
                        }
                    }
                }
            }
        }

        assert!(
            saw_partial_input_group,
            "sweep missed a partial input group"
        );
        assert!(saw_input_atom_bump, "sweep missed the 3-mod-4 atom bump");
        assert!(
            saw_partial_output_block,
            "sweep missed a partial output block"
        );
        assert!(saw_output_padding, "sweep missed programmed output padding");
    }

    /// An int8 rung whose weight zero point is 0 takes the fast path, and
    /// the affine packer's output is what it has to match: at a zero point
    /// of zero the neutral padding value and the zero fill coincide, which
    /// is the whole argument for routing it.
    #[test]
    fn a_zero_weight_zero_point_is_symmetric() {
        for (cin, cout, k) in [(20usize, 40usize, 1usize), (33, 17, 3), (3, 2, 7)] {
            let dense: Vec<u8> = (0..k * k * cin * cout)
                .map(|i| (i % 251 + 1) as u8)
                .collect();
            let packed_len = rocket_weight_storage_size(k, k, cin, cout, 1).unwrap();

            let mut affine = vec![0u8; packed_len];
            pack_hwcf_to_rocket_weights_affine_i8(
                &dense,
                k,
                k,
                cin,
                cout,
                &vec![0i8; cout],
                &mut affine,
            )
            .unwrap();

            let mut fast = vec![0u8; packed_len];
            pack_hwcf_to_rocket_weights(&dense, k, k, cin, cout, 1, &mut fast).unwrap();
            assert_eq!(
                fast, affine,
                "zero zero point differs at {k}x{k} {cin}->{cout}"
            );

            // And a nonzero one must *not* be the same bytes, or the
            // routing condition above is testing nothing.
            let mut nonzero = vec![0u8; packed_len];
            pack_hwcf_to_rocket_weights_affine_i8(
                &dense,
                k,
                k,
                cin,
                cout,
                &vec![3i8; cout],
                &mut nonzero,
            )
            .unwrap();
            if cin % 32 != 0 {
                assert_ne!(
                    nonzero, affine,
                    "a nonzero zero point must reach the padding"
                );
            }
        }
    }
}
