//! Host-side transforms between dense NHWC tensors and the NPU's
//! feature-atomic NC1HWC2 layout.
//!
//! The hardware's inner channel block is 16 bytes, not a fixed number of
//! elements: C2 is 16 for int8 and 8 for fp16. Each C1 surface contains
//! every HxW pixel before the next channel block begins.

/// Physical byte width of one NC1HWC2 inner-channel block.
pub const FEATURE_ATOMIC_BYTES: usize = 16;

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

/// Returns the NC1HWC2 storage required for `pixel_count` dense pixels.
///
/// `bytes_per_pixel` is the logical channel count times the element size.
/// The final C1 surface is padded to a complete 16-byte C2 block.
pub fn nc1hwc2_storage_size(
    pixel_count: usize,
    bytes_per_pixel: usize,
) -> Result<usize, &'static str> {
    if bytes_per_pixel == 0 {
        return Err("NC1HWC2 bytes per pixel must be nonzero");
    }
    let surface_count = bytes_per_pixel.div_ceil(FEATURE_ATOMIC_BYTES);
    pixel_count
        .checked_mul(surface_count)
        .and_then(|value| value.checked_mul(FEATURE_ATOMIC_BYTES))
        .ok_or("NC1HWC2 storage size overflows usize")
}

/// Packs dense NHWC bytes into feature-atomic NC1HWC2 surfaces.
///
/// For each pixel, consecutive 16-byte channel blocks are moved into
/// separate HxW surfaces. Any unused bytes in the final C2 block are zero,
/// which is required when the logical channel count is not atomic-aligned.
pub fn pack_nhwc_to_nc1hwc2(
    dense: &[u8],
    pixel_count: usize,
    bytes_per_pixel: usize,
    packed: &mut [u8],
) -> Result<usize, &'static str> {
    pack_nhwc_to_nc1hwc2_padded(dense, pixel_count, bytes_per_pixel, bytes_per_pixel, packed)
}

/// Packs dense NHWC bytes into NC1HWC2 with an explicitly padded pixel width.
///
/// `bytes_per_pixel` describes the logical dense input. The packed layout
/// has enough surfaces for `packed_bytes_per_pixel`, allowing callers to
/// match a hardware channel count that is wider than the logical tensor.
/// All padding surfaces and lanes are zero-filled.
pub fn pack_nhwc_to_nc1hwc2_padded(
    dense: &[u8],
    pixel_count: usize,
    bytes_per_pixel: usize,
    packed_bytes_per_pixel: usize,
    packed: &mut [u8],
) -> Result<usize, &'static str> {
    if packed_bytes_per_pixel < bytes_per_pixel {
        return Err("packed NC1HWC2 pixel width is smaller than the dense pixel width");
    }
    let dense_len = pixel_count
        .checked_mul(bytes_per_pixel)
        .ok_or("dense NHWC storage size overflows usize")?;
    if dense.len() < dense_len {
        return Err("dense NHWC input is smaller than its declared shape");
    }

    let packed_len = nc1hwc2_storage_size(pixel_count, packed_bytes_per_pixel)?;
    if packed.len() < packed_len {
        return Err("NC1HWC2 destination is smaller than its declared shape");
    }
    packed[..packed_len].fill(0);

    for pixel in 0..pixel_count {
        let dense_pixel = pixel * bytes_per_pixel;
        let mut copied = 0;
        while copied < bytes_per_pixel {
            let surface = copied / FEATURE_ATOMIC_BYTES;
            let chunk_len = (bytes_per_pixel - copied).min(FEATURE_ATOMIC_BYTES);
            let src_offset = dense_pixel + copied;
            let dst_offset =
                surface * pixel_count * FEATURE_ATOMIC_BYTES + pixel * FEATURE_ATOMIC_BYTES;
            packed[dst_offset..dst_offset + chunk_len]
                .copy_from_slice(&dense[src_offset..src_offset + chunk_len]);
            copied += chunk_len;
        }
    }

    Ok(packed_len)
}

/// Returns the storage needed for an uncompressed Rocket convolution filter.
///
/// The padding matches [`crate::rocket::conv::Shape::weight_channels`] and
/// [`crate::rocket::conv::Shape::programmed_kernels`]. FP16 pads input
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
    if filter_height == 0
        || filter_width == 0
        || input_channels == 0
        || output_channels == 0
        || element_size == 0
        || !matches!(element_size, 1 | 2)
    {
        return Err("invalid Rocket convolution filter shape");
    }

    let channels_per_atom = FEATURE_ATOMIC_BYTES / element_size;
    let input_atoms = input_channels.div_ceil(channels_per_atom);
    let padded_input_atoms = if element_size == 2 && input_atoms % 4 == 3 {
        input_atoms + 1
    } else {
        input_atoms
    };
    let padded_input_channels = padded_input_atoms * channels_per_atom;
    let programmed_output_channels = if element_size == 1 {
        output_channels.next_multiple_of(2)
    } else {
        output_channels
    };
    filter_height
        .checked_mul(filter_width)
        .and_then(|value| value.checked_mul(padded_input_channels))
        .and_then(|value| value.checked_mul(programmed_output_channels))
        .and_then(|value| value.checked_mul(element_size))
        .ok_or("Rocket convolution filter storage size overflows usize")
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
        element_size,
        None,
        packed,
    )
}

/// Packs a quantized int8 HWCF filter and fills physical input-channel
/// padding with each output channel's weight zero point.
///
/// The live bytes in `dense` are raw quantized coefficients and are copied
/// unchanged. A padded input lane participates in the hardware dot product,
/// so its neutral value is `weight_zero_points[output_channel]`, not
/// necessarily zero. The matching BS constant is `-weight_zero_point`; see
/// [`crate::rocket::conv::BsEntry::constant`]. A programmed odd-Cout padding
/// kernel remains all zero because it has no quantization parameters and is
/// never exposed as a logical output channel.
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
        1,
        Some(weight_zero_points),
        packed,
    )
}

fn pack_hwcf_to_rocket_weights_impl(
    dense: &[u8],
    filter_height: usize,
    filter_width: usize,
    input_channels: usize,
    output_channels: usize,
    element_size: usize,
    weight_zero_points: Option<&[i8]>,
    packed: &mut [u8],
) -> Result<usize, &'static str> {
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
        output_channels,
        element_size,
    )?;
    if packed.len() < packed_len {
        return Err("Rocket weight destination is smaller than its declared shape");
    }
    packed[..packed_len].fill(0);

    let output_block_channels = WEIGHT_ATOMIC_BYTES / element_size;
    let channels_per_atom = FEATURE_ATOMIC_BYTES / element_size;
    let input_atoms = input_channels.div_ceil(channels_per_atom);
    let padded_input_atoms = if element_size == 2 && input_atoms % 4 == 3 {
        input_atoms + 1
    } else {
        input_atoms
    };
    let padded_input_channels = padded_input_atoms * channels_per_atom;
    let programmed_output_channels = if element_size == 1 {
        output_channels.next_multiple_of(2)
    } else {
        output_channels
    };
    let input_groups = padded_input_channels.div_ceil(WEIGHT_INPUT_GROUP_CHANNELS);
    let output_blocks = programmed_output_channels.div_ceil(output_block_channels);
    let mut dst_offset = 0;

    for output_block in 0..output_blocks {
        for input_group in 0..input_groups {
            for filter_y in 0..filter_height {
                for filter_x in 0..filter_width {
                    for output_lane in 0..output_block_channels {
                        let output_channel = output_block * output_block_channels + output_lane;
                        if output_channel >= programmed_output_channels {
                            continue;
                        }
                        for input_lane in 0..WEIGHT_INPUT_GROUP_CHANNELS {
                            let input_channel =
                                input_group * WEIGHT_INPUT_GROUP_CHANNELS + input_lane;
                            if input_channel >= padded_input_channels {
                                continue;
                            }
                            if output_channel < output_channels {
                                if input_channel < input_channels {
                                    let src_element = (((filter_y * filter_width + filter_x)
                                        * input_channels
                                        + input_channel)
                                        * output_channels)
                                        + output_channel;
                                    let src_offset = src_element * element_size;
                                    packed[dst_offset..dst_offset + element_size].copy_from_slice(
                                        &dense[src_offset..src_offset + element_size],
                                    );
                                } else if let Some(zero_points) = weight_zero_points {
                                    packed[dst_offset] = zero_points[output_channel] as u8;
                                }
                            }
                            dst_offset += element_size;
                        }
                    }
                }
            }
        }
    }

    debug_assert_eq!(dst_offset, packed_len);
    Ok(packed_len)
}

/// Packs a depthwise filter into the RK3588 CNA coefficient order.
///
/// A depthwise filter is `[channels][filter_height][filter_width]` -- one
/// `kh x kw` kernel per input channel, with no `(input, output)` pairing at
/// all. The hardware wants it **tap-major within a channel group**: channels
/// are chunked into groups of [`WEIGHT_INPUT_GROUP_CHANNELS`] (32 -- the
/// same fixed grouping [`pack_hwcf_to_rocket_weights`]'s dense path already
/// uses, for both precisions), and every group's own 9 taps sit contiguously
/// before the next group starts:
///
/// ```text
/// slot = group_base(channel)
///      + (ky * filter_width + kx) * group_width(channel)
///      + (channel % WEIGHT_INPUT_GROUP_CHANNELS)
/// ```
///
/// where `group_base` is the running element offset of that channel's group
/// (`group * filter_height * filter_width * WEIGHT_INPUT_GROUP_CHANNELS`),
/// and `group_width` is 32 for every group except a final, shorter one when
/// `padded_channels` isn't a whole multiple of 32 (then it's the remainder).
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
/// [`WEIGHT_INPUT_GROUP_CHANNELS`] on the working assumption that it shares
/// the dense path's precision-independent grouping (true there), not its
/// own hardware confirmation.
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
/// [`Shape::weight_bytes`]: crate::rocket::conv::Shape::weight_bytes
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
        || element_size == 0
        || padded_channels < channels
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

    let full_groups = padded_channels / WEIGHT_INPUT_GROUP_CHANNELS;
    let tail_width = padded_channels - full_groups * WEIGHT_INPUT_GROUP_CHANNELS;

    for channel in 0..channels {
        let group = channel / WEIGHT_INPUT_GROUP_CHANNELS;
        let channel_in_group = channel % WEIGHT_INPUT_GROUP_CHANNELS;
        let group_width = if group < full_groups {
            WEIGHT_INPUT_GROUP_CHANNELS
        } else {
            tail_width
        };
        let group_base = group * filter_height * filter_width * WEIGHT_INPUT_GROUP_CHANNELS;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rocket::conv::Shape;

    #[test]
    fn packs_fp16_c32_as_four_channel_surfaces() {
        const PIXELS: usize = 2;
        const BYTES_PER_PIXEL: usize = 32 * 2;
        let dense: Vec<_> = (0..PIXELS * BYTES_PER_PIXEL)
            .map(|value| value as u8)
            .collect();
        let mut packed = vec![0xFF; nc1hwc2_storage_size(PIXELS, BYTES_PER_PIXEL).unwrap()];

        let written = pack_nhwc_to_nc1hwc2(&dense, PIXELS, BYTES_PER_PIXEL, &mut packed).unwrap();

        assert_eq!(written, PIXELS * 4 * FEATURE_ATOMIC_BYTES);
        for surface in 0..4 {
            for pixel in 0..PIXELS {
                let packed_offset = (surface * PIXELS + pixel) * FEATURE_ATOMIC_BYTES;
                let dense_offset = pixel * BYTES_PER_PIXEL + surface * FEATURE_ATOMIC_BYTES;
                assert_eq!(
                    &packed[packed_offset..packed_offset + FEATURE_ATOMIC_BYTES],
                    &dense[dense_offset..dense_offset + FEATURE_ATOMIC_BYTES]
                );
            }
        }
    }

    #[test]
    fn zero_pads_the_final_channel_surface() {
        const PIXELS: usize = 2;
        const BYTES_PER_PIXEL: usize = 3 * 2;
        let dense = vec![1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16];
        let mut packed = vec![0xFF; nc1hwc2_storage_size(PIXELS, BYTES_PER_PIXEL).unwrap()];

        pack_nhwc_to_nc1hwc2(&dense, PIXELS, BYTES_PER_PIXEL, &mut packed).unwrap();

        assert_eq!(&packed[0..6], &dense[0..6]);
        assert_eq!(&packed[6..16], &[0; 10]);
        assert_eq!(&packed[16..22], &dense[6..12]);
        assert_eq!(&packed[22..32], &[0; 10]);
    }

    #[test]
    fn zero_pads_fp16_c24_to_the_hardware_c32_width() {
        const PIXELS: usize = 2;
        const BYTES_PER_PIXEL: usize = 24 * 2;
        const PACKED_BYTES_PER_PIXEL: usize = 32 * 2;
        let dense: Vec<_> = (0..PIXELS * BYTES_PER_PIXEL)
            .map(|value| value as u8)
            .collect();
        let mut packed = vec![0xFF; nc1hwc2_storage_size(PIXELS, PACKED_BYTES_PER_PIXEL).unwrap()];

        let written = pack_nhwc_to_nc1hwc2_padded(
            &dense,
            PIXELS,
            BYTES_PER_PIXEL,
            PACKED_BYTES_PER_PIXEL,
            &mut packed,
        )
        .unwrap();

        assert_eq!(written, PIXELS * 4 * FEATURE_ATOMIC_BYTES);
        for surface in 0..3 {
            for pixel in 0..PIXELS {
                let packed_offset = (surface * PIXELS + pixel) * FEATURE_ATOMIC_BYTES;
                let dense_offset = pixel * BYTES_PER_PIXEL + surface * FEATURE_ATOMIC_BYTES;
                assert_eq!(
                    &packed[packed_offset..packed_offset + FEATURE_ATOMIC_BYTES],
                    &dense[dense_offset..dense_offset + FEATURE_ATOMIC_BYTES]
                );
            }
        }
        assert_eq!(&packed[PIXELS * 3 * FEATURE_ATOMIC_BYTES..], &[0; 32]);
    }

    #[test]
    fn rejects_short_buffers_and_size_overflow() {
        assert!(pack_nhwc_to_nc1hwc2(&[0; 3], 1, 4, &mut [0; 16]).is_err());
        assert!(pack_nhwc_to_nc1hwc2(&[0; 4], 1, 4, &mut [0; 15]).is_err());
        assert!(pack_nhwc_to_nc1hwc2_padded(&[0; 4], 1, 4, 3, &mut [0; 16]).is_err());
        assert!(nc1hwc2_storage_size(usize::MAX, 17).is_err());
    }

    #[test]
    fn packs_fp16_hwcf_in_output_and_input_blocks() {
        const H: usize = 1;
        const W: usize = 1;
        const C: usize = 32;
        const F: usize = 18;
        const BPE: usize = 2;
        let mut dense = vec![0u8; H * W * C * F * BPE];
        for input_channel in 0..C {
            for output_channel in 0..F {
                let value = (output_channel * 100 + input_channel) as u16;
                let offset = (input_channel * F + output_channel) * BPE;
                dense[offset..offset + BPE].copy_from_slice(&value.to_le_bytes());
            }
        }
        let mut packed = vec![0u8; rocket_weight_storage_size(H, W, C, F, BPE).unwrap()];

        pack_hwcf_to_rocket_weights(&dense, H, W, C, F, BPE, &mut packed).unwrap();

        let values = packed
            .chunks_exact(BPE)
            .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
            .collect::<Vec<_>>();
        assert_eq!(&values[0..32], &(0u16..32).collect::<Vec<_>>());
        assert_eq!(&values[32..64], &(100u16..132).collect::<Vec<_>>());
        assert_eq!(
            &values[16 * 32..16 * 32 + 32],
            &(1600u16..1632).collect::<Vec<_>>()
        );
        assert_eq!(
            &values[17 * 32..17 * 32 + 32],
            &(1700u16..1732).collect::<Vec<_>>()
        );
    }

    #[test]
    fn fp16_weight_storage_matches_ragged_plan_counts() {
        const BPE: usize = 2;
        // One fp16 feature atom is eight channels. Three atoms are the one
        // exceptional count that rounds to a four-atom coefficient group.
        assert_eq!(rocket_weight_storage_size(1, 1, 3, 2, BPE), Ok(32));
        assert_eq!(rocket_weight_storage_size(1, 1, 3, 3, BPE), Ok(48));
        assert_eq!(rocket_weight_storage_size(1, 1, 17, 3, BPE), Ok(192));
        // Five atoms pass through rather than rounding to a 16-channel
        // boundary.
        assert_eq!(rocket_weight_storage_size(1, 1, 40, 3, BPE), Ok(240));
    }

    #[test]
    fn fp16_weight_storage_agrees_with_conv_shape() {
        for input_channels in [1u32, 3, 8, 9, 17, 24, 25, 32, 40] {
            for output_channels in [1u32, 2, 3, 8, 17] {
                for kernels in [[1usize, 1], [3, 3], [3, 5], [4, 2]] {
                    let shape =
                        Shape::with_out_channels(32, 32, 1, input_channels, output_channels);
                    let packed = rocket_weight_storage_size(
                        kernels[0],
                        kernels[1],
                        input_channels as usize,
                        output_channels as usize,
                        2,
                    )
                    .unwrap();
                    assert_eq!(
                        packed,
                        shape.weight_bytes(kernels) as usize,
                        "Cin {input_channels}, Cout {output_channels}, kernel {kernels:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn packs_small_fp16_hwcf_without_phantom_channels_or_kernels() {
        const C: usize = 3;
        const F: usize = 3;
        const BPE: usize = 2;
        let mut dense = vec![0u8; C * F * BPE];
        for input_channel in 0..C {
            for output_channel in 0..F {
                let value = (10 * output_channel + input_channel + 1) as u16;
                let offset = (input_channel * F + output_channel) * BPE;
                dense[offset..offset + BPE].copy_from_slice(&value.to_le_bytes());
            }
        }
        let mut packed = vec![0u8; rocket_weight_storage_size(1, 1, C, F, BPE).unwrap()];

        let written = pack_hwcf_to_rocket_weights(&dense, 1, 1, C, F, BPE, &mut packed).unwrap();

        assert_eq!(written, 3 * 8 * BPE);
        let values = packed
            .chunks_exact(BPE)
            .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
            .collect::<Vec<_>>();
        assert_eq!(&values[0..8], &[1, 2, 3, 0, 0, 0, 0, 0]);
        assert_eq!(&values[8..16], &[11, 12, 13, 0, 0, 0, 0, 0]);
        assert_eq!(&values[16..24], &[21, 22, 23, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn packs_rectangular_fp16_taps_row_major() {
        const H: usize = 2;
        const W: usize = 3;
        const BPE: usize = 2;
        let dense = (1u16..=6).flat_map(u16::to_le_bytes).collect::<Vec<_>>();
        let mut packed = vec![0u8; rocket_weight_storage_size(H, W, 1, 1, BPE).unwrap()];

        pack_hwcf_to_rocket_weights(&dense, H, W, 1, 1, BPE, &mut packed).unwrap();

        let values = packed
            .chunks_exact(BPE)
            .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
            .collect::<Vec<_>>();
        for (tap, expected) in (1u16..=6).enumerate() {
            assert_eq!(values[tap * 8], expected, "row-major tap {tap}");
            assert_eq!(&values[tap * 8 + 1..tap * 8 + 8], &[0; 7]);
        }
    }

    #[test]
    fn int8_weight_storage_keeps_its_distinct_padding_rules() {
        assert_eq!(rocket_weight_storage_size(1, 1, 3, 3, 1), Ok(64));
        assert_eq!(rocket_weight_storage_size(1, 1, 17, 3, 1), Ok(128));
    }

    #[test]
    fn packs_affine_int8_weights_with_per_output_neutral_padding() {
        // Controlled vendor-style Cin=2/Cout=2 coefficients. Logical rows
        // are [-1,+1] and [+0.5,-0.5] after their per-output affine decode;
        // this test pins only the byte-level HWCF transform.
        let dense = [0x80, 0x7f, 0x7f, 0x80];
        let zero_points = [42i8, -43];
        let mut packed = vec![0; rocket_weight_storage_size(1, 1, 2, 2, 1).unwrap()];

        let written =
            pack_hwcf_to_rocket_weights_affine_i8(&dense, 1, 1, 2, 2, &zero_points, &mut packed)
                .unwrap();

        assert_eq!(written, 32);
        assert_eq!(&packed[0..2], &[0x80, 0x7f]);
        assert!(packed[2..16].iter().all(|&byte| byte == 42));
        assert_eq!(&packed[16..18], &[0x7f, 0x80]);
        assert!(packed[18..32].iter().all(|&byte| byte == (-43i8) as u8));
    }

    #[test]
    fn affine_int8_packer_rejects_wrong_zero_point_count() {
        let mut packed = vec![0; 32];
        assert_eq!(
            pack_hwcf_to_rocket_weights_affine_i8(&[0; 4], 1, 1, 2, 2, &[0], &mut packed),
            Err("int8 weight zero-point count does not match output channels")
        );
    }

    /// The exact slot mapping the hardware probe reported at Cin 8, 3x3:
    /// channel `c` at tap (0,0) lands at slot `c`, and channel 0's next tap
    /// (0,1) lands at slot 8 -- one whole channel row later, not adjacent.
    #[test]
    fn depthwise_packing_is_tap_major() {
        // One byte per element keeps the slot index and the byte offset the
        // same number, so the expectations read as slots.
        let (channels, kh, kw) = (8usize, 3usize, 3usize);
        let mut dense = vec![0u8; channels * kh * kw];
        for channel in 0..channels {
            for ky in 0..kh {
                for kx in 0..kw {
                    // Encode the source coordinate so a misplaced byte names
                    // where it came from.
                    dense[(channel * kh + ky) * kw + kx] = (channel * 16 + ky * 4 + kx) as u8;
                }
            }
        }

        let mut packed = vec![0xffu8; kh * kw * channels];
        let written =
            pack_depthwise_to_rocket_weights(&dense, kh, kw, channels, channels, 1, &mut packed)
                .expect("packing failed");
        assert_eq!(written, 72);

        for channel in 0..channels {
            for ky in 0..kh {
                for kx in 0..kw {
                    let slot = (ky * kw + kx) * channels + channel;
                    assert_eq!(
                        packed[slot],
                        (channel * 16 + ky * 4 + kx) as u8,
                        "slot {slot} (channel {channel}, tap ({ky}, {kx}))"
                    );
                }
            }
        }
        // The probe's own landmarks.
        assert_eq!(packed[0], 0, "channel 0 tap (0,0)");
        assert_eq!(packed[1], 16, "channel 1 tap (0,0)");
        assert_eq!(packed[8], 1, "channel 0 tap (0,1)");
        // channel 0, tap (2,2) -> 0*16 + 2*4 + 2
        assert_eq!(packed[64], 10, "channel 0 tap (2,2)");
    }

    /// Padding slots stay zero and the real channels keep the padded stride,
    /// which is what a Cin the atom granularity does not divide needs.
    #[test]
    fn depthwise_packing_honours_the_padded_stride() {
        let (channels, padded, kh, kw) = (12usize, 16usize, 3usize, 3usize);
        let dense = vec![0x5au8; channels * kh * kw];
        let mut packed = vec![0xffu8; kh * kw * padded];
        let written =
            pack_depthwise_to_rocket_weights(&dense, kh, kw, channels, padded, 1, &mut packed)
                .expect("packing failed");
        assert_eq!(written, kh * kw * padded);

        for ky in 0..kh {
            for kx in 0..kw {
                let base = (ky * kw + kx) * padded;
                for channel in 0..padded {
                    let want = if channel < channels { 0x5a } else { 0 };
                    assert_eq!(
                        packed[base + channel],
                        want,
                        "tap ({ky}, {kx}) channel {channel}"
                    );
                }
            }
        }
    }

    /// The two tests above never leave a single 32-channel group, so they
    /// cannot tell a per-group stride from one global stride spanning the
    /// whole padded channel count -- both formulas agree there. A hardware
    /// probe through the real compiled dispatch path (not this isolated
    /// packer), summing distinct known values on every tap of one channel,
    /// found real coefficients leaking into channels exactly
    /// `WEIGHT_INPUT_GROUP_CHANNELS` (32) apart at Cin 128 and 256, and
    /// -- with a genuine short final group -- at Cin 144. This is the
    /// smallest shape (one full group plus a real tail group) that
    /// reproduces the 144 probe's structure by hand. Values are 2 bytes so
    /// `channel * 16 + ky * 4 + kx` stays unique without wrapping past
    /// channel 15 the way the byte-sized encoding above would.
    #[test]
    fn depthwise_packing_groups_channels_past_the_first_thirty_two() {
        let (channels, padded, kh, kw) = (48usize, 48usize, 3usize, 3usize);
        let mut dense = vec![0u8; channels * kh * kw * 2];
        for channel in 0..channels {
            for ky in 0..kh {
                for kx in 0..kw {
                    let value = (channel * 16 + ky * 4 + kx) as u16;
                    let offset = ((channel * kh + ky) * kw + kx) * 2;
                    dense[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
                }
            }
        }

        let mut packed = vec![0xffu8; kh * kw * padded * 2];
        let written =
            pack_depthwise_to_rocket_weights(&dense, kh, kw, channels, padded, 2, &mut packed)
                .expect("packing failed");
        assert_eq!(written, kh * kw * padded * 2);

        // Group 0: channels 0..32, width 32, base 0. Group 1 (tail):
        // channels 32..48, width 16, base 9*32=288 elements.
        for channel in 0..channels {
            let (group_base, group_width) = if channel < 32 { (0, 32) } else { (288, 16) };
            let channel_in_group = channel % 32;
            for ky in 0..kh {
                for kx in 0..kw {
                    let slot = group_base + (ky * kw + kx) * group_width + channel_in_group;
                    let want = (channel * 16 + ky * 4 + kx) as u16;
                    let got = u16::from_le_bytes([packed[slot * 2], packed[slot * 2 + 1]]);
                    assert_eq!(got, want, "slot {slot} (channel {channel}, tap ({ky}, {kx}))");
                }
            }
        }
        // Channel 32 opens the tail group right after group 0's 9*32=288
        // elements -- not immediately after channel 31's own tap (0,0) the
        // way a single global stride would place it.
        let tail_start = u16::from_le_bytes([packed[288 * 2], packed[288 * 2 + 1]]);
        assert_eq!(tail_start, 32 * 16, "channel 32 tap (0,0), start of tail group");
        // Channel 47 (last real channel) tap (2,2): the tail group's own
        // last slot, at 288 + 8*16 + 15 = 431, the buffer's final element.
        let last = u16::from_le_bytes([packed[431 * 2], packed[431 * 2 + 1]]);
        assert_eq!(last, 47 * 16 + 2 * 4 + 2, "channel 47 tap (2,2)");
    }

    #[test]
    fn depthwise_packing_rejects_a_short_buffer() {
        let dense = vec![0u8; 8 * 9];
        let mut packed = vec![0u8; 8 * 9 - 1];
        assert!(
            pack_depthwise_to_rocket_weights(&dense, 3, 3, 8, 8, 1, &mut packed).is_err(),
            "a packed buffer one byte short must be refused"
        );
    }
}
