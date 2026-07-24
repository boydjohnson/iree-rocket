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
    let dense_len = pixel_count
        .checked_mul(bytes_per_pixel)
        .ok_or("dense NHWC storage size overflows usize")?;
    if dense.len() < dense_len {
        return Err("dense NHWC input is smaller than its declared shape");
    }

    let packed_len = nc1hwc2_storage_size(pixel_count, bytes_per_pixel)?;
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
/// `input_channels` are padded exactly as the convolution register builder
/// programs them: at least 16 channels and then to a multiple of 16.
/// Output kernels are padded only to the hardware's two-kernel granularity.
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
        || WEIGHT_ATOMIC_BYTES % element_size != 0
    {
        return Err("invalid Rocket convolution filter shape");
    }

    let padded_input_channels = input_channels.max(16).next_multiple_of(16);
    let padded_output_channels = output_channels.next_multiple_of(2);
    filter_height
        .checked_mul(filter_width)
        .and_then(|value| value.checked_mul(padded_input_channels))
        .and_then(|value| value.checked_mul(padded_output_channels))
        .and_then(|value| value.checked_mul(element_size))
        .ok_or("Rocket convolution filter storage size overflows usize")
}

/// Packs a logical HWCF filter into the RK3588 CNA coefficient order.
///
/// The physical nesting is:
///
/// `output_block -> input_group -> filter_x -> filter_y ->
/// output_lane -> input_lane`
///
/// An output block is one 32-byte weight atom: 32 kernels for int8 or 16
/// kernels for fp16. Input groups remain 32 channels for both precisions,
/// with a partial final group when the register-programmed input channel
/// count is not divisible by 32. Input channels are zero-padded to the
/// register-programmed 16-channel granularity and output kernels to a
/// multiple of two.
pub fn pack_hwcf_to_rocket_weights(
    dense: &[u8],
    filter_height: usize,
    filter_width: usize,
    input_channels: usize,
    output_channels: usize,
    element_size: usize,
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
    let padded_input_channels = input_channels.max(16).next_multiple_of(16);
    let padded_output_channels = output_channels.next_multiple_of(2);
    let input_groups = padded_input_channels.div_ceil(WEIGHT_INPUT_GROUP_CHANNELS);
    let output_blocks = padded_output_channels.div_ceil(output_block_channels);
    let mut dst_offset = 0;

    for output_block in 0..output_blocks {
        for input_group in 0..input_groups {
            for filter_x in 0..filter_width {
                for filter_y in 0..filter_height {
                    for output_lane in 0..output_block_channels {
                        let output_channel = output_block * output_block_channels + output_lane;
                        if output_channel >= padded_output_channels {
                            continue;
                        }
                        for input_lane in 0..WEIGHT_INPUT_GROUP_CHANNELS {
                            let input_channel =
                                input_group * WEIGHT_INPUT_GROUP_CHANNELS + input_lane;
                            if input_channel >= padded_input_channels {
                                continue;
                            }
                            if input_channel < input_channels && output_channel < output_channels {
                                let src_element = (((filter_y * filter_width + filter_x)
                                    * input_channels
                                    + input_channel)
                                    * output_channels)
                                    + output_channel;
                                let src_offset = src_element * element_size;
                                packed[dst_offset..dst_offset + element_size]
                                    .copy_from_slice(&dense[src_offset..src_offset + element_size]);
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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn rejects_short_buffers_and_size_overflow() {
        assert!(pack_nhwc_to_nc1hwc2(&[0; 3], 1, 4, &mut [0; 16]).is_err());
        assert!(pack_nhwc_to_nc1hwc2(&[0; 4], 1, 4, &mut [0; 15]).is_err());
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
}
