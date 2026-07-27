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
        || !WEIGHT_ATOMIC_BYTES.is_multiple_of(element_size)
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

/// Packs a depthwise filter into the RK3588 CNA coefficient order.
///
/// A depthwise filter is `[channels][filter_height][filter_width]` -- one
/// `kh x kw` kernel per input channel, with no `(input, output)` pairing at
/// all. The hardware wants it **tap-major**:
///
/// `slot = (ky * filter_width + kx) * padded_channels + channel`
///
/// i.e. all channels' coefficients for one tap sit contiguously, then all
/// channels' coefficients for the next. This is the transpose of how torch
/// and ONNX store it, and it is nothing like
/// [`pack_hwcf_to_rocket_weights`]'s blocked dense order.
///
/// `padded_channels` is the count the register program's
/// `CNA_WEIGHT_SIZE0.weight_bytes` is sized from -- [`Shape::weight_bytes`]
/// divided by `kh * kw * element_size` -- not the real channel count. The
/// two differ whenever the channel count is not a whole CBUF atom group;
/// padding slots are left zero and contribute nothing.
///
/// Derived by one-hot probing every slot of a real weight buffer on
/// hardware, reading back which output channel responded and which image
/// borders went zero (`tests/conv_depthwise_probe_hw.rs`). A capture could
/// never have shown this: it carries the register program, not the buffer.
/// The original exhaustive probe used fp16; the same tap-major order and a
/// padded stride of 16 were subsequently validated for int8 at Cin 12 by
/// `tests/conv_phase1_validation_hw.rs`.
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
    for channel in 0..channels {
        for ky in 0..filter_height {
            for kx in 0..filter_width {
                let from = ((channel * filter_height + ky) * filter_width + kx) * element_size;
                let to = ((ky * filter_width + kx) * padded_channels + channel) * element_size;
                packed[to..to + element_size].copy_from_slice(&dense[from..from + element_size]);
            }
        }
    }
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
