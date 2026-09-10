//! Host-side transforms between dense NHWC tensors and the NPU's
//! feature-atomic NC1HWC2 layout.
//!
//! The hardware's inner channel block is 16 bytes, not a fixed number of
//! elements: C2 is 16 for int8 and 8 for fp16. Each C1 surface contains
//! every HxW pixel before the next channel block begins.

use crate::rocket::conv::AccumulatorOutputTile;

/// Physical byte width of one NC1HWC2 inner-channel block.
pub const FEATURE_ATOMIC_BYTES: usize = 16;

// The coefficient layout and its packers live in `rocket_core::weights`
// (COMPILER_ROADMAP.md 6.3) and are re-exported here so every caller and
// test keeps its `tensor_layout::` path.
pub use rocket_core::weights::{
    TF32_WEIGHT_INPUT_GROUP_CHANNELS, TF32_WEIGHT_OUTPUT_BLOCK_CHANNELS, WEIGHT_ATOMIC_BYTES,
    WEIGHT_INPUT_GROUP_CHANNELS, pack_depthwise_to_rocket_weights, pack_hwcf_to_rocket_weights,
    pack_hwcf_to_rocket_weights_affine_i8, pack_hwcf_to_rocket_weights_int4,
    pack_hwcf_to_rocket_weights_int4_padded, pack_hwcf_to_rocket_weights_padded,
    pack_nhwc_to_nc1hwc2_int4, rocket_weight_storage_size, rocket_weight_storage_size_bits,
};

/// Returns the storage needed for an FP16 BRDMA bias operand stream.
///
/// FP16 enables only the BS ALU operand (`brdma_data_use = 1`), but that
/// operand is a 32-bit float at the RK3588 BRDMA boundary. The logical IREE
/// tensor remains FP16 and is widened by [`pack_fp16_bias_to_rocket`]. The
/// destination is sized to the DPU's programmed (padded) output-channel count
/// so BRDMA never reaches an adjacent allocation for the final partial
/// channel granule.
pub fn rocket_fp16_bias_storage_size(padded_output_channels: usize) -> Result<usize, &'static str> {
    if padded_output_channels == 0 {
        return Err("Rocket FP16 bias channel count must be nonzero");
    }
    padded_output_channels
        .checked_mul(4)
        .ok_or("Rocket FP16 bias storage size overflows usize")
}

/// Widens a logical dense FP16 bias vector into BRDMA's FP32 operand stream.
///
/// Only `output_channels * 2` bytes are read from `dense`; all physical tail
/// channels are zero. Public IREE bindings may be exact-sized subranges with
/// unrelated live data immediately before and after them, while the DPU is
/// programmed for `padded_output_channels`.
pub fn pack_fp16_bias_to_rocket(
    dense: &[u8],
    output_channels: usize,
    padded_output_channels: usize,
    packed: &mut [u8],
) -> Result<usize, &'static str> {
    if output_channels == 0 || padded_output_channels < output_channels {
        return Err("invalid Rocket FP16 bias channel counts");
    }
    let dense_len = output_channels
        .checked_mul(2)
        .ok_or("dense FP16 bias storage size overflows usize")?;
    if dense.len() < dense_len {
        return Err("dense FP16 bias is smaller than its declared shape");
    }
    let packed_len = rocket_fp16_bias_storage_size(padded_output_channels)?;
    if packed.len() < packed_len {
        return Err("Rocket FP16 bias destination is smaller than its declared shape");
    }
    packed[..packed_len].fill(0);
    for channel in 0..output_channels {
        let source = channel * 2;
        let fp16 = u16::from_le_bytes([dense[source], dense[source + 1]]);
        let destination = channel * 4;
        packed[destination..destination + 4]
            .copy_from_slice(&fp16_to_fp32_bits(fp16).to_le_bytes());
    }
    Ok(packed_len)
}

/// Exact IEEE-754 binary16 to binary32 widening, returned as raw bits.
fn fp16_to_fp32_bits(value: u16) -> u32 {
    let sign = (u32::from(value) & 0x8000) << 16;
    let exponent = (value >> 10) & 0x1f;
    let fraction = value & 0x03ff;
    match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let mut normalized = u32::from(fraction);
            let mut unbiased_exponent = -14i32;
            while normalized & 0x0400 == 0 {
                normalized <<= 1;
                unbiased_exponent -= 1;
            }
            normalized &= 0x03ff;
            sign | (((unbiased_exponent + 127) as u32) << 23) | (normalized << 13)
        }
        0x1f => sign | 0x7f80_0000 | (u32::from(fraction) << 13),
        _ => sign | ((u32::from(exponent) + (127 - 15)) << 23) | (u32::from(fraction) << 13),
    }
}

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

    // Only the bytes the copy below will not reach are zeroed, not the whole
    // destination. Zeroing everything first is a second full write pass over
    // the packed buffer, and for a channel count that is a whole number of
    // atoms -- every convolution in MobileNetV2 fp16 -- it is *entirely*
    // wasted: the copy overwrites every byte of it. What genuinely needs
    // zeroing is the channel padding: the tail of a partial last atom (filled
    // per pixel in the loop, where the line is already hot) and any whole
    // surface that exists only because `packed_bytes_per_pixel` is wider than
    // the logical pixel.
    let written_surfaces = bytes_per_pixel.div_ceil(FEATURE_ATOMIC_BYTES);
    let padding_surfaces_start = written_surfaces * pixel_count * FEATURE_ATOMIC_BYTES;
    if padding_surfaces_start < packed_len {
        packed[padding_surfaces_start..packed_len].fill(0);
    }

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
            if chunk_len < FEATURE_ATOMIC_BYTES {
                packed[dst_offset + chunk_len..dst_offset + FEATURE_ATOMIC_BYTES].fill(0);
            }
            copied += chunk_len;
        }
    }

    Ok(packed_len)
}

// The inverse direction: DPU output surfaces back to dense NHWC. Kept next to
// the packing above because the two are a pair -- every Rocket dispatch pays
// both, once each, and any change to how one walks memory has to be made to
// the other (see ISSUES.md P6, where together they are the largest host cost
// left in the driver).

/// Interleaves DPU feature-atomic output surfaces into dense NHWC pixels.
///
/// Each hardware surface stores one channel block for every spatial pixel;
/// `DPU_DST_SURF_STRIDE` advances between those full spatial surfaces.
/// Dense NHWC instead stores every channel group for one pixel contiguously,
/// so copy one surface chunk at a time into each destination pixel. Ordinary
/// output blocks are 16 bytes; accumulator blocks are 32 i32 lanes (128
/// bytes). The final logical surface may be partial.
pub fn compact_atomic_output(
    scratch: &[u8],
    source_pixel_count: usize,
    output_pixel_count: usize,
    bytes_per_pixel: usize,
    source_block_bytes: usize,
    dst: &mut [u8],
) -> usize {
    if source_block_bytes == 0 {
        return 0;
    }
    let mut written = 0;
    for pixel in 0..output_pixel_count {
        let mut pixel_written = 0;
        while pixel_written < bytes_per_pixel {
            let surface = pixel_written / source_block_bytes;
            let chunk_len = (bytes_per_pixel - pixel_written).min(source_block_bytes);
            let src_off =
                surface * source_pixel_count * source_block_bytes + pixel * source_block_bytes;
            let dst_off = pixel * bytes_per_pixel + pixel_written;
            if src_off + chunk_len > scratch.len() || dst_off + chunk_len > dst.len() {
                return written;
            }
            dst[dst_off..dst_off + chunk_len]
                .copy_from_slice(&scratch[src_off..src_off + chunk_len]);
            written += chunk_len;
            pixel_written += chunk_len;
        }
    }
    written
}

/// [`compact_atomic_output`] for one rectangle of the output image.
///
/// `scratch` holds the whole `source_pixel_count`-pixel image in
/// atomic-slot surfaces, exactly as [`compact_atomic_output`] expects; only
/// the pixels in rows `first_row..first_row + rows`, columns
/// `first_column..first_column + columns` of an `output_width`-wide image are
/// compacted, into their own places in `dst`. This is how a dispatch whose
/// tiles ran on several NPU contexts -- each writing its own copy of the
/// output scratch -- gathers one dense output: one call per tile, each
/// reading the scratch that tile's context wrote.
#[allow(clippy::too_many_arguments)]
pub fn compact_atomic_output_rect(
    scratch: &[u8],
    source_pixel_count: usize,
    output_width: usize,
    first_row: usize,
    rows: usize,
    first_column: usize,
    columns: usize,
    bytes_per_pixel: usize,
    source_block_bytes: usize,
    dst: &mut [u8],
) -> usize {
    if source_block_bytes == 0 || output_width == 0 {
        return 0;
    }
    let mut written = 0;
    for row in first_row..first_row + rows {
        for column in first_column..first_column + columns {
            let pixel = row * output_width + column;
            let mut pixel_written = 0;
            while pixel_written < bytes_per_pixel {
                let surface = pixel_written / source_block_bytes;
                let chunk_len = (bytes_per_pixel - pixel_written).min(source_block_bytes);
                let src_off =
                    surface * source_pixel_count * source_block_bytes + pixel * source_block_bytes;
                let dst_off = pixel * bytes_per_pixel + pixel_written;
                if src_off + chunk_len > scratch.len() || dst_off + chunk_len > dst.len() {
                    return written;
                }
                dst[dst_off..dst_off + chunk_len]
                    .copy_from_slice(&scratch[src_off..src_off + chunk_len]);
                written += chunk_len;
                pixel_written += chunk_len;
            }
        }
    }
    written
}

pub fn compact_tiled_accumulator_output(
    scratch: &[u8],
    tiles: &[AccumulatorOutputTile],
    output_width: usize,
    bytes_per_pixel: usize,
    source_block_bytes: usize,
    dst: &mut [u8],
) -> usize {
    if source_block_bytes == 0 || output_width == 0 {
        return 0;
    }
    let mut written = 0;
    for tile in tiles {
        let tile_pixels = tile.output_rows * tile.output_columns;
        let Some(tile_end) = tile.scratch_offset.checked_add(tile.scratch_bytes) else {
            return written;
        };
        if tile_end > scratch.len() {
            return written;
        }
        for row in 0..tile.output_rows {
            for column in 0..tile.output_columns {
                let local_pixel = row * tile.output_columns + column;
                let output_pixel =
                    (tile.output_row + row) * output_width + tile.output_column + column;
                let mut pixel_written = 0;
                while pixel_written < bytes_per_pixel {
                    let surface = pixel_written / source_block_bytes;
                    let chunk_len = (bytes_per_pixel - pixel_written).min(source_block_bytes);
                    let src_off = tile.scratch_offset
                        + surface * tile_pixels * source_block_bytes
                        + local_pixel * source_block_bytes;
                    let dst_off = output_pixel * bytes_per_pixel + pixel_written;
                    if src_off + chunk_len > tile_end || dst_off + chunk_len > dst.len() {
                        return written;
                    }
                    dst[dst_off..dst_off + chunk_len]
                        .copy_from_slice(&scratch[src_off..src_off + chunk_len]);
                    written += chunk_len;
                    pixel_written += chunk_len;
                }
            }
        }
    }
    written
}

#[cfg(test)]
// `as_chunks` would need every `bytes.try_into()`/index below re-typed for
// marginal benefit on these numeric decode paths; not worth the churn.
#[allow(clippy::chunks_exact_to_as_chunks)]
mod tests {
    use super::*;
    use crate::rocket::conv::Shape;

    #[test]
    fn packs_exact_fp16_bias_and_zeroes_programmed_tail() {
        let dense = [0x3C00u16, 0x4000, 0x4200]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>();
        let mut packed = vec![0xA5; rocket_fp16_bias_storage_size(16).unwrap()];

        let written = pack_fp16_bias_to_rocket(&dense, 3, 16, &mut packed).unwrap();

        assert_eq!(written, 64);
        let widened = packed[..12]
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(
            widened,
            [1.0f32.to_bits(), 2.0f32.to_bits(), 3.0f32.to_bits()]
        );
        assert!(packed[12..written].iter().all(|&byte| byte == 0));
    }

    #[test]
    fn fp16_bias_packing_reads_only_the_declared_subrange() {
        const PREFIX: usize = 11;
        let mut allocation = vec![0xD3; PREFIX];
        allocation.extend(
            [0x4900u16, 0x4980, 0x4A00]
                .into_iter()
                .flat_map(u16::to_le_bytes),
        );
        allocation.extend([0x7B; 13]);
        let logical_len = 3 * 2;
        let dense = &allocation[PREFIX..PREFIX + logical_len];
        let mut packed = vec![0xFF; 64];

        pack_fp16_bias_to_rocket(dense, 3, 16, &mut packed).unwrap();

        let widened = packed[..12]
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(widened, [10.0, 11.0, 12.0]);
        assert!(packed[12..].iter().all(|&byte| byte == 0));
    }

    #[test]
    fn fp16_bias_packing_rejects_short_and_invalid_buffers() {
        assert!(pack_fp16_bias_to_rocket(&[0; 5], 3, 16, &mut [0; 64]).is_err());
        assert!(pack_fp16_bias_to_rocket(&[0; 6], 3, 2, &mut [0; 64]).is_err());
        assert!(pack_fp16_bias_to_rocket(&[0; 6], 3, 16, &mut [0; 63]).is_err());
        assert!(rocket_fp16_bias_storage_size(0).is_err());
        assert!(rocket_fp16_bias_storage_size(usize::MAX).is_err());
    }

    #[test]
    fn fp16_bias_widening_handles_ieee_edges() {
        for (fp16, fp32) in [
            (0x0000, 0x0000_0000), // positive zero
            (0x8000, 0x8000_0000), // negative zero
            (0x0001, 0x3380_0000), // smallest subnormal
            (0x0400, 0x3880_0000), // smallest normal
            (0x3C00, 0x3F80_0000), // one
            (0xC000, 0xC000_0000), // negative two
            (0x7BFF, 0x477F_E000), // largest finite
            (0x7C00, 0x7F80_0000), // positive infinity
            (0xFC00, 0xFF80_0000), // negative infinity
        ] {
            assert_eq!(fp16_to_fp32_bits(fp16), fp32, "FP16 bits {fp16:#06x}");
        }
        let nan = fp16_to_fp32_bits(0x7E01);
        assert_eq!(nan & 0x7F80_0000, 0x7F80_0000);
        assert_ne!(nan & 0x007F_FFFF, 0);
    }

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

    /// A partial last atom *and* whole padding surfaces above it, which the
    /// packer zeroes by two different paths -- the per-pixel tail inside the
    /// copy loop, and one fill for the surfaces the copy never reaches. The
    /// two other padding tests each exercise only one of them.
    #[test]
    fn zero_pads_a_partial_atom_under_padding_surfaces() {
        const PIXELS: usize = 3;
        // 10 channels of fp16: one full atom plus 4 bytes of a second.
        const BYTES_PER_PIXEL: usize = 10 * 2;
        // Programmed as 32 channels, so two whole surfaces of padding above.
        const PACKED_BYTES_PER_PIXEL: usize = 32 * 2;
        let dense: Vec<_> = (0..PIXELS * BYTES_PER_PIXEL)
            .map(|value| value as u8 + 1)
            .collect();
        let mut packed = vec![0xFF; nc1hwc2_storage_size(PIXELS, PACKED_BYTES_PER_PIXEL).unwrap()];

        pack_nhwc_to_nc1hwc2_padded(
            &dense,
            PIXELS,
            BYTES_PER_PIXEL,
            PACKED_BYTES_PER_PIXEL,
            &mut packed,
        )
        .unwrap();

        for pixel in 0..PIXELS {
            let first = pixel * FEATURE_ATOMIC_BYTES;
            assert_eq!(
                &packed[first..first + 16],
                &dense[pixel * 20..pixel * 20 + 16]
            );
            let second = (PIXELS + pixel) * FEATURE_ATOMIC_BYTES;
            assert_eq!(
                &packed[second..second + 4],
                &dense[pixel * 20 + 16..pixel * 20 + 20]
            );
            assert_eq!(&packed[second + 4..second + 16], &[0; 12]);
        }
        assert_eq!(&packed[2 * PIXELS * FEATURE_ATOMIC_BYTES..], &[0; 96]);
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

    /// Int8 depthwise groups 64 channels, not 16.
    ///
    /// This test previously asserted a 16-channel group, and that was wrong
    /// on hardware. A delta-function probe on RK3588 (single nonzero tap,
    /// `Cin` 64, 3x3, `padding = [0, 0]`) came back with output channels
    /// 0..15, 16..31, 32..47 and 48..63 shifted by taps 0, 2, 4 and 6 -- an
    /// arithmetic progression that solves for exactly one address function,
    /// `tap * 64 + channel`. Nothing caught it earlier because the only
    /// hardware coverage of this path used uniform all-ones weights, which
    /// cannot observe a tap permutation at all, and because every unit test
    /// here checked the packer against the same assumed constant rather than
    /// against the hardware.
    ///
    /// `channels = 64` is deliberately one full group, and the case below
    /// covers a short final group.
    #[test]
    fn int8_depthwise_packing_uses_sixty_four_channel_groups() {
        let (channels, kh, kw) = (64usize, 3usize, 3usize);
        let mut dense = vec![0u8; channels * kh * kw];
        for channel in 0..channels {
            for tap in 0..kh * kw {
                // Distinct per (channel, tap) modulo 256 -- 9 taps and 64
                // channels fit without aliasing.
                dense[channel * kh * kw + tap] = (channel * 4 + tap) as u8;
            }
        }
        let mut packed = vec![0xffu8; channels * kh * kw];
        let written =
            pack_depthwise_to_rocket_weights(&dense, kh, kw, channels, channels, 1, &mut packed)
                .expect("packing failed");
        assert_eq!(written, kh * kw * channels);

        // One contiguous run of all 64 channels per tap.
        for tap in 0..kh * kw {
            for channel in 0..channels {
                assert_eq!(
                    packed[tap * channels + channel],
                    (channel * 4 + tap) as u8,
                    "tap {tap} channel {channel}"
                );
            }
        }
    }

    /// The int8 tail group, which the 64-channel case above cannot reach:
    /// `Cin` 48 pads to 48, leaving a single short group of width 48 rather
    /// than a full 64. Same address function, narrower run.
    #[test]
    fn int8_depthwise_packing_handles_a_short_final_group() {
        let (channels, padded, kh, kw) = (48usize, 48usize, 3usize, 3usize);
        let mut dense = vec![0u8; channels * kh * kw];
        for channel in 0..channels {
            for tap in 0..kh * kw {
                dense[channel * kh * kw + tap] = (channel * 4 + tap) as u8;
            }
        }
        let mut packed = vec![0xffu8; kh * kw * padded];
        let written =
            pack_depthwise_to_rocket_weights(&dense, kh, kw, channels, padded, 1, &mut packed)
                .expect("packing failed");
        assert_eq!(written, kh * kw * padded);

        for tap in 0..kh * kw {
            for channel in 0..padded {
                let want = if channel < channels {
                    (channel * 4 + tap) as u8
                } else {
                    0
                };
                assert_eq!(
                    packed[tap * padded + channel],
                    want,
                    "tap {tap} channel {channel}"
                );
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
                    assert_eq!(
                        got, want,
                        "slot {slot} (channel {channel}, tap ({ky}, {kx}))"
                    );
                }
            }
        }
        // Channel 32 opens the tail group right after group 0's 9*32=288
        // elements -- not immediately after channel 31's own tap (0,0) the
        // way a single global stride would place it.
        let tail_start = u16::from_le_bytes([packed[288 * 2], packed[288 * 2 + 1]]);
        assert_eq!(
            tail_start,
            32 * 16,
            "channel 32 tap (0,0), start of tail group"
        );
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

#[cfg(test)]
mod parity_padding_tests {
    use super::*;

    /// The surplus channels a parity pad introduces must pack to literal
    /// zero coefficients, and the real channels must land exactly where an
    /// unpadded pack would put them.
    ///
    /// Both halves matter: zeros are what make the padding channels compute
    /// nothing, and the real channels moving would corrupt the actual
    /// convolution. Verified against hardware by
    /// `int8_accumulator_cout_padding_probe`.
    #[test]
    fn padded_pack_zero_fills_surplus_and_preserves_real_channels() {
        for kernel in [1usize, 3] {
            let (cin, cout, padded) = (8usize, 32usize, 64usize);
            let dense: Vec<u8> = (0..kernel * kernel * cin * cout)
                .map(|index| ((index % 7) as i8 - 3) as u8)
                .collect();

            let unpadded_len = rocket_weight_storage_size(kernel, kernel, cin, cout, 1).unwrap();
            let mut unpadded = vec![0xffu8; unpadded_len];
            pack_hwcf_to_rocket_weights(&dense, kernel, kernel, cin, cout, 1, &mut unpadded)
                .unwrap();

            let padded_len = rocket_weight_storage_size(kernel, kernel, cin, padded, 1).unwrap();
            let mut packed = vec![0xffu8; padded_len];
            let written = pack_hwcf_to_rocket_weights_padded(
                &dense,
                kernel,
                kernel,
                cin,
                cout,
                padded,
                1,
                &mut packed,
            )
            .unwrap();

            assert_eq!(written, padded_len);
            assert_eq!(padded_len, 2 * unpadded_len, "k{kernel} should double");
            assert_eq!(
                &packed[..unpadded_len],
                &unpadded[..],
                "k{kernel}: real channels must pack identically to the unpadded filter"
            );
            assert!(
                packed[unpadded_len..].iter().all(|byte| *byte == 0),
                "k{kernel}: surplus channels must pack to zero coefficients"
            );
        }
    }

    #[test]
    fn padded_pack_rejects_a_narrower_padded_count() {
        let dense = vec![0u8; 8 * 32];
        let mut packed = vec![0u8; 4096];
        assert!(
            pack_hwcf_to_rocket_weights_padded(&dense, 1, 1, 8, 32, 16, 1, &mut packed).is_err()
        );
    }
}

/// The identity the driver's cross-dispatch chaining rests on (ISSUES.md P2
/// step 2).
///
/// A convolution writes its result as feature-atomic surfaces into scratch,
/// `compact_atomic_output` interleaves that into the dense IREE buffer, and
/// the next convolution repacks the dense buffer into surfaces again. When
/// the repack would reproduce the producer's scratch byte for byte, the
/// driver skips it and reads that scratch in place. Nothing at runtime can
/// check the claim, so it is pinned here: once that it holds, and once for
/// each way it stops holding, which is exactly the list
/// `rocket_core::layout::chain_identity` refuses on -- and each case asks
/// that pure verdict too, so the contract the compiler reads
/// (COMPILER_ROADMAP.md 6.1) is pinned to the bytes, not restated beside
/// them.
#[cfg(test)]
mod chain_identity_tests {
    use super::*;
    use rocket_core::layout::{
        ChainRefusal, CubeGeometry, CubeKind, chain_identity, cube_geometry,
    };

    fn conv(w: u32, h: u32, c: u32) -> CubeGeometry {
        cube_geometry(CubeKind::Conv, 2, w, h, c).unwrap()
    }

    fn pool(w: u32, h: u32, c: u32) -> CubeGeometry {
        cube_geometry(CubeKind::Pooling, 2, w, h, c).unwrap()
    }

    /// A cube with a distinct nonzero byte in every lane, padding channels
    /// included -- zeros there would hide the very mismatches these test.
    fn distinct_cube(pixels: usize, padded_bytes_per_pixel: usize) -> Vec<u8> {
        let surfaces = padded_bytes_per_pixel / FEATURE_ATOMIC_BYTES;
        (0..pixels * surfaces * FEATURE_ATOMIC_BYTES)
            .map(|index| (index % 251 + 1) as u8)
            .collect()
    }

    /// What the consumer would compact and repack, given its own geometry.
    fn compact(cube: &[u8], pixels: usize, bytes_per_pixel: usize) -> Vec<u8> {
        let mut dense = vec![0u8; pixels * bytes_per_pixel];
        let written = compact_atomic_output(
            cube,
            pixels,
            pixels,
            bytes_per_pixel,
            FEATURE_ATOMIC_BYTES,
            &mut dense,
        );
        assert_eq!(written, dense.len());
        dense
    }

    fn repack(dense: &[u8], pixels: usize, bytes_per_pixel: usize, packed: usize) -> Vec<u8> {
        let mut cube = vec![0u8; nc1hwc2_storage_size(pixels, packed).unwrap()];
        pack_nhwc_to_nc1hwc2_padded(dense, pixels, bytes_per_pixel, packed, &mut cube).unwrap();
        cube
    }

    fn round_trip(cube: &[u8], pixels: usize, bytes_per_pixel: usize, packed: usize) -> Vec<u8> {
        repack(
            &compact(cube, pixels, bytes_per_pixel),
            pixels,
            bytes_per_pixel,
            packed,
        )
    }

    #[test]
    fn a_whole_atom_cube_survives_compaction_and_repacking() {
        // 56x56 at Cin 64 fp16: ResNet50's conv1 -> conv2 edge, and the
        // shape class every chained edge in that model belongs to.
        const PIXELS: usize = 56 * 56;
        const BPP: usize = 64 * 2;
        let cube = distinct_cube(PIXELS, BPP);
        assert_eq!(round_trip(&cube, PIXELS, BPP, BPP), cube);
        let g = conv(56, 56, 64);
        assert_eq!((g.pixel_count, g.bytes_per_pixel), (PIXELS, BPP));
        assert_eq!(chain_identity(&g, &g), Ok(()));
    }

    /// The contract's storage size is the packer's, so a scratch sized by
    /// one is filled by the other.
    #[test]
    fn the_contract_sizes_the_cube_as_the_packer_does() {
        for (w, h, c) in [
            (56, 56, 64),
            (4, 4, 8),
            (4, 4, 20),
            (7, 7, 64),
            (1, 197, 768),
        ] {
            let g = conv(w, h, c);
            assert_eq!(
                g.storage_bytes().unwrap(),
                nc1hwc2_storage_size(g.surface_pixel_count, g.packed_bytes_per_pixel).unwrap()
            );
            let g = pool(w, h, c);
            assert_eq!(
                g.storage_bytes().unwrap(),
                nc1hwc2_storage_size(g.surface_pixel_count, g.packed_bytes_per_pixel).unwrap()
            );
        }
    }

    #[test]
    fn a_four_rounded_surface_stride_is_a_different_cube() {
        // The PPU strides its surfaces by the pixel count rounded up to
        // four. A 7x7 pool output therefore lives at stride 52 while a conv
        // or EW consumer of 49 pixels repacks at stride 49: compacting the
        // one and repacking as the other does not give the bytes back, which
        // is why `chainable_cube` compares surface strides and a pool only
        // chains at a multiple of four pixels. At 8x8 the strides agree.
        const BPP: usize = 64 * 2;
        let surfaces = BPP / FEATURE_ATOMIC_BYTES;
        let cube_52 = distinct_cube(52, BPP);
        let mut dense = vec![0u8; 49 * BPP];
        let written =
            compact_atomic_output(&cube_52, 52, 49, BPP, FEATURE_ATOMIC_BYTES, &mut dense);
        assert_eq!(written, dense.len());
        let repacked_49 = repack(&dense, 49, BPP, BPP);
        assert_eq!(repacked_49.len(), 49 * surfaces * FEATURE_ATOMIC_BYTES);
        assert_ne!(&cube_52[..repacked_49.len()], &repacked_49[..]);

        let cube_64 = distinct_cube(64, BPP);
        assert_eq!(round_trip(&cube_64, 64, BPP, BPP), cube_64);

        assert_eq!(
            chain_identity(&pool(7, 7, 64), &conv(7, 7, 64)),
            Err(ChainRefusal::SurfaceStride {
                producer: 52,
                consumer: 49,
            })
        );
        assert_eq!(chain_identity(&pool(8, 8, 64), &conv(8, 8, 64)), Ok(()));
    }

    #[test]
    fn a_producers_padding_surfaces_lie_outside_the_chained_region() {
        // A producer writes `padded_out_channels`, which can be wider than
        // the logical tensor. Those surfaces sit past everything the
        // consumer reads, so they cannot disturb the identity -- which is
        // why `chainable_cube` compares cube geometry and not scratch length.
        const PIXELS: usize = 8 * 8;
        const BPP: usize = 32 * 2;
        let cube = distinct_cube(PIXELS, BPP + 2 * FEATURE_ATOMIC_BYTES);
        assert_eq!(round_trip(&cube, PIXELS, BPP, BPP), cube[..PIXELS * BPP]);
        let producer = CubeGeometry {
            packed_bytes_per_pixel: BPP + 2 * FEATURE_ATOMIC_BYTES,
            ..conv(8, 8, 32)
        };
        assert_eq!(chain_identity(&producer, &conv(8, 8, 32)), Ok(()));
    }

    #[test]
    fn a_partial_trailing_atom_breaks_the_identity() {
        // Cin 20 fp16 is 40 bytes: two whole atoms and half of a third. A
        // repack zeroes that half; the producer left padding channels in it.
        const PIXELS: usize = 4 * 4;
        const BPP: usize = 20 * 2;
        let cube = distinct_cube(PIXELS, 3 * FEATURE_ATOMIC_BYTES);
        let repacked = round_trip(&cube, PIXELS, BPP, BPP);
        assert_ne!(repacked[..], cube[..repacked.len()]);
        assert_eq!(
            chain_identity(&conv(4, 4, 20), &conv(4, 4, 20)),
            Err(ChainRefusal::PartialAtom {
                bytes_per_pixel: BPP
            })
        );
    }

    #[test]
    fn a_consumer_that_pads_its_channels_breaks_the_identity() {
        // Cin 8 fp16 is 16 bytes logically and packs to 32: the second
        // surface is zero after a repack and is producer bytes here.
        const PIXELS: usize = 4 * 4;
        const BPP: usize = 8 * 2;
        const PACKED: usize = 16 * 2;
        let cube = distinct_cube(PIXELS, PACKED);
        assert_ne!(round_trip(&cube, PIXELS, BPP, PACKED), cube);
        let g = conv(4, 4, 8);
        assert_eq!((g.bytes_per_pixel, g.packed_bytes_per_pixel), (BPP, PACKED));
        assert_eq!(
            chain_identity(&g, &g),
            Err(ChainRefusal::ConsumerPadsChannels {
                bytes_per_pixel: BPP,
                packed_bytes_per_pixel: PACKED,
            })
        );
    }

    #[test]
    fn unequal_pixel_counts_read_the_wrong_surface() {
        // Surfaces are `pixel_count * 16` bytes apart, so a producer with
        // physical height padding (`fc.rs`'s padded row count) puts surface
        // 1 somewhere the consumer's geometry does not look. The bytes still
        // round-trip -- they are just the wrong bytes, which is why this is
        // checked against the producer's own dense result rather than
        // against the cube.
        const PRODUCER_PIXELS: usize = 8;
        const CONSUMER_PIXELS: usize = 4;
        const BPP: usize = 2 * FEATURE_ATOMIC_BYTES;
        let cube = distinct_cube(PRODUCER_PIXELS, BPP);
        let truth = compact(&cube, PRODUCER_PIXELS, BPP);
        let seen = compact(&cube, CONSUMER_PIXELS, BPP);
        assert_ne!(seen[..], truth[..seen.len()]);
        assert_eq!(
            chain_identity(&conv(8, 1, 16), &conv(4, 1, 16)),
            Err(ChainRefusal::PixelCount {
                producer: PRODUCER_PIXELS,
                consumer: CONSUMER_PIXELS,
            })
        );
    }
}
