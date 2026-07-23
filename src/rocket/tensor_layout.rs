//! Host-side transforms between dense NHWC tensors and the NPU's
//! feature-atomic NC1HWC2 layout.
//!
//! The hardware's inner channel block is 16 bytes, not a fixed number of
//! elements: C2 is 16 for int8 and 8 for fp16. Each C1 surface contains
//! every HxW pixel before the next channel block begins.

/// Physical byte width of one NC1HWC2 inner-channel block.
pub const FEATURE_ATOMIC_BYTES: usize = 16;

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
}
