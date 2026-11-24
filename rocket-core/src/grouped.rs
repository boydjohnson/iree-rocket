//! A grouped convolution as one job of dense per-channel-slice tasks.
//!
//! The compiler hands a grouped convolution to the NPU as the dense
//! convolution it equals, with a block-diagonal filter
//! (`rocket-block-diagonal-grouped-conv`): correct, but `G` times the
//! multiply-adds and filter bytes. On ResNeXt50 that was ~51 ms of NPU per
//! inference. A grouped convolution is independent dense convolutions over
//! channel slices, and in NC1HWC2 a slice of whole atoms is a contiguous run
//! of surfaces whose stride depends only on the pixel count -- so slice `s`
//! is the ordinary dense plan for `Cs -> Fs` channels with its input and
//! output bases moved `s * Cs` and `s * Fs` channels' worth of surfaces into
//! the full cubes, and every slice's tasks go in one job.
//!
//! Measured on RK3588 2026-11-24 (`grouped_conv_probe_hw.rs`), all 32 groups
//! of each ResNeXt stage, against the block-diagonal convolution: bit-
//! identical at every slicing, and 0.29-0.49 ms against 1.35-14.95 ms --
//! stage 4's 14x14 stride-2 site 34x faster.
//!
//! A slice is a run of whole groups, so the slice's own filter is block-
//! diagonal again whenever it holds more than one -- which is how a stage
//! with 4- or 8-channel groups reaches a whole-atom slice at all. How many
//! groups a slice holds is [`GroupedConv::slice_groups`]'s policy, which
//! the compile-time packer and the runtime driver both read here, so the
//! packed stream and the programs that index it cannot disagree.

use crate::{
    conv::{Kernels, Precision, Shape},
    error::{PlanError, PlanErrorCode},
    weights::WeightPlan,
};

/// Bytes in one feature atom.
const ATOM_BYTES: u32 = 16;

/// The widest group count [`detect_block_diagonal_groups`] looks for.
pub const MAX_DETECTED_GROUPS: u32 = 64;

/// A grouped convolution's slicing: the full dense shape, its kernel, and the
/// group count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupedConv {
    shape: Shape,
    kernels: Kernels,
    groups: u32,
}

impl GroupedConv {
    /// Validates `shape` (the full dense convolution) split into `groups`.
    ///
    /// fp16, dense, every slice a whole number of input atoms and of the
    /// output-channel granule, so slices tile both cubes exactly. Anything
    /// else is refused and the caller keeps the block-diagonal dense
    /// convolution.
    pub fn try_new(shape: Shape, kernels: Kernels, groups: u32) -> Result<GroupedConv, PlanError> {
        let refuse = |message: String| {
            Err(PlanError::new(
                PlanErrorCode::UnvalidatedConfiguration,
                message,
            ))
        };
        if groups < 2 {
            return refuse(format!(
                "a grouped convolution needs at least two groups, got {groups}"
            ));
        }
        if shape.precision != Precision::Fp16 || shape.depthwise {
            return refuse("grouped slicing is measured for dense fp16 only".to_string());
        }
        if !shape.in_channels.is_multiple_of(groups) || !shape.out_channels.is_multiple_of(groups) {
            return refuse(format!(
                "{} -> {} channels do not divide into {groups} groups",
                shape.in_channels, shape.out_channels
            ));
        }
        let grouped = GroupedConv {
            shape,
            kernels,
            groups,
        };
        let slice = grouped.slice_shape();
        let element = shape.precision.element_bytes();
        if !(slice.in_channels * element).is_multiple_of(ATOM_BYTES) {
            return refuse(format!(
                "a slice's {} input channels are not whole atoms",
                slice.in_channels
            ));
        }
        if slice.padded_out_channels() != slice.out_channels {
            return refuse(format!(
                "a slice's {} output channels pad to {}; the padding would land on the next slice",
                slice.out_channels,
                slice.padded_out_channels()
            ));
        }
        Ok(grouped)
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }

    pub fn groups(&self) -> u32 {
        self.groups
    }

    /// Groups per slice: the fewest whole groups that reach a slice of 32
    /// input channels, or 64 when the convolution is at least 512 wide. Read
    /// off the probe: 32-64-channel slices were fastest at every ResNeXt
    /// stage (16-channel ones paid ~2x per slice), and at 512 and 1024
    /// channels 64 beat 32 (stage 4 7x7: 0.36 against 0.65 ms).
    pub fn slice_groups(&self) -> u32 {
        let per_group = self.shape.in_channels / self.groups;
        let target = if self.shape.in_channels >= 512 {
            64
        } else {
            32
        };
        let mut slice_groups = (target / per_group).max(1);
        while !self.groups.is_multiple_of(slice_groups) {
            slice_groups -= 1;
        }
        slice_groups
    }

    pub fn slices(&self) -> u32 {
        self.groups / self.slice_groups()
    }

    /// One slice's dense convolution: the same extents, stride, padding,
    /// precision and activation over that slice's channels.
    pub fn slice_shape(&self) -> Shape {
        let slice_groups = self.slice_groups();
        Shape {
            in_channels: self.shape.in_channels / self.groups * slice_groups,
            out_channels: self.shape.out_channels / self.groups * slice_groups,
            ..self.shape
        }
    }

    /// Surfaces into the input cube slice `slice` starts at.
    pub fn input_surface(&self, slice: u32) -> u32 {
        slice * self.slice_shape().in_channels * self.shape.precision.element_bytes() / ATOM_BYTES
    }

    /// Surfaces into the output cube slice `slice` starts at.
    pub fn output_surface(&self, slice: u32) -> u32 {
        slice * self.slice_shape().out_channels * self.shape.precision.output_element_bytes()
            / ATOM_BYTES
    }

    /// First output channel of slice `slice` -- where its bias starts.
    pub fn output_channel(&self, slice: u32) -> u32 {
        slice * self.slice_shape().out_channels
    }

    /// One slice's coefficient plan; every slice has the same.
    pub fn slice_weight_plan(&self) -> WeightPlan {
        WeightPlan::for_conv(self.slice_shape(), self.kernels)
    }

    /// Bytes of one slice's packed coefficients: slice `s` starts at
    /// `s * slice_packed_bytes()` in the stream [`GroupedConv::pack`] writes.
    pub fn slice_packed_bytes(&self) -> Result<usize, PlanError> {
        self.slice_weight_plan().packed_bytes()
    }

    /// The whole packed stream: every slice's, back to back.
    pub fn packed_bytes(&self) -> Result<usize, PlanError> {
        Ok(self.slice_packed_bytes()? * self.slices() as usize)
    }

    /// Packs the full dense HWCF filter (block-diagonal across groups) slice
    /// by slice: each slice's `[kh, kw, Cs, Fs]` diagonal block through the
    /// ordinary dense packer, back to back.
    pub fn pack(&self, dense: &[u8], packed: &mut [u8]) -> Result<usize, PlanError> {
        let element = self.shape.precision.element_bytes() as usize;
        let [kh, kw] = self.kernels;
        let (cin, cout) = (
            self.shape.in_channels as usize,
            self.shape.out_channels as usize,
        );
        let expected = kh * kw * cin * cout * element;
        if dense.len() != expected {
            return Err(PlanError::new(
                PlanErrorCode::InvalidShape,
                format!(
                    "dense filter is {} bytes, the shape needs {expected}",
                    dense.len()
                ),
            ));
        }
        let total = self.packed_bytes()?;
        if packed.len() < total {
            return Err(PlanError::new(
                PlanErrorCode::InvalidShape,
                format!(
                    "packed buffer is {} bytes, the layout needs {total}",
                    packed.len()
                ),
            ));
        }
        let slice = self.slice_shape();
        let (cs, fs) = (slice.in_channels as usize, slice.out_channels as usize);
        let plan = self.slice_weight_plan();
        let slice_bytes = plan.packed_bytes()?;
        let mut block = vec![0u8; kh * kw * cs * fs * element];
        for s in 0..self.slices() as usize {
            for tap in 0..kh * kw {
                for i in 0..cs {
                    let from = ((tap * cin + s * cs + i) * cout + s * fs) * element;
                    let to = (tap * cs + i) * fs * element;
                    block[to..to + fs * element].copy_from_slice(&dense[from..from + fs * element]);
                }
            }
            plan.pack(&block, &mut packed[s * slice_bytes..(s + 1) * slice_bytes])?;
        }
        Ok(total)
    }
}

/// The largest group count, up to [`MAX_DETECTED_GROUPS`], for which the
/// dense HWCF filter `[kh, kw, cin, cout]` (`element` bytes each) is zero
/// wherever an output channel's group differs from an input channel's -- 1
/// when no such split exists.
///
/// Always safe to act on: a dense filter that happens to be zero off its
/// diagonal blocks computes exactly the same thing grouped as dense.
pub fn detect_block_diagonal_groups(
    dense: &[u8],
    kernels: Kernels,
    cin: usize,
    cout: usize,
    element: usize,
) -> u32 {
    let [kh, kw] = kernels;
    if dense.len() != kh * kw * cin * cout * element {
        return 1;
    }
    let zero_off_blocks = |groups: usize| {
        let (cg, fg) = (cin / groups, cout / groups);
        (0..kh * kw).all(|tap| {
            (0..cin).all(|i| {
                let row = (tap * cin + i) * cout * element;
                let own = i / cg;
                (0..cout).all(|o| {
                    o / fg == own
                        || dense[row + o * element..row + (o + 1) * element]
                            .iter()
                            .all(|byte| *byte == 0)
                })
            })
        })
    };
    (2..=MAX_DETECTED_GROUPS as usize)
        .rev()
        .filter(|groups| cin.is_multiple_of(*groups) && cout.is_multiple_of(*groups))
        .find(|&groups| zero_off_blocks(groups))
        .map_or(1, |groups| groups as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fp16(side: u32, stride: u32, c: u32, f: u32) -> Shape {
        Shape::try_with_precision(side, side, stride, c, f, Precision::Fp16)
            .unwrap()
            .try_with_padding([1, 1])
            .unwrap()
    }

    #[test]
    fn resnext_stages_slice_to_whole_atoms() {
        for (c, slice_channels, slices) in
            [(128, 32, 4), (256, 32, 8), (512, 64, 8), (1024, 64, 16)]
        {
            let grouped = GroupedConv::try_new(fp16(14, 1, c, c), [3, 3], 32).unwrap();
            assert_eq!(grouped.slice_shape().in_channels, slice_channels, "C {c}");
            assert_eq!(grouped.slices(), slices, "C {c}");
            assert_eq!(grouped.input_surface(1), slice_channels * 2 / 16);
            assert_eq!(
                grouped.output_surface(slices - 1),
                (slices - 1) * slice_channels * 2 / 16
            );
            assert_eq!(grouped.output_channel(2), 2 * slice_channels);
        }
    }

    #[test]
    fn refuses_what_does_not_tile_the_cubes() {
        assert!(GroupedConv::try_new(fp16(8, 1, 64, 64), [3, 3], 1).is_err());
        assert!(GroupedConv::try_new(fp16(8, 1, 60, 64), [3, 3], 32).is_err());
        // 2 groups of 4 channels in, 4 out: an 8-channel slice pads to 16.
        assert!(GroupedConv::try_new(fp16(8, 1, 8, 8), [3, 3], 2).is_err());
    }

    fn block_diagonal(kernels: Kernels, c: usize, f: usize, groups: usize) -> Vec<u8> {
        let [kh, kw] = kernels;
        let (cg, fg) = (c / groups, f / groups);
        let mut dense = vec![0u8; kh * kw * c * f * 2];
        for tap in 0..kh * kw {
            for i in 0..c {
                for o in 0..f {
                    if i / cg == o / fg {
                        let at = ((tap * c + i) * f + o) * 2;
                        let value = ((tap * 31 + i * 7 + o * 3) % 251 + 1) as u16;
                        dense[at..at + 2].copy_from_slice(&value.to_le_bytes());
                    }
                }
            }
        }
        dense
    }

    #[test]
    fn detects_the_finest_block_structure() {
        let dense = block_diagonal([3, 3], 128, 128, 32);
        assert_eq!(
            detect_block_diagonal_groups(&dense, [3, 3], 128, 128, 2),
            32
        );
        let coarse = block_diagonal([3, 3], 128, 128, 4);
        assert_eq!(
            detect_block_diagonal_groups(&coarse, [3, 3], 128, 128, 2),
            4
        );
        let full = block_diagonal([3, 3], 128, 128, 1);
        assert_eq!(detect_block_diagonal_groups(&full, [3, 3], 128, 128, 2), 1);
    }

    #[test]
    fn packs_each_slice_as_its_own_dense_filter() {
        let (c, f) = (128usize, 128usize);
        let shape = fp16(8, 1, c as u32, f as u32);
        let grouped = GroupedConv::try_new(shape, [3, 3], 32).unwrap();
        let dense = block_diagonal([3, 3], c, f, 32);
        let mut packed = vec![0u8; grouped.packed_bytes().unwrap()];
        assert_eq!(grouped.pack(&dense, &mut packed).unwrap(), packed.len());
        // Slice 1 packed on its own must equal the second chunk of the stream.
        let slice = grouped.slice_shape();
        let (cs, fs) = (slice.in_channels as usize, slice.out_channels as usize);
        let mut block = vec![0u8; 9 * cs * fs * 2];
        for tap in 0..9 {
            for i in 0..cs {
                for o in 0..fs {
                    let from = ((tap * c + cs + i) * f + fs + o) * 2;
                    let to = ((tap * cs + i) * fs + o) * 2;
                    block[to..to + 2].copy_from_slice(&dense[from..from + 2]);
                }
            }
        }
        let plan = grouped.slice_weight_plan();
        let mut alone = vec![0u8; plan.packed_bytes().unwrap()];
        plan.pack(&block, &mut alone).unwrap();
        let bytes = grouped.slice_packed_bytes().unwrap();
        assert_eq!(&packed[bytes..2 * bytes], alone.as_slice());
    }
}
