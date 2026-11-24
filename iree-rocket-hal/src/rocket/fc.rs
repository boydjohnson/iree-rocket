//! Capture-derived fully-connected lowering.
//!
//! RKNN Toolkit 2.3.2 was swept over 160 ONNX `Linear` models spanning
//! `[M,K] x [K,N]`, both fp16 and int8, with M from 1 through 32 and K/N
//! reaching 4096. Every one lowers FC as an ordinary 1x1 convolution:
//!
//! - M is the convolution width and the physical height is exactly one;
//! - K and N are the input and output channel counts;
//! - all three `CNA_FC_CON0/1/2` words are zero in all 666 programs;
//! - odd int8 N is rounded to an even programmed kernel count, exactly as
//!   [`crate::rocket::conv::Shape::programmed_kernels`] already does.
//!
//! Consequently FC needs no register builder of its own. [`Plan`] fixes the
//! captured shape mapping and delegates planning and emission to
//! [`ConvPlan`]. In particular, it does not enable the CNA zero-skipping
//! path: that path exists in the TRM, but it is not what this vendor
//! toolchain emits for any measured Dense/FC shape.
//!
//! Hardware validation on an RK3588 confirms this height-one lowering for
//! both an exact fp16 M=7/K=16/N=32 result and an int8 M=7/K=16/N=33 result.
//! The latter also exercises the captured odd-N kernel padding.
//!
//! The sweep's widest M is 32, and past it the geometry has a hardware
//! bound the corpus could not show: an input row is held in the CBUF in
//! 32-channel slabs whose base offset is 11 bits, so `(K/32 - 1) * M` must
//! stay at or below 2047 or the last slab is read from the front of the
//! line. [`ConvPlan`] enforces that through `Shape::max_tile_input_width`
//! and splits a wider M into column tiles; measured exact at M 90, 128, 197
//! and 296 on `planck` 2026-09-05 (ISSUES.md C10). The transposed `1 x M`
//! geometry `rockchip-npu-notes` uses row-tiles instead and was exact at the
//! same points; which of the two is the better lowering above 32 is open.

use crate::rocket::{
    builders::{
        Bits, RegCmd, Register, RegisterMeta,
        cna::{CnaConvCon1, CnaDmaCon1, CnaDmaCon2, CnaFeatureDataAddr},
        dpu::{DpuDataCubeNotchAddr, DpuDstBaseAddr, DpuDstSurfStride, DpuSurfaceAdd},
    },
    conv::{Buffers, ConvPlan, PlanError},
};

// The shape mapping itself lives in `rocket-core` (COMPILER_ROADMAP.md
// section 1); this module keeps the register emission and the hardware
// record above.
pub use rocket_core::fc::{KERNELS, Shape};

/// Standalone-job plan for a fully-connected operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    shape: Shape,
    conv: ConvPlan,
    row_major_output: bool,
}

impl Plan {
    /// Plans the captured height-1, 1x1-convolution lowering, panicking on
    /// a refusal.
    pub fn new(shape: Shape) -> Plan {
        Plan::try_new(shape).unwrap_or_else(|error| panic!("{error}"))
    }

    /// [`Plan::new`], returning the planner's refusal instead of panicking.
    pub fn try_new(shape: Shape) -> Result<Plan, PlanError> {
        Ok(Plan {
            shape,
            conv: ConvPlan::try_new(shape.try_as_conv_shape()?, KERNELS)?,
            row_major_output: false,
        })
    }

    /// Plans the width-1 lowering whose output the DPU writes straight into
    /// a row-major `[M, N]` buffer, or `None` where the shape does not
    /// support it -- see `rocket_core::fc::Plan::try_new_row_major_output`.
    ///
    /// The DPU has no destination line stride, but it skips
    /// `DPU_DATA_CUBE_NOTCH_ADDR` atoms at the end of every output line, and
    /// at width 1 every line is one output row. So with the surface stride
    /// at one atom and the notch at `N * 2 / 16 - 1`, channel group `c1` of
    /// row `m` lands at byte `m * N * 2 + c1 * 16` -- row-major, with no
    /// compaction pass. Measured on RK3588 2026-11-24
    /// (`matmul_dense_output_probe_hw.rs`): bit-identical to the NC1HWC2
    /// lowering on every BLIP matmul shape, nothing written past the dense
    /// extent, and `DPU_SURFACE_ADD` must be 2 (the fp16 writer's rule,
    /// twice the surface stride) -- 0, 1, 4 and the planner's whole-image
    /// value all misplace channel groups.
    pub fn try_new_row_major_output(shape: Shape) -> Option<Plan> {
        let core = rocket_core::fc::Plan::try_new_row_major_output(shape)?;
        Some(Plan {
            shape,
            conv: ConvPlan::from(core.conv_plan().clone()),
            row_major_output: true,
        })
    }

    /// Whether [`Plan::place_row_major_output`] applies to this plan's
    /// programs.
    pub fn row_major_output(&self) -> bool {
        self.row_major_output
    }

    /// Points tile `tile`'s already-relocated program at its rows of a
    /// row-major `[M, N]` output starting at `output`: the destination base
    /// moves to the tile's first row, and the surface stride, notch and
    /// surface add become the row-major writer's.
    ///
    /// Panics on a plan that is not [`Plan::row_major_output`], or when the
    /// program does not carry exactly one of each register.
    pub fn place_row_major_output(&self, program: &mut [RegCmd], tile: usize, output: u32) {
        assert!(
            self.row_major_output,
            "place_row_major_output needs a plan from Plan::try_new_row_major_output"
        );
        let row_bytes = self.shape.n * self.shape.precision.element_bytes();
        let notch = row_bytes / FEATURE_ATOM_BYTES - 1;
        let first_row = self.conv.tiles()[tile].rows.out_first;
        let base = output + first_row * row_bytes;
        assert_eq!(
            base % FEATURE_ATOM_BYTES,
            0,
            "row-major output must be 16-byte aligned"
        );
        replace_one::<DpuDstBaseAddr>(
            program,
            Register::<DpuDstBaseAddr>::new()
                .dst_base_addr(Bits::new(base))
                .build(),
        );
        replace_one::<DpuDstSurfStride>(
            program,
            Register::<DpuDstSurfStride>::new()
                .dst_surf_stride(Bits::new(1))
                .build(),
        );
        replace_one::<DpuDataCubeNotchAddr>(
            program,
            Register::<DpuDataCubeNotchAddr>::new()
                .notch_addr_0(Bits::new(notch))
                .notch_addr_1(Bits::new(notch))
                .build(),
        );
        replace_one::<DpuSurfaceAdd>(
            program,
            Register::<DpuSurfaceAdd>::new()
                .surf_add(Bits::new(2))
                .build(),
        );
    }

    /// Points tile `tile`'s already-relocated program at its rows of a
    /// row-major `[M, K]` input starting at `input`, so the driver copies the
    /// input instead of packing it into an NC1HWC2 cube.
    ///
    /// The width-1 lowering reads one pixel per line, so a line is one input
    /// row of `K * 2 / 16` atoms and a channel group is one atom. Measured on
    /// RK3588 2026-11-24 (`matmul_dense_input_probe`): grouped line fetch
    /// off, `line_stride` one row, `surf_stride` 0 -- the planner's
    /// horizontal-tile rule `surface_pitch - input_width` with a one-atom
    /// surface and a one-pixel line -- is bit-identical to the packed input
    /// from K 32 to 3072 and costs 1-8% NPU time. Grouped fetch with
    /// `line = 4 * Kc`, `surf = 1 - 4 * Kc` (the packed rule transcribed) and
    /// two other encodings hang the NPU.
    ///
    /// Panics on a plan that is not [`Plan::row_major_output`] or a shape
    /// without [`Shape::supports_row_major_input`].
    pub fn place_row_major_input(&self, program: &mut [RegCmd], tile: usize, input: u32) {
        assert!(
            self.row_major_output && self.shape.supports_row_major_input(),
            "place_row_major_input needs a row-major plan whose K is whole atoms"
        );
        let row_bytes = self.shape.k * self.shape.precision.element_bytes();
        let first_row = self.conv.tiles()[tile].rows.in_first;
        let base = input + first_row * row_bytes;
        assert_eq!(
            base % FEATURE_ATOM_BYTES,
            0,
            "row-major input must be 16-byte aligned"
        );
        replace_one::<CnaFeatureDataAddr>(
            program,
            Register::<CnaFeatureDataAddr>::new()
                .feature_base_addr(Bits::new(base))
                .build(),
        );
        replace_one::<CnaDmaCon1>(
            program,
            Register::<CnaDmaCon1>::new()
                .line_stride(Bits::new(row_bytes / FEATURE_ATOM_BYTES))
                .build(),
        );
        replace_one::<CnaDmaCon2>(
            program,
            Register::<CnaDmaCon2>::new()
                .surf_stride(Bits::new(0))
                .build(),
        );
        // The program writes CNA_CONV_CON1 more than once; every write
        // carries the flag, as in the probe that measured it.
        let mut written = 0;
        for slot in program.iter_mut().filter(|command| {
            (command.0 >> 48) as u32 == CnaConvCon1::DOMAIN
                && command.0 as u32 & 0xffff == CnaConvCon1::OFFSET
        }) {
            let value = (slot.0 >> 16) as u32 | GROUP_LINE_OFF;
            *slot = RegCmd::new(CnaConvCon1::DOMAIN, CnaConvCon1::OFFSET, value);
            written += 1;
        }
        assert!(written > 0, "CNA_CONV_CON1 is present");
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }

    /// Exposes the underlying convolution plan for common storage and
    /// scheduling queries.
    pub fn conv_plan(&self) -> &ConvPlan {
        &self.conv
    }

    /// Emits relocatable programs, one per planned standalone job.
    pub fn programs(&self) -> Vec<Vec<RegCmd>> {
        self.conv.programs()
    }

    /// Emits programs with their four DMA buffer addresses bound.
    pub fn programs_with_buffers(&self, buffers: Buffers) -> Vec<Vec<RegCmd>> {
        self.conv.programs_with_buffers(buffers)
    }
}

const FEATURE_ATOM_BYTES: u32 = 16;

/// `CNA_CONV_CON1.group_line_off`.
const GROUP_LINE_OFF: u32 = 0x2000_0000;

fn replace_one<R: RegisterMeta>(program: &mut [RegCmd], value: RegCmd) {
    let mut matches = program.iter_mut().filter(|command| {
        (command.0 >> 48) as u32 == R::DOMAIN && command.0 as u32 & 0xffff == R::OFFSET
    });
    let slot = matches
        .next()
        .unwrap_or_else(|| panic!("register {:#x}:{:#x} is absent", R::DOMAIN, R::OFFSET));
    *slot = value;
    assert!(
        matches.next().is_none(),
        "register {:#x}:{:#x} appears twice",
        R::DOMAIN,
        R::OFFSET
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rocket::{
        builders::{
            RegisterMeta,
            cna::{
                CnaCbufCon0, CnaCbufCon1, CnaDataSize0, CnaDataSize1, CnaDataSize2, CnaDataSize3,
                CnaDmaCon1, CnaDmaCon2, CnaFcCon0, CnaFcCon1, CnaFcCon2, CnaFcDataSize0,
                CnaFcDataSize1, CnaWeightSize0, CnaWeightSize1, CnaWeightSize2,
            },
        },
        conv::{Multiplier, Precision, Quantization},
    };

    #[test]
    fn row_major_output_rewrites_only_the_dpu_writer() {
        let shape = Shape::new(577, 768, 3072, Precision::Fp16);
        let plan = Plan::try_new_row_major_output(shape).unwrap();
        assert!(plan.row_major_output());
        let buffers = Buffers {
            input: 0x10_0000,
            weights: 0x20_0000,
            bias: 0x30_0000,
            output: 0x40_0000,
        };
        let programs = plan.programs_with_buffers(buffers);
        assert!(programs.len() > 1, "577 rows tile");
        let mut rows_seen = 0;
        for (tile, original) in programs.iter().enumerate() {
            let mut program: Vec<RegCmd> =
                original.iter().map(|command| RegCmd(command.0)).collect();
            plan.place_row_major_output(&mut program, tile, buffers.output);
            let first_row = plan.conv_plan().tiles()[tile].rows.out_first;
            assert_eq!(first_row, rows_seen);
            rows_seen += plan.conv_plan().tiles()[tile].rows.out_rows;
            assert_eq!(
                value_of::<DpuDstBaseAddr>(&program),
                0x40_0000 + first_row * 3072 * 2
            );
            assert_eq!(value_of::<DpuDstSurfStride>(&program), 16);
            assert_eq!(
                value_of::<DpuDataCubeNotchAddr>(&program),
                (383 << 16) | 383
            );
            let changed = original
                .iter()
                .zip(program.iter())
                .filter(|(before, after)| before.0 != after.0)
                .count();
            assert!(changed <= 4, "only the four writer registers move");
        }
        assert_eq!(rows_seen, 577);
        let mut program: Vec<RegCmd> = programs[1].iter().map(|c| RegCmd(c.0)).collect();
        plan.place_row_major_input(&mut program, 1, buffers.input);
        let first_row = plan.conv_plan().tiles()[1].rows.in_first;
        assert_eq!(
            value_of::<CnaFeatureDataAddr>(&program),
            0x10_0000 + first_row * 768 * 2
        );
        assert_eq!(value_of::<CnaDmaCon1>(&program), 96);
        assert_eq!(value_of::<CnaDmaCon2>(&program), 0);
        assert_ne!(value_of::<CnaConvCon1>(&program) & GROUP_LINE_OFF, 0);
        assert!(Plan::try_new_row_major_output(Shape::new(16, 768, 12, Precision::Fp16)).is_none());
    }

    fn value_of<R: RegisterMeta>(program: &[RegCmd]) -> u32 {
        let values = program
            .iter()
            .filter_map(|command| {
                let domain = (command.0 >> 48) as u32;
                let offset = command.0 as u32 & 0xffff;
                (domain == R::DOMAIN && offset == R::OFFSET).then_some((command.0 >> 16) as u32)
            })
            .collect::<Vec<_>>();
        assert!(!values.is_empty(), "register is absent from FC program");
        assert!(
            values.windows(2).all(|pair| pair[0] == pair[1]),
            "register has inconsistent repeated values: {values:?}"
        );
        values[0]
    }

    fn int8() -> Precision {
        Precision::Int8(Quantization {
            input_zero_point: 0,
            output_zero_point: 0,
            weight_zero_point: 0,
            input_scale: 1.0,
            weights_scale: 1.0,
            multiplier: Multiplier::for_unit_bs(1.0),
        })
    }

    #[test]
    fn maps_m_k_n_to_captured_height_one_convolution() {
        for precision in [Precision::Fp16, int8()] {
            let fc = Shape::new(7, 16, 32, precision);
            let conv = fc.as_conv_shape();
            assert_eq!(conv.width, 7);
            assert_eq!(conv.height, 1);
            assert_eq!(conv.in_channels, 16);
            assert_eq!(conv.out_channels, 32);
            assert_eq!(conv.output_width(KERNELS), 7);
            assert_eq!(conv.output_height(KERNELS), 1);
            assert_eq!(conv.padding, Some([0, 0]));
        }
    }

    #[test]
    fn representative_fp16_geometry_matches_vendor_capture() {
        // fc-m4-k16-n32.rknn, single-core plan 0. These are the complete CNA
        // geometry, coefficient-size and FC-control words that distinguish
        // the lowering from an ordinary image-shaped convolution.
        let program = Plan::new(Shape::new(4, 16, 32, Precision::Fp16))
            .programs()
            .remove(0);
        for (actual, expected, name) in [
            (value_of::<CnaCbufCon0>(&program), 0x0000_00b1, "cbuf0"),
            (value_of::<CnaCbufCon1>(&program), 0x0000_0002, "cbuf1"),
            (value_of::<CnaDataSize0>(&program), 0x0004_0001, "data0"),
            (value_of::<CnaDataSize1>(&program), 0x000f_0010, "data1"),
            (value_of::<CnaDataSize2>(&program), 0x0000_0004, "data2"),
            (value_of::<CnaDataSize3>(&program), 0x0000_0004, "data3"),
            (value_of::<CnaWeightSize0>(&program), 0x0000_0400, "weight0"),
            (value_of::<CnaWeightSize1>(&program), 0x0000_0020, "weight1"),
            (value_of::<CnaWeightSize2>(&program), 0x0101_0020, "weight2"),
            (value_of::<CnaDmaCon1>(&program), 0x0000_0010, "dma1"),
            (value_of::<CnaDmaCon2>(&program), 0x0fff_fff4, "dma2"),
            (
                value_of::<CnaFcDataSize0>(&program),
                0x0004_0001,
                "fc-data0",
            ),
            (
                value_of::<CnaFcDataSize1>(&program),
                0x0000_0010,
                "fc-data1",
            ),
        ] {
            assert_eq!(actual, expected, "{name}");
        }
        assert_eq!(value_of::<CnaFcCon0>(&program), 0);
        assert_eq!(value_of::<CnaFcCon1>(&program), 0);
        assert_eq!(value_of::<CnaFcCon2>(&program), 0);
    }

    #[test]
    fn representative_int8_sizes_match_vendor_capture() {
        // fc-m4-k16-n32-i8.rknn differs in coefficient bytes and CBUF data
        // entries, while retaining the same height-1 mapping and zeroed FC
        // controls.
        let program = Plan::new(Shape::new(4, 16, 32, int8()))
            .programs()
            .remove(0);
        assert_eq!(value_of::<CnaCbufCon0>(&program), 0x0000_00b1);
        assert_eq!(value_of::<CnaCbufCon1>(&program), 1);
        assert_eq!(value_of::<CnaWeightSize0>(&program), 512);
        assert_eq!(value_of::<CnaWeightSize1>(&program), 16);
        assert_eq!(value_of::<CnaWeightSize2>(&program), 0x0101_0020);
        assert_eq!(value_of::<CnaDmaCon2>(&program), 0x0fff_fff4);
        assert_eq!(value_of::<CnaFcCon0>(&program), 0);
        assert_eq!(value_of::<CnaFcCon1>(&program), 0);
        assert_eq!(value_of::<CnaFcCon2>(&program), 0);
    }

    #[test]
    fn odd_int8_n_uses_the_captured_even_kernel_count() {
        let conv = Shape::new(1, 64, 33, int8()).as_conv_shape();
        assert_eq!(conv.programmed_kernels(), 34);
        let program = Plan::new(Shape::new(1, 64, 33, int8()))
            .programs()
            .remove(0);
        assert_eq!(value_of::<CnaWeightSize2>(&program) & 0x3fff, 34);
    }

    #[test]
    fn supported_vendor_sweep_shapes_all_use_the_height_one_mapping() {
        let boundary = [
            1, 2, 3, 4, 7, 8, 15, 16, 17, 24, 31, 32, 33, 48, 63, 64, 65, 96, 127, 128, 129, 192,
            255, 256, 257, 384, 512,
        ];
        let mut shapes = [
            (1, 16, 32),
            (1, 32, 16),
            (1, 128, 256),
            (1, 256, 128),
            (1, 256, 256),
            (1, 512, 128),
            (1, 128, 512),
            (4, 16, 32),
            (7, 16, 32),
            (16, 256, 256),
        ]
        .to_vec();
        shapes.extend(
            [1, 2, 3, 4, 7, 8, 15, 16, 31, 32]
                .into_iter()
                .map(|m| (m, 64, 64)),
        );
        shapes.extend(boundary.into_iter().map(|k| (1, k, 64)));
        shapes.extend(boundary.into_iter().map(|n| (1, 64, n)));
        shapes.sort_unstable();
        shapes.dedup();

        for precision in [Precision::Fp16, int8()] {
            for &(m, k, n) in &shapes {
                let fc = Shape::new(m, k, n, precision);
                let conv = fc.as_conv_shape();
                let programs = Plan::new(fc).programs();
                assert_eq!(programs.len(), 1, "M={m} K={k} N={n} {precision:?}");
                let program = &programs[0];
                assert_eq!(value_of::<CnaDataSize0>(program), (m << 16) | 1);
                assert_eq!(value_of::<CnaDataSize2>(program), m);
                assert_eq!(value_of::<CnaDataSize3>(program), m);
                assert_eq!(
                    value_of::<CnaWeightSize2>(program) & 0x3fff,
                    conv.programmed_kernels()
                );
                assert_eq!(value_of::<CnaFcCon0>(program), 0);
                assert_eq!(value_of::<CnaFcCon1>(program), 0);
                assert_eq!(value_of::<CnaFcCon2>(program), 0);
                let expected_surface_stride = if k <= 4 {
                    0
                } else {
                    m.wrapping_mul(1u32.wrapping_sub(4)) & 0x0fff_ffff
                };
                assert_eq!(
                    value_of::<CnaDmaCon2>(program),
                    expected_surface_stride,
                    "M={m} K={k} N={n} {precision:?}"
                );
            }
        }
    }

    #[test]
    fn bound_programs_relocate_all_four_fc_buffers() {
        let plan = Plan::new(Shape::new(4, 16, 32, Precision::Fp16));
        let programs = plan.programs_with_buffers(Buffers {
            input: 0x1000,
            weights: 0x2000,
            bias: 0x3000,
            output: 0x4000,
        });
        assert_eq!(programs.len(), plan.conv_plan().tiles().len());
    }
}
