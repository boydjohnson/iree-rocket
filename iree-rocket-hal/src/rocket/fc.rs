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

use crate::rocket::{
    builders::RegCmd,
    conv::{Activation, Buffers, ConvPlan, Precision},
};

/// The 1x1 spatial kernel used by every captured FC program.
pub const KERNELS: [usize; 2] = [1, 1];

/// Logical `[M,K] x [K,N] -> [M,N]` fully-connected shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub precision: Precision,
    pub activation: Activation,
}

impl Shape {
    /// Constructs an FC shape and validates it against the convolution
    /// builder's capture-backed channel limits.
    pub fn new(m: u32, k: u32, n: u32, precision: Precision) -> Shape {
        let shape = Shape {
            m,
            k,
            n,
            precision,
            activation: Activation::None,
        };
        let _ = shape.as_conv_shape();
        shape
    }

    /// Fuses an activation through the convolution builder's captured BN
    /// path.
    pub fn with_activation(mut self, activation: Activation) -> Shape {
        self.activation = activation;
        self
    }

    /// Returns the physical convolution shape observed in the vendor sweep.
    pub fn as_conv_shape(self) -> crate::rocket::conv::Shape {
        crate::rocket::conv::Shape::with_precision(self.m, 1, 1, self.k, self.n, self.precision)
            .with_padding([0, 0])
            .with_activation(self.activation)
    }
}

/// Standalone-job plan for a fully-connected operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    shape: Shape,
    conv: ConvPlan,
}

impl Plan {
    /// Plans the captured height-1, 1x1-convolution lowering.
    pub fn new(shape: Shape) -> Plan {
        Plan {
            shape,
            conv: ConvPlan::new(shape.as_conv_shape(), KERNELS),
        }
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
        conv::{Multiplier, Quantization},
    };

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
