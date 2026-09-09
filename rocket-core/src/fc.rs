//! Capture-derived fully-connected lowering: the shape mapping only.
//!
//! RKNN Toolkit 2.3.2 was swept over 160 ONNX `Linear` models spanning
//! `[M,K] x [K,N]`, both fp16 and int8, with M from 1 through 32 and K/N
//! reaching 4096. Every one lowers FC as an ordinary 1x1 convolution:
//!
//! - M is the convolution width and the physical height is exactly one;
//! - K and N are the input and output channel counts;
//! - all three `CNA_FC_CON0/1/2` words are zero in all 666 programs;
//! - odd int8 N is rounded to an even programmed kernel count, exactly as
//!   [`crate::conv::Shape::programmed_kernels`] already does.
//!
//! Consequently FC needs no planner of its own. [`Shape::as_conv_shape`]
//! fixes the captured mapping and [`Plan`] delegates to [`ConvPlan`]. The
//! register program is the HAL's (`iree_rocket_hal::rocket::fc`), which
//! also carries the hardware-validation record and the corpus's register
//! tests.
//!
//! The sweep's widest M is 32, and past it the geometry has a hardware
//! bound the corpus could not show: an input row is held in the CBUF in
//! 32-channel slabs whose base offset is 11 bits, so `(K/32 - 1) * M` must
//! stay at or below 2047 or the last slab is read from the front of the
//! line. [`ConvPlan`] enforces that through `Shape::max_tile_input_width`
//! and splits a wider M into column tiles; measured exact at M 90, 128, 197
//! and 296 on `planck` 2026-09-05 (ISSUES.md C10).

use crate::{
    conv::{self, Activation, ConvPlan, Precision},
    error::PlanError,
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
    /// planner's capture-backed channel limits, panicking on a refusal.
    pub fn new(m: u32, k: u32, n: u32, precision: Precision) -> Shape {
        Shape::try_new(m, k, n, precision).unwrap_or_else(|error| panic!("{error}"))
    }

    /// [`Shape::new`], returning the planner's refusal instead of panicking.
    pub fn try_new(m: u32, k: u32, n: u32, precision: Precision) -> Result<Shape, PlanError> {
        let shape = Shape {
            m,
            k,
            n,
            precision,
            activation: Activation::None,
        };
        shape.try_as_conv_shape()?;
        Ok(shape)
    }

    /// Fuses an activation through the convolution builder's captured BN
    /// path.
    pub fn with_activation(mut self, activation: Activation) -> Shape {
        self.activation = activation;
        self
    }

    /// Returns the physical convolution shape observed in the vendor sweep.
    pub fn as_conv_shape(self) -> conv::Shape {
        self.try_as_conv_shape()
            .unwrap_or_else(|error| panic!("{error}"))
    }

    /// [`Shape::as_conv_shape`], returning the planner's refusal instead of
    /// panicking. Every field is validated here, including an activation
    /// set after construction.
    pub fn try_as_conv_shape(self) -> Result<conv::Shape, PlanError> {
        conv::Shape::try_with_precision(self.m, 1, 1, self.k, self.n, self.precision)?
            .try_with_padding([0, 0])?
            .try_with_activation(self.activation)
    }
}

/// Standalone-job plan for a fully-connected operation: the FC shape and
/// the convolution plan its height-one lowering resolves to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    shape: Shape,
    conv: ConvPlan,
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
        })
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }

    /// Exposes the underlying convolution plan for storage and scheduling
    /// queries.
    pub fn conv_plan(&self) -> &ConvPlan {
        &self.conv
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::PlanErrorCode;

    #[test]
    fn maps_m_k_n_to_the_captured_height_one_convolution() {
        let conv = Shape::try_new(7, 16, 32, Precision::Fp16)
            .unwrap()
            .try_as_conv_shape()
            .unwrap();
        assert_eq!(
            (conv.width, conv.height, conv.in_channels, conv.out_channels),
            (7, 1, 16, 32)
        );
        assert_eq!(conv.padding, Some([0, 0]));
        assert_eq!(conv.output_width(KERNELS), 7);
        assert_eq!(conv.output_height(KERNELS), 1);
    }

    #[test]
    fn refuses_through_the_convolution_planner() {
        let error = Shape::try_new(4, conv::MAX_INPUT_CHANNELS + 1, 32, Precision::Fp16)
            .expect_err("K past the ceiling");
        assert_eq!(error.code(), PlanErrorCode::UnvalidatedConfiguration);
        assert_eq!(
            code(Shape::try_new(0, 16, 32, Precision::Fp16)),
            PlanErrorCode::InvalidShape
        );
    }

    #[test]
    fn a_wide_m_splits_into_column_tiles() {
        // (K/32 - 1) * M must stay at or below 2047 per tile (ISSUES.md C10).
        let plan = Plan::try_new(Shape::try_new(296, 3584, 64, Precision::Fp16).unwrap()).unwrap();
        assert!(plan.conv_plan().tiles().len() > 1);
        assert_eq!(
            plan.conv_plan()
                .tiles()
                .iter()
                .map(|t| t.columns.out_cols)
                .sum::<u32>(),
            296
        );
        assert_eq!(plan, Plan::new(plan.shape()));
    }

    fn code<T: std::fmt::Debug>(result: Result<T, PlanError>) -> PlanErrorCode {
        result.expect_err("expected a refusal").code()
    }
}
