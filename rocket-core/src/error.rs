//! The planner's refusal type.
//!
//! A refusal carries a stable [`PlanErrorCode`] and a message that names the
//! actual values involved. Callers branch on the code -- the compiler needs
//! to tell an invalid operation from one the hardware could run but nobody
//! has measured -- and show the message; nothing should parse the message.
//! `Display` prints the message alone so the panicking constructors in
//! `conv` and `fc`, which are `try_*` plus `panic!("{error}")`, keep the
//! exact text their callers and `#[should_panic(expected = ...)]` tests
//! have always seen.

use std::fmt;

/// Why the planner refused, coarsely enough to act on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PlanErrorCode {
    /// The descriptor is malformed on its own terms: a zero extent, a kernel
    /// larger than its padded input, padding that is not smaller than the
    /// kernel, column widths that do not cover the output, a size that
    /// overflows the address space.
    InvalidShape,
    /// A well-formed operation whose meaning the lowering cannot express: a
    /// fused activation on an int32 accumulator, a depthwise channel
    /// multiplier above one, nonzero zero-points on the accumulator path.
    UnsupportedSemantics,
    /// A fixed hardware bound no decomposition of this operation gets
    /// around: a register field too narrow for the value, a working set
    /// that no CBUF partition holds.
    HardwareLimit,
    /// No standalone tile fits under the CBUF partition chosen for this
    /// shape. A different decomposition might; the planner's own search
    /// did not find one.
    CapacityExceeded,
    /// Register-representable, but past the capture corpus or the board
    /// measurements that back the planner's rules. The hardware may well
    /// run it; nobody has checked, and the planner does not guess.
    UnvalidatedConfiguration,
    /// The planner panicked. Only the HAL's runtime boundary produces this,
    /// by catching what should have been a refusal; it is a bug report, not
    /// a property of the shape.
    Internal,
}

impl PlanErrorCode {
    /// A fixed one-line description, for boundaries that can only carry a
    /// `&'static str`.
    pub fn description(self) -> &'static str {
        match self {
            PlanErrorCode::InvalidShape => "convolution shape is malformed",
            PlanErrorCode::UnsupportedSemantics => {
                "convolution semantics are not supported by the Rocket lowering"
            }
            PlanErrorCode::HardwareLimit => "convolution exceeds a fixed hardware limit",
            PlanErrorCode::CapacityExceeded => "no standalone tile fits the CBUF for this shape",
            PlanErrorCode::UnvalidatedConfiguration => {
                "convolution configuration has no capture or board backing"
            }
            PlanErrorCode::Internal => "the convolution planner failed internally",
        }
    }
}

/// A planning refusal: a [`PlanErrorCode`] plus the message that names the
/// offending values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanError {
    code: PlanErrorCode,
    message: String,
}

impl PlanError {
    pub fn new(code: PlanErrorCode, message: impl Into<String>) -> PlanError {
        PlanError {
            code,
            message: message.into(),
        }
    }

    pub fn code(&self) -> PlanErrorCode {
        self.code
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for PlanError {}

/// `assert!`'s shape for a refusal: `check!(cond, code, "fmt", args)`.
macro_rules! refuse_unless {
    ($cond:expr, $code:expr, $($arg:tt)+) => {
        if !$cond {
            return Err($crate::error::PlanError::new($code, format!($($arg)+)));
        }
    };
}
pub(crate) use refuse_unless;
