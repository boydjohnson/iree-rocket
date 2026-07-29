//! `iree_hal_executable_vtable_t`. For this driver an "executable" isn't
//! compiled machine code -- it's a stored `UkernelShape` (one of this
//! driver's fixed regcmd-template shapes) because the NPU pipeline itself
//! is a small fixed set of these rather than a general codegen target. No
//! real MLIR codegen target
//! exists for this hardware yet (see the `custom_dispatch` research this
//! crate started from -- that mechanism doesn't fit a regcmd-bitstream
//! device), so `executable_cache::prepare_executable` currently only picks
//! between a small number of hardcoded shapes via a one-byte tag prefix on
//! `executable_data` -- still a deliberate placeholder, not a real
//! executable-format parser (see that module's doc comment for the exact
//! tag convention).

use crate::{
    bindings::{
        iree_hal_buffer_t, iree_hal_executable_function_info_t,
        iree_hal_executable_function_parameter_t, iree_hal_executable_function_t,
        iree_hal_executable_t, iree_hal_executable_vtable_t, iree_hal_queue_affinity_t,
        iree_hal_resource_t, iree_host_size_t, iree_status_t, iree_string_view_t,
    },
    status,
};
use iree_rocket_hal::rocket::{
    conv::{self, Kernels},
    executable_format::validate_conv_shape,
    fc,
    pooling::PoolingShape,
};

/// A logical Conv2D shape/kernel field supplied by one uint32 dispatch push
/// constant.
///
/// Ordering is carried by [`Conv2dExecutable::runtime_dimensions`], not this
/// enum's numeric representation. The FlatBuffer decoder maps the schema enum
/// into this runtime-owned type so command recording never depends on generated
/// FlatBuffer objects remaining alive.
///
/// Unlike the retired Mesa-derived shape, [`conv::Shape`] has no independent
/// `OutputWidth`/`OutputHeight` fields -- they're always
/// `Shape::output_width(kernels)`/`output_height(kernels)`, derived from the
/// other five dimensions plus stride/padding, so making them independently
/// settable would reintroduce exactly the redundant/possibly-inconsistent
/// fields this type migration removes. Only six dimensions remain settable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeConv2dDimension {
    InputWidth,
    InputHeight,
    InputChannels,
    OutputChannels,
    WeightsWidth,
    WeightsHeight,
}

impl RuntimeConv2dDimension {
    fn index(self) -> usize {
        match self {
            Self::InputWidth => 0,
            Self::InputHeight => 1,
            Self::InputChannels => 2,
            Self::OutputChannels => 3,
            Self::WeightsWidth => 4,
            Self::WeightsHeight => 5,
        }
    }

    fn get(self, shape: &conv::Shape, kernels: Kernels) -> u32 {
        match self {
            Self::InputWidth => shape.width,
            Self::InputHeight => shape.height,
            Self::InputChannels => shape.in_channels,
            Self::OutputChannels => shape.out_channels,
            Self::WeightsWidth => kernels[1] as u32,
            Self::WeightsHeight => kernels[0] as u32,
        }
    }

    fn set(self, shape: &mut conv::Shape, kernels: &mut Kernels, value: u32) {
        match self {
            Self::InputWidth => shape.width = value,
            Self::InputHeight => shape.height = value,
            Self::InputChannels => shape.in_channels = value,
            Self::OutputChannels => shape.out_channels = value,
            Self::WeightsWidth => kernels[1] = value as usize,
            Self::WeightsHeight => kernels[0] = value as usize,
        }
    }
}

/// Conv2D executable metadata before per-dispatch runtime dimensions resolve.
#[derive(Clone, Debug, PartialEq)]
pub struct Conv2dExecutable {
    pub shape_template: conv::Shape,
    pub kernels: Kernels,
    pub runtime_dimensions: Vec<RuntimeConv2dDimension>,
}

impl Conv2dExecutable {
    pub fn new_static(shape: conv::Shape, kernels: Kernels) -> Self {
        Self {
            shape_template: shape,
            kernels,
            runtime_dimensions: Vec::new(),
        }
    }

    /// Validates the schema-level dynamic mapping independently of runtime
    /// values. Full hardware validation happens in [`resolve_shape`].
    pub fn validate_template(&self) -> Result<(), &'static str> {
        let mut seen = [false; 6];
        for dimension in &self.runtime_dimensions {
            let index = dimension.index();
            if seen[index] {
                return Err("runtime Conv2D dimensions must be unique");
            }
            seen[index] = true;
            if dimension.get(&self.shape_template, self.kernels) != 0 {
                return Err("runtime Conv2D dimensions must be zero in the executable template");
            }
        }

        let all_dimensions = [
            RuntimeConv2dDimension::InputWidth,
            RuntimeConv2dDimension::InputHeight,
            RuntimeConv2dDimension::InputChannels,
            RuntimeConv2dDimension::OutputChannels,
            RuntimeConv2dDimension::WeightsWidth,
            RuntimeConv2dDimension::WeightsHeight,
        ];
        for dimension in all_dimensions {
            if !seen[dimension.index()] && dimension.get(&self.shape_template, self.kernels) == 0 {
                return Err("static Conv2D dimensions must be nonzero in the executable template");
            }
        }

        if self.runtime_dimensions.is_empty() {
            validate_conv_shape(&self.shape_template, self.kernels)?;
        }
        Ok(())
    }

    /// Resolves runtime dimensions from native-endian uint32 push constants,
    /// then performs the same authoritative validation as static executables.
    pub fn resolve_shape(&self, constants: &[u8]) -> Result<(conv::Shape, Kernels), &'static str> {
        let expected_bytes = self
            .runtime_dimensions
            .len()
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or("runtime Conv2D push-constant byte count overflow")?;
        if constants.len() != expected_bytes {
            return Err("runtime Conv2D push-constant byte count does not match the executable");
        }

        let mut shape = self.shape_template;
        let mut kernels = self.kernels;
        for (dimension, bytes) in self
            .runtime_dimensions
            .iter()
            .zip(constants.chunks_exact(std::mem::size_of::<u32>()))
        {
            let value = u32::from_ne_bytes(bytes.try_into().unwrap());
            if value == 0 {
                return Err("runtime Conv2D dimensions must be nonzero");
            }
            dimension.set(&mut shape, &mut kernels, value);
        }
        validate_conv_shape(&shape, kernels)?;
        Ok((shape, kernels))
    }
}

/// One of this driver's fixed regcmd-template shapes -- see this crate's
/// `iree-rocket-hal::rocket::conv`/`fc` module doc comments for why the NPU
/// pipeline itself is a small, fixed set of these ("ukernels") rather than a
/// general codegen target. Extend this enum (and `command_buffer::dispatch`'s
/// match on it) as more of iree-rocket-hal's `build_*_regcmd`/`Plan` types
/// gain HAL-level wiring.
pub enum UkernelShape {
    Conv2d(Conv2dExecutable),
    FullyConnected(fc::Shape),
    Pooling(PoolingShape),
}

/// What every `iree_hal_executable_t*` this driver hands out actually
/// points to. `iree_hal_executable_t` is opaque (no public field
/// definition), so `resource` is the real base-at-offset-0 field.
#[repr(C)]
pub struct RocketExecutable {
    pub resource: iree_hal_resource_t,
    /// Exactly one "function" (ordinal 0) -- the hardcoded shape. A real
    /// executable format would carry N functions/entry points; this
    /// placeholder only ever has one.
    pub shape: UkernelShape,
}

unsafe fn cast(executable: *mut iree_hal_executable_t) -> *mut RocketExecutable {
    executable as *mut RocketExecutable
}

pub fn create(shape: UkernelShape) -> *mut iree_hal_executable_t {
    let executable = Box::new(RocketExecutable {
        resource: iree_hal_resource_t {
            ref_count: 1,
            vtable: &VTABLE as *const _ as *const std::ffi::c_void,
        },
        shape,
    });
    Box::into_raw(executable) as *mut iree_hal_executable_t
}

/// Not part of the vtable -- `command_buffer::dispatch` calls this
/// directly to get at the shape it needs for the matching `build_*_regcmd`
/// call.
pub unsafe fn shape(executable: *mut iree_hal_executable_t) -> *const UkernelShape {
    unsafe { &(*cast(executable)).shape }
}

unsafe extern "C" fn destroy(executable: *mut iree_hal_executable_t) {
    unsafe { drop(Box::from_raw(cast(executable))) }
}

#[allow(unused_variables)]
unsafe extern "C" fn function_count(executable: *mut iree_hal_executable_t) -> iree_host_size_t {
    1
}

status_stub!(function_info(
    executable: *mut iree_hal_executable_t,
    function: iree_hal_executable_function_t,
    out_info: *mut iree_hal_executable_function_info_t,
) -> iree_status_t);

status_stub!(function_parameters(
    executable: *mut iree_hal_executable_t,
    function: iree_hal_executable_function_t,
    capacity: iree_host_size_t,
    out_parameters: *mut iree_hal_executable_function_parameter_t,
) -> iree_status_t);

#[allow(unused_variables)]
unsafe extern "C" fn lookup_function_by_name(
    executable: *mut iree_hal_executable_t,
    name: iree_string_view_t,
    out_function: *mut iree_hal_executable_function_t,
) -> iree_status_t {
    // Only one function (ordinal 0) exists -- see module doc comment.
    unsafe {
        (*out_function).value = 0;
    }
    status::ok()
}

status_stub!(lookup_global_by_name(
    executable: *mut iree_hal_executable_t,
    name: iree_string_view_t,
    queue_affinity: iree_hal_queue_affinity_t,
    out_buffer: *mut *mut iree_hal_buffer_t,
) -> iree_status_t);

pub static VTABLE: iree_hal_executable_vtable_t = iree_hal_executable_vtable_t {
    destroy: Some(destroy),
    function_count: Some(function_count),
    function_info: Some(function_info),
    function_parameters: Some(function_parameters),
    lookup_function_by_name: Some(lookup_function_by_name),
    lookup_global_by_name: Some(lookup_global_by_name),
};

#[cfg(test)]
mod tests {
    use super::*;
    use iree_rocket_hal::rocket::conv::Activation;

    /// Dynamic input spatial extent, fixed channels and 1x1 kernel -- with
    /// stride 1 and no padding, `output_width(kernels)`/`output_height(kernels)`
    /// always equal `width`/`height` exactly, which is what the assertions
    /// below rely on.
    fn dynamic_spatial_executable() -> Conv2dExecutable {
        Conv2dExecutable {
            shape_template: conv::Shape {
                width: 0,
                height: 0,
                stride: 1,
                in_channels: 32,
                out_channels: 16,
                precision: conv::Precision::Fp16,
                padding: Some([0, 0]),
                activation: Activation::None,
                depthwise: false,
            },
            kernels: [1, 1],
            runtime_dimensions: vec![
                RuntimeConv2dDimension::InputHeight,
                RuntimeConv2dDimension::InputWidth,
            ],
        }
    }

    fn constants(values: &[u32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect()
    }

    #[test]
    fn runtime_dimensions_resolve_in_declared_order() {
        let executable = dynamic_spatial_executable();
        executable.validate_template().unwrap();
        let (shape, kernels) = executable.resolve_shape(&constants(&[112, 96])).unwrap();
        assert_eq!((shape.width, shape.height), (96, 112));
        assert_eq!(
            (shape.output_width(kernels), shape.output_height(kernels)),
            (96, 112)
        );
        assert_eq!((shape.in_channels, shape.out_channels), (32, 16));
    }

    #[test]
    fn runtime_dimensions_reject_wrong_constant_count_and_zero() {
        let executable = dynamic_spatial_executable();
        assert!(executable.resolve_shape(&constants(&[112])).is_err());
        assert!(executable.resolve_shape(&constants(&[112, 0])).is_err());
    }

    #[test]
    fn runtime_dimensions_reject_duplicate_mapping() {
        let mut executable = dynamic_spatial_executable();
        executable
            .runtime_dimensions
            .push(RuntimeConv2dDimension::InputWidth);
        assert!(executable.validate_template().is_err());
    }

    #[test]
    fn runtime_dimensions_reject_hardware_invalid_shape() {
        // Runtime kernel extent, far outside ConvPlan's capture-backed
        // 1..=11 range -- passes the "nonzero" check but must still be
        // rejected by validate_conv_shape.
        let executable = Conv2dExecutable {
            shape_template: conv::Shape {
                width: 96,
                height: 112,
                stride: 1,
                in_channels: 32,
                out_channels: 16,
                precision: conv::Precision::Fp16,
                padding: Some([0, 0]),
                activation: Activation::None,
                depthwise: false,
            },
            kernels: [0, 0],
            runtime_dimensions: vec![
                RuntimeConv2dDimension::WeightsHeight,
                RuntimeConv2dDimension::WeightsWidth,
            ],
        };
        assert!(executable.resolve_shape(&constants(&[99, 99])).is_err());
    }
}
