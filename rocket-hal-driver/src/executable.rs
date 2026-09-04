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

/// A logical pooling shape field supplied by one uint32 dispatch push
/// constant.
///
/// The same contract as [`RuntimeConv2dDimension`]: ordering is carried by
/// [`PoolingExecutable::runtime_dimensions`] rather than by this enum's
/// numeric values, and the schema decoder maps the wire enum into this
/// runtime-owned type so recording never depends on a FlatBuffer object
/// outliving it.
///
/// Output extents are absent for the reason they are absent from
/// `Conv2DDimension`: `PoolingShape::validate` derives them from the input,
/// kernel, stride and padding, so a settable output could state a shape the
/// register program was not built for. Padding is absent too, but for a
/// different reason -- it is 0..=7, it means different things per method,
/// and no measured model varies it per dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimePoolingDimension {
    InputWidth,
    InputHeight,
    Channels,
    KernelWidth,
    KernelHeight,
    StrideX,
    StrideY,
}

impl RuntimePoolingDimension {
    fn index(self) -> usize {
        match self {
            Self::InputWidth => 0,
            Self::InputHeight => 1,
            Self::Channels => 2,
            Self::KernelWidth => 3,
            Self::KernelHeight => 4,
            Self::StrideX => 5,
            Self::StrideY => 6,
        }
    }

    fn get(self, shape: &PoolingShape) -> u32 {
        match self {
            Self::InputWidth => shape.input_width,
            Self::InputHeight => shape.input_height,
            Self::Channels => shape.input_channels,
            Self::KernelWidth => shape.kernel_width,
            Self::KernelHeight => shape.kernel_height,
            Self::StrideX => shape.stride_x,
            Self::StrideY => shape.stride_y,
        }
    }

    fn set(self, shape: &mut PoolingShape, value: u32) {
        match self {
            Self::InputWidth => shape.input_width = value,
            Self::InputHeight => shape.input_height = value,
            Self::Channels => {
                // Pooling preserves the channel count; the wire format
                // carries one field and `PoolingShape` carries two, so this
                // is where they are kept equal.
                shape.input_channels = value;
                shape.output_channels = value;
            }
            Self::KernelWidth => shape.kernel_width = value,
            Self::KernelHeight => shape.kernel_height = value,
            Self::StrideX => shape.stride_x = value,
            Self::StrideY => shape.stride_y = value,
        }
    }
}

/// Pooling executable metadata before per-dispatch runtime dimensions
/// resolve.
///
/// `shape_template`'s output extents are whatever the executable declared;
/// [`PoolingExecutable::resolve_shape`] recomputes them from the resolved
/// input geometry, so a dynamic pool does not need the compiler to predict
/// them and a static one is checked against its own claim.
#[derive(Clone, Debug, PartialEq)]
pub struct PoolingExecutable {
    pub shape_template: PoolingShape,
    pub runtime_dimensions: Vec<RuntimePoolingDimension>,
}

impl PoolingExecutable {
    pub fn new_static(shape: PoolingShape) -> Self {
        Self {
            shape_template: shape,
            runtime_dimensions: Vec::new(),
        }
    }

    /// Validates the schema-level dynamic mapping independently of runtime
    /// values, exactly as `Conv2dExecutable::validate_template` does.
    pub fn validate_template(&self) -> Result<(), &'static str> {
        let mut seen = [false; 7];
        for dimension in &self.runtime_dimensions {
            let index = dimension.index();
            if seen[index] {
                return Err("runtime pooling dimensions must be unique");
            }
            seen[index] = true;
            if dimension.get(&self.shape_template) != 0 {
                return Err("runtime pooling dimensions must be zero in the executable template");
            }
        }

        let all_dimensions = [
            RuntimePoolingDimension::InputWidth,
            RuntimePoolingDimension::InputHeight,
            RuntimePoolingDimension::Channels,
            RuntimePoolingDimension::KernelWidth,
            RuntimePoolingDimension::KernelHeight,
            RuntimePoolingDimension::StrideX,
            RuntimePoolingDimension::StrideY,
        ];
        for dimension in all_dimensions {
            if !seen[dimension.index()] && dimension.get(&self.shape_template) == 0 {
                return Err("static pooling dimensions must be nonzero in the executable template");
            }
        }

        if self.runtime_dimensions.is_empty() {
            self.resolve_shape(&[])?;
        }
        Ok(())
    }

    /// Resolves runtime dimensions from native-endian uint32 push constants,
    /// then runs the same authoritative validation a static executable gets.
    ///
    /// The output extents are *derived* here rather than trusted: a dynamic
    /// pool cannot carry them (the compiler would have to predict them per
    /// dispatch) and a static one has already stated them, so this recomputes
    /// floor-mode geometry and rejects a template that disagreed.
    pub fn resolve_shape(&self, constants: &[u8]) -> Result<PoolingShape, &'static str> {
        let expected_bytes = self
            .runtime_dimensions
            .len()
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or("runtime pooling push-constant byte count overflow")?;
        if constants.len() != expected_bytes {
            return Err("runtime pooling push-constant byte count does not match the executable");
        }

        let mut shape = self.shape_template;
        let declared_output = (shape.output_width, shape.output_height);
        for (dimension, bytes) in self
            .runtime_dimensions
            .iter()
            .zip(constants.chunks_exact(std::mem::size_of::<u32>()))
        {
            let value = u32::from_ne_bytes(bytes.try_into().unwrap());
            if value == 0 {
                return Err("runtime pooling dimensions must be nonzero");
            }
            dimension.set(&mut shape, value);
        }

        shape.output_width = floor_output_extent(
            shape.input_width,
            shape.kernel_width,
            shape.stride_x,
            shape.pad_left,
            shape.pad_right,
        )?;
        shape.output_height = floor_output_extent(
            shape.input_height,
            shape.kernel_height,
            shape.stride_y,
            shape.pad_top,
            shape.pad_bottom,
        )?;
        if self.runtime_dimensions.is_empty() && declared_output != (0, 0) {
            // A static executable stated its own output extents. They are
            // not load-bearing -- the derivation above is -- but a
            // disagreement means the producer and the runtime do not share
            // a geometry model, which is worth failing on rather than
            // quietly overriding.
            if declared_output != (shape.output_width, shape.output_height) {
                return Err("pooling output extents disagree with floor-mode geometry");
            }
        }

        // `PoolingShape::validate` panics rather than returning, because
        // every other caller builds a shape in-process. Here the shape came
        // off a wire, so the panic is converted to an error at this
        // boundary the same way `fc::Shape::new` is in the executable cache.
        std::panic::catch_unwind(|| shape.validate())
            .map_err(|_| "pooling shape is outside what the PPU can program")?;
        Ok(shape)
    }
}

/// Floor-mode output extent, matching `PoolingShape::validate`'s own rule.
/// Returned as an error rather than a panic because the inputs come from a
/// dispatch.
fn floor_output_extent(
    input: u32,
    kernel: u32,
    stride: u32,
    before: u32,
    after: u32,
) -> Result<u32, &'static str> {
    if kernel == 0 || stride == 0 {
        return Err("pooling kernel and stride must be nonzero");
    }
    let padded = input
        .checked_add(before)
        .and_then(|value| value.checked_add(after))
        .ok_or("pooling padded extent overflows")?;
    if padded < kernel {
        return Err("pooling kernel exceeds its padded input extent");
    }
    Ok((padded - kernel) / stride + 1)
}

/// A logical matmul shape field supplied by one uint32 dispatch push
/// constant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeMatmulDimension {
    M,
    K,
    N,
}

impl RuntimeMatmulDimension {
    fn index(self) -> usize {
        match self {
            Self::M => 0,
            Self::K => 1,
            Self::N => 2,
        }
    }

    fn get(self, shape: &fc::Shape) -> u32 {
        match self {
            Self::M => shape.m,
            Self::K => shape.k,
            Self::N => shape.n,
        }
    }

    fn set(self, shape: &mut fc::Shape, value: u32) {
        match self {
            Self::M => shape.m = value,
            Self::K => shape.k = value,
            Self::N => shape.n = value,
        }
    }
}

/// Matmul executable metadata before per-dispatch runtime dimensions
/// resolve.
///
/// The shape is an [`fc::Shape`] because the *lowering* is the vendor's
/// fully-connected one -- a height-one 1x1 convolution, established over 160
/// captured ONNX `Linear` models, and the registers are literally named
/// `CNA_FC_CON*`. The *operation* is a matmul, which is what the input
/// dialect has and what the wire format now names. Both remain true.
#[derive(Clone, Debug, PartialEq)]
pub struct MatmulExecutable {
    pub shape_template: fc::Shape,
    pub runtime_dimensions: Vec<RuntimeMatmulDimension>,
}

impl MatmulExecutable {
    pub fn new_static(shape: fc::Shape) -> Self {
        Self {
            shape_template: shape,
            runtime_dimensions: Vec::new(),
        }
    }

    pub fn validate_template(&self) -> Result<(), &'static str> {
        let mut seen = [false; 3];
        for dimension in &self.runtime_dimensions {
            let index = dimension.index();
            if seen[index] {
                return Err("runtime matmul dimensions must be unique");
            }
            seen[index] = true;
            if dimension.get(&self.shape_template) != 0 {
                return Err("runtime matmul dimensions must be zero in the executable template");
            }
        }
        for dimension in [
            RuntimeMatmulDimension::M,
            RuntimeMatmulDimension::K,
            RuntimeMatmulDimension::N,
        ] {
            if !seen[dimension.index()] && dimension.get(&self.shape_template) == 0 {
                return Err("static matmul dimensions must be nonzero in the executable template");
            }
        }
        if self.runtime_dimensions.is_empty() {
            self.resolve_shape(&[])?;
        }
        Ok(())
    }

    /// Resolves runtime dimensions, then validates through the same
    /// convolution gate a Conv2D executable goes through -- `fc::Shape`'s
    /// own constructor only re-checks the channel-count bounds, while
    /// `validate_conv_shape` trial-plans the shape it will actually build.
    pub fn resolve_shape(&self, constants: &[u8]) -> Result<fc::Shape, &'static str> {
        let expected_bytes = self
            .runtime_dimensions
            .len()
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or("runtime matmul push-constant byte count overflow")?;
        if constants.len() != expected_bytes {
            return Err("runtime matmul push-constant byte count does not match the executable");
        }

        let mut shape = self.shape_template;
        for (dimension, bytes) in self
            .runtime_dimensions
            .iter()
            .zip(constants.chunks_exact(std::mem::size_of::<u32>()))
        {
            let value = u32::from_ne_bytes(bytes.try_into().unwrap());
            if value == 0 {
                return Err("runtime matmul dimensions must be nonzero");
            }
            dimension.set(&mut shape, value);
        }
        let conv = std::panic::catch_unwind(|| shape.as_conv_shape())
            .map_err(|_| "matmul shape is outside the convolution builder's bounds")?;
        validate_conv_shape(&conv, fc::KERNELS)?;
        Ok(shape)
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
    /// Both `MatmulDef` and the deprecated `FullyConnectedDef` decode into
    /// this: they describe the same operation and the runtime executes them
    /// identically, so there is nothing for a second variant to distinguish.
    Matmul(MatmulExecutable),
    Pooling(PoolingExecutable),
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
