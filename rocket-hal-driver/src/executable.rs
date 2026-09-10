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
    activation::{LutShape, LutTable},
    conv::{self, Kernels, Multiplier, PlanError, Precision},
    elementwise::{EwAddShape, EwBinaryOp, EwPrecision, EwUnaryAlgo, EwUnaryShape},
    executable_format::validate_conv_shape,
    fc,
    layout::DispatchLayout,
    pooling::PoolingShape,
};

/// Narrows a planner refusal to the `&'static str` this module's error
/// paths carry. The status that reaches IREE is a bare code either way
/// (`status::from_code`), so the full message -- the one that names the
/// actual bound -- goes to stderr here, where a developer running the
/// module can see why a dispatch was refused.
fn plan_refusal(error: PlanError) -> &'static str {
    eprintln!("rocket: refusing convolution plan: {error}");
    error.code().description()
}

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

/// A quantization parameter supplied by one uint32 dispatch push constant.
///
/// Separate from [`RuntimeConv2dDimension`] because the two have opposite
/// rules about zero. No extent of a real convolution is zero, so a zero
/// dimension constant means the adapter failed to push one and is rejected;
/// zero is an ordinary zero point, so nothing about the *value* can say
/// whether it was supplied. Only the list says who owns each field.
///
/// The push-constant payload is a bit pattern, not a number, because the
/// schema fields are `uint32` and neither value fits that unsigned reading:
/// `OutputScale` carries an IEEE-754 binary32 and the zero points carry
/// two's-complement `i32`s. `Conv2DQuantParam` in
/// `rocket_executable_def.fbs` is the statement of that convention;
/// `RocketTarget.cpp`'s serializer is its other implementation.
///
/// There is no `InputScale`/`WeightsScale` here. The requantized int8 target
/// holds both at 1.0 so `pack_int8_bias_to_bs` passes an already-accumulator-
/// unit bias through untouched, which leaves `OutputScale` carrying the whole
/// requantization ratio `output_scale / (input_scale * weights_scale)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeConv2dQuantParam {
    OutputScale,
    InputZeroPoint,
    OutputZeroPoint,
}

impl RuntimeConv2dQuantParam {
    fn index(self) -> usize {
        match self {
            Self::OutputScale => 0,
            Self::InputZeroPoint => 1,
            Self::OutputZeroPoint => 2,
        }
    }

    /// The template value a listed parameter must hold, so an unsupplied
    /// constant cannot pass for calibration data.
    fn is_unset(self, quantization: &conv::Quantization) -> bool {
        match self {
            // The scale never survives decode as a sentinel -- `decode_precision`
            // substitutes 1.0 for it to build a placeholder multiplier -- so
            // the zero check lives in the serializer and there is nothing to
            // re-check here.
            Self::OutputScale => true,
            Self::InputZeroPoint => quantization.input_zero_point == 0,
            Self::OutputZeroPoint => quantization.output_zero_point == 0,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::OutputScale => "output_scale",
            Self::InputZeroPoint => "input_zero_point",
            Self::OutputZeroPoint => "output_zero_point",
        }
    }
}

/// What the compiler declared about a dispatch's layout, carried as the
/// last push constant when the executable declared `runtime_layout`
/// (`rocket_core::layout::DispatchLayout`, COMPILER_ROADMAP.md 6.2); the
/// bare reader count when it declared the retired `runtime_dense_readers`
/// instead; and all-dense otherwise, which is "chain nothing, always write
/// the dense output". Not validated here: the command buffer chains only
/// the inputs declared packed and compares the count against the consumers
/// it actually saw chain. Shared by every kernel kind, since each `*Def`
/// declares the flag the same way.
pub fn trailing_layout(
    declared_dense_readers: bool,
    declared_layout: bool,
    constants: &[u8],
) -> DispatchLayout {
    if (!declared_dense_readers && !declared_layout) || constants.len() < std::mem::size_of::<u32>()
    {
        return DispatchLayout::default();
    }
    let tail = &constants[constants.len() - std::mem::size_of::<u32>()..];
    let word = u32::from_ne_bytes(tail.try_into().unwrap());
    if declared_layout {
        DispatchLayout::from_word(word)
    } else {
        DispatchLayout {
            packed_inputs: 0,
            packed_readers: u16::try_from(word).unwrap_or(u16::MAX),
        }
    }
}

/// Conv2D executable metadata before per-dispatch runtime dimensions resolve.
#[derive(Clone, Debug, PartialEq)]
pub struct Conv2dExecutable {
    pub shape_template: conv::Shape,
    pub kernels: Kernels,
    pub runtime_dimensions: Vec<RuntimeConv2dDimension>,
    /// Consumed after every `runtime_dimensions` constant, in order.
    pub runtime_quantization: Vec<RuntimeConv2dQuantParam>,
    /// A residual epilogue: after the tiles, one EW task adds the fourth
    /// binding to the output cube and applies `epilogue_activation` in the
    /// EW core (`Conv2DDef.epilogue_add`). Bindings are then input,
    /// weights, bias, residual, output.
    pub epilogue_add: bool,
    pub epilogue_activation: conv::Activation,
    /// One trailing push constant carries the compiler's count of Rocket
    /// dispatches that read this dispatch's result
    /// (`Conv2DDef.runtime_dense_readers`); see [`Self::layout`].
    pub runtime_dense_readers: bool,
    /// The trailing push constant is a layout word (`*Def.runtime_layout`);
    /// see [`trailing_layout`].
    pub runtime_layout: bool,
    /// `Conv2DDef.weights_packed` / `MatmulDef.weights_packed`: the weights
    /// binding is already the packed coefficient stream, packed at compile
    /// time by the same `rocket_core::weights::WeightPlan` the runtime packs
    /// with; bind it directly, run no packer. COMPILER_ROADMAP.md 6.3.
    pub weights_packed: bool,
}

impl Conv2dExecutable {
    pub fn new_static(shape: conv::Shape, kernels: Kernels) -> Self {
        Self {
            shape_template: shape,
            kernels,
            runtime_dimensions: Vec::new(),
            runtime_quantization: Vec::new(),
            epilogue_add: false,
            epilogue_activation: conv::Activation::None,
            runtime_dense_readers: false,
            runtime_layout: false,
            weights_packed: false,
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

        let mut seen_quantization = [false; 3];
        for param in &self.runtime_quantization {
            let index = param.index();
            if seen_quantization[index] {
                return Err("runtime Conv2D quantization parameters must be unique");
            }
            seen_quantization[index] = true;
            match self.shape_template.precision {
                // Only the requantized path has a stage to consume these:
                // fp16 does not requantize, and the accumulator mode bypasses
                // BS/CPEND/out-convert entirely and already demands zero zero
                // points.
                Precision::Int8(quantization) => {
                    if !param.is_unset(&quantization) {
                        return Err(
                            "runtime Conv2D quantization parameters must be zero in the \
                             executable template",
                        );
                    }
                }
                _ => {
                    return Err("runtime Conv2D quantization parameters require int8 precision");
                }
            }
        }

        if self.runtime_dimensions.is_empty() && self.runtime_quantization.is_empty() {
            validate_conv_shape(&self.shape_template, self.kernels).map_err(plan_refusal)?;
        }
        Ok(())
    }

    /// The compiler's count of Rocket dispatches that read this dispatch's
    /// result, carried as the last push constant when
    /// `runtime_dense_readers` or `runtime_layout` is set; all-dense otherwise, which is "always write
    /// the dense output". The count is not validated here: the command
    /// buffer compares it against the consumers it actually saw chain, and
    /// any mismatch in either direction keeps the dense write.
    pub fn layout(&self, constants: &[u8]) -> DispatchLayout {
        trailing_layout(self.runtime_dense_readers, self.runtime_layout, constants)
    }

    /// Resolves runtime dimensions and quantization from native-endian uint32
    /// push constants, then performs the same authoritative validation as
    /// static executables.
    ///
    /// Dimensions come first and quantization parameters after, matching the
    /// order `RocketTarget.cpp` counts them in when it checks a pipeline
    /// layout's constant count against the target.
    // `as_chunks` would need every `bytes.try_into()` below re-typed for
    // marginal benefit on a numeric decode path; not worth the churn.
    #[allow(clippy::chunks_exact_to_as_chunks)]
    pub fn resolve_shape(&self, constants: &[u8]) -> Result<(conv::Shape, Kernels), &'static str> {
        let expected_bytes = self
            .runtime_dimensions
            .len()
            .checked_add(self.runtime_quantization.len())
            .and_then(|count| {
                count.checked_add(usize::from(
                    self.runtime_dense_readers || self.runtime_layout,
                ))
            })
            .and_then(|count| count.checked_mul(std::mem::size_of::<u32>()))
            .ok_or("runtime Conv2D push-constant byte count overflow")?;
        if constants.len() != expected_bytes {
            return Err("runtime Conv2D push-constant byte count does not match the executable");
        }

        let mut shape = self.shape_template;
        let mut kernels = self.kernels;
        let mut words = constants.chunks_exact(std::mem::size_of::<u32>());
        for dimension in &self.runtime_dimensions {
            let bytes = words
                .next()
                .ok_or("runtime Conv2D push constants ran out")?;
            let value = u32::from_ne_bytes(bytes.try_into().unwrap());
            if value == 0 {
                return Err("runtime Conv2D dimensions must be nonzero");
            }
            dimension.set(&mut shape, &mut kernels, value);
        }

        if !self.runtime_quantization.is_empty() {
            let Precision::Int8(mut quantization) = shape.precision else {
                return Err("runtime Conv2D quantization parameters require int8 precision");
            };
            // The template's multiplier is a placeholder built from a
            // substituted unit output scale (see `decode_precision`), so the
            // real one is derived here from whatever this dispatch supplied.
            // `input_scale`/`weights_scale` stay static, which is what lets a
            // single output scale carry the whole ratio.
            let mut output_scale = None;
            for param in &self.runtime_quantization {
                let bytes = words
                    .next()
                    .ok_or("runtime Conv2D push constants ran out")?;
                let value = u32::from_ne_bytes(bytes.try_into().unwrap());
                match param {
                    RuntimeConv2dQuantParam::OutputScale => {
                        output_scale = Some(f32::from_bits(value));
                    }
                    RuntimeConv2dQuantParam::InputZeroPoint => {
                        quantization.input_zero_point = value as i32;
                    }
                    RuntimeConv2dQuantParam::OutputZeroPoint => {
                        quantization.output_zero_point = value as i32;
                    }
                }
            }

            // A target may list only zero points and keep a static scale; in
            // that case the template's own multiplier is already the real one.
            if let Some(output_scale) = output_scale {
                if !output_scale.is_finite() || output_scale <= 0.0 {
                    return Err("runtime Conv2D output scale must be finite and positive");
                }
                let ratio = f64::from(quantization.input_scale)
                    * f64::from(quantization.weights_scale)
                    / f64::from(output_scale);
                // Plain `try_from_ratio`; the BS plane's gain is 1 here. See
                // the measurement recorded in
                // `executable_cache::decode_precision`.
                quantization.multiplier = Multiplier::try_from_ratio(ratio)?;
            }
            shape.precision = Precision::Int8(quantization);
        }

        validate_conv_shape(&shape, kernels).map_err(plan_refusal)?;
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
    /// See [`trailing_layout`].
    pub runtime_dense_readers: bool,
    /// The trailing push constant is a layout word (`*Def.runtime_layout`);
    /// see [`trailing_layout`].
    pub runtime_layout: bool,
}

impl PoolingExecutable {
    pub fn new_static(shape: PoolingShape) -> Self {
        Self {
            shape_template: shape,
            runtime_dimensions: Vec::new(),
            runtime_dense_readers: false,
            runtime_layout: false,
        }
    }

    pub fn layout(&self, constants: &[u8]) -> DispatchLayout {
        trailing_layout(self.runtime_dense_readers, self.runtime_layout, constants)
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
    // `as_chunks` would need every `bytes.try_into()` below re-typed for
    // marginal benefit on a numeric decode path; not worth the churn.
    #[allow(clippy::chunks_exact_to_as_chunks)]
    pub fn resolve_shape(&self, constants: &[u8]) -> Result<PoolingShape, &'static str> {
        let expected_bytes = self
            .runtime_dimensions
            .len()
            .checked_add(usize::from(
                self.runtime_dense_readers || self.runtime_layout,
            ))
            .and_then(|count| count.checked_mul(std::mem::size_of::<u32>()))
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

/// The geometry both element-wise executables carry, and the only thing a
/// dispatch may vary about them.
///
/// These ops have no reduction -- input and output cubes are identical -- so
/// unlike [`PoolingExecutable`] there is no derived output extent to
/// recompute or to check a template's claim against. Three fields is the
/// whole shape story; what differs between the two kinds is the operation and
/// its parameters, not the geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ElementwiseGeometry {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
}

/// A logical element-wise shape field supplied by one uint32 dispatch push
/// constant. Shared by both element-wise executables, matching the single
/// `ElementwiseDimension` the wire format carries for them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeElementwiseDimension {
    Width,
    Height,
    Channels,
}

impl RuntimeElementwiseDimension {
    fn index(self) -> usize {
        match self {
            Self::Width => 0,
            Self::Height => 1,
            Self::Channels => 2,
        }
    }

    fn get(self, geometry: &ElementwiseGeometry) -> u32 {
        match self {
            Self::Width => geometry.width,
            Self::Height => geometry.height,
            Self::Channels => geometry.channels,
        }
    }

    fn set(self, geometry: &mut ElementwiseGeometry, value: u32) {
        match self {
            Self::Width => geometry.width = value,
            Self::Height => geometry.height = value,
            Self::Channels => geometry.channels = value,
        }
    }
}

const ALL_ELEMENTWISE_DIMENSIONS: [RuntimeElementwiseDimension; 3] = [
    RuntimeElementwiseDimension::Width,
    RuntimeElementwiseDimension::Height,
    RuntimeElementwiseDimension::Channels,
];

/// The template/runtime split every executable in this file shares: a listed
/// dimension must be zero in the template, an unlisted one must be nonzero,
/// and listed dimensions must be unique.
fn validate_elementwise_template(
    geometry: &ElementwiseGeometry,
    runtime_dimensions: &[RuntimeElementwiseDimension],
) -> Result<(), &'static str> {
    let mut seen = [false; 3];
    for dimension in runtime_dimensions {
        let index = dimension.index();
        if seen[index] {
            return Err("runtime element-wise dimensions must be unique");
        }
        seen[index] = true;
        if dimension.get(geometry) != 0 {
            return Err("runtime element-wise dimensions must be zero in the executable template");
        }
    }
    for dimension in ALL_ELEMENTWISE_DIMENSIONS {
        if !seen[dimension.index()] && dimension.get(geometry) == 0 {
            return Err(
                "static element-wise dimensions must be nonzero in the executable template",
            );
        }
    }
    Ok(())
}

/// Resolves the three extents from native-endian uint32 push constants.
// `as_chunks` would need every `bytes.try_into()` below re-typed for
// marginal benefit on a numeric decode path; not worth the churn.
#[allow(clippy::chunks_exact_to_as_chunks)]
fn resolve_elementwise_geometry(
    template: &ElementwiseGeometry,
    runtime_dimensions: &[RuntimeElementwiseDimension],
    runtime_dense_readers: bool,
    constants: &[u8],
) -> Result<ElementwiseGeometry, &'static str> {
    let expected_bytes = runtime_dimensions
        .len()
        .checked_add(usize::from(runtime_dense_readers))
        .and_then(|count| count.checked_mul(std::mem::size_of::<u32>()))
        .ok_or("runtime element-wise push-constant byte count overflow")?;
    if constants.len() != expected_bytes {
        return Err("runtime element-wise push-constant byte count does not match the executable");
    }

    let mut geometry = *template;
    for (dimension, bytes) in runtime_dimensions
        .iter()
        .zip(constants.chunks_exact(std::mem::size_of::<u32>()))
    {
        let value = u32::from_ne_bytes(bytes.try_into().unwrap());
        if value == 0 {
            return Err("runtime element-wise dimensions must be nonzero");
        }
        dimension.set(&mut geometry, value);
    }
    Ok(geometry)
}

/// Unary element-wise executable metadata before per-dispatch runtime
/// dimensions resolve.
///
/// fp16 only, which is why there is no precision field here or on the wire:
/// [`EwUnaryShape`] ships no int8 branch, because no capture confirms an int8
/// zero-point/scale recipe for this task shape.
#[derive(Clone, Debug, PartialEq)]
pub struct ElementwiseUnaryExecutable {
    pub geometry: ElementwiseGeometry,
    pub algo: EwUnaryAlgo,
    /// The `ADD_SCALAR` operand as an IEEE-754 binary32 bit pattern. Must be
    /// zero for every other opcode.
    pub operand: u32,
    pub runtime_dimensions: Vec<RuntimeElementwiseDimension>,
    /// See [`trailing_layout`].
    pub runtime_dense_readers: bool,
    /// The trailing push constant is a layout word (`*Def.runtime_layout`);
    /// see [`trailing_layout`].
    pub runtime_layout: bool,
}

impl ElementwiseUnaryExecutable {
    pub fn new_static(geometry: ElementwiseGeometry, algo: EwUnaryAlgo, operand: u32) -> Self {
        Self {
            geometry,
            algo,
            operand,
            runtime_dimensions: Vec::new(),
            runtime_dense_readers: false,
            runtime_layout: false,
        }
    }

    pub fn layout(&self, constants: &[u8]) -> DispatchLayout {
        trailing_layout(self.runtime_dense_readers, self.runtime_layout, constants)
    }

    pub fn validate_template(&self) -> Result<(), &'static str> {
        validate_elementwise_template(&self.geometry, &self.runtime_dimensions)?;
        // `build_unary_regcmd` asserts this. The shape came off a wire here,
        // so it has to be an error at this boundary rather than a panic
        // inside the builder.
        if self.operand != 0 && !matches!(self.algo, EwUnaryAlgo::Add) {
            return Err("an element-wise unary operand is only meaningful for ADD_SCALAR");
        }
        Ok(())
    }

    pub fn resolve_shape(&self, constants: &[u8]) -> Result<EwUnaryShape, &'static str> {
        let geometry = resolve_elementwise_geometry(
            &self.geometry,
            &self.runtime_dimensions,
            self.runtime_dense_readers || self.runtime_layout,
            constants,
        )?;
        Ok(EwUnaryShape {
            width: geometry.width,
            height: geometry.height,
            channels: geometry.channels,
            algo: self.algo,
            operand: self.operand,
        })
    }
}

/// Two-tensor element-wise executable metadata before per-dispatch runtime
/// dimensions resolve.
///
/// fp16 only, and no precision field, for a narrower reason than
/// [`ElementwiseUnaryExecutable`]'s: [`EwAddShape`] *does* carry an int8
/// branch, but its `EW_CVT_SCALE`/`OUT_CVT_SCALE` ratio semantics are
/// inferred from register shape rather than confirmed against a known value,
/// and `Mul` has no int8 recipe in any capture. The wire format declines to
/// carry that inference; see the schema's own comment.
#[derive(Clone, Debug, PartialEq)]
pub struct ElementwiseBinaryExecutable {
    pub geometry: ElementwiseGeometry,
    pub op: EwBinaryOp,
    pub runtime_dimensions: Vec<RuntimeElementwiseDimension>,
    /// See [`trailing_layout`].
    pub runtime_dense_readers: bool,
    /// The trailing push constant is a layout word (`*Def.runtime_layout`);
    /// see [`trailing_layout`].
    pub runtime_layout: bool,
}

impl ElementwiseBinaryExecutable {
    pub fn new_static(geometry: ElementwiseGeometry, op: EwBinaryOp) -> Self {
        Self {
            geometry,
            op,
            runtime_dimensions: Vec::new(),
            runtime_dense_readers: false,
            runtime_layout: false,
        }
    }

    pub fn layout(&self, constants: &[u8]) -> DispatchLayout {
        trailing_layout(self.runtime_dense_readers, self.runtime_layout, constants)
    }

    pub fn validate_template(&self) -> Result<(), &'static str> {
        validate_elementwise_template(&self.geometry, &self.runtime_dimensions)
    }

    pub fn resolve_shape(&self, constants: &[u8]) -> Result<EwAddShape, &'static str> {
        let geometry = resolve_elementwise_geometry(
            &self.geometry,
            &self.runtime_dimensions,
            self.runtime_dense_readers || self.runtime_layout,
            constants,
        )?;
        Ok(EwAddShape {
            width: geometry.width,
            height: geometry.height,
            channels: geometry.channels,
            precision: EwPrecision::Fp16,
            op: self.op,
            // Every field below is int8-only and ignored at fp16, which is
            // the only precision this executable can describe. They are not
            // on the wire for that reason, so there is nothing to carry
            // here either.
            output_zero_point: 0,
            w_cvt_offset: 0,
            w_scale_ratio: 1.0,
            output_scale_ratio: 1.0,
        })
    }
}

/// Which LUT curve an [`ElementwiseLutExecutable`] selects.
///
/// A runtime-owned mirror of the wire `LutFn`, for the same reason
/// [`RuntimeConv2dDimension`] mirrors `Conv2DDimension`: command recording
/// must not depend on generated FlatBuffer objects staying alive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LutFunction {
    Sigmoid,
    Tanh,
    Exp,
    Square,
    Erf,
    Sqrt,
    Rsqrt,
    Log,
    Reciprocal,
}

impl LutFunction {
    pub fn table(self) -> LutTable {
        match self {
            Self::Sigmoid => LutTable::sigmoid(),
            Self::Tanh => LutTable::tanh(),
            Self::Exp => LutTable::exp(),
            Self::Square => LutTable::square(),
            Self::Erf => LutTable::erf(),
            Self::Sqrt => LutTable::sqrt(),
            Self::Rsqrt => LutTable::rsqrt(),
            Self::Log => LutTable::log(),
            Self::Reciprocal => LutTable::reciprocal(),
        }
    }
}

/// The decoded input zero points `build_lut_regcmd` will program a `BN_ALU`
/// operand for.
///
/// Not a stylistic restriction: `lut_bn_alu`'s formula is exact by
/// construction at zero, and confirmed against captures at -128, -2 and 127.
/// Known-bad captures exist at 42. The builder asserts on anything else, so
/// the wire boundary rejects it instead.
const SUPPORTED_LUT_ZERO_POINTS: [i32; 4] = [-128, -2, 0, 127];

/// LUT executable metadata before per-dispatch runtime dimensions resolve.
///
/// int8 only, and again with no precision field: the curve is evaluated on the
/// dequantized real value, so the scales and zero points are how an input
/// reaches the table's fixed domain at all rather than optional metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct ElementwiseLutExecutable {
    pub geometry: ElementwiseGeometry,
    pub function: LutFunction,
    /// Decoded (real) zero points. The wire carries these decoded; the
    /// `0x80` bias `LutShape` takes is applied in [`Self::resolve_shape`].
    pub input_zero_point: i32,
    pub output_zero_point: i32,
    pub input_scale: f32,
    pub output_scale: f32,
    pub runtime_dimensions: Vec<RuntimeElementwiseDimension>,
    /// See [`trailing_layout`].
    pub runtime_dense_readers: bool,
    /// The trailing push constant is a layout word (`*Def.runtime_layout`);
    /// see [`trailing_layout`].
    pub runtime_layout: bool,
}

impl ElementwiseLutExecutable {
    pub fn layout(&self, constants: &[u8]) -> DispatchLayout {
        trailing_layout(self.runtime_dense_readers, self.runtime_layout, constants)
    }

    pub fn validate_template(&self) -> Result<(), &'static str> {
        validate_elementwise_template(&self.geometry, &self.runtime_dimensions)?;
        if !SUPPORTED_LUT_ZERO_POINTS.contains(&self.input_zero_point) {
            return Err("unsupported LUT input zero point");
        }
        // The output zero point only shifts `OUT_CVT_OFFSET` and has no
        // formula to be wrong about, but it still has to fit the byte the
        // bias produces.
        if !(-128..=127).contains(&self.output_zero_point) {
            return Err("LUT output zero point does not fit an int8");
        }
        // A zero or non-finite scale makes `lut_bn_mul`/`lut_out_cvt`'s
        // `log2` produce a nonsense shift rather than a wrong answer.
        if !self.input_scale.is_finite()
            || !self.output_scale.is_finite()
            || self.input_scale <= 0.0
            || self.output_scale <= 0.0
        {
            return Err("LUT scales must be finite and positive");
        }
        Ok(())
    }

    pub fn resolve_shape(&self, constants: &[u8]) -> Result<LutShape, &'static str> {
        let geometry = resolve_elementwise_geometry(
            &self.geometry,
            &self.runtime_dimensions,
            self.runtime_dense_readers || self.runtime_layout,
            constants,
        )?;
        Ok(LutShape {
            width: geometry.width,
            height: geometry.height,
            channels: geometry.channels,
            input_zero_point: (self.input_zero_point as u8 as u32).wrapping_add(0x80) & 0xff,
            output_zero_point: (self.output_zero_point as u8 as u32).wrapping_add(0x80) & 0xff,
            input_scale: self.input_scale,
            output_scale: self.output_scale,
        })
    }
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
    /// See [`trailing_layout`].
    pub runtime_dense_readers: bool,
    /// The trailing push constant is a layout word (`*Def.runtime_layout`);
    /// see [`trailing_layout`].
    pub runtime_layout: bool,
    /// `Conv2DDef.weights_packed` / `MatmulDef.weights_packed`: the weights
    /// binding is already the packed coefficient stream, packed at compile
    /// time by the same `rocket_core::weights::WeightPlan` the runtime packs
    /// with; bind it directly, run no packer. COMPILER_ROADMAP.md 6.3.
    pub weights_packed: bool,
}

impl MatmulExecutable {
    pub fn new_static(shape: fc::Shape) -> Self {
        Self {
            shape_template: shape,
            runtime_dimensions: Vec::new(),
            runtime_dense_readers: false,
            runtime_layout: false,
            weights_packed: false,
        }
    }

    pub fn layout(&self, constants: &[u8]) -> DispatchLayout {
        trailing_layout(self.runtime_dense_readers, self.runtime_layout, constants)
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
    // `as_chunks` would need every `bytes.try_into()` below re-typed for
    // marginal benefit on a numeric decode path; not worth the churn.
    #[allow(clippy::chunks_exact_to_as_chunks)]
    pub fn resolve_shape(&self, constants: &[u8]) -> Result<fc::Shape, &'static str> {
        let expected_bytes = self
            .runtime_dimensions
            .len()
            .checked_add(usize::from(
                self.runtime_dense_readers || self.runtime_layout,
            ))
            .and_then(|count| count.checked_mul(std::mem::size_of::<u32>()))
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
        let conv = shape.try_as_conv_shape().map_err(plan_refusal)?;
        validate_conv_shape(&conv, fc::KERNELS).map_err(plan_refusal)?;
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
    ElementwiseUnary(ElementwiseUnaryExecutable),
    ElementwiseLut(ElementwiseLutExecutable),
    ElementwiseBinary(ElementwiseBinaryExecutable),
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
///
/// # Safety
///
/// `executable` must be a valid, non-null pointer to a `RocketExecutable`
/// created by this module's `create` and still live.
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
            runtime_quantization: Vec::new(),
            epilogue_add: false,
            epilogue_activation: conv::Activation::None,
            runtime_dense_readers: false,
            runtime_layout: false,
            weights_packed: false,
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
            runtime_quantization: Vec::new(),
            epilogue_add: false,
            epilogue_activation: conv::Activation::None,
            runtime_dense_readers: false,
            runtime_layout: false,
            weights_packed: false,
        };
        assert!(executable.resolve_shape(&constants(&[99, 99])).is_err());
    }
}
