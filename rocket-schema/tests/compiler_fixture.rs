use rocket_schema::rocket;

#[test]
fn int8_accumulator_precision_round_trips() {
    let mut builder = flatbuffers::FlatBufferBuilder::new();
    let conv = rocket::Conv2DDef::create(
        &mut builder,
        &rocket::Conv2DDefArgs {
            input_width: 4,
            input_height: 4,
            input_channels: 1,
            output_width: 4,
            output_height: 4,
            output_channels: 8,
            weights_width: 1,
            weights_height: 1,
            stride: 1,
            precision: rocket::Precision::INT8_ACCUMULATOR,
            ..Default::default()
        },
    );
    let name = builder.create_string("conv_integer");
    let export = rocket::ExportDef::create(
        &mut builder,
        &rocket::ExportDefArgs {
            name: Some(name),
            kernel_type: rocket::KernelDef::Conv2DDef,
            kernel: Some(conv.as_union_value()),
        },
    );
    let exports = builder.create_vector(&[export]);
    let executable = rocket::ExecutableDef::create(
        &mut builder,
        &rocket::ExecutableDefArgs {
            exports: Some(exports),
        },
    );
    rocket::finish_executable_def_buffer(&mut builder, executable);

    let root = rocket::root_as_executable_def(builder.finished_data()).unwrap();
    let conv = root.exports().get(0).kernel_as_conv_2ddef().unwrap();
    assert_eq!(conv.precision(), rocket::Precision::INT8_ACCUMULATOR);
}

#[test]
fn compiler_fixture_verifies_and_preserves_conv_fields() {
    let executable = include_bytes!("../testdata/mnv2_conv0.rkt1");
    assert_eq!(&executable[0..4], b"RKT1");
    assert_eq!(u32::from_le_bytes(executable[4..8].try_into().unwrap()), 0);

    let content_size = u64::from_le_bytes(executable[8..16].try_into().unwrap()) as usize;
    assert_eq!(content_size, executable.len() - 64);
    let flatbuffer = &executable[64..];
    assert!(rocket::executable_def_buffer_has_identifier(flatbuffer));

    let root = rocket::root_as_executable_def(flatbuffer).unwrap();
    let exports = root.exports();
    assert_eq!(exports.len(), 1);

    let export = exports.get(0);
    assert_eq!(export.name(), "rocket_conv2d_0");
    assert_eq!(export.kernel_type(), rocket::KernelDef::Conv2DDef);

    let conv = export.kernel_as_conv_2ddef().unwrap();
    assert_eq!(conv.input_width(), 112);
    assert_eq!(conv.input_height(), 112);
    assert_eq!(conv.input_channels(), 32);
    assert_eq!(conv.output_width(), 112);
    assert_eq!(conv.output_height(), 112);
    assert_eq!(conv.output_channels(), 16);
    assert_eq!(conv.weights_width(), 1);
    assert_eq!(conv.weights_height(), 1);
    assert_eq!(conv.stride(), 1);
    assert_eq!((conv.pad_top(), conv.pad_left()), (0, 0));
    assert!(!conv.depthwise());
    assert_eq!(conv.precision(), rocket::Precision::FP16);
}

#[test]
fn fully_connected_definition_round_trips() {
    let mut builder = flatbuffers::FlatBufferBuilder::new();
    let name = builder.create_string("rocket_fc_0");
    let fc = rocket::FullyConnectedDef::create(
        &mut builder,
        &rocket::FullyConnectedDefArgs {
            m: 4,
            k: 32,
            n: 16,
            input_scale: 0.25,
            weights_scale: 0.5,
            output_scale: 0.125,
            activation: rocket::Activation::RELU,
            precision: rocket::Precision::FP16,
            ..Default::default()
        },
    );
    let export = rocket::ExportDef::create(
        &mut builder,
        &rocket::ExportDefArgs {
            name: Some(name),
            kernel_type: rocket::KernelDef::FullyConnectedDef,
            kernel: Some(fc.as_union_value()),
        },
    );
    let exports = builder.create_vector(&[export]);
    let executable = rocket::ExecutableDef::create(
        &mut builder,
        &rocket::ExecutableDefArgs {
            exports: Some(exports),
        },
    );
    rocket::finish_executable_def_buffer(&mut builder, executable);

    let bytes = builder.finished_data();
    assert!(rocket::executable_def_buffer_has_identifier(bytes));
    let root = rocket::root_as_executable_def(bytes).unwrap();
    let export = root.exports().get(0);
    assert_eq!(export.kernel_type(), rocket::KernelDef::FullyConnectedDef);

    let fc = export.kernel_as_fully_connected_def().unwrap();
    assert_eq!((fc.m(), fc.k(), fc.n()), (4, 32, 16));
    assert_eq!(fc.input_scale(), 0.25);
    assert_eq!(fc.weights_scale(), 0.5);
    assert_eq!(fc.output_scale(), 0.125);
    assert_eq!(fc.activation(), rocket::Activation::RELU);
    assert_eq!(fc.precision(), rocket::Precision::FP16);
}

#[test]
fn runtime_conv_dimensions_round_trip_in_push_constant_order() {
    let mut builder = flatbuffers::FlatBufferBuilder::new();
    let runtime_dimensions = builder.create_vector(&[
        rocket::Conv2DDimension::INPUT_HEIGHT,
        rocket::Conv2DDimension::INPUT_WIDTH,
        rocket::Conv2DDimension::WEIGHTS_HEIGHT,
        rocket::Conv2DDimension::WEIGHTS_WIDTH,
    ]);
    let conv = rocket::Conv2DDef::create(
        &mut builder,
        &rocket::Conv2DDefArgs {
            input_channels: 32,
            output_channels: 16,
            weights_width: 1,
            weights_height: 1,
            stride: 1,
            precision: rocket::Precision::FP16,
            runtime_dimensions: Some(runtime_dimensions),
            ..Default::default()
        },
    );
    let name = builder.create_string("dynamic_conv");
    let export = rocket::ExportDef::create(
        &mut builder,
        &rocket::ExportDefArgs {
            name: Some(name),
            kernel_type: rocket::KernelDef::Conv2DDef,
            kernel: Some(conv.as_union_value()),
        },
    );
    let exports = builder.create_vector(&[export]);
    let executable = rocket::ExecutableDef::create(
        &mut builder,
        &rocket::ExecutableDefArgs {
            exports: Some(exports),
        },
    );
    rocket::finish_executable_def_buffer(&mut builder, executable);

    let root = rocket::root_as_executable_def(builder.finished_data()).unwrap();
    let conv = root.exports().get(0).kernel_as_conv_2ddef().unwrap();
    let dimensions = conv.runtime_dimensions().unwrap();
    assert_eq!(dimensions.len(), 4);
    assert_eq!(dimensions.get(0), rocket::Conv2DDimension::INPUT_HEIGHT);
    assert_eq!(dimensions.get(1), rocket::Conv2DDimension::INPUT_WIDTH);
    assert_eq!(dimensions.get(2), rocket::Conv2DDimension::WEIGHTS_HEIGHT);
    assert_eq!(dimensions.get(3), rocket::Conv2DDimension::WEIGHTS_WIDTH);
}

/// The union tag is wire-format ABI: an executable built by an older
/// compiler names its kernel by this number, so a reordering of the union
/// members would silently reinterpret one kernel kind as another. The
/// deprecated `FullyConnectedDef` keeps tag 2 for exactly this reason.
#[test]
fn kernel_union_tags_are_stable() {
    assert_eq!(rocket::KernelDef::Conv2DDef.0, 1);
    assert_eq!(rocket::KernelDef::FullyConnectedDef.0, 2);
    assert_eq!(rocket::KernelDef::PoolingDef.0, 3);
    assert_eq!(rocket::KernelDef::MatmulDef.0, 4);
}

/// The pooling method values are wire format, chosen independently of the
/// PPU's register encoding even where the two happen to agree.
#[test]
fn pooling_method_values_are_stable() {
    assert_eq!(rocket::PoolingMethod::AVG.0, 0);
    assert_eq!(rocket::PoolingMethod::MAX.0, 1);
    assert_eq!(rocket::PoolingMethod::MIN.0, 2);
}

/// MobileNetV2's own global average pool, which is the shape this table was
/// added to carry: 7x7 over 1792 channels down to a single pixel.
#[test]
fn pooling_definition_round_trips() {
    let mut builder = flatbuffers::FlatBufferBuilder::new();
    let pooling = rocket::PoolingDef::create(
        &mut builder,
        &rocket::PoolingDefArgs {
            input_width: 7,
            input_height: 7,
            channels: 1792,
            output_width: 1,
            output_height: 1,
            kernel_width: 7,
            kernel_height: 7,
            stride_x: 1,
            stride_y: 1,
            method: rocket::PoolingMethod::AVG,
            precision: rocket::Precision::FP16,
            ..Default::default()
        },
    );
    let name = builder.create_string("rocket_pooling_0");
    let export = rocket::ExportDef::create(
        &mut builder,
        &rocket::ExportDefArgs {
            name: Some(name),
            kernel_type: rocket::KernelDef::PoolingDef,
            kernel: Some(pooling.as_union_value()),
        },
    );
    let exports = builder.create_vector(&[export]);
    let executable = rocket::ExecutableDef::create(
        &mut builder,
        &rocket::ExecutableDefArgs {
            exports: Some(exports),
        },
    );
    rocket::finish_executable_def_buffer(&mut builder, executable);

    let root = rocket::root_as_executable_def(builder.finished_data()).unwrap();
    let export = root.exports().get(0);
    assert_eq!(export.kernel_type(), rocket::KernelDef::PoolingDef);

    let pooling = export.kernel_as_pooling_def().unwrap();
    assert_eq!((pooling.input_width(), pooling.input_height()), (7, 7));
    assert_eq!(pooling.channels(), 1792);
    assert_eq!((pooling.output_width(), pooling.output_height()), (1, 1));
    assert_eq!((pooling.kernel_width(), pooling.kernel_height()), (7, 7));
    assert_eq!((pooling.stride_x(), pooling.stride_y()), (1, 1));
    assert_eq!(
        (
            pooling.pad_left(),
            pooling.pad_top(),
            pooling.pad_right(),
            pooling.pad_bottom(),
        ),
        (0, 0, 0, 0)
    );
    assert_eq!(pooling.method(), rocket::PoolingMethod::AVG);
    assert_eq!(pooling.precision(), rocket::Precision::FP16);
    assert!(pooling.runtime_dimensions().is_none());
}

/// A max pool with padding, which is the case the derived pad fill value
/// exists for: the runtime supplies -inf rather than reading a wire field a
/// producer could have set to zero.
#[test]
fn padded_max_pooling_definition_round_trips() {
    let mut builder = flatbuffers::FlatBufferBuilder::new();
    let pooling = rocket::PoolingDef::create(
        &mut builder,
        &rocket::PoolingDefArgs {
            input_width: 112,
            input_height: 112,
            channels: 64,
            output_width: 56,
            output_height: 56,
            kernel_width: 3,
            kernel_height: 3,
            stride_x: 2,
            stride_y: 2,
            pad_left: 1,
            pad_top: 1,
            pad_right: 1,
            pad_bottom: 1,
            method: rocket::PoolingMethod::MAX,
            precision: rocket::Precision::INT8,
            ..Default::default()
        },
    );
    let name = builder.create_string("rocket_pooling_max");
    let export = rocket::ExportDef::create(
        &mut builder,
        &rocket::ExportDefArgs {
            name: Some(name),
            kernel_type: rocket::KernelDef::PoolingDef,
            kernel: Some(pooling.as_union_value()),
        },
    );
    let exports = builder.create_vector(&[export]);
    let executable = rocket::ExecutableDef::create(
        &mut builder,
        &rocket::ExecutableDefArgs {
            exports: Some(exports),
        },
    );
    rocket::finish_executable_def_buffer(&mut builder, executable);

    let root = rocket::root_as_executable_def(builder.finished_data()).unwrap();
    let pooling = root.exports().get(0).kernel_as_pooling_def().unwrap();
    assert_eq!(
        (
            pooling.pad_left(),
            pooling.pad_top(),
            pooling.pad_right(),
            pooling.pad_bottom(),
        ),
        (1, 1, 1, 1)
    );
    assert_eq!(pooling.method(), rocket::PoolingMethod::MAX);
    assert_eq!(pooling.precision(), rocket::Precision::INT8);
}

#[test]
fn runtime_pooling_dimensions_round_trip_in_push_constant_order() {
    let mut builder = flatbuffers::FlatBufferBuilder::new();
    let runtime_dimensions = builder.create_vector(&[
        rocket::PoolingDimension::INPUT_HEIGHT,
        rocket::PoolingDimension::INPUT_WIDTH,
        rocket::PoolingDimension::CHANNELS,
        rocket::PoolingDimension::KERNEL_HEIGHT,
        rocket::PoolingDimension::KERNEL_WIDTH,
        rocket::PoolingDimension::STRIDE_Y,
        rocket::PoolingDimension::STRIDE_X,
    ]);
    let pooling = rocket::PoolingDef::create(
        &mut builder,
        &rocket::PoolingDefArgs {
            // Every listed field stays zero here and is filled from a push
            // constant, which is the contract Conv2DDef already documents.
            method: rocket::PoolingMethod::AVG,
            precision: rocket::Precision::FP16,
            runtime_dimensions: Some(runtime_dimensions),
            ..Default::default()
        },
    );
    let name = builder.create_string("rocket_dynamic_pooling");
    let export = rocket::ExportDef::create(
        &mut builder,
        &rocket::ExportDefArgs {
            name: Some(name),
            kernel_type: rocket::KernelDef::PoolingDef,
            kernel: Some(pooling.as_union_value()),
        },
    );
    let exports = builder.create_vector(&[export]);
    let executable = rocket::ExecutableDef::create(
        &mut builder,
        &rocket::ExecutableDefArgs {
            exports: Some(exports),
        },
    );
    rocket::finish_executable_def_buffer(&mut builder, executable);

    let root = rocket::root_as_executable_def(builder.finished_data()).unwrap();
    let pooling = root.exports().get(0).kernel_as_pooling_def().unwrap();
    let dimensions = pooling.runtime_dimensions().unwrap();
    assert_eq!(dimensions.len(), 7);
    assert_eq!(dimensions.get(0), rocket::PoolingDimension::INPUT_HEIGHT);
    assert_eq!(dimensions.get(2), rocket::PoolingDimension::CHANNELS);
    assert_eq!(dimensions.get(6), rocket::PoolingDimension::STRIDE_X);
    assert_eq!(pooling.input_width(), 0);
    assert_eq!(pooling.kernel_width(), 0);
}

/// MobileNetV2's classifier, the matmul this table was added to carry.
/// `K = 1792` is past the HAL's current `MAX_INPUT_CHANNELS`; that is a
/// runtime semantic limit, deliberately not a wire-format one, so the
/// schema round-trips it and the runtime is what refuses.
#[test]
fn matmul_definition_round_trips() {
    let mut builder = flatbuffers::FlatBufferBuilder::new();
    let matmul = rocket::MatmulDef::create(
        &mut builder,
        &rocket::MatmulDefArgs {
            m: 1,
            k: 1792,
            n: 1001,
            precision: rocket::Precision::FP16,
            ..Default::default()
        },
    );
    let name = builder.create_string("rocket_matmul_0");
    let export = rocket::ExportDef::create(
        &mut builder,
        &rocket::ExportDefArgs {
            name: Some(name),
            kernel_type: rocket::KernelDef::MatmulDef,
            kernel: Some(matmul.as_union_value()),
        },
    );
    let exports = builder.create_vector(&[export]);
    let executable = rocket::ExecutableDef::create(
        &mut builder,
        &rocket::ExecutableDefArgs {
            exports: Some(exports),
        },
    );
    rocket::finish_executable_def_buffer(&mut builder, executable);

    let root = rocket::root_as_executable_def(builder.finished_data()).unwrap();
    let export = root.exports().get(0);
    assert_eq!(export.kernel_type(), rocket::KernelDef::MatmulDef);

    let matmul = export.kernel_as_matmul_def().unwrap();
    assert_eq!((matmul.m(), matmul.k(), matmul.n()), (1, 1792, 1001));
    assert_eq!(matmul.precision(), rocket::Precision::FP16);
    assert_eq!(matmul.activation(), rocket::Activation::NONE);
    assert_eq!(matmul.input_scale(), 1.0);
    assert!(matmul.runtime_dimensions().is_none());
}

#[test]
fn runtime_matmul_dimensions_round_trip_in_push_constant_order() {
    let mut builder = flatbuffers::FlatBufferBuilder::new();
    let runtime_dimensions = builder.create_vector(&[
        rocket::MatmulDimension::M,
        rocket::MatmulDimension::K,
        rocket::MatmulDimension::N,
    ]);
    let matmul = rocket::MatmulDef::create(
        &mut builder,
        &rocket::MatmulDefArgs {
            input_zero_point: 0,
            precision: rocket::Precision::INT8,
            runtime_dimensions: Some(runtime_dimensions),
            ..Default::default()
        },
    );
    let name = builder.create_string("rocket_dynamic_matmul");
    let export = rocket::ExportDef::create(
        &mut builder,
        &rocket::ExportDefArgs {
            name: Some(name),
            kernel_type: rocket::KernelDef::MatmulDef,
            kernel: Some(matmul.as_union_value()),
        },
    );
    let exports = builder.create_vector(&[export]);
    let executable = rocket::ExecutableDef::create(
        &mut builder,
        &rocket::ExecutableDefArgs {
            exports: Some(exports),
        },
    );
    rocket::finish_executable_def_buffer(&mut builder, executable);

    let root = rocket::root_as_executable_def(builder.finished_data()).unwrap();
    let matmul = root.exports().get(0).kernel_as_matmul_def().unwrap();
    let dimensions = matmul.runtime_dimensions().unwrap();
    assert_eq!(dimensions.len(), 3);
    assert_eq!(dimensions.get(0), rocket::MatmulDimension::M);
    assert_eq!(dimensions.get(1), rocket::MatmulDimension::K);
    assert_eq!(dimensions.get(2), rocket::MatmulDimension::N);
    assert_eq!((matmul.m(), matmul.k(), matmul.n()), (0, 0, 0));
}
