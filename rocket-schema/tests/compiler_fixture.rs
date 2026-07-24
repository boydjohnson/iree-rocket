use rocket_schema::rocket;

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
        rocket::Conv2DDimension::OUTPUT_HEIGHT,
        rocket::Conv2DDimension::OUTPUT_WIDTH,
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
    assert_eq!(dimensions.get(2), rocket::Conv2DDimension::OUTPUT_HEIGHT);
    assert_eq!(dimensions.get(3), rocket::Conv2DDimension::OUTPUT_WIDTH);
}
