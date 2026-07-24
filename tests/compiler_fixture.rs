use rocket_schema::rocket;

#[test]
fn compiler_fixture_verifies_and_preserves_conv_fields() {
    let executable = include_bytes!("../testdata/mnv2_conv0.rkt1");
    assert_eq!(&executable[0..4], b"RKT1");
    assert_eq!(u32::from_le_bytes(executable[4..8].try_into().unwrap()), 0);

    let content_size =
        u64::from_le_bytes(executable[8..16].try_into().unwrap()) as usize;
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
