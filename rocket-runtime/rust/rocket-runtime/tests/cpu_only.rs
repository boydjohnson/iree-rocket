//! The wrapper end to end on the CPU device, which every machine has. The
//! fixture is VMVX bytecode (see tests/data/cpu_ops.mlir), so the same file
//! runs on an x86 host and on the board. The one test that touches the NPU
//! asserts whichever outcome the machine it runs on should give.

use rocket_runtime::{
    BufferView, CpuPlacement, Devices, ElementType, Instance, Session, SessionOptions, StatusCode,
};

const MODULE_BYTES: &[u8] = include_bytes!("data/cpu_ops.vmvx.vmfb");

/// The fixture as a file, for the from-file loader. Written out from the
/// embedded copy rather than read from the source tree, so the test binary
/// runs where it is copied to (the board) and not only where it was built.
fn module_path() -> &'static std::path::Path {
    static PATH: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();
    PATH.get_or_init(|| {
        let path = std::env::temp_dir().join(format!(
            "rocket-runtime-cpu_ops-{}.vmvx.vmfb",
            std::process::id()
        ));
        std::fs::write(&path, MODULE_BYTES).unwrap();
        path
    })
}

fn cpu_session(placement: CpuPlacement) -> Session {
    let instance = Instance::new().unwrap();
    let options = SessionOptions::new()
        .devices(Devices::CpuOnly)
        .cpu_placement(placement);
    let mut session = Session::new(&instance, &options).unwrap();
    session.load_module_file(module_path()).unwrap();
    session
}

fn mul_add(session: &Session, a: &[f32], b: &[f32]) -> Vec<f32> {
    let a = BufferView::from_slice(session, &[4], a).unwrap();
    let b = BufferView::from_slice(session, &[4], b).unwrap();
    let outputs = session.call("module.mul_add", &[&a, &b]).unwrap();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].shape(), vec![4]);
    assert_eq!(outputs[0].element_type(), ElementType::Float32);
    outputs[0].to_vec::<f32>().unwrap()
}

#[test]
fn api_version_matches_the_bindings() {
    assert_eq!(
        rocket_runtime::api_version(),
        rocket_runtime::sys::ROCKET_RUNTIME_API_VERSION
    );
}

#[test]
fn runs_a_module_loaded_from_a_file() {
    let session = cpu_session(CpuPlacement::default());
    assert_eq!(session.device_count(), 1);
    assert_eq!(
        mul_add(&session, &[1.0, 2.0, 3.0, 4.0], &[10.0; 4]),
        vec![11.0, 22.0, 33.0, 44.0]
    );
}

#[test]
fn runs_a_module_loaded_from_memory() {
    let instance = Instance::new().unwrap();
    let options = SessionOptions::new().devices(Devices::CpuOnly);
    let mut session = Session::new(&instance, &options).unwrap();
    session.load_module_bytes(MODULE_BYTES).unwrap();
    assert_eq!(
        mul_add(&session, &[0.5, -1.0, 2.0, 0.0], &[2.0, 3.0, -1.0, 7.0]),
        vec![1.5, -4.0, 0.0, 0.0]
    );
    // Dropping the session destroys the module, which frees the copied bytes
    // through the crate's deallocator.
    drop(session);
}

#[test]
fn returns_every_result() {
    let session = cpu_session(CpuPlacement::default());
    let a = BufferView::from_slice(&session, &[2, 3], &[1i32, 2, 3, 4, 5, 6]).unwrap();
    let outputs = session.call("module.twice", &[&a]).unwrap();
    assert_eq!(outputs.len(), 2);
    for output in &outputs {
        assert_eq!(output.shape(), vec![2, 3]);
        assert_eq!(output.element_type(), ElementType::Int32);
        assert_eq!(output.byte_length(), 24);
    }
    assert_eq!(
        outputs[0].to_vec::<i32>().unwrap(),
        vec![2, 4, 6, 8, 10, 12]
    );
    assert_eq!(
        outputs[1].to_vec::<i32>().unwrap(),
        vec![1, 4, 9, 16, 25, 36]
    );
    // A signless i32 reads as either signedness.
    assert_eq!(outputs[1].to_vec::<u32>().unwrap()[5], 36);
}

#[test]
fn a_function_is_reusable() {
    let session = cpu_session(CpuPlacement::default());
    let mut function = session.function("module.mul_add").unwrap();
    for scale in [1.0f32, 2.0, 3.0] {
        let a = BufferView::from_slice(&session, &[4], &[1.0f32; 4]).unwrap();
        let b = BufferView::from_slice(&session, &[4], &[scale; 4]).unwrap();
        let out = function.invoke(&[&a, &b]).unwrap();
        assert_eq!(out[0].to_vec::<f32>().unwrap(), vec![scale + 1.0; 4]);
    }
}

#[test]
fn from_bytes_matches_from_slice() {
    let session = cpu_session(CpuPlacement::default());
    let values = [1.0f32, 2.0, 3.0, 4.0];
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let view = BufferView::from_bytes(&session, &[4], ElementType::Float32, &bytes).unwrap();
    assert_eq!(view.to_bytes().unwrap(), bytes);
    assert_eq!(view.clone().to_vec::<f32>().unwrap(), values);
}

#[test]
fn every_cpu_placement_runs() {
    for placement in [
        CpuPlacement::Fastest,
        CpuPlacement::Cpus(vec![0]),
        CpuPlacement::IreeDefault,
    ] {
        let session = cpu_session(placement.clone());
        assert_eq!(
            mul_add(&session, &[1.0; 4], &[1.0; 4]),
            vec![2.0; 4],
            "{placement:?}"
        );
    }
}

#[test]
fn an_empty_cpu_list_is_refused() {
    let instance = Instance::new().unwrap();
    let options = SessionOptions::new()
        .devices(Devices::CpuOnly)
        .cpu_placement(CpuPlacement::Cpus(vec![]));
    let error = Session::new(&instance, &options).err().unwrap();
    assert_eq!(error.code(), StatusCode::InvalidArgument);
}

#[test]
fn an_unqualified_function_name_gets_a_hint() {
    let session = cpu_session(CpuPlacement::default());
    let error = session.function("mul_add").err().unwrap();
    // IREE's own code (InvalidArgument: the name has no module part) is kept.
    assert_eq!(error.code(), StatusCode::InvalidArgument);
    assert!(
        error.message().contains("try \"module.mul_add\""),
        "{error}"
    );
}

#[test]
fn a_wrong_sized_input_is_refused() {
    let session = cpu_session(CpuPlacement::default());
    let error = BufferView::from_slice(&session, &[5], &[1.0f32; 4]).unwrap_err();
    assert_eq!(error.code(), StatusCode::InvalidArgument);
}

#[test]
fn reading_as_the_wrong_type_is_refused() {
    let session = cpu_session(CpuPlacement::default());
    let view = BufferView::from_slice(&session, &[4], &[1.0f32; 4]).unwrap();
    let error = view.to_vec::<i32>().unwrap_err();
    assert_eq!(error.code(), StatusCode::InvalidArgument);
}

#[test]
fn a_missing_module_file_is_an_error_with_a_message() {
    let instance = Instance::new().unwrap();
    let options = SessionOptions::new().devices(Devices::CpuOnly);
    let mut session = Session::new(&instance, &options).unwrap();
    let error = session
        .load_module_file("/nonexistent/model.vmfb")
        .unwrap_err();
    assert_eq!(error.code(), StatusCode::NotFound);
    assert!(
        error.message().contains("/nonexistent/model.vmfb"),
        "{error}"
    );
}

#[test]
fn sessions_and_tensors_move_between_threads() {
    let session = cpu_session(CpuPlacement::default());
    let view = BufferView::from_slice(&session, &[4], &[3.0f32; 4]).unwrap();
    let result = std::thread::spawn(move || {
        let ones = BufferView::from_slice(&session, &[4], &[1.0f32; 4]).unwrap();
        session.call("module.mul_add", &[&view, &ones]).unwrap()[0]
            .to_vec::<f32>()
            .unwrap()
    })
    .join()
    .unwrap();
    assert_eq!(result, vec![6.0; 4]);
}

/// Whether the `rocket` kernel driver has NPU cores bound here. Not "does
/// /dev/accel/accel0 exist": the x86 development host has one too (an Intel
/// NPU under `intel_vpu`), and on the board accel0 hangs off a virtual `rknn`
/// platform device with no driver link of its own. The cores are what the
/// rocket platform driver binds (`fdab0000.npu`, ...).
fn has_rocket_npu() -> bool {
    std::fs::read_dir("/sys/bus/platform/drivers/rocket").is_ok_and(|entries| {
        entries
            .flatten()
            .any(|entry| entry.file_name().to_string_lossy().ends_with(".npu"))
    })
}

#[test]
fn an_npu_session_needs_the_npu() {
    let instance = Instance::new().unwrap();
    let result = Session::new(&instance, &SessionOptions::default());
    if has_rocket_npu() {
        assert_eq!(result.unwrap().device_count(), 2);
    } else {
        let error = result.err().unwrap();
        assert_eq!(error.code(), StatusCode::Unavailable, "{error}");
        assert!(error.message().contains("CPU_ONLY"), "{error}");
    }
}

#[cfg(feature = "half")]
#[test]
fn half_tensors_round_trip() {
    use half::f16;
    let session = cpu_session(CpuPlacement::default());
    let values: Vec<f16> = [0.1f32, -2.5, 65504.0, 0.0]
        .iter()
        .map(|&v| f16::from_f32(v))
        .collect();
    let view = BufferView::from_slice(&session, &[2, 2], &values).unwrap();
    assert_eq!(view.element_type(), ElementType::Float16);
    assert_eq!(view.byte_length(), 8);
    assert_eq!(view.to_vec::<f16>().unwrap(), values);
    // The raw bytes are IEEE half: 0.1 is 0x2E66.
    assert_eq!(&view.to_bytes().unwrap()[..2], &[0x66, 0x2E]);
}
