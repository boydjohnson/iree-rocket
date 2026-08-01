//! Hardware-in-the-loop tests for the direct PPU_RDMA -> PPU pooling path.
//!
//! A plain `cargo test` runs the planning test only. Cross-compile the ignored
//! numerical tests and copy the resulting binary to an RK3588 board:
//!
//!   CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!     cargo test --target aarch64-unknown-linux-gnu --release \
//!       --test pooling_hw --no-run
//!
//! Tests affected by the upstream kernel completion issue are only compiled
//! with `--features kernel-fix`. They remain ignored and must still be selected
//! with `--ignored` on the board.
//!
//! Each logical int8 pixel occupies one full 16-byte NC1HWC2 feature atom, all
//! `TEST_CHANNELS` of it carrying data that is checked on readback. Wide cases
//! submit every independently kicked tile as the ordered task array of one
//! kernel job, with all tasks writing disjoint columns of one shared output BO.

use std::{
    fs::{File, OpenOptions},
    mem,
    os::unix::io::AsRawFd,
    ptr,
    time::Instant,
};

use iree_rocket_hal::rocket::{
    device::{Buffer, close_bo, fini_bo, prep_bo, submit_tasks, unmap_bo},
    pooling::{PoolingBuffers, PoolingMethod, PoolingPlan, PoolingPrecision, PoolingShape},
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const FEATURE_ATOM_BYTES: usize = 16;

/// Logical channels every numerical case carries: one full int8 NC1HWC2 atom.
///
/// These ran at one channel for a long time, which left fifteen of every
/// sixteen atom bytes zero and unverified. It also made the vendor comparison
/// meaningless: rknn-toolkit2 prefixes a channel-repacking `0x0d` stage for
/// *every* geometry at c=1, so no test shape matched the vendor structurally.
/// At a full atom the toolkit emits a bare `0x60` for the geometries marked
/// `vendor_bare`, which is exactly what `PoolingPlan` emits.
///
/// `PoolingPlan` rounds `input_channels` up to the atom either way, so this
/// does not change a single emitted register -- only how much of each atom
/// carries real data and gets checked.
const TEST_CHANNELS: u32 = 16;

#[derive(Clone, Copy, Debug)]
struct NumericCase {
    name: &'static str,
    width: u32,
    height: u32,
    kernel_width: u32,
    kernel_height: u32,
    stride_x: u32,
    stride_y: u32,
    /// Whether rknn-toolkit2 compiles this geometry to a bare `PPU | PPU_RDMA`
    /// (`0x60`) kick, the structure `PoolingPlan` emits, or prefixes it with a
    /// `CNA | CORE | DPU` (`0x0d`) stage.
    ///
    /// Measured, not guessed: `~/projects/rknn-files/sweep-ourshapes` holds a
    /// vendor model per geometry, classified by the exact kick word
    /// `(0x0081 << 48) | (mask << 16) | 0x0008`. That prefix stage is a 1x1
    /// stride-1 convolution which preserves width and height and only repacks
    /// the channel axis into NC1HWC2 -- a layout pass the toolkit needs because
    /// its input arrives as graph-level NCHW, and which this crate does not
    /// need because it owns its own buffer layout.
    ///
    /// Classified at `TEST_CHANNELS`, which is what these cases run. (At one
    /// channel the toolkit inserts the layout stage for every geometry, so the
    /// flag would be `false` everywhere and separate nothing -- that is why
    /// these tests no longer run at c=1.)
    vendor_bare: bool,
}

const K3_TWO_TILES: NumericCase = NumericCase {
    name: "k3_two_tiles",
    width: 256,
    height: 9,
    kernel_width: 3,
    kernel_height: 3,
    stride_x: 2,
    stride_y: 2,
    vendor_bare: true,
};

const K2_CAPTURE_BOUNDARY: NumericCase = NumericCase {
    name: "k2_capture_boundary",
    width: 257,
    height: 8,
    kernel_width: 2,
    kernel_height: 2,
    stride_x: 2,
    stride_y: 2,
    vendor_bare: true,
};

const K2_TWO_TILES: NumericCase = NumericCase {
    name: "k2_two_tiles",
    width: 258,
    height: 8,
    kernel_width: 2,
    kernel_height: 2,
    stride_x: 2,
    stride_y: 2,
    vendor_bare: true,
};

const NUMERIC_CASES: [NumericCase; 9] = [
    NumericCase {
        name: "small_square",
        width: 4,
        height: 4,
        kernel_width: 2,
        kernel_height: 2,
        stride_x: 2,
        stride_y: 2,
        vendor_bare: true,
    },
    NumericCase {
        name: "rectangular_kernel",
        width: 7,
        height: 5,
        kernel_width: 3,
        kernel_height: 2,
        stride_x: 2,
        stride_y: 1,
        vendor_bare: false,
    },
    NumericCase {
        name: "overlapping_windows",
        width: 13,
        height: 9,
        kernel_width: 3,
        kernel_height: 3,
        stride_x: 2,
        stride_y: 2,
        vendor_bare: false,
    },
    NumericCase {
        name: "asymmetric_stride",
        width: 17,
        height: 13,
        kernel_width: 4,
        kernel_height: 3,
        stride_x: 3,
        stride_y: 2,
        vendor_bare: false,
    },
    NumericCase {
        name: "largest_direct_window",
        width: 64,
        height: 31,
        kernel_width: 8,
        kernel_height: 8,
        stride_x: 3,
        stride_y: 3,
        vendor_bare: true,
    },
    NumericCase {
        name: "default_width_boundary",
        width: 129,
        height: 17,
        kernel_width: 3,
        kernel_height: 3,
        stride_x: 2,
        stride_y: 2,
        vendor_bare: false,
    },
    K3_TWO_TILES,
    K2_CAPTURE_BOUNDARY,
    K2_TWO_TILES,
];

fn pooling_shape(case: NumericCase, method: PoolingMethod) -> PoolingShape {
    PoolingShape {
        input_width: case.width,
        input_height: case.height,
        input_channels: TEST_CHANNELS,
        output_width: (case.width - case.kernel_width) / case.stride_x + 1,
        output_height: (case.height - case.kernel_height) / case.stride_y + 1,
        output_channels: TEST_CHANNELS,
        precision: PoolingPrecision::Int8,
        kernel_width: case.kernel_width,
        kernel_height: case.kernel_height,
        stride_x: case.stride_x,
        stride_y: case.stride_y,
        method,
        pad_left: 0,
        pad_top: 0,
        pad_right: 0,
        pad_bottom: 0,
        pad_value: 0,
    }
}

/// Channel-minor within each pixel, matching the NC1HWC2 atom the hardware
/// consumes: index `(y * width + x) * TEST_CHANNELS + channel`.
fn numeric_input(case: NumericCase) -> Vec<i8> {
    let mut input = Vec::with_capacity((case.width * case.height * TEST_CHANNELS) as usize);
    for y in 0..case.height {
        for x in 0..case.width {
            for channel in 0..TEST_CHANNELS {
                // Non-monotonic and non-negative: max/min cannot accidentally
                // pass through a fixed-corner bug, while int8 ordering is
                // clear. The channel term is coprime-ish with the rest so no
                // two channels of a pixel share a value, which is what makes a
                // cross-channel mix-up visible.
                input.push(((13 * x + 7 * y + 3 * x * y + 29 * channel) % 61) as i8);
            }
        }
    }
    input
}

fn cpu_pool(input: &[i8], shape: &PoolingShape) -> Vec<i8> {
    let channels = shape.input_channels;
    let mut output =
        Vec::with_capacity((shape.output_width * shape.output_height * channels) as usize);
    for output_y in 0..shape.output_height {
        for output_x in 0..shape.output_width {
            for channel in 0..channels {
                let input_y = output_y * shape.stride_y;
                let input_x = output_x * shape.stride_x;
                let mut minimum = i8::MAX;
                let mut maximum = i8::MIN;
                let mut sum = 0i32;
                for kernel_y in 0..shape.kernel_height {
                    for kernel_x in 0..shape.kernel_width {
                        let pixel = (input_y + kernel_y) * shape.input_width + input_x + kernel_x;
                        let value = input[(pixel * channels + channel) as usize];
                        minimum = minimum.min(value);
                        maximum = maximum.max(value);
                        sum += i32::from(value);
                    }
                }
                output.push(match shape.method {
                    PoolingMethod::Max => maximum,
                    PoolingMethod::Min => minimum,
                    PoolingMethod::Avg => {
                        let count = (shape.kernel_width * shape.kernel_height) as i32;
                        ((sum + count / 2) / count) as i8
                    }
                });
            }
        }
    }
    output
}

fn page_aligned(bytes: usize) -> usize {
    bytes.max(1).next_multiple_of(4096)
}

fn open_device() -> File {
    OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open NPU device")
}

fn run_numeric_case_on_file(
    file: &File,
    case: NumericCase,
    method: PoolingMethod,
) -> (Vec<i8>, Vec<i8>, u128) {
    let shape = pooling_shape(case, method);
    let input = numeric_input(case);
    let expected = cpu_pool(&input, &shape);
    let fd = file.as_raw_fd();

    unsafe {
        // One atom per pixel; TEST_CHANNELS of its bytes now carry real data.
        let channels = TEST_CHANNELS as usize;
        let input_pixels = (case.width * case.height) as usize;
        let input_bytes = input_pixels * FEATURE_ATOM_BYTES;
        let buf_in = Buffer::new(fd, page_aligned(input_bytes), file);
        ptr::write_bytes(buf_in.host_ptr, 0, buf_in.size);
        for pixel in 0..input_pixels {
            for channel in 0..channels {
                *buf_in.host_ptr.add(pixel * FEATURE_ATOM_BYTES + channel) =
                    input[pixel * channels + channel] as u8;
            }
        }

        let output_atoms = (shape.output_width * shape.output_height).next_multiple_of(4) as usize;
        let output_bytes = output_atoms * FEATURE_ATOM_BYTES;
        let buf_out = Buffer::new(fd, page_aligned(output_bytes), file);
        ptr::write_bytes(buf_out.host_ptr, 0xa5, buf_out.size);

        fini_bo(fd, buf_in.handle).expect("failed to sync pooling input BO for the NPU");
        fini_bo(fd, buf_out.handle).expect("failed to sync pooling output BO for the NPU");

        let programs = PoolingPlan::new(shape).programs_with_buffers(&PoolingBuffers {
            input_addr: buf_in.dma_address,
            output_addr: buf_out.dma_address,
        });
        let mut command_buffers = Vec::with_capacity(programs.len());
        for program in &programs {
            let command_bytes = program.len() * mem::size_of::<u64>();
            let command_buffer = Buffer::new(fd, page_aligned(command_bytes), file);
            let command_slice = std::slice::from_raw_parts_mut(
                command_buffer.host_ptr.cast::<u64>(),
                program.len(),
            );
            for (slot, command) in command_slice.iter_mut().zip(program) {
                *slot = command.0;
            }
            fini_bo(fd, command_buffer.handle)
                .expect("failed to sync pooling command BO for the NPU");
            command_buffers.push(command_buffer);
        }

        let tasks: Vec<_> = command_buffers
            .iter()
            .zip(&programs)
            .map(|(buffer, program)| (buffer.dma_address, program.len() as u32))
            .collect();
        let mut in_handles: Vec<_> = command_buffers.iter().map(|buffer| buffer.handle).collect();
        in_handles.push(buf_in.handle);
        let out_handles = [buf_out.handle];

        let submitted_at = Instant::now();
        submit_tasks(fd, &tasks, &in_handles, &out_handles)
            .expect("direct pooling multi-task SUBMIT ioctl failed");
        prep_bo(fd, buf_out.handle, 2_000_000_000)
            .expect("direct pooling job did not complete within timeout");
        let dispatch_ms = submitted_at.elapsed().as_millis();

        let raw_output = std::slice::from_raw_parts(buf_out.host_ptr, output_bytes);
        let actual = (0..expected.len())
            .map(|index| {
                let (pixel, channel) = (index / channels, index % channels);
                raw_output[pixel * FEATURE_ATOM_BYTES + channel] as i8
            })
            .collect();

        for command_buffer in command_buffers {
            unmap_bo(&command_buffer).expect("failed to unmap pooling command BO");
            close_bo(fd, command_buffer.handle).ok();
        }
        unmap_bo(&buf_in).expect("failed to unmap pooling input BO");
        unmap_bo(&buf_out).expect("failed to unmap pooling output BO");
        close_bo(fd, buf_in.handle).ok();
        close_bo(fd, buf_out.handle).ok();
        (actual, expected, dispatch_ms)
    }
}

fn assert_numeric_case_on_file(
    file: &File,
    case: NumericCase,
    method: PoolingMethod,
    tolerance: i16,
) {
    let (actual, expected, dispatch_ms) = run_numeric_case_on_file(file, case, method);
    let shape = pooling_shape(case, method);
    let mut mismatches = Vec::new();
    for (index, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
        if (i16::from(got) - i16::from(want)).abs() > tolerance && mismatches.len() < 8 {
            let index = index as u32;
            let channel = index % shape.output_channels;
            let pixel = index / shape.output_channels;
            let y = pixel / shape.output_width;
            let x = pixel % shape.output_width;
            mismatches.push(format!("[{y}, {x}, c{channel}] want {want} got {got}"));
        }
    }
    assert!(
        mismatches.is_empty(),
        "{} {method:?}: numerical pooling mismatches after {dispatch_ms} ms \
         (a duration near the driver's 500 ms watchdog means the scheduler reset the job): {}",
        case.name,
        mismatches.join(", ")
    );
}

fn assert_numeric_case(case: NumericCase, method: PoolingMethod, tolerance: i16) {
    let file = open_device();
    assert_numeric_case_on_file(&file, case, method, tolerance);
}

/// Whether a geometry should be exercised on hardware.
///
/// Default builds run only the geometries the vendor compiles to a bare
/// `0x60`, i.e. those whose vendor program is structurally identical to what
/// `PoolingPlan` emits. `--features vendor-bypass-shapes` restores the rest.
#[cfg(feature = "kernel-fix")]
fn enabled(case: NumericCase) -> bool {
    cfg!(feature = "vendor-bypass-shapes") || case.vendor_bare
}

#[cfg(feature = "kernel-fix")]
fn assert_numeric_matrix(method: PoolingMethod, tolerance: i16) {
    for case in NUMERIC_CASES.into_iter().filter(|&case| enabled(case)) {
        assert_numeric_case(case, method, tolerance);
    }
}

#[test]
fn numerical_dimension_matrix_has_expected_direct_task_counts() {
    for case in NUMERIC_CASES {
        for method in [PoolingMethod::Max, PoolingMethod::Min, PoolingMethod::Avg] {
            let shape = pooling_shape(case, method);
            let plan = PoolingPlan::new(shape);
            let programs = plan.programs_with_buffers(&PoolingBuffers {
                input_addr: 0x1000,
                output_addr: 0x8000,
            });
            assert_eq!(
                programs.len(),
                plan.tiles().len(),
                "{} {method:?}",
                case.name
            );
            assert!(
                programs.iter().all(|program| !program.is_empty()),
                "{} {method:?} emitted an empty task",
                case.name
            );
            assert_eq!(
                cpu_pool(&numeric_input(case), &shape).len(),
                (shape.output_width * shape.output_height * shape.output_channels) as usize,
                "{} {method:?} CPU reference extent",
                case.name
            );
        }
    }

    assert_eq!(
        PoolingPlan::new(pooling_shape(K2_CAPTURE_BOUNDARY, PoolingMethod::Max))
            .tiles()
            .len(),
        2
    );
    assert_eq!(
        PoolingPlan::new(pooling_shape(K2_TWO_TILES, PoolingMethod::Max))
            .tiles()
            .len(),
        2
    );
    assert_eq!(
        PoolingPlan::new(pooling_shape(K3_TWO_TILES, PoolingMethod::Max))
            .tiles()
            .len(),
        2
    );
}

#[cfg(feature = "kernel-fix")]
#[test]
#[ignore = "needs the real NPU device and the upstream kernel completion fix -- validates max pooling numerically across dimensions"]
fn max_pooling_dimension_matrix_matches_cpu_reference() {
    assert_numeric_matrix(PoolingMethod::Max, 0);
}

#[cfg(feature = "kernel-fix")]
#[test]
#[ignore = "needs the real NPU device and the upstream kernel completion fix -- validates min pooling numerically across dimensions"]
fn min_pooling_dimension_matrix_matches_cpu_reference() {
    assert_numeric_matrix(PoolingMethod::Min, 0);
}

#[cfg(feature = "kernel-fix")]
#[test]
#[ignore = "needs the real NPU device and the upstream kernel completion fix -- validates average pooling numerically across dimensions"]
fn average_pooling_dimension_matrix_matches_cpu_reference() {
    assert_numeric_matrix(PoolingMethod::Avg, 1);
}

#[cfg(feature = "kernel-fix")]
#[test]
#[ignore = "needs the real NPU device and the upstream kernel completion fix -- focused PPU_RDMA -> PPU tile 0 -> tile 1 job"]
fn direct_two_task_tiled_pooling_matches_cpu_reference() {
    assert_numeric_case(K2_TWO_TILES, PoolingMethod::Max, 0);
}

#[cfg(feature = "kernel-fix")]
#[test]
#[ignore = "needs the real NPU device and the upstream kernel completion fix -- isolates the int8 63+65 tile split"]
fn direct_equal_half_boundary_pooling_matches_cpu_reference() {
    assert_numeric_case(K2_CAPTURE_BOUNDARY, PoolingMethod::Max, 0);
}

#[cfg(feature = "vendor-bypass-shapes")]
#[test]
#[ignore = "needs the real NPU device -- isolates non-square min pooling"]
fn direct_rectangular_min_pooling_matches_cpu_reference() {
    assert_numeric_case(NUMERIC_CASES[1], PoolingMethod::Min, 0);
}

#[cfg(feature = "vendor-bypass-shapes")]
#[test]
#[ignore = "needs the real NPU device -- distinguishes min method from non-square geometry"]
fn direct_square_k3_min_pooling_matches_cpu_reference() {
    assert_numeric_case(NUMERIC_CASES[2], PoolingMethod::Min, 0);
}

#[cfg(feature = "kernel-fix")]
#[test]
#[ignore = "needs the real NPU device and the upstream kernel completion fix -- checks repeated GEM/VMA/domain teardown"]
fn repeated_pooling_resource_lifetime_matches_cpu_reference() {
    for _ in 0..4 {
        assert_numeric_case(NUMERIC_CASES[0], PoolingMethod::Max, 0);
        assert_numeric_case(K2_CAPTURE_BOUNDARY, PoolingMethod::Max, 0);
        assert_numeric_case(NUMERIC_CASES[0], PoolingMethod::Min, 0);
        #[cfg(feature = "vendor-bypass-shapes")]
        assert_numeric_case(NUMERIC_CASES[1], PoolingMethod::Min, 0);
        assert_numeric_case(K2_CAPTURE_BOUNDARY, PoolingMethod::Avg, 1);
    }
}

#[cfg(feature = "vendor-bypass-shapes")]
#[test]
#[ignore = "needs the real NPU device -- repeats the single-task width-129 PC launch boundary"]
fn repeated_default_width_boundary_matches_cpu_reference() {
    let case = NUMERIC_CASES[5];

    for _ in 0..4 {
        assert_numeric_case(case, PoolingMethod::Avg, 1);
        assert_numeric_case(case, PoolingMethod::Max, 0);
        assert_numeric_case(case, PoolingMethod::Min, 0);
    }
}

// Gated whole rather than filtered: these two exist to exercise the
// largest_direct_window -> default_width_boundary transition, and the second
// half of that pair is a vendor-bypass geometry. Dropping it would leave the
// test repeating one shape and silently stop testing what it is named for.
#[cfg(feature = "vendor-bypass-shapes")]
#[test]
#[ignore = "needs the real NPU device -- reproduces deferred BO cleanup across file close"]
fn pooling_file_lifetime_boundary_matches_cpu_reference() {
    for _ in 0..4 {
        assert_numeric_case(NUMERIC_CASES[4], PoolingMethod::Max, 0);
        assert_numeric_case(NUMERIC_CASES[5], PoolingMethod::Max, 0);
    }
}

#[cfg(feature = "vendor-bypass-shapes")]
#[test]
#[ignore = "needs the real NPU device -- isolates retained engine state within one DRM file"]
fn same_file_pooling_geometry_transition_matches_cpu_reference() {
    let file = open_device();

    for _ in 0..4 {
        assert_numeric_case_on_file(&file, NUMERIC_CASES[4], PoolingMethod::Max, 0);
        assert_numeric_case_on_file(&file, NUMERIC_CASES[5], PoolingMethod::Max, 0);
    }
}

// ============================================================================
// Wedging-predecessor sweep.
//
// `largest_direct_window` (64x31, k8x8, s3x3) is the only geometry that leaves
// the *next* job hanging, and the next job's own shape is irrelevant: with all
// shapes enabled the casualty is `default_width_boundary` (129x17), with only
// vendor-bare shapes it is `k3_two_tiles` (256x9). Both time out around 520 ms
// with their output BO still holding the 0xa5 poison fill.
//
// Three properties of that shape could be responsible, and the matrix tests
// cannot separate them because they only ever run it in one configuration.
// These probes move one axis at a time off the baseline, holding the successor
// fixed at the geometry directly observed hanging after it:
//
//   small_kernel   64x31 k3x3 s3x3  same extent and stride, 9 taps not 64
//   no_overlap     64x31 k8x8 s8x8  same extent and kernel, windows disjoint
//   short_extent   16x16 k8x8 s3x3  same kernel and overlap, far less work
//
// Read the result by which probe stops wedging: `small_kernel` clean implicates
// kernel area, `no_overlap` clean implicates window overlap, `short_extent`
// clean implicates dispatch duration (i.e. a quiesce race, not a geometry
// property). The reported predecessor timings distinguish the last case on
// their own. All three are vendor-bare at c=16, like the baseline, so none of
// them reintroduces the `0x0d` layout stage as a confound.
// ============================================================================

#[cfg(feature = "wedge-sweep")]
const WEDGE_SMALL_KERNEL: NumericCase = NumericCase {
    name: "small_kernel",
    width: 64,
    height: 31,
    kernel_width: 3,
    kernel_height: 3,
    stride_x: 3,
    stride_y: 3,
    vendor_bare: true,
};

#[cfg(feature = "wedge-sweep")]
const WEDGE_NO_OVERLAP: NumericCase = NumericCase {
    name: "no_overlap",
    width: 64,
    height: 31,
    kernel_width: 8,
    kernel_height: 8,
    stride_x: 8,
    stride_y: 8,
    vendor_bare: true,
};

#[cfg(feature = "wedge-sweep")]
const WEDGE_SHORT_EXTENT: NumericCase = NumericCase {
    name: "short_extent",
    width: 16,
    height: 16,
    kernel_width: 8,
    kernel_height: 8,
    stride_x: 3,
    stride_y: 3,
    vendor_bare: true,
};

/// Runs each probe as `predecessor -> successor` and reports which
/// predecessors left the successor broken.
///
/// A fresh DRM file per dispatch, matching how the matrix tests run -- the hang
/// reproduces across separate files, so this is not a within-file effect.
#[cfg(feature = "wedge-sweep")]
fn run_wedge_probes(successor: NumericCase, probes: &[(&'static str, Option<NumericCase>)]) {
    let mut rows = Vec::new();
    let mut wedged = Vec::new();

    for &(label, predecessor) in probes {
        let predecessor_ms = predecessor.map(|case| {
            let file = open_device();
            let (_, _, dispatch_ms) = run_numeric_case_on_file(&file, case, PoolingMethod::Max);
            dispatch_ms
        });

        let file = open_device();
        let (actual, expected, successor_ms) =
            run_numeric_case_on_file(&file, successor, PoolingMethod::Max);
        let wrong = actual
            .iter()
            .zip(&expected)
            .filter(|(got, want)| got != want)
            .count();

        rows.push(format!(
            "  {label:<22} predecessor {:>5} -> successor {successor_ms:>4} ms, {wrong:>6} of {} wrong",
            predecessor_ms.map_or("--".to_string(), |ms| format!("{ms} ms")),
            expected.len(),
        ));
        if wrong > 0 {
            wedged.push(label);
        }
    }

    assert!(
        wedged.is_empty(),
        "successor {} was wedged by: {}\n{}",
        successor.name,
        wedged.join(", "),
        rows.join("\n"),
    );
}

#[cfg(feature = "wedge-sweep")]
#[test]
#[ignore = "needs the real NPU device -- isolates which property of largest_direct_window wedges the following job"]
fn wedging_predecessor_sweep() {
    // Held fixed: the geometry observed hanging after largest_direct_window in
    // the vendor-bare configuration.
    run_wedge_probes(
        K3_TWO_TILES,
        &[
            // Controls first: the successor must be clean on its own and after
            // a geometry known not to wedge, or nothing below means anything.
            ("(no predecessor)", None),
            ("benign_small_square", Some(NUMERIC_CASES[0])),
            // Positive control: this one is expected to wedge.
            ("baseline_64x31_k8_s3", Some(NUMERIC_CASES[4])),
            ("small_kernel_k3_s3", Some(WEDGE_SMALL_KERNEL)),
            ("no_overlap_k8_s8", Some(WEDGE_NO_OVERLAP)),
            ("short_extent_16x16", Some(WEDGE_SHORT_EXTENT)),
        ],
    );
}

/// One predecessor geometry for the stride sweep: 64x31 with a 3x3 kernel, so
/// every stride below stays inside the input.
///
/// `vendor_bare` is measured per combination in `~/projects/rknn-files/
/// sweep-wedge`: the toolkit inserts its layout stage once stride exceeds the
/// kernel, so sy=4, sy=5 and sx=8 are bypass and the rest are bare. That has no
/// bearing on this experiment -- these run as predecessors through our own
/// `PoolingPlan`, whose structure does not vary with stride -- but the field
/// should not lie.
#[cfg(feature = "wedge-sweep")]
const fn stride_probe(stride_x: u32, stride_y: u32, vendor_bare: bool) -> NumericCase {
    NumericCase {
        name: "stride_probe",
        width: 64,
        height: 31,
        kernel_width: 3,
        kernel_height: 3,
        stride_x,
        stride_y,
        vendor_bare,
    }
}

#[cfg(feature = "wedge-sweep")]
#[test]
#[ignore = "needs the real NPU device -- confirms the stride_y threshold and that stride_x is irrelevant"]
fn wedging_stride_y_sweep() {
    // The predecessor sweep ruled out kernel area, window overlap and dispatch
    // duration, leaving stride_y >= 3 as the separator. Here stride is the only
    // thing that moves, so the threshold is measured rather than inferred.
    //
    // Result on hardware: sy1 and sy2 clean, sy3 wedges, sx3 clean. Since
    // `sx3_sy2` and `sy3_sx2` are the same 64x31 k3x3 geometry with the strides
    // transposed and only the vertical one wedges, the asymmetry is real and
    // not an artifact of some other axis.
    //
    // The original run also probed sy=4, sy=5 and sx=8. Those did not wedge a
    // successor -- they timed out on their *own* dispatch (500, 530 and 513 ms)
    // because a stride wider than the kernel hangs the direct PPU path.
    // `PoolingShape::validate` now rejects those shapes, so they would panic in
    // `PoolingPlan::new` before reaching hardware and are no longer probed
    // here; see the stride-beyond-kernel assertion in pooling.rs.
    run_wedge_probes(
        K3_TWO_TILES,
        &[
            ("(no predecessor)", None),
            ("benign_small_square", Some(NUMERIC_CASES[0])),
            ("sy1_sx2", Some(stride_probe(2, 1, true))),
            ("sy2_sx2", Some(stride_probe(2, 2, true))),
            ("sy3_sx2", Some(stride_probe(2, 3, true))),
            // Horizontal control at a known-clean sy. sx must stay <= kernel
            // now, so 3 is the largest probe available at a 3x3 kernel.
            ("sx3_sy2", Some(stride_probe(3, 2, true))),
        ],
    );
}
