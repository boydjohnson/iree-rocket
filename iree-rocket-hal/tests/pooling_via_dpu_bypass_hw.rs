//! Hardware-in-the-loop tests for `build_pooling_via_dpu_bypass_tasks` --
//! the real multi-task shape a vendor-compiled model actually emits (see
//! `pooling.rs`'s module doc comment above that function, and
//! rknpu-spelunking/NOTES.md's "Decoding a real regcmd program for a
//! pooling-only op"). The standalone-only hardware suite was retired after
//! its large-window and tiled-width probes failed; a third, on-chip
//! pipelined shape was tried and then retired after a sweep found no compiler
//! evidence for it. See `pooling.rs`'s module doc comment for both histories.
//!
//! Not run by a plain `cargo test` -- see `conv_phase1_validation_hw.rs`'s
//! doc comment for the cross-compile-and-copy-to-the-board workflow;
//! identical here:
//!
//!   CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!     cargo test --target aarch64-unknown-linux-gnu --release \
//!       --test pooling_via_dpu_bypass_hw --no-run
//!
//! The numerical matrix in this file makes the bypass convolution an exact
//! int8 identity, fills every input pixel with a position-dependent value,
//! and compares every pooling output against a CPU reference. It spans
//! rectangular inputs, kernels and strides, the direct PPU width boundary,
//! the largest directly hardware-backed 8x8 pooling window, and a tiled
//! width whose job is `DPU bypass -> PPU tile 0 -> PPU tile 1`.
//!
//! An early hardware round concatenated the bypass and PPU register streams
//! into one `rocket_task`; its second kick replaced the first before the
//! bypass ran. That demonstrates that each kick needs a complete task
//! boundary. It does not require separate jobs: this suite submits all task
//! programs as the ordered `rocket_task` array of one `rocket_job`, matching
//! the driver's task-advance mechanism.

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    activation::Activation,
    conv::{self, BsEntry, Kernels, Multiplier, Quantization, write_bs_buffer},
    device::{Buffer, close_bo, fini_bo, prep_bo, submit_tasks},
    pooling::{
        PoolingMethod, PoolingPlan, PoolingPrecision, PoolingShape, PoolingViaBypassBuffers,
        build_pooling_via_dpu_bypass_tasks,
    },
    tensor_layout::pack_hwcf_to_rocket_weights,
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const TENSOR_SIZE: usize = 4096;

/// 1x1 kernel, matching every `bypass_shape()` call in this file.
const BYPASS_KERNELS: Kernels = [1, 1];

/// Bypass stage: 4x4x1 -> 4x4x1, 1x1 kernel, stride 1 -- a real (if
/// numerically arbitrary under uniform fill) CNA->CORE->DPU task sized to
/// match the pooling stage's own 4x4 input, reusing the same safe (>=4)
/// input_height this module's other hardware-validated conv shapes use
/// (see conv.rs's own `input_height/4 - 1`-derived underflow risk,
/// documented in the roadmap plan's Phase 2 notes).
fn bypass_shape() -> conv::Shape {
    conv::Shape {
        width: 4,
        height: 4,
        stride: 1,
        in_channels: 1,
        out_channels: 1,
        precision: conv::Precision::Int8(Quantization {
            input_zero_point: 0,
            output_zero_point: 0,
            multiplier: Multiplier::from_ratio(1.0),
        }),
        padding: Some([0, 0]),
        activation: conv::Activation::None,
        depthwise: false,
    }
}

/// Pooling stage: 4x4x1 -> 2x2x1, 2x2 kernel, stride 2. Its "input" is the
/// bypass stage's real memory output, not an external buffer.
fn pooling_shape(method: PoolingMethod) -> PoolingShape {
    pooling_shape_with_activation(method, Activation::None)
}

/// Same geometry as `pooling_shape`, with an explicit fused activation --
/// Phase 3 of the ukernel roadmap. `build_pooling_via_dpu_bypass_tasks`
/// applies this to the bypass conv stage's own fused-activation stage (see
/// its own doc comment); `pooling_shape`'s `activation: Activation::None`
/// above is just the common case of this with `None`.
fn pooling_shape_with_activation(method: PoolingMethod, activation: Activation) -> PoolingShape {
    PoolingShape {
        input_width: 4,
        input_height: 4,
        input_channels: 1,
        output_width: 2,
        output_height: 2,
        output_channels: 1,
        precision: PoolingPrecision::Int8,
        kernel_width: 2,
        kernel_height: 2,
        stride_x: 2,
        stride_y: 2,
        method,
        pad_left: 0,
        pad_top: 0,
        pad_right: 0,
        pad_bottom: 0,
        pad_value: 0,
        activation,
    }
}

struct Bufs {
    file: std::fs::File,
    fd: i32,
    buf_in: Buffer,
    buf_w: Buffer,
    buf_bias: Buffer,
    buf_mid: Buffer,
    buf_out: Buffer,
}

/// Allocates and fills every buffer this shape needs, but does NOT submit
/// anything -- shared setup for both the real multi-task job and the
/// diagnostic (which also sentinel-fills buf_mid/buf_out differently).
fn setup(input_fill: u8, weight_fill: u8, out_fill: u8) -> Bufs {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let buf_in = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_in.host_ptr, input_fill, TENSOR_SIZE);

        let buf_w = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_w.host_ptr, weight_fill, TENSOR_SIZE);

        // A plain zero-fill supplies a zero BS multiplier (not just a zero
        // bias) for this int8 bypass shape -- conv.rs's write_bs_buffer doc
        // comment warns about exactly this: "the fp16 tests' habit of
        // zeroing the bias buffer does not carry over" for int8. A real
        // RK3588 run confirmed it: buf_mid came back correctly written but
        // to all-zero regardless of input, not the "unwritten" symptom a
        // dispatch/kick bug would show. BsEntry::default()'s unit multiplier
        // is the real "no rescale" value this near-identity bypass conv
        // needs, paired with bypass_shape()'s own Multiplier::from_ratio(1.0).
        let buf_bias = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_bias.host_ptr, 0, TENSOR_SIZE);
        let bs_entries = vec![BsEntry::default(); bypass_shape().padded_out_channels() as usize];
        let bs_bytes =
            std::slice::from_raw_parts_mut(buf_bias.host_ptr, bypass_shape().bs_buffer_bytes());
        write_bs_buffer(bs_bytes, &bs_entries);

        let buf_mid = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_mid.host_ptr, out_fill, TENSOR_SIZE);

        let buf_out = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_out.host_ptr, out_fill, TENSOR_SIZE);

        fini_bo(fd, buf_in.handle).ok();
        fini_bo(fd, buf_w.handle).ok();
        fini_bo(fd, buf_bias.handle).ok();
        fini_bo(fd, buf_mid.handle).ok();
        fini_bo(fd, buf_out.handle).ok();

        Bufs {
            file,
            fd,
            buf_in,
            buf_w,
            buf_bias,
            buf_mid,
            buf_out,
        }
    }
}

impl Bufs {
    /// Releases every GEM handle this struct owns. `Buffer` has no `Drop`
    /// impl (nothing in `iree-rocket-hal` did before this session's
    /// `close_bo` addition -- see `device.rs`'s doc comment on it), so
    /// without an explicit call like this, every one of these 5 buffers
    /// leaks for the rest of the process's lifetime. Added after repeated
    /// calls to `run_uniform_pooling_via_bypass` within one test (up to 6x
    /// in `pooling_via_bypass_repeat_dispatch_dump`, each leaking 7 GEM
    /// handles including `run_task_chain`'s own regcmd buffers) were found
    /// to produce corrupted/stale results, while the identical dispatch
    /// run exactly once in a fresh process was clean.
    unsafe fn close(self) {
        unsafe {
            close_bo(self.fd, self.buf_in.handle).ok();
            close_bo(self.fd, self.buf_w.handle).ok();
            close_bo(self.fd, self.buf_bias.handle).ok();
            close_bo(self.fd, self.buf_mid.handle).ok();
            close_bo(self.fd, self.buf_out.handle).ok();
        }
    }
}

/// Runs the bypass-then-pool shape as one ordered multi-task job and returns
/// the real output pixels at their 16-byte atomic stride.
fn run_uniform_pooling_via_bypass(
    method: PoolingMethod,
    input_fill: u8,
    weight_fill: u8,
) -> Vec<u8> {
    run_uniform_pooling_via_bypass_with_activation(
        method,
        input_fill,
        weight_fill,
        Activation::None,
    )
}

/// Same as `run_uniform_pooling_via_bypass`, with an explicit fused
/// activation on the pooling op (Phase 3, applied to the bypass conv
/// stage's own fused-activation stage by `build_pooling_via_dpu_bypass_tasks`)
/// instead of always `None`.
fn run_uniform_pooling_via_bypass_with_activation(
    method: PoolingMethod,
    input_fill: u8,
    weight_fill: u8,
    activation: Activation,
) -> Vec<u8> {
    let b = setup(input_fill, weight_fill, 0);
    unsafe {
        run_task_chain(
            &b,
            &bypass_shape(),
            &pooling_shape_with_activation(method, activation),
        )
    };

    let raw = unsafe { std::slice::from_raw_parts(b.buf_out.host_ptr, 256) };
    let pixels = (0..4).map(|i| raw[i * 16]).collect();
    unsafe { b.close() };
    pixels
}

unsafe fn run_task_chain(b: &Bufs, bypass: &conv::Shape, pooling: &PoolingShape) {
    unsafe {
        let bufs = PoolingViaBypassBuffers {
            input_addr: b.buf_in.dma_address,
            weights_addr: b.buf_w.dma_address,
            bias_addr: b.buf_bias.dma_address,
            bypass_output_addr: b.buf_mid.dma_address,
            output_addr: b.buf_out.dma_address,
        };
        let programs = build_pooling_via_dpu_bypass_tasks(bypass, BYPASS_KERNELS, pooling, &bufs);
        let mut command_buffers = Vec::with_capacity(programs.len());
        for program in &programs {
            let command_bytes = program.len() * mem::size_of::<u64>();
            let command_buffer = Buffer::new(b.fd, command_bytes.next_multiple_of(4096), &b.file);
            let command_slice =
                std::slice::from_raw_parts_mut(command_buffer.host_ptr as *mut u64, program.len());
            for (slot, command) in command_slice.iter_mut().zip(program) {
                *slot = command.0;
            }
            fini_bo(b.fd, command_buffer.handle)
                .expect("failed to sync pooling command BO for the NPU");
            command_buffers.push(command_buffer);
        }

        let tasks: Vec<_> = command_buffers
            .iter()
            .zip(&programs)
            .map(|(buffer, program)| (buffer.dma_address, program.len() as u32))
            .collect();
        let mut in_handles: Vec<_> = command_buffers.iter().map(|buffer| buffer.handle).collect();
        in_handles.extend([b.buf_in.handle, b.buf_w.handle, b.buf_bias.handle]);
        let out_handles = [b.buf_mid.handle, b.buf_out.handle];

        submit_tasks(b.fd, &tasks, &in_handles, &out_handles)
            .expect("bypass-plus-pooling multi-task SUBMIT ioctl failed");

        // Waiting on either output waits for the one job fence, hence every
        // task. PREP both only after that wait so both CPU mappings are
        // synchronized before the numerical checks inspect them.
        prep_bo(b.fd, b.buf_out.handle, 2_000_000_000)
            .expect("bypass-plus-pooling job did not complete within timeout");
        prep_bo(b.fd, b.buf_mid.handle, 2_000_000_000)
            .expect("failed to synchronize bypass intermediate after job completion");

        for command_buffer in command_buffers {
            close_bo(b.fd, command_buffer.handle).ok();
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct NumericCase {
    name: &'static str,
    width: u32,
    height: u32,
    kernel_width: u32,
    kernel_height: u32,
    stride_x: u32,
    stride_y: u32,
}

const TWO_HORIZONTAL_TILES: NumericCase = NumericCase {
    name: "two_horizontal_tiles",
    width: 256,
    height: 9,
    kernel_width: 3,
    kernel_height: 3,
    stride_x: 2,
    stride_y: 2,
};

const NUMERIC_CASES: [NumericCase; 7] = [
    NumericCase {
        name: "small_square",
        width: 4,
        height: 4,
        kernel_width: 2,
        kernel_height: 2,
        stride_x: 2,
        stride_y: 2,
    },
    NumericCase {
        name: "rectangular_kernel",
        width: 7,
        height: 5,
        kernel_width: 3,
        kernel_height: 2,
        stride_x: 2,
        stride_y: 1,
    },
    NumericCase {
        name: "overlapping_windows",
        width: 13,
        height: 9,
        kernel_width: 3,
        kernel_height: 3,
        stride_x: 2,
        stride_y: 2,
    },
    NumericCase {
        name: "asymmetric_stride",
        width: 17,
        height: 13,
        kernel_width: 4,
        kernel_height: 3,
        stride_x: 3,
        stride_y: 2,
    },
    NumericCase {
        name: "largest_direct_window",
        width: 64,
        height: 31,
        kernel_width: 8,
        kernel_height: 8,
        stride_x: 3,
        stride_y: 3,
    },
    NumericCase {
        name: "direct_width_boundary",
        width: 129,
        height: 17,
        kernel_width: 3,
        kernel_height: 3,
        stride_x: 2,
        stride_y: 2,
    },
    TWO_HORIZONTAL_TILES,
];

fn numeric_bypass_shape(case: NumericCase) -> conv::Shape {
    conv::Shape {
        width: case.width,
        height: case.height,
        stride: 1,
        in_channels: 1,
        out_channels: 1,
        precision: conv::Precision::Int8(Quantization {
            input_zero_point: 0,
            output_zero_point: 0,
            // BsEntry::default() contributes the unit BS multiplier. Cancel
            // that stage's measured gain so this 1x1, weight-1 convolution
            // is an exact identity for the 0..60 values used below.
            multiplier: Multiplier::for_unit_bs(1.0),
        }),
        padding: Some([0, 0]),
        activation: conv::Activation::None,
        depthwise: false,
    }
}

fn numeric_pooling_shape(case: NumericCase, method: PoolingMethod) -> PoolingShape {
    PoolingShape {
        input_width: case.width,
        input_height: case.height,
        input_channels: 1,
        output_width: (case.width - case.kernel_width) / case.stride_x + 1,
        output_height: (case.height - case.kernel_height) / case.stride_y + 1,
        output_channels: 1,
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
        activation: Activation::None,
    }
}

fn numeric_input(case: NumericCase) -> Vec<i8> {
    let mut input = Vec::with_capacity((case.width * case.height) as usize);
    for y in 0..case.height {
        for x in 0..case.width {
            // Bounded below the int8 convolution's measured exact-through-64
            // region, but deliberately non-monotonic so max/min cannot pass
            // through a fixed corner-selection bug.
            input.push(((13 * x + 7 * y + 3 * x * y) % 61) as i8);
        }
    }
    input
}

fn cpu_pool(input: &[i8], shape: &PoolingShape) -> Vec<i8> {
    let mut output = Vec::with_capacity((shape.output_width * shape.output_height) as usize);
    for output_y in 0..shape.output_height {
        for output_x in 0..shape.output_width {
            let input_y = output_y * shape.stride_y;
            let input_x = output_x * shape.stride_x;
            let mut minimum = i8::MAX;
            let mut maximum = i8::MIN;
            let mut sum = 0i32;
            for kernel_y in 0..shape.kernel_height {
                for kernel_x in 0..shape.kernel_width {
                    let index =
                        ((input_y + kernel_y) * shape.input_width + input_x + kernel_x) as usize;
                    let value = input[index];
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
                    // Inputs are non-negative. The PPU's fixed-point
                    // reciprocals can differ from ideal rounding by one LSB;
                    // the hardware assertion below carries that tolerance.
                    ((sum + count / 2) / count) as i8
                }
            });
        }
    }
    output
}

fn page_aligned(bytes: usize) -> usize {
    bytes.max(1).next_multiple_of(4096)
}

fn run_numeric_case(case: NumericCase, method: PoolingMethod) -> (Vec<i8>, Vec<i8>) {
    let bypass = numeric_bypass_shape(case);
    let pooling = numeric_pooling_shape(case, method);
    let input = numeric_input(case);
    let expected = cpu_pool(&input, &pooling);
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let buf_in = Buffer::new(fd, page_aligned(input.len()), &file);
        ptr::write_bytes(buf_in.host_ptr, 0, buf_in.size);
        ptr::copy_nonoverlapping(input.as_ptr().cast::<u8>(), buf_in.host_ptr, input.len());

        let weight_bytes = bypass.weight_bytes(BYPASS_KERNELS) as usize;
        let mut packed_weights = vec![0; weight_bytes];
        pack_hwcf_to_rocket_weights(&[1], 1, 1, 1, 1, 1, &mut packed_weights)
            .expect("failed to pack the identity convolution weight");
        let buf_w = Buffer::new(fd, page_aligned(weight_bytes), &file);
        ptr::write_bytes(buf_w.host_ptr, 0, buf_w.size);
        ptr::copy_nonoverlapping(packed_weights.as_ptr(), buf_w.host_ptr, weight_bytes);

        let bias_bytes = bypass.bs_buffer_bytes();
        let buf_bias = Buffer::new(fd, page_aligned(bias_bytes), &file);
        ptr::write_bytes(buf_bias.host_ptr, 0, buf_bias.size);
        let entries = vec![BsEntry::default(); bypass.padded_out_channels() as usize];
        write_bs_buffer(
            std::slice::from_raw_parts_mut(buf_bias.host_ptr, bias_bytes),
            &entries,
        );

        let intermediate_bytes = bypass.output_scratch_bytes(BYPASS_KERNELS);
        let buf_mid = Buffer::new(fd, page_aligned(intermediate_bytes), &file);
        ptr::write_bytes(buf_mid.host_ptr, 0xA5, buf_mid.size);

        let output_bytes =
            ((pooling.output_width * pooling.output_height).next_multiple_of(4) * 16) as usize;
        let buf_out = Buffer::new(fd, page_aligned(output_bytes), &file);
        ptr::write_bytes(buf_out.host_ptr, 0xA5, buf_out.size);

        for buffer in [&buf_in, &buf_w, &buf_bias, &buf_mid, &buf_out] {
            fini_bo(fd, buffer.handle).expect("failed to sync pooling test BO for the NPU");
        }

        let buffers = Bufs {
            file,
            fd,
            buf_in,
            buf_w,
            buf_bias,
            buf_mid,
            buf_out,
        };
        run_task_chain(&buffers, &bypass, &pooling);

        let intermediate = std::slice::from_raw_parts(buffers.buf_mid.host_ptr, intermediate_bytes);
        for (pixel, &want) in input.iter().enumerate() {
            let got = intermediate[pixel * 16] as i8;
            assert_eq!(
                got, want,
                "{} identity bypass mismatch at pixel {pixel}: expected {want}, got {got}",
                case.name
            );
        }

        let raw_output = std::slice::from_raw_parts(buffers.buf_out.host_ptr, output_bytes);
        let actual = (0..expected.len())
            .map(|pixel| raw_output[pixel * 16] as i8)
            .collect();
        buffers.close();
        (actual, expected)
    }
}

fn assert_numeric_case(case: NumericCase, method: PoolingMethod, tolerance: i16) {
    let (actual, expected) = run_numeric_case(case, method);
    assert_eq!(actual.len(), expected.len(), "{} output length", case.name);
    let mut mismatches = Vec::new();
    for (index, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
        if (i16::from(got) - i16::from(want)).abs() > tolerance && mismatches.len() < 8 {
            let shape = numeric_pooling_shape(case, method);
            let y = index as u32 / shape.output_width;
            let x = index as u32 % shape.output_width;
            mismatches.push(format!("[{y}, {x}] want {want} got {got}"));
        }
    }
    assert!(
        mismatches.is_empty(),
        "{} {method:?}: numerical pooling mismatches (tolerance {tolerance}): {}",
        case.name,
        mismatches.join(", ")
    );
}

fn assert_numeric_matrix(method: PoolingMethod, tolerance: i16) {
    for case in NUMERIC_CASES {
        assert_numeric_case(case, method, tolerance);
    }
}

#[test]
fn numerical_dimension_matrix_has_expected_task_chain() {
    for case in NUMERIC_CASES {
        for method in [PoolingMethod::Max, PoolingMethod::Min, PoolingMethod::Avg] {
            let bypass = numeric_bypass_shape(case);
            let pooling = numeric_pooling_shape(case, method);
            let programs = build_pooling_via_dpu_bypass_tasks(
                &bypass,
                BYPASS_KERNELS,
                &pooling,
                &PoolingViaBypassBuffers {
                    input_addr: 0x1000,
                    weights_addr: 0x2000,
                    bias_addr: 0x3000,
                    bypass_output_addr: 0x4000,
                    output_addr: 0x8000,
                },
            );
            assert_eq!(
                programs.len(),
                1 + PoolingPlan::new(pooling).tiles().len(),
                "{} {method:?} task count",
                case.name
            );
            assert!(
                programs.iter().all(|program| !program.is_empty()),
                "{} {method:?} emitted an empty task",
                case.name
            );
            assert_eq!(
                cpu_pool(&numeric_input(case), &pooling).len(),
                (pooling.output_width * pooling.output_height) as usize,
                "{} {method:?} CPU reference extent",
                case.name
            );
        }
    }
}

#[test]
#[ignore = "needs the real NPU device -- validates max pooling numerically across dimensions"]
fn max_pooling_dimension_matrix_matches_cpu_reference() {
    assert_numeric_matrix(PoolingMethod::Max, 0);
}

#[test]
#[ignore = "needs the real NPU device -- validates min pooling numerically across dimensions"]
fn min_pooling_dimension_matrix_matches_cpu_reference() {
    assert_numeric_matrix(PoolingMethod::Min, 0);
}

#[test]
#[ignore = "needs the real NPU device -- validates average pooling numerically across dimensions"]
fn average_pooling_dimension_matrix_matches_cpu_reference() {
    assert_numeric_matrix(PoolingMethod::Avg, 1);
}

#[test]
#[ignore = "needs the real NPU device -- focused DPU bypass -> PPU tile 0 -> PPU tile 1 job"]
fn vendor_style_three_task_tiled_pooling_matches_cpu_reference() {
    assert_numeric_case(TWO_HORIZONTAL_TILES, PoolingMethod::Max, 0);
}

macro_rules! completes_and_tracks_input_test {
    ($name:ident, $method:expr) => {
        #[test]
        #[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
        fn $name() {
            for input_fill in [10u8, 118, 200] {
                let pixels = run_uniform_pooling_via_bypass($method, input_fill, 2);
                assert!(
                    pixels.iter().all(|&p| p == pixels[0]),
                    "input_fill={input_fill}: expected all 4 output pixels identical \
                     (uniform input), got {pixels:?}"
                );
            }
            let low = run_uniform_pooling_via_bypass($method, 10, 2)[0];
            let high = run_uniform_pooling_via_bypass($method, 200, 2)[0];
            assert_ne!(
                low, high,
                "output pixel value didn't change between input_fill=10 ({low}) and \
                 input_fill=200 ({high}) -- suggests the op isn't really reading the input"
            );
        }
    };
}

completes_and_tracks_input_test!(
    pooling_via_bypass_max_completes_and_output_tracks_input,
    PoolingMethod::Max
);
completes_and_tracks_input_test!(
    pooling_via_bypass_min_completes_and_output_tracks_input,
    PoolingMethod::Min
);
completes_and_tracks_input_test!(
    pooling_via_bypass_avg_completes_and_output_tracks_input,
    PoolingMethod::Avg
);

/// Diagnostic, not a correctness check. Second round update: with the
/// DPU_RDMA kick fix, stage 1 now genuinely writes buf_mid (no longer
/// stuck at the 0xAA sentinel) -- but `pooling_via_bypass_min_...` then
/// found a *new* bug: pooling a uniform-input result produced
/// non-identical output pixels ([255,255,0,0]), the first real exercise
/// of PPU_RDMA's fetch-side stride math against a genuinely DPU-written
/// (not CPU-written) buffer. Dumps a wider region (512 bytes, covering the
/// full 4x4x1 image at 16 bytes/pixel plus the padded second
/// task_output_channels atomic group) of both buffers using `Min` (the
/// method that showed the clearest non-uniform symptom) to see whether
/// DPU's write-side layout is genuinely uniform across all 16 pixel
/// positions (bug is in PPU_RDMA's read-side stride formulas) or not
/// (bug is upstream, in the bypass conv's own write).
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there; \
            diagnostic only, not a pass/fail check -- read the printed hex dumps"]
fn pooling_via_bypass_dump_intermediate_and_output_buffers() {
    let b = setup(200, 2, 0xAA);
    unsafe { run_task_chain(&b, &bypass_shape(), &pooling_shape(PoolingMethod::Min)) };

    unsafe {
        prep_bo(b.fd, b.buf_mid.handle, 2_000_000_000).ok();
        for (label, buf) in [
            ("buf_mid (stage 1 output)", &b.buf_mid),
            ("buf_out (stage 2 output)", &b.buf_out),
        ] {
            let raw = std::slice::from_raw_parts(buf.host_ptr, 512);
            eprintln!("{label}, first 512 bytes:");
            for (row, chunk) in raw.chunks(32).enumerate() {
                let hex: String = chunk.iter().map(|b| format!("{b:02x} ")).collect();
                eprintln!("  {:04x}: {hex}", row * 32);
            }
            let unchanged = raw.iter().filter(|&&b| b == 0xAA).count();
            eprintln!("{label}: {unchanged}/512 bytes still == 0xAA");
        }
    }
    unsafe { b.close() };
}

/// Diagnostic, not a correctness check. Added after the *same* nominal
/// dispatch (Min pooling, input_fill=200, weight_fill=2) produced two
/// different results across separate test-process invocations even under
/// `--test-threads=1` (serial execution, ruling out cross-thread races):
/// [255,255,0,0] from `pooling_via_bypass_min_completes_and_output_tracks_input`
/// vs. a uniform 250 from `pooling_via_bypass_dump_intermediate_and_output_buffers`,
/// with `buf_mid` confirmed byte-identical (uniform, correct) in both
/// cases. Since the input to stage 2 is proven identical, whatever's
/// non-deterministic must be internal to PPU/PPU_RDMA's own hardware state
/// -- the leading hypothesis is the ping-pong buffering bits
/// (`PpuSPointer`/`PpuRdmaSPointer`'s `pointer_pp_mode`/`pp_en`, set
/// identically on every dispatch here) not being correctly synchronized
/// with the engine's actual internal toggle state, which may persist
/// across dispatches regardless of which process/fd issued them.
///
/// Repeats the identical dispatch (fresh buffers each time, same shape,
/// same fills) several times in a row within one process and prints each
/// run's first 4 output bytes, to see whether results alternate between a
/// small fixed set of states (supports the ping-pong theory) or vary
/// unpredictably (points to something else, e.g. a genuine race against
/// the job's actual hardware completion rather than just PREP_BO
/// returning).
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there; \
            diagnostic only, not a pass/fail check -- read the printed sequence"]
fn pooling_via_bypass_repeat_dispatch_dump() {
    for i in 0..6 {
        let pixels = run_uniform_pooling_via_bypass(PoolingMethod::Min, 200, 2);
        eprintln!("repeat #{i}: output pixels = {pixels:?}");
    }
}

/// Diagnostic, not a correctness check. Extends
/// `pooling_via_bypass_repeat_dispatch_dump`'s same-process repeat
/// experiment to *alternating* methods, to mirror the actual failure shape
/// found running the full ignored suite together on real hardware:
/// `max_pooling_dimension_matrix_matches_cpu_reference` passed repeatedly
/// run alone, but hung with a kernel-side "NPU job timed out" (task 2 of 2
/// never completing) when run in the same process alongside the avg/min
/// matrix tests. That points at hardware state left over from a *different*
/// method's dispatch, not a per-method register/opcode bug -- the kernel
/// driver programs an identical CNA/CORE/PC register sequence regardless of
/// method, and the PPU itself is configured entirely from the regcmd stream
/// (the kernel driver never maps PPU registers at all, so nothing on that
/// side varies by method either).
///
/// Uses the same small uniform 4x4 shape as `pooling_via_bypass_repeat_dispatch_dump`
/// (not the large numeric matrix) so a hang surfaces in seconds, on a
/// specific (round, method) pair, instead of requiring the multi-minute
/// full-matrix sweep to reproduce it.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there; \
            diagnostic only, not a pass/fail check -- if it hangs instead of finishing, that IS \
            the finding; check the round/method it stopped after and cross-reference the kernel's \
            'timeout state' dev_err (task index, PC raw_status, CNA_S_STATUS, CORE_S_STATUS)"]
fn pooling_via_bypass_interleaved_methods_dump() {
    let methods = [PoolingMethod::Avg, PoolingMethod::Min, PoolingMethod::Max];
    for round in 0..6 {
        for &method in &methods {
            let start = std::time::Instant::now();
            let pixels = run_uniform_pooling_via_bypass(method, 200, 2);
            eprintln!(
                "round {round} {method:?}: output pixels = {pixels:?} ({:?})",
                start.elapsed()
            );
        }
    }
}

/// Diagnostic, not a correctness check. `pooling_via_bypass_interleaved_methods_dump`
/// alternated methods but only on the small single-tile 4x4 uniform shape, and
/// ran clean 18/18 -- it did not reproduce the hang. The kernel's new
/// timeout-state dev_err (see rocket_job.c's rocket_job_timedout) shows that
/// when this hangs for real, the stuck task is PPU-only (CNA_S_STATUS and
/// CORE_S_STATUS both idle -- expected, since a single-tile job's task 1 never
/// touches those blocks) and PC's raw interrupt status has no completion or
/// error bit set at all: the PPU genuinely never signals anything, not a
/// lost/masked interrupt.
///
/// This version adds `TWO_HORIZONTAL_TILES`, the one shape in NUMERIC_CASES
/// whose job is 3 tasks -- DPU bypass, then *two* PPU tiles back-to-back --
/// since that is the only shape where the PPU itself dispatches twice in
/// immediate succession, the most direct way to exercise any state the PPU
/// carries between dispatches. Also runs more rounds and mixes in one bigger
/// single-tile shape, since the real failing suite issues far more total
/// dispatches than the small-shape repeat test did.
///
/// First run (fixed `[Avg, Min, Max]` order every time) found `Avg` on
/// `two_horizontal_tiles` timing out 20/20 rounds while `Min`/`Max` on the
/// same shape were clean -- but that run always dispatched `Avg` *first*
/// after switching to a new case's tile geometry, since the inner loop order
/// never varied. That confounds "this method" with "the first dispatch after
/// this shape switch"; a bug in whatever state carries over between a shape
/// switch and the next dispatch would look identical to a per-method bug
/// under that fixed ordering. This version rotates the method order by
/// `(round + case_index) % 3` so each method takes a turn being first (and
/// last) across the run, and prints the actual order used each time so a
/// hang's real correlate -- specific method, or specific position -- is
/// readable directly from the log instead of inferred.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there; \
            diagnostic only, not a pass/fail check -- if it hangs instead of finishing, that IS \
            the finding; check whether the round/case/method it stopped after correlates with a \
            specific method or a specific position in the printed order, and cross-reference the \
            kernel's 'timeout state' dev_err (task index, PC raw_status, CNA_S_STATUS, CORE_S_STATUS)"]
fn pooling_via_bypass_numeric_interleaved_methods_dump() {
    let cases = [NUMERIC_CASES[0], NUMERIC_CASES[4], TWO_HORIZONTAL_TILES];
    let base_methods = [PoolingMethod::Avg, PoolingMethod::Min, PoolingMethod::Max];
    for round in 0..20 {
        for (case_index, case) in cases.iter().enumerate() {
            let mut methods = base_methods;
            methods.rotate_left((round + case_index) % base_methods.len());
            for (position, method) in methods.into_iter().enumerate() {
                let start = std::time::Instant::now();
                let (actual, expected) = run_numeric_case(*case, method);
                eprintln!(
                    "round {round} {} order={methods:?} pos={position} {method:?}: match={} ({:?})",
                    case.name,
                    actual == expected,
                    start.elapsed()
                );
            }
        }
    }
}

/// Phase 3 of the ukernel roadmap: fused activation on a pooling op that
/// has a real DPU stage ahead of PPU (this bypass path). Same
/// domain-independent proof `conv_phase1_validation_hw.rs`'s clamped-
/// activation test used for conv: `cmp: 0`
/// clamps to 0 in any numeric scale, so if the bypass stage is really
/// applying `pooling_shape.activation` (not silently dropping it),
/// every input_fill should read back as the same constant regardless of
/// what the un-clamped value would have been -- proves the fusion wiring
/// without needing to know this suite's real (placeholder scale=1.0)
/// numeric domain, same caveat as conv's own `cmp` open question.
///
/// The bypass stage's conv now goes through `conv.rs`'s capture-derived
/// builder, which fuses activation via the BN stage instead of the retired
/// Mesa-derived builder's BS stage (see `pooling.rs`'s own doc comment on
/// `build_pooling_via_dpu_bypass_tasks`) -- `cmp: 0` clamping to a
/// constant zero is domain- and stage-invariant, so this specific test
/// still proves the fusion wiring works, but a real hardware run after
/// this migration is the first one to exercise BN-stage fusion in this
/// exact bypass-then-pool composition.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn pooling_via_bypass_relux_cmp_zero_forces_constant_output() {
    let mut outputs = Vec::new();
    for input_fill in [0u8, 50, 100, 150, 200, 255] {
        let pixels = run_uniform_pooling_via_bypass_with_activation(
            PoolingMethod::Max,
            input_fill,
            2,
            Activation::Relux { cmp: 0 },
        );
        assert!(
            pixels.iter().all(|&p| p == pixels[0]),
            "input_fill={input_fill}: expected all 4 output pixels identical \
             (uniform input), got {pixels:?}"
        );
        outputs.push(pixels[0]);
    }
    assert!(
        outputs.iter().all(|&o| o == outputs[0]),
        "Relux{{cmp: 0}} should force the same constant output regardless of input_fill \
         in any numeric domain, but got {outputs:?} across fills [0,50,100,150,200,255] -- \
         fused activation on the bypass stage isn't being applied, or \
         pooling_shape.activation isn't reaching it"
    );
}
