//! Oracle-checked PPU pooling on real RK3588 hardware.
//!
//! # Why this exists
//!
//! `pooling.rs` is 1200 lines of capture-derived register programming with a
//! healthy unit-test suite -- and until this file, **nothing in this repo ran
//! it on the device**. Its own module doc points at a standalone hardware
//! suite that was retired, and the only thing that ever built a
//! `UkernelShape::Pooling` was a legacy one-byte test tag whose own comment
//! said "NOT hardware-validated" and which has since been removed. The
//! `PoolingDef` wire format now makes the PPU reachable from a compiled
//! model, which is exactly the moment that gap stops being academic.
//!
//! # What the oracle is
//!
//! A CPU reference over the same logical window, compared per element. The
//! input varies in y, x **and** channel, which is the whole point: the
//! retired suite's uniform-buffer tests stayed green through a
//! `PPU_RDMA_SRC_LINE_STRIDE` that was 16x too large, because reading the
//! wrong row of an everywhere-identical buffer still returns the right byte
//! (see `build_pooling_tile_task`'s own comment on that bug). A pattern that
//! differs per pixel cannot hide a stride, a tile offset or a surface.
//!
//! # What is checked exactly, and what is not
//!
//! Max and min are comparisons -- the result is one of the inputs, so those
//! run at tolerance zero. Average is not: the PPU has no divider and
//! multiplies by a per-axis reciprocal held as `fp16(65536/k)`, so a divisor
//! that is not a power of two is approximate by construction (~0.02-0.05%,
//! bit-identical to the vendor's own runtime). Those cases carry a tolerance
//! and say so.
//!
//! Cross-compile and run on the board:
//!
//! ```text
//! CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!   cargo test --target aarch64-unknown-linux-gnu --release \
//!     --test pooling_oracle_hw --no-run
//!
//! ./pooling_oracle_hw-<hash> --ignored --nocapture
//! ```

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr, sync::Mutex, time::Instant};

#[path = "support/dispatch.rs"]
mod dispatch;

use iree_rocket_hal::rocket::{
    builders::RegCmd,
    device::{Buffer, JobDesc, close_bo, fini_bo, prep_bo, submit_jobs},
    pooling::{PoolingBuffers, PoolingMethod, PoolingPlan, PoolingPrecision, PoolingShape},
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const PAGE_BYTES: usize = 4096;
const FEATURE_ATOM_BYTES: usize = 16;
const COMPLETION_TIMEOUT_NS: u64 = 5_000_000_000;

/// Bytes the DPU never wrote read back as this, so a tile the PPU skipped
/// entirely is a loud mismatch rather than a plausible zero.
const OUTPUT_SENTINEL: u8 = 0xa5;

/// The RK3588 NPU is one shared device and Rust's harness runs `#[ignore]`d
/// tests in a binary concurrently, so serialize them the way every other
/// hardware test in this crate does.
static NPU_TEST_LOCK: Mutex<()> = Mutex::new(());

fn page_aligned_size(size: usize) -> usize {
    size.div_ceil(PAGE_BYTES) * PAGE_BYTES
}

fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    if value == 0.0 {
        return sign;
    }
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = bits & 0x7f_ffff;
    assert!(
        (1..=30).contains(&exponent) && mantissa & 0x1fff == 0,
        "{value} is not exactly representable in fp16"
    );
    sign | ((exponent as u16) << 10) | ((mantissa >> 13) as u16)
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exponent = ((bits >> 10) & 0x1f) as u32;
    let mantissa = (bits & 0x3ff) as u32;
    let word = match exponent {
        0 if mantissa == 0 => sign << 31,
        0x1f => (sign << 31) | 0x7f80_0000 | (mantissa << 13),
        0 => {
            let mut exponent = -1i32;
            let mut mantissa = mantissa;
            while mantissa & 0x400 == 0 {
                mantissa <<= 1;
                exponent -= 1;
            }
            (sign << 31) | (((exponent + 127 - 15) as u32) << 23) | ((mantissa & 0x3ff) << 13)
        }
        _ => (sign << 31) | ((exponent + 127 - 15) << 23) | (mantissa << 13),
    };
    f32::from_bits(word)
}

#[derive(Clone, Copy, Debug)]
struct PoolCase {
    label: &'static str,
    width: u32,
    height: u32,
    channels: u32,
    kernel: (u32, u32),
    stride: (u32, u32),
    /// `[left, top, right, bottom]`.
    pad: [u32; 4],
    method: PoolingMethod,
    precision: PoolingPrecision,
}

impl PoolCase {
    fn shape(self) -> PoolingShape {
        let output_width = output_extent(
            self.width,
            self.kernel.0,
            self.stride.0,
            self.pad[0],
            self.pad[2],
        );
        let output_height = output_extent(
            self.height,
            self.kernel.1,
            self.stride.1,
            self.pad[1],
            self.pad[3],
        );
        PoolingShape {
            input_width: self.width,
            input_height: self.height,
            input_channels: self.channels,
            output_width,
            output_height,
            output_channels: self.channels,
            precision: self.precision,
            kernel_width: self.kernel.0,
            kernel_height: self.kernel.1,
            stride_x: self.stride.0,
            stride_y: self.stride.1,
            method: self.method,
            pad_left: self.pad[0],
            pad_top: self.pad[1],
            pad_right: self.pad[2],
            pad_bottom: self.pad[3],
            // Derived, never chosen -- the same rule the runtime applies when
            // it decodes a `PoolingDef`, and the reason that wire format has
            // no field for it.
            pad_value: self.method.pad_fill_value(self.precision),
        }
    }

    /// Whether the comparison can run at tolerance zero.
    ///
    /// Max and min return one of their inputs. Average multiplies by
    /// `fp16(65536/k)`, which is exact only when `k` is a power of two --
    /// and even then the fp16 product rounds unless the sum is small.
    fn tolerance(self) -> f32 {
        match (self.method, self.precision) {
            (PoolingMethod::Max | PoolingMethod::Min, _) => 0.0,
            // Measured slack, not a guess at rounding: the reciprocal itself
            // carries ~0.05% and the fp16 product another ulp.
            (PoolingMethod::Avg, PoolingPrecision::Fp16) => 0.05,
            (PoolingMethod::Avg, PoolingPrecision::Int8) => 1.0,
        }
    }
}

fn output_extent(input: u32, kernel: u32, stride: u32, before: u32, after: u32) -> u32 {
    (input + before + after - kernel) / stride + 1
}

/// A value that differs in y, x and channel, so no permutation of the three
/// can cancel out. Kept inside `[-8, 8]`: exact in fp16, inside int8, and
/// small enough that a 7x7 average's sum stays exact in fp16 too.
fn input_at(y: usize, x: usize, channel: usize) -> f32 {
    (((y * 13 + x * 7 + channel * 3 + (y * x) % 5) % 17) as i32 - 8) as f32
}

/// The CPU reference for one output element.
///
/// Padded taps are the reduction's identity: `-inf` for max and `0` for
/// average, matching what the hardware is programmed to fill with. Average
/// divides by the whole `kh * kw` whether or not a tap was padding, which is
/// count-include-pad -- the PPU has no way to count valid taps, and it is why
/// a padded average from a framework that excludes them cannot be offloaded
/// as-is.
fn expected(case: PoolCase, out_y: usize, out_x: usize, channel: usize) -> f32 {
    let origin_y = out_y as isize * case.stride.1 as isize - case.pad[1] as isize;
    let origin_x = out_x as isize * case.stride.0 as isize - case.pad[0] as isize;
    let mut sum = 0.0f32;
    let mut best: Option<f32> = None;
    for ky in 0..case.kernel.1 as isize {
        for kx in 0..case.kernel.0 as isize {
            let y = origin_y + ky;
            let x = origin_x + kx;
            let inside =
                (0..case.height as isize).contains(&y) && (0..case.width as isize).contains(&x);
            if !inside {
                continue;
            }
            let value = input_at(y as usize, x as usize, channel);
            sum += value;
            best = Some(match case.method {
                PoolingMethod::Max => best.map_or(value, |best: f32| best.max(value)),
                PoolingMethod::Min => best.map_or(value, |best: f32| best.min(value)),
                PoolingMethod::Avg => 0.0,
            });
        }
    }
    match case.method {
        PoolingMethod::Avg => sum / (case.kernel.0 * case.kernel.1) as f32,
        _ => best.expect("every output window overlaps the image"),
    }
}

/// What one pooling dispatch left in its output cube.
struct Dispatched {
    raw: Vec<u8>,
    elapsed: std::time::Duration,
    tiles: usize,
}

/// Runs `shape` once on the device and returns the raw output cube.
///
/// `input` gives each logical element; `patch` sees every tile's register
/// program before it is written to a command buffer, which is how the pad
/// fill probe programs `PPU_PADDING_VALUE_2_CFG` -- a field the planner
/// always zeroes.
fn dispatch_pool(
    shape: PoolingShape,
    input: &dyn Fn(usize, usize, usize) -> f32,
    patch: &dyn Fn(&mut Vec<RegCmd>),
) -> Dispatched {
    let precision = shape.precision;
    let channels_per_atom = precision.channels_per_atom() as usize;
    let element_bytes = precision.element_bytes() as usize;
    let surfaces = shape.programmed_channels() as usize / channels_per_atom;
    let in_pixels = (shape.input_width * shape.input_height) as usize;
    let out_pixels = (shape.output_width * shape.output_height) as usize;
    // Both cube strides are rounded up to four pixels by
    // `build_pooling_tile_task`; a buffer laid out any other way puts every
    // surface past the first at the wrong offset.
    let in_surface_pixels = in_pixels.next_multiple_of(4);
    let out_surface_pixels = out_pixels.next_multiple_of(4);

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        // ISSUES C16: push this case's buffers up the file's IOVA space. The
        // allocator is first-fit from zero per file, so a ballast BO created
        // first places everything after it. Unset, nothing changes.
        let ballast = BALLAST_BYTES.with(std::cell::Cell::get);
        let ballast = (ballast > 0).then(|| Buffer::new(fd, ballast, &file));
        let input_buffer = Buffer::new(
            fd,
            page_aligned_size(surfaces * in_surface_pixels * FEATURE_ATOM_BYTES),
            &file,
        );
        ptr::write_bytes(input_buffer.host_ptr, 0, input_buffer.size);
        for channel in 0..shape.input_channels as usize {
            for y in 0..shape.input_height as usize {
                for x in 0..shape.input_width as usize {
                    let offset =
                        (channel / channels_per_atom) * in_surface_pixels * FEATURE_ATOM_BYTES
                            + (y * shape.input_width as usize + x) * FEATURE_ATOM_BYTES
                            + (channel % channels_per_atom) * element_bytes;
                    let value = input(y, x, channel);
                    match precision {
                        PoolingPrecision::Fp16 => ptr::write(
                            input_buffer.host_ptr.add(offset) as *mut u16,
                            f32_to_f16(value),
                        ),
                        PoolingPrecision::Int8 => {
                            ptr::write(input_buffer.host_ptr.add(offset), value as i8 as u8)
                        }
                    }
                }
            }
        }

        let output = Buffer::new(
            fd,
            page_aligned_size(surfaces * out_surface_pixels * FEATURE_ATOM_BYTES),
            &file,
        );
        ptr::write_bytes(output.host_ptr, OUTPUT_SENTINEL, output.size);

        let plan = PoolingPlan::new(shape);
        let mut programs = plan.programs_with_buffers(&PoolingBuffers {
            input_addr: input_buffer.dma_address,
            output_addr: output.dma_address,
        });
        for program in &mut programs {
            patch(program);
        }
        let mut command_buffers = Vec::with_capacity(programs.len());
        for commands in &programs {
            let buffer = Buffer::new(
                fd,
                page_aligned_size(commands.len() * mem::size_of::<u64>()),
                &file,
            );
            ptr::write_bytes(buffer.host_ptr, 0, buffer.size);
            let words = std::slice::from_raw_parts_mut(buffer.host_ptr as *mut u64, commands.len());
            for (word, command) in words.iter_mut().zip(commands) {
                *word = command.0;
            }
            command_buffers.push((buffer, commands.len() as u32));
        }

        for handle in [input_buffer.handle, output.handle] {
            fini_bo(fd, handle).unwrap();
        }
        for (buffer, _) in &command_buffers {
            fini_bo(fd, buffer.handle).unwrap();
        }

        // Every horizontal tile is one task of a single job, which is what
        // the driver's dispatch arm builds -- tasks within a job run in order
        // on one core, and the tiles write disjoint column ranges of the same
        // output cube.
        let tasks: Vec<(u32, u32)> = command_buffers
            .iter()
            .map(|(buffer, count)| (buffer.dma_address, *count))
            .collect();
        let mut in_handles: Vec<u32> = command_buffers
            .iter()
            .map(|(buffer, _)| buffer.handle)
            .collect();
        in_handles.push(input_buffer.handle);
        let out_handles = [output.handle];
        let jobs = [JobDesc {
            tasks: &tasks,
            in_handles: &in_handles,
            out_handles: &out_handles,
        }];

        let started = Instant::now();
        submit_jobs(fd, &jobs)
            .unwrap_or_else(|error| panic!("SUBMIT failed ({} tile(s)): {error}", tasks.len()));
        prep_bo(fd, output.handle, COMPLETION_TIMEOUT_NS)
            .unwrap_or_else(|error| panic!("did not complete: {error}"));
        let elapsed = started.elapsed();

        let raw = std::slice::from_raw_parts(output.host_ptr, output.size).to_vec();

        for (buffer, _) in &command_buffers {
            close_bo(fd, buffer.handle).unwrap();
        }
        close_bo(fd, input_buffer.handle).unwrap();
        close_bo(fd, output.handle).unwrap();
        if let Some(ballast) = ballast {
            close_bo(fd, ballast.handle).unwrap();
        }
        Dispatched {
            raw,
            elapsed,
            tiles: tasks.len(),
        }
    }
}

/// Reads one logical output element of a cube `dispatch_pool` returned.
fn read_output(shape: PoolingShape, raw: &[u8], out_y: usize, out_x: usize, channel: usize) -> f32 {
    let channels_per_atom = shape.precision.channels_per_atom() as usize;
    let element_bytes = shape.precision.element_bytes() as usize;
    let out_surface_pixels = (shape.output_width * shape.output_height) as usize;
    let out_surface_pixels = out_surface_pixels.next_multiple_of(4);
    let offset = (channel / channels_per_atom) * out_surface_pixels * FEATURE_ATOM_BYTES
        + (out_y * shape.output_width as usize + out_x) * FEATURE_ATOM_BYTES
        + (channel % channels_per_atom) * element_bytes;
    match shape.precision {
        PoolingPrecision::Fp16 => f16_to_f32(u16::from_le_bytes([raw[offset], raw[offset + 1]])),
        PoolingPrecision::Int8 => f32::from(raw[offset] as i8),
    }
}

fn run_case(case: PoolCase) {
    let _guard = NPU_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let shape = case.shape();
    shape.validate();

    let label = format!(
        "{} {}x{} C{} k{}x{} s{}x{} pad{:?} {:?}",
        case.label,
        case.width,
        case.height,
        case.channels,
        case.kernel.1,
        case.kernel.0,
        case.stride.1,
        case.stride.0,
        case.pad,
        case.precision,
    );
    let Dispatched {
        raw,
        elapsed,
        tiles,
    } = dispatch_pool(shape, &input_at, &|_| {});

    let tolerance = case.tolerance();
    let mut mismatches = 0usize;
    let mut first = None;
    let mut worst = 0.0f32;
    for out_y in 0..shape.output_height as usize {
        for out_x in 0..shape.output_width as usize {
            for channel in 0..case.channels as usize {
                let want = expected(case, out_y, out_x, channel);
                let got = read_output(shape, &raw, out_y, out_x, channel);
                let difference = (got - want).abs();
                worst = worst.max(difference);
                if !got.is_finite() || difference > tolerance {
                    mismatches += 1;
                    first.get_or_insert((out_y, out_x, channel, want, got));
                }
            }
        }
    }

    let total = (shape.output_width * shape.output_height * case.channels) as usize;
    // The dispatch clock is what separates a shape result from a
    // watchdog kill: `prep_bo` returns success over an error-signalled
    // fence, so a killed job looks like a wrong answer unless the wall
    // time is in the report. Healthy pools here run in single-digit
    // milliseconds; a kill costs ~500 ms per tile.
    assert!(
        mismatches == 0,
        "{label}: {mismatches} of {total} elements wrong (tolerance {tolerance}), \
         max|diff| {worst}, first at {:?}, dispatch {:.1} ms{}",
        first.unwrap(),
        elapsed.as_secs_f64() * 1e3,
        if elapsed.as_millis() >= 150 {
            " -- OVER THE TIMEOUT FLOOR, suspect a watchdog kill rather than a shape result"
        } else {
            ""
        },
    );
    println!(
        "  {label}: ok, {tiles} tile(s), max|diff| {worst}{}",
        dispatch::note(elapsed),
    );
}

/// Runs every case before reporting, so one bad shape does not hide the
/// ones after it -- the same reason `run_hardware_case_matrix` accumulates.
///
/// The pause is not politeness. This board hangs a dispatch far more often
/// when the machine was busy in the preceding second -- measured elsewhere
/// in this repo at 6-8 failures in 8 back-to-back runs against 0 in 10 with
/// a one-second gap, on shapes that are otherwise exact. Without it the
/// padded 3x3/stride-2 case here fails in sequence and passes 3/3 in
/// isolation, which is a statement about the device rather than about
/// pooling. The dispatch clock in the failure message is what still tells
/// a real hang from that.
const SETTLE: std::time::Duration = std::time::Duration::from_millis(1200);

thread_local! {
    /// Bytes of ballast allocated ahead of a case's buffers; see `run_case`.
    static BALLAST_BYTES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn run_cases(cases: &[PoolCase]) {
    let mut failures = Vec::new();
    for case in cases {
        std::thread::sleep(SETTLE);
        if let Err(failure) = std::panic::catch_unwind(|| run_case(*case)) {
            let message = failure
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| {
                    failure
                        .downcast_ref::<&str>()
                        .map(|text| (*text).to_string())
                })
                .unwrap_or_else(|| "non-string panic".to_string());
            println!("  FAIL {message}");
            failures.push(message);
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} pooling cases failed; every failure is printed above",
        failures.len(),
        cases.len(),
    );
}

/// An env-driven width sweep, for bisecting a boundary this ladder finds.
///
/// `tiled_pooling_matches_oracle` turned up a 3x3/stride-1 hang and the
/// first question was always going to be "is it the tiling or the width",
/// which needs a sweep rather than another committed case.
///
///     ROCKET_POOL_WIDTHS=51,64,100,129,200 \
///       ./pooling_oracle_hw pooling_width_probe --ignored --nocapture
///
/// `ROCKET_POOL_KERNEL` and `ROCKET_POOL_STRIDE` default to 3 and 1.
#[test]
#[ignore = "needs /dev/accel/accel0 -- env-driven, see the doc comment"]
fn pooling_width_probe() {
    let Ok(widths) = std::env::var("ROCKET_POOL_WIDTHS") else {
        println!("\n  pooling_width_probe: set ROCKET_POOL_WIDTHS to run it, e.g.");
        println!("    ROCKET_POOL_WIDTHS=51,64,100,129,200");
        return;
    };
    // Both accept `N` or `WxH`: the two axes are programmed by separate
    // register fields and a hang that follows one of them is a different
    // finding from one that follows both.
    let pair = |name: &str, default: u32| -> (u32, u32) {
        let Ok(value) = std::env::var(name) else {
            return (default, default);
        };
        match value.split_once('x') {
            Some((width, height)) => (
                width.parse().expect("width"),
                height.parse().expect("height"),
            ),
            None => {
                let value: u32 = value.parse().expect("value");
                (value, value)
            }
        }
    };
    let kernel = pair("ROCKET_POOL_KERNEL", 3);
    let stride = pair("ROCKET_POOL_STRIDE", 1);
    let height: u32 = std::env::var("ROCKET_POOL_HEIGHT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(8);
    let cases: Vec<PoolCase> = widths
        .split(',')
        .map(|width| PoolCase {
            label: "probe",
            width: width.parse().expect("width"),
            height,
            channels: 8,
            kernel,
            stride,
            pad: [0; 4],
            method: PoolingMethod::Max,
            precision: PoolingPrecision::Fp16,
        })
        .collect();
    run_cases(&cases);
}

/// Max pooling, the shape every classifier stem uses, at both element
/// widths and across the channel-atom boundary.
#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn max_pooling_matches_oracle() {
    run_cases(&[
        PoolCase {
            label: "max 2x2s2",
            width: 32,
            height: 32,
            channels: 32,
            kernel: (2, 2),
            stride: (2, 2),
            pad: [0; 4],
            method: PoolingMethod::Max,
            precision: PoolingPrecision::Fp16,
        },
        PoolCase {
            label: "max 2x2s2",
            width: 32,
            height: 32,
            channels: 32,
            kernel: (2, 2),
            stride: (2, 2),
            pad: [0; 4],
            method: PoolingMethod::Max,
            precision: PoolingPrecision::Int8,
        },
        // Neither channel count fills a whole 16-byte atom: fp16 20 rounds to
        // 24 and int8 12 rounds to 16, so the last surface is part padding.
        PoolCase {
            label: "max partial atom",
            width: 16,
            height: 16,
            channels: 20,
            kernel: (3, 3),
            stride: (1, 1),
            pad: [0; 4],
            method: PoolingMethod::Max,
            precision: PoolingPrecision::Fp16,
        },
        PoolCase {
            label: "max partial atom",
            width: 16,
            height: 16,
            channels: 12,
            kernel: (3, 3),
            stride: (1, 1),
            pad: [0; 4],
            method: PoolingMethod::Max,
            precision: PoolingPrecision::Int8,
        },
        // ResNet's stem pool at int8, on the input the CPU has already
        // padded (112 + 2): the shape the int8 max-pool matcher hands the
        // NPU. 64 channels is four whole int8 atoms; the extent is past
        // every other int8 case here.
        PoolCase {
            label: "max 3x3s2 resnet stem",
            width: 114,
            height: 114,
            channels: 64,
            kernel: (3, 3),
            stride: (2, 2),
            pad: [0; 4],
            method: PoolingMethod::Max,
            precision: PoolingPrecision::Int8,
        },
        // A non-square window and a non-square stride, which move the two
        // axes' kernel and stride fields independently.
        PoolCase {
            label: "max 3x2s2x1",
            width: 24,
            height: 24,
            channels: 16,
            kernel: (2, 3),
            stride: (1, 2),
            pad: [0; 4],
            method: PoolingMethod::Max,
            precision: PoolingPrecision::Fp16,
        },
    ]);
}

/// The one padded case the hardware has a documented fill value for: fp16
/// max, whose padded taps are `-inf` and therefore never win. This is what
/// says the fill reaches the border windows -- a fill of zero would beat
/// every negative input and the oracle would see it.
#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn padded_max_pooling_matches_oracle() {
    run_cases(&[
        PoolCase {
            label: "max 3x3s2 pad1",
            width: 32,
            height: 32,
            channels: 16,
            kernel: (3, 3),
            stride: (2, 2),
            pad: [1, 1, 1, 1],
            method: PoolingMethod::Max,
            precision: PoolingPrecision::Fp16,
        },
        // ResNet's stem pool at int8, with its padding on the PPU: the shape
        // `@match_pooling_nhwc_max_s2_i8_pad1` claims. The int8 fill is inert
        // (`int8_max_pad_fill_probe`), so this is the clamp-to-image answer.
        PoolCase {
            label: "max 3x3s2 pad1 resnet stem",
            width: 112,
            height: 112,
            channels: 64,
            kernel: (3, 3),
            stride: (2, 2),
            pad: [1, 1, 1, 1],
            method: PoolingMethod::Max,
            precision: PoolingPrecision::Int8,
        },
        PoolCase {
            label: "max 3x3s1 pad-lead",
            width: 16,
            height: 16,
            channels: 16,
            kernel: (3, 3),
            stride: (1, 1),
            pad: [2, 1, 0, 0],
            method: PoolingMethod::Max,
            precision: PoolingPrecision::Int8,
        },
        // Leading padding only, so the two axes' pad fields cannot be
        // confused with each other.
        PoolCase {
            label: "max 3x3s1 pad-lead",
            width: 16,
            height: 16,
            channels: 8,
            kernel: (3, 3),
            stride: (1, 1),
            pad: [2, 1, 0, 0],
            method: PoolingMethod::Max,
            precision: PoolingPrecision::Fp16,
        },
    ]);
}

/// Min pooling, unpadded and padded. A padded min is clamp-to-image on the
/// PPU whatever the fill (`int8_min_pad_fill_probe`,
/// `fp16_min_pad_fill_probe`), which is the oracle's answer while pad <
/// kernel.
#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn min_pooling_matches_oracle() {
    run_cases(&[
        PoolCase {
            label: "min 3x3s1",
            width: 16,
            height: 16,
            channels: 16,
            kernel: (3, 3),
            stride: (1, 1),
            pad: [0; 4],
            method: PoolingMethod::Min,
            precision: PoolingPrecision::Fp16,
        },
        PoolCase {
            label: "min 2x2s2",
            width: 32,
            height: 32,
            channels: 16,
            kernel: (2, 2),
            stride: (2, 2),
            pad: [0; 4],
            method: PoolingMethod::Min,
            precision: PoolingPrecision::Int8,
        },
        PoolCase {
            label: "min 3x3s2 pad1",
            width: 32,
            height: 32,
            channels: 16,
            kernel: (3, 3),
            stride: (2, 2),
            pad: [1, 1, 1, 1],
            method: PoolingMethod::Min,
            precision: PoolingPrecision::Fp16,
        },
        PoolCase {
            label: "min 3x3s2 pad1",
            width: 112,
            height: 112,
            channels: 64,
            kernel: (3, 3),
            stride: (2, 2),
            pad: [1, 1, 1, 1],
            method: PoolingMethod::Min,
            precision: PoolingPrecision::Int8,
        },
        PoolCase {
            label: "min 3x3s1 pad-lead",
            width: 16,
            height: 16,
            channels: 8,
            kernel: (3, 3),
            stride: (1, 1),
            pad: [2, 1, 0, 0],
            method: PoolingMethod::Min,
            precision: PoolingPrecision::Fp16,
        },
        PoolCase {
            label: "min 3x3s1 pad-lead",
            width: 16,
            height: 16,
            channels: 16,
            kernel: (3, 3),
            stride: (1, 1),
            pad: [2, 1, 0, 0],
            method: PoolingMethod::Min,
            precision: PoolingPrecision::Int8,
        },
    ]);
}

/// Average pooling, including MobileNetV2's own global average pool -- the
/// shape `PoolingDef` was added to carry.
///
/// 7x7 is inside the direct-kernel limit of 8, so the whole thing is one PPU
/// task. The divisor is 49, which is not a power of two, so this is the case
/// the reciprocal's precision actually shows up in.
#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn average_pooling_matches_oracle() {
    run_cases(&[
        // MobileNetV2's classifier tail, at its real channel count.
        PoolCase {
            label: "avg global 7x7",
            width: 7,
            height: 7,
            channels: 1792,
            kernel: (7, 7),
            stride: (1, 1),
            pad: [0; 4],
            method: PoolingMethod::Avg,
            precision: PoolingPrecision::Fp16,
        },
        // A power-of-two divisor, where the reciprocal is exact.
        PoolCase {
            label: "avg 2x2s2",
            width: 32,
            height: 32,
            channels: 32,
            kernel: (2, 2),
            stride: (2, 2),
            pad: [0; 4],
            method: PoolingMethod::Avg,
            precision: PoolingPrecision::Fp16,
        },
        PoolCase {
            label: "avg 2x2s2",
            width: 32,
            height: 32,
            channels: 16,
            kernel: (2, 2),
            stride: (2, 2),
            pad: [0; 4],
            method: PoolingMethod::Avg,
            precision: PoolingPrecision::Int8,
        },
        // Padded average is admissible: the fill is zero, and the PPU
        // divides by the whole window whether or not a tap was padding, so
        // count-include-pad is what both sides compute.
        PoolCase {
            label: "avg 3x3s1 pad1",
            width: 16,
            height: 16,
            channels: 16,
            kernel: (3, 3),
            stride: (1, 1),
            pad: [1, 1, 1, 1],
            method: PoolingMethod::Avg,
            precision: PoolingPrecision::Fp16,
        },
    ]);
}

/// Widths past the direct-programming limit, where `PoolingPlan` splits the
/// image into several horizontal tiles.
///
/// This is the part a single-tile test cannot reach: each tile carries its
/// own input offset, its own column count and its own kernel overlap, and a
/// wrong offset moves a whole column range rather than corrupting one value.
#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn tiled_pooling_matches_oracle() {
    run_cases(&[
        // The exact fp16 2x2/stride-2 path reaches 256 input columns per
        // task, so 258 needs two.
        PoolCase {
            label: "max 2x2s2 tiled",
            width: 258,
            height: 8,
            channels: 8,
            kernel: (2, 2),
            stride: (2, 2),
            pad: [0; 4],
            method: PoolingMethod::Max,
            precision: PoolingPrecision::Fp16,
        },
        // Everything else is capped at 129 input / 64 output columns, so a
        // 200-wide 3x3 splits several ways.
        PoolCase {
            label: "max 3x3s1 tiled",
            width: 200,
            height: 8,
            channels: 8,
            kernel: (3, 3),
            stride: (1, 1),
            pad: [0; 4],
            method: PoolingMethod::Max,
            precision: PoolingPrecision::Fp16,
        },
        // A tiled average, so the reciprocal is programmed identically in
        // every tile.
        PoolCase {
            label: "avg 2x2s2 tiled",
            width: 260,
            height: 8,
            channels: 8,
            kernel: (2, 2),
            stride: (2, 2),
            pad: [0; 4],
            method: PoolingMethod::Avg,
            precision: PoolingPrecision::Fp16,
        },
    ]);
}

/// ISSUES C16: does the hanging pool -- ResNet50's global `Avg 7x7x2048
/// k7x7` -- hang because of *where* its buffers sit? In the hanging model
/// build its input and output IOVAs were 0x0475a000 / 0x0479a000; in clean
/// builds they were 0x03f7a000, 0x0425e000 and 0x0450a000. This places the
/// same pool's buffers at a ladder of offsets with a ballast BO, from a
/// fresh file, with nothing else run first. `ROCKET_POOL_BALLAST_MB` takes a
/// comma-separated list instead of the default ladder.
/// ISSUES C16: overlapping windows at stride 2 on an even extent leave a
/// trailing input row that no window reads; the PPU's cube height is now
/// programmed to the rows actually read. These are the shapes that hit it --
/// ResNet50's own stem pool among them -- checked against the oracle so the
/// trimmed height cannot silently drop a row that *is* read.
#[test]
#[ignore = "needs /dev/accel/accel0"]
fn overlapping_stride2_pooling_matches_oracle() {
    run_cases(&[
        PoolCase {
            label: "max 3x3s2 even",
            width: 16,
            height: 16,
            channels: 8,
            kernel: (3, 3),
            stride: (2, 2),
            pad: [0; 4],
            method: PoolingMethod::Max,
            precision: PoolingPrecision::Fp16,
        },
        PoolCase {
            label: "max 3x3s2 ResNet50 stem",
            width: 114,
            height: 114,
            channels: 64,
            kernel: (3, 3),
            stride: (2, 2),
            pad: [0; 4],
            method: PoolingMethod::Max,
            precision: PoolingPrecision::Fp16,
        },
        PoolCase {
            label: "avg 3x3s2 even",
            width: 30,
            height: 30,
            channels: 16,
            kernel: (3, 3),
            stride: (2, 2),
            pad: [0; 4],
            method: PoolingMethod::Avg,
            precision: PoolingPrecision::Fp16,
        },
        PoolCase {
            label: "max 3x3s2 even int8",
            width: 32,
            height: 20,
            channels: 16,
            kernel: (3, 3),
            stride: (2, 2),
            pad: [0; 4],
            method: PoolingMethod::Max,
            precision: PoolingPrecision::Int8,
        },
    ]);
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- characterizes the C16 pool hang against IOVA"]
fn pool_iova_probe() {
    let ladder: Vec<usize> = std::env::var("ROCKET_POOL_BALLAST_MB")
        .map(|list| list.split(',').map(|v| v.trim().parse().unwrap()).collect())
        .unwrap_or_else(|_| vec![0, 60, 64, 66, 69, 70, 71, 72, 80, 128]);
    for mb in ladder {
        BALLAST_BYTES.with(|b| b.set(mb << 20));
        println!("ballast {mb} MiB:");
        run_cases(&[PoolCase {
            label: "avg 7x7x2048 k7 (ResNet50 global pool)",
            width: 7,
            height: 7,
            channels: 2048,
            kernel: (7, 7),
            stride: (1, 1),
            pad: [0; 4],
            method: PoolingMethod::Avg,
            precision: PoolingPrecision::Fp16,
        }]);
    }
    BALLAST_BYTES.with(|b| b.set(0));
}

/// What the PPU fills a padded window with, per encoding of the 35-bit
/// `PPU_PADDING_VALUE` (`_1_CFG` holds bits [31:0], `_2_CFG` bits [34:32],
/// which the planner always zeroes and this patches).
///
/// A characterization, not a gate: it asserts nothing about the fill and
/// prints what each candidate does. 6x6 C16, 3x3 stride 1, padding 2 on
/// every side, so each corner window holds exactly one image pixel. The
/// default input is the method's worst value everywhere (the smallest code
/// for max, the largest for min), so whatever a border window returns past
/// it is the fill as the PPU read it; `ROCKET_FILL_INPUT=varied` runs
/// `input_at` instead, whose border windows a mid-range fill would win and
/// which cannot hide a misplaced window.
///
/// A window lying *entirely* in the padding would read the fill directly,
/// but that shape (pad 3) hangs the PPU -- 13 of 13 dispatches killed -- which
/// is why `PoolingShape::validate` refuses pad >= kernel.
///
///     ./pooling_oracle_hw-<hash> pad_fill_probe --ignored --nocapture
fn pad_fill_probe(
    method: PoolingMethod,
    precision: PoolingPrecision,
    worst_input: f32,
    candidates: &[(u32, u32, &str)],
) {
    const SIDE: u32 = 6;
    const PAD: u32 = 2;
    const KERNEL: u32 = 3;
    let channels = precision.channels_per_atom() * 2;
    let out_side = output_extent(SIDE, KERNEL, 1, PAD, PAD);
    let varied = std::env::var("ROCKET_FILL_INPUT").is_ok_and(|value| value == "varied");
    let input = move |y: usize, x: usize, channel: usize| -> f32 {
        if varied {
            input_at(y, x, channel)
        } else {
            worst_input
        }
    };
    let reduce = |a: f32, b: f32| match method {
        PoolingMethod::Max => a.max(b),
        PoolingMethod::Min => a.min(b),
        PoolingMethod::Avg => unreachable!("the probe is for max and min"),
    };
    println!(
        "  {method:?} {precision:?}, input {}",
        if varied {
            "input_at".to_string()
        } else {
            format!("{worst_input} everywhere")
        }
    );
    for &(low, high, meaning) in candidates {
        std::thread::sleep(SETTLE);
        let _guard = NPU_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let shape = PoolingShape {
            input_width: SIDE,
            input_height: SIDE,
            input_channels: channels,
            output_width: out_side,
            output_height: out_side,
            output_channels: channels,
            precision,
            kernel_width: KERNEL,
            kernel_height: KERNEL,
            stride_x: 1,
            stride_y: 1,
            method,
            pad_left: PAD,
            pad_top: PAD,
            pad_right: PAD,
            pad_bottom: PAD,
            pad_value: low,
        };
        shape.validate();
        let patch = |program: &mut Vec<RegCmd>| {
            let mut patched = 0;
            for command in program.iter_mut() {
                if command.0 & 0xFFFF == 0x6048 {
                    let domain = (command.0 >> 48) as u32;
                    *command = RegCmd::new(domain, 0x6048, high);
                    patched += 1;
                }
            }
            assert_eq!(patched, 1, "one PPU_PADDING_VALUE_2_CFG per tile");
        };
        let dispatched = dispatch_pool(shape, &input, &patch);

        let mut mixed = 0usize;
        let mut mixed_clamp = 0usize;
        let mut mixed_wrong = None;
        let mut mixed_values = std::collections::BTreeMap::<String, usize>::new();
        let mut interior_wrong = 0usize;
        for out_y in 0..out_side as usize {
            for out_x in 0..out_side as usize {
                for channel in 0..channels as usize {
                    let got = read_output(shape, &dispatched.raw, out_y, out_x, channel);
                    let mut best: Option<f32> = None;
                    let mut padded_tap = false;
                    for ky in 0..KERNEL as isize {
                        for kx in 0..KERNEL as isize {
                            let y = out_y as isize - PAD as isize + ky;
                            let x = out_x as isize - PAD as isize + kx;
                            if (0..SIDE as isize).contains(&y) && (0..SIDE as isize).contains(&x) {
                                let value = input(y as usize, x as usize, channel);
                                best = Some(best.map_or(value, |best| reduce(best, value)));
                            } else {
                                padded_tap = true;
                            }
                        }
                    }
                    let clamp = best.expect("pad < kernel: every window holds a pixel");
                    if padded_tap {
                        mixed += 1;
                        *mixed_values.entry(format!("{got}")).or_default() += 1;
                        if got == clamp {
                            mixed_clamp += 1;
                        } else {
                            mixed_wrong.get_or_insert((out_y, out_x, channel, clamp, got));
                        }
                    } else if got != clamp {
                        interior_wrong += 1;
                    }
                }
            }
        }
        println!(
            "    fill {low:#010x}/{high} ({meaning}): border windows {mixed_clamp}/{mixed} \
             equal clamp-to-image, values {mixed_values:?}{}; interior wrong {interior_wrong}{}",
            mixed_wrong
                .map(|(y, x, c, want, got)| format!(
                    ", first other at ({y},{x},c{c}) clamp {want} got {got}"
                ))
                .unwrap_or_default(),
            if dispatch::is_device_timeout(dispatched.elapsed) {
                format!(
                    " -- DISPATCH {:.0} ms, watchdog kill",
                    dispatched.elapsed.as_secs_f64() * 1e3
                )
            } else {
                String::new()
            },
        );
    }
}

/// Measured 2026-11-24: every candidate gives the clamp-to-image answer
/// under both inputs -- the int8 max ignores the fill
/// (`PoolingMethod::pad_fill_value`).
#[test]
#[ignore = "needs /dev/accel/accel0 -- characterization, prints rather than asserts"]
fn int8_max_pad_fill_probe() {
    pad_fill_probe(
        PoolingMethod::Max,
        PoolingPrecision::Int8,
        -128.0,
        &[
            (0x0000_0000, 0, "0"),
            (0x0000_0005, 0, "+5"),
            (0x0000_007F, 0, "+127 / 0x7F"),
            (0x0000_0080, 0, "0x80: -128 as 8 bits, +128 wider"),
            (0x0000_00FB, 0, "0xFB: -5 as 8 bits"),
            (0x0000_FFFB, 0, "0xFFFB: -5 as 16 bits"),
            (0xFFFF_FFFB, 0, "0xFFFFFFFB: -5 as 32 bits"),
            (0xFFFF_FFFB, 7, "-5 as 35 bits"),
            (0xFFFF_FF80, 7, "-128 as 35 bits"),
            (0x0000_FC00, 0, "0xFC00: fp16 -inf"),
            (0x0000_8000, 0, "0x8000: -32768 as 16 bits"),
            (0x8000_0000, 0, "0x80000000"),
            (0x0000_0000, 4, "-2^34: 35-bit minimum"),
        ],
    );
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- characterization, prints rather than asserts"]
fn int8_min_pad_fill_probe() {
    pad_fill_probe(
        PoolingMethod::Min,
        PoolingPrecision::Int8,
        127.0,
        &[
            (0x0000_0000, 0, "0"),
            (0x0000_0005, 0, "+5"),
            (0x0000_007F, 0, "+127 / 0x7F"),
            (0x0000_0080, 0, "0x80: -128 as 8 bits, +128 wider"),
            (0x0000_00FB, 0, "0xFB: -5 as 8 bits, +251 wider"),
            (0x0000_FFFB, 0, "0xFFFB: -5 as 16 bits"),
            (0xFFFF_FFFB, 7, "-5 as 35 bits"),
            (0x0000_7C00, 0, "0x7C00: fp16 +inf"),
            (0x7FFF_FFFF, 0, "0x7FFFFFFF"),
            (0xFFFF_FFFF, 3, "2^34 - 1: 35-bit maximum"),
        ],
    );
}

/// The control: fp16 max, where the vendor's own captures program -inf. If
/// the PPU reads the fill anywhere, a 0 or +1.0 fill shows here.
#[test]
#[ignore = "needs /dev/accel/accel0 -- characterization, prints rather than asserts"]
fn fp16_max_pad_fill_probe() {
    pad_fill_probe(
        PoolingMethod::Max,
        PoolingPrecision::Fp16,
        -8.0,
        &[
            (0x0000_FC00, 0, "-inf (the vendor's)"),
            (0x0000_0000, 0, "0.0"),
            (0x0000_3C00, 0, "+1.0"),
            (0x0000_BC00, 0, "-1.0"),
            (0x0000_7C00, 0, "+inf"),
            (0x0000_FBFF, 0, "-65504"),
        ],
    );
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- characterization, prints rather than asserts"]
fn fp16_min_pad_fill_probe() {
    pad_fill_probe(
        PoolingMethod::Min,
        PoolingPrecision::Fp16,
        8.0,
        &[
            (0x0000_7C00, 0, "+inf"),
            (0x0000_0000, 0, "0.0"),
            (0x0000_3C00, 0, "+1.0"),
            (0x0000_BC00, 0, "-1.0"),
            (0x0000_FC00, 0, "-inf"),
            (0x0000_7BFF, 0, "+65504"),
        ],
    );
}

/// The control the max/min probes need: an *average* pool has to add the
/// fill into its sum (it divides by the whole window), so if
/// `PPU_PADDING_VALUE` reaches the hardware at all, a nonzero fill moves an
/// average's border outputs. Input 0 everywhere, so a border window reads
/// `padded_taps * fill / 9` and nothing else.
#[test]
#[ignore = "needs /dev/accel/accel0 -- characterization, prints rather than asserts"]
fn avg_pad_fill_control_probe() {
    const SIDE: u32 = 6;
    const PAD: u32 = 2;
    const KERNEL: u32 = 3;
    let out_side = output_extent(SIDE, KERNEL, 1, PAD, PAD);
    for (precision, candidates) in [
        (
            PoolingPrecision::Fp16,
            &[
                (0x0000u32, "0.0"),
                (0x3C00, "+1.0"),
                (0x4880, "+9.0"),
                (0xC880, "-9.0"),
            ][..],
        ),
        (
            PoolingPrecision::Int8,
            &[
                (0x00, "0"),
                (0x09, "+9"),
                (0x12, "+18"),
                (0xF7, "0xF7: -9 as 8 bits"),
            ][..],
        ),
    ] {
        let channels = precision.channels_per_atom();
        for &(fill, meaning) in candidates {
            std::thread::sleep(SETTLE);
            let _guard = NPU_TEST_LOCK
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let shape = PoolingShape {
                input_width: SIDE,
                input_height: SIDE,
                input_channels: channels,
                output_width: out_side,
                output_height: out_side,
                output_channels: channels,
                precision,
                kernel_width: KERNEL,
                kernel_height: KERNEL,
                stride_x: 1,
                stride_y: 1,
                method: PoolingMethod::Avg,
                pad_left: PAD,
                pad_top: PAD,
                pad_right: PAD,
                pad_bottom: PAD,
                pad_value: fill,
            };
            shape.validate();
            let dispatched = dispatch_pool(shape, &|_, _, _| 0.0, &|_| {});
            // padded taps -> the outputs seen for that count
            let mut by_taps =
                std::collections::BTreeMap::<usize, std::collections::BTreeSet<String>>::new();
            for out_y in 0..out_side as usize {
                for out_x in 0..out_side as usize {
                    let mut padded = 0;
                    for ky in 0..KERNEL as isize {
                        for kx in 0..KERNEL as isize {
                            let y = out_y as isize - PAD as isize + ky;
                            let x = out_x as isize - PAD as isize + kx;
                            if !((0..SIDE as isize).contains(&y) && (0..SIDE as isize).contains(&x))
                            {
                                padded += 1;
                            }
                        }
                    }
                    let got = read_output(shape, &dispatched.raw, out_y, out_x, 0);
                    by_taps.entry(padded).or_default().insert(format!("{got}"));
                }
            }
            println!(
                "  avg {precision:?} fill {fill:#06x} ({meaning}): padded taps -> outputs {by_taps:?}{}",
                if dispatch::is_device_timeout(dispatched.elapsed) {
                    " -- watchdog kill"
                } else {
                    ""
                }
            );
        }
    }
}
