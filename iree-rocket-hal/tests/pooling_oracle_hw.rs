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
            pad_value: self
                .method
                .required_pad_fill(self.precision, self.pad.iter().any(|pad| *pad != 0))
                .expect("this ladder only pads where the fill value is measured"),
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

fn run_case(case: PoolCase) {
    let _guard = NPU_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let shape = case.shape();
    shape.validate();

    let channels_per_atom = case.precision.channels_per_atom() as usize;
    let element_bytes = case.precision.element_bytes() as usize;
    let surfaces = shape.programmed_channels() as usize / channels_per_atom;
    let in_pixels = (case.width * case.height) as usize;
    let out_pixels = (shape.output_width * shape.output_height) as usize;
    // Both cube strides are rounded up to four pixels by
    // `build_pooling_tile_task`; a buffer laid out any other way puts every
    // surface past the first at the wrong offset.
    let in_surface_pixels = in_pixels.next_multiple_of(4);
    let out_surface_pixels = out_pixels.next_multiple_of(4);

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

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let input = Buffer::new(
            fd,
            page_aligned_size(surfaces * in_surface_pixels * FEATURE_ATOM_BYTES),
            &file,
        );
        ptr::write_bytes(input.host_ptr, 0, input.size);
        for channel in 0..case.channels as usize {
            for y in 0..case.height as usize {
                for x in 0..case.width as usize {
                    let offset =
                        (channel / channels_per_atom) * in_surface_pixels * FEATURE_ATOM_BYTES
                            + (y * case.width as usize + x) * FEATURE_ATOM_BYTES
                            + (channel % channels_per_atom) * element_bytes;
                    let value = input_at(y, x, channel);
                    match case.precision {
                        PoolingPrecision::Fp16 => {
                            ptr::write(input.host_ptr.add(offset) as *mut u16, f32_to_f16(value))
                        }
                        PoolingPrecision::Int8 => {
                            ptr::write(input.host_ptr.add(offset), value as i8 as u8)
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
        let programs = plan.programs_with_buffers(&PoolingBuffers {
            input_addr: input.dma_address,
            output_addr: output.dma_address,
        });
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

        for handle in [input.handle, output.handle] {
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
        in_handles.push(input.handle);
        let out_handles = [output.handle];
        let jobs = [JobDesc {
            tasks: &tasks,
            in_handles: &in_handles,
            out_handles: &out_handles,
        }];

        let started = Instant::now();
        submit_jobs(fd, &jobs).unwrap_or_else(|error| {
            panic!("{label} ({} tile(s)): SUBMIT failed: {error}", tasks.len())
        });
        prep_bo(fd, output.handle, COMPLETION_TIMEOUT_NS)
            .unwrap_or_else(|error| panic!("{label}: did not complete: {error}"));
        let elapsed = started.elapsed();

        let raw = std::slice::from_raw_parts(output.host_ptr, output.size);
        let read = |out_y: usize, out_x: usize, channel: usize| -> f32 {
            let offset = (channel / channels_per_atom) * out_surface_pixels * FEATURE_ATOM_BYTES
                + (out_y * shape.output_width as usize + out_x) * FEATURE_ATOM_BYTES
                + (channel % channels_per_atom) * element_bytes;
            match case.precision {
                PoolingPrecision::Fp16 => {
                    f16_to_f32(u16::from_le_bytes([raw[offset], raw[offset + 1]]))
                }
                PoolingPrecision::Int8 => f32::from(raw[offset] as i8),
            }
        };

        let tolerance = case.tolerance();
        let mut mismatches = 0usize;
        let mut first = None;
        let mut worst = 0.0f32;
        for out_y in 0..shape.output_height as usize {
            for out_x in 0..shape.output_width as usize {
                for channel in 0..case.channels as usize {
                    let want = expected(case, out_y, out_x, channel);
                    let got = read(out_y, out_x, channel);
                    let difference = (got - want).abs();
                    worst = worst.max(difference);
                    if !got.is_finite() || difference > tolerance {
                        mismatches += 1;
                        first.get_or_insert((out_y, out_x, channel, want, got));
                    }
                }
            }
        }

        for (buffer, _) in &command_buffers {
            close_bo(fd, buffer.handle).unwrap();
        }
        close_bo(fd, input.handle).unwrap();
        close_bo(fd, output.handle).unwrap();

        let total = out_pixels * case.channels as usize;
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
            "  {label}: ok, {} tile(s), max|diff| {worst}{}",
            tasks.len(),
            dispatch::note(elapsed),
        );
    }
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

/// Min pooling, unpadded only: no measurement says what the PPU should fill
/// a padded min window with, and the runtime refuses that combination
/// rather than guessing (`PoolingMethod::pad_fill_value`).
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
