//! Hardware check for the affine int8 coefficient encoding emitted by RKNN.
//!
//! This deliberately does not submit a vendor command stream. `ConvPlan`
//! and the Rocket builders generate every register command; only the 32-byte
//! coefficient payload and 256-byte BS payload are captured fixtures. The
//! five cases cover symmetric, asymmetric, and one-sided weight ranges. The
//! captured BS multiplier is `2^14`, so the output conversion uses the
//! ordinary centered-coefficient scale; it does not apply the separate
//! divide-by-128 compensation used with Rocket's legacy `+128` BS constant.
//!
//! Cross-compile and run on the board:
//!
//! ```text
//! CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!   cargo test --target aarch64-unknown-linux-gnu --release \
//!     -p iree-rocket-hal --test conv_int8_vendor_affine_hw --no-run
//!
//! ./conv_int8_vendor_affine_hw-<hash> --ignored --nocapture
//! ```

use std::{collections::BTreeMap, fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    conv::{Buffers, ConvPlan, Multiplier, Precision, Quantization, Shape},
    device::{Buffer, JobDesc, close_bo, fini_bo, prep_bo, submit_jobs, unmap_bo},
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const PAGE_BYTES: usize = 4096;
const TIMEOUT_NS: u64 = 5_000_000_000;
const WIDTH: usize = 4;
const HEIGHT: usize = 4;
const INPUT_CHANNELS: usize = 2;
const OUTPUT_CHANNELS: usize = 2;
const OUTPUT_ATOM_BYTES: usize = 16;
const REPEATS: usize = 3;

const M0_COEFFICIENTS: &[u8; 32] =
    include_bytes!("fixtures/controlled_int8_vendor/weight-c2_m0.coefficients.bin");
const M0_BS: &[u8; 256] = include_bytes!("fixtures/controlled_int8_vendor/weight-c2_m0.bs.bin");
const H0_COEFFICIENTS: &[u8; 32] =
    include_bytes!("fixtures/controlled_int8_vendor/weight-c2_h0.coefficients.bin");
const H0_BS: &[u8; 256] = include_bytes!("fixtures/controlled_int8_vendor/weight-c2_h0.bs.bin");
const H1_COEFFICIENTS: &[u8; 32] =
    include_bytes!("fixtures/controlled_int8_vendor/weight-c2_h1.coefficients.bin");
const H1_BS: &[u8; 256] = include_bytes!("fixtures/controlled_int8_vendor/weight-c2_h1.bs.bin");
const Z0_COEFFICIENTS: &[u8; 32] =
    include_bytes!("fixtures/controlled_int8_vendor/weight-c2_z0.coefficients.bin");
const Z0_BS: &[u8; 256] = include_bytes!("fixtures/controlled_int8_vendor/weight-c2_z0.bs.bin");
const Z1_COEFFICIENTS: &[u8; 32] =
    include_bytes!("fixtures/controlled_int8_vendor/weight-c2_z1.coefficients.bin");
const Z1_BS: &[u8; 256] = include_bytes!("fixtures/controlled_int8_vendor/weight-c2_z1.bs.bin");

#[derive(Clone, Copy)]
struct VendorCase {
    name: &'static str,
    coefficients: &'static [u8; 32],
    bs: &'static [u8; 256],
    /// Independently recorded `i8(coefficient) + BS constant` values in
    /// `[output_channel][input_channel]` order.
    centered: [[i32; INPUT_CHANNELS]; OUTPUT_CHANNELS],
    /// The centered coefficients are exact multiples of this value. The
    /// output conversion divides by it, keeping every expected result exact.
    divisor: i32,
}

fn vendor_cases() -> [VendorCase; 5] {
    [
        VendorCase {
            name: "c2_m0 symmetric [-1,+1] / [+1,-1]",
            coefficients: M0_COEFFICIENTS,
            bs: M0_BS,
            centered: [[-127, 127], [127, -127]],
            divisor: 127,
        },
        VendorCase {
            name: "c2_h0 asymmetric [-1,+0.5] / [+1,-0.5]",
            coefficients: H0_COEFFICIENTS,
            bs: H0_BS,
            centered: [[-170, 85], [170, -85]],
            divisor: 85,
        },
        VendorCase {
            name: "c2_h1 asymmetric sign swap",
            coefficients: H1_COEFFICIENTS,
            bs: H1_BS,
            centered: [[170, -85], [-170, 85]],
            divisor: 85,
        },
        VendorCase {
            name: "c2_z0 one-sided first input lane",
            coefficients: Z0_COEFFICIENTS,
            bs: Z0_BS,
            centered: [[-255, 0], [255, 0]],
            divisor: 255,
        },
        VendorCase {
            name: "c2_z1 one-sided second input lane",
            coefficients: Z1_COEFFICIENTS,
            bs: Z1_BS,
            centered: [[0, -255], [0, 255]],
            divisor: 255,
        },
    ]
}

fn bs_i16(bs: &[u8], offset: usize) -> i16 {
    i16::from_le_bytes([bs[offset], bs[offset + 1]])
}

fn bs_i32(bs: &[u8], offset: usize) -> i32 {
    i32::from_le_bytes([bs[offset], bs[offset + 1], bs[offset + 2], bs[offset + 3]])
}

fn verify_payload(case: VendorCase) {
    for output_channel in 0..OUTPUT_CHANNELS {
        assert_eq!(bs_i32(case.bs, output_channel * 4), 0, "{} bias", case.name);
        let constant = i32::from(bs_i16(case.bs, 32 + output_channel * 2));
        assert_eq!(
            bs_i16(case.bs, 48 + output_channel * 2),
            16_384,
            "{} multiplier",
            case.name
        );

        let row = &case.coefficients[output_channel * 16..(output_channel + 1) * 16];
        for input_channel in 0..INPUT_CHANNELS {
            let centered = i32::from(row[input_channel] as i8) + constant;
            assert_eq!(
                centered, case.centered[output_channel][input_channel],
                "{} output {output_channel} input {input_channel}",
                case.name
            );
        }
        for (input_channel, raw) in row.iter().enumerate().skip(INPUT_CHANNELS) {
            assert_eq!(
                i32::from(*raw as i8) + constant,
                0,
                "{} output {output_channel} padding lane {input_channel}",
                case.name
            );
        }
    }
}

#[test]
fn controlled_vendor_payloads_decode_to_expected_centered_coefficients() {
    for case in vendor_cases() {
        verify_payload(case);
    }
}

fn page_aligned_size(size: usize) -> usize {
    size.max(1).div_ceil(PAGE_BYTES) * PAGE_BYTES
}

struct OwnedBuffer {
    fd: i32,
    buffer: Buffer,
}

impl OwnedBuffer {
    unsafe fn new(fd: i32, size: usize, file: &std::fs::File) -> Self {
        Self {
            fd,
            buffer: unsafe { Buffer::new(fd, page_aligned_size(size), file) },
        }
    }

    unsafe fn from_bytes(fd: i32, bytes: &[u8], file: &std::fs::File) -> Self {
        let buffer = unsafe { Self::new(fd, bytes.len(), file) };
        unsafe {
            ptr::write_bytes(buffer.buffer.host_ptr, 0, buffer.buffer.size);
            ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.buffer.host_ptr, bytes.len());
        }
        buffer
    }
}

impl Drop for OwnedBuffer {
    fn drop(&mut self) {
        unsafe {
            let _ = unmap_bo(&self.buffer);
            let _ = close_bo(self.fd, self.buffer.handle);
        }
    }
}

const INPUT_VECTORS: [[i8; INPUT_CHANNELS]; WIDTH * HEIGHT] = [
    [1, 0],
    [0, 1],
    [1, 1],
    [-1, 0],
    [0, -1],
    [-1, -1],
    [2, 1],
    [-2, -1],
    [1, 2],
    [-1, -2],
    [3, -1],
    [-3, 1],
    [2, -2],
    [-2, 2],
    [3, 3],
    [-3, -3],
];

fn input_bytes() -> Vec<u8> {
    INPUT_VECTORS
        .iter()
        .flat_map(|vector| vector.iter().map(|value| *value as u8))
        .collect()
}

fn shape_for(case: VendorCase) -> Shape {
    Shape::with_precision(
        WIDTH as u32,
        HEIGHT as u32,
        1,
        INPUT_CHANNELS as u32,
        OUTPUT_CHANNELS as u32,
        Precision::Int8(Quantization {
            input_zero_point: 0,
            output_zero_point: 0,
            multiplier: Multiplier::from_ratio(1.0 / f64::from(case.divisor)),
        }),
    )
    .with_padding([0, 0])
}

fn expected(case: VendorCase, pixel: usize, output_channel: usize) -> i8 {
    let input = INPUT_VECTORS[pixel];
    let accumulator = (0..INPUT_CHANNELS)
        .map(|input_channel| {
            i32::from(input[input_channel]) * case.centered[output_channel][input_channel]
        })
        .sum::<i32>();
    assert_eq!(
        accumulator % case.divisor,
        0,
        "{} expected result is not integral",
        case.name
    );
    i8::try_from(accumulator / case.divisor).expect("test vectors must stay in int8")
}

unsafe fn execute_once(
    file: &std::fs::File,
    case: VendorCase,
) -> Result<(ConvPlan, Vec<u8>), String> {
    let fd = file.as_raw_fd();
    let shape = shape_for(case);
    let kernels = [1, 1];
    let plan = ConvPlan::new(shape, kernels);
    let output_len = shape.output_scratch_bytes(kernels);

    let input = unsafe { OwnedBuffer::from_bytes(fd, &input_bytes(), file) };
    let weights = unsafe { OwnedBuffer::from_bytes(fd, case.coefficients, file) };
    let bs = unsafe { OwnedBuffer::from_bytes(fd, case.bs, file) };
    let output = unsafe { OwnedBuffer::new(fd, output_len, file) };
    unsafe { ptr::write_bytes(output.buffer.host_ptr, 0xa5, output.buffer.size) };

    let programs = plan.programs_with_buffers(Buffers {
        input: input.buffer.dma_address,
        weights: weights.buffer.dma_address,
        bias: bs.buffer.dma_address,
        output: output.buffer.dma_address,
    });
    let mut command_buffers = Vec::with_capacity(programs.len());
    for program in &programs {
        let command_bytes = program.len() * mem::size_of::<u64>();
        let buffer = unsafe { OwnedBuffer::new(fd, command_bytes, file) };
        unsafe {
            ptr::write_bytes(buffer.buffer.host_ptr, 0, buffer.buffer.size);
            let words =
                std::slice::from_raw_parts_mut(buffer.buffer.host_ptr as *mut u64, program.len());
            for (destination, command) in words.iter_mut().zip(program) {
                *destination = command.0;
            }
        }
        command_buffers.push((buffer, program.len() as u32));
    }

    for handle in [
        input.buffer.handle,
        weights.buffer.handle,
        bs.buffer.handle,
        output.buffer.handle,
    ] {
        unsafe { fini_bo(fd, handle) }.map_err(|error| format!("sync data BO: {error}"))?;
    }
    for (buffer, _) in &command_buffers {
        unsafe { fini_bo(fd, buffer.buffer.handle) }
            .map_err(|error| format!("sync regcmd BO: {error}"))?;
    }

    let tasks = command_buffers
        .iter()
        .map(|(buffer, count)| [(buffer.buffer.dma_address, *count)])
        .collect::<Vec<_>>();
    let input_handles = command_buffers
        .iter()
        .map(|(buffer, _)| {
            [
                buffer.buffer.handle,
                input.buffer.handle,
                weights.buffer.handle,
                bs.buffer.handle,
            ]
        })
        .collect::<Vec<_>>();
    let output_handles = [output.buffer.handle];
    let jobs = tasks
        .iter()
        .zip(&input_handles)
        .map(|(tasks, input_handles)| JobDesc {
            tasks,
            in_handles: input_handles,
            out_handles: &output_handles,
        })
        .collect::<Vec<_>>();

    unsafe { submit_jobs(fd, &jobs) }.map_err(|error| format!("submit: {error}"))?;
    unsafe { prep_bo(fd, output.buffer.handle, TIMEOUT_NS) }
        .map_err(|error| format!("completion wait: {error}"))?;

    let bytes = unsafe { std::slice::from_raw_parts(output.buffer.host_ptr, output_len).to_vec() };
    Ok((plan, bytes))
}

fn compare(case: VendorCase, output: &[u8]) -> Result<BTreeMap<i32, usize>, String> {
    let mut differences = BTreeMap::new();
    let mut mismatches = 0usize;
    let mut samples = Vec::new();
    for pixel in 0..WIDTH * HEIGHT {
        for output_channel in 0..OUTPUT_CHANNELS {
            let want = expected(case, pixel, output_channel);
            let got = output[pixel * OUTPUT_ATOM_BYTES + output_channel] as i8;
            let difference = i32::from(got) - i32::from(want);
            *differences.entry(difference).or_insert(0) += 1;
            if difference.abs() > 1 {
                mismatches += 1;
                if samples.len() < 12 {
                    let y = pixel / WIDTH;
                    let x = pixel % WIDTH;
                    samples.push(format!(
                        "[y={y}, x={x}, c={output_channel}] input={:?} want {want} got {got}",
                        INPUT_VECTORS[pixel]
                    ));
                }
            }
        }
    }
    if mismatches == 0 {
        Ok(differences)
    } else {
        Err(format!(
            "{mismatches} mismatches outside one LSB, differences={differences:?}\n      {}",
            samples.join("\n      ")
        ))
    }
}

#[test]
#[ignore = "needs an RK3588 NPU at /dev/accel/accel0"]
fn builder_commands_execute_controlled_vendor_affine_payloads() {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("open RK3588 NPU");
    let cases = vendor_cases();
    let mut failures = Vec::new();

    println!("\n=== builder commands + controlled RKNN int8 coefficient/BS payloads ===");
    println!("  BS multiplier 0x4000 is unit; OUT_CVT scales centered coefficients directly");
    for (case_index, case) in cases.into_iter().enumerate() {
        verify_payload(case);
        for repeat in 0..REPEATS {
            let result = unsafe { execute_once(&file, case) };
            match result.and_then(|(plan, output)| {
                let differences = compare(case, &output)?;
                Ok((plan, differences))
            }) {
                Ok((plan, differences)) => println!(
                    "[{}/{} rep {repeat}] ok   {} differences={differences:?} banks={}/{} tiles={}",
                    case_index + 1,
                    cases.len(),
                    case.name,
                    plan.data_banks(),
                    plan.weight_banks(),
                    plan.tiles().len(),
                ),
                Err(error) => {
                    println!(
                        "[{}/{} rep {repeat}] FAIL {}: {error}",
                        case_index + 1,
                        cases.len(),
                        case.name
                    );
                    failures.push(format!("{} rep {repeat}: {error}", case.name));
                }
            }
        }
    }

    println!("\n=== summary ===");
    println!("  runs:   {}", cases.len() * REPEATS);
    println!("  passed: {}", cases.len() * REPEATS - failures.len());
    println!("  failed: {}", failures.len());
    assert!(
        failures.is_empty(),
        "{} controlled vendor payload run(s) failed:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}
