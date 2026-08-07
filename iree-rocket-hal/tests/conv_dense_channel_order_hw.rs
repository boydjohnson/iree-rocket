//! Isolated hardware test: does the Cin<=4 dense/ARGB path preserve input
//! channel order once the data on each channel actually differs?
//!
//! Every existing hand-rolled hw test that has ever exercised a dense-layout
//! shape (`features.0`'s Cin=3 included, validated 5/5 by the original
//! five-shape sweep) fills every real channel with a uniform 1.0 and checks
//! a single expected sum. That is structurally blind to a channel-order
//! swap: swapping two identically-valued, identically-weighted input
//! channels changes nothing about the output. `rocket_conv_harness.py` (a
//! separate real-compiler-path harness in iree-rocket-design-spike) found a
//! real, deterministic, non-zero mismatch against the CPU reference for
//! `features.0`'s exact shape once fed genuinely varying random data -- the
//! first Cin<=4 dense-layout shape that harness has tested, and the first
//! time any test here has used non-uniform per-channel data on the dense
//! path at all.
//!
//! This isolates the channel-order hypothesis specifically, independent of
//! spatial-tap order: a 1x1 kernel has exactly one tap, so there is no
//! spatial ordering left to get wrong, only channel ordering. Cin=Cout=3,
//! weight is the identity matrix (`weight[cin][cout] = 1 iff cin == cout`,
//! built as a logical HWCF buffer and packed with
//! `tensor_layout::pack_hwcf_to_rocket_weights`, the same packer
//! `rocket-hal-driver`'s real dispatch path uses via `WeightPacking`), and
//! each input channel is filled with a distinguishable constant (channel 0
//! = 1.0, channel 1 = 2.0, channel 2 = 3.0, uniform across every pixel). If
//! channel order round-trips correctly end to end, every pixel's output
//! channel `c` must read back exactly the same constant channel `c` was
//! given -- a pure per-channel passthrough. Any permutation (a swap, an
//! off-by-one rotation) shows up immediately as output channel `c` reading
//! back a *different* channel's constant instead of 0/mismatched-with-want,
//! which is the point: it is diagnostic of the exact permutation, not just
//! pass/fail.
//!
//!   CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!     cargo test --target aarch64-unknown-linux-gnu --release \
//!       --test conv_dense_channel_order_hw --no-run
//!
//!   ./conv_dense_channel_order_hw-<hash> --ignored --nocapture

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    conv::{Buffers, ConvPlan, Kernels, Shape},
    device::{Buffer, JobDesc, close_bo, fini_bo, prep_bo, submit_jobs},
    tensor_layout::{pack_hwcf_to_rocket_weights, rocket_weight_storage_size},
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const FP16_BYTES: usize = 2;
const FEATURE_ATOM_BYTES: usize = 16;
const CHANNELS_PER_ATOM: usize = 8;
const PAGE_BYTES: usize = 4096;

// fp16 1.0, 2.0, 3.0 -- distinguishable per-channel constants. Index i is
// channel i's fill value and also its expected passthrough output.
const CHANNEL_VALUES: [u16; 3] = [0x3c00, 0x4000, 0x4200];

const CIN: usize = 3;
const COUT: usize = 3;
const REPS: usize = 3;

fn page_aligned_size(size: usize) -> usize {
    size.div_ceil(PAGE_BYTES) * PAGE_BYTES
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    let f32_bits = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            let subnormal = (frac as f32) * 2f32.powi(-24);
            return if sign == 1 { -subnormal } else { subnormal };
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (frac << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13)
    };
    f32::from_bits(f32_bits)
}

struct Failure {
    mismatches: usize,
    samples: Vec<String>,
    timed_out: bool,
}

/// Built from ConvPlan directly, the same call
/// `rocket-hal-driver/src/command_buffer.rs`'s real dispatch path makes
/// (`ConvPlan::new(*shape, kernels).programs_with_buffers(bufs)`) once past
/// its input-packing step -- which for Cin<=4 dense shapes is a no-op
/// (`input_packing = None`, the raw IREE buffer address is used directly),
/// so this exercises the identical register-program-building code path a
/// real compiled dispatch would.
fn run(fd: i32, file: &std::fs::File, width: u32, height: u32) -> Result<(), Failure> {
    let kernels: Kernels = [1, 1];
    let shape = Shape::with_out_channels(width, height, 1, CIN as u32, COUT as u32);
    debug_assert!(matches!(
        shape.layout(),
        iree_rocket_hal::rocket::conv::FeatureLayout::Dense
    ));

    let plan = ConvPlan::new(shape, kernels);
    let pixel_count = width as usize * height as usize;
    let input_bytes = pixel_count * CIN * FP16_BYTES;
    let output_bytes = shape.output_scratch_bytes(kernels);
    assert_eq!(
        output_bytes,
        pixel_count * FEATURE_ATOM_BYTES * 2,
        "fp16 Cout=3 must allocate both padded output surfaces"
    );

    unsafe {
        let buf_input = Buffer::new(fd, page_aligned_size(input_bytes), file);
        let input_words =
            std::slice::from_raw_parts_mut(buf_input.host_ptr as *mut u16, input_bytes / 2);
        for pixel in 0..pixel_count {
            for channel in 0..CIN {
                input_words[pixel * CIN + channel] = CHANNEL_VALUES[channel];
            }
        }

        // Logical HWCF identity filter: weight[0][0][cin][cout] = 1.0 iff
        // cin == cout, else 0.0. HWCF flattens as ((h*W+w)*Cin+cin)*Cout+cout
        // -- see pack_hwcf_to_rocket_weights's own src_element computation.
        let mut dense_weights = vec![0u8; CIN * COUT * FP16_BYTES];
        for cin in 0..CIN {
            for cout in 0..COUT {
                let value: u16 = if cin == cout { 0x3c00 } else { 0 };
                let offset = (cin * COUT + cout) * FP16_BYTES;
                dense_weights[offset..offset + FP16_BYTES].copy_from_slice(&value.to_le_bytes());
            }
        }
        let packed_len = rocket_weight_storage_size(1, 1, CIN, COUT, FP16_BYTES)
            .expect("identity filter shape should be valid");
        let buf_weights = Buffer::new(fd, page_aligned_size(packed_len), file);
        let packed_slice = std::slice::from_raw_parts_mut(buf_weights.host_ptr, buf_weights.size);
        pack_hwcf_to_rocket_weights(&dense_weights, 1, 1, CIN, COUT, FP16_BYTES, packed_slice)
            .expect("packing the identity filter should not fail");

        let buf_bias = Buffer::new(fd, PAGE_BYTES, file);
        ptr::write_bytes(buf_bias.host_ptr, 0, buf_bias.size);
        let buf_output = Buffer::new(fd, page_aligned_size(output_bytes), file);
        ptr::write_bytes(buf_output.host_ptr, 0, buf_output.size);

        let programs = plan.programs_with_buffers(Buffers {
            input: buf_input.dma_address,
            weights: buf_weights.dma_address,
            bias: buf_bias.dma_address,
            output: buf_output.dma_address,
        });
        let mut command_buffers = Vec::with_capacity(programs.len());
        for commands in &programs {
            let command_bytes = commands.len() * mem::size_of::<u64>();
            let buffer = Buffer::new(fd, page_aligned_size(command_bytes), file);
            ptr::write_bytes(buffer.host_ptr, 0, buffer.size);
            let words = std::slice::from_raw_parts_mut(buffer.host_ptr as *mut u64, commands.len());
            for (destination, command) in words.iter_mut().zip(commands.iter()) {
                *destination = command.0;
            }
            command_buffers.push((buffer, commands.len() as u32));
        }

        for handle in [
            buf_input.handle,
            buf_weights.handle,
            buf_bias.handle,
            buf_output.handle,
        ] {
            fini_bo(fd, handle).expect("failed to sync BO for the NPU");
        }
        for (buffer, _) in &command_buffers {
            fini_bo(fd, buffer.handle).expect("failed to sync regcmd BO");
        }

        let tasks: Vec<[(u32, u32); 1]> = command_buffers
            .iter()
            .map(|(buffer, count)| [(buffer.dma_address, *count)])
            .collect();
        let in_handles: Vec<[u32; 4]> = command_buffers
            .iter()
            .map(|(buffer, _)| {
                [
                    buffer.handle,
                    buf_input.handle,
                    buf_weights.handle,
                    buf_bias.handle,
                ]
            })
            .collect();
        let out_handles = [buf_output.handle];
        let jobs: Vec<JobDesc<'_>> = tasks
            .iter()
            .zip(&in_handles)
            .map(|(tasks, in_handles)| JobDesc {
                tasks,
                in_handles,
                out_handles: &out_handles,
            })
            .collect();

        submit_jobs(fd, &jobs)
            .unwrap_or_else(|error| panic!("{width}x{height} SUBMIT failed: {error}"));

        let mut failure = Failure {
            mismatches: 0,
            samples: Vec::new(),
            timed_out: false,
        };

        if let Err(error) = prep_bo(fd, buf_output.handle, 5_000_000_000) {
            failure.timed_out = true;
            failure
                .samples
                .push(format!("prep_bo did not complete: {error}"));
        } else {
            let raw = std::slice::from_raw_parts(buf_output.host_ptr, output_bytes);
            for y in 0..height as usize {
                for x in 0..width as usize {
                    for channel in 0..COUT {
                        let surface = channel / CHANNELS_PER_ATOM;
                        let lane = channel % CHANNELS_PER_ATOM;
                        let offset =
                            surface * width as usize * height as usize * FEATURE_ATOM_BYTES
                                + (y * width as usize + x) * FEATURE_ATOM_BYTES
                                + lane * FP16_BYTES;
                        let got = f16_to_f32(u16::from_le_bytes([raw[offset], raw[offset + 1]]));
                        let want = f16_to_f32(CHANNEL_VALUES[channel]);
                        if got != want {
                            failure.mismatches += 1;
                            if failure.samples.len() < 8 {
                                failure.samples.push(format!(
                                    "[y={y}, x={x}, out_channel={channel}] want {want} got {got}"
                                ));
                            }
                        }
                    }
                }
            }
        }

        for handle in [
            buf_input.handle,
            buf_weights.handle,
            buf_bias.handle,
            buf_output.handle,
        ] {
            let _ = close_bo(fd, handle);
        }
        for (buffer, _) in &command_buffers {
            let _ = close_bo(fd, buffer.handle);
        }

        if failure.mismatches == 0 && !failure.timed_out {
            Ok(())
        } else {
            Err(failure)
        }
    }
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn dense_channel_order_survives_a_1x1_identity_conv() {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();

    let mut passed = 0;
    for i in 0..REPS {
        match run(fd, &file, 8, 8) {
            Ok(()) => {
                println!("rep {i}: ok");
                passed += 1;
            }
            Err(failure) => {
                println!(
                    "rep {i}: FAIL ({} mismatches, timed_out={})",
                    failure.mismatches, failure.timed_out
                );
                for sample in &failure.samples {
                    println!("         {sample}");
                }
            }
        }
    }

    println!("\n=== summary: dense_channel_order_survives_a_1x1_identity_conv ===");
    println!("  {passed}/{REPS} passed");

    assert_eq!(
        passed, REPS,
        "the Cin<=4 dense/ARGB path did not preserve channel order at least once -- \
         see samples above for which output channel read back which input channel's value"
    );
}
