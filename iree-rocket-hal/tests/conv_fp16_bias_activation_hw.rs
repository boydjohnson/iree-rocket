#![cfg(feature = "hardware-characterization")]

//! Hardware characterization: a **nonzero** fp16 bias on the BS plane, alone
//! and underneath each fused activation.
//!
//!   CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!     cargo test --target aarch64-unknown-linux-gnu --release \
//!       --features hardware-characterization \
//!       --test conv_fp16_bias_activation_hw --no-run
//!
//!   ./conv_fp16_bias_activation_hw-<hash> --ignored --nocapture
//!
//! # Why
//!
//! Every fp16 convolution this project compiles passes the NPU a **zero**
//! bias and adds the real one back in a CPU shim
//! (`@call_rocket_dynamic_conv2d` in the transform spec fills
//! `%zero_bias` and the epilogue does `extf` then `addf %initial`). The
//! driver's `pack_fp16_bias_to_rocket` widens an fp16 bias into BRDMA's fp32
//! operand stream and is wired up (`command_buffer.rs`), but nothing has ever
//! driven it with a nonzero value on hardware: `conv_activation_fused_hw`,
//! the only fp16 activation coverage, zero-fills its bias buffer.
//!
//! That matters now because fusing a model's ReLU6 into the convolution
//! **requires** the bias to move. The hardware order is
//! accumulate -> BS (bias) -> BN (activation) -> OUT_CVT, so a clamp in BN
//! sees the biased value. A model clamps after its bias too -- but only if
//! the bias is on the BS plane. Leaving the bias in the CPU epilogue and
//! turning BN on would clamp the *unbiased* accumulator, which is a
//! different function.
//!
//! So this file answers two questions before any compiler work depends on
//! them: does a nonzero fp16 bias reach the output correctly, and does the
//! activation see the biased value?
//!
//! # Method
//!
//! A 1x1 kernel at `Cin` 1 with unit coefficients makes the pre-bias
//! accumulator at each pixel exactly that pixel's input, so the expected
//! output is `activation(input[pixel] + bias[channel])` with no reduction to
//! reason about. Inputs run -4..11 and biases -2, 0, 2 and 5 across the
//! output channels, so `acc + bias` lands below 0, inside [0, 6] and above 6
//! for every activation -- a test that only produced in-range values would
//! pass with the clamp disabled. Every value is a small integer and exact in
//! fp16, so this takes no tolerance.

#[path = "support/conv2d_oracle.rs"]
mod conv2d_oracle;

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use conv2d_oracle::{feature_offset, input_storage_bytes, output_offset, output_storage_bytes};

use iree_rocket_hal::rocket::{
    conv::{Activation, Buffers, Kernels, Shape, Tile, conv_2d_tile, relocate},
    device::{Buffer, JobDesc, close_bo, fini_bo, prep_bo, submit_jobs},
    tensor_layout::pack_depthwise_to_rocket_weights,
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const PAGE_BYTES: usize = 4096;
const FP16_BYTES: usize = 2;
const WIDTH: u32 = 16;
const HEIGHT: u32 = 16;
const IN_CHANNELS: u32 = 1;
const OUT_CHANNELS: usize = 4;
const KERNELS: Kernels = [1, 1];

/// One bias per output channel, chosen so `input + bias` crosses both ends
/// of the ReLU6 range within a single job.
const BIASES: [f32; OUT_CHANNELS] = [-2.0, 0.0, 2.0, 5.0];

fn page_aligned_size(size: usize) -> usize {
    size.div_ceil(PAGE_BYTES) * PAGE_BYTES
}

fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    if value == 0.0 {
        return ((bits >> 16) & 0x8000) as u16;
    }
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let fraction = bits & 0x7f_ffff;
    assert!(
        (1..31).contains(&exponent) && fraction & 0x1fff == 0,
        "{value} is not exactly representable in fp16"
    );
    sign | ((exponent as u16) << 10) | ((fraction >> 13) as u16)
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exponent = u32::from((bits >> 10) & 0x1f);
    let fraction = u32::from(bits & 0x3ff);
    let word = match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let mut exponent = -1i32;
            let mut fraction = fraction;
            while fraction & 0x400 == 0 {
                fraction <<= 1;
                exponent -= 1;
            }
            sign | (((exponent + 127 - 14) as u32) << 23) | ((fraction & 0x3ff) << 13)
        }
        0x1f => sign | (0xffu32 << 23) | (fraction << 13),
        _ => sign | ((exponent + 127 - 15) << 23) | (fraction << 13),
    };
    f32::from_bits(word)
}

/// The pre-bias accumulator this fixture puts at each pixel.
fn input_value(pixel: usize) -> f32 {
    (pixel % 16) as f32 - 4.0
}

struct Failure {
    mismatches: usize,
    samples: Vec<String>,
}

/// Runs one job with `activation` fused over a nonzero per-channel bias, and
/// checks every output against `expected(input + bias)`.
///
/// `depthwise` selects the other register program -- `CNA_CONV_CON1.CONV_MODE
/// = 3`, `CORE_MISC_CFG.DW_EN = 1` and a tap-major coefficient layout -- which
/// is the reason it is worth running the same checks twice rather than
/// assuming the BS and BN stages behave identically under both.
fn run(
    depthwise: bool,
    activation: Activation,
    expected: impl Fn(f32) -> f32,
) -> Result<(), Failure> {
    // Depthwise has one filter per channel, so `Cin` must equal `Cout`; the
    // dense case keeps `Cin` 1 so the accumulator is the input byte alone.
    let shape = if depthwise {
        Shape::with_out_channels(WIDTH, HEIGHT, 1, OUT_CHANNELS as u32, OUT_CHANNELS as u32)
            .with_depthwise()
            .with_activation(activation)
    } else {
        Shape::with_channels(WIDTH, HEIGHT, IN_CHANNELS, OUT_CHANNELS as u32)
            .with_activation(activation)
    };
    let width = WIDTH as usize;
    let pixels = width * HEIGHT as usize;
    // Both layouts come from the oracle support module rather than being
    // spelled here: an fp16 feature cube is dense at small `Cin` and
    // NC1HWC2 above it, and getting that wrong is invisible to a uniform
    // input -- which is exactly why `conv_activation_fused_hw`, whose input
    // is a single repeated value, could never have caught it.
    let output_bytes = output_storage_bytes(shape, KERNELS);

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let input_bytes = input_storage_bytes(shape);
        let buf_input = Buffer::new(fd, page_aligned_size(input_bytes), &file);
        ptr::write_bytes(buf_input.host_ptr, 0, buf_input.size);
        let input = std::slice::from_raw_parts_mut(buf_input.host_ptr, buf_input.size);
        let input_channels = if depthwise { OUT_CHANNELS } else { 1 };
        for pixel in 0..pixels {
            for channel in 0..input_channels {
                let offset = feature_offset(shape, channel, pixel / width, pixel % width);
                input[offset..offset + FP16_BYTES]
                    .copy_from_slice(&f32_to_f16(input_value(pixel)).to_le_bytes());
            }
        }

        // Unit coefficients. Depthwise takes the tap-major packed layout its
        // own builder needs; dense can fill raw, since every value is 1.0.
        let weight_bytes = if depthwise {
            shape.weight_bytes(KERNELS) as usize
        } else {
            KERNELS[0]
                * KERNELS[1]
                * shape.weight_channels() as usize
                * shape.padded_out_channels() as usize
                * FP16_BYTES
        };
        let buf_weights = Buffer::new(fd, page_aligned_size(weight_bytes), &file);
        ptr::write_bytes(buf_weights.host_ptr, 0, buf_weights.size);
        if depthwise {
            let dense = vec![f32_to_f16(1.0); OUT_CHANNELS * KERNELS[0] * KERNELS[1]]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<u8>>();
            // The depthwise packer's channel stride comes from the weight
            // buffer the shape asks for, not from `padded_out_channels` --
            // they are padded to different granules.
            let padded_channels = weight_bytes / (KERNELS[0] * KERNELS[1] * FP16_BYTES);
            pack_depthwise_to_rocket_weights(
                &dense,
                KERNELS[0],
                KERNELS[1],
                OUT_CHANNELS,
                padded_channels,
                FP16_BYTES,
                std::slice::from_raw_parts_mut(buf_weights.host_ptr, weight_bytes),
            )
            .expect("depthwise coefficient packing failed");
        } else {
            std::slice::from_raw_parts_mut(buf_weights.host_ptr as *mut u16, weight_bytes / 2)
                .fill(f32_to_f16(1.0));
        }

        // BRDMA reads the bias as fp32, one word per *padded* output
        // channel -- the widened form `pack_fp16_bias_to_rocket` produces.
        // Written here directly so this test exercises the hardware's
        // operand format rather than the driver's packer.
        let bias_bytes = shape.padded_out_channels() as usize * 4;
        let buf_bias = Buffer::new(fd, page_aligned_size(bias_bytes), &file);
        ptr::write_bytes(buf_bias.host_ptr, 0, buf_bias.size);
        let bias = std::slice::from_raw_parts_mut(
            buf_bias.host_ptr as *mut f32,
            shape.padded_out_channels() as usize,
        );
        for (channel, slot) in bias.iter_mut().enumerate() {
            *slot = BIASES.get(channel).copied().unwrap_or(0.0);
        }

        let buf_output = Buffer::new(fd, page_aligned_size(output_bytes), &file);
        ptr::write_bytes(buf_output.host_ptr, 0, buf_output.size);

        let mut commands = conv_2d_tile(shape, KERNELS, &Tile::whole(shape, KERNELS));
        relocate(
            &mut commands,
            Buffers {
                input: buf_input.dma_address,
                weights: buf_weights.dma_address,
                bias: buf_bias.dma_address,
                output: buf_output.dma_address,
            },
        );

        let command_bytes = commands.len() * mem::size_of::<u64>();
        let buf_commands = Buffer::new(fd, page_aligned_size(command_bytes), &file);
        ptr::write_bytes(buf_commands.host_ptr, 0, buf_commands.size);
        let words =
            std::slice::from_raw_parts_mut(buf_commands.host_ptr as *mut u64, commands.len());
        for (destination, command) in words.iter_mut().zip(&commands) {
            *destination = command.0;
        }

        let handles = [
            buf_input.handle,
            buf_weights.handle,
            buf_bias.handle,
            buf_output.handle,
            buf_commands.handle,
        ];
        for handle in handles {
            fini_bo(fd, handle).expect("failed to sync BO for the NPU");
        }

        let tasks = [(buf_commands.dma_address, commands.len() as u32)];
        let in_handles = [
            buf_commands.handle,
            buf_input.handle,
            buf_weights.handle,
            buf_bias.handle,
        ];
        let out_handles = [buf_output.handle];
        let jobs = [JobDesc {
            tasks: &tasks,
            in_handles: &in_handles,
            out_handles: &out_handles,
        }];

        submit_jobs(fd, &jobs).expect("SUBMIT failed");
        prep_bo(fd, buf_output.handle, 5_000_000_000).expect("job did not complete");

        let raw = std::slice::from_raw_parts(buf_output.host_ptr, output_bytes);
        let mut failure = Failure {
            mismatches: 0,
            samples: Vec::new(),
        };
        for pixel in 0..pixels {
            for channel in 0..OUT_CHANNELS {
                let want = expected(input_value(pixel) + BIASES[channel]);
                let offset = output_offset(shape, KERNELS, channel, pixel / width, pixel % width);
                let got = f16_to_f32(u16::from_le_bytes([raw[offset], raw[offset + 1]]));
                if got != want {
                    failure.mismatches += 1;
                    if failure.samples.len() < 8 {
                        failure.samples.push(format!(
                            "pixel {pixel} channel {channel}: acc {} bias {} want {want} got {got}",
                            input_value(pixel),
                            BIASES[channel]
                        ));
                    }
                }
            }
        }

        for handle in handles {
            let _ = close_bo(fd, handle);
        }
        if failure.mismatches == 0 {
            Ok(())
        } else {
            Err(failure)
        }
    }
}

fn check(label: &str, depthwise: bool, activation: Activation, expected: impl Fn(f32) -> f32) {
    match run(depthwise, activation, expected) {
        Ok(()) => println!("  {label}: exact over all {} outputs", 256 * OUT_CHANNELS),
        Err(failure) => panic!(
            "{label}: {} of {} outputs wrong\n  {}",
            failure.mismatches,
            256 * OUT_CHANNELS,
            failure.samples.join("\n  ")
        ),
    }
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn a_nonzero_fp16_bias_reaches_the_output() {
    check("bias only", false, Activation::None, |biased| biased);
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn relu_sees_the_biased_value() {
    check("bias + relu", false, Activation::Relu, |biased| {
        biased.max(0.0)
    });
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn relu6_clamps_the_biased_value_at_both_ends() {
    check(
        "bias + relu6",
        false,
        Activation::clamped_fp16(6.0),
        |biased| biased.clamp(0.0, 6.0),
    );
}

// The same three under the depthwise register program. ISSUES.md P7's
// depthwise offload needs the bias on the BS plane for exactly the reason the
// dense path did, and the depthwise program moves six register fields the
// dense one does not (`depthwise-conv.md` lists them), so this is measured
// rather than inherited.

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn a_nonzero_fp16_bias_reaches_a_depthwise_output() {
    check("depthwise bias only", true, Activation::None, |biased| {
        biased
    });
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn depthwise_relu_sees_the_biased_value() {
    check("depthwise bias + relu", true, Activation::Relu, |biased| {
        biased.max(0.0)
    });
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn depthwise_relu6_clamps_the_biased_value_at_both_ends() {
    check(
        "depthwise bias + relu6",
        true,
        Activation::clamped_fp16(6.0),
        |biased| biased.clamp(0.0, 6.0),
    );
}
