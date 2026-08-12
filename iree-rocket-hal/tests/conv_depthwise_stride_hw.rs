//! Hardware validation of depthwise convolution at stride > 1.
//!
//! `conv_depthwise_hw.rs` confirmed the tap-major weight-packing layout via
//! an impulse probe, but only ever at stride 1 (`Shape::with_out_channels`'s
//! stride argument is hardcoded there). `conv_wide_shape_hw.rs` separately
//! confirmed stride 2/3/4 for *dense* convolution via an all-ones/tap-count
//! oracle: uniform input times uniform weights makes every output pixel
//! equal to however many taps landed inside the image, which is exact in
//! fp16 and needs no per-tap discrimination. Depthwise's `DW_EN` register
//! mode (`CNA_CONV_CON1.CONV_MODE=3`, doubled channel/weight-byte granule)
//! has never been run at stride > 1 on real hardware -- this file is that
//! gap.
//!
//! This reuses the tap-count oracle rather than adapting the impulse trick
//! to depthwise-at-stride: the impulse test's discriminating power comes
//! from *overlapping* output receptive fields each reading a different tap
//! of the same impulse, which only holds while `stride < kernel`. At
//! stride 3 (kernel 3) receptive fields exactly tile the input with no
//! overlap, and at stride 4 they leave gaps -- an impulse placed in a gap
//! is read by no output pixel at all and the test would silently pass
//! against an all-zero expectation. The tap-count oracle has no such dead
//! zone: `valid_taps` already accounts for exactly which taps of a
//! `y*stride, x*stride`-centred window survive the image boundary, so it
//! stays meaningful at every stride. What it does *not* independently
//! re-confirm is per-tap placement inside the packed buffer -- that is
//! `conv_depthwise_hw.rs`'s job, and nothing here suggests the packer's
//! layout (a pure function of channel count and kernel size) has any
//! stride dependence to re-check.
//!
//! # Scope
//!
//! fp16 depthwise, `Cin=Cout` in {8, 12} (whole atom vs. channel-padded,
//! mirroring `conv_depthwise_hw.rs`'s own two cases), kernels 1x1 and 3x3,
//! strides 2/3/4. `EXTENT=32` throughout -- small enough that the CBUF
//! capacity/tiling questions `conv_wide_shape_hw.rs` exists for don't come
//! up; this is purely about the stride/`DW_EN` combination.

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    conv::{Buffers, Kernels, Shape, Tile, conv_2d_tile, relocate},
    device::{Buffer, JobDesc, close_bo, fini_bo, prep_bo, submit_jobs},
    tensor_layout::pack_depthwise_to_rocket_weights,
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const FP16_BYTES: usize = 2;
const FEATURE_ATOM_BYTES: usize = 16;
const CHANNELS_PER_ATOM: usize = 8;
const PAGE_BYTES: usize = 4096;
const FP16_ONE: u16 = 0x3c00;

const EXTENT: u32 = 32;

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

fn valid_taps(coordinate: usize, extent: usize, kernel: usize) -> usize {
    match kernel {
        1 => 1,
        3 => 3 - usize::from(coordinate == 0) - usize::from(coordinate + 1 == extent),
        _ => unreachable!("only 1x1 and 3x3 have vendor reference data"),
    }
}

/// Uniform input and weights of 1.0 make every output channel equal to the
/// count of taps that landed inside the image -- depthwise has no
/// cross-channel sum, unlike `conv_wide_shape_hw.rs`'s dense `expected`,
/// which multiplies by `INPUT_CHANNELS`.
fn expected(kernels: Kernels, shape: Shape, y: usize, x: usize) -> f32 {
    let stride = shape.stride as usize;
    (valid_taps(y * stride, shape.height as usize, kernels[0])
        * valid_taps(x * stride, shape.width as usize, kernels[1])) as f32
}

struct Failure {
    mismatches: usize,
    samples: Vec<String>,
}

fn run(channels: u32, kernels: Kernels, stride: u32) -> Result<(), Failure> {
    let shape =
        Shape::with_out_channels(EXTENT, EXTENT, stride, channels, channels).with_depthwise();
    let channels = channels as usize;
    let (kh, kw) = (kernels[0], kernels[1]);
    let out_width = shape.output_width(kernels) as usize;
    let out_height = shape.output_height(kernels) as usize;

    let weight_bytes = shape.weight_bytes(kernels) as usize;
    let padded_channels = weight_bytes / (kh * kw * FP16_BYTES);
    assert!(
        padded_channels >= channels,
        "padded {padded_channels} < real {channels}"
    );

    let in_surfaces = (shape.weight_channels() as usize).div_ceil(CHANNELS_PER_ATOM);
    let out_surfaces = (shape.padded_out_channels() as usize).div_ceil(CHANNELS_PER_ATOM);

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let input = Buffer::new(
            fd,
            page_aligned_size(in_surfaces * EXTENT as usize * EXTENT as usize * FEATURE_ATOM_BYTES),
            &file,
        );
        std::slice::from_raw_parts_mut(input.host_ptr as *mut u16, input.size / FP16_BYTES)
            .fill(FP16_ONE);

        // Dense [channels][kh][kw], all-ones, then packed by the routine
        // conv_depthwise_hw.rs already validated at stride 1 -- packing is
        // a pure function of channel count and kernel size, not stride.
        let dense_bytes = channels * kh * kw * FP16_BYTES;
        let mut dense = vec![0u8; dense_bytes];
        for entry in dense.chunks_exact_mut(FP16_BYTES) {
            entry.copy_from_slice(&FP16_ONE.to_le_bytes());
        }
        let mut packed = vec![0u8; weight_bytes];
        pack_depthwise_to_rocket_weights(
            &dense,
            kh,
            kw,
            channels,
            padded_channels,
            FP16_BYTES,
            &mut packed,
        )
        .expect("depthwise packing failed");

        let weights = Buffer::new(fd, page_aligned_size(weight_bytes), &file);
        ptr::write_bytes(weights.host_ptr, 0, weights.size);
        ptr::copy_nonoverlapping(packed.as_ptr(), weights.host_ptr, packed.len());

        let bias = Buffer::new(fd, PAGE_BYTES, &file);
        ptr::write_bytes(bias.host_ptr, 0, bias.size);
        let output = Buffer::new(
            fd,
            page_aligned_size(out_surfaces * out_width * out_height * FEATURE_ATOM_BYTES),
            &file,
        );
        ptr::write_bytes(output.host_ptr, 0, output.size);

        let mut words = conv_2d_tile(shape, kernels, &Tile::whole(shape, kernels));
        relocate(
            &mut words,
            Buffers {
                input: input.dma_address,
                weights: weights.dma_address,
                bias: bias.dma_address,
                output: output.dma_address,
            },
        );
        let commands = Buffer::new(
            fd,
            page_aligned_size(words.len() * mem::size_of::<u64>()),
            &file,
        );
        ptr::write_bytes(commands.host_ptr, 0, commands.size);
        let slots = std::slice::from_raw_parts_mut(commands.host_ptr as *mut u64, words.len());
        for (destination, command) in slots.iter_mut().zip(&words) {
            *destination = command.0;
        }

        for handle in [
            input.handle,
            weights.handle,
            bias.handle,
            output.handle,
            commands.handle,
        ] {
            fini_bo(fd, handle).expect("failed to sync BO for the NPU");
        }

        let tasks = [(commands.dma_address, words.len() as u32)];
        let in_handles = [commands.handle, input.handle, weights.handle, bias.handle];
        let out_handles = [output.handle];
        submit_jobs(
            fd,
            &[JobDesc {
                tasks: &tasks,
                in_handles: &in_handles,
                out_handles: &out_handles,
            }],
        )
        .unwrap_or_else(|error| {
            panic!("Cin {channels} {kernels:?} s{stride}: SUBMIT failed: {error}")
        });
        prep_bo(fd, output.handle, 5_000_000_000).unwrap_or_else(|error| {
            panic!("Cin {channels} {kernels:?} s{stride}: did not complete: {error}")
        });

        let raw = std::slice::from_raw_parts(
            output.host_ptr,
            out_surfaces * out_width * out_height * FEATURE_ATOM_BYTES,
        );
        let at = |channel: usize, y: usize, x: usize| -> f32 {
            let offset =
                (channel / CHANNELS_PER_ATOM) * out_width * out_height * FEATURE_ATOM_BYTES
                    + (y * out_width + x) * FEATURE_ATOM_BYTES
                    + (channel % CHANNELS_PER_ATOM) * FP16_BYTES;
            f16_to_f32(u16::from_le_bytes([raw[offset], raw[offset + 1]]))
        };

        let mut failure = Failure {
            mismatches: 0,
            samples: Vec::new(),
        };
        for y in 0..out_height {
            for x in 0..out_width {
                let want = expected(kernels, shape, y, x);
                for channel in 0..channels {
                    let got = at(channel, y, x);
                    if got != want {
                        failure.mismatches += 1;
                        if failure.samples.len() < 8 {
                            failure
                                .samples
                                .push(format!("[c{channel}, {y}, {x}] want {want} got {got}"));
                        }
                    }
                }
            }
        }

        for handle in [
            input.handle,
            weights.handle,
            bias.handle,
            output.handle,
            commands.handle,
        ] {
            let _ = close_bo(fd, handle);
        }

        if failure.mismatches == 0 {
            Ok(())
        } else {
            Err(failure)
        }
    }
}

#[test]
#[ignore = "requires a real RK3588 NPU at /dev/accel/accel0"]
fn depthwise_strided_convs_run_on_npu() {
    // (channels, kernel, stride). Channels cover both the whole-atom case
    // (8) and the channel-padded case (12, per conv_depthwise_hw.rs's own
    // rationale for that value) at each stride; kernels cover both matcher
    // shapes (rocket-compiler-plugin's @match_dynamic_depthwise_conv2d is
    // 1x1, its _3x3 sibling is 3x3); strides cover the full hardware-
    // confirmed dense range (conv_wide_shape_hw.rs), which this file exists
    // to check depthwise against.
    const CASES: [(u32, Kernels, u32); 12] = [
        (8, [1, 1], 2),
        (8, [1, 1], 3),
        (8, [1, 1], 4),
        (8, [3, 3], 2),
        (8, [3, 3], 3),
        (8, [3, 3], 4),
        (12, [1, 1], 2),
        (12, [1, 1], 3),
        (12, [1, 1], 4),
        (12, [3, 3], 2),
        (12, [3, 3], 3),
        (12, [3, 3], 4),
    ];

    let mut failures = Vec::new();
    for (channels, kernels, stride) in CASES {
        match run(channels, kernels, stride) {
            Ok(()) => println!("  ok   Cin{channels:<3} {kernels:?} s{stride}"),
            Err(failure) => {
                println!(
                    "  FAIL Cin{channels:<3} {kernels:?} s{stride}  {} mismatches",
                    failure.mismatches
                );
                for sample in &failure.samples {
                    println!("         {sample}");
                }
                failures.push(format!(
                    "Cin{channels} {kernels:?} s{stride}: {} mismatches",
                    failure.mismatches
                ));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "depthwise stride cases failed:\n  {}",
        failures.join("\n  ")
    );
}
