//! Hardware validation of convolutions with more than four input channels.
//!
//! This test is ignored on the development host because it needs the RK3588
//! NPU device. Cross-compile it, copy the printed test binary to the board,
//! and run it there:
//!
//!   CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!     cargo test --target aarch64-unknown-linux-gnu --release \
//!       --test conv_multichannel_hw --no-run
//!
//!   ./conv_multichannel_hw-<hash> --ignored --nocapture
//!
//! # Why this is separate
//!
//! Every earlier convolution test feeds a dense NHWC buffer, which is what
//! the vendor uses while a pixel fits in half a feature atom -- `Cin` up to
//! 4. From `Cin` 5 the feature map becomes NC1HWC2 surfaces: one 16-byte
//! atom per pixel per surface, surfaces `width * height * 16` bytes apart.
//! The input has to be packed accordingly, so the buffer setup differs
//! enough from the dense tests to keep apart.
//!
//! # What makes the check independent of coefficient order
//!
//! Real channels are filled with 1.0 and every padding channel with zero,
//! and all weights are 1.0. Each output is then exactly the number of real
//! input channels times the taps that landed inside the image, whatever
//! order the hardware walks the coefficients in, because the padding
//! channels contribute nothing. Values stay small enough to be exact in
//! fp16 -- the largest here is `80 * 9 = 720`.
//!
//! # Scope
//!
//! `Cout=8`, stride 1, 1x1 and 3x3 kernels, `Cin` 1..=80. The channel
//! padding beyond 80 has no capture backing and `Shape::with_channels`
//! rejects it.

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    builders::{
        RegCmd, RegisterMeta,
        cna::{CnaDcompAddr0, CnaFeatureDataAddr},
        dpu::DpuDstBaseAddr,
        dpu_rdma::DpuRdmaBsBaseAddr,
    },
    conv::{FeatureLayout, Kernels, Shape, Tile, conv_2d_tile},
    device::{Buffer, JobDesc, close_bo, fini_bo, prep_bo, submit_jobs},
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const OUTPUT_CHANNELS: usize = 8;
const FP16_BYTES: usize = 2;
const FEATURE_ATOM_BYTES: usize = 16;
const CHANNELS_PER_ATOM: usize = 8;
const PAGE_BYTES: usize = 4096;
const FP16_ONE: u16 = 0x3c00;

fn page_aligned_size(size: usize) -> usize {
    size.div_ceil(PAGE_BYTES) * PAGE_BYTES
}

fn decode_identity(command: &RegCmd) -> (u32, u32) {
    ((command.0 >> 48) as u32, command.0 as u32 & 0xffff)
}

fn relocate<R: RegisterMeta>(commands: &mut [RegCmd], address: u32) {
    let matches: Vec<_> = commands
        .iter()
        .enumerate()
        .filter_map(|(index, command)| {
            (decode_identity(command) == (R::DOMAIN, R::OFFSET)).then_some(index)
        })
        .collect();
    assert_eq!(matches.len(), 1, "expected exactly one relocation site");
    let tile_offset = (commands[matches[0]].0 >> 16) as u32;
    commands[matches[0]] = RegCmd::new(R::DOMAIN, R::OFFSET, address + tile_offset);
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

/// Writes the input feature map, real channels 1.0 and padding zero.
///
/// Surfaces are allocated from the *weight* padding rather than
/// `datain_channel`, because the two disagree at three atoms and allocating
/// the larger of them satisfies either reading.
unsafe fn fill_input(base: *mut u8, size: usize, shape: Shape, surfaces: usize) {
    unsafe {
        ptr::write_bytes(base, 0, size);
        let width = shape.width as usize;
        let height = shape.height as usize;
        for channel in 0..shape.in_channels as usize {
            let surface = channel / CHANNELS_PER_ATOM;
            let lane = channel % CHANNELS_PER_ATOM;
            if surface >= surfaces {
                continue;
            }
            for y in 0..height {
                for x in 0..width {
                    let offset = match shape.layout() {
                        FeatureLayout::Dense => {
                            (y * width + x) * shape.in_channels as usize * FP16_BYTES
                                + channel * FP16_BYTES
                        }
                        FeatureLayout::Surfaces => {
                            surface * width * height * FEATURE_ATOM_BYTES
                                + (y * width + x) * FEATURE_ATOM_BYTES
                                + lane * FP16_BYTES
                        }
                    };
                    ptr::write((base.add(offset)) as *mut u16, FP16_ONE);
                }
            }
        }
    }
}

struct Failure {
    mismatches: usize,
    samples: Vec<String>,
}

fn run(shape: Shape, kernels: Kernels, tiles: u32) -> Result<(), Failure> {
    let width = shape.width as usize;
    let height = shape.height as usize;
    let out_width = shape.output_width(kernels) as usize;
    let out_height = shape.output_height(kernels) as usize;
    let surfaces = (shape.weight_channels() / 8) as usize;

    let input_bytes = match shape.layout() {
        FeatureLayout::Dense => width * height * shape.in_channels as usize * FP16_BYTES,
        FeatureLayout::Surfaces => surfaces * width * height * FEATURE_ATOM_BYTES,
    };
    let output_bytes = out_width * out_height * FEATURE_ATOM_BYTES * 2;

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let buf_input = Buffer::new(fd, page_aligned_size(input_bytes), &file);
        fill_input(buf_input.host_ptr, buf_input.size, shape, surfaces);

        // Coefficients cover the padded channel count, all ones. Padding
        // channels multiply zeroed input, so they contribute nothing.
        let weight_bytes = kernels[0]
            * kernels[1]
            * shape.weight_channels() as usize
            * OUTPUT_CHANNELS
            * FP16_BYTES;
        let buf_weights = Buffer::new(fd, page_aligned_size(weight_bytes), &file);
        ptr::write_bytes(buf_weights.host_ptr, 0, buf_weights.size);
        std::slice::from_raw_parts_mut(buf_weights.host_ptr as *mut u16, weight_bytes / 2)
            .fill(FP16_ONE);

        let buf_bias = Buffer::new(fd, PAGE_BYTES, &file);
        ptr::write_bytes(buf_bias.host_ptr, 0, buf_bias.size);
        let buf_output = Buffer::new(fd, page_aligned_size(output_bytes), &file);
        ptr::write_bytes(buf_output.host_ptr, 0, buf_output.size);

        let split = Tile::split(shape, kernels, tiles);
        let mut command_buffers = Vec::with_capacity(split.len());
        for tile in &split {
            let mut commands = conv_2d_tile(shape, kernels, tile);
            relocate::<CnaFeatureDataAddr>(&mut commands, buf_input.dma_address);
            relocate::<CnaDcompAddr0>(&mut commands, buf_weights.dma_address);
            relocate::<DpuRdmaBsBaseAddr>(&mut commands, buf_bias.dma_address);
            relocate::<DpuDstBaseAddr>(&mut commands, buf_output.dma_address);

            let command_bytes = commands.len() * mem::size_of::<u64>();
            let buffer = Buffer::new(fd, page_aligned_size(command_bytes), &file);
            ptr::write_bytes(buffer.host_ptr, 0, buffer.size);
            let words = std::slice::from_raw_parts_mut(buffer.host_ptr as *mut u64, commands.len());
            for (destination, command) in words.iter_mut().zip(&commands) {
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
            .unwrap_or_else(|error| panic!("{shape:?} {kernels:?} SUBMIT failed: {error}"));
        prep_bo(fd, buf_output.handle, 5_000_000_000)
            .unwrap_or_else(|error| panic!("{shape:?} {kernels:?} did not complete: {error}"));

        let raw = std::slice::from_raw_parts(buf_output.host_ptr, output_bytes);
        let mut failure = Failure {
            mismatches: 0,
            samples: Vec::new(),
        };
        for y in 0..out_height {
            for x in 0..out_width {
                let stride = shape.stride as usize;
                let want = (shape.in_channels as usize
                    * valid_taps(y * stride, height, kernels[0])
                    * valid_taps(x * stride, width, kernels[1])) as f32;
                for channel in 0..OUTPUT_CHANNELS {
                    let offset = (y * out_width + x) * FEATURE_ATOM_BYTES + channel * FP16_BYTES;
                    let got = f16_to_f32(u16::from_le_bytes([raw[offset], raw[offset + 1]]));
                    if got != want {
                        failure.mismatches += 1;
                        if failure.samples.len() < 8 {
                            failure
                                .samples
                                .push(format!("[{y}, {x}, {channel}] want {want} got {got}"));
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

        if failure.mismatches == 0 {
            Ok(())
        } else {
            Err(failure)
        }
    }
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn multi_channel_convs_run_on_npu() {
    // Channel counts chosen to cover both sides of the dense/surface
    // boundary, every distinct padding row in the table, and both of the
    // measured exceptions: Cin 20 and 24 pad their coefficients to 32 while
    // datain_channel stays 24, and Cin 56 rounds both up to 64.
    const CHANNELS: [u32; 12] = [3, 4, 5, 8, 12, 16, 20, 24, 32, 40, 56, 64];

    let mut failures = Vec::new();
    for cin in CHANNELS {
        for kernels in [[1usize, 1], [3, 3]] {
            let shape = Shape::with_channels(64, 32, 1, cin);
            let tiles = shape.min_tiles(kernels);
            match run(shape, kernels, tiles) {
                Ok(()) => println!(
                    "  ok   Cin {cin:>3} {kernels:?} {tiles} tile(s)  {:?}  \
                     dchan {} weights {} banks d{}",
                    shape.layout(),
                    shape.padded_channels(),
                    shape.weight_channels(),
                    shape.data_banks(),
                ),
                Err(failure) => {
                    println!(
                        "  FAIL Cin {cin:>3} {kernels:?} {tiles} tile(s)  {:?}  {} mismatches",
                        shape.layout(),
                        failure.mismatches
                    );
                    for sample in &failure.samples {
                        println!("         {sample}");
                    }
                    failures.push(format!("Cin {cin} {kernels:?}"));
                }
            }
        }
    }

    assert!(
        failures.is_empty(),
        "{} channel counts produced wrong output: {}",
        failures.len(),
        failures.join(", ")
    );
}
