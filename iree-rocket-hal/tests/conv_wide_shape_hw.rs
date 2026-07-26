//! Hardware validation of the shape-generalised convolution builder.
//!
//! This test is ignored on the development host because it needs the RK3588
//! NPU device. Cross-compile it, copy the printed test binary to the board,
//! and run it there:
//!
//!   CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!     cargo test --target aarch64-unknown-linux-gnu --release \
//!       --test conv_wide_shape_hw --no-run
//!
//!   ./conv_wide_shape_hw-<hash> --ignored --nocapture
//!
//! Everything before this test ran at the captured `32x32` geometry.
//! `conv::Shape` generalises the builder to arbitrary width and height, with
//! every width-dependent register formula validated against 212 convolution
//! programs from 35 vendor captures spanning widths 32..256 and heights
//! 32..256. This test is the hardware half of that: it runs shapes the
//! builder has never executed and checks every output element.
//!
//! # Scope
//!
//! `Cin=3` dense NHWC, `Cout=8`, 1x1 and 3x3 kernels, strides 1 to 4 -- the
//! case the register formulas were validated for. Wider channel counts move
//! the feature map to multiple NC1HWC2 surfaces and change the row strides,
//! and nothing here covers that.
//!
//! # The capacity ceiling
//!
//! Two bounds limit how much of a feature map one program may cover.
//! `CNA_CBUF_CON1.data_entries` is 14 bits and holds `tile_input_rows *
//! width`, a hard 16383. The vendor's own rule is tighter -- it keeps that
//! product within `data_banks * 1024`, giving 32 rows at 256 wide and 7 at
//! 1536 -- and `Shape::max_tile_input_rows` follows the vendor. Shapes below
//! exercise both sides: geometries that fit whole, and geometries that must
//! be split, which is the case real feature maps hit.
//!
//! # Stride
//!
//! At stride greater than one the output geometry shrinks, so output-side
//! registers and the output buffer follow `output_width`/`output_height`
//! rather than the input extents, and a tile's halo projects back through
//! the stride as `out_first * stride - pad`. Both were confirmed on 150
//! stride-2, -3 and -4 programs in the sweep corpus; these runs are the
//! hardware half.

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    builders::{
        RegCmd, RegisterMeta,
        cna::{CnaDcompAddr0, CnaFeatureDataAddr},
        dpu::DpuDstBaseAddr,
        dpu_rdma::DpuRdmaBsBaseAddr,
    },
    conv::{Kernels, Shape, Tile, conv_2d_tile},
    device::{Buffer, JobDesc, close_bo, fini_bo, prep_bo, submit_jobs},
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const INPUT_CHANNELS: usize = 3;
const WEIGHT_INPUT_CHANNELS: usize = 8;
const OUTPUT_CHANNELS: usize = 8;
const FP16_BYTES: usize = 2;
const FEATURE_ATOM_BYTES: usize = 16;
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

/// Uniform inputs and weights of 1.0 make every output the count of taps that
/// landed inside the image, which is exact in fp16 for these sizes.
fn expected(kernels: Kernels, shape: Shape, y: usize, x: usize) -> f32 {
    // Output pixel (y, x) is centred on input (y*s, x*s); the tap count is
    // whatever lands inside the image.
    let stride = shape.stride as usize;
    (INPUT_CHANNELS
        * valid_taps(y * stride, shape.height as usize, kernels[0])
        * valid_taps(x * stride, shape.width as usize, kernels[1])) as f32
}

struct Failure {
    mismatches: usize,
    samples: Vec<String>,
}

/// Runs the whole convolution as `tiles` independent jobs and checks every
/// output element.
fn run(shape: Shape, kernels: Kernels, tiles: u32) -> Result<(), Failure> {
    let width = shape.width as usize;
    let height = shape.height as usize;
    let out_width = shape.output_width(kernels) as usize;
    let out_height = shape.output_height(kernels) as usize;
    let input_bytes = width * height * INPUT_CHANNELS * FP16_BYTES;
    // Two NC1HWC2 surfaces: eight real fp16 channels sit in the first.
    let output_bytes = out_width * out_height * FEATURE_ATOM_BYTES * 2;

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let buf_input = Buffer::new(fd, page_aligned_size(input_bytes), &file);
        ptr::write_bytes(buf_input.host_ptr, 0, buf_input.size);
        std::slice::from_raw_parts_mut(buf_input.host_ptr as *mut u16, input_bytes / FP16_BYTES)
            .fill(FP16_ONE);

        let weight_bytes =
            kernels[0] * kernels[1] * WEIGHT_INPUT_CHANNELS * OUTPUT_CHANNELS * FP16_BYTES;
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

        submit_jobs(fd, &jobs).unwrap_or_else(|error| {
            panic!("{shape:?} {kernels:?} {tiles}-tile SUBMIT failed: {error}")
        });
        prep_bo(fd, buf_output.handle, 5_000_000_000).unwrap_or_else(|error| {
            panic!("{shape:?} {kernels:?} {tiles}-tile did not complete: {error}")
        });

        let raw = std::slice::from_raw_parts(buf_output.host_ptr, output_bytes);
        let mut failure = Failure {
            mismatches: 0,
            samples: Vec::new(),
        };
        for y in 0..out_height {
            for x in 0..out_width {
                let want = expected(kernels, shape, y, x);
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
fn shape_generalised_convs_run_on_npu() {
    // Widths and heights the builder has never executed, chosen to cover
    // non-square geometry, both kernels, and both sides of the 14-bit
    // data_entries ceiling. 256x64 and 128x128 exceed it whole and are only
    // reachable by splitting.
    const SHAPES: [(u32, u32, u32); 10] = [
        // stride 1
        (64, 32, 1),
        (32, 64, 1),
        (64, 64, 1),
        (128, 32, 1),
        (256, 64, 1),
        (128, 128, 1),
        // stride > 1: output geometry, halo projection, and CONV_CON3
        (64, 64, 2),
        (128, 64, 2),
        (128, 64, 3),
        (64, 64, 4),
    ];

    let mut failures = Vec::new();
    for (width, height, stride) in SHAPES {
        let shape = Shape::with_stride(width, height, stride);
        for kernels in [[1usize, 1], [3, 3]] {
            let minimum = shape.min_tiles(kernels);
            // The smallest legal split, and one more, so multi-job submission
            // is exercised even where a single program would have fitted.
            for tiles in [minimum, minimum + 1] {
                match run(shape, kernels, tiles) {
                    Ok(()) => println!(
                        "  ok   {width:>3}x{height:<3} s{stride} {kernels:?} {tiles} tile(s)  \
                         out {}x{}  banks d{}/w{}  max_rows {}",
                        shape.output_width(kernels),
                        shape.output_height(kernels),
                        shape.data_banks(kernels),
                        shape.weight_banks(kernels),
                        shape.max_tile_input_rows(kernels),
                    ),
                    Err(failure) => {
                        println!(
                            "  FAIL {width:>3}x{height:<3} s{stride} {kernels:?} {tiles} tile(s)  \
                             {} mismatches",
                            failure.mismatches
                        );
                        for sample in &failure.samples {
                            println!("         {sample}");
                        }
                        failures.push(format!(
                            "{width}x{height} s{stride} {kernels:?} {tiles} tiles"
                        ));
                    }
                }
            }
        }
    }

    assert!(
        failures.is_empty(),
        "{} shape/tiling combinations produced wrong output: {}",
        failures.len(),
        failures.join(", ")
    );
}
