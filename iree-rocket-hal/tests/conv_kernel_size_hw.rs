//! Hardware probe for 7x7, 9x9, and 11x11 convolution kernels.
//!
//! This test is ignored on the development host because it needs the RK3588
//! NPU device. Cross-compile it, copy the printed test binary to the board,
//! and run it there:
//!
//!   CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!     cargo test --target aarch64-unknown-linux-gnu --release \
//!       --test conv_kernel_size_hw --no-run
//!
//!   ./conv_kernel_size_hw-<hash> --ignored --nocapture
//!
//! # What this is checking
//!
//! The focused capture sweep found typical CBUF splits of 8/4 for 7x7, 6/6
//! for 9x9, and 7/5 for 11x11, but also showed that the vendor's allocation
//! is not a simple function of whole feature and coefficient footprints.
//! The default run covers 24 points from that sweep with explicit bank
//! counts. The three high-`Cin` cases where the typical split cannot hold one
//! full-width kernel footprint are isolated behind `KERNEL_PROBE_CASE` so a
//! hardware timeout cannot hide the rest of the matrix:
//!
//!   KERNEL_PROBE_CASE=k9-ci64-w3 ./conv_kernel_size_hw-<hash> --ignored --nocapture
//!   KERNEL_PROBE_CASE=k11-ci48 ./conv_kernel_size_hw-<hash> --ignored --nocapture
//!   KERNEL_PROBE_CASE=k11-ci64 ./conv_kernel_size_hw-<hash> --ignored --nocapture
//!
//! The original 9x9/Cin64 experiment at 10/2 banks is retained as
//! `k9-ci64-w2`; it completes but leaves the whole output zero. The `w3`
//! case is the only remaining full-width allocation: nine data banks are the
//! minimum that can hold a nine-row footprint. A passing result would show
//! that standalone row-tile jobs can avoid reproducing the vendor's
//! width-partitioned multi-core schedule.
//!
//! Input and weights are 1.0, with padded input lanes zero. Each output is
//! therefore `Cin * valid_y_taps * valid_x_taps`, independent of coefficient
//! order. All tested values are exactly representable in fp16.

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    builders::{
        RegCmd, RegisterMeta,
        cna::{CnaDcompAddr0, CnaFeatureDataAddr},
        dpu::DpuDstBaseAddr,
        dpu_rdma::DpuRdmaBsBaseAddr,
    },
    conv::{FeatureLayout, Kernels, Shape, Tile, conv_2d_tile_with_cbuf_banks},
    device::{Buffer, JobDesc, close_bo, fini_bo, prep_bo, submit_jobs},
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const FP16_BYTES: usize = 2;
const FEATURE_ATOM_BYTES: usize = 16;
const CHANNELS_PER_ATOM: usize = 8;
const PAGE_BYTES: usize = 4096;
const FP16_ONE: u16 = 0x3c00;
const COMPLETION_TIMEOUT_NS: u64 = 10_000_000_000;
const ISOLATED_COMPLETION_TIMEOUT_NS: u64 = 30_000_000_000;

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
    let radius = kernel / 2;
    1 + coordinate.min(radius) + (extent - coordinate - 1).min(radius)
}

/// Writes the input feature map, real channels 1.0 and padding zero.
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

fn run(shape: Shape, kernels: Kernels, data_banks: u32, weight_banks: u32) -> Result<u32, Failure> {
    let width = shape.width as usize;
    let height = shape.height as usize;
    let out_width = shape.output_width(kernels) as usize;
    let out_height = shape.output_height(kernels) as usize;
    let in_surfaces = (shape.weight_channels() / 8) as usize;
    // The DPU writes whole granules, so the destination has to hold the
    // padded channel count even when the caller only wants `out_channels`.
    let out_surfaces = (shape.padded_out_channels() / 8) as usize;

    let input_bytes = match shape.layout() {
        FeatureLayout::Dense => width * height * shape.in_channels as usize * FP16_BYTES,
        FeatureLayout::Surfaces => in_surfaces * width * height * FEATURE_ATOM_BYTES,
    };
    let output_bytes = out_surfaces * out_width * out_height * FEATURE_ATOM_BYTES;

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let buf_input = Buffer::new(fd, page_aligned_size(input_bytes), &file);
        fill_input(buf_input.host_ptr, buf_input.size, shape, in_surfaces);

        // Coefficients cover the padded input channel count and the real
        // output count, all ones. Padding channels multiply zeroed input, so
        // they contribute nothing.
        let weight_bytes = shape.weight_bytes(kernels) as usize;
        let buf_weights = Buffer::new(fd, page_aligned_size(weight_bytes), &file);
        ptr::write_bytes(buf_weights.host_ptr, 0, buf_weights.size);
        std::slice::from_raw_parts_mut(buf_weights.host_ptr as *mut u16, weight_bytes / 2)
            .fill(FP16_ONE);

        let buf_bias = Buffer::new(fd, PAGE_BYTES, &file);
        ptr::write_bytes(buf_bias.host_ptr, 0, buf_bias.size);
        let buf_output = Buffer::new(fd, page_aligned_size(output_bytes), &file);
        ptr::write_bytes(buf_output.host_ptr, 0, buf_output.size);

        let tiles = shape.min_tiles_for_data_banks(kernels, data_banks);
        let split = Tile::split(shape, kernels, tiles);
        let mut command_buffers = Vec::with_capacity(split.len());
        for tile in &split {
            let mut commands =
                conv_2d_tile_with_cbuf_banks(shape, kernels, tile, data_banks, weight_banks);
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
        let timeout_ns = if std::env::var_os("KERNEL_PROBE_CASE").is_some() {
            ISOLATED_COMPLETION_TIMEOUT_NS
        } else {
            COMPLETION_TIMEOUT_NS
        };
        prep_bo(fd, buf_output.handle, timeout_ns)
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
                // Only the real output channels are checked. What the
                // hardware leaves in the padding lanes of the last granule
                // is not something the vendor captures constrain.
                for channel in 0..shape.out_channels as usize {
                    let surface = channel / CHANNELS_PER_ATOM;
                    let lane = channel % CHANNELS_PER_ATOM;
                    let offset = surface * out_width * out_height * FEATURE_ATOM_BYTES
                        + (y * out_width + x) * FEATURE_ATOM_BYTES
                        + lane * FP16_BYTES;
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
            Ok(tiles)
        } else {
            Err(failure)
        }
    }
}

fn attempt(
    shape: Shape,
    kernel: usize,
    data_banks: u32,
    weight_banks: u32,
    failures: &mut Vec<String>,
) {
    let kernels = [kernel, kernel];
    let label = format!(
        "k{kernel:<2} Cin {:>2} Cout {:>2} {}x{} d{data_banks}/w{weight_banks}",
        shape.in_channels, shape.out_channels, shape.width, shape.height,
    );
    let tiles = shape.min_tiles_for_data_banks(kernels, data_banks);
    match run(shape, kernels, data_banks, weight_banks) {
        Ok(tiles) => println!(
            "  ok   {label} {tiles:>2} tile(s)  padded {}  weights {}B",
            shape.padded_out_channels(),
            shape.weight_bytes(kernels),
        ),
        Err(failure) => {
            println!(
                "  FAIL {label} {tiles} tile(s)  {} mismatches",
                failure.mismatches
            );
            for sample in &failure.samples {
                println!("         {sample}");
            }
            failures.push(label);
        }
    }
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn large_kernel_cbuf_partitions_run_on_npu() {
    let mut failures = Vec::new();

    if let Ok(probe) = std::env::var("KERNEL_PROBE_CASE") {
        let (kernel, cin, data_banks, weight_banks) = match probe.as_str() {
            "k9-ci64" | "k9-ci64-w2" => (9usize, 64u32, 10u32, 2u32),
            "k9-ci64-w3" => (9, 64, 9, 3),
            "k11-ci48" => (11, 48, 9, 3),
            "k11-ci64" => (11, 64, 11, 1),
            _ => panic!(
                "unknown KERNEL_PROBE_CASE={probe:?}; expected k9-ci64-w2, k9-ci64-w3, \
                 k11-ci48, or k11-ci64"
            ),
        };
        println!("isolated timeout-risk probe: {probe}");
        attempt(
            Shape::with_out_channels(256, 32, 1, cin, 64),
            kernel,
            data_banks,
            weight_banks,
            &mut failures,
        );
        assert!(
            failures.is_empty(),
            "{} configuration(s) produced wrong output: {}",
            failures.len(),
            failures.join(", ")
        );
        return;
    }

    // Cin axis of the focused sweep. The center Cout is 64. The first split
    // for each kernel is the stable capture regime; the high-Cin overrides
    // that can time out the NPU are selected separately with
    // KERNEL_PROBE_CASE, above.
    for &(kernel, cin, data_banks, weight_banks) in &[
        (7usize, 16u32, 8u32, 4u32),
        (7, 24, 8, 4),
        (7, 32, 8, 4),
        (7, 48, 8, 4),
        (7, 64, 8, 4),
        (9, 16, 6, 6),
        (9, 24, 6, 6),
        (9, 32, 6, 6),
        (9, 48, 7, 5),
        (11, 16, 7, 5),
        (11, 24, 7, 5),
        (11, 32, 7, 5),
    ] {
        attempt(
            Shape::with_out_channels(256, 32, 1, cin, 64),
            kernel,
            data_banks,
            weight_banks,
            &mut failures,
        );
    }

    // Cout axis of the focused sweep at Cin 32. Cout 64 was covered above;
    // run the remaining four points with the same typical per-kernel split.
    for &(kernel, data_banks, weight_banks) in &[(7usize, 8u32, 4u32), (9, 6, 6), (11, 7, 5)] {
        for cout in [16u32, 32, 48, 96] {
            attempt(
                Shape::with_out_channels(256, 32, 1, 32, cout),
                kernel,
                data_banks,
                weight_banks,
                &mut failures,
            );
        }
    }

    println!(
        "  note: run KERNEL_PROBE_CASE=k9-ci64-w3, k11-ci48, and k11-ci64 separately \
         to cover the three timeout-risk points"
    );

    assert!(
        failures.is_empty(),
        "{} configuration(s) produced wrong output: {}",
        failures.len(),
        failures.join(", ")
    );
}
