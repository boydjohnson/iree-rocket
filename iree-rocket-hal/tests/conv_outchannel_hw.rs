//! Hardware validation of convolutions with more than eight output channels.
//!
//! This test is ignored on the development host because it needs the RK3588
//! NPU device. Cross-compile it, copy the printed test binary to the board,
//! and run it there:
//!
//!   CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!     cargo test --target aarch64-unknown-linux-gnu --release \
//!       --test conv_outchannel_hw --no-run
//!
//!   ./conv_outchannel_hw-<hash> --ignored --nocapture
//!
//! # What this is checking
//!
//! `Cout` splits across two register groups. The CNA counts real kernels
//! (`weight_kernels`, and `weight_bytes` scaled by them) while the DPU counts
//! whole 16-channel granules (`dataout_channel`, `data_cube_channel.channel`,
//! the RDMA's copy of it, and `channel_wdma`), with `orig_channel` the odd one
//! out on the DPU side carrying the true count. Getting the two mixed up is
//! invisible at `Cout` 8 and 16, where the padded value is 16 either way --
//! which is exactly the range every earlier test covers.
//!
//! The second thing under test is the CBUF split. `Cout` reaches it only
//! through the coefficient footprint, and only when the feature data is
//! already asking for more banks than exist. The corpus contains one such
//! shape and it is included below.
//!
//! # What makes the check independent of coefficient order
//!
//! Real input channels are filled with 1.0 and every padding channel with
//! zero, and all weights are 1.0. Each output is then exactly the number of
//! real input channels times the taps that landed inside the image, whatever
//! order the hardware walks the coefficients in. That holds per output
//! channel too, so every real output channel must carry the same value --
//! and a program that miscounts them writes some of them not at all, which
//! shows up as a zero rather than a wrong number.
//!
//! Values stay exact in fp16: the largest here is `32 * 9 = 288`.
//!
//! # Scope
//!
//! `Cout` 1..=128, `Cin` 3..=32, stride 1, 1x1 and 3x3 kernels.

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
            Ok(())
        } else {
            Err(failure)
        }
    }
}

fn attempt(shape: Shape, kernels: Kernels, failures: &mut Vec<String>) {
    let tiles = shape.min_tiles(kernels);
    let label = format!(
        "Cin {:>2} Cout {:>3} {}x{} {kernels:?}",
        shape.in_channels, shape.out_channels, shape.width, shape.height
    );
    match run(shape, kernels, tiles) {
        Ok(()) => println!(
            "  ok   {label} {tiles} tile(s)  padded {} banks d{}/w{}  weights {}B",
            shape.padded_out_channels(),
            shape.data_banks(kernels),
            shape.weight_banks(kernels),
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

/// Every `(shape, kernels)` the device test runs, as one list so the
/// buildability check below cannot drift from it.
fn matrix() -> Vec<(Shape, Kernels)> {
    // Values chosen so that the true and padded counts disagree in every way
    // they can: below one granule (1, 3), exactly one (8, 16), and each of
    // the ragged points between granules (9, 20, 40, 56, 72). 8 and 16 are
    // kept as the controls -- they are the only ones earlier tests covered.
    //
    // The tail past 128 is new. Cout has capture backing to 512 and the
    // 14-bit `weight_kernels` field encodes it, but hardware had only ever
    // run to 128. Unlike Cin, the output padding is a clean granule with no
    // exceptions, so these cover the range rather than probe a rule -- the
    // interesting part is that the coefficient footprint grows with Cout
    // until it no longer fits the CBUF, and the vendor's own single-core
    // plan streams it rather than splitting the kernel set.
    const OUT_CHANNELS: [u32; 18] = [
        1, 3, 8, 9, 16, 20, 32, 40, 48, 56, 64, 128, 160, 192, 256, 320, 384, 512,
    ];

    let mut cases = Vec::new();
    for cout in OUT_CHANNELS {
        for kernels in [[1usize, 1], [3, 3]] {
            cases.push((Shape::with_out_channels(64, 32, 1, 8, cout), kernels));
        }
    }

    // A dense input at a large kernel count: the ARGB path and the output
    // granule padding have never been exercised together.
    for cout in [16u32, 40, 128] {
        cases.push((Shape::with_out_channels(64, 32, 1, 3, cout), [3, 3]));
    }

    // Both channel axes large at once, where the coefficient footprint stops
    // fitting the CBUF and has to stream. Cin 64 / Cout 512 is the vendor's
    // own captured shape: 589824 bytes of coefficients, an 18-bank demand
    // against the eight the builder grants it, and the builder computes the
    // 4/8 split the capture programs. Cin 128 / Cout 256 is the same
    // footprint reached from the other axis.
    //
    // These are the shapes that would need kernel-set splitting if streaming
    // did not work. The vendor's single-core plan does not split -- no
    // plan-0 program in 576 captures covers a partial kernel set -- so what
    // is under test is whether the builder's one task is enough.
    for (in_channels, out_channels) in [(64u32, 512u32), (128, 256)] {
        cases.push((
            Shape::with_out_channels(32, 32, 1, in_channels, out_channels),
            [3, 3],
        ));
    }

    // The one shape in the corpus where Cout moves the CBUF bank split: at
    // 256x32 with Cin 32 the feature data wants 16 banks, so the two extra
    // banks the Cout 64 coefficients need come straight off it (11/1 at
    // Cout 16, 10/2 at Cout 64). Both sides are run so a regression in the
    // contention rule cannot pass by accident.
    for cout in [16u32, 64] {
        cases.push((Shape::with_out_channels(256, 32, 1, 32, cout), [3, 3]));
    }
    cases
}

/// Checks every case is buildable, without a device.
///
/// A tile whose input rows exceed what its data banks hold does not fault --
/// the hardware reads what it has and the tile loses its last rows. At large
/// Cout the coefficients take banks away from the feature data, so the
/// capacity has to be checked at every point rather than assumed.
#[test]
fn output_channel_matrix_tiles_fit_their_data_banks() {
    for (shape, kernels) in matrix() {
        let capacity = shape.max_tile_input_rows(kernels);
        let tiles = Tile::split(shape, kernels, shape.min_tiles(kernels));
        for tile in &tiles {
            assert!(
                tile.in_rows <= capacity,
                "Cin {} Cout {} {kernels:?}: tile reads {} input rows against a \
                 {capacity}-row capacity in {} data banks",
                shape.in_channels,
                shape.out_channels,
                tile.in_rows,
                shape.data_banks(kernels),
            );
        }
        let covered: u32 = tiles.iter().map(|tile| tile.out_rows).sum();
        assert_eq!(
            covered,
            shape.output_height(kernels),
            "Cin {} Cout {} {kernels:?}: tiles do not cover the output",
            shape.in_channels,
            shape.out_channels,
        );
    }
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn multi_output_channel_convs_run_on_npu() {
    let mut failures = Vec::new();
    for (shape, kernels) in matrix() {
        attempt(shape, kernels, &mut failures);
    }

    assert!(
        failures.is_empty(),
        "{} configuration(s) produced wrong output: {}",
        failures.len(),
        failures.join(", ")
    );
}
