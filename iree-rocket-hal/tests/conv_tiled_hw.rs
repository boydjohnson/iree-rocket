//! Hardware-in-the-loop test for height-tiled convolution submitted as
//! several independent jobs in one SUBMIT ioctl.
//!
//! This test is ignored on the development host because it needs the RK3588
//! NPU device. Cross-compile it, copy the printed test binary to the board,
//! and run it there:
//!
//!   CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!     cargo test --target aarch64-unknown-linux-gnu --release \
//!       --test conv_tiled_hw --no-run
//!
//!   ./conv_tiled_hw-<hash> --ignored --nocapture
//!
//! It runs the same `32x32x3 -> 32x32x8` fp16 convolution as
//! `conv_vendor_reference_hw`, with the same uniform inputs and the same
//! exact expected values, but splits the output rows across 1, 2, and 3
//! tiles. Each tile is a separate `drm_rocket_job`; all of them go in one
//! `drm_rocket_submit`. The three splits are the vendor's own: 32, 16+16,
//! and 11+11+10.
//!
//! # What a pass does and does not prove
//!
//! The exact-value assertions confirm the tile-dependent register formulas
//! derived from the cross-group capture diff -- `feature_grains`, the split
//! `pad_top`/`pad_left` nibbles, and the input/output offsets in particular.
//!
//! It does **not** prove the jobs ran on different cores. `drm_rocket_job`
//! has no core field; placement is entirely the kernel scheduler's and is
//! not observable from userspace. A pass is equally consistent with all
//! three jobs running sequentially on one core. Two things worth watching
//! for with the out-of-tree kprobe tracer while this runs:
//!
//! - per-core operation counts, to see whether the scheduler spreads jobs
//! - whether jobs writing the same output BO get an implicit write-after-
//!   write dependency and serialise despite covering disjoint rows
//!
//! # Where a wrong halo shows up
//!
//! The tiles write disjoint output rows, but for 3x3 each tile's first and
//! last output row depends on reading a halo row from its neighbour. A bad
//! `feature_grains` or feature-base offset therefore fails at tile boundary
//! rows specifically -- rows 15/16 for the 2-tile split, rows 10/11 and
//! 21/22 for the 3-tile split -- so the failure report below groups
//! mismatches by row and marks boundary rows.

use std::{collections::BTreeMap, fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    builders::{
        RegCmd, RegisterMeta,
        cna::{CnaDcompAddr0, CnaFeatureDataAddr},
        dpu::DpuDstBaseAddr,
        dpu_rdma::DpuRdmaBsBaseAddr,
    },
    conv::{Kernels, Tile, conv_2d_tile},
    device::{Buffer, JobDesc, close_bo, fini_bo, prep_bo, submit_jobs},
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const WIDTH: usize = 32;
const HEIGHT: usize = 32;
const INPUT_CHANNELS: usize = 3;
const WEIGHT_INPUT_CHANNELS: usize = 8;
const OUTPUT_CHANNELS: usize = 8;
const FP16_BYTES: usize = 2;
const FEATURE_ATOM_BYTES: usize = 16;
const PAGE_BYTES: usize = 4096;

const INPUT_BYTES: usize = WIDTH * HEIGHT * INPUT_CHANNELS * FP16_BYTES;
const OUTPUT_BYTES: usize = WIDTH * HEIGHT * FEATURE_ATOM_BYTES * 2;

const FP16_ONE: u16 = 0x3c00;

fn page_aligned_size(size: usize) -> usize {
    size.div_ceil(PAGE_BYTES) * PAGE_BYTES
}

fn decode_identity(command: &RegCmd) -> (u32, u32) {
    ((command.0 >> 48) as u32, command.0 as u32 & 0xffff)
}

/// Adds `address` to the tile offset a program already carries in register
/// `R`, matching `conv_vendor_reference_hw`'s relocation.
fn relocate<R: RegisterMeta>(commands: &mut [RegCmd], address: u32) {
    assert_eq!(
        address & 0xf,
        0,
        "NPU DMA address for register {:#x}:{:#x} is not 16-byte aligned",
        R::DOMAIN,
        R::OFFSET
    );

    let matches: Vec<_> = commands
        .iter()
        .enumerate()
        .filter_map(|(index, command)| {
            (decode_identity(command) == (R::DOMAIN, R::OFFSET)).then_some(index)
        })
        .collect();
    assert_eq!(
        matches.len(),
        1,
        "expected exactly one {:#x}:{:#x} relocation, found {matches:?}",
        R::DOMAIN,
        R::OFFSET
    );

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
        _ => unreachable!("conv_2d_tile rejects kernels other than 1x1 and 3x3"),
    }
}

/// Expected value is a property of the whole convolution, not of the tiling:
/// splitting the output rows must not change any value.
fn expected_output(kernels: Kernels, y: usize, x: usize) -> f32 {
    (INPUT_CHANNELS * valid_taps(y, HEIGHT, kernels[0]) * valid_taps(x, WIDTH, kernels[1])) as f32
}

/// Runs the convolution split across `tiles` jobs and returns the decoded
/// `32x32x8` output.
fn run_tiled_conv(kernels: Kernels, tiles: u32) -> Vec<f32> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let buf_input = Buffer::new(fd, page_aligned_size(INPUT_BYTES), &file);
        ptr::write_bytes(buf_input.host_ptr, 0, buf_input.size);
        let input = std::slice::from_raw_parts_mut(
            buf_input.host_ptr as *mut u16,
            INPUT_BYTES / FP16_BYTES,
        );
        input.fill(FP16_ONE);

        let weight_bytes =
            kernels[0] * kernels[1] * WEIGHT_INPUT_CHANNELS * OUTPUT_CHANNELS * FP16_BYTES;
        let buf_weights = Buffer::new(fd, page_aligned_size(weight_bytes), &file);
        ptr::write_bytes(buf_weights.host_ptr, 0, buf_weights.size);
        let weights =
            std::slice::from_raw_parts_mut(buf_weights.host_ptr as *mut u16, weight_bytes / 2);
        weights.fill(FP16_ONE);

        let buf_bias = Buffer::new(fd, PAGE_BYTES, &file);
        ptr::write_bytes(buf_bias.host_ptr, 0, buf_bias.size);

        let buf_output = Buffer::new(fd, OUTPUT_BYTES, &file);
        ptr::write_bytes(buf_output.host_ptr, 0, buf_output.size);

        // One regcmd buffer per tile. All tiles share the four data buffers;
        // only the base addresses inside each program differ.
        let split = Tile::split(kernels, tiles);
        let mut command_buffers = Vec::with_capacity(split.len());
        for tile in &split {
            let mut commands = conv_2d_tile(kernels, tile);
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

        fini_bo(fd, buf_input.handle).expect("failed to sync input BO for the NPU");
        fini_bo(fd, buf_weights.handle).expect("failed to sync weight BO for the NPU");
        fini_bo(fd, buf_bias.handle).expect("failed to sync bias BO for the NPU");
        fini_bo(fd, buf_output.handle).expect("failed to sync output BO for the NPU");
        for (buffer, _) in &command_buffers {
            fini_bo(fd, buffer.handle).expect("failed to sync regcmd BO for the NPU");
        }

        // Handle lists must outlive the ioctl, so they are materialised
        // before any JobDesc borrows them.
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
            panic!("{kernels:?} {tiles}-tile convolution SUBMIT ioctl failed: {error}")
        });

        // Every job lists the output BO in out_bo_handles, so waiting on it
        // waits for all of them.
        prep_bo(fd, buf_output.handle, 2_000_000_000).unwrap_or_else(|error| {
            panic!(
                "{kernels:?} {tiles}-tile convolution did not complete within two seconds: {error}"
            )
        });

        let raw = std::slice::from_raw_parts(buf_output.host_ptr, OUTPUT_BYTES);
        let mut output = Vec::with_capacity(WIDTH * HEIGHT * OUTPUT_CHANNELS);
        for pixel in 0..WIDTH * HEIGHT {
            for channel in 0..OUTPUT_CHANNELS {
                let offset = pixel * FEATURE_ATOM_BYTES + channel * FP16_BYTES;
                output.push(f16_to_f32(u16::from_le_bytes([
                    raw[offset],
                    raw[offset + 1],
                ])));
            }
        }

        close_bo(fd, buf_input.handle).expect("failed to close input BO");
        close_bo(fd, buf_weights.handle).expect("failed to close weight BO");
        close_bo(fd, buf_bias.handle).expect("failed to close bias BO");
        close_bo(fd, buf_output.handle).expect("failed to close output BO");
        for (buffer, _) in &command_buffers {
            close_bo(fd, buffer.handle).expect("failed to close regcmd BO");
        }

        output
    }
}

/// Output rows that begin a tile other than the first -- where a wrong halo
/// or feature-base offset shows up.
fn boundary_rows(kernels: Kernels, tiles: u32) -> Vec<usize> {
    Tile::split(kernels, tiles)
        .iter()
        .skip(1)
        .flat_map(|tile| {
            let first = tile.out_first as usize;
            [first.saturating_sub(1), first]
        })
        .collect()
}

fn check(kernels: Kernels, tiles: u32) {
    let actual = run_tiled_conv(kernels, tiles);
    let boundaries = boundary_rows(kernels, tiles);

    let mut per_row: BTreeMap<usize, usize> = BTreeMap::new();
    let mut samples = Vec::new();
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let expected = expected_output(kernels, y, x);
            for channel in 0..OUTPUT_CHANNELS {
                let index = (y * WIDTH + x) * OUTPUT_CHANNELS + channel;
                if actual[index] != expected {
                    *per_row.entry(y).or_default() += 1;
                    if samples.len() < 16 {
                        samples.push(format!(
                            "[{y}, {x}, {channel}]: expected {expected}, got {}",
                            actual[index]
                        ));
                    }
                }
            }
        }
    }

    if per_row.is_empty() {
        println!(
            "{kernels:?} {tiles}-tile: all {} values correct",
            actual.len()
        );
        return;
    }

    let rows: Vec<String> = per_row
        .iter()
        .map(|(row, count)| {
            let mark = if boundaries.contains(row) {
                " <- tile boundary"
            } else {
                ""
            };
            format!("  row {row:2}: {count} mismatches{mark}")
        })
        .collect();
    panic!(
        "{kernels:?} {tiles}-tile convolution had {} mismatches\n\
         tile boundary rows: {boundaries:?}\n\
         mismatches by row:\n{}\nfirst mismatches:\n{}",
        per_row.values().sum::<usize>(),
        rows.join("\n"),
        samples.join("\n")
    );
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn tiled_convs_match_untiled_results() {
    // tiles = 1 reproduces the whole-image program already covered by
    // conv_vendor_reference_hw, and is included so a failure there separates
    // "tiling is wrong" from "multi-job submission is wrong".
    for tiles in [1, 2, 3] {
        for kernels in [[1, 1], [3, 3]] {
            check(kernels, tiles);
        }
    }
}
