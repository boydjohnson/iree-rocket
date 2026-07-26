//! Measures which `CNA_CONV_CON2.feature_grains` values the hardware accepts.
//!
//! This test is ignored on the development host because it needs the RK3588
//! NPU device. Cross-compile it, copy the printed test binary to the board,
//! and run it there:
//!
//!   CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!     cargo test --target aarch64-unknown-linux-gnu --release \
//!       --test conv_grains_probe_hw --no-run
//!
//!   ./conv_grains_probe_hw-<hash> --ignored --nocapture
//!
//! # Why probe instead of deriving
//!
//! A shape sweep of 49 vendor captures (297 convolution programs) failed to
//! yield a formula. The vendor's value equals `in_rows + weight_height +
//! pad_top` in 63% of programs, exactly `in_rows` in 28%, and drops *below*
//! `in_rows` in 6%. No register field in the corpus separates those cases,
//! so the choice appears to come from compiler allocator state that never
//! reaches the register program.
//!
//! The runtime does not need the vendor's value, though -- it needs a value
//! that works, computable from a dynamic shape. The TRM calls its own
//! formula "suggested", implying a range of valid settings. This test
//! measures that range directly: for each tile geometry it walks
//! `feature_grains` across a span and records, for each value, whether the
//! job completed and whether every output element was correct.
//!
//! # Reading the output
//!
//! Each row is one tile geometry, with one character per probed value:
//!
//!   `.` correct output      `X` wrong output      `T` job did not complete
//!
//! `D` marks the value this crate derives and `V` the value the vendor
//! programs for that tile, so the printed range shows whether both sit
//! inside the working span and how much margin each has.
//!
//! # Safety
//!
//! A rejected value may leave a job that never signals. Every probe point
//! gets its own device handle and its own buffers so a stuck job cannot
//! corrupt a later measurement, and the completion wait is short. If the run
//! wedges the NPU, the kernel's own reset path is what recovers it -- watch
//! the tracer's reset counter while this runs.

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    builders::{
        RegCmd, RegisterMeta,
        cna::{CnaDcompAddr0, CnaFeatureDataAddr},
        dpu::DpuDstBaseAddr,
        dpu_rdma::DpuRdmaBsBaseAddr,
    },
    conv::{Kernels, Shape, Tile, conv_2d_tile_with_grains, feature_grains},
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

/// Short on purpose: a rejected value should cost a fraction of a second,
/// not the two seconds a real conv is given.
const TIMEOUT_NS: u64 = 250_000_000;

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

#[derive(PartialEq, Clone, Copy)]
enum Outcome {
    Correct,
    Wrong,
    Timeout,
}

impl Outcome {
    fn mark(self) -> char {
        match self {
            Outcome::Correct => '.',
            Outcome::Wrong => 'X',
            Outcome::Timeout => 'T',
        }
    }
}

/// Runs one tile with an explicit `grains`, checking only the rows this tile
/// produces -- neighbouring rows belong to other tiles and are left zero.
fn probe(kernels: Kernels, tile: &Tile, grains: u32) -> Outcome {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let buf_input = Buffer::new(fd, page_aligned_size(INPUT_BYTES), &file);
        ptr::write_bytes(buf_input.host_ptr, 0, buf_input.size);
        std::slice::from_raw_parts_mut(buf_input.host_ptr as *mut u16, INPUT_BYTES / FP16_BYTES)
            .fill(FP16_ONE);

        let weight_bytes =
            kernels[0] * kernels[1] * WEIGHT_INPUT_CHANNELS * OUTPUT_CHANNELS * FP16_BYTES;
        let buf_weights = Buffer::new(fd, page_aligned_size(weight_bytes), &file);
        ptr::write_bytes(buf_weights.host_ptr, 0, buf_weights.size);
        std::slice::from_raw_parts_mut(buf_weights.host_ptr as *mut u16, weight_bytes / 2)
            .fill(FP16_ONE);

        let buf_bias = Buffer::new(fd, PAGE_BYTES, &file);
        ptr::write_bytes(buf_bias.host_ptr, 0, buf_bias.size);
        let buf_output = Buffer::new(fd, OUTPUT_BYTES, &file);
        ptr::write_bytes(buf_output.host_ptr, 0, buf_output.size);

        let mut commands = conv_2d_tile_with_grains(Shape::CAPTURED, kernels, tile, grains);
        relocate::<CnaFeatureDataAddr>(&mut commands, buf_input.dma_address);
        relocate::<CnaDcompAddr0>(&mut commands, buf_weights.dma_address);
        relocate::<DpuRdmaBsBaseAddr>(&mut commands, buf_bias.dma_address);
        relocate::<DpuDstBaseAddr>(&mut commands, buf_output.dma_address);

        let command_bytes = commands.len() * mem::size_of::<u64>();
        let buf_commands = Buffer::new(fd, page_aligned_size(command_bytes), &file);
        ptr::write_bytes(buf_commands.host_ptr, 0, buf_commands.size);
        let words =
            std::slice::from_raw_parts_mut(buf_commands.host_ptr as *mut u64, commands.len());
        for (destination, command) in words.iter_mut().zip(&commands) {
            *destination = command.0;
        }

        for handle in [
            buf_input.handle,
            buf_weights.handle,
            buf_bias.handle,
            buf_output.handle,
            buf_commands.handle,
        ] {
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

        // A value the hardware rejects can fail either at submit or by never
        // signalling completion; both mean "this value does not work".
        let accepted =
            submit_jobs(fd, &jobs).is_ok() && prep_bo(fd, buf_output.handle, TIMEOUT_NS).is_ok();
        let outcome = if !accepted {
            Outcome::Timeout
        } else {
            let raw = std::slice::from_raw_parts(buf_output.host_ptr, OUTPUT_BYTES);
            let mut outcome = Outcome::Correct;
            'rows: for row in 0..tile.out_rows as usize {
                let y = tile.out_first as usize + row;
                for x in 0..WIDTH {
                    let expected = (INPUT_CHANNELS
                        * valid_taps(y, HEIGHT, kernels[0])
                        * valid_taps(x, WIDTH, kernels[1]))
                        as f32;
                    for channel in 0..OUTPUT_CHANNELS {
                        let offset = (y * WIDTH + x) * FEATURE_ATOM_BYTES + channel * FP16_BYTES;
                        let actual = f16_to_f32(u16::from_le_bytes([raw[offset], raw[offset + 1]]));
                        if actual != expected {
                            outcome = Outcome::Wrong;
                            break 'rows;
                        }
                    }
                }
            }
            outcome
        };

        for handle in [
            buf_input.handle,
            buf_weights.handle,
            buf_bias.handle,
            buf_output.handle,
            buf_commands.handle,
        ] {
            let _ = close_bo(fd, handle);
        }
        outcome
    }
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board; \
            diagnostic only, read the printed table"]
fn feature_grains_working_range() {
    // The vendor's own values for the captured 32x32 plans, in tile order:
    // groups 1 / 2-3 / 4-6 of conv.rknn and conv-1x1-oc-8.rknn.
    const VENDOR_3X3: [[u32; 3]; 3] = [[36, 0, 0], [21, 20, 0], [16, 16, 14]];
    const VENDOR_1X1: [[u32; 3]; 3] = [[33, 0, 0], [17, 17, 0], [12, 12, 11]];

    println!("\n32x32x3 -> 32x32x8 fp16, probing CNA_CONV_CON2.feature_grains");
    println!("  . correct    X wrong output    T did not complete");
    println!("  D derived value    V vendor value\n");

    for kernels in [[1usize, 1], [3, 3]] {
        let vendor = if kernels[0] == 3 {
            VENDOR_3X3
        } else {
            VENDOR_1X1
        };
        for tiles in [1u32, 2, 3] {
            for (index, tile) in Tile::split(Shape::CAPTURED, kernels, tiles)
                .iter()
                .enumerate()
            {
                let derived = feature_grains(kernels, tile);
                let expected = vendor[tiles as usize - 1][index];
                let high = derived + 12;

                let mut marks = String::new();
                let (mut first_ok, mut last_ok) = (None, None);
                for grains in 1..=high {
                    let outcome = probe(kernels, tile, grains);
                    marks.push(outcome.mark());
                    if outcome == Outcome::Correct {
                        first_ok.get_or_insert(grains);
                        last_ok = Some(grains);
                    }
                }

                let annotate = |value: u32| {
                    if value == 0 || value > high {
                        String::from("-")
                    } else {
                        format!("{value}")
                    }
                };
                println!(
                    "{:?} {tiles}-tile[{index}] in_rows={:>2} pad_top={} \
                     D={:>2} V={:>2}  1..{high}: {marks}",
                    kernels,
                    tile.in_rows,
                    tile.pad_top,
                    annotate(derived),
                    annotate(expected),
                );
                match (first_ok, last_ok) {
                    (Some(low), Some(high_ok)) => println!(
                        "        working range {low}..{high_ok}  \
                         derived {}  vendor {}",
                        if (low..=high_ok).contains(&derived) {
                            "inside"
                        } else {
                            "OUTSIDE"
                        },
                        if expected == 0 {
                            "n/a"
                        } else if (low..=high_ok).contains(&expected) {
                            "inside"
                        } else {
                            "OUTSIDE"
                        },
                    ),
                    _ => println!("        no value produced correct output"),
                }
            }
        }
    }

    println!(
        "\nIf every geometry has a wide working range containing both D and V,\n\
         the runtime can compute feature_grains from a dynamic shape without\n\
         reproducing the vendor's choice.\n"
    );
}
