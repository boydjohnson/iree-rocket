#![cfg(feature = "hardware-characterization")]

//! Hardware characterization. Follow-up to
//! `conv_dense_alignment_channel_sweep_hw.rs`, which broke the
//! "byte offset `>= 8` is unsafe" model: `Cin=1` started failing at offset 4
//! (not 8) with a corrupted-column count that grew *linearly with
//! `in_first`* (1, 2, 3, 4, 5, 6 columns at `in_first` 2 through 7, no sign
//! of levelling off), while `Cin=4` did not fail at all at the one nonzero
//! offset (8) it could reach.
//!
//! Every earlier probe in this line of tests only ever swept `in_first`
//! across a small range (0..7 or 0..3), because that was enough to cover
//! every reachable byte offset once. It was never enough to see whether
//! corruption severity keeps growing with `in_first` itself, at any `Cin`
//! including 3 -- the "1 or 2 columns, then apparently flat" reading of the
//! original `Cin=3` sweep might just be an artifact of not having looked
//! past `in_first=7`.
//!
//! # The hypothesis this checks
//!
//! A fixed byte-scale drift that *accumulates with `in_first`* (not a
//! per-fetch constant tied to the byte offset alone) would explain every
//! observation so far: it becomes a visible wrong *column* once it exceeds
//! one dense pixel's width (`Cin * 2` bytes at fp16) -- sooner, in terms of
//! `in_first`, for narrower `Cin` (2 bytes/pixel at `Cin=1`) than wider
//! (8 bytes/pixel at `Cin=4`), which would explain both why `Cin=1` grows
//! fastest and why `Cin=4` might simply need a larger `in_first` than 1 to
//! ever show anything.
//!
//! This sweeps `in_first` far past every earlier probe's range (0..24,
//! against 0..7 or 0..3 before) at all four dense `Cin` values, holding
//! each `Cin`'s width fixed (the same ones already used, so the results are
//! directly comparable to what came before), and reports the corrupted-
//! column count as a function of `in_first` rather than of offset. If
//! `Cin=3`'s growth also continues past where the first sweep stopped
//! looking, or `Cin=4` starts failing once `in_first` is large enough, that
//! confirms the accumulating-drift reading over the original fixed-offset-
//! threshold one.
//!
//!   CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!     cargo test --target aarch64-unknown-linux-gnu --release \
//!       --test conv_dense_alignment_in_first_growth_hw --no-run
//!
//!   ./conv_dense_alignment_in_first_growth_hw-<hash> --ignored --nocapture

use std::{collections::BTreeSet, fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    conv::{Buffers, Kernels, Shape, Tile, conv_2d_tile_with_grains, relocate},
    device::{Buffer, close_bo, fini_bo, prep_bo, submit},
    tensor_layout::{pack_hwcf_to_rocket_weights, rocket_weight_storage_size},
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const FP16_BYTES: usize = 2;
const FEATURE_ATOM_BYTES: usize = 16;
const PAGE_BYTES: usize = 4096;

const COUT: u32 = 256;
const KERNEL: usize = 3;
const OUT_ROWS: u32 = 8;
const IN_ROWS: u32 = 10; // OUT_ROWS + KERNEL - 1, padding=[0,0].
const MAX_IN_FIRST: u32 = 24;
const HEIGHT: u32 = MAX_IN_FIRST + IN_ROWS + 10; // generous runway throughout.

const ROW_VALUES: [u16; 3] = [0x3c00, 0x4000, 0x4200];

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

struct Probe {
    mismatches: usize,
    distinct_x: BTreeSet<usize>,
    timed_out: bool,
}

fn run(fd: i32, file: &std::fs::File, cin: u32, width: u32, in_first: u32) -> Probe {
    let kernels: Kernels = [KERNEL, KERNEL];
    let shape = Shape::with_out_channels(width, HEIGHT, 1, cin, COUT).with_padding([0, 0]);
    let tile = Tile {
        out_first: in_first,
        out_rows: OUT_ROWS,
        in_first,
        in_rows: IN_ROWS,
        pad_top: 0,
    };
    let grains = IN_ROWS + KERNEL as u32;
    assert!(shape.output_height(kernels) >= tile.out_first + tile.out_rows);
    assert!(HEIGHT >= in_first + IN_ROWS);

    unsafe {
        let input_bytes = width as usize * HEIGHT as usize * cin as usize * FP16_BYTES;
        let buf_input = Buffer::new(fd, page_aligned_size(input_bytes), file);
        let input_words =
            std::slice::from_raw_parts_mut(buf_input.host_ptr as *mut u16, input_bytes / 2);
        for y in 0..HEIGHT as usize {
            let value = ROW_VALUES[y % ROW_VALUES.len()];
            for x in 0..width as usize {
                for c in 0..cin as usize {
                    input_words[(y * width as usize + x) * cin as usize + c] = value;
                }
            }
        }

        let dense_weight_elems = KERNEL * KERNEL * cin as usize * COUT as usize;
        let mut dense_weights = vec![0u16; dense_weight_elems];
        let center_index = (1 * KERNEL + 1) * cin as usize * COUT as usize + 0 * COUT as usize + 0;
        dense_weights[center_index] = 0x3c00;
        let dense_weight_bytes: Vec<u8> =
            dense_weights.iter().flat_map(|w| w.to_le_bytes()).collect();

        let packed_weight_bytes =
            rocket_weight_storage_size(KERNEL, KERNEL, cin as usize, COUT as usize, FP16_BYTES)
                .expect("weight storage size");
        let buf_weights = Buffer::new(fd, page_aligned_size(packed_weight_bytes), file);
        ptr::write_bytes(buf_weights.host_ptr, 0, buf_weights.size);
        let weight_dst = std::slice::from_raw_parts_mut(buf_weights.host_ptr, packed_weight_bytes);
        pack_hwcf_to_rocket_weights(
            &dense_weight_bytes,
            KERNEL,
            KERNEL,
            cin as usize,
            COUT as usize,
            FP16_BYTES,
            weight_dst,
        )
        .expect("weight packing");

        let buf_bias = Buffer::new(fd, page_aligned_size(shape.bs_buffer_bytes()), file);
        ptr::write_bytes(buf_bias.host_ptr, 0, buf_bias.size);

        let output_bytes = shape.output_scratch_bytes(kernels);
        let buf_output = Buffer::new(fd, page_aligned_size(output_bytes), file);
        ptr::write_bytes(buf_output.host_ptr, 0, buf_output.size);

        let mut commands = conv_2d_tile_with_grains(shape, kernels, &tile, grains);
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
        let cmd_buf = Buffer::new(fd, page_aligned_size(command_bytes), file);
        ptr::write_bytes(cmd_buf.host_ptr, 0, cmd_buf.size);
        let words = std::slice::from_raw_parts_mut(cmd_buf.host_ptr as *mut u64, commands.len());
        for (destination, command) in words.iter_mut().zip(&commands) {
            *destination = command.0;
        }

        for handle in [
            buf_input.handle,
            buf_weights.handle,
            buf_bias.handle,
            buf_output.handle,
            cmd_buf.handle,
        ] {
            fini_bo(fd, handle).expect("failed to sync BO for the NPU");
        }

        let in_handles = [
            cmd_buf.handle,
            buf_input.handle,
            buf_weights.handle,
            buf_bias.handle,
        ];
        let out_handles = [buf_output.handle];

        let mut probe = Probe {
            mismatches: 0,
            distinct_x: BTreeSet::new(),
            timed_out: false,
        };

        let out_width = shape.output_width(kernels) as usize;
        if submit(
            fd,
            cmd_buf.dma_address,
            commands.len() as u32,
            &in_handles,
            &out_handles,
        )
        .is_err()
            || prep_bo(fd, buf_output.handle, 5_000_000_000).is_err()
        {
            probe.timed_out = true;
        } else {
            let raw = std::slice::from_raw_parts(buf_output.host_ptr, output_bytes);
            for y in tile.out_first as usize..(tile.out_first + tile.out_rows) as usize {
                let want = f16_to_f32(ROW_VALUES[(y + 1) % ROW_VALUES.len()]);
                for x in 0..out_width {
                    let offset = (y * out_width + x) * FEATURE_ATOM_BYTES;
                    let got = f16_to_f32(u16::from_le_bytes([raw[offset], raw[offset + 1]]));
                    if got != want {
                        probe.mismatches += 1;
                        probe.distinct_x.insert(x);
                    }
                }
            }
        }

        for handle in [
            buf_input.handle,
            buf_weights.handle,
            buf_bias.handle,
            buf_output.handle,
            cmd_buf.handle,
        ] {
            let _ = close_bo(fd, handle);
        }

        probe
    }
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn corrupted_columns_vs_in_first() {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();

    // Same (Cin, width) pairs the previous sweep used, so results line up
    // directly; only the in_first range is new.
    let cases: [(&str, u32, u32); 4] = [
        ("Cin=1 w225", 1, 225),
        ("Cin=2 w225", 2, 225),
        ("Cin=3 w227", 3, 227),
        ("Cin=4 w225", 4, 225),
    ];

    for (label, cin, width) in cases {
        let stride = width * cin * FP16_BYTES as u32;
        println!(
            "\n=== {label}: input_row_stride={stride} mod16={} -- in_first 0..{MAX_IN_FIRST} ===",
            stride % 16
        );
        for in_first in 0..=MAX_IN_FIRST {
            let offset = (in_first * stride) % 16;
            let probe = run(fd, &file, cin, width, in_first);
            let status = if probe.timed_out {
                "TIMEOUT".to_string()
            } else if probe.mismatches == 0 {
                "ok".to_string()
            } else {
                format!(
                    "FAIL {} mismatches, {} distinct columns {:?}",
                    probe.mismatches,
                    probe.distinct_x.len(),
                    probe.distinct_x
                )
            };
            println!("  in_first={in_first:2} offset={offset:2}  {status}");
        }
    }
    println!(
        "\nLook for: does Cin=3's column count keep growing past where the first sweep stopped \
         (in_first=7)? Does Cin=4 ever fail, and at what in_first? Does Cin=1's growth continue \
         linearly, accelerate, or saturate? If corrupted-column count tracks in_first (scaled \
         inversely by Cin's bytes-per-pixel) rather than the byte offset alone, that confirms an \
         accumulating-drift mechanism over the original fixed-per-fetch-offset one."
    );
}
