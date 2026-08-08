//! Follow-up to `conv_dense_shared_buffer_dispatch_hw.rs`, which found a
//! clean, deterministic, 1-row shift on `run1`'s exact shape (Cin=3 dense,
//! Cout=256, 3x3, 228x228 physically-padded input): every output row of
//! `ConvPlan`'s sixth and last row tile (`out_first=189`) read back the
//! value of the input row *one above* the one it should have -- output row
//! `y` returned `input(y)` instead of `input(y+1)`, for every row of that
//! tile, deterministically. That test went through this crate's own
//! `submit`/`prep_bo` (not `rocket-hal-driver`), so the shift is not a
//! driver/dispatch bug -- it reproduces from the register program alone.
//!
//! # The lead this test checks
//!
//! `feature_grains(kernels, &tile.rows)` is `in_rows + weight_height +
//! pad_top`. For tile 5 specifically: `39 + 3 + 0 = 42`, three rows *more*
//! than the tile's own `in_rows` (39). Every other tile in the same plan has
//! the identical three-row margin (`feature_grains - in_rows == 3`
//! everywhere, since `weight_height=3` and `pad_top=0` throughout this
//! plan) but sits well before the end of the 228-row input buffer, so
//! reading three rows past its own window still lands inside real,
//! allocated, subsequent-tile data. Tile 5 is the only one whose window
//! reaches the *true* end of the buffer (`in_first + in_rows == 189 + 39 ==
//! 228 == the buffer's total height`) -- three rows of "margin" past it run
//! off the end of the allocation entirely.
//!
//! DESIGN_NOTES.md's existing `feature_grains` hardware probe
//! (`conv_grains_probe_hw.rs`) found the field did not gate correctness
//! anywhere from 1 up to 12 above the derived value, at 32x32-scale surface
//! shapes -- but every one of those 369 probe points had buffer runway past
//! its own tile, the same as every *other* tile in `run1`'s plan. Nothing
//! in this repository has probed `feature_grains` for a dense-layout tile
//! sitting flush against its buffer's true end, which is exactly what tile
//! 5 is and every earlier probe point was not.
//!
//! This test isolates that one variable. It reproduces tile 5's exact
//! geometry -- same `Shape`, same `Tile { out_first: 189, out_rows: 37,
//! in_first: 189, in_rows: 39, pad_top: 0 }`, same 228-row buffer with
//! nothing past row 228 -- as a single standalone job (not the full 6-tile
//! plan), and walks `feature_grains` explicitly via
//! `conv_2d_tile_with_grains` from below `in_rows` up past the derived
//! value, using the same one-hot center-tap weight and per-row-
//! distinguishable input `conv_dense_shared_buffer_dispatch_hw.rs` uses, so
//! a wrong result reports *which* row it actually read, not just
//! pass/fail. If capping `feature_grains` at `in_rows` (no margin at all)
//! fixes it while the derived value (42) still breaks it, that identifies
//! the mechanism precisely: reading `feature_grains` rows past a tile's own
//! `in_rows` is unsafe specifically when there is no real data past the
//! buffer to (harmlessly) over-read -- narrowing, not contradicting, the
//! existing probe's finding.
//!
//!   CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!     cargo test --target aarch64-unknown-linux-gnu --release \
//!       --test conv_dense_last_tile_grains_probe_hw --no-run
//!
//!   ./conv_dense_last_tile_grains_probe_hw-<hash> --ignored --nocapture

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    conv::{Buffers, Kernels, Shape, Tile, conv_2d_tile_with_grains, relocate},
    device::{Buffer, close_bo, fini_bo, prep_bo, submit},
    tensor_layout::{pack_hwcf_to_rocket_weights, rocket_weight_storage_size},
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const FP16_BYTES: usize = 2;
const FEATURE_ATOM_BYTES: usize = 16;
const PAGE_BYTES: usize = 4096;

// run1 / conv_dense_shared_buffer_dispatch_hw.rs's exact shape and tile 5.
const CIN: u32 = 3;
const COUT: u32 = 256;
const PADDED: u32 = 228; // buffer height -- tile 5's window ends exactly here.
const OUTPUT: u32 = 226;
const KERNEL: usize = 3;
const TILE: Tile = Tile {
    out_first: 189,
    out_rows: 37,
    in_first: 189,
    in_rows: 39,
    pad_top: 0,
};
const DERIVED_GRAINS: u32 = 42; // TILE.in_rows + weight_height(3) + pad_top(0).

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
    timed_out: bool,
    first_bad: Option<(usize, usize, f32, f32)>, // (y, x, want, got)
}

fn run(fd: i32, file: &std::fs::File, grains: u32) -> Probe {
    let kernels: Kernels = [KERNEL, KERNEL];
    let shape = Shape::with_out_channels(PADDED, PADDED, 1, CIN, COUT).with_padding([0, 0]);
    assert_eq!(shape.output_width(kernels), OUTPUT);
    assert_eq!(shape.output_height(kernels), OUTPUT);

    unsafe {
        let input_bytes = PADDED as usize * PADDED as usize * CIN as usize * FP16_BYTES;
        let buf_input = Buffer::new(fd, page_aligned_size(input_bytes), file);
        let input_words =
            std::slice::from_raw_parts_mut(buf_input.host_ptr as *mut u16, input_bytes / 2);
        for y in 0..PADDED as usize {
            let value = ROW_VALUES[y % ROW_VALUES.len()];
            for x in 0..PADDED as usize {
                for c in 0..CIN as usize {
                    input_words[(y * PADDED as usize + x) * CIN as usize + c] = value;
                }
            }
        }

        let dense_weight_elems = KERNEL * KERNEL * CIN as usize * COUT as usize;
        let mut dense_weights = vec![0u16; dense_weight_elems];
        let center_index = (1 * KERNEL + 1) * CIN as usize * COUT as usize + 0 * COUT as usize + 0;
        dense_weights[center_index] = 0x3c00;
        let dense_weight_bytes: Vec<u8> =
            dense_weights.iter().flat_map(|w| w.to_le_bytes()).collect();

        let packed_weight_bytes =
            rocket_weight_storage_size(KERNEL, KERNEL, CIN as usize, COUT as usize, FP16_BYTES)
                .expect("weight storage size");
        let buf_weights = Buffer::new(fd, page_aligned_size(packed_weight_bytes), file);
        ptr::write_bytes(buf_weights.host_ptr, 0, buf_weights.size);
        let weight_dst = std::slice::from_raw_parts_mut(buf_weights.host_ptr, packed_weight_bytes);
        pack_hwcf_to_rocket_weights(
            &dense_weight_bytes,
            KERNEL,
            KERNEL,
            CIN as usize,
            COUT as usize,
            FP16_BYTES,
            weight_dst,
        )
        .expect("weight packing");

        let buf_bias = Buffer::new(fd, page_aligned_size(shape.bs_buffer_bytes()), file);
        ptr::write_bytes(buf_bias.host_ptr, 0, buf_bias.size);

        // Full-canvas output scratch, same as the multi-tile test -- only
        // rows 189..225 get written by this single-tile job, and only those
        // are checked.
        let output_bytes = shape.output_scratch_bytes(kernels);
        let buf_output = Buffer::new(fd, page_aligned_size(output_bytes), file);
        ptr::write_bytes(buf_output.host_ptr, 0, buf_output.size);

        let mut commands = conv_2d_tile_with_grains(shape, kernels, &TILE, grains);
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
            timed_out: false,
            first_bad: None,
        };

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
            for y in TILE.out_first as usize..(TILE.out_first + TILE.out_rows) as usize {
                let want = f16_to_f32(ROW_VALUES[(y + 1) % ROW_VALUES.len()]);
                for x in 0..OUTPUT as usize {
                    let offset = (y * OUTPUT as usize + x) * FEATURE_ATOM_BYTES;
                    let got = f16_to_f32(u16::from_le_bytes([raw[offset], raw[offset + 1]]));
                    if got != want {
                        probe.mismatches += 1;
                        if probe.first_bad.is_none() {
                            probe.first_bad = Some((y, x, want, got));
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
            cmd_buf.handle,
        ] {
            let _ = close_bo(fd, handle);
        }

        probe
    }
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn last_tile_grains_sweep() {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();

    println!(
        "\n=== feature_grains sweep, tile 5's exact geometry (in_rows={}, derived={DERIVED_GRAINS}) ===",
        TILE.in_rows
    );
    // Below in_rows (never captured or probed anywhere), at in_rows exactly
    // (zero margin -- the value under test), through the derived value and
    // a few above it (the existing probe's own span at 32x32 scale).
    for grains in [30u32, 35, 38, 39, 40, 41, 42, 43, 45, 50] {
        let probe = run(fd, &file, grains);
        let margin = grains as i64 - TILE.in_rows as i64;
        if probe.timed_out {
            println!("grains={grains:3} (in_rows{margin:+3})  TIMEOUT / did not complete");
        } else if probe.mismatches == 0 {
            println!("grains={grains:3} (in_rows{margin:+3})  ok  (0 mismatches)");
        } else {
            let (y, x, want, got) = probe.first_bad.unwrap();
            println!(
                "grains={grains:3} (in_rows{margin:+3})  FAIL  {} mismatches  first_bad=[y={y}, x={x}] want={want} got={got}",
                probe.mismatches
            );
        }
    }
    println!(
        "\nIf grains<=in_rows (margin <= 0) all pass and grains>in_rows (margin > 0, including \
         the derived value {DERIVED_GRAINS}) all fail the same way conv_dense_shared_buffer_dispatch_hw.rs \
         did, that confirms: feature_grains reading past a tile's own in_rows is unsafe when the tile \
         sits flush against the true end of its input buffer, and capping it at in_rows there is a fix."
    );
}
