//! Follow-up to `conv_dense_last_tile_grains_probe_hw.rs`, which ruled out
//! `feature_grains` as the cause of the defect
//! `conv_dense_shared_buffer_dispatch_hw.rs` found on `run1`'s shape (Cin=3
//! dense, Cout=256, 3x3): every value from below `in_rows` to well above
//! the derived one failed identically.
//!
//! Re-reading that failure's mismatch count corrects an earlier
//! misdiagnosis in this file's own history: 37 mismatches across a 37-row x
//! 226-column tile is not a whole-row shift, it is **one wrong pixel per
//! row, at column 0**, every other column in every one of tile 5's rows
//! reading back correct. The `[y, x=0]` value observed is the previous
//! row's value.
//!
//! # The new lead
//!
//! `CNA_FEATURE_DATA_ADDR` (`Tile::input_offset`) is `in_first *
//! input_row_stride`. For this shape `input_row_stride = 228 * 3 * 2 =
//! 1368`, which is not a multiple of 16 -- so the byte address is 16-byte
//! aligned only when `in_first` is even (`1368 mod 16 == 8`, so an odd
//! `in_first` lands the feature base 8 bytes off a 16-byte boundary).
//! Checked directly: tile 5's `in_first` (189) is the *only* odd one among
//! this plan's six tiles (0, 38, 76, 114, 152, 189) -- every other tile's
//! feature base is exactly 16-byte aligned.
//!
//! Dense mode sets `CNA_CONV_CON1.nonalign_dma = 1` unconditionally
//! (`conv_2d_tile_program`), which exists specifically to let the CNA start
//! a dense fetch from a non-16-byte-aligned address -- so a single wrong
//! *leading* pixel per row, with everything after it correct, is exactly
//! the shape of bug a resync-to-pixel-0 compensation getting the leading
//! partial atom wrong would produce.
//!
//! # What this test isolates
//!
//! `conv_dense_last_tile_grains_probe_hw.rs` held `in_first=189` (odd) and
//! `height=228` (the tile's window flush against the buffer's true end)
//! both fixed while sweeping `feature_grains`. This test instead holds
//! `feature_grains` at its ordinary derived value and varies `in_first`'s
//! parity and whether the tile's window is flush against its buffer's true
//! end, independently, as a small factorial:
//!
//! | case | in_first | parity | flush against buffer end |
//! |---|---:|---|---|
//! | A | 189 | odd  | yes (228 = 189+39, same as `run1`) |
//! | B |  39 | odd  | no  (buffer is 300 rows) |
//! | C |  40 | even | yes (79 = 40+39) |
//! | D | 190 | even | no  (buffer is 300 rows) |
//!
//! If B fails the same single-leading-pixel way A does, and C passes same
//! as D, parity alone is the cause, independent of flushness -- which would
//! also mean the buffer-end/`feature_grains` framing in
//! `conv_dense_last_tile_grains_probe_hw.rs`'s doc comment was the wrong
//! half of a coincidence (tile 5 in `run1`'s real plan happens to be both
//! odd *and* the flush one, since row tiling there always lands on an
//! even split except for the remainder).
//!
//! For each case this reports every mismatch's column, not just the first,
//! so "only column 0" can be confirmed or refuted directly rather than
//! assumed from the first-bad sample the way the earlier tests did.
//!
//!   CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!     cargo test --target aarch64-unknown-linux-gnu --release \
//!       --test conv_dense_odd_in_first_probe_hw --no-run
//!
//!   ./conv_dense_odd_in_first_probe_hw-<hash> --ignored --nocapture

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

const CIN: u32 = 3;
const COUT: u32 = 256;
const WIDTH: u32 = 228;
const KERNEL: usize = 3;
const OUT_ROWS: u32 = 37;
const IN_ROWS: u32 = 39; // OUT_ROWS + KERNEL - 1, padding=[0,0].

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
    distinct_y: BTreeSet<usize>,
    samples: Vec<String>,
    timed_out: bool,
}

/// `in_first == out_first` throughout (padding=[0,0]); `height` is the
/// buffer's total row count, i.e. how much runway exists past the tile.
fn run(fd: i32, file: &std::fs::File, in_first: u32, height: u32) -> Probe {
    let kernels: Kernels = [KERNEL, KERNEL];
    let shape = Shape::with_out_channels(WIDTH, height, 1, CIN, COUT).with_padding([0, 0]);
    let tile = Tile {
        out_first: in_first,
        out_rows: OUT_ROWS,
        in_first,
        in_rows: IN_ROWS,
        pad_top: 0,
    };
    let grains = IN_ROWS + KERNEL as u32 + 0; // ordinary derived value.
    assert!(
        shape.output_height(kernels) >= tile.out_first + tile.out_rows,
        "case is malformed: tile's output rows fall outside the {}-row output",
        shape.output_height(kernels)
    );
    assert!(
        height >= in_first + IN_ROWS,
        "case is malformed: tile's input rows fall outside the {height}-row buffer"
    );

    unsafe {
        let input_bytes = WIDTH as usize * height as usize * CIN as usize * FP16_BYTES;
        let buf_input = Buffer::new(fd, page_aligned_size(input_bytes), file);
        let input_words =
            std::slice::from_raw_parts_mut(buf_input.host_ptr as *mut u16, input_bytes / 2);
        for y in 0..height as usize {
            let value = ROW_VALUES[y % ROW_VALUES.len()];
            for x in 0..WIDTH as usize {
                for c in 0..CIN as usize {
                    input_words[(y * WIDTH as usize + x) * CIN as usize + c] = value;
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
            distinct_y: BTreeSet::new(),
            samples: Vec::new(),
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
                        probe.distinct_y.insert(y);
                        if probe.samples.len() < 8 {
                            probe
                                .samples
                                .push(format!("[y={y}, x={x}] want={want} got={got}"));
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
fn odd_in_first_factorial() {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();

    let cases: [(&str, u32, u32); 4] = [
        ("A odd+flush  (run1's tile 5)", 189, 228),
        ("B odd+runway", 39, 300),
        ("C even+flush", 40, 79),
        ("D even+runway", 190, 300),
    ];

    println!("\n=== odd in_first vs flush-against-buffer-end, factorial ===");
    for (label, in_first, height) in cases {
        let probe = run(fd, &file, in_first, height);
        if probe.timed_out {
            println!("{label}: in_first={in_first} height={height}  TIMEOUT");
        } else if probe.mismatches == 0 {
            println!("{label}: in_first={in_first} height={height}  ok (0 mismatches)");
        } else {
            println!(
                "{label}: in_first={in_first} height={height}  FAIL {} mismatches, \
                 distinct_x={:?}, distinct_y_count={}",
                probe.mismatches,
                probe.distinct_x,
                probe.distinct_y.len()
            );
            for sample in &probe.samples {
                println!("    {sample}");
            }
        }
    }
    println!(
        "\nIf A and B fail the same way (regardless of flushness) and C and D both pass, \
         parity of in_first -- not buffer-end flushness -- is the cause. distinct_x should be \
         a single-element set ({{0}}) if this is really the same single-leading-pixel defect \
         found in run1."
    );
}
