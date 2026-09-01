#![cfg(feature = "hardware-characterization")]

//! Hardware characterization: data-rich width sweep for the `nonalign_dma`
//! dense-input alignment defect
//! `conv_dense_odd_in_first_probe_hw.rs` confirmed: on `run1`'s shape
//! (Cin=3 dense, width=228, so `input_row_stride = 228*3*2 = 1368`, `mod 16
//! == 8`), an odd `in_first` -- landing `CNA_FEATURE_DATA_ADDR` 8 bytes off
//! a 16-byte boundary -- reliably corrupts exactly one leading pixel per
//! output row, independent of `feature_grains` or buffer-end flushness.
//!
//! That shape only ever produces a 2-cycle: `in_first * 1368 mod 16` is
//! either 0 (aligned, correct) or 8 (misaligned, broken). Every other
//! offset the register field can take -- 2, 4, 6, 10, 12, 14 bytes off --
//! has never been produced by any capture or test in this repository. This
//! sweep produces them directly, by choosing `width` so `input_row_stride
//! mod 16` takes a different value, then walking `in_first` across a full
//! period so every reachable offset at that width gets a probe point.
//!
//! `input_row_stride = width * 3 * 2` is even for every width, so `mod 16`
//! is always even; only 8 residues are possible (0, 2, 4, 6, 8, 10, 12,
//! 14), and `width mod 8` determines which one a given width lands on:
//!
//! | width mod 8 | stride mod 16 |
//! |---:|---:|
//! | 0 | 0  (never misaligned -- the control) |
//! | 3 | 2 |
//! | 6 | 4 |
//! | 1 | 6 |
//! | 4 | 8  (`run1`'s own case) |
//! | 7 | 10 |
//! | 2 | 12 |
//! | 5 | 14 |
//!
//! Widths 227 (stride mod 16 = 2) and 229 (= 14) each have `gcd(stride mod
//! 16, 16) == 2`, so walking `in_first` 0..7 at either one cycles through
//! *all eight* residues by itself -- 227 ascending (0, 2, 4, .., 14), 229
//! descending in a different order (0, 14, 12, .., 2). Running both is a
//! cheap independent cross-check that a given byte offset behaves the same
//! regardless of which width produced it. Width 224 (stride mod 16 = 0) is
//! the negative control: no `in_first` can misalign it, so every point
//! there should pass -- if it doesn't, something in this test's own
//! methodology is wrong, not the hardware. Width 228 repeats the original
//! `run1`-scale case (only offsets 0 and 8 reachable) for a like-for-like
//! anchor against the earlier results.
//!
//! The first version of this sweep used the same one-hot center-tap weight
//! and per-row-distinguishable but x/channel-uniform input as
//! `conv_dense_odd_in_first_probe_hw.rs`. That was structurally blind to a
//! sub-pixel or channel displacement: offsets 2/4/6 appeared safe because
//! the displaced values were identical. The exact VGG-19 `features.0`
//! regression subsequently found every offset-4 tile about 94% wrong once
//! x and channel varied. This corrected sweep keeps the one-hot center tap
//! to isolate feature fetching, but varies input values with x, y, and
//! channel. Every mismatch's column is still reported.
//!
//!   CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!     cargo test --target aarch64-unknown-linux-gnu --release \
//!       --test conv_dense_alignment_width_sweep_hw --no-run
//!
//!   ./conv_dense_alignment_width_sweep_hw-<hash> --ignored --nocapture

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
const KERNEL: usize = 3;
const OUT_ROWS: u32 = 8;
const IN_ROWS: u32 = 10; // OUT_ROWS + KERNEL - 1, padding=[0,0].
const HEIGHT: u32 = 60; // Generous runway past every in_first probed (max 7+10=17) -- flushness already ruled out.

fn page_aligned_size(size: usize) -> usize {
    size.div_ceil(PAGE_BYTES) * PAGE_BYTES
}

fn small_integer_f16_bits(value: i16) -> u16 {
    match value {
        -3 => 0xc200,
        -2 => 0xc000,
        -1 => 0xbc00,
        0 => 0x0000,
        1 => 0x3c00,
        2 => 0x4000,
        3 => 0x4200,
        _ => panic!("test value {value} is outside the exact lookup table"),
    }
}

fn input_value(y: usize, x: usize, channel: usize) -> i16 {
    ((y * 13 + x * 7 + channel * 3 + (y * x) % 5) % 7) as i16 - 3
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

fn run(fd: i32, file: &std::fs::File, width: u32, in_first: u32) -> Probe {
    let kernels: Kernels = [KERNEL, KERNEL];
    let shape = Shape::with_out_channels(width, HEIGHT, 1, CIN, COUT).with_padding([0, 0]);
    let tile = Tile {
        out_first: in_first,
        out_rows: OUT_ROWS,
        in_first,
        in_rows: IN_ROWS,
        pad_top: 0,
    };
    let grains = IN_ROWS + KERNEL as u32; // ordinary derived value; already shown not to matter.
    assert!(shape.output_height(kernels) >= tile.out_first + tile.out_rows);
    assert!(HEIGHT >= in_first + IN_ROWS);

    unsafe {
        let input_bytes = width as usize * HEIGHT as usize * CIN as usize * FP16_BYTES;
        let buf_input = Buffer::new(fd, page_aligned_size(input_bytes), file);
        let input_words =
            std::slice::from_raw_parts_mut(buf_input.host_ptr as *mut u16, input_bytes / 2);
        for y in 0..HEIGHT as usize {
            for x in 0..width as usize {
                for c in 0..CIN as usize {
                    input_words[(y * width as usize + x) * CIN as usize + c] =
                        small_integer_f16_bits(input_value(y, x, c));
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
                for x in 0..out_width {
                    // Center tap, input channel 0 -> output channel 0.
                    let want = f32::from(input_value(y + 1, x + 1, 0));
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
fn alignment_offset_width_sweep() {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();

    // (label, width) -- in_first walks 0..=7 at each, covering every offset
    // that width's stride can produce.
    let widths: [(&str, u32); 4] = [
        ("control, stride mod16=0", 224),
        ("stride mod16=2, ascending", 227),
        ("stride mod16=14, descending", 229),
        ("run1's own case, stride mod16=8", 228),
    ];

    println!("\n=== nonalign_dma offset sweep: width, in_first -> byte offset -> result ===");
    let mut offset_results: std::collections::BTreeMap<u32, Vec<(u32, u32, bool)>> =
        std::collections::BTreeMap::new();
    for (label, width) in widths {
        let stride = width * CIN * FP16_BYTES as u32;
        println!(
            "--- width={width} ({label}), input_row_stride={stride}, mod16={} ---",
            stride % 16
        );
        for in_first in 0u32..8 {
            let offset = (in_first * stride) % 16;
            let probe = run(fd, &file, width, in_first);
            let ok = !probe.timed_out && probe.mismatches == 0;
            let status = if probe.timed_out {
                "TIMEOUT".to_string()
            } else if ok {
                "ok".to_string()
            } else {
                let first_x = probe.distinct_x.first().copied();
                let last_x = probe.distinct_x.last().copied();
                format!(
                    "FAIL {} mismatches across {} x positions ({first_x:?}..={last_x:?})",
                    probe.mismatches,
                    probe.distinct_x.len()
                )
            };
            println!("  in_first={in_first} offset={offset:2}  {status}");
            offset_results
                .entry(offset)
                .or_default()
                .push((width, in_first, ok));
        }
    }

    println!("\n=== summary by byte offset, across all widths that produced it ===");
    for (offset, results) in &offset_results {
        let all_ok = results.iter().all(|(_, _, ok)| *ok);
        let all_fail = results.iter().all(|(_, _, ok)| !ok);
        let consistency = if all_ok {
            "all pass"
        } else if all_fail {
            "all fail"
        } else {
            "MIXED -- inconsistent across widths/in_first at the same offset"
        };
        println!(
            "offset={offset:2}  {consistency}  ({} points: {:?})",
            results.len(),
            results
        );
    }
    println!(
        "\nIf offset=0 always passes and every nonzero offset always fails, the defect is any \
         nonzero misalignment, not specifically 8 bytes. If some nonzero offsets pass, the \
         hardware tolerates certain misalignments -- worth knowing exactly which before choosing \
         a fix. A 'MIXED' line would mean offset alone doesn't determine the outcome."
    );
}
