//! Generalizes `conv_dense_alignment_width_sweep_hw.rs`'s finding --
//! `nonalign_dma`'s dense-mode fetch corrupts leading pixels exactly when
//! `(in_first * input_row_stride) % 16 >= 8`, confirmed at Cin=3/fp16 across
//! every reachable byte offset -- across the other channel counts dense
//! mode covers (`Cin` 1, 2, 4) and across `Cout`, to check whether the
//! threshold is a `Cin=3`-specific coincidence or a property of the
//! `nonalign_dma` path itself.
//!
//! `input_row_stride = width * in_channels * element_bytes`. Each `Cin`
//! reaches a different, and differently rich, set of byte offsets:
//!
//! | Cin | stride factor | reachable offsets (fp16) |
//! |---:|---|---|
//! | 1 | `2 * width` | all 8 even values (0, 2, .., 14), same richness as `Cin=3` |
//! | 2 | `4 * width` | only 0, 4, 8, 12 -- coarser, but still crosses the threshold twice |
//! | 3 | `6 * width` | all 8 (see `conv_dense_alignment_width_sweep_hw.rs`) |
//! | 4 | `8 * width` | only 0, 8 -- the same 2-cycle `run1`'s own shape has |
//!
//! Widths are chosen so each `Cin`'s `in_first` sweep reaches every offset
//! that `Cin` can produce at all (`Cin=2` and `Cin=4` structurally cannot
//! reach the odd-numbered-quarter offsets `Cin=1`/`Cin=3` can, since their
//! stride factors are already multiples of 4 and 8 respectively -- that is
//! a property of the arithmetic, not a gap in this probe).
//!
//! `Cout` is checked separately, at `Cin=3` (already fully characterized),
//! comparing `Cout=256` (every previous test in this line) against
//! `Cout=8` at the same two offsets (0 and 8) -- `input_row_stride` and
//! `CNA_FEATURE_DATA_ADDR` do not depend on `Cout` at all, so this is a
//! sanity confirmation that the mechanism really is address-only, not a
//! search for a plausible alternative.
//!
//! Same one-hot center-tap weight and per-row-distinguishable input as the
//! rest of this line of tests, reporting every mismatched column.
//!
//!   CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!     cargo test --target aarch64-unknown-linux-gnu --release \
//!       --test conv_dense_alignment_channel_sweep_hw --no-run
//!
//!   ./conv_dense_alignment_channel_sweep_hw-<hash> --ignored --nocapture

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

const KERNEL: usize = 3;
const OUT_ROWS: u32 = 8;
const IN_ROWS: u32 = 10; // OUT_ROWS + KERNEL - 1, padding=[0,0].
const HEIGHT: u32 = 60; // Generous runway -- flushness already ruled out as a factor.

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

fn run(fd: i32, file: &std::fs::File, cin: u32, cout: u32, width: u32, in_first: u32) -> Probe {
    let kernels: Kernels = [KERNEL, KERNEL];
    let shape = Shape::with_out_channels(width, HEIGHT, 1, cin, cout).with_padding([0, 0]);
    debug_assert!(matches!(
        shape.layout(),
        iree_rocket_hal::rocket::conv::FeatureLayout::Dense
    ));
    let tile = Tile {
        out_first: in_first,
        out_rows: OUT_ROWS,
        in_first,
        in_rows: IN_ROWS,
        pad_top: 0,
    };
    let grains = IN_ROWS + KERNEL as u32; // shown not to matter in the earlier grains sweep.
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

        let dense_weight_elems = KERNEL * KERNEL * cin as usize * cout as usize;
        let mut dense_weights = vec![0u16; dense_weight_elems];
        let center_index = (1 * KERNEL + 1) * cin as usize * cout as usize + 0 * cout as usize + 0;
        dense_weights[center_index] = 0x3c00;
        let dense_weight_bytes: Vec<u8> =
            dense_weights.iter().flat_map(|w| w.to_le_bytes()).collect();

        let packed_weight_bytes =
            rocket_weight_storage_size(KERNEL, KERNEL, cin as usize, cout as usize, FP16_BYTES)
                .expect("weight storage size");
        let buf_weights = Buffer::new(fd, page_aligned_size(packed_weight_bytes), file);
        ptr::write_bytes(buf_weights.host_ptr, 0, buf_weights.size);
        let weight_dst = std::slice::from_raw_parts_mut(buf_weights.host_ptr, packed_weight_bytes);
        pack_hwcf_to_rocket_weights(
            &dense_weight_bytes,
            KERNEL,
            KERNEL,
            cin as usize,
            cout as usize,
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

fn report(label: &str, expect_offset_ge_8_to_fail: bool, offset: u32, probe: &Probe) {
    let ok = !probe.timed_out && probe.mismatches == 0;
    let expected_ok = offset < 8;
    let flag = if ok == expected_ok {
        ""
    } else {
        "  <-- UNEXPECTED"
    };
    let status = if probe.timed_out {
        "TIMEOUT".to_string()
    } else if ok {
        "ok".to_string()
    } else {
        format!(
            "FAIL {} mismatches distinct_x={:?}",
            probe.mismatches, probe.distinct_x
        )
    };
    let _ = expect_offset_ge_8_to_fail;
    println!("{label}  offset={offset:2}  {status}{flag}");
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn alignment_threshold_across_channels_and_cout() {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();

    println!("\n=== Cin sweep, Cout=256 fixed, threshold should stay at offset>=8 ===");
    // (label, cin, width, in_first range)
    let cin_cases: [(&str, u32, u32, std::ops::Range<u32>); 3] = [
        ("Cin=1 w225", 1, 225, 0..8),
        ("Cin=2 w225", 2, 225, 0..4),
        ("Cin=4 w225", 4, 225, 0..2),
    ];
    for (label, cin, width, in_firsts) in cin_cases {
        let stride = width * cin * FP16_BYTES as u32;
        println!(
            "--- {label}: input_row_stride={stride} mod16={} ---",
            stride % 16
        );
        for in_first in in_firsts {
            let offset = (in_first * stride) % 16;
            let probe = run(fd, &file, cin, 256, width, in_first);
            report(&format!("  in_first={in_first}"), true, offset, &probe);
        }
    }

    println!("\n=== Cout sweep, Cin=3/width=227 fixed (stride mod16=2) ===");
    for (cout, in_first) in [(8u32, 0u32), (8, 4), (256, 0), (256, 4)] {
        let stride = 227 * 3 * FP16_BYTES as u32;
        let offset = (in_first * stride) % 16;
        let probe = run(fd, &file, 3, cout, 227, in_first);
        report(
            &format!("  Cout={cout} in_first={in_first}"),
            true,
            offset,
            &probe,
        );
    }

    println!(
        "\nEvery line should read 'ok' when offset<8 and 'FAIL ... distinct_x={{0}}' (or {{0,1}} at \
         offset 14) when offset>=8, exactly as conv_dense_alignment_width_sweep_hw.rs found at \
         Cin=3. Any '<-- UNEXPECTED' line means the threshold does not generalize to that Cin or \
         that Cout matters after all -- worth flagging specifically."
    );
}
