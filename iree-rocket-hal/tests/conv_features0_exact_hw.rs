//! Exact VGG-19 `features.0` convolution regression.
//!
//! This matches the shape emitted by `rocket_conv_harness.py` after IREE has
//! materialized same-padding explicitly:
//!
//! ```text
//! input:   1x226x226x3 f16 (one-pixel zero border)
//! filter:  3x3x3x64 f16 (HWCF before Rocket packing)
//! output:  1x224x224x64 f16 in Rocket's raw scratch layout
//! ```
//!
//! The production compiler/runtime path is deterministic but differs from
//! the CPU by roughly 6-7 for this shape. Existing tests cover either the
//! nearby 228x228/Cin=3/Cout=256/3x3 case, or a simplified
//! 226x226/Cin=3/Cout=1/1x1 case; neither is this convolution.
//!
//! This test uses `ConvPlan::new`, the production HWCF weight packer, one
//! scratch output shared by every tile, and the driver's sequential
//! submit-then-wait pattern. Input values vary with x, y, and channel. Each
//! output channel has three nonzero integer weights at distinct
//! (ky, kx, cin) positions, collectively covering every input channel and
//! all nine spatial taps. The resulting sums are small exactly-representable
//! integers, avoiding a tolerance question while retaining enough data
//! variation to expose row, column, channel, tap, surface, or tile mistakes.
//!
//! Raw scratch is checked directly. This deliberately excludes the driver's
//! post-dispatch NC1HWC2-to-NHWC compaction: a failure here is in the
//! ConvPlan/register/hardware side, while a pass narrows the end-to-end
//! harness failure to the driver/runtime path after hardware completion.
//!
//! Cross-compile and run on the RK3588 board:
//!
//! ```text
//! CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!   cargo test --target aarch64-unknown-linux-gnu --release \
//!     --test conv_features0_exact_hw --no-run
//!
//! ./conv_features0_exact_hw-<hash> --ignored --nocapture
//! ```

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    conv::{Buffers, ConvPlan, FeatureLayout, Kernels, Shape},
    device::{Buffer, close_bo, fini_bo, prep_bo, submit},
    tensor_layout::{pack_hwcf_to_rocket_weights, rocket_weight_storage_size},
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const FP16_BYTES: usize = 2;
const FEATURE_ATOM_BYTES: usize = 16;
const CHANNELS_PER_ATOM: usize = FEATURE_ATOM_BYTES / FP16_BYTES;
const PAGE_BYTES: usize = 4096;
const PER_TILE_TIMEOUT_NS: u64 = 5_000_000_000;

const CIN: usize = 3;
const COUT: usize = 64;
const KERNEL: usize = 3;
const PADDED: usize = 226;
const OUTPUT: usize = 224;
const REPS: usize = 3;

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

/// Value in the physically padded input. The one-pixel border is zero,
/// matching `rocket_conv_harness.py`; the interior varies in every axis.
fn input_value(y: usize, x: usize, channel: usize) -> i16 {
    if y == 0 || x == 0 || y + 1 == PADDED || x + 1 == PADDED {
        return 0;
    }
    let logical_y = y - 1;
    let logical_x = x - 1;
    ((logical_y * 13 + logical_x * 7 + channel * 3 + (logical_y * logical_x) % 5) % 7) as i16 - 3
}

/// Three distinct HWCF positions and their integer weights for one output
/// channel. Across Cout=64 these rotate through all 27 tap/channel slots.
fn selected_weights(cout: usize) -> [(usize, i16); 3] {
    let first = cout % (KERNEL * KERNEL * CIN);
    [
        (first, 1),
        ((first + 7) % (KERNEL * KERNEL * CIN), -1),
        ((first + 17) % (KERNEL * KERNEL * CIN), 2),
    ]
}

fn decode_selector(selector: usize) -> (usize, usize, usize) {
    let ky = selector / (KERNEL * CIN);
    let remainder = selector % (KERNEL * CIN);
    let kx = remainder / CIN;
    let cin = remainder % CIN;
    (ky, kx, cin)
}

fn expected_value(y: usize, x: usize, cout: usize) -> i16 {
    selected_weights(cout)
        .into_iter()
        .map(|(selector, weight)| {
            let (ky, kx, cin) = decode_selector(selector);
            input_value(y + ky, x + kx, cin) * weight
        })
        .sum()
}

fn feature_0_shape() -> Shape {
    Shape::with_out_channels(PADDED as u32, PADDED as u32, 1, CIN as u32, COUT as u32)
        .with_padding([0, 0])
}

#[test]
fn feature_0_plan_is_gap_free_and_dense_alignment_safe() {
    let kernels: Kernels = [KERNEL, KERNEL];
    let shape = feature_0_shape();
    assert_eq!(shape.layout(), FeatureLayout::Dense);
    assert_eq!(shape.output_width(kernels), OUTPUT as u32);
    assert_eq!(shape.output_height(kernels), OUTPUT as u32);

    let plan = ConvPlan::new(shape, kernels);
    assert_eq!((plan.data_banks(), plan.weight_banks()), (11, 1));

    let mut covered = 0u32;
    for tile in plan.tiles() {
        assert_eq!(tile.rows.out_first, covered, "tile gap or overlap");
        assert_eq!(
            (tile.rows.in_first * shape.input_row_stride()) % FEATURE_ATOM_BYTES as u32,
            0,
            "tile at in_first={} is not 16-byte aligned",
            tile.rows.in_first
        );
        assert!(
            shape.dense_feature_offset_safe(tile.rows.in_first),
            "unsafe tile boundary at in_first={}",
            tile.rows.in_first
        );
        covered += tile.rows.out_rows;
    }
    assert_eq!(covered, OUTPUT as u32);
}

struct Failure {
    mismatches: usize,
    samples: Vec<String>,
    timed_out: bool,
    failing_tile: Option<usize>,
    first_bad: Option<(usize, usize, usize)>,
    tile_mismatches: Vec<usize>,
}

fn run(fd: i32, file: &std::fs::File) -> Result<(), Failure> {
    let kernels: Kernels = [KERNEL, KERNEL];
    let shape = feature_0_shape();
    let plan = ConvPlan::new(shape, kernels);
    let tiles = plan.tiles().len();
    let output_pixels = OUTPUT * OUTPUT;

    unsafe {
        let input_bytes = PADDED * PADDED * CIN * FP16_BYTES;
        let buf_input = Buffer::new(fd, page_aligned_size(input_bytes), file);
        let input_words =
            std::slice::from_raw_parts_mut(buf_input.host_ptr as *mut u16, input_bytes / 2);
        for y in 0..PADDED {
            for x in 0..PADDED {
                for channel in 0..CIN {
                    input_words[(y * PADDED + x) * CIN + channel] =
                        small_integer_f16_bits(input_value(y, x, channel));
                }
            }
        }

        let mut dense_weights = vec![0u16; KERNEL * KERNEL * CIN * COUT];
        for cout in 0..COUT {
            for (selector, weight) in selected_weights(cout) {
                let (ky, kx, cin) = decode_selector(selector);
                let index = ((ky * KERNEL + kx) * CIN + cin) * COUT + cout;
                dense_weights[index] = small_integer_f16_bits(weight);
            }
        }
        let dense_weight_bytes: Vec<u8> = dense_weights
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        let packed_weight_bytes = rocket_weight_storage_size(KERNEL, KERNEL, CIN, COUT, FP16_BYTES)
            .expect("weight storage size");
        let buf_weights = Buffer::new(fd, page_aligned_size(packed_weight_bytes), file);
        ptr::write_bytes(buf_weights.host_ptr, 0, buf_weights.size);
        let weight_dst = std::slice::from_raw_parts_mut(buf_weights.host_ptr, packed_weight_bytes);
        pack_hwcf_to_rocket_weights(
            &dense_weight_bytes,
            KERNEL,
            KERNEL,
            CIN,
            COUT,
            FP16_BYTES,
            weight_dst,
        )
        .expect("weight packing");

        let buf_bias = Buffer::new(fd, page_aligned_size(shape.bs_buffer_bytes()), file);
        ptr::write_bytes(buf_bias.host_ptr, 0, buf_bias.size);

        let output_bytes = shape.output_scratch_bytes(kernels);
        assert_eq!(
            output_bytes,
            output_pixels * COUT.div_ceil(CHANNELS_PER_ATOM) * FEATURE_ATOM_BYTES
        );
        let buf_output = Buffer::new(fd, page_aligned_size(output_bytes), file);
        ptr::write_bytes(buf_output.host_ptr, 0, buf_output.size);

        let programs = plan.programs_with_buffers(Buffers {
            input: buf_input.dma_address,
            weights: buf_weights.dma_address,
            bias: buf_bias.dma_address,
            output: buf_output.dma_address,
        });
        assert_eq!(programs.len(), tiles);

        let mut command_buffers = Vec::with_capacity(programs.len());
        for commands in &programs {
            let command_bytes = commands.len() * mem::size_of::<u64>();
            let buffer = Buffer::new(fd, page_aligned_size(command_bytes), file);
            ptr::write_bytes(buffer.host_ptr, 0, buffer.size);
            let words = std::slice::from_raw_parts_mut(buffer.host_ptr as *mut u64, commands.len());
            for (destination, command) in words.iter_mut().zip(commands) {
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

        let out_handles = [buf_output.handle];
        let mut failure = Failure {
            mismatches: 0,
            samples: Vec::new(),
            timed_out: false,
            failing_tile: None,
            first_bad: None,
            tile_mismatches: vec![0; tiles],
        };

        for (index, (buffer, count)) in command_buffers.iter().enumerate() {
            let in_handles = [
                buffer.handle,
                buf_input.handle,
                buf_weights.handle,
                buf_bias.handle,
            ];
            if submit(fd, buffer.dma_address, *count, &in_handles, &out_handles).is_err() {
                failure.timed_out = true;
                failure.failing_tile = Some(index);
                failure.samples.push(format!("tile {index}: SUBMIT failed"));
                break;
            }
            if prep_bo(fd, buf_output.handle, PER_TILE_TIMEOUT_NS).is_err() {
                failure.timed_out = true;
                failure.failing_tile = Some(index);
                failure
                    .samples
                    .push(format!("tile {index}: prep_bo did not complete"));
                break;
            }
        }

        if !failure.timed_out {
            let raw = std::slice::from_raw_parts(buf_output.host_ptr, output_bytes);
            let surface_stride = output_pixels * FEATURE_ATOM_BYTES;
            for y in 0..OUTPUT {
                let tile_index = plan
                    .tiles()
                    .iter()
                    .position(|tile| {
                        y as u32 >= tile.rows.out_first
                            && (y as u32) < tile.rows.out_first + tile.rows.out_rows
                    })
                    .expect("output row belongs to a tile");
                for x in 0..OUTPUT {
                    let pixel = y * OUTPUT + x;
                    for cout in 0..COUT {
                        let surface = cout / CHANNELS_PER_ATOM;
                        let lane = cout % CHANNELS_PER_ATOM;
                        let offset = surface * surface_stride
                            + pixel * FEATURE_ATOM_BYTES
                            + lane * FP16_BYTES;
                        let bits = u16::from_le_bytes([raw[offset], raw[offset + 1]]);
                        let got = f16_to_f32(bits);
                        let want = f32::from(expected_value(y, x, cout));
                        if got != want {
                            failure.mismatches += 1;
                            failure.tile_mismatches[tile_index] += 1;
                            if failure.first_bad.is_none() {
                                failure.first_bad = Some((y, x, cout));
                            }
                            if failure.samples.len() < 16 {
                                failure.samples.push(format!(
                                    "[y={y}, x={x}, c={cout}] want {want} got {got} ({bits:#06x})"
                                ));
                            }
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

        if failure.mismatches == 0 && !failure.timed_out {
            Ok(())
        } else {
            Err(failure)
        }
    }
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn feature_0_exact_shape_matches_sparse_integer_reference() {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();

    let shape = feature_0_shape();
    let plan = ConvPlan::new(shape, [KERNEL, KERNEL]);
    println!(
        "\n=== VGG-19 features.0 exact hardware regression: padded 226x226, Cin=3, Cout=64, K3 ==="
    );
    println!(
        "banks {}/{} tiles={}",
        plan.data_banks(),
        plan.weight_banks(),
        plan.tiles().len()
    );
    for (index, tile) in plan.tiles().iter().enumerate() {
        println!("  tile {index}: {:?}", tile.rows);
    }

    let mut passed = 0;
    for rep in 0..REPS {
        match run(fd, &file) {
            Ok(()) => {
                println!("rep {rep}: ok");
                passed += 1;
            }
            Err(failure) => {
                println!(
                    "rep {rep}: FAIL ({} mismatches, timed_out={}, failing_tile={:?}, first_bad={:?}, tile_mismatches={:?})",
                    failure.mismatches,
                    failure.timed_out,
                    failure.failing_tile,
                    failure.first_bad,
                    failure.tile_mismatches
                );
                for sample in &failure.samples {
                    println!("         {sample}");
                }
            }
        }
    }

    assert_eq!(
        passed, REPS,
        "features.0 exact shape failed -- see diagnostics above"
    );
}
