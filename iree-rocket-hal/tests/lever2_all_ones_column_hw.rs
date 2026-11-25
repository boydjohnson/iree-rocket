//! ROADMAP lever 2's one genuinely new mechanism: **the denominator comes
//! free from an all-ones column appended to `V^T`.**
//!
//! SOFTMAX.md §1.1 states it and nothing has run it. The second attention
//! matmul is `O^T = V^T P^T` -- `V^T` on width, keys on `Cin`, queries on
//! `Cout`, the probabilities down the weight path. Extend `V^T` by one width
//! position holding all ones and output width position `d_v` must hold
//! `sum_k P^T[k, q]`, which is exactly the softmax denominator for query
//! `q`, computed in the DPU's own accumulator with no extra dispatch and no
//! pooling tree.
//!
//! If that holds, it deletes the five-level sum tree
//! (`rocket_core::softmax`'s `SumLevel` stages) -- 11 of the 19 device tasks
//! SOFTMAX.md §10.3 measured at 1792 rows -- and it lifts the op's 1792-row
//! ceiling, which is the sum tree's depthwise channel limit rather than
//! anything the planes impose.
//!
//! What this file does **not** claim is that lever 2 is therefore a win.
//! That needs the PV matmul to be on the NPU at all, and
//! `lever2_pack_cost_hw.rs` measures the gate standing in the way.
//!
//! Not run by a plain `cargo test` -- see `conv_phase1_validation_hw.rs`'s
//! doc comment for the cross-compile-and-copy workflow:
//!
//!   CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!     cargo test --target aarch64-unknown-linux-gnu --release \
//!       --test lever2_all_ones_column_hw --no-run
//!
//! `run_fc` is `fc_hw.rs`'s own helper, duplicated rather than shared (this
//! crate's convention for hand-rolled hardware tests), with the buffers
//! sized from the shape instead of pinned to one page -- these operands are
//! tens of kilobytes, not hundreds of bytes.

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    conv::{Buffers, Precision},
    device::{Buffer, close_bo, fini_bo, prep_bo, submit},
    fc::{Plan, Shape},
    tensor_layout::{
        FEATURE_ATOMIC_BYTES, nc1hwc2_storage_size, pack_hwcf_to_rocket_weights,
        pack_nhwc_to_nc1hwc2, rocket_weight_storage_size,
    },
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const PAGE_BYTES: usize = 4096;

/// ViT-B/16's head width, which is also `V^T`'s width before the extra
/// column.
const HEAD_WIDTH: usize = 64;

fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = (bits >> 31) & 1;
    if value == 0.0 {
        return (sign << 15) as u16;
    }
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let fraction = bits & 0x7f_ffff;
    if exponent <= 0 {
        // Subnormal or below: round to zero. The probabilities this test
        // feeds reach 1e-9, well under fp16's normal range, and zero is
        // what the weight path would store for them anyway.
        return (sign << 15) as u16;
    }
    assert!(exponent < 31, "value {value} overflows fp16");
    ((sign << 15) | ((exponent as u32) << 10) | (fraction >> 13)) as u16
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15);
    let exponent = u32::from((bits >> 10) & 0x1f);
    let fraction = u32::from(bits & 0x3ff);
    if exponent == 0 {
        // Subnormal: fp16's own 2^-24 scale, counting from one not zero.
        // Getting this branch 4x wrong is a trap this repo has hit before.
        let mut value = 0.0f32;
        for bit in 0i32..10 {
            if fraction & (1u32 << bit) != 0 {
                value += 2.0f32.powi(bit - 24);
            }
        }
        return if sign == 1 { -value } else { value };
    }
    assert!(exponent != 0x1f, "fixture should hold no inf or NaN");
    f32::from_bits((sign << 31) | ((exponent + (127 - 15)) << 23) | (fraction << 13))
}

fn encode_fp16(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| f32_to_f16_bits(*value).to_le_bytes())
        .collect()
}

fn decode_fp16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| f16_bits_to_f32(u16::from_le_bytes([pair[0], pair[1]])))
        .collect()
}

fn decode_nc1hwc2(raw: &[u8], pixels: usize, channels: usize, element_bytes: usize) -> Vec<u8> {
    let bytes_per_pixel = channels * element_bytes;
    let surface_stride = pixels * FEATURE_ATOMIC_BYTES;
    let mut dense = vec![0; pixels * bytes_per_pixel];
    for pixel in 0..pixels {
        for byte in 0..bytes_per_pixel {
            let surface = byte / FEATURE_ATOMIC_BYTES;
            let lane = byte % FEATURE_ATOMIC_BYTES;
            dense[pixel * bytes_per_pixel + byte] =
                raw[surface * surface_stride + pixel * FEATURE_ATOMIC_BYTES + lane];
        }
    }
    dense
}

/// `fc_hw.rs`'s helper with shape-sized buffers. Returns the dense logical
/// `[M, N]` output bytes.
fn run_fc(shape: Shape, dense_input: &[u8], dense_weights: &[u8]) -> Vec<u8> {
    let element_bytes = shape.precision.element_bytes() as usize;
    let m = shape.m as usize;
    let k = shape.k as usize;
    let n = shape.n as usize;
    assert_eq!(dense_input.len(), m * k * element_bytes);
    assert_eq!(dense_weights.len(), k * n * element_bytes);

    let input_len = nc1hwc2_storage_size(m, k * element_bytes).unwrap();
    let mut packed_input = vec![0; input_len];
    pack_nhwc_to_nc1hwc2(dense_input, m, k * element_bytes, &mut packed_input).unwrap();

    let weight_len = rocket_weight_storage_size(1, 1, k, n, element_bytes).unwrap();
    let mut packed_weights = vec![0; weight_len];
    pack_hwcf_to_rocket_weights(
        dense_weights,
        1,
        1,
        k,
        n,
        element_bytes,
        &mut packed_weights,
    )
    .unwrap();

    let conv_shape = shape.as_conv_shape();
    let padded_outputs = conv_shape.padded_out_channels() as usize;
    let output_len = nc1hwc2_storage_size(m, padded_outputs * element_bytes).unwrap();
    let bias_len = padded_outputs * 4;

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let buf_in = Buffer::new(fd, input_len.next_multiple_of(PAGE_BYTES), &file);
        ptr::write_bytes(buf_in.host_ptr, 0, input_len.next_multiple_of(PAGE_BYTES));
        ptr::copy_nonoverlapping(packed_input.as_ptr(), buf_in.host_ptr, input_len);

        let buf_weights = Buffer::new(fd, weight_len.next_multiple_of(PAGE_BYTES), &file);
        ptr::write_bytes(
            buf_weights.host_ptr,
            0,
            weight_len.next_multiple_of(PAGE_BYTES),
        );
        ptr::copy_nonoverlapping(packed_weights.as_ptr(), buf_weights.host_ptr, weight_len);

        // fp16 bias is an f32 per output channel; all-zero is the identity.
        let buf_bias = Buffer::new(fd, bias_len.next_multiple_of(PAGE_BYTES), &file);
        ptr::write_bytes(buf_bias.host_ptr, 0, bias_len.next_multiple_of(PAGE_BYTES));

        let buf_out = Buffer::new(fd, output_len.next_multiple_of(PAGE_BYTES), &file);
        ptr::write_bytes(buf_out.host_ptr, 0, output_len.next_multiple_of(PAGE_BYTES));

        let programs = Plan::new(shape).programs_with_buffers(Buffers {
            input: buf_in.dma_address,
            weights: buf_weights.dma_address,
            bias: buf_bias.dma_address,
            output: buf_out.dma_address,
        });
        assert_eq!(programs.len(), 1, "this shape should fit one job");
        let commands = &programs[0];
        let command_bytes = commands.len() * mem::size_of::<u64>();
        let buf_commands = Buffer::new(fd, command_bytes.next_multiple_of(PAGE_BYTES), &file);
        let command_words =
            std::slice::from_raw_parts_mut(buf_commands.host_ptr as *mut u64, commands.len());
        for (destination, command) in command_words.iter_mut().zip(commands) {
            *destination = command.0;
        }

        fini_bo(fd, buf_in.handle).ok();
        fini_bo(fd, buf_weights.handle).ok();
        fini_bo(fd, buf_bias.handle).ok();
        fini_bo(fd, buf_out.handle).ok();
        fini_bo(fd, buf_commands.handle).ok();

        submit(
            fd,
            buf_commands.dma_address,
            commands.len() as u32,
            &[
                buf_commands.handle,
                buf_in.handle,
                buf_weights.handle,
                buf_bias.handle,
            ],
            &[buf_out.handle],
        )
        .expect("SUBMIT ioctl failed");
        prep_bo(fd, buf_out.handle, 2_000_000_000).expect("PV matmul timed out");

        let raw = std::slice::from_raw_parts(buf_out.host_ptr, output_len);
        let output = decode_nc1hwc2(raw, m, padded_outputs, element_bytes);

        close_bo(fd, buf_commands.handle).ok();
        close_bo(fd, buf_in.handle).ok();
        close_bo(fd, buf_weights.handle).ok();
        close_bo(fd, buf_bias.handle).ok();
        close_bo(fd, buf_out.handle).ok();

        // Trim the parity padding back to the logical output channels.
        let mut logical = vec![0u8; m * n * element_bytes];
        for pixel in 0..m {
            let from = pixel * padded_outputs * element_bytes;
            let to = pixel * n * element_bytes;
            logical[to..to + n * element_bytes]
                .copy_from_slice(&output[from..from + n * element_bytes]);
        }
        logical
    }
}

/// The fixture's 64 real ViT-B/16 attention rows, `[rows, keys]`.
fn attention_rows() -> (Vec<f32>, usize, usize) {
    const RAW: &[u8] = include_bytes!("fixtures/vit_b_16_attention_rows.f32");
    let values: Vec<f32> = RAW
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect();
    let keys = 197;
    let rows = values.len() / keys;
    assert_eq!(rows * keys, values.len(), "fixture is [rows, 197]");
    (values, rows, keys)
}

/// An exact f32 softmax of each row, which is what the probabilities would
/// be if `exp` were perfect. The point here is the *matmul*, so the
/// probabilities are exact by construction and the only error measured is
/// the contraction's own.
fn exact_probabilities(rows: &[f32], row_count: usize, keys: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; row_count * keys];
    for row in 0..row_count {
        let slice = &rows[row * keys..(row + 1) * keys];
        let max = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exponentials: Vec<f32> = slice.iter().map(|value| (value - max).exp()).collect();
        let total: f32 = exponentials.iter().sum();
        for (key, value) in exponentials.iter().enumerate() {
            out[row * keys + key] = value / total;
        }
    }
    out
}

/// A deterministic stand-in for `V`, `[keys, d_v]`, in a range fp16 holds
/// exactly enough that the reference below is about the contraction and not
/// about the operands.
fn values_matrix(keys: usize, width: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; keys * width];
    for key in 0..keys {
        for lane in 0..width {
            let index = (key * 37 + lane * 11) % 64;
            out[key * width + lane] = (index as f32 - 32.0) / 8.0;
        }
    }
    out
}

/// Lever 2's claim, on hardware: output width position `d_v` is the
/// denominator.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn the_all_ones_column_gives_the_denominator() {
    let (rows_f32, queries, keys) = attention_rows();
    let probabilities = exact_probabilities(&rows_f32, queries, keys);
    let values = values_matrix(keys, HEAD_WIDTH);

    // Activations: `V^T` with one appended all-ones width position.
    // Logical `[m = d_v + 1, k = keys]`.
    let width = HEAD_WIDTH + 1;
    let mut dense_activations = vec![0.0f32; width * keys];
    for lane in 0..HEAD_WIDTH {
        for key in 0..keys {
            dense_activations[lane * keys + key] = values[key * HEAD_WIDTH + lane];
        }
    }
    for key in 0..keys {
        dense_activations[HEAD_WIDTH * keys + key] = 1.0;
    }

    // Weights: `P^T`, logical `[k = keys, n = queries]`.
    let mut dense_weights = vec![0.0f32; keys * queries];
    for key in 0..keys {
        for query in 0..queries {
            dense_weights[key * queries + query] = probabilities[query * keys + key];
        }
    }

    let shape = Shape::new(width as u32, keys as u32, queries as u32, Precision::Fp16);
    let activation_bytes = encode_fp16(&dense_activations);
    let weight_bytes = encode_fp16(&dense_weights);
    let output = decode_fp16(&run_fc(shape, &activation_bytes, &weight_bytes));
    assert_eq!(output.len(), width * queries);

    // The reference contracts the *fp16-rounded* operands in f32, which is
    // what the DPU's accumulator does, so any disagreement is the
    // hardware's and not the encoding's.
    let activations_f16 = decode_fp16(&activation_bytes);
    let weights_f16 = decode_fp16(&weight_bytes);
    let mut reference = vec![0.0f32; width * queries];
    for lane in 0..width {
        for query in 0..queries {
            let mut total = 0.0f32;
            for key in 0..keys {
                total += activations_f16[lane * keys + key] * weights_f16[key * queries + query];
            }
            reference[lane * queries + query] = total;
        }
    }

    // 1. The denominator column. Every row of an exact softmax sums to one,
    //    so this column should be one to fp16's resolution -- and that is a
    //    real check, not a tautology: it is the hardware adding up 197
    //    separately rounded fp16 probabilities.
    let mut worst_denominator = 0.0f32;
    for query in 0..queries {
        let measured = output[HEAD_WIDTH * queries + query];
        let error = (measured - 1.0).abs();
        worst_denominator = worst_denominator.max(error);
        assert!(
            error < 0.01,
            "query {query}: denominator column gave {measured}, not 1.0"
        );
    }

    // 2. The value columns, against the f32 contraction of the same fp16
    //    operands.
    let mut worst_value = 0.0f32;
    for lane in 0..HEAD_WIDTH {
        for query in 0..queries {
            let measured = output[lane * queries + query];
            let expected = reference[lane * queries + query];
            worst_value = worst_value.max((measured - expected).abs());
        }
    }

    println!(
        "all-ones column: worst |denominator - 1| = {worst_denominator:.2e} over {queries} \
         queries; worst |O^T - reference| = {worst_value:.2e} over {HEAD_WIDTH}x{queries}"
    );
    assert!(
        worst_value < 0.02,
        "value columns disagree with the reference by {worst_value}"
    );
}

/// And the part that makes it *free*: the extra width position buys no extra
/// task. If appending a column split the dispatch, lever 2 would be paying
/// for its denominator after all.
#[test]
fn the_all_ones_column_costs_no_extra_task() {
    let (_, queries, keys) = attention_rows();
    let bare = Shape::new(
        HEAD_WIDTH as u32,
        keys as u32,
        queries as u32,
        Precision::Fp16,
    );
    let augmented = Shape::new(
        HEAD_WIDTH as u32 + 1,
        keys as u32,
        queries as u32,
        Precision::Fp16,
    );

    let bare_programs = Plan::new(bare).programs();
    let augmented_programs = Plan::new(augmented).programs();
    println!(
        "PV matmul [{HEAD_WIDTH}, {keys}, {queries}]: {} job(s); with the all-ones column: {} job(s)",
        bare_programs.len(),
        augmented_programs.len()
    );
    assert_eq!(
        bare_programs.len(),
        augmented_programs.len(),
        "the all-ones column must not split the dispatch"
    );
    assert_eq!(
        Plan::new(bare).conv_plan().tiles().len(),
        Plan::new(augmented).conv_plan().tiles().len(),
        "the all-ones column must not add a tile"
    );
}
