//! Hardware-in-the-loop tests for `fc::Plan` -- the capture-derived FC
//! lowering (physical height exactly one, see `fc.rs`'s own module doc
//! comment), structural checks `fc_phase3_hw.rs`'s own two geometry-sanity
//! tests don't cover: real weight-ABI/channel-selection isolation, per-row
//! (per-M) independence, and the full `[K,N]` fp16 weight-packing
//! permutation with an exact per-element reference.
//!
//! Not run by a plain `cargo test` -- see `conv_phase1_validation_hw.rs`'s
//! doc comment for the cross-compile-and-copy-to-the-board workflow;
//! identical here:
//!
//!   CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!     cargo test --target aarch64-unknown-linux-gnu --release \
//!       --test fc_hw --no-run
//!
//! `run_fc` is `fc_phase3_hw.rs`'s own helper, duplicated rather than shared
//! (this crate's established convention for hand-rolled hardware tests) --
//! it already handles both precisions and the real `NC1HWC2`/rocket-weight
//! packing this op needs, so every test here just varies the dense
//! `input`/`weights` byte content passed through it.
//!
//! int8 shapes use `conv::Quantization`'s real (zero-centered) convention
//! throughout, not Mesa's raw-byte-plus-0x80-offset one: a weight byte IS
//! the real signed value directly (no separate weights zero point exists in
//! this model), which makes an isolation probe like
//! `fc_packed_weights_select_one_output_channel` below simpler than its old
//! retired Mesa-derived equivalent -- zero really is neutral.

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    conv::{BsEntry, Buffers, Multiplier, Precision, Quantization, write_bs_buffer},
    device::{Buffer, close_bo, fini_bo, prep_bo, submit},
    fc::{Plan, Shape},
    tensor_layout::{
        FEATURE_ATOMIC_BYTES, nc1hwc2_storage_size, pack_hwcf_to_rocket_weights,
        pack_nhwc_to_nc1hwc2, rocket_weight_storage_size,
    },
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const PAGE_BYTES: usize = 4096;

fn int8_shape(m: u32, k: u32, n: u32) -> Shape {
    Shape::new(
        m,
        k,
        n,
        Precision::Int8(Quantization {
            input_zero_point: 0,
            output_zero_point: 0,
            weight_zero_point: 0,
            multiplier: Multiplier::for_unit_bs(1.0),
        }),
    )
}

fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = (bits >> 31) & 1;
    if value == 0.0 {
        return (sign << 15) as u16;
    }
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let fraction = bits & 0x7f_ffff;
    assert!((1..31).contains(&exponent));
    ((sign << 15) | ((exponent as u32) << 10) | (fraction >> 13)) as u16
}

fn encode_fp16(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values
        .into_iter()
        .flat_map(|value| f32_to_f16_bits(value).to_le_bytes())
        .collect()
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15);
    let exponent = u32::from((bits >> 10) & 0x1f);
    let fraction = u32::from(bits & 0x3ff);
    assert!(
        exponent != 0 && exponent != 0x1f,
        "test only emits normal finite fp16 values"
    );
    f32::from_bits((sign << 31) | ((exponent + (127 - 15)) << 23) | (fraction << 13))
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

/// Runs a real, single-job FC dispatch through `fc::Plan` and returns the
/// dense logical `[M,N]` output bytes (`fc_phase3_hw.rs`'s own helper).
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
    let padded_outputs = conv_shape.padded_out_channels();
    let output_len = nc1hwc2_storage_size(m, padded_outputs as usize * element_bytes).unwrap();
    let bias_len = match shape.precision {
        Precision::Fp16 => padded_outputs as usize * 4,
        Precision::Int8(_) => conv_shape.bs_buffer_bytes(),
    };

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let buf_in = Buffer::new(fd, PAGE_BYTES, &file);
        ptr::write_bytes(buf_in.host_ptr, 0, PAGE_BYTES);
        ptr::copy_nonoverlapping(packed_input.as_ptr(), buf_in.host_ptr, input_len);

        let buf_weights = Buffer::new(fd, PAGE_BYTES, &file);
        ptr::write_bytes(buf_weights.host_ptr, 0, PAGE_BYTES);
        ptr::copy_nonoverlapping(packed_weights.as_ptr(), buf_weights.host_ptr, weight_len);

        let buf_bias = Buffer::new(fd, PAGE_BYTES, &file);
        ptr::write_bytes(buf_bias.host_ptr, 0, PAGE_BYTES);
        if matches!(shape.precision, Precision::Int8(_)) {
            let entries = vec![BsEntry::default(); padded_outputs as usize];
            let bytes = std::slice::from_raw_parts_mut(buf_bias.host_ptr, bias_len);
            write_bs_buffer(bytes, &entries);
        }

        let buf_out = Buffer::new(fd, PAGE_BYTES, &file);
        ptr::write_bytes(buf_out.host_ptr, 0, PAGE_BYTES);

        let programs = Plan::new(shape).programs_with_buffers(Buffers {
            input: buf_in.dma_address,
            weights: buf_weights.dma_address,
            bias: buf_bias.dma_address,
            output: buf_out.dma_address,
        });
        assert_eq!(programs.len(), 1, "test FC shape must fit one job");
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
        prep_bo(fd, buf_out.handle, 2_000_000_000).expect("FC job timed out");

        let raw = std::slice::from_raw_parts(buf_out.host_ptr, output_len);
        let output = decode_nc1hwc2(raw, m, n, element_bytes);

        close_bo(fd, buf_commands.handle).ok();
        close_bo(fd, buf_in.handle).ok();
        close_bo(fd, buf_weights.handle).ok();
        close_bo(fd, buf_bias.handle).ok();
        close_bo(fd, buf_out.handle).ok();
        output
    }
}

/// Uniform-fill sanity check: every one of `M*N` int8 outputs should agree
/// (same input/weight everywhere, so no per-position difference should be
/// possible regardless of any unknown weight-buffer packing order). Small
/// real magnitudes (`k=16` lanes each contributing at most `3*2=6`, capped
/// at `96`) stay well inside int8 range under `Multiplier::for_unit_bs(1.0)`
/// (accumulator passes through as the real output, no rescale).
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn fc_uniform_fill_columns_agree() {
    let shape = int8_shape(4, 16, 32);
    for input_fill in [-3i8, 0, 3] {
        let input = vec![input_fill as u8; shape.m as usize * shape.k as usize];
        let weights = vec![2u8; shape.k as usize * shape.n as usize];
        let output = run_fc(shape, &input, &weights);
        assert!(
            output.iter().all(|&value| value == output[0]),
            "input_fill={input_fill}: expected all {}x{} outputs identical (uniform \
             input/weights), got {output:?}",
            shape.m,
            shape.n
        );
    }
}

/// A real matmul accumulation should actually respond to the input --
/// guards against a hollow "completes but never touches the data" pass.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn fc_output_tracks_input() {
    let shape = int8_shape(4, 16, 32);
    let weights = vec![2u8; shape.k as usize * shape.n as usize];
    let low = run_fc(
        shape,
        &vec![(-3i8) as u8; shape.m as usize * shape.k as usize],
        &weights,
    )[0];
    let high = run_fc(
        shape,
        &vec![3i8 as u8; shape.m as usize * shape.k as usize],
        &weights,
    )[0];
    assert_ne!(
        low, high,
        "output didn't change between input real=-3 ({low}) and input real=3 ({high}) -- \
         suggests the op isn't really reading the input"
    );
}

/// Validates the real logical `[K,N]` weight ABI: only logical output
/// channel `SELECTED_CHANNEL` should respond when its own weight column
/// changes, every other channel's weights (and this test's input) held at
/// a fixed nonzero value that produces a real, nonzero accumulator baseline
/// everywhere -- a stuck-at-zero bug in the weight ABI wouldn't be
/// distinguishable from "correctly isolated" if every other channel were
/// silently zero too.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn fc_packed_weights_select_one_output_channel() {
    const SELECTED_CHANNEL: usize = 17;
    let shape = int8_shape(4, 16, 32);
    let input = vec![2i8 as u8; shape.m as usize * shape.k as usize];
    let mut negative_weights = vec![1u8; shape.k as usize * shape.n as usize];
    let mut positive_weights = negative_weights.clone();
    for k in 0..shape.k as usize {
        let index = k * shape.n as usize + SELECTED_CHANNEL;
        negative_weights[index] = (-10i8) as u8;
        positive_weights[index] = 10i8 as u8;
    }
    let negative = run_fc(shape, &input, &negative_weights);
    let positive = run_fc(shape, &input, &positive_weights);

    let changed_channels = (0..shape.n as usize)
        .filter(|&n| {
            (0..shape.m as usize).any(|m| {
                let index = m * shape.n as usize + n;
                negative[index] != positive[index]
            })
        })
        .collect::<Vec<_>>();
    eprintln!(
        "logical output channels responding to selected-weight sign change: {changed_channels:?}"
    );
    assert_eq!(
        changed_channels,
        vec![SELECTED_CHANNEL],
        "expected only logical output channel {SELECTED_CHANNEL} to respond when its \
         weights changed sign, got {negative:?} vs {positive:?}"
    );
}

/// Load-bearing check for the M-as-width mapping: gives each of the `M=7`
/// logical rows a distinct real input value across its whole `K` depth and
/// checks the corresponding output rows track that distinction in the same
/// (increasing) order. A uniform whole-buffer fill can't catch a bug where
/// rows cross-contaminate (wrong per-row stride, wrong CBUF geometry
/// assumption) -- this can. Deliberately non-square (`M != K, N`) so the
/// mapping can't accidentally rely on a coincidental equal extent.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn fc_columns_are_independent() {
    let shape = int8_shape(7, 16, 8);
    let row_value = |m: usize| ((m as i32 - 3) * 2) as i8; // -6, -4, -2, 0, 2, 4, 6
    let mut input = vec![0u8; shape.m as usize * shape.k as usize];
    for m in 0..shape.m as usize {
        input[m * shape.k as usize..(m + 1) * shape.k as usize].fill(row_value(m) as u8);
    }
    let weights = vec![1u8; shape.k as usize * shape.n as usize];
    let output = run_fc(shape, &input, &weights);
    let rows: Vec<i8> = (0..shape.m as usize)
        .map(|m| output[m * shape.n as usize] as i8)
        .collect();

    eprintln!(
        "fc_columns_are_independent: per-row real input {:?} (x{} lanes) -> output rows {rows:?}",
        (0..shape.m as usize).map(row_value).collect::<Vec<_>>(),
        shape.k
    );
    assert!(
        rows.windows(2).all(|pair| pair[0] < pair[1]),
        "expected output rows to be strictly increasing, got {rows:?}"
    );
}

/// Gives every `[M,K]` input element a distinct exactly-representable fp16
/// integer. Output channel N selects exactly one K channel with a weight of
/// 1.0 (every other weight 0.0), so this checks the full logical `[K,N]`
/// packing permutation with an exact per-element reference.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn fc_fp16_distinct_weights_follow_channels() {
    let shape = Shape::new(4, 32, 16, Precision::Fp16);
    let input_value = |m: usize, k: usize| (m * shape.k as usize + k + 1) as f32;
    let input = encode_fp16(
        (0..shape.m as usize).flat_map(|m| (0..shape.k as usize).map(move |k| input_value(m, k))),
    );

    let mut weight_values = vec![0.0f32; shape.k as usize * shape.n as usize];
    for n in 0..shape.n as usize {
        let selected_k = (n * 2 + 1) % shape.k as usize;
        weight_values[selected_k * shape.n as usize + n] = 1.0;
    }
    let weights = encode_fp16(weight_values);
    let raw = run_fc(shape, &input, &weights);

    for m in 0..shape.m as usize {
        for n in 0..shape.n as usize {
            let selected_k = (n * 2 + 1) % shape.k as usize;
            let expected = input_value(m, selected_k);
            let offset = (m * shape.n as usize + n) * 2;
            let actual = f16_to_f32(u16::from_le_bytes([raw[offset], raw[offset + 1]]));
            assert_eq!(
                actual, expected,
                "fp16 FC [{m}, {n}] selects K={selected_k}: expected {expected}, got {actual}"
            );
        }
    }
}
