//! Hardware-in-the-loop tests for `build_fc_regcmd` -- Phase 2 of the
//! ukernel roadmap. FC as a 1x1-kernel conv, M mapped onto `input_width`,
//! height fixed internally at `FC_SAFE_HEIGHT` (see `regcmd.rs`'s FC
//! section doc comment for the full rationale, especially the
//! `input_height >= 4` underflow risk this sidesteps).
//!
//! Not run by a plain `cargo test` -- see conv_hw.rs's doc comment for the
//! cross-compile-and-copy-to-the-board workflow; identical here:
//!
//!   CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!     cargo test --target aarch64-unknown-linux-gnu --release \
//!       --test fc_hw --no-run
//!
//! Smallest atomic-aligned shape per the roadmap plan: m=4, k=16, n=32
//! (k already a full FEATURE_ATOMIC_SIZE group, n a full task_output_
//! channels group -- no padding-related surprises to also debug on the
//! first hardware round).
//!
//! Two kinds of test, same uniform-fill-sidesteps-packing-order strategy
//! established by conv_hw.rs/pooling_hw.rs throughout this crate:
//! - `fc_uniform_fill_*`/`fc_output_tracks_input`: whole-input-buffer
//!   uniform fill, proves the dispatch completes and genuinely reads the
//!   input, regardless of any unknown per-row/per-channel weight packing.
//! - `fc_rows_are_independent`: NOT a uniform fill -- gives each of the 4
//!   logical M positions (columns, since M maps onto width) a distinct
//!   per-column fill value and checks the corresponding output columns
//!   differ from each other in the same relative order. This is the load-
//!   bearing check for the M-as-width mapping itself: if rows/columns were
//!   secretly cross-contaminating (wrong stride math, wrong CBUF geometry
//!   assumption), this is what would catch it -- a uniform fill alone
//!   cannot.
//! - `fc_uniform_fill_all_output_channels_agree`: decodes all N=32 output
//!   channels, crossing the 16-byte feature-surface boundary that the older
//!   row helpers never inspected.
//! - `fc_packed_weights_select_one_output_channel`: starts from logical
//!   `[M,K]`/`[K,N]` tensors, packs them with the same layout helpers the HAL
//!   driver uses, and proves that one nonzero logical weight column changes
//!   only the matching output channel.
//! - `fc_non_square_m_columns_are_independent`: repeats the M-as-width check
//!   at M=7, so the original square M=4/height=4 geometry cannot accidentally
//!   hide a line/surface-stride error.

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    device::{Buffer, close_bo, fini_bo, prep_bo, submit},
    regcmd::{Activation, FcBuffers, FcShape, build_fc_regcmd},
    tensor_layout::{
        nc1hwc2_storage_size, pack_hwcf_to_rocket_weights, pack_nhwc_to_nc1hwc2,
        rocket_weight_storage_size,
    },
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const TENSOR_SIZE: usize = 4096;
const FC_SAFE_HEIGHT: usize = 4;
const FEATURE_ATOMIC_BYTES: usize = 16;

/// Smallest atomic-aligned FC shape: m=4 (maps onto input_width, matches
/// FC_SAFE_HEIGHT so the padded cube is square), k=16 (one full
/// FEATURE_ATOMIC_SIZE group, no channel padding), n=32 (one full
/// task_output_channels group, no output-channel padding).
fn small_shape() -> FcShape {
    FcShape {
        m: 4,
        k: 16,
        n: 32,
        input_zero_point: 0,
        output_zero_point: 0,
        weights_zero_point: 0,
        input_scale: 1.0,
        weights_scale: 1.0,
        output_scale: 1.0,
        truncate_bits: 0,
        activation: Activation::None,
    }
}

/// Runs `shape` with the whole input plane filled with `input_fill` and
/// the whole weight buffer filled with `weight_fill`, returning row 0's
/// `m` real output pixels (the only row with real, not garbage, input --
/// see regcmd.rs's FC section doc comment).
///
/// Read at stride 16 bytes/pixel, same convention as conv_hw.rs/
/// pooling_hw.rs -- output_channels padding still lands each pixel at a
/// full 16-byte-aligned atomic slot regardless of real channel count.
fn run_uniform_fc(shape: &FcShape, input_fill: u8, weight_fill: u8) -> Vec<u8> {
    let dense = run_uniform_fc_all_channels(shape, input_fill, weight_fill);
    first_output_channel(shape, &dense)
}

/// Uniform-fill runner that preserves all `N` channels of row 0 in dense
/// `[M,N]` order. The older FC tests intentionally sampled only channel 0;
/// tests that need to validate output-surface addressing use this instead.
fn run_uniform_fc_all_channels(shape: &FcShape, input_fill: u8, weight_fill: u8) -> Vec<u8> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let buf_in = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_in.host_ptr, input_fill, TENSOR_SIZE);

        let buf_w = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_w.host_ptr, weight_fill, TENSOR_SIZE);

        let buf_bias = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_bias.host_ptr, 0, TENSOR_SIZE);

        let buf_out = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_out.host_ptr, 0, TENSOR_SIZE);

        let output = run_fc(&file, fd, shape, &buf_in, &buf_w, &buf_bias, &buf_out);

        close_bo(fd, buf_in.handle).ok();
        close_bo(fd, buf_w.handle).ok();
        close_bo(fd, buf_bias.handle).ok();
        close_bo(fd, buf_out.handle).ok();

        output
    }
}

fn first_output_channel(shape: &FcShape, dense_output: &[u8]) -> Vec<u8> {
    (0..shape.m as usize)
        .map(|m| dense_output[m * shape.n as usize])
        .collect()
}

/// Converts the DPU's feature-surface output into the logical row-0 `[M,N]`
/// tensor. The physical cube contains four spatial rows; only row 0 belongs
/// to the public FC result, but every channel surface is still strided over
/// all `M * FC_SAFE_HEIGHT` physical pixels.
fn decode_fc_row0(shape: &FcShape, raw: &[u8]) -> Vec<u8> {
    let m = shape.m as usize;
    let n = shape.n as usize;
    let physical_pixels = m * FC_SAFE_HEIGHT;
    let surface_stride = physical_pixels * FEATURE_ATOMIC_BYTES;
    let mut dense = vec![0u8; m * n];

    for row in 0..m {
        for channel in 0..n {
            let surface = channel / FEATURE_ATOMIC_BYTES;
            let lane = channel % FEATURE_ATOMIC_BYTES;
            let src = surface * surface_stride + row * FEATURE_ATOMIC_BYTES + lane;
            dense[row * n + channel] = raw[src];
        }
    }
    dense
}

/// Shared dispatch/readback plumbing -- factored out so the per-row-
/// independence test can fill its own per-column pattern instead of a
/// single uniform fill.
unsafe fn run_fc(
    file: &std::fs::File,
    fd: i32,
    shape: &FcShape,
    buf_in: &Buffer,
    buf_w: &Buffer,
    buf_bias: &Buffer,
    buf_out: &Buffer,
) -> Vec<u8> {
    unsafe {
        let bufs = FcBuffers {
            input_addr: buf_in.dma_address,
            weights_addr: buf_w.dma_address,
            bias_addr: buf_bias.dma_address,
            output_addr: buf_out.dma_address,
        };
        let cmds = build_fc_regcmd(shape, &bufs);

        let cmd_bytes = cmds.len() * mem::size_of::<u64>();
        let cmd_len = cmd_bytes.next_multiple_of(4096);
        let buf_cmd = Buffer::new(fd, cmd_len, file);
        let cmd_slice = std::slice::from_raw_parts_mut(buf_cmd.host_ptr as *mut u64, cmds.len());
        for (i, c) in cmds.iter().enumerate() {
            cmd_slice[i] = c.0;
        }

        fini_bo(fd, buf_in.handle).ok();
        fini_bo(fd, buf_w.handle).ok();
        fini_bo(fd, buf_bias.handle).ok();
        fini_bo(fd, buf_out.handle).ok();
        fini_bo(fd, buf_cmd.handle).ok();

        let in_handles = [buf_cmd.handle, buf_in.handle, buf_w.handle, buf_bias.handle];
        let out_handles = [buf_out.handle];

        submit(
            fd,
            buf_cmd.dma_address,
            cmds.len() as u32,
            &in_handles,
            &out_handles,
        )
        .expect("SUBMIT ioctl failed");

        prep_bo(fd, buf_out.handle, 2_000_000_000).expect("job did not complete within timeout");
        close_bo(fd, buf_cmd.handle).ok();

        // Row 0 only (h=0) is logically returned, but decoding channel
        // surfaces requires retaining their full four-row physical stride.
        let physical_pixels = shape.m as usize * FC_SAFE_HEIGHT;
        let surface_count = (shape.n as usize).div_ceil(FEATURE_ATOMIC_BYTES);
        let raw_len = physical_pixels * surface_count * FEATURE_ATOMIC_BYTES;
        assert!(
            raw_len <= TENSOR_SIZE,
            "test output buffer too small for m={} n={}",
            shape.m,
            shape.n
        );
        let raw = std::slice::from_raw_parts(buf_out.host_ptr, raw_len);
        decode_fc_row0(shape, raw)
    }
}

/// Runs logical dense `[M,K]` input and `[K,N]` weights through the explicit
/// host-side layouts required by the convolution-backed FC implementation.
/// Rows 1..3 of the physical input cube are filled with the input zero point;
/// only row 0 receives the caller's logical M rows.
fn run_packed_fc(shape: &FcShape, input: &[u8], weights: &[u8]) -> Vec<u8> {
    let m = shape.m as usize;
    let k = shape.k as usize;
    let n = shape.n as usize;
    assert_eq!(input.len(), m * k);
    assert_eq!(weights.len(), k * n);

    let physical_pixels = m * FC_SAFE_HEIGHT;
    let mut physical_dense_input = vec![shape.input_zero_point as u8; physical_pixels * k];
    physical_dense_input[..input.len()].copy_from_slice(input);
    let packed_input_len = nc1hwc2_storage_size(physical_pixels, k).unwrap();
    let mut packed_input = vec![0u8; packed_input_len];
    pack_nhwc_to_nc1hwc2(&physical_dense_input, physical_pixels, k, &mut packed_input).unwrap();

    let packed_weights_len = rocket_weight_storage_size(1, 1, k, n, 1).unwrap();
    let mut packed_weights = vec![shape.weights_zero_point as u8; packed_weights_len];
    pack_hwcf_to_rocket_weights(weights, 1, 1, k, n, 1, &mut packed_weights).unwrap();

    assert!(packed_input_len <= TENSOR_SIZE);
    assert!(packed_weights_len <= TENSOR_SIZE);

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let buf_in = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_in.host_ptr, shape.input_zero_point as u8, TENSOR_SIZE);
        ptr::copy_nonoverlapping(packed_input.as_ptr(), buf_in.host_ptr, packed_input_len);

        let buf_w = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_w.host_ptr, shape.weights_zero_point as u8, TENSOR_SIZE);
        ptr::copy_nonoverlapping(packed_weights.as_ptr(), buf_w.host_ptr, packed_weights_len);

        let buf_bias = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_bias.host_ptr, 0, TENSOR_SIZE);

        let buf_out = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_out.host_ptr, 0, TENSOR_SIZE);

        let output = run_fc(&file, fd, shape, &buf_in, &buf_w, &buf_bias, &buf_out);

        close_bo(fd, buf_in.handle).ok();
        close_bo(fd, buf_w.handle).ok();
        close_bo(fd, buf_bias.handle).ok();
        close_bo(fd, buf_out.handle).ok();

        output
    }
}

/// Uniform-fill sanity check: every one of m=4 output columns should agree
/// (same input/weight fill everywhere, so no per-column difference should
/// be possible regardless of any unknown weight-buffer packing order).
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn fc_uniform_fill_columns_agree() {
    let shape = small_shape();
    for input_fill in [10u8, 118, 200] {
        let pixels = run_uniform_fc(&shape, input_fill, 2);
        assert!(
            pixels.iter().all(|&p| p == pixels[0]),
            "input_fill={input_fill}: expected all {} output columns identical \
             (uniform input/weights), got {pixels:?}",
            shape.m
        );
    }
}

/// A real matmul accumulation should actually respond to the input --
/// guards against a hollow "completes but never touches the data" pass.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn fc_output_tracks_input() {
    let shape = small_shape();
    let low = run_uniform_fc(&shape, 10, 2)[0];
    let high = run_uniform_fc(&shape, 200, 2)[0];
    assert_ne!(
        low, high,
        "output column value didn't change between input_fill=10 ({low}) and \
         input_fill=200 ({high}) -- suggests the op isn't really reading the input"
    );
}

/// The original FC tests used N=32 but sampled only channel 0. This decodes
/// both 16-byte output surfaces and verifies that uniform input/weights
/// produce the same value in every logical `[M,N]` position.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn fc_uniform_fill_all_output_channels_agree() {
    let shape = small_shape();
    let output = run_uniform_fc_all_channels(&shape, 118, 2);
    assert!(
        output.iter().all(|&value| value == output[0]),
        "uniform input/weights should agree across all {}x{} logical FC outputs, got {output:?}",
        shape.m,
        shape.n
    );
}

/// Validates the logical `[K,N]` weight ABI and the second output surface.
/// Only logical output channel 17 receives the same nonzero coefficient used
/// by `fc_output_tracks_input`; changing the input must therefore change that
/// channel and no other channel.
///
/// Deliberately uses the already-proven zero-point=0, weight=2 regime. The
/// tempting symmetric-zero-point version (`0x80` baseline, `0x82` selected
/// weight) is not diagnostic: the shared int8 1x1-conv output-conversion bug
/// collapses both its zero and small-positive accumulators to raw 127.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn fc_packed_weights_select_one_output_channel() {
    const SELECTED_CHANNEL: usize = 17;
    let shape = small_shape();
    let mut selected_weights = vec![0u8; shape.k as usize * shape.n as usize];
    for k in 0..shape.k as usize {
        selected_weights[k * shape.n as usize + SELECTED_CHANNEL] = 2;
    }
    let low_input = vec![10u8; shape.m as usize * shape.k as usize];
    let high_input = vec![200u8; shape.m as usize * shape.k as usize];
    let low = run_packed_fc(&shape, &low_input, &selected_weights);
    let high = run_packed_fc(&shape, &high_input, &selected_weights);

    let changed_channels = (0..shape.n as usize)
        .filter(|&n| {
            (0..shape.m as usize).any(|m| {
                let index = m * shape.n as usize + n;
                low[index] != high[index]
            })
        })
        .collect::<Vec<_>>();
    let selected_values = (0..shape.m as usize)
        .map(|m| {
            let index = m * shape.n as usize + SELECTED_CHANNEL;
            (low[index], high[index])
        })
        .collect::<Vec<_>>();
    eprintln!(
        "logical output channels responding to input sweep: {changed_channels:?}; \
         selected channel {SELECTED_CHANNEL} (low, high) by M: {selected_values:?}"
    );
    assert_eq!(
        changed_channels,
        vec![SELECTED_CHANNEL],
        "expected only logical output channel {SELECTED_CHANNEL} to respond to input changing from 10 to 200"
    );
}

/// Exercises M-as-width when M differs from the internal fixed height (4).
/// Logical input rows have increasing fills and are packed into physical row
/// 0; output channel 0 must preserve that order without cross-contamination.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn fc_non_square_m_columns_are_independent() {
    let shape = FcShape {
        m: 7,
        ..small_shape()
    };
    let mut input = vec![0u8; shape.m as usize * shape.k as usize];
    for m in 0..shape.m as usize {
        input[m * shape.k as usize..(m + 1) * shape.k as usize].fill((10 + m * 30) as u8);
    }
    let weights = vec![2u8; shape.k as usize * shape.n as usize];
    let dense = run_packed_fc(&shape, &input, &weights);
    let columns = first_output_channel(&shape, &dense);

    assert!(
        columns.windows(2).all(|pair| pair[0] <= pair[1]),
        "expected M=7 output columns to be non-decreasing, got {columns:?}"
    );
    assert_ne!(
        columns.first(),
        columns.last(),
        "first and last M=7 outputs are identical despite distinct logical input rows: {columns:?}"
    );
}

/// Load-bearing check for the M-as-width mapping itself: gives each of the
/// 4 logical M positions (columns) a distinct fill value across its whole
/// K depth, and checks the m real output columns track that per-column
/// distinction in the same relative order (monotonically, since a higher
/// uniform input column should produce a higher accumulator everywhere
/// weights/bias are uniform). A uniform whole-buffer fill can't catch a
/// bug where M positions cross-contaminate (wrong per-column stride, wrong
/// CBUF geometry) -- this can.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn fc_columns_are_independent() {
    let shape = small_shape();
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open NPU device");
    let fd = file.as_raw_fd();

    // Column-major per-16-byte-atomic-pixel fill: FEATURE_ATOMIC_SIZE=16
    // bytes per (w, h) position regardless of real channel count, same
    // atomic convention as every other builder in this module. Column w
    // gets fill value col_fill(w), replicated across every row (only row 0
    // is real, but filling all rows identically is simplest and harmless).
    let col_fill = |w: u32| -> u8 { (10 + w * 60) as u8 }; // 10, 70, 130, 190

    unsafe {
        let buf_in = Buffer::new(fd, TENSOR_SIZE, &file);
        for h in 0..4u32 {
            for w in 0..shape.m {
                let row_base = (h * shape.m + w) as usize * 16;
                ptr::write_bytes(buf_in.host_ptr.add(row_base), col_fill(w), 16);
            }
        }

        let buf_w = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_w.host_ptr, 2, TENSOR_SIZE);

        let buf_bias = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_bias.host_ptr, 0, TENSOR_SIZE);

        let buf_out = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_out.host_ptr, 0, TENSOR_SIZE);

        let dense = run_fc(&file, fd, &shape, &buf_in, &buf_w, &buf_bias, &buf_out);
        let pixels = first_output_channel(&shape, &dense);

        close_bo(fd, buf_in.handle).ok();
        close_bo(fd, buf_w.handle).ok();
        close_bo(fd, buf_bias.handle).ok();
        close_bo(fd, buf_out.handle).ok();

        eprintln!(
            "fc_columns_are_independent: per-column input fills {:?} -> output columns {pixels:?}",
            (0..shape.m).map(col_fill).collect::<Vec<_>>()
        );

        assert!(
            pixels.windows(2).all(|w| w[0] <= w[1]),
            "expected output columns to be non-decreasing (input fills strictly \
             increase per column: {:?}), got {pixels:?} -- suggests columns are \
             cross-contaminating rather than independent",
            (0..shape.m).map(col_fill).collect::<Vec<_>>()
        );
        assert!(
            pixels[0] != pixels[shape.m as usize - 1],
            "first and last output columns are identical ({}) despite very \
             different input fills ({} vs {}) -- suggests the op isn't really \
             reading per-column input",
            pixels[0],
            col_fill(0),
            col_fill(shape.m - 1)
        );
    }
}
