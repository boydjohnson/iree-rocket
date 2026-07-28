//! RK3588 validation for the capture-derived Phase 3 FC lowering.
//!
//! These tests exercise the vendor mapping that the old Mesa-backed FC path
//! could not: physical height is one, not four. Cross-compile and run with:
//!
//! ```text
//! CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!   cargo test --target aarch64-unknown-linux-gnu --release \
//!     --test fc_phase3_hw --no-run
//! ./fc_phase3_hw-<hash> --ignored --nocapture
//! ```

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
        assert_eq!(
            programs.len(),
            1,
            "small FC validation shape must be one job"
        );
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

#[test]
#[ignore = "needs /dev/accel/accel0 on an RK3588"]
fn fp16_height_one_fc_runs_on_npu() {
    let shape = Shape::new(7, 16, 32, Precision::Fp16);
    let input = encode_fp16(
        (0..shape.m).flat_map(|m| std::iter::repeat_n((m + 1) as f32, shape.k as usize)),
    );
    let weights = encode_fp16(std::iter::repeat_n(
        1.0 / shape.k as f32,
        shape.k as usize * shape.n as usize,
    ));
    let output = run_fc(shape, &input, &weights);
    for (m, row) in output.chunks_exact(shape.n as usize * 2).enumerate() {
        for (n, value) in row.chunks_exact(2).enumerate() {
            let actual = f16_to_f32(u16::from_le_bytes([value[0], value[1]]));
            assert_eq!(actual, (m + 1) as f32, "FC [{m}, {n}]");
        }
    }
}

#[test]
#[ignore = "needs /dev/accel/accel0 on an RK3588"]
fn odd_int8_n_height_one_fc_runs_on_npu() {
    let shape = Shape::new(
        7,
        16,
        33,
        Precision::Int8(Quantization {
            input_zero_point: 0,
            output_zero_point: 0,
            multiplier: Multiplier::for_unit_bs(1.0),
        }),
    );
    let input = vec![1; shape.m as usize * shape.k as usize];
    let weights = vec![1; shape.k as usize * shape.n as usize];
    let output = run_fc(shape, &input, &weights);
    assert!(
        output.iter().all(|&value| value == shape.k as u8),
        "expected every FC output to equal K={}, got {output:?}",
        shape.k
    );
}
