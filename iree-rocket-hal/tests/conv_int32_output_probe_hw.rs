//! Hardware probe for the DPU's documented int32 convolution output mode.
//!
//! `ConvInteger` needs an exact i32 accumulator, while Rocket's validated
//! int8 path requantizes to i8. The DPU documents `out_precision = 4` as
//! int32, but no vendor capture in this repository exercises it. This probe
//! establishes the required combination: retain int8 input and processing
//! precision, select int32 output, and bypass the BS and CPEND stages that
//! otherwise requantize the accumulator.
//!
//! Cross-build and run on an RK3588 board:
//!
//! ```text
//! CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!   cargo test --target aarch64-unknown-linux-gnu --release \
//!     --test conv_int32_output_probe_hw --no-run
//! ./conv_int32_output_probe_hw-<hash> --ignored --nocapture
//! ```

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    conv::{
        BsEntry, Buffers, Kernels, Multiplier, Precision, Quantization, Shape, Tile, conv_2d_tile,
        relocate, write_bs_buffer,
    },
    device::{Buffer, JobDesc, close_bo, fini_bo, prep_bo, submit_jobs},
    tensor_layout::pack_hwcf_to_rocket_weights,
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const PAGE_BYTES: usize = 4096;
const WIDTH: u32 = 4;
const HEIGHT: u32 = 4;
const OUT_CHANNELS: u32 = 8;
const FEATURE_ATOM_BYTES: usize = 16;
const I32_CHANNELS_PER_ATOM: usize = FEATURE_ATOM_BYTES / mem::size_of::<i32>();
const OUTPUT_PIXELS: usize = WIDTH as usize * HEIGHT as usize;
const OUTPUT_SURFACES: usize = (OUT_CHANNELS as usize).div_ceil(I32_CHANNELS_PER_ATOM);
const EXPECTED_OUTPUT_BYTES: usize = OUTPUT_PIXELS * OUTPUT_SURFACES * FEATURE_ATOM_BYTES;
const OUTPUT_BYTES: usize = 64 * 1024;
const SENTINEL: u8 = 0xa5;

fn page_aligned_size(size: usize) -> usize {
    size.div_ceil(PAGE_BYTES) * PAGE_BYTES
}

fn changed_extent(bytes: &[u8]) -> Option<(usize, usize)> {
    let first = bytes.iter().position(|&byte| byte != SENTINEL)?;
    let last = bytes.iter().rposition(|&byte| byte != SENTINEL)?;
    Some((first, last + 1))
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn int8_convolution_can_write_exact_i32_accumulators() {
    let kernels: Kernels = [1, 1];
    let precision = Precision::Int8Accumulator(Quantization {
        input_zero_point: 0,
        output_zero_point: 0,
        weight_zero_point: 0,
        input_scale: 1.0,
        weights_scale: 1.0,
        multiplier: Multiplier::from_ratio(1.0),
    });
    let shape = Shape::with_precision(WIDTH, HEIGHT, 1, 1, OUT_CHANNELS, precision);
    let input_values: Vec<u8> = (1..=OUTPUT_PIXELS as u8).collect();
    let weight_values: [i8; OUT_CHANNELS as usize] = [-1, -2, -3, -4, -5, -6, -7, -8];

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();

    let raw_output = unsafe {
        let input_bytes = (WIDTH * HEIGHT) as usize;
        let buf_input = Buffer::new(fd, page_aligned_size(input_bytes), &file);
        ptr::write_bytes(buf_input.host_ptr, 0, buf_input.size);
        ptr::copy_nonoverlapping(
            input_values.as_ptr(),
            buf_input.host_ptr,
            input_values.len(),
        );

        let weight_bytes = shape.weight_bytes(kernels) as usize;
        let buf_weights = Buffer::new(fd, page_aligned_size(weight_bytes), &file);
        ptr::write_bytes(buf_weights.host_ptr, 0, buf_weights.size);
        let dense_weights: Vec<u8> = weight_values.iter().map(|&value| value as u8).collect();
        let mut packed_weights = vec![0; weight_bytes];
        let packed_bytes = pack_hwcf_to_rocket_weights(
            &dense_weights,
            1,
            1,
            1,
            OUT_CHANNELS as usize,
            1,
            &mut packed_weights,
        )
        .expect("failed to pack int8 weights");
        assert_eq!(packed_bytes, weight_bytes);
        ptr::copy_nonoverlapping(
            packed_weights.as_ptr(),
            buf_weights.host_ptr,
            packed_weights.len(),
        );

        let bs_bytes = shape.bs_buffer_bytes();
        let buf_bs = Buffer::new(fd, page_aligned_size(bs_bytes), &file);
        ptr::write_bytes(buf_bs.host_ptr, 0, buf_bs.size);
        let entries = vec![BsEntry::default(); shape.padded_out_channels() as usize];
        write_bs_buffer(
            std::slice::from_raw_parts_mut(buf_bs.host_ptr, buf_bs.size),
            &entries,
        );

        // Deliberately oversized: the validated i8 path writes far less, but
        // int32 may alter the hardware's channel/surface byte stride.
        let buf_output = Buffer::new(fd, OUTPUT_BYTES, &file);
        ptr::write_bytes(buf_output.host_ptr, SENTINEL, buf_output.size);

        let mut commands = conv_2d_tile(shape, kernels, &Tile::whole(shape, kernels));
        relocate(
            &mut commands,
            Buffers {
                input: buf_input.dma_address,
                weights: buf_weights.dma_address,
                bias: buf_bs.dma_address,
                output: buf_output.dma_address,
            },
        );

        let command_bytes = commands.len() * mem::size_of::<u64>();
        let buf_commands = Buffer::new(fd, page_aligned_size(command_bytes), &file);
        ptr::write_bytes(buf_commands.host_ptr, 0, buf_commands.size);
        let words =
            std::slice::from_raw_parts_mut(buf_commands.host_ptr as *mut u64, commands.len());
        for (destination, command) in words.iter_mut().zip(&commands) {
            *destination = command.0;
        }

        for handle in [
            buf_input.handle,
            buf_weights.handle,
            buf_bs.handle,
            buf_output.handle,
            buf_commands.handle,
        ] {
            fini_bo(fd, handle).expect("failed to sync BO for the NPU");
        }

        let tasks = [(buf_commands.dma_address, commands.len() as u32)];
        let in_handles = [
            buf_commands.handle,
            buf_input.handle,
            buf_weights.handle,
            buf_bs.handle,
        ];
        let out_handles = [buf_output.handle];
        let jobs = [JobDesc {
            tasks: &tasks,
            in_handles: &in_handles,
            out_handles: &out_handles,
        }];
        submit_jobs(fd, &jobs).expect("SUBMIT failed");
        prep_bo(fd, buf_output.handle, 5_000_000_000).expect("job did not complete");

        let output = std::slice::from_raw_parts(buf_output.host_ptr, OUTPUT_BYTES).to_vec();
        for handle in [
            buf_input.handle,
            buf_weights.handle,
            buf_bs.handle,
            buf_output.handle,
            buf_commands.handle,
        ] {
            let _ = close_bo(fd, handle);
        }
        output
    };

    let (first, end) = changed_extent(&raw_output).expect("NPU did not modify the output buffer");
    println!(
        "changed output byte extent: {first}..{end} ({} bytes)",
        end - first
    );

    assert_eq!(
        (first, end),
        (0, EXPECTED_OUTPUT_BYTES),
        "unexpected int32 output write extent"
    );

    let surface_stride = OUTPUT_PIXELS * FEATURE_ATOM_BYTES;
    let mut mismatches = Vec::new();
    for (pixel, &input) in input_values.iter().enumerate() {
        for (channel, &weight) in weight_values.iter().enumerate() {
            let surface = channel / I32_CHANNELS_PER_ATOM;
            let lane = channel % I32_CHANNELS_PER_ATOM;
            let offset = surface * surface_stride
                + pixel * FEATURE_ATOM_BYTES
                + lane * mem::size_of::<i32>();
            let value = i32::from_le_bytes(raw_output[offset..offset + 4].try_into().unwrap());
            let expected = i32::from(input) * i32::from(weight);
            if pixel < 2 {
                println!(
                    "  pixel {pixel:02}, channel {channel}: output[{offset:04x}] = {value:4} (expected {expected:4})"
                );
            }
            if value != expected && mismatches.len() < 16 {
                mismatches.push((pixel, channel, offset, expected, value));
            }
        }
    }
    assert!(
        mismatches.is_empty(),
        "int32 output did not preserve exact accumulators in NC1HWC2 surface order; first mismatches: {mismatches:?}"
    );
}
