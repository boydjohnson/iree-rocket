//! Raw RK3588 probe for the int8 depthwise accumulator path.
//!
//! This intentionally bypasses the driver compaction layer. It makes the
//! channel/tap layout observable before comparing it with the dense ABI.

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    conv::{
        BsEntry, Buffers, Multiplier, Precision, Quantization, Shape, Tile, conv_2d_tile, relocate,
        write_bs_buffer,
    },
    device::{Buffer, JobDesc, close_bo, fini_bo, prep_bo, submit_jobs},
    tensor_layout::{pack_depthwise_to_rocket_weights, pack_nhwc_to_nc1hwc2},
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const PAGE_BYTES: usize = 4096;
const WIDTH: usize = 34;
const HEIGHT: usize = 34;
const CHANNELS: usize = 64;
const OUT_WIDTH: usize = 32;
const OUT_HEIGHT: usize = 32;
const BLOCK_CHANNELS: usize = 32;
const BLOCK_BYTES: usize = BLOCK_CHANNELS * mem::size_of::<i32>();
const OUTPUT_BYTES: usize = 1024 * 1024;
const SENTINEL: u8 = 0xa5;

fn page_aligned_size(size: usize) -> usize {
    size.div_ceil(PAGE_BYTES) * PAGE_BYTES
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn int8_depthwise_raw_accumulator_layout() {
    let precision = Precision::Int8Accumulator(Quantization {
        input_zero_point: 0,
        output_zero_point: 0,
        weight_zero_point: 0,
        input_scale: 1.0,
        weights_scale: 1.0,
        multiplier: Multiplier::from_ratio(1.0),
    });
    let shape = Shape::with_precision(
        WIDTH as u32,
        HEIGHT as u32,
        1,
        CHANNELS as u32,
        CHANNELS as u32,
        precision,
    )
    .with_depthwise();
    let kernels = [3, 3];
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .unwrap();
    let fd = file.as_raw_fd();

    unsafe {
        let input_bytes = WIDTH * HEIGHT * CHANNELS;
        let input = Buffer::new(fd, page_aligned_size(input_bytes), &file);
        let dense_input: Vec<u8> = (0..HEIGHT * WIDTH)
            .flat_map(|_| (0..CHANNELS).map(|channel| (channel + 1) as u8))
            .collect();
        let mut packed_input = vec![0; input_bytes];
        pack_nhwc_to_nc1hwc2(&dense_input, WIDTH * HEIGHT, CHANNELS, &mut packed_input).unwrap();
        ptr::copy_nonoverlapping(packed_input.as_ptr(), input.host_ptr, packed_input.len());

        let weight_bytes = shape.weight_bytes(kernels) as usize;
        let weights = Buffer::new(fd, page_aligned_size(weight_bytes), &file);
        let dense_weights: Vec<u8> = (0..CHANNELS)
            .flat_map(|_| std::iter::repeat_n(1u8, 9))
            .collect();
        let mut packed_weights = vec![0; weight_bytes];
        pack_depthwise_to_rocket_weights(
            &dense_weights,
            3,
            3,
            CHANNELS,
            shape.depthwise_padded_channels() as usize,
            1,
            &mut packed_weights,
        )
        .unwrap();
        ptr::copy_nonoverlapping(
            packed_weights.as_ptr(),
            weights.host_ptr,
            packed_weights.len(),
        );

        let bs = Buffer::new(fd, page_aligned_size(shape.bs_buffer_bytes()), &file);
        ptr::write_bytes(bs.host_ptr, 0, bs.size);
        write_bs_buffer(
            std::slice::from_raw_parts_mut(bs.host_ptr, bs.size),
            &vec![BsEntry::default(); shape.padded_out_channels() as usize],
        );

        let output = Buffer::new(fd, OUTPUT_BYTES, &file);
        ptr::write_bytes(output.host_ptr, SENTINEL, output.size);
        let mut commands = conv_2d_tile(shape, kernels, &Tile::whole(shape, kernels));
        relocate(
            &mut commands,
            Buffers {
                input: input.dma_address,
                weights: weights.dma_address,
                bias: bs.dma_address,
                output: output.dma_address,
            },
        );
        let command_buffer = Buffer::new(
            fd,
            page_aligned_size(commands.len() * mem::size_of::<u64>()),
            &file,
        );
        let words =
            std::slice::from_raw_parts_mut(command_buffer.host_ptr as *mut u64, commands.len());
        for (word, command) in words.iter_mut().zip(&commands) {
            *word = command.0;
        }
        for handle in [
            input.handle,
            weights.handle,
            bs.handle,
            output.handle,
            command_buffer.handle,
        ] {
            fini_bo(fd, handle).unwrap();
        }
        let tasks = [(command_buffer.dma_address, commands.len() as u32)];
        let input_handles = [
            command_buffer.handle,
            input.handle,
            weights.handle,
            bs.handle,
        ];
        let output_handles = [output.handle];
        submit_jobs(
            fd,
            &[JobDesc {
                tasks: &tasks,
                in_handles: &input_handles,
                out_handles: &output_handles,
            }],
        )
        .unwrap();
        prep_bo(fd, output.handle, 5_000_000_000).unwrap();
        let raw = std::slice::from_raw_parts(output.host_ptr, OUTPUT_BYTES);
        let extent = raw
            .iter()
            .position(|&byte| byte != SENTINEL)
            .unwrap_or(OUTPUT_BYTES);
        let end = raw
            .iter()
            .rposition(|&byte| byte != SENTINEL)
            .map_or(0, |i| i + 1);
        println!(
            "changed extent: {extent}..{end} ({} bytes)",
            end.saturating_sub(extent)
        );
        for index in 0..64 {
            let offset = index * 4;
            let value = i32::from_le_bytes(raw[offset..offset + 4].try_into().unwrap());
            print!(" {value}");
        }
        println!();
        println!("expected channel values: {} {} {}", 9, 18, 27);
        println!(
            "native block bytes={BLOCK_BYTES}, expected output bytes={}",
            OUT_WIDTH * OUT_HEIGHT * 2 * BLOCK_BYTES
        );
        for handle in [
            input.handle,
            weights.handle,
            bs.handle,
            output.handle,
            command_buffer.handle,
        ] {
            let _ = close_bo(fd, handle);
        }
    }
}
