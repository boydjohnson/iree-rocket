//! Measures the int8 requantisation gain instead of assuming it.
//!
//!   CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!     cargo test --target aarch64-unknown-linux-gnu --release \
//!       --test conv_int8_probe_hw --no-run
//!
//!   ./conv_int8_probe_hw-<hash> --ignored --nocapture
//!
//! # Why
//!
//! `conv_int8_hw` returned a saturated 127 in all 27 configurations,
//! including `Cin` 1 with a 1x1 kernel where the accumulator is exactly 1.
//! A uniform saturation independent of the accumulator is a constant gain
//! error, not a layout error, so the useful thing to know is the constant.
//!
//! The datapath is assumed to be
//!
//!     output = accumulator * (bs_multiplier / 2^a) * (cvt_scale / 2^shift)
//!
//! with `a` unknown. The builder currently assumes `a = 14`, from
//! `DPU_BS_MUL_CFG.bs_mul_shift_value` reading 14 in every capture and the
//! BS multiplier plane normalising to `2^14`.
//!
//! Holding the accumulator at 1, the multiplier at `2^14` and `cvt_scale` at
//! `2^14`, the output is `2^(28 - a - shift)`. Sweeping `shift` and finding
//! where the output reaches 1 measures `a` directly: the crossing sits at
//! `shift = 28 - a`. So a crossing at 14 confirms the current assumption, at
//! 28 says the BS stage applies no shift of its own, and at 21 says
//! something is reading the constant-128 plane as the multiplier.
//!
//! The second sweep does the same over the BS multiplier at a fixed shift,
//! which distinguishes "the multiplier is being read" from "the multiplier
//! is being ignored" -- if the output does not move with it, the BS operand
//! is not reaching the multiply at all and the whole plane is misplaced.

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    builders::{
        RegCmd, RegisterMeta,
        cna::{CnaDcompAddr0, CnaFeatureDataAddr},
        dpu::DpuDstBaseAddr,
        dpu_rdma::DpuRdmaBsBaseAddr,
    },
    conv::{
        BsEntry, Kernels, Multiplier, Precision, Quantization, Shape, Tile, conv_2d_tile,
        write_bs_buffer,
    },
    device::{Buffer, JobDesc, close_bo, fini_bo, prep_bo, submit_jobs},
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const FEATURE_ATOM_BYTES: usize = 16;
const PAGE_BYTES: usize = 4096;
const WIDTH: u32 = 8;
const HEIGHT: u32 = 8;
const OUT_CHANNELS: u32 = 8;

fn page_aligned_size(size: usize) -> usize {
    size.div_ceil(PAGE_BYTES) * PAGE_BYTES
}

fn decode_identity(command: &RegCmd) -> (u32, u32) {
    ((command.0 >> 48) as u32, command.0 as u32 & 0xffff)
}

fn relocate<R: RegisterMeta>(commands: &mut [RegCmd], address: u32) {
    let matches: Vec<_> = commands
        .iter()
        .enumerate()
        .filter_map(|(index, command)| {
            (decode_identity(command) == (R::DOMAIN, R::OFFSET)).then_some(index)
        })
        .collect();
    assert_eq!(matches.len(), 1, "expected exactly one relocation site");
    let tile_offset = (commands[matches[0]].0 >> 16) as u32;
    commands[matches[0]] = RegCmd::new(R::DOMAIN, R::OFFSET, address + tile_offset);
}

/// One run at `Cin` 1 with a 1x1 kernel, so the accumulator is exactly 1 at
/// every output pixel. Returns the value at an interior pixel.
fn probe(bs_multiplier: i16, cvt_scale: u32, cvt_shift: u32) -> i32 {
    let kernels: Kernels = [1, 1];
    let precision = Precision::Int8(Quantization {
        input_zero_point: 0,
        output_zero_point: 0,
        multiplier: Multiplier {
            scale: cvt_scale,
            shift: cvt_shift,
        },
    });
    let shape = Shape::with_precision(WIDTH, HEIGHT, 1, 1, OUT_CHANNELS, precision);
    let width = WIDTH as usize;
    let height = HEIGHT as usize;
    // Dense at Cin 1: one byte per pixel.
    let input_bytes = width * height;
    let out_surfaces = (shape.padded_out_channels() as usize).div_ceil(16);
    let output_bytes = out_surfaces * width * height * FEATURE_ATOM_BYTES;

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let buf_input = Buffer::new(fd, page_aligned_size(input_bytes), &file);
        ptr::write_bytes(buf_input.host_ptr, 0, buf_input.size);
        ptr::write_bytes(buf_input.host_ptr, 1, input_bytes);

        let weight_bytes = shape.weight_bytes(kernels) as usize;
        let buf_weights = Buffer::new(fd, page_aligned_size(weight_bytes), &file);
        ptr::write_bytes(buf_weights.host_ptr, 0, buf_weights.size);
        ptr::write_bytes(buf_weights.host_ptr, 1, weight_bytes);

        let bs_bytes = shape.bs_buffer_bytes();
        let buf_bs = Buffer::new(fd, page_aligned_size(bs_bytes), &file);
        ptr::write_bytes(buf_bs.host_ptr, 0, buf_bs.size);
        let entries = vec![
            BsEntry {
                bias: 0,
                multiplier: bs_multiplier,
            };
            shape.padded_out_channels() as usize
        ];
        write_bs_buffer(
            std::slice::from_raw_parts_mut(buf_bs.host_ptr, buf_bs.size),
            &entries,
        );

        let buf_output = Buffer::new(fd, page_aligned_size(output_bytes), &file);
        ptr::write_bytes(buf_output.host_ptr, 0, buf_output.size);

        let mut commands = conv_2d_tile(shape, kernels, &Tile::whole(shape, kernels));
        relocate::<CnaFeatureDataAddr>(&mut commands, buf_input.dma_address);
        relocate::<CnaDcompAddr0>(&mut commands, buf_weights.dma_address);
        relocate::<DpuRdmaBsBaseAddr>(&mut commands, buf_bs.dma_address);
        relocate::<DpuDstBaseAddr>(&mut commands, buf_output.dma_address);

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

        // An interior pixel, channel 0.
        let raw = std::slice::from_raw_parts(buf_output.host_ptr as *const i8, output_bytes);
        let value = i32::from(raw[(2 * width + 2) * FEATURE_ATOM_BYTES]);

        for handle in [
            buf_input.handle,
            buf_weights.handle,
            buf_bs.handle,
            buf_output.handle,
            buf_commands.handle,
        ] {
            let _ = close_bo(fd, handle);
        }
        value
    }
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn measures_the_int8_requantisation_gain() {
    const UNIT: i16 = 1 << 14;
    const MANTISSA: u32 = 1 << 14;

    println!("accumulator is exactly 1 (Cin 1, 1x1 kernel, all inputs and weights 1)");
    println!("bs_multiplier = 2^14, cvt_scale = 2^14, sweeping cvt_shift\n");
    println!(
        "  {:>5}  {:>8}   expected if the BS stage shifts by...",
        "shift", "output"
    );
    println!(
        "  {:>5}  {:>8}   {:>6} {:>6} {:>6}",
        "", "", "a=14", "a=7", "a=0"
    );

    let mut crossing = None;
    for shift in 10..=34u32 {
        let got = probe(UNIT, MANTISSA, shift);
        let predict = |a: i32| {
            let exponent = 28 - a - shift as i32;
            if exponent >= 0 {
                (1i64 << exponent.min(20)).min(127).to_string()
            } else {
                "0".to_string()
            }
        };
        println!(
            "  {shift:>5}  {got:>8}   {:>6} {:>6} {:>6}",
            predict(14),
            predict(7),
            predict(0)
        );
        if got == 1 && crossing.is_none() {
            crossing = Some(shift);
        }
    }

    match crossing {
        Some(shift) => {
            let implied = 28 - shift as i32;
            println!(
                "\noutput reaches 1 at cvt_shift {shift}, implying the BS stage \
                 applies a shift of {implied}"
            );
        }
        None => println!("\noutput never reached 1 -- the gain is outside the swept range"),
    }

    println!("\nnow sweeping bs_multiplier at cvt_shift 14, to confirm the BS");
    println!("operand reaches the multiply at all:\n");
    println!("  {:>10}  {:>8}", "bs_mul", "output");
    let mut moved = false;
    let baseline = probe(UNIT, MANTISSA, 14);
    for exponent in 0..=14u32 {
        let multiplier = (1i32 << exponent) as i16;
        let got = probe(multiplier, MANTISSA, 14);
        println!("  {multiplier:>10}  {got:>8}");
        if got != baseline {
            moved = true;
        }
    }
    if !moved {
        println!("\nthe output does not move with bs_multiplier at all -- the BS");
        println!("operand is not reaching the multiply, so the plane is misplaced.");
    }
}
