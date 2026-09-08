//! A residual block's tail on the NPU: a multi-tile fp16 convolution writes
//! its output cube, then one EW task adds a second tensor to that cube and
//! applies the EW core's own ReLU, all without the host touching the
//! intermediate. This is ResNet50's `relu(conv3(x) + skip)`, the edge
//! ISSUES.md P2 sized as the largest remaining lever on that model.
//!
//! Two things this settles that `conv_with_add_hw` could not: the conv is
//! **two CBUF row tiles** (56x56x64 -> 256, ResNet50's own shape), so the EW
//! task reads a cube assembled by several jobs, not one program's output;
//! and `DPU_EW_CFG.ew_relu_bypass = 0` is driven for the first time. Every
//! operand varies per pixel and per channel and every value is a small
//! integer, so the comparison is exact.
//!
//! The fp16 output cube and the fp16 feature cube share one layout here --
//! `output_offset` and `feature_offset` are the same formula -- which is
//! what lets the EW task consume the conv's scratch directly.

mod support {
    pub mod conv2d_oracle;
}

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    conv::{Buffers, ConvPlan, Kernels, Shape},
    device::{Buffer, JobDesc, close_bo, fini_bo, prep_bo, submit_jobs},
    elementwise::{EwAddBuffers, EwAddShape, EwBinaryOp, EwPrecision, build_add_regcmd_with_relu},
    tensor_layout::{pack_hwcf_to_rocket_weights, rocket_weight_storage_size},
};
use support::conv2d_oracle::{
    f16_to_f32, f32_to_f16, feature_offset, input_storage_bytes, output_storage_bytes,
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const WIDTH: u32 = 56;
const HEIGHT: u32 = 56;
const CIN: usize = 64;
const COUT: usize = 256;
const KERNELS: Kernels = [1, 1];
const FP16_BYTES: usize = 2;
const ATOM_BYTES: usize = 16;

fn page_aligned(size: usize) -> usize {
    size.next_multiple_of(4096)
}

fn input_value(y: usize, x: usize, c: usize) -> i32 {
    ((y * 7 + x * 3 + c * 5) % 7) as i32 - 3
}
fn weight_value(c: usize, o: usize) -> i32 {
    ((c * 3 + o * 5) % 5) as i32 - 2
}
fn bias_value(o: usize) -> i32 {
    (o % 5) as i32 - 2
}
fn residual_value(y: usize, x: usize, o: usize) -> i32 {
    ((y * 5 + x * 11 + o * 3) % 9) as i32 - 4
}

/// NC1HWC2 byte offset of an fp16 element in a `width` x `height` cube.
fn cube_offset(width: usize, height: usize, channel: usize, y: usize, x: usize) -> usize {
    (channel / 8) * width * height * ATOM_BYTES
        + (y * width + x) * ATOM_BYTES
        + (channel % 8) * FP16_BYTES
}

fn run(relu: bool) -> Result<usize, String> {
    let shape = Shape::with_out_channels(WIDTH, HEIGHT, 1, CIN as u32, COUT as u32);
    let width = WIDTH as usize;
    let height = HEIGHT as usize;
    let plan = ConvPlan::new(shape, KERNELS);
    let tiles = plan.tiles().len();
    assert!(
        tiles >= 2,
        "this test wants a multi-tile conv; the planner gave {tiles} tile(s)"
    );

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();
    unsafe {
        // Input feature cube.
        let input_bytes = input_storage_bytes(shape);
        let buf_input = Buffer::new(fd, page_aligned(input_bytes), &file);
        ptr::write_bytes(buf_input.host_ptr, 0, buf_input.size);
        let input = std::slice::from_raw_parts_mut(buf_input.host_ptr, buf_input.size);
        for y in 0..height {
            for x in 0..width {
                for c in 0..CIN {
                    let offset = feature_offset(shape, c, y, x);
                    input[offset..offset + FP16_BYTES]
                        .copy_from_slice(&f32_to_f16(input_value(y, x, c) as f32).to_le_bytes());
                }
            }
        }
        // Weights, packed from dense HWCF.
        let mut dense = Vec::with_capacity(CIN * COUT * FP16_BYTES);
        for c in 0..CIN {
            for o in 0..COUT {
                dense.extend_from_slice(&f32_to_f16(weight_value(c, o) as f32).to_le_bytes());
            }
        }
        let weight_bytes =
            rocket_weight_storage_size(1, 1, CIN, COUT, FP16_BYTES).expect("weight storage size");
        let buf_weights = Buffer::new(fd, page_aligned(weight_bytes), &file);
        ptr::write_bytes(buf_weights.host_ptr, 0, buf_weights.size);
        pack_hwcf_to_rocket_weights(
            &dense,
            1,
            1,
            CIN,
            COUT,
            FP16_BYTES,
            std::slice::from_raw_parts_mut(buf_weights.host_ptr, weight_bytes),
        )
        .expect("weight packing failed");
        // Bias: one f32 per padded output channel.
        let bias_bytes = shape.padded_out_channels() as usize * 4;
        let buf_bias = Buffer::new(fd, page_aligned(bias_bytes), &file);
        ptr::write_bytes(buf_bias.host_ptr, 0, buf_bias.size);
        let bias = std::slice::from_raw_parts_mut(
            buf_bias.host_ptr as *mut f32,
            shape.padded_out_channels() as usize,
        );
        for (o, slot) in bias.iter_mut().enumerate().take(COUT) {
            *slot = bias_value(o) as f32;
        }
        // The conv's output cube: the EW task's primary operand, never
        // touched by the host.
        let mid_bytes = output_storage_bytes(shape, KERNELS);
        let buf_mid = Buffer::new(fd, page_aligned(mid_bytes), &file);
        ptr::write_bytes(buf_mid.host_ptr, 0, buf_mid.size);
        // The residual, as a feature cube of the output geometry.
        let cube_bytes = COUT.div_ceil(8) * width * height * ATOM_BYTES;
        let buf_residual = Buffer::new(fd, page_aligned(cube_bytes), &file);
        ptr::write_bytes(buf_residual.host_ptr, 0, buf_residual.size);
        let residual = std::slice::from_raw_parts_mut(buf_residual.host_ptr, buf_residual.size);
        for y in 0..height {
            for x in 0..width {
                for o in 0..COUT {
                    let offset = cube_offset(width, height, o, y, x);
                    residual[offset..offset + FP16_BYTES]
                        .copy_from_slice(&f32_to_f16(residual_value(y, x, o) as f32).to_le_bytes());
                }
            }
        }
        let buf_out = Buffer::new(fd, page_aligned(cube_bytes), &file);
        ptr::write_bytes(buf_out.host_ptr, 0, buf_out.size);

        // One program per CBUF tile, each its own job, all on this fd so the
        // kernel runs them in order; then the EW task, after them.
        let programs = plan.programs_with_buffers(Buffers {
            input: buf_input.dma_address,
            weights: buf_weights.dma_address,
            bias: buf_bias.dma_address,
            output: buf_mid.dma_address,
        });
        let add = EwAddShape {
            width: WIDTH,
            height: HEIGHT,
            channels: COUT as u32,
            precision: EwPrecision::Fp16,
            op: EwBinaryOp::Add,
            output_zero_point: 0,
            w_cvt_offset: 0,
            w_scale_ratio: 1.0,
            output_scale_ratio: 1.0,
        };
        let add_cmds = build_add_regcmd_with_relu(
            &add,
            &EwAddBuffers {
                intermediate_addr: buf_mid.dma_address,
                w_addr: buf_residual.dma_address,
                output_addr: buf_out.dma_address,
            },
            relu,
        );
        let mut cmd_bufs = Vec::new();
        for cmds in programs.iter().chain(std::iter::once(&add_cmds)) {
            let bytes = cmds.len() * mem::size_of::<u64>();
            let buf = Buffer::new(fd, page_aligned(bytes), &file);
            let words = std::slice::from_raw_parts_mut(buf.host_ptr as *mut u64, cmds.len());
            for (dst, cmd) in words.iter_mut().zip(cmds) {
                *dst = cmd.0;
            }
            cmd_bufs.push((buf, cmds.len() as u32));
        }
        let mut all_handles = vec![
            buf_input.handle,
            buf_weights.handle,
            buf_bias.handle,
            buf_mid.handle,
            buf_residual.handle,
            buf_out.handle,
        ];
        all_handles.extend(cmd_bufs.iter().map(|(b, _)| b.handle));
        for &handle in &all_handles {
            fini_bo(fd, handle).expect("failed to sync BO for the NPU");
        }
        let tile_tasks: Vec<[(u32, u32); 1]> = cmd_bufs[..tiles]
            .iter()
            .map(|(b, n)| [(b.dma_address, *n)])
            .collect();
        let tile_ins: Vec<[u32; 4]> = cmd_bufs[..tiles]
            .iter()
            .map(|(b, _)| {
                [
                    b.handle,
                    buf_input.handle,
                    buf_weights.handle,
                    buf_bias.handle,
                ]
            })
            .collect();
        let mid_out = [buf_mid.handle];
        let ew_task = [(cmd_bufs[tiles].0.dma_address, cmd_bufs[tiles].1)];
        let ew_ins = [
            cmd_bufs[tiles].0.handle,
            buf_mid.handle,
            buf_residual.handle,
        ];
        let ew_out = [buf_out.handle];
        let mut jobs: Vec<JobDesc> = tile_tasks
            .iter()
            .zip(&tile_ins)
            .map(|(tasks, ins)| JobDesc {
                tasks,
                in_handles: ins,
                out_handles: &mid_out,
            })
            .collect();
        jobs.push(JobDesc {
            tasks: &ew_task,
            in_handles: &ew_ins,
            out_handles: &ew_out,
        });
        submit_jobs(fd, &jobs).expect("SUBMIT failed");
        prep_bo(fd, buf_out.handle, 5_000_000_000).expect("job did not complete");

        let raw = std::slice::from_raw_parts(buf_out.host_ptr, cube_bytes);
        let mut mismatches = 0usize;
        let mut samples = Vec::new();
        for y in 0..height {
            for x in 0..width {
                for o in 0..COUT {
                    let acc: i32 = (0..CIN)
                        .map(|c| input_value(y, x, c) * weight_value(c, o))
                        .sum();
                    let sum = acc + bias_value(o) + residual_value(y, x, o);
                    let want = if relu { sum.max(0) } else { sum } as f32;
                    let offset = cube_offset(width, height, o, y, x);
                    let got = f16_to_f32(u16::from_le_bytes([raw[offset], raw[offset + 1]]));
                    if got != want {
                        mismatches += 1;
                        if samples.len() < 8 {
                            samples.push(format!(
                                "y {y} x {x} o {o}: acc {acc} bias {} residual {} want {want} got {got}",
                                bias_value(o),
                                residual_value(y, x, o)
                            ));
                        }
                    }
                }
            }
        }
        for &handle in &all_handles {
            let _ = close_bo(fd, handle);
        }
        if mismatches == 0 {
            Ok(tiles)
        } else {
            Err(format!(
                "{mismatches} of {} mismatches (relu {relu}, {tiles} tiles):\n  {}",
                width * height * COUT,
                samples.join("\n  ")
            ))
        }
    }
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64, copy to the board, run there"]
fn multi_tile_conv_then_residual_add() {
    let tiles = run(false).unwrap_or_else(|e| panic!("{e}"));
    println!("conv 56x56x64->256 in {tiles} tiles, then EW add: exact");
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64, copy to the board, run there"]
fn multi_tile_conv_then_residual_add_relu() {
    let tiles = run(true).unwrap_or_else(|e| panic!("{e}"));
    println!("conv 56x56x64->256 in {tiles} tiles, then EW add + EW relu: exact");
}
