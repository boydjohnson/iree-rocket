//! A 3x3 depthwise convolution with the CNA doing its pad-1 "same" padding,
//! at every depthwise shape MobileNetV2 runs: exact against an f32
//! reference of the same fp16 values, with the NPU time printed.
//!
//! The compiler folds a model's `tensor.pad` into a depthwise convolution
//! only if this holds. Every padded convolution measured on hardware before
//! it was dense (`rocket_dynamic_pad1_*`). Depthwise is a different register
//! program (`CONV_MODE = 3`, `DW_EN`, tap-major coefficients) whose row and
//! column tiling the planner shares with dense, so "dense pad 1 is exact"
//! is no evidence for it. The probe pattern varies with every index, so a
//! window shifted by the padding, or a tile edge that drops the border, shows
//! up as an error, not as a coincidentally equal value.
//!
//!     ./depthwise_pad1_hw-<hash> --ignored --nocapture --test-threads=1
//!
//! `ROCKET_PAD1_SHAPES=56:144:1,56:128:1` runs other `side:channels:stride`
//! shapes, and `ROCKET_CBUF_SPLIT=<data>/<weight>` forces the CBUF split --
//! which tells a capacity fault (exact with fewer rows) from a geometry one.
//! `ROCKET_DW_PAD=0` runs the same shapes unpadded.
//!
//! **Size the output from `Shape::padded_out_channels`, not `Cin`.** A
//! depthwise output granule is 32 fp16 channels, so the DPU writes 160
//! channels' worth of surfaces for `Cin` 144. A cube sized for 144 is
//! overrun, and the job hangs with most of the output unwritten -- at every
//! `Cin % 32 != 0`, padded or not, which reads exactly like a padding or
//! channel-count hardware limit. This file's first version did that and
//! briefly "found" one; the 7x7 probe it was copied from has the same sizing
//! and only ever ran multiples of 32.

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr, time::Instant};

use iree_rocket_hal::rocket::{
    conv::{Buffers, ConvPlan, Precision, Shape},
    device::{Buffer, JobDesc, close_bo, fini_bo, prep_bo, submit_jobs},
    tensor_layout::{nc1hwc2_storage_size, pack_nhwc_to_nc1hwc2},
    weights::WeightPlan,
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const PAGE_BYTES: usize = 4096;
const ATOM: usize = 16;
const SENTINEL: u8 = 0xa5;
const KERNEL: usize = 3;

fn f32_to_f16_bits(value: f32) -> u16 {
    if value == 0.0 {
        return 0;
    }
    let bits = value.to_bits();
    let sign = (bits >> 31) & 1;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let fraction = bits & 0x7f_ffff;
    if exponent <= 0 {
        return (sign << 15) as u16;
    }
    ((sign << 15) | ((exponent as u32) << 10) | (fraction >> 13)) as u16
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits >> 15 == 1 { -1.0 } else { 1.0 };
    let exponent = ((bits >> 10) & 0x1f) as i32;
    let fraction = (bits & 0x3ff) as f32;
    match exponent {
        0 => sign * fraction * 2f32.powi(-24),
        0x1f => f32::NAN,
        e => sign * (1.0 + fraction / 1024.0) * 2f32.powi(e - 15),
    }
}

/// A deterministic small fp16-exact value per index.
fn value(seed: usize) -> f32 {
    ((seed.wrapping_mul(2_654_435_761) >> 7) % 17) as f32 / 16.0 - 0.5
}

struct Run {
    raw: Vec<u8>,
    best_ms: f64,
    tasks: usize,
}

/// One job running `plan` over a fresh input and a sentinel-filled output.
fn run_job(
    plan: &ConvPlan,
    in_pixels: usize,
    out_pixels: usize,
    channels: usize,
    written_channels: usize,
    input_nhwc: &[u8],
    weights: &[u8],
    bias_len: usize,
    reps: usize,
) -> Run {
    let in_len = nc1hwc2_storage_size(in_pixels, channels * 2).unwrap();
    let mut packed_in = vec![0u8; in_len];
    pack_nhwc_to_nc1hwc2(input_nhwc, in_pixels, channels * 2, &mut packed_in).unwrap();
    // What the DPU writes, which is more than the real channels.
    let out_len = nc1hwc2_storage_size(out_pixels, written_channels * 2).unwrap();

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open NPU device");
    let fd = file.as_raw_fd();
    let page = |bytes: usize| bytes.max(1).next_multiple_of(PAGE_BYTES);
    unsafe {
        // Zero surfaces past `Cin` up to the written granule, so a reader
        // that fetches whole groups stays inside the buffer.
        let in_alloc = in_len.max(nc1hwc2_storage_size(in_pixels, written_channels * 2).unwrap());
        let buf_in = Buffer::new(fd, page(in_alloc), &file);
        ptr::write_bytes(buf_in.host_ptr, 0, buf_in.size);
        ptr::copy_nonoverlapping(packed_in.as_ptr(), buf_in.host_ptr, in_len);
        let buf_out = Buffer::new(fd, page(out_len), &file);
        ptr::write_bytes(buf_out.host_ptr, SENTINEL, buf_out.size);
        let buf_w = Buffer::new(fd, page(weights.len()), &file);
        ptr::copy_nonoverlapping(weights.as_ptr(), buf_w.host_ptr, weights.len());
        let buf_b = Buffer::new(fd, page(bias_len), &file);
        ptr::write_bytes(buf_b.host_ptr, 0, buf_b.size);

        let programs = plan.programs_with_buffers(Buffers {
            input: buf_in.dma_address,
            weights: buf_w.dma_address,
            bias: buf_b.dma_address,
            output: buf_out.dma_address,
        });
        let mut command_buffers = Vec::new();
        let mut in_handles = Vec::new();
        for program in &programs {
            let buffer = Buffer::new(fd, page(program.len() * mem::size_of::<u64>()), &file);
            let words = std::slice::from_raw_parts_mut(buffer.host_ptr as *mut u64, program.len());
            for (word, command) in words.iter_mut().zip(program) {
                *word = command.0;
            }
            fini_bo(fd, buffer.handle).ok();
            in_handles.push(buffer.handle);
            command_buffers.push((buffer, program.len() as u32));
        }
        for buffer in [&buf_w, &buf_b, &buf_in, &buf_out] {
            fini_bo(fd, buffer.handle).ok();
        }
        in_handles.extend([buf_w.handle, buf_b.handle, buf_in.handle]);
        let tasks: Vec<(u32, u32)> = command_buffers
            .iter()
            .map(|(buffer, count)| (buffer.dma_address, *count))
            .collect();
        let out_handles = [buf_out.handle];
        let mut best = f64::MAX;
        for _ in 0..reps {
            let jobs = [JobDesc {
                tasks: &tasks,
                in_handles: &in_handles,
                out_handles: &out_handles,
            }];
            let started = Instant::now();
            submit_jobs(fd, &jobs).expect("SUBMIT failed");
            prep_bo(fd, buf_out.handle, 5_000_000_000).ok();
            best = best.min(started.elapsed().as_secs_f64() * 1e3);
        }
        let raw = std::slice::from_raw_parts(buf_out.host_ptr, out_len).to_vec();
        for (buffer, _) in &command_buffers {
            close_bo(fd, buffer.handle).ok();
        }
        for buffer in [&buf_w, &buf_b, &buf_in, &buf_out] {
            close_bo(fd, buffer.handle).ok();
        }
        Run {
            raw,
            best_ms: best,
            tasks: tasks.len(),
        }
    }
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn depthwise_pad1_mobilenetv2_shapes() {
    let mut failures = Vec::new();
    // `ROCKET_DW_PAD=0` runs the same shapes unpadded: the harness check.
    let pad: usize = std::env::var("ROCKET_DW_PAD")
        .ok()
        .map_or(1, |v| v.parse().unwrap());
    let split = std::env::var("ROCKET_CBUF_SPLIT").ok().map(|spec| {
        let (data, weight) = spec
            .split_once('/')
            .expect("ROCKET_CBUF_SPLIT=<data>/<weight>");
        (data.parse::<u32>().unwrap(), weight.parse::<u32>().unwrap())
    });
    let custom: Option<Vec<(u32, u32, u32)>> =
        std::env::var("ROCKET_PAD1_SHAPES").ok().map(|spec| {
            spec.split(',')
                .map(|shape| {
                    let fields: Vec<u32> = shape.split(':').map(|f| f.parse().unwrap()).collect();
                    (fields[0], fields[1], fields[2])
                })
                .collect()
        });
    // MobileNetV2's seventeen depthwise convolutions, deduplicated: input
    // side, channels, stride. Every one of them is 3x3 with symmetric pad 1.
    let mobilenet = vec![
        (112u32, 32u32, 1u32),
        (112, 96, 2),
        (56, 144, 1),
        (56, 144, 2),
        (28, 192, 1),
        (28, 192, 2),
        (14, 384, 1),
        (14, 576, 1),
        (14, 576, 2),
        (7, 960, 1),
    ];
    for (side, channels, stride) in custom.unwrap_or(mobilenet) {
        let c = channels as usize;
        let s = stride as usize;
        let label = format!("{side}x{side} C{c} s{s}");
        let shape =
            Shape::try_with_precision(side, side, stride, channels, channels, Precision::Fp16)
                .unwrap()
                .try_with_padding([pad, pad])
                .unwrap()
                .try_with_depthwise()
                .unwrap();
        let planned = match split {
            Some((data, weight)) => Ok(ConvPlan::with_cbuf_banks(
                shape,
                [KERNEL, KERNEL],
                data,
                weight,
            )),
            None => ConvPlan::try_new(shape, [KERNEL, KERNEL]),
        };
        let plan = match planned {
            Ok(plan) => plan,
            Err(error) => {
                println!("  {label}: planner refuses: {error}");
                failures.push(format!("{label}: planner refuses: {error}"));
                continue;
            }
        };
        let side = side as usize;
        let out_side = (side + 2 * pad - KERNEL) / s + 1;
        let in_pixels = side * side;
        let out_pixels = out_side * out_side;
        let input_f32: Vec<f32> = (0..in_pixels * c).map(|i| value(i * 3 + 1)).collect();
        let filter_f32: Vec<f32> = (0..KERNEL * KERNEL * c).map(|i| value(i * 7 + 5)).collect();
        let input: Vec<u8> = input_f32
            .iter()
            .flat_map(|v| f32_to_f16_bits(*v).to_le_bytes())
            .collect();
        // The depthwise packer takes the filter channel-major, `[C][kh][kw]`.
        let filter: Vec<u8> = (0..c)
            .flat_map(|ch| (0..KERNEL * KERNEL).map(move |tap| (ch, tap)))
            .flat_map(|(ch, tap)| f32_to_f16_bits(filter_f32[tap * c + ch]).to_le_bytes())
            .collect();
        let weight_plan = WeightPlan::for_conv(shape, [KERNEL, KERNEL]);
        let mut packed = vec![0u8; weight_plan.packed_bytes().unwrap()];
        weight_plan.pack(&filter, &mut packed).unwrap();
        let bias = shape.padded_out_channels() as usize * 4;
        let written = shape.padded_out_channels() as usize;
        let run = run_job(
            &plan, in_pixels, out_pixels, c, written, &input, &packed, bias, 5,
        );

        let mut worst = 0.0f32;
        let mut worst_at = (0, 0, 0);
        let mut unwritten = 0;
        let mut border_worst = 0.0f32;
        for oy in 0..out_side {
            for ox in 0..out_side {
                let border = oy * s < pad
                    || ox * s < pad
                    || oy * s + KERNEL > side + pad
                    || ox * s + KERNEL > side + pad;
                for ch in 0..c {
                    let mut want = 0.0f32;
                    for ky in 0..KERNEL {
                        for kx in 0..KERNEL {
                            let y = (oy * s + ky) as isize - pad as isize;
                            let x = (ox * s + kx) as isize - pad as isize;
                            if y < 0 || x < 0 || y >= side as isize || x >= side as isize {
                                continue;
                            }
                            let pixel = y as usize * side + x as usize;
                            want +=
                                input_f32[pixel * c + ch] * filter_f32[(ky * KERNEL + kx) * c + ch];
                        }
                    }
                    let offset =
                        (ch / 8) * out_pixels * ATOM + (oy * out_side + ox) * ATOM + (ch % 8) * 2;
                    let bits = u16::from_le_bytes([run.raw[offset], run.raw[offset + 1]]);
                    if bits == 0xa5a5 {
                        unwritten += 1;
                    }
                    let error = (f16_to_f32(bits) - want).abs();
                    if !(error <= worst) {
                        worst = error;
                        worst_at = (oy, ox, ch);
                    }
                    if border {
                        border_worst = border_worst.max(error);
                    }
                }
            }
        }
        let killed = run.best_ms > 150.0;
        if !(worst <= 0.01) || unwritten > 0 || killed {
            failures.push(format!(
                "{label}: max|err| {worst} at {worst_at:?}, {unwritten} unwritten{}",
                if killed { ", watchdog kill" } else { "" }
            ));
        }
        let rows: Vec<String> = plan
            .tiles()
            .iter()
            .map(|tile| format!("{}r{}c", tile.rows.in_rows, tile.columns.in_cols))
            .collect();
        println!(
            "  {label} -> {out_side}x{out_side}: {} task(s) [{}], npu {:.2} ms, max|err| {worst:.4} \
             (border {border_worst:.4}), {unwritten} unwritten{}",
            run.tasks,
            rows.join(" "),
            run.best_ms,
            if killed { " -- WATCHDOG KILL" } else { "" }
        );
    }
    assert!(failures.is_empty(), "{failures:?}");
}
