//! A 7x7 depthwise convolution at ConvNeXt's four shapes on the NPU:
//! exact against an f32 reference (gate), with the NPU time printed.
//! `ROCKET_DW_KERNEL=3|5` runs the same shapes at another kernel.
//!
//! ConvNeXt-T's 18 depthwise 7x7 convolutions run on the CPU: the demotion
//! that would make them fp16 is gated off (measured on MobileNetV2's small
//! 3x3 ones), and depthwise admission stops at 3x3. Before admitting 7x7
//! this measures it: each ConvNeXt stage's depthwise convolution on its
//! padded input (the model's pad 3 is a CPU `tensor.pad`), against an f32
//! reference of the same fp16 values, with the NPU time.
//!
//!     ./depthwise_7x7_probe_hw-<hash> --ignored --nocapture

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

/// A deterministic small fp16 value per index.
fn value(seed: usize) -> f32 {
    ((seed.wrapping_mul(2_654_435_761) >> 7) % 17) as f32 / 16.0 - 0.5
}

struct Run {
    raw: Vec<u8>,
    best_ms: f64,
    tasks: usize,
}

/// One job: `slices` tasks lists, each `(plan, input surface offset, output
/// surface offset, packed weights)`, over one shared input and output cube.
fn run_job(
    in_pixels: usize,
    in_channels: usize,
    out_pixels: usize,
    out_channels: usize,
    input_nhwc: &[u8],
    slices: &[(ConvPlan, usize, usize, Vec<u8>, usize)],
    reps: usize,
) -> Run {
    let in_len = nc1hwc2_storage_size(in_pixels, in_channels * 2).unwrap();
    let mut packed_in = vec![0u8; in_len];
    pack_nhwc_to_nc1hwc2(input_nhwc, in_pixels, in_channels * 2, &mut packed_in).unwrap();
    let out_len = nc1hwc2_storage_size(out_pixels, out_channels * 2).unwrap();

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open NPU device");
    let fd = file.as_raw_fd();
    let page = |bytes: usize| bytes.max(1).next_multiple_of(PAGE_BYTES);
    unsafe {
        let buf_in = Buffer::new(fd, page(in_len), &file);
        ptr::copy_nonoverlapping(packed_in.as_ptr(), buf_in.host_ptr, in_len);
        let buf_out = Buffer::new(fd, page(out_len), &file);
        ptr::write_bytes(buf_out.host_ptr, SENTINEL, buf_out.size);

        let mut owned = Vec::new();
        let mut command_buffers = Vec::new();
        let mut in_handles = Vec::new();
        for (plan, in_surface, out_surface, weights, bias_len) in slices {
            let buf_w = Buffer::new(fd, page(weights.len()), &file);
            ptr::copy_nonoverlapping(weights.as_ptr(), buf_w.host_ptr, weights.len());
            let buf_b = Buffer::new(fd, page(*bias_len), &file);
            ptr::write_bytes(buf_b.host_ptr, 0, buf_b.size);
            let programs = plan.programs_with_buffers(Buffers {
                input: buf_in.dma_address + (in_surface * in_pixels * ATOM) as u32,
                weights: buf_w.dma_address,
                bias: buf_b.dma_address,
                output: buf_out.dma_address + (out_surface * out_pixels * ATOM) as u32,
            });
            for program in &programs {
                let buffer = Buffer::new(fd, page(program.len() * mem::size_of::<u64>()), &file);
                let words =
                    std::slice::from_raw_parts_mut(buffer.host_ptr as *mut u64, program.len());
                for (word, command) in words.iter_mut().zip(program) {
                    *word = command.0;
                }
                fini_bo(fd, buffer.handle).ok();
                in_handles.push(buffer.handle);
                command_buffers.push((buffer, program.len() as u32));
            }
            fini_bo(fd, buf_w.handle).ok();
            fini_bo(fd, buf_b.handle).ok();
            in_handles.push(buf_w.handle);
            in_handles.push(buf_b.handle);
            owned.push(buf_w);
            owned.push(buf_b);
        }
        fini_bo(fd, buf_in.handle).ok();
        fini_bo(fd, buf_out.handle).ok();
        in_handles.push(buf_in.handle);
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
        for buffer in &owned {
            close_bo(fd, buffer.handle).ok();
        }
        close_bo(fd, buf_in.handle).ok();
        close_bo(fd, buf_out.handle).ok();
        Run {
            raw,
            best_ms: best,
            tasks: tasks.len(),
        }
    }
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

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn depthwise_7x7_convnext_probe() {
    let mut failures = Vec::new();
    let kernel: usize = std::env::var("ROCKET_DW_KERNEL")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(7);
    for (label, side, channels) in [
        ("stage1", 62u32, 96u32),
        ("stage2", 34, 192),
        ("stage3", 20, 384),
        ("stage4", 13, 768),
    ] {
        let c = channels as usize;
        let shape = Shape::try_with_precision(side, side, 1, channels, channels, Precision::Fp16)
            .unwrap()
            .try_with_padding([0, 0])
            .unwrap()
            .try_with_depthwise()
            .unwrap();
        let plan = match ConvPlan::try_new(shape, [kernel, kernel]) {
            Ok(plan) => plan,
            Err(error) => {
                println!("  {label} {side}x{side} C{c}: planner refuses: {error}");
                failures.push(format!("{label}: planner refuses: {error}"));
                continue;
            }
        };
        let out_side = side as usize - kernel + 1;
        let in_pixels = (side * side) as usize;
        let out_pixels = out_side * out_side;
        let input_f32: Vec<f32> = (0..in_pixels * c).map(|i| value(i * 3 + 1)).collect();
        let filter_f32: Vec<f32> = (0..kernel * kernel * c).map(|i| value(i * 7 + 5)).collect();
        let input: Vec<u8> = input_f32
            .iter()
            .flat_map(|v| f32_to_f16_bits(*v).to_le_bytes())
            .collect();
        // The depthwise packer takes the filter channel-major, `[C][kh][kw]`
        // (what the compiler's depthwise matcher hands the driver), not HWC.
        let filter: Vec<u8> = (0..c)
            .flat_map(|ch| (0..kernel * kernel).map(move |tap| (ch, tap)))
            .flat_map(|(ch, tap)| f32_to_f16_bits(filter_f32[tap * c + ch]).to_le_bytes())
            .collect();
        let weight_plan = WeightPlan::for_conv(shape, [kernel, kernel]);
        let mut packed = vec![0u8; weight_plan.packed_bytes().unwrap()];
        weight_plan.pack(&filter, &mut packed).unwrap();
        let bias = shape.padded_out_channels() as usize * 4;
        std::thread::sleep(std::time::Duration::from_millis(800));
        // Size the output by what the DPU writes: a depthwise output granule
        // is 32 fp16 channels, so a `Cin % 32 != 0` cube sized for `Cin` is
        // overrun and the job hangs (`depthwise_pad1_hw`'s doc). Every
        // ConvNeXt stage is a multiple of 32, which is why this went unseen.
        let run = run_job(
            in_pixels,
            c,
            out_pixels,
            shape.padded_out_channels() as usize,
            &input,
            &[(plan, 0, 0, packed, bias)],
            5,
        );
        let mut worst = 0.0f32;
        let mut unwritten = 0;
        for oy in 0..out_side {
            for ox in 0..out_side {
                for ch in 0..c {
                    let mut want = 0.0f32;
                    for ky in 0..kernel {
                        for kx in 0..kernel {
                            let x = input_f32[((oy + ky) * side as usize + ox + kx) * c + ch];
                            want += x * filter_f32[(ky * kernel + kx) * c + ch];
                        }
                    }
                    let offset =
                        (ch / 8) * out_pixels * ATOM + (oy * out_side + ox) * ATOM + (ch % 8) * 2;
                    let bits = u16::from_le_bytes([run.raw[offset], run.raw[offset + 1]]);
                    if bits == 0xa5a5 {
                        unwritten += 1;
                    }
                    worst = worst.max((f16_to_f32(bits) - want).abs());
                }
            }
        }
        if worst > 0.01 || unwritten > 0 || run.best_ms > 150.0 {
            failures.push(format!(
                "{label} k{kernel}: max|err| {worst}, {unwritten} unwritten"
            ));
        }
        println!(
            "  {label} {side}x{side} C{c} k{kernel}: {} task(s), npu {:.2} ms, max|err| {worst:.4}, {unwritten} unwritten{}",
            run.tasks,
            run.best_ms,
            if run.best_ms > 150.0 {
                " -- WATCHDOG KILL"
            } else {
                ""
            }
        );
    }
    assert!(failures.is_empty(), "{failures:?}");
}
