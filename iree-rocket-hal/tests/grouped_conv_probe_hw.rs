//! Can a grouped convolution run as one job of per-channel-slice tasks?
//!
//! ResNeXt's 32-group 3x3 convolutions run today as dense convolutions with
//! a block-diagonal filter (`rocket-block-diagonal-grouped-conv`): G x the
//! multiply-adds and G x the filter bytes, ~51 ms of NPU per inference, half
//! of it stage 4. A grouped convolution is independent dense convolutions
//! over channel slices, and in NC1HWC2 a slice of whole atoms is a
//! contiguous run of surfaces whose stride depends only on the pixel count.
//! So bundle `b` (a run of `bundle` groups, `Cb` in and `Fb` out channels) is
//! the ordinary dense plan for `Cb -> Fb`, with its input and output bases
//! moved `b * Cb / 8` and `b * Fb / 8` surfaces into the full cubes -- and all
//! bundles' tasks go in one job.
//!
//! A probe, not a gate: per ResNeXt stage and bundle size it prints whether
//! the sliced job is bit-identical to the block-diagonal dense convolution
//! on the NPU (the extra zero products should not change a sum) and both
//! NPU times.
//!
//!     ./grouped_conv_probe_hw-<hash> --ignored --nocapture

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr, time::Instant};

use iree_rocket_hal::rocket::{
    conv::{Buffers, ConvPlan, Precision, Shape},
    device::{Buffer, JobDesc, close_bo, fini_bo, prep_bo, submit_jobs},
    tensor_layout::{
        nc1hwc2_storage_size, pack_hwcf_to_rocket_weights, pack_nhwc_to_nc1hwc2,
        rocket_weight_storage_size,
    },
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

#[derive(Clone, Copy, Debug)]
struct Grouped {
    side: u32,
    stride: u32,
    channels: u32,
    features: u32,
    groups: u32,
}

impl Grouped {
    fn cg(self) -> usize {
        (self.channels / self.groups) as usize
    }
    fn fg(self) -> usize {
        (self.features / self.groups) as usize
    }
    fn out_side(self) -> u32 {
        (self.side + 2 - 3) / self.stride + 1
    }
    /// Grouped filter value `w[g][f][c][y][x]`.
    fn weight(self, g: usize, f: usize, c: usize, tap: usize) -> f32 {
        value(((g * self.fg() + f) * self.cg() + c) * 9 + tap + 7)
    }
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

/// HWCF filter bytes for a dense conv over `groups_in_block` consecutive
/// groups starting at `first_group`: block-diagonal within the block.
fn block_filter(shape: Grouped, first_group: usize, groups_in_block: usize) -> Vec<u8> {
    let (cg, fg) = (shape.cg(), shape.fg());
    let (cin, cout) = (cg * groups_in_block, fg * groups_in_block);
    let mut bytes = Vec::with_capacity(9 * cin * cout * 2);
    for tap in 0..9 {
        for i in 0..cin {
            for o in 0..cout {
                let (gi, go) = (i / cg, o / fg);
                let w = if gi == go {
                    shape.weight(first_group + go, o % fg, i % cg, tap)
                } else {
                    0.0
                };
                bytes.extend_from_slice(&f32_to_f16_bits(w).to_le_bytes());
            }
        }
    }
    bytes
}

fn plan_for(shape: Grouped, cin: usize, cout: usize) -> (ConvPlan, usize) {
    let conv = Shape::try_with_precision(
        shape.side,
        shape.side,
        shape.stride,
        cin as u32,
        cout as u32,
        Precision::Fp16,
    )
    .unwrap()
    .try_with_padding([1, 1])
    .unwrap();
    let bias = conv.padded_out_channels() as usize * 4;
    (ConvPlan::try_new(conv, [3, 3]).unwrap(), bias)
}

fn packed(filter: &[u8], cin: usize, cout: usize) -> Vec<u8> {
    let len = rocket_weight_storage_size(3, 3, cin, cout, 2).unwrap();
    let mut out = vec![0u8; len];
    pack_hwcf_to_rocket_weights(filter, 3, 3, cin, cout, 2, &mut out).unwrap();
    out
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- characterization, prints rather than asserts"]
fn grouped_conv_slices_probe() {
    let stages = [
        (
            "stage4 7x7 s1",
            Grouped {
                side: 7,
                stride: 1,
                channels: 1024,
                features: 1024,
                groups: 32,
            },
            vec![1, 2, 4],
        ),
        (
            "stage4 14x14 s2",
            Grouped {
                side: 14,
                stride: 2,
                channels: 1024,
                features: 1024,
                groups: 32,
            },
            vec![1, 2, 4],
        ),
        (
            "stage3 14x14 s1",
            Grouped {
                side: 14,
                stride: 1,
                channels: 512,
                features: 512,
                groups: 32,
            },
            vec![1, 2, 4],
        ),
        (
            "stage3 28x28 s2",
            Grouped {
                side: 28,
                stride: 2,
                channels: 512,
                features: 512,
                groups: 32,
            },
            vec![1, 2],
        ),
        (
            "stage2 28x28 s1",
            Grouped {
                side: 28,
                stride: 1,
                channels: 256,
                features: 256,
                groups: 32,
            },
            vec![2, 4, 8],
        ),
        (
            "stage1 56x56 s1",
            Grouped {
                side: 56,
                stride: 1,
                channels: 128,
                features: 128,
                groups: 32,
            },
            vec![4, 8, 16],
        ),
    ];
    for (label, shape, bundles) in stages {
        let (c, f) = (shape.channels as usize, shape.features as usize);
        let in_pixels = (shape.side * shape.side) as usize;
        let out_pixels = (shape.out_side() * shape.out_side()) as usize;
        let input: Vec<u8> = (0..in_pixels * c)
            .flat_map(|i| f32_to_f16_bits(value(i * 3 + 1)).to_le_bytes())
            .collect();

        // The block-diagonal dense convolution: the reference and the baseline.
        let (plan, bias) = plan_for(shape, c, f);
        let dense_w = packed(&block_filter(shape, 0, shape.groups as usize), c, f);
        std::thread::sleep(std::time::Duration::from_millis(800));
        let dense = run_job(
            in_pixels,
            c,
            out_pixels,
            f,
            &input,
            &[(plan, 0, 0, dense_w, bias)],
            5,
        );
        println!(
            "  {label} C{c} G{}: block-diagonal {:.2} ms, {} task(s){}",
            shape.groups,
            dense.best_ms,
            dense.tasks,
            if dense.best_ms > 150.0 {
                " -- WATCHDOG KILL"
            } else {
                ""
            }
        );

        for bundle in bundles {
            let (cb, fb) = (shape.cg() * bundle, shape.fg() * bundle);
            if cb % 8 != 0 || fb % 16 != 0 {
                println!("    bundle {bundle}: Cb {cb} / Fb {fb} not whole atoms, skipped");
                continue;
            }
            let blocks = shape.groups as usize / bundle;
            let slices: Vec<_> = (0..blocks)
                .map(|b| {
                    let (plan, bias) = plan_for(shape, cb, fb);
                    let w = packed(&block_filter(shape, b * bundle, bundle), cb, fb);
                    (plan, b * cb / 8, b * fb / 8, w, bias)
                })
                .collect();
            std::thread::sleep(std::time::Duration::from_millis(800));
            let sliced = run_job(in_pixels, c, out_pixels, f, &input, &slices, 5);
            let differ = dense
                .raw
                .chunks(2)
                .zip(sliced.raw.chunks(2))
                .filter(|(a, b)| a != b)
                .count();
            let unwritten = sliced
                .raw
                .chunks(2)
                .filter(|v| v == &[SENTINEL, SENTINEL])
                .count();
            println!(
                "    bundle {bundle} (Cb {cb} -> Fb {fb}, {blocks} slices): {:.2} ms, {} task(s), \
                 {differ} differ, {unwritten} unwritten{}",
                sliced.best_ms,
                sliced.tasks,
                if sliced.best_ms > 150.0 {
                    " -- WATCHDOG KILL"
                } else {
                    ""
                }
            );
        }
    }
}

/// The gate: rocket-core's `GroupedConv` -- the slicing, surface offsets and
/// packed stream the compiler and the driver share -- run as one job of
/// slice tasks, bit-identical to the block-diagonal dense convolution at
/// every ResNeXt stage shape, stride 1 and 2.
#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn grouped_conv_matches_block_diagonal() {
    use iree_rocket_hal::rocket::grouped::GroupedConv;
    let mut failures = Vec::new();
    for shape in [
        Grouped {
            side: 56,
            stride: 1,
            channels: 128,
            features: 128,
            groups: 32,
        },
        Grouped {
            side: 56,
            stride: 2,
            channels: 256,
            features: 256,
            groups: 32,
        },
        Grouped {
            side: 28,
            stride: 1,
            channels: 256,
            features: 256,
            groups: 32,
        },
        Grouped {
            side: 14,
            stride: 1,
            channels: 512,
            features: 512,
            groups: 32,
        },
        Grouped {
            side: 14,
            stride: 2,
            channels: 1024,
            features: 1024,
            groups: 32,
        },
        Grouped {
            side: 7,
            stride: 1,
            channels: 1024,
            features: 1024,
            groups: 32,
        },
        Grouped {
            side: 9,
            stride: 1,
            channels: 64,
            features: 64,
            groups: 8,
        },
    ] {
        let (c, f) = (shape.channels as usize, shape.features as usize);
        let in_pixels = (shape.side * shape.side) as usize;
        let out_pixels = (shape.out_side() * shape.out_side()) as usize;
        let input: Vec<u8> = (0..in_pixels * c)
            .flat_map(|i| f32_to_f16_bits(value(i * 5 + 3)).to_le_bytes())
            .collect();
        let dense_filter = block_filter(shape, 0, shape.groups as usize);
        let (plan, bias) = plan_for(shape, c, f);
        std::thread::sleep(std::time::Duration::from_millis(800));
        let dense = run_job(
            in_pixels,
            c,
            out_pixels,
            f,
            &input,
            &[(plan, 0, 0, packed(&dense_filter, c, f), bias)],
            1,
        );

        let full = Shape::try_with_precision(
            shape.side,
            shape.side,
            shape.stride,
            c as u32,
            f as u32,
            Precision::Fp16,
        )
        .unwrap()
        .try_with_padding([1, 1])
        .unwrap();
        let grouped = GroupedConv::try_new(full, [3, 3], shape.groups).unwrap();
        let mut stream = vec![0u8; grouped.packed_bytes().unwrap()];
        grouped.pack(&dense_filter, &mut stream).unwrap();
        let slice_bytes = grouped.slice_packed_bytes().unwrap();
        let slice = grouped.slice_shape();
        let slices: Vec<_> = (0..grouped.slices())
            .map(|s| {
                let (plan, bias) = plan_for(
                    shape,
                    slice.in_channels as usize,
                    slice.out_channels as usize,
                );
                (
                    plan,
                    grouped.input_surface(s) as usize,
                    grouped.output_surface(s) as usize,
                    stream[s as usize * slice_bytes..(s as usize + 1) * slice_bytes].to_vec(),
                    bias,
                )
            })
            .collect();
        std::thread::sleep(std::time::Duration::from_millis(800));
        let sliced = run_job(in_pixels, c, out_pixels, f, &input, &slices, 1);
        let differ = dense
            .raw
            .chunks(2)
            .zip(sliced.raw.chunks(2))
            .filter(|(a, b)| a != b)
            .count();
        println!(
            "  {}x{} s{} C{c} G{}: {} slices of {}, {differ} differ, {:.2} vs {:.2} ms",
            shape.side,
            shape.side,
            shape.stride,
            shape.groups,
            grouped.slices(),
            slice.in_channels,
            sliced.best_ms,
            dense.best_ms,
        );
        if differ != 0 || sliced.best_ms > 150.0 || dense.best_ms > 150.0 {
            failures.push(format!("{shape:?}: {differ} differ"));
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
}
