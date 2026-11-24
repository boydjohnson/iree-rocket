//! Can the DPU write a matmul's result straight into a row-major `[M, N]`
//! buffer, so the driver never compacts it?
//!
//! The DPU has no destination line-stride register, but it has two knobs
//! that together make one: `DPU_DATA_CUBE_NOTCH_ADDR` skips that many atoms
//! at the end of every output line (validated: horizontal column tiles use
//! it as `full_width - tile_width`), and `DPU_DST_SURF_STRIDE` places
//! channel group `c1` at `c1 * stride` atoms. Plan the matmul as a 1x1
//! convolution of width 1 and height M -- one output pixel per line -- and
//! set the surface stride to one atom and the notch to `N * bytes / 16 - 1`:
//! pixel `m`, channel group `c1` then lands at `(m * N * bytes / 16 + c1)`
//! atoms, which is exactly row-major.
//!
//! A probe, not a gate: it sweeps `DPU_SURFACE_ADD`, whose meaning at a
//! one-atom surface stride nothing has measured, and prints what each value
//! does.
//!
//!     ./matmul_dense_output_probe_hw-<hash> --ignored --nocapture

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr, time::Instant};

use iree_rocket_hal::rocket::{
    builders::{
        Bits, RegCmd, Register, RegisterMeta,
        dpu::{DpuDataCubeNotchAddr, DpuDstSurfStride, DpuSurfaceAdd},
    },
    conv::{Buffers, ConvPlan, Precision, Shape},
    device::{Buffer, close_bo, fini_bo, prep_bo, submit},
    tensor_layout::{
        nc1hwc2_storage_size, pack_hwcf_to_rocket_weights, pack_nhwc_to_nc1hwc2,
        rocket_weight_storage_size,
    },
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const PAGE_BYTES: usize = 4096;
const SENTINEL: u8 = 0xa5;

fn f32_to_f16_bits(value: f32) -> u16 {
    if value == 0.0 {
        return 0;
    }
    let bits = value.to_bits();
    let sign = (bits >> 31) & 1;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let fraction = bits & 0x7f_ffff;
    assert!((1..31).contains(&exponent));
    ((sign << 15) | ((exponent as u32) << 10) | (fraction >> 13)) as u16
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15);
    let exponent = u32::from((bits >> 10) & 0x1f);
    let fraction = u32::from(bits & 0x3ff);
    if exponent == 0 && fraction == 0 {
        return if sign == 1 { -0.0 } else { 0.0 };
    }
    if exponent == 0x1f {
        return f32::NAN;
    }
    if exponent == 0 {
        return (if sign == 1 { -1.0 } else { 1.0 }) * fraction as f32 * 2f32.powi(-24);
    }
    f32::from_bits((sign << 31) | ((exponent + (127 - 15)) << 23) | (fraction << 13))
}

fn replace<R: RegisterMeta>(program: &mut [RegCmd], value: RegCmd) -> usize {
    let mut count = 0;
    for command in program.iter_mut() {
        let domain = (command.0 >> 48) as u32;
        let offset = command.0 as u32 & 0xffff;
        if domain == R::DOMAIN && offset == R::OFFSET {
            *command = RegCmd(value.0);
            count += 1;
        }
    }
    count
}

fn raw_value<R: RegisterMeta>(program: &[RegCmd]) -> u32 {
    program
        .iter()
        .find(|command| {
            (command.0 >> 48) as u32 == R::DOMAIN && command.0 as u32 & 0xffff == R::OFFSET
        })
        .map(|command| (command.0 >> 16) as u32)
        .expect("register present")
}

/// Small integers, so every fp16 product and sum is exact.
fn input_at(m: usize, k: usize) -> f32 {
    (((m * 7 + k * 3) % 9) as i32 - 4) as f32
}
fn weight_at(k: usize, n: usize) -> f32 {
    (((k * 5 + n * 11) % 7) as i32 - 3) as f32
}

struct Outcome {
    exact: usize,
    total: usize,
    sentinel: usize,
    first_wrong: Option<(usize, usize, f32, f32)>,
    elapsed_ms: f64,
}

/// `surface_add`: `None` keeps the planner's value.
fn run(m: usize, k: usize, n: usize, surface_add: Option<u32>) -> Outcome {
    let shape = Shape::try_with_precision(1, m as u32, 1, k as u32, n as u32, Precision::Fp16)
        .unwrap()
        .try_with_padding([0, 0])
        .unwrap();
    let plan = ConvPlan::try_new(shape, [1, 1]).unwrap();

    let dense_input: Vec<u8> = (0..m)
        .flat_map(|row| (0..k).map(move |col| input_at(row, col)))
        .flat_map(|value| f32_to_f16_bits(value).to_le_bytes())
        .collect();
    let dense_weights: Vec<u8> = (0..k)
        .flat_map(|row| (0..n).map(move |col| weight_at(row, col)))
        .flat_map(|value| f32_to_f16_bits(value).to_le_bytes())
        .collect();
    let input_len = nc1hwc2_storage_size(m, k * 2).unwrap();
    let mut packed_input = vec![0; input_len];
    pack_nhwc_to_nc1hwc2(&dense_input, m, k * 2, &mut packed_input).unwrap();
    let weight_len = rocket_weight_storage_size(1, 1, k, n, 2).unwrap();
    let mut packed_weights = vec![0; weight_len];
    pack_hwcf_to_rocket_weights(&dense_weights, 1, 1, k, n, 2, &mut packed_weights).unwrap();
    let bias_len = shape.padded_out_channels() as usize * 4;
    let output_len = m * n * 2;

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open NPU device");
    let fd = file.as_raw_fd();
    let page = |bytes: usize| bytes.max(1).next_multiple_of(PAGE_BYTES);
    unsafe {
        let buf_in = Buffer::new(fd, page(input_len), &file);
        ptr::copy_nonoverlapping(packed_input.as_ptr(), buf_in.host_ptr, input_len);
        let buf_weights = Buffer::new(fd, page(weight_len), &file);
        ptr::copy_nonoverlapping(packed_weights.as_ptr(), buf_weights.host_ptr, weight_len);
        let buf_bias = Buffer::new(fd, page(bias_len), &file);
        ptr::write_bytes(buf_bias.host_ptr, 0, buf_bias.size);
        // Twice the output, sentinel-filled: a write past the dense extent
        // shows up in the tail.
        let buf_out = Buffer::new(fd, page(output_len * 2), &file);
        ptr::write_bytes(buf_out.host_ptr, SENTINEL, buf_out.size);

        let mut programs = plan.programs_with_buffers(Buffers {
            input: buf_in.dma_address,
            weights: buf_weights.dma_address,
            bias: buf_bias.dma_address,
            output: buf_out.dma_address,
        });
        assert_eq!(programs.len(), 1, "probe shape must be one job");
        let program = &mut programs[0];
        let notch = (n * 2 / 16 - 1) as u32;
        assert_eq!(
            replace::<DpuDstSurfStride>(
                program,
                Register::<DpuDstSurfStride>::new()
                    .dst_surf_stride(Bits::new(1))
                    .build()
            ),
            1
        );
        assert_eq!(
            replace::<DpuDataCubeNotchAddr>(
                program,
                Register::<DpuDataCubeNotchAddr>::new()
                    .notch_addr_0(Bits::new(notch))
                    .notch_addr_1(Bits::new(notch))
                    .build()
            ),
            1
        );
        if let Some(value) = surface_add {
            assert_eq!(
                replace::<DpuSurfaceAdd>(
                    program,
                    Register::<DpuSurfaceAdd>::new()
                        .surf_add(Bits::new(value))
                        .build()
                ),
                1
            );
        }
        if surface_add.is_none() {
            println!(
                "    planner SURFACE_ADD raw {:#x}",
                raw_value::<DpuSurfaceAdd>(program)
            );
        }

        let command_bytes = program.len() * mem::size_of::<u64>();
        let buf_commands = Buffer::new(fd, page(command_bytes), &file);
        let words =
            std::slice::from_raw_parts_mut(buf_commands.host_ptr as *mut u64, program.len());
        for (word, command) in words.iter_mut().zip(program.iter()) {
            *word = command.0;
        }
        for handle in [
            buf_in.handle,
            buf_weights.handle,
            buf_bias.handle,
            buf_out.handle,
            buf_commands.handle,
        ] {
            fini_bo(fd, handle).ok();
        }
        let started = Instant::now();
        submit(
            fd,
            buf_commands.dma_address,
            program.len() as u32,
            &[
                buf_commands.handle,
                buf_in.handle,
                buf_weights.handle,
                buf_bias.handle,
            ],
            &[buf_out.handle],
        )
        .expect("SUBMIT failed");
        prep_bo(fd, buf_out.handle, 2_000_000_000).ok();
        let elapsed_ms = started.elapsed().as_secs_f64() * 1e3;

        let raw = std::slice::from_raw_parts(buf_out.host_ptr, buf_out.size);
        let mut exact = 0;
        let mut first_wrong = None;
        for row in 0..m {
            for col in 0..n {
                let want: f32 = (0..k).map(|i| input_at(row, i) * weight_at(i, col)).sum();
                let offset = (row * n + col) * 2;
                let got = f16_to_f32(u16::from_le_bytes([raw[offset], raw[offset + 1]]));
                if got == want {
                    exact += 1;
                } else {
                    first_wrong.get_or_insert((row, col, want, got));
                }
            }
        }
        let sentinel = raw[output_len..]
            .iter()
            .filter(|byte| **byte != SENTINEL)
            .count();

        for handle in [
            buf_commands.handle,
            buf_in.handle,
            buf_weights.handle,
            buf_bias.handle,
            buf_out.handle,
        ] {
            close_bo(fd, handle).ok();
        }
        Outcome {
            exact,
            total: m * n,
            sentinel,
            first_wrong,
            elapsed_ms,
        }
    }
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- characterization, prints rather than asserts"]
fn matmul_dense_output_probe() {
    for (m, k, n) in [(16, 64, 64), (32, 64, 64), (8, 32, 16)] {
        println!("  M {m} K {k} N {n}");
        for surface_add in [None, Some(2), Some(1), Some(0), Some(4)] {
            std::thread::sleep(std::time::Duration::from_millis(800));
            let outcome = run(m, k, n, surface_add);
            println!(
                "    SURFACE_ADD {}: {}/{} exact, {} bytes written past the dense extent{}, \
                 dispatch {:.1} ms{}",
                surface_add.map_or("planner".to_string(), |value| value.to_string()),
                outcome.exact,
                outcome.total,
                outcome.sentinel,
                outcome
                    .first_wrong
                    .map(|(row, col, want, got)| format!(
                        ", first wrong ({row},{col}) want {want} got {got}"
                    ))
                    .unwrap_or_default(),
                outcome.elapsed_ms,
                if outcome.elapsed_ms > 150.0 {
                    " -- WATCHDOG KILL"
                } else {
                    ""
                },
            );
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Variant {
    /// Today's lowering: width M, height 1, NC1HWC2 output.
    WideCube,
    /// Width 1, height M, NC1HWC2 output.
    TallCube,
    /// Width 1, height M, written straight into a row-major `[M, N]`.
    TallDense,
}

/// Runs a whole multi-tile plan as one job (one task per tile, as the driver
/// submits it) and returns the dense `[M, N]` fp16 output bits plus the
/// best-of-`reps` dispatch time.
fn run_full(
    m: usize,
    k: usize,
    n: usize,
    variant: Variant,
    reps: usize,
) -> (Vec<u16>, f64, usize, usize) {
    use iree_rocket_hal::rocket::device::{JobDesc, submit_jobs};
    let (width, height) = match variant {
        Variant::WideCube => (m as u32, 1),
        Variant::TallCube | Variant::TallDense => (1, m as u32),
    };
    let shape = Shape::try_with_precision(width, height, 1, k as u32, n as u32, Precision::Fp16)
        .unwrap()
        .try_with_padding([0, 0])
        .unwrap();
    let plan = ConvPlan::try_new(shape, [1, 1]).unwrap();

    let dense_input: Vec<u8> = (0..m)
        .flat_map(|row| (0..k).map(move |col| input_at(row, col) / 4.0))
        .flat_map(|value| f32_to_f16_bits(value).to_le_bytes())
        .collect();
    let dense_weights: Vec<u8> = (0..k)
        .flat_map(|row| (0..n).map(move |col| weight_at(row, col) / 4.0))
        .flat_map(|value| f32_to_f16_bits(value).to_le_bytes())
        .collect();
    let input_len = nc1hwc2_storage_size(m, k * 2).unwrap();
    let mut packed_input = vec![0; input_len];
    pack_nhwc_to_nc1hwc2(&dense_input, m, k * 2, &mut packed_input).unwrap();
    let weight_len = rocket_weight_storage_size(1, 1, k, n, 2).unwrap();
    let mut packed_weights = vec![0; weight_len];
    pack_hwcf_to_rocket_weights(&dense_weights, 1, 1, k, n, 2, &mut packed_weights).unwrap();
    let bias_len = shape.padded_out_channels() as usize * 4;
    let cube_len = nc1hwc2_storage_size(m, n * 2).unwrap();
    let output_len = cube_len.max(m * n * 2);

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open NPU device");
    let fd = file.as_raw_fd();
    let page = |bytes: usize| bytes.max(1).next_multiple_of(PAGE_BYTES);
    unsafe {
        let buf_in = Buffer::new(fd, page(input_len), &file);
        ptr::copy_nonoverlapping(packed_input.as_ptr(), buf_in.host_ptr, input_len);
        let buf_weights = Buffer::new(fd, page(weight_len), &file);
        ptr::copy_nonoverlapping(packed_weights.as_ptr(), buf_weights.host_ptr, weight_len);
        let buf_bias = Buffer::new(fd, page(bias_len), &file);
        ptr::write_bytes(buf_bias.host_ptr, 0, buf_bias.size);
        let buf_out = Buffer::new(fd, page(output_len), &file);
        ptr::write_bytes(buf_out.host_ptr, SENTINEL, buf_out.size);

        let mut programs = plan.programs_with_buffers(Buffers {
            input: buf_in.dma_address,
            weights: buf_weights.dma_address,
            bias: buf_bias.dma_address,
            output: buf_out.dma_address,
        });
        if variant == Variant::TallDense {
            // The shipping path: `fc::Plan::try_new_row_major_output` and
            // `place_row_major_output`, not hand-patched registers.
            let fc_plan = iree_rocket_hal::rocket::fc::Plan::try_new_row_major_output(
                iree_rocket_hal::rocket::fc::Shape::new(
                    m as u32,
                    k as u32,
                    n as u32,
                    Precision::Fp16,
                ),
            )
            .expect("row-major output supported");
            assert_eq!(fc_plan.conv_plan().tiles(), plan.tiles());
            for (tile, program) in programs.iter_mut().enumerate() {
                fc_plan.place_row_major_output(program, tile, buf_out.dma_address);
            }
        }
        let mut command_buffers = Vec::new();
        for program in &programs {
            let buffer = Buffer::new(fd, page(program.len() * mem::size_of::<u64>()), &file);
            let words = std::slice::from_raw_parts_mut(buffer.host_ptr as *mut u64, program.len());
            for (word, command) in words.iter_mut().zip(program) {
                *word = command.0;
            }
            fini_bo(fd, buffer.handle).ok();
            command_buffers.push((buffer, program.len() as u32));
        }
        for handle in [
            buf_in.handle,
            buf_weights.handle,
            buf_bias.handle,
            buf_out.handle,
        ] {
            fini_bo(fd, handle).ok();
        }
        let tasks: Vec<(u32, u32)> = command_buffers
            .iter()
            .map(|(buffer, count)| (buffer.dma_address, *count))
            .collect();
        let mut in_handles: Vec<u32> = command_buffers.iter().map(|(b, _)| b.handle).collect();
        in_handles.extend([buf_in.handle, buf_weights.handle, buf_bias.handle]);
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

        let raw = std::slice::from_raw_parts(buf_out.host_ptr, buf_out.size);
        let mut out = vec![0u16; m * n];
        for row in 0..m {
            for col in 0..n {
                let offset = match variant {
                    Variant::TallDense => (row * n + col) * 2,
                    _ => (col / 8) * (m.next_multiple_of(1)) * 16 + row * 16 + (col % 8) * 2,
                };
                out[row * n + col] = u16::from_le_bytes([raw[offset], raw[offset + 1]]);
            }
        }
        // Bytes past the dense [M, N] extent the writer should never touch
        // (only meaningful for the row-major variant; a cube is larger).
        let past = if variant == Variant::TallDense {
            raw[m * n * 2..]
                .iter()
                .filter(|byte| **byte != SENTINEL)
                .count()
        } else {
            0
        };
        for (buffer, _) in &command_buffers {
            close_bo(fd, buffer.handle).ok();
        }
        for handle in [
            buf_in.handle,
            buf_weights.handle,
            buf_bias.handle,
            buf_out.handle,
        ] {
            close_bo(fd, handle).ok();
        }
        (out, best, tasks.len(), past)
    }
}

/// BLIP's matmul shapes in all three variants: NPU time (best of 5) and
/// whether each variant is bit-identical to today's lowering.
#[test]
#[ignore = "needs /dev/accel/accel0 -- characterization, prints rather than asserts"]
fn matmul_dense_output_timing_probe() {
    let shapes: Vec<(usize, usize, usize)> = std::env::var("ROCKET_MM_SHAPES")
        .ok()
        .map(|text| {
            text.split(';')
                .map(|shape| {
                    let dims: Vec<usize> = shape.split('x').map(|d| d.parse().unwrap()).collect();
                    (dims[0], dims[1], dims[2])
                })
                .collect()
        })
        .unwrap_or_else(|| {
            vec![
                (577, 768, 768),
                (577, 768, 3072),
                (577, 3072, 768),
                (577, 768, 2304),
                (16, 768, 768),
                (16, 768, 3072),
                (16, 3072, 768),
            ]
        });
    for (m, k, n) in shapes {
        let mut reference: Option<Vec<u16>> = None;
        for variant in [Variant::WideCube, Variant::TallCube, Variant::TallDense] {
            std::thread::sleep(std::time::Duration::from_millis(800));
            let (out, ms, tasks, _) = run_full(m, k, n, variant, 5);
            let sentinel = out.iter().filter(|v| **v == 0xa5a5).count();
            let verdict = match &reference {
                None => {
                    reference = Some(out);
                    "reference".to_string()
                }
                Some(want) => {
                    let wrong = want.iter().zip(&out).filter(|(a, b)| a != b).count();
                    if wrong == 0 {
                        "bit-identical".to_string()
                    } else {
                        format!("{wrong} differ")
                    }
                }
            };
            println!(
                "  {m}x{k}x{n} {variant:?}: {tasks} task(s), npu {ms:.2} ms, {verdict}, {sentinel} unwritten{}",
                if ms > 150.0 { " -- WATCHDOG KILL" } else { "" }
            );
        }
    }
}

/// The gate: `fc::Plan::try_new_row_major_output` is bit-identical to the
/// NC1HWC2 lowering, writes nothing past `[M, N]`, across one-row, odd-row,
/// two-atom, six-atom and multi-tile shapes. (N 8 and N 24 -- padded
/// channels -- overrun and corrupt here, which is why the plan refuses them.)
#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn row_major_matmul_output_matches_cube() {
    let mut failures = Vec::new();
    for (m, k, n) in [
        (1, 64, 64),
        (7, 32, 16),
        (16, 768, 48),
        (197, 768, 768),
        (577, 3072, 768),
        (577, 768, 2304),
    ] {
        std::thread::sleep(std::time::Duration::from_millis(800));
        let (reference, _, _, _) = run_full(m, k, n, Variant::WideCube, 1);
        std::thread::sleep(std::time::Duration::from_millis(800));
        let (dense, ms, tasks, past) = run_full(m, k, n, Variant::TallDense, 1);
        let wrong = reference.iter().zip(&dense).filter(|(a, b)| a != b).count();
        println!(
            "  {m}x{k}x{n}: {tasks} task(s), {wrong} differ, {past} bytes past [M, N], {ms:.2} ms"
        );
        if wrong != 0 || past != 0 || ms > 150.0 {
            failures.push(format!(
                "{m}x{k}x{n}: {wrong} differ, {past} past, {ms:.1} ms"
            ));
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
}
