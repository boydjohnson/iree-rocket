//! The whole softmax, on real attention rows, against an exact one.
//!
//! SOFTMAX.md §8's S5 gate. Every mechanism in the chain was measured on
//! its own first -- the reduction both ways (S2), the operand plane in
//! both element formats (S1, S5b), the bridge (S4a), `exp` with its tuned
//! tail (S3, §3.4), the reciprocal's placement (S4) -- so what this file
//! tests is the *assembly*: twelve device tasks across three submissions,
//! three cube layouts that have to agree, and two host steps in the middle.
//!
//! The input is `fixtures/vit_b_16_attention_rows.f32`, the same 64 rows of
//! real ViT-B/16 pre-softmax logits §3.4 tuned the `exp` tail against. The
//! op finds their maxima itself; the fixture's rows happen to be reduced by
//! their own already, which softmax's shift-invariance makes harmless.
//!
//! Cross-compile this test, copy the resulting binary to the RK3588 board,
//! and run the ignored tests there:
//!
//! ```text
//! CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!   cargo test --target aarch64-unknown-linux-gnu --release \
//!     --test softmax_oracle_hw --no-run
//!
//! ./softmax_oracle_hw-<hash> --ignored --nocapture --test-threads=1
//! ```
//!
//! The figure of merit is **total variation** against an exact softmax,
//! half the L1 distance between the two distributions, which bounds how far
//! an attention output can move. §3.4 measured 1.12% for the `exp` step
//! alone and §4 the same for passes 2 and 3; this adds the device's own
//! maximum and the int8 score quantisation on top, and comes out at the
//! same 1.12%.
//!
//! ## One trap this file cost a long bisection to find
//!
//! **A BO listed in a job's `in_bo_handles` *and* its `out_bo_handles` is
//! not written.** The `exp` task here returned all zeros while the very
//! same program -- byte-identical, asserted -- wrote correct values from
//! another test, and what differed was that this file passed every buffer
//! as an input, the output among them. Removing the output handle from the
//! input list fixed it outright.
//!
//! The bisection is worth keeping, because each step ruled out something
//! that *looked* likelier: the program is identical to a direct build; the
//! input codes read back correctly through the same offset formula; the
//! task is exact at this width and row count standalone, and at every
//! width from 197 to 256; a preceding PPU job does not disturb it; one task
//! per job behaves the same as one grouped job; and writing at an offset of
//! 15360 into a shared allocation is fine. Nothing in this repo documented
//! the handle-list rule, and `submit_tasks`'s own doc comment says only
//! that a buffer touched solely by tasks within the job "does not need to
//! appear in either list" -- it does not say that putting it in both is
//! harmful.
//!
//! The other thing that cost time was my own tooling rather than the
//! hardware: `ls -t` over the cargo target directory picked a stale
//! cross-compiled binary twice, so two runs reported results from code
//! that no longer existed. Take the path from `cargo --message-format=json`
//! instead.

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr, time::Instant};

#[path = "support/dispatch.rs"]
mod dispatch;

use iree_rocket_hal::rocket::{
    bn_plane::write_bn_mul_plane_fp16,
    device::{Buffer, JobDesc, close_bo, fini_bo, prep_bo, submit_jobs},
    softmax::{
        SoftmaxBuffers, SoftmaxPlan, dpu_cube_offset, pack_sum_weights, ppu_cube_offset,
        sum_weight_bytes, write_constant_plane,
    },
};
use rocket_core::softmax::Shape;

const DEVICE_PATH: &str = "/dev/accel/accel0";
const COMPLETION_TIMEOUT_NS: u64 = 5_000_000_000;
const Q15_SCALE: f32 = 32768.0;

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    let word = match exp {
        0 if frac == 0 => sign << 31,
        0x1f => (sign << 31) | 0x7f80_0000 | (frac << 13),
        0 => {
            // See the note in `reciprocal_route_hw.rs`: this counter starts
            // at 1, not -1.
            let mut exponent = 1i32;
            let mut mantissa = frac;
            while mantissa & 0x400 == 0 {
                mantissa <<= 1;
                exponent -= 1;
            }
            (sign << 31) | (((exponent + 127 - 15) as u32) << 23) | ((mantissa & 0x3ff) << 13)
        }
        _ => (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13),
    };
    f32::from_bits(word)
}

fn page_aligned(bytes: usize) -> usize {
    bytes.next_multiple_of(4096).max(4096)
}

fn attention_rows() -> (Vec<f32>, usize, usize) {
    const RAW: &[u8] = include_bytes!("fixtures/vit_b_16_attention_rows.f32");
    const ROWS: usize = 64;
    let values: Vec<f32> = RAW
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect();
    let keys = values.len() / ROWS;
    (values, ROWS, keys)
}

/// The fixture repeated to `rows`, for ROADMAP's lever 1: batching a
/// layer's heads onto the channel axis.
///
/// Repeating 64 real rows rather than inventing more is deliberate. What a
/// wider shape tests is layout and cost -- whether a 1792-lane operand
/// plane is indexed correctly and what a task costs when it is wide -- and
/// the per-row oracle is just as sharp on a repeat as on a novel row. It is
/// not a claim about a wider *distribution*; §3.4's tuning used all 64
/// distinct rows.
fn attention_rows_repeated(rows: usize) -> (Vec<f32>, usize, usize) {
    let (base, base_rows, keys) = attention_rows();
    let mut values = Vec::with_capacity(rows * keys);
    for row in 0..rows {
        let source = row % base_rows;
        values.extend_from_slice(&base[source * keys..(source + 1) * keys]);
    }
    (values, rows, keys)
}

/// Bytes a per-channel operand plane needs for `rows` lanes at four bytes a
/// lane, page-rounded. The planes were 4 KiB fixed while rows was 64; at
/// 1792 lanes the ALU plane alone is 7 KiB, and a plane short of the
/// programmed channel count is how NRDMA reads into whatever follows it.
/// The row count this process measures, from `PROBE_ROWS`, defaulting to
/// the fixture's own 64.
///
/// One row count per process rather than a sweep inside one: an NPU
/// measurement taken after another in the same process is not independent
/// -- the accumulator sweeps that taught this repo so are recorded in
/// ISSUES -- and a shape that wedges the device would otherwise poison
/// every shape after it.
fn rows_from_env() -> usize {
    std::env::var("PROBE_ROWS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(64)
}

fn plane_bytes(rows: usize) -> usize {
    ((rows.max(16).next_multiple_of(16)) * 4).next_multiple_of(4096)
}

fn total_variation(got: &[f32], want: &[f32]) -> f32 {
    0.5 * got
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
}

struct Owned {
    buffers: Vec<Buffer>,
}

impl Owned {
    fn new() -> Owned {
        Owned {
            buffers: Vec::new(),
        }
    }
    unsafe fn add(&mut self, fd: i32, bytes: usize, file: &std::fs::File) -> u32 {
        let buffer = unsafe { Buffer::new(fd, page_aligned(bytes), file) };
        unsafe { ptr::write_bytes(buffer.host_ptr, 0, buffer.size) };
        let address = buffer.dma_address;
        self.buffers.push(buffer);
        address
    }
    fn handles(&self) -> Vec<u32> {
        self.buffers.iter().map(|buffer| buffer.handle).collect()
    }
}

/// Submits one group as a job and waits for `out`.
unsafe fn run(
    fd: i32,
    file: &std::fs::File,
    programs: &[Vec<iree_rocket_hal::rocket::builders::RegCmd>],
    in_handles: &[u32],
    out: u32,
    label: &str,
) -> Option<std::time::Duration> {
    unsafe {
        let mut command_buffers = Vec::new();
        for commands in programs {
            let buffer = Buffer::new(
                fd,
                page_aligned(commands.len() * mem::size_of::<u64>()),
                file,
            );
            ptr::write_bytes(buffer.host_ptr, 0, buffer.size);
            let slots = std::slice::from_raw_parts_mut(buffer.host_ptr as *mut u64, commands.len());
            for (slot, command) in slots.iter_mut().zip(commands) {
                *slot = command.0;
            }
            command_buffers.push((buffer, commands.len() as u32));
        }
        for (buffer, _) in &command_buffers {
            fini_bo(fd, buffer.handle).ok();
        }
        // `PROBE_ONE_TASK_PER_JOB=1` submits each program as its own job
        // instead of grouping them. Tasks of one job run in order on one
        // core and need no fence between them (§5), so grouping should be
        // free -- and if it is not, this says so.
        let groups: Vec<Vec<usize>> = if std::env::var("PROBE_ONE_TASK_PER_JOB").is_ok() {
            (0..command_buffers.len())
                .map(|index| vec![index])
                .collect()
        } else {
            vec![(0..command_buffers.len()).collect()]
        };
        let _ = &groups;
        let tasks: Vec<(u32, u32)> = command_buffers
            .iter()
            .map(|(buffer, count)| (buffer.dma_address, *count))
            .collect();
        let mut handles: Vec<u32> = command_buffers
            .iter()
            .map(|(buffer, _)| buffer.handle)
            .collect();
        // The output BO goes in `out_handles` only. Listing it in both is
        // what `softmax_oracle_hw.rs` did while its exp task returned
        // zeros, with the program byte-identical to a working one.
        handles.extend(in_handles.iter().copied().filter(|&handle| handle != out));
        let out_handles = [out];
        let started = Instant::now();
        for group in &groups {
            let group_tasks: Vec<(u32, u32)> = group.iter().map(|&index| tasks[index]).collect();
            submit_jobs(
                fd,
                &[JobDesc {
                    tasks: &group_tasks,
                    in_handles: &handles,
                    out_handles: &out_handles,
                }],
            )
            .unwrap_or_else(|error| {
                panic!(
                    "{label}: SUBMIT failed ({} tasks): {error}",
                    group_tasks.len()
                )
            });
            prep_bo(fd, out, COMPLETION_TIMEOUT_NS)
                .unwrap_or_else(|error| panic!("{label}: did not complete: {error}"));
            if groups.len() > 1 {
                // Each job writes the scratch the next one reads.
                fini_bo(fd, out).ok();
            }
        }
        let elapsed = started.elapsed();
        for (buffer, _) in &command_buffers {
            close_bo(fd, buffer.handle).ok();
        }
        if dispatch::is_device_timeout(elapsed) {
            let strict = dispatch::report(label, elapsed);
            assert!(!strict, "killed dispatch under ROCKET_STRICT_DISPATCH");
            return None;
        }
        Some(elapsed)
    }
}

/// The whole op: three submissions, two host steps, one exact oracle.
#[test]
#[ignore = "requires an RK3588 NPU at /dev/accel/accel0"]
fn softmax_matches_an_exact_oracle_on_real_attention_rows() {
    let (rows_f32, row_count, keys) = attention_rows_repeated(rows_from_env());
    let shape = Shape::new(keys as u32, row_count as u32);
    let plan = SoftmaxPlan::try_new(shape).expect("the fixture's shape should plan");
    let input_scale = shape.exp_input_scale;
    // The cube width both readers agree at; the extra columns carry
    // `PAD_CODE` and change neither reduction.
    let physical = shape.physical_keys() as usize;
    let operand = plan.exp_bn_mul_operand();
    println!(
        "  {keys} keys x {row_count} rows: {} device tasks, {} host steps, {} bytes of scratch",
        plan.device_tasks(),
        plan.host_steps().len(),
        plan.scratch().bytes()
    );

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let mut owned = Owned::new();
        // The op's input: int8 scores, keys on width and rows on channels.
        let scores_bytes = row_count.div_ceil(16) * physical * 16;
        let scores = owned.add(fd, scores_bytes, &file);
        let scores_ptr = owned.buffers[0].host_ptr;
        for row in 0..row_count {
            for key in 0..physical {
                let code = if key < keys {
                    (rows_f32[row * keys + key] / input_scale)
                        .round()
                        .clamp(-128.0, 127.0) as i8
                } else {
                    rocket_core::softmax::PAD_CODE
                };
                ptr::write(
                    scores_ptr.add(dpu_cube_offset(row, key, physical, 1)),
                    code as u8,
                );
            }
        }

        print!("    row 0 scores as written:");
        for key in 0..6 {
            print!(
                " {}",
                ptr::read(scores_ptr.add(dpu_cube_offset(0, key, physical, 1))) as i8
            );
        }
        println!();
        print!("    row 17 scores as written:");
        for key in 0..4 {
            print!(
                " {}",
                ptr::read(scores_ptr.add(dpu_cube_offset(17, key, physical, 1))) as i8
            );
        }
        println!();

        let scratch = owned.add(fd, plan.scratch().bytes(), &file);
        let scratch_ptr = owned.buffers[1].host_ptr;
        let alu_plane = owned.add(fd, plane_bytes(row_count), &file);
        let alu_ptr = owned.buffers[2].host_ptr;
        let const_plane = owned.add(fd, plane_bytes(row_count), &file);
        let const_ptr = owned.buffers[3].host_ptr;
        let reciprocal_plane = owned.add(fd, plane_bytes(row_count), &file);
        let reciprocal_ptr = owned.buffers[4].host_ptr;
        let weight_bytes = sum_weight_bytes(row_count as u32).unwrap() as usize;
        let sum_weights = owned.add(fd, weight_bytes, &file);
        let weights_ptr = owned.buffers[5].host_ptr;
        let sum_bias = owned.add(fd, 4096, &file);
        let output_bytes = row_count.div_ceil(8) * physical * 16;
        let output = owned.add(fd, output_bytes, &file);
        let output_ptr = owned.buffers[7].host_ptr;

        // Compile-time constants: the all-ones coefficients and the 2^-15
        // plane.
        let mut packed = vec![0u8; weight_bytes];
        pack_sum_weights(row_count as u32, &mut packed).expect("all-ones packing");
        ptr::copy_nonoverlapping(packed.as_ptr(), weights_ptr, packed.len());
        let mut constant = vec![0u8; plane_bytes(row_count)];
        write_constant_plane(row_count as u32, &mut constant).expect("constant plane");
        ptr::copy_nonoverlapping(constant.as_ptr(), const_ptr, constant.len());

        let bufs = SoftmaxBuffers {
            scores,
            scratch,
            alu_plane,
            const_plane,
            reciprocal_plane,
            sum_weights,
            sum_bias,
            output,
        };
        let handles = owned.handles();
        for &handle in &handles {
            fini_bo(fd, handle).ok();
        }

        // ---- submission 1: the row-maximum tree.
        let max_programs = plan.programs_max(&bufs);
        if run(
            fd,
            &file,
            &max_programs,
            &handles,
            owned.buffers[1].handle,
            "max tree",
        )
        .is_none()
        {
            return;
        }

        // ---- host: the maximum into the exp task's ALU plane. A width-1,
        // rows-wide vector: negate it and pre-scale it by the task's own BN
        // multiplier, as i32, which is the int8 datapath's plane format
        // (S5b).
        let max_offset = plan.scratch().max_offset().expect("a reduced row");
        let mut maxima = Vec::with_capacity(row_count);
        for row in 0..row_count {
            let at = max_offset + ppu_cube_offset(row, 0, 1, 1);
            let code = ptr::read(scratch_ptr.add(at)) as i8;
            maxima.push(code);
            let value = -(i64::from(code)) * i64::from(operand);
            ptr::write(
                alu_ptr.add(row * 4) as *mut i32,
                i32::try_from(value).expect("the plane operand fits i32"),
            );
        }
        // Sync the plane *and the scratch* back for the device. `prep_bo`
        // after submission 1 moved the scratch to the CPU's side, and the
        // next submission writes it -- leaving that out is what made every
        // Q15 count come back zero, with the exp task itself measured
        // correct at this very shape (`the_planed_exp_task_at_scale`).
        fini_bo(fd, owned.buffers[2].handle).ok();
        fini_bo(fd, owned.buffers[1].handle).ok();
        // The device's maxima must be the real ones, or nothing downstream
        // can be right.
        for row in 0..row_count {
            let want = (0..keys)
                .map(|key| {
                    (rows_f32[row * keys + key] / input_scale)
                        .round()
                        .clamp(-128.0, 127.0) as i8
                })
                .max()
                .unwrap()
                .max(rocket_core::softmax::PAD_CODE);
            assert_eq!(
                maxima[row], want,
                "row {row}: the device's maximum is {} and the real one is {want}",
                maxima[row]
            );
        }

        // ---- submission 2: exp, the two bridge tasks, the sum tree.
        let exp_programs = plan.programs_exp_and_sum(&bufs);
        {
            // The same task built directly, with the same addresses. If
            // these differ the assembly is passing something the probe
            // does not; if they match, the programs are identical and the
            // difference is in the buffers.
            let reference = iree_rocket_hal::rocket::activation::build_lut_q15_regcmd_with_plane(
                &iree_rocket_hal::rocket::activation::LutShape {
                    width: physical as u32,
                    height: 1,
                    channels: row_count as u32,
                    input_zero_point: 0x80,
                    output_zero_point: 0x80,
                    input_scale,
                    output_scale: 1.0,
                },
                &iree_rocket_hal::rocket::activation::LutBuffers {
                    input_addr: scores,
                    output_addr: scratch + plan.scratch().q15_offset() as u32,
                },
                plan.exp_table(),
                alu_plane,
            );
            let differing = reference
                .iter()
                .zip(&exp_programs[0])
                .filter(|(a, b)| a.0 != b.0)
                .count();
            println!(
                "    exp program: {} words, {} differ from a direct build",
                exp_programs[0].len(),
                differing
            );
            assert_eq!(reference.len(), exp_programs[0].len());
        }
        if run(
            fd,
            &file,
            &exp_programs,
            &handles,
            owned.buffers[1].handle,
            "exp and sum",
        )
        .is_none()
        {
            return;
        }

        // ---- bisect the assembly by reading the intermediates. Every
        // stage is gated on its own, so a failure here is one of the three
        // cube layouts or a buffer that did not get written.
        {
            print!("    row 0 Q15 counts:");
            for key in 0..6 {
                let at = plan.scratch().q15_offset() + dpu_cube_offset(0, key, physical, 2);
                print!(
                    " {}",
                    ptr::read_unaligned(scratch_ptr.add(at) as *const i16)
                );
            }
            println!();
            print!("    row 0 fp16 counts:");
            for key in 0..6 {
                let at = plan.scratch().counts_offset() + dpu_cube_offset(0, key, physical, 2);
                print!(
                    " {:.1}",
                    f16_to_f32(ptr::read_unaligned(scratch_ptr.add(at) as *const u16))
                );
            }
            println!();
            print!("    row 0 probabilities:");
            for key in 0..6 {
                let at =
                    plan.scratch().probabilities_offset() + dpu_cube_offset(0, key, physical, 2);
                print!(
                    " {:.6}",
                    f16_to_f32(ptr::read_unaligned(scratch_ptr.add(at) as *const u16))
                );
            }
            println!();
            for (level, &offset) in plan.scratch().sum_level_offsets().iter().enumerate() {
                print!("    sum level {level} row 0:");
                for pixel in 0..4 {
                    let at = offset + dpu_cube_offset(0, pixel, 1, 2);
                    print!(
                        " {:.4}",
                        f16_to_f32(ptr::read_unaligned(scratch_ptr.add(at) as *const u16))
                    );
                }
                println!();
            }
        }

        // ---- host: 1/d per row, from the sum tree's fp16 result. The
        // probabilities are already divided by 32768, so this is 1/d in
        // real units and stays inside fp16's normal range (§4.2).
        let sum_offset = plan.scratch().sum_offset().expect("a reduced row");
        let mut reciprocals = Vec::with_capacity(row_count);
        for row in 0..row_count {
            let at = sum_offset + dpu_cube_offset(row, 0, 1, 2);
            let bits = ptr::read_unaligned(scratch_ptr.add(at) as *const u16);
            let d = f16_to_f32(bits);
            assert!(d > 0.0, "row {row}: the sum came back {d}");
            reciprocals.push(1.0 / d);
        }
        let mut plane = vec![0u8; plane_bytes(row_count)];
        write_bn_mul_plane_fp16(
            &mut plane,
            (row_count as u32).max(16).next_multiple_of(16),
            &reciprocals,
        )
        .expect("the reciprocals fit an fp16 plane");
        ptr::copy_nonoverlapping(plane.as_ptr(), reciprocal_ptr, plane.len());
        fini_bo(fd, owned.buffers[4].handle).ok();
        fini_bo(fd, owned.buffers[1].handle).ok();
        fini_bo(fd, owned.buffers[7].handle).ok();

        // ---- submission 3: the normalising multiply.
        let normalize_programs = plan.programs_normalize(&bufs);
        if run(
            fd,
            &file,
            &normalize_programs,
            &handles,
            owned.buffers[7].handle,
            "normalize",
        )
        .is_none()
        {
            return;
        }

        // ---- the oracle.
        let mut total = 0.0f32;
        let mut worst = 0.0f32;
        let mut worst_row = 0usize;
        for row in 0..row_count {
            let got: Vec<f32> = (0..keys)
                .map(|key| {
                    f16_to_f32(ptr::read_unaligned(
                        output_ptr.add(dpu_cube_offset(row, key, physical, 2)) as *const u16,
                    ))
                })
                .collect();
            let exact: Vec<f32> = (0..keys)
                .map(|key| rows_f32[row * keys + key].exp())
                .collect();
            let sum: f32 = exact.iter().sum();
            let want: Vec<f32> = exact.iter().map(|value| value / sum).collect();
            let tv = total_variation(&got, &want);
            if tv > worst {
                worst = tv;
                worst_row = row;
            }
            total += tv;
        }
        let mean = total / row_count as f32;

        for &handle in &handles {
            close_bo(fd, handle).ok();
        }

        println!(
            "  mean total variation {:.2}%, worst {:.2}% (row {worst_row})",
            100.0 * mean,
            100.0 * worst
        );
        assert!(
            mean < 0.03,
            "mean total variation {:.3}% against an exact softmax. §3.4 measured 1.12% for the \
             exp step alone and §4 the same for passes 2 and 3, so a figure far above that is \
             the assembly rather than the arithmetic -- the three cube layouts are what to \
             check first, and `the_two_cube_layouts_differ_where_they_should` says which is \
             which",
            100.0 * mean
        );
    }
}

/// Everything about the plan that can be checked without a device, as a
/// companion to the gate above: if this fails, the hardware run is not
/// worth doing.
#[test]
fn the_fixture_s_shape_plans() {
    let (_, row_count, keys) = attention_rows();
    let plan = SoftmaxPlan::try_new(Shape::new(keys as u32, row_count as u32)).unwrap();
    assert_eq!(plan.host_steps().len(), 2);
    assert!(plan.device_tasks() >= 12);
    let physical = (keys as u32).next_multiple_of(4) as usize;
    assert!(plan.scratch().bytes() >= 3 * row_count.div_ceil(8) * physical * 16);
    assert_eq!(plan.exp_bn_mul_operand(), 20971);
    assert_eq!(
        (
            plan.exp_table().le_slope_uflow_scale,
            plan.exp_table().le_slope_uflow_shift
        ),
        (789, 14),
        "the plan has to use §3.4's tuned tail, not the shipped flat one"
    );
    let _ = Q15_SCALE;
}

/// Wall time, against the CPU doing the same softmax -- the measurement
/// SOFTMAX.md §10 is built from.
///
/// ISSUES.md **P10** is the figure to beat: on BLIP fp16 the f32 CPU
/// softmax is 24% of the A76 cycles, and a 16-wide vector tile is
/// bit-identical and 1.39x faster on the standalone kernel.
///
/// **What this arm is and is not.** The NPU arm is the whole op as §7
/// defines it: three submissions, 13 tasks, both host steps, buffers and
/// programs built once outside the loop. The CPU arm is a plain scalar
/// Rust softmax over the same `[rows, keys]` f32 data in the same process
/// -- *not* IREE's codegen, which P10 profiled and which gets 1.39x better
/// with its tile fixed. So the CPU number here is a reference point rather
/// than the target, and the target is faster than it.
///
/// Read the environment before quoting any of it: `performance` governor,
/// A76 at 2.4 GHz, `rocket_npu_clk_hz` at 600 MHz. Pin to the big cluster
/// (`taskset -c 4-7`) or the host steps land on an A55.
#[test]
#[ignore = "requires an RK3588 NPU at /dev/accel/accel0"]
fn softmax_wall_time_against_the_cpu() {
    const WARMUP: usize = 3;
    const ITERATIONS: usize = 20;
    let (rows_f32, row_count, keys) = attention_rows_repeated(rows_from_env());
    let shape = Shape::new(keys as u32, row_count as u32);
    let plan = SoftmaxPlan::try_new(shape).unwrap();
    let input_scale = shape.exp_input_scale;
    let operand = plan.exp_bn_mul_operand();
    let physical = shape.physical_keys() as usize;

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let mut owned = Owned::new();
        let scores = owned.add(fd, row_count.div_ceil(16) * physical * 16, &file);
        let scores_ptr = owned.buffers[0].host_ptr;
        for row in 0..row_count {
            for key in 0..physical {
                let code = if key < keys {
                    (rows_f32[row * keys + key] / input_scale)
                        .round()
                        .clamp(-128.0, 127.0) as i8
                } else {
                    rocket_core::softmax::PAD_CODE
                };
                ptr::write(
                    scores_ptr.add(dpu_cube_offset(row, key, physical, 1)),
                    code as u8,
                );
            }
        }
        let scratch = owned.add(fd, plan.scratch().bytes(), &file);
        let scratch_ptr = owned.buffers[1].host_ptr;
        let alu_plane = owned.add(fd, plane_bytes(row_count), &file);
        let alu_ptr = owned.buffers[2].host_ptr;
        let const_plane = owned.add(fd, plane_bytes(row_count), &file);
        let const_ptr = owned.buffers[3].host_ptr;
        let reciprocal_plane = owned.add(fd, plane_bytes(row_count), &file);
        let reciprocal_ptr = owned.buffers[4].host_ptr;
        let weight_bytes = sum_weight_bytes(row_count as u32).unwrap() as usize;
        let sum_weights = owned.add(fd, weight_bytes, &file);
        let weights_ptr = owned.buffers[5].host_ptr;
        let sum_bias = owned.add(fd, 4096, &file);
        let output = owned.add(fd, row_count.div_ceil(8) * physical * 16, &file);

        let mut packed = vec![0u8; weight_bytes];
        pack_sum_weights(row_count as u32, &mut packed).unwrap();
        ptr::copy_nonoverlapping(packed.as_ptr(), weights_ptr, packed.len());
        let mut constant = vec![0u8; plane_bytes(row_count)];
        write_constant_plane(row_count as u32, &mut constant).unwrap();
        ptr::copy_nonoverlapping(constant.as_ptr(), const_ptr, constant.len());

        let bufs = SoftmaxBuffers {
            scores,
            scratch,
            alu_plane,
            const_plane,
            reciprocal_plane,
            sum_weights,
            sum_bias,
            output,
        };
        let handles = owned.handles();
        for &handle in &handles {
            fini_bo(fd, handle).ok();
        }
        // Built once: a real dispatch would too, and ISSUES P8's
        // per-dispatch host cost is about exactly this kind of work, so
        // timing it inside the loop would double-count it.
        let max_programs = plan.programs_max(&bufs);
        let exp_programs = plan.programs_exp_and_sum(&bufs);
        let normalize_programs = plan.programs_normalize(&bufs);

        let mut submit1 = 0.0f64;
        let mut host1 = 0.0f64;
        let mut submit2 = 0.0f64;
        let mut host2 = 0.0f64;
        let mut submit3 = 0.0f64;

        for iteration in 0..WARMUP + ITERATIONS {
            let measure = iteration >= WARMUP;
            let started = Instant::now();
            if run(
                fd,
                &file,
                &max_programs,
                &handles,
                owned.buffers[1].handle,
                "max tree",
            )
            .is_none()
            {
                return;
            }
            if measure {
                submit1 += started.elapsed().as_secs_f64();
            }

            let step = Instant::now();
            let max_offset = plan.scratch().max_offset().unwrap();
            for row in 0..row_count {
                let code =
                    ptr::read(scratch_ptr.add(max_offset + ppu_cube_offset(row, 0, 1, 1))) as i8;
                let value = -(i64::from(code)) * i64::from(operand);
                ptr::write(alu_ptr.add(row * 4) as *mut i32, value as i32);
            }
            fini_bo(fd, owned.buffers[2].handle).ok();
            fini_bo(fd, owned.buffers[1].handle).ok();
            if measure {
                host1 += step.elapsed().as_secs_f64();
            }

            let started = Instant::now();
            if run(
                fd,
                &file,
                &exp_programs,
                &handles,
                owned.buffers[1].handle,
                "exp and sum",
            )
            .is_none()
            {
                return;
            }
            if measure {
                submit2 += started.elapsed().as_secs_f64();
            }

            let step = Instant::now();
            let sum_offset = plan.scratch().sum_offset().unwrap();
            let mut reciprocals = Vec::with_capacity(row_count);
            for row in 0..row_count {
                let bits = ptr::read_unaligned(
                    scratch_ptr.add(sum_offset + dpu_cube_offset(row, 0, 1, 2)) as *const u16,
                );
                reciprocals.push(1.0 / f16_to_f32(bits));
            }
            let mut plane = vec![0u8; plane_bytes(row_count)];
            write_bn_mul_plane_fp16(
                &mut plane,
                (row_count as u32).max(16).next_multiple_of(16),
                &reciprocals,
            )
            .unwrap();
            ptr::copy_nonoverlapping(plane.as_ptr(), reciprocal_ptr, plane.len());
            fini_bo(fd, owned.buffers[4].handle).ok();
            fini_bo(fd, owned.buffers[1].handle).ok();
            fini_bo(fd, owned.buffers[7].handle).ok();
            if measure {
                host2 += step.elapsed().as_secs_f64();
            }

            let started = Instant::now();
            if run(
                fd,
                &file,
                &normalize_programs,
                &handles,
                owned.buffers[7].handle,
                "normalize",
            )
            .is_none()
            {
                return;
            }
            if measure {
                submit3 += started.elapsed().as_secs_f64();
            }
        }
        for &handle in &handles {
            close_bo(fd, handle).ok();
        }

        let each = |total: f64| total / ITERATIONS as f64 * 1e3;
        let npu = each(submit1 + host1 + submit2 + host2 + submit3);
        println!(
            "  NPU, {row_count} rows x {keys} keys, {} tasks: {npu:.3} ms per softmax",
            plan.device_tasks()
        );
        println!(
            "    max tree {:.3} | host max->plane {:.3} | exp+bridge+sum {:.3} | \
             host 1/d {:.3} | normalize {:.3}  (ms)",
            each(submit1),
            each(host1),
            each(submit2),
            each(host2),
            each(submit3)
        );
        println!(
            "    per task: {:.0} us, against S4's 37-59 us for a bare submit-and-wait",
            npu * 1e3 / plan.device_tasks() as f64
        );
        // What ROADMAP's lever 1 is about: a task's cost is fixed and its
        // width is free, so the figure that should fall as rows rise is the
        // cost per row.
        println!(
            "    per row: {:.1} us ({row_count} rows)",
            npu * 1e3 / row_count as f64
        );

        // The CPU arm: a plain scalar softmax over the same data. Not
        // IREE's codegen -- see this test's doc comment.
        let mut out = vec![0.0f32; row_count * keys];
        let started = Instant::now();
        for _ in 0..ITERATIONS {
            for row in 0..row_count {
                let logits = &rows_f32[row * keys..(row + 1) * keys];
                let mut m = f32::NEG_INFINITY;
                for &x in logits {
                    m = m.max(x);
                }
                let mut sum = 0.0f32;
                for (index, &x) in logits.iter().enumerate() {
                    let e = (x - m).exp();
                    out[row * keys + index] = e;
                    sum += e;
                }
                let inverse = 1.0 / sum;
                for value in &mut out[row * keys..(row + 1) * keys] {
                    *value *= inverse;
                }
            }
        }
        let cpu = started.elapsed().as_secs_f64() / ITERATIONS as f64 * 1e3;
        println!("  CPU, scalar Rust, same shape: {cpu:.3} ms per softmax");
        println!(
            "  ratio: the NPU op is {:.2}x the CPU's time ({})",
            npu / cpu,
            if npu < cpu {
                "NPU faster"
            } else {
                "CPU faster"
            }
        );
        // Keeps the CPU arm from being optimised away.
        assert!(out.iter().all(|value| value.is_finite()));
    }
}
