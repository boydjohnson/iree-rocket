//! ROADMAP lever 3: replace the exact row maximum with a calibrated upper
//! bound, deleting the max tree and the first host boundary with it.
//!
//! Softmax is shift-invariant, so the subtrahend need only be a true upper
//! bound. The max tree is **4 device tasks at every row count** (SOFTMAX.md
//! §11.1's census) and §10's breakdown puts it at 0.134 ms of 0.495 -- a
//! quarter of the op -- so this is worth about as much as lever 2's sum
//! tree is at small row counts and rather less at 1792.
//!
//! What nobody has measured is the two things that decide it, and both are
//! measurable through **one knob on the real op** rather than through a
//! model of it. The op's host step already writes the subtrahend by hand
//! into the `exp` task's BN ALU plane (`softmax_oracle_hw.rs`, submission 1
//! to 2), so writing `max + slack` there *is* lever 3, with no new
//! mechanism at all:
//!
//! 1. **What a loose bound costs in accuracy.** A bound loose by `slack`
//!    codes shifts every exponent down by `slack * input_scale`, which
//!    compresses the distribution toward the `exp` table's decaying tail
//!    and shrinks the Q15 numerators by `exp(-slack * input_scale)`. At
//!    `input_scale = 1/20` the accurate domain is 100 codes wide (§3.2's
//!    `-5.0` edge), so slack eats it directly.
//! 2. **What happens when the bound is wrong** -- a logit *above* it, which
//!    a calibrated bound can always meet on out-of-distribution input.
//!    ROADMAP says such a logit is "silently wrong by an unbounded amount".
//!    That is a guess about a mechanism nobody has driven, and there are
//!    two possibilities with very different consequences. The subtraction
//!    happens in the int8 code domain inside the DPU: if it **saturates**,
//!    the element lands at the table's top and returns `exp(0) = 1.0`, so
//!    the result is a valid distribution computed as if that logit sat
//!    exactly at the bound -- graceful, bounded, shippable behind a
//!    calibration check. If it **wraps**, a positive overflow becomes a
//!    large negative index, the largest logit in the row gets a probability
//!    of *zero*, and the answer is catastrophically wrong in the worst
//!    possible direction.
//!
//! A negative `PROBE_BOUND_SLACK` drives exactly that second case.
//!
//! One slack value per process, like `softmax_oracle_hw.rs`'s `PROBE_ROWS`
//! and for the same reason -- an NPU measurement taken after another in the
//! same process is not independent (ISSUES, the accumulator sweeps).
//!
//! Not run by a plain `cargo test`:
//!
//!   CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!     cargo test --target aarch64-unknown-linux-gnu --release \
//!       --test lever3_bound_slack_hw --no-run
//!
//! then on the board, one value per run:
//!
//!   for s in 0 5 10 20 40 80; do PROBE_BOUND_SLACK=$s ./lever3_... ; done
//!
//! The harness is `softmax_oracle_hw.rs`'s, duplicated rather than shared,
//! which is this crate's convention for hand-rolled hardware tests.

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

fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15);
    let exponent = u32::from((bits >> 10) & 0x1f);
    let fraction = u32::from(bits & 0x3ff);
    if exponent == 0 {
        // Subnormals count from one, not zero: starting this at `-1i32` is
        // a 4x error this repo has made in fifteen test files at once.
        let mut value = 0.0f32;
        for bit in 1i32..=10 {
            if fraction & (1 << (10 - bit)) != 0 {
                value += 2.0f32.powi(-14 - bit);
            }
        }
        return if sign == 1 { -value } else { value };
    }
    if exponent == 0x1f {
        return if sign == 1 {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };
    }
    f32::from_bits((sign << 31) | ((exponent + (127 - 15)) << 23) | (fraction << 13))
}

fn page_aligned(bytes: usize) -> usize {
    bytes.next_multiple_of(4096).max(4096)
}

fn attention_rows() -> (Vec<f32>, usize, usize) {
    const RAW: &[u8] = include_bytes!("fixtures/vit_b_16_attention_rows.f32");
    const ROWS: usize = 64;
    let values: Vec<f32> = RAW
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect();
    let keys = values.len() / ROWS;
    (values, ROWS, keys)
}

/// The bound's slack in **int8 code units**, from `PROBE_BOUND_SLACK`.
/// Positive is a loose bound (lever 3's accuracy question); negative is a
/// bound below the true maximum (lever 3's safety question); zero is the
/// op as it ships.
fn slack_from_env() -> i32 {
    std::env::var("PROBE_BOUND_SLACK")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
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

/// The score codes the op's input carries, as `softmax_oracle_hw.rs` writes
/// them.
fn score_code(value: f32, input_scale: f32) -> i8 {
    (value / input_scale).round().clamp(-128.0, 127.0) as i8
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
        let tasks: Vec<(u32, u32)> = command_buffers
            .iter()
            .map(|(buffer, count)| (buffer.dma_address, *count))
            .collect();
        let mut handles: Vec<u32> = command_buffers
            .iter()
            .map(|(buffer, _)| buffer.handle)
            .collect();
        // The output BO goes in `out_handles` only; listing it in both is
        // what made the oracle's exp task return zeros.
        handles.extend(in_handles.iter().copied().filter(|&handle| handle != out));
        let started = Instant::now();
        submit_jobs(
            fd,
            &[JobDesc {
                tasks: &tasks,
                in_handles: &handles,
                out_handles: &[out],
            }],
        )
        .unwrap_or_else(|error| panic!("{label}: SUBMIT failed: {error}"));
        prep_bo(fd, out, COMPLETION_TIMEOUT_NS)
            .unwrap_or_else(|error| panic!("{label}: did not complete: {error}"));
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

/// The fixture cannot measure lever 3's slack, and this pins why.
///
/// `tools/fixtures/dump_vit_attention_rows.py` stores
/// `scores - scores.max(dim=-1)`, so every row's maximum is already zero
/// and the row-max spread -- the whole quantity lever 3 turns on -- is zero
/// by construction. A first pass at §12.1 measured that zero and nearly
/// believed it.
///
/// So this is a tripwire, not a measurement: it asserts the fixture really
/// is pre-reduced, and if someone regenerates it without the reduction this
/// fires and points at the script that does measure the spread
/// (`tools/fixtures/head_bound_spread.py`, which needs torch and the raw
/// logits). The answer it found is in SOFTMAX.md §12.1: a median of 78
/// codes of slack on the typical row and 495 on the worst, against an
/// accurate `exp` domain 100 codes wide.
#[test]
fn the_fixture_is_pre_reduced_so_it_cannot_price_a_bound() {
    let (rows_f32, row_count, keys) = attention_rows();
    let input_scale = Shape::new(keys as u32, row_count as u32).exp_input_scale;

    let maxima: Vec<i8> = (0..row_count)
        .map(|row| {
            (0..keys)
                .map(|key| score_code(rows_f32[row * keys + key], input_scale))
                .max()
                .expect("a non-empty row")
        })
        .collect();

    let highest = *maxima.iter().max().expect("rows");
    let lowest = *maxima.iter().min().expect("rows");
    let domain_codes = (rocket_core::softmax::EXP_DOMAIN_EDGE / input_scale).round() as i32;
    println!(
        "fixture row maxima: all {row_count} rows peak at code {lowest}..{highest}; the \
         accurate exp domain is {domain_codes} codes wide"
    );
    assert_eq!(
        (lowest, highest),
        (0, 0),
        "the fixture is supposed to be stored already reduced by each row's own maximum \
         (dump_vit_attention_rows.py). It is not, so its row-max spread is now meaningful -- \
         which means SOFTMAX.md §12.1's 78-to-495-code figure should be re-derived from this \
         fixture rather than from head_bound_spread.py"
    );
}

/// The op end to end with `max + PROBE_BOUND_SLACK` as the subtrahend.
///
/// Prints mean and worst total variation, and the Q15 count the table
/// returned for each row's own largest code -- which is the number that
/// answers the safety question when the slack is negative: `32768` is
/// `exp(0)`, so a saturating subtraction holds there, while a wrap shows up
/// as a count near zero for the very element that should dominate.
#[test]
#[ignore = "requires an RK3588 NPU at /dev/accel/accel0"]
fn softmax_under_a_bound_with_slack() {
    let slack = slack_from_env();
    let (rows_f32, row_count, keys) = attention_rows();
    let shape = Shape::new(keys as u32, row_count as u32);
    let plan = SoftmaxPlan::try_new(shape).expect("the fixture's shape should plan");
    let input_scale = shape.exp_input_scale;
    let physical = shape.physical_keys() as usize;
    let operand = plan.exp_bn_mul_operand();

    println!(
        "bound slack {slack} codes ({:+.3} in real units), {} device tasks",
        slack as f32 * input_scale,
        plan.device_tasks()
    );

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let mut owned = Owned::new();
        let scores_bytes = row_count.div_ceil(16) * physical * 16;
        let scores = owned.add(fd, scores_bytes, &file);
        let scores_ptr = owned.buffers[0].host_ptr;
        for row in 0..row_count {
            for key in 0..physical {
                let code = if key < keys {
                    score_code(rows_f32[row * keys + key], input_scale)
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
        let output_bytes = row_count.div_ceil(8) * physical * 16;
        let output = owned.add(fd, output_bytes, &file);
        let output_ptr = owned.buffers[7].host_ptr;

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

        // ---- submission 1: the max tree. Lever 3 deletes this; here it
        // still runs, because the device's own maximum is what the slack is
        // measured *from*. A real lever 3 would skip it and take the bound
        // from calibration.
        if run(
            fd,
            &file,
            &plan.programs_max(&bufs),
            &handles,
            owned.buffers[1].handle,
            "max tree",
        )
        .is_none()
        {
            return;
        }

        // ---- host: the bound into the ALU plane. This single line is
        // lever 3.
        let max_offset = plan.scratch().max_offset().expect("a reduced row");
        let mut bounds = Vec::with_capacity(row_count);
        for row in 0..row_count {
            let at = max_offset + ppu_cube_offset(row, 0, 1, 1);
            let measured = i32::from(ptr::read(scratch_ptr.add(at)) as i8);
            let bound = measured + slack;
            bounds.push(bound);
            let value = -i64::from(bound) * i64::from(operand);
            ptr::write(
                alu_ptr.add(row * 4) as *mut i32,
                i32::try_from(value).expect("the plane operand fits i32"),
            );
        }
        fini_bo(fd, owned.buffers[2].handle).ok();
        fini_bo(fd, owned.buffers[1].handle).ok();

        // ---- submission 2: exp, the bridges, the sum tree.
        if run(
            fd,
            &file,
            &plan.programs_exp_and_sum(&bufs),
            &handles,
            owned.buffers[1].handle,
            "exp and sum",
        )
        .is_none()
        {
            return;
        }

        // The safety read: the Q15 count for each row's own largest code.
        // `exp(0)` is 32768, so a saturating subtraction pins the top
        // element there however far above the bound it sits; a wrap sends
        // it toward zero.
        let mut top_counts = Vec::with_capacity(row_count);
        for row in 0..row_count {
            let argmax = (0..keys)
                .max_by_key(|&key| score_code(rows_f32[row * keys + key], input_scale))
                .expect("a non-empty row");
            let at = plan.scratch().q15_offset() + dpu_cube_offset(row, argmax, physical, 2);
            top_counts.push(ptr::read_unaligned(scratch_ptr.add(at) as *const i16));
        }
        let lowest_top = *top_counts.iter().min().expect("rows");
        let highest_top = *top_counts.iter().max().expect("rows");
        println!(
            "  Q15 at each row's own largest code: lowest {lowest_top}, highest {highest_top} \
             (exp(0) is 32768, and exp({:+.3}) would be {:.0})",
            -(slack as f32) * input_scale,
            32768.0 * (-(slack as f32) * input_scale).exp()
        );

        // ---- host: 1/d, then submission 3.
        let sum_offset = plan.scratch().sum_offset().expect("a reduced row");
        let mut reciprocals = Vec::with_capacity(row_count);
        let mut dead_rows = 0;
        for row in 0..row_count {
            let at = sum_offset + dpu_cube_offset(row, 0, 1, 2);
            let d = f16_to_f32(ptr::read_unaligned(scratch_ptr.add(at) as *const u16));
            if d <= 0.0 {
                dead_rows += 1;
                reciprocals.push(0.0);
            } else {
                reciprocals.push(1.0 / d);
            }
        }
        if dead_rows > 0 {
            println!(
                "  {dead_rows} of {row_count} rows summed to zero -- the whole row underflowed"
            );
        }
        // Is the denominator itself right? `exp` is measurably exact above,
        // so if the output is inflated the sum tree is where it happened.
        // The true denominator under this bound is
        // `exp(-slack * input_scale) * sum_k exp(x_k - max_k)`, and the
        // Q15 chain's `2^-15` is already divided out (§4.2).
        {
            let mut worst_ratio = 1.0f32;
            let mut worst_at = 0usize;
            let mut live = 0usize;
            for row in 0..row_count {
                let at = sum_offset + dpu_cube_offset(row, 0, 1, 2);
                let measured = f16_to_f32(ptr::read_unaligned(scratch_ptr.add(at) as *const u16));
                if measured <= 0.0 {
                    continue;
                }
                let max_code = i32::from(
                    (0..keys)
                        .map(|key| score_code(rows_f32[row * keys + key], input_scale))
                        .max()
                        .expect("a non-empty row"),
                );
                let shift = (max_code + slack) as f32 * input_scale;
                let want: f32 = (0..keys)
                    .map(|key| (rows_f32[row * keys + key] - shift).exp())
                    .sum();
                live += 1;
                let ratio = want / measured;
                if ratio > worst_ratio {
                    worst_ratio = ratio;
                    worst_at = row;
                }
            }
            println!(
                "  denominator: over {live} live rows the sum tree is at worst {worst_ratio:.1}x                  too small (row {worst_at}) -- every factor of that inflates the output directly"
            );
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

        if run(
            fd,
            &file,
            &plan.programs_normalize(&bufs),
            &handles,
            owned.buffers[7].handle,
            "normalize",
        )
        .is_none()
        {
            return;
        }

        // ---- the oracle, unchanged: an exact f32 softmax of the same
        // rows. Shift invariance means the slack should not change it at
        // all, so every bit of the degradation below is the quantisation
        // the slack costs.
        let mut total = 0.0f32;
        let mut worst = 0.0f32;
        let mut worst_row = 0usize;
        let mut zeroed = 0usize;
        let mut exact_mass_zeroed = 0.0f32;
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
            for (key, &value) in got.iter().enumerate() {
                if value == 0.0 {
                    zeroed += 1;
                    exact_mass_zeroed += want[key];
                }
            }
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
        println!(
            "  {zeroed} of {} weights came back exactly zero, carrying {:.2}% of the true mass",
            row_count * keys,
            100.0 * exact_mass_zeroed / row_count as f32
        );
        println!(
            "  bound codes: lowest {}, highest {}",
            bounds.iter().min().expect("rows"),
            bounds.iter().max().expect("rows")
        );

        // The only assertion: at the shipped slack of zero this must still
        // be the op §10 measured. Every other slack is a measurement, and
        // judging it is the write-up's job, not an assert's.
        if slack == 0 {
            assert!(
                mean < 0.03,
                "at zero slack this is the shipped op and should be §10's 1.12%, not {:.2}%",
                100.0 * mean
            );
        }
    }
}
