//! SOFTMAX.md probe **S3**: `exp` on an fp16 datapath, its real domain, and
//! what lies past the edge.
//!
//! §3 of that document reasons about the `exp` LUT's domain as a function of
//! `input_scale` and concludes that `1/2` plus one squaring is the right
//! configuration. Two things had to be checked before any of that could be
//! acted on, and the first was not the arithmetic.
//!
//! **The LUT path in this crate is int8.** `build_lut_regcmd` programs
//! neither `in_precision`, `out_precision`, `proc_precision` nor
//! `mrdma_fp16tofp32_en` anywhere, so all four sit at 0 -- "Integer 8bit" --
//! and `LutShape` carries int8 zero points and quantization scales to match.
//! Every table in `activation.rs` has only ever run that way, `exp`
//! included (`lut_exp_hw.rs` uses `input_scale = 1/32` on int8 bytes). But
//! softmax's `exp` consumes a matmul's fp16 score cube. So this file gates
//! `build_lut_fp16_regcmd`, the variant that sets those four fields the way
//! `build_unary_regcmd` sets them for its own fp16 datapath, and drops the
//! two int8 concepts that do not carry over (no input zero point to
//! subtract, no output quantization to undo).
//!
//! Cross-compile this test, copy the resulting binary to the RK3588 board,
//! and run the ignored tests there:
//!
//! ```text
//! CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!   cargo test --target aarch64-unknown-linux-gnu --release \
//!     --test lut_exp_fp16_hw --no-run
//!
//! ./lut_exp_fp16_hw-<hash> --ignored --nocapture --test-threads=1
//! ```
//!
//! ## What §3 predicts, and where this file disagrees with it
//!
//! The LE table holds `exp(index / K_EXP) * 32768` for `index` in
//! `[-16384, 0)`, with `K_EXP = EXP_LUT_BN_SCALE_K = 3276.7184`, and the
//! hardware indexes it with `real_x * input_scale * K_EXP`. So at
//! `input_scale = s` the task computes `exp(x * s)` over a real domain of
//! `[-5/s, 0]`, and `le_slope_uflow_scale = 0` makes everything below that a
//! **flat clamp at the table's edge entry**, which is `exp(-5) = 6.74e-3`.
//!
//! §3 argues that the useful range is bounded not by that domain but by the
//! Q15 output encoding, "at about `x = -10.4`, not by the input domain", and
//! that `input_scale = 1/4` therefore "buys domain the output encoding
//! cannot express". **That is wrong, and arithmetic alone shows it**: the
//! table's edge entry is `round(exp(-5) * 32768) = 221`, nowhere near the
//! Q15 floor of 1. Nothing inside the table underflows, so the domain edge
//! binds first at every scale, and a wider domain is worth real accuracy
//! rather than nothing. `the_clamp_is_the_table_edge_not_the_q15_floor`
//! measures which it is.
//!
//! What survives unchanged is the recommendation: `input_scale = 1/2` with
//! one squaring covers `exp(x)` for `x` in `[-10, 0]` and clamps to
//! `(6.74e-3)^2 = 4.54e-5` below it, which over 384 keys is 0.017 of
//! spurious mass against a denominator of at least 1.

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr, time::Instant};

#[path = "support/dispatch.rs"]
mod dispatch;

use iree_rocket_hal::rocket::{
    activation::{LutBuffers, LutFp16Shape, LutTable, build_lut_fp16_regcmd},
    builders::{
        Bits, Register, RegisterMeta,
        dpu::{DpuOutCvtScale, DpuOutCvtShift},
    },
    device::{Buffer, close_bo, fini_bo, prep_bo, submit, submit_tasks},
    elementwise::{EwAddBuffers, EwAddShape, EwBinaryOp, EwPrecision, build_add_regcmd},
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const BUFFER_SIZE: usize = 64 * 1024;
const FEATURE_ATOM_BYTES: usize = 16;
/// `exp(-16384 / EXP_LUT_BN_SCALE_K)`, the LE table's edge entry as a real
/// value -- what a flat clamp returns below the domain.
const TABLE_EDGE: f32 = 0.006_737_947;

fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    if value == 0.0 {
        return sign;
    }
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let fraction = bits & 0x7f_ffff;
    if exponent < 1 {
        // Subnormal or zero in fp16; the probe never needs one as input.
        return sign;
    }
    assert!(exponent < 31, "{value} overflows fp16");
    let mut half = sign | ((exponent as u16) << 10) | ((fraction >> 13) as u16);
    let remainder = fraction & 0x1fff;
    if remainder > 0x1000 || (remainder == 0x1000 && half & 1 == 1) {
        half += 1;
    }
    half
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    let word = match exp {
        0 if frac == 0 => sign << 31,
        0x1f => (sign << 31) | 0x7f80_0000 | (frac << 13),
        0 => {
            // An fp16 subnormal is `mantissa * 2^-24`. Normalising it so
            // bit 10 is set after `j` shifts leaves `1.m * 2^(-14 - j)`, so
            // the f32 exponent field wants `113 - j` and this counter has
            // to start at **1**, not -1: `exponent + 127 - 15` is the field.
            // Starting at -1 reports every subnormal a factor of four low,
            // which is what it did until 2026-11-24 -- harmless in tests
            // whose values are of order 1, and not harmless in one reading
            // a softmax's small weights.
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

/// One fp16 value per pixel of a height-1 cube, channel 0 of each atom.
fn write_row(ptr_base: *mut u8, xs: &[f32]) {
    unsafe {
        for (pixel, &x) in xs.iter().enumerate() {
            ptr::write(
                ptr_base.add(pixel * FEATURE_ATOM_BYTES) as *mut u16,
                f32_to_f16(x),
            );
        }
    }
}

fn read_row(ptr_base: *const u8, count: usize) -> Vec<f32> {
    unsafe {
        (0..count)
            .map(|pixel| {
                f16_to_f32(ptr::read_unaligned(
                    ptr_base.add(pixel * FEATURE_ATOM_BYTES) as *const u16,
                ))
            })
            .collect()
    }
}

/// Runs the fp16 `exp` LUT over `xs`, optionally squaring the result with an
/// `EwBinaryOp::Mul` task chained into the same job.
///
/// Squaring is how §3 trades the table's resolution for its domain:
/// `exp(x/2)^2 = exp(x)`. Both operands of the multiply are the same buffer,
/// which `ew_binary_hw.rs` already exercises.
fn run_exp(input_scale: f32, xs: &[f32], square: bool) -> Option<Vec<f32>> {
    let width = xs.len() as u32;
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let buf_in = Buffer::new(fd, BUFFER_SIZE, &file);
        ptr::write_bytes(buf_in.host_ptr, 0, BUFFER_SIZE);
        write_row(buf_in.host_ptr, xs);

        let buf_lut = Buffer::new(fd, BUFFER_SIZE, &file);
        ptr::write_bytes(buf_lut.host_ptr, 0x5A, BUFFER_SIZE);
        let buf_out = Buffer::new(fd, BUFFER_SIZE, &file);
        ptr::write_bytes(buf_out.host_ptr, 0x5A, BUFFER_SIZE);

        let lut = build_lut_fp16_regcmd(
            &LutFp16Shape {
                width,
                height: 1,
                channels: 1,
                input_scale,
            },
            &LutBuffers {
                input_addr: buf_in.dma_address,
                output_addr: buf_lut.dma_address,
            },
            LutTable::exp(),
        );
        // PROBE_OUT_CVT_SCALE / _SHIFT override the Q15 undo, which is the
        // one part of the fp16 recipe with no capture behind it: the int8
        // path's `lut_out_cvt` targets an integer output, and the fp16
        // pass-throughs in this crate use scale 1 / shift 0 with
        // `fp32tofp16_en` doing the whole job.
        let mut lut = lut;
        if let (Ok(scale), Ok(shift)) = (
            std::env::var("PROBE_OUT_CVT_SCALE"),
            std::env::var("PROBE_OUT_CVT_SHIFT"),
        ) {
            let scale: u32 = scale.parse().unwrap();
            let shift: u32 = shift.parse().unwrap();
            let mut patched = 0;
            for cmd in &mut lut {
                if (cmd.0 >> 48) as u32 == DpuOutCvtScale::DOMAIN
                    && (cmd.0 as u32 & 0xffff) == DpuOutCvtScale::OFFSET
                {
                    *cmd = Register::<DpuOutCvtScale>::new()
                        .fp32tofp16_en(Bits::new(1))
                        .out_cvt_scale(Bits::new(scale))
                        .build();
                    patched += 1;
                }
                if (cmd.0 >> 48) as u32 == DpuOutCvtShift::DOMAIN
                    && (cmd.0 as u32 & 0xffff) == DpuOutCvtShift::OFFSET
                {
                    *cmd = Register::<DpuOutCvtShift>::new()
                        .out_cvt_shift(Bits::new(shift))
                        .build();
                    patched += 1;
                }
            }
            assert_eq!(patched, 2, "OUT_CVT_SCALE and _SHIFT once each");
        }
        let mut programs = vec![lut];
        if square {
            programs.push(build_add_regcmd(
                &EwAddShape {
                    width,
                    height: 1,
                    channels: 1,
                    precision: EwPrecision::Fp16,
                    op: EwBinaryOp::Mul,
                    int8: None,
                },
                &EwAddBuffers {
                    intermediate_addr: buf_lut.dma_address,
                    w_addr: buf_lut.dma_address,
                    output_addr: buf_out.dma_address,
                },
            ));
        }

        let mut cmd_bufs = Vec::new();
        for commands in &programs {
            let buffer = Buffer::new(
                fd,
                (commands.len() * mem::size_of::<u64>()).next_multiple_of(4096),
                &file,
            );
            ptr::write_bytes(buffer.host_ptr, 0, buffer.size);
            let slots = std::slice::from_raw_parts_mut(buffer.host_ptr as *mut u64, commands.len());
            for (slot, command) in slots.iter_mut().zip(commands) {
                *slot = command.0;
            }
            cmd_bufs.push((buffer, commands.len() as u32));
        }

        for handle in [buf_in.handle, buf_lut.handle, buf_out.handle] {
            fini_bo(fd, handle).ok();
        }
        for (buffer, _) in &cmd_bufs {
            fini_bo(fd, buffer.handle).ok();
        }

        let read_handle = if square {
            buf_out.handle
        } else {
            buf_lut.handle
        };
        let read_ptr = if square {
            buf_out.host_ptr
        } else {
            buf_lut.host_ptr
        };
        let mut in_handles: Vec<u32> = cmd_bufs.iter().map(|(b, _)| b.handle).collect();
        in_handles.push(buf_in.handle);

        let started = Instant::now();
        if square {
            let tasks: Vec<(u32, u32)> = cmd_bufs
                .iter()
                .map(|(buffer, count)| (buffer.dma_address, *count))
                .collect();
            submit_tasks(fd, &tasks, &in_handles, &[read_handle]).expect("SUBMIT failed");
        } else {
            submit(
                fd,
                cmd_bufs[0].0.dma_address,
                cmd_bufs[0].1,
                &in_handles,
                &[read_handle],
            )
            .expect("SUBMIT failed");
        }
        prep_bo(fd, read_handle, 2_000_000_000).unwrap_or_else(|error| {
            panic!(
                "the fp16 exp LUT (input_scale {input_scale}, square {square}) did not \
                 complete. This is the first use of the LUT block on an fp16 datapath, so a \
                 hang is a plausible outcome of the four precision fields being wrong rather \
                 than a surprise: {error}"
            )
        });
        let elapsed = started.elapsed();

        let out = read_row(read_ptr, xs.len());

        for handle in [buf_in.handle, buf_lut.handle, buf_out.handle] {
            close_bo(fd, handle).ok();
        }
        for (buffer, _) in &cmd_bufs {
            close_bo(fd, buffer.handle).ok();
        }

        if dispatch::is_device_timeout(elapsed) {
            let strict = dispatch::report(&format!("fp16 exp s={input_scale}"), elapsed);
            assert!(!strict, "killed dispatch under ROCKET_STRICT_DISPATCH");
            return None;
        }
        Some(out)
    }
}

/// What the fp16 LUT task does reach: the datapath, and `exp(0)`.
///
/// The four precision fields do something -- the job completes, in
/// milliseconds, and `exp(0)` comes back exactly 1.0 through an output
/// converter programmed `(1, 15)`, which is the Q15 value 32768 shifted
/// down by 15. So `build_lut_fp16_regcmd` is a valid program and the fp16
/// cube reaches the LUT. Everything past zero is the problem, and the two
/// tests below are what that problem is.
#[test]
#[ignore = "requires an RK3588 NPU at /dev/accel/accel0"]
fn the_fp16_lut_task_runs_and_exp_zero_is_exact() {
    let Some(got) = run_exp(1.0, &[0.0, 0.0, 0.0, 0.0], false) else {
        return;
    };
    for (pixel, &value) in got.iter().enumerate() {
        assert!(
            (value - 1.0).abs() <= 1e-3,
            "exp(0) at pixel {pixel} came back {value}; the fp16 LUT program does not even \
             reach the table"
        );
    }
    println!("  the fp16 LUT task runs and exp(0) is exact");
}

/// **The finding that decides §3: `OUT_CVT` is an integer multiply-shift,
/// so a Q15 table value cannot leave the task as a fraction.**
///
/// The table's entries are `real_value * 32768` -- a fixed-point encoding
/// independent of any op's quantization, as `lut_out_cvt`'s own doc comment
/// says. Turning one into a real number needs a multiply by `1/32768`, and
/// `OUT_CVT` cannot do it: at `(scale 1, shift 15)` the entry for `exp(0)`,
/// 32768, comes out as exactly 1.0, while the LE table's edge entry, 221,
/// comes out as exactly **0**. `221 >> 15` is 0 in integer arithmetic and
/// 0.006_74 in real arithmetic, so this is decisive, and it does not depend
/// on the indexing question below.
///
/// Consequence for SOFTMAX.md: the `exp` step cannot emit fp16
/// probabilities. Its output is a fixed-point Q15 integer, and §7 has to
/// either carry that through the sum and the divide -- where the 32768
/// cancels in the ratio, which is the promising direction -- or quantize
/// and dequantize around the LUT.
#[test]
#[ignore = "requires an RK3588 NPU at /dev/accel/accel0"]
fn out_cvt_cannot_turn_a_q15_entry_into_a_fraction() {
    // `(1, 0)` leaves the Q15 integer alone; `(1, 15)` is the 1/32768 the
    // real value needs. Both are run so the pair is the evidence.
    let xs = [0.0f32, -1.0];
    let raw = std::env::var("PROBE_OUT_CVT_SCALE").is_ok();
    assert!(
        !raw,
        "this test programs the converter itself; unset PROBE_OUT_CVT_*"
    );

    // The builder's own `lut_out_cvt(1.0)` is `(16384, 29)`, the int8
    // formula's answer for an output scale of 1. Measured: every value
    // underflows to 0 except an exact 1.0 at x = 0.
    let Some(builder) = run_exp(1.0, &xs, false) else {
        return;
    };
    assert!(
        (builder[0] - 1.0).abs() <= 1e-3,
        "exp(0) through the builder's converter came back {}, not 1.0",
        builder[0]
    );
    assert_eq!(
        builder[1], 0.0,
        "exp(-1) through the builder's converter came back {}. If it is a real fraction then \
         OUT_CVT is not integer-only and §3's exp step can stay in fp16 after all -- which \
         would be the better answer, so check it before believing this test's name.",
        builder[1]
    );
    println!(
        "  OUT_CVT is integer-only: the Q15 entry for exp(0) survives as 1.0 and every \
         smaller entry underflows to 0"
    );
}

/// The second, separate problem: the index does not track the real value.
///
/// At `input_scale = 1` the LE table's edge is reached by `x` of about
/// `-1.2e-4`, where the int8 formula's multiplier of 3276.72 predicts
/// `-5.0`. So the effective multiplier is four orders of magnitude too
/// large. `the_effective_index_multiplier` dumps the ramp this is read off.
///
/// Recorded rather than diagnosed. Two candidate mechanisms were tried and
/// neither fits: the BN stage scaling the widened float's *bit pattern* as
/// an integer predicts a particular near-constant output across `-0.5` to
/// `-4` under a compensating `input_scale`, and the hardware returned a
/// flat 1.0 instead; the BN stage truncating the float to an integer
/// predicts `exp(0)` for every `|x| < 1`, and the hardware returns the edge
/// entry. Settling it needs the BN stage's own float behaviour, which is a
/// probe of its own and is not on SOFTMAX.md's path while
/// `out_cvt_cannot_turn_a_q15_entry_into_a_fraction` stands.
#[test]
#[ignore = "requires an RK3588 NPU at /dev/accel/accel0"]
fn the_fp16_index_does_not_track_the_real_value() {
    let xs = [0.0f32, -0.000061, -0.00012, -0.5, -4.0];
    let Some(got) = run_exp(1.0, &xs, false) else {
        return;
    };
    // Past the measured transition everything is the table edge, which
    // under the builder's converter underflows to 0.
    for (&x, &value) in xs.iter().zip(&got).skip(2) {
        assert_eq!(
            value, 0.0,
            "x = {x} came back {value}. If the real exp values appear here the index is \
             tracking after all and §3's domain arithmetic applies."
        );
    }
    println!(
        "  the index reaches the table edge by x = -1.2e-4, where 3276.72 predicts -5.0: \
         four orders of magnitude out"
    );
}

/// Diagnostic for the output converter: dumps the raw fp16 words for a few
/// known inputs under one `(out_cvt_scale, out_cvt_shift)` pair from the
/// environment, one combination per process.
///
/// The first fp16 attempt used `lut_out_cvt(1.0)` -- `(16384, 29)`, the int8
/// path's formula asked for an output scale of 1 -- and every value came
/// back 0 while the job itself completed. So the question is what the LUT
/// leaves in the fp32 datapath: the table's Q15 *integer*, which needs the
/// 1/32768, or a value already scaled, in which case the fp16 convention of
/// scale 1 / shift 0 is right and the 1/32768 is applied twice.
#[test]
#[ignore = "requires an RK3588 NPU at /dev/accel/accel0"]
fn the_out_cvt_probe() {
    let xs = [0.0f32, -0.5, -1.0, -2.0, -4.0];
    let label = format!(
        "out_cvt scale {} shift {}",
        std::env::var("PROBE_OUT_CVT_SCALE").unwrap_or_else(|_| "builder".into()),
        std::env::var("PROBE_OUT_CVT_SHIFT").unwrap_or_else(|_| "builder".into()),
    );
    let input_scale = std::env::var("PROBE_INPUT_SCALE")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(1.0);
    match run_exp(input_scale, &xs, false) {
        None => println!("  {label}: NOT MEASURED"),
        Some(got) => {
            print!("  {label}:");
            for (&x, &value) in xs.iter().zip(&got) {
                print!("  exp({x})={value:.6}/{:.6}", x.exp());
            }
            println!();
        }
    }
}

/// Measures the effective index multiplier instead of guessing it.
///
/// At `input_scale = 1` the earlier dump showed `exp(0)` exact and every one
/// of -0.5, -1, -2, -4 returning the LE table's edge entry, so the
/// transition from "index 0" to "index <= -16384" happens somewhere inside
/// (-0.5, 0). Wherever the output first reaches that edge, `x_edge`, gives
/// the multiplier the hardware actually applied: `m = -16384 / x_edge`,
/// against the `3276.72` the int8 formula asks for.
///
/// Reports; asserts nothing. `PROBE_OUT_CVT_SCALE=1 PROBE_OUT_CVT_SHIFT=15`
/// makes the output a real value rather than a Q15 integer.
#[test]
#[ignore = "requires an RK3588 NPU at /dev/accel/accel0"]
fn the_effective_index_multiplier() {
    let xs: Vec<f32> = [
        0.0f32, -0.000061, -0.00012, -0.00024, -0.00049, -0.00098, -0.002, -0.004, -0.008, -0.0156,
        -0.0312, -0.0625, -0.125, -0.25, -0.5,
    ]
    .to_vec();
    let Some(got) = run_exp(1.0, &xs, false) else {
        return;
    };
    println!("  x, got, exp(x), implied index (from got = exp(index/K)):");
    for (&x, &value) in xs.iter().zip(&got) {
        let implied = if value > 0.0 && value <= 1.0 {
            format!("{:.1}", value.ln() * 3276.7184)
        } else {
            "-".into()
        };
        println!(
            "    {x:>10}  {value:>12.6}  {:>12.6}  {implied:>10}",
            x.exp()
        );
    }
    // The first x whose output has reached the edge bounds the multiplier.
    if let Some((edge_x, _)) = xs
        .iter()
        .zip(&got)
        .find(|(_, value)| (**value - TABLE_EDGE).abs() <= TABLE_EDGE * 0.02)
    {
        println!(
            "  first x at the table edge: {edge_x} -> effective multiplier about {:.1} \
             (the int8 formula asks for 3276.72)",
            -16384.0 / edge_x
        );
    } else {
        println!("  no x in this ramp reached the table edge");
    }
}
