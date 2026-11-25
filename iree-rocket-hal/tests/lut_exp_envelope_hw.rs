//! What `exp` can actually be asked for, on the int8 path that is already
//! hardware-validated -- SOFTMAX.md §3.1's follow-up.
//!
//! S3 established that the LUT's output converter is an integer
//! multiply-shift, so the `exp` step's result is a fixed-point Q15 integer
//! rather than an fp16 probability, and §9 asks whether §7 can be rebuilt
//! around that. Before any of that is worth designing, the *envelope* has
//! to be known, and one number decides it: what the table returns below its
//! domain edge.
//!
//! `LutTable::exp()` programs `le_slope_uflow_scale = 0`, a flat clamp,
//! which predicts a constant `exp(-5) = 6.74e-3` for every input past the
//! edge rather than a decay toward zero. If that is what the hardware does,
//! a softmax over `N` keys carries up to `N * 6.74e-3` of spurious mass in
//! its denominator -- 2.6 against a true denominator of about 1 to 2 at
//! `N = 384` -- and no amount of care downstream recovers it. If instead
//! the hardware returns zero there, the whole problem goes away.
//!
//! The fp16 probe suggested the edge entry, but that was the path whose
//! indexing is four orders of magnitude out, so the number came from a
//! configuration that is wrong for other reasons. This file asks the same
//! question on the int8 path, which `lut_exp_hw.rs` has gated against an
//! oracle.
//!
//! Cross-compile this test, copy the resulting binary to the RK3588 board,
//! and run the ignored tests there:
//!
//! ```text
//! CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!   cargo test --target aarch64-unknown-linux-gnu --release \
//!     --test lut_exp_envelope_hw --no-run
//!
//! ./lut_exp_envelope_hw-<hash> --ignored --nocapture --test-threads=1
//! ```
//!
//! The scales are chosen to make the tail visible rather than the peak.
//! `input_scale = 1/16` puts int8's 128 negative codes over real `x` in
//! `[-8, 0]`, so the codes run past the table's own `-5` edge.
//! `output_scale = 1/4096` resolves `6.74e-3` as 28 counts, where
//! `lut_exp_hw.rs`'s `1/128` would resolve it as 1 -- at the cost of
//! saturating everything above `0.031`, which is deliberate: this file is
//! about the tail.

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr, time::Instant};

#[path = "support/dispatch.rs"]
mod dispatch;

use iree_rocket_hal::rocket::{
    activation::{LutBuffers, LutShape, LutTable, build_lut_regcmd},
    builders::{Bits, Register, RegisterMeta, dpu::DpuEwCfg},
    device::{Buffer, close_bo, fini_bo, prep_bo, submit},
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const BUFFER_SIZE: usize = 64 * 1024;
const FEATURE_ATOM_BYTES: usize = 16;
/// `exp(-16384 / EXP_LUT_BN_SCALE_K)`: the LE table's edge entry as a real
/// value, and what a flat clamp predicts below the domain.
const TABLE_EDGE: f32 = 0.006_737_947;

/// Runs the int8 `exp` LUT over one signed code per pixel.
fn run_int8_exp(input_scale: f32, output_scale: f32, codes: &[i8]) -> Option<Vec<i8>> {
    run_int8_exp_with_slope(input_scale, output_scale, codes, None)
}

/// `run_int8_exp`, with `LutTable::exp()`'s underflow extrapolation slope
/// replaced.
///
/// The flat clamp past the domain edge is a *programmed* choice --
/// `le_slope_uflow_scale = 0`, which the vendor capture confirms for `exp`
/// and `tanh` while `sigmoid` carries a real nonzero slope, "consistent
/// with sigmoid's tail not being fully flat even far from zero". The field
/// is public on [`LutTable`], and a slope that took the tail to zero
/// instead of holding it at `exp(-5)` would remove the one thing that makes
/// this table unusable for a wide softmax. Worth a register.
fn run_int8_exp_with_slope(
    input_scale: f32,
    output_scale: f32,
    codes: &[i8],
    slope: Option<(u16, u8)>,
) -> Option<Vec<i8>> {
    // PROBE_RELU clears `EW_CFG.ew_relu_bypass`, which `build_lut_regcmd`
    // sets. A linear underflow extrapolation runs straight through zero
    // into negative output; a ReLU on the same stage is what would stop it
    // there, and a probability must not be negative.
    let relu = std::env::var("PROBE_RELU").is_ok();
    let width = codes.len() as u32;
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let buf_in = Buffer::new(fd, BUFFER_SIZE, &file);
        ptr::write_bytes(buf_in.host_ptr, 0, BUFFER_SIZE);
        for (pixel, &code) in codes.iter().enumerate() {
            ptr::write(buf_in.host_ptr.add(pixel * FEATURE_ATOM_BYTES), code as u8);
        }

        let buf_out = Buffer::new(fd, BUFFER_SIZE, &file);
        ptr::write_bytes(buf_out.host_ptr, 0x5A, BUFFER_SIZE);

        let cmds = build_lut_regcmd(
            &LutShape {
                width,
                height: 1,
                channels: 1,
                // Real zero point 0 on both sides: the one value
                // `lut_bn_alu` is exact for by construction.
                input_zero_point: 0x80,
                output_zero_point: 0x80,
                input_scale,
                output_scale,
            },
            &LutBuffers {
                input_addr: buf_in.dma_address,
                output_addr: buf_out.dma_address,
            },
            {
                let mut table = LutTable::exp();
                if let Some((scale, shift)) = slope {
                    table.le_slope_uflow_scale = scale;
                    table.le_slope_uflow_shift = shift;
                }
                table
            },
        );

        let mut cmds = cmds;
        if relu {
            let mut patched = 0;
            for cmd in &mut cmds {
                if (cmd.0 >> 48) as u32 == DpuEwCfg::DOMAIN
                    && (cmd.0 as u32 & 0xffff) == DpuEwCfg::OFFSET
                {
                    // build_lut_regcmd's own word with the ReLU engaged.
                    *cmd = Register::<DpuEwCfg>::new()
                        .ew_relu_bypass(Bits::new(0))
                        .ew_op_cvt_bypass(Bits::new(1))
                        .ew_lut_bypass(Bits::new(0))
                        .ew_op_bypass(Bits::new(1))
                        .ew_bypass(Bits::new(0))
                        .build();
                    patched += 1;
                }
            }
            assert_eq!(patched, 1, "EW_CFG is written exactly once");
        }

        let buf_cmd = Buffer::new(
            fd,
            (cmds.len() * mem::size_of::<u64>()).next_multiple_of(4096),
            &file,
        );
        ptr::write_bytes(buf_cmd.host_ptr, 0, buf_cmd.size);
        let slots = std::slice::from_raw_parts_mut(buf_cmd.host_ptr as *mut u64, cmds.len());
        for (slot, cmd) in slots.iter_mut().zip(&cmds) {
            *slot = cmd.0;
        }

        for handle in [buf_in.handle, buf_out.handle, buf_cmd.handle] {
            fini_bo(fd, handle).ok();
        }

        let started = Instant::now();
        submit(
            fd,
            buf_cmd.dma_address,
            cmds.len() as u32,
            &[buf_cmd.handle, buf_in.handle],
            &[buf_out.handle],
        )
        .expect("SUBMIT ioctl failed");
        prep_bo(fd, buf_out.handle, 2_000_000_000)
            .unwrap_or_else(|error| panic!("the int8 exp LUT did not complete: {error}"));
        let elapsed = started.elapsed();

        let out: Vec<i8> = (0..codes.len())
            .map(|pixel| ptr::read(buf_out.host_ptr.add(pixel * FEATURE_ATOM_BYTES)) as i8)
            .collect();

        for handle in [buf_in.handle, buf_out.handle, buf_cmd.handle] {
            close_bo(fd, handle).ok();
        }

        if dispatch::is_device_timeout(elapsed) {
            let strict = dispatch::report("int8 exp envelope", elapsed);
            assert!(!strict, "killed dispatch under ROCKET_STRICT_DISPATCH");
            return None;
        }
        Some(out)
    }
}

/// **The number SOFTMAX.md's remaining design question turns on.**
///
/// Codes spanning real `x` in `[-8, 0]` at `input_scale = 1/16`, so they run
/// past the table's own `-5` domain edge. Inside the domain the output must
/// track `exp`; past it, the flat clamp predicts a constant `exp(-5)`.
///
/// This test asserts the prediction and prints the whole ramp, because the
/// *alternative* is the better answer for the design: a clamp to zero would
/// mean a softmax's tail contributes nothing and §7 survives whole.
#[test]
#[ignore = "requires an RK3588 NPU at /dev/accel/accel0"]
fn the_int8_tail_below_the_domain_edge_is_the_table_edge() {
    let input_scale = 1.0 / 16.0;
    let output_scale = 1.0 / 4096.0;
    // -8.0 up to 0 in steps of 0.25, which is 4 codes at this scale.
    let codes: Vec<i8> = (0..=32).map(|i| -(i as i32 * 4) as i8).collect();
    let Some(out) = run_int8_exp(input_scale, output_scale, &codes) else {
        return;
    };

    println!("  real x, got, exp(x), counts");
    for (&code, &value) in codes.iter().zip(&out) {
        let x = f32::from(code) * input_scale;
        println!(
            "    {x:>6.2}  {:>10.6}  {:>10.6}  {value:>5}",
            f32::from(value) * output_scale,
            x.exp()
        );
    }

    // Inside the domain, and above the output's own saturation, nothing is
    // asserted -- `lut_exp_hw.rs` gates that against an oracle at a scale
    // chosen for the peak. This test is only about what happens past -5.
    let tail: Vec<(f32, f32)> = codes
        .iter()
        .zip(&out)
        .map(|(&code, &value)| {
            (
                f32::from(code) * input_scale,
                f32::from(value) * output_scale,
            )
        })
        .filter(|&(x, _)| x < -5.2)
        .collect();
    assert!(!tail.is_empty(), "the ramp did not reach past the edge");

    let flat = tail
        .iter()
        .all(|&(_, got)| (got - TABLE_EDGE).abs() <= 2.0 * output_scale);
    let zero = tail.iter().all(|&(_, got)| got == 0.0);
    assert!(
        flat || zero,
        "past the domain edge the output is neither the table edge ({TABLE_EDGE:.3e}) nor \
         zero: {tail:?}. Whatever it is decides SOFTMAX.md section 9's open question, so \
         record it."
    );
    if zero {
        println!(
            "  past -5 the output is ZERO, not the table edge: a softmax tail contributes \
             nothing and section 7 survives the Q15 output whole"
        );
    } else {
        println!(
            "  past -5 the output is a flat {TABLE_EDGE:.3e}. Over N keys that is \
             N * 6.74e-3 of spurious denominator -- 2.6 at N = 384, against a true \
             denominator of order 1 -- so a softmax wider than a few dozen keys cannot use \
             this table as it stands"
        );
    }
    assert!(
        flat,
        "the flat clamp is what `le_slope_uflow_scale = 0` predicts; a zero would be better \
         and this assertion is how it gets noticed"
    );
}

/// Can the underflow slope take the tail to zero?
///
/// The flat clamp at `exp(-5)` is what makes this table unusable for a
/// softmax wider than a few dozen keys, and it is one register. The
/// candidates decay the edge entry's 221 Q15 counts to zero over a chosen
/// distance below the edge: `(4420, 14)` by `x = -5.25`, `(2210, 14)` by
/// `-5.5`, `(1105, 14)` by `-6.0`, `(553, 14)` by `-7.0`. One combination
/// per process via `PROBE_SLOPE_SCALE` / `PROBE_SLOPE_SHIFT`; with neither
/// set it reports the shipped flat clamp for comparison.
///
/// Reports rather than asserts: the shape of the extrapolation is the
/// question, and the slope field's sign convention is not documented in
/// this tree.
#[test]
#[ignore = "requires an RK3588 NPU at /dev/accel/accel0"]
fn the_underflow_slope_probe() {
    let slope = match (
        std::env::var("PROBE_SLOPE_SCALE").ok(),
        std::env::var("PROBE_SLOPE_SHIFT").ok(),
    ) {
        (Some(scale), Some(shift)) => {
            Some((scale.parse::<u16>().unwrap(), shift.parse::<u8>().unwrap()))
        }
        _ => None,
    };
    let input_scale = 1.0 / 16.0;
    let output_scale = 1.0 / 4096.0;
    // -4.5 to -8.0, which straddles the edge and goes well past it.
    let codes: Vec<i8> = (18..=32).map(|i| -(i as i32 * 4) as i8).collect();
    let Some(out) = run_int8_exp_with_slope(input_scale, output_scale, &codes, slope) else {
        println!("  {slope:?}: NOT MEASURED");
        return;
    };
    print!("  slope {slope:?}:");
    for (&code, &value) in codes.iter().zip(&out) {
        print!(" {:.2}={}", f32::from(code) * input_scale, value);
    }
    println!();
}

/// The configuration SOFTMAX.md §3.3 settles on, asserted rather than
/// reported: `LutTable::exp_with_decaying_tail` makes the tail decay
/// monotonically to exactly zero at the most negative input the scale can
/// express, and never below it.
///
/// A probability must not be negative, and a steeper slope produces exactly
/// that -- `(4420, 14)` crosses zero at `x = -5.25` and reaches the int8
/// floor of -128 by `-6.5`. Since the EW ReLU is inert on a LUT task
/// (`the_ew_relu_is_inert_on_a_lut_task`), tuning the slope to the domain is
/// the only mechanism that keeps the tail non-negative without a second
/// task.
#[test]
#[ignore = "requires an RK3588 NPU at /dev/accel/accel0"]
fn the_tuned_tail_decays_to_zero_and_never_below() {
    let input_scale = 1.0 / 16.0;
    let output_scale = 1.0 / 4096.0;
    let table = LutTable::exp_with_decaying_tail(input_scale);
    assert_eq!(
        (table.le_slope_uflow_scale, table.le_slope_uflow_shift),
        (368, 14),
        "the slope for input_scale 1/16, which is the one measured on hardware"
    );

    let codes: Vec<i8> = (0..=32).map(|i| -(i as i32 * 4) as i8).collect();
    let Some(out) = run_int8_exp_with_slope(
        input_scale,
        output_scale,
        &codes,
        Some((table.le_slope_uflow_scale, table.le_slope_uflow_shift)),
    ) else {
        return;
    };

    let mut previous = i8::MAX;
    for (&code, &value) in codes.iter().zip(&out) {
        let x = f32::from(code) * input_scale;
        assert!(
            value >= 0,
            "x = {x} came back {value} counts: a probability must not be negative, and a \
             slope tuned to the domain is what should prevent it"
        );
        assert!(
            value <= previous,
            "x = {x} came back {value} counts after {previous}: the tail must decrease"
        );
        previous = value;
    }
    assert_eq!(
        out[out.len() - 1],
        0,
        "at the most negative int8 code the tail should reach exactly zero"
    );
    println!(
        "  the tuned tail decays monotonically from the edge to exactly 0 at x = -8.0, \
         non-negative throughout"
    );
}

/// `EW_CFG.ew_relu_bypass = 0` has no effect on a LUT task: the output is
/// byte-identical with the ReLU engaged and bypassed, negative
/// extrapolation included.
///
/// Worth its own test because it is the obvious way to stop a linear
/// underflow ramp going negative, and it does not work -- which is why
/// `exp_with_decaying_tail` tunes the slope instead.
#[test]
#[ignore = "requires an RK3588 NPU at /dev/accel/accel0"]
fn the_ew_relu_is_inert_on_a_lut_task() {
    let input_scale = 1.0 / 16.0;
    let output_scale = 1.0 / 4096.0;
    // A slope steep enough to drive the output well negative.
    let slope = Some((1105u16, 14u8));
    let codes: Vec<i8> = (24..=32).map(|i| -(i as i32 * 4) as i8).collect();

    let Some(bypassed) = run_int8_exp_with_slope(input_scale, output_scale, &codes, slope) else {
        return;
    };
    assert!(
        bypassed.iter().any(|&value| value < 0),
        "this test needs a slope that drives the output negative; got {bypassed:?}"
    );
    // SAFETY-free: the probe reads PROBE_RELU itself, and the tests run
    // single-threaded by instruction at the top of this file.
    unsafe { std::env::set_var("PROBE_RELU", "1") };
    let engaged = run_int8_exp_with_slope(input_scale, output_scale, &codes, slope);
    unsafe { std::env::remove_var("PROBE_RELU") };
    let Some(engaged) = engaged else { return };

    assert_eq!(
        bypassed, engaged,
        "the EW ReLU changed the output, so it is not inert after all and a LUT task can \
         clamp its own negative extrapolation -- which would be the better answer"
    );
    println!("  the EW ReLU is inert on a LUT task: {bypassed:?} either way");
}
