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
    activation::{LutBuffers, LutShape, LutTable, build_lut_q15_regcmd, build_lut_regcmd},
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

/// Runs the int8 `exp` LUT with the Q15 integer written straight to memory
/// as `int16` -- `build_lut_q15_regcmd`. One code per pixel, one `i16` back.
fn run_int8_exp_q15(input_scale: f32, codes: &[i8], table: &LutTable) -> Option<Vec<i16>> {
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

        let cmds = build_lut_q15_regcmd(
            &LutShape {
                width,
                height: 1,
                channels: 1,
                input_zero_point: 0x80,
                output_zero_point: 0x80,
                input_scale,
                // Ignored by this builder; the converter sits at identity.
                output_scale: 1.0,
            },
            &LutBuffers {
                input_addr: buf_in.dma_address,
                output_addr: buf_out.dma_address,
            },
            *table,
        );
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
        prep_bo(fd, buf_out.handle, 2_000_000_000).unwrap_or_else(|error| {
            panic!("the Q15 exp LUT did not complete (first use of int16 output): {error}")
        });
        let elapsed = started.elapsed();
        let out: Vec<i16> = (0..codes.len())
            .map(|pixel| {
                ptr::read_unaligned(buf_out.host_ptr.add(pixel * FEATURE_ATOM_BYTES) as *const i16)
            })
            .collect();
        for handle in [buf_in.handle, buf_out.handle, buf_cmd.handle] {
            close_bo(fd, handle).ok();
        }
        if dispatch::is_device_timeout(elapsed) {
            let strict = dispatch::report("Q15 exp", elapsed);
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

// ===========================================================================
// Tuning the tail against a real attention distribution.
// ===========================================================================
//
// §3.3 left one judgement open: which `(input_scale, slope)` pair to use.
// The two error terms pull against each other -- the int8 input's
// quantization wants a *fine* scale, since a code step of `s` puts a
// relative error of up to `s/2` on every weight, while the table's `-5`
// domain edge wants a *coarse* one, since anything past the reach of
// `-128 * s` is approximated by the ramp. Picking the pair at the domain's
// corner, as §3.3 did, is not the same as picking it where the data is.
//
// `fixtures/vit_b_16_attention_rows.f32` is 64 rows of real pre-softmax
// attention logits, already reduced by their own row maximum: torchvision's
// ViT-B/16 at IMAGENET1K_V1 weights on `mlir-spike/Unicorn.jpg`, scores
// recomputed from each block's `in_proj_weight` rather than read from
// `output_attentions` (which gives probabilities, not logits), sampled every
// (12 layers x 12 heads x 197 rows)/64'th row. Its shape is what makes the
// choice: median `-3.44`, and **28% of all entries fall past the table's
// `-5` edge**, 8.7% past `-8`, 1.4% past `-16`.
//
// A host sweep over that fixture under the measured LUT shape puts the
// optimum at `input_scale = 1/20`, and the curve is shallow around it:
//
// ```text
//   1/s   reach   slope   d err %   TV %    flat-clamp TV %
//    24    -5.3    3315      1.98   2.13
//    20    -6.4     789      0.70   1.11              6.69
//    16    -8.0     368      1.38   1.60              6.77
//    12   -10.7     195      3.51   3.25
// ```
//
// TV is total variation -- half the L1 distance between the LUT's softmax
// and an exact one, which bounds how far the attention output can move.
// Relative error on an individual weight is not a useful figure here: it is
// dominated by weights of order 1e-9 whose absolute error is nil.

/// The 64 real rows, as `(rows, keys)`.
fn attention_rows() -> (Vec<f32>, usize, usize) {
    const RAW: &[u8] = include_bytes!("fixtures/vit_b_16_attention_rows.f32");
    const ROWS: usize = 64;
    let values: Vec<f32> = RAW
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect();
    let keys = values.len() / ROWS;
    assert_eq!(keys, 197, "ViT-B/16 has 197 tokens");
    (values, ROWS, keys)
}

/// Half the L1 distance between two distributions.
fn total_variation(got: &[f32], want: &[f32]) -> f32 {
    0.5 * got
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
}

/// The tuned pair, measured end to end on real attention logits.
///
/// Every value the device touches is real: the input codes come from
/// ViT-B/16's own logits, the `exp` is the hardware's, and only the
/// denominator and the division happen on the host (which is where §4 puts
/// them anyway, pending the reciprocal). The output is the table's Q15
/// integer read back through `build_lut_q15_regcmd`, because an int8 output
/// quantizes a probability to 1/128 and would swamp the figure being
/// measured.
///
/// The bound is the host sweep's prediction with room for the hardware to
/// differ from the model of it: 1.11% predicted, asserted under 2%.
#[test]
#[ignore = "requires an RK3588 NPU at /dev/accel/accel0"]
fn the_tuned_pair_reaches_one_percent_on_real_attention_rows() {
    let input_scale = 1.0 / 20.0;
    let (rows, row_count, keys) = attention_rows();
    let table = LutTable::exp_with_decaying_tail(input_scale);
    assert_eq!(
        (table.le_slope_uflow_scale, table.le_slope_uflow_shift),
        (789, 14),
        "the pair the host sweep picked"
    );

    let mut worst = 0.0f32;
    let mut total = 0.0f32;
    let mut measured = 0usize;
    for row in 0..row_count {
        let logits = &rows[row * keys..(row + 1) * keys];
        let codes: Vec<i8> = logits
            .iter()
            .map(|&x| (x / input_scale).round().clamp(-128.0, 0.0) as i8)
            .collect();
        let Some(counts) = run_int8_exp_q15(input_scale, &codes, &table) else {
            continue;
        };
        measured += 1;

        // Q15 integers, so the 32768 cancels in the ratio and never has to
        // be divided out at all -- which is the whole point of §9's plan.
        let sum: f32 = counts.iter().map(|&c| f32::from(c)).sum();
        assert!(sum > 0.0, "row {row} summed to zero");
        let got: Vec<f32> = counts.iter().map(|&c| f32::from(c) / sum).collect();

        let exact: Vec<f32> = logits.iter().map(|&x| x.exp()).collect();
        let exact_sum: f32 = exact.iter().sum();
        let want: Vec<f32> = exact.iter().map(|&value| value / exact_sum).collect();

        let tv = total_variation(&got, &want);
        worst = worst.max(tv);
        total += tv;
    }
    assert!(measured > 0, "no row was measured");
    let mean = total / measured as f32;
    println!(
        "  {measured} real ViT-B/16 rows at input_scale 1/20, slope (789, 14): mean total \
         variation {:.2}%, worst {:.2}%",
        100.0 * mean,
        100.0 * worst
    );
    assert!(
        mean < 0.02,
        "mean total variation {:.3}% against an exact softmax; the host sweep over the same \
         fixture predicted 1.11%, so a figure far above this is the hardware departing from \
         the piecewise model §3.3 was tuned against",
        100.0 * mean
    );
}

/// The flat clamp on the same rows, for the comparison that justifies the
/// slope at all. Reported, not asserted: it is the arm this design does not
/// use.
#[test]
#[ignore = "requires an RK3588 NPU at /dev/accel/accel0"]
fn the_flat_clamp_on_real_attention_rows() {
    let input_scale = 1.0 / 20.0;
    let (rows, row_count, keys) = attention_rows();
    let mut total = 0.0f32;
    let mut measured = 0usize;
    for row in 0..row_count {
        let logits = &rows[row * keys..(row + 1) * keys];
        let codes: Vec<i8> = logits
            .iter()
            .map(|&x| (x / input_scale).round().clamp(-128.0, 0.0) as i8)
            .collect();
        let Some(counts) = run_int8_exp_q15(input_scale, &codes, &LutTable::exp()) else {
            continue;
        };
        measured += 1;
        let sum: f32 = counts.iter().map(|&c| f32::from(c)).sum();
        let got: Vec<f32> = counts.iter().map(|&c| f32::from(c) / sum).collect();
        let exact: Vec<f32> = logits.iter().map(|&x| x.exp()).collect();
        let exact_sum: f32 = exact.iter().sum();
        let want: Vec<f32> = exact.iter().map(|&value| value / exact_sum).collect();
        total += total_variation(&got, &want);
    }
    assert!(measured > 0);
    println!(
        "  the same {measured} rows with the shipped flat clamp: mean total variation {:.2}% \
         (the host sweep predicted 6.69%)",
        100.0 * total / measured as f32
    );
}
