//! Hardware-in-the-loop probe of the two LUT hazards `ISSUES.md` C5 raises:
//! a `q = 0` table entry that mis-decodes to a garbage `~4.0` (QUIRK 4),
//! and a discrete `+128` mux spike in a razor-thin band around `x = 0`
//! (QUIRK 2). Both are documented in `../rockchip-npu-notes/encodings/
//! dpu-lut-activation.md`, found on a *different* stack (an fp16-output
//! emitter driving self-built `build_lut_shifted` tables); C5's open
//! question is whether either reaches this crate's int8-output
//! `build_lut_regcmd` with its vendor-captured / self-derived tables.
//!
//! Cross-compile, copy to the RK3588 board (`planck`), run there:
//!
//! ```text
//! CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!   cargo test --target aarch64-unknown-linux-gnu --release \
//!     --test lut_zero_join_hw --no-run
//!
//! ./lut_zero_join_hw-<hash> --ignored --nocapture
//! ```
//!
//! # Why the existing per-kind tests cannot see either hazard
//!
//! Every LUT test in this tree (`lut_hw.rs`, `lut_erf_hw.rs`, ...) drives
//! `ptr::write_bytes(buf, fill, ..)` -- one *uniform* input value per NPU
//! job -- and reads back the first 16 bytes. So a kind is gated at ~6-13
//! of the 256 possible int8 input codes, and the tails and the join
//! neighbourhood are whatever those hand-picked fills happen to cover.
//! The notes are explicit that a sparse gate steps straight over QUIRK 2's
//! band and that only dense sampling finds it.
//!
//! This file replaces the uniform fill with a **per-element input
//! pattern**: `build_lut_regcmd` gives the input and output cubes
//! identical geometry (`w=4,h=4,c=16`, `surface_stride = w*h*c`, no
//! padding), and the LUT is pointwise, so element `i` of the output is
//! `f(element i of the input)` whatever the NC1HWC2 walk order actually
//! is. One job therefore covers all 256 input codes instead of one. That
//! also keeps the job count *down* -- a full-domain sweep costs 1 job per
//! kind, not 256 -- which matters given how badly a wedged NPU
//! contaminates a long test run.
//!
//! `lut_ramp_agrees_with_uniform_fill` is the guard on that reasoning: if
//! the walk order were not shared between DPU_RDMA and DPU_WDMA, the ramp
//! would come back permuted and every oracle below would fail for a
//! harness reason rather than a hardware one. Run it first.
//!
//! # Where the `q = 0` entries actually are
//!
//! C5 lists them; reading the tables confirms the list, and adds the
//! detail that makes this file short. `push_lut_tables_and_config` sets
//! `LUT_LE_START = -16384`, `LUT_LE_END = 0`, `LUT_LO_START = 0`,
//! `LUT_LO_END = 16384`, so **LE[512] and LO[0] are both the fixed-domain
//! index 0, i.e. real `x = 0`** -- the LE/LO join. Every `q = 0` entry
//! C5 names except one sits exactly there:
//!
//! | table | zero entries | real `x` |
//! |---|---|---|
//! | `TANH_LE` / `TANH_LO` | 512 / 0 | 0 |
//! | `ERF_LE` / `ERF_LO` | 512 / 0 | 0 |
//! | `SQUARE_LE` / `SQUARE_LO` | 510-512 / 0-2 | `|x| <= 2/16384` |
//! | `SQRT_LO` | 0 | 0 |
//! | `LOG_LO` | 128 | `x = 1` (`log 1 = 0`) |
//! | `SQRT_LE`, `RSQRT_LE`, `LOG_LE` | **all 513** | every `x < 0` |
//!
//! So QUIRK 4's target and QUIRK 2's band are the *same* neighbourhood for
//! six of the nine kinds, and one dense probe around zero tests both.
//! `LOG_LO[128]` is the exception and is covered by the full-domain sweep
//! (`log`'s existing gate already drives `x = 1` at fill 64, so it is the
//! one `q = 0` entry with prior hardware evidence).
//!
//! The last row is the interesting one: `sqrt`/`rsqrt`/`log` carry an
//! all-zero *placeholder* LE table, so if QUIRK 4 reaches this stack, a
//! negative input to any of them returns garbage rather than the harmless
//! ~0 the placeholder was chosen to give. Those codes are outside each
//! kind's declared domain so the sweep does not *assert* on them, but it
//! does report them -- see `report_out_of_domain`.
//!
//! # What a hazard looks like in *this* decode
//!
//! The notes quote QUIRK 4 as "emits ~4.0" and QUIRK 2 as "+128", both in
//! an fp16-output world. Here the output is int8: a table entry `q`
//! decodes as `q / 32768 / output_scale` codes (`lut_out_cvt`), so a real
//! `~4.0` where `~0` was expected saturates the byte. Both hazards
//! therefore show up as **a discrete jump of tens of codes, or a
//! saturation to +-127, at or beside `x = 0`** -- nothing like the 1-3 LSB
//! interpolation error the oracles otherwise see. `SPIKE_LSB` splits the
//! two so a mild systematic drift is not reported as a spike.

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    activation::{LutBuffers, LutShape, LutTable, build_lut_regcmd},
    device::{Buffer, close_bo, fini_bo, prep_bo, submit},
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const TENSOR_SIZE: usize = 4096;

/// `w * h * c` for every shape in this file, and therefore exactly the
/// number of distinct int8 input codes -- which is why one job can carry
/// the whole domain.
const ELEMENTS: usize = 256;

/// An error this large is not interpolation or quantization: it is a
/// QUIRK-2/QUIRK-4 style discrete garbage value. Kept well above every
/// per-kind `tolerance_lsb` (the widest is 4) and well below the ~128
/// codes either hazard would actually produce.
const SPIKE_LSB: f32 = 16.0;

fn shape_with(input_scale: f32, output_scale: f32) -> LutShape {
    LutShape {
        width: 4,
        height: 4,
        channels: 16,
        // Decoded zero point 0 for both sides: the only zero point
        // `lut_bn_alu` is exact for by construction, and the one every
        // per-kind oracle test in this tree already uses.
        input_zero_point: 0x80,
        output_zero_point: 0x80,
        input_scale,
        output_scale,
    }
}

/// Element `i` carries input code `i - 128`, so real input rises
/// monotonically with the element index. Inverse: code `c` lives at
/// element `(c as u8).wrapping_sub(128)`.
fn ramp() -> [u8; ELEMENTS] {
    let mut pattern = [0u8; ELEMENTS];
    for (i, byte) in pattern.iter_mut().enumerate() {
        *byte = (i as u8).wrapping_add(128);
    }
    pattern
}

fn element_of_code(code: i8) -> usize {
    (code as u8).wrapping_sub(128) as usize
}

/// One NPU job with an arbitrary per-element input, returning all
/// `ELEMENTS` output bytes -- the uniform-fill harness the other LUT
/// tests share, generalized in exactly those two places.
fn run_patterned_standalone_lut(
    shape: &LutShape,
    table: LutTable,
    input: &[u8; ELEMENTS],
    what: &str,
) -> [u8; ELEMENTS] {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let buf_a = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_a.host_ptr, 0, TENSOR_SIZE);
        ptr::copy_nonoverlapping(input.as_ptr(), buf_a.host_ptr, ELEMENTS);

        let buf_c = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_c.host_ptr, 0, TENSOR_SIZE);

        let bufs = LutBuffers {
            input_addr: buf_a.dma_address,
            output_addr: buf_c.dma_address,
        };
        let cmds = build_lut_regcmd(shape, &bufs, table);

        let cmd_bytes = cmds.len() * mem::size_of::<u64>();
        let cmd_len = cmd_bytes.next_multiple_of(4096);
        let buf_cmd = Buffer::new(fd, cmd_len, &file);
        let cmd_slice = std::slice::from_raw_parts_mut(buf_cmd.host_ptr as *mut u64, cmds.len());
        for (i, c) in cmds.iter().enumerate() {
            cmd_slice[i] = c.0;
        }

        fini_bo(fd, buf_a.handle).ok();
        fini_bo(fd, buf_c.handle).ok();
        fini_bo(fd, buf_cmd.handle).ok();

        let in_handles = [buf_cmd.handle, buf_a.handle];
        let out_handles = [buf_c.handle];

        submit(
            fd,
            buf_cmd.dma_address,
            cmds.len() as u32,
            &in_handles,
            &out_handles,
        )
        .expect("SUBMIT ioctl failed");

        prep_bo(fd, buf_c.handle, 2_000_000_000).unwrap_or_else(|e| {
            panic!(
                "standalone LUT job ({what}) did not complete within timeout -- DPU/MRDMA \
                 flying-mode LUT config may have hung the NPU: {e}"
            )
        });

        let mut out = [0u8; ELEMENTS];
        ptr::copy_nonoverlapping(buf_c.host_ptr, out.as_mut_ptr(), ELEMENTS);

        close_bo(fd, buf_a.handle).ok();
        close_bo(fd, buf_c.handle).ok();
        close_bo(fd, buf_cmd.handle).ok();

        out
    }
}

/// The uniform-fill harness the other LUT tests use, expressed through the
/// patterned one so `lut_ramp_agrees_with_uniform_fill` compares two paths
/// that differ *only* in the input bytes.
fn run_uniform_standalone_lut(shape: &LutShape, table: LutTable, fill: u8) -> [u8; ELEMENTS] {
    run_patterned_standalone_lut(
        shape,
        table,
        &[fill; ELEMENTS],
        &format!("uniform fill {fill}"),
    )
}

/// `erf` matching Rust's not-yet-stable `f32::erf` -- Abramowitz & Stegun
/// 7.1.26, max error ~1.5e-7. Copied verbatim from `lut_erf_hw.rs` so the
/// two tests cannot disagree for an oracle reason rather than a hardware
/// one.
#[allow(clippy::excessive_precision)]
fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

/// One kind under one shape. `valid` is the inclusive int8 input-code
/// range over which that kind's tables hold real data -- outside it the
/// tables are clamps or placeholders (see each `LutTable` constructor's
/// doc comment), so the oracle does not apply and the codes are reported
/// rather than asserted.
struct Sweep {
    name: &'static str,
    table: fn() -> LutTable,
    oracle: fn(f32) -> f32,
    input_scale: f32,
    output_scale: f32,
    valid: (i8, i8),
    tolerance_lsb: f32,
}

/// Per-code result of one sweep, in real units.
struct Outcome {
    worst_err: f32,
    worst_code: i8,
    violations: usize,
    spikes: Vec<String>,
}

fn evaluate(sweep: &Sweep, out: &[u8; ELEMENTS]) -> Outcome {
    let shape = shape_with(sweep.input_scale, sweep.output_scale);
    let tolerance = sweep.tolerance_lsb * sweep.output_scale;
    let spike = SPIKE_LSB * sweep.output_scale;

    let mut outcome = Outcome {
        worst_err: 0.0,
        worst_code: 0,
        violations: 0,
        spikes: Vec::new(),
    };

    for code in sweep.valid.0..=sweep.valid.1 {
        let x = code as f32 * shape.input_scale;
        let expected = (sweep.oracle)(x);
        let got = (out[element_of_code(code)] as i8) as f32 * shape.output_scale;
        let err = (got - expected).abs();

        if err > outcome.worst_err {
            outcome.worst_err = err;
            outcome.worst_code = code;
        }
        if err > tolerance {
            outcome.violations += 1;
        }
        if err > spike && outcome.spikes.len() < 8 {
            outcome.spikes.push(format!(
                "code {code} (x={x}): want {expected}, got {got}, err {err}"
            ));
        }
    }
    outcome
}

/// Prints what the hardware did outside the kind's declared domain. Not
/// asserted -- the tables there are clamps or all-zero placeholders, so
/// there is no oracle -- but it is exactly where QUIRK 4 would bite
/// (`SQRT_LE`/`RSQRT_LE`/`LOG_LE` are 513 `q = 0` entries), and a
/// saturated `+-127` here is the signature.
fn report_out_of_domain(sweep: &Sweep, out: &[u8; ELEMENTS]) {
    let mut saturated = Vec::new();
    for code in -128i32..=127 {
        let code = code as i8;
        if code >= sweep.valid.0 && code <= sweep.valid.1 {
            continue;
        }
        let got = (out[element_of_code(code)] as i8) as i32;
        if got.abs() >= 127 {
            saturated.push(code);
        }
    }
    if saturated.is_empty() {
        eprintln!("  {}: out-of-domain codes: none saturated", sweep.name);
    } else {
        eprintln!(
            "  {}: out-of-domain codes saturated to +-127: {} of them, first {:?}",
            sweep.name,
            saturated.len(),
            &saturated[..saturated.len().min(12)]
        );
    }
}

fn run_sweeps(sweeps: &[Sweep], title: &str) {
    eprintln!("=== {title} ===");
    let mut failures = Vec::new();

    for sweep in sweeps {
        let shape = shape_with(sweep.input_scale, sweep.output_scale);
        let out = run_patterned_standalone_lut(&shape, (sweep.table)(), &ramp(), sweep.name);
        let outcome = evaluate(sweep, &out);

        let at_zero = (out[element_of_code(0)] as i8) as i32;
        eprintln!(
            "  {}: worst |err| {:.6} ({:.2} LSB) at code {}, {} violations of {} LSB, \
             {} spikes; raw byte at x=0 (the LE/LO join) = {}",
            sweep.name,
            outcome.worst_err,
            outcome.worst_err / sweep.output_scale,
            outcome.worst_code,
            outcome.violations,
            sweep.tolerance_lsb,
            outcome.spikes.len(),
            at_zero,
        );
        for spike in &outcome.spikes {
            eprintln!("    SPIKE {spike}");
        }
        report_out_of_domain(sweep, &out);

        if outcome.violations > 0 {
            failures.push(format!(
                "{}: {} of {} in-domain codes exceed {} LSB (worst {:.6} = {:.2} LSB at code {}){}",
                sweep.name,
                outcome.violations,
                sweep.valid.1 as i32 - sweep.valid.0 as i32 + 1,
                sweep.tolerance_lsb,
                outcome.worst_err,
                outcome.worst_err / sweep.output_scale,
                outcome.worst_code,
                if outcome.spikes.is_empty() {
                    String::new()
                } else {
                    format!("\n      {}", outcome.spikes.join("\n      "))
                },
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "{title}: {} of {} kinds failed:\n  {}",
        failures.len(),
        sweeps.len(),
        failures.join("\n  ")
    );
}

/// Guard on this file's central assumption: that DPU_RDMA and DPU_WDMA
/// walk the cube in the same order, so a patterned input can be compared
/// element-wise against an oracle. If they did not, every sweep below
/// would fail for a harness reason.
///
/// Checks two independent things. First, that a monotone input ramp
/// produces a monotone output for a monotone kind (`tanh`) -- a permuted
/// walk would shred that ordering. Second, that the ramp's answer at four
/// individual codes equals what the established uniform-fill harness
/// returns for those same codes, which is the direct statement that the
/// two harnesses measure the same thing.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn lut_ramp_agrees_with_uniform_fill() {
    // tanh's own gated shape (`lut_hw.rs::lut_standalone_tanh_matches_oracle`).
    let shape = shape_with(1.0 / 32.0, 1.0 / 128.0);
    let out = run_patterned_standalone_lut(&shape, LutTable::tanh(), &ramp(), "tanh ramp");

    let mut prev = (out[0] as i8) as i32;
    for i in 1..ELEMENTS {
        let code = (out[i] as i8) as i32;
        assert!(
            code >= prev,
            "tanh output is not monotone across the input ramp at element {i} \
             (previous {prev}, current {code}) -- the input and output cubes are being \
             walked in different orders, so every element-wise oracle in this file is \
             invalid. Full output: {:?}",
            out.iter().map(|&b| (b as i8) as i32).collect::<Vec<_>>()
        );
        prev = code;
    }

    for fill in [128u8, 250, 0, 127] {
        let code = fill as i8;
        let uniform = run_uniform_standalone_lut(&shape, LutTable::tanh(), fill)[0];
        let from_ramp = out[element_of_code(code)];
        eprintln!("tanh code {code}: uniform-fill {uniform}, ramp {from_ramp}");
        assert_eq!(
            uniform, from_ramp,
            "tanh at input code {code}: the uniform-fill harness returned {uniform} but the \
             ramp returned {from_ramp}. The two disagree, so the LUT result depends on the \
             *other* elements of the cube, not just the element's own value -- which would \
             invalidate both this file and the uniform-fill gates it generalizes."
        );
    }
}

/// Every kind, every one of the 256 int8 input codes, one NPU job each.
/// This is the "drive the tails at least once" half of C5: it reaches
/// `TANH_LE[512]`/`TANH_LO[0]`, `ERF_LE[512]`/`ERF_LO[0]`,
/// `SQUARE_LE[510..512]`/`SQUARE_LO[0..2]`, `SQRT_LO[0]` and
/// `LOG_LO[128]` -- the whole `q = 0` inventory C5 lists -- plus both
/// saturating ends of every table, none of which the 6-13 hand-picked
/// fills per kind reach today.
///
/// Shapes and tolerances are inherited from each kind's own gated test so
/// a failure here is a coverage finding and not a re-litigation of a shape
/// choice; `sigmoid` is the exception (see its entry).
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn lut_full_code_sweep_matches_oracle() {
    let sweeps = [
        // Saturating, valid across the whole code range. `lut_hw.rs`.
        Sweep {
            name: "tanh",
            table: LutTable::tanh,
            oracle: |x| x.tanh(),
            input_scale: 1.0 / 32.0,
            output_scale: 1.0 / 128.0,
            valid: (-128, 127),
            tolerance_lsb: 2.0,
        },
        // Saturating within `x in [-4, 4]`. `lut_erf_hw.rs`.
        Sweep {
            name: "erf",
            table: LutTable::erf,
            oracle: erf,
            input_scale: 1.0 / 32.0,
            output_scale: 1.0 / 128.0,
            valid: (-128, 127),
            tolerance_lsb: 2.0,
        },
        // `EXP_LO` is a placeholder constant 1.0 and cannot be fixed in
        // this Q15 encoding (`LutTable::exp`'s doc comment), so only the
        // `x <= 0` half has an oracle. `lut_exp_hw.rs`.
        Sweep {
            name: "exp",
            table: LutTable::exp,
            oracle: |x| x.exp(),
            input_scale: 1.0 / 32.0,
            output_scale: 1.0 / 128.0,
            valid: (-128, 0),
            tolerance_lsb: 2.0,
        },
        // Accurate only inside `|x| <= 1`, which at this scale is exactly
        // the code range. `ew_square_hw.rs`.
        Sweep {
            name: "square",
            table: LutTable::square,
            oracle: |x| x * x,
            input_scale: 1.0 / 128.0,
            output_scale: 1.0 / 128.0,
            valid: (-128, 127),
            tolerance_lsb: 2.0,
        },
        // `SQRT_LE` is an all-zero placeholder: `x < 0` is undefined, not
        // merely inaccurate. `lut_sqrt_hw.rs`.
        Sweep {
            name: "sqrt",
            table: LutTable::sqrt,
            oracle: |x| x.sqrt(),
            input_scale: 1.0 / 128.0,
            output_scale: 1.0 / 128.0,
            valid: (0, 127),
            tolerance_lsb: 4.0,
        },
        // Accurate only for `x in [1, 16)` -> codes 8..=127 at scale 1/8.
        // `lut_rsqrt_hw.rs`.
        Sweep {
            name: "rsqrt",
            table: LutTable::rsqrt,
            oracle: |x| 1.0 / x.sqrt(),
            input_scale: 1.0 / 8.0,
            output_scale: 1.0 / 128.0,
            valid: (8, 127),
            tolerance_lsb: 3.0,
        },
        // Accurate only for `x in [1/e, e)` -- `|log x| <= 1.0` is all
        // the Q15 table encoding holds -- which at scale 1/64 is codes
        // 24..=127 (the table's own grid puts the first non-clamped entry
        // at `x = 0.375`, code 24). That is exactly where the existing
        // gate's fills start.
        //
        // The first run of this sweep set `valid` to `(2, 127)`, from the
        // `[0.02, e)` bound `LutTable::log`'s doc comment then claimed,
        // and hardware returned a flat `-1.0` for codes 2..=23. That was
        // the table's documented `-32768` clamp behaving correctly and a
        // wrong doc comment, not a hardware hazard; both doc comments are
        // now fixed. Kept as a comment because "log silently returns -1.0
        // below 0.375" is the kind of thing a model author needs to know.
        //
        // `LOG_LO[128]` (`log 1 = 0`, code 64) is the one `q = 0` entry
        // C5 lists that already had hardware evidence. `lut_log_hw.rs`.
        Sweep {
            name: "log",
            table: LutTable::log,
            oracle: |x| x.ln(),
            input_scale: 1.0 / 64.0,
            output_scale: 1.0 / 128.0,
            valid: (24, 127),
            tolerance_lsb: 3.0,
        },
        // Odd function, real data on both halves, accurate for
        // `|x| in [1, 16)` -> codes +-8..=+-127 at scale 1/8. Swept as two
        // entries because `valid` is a single interval and the gap around
        // zero is genuinely out of domain. `lut_reciprocal_hw.rs`.
        Sweep {
            name: "reciprocal (positive half)",
            table: LutTable::reciprocal,
            oracle: |x| 1.0 / x,
            input_scale: 1.0 / 8.0,
            output_scale: 1.0 / 128.0,
            valid: (8, 127),
            tolerance_lsb: 3.0,
        },
        Sweep {
            name: "reciprocal (negative half)",
            table: LutTable::reciprocal,
            oracle: |x| 1.0 / x,
            input_scale: 1.0 / 8.0,
            output_scale: 1.0 / 128.0,
            valid: (-128, -8),
            tolerance_lsb: 3.0,
        },
        // The only kind here without an oracle-gated shape of its own --
        // `lut_hw.rs` gates sigmoid for monotonicity only, at an
        // `output_zero_point` of 0 whose decode "wraps around byte zero".
        // This entry re-frames it onto the signed-decode convention every
        // other kind uses, which needs `output_scale` no finer than 1/64
        // to keep sigmoid's `[0, 1]` range inside a signed byte. So a
        // sigmoid-only failure here is more likely this shape choice than
        // a hardware hazard; a *spike* would not be.
        Sweep {
            name: "sigmoid",
            table: LutTable::sigmoid,
            oracle: |x| 1.0 / (1.0 + (-x).exp()),
            input_scale: 1.0 / 32.0,
            output_scale: 1.0 / 64.0,
            valid: (-128, 127),
            tolerance_lsb: 3.0,
        },
    ];

    run_sweeps(&sweeps, "full 256-code sweep");
}

/// The "dense sweep near 0" half of C5, and the direct test of both
/// hazards.
///
/// `input_scale = 1/32768` collapses all 256 codes into real `x` in
/// `[-0.0039, +0.0039]` at a step of `3.05e-5`. QUIRK 2's band is quoted
/// as `~+-0.0015`, so roughly 100 of the 256 codes land *inside* it and
/// the rest sit just outside -- far denser than the notes' own
/// `step-1e-5` probe needs to be, because int8 input means there is
/// nothing finer to sample. In fixed-domain terms every code lands in the
/// single table cell on each side of the join (`|index| <= 21` against a
/// 32-wide cell), so this is the most concentrated probe of `LE[512]`,
/// `LO[0]` and the mux between them that the hardware allows.
///
/// `output_scale` is chosen per kind so the *expected* answer occupies a
/// small, well-separated part of the byte: a discrete `+128` code spike or
/// a `~4.0` mis-decode saturates, and cannot be confused with
/// interpolation error.
///
/// Only kinds whose tables hold real data at `x = 0` are here.
/// `sqrt`/`rsqrt`/`log`/`reciprocal` are undefined or clamped at zero, so
/// a dense probe there would be gating a placeholder; the full sweep above
/// reports what the hardware does at their zero instead.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn lut_dense_near_zero_has_no_join_spike() {
    let sweeps = [
        // Signed output, `TANH_LE[512] = TANH_LO[0] = 0`. The notes'
        // primary QUIRK-2 candidate. tanh(x) ~ x here, so at 1/4096 the
        // expected codes span only +-16.
        Sweep {
            name: "tanh",
            table: LutTable::tanh,
            oracle: |x| x.tanh(),
            input_scale: 1.0 / 32768.0,
            output_scale: 1.0 / 4096.0,
            valid: (-128, 127),
            tolerance_lsb: 2.0,
        },
        // Signed output, `ERF_LE[512] = ERF_LO[0] = 0`. erf(x) ~ 1.128x
        // here, expected codes span +-18.
        Sweep {
            name: "erf",
            table: LutTable::erf,
            oracle: erf,
            input_scale: 1.0 / 32768.0,
            output_scale: 1.0 / 4096.0,
            valid: (-128, 127),
            tolerance_lsb: 3.0,
        },
        // The dedicated QUIRK-4 probe: `SQUARE_LE[510..512]` and
        // `SQUARE_LO[0..2]` are six `q = 0` entries and this shape drives
        // nothing else -- every code lands within `|index| <= 63`, i.e.
        // inside them. `x^2 <= 1.5e-5` so every correct output is the
        // byte 0, and *any* nonzero byte is the hazard.
        Sweep {
            name: "square",
            table: LutTable::square,
            oracle: |x| x * x,
            input_scale: 1.0 / 32768.0,
            output_scale: 1.0 / 128.0,
            valid: (-128, 127),
            tolerance_lsb: 1.0,
        },
        // Unsigned-output control: the notes predict sigmoid is
        // x~0-clean, so a spike here would mean the hazard is not the
        // signed-output-only effect they describe. sigmoid(x) ~ 0.5,
        // expected code ~32 throughout.
        Sweep {
            name: "sigmoid",
            table: LutTable::sigmoid,
            oracle: |x| 1.0 / (1.0 + (-x).exp()),
            input_scale: 1.0 / 32768.0,
            output_scale: 1.0 / 64.0,
            valid: (-128, 127),
            tolerance_lsb: 2.0,
        },
        // Second unsigned-output control, and the kind the notes found
        // QUIRK 4 with in the first place (in its deep tail, not here).
        // `x <= 0` only, per `EXP_LO`. exp(x) ~ 1, expected code ~64.
        Sweep {
            name: "exp",
            table: LutTable::exp,
            oracle: |x| x.exp(),
            input_scale: 1.0 / 32768.0,
            output_scale: 1.0 / 64.0,
            valid: (-128, 0),
            tolerance_lsb: 2.0,
        },
    ];

    run_sweeps(&sweeps, "dense near-zero sweep (QUIRK 2 / QUIRK 4)");
}
