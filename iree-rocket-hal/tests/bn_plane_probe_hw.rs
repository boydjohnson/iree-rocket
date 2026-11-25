//! SOFTMAX.md probe **S1**: the DPU's BN per-channel operand plane, fed by
//! NRDMA.
//!
//! This is the first time anything in this repo drives that plane. Every
//! builder here writes `RDMA_NRDMA_CFG` and `RDMA_BN_BASE_ADDR` as zero and
//! leaves `BN_CFG.bn_alu_src` / `BN_MUL_CFG.bn_mul_src` at 0, so the
//! operands have only ever come from config registers. SOFTMAX.md's whole
//! design stands on the plane working: in the width-mapped layout it argues
//! for, a softmax row maximum *is* a per-channel value at physical height
//! 1, and a per-channel plane is what broadcasts it back across the row
//! (and later applies the row's reciprocal). Everything downstream is
//! uninteresting if this fails, which is why it is probe number one.
//!
//! Cross-compile this test, copy the resulting binary to the RK3588 board,
//! and run the ignored tests there:
//!
//! ```text
//! CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!   cargo test --target aarch64-unknown-linux-gnu --release \
//!     --test bn_plane_probe_hw --no-run
//!
//! ./bn_plane_probe_hw-<hash> --ignored --nocapture --test-threads=1
//! ```
//!
//! `--test-threads=1` matters here beyond the usual single-device reason:
//! this probe deliberately enables a DMA fetch that ISSUES.md C8 showed can
//! poison the core for the *following* job, and
//! `a_following_job_still_runs_after_an_nrdma_fetch` only means something
//! if nothing else is submitting in between.
//!
//! ## What is being asked, in order
//!
//! Each question is a separate test, and each decides between *named
//! candidates* rather than asserting one guess -- a probe's job is to come
//! back with the answer whichever way it goes, so a failing run prints
//! which candidate did match.
//!
//! 1. **Is the plane read at all** (`bn_alu_plane_is_read`): does a nonzero
//!    plane change the output against a zeroed one.
//! 2. **What is the ALU operand's encoding**
//!    (`bn_alu_plane_is_ieee_fp32_per_channel`). The leading candidate is a
//!    flat IEEE `f32` per channel, because that is what the one confirmed
//!    plane in this tree does: the fp16 BS ALU operand "is a 32-bit float
//!    at the RK3588 BRDMA boundary", widened from the logical fp16 tensor
//!    by `pack_fp16_bias_to_rocket`. The alternative is the fixed-point
//!    form BS takes at int8 (an `i32` bias at a 14-bit shift).
//! 3. **Does it broadcast** (`bn_alu_plane_broadcasts_over_width_and_height`)
//!    -- the property softmax actually needs, checked at softmax's own
//!    height-1 shape as well as a square one.
//! 4. **Does it index past one atom**
//!    (`bn_alu_plane_indexes_every_programmed_channel`). `ew_round_hw.rs`
//!    found `EW_OP_VALUE_0` alone serving only channels 0-7 of a 16-wide
//!    padded atom while 8-15 silently read zero; the same shape of bug in a
//!    memory plane would be a layout that repeats per atom or per 8-channel
//!    block (which is what BS's int8 buffer really does -- planar within a
//!    64-byte block, `conv::write_bs_buffer`).
//! 5. **The MUL operand's encoding** (`bn_mul_plane_scales_per_channel`).
//!    `BN_MUL_CFG.bn_mul_operand` is 16 bits with a 6-bit shift beside it,
//!    so a fixed-point `i16` is the leading candidate here, unlike the ALU.
//! 6. **The stage order** (`bn_plane_add_then_mul_order`): `(x + a) * m` or
//!    `x * m + a`. No register description in this tree says, and
//!    `lut_bn_alu`'s own formula (`-zero_point * bn_mul_operand`) implies
//!    the operand is pre-scaled by the multiplier, i.e. the add lands
//!    *after*. SOFTMAX.md's two-pass form never engages both at once, so
//!    this one is for the record rather than the critical path.
//! 7. **The C8 hazard** (`a_following_job_still_runs_after_an_nrdma_fetch`).
//!
//! Every value used is exactly representable in fp16 and every oracle is
//! exact, so the tolerance is for the fp32 datapath's narrowing on the way
//! out, not for the arithmetic.

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr, time::Instant};

#[path = "support/dispatch.rs"]
mod dispatch;

use iree_rocket_hal::rocket::{
    bn_plane::{
        BnPlaneBuffers, BnPlaneOp, BnPlaneShape, PlaneElement, bn_mul_plane_bytes, bn_plane_bytes,
        build_bn_plane_regcmd, build_plane_gather_regcmd, write_bn_alu_plane_fp32,
        write_bn_mul_plane_fp16,
    },
    builders::{
        Bits, Register, RegisterMeta,
        dpu::{DpuBsOwCfg, DpuDataFormat, DpuSurfaceAdd},
    },
    device::{Buffer, close_bo, fini_bo, prep_bo, submit, submit_tasks},
    elementwise::{EwUnaryAlgo, EwUnaryBuffers, EwUnaryShape, build_unary_regcmd},
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const BUFFER_SIZE: usize = 16 * 1024;
const FEATURE_ATOM_BYTES: usize = 16;
const FP16_LANES_PER_ATOM: usize = 8;
/// fp16 round trips through an fp32 datapath; the oracle values are exact,
/// so anything above this is a real disagreement.
const TOLERANCE: f32 = 1e-3;

fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    if value == 0.0 {
        return sign;
    }
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let fraction = bits & 0x7f_ffff;
    assert!(
        (1..31).contains(&exponent),
        "{value} is outside the fp16 normal range"
    );
    assert_eq!(fraction & 0x1fff, 0, "{value} is not exact in fp16");
    sign | ((exponent as u16) << 10) | ((fraction >> 13) as u16)
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

/// Byte offset of logical channel `channel` at pixel `pixel` in an fp16
/// NC1HWC2 cube whose surface stride is `pixels` atoms -- the layout both
/// `build_bn_plane_regcmd` and `build_unary_regcmd` program
/// (`DST_SURF_STRIDE` and `SURFACE_ADD` count 16-byte atoms, so a surface
/// is `width * height` atoms and the channel count does not enter).
fn nc1hwc2_offset(channel: usize, pixel: usize, pixels: usize) -> usize {
    let surface = channel / FP16_LANES_PER_ATOM;
    let lane = channel % FP16_LANES_PER_ATOM;
    surface * pixels * FEATURE_ATOM_BYTES + pixel * FEATURE_ATOM_BYTES + lane * 2
}

/// The probe's input: distinct in both axes, so a plane that were really
/// per-pixel, or an output that silently reused one channel's value, could
/// not pass by coincidence. Every value is exact in fp16.
fn input_value(channel: usize, pixel: usize) -> f32 {
    1.0 + channel as f32 + pixel as f32 * 0.25
}

/// A per-channel ALU operand that is negative for some channels: softmax
/// subtracts a maximum, so the sign matters and a probe that only ever
/// added would not have checked it.
fn alu_value(channel: usize) -> f32 {
    0.5 * channel as f32 - 2.0
}

/// A per-channel multiplier, kept to exact powers of two and halves so the
/// fixed-point candidates encode without rounding.
fn mul_value(channel: usize) -> f32 {
    match channel % 4 {
        0 => 1.0,
        1 => 2.0,
        2 => 0.5,
        _ => -1.0,
    }
}

struct Case {
    width: u32,
    height: u32,
    channels: u32,
    op: BnPlaneOp,
    mul_shift: u32,
}

impl Case {
    fn pixels(&self) -> usize {
        (self.width * self.height) as usize
    }

    fn shape(&self) -> BnPlaneShape {
        BnPlaneShape {
            width: self.width,
            height: self.height,
            channels: self.channels,
            op: self.op,
            mul_shift: self.mul_shift,
        }
    }

    fn label(&self) -> String {
        format!(
            "{}x{} c{} {:?} shift{}",
            self.width, self.height, self.channels, self.op, self.mul_shift
        )
    }
}

/// Runs one BN-plane job. `plane` is the exact bytes `RDMA_BN_BASE_ADDR`
/// will point at. Returns the output as `[channel][pixel]`, or `None` when
/// the watchdog killed the job (see `support/dispatch.rs` -- `PREP_BO`
/// reports a killed job as success, so the wall clock is the only
/// discriminator).
fn run_case(case: &Case, plane: &[u8]) -> Option<Vec<Vec<f32>>> {
    let pixels = case.pixels();
    let shape = case.shape();
    let channels = case.channels as usize;
    assert!(
        plane.len() >= bn_plane_bytes(shape.programmed_channels()),
        "plane must cover every programmed channel, or NRDMA reads past it"
    );

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let buf_in = Buffer::new(fd, BUFFER_SIZE, &file);
        ptr::write_bytes(buf_in.host_ptr, 0, BUFFER_SIZE);
        for channel in 0..channels {
            for pixel in 0..pixels {
                let at = nc1hwc2_offset(channel, pixel, pixels);
                let word = f32_to_f16(input_value(channel, pixel));
                ptr::write_unaligned(buf_in.host_ptr.add(at) as *mut u16, word);
            }
        }

        let buf_plane = Buffer::new(fd, BUFFER_SIZE, &file);
        ptr::write_bytes(buf_plane.host_ptr, 0, BUFFER_SIZE);
        ptr::copy_nonoverlapping(plane.as_ptr(), buf_plane.host_ptr, plane.len());

        let buf_out = Buffer::new(fd, BUFFER_SIZE, &file);
        // A sentinel rather than zero: zero is a plausible wrong answer
        // here (it is what an unread plane plus a bypassed stage would
        // look like), so an unwritten output has to be distinguishable
        // from a computed one.
        ptr::write_bytes(buf_out.host_ptr, 0x5A, BUFFER_SIZE);

        let bufs = BnPlaneBuffers {
            input_addr: buf_in.dma_address,
            plane_addr: buf_plane.dma_address,
            output_addr: buf_out.dma_address,
        };
        let cmds = build_bn_plane_regcmd(&shape, &bufs);

        let cmd_bytes = cmds.len() * mem::size_of::<u64>();
        let buf_cmd = Buffer::new(fd, cmd_bytes.next_multiple_of(4096), &file);
        let cmd_slice = std::slice::from_raw_parts_mut(buf_cmd.host_ptr as *mut u64, cmds.len());
        for (i, c) in cmds.iter().enumerate() {
            cmd_slice[i] = c.0;
        }

        fini_bo(fd, buf_in.handle).ok();
        fini_bo(fd, buf_plane.handle).ok();
        fini_bo(fd, buf_out.handle).ok();
        fini_bo(fd, buf_cmd.handle).ok();

        let in_handles = [buf_cmd.handle, buf_in.handle, buf_plane.handle];
        let out_handles = [buf_out.handle];

        let started = Instant::now();
        submit(
            fd,
            buf_cmd.dma_address,
            cmds.len() as u32,
            &in_handles,
            &out_handles,
        )
        .expect("SUBMIT ioctl failed");
        prep_bo(fd, buf_out.handle, 2_000_000_000).unwrap_or_else(|e| {
            panic!(
                "BN-plane job ({}) did not complete within the timeout. This is the first \
                 hardware round for the NRDMA-fed BN plane, so a hang is a plausible outcome \
                 of a wrong register program rather than a surprise: {e}",
                case.label()
            )
        });
        let elapsed = started.elapsed();

        let out = (0..channels)
            .map(|channel| {
                (0..pixels)
                    .map(|pixel| {
                        let at = nc1hwc2_offset(channel, pixel, pixels);
                        f16_to_f32(ptr::read_unaligned(buf_out.host_ptr.add(at) as *const u16))
                    })
                    .collect()
            })
            .collect();

        close_bo(fd, buf_in.handle).ok();
        close_bo(fd, buf_plane.handle).ok();
        close_bo(fd, buf_out.handle).ok();
        close_bo(fd, buf_cmd.handle).ok();

        if dispatch::is_device_timeout(elapsed) {
            let strict = dispatch::report(&case.label(), elapsed);
            assert!(!strict, "killed dispatch under ROCKET_STRICT_DISPATCH");
            return None;
        }
        println!("  ran {}{}", case.label(), dispatch::note(elapsed));
        Some(out)
    }
}

/// A flat plane of `i32` values at `shift` fractional bits -- the form the
/// BS plane's operands take at int8 (`conv::BsEntry::bias` normalised by
/// the scales, with `BS_MUL_SHIFT_VALUE = 14`).
fn flat_i32_plane(programmed_channels: u32, values: &[f32], shift: u32) -> Vec<u8> {
    let mut plane = vec![0u8; bn_plane_bytes(programmed_channels)];
    for (channel, &value) in values.iter().enumerate() {
        let fixed = (value * (1u32 << shift) as f32).round() as i32;
        plane[channel * 4..channel * 4 + 4].copy_from_slice(&fixed.to_le_bytes());
    }
    plane
}

/// A flat plane of `i16` values at `shift` fractional bits, `stride` bytes
/// apart -- the shape `BN_MUL_CFG.bn_mul_operand`'s 16-bit width suggests.
/// `stride` is swept because nothing pins the slot size: BS's own int8
/// multiplier is an `i16` at a 2-byte pitch inside a 64-byte block of eight
/// channels, not a flat array.
fn flat_i16_plane(programmed_channels: u32, values: &[f32], shift: u32, stride: usize) -> Vec<u8> {
    let mut plane = vec![0u8; bn_plane_bytes(programmed_channels).max(values.len() * stride)];
    for (channel, &value) in values.iter().enumerate() {
        let fixed = (value * (1u32 << shift) as f32).round() as i16;
        let at = channel * stride;
        plane[at..at + 2].copy_from_slice(&fixed.to_le_bytes());
    }
    plane
}

/// The confirmed MUL form: flat IEEE fp16 at a 2-byte pitch.
fn fp16_plane(programmed_channels: u32, values: &[f32]) -> Vec<u8> {
    let mut plane = vec![0u8; bn_plane_bytes(programmed_channels)];
    write_bn_mul_plane_fp16(&mut plane, programmed_channels, values).expect("plane fits");
    plane
}

fn fp32_plane(programmed_channels: u32, values: &[f32]) -> Vec<u8> {
    let mut plane = vec![0u8; bn_plane_bytes(programmed_channels)];
    write_bn_alu_plane_fp32(&mut plane, programmed_channels, values).expect("plane fits");
    plane
}

/// Whether every element of `got` matches `oracle(channel, pixel)`, and
/// the first few that do not.
fn compare(got: &[Vec<f32>], oracle: &dyn Fn(usize, usize) -> f32) -> (bool, String) {
    let mut samples = Vec::new();
    let mut mismatches = 0usize;
    for (channel, row) in got.iter().enumerate() {
        for (pixel, &value) in row.iter().enumerate() {
            let want = oracle(channel, pixel);
            if (value - want).abs() > TOLERANCE {
                mismatches += 1;
                if samples.len() < 6 {
                    samples.push(format!("c{channel} p{pixel}: want {want} got {value}"));
                }
            }
        }
    }
    (
        mismatches == 0,
        if mismatches == 0 {
            String::new()
        } else {
            format!("{mismatches} mismatches: {}", samples.join("; "))
        },
    )
}

/// One named hypothesis: the exact plane bytes to load, and the oracle the
/// output must reproduce if that hypothesis is the right one.
type Candidate<'a> = (&'a str, Vec<u8>, &'a dyn Fn(usize, usize) -> f32);

/// Prints what the hardware actually returned, per channel, next to the
/// input and the plane value that was loaded for it. A candidate sweep
/// says only "none of these"; this says what the answer looks like, which
/// is what a probe is for when the sweep comes back empty.
fn dump(label: &str, got: &[Vec<f32>], plane_values: &[f32]) {
    println!("  {label}:");
    for (channel, row) in got.iter().enumerate() {
        let input = input_value(channel, 0);
        let out = row[0];
        let ratio = if input == 0.0 { f32::NAN } else { out / input };
        println!(
            "    c{channel}: plane {:>8}  in {:>8}  out {:>10}  out/in {:>10}  out-in {:>10}",
            plane_values.get(channel).copied().unwrap_or(f32::NAN),
            input,
            out,
            ratio,
            out - input
        );
    }
}

/// Runs each named candidate and reports which ones reproduce their own
/// oracle. The whole point of a probe is to come back with the answer, so
/// this prints every candidate's verdict before the caller asserts on it.
fn identify(case: &Case, candidates: &[Candidate<'_>]) -> Vec<String> {
    let mut matched = Vec::new();
    for (name, plane, oracle) in candidates {
        match run_case(case, plane) {
            None => println!("  {name}: NOT MEASURED (killed dispatch)"),
            Some(got) => {
                let (ok, detail) = compare(&got, *oracle);
                if ok {
                    println!("  {name}: MATCHES");
                    matched.push((*name).to_string());
                } else {
                    println!("  {name}: no -- {detail}");
                }
            }
        }
    }
    matched
}

/// Question 1. Nothing else in this file means anything if the plane is
/// inert: a zeroed plane and a nonzero one have to produce different
/// output, whatever the encoding turns out to be.
#[test]
#[ignore = "requires an RK3588 NPU at /dev/accel/accel0"]
fn bn_alu_plane_is_read() {
    let case = Case {
        width: 4,
        height: 4,
        channels: 8,
        op: BnPlaneOp::Add,
        mul_shift: 0,
    };
    let channels = case.channels as usize;
    let values: Vec<f32> = (0..channels).map(alu_value).collect();
    let programmed = case.shape().programmed_channels();

    let Some(zeroed) = run_case(&case, &vec![0u8; bn_plane_bytes(programmed)]) else {
        println!("zeroed-plane arm not measured; nothing to compare");
        return;
    };
    let Some(loaded) = run_case(&case, &fp32_plane(programmed, &values)) else {
        println!("loaded-plane arm not measured; nothing to compare");
        return;
    };

    // The zeroed arm is also the pass-through check: BN engaged with a zero
    // addend must be the identity, which is what makes a difference in the
    // loaded arm attributable to the plane rather than to engaging BN.
    let (passthrough, detail) = compare(&zeroed, &|c, p| input_value(c, p));
    assert!(
        passthrough,
        "BN engaged with an all-zero ALU plane is not the identity, so this probe cannot \
         attribute anything to the plane's contents. Either the plane is being read as \
         something other than zero, or engaging BN at all perturbs the datapath: {detail}"
    );

    assert!(
        zeroed != loaded,
        "a nonzero BN ALU plane produced byte-identical output to a zeroed one: NRDMA is not \
         reading it. SOFTMAX.md section 9 is the fallback -- an ERDMA broadcast operand \
         (EW_SURF_STRIDE/EW_SURF_NOTCH, zero in every builder here) or a materialised \
         broadcast tensor, which costs a full-size pass and probably puts softmax back under \
         P8's dispatch-tax line."
    );
}

/// Question 2. The leading candidate is a flat IEEE `f32` per channel,
/// mirroring the fp16 BS bias plane exactly.
#[test]
#[ignore = "requires an RK3588 NPU at /dev/accel/accel0"]
fn bn_alu_plane_is_ieee_fp32_per_channel() {
    let case = Case {
        width: 4,
        height: 4,
        channels: 8,
        op: BnPlaneOp::Add,
        mul_shift: 0,
    };
    let channels = case.channels as usize;
    let values: Vec<f32> = (0..channels).map(alu_value).collect();
    let programmed = case.shape().programmed_channels();

    let added = |c: usize, p: usize| input_value(c, p) + alu_value(c);
    let matched = identify(
        &case,
        &[
            (
                "flat IEEE f32 (BS fp16 bias's own form)",
                fp32_plane(programmed, &values),
                &added,
            ),
            (
                "flat i32, 14 fractional bits (BS int8's form)",
                flat_i32_plane(programmed, &values, 14),
                &added,
            ),
            (
                "flat i32, no fractional bits",
                flat_i32_plane(programmed, &values, 0),
                &added,
            ),
        ],
    );

    assert!(
        !matched.is_empty(),
        "no candidate ALU operand encoding reproduced `input + plane[channel]`. The plane is \
         read (see bn_alu_plane_is_read) but this file does not know as what -- record the \
         per-candidate dumps above in SOFTMAX.md section 8 before guessing further."
    );
    assert_eq!(
        matched,
        vec!["flat IEEE f32 (BS fp16 bias's own form)".to_string()],
        "the ALU operand encoding is not (only) the predicted flat IEEE f32; matched: {matched:?}. \
         That is a real finding either way -- update SOFTMAX.md section 7's plane-format note \
         and `write_bn_alu_plane_fp32`'s doc comment, which currently call f32 a hypothesis."
    );
}

/// Question 3, and the property softmax actually needs: one value per
/// channel, constant across every spatial position. Checked at softmax's
/// own height-1 shape and at a square one, because a plane indexed by
/// anything other than the channel would still pass a 1x1 cube.
#[test]
#[ignore = "requires an RK3588 NPU at /dev/accel/accel0"]
fn bn_alu_plane_broadcasts_over_width_and_height() {
    for (width, height) in [(16, 1), (4, 4), (1, 1)] {
        let case = Case {
            width,
            height,
            channels: 8,
            op: BnPlaneOp::Add,
            mul_shift: 0,
        };
        let channels = case.channels as usize;
        let values: Vec<f32> = (0..channels).map(alu_value).collect();
        let programmed = case.shape().programmed_channels();
        let Some(got) = run_case(&case, &fp32_plane(programmed, &values)) else {
            continue;
        };

        // Stated as "the same addend at every pixel" rather than just
        // re-checking the oracle: that is the claim SOFTMAX.md relies on,
        // and this way the failure message says so.
        for (channel, row) in got.iter().enumerate() {
            let addends: Vec<f32> = row
                .iter()
                .enumerate()
                .map(|(pixel, &value)| value - input_value(channel, pixel))
                .collect();
            let first = addends[0];
            for (pixel, &addend) in addends.iter().enumerate() {
                assert!(
                    (addend - first).abs() <= TOLERANCE,
                    "{}x{height} channel {channel}: the BN addend varies along the cube \
                     ({first} at pixel 0, {addend} at pixel {pixel}), so it is not a \
                     per-channel broadcast. SOFTMAX.md's max-subtraction needs it constant \
                     across the reduced axis.",
                    width
                );
            }
            assert!(
                (first - alu_value(channel)).abs() <= TOLERANCE,
                "{width}x{height} channel {channel}: constant addend {first}, wanted \
                 {}",
                alu_value(channel)
            );
        }
    }
}

/// Question 4. A plane that repeated per 8-channel atom or per 16-channel
/// programmed granule would pass every test above, which only use 8
/// channels.
#[test]
#[ignore = "requires an RK3588 NPU at /dev/accel/accel0"]
fn bn_alu_plane_indexes_every_programmed_channel() {
    let case = Case {
        width: 4,
        height: 1,
        channels: 32,
        op: BnPlaneOp::Add,
        mul_shift: 0,
    };
    let channels = case.channels as usize;
    let values: Vec<f32> = (0..channels).map(alu_value).collect();
    let programmed = case.shape().programmed_channels();
    let Some(got) = run_case(&case, &fp32_plane(programmed, &values)) else {
        return;
    };

    let (ok, detail) = compare(&got, &|c, p| input_value(c, p) + alu_value(c));
    if !ok {
        // Name the two layouts that would produce this, so the failure is
        // a diagnosis rather than a mismatch count.
        let (per_atom, _) = compare(&got, &|c, p| {
            input_value(c, p) + alu_value(c % FP16_LANES_PER_ATOM)
        });
        let (per_granule, _) = compare(&got, &|c, p| input_value(c, p) + alu_value(c % 16));
        panic!(
            "the ALU plane is not indexed by logical channel across 32 channels.\n  \
             repeats every 8 channels (one fp16 atom): {per_atom}\n  \
             repeats every 16 channels (one programmed granule): {per_granule}\n  \
             {detail}"
        );
    }
}

/// Question 5. `bn_mul_operand`'s 16-bit register width plus a 6-bit shift
/// beside it is what a fixed-point multiplier looks like, so unlike the
/// ALU this one does not have an obvious float candidate -- all four
/// combinations get run.
#[test]
#[ignore = "requires an RK3588 NPU at /dev/accel/accel0"]
fn bn_mul_plane_scales_per_channel() {
    let channels = 8usize;
    let values: Vec<f32> = (0..channels).map(mul_value).collect();
    let scaled = |c: usize, p: usize| input_value(c, p) * mul_value(c);

    let mut all_matched = Vec::new();
    for shift in [0u32, 14] {
        let case = Case {
            width: 4,
            height: 4,
            channels: channels as u32,
            op: BnPlaneOp::Mul,
            mul_shift: shift,
        };
        let programmed = case.shape().programmed_channels();
        let mut candidates: Vec<Candidate<'_>> = Vec::new();
        if shift == 0 {
            // The confirmed form, found by `bn_mul_plane_dump`: a 16-bit
            // IEEE fp16 per channel at a 2-byte pitch, no shift. Kept as a
            // *candidate* rather than a bare oracle so the two encodings
            // the register widths made plausible stay on the record next
            // to it.
            candidates.push((
                "flat IEEE fp16 at 2-byte pitch, shift 0",
                fp16_plane(programmed, &values),
                &scaled,
            ));
            candidates.push((
                "flat IEEE f32 at 4-byte pitch, shift 0",
                fp32_plane(programmed, &values),
                &scaled,
            ));
        } else {
            candidates.push((
                "flat i16 at 2-byte pitch, shift 14",
                flat_i16_plane(programmed, &values, 14, 2),
                &scaled,
            ));
            candidates.push((
                "flat i32, shift 14",
                flat_i32_plane(programmed, &values, 14),
                &scaled,
            ));
        }
        println!("mul_shift = {shift}:");
        all_matched.extend(identify(&case, &candidates));
    }

    assert_eq!(
        all_matched,
        vec!["flat IEEE fp16 at 2-byte pitch, shift 0".to_string()],
        "the MUL operand encoding is not (only) the fp16-at-2-bytes form the first hardware \
         round established; matched: {all_matched:?}. SOFTMAX.md section 4 applies the row \
         reciprocal through this stage, so this changes that -- update `bn_mul_plane_bytes`'s \
         doc comment, which states the format as measured rather than assumed."
    );
}

/// A MUL plane sized for the operand's real width has no room for a stray
/// read past it: at two bytes per channel the plane is *half* the size the
/// ALU's is, which is the kind of asymmetry that produces an
/// adjacent-allocation read when one helper is used for both.
#[test]
#[ignore = "requires an RK3588 NPU at /dev/accel/accel0"]
fn bn_mul_plane_reads_only_its_own_two_bytes_per_channel() {
    let channels = 8usize;
    let values: Vec<f32> = (0..channels).map(mul_value).collect();
    let case = Case {
        width: 4,
        height: 1,
        channels: channels as u32,
        op: BnPlaneOp::Mul,
        mul_shift: 0,
    };
    let programmed = case.shape().programmed_channels();

    // Exactly the bytes the operand needs, then a sentinel over the rest
    // of what an ALU-sized plane would have occupied. If the hardware read
    // four bytes per channel, or reached past `bn_mul_plane_bytes`, the
    // sentinel would show up as a wrong multiplier.
    let mut plane = vec![0xA5u8; bn_plane_bytes(programmed)];
    write_bn_mul_plane_fp16(&mut plane, programmed, &values).expect("plane fits");
    for byte in plane.iter_mut().skip(bn_mul_plane_bytes(programmed)) {
        *byte = 0xA5;
    }

    let Some(got) = run_case(&case, &plane) else {
        return;
    };
    let (ok, detail) = compare(&got, &|c, p| input_value(c, p) * mul_value(c));
    assert!(
        ok,
        "a MUL plane of exactly `bn_mul_plane_bytes` did not reproduce `input * plane[c]` \
         with a sentinel past it, so the fetch is wider than two bytes per channel or \
         reaches past the plane: {detail}"
    );
}

/// Question 6, recorded rather than gated: with **both** sub-stages
/// engaged, where do the two operands live relative to each other, and in
/// which order are they applied?
///
/// SOFTMAX.md's two-pass form never engages both at once -- the max
/// subtraction rides the `exp` task's ALU and the row reciprocal rides a
/// later task's MUL -- so nothing in the design waits on this. It is a
/// diagnostic, not an assertion: it fits a line through each channel's
/// output and prints the slope and intercept, which separates the two
/// questions. A slope of `m` with an intercept of `a` means `x * m + a`; a
/// slope of `m` with an intercept of `a * m` means `(x + a) * m`; and a
/// slope or intercept that is neither says the operand came from the wrong
/// bytes, which is a layout answer rather than an order one.
///
/// **First run, 2026-11-24, and the order is settled: `x * m + a`.** The
/// "MUL region after ALU region" arm fitted `0 * x + 1` on every channel.
/// The slope says that layout put the multiplier stream somewhere the
/// hardware did not read, so it multiplied by zero -- and the intercept
/// says the addend survived that *unscaled*, at exactly the loaded `a`.
/// Under `(x + a) * m` a zero multiplier annihilates the addend too and
/// the output would have been 0. So the add lands after the multiply,
/// which independently corroborates `activation.rs`'s `lut_bn_alu`: that
/// formula pre-scales the register-sourced ALU operand by
/// `bn_mul_operand` precisely because it has to survive a later multiply.
///
/// **The combined layout is open**, and the interleaved arm says why it
/// will not be guessed: its fit repeats with period 3 (slope/intercept
/// 1.875/1, 2/0, 0/2.004), which is the aliasing signature of a 6-byte
/// caller slot against two independent hardware strides -- `lcm(4, 6)` is
/// three channels. Reading the bytes back out, the ALU stream is at
/// `channel * 4` from the plane base, as `Add` alone already established,
/// while the MUL stream is at `2 + channel * 2` -- two bytes further on
/// than where `Mul` alone reads it. Two enabled streams are therefore not
/// simply both based at `BN_BASE_ADDR`, and one more probe would be needed
/// to say what they are based at. SOFTMAX.md section 9 carries it as open;
/// nothing in the design waits on it.
#[test]
#[ignore = "requires an RK3588 NPU at /dev/accel/accel0"]
fn bn_plane_add_then_mul_fit() {
    let channels = 8usize;
    let case = Case {
        width: 4,
        height: 1,
        channels: channels as u32,
        op: BnPlaneOp::AddThenMul,
        mul_shift: 0,
    };
    let programmed = case.shape().programmed_channels();
    let (alu, mul) = (1.0f32, 2.0f32);
    let alu_bytes = fp32_plane(programmed, &vec![alu; channels]);
    let mul_bytes = fp16_plane(programmed, &vec![mul; channels]);

    let mut appended = vec![0u8; bn_plane_bytes(programmed) * 2];
    appended[..alu_bytes.len()].copy_from_slice(&alu_bytes);
    appended[bn_plane_bytes(programmed)..bn_plane_bytes(programmed) + mul_bytes.len()]
        .copy_from_slice(&mul_bytes);

    let mut interleaved = vec![0u8; bn_plane_bytes(programmed) * 2];
    for channel in 0..channels {
        let at = channel * 6;
        interleaved[at..at + 4].copy_from_slice(&alu.to_bits().to_le_bytes());
        interleaved[at + 4..at + 6].copy_from_slice(&mul_bytes[channel * 2..channel * 2 + 2]);
    }

    println!(
        "loaded a = {alu}, m = {mul}; `x * m + a` predicts slope {mul} intercept {alu}, \
         `(x + a) * m` predicts slope {mul} intercept {}",
        alu * mul
    );
    for (label, plane) in [
        ("MUL region after ALU region", appended),
        ("interleaved 6-byte slots", interleaved),
    ] {
        let Some(got) = run_case(&case, &plane) else {
            println!("  {label}: NOT MEASURED");
            continue;
        };
        println!("  {label}:");
        for (channel, row) in got.iter().enumerate() {
            // The input is affine in the pixel index by construction
            // (`input_value` steps by 0.25), so two pixels determine the
            // line and the rest confirm it.
            let (x0, x1) = (input_value(channel, 0), input_value(channel, 1));
            let slope = (row[1] - row[0]) / (x1 - x0);
            let intercept = row[0] - slope * x0;
            let affine = row.iter().enumerate().all(|(pixel, &value)| {
                (value - (slope * input_value(channel, pixel) + intercept)).abs() <= TOLERANCE
            });
            println!(
                "    c{channel}: slope {slope:>10}  intercept {intercept:>10}  affine {affine}"
            );
        }
    }
}

/// ISSUES.md C8's shape, which is the reason this module derives
/// `nrdma_data_use` from the engaged stages instead of taking it from the
/// caller. C8 was BRDMA -- the sibling of the DMA this probe turns on --
/// left fetching with the BS plane bypassed, and the symptom was not a
/// wrong result in *that* job but the *following* wide fp16 job hanging
/// until the power domain cycled. So the check has to be a second job,
/// after the plane job, with nothing in between.
#[test]
#[ignore = "requires an RK3588 NPU at /dev/accel/accel0"]
fn a_following_job_still_runs_after_an_nrdma_fetch() {
    let case = Case {
        width: 4,
        height: 4,
        channels: 8,
        op: BnPlaneOp::Add,
        mul_shift: 0,
    };
    let channels = case.channels as usize;
    let values: Vec<f32> = (0..channels).map(alu_value).collect();
    let programmed = case.shape().programmed_channels();
    if run_case(&case, &fp32_plane(programmed, &values)).is_none() {
        println!("the NRDMA job itself was not measured; the follow-on check is meaningless");
        return;
    }

    // A plain `build_unary_regcmd` Abs job: no BN, no NRDMA, and already
    // hardware-confirmed by `ew_unary_hw.rs`, so a failure here is about
    // the state the previous job left the core in and nothing else.
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let buf_in = Buffer::new(fd, BUFFER_SIZE, &file);
        let words = std::slice::from_raw_parts_mut(buf_in.host_ptr as *mut u16, BUFFER_SIZE / 2);
        words.fill(f32_to_f16(-3.5));

        let buf_out = Buffer::new(fd, BUFFER_SIZE, &file);
        ptr::write_bytes(buf_out.host_ptr, 0x5A, BUFFER_SIZE);

        let cmds = build_unary_regcmd(
            &EwUnaryShape {
                width: 4,
                height: 4,
                channels: 1,
                algo: EwUnaryAlgo::Abs,
                operand: 0,
            },
            &EwUnaryBuffers {
                input_addr: buf_in.dma_address,
                output_addr: buf_out.dma_address,
            },
        );
        let cmd_bytes = cmds.len() * mem::size_of::<u64>();
        let buf_cmd = Buffer::new(fd, cmd_bytes.next_multiple_of(4096), &file);
        let cmd_slice = std::slice::from_raw_parts_mut(buf_cmd.host_ptr as *mut u64, cmds.len());
        for (i, c) in cmds.iter().enumerate() {
            cmd_slice[i] = c.0;
        }

        fini_bo(fd, buf_in.handle).ok();
        fini_bo(fd, buf_out.handle).ok();
        fini_bo(fd, buf_cmd.handle).ok();

        let started = Instant::now();
        submit(
            fd,
            buf_cmd.dma_address,
            cmds.len() as u32,
            &[buf_cmd.handle, buf_in.handle],
            &[buf_out.handle],
        )
        .expect("SUBMIT ioctl failed");
        prep_bo(fd, buf_out.handle, 2_000_000_000).unwrap_or_else(|e| {
            panic!(
                "the job after an NRDMA fetch did not complete: this is ISSUES.md C8's \
                 symptom exactly (the poisoning job returns fine, the next one hangs until \
                 the power domain cycles). `build_bn_plane_regcmd` would need the same \
                 treatment C8's fix gave `brdma_data_use`: {e}"
            )
        });
        let elapsed = started.elapsed();

        let got: Vec<f32> = (0..16)
            .map(|pixel| {
                f16_to_f32(ptr::read_unaligned(
                    buf_out.host_ptr.add(pixel * FEATURE_ATOM_BYTES) as *const u16,
                ))
            })
            .collect();

        close_bo(fd, buf_in.handle).ok();
        close_bo(fd, buf_out.handle).ok();
        close_bo(fd, buf_cmd.handle).ok();

        if dispatch::is_device_timeout(elapsed) {
            let strict = dispatch::report("Abs after an NRDMA fetch", elapsed);
            assert!(!strict, "killed dispatch under ROCKET_STRICT_DISPATCH");
            return;
        }
        for (pixel, &value) in got.iter().enumerate() {
            assert!(
                (value - 3.5).abs() <= TOLERANCE,
                "Abs(-3.5) came back {value} at pixel {pixel} in the job after an NRDMA \
                 fetch. ew_unary_hw.rs has this op confirmed standalone, so the plane job \
                 left the core in a bad state -- ISSUES.md C8, one DMA over."
            );
        }
    }
}

/// Diagnostic for `bn_mul_plane_scales_per_channel` when its sweep comes
/// back empty: prints the raw per-channel result for each MUL
/// configuration instead of a mismatch count, so the encoding can be read
/// off the `out/in` column rather than guessed at.
///
/// It also engages the ALU with a *zero* addend alongside the MUL, which
/// the sweep does not. `lut_bn_alu`'s formula for the register-sourced
/// case is `-zero_point * bn_mul_operand` -- the ALU operand pre-scaled by
/// the multiplier -- so the two sub-stages are coupled in at least one
/// direction, and "the MUL stage only reads its operand while the ALU
/// stage is also engaged" is a cheap hypothesis to rule in or out.
#[test]
#[ignore = "requires an RK3588 NPU at /dev/accel/accel0"]
fn bn_mul_plane_dump() {
    let channels = 8usize;
    let values: Vec<f32> = (0..channels).map(mul_value).collect();
    println!("oracle is out = in * plane, so the out/in column should be the plane column");

    for (op, shift, encoding) in [
        (BnPlaneOp::Mul, 0u32, "f32"),
        (BnPlaneOp::Mul, 14, "i16@14"),
        (BnPlaneOp::AddThenMul, 0, "f32"),
        (BnPlaneOp::AddThenMul, 14, "i16@14"),
    ] {
        let case = Case {
            width: 4,
            height: 1,
            channels: channels as u32,
            op,
            mul_shift: shift,
        };
        let programmed = case.shape().programmed_channels();
        // With AddThenMul the ALU operand has to be present and zero, so
        // the two planes go in back to back: ALU first, MUL after it.
        let plane = match (op, encoding) {
            (BnPlaneOp::Mul, "f32") => fp32_plane(programmed, &values),
            (BnPlaneOp::Mul, _) => flat_i16_plane(programmed, &values, 14, 2),
            (_, "f32") => {
                let mut both = vec![0u8; bn_plane_bytes(programmed) * 2];
                let zeros = fp32_plane(programmed, &vec![0.0f32; channels]);
                both[..zeros.len()].copy_from_slice(&zeros);
                let mul = fp32_plane(programmed, &values);
                both[zeros.len()..zeros.len() + mul.len()].copy_from_slice(&mul);
                both
            }
            (_, _) => {
                let mut both = vec![0u8; bn_plane_bytes(programmed) * 2];
                let mul = flat_i16_plane(programmed, &values, 14, 2);
                both[bn_plane_bytes(programmed)..bn_plane_bytes(programmed) + mul.len()]
                    .copy_from_slice(&mul);
                both
            }
        };
        match run_case(&case, &plane) {
            None => println!("  {op:?} shift{shift} {encoding}: NOT MEASURED"),
            Some(got) => dump(&format!("{op:?} shift{shift} {encoding}"), &got, &values),
        }
    }
}

// ===========================================================================
// SOFTMAX.md probe S2a: the bridge from a strided reduction output to a
// dense plane operand.
// ===========================================================================
//
// S1 established what the plane wants: a flat `f32` at `channel * 4` for the
// ALU, a flat `fp16` at `channel * 2` for the MUL. S2 established what the
// row maximum arrives as: a width-1 cube whose *surfaces* are four pixels
// apart, because the PPU's writer rounds both cube strides up to four
// pixels. So channel `c` of the maximum sits at `(c / 8) * 64 + (c % 8) * 2`
// and the ALU plane wants it at `c * 4`.
//
// `build_plane_gather_regcmd` is one DPU pass-through that closes both gaps:
// `RDMA_SURF_NOTCH` gives the read the four-pixel slack, and the write is a
// width-1 cube, which is dense because `DST_SURF_STRIDE` counts atoms.
//
// Two questions, deliberately separate so a failure localises:
//
// 1. `the_gather_reads_a_four_pixel_strided_cube` -- fp16 out, which S2
//    already showed is the easy half of the write. A failure here is the
//    notch.
// 2. `the_gather_writes_a_dense_f32_plane` -- the same read with `Fp32` out.
//    A failure here is the writer, and the fp16 arm above says so.

/// The strided source this bridge exists to read: the four-pixel surface
/// pitch a width-1 PPU output has.
const POOLED_SURFACE_PIXELS: u32 = 4;

fn pooled_offset(channel: usize) -> usize {
    (channel / FP16_LANES_PER_ATOM) * POOLED_SURFACE_PIXELS as usize * FEATURE_ATOM_BYTES
        + (channel % FP16_LANES_PER_ATOM) * 2
}

/// Writes a width-1 cube with the PPU's four-pixel surface pitch, runs the
/// gather over it, and returns the output buffer's raw bytes.
fn run_gather(channels: u32, element: PlaneElement) -> Option<Vec<u8>> {
    run_gather_with_size_e(channels, element, None)
}

/// `run_gather`, with the 4-byte writer's whole triple overridden from the
/// environment: `PROBE_MC_SURF_OUT`, `PROBE_SIZE_E`, `PROBE_SURF_ADD`.
///
/// One combination per process, which is how this repo measures anything
/// on the NPU after a failed job -- a hung job can leave the core sick, so
/// a 32-case in-process sweep is exactly the shape that produces results
/// nobody can reproduce.
fn writer_override() -> Option<(u32, u32, u32)> {
    let read = |name: &str| {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
    };
    match (
        read("PROBE_MC_SURF_OUT"),
        read("PROBE_SIZE_E"),
        read("PROBE_SURF_ADD"),
    ) {
        (Some(mc), Some(size_e), Some(surf_add)) => Some((mc, size_e, surf_add)),
        _ => None,
    }
}

/// `run_gather`, with `BS_OW_CFG.size_e_0..2` overridden.
///
/// The field exists in this harness because the first fp32 attempt hung at
/// 522 ms -- the watchdog -- and `conv.rs`'s `bs_ow_size_e` records exactly
/// that symptom for a wrong value on a *different* path: "at 3 or 7 writes
/// 1024 of 65536 bytes and takes ~525 ms per tile -- the watchdog killing
/// the job, with PREP_BO still returning success". So the value is swept
/// rather than reasoned about.
fn run_gather_with_size_e(
    channels: u32,
    element: PlaneElement,
    size_e: Option<u32>,
) -> Option<Vec<u8>> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let buf_in = Buffer::new(fd, BUFFER_SIZE, &file);
        // A sentinel in the slack pixels: if the notch is wrong the gather
        // reads these instead of the next surface's real value, and a
        // recognisable number is easier to diagnose than a zero.
        let words = std::slice::from_raw_parts_mut(buf_in.host_ptr as *mut u16, BUFFER_SIZE / 2);
        words.fill(f32_to_f16(-99.0));
        for channel in 0..channels as usize {
            ptr::write(
                buf_in.host_ptr.add(pooled_offset(channel)) as *mut u16,
                f32_to_f16(alu_value(channel)),
            );
        }

        let buf_out = Buffer::new(fd, BUFFER_SIZE, &file);
        ptr::write_bytes(buf_out.host_ptr, 0x5A, BUFFER_SIZE);

        let mut cmds = build_plane_gather_regcmd(
            channels,
            POOLED_SURFACE_PIXELS,
            element,
            buf_in.dma_address,
            buf_out.dma_address,
        );
        // `conv.rs`'s own lesson about this writer: "the three move
        // together or not at all ... sweeping surf_add alone and size_e
        // alone both read as 'no effect / no value helps'". So the
        // environment sets all three or none.
        let triple = writer_override();
        if let Some(size_e) = size_e.or(triple.map(|(_, size_e, _)| size_e)) {
            let mut patched = 0;
            for cmd in &mut cmds {
                if (cmd.0 >> 48) as u32 == DpuBsOwCfg::DOMAIN
                    && (cmd.0 as u32 & 0xffff) == DpuBsOwCfg::OFFSET
                {
                    *cmd = Register::<DpuBsOwCfg>::new()
                        .size_e_0(Bits::new(size_e))
                        .size_e_1(Bits::new(size_e))
                        .size_e_2(Bits::new(size_e))
                        .od_bypass(Bits::new(1))
                        .build();
                    patched += 1;
                }
            }
            assert_eq!(patched, 1, "BS_OW_CFG is written exactly once");
        }
        if let Some((mc_surf_out, _, surf_add)) = triple {
            let mut patched = 0;
            for cmd in &mut cmds {
                if (cmd.0 >> 48) as u32 == DpuDataFormat::DOMAIN
                    && (cmd.0 as u32 & 0xffff) == DpuDataFormat::OFFSET
                {
                    *cmd = Register::<DpuDataFormat>::new()
                        .in_precision(Bits::new(2))
                        .out_precision(Bits::new(5))
                        .proc_precision(Bits::new(2))
                        .mc_surf_out(Bits::new(mc_surf_out))
                        .build();
                    patched += 1;
                }
                if (cmd.0 >> 48) as u32 == DpuSurfaceAdd::DOMAIN
                    && (cmd.0 as u32 & 0xffff) == DpuSurfaceAdd::OFFSET
                {
                    *cmd = Register::<DpuSurfaceAdd>::new()
                        .surf_add(Bits::new(surf_add))
                        .build();
                    patched += 1;
                }
            }
            assert_eq!(patched, 2, "DATA_FORMAT and SURFACE_ADD once each");
        }
        let cmd_bytes = cmds.len() * mem::size_of::<u64>();
        let buf_cmd = Buffer::new(fd, cmd_bytes.next_multiple_of(4096), &file);
        let cmd_slice = std::slice::from_raw_parts_mut(buf_cmd.host_ptr as *mut u64, cmds.len());
        for (i, c) in cmds.iter().enumerate() {
            cmd_slice[i] = c.0;
        }

        fini_bo(fd, buf_in.handle).ok();
        fini_bo(fd, buf_out.handle).ok();
        fini_bo(fd, buf_cmd.handle).ok();

        let started = Instant::now();
        submit(
            fd,
            buf_cmd.dma_address,
            cmds.len() as u32,
            &[buf_cmd.handle, buf_in.handle],
            &[buf_out.handle],
        )
        .expect("SUBMIT ioctl failed");
        prep_bo(fd, buf_out.handle, 2_000_000_000).unwrap_or_else(|e| {
            panic!(
                "the plane gather ({channels} channels, {element:?}) did not complete. Both \
                 RDMA_SURF_NOTCH and the fp32 writer are first-use here, so a hang is a \
                 plausible outcome of a wrong program: {e}"
            )
        });
        let elapsed = started.elapsed();

        let raw = std::slice::from_raw_parts(buf_out.host_ptr, 512).to_vec();
        close_bo(fd, buf_in.handle).ok();
        close_bo(fd, buf_out.handle).ok();
        close_bo(fd, buf_cmd.handle).ok();

        if dispatch::is_device_timeout(elapsed) {
            let strict = dispatch::report(&format!("gather {channels}ch {element:?}"), elapsed);
            assert!(!strict, "killed dispatch under ROCKET_STRICT_DISPATCH");
            return None;
        }
        Some(raw)
    }
}

/// Question 1: does `RDMA_SURF_NOTCH` give the read its four-pixel slack?
///
/// fp16 out, so the write is the arrangement S2 already measured and any
/// failure is the notch. 32 channels is four source surfaces, so a notch
/// that were ignored would read channel 8 from the slack of surface 0 --
/// the -99.0 sentinel -- rather than from surface 1.
#[test]
#[ignore = "requires an RK3588 NPU at /dev/accel/accel0"]
fn the_gather_reads_a_four_pixel_strided_cube() {
    let channels = 32u32;
    let Some(raw) = run_gather(channels, PlaneElement::Fp16) else {
        return;
    };
    for channel in 0..channels as usize {
        let at = channel * 2;
        let got = f16_to_f32(u16::from_le_bytes([raw[at], raw[at + 1]]));
        let want = alu_value(channel);
        assert!(
            (got - want).abs() <= TOLERANCE,
            "channel {channel} (source surface {}) came back {got}, wanted {want}. -99 means \
             RDMA_SURF_NOTCH was ignored and the read ran straight through the slack pixels; \
             anything else means the write, not the notch.",
            channel / FP16_LANES_PER_ATOM
        );
    }
    println!("  RDMA_SURF_NOTCH reads a 4-pixel-pitched width-1 cube at {channels} channels");
}

/// One `(mc_surf_out, size_e, surf_add)` combination for the fp32 writer,
/// taken from the environment and reported rather than asserted.
///
/// The first fp32 attempt -- `mc_surf_out 0`, `size_e 3`, `surf_add 1`, the
/// reading of `Precision::Fp16Accumulator`'s recipe that seemed to follow
/// from `conv.rs` -- hung at the watchdog. An in-process sweep of `size_e`
/// alone then showed 0 and 1 running and writing channel 0 *correctly* in
/// `f32` at `channel * 4`, with 28 of 32 channels wrong: exactly one atom's
/// worth right, which points at the surface stride rather than the element
/// type.
///
/// `conv.rs` says why sweeping one field reads as "nothing helps": the
/// 4-byte writer's three fields are one configuration. It records two
/// known-good triples, both for int32 -- `(0, 7, dst_surf_stride * 8)` from
/// `rocket-userspace`'s matmul, and `(1, 1, 16)`, this crate's own
/// accumulator mode, "a different writer, not a variant of the same one".
/// For fp32 the multiplier should be smaller: the int4 note calls 8 the
/// integer quirk, "not the float path's natural 1 and 2x".
#[test]
#[ignore = "requires an RK3588 NPU at /dev/accel/accel0"]
fn the_fp32_writer_triple_probe() {
    let Some((mc_surf_out, size_e, surf_add)) = writer_override() else {
        println!(
            "set PROBE_MC_SURF_OUT, PROBE_SIZE_E and PROBE_SURF_ADD; one combination per \
             process, since a hung job can leave the core sick"
        );
        return;
    };
    let channels = 32u32;
    print!("  mc_surf_out {mc_surf_out} size_e {size_e} surf_add {surf_add}: ");
    match run_gather_with_size_e(channels, PlaneElement::Fp32, None) {
        None => println!("HUNG (watchdog)"),
        Some(raw) => {
            let wrong: Vec<usize> = (0..channels as usize)
                .filter(|&channel| {
                    let at = channel * 4;
                    let got = f32::from_bits(u32::from_le_bytes([
                        raw[at],
                        raw[at + 1],
                        raw[at + 2],
                        raw[at + 3],
                    ]));
                    (got - alu_value(channel)).abs() > TOLERANCE
                })
                .collect();
            if wrong.is_empty() {
                println!("MATCHES a dense f32 plane on all {channels} channels");
            } else {
                println!(
                    "ran, {}/{channels} wrong, first wrong c{}",
                    wrong.len(),
                    wrong[0]
                );
            }
        }
    }
}

/// Question 2: does the `Fp16Accumulator` writer recipe produce a dense
/// `f32` plane -- the thing the BN ALU plane actually consumes?
#[test]
#[ignore = "requires an RK3588 NPU at /dev/accel/accel0"]
fn the_gather_writes_a_dense_f32_plane() {
    let channels = 32u32;
    let Some(raw) = run_gather(channels, PlaneElement::Fp32) else {
        return;
    };
    for channel in 0..channels as usize {
        let at = channel * 4;
        let got = f32::from_bits(u32::from_le_bytes([
            raw[at],
            raw[at + 1],
            raw[at + 2],
            raw[at + 3],
        ]));
        let want = alu_value(channel);
        assert!(
            (got - want).abs() <= TOLERANCE,
            "channel {channel} of the f32 plane came back {got} ({:#010x}), wanted {want}. \
             If the fp16 arm passed, this is the writer: out_precision 5, size_e 3 and a \
             cleared fp32tofp16_en are `Precision::Fp16Accumulator`'s recipe, and 4 channels \
             to a 16-byte atom is the geometry it implies at width 1.",
            u32::from_le_bytes([raw[at], raw[at + 1], raw[at + 2], raw[at + 3]])
        );
    }
    println!("  the gather writes a dense f32 plane at {channels} channels");
}

/// The bridge end to end: gather a strided maximum into an `f32` plane,
/// then have the BN ALU stage consume *that buffer* -- no host write
/// between them. This is SOFTMAX.md section 7's pass-2 edge, and it is the
/// claim S2a exists to support rather than either half on its own.
#[test]
#[ignore = "requires an RK3588 NPU at /dev/accel/accel0"]
fn a_gathered_plane_feeds_the_bn_alu_stage() {
    let channels = 8u32;
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        // The "pooled maximum": a strided width-1 cube, negated, because
        // what pass 2 adds is `-m`.
        let pooled = Buffer::new(fd, BUFFER_SIZE, &file);
        let words = std::slice::from_raw_parts_mut(pooled.host_ptr as *mut u16, BUFFER_SIZE / 2);
        words.fill(f32_to_f16(-99.0));
        for channel in 0..channels as usize {
            ptr::write(
                pooled.host_ptr.add(pooled_offset(channel)) as *mut u16,
                f32_to_f16(-alu_value(channel)),
            );
        }

        // The plane the gather writes and the BN stage reads. Nothing on
        // the host touches it in between.
        let plane = Buffer::new(fd, BUFFER_SIZE, &file);
        ptr::write_bytes(plane.host_ptr, 0, plane.size);

        let case = Case {
            width: 4,
            height: 4,
            channels,
            op: BnPlaneOp::Add,
            mul_shift: 0,
        };
        let pixels = case.pixels();
        let buf_in = Buffer::new(fd, BUFFER_SIZE, &file);
        ptr::write_bytes(buf_in.host_ptr, 0, BUFFER_SIZE);
        for channel in 0..channels as usize {
            for pixel in 0..pixels {
                ptr::write(
                    buf_in.host_ptr.add(nc1hwc2_offset(channel, pixel, pixels)) as *mut u16,
                    f32_to_f16(input_value(channel, pixel)),
                );
            }
        }
        let buf_out = Buffer::new(fd, BUFFER_SIZE, &file);
        ptr::write_bytes(buf_out.host_ptr, 0x5A, BUFFER_SIZE);

        let gather = build_plane_gather_regcmd(
            channels,
            POOLED_SURFACE_PIXELS,
            PlaneElement::Fp32,
            pooled.dma_address,
            plane.dma_address,
        );
        let consume = build_bn_plane_regcmd(
            &case.shape(),
            &BnPlaneBuffers {
                input_addr: buf_in.dma_address,
                plane_addr: plane.dma_address,
                output_addr: buf_out.dma_address,
            },
        );

        let mut cmd_bufs = Vec::new();
        for cmds in [&gather, &consume] {
            let buffer = Buffer::new(
                fd,
                (cmds.len() * mem::size_of::<u64>()).next_multiple_of(4096),
                &file,
            );
            let slots = std::slice::from_raw_parts_mut(buffer.host_ptr as *mut u64, cmds.len());
            for (slot, cmd) in slots.iter_mut().zip(cmds.iter()) {
                *slot = cmd.0;
            }
            cmd_bufs.push((buffer, cmds.len() as u32));
        }

        for handle in [pooled.handle, plane.handle, buf_in.handle, buf_out.handle] {
            fini_bo(fd, handle).ok();
        }
        for (buffer, _) in &cmd_bufs {
            fini_bo(fd, buffer.handle).ok();
        }

        // Both tasks in one job: the plane buffer is written by task 1 and
        // read by task 2, and the kernel dispatches task 2 only after task
        // 1's completion IRQ, so nothing waits on the host.
        let tasks: Vec<(u32, u32)> = cmd_bufs
            .iter()
            .map(|(buffer, count)| (buffer.dma_address, *count))
            .collect();
        let in_handles = [
            cmd_bufs[0].0.handle,
            cmd_bufs[1].0.handle,
            pooled.handle,
            buf_in.handle,
        ];
        let out_handles = [buf_out.handle];

        let started = Instant::now();
        submit_tasks(fd, &tasks, &in_handles, &out_handles).expect("SUBMIT ioctl failed");
        prep_bo(fd, buf_out.handle, 2_000_000_000)
            .unwrap_or_else(|e| panic!("the gather-then-consume job did not complete: {e}"));
        let elapsed = started.elapsed();

        let out: Vec<f32> = (0..channels as usize)
            .flat_map(|channel| {
                (0..pixels).map(move |pixel| {
                    f16_to_f32(ptr::read_unaligned(
                        buf_out.host_ptr.add(nc1hwc2_offset(channel, pixel, pixels)) as *const u16,
                    ))
                })
            })
            .collect();

        for handle in [pooled.handle, plane.handle, buf_in.handle, buf_out.handle] {
            close_bo(fd, handle).ok();
        }
        for (buffer, _) in &cmd_bufs {
            close_bo(fd, buffer.handle).ok();
        }

        if dispatch::is_device_timeout(elapsed) {
            let strict = dispatch::report("gather then consume", elapsed);
            assert!(!strict, "killed dispatch under ROCKET_STRICT_DISPATCH");
            return;
        }

        for channel in 0..channels as usize {
            for pixel in 0..pixels {
                let got = out[channel * pixels + pixel];
                let want = input_value(channel, pixel) - alu_value(channel);
                assert!(
                    (got - want).abs() <= TOLERANCE,
                    "channel {channel} pixel {pixel}: {got}, wanted {want}. Both halves pass \
                     on their own if the two tests above are green, so a failure here is the \
                     edge between them -- the plane the gather wrote is not what the BN stage \
                     read."
                );
            }
        }
        println!("  a gathered f32 plane feeds the BN ALU stage in one job, no host write");
    }
}
