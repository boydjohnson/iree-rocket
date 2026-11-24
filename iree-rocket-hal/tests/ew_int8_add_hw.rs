//! Hardware gate for the **int8** two-tensor EW add,
//! `out = clamp(round(relu?(sa*(a - za) + sb*(b - zb)) / so) + zo)` -- a
//! quantized residual block's tail.
//!
//! `build_add_regcmd`'s int8 branch used to be capture-inferred and was
//! never run. Measured here on 2026-11-24, that inference saturated nearly
//! every element: the sum lives in a common unit with 2^14 of headroom
//! (see [`EwInt8Quant`]), not in the primary operand's units. Fixing only
//! the 2^14 left the second operand's gain overflowing its 16-bit mantissa
//! once `sb > 2 * sa`, so both gains are taken relative to the larger of the
//! two scales -- the primary's through the BS multiply, the second's
//! through EW_CVT.
//!
//! Every operand is distinct per element and the geometries span several
//! 16-channel surfaces, so neither an ERDMA that re-reads one atom nor a
//! surface-stride error can pass. The criterion is strict: every element
//! exact against the real-valued formula, except an off-by-one where that
//! value sits on a rounding tie. Encoded gains carry ~2^-15 relative error,
//! which moves a value that is exactly `x.5` in real arithmetic to one side
//! of it; nothing else may differ.
//!
//! [`EwInt8Quant`]: iree_rocket_hal::rocket::elementwise::EwInt8Quant

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    device::{Buffer, close_bo, fini_bo, prep_bo, submit},
    elementwise::{
        EwAddBuffers, EwAddShape, EwBinaryOp, EwInt8Quant, EwPrecision, build_add_regcmd_with_relu,
    },
};

const DEVICE_PATH: &str = "/dev/accel/accel0";

#[derive(Clone, Copy, Debug)]
struct Geometry {
    width: u32,
    height: u32,
    channels: u32,
}

impl Geometry {
    fn elements(self) -> usize {
        (self.width * self.height * self.channels) as usize
    }

    /// int8 NC1HWC2: surfaces of 16 channels.
    fn cube_index(self, channel: u32, y: u32, x: u32) -> usize {
        let surface = channel / 16;
        let c2 = channel % 16;
        ((surface * self.width * self.height + y * self.width + x) * 16 + c2) as usize
    }

    fn cube_bytes(self) -> usize {
        (self.channels.next_multiple_of(16) * self.width * self.height) as usize
    }

    fn logical(self, i: usize) -> (u32, u32, u32) {
        let i = i as u32;
        (
            i / (self.width * self.height),
            (i / self.width) % self.height,
            i % self.width,
        )
    }
}

fn exact(q: &EwInt8Quant, relu: bool, a: i8, b: i8) -> f64 {
    let mut y = q.a_scale * f64::from(i32::from(a) - q.a_zero_point)
        + q.b_scale * f64::from(i32::from(b) - q.b_zero_point);
    if relu && y < 0.0 {
        y = 0.0;
    }
    y / q.output_scale + f64::from(q.output_zero_point)
}

/// Deterministic, per-element-distinct int8 operands.
fn operand(seed: u64, i: usize) -> i8 {
    let mut x = seed ^ (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    (x & 0xff) as u8 as i8
}

/// Runs one int8 add task and returns `(a, b, out)` per logical element.
fn run(geometry: Geometry, q: EwInt8Quant, relu: bool) -> Vec<(i8, i8, i8)> {
    let elements = geometry.elements();
    let bytes = geometry.cube_bytes().next_multiple_of(4096);
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open NPU device");
    let fd = file.as_raw_fd();
    unsafe {
        let buf_a = Buffer::new(fd, bytes, &file);
        let buf_b = Buffer::new(fd, bytes, &file);
        let buf_out = Buffer::new(fd, bytes, &file);
        ptr::write_bytes(buf_a.host_ptr, 0, bytes);
        ptr::write_bytes(buf_b.host_ptr, 0, bytes);
        ptr::write_bytes(buf_out.host_ptr, 0x5a, bytes);
        let a_cube = std::slice::from_raw_parts_mut(buf_a.host_ptr as *mut i8, bytes);
        let b_cube = std::slice::from_raw_parts_mut(buf_b.host_ptr as *mut i8, bytes);
        for i in 0..elements {
            let (c, y, x) = geometry.logical(i);
            a_cube[geometry.cube_index(c, y, x)] = operand(0x1234, i);
            b_cube[geometry.cube_index(c, y, x)] = operand(0xabcdef, i);
        }

        let cmds = build_add_regcmd_with_relu(
            &EwAddShape {
                width: geometry.width,
                height: geometry.height,
                channels: geometry.channels,
                precision: EwPrecision::Int8,
                op: EwBinaryOp::Add,
                int8: Some(q),
            },
            &EwAddBuffers {
                intermediate_addr: buf_a.dma_address,
                w_addr: buf_b.dma_address,
                output_addr: buf_out.dma_address,
            },
            relu,
        );

        let cmd_bytes = cmds.len() * mem::size_of::<u64>();
        let buf_cmd = Buffer::new(fd, cmd_bytes.next_multiple_of(4096), &file);
        let cmd_slice = std::slice::from_raw_parts_mut(buf_cmd.host_ptr as *mut u64, cmds.len());
        for (i, c) in cmds.iter().enumerate() {
            cmd_slice[i] = c.0;
        }
        for handle in [buf_a.handle, buf_b.handle, buf_out.handle, buf_cmd.handle] {
            fini_bo(fd, handle).ok();
        }
        submit(
            fd,
            buf_cmd.dma_address,
            cmds.len() as u32,
            &[buf_cmd.handle, buf_a.handle, buf_b.handle],
            &[buf_out.handle],
        )
        .expect("SUBMIT ioctl failed");
        prep_bo(fd, buf_out.handle, 2_000_000_000)
            .unwrap_or_else(|e| panic!("int8 EW add at {geometry:?} timed out: {e}"));

        let out_cube = std::slice::from_raw_parts(buf_out.host_ptr as *const i8, bytes);
        let result = (0..elements)
            .map(|i| {
                let (c, y, x) = geometry.logical(i);
                let at = geometry.cube_index(c, y, x);
                (a_cube[at], b_cube[at], out_cube[at])
            })
            .collect();
        for handle in [buf_a.handle, buf_b.handle, buf_out.handle, buf_cmd.handle] {
            close_bo(fd, handle).ok();
        }
        result
    }
}

/// `(exact, off-by-one at a tie, wrong)` against the real-valued formula;
/// `wrong` keeps the first few as text.
fn score(q: &EwInt8Quant, relu: bool, data: &[(i8, i8, i8)]) -> (usize, usize, usize, Vec<String>) {
    let (mut exact_count, mut ties, mut wrong) = (0, 0, 0);
    let mut samples = Vec::new();
    for &(a, b, y) in data {
        let e = exact(q, relu, a, b);
        let want = e.round().clamp(-128.0, 127.0) as i32;
        let d = (i32::from(y) - want).abs();
        let at_tie = (e - e.floor() - 0.5).abs() < 0.01;
        if d == 0 {
            exact_count += 1;
        } else if d == 1 && at_tie {
            ties += 1;
        } else {
            wrong += 1;
            if samples.len() < 8 {
                samples.push(format!("a={a} b={b} got={y} exact={e:.4}"));
            }
        }
    }
    (exact_count, ties, wrong, samples)
}

fn quant(za: i32, zb: i32, zo: i32, sa: f64, sb: f64, so: f64) -> EwInt8Quant {
    EwInt8Quant {
        a_zero_point: za,
        b_zero_point: zb,
        output_zero_point: zo,
        a_scale: sa,
        b_scale: sb,
        output_scale: so,
    }
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- int8 EW add, the quantized residual tail"]
fn int8_ew_add_matches_formula() {
    let small = Geometry {
        width: 8,
        height: 8,
        channels: 48,
    };
    // Scale ratios 0.08..12 both ways round, distinct zero points, a
    // nonzero output zero point under ReLU (where the floor is visible),
    // and ResNet50 int8's own residual geometries.
    let cases = [
        (small, quant(3, -128, -128, 0.035, 0.042, 0.07), false),
        (small, quant(3, -128, -128, 0.035, 0.042, 0.07), true),
        (small, quant(10, -20, 5, 0.02, 0.09, 0.05), false),
        (small, quant(10, -20, 5, 0.02, 0.09, 0.05), true),
        (small, quant(10, -20, 5, 0.09, 0.02, 0.05), true),
        (small, quant(40, -128, -128, 0.17, 0.035, 0.04), true),
        (small, quant(-5, 7, -128, 0.011, 0.13, 0.06), true),
        (small, quant(0, 0, 0, 0.05, 0.05, 0.05), false),
        (
            Geometry {
                width: 56,
                height: 56,
                channels: 256,
            },
            quant(3, -128, -128, 0.035, 0.044, 0.042),
            true,
        ),
        (
            Geometry {
                width: 14,
                height: 14,
                channels: 1024,
            },
            quant(-12, -128, -128, 0.08, 0.05, 0.07),
            true,
        ),
        (
            Geometry {
                width: 7,
                height: 7,
                channels: 2048,
            },
            quant(25, -128, -128, 0.12, 0.09, 0.11),
            true,
        ),
    ];
    let mut failures = Vec::new();
    for (geometry, q, relu) in cases {
        let data = run(geometry, q, relu);
        let (exact_count, ties, wrong, samples) = score(&q, relu, &data);
        println!(
            "{geometry:?} {q:?} relu={relu}: exact {exact_count}, tie off-by-one {ties}, wrong {wrong}"
        );
        for line in &samples {
            println!("    {line}");
        }
        if wrong > 0 {
            failures.push(format!("{geometry:?} relu={relu}: {wrong} wrong"));
        }
    }
    assert!(failures.is_empty(), "int8 EW add mismatches: {failures:?}");
}

/// The encoding refuses what the registers cannot hold rather than
/// programming a wrong gain.
#[test]
fn int8_ew_add_encoding_bounds() {
    let ok = quant(0, 0, 0, 0.035, 0.042, 0.07).encoding().unwrap();
    // The second operand carries the larger scale: its gain is the full
    // 2^14 and the primary's is below it.
    assert_eq!(ok.b_gain.scale >> ok.b_gain.shift, 16384);
    assert_eq!(ok.a_gain, (0.035_f64 / 0.042 * 16384.0).round() as u32);
    // The primary's gain has no shift: a primary 2^15 smaller rounds to 0.
    assert!(quant(0, 0, 0, 1e-6, 1.0, 1.0).encoding().is_err());
    // The second's gain does: the same ratio the other way round encodes.
    assert!(quant(0, 0, 0, 1.0, 1e-6, 1.0).encoding().is_ok());
    assert!(quant(0, 0, 0, 0.0, 1.0, 1.0).encoding().is_err());
    assert!(quant(0, 200, 0, 1.0, 1.0, 1.0).encoding().is_err());
}
