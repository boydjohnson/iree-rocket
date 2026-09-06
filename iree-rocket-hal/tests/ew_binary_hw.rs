//! Hardware-in-the-loop tests for the **standalone two-tensor** EW task --
//! `build_add_regcmd` driven with no producing conv at all, over
//! non-uniform data, for every [`EwBinaryOp`].
//!
//! This is ROADMAP.md's `mulf` investigation. That document records
//! `arith.mulf` as blocked by a "known, unresolved hardware failure": a
//! prior `build_square_regcmd` used DPU MUL mode with ERDMA self-aliased
//! to the primary input and produced all-zero output. Its own note says
//! the failing configuration was specifically the *self-aliased* one and
//! that a genuine two-tensor MUL through ERDMA's ordinary path had never
//! been tried. This file tries it.
//!
//! Two things this file exists to cover that no other hardware test does:
//!
//! 1. **Non-uniform operands.** `conv_with_add_hw.rs` fills both tensors
//!    with a single repeated value and reads pixel `[0]`, so it cannot
//!    tell a correctly addressed ERDMA fetch from one that reads the same
//!    atom for every position. The reference emitter in
//!    `../rocket-userspace` reports exactly that failure mode for its own
//!    flying-main elementwise multiply (`gen_ew_mul_fp16`: "leaving
//!    SURF_NOTCH 0 makes the ERDMA never advance, so B is effectively
//!    unread"), and this builder leaves `SURF_NOTCH`/`EW_SURF_NOTCH` at
//!    zero. Every tensor here is distinct per element and every element is
//!    checked.
//! 2. **No conv in front.** `build_add_regcmd`'s doc comment says it is
//!    "usable on its own", but every hardware run so far has been the
//!    two-task `build_conv_then_add_regcmd` chain. `../rocket-userspace`
//!    claims a conv/CACC main feed is *required* for the EW unit to read
//!    its second operand at all -- if that holds here, every test in this
//!    file reads back the primary operand alone.
//!
//! The default geometry is deliberately 16 real channels, i.e. **two
//! surfaces**, so a surface-stride or notch error shows up as a wrong
//! upper half rather than passing by accident on a single-surface cube;
//! `standalone_ew_mul_geometry_ladder` widens that to shapes whose surface
//! stride is larger, unaligned, and past the builder's own `max(12)`
//! floor.
//!
//! Cross-compile, copy to `planck`, run the ignored tests there:
//!
//! ```text
//! CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!   cargo test -p iree-rocket-hal --release \
//!     --target aarch64-unknown-linux-gnu --test ew_binary_hw --no-run
//!
//! ./ew_binary_hw-<hash> --ignored --nocapture --test-threads=1
//! ```

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    device::{Buffer, close_bo, fini_bo, prep_bo, submit},
    elementwise::{EwAddBuffers, EwAddShape, EwBinaryOp, EwPrecision, build_add_regcmd},
};

const DEVICE_PATH: &str = "/dev/accel/accel0";

/// A host-side reference for one binary op, applied element by element.
type Oracle = fn(f32, f32) -> f32;

/// One cube shape to run the op over. The default
/// ([`Geometry::TWO_SURFACES`]) is the smallest shape that still has to
/// advance the ERDMA fetch across a surface boundary; the ladder tests
/// below add larger and deliberately unaligned ones.
#[derive(Clone, Copy, Debug)]
struct Geometry {
    width: u32,
    height: u32,
    channels: u32,
}

impl Geometry {
    /// Two full fp16 surfaces (`C2 = 8` channels each), so the ERDMA
    /// operand fetch has to cross a surface boundary to be right.
    const TWO_SURFACES: Geometry = Geometry {
        width: 4,
        height: 4,
        channels: 16,
    };

    fn elements(self) -> usize {
        (self.width * self.height * self.channels) as usize
    }

    /// Position of one real element in the NC1HWC2 cube this task reads
    /// and writes, in fp16 elements: surfaces of `C2 = 8` channels, each
    /// surface `width * height` positions of 8 contiguous channels.
    fn cube_index(self, channel: u32, y: u32, x: u32) -> usize {
        let surface = channel / 8;
        let c2 = channel % 8;
        ((surface * self.width * self.height + y * self.width + x) * 8 + c2) as usize
    }

    /// Logical (channel, y, x) of the `i`th element, in a fixed order the
    /// operand generators below use to make every element distinct.
    fn logical(self, i: usize) -> (u32, u32, u32) {
        let i = i as u32;
        (
            i / (self.width * self.height),
            (i / self.width) % self.height,
            i % self.width,
        )
    }
}

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
        "{value} is out of fp16 normal range"
    );
    assert_eq!(fraction & 0x1fff, 0, "{value} is not exact in fp16");
    sign | ((exponent as u16) << 10) | ((fraction >> 13) as u16)
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) & 0x8000) << 16;
    let exponent = ((bits >> 10) & 0x1f) as u32;
    let fraction = ((bits as u32) & 0x3ff) << 13;
    if exponent == 0 {
        assert_eq!(fraction, 0, "subnormal fp16 output {bits:#x} not expected");
        return f32::from_bits(sign);
    }
    assert_ne!(exponent, 31, "inf/NaN fp16 output {bits:#x} not expected");
    f32::from_bits(sign | ((exponent + 127 - 15) << 23) | fraction)
}

/// The primary operand: every value a multiple of `0.25` in `[-1.5, 1.5]`,
/// so it is exact in fp16 and so is every sum, difference and product with
/// [`operand_b`] below.
fn operand_a(i: usize) -> f32 {
    ((i as i32 * 7).rem_euclid(13) - 6) as f32 * 0.25
}

/// The second operand, fetched through ERDMA. Deliberately a different
/// period from [`operand_a`] (11 vs 13) so no element pair repeats across
/// the cube -- an ERDMA that re-reads one atom for every position cannot
/// look right by coincidence.
fn operand_b(i: usize) -> f32 {
    ((i as i32 * 5).rem_euclid(11) - 5) as f32 * 0.5
}

fn shape(op: EwBinaryOp, geometry: Geometry) -> EwAddShape {
    EwAddShape {
        width: geometry.width,
        height: geometry.height,
        channels: geometry.channels,
        precision: EwPrecision::Fp16,
        op,
        // int8-only fields, unused at fp16.
        output_zero_point: 0,
        w_cvt_offset: 0,
        w_scale_ratio: 1.0,
        output_scale_ratio: 1.0,
    }
}

/// Submits one standalone EW task -- no conv, both tensors filled by the
/// CPU -- and returns the decoded output in the same logical order
/// [`logical`] defines.
fn run_binary(op: EwBinaryOp, geometry: Geometry) -> Vec<f32> {
    let elements = geometry.elements();
    let buffer_size = (elements * mem::size_of::<u16>()).next_multiple_of(4096);
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let buf_a = Buffer::new(fd, buffer_size, &file);
        ptr::write_bytes(buf_a.host_ptr, 0, buffer_size);
        let buf_b = Buffer::new(fd, buffer_size, &file);
        ptr::write_bytes(buf_b.host_ptr, 0, buffer_size);
        let buf_out = Buffer::new(fd, buffer_size, &file);
        // 0xa5a5 reads back as -0.02204895 in fp16 -- this crate's standard
        // "the DPU never wrote here" sentinel, so an unwritten position is
        // distinguishable from a real zero.
        ptr::write_bytes(buf_out.host_ptr, 0xa5, buffer_size);

        let a_cube = std::slice::from_raw_parts_mut(buf_a.host_ptr as *mut u16, elements);
        let b_cube = std::slice::from_raw_parts_mut(buf_b.host_ptr as *mut u16, elements);
        for i in 0..elements {
            let (c, y, x) = geometry.logical(i);
            a_cube[geometry.cube_index(c, y, x)] = f32_to_f16(operand_a(i));
            b_cube[geometry.cube_index(c, y, x)] = f32_to_f16(operand_b(i));
        }

        let cmds = build_add_regcmd(
            &shape(op, geometry),
            &EwAddBuffers {
                intermediate_addr: buf_a.dma_address,
                w_addr: buf_b.dma_address,
                output_addr: buf_out.dma_address,
            },
        );

        let cmd_bytes = cmds.len() * mem::size_of::<u64>();
        let buf_cmd = Buffer::new(fd, cmd_bytes.next_multiple_of(4096), &file);
        let cmd_slice = std::slice::from_raw_parts_mut(buf_cmd.host_ptr as *mut u64, cmds.len());
        for (i, c) in cmds.iter().enumerate() {
            cmd_slice[i] = c.0;
        }

        fini_bo(fd, buf_a.handle).ok();
        fini_bo(fd, buf_b.handle).ok();
        fini_bo(fd, buf_out.handle).ok();
        fini_bo(fd, buf_cmd.handle).ok();

        let in_handles = [buf_cmd.handle, buf_a.handle, buf_b.handle];
        let out_handles = [buf_out.handle];
        submit(
            fd,
            buf_cmd.dma_address,
            cmds.len() as u32,
            &in_handles,
            &out_handles,
        )
        .expect("SUBMIT ioctl failed");

        prep_bo(fd, buf_out.handle, 2_000_000_000).unwrap_or_else(|e| {
            panic!("standalone EW {op:?} job at {geometry:?} did not complete within timeout: {e}")
        });

        let raw = std::slice::from_raw_parts(buf_out.host_ptr as *const u16, elements);
        let out = (0..elements)
            .map(|i| {
                let (c, y, x) = geometry.logical(i);
                f16_to_f32(raw[geometry.cube_index(c, y, x)])
            })
            .collect();

        close_bo(fd, buf_a.handle).ok();
        close_bo(fd, buf_b.handle).ok();
        close_bo(fd, buf_out.handle).ok();
        close_bo(fd, buf_cmd.handle).ok();

        out
    }
}

/// What the readback actually looks like, independent of what was asked
/// for. A wrong answer here is far more useful than a bare mismatch count:
/// `A` alone means the ERDMA operand was never read (the failure
/// `../rocket-userspace` predicts for a flying main feed), `B` alone means
/// the main feed was lost, and a hypothesis matching for a *different* op
/// than the one requested means the opcode mapping is wrong rather than
/// the datapath.
fn classify(got: &[f32]) -> String {
    let elements = got.len();
    let hypotheses: [(&str, Oracle); 7] = [
        ("A+B", |a, b| a + b),
        ("A-B", |a, b| a - b),
        ("A*B", |a, b| a * b),
        ("max(A,B)", |a, b| a.max(b)),
        ("min(A,B)", |a, b| a.min(b)),
        ("A alone (operand unread)", |a, _| a),
        ("B alone (main feed lost)", |_, b| b),
    ];
    for (name, f) in hypotheses {
        if (0..elements).all(|i| got[i] == f(operand_a(i), operand_b(i))) {
            return format!("{name}, exactly");
        }
    }
    let unwritten = got.iter().filter(|&&v| v == f16_to_f32(0xa5a5)).count();
    if unwritten > 0 {
        return format!("{unwritten}/{elements} positions never written by the DPU");
    }
    if got.iter().all(|&v| v == 0.0) {
        return "all zero".to_string();
    }
    "none of the hypotheses".to_string()
}

/// Runs one op and checks every element against its host oracle. fp16 is
/// exact for these operands by construction (see [`operand_a`]), so the
/// comparison is `==`, not a tolerance.
fn check_geometry(op: EwBinaryOp, geometry: Geometry, oracle: Oracle) {
    let elements = geometry.elements();
    let got = run_binary(op, geometry);
    eprintln!(
        "standalone EW {op:?} {}x{}x{}: readback is {}",
        geometry.width,
        geometry.height,
        geometry.channels,
        classify(&got)
    );

    let mut mismatches = 0usize;
    let mut first: Option<(usize, f32, f32)> = None;
    for (i, &saw) in got.iter().enumerate() {
        let want = oracle(operand_a(i), operand_b(i));
        if saw != want {
            mismatches += 1;
            first.get_or_insert((i, want, saw));
        }
    }
    if let Some((i, want, saw)) = first {
        let (c, y, x) = geometry.logical(i);
        panic!(
            "standalone EW {op:?} at {geometry:?}: {mismatches}/{elements} elements wrong; \
             readback is {}. First at element {i} (channel {c}, y {y}, x {x}): want {want}, \
             got {saw}",
            classify(&got)
        );
    }
}

fn check(op: EwBinaryOp, oracle: Oracle) {
    check_geometry(op, Geometry::TWO_SURFACES, oracle);
}

/// The op the investigation is about. Everything else in this file is
/// context for reading its result.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn standalone_ew_mul_matches_oracle() {
    check(EwBinaryOp::Mul, |a, b| a * b);
}

/// The control: the one opcode already confirmed on silicon (through the
/// conv-then-add chain). If this fails over non-uniform data while
/// `conv_with_add_hw.rs` passes over uniform data, the defect is in the
/// ERDMA addressing, not in the multiply.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn standalone_ew_add_matches_oracle() {
    check(EwBinaryOp::Add, |a, b| a + b);
}

#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn standalone_ew_sub_matches_oracle() {
    check(EwBinaryOp::Sub, |a, b| a - b);
}

/// `ew_alu_algo=0`. TRM-documented, used by the reference emitter, never
/// measured in this repo -- ROADMAP.md lists it as needing exactly this
/// gate before anything depends on it.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn standalone_ew_max_matches_oracle() {
    check(EwBinaryOp::Max, |a, b| a.max(b));
}

/// `ew_alu_algo=1`, same status as [`standalone_ew_max_matches_oracle`].
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn standalone_ew_min_matches_oracle() {
    check(EwBinaryOp::Min, |a, b| a.min(b));
}

/// Every element of `A*B` across four cube shapes, run in one process:
/// the two-surface default, a shape whose `width * height` sits *below*
/// `build_add_regcmd`'s `EW_SURF_STRIDE` floor of 12 (where the register
/// over-states the real surface stride), an odd non-square one, and a
/// 64-channel cube spanning 8 surfaces. Geometry is where an ERDMA
/// addressing bug would hide, and a single small square cube is exactly
/// the shape that cannot find one.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn standalone_ew_mul_geometry_ladder() {
    let ladder = [
        Geometry::TWO_SURFACES,
        // width * height = 9 < 12, the EW_SURF_STRIDE floor.
        Geometry {
            width: 3,
            height: 3,
            channels: 16,
        },
        Geometry {
            width: 7,
            height: 5,
            channels: 24,
        },
        Geometry {
            width: 14,
            height: 14,
            channels: 64,
        },
        // ViT's own token-by-embedding shape, which is where the element-wise
        // sites in a real model are: `1x197x768` maps to 197 pixels of 768
        // channels at height one. That is **96 fp16 surfaces**, an order of
        // magnitude past the 8 the entry above reaches, and the compiler
        // matcher's channel bound has no business admitting it on the
        // strength of a 64-channel cube. `build_lut_regcmd`'s surface stride
        // was wrong for exactly this reason and nothing caught it, because
        // every test of it used one surface (see `lut_multi_surface_hw.rs`).
        Geometry {
            width: 197,
            height: 1,
            channels: 768,
        },
    ];
    for geometry in ladder {
        check_geometry(EwBinaryOp::Mul, geometry, |a, b| a * b);
    }
}

/// ViT's shape across every operator, not just `Mul`.
///
/// The ladder above widens geometry for one op; this widens ops at the one
/// geometry a compiled model will actually ask for. Both matter: an ERDMA
/// addressing fault would show in the first, and an operator that happens to
/// mis-handle a deep cube would show only here.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn standalone_ew_vit_shape_matches_oracle_for_every_op() {
    let vit = Geometry {
        width: 197,
        height: 1,
        channels: 768,
    };
    let cases: [(EwBinaryOp, Oracle); 5] = [
        (EwBinaryOp::Add, |a, b| a + b),
        (EwBinaryOp::Sub, |a, b| a - b),
        (EwBinaryOp::Mul, |a, b| a * b),
        (EwBinaryOp::Max, |a, b| a.max(b)),
        (EwBinaryOp::Min, |a, b| a.min(b)),
    ];
    for (op, oracle) in cases {
        check_geometry(op, vit, oracle);
    }

    // ViT-base's MLP is 768 -> 3072 -> 768, so the element-wise ops inside
    // the feed-forward block run at four times the embedding width: 384 fp16
    // surfaces. This is the widest cube any measured model asks of this path,
    // and it is what the compiler matcher's channel bound is set from.
    check_geometry(
        EwBinaryOp::Mul,
        Geometry {
            width: 197,
            height: 1,
            channels: 3072,
        },
        |a, b| a * b,
    );
}
