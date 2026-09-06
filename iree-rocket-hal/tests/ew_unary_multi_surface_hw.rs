//! Does `build_unary_regcmd` write the right bytes on a cube with more than
//! one surface, and with a position-dependent input?
//!
//! `ew_unary_hw.rs` and `ew_round_hw.rs` are the only tests of this builder
//! and both run `channels: 1` with a **uniform fill**, deliberately --
//! `ew_unary_hw.rs`'s module comment says so, and says a multi-channel round
//! is "real follow-up work, not done here". This is that round.
//!
//! Two things follow from those choices that a reader of the older tests
//! should know, and that this file removes:
//!
//! 1. A uniform fill cannot see a layout fault. `ew_unary_hw.rs` reads
//!    `width * height` consecutive fp16 words and calls them pixels, but the
//!    cube is NC1HWC2: pixel `p`'s channel 0 sits at fp16 element `p * 8`, so
//!    those 16 words are actually the first two pixels' 8 channels each. Every
//!    channel holds the same value under a uniform fill, so the assertion
//!    passes either way. What that test really proves is the ALU opcode
//!    arithmetic, which is what it set out to prove.
//! 2. `ew_unary_hw.rs`'s stated reason for staying at one channel -- that
//!    `build_add_regcmd`'s multi-channel layout was itself unconfirmed -- is
//!    now stale. `ew_binary_hw.rs` (2026-09-06) gates that builder at 16, 24
//!    and 64 fp16 channels, i.e. 2, 4 and 8 surfaces.
//!
//! The specific hazard is the surface stride. `build_unary_regcmd` programs
//! `DPU_DST_SURF_STRIDE`/`DPU_SURFACE_ADD` as `output_area = width * height`,
//! which is the atom-unit rule `pooling.rs` and `build_add_regcmd` both
//! independently state. `build_lut_regcmd` had the same registers wrong by a
//! factor of `task_channels` and nothing caught it, for exactly the reason
//! above: every LUT test used a single surface. See
//! `lut_multi_surface_hw.rs`. This file checks the unary EW path is not the
//! same story.
//!
//! fp16 only, which is all `EwUnaryShape` supports. At fp16 a 16-byte feature
//! atom holds 8 channels, so a cube has `task_channels / 8` surfaces --
//! `channels: 16` is already two.
//!
//! Inputs are multiples of 0.5 in `[-128, 127.5]`: exact in fp16, distinct per
//! element so a misplaced surface cannot pass as a duplicate, and non-integral
//! so `Floor` and `Ceil` are doing something rather than acting as identity.
//! The oracle is therefore exact and the tolerance is zero.
//!
//! ```text
//! CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!   cargo test --target aarch64-unknown-linux-gnu --release \
//!     --test ew_unary_multi_surface_hw --no-run
//!
//! ./ew_unary_multi_surface_hw-<hash> --ignored --nocapture
//! ```

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    device::{Buffer, close_bo, fini_bo, prep_bo, submit},
    elementwise::{EwUnaryAlgo, EwUnaryBuffers, EwUnaryShape, build_unary_regcmd},
};

const DEVICE_PATH: &str = "/dev/accel/accel0";

/// Comfortably past the largest cube on the ladder (14x14x64 is 8 surfaces of
/// 196 pixels, 25088 bytes), so a wrong stride writes somewhere observable
/// inside our own allocation rather than out of it.
const TENSOR_SIZE: usize = 131072;

/// fp16 channels per 16-byte feature atom.
const C2: usize = 8;

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
            let mut exponent = -1i32;
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

/// The channel count the task actually programs, and from it the surface
/// count. `build_unary_regcmd` rounds to a multiple of 16 regardless of
/// precision (the same formula `build_add_regcmd` and `build_lut_regcmd`
/// use), while fp16 packs 8 channels per surface -- so `channels: 24` is a
/// 32-channel task and four surfaces, not three.
fn task_channels(channels: u32) -> usize {
    (channels as usize).max(16).next_multiple_of(16)
}

/// Multiples of 0.5 in `[-128, 127.5]`, exact in fp16 and distinct across a
/// 512-element window.
fn input_value(element: usize) -> f32 {
    ((element % 512) as f32 - 256.0) * 0.5
}

fn run(shape: &EwUnaryShape, input: &[u16]) -> Vec<u16> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let buf_in = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_in.host_ptr, 0, TENSOR_SIZE);
        ptr::copy_nonoverlapping(
            input.as_ptr() as *const u8,
            buf_in.host_ptr,
            mem::size_of_val(input),
        );

        let buf_out = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_out.host_ptr, 0, TENSOR_SIZE);

        let bufs = EwUnaryBuffers {
            input_addr: buf_in.dma_address,
            output_addr: buf_out.dma_address,
        };
        let cmds = build_unary_regcmd(shape, &bufs);

        let cmd_bytes = cmds.len() * mem::size_of::<u64>();
        let cmd_len = cmd_bytes.next_multiple_of(4096);
        let buf_cmd = Buffer::new(fd, cmd_len, &file);
        let cmd_slice = std::slice::from_raw_parts_mut(buf_cmd.host_ptr as *mut u64, cmds.len());
        for (i, c) in cmds.iter().enumerate() {
            cmd_slice[i] = c.0;
        }

        fini_bo(fd, buf_in.handle).ok();
        fini_bo(fd, buf_out.handle).ok();
        fini_bo(fd, buf_cmd.handle).ok();

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
                "multi-surface unary EW-ALU job did not complete within timeout \
                 ({}x{}x{} {:?}): {e}",
                shape.width, shape.height, shape.channels, shape.algo
            )
        });

        let out =
            std::slice::from_raw_parts(buf_out.host_ptr as *const u16, TENSOR_SIZE / 2).to_vec();

        close_bo(fd, buf_in.handle).ok();
        close_bo(fd, buf_out.handle).ok();
        close_bo(fd, buf_cmd.handle).ok();

        out
    }
}

/// One cube and one opcode. Returns a failure description, or `None` when
/// every real channel of every pixel of every surface is exact.
fn check(width: u32, height: u32, channels: u32, algo: EwUnaryAlgo) -> Option<String> {
    let shape = EwUnaryShape {
        width,
        height,
        channels,
        algo,
        operand: 0,
    };
    let pixels = (width * height) as usize;
    let surfaces = task_channels(channels) / C2;
    let elements = surfaces * pixels * C2;
    assert!(
        elements * 2 <= TENSOR_SIZE,
        "ladder entry {width}x{height}x{channels} needs {} bytes, past TENSOR_SIZE",
        elements * 2
    );

    let input: Vec<u16> = (0..elements).map(|i| f32_to_f16(input_value(i))).collect();
    let out = run(&shape, &input);

    let oracle = |x: f32| match algo {
        EwUnaryAlgo::Abs => x.abs(),
        EwUnaryAlgo::Neg => -x,
        EwUnaryAlgo::Floor => x.floor(),
        EwUnaryAlgo::Ceil => x.ceil(),
        EwUnaryAlgo::Add => x,
    };

    let mut per_surface = vec![0usize; surfaces];
    let mut samples = Vec::new();
    for (surface, wrong) in per_surface.iter_mut().enumerate() {
        for pixel in 0..pixels {
            for c2 in 0..C2 {
                // Channels past the real count are cube padding no caller
                // reads; the hardware may write anything there.
                if surface * C2 + c2 >= channels as usize {
                    continue;
                }
                let element = (surface * pixels + pixel) * C2 + c2;
                let got = f16_to_f32(out[element]);
                let want = oracle(input_value(element));
                if got != want {
                    *wrong += 1;
                    if samples.len() < 6 {
                        samples.push(format!(
                            "surface {surface} pixel {pixel} c2 {c2} (fp16 element {element}): \
                             x={}, want {want}, got {got}",
                            input_value(element)
                        ));
                    }
                }
            }
        }
    }

    let total: usize = per_surface.iter().sum();
    eprintln!(
        "  {algo:?} {width}x{height}x{channels}: {surfaces} surface(s), \
         {elements} fp16 elements, wrong per surface {per_surface:?}"
    );
    if total == 0 {
        return None;
    }
    Some(format!(
        "{algo:?} {width}x{height}x{channels}: {total} wrong, per surface {per_surface:?}\n      {}",
        samples.join("\n      ")
    ))
}

/// Every opcode across a geometry ladder past one surface, with a
/// position-dependent input and an exact oracle.
///
/// `EwUnaryAlgo::Add` is excluded: its scalar operand is the one part of this
/// builder with no hardware confirmation at all (see `EwUnaryShape::operand`),
/// and `ew_round_hw.rs` owns that question. Mixing it in here would confuse a
/// layout result with an operand-encoding one.
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn ew_unary_multi_surface_cubes_are_exact() {
    let ladder = [
        // Two fp16 surfaces -- the minimum multi-surface cube.
        (4, 4, 16),
        // `channels: 24` rounds to a 32-channel task: four surfaces, with the
        // top eight channels cube padding.
        (4, 4, 24),
        // Odd, non-square, unaligned area (35 pixels).
        (7, 5, 32),
        // The top of `ew_binary_hw.rs`'s own ladder: eight surfaces.
        (14, 14, 64),
    ];
    let algos = [
        EwUnaryAlgo::Abs,
        EwUnaryAlgo::Neg,
        EwUnaryAlgo::Floor,
        EwUnaryAlgo::Ceil,
    ];

    eprintln!("=== EW unary multi-surface ladder ===");
    let mut failures = Vec::new();
    for algo in algos {
        for (w, h, c) in ladder {
            if let Some(failure) = check(w, h, c, algo) {
                failures.push(failure);
            }
        }
    }

    assert!(
        failures.is_empty(),
        "{} of {} multi-surface unary EW cubes are wrong -- see this test's module comment:\n  {}",
        failures.len(),
        algos.len() * ladder.len(),
        failures.join("\n  ")
    );
}
