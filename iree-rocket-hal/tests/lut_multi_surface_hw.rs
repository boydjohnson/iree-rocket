//! Does the standalone DPU LUT work on a cube with more than one surface?
//!
//! Every existing LUT test -- `lut_hw.rs`, `lut_erf_hw.rs`, `lut_exp_hw.rs`,
//! `lut_sqrt_hw.rs`, `lut_rsqrt_hw.rs`, `lut_log_hw.rs`,
//! `lut_reciprocal_hw.rs`, `ew_square_hw.rs`, `lut_zero_join_hw.rs` -- uses
//! `channels: 16`. At int8 a feature atom holds 16 channels, so all of them
//! run a **single-surface** cube, where `DPU_DST_SURF_STRIDE` and
//! `DPU_SURFACE_ADD` are never used to advance anywhere. `conv_then_lut_hw.rs`
//! uses `channels: 1`, which is the same case.
//!
//! That matters because `build_lut_regcmd` computes
//!
//! ```text
//! surface_stride = width * height * task_channels
//! ```
//!
//! while the two builders that *are* validated across surface boundaries
//! compute it without the channel factor:
//!
//! - `elementwise.rs`'s `build_add_regcmd`/`build_unary_regcmd` use
//!   `output_area = width * height`, and `ew_binary_hw.rs` gates that over a
//!   two-surface default plus a geometry ladder up to a 64-channel,
//!   8-surface cube.
//! - `pooling.rs` computes `(width * ATOMIC_K_SIZE * height) /
//!   FEATURE_ATOMIC_SIZE` with both constants 16, i.e. `width * height`
//!   again (rounded up to four pixels), and that is capture-derived.
//!
//! So the register's unit is the 16-byte feature atom, and `build_lut_regcmd`
//! looks 16x too large for `channels = 16`. `build_add_regcmd`'s own doc
//! comment already flags the disagreement in passing -- "`width*height`, no
//! channel factor -- unlike `build_lut_regcmd`'s own" -- without saying which
//! is right, because at the shapes tested it could not matter.
//!
//! ROADMAP Phase 1 puts the LUT path on the wire, where any real tensor has
//! more than 16 channels, so this has to be settled before a `LutDef` exists.
//! This test settles it on hardware rather than by reading the two builders
//! against each other.
//!
//! The buffers here are deliberately 64 KiB for a 512-byte cube. If the
//! stride really is 16x too large the second surface is written ~8 KiB out;
//! a tight buffer would put that in someone else's memory, and a large one
//! turns a potential corruption into an observation. `surface_one_offset`
//! reports where the second surface actually landed, which measures the
//! register's unit directly instead of inferring it.
//!
//! ```text
//! CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!   cargo test --target aarch64-unknown-linux-gnu --release \
//!     --test lut_multi_surface_hw --no-run
//!
//! ./lut_multi_surface_hw-<hash> --ignored --nocapture
//! ```

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    activation::{LutBuffers, LutShape, LutTable, build_lut_regcmd},
    device::{Buffer, close_bo, fini_bo, prep_bo, submit},
};

const DEVICE_PATH: &str = "/dev/accel/accel0";

/// Far larger than the biggest cube on the ladder (14x14x64 = 12544 bytes),
/// so a wrong surface stride writes somewhere observable inside our own
/// allocation instead of out of it.
const TENSOR_SIZE: usize = 65536;

/// An int8 feature atom holds 16 channels, so a cube's surface count is
/// `ceil(channels / 16)` and one surface is `width * height` atoms.
const FEATURE_ATOMIC_SIZE: usize = 16;

/// tanh at the shape `lut_hw.rs::lut_standalone_tanh_matches_oracle` gates,
/// widened past one surface.
fn shape_of(width: u32, height: u32, channels: u32) -> LutShape {
    LutShape {
        width,
        height,
        channels,
        input_zero_point: 0x80,
        output_zero_point: 0x80,
        input_scale: 1.0 / 32.0,
        output_scale: 1.0 / 128.0,
    }
}

/// Number of bytes the cube occupies if the register unit is the 16-byte
/// feature atom, which is what `pooling.rs` and `elementwise.rs` both assume:
/// `surfaces * width * height * 16`.
fn cube_bytes(shape: &LutShape) -> usize {
    let task_channels = (shape.channels as usize)
        .max(FEATURE_ATOMIC_SIZE)
        .next_multiple_of(FEATURE_ATOMIC_SIZE);
    let surfaces = task_channels / FEATURE_ATOMIC_SIZE;
    surfaces * (shape.width * shape.height) as usize * FEATURE_ATOMIC_SIZE
}

/// Element `i` carries input code `i - 128`, wrapping, so each surface holds a
/// different slice of the ramp where the cube is not a multiple of 256 and, in
/// any case, a surface that was never written reads as zero rather than as a
/// plausible duplicate of its neighbour.
fn ramp(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i as u8).wrapping_add(128)).collect()
}

fn run(shape: &LutShape, table: LutTable, input: &[u8]) -> Vec<u8> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let buf_a = Buffer::new(fd, TENSOR_SIZE, &file);
        ptr::write_bytes(buf_a.host_ptr, 0, TENSOR_SIZE);
        ptr::copy_nonoverlapping(input.as_ptr(), buf_a.host_ptr, input.len());

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

        submit(
            fd,
            buf_cmd.dma_address,
            cmds.len() as u32,
            &[buf_cmd.handle, buf_a.handle],
            &[buf_c.handle],
        )
        .expect("SUBMIT ioctl failed");

        prep_bo(fd, buf_c.handle, 2_000_000_000).unwrap_or_else(|e| {
            panic!(
                "LUT job for {}x{}x{} did not complete within timeout: {e}",
                shape.width, shape.height, shape.channels
            )
        });

        let out = std::slice::from_raw_parts(buf_c.host_ptr, TENSOR_SIZE).to_vec();

        close_bo(fd, buf_a.handle).ok();
        close_bo(fd, buf_c.handle).ok();
        close_bo(fd, buf_cmd.handle).ok();

        out
    }
}

/// Where a surface's first pixel actually landed, searched over the whole
/// output buffer on 16-byte boundaries. Reported rather than asserted: it
/// measures the register's unit directly, which is what turns "surface 1 is
/// wrong" into "the stride is 16x too large".
fn find_pixel(out: &[u8], expected: &[u8], skip_before: usize) -> Option<usize> {
    (0..out.len().saturating_sub(expected.len()))
        .step_by(FEATURE_ATOMIC_SIZE)
        .find(|&offset| offset >= skip_before && &out[offset..offset + expected.len()] == expected)
}

/// One cube: run it, check every element against the tanh oracle, and report
/// per-surface so a stride fault reads as "surface N onward" rather than as a
/// scattered error count.
///
/// Returns a failure description, or `None` when the cube is correct.
fn check(width: u32, height: u32, channels: u32) -> Option<String> {
    let shape = shape_of(width, height, channels);
    let bytes = cube_bytes(&shape);
    assert!(
        bytes <= TENSOR_SIZE,
        "ladder entry {width}x{height}x{channels} needs {bytes} bytes, past TENSOR_SIZE"
    );
    let pixels = (width * height) as usize;
    let surface_bytes = pixels * FEATURE_ATOMIC_SIZE;
    let surfaces = bytes / surface_bytes;

    let input = ramp(bytes);
    let out = run(&shape, LutTable::tanh(), &input);

    let tolerance = 2.0 * shape.output_scale;
    let expect = |code: i8| ((code as f32) * shape.input_scale).tanh();

    // Channels past the real count are padding the cube carries but no caller
    // reads; the hardware may write anything there, so only the real channels
    // are checked. `orig_channel` vs `channel` is exactly this distinction.
    let real_c2_of =
        |surface: usize, c2: usize| surface * FEATURE_ATOMIC_SIZE + c2 < channels as usize;

    let mut per_surface = vec![0usize; surfaces];
    let mut samples = Vec::new();
    for (surface, wrong) in per_surface.iter_mut().enumerate() {
        for pixel in 0..pixels {
            for c2 in 0..FEATURE_ATOMIC_SIZE {
                if !real_c2_of(surface, c2) {
                    continue;
                }
                let element = (surface * pixels + pixel) * FEATURE_ATOMIC_SIZE + c2;
                let got = (out[element] as i8) as f32 * shape.output_scale;
                let want = expect(input[element] as i8);
                if (got - want).abs() > tolerance {
                    *wrong += 1;
                    if samples.len() < 6 {
                        samples.push(format!(
                            "surface {surface} pixel {pixel} c2 {c2} (byte {element}): \
                             input code {}, want {want}, got {got}",
                            input[element] as i8
                        ));
                    }
                }
            }
        }
    }

    let total: usize = per_surface.iter().sum();
    eprintln!(
        "  {width}x{height}x{channels}: {surfaces} surface(s), {bytes} bytes, \
         wrong per surface {per_surface:?}"
    );

    if total == 0 {
        return None;
    }

    // Only worth the search when something is wrong: locate surface 1 so the
    // failure names the stride the hardware actually used.
    if surfaces > 1 {
        let first_pixel: Vec<u8> = (0..FEATURE_ATOMIC_SIZE)
            .map(|c2| {
                let element = pixels * FEATURE_ATOMIC_SIZE + c2;
                (expect(input[element] as i8) / shape.output_scale).round() as i8 as u8
            })
            .collect();
        match find_pixel(&out, &first_pixel, FEATURE_ATOMIC_SIZE) {
            Some(offset) => eprintln!(
                "    surface 1's first pixel landed at byte {offset} ({} atoms); the correct \
                 stride is {pixels} atoms, and this build programmed {}",
                offset / FEATURE_ATOMIC_SIZE,
                width * height,
            ),
            None => eprintln!("    surface 1's first pixel is nowhere in the output buffer"),
        }
    }

    Some(format!(
        "{width}x{height}x{channels}: {total} wrong elements, per surface {per_surface:?}\n      {}",
        samples.join("\n      ")
    ))
}

/// A geometry ladder past one surface, matching the depth `ew_binary_hw.rs`
/// holds the EW builders to (which is what makes `elementwise.rs`'s
/// `output_area` the trustworthy statement of this register's unit).
///
/// Before the fix this test drove, `build_lut_regcmd` programmed
/// `width * height * task_channels`: at 4x4x32 the second surface was
/// addressed 8 KiB out and came back entirely unwritten (255 of 256 bytes
/// wrong, and its data nowhere in a 64 KiB buffer).
#[test]
#[ignore = "needs the real NPU device -- cross-compile for aarch64, copy to the board, run there"]
fn lut_multi_surface_cubes_are_correct() {
    let ladder = [
        // The minimum failing case: two surfaces, everything else identical
        // to the shape `lut_hw.rs` already gates.
        (4, 4, 32),
        // Four surfaces.
        (4, 4, 64),
        // Channels not a multiple of the atom: `orig_channel` 19 against a
        // `channel` of 31, so the cube carries padding no caller reads.
        (4, 4, 20),
        // Odd, non-square, unaligned area (35 pixels) across three surfaces --
        // the geometry that exposed pooling's own four-pixel surface rounding.
        (7, 5, 48),
        // The top of `ew_binary_hw.rs`'s own geometry ladder. That test is
        // fp16, where a 16-byte atom holds 8 channels and this is eight
        // surfaces; `LutShape` is int8-only, so here it is four -- the point
        // it carries is the large, non-trivial `width * height`, not the
        // surface count, which the entries above already vary.
        (14, 14, 64),
    ];

    eprintln!("=== LUT multi-surface ladder (tanh) ===");
    let failures: Vec<String> = ladder
        .into_iter()
        .filter_map(|(w, h, c)| check(w, h, c))
        .collect();

    assert!(
        failures.is_empty(),
        "{} of {} multi-surface LUT cubes are wrong -- see this test's module comment:\n  {}",
        failures.len(),
        ladder.len(),
        failures.join("\n  ")
    );
}
