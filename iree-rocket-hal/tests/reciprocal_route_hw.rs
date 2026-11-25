//! SOFTMAX.md probe **S4**: where the row reciprocal `1/d` comes from.
//!
//! §4 ranks four routes and puts the host last. S4 inverts that, and the
//! reason is `d`'s real range rather than any preference.
//!
//! **What `d` actually is.** Over the 64 real ViT-B/16 attention rows in
//! `fixtures/vit_b_16_attention_rows.f32` (§3.4), `d = sum(exp(x - m))`
//! runs from 1.03 to 122.8 with a median of 10.8, and **44% of rows are at
//! or past 16** -- the edge of `LutTable::reciprocal`'s real data. The
//! worst case is `N` itself, 197 for ViT, since a uniform row sums to `N`.
//! So the table's `[1, 16)` window covers a bit over half of a real
//! model's rows, and a widened table does not fix it: the blocker is the
//! *input*. An int8 code space is 128 linear steps, and `d`'s range is
//! 200:1, so the step at `d = 1` is worth more than `d` itself. A Newton
//! iteration cannot recover what the input quantization threw away, since
//! it multiplies by the same quantized `d`.
//!
//! An int16 LUT input would have the codes, and a self-derived table over
//! `[1, 256]` would have the range. But the Newton step then needs
//! `y0 * d` as a *fraction*, and probe S3 measured that the LUT's output
//! converter is an integer multiply-shift: a Q15 entry cannot leave the
//! task as a fraction. So the NPU route needs two unprobed mechanisms and
//! a new table to replace what the host does exactly, in `M` divides, on a
//! width-1 vector of a few hundred values.
//!
//! The one real argument against the host was §5's one-job structure: the
//! reciprocal sits between the sum and the normalize, so computing it on
//! the host splits one dispatch into two submissions.
//! `the_extra_submission_boundary_costs` prices that.
//!
//! Cross-compile this test, copy the resulting binary to the RK3588 board,
//! and run the ignored tests there:
//!
//! ```text
//! CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!   cargo test --target aarch64-unknown-linux-gnu --release \
//!     --test reciprocal_route_hw --no-run
//!
//! ./reciprocal_route_hw-<hash> --ignored --nocapture --test-threads=1
//! ```

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr, time::Instant};

#[path = "support/dispatch.rs"]
mod dispatch;

use iree_rocket_hal::rocket::{
    activation::{
        LutBuffers, LutShape, LutTable, build_lut_q15_regcmd, build_lut_q15_regcmd_with_plane,
        lut_bn_mul_operand,
    },
    bn_plane::{
        BnPlaneBuffers, BnPlaneOp, BnPlaneShape, PlaneElement, Q15BridgeBuffers, Q15BridgeShape,
        build_bn_plane_regcmd, build_plane_gather_regcmd, build_q15_bridge_regcmd,
        write_bn_alu_plane_fp32, write_bn_mul_plane_fp16,
    },
    builders::{
        Bits, Register, RegisterMeta,
        dpu::{DpuBsOwCfg, DpuSurfaceAdd},
    },
    device::{Buffer, close_bo, fini_bo, prep_bo, submit, submit_tasks},
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const BUFFER_SIZE: usize = 1024 * 1024;
const FEATURE_ATOM_BYTES: usize = 16;
const FP16_LANES_PER_ATOM: usize = 8;
const INT8_LANES_PER_ATOM: usize = 16;
/// `input_scale` and slope chosen in §3.4.
const EXP_INPUT_SCALE: f32 = 1.0 / 20.0;

fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    if value == 0.0 {
        return sign;
    }
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let fraction = bits & 0x7f_ffff;
    if exponent < 1 {
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

fn attention_rows() -> (Vec<f32>, usize, usize) {
    const RAW: &[u8] = include_bytes!("fixtures/vit_b_16_attention_rows.f32");
    const ROWS: usize = 64;
    let values: Vec<f32> = RAW
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect();
    let keys = values.len() / ROWS;
    (values, ROWS, keys)
}

fn total_variation(got: &[f32], want: &[f32]) -> f32 {
    0.5 * got
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
}

/// Byte offset in a DPU cube (surface stride `width * height` atoms, no
/// four-pixel rounding -- that is the PPU's).
fn cube_offset(channel: usize, pixel: usize, pixels: usize, lanes: usize, bytes: usize) -> usize {
    (channel / lanes) * pixels * FEATURE_ATOM_BYTES
        + pixel * FEATURE_ATOM_BYTES
        + (channel % lanes) * bytes
}

/// §7 passes 2 and 3 on the real rows, in §1.1's layout: keys on width,
/// rows on channels.
///
/// `exp` and the normalizing multiply both run on the device; the sum, the
/// reciprocal and one format conversion run on the host. What each does is
/// spelled out in the test body, because which side of the line a step
/// falls on is the result.
fn run_exp_then_normalize(
    rows: &[f32],
    row_count: usize,
    keys: usize,
) -> Option<(Vec<f32>, f32, f32)> {
    let table = LutTable::exp_with_decaying_tail(EXP_INPUT_SCALE);
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        // ---- device input: int8 codes, keys on width, rows on channels.
        let buf_codes = Buffer::new(fd, BUFFER_SIZE, &file);
        ptr::write_bytes(buf_codes.host_ptr, 0, BUFFER_SIZE);
        for row in 0..row_count {
            for key in 0..keys {
                let x = rows[row * keys + key];
                let code = (x / EXP_INPUT_SCALE).round().clamp(-128.0, 0.0) as i8;
                let at = cube_offset(row, key, keys, INT8_LANES_PER_ATOM, 1);
                ptr::write(buf_codes.host_ptr.add(at), code as u8);
            }
        }

        // ---- device: exp, Q15 out.
        let buf_q15 = Buffer::new(fd, BUFFER_SIZE, &file);
        ptr::write_bytes(buf_q15.host_ptr, 0, BUFFER_SIZE);
        let exp_cmds = build_lut_q15_regcmd(
            &LutShape {
                width: keys as u32,
                height: 1,
                channels: row_count as u32,
                input_zero_point: 0x80,
                output_zero_point: 0x80,
                input_scale: EXP_INPUT_SCALE,
                output_scale: 1.0,
            },
            &LutBuffers {
                input_addr: buf_codes.dma_address,
                output_addr: buf_q15.dma_address,
            },
            table,
        );

        let buf_exp_cmd = Buffer::new(
            fd,
            (exp_cmds.len() * mem::size_of::<u64>()).next_multiple_of(4096),
            &file,
        );
        ptr::write_bytes(buf_exp_cmd.host_ptr, 0, buf_exp_cmd.size);
        let slots =
            std::slice::from_raw_parts_mut(buf_exp_cmd.host_ptr as *mut u64, exp_cmds.len());
        for (slot, cmd) in slots.iter_mut().zip(&exp_cmds) {
            *slot = cmd.0;
        }

        for handle in [buf_codes.handle, buf_q15.handle, buf_exp_cmd.handle] {
            fini_bo(fd, handle).ok();
        }
        submit(
            fd,
            buf_exp_cmd.dma_address,
            exp_cmds.len() as u32,
            &[buf_exp_cmd.handle, buf_codes.handle],
            &[buf_q15.handle],
        )
        .expect("exp SUBMIT failed");
        prep_bo(fd, buf_q15.handle, 2_000_000_000).expect("exp did not complete");

        // ---- host, and the boundary S4 is about: sum the Q15 counts per
        // row and invert. `M` f32 divides, exactly.
        // Two host costs, timed apart because they are not the same
        // thing: the reciprocal is `M` divides on a width-1 vector and is
        // what S4 is about, while the Q15 -> fp16 conversion is a pass over
        // the whole N x M cube and is the open item.
        let mut convert_ms = 0.0f32;
        let mut reciprocal_ms = 0.0f32;
        let mut reciprocals = Vec::with_capacity(row_count);
        // The Q15 -> fp16 conversion is a *second* host pass, over the
        // whole N x M cube rather than the M-wide statistic vector, and it
        // is here only because the BN plane task is fp16 while the LUT
        // writes int16. A real section 7 has to remove it; see this file's
        // own report.
        let buf_p = Buffer::new(fd, BUFFER_SIZE, &file);
        ptr::write_bytes(buf_p.host_ptr, 0, BUFFER_SIZE);
        for row in 0..row_count {
            let mut sum = 0.0f32;
            let convert = Instant::now();
            for key in 0..keys {
                let at = cube_offset(row, key, keys, FP16_LANES_PER_ATOM, 2);
                let count = ptr::read_unaligned(buf_q15.host_ptr.add(at) as *const i16);
                sum += f32::from(count);
                // The 1/32768 that turns a Q15 count into a probability
                // rides here, and it has to: the plane's operand is fp16
                // (S1), and `1/sum` over Q15 counts is 1.5e-7 to 3e-5,
                // *below* fp16's normal range of 6.1e-5 -- the first
                // attempt refused it for exactly that reason. Dividing by
                // 32768 is exact in fp16, being a power of two, so it
                // costs nothing here; doing it on the device is the open
                // item this test leaves.
                ptr::write(
                    buf_p.host_ptr.add(at) as *mut u16,
                    f32_to_f16(f32::from(count) / 32768.0),
                );
            }
            convert_ms += convert.elapsed().as_secs_f32() * 1e3;
            let invert = Instant::now();
            assert!(sum > 0.0, "row {row} summed to zero");
            reciprocals.push(32768.0 / sum);
            reciprocal_ms += invert.elapsed().as_secs_f32() * 1e3;
        }

        // ---- device: the normalizing multiply, 1/d on the BN MUL plane.
        let plane_values: Vec<f32> = reciprocals.clone();
        let shape = BnPlaneShape {
            width: keys as u32,
            height: 1,
            channels: row_count as u32,
            op: BnPlaneOp::Mul,
            mul_shift: 0,
        };
        let buf_plane = Buffer::new(fd, BUFFER_SIZE, &file);
        ptr::write_bytes(buf_plane.host_ptr, 0, BUFFER_SIZE);
        let mut plane = vec![0u8; 4 * shape.programmed_channels() as usize];
        write_bn_mul_plane_fp16(&mut plane, shape.programmed_channels(), &plane_values)
            .expect("the reciprocals do not fit an fp16 plane");
        ptr::copy_nonoverlapping(plane.as_ptr(), buf_plane.host_ptr, plane.len());

        let buf_out = Buffer::new(fd, BUFFER_SIZE, &file);
        ptr::write_bytes(buf_out.host_ptr, 0x5A, BUFFER_SIZE);
        let norm_cmds = build_bn_plane_regcmd(
            &shape,
            &BnPlaneBuffers {
                input_addr: buf_p.dma_address,
                plane_addr: buf_plane.dma_address,
                output_addr: buf_out.dma_address,
            },
        );
        let buf_norm_cmd = Buffer::new(
            fd,
            (norm_cmds.len() * mem::size_of::<u64>()).next_multiple_of(4096),
            &file,
        );
        ptr::write_bytes(buf_norm_cmd.host_ptr, 0, buf_norm_cmd.size);
        let slots =
            std::slice::from_raw_parts_mut(buf_norm_cmd.host_ptr as *mut u64, norm_cmds.len());
        for (slot, cmd) in slots.iter_mut().zip(&norm_cmds) {
            *slot = cmd.0;
        }

        for handle in [
            buf_p.handle,
            buf_plane.handle,
            buf_out.handle,
            buf_norm_cmd.handle,
        ] {
            fini_bo(fd, handle).ok();
        }
        let started = Instant::now();
        submit(
            fd,
            buf_norm_cmd.dma_address,
            norm_cmds.len() as u32,
            &[buf_norm_cmd.handle, buf_p.handle, buf_plane.handle],
            &[buf_out.handle],
        )
        .expect("normalize SUBMIT failed");
        prep_bo(fd, buf_out.handle, 2_000_000_000).expect("normalize did not complete");
        let elapsed = started.elapsed();

        let mut weights = vec![0.0f32; row_count * keys];
        for row in 0..row_count {
            for key in 0..keys {
                let at = cube_offset(row, key, keys, FP16_LANES_PER_ATOM, 2);
                weights[row * keys + key] =
                    f16_to_f32(ptr::read_unaligned(buf_out.host_ptr.add(at) as *const u16));
            }
        }

        for handle in [
            buf_codes.handle,
            buf_q15.handle,
            buf_exp_cmd.handle,
            buf_p.handle,
            buf_plane.handle,
            buf_out.handle,
            buf_norm_cmd.handle,
        ] {
            close_bo(fd, handle).ok();
        }

        if dispatch::is_device_timeout(elapsed) {
            let strict = dispatch::report("exp then normalize", elapsed);
            assert!(!strict, "killed dispatch under ROCKET_STRICT_DISPATCH");
            return None;
        }
        Some((weights, reciprocal_ms, convert_ms))
    }
}

/// §7's passes 2 and 3 end to end on real attention rows, with the
/// reciprocal on the host.
///
/// The bound is §3.4's measured 1.12% plus room for the two fp16 steps
/// this adds: the Q15 counts narrowed to fp16, where integers above 2048
/// are no longer exact, and the plane's own fp16 multiplier.
#[test]
#[ignore = "requires an RK3588 NPU at /dev/accel/accel0"]
fn exp_then_a_host_reciprocal_normalizes_real_attention_rows() {
    let (rows, row_count, keys) = attention_rows();
    let Some((weights, reciprocal_ms, convert_ms)) = run_exp_then_normalize(&rows, row_count, keys)
    else {
        return;
    };

    let mut total = 0.0f32;
    let mut worst = 0.0f32;
    for row in 0..row_count {
        let logits = &rows[row * keys..(row + 1) * keys];
        let exact: Vec<f32> = logits.iter().map(|&x| x.exp()).collect();
        let sum: f32 = exact.iter().sum();
        let want: Vec<f32> = exact.iter().map(|&value| value / sum).collect();
        let got = &weights[row * keys..(row + 1) * keys];
        let tv = total_variation(got, &want);
        total += tv;
        worst = worst.max(tv);
    }
    let mean = total / row_count as f32;
    println!(
        "  {row_count} real ViT-B/16 rows, exp and normalize on the device: mean total \
         variation {:.2}%, worst {:.2}%. Host share, timed apart: the reciprocal itself is \
         {reciprocal_ms:.3} ms for {row_count} rows, while the Q15->fp16 conversion over the \
         whole {row_count}x{keys} cube is {convert_ms:.2} ms -- the first is what S4 chose to \
         put on the host, the second is the open item",
        100.0 * mean,
        100.0 * worst
    );
    assert!(
        mean < 0.03,
        "mean total variation {:.3}%; §3.4 measured 1.12% for the exp step alone, so a figure \
         far above that is the two fp16 narrowings this adds",
        100.0 * mean
    );
}

/// What the host reciprocal really costs: the submission boundary it forces
/// between the sum and the normalize.
///
/// §4's one argument against the host route was §5's one-job structure. A
/// second submission inside the same dispatch is not a second IREE dispatch
/// -- it is one more `SUBMIT` ioctl and one more `PREP_BO` wait -- so this
/// measures that round trip on the smallest real job there is, a width-1
/// plane gather.
#[test]
#[ignore = "requires an RK3588 NPU at /dev/accel/accel0"]
fn the_extra_submission_boundary_costs() {
    const ITERATIONS: usize = 200;
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let buf_in = Buffer::new(fd, 4096, &file);
        ptr::write_bytes(buf_in.host_ptr, 0, 4096);
        let buf_out = Buffer::new(fd, 4096, &file);
        ptr::write_bytes(buf_out.host_ptr, 0, 4096);
        let cmds = build_plane_gather_regcmd(
            8,
            1,
            PlaneElement::Fp16,
            buf_in.dma_address,
            buf_out.dma_address,
        );
        let buf_cmd = Buffer::new(
            fd,
            (cmds.len() * mem::size_of::<u64>()).next_multiple_of(4096),
            &file,
        );
        let slots = std::slice::from_raw_parts_mut(buf_cmd.host_ptr as *mut u64, cmds.len());
        for (slot, cmd) in slots.iter_mut().zip(&cmds) {
            *slot = cmd.0;
        }
        for handle in [buf_in.handle, buf_out.handle, buf_cmd.handle] {
            fini_bo(fd, handle).ok();
        }

        // One warm pass, then the measurement.
        for _ in 0..4 {
            submit(
                fd,
                buf_cmd.dma_address,
                cmds.len() as u32,
                &[buf_cmd.handle, buf_in.handle],
                &[buf_out.handle],
            )
            .expect("SUBMIT failed");
            prep_bo(fd, buf_out.handle, 2_000_000_000).expect("did not complete");
        }
        let started = Instant::now();
        for _ in 0..ITERATIONS {
            submit(
                fd,
                buf_cmd.dma_address,
                cmds.len() as u32,
                &[buf_cmd.handle, buf_in.handle],
                &[buf_out.handle],
            )
            .expect("SUBMIT failed");
            prep_bo(fd, buf_out.handle, 2_000_000_000).expect("did not complete");
        }
        let each = started.elapsed().as_secs_f64() * 1e3 / ITERATIONS as f64;

        for handle in [buf_in.handle, buf_out.handle, buf_cmd.handle] {
            close_bo(fd, handle).ok();
        }
        println!(
            "  a SUBMIT plus its PREP_BO wait on the smallest real job: {each:.3} ms. That is \
             what a host reciprocal adds per attention block, against ISSUES P8's 1.6 to 7.4 \
             ms of flat *per-dispatch* host cost -- the boundary is not a new dispatch"
        );
        assert!(
            each < 1.0,
            "{each:.3} ms per submission boundary; above a millisecond the host reciprocal \
             stops being obviously the right call and §4's NPU routes are worth the two \
             unprobed mechanisms they need"
        );
    }
}

/// Which channels the Q15 LUT actually writes, with a uniform input.
///
/// `exp_then_a_host_reciprocal_normalizes_real_attention_rows` found row 40
/// summing to zero at 64 channels, and every earlier Q15 run used *one*
/// channel -- where the output surface stride cannot be observed at all. So
/// this feeds code 0 everywhere, whose answer is the table's `exp(0)` entry
/// of 32768 in every channel, and reports which ones get it.
///
/// `PROBE_SURF_ADD` and `PROBE_SIZE_E` override the int16 writer's pair.
/// The fp32 writer needed `(size_e 1, surf_add = dst_surf_stride * 2)` and
/// `conv.rs` records `* 8` with `size_e 7` for the int32 integer path, so
/// the candidates are not free choices.
#[test]
#[ignore = "requires an RK3588 NPU at /dev/accel/accel0"]
fn the_q15_writer_channel_map() {
    let keys = 197usize;
    let channels = 64u32;
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let buf_in = Buffer::new(fd, BUFFER_SIZE, &file);
        ptr::write_bytes(buf_in.host_ptr, 0, BUFFER_SIZE);
        let buf_out = Buffer::new(fd, BUFFER_SIZE, &file);
        ptr::write_bytes(buf_out.host_ptr, 0, BUFFER_SIZE);

        let mut cmds = build_lut_q15_regcmd(
            &LutShape {
                width: keys as u32,
                height: 1,
                channels,
                input_zero_point: 0x80,
                output_zero_point: 0x80,
                input_scale: EXP_INPUT_SCALE,
                output_scale: 1.0,
            },
            &LutBuffers {
                input_addr: buf_in.dma_address,
                output_addr: buf_out.dma_address,
            },
            LutTable::exp(),
        );
        let surf_add = std::env::var("PROBE_SURF_ADD")
            .ok()
            .and_then(|value| value.parse::<u32>().ok());
        let size_e = std::env::var("PROBE_SIZE_E")
            .ok()
            .and_then(|value| value.parse::<u32>().ok());
        for cmd in &mut cmds {
            if let Some(surf_add) = surf_add
                && (cmd.0 >> 48) as u32 == DpuSurfaceAdd::DOMAIN
                && (cmd.0 as u32 & 0xffff) == DpuSurfaceAdd::OFFSET
            {
                *cmd = Register::<DpuSurfaceAdd>::new()
                    .surf_add(Bits::new(surf_add * keys as u32))
                    .build();
            }
            if let Some(size_e) = size_e
                && (cmd.0 >> 48) as u32 == DpuBsOwCfg::DOMAIN
                && (cmd.0 as u32 & 0xffff) == DpuBsOwCfg::OFFSET
            {
                *cmd = Register::<DpuBsOwCfg>::new()
                    .size_e_0(Bits::new(size_e))
                    .size_e_1(Bits::new(size_e))
                    .size_e_2(Bits::new(size_e))
                    .od_bypass(Bits::new(1))
                    .build();
            }
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
        .expect("SUBMIT failed");
        prep_bo(fd, buf_out.handle, 2_000_000_000).expect("did not complete");
        let elapsed = started.elapsed();

        let mut good = Vec::new();
        for channel in 0..channels as usize {
            let at = cube_offset(channel, 0, keys, FP16_LANES_PER_ATOM, 2);
            let count = ptr::read_unaligned(buf_out.host_ptr.add(at) as *const i16);
            if count == 32768i32 as i16 || count == 32767 {
                good.push(channel);
            }
        }
        for handle in [buf_in.handle, buf_out.handle, buf_cmd.handle] {
            close_bo(fd, handle).ok();
        }
        if dispatch::is_device_timeout(elapsed) {
            println!("  surf_add {surf_add:?} size_e {size_e:?}: HUNG");
            return;
        }
        println!(
            "  surf_add x{:?} size_e {:?}: {} of {channels} channels carry exp(0); {:?}",
            surf_add.unwrap_or(1),
            size_e.unwrap_or(1),
            good.len(),
            if good.len() > 12 {
                format!("{:?}..{:?}", &good[..6], &good[good.len() - 3..])
            } else {
                format!("{good:?}")
            }
        );
    }
}

// ===========================================================================
// The Q15-to-fp16 bridge: SOFTMAX.md section 9's last host step.
// ===========================================================================
//
// `exp` emits the table's Q15 integer as `int16`, and a standalone softmax
// owes the rest of the model fp16 probabilities. That conversion is the only
// step left in section 7 that is a pass over the whole `[N, M]` cube rather
// than a width-1 vector -- 0.18 ms for 64x197 on the host, against the
// reciprocal's 0.006 -- so it is the one worth moving onto the device.
//
// `build_q15_bridge_regcmd` is one DPU task that would do all of it: an
// integer read, an exact `2^-15` on the BS MUL stage from the config
// register, and the per-row `1/d` on the BN MUL plane. The three tests below
// take it in stages so a failure says which part, because the open question
// is the first one: MRDMA has an `mrdma_fp16tofp32_en` and no integer
// equivalent, so whether it can feed an integer cube into a float datapath
// at all is not established anywhere in this crate.

/// Stage 0, the control: fp16 in, fp16 out, no scaling. If this is not the
/// identity the harness is wrong and the other two mean nothing.
#[test]
#[ignore = "requires an RK3588 NPU at /dev/accel/accel0"]
fn the_bridge_passes_fp16_through_unchanged() {
    let values = [1.0f32, 0.5, 0.25, -2.0, 8.0, 0.0625];
    let Some(got) = run_bridge(&values, false, 0, None) else {
        return;
    };
    for (&want, &value) in values.iter().zip(&got) {
        assert!(
            (value - want).abs() <= 1e-3,
            "fp16 pass-through gave {value} for {want}"
        );
    }
    println!("  the bridge task is the identity on fp16 in, fp16 out");
}

/// Stage 1, the open question: can MRDMA read an `int16` cube into the
/// float datapath? Reported, not asserted -- a negative answer is the
/// result, and it keeps the conversion on the host.
#[test]
#[ignore = "requires an RK3588 NPU at /dev/accel/accel0"]
fn the_bridge_int16_read_probe() {
    // Q15 counts as they come out of `exp`: the peak, a half, and small
    // tail values.
    let counts = [32767.0f32, 16384.0, 4096.0, 221.0, 28.0, 1.0];
    match run_bridge(&counts, true, 0, None) {
        None => println!("  int16 read: NOT MEASURED"),
        Some(got) => {
            print!("  int16 read, no scaling:");
            for (&want, &value) in counts.iter().zip(&got) {
                print!(" {want}->{value}");
            }
            println!();
        }
    }
}

/// Stage 2: the same read with the BS MUL stage's `2^-15`, which is what
/// turns a Q15 count into a probability. Reported for the same reason.
#[test]
#[ignore = "requires an RK3588 NPU at /dev/accel/accel0"]
fn the_bridge_constant_scale_probe() {
    let counts = [32767.0f32, 16384.0, 4096.0, 221.0];
    for (label, int16) in [("int16 in", true), ("fp16 in", false)] {
        match run_bridge(&counts, int16, 15, None) {
            None => println!("  {label} with 2^-15: NOT MEASURED"),
            Some(got) => {
                print!("  {label} with 2^-15:");
                for (&want, &value) in counts.iter().zip(&got) {
                    print!(" {want}->{value:.6} (want {:.6})", want / 32768.0);
                }
                println!();
            }
        }
    }
}

/// Which plane constants the hardware honours: `2^-15` is an exact fp16
/// *subnormal*, `2^-14` the smallest *normal*. S1 only ever measured normal
/// operands.
#[test]
#[ignore = "requires an RK3588 NPU at /dev/accel/accel0"]
fn the_plane_constant_probe() {
    let counts = [32767.0f32, 16384.0, 4096.0];
    for (label, int16) in [("int16 in", true), ("fp16 in", false)] {
        for exponent in [-13i32, -14, -15, -16] {
            let plane = vec![2.0f32.powi(exponent)];
            match run_bridge(&counts, int16, 0, Some(&plane)) {
                None => println!("  {label} plane 2^{exponent}: NOT MEASURED"),
                Some(got) => {
                    print!("  {label} plane 2^{exponent}:");
                    for (&count, &value) in counts.iter().zip(&got) {
                        print!(
                            " {count}->{value:.6} (want {:.6})",
                            count * 2.0f32.powi(exponent)
                        );
                    }
                    println!();
                }
            }
        }
    }
}

/// Stage 3: the bridge that works, and it is **two** tasks rather than
/// one.
///
/// An `int16` input makes the datapath integer -- measured in
/// `the_plane_constant_probe`, where a plane of `2^-13` turns 16384 into
/// 2.0 correctly and 4096 into **0** instead of 0.5. So the int16 read
/// cannot also scale: it can only hand on exact integer counts as fp16,
/// which it does.
///
/// Everything fractional then happens on the fp16 path, where the plane
/// honours fractional operands down to at least `2^-16`, exact power-of-two
/// subnormals included. The constant `2^-15` turns a Q15 count into a
/// probability, and §7's pass 3 multiplies by `1/d` afterwards.
///
/// Why the two cannot be folded into one plane: their product is
/// `1/d_q15`, a non-power-of-two subnormal worth about 2% of relative
/// precision at a real row's median `d` and 24% at the worst (§4).
#[test]
#[ignore = "requires an RK3588 NPU at /dev/accel/accel0"]
fn the_bridge_turns_q15_counts_into_fp16_probabilities() {
    let counts = [32767.0f32, 16384.0, 4096.0, 221.0, 28.0, 1.0];

    // Task 1: int16 in, fp16 out, no scaling.
    let Some(as_fp16) = run_bridge(&counts, true, 0, None) else {
        return;
    };
    for (&count, &value) in counts.iter().zip(&as_fp16) {
        // fp16 is exact on integers to 2048 and steps by 16 at 32768, so
        // the peak rounds; everything below 2048 is exact.
        let tolerance = (count * 1e-3).max(1.0);
        assert!(
            (value - count).abs() <= tolerance,
            "the int16 read gave {value} for a count of {count}"
        );
    }

    // Task 2: fp16 in, the constant 2^-15 on the plane.
    let plane = vec![2.0f32.powi(-15)];
    let Some(probabilities) = run_bridge(&as_fp16, false, 0, Some(&plane)) else {
        return;
    };
    for (&count, &value) in counts.iter().zip(&probabilities) {
        let want = count / 32768.0;
        let tolerance = (want * 2e-3).max(2.0f32.powi(-24));
        assert!(
            (value - want).abs() <= tolerance,
            "a Q15 count of {count} became {value}, wanted {want}"
        );
    }
    println!(
        "  the bridge is two tasks: an int16 read that cannot scale, then a 2^-15 plane on \
         the fp16 path. Q15 counts become probabilities exactly."
    );
}

/// Runs the bridge over one row of `values`, written as fp16 or as `int16`
/// depending on `int16`, and returns the fp16 output.
fn run_bridge(
    values: &[f32],
    int16: bool,
    constant_shift: u32,
    plane: Option<&[f32]>,
) -> Option<Vec<f32>> {
    let width = values.len() as u32;
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let buf_in = Buffer::new(fd, BUFFER_SIZE, &file);
        ptr::write_bytes(buf_in.host_ptr, 0, BUFFER_SIZE);
        for (pixel, &value) in values.iter().enumerate() {
            let at = pixel * FEATURE_ATOM_BYTES;
            if int16 {
                ptr::write(buf_in.host_ptr.add(at) as *mut i16, value as i16);
            } else {
                ptr::write(buf_in.host_ptr.add(at) as *mut u16, f32_to_f16(value));
            }
        }
        let buf_plane = Buffer::new(fd, BUFFER_SIZE, &file);
        ptr::write_bytes(buf_plane.host_ptr, 0, BUFFER_SIZE);
        let shape = Q15BridgeShape {
            width,
            height: 1,
            channels: 1,
            input_int16: int16,
            constant_shift,
            per_channel: plane.is_some(),
            per_channel_add: false,
            out_scale: 0,
        };
        if let Some(plane) = plane {
            let mut bytes = vec![0u8; 4 * shape.channels.max(16) as usize];
            write_bn_mul_plane_fp16(&mut bytes, shape.channels.max(16), plane).expect("plane fits");
            ptr::copy_nonoverlapping(bytes.as_ptr(), buf_plane.host_ptr, bytes.len());
        }
        let buf_out = Buffer::new(fd, BUFFER_SIZE, &file);
        ptr::write_bytes(buf_out.host_ptr, 0x5A, BUFFER_SIZE);

        let cmds = build_q15_bridge_regcmd(
            &shape,
            &Q15BridgeBuffers {
                input_addr: buf_in.dma_address,
                plane_addr: if plane.is_some() {
                    buf_plane.dma_address
                } else {
                    0
                },
                output_addr: buf_out.dma_address,
            },
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
        for handle in [
            buf_in.handle,
            buf_plane.handle,
            buf_out.handle,
            buf_cmd.handle,
        ] {
            fini_bo(fd, handle).ok();
        }
        let started = Instant::now();
        submit(
            fd,
            buf_cmd.dma_address,
            cmds.len() as u32,
            &[buf_cmd.handle, buf_in.handle, buf_plane.handle],
            &[buf_out.handle],
        )
        .expect("SUBMIT failed");
        prep_bo(fd, buf_out.handle, 2_000_000_000)
            .unwrap_or_else(|error| panic!("the bridge task did not complete: {error}"));
        let elapsed = started.elapsed();
        let out: Vec<f32> = (0..values.len())
            .map(|pixel| {
                f16_to_f32(ptr::read_unaligned(
                    buf_out.host_ptr.add(pixel * FEATURE_ATOM_BYTES) as *const u16,
                ))
            })
            .collect();
        for handle in [
            buf_in.handle,
            buf_plane.handle,
            buf_out.handle,
            buf_cmd.handle,
        ] {
            close_bo(fd, handle).ok();
        }
        if dispatch::is_device_timeout(elapsed) {
            let strict = dispatch::report("q15 bridge", elapsed);
            assert!(!strict, "killed dispatch under ROCKET_STRICT_DISPATCH");
            return None;
        }
        Some(out)
    }
}

/// §7's subtract-and-quantize stage, the last mechanism the chain needed
/// and the one that was not measured: fp16 scores in, the per-row `-m` on
/// the BN **ALU** plane, and an int8 output scaled by 20 for §3.4's
/// `input_scale = 1/20`.
///
/// **First run, 2026-11-24: both arms hang at the watchdog**, and that is
/// the finding rather than a flaky kill -- it reproduces, and the same task
/// shape with an fp16 output is the identity
/// (`the_bridge_passes_fp16_through_unchanged`). So an int8 *output* is not
/// something this pass-through shape can be talked into by setting
/// `out_precision` to 0.
///
/// Which is unsurprising in hindsight, and says where the stage belongs:
/// subtract-and-quantize is a **requantisation with a per-channel bias**,
/// and that is exactly what `conv.rs`'s int8 path does every time it runs
/// -- bias on the BS plane, the multiplier and the output zero point on
/// `OUT_CVT`, `od_bypass` clear, `ow_src` set, the `BS_OW_OP` word
/// programmed. This probe set none of that. The stage should be derived
/// from that configuration rather than from an fp16 pass-through, which is
/// SOFTMAX.md §8's S5b.
#[test]
#[ignore = "requires an RK3588 NPU at /dev/accel/accel0"]
fn the_subtract_and_quantize_probe() {
    // One channel, so one `-m`. Scores already relative to a maximum of 0,
    // as they arrive in §7.
    let scores = [0.0f32, -0.25, -1.0, -2.5, -5.0, -6.4];
    let minus_m = 0.0f32;
    let want: Vec<i8> = scores
        .iter()
        .map(|&x| ((x + minus_m) * 20.0).round().clamp(-128.0, 127.0) as i8)
        .collect();

    for (label, out_scale) in [("scale 20", 20u32), ("scale 1", 1)] {
        match run_quantize(&scores, minus_m, out_scale) {
            None => println!(
                "  {label}: HUNG. The fp16 arm of this same shape is the identity, so this \
                 is the int8 output and not the task"
            ),
            Some(got) => {
                print!("  {label}:");
                for (index, &code) in got.iter().enumerate() {
                    print!(" {}->{code} (want {})", scores[index], want[index]);
                }
                println!();
            }
        }
    }
}

/// Runs one subtract-and-quantize task: fp16 in, int8 out.
fn run_quantize(scores: &[f32], minus_m: f32, out_scale: u32) -> Option<Vec<i8>> {
    let width = scores.len() as u32;
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();
    unsafe {
        let buf_in = Buffer::new(fd, BUFFER_SIZE, &file);
        ptr::write_bytes(buf_in.host_ptr, 0, BUFFER_SIZE);
        for (pixel, &x) in scores.iter().enumerate() {
            ptr::write(
                buf_in.host_ptr.add(pixel * FEATURE_ATOM_BYTES) as *mut u16,
                f32_to_f16(x),
            );
        }
        let buf_plane = Buffer::new(fd, BUFFER_SIZE, &file);
        ptr::write_bytes(buf_plane.host_ptr, 0, BUFFER_SIZE);
        // The ALU plane is a flat f32 per channel (S1).
        let mut plane = vec![0u8; 4 * 16];
        write_bn_alu_plane_fp32(&mut plane, 16, &[minus_m]).expect("plane fits");
        ptr::copy_nonoverlapping(plane.as_ptr(), buf_plane.host_ptr, plane.len());

        let buf_out = Buffer::new(fd, BUFFER_SIZE, &file);
        ptr::write_bytes(buf_out.host_ptr, 0x5A, BUFFER_SIZE);
        let cmds = build_q15_bridge_regcmd(
            &Q15BridgeShape {
                width,
                height: 1,
                channels: 1,
                input_int16: false,
                constant_shift: 0,
                per_channel: false,
                per_channel_add: true,
                out_scale,
            },
            &Q15BridgeBuffers {
                input_addr: buf_in.dma_address,
                plane_addr: buf_plane.dma_address,
                output_addr: buf_out.dma_address,
            },
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
        for handle in [
            buf_in.handle,
            buf_plane.handle,
            buf_out.handle,
            buf_cmd.handle,
        ] {
            fini_bo(fd, handle).ok();
        }
        let started = Instant::now();
        submit(
            fd,
            buf_cmd.dma_address,
            cmds.len() as u32,
            &[buf_cmd.handle, buf_in.handle, buf_plane.handle],
            &[buf_out.handle],
        )
        .expect("SUBMIT failed");
        prep_bo(fd, buf_out.handle, 2_000_000_000)
            .unwrap_or_else(|error| panic!("the quantize task did not complete: {error}"));
        let elapsed = started.elapsed();
        let out: Vec<i8> = (0..scores.len())
            .map(|pixel| ptr::read(buf_out.host_ptr.add(pixel * FEATURE_ATOM_BYTES)) as i8)
            .collect();
        for handle in [
            buf_in.handle,
            buf_plane.handle,
            buf_out.handle,
            buf_cmd.handle,
        ] {
            close_bo(fd, handle).ok();
        }
        if dispatch::is_device_timeout(elapsed) {
            let strict = dispatch::report("subtract and quantize", elapsed);
            assert!(!strict, "killed dispatch under ROCKET_STRICT_DISPATCH");
            return None;
        }
        Some(out)
    }
}

/// **S5b**: the per-row maximum subtracted *inside* the `exp` task, from an
/// NRDMA plane, which removes §7's subtract-and-quantize stage rather than
/// fixing it.
///
/// Four rows, each with its own maximum, fed as int8 codes that have *not*
/// been reduced by it. The plane carries `-m_code * lut_bn_mul_operand(..)`
/// per row -- the same arithmetic `lut_bn_alu` does for the scalar int8
/// zero point, since the operand is pre-scaled by the multiplier because
/// the add lands after the multiply (S1).
///
/// The plane element format is the open part: S1 measured the fp16 ALU
/// plane as a flat `f32` per channel, while the BS plane's int8 operands
/// are `i32`. The LUT runs on the int8 datapath, so `i32` is the
/// expectation; both are tried and the result says which.
#[test]
#[ignore = "requires an RK3588 NPU at /dev/accel/accel0"]
fn the_exp_task_subtracts_a_per_row_max_from_its_own_plane() {
    let input_scale = 1.0 / 20.0;
    let table = LutTable::exp_with_decaying_tail(input_scale);
    let operand = lut_bn_mul_operand(input_scale, table);
    // Row maxima as int8 codes, and scores at or below them.
    let maxima = [0i8, -20, -60, -100];
    let offsets = [0i8, -5, -20, -40];

    for (label, as_f32) in [("i32 plane", false), ("f32 plane", true)] {
        let mut plane = vec![0u8; 4 * 16];
        for (row, &m) in maxima.iter().enumerate() {
            let value = -(i64::from(m)) * i64::from(operand);
            let bytes = if as_f32 {
                (value as f32).to_bits().to_le_bytes()
            } else {
                (value as i32).to_le_bytes()
            };
            plane[row * 4..row * 4 + 4].copy_from_slice(&bytes);
        }
        match run_exp_with_plane(input_scale, table, &maxima, &offsets, &plane) {
            None => println!("  {label}: NOT MEASURED"),
            Some(counts) => {
                print!("  {label}:");
                let mut worst = 0.0f32;
                for (row, &m) in maxima.iter().enumerate() {
                    for (column, &offset) in offsets.iter().enumerate() {
                        let code = m.saturating_add(offset);
                        let x = f32::from(code - m) * input_scale;
                        let want = x.exp();
                        let got = f32::from(counts[row * offsets.len() + column]) / 32768.0;
                        worst = worst.max((got - want).abs());
                    }
                }
                println!(" worst absolute error {worst:.4} over 16 (row, column) pairs");
                if as_f32 {
                    assert!(
                        worst > 0.1,
                        "the f32 reading of the plane also works, so the element format is \
                         not datapath-determined after all and both are usable"
                    );
                } else {
                    assert!(
                        worst < 0.01,
                        "the i32 plane gave {worst} of absolute error, so the per-row maximum \
                         is not reaching the exp task's ALU stage and SOFTMAX.md section 7 \
                         still needs a subtract stage of its own"
                    );
                }
            }
        }
    }
}

/// Runs one `exp` task whose BN ALU operand comes per channel from `plane`.
/// Rows are channels, columns are width: §1.1's layout.
fn run_exp_with_plane(
    input_scale: f32,
    table: LutTable,
    maxima: &[i8],
    offsets: &[i8],
    plane: &[u8],
) -> Option<Vec<i16>> {
    run_exp_with_plane_at(input_scale, table, maxima, offsets, plane, 0)
}

/// `run_exp_with_plane` with the output written at `out_offset` into the
/// output BO rather than at its base.
///
/// `softmax_oracle_hw.rs` writes every intermediate at an offset into one
/// shared scratch allocation, and its `exp` task returns zeros while the
/// same program -- byte-identical, verified -- works here. An output offset
/// is the last difference left.
fn run_exp_with_plane_at(
    input_scale: f32,
    table: LutTable,
    maxima: &[i8],
    offsets: &[i8],
    plane: &[u8],
    out_offset: u32,
) -> Option<Vec<i16>> {
    let rows = maxima.len();
    let keys = offsets.len();
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();
    unsafe {
        let buf_in = Buffer::new(fd, BUFFER_SIZE, &file);
        ptr::write_bytes(buf_in.host_ptr, 0, BUFFER_SIZE);
        for (row, &m) in maxima.iter().enumerate() {
            for (column, &offset) in offsets.iter().enumerate() {
                let at = cube_offset(row, column, keys, INT8_LANES_PER_ATOM, 1);
                ptr::write(buf_in.host_ptr.add(at), m.saturating_add(offset) as u8);
            }
        }
        let buf_plane = Buffer::new(fd, BUFFER_SIZE, &file);
        ptr::write_bytes(buf_plane.host_ptr, 0, BUFFER_SIZE);
        ptr::copy_nonoverlapping(plane.as_ptr(), buf_plane.host_ptr, plane.len());
        let buf_out = Buffer::new(fd, BUFFER_SIZE, &file);
        ptr::write_bytes(buf_out.host_ptr, 0, BUFFER_SIZE);

        let cmds = build_lut_q15_regcmd_with_plane(
            &LutShape {
                width: keys as u32,
                height: 1,
                channels: rows as u32,
                input_zero_point: 0x80,
                output_zero_point: 0x80,
                input_scale,
                output_scale: 1.0,
            },
            &LutBuffers {
                input_addr: buf_in.dma_address,
                output_addr: buf_out.dma_address + out_offset,
            },
            table,
            buf_plane.dma_address,
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
        for handle in [
            buf_in.handle,
            buf_plane.handle,
            buf_out.handle,
            buf_cmd.handle,
        ] {
            fini_bo(fd, handle).ok();
        }
        let started = Instant::now();
        submit(
            fd,
            buf_cmd.dma_address,
            cmds.len() as u32,
            &[buf_cmd.handle, buf_in.handle, buf_plane.handle],
            &[buf_out.handle],
        )
        .expect("SUBMIT failed");
        prep_bo(fd, buf_out.handle, 2_000_000_000)
            .unwrap_or_else(|error| panic!("the planed exp task did not complete: {error}"));
        let elapsed = started.elapsed();
        let out: Vec<i16> = (0..rows * keys)
            .map(|index| {
                let at = out_offset as usize
                    + cube_offset(index / keys, index % keys, keys, FP16_LANES_PER_ATOM, 2);
                ptr::read_unaligned(buf_out.host_ptr.add(at) as *const i16)
            })
            .collect();
        for handle in [
            buf_in.handle,
            buf_plane.handle,
            buf_out.handle,
            buf_cmd.handle,
        ] {
            close_bo(fd, handle).ok();
        }
        if dispatch::is_device_timeout(elapsed) {
            let strict = dispatch::report("planed exp", elapsed);
            assert!(!strict, "killed dispatch under ROCKET_STRICT_DISPATCH");
            return None;
        }
        Some(out)
    }
}

/// S5b at scale: the same per-row-max exp task at the row and key counts
/// the assembled op uses.
///
/// `the_exp_task_subtracts_a_per_row_max_from_its_own_plane` ran 4 rows by
/// 4 keys, and `softmax_oracle_hw.rs` found the task returning zeros at 64
/// by 197. This bisects which extent does it: rows scale the ALU plane's
/// lane count, keys scale the cube.
#[test]
#[ignore = "requires an RK3588 NPU at /dev/accel/accel0"]
fn the_planed_exp_task_at_scale() {
    let input_scale = 1.0 / 20.0;
    let table = LutTable::exp_with_decaying_tail(input_scale);
    let operand = lut_bn_mul_operand(input_scale, table);

    for (rows, keys) in [
        (64usize, 197usize),
        (64, 198),
        (64, 199),
        (64, 200),
        (64, 204),
        (64, 256),
        (8, 200),
    ] {
        // Row r has maximum -r, and its keys step down from it.
        let maxima: Vec<i8> = (0..rows).map(|r| -(r as i8)).collect();
        let offsets: Vec<i8> = (0..keys).map(|k| -((k % 8) as i8)).collect();
        let mut plane = vec![0u8; 4 * rows.max(16)];
        for (row, &m) in maxima.iter().enumerate() {
            let value = -(i64::from(m)) * i64::from(operand);
            plane[row * 4..row * 4 + 4].copy_from_slice(&(value as i32).to_le_bytes());
        }
        match run_exp_with_plane(input_scale, table, &maxima, &offsets, &plane) {
            None => println!("  {rows} rows x {keys} keys: NOT MEASURED"),
            Some(counts) => {
                let mut worst = 0.0f32;
                let mut zero_rows = 0;
                for (row, &m) in maxima.iter().enumerate() {
                    let mut row_zero = true;
                    for (column, &offset) in offsets.iter().enumerate() {
                        let code = m.saturating_add(offset);
                        let want = (f32::from(code - m) * input_scale).exp();
                        let got = f32::from(counts[row * keys + column]) / 32768.0;
                        if got != 0.0 {
                            row_zero = false;
                        }
                        worst = worst.max((got - want).abs());
                    }
                    if row_zero {
                        zero_rows += 1;
                    }
                }
                println!(
                    "  {rows} rows x {keys} keys: worst absolute error {worst:.4}, \
                     {zero_rows}/{rows} rows all zero"
                );
            }
        }
    }
}

/// Does a PPU job before the `exp` task break it?
///
/// `softmax_oracle_hw.rs` assembles the op as a max tree (PPU) and then
/// `exp` (DPU LUT), and the `exp` task returns zeros there while being
/// measured correct at the same shape standalone
/// (`the_planed_exp_task_at_scale`, every width through 256). One job
/// leaving the core in a state that breaks the next is a class of problem
/// this repo already has on record -- ISSUES.md C8 -- so this runs a
/// pooling job first and then the same `exp` task.
#[test]
#[ignore = "requires an RK3588 NPU at /dev/accel/accel0"]
fn a_pooling_job_before_the_exp_task() {
    use iree_rocket_hal::rocket::pooling::{
        PoolingBuffers, PoolingMethod, PoolingPlan, PoolingPrecision, PoolingShape,
    };

    let input_scale = 1.0 / 20.0;
    let table = LutTable::exp_with_decaying_tail(input_scale);
    let operand = lut_bn_mul_operand(input_scale, table);
    let (rows, keys) = (64usize, 200usize);
    let maxima: Vec<i8> = (0..rows).map(|r| -(r as i8)).collect();
    let offsets: Vec<i8> = (0..keys).map(|k| -((k % 8) as i8)).collect();
    let mut plane = vec![0u8; 4 * rows];
    for (row, &m) in maxima.iter().enumerate() {
        let value = -(i64::from(m)) * i64::from(operand);
        plane[row * 4..row * 4 + 4].copy_from_slice(&(value as i32).to_le_bytes());
    }

    // A pooling job of exactly the shape the max tree's first level uses.
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();
    unsafe {
        let pool_in = Buffer::new(fd, BUFFER_SIZE, &file);
        ptr::write_bytes(pool_in.host_ptr, 0, BUFFER_SIZE);
        let pool_out = Buffer::new(fd, BUFFER_SIZE, &file);
        ptr::write_bytes(pool_out.host_ptr, 0, BUFFER_SIZE);
        let shape = PoolingShape {
            input_width: keys as u32,
            input_height: 1,
            input_channels: rows as u32,
            output_width: 25,
            output_height: 1,
            output_channels: rows as u32,
            precision: PoolingPrecision::Int8,
            kernel_width: 8,
            kernel_height: 1,
            stride_x: 8,
            stride_y: 1,
            method: PoolingMethod::Max,
            pad_left: 0,
            pad_top: 0,
            pad_right: 0,
            pad_bottom: 0,
            pad_value: PoolingMethod::Max.pad_fill_value(PoolingPrecision::Int8),
        };
        let programs = PoolingPlan::new(shape).programs_with_buffers(&PoolingBuffers {
            input_addr: pool_in.dma_address,
            output_addr: pool_out.dma_address,
        });
        let mut cmd_bufs = Vec::new();
        for commands in &programs {
            let buffer = Buffer::new(
                fd,
                (commands.len() * mem::size_of::<u64>()).next_multiple_of(4096),
                &file,
            );
            let slots = std::slice::from_raw_parts_mut(buffer.host_ptr as *mut u64, commands.len());
            for (slot, command) in slots.iter_mut().zip(commands) {
                *slot = command.0;
            }
            cmd_bufs.push((buffer, commands.len() as u32));
        }
        for handle in [pool_in.handle, pool_out.handle] {
            fini_bo(fd, handle).ok();
        }
        for (buffer, _) in &cmd_bufs {
            fini_bo(fd, buffer.handle).ok();
        }
        let tasks: Vec<(u32, u32)> = cmd_bufs
            .iter()
            .map(|(buffer, count)| (buffer.dma_address, *count))
            .collect();
        let mut in_handles: Vec<u32> = cmd_bufs.iter().map(|(b, _)| b.handle).collect();
        in_handles.push(pool_in.handle);
        submit_tasks(fd, &tasks, &in_handles, &[pool_out.handle]).expect("pooling SUBMIT failed");
        prep_bo(fd, pool_out.handle, 2_000_000_000).expect("pooling did not complete");
        println!("  ran a {}-task int8 max pool first", tasks.len());
        for (buffer, _) in &cmd_bufs {
            close_bo(fd, buffer.handle).ok();
        }
        close_bo(fd, pool_in.handle).ok();
        close_bo(fd, pool_out.handle).ok();
    }

    // Now the same exp task that passes on its own.
    match run_exp_with_plane(input_scale, table, &maxima, &offsets, &plane) {
        None => println!("  exp after pooling: NOT MEASURED"),
        Some(counts) => {
            let zero_rows = (0..rows)
                .filter(|&row| (0..keys).all(|key| counts[row * keys + key] == 0))
                .count();
            println!("  exp after pooling: {zero_rows}/{rows} rows all zero");
            assert_eq!(
                zero_rows, 0,
                "a pooling job before the exp task zeroes it, which is what \
                 softmax_oracle_hw.rs sees -- ISSUES.md C8's shape, one job leaving the core \
                 in a state the next cannot run in"
            );
        }
    }
}

/// Does writing at an offset into a shared allocation break the `exp` task?
///
/// The last difference between `softmax_oracle_hw.rs`'s failing `exp` and
/// the passing one here. Everything else is eliminated: the program is
/// byte-identical, the input codes are correct, the shape is measured at
/// this width and row count, and a preceding pooling job does not break it.
#[test]
#[ignore = "requires an RK3588 NPU at /dev/accel/accel0"]
fn the_exp_task_output_at_an_offset() {
    let input_scale = 1.0 / 20.0;
    let table = LutTable::exp_with_decaying_tail(input_scale);
    let operand = lut_bn_mul_operand(input_scale, table);
    let (rows, keys) = (64usize, 200usize);
    let maxima: Vec<i8> = (0..rows).map(|r| -(r as i8)).collect();
    let offsets: Vec<i8> = (0..keys).map(|k| -((k % 8) as i8)).collect();
    let mut plane = vec![0u8; 4 * rows];
    for (row, &m) in maxima.iter().enumerate() {
        let value = -(i64::from(m)) * i64::from(operand);
        plane[row * 4..row * 4 + 4].copy_from_slice(&(value as i32).to_le_bytes());
    }
    for out_offset in [0u32, 16, 4096, 15360] {
        match run_exp_with_plane_at(input_scale, table, &maxima, &offsets, &plane, out_offset) {
            None => println!("  output offset {out_offset}: NOT MEASURED"),
            Some(counts) => {
                let zero_rows = (0..rows)
                    .filter(|&row| (0..keys).all(|key| counts[row * keys + key] == 0))
                    .count();
                println!("  output offset {out_offset}: {zero_rows}/{rows} rows all zero");
            }
        }
    }
}
