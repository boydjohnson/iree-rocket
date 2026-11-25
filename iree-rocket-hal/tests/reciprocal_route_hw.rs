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
    activation::{LutBuffers, LutShape, LutTable, build_lut_q15_regcmd},
    bn_plane::{
        BnPlaneBuffers, BnPlaneOp, BnPlaneShape, PlaneElement, build_bn_plane_regcmd,
        build_plane_gather_regcmd, write_bn_mul_plane_fp16,
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
