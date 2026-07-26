//! Dumps the int8 output map, to see *where* a wrong value sits.
//!
//!   CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!     cargo test --target aarch64-unknown-linux-gnu --release \
//!       --test conv_int8_probe_hw --no-run
//!
//!   ./conv_int8_probe_hw-<hash> --ignored --nocapture
//!
//! # Why
//!
//! `conv_int8_hw` leaves one real anomaly: at `Cin` >= 8 with a 3x3 kernel,
//! interior pixels come back exactly one too high while the perimeter is
//! correct. In accumulator units the excess is `Cin / 8` -- one unit per
//! eight input channels -- and it disappears wherever a tap is supplied by
//! padding rather than by memory.
//!
//! Reasoning further from aggregate counts is guesswork, so this prints the
//! grid. A small image makes the whole map readable at once, and sweeping
//! `Cout` covers the one observation that does not fit an input-read story:
//! the same shape at `Cout` 1 is correct.

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    builders::{
        RegCmd, RegisterMeta,
        cna::{CnaDcompAddr0, CnaFeatureDataAddr},
        dpu::DpuDstBaseAddr,
        dpu_rdma::DpuRdmaBsBaseAddr,
    },
    conv::{
        BS_UNIT_MULTIPLIER, BsEntry, FeatureLayout, Kernels, Multiplier, Precision, Quantization,
        Shape, Tile, bs_buffer_bytes, conv_2d_tile, write_bs_buffer,
    },
    device::{Buffer, JobDesc, close_bo, fini_bo, prep_bo, submit_jobs},
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const FEATURE_ATOM_BYTES: usize = 16;
const PAGE_BYTES: usize = 4096;
const WIDTH: u32 = 8;
const HEIGHT: u32 = 8;

fn page_aligned_size(size: usize) -> usize {
    size.div_ceil(PAGE_BYTES) * PAGE_BYTES
}

fn decode_identity(command: &RegCmd) -> (u32, u32) {
    ((command.0 >> 48) as u32, command.0 as u32 & 0xffff)
}

fn relocate<R: RegisterMeta>(commands: &mut [RegCmd], address: u32) {
    let matches: Vec<_> = commands
        .iter()
        .enumerate()
        .filter_map(|(index, command)| {
            (decode_identity(command) == (R::DOMAIN, R::OFFSET)).then_some(index)
        })
        .collect();
    assert_eq!(matches.len(), 1, "expected exactly one relocation site");
    let tile_offset = (commands[matches[0]].0 >> 16) as u32;
    commands[matches[0]] = RegCmd::new(R::DOMAIN, R::OFFSET, address + tile_offset);
}

/// Runs one convolution and returns the output map for channel `channel`,
/// alongside the accumulator the test believes each pixel should have.
fn map(in_channels: u32, out_channels: u32, kernel: usize, channel: usize) -> Vec<i32> {
    map_at(in_channels, out_channels, kernel, channel, 0)
}

/// As [`map`], at a gain of `2^-shift`.
fn map_at(
    in_channels: u32,
    out_channels: u32,
    kernel: usize,
    channel: usize,
    shift: u32,
) -> Vec<i32> {
    map_with(
        in_channels,
        out_channels,
        kernel,
        channel,
        BS_UNIT_MULTIPLIER,
        Multiplier::for_unit_bs(1.0 / f64::from(1u32 << shift)),
    )
}

/// As [`map`], with the BS multiplier and the output conversion set
/// explicitly. Several pairs encode the same nominal gain, which is what
/// makes it possible to ask *where* a gain error lives.
fn map_with(
    in_channels: u32,
    out_channels: u32,
    kernel: usize,
    channel: usize,
    bs_multiplier: i16,
    cvt: Multiplier,
) -> Vec<i32> {
    let kernels: Kernels = [kernel, kernel];
    // At shift 0 the printed value *is* the accumulator, with nothing hidden
    // behind a scale.
    let precision = Precision::Int8(Quantization {
        input_zero_point: 0,
        output_zero_point: 0,
        multiplier: cvt,
    });
    let shape = Shape::with_precision(WIDTH, HEIGHT, 1, in_channels, out_channels, precision);
    let width = WIDTH as usize;
    let height = HEIGHT as usize;
    let in_surfaces = (shape.weight_channels() as usize).div_ceil(16);
    let input_bytes = match shape.layout() {
        FeatureLayout::Dense => width * height * in_channels as usize,
        FeatureLayout::Surfaces => in_surfaces * width * height * FEATURE_ATOM_BYTES,
    };
    let out_surfaces = (shape.padded_out_channels() as usize).div_ceil(16);
    let output_bytes = out_surfaces * width * height * FEATURE_ATOM_BYTES;

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let buf_input = Buffer::new(fd, page_aligned_size(input_bytes), &file);
        ptr::write_bytes(buf_input.host_ptr, 0, buf_input.size);
        for c in 0..in_channels as usize {
            let surface = c / 16;
            let lane = c % 16;
            for y in 0..height {
                for x in 0..width {
                    let offset = match shape.layout() {
                        FeatureLayout::Dense => (y * width + x) * in_channels as usize + c,
                        FeatureLayout::Surfaces => {
                            surface * width * height * FEATURE_ATOM_BYTES
                                + (y * width + x) * FEATURE_ATOM_BYTES
                                + lane
                        }
                    };
                    ptr::write(buf_input.host_ptr.add(offset), 1u8);
                }
            }
        }

        let weight_bytes = shape.weight_bytes(kernels) as usize;
        let buf_weights = Buffer::new(fd, page_aligned_size(weight_bytes), &file);
        ptr::write_bytes(buf_weights.host_ptr, 0, buf_weights.size);
        ptr::write_bytes(buf_weights.host_ptr, 1, weight_bytes);

        let bs_bytes = bs_buffer_bytes(out_channels);
        let buf_bs = Buffer::new(fd, page_aligned_size(bs_bytes), &file);
        ptr::write_bytes(buf_bs.host_ptr, 0, buf_bs.size);
        let entries = vec![
            BsEntry {
                bias: 0,
                multiplier: bs_multiplier,
            };
            out_channels as usize
        ];
        write_bs_buffer(
            std::slice::from_raw_parts_mut(buf_bs.host_ptr, buf_bs.size),
            &entries,
        );

        let buf_output = Buffer::new(fd, page_aligned_size(output_bytes), &file);
        ptr::write_bytes(buf_output.host_ptr, 0, buf_output.size);

        let mut commands = conv_2d_tile(shape, kernels, &Tile::whole(shape, kernels));
        relocate::<CnaFeatureDataAddr>(&mut commands, buf_input.dma_address);
        relocate::<CnaDcompAddr0>(&mut commands, buf_weights.dma_address);
        relocate::<DpuRdmaBsBaseAddr>(&mut commands, buf_bs.dma_address);
        relocate::<DpuDstBaseAddr>(&mut commands, buf_output.dma_address);

        let command_bytes = commands.len() * mem::size_of::<u64>();
        let buf_commands = Buffer::new(fd, page_aligned_size(command_bytes), &file);
        ptr::write_bytes(buf_commands.host_ptr, 0, buf_commands.size);
        let words =
            std::slice::from_raw_parts_mut(buf_commands.host_ptr as *mut u64, commands.len());
        for (destination, command) in words.iter_mut().zip(&commands) {
            *destination = command.0;
        }

        for handle in [
            buf_input.handle,
            buf_weights.handle,
            buf_bs.handle,
            buf_output.handle,
            buf_commands.handle,
        ] {
            fini_bo(fd, handle).expect("failed to sync BO for the NPU");
        }

        let tasks = [(buf_commands.dma_address, commands.len() as u32)];
        let in_handles = [
            buf_commands.handle,
            buf_input.handle,
            buf_weights.handle,
            buf_bs.handle,
        ];
        let out_handles = [buf_output.handle];
        let jobs = [JobDesc {
            tasks: &tasks,
            in_handles: &in_handles,
            out_handles: &out_handles,
        }];
        submit_jobs(fd, &jobs).expect("SUBMIT failed");
        prep_bo(fd, buf_output.handle, 5_000_000_000).expect("job did not complete");

        let raw = std::slice::from_raw_parts(buf_output.host_ptr as *const i8, output_bytes);
        let surface = channel / 16;
        let lane = channel % 16;
        let values: Vec<i32> = (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    surface * width * height * FEATURE_ATOM_BYTES
                        + (y * width + x) * FEATURE_ATOM_BYTES
                        + lane
                })
            })
            .map(|offset| i32::from(raw[offset]))
            .collect();

        for handle in [
            buf_input.handle,
            buf_weights.handle,
            buf_bs.handle,
            buf_output.handle,
            buf_commands.handle,
        ] {
            let _ = close_bo(fd, handle);
        }
        values
    }
}

fn taps(coordinate: usize, extent: usize, kernel: usize) -> usize {
    match kernel {
        1 => 1,
        3 => 3 - usize::from(coordinate == 0) - usize::from(coordinate + 1 == extent),
        _ => unreachable!(),
    }
}

fn show(label: &str, in_channels: u32, out_channels: u32, kernel: usize, channel: usize) {
    let got = map(in_channels, out_channels, kernel, channel);
    let width = WIDTH as usize;
    let height = HEIGHT as usize;
    println!("{label}: Cin {in_channels} Cout {out_channels} {kernel}x{kernel} channel {channel}");
    println!("  got / expected, and the difference:");
    for y in 0..height {
        let mut got_row = String::new();
        let mut want_row = String::new();
        let mut diff_row = String::new();
        for x in 0..width {
            let want =
                (in_channels as usize * taps(y, height, kernel) * taps(x, width, kernel)) as i32;
            let value = got[y * width + x];
            got_row.push_str(&format!("{value:>5}"));
            want_row.push_str(&format!("{want:>5}"));
            diff_row.push_str(&format!("{:>5}", value - want));
        }
        println!("    {got_row}   |{want_row}   |{diff_row}");
    }
    println!();
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn dumps_int8_output_maps() {
    // The two shapes that disagree in `conv_int8_hw`: identical but for the
    // output channel count, one correct and one uniformly one too high.
    show("A", 8, 1, 3, 0);
    show("B", 8, 8, 3, 0);

    // Is it the channel read, or the convolution? If only channel 0 of the
    // Cout 8 case is wrong, the fault is in how output channels are placed.
    show("C", 8, 8, 3, 7);

    // 1x1 at the same Cout isolates the 3x3 tap pattern from everything
    // else about having eight output channels.
    show("D", 8, 8, 1, 0);

    // Dense at Cout 8 with a 3x3: the layout that already passes, as the
    // control for "surfaces is what differs".
    show("E", 4, 8, 3, 0);

    // Halving and doubling the input channels within one atom: if the
    // excess really is `Cin / 8` it should be 2 here and absent at Cin 4.
    show("F", 16, 8, 3, 0);
}

/// Interior output value for a convolution whose accumulator is known.
fn interior(in_channels: u32, kernel: usize) -> i32 {
    let values = map(in_channels, 8, kernel, 0);
    // Row 2, column 2 of an 8x8 map is interior at both kernel sizes.
    values[2 * WIDTH as usize + 2]
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn traces_the_requantisation_transfer_function() {
    // `conv_int8_hw` leaves a uniform +1 on interior pixels in the surface
    // layout with a 3x3 kernel. Five points from the output maps fit
    // `excess = floor((accumulator - 1) / 64)`, but five points and a
    // two-parameter guess is not a derivation, and an accumulator of 64
    // landing on 0 rules out the obvious 65/64 gain.
    //
    // Both sweeps hold the gain at unit, so the printed output *is* the
    // transfer function. The 1x1 sweep sets the accumulator to `Cin`
    // exactly, one step at a time, with none of the 3x3 tap machinery in
    // play; the 3x3 sweep sets it to `9 * Cin`. If 1x1 is the identity
    // across the whole range then the excess belongs to the 3x3 path, and
    // if both show it then it belongs to the output stage.
    println!("unit gain throughout, so output should equal the accumulator\n");

    println!("1x1: accumulator is Cin exactly, no 3x3 taps involved");
    println!(
        "  {:>5}  {:>4}  {:>4}  {:>7}",
        "Cin", "acc", "out", "excess"
    );
    let mut first_1x1 = None;
    for in_channels in 1..=127u32 {
        let got = interior(in_channels, 1);
        let excess = got - in_channels as i32;
        if excess != 0 && first_1x1.is_none() {
            first_1x1 = Some(in_channels);
        }
        // Print sparsely away from the interesting region.
        if in_channels <= 8 || in_channels % 8 == 0 || excess != 0 {
            println!("  {in_channels:>5}  {in_channels:>4}  {got:>4}  {excess:>7}");
        }
    }

    println!("\n3x3: accumulator is 9 * Cin, surfaces from Cin 5 up");
    println!(
        "  {:>5}  {:>4}  {:>4}  {:>7}  {:>9}",
        "Cin", "acc", "out", "excess", "(acc-1)/64"
    );
    for in_channels in 1..=14u32 {
        let accumulator = 9 * in_channels as i32;
        let got = interior(in_channels, 3);
        println!(
            "  {in_channels:>5}  {accumulator:>4}  {got:>4}  {:>7}  {:>9}",
            got - accumulator,
            (accumulator - 1) / 64
        );
    }

    match first_1x1 {
        Some(in_channels) => println!(
            "\n1x1 first departs from the identity at Cin {in_channels}, so the \
             excess is in the output stage, not the 3x3 tap path"
        ),
        None => println!(
            "\n1x1 is the identity across the whole range, so the excess belongs \
             to the 3x3 path and not to the output stage"
        ),
    }
}

/// Interior value at an explicit gain, so the accumulator and the output
/// live in different domains.
fn interior_at(in_channels: u32, kernel: usize, shift: u32) -> i32 {
    map_at(in_channels, 8, kernel, 0, shift)[2 * WIDTH as usize + 2]
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn locates_the_excess_before_or_after_the_requantisation() {
    // Every measurement so far used unit gain, which makes the accumulator
    // and the output the same number and so cannot say which one carries
    // the `+1 per 64` inflation. A shift separates them.
    //
    // If the excess is in the accumulator, it steps every 64 units of
    // `9 * Cin` and the output shows a step every `64 >> shift`.
    // If it is in the output stage, it steps every 64 units of the *output*
    // and so needs `9 * Cin >> shift` to pass 64.
    //
    // Those predictions diverge immediately: at shift 2 the first is a step
    // every 16 output units starting near Cin 8, the second a single step
    // near Cin 29.
    for shift in [1u32, 2, 3] {
        println!("\n3x3 at gain 2^-{shift}: accumulator is 9 * Cin");
        println!(
            "  {:>4}  {:>5}  {:>5}  {:>5}  {:>10}  {:>10}",
            "Cin", "acc", "out", "ideal", "if in acc", "if in out"
        );
        for in_channels in (4..=64u32).step_by(4) {
            let accumulator = 9 * in_channels as i64;
            let ideal = (accumulator + (1 << (shift - 1))) >> shift;
            if ideal > 127 {
                break;
            }
            let got = interior_at(in_channels, 3, shift);
            // Inflation applied to the accumulator, then requantised.
            let in_acc = {
                let inflated = accumulator + (accumulator - 1) / 64;
                (inflated + (1 << (shift - 1))) >> shift
            };
            // Requantised first, then inflated.
            let in_out = ideal + (ideal - 1) / 64;
            println!(
                "  {in_channels:>4}  {accumulator:>5}  {got:>5}  {ideal:>5}  \
                 {in_acc:>10}  {in_out:>10}"
            );
        }
    }
    println!(
        "\nwhichever column tracks `out` is the domain the inflation lives in; \
         if neither does, it is not a simple `+1 per 64` in either."
    );
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn attributes_the_gain_error_to_a_stage() {
    // `out = floor((65 * acc - 1) / 64)` fits every measurement taken with a
    // BS multiplier of 2^14. Several (bs_multiplier, cvt) pairs encode the
    // same nominal unit gain, and they put the 65/64 in different places:
    // if it belongs to the BS multiply it tracks the multiplier, and if it
    // belongs to the output conversion it appears in all of them.
    //
    // The first pair is what the builder programs today. The last is the one
    // the original shift probe used, which could not have seen this because
    // it only ever measured an accumulator of 1.
    const PAIRS: [(i16, u32, u32); 5] = [
        (16384, 16384, 21),
        (8192, 16384, 20),
        (4096, 16384, 19),
        (1024, 16384, 17),
        (128, 16384, 14),
    ];

    println!("1x1, accumulator = Cin, every pair a nominal gain of 1\n");
    println!(
        "  {:>8}  {:>6}  {:>5}  {:>9}  {:>26}",
        "bs_mul", "scale", "shift", "first bad", "out at acc 64 / 65 / 126"
    );
    for (bs_multiplier, scale, shift) in PAIRS {
        let cvt = Multiplier { scale, shift };
        let at = |in_channels: u32| {
            map_with(in_channels, 8, 1, 0, bs_multiplier, cvt)[2 * WIDTH as usize + 2]
        };
        let mut first_bad = None;
        for in_channels in 1..=126u32 {
            if at(in_channels) != in_channels as i32 && first_bad.is_none() {
                first_bad = Some(in_channels);
                break;
            }
        }
        println!(
            "  {bs_multiplier:>8}  {scale:>6}  {shift:>5}  {:>9}  {:>8} {:>8} {:>8}",
            first_bad.map_or("none".to_string(), |c| c.to_string()),
            at(64),
            at(65),
            at(126)
        );
    }
    println!(
        "\na `first bad` of 65 everywhere puts the 65/64 in the output stage; \
         only at bs_mul 16384 puts it in the BS multiply."
    );
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn checks_the_output_stage_compensation() {
    // The output conversion realises 65/64 of the gain it is asked for --
    // measured across five (bs_multiplier, cvt) pairs that all encode unit
    // gain, every one of which inflates identically. Its multiplier is ours
    // to choose, so the compensation is to ask for 64/65 of what is wanted.
    //
    // This is a correction, not an explanation: nothing so far says *why*
    // the stage does this, and the factor is applied because it is measured
    // rather than because it is understood. If the transfer function below
    // is the identity across the whole int8 range, the correction holds over
    // every accumulator the output can represent, which is the most that can
    // be claimed without a mechanism.
    const COMPENSATION: f64 = 64.0 / 65.0;

    for (label, factor) in [("uncompensated", 1.0), ("compensated", COMPENSATION)] {
        let cvt = Multiplier::for_unit_bs(factor);
        let at = |in_channels: u32| {
            map_with(in_channels, 8, 1, 0, BS_UNIT_MULTIPLIER, cvt)[2 * WIDTH as usize + 2]
        };
        let mut bad = Vec::new();
        for in_channels in 1..=126u32 {
            let got = at(in_channels);
            if got != in_channels as i32 {
                bad.push((in_channels, got));
            }
        }
        println!(
            "{label:>14}: cvt = {}/2^{}, {} of 126 accumulators wrong",
            cvt.scale,
            cvt.shift,
            bad.len()
        );
        for (accumulator, got) in bad.iter().take(6) {
            println!("                  acc {accumulator} -> {got}");
        }
        if bad.len() > 6 {
            println!("                  ... and {} more", bad.len() - 6);
        }
    }

    // If the correction holds it should also hold at 3x3, where the
    // accumulator is nine times larger for the same channel count and the
    // surface layout is in play.
    println!("\n3x3 with the compensation applied, accumulator = 9 * Cin:");
    let cvt = Multiplier::for_unit_bs(COMPENSATION);
    let mut wrong = 0;
    for in_channels in 1..=14u32 {
        let accumulator = 9 * in_channels as i32;
        let got = map_with(in_channels, 8, 3, 0, BS_UNIT_MULTIPLIER, cvt)[2 * WIDTH as usize + 2];
        if got != accumulator {
            wrong += 1;
            println!("  Cin {in_channels:>3}  acc {accumulator:>4}  out {got:>4}");
        }
    }
    println!("  {wrong} of 14 wrong");
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn measures_the_output_scale_directly() {
    // Compensating by a ratio overshot, so the excess is not multiplicative.
    // Two configurations both imply an effective mantissa about 256 above
    // the programmed one, which would look like +1/64 at a scale of 2^14 and
    // +1/128 at 2^15 -- exactly the two ratios measured, and exactly why a
    // single correction factor could not fit both.
    //
    // An additive constant is a different kind of bug from a gain: it points
    // at a field, not at an arithmetic misunderstanding. So sweep the scale
    // itself, hold everything else fixed, and read the effective value back
    // out of the result. The accumulator is held at 120, large enough that
    // one unit of output resolves about 137 units of mantissa.
    const ACCUMULATOR: u32 = 120;
    const SHIFT: u32 = 21;

    println!("Cin {ACCUMULATOR} at 1x1, bs_mul 2^14, cvt_shift {SHIFT}");
    println!("out should be acc * scale / 2^14, so scale 16384 means out 120\n");
    println!(
        "  {:>7}  {:>8}  {:>5}  {:>10}  {:>8}",
        "scale", "ideal", "out", "implied", "diff"
    );
    for scale in (2048..=17408u32).step_by(512) {
        let cvt = Multiplier {
            scale,
            shift: SHIFT,
        };
        let out = map_with(ACCUMULATOR, 8, 1, 0, BS_UNIT_MULTIPLIER, cvt)[2 * WIDTH as usize + 2];
        let ideal = f64::from(ACCUMULATOR) * f64::from(scale) / 16384.0;
        // What mantissa would have produced this output exactly.
        let implied = f64::from(out) * 16384.0 / f64::from(ACCUMULATOR);
        println!(
            "  {scale:>7}  {ideal:>8.2}  {out:>5}  {implied:>10.0}  {:>8.0}",
            implied - f64::from(scale)
        );
    }
    println!(
        "\na roughly constant `diff` is an additive offset on the mantissa; \
         a diff proportional to `scale` is a gain after all."
    );
}
