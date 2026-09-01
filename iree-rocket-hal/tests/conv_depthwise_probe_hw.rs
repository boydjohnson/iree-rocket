//! Hardware probe for the one thing the depthwise capture sweep could not
//! reach: how the coefficient bytes are arranged in the weight buffer.
//!
//! The sweep pinned the register programming -- mode bits, channel padding,
//! and the `CNA_WEIGHT_SIZE0.weight_bytes` footprint -- but a capture only
//! contains the register program, never the buffer it points at. A depthwise
//! filter is `[Cin, 1, kh, kw]`, not the `[kh, kw, Cin, Cout]` that
//! `tensor_layout::pack_hwcf_to_rocket_weights` produces.
//!
//! # Result
//!
//! The Cin 8 run mapped all 72 slots with none silent and none ambiguous:
//! the layout is **tap-major**, `slot = (ky * kw + kx) * stride + channel`.
//! Its warm-up also peaked at 9 on all eight channels, which is what first
//! said the depthwise register programming is right on real silicon.
//!
//! What that run could not settle is the row `stride`: at Cin 8 the real and
//! padded channel counts are both 8. `depthwise_weight_layout_probe_with_channel_padding`
//! is the case that separates them.

#![cfg(feature = "hardware-characterization")]
//!
//! # Method
//!
//! One-hot the weight buffer. At Cin 8 with a 3x3 kernel the buffer is
//! exactly 144 bytes -- 72 fp16 slots -- so every slot can be probed
//! individually. For each slot the buffer is zeroed except that one element,
//! set to 1.0, and the input is 1.0 everywhere.
//!
//! A single live coefficient at kernel tap `(ky, kx)` of channel `c` makes
//! `output[c][y][x] = input[c][y + ky - 1][x + kx - 1]`, which is 1.0 except
//! where the tap reaches outside the image. So:
//!
//! - *which* `(surface, lane)` responds identifies the output channel, and
//! - *which borders read zero* identifies the tap: a zero first row means
//!   `ky == 0`, a zero last row means `ky == 2`, neither means `ky == 1`.
//!   Columns give `kx` the same way.
//!
//! That is a complete `slot -> (channel, ky, kx)` mapping, which is the
//! layout.
//!
//! # Reading the output
//!
//! Run with `--nocapture`; the probe reports a table and then a summary
//! naming the layout if it recognises one. It asserts only that the mapping
//! is *total and unambiguous* -- every slot accounted for, no slot driving
//! two channels -- not that it takes any particular form. Guessing the
//! answer in an assertion is what this test exists to avoid.
//!
//! # The warm-up
//!
//! Each probe fires one all-ones job before the sweep, so that a hang or a
//! dead output is attributed to the register programming rather than to the
//! layout sweep. This mattered on the first run, when nothing had dispatched
//! a depthwise program on silicon yet; it stays as the same guard for every
//! new shape the probe is pointed at.

use std::{collections::BTreeMap, fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    conv::{Buffers, Kernels, Shape, Tile, conv_2d_tile, relocate},
    device::{Buffer, JobDesc, close_bo, fini_bo, prep_bo, submit_jobs},
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const FP16_BYTES: usize = 2;
const FEATURE_ATOM_BYTES: usize = 16;
const CHANNELS_PER_ATOM: usize = 8;
const PAGE_BYTES: usize = 4096;
const FP16_ONE: u16 = 0x3c00;

const EXTENT: usize = 32;
const CIN: u32 = 8;
const KERNEL: Kernels = [3, 3];

fn page_aligned_size(size: usize) -> usize {
    size.div_ceil(PAGE_BYTES) * PAGE_BYTES
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

/// What one output surface-lane looked like for a single-coefficient run.
struct Response {
    surface: usize,
    lane: usize,
    peak: f32,
    row0_zero: bool,
    last_row_zero: bool,
    col0_zero: bool,
    last_col_zero: bool,
}

impl Response {
    /// The tap the border pattern implies, or `None` if the pattern is not
    /// one a single tap can produce.
    fn tap(&self) -> Option<(usize, usize)> {
        let axis = |low: bool, high: bool| match (low, high) {
            (true, false) => Some(0),
            (false, false) => Some(1),
            (false, true) => Some(2),
            (true, true) => None,
        };
        Some((
            axis(self.row0_zero, self.last_row_zero)?,
            axis(self.col0_zero, self.last_col_zero)?,
        ))
    }

    fn channel(&self) -> usize {
        self.surface * CHANNELS_PER_ATOM + self.lane
    }
}

struct Probe {
    file: std::fs::File,
    fd: i32,
    input: Buffer,
    weights: Buffer,
    bias: Buffer,
    output: Buffer,
    commands: Buffer,
    command_count: u32,
    weight_slots: usize,
    out_surfaces: usize,
}

impl Probe {
    unsafe fn new(shape: Shape) -> Probe {
        unsafe {
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(DEVICE_PATH)
                .expect("failed to open RK3588 NPU device");
            let fd = file.as_raw_fd();

            let in_surfaces = (shape.weight_channels() as usize).div_ceil(CHANNELS_PER_ATOM);
            let input_bytes = in_surfaces * EXTENT * EXTENT * FEATURE_ATOM_BYTES;
            let input = Buffer::new(fd, page_aligned_size(input_bytes), &file);
            ptr::write_bytes(input.host_ptr, 0, input.size);
            // 1.0 in every real lane of every pixel. Padding lanes stay zero
            // so they cannot contribute whatever the layout turns out to be.
            for surface in 0..in_surfaces {
                for pixel in 0..EXTENT * EXTENT {
                    for lane in 0..CHANNELS_PER_ATOM {
                        if surface * CHANNELS_PER_ATOM + lane >= shape.in_channels as usize {
                            continue;
                        }
                        let offset = surface * EXTENT * EXTENT * FEATURE_ATOM_BYTES
                            + pixel * FEATURE_ATOM_BYTES
                            + lane * FP16_BYTES;
                        ptr::write(input.host_ptr.add(offset) as *mut u16, FP16_ONE);
                    }
                }
            }

            let weight_bytes = shape.weight_bytes(KERNEL) as usize;
            let weights = Buffer::new(fd, page_aligned_size(weight_bytes), &file);
            let bias = Buffer::new(fd, PAGE_BYTES, &file);
            ptr::write_bytes(bias.host_ptr, 0, bias.size);

            let out_surfaces = (shape.padded_out_channels() as usize).div_ceil(CHANNELS_PER_ATOM);
            let output_bytes = out_surfaces * EXTENT * EXTENT * FEATURE_ATOM_BYTES;
            let output = Buffer::new(fd, page_aligned_size(output_bytes), &file);

            let mut words = conv_2d_tile(shape, KERNEL, &Tile::whole(shape, KERNEL));
            relocate(
                &mut words,
                Buffers {
                    input: input.dma_address,
                    weights: weights.dma_address,
                    bias: bias.dma_address,
                    output: output.dma_address,
                },
            );
            let commands = Buffer::new(
                fd,
                page_aligned_size(words.len() * mem::size_of::<u64>()),
                &file,
            );
            ptr::write_bytes(commands.host_ptr, 0, commands.size);
            let slots = std::slice::from_raw_parts_mut(commands.host_ptr as *mut u64, words.len());
            for (destination, command) in slots.iter_mut().zip(&words) {
                *destination = command.0;
            }
            for handle in [input.handle, bias.handle, commands.handle] {
                fini_bo(fd, handle).expect("failed to sync BO for the NPU");
            }

            Probe {
                file,
                fd,
                input,
                weights,
                bias,
                output,
                commands,
                command_count: words.len() as u32,
                weight_slots: weight_bytes / FP16_BYTES,
                out_surfaces,
            }
        }
    }

    /// Loads `coefficients` into the weight buffer, runs one job, and
    /// returns every output surface-lane that came back non-zero.
    unsafe fn run(&self, coefficients: &[u16], label: &str) -> Vec<Response> {
        unsafe {
            ptr::write_bytes(self.weights.host_ptr, 0, self.weights.size);
            let destination = std::slice::from_raw_parts_mut(
                self.weights.host_ptr as *mut u16,
                self.weight_slots,
            );
            destination.copy_from_slice(coefficients);
            ptr::write_bytes(self.output.host_ptr, 0, self.output.size);
            fini_bo(self.fd, self.weights.handle).expect("failed to sync weights");
            fini_bo(self.fd, self.output.handle).expect("failed to sync output");

            let tasks = [(self.commands.dma_address, self.command_count)];
            let in_handles = [
                self.commands.handle,
                self.input.handle,
                self.weights.handle,
                self.bias.handle,
            ];
            let out_handles = [self.output.handle];
            submit_jobs(
                self.fd,
                &[JobDesc {
                    tasks: &tasks,
                    in_handles: &in_handles,
                    out_handles: &out_handles,
                }],
            )
            .unwrap_or_else(|error| panic!("{label}: SUBMIT failed: {error}"));
            prep_bo(self.fd, self.output.handle, 5_000_000_000).unwrap_or_else(|error| {
                panic!(
                    "{label}: job did not complete: {error}. A depthwise program \
                     that never retires points at the register programming, not \
                     at the weight layout this probe is after."
                )
            });

            let raw = std::slice::from_raw_parts(
                self.output.host_ptr,
                self.out_surfaces * EXTENT * EXTENT * FEATURE_ATOM_BYTES,
            );
            let at = |surface: usize, lane: usize, y: usize, x: usize| -> f32 {
                let offset = surface * EXTENT * EXTENT * FEATURE_ATOM_BYTES
                    + (y * EXTENT + x) * FEATURE_ATOM_BYTES
                    + lane * FP16_BYTES;
                f16_to_f32(u16::from_le_bytes([raw[offset], raw[offset + 1]]))
            };

            let mut responses = Vec::new();
            for surface in 0..self.out_surfaces {
                for lane in 0..CHANNELS_PER_ATOM {
                    let mut peak: f32 = 0.0;
                    for y in 0..EXTENT {
                        for x in 0..EXTENT {
                            peak = peak.max(at(surface, lane, y, x).abs());
                        }
                    }
                    if peak == 0.0 {
                        continue;
                    }
                    let row = |y: usize| (0..EXTENT).all(|x| at(surface, lane, y, x) == 0.0);
                    let col = |x: usize| (0..EXTENT).all(|y| at(surface, lane, y, x) == 0.0);
                    responses.push(Response {
                        surface,
                        lane,
                        peak,
                        row0_zero: row(0),
                        last_row_zero: row(EXTENT - 1),
                        col0_zero: col(0),
                        last_col_zero: col(EXTENT - 1),
                    });
                }
            }
            responses
        }
    }
}

impl Drop for Probe {
    fn drop(&mut self) {
        for handle in [
            self.input.handle,
            self.weights.handle,
            self.bias.handle,
            self.output.handle,
            self.commands.handle,
        ] {
            // Safe here: these handles were opened by `Probe::new` against
            // `self.fd`, which the still-live `self.file` keeps open, and
            // nothing else closes them.
            let _ = unsafe { close_bo(self.fd, handle) };
        }
        let _ = &self.file;
    }
}

fn probe_layout(cin: u32) {
    let shape =
        Shape::with_out_channels(EXTENT as u32, EXTENT as u32, 1, cin, cin).with_depthwise();

    unsafe {
        let probe = Probe::new(shape);
        println!(
            "\ndepthwise probe: {}x{} Cin/Cout {cin} {KERNEL:?} fp16, \
             {} weight slots, {} output surfaces\n",
            EXTENT, EXTENT, probe.weight_slots, probe.out_surfaces
        );

        // Warm-up: every coefficient live. This says whether a depthwise
        // program dispatches and writes anything at all, before any
        // conclusion is drawn from a single-coefficient run.
        let all_ones = vec![FP16_ONE; probe.weight_slots];
        let warm = probe.run(&all_ones, "all-ones warm-up");
        println!("all-ones warm-up: {} responding channel(s)", warm.len());
        for response in &warm {
            println!(
                "  surface {} lane {} (channel {:2}) peak {}",
                response.surface,
                response.lane,
                response.channel(),
                response.peak
            );
        }
        assert!(
            !warm.is_empty(),
            "a depthwise program with every coefficient set produced an \
             all-zero output: the register programming is wrong, and no \
             weight layout can be read until that is fixed"
        );
        println!();

        // One coefficient at a time.
        let mut mapping: BTreeMap<usize, (usize, usize, usize)> = BTreeMap::new();
        let mut silent = Vec::new();
        let mut ambiguous = Vec::new();
        for slot in 0..probe.weight_slots {
            let mut coefficients = vec![0u16; probe.weight_slots];
            coefficients[slot] = FP16_ONE;
            let responses = probe.run(&coefficients, &format!("slot {slot}"));

            match responses.as_slice() {
                [] => {
                    silent.push(slot);
                    println!("slot {slot:3}: no response");
                }
                [response] => match response.tap() {
                    Some((ky, kx)) => {
                        println!(
                            "slot {slot:3}: channel {:2} (surface {} lane {}) tap ({ky}, {kx}) peak {}",
                            response.channel(),
                            response.surface,
                            response.lane,
                            response.peak
                        );
                        mapping.insert(slot, (response.channel(), ky, kx));
                    }
                    None => {
                        println!(
                            "slot {slot:3}: channel {:2} but border pattern is not a single tap \
                             (rows {}/{}, cols {}/{})",
                            response.channel(),
                            response.row0_zero,
                            response.last_row_zero,
                            response.col0_zero,
                            response.last_col_zero
                        );
                        ambiguous.push(slot);
                    }
                },
                many => {
                    println!(
                        "slot {slot:3}: {} channels responded: {:?}",
                        many.len(),
                        many.iter().map(Response::channel).collect::<Vec<_>>()
                    );
                    ambiguous.push(slot);
                }
            }
        }

        println!("\n--- summary ---");
        println!(
            "{} of {} slots mapped, {} silent, {} ambiguous",
            mapping.len(),
            probe.weight_slots,
            silent.len(),
            ambiguous.len()
        );

        // Per-channel slot order, which is what a packer needs.
        let mut by_channel: BTreeMap<usize, Vec<(usize, usize, usize)>> = BTreeMap::new();
        for (slot, (channel, ky, kx)) in &mapping {
            by_channel
                .entry(*channel)
                .or_default()
                .push((*slot, *ky, *kx));
        }
        for (channel, mut entries) in by_channel {
            entries.sort();
            let taps: Vec<String> = entries
                .iter()
                .map(|(slot, ky, kx)| format!("{slot}->({ky},{kx})"))
                .collect();
            println!("  channel {channel:2}: {}", taps.join(" "));
        }

        // Name the layouts worth recognising, without assuming any of them.
        //
        // The two tap-major candidates differ only in the row stride: the
        // real channel count against the padded one the weight footprint is
        // sized from. At Cin 8 those coincide, which is why the Cin 12 case
        // exists -- there the padded count is 16 and the two disagree.
        let padded = probe.weight_slots / (KERNEL[0] * KERNEL[1]);
        let channel_major = mapping.iter().all(|(slot, (channel, ky, kx))| {
            *slot == channel * KERNEL[0] * KERNEL[1] + ky * KERNEL[1] + kx
        });
        let tap_major_real = mapping.iter().all(|(slot, (channel, ky, kx))| {
            *slot == (ky * KERNEL[1] + kx) * cin as usize + channel
        });
        let tap_major_padded = mapping
            .iter()
            .all(|(slot, (channel, ky, kx))| *slot == (ky * KERNEL[1] + kx) * padded + channel);
        let verdict = |matched: bool| if matched { "MATCHES" } else { "no" };
        println!(
            "\nchannel-major [Cin][kh][kw]:            {}\n\
             tap-major     [kh][kw][Cin={cin}]:        {}\n\
             tap-major     [kh][kw][padded={padded}]:   {}",
            verdict(channel_major),
            verdict(tap_major_real),
            verdict(tap_major_padded)
        );
        if cin as usize == padded {
            println!(
                "(Cin and the padded count coincide here, so the last two \
                 cannot be told apart at this shape)"
            );
        }
        if !channel_major && !tap_major_real && !tap_major_padded {
            println!(
                "none -- read the per-channel table above; the slot order \
                 within each channel is the packing rule"
            );
        }

        assert!(
            ambiguous.is_empty(),
            "{} slot(s) drove more than one channel or produced a border \
             pattern no single tap explains: {ambiguous:?}. The one-hot \
             assumption does not hold and the table above cannot be read as \
             a layout.",
            ambiguous.len()
        );
        assert_eq!(
            mapping.len() + silent.len(),
            probe.weight_slots,
            "slot accounting is inconsistent"
        );
        assert!(
            !mapping.is_empty(),
            "no single coefficient produced any output, though the all-ones \
             warm-up did: the buffer is being read, but not one element at a \
             time the way this probe assumes"
        );
    }
}

/// The original probe. Cin 8 is one whole fp16 atom, so the buffer holds
/// exactly 72 slots and every one of them is a real channel.
#[test]
#[ignore = "requires a real RK3588 NPU at /dev/accel/accel0; run with --nocapture"]
fn depthwise_weight_layout_probe() {
    probe_layout(CIN);
}

/// The disambiguating probe.
///
/// Cin 8 leaves one question open: its real and padded channel counts are
/// both 8, so a row stride of `Cin` and a row stride of `padded` are the
/// same number and the first probe cannot separate them. Cin 12 pads to 16,
/// so the two predict different slots for every tap after the first -- tap
/// (0,1) starts at slot 12 under one rule and slot 16 under the other.
///
/// 144 slots, so this takes twice as long as the Cin 8 run.
#[test]
#[ignore = "requires a real RK3588 NPU at /dev/accel/accel0; run with --nocapture"]
fn depthwise_weight_layout_probe_with_channel_padding() {
    probe_layout(12);
}
