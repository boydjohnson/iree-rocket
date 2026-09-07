#![cfg(feature = "hardware-characterization")]

//! Hardware characterization: classifies the int8 requantisation **tie rule**
//! on RK3588 instead of assuming it.
//!
//!   CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!     cargo test --target aarch64-unknown-linux-gnu --release \
//!       --features hardware-characterization \
//!       --test conv_requant_tie_rule_hw --no-run
//!
//!   ./conv_requant_tie_rule_hw-<hash> --ignored --nocapture
//!
//! # Why
//!
//! `DPU_OUT_CVT` computes `(accumulator * SCALE) >> SHIFT`, and three
//! different tie rules are asserted in three places in this project:
//!
//! - `tests/support/conv2d_oracle.rs`'s `rounded_shift` models
//!   **round-half-away-from-zero**;
//! - `tests/conv_kernel_shape_hw.rs` models `(acc + half) >> shift`, which is
//!   **round-half-up**, under a comment claiming half-away-from-zero
//!   "measured by `conv_int8_probe_hw`" -- that probe measured the BS *gain*
//!   and never classified a tie, and its `Int8` tolerance of 1.0 is exactly
//!   wide enough to hide the disagreement;
//! - `builders/dpu.rs`'s `DPU_OUT_CVT_SHIFT.cvt_round` documents
//!   `0 = if the integer is odd, carry 1`, i.e. **round-half-to-even**, and
//!   `conv.rs` builds that register without ever setting the field, so every
//!   convolution this project has ever run asked for half-to-even.
//!   `../rockchip-npu-notes/encodings/out-cvt-converter.md` measured
//!   half-to-even on RK3576 and explicitly records RK3588 as *predicted*,
//!   never probed.
//!
//! The three rules agree everywhere except on an exact tie, so no existing
//! test separates them. This one drives the accumulator onto exact ties at
//! two shifts and both signs and classifies the result.
//!
//! # Method
//!
//! A 1x1 convolution at `Cin` 1 makes the accumulator at each output pixel
//! exactly that pixel's input byte, so one 16x16 job sweeps every `i8`
//! accumulator from -128 to 127 in a single dispatch. All coefficients are 1
//! and the BS plane is `BsEntry::default()` (unit multiplier, zero bias),
//! whose `2^14` multiplier and `>> BS_MULTIPLIER_SHIFT` are exact -- so the
//! only rounding in the datapath is the one under test.
//! `Multiplier::for_unit_bs(1 / 2^s)` cancels the BS gain, leaving the
//! hardware computing `accumulator >> s` under its own tie rule.
//!
//! **The probe validates itself.** Off a tie all three rules agree, so any
//! non-tie accumulator that does not come back as `accumulator >> s` means
//! the gain or the layout is wrong and the tie classification below is
//! meaningless. That is checked and reported before anything is concluded.

#[path = "support/conv2d_oracle.rs"]
mod conv2d_oracle;

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use conv2d_oracle::{feature_offset, input_storage_bytes, output_offset, output_storage_bytes};

use iree_rocket_hal::rocket::{
    builders::{RegCmd, RegisterMeta, dpu::DpuOutCvtShift},
    conv::{
        BsEntry, Buffers, Kernels, Multiplier, Precision, Quantization, Shape, Tile, conv_2d_tile,
        relocate, write_bs_buffer,
    },
    device::{Buffer, JobDesc, close_bo, fini_bo, prep_bo, submit_jobs},
    tensor_layout::pack_hwcf_to_rocket_weights,
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const PAGE_BYTES: usize = 4096;
const WIDTH: u32 = 16;
const HEIGHT: u32 = 16;
const IN_CHANNELS: u32 = 1;
const OUT_CHANNELS: u32 = 16;
const KERNELS: Kernels = [1, 1];

fn page_aligned_size(size: usize) -> usize {
    size.div_ceil(PAGE_BYTES) * PAGE_BYTES
}

/// The three candidate rules for `accumulator >> shift`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rule {
    HalfUp,
    HalfAwayFromZero,
    HalfToEven,
}

impl Rule {
    const ALL: [Rule; 3] = [Rule::HalfUp, Rule::HalfAwayFromZero, Rule::HalfToEven];

    fn name(self) -> &'static str {
        match self {
            Rule::HalfUp => "round-half-up",
            Rule::HalfAwayFromZero => "round-half-away-from-zero",
            Rule::HalfToEven => "round-half-to-even",
        }
    }

    fn apply(self, value: i32, shift: u32) -> i32 {
        if shift == 0 {
            return value;
        }
        let half = 1i32 << (shift - 1);
        match self {
            Rule::HalfUp => (value + half) >> shift,
            Rule::HalfAwayFromZero => {
                if value >= 0 {
                    (value + half) >> shift
                } else {
                    -((-value + half) >> shift)
                }
            }
            Rule::HalfToEven => {
                let quotient = value >> shift;
                let remainder = value & ((1 << shift) - 1);
                match remainder.cmp(&half) {
                    std::cmp::Ordering::Less => quotient,
                    std::cmp::Ordering::Greater => quotient + 1,
                    std::cmp::Ordering::Equal => quotient + (quotient & 1),
                }
            }
        }
    }
}

/// Sets `DPU_OUT_CVT_SHIFT.cvt_round`, which `conv_2d_tile` always leaves
/// clear.
///
/// The field is bit 30 of the register and the command word carries the
/// value at bits 16..48, so the bit to set is 46. `0` is documented as
/// "if the integer is odd, carry 1" and `1` as "carry 1 no matter what",
/// i.e. half-to-even and half-up -- neither of which is what the shipped
/// `0` actually does, which is the point of measuring both states.
fn set_cvt_round(commands: &mut [RegCmd]) {
    let mut patched = 0;
    for command in commands.iter_mut() {
        if (command.0 >> 48) as u32 == DpuOutCvtShift::DOMAIN
            && (command.0 as u32 & 0xffff) == DpuOutCvtShift::OFFSET
        {
            command.0 |= 1u64 << 46;
            patched += 1;
        }
    }
    assert_eq!(patched, 1, "expected exactly one DPU_OUT_CVT_SHIFT write");
}

/// True when `value >> shift` lands on an exact half.
fn is_tie(value: i32, shift: u32) -> bool {
    shift > 0 && (value & ((1 << shift) - 1)) == 1 << (shift - 1)
}

/// Runs one 16x16 job whose accumulator at pixel `p` is `accumulators[p]`,
/// returning the requantised `i8` the hardware wrote at channel 0.
///
/// `shift` is the *logical* right shift asked of the datapath;
/// `Multiplier::for_unit_bs` turns it into the `SCALE`/`SHIFT` pair that
/// cancels the BS stage's gain.
fn run(shift: u32, cvt_round: bool, accumulators: &[i32]) -> Vec<i32> {
    let precision = Precision::Int8(Quantization {
        input_zero_point: 0,
        output_zero_point: 0,
        weight_zero_point: 0,
        input_scale: 1.0,
        weights_scale: 1.0,
        multiplier: Multiplier::for_unit_bs(1.0 / f64::from(1u32 << shift)),
    });
    let shape = Shape::with_precision(WIDTH, HEIGHT, 1, IN_CHANNELS, OUT_CHANNELS, precision);
    let width = WIDTH as usize;
    let height = HEIGHT as usize;
    let pixels = width * height;
    assert_eq!(
        accumulators.len(),
        pixels,
        "the sweep must name one accumulator per pixel"
    );

    let input_bytes = input_storage_bytes(shape);
    let output_bytes = output_storage_bytes(shape, KERNELS);

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let buf_input = Buffer::new(fd, page_aligned_size(input_bytes), &file);
        ptr::write_bytes(buf_input.host_ptr, 0, buf_input.size);
        let input = std::slice::from_raw_parts_mut(buf_input.host_ptr, buf_input.size);
        for (pixel, &accumulator) in accumulators.iter().enumerate() {
            let value = i8::try_from(accumulator).expect("accumulator must fit in one int8 input");
            input[feature_offset(shape, 0, pixel / width, pixel % width)] = value as u8;
        }

        // Every coefficient is 1, so the accumulator is the input byte.
        let weight_bytes = shape.weight_bytes(KERNELS) as usize;
        let buf_weights = Buffer::new(fd, page_aligned_size(weight_bytes), &file);
        ptr::write_bytes(buf_weights.host_ptr, 0, buf_weights.size);
        let dense_weights = vec![1u8; (IN_CHANNELS * OUT_CHANNELS) as usize];
        pack_hwcf_to_rocket_weights(
            &dense_weights,
            KERNELS[0] as usize,
            KERNELS[1] as usize,
            IN_CHANNELS as usize,
            OUT_CHANNELS as usize,
            1,
            std::slice::from_raw_parts_mut(buf_weights.host_ptr, weight_bytes),
        )
        .expect("failed to pack coefficients");

        let bs_bytes = shape.bs_buffer_bytes();
        let buf_bs = Buffer::new(fd, page_aligned_size(bs_bytes), &file);
        ptr::write_bytes(buf_bs.host_ptr, 0, buf_bs.size);
        let entries = vec![BsEntry::default(); shape.padded_out_channels() as usize];
        write_bs_buffer(
            std::slice::from_raw_parts_mut(buf_bs.host_ptr, buf_bs.size),
            &entries,
        );

        let buf_output = Buffer::new(fd, page_aligned_size(output_bytes), &file);
        ptr::write_bytes(buf_output.host_ptr, 0, buf_output.size);

        let mut commands = conv_2d_tile(shape, KERNELS, &Tile::whole(shape, KERNELS));
        if cvt_round {
            set_cvt_round(&mut commands);
        }
        relocate(
            &mut commands,
            Buffers {
                input: buf_input.dma_address,
                weights: buf_weights.dma_address,
                bias: buf_bs.dma_address,
                output: buf_output.dma_address,
            },
        );

        let command_bytes = commands.len() * mem::size_of::<u64>();
        let buf_commands = Buffer::new(fd, page_aligned_size(command_bytes), &file);
        ptr::write_bytes(buf_commands.host_ptr, 0, buf_commands.size);
        let words =
            std::slice::from_raw_parts_mut(buf_commands.host_ptr as *mut u64, commands.len());
        for (destination, command) in words.iter_mut().zip(&commands) {
            *destination = command.0;
        }

        let handles = [
            buf_input.handle,
            buf_weights.handle,
            buf_bs.handle,
            buf_output.handle,
            buf_commands.handle,
        ];
        for handle in handles {
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
        let observed = (0..pixels)
            .map(|pixel| {
                i32::from(raw[output_offset(shape, KERNELS, 0, pixel / width, pixel % width)])
            })
            .collect::<Vec<_>>();

        for handle in handles {
            let _ = close_bo(fd, handle);
        }
        observed
    }
}

/// Classifies one shift, returning the rules that explain every observation.
fn classify(shift: u32, cvt_round: bool) -> Vec<Rule> {
    // Every int8 accumulator, so both signs and both tie parities are swept.
    let accumulators = (-128..128).collect::<Vec<i32>>();
    let observed = run(shift, cvt_round, &accumulators);

    // Self-check first: off a tie the three rules agree, so a non-tie
    // disagreement means the gain or the layout is wrong and nothing below
    // is worth reading.
    let mut off_tie_wrong = Vec::new();
    for (&accumulator, &got) in accumulators.iter().zip(&observed) {
        if is_tie(accumulator, shift) {
            continue;
        }
        let want = Rule::HalfToEven.apply(accumulator, shift);
        if got != want {
            off_tie_wrong.push((accumulator, want, got));
        }
    }
    println!(
        "\n=== logical shift {shift}, cvt_round = {} ===",
        u32::from(cvt_round)
    );
    println!(
        "  {} of {} non-tie accumulators match `accumulator >> {shift}`",
        accumulators.len()
            - accumulators.iter().filter(|a| is_tie(**a, shift)).count()
            - off_tie_wrong.len(),
        accumulators.len() - accumulators.iter().filter(|a| is_tie(**a, shift)).count(),
    );
    let non_ties = accumulators.len() - accumulators.iter().filter(|a| is_tie(**a, shift)).count();
    if !off_tie_wrong.is_empty() {
        println!(
            "  {} non-tie accumulators disagree (accumulator, want, got): {:?}",
            off_tie_wrong.len(),
            &off_tie_wrong[..off_tie_wrong.len().min(8)]
        );
        // A handful of stragglers is a separate finding worth printing; a
        // wholesale disagreement means the gain or the layout is wrong and
        // the tie classification below would be measuring neither.
        if off_tie_wrong.len() * 20 > non_ties {
            println!("  NOT VALID -- more than 5% of non-ties disagree");
            return Vec::new();
        }
    }

    println!("  ties (accumulator, half-up, half-away, half-even, hardware):");
    let mut mismatches = [0usize; 3];
    for (&accumulator, &got) in accumulators.iter().zip(&observed) {
        if !is_tie(accumulator, shift) {
            continue;
        }
        let predictions = Rule::ALL.map(|rule| rule.apply(accumulator, shift));
        for (index, prediction) in predictions.iter().enumerate() {
            if *prediction != got {
                mismatches[index] += 1;
            }
        }
        // Only the ties where the rules actually disagree are interesting.
        if predictions[0] != predictions[1] || predictions[1] != predictions[2] {
            println!(
                "    {accumulator:>5}  {:>4} {:>4} {:>4}   {got:>4}",
                predictions[0], predictions[1], predictions[2]
            );
        }
    }

    let ties = accumulators.iter().filter(|a| is_tie(**a, shift)).count();
    let mut explains = Vec::new();
    for (index, rule) in Rule::ALL.into_iter().enumerate() {
        println!(
            "  {:<26} {} of {ties} ties wrong",
            rule.name(),
            mismatches[index]
        );
        if mismatches[index] == 0 {
            explains.push(rule);
        }
    }
    explains
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn classifies_the_requantisation_tie_rule() {
    println!(
        "1x1 conv at Cin 1: the accumulator at each pixel is that pixel's input byte.\n\
         Coefficients 1, BsEntry::default(), Multiplier::for_unit_bs(1 / 2^s)."
    );

    // The shipped configuration: `conv_2d_tile` never sets `cvt_round`.
    let mut shipped: Option<Vec<Rule>> = None;
    for shift in [1u32, 2] {
        let explains = classify(shift, false);
        assert!(
            !explains.is_empty(),
            "shift {shift}: no candidate rule explains the hardware -- \
             see the table above"
        );
        shipped = Some(match shipped {
            None => explains,
            Some(previous) => previous
                .into_iter()
                .filter(|rule| explains.contains(rule))
                .collect(),
        });
    }
    let shipped = shipped.expect("at least one shift was measured");
    println!(
        "\nrules consistent with every shift at cvt_round = 0: {:?}",
        shipped.iter().map(|rule| rule.name()).collect::<Vec<_>>()
    );

    // Whether the field does anything at all on this path. If setting it
    // moves the tie rule, `DPU_OUT_CVT` is the rounding stage and its `0`
    // state simply is not the half-to-even the register documentation
    // claims; if it does not, the rounding happens somewhere this field
    // does not reach.
    let flipped = classify(1, true);
    println!(
        "\nrules consistent at cvt_round = 1, shift 1: {:?}",
        flipped.iter().map(|rule| rule.name()).collect::<Vec<_>>()
    );
    if flipped == shipped {
        println!(
            "cvt_round is INERT on this path -- the tie rule is unchanged by \
             the only field that documents one."
        );
    } else {
        println!("cvt_round MOVES the tie rule, so DPU_OUT_CVT is the rounding stage.");
    }

    assert_eq!(
        shipped,
        vec![Rule::HalfAwayFromZero],
        "the shipped configuration's tie rule changed; \
         `conv2d_oracle.rs`'s `rounded_shift` models half-away-from-zero"
    );
}
