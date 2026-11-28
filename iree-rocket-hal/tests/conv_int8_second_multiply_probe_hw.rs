#![cfg(feature = "hardware-characterization")]

//! Hardware characterization: can an int8 convolution carry a second
//! per-channel multiply, or a per-channel shift, so a per-channel
//! requantization factor gets more range than one 16-bit BS lane?
//!
//!   CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!     cargo test --target aarch64-unknown-linux-gnu --release \
//!       --features hardware-characterization \
//!       --test conv_int8_second_multiply_probe_hw --no-run
//!
//!   ./conv_int8_second_multiply_probe_hw-<hash> --ignored --nocapture \
//!       --test-threads=1 <one test name>
//!
//! One test per process: long NPU sweeps contaminate themselves.
//!
//! # Why
//!
//! The requantized int8 path applies a per-channel factor `m[c]` on the BS
//! plane's 16-bit multiplier lane, `round(2^14 * m)`. A stock per-channel
//! ResNet50 has 15 convolutions whose factors span more than that lane
//! resolves (smallest live `m` 5e-7) and so stay fp16. Two hardware
//! mechanisms could carry the rest, and neither has been measured inside an
//! int8 convolution:
//!
//! 1. **BN MUL from NRDMA** (`bn_mul_src = 1`, `nrdma_data_use` bit 2): a
//!    second per-channel multiply after BS. `bn_plane.rs` measured it in a
//!    standalone *fp16* pass-through, where the operand is an IEEE fp16 at
//!    `channel * 2`. In an int8 convolution the datapath is fixed point;
//!    whether the operand is still fp16 or an `i16` under
//!    `bn_mul_shift_value` is the first question.
//! 2. **A per-channel shift** (`*_truncate_src = 1`, the `TRT` bit of
//!    `brdma_data_use` / `nrdma_data_use`): the TRM lists it, while
//!    `rockchip-npu-notes/encodings/out-cvt-converter.md` reads Mesa's
//!    register file as "one shift per stage". Unmeasured either way.
//!
//! # Method
//!
//! As `conv_requant_tie_rule_hw`: a 1x1 convolution at `Cin` 1 with every
//! coefficient 1 makes each output pixel's accumulator its input byte, so a
//! 16x16 job sweeps every accumulator from -128 to 127 on every one of 16
//! output channels. The BS plane is at unit (or, for the TRT arm, under
//! test), and `DPU_OUT_CVT` cancels the BS gain, so the output is
//! `round(accumulator * g[c])` for the per-channel gain `g[c]` of the stage
//! under test. Each arm patches `conv_2d_tile`'s program to engage that
//! stage, fits `g[c]` over the unsaturated outputs, and prints it beside
//! each hypothesis' prediction.

#[path = "support/conv2d_oracle.rs"]
mod conv2d_oracle;

use std::{fs::OpenOptions, mem, os::unix::io::AsRawFd, ptr};

use conv2d_oracle::{feature_offset, input_storage_bytes, output_offset, output_storage_bytes};

use iree_rocket_hal::rocket::{
    builders::{
        Bits, RegCmd, Register, RegisterMeta,
        dpu::{DpuBnCfg, DpuBnMulCfg, DpuBsMulCfg, DpuDataFormat},
        dpu_rdma::{DpuRdmaBnBaseAddr, DpuRdmaBrdmaCfg, DpuRdmaNrdmaCfg},
    },
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
/// Bytes of every operand buffer the arms allocate beyond what they write,
/// filled with `FILL` so a read past the written region is recognizable.
const PLANE_BYTES: usize = 4096;
const FILL: u8 = 0;

fn page_aligned_size(size: usize) -> usize {
    size.div_ceil(PAGE_BYTES) * PAGE_BYTES
}

const VALUE_MASK: u64 = 0xffff_ffff << 16;

fn identity(command: &RegCmd) -> (u32, u32) {
    ((command.0 >> 48) as u32, command.0 as u32 & 0xffff)
}

fn find<R: RegisterMeta>(commands: &mut [RegCmd]) -> &mut RegCmd {
    let mut found = commands
        .iter_mut()
        .filter(|command| identity(command) == (R::DOMAIN, R::OFFSET));
    let command = found.next().expect("register not in the program");
    assert!(found.next().is_none(), "register written twice");
    command
}

/// Replaces the whole value of `R`'s write with `value`'s.
fn replace<R: RegisterMeta>(commands: &mut [RegCmd], value: RegCmd) {
    let command = find::<R>(commands);
    command.0 = (command.0 & !VALUE_MASK) | (value.0 & VALUE_MASK);
}

/// Clears the bits `mask` sets in `R`'s write, then ORs in `value`'s.
fn set_field<R: RegisterMeta>(commands: &mut [RegCmd], mask: RegCmd, value: RegCmd) {
    let command = find::<R>(commands);
    command.0 = (command.0 & !(mask.0 & VALUE_MASK)) | (value.0 & VALUE_MASK);
}

/// What one arm patches into the program and the operand planes it binds.
struct Arm {
    /// The BS buffer `DPU_RDMA_BS_BASE_ADDR` points at.
    bs: Vec<u8>,
    /// The BN plane `DPU_RDMA_BN_BASE_ADDR` points at, when NRDMA is on.
    bn: Option<Vec<u8>>,
    patch: Box<dyn Fn(&mut Vec<RegCmd>)>,
}

fn unit_bs() -> Vec<u8> {
    let mut bs = vec![FILL; PLANE_BYTES];
    let entries = vec![BsEntry::default(); OUT_CHANNELS as usize];
    write_bs_buffer(&mut bs, &entries);
    bs
}

/// Runs the sweep under `arm`, returning `out[c][accumulator + 128]`.
fn run(arm: &Arm) -> Vec<Vec<i32>> {
    let precision = Precision::Int8(Quantization {
        input_zero_point: 0,
        output_zero_point: 0,
        weight_zero_point: 0,
        input_scale: 1.0,
        weights_scale: 1.0,
        multiplier: Multiplier::for_unit_bs(1.0),
        per_channel_shift: false,
    });
    let shape = Shape::with_precision(WIDTH, HEIGHT, 1, IN_CHANNELS, OUT_CHANNELS, precision);
    let width = WIDTH as usize;
    let pixels = width * HEIGHT as usize;
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
        for pixel in 0..pixels {
            let value = pixel as i32 - 128;
            input[feature_offset(shape, 0, pixel / width, pixel % width)] = value as i8 as u8;
        }

        let weight_bytes = shape.weight_bytes(KERNELS) as usize;
        let buf_weights = Buffer::new(fd, page_aligned_size(weight_bytes), &file);
        ptr::write_bytes(buf_weights.host_ptr, 0, buf_weights.size);
        pack_hwcf_to_rocket_weights(
            &vec![1u8; (IN_CHANNELS * OUT_CHANNELS) as usize],
            1,
            1,
            IN_CHANNELS as usize,
            OUT_CHANNELS as usize,
            1,
            std::slice::from_raw_parts_mut(buf_weights.host_ptr, weight_bytes),
        )
        .expect("failed to pack coefficients");

        let buf_bs = Buffer::new(fd, page_aligned_size(arm.bs.len()), &file);
        ptr::write_bytes(buf_bs.host_ptr, FILL, buf_bs.size);
        ptr::copy_nonoverlapping(arm.bs.as_ptr(), buf_bs.host_ptr, arm.bs.len());

        let buf_bn = arm.bn.as_ref().map(|plane| {
            let buffer = Buffer::new(fd, page_aligned_size(plane.len()), &file);
            ptr::write_bytes(buffer.host_ptr, FILL, buffer.size);
            ptr::copy_nonoverlapping(plane.as_ptr(), buffer.host_ptr, plane.len());
            buffer
        });

        let buf_output = Buffer::new(fd, page_aligned_size(output_bytes), &file);
        ptr::write_bytes(buf_output.host_ptr, 0xa5, buf_output.size);

        let mut commands = conv_2d_tile(shape, KERNELS, &Tile::whole(shape, KERNELS));
        (arm.patch)(&mut commands);
        relocate(
            &mut commands,
            Buffers {
                input: buf_input.dma_address,
                weights: buf_weights.dma_address,
                bias: buf_bs.dma_address,
                output: buf_output.dma_address,
            },
        );
        if let Some(bn) = &buf_bn {
            replace::<DpuRdmaBnBaseAddr>(
                &mut commands,
                Register::<DpuRdmaBnBaseAddr>::new()
                    .bn_base_addr(Bits::new(bn.dma_address))
                    .build(),
            );
        }

        let command_bytes = commands.len() * mem::size_of::<u64>();
        let buf_commands = Buffer::new(fd, page_aligned_size(command_bytes), &file);
        ptr::write_bytes(buf_commands.host_ptr, 0, buf_commands.size);
        let words =
            std::slice::from_raw_parts_mut(buf_commands.host_ptr as *mut u64, commands.len());
        for (destination, command) in words.iter_mut().zip(&commands) {
            *destination = command.0;
        }

        let mut handles = vec![
            buf_input.handle,
            buf_weights.handle,
            buf_bs.handle,
            buf_output.handle,
            buf_commands.handle,
        ];
        if let Some(bn) = &buf_bn {
            handles.push(bn.handle);
        }
        for &handle in &handles {
            fini_bo(fd, handle).expect("failed to sync BO for the NPU");
        }

        let tasks = [(buf_commands.dma_address, commands.len() as u32)];
        let mut in_handles = vec![
            buf_commands.handle,
            buf_input.handle,
            buf_weights.handle,
            buf_bs.handle,
        ];
        if let Some(bn) = &buf_bn {
            in_handles.push(bn.handle);
        }
        let out_handles = [buf_output.handle];
        let jobs = [JobDesc {
            tasks: &tasks,
            in_handles: &in_handles,
            out_handles: &out_handles,
        }];
        submit_jobs(fd, &jobs).expect("SUBMIT failed");
        prep_bo(fd, buf_output.handle, 5_000_000_000).expect("job did not complete");

        let raw = std::slice::from_raw_parts(buf_output.host_ptr as *const i8, output_bytes);
        let observed = (0..OUT_CHANNELS as usize)
            .map(|channel| {
                (0..pixels)
                    .map(|pixel| {
                        i32::from(
                            raw[output_offset(
                                shape,
                                KERNELS,
                                channel,
                                pixel / width,
                                pixel % width,
                            )],
                        )
                    })
                    .collect()
            })
            .collect();
        for handle in handles {
            let _ = close_bo(fd, handle);
        }
        observed
    }
}

/// Least-squares gain over the outputs that did not clamp, and how many
/// clamped; `None` when every nonzero accumulator clamped.
fn fit(out: &[i32]) -> (Option<f64>, usize) {
    let mut num = 0.0;
    let mut den = 0.0;
    let mut clamped = 0;
    for (index, &y) in out.iter().enumerate() {
        let acc = index as f64 - 128.0;
        if y <= -128 || y >= 127 {
            clamped += 1;
            continue;
        }
        num += acc * f64::from(y);
        den += acc * acc;
    }
    (if den > 0.0 { Some(num / den) } else { None }, clamped)
}

fn fp16_value(bits: u16) -> f64 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let mantissa = f64::from(bits & 0x3ff);
    match exponent {
        0 => sign * mantissa * 2f64.powi(-24),
        31 => f64::NAN,
        _ => sign * (1.0 + mantissa / 1024.0) * 2f64.powi(exponent - 15),
    }
}

/// Rounds `value` half away from zero, the DPU's tie rule (memory:
/// int8-tie-rule-half-away).
fn round_away(value: f64) -> i32 {
    value.signum() as i32 * (value.abs() + 0.5).floor() as i32
}

/// Mismatches of `out` against `round(acc * gain)` clamped to int8.
fn mismatches(out: &[i32], gain: f64) -> usize {
    out.iter()
        .enumerate()
        .filter(|&(index, &y)| {
            let acc = index as f64 - 128.0;
            round_away(acc * gain).clamp(-128, 127) != y
        })
        .count()
}

fn report(title: &str, observed: &[Vec<i32>], hypotheses: &[(&str, Vec<f64>)]) {
    println!("\n=== {title} ===");
    let sentinel = observed
        .iter()
        .flatten()
        .filter(|&&y| y == 0xa5u8 as i8 as i32)
        .count();
    println!(
        "  bytes still at the 0xa5 sentinel: {sentinel} of {}",
        16 * 256
    );
    for (channel, out) in observed.iter().enumerate() {
        let (gain, clamped) = fit(out);
        let gain = gain.map_or("-".to_string(), |g| format!("{g:.6}"));
        let mut line = format!("  c{channel:2}: fitted gain {gain:>10} ({clamped:3} clamped)");
        for (name, gains) in hypotheses {
            let wrong = mismatches(out, gains[channel]);
            line += &format!(" | {name} {:.6} -> {wrong:3} off", gains[channel]);
        }
        println!("{line}");
        if channel < 3 {
            let gain = hypotheses.first().map_or(1.0, |(_, gains)| gains[channel]);
            let wrong: Vec<(i32, i32, i32)> = out
                .iter()
                .enumerate()
                .map(|(index, &y)| {
                    let acc = index as i32 - 128;
                    (acc, round_away(f64::from(acc) * gain).clamp(-128, 127), y)
                })
                .filter(|&(_, want, got)| want != got)
                .collect();
            println!(
                "       (acc, want, got) off: {:?}",
                &wrong[..wrong.len().min(24)]
            );
        }
    }
}

/// The 16-bit operands the BN arms give channels 0..16: fixed-point values
/// that are exact under `shift = 14`, then four fp16 encodings (1.0, 0.5,
/// 0.75, 0.25), so one run separates the two encodings.
const BN_OPERANDS: [u16; 16] = [
    16384, 8192, 12288, 4096, 2048, 24576, 1024, 16383, 32767, 20000, 10000, 5000, 0x3c00, 0x3800,
    0x3a00, 0x3400,
];

fn bn_mul_arm(shift: u32) -> Arm {
    let mut plane = vec![FILL; PLANE_BYTES];
    for (channel, &operand) in BN_OPERANDS.iter().enumerate() {
        plane[channel * 2..channel * 2 + 2].copy_from_slice(&operand.to_le_bytes());
    }
    Arm {
        bs: unit_bs(),
        bn: Some(plane),
        patch: Box::new(move |commands| {
            replace::<DpuBnCfg>(
                commands,
                Register::<DpuBnCfg>::new()
                    .bn_bypass(Bits::new(0))
                    .bn_alu_bypass(Bits::new(1))
                    .bn_mul_bypass(Bits::new(0))
                    .bn_relu_bypass(Bits::new(1))
                    .build(),
            );
            replace::<DpuBnMulCfg>(
                commands,
                Register::<DpuBnMulCfg>::new()
                    .bn_mul_src(Bits::new(1))
                    .bn_mul_shift_value(Bits::new(shift))
                    .build(),
            );
            set_field::<DpuDataFormat>(
                commands,
                Register::<DpuDataFormat>::new()
                    .bn_mul_shift_value_neg(Bits::new(63))
                    .build(),
                Register::<DpuDataFormat>::new()
                    .bn_mul_shift_value_neg(Bits::new(shift))
                    .build(),
            );
            replace::<DpuRdmaNrdmaCfg>(
                commands,
                Register::<DpuRdmaNrdmaCfg>::new()
                    .nrdma_data_use(Bits::new(0b0100))
                    .build(),
            );
        }),
    }
}

fn bn_hypotheses(shift: u32) -> Vec<(&'static str, Vec<f64>)> {
    let fixed = |s: i32| {
        BN_OPERANDS
            .iter()
            .map(|&r| f64::from(r as i16) / 2f64.powi(s))
            .collect::<Vec<_>>()
    };
    vec![
        ("i16>>S", fixed(shift as i32)),
        ("i16>>(S-7)", fixed(shift as i32 - 7)),
        ("fp16", BN_OPERANDS.iter().map(|&r| fp16_value(r)).collect()),
    ]
}

/// Q1: the BN MUL operand inside an int8 conv, at `bn_mul_shift_value` 14.
#[test]
#[ignore]
fn bn_mul_operand_at_shift_14() {
    let observed = run(&bn_mul_arm(14));
    report("BN MUL from NRDMA, shift 14", &observed, &bn_hypotheses(14));
}

/// Q1 again at shift 0: the fp16 hypothesis predicts the same gains as at
/// 14 (a float operand does not shift); fixed point predicts saturation.
#[test]
#[ignore]
fn bn_mul_operand_at_shift_0() {
    let observed = run(&bn_mul_arm(0));
    report("BN MUL from NRDMA, shift 0", &observed, &bn_hypotheses(0));
}

/// Control: the unpatched program on the same sweep. Every channel should
/// fit gain 1 with 0 off; if not, the harness is wrong and the arms above
/// mean nothing.
#[test]
#[ignore]
fn control_unit_gain() {
    let observed = run(&Arm {
        bs: unit_bs(),
        bn: None,
        patch: Box::new(|_| {}),
    });
    report(
        "control: unit BS, BN bypassed",
        &observed,
        &[("unit", vec![1.0; 16])],
    );
}

/// Q2: a per-channel BS shift. `brdma_data_use` gains the TRT bit and
/// `bs_truncate_src` selects BRDMA for the shift; where the shift lives in
/// the BS buffer is unknown, so `ROCKET_PROBE_TRT_LAYOUT` picks one
/// candidate per process:
///
/// - `after8` (default): each 8-channel block grows from 64 to 72 bytes,
///   one shift byte per channel at `+64`;
/// - `after16`: blocks of 80 bytes, one `u16` per channel at `+64`;
/// - `plane`: blocks stay 64 bytes and the shifts are a separate byte
///   plane after the last block.
///
/// Channel `c` asks for shift `14 + (c % 4)`, so the gains a correct
/// layout produces are `2^-(c % 4)` -- and the fitted gains of a wrong
/// one name whatever the hardware actually read.
#[test]
#[ignore]
fn bs_per_channel_shift() {
    let layout = std::env::var("ROCKET_PROBE_TRT_LAYOUT").unwrap_or_else(|_| "after8".into());
    let shifts: Vec<u8> = (0..OUT_CHANNELS).map(|c| 14 + (c % 4) as u8).collect();
    let mut bs = vec![FILL; PLANE_BYTES];
    let blocks = (OUT_CHANNELS as usize).div_ceil(8);
    let (block_bytes, trt_bytes) = match layout.as_str() {
        "after8" => (72, 1),
        "after16" => (80, 2),
        "plane" => (64, 1),
        // The layout `after16` decoded: 80-byte blocks, one byte per
        // channel at +64 and another at +72. The two lanes get different
        // shifts here (+64: 14 + c % 4, +72: always 15), so which lane
        // serves which sign is visible in one run.
        "posneg" => (80, 1),
        other => panic!("unknown ROCKET_PROBE_TRT_LAYOUT {other}"),
    };
    for channel in 0..OUT_CHANNELS as usize {
        let base = (channel / 8) * block_bytes;
        let lane = channel % 8;
        let entry = BsEntry::default();
        bs[base + lane * 4..base + lane * 4 + 4].copy_from_slice(&entry.bias.to_le_bytes());
        bs[base + 32 + lane * 2..base + 34 + lane * 2]
            .copy_from_slice(&entry.constant.to_le_bytes());
        bs[base + 48 + lane * 2..base + 50 + lane * 2]
            .copy_from_slice(&entry.multiplier.to_le_bytes());
        let at = if layout == "plane" {
            blocks * 64 + channel
        } else {
            base + 64 + lane * trt_bytes
        };
        bs[at] = shifts[channel];
        if layout == "posneg" {
            bs[base + 72 + lane] = 15;
        }
    }
    let gains = shifts
        .iter()
        .map(|&s| 2f64.powi(14 - i32::from(s)))
        .collect::<Vec<_>>();
    let observed = run(&Arm {
        bs,
        bn: None,
        patch: Box::new(|commands| {
            replace::<DpuRdmaBrdmaCfg>(
                commands,
                Register::<DpuRdmaBrdmaCfg>::new()
                    .brdma_data_use(Bits::new(0b1111))
                    .build(),
            );
            let mask = Register::<DpuBsMulCfg>::new()
                .bs_truncate_src(Bits::new(1))
                .build();
            let value = RegCmd(mask.0);
            set_field::<DpuBsMulCfg>(commands, mask, value);
        }),
    });
    if layout == "posneg" {
        // Sign-dependent check of both lane assignments.
        for (name, positive_lane_first) in [
            ("+64 positive, +72 negative", true),
            ("+64 negative, +72 positive", false),
        ] {
            let mut wrong = 0;
            for (channel, out) in observed.iter().enumerate() {
                let lane64 = 2f64.powi(14 - i32::from(shifts[channel]));
                let lane72 = 0.5;
                for (index, &y) in out.iter().enumerate() {
                    let acc = index as f64 - 128.0;
                    let gain = if (acc >= 0.0) == positive_lane_first {
                        lane64
                    } else {
                        lane72
                    };
                    if round_away(acc * gain).clamp(-128, 127) != y {
                        wrong += 1;
                    }
                }
            }
            println!("  posneg, {name}: {wrong} of 4096 off");
        }
    }
    report(
        &format!("BS per-channel shift, layout {layout}"),
        &observed,
        &[
            ("per-channel shift", gains),
            ("unit (TRT ignored)", vec![1.0; 16]),
        ],
    );
}
