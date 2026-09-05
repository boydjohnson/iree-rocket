//! ISSUES.md C8 at the HAL level: an `Int8Accumulator` job used to poison the
//! core it ran on, so a following wide fp16 job hung until the power domain
//! cycled. **Fixed 2026-09-05** -- the cause was this crate leaving
//! `DPU_RDMA_BRDMA_CFG.brdma_data_use` at 7 on a path that bypasses the BS
//! plane, so BRDMA fetched a bias/scale/shift triple nothing consumed. The
//! vendor emitter (`rocket-userspace`'s `gen_conv2d_int8`) leaves that
//! register at 0 and never poisoned; bisecting the two programs field by
//! field pinned it to that register alone.
//!
//! Every arm here therefore expects **Clean** now. `ROCKET_ACC_BRDMA=1` puts
//! the old value back, which reproduces the hang on demand and is how this
//! test doubles as a regression guard: run it once plain (all clean) and once
//! with that variable (the poisoning arms hang again).
//!
//! The arms remain as the characterisation that found it, and they still
//! record what the symptom looked like: the state was per core (each job's
//! core is read off `/proc/interrupts`, and `drm_sched` alternates an idle
//! entity between cores, so a victim hung only when it landed on the
//! aggressor's core), it only appeared with the int32 output writer, it needed
//! the victim's output past ~256 KiB, and only the core's runtime suspend
//! cleared it. Each job is prepared, packed and cache-synced before an arm
//! starts, the aggressors and victim are submitted back to back with the gap
//! under the test's control, and every verdict is computed after the victim's
//! fence.
//!
//! Cross-compile and run on the board:
//!
//! ```text
//! CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!   cargo test --target aarch64-unknown-linux-gnu --release \
//!     --test c8_precision_transition_hw --no-run
//!
//! ROCKET_DISPATCH_TIMES=1 ./c8_precision_transition_hw-<hash> --ignored --nocapture
//! ```
//!
//! `ROCKET_C8_ARMS=name,name` runs a subset; `ROCKET_C8_REPEAT=n` (default 2)
//! repeats each arm. A hang costs ~500 ms and a core reset, and the state
//! crosses processes, so every arm starts from a suspended domain and a
//! canary runs after any timeout.
//!
//! 2026-09-05 the arms below narrowed the mechanism to a single register
//! field. A register-program diff of one shape at every precision
//! (`examples/dump_conv_plan_regcmd.rs`) found all precisions write the same
//! register set -- nothing is inherited stale -- and only four registers carry
//! a value unique to the poisoner. `bs_engage` clears `DPU_BS_CFG.bs_bypass`
//! and still hangs with exact output, as does `od_engage` clearing
//! `BS_OW_CFG.od_bypass`, alone or with it; `Fp16Accumulator` aggressors
//! (fp32 out, the other 4-byte writer) are clean. So it is `DATA_FORMAT.out_precision = 4`, the int32
//! writer, and not output width or the bypassed BS plane. See ISSUES.md C8.
//!
//! What it established on planck, 2026-09-04 (every arm 2-3 trials, all
//! consistent):
//!
//! * The state is **per core**. Each job's core is read off
//!   `/proc/interrupts`; `drm_sched` alternates an idle entity between cores,
//!   so after one `Int8Accumulator` job on c0 the first wide fp16 victim
//!   lands on c1 and is clean, and the *second* lands on c0 and hangs
//!   (`q1_then_k1x4_gap0`, 3/3). The IREE-level "the stem hangs on its 4th
//!   submission" and "one aggressor is not enough for `k1`" were both
//!   placement, not an accumulating dose.
//! * It is the **int32 output writer**. The same three shapes run as
//!   requantized `Int8` (one-byte output) or as `Fp16` poison neither core
//!   (`int8req_then_k1x4_gap0`, `fp16_then_k1x4_gap0`: the victim visits
//!   both cores four times, clean).
//! * The victim's size still matters: `bigin` (128 KiB of fp16 out) is clean
//!   on a poisoned core, `px33` (139 KiB) and up hang.
//! * A 30 ms gap (cores still `active`) hangs; 150 ms, or polling until
//!   every core reads `suspended`, is clean. The domain cycling is what
//!   clears it, on the core that cycled.

#[path = "support/conv2d_oracle.rs"]
mod conv2d_oracle;
#[path = "support/dispatch.rs"]
mod dispatch;

use std::{
    fs::OpenOptions,
    mem,
    os::unix::io::AsRawFd,
    ptr,
    sync::Mutex,
    time::{Duration, Instant},
};

use conv2d_oracle::{
    Conv2dCase, Conv2dFixture, OraclePattern, OraclePrecision, build_fixture, expected_output,
    f16_to_f32, output_offset, output_storage_bytes,
};
use dispatch::DISPATCH_TIMEOUT_FLOOR;
use iree_rocket_hal::rocket::{
    conv::{AccumulatorOutputTile, Buffers, ConvPlan, Shape},
    device::{Buffer, JobDesc, close_bo, fini_bo, prep_bo, submit_jobs, unmap_bo},
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const PAGE_BYTES: usize = 4096;
const PER_JOB_TIMEOUT_NS: u64 = 5_000_000_000;
const OUTPUT_SENTINEL: u8 = 0xa5;
/// How long to wait for every core to report `suspended` before an arm, and
/// after a timeout. Autosuspend is 50 ms on planck; a hang adds the 500 ms
/// watchdog and a reset.
const SUSPEND_BUDGET: Duration = Duration::from_secs(5);
/// Quiet gap before every arm, on top of the suspended-domain wait, so the
/// per-process contamination the wedge protocol describes stays out of it.
const SETTLE: Duration = Duration::from_millis(1000);

static NPU_TEST_LOCK: Mutex<()> = Mutex::new(());

// ---------------------------------------------------------------------------
// Runtime PM
// ---------------------------------------------------------------------------

fn npu_runtime_status_paths() -> Vec<std::path::PathBuf> {
    let mut paths: Vec<_> = std::fs::read_dir("/sys/devices/platform")
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.to_string_lossy().ends_with(".npu"))
        .map(|path| path.join("power/runtime_status"))
        .collect();
    paths.sort();
    paths
}

/// One character per core: `S` suspended, `A` active, `?` anything else.
fn runtime_status_summary(paths: &[std::path::PathBuf]) -> String {
    paths
        .iter()
        .map(|path| match std::fs::read_to_string(path) {
            Ok(status) => match status.trim() {
                "suspended" => 'S',
                "active" => 'A',
                _ => '?',
            },
            Err(_) => '?',
        })
        .collect()
}

fn all_suspended(paths: &[std::path::PathBuf]) -> bool {
    !paths.is_empty()
        && paths
            .iter()
            .all(|path| std::fs::read_to_string(path).is_ok_and(|s| s.trim() == "suspended"))
}

/// Polls until every core reports `suspended`; returns how long it took and
/// whether it got there inside `budget`.
fn wait_all_suspended(paths: &[std::path::PathBuf], budget: Duration) -> (Duration, bool) {
    let started = Instant::now();
    loop {
        if all_suspended(paths) {
            return (started.elapsed(), true);
        }
        if started.elapsed() > budget {
            return (started.elapsed(), false);
        }
        std::thread::sleep(Duration::from_micros(500));
    }
}

/// Completion-interrupt counts per NPU core, from `/proc/interrupts`.
///
/// IRQ 82/83/84 are the three cores' lines (shared with their IOMMUs), so a
/// delta across one job names the core that completed it. The kernel's
/// `drm_sched` picks the core; one fd is one entity, but an idle entity can
/// land on a different core for its next job, and that is exactly the
/// question the arms below need answered.
fn npu_irq_counts() -> Vec<u64> {
    std::fs::read_to_string("/proc/interrupts")
        .unwrap_or_default()
        .lines()
        .filter(|line| line.contains(".npu"))
        .map(|line| {
            line.split_whitespace()
                .skip(1)
                .take_while(|field| field.chars().all(|c| c.is_ascii_digit()))
                .filter_map(|field| field.parse::<u64>().ok())
                .sum()
        })
        .collect()
}

/// Which core(s) completed interrupts between two readings, as a string like
/// `c0` or `c0+c1`; `?` when nothing moved (a job the watchdog killed never
/// raises one).
fn cores_between(before: &[u64], after: &[u64]) -> String {
    let moved: Vec<String> = before
        .iter()
        .zip(after)
        .enumerate()
        .filter(|(_, (b, a))| a > b)
        .map(|(index, _)| format!("c{index}"))
        .collect();
    if moved.is_empty() {
        "?".to_string()
    } else {
        moved.join("+")
    }
}

// ---------------------------------------------------------------------------
// Buffers and prepared jobs
// ---------------------------------------------------------------------------

fn page_aligned_size(size: usize) -> usize {
    size.max(1).div_ceil(PAGE_BYTES) * PAGE_BYTES
}

struct OwnedBuffer {
    fd: i32,
    buffer: Buffer,
}

impl OwnedBuffer {
    unsafe fn new(fd: i32, size: usize, file: &std::fs::File) -> Self {
        Self {
            fd,
            buffer: unsafe { Buffer::new(fd, page_aligned_size(size), file) },
        }
    }

    unsafe fn from_bytes(fd: i32, bytes: &[u8], file: &std::fs::File) -> Self {
        let buffer = unsafe { Self::new(fd, bytes.len(), file) };
        unsafe {
            ptr::write_bytes(buffer.buffer.host_ptr, 0, buffer.buffer.size);
            ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.buffer.host_ptr, bytes.len());
        }
        buffer
    }
}

impl Drop for OwnedBuffer {
    fn drop(&mut self) {
        unsafe {
            let _ = unmap_bo(&self.buffer);
            let _ = close_bo(self.fd, self.buffer.handle);
        }
    }
}

/// A conv job with every buffer allocated, packed and synced for the device,
/// so that submitting it is one ioctl and one fence wait and nothing else.
struct PreparedJob {
    fixture: Conv2dFixture,
    plan: ConvPlan,
    accumulator_tiles: Option<Vec<AccumulatorOutputTile>>,
    _input: OwnedBuffer,
    _weights: OwnedBuffer,
    _bias: OwnedBuffer,
    output: OwnedBuffer,
    output_len: usize,
    _command_buffers: Vec<OwnedBuffer>,
    tasks: Vec<Vec<(u32, u32)>>,
    in_handles: Vec<Vec<u32>>,
    out_handles: Vec<u32>,
}

impl PreparedJob {
    fn label(&self) -> String {
        let shape = self.fixture.shape;
        let case = self.fixture.case;
        let out_bytes = shape.output_width(case.kernel) as usize
            * shape.output_height(case.kernel) as usize
            * shape.padded_out_channels() as usize
            * shape.precision.output_element_bytes() as usize;
        format!(
            "{} {}x{} Cin{} Cout{} k{} s{} -> {} KiB out, {} task(s)",
            case.precision.name(),
            case.width,
            case.height,
            case.cin,
            case.cout,
            case.kernel[0],
            case.stride,
            out_bytes / 1024,
            self.tasks.len(),
        )
    }

    unsafe fn prepare(file: &std::fs::File, case: Conv2dCase) -> Result<Self, String> {
        let fd = file.as_raw_fd();
        let fixture = build_fixture(case)?;
        let shape = fixture.shape;
        let plan = ConvPlan::new(shape, case.kernel);
        let output_len = output_storage_bytes(shape, case.kernel);

        unsafe {
            let input = OwnedBuffer::from_bytes(fd, &fixture.input, file);
            let weights = OwnedBuffer::from_bytes(fd, &fixture.weights, file);
            let bias = OwnedBuffer::from_bytes(fd, &fixture.bias, file);
            let output = OwnedBuffer::new(fd, output_len, file);
            ptr::write_bytes(output.buffer.host_ptr, OUTPUT_SENTINEL, output.buffer.size);

            let buffers = Buffers {
                input: input.buffer.dma_address,
                weights: weights.buffer.dma_address,
                bias: bias.buffer.dma_address,
                output: output.buffer.dma_address,
            };
            let (programs, accumulator_tiles) = if shape.precision.writes_accumulators() {
                let staged = plan.programs_with_staged_accumulator_output(buffers);
                (staged.programs, Some(staged.tiles))
            } else {
                (plan.programs_with_buffers(buffers), None)
            };

            let mut command_buffers = Vec::with_capacity(programs.len());
            let mut tasks = Vec::with_capacity(programs.len());
            for program in &programs {
                let command_bytes = program.len() * mem::size_of::<u64>();
                let buffer = OwnedBuffer::new(fd, command_bytes, file);
                ptr::write_bytes(buffer.buffer.host_ptr, 0, buffer.buffer.size);
                let words = std::slice::from_raw_parts_mut(
                    buffer.buffer.host_ptr as *mut u64,
                    program.len(),
                );
                for (destination, command) in words.iter_mut().zip(program) {
                    *destination = command.0;
                }
                tasks.push(vec![(buffer.buffer.dma_address, program.len() as u32)]);
                command_buffers.push(buffer);
            }

            for handle in [
                input.buffer.handle,
                weights.buffer.handle,
                bias.buffer.handle,
                output.buffer.handle,
            ] {
                fini_bo(fd, handle).map_err(|error| format!("sync data BO: {error}"))?;
            }
            for buffer in &command_buffers {
                fini_bo(fd, buffer.buffer.handle)
                    .map_err(|error| format!("sync regcmd BO: {error}"))?;
            }

            let in_handles = command_buffers
                .iter()
                .map(|buffer| {
                    vec![
                        buffer.buffer.handle,
                        input.buffer.handle,
                        weights.buffer.handle,
                        bias.buffer.handle,
                    ]
                })
                .collect();
            let out_handles = vec![output.buffer.handle];

            Ok(Self {
                fixture,
                plan,
                accumulator_tiles,
                _input: input,
                _weights: weights,
                _bias: bias,
                output,
                output_len,
                _command_buffers: command_buffers,
                tasks,
                in_handles,
                out_handles,
            })
        }
    }

    /// One SUBMIT of every task as its own job, then the output fence.
    /// Returns the wall time, the only thing that tells a killed job from a
    /// result (see `support/dispatch.rs`).
    unsafe fn submit(&self, fd: i32) -> Result<(Duration, String), String> {
        let irqs_before = npu_irq_counts();
        let jobs: Vec<JobDesc> = self
            .tasks
            .iter()
            .zip(&self.in_handles)
            .map(|(tasks, in_handles)| JobDesc {
                tasks,
                in_handles,
                out_handles: &self.out_handles,
            })
            .collect();
        let started = Instant::now();
        unsafe {
            submit_jobs(fd, &jobs).map_err(|error| format!("submit: {error}"))?;
            prep_bo(fd, self.output.buffer.handle, PER_JOB_TIMEOUT_NS)
                .map_err(|error| format!("completion wait: {error}"))?;
        }
        let elapsed = started.elapsed();
        Ok((elapsed, cores_between(&irqs_before, &npu_irq_counts())))
    }

    /// Mismatch count against the oracle, computed after the fact.
    fn mismatches(&self) -> Result<usize, String> {
        let raw = unsafe {
            std::slice::from_raw_parts(self.output.buffer.host_ptr, self.output_len).to_vec()
        };
        let shape = self.fixture.shape;
        let case = self.fixture.case;
        let output = match &self.accumulator_tiles {
            Some(tiles) => assemble_staged_accumulator_output(shape, case.kernel, &raw, tiles)?,
            None => raw,
        };
        let tolerance = if case.precision == OraclePrecision::Int8 {
            1.0
        } else {
            0.0
        };
        let mut mismatches = 0;
        for y in 0..shape.output_height(case.kernel) as usize {
            for x in 0..shape.output_width(case.kernel) as usize {
                for channel in 0..case.cout as usize {
                    let offset = output_offset(shape, case.kernel, channel, y, x);
                    let got = match case.precision {
                        OraclePrecision::Fp16 => {
                            f16_to_f32(u16::from_le_bytes([output[offset], output[offset + 1]]))
                        }
                        OraclePrecision::Int8 => f32::from(output[offset] as i8),
                        OraclePrecision::Int8Accumulator => {
                            i32::from_le_bytes(output[offset..offset + 4].try_into().unwrap())
                                as f32
                        }
                        other => return Err(format!("no readback for {}", other.name())),
                    };
                    let want = expected_output(case, channel, y, x) as f32;
                    if !got.is_finite() || (got - want).abs() > tolerance {
                        mismatches += 1;
                    }
                }
            }
        }
        Ok(mismatches)
    }

    fn tiles(&self) -> usize {
        self.plan.tiles().len()
    }
}

/// Copied from `conv2d_oracle_hw.rs`: the DPU's atomic-slot-strided staging
/// reinterpreted as the logical output cube.
fn assemble_staged_accumulator_output(
    shape: Shape,
    kernels: [usize; 2],
    scratch: &[u8],
    tiles: &[AccumulatorOutputTile],
) -> Result<Vec<u8>, String> {
    let output_width = shape.output_width(kernels) as usize;
    let output_height = shape.output_height(kernels) as usize;
    let output_pixels = output_width * output_height;
    let block_bytes = shape.output_atom_bytes() as usize;
    let bytes_per_pixel =
        shape.padded_out_channels() as usize * shape.precision.output_element_bytes() as usize;
    let blocks_per_pixel = bytes_per_pixel.div_ceil(block_bytes);
    let mut output = vec![OUTPUT_SENTINEL; output_storage_bytes(shape, kernels)];

    for (index, tile) in tiles.iter().enumerate() {
        let tile_pixels = tile.output_rows * tile.output_columns;
        let tile_end = tile.scratch_offset + tile.scratch_bytes;
        if tile_end > scratch.len() {
            return Err(format!(
                "tile {index} scratch range exceeds {} bytes",
                scratch.len()
            ));
        }
        for surface in 0..blocks_per_pixel {
            for row in 0..tile.output_rows {
                for column in 0..tile.output_columns {
                    let local_pixel = row * tile.output_columns + column;
                    let output_row = tile.output_row + row;
                    let output_column = tile.output_column + column;
                    if output_row >= output_height || output_column >= output_width {
                        return Err(format!("tile {index} output exceeds the cube"));
                    }
                    let source = tile.scratch_offset
                        + surface * tile_pixels * block_bytes
                        + local_pixel * block_bytes;
                    let destination = surface * output_pixels * block_bytes
                        + (output_row * output_width + output_column) * block_bytes;
                    output[destination..destination + block_bytes]
                        .copy_from_slice(&scratch[source..source + block_bytes]);
                }
            }
        }
    }
    Ok(output)
}

// ---------------------------------------------------------------------------
// Shapes, from tools/c8_precision_transition_probe.py
// ---------------------------------------------------------------------------

fn case(
    width: u32,
    height: u32,
    cin: u32,
    cout: u32,
    kernel: usize,
    stride: u32,
    precision: OraclePrecision,
) -> Conv2dCase {
    Conv2dCase {
        width,
        height,
        cin,
        cout,
        kernel: [kernel, kernel],
        stride,
        padding: [0, 0],
        precision,
        pattern: OraclePattern::Dense { phase: 1 },
    }
}

/// The probe's three int8 aggressors, at a given precision so the same
/// shapes can be run as `Int8Accumulator`, requantized `Int8`, or `Fp16`.
fn aggressors(precision: OraclePrecision) -> Vec<Conv2dCase> {
    vec![
        case(32, 32, 64, 128, 1, 1, precision), // q1: 512 KiB of int32 out
        case(34, 34, 32, 64, 3, 1, precision),  // q2: 3x3, 256 KiB
        case(32, 32, 16, 512, 1, 1, precision), // q3: 2 MiB
    ]
}

/// fp16 victims. The probe's boundary sits between `bigin` (clean) and
/// `px33` (hangs); `k1` and `bigout` are well past it, `stem` is the model's.
fn victim(name: &str) -> Conv2dCase {
    match name {
        "bigin" => case(32, 32, 64, 64, 1, 1, OraclePrecision::Fp16),
        "px33" => case(33, 33, 16, 64, 1, 1, OraclePrecision::Fp16),
        "k1" => case(32, 32, 64, 128, 1, 1, OraclePrecision::Fp16),
        "bigout" => case(32, 32, 16, 256, 1, 1, OraclePrecision::Fp16),
        "stem" => case(225, 225, 3, 32, 3, 2, OraclePrecision::Fp16),
        other => panic!("unknown victim {other}"),
    }
}

// ---------------------------------------------------------------------------
// Arms
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
enum Gap {
    None,
    Sleep(Duration),
    UntilSuspended,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Expect {
    Hang,
    Clean,
    /// A question the arm is asked to answer, not a prediction.
    Open,
}

struct Arm {
    name: &'static str,
    aggressors: Vec<Conv2dCase>,
    victim: Conv2dCase,
    /// How many times the victim is submitted, each its own SUBMIT and
    /// fence, back to back. More than one lets the scheduler place the
    /// same job on different cores, which separates "the amount of int8
    /// work" from "the core that did it".
    victim_repeats: usize,
    gap: Gap,
    expect: Expect,
    /// Engage the BS plane as a pass-through on the `Int8Accumulator`
    /// aggressors (`conv::set_accumulator_bs_engage`). Off everywhere except
    /// the two arms that ask whether `bs_bypass` or `out_precision` is the
    /// poisoner; see those arms' comment.
    bs_engage: bool,
    /// Clear `BS_OW_CFG.od_bypass` on the `Int8Accumulator` aggressors
    /// (`conv::set_od_engage`), putting the output converter in
    /// the path. Safe only at the accumulator's own `size_e` of 7; see the
    /// arms that use it.
    od_engage: bool,
}

fn arms() -> Vec<Arm> {
    let acc = OraclePrecision::Int8Accumulator;
    vec![
        // The mechanism's predictions.
        Arm {
            name: "int8acc_then_k1_gap0",
            aggressors: aggressors(acc),
            victim: victim("k1"),
            victim_repeats: 1,
            gap: Gap::None,
            expect: Expect::Clean,
            bs_engage: false,
            od_engage: false,
        },
        Arm {
            name: "int8acc_then_bigin_gap0",
            aggressors: aggressors(acc),
            victim: victim("bigin"),
            victim_repeats: 1,
            gap: Gap::None,
            expect: Expect::Clean,
            bs_engage: false,
            od_engage: false,
        },
        Arm {
            name: "int8acc_then_px33_gap0",
            aggressors: aggressors(acc),
            victim: victim("px33"),
            victim_repeats: 1,
            gap: Gap::None,
            expect: Expect::Clean,
            bs_engage: false,
            od_engage: false,
        },
        Arm {
            name: "int8acc_then_k1_sleep30",
            aggressors: aggressors(acc),
            victim: victim("k1"),
            victim_repeats: 1,
            gap: Gap::Sleep(Duration::from_millis(30)),
            expect: Expect::Clean,
            bs_engage: false,
            od_engage: false,
        },
        Arm {
            name: "int8acc_then_k1_sleep150",
            aggressors: aggressors(acc),
            victim: victim("k1"),
            victim_repeats: 1,
            gap: Gap::Sleep(Duration::from_millis(150)),
            expect: Expect::Clean,
            bs_engage: false,
            od_engage: false,
        },
        Arm {
            name: "int8acc_then_k1_suspend",
            aggressors: aggressors(acc),
            victim: victim("k1"),
            victim_repeats: 1,
            gap: Gap::UntilSuspended,
            expect: Expect::Clean,
            bs_engage: false,
            od_engage: false,
        },
        Arm {
            name: "k1_alone",
            aggressors: Vec::new(),
            victim: victim("k1"),
            victim_repeats: 1,
            gap: Gap::None,
            expect: Expect::Clean,
            bs_engage: false,
            od_engage: false,
        },
        // The questions the fix design depended on, now answered (see the
        // module doc); kept with their expectations so a regression shows.
        // `q1_then_k1_gap0` and `q1_then_bigout_gap0` stay Open: with one
        // victim submit the verdict is whichever core the scheduler picks.
        // Four victim submits so the victim visits every core the
        // aggressors ran on (the scheduler alternates), not just one.
        Arm {
            name: "int8req_then_k1x4_gap0",
            aggressors: aggressors(OraclePrecision::Int8),
            victim: victim("k1"),
            victim_repeats: 4,
            gap: Gap::None,
            expect: Expect::Clean,
            bs_engage: false,
            od_engage: false,
        },
        Arm {
            name: "fp16_then_k1x4_gap0",
            aggressors: aggressors(OraclePrecision::Fp16),
            victim: victim("k1"),
            victim_repeats: 4,
            gap: Gap::None,
            expect: Expect::Clean,
            bs_engage: false,
            od_engage: false,
        },
        Arm {
            name: "q1_then_k1_gap0",
            aggressors: vec![aggressors(acc)[0]],
            victim: victim("k1"),
            victim_repeats: 1,
            gap: Gap::None,
            expect: Expect::Open,
            bs_engage: false,
            od_engage: false,
        },
        Arm {
            name: "q1_then_stem_gap0",
            aggressors: vec![aggressors(acc)[0]],
            victim: victim("stem"),
            victim_repeats: 1,
            gap: Gap::None,
            expect: Expect::Open,
            bs_engage: false,
            od_engage: false,
        },
        // Dose or core? One aggressor on one core, then the victim four
        // times as four submits. A later repeat hangs where the first was
        // clean: the state is per core and the "dose" was placement.
        Arm {
            name: "q1_then_k1x4_gap0",
            aggressors: vec![aggressors(acc)[0]],
            victim: victim("k1"),
            victim_repeats: 4,
            gap: Gap::None,
            expect: Expect::Clean,
            bs_engage: false,
            od_engage: false,
        },
        Arm {
            name: "q1_then_bigout_gap0",
            aggressors: vec![aggressors(acc)[0]],
            victim: victim("bigout"),
            victim_repeats: 1,
            gap: Gap::None,
            expect: Expect::Open,
            bs_engage: false,
            od_engage: false,
        },
        Arm {
            name: "q1_then_stemx3_gap0",
            aggressors: vec![aggressors(acc)[0]],
            victim: victim("stem"),
            victim_repeats: 3,
            gap: Gap::None,
            expect: Expect::Clean,
            bs_engage: false,
            od_engage: false,
        },
        Arm {
            name: "int8acc_then_bigout_gap0",
            aggressors: aggressors(acc),
            victim: victim("bigout"),
            victim_repeats: 1,
            gap: Gap::None,
            expect: Expect::Clean,
            bs_engage: false,
            od_engage: false,
        },
        // `bs_bypass` or `out_precision`? Accumulator mode is the only path
        // that sets `DPU_BS_CFG.bs_bypass`, and the only one that poisons, but
        // it also sets `DATA_FORMAT.out_precision = 4` -- the two are
        // confounded in the shipped program. These arms repeat the two
        // decisive ones above with the BS plane engaged as a pass-through
        // (`conv::set_accumulator_bs_engage`), which changes `BS_CFG` alone,
        // 0x141 -> 0x152.
        //
        // Answered 2026-09-05, both **hang 3/3 with every aggressor exact**:
        // clocking the BS plane does not clear the state, so `bs_bypass` is
        // not the cause. `fp16acc_then_k1x4_gap0` below eliminates it a second
        // way, from the clean side.
        //
        // The first version of this arm returned an all-zero buffer
        // (103296/131072 wrong, every non-zero lane zeroed). That was our bug,
        // not the hardware's: `bs_mul_bypass` bypasses the *multiply* and not
        // the shift after it, so the shipped `BS_MUL_SHIFT_VALUE` of 14 was
        // still right-shifting each accumulator into zero. The pass-through
        // zeroes the shift in both `BS_MUL_CFG` and
        // `DATA_FORMAT.bs_mul_shift_value_neg`, and the aggressors are exact.
        Arm {
            name: "q1_bsengage_then_k1x4_gap0",
            aggressors: vec![aggressors(acc)[0]],
            victim: victim("k1"),
            victim_repeats: 4,
            gap: Gap::None,
            expect: Expect::Clean,
            bs_engage: true,
            od_engage: false,
        },
        Arm {
            name: "int8acc_bsengage_then_k1_gap0",
            aggressors: aggressors(acc),
            victim: victim("k1"),
            victim_repeats: 1,
            gap: Gap::None,
            expect: Expect::Clean,
            bs_engage: true,
            od_engage: false,
        },
        // int32 specifically, or any output element wider than the fp16 the
        // victim writes? `Fp16Accumulator` is fp16 in / fp32 out: a 4-byte
        // output writer like the accumulator, with a different
        // `out_precision` (5, not 4), `bs_bypass = 0` and `od_bypass = 1`.
        //
        // Answered 2026-09-05: **clean 3/3**, with the victim landing on both
        // c0 and c1 across its four submits, so this is not the scheduler
        // missing a poisoned core. A 4-byte output writer does not poison, so
        // C8 is the int32 writer and not output width. Together with the
        // `bs_engage` arms that leaves `DATA_FORMAT.out_precision = 4` as the
        // only field of the four that separated the poisoner still standing:
        // `bs_bypass` is 0 here and 1 there, `od_bypass` is 1 in both, and
        // `size_e` was already measured inert on the accumulator path.
        //
        // The oracle has no readback for fp32 output, so this arm asserts that
        // the aggressors *ran* (1.6-3.7 ms dispatches) and did not poison, not
        // that their output was right.
        // The last untested combination: `out_precision = 4` with the output
        // converter **in the path**. `od_bypass` was already eliminated as a
        // discriminator (it is 1 in the poisoner and in two clean paths), but
        // no arm had run the int32 writer with the OW stage engaged.
        //
        // Safe only at the accumulator's own `size_e` of 7. Characterized
        // first with `accumulator_size_e_probe` behind `ROCKET_PAD_OUTPUT`
        // (32x32 Cin 64 Cout 128 k1): `od_engage` alone and `od_engage` with
        // `bs_engage` are both 0 mismatches, 100% written, `past_end` 0, 17
        // ms. Forcing `size_e = 3` there instead writes 1024 of 524288 bytes
        // and takes a 540 ms watchdog kill -- so `size_e` is gated by
        // `od_bypass`, not `bs_bypass`, and the notes' "integer outputs stride
        // as `size_e = 7`" quirk is exactly what the engaged OW stage wants.
        //
        // Answered 2026-09-05: both **hang 3/3, every aggressor exact**. The
        // int32 writer poisons whatever the BS and OW stages are doing, so
        // `out_precision = 4` is the cause on its own.
        Arm {
            name: "q1_odengage_then_k1x4_gap0",
            aggressors: vec![aggressors(acc)[0]],
            victim: victim("k1"),
            victim_repeats: 4,
            gap: Gap::None,
            expect: Expect::Clean,
            bs_engage: false,
            od_engage: true,
        },
        // Both stages engaged: the closest an int32-output program gets to the
        // structure of the clean requantized one, differing in
        // `out_precision`, `size_e` and `surf_add`.
        Arm {
            name: "q1_odbsengage_then_k1x4_gap0",
            aggressors: vec![aggressors(acc)[0]],
            victim: victim("k1"),
            victim_repeats: 4,
            gap: Gap::None,
            expect: Expect::Clean,
            bs_engage: true,
            od_engage: true,
        },
        Arm {
            name: "fp16acc_then_k1x4_gap0",
            aggressors: aggressors(OraclePrecision::Fp16Accumulator),
            victim: victim("k1"),
            victim_repeats: 4,
            gap: Gap::None,
            expect: Expect::Clean,
            bs_engage: false,
            od_engage: false,
        },
    ]
}

struct ArmOutcome {
    /// Every core's runtime status right before the victim submit.
    status_before_victim: String,
    started_suspended: bool,
    gap: Duration,
    aggressor_dispatches: Vec<Duration>,
    aggressor_cores: Vec<String>,
    /// One per victim submit, in order.
    victim_dispatches: Vec<Duration>,
    victim_cores: Vec<String>,
    victim_mismatches: Result<usize, String>,
    aggressor_mismatches: Vec<Result<usize, String>>,
}

impl ArmOutcome {
    fn hung(&self) -> bool {
        self.victim_dispatches
            .iter()
            .any(|d| *d >= DISPATCH_TIMEOUT_FLOOR)
            || self
                .aggressor_dispatches
                .iter()
                .any(|d| *d >= DISPATCH_TIMEOUT_FLOOR)
    }
}

fn run_arm(
    file: &std::fs::File,
    paths: &[std::path::PathBuf],
    arm: &Arm,
) -> Result<ArmOutcome, String> {
    let fd = file.as_raw_fd();
    // Before `prepare`, which is what builds the register program.
    iree_rocket_hal::rocket::conv::set_accumulator_bs_engage(arm.bs_engage);
    iree_rocket_hal::rocket::conv::set_od_engage(arm.od_engage);
    // Everything that costs host time happens here, before the clock matters.
    let aggressors = arm
        .aggressors
        .iter()
        .map(|case| unsafe { PreparedJob::prepare(file, *case) })
        .collect::<Result<Vec<_>, _>>()?;
    let victim = unsafe { PreparedJob::prepare(file, arm.victim)? };
    for job in &aggressors {
        println!("    aggressor {}", job.label());
    }
    println!("    victim    {}", victim.label());

    std::thread::sleep(SETTLE);
    let (_, started_suspended) = wait_all_suspended(paths, SUSPEND_BUDGET);

    let mut aggressor_dispatches = Vec::with_capacity(aggressors.len());
    let mut aggressor_cores = Vec::with_capacity(aggressors.len());
    for job in &aggressors {
        let (elapsed, core) = unsafe { job.submit(fd)? };
        aggressor_dispatches.push(elapsed);
        aggressor_cores.push(core);
    }

    let gap_started = Instant::now();
    match arm.gap {
        Gap::None => {}
        Gap::Sleep(duration) => std::thread::sleep(duration),
        Gap::UntilSuspended => {
            wait_all_suspended(paths, SUSPEND_BUDGET);
        }
    }
    let status_before_victim = runtime_status_summary(paths);
    let gap = gap_started.elapsed();

    let mut victim_dispatches = Vec::with_capacity(arm.victim_repeats);
    let mut victim_cores = Vec::with_capacity(arm.victim_repeats);
    for _ in 0..arm.victim_repeats.max(1) {
        let (elapsed, core) = unsafe { victim.submit(fd)? };
        victim_dispatches.push(elapsed);
        victim_cores.push(core);
        if elapsed >= DISPATCH_TIMEOUT_FLOOR {
            // A killed job resets the core; anything after it is a
            // different experiment.
            break;
        }
    }

    Ok(ArmOutcome {
        status_before_victim,
        started_suspended,
        gap,
        aggressor_dispatches,
        aggressor_cores,
        victim_dispatches,
        victim_cores,
        victim_mismatches: victim.mismatches(),
        aggressor_mismatches: aggressors.iter().map(PreparedJob::mismatches).collect(),
    })
}

/// A wide fp16 job alone, from a suspended domain: the thing that must be
/// clean before the next arm's row means anything.
fn canary_passes(file: &std::fs::File, paths: &[std::path::PathBuf]) -> bool {
    let fd = file.as_raw_fd();
    for extra in [0u64, 3, 8] {
        std::thread::sleep(SETTLE + Duration::from_secs(extra));
        wait_all_suspended(paths, SUSPEND_BUDGET);
        let job = match unsafe { PreparedJob::prepare(file, victim("k1")) } {
            Ok(job) => job,
            Err(_) => return false,
        };
        match unsafe { job.submit(fd) }.map(|(elapsed, _)| elapsed) {
            Ok(elapsed) if elapsed < DISPATCH_TIMEOUT_FLOOR => {
                if job.mismatches() == Ok(0) {
                    println!(
                        "    canary: healthy ({:.1} ms)",
                        elapsed.as_secs_f64() * 1e3
                    );
                    return true;
                }
                println!("    canary: wrong values, retrying");
            }
            Ok(elapsed) => println!(
                "    canary: SICK ({:.0} ms), retrying after a longer gap",
                elapsed.as_secs_f64() * 1e3
            ),
            Err(error) => println!("    canary: {error}, retrying"),
        }
    }
    false
}

fn selected_arms() -> Vec<Arm> {
    let all = arms();
    match std::env::var("ROCKET_C8_ARMS") {
        Ok(list) if !list.is_empty() => {
            let wanted: Vec<&str> = list.split(',').map(str::trim).collect();
            let selected: Vec<Arm> = all
                .into_iter()
                .filter(|arm| wanted.contains(&arm.name))
                .collect();
            assert!(!selected.is_empty(), "ROCKET_C8_ARMS matched no arm");
            selected
        }
        _ => all,
    }
}

fn repeat() -> usize {
    std::env::var("ROCKET_C8_REPEAT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(2)
}

fn ms(duration: Duration) -> String {
    format!("{:.1}", duration.as_secs_f64() * 1e3)
}

/// The C8 transition, arm by arm, with the gap and the power domain as the
/// variables.
///
/// Arms with a `Hang`/`Clean` expectation are the mechanism's predictions and
/// a wrong one fails the test; `Open` arms are questions and only print.
#[test]
#[ignore = "needs /dev/accel/accel0 -- see the module doc comment"]
fn int8acc_to_fp16_transition_no_longer_poisons_a_core() {
    let _guard = NPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("open /dev/accel/accel0");
    let paths = npu_runtime_status_paths();
    println!(
        "runtime PM: {} NPU core(s) visible, status now {}",
        paths.len(),
        runtime_status_summary(&paths)
    );

    let arms = selected_arms();
    let repeat = repeat();
    let mut rows: Vec<String> = Vec::new();
    let mut wrong: Vec<String> = Vec::new();
    let mut device_dead = false;

    'arms: for arm in &arms {
        for trial in 1..=repeat {
            println!(
                "\n== {} trial {trial} (gap {:?}, expect {:?})",
                arm.name, arm.gap, arm.expect
            );
            let outcome = match run_arm(&file, &paths, arm) {
                Ok(outcome) => outcome,
                Err(error) => {
                    println!("    unrunnable: {error}");
                    rows.push(format!(
                        "{:<28} trial {trial}  UNRUNNABLE  {error}",
                        arm.name
                    ));
                    continue;
                }
            };
            let verdict = if outcome.hung() { "HUNG" } else { "ok" };
            let victim = match &outcome.victim_mismatches {
                Ok(0) => "exact".to_string(),
                Ok(n) => format!("{n} wrong"),
                Err(e) => e.clone(),
            };
            let aggressors = outcome
                .aggressor_mismatches
                .iter()
                .map(|m| match m {
                    Ok(0) => "exact".to_string(),
                    Ok(n) => format!("{n} wrong"),
                    Err(e) => e.clone(),
                })
                .collect::<Vec<_>>()
                .join("/");
            let with_cores = |dispatches: &[Duration], cores: &[String]| {
                dispatches
                    .iter()
                    .zip(cores)
                    .map(|(d, core)| format!("{}@{core}", ms(*d)))
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            let row = format!(
                "{:<28} trial {trial}  {verdict:<5} victim [{}] ms ({victim})  aggressors [{}] ms ({aggressors})  \
                 gap {} ms  status before victim {}  started suspended {}",
                arm.name,
                with_cores(&outcome.victim_dispatches, &outcome.victim_cores),
                with_cores(&outcome.aggressor_dispatches, &outcome.aggressor_cores),
                ms(outcome.gap),
                outcome.status_before_victim,
                outcome.started_suspended,
            );
            println!("    {row}");
            let unexpected = match arm.expect {
                Expect::Hang => !outcome.hung(),
                Expect::Clean => outcome.hung(),
                Expect::Open => false,
            };
            if unexpected {
                wrong.push(row.clone());
            }
            rows.push(row);

            if outcome.hung() && !canary_passes(&file, &paths) {
                println!(
                    "    the canary did not recover; stopping, later rows would measure the device"
                );
                device_dead = true;
                break 'arms;
            }
        }
    }

    println!("\n== summary ==");
    for row in &rows {
        println!("{row}");
    }
    assert!(
        !device_dead,
        "the device did not recover after a hang; reboot before reading anything above"
    );
    assert!(
        wrong.is_empty(),
        "{} arm(s) contradicted the mechanism's prediction:\n  {}",
        wrong.len(),
        wrong.join("\n  ")
    );
}
