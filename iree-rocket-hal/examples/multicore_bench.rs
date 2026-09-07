//! Does opening `/dev/accel/accel0` N times buy N cores?
//!
//! `rocket-hal-driver/MULTICORE.md` §8 step 1: before any worker pool is
//! built, measure the hardware term on its own. One `open()` of the device
//! is one DRM scheduler entity, which occupies at most one NPU core at a
//! time (mainline `rocket_job.c`). This opens the device N times, gives
//! each file its own copy of one dispatch's buffers, and has N pinned
//! threads replay that dispatch as fast as the hardware completes it. The
//! notes' measurement (`../rockchip-npu-notes/perf/iova-and-multicore.md`)
//! is 1 / 2.13 / 2.94 / 3.06x jobs/s for 1..4 files on RK3588. If that is
//! not here, nothing in MULTICORE.md is worth building.
//!
//! Two shapes, both fp16, both from real models:
//!
//! - `conv`: MobileNetV2's `112x112x24 -> 144` 1x1 expansion, a multi-tile
//!   dispatch (ConvPlan splits it by CBUF rows; the tile count is printed).
//! - `fc`:   ViT-B/16's `197x768 @ 768x768` projection as an `fc::Plan`.
//!
//! Every worker replays production's exact submission pattern for one
//! dispatch: `submit` then a blocking `prep_bo` on the output BO, one tile at
//! a time, in order. One "job" below is one whole dispatch (all its tiles).
//!
//! Modes (MULTICORE.md §8 steps 1 and 2):
//!
//! - default: hardware term only -- submit + wait, buffers packed once.
//! - `--host-phases`: each job also re-packs the NHWC input into the
//!   context's NC1HWC2 input BO and compacts the NC1HWC2 output back to a
//!   dense heap buffer, the way `queue_execute` does per dispatch. This is
//!   the overlap measurement: with N files, worker A's pack/compact runs
//!   while worker B's job is on the NPU.
//! - `--shared-fd`: every context is a `try_clone()` of one file instead of
//!   its own `open()`. MULTICORE.md's claim is that a dup is the same
//!   scheduler entity and therefore *not* a second core; this arm measures
//!   that directly.
//!
//! Reported per arm: jobs/s and its ratio to the N=1 arm of the same pass,
//! mean per-context `wait` (time inside `prep_bo`), and `overlap`, the
//! fraction of wall during which two or more contexts had a job in flight.
//! Overlap is the number that says whether N > 1 did anything at all: N
//! files that serialise on one core score ~0 no matter what jobs/s says.
//!
//! Every context is verified once after warm-up (input all 1.0, weights all
//! 1.0, so every output element must equal Cin exactly); a wrong or unwritten
//! output aborts the run, since a sick device makes every number meaningless
//! (memory `npu-wedges-after-failed-job`).
//!
//! Board only. Read `planck-measurement-environment` first: quote the
//! governor and `taskset` with every number. Workers pin themselves to
//! `--cpus` (default `4,5,6,7`, the A76 cluster) so a `PREP_BO` wake never
//! lands on an A55; `--cpus none` leaves placement to the scheduler.
//!
//! ```sh
//! CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!   cargo build -p iree-rocket-hal --release \
//!   --target aarch64-unknown-linux-gnu --example multicore_bench
//! scp target/aarch64-unknown-linux-gnu/release/examples/multicore_bench planck:
//! ssh planck ./multicore_bench --shape conv --contexts 1,2,3,4 --jobs 200
//! ssh planck ./multicore_bench --shape fc --host-phases
//! ssh planck ./multicore_bench --shape conv --shared-fd
//! ```

use std::{
    fs::{File, OpenOptions},
    mem,
    os::fd::AsRawFd,
    ptr,
    sync::{Arc, Barrier},
    time::{Duration, Instant},
};

use iree_rocket_hal::rocket::{
    builders::RegCmd,
    conv::{Buffers, ConvPlan, Precision, Shape},
    device::{OwnedBuffer, fini_bo, prep_bo, submit},
    fc,
    tensor_layout::{
        compact_atomic_output, nc1hwc2_storage_size, pack_hwcf_to_rocket_weights,
        pack_nhwc_to_nc1hwc2_padded, rocket_weight_storage_size,
    },
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const PAGE_BYTES: usize = 4096;
const FP16_BYTES: usize = 2;
const FP16_ONE: u16 = 0x3c00;
/// The fp16 reading of `0xa5a5`, the never-written sentinel the hw tests use.
const OUTPUT_SENTINEL: u8 = 0xa5;
const PER_TILE_TIMEOUT_NS: u64 = 5_000_000_000;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Workload {
    Conv,
    Fc,
}

impl Workload {
    fn label(self) -> &'static str {
        match self {
            Workload::Conv => "conv 112x112x24->144 k1 fp16",
            Workload::Fc => "fc 197x768 @ 768x768 fp16",
        }
    }
}

struct Options {
    workloads: Vec<Workload>,
    contexts: Vec<usize>,
    jobs: usize,
    warmup: usize,
    passes: usize,
    host_phases: bool,
    shared_fd: bool,
    cpus: Option<Vec<usize>>,
}

fn usage() -> ! {
    eprintln!(
        "usage: multicore_bench [--shape conv|fc|both] [--contexts 1,2,3,4] [--jobs N] \
         [--warmup N] [--passes N] [--host-phases] [--shared-fd] [--cpus 4,5,6,7|none]"
    );
    std::process::exit(2);
}

fn parse_list(text: &str) -> Vec<usize> {
    text.split(',')
        .map(|item| item.trim().parse().unwrap_or_else(|_| usage()))
        .collect()
}

fn parse_options() -> Options {
    let mut options = Options {
        workloads: vec![Workload::Conv, Workload::Fc],
        contexts: vec![1, 2, 3, 4],
        jobs: 100,
        warmup: 5,
        passes: 3,
        host_phases: false,
        shared_fd: false,
        cpus: Some(vec![4, 5, 6, 7]),
    };
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut value = || args.next().unwrap_or_else(|| usage());
        match flag.as_str() {
            "--shape" => {
                options.workloads = match value().as_str() {
                    "conv" => vec![Workload::Conv],
                    "fc" => vec![Workload::Fc],
                    "both" => vec![Workload::Conv, Workload::Fc],
                    _ => usage(),
                }
            }
            "--contexts" => options.contexts = parse_list(&value()),
            "--jobs" => options.jobs = value().parse().unwrap_or_else(|_| usage()),
            "--warmup" => options.warmup = value().parse().unwrap_or_else(|_| usage()),
            "--passes" => options.passes = value().parse().unwrap_or_else(|_| usage()),
            "--host-phases" => options.host_phases = true,
            "--shared-fd" => options.shared_fd = true,
            "--cpus" => {
                let text = value();
                options.cpus = (text != "none").then(|| parse_list(&text));
            }
            _ => usage(),
        }
    }
    if options.contexts.iter().any(|&n| n == 0) || options.jobs == 0 {
        usage();
    }
    options
}

/// The logical dispatch a context replays, independent of any file.
struct Dispatch {
    /// Dense NHWC input the host-phases mode re-packs every job.
    dense_input: Vec<u8>,
    pixel_count: usize,
    input_bytes_per_pixel: usize,
    /// Production's programmed pixel width (`command_buffer.rs`).
    packed_input_bytes_per_pixel: usize,
    packed_input_bytes: usize,
    packed_weights: Vec<u8>,
    bias_bytes: usize,
    output_bytes: usize,
    output_pixel_count: usize,
    output_bytes_per_pixel: usize,
    output_block_bytes: usize,
    /// Every fp16 output element must read back as this.
    expected_output: f32,
    /// Relocatable programs, one per tile; bound per context.
    programs: Vec<Vec<RegCmd>>,
}

impl Dispatch {
    fn new(workload: Workload) -> Dispatch {
        let (conv_shape, kernels, fc_plan) = match workload {
            Workload::Conv => (
                Shape::with_precision(112, 112, 1, 24, 144, Precision::Fp16),
                [1usize, 1usize],
                None,
            ),
            Workload::Fc => {
                let shape = fc::Shape::new(197, 768, 768, Precision::Fp16);
                (
                    shape.as_conv_shape(),
                    [1usize, 1usize],
                    Some(fc::Plan::new(shape)),
                )
            }
        };
        let cin = conv_shape.in_channels as usize;
        let cout = conv_shape.out_channels as usize;
        let pixel_count = conv_shape.width as usize * conv_shape.height as usize;
        let input_bytes_per_pixel = cin * FP16_BYTES;
        let packed_input_bytes_per_pixel = cin.max(16).next_multiple_of(16) * FP16_BYTES;
        let packed_input_bytes =
            nc1hwc2_storage_size(pixel_count, packed_input_bytes_per_pixel).unwrap();

        let dense_input: Vec<u8> = (0..pixel_count * cin)
            .flat_map(|_| FP16_ONE.to_le_bytes())
            .collect();

        let dense_weights: Vec<u8> = (0..kernels[0] * kernels[1] * cin * cout)
            .flat_map(|_| FP16_ONE.to_le_bytes())
            .collect();
        let weight_bytes =
            rocket_weight_storage_size(kernels[0], kernels[1], cin, cout, FP16_BYTES).unwrap();
        let mut packed_weights = vec![0u8; weight_bytes];
        pack_hwcf_to_rocket_weights(
            &dense_weights,
            kernels[0],
            kernels[1],
            cin,
            cout,
            FP16_BYTES,
            &mut packed_weights,
        )
        .unwrap();

        let padded_out = conv_shape.padded_out_channels() as usize;
        let bias_bytes = conv_shape.bs_buffer_bytes().max(padded_out * 4);
        let output_pixel_count =
            conv_shape.output_width(kernels) as usize * conv_shape.output_height(kernels) as usize;
        let output_bytes = conv_shape.output_scratch_bytes(kernels);

        let programs = match fc_plan {
            Some(plan) => plan.programs(),
            None => ConvPlan::new(conv_shape, kernels).programs(),
        };

        Dispatch {
            dense_input,
            pixel_count,
            input_bytes_per_pixel,
            packed_input_bytes_per_pixel,
            packed_input_bytes,
            packed_weights,
            bias_bytes,
            output_bytes,
            output_pixel_count,
            output_bytes_per_pixel: cout * FP16_BYTES,
            output_block_bytes: conv_shape.output_atom_bytes() as usize,
            expected_output: cin as f32,
            programs,
        }
    }
}

fn page_aligned(size: usize) -> usize {
    size.div_ceil(PAGE_BYTES) * PAGE_BYTES
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    let f32_bits = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            let subnormal = (frac as f32) * 2f32.powi(-24);
            return if sign == 1 { -subnormal } else { subnormal };
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (frac << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13)
    };
    f32::from_bits(f32_bits)
}

/// Everything one worker owns: a file and the BOs a job on it may name.
///
/// MULTICORE.md §2: a job submitted on file k may reference only BOs created
/// on file k. So every context carries its own input, weights, bias, output
/// and per-tile regcmd BOs -- exactly the `NpuContext` of §5.2.
struct Context {
    id: usize,
    file: File,
    input: OwnedBuffer,
    weights: OwnedBuffer,
    bias: OwnedBuffer,
    output: OwnedBuffer,
    regcmds: Vec<(OwnedBuffer, u32)>,
    /// Dense destination for the host-phases compaction.
    compacted: Vec<u8>,
}

// `OwnedBuffer` holds a raw host pointer; each context is used by exactly
// one worker thread, which is the whole point of the design under test.
unsafe impl Send for Context {}

impl Context {
    fn new(id: usize, file: File, dispatch: &Dispatch) -> Context {
        let fd = file.as_raw_fd();
        unsafe {
            let input = OwnedBuffer::new(fd, page_aligned(dispatch.packed_input_bytes), &file);
            ptr::write_bytes(input.host_ptr, 0, input.size);
            let packed =
                std::slice::from_raw_parts_mut(input.host_ptr, dispatch.packed_input_bytes);
            pack_nhwc_to_nc1hwc2_padded(
                &dispatch.dense_input,
                dispatch.pixel_count,
                dispatch.input_bytes_per_pixel,
                dispatch.packed_input_bytes_per_pixel,
                packed,
            )
            .unwrap();

            let weights = OwnedBuffer::new(fd, page_aligned(dispatch.packed_weights.len()), &file);
            ptr::write_bytes(weights.host_ptr, 0, weights.size);
            ptr::copy_nonoverlapping(
                dispatch.packed_weights.as_ptr(),
                weights.host_ptr,
                dispatch.packed_weights.len(),
            );

            let bias = OwnedBuffer::new(fd, page_aligned(dispatch.bias_bytes), &file);
            ptr::write_bytes(bias.host_ptr, 0, bias.size);

            let output = OwnedBuffer::new(fd, page_aligned(dispatch.output_bytes), &file);
            ptr::write_bytes(output.host_ptr, OUTPUT_SENTINEL, output.size);

            let buffers = Buffers {
                input: input.dma_address,
                weights: weights.dma_address,
                bias: bias.dma_address,
                output: output.dma_address,
            };
            let mut regcmds = Vec::with_capacity(dispatch.programs.len());
            for program in &dispatch.programs {
                let mut commands: Vec<RegCmd> =
                    program.iter().map(|command| RegCmd(command.0)).collect();
                iree_rocket_hal::rocket::conv::relocate(&mut commands, buffers);
                let bytes = commands.len() * mem::size_of::<u64>();
                let buffer = OwnedBuffer::new(fd, page_aligned(bytes), &file);
                ptr::write_bytes(buffer.host_ptr, 0, buffer.size);
                let words =
                    std::slice::from_raw_parts_mut(buffer.host_ptr as *mut u64, commands.len());
                for (word, command) in words.iter_mut().zip(commands.iter()) {
                    *word = command.0;
                }
                regcmds.push((buffer, commands.len() as u32));
            }

            for handle in [input.handle, weights.handle, bias.handle, output.handle] {
                fini_bo(fd, handle).expect("FINI_BO");
            }
            for (buffer, _) in &regcmds {
                fini_bo(fd, buffer.handle).expect("FINI_BO regcmd");
            }

            Context {
                id,
                file,
                input,
                weights,
                bias,
                output,
                regcmds,
                compacted: vec![0u8; dispatch.output_pixel_count * dispatch.output_bytes_per_pixel],
            }
        }
    }

    /// One job: production's per-tile submit + blocking prep_bo chain.
    /// Returns the in-flight interval of every tile and the phase totals, or
    /// the tile that failed.
    fn run_job(
        &mut self,
        dispatch: &Dispatch,
        host_phases: bool,
        intervals: &mut Vec<(Instant, Instant)>,
        phases: &mut Phases,
    ) -> Result<(), String> {
        let fd = self.file.as_raw_fd();
        unsafe {
            if host_phases {
                let started = Instant::now();
                let packed = std::slice::from_raw_parts_mut(
                    self.input.host_ptr,
                    dispatch.packed_input_bytes,
                );
                pack_nhwc_to_nc1hwc2_padded(
                    &dispatch.dense_input,
                    dispatch.pixel_count,
                    dispatch.input_bytes_per_pixel,
                    dispatch.packed_input_bytes_per_pixel,
                    packed,
                )
                .map_err(|err| err.to_string())?;
                fini_bo(fd, self.input.handle).map_err(|err| format!("FINI_BO input: {err}"))?;
                phases.pack += started.elapsed();
            }
            for (tile, (regcmd, count)) in self.regcmds.iter().enumerate() {
                let in_handles = [
                    regcmd.handle,
                    self.input.handle,
                    self.weights.handle,
                    self.bias.handle,
                ];
                let out_handles = [self.output.handle];
                let submitted = Instant::now();
                submit(fd, regcmd.dma_address, *count, &in_handles, &out_handles)
                    .map_err(|err| format!("ctx {} tile {tile}: SUBMIT failed: {err}", self.id))?;
                let waiting = Instant::now();
                phases.submit += waiting - submitted;
                prep_bo(fd, self.output.handle, PER_TILE_TIMEOUT_NS)
                    .map_err(|err| format!("ctx {} tile {tile}: PREP_BO failed: {err}", self.id))?;
                let done = Instant::now();
                phases.wait += done - waiting;
                intervals.push((submitted, done));
            }
            if host_phases {
                let started = Instant::now();
                let scratch =
                    std::slice::from_raw_parts(self.output.host_ptr, dispatch.output_bytes);
                let written = compact_atomic_output(
                    scratch,
                    dispatch.output_pixel_count,
                    dispatch.output_pixel_count,
                    dispatch.output_bytes_per_pixel,
                    dispatch.output_block_bytes,
                    &mut self.compacted,
                );
                if written != self.compacted.len() {
                    return Err(format!(
                        "ctx {}: compaction wrote {written} of {} bytes",
                        self.id,
                        self.compacted.len()
                    ));
                }
                phases.compact += started.elapsed();
            }
        }
        Ok(())
    }

    /// Compacts the output and checks every element against `expected`.
    fn verify(&mut self, dispatch: &Dispatch) -> Result<(), String> {
        let scratch =
            unsafe { std::slice::from_raw_parts(self.output.host_ptr, dispatch.output_bytes) };
        let written = compact_atomic_output(
            scratch,
            dispatch.output_pixel_count,
            dispatch.output_pixel_count,
            dispatch.output_bytes_per_pixel,
            dispatch.output_block_bytes,
            &mut self.compacted,
        );
        if written != self.compacted.len() {
            return Err(format!(
                "ctx {}: compaction wrote {written} of {} bytes",
                self.id,
                self.compacted.len()
            ));
        }
        let mut wrong = 0usize;
        let mut sentinel = 0usize;
        let mut first: Option<(usize, f32)> = None;
        for (index, pair) in self.compacted.chunks_exact(2).enumerate() {
            let bits = u16::from_le_bytes([pair[0], pair[1]]);
            let value = f16_to_f32(bits);
            if value != dispatch.expected_output {
                wrong += 1;
                if bits == u16::from_le_bytes([OUTPUT_SENTINEL, OUTPUT_SENTINEL]) {
                    sentinel += 1;
                }
                if first.is_none() {
                    first = Some((index, value));
                }
            }
        }
        if wrong == 0 {
            Ok(())
        } else {
            Err(format!(
                "ctx {}: {wrong} of {} outputs != {} ({sentinel} never written); first at element {:?}",
                self.id,
                self.compacted.len() / 2,
                dispatch.expected_output,
                first
            ))
        }
    }
}

#[derive(Default, Clone, Copy)]
struct Phases {
    pack: Duration,
    submit: Duration,
    wait: Duration,
    compact: Duration,
}

struct WorkerResult {
    phases: Phases,
    intervals: Vec<(Instant, Instant)>,
    /// Wall from the barrier release to this worker's last completion.
    finished: Instant,
    error: Option<String>,
}

fn pin_to_cpu(cpu: usize) -> Result<(), String> {
    use nix::{sched::CpuSet, unistd::Pid};
    let mut set = CpuSet::new();
    set.set(cpu).map_err(|err| format!("cpu {cpu}: {err}"))?;
    nix::sched::sched_setaffinity(Pid::from_raw(0), &set)
        .map_err(|err| format!("sched_setaffinity({cpu}): {err}"))
}

fn open_device() -> File {
    OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .unwrap_or_else(|err| {
            eprintln!("open {DEVICE_PATH}: {err} (run this on the board)");
            std::process::exit(1);
        })
}

/// Fraction of `[start, end]` during which at least two intervals overlap.
fn overlap_fraction(intervals: &[(Instant, Instant)], start: Instant, end: Instant) -> f64 {
    let mut edges: Vec<(Instant, i32)> = Vec::with_capacity(intervals.len() * 2);
    for &(begin, finish) in intervals {
        edges.push((begin, 1));
        edges.push((finish, -1));
    }
    edges.sort_by_key(|&(at, delta)| (at, delta));
    let mut depth = 0i32;
    let mut covered = Duration::ZERO;
    let mut previous = start;
    for (at, delta) in edges {
        if depth >= 2 {
            covered += at.saturating_duration_since(previous);
        }
        previous = at;
        depth += delta;
    }
    let wall = end.saturating_duration_since(start);
    if wall.is_zero() {
        0.0
    } else {
        covered.as_secs_f64() / wall.as_secs_f64()
    }
}

struct ArmResult {
    contexts: usize,
    jobs_per_second: f64,
    wall: Duration,
    mean_wait: Duration,
    mean_pack: Duration,
    mean_compact: Duration,
    overlap: f64,
}

fn run_arm(
    dispatch: &Arc<Dispatch>,
    contexts: usize,
    options: &Options,
) -> Result<ArmResult, String> {
    let base = open_device();
    let mut built = Vec::with_capacity(contexts);
    for id in 0..contexts {
        let file = if options.shared_fd {
            base.try_clone().map_err(|err| format!("dup: {err}"))?
        } else if id == 0 {
            base.try_clone().map_err(|err| format!("dup: {err}"))?
        } else {
            open_device()
        };
        built.push(Context::new(id, file, dispatch));
    }

    // Warm every context before timing: each core autosuspends on its own,
    // and the first job on a cold core pays the resume (MULTICORE.md §5.6).
    // Verification happens here too, so a sick device aborts before any
    // number is printed.
    for context in &mut built {
        let mut scratch_intervals = Vec::new();
        let mut scratch_phases = Phases::default();
        for _ in 0..options.warmup.max(1) {
            context.run_job(
                dispatch,
                options.host_phases,
                &mut scratch_intervals,
                &mut scratch_phases,
            )?;
        }
        context.verify(dispatch)?;
    }

    let barrier = Arc::new(Barrier::new(contexts + 1));
    let mut handles = Vec::with_capacity(contexts);
    for mut context in built {
        let barrier = Arc::clone(&barrier);
        let dispatch = Arc::clone(dispatch);
        let cpu = options
            .cpus
            .as_ref()
            .map(|cpus| cpus[context.id % cpus.len()]);
        let jobs = options.jobs;
        let host_phases = options.host_phases;
        handles.push(std::thread::spawn(move || {
            if let Some(cpu) = cpu {
                if let Err(err) = pin_to_cpu(cpu) {
                    eprintln!("warning: {err}");
                }
            }
            let mut intervals = Vec::with_capacity(jobs * dispatch.programs.len());
            let mut phases = Phases::default();
            let mut error = None;
            barrier.wait();
            for _ in 0..jobs {
                if let Err(err) =
                    context.run_job(&dispatch, host_phases, &mut intervals, &mut phases)
                {
                    error = Some(err);
                    break;
                }
            }
            let finished = Instant::now();
            // Check the last concurrent job too: warm-up ran each context
            // alone, and a placement bug would only show with N in flight.
            if error.is_none() {
                error = context.verify(&dispatch).err();
            }
            // Drop the context on its own thread, after timing.
            drop(context);
            WorkerResult {
                phases,
                intervals,
                finished,
                error,
            }
        }));
    }

    barrier.wait();
    let started = Instant::now();
    let mut results = Vec::with_capacity(contexts);
    for handle in handles {
        results.push(handle.join().map_err(|_| "worker panicked".to_string())?);
    }
    if let Some(err) = results.iter().find_map(|result| result.error.clone()) {
        return Err(err);
    }
    let end = results.iter().map(|result| result.finished).max().unwrap();
    let wall = end - started;
    let all_intervals: Vec<(Instant, Instant)> = results
        .iter()
        .flat_map(|result| result.intervals.iter().copied())
        .collect();
    let mean = |pick: fn(&Phases) -> Duration| {
        results
            .iter()
            .map(|result| pick(&result.phases))
            .sum::<Duration>()
            / contexts as u32
    };
    Ok(ArmResult {
        contexts,
        jobs_per_second: (contexts * options.jobs) as f64 / wall.as_secs_f64(),
        wall,
        mean_wait: mean(|phases| phases.wait),
        mean_pack: mean(|phases| phases.pack),
        mean_compact: mean(|phases| phases.compact),
        overlap: overlap_fraction(&all_intervals, started, end),
    })
}

fn read_sys(path: &str) -> String {
    std::fs::read_to_string(path)
        .map(|text| text.trim().to_string())
        .unwrap_or_else(|_| "?".to_string())
}

fn main() {
    let options = parse_options();
    println!(
        "governor cpu4={} min={} kHz; cpu0={} min={} kHz",
        read_sys("/sys/devices/system/cpu/cpu4/cpufreq/scaling_governor"),
        read_sys("/sys/devices/system/cpu/cpu4/cpufreq/scaling_min_freq"),
        read_sys("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        read_sys("/sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq"),
    );
    println!(
        "mode: {}{}; workers pinned to {:?}; {} jobs/context, {} warm-up, {} passes",
        if options.host_phases {
            "host phases (pack + submit + wait + compact)"
        } else {
            "hardware term (submit + wait)"
        },
        if options.shared_fd {
            ", shared fd (dups of one open())"
        } else {
            ""
        },
        options.cpus,
        options.jobs,
        options.warmup,
        options.passes,
    );

    let mut failed = false;
    for workload in &options.workloads {
        let dispatch = Arc::new(Dispatch::new(*workload));
        println!(
            "\n== {} : {} tile(s) per job, input {} KB, weights {} KB, output {} KB ==",
            workload.label(),
            dispatch.programs.len(),
            dispatch.packed_input_bytes / 1024,
            dispatch.packed_weights.len() / 1024,
            dispatch.output_bytes / 1024,
        );
        println!(
            "{:>4} {:>5} {:>9} {:>9} {:>8} {:>11} {:>10} {:>10} {:>8}",
            "pass",
            "N",
            "wall ms",
            "jobs/s",
            "vs N=1",
            "job ms/ctx",
            "wait ms",
            "pack ms",
            "overlap"
        );
        for pass in 0..options.passes {
            let mut baseline: Option<f64> = None;
            for &contexts in &options.contexts {
                match run_arm(&dispatch, contexts, &options) {
                    Ok(result) => {
                        if contexts == 1 || baseline.is_none() {
                            baseline.get_or_insert(result.jobs_per_second);
                        }
                        let ratio = result.jobs_per_second / baseline.unwrap();
                        println!(
                            "{:>4} {:>5} {:>9.1} {:>9.1} {:>7.2}x {:>11.3} {:>10.3} {:>10.3} {:>7.0}%",
                            pass,
                            result.contexts,
                            result.wall.as_secs_f64() * 1e3,
                            result.jobs_per_second,
                            ratio,
                            result.wall.as_secs_f64() * 1e3 / options.jobs as f64,
                            result.mean_wait.as_secs_f64() * 1e3 / options.jobs as f64,
                            (result.mean_pack + result.mean_compact).as_secs_f64() * 1e3
                                / options.jobs as f64,
                            result.overlap * 100.0,
                        );
                    }
                    Err(err) => {
                        println!("{:>4} {:>5} FAILED: {err}", pass, contexts);
                        failed = true;
                    }
                }
            }
        }
    }
    println!(
        "\n'job ms/ctx' is wall / jobs: the time one context sees per dispatch. 'wait ms' is \
         the mean per-job time inside PREP_BO per context; 'pack ms' is pack + compact per job \
         (host-phases mode only). 'overlap' is the share of wall with >= 2 jobs in flight."
    );
    if failed {
        std::process::exit(1);
    }
}
