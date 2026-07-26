//! Wall-clock comparison of height-tiled versus single-job convolution.
//!
//! This test is ignored on the development host because it needs the RK3588
//! NPU device. Cross-compile it, copy the printed test binary to the board,
//! and run it there:
//!
//!   CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!     cargo test --target aarch64-unknown-linux-gnu --release \
//!       --test conv_tiling_bench_hw --no-run
//!
//!   ./conv_tiling_bench_hw-<hash> --ignored --nocapture
//!
//! Every configuration computes the identical `32x32x3 -> 32x32x8` fp16
//! convolution over the whole image. Only the number of jobs the work is
//! split into varies: 1 job covering 32 output rows, 2 jobs covering 16+16,
//! or 3 jobs covering 11+11+10, all submitted in a single
//! `drm_rocket_submit`. Total work is therefore held constant and tile count
//! is the only independent variable.
//!
//! # What is timed
//!
//! The timed region is submit-to-completion: the SUBMIT ioctl plus the
//! PREP_BO wait on the output buffer. GEM allocation, regcmd construction,
//! and address relocation all happen once, before timing, because they are
//! properties of the caller rather than of the hardware. FINI_BO between
//! iterations is outside the timed region and is paid identically by every
//! configuration.
//!
//! # Ordering
//!
//! All six configurations are built up front and their timed iterations are
//! interleaved round-robin. An earlier version ran every iteration of one
//! configuration before starting the next, which made whichever ran first
//! absorb the CPU-governor and NPU devfreq ramp. That bias was large enough
//! to report 1x1 as slower than 3x3 at equal job counts -- impossible, since
//! 1x1 does a ninth of the taps, loads a ninth of the weights, and reads no
//! halo rows. Round-robin spreads any remaining drift evenly across
//! configurations, so the comparison holds even while absolute numbers move.
//!
//! # What to expect
//!
//! This convolution is small. The whole-image 3x3 case is roughly
//! `32 * 32 * 9 = 9216` array cycles, under ten microseconds of compute,
//! against observed per-operation latencies in the tens of microseconds.
//! The workload is dominated by fixed per-job cost, not by compute, so
//! splitting it three ways triples that fixed cost while dividing compute
//! by three. Tiling is not expected to win at this size, and a result
//! showing it losing is a real result, not a failure.
//!
//! Read this as a measurement of where the crossover is, not as a
//! recommendation. The interesting number is how much per-job overhead
//! tiling has to overcome before it pays.

use std::{
    fs::OpenOptions,
    mem,
    os::unix::io::AsRawFd,
    ptr,
    time::{Duration, Instant},
};

use iree_rocket_hal::rocket::{
    builders::{
        RegCmd, RegisterMeta,
        cna::{CnaDcompAddr0, CnaFeatureDataAddr},
        dpu::DpuDstBaseAddr,
        dpu_rdma::DpuRdmaBsBaseAddr,
    },
    conv::{Kernels, Shape, Tile, conv_2d_tile},
    device::{Buffer, JobDesc, close_bo, fini_bo, prep_bo, submit_jobs},
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const WIDTH: usize = 32;
const HEIGHT: usize = 32;
const INPUT_CHANNELS: usize = 3;
const WEIGHT_INPUT_CHANNELS: usize = 8;
const OUTPUT_CHANNELS: usize = 8;
const FP16_BYTES: usize = 2;
const FEATURE_ATOM_BYTES: usize = 16;
const PAGE_BYTES: usize = 4096;

const INPUT_BYTES: usize = WIDTH * HEIGHT * INPUT_CHANNELS * FP16_BYTES;
const OUTPUT_BYTES: usize = WIDTH * HEIGHT * FEATURE_ATOM_BYTES * 2;

const FP16_ONE: u16 = 0x3c00;

const WARMUP_ROUNDS: usize = 200;
const ITERATIONS: usize = 500;
const TIMEOUT_NS: u64 = 2_000_000_000;

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

fn valid_taps(coordinate: usize, extent: usize, kernel: usize) -> usize {
    match kernel {
        1 => 1,
        3 => 3 - usize::from(coordinate == 0) - usize::from(coordinate + 1 == extent),
        _ => unreachable!("conv_2d_tile rejects kernels other than 1x1 and 3x3"),
    }
}

fn expected_output(kernels: Kernels, y: usize, x: usize) -> f32 {
    (INPUT_CHANNELS * valid_taps(y, HEIGHT, kernels[0]) * valid_taps(x, WIDTH, kernels[1])) as f32
}

struct Stats {
    min: Duration,
    median: Duration,
    p90: Duration,
}

fn summarise(mut samples: Vec<Duration>) -> Stats {
    samples.sort_unstable();
    Stats {
        min: samples[0],
        median: samples[samples.len() / 2],
        p90: samples[samples.len() * 9 / 10],
    }
}

/// One configuration: every buffer and job descriptor needed to run the
/// whole convolution split across `tiles` jobs.
///
/// All configurations are constructed up front and kept alive together so
/// their timed iterations can be interleaved.
struct Config {
    kernels: Kernels,
    tiles: u32,
    buffers: Vec<Buffer>,
    output_handle: u32,
    output_ptr: *mut u8,
    tasks: Vec<[(u32, u32); 1]>,
    in_handles: Vec<[u32; 4]>,
    out_handles: [u32; 1],
}

impl Config {
    unsafe fn new(fd: i32, file: &std::fs::File, kernels: Kernels, tiles: u32) -> Config {
        unsafe {
            let buf_input = Buffer::new(fd, page_aligned_size(INPUT_BYTES), file);
            ptr::write_bytes(buf_input.host_ptr, 0, buf_input.size);
            std::slice::from_raw_parts_mut(
                buf_input.host_ptr as *mut u16,
                INPUT_BYTES / FP16_BYTES,
            )
            .fill(FP16_ONE);

            let weight_bytes =
                kernels[0] * kernels[1] * WEIGHT_INPUT_CHANNELS * OUTPUT_CHANNELS * FP16_BYTES;
            let buf_weights = Buffer::new(fd, page_aligned_size(weight_bytes), file);
            ptr::write_bytes(buf_weights.host_ptr, 0, buf_weights.size);
            std::slice::from_raw_parts_mut(buf_weights.host_ptr as *mut u16, weight_bytes / 2)
                .fill(FP16_ONE);

            let buf_bias = Buffer::new(fd, PAGE_BYTES, file);
            ptr::write_bytes(buf_bias.host_ptr, 0, buf_bias.size);

            let buf_output = Buffer::new(fd, OUTPUT_BYTES, file);
            ptr::write_bytes(buf_output.host_ptr, 0, buf_output.size);

            let output_handle = buf_output.handle;
            let output_ptr = buf_output.host_ptr;

            let mut buffers = vec![buf_input, buf_weights, buf_bias, buf_output];
            let (input_addr, weights_addr, bias_addr, output_addr) = (
                buffers[0].dma_address,
                buffers[1].dma_address,
                buffers[2].dma_address,
                buffers[3].dma_address,
            );
            let (input_handle, weights_handle, bias_handle) =
                (buffers[0].handle, buffers[1].handle, buffers[2].handle);

            // Regcmd construction and relocation are caller costs, not
            // hardware costs, and must not land inside the timed region.
            let mut tasks = Vec::new();
            let mut in_handles = Vec::new();
            for tile in &Tile::split(Shape::CAPTURED, kernels, tiles) {
                let mut commands = conv_2d_tile(Shape::CAPTURED, kernels, tile);
                relocate::<CnaFeatureDataAddr>(&mut commands, input_addr);
                relocate::<CnaDcompAddr0>(&mut commands, weights_addr);
                relocate::<DpuRdmaBsBaseAddr>(&mut commands, bias_addr);
                relocate::<DpuDstBaseAddr>(&mut commands, output_addr);

                let command_bytes = commands.len() * mem::size_of::<u64>();
                let buffer = Buffer::new(fd, page_aligned_size(command_bytes), file);
                ptr::write_bytes(buffer.host_ptr, 0, buffer.size);
                let words =
                    std::slice::from_raw_parts_mut(buffer.host_ptr as *mut u64, commands.len());
                for (destination, command) in words.iter_mut().zip(&commands) {
                    *destination = command.0;
                }

                tasks.push([(buffer.dma_address, commands.len() as u32)]);
                in_handles.push([buffer.handle, input_handle, weights_handle, bias_handle]);
                fini_bo(fd, buffer.handle).expect("failed to sync regcmd BO");
                buffers.push(buffer);
            }

            fini_bo(fd, input_handle).expect("failed to sync input BO");
            fini_bo(fd, weights_handle).expect("failed to sync weight BO");
            fini_bo(fd, bias_handle).expect("failed to sync bias BO");

            Config {
                kernels,
                tiles,
                buffers,
                output_handle,
                output_ptr,
                tasks,
                in_handles,
                out_handles: [output_handle],
            }
        }
    }

    /// Runs one submit-to-completion cycle, returning its wall time when
    /// `timed`. Building the job descriptors is deliberately outside the
    /// timed region.
    unsafe fn run(&self, fd: i32, timed: bool) -> Option<Duration> {
        unsafe {
            let jobs: Vec<JobDesc<'_>> = self
                .tasks
                .iter()
                .zip(&self.in_handles)
                .map(|(tasks, in_handles)| JobDesc {
                    tasks,
                    in_handles,
                    out_handles: &self.out_handles,
                })
                .collect();

            fini_bo(fd, self.output_handle).expect("failed to sync output BO for the NPU");
            let start = Instant::now();
            submit_jobs(fd, &jobs).unwrap_or_else(|error| {
                panic!(
                    "{:?} {}-tile SUBMIT failed: {error}",
                    self.kernels, self.tiles
                )
            });
            prep_bo(fd, self.output_handle, TIMEOUT_NS).unwrap_or_else(|error| {
                panic!(
                    "{:?} {}-tile did not complete in two seconds: {error}",
                    self.kernels, self.tiles
                )
            });
            timed.then(|| start.elapsed())
        }
    }

    /// A fast wrong answer is not a result.
    unsafe fn verify(&self) {
        unsafe {
            let raw = std::slice::from_raw_parts(self.output_ptr, OUTPUT_BYTES);
            for y in 0..HEIGHT {
                for x in 0..WIDTH {
                    let expected = expected_output(self.kernels, y, x);
                    for channel in 0..OUTPUT_CHANNELS {
                        let offset = (y * WIDTH + x) * FEATURE_ATOM_BYTES + channel * FP16_BYTES;
                        let actual = f16_to_f32(u16::from_le_bytes([raw[offset], raw[offset + 1]]));
                        assert_eq!(
                            actual, expected,
                            "{:?} {}-tile wrong at [{y}, {x}, {channel}] -- \
                             benchmark aborted, timings would be meaningless",
                            self.kernels, self.tiles
                        );
                    }
                }
            }
        }
    }

    unsafe fn close(&self, fd: i32) {
        unsafe {
            for buffer in &self.buffers {
                close_bo(fd, buffer.handle).expect("failed to close BO");
            }
        }
    }
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board; \
            diagnostic only, read the printed table"]
fn tiling_versus_single_job_wall_clock() {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();

    unsafe {
        let configs: Vec<Config> = [[1usize, 1], [3, 3]]
            .into_iter()
            .flat_map(|kernels| [1u32, 2, 3].map(move |tiles| (kernels, tiles)))
            .map(|(kernels, tiles)| Config::new(fd, &file, kernels, tiles))
            .collect();

        for config in &configs {
            config.run(fd, false);
            config.verify();
        }

        // Interleaved warmup and measurement. Running every iteration of one
        // configuration before starting the next makes the first one absorb
        // CPU-governor and NPU devfreq ramp, which is large enough here to
        // invert the true ordering. Round-robin spreads any drift evenly.
        for _ in 0..WARMUP_ROUNDS {
            for config in &configs {
                config.run(fd, false);
            }
        }

        let mut samples: Vec<Vec<Duration>> = vec![Vec::with_capacity(ITERATIONS); configs.len()];
        for _ in 0..ITERATIONS {
            for (index, config) in configs.iter().enumerate() {
                samples[index].push(
                    config
                        .run(fd, true)
                        .expect("timed run returned no duration"),
                );
            }
        }

        println!(
            "\n32x32x3 -> 32x32x8 fp16, whole image per configuration\n\
             {ITERATIONS} timed iterations after {WARMUP_ROUNDS} warmup rounds,\n\
             interleaved round-robin across configurations, submit-to-completion\n"
        );
        println!(
            "{:<8} {:>6} {:>12} {:>12} {:>12} {:>10}",
            "kernel", "jobs", "min", "median", "p90", "vs 1 job"
        );

        let mut baseline: Option<Duration> = None;
        for (config, samples) in configs.iter().zip(samples) {
            let stats = summarise(samples);
            if config.tiles == 1 {
                baseline = Some(stats.median);
            }
            let ratio = baseline
                .expect("1-job configuration must run first")
                .as_secs_f64()
                / stats.median.as_secs_f64();
            println!(
                "{:<8} {:>6} {:>10.1?} {:>10.1?} {:>10.1?} {:>9.2}x",
                format!("{}x{}", config.kernels[0], config.kernels[1]),
                config.tiles,
                stats.min,
                stats.median,
                stats.p90,
                ratio
            );
        }

        println!(
            "\n'vs 1 job' is median speedup over the single-job configuration\n\
             of the same kernel: above 1.00x means tiling is faster, below\n\
             means per-job cost outweighs the parallelism at this size.\n"
        );

        for config in &configs {
            config.close(fd);
        }
    }
}
