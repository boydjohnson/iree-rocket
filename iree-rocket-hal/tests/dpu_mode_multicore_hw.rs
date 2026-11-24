//! Is the depthwise -> dense DPU hazard per core, or per device?
//!
//! `rocket-hal-driver` dwells 1 ms between a depthwise completion and the
//! next dense submit (`DEPTHWISE_TO_DENSE_QUIESCENCE`): after the fence
//! the DPU can still hold depthwise write-back state, and a dense job
//! started into it intermittently writes only its first 16 channels. With
//! several NPU contexts (MULTICORE.md §5.6) the driver applies that rule
//! device-wide, because userspace cannot see which core a job ran on --
//! and it assumes, without measurement, that a depthwise job on core A
//! cannot corrupt a dense job on core B. This test measures that.
//!
//! Three files, three threads, no dwell anywhere:
//!
//! - context 0 submits only depthwise jobs, contexts 1 and 2 only dense
//!   ones, each keeping its queue several jobs deep. A DRM scheduler
//!   entity stays on its core while it has queued work, so the depthwise
//!   stream and the dense streams run on different cores throughout and
//!   a dense job never follows a depthwise one on its own DPU. Every
//!   output is checked exactly. A failure here is the cross-core hazard
//!   the driver's rule would not cover.
//! - As a control, one file alternates depthwise and dense jobs back to
//!   back on one queue -- the same-core transition the dwell exists for.
//!   This arm is reported, not asserted: the hazard was intermittent when
//!   it was found, and its absence here would only mean this shape did not
//!   show it today.
//!
//! The shapes are the ones the e2e gate's `mixed_*` cases use, where the
//! hazard was originally seen: dense `32x32x64 -> 128` 1x1 fp16 and
//! depthwise `113x113x144` 3x3 stride 2 fp16.
//!
//!   CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!     cargo test --target aarch64-unknown-linux-gnu --release \
//!       --test dpu_mode_multicore_hw --no-run
//!
//!   ./dpu_mode_multicore_hw-<hash> --ignored --nocapture --test-threads=1

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
    conv::{Buffers, ConvPlan, Kernels, Shape, relocate},
    device::{OwnedBuffer, fini_bo, prep_bo, submit, submit_tasks},
    tensor_layout::{
        compact_atomic_output, nc1hwc2_storage_size, pack_depthwise_to_rocket_weights,
        pack_hwcf_to_rocket_weights, pack_nhwc_to_nc1hwc2_padded, rocket_weight_storage_size,
    },
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const PAGE_BYTES: usize = 4096;
const FP16_BYTES: usize = 2;
const FP16_ONE: u16 = 0x3c00;
const OUTPUT_SENTINEL: u8 = 0xa5;
const PER_TILE_TIMEOUT_NS: u64 = 5_000_000_000;
/// Jobs each stream runs in the timed phase.
const JOBS: usize = 60;
/// Jobs kept in flight per queue so an entity never goes idle and moves.
const QUEUE_DEPTH: usize = 4;

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

/// Kernel taps that land inside the image for an output whose window starts
/// at input `start` (its top-left tap, padding included).
fn valid_taps(start: isize, extent: usize, kernel: usize) -> usize {
    (0..kernel as isize)
        .filter(|tap| (0..extent as isize).contains(&(start + tap)))
        .count()
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Dense,
    Depthwise,
}

/// A depthwise -> dense pair the transition campaign runs.
struct Pair {
    label: &'static str,
    depthwise: (Shape, Kernels),
    dense: (Shape, Kernels),
}

/// The original `mixed_*` pair, plus ConvNeXt-T's dwconv 7x7 -> pwconv1 at
/// stages 1 and 3. The 7x7 depthwise jobs are the longest depthwise
/// write-backs the compiler now emits, and each pwconv1 widens 4x, so a
/// truncation to the first 16 channels leaves most of the output unwritten.
fn transition_pairs() -> Vec<Pair> {
    // Padding 0 on the pre-padded extent, as the compiled path runs it (the
    // pad is a separate op) and as `depthwise_7x7_probe_hw` validated it.
    let fp16_dw7 = |side: u32, c: u32| {
        (
            Shape::with_out_channels(side + 6, side + 6, 1, c, c)
                .with_padding([0, 0])
                .with_depthwise(),
            [7, 7],
        )
    };
    vec![
        Pair {
            label: "mnv2 dw3x3s2 113x113x144 -> dense 32x32 64->128",
            depthwise: Dispatch::spec(Kind::Depthwise),
            dense: Dispatch::spec(Kind::Dense),
        },
        Pair {
            label: "convnext s1 dw7x7 62x62x96 -> pw 56x56 96->384",
            depthwise: fp16_dw7(56, 96),
            dense: (Shape::with_out_channels(56, 56, 1, 96, 384), [1, 1]),
        },
        Pair {
            label: "convnext s3 dw7x7 20x20x384 -> pw 14x14 384->1536",
            depthwise: fp16_dw7(14, 384),
            dense: (Shape::with_out_channels(14, 14, 1, 384, 1536), [1, 1]),
        },
    ]
}

fn transition_rounds() -> usize {
    std::env::var("ROCKET_TRANSITION_ROUNDS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(500)
}

/// One dispatch: uniform 1.0 input and weights, so a dense output element
/// is `Cin` and a depthwise one is the number of taps inside the image.
struct Dispatch {
    kind: Kind,
    shape: Shape,
    kernels: Kernels,
    packed_input: Vec<u8>,
    packed_weights: Vec<u8>,
    bias_bytes: usize,
    output_bytes: usize,
    output_pixel_count: usize,
    output_width: usize,
    output_height: usize,
    output_bytes_per_pixel: usize,
    output_block_bytes: usize,
    programs: Vec<Vec<RegCmd>>,
}

impl Dispatch {
    fn spec(kind: Kind) -> (Shape, Kernels) {
        match kind {
            Kind::Dense => (Shape::with_out_channels(32, 32, 1, 64, 128), [1, 1]),
            Kind::Depthwise => (
                Shape::with_out_channels(113, 113, 2, 144, 144).with_depthwise(),
                [3, 3],
            ),
        }
    }

    fn new(kind: Kind) -> Dispatch {
        let (shape, kernels) = Dispatch::spec(kind);
        Dispatch::from_shape(shape, kernels)
    }

    fn from_shape(shape: Shape, kernels: Kernels) -> Dispatch {
        let kind = if shape.depthwise {
            Kind::Depthwise
        } else {
            Kind::Dense
        };
        let cin = shape.in_channels as usize;
        let cout = shape.out_channels as usize;
        let pixel_count = shape.width as usize * shape.height as usize;
        let input_bytes_per_pixel = cin * FP16_BYTES;
        let packed_input_bytes_per_pixel = cin.max(16).next_multiple_of(16) * FP16_BYTES;
        let dense_input: Vec<u8> = (0..pixel_count * cin)
            .flat_map(|_| FP16_ONE.to_le_bytes())
            .collect();
        let mut packed_input =
            vec![0u8; nc1hwc2_storage_size(pixel_count, packed_input_bytes_per_pixel).unwrap()];
        pack_nhwc_to_nc1hwc2_padded(
            &dense_input,
            pixel_count,
            input_bytes_per_pixel,
            packed_input_bytes_per_pixel,
            &mut packed_input,
        )
        .unwrap();

        let packed_weights = match kind {
            Kind::Dense => {
                let dense: Vec<u8> = (0..kernels[0] * kernels[1] * cin * cout)
                    .flat_map(|_| FP16_ONE.to_le_bytes())
                    .collect();
                let mut packed =
                    vec![
                        0u8;
                        rocket_weight_storage_size(kernels[0], kernels[1], cin, cout, FP16_BYTES)
                            .unwrap()
                    ];
                pack_hwcf_to_rocket_weights(
                    &dense,
                    kernels[0],
                    kernels[1],
                    cin,
                    cout,
                    FP16_BYTES,
                    &mut packed,
                )
                .unwrap();
                packed
            }
            Kind::Depthwise => {
                let weight_bytes = shape.weight_bytes(kernels) as usize;
                let padded_channels = weight_bytes / (kernels[0] * kernels[1] * FP16_BYTES);
                let dense: Vec<u8> = (0..cin * kernels[0] * kernels[1])
                    .flat_map(|_| FP16_ONE.to_le_bytes())
                    .collect();
                let mut packed = vec![0u8; weight_bytes];
                pack_depthwise_to_rocket_weights(
                    &dense,
                    kernels[0],
                    kernels[1],
                    cin,
                    padded_channels,
                    FP16_BYTES,
                    &mut packed,
                )
                .unwrap();
                packed
            }
        };
        let padded_out = shape.padded_out_channels() as usize;
        let output_width = shape.output_width(kernels) as usize;
        let output_height = shape.output_height(kernels) as usize;
        Dispatch {
            kind,
            shape,
            kernels,
            packed_input,
            packed_weights,
            bias_bytes: shape.bs_buffer_bytes().max(padded_out * 4),
            output_bytes: shape.output_scratch_bytes(kernels),
            output_pixel_count: output_width * output_height,
            output_width,
            output_height,
            output_bytes_per_pixel: cout * FP16_BYTES,
            output_block_bytes: shape.output_atom_bytes() as usize,
            programs: ConvPlan::new(shape, kernels).programs(),
        }
    }

    fn expected(&self, y: usize, x: usize) -> f32 {
        match self.kind {
            Kind::Dense => self.shape.in_channels as f32,
            Kind::Depthwise => {
                let stride = self.shape.stride as usize;
                let [pad_top, pad_left] = self
                    .shape
                    .padding
                    .unwrap_or([self.kernels[0] / 2, self.kernels[1] / 2]);
                (valid_taps(
                    (y * stride) as isize - pad_top as isize,
                    self.shape.height as usize,
                    self.kernels[0],
                ) * valid_taps(
                    (x * stride) as isize - pad_left as isize,
                    self.shape.width as usize,
                    self.kernels[1],
                )) as f32
            }
        }
    }
}

/// One dispatch's BOs on one file, plus `QUEUE_DEPTH` independent output
/// buffers so several jobs can be in flight on the queue at once.
struct Staged {
    file: File,
    input: OwnedBuffer,
    weights: OwnedBuffer,
    bias: OwnedBuffer,
    /// (output, regcmds per tile) per slot.
    slots: Vec<(OwnedBuffer, Vec<(OwnedBuffer, u32)>)>,
    compacted: Vec<u8>,
}

unsafe impl Send for Staged {}

impl Staged {
    fn new(file: File, dispatch: &Dispatch, slot_count: usize) -> Staged {
        let fd = file.as_raw_fd();
        unsafe {
            let input = OwnedBuffer::new(fd, page_aligned(dispatch.packed_input.len()), &file);
            ptr::write_bytes(input.host_ptr, 0, input.size);
            ptr::copy_nonoverlapping(
                dispatch.packed_input.as_ptr(),
                input.host_ptr,
                dispatch.packed_input.len(),
            );
            let weights = OwnedBuffer::new(fd, page_aligned(dispatch.packed_weights.len()), &file);
            ptr::write_bytes(weights.host_ptr, 0, weights.size);
            ptr::copy_nonoverlapping(
                dispatch.packed_weights.as_ptr(),
                weights.host_ptr,
                dispatch.packed_weights.len(),
            );
            let bias = OwnedBuffer::new(fd, page_aligned(dispatch.bias_bytes), &file);
            ptr::write_bytes(bias.host_ptr, 0, bias.size);
            for handle in [input.handle, weights.handle, bias.handle] {
                fini_bo(fd, handle).expect("FINI_BO");
            }
            let mut slots = Vec::with_capacity(slot_count);
            for _ in 0..slot_count {
                let output = OwnedBuffer::new(fd, page_aligned(dispatch.output_bytes), &file);
                ptr::write_bytes(output.host_ptr, OUTPUT_SENTINEL, output.size);
                fini_bo(fd, output.handle).expect("FINI_BO");
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
                    relocate(&mut commands, buffers);
                    let bytes = commands.len() * mem::size_of::<u64>();
                    let buffer = OwnedBuffer::new(fd, page_aligned(bytes), &file);
                    ptr::write_bytes(buffer.host_ptr, 0, buffer.size);
                    let words =
                        std::slice::from_raw_parts_mut(buffer.host_ptr as *mut u64, commands.len());
                    for (word, command) in words.iter_mut().zip(commands.iter()) {
                        *word = command.0;
                    }
                    fini_bo(fd, buffer.handle).expect("FINI_BO regcmd");
                    regcmds.push((buffer, commands.len() as u32));
                }
                slots.push((output, regcmds));
            }
            Staged {
                file,
                input,
                weights,
                bias,
                slots,
                compacted: vec![0u8; dispatch.output_pixel_count * dispatch.output_bytes_per_pixel],
            }
        }
    }

    /// Submits every tile of the dispatch into `slot` without waiting.
    fn submit_slot(&self, slot: usize) -> Result<(), String> {
        let fd = self.file.as_raw_fd();
        let (output, regcmds) = &self.slots[slot];
        for (tile, (regcmd, count)) in regcmds.iter().enumerate() {
            let in_handles = [
                regcmd.handle,
                self.input.handle,
                self.weights.handle,
                self.bias.handle,
            ];
            unsafe {
                submit(
                    fd,
                    regcmd.dma_address,
                    *count,
                    &in_handles,
                    &[output.handle],
                )
            }
            .map_err(|err| format!("slot {slot} tile {tile}: SUBMIT failed: {err}"))?;
        }
        Ok(())
    }

    fn wait_slot(&self, slot: usize) -> Result<(), String> {
        let fd = self.file.as_raw_fd();
        unsafe { prep_bo(fd, self.slots[slot].0.handle, PER_TILE_TIMEOUT_NS) }
            .map_err(|err| format!("slot {slot}: PREP_BO failed: {err}"))
    }

    /// Checks `slot`'s output exactly, then re-arms it with the sentinel.
    fn verify_slot(&mut self, slot: usize, dispatch: &Dispatch, label: &str) -> Result<(), String> {
        self.verify_slot_from(slot, dispatch, label, None)
    }

    /// Copies `slot`'s raw output now, for checking later with
    /// [`Staged::verify_slot_from`] -- so what is checked is what the CPU
    /// saw at this instant, not after some later job has also finished.
    fn snapshot_slot(&self, slot: usize, dispatch: &Dispatch, into: &mut Vec<u8>) {
        let (output, _) = &self.slots[slot];
        into.clear();
        into.extend_from_slice(unsafe {
            std::slice::from_raw_parts(output.host_ptr, dispatch.output_bytes)
        });
    }

    /// [`Staged::verify_slot`], checking `source` (a snapshot) in place of
    /// the live output when given. The live output is re-armed either way.
    fn verify_slot_from(
        &mut self,
        slot: usize,
        dispatch: &Dispatch,
        label: &str,
        source: Option<&[u8]>,
    ) -> Result<(), String> {
        let (output, _) = &self.slots[slot];
        let scratch = source.unwrap_or_else(|| unsafe {
            std::slice::from_raw_parts(output.host_ptr, dispatch.output_bytes)
        });
        let written = compact_atomic_output(
            scratch,
            dispatch.output_pixel_count,
            dispatch.output_pixel_count,
            dispatch.output_bytes_per_pixel,
            dispatch.output_block_bytes,
            &mut self.compacted,
        );
        if written != self.compacted.len() {
            return Err(format!("{label}: compaction wrote {written} bytes"));
        }
        let channels = dispatch.output_bytes_per_pixel / FP16_BYTES;
        let mut wrong = 0usize;
        let mut sentinel = 0usize;
        let mut first: Option<(usize, usize, usize, f32)> = None;
        let mut worst_channel = 0usize;
        for y in 0..dispatch.output_height {
            for x in 0..dispatch.output_width {
                let want = dispatch.expected(y, x);
                let pixel = y * dispatch.output_width + x;
                for c in 0..channels {
                    let offset = pixel * dispatch.output_bytes_per_pixel + c * FP16_BYTES;
                    let bits =
                        u16::from_le_bytes([self.compacted[offset], self.compacted[offset + 1]]);
                    let got = f16_to_f32(bits);
                    if got != want {
                        wrong += 1;
                        worst_channel = worst_channel.max(c);
                        if bits == u16::from_le_bytes([OUTPUT_SENTINEL; 2]) {
                            sentinel += 1;
                        }
                        if first.is_none() {
                            first = Some((y, x, c, got));
                        }
                    }
                }
            }
        }
        unsafe { ptr::write_bytes(output.host_ptr, OUTPUT_SENTINEL, output.size) };
        unsafe { fini_bo(self.file.as_raw_fd(), output.handle) }.map_err(|e| e.to_string())?;
        if wrong == 0 {
            Ok(())
        } else {
            Err(format!(
                "{label}: {wrong} of {} elements wrong ({sentinel} never written); first (y, x, c, got) = {first:?}; \
                 highest wrong channel {worst_channel}",
                self.compacted.len() / 2
            ))
        }
    }
}

fn open_device() -> File {
    OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("open /dev/accel/accel0 -- run this on the board")
}

/// Runs `JOBS` jobs of one dispatch on one file, `QUEUE_DEPTH` in flight,
/// verifying each as it completes. Returns the failures.
fn stream(mut staged: Staged, dispatch: &Dispatch, label: &str) -> Vec<String> {
    let mut failures = Vec::new();
    let depth = staged.slots.len();
    let mut submitted = 0usize;
    let mut completed = 0usize;
    while submitted < depth.min(JOBS) {
        if let Err(err) = staged.submit_slot(submitted % depth) {
            failures.push(format!("{label}: {err}"));
            return failures;
        }
        submitted += 1;
    }
    while completed < JOBS {
        let slot = completed % depth;
        if let Err(err) = staged.wait_slot(slot) {
            failures.push(format!("{label}: {err}"));
            return failures;
        }
        if let Err(err) = staged.verify_slot(slot, dispatch, label) {
            failures.push(err);
            if failures.len() >= 8 {
                return failures;
            }
        }
        completed += 1;
        if submitted < JOBS {
            if let Err(err) = staged.submit_slot(submitted % depth) {
                failures.push(format!("{label}: {err}"));
                return failures;
            }
            submitted += 1;
        }
    }
    failures
}

/// Depthwise on one file, dense on two others, all at once, no dwell.
#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn dense_on_other_cores_survives_concurrent_depthwise() {
    let dense = Arc::new(Dispatch::new(Kind::Dense));
    let depthwise = Arc::new(Dispatch::new(Kind::Depthwise));
    println!(
        "dense {} tile(s), depthwise {} tile(s), {JOBS} jobs per stream, queue depth {QUEUE_DEPTH}",
        dense.programs.len(),
        depthwise.programs.len()
    );
    let streams: Vec<(Kind, Arc<Dispatch>)> = vec![
        (Kind::Depthwise, Arc::clone(&depthwise)),
        (Kind::Dense, Arc::clone(&dense)),
        (Kind::Dense, Arc::clone(&dense)),
    ];
    let barrier = Arc::new(Barrier::new(streams.len()));
    let handles: Vec<_> = streams
        .into_iter()
        .enumerate()
        .map(|(index, (kind, dispatch))| {
            let staged = Staged::new(open_device(), &dispatch, QUEUE_DEPTH);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                stream(staged, &dispatch, &format!("ctx {index} {kind:?}"))
            })
        })
        .collect();
    let failures: Vec<String> = handles
        .into_iter()
        .flat_map(|handle| handle.join().expect("stream thread panicked"))
        .collect();
    for failure in &failures {
        println!("  {failure}");
    }
    assert!(
        failures.is_empty(),
        "{} failures with depthwise and dense streams on separate files -- the depthwise -> dense \
         hazard reaches across cores, and the driver's per-device dwell does not cover an \
         in-flight depthwise on another context",
        failures.len()
    );
}

/// Control: the same-core transition, one file, alternating, no dwell.
/// Reported only -- see the module doc comment.
#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn same_queue_alternation_without_dwell_is_reported() {
    let dense = Dispatch::new(Kind::Dense);
    let depthwise = Dispatch::new(Kind::Depthwise);
    let file = open_device();
    let mut dense_staged = Staged::new(file.try_clone().unwrap(), &dense, 1);
    let mut depthwise_staged = Staged::new(file, &depthwise, 1);
    let mut wrong = 0usize;
    let mut samples = Vec::new();
    for round in 0..JOBS {
        depthwise_staged.submit_slot(0).expect("depthwise SUBMIT");
        depthwise_staged.wait_slot(0).expect("depthwise PREP_BO");
        // Straight into the dense job: no dwell.
        dense_staged.submit_slot(0).expect("dense SUBMIT");
        dense_staged.wait_slot(0).expect("dense PREP_BO");
        if let Err(err) = depthwise_staged.verify_slot(0, &depthwise, "depthwise") {
            wrong += 1;
            if samples.len() < 4 {
                samples.push(format!("round {round}: {err}"));
            }
        }
        if let Err(err) = dense_staged.verify_slot(0, &dense, "dense after depthwise") {
            wrong += 1;
            if samples.len() < 4 {
                samples.push(format!("round {round}: {err}"));
            }
        }
        std::thread::sleep(Duration::from_micros(200));
    }
    println!(
        "same-queue depthwise -> dense without dwell: {wrong} wrong of {} jobs",
        JOBS * 2
    );
    for sample in samples {
        println!("  {sample}");
    }
}

/// How a dense job follows a depthwise one in the transition campaign.
#[derive(Clone, Copy, Debug)]
enum Gap {
    /// Two jobs on one file: the dense SUBMIT goes in the moment the
    /// depthwise PREP_BO returns, and the depthwise output is copied while
    /// the dense job runs -- the driver's own order (it compacts the
    /// depthwise output right after its fence), with no dwell and no
    /// compaction time between the two submits.
    Userspace,
    /// One job, depthwise tasks then dense tasks. The kernel's IRQ thread
    /// starts the first dense task straight from the depthwise retire, a
    /// few microseconds later: tighter than any path the driver can take.
    SameJob,
}

/// Submits `depthwise` then `dense` as the tasks of a single job.
fn submit_same_job(depthwise: &Staged, dense: &Staged) -> Result<(), String> {
    let fd = depthwise.file.as_raw_fd();
    let mut tasks = Vec::new();
    let mut in_handles = Vec::new();
    for staged in [depthwise, dense] {
        let (_, regcmds) = &staged.slots[0];
        for (regcmd, count) in regcmds {
            tasks.push((regcmd.dma_address, *count));
            in_handles.push(regcmd.handle);
        }
        in_handles.extend([
            staged.input.handle,
            staged.weights.handle,
            staged.bias.handle,
        ]);
    }
    let out_handles = [depthwise.slots[0].0.handle, dense.slots[0].0.handle];
    unsafe { submit_tasks(fd, &tasks, &in_handles, &out_handles) }
        .map_err(|err| format!("same-job SUBMIT failed: {err}"))
}

/// Depthwise -> dense with no dwell, `ROCKET_TRANSITION_ROUNDS` times (500 by
/// default) per pair and gap, every output checked exactly.
///
/// This is the question `DEPTHWISE_TO_DENSE_QUIESCENCE` (rocket-hal-driver
/// device.rs) leaves open: does a dense job that starts right after a
/// depthwise one ever write only its first 16 channels, and is a depthwise
/// output ever incomplete when its fence has signaled. Both pairs of shapes
/// that matter are here -- the MobileNetV2 pair where the hazard was
/// reported, and ConvNeXt-T's 7x7 depthwise -> pwconv1 -- each run both the
/// way the driver submits and inside one job, where the dense task starts
/// microseconds after the depthwise one retires.
///
/// Uniform data is enough for this question: an element the DPU never wrote
/// keeps the 0xa5a5 sentinel, and a dense job run in depthwise mode writes a
/// tap count rather than `Cin`. The test checks first that the detector sees
/// an unwritten output.
///
///   ROCKET_TRANSITION_ROUNDS=5000 ./dpu_mode_multicore_hw-<hash> --ignored \
///     --nocapture --test-threads=1 depthwise_to_dense_transition_campaign
#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn depthwise_to_dense_transition_campaign() {
    let rounds = transition_rounds();
    let mut failures = Vec::new();
    let mut transitions = 0usize;
    for pair in transition_pairs() {
        let depthwise = Dispatch::from_shape(pair.depthwise.0, pair.depthwise.1);
        let dense = Dispatch::from_shape(pair.dense.0, pair.dense.1);

        // Detector self-check: a slot no job has written must fail, entirely.
        let mut probe = Staged::new(open_device(), &dense, 1);
        match probe.verify_slot(0, &dense, "unwritten") {
            Ok(()) => panic!("{}: an unwritten output verified clean", pair.label),
            Err(err) => assert!(
                err.contains(&format!(
                    "{0} of {0} elements wrong ({0} never written)",
                    dense.output_pixel_count * dense.output_bytes_per_pixel / FP16_BYTES
                )),
                "{}: detector self-check: {err}",
                pair.label
            ),
        }
        drop(probe);

        for gap in [Gap::Userspace, Gap::SameJob] {
            let file = open_device();
            let mut dw = Staged::new(file.try_clone().unwrap(), &depthwise, 1);
            let mut dn = Staged::new(file, &dense, 1);
            let mut snapshot = Vec::with_capacity(depthwise.output_bytes);
            let mut wrong = 0usize;
            let mut samples = Vec::new();
            let mut aborted = None;
            let start = Instant::now();
            let mut done = 0usize;
            for round in 0..rounds {
                let ran = match gap {
                    Gap::Userspace => dw
                        .submit_slot(0)
                        .and_then(|()| dw.wait_slot(0))
                        .and_then(|()| dn.submit_slot(0))
                        .map(|()| dw.snapshot_slot(0, &depthwise, &mut snapshot))
                        .and_then(|()| dn.wait_slot(0)),
                    Gap::SameJob => submit_same_job(&dw, &dn)
                        .and_then(|()| dn.wait_slot(0))
                        .and_then(|()| dw.wait_slot(0)),
                };
                if let Err(err) = ran {
                    aborted = Some(format!("round {round}: {err}"));
                    break;
                }
                let source = matches!(gap, Gap::Userspace).then_some(snapshot.as_slice());
                for result in [
                    dw.verify_slot_from(0, &depthwise, "depthwise", source),
                    dn.verify_slot(0, &dense, "dense after depthwise"),
                ] {
                    if let Err(err) = result {
                        wrong += 1;
                        if samples.len() < 4 {
                            samples.push(format!("round {round}: {err}"));
                        }
                    }
                }
                done += 1;
            }
            let elapsed = start.elapsed();
            transitions += done;
            println!(
                "{} [{gap:?}]: {done} transitions, {wrong} wrong outputs, {:.2} ms/round{}",
                pair.label,
                elapsed.as_secs_f64() * 1e3 / done.max(1) as f64,
                aborted
                    .as_deref()
                    .map(|err| format!(", ABORTED at {err}"))
                    .unwrap_or_default()
            );
            for sample in &samples {
                println!("    {sample}");
            }
            if wrong > 0 || aborted.is_some() {
                failures.push(format!(
                    "{} [{gap:?}]: {wrong} wrong, aborted: {aborted:?}",
                    pair.label
                ));
            }
        }
    }
    println!("{transitions} depthwise -> dense transitions without a dwell");
    assert!(
        failures.is_empty(),
        "depthwise -> dense without a dwell went wrong: {failures:#?}"
    );
}
