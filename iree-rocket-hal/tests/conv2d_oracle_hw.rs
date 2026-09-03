//! Cartesian Conv2D correctness sweep against an independent logical oracle.
//!
//! Every supported case executes through `Shape::parity_padded_shape`,
//! `ConvPlan::new`, production HWCF weight packing, and the RK3588 device.
//! The case and oracle remain logical while planning, coefficients, and
//! scratch output use the HAL's physical programmed shape. Failures are
//! accumulated: one bad case
//! never prevents later cases from running, and the test asserts only after
//! printing the complete summary. The one exception is a *sick device* --
//! see `run_hardware_case_matrix`, which halts there because past that point
//! a sweep measures the NPU's health rather than the shapes.
//!
//! Cross-compile and run on the board:
//!
//! ```text
//! CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!   cargo test --target aarch64-unknown-linux-gnu --release \
//!     --test conv2d_oracle_hw --no-run
//!
//! ./conv2d_oracle_hw-<hash> --ignored --nocapture
//! ```

#[path = "support/conv2d_oracle.rs"]
mod conv2d_oracle;

use std::{
    any::Any,
    collections::BTreeMap,
    fs::OpenOptions,
    mem,
    os::unix::io::AsRawFd,
    panic::{AssertUnwindSafe, catch_unwind},
    ptr,
    sync::{Mutex, OnceLock},
};

use conv2d_oracle::{
    Conv2dCase, Conv2dFixture, OraclePattern, OraclePrecision, build_fixture, expected_output,
    expected_output_carries_signal, f16_to_f32, output_offset, output_storage_bytes,
};
#[cfg(feature = "hardware-characterization")]
use conv2d_oracle::{build_raw_fixture, feature_offset};
use iree_rocket_hal::rocket::{
    conv::{AccumulatorOutputTile, Buffers, ConvPlan, Shape},
    device::{Buffer, JobDesc, close_bo, fini_bo, prep_bo, submit_jobs, unmap_bo},
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const PAGE_BYTES: usize = 4096;
const PER_CASE_TIMEOUT_NS: u64 = 5_000_000_000;
const OUTPUT_SENTINEL: u8 = 0xa5;
// `nextest -j1` serializes test *processes*, but Rust's harness still runs
// ignored tests in this binary concurrently. The RK3588 NPU is a single
// shared device, so concurrent jobs turn independent oracle failures into
// cross-test contamination.
static NPU_TEST_LOCK: Mutex<()> = Mutex::new(());

fn page_aligned_size(size: usize) -> usize {
    size.max(1).div_ceil(PAGE_BYTES) * PAGE_BYTES
}

struct OwnedBuffer {
    fd: i32,
    buffer: Buffer,
}

impl OwnedBuffer {
    unsafe fn new(fd: i32, size: usize, file: &std::fs::File, role: &'static str) -> Self {
        let buffer = unsafe { Buffer::new(fd, page_aligned_size(size), file) };
        ledger_record_alloc(role, &buffer);
        Self { fd, buffer }
    }

    /// Extra mapped bytes to append to a named input buffer.
    ///
    /// The accumulator failure raises a DMA **read** error, so the question is
    /// which buffer the NPU reads past. Padding one at a time separates them:
    /// if a shape becomes exact only when a particular buffer is grown, that
    /// buffer is the one being over-read. The padding is zero-filled, so it is
    /// inert as data -- it only makes the pages mapped.
    fn pad_bytes(which: &str) -> usize {
        std::env::var(format!("ROCKET_PAD_{which}"))
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(0)
    }

    /// Byte the padding region is filled with (`ROCKET_POISON_<WHICH>`).
    ///
    /// Zero padding proves *that* the NPU reads outside the buffer but says
    /// nothing about *which* buffer, because a zero contributes nothing to the
    /// sum. A nonzero fill does: under the `Counting` pattern every real input
    /// and coefficient is 1, so the accumulator is a plain count, and any
    /// over-read of a region filled with `p` shifts the result by a multiple
    /// of `p`. Poisoning one buffer at a time therefore names the source.
    fn poison_byte(which: &str) -> u8 {
        std::env::var(format!("ROCKET_POISON_{which}"))
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(0)
    }

    unsafe fn from_bytes_padded(
        fd: i32,
        bytes: &[u8],
        which: &'static str,
        file: &std::fs::File,
    ) -> Self {
        let buffer = unsafe { Self::new(fd, bytes.len() + Self::pad_bytes(which), file, which) };
        unsafe {
            ptr::write_bytes(buffer.buffer.host_ptr, 0, buffer.buffer.size);
            ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.buffer.host_ptr, bytes.len());
            // Everything past the logical data is padding; poison it so an
            // over-read shows up in the arithmetic.
            let poison = Self::poison_byte(which);
            if poison != 0 && buffer.buffer.size > bytes.len() {
                ptr::write_bytes(
                    buffer.buffer.host_ptr.add(bytes.len()),
                    poison,
                    buffer.buffer.size - bytes.len(),
                );
            }
        }
        buffer
    }

    #[allow(dead_code)]
    unsafe fn from_bytes(fd: i32, bytes: &[u8], file: &std::fs::File) -> Self {
        let buffer = unsafe { Self::new(fd, bytes.len(), file, "SCRATCH") };
        unsafe {
            ptr::write_bytes(buffer.buffer.host_ptr, 0, buffer.buffer.size);
            ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.buffer.host_ptr, bytes.len());
        }
        buffer
    }
}

impl Drop for OwnedBuffer {
    fn drop(&mut self) {
        ledger_record_free(self.buffer.handle);
        unsafe {
            let _ = unmap_bo(&self.buffer);
            let _ = close_bo(self.fd, self.buffer.handle);
        }
    }
}

// ---------------------------------------------------------------------------
// IOVA ledger -- `ROCKET_IOVA_LEDGER=1`
//
// Instrumentation for the buffer/DMA-address-reuse theory of this file's
// per-process contamination. Every case allocates five GEM buffers and frees
// them again, so the IOVAs `CREATE_BO` hands back are recycled continuously,
// and the RK3588 NPU sits behind the Rockchip IOMMU (three cores, each with
// `iommus` in the device tree). If a freed IOVA is reissued while a stale
// translation survives, the DPU reads and writes the *previous* case's
// physical pages -- which is exactly the symptom set this sweep shows:
// sentinel-valued output (the DPU wrote somewhere else), fully written wrong
// values (it read someone else's weights), contamination that only a fresh
// process clears (IOVA allocation is per `drm_file`), and order dependence.
//
// This tests that theory without a kernel patch, because `CREATE_BO` already
// returns the IOVA and `Buffer` keeps it in `dma_address`. The ledger records
// every lease, flags any overlap between simultaneously live buffers,
// attributes each new lease to the freed one whose address it recycles, and
// after each dispatch re-reads the buffers the DPU was only supposed to read
// to see whether it wrote outside its output.
//
// Deliberately opt-in, and it allocates no GEM buffers and submits no jobs of
// its own -- `run_hardware_case_matrix`'s doc comment records what happened
// the last time this loop grew work on its failure path. The scan's `PREP_BO`
// calls are cache maintenance on buffers the job has already finished with,
// not new work for the DPU, but they are still a change to the sequence. The
// ledger's own output is therefore only trustworthy when the run's failure
// set matches a run with the knob off. Check that first, every time.
// ---------------------------------------------------------------------------

/// Wall-clock time of the most recent `SUBMIT` -> `PREP_BO` round trip.
///
/// This is how a hung job becomes visible to userspace. When the NPU stalls,
/// `rocket_job.c`'s watchdog resets the core and signals the job's fence with
/// an error -- but `PREP_BO` waits on the `dma_resv` fence, and a fence
/// signaled with an error is still *signaled*, so it returns success and the
/// sweep reads a half-written output buffer. The round trip is the only thing
/// that changes: normal dispatches here run in single-digit milliseconds, a
/// timed-out one costs the scheduler's full timeout.
static LAST_DISPATCH: Mutex<Option<std::time::Duration>> = Mutex::new(None);

/// Round trip above which a case is a hung job rather than a slow shape.
///
/// Measured on `planck` 2026-09-02: with `dense_geometry_regression`, clean
/// runs of all 22 cases take 220-317 ms end to end and every run with one
/// failure takes 694-875 ms -- about 450 ms of extra wait per failure, and
/// each one has a matching `rocket ...npu: NPU job timed out` in `dmesg`. No
/// single healthy dispatch in this file comes near this bound; see
/// `dispatch_ms` in the ledger output for the distribution.
const DISPATCH_TIMEOUT_FLOOR: std::time::Duration = std::time::Duration::from_millis(150);

fn record_dispatch_duration(elapsed: std::time::Duration) {
    *LAST_DISPATCH
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(elapsed);
}

/// The last dispatch's round trip, if it was long enough to be a hung job.
fn last_dispatch_if_timed_out() -> Option<std::time::Duration> {
    (*LAST_DISPATCH
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()))
    .filter(|elapsed| *elapsed >= DISPATCH_TIMEOUT_FLOOR)
}

/// One GEM buffer's IOVA lease, exactly as `CREATE_BO` handed it out.
#[derive(Clone)]
struct BoLease {
    seq: u64,
    case: usize,
    role: &'static str,
    dma: u32,
    size: usize,
    handle: u32,
}

impl BoLease {
    fn end(&self) -> u64 {
        u64::from(self.dma) + self.size as u64
    }

    fn overlaps(&self, other: &BoLease) -> bool {
        u64::from(self.dma) < other.end() && u64::from(other.dma) < self.end()
    }

    fn range(&self) -> String {
        format!("{:#x}+{:#x}", self.dma, self.size)
    }
}

struct IovaLedger {
    /// Leases the kernel currently considers outstanding.
    live: Vec<BoLease>,
    /// Every released lease, oldest first -- the recycling history.
    freed: Vec<BoLease>,
    /// Leases taken by the case being measured right now.
    case_leases: Vec<BoLease>,
    notes: Vec<String>,
    seq: u64,
    case: usize,
}

impl IovaLedger {
    const fn new() -> Self {
        Self {
            live: Vec::new(),
            freed: Vec::new(),
            case_leases: Vec::new(),
            notes: Vec::new(),
            seq: 0,
            case: 0,
        }
    }
}

static IOVA_LEDGER: Mutex<IovaLedger> = Mutex::new(IovaLedger::new());

fn ledger_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("ROCKET_IOVA_LEDGER").is_some())
}

fn lock_ledger() -> std::sync::MutexGuard<'static, IovaLedger> {
    IOVA_LEDGER
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Opens a fresh page for `case`. `freed` deliberately survives: the whole
/// point is what an address held *before* this case got it.
fn ledger_begin_case(case: usize) {
    if !ledger_enabled() {
        return;
    }
    let mut ledger = lock_ledger();
    ledger.case = case;
    ledger.case_leases.clear();
    ledger.notes.clear();
}

fn ledger_note(note: String) {
    if !ledger_enabled() {
        return;
    }
    lock_ledger().notes.push(note);
}

fn ledger_record_alloc(role: &'static str, buffer: &Buffer) {
    if !ledger_enabled() {
        return;
    }
    let mut ledger = lock_ledger();
    ledger.seq += 1;
    let lease = BoLease {
        seq: ledger.seq,
        case: ledger.case,
        role,
        dma: buffer.dma_address,
        size: buffer.size,
        handle: buffer.handle,
    };

    let mut notes = Vec::new();
    // Two simultaneously live buffers sharing an address would be a kernel
    // allocator bug outright, and would explain every symptom on its own.
    for other in &ledger.live {
        if lease.overlaps(other) {
            notes.push(format!(
                "OVERLAP {} {} with live case {} {} {}",
                lease.role,
                lease.range(),
                other.case,
                other.role,
                other.range(),
            ));
        }
    }
    // Whose address this used to be. Reuse is expected and benign by itself;
    // it only means something next to a failure, which is why it is recorded
    // per case rather than asserted on.
    if let Some(previous) = ledger
        .freed
        .iter()
        .rev()
        .find(|freed| freed.overlaps(&lease))
    {
        notes.push(format!(
            "{} {} recycles case {} {} {} (seq {})",
            lease.role,
            lease.range(),
            previous.case,
            previous.role,
            previous.range(),
            previous.seq,
        ));
    }
    ledger.notes.extend(notes);
    ledger.case_leases.push(lease.clone());
    ledger.live.push(lease);
}

fn ledger_record_free(handle: u32) {
    if !ledger_enabled() {
        return;
    }
    let mut ledger = lock_ledger();
    // GEM handles are recycled too, but only one live buffer holds a given
    // handle at a time, so this is unambiguous.
    if let Some(position) = ledger.live.iter().position(|lease| lease.handle == handle) {
        let lease = ledger.live.remove(position);
        ledger.freed.push(lease);
    }
}

/// The measured case's buffer table plus whatever the scan flagged, ready to
/// print under the case's own result line. `None` when the ledger is off.
fn ledger_case_report() -> Option<String> {
    if !ledger_enabled() {
        return None;
    }
    let ledger = lock_ledger();
    let map = ledger
        .case_leases
        .iter()
        .map(|lease| format!("{}={}", lease.role, lease.range()))
        .collect::<Vec<_>>()
        .join(" ");
    let dispatch = *LAST_DISPATCH
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let mut lines = vec![format!(
        "      iova: {map} dispatch_ms={:.1}",
        dispatch.unwrap_or_default().as_secs_f64() * 1000.0,
    )];
    lines.extend(
        ledger
            .notes
            .iter()
            .map(|note| format!("      iova: {note}")),
    );
    Some(lines.join("\n"))
}

/// Re-reads a buffer the DPU was only supposed to read and reports any byte
/// that changed.
///
/// `PREP_BO` first, or the CPU's own clean lines from before `FINI_BO` would
/// hide a device write to the same address. `written` is what this process
/// put there; bytes past it are `from_bytes_padded`'s `fill`, which is zero
/// unless `ROCKET_POISON_*` asked for something else.
unsafe fn ledger_scan_readonly_bo(
    fd: i32,
    buffer: &OwnedBuffer,
    written: &[u8],
    fill: u8,
    role: &str,
) {
    if !ledger_enabled() {
        return;
    }
    if let Err(error) = unsafe { prep_bo(fd, buffer.buffer.handle, PER_CASE_TIMEOUT_NS) } {
        ledger_note(format!(
            "{role}: PREP_BO for read-back scan failed: {error}"
        ));
        return;
    }
    let mapped = unsafe { std::slice::from_raw_parts(buffer.buffer.host_ptr, buffer.buffer.size) };
    let intact = |offset: usize, byte: u8| byte == written.get(offset).copied().unwrap_or(fill);
    let Some((offset, &byte)) = mapped
        .iter()
        .enumerate()
        .find(|(offset, byte)| !intact(*offset, **byte))
    else {
        return;
    };
    let changed = mapped
        .iter()
        .enumerate()
        .filter(|(offset, byte)| !intact(*offset, **byte))
        .count();
    ledger_note(format!(
        "STRAY WRITE into {role} {:#x}+{:#x}: {changed} bytes changed, first at +{offset:#x} \
         (wrote {:#04x}, read back {byte:#04x})",
        buffer.buffer.dma_address,
        buffer.buffer.size,
        written.get(offset).copied().unwrap_or(fill),
    ));
}

/// Re-reads the output after the read-only scan, to separate "the DPU never
/// wrote these bytes" from "it had not written them *yet*".
///
/// `rocket-hal-driver`'s `DEPTHWISE_TO_DENSE_QUIESCENCE` records the same
/// hazard from the production side, hardware-validated: `PREP_BO` observes
/// the output fence, but the RK3588 DPU can still retain write-back state
/// briefly after that fence has signaled. If a case's missing tail turns up
/// here, this sweep raced the drain rather than measuring the shape.
unsafe fn ledger_recheck_output_drain(fd: i32, output: &OwnedBuffer, first_read: &[u8]) {
    if !ledger_enabled() {
        return;
    }
    std::thread::sleep(std::time::Duration::from_millis(2));
    if let Err(error) = unsafe { prep_bo(fd, output.buffer.handle, PER_CASE_TIMEOUT_NS) } {
        ledger_note(format!("output drain re-check: PREP_BO failed: {error}"));
        return;
    }
    let mapped = unsafe { std::slice::from_raw_parts(output.buffer.host_ptr, first_read.len()) };
    let settled = mapped
        .iter()
        .zip(first_read)
        .filter(|(now, before)| now != before)
        .count();
    if settled == 0 {
        return;
    }
    let still_sentinel = mapped
        .iter()
        .filter(|&&byte| byte == OUTPUT_SENTINEL)
        .count();
    ledger_note(format!(
        "LATE WRITE-BACK into output {:#x}: {settled} bytes changed after the completion fence, \
         {still_sentinel} of {} still sentinel",
        output.buffer.dma_address,
        first_read.len(),
    ));
}

/// How much of the output buffer the DPU actually touched. An output left
/// entirely at the sentinel is the signature the address-reuse theory
/// predicts: the job completed, and its writes landed somewhere else.
fn ledger_note_output_coverage(raw: &[u8], buffer: &Buffer) {
    if !ledger_enabled() {
        return;
    }
    let untouched = raw.iter().filter(|&&byte| byte == OUTPUT_SENTINEL).count();
    if untouched == raw.len() {
        ledger_note(format!(
            "OUTPUT UNTOUCHED {:#x}+{:#x}: all {} bytes still sentinel",
            buffer.dma_address,
            buffer.size,
            raw.len(),
        ));
    } else if untouched != 0 {
        ledger_note(format!(
            "output {:#x}+{:#x}: {untouched} of {} bytes still sentinel",
            buffer.dma_address,
            buffer.size,
            raw.len(),
        ));
    }
}

struct MismatchReport {
    mismatches: usize,
    max_abs_difference: f32,
    samples: Vec<String>,
    tile_mismatches: Vec<usize>,
}

struct CaseSuccess {
    data_banks: u32,
    weight_banks: u32,
    tiles: usize,
}

struct CaseExecution {
    plan: ConvPlan,
    output: Vec<u8>,
    /// The staging buffer exactly as the DPU left it, before
    /// `assemble_staged_accumulator_output` reinterprets it. The layout
    /// probes read this: the assembled `output` already assumes the answer
    /// they are trying to measure.
    #[cfg(feature = "hardware-characterization")]
    raw: Vec<u8>,
}

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
        let expected_tile_bytes = tile_pixels * blocks_per_pixel * block_bytes;
        if tile.scratch_bytes != expected_tile_bytes {
            return Err(format!(
                "tile {index} declares {} scratch bytes, expected {expected_tile_bytes}",
                tile.scratch_bytes
            ));
        }
        let tile_end = tile
            .scratch_offset
            .checked_add(tile.scratch_bytes)
            .ok_or_else(|| format!("tile {index} scratch range overflow"))?;
        if tile_end > scratch.len() {
            return Err(format!(
                "tile {index} scratch range {}..{tile_end} exceeds {} bytes",
                tile.scratch_offset,
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
                        return Err(format!(
                            "tile {index} output ({output_row}, {output_column}) exceeds \
                             {output_height}x{output_width}"
                        ));
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

fn tile_for_output(plan: &ConvPlan, y: usize, x: usize) -> Option<usize> {
    plan.tiles().iter().position(|tile| {
        (tile.rows.out_first as usize..(tile.rows.out_first + tile.rows.out_rows) as usize)
            .contains(&y)
            && (tile.columns.out_first as usize
                ..(tile.columns.out_first + tile.columns.out_cols) as usize)
                .contains(&x)
    })
}

fn compare_output(fixture: &Conv2dFixture, plan: &ConvPlan, output: &[u8]) -> MismatchReport {
    let case = fixture.case;
    let shape = fixture.shape;
    let out_height = shape.output_height(case.kernel) as usize;
    let out_width = shape.output_width(case.kernel) as usize;
    let tolerance = if case.precision == OraclePrecision::Int8 {
        1.0
    } else {
        0.0
    };
    let mut report = MismatchReport {
        mismatches: 0,
        max_abs_difference: 0.0,
        samples: Vec::new(),
        tile_mismatches: vec![0; plan.tiles().len()],
    };

    for y in 0..out_height {
        for x in 0..out_width {
            for channel in 0..case.cout as usize {
                let offset = output_offset(shape, case.kernel, channel, y, x);
                let got = match case.precision {
                    OraclePrecision::Fp16 => {
                        f16_to_f32(u16::from_le_bytes([output[offset], output[offset + 1]]))
                    }
                    OraclePrecision::Int8 => f32::from(output[offset] as i8),
                    OraclePrecision::Int8Accumulator => {
                        i32::from_le_bytes(output[offset..offset + 4].try_into().unwrap()) as f32
                    }
                };
                let want = expected_output(case, channel, y, x) as f32;
                let difference = (got - want).abs();
                report.max_abs_difference = report.max_abs_difference.max(difference);
                if !got.is_finite() || difference > tolerance {
                    report.mismatches += 1;
                    if let Some(tile) = tile_for_output(plan, y, x) {
                        report.tile_mismatches[tile] += 1;
                    }
                    if report.samples.len() < 12 {
                        report
                            .samples
                            .push(format!("[y={y}, x={x}, c={channel}] want {want} got {got}"));
                    }
                }
            }
        }
    }
    report
}

fn execute_case_output(
    file: &std::fs::File,
    fixture: &Conv2dFixture,
) -> Result<CaseExecution, String> {
    let plan = ConvPlan::new(fixture.shape, fixture.case.kernel);
    execute_case_output_with_plan(file, fixture, plan)
}

fn execute_case_output_with_plan(
    file: &std::fs::File,
    fixture: &Conv2dFixture,
    plan: ConvPlan,
) -> Result<CaseExecution, String> {
    let fd = file.as_raw_fd();
    let shape = fixture.shape;
    let kernels = fixture.case.kernel;
    let output_len = output_storage_bytes(shape, kernels);

    unsafe {
        let input = OwnedBuffer::from_bytes_padded(fd, &fixture.input, "INPUT", file);
        let weights = OwnedBuffer::from_bytes_padded(fd, &fixture.weights, "WEIGHTS", file);
        let bias = OwnedBuffer::from_bytes_padded(fd, &fixture.bias, "BIAS", file);
        let output = OwnedBuffer::new(fd, output_len, file, "OUTPUT");
        // A zero fill can turn an unwritten output lane into a plausible
        // convolution result. Poison the whole allocation so missing tail
        // pixels/blocks fail loudly and remain distinguishable from a real
        // all-zero accumulator.
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
        for program in &programs {
            let command_bytes = program.len() * mem::size_of::<u64>();
            let buffer = OwnedBuffer::new(fd, command_bytes, file, "REGCMD");
            ptr::write_bytes(buffer.buffer.host_ptr, 0, buffer.buffer.size);
            let words =
                std::slice::from_raw_parts_mut(buffer.buffer.host_ptr as *mut u64, program.len());
            for (destination, command) in words.iter_mut().zip(program) {
                *destination = command.0;
            }
            command_buffers.push((buffer, program.len() as u32));
        }

        for handle in [
            input.buffer.handle,
            weights.buffer.handle,
            bias.buffer.handle,
            output.buffer.handle,
        ] {
            fini_bo(fd, handle).map_err(|error| format!("sync data BO: {error}"))?;
        }
        for (buffer, _) in &command_buffers {
            fini_bo(fd, buffer.buffer.handle)
                .map_err(|error| format!("sync regcmd BO: {error}"))?;
        }

        let tasks = command_buffers
            .iter()
            .map(|(buffer, count)| [(buffer.buffer.dma_address, *count)])
            .collect::<Vec<_>>();
        let input_handles = command_buffers
            .iter()
            .map(|(buffer, _)| {
                [
                    buffer.buffer.handle,
                    input.buffer.handle,
                    weights.buffer.handle,
                    bias.buffer.handle,
                ]
            })
            .collect::<Vec<_>>();
        let output_handles = [output.buffer.handle];
        let jobs = tasks
            .iter()
            .zip(&input_handles)
            .map(|(tasks, input_handles)| JobDesc {
                tasks,
                in_handles: input_handles,
                out_handles: &output_handles,
            })
            .collect::<Vec<_>>();

        let dispatch_started = std::time::Instant::now();
        submit_jobs(fd, &jobs).map_err(|error| format!("submit: {error}"))?;
        prep_bo(fd, output.buffer.handle, PER_CASE_TIMEOUT_NS)
            .map_err(|error| format!("completion wait: {error}"))?;
        record_dispatch_duration(dispatch_started.elapsed());

        let raw_output = std::slice::from_raw_parts(output.buffer.host_ptr, output_len).to_vec();

        // Everything below is off unless ROCKET_IOVA_LEDGER is set, and adds
        // no GEM buffers and no jobs. It runs here because this is the only
        // point where the dispatch is complete and every buffer it named is
        // still mapped.
        ledger_note_output_coverage(&raw_output, &output.buffer);
        ledger_scan_readonly_bo(
            fd,
            &input,
            &fixture.input,
            OwnedBuffer::poison_byte("INPUT"),
            "INPUT",
        );
        ledger_scan_readonly_bo(
            fd,
            &weights,
            &fixture.weights,
            OwnedBuffer::poison_byte("WEIGHTS"),
            "WEIGHTS",
        );
        ledger_scan_readonly_bo(
            fd,
            &bias,
            &fixture.bias,
            OwnedBuffer::poison_byte("BIAS"),
            "BIAS",
        );
        if ledger_enabled() {
            for ((buffer, _), program) in command_buffers.iter().zip(&programs) {
                let bytes = program
                    .iter()
                    .flat_map(|command| command.0.to_le_bytes())
                    .collect::<Vec<u8>>();
                ledger_scan_readonly_bo(fd, buffer, &bytes, 0, "REGCMD");
            }
        }
        ledger_recheck_output_drain(fd, &output, &raw_output);

        let output = match &accumulator_tiles {
            Some(tiles) => assemble_staged_accumulator_output(shape, kernels, &raw_output, tiles)?,
            None => raw_output.clone(),
        };
        Ok(CaseExecution {
            plan,
            output,
            #[cfg(feature = "hardware-characterization")]
            raw: raw_output,
        })
    }
}

fn execute_case(file: &std::fs::File, fixture: &Conv2dFixture) -> Result<CaseSuccess, String> {
    let execution = execute_case_output(file, fixture)?;
    let report = compare_output(fixture, &execution.plan, &execution.output);
    if report.mismatches != 0 {
        return Err(format!(
            "{} mismatches, max|diff|={}, tile_mismatches={:?}\n      {}",
            report.mismatches,
            report.max_abs_difference,
            report.tile_mismatches,
            report.samples.join("\n      "),
        ));
    }
    Ok(CaseSuccess {
        data_banks: execution.plan.data_banks(),
        weight_banks: execution.plan.weight_banks(),
        tiles: execution.plan.tiles().len(),
    })
}

fn cartesian_cases() -> Vec<Conv2dCase> {
    let mut cases = Vec::new();
    let input_channels = [3u32, 4, 5, 64, 128, 192, 256, 320, 384, 448, 512];
    let output_channels = [64u32, 128, 256, 512];

    // The broad Cartesian core: exactly the extent used by the expanded
    // vendor corpus, with both kernel sizes and both precisions.
    for precision in [OraclePrecision::Fp16, OraclePrecision::Int8] {
        for cin in input_channels {
            for cout in output_channels {
                for kernel in [1usize, 3] {
                    cases.push(Conv2dCase {
                        width: 28,
                        height: 28,
                        cin,
                        cout,
                        kernel: [kernel, kernel],
                        stride: 1,
                        padding: [kernel / 2, kernel / 2],
                        precision,
                        pattern: OraclePattern::Counting,
                    });
                }
            }
        }
    }

    // A smaller Cartesian selector grid catches indexing/permutation bugs
    // without doubling every expensive high-channel case.
    for precision in [OraclePrecision::Fp16, OraclePrecision::Int8] {
        for cin in [3u32, 5, 128, 256, 512] {
            for cout in [64u32, 256, 512] {
                cases.push(Conv2dCase {
                    width: 28,
                    height: 28,
                    cin,
                    cout,
                    kernel: [3, 3],
                    stride: 1,
                    padding: [1, 1],
                    precision,
                    pattern: if precision == OraclePrecision::Int8 {
                        OraclePattern::SelectorsAffine { phase: 0 }
                    } else {
                        OraclePattern::Selectors { phase: 0 }
                    },
                });
            }
        }
    }

    // IREE materializes VGG's padding as real zero-border pixels. These are
    // the nine unique physical shapes behind all sixteen convolutions.
    let vgg = [
        (226, 3, 64),
        (226, 64, 64),
        (114, 64, 128),
        (114, 128, 128),
        (58, 128, 256),
        (58, 256, 256),
        (30, 256, 512),
        (30, 512, 512),
        (16, 512, 512),
    ];
    for precision in [OraclePrecision::Fp16, OraclePrecision::Int8] {
        for (extent, cin, cout) in vgg {
            for selector_case in [false, true] {
                let pattern = if selector_case {
                    if precision == OraclePrecision::Int8 {
                        OraclePattern::SelectorsAffine { phase: 1 }
                    } else {
                        OraclePattern::Selectors { phase: 1 }
                    }
                } else {
                    OraclePattern::Counting
                };
                cases.push(Conv2dCase {
                    width: extent,
                    height: extent,
                    cin,
                    cout,
                    kernel: [3, 3],
                    stride: 1,
                    padding: [0, 0],
                    precision,
                    pattern,
                });
            }
        }
    }
    cases
}

/// Confirms the mechanism behind an apparent output-channel-group cutoff
/// observed during the original Cartesian investigation.
///
/// Focused measurements found every one of 224/225/32/33 exact -- the cutoff
/// is not a fixed per-task channel count, and 256/512 failing is not evidence
/// of missing output-channel-group task splitting after all. Host-side
/// `ConvPlan::data_banks`/`weight_banks` explains it instead:
/// `demand_based_cbuf_partition`'s starved-to-`streamed_preference` branch
/// (`conv.rs` around line 1183) returns as soon as coefficient demand
/// exceeds one bank, without ever comparing against the independently
/// hardware-validated `weight_banks_floor` minimum a few lines below it --
/// that comparison is unreachable once the earlier branch has already
/// returned. For Cin 3/4/5 K3 fp16, weight_channels is 8: `floor(8) = 3`
/// banks, but `streamed_preference` is only 1, so the plan starves itself to
/// 1 weight bank the moment coefficient demand needs more than one. Walking
/// `ConvPlan::data_banks`/`weight_banks` across Cout by hand finds the exact
/// bank-count flip (1/11 -> 11/1, etc.) at:
///
/// - fp16 K3, Cin 3/4/5: Cout 227 (still 1/11) -> 228 (11/1)
/// - int8 K3, Cin 3/4/5: Cout 226 (still 1/11) -> 227 (11/1)
/// - fp16 K1, Cin 384: Cout 426 (still 2/10) -> 427 (11/1)
/// - fp16 K1, Cin 448: Cout 365 (still 2/10) -> 366 (11/1)
///
/// If this is the real mechanism, each pair here should flip from correct to
/// all-zero-past-some-channel at exactly its listed boundary, with no gap
/// like the 225-vs-256 one that motivated this probe.
fn bank_partition_flip_boundary_cases() -> Vec<Conv2dCase> {
    let mut cases = Vec::new();
    for (precision, boundary) in [
        (OraclePrecision::Fp16, 227u32),
        (OraclePrecision::Int8, 226),
    ] {
        for cin in [3u32, 4, 5] {
            for cout in [boundary, boundary + 1] {
                cases.push(Conv2dCase {
                    width: 28,
                    height: 28,
                    cin,
                    cout,
                    kernel: [3, 3],
                    stride: 1,
                    padding: [1, 1],
                    precision,
                    pattern: OraclePattern::Counting,
                });
            }
        }
    }
    for (cin, boundary) in [(384u32, 426u32), (448, 365)] {
        for cout in [boundary, boundary + 1] {
            cases.push(Conv2dCase {
                width: 28,
                height: 28,
                cin,
                cout,
                kernel: [1, 1],
                stride: 1,
                padding: [0, 0],
                precision: OraclePrecision::Fp16,
                pattern: OraclePattern::Counting,
            });
        }
    }
    cases
}

/// A real fp16 VGG-19 run through blocks 4-5 (all four Cin/Cout-512
/// convolutions) previously showed severe end-to-end disagreement against
/// CPU -- MAE roughly 30x worse than the same run capped at Cin/Cout <= 256,
/// with only 20.70% of logits within the 0.05 tolerance that the <= 256 run hit
/// 100% on. None of these shapes are affected by the CBUF bank-partition
/// bug fixed above (VGG never asks for small Cin with large Cout, or the
/// specific high-Cin K1 combinations that bug needed), and the Cartesian
/// oracle's `Counting`/`Selectors` patterns already pass at exactly these
/// shapes. The leading hypothesis is that something specific to *dense,
/// fully-diverse* coefficient data -- every tap and channel nonzero and
/// distinct, as real trained weights are -- breaks in a way neither of
/// those patterns exercises (`Counting` is uniform 1s; `Selectors` is
/// three nonzero taps per output with everything else zero).
///
/// This runs `Dense` at the two suspect Cin/Cout-512 shapes (VGG blocks 4
/// and 5) alongside the same pattern at a known-safe Cin/Cout-256 shape
/// (VGG block 3) as a control: if block 3 passes and blocks 4/5 fail, that
/// isolates the effect to Cin/Cout 512 specifically rather than to
/// something wrong with the `Dense` pattern or harness itself.
///
/// Status (2026-08-30, `f6843e4`): all five cases pass on RK3588 hardware
/// after the CBUF partition fixes. Keep this as a regression gate because
/// the earlier Counting/Selectors coverage passed while a real trained-weight
/// model did not. `tools/e2e_conv_regression.py` cross-builds and runs this exact
/// test before checking the compiler -> VMFB -> public-driver path too.
fn dense_coefficient_vgg_block_cases() -> Vec<Conv2dCase> {
    let mut cases = Vec::new();
    for (extent, cin, cout) in [
        // VGG block 3 control -- known-safe, Counting/Selectors already
        // pass here, and it is well outside the CBUF bank-partition bug's
        // range.
        (58u32, 128u32, 256u32),
        (58, 256, 256),
        // VGG block 4 -- the two suspect shapes.
        (30, 256, 512),
        (30, 512, 512),
        // VGG block 5 -- the other suspect shape.
        (16, 512, 512),
    ] {
        cases.push(Conv2dCase {
            width: extent,
            height: extent,
            cin,
            cout,
            kernel: [3, 3],
            stride: 1,
            padding: [0, 0],
            precision: OraclePrecision::Fp16,
            pattern: OraclePattern::Dense { phase: 0 },
        });
    }
    cases
}

/// Hardware-confirmed exact i32-accumulator regression cases through the
/// shared production oracle. Together these cover a partial second output
/// block, a partial third block at stride 2, the K3 Cin=32 ceiling, the K1
/// transition from one tile at Cin=352 to two at Cin=353, the tile-local
/// contract's current Cin=384 ceiling, a large three-tile image, and Cout=512
/// ceilings with dense signed data.
fn int8_accumulator_regression_cases() -> Vec<Conv2dCase> {
    vec![
        Conv2dCase {
            width: 9,
            height: 7,
            cin: 5,
            cout: 63,
            kernel: [1, 1],
            stride: 1,
            padding: [0, 0],
            precision: OraclePrecision::Int8Accumulator,
            pattern: OraclePattern::Dense { phase: 0 },
        },
        Conv2dCase {
            width: 34,
            height: 34,
            cin: 32,
            cout: 64,
            kernel: [3, 3],
            stride: 1,
            padding: [0, 0],
            precision: OraclePrecision::Int8Accumulator,
            pattern: OraclePattern::Dense { phase: 2 },
        },
        Conv2dCase {
            width: 32,
            height: 32,
            cin: 352,
            cout: 64,
            kernel: [1, 1],
            stride: 1,
            padding: [0, 0],
            precision: OraclePrecision::Int8Accumulator,
            pattern: OraclePattern::Dense { phase: 1 },
        },
        Conv2dCase {
            width: 32,
            height: 32,
            cin: 353,
            cout: 64,
            kernel: [1, 1],
            stride: 1,
            padding: [0, 0],
            precision: OraclePrecision::Int8Accumulator,
            pattern: OraclePattern::Dense { phase: 1 },
        },
        Conv2dCase {
            width: 32,
            height: 32,
            cin: 384,
            cout: 64,
            kernel: [1, 1],
            stride: 1,
            padding: [0, 0],
            precision: OraclePrecision::Int8Accumulator,
            pattern: OraclePattern::Dense { phase: 1 },
        },
        Conv2dCase {
            width: 32,
            height: 32,
            cin: 16,
            cout: 512,
            kernel: [1, 1],
            stride: 1,
            padding: [0, 0],
            precision: OraclePrecision::Int8Accumulator,
            pattern: OraclePattern::Dense { phase: 3 },
        },
        Conv2dCase {
            width: 34,
            height: 34,
            cin: 16,
            cout: 512,
            kernel: [3, 3],
            stride: 1,
            padding: [0, 0],
            precision: OraclePrecision::Int8Accumulator,
            pattern: OraclePattern::Dense { phase: 4 },
        },
        Conv2dCase {
            width: 33,
            height: 33,
            cin: 16,
            cout: 65,
            kernel: [3, 3],
            stride: 2,
            padding: [0, 0],
            precision: OraclePrecision::Int8Accumulator,
            pattern: OraclePattern::Dense { phase: 5 },
        },
        Conv2dCase {
            width: 226,
            height: 226,
            cin: 3,
            cout: 64,
            kernel: [3, 3],
            stride: 1,
            padding: [0, 0],
            precision: OraclePrecision::Int8Accumulator,
            pattern: OraclePattern::Selectors { phase: 6 },
        },
    ]
}

/// Sweeps every supported native 16-channel input atom through the measured
/// dense K1 accumulator ceiling. Cin 16..=384 passes exactly, including the
/// transition to two tiles above Cin=352. Cin 385+ is rejected by
/// `Shape::parity_padded_shape` rather than submitted to hardware.
fn int8_accumulator_k1_supported_cin_atom_cases() -> Vec<Conv2dCase> {
    (16..=384)
        .step_by(16)
        .map(|cin| Conv2dCase {
            width: 32,
            height: 32,
            cin,
            cout: 64,
            kernel: [1, 1],
            stride: 1,
            padding: [0, 0],
            precision: OraclePrecision::Int8Accumulator,
            pattern: OraclePattern::Dense { phase: 1 },
        })
        .collect()
}

/// Guards the exact logical-Cin planning transition: Cin 351 and 352 use one
/// tile, while 353, 354, and 367 use two. All five pass on RK3588 when each
/// accumulator tile owns its destination geometry (2026-08-31).
fn int8_accumulator_k1_cin_boundary_cases() -> Vec<Conv2dCase> {
    [351u32, 352, 353, 354, 367]
        .into_iter()
        .map(|cin| Conv2dCase {
            width: 32,
            height: 32,
            cin,
            cout: 64,
            kernel: [1, 1],
            stride: 1,
            padding: [0, 0],
            precision: OraclePrecision::Int8Accumulator,
            pattern: OraclePattern::Dense { phase: 1 },
        })
        .collect()
}

/// Shapes the first expanded oracle run (RK3588, 2026-08-31) proved require
/// programmed-Cout parity padding. They are now a compact HAL regression
/// matrix: the logical cases below are translated by `build_fixture` through
/// `Shape::parity_padded_shape`, packed with zero surplus coefficients, and
/// compared only across their logical output channels.
///
/// * odd 9x7 output extents leave the final logical accumulator(s) unwritten;
/// * several small-Cin shapes corrupt most values at exact 32-lane block
///   boundaries.
/// Re-measured on `planck` 2026-08-31, each case alone in a fresh process
/// via `ROCKET_PROBE_ONLY` -- the only way to get an uncontaminated verdict,
/// since a case's result shifts with what ran before it. Counts below are
/// isolated runs on a healthy device; two entries turned out not to be
/// limitations at all:
///
///   * `Cin=3  Cout=1  1x1` -- fails 12/12. Real.
///   * `Cin=3  Cout=31 3x3` -- fails 12/12. Real.
///   * `Cin=3  Cout=32 3x3` -- fails 12/12. Real.
///   * `Cin=3  Cout=33 3x3` -- **passes 48/54**. Every one of the six
///     failures landed in a window right after a heavy failing sweep, so
///     this is device contamination, not a shape limit.
///   * `Cin=5  Cout=64 1x1` -- **passes 37/37**. Not a limitation; it only
///     ever failed as the fifth case of a sweep.
///   * `Cin=5  Cout=65 1x1` -- fails 12/12. Real.
///
/// All six points are explained by the parity rule in
/// `int8_accumulator_output_parity_cases`: a dense accumulator conv is
/// correct exactly when `tile_pixels * blocks_per_pixel` is even. Every case
/// here is 9x7, which is 63 pixels -- odd -- so parity falls to the block
/// count. Cout 33 and 64 give two blocks and pass; Cout 1, 31, 32 give one
/// and Cout 65 gives three, and those fail. Nothing here is specific to Cin
/// or the kernel, and none of it is specific to Cout either: the same Cout
/// values all pass at an even pixel count.
/// Dense fp16 geometries that the existing stride coverage misses.
///
/// Before this list, dense convolution above stride 1 was checked in exactly
/// two places, and neither could see the two defects below:
///
///   * `conv_wide_shape_hw.rs` runs Cin=3, Cout=8, 1x1 and 3x3 at strides 1
///     to 4 -- but with uniform 1.0 inputs *and* weights, so every output is
///     just a count of valid taps. That is blind to which input pixels were
///     read, as long as the right number of them were.
///   * `cartesian_cases` covers Cin 3/4/5 densely, but only at stride 1.
///
/// The one stride-2 dense case anywhere (33x33, Cin=16, Cout=65, in
/// `int8_accumulator_phase_cases`) happens to use an extent where
/// `(width - kernel)` divides by the stride exactly, which is precisely the
/// case that works.
///
/// # What these cover
///
/// *Window parity.* At stride > 1 an input extent where
/// `(extent - kernel) % stride != 0` leaves a partial trailing window that no
/// output tap consumes. `ColumnTile::from_output_range` then derives an
/// `in_cols` *smaller* than the row it sits in, and that value is programmed
/// as `CNA_DATA_SIZE0.datain_width` while `line_stride` keeps the full row
/// pitch. The pairs below straddle that boundary at three sizes: 33/35/65
/// divide exactly, 34/36/66 do not.
///
/// *Sub-atom input channels.* One 16-byte feature atom holds 8 fp16
/// channels. Cin 1/2/4 do not fill one; 8/16 do. Run at stride 1 so the
/// window-parity axis above is held fixed.
///
/// # A known flake in this list
///
/// `34x34 Cin=8` intermittently fails when the matrix runs in order, and
/// passes every time under `ROCKET_PROBE_ONLY`. That is the per-process
/// contamination `run_hardware_case_matrix` documents, not a verdict on the
/// shape: it flakes identically with the CBUF residency fix reverted, and its
/// plan (one tile, 34 rows) is unaffected by anything here. Isolate before
/// believing a failure on this row.
///
/// `OraclePattern::Dense` is the point of all of these: every tap and channel
/// carries a distinct nonzero weight, so a wrong-pixel or wrong-channel read
/// changes the result instead of cancelling out. Inputs are in [-3, 3] and
/// weights in {-2, -1, 1, 2}, so the largest accumulator any of these can
/// produce is `6 * 16 * 9 = 864`, well inside the 2048 where fp16 still
/// represents every integer exactly and the oracle's zero tolerance holds.
/// Geometries either side of the CBUF residency boundary.
///
/// The surface feature charge is counted in whole `data_entries` entries of
/// four atoms and rounds *up*, so a row whose atom count is not a multiple of
/// four still occupies its whole final entry.
/// `Shape::max_tile_input_rows_for_width_and_data_banks` used to charge bare
/// atoms instead, letting a tile claim rows that do not fit. The overflow
/// lands on the tail of the tile's last input row, which surfaces as the last
/// output row of that tile being wrong from some column onwards while every
/// earlier row is exact -- a single wrong row in an otherwise perfect result,
/// which is exactly the shape of failure a coarse tolerance check misses.
///
/// At Cin=16 fp16 a row is two atoms per pixel, so odd widths are the ones
/// that round up. The pairs below sit either side of the bound at three
/// widths, and the height sweep walks one width across it a row at a time:
/// 113x113 wanted 5643 entries against the 5632 available, while 115x113
/// fits at 5626 and has to keep working.
fn cbuf_residency_boundary_cases() -> Vec<Conv2dCase> {
    let dense = |width: u32, height: u32| Conv2dCase {
        width,
        height,
        cin: 16,
        cout: 16,
        kernel: [3, 3],
        stride: 2,
        padding: [0, 0],
        precision: OraclePrecision::Fp16,
        pattern: OraclePattern::Dense { phase: 0 },
    };

    let mut cases = Vec::new();
    // Widths that round up (odd) against their even neighbours, at a height
    // that forces a multi-tile split.
    for width in [109u32, 111, 112, 113, 114, 115, 117, 119] {
        cases.push(dense(width, 113));
    }
    // One width walked across the bound by height alone. Every one of these
    // is a single tile, which is what rules out tile *splitting* as the
    // cause: 113x99 is one tile that does not fit.
    for height in [91u32, 93, 95, 97, 99] {
        cases.push(dense(113, height));
    }
    cases
}

/// Every distinct CPU-routed fp16 pointwise MobileNetV2 shape.
///
/// These are the nine signatures behind the model's 17 remaining pointwise
/// placements. They are deliberately isolated from the model: a passing case
/// remains an ordinary ignored board test, while a failing case is evidence to
/// keep (or add) a compiler feature gate rather than widen the matcher.
///
/// `ROCKET_ALLOW_UNBACKED_CHANNELS` is set by the two callers below. It is a
/// characterization-only escape hatch for the capture-backed Cin/Cout limits;
/// the production compiler never sets it. Each signature is checked with
/// `Counting`, `Selectors`, and varied `Dense` values, covering complete
/// accumulation, production weight layout, and ordinary arithmetic at every
/// output lane.
fn fp16_mobilenetv2_remaining_pointwise_cases() -> Vec<Conv2dCase> {
    [
        // 14x14: 3 + 1 + 3 + 2 = 9 model placements.
        (14, 528, 88),
        (14, 528, 136),
        (14, 136, 816),
        (14, 816, 136),
        // 7x7: 1 + 3 + 2 + 1 + 1 = 8 model placements.
        (7, 816, 224),
        (7, 224, 1344),
        (7, 1344, 224),
        (7, 1344, 448),
        (7, 448, 1792),
    ]
    .into_iter()
    .flat_map(|(width, cin, cout)| {
        [
            OraclePattern::Counting,
            OraclePattern::Selectors { phase: 0 },
            OraclePattern::Dense { phase: 0 },
        ]
        .into_iter()
        .map(move |pattern| Conv2dCase {
            width,
            height: width,
            cin,
            cout,
            kernel: [1, 1],
            stride: 1,
            padding: [0, 0],
            precision: OraclePrecision::Fp16,
            pattern,
        })
    })
    .collect()
}

fn dense_geometry_regression_cases() -> Vec<Conv2dCase> {
    let mut cases = Vec::new();
    let dense = |width: u32, height: u32, cin: u32, kernel: usize, stride: u32| Conv2dCase {
        width,
        height,
        cin,
        cout: 16,
        kernel: [kernel, kernel],
        stride,
        padding: [0, 0],
        precision: OraclePrecision::Fp16,
        pattern: OraclePattern::Dense { phase: 0 },
    };

    // Window parity at stride 2: exact / leftover / exact / leftover / ...
    for extent in [33u32, 34, 35, 36, 65, 66] {
        cases.push(dense(extent, extent, 16, 3, 2));
    }
    // The same boundary at stride 3 and 4. `(extent - 3) % stride`: 33 and 36
    // divide at stride 3, 35 divides at stride 4, the others do not.
    for extent in [33u32, 34, 35, 36] {
        cases.push(dense(extent, extent, 16, 3, 3));
        cases.push(dense(extent, extent, 16, 3, 4));
    }
    // A 1x1 control: at kernel 1 the derived `in_cols` always reaches the end
    // of the row, so these should pass at every extent and pin that down.
    for extent in [32u32, 33] {
        cases.push(dense(extent, extent, 16, 1, 2));
    }
    // Sub-atom input-channel counts, stride 1 so only Cin varies. 8 fp16
    // channels fill one 16-byte feature atom; 1/2/4 do not.
    for cin in [1u32, 2, 4, 8, 16, 32] {
        cases.push(dense(34, 34, cin, 3, 1));
    }
    cases
}

fn int8_accumulator_parity_regression_cases() -> Vec<Conv2dCase> {
    let mut cases = Vec::new();
    for (cin, cout, kernel) in [
        (3u32, 1u32, 1usize),
        (3, 31, 3),
        (3, 32, 3),
        (3, 33, 3),
        (5, 64, 1),
        (5, 65, 1),
    ] {
        cases.push(Conv2dCase {
            width: 9,
            height: 7,
            cin,
            cout,
            kernel: [kernel, kernel],
            stride: 1,
            padding: [kernel / 2, kernel / 2],
            precision: OraclePrecision::Int8Accumulator,
            pattern: OraclePattern::Dense { phase: 0 },
        });
    }
    cases
}

/// Cout swept across the accumulator's 128-byte output-surface boundaries.
///
/// `Shape::output_atom_bytes` is 128 for a dense int8 accumulator -- 32 i32
/// channels per surface -- so this walks Cout over every multiple of 32 and
/// the values either side of it.
///
/// Measured on `planck` 2026-08-31, isolated via `ROCKET_PROBE_ONLY`. At
/// 9x7 the result is exact at all 21 points in both families: every odd
/// block count fails and every even one passes.
///
/// **That reading was too narrow.** Both families here are 9x7, which is 63
/// pixels -- odd -- so the block count alone decided the parity of
/// `pixels * blocks`. At 4x4, 16 pixels, *every* Cout passes including the
/// odd block counts. The rule this sweep was really measuring is the one in
/// `int8_accumulator_output_parity_cases`: `tile_pixels * blocks_per_pixel`
/// must be even. Read the table below as "at 63 pixels", not as a law about
/// Cout.
///
/// | blocks/pixel | Cout range   | result |
/// |--------------|--------------|--------|
/// | 1            | 1..=32       | fails  |
/// | 2            | 33..=64      | passes |
/// | 3            | 65..=96      | fails  |
/// | 4            | 97..=128     | passes |
/// | 5            | 129..=160    | fails  |
/// | 6, 8         | 161..=192,256| passes |
///
/// This subsumes every entry of `int8_accumulator_parity_regression_cases`,
/// all of which are 9x7.
///
/// Raising the accumulator output granule to 64, so the padded count always
/// lands on an even block boundary, changes nothing on hardware -- so the
/// rounding in `Shape::padded_out_channels` is not the mechanism. The
/// mechanism is in `int8_accumulator_output_layout_probe`: the DPU commits
/// accumulator output in 256-byte units, and the 128-byte block model only
/// coincides with that when the total block count is even.
///
/// Measure with `ROCKET_PROBE_ONLY`, one case per process: a plain sweep
/// contaminates its own later rows, and even isolated runs are skewed by a
/// recent history of failures -- Cout 33/40/48 at K1x1 read as failures in a
/// first pass and pass 12/12 once measured on a rested device.
fn int8_accumulator_cout_sweep_cases() -> Vec<Conv2dCase> {
    const COUT: [u32; 21] = [
        1, 2, 8, 16, 31, 32, 33, 40, 48, 63, 64, 65, 80, 96, 127, 128, 129, 160, 192, 255, 256,
    ];
    let mut cases = Vec::new();
    for (cin, kernel) in [(5u32, 1usize), (3, 3)] {
        for cout in COUT {
            cases.push(Conv2dCase {
                width: 9,
                height: 7,
                cin,
                cout,
                kernel: [kernel, kernel],
                stride: 1,
                padding: [kernel / 2, kernel / 2],
                precision: OraclePrecision::Int8Accumulator,
                pattern: OraclePattern::Dense { phase: 0 },
            });
        }
    }
    cases
}

/// Does the even-blocks Cout rule actually depend only on Cout?
///
/// `int8_accumulator_cout_sweep_cases` established the rule at 9x7 with
/// Cin=5 and Cin=3, and it held at all 21 Cout values in both. But the
/// layout probe's own control shape, 4x4 with Cin=8, *passes* at Cout 32 and
/// 96 -- both odd block counts the rule says must fail. So the rule is not a
/// property of Cout alone, and this matrix is what separates the spatial
/// size from Cin.
///
/// Ordered Cin-major within each spatial size so a sweep walks one variable
/// at a time. Measure with `ROCKET_PROBE_ONLY`.
fn int8_accumulator_cout_shape_interaction_cases() -> Vec<Conv2dCase> {
    let mut cases = Vec::new();
    for (width, height) in [(4u32, 4u32), (9, 7)] {
        for cin in [3u32, 5, 8, 16] {
            for cout in [32u32, 64, 96] {
                cases.push(Conv2dCase {
                    width,
                    height,
                    cin,
                    cout,
                    kernel: [1, 1],
                    stride: 1,
                    padding: [0, 0],
                    precision: OraclePrecision::Int8Accumulator,
                    pattern: OraclePattern::Dense { phase: 0 },
                });
            }
        }
    }
    cases
}

/// Tests the unified parity rule the Cout and shape sweeps converge on.
///
/// The Cout rule was never about Cout. 4x4 passes at every Cout, including
/// the odd block counts that fail at 9x7 -- and 4x4 has 16 pixels while 9x7
/// has 63. What both sweeps were really measuring is
///
/// > `tile_pixels * blocks_per_pixel` must be **even**,
///
/// i.e. a tile's output must be a whole number of 256-byte units, as if the
/// DPU commits output two 128-byte blocks at a time. An even pixel count
/// satisfies it for any Cout, which is why 4x4 never fails; an odd one
/// leaves the parity to the block count, which is why 9x7 tracks Cout.
///
/// These twelve cases are chosen so pixel parity and block parity disagree,
/// which the earlier sweeps never did -- both held one fixed while moving
/// the other. Predictions, all at Cin=8 and 1x1:
///
/// | shape | pixels | Cout | blocks | product | predicted |
/// |-------|--------|------|--------|---------|-----------|
/// | 3x3   | 9 odd  |  32  | 1      |  9 odd  | fails     |
/// | 3x3   | 9 odd  |  64  | 2      | 18 even | passes    |
/// | 3x3   | 9 odd  |  96  | 3      | 27 odd  | fails     |
/// | 4x4   | 16 even|  32  | 1      | 16 even | passes    |
/// | 4x4   | 16 even|  96  | 3      | 48 even | passes    |
/// | 5x5   | 25 odd |  64  | 2      | 50 even | passes    |
/// | 5x5   | 25 odd |  96  | 3      | 75 odd  | fails     |
/// | 6x6   | 36 even|  32  | 1      | 36 even | passes    |
/// | 9x7   | 63 odd | 128  | 4      | 252 even| passes    |
/// | 9x7   | 63 odd | 160  | 5      | 315 odd | fails     |
/// | 8x8   | 64 even| 160  | 5      | 320 even| passes    |
/// | 8x8   | 64 even|  32  | 1      | 64 even | passes    |
/// | 3x4   | 12 even|  32  | 1      | 12 even | passes    |
/// | 4x3   | 12 even|  32  | 1      | 12 even | passes    |
/// | 9x8   | 72 even|  32  | 1      | 72 even | passes    |
/// | 5x5   | 25 odd |  32  | 1      | 25 odd  | fails     |
///
/// The last four are the padding question, and they **pass on hardware**:
/// 3x3 and 9x7 fail at Cout=32, while 3x4, 4x3 and 9x8 -- the same shapes
/// one output row or column larger -- are all correct, with 5x5 staying odd
/// as the control. Computing one extra output row or column and discarding
/// it is therefore a real fix lever, not just an arithmetic one.
///
/// **Pad the width, not the height.** `tile_pixels` is per *tile*:
/// `tile.rows.out_rows * tile.columns.out_cols`, a 2D subsection of the
/// output that `plan_grid` produces by splitting columns and then greedily
/// splitting rows to CBUF capacity. Every tile has to satisfy the parity
/// independently. An even `out_cols` makes `rows * out_cols` even for *any*
/// row split, so padding the width to even is robust against however the
/// planner tiles. An even output *height* only helps while the shape stays
/// one tile -- greedy row splitting readily produces odd per-tile row counts
/// (8 rows becoming 5+3), and each of those tiles would then violate the
/// parity on its own. The shapes here are all single-tile, so they cannot
/// distinguish the two; that distinction is untested.
/// | 8x8   | 64 even|  32  | 1      | 64 even | passes    |
fn int8_accumulator_output_parity_cases() -> Vec<Conv2dCase> {
    [
        (3u32, 3u32, 32u32),
        (3, 3, 64),
        (3, 3, 96),
        (4, 4, 32),
        (4, 4, 96),
        (5, 5, 64),
        (5, 5, 96),
        (6, 6, 32),
        (9, 7, 128),
        (9, 7, 160),
        (8, 8, 160),
        (8, 8, 32),
        // Does *one extra output row or column* rescue a failing shape?
        // 3x3 Cout=32 fails at 9 pixels; 3x4 and 4x3 are the same shape one
        // row or one column larger, 12 pixels, even. 9x7 Cout=32 fails at 63;
        // 9x8 is one row more, 72, even. 5x5 stays odd at 25 as the control
        // that the extra row is what matters, not the size change.
        (3, 4, 32),
        (4, 3, 32),
        (9, 8, 32),
        (5, 5, 32),
    ]
    .into_iter()
    .map(|(width, height, cout)| Conv2dCase {
        width,
        height,
        cin: 8,
        cout,
        kernel: [1, 1],
        stride: 1,
        padding: [0, 0],
        precision: OraclePrecision::Int8Accumulator,
        pattern: OraclePattern::Dense { phase: 0 },
    })
    .collect()
}

#[cfg(feature = "hardware-characterization")]
fn int8_neutral80_one_hot_four_way_cases() -> Vec<Conv2dCase> {
    let mut cases = Vec::with_capacity(4);
    for signed_input in [false, true] {
        for kernel in [1usize, 3] {
            cases.push(Conv2dCase {
                width: 9,
                height: 7,
                cin: 3,
                cout: 32,
                kernel: [kernel, kernel],
                stride: 1,
                // No padding keeps every selected K3 tap in-bounds, making
                // a tap-order failure distinct from padding semantics.
                padding: [0, 0],
                precision: OraclePrecision::Int8,
                pattern: OraclePattern::OneHotNeutral80 {
                    phase: 2,
                    signed_input,
                },
            });
        }
    }
    cases
}

#[cfg(feature = "hardware-characterization")]
fn int8_raw_coefficient_byte_sweep_case(unit_gain: bool) -> Conv2dCase {
    Conv2dCase {
        width: 1,
        height: 1,
        cin: 1,
        cout: 256,
        kernel: [1, 1],
        stride: 1,
        padding: [0, 0],
        precision: OraclePrecision::Int8,
        pattern: if unit_gain {
            OraclePattern::RawByteSweepUnit
        } else {
            OraclePattern::RawByteSweep
        },
    }
}

fn panic_message(payload: Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else {
        "non-string panic payload".to_string()
    }
}

fn assert_planable_and_gap_free(cases: Vec<Conv2dCase>) {
    for case in cases {
        // Checked here rather than only in `build_fixture` so a vacuous case
        // fails in the default host run, not on the board: every case set in
        // this file that has a hardware test also has one of these.
        if let Err(error) = expected_output_carries_signal(case) {
            panic!("{error}");
        }
        let shape = case
            .shape()
            .parity_padded_shape(case.kernel)
            .expect("supported oracle case should have a programmed HAL shape");
        let plan = ConvPlan::new(shape, case.kernel);
        let out_width = shape.output_width(case.kernel) as usize;
        let out_height = shape.output_height(case.kernel) as usize;
        let mut coverage = vec![0u8; out_width * out_height];
        for tile in plan.tiles() {
            let tile_row_capacity = shape.max_tile_input_rows_for_width_and_data_banks(
                tile.columns.in_cols,
                plan.data_banks(),
            );
            assert!(
                tile.rows.in_rows <= tile_row_capacity,
                "{}: tile reads {} rows beyond capacity {}",
                case.label(),
                tile.rows.in_rows,
                tile_row_capacity,
            );
            assert!(
                shape.dense_feature_offset_safe(tile.rows.in_first),
                "{}: tile at in_first={} has input byte offset {} mod 16",
                case.label(),
                tile.rows.in_first,
                tile.rows.in_first * shape.input_row_stride() % 16,
            );
            for y in
                tile.rows.out_first as usize..(tile.rows.out_first + tile.rows.out_rows) as usize
            {
                for x in tile.columns.out_first as usize
                    ..(tile.columns.out_first + tile.columns.out_cols) as usize
                {
                    coverage[y * out_width + x] += 1;
                }
            }
        }
        assert!(
            coverage.iter().all(|count| *count == 1),
            "{}: output coverage has gaps or overlaps",
            case.label(),
        );
    }
}

#[test]
fn cartesian_matrix_is_planable_and_gap_free() {
    let cases = cartesian_cases();
    assert_eq!(cases.len(), 242);
    assert_planable_and_gap_free(cases);
}

#[test]
fn bank_partition_flip_boundary_matrix_is_planable_and_gap_free() {
    let cases = bank_partition_flip_boundary_cases();
    assert_eq!(cases.len(), 16);
    assert_planable_and_gap_free(cases);
}

#[test]
fn dense_coefficient_vgg_block_matrix_is_planable_and_gap_free() {
    let cases = dense_coefficient_vgg_block_cases();
    assert_eq!(cases.len(), 5);
    assert_planable_and_gap_free(cases);
}

#[test]
fn int8_accumulator_matrices_are_planable_and_gap_free() {
    let regression = int8_accumulator_regression_cases();
    assert_eq!(regression.len(), 9);
    assert_eq!(
        ConvPlan::new(regression[3].shape(), regression[3].kernel)
            .tiles()
            .len(),
        2
    );
    assert_eq!(
        ConvPlan::new(regression[8].shape(), regression[8].kernel)
            .tiles()
            .len(),
        3
    );
    assert_planable_and_gap_free(regression);

    let parity_regression = int8_accumulator_parity_regression_cases();
    assert_eq!(parity_regression.len(), 6);
    assert_planable_and_gap_free(parity_regression);
}

#[test]
fn int8_accumulator_k1_supported_cin_atoms_are_planable_and_gap_free() {
    let cases = int8_accumulator_k1_supported_cin_atom_cases();
    assert_eq!(cases.len(), 24);
    assert_eq!(cases.first().unwrap().cin, 16);
    assert_eq!(cases.last().unwrap().cin, 384);
    assert!(cases.windows(2).all(|pair| pair[1].cin - pair[0].cin == 16));
    assert_planable_and_gap_free(cases);
}

#[test]
fn int8_accumulator_k1_cin_boundary_is_planable_and_gap_free() {
    let cases = int8_accumulator_k1_cin_boundary_cases();
    assert_eq!(
        cases.iter().map(|case| case.cin).collect::<Vec<_>>(),
        [351, 352, 353, 354, 367]
    );
    assert_eq!(
        ConvPlan::new(cases[1].shape(), cases[1].kernel)
            .tiles()
            .len(),
        1
    );
    assert_eq!(
        ConvPlan::new(cases[2].shape(), cases[2].kernel)
            .tiles()
            .len(),
        2
    );
    assert_planable_and_gap_free(cases);
}

#[cfg(feature = "hardware-characterization")]
#[test]
fn int8_accumulator_cbuf_split_characterization_matrix_is_planable() {
    for cin in [384u32, 385, 400, 512] {
        let case = Conv2dCase {
            width: 32,
            height: 32,
            cin,
            cout: 64,
            kernel: [1, 1],
            stride: 1,
            padding: [0, 0],
            precision: OraclePrecision::Int8Accumulator,
            pattern: OraclePattern::Dense { phase: 1 },
        };
        for weight_banks in 1..=5 {
            let data_banks = 12 - weight_banks;
            let plan =
                ConvPlan::with_cbuf_banks(case.shape(), case.kernel, data_banks, weight_banks);
            assert_eq!(plan.data_banks(), data_banks, "Cin={cin}");
            assert_eq!(plan.weight_banks(), weight_banks, "Cin={cin}");
            assert!(
                !plan.tiles().is_empty(),
                "Cin={cin}, {data_banks}/{weight_banks}"
            );
        }
    }
}

/// Index to resume a hardware sweep at, for picking one back up after a
/// reboot has cleared a sick device. Zero, the default, starts at the top.
fn probe_resume_index() -> usize {
    std::env::var("ROCKET_PROBE_RESUME_AT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}

/// Case index to run entirely on its own, for an uncontaminated verdict on a
/// single shape.
///
/// Results in these sweeps are order dependent -- a case can pass when it
/// runs first and fail from the same sweep a few cases later -- so the only
/// trustworthy measurement of one shape is that shape running first in a
/// fresh process. Loop this over the indices, several passes each, rather
/// than reading one row out of one sweep. Takes precedence over
/// `ROCKET_PROBE_RESUME_AT`.
fn probe_only_index() -> Option<usize> {
    std::env::var("ROCKET_PROBE_ONLY")
        .ok()
        .and_then(|value| value.parse().ok())
}

/// A shape well inside the known-good region, used to prove the device is
/// still healthy. Dense signed K1 accumulator at Cin=64 passes on any
/// healthy RK3588, so a failure here is a statement about the *device*
/// rather than about whatever shape is under test.
fn accumulator_canary_passes(file: &std::fs::File) -> bool {
    let case = Conv2dCase {
        width: 32,
        height: 32,
        cin: 64,
        cout: 64,
        kernel: [1, 1],
        stride: 1,
        padding: [0, 0],
        precision: OraclePrecision::Int8Accumulator,
        pattern: OraclePattern::Dense { phase: 1 },
    };
    let result = catch_unwind(AssertUnwindSafe(|| {
        let fixture = build_fixture(case)?;
        execute_case(file, &fixture)
    }));
    matches!(result, Ok(Ok(_)))
}

/// Runs a case matrix, accumulating shape failures but halting on a sick
/// device.
///
/// Shape failures accumulate exactly as they always did: one bad case never
/// prevents later cases from running. The one exception is the RK3588's sick
/// state. After roughly fifteen to twenty failing jobs the NPU starts
/// returning wrong results for *every* shape, including ones that passed
/// seconds earlier, and nothing reports it -- `prep_bo` still succeeds well
/// inside its timeout. Past that point a sweep measures the device rather
/// than the shapes, so every failure re-checks a known-good canary and the
/// run stops the moment the canary goes.
///
/// The canary is the only reliable discriminator *among the values*, so do
/// not try to read the device's health off them. A `got -1515870800` means
/// the DPU never wrote those bytes, which is how the dense Cin>384 failures
/// present, but fully written *wrong* values occur both for genuine shape
/// failures (the Cout=31/33 partial-block cases in the known-limitations
/// probe report `max|diff|` of 45 on a perfectly healthy device) and for a
/// sick one.
///
/// The clock, however, is decisive, and it is why this loop times every
/// dispatch. A hung job leaves exactly the same sentinel bytes as a shape
/// that computes nothing, because `rocket_job.c`'s watchdog resets the core
/// and signals the fence with an error, and `PREP_BO` -- which waits on the
/// `dma_resv` fence -- returns success on a fence that is signaled at all.
/// The round trip is the tell: healthy dispatches in this file run under
/// 3.4 ms including the Cin=1792 shapes, and a hung one costs the scheduler's
/// full timeout, measured at 534 ms. See `DISPATCH_TIMEOUT_FLOOR`. The kernel
/// does log it -- `rocket ...npu: NPU job timed out`, with a follow-up line
/// giving `PC raw_status`, `CNA_S_STATUS` and `CORE_S_STATUS` -- but reading
/// dmesg needs root on `planck`, so the timer is what this sweep can act on.
/// In every sample decoded so far the CNA had signaled its feature and weight
/// loads for the live ping-pong group, CORE and DPU never signaled
/// completion, that executer sat in `operating`, and both `DMA_READ_ERROR`
/// and `DMA_WRITE_ERROR` were clear.
///
/// Only a reboot clears the sick state -- not idle, not a fresh process, not
/// a reopened fd. Reboot the board, then resume with
/// `ROCKET_PROBE_RESUME_AT`, which this prints for you.
///
/// **The canary is necessary but not sufficient.** Results here are also
/// order dependent, and at small shapes outright flaky, in ways a Cin=64
/// canary does not detect. In the original raw parity probe,
/// Cin=3 Cout=33 and Cin=5 Cout=64 both *pass* when they run first and fail
/// mid-sweep, and even in isolation Cin=3 Cout=33 failed once in six
/// otherwise identical runs. Contamination is per-process: a fresh process
/// clears it, intervening successful jobs do not. So a verdict on one shape
/// needs `ROCKET_PROBE_RESUME_AT` to run it *first*, repeated a few times --
/// never one row from one sweep.
///
/// **It is not DMA-address reuse**, which this comment used to claim on the
/// strength of the per-process part alone. Measured with `ROCKET_IOVA_LEDGER`
/// on 2026-09-02, using this test's own 34x34 Cin=8 Cout=16 K3 dense case
/// (index 19), which fails about 40% of the time here and never in isolation:
/// the IOVA allocator is first-fit from 0 and resets every case, so the
/// layout for that case is byte-identical -- INPUT=0x0+0x5000,
/// WEIGHTS=0x5000+0x1000, BIAS=0x6000+0x1000, OUTPUT=0x7000+0x8000,
/// REGCMD=0xf000+0x1000 -- in the runs where it passes, the runs where it
/// fails, and the isolated runs where it never fails. Same shape, same plan
/// (banks 1/11), same program, same addresses; only the preceding jobs
/// differ. No pair of simultaneously live buffers ever overlapped, and the
/// input, weight, bias and regcmd buffers all read back byte-intact after
/// every failing job, so the DPU is not writing into another live buffer.
///
/// What it actually looks like is a *quantized truncation*: the failing case
/// leaves exactly 3584 of 32768 output bytes at the sentinel, every time it
/// fails -- the last 112 of 1024 pixels, never written, with everything
/// before them correct. Not a drain race either: a 2 ms dwell and a second
/// `PREP_BO` never recovered a byte, so those writes are not late, they never
/// happen. That leaves device state carried across jobs within a process,
/// which is the same hazard class `rocket-hal-driver`'s
/// `DEPTHWISE_TO_DENSE_QUIESCENCE` already documents from the production
/// side. The open question is whether the DPU or the IOMMU reports anything
/// when it happens, and that one does need kernel-side visibility.
///
/// So the summary tells you to settle every failure per-process rather than
/// implying the sweep already did. On 2026-09-02 all three failures in the
/// full `--include-ignored` gate (`dense_geometry_regression`, two of the
/// MobileNetV2 pointwise cases, and `conv_features0_exact_hw`) left the
/// canary passing, and every one of them passed alone under
/// `ROCKET_PROBE_ONLY`.
///
/// **Replaying a witness here was tried and rejected**, on the theory that a
/// case this run already saw pass would stop passing once the process was
/// contaminated, catching what the fixed Cin=64 canary is too small to see.
/// It does not work, and it is actively harmful. Measured on `planck`, the
/// MobileNetV2 pointwise sweep without the replay fails the same two cases
/// three runs out of three; with a witness replayed after each failure it
/// failed three, four and three cases, and the extra failures were the
/// neighbours of the first one. The replay is itself dispatches, and
/// dispatches issued after a failure spread the corruption -- so the
/// instrument changed what it was measuring, and traded a deterministic
/// result for a drifting one. Anything that adds work to this loop on the
/// failure path has the same problem. Diagnose with a fresh process, which
/// is the one thing known to clear it, not with more work inside this one.
fn run_hardware_case_matrix(title: &str, cases: Vec<Conv2dCase>) {
    let _device_guard = NPU_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let total_cases = cases.len();
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let resume = probe_resume_index();
    let only = probe_only_index();
    let mut failures = Vec::new();
    let mut sick_after = None;
    let mut timeouts = 0usize;
    let mut attempted = 0usize;
    let mut skipped = 0usize;

    println!("\n=== {title} ===");
    match only {
        Some(index) => println!("  running index {index} of {total_cases} on its own"),
        None if resume != 0 => println!("  resuming at index {resume} of {total_cases}"),
        None => {}
    }

    if !accumulator_canary_passes(&file) {
        println!(
            "  CANARY FAILED before any measurement (dense K1 Cin=64, which passes on any\n  \
             healthy device). The NPU is sick -- reboot the board and re-run. Nothing this\n  \
             sweep could print below would be trustworthy."
        );
        panic!("{title} did not run: the NPU is sick, reboot the board");
    }

    for (index, case) in cases.into_iter().enumerate() {
        let wanted = match only {
            Some(only) => index == only,
            None => index >= resume,
        };
        if !wanted {
            skipped += 1;
            continue;
        }
        attempted += 1;
        ledger_begin_case(index);
        let label = case.label();
        let result = catch_unwind(AssertUnwindSafe(|| {
            let fixture = build_fixture(case)?;
            execute_case(&file, &fixture)
        }));
        let failure = match result {
            Ok(Ok(success)) => {
                println!(
                    "[{}/{}] ok   {label} banks={}/{} tiles={}",
                    index + 1,
                    total_cases,
                    success.data_banks,
                    success.weight_banks,
                    success.tiles,
                );
                if let Some(report) = ledger_case_report() {
                    println!("{report}");
                }
                continue;
            }
            Ok(Err(error)) => {
                println!(
                    "[{}/{}] FAIL {label}\n      {error}",
                    index + 1,
                    total_cases
                );
                if let Some(report) = ledger_case_report() {
                    println!("{report}");
                }
                format!("{label}: {error}")
            }
            Err(payload) => {
                let error = panic_message(payload);
                println!(
                    "[{}/{}] PANIC {label}\n      {error}",
                    index + 1,
                    total_cases
                );
                if let Some(report) = ledger_case_report() {
                    println!("{report}");
                }
                format!("{label}: panic: {error}")
            }
        };
        // A hung job and a wrong answer are indistinguishable in the output
        // buffer -- both leave sentinel bytes -- but not on the clock.
        let failure = match last_dispatch_if_timed_out() {
            Some(elapsed) => {
                timeouts += 1;
                println!(
                    "      DEVICE TIMEOUT: that dispatch took {:.0} ms against a {} ms healthy\n      \
                     ceiling, so the NPU hung and the driver reset it mid-job. The output is a\n      \
                     partial write, not this shape's answer. Expect a matching \"NPU job timed\n      \
                     out\" in dmesg at this moment.",
                    elapsed.as_secs_f64() * 1000.0,
                    DISPATCH_TIMEOUT_FLOOR.as_millis(),
                );
                format!("{failure} [DEVICE TIMEOUT, not a shape result]")
            }
            None => failure,
        };
        failures.push(failure);

        if !accumulator_canary_passes(&file) {
            sick_after = Some(index);
            break;
        }
    }

    let not_run = total_cases - attempted - skipped;

    println!("\n=== {title} summary ===");
    println!(
        "  passed: {} of {attempted} attempted",
        attempted - failures.len()
    );
    println!("  failed: {}", failures.len());
    if timeouts != 0 {
        println!("  of those, device timeouts (no shape verdict): {timeouts}");
    }
    if skipped != 0 {
        println!("  skipped: {skipped}");
    }
    if sick_after.is_some() {
        println!("  not run (device went sick): {not_run}");
    }
    for (index, failure) in failures.iter().enumerate() {
        println!("    {}. {failure}", index + 1);
    }

    if timeouts != 0 {
        println!(
            "\n  {timeouts} of these were hung jobs, not measurements. The NPU stalled with one\n  \
             CNA executer still `operating` and no CORE/DPU completion, the driver's watchdog\n  \
             reset it, and PREP_BO returned success anyway -- a fence signaled with an error is\n  \
             still signaled, which is why this sweep has always seen a silent half-written\n  \
             buffer. Confirm against the kernel's own account, which needs root:\n    \
             sudo dmesg | grep -A1 'NPU job timed out'\n  \
             Both DMA error bits were clear in every sample so far, so these are not\n  \
             out-of-range accesses. Do not record any of them as a shape limit."
        );
    }

    if !failures.is_empty() && sick_after.is_none() {
        println!(
            "\n  The canary still passes, so the device is not sick -- but that does not make\n  \
             these failures real. This process can corrupt its own later cases while the\n  \
             canary stays clean, and every failure investigated that way so far has passed\n  \
             when run on its own. Settle each one before recording it as a shape limit:\n    \
             ROCKET_PROBE_ONLY=<index-1> <this binary> <test name> --ignored --nocapture"
        );
    }

    if let Some(index) = sick_after {
        println!(
            "\n  CANARY FAILED after case {} of {total_cases}. The device is sick from here on,\n  \
             so that case is the last trustworthy row and the remaining {not_run} were not run.\n  \
             Reboot the board, then continue with:\n    \
             ROCKET_PROBE_RESUME_AT={} <this binary> <test name> --ignored --nocapture",
            index + 1,
            index + 1,
        );
        panic!(
            "{title} stopped after case {} of {total_cases}: the NPU went sick. Reboot the \
             board and resume with ROCKET_PROBE_RESUME_AT={}",
            index + 1,
            index + 1,
        );
    }

    assert!(
        failures.is_empty(),
        "{} of {attempted} cases failed in {title}; complete diagnostics are above",
        failures.len(),
    );
}
#[cfg(feature = "hardware-characterization")]
#[test]
#[ignore = "needs /dev/accel/accel0 -- confirms failure starts exactly at ConvPlan's bank-partition flip"]
fn bank_partition_flip_boundary_probe_runs_every_case_before_failing() {
    let cases = bank_partition_flip_boundary_cases();
    assert_eq!(cases.len(), 16);
    run_hardware_case_matrix("bank-partition flip boundary probe", cases);
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- guards dense diverse coefficients at VGG's Cin/Cout-512 blocks"]
fn dense_coefficient_vgg_blocks_match_oracle() {
    let cases = dense_coefficient_vgg_block_cases();
    assert_eq!(cases.len(), 5);
    run_hardware_case_matrix("dense-coefficient VGG block regression", cases);
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- validates exact i32 accumulator values, partial blocks, channel ceilings, and stride"]
fn int8_accumulator_regression_matrix_matches_oracle() {
    let cases = int8_accumulator_regression_cases();
    assert_eq!(cases.len(), 9);
    run_hardware_case_matrix("int8 accumulator regression matrix", cases);
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- validates every supported dense K1 accumulator Cin atom"]
fn int8_accumulator_k1_supported_cin_atom_sweep() {
    let cases = int8_accumulator_k1_supported_cin_atom_cases();
    assert_eq!(cases.len(), 24);
    run_hardware_case_matrix("int8 accumulator K1 supported Cin atom sweep", cases);
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- resolves the exact dense signed K1 Int8Accumulator Cin transition"]
fn int8_accumulator_k1_cin_boundary_matches_oracle() {
    let cases = int8_accumulator_k1_cin_boundary_cases();
    assert_eq!(cases.len(), 5);
    run_hardware_case_matrix("int8 accumulator K1 Cin boundary regression", cases);
}

/// Resolves the Cin/CBUF split matrix, guarding every row with a canary.
///
/// The RK3588 NPU can drop into a state where *every* job returns wrong
/// results, including shapes that passed seconds earlier. Observed on
/// `planck` 2026-08-31 during a wide accumulator sweep: from one case
/// onwards nothing was correct again, `int8_accumulator_k1_cin_boundary_matches_oracle`
/// went 5/5 pass to 5/5 fail, and only rebooting the board restored it.
/// Nothing reports an error when this happens -- `prep_bo` still succeeds
/// well inside its timeout -- so a long sweep silently stops measuring
/// shapes and starts measuring the sick device.
///
/// The canary is the only reliable discriminator; the values themselves are
/// not. Every failure in this matrix presents as `got -1515870800`, the
/// sentinel the DPU never overwrote, but elsewhere a healthy device produces
/// fully written wrong values for genuine shape failures too, so that alone
/// proves nothing about the device.
///
/// A single bad shape does not cause it: 15 consecutive Cin=385 failures each
/// left the canary clean, and the whole Cin=384 block still passes 5/5
/// immediately after a Cin=385 failure. It takes roughly fifteen to twenty
/// failing jobs in one process -- one full pass over this matrix is enough,
/// and did it twice. So passes chain freely and failures are worth
/// continuing past, but every failure re-checks the canary and the probe
/// stops the moment the device itself is implicated.
/// `ROCKET_PROBE_RESUME_AT` then picks the matrix back up, by index, once
/// the board has been rebooted.
///
/// The canary between cases also makes the rows *deterministic*: without it
/// the failing rows had ragged mismatch counts and a first-bad channel of
/// 32; with it every failing row is a clean 65536/65536 from c=0.
#[cfg(feature = "hardware-characterization")]
#[test]
#[ignore = "needs /dev/accel/accel0 -- isolates the accumulator Cin>384 CBUF split boundary"]
fn int8_accumulator_k1_cin_cbuf_split_probe() {
    const CIN: [u32; 4] = [384, 385, 400, 512];
    const SPLITS: u32 = 5;

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");

    let resume = probe_resume_index();

    println!("\n=== int8 accumulator K1 Cin/CBUF split probe ===");
    println!("  matrix: Cin 384/385/400/512 x weight banks 1..=5, ascending");
    println!(
        "  every failure re-checks a Cin=64 canary; the probe stops only if the device is sick"
    );
    if resume != 0 {
        println!("  resuming at index {resume}");
    }

    if !accumulator_canary_passes(&file) {
        println!(
            "\n  CANARY FAILED before any measurement (dense K1 Cin=64, which passes on any\n  \
             healthy device). The NPU is sick -- reboot the board and re-run. Nothing this\n  \
             probe could print below would be trustworthy."
        );
        return;
    }
    println!("  canary ok (dense K1 Cin=64): the device is healthy");

    let mut failures = 0usize;
    for (cin_index, cin) in CIN.into_iter().enumerate() {
        let base = cin_index * SPLITS as usize;
        if base + SPLITS as usize <= resume {
            continue;
        }

        let case = Conv2dCase {
            width: 32,
            height: 32,
            cin,
            cout: 64,
            kernel: [1, 1],
            stride: 1,
            padding: [0, 0],
            precision: OraclePrecision::Int8Accumulator,
            pattern: OraclePattern::Dense { phase: 1 },
        };
        let fixture = build_raw_fixture(case).expect("build focused accumulator fixture");
        println!("\n  Cin={cin}, coefficient_bytes={}", fixture.weights.len());

        for (split_index, weight_banks) in (1..=SPLITS).enumerate() {
            let index = base + split_index;
            let data_banks = 12 - weight_banks;
            if index < resume {
                println!("    [{index:2}] skip banks={data_banks}/{weight_banks} (before resume)");
                continue;
            }

            let result = catch_unwind(AssertUnwindSafe(|| {
                let plan = ConvPlan::with_cbuf_banks(
                    fixture.shape,
                    fixture.case.kernel,
                    data_banks,
                    weight_banks,
                );
                let execution = execute_case_output_with_plan(&file, &fixture, plan)?;
                let report = compare_output(&fixture, &execution.plan, &execution.output);
                Ok::<_, String>((execution.plan.tiles().len(), report))
            }));

            let failure = match result {
                Ok(Ok((tiles, report))) if report.mismatches == 0 => {
                    println!(
                        "    [{index:2}] PASS banks={data_banks}/{weight_banks} tiles={tiles}"
                    );
                    continue;
                }
                Ok(Ok((tiles, report))) => format!(
                    "FAIL banks={data_banks}/{weight_banks} tiles={tiles} mismatches={} \
                     tile_mismatches={:?} first={}",
                    report.mismatches,
                    report.tile_mismatches,
                    report.samples.first().map(String::as_str).unwrap_or("none"),
                ),
                Ok(Err(error)) => format!("ERROR banks={data_banks}/{weight_banks}: {error}"),
                Err(payload) => format!(
                    "PANIC banks={data_banks}/{weight_banks}: {}",
                    panic_message(payload)
                ),
            };

            failures += 1;
            println!("    [{index:2}] {failure}");

            if accumulator_canary_passes(&file) {
                println!("         canary still ok: the device is healthy, this row is real");
                continue;
            }

            println!(
                "\n  CANARY FAILED after index {index} (Cin={cin}, banks={data_banks}/{weight_banks}).\n  \
                 The device is now sick, so this row is the last trustworthy one and every\n  \
                 later point would measure the device rather than its own shape.\n  \
                 Reboot the board, then continue with:\n    \
                 ROCKET_PROBE_RESUME_AT={} <this binary> int8_accumulator_k1_cin_cbuf_split_probe \
                 --ignored --nocapture",
                index + 1,
            );
            return;
        }
    }

    println!("\n=== int8 accumulator K1 Cin/CBUF split probe summary ===");
    println!(
        "  covered the whole matrix: {} of {} points failed, device healthy throughout",
        failures,
        CIN.len() * SPLITS as usize,
    );
}

#[test]
fn int8_accumulator_cout_sweep_is_planable_and_gap_free() {
    let cases = int8_accumulator_cout_sweep_cases();
    assert_eq!(cases.len(), 42);
    assert_planable_and_gap_free(cases);
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- validates HAL Cout padding across every output block boundary"]
fn int8_accumulator_cout_padding_sweep_matches_oracle() {
    run_hardware_case_matrix(
        "int8 accumulator HAL Cout-padding sweep",
        int8_accumulator_cout_sweep_cases(),
    );
}

#[test]
fn int8_accumulator_cout_shape_interaction_is_planable_and_gap_free() {
    let cases = int8_accumulator_cout_shape_interaction_cases();
    assert_eq!(cases.len(), 24);
    assert_planable_and_gap_free(cases);
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- validates HAL Cout padding across even and odd shapes"]
fn int8_accumulator_cout_shape_interaction_matches_oracle() {
    run_hardware_case_matrix(
        "int8 accumulator HAL Cout/shape interaction",
        int8_accumulator_cout_shape_interaction_cases(),
    );
}

#[test]
fn int8_accumulator_output_parity_is_planable_and_gap_free() {
    let cases = int8_accumulator_output_parity_cases();
    assert_eq!(cases.len(), 16);
    assert_planable_and_gap_free(cases);
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- validates HAL padding on every measured parity combination"]
fn int8_accumulator_output_parity_padding_matches_oracle() {
    run_hardware_case_matrix(
        "int8 accumulator HAL output-parity padding",
        int8_accumulator_output_parity_cases(),
    );
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- guards the HAL Cout-parity workaround at former failure points"]
fn int8_accumulator_parity_regression_matrix_matches_oracle() {
    let cases = int8_accumulator_parity_regression_cases();
    assert_eq!(cases.len(), 6);
    run_hardware_case_matrix("int8 accumulator parity regression matrix", cases);
}

#[test]
fn dense_geometry_regression_is_planable_and_gap_free() {
    let cases = dense_geometry_regression_cases();
    assert_eq!(cases.len(), 22);
    assert_planable_and_gap_free(cases);
}

#[test]
fn cbuf_residency_boundary_is_planable_and_gap_free() {
    let cases = cbuf_residency_boundary_cases();
    assert_eq!(cases.len(), 13);
    assert_planable_and_gap_free(cases);
}

#[test]
fn fp16_mobilenetv2_remaining_pointwise_is_planable_and_gap_free() {
    unsafe { std::env::set_var("ROCKET_ALLOW_UNBACKED_CHANNELS", "1") };
    let cases = fp16_mobilenetv2_remaining_pointwise_cases();
    assert_eq!(cases.len(), 27);
    assert_planable_and_gap_free(cases);
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- CBUF entry-rounding residency boundary"]
fn cbuf_residency_boundary_matches_oracle() {
    run_hardware_case_matrix(
        "CBUF entry-rounding residency boundary",
        cbuf_residency_boundary_cases(),
    );
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- dense stride/window-parity and sub-atom Cin geometries"]
fn dense_geometry_regression_matches_oracle() {
    run_hardware_case_matrix(
        "dense fp16 stride/window-parity and sub-atom Cin geometries",
        dense_geometry_regression_cases(),
    );
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- characterizes the 9 CPU-routed fp16 MobileNetV2 pointwise shapes"]
fn fp16_mobilenetv2_remaining_pointwise_matches_oracle() {
    unsafe { std::env::set_var("ROCKET_ALLOW_UNBACKED_CHANNELS", "1") };
    run_hardware_case_matrix(
        "MobileNetV2 fp16 CPU-routed pointwise shapes",
        fp16_mobilenetv2_remaining_pointwise_cases(),
    );
}

#[cfg(feature = "hardware-characterization")]
#[test]
#[ignore = "needs /dev/accel/accel0 -- characterizes 0x80 neutral int8 coefficient bytes"]
fn int8_neutral80_one_hot_four_way_confirmation_matches_oracle() {
    println!("  packed background/padding byte: 0x80");
    println!("  packed live logical +1 byte:    0x00");
    run_hardware_case_matrix(
        "int8 neutral80 one-hot four-way confirmation",
        int8_neutral80_one_hot_four_way_cases(),
    );
}

#[cfg(feature = "hardware-characterization")]
fn run_raw_coefficient_byte_sweep(unit_gain: bool, title: &str, conversion: &str) -> Vec<i8> {
    let _device_guard = NPU_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let case = int8_raw_coefficient_byte_sweep_case(unit_gain);
    let fixture = build_fixture(case).expect("build 256-byte coefficient sweep");
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let execution = execute_case_output(&file, &fixture).expect("execute coefficient-byte sweep");
    let observed = (0usize..256)
        .map(|channel| {
            let offset = output_offset(fixture.shape, case.kernel, channel, 0, 0);
            execution.output[offset] as i8
        })
        .collect::<Vec<_>>();

    println!("\n=== {title} ===");
    println!("  input: 1, K1/Cin1, output channel c gets raw coefficient byte c");
    println!("  padding: 0x80, output conversion: {conversion}");
    println!(
        "  banks={}/{} tiles={}",
        execution.plan.data_banks(),
        execution.plan.weight_banks(),
        execution.plan.tiles().len(),
    );
    for base in (0usize..256).step_by(16) {
        let values = observed[base..base + 16]
            .iter()
            .map(|value| format!("{value:>4}"))
            .collect::<Vec<_>>()
            .join(" ");
        println!("  raw 0x{base:02x}..0x{:02x}: {values}", base + 15);
    }
    observed
}

#[cfg(feature = "hardware-characterization")]
#[test]
#[ignore = "needs /dev/accel/accel0 -- maps every dense int8 coefficient byte"]
fn int8_dense_coefficient_raw_byte_sweep() {
    let observed = run_raw_coefficient_byte_sweep(
        false,
        "dense int8 coefficient raw-byte sweep",
        "gain 128, zero point -128",
    );
    println!("  hypothesis: output(raw) = raw interpreted as signed i8");

    let differences = observed
        .iter()
        .enumerate()
        .map(|(raw, &got)| i32::from(got) - i32::from(raw as u8 as i8))
        .collect::<Vec<_>>();
    let exact = differences
        .iter()
        .filter(|&&difference| difference == 0)
        .count();
    let within_one = differences
        .iter()
        .filter(|&&difference| difference.abs() <= 1)
        .count();
    let max_difference = differences
        .iter()
        .map(|difference| difference.abs())
        .max()
        .unwrap_or(0);
    println!("  exact: {exact}/256");
    println!("  within one LSB: {within_one}/256");
    println!("  max |difference|: {max_difference}");
    for (raw, (&got, &difference)) in observed.iter().zip(&differences).enumerate() {
        if difference.abs() > 1 {
            println!(
                "    raw=0x{raw:02x} signed={} expected={} got={got} difference={difference}",
                raw as u8 as i8, raw as u8 as i8,
            );
        }
    }

    assert_eq!(
        within_one, 256,
        "coefficient-byte mapping differed from signed-byte hypothesis by more than one LSB; \
         complete map is printed above",
    );
}

#[cfg(feature = "hardware-characterization")]
#[test]
#[ignore = "needs /dev/accel/accel0 -- maps dense int8 coefficient bytes at unit gain"]
fn int8_dense_coefficient_raw_byte_unit_gain_sweep() {
    let observed = run_raw_coefficient_byte_sweep(
        true,
        "dense int8 coefficient raw-byte unit-gain sweep",
        "ordinary unit gain, zero point 0",
    );
    let distinct = observed
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    println!("  distinct outputs ({}): {distinct:?}", distinct.len());
    for raw in [0x00usize, 0x01, 0x7f, 0x80, 0x81, 0xff] {
        println!(
            "  landmark raw=0x{raw:02x} signed={:>4} -> output={:>4}",
            raw as u8 as i8, observed[raw],
        );
    }
    assert_eq!(
        observed[0x80], 0,
        "raw 0x80 stopped being neutral at ordinary unit gain; complete map is above",
    );
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn cartesian_conv2d_oracle_sweep_matches_oracle() {
    let _device_guard = NPU_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let cases = cartesian_cases();
    let total_cases = cases.len();
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let mut failures = Vec::new();
    let mut passed_by_kind = BTreeMap::<(&str, &str), usize>::new();

    println!(
        "\n=== Conv2D oracle Cartesian sweep: {} cases ===",
        cases.len()
    );
    for (index, case) in cases.into_iter().enumerate() {
        let label = case.label();
        let result = catch_unwind(AssertUnwindSafe(|| {
            let fixture = build_fixture(case)?;
            execute_case(&file, &fixture)
        }));
        match result {
            Ok(Ok(success)) => {
                println!(
                    "[{}/{}] ok   {label} banks={}/{} tiles={}",
                    index + 1,
                    total_cases,
                    success.data_banks,
                    success.weight_banks,
                    success.tiles,
                );
                *passed_by_kind
                    .entry((case.precision.name(), case.pattern.name()))
                    .or_default() += 1;
            }
            Ok(Err(error)) => {
                println!(
                    "[{}/{}] FAIL {label}\n      {error}",
                    index + 1,
                    total_cases,
                );
                failures.push(format!("{label}: {error}"));
            }
            Err(payload) => {
                let error = panic_message(payload);
                println!(
                    "[{}/{}] PANIC {label}\n      {error}",
                    index + 1,
                    total_cases,
                );
                failures.push(format!("{label}: panic: {error}"));
            }
        }
    }

    println!("\n=== Conv2D oracle summary ===");
    for ((precision, pattern), passed) in passed_by_kind {
        println!("  {precision:>4} {pattern:<9}: {passed} passed");
    }
    println!("  failures: {}", failures.len());
    for (index, failure) in failures.iter().enumerate() {
        println!("    {}. {failure}", index + 1);
    }

    assert!(
        failures.is_empty(),
        "{} of {} Conv2D oracle cases failed; complete diagnostics are printed above",
        failures.len(),
        total_cases,
    );
}

/// Where the DPU physically writes accumulator output, measured rather than
/// assumed.
///
/// `Shape::output_atom_bytes` keeps the 128-byte block for the dense
/// accumulator path on the strength of one measurement at Cin=Cout=64. That
/// shape is two blocks per pixel -- an even count -- which is exactly the
/// case a 128-byte-block model and a 256-byte-atom model cannot be told
/// apart by, since both put channel `c` at the same address. Every shape
/// that fails the even-blocks rule has an *odd* count, where the two models
/// diverge, and none of them had ever been measured.
///
/// Method: drive one input element to a nonzero value and zero the rest.
/// With a 1x1 kernel and the `Dense` pattern -- whose every coefficient is
/// nonzero -- exactly one output pixel is nonzero, and its whole channel
/// vector is. The offsets of the nonzero i32 lanes in the untouched staging
/// buffer are then the layout, with nothing inferred.
///
/// **Result, measured 2026-08-31.** Where the parity rule holds the
/// 128-byte block model matches every lane at every hot pixel, and exactly
/// `Cout` lanes come back. Where it is violated, two things change at once:
///
///   * exactly one extra 128-byte block's worth of lanes (32) comes back
///     non-zero -- 64 for Cout=32, 128 for Cout=96 -- and those are the
///     `OUTPUT_SENTINEL` still sitting in the **trailing block the DPU never
///     wrote**. The excess is one block in every odd case and zero in every
///     even one, so the DPU commits whole 256-byte units and drops the odd
///     trailing 128-byte block;
///   * the pixel stride becomes 256 bytes rather than 128. At 3x3 Cout=32
///     with the hot pixel at (0,1), the 128-byte model matches 0/32 lanes
///     while the 256-byte model matches 32/32.
///
/// So the DPU commits accumulator output in **256-byte units**, and the
/// 128-byte block model is not wrong so much as *coincidental*: it agrees
/// exactly when the total block count is even, which is the parity rule.
/// `int8_accumulator_output_address_map_probe` sweeps the hot pixel over
/// every position and settles what the addressing does. The **passing**
/// shapes come back as exact bijections onto the shipped model -- 4x4
/// Cout=32 maps pixel `p` to block `p` for all sixteen, and 3x3 Cout=64 maps
/// `p` to blocks `p` and `p + 9`, surface-major, for all nine -- which is
/// what validates the method. The **failing** shape does not:
///
/// ```text
/// 3x3 Cout=32, pixel -> block written (correct would be p -> p)
///   0 -> 0     1 -> 0 and 2   2 -> 3    3 -> 3
///   4 -> 4     5 -> 6         6 -> 7    7 -> 7    8 -> nothing
/// ```
///
/// Blocks 0, 2, 3, 4, 6, 7 receive data; 1 and 5 receive only zeros; block
/// 8, the odd trailing one, is never written at all. The map is
/// *non-injective*: pixels 2 and 3 alias onto one block, so do 6 and 7,
/// pixel 1 lands twice and pixel 8 vanishes. That is a corrupted address
/// computation, not a different stride constant.
///
/// Note the counting caveat that produced an earlier misreading here: the
/// staging buffer is poisoned with `OUTPUT_SENTINEL`, which is non-zero, so
/// "non-zero lane" means *written data or untouched sentinel*. Separate the
/// two before drawing conclusions from lane counts.
///
/// **Why the granule change did not help, settled by diffing the programs**
/// (`--example dump_conv_plan_regcmd`, `ROCKET_DUMP_PRECISION=int8acc`).
/// `DPU 0x403c` packs two channel fields -- low is the true `out_channels`
/// minus one, high is the padded count minus one:
///
/// ```text
/// 3x3 Cout=32 stock            0x001f001f   fails
/// 3x3 Cout=32 + granule 64     0x001f003f   still fails
/// 3x3 Cout=64 real             0x003f003f   passes
/// ```
///
/// The granule moved only the *padded* half. The DPU still writes
/// `true_out_channels` worth of data -- one block per pixel for Cout=32 --
/// so the block count it actually commits never changed and the parity never
/// changed with it. Padding the allocation was never going to help.
///
/// **And there is no misprogrammed register.** The minimal pair, 3x3 Cout=32
/// against 3x3 Cout=64 -- same spatial shape, parity differing only through
/// Cout -- differs in exactly six registers, and every one of them is a
/// channel count or a weight size. Not one addressing, stride or geometry
/// register differs between a shape that works and a shape that does not.
/// `DPU_SURFACE_ADD` is the constant 256 in both, while the measured surface
/// stride at the passing shape is `tile_pixels * 128` = 1152, so it is not
/// the stride either.
///
/// So this is a **hardware constraint, not a driver bug**: the DPU requires
/// the tile's committed block count to be even, and nothing in the program
/// tells it otherwise. The lever a fix has is the parity itself -- the
/// number of blocks actually written (true `Cout`) or `tile_pixels` -- not
/// any padding or register correction.
#[cfg(feature = "hardware-characterization")]
fn accumulator_layout_probe(
    file: &std::fs::File,
    extent: (u32, u32),
    cout: u32,
    hot: (usize, usize, usize),
) -> Result<Vec<u8>, String> {
    let (hot_y, hot_x, hot_c) = hot;
    let case = Conv2dCase {
        width: extent.0,
        height: extent.1,
        cin: 8,
        cout,
        kernel: [1, 1],
        stride: 1,
        padding: [0, 0],
        precision: OraclePrecision::Int8Accumulator,
        pattern: OraclePattern::Dense { phase: 0 },
    };
    let mut fixture = build_raw_fixture(case)?;

    // One-hot the *input*, keeping the fixture's own coefficients: output
    // then vanishes everywhere except the single pixel (hot_y, hot_x).
    fixture.input.iter_mut().for_each(|byte| *byte = 0);
    fixture.input[feature_offset(fixture.shape, hot_c, hot_y, hot_x)] = 1;

    Ok(execute_case_output(file, &fixture)?.raw)
}

/// One i32 lane of the staging buffer, classified.
///
/// The buffer is pre-poisoned with `OUTPUT_SENTINEL`, which is *non-zero*,
/// so "non-zero" alone cannot separate written data from bytes the DPU never
/// touched. Conflating the two is what made an earlier reading of this probe
/// report values written twice, when the excess was an untouched trailing
/// block.
#[cfg(feature = "hardware-characterization")]
#[derive(PartialEq)]
enum Lane {
    Sentinel,
    Zero,
    Data,
}

#[cfg(feature = "hardware-characterization")]
fn classify_lanes(raw: &[u8]) -> Vec<Lane> {
    let sentinel = i32::from_le_bytes([OUTPUT_SENTINEL; 4]);
    raw.chunks_exact(4)
        .map(
            |chunk| match i32::from_le_bytes(chunk.try_into().unwrap()) {
                value if value == sentinel => Lane::Sentinel,
                0 => Lane::Zero,
                _ => Lane::Data,
            },
        )
        .collect()
}

#[cfg(feature = "hardware-characterization")]
#[test]
#[ignore = "needs /dev/accel/accel0 -- maps where the DPU physically addresses accumulator output"]
fn int8_accumulator_output_address_map_probe() {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");

    println!("\n=== accumulator output address map ===");

    // Passing shapes first, deliberately: only failures walk the device
    // toward the sick state, so the clean control maps are collected while
    // the device is certainly healthy.
    for (extent, cout) in [
        ((4u32, 4u32), 32u32),
        ((3, 3), 64),
        ((3, 3), 32),
        ((3, 3), 96),
    ] {
        if !accumulator_canary_passes(&file) {
            println!("\n  CANARY FAILED -- the NPU is sick, reboot the board");
            return;
        }

        let control = Conv2dCase {
            width: extent.0,
            height: extent.1,
            cin: 8,
            cout,
            kernel: [1, 1],
            stride: 1,
            padding: [0, 0],
            precision: OraclePrecision::Int8Accumulator,
            pattern: OraclePattern::Dense { phase: 0 },
        };
        let shape = control.shape();
        let width = shape.output_width([1, 1]) as usize;
        let height = shape.output_height([1, 1]) as usize;
        let blocks = (shape.padded_out_channels() as usize * 4) / 128;
        let verdict = match build_raw_fixture(control).and_then(|fixture| {
            let execution = execute_case_output(&file, &fixture)?;
            Ok(compare_output(&fixture, &execution.plan, &execution.output).mismatches)
        }) {
            Ok(0) => "PASSES".to_string(),
            Ok(mismatches) => format!("FAILS ({mismatches} mismatches)"),
            Err(error) => format!("ERROR {error}"),
        };
        println!(
            "\n  {width}x{height} Cout={cout}: pixels={} blocks/pixel={blocks} product={} -> control {verdict}",
            width * height,
            width * height * blocks,
        );
        // Surface-major: a pixel's block advances by 128 within a surface,
        // and each surface is tile_pixels blocks further on. So a
        // multi-block shape's lanes are legitimately split across surfaces,
        // and "SPLIT" there is correct rather than a symptom.
        println!(
            "    shipped model predicts block index {} for pixel p, surface stride {}",
            "p",
            width * height * 128,
        );

        let mut sentinel_blocks = Vec::new();
        let mut total_blocks = 0;
        for y in 0..height {
            for x in 0..width {
                let raw = match accumulator_layout_probe(&file, extent, cout, (y, x, 0)) {
                    Ok(raw) => raw,
                    Err(error) => {
                        println!("    hot=({y},{x}): ERROR {error}");
                        continue;
                    }
                };
                let lanes = classify_lanes(&raw);
                let data: Vec<usize> = lanes
                    .iter()
                    .enumerate()
                    .filter(|(_, lane)| **lane == Lane::Data)
                    .map(|(index, _)| index * 4)
                    .collect();

                if total_blocks == 0 {
                    total_blocks = raw.len() / 128;
                    sentinel_blocks = (0..total_blocks)
                        .filter(|block| {
                            (0..32).all(|lane| lanes[block * 32 + lane] == Lane::Sentinel)
                        })
                        .collect();
                }

                match (data.first(), data.last()) {
                    (Some(first), Some(last)) => println!(
                        "    hot=({y},{x}): {:3} data lanes {first:5}..={last:5}  block {:<3} {}",
                        data.len(),
                        first / 128,
                        if data.len() == (last - first) / 4 + 1 {
                            "contiguous"
                        } else {
                            "SPLIT"
                        },
                    ),
                    _ => println!("    hot=({y},{x}): no data written anywhere"),
                }
            }
        }
        println!(
            "    blocks the DPU never wrote: {sentinel_blocks:?} of {total_blocks} \
             ({} bytes allocated)",
            total_blocks * 128,
        );
    }
}

/// Is the parity rule per *tile* or per dispatch?
///
/// Every case here has an **even** total output pixel count, so a
/// whole-dispatch reading of the rule predicts all of them pass. They differ
/// only in the CBUF bank split, which changes nothing about the convolution
/// -- same extent, same Cin, same Cout, same coefficients -- and only moves
/// where `plan_grid` puts its row boundaries. If the rule is per tile, the
/// splits whose individual tiles are odd must fail while the all-even splits
/// pass.
///
/// **Measured 9/9 on hardware: the rule is per tile.** Every case has an even
/// total pixel count, so a per-dispatch reading predicts all nine pass. The
/// all-even splits pass and the odd ones fail, with only the bank split
/// changing between them.
///
/// The width A/B settles the fix, too. `33x8` at banks 3/9 tiles as
/// `[7, 1]` and **fails**; `34x8` at the same banks tiles as the *identical*
/// `[7, 1]` and **passes**, because an even width makes both tiles even.
/// `64x8` split into eight single-row tiles -- every row count odd -- also
/// passes. So padding the width to even is robust however the planner tiles,
/// while an even output *height* is not: it survives only while the shape
/// stays one tile, and greedy row splitting readily produces odd per-tile
/// row counts.
///
/// The row splits below come from `ConvPlan::with_cbuf_banks` and are
/// asserted in `int8_accumulator_multitile_parity_is_planable`, so a planner
/// change that invalidates the experiment fails loudly rather than silently
/// measuring something else.
///
/// The last three carry the other half of the argument: an **even width**
/// should make every tile even no matter how the rows split. 34x8 at banks
/// 3/9 tiles as `[7, 1]`, the same odd row split that fails at 33x8, and
/// 64x8 at 1/11 splits into eight single-row tiles -- every row count odd,
/// every tile still even.
///
/// Cin stays at or below 384 throughout: dense int8 1x1 fails above that for
/// an unrelated reason that would confound this.
#[cfg(feature = "hardware-characterization")]
fn int8_accumulator_multitile_parity_cases() -> Vec<(Conv2dCase, u32, u32, &'static [u32])> {
    [
        // (width, height, cin, data_banks, weight_banks, expected out_rows)
        (33u32, 8u32, 384u32, 1u32, 11u32, &[2u32, 2, 2, 2] as &[u32]),
        (33, 8, 384, 3, 9, &[7, 1]),
        (63, 8, 384, 5, 7, &[6, 2]),
        (63, 8, 384, 4, 8, &[5, 3]),
        (15, 32, 256, 1, 11, &[8, 8, 8, 8]),
        (15, 32, 256, 3, 9, &[25, 7]),
        // The width A/B. 34x8 at banks 3/9 tiles as [7, 1] -- the *identical*
        // odd row split that fails at 33x8 -- but an even width makes both
        // tiles even anyway. 64x8 at 1/11 is the extreme: eight tiles of one
        // row each, every row count odd, every tile still even.
        (34, 8, 384, 3, 9, &[7, 1]),
        (34, 8, 384, 2, 10, &[5, 3]),
        (64, 8, 384, 1, 11, &[1, 1, 1, 1, 1, 1, 1, 1]),
    ]
    .into_iter()
    .map(|(width, height, cin, data_banks, weight_banks, rows)| {
        (
            Conv2dCase {
                width,
                height,
                cin,
                cout: 32,
                kernel: [1, 1],
                stride: 1,
                padding: [0, 0],
                precision: OraclePrecision::Int8Accumulator,
                pattern: OraclePattern::Dense { phase: 0 },
            },
            data_banks,
            weight_banks,
            rows,
        )
    })
    .collect()
}

#[cfg(feature = "hardware-characterization")]
#[test]
fn int8_accumulator_multitile_parity_is_planable() {
    for (case, data_banks, weight_banks, expected_rows) in int8_accumulator_multitile_parity_cases()
    {
        let shape = case.shape();
        let plan = ConvPlan::with_cbuf_banks(shape, case.kernel, data_banks, weight_banks);
        let rows: Vec<u32> = plan.tiles().iter().map(|tile| tile.rows.out_rows).collect();
        assert_eq!(
            rows, expected_rows,
            "{}x{} Cin={} banks={data_banks}/{weight_banks} no longer tiles as the experiment assumes",
            case.width, case.height, case.cin,
        );
        assert_eq!(
            (case.width as usize * case.height as usize) % 2,
            0,
            "every case must have an even total pixel count, or it cannot discriminate"
        );
    }
}

#[cfg(feature = "hardware-characterization")]
#[test]
#[ignore = "needs /dev/accel/accel0 -- characterizes raw parity across forced CBUF splits"]
fn int8_accumulator_multitile_parity_probe() {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");

    let only = probe_only_index();
    println!("\n=== accumulator programmed Cout across tile splits ===");
    if !accumulator_canary_passes(&file) {
        println!("  CANARY FAILED -- the NPU is sick, reboot the board");
        return;
    }

    for (index, (case, data_banks, weight_banks, expected_rows)) in
        int8_accumulator_multitile_parity_cases()
            .into_iter()
            .enumerate()
    {
        if only.is_some_and(|only| only != index) {
            continue;
        }
        let fixture = match build_raw_fixture(case) {
            Ok(fixture) => fixture,
            Err(error) => {
                println!("  [{index}] build ERROR {error}");
                continue;
            }
        };
        let plan =
            ConvPlan::with_cbuf_banks(fixture.shape, fixture.case.kernel, data_banks, weight_banks);
        let per_tile: Vec<usize> = expected_rows
            .iter()
            .map(|rows| *rows as usize * case.width as usize)
            .collect();
        let all_even = per_tile.iter().all(|pixels| pixels % 2 == 0);
        let verdict = match execute_case_output_with_plan(&file, &fixture, plan) {
            Ok(execution) => {
                let report = compare_output(&fixture, &execution.plan, &execution.output);
                if report.mismatches == 0 {
                    "pass".to_string()
                } else {
                    format!("FAIL ({} mismatches)", report.mismatches)
                }
            }
            Err(error) => format!("ERROR {error}"),
        };

        println!(
            "  [{index}] {}x{} Cin={} banks={data_banks}/{weight_banks} rows={expected_rows:?} \
             tile_pixels={per_tile:?} -> per-tile predicts {}, actual {verdict}",
            case.width,
            case.height,
            case.cin,
            if all_even { "pass" } else { "FAIL" },
        );
    }
}

/// Does padding `Cout` up to an even block count rescue a failing shape,
/// with the padding channels carrying zero coefficients?
///
/// This is the proposed fix for the output-parity rule. The parity is
/// `tile_pixels * blocks`, and `blocks = ceil(Cout/32)` is *static*, so
/// rounding `Cout` up to a multiple of 64 makes the product even however the
/// planner tiles -- no planner constraint, no matcher rejection, and nothing
/// added to the per-dispatch activation path, which is debt slated for
/// removal rather than something to build on.
///
/// Every case has an odd output pixel count and fails at its true `Cout`.
/// The padded form runs the same shape at the next multiple of 64 with the
/// surplus output channels zeroed, which is what padded weights produce.
/// Coefficients pack output-block-major in 32-channel blocks
/// (`WEIGHT_ATOMIC_BYTES`) with every block the same size, so zeroing from
/// `true_cout / 32` blocks onward zeroes precisely the surplus channels --
/// and that geometry is kernel- and stride-independent, which is why 3x3 and
/// stride 2 are here rather than assumed.
///
/// Only coefficients need zeroing: `Int8Accumulator` bypasses BS (asserted
/// by `int8_accumulator_output_uses_the_hardware_validated_bypasses`), so no
/// bias or per-channel multiplier reaches the padding channels.
///
/// Real channels are checked against the oracle for the *padded* case -- a
/// convolution's output channels are independent, so zeroing the surplus
/// must not move them -- and the surplus must come back exactly zero.
///
/// **Result, measured 2026-08-31, four isolated repeats per case.** Eight of
/// the nine cases give `real_bad=0 pad_nonzero=0` every time: 1x1 and 3x3,
/// stride 1 and 2, Cout 32->64 and 96->128. The fix generalizes.
///
/// The exception is index 4, 3x3 extent with a 3x3 kernel, where the
/// *unzeroed* control also fails 576 every time. The padded configuration is
/// broken there on its own, independent of any zeroing, so the case cannot
/// evaluate the fix -- and a compiler applying this pad would emit a broken
/// configuration for that shape. Its cause is not known; nothing else here
/// is that small relative to its kernel.
///
/// Note the trap this probe fell into first. A run over the whole list
/// reported cases 5-8 as failures, and so did a first pass with one case per
/// process, because index 4 fails for real and its failures contaminate
/// *subsequent processes*. Only re-running 5-8 on a rested device showed
/// them clean. Any sweep containing a genuinely failing case invalidates
/// everything measured after it, process isolation included.
#[cfg(feature = "hardware-characterization")]
#[test]
#[ignore = "needs /dev/accel/accel0 -- validates padding Cout to an even block count"]
fn int8_accumulator_cout_padding_probe() {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");

    println!("\n=== padding Cout to an even block count ===");
    if !accumulator_canary_passes(&file) {
        println!("  CANARY FAILED -- the NPU is sick, reboot the board");
        return;
    }

    let make = |width: u32, height: u32, kernel: usize, stride: u32, cout: u32| Conv2dCase {
        width,
        height,
        cin: 8,
        cout,
        kernel: [kernel, kernel],
        stride,
        padding: [kernel / 2, kernel / 2],
        precision: OraclePrecision::Int8Accumulator,
        pattern: OraclePattern::Dense { phase: 0 },
    };

    // Three device jobs per case, two of them deliberately failing, so a
    // whole-list run walks the device into the degraded state and the later
    // rows measure that rather than the shape. Run one case per process:
    //   for i in $(seq 0 8); do ROCKET_PROBE_ONLY=$i <bin> ... ; done
    let only = probe_only_index();

    // (width, height, kernel, stride, true_cout, padded_cout)
    for (index, (width, height, kernel, stride, true_cout, padded_cout)) in [
        (3u32, 3u32, 1usize, 1u32, 32u32, 64u32),
        (5, 5, 1, 1, 32, 64),
        (9, 7, 1, 1, 32, 64),
        (5, 5, 1, 1, 96, 128),
        (3, 3, 3, 1, 32, 64),
        (5, 5, 3, 1, 32, 64),
        (9, 7, 3, 1, 32, 64),
        (5, 5, 3, 1, 96, 128),
        (9, 9, 1, 2, 32, 64),
    ]
    .into_iter()
    .enumerate()
    {
        if only.is_some_and(|only| only != index) {
            continue;
        }
        let case = make(width, height, kernel, stride, padded_cout);
        let shape = case.shape();
        let kernels = case.kernel;
        let pixels = shape.output_width(kernels) as usize * shape.output_height(kernels) as usize;

        // `ROCKET_PADDING_MODE` selects which single variant this process
        // measures, so each one lands on a device the canary just proved
        // clean. Running them together silently corrupts whichever comes
        // after the first failure:
        //   zeroed   (default) the proposed fix -- Cout padded, surplus
        //                      channels zeroed
        //   unzeroed           padded Cout, coefficients untouched: should
        //                      pass, so a `zeroed` failure is the zeroing
        //   unpadded           the true Cout: should fail, proving the case
        //                      is a real one
        let mode = std::env::var("ROCKET_PADDING_MODE").unwrap_or_else(|_| "zeroed".into());
        if mode == "unpadded" || mode == "unzeroed" {
            let probe = if mode == "unpadded" {
                make(width, height, kernel, stride, true_cout)
            } else {
                case
            };
            let verdict = match build_raw_fixture(probe).and_then(|fixture| {
                let execution = execute_case_output(&file, &fixture)?;
                Ok(compare_output(&fixture, &execution.plan, &execution.output).mismatches)
            }) {
                Ok(0) => "ok".to_string(),
                Ok(mismatches) => format!("FAILS {mismatches}"),
                Err(error) => format!("ERROR {error}"),
            };
            println!(
                "  [{index}] {width}x{height} k{kernel}x{kernel} s{stride} {pixels}px \
                 Cout {true_cout}->{padded_cout}: {mode} {verdict}"
            );
            continue;
        }

        let mut fixture = match build_raw_fixture(case) {
            Ok(fixture) => fixture,
            Err(error) => {
                println!("  {width}x{height} k{kernel} s{stride}: build ERROR {error}");
                continue;
            }
        };
        let blocks = (padded_cout / 32) as usize;
        assert_eq!(
            fixture.weights.len() % blocks,
            0,
            "packed coefficients should divide evenly into {blocks} output blocks"
        );
        let block_bytes = fixture.weights.len() / blocks;
        fixture.weights[(true_cout as usize / 32) * block_bytes..].fill(0);

        let execution = match execute_case_output(&file, &fixture) {
            Ok(execution) => execution,
            Err(error) => {
                println!("  {width}x{height} k{kernel} s{stride}: padded ERROR {error}");
                continue;
            }
        };

        let (mut real_bad, mut pad_nonzero) = (0usize, 0usize);
        let mut sample = None;
        for y in 0..shape.output_height(kernels) as usize {
            for x in 0..shape.output_width(kernels) as usize {
                for channel in 0..padded_cout as usize {
                    let offset = output_offset(shape, kernels, channel, y, x);
                    let got = i32::from_le_bytes(
                        execution.output[offset..offset + 4].try_into().unwrap(),
                    );
                    if channel < true_cout as usize {
                        let want = expected_output(case, channel, y, x);
                        if got != want {
                            real_bad += 1;
                            sample.get_or_insert(format!(
                                "[y={y},x={x},c={channel}] want {want} got {got}"
                            ));
                        }
                    } else if got != 0 {
                        pad_nonzero += 1;
                    }
                }
            }
        }

        println!(
            "  [{index}] {width}x{height} k{kernel}x{kernel} s{stride} {pixels}px \
             Cout {true_cout}->{padded_cout}: padded real_bad={real_bad} pad_nonzero={pad_nonzero}{}",
            sample.map(|s| format!("  first {s}")).unwrap_or_default(),
        );
    }
}

/// The high-`Cin` shapes whose CBUF split comes from dividing the streamed
/// output-channel group.
///
/// A first attempt at this was reverted on 2026-09-02: it gated the division
/// on whether the *feature map* still fit, which is spatial, and drove 56x56
/// `Cin` 640 to five weight banks against the vendor's seven -- wrong values
/// on hardware, and a hang at `Cin` 768. The spatial corpus
/// (`conv_vendor_fixtures_spatial*.json`, six extents) showed the divisor is
/// only ever one or two and is decided by the coefficient working set alone,
/// so the gate is now "does the undivided grant leave at least two data
/// banks", with no spatial term. All four 56x56 shapes now plan the vendor's
/// 6/7/8/8.
///
/// Run one case per process -- this sweep contaminates itself, and a case can
/// fail in the batch while passing alone:
///
/// ```text
/// for i in $(seq 0 N); do ROCKET_PROBE_ONLY=$i ./conv2d_oracle_hw \
///     group_division_high_channel_splits_match_oracle --ignored --nocapture; done
/// ```
#[cfg(feature = "hardware-characterization")]
fn group_division_split_cases() -> Vec<Conv2dCase> {
    let mut cases = Vec::new();
    for precision in [OraclePrecision::Fp16, OraclePrecision::Int8] {
        // 512 is the last undivided grant and is already capture-backed --
        // it is here as the control that must keep passing.
        for cin in [512u32, 576, 640, 704, 768] {
            let mut patterns = vec![
                // Uniform 1s: the accumulator is exactly `Cin * valid taps`,
                // so a split that reads short of the resident window shows up
                // directly as a low sum. It cannot see a *reordering*, which
                // is why it is not the only pattern here -- uniform data has
                // hidden real layout bugs in this HAL before.
                OraclePattern::Counting,
                // Three signed taps per output at distinct HWCF positions,
                // with inputs varying in y, x and channel: this is the one
                // that catches a permutation. Its term count is tiny, so the
                // fp16 comparison stays exact at every Cin here.
                //
                // int8 must use the *affine* encoding of the same logical
                // filter -- raw `Selectors` is a signed coefficient form that
                // is not the ordinary int8 ABI, and mismatches under it say
                // nothing about the device. The Cartesian sweep above makes
                // the same split for the same reason.
                if precision == OraclePrecision::Int8 {
                    OraclePattern::SelectorsAffine { phase: 0 }
                } else {
                    OraclePattern::Selectors { phase: 0 }
                },
            ];
            // `Dense` is deliberately not used here. At fp16 its exactness
            // argument does not survive this term count -- inputs reach +-3
            // and coefficients +-2 over `Cin * 9` taps, so Cin 768 admits
            // accumulators near 41k, far past the 2048 below which fp16 still
            // represents every integer -- and the rest of the suite only
            // exercises it at Fp16 and Int8Accumulator, never plain Int8.
            let _ = &mut patterns;
            for pattern in patterns {
                cases.push(Conv2dCase {
                    width: 28,
                    height: 28,
                    cin,
                    cout: 256,
                    kernel: [3, 3],
                    stride: 1,
                    padding: [1, 1],
                    precision,
                    pattern,
                });
            }
        }
    }
    // 28x28/Cout 256 is the geometry the vendor channel grid captures, so it
    // is where a split disagreement is attributable. It is a single point in
    // (spatial, Cout) though, and the grant is a function of both -- the
    // division threshold depends on how many data banks the feature map needs.
    // Vary each around a divided-group Cin so the region is covered rather
    // than one cell of it.
    for precision in [OraclePrecision::Fp16, OraclePrecision::Int8] {
        let permuting = if precision == OraclePrecision::Int8 {
            OraclePattern::SelectorsAffine { phase: 1 }
        } else {
            OraclePattern::Selectors { phase: 1 }
        };
        for (width, height, cout) in [
            (14u32, 14u32, 256u32),
            (56, 56, 256),
            (112, 112, 256),
            (28, 28, 64),
            (28, 28, 768),
            (14, 14, 768),
        ] {
            for cin in [640u32, 768] {
                for pattern in [OraclePattern::Counting, permuting] {
                    cases.push(Conv2dCase {
                        width,
                        height,
                        cin,
                        cout,
                        kernel: [3, 3],
                        stride: 1,
                        padding: [1, 1],
                        precision,
                        pattern,
                    });
                }
            }
        }
    }
    cases
}

/// The horizontally tiled shapes whose column partition moved when
/// `dispatch_optimal_column_widths` replaced an even division.
///
/// A column tile's row capacity is a floor function of its width, so two
/// equal columns can both land just under the same capacity step and waste
/// the slack. Dividing unevenly pushes one of them over the step and buys it
/// a whole extra row per tile. Across the corpora this is -14.6% dispatches
/// with no case getting worse and no case changing its column *count*.
///
/// The column count is what governs how much kernel halo gets re-read, and
/// it is unchanged here -- so unlike the CBUF split work, this moves only
/// the tile geometry, not the CBUF programming. What still has to be proved
/// on silicon is that these uneven partitions are *correct*: every column
/// tile programs its own `datain_width`, `surf_stride` and grouped-line
/// mode, and an off-by-one in a partition the planner never emitted before
/// would show up as a corrupt column seam.
///
/// `Counting` catches a tile reading short of its window; the selector
/// patterns catch a column landing at the wrong offset, which uniform data
/// cannot see. The spread covers 2-, 3-, 4- and 5-column partitions in both
/// precisions. Shapes above ~450 dispatches are left to the fixture corpus.
///
/// Run one case per process.
#[cfg(feature = "hardware-characterization")]
#[test]
#[ignore = "requires the RK3588 NPU"]
fn column_partition_widths_match_oracle() {
    let mut cases = Vec::new();
    for (extent, cin, int8) in [
        (112u32, 384u32, false),
        (112, 448, false),
        (160, 320, false),
        (128, 448, false),
        (160, 512, true),
        (224, 384, true),
        (224, 448, true),
        (192, 512, true),
    ] {
        let precision = if int8 {
            OraclePrecision::Int8
        } else {
            OraclePrecision::Fp16
        };
        let permuting = if int8 {
            OraclePattern::SelectorsAffine { phase: 0 }
        } else {
            OraclePattern::Selectors { phase: 0 }
        };
        for pattern in [OraclePattern::Counting, permuting] {
            cases.push(Conv2dCase {
                width: extent,
                height: extent,
                cin,
                cout: 256,
                kernel: [3, 3],
                stride: 1,
                padding: [1, 1],
                precision,
                pattern,
            });
        }
    }
    run_hardware_case_matrix("uneven column partitions", cases);
}

/// The shapes whose CBUF split moved when `streamed_weight_bank_grant`
/// learned the vendor's kernel-window residency gate.
///
/// The size gate alone (`streamed_weight_bank_preference`) divides the
/// streamed output-channel group only when the coefficient working set is
/// too big on its own. The vendor also divides when the undivided grant
/// would leave too few feature banks to hold one kernel window -- which is
/// why `Cin` 448/512 from extent 56 up are programmed 7/5 rather than the
/// 4/8 and 3/9 the size gate predicts.
///
/// Every case here moves the grant **down** (fewer weight banks, more data
/// banks) onto the vendor's own value. Down is the direction that produced
/// wrong values and an NPU hang when a spatial gate was tried and reverted
/// on 2026-09-02, so it is the direction that has to be re-measured rather
/// than argued from the corpus -- even though the corpus now agrees with the
/// vendor exactly at each of these points.
///
/// Extents stop at 160. The rule also moves 192^2 and 224^2 cases, but those
/// plan into the low hundreds of dispatches each and the sweep already
/// contaminates itself; they are covered by the fixture corpus only.
///
/// Run one case per process.
#[cfg(feature = "hardware-characterization")]
#[test]
#[ignore = "requires the RK3588 NPU"]
fn kernel_window_residency_splits_match_oracle() {
    let mut cases = Vec::new();
    // (extent, Cin, precision) -- exactly the moved cases at extents <= 160.
    for (extent, cin, int8) in [
        (56u32, 448u32, false),
        (56, 512, false),
        (112, 320, false),
        (112, 448, true),
        (112, 512, true),
        (128, 320, false),
        (128, 448, true),
        (128, 512, true),
        (160, 256, false),
    ] {
        let precision = if int8 {
            OraclePrecision::Int8
        } else {
            OraclePrecision::Fp16
        };
        let permuting = if int8 {
            OraclePattern::SelectorsAffine { phase: 0 }
        } else {
            OraclePattern::Selectors { phase: 0 }
        };
        for pattern in [OraclePattern::Counting, permuting] {
            cases.push(Conv2dCase {
                width: extent,
                height: extent,
                cin,
                cout: 256,
                kernel: [3, 3],
                stride: 1,
                padding: [1, 1],
                precision,
                pattern,
            });
        }
    }
    run_hardware_case_matrix("kernel-window residency splits", cases);
}

/// The small-extent shapes whose CBUF split changed when
/// `demand_based_cbuf_partition`'s starvation correction learned to tell
/// "clamped down to one data bank" from "only ever wanted one".
///
/// At 7x7 and 14x14 the feature map fits a single CBUF bank at every `Cin` in
/// the corpus, so the old `data_banks == 1` trigger fired on shapes that were
/// never starved and rewrote them from the vendor's 1/11 to 9/3 or 8/4. These
/// are the extents the MobileNetV2 and ResNet tails run at, so the split is
/// on the compiled path, not just in the corpus.
///
/// What this has to establish is that the *new* value is right on silicon --
/// one data bank really does hold these maps -- since the host fixtures only
/// show it is what the vendor programs. `Counting` catches a split that reads
/// short of the resident window (the accumulator comes back low); the
/// selector patterns catch a permutation, which uniform data cannot see.
///
/// Run one case per process; this sweep contaminates itself like the others.
#[cfg(feature = "hardware-characterization")]
#[test]
#[ignore = "requires the RK3588 NPU"]
fn single_data_bank_small_extent_splits_match_oracle() {
    let mut cases = Vec::new();
    for precision in [OraclePrecision::Fp16, OraclePrecision::Int8] {
        let permuting = if precision == OraclePrecision::Int8 {
            OraclePattern::SelectorsAffine { phase: 0 }
        } else {
            OraclePattern::Selectors { phase: 0 }
        };
        // 7x7 moves at every one of these; 14x14 moves at 64 and 128 only,
        // and is here at 256 as the control that must not move.
        for (extent, cin) in [
            (7u32, 64u32),
            (7, 128),
            (7, 256),
            (7, 320),
            (7, 512),
            (14, 64),
            (14, 128),
            (14, 256),
        ] {
            for pattern in [OraclePattern::Counting, permuting] {
                cases.push(Conv2dCase {
                    width: extent,
                    height: extent,
                    cin,
                    cout: 256,
                    kernel: [3, 3],
                    stride: 1,
                    padding: [1, 1],
                    precision,
                    pattern,
                });
            }
        }
    }
    run_hardware_case_matrix("single-data-bank small extents", cases);
}

/// Re-measures the dense 3x3 `Cin` cap in **both** int8 output modes.
///
/// The transform spec caps dense int8 3x3 at `Cin` 32 on the strength of
/// "exact to 32, wrong from 33 up". That measurement was made through the
/// accumulator dispatch, and the 1x1 cliff turned out to be a property of the
/// accumulator *output* path rather than of int8 convolution -- plain int8 is
/// exact at every `Cin` where the accumulator is wrong. If the same holds at
/// 3x3, the cap is an artifact of the output mode and far more of MobileNetV2
/// could be offloaded than the spec currently allows.
///
/// Both patterns matter here: `Counting` is uniform 1s and catches a short
/// read, `SelectorsAffine` is the int8 permutation probe (raw `Selectors` is a
/// signed coefficient form that is not the ordinary int8 ABI and says nothing
/// about the device).
///
/// Run one case per process; this sweep contaminates itself.
#[cfg(feature = "hardware-characterization")]
#[test]
#[ignore = "requires the RK3588 NPU"]
fn dense_int8_3x3_channel_cap_control() {
    let mut cases = Vec::new();
    for precision in [OraclePrecision::Int8, OraclePrecision::Int8Accumulator] {
        for cin in [16u32, 32, 33, 64, 128, 256, 512] {
            for pattern in [
                OraclePattern::Counting,
                OraclePattern::SelectorsAffine { phase: 0 },
            ] {
                cases.push(Conv2dCase {
                    width: 28,
                    height: 28,
                    cin,
                    cout: 256,
                    kernel: [3, 3],
                    stride: 1,
                    padding: [1, 1],
                    precision,
                    pattern,
                });
            }
        }
    }
    run_hardware_case_matrix("dense int8 3x3 cap, both output modes", cases);
}

#[cfg(feature = "hardware-characterization")]
#[test]
#[ignore = "requires the RK3588 NPU"]
fn group_division_high_channel_splits_match_oracle() {
    // NOTE: this sweep contaminates itself -- a case can fail in the batch and
    // pass in isolation (fp16 Cin 512 selectors does exactly that). For a
    // verdict, drive it one case per process with `ROCKET_PROBE_ONLY=<index>`.
    // The planner's channel cap is what this run exists to justify raising, so
    // the shapes have to be reachable before the cap moves.
    unsafe { std::env::set_var("ROCKET_ALLOW_UNBACKED_CHANNELS", "1") };
    run_hardware_case_matrix(
        "group-division high-channel CBUF splits",
        group_division_split_cases(),
    );
}

/// Dense int8 **accumulator** 1x1 across the `Cin` cliff, for the grains
/// experiment.
///
/// This is the shape family that is exact to `Cin` 384 and wrong from 400 up
/// with exactly one output row correct. The vendor capture diff
/// (`k1_residency_diff`) showed our tile count, bank split, `in_rows` and
/// `cbuf_data_entries` match the vendor *exactly* across the cliff, so the CBUF
/// residency arithmetic is not the cause. The one field that differs is
/// `feature_grains`, which no gate test checks: ours is rows-driven and sits
/// near 33 while the vendor drives it down with channel pressure to 9 and then
/// 6.
///
/// Drive the arms with `ROCKET_FEATURE_GRAINS=<n>` or
/// `ROCKET_FEATURE_GRAINS_MAX=<n>`; `Cin > 384` needs
/// `ROCKET_ALLOW_KNOWN_BAD_SHAPES=1`. `Cin` 384 is the control and must stay
/// exact in every arm -- if it breaks, the arm is invalid, not informative.
/// Locates the accumulator boundary for several kernels at *one* geometry.
///
/// The two boundaries measured so far come from shapes that differ in three
/// ways at once -- k=3 fails above `Cin` 32 at 28x28 Cout 256, k=1 is exact to
/// `Cin` 384 at 32x32 Cout 64 -- so nothing can be concluded about what the
/// boundary is a function of. This holds 32x32 and Cout 64 fixed and sweeps
/// `Cin` finely for k=1, 3 and 5, so the three boundaries are directly
/// comparable.
///
/// `Cin > 384` at k=1 needs `ROCKET_ALLOW_KNOWN_BAD_SHAPES=1`, which the test
/// sets. Run one case per process.
/// Tests the coefficient-footprint hypothesis by varying **Cout** at fixed
/// `Cin`, kernel and geometry.
///
/// Both boundaries in `accumulator_boundary_sweep` bracket the same total
/// coefficient size (`padded_Cin * taps * Cout`): k=3 `Cin` 32 is 18432 bytes
/// and passes, k=1 `Cin` 384 is 24576 and passes, k=1 `Cin` 400 is 25600 and
/// fails, k=3 `Cin` 33 pads to 48 for 27648 and fails. If the accumulator is
/// correct iff that footprint stays under roughly 24-25 KiB, then at k=3
/// `Cin` 48 -- which fails at Cout 64 -- **shrinking Cout alone must fix it**,
/// with the boundary between Cout 56 (24192 bytes) and Cout 64 (27648).
/// Pins the accumulator threshold in coefficient bytes **per output channel**.
///
/// Total coefficient size is ruled out: at k=3 `Cin` 48 every Cout from 8 to
/// 64 fails, including Cout 8 at 3456 bytes. Normalising per output channel
/// fits every point instead -- `padded_Cin * taps` is 288 and 384 where the
/// device is exact, 400 and 432 where it is wrong. This walks k=1 across
/// 384 -> 400 one atom at a time, where `padded_Cin * taps` is just the padded
/// channel count, to place the edge exactly. 384 is 6 whole 64-byte
/// coefficient groups.
/// Discriminating test for the per-output-channel coefficient limit.
///
/// k=1 places the edge exactly between `Cin` 384 (exact) and 385 (wrong), and
/// k=3 between padded `Cin` 32 (288 bytes) and 48 (432). Both fit
/// "correct iff `padded_Cin * taps <= 384`", but they cannot separate that
/// from a plain `Cin` limit that happens to coincide.
///
/// A **1x3** kernel does separate them: taps is 3, so the rule predicts the
/// edge at padded `Cin` 128 (384 bytes) -- nowhere near 384 channels. `Cin` 112
/// (336) and 128 (384) must be exact; `Cin` 129 pads to 144 (432) and must
/// fail. A plain channel limit predicts all four exact.
#[cfg(feature = "hardware-characterization")]
fn accumulator_rectangular_kernel_cases() -> Vec<Conv2dCase> {
    [96u32, 112, 128, 129]
        .into_iter()
        .map(|cin| Conv2dCase {
            width: 32,
            height: 32,
            cin,
            cout: 64,
            kernel: [1, 3],
            stride: 1,
            padding: [0, 1],
            precision: OraclePrecision::Int8Accumulator,
            pattern: OraclePattern::Counting,
        })
        .collect()
}

#[cfg(feature = "hardware-characterization")]
#[test]
#[ignore = "requires the RK3588 NPU"]
fn accumulator_rectangular_kernel_probe() {
    unsafe { std::env::set_var("ROCKET_ALLOW_KNOWN_BAD_SHAPES", "1") };
    unsafe { std::env::set_var("ROCKET_ALLOW_UNBACKED_CHANNELS", "1") };
    run_hardware_case_matrix(
        "int8 accumulator 1x3 kernel, per-channel coefficient limit",
        accumulator_rectangular_kernel_cases(),
    );
}

/// What is the number of *written* output lanes a function of?
///
/// Past the coefficient limit the DPU leaves most lanes untouched
/// (`OUTPUT_SENTINEL`) rather than writing wrong values. With all three
/// `ROCKET_PAD_*` set the job no longer faults, so the written count is stable
/// run to run and the device never wedges -- which makes this sweepable.
///
/// Three axes, each varied with the others held fixed: `Cin` past the limit,
/// `Cout` (which sets `blocks_per_pixel`), and the spatial extent (which sets
/// pixels and tile count).
#[cfg(feature = "hardware-characterization")]
fn accumulator_written_lanes_cases() -> Vec<Conv2dCase> {
    let base = |width, height, cin, cout, kernel: usize| Conv2dCase {
        width,
        height,
        cin,
        cout,
        kernel: [kernel, kernel],
        stride: 1,
        padding: [kernel / 2, kernel / 2],
        precision: OraclePrecision::Int8Accumulator,
        pattern: OraclePattern::Counting,
    };
    let mut cases = Vec::new();
    // Cin axis: 32x32, Cout 64, k=1.
    for cin in [385u32, 400, 448, 512] {
        cases.push(base(32, 32, cin, 64, 1));
    }
    // Cout axis: 32x32, Cin 400, k=1.
    for cout in [32u32, 128, 256] {
        cases.push(base(32, 32, 400, cout, 1));
    }
    // Spatial axis: Cin 400, Cout 64, k=1.
    for extent in [16u32, 64] {
        cases.push(base(extent, extent, 400, 64, 1));
    }
    // k=3 cross-check, just past its own limit.
    cases.push(base(32, 32, 33, 64, 3));
    cases.push(base(32, 32, 64, 64, 3));
    cases
}

#[cfg(feature = "hardware-characterization")]
#[test]
#[ignore = "requires the RK3588 NPU"]
fn accumulator_written_lanes_probe() {
    unsafe { std::env::set_var("ROCKET_ALLOW_KNOWN_BAD_SHAPES", "1") };
    unsafe { std::env::set_var("ROCKET_ALLOW_UNBACKED_CHANNELS", "1") };
    run_hardware_case_matrix(
        "int8 accumulator written-lane count",
        accumulator_written_lanes_cases(),
    );
}

#[cfg(feature = "hardware-characterization")]
fn accumulator_per_channel_threshold_cases() -> Vec<Conv2dCase> {
    [352u32, 368, 384, 385, 392, 400, 416]
        .into_iter()
        .map(|cin| Conv2dCase {
            width: 32,
            height: 32,
            cin,
            cout: 64,
            kernel: [1, 1],
            stride: 1,
            padding: [0, 0],
            precision: OraclePrecision::Int8Accumulator,
            pattern: OraclePattern::Counting,
        })
        .collect()
}

#[cfg(feature = "hardware-characterization")]
#[test]
#[ignore = "requires the RK3588 NPU"]
fn accumulator_per_channel_threshold_probe() {
    unsafe { std::env::set_var("ROCKET_ALLOW_KNOWN_BAD_SHAPES", "1") };
    unsafe { std::env::set_var("ROCKET_ALLOW_UNBACKED_CHANNELS", "1") };
    run_hardware_case_matrix(
        "int8 accumulator per-output-channel coefficient threshold",
        accumulator_per_channel_threshold_cases(),
    );
}

#[cfg(feature = "hardware-characterization")]
fn accumulator_coefficient_footprint_cases() -> Vec<Conv2dCase> {
    [8u32, 16, 32, 48, 56, 64]
        .into_iter()
        .map(|cout| Conv2dCase {
            width: 32,
            height: 32,
            cin: 48,
            cout,
            kernel: [3, 3],
            stride: 1,
            padding: [1, 1],
            precision: OraclePrecision::Int8Accumulator,
            pattern: OraclePattern::Counting,
        })
        .collect()
}

#[cfg(feature = "hardware-characterization")]
#[test]
#[ignore = "requires the RK3588 NPU"]
fn accumulator_coefficient_footprint_probe() {
    unsafe { std::env::set_var("ROCKET_ALLOW_KNOWN_BAD_SHAPES", "1") };
    run_hardware_case_matrix(
        "int8 accumulator coefficient footprint, k=3 Cin 48",
        accumulator_coefficient_footprint_cases(),
    );
}

#[cfg(feature = "hardware-characterization")]
fn accumulator_boundary_sweep_cases() -> Vec<Conv2dCase> {
    let mut cases = Vec::new();
    for kernel in [1usize, 3, 5] {
        for cin in [
            16u32, 24, 32, 33, 40, 48, 64, 96, 128, 192, 256, 320, 352, 384, 400, 448, 512,
        ] {
            cases.push(Conv2dCase {
                width: 32,
                height: 32,
                cin,
                cout: 64,
                kernel: [kernel, kernel],
                stride: 1,
                padding: [kernel / 2, kernel / 2],
                precision: OraclePrecision::Int8Accumulator,
                pattern: OraclePattern::Counting,
            });
        }
    }
    cases
}

#[cfg(feature = "hardware-characterization")]
#[test]
#[ignore = "requires the RK3588 NPU"]
fn accumulator_boundary_sweep() {
    unsafe { std::env::set_var("ROCKET_ALLOW_KNOWN_BAD_SHAPES", "1") };
    unsafe { std::env::set_var("ROCKET_ALLOW_UNBACKED_CHANNELS", "1") };
    run_hardware_case_matrix(
        "int8 accumulator boundary, 32x32 Cout 64, k=1/3/5",
        accumulator_boundary_sweep_cases(),
    );
}

/// Does the accumulator failure track `Cin`, or output-channel coverage?
///
/// At 3x3 `Cin` 33 Cout 256 the device writes exactly one 128-byte output
/// block -- channels 0..31 are right and the rest is untouched `OUTPUT_SENTINEL`
/// -- so the symptom is *coverage*, not corruption. `blocks_per_pixel` is
/// `padded_out_channels * 4 / 128`, i.e. 1, 2, 4 and 8 at Cout 32, 64, 128 and
/// 256. If only the multi-block Couts fail, the cap is a Cout property that
/// `Cin` merely correlates with.
#[cfg(feature = "hardware-characterization")]
fn accumulator_cout_coverage_cases() -> Vec<Conv2dCase> {
    let mut cases = Vec::new();
    for cin in [32u32, 33, 64] {
        for cout in [32u32, 64, 128, 256] {
            cases.push(Conv2dCase {
                width: 28,
                height: 28,
                cin,
                cout,
                kernel: [3, 3],
                stride: 1,
                padding: [1, 1],
                precision: OraclePrecision::Int8Accumulator,
                pattern: OraclePattern::Counting,
            });
        }
    }
    cases
}

#[cfg(feature = "hardware-characterization")]
#[test]
#[ignore = "requires the RK3588 NPU"]
fn accumulator_cout_coverage_probe() {
    run_hardware_case_matrix(
        "int8 accumulator output-block coverage",
        accumulator_cout_coverage_cases(),
    );
}

#[cfg(feature = "hardware-characterization")]
fn accumulator_grains_cases() -> Vec<Conv2dCase> {
    // Both precisions at the same shapes. The register diff showed our conv
    // programming is bit-identical to the vendor's across the cliff for every
    // register both sides write, so the remaining suspects are the accumulator
    // *output* path (which the vendor never exercises, so no capture can
    // adjudicate it) and buffer packing. Plain int8 shares the packing and the
    // whole input side but takes the ordinary requantized output, so it
    // separates the two: if plain int8 is exact where the accumulator is
    // wrong, the fault is downstream of the DPU accumulate.
    [OraclePrecision::Int8Accumulator, OraclePrecision::Int8]
        .into_iter()
        .flat_map(|precision| {
            [384u32, 400, 448, 512]
                .into_iter()
                .map(move |cin| (precision, cin))
        })
        .map(|(precision, cin)| Conv2dCase {
            width: 32,
            height: 32,
            cin,
            cout: 64,
            kernel: [1, 1],
            stride: 1,
            padding: [0, 0],
            precision,
            pattern: OraclePattern::Counting,
        })
        .collect()
}

#[cfg(feature = "hardware-characterization")]
#[test]
#[ignore = "requires the RK3588 NPU"]
fn accumulator_high_channel_grains_probe() {
    unsafe { std::env::set_var("ROCKET_ALLOW_KNOWN_BAD_SHAPES", "1") };
    unsafe { std::env::set_var("ROCKET_ALLOW_UNBACKED_CHANNELS", "1") };
    let grains = std::env::var("ROCKET_FEATURE_GRAINS")
        .or_else(|_| std::env::var("ROCKET_FEATURE_GRAINS_MAX"))
        .unwrap_or_else(|_| "default".to_string());
    run_hardware_case_matrix(
        &format!("int8 accumulator 1x1 Cin cliff, grains={grains}"),
        accumulator_grains_cases(),
    );
}

/// Maps *which* bytes of the accumulator staging buffer the DPU actually
/// wrote, past the per-output-channel coefficient limit.
///
/// The written-lane counts are deterministic under padding but do not fit a
/// clean function of `Cin`, `Cout` or extent, and some are not even pixel
/// aligned. Counting lanes cannot distinguish a prefix from a stride from a
/// scattered pattern, so this reads the staging buffer byte by byte and prints
/// the run structure instead. `OUTPUT_SENTINEL` marks untouched bytes.
///
/// Worth noting up front: bad lanes read `0xA5A5A5B0`, i.e. only the *low*
/// byte of a 4-byte accumulator lane was disturbed, and 0xB0 is not the low
/// byte of the expected 385 (0x181) either. So the write granularity may be
/// finer than a lane.
#[cfg(feature = "hardware-characterization")]
#[test]
#[ignore = "requires the RK3588 NPU"]
fn accumulator_written_region_map() {
    unsafe { std::env::set_var("ROCKET_ALLOW_KNOWN_BAD_SHAPES", "1") };
    unsafe { std::env::set_var("ROCKET_ALLOW_UNBACKED_CHANNELS", "1") };
    let _guard = NPU_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");

    // One axis varied at a time. Tile 0's prefix stopped at exactly 24 pixels
    // for both Cin 385 and 400, so the question is what that 24 is a function
    // of -- and tile 1 differed (63 vs 2 px), so the tiles must be read apart.
    let mut cases: Vec<(u32, u32, u32, usize)> = Vec::new();
    for cin in [385u32, 392, 400, 416, 448, 512] {
        cases.push((32, cin, 64, 1));
    }
    for cout in [32u32, 128, 256] {
        cases.push((32, 400, cout, 1));
    }
    for extent in [16u32, 64, 128] {
        cases.push((extent, 400, 64, 1));
    }
    for cin in [33u32, 48, 64] {
        cases.push((32, cin, 64, 3));
    }

    for (extent, cin, cout, kernel) in cases {
        let case = Conv2dCase {
            width: extent,
            height: extent,
            cin,
            cout,
            kernel: [kernel, kernel],
            stride: 1,
            padding: [kernel / 2, kernel / 2],
            precision: OraclePrecision::Int8Accumulator,
            pattern: OraclePattern::Counting,
        };
        let fixture = match build_fixture(case) {
            Ok(fixture) => fixture,
            Err(error) => {
                println!("  {extent}^2 Cin={cin} Cout={cout} k={kernel}: fixture failed: {error}");
                continue;
            }
        };
        let execution = match execute_case_output(&file, &fixture) {
            Ok(execution) => execution,
            Err(error) => {
                println!("  {extent}^2 Cin={cin} Cout={cout} k={kernel}: execute failed: {error}");
                continue;
            }
        };
        let raw = execution.raw;
        let pixel_bytes = fixture.shape.padded_out_channels() as usize
            * fixture.shape.precision.output_element_bytes() as usize;

        // Written runs with their offsets: the tile partition shows up as the
        // gaps between them, and a prefix per tile as a run at each tile base.
        let mut runs: Vec<(usize, usize)> = Vec::new();
        let mut offset = 0usize;
        while offset < raw.len() {
            if raw[offset] != OUTPUT_SENTINEL {
                let start = offset;
                while offset < raw.len() && raw[offset] != OUTPUT_SENTINEL {
                    offset += 1;
                }
                runs.push((start, offset - start));
            } else {
                offset += 1;
            }
        }
        let total: usize = runs.iter().map(|(_, len)| len).sum();
        let described: Vec<String> = runs
            .iter()
            .take(6)
            .map(|(start, len)| {
                format!(
                    "@{start}+{len}({}px)",
                    if pixel_bytes > 0 {
                        len / pixel_bytes
                    } else {
                        0
                    }
                )
            })
            .collect();
        println!(
            "  {extent}^2 Cin={cin:<4} Cout={cout:<4} k={kernel}  staging={:<7} written={total:<7} tiles={}  runs: {}",
            raw.len(),
            execution.plan.tiles().len(),
            described.join(" ")
        );
    }
}
