//! Wall-clock accounting for where an inference's time actually goes
//! (`ROCKET_PROFILE=1`).
//!
//! `ROCKET_DISPATCH_TIMES` (device.rs) answers "how long did this hardware
//! job take", which is what the hung-job floor needs. It cannot answer "why
//! is this model slow", because on this driver the hardware job is only one
//! of the costs a dispatch pays. Every Rocket dispatch also repacks its
//! input NHWC -> NC1HWC2, repacks its weights into the CNA's blocked
//! coefficient order, widens its bias, and compacts the DPU's atomic-slot
//! output back into IREE's dense ABI buffer -- all on the host CPU, all per
//! dispatch, all invisible to a per-job timer (see the "layout repack per
//! dispatch" note: nothing propagates a packed layout between dispatches
//! yet, so a chain of NPU ops pays the round trip at every link).
//!
//! This module times each of those phases separately and prints two tables
//! at process exit: totals per phase, and a per-op breakdown keyed by the
//! op's shape. Reading a model's bottleneck off that is the point --
//! "the NPU is fast and the repacking is eating it" and "one dispatch
//! dominates" look nothing alike in these numbers, and neither is visible
//! from end-to-end wall time.
//!
//! [`Phase::Outside`] closes the loop: it is the time between one
//! `queue_execute` returning and the next one starting, i.e. everything
//! IREE did that was *not* this driver -- CPU dispatches, mostly. Summed
//! against the phases below it says whether the NPU half is even the half
//! worth optimizing.
//!
//! Enabled by env var and read once (a per-dispatch `getenv` would sit in
//! the inference loop), same convention as device.rs's diagnostic knobs.
//! Disabled, every entry point here is an atomic load and a branch.
//!
//! ```text
//! ROCKET_PROFILE=1      summary tables at exit
//! ROCKET_PROFILE=trace  the above, plus a line per timed phase as it happens
//! ```

use std::{
    collections::HashMap,
    sync::{
        Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

/// One timed stage of a dispatch's life, in the order a dispatch pays them.
///
/// `Record` happens at command-buffer record time (`command_buffer::
/// dispatch`); everything from `PackInput` on happens at submit time
/// (`device::queue_execute`, via `command_buffer::apply_ops`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(usize)]
pub enum Phase {
    /// Between `queue_execute` calls: IREE doing something that is not this
    /// driver. Not attributable to any one op.
    Outside = 0,
    /// Planning the convolution and building its regcmd program.
    Record = 1,
    /// Dense NHWC -> NC1HWC2 input repack.
    PackInput = 2,
    /// Logical HWCF -> CNA blocked coefficient repack.
    PackWeights = 3,
    /// Bias widening/padding.
    PackBias = 4,
    /// `fini_bo` cache maintenance over every buffer the NPU will read.
    SyncInputs = 5,
    /// Allocating the regcmd BOs and copying the program words in.
    Regcmd = 6,
    /// The `DRM_ROCKET_SUBMIT` ioctl itself.
    Submit = 7,
    /// `PREP_BO`, i.e. waiting for the hardware to finish. The only phase
    /// that is actually NPU time.
    Wait = 8,
    /// Atomic-slot scratch -> dense IREE output buffer.
    Compact = 9,
    /// The deliberate inter-dispatch dwell (`DEPTHWISE_TO_DENSE_QUIESCENCE`).
    Quiesce = 10,
    /// The whole `queue_execute` callback, as a check total: everything
    /// above it minus the phases above sums to unaccounted host overhead.
    Execute = 11,
}

impl Phase {
    pub const ALL: [Phase; PHASE_COUNT] = [
        Phase::Outside,
        Phase::Record,
        Phase::PackInput,
        Phase::PackWeights,
        Phase::PackBias,
        Phase::SyncInputs,
        Phase::Regcmd,
        Phase::Submit,
        Phase::Wait,
        Phase::Compact,
        Phase::Quiesce,
        Phase::Execute,
    ];

    fn name(self) -> &'static str {
        match self {
            Phase::Outside => "outside",
            Phase::Record => "record",
            Phase::PackInput => "pack.input",
            Phase::PackWeights => "pack.weights",
            Phase::PackBias => "pack.bias",
            Phase::SyncInputs => "sync.inputs",
            Phase::Regcmd => "regcmd",
            Phase::Submit => "submit",
            Phase::Wait => "wait.npu",
            Phase::Compact => "compact",
            Phase::Quiesce => "quiesce",
            Phase::Execute => "execute",
        }
    }

    /// Short column heading for the per-op table.
    fn short_name(self) -> &'static str {
        match self {
            Phase::Outside => "outsd",
            Phase::Record => "rec",
            Phase::PackInput => "pk.in",
            Phase::PackWeights => "pk.wt",
            Phase::PackBias => "pk.bs",
            Phase::SyncInputs => "sync",
            Phase::Regcmd => "regcmd",
            Phase::Submit => "submit",
            Phase::Wait => "npu",
            Phase::Compact => "cmpct",
            Phase::Quiesce => "quies",
            Phase::Execute => "exec",
        }
    }

    /// Whether this phase nests inside `Execute` and so must not be added to
    /// it when totalling. `Record` runs before `queue_execute` and `Outside`
    /// runs between calls, so both are genuinely disjoint from it.
    fn nested_in_execute(self) -> bool {
        !matches!(self, Phase::Outside | Phase::Record | Phase::Execute)
    }
}

/// Number of [`Phase`] variants, as an array length.
const PHASE_COUNT: usize = 12;

/// The label used for phases that belong to no particular op.
pub const NO_OP: &str = "-";

fn level() -> u8 {
    static LEVEL: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *LEVEL.get_or_init(|| match std::env::var("ROCKET_PROFILE") {
        Ok(value) if value == "trace" => 2,
        Ok(value) if value != "0" && !value.is_empty() => 1,
        _ => 0,
    })
}

/// Whether any profiling is on. Checked at every call site, so it is one
/// initialized-`OnceLock` load after the first call.
#[inline]
pub fn enabled() -> bool {
    level() > 0
}

fn tracing() -> bool {
    level() >= 2
}

/// Starts a phase timer, or `None` when profiling is off.
///
/// Paired with [`stop`]. Returning an `Option` rather than a guard keeps the
/// disabled path free of any drop glue, and lets a call site that bails out
/// early (every packing step has error returns) simply not record.
#[inline]
pub fn start() -> Option<Instant> {
    enabled().then(Instant::now)
}

/// Records a phase against an op label, ignoring a `None` start.
///
/// `bytes` is how much data the phase moved, used for the MB/s column; pass
/// 0 where that is meaningless (a wait, a sleep, an ioctl).
pub fn stop(started: Option<Instant>, phase: Phase, label: &str, bytes: usize) {
    let Some(started) = started else { return };
    record(phase, label, started.elapsed(), bytes);
}

/// Records an already-measured duration -- for the few call sites that need
/// the elapsed time for their own reasons anyway (the hung-job floor).
pub fn record(phase: Phase, label: &str, elapsed: Duration, bytes: usize) {
    if !enabled() {
        return;
    }
    if tracing() {
        eprintln!(
            "rocket profile: {:<12} {:>8.3} ms  {}",
            phase.name(),
            elapsed.as_secs_f64() * 1e3,
            label,
        );
    }
    install_exit_hook();
    let mut registry = registry().lock().unwrap_or_else(|e| e.into_inner());
    registry.add(phase, label, elapsed, bytes);
}

/// The time between `queue_execute` calls -- everything IREE did that was
/// not this driver. Call on entry to `queue_execute`; the first call has no
/// predecessor and only arms the clock.
pub fn mark_outside_start() {
    if !enabled() {
        return;
    }
    let mut registry = registry().lock().unwrap_or_else(|e| e.into_inner());
    if let Some(last) = registry.last_execute_end.take() {
        let elapsed = last.elapsed();
        drop(registry);
        record(Phase::Outside, NO_OP, elapsed, 0);
    }
}

/// Call on exit from `queue_execute`, to start the [`Phase::Outside`] clock.
pub fn mark_outside_end() {
    if !enabled() {
        return;
    }
    let mut registry = registry().lock().unwrap_or_else(|e| e.into_inner());
    registry.last_execute_end = Some(Instant::now());
}

/// Builds an op label only when profiling is on, so the `format!` a caller
/// passes never runs otherwise.
#[inline]
pub fn label(f: impl FnOnce() -> String) -> String {
    if enabled() { f() } else { String::new() }
}

#[derive(Clone, Copy, Default)]
struct PhaseStat {
    calls: u64,
    nanos: u128,
    max_nanos: u128,
    bytes: u64,
}

impl PhaseStat {
    fn add(&mut self, elapsed: Duration, bytes: usize) {
        self.calls += 1;
        self.nanos += elapsed.as_nanos();
        self.max_nanos = self.max_nanos.max(elapsed.as_nanos());
        self.bytes += bytes as u64;
    }

    fn ms(&self) -> f64 {
        self.nanos as f64 / 1e6
    }
}

/// How many CPUs the per-CPU histogram tracks. Past this the sample is
/// dropped rather than grown into: it is a diagnostic, and RK3588 has eight.
const MAX_TRACKED_CPUS: usize = 64;

#[derive(Default)]
struct Registry {
    /// Label insertion order, so the report reads in first-seen (i.e. model)
    /// order rather than hash order.
    order: Vec<String>,
    ops: HashMap<String, [PhaseStat; PHASE_COUNT]>,
    totals: [PhaseStat; PHASE_COUNT],
    /// Host nanoseconds attributed to the CPU the phase finished on.
    ///
    /// The driver's host-side work is memory-bound, and on a big.LITTLE part
    /// it matters enormously which cluster runs it: the same transforms
    /// measured 13.8 ms on RK3588's A76s and 52.4 ms on its A55s. The thread
    /// that runs `queue_execute` blocks in `PREP_BO` waiting for an NPU
    /// completion IRQ, and Linux wakes it near whichever CPU serviced that
    /// IRQ -- which, per ISSUES.md M3, is cpu0. This is how to see whether
    /// that is what is happening rather than assume it.
    cpu_nanos: Vec<u128>,
    last_execute_end: Option<Instant>,
}

impl Registry {
    fn add(&mut self, phase: Phase, label: &str, elapsed: Duration, bytes: usize) {
        self.totals[phase as usize].add(elapsed, bytes);
        // `Execute` contains the other submit-time phases, so counting it too
        // would double every nanosecond it already covers.
        if phase != Phase::Execute {
            let cpu = unsafe { nix::libc::sched_getcpu() };
            if (0..MAX_TRACKED_CPUS as i32).contains(&cpu) {
                if self.cpu_nanos.is_empty() {
                    self.cpu_nanos = vec![0; MAX_TRACKED_CPUS];
                }
                self.cpu_nanos[cpu as usize] += elapsed.as_nanos();
            }
        }
        if label == NO_OP {
            return;
        }
        if !self.ops.contains_key(label) {
            self.order.push(label.to_string());
        }
        self.ops
            .entry(label.to_string())
            .or_insert_with(|| [PhaseStat::default(); Phase::ALL.len()])[phase as usize]
            .add(elapsed, bytes);
    }
}

fn registry() -> &'static Mutex<Registry> {
    static REGISTRY: std::sync::OnceLock<Mutex<Registry>> = std::sync::OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(Registry::default()))
}

/// Prints the report from an `atexit` handler.
///
/// The driver is a staticlib inside a C host (`iree-run-module`), so there
/// is no Rust `main` to return from, and nothing guarantees the HAL device
/// is destroyed before the process ends. `report` is also called from
/// `device::destroy` and is idempotent, so whichever happens first wins.
fn install_exit_hook() {
    static INSTALLED: std::sync::Once = std::sync::Once::new();
    INSTALLED.call_once(|| {
        extern "C" fn at_exit() {
            report();
        }
        // Failure here just means no report at exit; device::destroy still
        // prints one on the ordinary path.
        unsafe { nix::libc::atexit(at_exit) };
    });
}

/// Prints both tables, once per process.
pub fn report() {
    if !enabled() {
        return;
    }
    static REPORTED: AtomicBool = AtomicBool::new(false);
    if REPORTED.swap(true, Ordering::SeqCst) {
        return;
    }
    let registry = registry().lock().unwrap_or_else(|e| e.into_inner());
    if registry.totals.iter().all(|stat| stat.calls == 0) {
        return;
    }

    eprintln!();
    eprintln!("rocket profile");
    eprintln!(
        "  {:<13} {:>6} {:>11} {:>9} {:>9} {:>10}",
        "phase", "calls", "total ms", "avg ms", "max ms", "MB/s"
    );
    for phase in Phase::ALL {
        let stat = &registry.totals[phase as usize];
        if stat.calls == 0 {
            continue;
        }
        let throughput = if stat.bytes == 0 || stat.nanos == 0 {
            String::from("-")
        } else {
            format!("{:.0}", stat.bytes as f64 / (stat.nanos as f64 / 1e9) / 1e6)
        };
        eprintln!(
            "  {:<13} {:>6} {:>11.3} {:>9.3} {:>9.3} {:>10}",
            phase.name(),
            stat.calls,
            stat.ms(),
            stat.ms() / stat.calls as f64,
            stat.max_nanos as f64 / 1e6,
            throughput,
        );
    }

    // What `execute` spent outside every phase it contains: mmap faults,
    // scratch allocation, IREE's own command-buffer walk. A large residual
    // here means the next thing to instrument is inside queue_execute.
    let execute = registry.totals[Phase::Execute as usize].nanos;
    let nested: u128 = Phase::ALL
        .iter()
        .filter(|p| p.nested_in_execute())
        .map(|p| registry.totals[*p as usize].nanos)
        .sum();
    eprintln!(
        "  {:<13} {:>6} {:>11.3}",
        "execute.other",
        "",
        execute.saturating_sub(nested) as f64 / 1e6,
    );
    let host: u128 = Phase::ALL
        .iter()
        .filter(|p| !matches!(p, Phase::Execute | Phase::Wait | Phase::Submit))
        .map(|p| registry.totals[*p as usize].nanos)
        .sum::<u128>()
        + execute.saturating_sub(nested);
    let npu =
        registry.totals[Phase::Wait as usize].nanos + registry.totals[Phase::Submit as usize].nanos;
    // Everything the profiler saw, end to end: the phases that nest inside
    // `execute` are already counted in it, so the three disjoint ones sum to
    // the run's wall time from the first `queue_execute` onward.
    let wall = registry.totals[Phase::Outside as usize].nanos
        + registry.totals[Phase::Record as usize].nanos
        + execute;
    eprintln!("  {:<13} {:>6} {:>11.3}", "wall", "", wall as f64 / 1e6);
    eprintln!(
        "  host {:.3} ms, npu {:.3} ms, npu share {:.1}%",
        host as f64 / 1e6,
        npu as f64 / 1e6,
        if host + npu == 0 {
            0.0
        } else {
            npu as f64 / (host + npu) as f64 * 100.0
        },
    );

    // Which CPUs the host half actually ran on. A driver whose transforms all
    // land on cpu0 is not a driver with slow transforms; see `cpu_nanos`.
    let total_cpu_nanos: u128 = registry.cpu_nanos.iter().sum();
    if total_cpu_nanos > 0 {
        let mut busiest: Vec<(usize, u128)> = registry
            .cpu_nanos
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, nanos)| *nanos > 0)
            .collect();
        busiest.sort_by_key(|(_, nanos)| std::cmp::Reverse(*nanos));
        let shares: Vec<String> = busiest
            .iter()
            .take(8)
            .map(|(cpu, nanos)| {
                format!(
                    "cpu{cpu} {:.0}%",
                    *nanos as f64 / total_cpu_nanos as f64 * 100.0
                )
            })
            .collect();
        eprintln!("  host time by cpu: {}", shares.join(", "));
    }

    // How much of `pack.weights` above was avoided rather than paid. A run
    // with no hits is the first inference of a process (there is no
    // within-inference reuse: every dispatch has different filters); a later
    // one with misses still in it has had something write a weight binding.
    let cache = crate::weight_cache::stats();
    if cache.hits + cache.misses_absent + cache.misses_stale > 0 {
        eprintln!(
            "  weight cache: {} hit, {} miss (new), {} miss (rewritten), {} refused (recorded \
             writer), {} refused (budget), {:.1} MiB peak",
            cache.hits,
            cache.misses_absent,
            cache.misses_stale,
            cache.recorded_writers,
            cache.over_budget,
            cache.peak_bytes as f64 / (1024.0 * 1024.0),
        );
    }

    // Per-op: which dispatch shapes the time is in, and for each, how it
    // splits between hardware and the host-side layout bridging around it.
    let columns: Vec<Phase> = Phase::ALL
        .iter()
        .copied()
        .filter(|p| {
            !matches!(p, Phase::Outside)
                && registry
                    .order
                    .iter()
                    .any(|label| registry.ops[label][*p as usize].calls > 0)
        })
        .collect();
    if columns.is_empty() {
        return;
    }
    eprintln!();
    let mut header = format!("  {:<44} {:>5} {:>9}", "op", "calls", "total ms");
    for phase in &columns {
        header.push_str(&format!(" {:>7}", phase.short_name()));
    }
    eprintln!("{header}");
    let mut rows: Vec<(&String, f64)> = registry
        .order
        .iter()
        .map(|label| {
            let stats = &registry.ops[label];
            let total: u128 = columns
                .iter()
                .filter(|p| !matches!(p, Phase::Execute))
                .map(|p| stats[*p as usize].nanos)
                .sum();
            (label, total as f64 / 1e6)
        })
        .collect();
    // Slowest first: the bottleneck is the point of the table.
    rows.sort_by(|a, b| b.1.total_cmp(&a.1));
    for (label, total_ms) in rows {
        let stats = &registry.ops[label];
        // Dispatches, not hardware jobs: one dispatch whose CBUF split makes
        // it several individually fenced jobs pays `Record` once and `Submit`
        // /`Wait` per split, so the max across phases would report a
        // six-task stem convolution as six dispatches.
        let calls = stats[Phase::Record as usize].calls;
        let mut line = format!(
            "  {:<44} {:>5} {:>9.3}",
            truncate(label, 44),
            calls,
            total_ms
        );
        for phase in &columns {
            let stat = &stats[*phase as usize];
            if stat.calls == 0 {
                line.push_str(&format!(" {:>7}", "-"));
            } else {
                line.push_str(&format!(" {:>7.2}", stat.ms()));
            }
        }
        eprintln!("{line}");
    }
    eprintln!();
}

fn truncate(label: &str, width: usize) -> String {
    if label.len() <= width {
        label.to_string()
    } else {
        format!("{}…", &label[..width - 1])
    }
}

/// Short, stable name for a precision, for op labels.
///
/// `{:?}` would work but drags a whole `Quantization` struct into the label
/// for the int8 rungs, which makes every otherwise-identical dispatch look
/// distinct.
pub fn precision_name(precision: iree_rocket_hal::rocket::conv::Precision) -> &'static str {
    use iree_rocket_hal::rocket::conv::Precision;
    match precision {
        Precision::Fp16 => "fp16",
        Precision::Bf16 => "bf16",
        Precision::Int16 => "int16",
        Precision::Fp16Accumulator => "fp16acc",
        Precision::Tf32 => "tf32",
        Precision::Int4 => "int4",
        Precision::Int8(_) => "int8",
        Precision::Int8Accumulator(_) => "int8acc",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_aggregates_per_phase_and_per_op() {
        let mut registry = Registry::default();
        registry.add(Phase::Wait, "conv a", Duration::from_millis(3), 0);
        registry.add(Phase::Wait, "conv a", Duration::from_millis(1), 0);
        registry.add(Phase::PackInput, "conv a", Duration::from_millis(2), 64);
        registry.add(Phase::Wait, "conv b", Duration::from_millis(5), 0);

        let wait = &registry.totals[Phase::Wait as usize];
        assert_eq!(wait.calls, 3);
        assert_eq!(wait.ms(), 9.0);
        assert_eq!(wait.max_nanos, Duration::from_millis(5).as_nanos());

        let a = &registry.ops["conv a"];
        assert_eq!(a[Phase::Wait as usize].calls, 2);
        assert_eq!(a[Phase::Wait as usize].ms(), 4.0);
        assert_eq!(a[Phase::PackInput as usize].bytes, 64);
        assert_eq!(registry.order, vec!["conv a", "conv b"]);
    }

    /// Phases with no op of their own must still total, without inventing a
    /// row in the per-op table.
    #[test]
    fn unattributed_phases_total_without_an_op_row() {
        let mut registry = Registry::default();
        registry.add(Phase::Outside, NO_OP, Duration::from_millis(7), 0);
        assert_eq!(registry.totals[Phase::Outside as usize].ms(), 7.0);
        assert!(registry.ops.is_empty());
        assert!(registry.order.is_empty());
    }

    /// The report subtracts the nested phases from `execute` to get the
    /// unaccounted host overhead, so exactly the phases that run inside
    /// `queue_execute` must be marked nested.
    #[test]
    fn only_submit_time_phases_nest_in_execute() {
        for phase in Phase::ALL {
            let expected = !matches!(phase, Phase::Outside | Phase::Record | Phase::Execute);
            assert_eq!(phase.nested_in_execute(), expected, "{phase:?}");
        }
    }

    #[test]
    fn labels_are_empty_unless_profiling_is_on() {
        // The env var is not set in the test process, so the closure that
        // would allocate the label must not run at all.
        assert!(!enabled());
        assert_eq!(label(|| panic!("must not be called")), "");
    }
}
