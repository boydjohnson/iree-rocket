//! Runs the driver's host-side work on the CPUs that are good at it.
//!
//! Every Rocket dispatch pays a host-side layout bridge -- pack the input
//! NHWC -> NC1HWC2, compact the DPU's atomic-slot output back to dense -- and
//! those transforms are pure memory movement. On a big.LITTLE part that makes
//! the choice of core worth more than any amount of tuning inside the loops.
//! Measured on planck (RK3588, 4x A55 + 4x A76) with
//! `iree-rocket-hal/examples/layout_bench`, MobileNetV2 fp16's full set of
//! shapes, cold caches:
//!
//! ```text
//!   A76 (cpu4-7)   pack  4.1 ms   compact  9.7 ms    13.8 ms
//!   A55 (cpu0-3)   pack 12.2 ms   compact 40.2 ms    52.4 ms
//! ```
//!
//! 3.8x, for the same code on the same data. And `ROCKET_PROFILE`'s
//! `host time by cpu` line showed the driver was spending 59% of its host
//! time on cpu0-3 -- the slow half -- which is what the A55 column costs in
//! a real inference. The whole process pinned to the big cluster ran
//! MobileNetV2 fp16 at 129 ms against 208 ms unpinned.
//!
//! So the transforms ask the scheduler for the big cluster while they run,
//! and give the thread's original affinity back afterwards. Asking rather
//! than keeping matters: on the fast path `queue_execute`'s work runs on
//! IREE's own calling thread (`run_after_wait`), and permanently narrowing
//! a thread this driver does not own would change scheduling for work that
//! has nothing to do with Rocket.
//!
//! "Big" is read from `cpu_capacity`, the scheduler's own normalized
//! capacity for each CPU, rather than from a hardcoded RK3588 core map. A
//! uniform machine reports one capacity for everything, this finds nothing
//! to prefer, and every entry point below becomes a no-op.
//!
//! Applied once per `queue_execute`, which covers every submit-time phase.
//! Record time (`command_buffer::dispatch`, ~14 ms per MobileNetV2 fp16
//! inference) is not covered: a guard per `dispatch` call measured as noise
//! (144/148 ms against 146/146 ms), because 37 back-to-back set/restore pairs
//! migrate the thread off the big cluster again between every one of them.
//! Holding one guard across a whole command buffer's recording would be the
//! way to get that 14 ms, and it needs state on the command buffer rather
//! than a local.
//!
//! ```text
//! ROCKET_HOST_CPUS=off     never change affinity
//! ROCKET_HOST_CPUS=0-3,7   use this CPU list instead of the highest-capacity one
//! ```

use nix::{
    sched::{CpuSet, sched_getaffinity, sched_setaffinity},
    unistd::Pid,
};

/// The CPUs the host-side transforms should run on, or `None` when there is
/// nothing to prefer (a uniform machine, an unreadable `cpu_capacity`, or
/// `ROCKET_HOST_CPUS=off`).
fn preferred() -> Option<&'static CpuSet> {
    static PREFERRED: std::sync::OnceLock<Option<CpuSet>> = std::sync::OnceLock::new();
    PREFERRED
        .get_or_init(|| match std::env::var("ROCKET_HOST_CPUS") {
            Ok(value) if value == "off" => None,
            Ok(value) => parse_cpu_list(&value),
            Err(_) => highest_capacity_cpus(),
        })
        .as_ref()
}

/// Parses a `0-3,7` style CPU list, the same spelling `taskset -c` takes.
fn parse_cpu_list(list: &str) -> Option<CpuSet> {
    let mut set = CpuSet::new();
    let mut any = false;
    for part in list.split(',').filter(|part| !part.is_empty()) {
        let (first, last) = match part.split_once('-') {
            Some((first, last)) => (first.trim().parse().ok()?, last.trim().parse().ok()?),
            None => {
                let only: usize = part.trim().parse().ok()?;
                (only, only)
            }
        };
        for cpu in first..=last {
            set.set(cpu).ok()?;
            any = true;
        }
    }
    any.then_some(set)
}

/// Every CPU sharing the machine's highest `cpu_capacity`.
///
/// `cpu_capacity` is the scheduler's normalized compute capacity (1024 for
/// the fastest core in the system). It is absent on uniform machines and on
/// kernels without the energy model, which is the same answer as "nothing to
/// prefer" -- and it is also identical across CPUs on a machine whose cores
/// really are all the same, so a big.LITTLE assumption never leaks onto a
/// machine that is not one.
fn highest_capacity_cpus() -> Option<CpuSet> {
    let mut capacities = Vec::new();
    for cpu in 0..CpuSet::count() {
        let path = format!("/sys/devices/system/cpu/cpu{cpu}/cpu_capacity");
        let Ok(text) = std::fs::read_to_string(&path) else {
            // CPUs are numbered contiguously; the first miss is the end of
            // the list (or a kernel that does not publish capacities at all).
            break;
        };
        let Ok(capacity) = text.trim().parse::<u64>() else {
            return None;
        };
        capacities.push((cpu, capacity));
    }
    let highest = capacities.iter().map(|(_, c)| *c).max()?;
    if capacities.iter().all(|(_, c)| *c == highest) {
        // Uniform: every CPU is equally good, so there is nothing to ask for.
        return None;
    }
    let mut set = CpuSet::new();
    for (cpu, _) in capacities.iter().filter(|(_, c)| *c == highest) {
        set.set(*cpu).ok()?;
    }
    Some(set)
}

/// Restores the calling thread's affinity when dropped.
///
/// Constructed by [`prefer_fast_cpus`]. Holding one across a whole
/// `queue_execute` rather than around each transform keeps it to two syscalls
/// per submission instead of two per phase, and lets the thread settle on its
/// new CPU instead of being migrated back and forth between them.
pub struct Restore(Option<CpuSet>);

impl Drop for Restore {
    fn drop(&mut self) {
        if let Some(previous) = self.0.take() {
            let _ = sched_setaffinity(Pid::from_raw(0), &previous);
        }
    }
}

/// Asks to run on the highest-capacity CPUs until the returned guard drops.
///
/// A no-op -- and no syscalls -- when there is nothing to prefer, or when the
/// thread is already confined to a subset of the preferred CPUs, which is the
/// case for a caller that has already pinned itself (`taskset`, or IREE's own
/// task-topology flags). Deliberately quiet on failure: a restricted cpuset,
/// a container, or a kernel that refuses the call all mean "run where you
/// are", not "fail this dispatch".
pub fn prefer_fast_cpus() -> Restore {
    let Some(preferred) = preferred() else {
        return Restore(None);
    };
    let Ok(current) = sched_getaffinity(Pid::from_raw(0)) else {
        return Restore(None);
    };
    if is_subset(&current, preferred) {
        return Restore(None);
    }
    // Only the CPUs the caller is already allowed to use: a thread confined
    // to cpu0-3 by a cpuset must not be handed cpu4-7 by this.
    let mut wanted = CpuSet::new();
    let mut any = false;
    for cpu in 0..CpuSet::count() {
        if preferred.is_set(cpu).unwrap_or(false) && current.is_set(cpu).unwrap_or(false) {
            if wanted.set(cpu).is_err() {
                return Restore(None);
            }
            any = true;
        }
    }
    if !any || sched_setaffinity(Pid::from_raw(0), &wanted).is_err() {
        return Restore(None);
    }
    Restore(Some(current))
}

fn is_subset(inner: &CpuSet, outer: &CpuSet) -> bool {
    (0..CpuSet::count())
        .all(|cpu| !inner.is_set(cpu).unwrap_or(false) || outer.is_set(cpu).unwrap_or(false))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_taskset_style_cpu_lists() {
        let set = parse_cpu_list("0-3,7").expect("a valid list");
        for cpu in 0..=3 {
            assert!(set.is_set(cpu).unwrap());
        }
        assert!(!set.is_set(4).unwrap());
        assert!(set.is_set(7).unwrap());
    }

    #[test]
    fn rejects_lists_that_name_no_cpu() {
        assert!(parse_cpu_list("").is_none());
        assert!(parse_cpu_list("not-a-cpu").is_none());
    }

    /// A thread already confined to a subset of the preferred CPUs is left
    /// alone -- the guard has nothing to restore.
    #[test]
    fn a_subset_is_left_alone() {
        let mut inner = CpuSet::new();
        inner.set(0).unwrap();
        let mut outer = CpuSet::new();
        outer.set(0).unwrap();
        outer.set(1).unwrap();
        assert!(is_subset(&inner, &outer));
        assert!(!is_subset(&outer, &inner));
    }
}
