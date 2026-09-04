//! Telling a watchdog-killed NPU job apart from a wrong result.
//!
//! `PREP_BO` is not a completion check. It waits on the output BO's
//! `dma_resv` fence, and when the RK3588 watchdog gives up on a hung job it
//! resets the core and signals that fence **with an error** -- which is
//! still signalled. The ioctl returns success and the test reads an output
//! buffer the DPU wrote part of, or none of. It looks exactly like a
//! catastrophic layout bug: every element wrong, all of them the sentinel.
//!
//! A wall clock is the only thing that separates the two, and it separates
//! them by two orders of magnitude, so this is a discriminator rather than
//! a heuristic. See [`DISPATCH_TIMEOUT_FLOOR`].
//!
//! Every hardware test that submits its own job wants this. Twenty-six
//! files in this directory do their own `submit_jobs` + `prep_bo`, and
//! without it each one reports a device event as a shape result -- which is
//! how two of six once-recorded "known limitations" got written down.

#![allow(dead_code)]

use std::time::Duration;

/// Dispatch wall time above which a failure is a killed job, not a result.
///
/// Measured on planck 2026-09-03, and the gap is wide in both directions:
///
/// ```text
///   3.13 ms   slowest of MobileNetV2 fp16's 54 real dispatches
///   58.5 ms   slowest across every hardware ladder here (226x226, 28 tiles)
///   ~500 ms   a job the watchdog killed (JOB_TIMEOUT_MS plus a tick)
/// ```
///
/// Dispatch time is *not* proportional to tile count -- 28 large tiles cost
/// more than 112 small ones -- so a flat floor fits better than a per-tile
/// budget. Re-measure with `ROCKET_DISPATCH_TIMES=1` on a shape family
/// larger than anything above rather than assuming the headroom survived.
pub const DISPATCH_TIMEOUT_FLOOR: Duration = Duration::from_millis(150);

/// Whether a dispatch ran long enough to be a killed job.
pub fn is_device_timeout(elapsed: Duration) -> bool {
    elapsed >= DISPATCH_TIMEOUT_FLOOR
}

/// Whether a killed dispatch should fail the run (`ROCKET_STRICT_DISPATCH`).
///
/// Off by default. A killed job says nothing about the code under test, and
/// failing on it turns the gate into noise that stops being read -- worse
/// than the alternative, because the report below is loud and counted. On
/// for a run that must be clean.
pub fn strict() -> bool {
    std::env::var("ROCKET_STRICT_DISPATCH").is_ok_and(|value| value != "0")
}

/// The dispatch time, printed only when asked for
/// (`ROCKET_DISPATCH_TIMES=1`). This is how the headroom under
/// [`DISPATCH_TIMEOUT_FLOOR`] gets re-measured instead of trusted.
pub fn note(elapsed: Duration) -> String {
    if std::env::var("ROCKET_DISPATCH_TIMES").is_ok() {
        format!(" dispatch={:.1}ms", elapsed.as_secs_f64() * 1e3)
    } else {
        String::new()
    }
}

/// Reports a killed dispatch, and says whether the caller must still fail.
///
/// Returns `true` under `ROCKET_STRICT_DISPATCH`; otherwise the caller
/// should treat the case as **not measured** and carry on, exactly as
/// `run_hardware_case_matrix` does.
#[must_use]
pub fn report(label: &str, elapsed: Duration) -> bool {
    println!(
        "  DEVICE TIMEOUT, not a shape result: {label}\n    \
         dispatch took {:.0} ms against a {:.0} ms floor (healthy is milliseconds). The\n    \
         kernel watchdog killed the job and signalled its fence with an error, which\n    \
         PREP_BO reports as success, so the output is partly or wholly unwritten. This\n    \
         case was NOT measured and is excluded from the verdict.\n    \
         Confirm with `sudo dmesg | grep -i npu` (`NPU job timed out`). These cluster when\n    \
         the machine was busy in the preceding second, which is why `cargo nextest`\n    \
         provokes them and a one-second idle gap almost never does.\n    \
         Set ROCKET_STRICT_DISPATCH=1 to make them fail the run instead.",
        elapsed.as_secs_f64() * 1e3,
        DISPATCH_TIMEOUT_FLOOR.as_secs_f64() * 1e3,
    );
    strict()
}
