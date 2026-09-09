//! Research and diagnostic overrides that widen what the planner admits.
//!
//! These are the only environment reads on the planning path. Each one
//! lifts a capture-backing limit rather than a hardware limit: the planner
//! will then plan a configuration that has not been measured, which is the
//! point of the override and the reason it is not the default. The compiler
//! and the runtime read the same variables, so a model built under one has
//! to run under the same one. A structured `PlanningPolicy` argument is the
//! intended replacement (COMPILER_ROADMAP.md section 1). [`PlanningPolicy`]
//! is the first step: the environment is still the default, but a caller
//! that holds an explicit policy -- the compiler plugin, through the C ABI
//! in `rocket-plan-ffi` -- runs the planner under [`with_policy`] and the
//! environment is not consulted at all. Threading the policy through every
//! planner signature instead of a thread-local is the step after this one.

use std::cell::Cell;

/// What the planner may admit beyond its capture backing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PlanningPolicy {
    /// [`unbacked_channels_allowed`].
    pub allow_unbacked_channels: bool,
    /// [`large_kernel_probing_allowed`].
    pub allow_large_kernel_probing: bool,
}

thread_local! {
    static OVERRIDE: Cell<Option<PlanningPolicy>> = const { Cell::new(None) };
}

impl PlanningPolicy {
    /// The policy the environment variables spell: what every in-process
    /// caller has always got.
    pub fn from_env() -> PlanningPolicy {
        PlanningPolicy {
            allow_unbacked_channels: std::env::var_os("ROCKET_ALLOW_UNBACKED_CHANNELS").is_some(),
            allow_large_kernel_probing: std::env::var_os("ROCKET_ALLOW_LARGE_KERNEL_PROBING")
                .is_some(),
        }
    }

    /// The policy in force on this thread: the [`with_policy`] override if
    /// one is active, the environment otherwise.
    pub fn current() -> PlanningPolicy {
        OVERRIDE
            .with(Cell::get)
            .unwrap_or_else(PlanningPolicy::from_env)
    }
}

/// Runs `f` with `policy` in force on this thread, restoring whatever was in
/// force before -- so an explicit policy from across the C ABI never leaks
/// into the next caller, and nesting behaves.
pub fn with_policy<R>(policy: PlanningPolicy, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<PlanningPolicy>);
    impl Drop for Restore {
        fn drop(&mut self) {
            OVERRIDE.with(|cell| cell.set(self.0));
        }
    }
    let _restore = Restore(OVERRIDE.with(|cell| cell.replace(Some(policy))));
    f()
}

/// Whether `Shape` may be built with more input channels than the capture
/// corpus backs.
///
/// This exists so the CBUF-split scoring harness
/// (`tests/cbuf_split_score.rs`) and the high-channel hardware probes can
/// reach past the cap; without it `Shape` refuses to build and the most
/// interesting part of the vendor corpus is invisible. Nothing on the compiled
/// path sets this.
///
/// It lifts **both** channel ceilings. It used to lift only the input one,
/// which made a whole class of shape unreachable for characterization:
/// MobileNetV2's widest dense 1x1 is `Cin` 448 -> `Cout` 1792, and no probe
/// could construct it to find out whether the `Cout` ceiling was a real limit
/// or just the extent of the measurement. It was the latter.
pub fn unbacked_channels_allowed() -> bool {
    PlanningPolicy::current().allow_unbacked_channels
}

/// Lifts [`check_large_kernel_plan_case`] entirely, so a probe can build the
/// shapes above 3x3 that the refusals exist to keep off hardware.
///
/// The `Cin` cliff hangs the NPU and the watchdog kills the job, so a sweep
/// that walks past a ceiling will leave the device sick -- see the wedge
/// protocol in `dtype_boundary_probe`. Nothing on the compiled path sets
/// this; it is the counterpart of [`unbacked_channels_allowed`] for kernel
/// size rather than channel count.
pub fn large_kernel_probing_allowed() -> bool {
    PlanningPolicy::current().allow_large_kernel_probing
}
