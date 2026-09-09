//! Research and diagnostic overrides that widen what the planner admits.
//!
//! These are the only environment reads on the planning path. Each one
//! lifts a capture-backing limit rather than a hardware limit: the planner
//! will then plan a configuration that has not been measured, which is the
//! point of the override and the reason it is not the default. The compiler
//! and the runtime read the same variables, so a model built under one has
//! to run under the same one. A structured `PlanningPolicy` argument is the
//! intended replacement (COMPILER_ROADMAP.md section 1); until then these
//! two functions are the whole policy surface.

/// Feature atoms the CBUF charges per `data_entries` entry.
///
/// The surface feature charge is counted in whole entries of four atoms, and
/// it rounds *up*: a row whose atom count is not a multiple of four still
/// occupies the whole final entry. `CNA_CBUF_CON1.data_entries` has always
/// been programmed that way (see its `div_ceil` below); the residency bound
/// in [`Shape::max_tile_input_rows_for_width_and_data_banks`] has to charge
/// the same way or it over-commits the CBUF.
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
    std::env::var_os("ROCKET_ALLOW_UNBACKED_CHANNELS").is_some()
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
    std::env::var_os("ROCKET_ALLOW_LARGE_KERNEL_PROBING").is_some()
}
