//! Element-wise (EW) tensor-tensor ops on the DPU's EW/ERDMA block.
//!
//! Currently one shape: Mesa's `add_tensor`, fused onto a convolution's own
//! DPU pass ([`AddTensor`], [`build_conv_with_add_regcmd`]). The EW block
//! is otherwise fully bypassed by every other builder in this crate.
//!
//! The register recipe here is bit-exact confirmed against a live capture
//! of a standalone `x + y` model; what is *not* confirmed is the data flow
//! around it -- see [`build_conv_with_add_regcmd`]'s doc comment for the
//! one-task-vs-two-task gap, which is the first thing to suspect if this
//! path misbehaves on hardware.
//!
//! Mesa-derived, like [`crate::rocket::mesa_conv`]. A standalone
//! element-wise op (EW without a producing conv) has no builder yet; it
//! belongs in this module when one is derived.

use crate::rocket::{
    builders::RegCmd,
    mesa_conv::{
        ConvBuffers, ConvShape, build_conv_cna_core_dpu_dpu_rdma, require_single_conv_task,
    },
    regcmd::{KICK_CNA, KICK_CORE, KICK_DPU, KICK_DPU_RDMA, push_kick},
};

/// `EW_CVT_SCALE_VALUE`'s packed `(scale, shift)` for `AddTensor` --
/// the magnitude computation is ported directly from Mesa's
/// `rkt_regcmd.c` (`float add_scale = operation->addition_scale /
/// (task->input_scale * task->weights_scale); ... add_shift =
/// 127+31-32-(add_scale_bits>>23)+16; scale = (add_scale_bits>>9) &
/// 0x7fff; if (scale < 1<<14) scale |= 1<<14;`) -- the same float-bits
/// mantissa/exponent split `build_conv_cna_core_dpu_dpu_rdma`'s own
/// `out_scale`/`out_shift` already uses, just a different formula input
/// and NO `+ 1` on the scale (unlike `out_scale`).
///
/// **The sign handling below is NOT in Mesa's C** -- literally as
/// written, Mesa's formula masks `add_scale_bits` down to 15 bits
/// (`& 0x7fff`), which discards the float's sign bit entirely and
/// produces the IDENTICAL scale for `+1.0` and `-1.0`. Mesa never
/// needed a negative `addition_scale` (its only real use is a residual
/// add, always positive), so this was presumably just never exercised
/// upstream, not a deliberate design. Comparing a live hardware capture
/// of a standalone `x + y` model against a standalone `x - y` model
/// (same shape, same EW_CFG/ew_alu_algo=2, only `EW_CVT_SCALE_VALUE`
/// differs: `0x4000` vs `0xc000`) showed the real vendor compiler DOES
/// distinguish them -- `0xc000`, read as a signed 16-bit value, is
/// exactly `-0x4000` -- i.e. the real hardware field is a signed
/// fixed-point scale, and subtraction is implemented as this exact
/// same Add-fusion path with a negated scale, not a different
/// `ew_alu_algo` opcode. Two's-complement-negating the magnitude
/// result when `addition_scale` is negative reproduces the real
/// captured `0xc000` bit-exact (see `rknpu-spelunking/NOTES.md`'s
/// "Elementwise tensor-tensor ops" section).
pub(crate) fn ew_add_cvt(addition_scale: f32, input_scale: f32, weights_scale: f32) -> (u32, u32) {
    let add_scale = addition_scale.abs() / (input_scale * weights_scale);
    let add_scale_bits = add_scale.to_bits();
    let add_scale_exponent = ((add_scale_bits >> 23) & 0xff) as i32;
    let add_shift = 127 + 31 - 32 - add_scale_exponent + 16;
    let mut scale = (add_scale_bits >> 9) & 0x7fff;
    if scale < (1 << 14) {
        scale |= 1 << 14;
    }
    if addition_scale < 0.0 {
        scale = scale.wrapping_neg() & 0xffff;
    }
    (scale, (add_shift - 1) as u32)
}

/// Element-wise "add_tensor" fusion onto a conv's own DPU pass -- Mesa's
/// `rkt_regcmd.c` `operation->add_tensor != -1` branch, ported directly
/// from that C source (see `build_conv_cna_core_dpu_dpu_rdma`'s doc
/// comment for exactly which fields are hardware-confirmed vs. taken
/// as-is from Mesa without independent re-derivation). Phase 5's EW-core
/// research spike found this dispatch's real register values match a
/// live hardware capture of a standalone `x + y` model bit-exact
/// (`rknpu-spelunking/NOTES.md`'s "Elementwise tensor-tensor ops"
/// section) -- `ew_alu_algo=2` (Add) is the confirmed opcode; the TRM
/// documents `3=Div, 4=Minus, 0=Max, 1=Min, 5=Abs, 6=Neg, 7=Floor,
/// 8=Ceil` as the same field's other values.
///
/// **This same struct also covers subtraction** -- comparing a static
/// decode of a standalone `x + y` model against a standalone `x - y`
/// model showed the real vendor compiler does NOT use a different
/// `ew_alu_algo` for subtract; it reuses this exact `algo=2` Add fusion
/// with a NEGATED `scale` (`EW_CVT_SCALE_VALUE` was `0x4000` for `+1.0`
/// vs. `0xc000` -- exactly `-0x4000` read as signed 16-bit -- for
/// `-1.0`). Pass a negative `scale` here for `x - y`; see
/// `ew_add_cvt`'s doc comment for the sign-handling fix this needed
/// (Mesa's own C silently discards the sign, since it never uses a
/// negative `addition_scale` itself). `x_real - y_real` (rather than
/// `y_real - x_real`) is achieved by negating `scale`, not by swapping
/// which tensor is the primary conv input vs. the fused operand.
#[derive(Clone, Copy)]
pub struct AddTensor {
    /// Real DMA address `DPU_RDMA_RDMA_SRC_BASE_ADDR` reads from. Mesa's
    /// own C computes this as the add_tensor's own `phys_addr +
    /// task->output_offset` -- NOT simply "the conv's usual primary
    /// input" (a plain conv's DPU pass is pipelined on-chip from
    /// CNA/CORE and never uses this register at all -- it reads 0 in
    /// every other builder in this crate; it's only ever nonzero in
    /// this fused case, confirmed in the live capture too). Exposed here
    /// as a plain resolved address like every other `*_addr` field in
    /// this crate's buffer structs, rather than replicating Mesa's own
    /// internal tensor/subgraph offset arithmetic.
    pub src_addr: u32,
    /// Real DMA address `DPU_RDMA_RDMA_EW_BASE_ADDR`/ERDMA reads from --
    /// the actual second operand EW's ALU adds to the conv's own
    /// accumulator. In Mesa's own C this is the SAME tensor as
    /// `src_addr` above, offset by one whole output plane
    /// (`output_width * output_height * ATOMIC_K_SIZE` bytes) -- a
    /// caller that wants to match Mesa's own buffer layout can replicate
    /// that itself; this crate doesn't assume or enforce the
    /// relationship between the two addresses.
    pub ew_addr: u32,
    /// The second tensor's own quantization scale. `EW_CVT_SCALE_VALUE`'s
    /// packed shift/scale is `addition_scale / (input_scale *
    /// weights_scale)`, ported directly from Mesa's C formula (not
    /// reverse-engineered) -- confirmed bit-exact against a real
    /// hardware capture for one calibration.
    pub scale: f32,
    /// Raw value written to `EW_CVT_OFFSET_VALUE` verbatim. Mesa's own
    /// `operation->addition_offset` is used as-is in its C source with
    /// no zero-point-based formula shown deriving it -- this crate does
    /// the same rather than guessing one. UNCONFIRMED what real-world
    /// zero point this should be derived from in general; the one real
    /// capture behind this code needed `1` for its own particular
    /// calibration.
    pub cvt_offset: u32,
    /// Raw `ew_alu_algo` opcode (bits 19:16 of `EW_CFG`) -- the TRM
    /// documents `0=Max, 1=Min, 2=Add, 3=Div, 4=Minus, 5=Abs, 6=Neg,
    /// 7=Floor, 8=Ceil` for this field. Only `2` (Add, with `scale`'s
    /// sign giving Add vs. Subtract -- see this struct's own doc
    /// comment) is hardware-confirmed correct end-to-end (register
    /// recipe AND live numeric output). Other values, including `3`
    /// (Div), are driven directly from the TRM's documented opcode list
    /// as a genuine hardware experiment -- no compiled model was ever
    /// observed emitting them (a standalone `x / y` ONNX export compiled
    /// with NO active EW/BN/BS-mul dispatch anywhere in the file, i.e.
    /// the real vendor compiler doesn't appear to route division through
    /// this opcode at all, for reasons not otherwise understood -- see
    /// `rknpu-spelunking/NOTES.md`'s "Elementwise tensor-tensor ops"
    /// section), so `ew_add_cvt`'s scale/shift formula (derived from
    /// Mesa's ADD-specific math) is NOT known to apply for these other
    /// opcodes at all -- treat any non-`2` result as exploratory.
    pub algo: u32,
}

/// A single conv task with Mesa's `add_tensor` element-wise-add fused
/// onto its own DPU pass -- see `AddTensor`'s doc comment for the full
/// derivation (TRM's documented `ew_alu_algo` opcodes, Mesa's
/// `rkt_regcmd.c` source this is ported from, and the live hardware
/// capture of a standalone `x + y` model that confirmed the resulting
/// register values bit-exact). Same single task/kick shape as
/// `build_conv_regcmd` -- `addition` only changes what the DPU pass's
/// own EW/ERDMA block does, not the task structure around it.
///
/// NOT YET HARDWARE-VALIDATED THROUGH THIS EXACT FUNCTION: the real
/// captured model that confirmed `AddTensor`'s register recipe compiled
/// to a 3-task chain (a real conv-shaped task producing an intermediate,
/// paired with a second task doing the actual EW-add dispatch), not the
/// single conv-with-fused-EW task this function builds -- the data-flow
/// difference between "one task, real primary input direct to DPU_RDMA/
/// ERDMA" (what this function assumes, mirroring `build_lut_regcmd`'s
/// existing single-task shape) and "two tasks, one producing an
/// intermediate the other reads" (what the one real capture actually
/// showed) has NOT been independently distinguished by hardware testing
/// -- see `rknpu-spelunking/NOTES.md`'s "Elementwise tensor-tensor ops"
/// section. If this hangs or produces wrong output on real hardware,
/// suspect that gap first, not the `EW_CFG`/`EW_CVT_SCALE_VALUE` values
/// themselves (those parts are bit-exact confirmed).
pub fn build_conv_with_add_regcmd(
    shape: &ConvShape,
    bufs: &ConvBuffers,
    addition: &AddTensor,
) -> Vec<RegCmd> {
    let task = require_single_conv_task(shape);
    let mut cmds = build_conv_cna_core_dpu_dpu_rdma(shape, bufs, &task, 2, Some(addition));
    push_kick(&mut cmds, KICK_CNA | KICK_CORE | KICK_DPU | KICK_DPU_RDMA);
    cmds
}
