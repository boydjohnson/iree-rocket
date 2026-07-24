//! Shared, Mesa-faithful regcmd construction for conv operations.
//!
//! Ported field-for-field from `mesa-rocket-userspace/rkt_regcmd.c`'s
//! `fill_first_regcmd()` and `rkt_task.c`'s `fill_task()`/
//! `rkt_split_tasks()` (see rknpu-spelunking/NOTES.md for the full
//! derivation history and hardware validation -- this is what turned
//! `rkt-basic.rs` from a permanently-hanging/wrong-output test into one
//! that dispatches, completes, and produces real, input-tracking output).
//! Originally written directly into `rkt-basic.rs` for that file's one
//! fixed shape; generalized here so `rkt-job.rs`/`rkt-simple-job.rs` can
//! share it instead of each carrying their own independently incomplete
//! regcmd builders.
//!
//! Scope, deliberately not general beyond what this repo currently needs:
//! - Convolutions larger than CBUF are split along output height using
//!   Mesa's `rkt_split_tasks()` algorithm. The public multi-task builder
//!   returns one regcmd buffer per split. After allocating the buffers, callers
//!   patch their DMA addresses with `link_regcmd_tasks` and pass them to
//!   `device::submit_tasks`.
//! - The serialized shape currently has no padding fields, so convolution
//!   planning supports zero-padding valid convolutions only.
//! - No `addition_input`/`add_tensor` support -- none of the three bins
//!   use a second input tensor, so the DPU_RDMA/EW "add_tensor != -1"
//!   branches in Mesa's source are simply not implemented here.
//! - The `input_channels_real == 1 && output_channels_real > 1` "wide
//!   atomic" branch in `fill_task()` isn't implemented either -- no
//!   current client's shape reaches it, and porting an untested branch
//!   with no way to validate it would just be a latent bug. `build()`
//!   asserts this combination isn't requested rather than silently
//!   producing wrong output for it.

use crate::rocket::{
    builders::{
        Bits, DOMAIN_CORE, DOMAIN_DPU, DOMAIN_PC, RegCmd, Register, RegisterMeta, cna::*, core::*,
        dpu::*, dpu_rdma::*, ppu::*, ppu_rdma::*,
    },
    registers::{
        REG_DPU_LUT_ACCESS_DATA, REG_PC_BASE_ADDRESS, REG_PC_OPERATION_ENABLE,
        REG_PC_REGISTER_AMOUNTS,
    },
};

fn zero<R: RegisterMeta>() -> RegCmd {
    Register::<R>::new().build()
}

/// Fused post-processing applied to a conv/FC/pooling op's own accumulator
/// before write-back, mirroring how a real compiler would fuse a
/// `linalg.generic` elementwise op into its producer rather than emit a
/// standalone op -- this hardware has no dedicated activation unit, only a
/// bypassable post-processing stage on the op that already runs (DPU's BS
/// core, for conv/FC; only reachable on pooling paths that have a real DPU
/// stage ahead of PPU).
///
/// Hardware-validated (`iree-rocket-hal/tests/conv_activation_hw.rs`, real
/// RK3588): BS's ALU-then-RELU-then-MUL bit ordering means this applies as
/// `relu(bias_add(x))`, the standard fusion order -- confirmed by `Relu`
/// never decreasing output vs. `None` across a fill sweep, and `Relux`'s
/// independent upper clamp confirmed by `cmp: 0` forcing constant output
/// regardless of input (domain-independent, since 0 clamps in any scale).
/// Still open: what real-world numeric domain a non-zero `cmp` needs to be
/// in for a given quantization scale -- `cmp` values of 1000/3000 showed no
/// visible effect across accumulator magnitudes reached by uniform-fill
/// inputs 100..140 at this test's placeholder scale=1.0, so a useful `cmp`
/// for a real (calibrated) shape needs deriving from that shape's own
/// scale/shift math, not assumed to transfer from this test's constants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    None,
    Relu,
    /// Clamps to `cmp` in addition to the zero floor (`DPU_BS_RELUX_CMP_VALUE`).
    Relux {
        cmp: u32,
    },
}

/// Which numeric domain a conv op computes in. Hardware-validated:
/// `Int8` by essentially every hw test in this repo; `Fp16` by
/// `rkt-fp16.rs`'s round-7 recipe (rknpu-spelunking/NOTES.md's
/// "Attempted real fp16 dispatch on hardware" section through its
/// resolution) -- a real, non-uniform weight/input pair (`0.25 * 10.5`)
/// computing a bit-exact correct product (`2.625`) on real hardware.
///
/// `Fp16` is only wired up for the `input_channels == 1` path below --
/// every fp16 hardware test used that shape, and the multi-channel
/// branch's CVT convention (already its own distinct code path for
/// int8) was never independently re-derived or validated for fp16.
/// `build_conv_cna_core_dpu_dpu_rdma` asserts this rather than silently
/// emitting untested output for it, matching this file's existing
/// convention for other untested shape combinations (see the module doc
/// comment's "wide atomic" note).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    Int8,
    Fp16,
}

impl Precision {
    /// Bytes per element -- scales the byte-footprint-only fields
    /// (`CNA_WEIGHT_SIZE0/1`) that Mesa's original `fill_task()` formula
    /// assumes are always 1 byte/element. Flagged as an open question in
    /// early fp16 rounds ("not patched or ruled out"), later confirmed
    /// load-bearing once a real, non-uniform weight/input pair exposed
    /// it (a uniformly-filled weight buffer can't distinguish a wrong
    /// byte count from a correct one).
    pub fn bytes_per_element(self) -> u32 {
        match self {
            Precision::Int8 => 1,
            Precision::Fp16 => 2,
        }
    }

    /// The shared 3-bit precision enum `CNA_CONV_CON1`/`CORE_MISC_CFG`
    /// use, which also matches `DPU_DATA_FORMAT`/`DPU_RDMA_FEATURE_
    /// MODE_CFG`'s own wider 6-value enum for every value in use here
    /// (int8=0, fp16=2 in both) -- per TRM chapter36.txt.
    fn enum_value(self) -> u32 {
        match self {
            Precision::Int8 => 0,
            Precision::Fp16 => 2,
        }
    }
}

/// A DPU LUT (piecewise lookup table) configuration, as used by
/// `build_lut_regcmd`. Confirmed via `rknpu-spelunking/NOTES.md`'s
/// "Decoding the DPU LUT block: sigmoid/tanh regcmd capture" section --
/// exported standalone `nn.Sigmoid()`/`nn.Tanh()` ONNX models through
/// `rknn-convert` (real `rknn-toolkit2` under the hood) and decoded the
/// compiled `.rknn`'s regcmd blob directly, cross-validating the recipe
/// across two independent activation functions.
///
/// The domain/indexing recipe is compiler-generic, confirmed byte-
/// identical between the sigmoid and tanh captures, and is therefore
/// hardcoded inside the builder rather than exposed here (same
/// risk-avoidance choice as `FC_SAFE_HEIGHT` in the FC section below --
/// narrow the API to only what's actually been observed): the LUT domain
/// is always split at `x=0` into `LE = [-16384, 0)` and `LO = [0, 16384]`,
/// each table addressed as `index = (x - region_start) >> 5`, giving
/// exactly 513 entries per table (both known-good captures agree). Only
/// the table contents, EW op bypass state, and the LE-side underflow
/// extrapolation slope vary
/// per function -- `lut_tables::SIGMOID_LE`/`SIGMOID_LO`/`TANH_LE`/
/// `TANH_LO` are the exact captured vendor data, not reimplemented math.
#[derive(Clone, Copy, Debug)]
pub struct LutTable {
    /// 513 raw 16-bit table entries for the `x < 0` half.
    pub le_entries: &'static [u16; 513],
    /// 513 raw 16-bit table entries for the `x >= 0` half.
    pub lo_entries: &'static [u16; 513],
    /// `DPU_EW_CFG.EW_OP_BYPASS` captured for this specific function.
    /// Sigmoid uses `EW_CFG=0x300`; tanh uses `EW_CFG=0x302`.
    pub ew_op_bypass: u8,
    /// Extrapolation slope for `x` below `-16384` (`LUT_LE_SLOPE_SCALE`/
    /// `_SHIFT`'s `UFLOW` sub-fields). Zero in the tanh capture (flat
    /// clamp); nonzero in sigmoid's (a small residual slope, consistent
    /// with sigmoid's tail not being fully flat even far from zero).
    pub le_slope_uflow_scale: u16,
    pub le_slope_uflow_shift: u8,
}

impl LutTable {
    pub fn sigmoid() -> Self {
        LutTable {
            le_entries: &crate::rocket::lut_tables::SIGMOID_LE,
            lo_entries: &crate::rocket::lut_tables::SIGMOID_LO,
            ew_op_bypass: 0,
            le_slope_uflow_scale: 23107,
            le_slope_uflow_shift: 22,
        }
    }

    pub fn tanh() -> Self {
        LutTable {
            le_entries: &crate::rocket::lut_tables::TANH_LE,
            lo_entries: &crate::rocket::lut_tables::TANH_LO,
            ew_op_bypass: 1,
            le_slope_uflow_scale: 0,
            le_slope_uflow_shift: 0,
        }
    }

    /// Softmax's exp(x) step (Phase 5 of the ukernel roadmap), routed
    /// through this exact same standalone DPU LUT path -- confirmed via
    /// live hardware trace of the vendor runtime, byte-identical recipe to
    /// `sigmoid()`/`tanh()` above (`rknpu-spelunking/NOTES.md`'s softmax
    /// Phase 5 "Follow-up 4" section): same domain split, same 513-entry
    /// indexing, same `EW_CFG` field layout. Two things are specific to
    /// exp and confirmed only from this one real capture (not fit across
    /// multiple scales/zero-points the way `sigmoid()`/`tanh()`'s tables
    /// were cross-validated against each other):
    /// - `le_slope_uflow_scale`/`_shift` = 0 (flat clamp below `x=-16384`,
    ///   same choice as `tanh()`, confirmed by direct register capture,
    ///   not inferred by analogy).
    /// - `EXP_LO` (the `x >= 0` half) is a placeholder, not real captured
    ///   data -- see its own doc comment in `lut_tables.rs`. Softmax's
    ///   max-subtraction guarantees this half is never legitimately read,
    ///   but a caller building a *different* op around this table (not
    ///   softmax's own max-subtract-then-exp pattern) would get an
    ///   unvalidated constant `1.0` for any `x >= 0` input.
    ///
    /// One piece of good, independent supporting evidence this recipe
    /// generalizes correctly: the real captured `BN_MUL_CFG=0x6a660000`
    /// for this op backs out to `input_scale ~= 10.49` via the *existing*
    /// `lut_bn_mul()`/`LUT_BN_SCALE_K=2596.513` formula (fit from
    /// sigmoid/tanh captures, not exp) -- i.e. `build_lut_regcmd`'s
    /// generic `input_scale`/`input_zero_point` handling doesn't need any
    /// exp-specific change, only this table.
    pub fn exp() -> Self {
        LutTable {
            le_entries: &crate::rocket::lut_tables::EXP_LE,
            lo_entries: &crate::rocket::lut_tables::EXP_LO,
            ew_op_bypass: 1,
            le_slope_uflow_scale: 0,
            le_slope_uflow_shift: 0,
        }
    }
}

/// Writes one LUT table bank via the auto-incrementing `LUT_ACCESS_DATA`
/// address port: write `LUT_ACCESS_CFG` exactly once (write mode,
/// `table_id`, addr 0), then one `LUT_ACCESS_DATA` write per entry -- no
/// separate address register write between entries, confirmed by the
/// vendor capture never interleaving one (see `LutTable`'s doc comment).
fn push_lut_table(cmds: &mut Vec<RegCmd>, table_id: u32, entries: &[u16; 513]) {
    cmds.push(
        Register::<DpuLutAccessCfg>::new()
            .lut_access_type(Bits::new(1))
            .lut_table_id(Bits::new(table_id))
            .lut_addr(Bits::new(0))
            .build(),
    );
    for &entry in entries.iter() {
        let data = entry as i16 as i32 as u32;
        cmds.push(RegCmd::new(DOMAIN_DPU, REG_DPU_LUT_ACCESS_DATA, data));
    }
}

fn push_lut_tables_and_config(cmds: &mut Vec<RegCmd>, table: LutTable) {
    push_lut_table(cmds, 0, table.le_entries);
    push_lut_table(cmds, 1, table.lo_entries);
    cmds.push(
        Register::<DpuLutCfg>::new()
            .lut_hybrid_priority(Bits::new(1))
            .lut_oflow_priority(Bits::new(1))
            .lut_lo_le_mux(Bits::new(2))
            .build(),
    );
    cmds.push(
        Register::<DpuLutInfo>::new()
            .lut_le_index_select(Bits::new(5))
            .lut_lo_index_select(Bits::new(5))
            .build(),
    );
    cmds.push(
        Register::<DpuLutLeStart>::new()
            .lut_le_start(Bits::new((-16384i32) as u32))
            .build(),
    );
    cmds.push(
        Register::<DpuLutLeEnd>::new()
            .lut_le_end(Bits::new(0))
            .build(),
    );
    cmds.push(
        Register::<DpuLutLoStart>::new()
            .lut_lo_start(Bits::new(0))
            .build(),
    );
    cmds.push(
        Register::<DpuLutLoEnd>::new()
            .lut_lo_end(Bits::new(16384))
            .build(),
    );
    cmds.push(
        Register::<DpuLutLeSlopeScale>::new()
            .lut_le_slope_uflow_scale(Bits::new(table.le_slope_uflow_scale as u32))
            .build(),
    );
    cmds.push(
        Register::<DpuLutLeSlopeShift>::new()
            .lut_le_slope_uflow_shift(Bits::new(table.le_slope_uflow_shift as u32))
            .build(),
    );
    cmds.push(zero::<DpuLutLoSlopeScale>());
    cmds.push(zero::<DpuLutLoSlopeShift>());
}

/// Empirically-fit constant relating a shape's `input_scale` to DPU BN's
/// multiply stage in `build_lut_regcmd` -- reverse-engineered from
/// 5 independent int8-quantized `rknn-toolkit2` sigmoid captures at
/// different calibration scales (see `lut_bn_mul`'s doc comment and
/// `rknpu-spelunking/NOTES.md`). Not a documented hardware constant.
const LUT_BN_SCALE_K: f32 = 2596.513;

/// `BN_MUL_OPERAND`/`BN_MUL_SHIFT` for `build_lut_regcmd`:
/// `multiplier = input_scale * LUT_BN_SCALE_K`, normalized (standard
/// mantissa/exponent split) so `operand = round(multiplier * 2^shift)`
/// lands in `[16384, 32768)` -- reconstructs the captured register values
/// exactly across all 5 known data points (see the call site's doc
/// comment for which zero points were used and the caveat on
/// `lut_bn_alu`).
fn lut_bn_mul(input_scale: f32) -> (u32, u32) {
    let multiplier = input_scale * LUT_BN_SCALE_K;
    let e = multiplier.log2().floor() as i32;
    let shift = 14 - e;
    let operand = (multiplier * 2f32.powi(shift)).round() as u32;
    (operand, shift as u32)
}

/// `BN_ALU_OPERAND` for `build_lut_regcmd`: `-real_zero_point *
/// bn_mul_operand`. Exact for 3 of 5 known data points; an unresolved ~1/16
/// discrepancy showed up for the other 2 (both `real_zero_point == 42`) --
/// see the call site's doc comment. Always exactly right when
/// `real_zero_point == 0`.
fn lut_bn_alu(real_zero_point: i32, bn_mul_operand: u32) -> u32 {
    let alu = -(real_zero_point as i64) * (bn_mul_operand as i64);
    alu as i32 as u32
}

fn lut_bn_alu_supports_zero_point(real_zero_point: i32) -> bool {
    // Exact against the current captured data set for -128, -2, and 127.
    // Zero is exact by construction because the ALU operand is 0 regardless
    // of the multiply operand. Known bad captures exist for 42.
    matches!(real_zero_point, -128 | -2 | 0 | 127)
}

/// `OUT_CVT_SCALE`/`OUT_CVT_SHIFT` for `build_lut_regcmd`.
/// `build_conv_cna_core_dpu_dpu_rdma`'s normal `out_scale`/`out_shift`
/// computation targets a raw conv accumulator's magnitude (via
/// `conv_scale = input_scale*weights_scale/output_scale`) -- the wrong
/// domain entirely for a LUT table's own already-normalized output
/// (`lut_tables.rs`'s entries represent `real_value * 32768`, a fixed
/// Q15-like encoding independent of this op's quantization params).
/// Reverse-engineered from the same 4 int8-quantized captures used for
/// `lut_bn_mul`: `multiplier = 1.0 / (32768.0 * output_scale)`, normalized
/// the same way (`shift = 14 - floor(log2(multiplier))`, `operand =
/// round(multiplier * 2^shift)`) -- reconstructs all 4 captured
/// `OUT_CVT_SCALE` values closely (all `OUT_CVT_SHIFT=21` in this
/// output_scale range, itself a consequence of the shared normalization
/// formula rather than a separately-confirmed constant).
fn lut_out_cvt(output_scale: f32) -> (u32, u32) {
    let multiplier = 1.0 / (32768.0 * output_scale);
    let e = multiplier.log2().floor() as i32;
    let shift = 14 - e;
    let operand = (multiplier * 2f32.powi(shift)).round() as u32;
    (operand, shift as u32)
}

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
fn ew_add_cvt(addition_scale: f32, input_scale: f32, weights_scale: f32) -> (u32, u32) {
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
    /// every other builder in this module; it's only ever nonzero in
    /// this fused case, confirmed in the live capture too). Exposed here
    /// as a plain resolved address like every other `*_addr` field in
    /// this module, rather than replicating Mesa's own internal
    /// tensor/subgraph offset arithmetic.
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

/// Logical shape of a single conv operation (`operation->*` in Mesa).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConvShape {
    pub input_width: u32,
    pub input_height: u32,
    pub input_channels: u32,
    pub output_width: u32,
    pub output_height: u32,
    pub output_channels: u32,
    pub weights_width: u32,
    pub weights_height: u32,
    pub stride: u32,
    pub depthwise: bool,
    pub input_zero_point: u32,
    pub output_zero_point: u32,
    pub weights_zero_point: u32,
    /// Quantization scale factors. None of this repo's clients have real
    /// calibrated tensors -- 1.0 (no rescaling) for all three -- but a
    /// real caller with an actual quantized model would need real values
    /// here, so these are plumbed through rather than hardcoded.
    pub input_scale: f32,
    pub weights_scale: f32,
    pub output_scale: f32,
    pub truncate_bits: u32,
    pub activation: Activation,
    pub precision: Precision,
}

/// DMA addresses for the four buffers a single-task conv op needs.
pub struct ConvBuffers {
    pub input_addr: u32,
    pub weights_addr: u32,
    /// DPU's BS (bias-subtract) block is never actually bypassed by Mesa
    /// -- it always runs its ALU against a real biases buffer, even for
    /// operations with no logical bias (in which case this should just be
    /// zero-filled). See DPU_BS_CFG below.
    pub bias_addr: u32,
    pub output_addr: u32,
}

/// One height split produced by Mesa's `rkt_split_tasks()` policy.
///
/// Offsets use the NPU's 16-byte feature-atomic row layout, matching the
/// register builder's existing `calc_line_stride` port. A caller should
/// normally consume this through [`build_conv_regcmd_tasks`] rather than
/// applying the offsets itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConvTask {
    pub index: u32,
    pub input_top: u32,
    pub input_height: u32,
    pub output_top: u32,
    pub output_height: u32,
    pub input_offset_bytes: u32,
    pub output_offset_bytes: u32,
    pub input_banks: u32,
    pub weights_banks: u32,
    pub overlap_slices: u32,
    pub retain_slices: u32,
    /// Whether this task reuses weights left in CBUF by the preceding task.
    pub reuse_weights: bool,
}

#[derive(Clone, Copy)]
struct ConvTaskDraft {
    top: u32,
    bottom: u32,
    overlap_slices: u32,
    retain_slices: u32,
}

const CBUF_BANKS: u32 = 12;
const CBUF_ENTRIES_PER_BANK: u64 = 256;
const FEATURE_ATOMIC_SIZE: u64 = 16;
const CBUF_ENTRY_SIZE: u64 = 128;
const ATOMIC_K_SIZE: u64 = 16;

fn conv_entries_per_slice(shape: &ConvShape) -> Result<u64, &'static str> {
    let bpe = u64::from(shape.precision.bytes_per_element());
    let input_channels = u64::from(shape.input_channels);
    let input_width = u64::from(shape.input_width);
    let atomics_per_entry = CBUF_ENTRY_SIZE / FEATURE_ATOMIC_SIZE;
    let channel_bytes = input_channels
        .checked_mul(bpe)
        .ok_or("input channel byte count overflows task planning")?;
    let total_c_atomics = channel_bytes.div_ceil(FEATURE_ATOMIC_SIZE);
    let last_c_atomics = total_c_atomics % atomics_per_entry;
    let integer_entries = (total_c_atomics / atomics_per_entry)
        .checked_mul(input_width)
        .ok_or("input entries per slice overflow task planning")?;
    let fractional_entries = if last_c_atomics == 3 {
        input_width
    } else {
        last_c_atomics
            .checked_mul(input_width)
            .ok_or("input entries per slice overflow task planning")?
            .div_ceil(atomics_per_entry)
    };
    integer_entries
        .checked_add(fractional_entries)
        .ok_or("input entries per slice overflow task planning")
}

fn conv_weights_banks(shape: &ConvShape) -> Result<u64, &'static str> {
    let mut bytes = u64::from(shape.weights_width)
        .checked_mul(u64::from(shape.weights_height))
        .and_then(|value| value.checked_mul(u64::from(shape.input_channels)))
        .and_then(|value| value.checked_mul(u64::from(shape.precision.bytes_per_element())))
        .ok_or("weight byte count overflows task planning")?;
    if !shape.depthwise {
        bytes = bytes
            .checked_mul(u64::from(shape.output_channels))
            .ok_or("weight byte count overflows task planning")?;
    }

    // Mesa reserves one extra bank in addition to the rounded weight size.
    let entries = bytes.div_ceil(CBUF_ENTRY_SIZE);
    entries
        .div_ceil(CBUF_ENTRIES_PER_BANK)
        .checked_add(1)
        .ok_or("weight bank count overflows task planning")
}

/// Plans zero-padding convolution height splits using Mesa's
/// `rkt_split_tasks()` CBUF policy.
///
/// The current executable schema has no padding fields, so this requires
/// valid-convolution geometry:
/// `output = (input - kernel) / stride + 1`.
pub fn plan_conv_tasks(shape: &ConvShape) -> Result<Vec<ConvTask>, &'static str> {
    if shape.input_width == 0
        || shape.input_height == 0
        || shape.input_channels == 0
        || shape.output_width == 0
        || shape.output_height == 0
        || shape.output_channels == 0
        || shape.weights_width == 0
        || shape.weights_height == 0
    {
        return Err("convolution dimensions and channel counts must be nonzero");
    }
    if shape.stride == 0 {
        return Err("convolution stride must be nonzero");
    }
    if shape.weights_width > shape.input_width || shape.weights_height > shape.input_height {
        return Err("convolution kernel exceeds the zero-padded input extent");
    }

    let expected_output_width = (shape.input_width - shape.weights_width) / shape.stride + 1;
    let expected_output_height = (shape.input_height - shape.weights_height) / shape.stride + 1;
    if shape.output_width != expected_output_width || shape.output_height != expected_output_height
    {
        return Err("output shape does not match zero-padding valid-convolution geometry");
    }

    let entries_per_slice = conv_entries_per_slice(shape)?;
    if entries_per_slice == 0 {
        return Err("convolution requires zero CBUF entries per input slice");
    }
    let input_banks_required = entries_per_slice
        .checked_mul(u64::from(shape.input_height))
        .ok_or("input bank count overflows task planning")?
        .div_ceil(CBUF_ENTRIES_PER_BANK);
    let weights_banks_required = conv_weights_banks(shape)?;

    let weights_with_guard_bank = weights_banks_required
        .checked_add(1)
        .ok_or("weight bank count overflows task planning")?;
    let (available_input_banks, available_weights_banks, reuse_weights) =
        if weights_with_guard_bank < u64::from(CBUF_BANKS) {
            (
                u64::from(CBUF_BANKS) - weights_banks_required,
                weights_banks_required,
                true,
            )
        } else {
            // Mesa's partial-weights/partial-input split.
            (7, 5, false)
        };

    if input_banks_required <= available_input_banks {
        return Ok(vec![ConvTask {
            index: 0,
            input_top: 0,
            input_height: shape.input_height,
            output_top: 0,
            output_height: shape.output_height,
            input_offset_bytes: 0,
            output_offset_bytes: 0,
            input_banks: u32::try_from(input_banks_required)
                .map_err(|_| "input bank count exceeds u32")?,
            weights_banks: CBUF_BANKS
                - u32::try_from(input_banks_required)
                    .map_err(|_| "input bank count exceeds u32")?,
            overlap_slices: 0,
            retain_slices: 0,
            reuse_weights: false,
        }]);
    }

    let available_slices = CBUF_ENTRIES_PER_BANK * available_input_banks / entries_per_slice;
    if available_slices == 0 {
        return Err("one input row does not fit in the CBUF banks reserved for input");
    }
    let available_slices =
        u32::try_from(available_slices).map_err(|_| "available slice count exceeds u32")?;
    if available_slices < shape.weights_height {
        return Err("CBUF input slice capacity is smaller than the convolution kernel height");
    }

    let mut drafts = vec![ConvTaskDraft {
        top: 0,
        bottom: available_slices - 1,
        overlap_slices: 0,
        retain_slices: 0,
    }];
    let mut slice = shape.weights_height - 1;
    while slice < shape.input_height {
        let previous = *drafts.last().expect("first task is present");
        while slice <= previous.bottom {
            slice = slice
                .checked_add(shape.stride)
                .ok_or("split slice index overflows u32")?;
        }
        slice -= shape.stride;

        let top = slice
            .min(previous.bottom)
            .checked_sub(shape.weights_height - 1)
            .and_then(|value| value.checked_add(shape.stride))
            .ok_or("split top slice underflows or overflows u32")?;
        let mut bottom = top
            .checked_add(available_slices - 1)
            .ok_or("split bottom slice overflows u32")?;
        if bottom >= shape.input_height - 1 {
            bottom = shape.input_height - 1;
            drafts.push(ConvTaskDraft {
                top,
                bottom,
                overlap_slices: 0,
                retain_slices: 0,
            });
            break;
        }

        slice = top
            .checked_add(shape.weights_height - 1)
            .ok_or("split slice index overflows u32")?;
        drafts.push(ConvTaskDraft {
            top,
            bottom,
            overlap_slices: 0,
            retain_slices: 0,
        });
    }

    if drafts.len() < 2 {
        return Err("CBUF split planning did not make forward progress");
    }

    for index in 1..drafts.len() {
        let previous_bottom = drafts[index - 1].bottom;
        let current_top = drafts[index].top;
        if previous_bottom >= current_top {
            let overlap = previous_bottom - current_top + 1;
            drafts[index].overlap_slices = overlap;
            drafts[index - 1].retain_slices = overlap;
        }
    }

    let input_line_stride = u64::from(shape.input_width) * ATOMIC_K_SIZE;
    let output_line_stride = u64::from(shape.output_width) * ATOMIC_K_SIZE;
    let mut output_height_processed = 0u32;
    let mut tasks = Vec::with_capacity(drafts.len());
    for (index, draft) in drafts.into_iter().enumerate() {
        let input_height = draft.bottom - draft.top + 1;
        let output_height = (input_height - shape.weights_height) / shape.stride + 1;
        if output_height == 0 {
            return Err("CBUF split produces a task with no output rows");
        }

        let input_offset = input_line_stride
            .checked_mul(u64::from(draft.top))
            .ok_or("input task offset overflows")?;
        let output_offset = output_line_stride
            .checked_mul(u64::from(output_height_processed))
            .ok_or("output task offset overflows")?;
        tasks.push(ConvTask {
            index: u32::try_from(index).map_err(|_| "task count exceeds u32")?,
            input_top: draft.top,
            input_height,
            output_top: output_height_processed,
            output_height,
            input_offset_bytes: u32::try_from(input_offset)
                .map_err(|_| "input task offset exceeds u32")?,
            output_offset_bytes: u32::try_from(output_offset)
                .map_err(|_| "output task offset exceeds u32")?,
            input_banks: u32::try_from(available_input_banks)
                .map_err(|_| "input bank count exceeds u32")?,
            weights_banks: u32::try_from(available_weights_banks)
                .map_err(|_| "weight bank count exceeds u32")?,
            overlap_slices: draft.overlap_slices,
            retain_slices: draft.retain_slices,
            reuse_weights: reuse_weights && index > 0,
        });
        output_height_processed = output_height_processed
            .checked_add(output_height)
            .ok_or("planned output height overflows")?;
    }

    if output_height_processed != shape.output_height {
        return Err("split tasks do not cover the declared output height exactly");
    }
    Ok(tasks)
}

/// Logical shape for a standalone DPU LUT pass -- the only LUT-activation
/// path this crate builds, matching the vendor compiler's decoded
/// sigmoid/tanh routing (confirmed via live hardware trace of the vendor
/// runtime itself, see `rknpu-spelunking/NOTES.md`): DPU runs in flying
/// mode, DPU_RDMA/MRDMA supplies the input tensor directly from memory,
/// and CNA/CORE are not kicked. Chain a `build_conv_regcmd` task (with
/// `Activation::None`) into this via `build_conv_then_lut_regcmd` for a
/// conv-then-sigmoid/tanh pipeline -- LUT activations are never fused
/// into the conv's own DPU pass, unlike `Relu`/`Relux`.
#[derive(Clone, Copy)]
pub struct LutShape {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub input_zero_point: u32,
    pub output_zero_point: u32,
    /// Quantization scale for the bytes read by DPU_RDMA/MRDMA.
    pub input_scale: f32,
    /// Quantization scale for the bytes written by DPU.
    pub output_scale: f32,
}

pub struct LutBuffers {
    pub input_addr: u32,
    pub output_addr: u32,
}

/// Shared CNA->CORE->DPU->DPU_RDMA emission, factored out of
/// `build_conv_regcmd` so `build_conv_then_pooling_regcmd` can reuse the
/// exact same hardware-validated sequence instead of duplicating it. Pure
/// extraction -- no behavior change versus the original single function
/// (verified by inspection: every `cmds.push` below is unchanged from
/// before the split, in the same order).
///
/// `dpu_output_mode` is the one new degree of freedom the pooling caller
/// needs: `DPU_FEATURE_MODE_CFG.output_mode` is bit0->PPU (on-chip,
/// pipelined), bit1->outside/memory (can be both). `build_conv_regcmd`
/// always passes `2` (outside only, its original hardcoded behavior,
/// unchanged). When bit1 is clear, `DPU_DST_BASE_ADDR`/`DPU_DST_SURF_
/// STRIDE` are written as 0 rather than `bufs.output_addr`-derived values
/// -- UNCONFIRMED whether the hardware truly ignores them in that case or
/// just doesn't care what's there, but 0 matches this codebase's existing
/// "safe default when unused" convention.
///
/// `addition`, when `Some`, fuses Mesa's `add_tensor` element-wise-add
/// path into this same DPU pass instead of the usual fully-bypassed
/// EW/ERDMA block -- see `AddTensor`'s own doc comment for exactly
/// what's hardware-confirmed. Every existing caller passes `None`
/// (unchanged behavior); only `build_conv_with_add_regcmd` passes
/// `Some`.
fn build_conv_cna_core_dpu_dpu_rdma(
    shape: &ConvShape,
    bufs: &ConvBuffers,
    task: &ConvTask,
    dpu_output_mode: u32,
    addition: Option<&AddTensor>,
) -> Vec<RegCmd> {
    let input_channels_real_is_one = shape.input_channels == 1;
    assert!(
        !(input_channels_real_is_one && shape.output_channels > 1),
        "build_conv_regcmd: input_channels==1 && output_channels>1 needs \
         fill_task()'s \"wide atomic\" branch, not implemented here (no \
         current client's shape exercises it)"
    );
    // EXPERIMENTAL, awaiting hardware validation (see project memory /
    // MobileNetV2 Fp16 multi-channel investigation): the multi-channel
    // branch below is precision-generic in every place it was checked
    // (CnaCvtCon0-4's bypass convention already covers multi-channel int8
    // AND single-channel fp16 identically; DpuOutCvtScale/PadCon1/BsOwCfg
    // etc. all branch on `shape.precision` alone, never on channel count).
    // The two real channel-count-AND-byte-width-dependent formulas
    // (`conv_entries_per_slice`, `input_data_entries` below) have been
    // generalized to scale by `Precision::bytes_per_element()`, mirroring
    // Mesa's own `calc_entries_per_slice()` (`rkt_task.c`), which threads a
    // `bpe` factor through the exact same math -- hardcoded to
    // `sizeof(uint8_t)` there since Mesa never emits non-int8 ops, but
    // structurally the same generalization. Not yet hardware-confirmed for
    // input_channels>1 -- assert relaxed to allow real hardware testing,
    // NOT because this has been proven correct.

    let input_banks = task.input_banks;
    let weight_banks = task.weights_banks;

    // fill_task(): channel/kernel counts round up to the hardware's
    // atomic granularity, independent of the op's logical shape. See
    // compute_task_input_channels/compute_task_output_channels's own doc
    // comments for why these are separate functions.
    let task_input_channels = compute_task_input_channels(shape);
    let task_output_channels = compute_task_output_channels(shape);
    let weights_kernels = if shape.depthwise {
        1
    } else {
        shape.output_channels.next_multiple_of(2)
    };

    // fill_task(): input_channels_real==1 && (output_channels_real>1 ||
    // addition) selects a "wide atomic" branch -- asserted unreachable
    // above, so this is always the plain else branch for every shape this
    // function actually gets called with.
    let line_stride = |w: u32| w * ATOMIC_K_SIZE as u32;
    let input_line_stride = line_stride(shape.input_width) / 4;
    let input_surface_stride = input_line_stride * (shape.input_height / 4 - 1);
    let output_surface_stride =
        (line_stride(shape.output_width) * shape.output_height) / FEATURE_ATOMIC_SIZE as u32;
    let surfaces_per_row =
        shape.output_width * shape.output_height * 2 * if shape.depthwise { 2 } else { 1 };

    // fill_task(): input_data_entries, three-way branch. The multi-channel
    // (`else`) arm is EXPERIMENTAL for Fp16 (bpe>1): Mesa's own source
    // (`rkt_task.c`) computes this arm with a hardcoded `bpe=sizeof(uint8_t)`
    // implicitly baked into the channel-count term (never generalized,
    // since Mesa never emits non-int8 ops) -- generalized here by analogy
    // with that same file's OWN `calc_entries_per_slice()`, which DOES
    // thread a `bpe` factor through the identical
    // `DIV_ROUND_UP(channels * bpe, FEATURE_ATOMIC_SIZE)` pattern (also
    // hardcoded to 1 there, but structurally the general form). Not yet
    // hardware-confirmed for bpe>1.
    let bpe = shape.precision.bytes_per_element();
    let input_data_entries = if input_channels_real_is_one {
        shape.input_width * shape.input_height
    } else if shape.input_width == 40 && shape.input_channels == 40 {
        40
    } else {
        (shape.input_width * 2 * (shape.input_channels * bpe).div_ceil(FEATURE_ATOMIC_SIZE as u32))
            .div_ceil(8)
    };

    // rkt_regcmd.c: DPU_OUT_CVT_OFFSET/SCALE/SHIFT -- float-bits-based
    // fixed-point conversion (f32::to_bits() is bit-exact for C's fui()).
    let out_offset = shape.output_zero_point.wrapping_sub(0x80);
    // See compute_out_shift's own doc comment for why the shift itself is
    // a separate function; out_scale isn't part of that assert's guarded
    // value so it's still derived inline here.
    let out_shift_raw = compute_out_shift(shape);
    assert!(
        out_shift_raw >= 0,
        "unsupported output conversion scale: input_scale={}, weights_scale={}, output_scale={}, \
         computed out_shift={out_shift_raw}",
        shape.input_scale,
        shape.weights_scale,
        shape.output_scale,
    );
    let out_shift = out_shift_raw as u32;
    let conv_scale = (shape.input_scale * shape.weights_scale) / shape.output_scale;
    let scale_bits = conv_scale.to_bits();
    let mut out_scale = ((scale_bits >> 9) & 0x7fff) + 1;
    if out_scale < (1 << 14) {
        out_scale |= 1 << 14;
    }

    let mut cmds: Vec<RegCmd> = Vec::new();

    // ========================================================================
    // CNA
    // ========================================================================

    let mut cbuf_con0_builder = Register::<CnaCbufCon0>::new();
    cbuf_con0_builder
        .weight_bank(Bits::new(weight_banks))
        .data_bank(Bits::new(input_banks));
    if task.reuse_weights {
        cbuf_con0_builder.weight_reuse(Bits::new(1));
    }
    cmds.push(cbuf_con0_builder.build()); // written again after WEIGHT_SIZE2 below -- Mesa emits it twice

    cmds.push(zero::<CnaDcompRegnum>());
    cmds.push(zero::<CnaDcompCtrl>());

    let mut conv_con1_builder = Register::<CnaConvCon1>::new();
    if input_channels_real_is_one {
        conv_con1_builder
            .nonalign_dma(Bits::new(1))
            .group_line_off(Bits::new(1))
            .argb_in(Bits::new(8));
    }
    if shape.depthwise {
        conv_con1_builder.conv_mode(Bits::new(3));
    }
    conv_con1_builder
        .in_precision(Bits::new(shape.precision.enum_value()))
        .proc_precision(Bits::new(shape.precision.enum_value()));
    cmds.push(conv_con1_builder.build());

    cmds.push(
        Register::<DpuSPointer>::new()
            .pointer_pp_mode(Bits::new(1))
            .executer_pp_en(Bits::new(1))
            .pointer_pp_en(Bits::new(1))
            .build(),
    );
    cmds.push(
        Register::<DpuRdmaSPointer>::new()
            .pointer_pp_mode(Bits::new(1))
            .executer_pp_en(Bits::new(1))
            .pointer_pp_en(Bits::new(1))
            .build(),
    );
    cmds.push(conv_con1_builder.build()); // second write, same value

    cmds.push(
        Register::<CnaConvCon2>::new()
            .feature_grains(Bits::new(50 + shape.stride + 1)) // Mesa's own comment: "Magic"
            .build(),
    );
    cmds.push(
        Register::<CnaConvCon3>::new()
            .conv_x_stride(Bits::new(shape.stride))
            .conv_y_stride(Bits::new(shape.stride))
            .build(),
    );

    cmds.push(
        Register::<CnaDataSize0>::new()
            .datain_width(Bits::new(shape.input_width))
            .datain_height(Bits::new(task.input_height))
            .build(),
    );
    cmds.push(
        Register::<CnaDataSize1>::new()
            .datain_channel_real(Bits::new(shape.input_channels - 1))
            .datain_channel(Bits::new(task_input_channels))
            .build(),
    );
    cmds.push(
        Register::<CnaDataSize2>::new()
            .dataout_width(Bits::new(shape.output_width))
            .build(),
    );
    cmds.push(
        Register::<CnaDataSize3>::new()
            .dataout_atomics(Bits::new(shape.output_width * task.output_height))
            .build(),
    );

    let bpe = shape.precision.bytes_per_element();
    cmds.push(
        Register::<CnaWeightSize0>::new()
            .weight_bytes(Bits::new(
                shape.weights_width
                    * shape.weights_height
                    * task_input_channels
                    * weights_kernels
                    * bpe,
            ))
            .build(),
    );
    cmds.push(
        Register::<CnaWeightSize1>::new()
            .weight_bytes_per_kernel(Bits::new(
                shape.weights_width * shape.weights_height * task_input_channels * bpe,
            ))
            .build(),
    );
    cmds.push(
        Register::<CnaWeightSize2>::new()
            .weight_width(Bits::new(shape.weights_width))
            .weight_height(Bits::new(shape.weights_height))
            .weight_kernels(Bits::new(weights_kernels))
            .build(),
    );

    cmds.push(cbuf_con0_builder.build());

    cmds.push(
        Register::<CnaCbufCon1>::new()
            .data_entries(Bits::new(input_data_entries))
            .build(),
    );

    if input_channels_real_is_one && shape.precision == Precision::Int8 {
        const CVT_TRUNCATE: u32 = 14;
        const CVT_SCALE: u32 = 16384;
        const CVT_OFFSET: u32 = 65408;
        cmds.push(
            Register::<CnaCvtCon0>::new()
                .cvt_truncate_0(Bits::new(CVT_TRUNCATE))
                .cvt_truncate_1(Bits::new(CVT_TRUNCATE))
                .cvt_truncate_2(Bits::new(CVT_TRUNCATE))
                .cvt_truncate_3(Bits::new(CVT_TRUNCATE))
                .build(),
        );
        cmds.push(
            Register::<CnaCvtCon1>::new()
                .cvt_scale0(Bits::new(CVT_SCALE))
                .cvt_offset0(Bits::new(CVT_OFFSET))
                .build(),
        );
        cmds.push(
            Register::<CnaCvtCon2>::new()
                .cvt_scale1(Bits::new(CVT_SCALE))
                .cvt_offset1(Bits::new(CVT_OFFSET))
                .build(),
        );
        cmds.push(
            Register::<CnaCvtCon3>::new()
                .cvt_scale2(Bits::new(CVT_SCALE))
                .cvt_offset2(Bits::new(CVT_OFFSET))
                .build(),
        );
        cmds.push(
            Register::<CnaCvtCon4>::new()
                .cvt_scale3(Bits::new(CVT_SCALE))
                .cvt_offset3(Bits::new(CVT_OFFSET))
                .build(),
        );
    } else {
        // Two cases share this bypassed-CVT convention: the multi-channel
        // int8 branch (its own original use), and single-channel Fp16
        // (round 5's diff against real `conv.rknn` ground truth found
        // this exact recipe -- `cvt_bypass=1, data_sign=1, cvt_type=1`
        // on CON0, `cvt_scale=1, cvt_offset=0` on CON1-4 -- byte-for-byte
        // identical to what the real compiler already emits here, no
        // separate fp16-specific formula needed).
        cmds.push(
            Register::<CnaCvtCon0>::new()
                .data_sign(Bits::new(1))
                .cvt_type(Bits::new(1))
                .cvt_bypass(Bits::new(1))
                .build(),
        );
        cmds.push(
            Register::<CnaCvtCon1>::new()
                .cvt_scale0(Bits::new(1))
                .build(),
        );
        cmds.push(
            Register::<CnaCvtCon2>::new()
                .cvt_scale1(Bits::new(1))
                .build(),
        );
        cmds.push(
            Register::<CnaCvtCon3>::new()
                .cvt_scale2(Bits::new(1))
                .build(),
        );
        cmds.push(
            Register::<CnaCvtCon4>::new()
                .cvt_scale3(Bits::new(1))
                .build(),
        );
    }

    cmds.push(zero::<CnaFcCon0>());
    cmds.push(zero::<CnaFcCon1>());
    cmds.push(
        Register::<CnaPadCon0>::new()
            .pad_top(Bits::new(0))
            .pad_left(Bits::new(0))
            .build(),
    );
    cmds.push(
        Register::<CnaFeatureDataAddr>::new()
            .feature_base_addr(Bits::new(
                bufs.input_addr
                    .checked_add(task.input_offset_bytes)
                    .expect("input task address overflows u32"),
            ))
            .build(),
    );
    cmds.push(zero::<CnaFcCon2>());
    cmds.push(
        Register::<CnaDmaCon0>::new()
            .data_burst_len(Bits::new(15))
            .weight_burst_len(Bits::new(15))
            .build(),
    );
    cmds.push(
        Register::<CnaDmaCon1>::new()
            .line_stride(Bits::new(input_line_stride))
            .build(),
    );
    cmds.push(
        Register::<CnaDmaCon2>::new()
            .surf_stride(Bits::new(input_surface_stride))
            .build(),
    );

    cmds.push(
        Register::<CnaFcDataSize0>::new()
            .dma_width(Bits::new(shape.input_width))
            .dma_height(Bits::new(task.input_height))
            .build(),
    );
    cmds.push(
        Register::<CnaFcDataSize1>::new()
            .dma_channel(Bits::new(task_input_channels))
            .build(),
    );

    cmds.push(zero::<CnaDcompCtrl>());
    cmds.push(zero::<CnaDcompRegnum>());
    cmds.push(
        Register::<CnaDcompAddr0>::new()
            .decompress_addr0(Bits::new(bufs.weights_addr))
            .build(),
    );
    cmds.push(zero::<CnaDcompAmount0>());
    cmds.push(zero::<CnaDcompAmount1>());
    cmds.push(zero::<CnaDcompAmount2>());
    cmds.push(zero::<CnaDcompAmount3>());
    cmds.push(zero::<CnaDcompAmount4>());
    cmds.push(zero::<CnaDcompAmount5>());
    cmds.push(zero::<CnaDcompAmount6>());
    cmds.push(zero::<CnaDcompAmount7>());
    cmds.push(zero::<CnaDcompAmount8>());
    cmds.push(zero::<CnaDcompAmount9>());
    cmds.push(zero::<CnaDcompAmount10>());
    cmds.push(zero::<CnaDcompAmount11>());
    cmds.push(zero::<CnaDcompAmount12>());
    cmds.push(zero::<CnaDcompAmount13>());
    cmds.push(zero::<CnaDcompAmount14>());
    cmds.push(zero::<CnaDcompAmount15>());

    cmds.push(
        if input_channels_real_is_one && shape.precision == Precision::Int8 {
            Register::<CnaCvtCon5>::new()
                .per_channel_cvt_en(Bits::new(65535))
                .build()
        } else {
            zero::<CnaCvtCon5>()
        },
    );

    // Fp16's ground truth (round 5) has PAD_CON1 at a plain 0 -- the
    // int8-only zero-point-centering convention below doesn't apply once
    // CVT is genuinely bypassed.
    let mut pad_con1 = if shape.precision == Precision::Fp16 {
        0
    } else if shape.weights_width >= 3 && shape.input_zero_point == 0 {
        0xffff8080u32
    } else {
        shape.input_zero_point.wrapping_sub(0x80)
    };
    if shape.depthwise && shape.input_zero_point == 0x8b {
        pad_con1 = 0x0b0b;
    }
    cmds.push(
        Register::<CnaPadCon1>::new()
            .pad_value(Bits::new(pad_con1))
            .build(),
    );

    // ========================================================================
    // CORE
    // ========================================================================

    let mut misc_cfg_builder = Register::<CoreMiscCfg>::new();
    // qd_en is int8-quantization-specific -- real fp16 ground truth
    // (round 5) has it clear.
    if shape.precision == Precision::Int8 {
        misc_cfg_builder.qd_en(Bits::new(1));
    }
    misc_cfg_builder.proc_precision(Bits::new(shape.precision.enum_value()));
    if shape.depthwise {
        misc_cfg_builder.dw_en(Bits::new(1));
    }
    cmds.push(misc_cfg_builder.build());

    cmds.push(
        Register::<CoreDataoutSize0>::new()
            .dataout_width(Bits::new(shape.output_width - 1))
            .dataout_height(Bits::new(task.output_height - 1))
            .build(),
    );
    cmds.push(
        Register::<CoreDataoutSize1>::new()
            .dataout_channel(Bits::new(task_output_channels - 1))
            .build(),
    );
    cmds.push(
        Register::<CoreClipTruncate>::new()
            .clip_truncate(Bits::new(shape.truncate_bits))
            .build(),
    );
    cmds.push(RegCmd::new(DOMAIN_CORE, 0x3030, 0)); // TRM-mandated reserved write, no REG_CORE_* name

    // ========================================================================
    // DPU
    // ========================================================================

    let mut feat_mode_builder = Register::<DpuFeatureModeCfg>::new();
    feat_mode_builder
        .burst_len(Bits::new(15))
        .output_mode(Bits::new(dpu_output_mode));
    if shape.depthwise {
        feat_mode_builder.conv_mode(Bits::new(3));
    }
    cmds.push(feat_mode_builder.build());

    // build_conv_regcmd used to hardcode int8 here regardless of what CNA
    // says -- confirmed wrong for non-int8 by the real conv_w16a16i.rknn
    // capture (Step 1 of the dtype-coverage investigation) and by
    // conv.rknn's own fp16-shaped ground truth (round 5): the real
    // compiler sets all three precision fields here too.
    cmds.push(if shape.precision == Precision::Int8 {
        zero::<DpuDataFormat>()
    } else {
        Register::<DpuDataFormat>::new()
            .in_precision(Bits::new(shape.precision.enum_value()))
            .out_precision(Bits::new(shape.precision.enum_value()))
            .proc_precision(Bits::new(shape.precision.enum_value()))
            .build()
    });
    cmds.push(zero::<DpuOffsetPend>());
    // bit1 clear (not writing "outside"/memory) -> 0, see this function's
    // own doc comment.
    let dpu_writes_to_memory = dpu_output_mode & 0b10 != 0;
    cmds.push(
        Register::<DpuDstBaseAddr>::new()
            .dst_base_addr(Bits::new(if dpu_writes_to_memory {
                bufs.output_addr
                    .checked_add(task.output_offset_bytes)
                    .expect("output task address overflows u32")
            } else {
                0
            }))
            .build(),
    );
    cmds.push(
        Register::<DpuDstSurfStride>::new()
            .dst_surf_stride(Bits::new(if dpu_writes_to_memory {
                output_surface_stride
            } else {
                0
            }))
            .build(),
    );
    cmds.push(
        Register::<DpuDataCubeWidth>::new()
            .width(Bits::new(shape.output_width - 1))
            .build(),
    );
    cmds.push(
        Register::<DpuDataCubeHeight>::new()
            .height(Bits::new(task.output_height - 1))
            .build(),
    );
    cmds.push(zero::<DpuDataCubeNotchAddr>());
    cmds.push(
        Register::<DpuDataCubeChannel>::new()
            .orig_channel(Bits::new(shape.output_channels - 1))
            .channel(Bits::new(task_output_channels - 1))
            .build(),
    );

    // BS (bias-subtract): Mesa never bypasses the ALU outright -- always
    // runs it (ALGO(2)|SRC(1)) against a real (if zero-filled) biases
    // buffer, wired in via DPU_RDMA_RDMA_BS_BASE_ADDR below. RELU/RELUX are
    // an additional fused-activation degree of freedom this op didn't
    // originally expose (see `Activation` above) -- MUL stays permanently
    // bypassed, matching Mesa (no client of this function needs BS's mul
    // stage).
    let (bs_relu_bypass, bs_relux_en, bs_relux_cmp) = match shape.activation {
        Activation::None => (1, 0, 0),
        Activation::Relu => (0, 0, 0),
        Activation::Relux { cmp } => (0, 1, cmp),
    };
    cmds.push(
        Register::<DpuBsCfg>::new()
            .bs_alu_algo(Bits::new(2))
            .bs_alu_src(Bits::new(1))
            .bs_relu_bypass(Bits::new(bs_relu_bypass))
            .bs_relux_en(Bits::new(bs_relux_en))
            .bs_mul_bypass(Bits::new(1))
            .build(),
    );
    cmds.push(zero::<DpuBsAluCfg>());
    cmds.push(zero::<DpuBsMulCfg>());
    cmds.push(
        Register::<DpuBsReluxCmpValue>::new()
            .bs_relux_cmp_dat(Bits::new(bs_relux_cmp))
            .build(),
    );
    let mut bs_ow_cfg_builder = Register::<DpuBsOwCfg>::new();
    if shape.depthwise {
        bs_ow_cfg_builder
            .size_e_0(Bits::new(3))
            .size_e_1(Bits::new(3))
            .size_e_2(Bits::new(3));
    } else {
        bs_ow_cfg_builder
            .size_e_0(Bits::new(1))
            .size_e_1(Bits::new(1))
            .size_e_2(Bits::new(1));
    }
    // Round 6's fp16 fix: real ground truth bypasses the CPEND stage
    // entirely here (`od_bypass=1`) rather than feeding it the int8
    // zero-point-derived operand below.
    if shape.precision == Precision::Fp16 {
        bs_ow_cfg_builder.od_bypass(Bits::new(1));
    }
    cmds.push(bs_ow_cfg_builder.build());
    cmds.push(
        Register::<DpuBsOwOp>::new()
            .ow_op(Bits::new(match shape.precision {
                Precision::Int8 => 0x80 - shape.weights_zero_point,
                Precision::Fp16 => 0,
            }))
            .build(),
    );
    cmds.push(
        Register::<DpuWdmaSize0>::new()
            .channel_wdma(Bits::new(task_output_channels - 1))
            .build(),
    );
    cmds.push(
        Register::<DpuWdmaSize1>::new()
            .height_wdma(Bits::new(task.output_height - 1))
            .width_wdma(Bits::new(shape.output_width - 1))
            .build(),
    );

    // BN (batchnorm): genuinely fully bypassed, all four bits.
    cmds.push(
        Register::<DpuBnCfg>::new()
            .bn_relu_bypass(Bits::new(1))
            .bn_mul_bypass(Bits::new(1))
            .bn_alu_bypass(Bits::new(1))
            .bn_bypass(Bits::new(1))
            .build(),
    );
    cmds.push(zero::<DpuBnAluCfg>());
    cmds.push(zero::<DpuBnMulCfg>());
    cmds.push(zero::<DpuBnReluxCmpValue>());

    // EW (elementwise): `addition.is_none()` is the original add_tensor==-1
    // branch (fully bypassed, all five bits), unchanged. `Some` fuses
    // Mesa's add_tensor path -- ported directly from `rkt_regcmd.c`, see
    // `AddTensor`'s doc comment for what's hardware-confirmed.
    if let Some(add) = addition {
        cmds.push(
            Register::<DpuEwCfg>::new()
                .ew_cvt_type(Bits::new(1))
                .ew_data_mode(Bits::new(1))
                .edata_size(Bits::new(1))
                .ew_alu_algo(Bits::new(add.algo)) // see AddTensor::algo's doc comment
                .ew_relu_bypass(Bits::new(1))
                .ew_lut_bypass(Bits::new(1))
                .ew_op_src(Bits::new(1)) // operand from outside (the second tensor)
                .build(),
        );
        cmds.push(
            Register::<DpuEwCvtOffsetValue>::new()
                .ew_op_cvt_offset(Bits::new(add.cvt_offset))
                .build(),
        );
        let (ew_scale, ew_shift) = ew_add_cvt(add.scale, shape.input_scale, shape.weights_scale);
        cmds.push(
            Register::<DpuEwCvtScaleValue>::new()
                .ew_op_cvt_scale(Bits::new(ew_scale))
                .ew_op_cvt_shift(Bits::new(ew_shift))
                .build(),
        );
        cmds.push(zero::<DpuEwReluxCmpValue>());
    } else {
        cmds.push(
            Register::<DpuEwCfg>::new()
                .ew_relu_bypass(Bits::new(1))
                .ew_op_cvt_bypass(Bits::new(1))
                .ew_lut_bypass(Bits::new(1))
                .ew_op_bypass(Bits::new(1))
                .ew_bypass(Bits::new(1))
                .build(),
        );
        cmds.push(zero::<DpuEwCvtOffsetValue>());
        cmds.push(
            Register::<DpuEwCvtScaleValue>::new()
                .ew_op_cvt_scale(Bits::new(1))
                .build(),
        );
        cmds.push(zero::<DpuEwReluxCmpValue>());
    }

    match shape.precision {
        Precision::Int8 => {
            cmds.push(
                Register::<DpuOutCvtOffset>::new()
                    .out_cvt_offset(Bits::new(out_offset))
                    .build(),
            );
            cmds.push(
                Register::<DpuOutCvtScale>::new()
                    .out_cvt_scale(Bits::new(out_scale))
                    .build(),
            );
            cmds.push(
                Register::<DpuOutCvtShift>::new()
                    .out_cvt_shift(Bits::new(out_shift - 1))
                    .build(),
            );
        }
        Precision::Fp16 => {
            // int8's OUT_CVT formula above (`out_offset`/`out_scale`/
            // `out_shift`) is a fixed-point requantization derived by
            // bit-tricking a float32's mantissa/exponent -- it doesn't
            // apply to real fp16 output. Ground truth (rounds 1-3 for
            // `fp32tofp16_en`, round 5 for `out_cvt_scale=1`): offset and
            // shift both zeroed, scale is the fp32->fp16 conversion
            // enable bit plus a plain identity scale of 1.
            cmds.push(zero::<DpuOutCvtOffset>());
            cmds.push(
                Register::<DpuOutCvtScale>::new()
                    .fp32tofp16_en(Bits::new(1))
                    .out_cvt_scale(Bits::new(1))
                    .build(),
            );
            cmds.push(zero::<DpuOutCvtShift>());
        }
    }

    cmds.push(zero::<DpuEwOpValue0>());
    cmds.push(zero::<DpuEwOpValue1>());
    cmds.push(zero::<DpuEwOpValue2>());
    cmds.push(zero::<DpuEwOpValue3>());
    cmds.push(zero::<DpuEwOpValue4>());
    cmds.push(zero::<DpuEwOpValue5>());
    cmds.push(zero::<DpuEwOpValue6>());
    cmds.push(zero::<DpuEwOpValue7>());

    cmds.push(
        Register::<DpuSurfaceAdd>::new()
            .surf_add(Bits::new(surfaces_per_row))
            .build(),
    );
    cmds.push(RegCmd::new(DOMAIN_DPU, 0x40c4, 0)); // TRM-mandated reserved write, no REG_DPU_* name

    cmds.push(zero::<DpuLutAccessCfg>());
    cmds.push(zero::<DpuLutAccessData>());
    cmds.push(zero::<DpuLutCfg>());
    cmds.push(zero::<DpuLutInfo>());
    cmds.push(zero::<DpuLutLeStart>());
    cmds.push(zero::<DpuLutLeEnd>());
    cmds.push(zero::<DpuLutLoStart>());
    cmds.push(zero::<DpuLutLoEnd>());
    cmds.push(zero::<DpuLutLeSlopeScale>());
    cmds.push(zero::<DpuLutLeSlopeShift>());
    cmds.push(zero::<DpuLutLoSlopeScale>());
    cmds.push(zero::<DpuLutLoSlopeShift>());

    // ========================================================================
    // DPU_RDMA
    // ========================================================================

    cmds.push(
        Register::<DpuRdmaDataCubeWidth>::new()
            .width(Bits::new(shape.output_width - 1))
            .build(),
    );
    cmds.push(
        Register::<DpuRdmaDataCubeHeight>::new()
            .height(Bits::new(task.output_height - 1))
            .build(),
    );
    cmds.push(
        Register::<DpuRdmaDataCubeChannel>::new()
            .channel(Bits::new(task_output_channels - 1))
            .build(),
    );

    if let Some(add) = addition {
        cmds.push(
            Register::<DpuRdmaSrcBaseAddr>::new()
                .src_base_addr(Bits::new(add.src_addr))
                .build(),
        );
    } else {
        cmds.push(zero::<DpuRdmaSrcBaseAddr>()); // add_tensor == -1
    }
    cmds.push(
        Register::<DpuRdmaBrdmaCfg>::new()
            .brdma_data_use(Bits::new(1))
            .build(),
    );
    cmds.push(
        Register::<DpuRdmaBsBaseAddr>::new()
            .bs_base_addr(Bits::new(bufs.bias_addr))
            .build(),
    );
    cmds.push(zero::<DpuRdmaNrdmaCfg>());
    cmds.push(zero::<DpuRdmaBnBaseAddr>());

    if let Some(add) = addition {
        cmds.push(
            Register::<DpuRdmaErdmaCfg>::new()
                .erdma_data_mode(Bits::new(1))
                .erdma_data_size(Bits::new(1))
                .build(),
        );
        cmds.push(
            Register::<DpuRdmaEwBaseAddr>::new()
                .ew_base_addr(Bits::new(add.ew_addr))
                .build(),
        );
        // Mesa: `MAX2(operation->output_width * operation->output_height, 12)`.
        let ew_stride = (shape.output_width * shape.output_height).max(12);
        cmds.push(
            Register::<DpuRdmaEwSurfStride>::new()
                .ew_surf_stride(Bits::new(ew_stride))
                .build(),
        );
    } else {
        cmds.push(
            Register::<DpuRdmaErdmaCfg>::new()
                .erdma_disable(Bits::new(1))
                .build(),
        ); // add_tensor == -1
        cmds.push(zero::<DpuRdmaEwBaseAddr>());
        cmds.push(zero::<DpuRdmaEwSurfStride>());
    }

    let mut rdma_feat_mode_builder = Register::<DpuRdmaFeatureModeCfg>::new();
    rdma_feat_mode_builder
        .burst_len(Bits::new(15))
        .mrdma_disable(Bits::new(1))
        .in_precision(Bits::new(shape.precision.enum_value()))
        .proc_precision(Bits::new(shape.precision.enum_value()));
    if shape.depthwise {
        rdma_feat_mode_builder.conv_mode(Bits::new(3));
    }
    cmds.push(rdma_feat_mode_builder.build());
    cmds.push(zero::<DpuRdmaSrcDmaCfg>());
    cmds.push(zero::<DpuRdmaSurfNotch>()); // add_tensor == -1
    cmds.push(zero::<DpuRdmaPadCfg>());
    cmds.push(
        Register::<DpuRdmaWeight>::new()
            .e_weight(Bits::new(1))
            .n_weight(Bits::new(1))
            .b_weight(Bits::new(1))
            .m_weight(Bits::new(1))
            .build(),
    );
    cmds.push(zero::<DpuRdmaEwSurfNotch>()); // add_tensor == -1

    cmds
}

// PC_OPERATION_ENABLE's bitmask, one bit per engine block, confirmed
// against real rknn-toolkit2-compiled regcmd programs (both conv.rknn and
// the pooling.rknn capture in rknpu-spelunking/NOTES.md's "Decoding a real
// regcmd program for a pooling-only op" section): bit = log2(mesa_target)
// - 9, where mesa_target is Mesa registers.xml's `target` enum value for
// that block (PC=0x100 CNA=0x200 CORE=0x800 DPU=0x1000 DPU_RDMA=0x2000
// PPU=0x4000 PPU_RDMA=0x8000). Bit 1 has no corresponding block (the gap
// between CNA=0x200 and CORE=0x800) and is never set in any real capture.
//
// conv.rknn's single kick is always `KICK_CNA | KICK_CORE | KICK_DPU |
// KICK_DPU_RDMA` (0x1d). The pooling.rknn capture fires *two* kicks per
// tile: a bypass CNA/CORE/DPU stage (`0x0d`, no DPU_RDMA) followed
// separately by `KICK_PPU | KICK_PPU_RDMA` (0x60) -- notably with bit 0
// (`PC_OPERATION_ENABLE_OP_EN` in rkt_registers.h's naming) clear, which
// is what revealed that bit isn't a generic mandatory "go" flag -- it's
// just CNA's own enable bit, like every other bit here.
const KICK_CNA: u32 = 1 << 0;
const KICK_CORE: u32 = 1 << 2;
const KICK_DPU: u32 = 1 << 3;
const KICK_DPU_RDMA: u32 = 1 << 4;
const KICK_PPU: u32 = 1 << 5;
const KICK_PPU_RDMA: u32 = 1 << 6;

/// Appends the standard single-task "kick" tail (ported verbatim from
/// Mesa's fill_first_regcmd(), num_tasks == 1 branch -- see rkt-basic.rs's
/// top-of-file doc comment / NOTES.md) and pads to an even length. Shared
/// by every regcmd builder in this module -- every one of them ends a
/// single task the same way, differing only in which blocks that task's
/// kick should actually enable (see `KICK_*` above -- pass exactly the
/// bits for the blocks this task configured, not a fixed value).
fn push_kick_for_task_count(cmds: &mut Vec<RegCmd>, enable_mask: u32, task_count: usize) {
    if task_count == 1 {
        cmds.push(RegCmd::new_raw(0x0)); // Mesa's single-task placeholder
    } else {
        cmds.push(RegCmd::new(DOMAIN_PC, REG_PC_BASE_ADDRESS, 0));
    }
    cmds.push(RegCmd::new(DOMAIN_PC, REG_PC_REGISTER_AMOUNTS, 0));
    cmds.push(RegCmd::new_raw(0x0041000000000000)); // TRM: required immediately before op_en
    cmds.push(RegCmd::new(0x81, REG_PC_OPERATION_ENABLE, enable_mask));

    if cmds.len() % 2 != 0 {
        cmds.push(RegCmd::new_raw(0x0));
    }
}

fn push_kick(cmds: &mut Vec<RegCmd>, enable_mask: u32) {
    push_kick_for_task_count(cmds, enable_mask, 1);
}

pub fn build_lut_regcmd(shape: &LutShape, bufs: &LutBuffers, table: LutTable) -> Vec<RegCmd> {
    assert!(
        shape.width > 0 && shape.height > 0 && shape.channels > 0,
        "build_lut_regcmd: width, height, and channels must be nonzero"
    );

    const FEATURE_ATOMIC_SIZE: u32 = 16;

    let task_channels = shape
        .channels
        .max(FEATURE_ATOMIC_SIZE)
        .next_multiple_of(FEATURE_ATOMIC_SIZE);
    let surface_stride = shape.width * shape.height * task_channels;
    let out_offset = shape.output_zero_point.wrapping_sub(0x80);
    let (bn_mul_operand, bn_mul_shift) = lut_bn_mul(shape.input_scale);
    let real_zero_point = shape.input_zero_point.wrapping_sub(0x80) as i8 as i32;
    assert!(
        lut_bn_alu_supports_zero_point(real_zero_point),
        "build_lut_regcmd: only input_zero_point values whose decoded signed zero point is one of \
         -128, -2, 0, or 127 are supported for now; got raw input_zero_point={} (decoded {})",
        shape.input_zero_point,
        real_zero_point
    );
    let bn_alu_operand = lut_bn_alu(real_zero_point, bn_mul_operand);
    let (out_cvt_scale, out_cvt_shift) = lut_out_cvt(shape.output_scale);

    let mut cmds = Vec::new();

    cmds.push(
        Register::<DpuSPointer>::new()
            .pointer_pp_mode(Bits::new(1))
            .executer_pp_en(Bits::new(1))
            .pointer_pp_en(Bits::new(1))
            .build(),
    );
    cmds.push(
        Register::<DpuRdmaSPointer>::new()
            .pointer_pp_mode(Bits::new(1))
            .executer_pp_en(Bits::new(1))
            .pointer_pp_en(Bits::new(1))
            .build(),
    );

    cmds.push(
        Register::<DpuFeatureModeCfg>::new()
            .flying_mode(Bits::new(1))
            .output_mode(Bits::new(2))
            .burst_len(Bits::new(15))
            .build(),
    );
    cmds.push(
        Register::<DpuDataFormat>::new()
            .bn_mul_shift_value_neg(Bits::new(bn_mul_shift))
            .build(),
    );
    cmds.push(zero::<DpuOffsetPend>());
    cmds.push(
        Register::<DpuDstBaseAddr>::new()
            .dst_base_addr(Bits::new(bufs.output_addr))
            .build(),
    );
    cmds.push(
        Register::<DpuDstSurfStride>::new()
            .dst_surf_stride(Bits::new(surface_stride))
            .build(),
    );
    cmds.push(
        Register::<DpuDataCubeWidth>::new()
            .width(Bits::new(shape.width - 1))
            .build(),
    );
    cmds.push(
        Register::<DpuDataCubeHeight>::new()
            .height(Bits::new(shape.height - 1))
            .build(),
    );
    cmds.push(zero::<DpuDataCubeNotchAddr>());
    cmds.push(
        Register::<DpuDataCubeChannel>::new()
            .orig_channel(Bits::new(shape.channels - 1))
            .channel(Bits::new(task_channels - 1))
            .build(),
    );

    cmds.push(
        Register::<DpuBsCfg>::new()
            .bs_bypass(Bits::new(1))
            .bs_alu_bypass(Bits::new(1))
            .bs_mul_bypass(Bits::new(1))
            .bs_relu_bypass(Bits::new(1))
            .build(),
    );
    cmds.push(zero::<DpuBsAluCfg>());
    cmds.push(zero::<DpuBsMulCfg>());
    cmds.push(zero::<DpuBsReluxCmpValue>());
    cmds.push(
        Register::<DpuBsOwCfg>::new()
            .od_bypass(Bits::new(1))
            .build(),
    );
    cmds.push(zero::<DpuBsOwOp>());
    cmds.push(
        Register::<DpuWdmaSize0>::new()
            .channel_wdma(Bits::new(task_channels - 1))
            .build(),
    );
    cmds.push(
        Register::<DpuWdmaSize1>::new()
            .height_wdma(Bits::new(shape.height - 1))
            .width_wdma(Bits::new(shape.width - 1))
            .build(),
    );

    cmds.push(
        Register::<DpuBnCfg>::new()
            .bn_alu_algo(Bits::new(2))
            .bn_alu_src(Bits::new(0))
            .bn_relu_bypass(Bits::new(1))
            .bn_mul_bypass(Bits::new(0))
            .bn_alu_bypass(Bits::new(0))
            .bn_bypass(Bits::new(0))
            .build(),
    );
    cmds.push(
        Register::<DpuBnAluCfg>::new()
            .bn_alu_operand(Bits::new(bn_alu_operand))
            .build(),
    );
    cmds.push(
        Register::<DpuBnMulCfg>::new()
            .bn_mul_operand(Bits::new(bn_mul_operand))
            .bn_mul_shift_value(Bits::new(bn_mul_shift))
            .bn_mul_src(Bits::new(0))
            .bn_truncate_src(Bits::new(0))
            .build(),
    );
    cmds.push(zero::<DpuBnReluxCmpValue>());

    cmds.push(
        Register::<DpuEwCfg>::new()
            .ew_relu_bypass(Bits::new(1))
            .ew_op_cvt_bypass(Bits::new(1))
            .ew_lut_bypass(Bits::new(0))
            .ew_op_bypass(Bits::new(table.ew_op_bypass as u32))
            .ew_bypass(Bits::new(0))
            .build(),
    );
    cmds.push(zero::<DpuEwCvtOffsetValue>());
    cmds.push(
        Register::<DpuEwCvtScaleValue>::new()
            .ew_op_cvt_scale(Bits::new(1))
            .build(),
    );
    cmds.push(zero::<DpuEwReluxCmpValue>());

    cmds.push(
        Register::<DpuOutCvtOffset>::new()
            .out_cvt_offset(Bits::new(out_offset))
            .build(),
    );
    cmds.push(
        Register::<DpuOutCvtScale>::new()
            .out_cvt_scale(Bits::new(out_cvt_scale))
            .build(),
    );
    cmds.push(
        Register::<DpuOutCvtShift>::new()
            .out_cvt_shift(Bits::new(out_cvt_shift))
            .build(),
    );
    cmds.push(zero::<DpuEwOpValue0>());
    cmds.push(zero::<DpuEwOpValue1>());
    cmds.push(zero::<DpuEwOpValue2>());
    cmds.push(zero::<DpuEwOpValue3>());
    cmds.push(zero::<DpuEwOpValue4>());
    cmds.push(zero::<DpuEwOpValue5>());
    cmds.push(zero::<DpuEwOpValue6>());
    cmds.push(zero::<DpuEwOpValue7>());
    cmds.push(
        Register::<DpuSurfaceAdd>::new()
            .surf_add(Bits::new(surface_stride))
            .build(),
    );
    cmds.push(RegCmd::new(DOMAIN_DPU, 0x40c4, 0));

    push_lut_tables_and_config(&mut cmds, table);

    cmds.push(
        Register::<DpuRdmaDataCubeWidth>::new()
            .width(Bits::new(shape.width - 1))
            .build(),
    );
    cmds.push(
        Register::<DpuRdmaDataCubeHeight>::new()
            .height(Bits::new(shape.height - 1))
            .build(),
    );
    cmds.push(
        Register::<DpuRdmaDataCubeChannel>::new()
            .channel(Bits::new(task_channels - 1))
            .build(),
    );
    cmds.push(
        Register::<DpuRdmaSrcBaseAddr>::new()
            .src_base_addr(Bits::new(bufs.input_addr))
            .build(),
    );
    cmds.push(zero::<DpuRdmaBrdmaCfg>());
    cmds.push(zero::<DpuRdmaBsBaseAddr>());
    cmds.push(zero::<DpuRdmaNrdmaCfg>());
    cmds.push(zero::<DpuRdmaBnBaseAddr>());
    cmds.push(
        Register::<DpuRdmaErdmaCfg>::new()
            .erdma_disable(Bits::new(1))
            .build(),
    );
    cmds.push(zero::<DpuRdmaEwBaseAddr>());
    cmds.push(zero::<DpuRdmaEwSurfStride>());
    cmds.push(
        Register::<DpuRdmaFeatureModeCfg>::new()
            .flying_mode(Bits::new(1))
            .burst_len(Bits::new(15))
            .build(),
    );
    cmds.push(zero::<DpuRdmaSrcDmaCfg>());
    cmds.push(zero::<DpuRdmaSurfNotch>());
    cmds.push(zero::<DpuRdmaPadCfg>());
    cmds.push(
        Register::<DpuRdmaWeight>::new()
            .e_weight(Bits::new(1))
            .n_weight(Bits::new(1))
            .b_weight(Bits::new(1))
            .m_weight(Bits::new(1))
            .build(),
    );
    cmds.push(zero::<DpuRdmaEwSurfNotch>());

    push_kick(&mut cmds, KICK_DPU | KICK_DPU_RDMA);
    cmds
}

/// rkt_regcmd.c's `DPU_OUT_CVT_SHIFT` formula (float-bits-based fixed-point
/// conversion, `conv_scale = input_scale*weights_scale/output_scale`).
/// Returns the raw shift value, NOT yet checked for non-negativity or cast
/// to `u32` -- the caller is responsible for both (see
/// `build_conv_cna_core_dpu_dpu_rdma`'s own `assert!`). Extracted as a pure
/// function (no behavior change) for the same single-source-of-truth
/// reason as the task-planning helpers above: `validate_conv_shape` needs
/// to check the exact value the builder's `assert!` protects.
pub(crate) fn compute_out_shift(shape: &ConvShape) -> i32 {
    let conv_scale = (shape.input_scale * shape.weights_scale) / shape.output_scale;
    let scale_bits = conv_scale.to_bits();
    let scale_exponent = ((scale_bits >> 23) & 0xff) as i32;
    let mut out_shift = 127 + 31 - 32 - scale_exponent + 16;
    if shape.truncate_bits > 0 {
        out_shift -= 1;
    }
    out_shift
}

/// fill_task()'s input-channel atomic rounding (`CNA_DATA_SIZE1.datain_channel`,
/// a 16-bit field -- see `builders/cna.rs`). Extracted as a pure function
/// for the same single-source-of-truth reason as the task-planning helpers:
/// `validate_conv_shape` needs to check this against the field's real bit
/// width without re-deriving the rounding formula a second time.
pub(crate) fn compute_task_input_channels(shape: &ConvShape) -> u32 {
    const FEATURE_ATOMIC_SIZE: u32 = 16;
    shape
        .input_channels
        .max(FEATURE_ATOMIC_SIZE)
        .next_multiple_of(FEATURE_ATOMIC_SIZE)
}

/// fill_task()'s output-channel atomic rounding, feeding `DPU_..._CHANNEL`
/// (13-bit fields, see `builders/dpu.rs`'s `channel`/`channel_wdma`) as
/// `task_output_channels - 1`. Extracted as a pure function for the same
/// reason as `compute_task_input_channels` above.
pub(crate) fn compute_task_output_channels(shape: &ConvShape) -> u32 {
    let mut task_output_channels = shape.output_channels.max(32).next_multiple_of(32);
    if shape.depthwise {
        if shape.output_channels <= 32 {
            task_output_channels *= 2;
        }
        task_output_channels = task_output_channels.next_multiple_of(64);
    }
    task_output_channels
}

/// Byte width of one DPU feature-atomic output surface.
pub const CONV_OUTPUT_ATOMIC_STRIDE: u32 = 16;

/// Conservative physical DPU output allocation for a convolution.
///
/// DPU write-back addresses output in 16-byte feature-atomic surfaces and
/// programs the atomically-rounded task channel count, not just the logical
/// output channel count. Allocate enough complete surfaces for that padded
/// channel count so multi-channel dispatch cannot write past the backing BO.
/// This is a physical allocation size, not a statement that the resulting
/// surface layout is already the dense NHWC ABI expected by IREE.
pub fn conv_output_scratch_bytes(shape: &ConvShape) -> usize {
    let channel_bytes =
        compute_task_output_channels(shape) as usize * shape.precision.bytes_per_element() as usize;
    let surface_count = channel_bytes.div_ceil(CONV_OUTPUT_ATOMIC_STRIDE as usize);
    shape.output_width as usize
        * shape.output_height as usize
        * surface_count
        * CONV_OUTPUT_ATOMIC_STRIDE as usize
}

fn require_single_conv_task(shape: &ConvShape) -> ConvTask {
    let tasks = plan_conv_tasks(shape).expect("build_conv_regcmd: invalid convolution task plan");
    assert!(
        tasks.len() == 1,
        "build_conv_regcmd: convolution requires {} CBUF height splits; \
         use build_conv_regcmd_tasks and device::submit_tasks",
        tasks.len()
    );
    tasks[0]
}

/// Builds one regcmd buffer per CBUF height split.
///
/// Allocate each returned vector in its own command buffer, call
/// [`link_regcmd_tasks`] with their DMA addresses, and submit the
/// `(address, command_count)` pairs together through
/// [`crate::rocket::device::submit_tasks`]. Keeping the splits in one kernel
/// job is required for `reuse_weights` tasks to observe the preceding task's
/// CBUF contents.
pub fn build_conv_regcmd_tasks(
    shape: &ConvShape,
    bufs: &ConvBuffers,
) -> Result<Vec<Vec<RegCmd>>, &'static str> {
    let tasks = plan_conv_tasks(shape)?;
    for task in &tasks {
        bufs.input_addr
            .checked_add(task.input_offset_bytes)
            .ok_or("input task DMA address exceeds u32")?;
        bufs.output_addr
            .checked_add(task.output_offset_bytes)
            .ok_or("output task DMA address exceeds u32")?;
    }

    let task_count = tasks.len();
    Ok(tasks
        .iter()
        .map(|task| {
            let mut cmds = build_conv_cna_core_dpu_dpu_rdma(shape, bufs, task, 2, None);
            push_kick_for_task_count(
                &mut cmds,
                KICK_CNA | KICK_CORE | KICK_DPU | KICK_DPU_RDMA,
                task_count,
            );
            cmds
        })
        .collect())
}

fn task_link_trailer_index(commands: &[RegCmd]) -> Result<usize, &'static str> {
    let is_register = |command: &RegCmd, domain: u32, offset: u32| {
        ((command.0 >> 48) & 0xff) == u64::from(domain) && (command.0 & 0xffff) == u64::from(offset)
    };
    if commands.len() < 4 {
        return Err("regcmd task is too short to contain a PC link trailer");
    }

    for index in (0..=commands.len() - 4).rev() {
        if is_register(&commands[index], DOMAIN_PC, REG_PC_BASE_ADDRESS)
            && is_register(&commands[index + 1], DOMAIN_PC, REG_PC_REGISTER_AMOUNTS)
            && commands[index + 2].0 == 0x0041_0000_0000_0000
            && is_register(&commands[index + 3], 0x81, REG_PC_OPERATION_ENABLE)
        {
            return Ok(index);
        }
    }
    Err("regcmd task has no PC link trailer")
}

/// Patches Mesa's embedded next-task PC links after command-buffer DMA
/// addresses are known.
///
/// Multi-split operations use both the kernel's ordered task array and the
/// trailing `PC_BASE_ADDRESS`/`PC_REGISTER_AMOUNTS` pair in each regcmd.
/// Mesa patches those registers to the next task before submission so the
/// alternate ping-pong register group is prepared while the current task
/// runs. The final task retains its zero-valued link.
pub fn link_regcmd_tasks(
    tasks: &mut [Vec<RegCmd>],
    task_dma_addresses: &[u32],
) -> Result<(), &'static str> {
    if tasks.len() != task_dma_addresses.len() {
        return Err("regcmd task and DMA-address counts differ");
    }
    if tasks.len() <= 1 {
        return Ok(());
    }

    let trailer_indices = tasks
        .iter()
        .map(|commands| task_link_trailer_index(commands))
        .collect::<Result<Vec<_>, _>>()?;
    for index in 0..tasks.len() - 1 {
        let next_payload_commands = trailer_indices[index + 1];
        let next_register_amount = (next_payload_commands / 2).next_multiple_of(2);
        let next_register_amount = u32::try_from(next_register_amount)
            .map_err(|_| "next regcmd register amount exceeds u32")?;
        let trailer = trailer_indices[index];
        tasks[index][trailer] = RegCmd::new(
            DOMAIN_PC,
            REG_PC_BASE_ADDRESS,
            task_dma_addresses[index + 1],
        );
        tasks[index][trailer + 1] =
            RegCmd::new(DOMAIN_PC, REG_PC_REGISTER_AMOUNTS, next_register_amount);
    }
    Ok(())
}

/// Builds the historical one-task convolution command buffer.
///
/// Shapes that need CBUF splitting must use [`build_conv_regcmd_tasks`].
pub fn build_conv_regcmd(shape: &ConvShape, bufs: &ConvBuffers) -> Vec<RegCmd> {
    let task = require_single_conv_task(shape);
    let mut cmds = build_conv_cna_core_dpu_dpu_rdma(shape, bufs, &task, 2, None);
    push_kick(&mut cmds, KICK_CNA | KICK_CORE | KICK_DPU | KICK_DPU_RDMA);
    cmds
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

/// DMA addresses for a conv-then-LUT-activation pipeline. Unlike
/// `ConvThenPoolingBuffers` (whose DPU->PPU handoff stays on-chip), `Relu`/
/// `Relux` fuse into the conv's own DPU pass but sigmoid/tanh never do --
/// confirmed by live hardware tracing of the vendor runtime itself
/// (`rknpu-spelunking/NOTES.md`), which showed a real separate task
/// reading the conv's output back from memory, not on-chip pipelining.
/// `intermediate_addr` is that real round-trip buffer: pure inter-task DMA
/// memory the CPU never touches, so it doesn't need `prep_bo()`/`fini_bo()`
/// the way a job-boundary buffer would.
pub struct ConvThenLutBuffers {
    pub input_addr: u32,
    pub weights_addr: u32,
    pub bias_addr: u32,
    pub intermediate_addr: u32,
    pub output_addr: u32,
}

/// Builds a conv task (forced `Activation::None` -- LUT activations are
/// never fused into the conv's own DPU pass, see `ConvThenLutBuffers`'s
/// doc comment) chained into a standalone LUT task, matching the vendor
/// runtime's own real routing for conv->sigmoid/tanh. Returns
/// `(conv_cmds, lut_cmds)` -- submit both as one multi-task job via
/// `device::submit_tasks(fd, &[(conv_cmd_addr, conv_cmds.len() as u32), \
/// (lut_cmd_addr, lut_cmds.len() as u32)], ...)`, not as two separate
/// jobs (the kernel dispatches task 2 only after task 1's own hardware
/// completion IRQ fires, so no CPU-side wait on `intermediate_addr` is
/// needed in between -- see `submit_tasks`'s doc comment).
pub fn build_conv_then_lut_regcmd(
    conv_shape: &ConvShape,
    lut_shape: &LutShape,
    table: LutTable,
    bufs: &ConvThenLutBuffers,
) -> (Vec<RegCmd>, Vec<RegCmd>) {
    assert!(
        matches!(conv_shape.activation, Activation::None),
        "build_conv_then_lut_regcmd: conv_shape.activation must be Activation::None -- \
         LUT activations always run as a separate task, never fused into the conv's own DPU pass"
    );
    let conv_cmds = build_conv_regcmd(
        conv_shape,
        &ConvBuffers {
            input_addr: bufs.input_addr,
            weights_addr: bufs.weights_addr,
            bias_addr: bufs.bias_addr,
            output_addr: bufs.intermediate_addr,
        },
    );
    let lut_cmds = build_lut_regcmd(
        lut_shape,
        &LutBuffers {
            input_addr: bufs.intermediate_addr,
            output_addr: bufs.output_addr,
        },
        table,
    );
    (conv_cmds, lut_cmds)
}

//===========================================================================
// Fully-connected (FC), Phase 2 of the ukernel roadmap. RKNN_TRM_Ch36
// (verified directly, ~line 426) confirms FC shares conv's exact
// CNA->CORE->DPU pipeline ("Fig. 36-5 -- Convolution flow 2 (zero-skipping/
// fully-connected path): Same skeleton as flow 1 [plain conv]... When
// zero-skipping is enabled: DPU's conv_mode must be 3, BS_CORE must be
// bypassed... the convolution accumulation itself is done by BN_CORE
// instead of BS_CORE"), i.e. zero-skip (`CnaFcCon0/1/2`, unused by any
// builder in this module) is an optional perf mode, not required for
// correctness -- FC v1 needs only a correctly-shaped conv with 1x1 spatial
// weights, reusing `build_conv_regcmd` unchanged rather than duplicating
// its CNA/CORE/DPU emission.
//
// The real risk, NOT sidestepped by a thin "just call build_conv_regcmd
// with weights_width=weights_height=1" wrapper: `build_conv_cna_core_dpu_
// dpu_rdma`'s `input_surface_stride = input_line_stride * (shape.
// input_height / 4 - 1)` underflows (`u32` `0 - 1`) for `input_height < 4`
// -- not a Rust-port bug, Mesa's own `rkt_task.c` has the same implicit
// `height >= 4` assumption via float math (`height/4.0 - 1`). A caller
// thinking purely in FC terms (an M x K input, no natural "height" at
// all) has no reason to know this, so `FcShape` doesn't expose an
// `input_height` field for a caller to get wrong -- `build_fc_regcmd`
// fixes it internally to `FC_SAFE_HEIGHT` (4, the smallest value that
// cannot underflow: `4 / 4 - 1 == 0`) and maps FC's M dimension onto
// `input_width` instead (per the roadmap plan's Phase 2 research
// sequence, tried first because `input_line_stride`/`input_surface_stride`
// only ever multiply width, never subtract from it).
//
// Consequence: with a 1x1 kernel, stride 1, no padding, every output
// pixel `(w, h)` depends only on input pixel `(w, h)` -- rows are fully
// independent, so only "row 0" (`h=0`) of the input needs real M*K data;
// rows 1..3 can be anything (garbage/uninitialized), they just waste
// compute on values nobody reads back. Callers must still allocate input/
// output buffers sized for the full `M x FC_SAFE_HEIGHT` cube (not just
// one row) and only read back row 0 of the output.
//
// UNCONFIRMED, not yet hardware-validated: whether M-as-width (rather than
// M-as-height, or some other mapping) is really what the hardware expects
// for a "batch of independent FC evaluations" -- and this is also the
// first hardware exercise anywhere in this module of `weights_width ==
// weights_height == 1` (every previously-validated conv shape used a 3x3
// kernel; note `build_conv_cna_core_dpu_dpu_rdma`'s `pad_con1` special-case
// for `weights_width >= 3` simply doesn't trigger here, falling back to the
// plain `input_zero_point.wrapping_sub(0x80)` path -- untested combination
// until now).
//===========================================================================

/// Fixed input/output height every `build_fc_regcmd` shape uses internally,
/// regardless of the logical FC shape's own M/K/N -- see this section's
/// module doc comment for why 4 specifically (avoids `build_conv_cna_core_
/// dpu_dpu_rdma`'s `input_height / 4 - 1` underflow) and why callers don't
/// get to override it.
const FC_SAFE_HEIGHT: u32 = 4;

/// Logical shape of a single fully-connected (matmul + bias) operation:
/// `[m, k] x [k, n] + [n] -> [m, n]`, quantized the same way `ConvShape` is.
/// Internally built as a 1x1-kernel conv with M mapped onto `input_width`
/// and height fixed at `FC_SAFE_HEIGHT` -- see this module's FC section
/// doc comment.
pub struct FcShape {
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub input_zero_point: u32,
    pub output_zero_point: u32,
    pub weights_zero_point: u32,
    pub input_scale: f32,
    pub weights_scale: f32,
    pub output_scale: f32,
    pub truncate_bits: u32,
    pub activation: Activation,
}

/// DMA addresses for the four buffers a single-task FC op needs -- same
/// four roles as `ConvBuffers`, just renamed to FC's own vocabulary.
/// `input_addr`/`output_addr` must point at buffers sized for the full
/// `m x FC_SAFE_HEIGHT` cube (not just one logical row) -- see this
/// module's FC section doc comment for why rows 1..3 are real but unread.
pub struct FcBuffers {
    pub input_addr: u32,
    pub weights_addr: u32,
    pub bias_addr: u32,
    pub output_addr: u32,
}

pub fn build_fc_regcmd(shape: &FcShape, bufs: &FcBuffers) -> Vec<RegCmd> {
    let conv_shape = ConvShape {
        input_width: shape.m,
        input_height: FC_SAFE_HEIGHT,
        input_channels: shape.k,
        output_width: shape.m,
        output_height: FC_SAFE_HEIGHT,
        output_channels: shape.n,
        weights_width: 1,
        weights_height: 1,
        stride: 1,
        depthwise: false,
        input_zero_point: shape.input_zero_point,
        output_zero_point: shape.output_zero_point,
        weights_zero_point: shape.weights_zero_point,
        input_scale: shape.input_scale,
        weights_scale: shape.weights_scale,
        output_scale: shape.output_scale,
        truncate_bits: shape.truncate_bits,
        activation: shape.activation,
        precision: Precision::Int8,
    };
    let conv_bufs = ConvBuffers {
        input_addr: bufs.input_addr,
        weights_addr: bufs.weights_addr,
        bias_addr: bufs.bias_addr,
        output_addr: bufs.output_addr,
    };
    build_conv_regcmd(&conv_shape, &conv_bufs)
}

//===========================================================================
// Pooling (standalone PPU, "flying mode" -- TRM Ch.36 Fig 36-6): PPU_RDMA
// reads the input straight from memory and feeds PPU directly, bypassing
// CNA/CORE/DPU entirely. There is NO Mesa/Teflon reference for this path
// (`rkt_ml.c` only ever implements convolution) -- every field below is
// derived from the PPU/PPU_RDMA register layout in `builders/ppu.rs` /
// `builders/ppu_rdma.rs` (bindgen'd from Mesa's own `registers.xml`, see
// builders.rs's DOMAIN_* comment) plus the TRM Ch.36 §4.6/§4.7 prose and
// `build_conv_regcmd`'s established conventions (N-1 encoding on every
// *_RDMA/CORE/DPU cube dimension, the same four-entry PC kick tail). NONE
// of the following has been hardware-validated yet -- see the UNCONFIRMED
// markers below and iree-rocket-hal/tests/pooling_hw.rs's doc comment for
// the sweep tests needed before trusting this in production:
//
// - RESOLVED (hardware-confirmed, real RK3588, via
//   `pooling_method_encoding_discovery`): `PoolingMethod`'s bit encoding is
//   Avg=0, Max=1, Min=2 -- see `PoolingMethod::bits()`'s doc comment.
// - PPU_RDMA's `src_line_stride`/`src_surf_stride` and PPU's
//   `dst_surf_stride`/`misc_ctrl.surf_len` formulas -- derived by analogy
//   to CNA's input-side and DPU's output-side stride math in
//   `build_conv_regcmd`; still not independently confirmed against a
//   non-uniform/asymmetric shape, but exercised without issue by every
//   `completes_and_output_tracks_input` test using this path.
// - RESOLVED (was open when this task hung real hardware, now hardware
//   re-confirmed working -- all of `pooling_hw.rs`'s tests pass): the kick
//   used to fire `build_conv_regcmd`'s fixed CNA/CORE/DPU/DPU_RDMA bitmask
//   unconditionally, which never actually enabled PPU/PPU_RDMA for this
//   task -- the root cause of the original hang. Now kicks `KICK_PPU |
//   KICK_PPU_RDMA` only, confirmed against a real rknn-toolkit2-compiled
//   pooling.rknn capture (see NOTES.md in rknpu-spelunking) and against
//   real hardware.
// - SUPERSEDED: this standalone-only shape is not actually what the real
//   vendor compiler emits at all -- the same NOTES.md capture shows a
//   *third* shape (a real CNA/CORE/DPU bypass task, memory round-trip,
//   *then* a separately-kicked standalone-PPU task) that matches neither
//   this path nor the pipelined `dpu_flyin` path below. See
//   `build_pooling_via_dpu_bypass_regcmd` further down, which is the
//   recommended default going forward; this function is kept for hardware
//   comparison/reference.
//===========================================================================

/// Hardware-confirmed bit encoding (`iree-rocket-hal/tests/pooling_hw.rs`'s
/// `pooling_method_encoding_discovery`, real RK3588, via
/// `build_pooling_regcmd`'s standalone-flying path): a half-10/half-200
/// input produced raw=0 -> 249, raw=1 -> 250, raw=2 -> 248 -- i.e. raw=1 is
/// the real max, raw=2 is the real min, raw=0 sits in between (avg). The
/// original guess (Max=0, Min=1, Avg=2) had max and min swapped relative to
/// avg; corrected below.
#[derive(Clone, Copy)]
pub enum PoolingMethod {
    Max,
    Min,
    Avg,
}

impl PoolingMethod {
    fn bits(self) -> u32 {
        match self {
            PoolingMethod::Avg => 0,
            PoolingMethod::Max => 1,
            PoolingMethod::Min => 2,
        }
    }
}

/// Logical shape of a single standalone pooling operation. Single-task
/// only, no CBUF-budget splitting to worry about (PPU has no CBUF -- that
/// concern is CNA/CORE-specific), no `index_en` output wiring yet (that
/// needs a second output buffer for argmax/argmin positions, not plumbed
/// here).
pub struct PoolingShape {
    pub input_width: u32,
    pub input_height: u32,
    pub input_channels: u32,
    pub output_width: u32,
    pub output_height: u32,
    pub output_channels: u32,
    pub kernel_width: u32,
    pub kernel_height: u32,
    pub stride_x: u32,
    pub stride_y: u32,
    pub method: PoolingMethod,
    pub pad_left: u32,
    pub pad_top: u32,
    pub pad_right: u32,
    pub pad_bottom: u32,
    /// Fill value for padded taps (e.g. -inf-ish for max, 0 for avg --
    /// caller's responsibility to pick something sane for `method`).
    pub pad_value: u32,
    /// Fused post-processing for this pooling op, per [[Activation]]'s own
    /// doc comment. PPU itself has no ALU/activation capability at all
    /// (confirmed by enumerating every register in `builders/ppu.rs`) --
    /// this field is only meaningful on a pooling path that runs a real DPU
    /// stage ahead of PPU. `build_pooling_via_dpu_bypass_regcmd` applies it
    /// to that DPU stage's BS core (the same fusion point Phase 1 validated
    /// for conv); `build_pooling_regcmd`'s pure standalone-flying path (no
    /// DPU at all) cannot honor this and asserts it's `Activation::None`
    /// rather than silently ignoring a non-`None` value.
    pub activation: Activation,
}

/// DMA addresses for the two buffers a standalone pooling op needs.
pub struct PoolingBuffers {
    pub input_addr: u32,
    /// NOTE: `PPU_DST_BASE_ADDR` is a genuinely 28-bit hardware field
    /// (unlike every other block's 32-bit `*_BASE_ADDR` registers -- see
    /// `PpuDstBaseAddr::dst_base_addr`'s `Bits<28>` parameter, confirmed
    /// against the bindgen'd header, not a guess). A `CREATE_BO`
    /// allocation whose `dma_address` lands >= 0x1000_0000 (256MiB) into
    /// DMA-visible space cannot be addressed by PPU's output stage as-is.
    /// `build_pooling_regcmd` asserts this explicitly (a clear panic
    /// message) rather than letting `Bits::<28>::new` fail with its
    /// generic "value exceeds designated bit width" message.
    ///
    /// UPDATE (post first hardware run): this was wrong -- writing the raw
    /// byte address here produced a job that completed (no hang, PREP_BO
    /// returned) but left the *entire* output buffer at zero (see
    /// pooling_hw.rs's `pooling_dump_full_output_buffer`, added
    /// specifically to check this). The 28-bit width is the same
    /// bits[31:4] convention TRM Ch.36 documents explicitly for
    /// `pc_base_address` ("bits 31:4 regcmd DMA address") -- a 32-bit byte
    /// address shifted right by 4 (assuming 16-byte alignment) always fits
    /// exactly in 28 bits with zero waste, unlike a genuine "only 256MiB
    /// addressable" hardware limit (which would be an odd, wasteful design
    /// choice and doesn't match anything else in this chapter). `DPU_DST_
    /// BASE_ADDR` being 32-bit and used unshifted in the hardware-
    /// validated `build_conv_regcmd` shows the shift isn't universal --
    /// only registers too narrow to hold a full byte address seem to use
    /// it, and PPU_DST_BASE_ADDR is the only such *_BASE_ADDR field found
    /// so far. `build_pooling_regcmd` now asserts 16-byte alignment
    /// (trivially satisfied by any page-aligned CREATE_BO allocation) and
    /// shifts right by 4 instead. NOT YET RE-CONFIRMED ON HARDWARE.
    pub output_addr: u32,
}

/// Standalone-flying PPU_RDMA+PPU register emission, factored out of
/// `build_pooling_regcmd` so `build_pooling_via_dpu_bypass_regcmd` (the real
/// two-kick vendor shape, see its own doc comment) can reuse the exact same
/// PPU-stage sequence, differing only in where PPU_RDMA fetches from (a
/// caller-supplied real input buffer for the standalone path; a bypass
/// conv's memory-written output for the two-kick path) and in which kick
/// mask the caller appends afterwards. Pure extraction -- no behavior
/// change versus the original single function.
fn build_ppu_standalone_flying(
    shape: &PoolingShape,
    input_addr: u32,
    output_addr: u32,
) -> Vec<RegCmd> {
    assert!(
        output_addr % 16 == 0,
        "build_ppu_standalone_flying: output_addr {output_addr:#x} is not 16-byte aligned -- \
         PPU_DST_BASE_ADDR is written as address >> 4 (see PoolingBuffers::output_addr's \
         doc comment), which silently drops any non-zero low 4 bits instead of failing \
         loudly, so this must be checked explicitly"
    );
    let dst_base_addr_shifted = output_addr >> 4;

    const ATOMIC_K_SIZE: u32 = 16;
    const FEATURE_ATOMIC_SIZE: u32 = 16;

    // Confirmed against RKNN_TRM_Ch36 (`rknpu-spelunking/chapter36.txt`):
    // PPU_RDMA_SRC_LINE_STRIDE/SRC_SURF_STRIDE are documented as "Pooling
    // cube shape width"/"Pooling cube shape area" -- plain pixel-count
    // values, not byte offsets -- exactly like DPU_DST_SURF_STRIDE just
    // below (also `31:4 RW`/`3:0 reserved`), whose already-correct formula
    // cancels the same way. The previous version omitted the `/
    // FEATURE_ATOMIC_SIZE` cancellation, writing a byte-stride value 16x
    // too large; row 0 of any pooling window (offset = base + 0*stride)
    // still read correctly regardless, which is why every hardware test
    // using this function against a CPU-filled *uniform-whole-buffer* input
    // (pooling_hw.rs) never caught it -- reading 16x too far still landed on
    // the same repeated fill byte anywhere within the buffer. It only
    // surfaced once a real DPU-written (not uniformly-filled) source buffer
    // was read in `pooling_via_dpu_bypass_hw.rs`, where rows beyond row 0
    // jumped past the DPU's actual (small) write footprint into untouched
    // buffer contents.
    let src_line_stride = (shape.input_width * ATOMIC_K_SIZE) / FEATURE_ATOMIC_SIZE;
    let src_surf_stride =
        (shape.input_width * ATOMIC_K_SIZE * shape.input_height) / FEATURE_ATOMIC_SIZE;

    // UNCONFIRMED, derived by analogy to DPU's output-side stride math in
    // build_conv_regcmd -- see module doc comment.
    let dst_surf_stride =
        (shape.output_width * ATOMIC_K_SIZE * shape.output_height) / FEATURE_ATOMIC_SIZE;
    let surf_len = shape.output_width * shape.output_height;

    // Average-pooling's divide-as-multiply trick (TRM §4.6): precomputed
    // reciprocal of the kernel dimension, x2^16. Always computed (not just
    // under Avg) -- build_conv_regcmd's own precedent is to always fill
    // every register regardless of which branch is logically active.
    let recip_kernel_width = ((1u64 << 16) / shape.kernel_width as u64) as u32;
    let recip_kernel_height = ((1u64 << 16) / shape.kernel_height as u64) as u32;

    let mut cmds: Vec<RegCmd> = Vec::new();

    // ========================================================================
    // Ping-pong pointers -- same pattern/values as build_conv_regcmd's
    // DPU/DPU_RDMA pair, applied to the two blocks this op actually uses.
    // ========================================================================

    cmds.push(
        Register::<PpuSPointer>::new()
            .pointer_pp_mode(Bits::new(1))
            .executer_pp_en(Bits::new(1))
            .pointer_pp_en(Bits::new(1))
            .build(),
    );
    cmds.push(
        Register::<PpuRdmaSPointer>::new()
            .pointer_pp_mode(Bits::new(1))
            .executer_pp_en(Bits::new(1))
            .pointer_pp_en(Bits::new(1))
            .build(),
    );

    // ========================================================================
    // PPU_RDMA -- standalone read side, feeds PPU directly from memory.
    // ========================================================================

    cmds.push(
        Register::<PpuRdmaCubeInWidth>::new()
            .cube_in_width(Bits::new(shape.input_width - 1))
            .build(),
    );
    cmds.push(
        Register::<PpuRdmaCubeInHeight>::new()
            .cube_in_height(Bits::new(shape.input_height - 1))
            .build(),
    );
    cmds.push(
        Register::<PpuRdmaCubeInChannel>::new()
            .cube_in_channel(Bits::new(shape.input_channels - 1))
            .build(),
    );
    cmds.push(
        Register::<PpuRdmaSrcBaseAddr>::new()
            .src_base_addr(Bits::new(input_addr))
            .build(),
    );
    cmds.push(
        Register::<PpuRdmaSrcLineStride>::new()
            .src_line_stride(Bits::new(src_line_stride))
            .build(),
    );
    cmds.push(
        Register::<PpuRdmaSrcSurfStride>::new()
            .src_surf_stride(Bits::new(src_surf_stride))
            .build(),
    );
    cmds.push(zero::<PpuRdmaDataFormat>()); // in_precision = 0 (int8)

    // ========================================================================
    // PPU
    // ========================================================================

    cmds.push(
        Register::<PpuDataCubeInWidth>::new()
            .cube_in_width(Bits::new(shape.input_width - 1))
            .build(),
    );
    cmds.push(
        Register::<PpuDataCubeInHeight>::new()
            .cube_in_height(Bits::new(shape.input_height - 1))
            .build(),
    );
    cmds.push(
        Register::<PpuDataCubeInChannel>::new()
            .cube_in_channel(Bits::new(shape.input_channels - 1))
            .build(),
    );
    cmds.push(
        Register::<PpuDataCubeOutWidth>::new()
            .cube_out_width(Bits::new(shape.output_width - 1))
            .build(),
    );
    cmds.push(
        Register::<PpuDataCubeOutHeight>::new()
            .cube_out_height(Bits::new(shape.output_height - 1))
            .build(),
    );
    cmds.push(
        Register::<PpuDataCubeOutChannel>::new()
            .cube_out_channel(Bits::new(shape.output_channels - 1))
            .build(),
    );

    cmds.push(
        Register::<PpuOperationModeCfg>::new()
            .pooling_method(Bits::new(shape.method.bits()))
            .flying_mode(Bits::new(1)) // standalone via PPU_RDMA, not pipelined after DPU
            .index_en(Bits::new(0)) // no argmax/argmin output wiring yet
            .use_cnt(Bits::new(0))
            .notch_addr(Bits::new(0))
            .build(),
    );
    cmds.push(
        Register::<PpuPoolingKernelCfg>::new()
            .kernel_width(Bits::new(shape.kernel_width - 1))
            .kernel_height(Bits::new(shape.kernel_height - 1))
            .kernel_stride_width(Bits::new(shape.stride_x - 1))
            .kernel_stride_height(Bits::new(shape.stride_y - 1))
            .build(),
    );
    cmds.push(
        Register::<PpuRecipKernelWidth>::new()
            .recip_kernel_width(Bits::new(recip_kernel_width))
            .build(),
    );
    cmds.push(
        Register::<PpuRecipKernelHeight>::new()
            .recip_kernel_height(Bits::new(recip_kernel_height))
            .build(),
    );
    cmds.push(
        Register::<PpuPoolingPaddingCfg>::new()
            .pad_left(Bits::new(shape.pad_left))
            .pad_top(Bits::new(shape.pad_top))
            .pad_right(Bits::new(shape.pad_right))
            .pad_bottom(Bits::new(shape.pad_bottom))
            .build(),
    );
    cmds.push(
        Register::<PpuPaddingValue1Cfg>::new()
            .pad_value_0(Bits::new(shape.pad_value))
            .build(),
    );
    cmds.push(zero::<PpuPaddingValue2Cfg>());

    cmds.push(
        Register::<PpuDstBaseAddr>::new()
            .dst_base_addr(Bits::new(dst_base_addr_shifted))
            .build(),
    );
    cmds.push(
        Register::<PpuDstSurfStride>::new()
            .dst_surf_stride(Bits::new(dst_surf_stride))
            .build(),
    );
    cmds.push(zero::<PpuDataFormat>()); // proc_precision=0 (int8), dpu_flyin=0 (standalone)
    cmds.push(
        Register::<PpuMiscCtrl>::new()
            .burst_len(Bits::new(15))
            .nonalign(Bits::new(0))
            .mc_surf_out(Bits::new(0))
            .surf_len(Bits::new(surf_len))
            .build(),
    );

    cmds
}

/// Standalone-flying PPU pooling (TRM Ch.36 Fig 36-6): PPU_RDMA fetches
/// straight from `bufs.input_addr`, CNA/CORE/DPU untouched. Kicked with just
/// `KICK_PPU | KICK_PPU_RDMA`, confirmed against a real rknn-toolkit2-
/// compiled pooling.rknn capture's standalone-flying task (see NOTES.md
/// "Decoding a real regcmd program for a pooling-only op") -- previously
/// reused `build_conv_regcmd`'s fixed CNA/CORE/DPU/DPU_RDMA kick verbatim,
/// which never set PPU's/PPU_RDMA's enable bit at all and is what hung real
/// hardware. NOT YET RE-VALIDATED ON HARDWARE with this fix -- and per the
/// real capture, this standalone-only shape isn't actually what the vendor
/// compiler emits at all; see `build_pooling_via_dpu_bypass_regcmd` below
/// for that real two-kick shape. Kept for hardware comparison/reference,
/// not as the recommended default.
pub fn build_pooling_regcmd(shape: &PoolingShape, bufs: &PoolingBuffers) -> Vec<RegCmd> {
    assert!(
        matches!(shape.activation, Activation::None),
        "build_pooling_regcmd: fused activation ({:?}) requested but this standalone-flying \
         path has no DPU stage at all -- PPU has no ALU/activation capability of its own (see \
         PoolingShape::activation's doc comment). Use build_pooling_via_dpu_bypass_regcmd \
         instead if fused activation is needed.",
        shape.activation
    );
    let mut cmds = build_ppu_standalone_flying(shape, bufs.input_addr, bufs.output_addr);
    push_kick(&mut cmds, KICK_PPU | KICK_PPU_RDMA);
    cmds
}

//===========================================================================
// Pooling via a real DPU bypass stage, two separately-kicked tasks -- the
// shape a real rknn-toolkit2-compiled model actually emits (see NOTES.md's
// "Decoding a real regcmd program for a pooling-only op" and its follow-up
// "Checked against iree-rocket-hal/src/" section), matching NEITHER of the
// two paths above:
//
// 1. A real (but trivial/near-identity) CNA->CORE->DPU task, same sequence
//    `build_conv_regcmd` already proves on hardware
//    (`build_conv_cna_core_dpu_dpu_rdma`, `dpu_output_mode=2` i.e. outside/
//    memory, matching the real capture's `DPU_FEATURE_MODE_CFG.output_mode
//    =2`), writing its output to a real intermediate buffer
//    (`bufs.bypass_output_addr`). Kicked `KICK_CNA | KICK_CORE | KICK_DPU |
//    KICK_DPU_RDMA` -- despite the real capture's kick reading `0x0d` (no
//    DPU_RDMA bit), real hardware evidence overrides that reading: even
//    with genuinely separate submit()/prep_bo() calls per task, omitting
//    DPU_RDMA left the intermediate buffer completely unwritten (see the
//    kick-fix note further down). `CNA_WEIGHT_SIZE0/1/2`'s tiny values in
//    the real capture, vs. conv.rknn's real-kernel values, are consistent
//    with this stage doing a near-identity passthrough rather than real
//    conv math -- reflected here by `bypass_shape` being caller-supplied
//    rather than hardcoded, so a 1x1, zero-point=0, scale=1.0 identity-ish
//    shape can be passed in without this function assuming what "trivial"
//    means numerically.
// 2. A second, separately-kicked task: PPU_RDMA fetches from
//    `bufs.bypass_output_addr` (the first stage's real memory output, not
//    an external caller-supplied input) and PPU pools it exactly like
//    `build_pooling_regcmd`'s standalone path -- reuses
//    `build_ppu_standalone_flying` verbatim. Kicked `KICK_PPU |
//    KICK_PPU_RDMA`, same as the real capture's `0x60` second kick.
//
// RESOLVED (real RK3588, first hardware round): the roadmap plan's Phase 0
// "one or two submit() calls?" question was originally guessed at "one" --
// reasoning that the real vendor capture's two kicks live inside one
// contiguous regcmd dump, and `drm_rocket_job`/`drm_rocket_task` (device.rs)
// carry a single opaque `regcmd`/`regcmd_count` per task, so nothing
// *structurally* prevented one combined blob. That guess was wrong: a
// single-`submit()` version of this (writing `PC_OPERATION_ENABLE=0x0d`
// then immediately `0x60`, no wait in between) left the bypass stage's
// intermediate buffer completely untouched (sentinel-fill diagnostic,
// `tests/pooling_via_dpu_bypass_hw.rs`'s
// `pooling_via_bypass_dump_intermediate_and_output_buffers`, real
// hardware) -- i.e. the CNA/CORE/DPU stage never got to actually run
// before the second write replaced which engines were enabled. The real
// vendor capture being one contiguous *file dump* doesn't mean the
// original driver issued it as one *submission* -- that inference doesn't
// hold. This function now returns two separate task command lists; the
// caller must `submit()` the first, `prep_bo()`-wait on
// `bufs.bypass_output_addr`'s buffer, *then* `submit()` the second --
// exactly like two independent dispatches, not one.
//===========================================================================

/// DMA addresses for the two-kick bypass-then-pool shape. `bypass_output_addr`
/// is a real intermediate buffer (unlike the pipelined path's on-chip
/// `dpu_flyin`, this shape genuinely round-trips through memory between the
/// two kicks, matching the real vendor capture) -- must be 16-byte aligned
/// for the same `PPU_RDMA` fetch-side reason `output_addr` must be (see
/// `PoolingBuffers::output_addr`'s doc comment), though PPU_RDMA's own
/// `src_base_addr` field is a full 32-bit address (unshifted), unlike
/// `PPU_DST_BASE_ADDR` -- 16-byte alignment is still required so the
/// bypass stage's own DPU output-side stride math stays consistent with
/// every other buffer in this module, not because PPU_RDMA's register width
/// demands it.
pub struct PoolingViaBypassBuffers {
    pub input_addr: u32,
    pub weights_addr: u32,
    pub bias_addr: u32,
    pub bypass_output_addr: u32,
    pub output_addr: u32,
}

/// Returns `(stage_1_bypass_cmds, stage_2_pooling_cmds)` -- two independent
/// tasks, NOT one combined blob (see the module doc comment above for why:
/// a single-`submit()` version of this left the bypass stage's output
/// completely unwritten on real hardware). The caller must `submit()` and
/// fully `prep_bo()`-wait on stage 1 (specifically on
/// `bufs.bypass_output_addr`'s buffer) before `submit()`ing stage 2 --
/// exactly like two ordinary, separately-fenced dispatches.
pub fn build_pooling_via_dpu_bypass_regcmd(
    bypass_shape: &ConvShape,
    pooling_shape: &PoolingShape,
    bufs: &PoolingViaBypassBuffers,
) -> (Vec<RegCmd>, Vec<RegCmd>) {
    assert!(
        bufs.bypass_output_addr % 16 == 0,
        "build_pooling_via_dpu_bypass_regcmd: bypass_output_addr {:#x} is not \
         16-byte aligned",
        bufs.bypass_output_addr
    );
    assert!(
        matches!(bypass_shape.activation, Activation::None),
        "build_pooling_via_dpu_bypass_regcmd: bypass_shape.activation must be None -- \
         fused activation for this pooling path is expressed on `pooling_shape.activation` \
         (the op's own logical shape), not on the internal near-identity bypass conv shape, \
         so there's one canonical place a caller/HAL layer needs to set it. This function \
         applies `pooling_shape.activation` to the bypass stage's BS core itself."
    );
    // Phase 3: the bypass stage is the only real DPU (BS-core) instance in
    // this pooling path, so fused activation rides on it exactly like
    // Phase 1's conv activation -- same enum, same BS-core fusion point,
    // already hardware-validated for conv. Only the `activation` field is
    // overridden here; every other geometry/quant field comes from the
    // caller-supplied `bypass_shape` unchanged.
    let bypass_shape_with_activation = ConvShape {
        activation: pooling_shape.activation,
        ..*bypass_shape
    };

    // Stage 1: real (near-identity) CNA->CORE->DPU task, output to memory.
    let bypass_task = require_single_conv_task(&bypass_shape_with_activation);
    let mut bypass_cmds = build_conv_cna_core_dpu_dpu_rdma(
        &bypass_shape_with_activation,
        &ConvBuffers {
            input_addr: bufs.input_addr,
            weights_addr: bufs.weights_addr,
            bias_addr: bufs.bias_addr,
            output_addr: bufs.bypass_output_addr,
        },
        &bypass_task,
        2, // outside/memory, matching the real capture
        None,
    );
    // KICK_DPU_RDMA included despite the real vendor capture's kick reading
    // 0x0d (no DPU_RDMA bit) -- hardware evidence from the two-submit fix
    // above overrides that reading: even with genuinely separate
    // submit()/prep_bo() calls, a 0x0d-kicked stage 1 left buf_mid
    // completely unwritten (sentinel-fill diagnostic, real hardware).
    // Every other memory-writing conv task in this module (build_conv_regcmd,
    // and even build_conv_then_pooling_regcmd's on-chip-routed stage) kicks
    // DPU_RDMA unconditionally -- DPU_RDMA's enable bit is apparently
    // required for the memory write-back to actually commit, regardless of
    // dpu_output_mode, contradicting the vendor capture's literal decoded
    // value (which may reflect something else about how the vendor
    // compiler's tasks share DPU_RDMA state across the larger multi-tile
    // program it came from, not a standalone single-task requirement).
    push_kick(
        &mut bypass_cmds,
        KICK_CNA | KICK_CORE | KICK_DPU | KICK_DPU_RDMA,
    );

    // Stage 2: standalone-flying PPU, fetching from stage 1's real output.
    let mut pooling_cmds =
        build_ppu_standalone_flying(pooling_shape, bufs.bypass_output_addr, bufs.output_addr);
    push_kick(&mut pooling_cmds, KICK_PPU | KICK_PPU_RDMA);

    (bypass_cmds, pooling_cmds)
}

//===========================================================================
// Pooling, pipelined directly after a real DPU stage (TRM Ch.36 Fig 36-4/
// 36-5's "do some pipeline surface ops" step, NOT the standalone/flying
// Fig 36-6 path `build_pooling_regcmd` implements above).
//
// Chosen after the standalone path hung real hardware (NPU job timeout +
// IOMMU "Enable stall request timed out"). Re-reading TRM Fig 36-6 itself
// ruled out one leading hypothesis for that hang -- the official standalone
// flow never configures CNA/CORE/DPU/DPU_RDMA at all, so leaving them
// zeroed (as `build_pooling_regcmd` already does) is not the bug. That left
// PPU_RDMA's own fetch-stride math (also unconfirmed) or the borrowed
// all-blocks "kick" as the remaining suspects -- the kick one is now
// confirmed and fixed (see `build_pooling_regcmd`'s kick-sequence comment
// and the `KICK_*` constants above `push_kick`); PPU_RDMA's stride math
// remains unconfirmed since this pipelined path doesn't use PPU_RDMA at
// all (see below).
//
// This path sidesteps the still-open stride-math question rather than
// resolving it: it runs a real, hardware-validated CNA->CORE->DPU
// convolution (the exact same sequence `build_conv_regcmd` already proves
// on hardware, reused via `build_conv_cna_core_dpu_dpu_rdma`) and routes
// DPU's output on-chip straight into PPU (`DPU_FEATURE_MODE_CFG.output_mode`
// bit0, `PPU_DATA_FORMAT.dpu_flyin`) instead of through PPU_RDMA. No
// PPU_RDMA fetch is used at all, so the kick is `KICK_CNA | KICK_CORE |
// KICK_DPU | KICK_DPU_RDMA | KICK_PPU` -- every block this task actually
// configures, PPU_RDMA correctly left both unconfigured and unkicked.
//
// Still NOT hardware-validated as of writing -- see tests/ for the
// hardware-in-the-loop test this needs before trusting it.
//===========================================================================

/// Pooling-specific parameters for the pipelined path. `conv_shape`'s own
/// `output_width`/`output_height`/`output_channels` become PPU's INPUT
/// shape (DPU's output, routed on-chip) -- this struct only carries
/// pooling's own output shape and kernel/pad/method parameters.
pub struct PipelinedPoolingShape {
    pub kernel_width: u32,
    pub kernel_height: u32,
    pub stride_x: u32,
    pub stride_y: u32,
    pub method: PoolingMethod,
    pub pad_left: u32,
    pub pad_top: u32,
    pub pad_right: u32,
    pub pad_bottom: u32,
    pub pad_value: u32,
    pub output_width: u32,
    pub output_height: u32,
    pub output_channels: u32,
}

/// DMA addresses for a conv-then-pipelined-pooling op. Unlike
/// `ConvBuffers`, there is no separate "conv output" address: DPU's output
/// is routed on-chip directly into PPU (`output_mode` bit0 only, bit1/
/// "outside" left clear), so it never needs a memory round-trip. Only the
/// final pooled result needs a real buffer.
pub struct ConvThenPoolingBuffers {
    pub input_addr: u32,
    pub weights_addr: u32,
    pub bias_addr: u32,
    pub output_addr: u32,
}

pub fn build_conv_then_pooling_regcmd(
    conv_shape: &ConvShape,
    pooling: &PipelinedPoolingShape,
    bufs: &ConvThenPoolingBuffers,
) -> Vec<RegCmd> {
    assert!(
        bufs.output_addr % 16 == 0,
        "build_conv_then_pooling_regcmd: output_addr {:#x} is not 16-byte aligned -- \
         PPU_DST_BASE_ADDR is written as address >> 4 (see build_pooling_regcmd's \
         PoolingBuffers::output_addr doc comment for why)",
        bufs.output_addr
    );

    const FEATURE_ATOMIC_SIZE: u32 = 16;
    const ATOMIC_K_SIZE: u32 = 16;

    // dpu_output_mode = 1: bit0 only ("to PPU", on-chip) -- bit1
    // ("outside"/memory) deliberately left clear, see module doc comment.
    // The conv-stage buffers (input/weights/bias) are real; output_addr
    // there is irrelevant (never written) since bit1 is clear.
    let conv_task = require_single_conv_task(conv_shape);
    let mut cmds = build_conv_cna_core_dpu_dpu_rdma(
        conv_shape,
        &ConvBuffers {
            input_addr: bufs.input_addr,
            weights_addr: bufs.weights_addr,
            bias_addr: bufs.bias_addr,
            output_addr: 0,
        },
        &conv_task,
        1,
        None,
    );

    // ========================================================================
    // PPU -- pipelined directly after DPU (dpu_flyin=1), no PPU_RDMA
    // involved. TRM Fig 36-4/36-5's "do some pipeline surface ops" step.
    // ========================================================================

    cmds.push(
        Register::<PpuSPointer>::new()
            .pointer_pp_mode(Bits::new(1))
            .executer_pp_en(Bits::new(1))
            .pointer_pp_en(Bits::new(1))
            .build(),
    );

    cmds.push(
        Register::<PpuDataCubeInWidth>::new()
            .cube_in_width(Bits::new(conv_shape.output_width - 1))
            .build(),
    );
    cmds.push(
        Register::<PpuDataCubeInHeight>::new()
            .cube_in_height(Bits::new(conv_shape.output_height - 1))
            .build(),
    );
    cmds.push(
        Register::<PpuDataCubeInChannel>::new()
            .cube_in_channel(Bits::new(conv_shape.output_channels - 1))
            .build(),
    );
    cmds.push(
        Register::<PpuDataCubeOutWidth>::new()
            .cube_out_width(Bits::new(pooling.output_width - 1))
            .build(),
    );
    cmds.push(
        Register::<PpuDataCubeOutHeight>::new()
            .cube_out_height(Bits::new(pooling.output_height - 1))
            .build(),
    );
    cmds.push(
        Register::<PpuDataCubeOutChannel>::new()
            .cube_out_channel(Bits::new(pooling.output_channels - 1))
            .build(),
    );

    cmds.push(
        Register::<PpuOperationModeCfg>::new()
            .pooling_method(Bits::new(pooling.method.bits()))
            .flying_mode(Bits::new(0)) // pipelined after DPU, not standalone
            .index_en(Bits::new(0))
            .use_cnt(Bits::new(0))
            .notch_addr(Bits::new(0))
            .build(),
    );
    cmds.push(
        Register::<PpuPoolingKernelCfg>::new()
            .kernel_width(Bits::new(pooling.kernel_width - 1))
            .kernel_height(Bits::new(pooling.kernel_height - 1))
            .kernel_stride_width(Bits::new(pooling.stride_x - 1))
            .kernel_stride_height(Bits::new(pooling.stride_y - 1))
            .build(),
    );

    let recip_kernel_width = ((1u64 << 16) / pooling.kernel_width as u64) as u32;
    let recip_kernel_height = ((1u64 << 16) / pooling.kernel_height as u64) as u32;
    cmds.push(
        Register::<PpuRecipKernelWidth>::new()
            .recip_kernel_width(Bits::new(recip_kernel_width))
            .build(),
    );
    cmds.push(
        Register::<PpuRecipKernelHeight>::new()
            .recip_kernel_height(Bits::new(recip_kernel_height))
            .build(),
    );
    cmds.push(
        Register::<PpuPoolingPaddingCfg>::new()
            .pad_left(Bits::new(pooling.pad_left))
            .pad_top(Bits::new(pooling.pad_top))
            .pad_right(Bits::new(pooling.pad_right))
            .pad_bottom(Bits::new(pooling.pad_bottom))
            .build(),
    );
    cmds.push(
        Register::<PpuPaddingValue1Cfg>::new()
            .pad_value_0(Bits::new(pooling.pad_value))
            .build(),
    );
    cmds.push(zero::<PpuPaddingValue2Cfg>());

    // Tested both this (>>4, by analogy to PC_BASE_ADDRESS's documented
    // bits[31:4] convention) and the raw address directly against real
    // hardware with a small (0x3000) buffer address -- both produced
    // identical zero-effect results, ruling out address encoding as the
    // actual blocker. Keeping the shifted form since it's still the more
    // principled default absent evidence either way.
    cmds.push(
        Register::<PpuDstBaseAddr>::new()
            .dst_base_addr(Bits::new(bufs.output_addr >> 4))
            .build(),
    );
    let ppu_dst_surf_stride =
        (pooling.output_width * ATOMIC_K_SIZE * pooling.output_height) / FEATURE_ATOMIC_SIZE;
    cmds.push(
        Register::<PpuDstSurfStride>::new()
            .dst_surf_stride(Bits::new(ppu_dst_surf_stride))
            .build(),
    );
    cmds.push(
        Register::<PpuDataFormat>::new()
            .dpu_flyin(Bits::new(1)) // source is DPU's on-chip output, not PPU_RDMA
            .build(),
    );
    let surf_len = pooling.output_width * pooling.output_height;
    cmds.push(
        Register::<PpuMiscCtrl>::new()
            .burst_len(Bits::new(15))
            .nonalign(Bits::new(0))
            .mc_surf_out(Bits::new(0))
            .surf_len(Bits::new(surf_len))
            .build(),
    );

    // Previously added an explicit PPU_OPERATION_ENABLE write here (PPU's
    // own DOMAIN_PPU-tagged register) because the shared kick's fixed
    // CNA/CORE/DPU/DPU_RDMA bitmask never set PPU's enable bit, and there
    // was no evidence either mechanism actually enabled PPU. There's now
    // real evidence: a real rknn-toolkit2-compiled pooling.rknn capture
    // (see NOTES.md in rknpu-spelunking, "Decoding a real regcmd program
    // for a pooling-only op") never writes any per-block *_OPERATION_ENABLE
    // register for *any* block, including PPU -- only the broadcast PC
    // kick, with a bitmask scoped to exactly the blocks that task uses.
    // Dropped the per-block write and folded KICK_PPU into the kick below
    // instead, matching that real pattern. NOT YET HARDWARE-VALIDATED.
    push_kick(
        &mut cmds,
        KICK_CNA | KICK_CORE | KICK_DPU | KICK_DPU_RDMA | KICK_PPU,
    );

    cmds
}

//===========================================================================
// Softmax, Phase 5 of the ukernel roadmap. Research spike, NOT a complete
// hardware-validated op yet -- see `rknpu-spelunking/NOTES.md`'s softmax
// Phase 5 section (all 5 "Follow-up"s) and
// `project_softmax_phase5_status.md` in this repo's memory for the full
// derivation. Two of three pieces are confirmed:
//
// 1. exp(x) itself is the standalone DPU LUT path above, unchanged --
//    `LutTable::exp()` plus `build_conv_then_lut_regcmd`/`build_lut_regcmd`
//    already handle it, no new code needed.
// 2. Max-reduction is a confirmed **N-stage tree** of (bypass-conv,
//    PPU-max-pool) pairs -- structurally identical to
//    `build_pooling_via_dpu_bypass_regcmd`'s single-stage shape, just
//    repeated with each stage's output feeding the next stage's input.
//    Live, self-consistent hardware capture found exactly 5 such stages
//    for one specific (32-channel) test shape, exactly matching the
//    static `.rknn` file's original prediction of 5
//    `PPU_OPERATION_MODE_CFG=0x11` blocks. `build_max_reduction_tree_regcmd`
//    below generalizes the PAIRED-STAGE SHAPE (not a specific stage count
//    or kernel-size schedule -- that part is caller-supplied, like every
//    other shape struct in this file) to arbitrary `N`.
//
// What's still missing, deliberately NOT implemented here (would be
// fabricating unconfirmed behavior rather than porting a proven recipe):
// - How the just-computed max actually gets subtracted from each element
//   before the exp LUT reads it. The confirmed exp dispatch's own
//   `BN_BASE_ADDR` was captured as 0, and its `SRC_BASE_ADDR` reads
//   directly from the graph's real conv output, not from the max-tree's
//   output buffer -- so the subtraction is NOT wired through DPU_RDMA's
//   per-position bias-buffer mechanism the way a first guess might
//   assume. Real data-flow between the max tree and the exp step is
//   still unmapped (see NOTES.md's Follow-up 5 "working hypothesis"
//   section).
// - The reciprocal/normalize (divide-by-sum) step. Exhaustively searched
//   for as a second DPU LUT/BN dispatch or a Sum/Avg-mode PPU block
//   across a complete, self-consistent capture -- found nowhere. Current
//   best guess is it isn't a distinct on-chip step in this graph at all
//   (folded into output requant, or done off-chip) -- nothing to port
//   until that's settled.
//
// There is deliberately no `build_softmax_regcmd` top-level entry point
// yet -- assembling one now would silently paper over the two gaps
// above with guesses. What's below is the reusable, structurally-
// confirmed piece (the max-reduction tree) plus a pointer to the
// already-existing exp LUT piece; wire them into a real op once the
// missing data-flow is confirmed on hardware.
//===========================================================================

/// One stage of a max-reduction tree: a near-identity bypass conv (same
/// role as `build_pooling_via_dpu_bypass_regcmd`'s `bypass_shape`) paired
/// with a PPU pooling stage that actually does the reduction. Caller
/// supplies both shapes explicitly per stage -- this module does not
/// guess a general "how many stages / what kernel size" schedule from an
/// element count; the one real hardware capture behind this code only
/// confirms the PAIRED-STAGE SHAPE generalizes, not any particular
/// tiling algorithm (see this section's module doc comment).
pub struct MaxReductionStage {
    pub bypass_shape: ConvShape,
    pub pooling_shape: PoolingShape,
}

/// DMA addresses for one stage of the tree. `output_addr` is this stage's
/// real memory output -- either the next stage's `bypass_input_addr`, or
/// (for the last stage) the tree's final reduced-max output.
pub struct MaxReductionStageBuffers {
    pub bypass_input_addr: u32,
    pub bypass_weights_addr: u32,
    pub bypass_bias_addr: u32,
    pub bypass_output_addr: u32,
    pub output_addr: u32,
}

/// Builds an `N`-stage max-reduction tree, each stage reusing
/// `build_pooling_via_dpu_bypass_regcmd` unchanged. Returns one
/// `(bypass_cmds, pooling_cmds)` pair per stage, in order -- the caller
/// must `submit()`+`prep_bo()`-wait through EVERY pair sequentially
/// (bypass stage, then pooling stage, then the next stage's bypass,
/// ...), exactly like `build_pooling_via_dpu_bypass_regcmd`'s own
/// single-stage discipline (see its doc comment for why: a combined
/// single-`submit_tasks()` job silently left one real hardware round
/// unwritten). Chaining `N>1` stages this way has the same *mechanism*
/// as the proven `N=1` case, but the specific 5-stage chain itself is
/// NOT YET independently hardware-validated end-to-end through this
/// exact function -- only the underlying single-stage building block is.
pub fn build_max_reduction_tree_regcmd(
    stages: &[MaxReductionStage],
    bufs: &[MaxReductionStageBuffers],
) -> Vec<(Vec<RegCmd>, Vec<RegCmd>)> {
    assert!(
        !stages.is_empty(),
        "build_max_reduction_tree_regcmd: at least one stage is required"
    );
    assert_eq!(
        stages.len(),
        bufs.len(),
        "build_max_reduction_tree_regcmd: one MaxReductionStageBuffers entry is required per stage"
    );
    stages
        .iter()
        .zip(bufs.iter())
        .map(|(stage, buf)| {
            build_pooling_via_dpu_bypass_regcmd(
                &stage.bypass_shape,
                &stage.pooling_shape,
                &PoolingViaBypassBuffers {
                    input_addr: buf.bypass_input_addr,
                    weights_addr: buf.bypass_weights_addr,
                    bias_addr: buf.bypass_bias_addr,
                    bypass_output_addr: buf.bypass_output_addr,
                    output_addr: buf.output_addr,
                },
            )
        })
        .collect()
}

#[cfg(test)]
mod conv_task_tests {
    use super::*;

    fn mobilenet_1x1(
        height: u32,
        width: u32,
        input_channels: u32,
        output_channels: u32,
    ) -> ConvShape {
        ConvShape {
            input_width: width,
            input_height: height,
            input_channels,
            output_width: width,
            output_height: height,
            output_channels,
            weights_width: 1,
            weights_height: 1,
            stride: 1,
            depthwise: false,
            input_zero_point: 0,
            output_zero_point: 0,
            weights_zero_point: 0,
            input_scale: 1.0,
            weights_scale: 1.0,
            output_scale: 1.0,
            truncate_bits: 0,
            activation: Activation::None,
            precision: Precision::Fp16,
        }
    }

    #[test]
    fn plans_mobilenet_fp16_height_splits() {
        let cases = [
            (mobilenet_1x1(112, 112, 32, 16), vec![45, 45, 22]),
            (mobilenet_1x1(56, 56, 96, 24), vec![30, 26]),
            (mobilenet_1x1(56, 56, 24, 144), vec![45, 11]),
        ];

        for (shape, expected_heights) in cases {
            let tasks = plan_conv_tasks(&shape).unwrap();
            assert_eq!(
                tasks
                    .iter()
                    .map(|task| task.output_height)
                    .collect::<Vec<_>>(),
                expected_heights
            );
            assert_eq!(tasks[0].input_banks, 10);
            assert_eq!(tasks[0].weights_banks, 2);
            assert!(!tasks[0].reuse_weights);
            assert!(tasks.iter().skip(1).all(|task| task.reuse_weights));
            assert_eq!(
                tasks.iter().map(|task| task.output_height).sum::<u32>(),
                shape.output_height
            );
        }
    }

    #[test]
    fn scratch_footprint_covers_padded_fp16_output_surfaces() {
        let shape = mobilenet_1x1(112, 112, 32, 16);
        let padded_channel_bytes =
            compute_task_output_channels(&shape) * shape.precision.bytes_per_element();
        let expected = shape.output_width as usize
            * shape.output_height as usize
            * padded_channel_bytes as usize;
        assert_eq!(conv_output_scratch_bytes(&shape), expected);
        assert!(
            conv_output_scratch_bytes(&shape)
                >= shape.output_width as usize
                    * shape.output_height as usize
                    * shape.output_channels as usize
                    * shape.precision.bytes_per_element() as usize
        );
    }

    #[test]
    fn plans_kernel_overlap_for_strided_3x3_splits() {
        let mut shape = mobilenet_1x1(100, 100, 32, 16);
        shape.weights_width = 3;
        shape.weights_height = 3;
        shape.stride = 2;
        shape.output_width = 49;
        shape.output_height = 49;

        let tasks = plan_conv_tasks(&shape).unwrap();
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0].input_height, 51);
        assert_eq!(tasks[0].output_height, 25);
        assert_eq!(tasks[0].retain_slices, 1);
        assert_eq!(tasks[1].input_top, 50);
        assert_eq!(tasks[1].input_height, 50);
        assert_eq!(tasks[1].output_top, 25);
        assert_eq!(tasks[1].output_height, 24);
        assert_eq!(tasks[1].overlap_slices, 1);
    }

    #[test]
    fn split_regcmd_builder_returns_one_buffer_per_task() {
        let shape = mobilenet_1x1(112, 112, 32, 16);
        let buffers = ConvBuffers {
            input_addr: 0x1000,
            weights_addr: 0x2000,
            bias_addr: 0x3000,
            output_addr: 0x4000,
        };
        let plans = plan_conv_tasks(&shape).unwrap();
        let command_buffers = build_conv_regcmd_tasks(&shape, &buffers).unwrap();

        assert_eq!(command_buffers.len(), plans.len());
        assert!(command_buffers.iter().all(|commands| !commands.is_empty()));

        for (plan, commands) in plans.iter().zip(&command_buffers) {
            let expected_input_addr = Register::<CnaFeatureDataAddr>::new()
                .feature_base_addr(Bits::new(buffers.input_addr + plan.input_offset_bytes))
                .build()
                .0;
            let expected_output_addr = Register::<DpuDstBaseAddr>::new()
                .dst_base_addr(Bits::new(buffers.output_addr + plan.output_offset_bytes))
                .build()
                .0;
            let expected_input_size = Register::<CnaDataSize0>::new()
                .datain_width(Bits::new(shape.input_width))
                .datain_height(Bits::new(plan.input_height))
                .build()
                .0;
            let expected_output_size = Register::<CoreDataoutSize0>::new()
                .dataout_width(Bits::new(shape.output_width - 1))
                .dataout_height(Bits::new(plan.output_height - 1))
                .build()
                .0;
            let mut cbuf = Register::<CnaCbufCon0>::new();
            cbuf.weight_bank(Bits::new(plan.weights_banks))
                .data_bank(Bits::new(plan.input_banks));
            if plan.reuse_weights {
                cbuf.weight_reuse(Bits::new(1));
            }

            for expected in [
                expected_input_addr,
                expected_output_addr,
                expected_input_size,
                expected_output_size,
                cbuf.build().0,
                RegCmd::new(DOMAIN_PC, REG_PC_BASE_ADDRESS, 0).0,
            ] {
                assert!(
                    commands.iter().any(|command| command.0 == expected),
                    "task {} is missing expected regcmd {expected:#018x}",
                    plan.index
                );
            }
        }
    }

    #[test]
    fn links_split_regcmd_ping_pong_trailers() {
        let shape = mobilenet_1x1(112, 112, 32, 16);
        let buffers = ConvBuffers {
            input_addr: 0x1000,
            weights_addr: 0x2000,
            bias_addr: 0x3000,
            output_addr: 0x4000,
        };
        let mut tasks = build_conv_regcmd_tasks(&shape, &buffers).unwrap();
        let addresses = [0x1000_0000, 0x1000_1000, 0x1000_2000];
        let trailers = tasks
            .iter()
            .map(|task| task_link_trailer_index(task).unwrap())
            .collect::<Vec<_>>();

        link_regcmd_tasks(&mut tasks, &addresses).unwrap();

        for index in 0..tasks.len() - 1 {
            let next_amount = (trailers[index + 1] / 2).next_multiple_of(2) as u32;
            assert_eq!(
                tasks[index][trailers[index]].0,
                RegCmd::new(DOMAIN_PC, REG_PC_BASE_ADDRESS, addresses[index + 1]).0
            );
            assert_eq!(
                tasks[index][trailers[index] + 1].0,
                RegCmd::new(DOMAIN_PC, REG_PC_REGISTER_AMOUNTS, next_amount).0
            );
        }
        let last = tasks.len() - 1;
        assert_eq!(
            tasks[last][trailers[last]].0,
            RegCmd::new(DOMAIN_PC, REG_PC_BASE_ADDRESS, 0).0
        );
        assert_eq!(
            tasks[last][trailers[last] + 1].0,
            RegCmd::new(DOMAIN_PC, REG_PC_REGISTER_AMOUNTS, 0).0
        );
    }

    #[test]
    fn rejects_mismatched_regcmd_link_address_count() {
        let shape = mobilenet_1x1(112, 112, 32, 16);
        let buffers = ConvBuffers {
            input_addr: 0x1000,
            weights_addr: 0x2000,
            bias_addr: 0x3000,
            output_addr: 0x4000,
        };
        let mut tasks = build_conv_regcmd_tasks(&shape, &buffers).unwrap();
        assert!(link_regcmd_tasks(&mut tasks, &[0x1000_0000]).is_err());
    }

    #[test]
    fn keeps_small_convolution_on_the_historical_single_task_path() {
        let shape = mobilenet_1x1(4, 4, 16, 16);
        let tasks = plan_conv_tasks(&shape).unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].input_banks, 1);
        assert_eq!(tasks[0].weights_banks, CBUF_BANKS - tasks[0].input_banks);
    }
}
