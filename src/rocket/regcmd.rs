//! Shared, Mesa-faithful regcmd construction for a single conv operation.
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
//! Scope, deliberately not general beyond what any of this repo's test
//! clients actually need:
//! - Single-task operations only (`rkt_split_tasks()`'s "full weights,
//!   full input" branch) -- covers every shape small enough not to need
//!   multi-task CBUF splitting, which is all three clients in this repo.
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
    registers::{REG_DPU_LUT_ACCESS_DATA, REG_PC_OPERATION_ENABLE, REG_PC_REGISTER_AMOUNTS},
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
#[derive(Clone, Copy, Debug)]
pub enum Activation {
    None,
    Relu,
    /// Clamps to `cmp` in addition to the zero floor (`DPU_BS_RELUX_CMP_VALUE`).
    Relux {
        cmp: u32,
    },
    /// LUT-based nonlinear activation (sigmoid/tanh/...), routed through
    /// DPU's EW core rather than BS -- see `LutTable`'s doc comment. Not
    /// yet hardware-validated through this crate's own builder (the
    /// recipe below was decoded from a vendor-compiler capture, not run
    /// on real RK3588 through `build_conv_regcmd` before this variant
    /// existed).
    Lut(LutTable),
}

/// A DPU LUT (piecewise lookup table) configuration, as used by
/// `Activation::Lut`. Confirmed via `rknpu-spelunking/NOTES.md`'s
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
/// multiply stage on the `Activation::Lut` path -- reverse-engineered from
/// 5 independent int8-quantized `rknn-toolkit2` sigmoid captures at
/// different calibration scales (see `lut_bn_mul`'s doc comment and
/// `rknpu-spelunking/NOTES.md`). Not a documented hardware constant.
const LUT_BN_SCALE_K: f32 = 2596.513;

/// `BN_MUL_OPERAND`/`BN_MUL_SHIFT` for the `Activation::Lut` path:
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

/// `BN_ALU_OPERAND` for the `Activation::Lut` path: `-real_zero_point *
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

/// `OUT_CVT_SCALE`/`OUT_CVT_SHIFT` for the `Activation::Lut` path.
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

/// Logical shape of a single conv operation (`operation->*` in Mesa).
#[derive(Clone, Copy)]
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

/// Logical shape for a standalone DPU LUT pass. Unlike `Activation::Lut`
/// on `ConvShape`, this path matches the vendor compiler's decoded
/// sigmoid/tanh routing: DPU runs in flying mode, DPU_RDMA/MRDMA supplies
/// the input tensor directly from memory, and CNA/CORE are not kicked.
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
fn build_conv_cna_core_dpu_dpu_rdma(
    shape: &ConvShape,
    bufs: &ConvBuffers,
    dpu_output_mode: u32,
) -> Vec<RegCmd> {
    let input_channels_real_is_one = shape.input_channels == 1;
    assert!(
        !(input_channels_real_is_one && shape.output_channels > 1),
        "build_conv_regcmd: input_channels==1 && output_channels>1 needs \
         fill_task()'s \"wide atomic\" branch, not implemented here (no \
         current client's shape exercises it)"
    );

    // rkt_ml.h geometry constants.
    const CBUF_ENTRY_SIZE: u32 = 128; // CBUF_BANK_SIZE(32768) / CBUF_ENTRIES_PER_BANK(256)
    const CBUF_ENTRIES_PER_BANK: u32 = 256;
    const CBUF_BANKS: u32 = 12;
    const FEATURE_ATOMIC_SIZE: u32 = 16;
    const ATOMIC_K_SIZE: u32 = 16;

    // calc_entries_per_slice() (bpe=1, int8).
    let atomics_per_entry = CBUF_ENTRY_SIZE / FEATURE_ATOMIC_SIZE;
    let total_c_atomics = shape.input_channels.div_ceil(FEATURE_ATOMIC_SIZE);
    let last_c_atomics = total_c_atomics % atomics_per_entry;
    let int_c_entries = (total_c_atomics / atomics_per_entry) * shape.input_width;
    let frac_c_entries = if last_c_atomics == 3 {
        shape.input_width
    } else {
        (last_c_atomics * shape.input_width).div_ceil(atomics_per_entry)
    };
    let entries_per_slice = int_c_entries + frac_c_entries;

    // calc_input_banks().
    let input_banks = (entries_per_slice * shape.input_height).div_ceil(CBUF_ENTRIES_PER_BANK);
    assert!(
        input_banks < CBUF_BANKS,
        "build_conv_regcmd: shape needs {input_banks} input CBUF banks (only \
         {CBUF_BANKS} total) -- too big for the single-task path this \
         function implements (rkt_split_tasks()'s general multi-task \
         splitting branch isn't ported); shrink the operation or add that \
         support"
    );

    // rkt_split_tasks(): the single-task branch sets weight_banks to
    // CBUF_BANKS - input_banks directly -- NOT calc_weights_banks()'s own
    // result (that function's output only decides the reuse/no-reuse
    // branch elsewhere, never reaches CBUF_CON0 here).
    let weight_banks = CBUF_BANKS - input_banks;

    // fill_task(): channel/kernel counts round up to the hardware's
    // atomic granularity, independent of the op's logical shape.
    let task_input_channels = shape
        .input_channels
        .max(FEATURE_ATOMIC_SIZE)
        .next_multiple_of(FEATURE_ATOMIC_SIZE);
    let mut task_output_channels = shape.output_channels.max(32).next_multiple_of(32);
    if shape.depthwise {
        if shape.output_channels <= 32 {
            task_output_channels *= 2;
        }
        task_output_channels = task_output_channels.next_multiple_of(64);
    }
    let weights_kernels = if shape.depthwise {
        1
    } else {
        shape.output_channels.next_multiple_of(2)
    };

    // fill_task(): input_channels_real==1 && (output_channels_real>1 ||
    // addition) selects a "wide atomic" branch -- asserted unreachable
    // above, so this is always the plain else branch for every shape this
    // function actually gets called with.
    let line_stride = |w: u32| w * ATOMIC_K_SIZE;
    let input_line_stride = line_stride(shape.input_width) / 4;
    let input_surface_stride = input_line_stride * (shape.input_height / 4 - 1);
    let output_surface_stride =
        (line_stride(shape.output_width) * shape.output_height) / FEATURE_ATOMIC_SIZE;
    let surfaces_per_row =
        shape.output_width * shape.output_height * 2 * if shape.depthwise { 2 } else { 1 };

    // fill_task(): input_data_entries, three-way branch.
    let input_data_entries = if input_channels_real_is_one {
        shape.input_width * shape.input_height
    } else if shape.input_width == 40 && shape.input_channels == 40 {
        40
    } else {
        (shape.input_width * 2 * shape.input_channels.div_ceil(FEATURE_ATOMIC_SIZE)).div_ceil(8)
    };

    // rkt_regcmd.c: DPU_OUT_CVT_OFFSET/SCALE/SHIFT -- float-bits-based
    // fixed-point conversion (f32::to_bits() is bit-exact for C's fui()).
    let out_offset = shape.output_zero_point.wrapping_sub(0x80);
    let conv_scale = (shape.input_scale * shape.weights_scale) / shape.output_scale;
    let scale_bits = conv_scale.to_bits();
    let scale_exponent = ((scale_bits >> 23) & 0xff) as i32;
    let mut out_shift = 127 + 31 - 32 - scale_exponent + 16;
    if shape.truncate_bits > 0 {
        out_shift -= 1;
    }
    assert!(
        out_shift >= 0,
        "unsupported output conversion scale: conv_scale={conv_scale}, exponent={scale_exponent}, \
         computed out_shift={out_shift}"
    );
    let out_shift = out_shift as u32;
    let mut out_scale = ((scale_bits >> 9) & 0x7fff) + 1;
    if out_scale < (1 << 14) {
        out_scale |= 1 << 14;
    }

    let lut = match shape.activation {
        Activation::Lut(table) => Some(table),
        _ => None,
    };
    // Computed early (rather than inline in the BN block below) because
    // `DPU_DATA_FORMAT.bn_mul_shift_value_neg` -- emitted well before BN's
    // own registers -- needs the same shift value: confirmed via 4
    // independent int8-quantized sigmoid captures (see `lut_bn_mul`'s doc
    // comment) that this field always exactly mirrors `BN_MUL_CFG`'s own
    // `bn_mul_shift_value`, not a separately-derived quantity. Previously
    // missed entirely -- `DpuDataFormat` was emitted unconditionally zero
    // for every activation, silently leaving this field 0 whenever the
    // real shift was nonzero (which is virtually always).
    let lut_bn_mul_result = lut.map(|_| lut_bn_mul(shape.input_scale));

    let mut cmds: Vec<RegCmd> = Vec::new();

    // ========================================================================
    // CNA
    // ========================================================================

    let mut cbuf_con0_builder = Register::<CnaCbufCon0>::new();
    cbuf_con0_builder
        .weight_bank(Bits::new(weight_banks))
        .data_bank(Bits::new(input_banks));
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
            .datain_height(Bits::new(shape.input_height))
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
            .dataout_atomics(Bits::new(shape.output_width * shape.output_height))
            .build(),
    );

    cmds.push(
        Register::<CnaWeightSize0>::new()
            .weight_bytes(Bits::new(
                shape.weights_width * shape.weights_height * task_input_channels * weights_kernels,
            ))
            .build(),
    );
    cmds.push(
        Register::<CnaWeightSize1>::new()
            .weight_bytes_per_kernel(Bits::new(
                shape.weights_width * shape.weights_height * task_input_channels,
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

    if input_channels_real_is_one {
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
            .feature_base_addr(Bits::new(bufs.input_addr))
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
            .dma_height(Bits::new(shape.input_height))
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

    cmds.push(if input_channels_real_is_one {
        Register::<CnaCvtCon5>::new()
            .per_channel_cvt_en(Bits::new(65535))
            .build()
    } else {
        zero::<CnaCvtCon5>()
    });

    let mut pad_con1 = if shape.weights_width >= 3 && shape.input_zero_point == 0 {
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
    misc_cfg_builder.qd_en(Bits::new(1));
    if shape.depthwise {
        misc_cfg_builder.dw_en(Bits::new(1));
    }
    cmds.push(misc_cfg_builder.build());

    cmds.push(
        Register::<CoreDataoutSize0>::new()
            .dataout_width(Bits::new(shape.output_width - 1))
            .dataout_height(Bits::new(shape.output_height - 1))
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

    cmds.push(match lut_bn_mul_result {
        Some((_, bn_mul_shift)) => Register::<DpuDataFormat>::new()
            .bn_mul_shift_value_neg(Bits::new(bn_mul_shift))
            .build(),
        None => zero::<DpuDataFormat>(),
    });
    cmds.push(zero::<DpuOffsetPend>());
    // bit1 clear (not writing "outside"/memory) -> 0, see this function's
    // own doc comment.
    let dpu_writes_to_memory = dpu_output_mode & 0b10 != 0;
    cmds.push(
        Register::<DpuDstBaseAddr>::new()
            .dst_base_addr(Bits::new(if dpu_writes_to_memory {
                bufs.output_addr
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
            .height(Bits::new(shape.output_height - 1))
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
    //
    // *Except* for `Activation::Lut`: the vendor int8 capture used for
    // `lut_bn_mul`/`lut_out_cvt` has BS **fully** bypassed for the
    // LUT-using task (`BS_CFG=0x53` -- `bs_bypass`/`bs_alu_bypass`/
    // `bs_mul_bypass`/`bs_relu_bypass` all set, `BS_ALU_CFG`/`BS_MUL_CFG`
    // both zero), not just "relu/relux off" -- found by cross-checking
    // this register against the same capture after BN/OUT_CVT fixes alone
    // still didn't reveal a sigmoid/tanh difference. Bias-add-of-zero
    // should be a numeric no-op regardless, so this may not be
    // load-bearing, but it's a confirmed, real byte-for-byte discrepancy
    // from the vendor's own working recipe, worth matching exactly.
    let (bs_relu_bypass, bs_relux_en, bs_relux_cmp) = match shape.activation {
        Activation::None => (1, 0, 0),
        Activation::Relu => (0, 0, 0),
        Activation::Relux { cmp } => (0, 1, cmp),
        Activation::Lut(_) => (1, 0, 0),
    };
    cmds.push(match lut {
        Some(_) => Register::<DpuBsCfg>::new()
            .bs_bypass(Bits::new(1))
            .bs_alu_bypass(Bits::new(1))
            .bs_mul_bypass(Bits::new(1))
            .bs_relu_bypass(Bits::new(1))
            .build(),
        None => Register::<DpuBsCfg>::new()
            .bs_alu_algo(Bits::new(2))
            .bs_alu_src(Bits::new(1))
            .bs_relu_bypass(Bits::new(bs_relu_bypass))
            .bs_relux_en(Bits::new(bs_relux_en))
            .bs_mul_bypass(Bits::new(1))
            .build(),
    });
    cmds.push(zero::<DpuBsAluCfg>());
    cmds.push(zero::<DpuBsMulCfg>());
    cmds.push(
        Register::<DpuBsReluxCmpValue>::new()
            .bs_relux_cmp_dat(Bits::new(bs_relux_cmp))
            .build(),
    );
    cmds.push(match lut {
        Some(_) => Register::<DpuBsOwCfg>::new()
            .od_bypass(Bits::new(1))
            .build(),
        None if shape.depthwise => Register::<DpuBsOwCfg>::new()
            .size_e_0(Bits::new(3))
            .size_e_1(Bits::new(3))
            .size_e_2(Bits::new(3))
            .build(),
        None => Register::<DpuBsOwCfg>::new()
            .size_e_0(Bits::new(1))
            .size_e_1(Bits::new(1))
            .size_e_2(Bits::new(1))
            .build(),
    });
    cmds.push(match lut {
        Some(_) => zero::<DpuBsOwOp>(),
        None => Register::<DpuBsOwOp>::new()
            .ow_op(Bits::new(0x80 - shape.weights_zero_point))
            .build(),
    });
    cmds.push(
        Register::<DpuWdmaSize0>::new()
            .channel_wdma(Bits::new(task_output_channels - 1))
            .build(),
    );
    cmds.push(
        Register::<DpuWdmaSize1>::new()
            .height_wdma(Bits::new(shape.output_height - 1))
            .width_wdma(Bits::new(shape.output_width - 1))
            .build(),
    );

    // BN (batchnorm): genuinely fully bypassed, all four bits -- *except*
    // for `Activation::Lut`, where the vendor capture shows BN very much
    // *not* bypassed (`BN_CFG=0x00020040` plus a nonzero `BN_ALU_CFG`/
    // `BN_MUL_CFG`). First round of hardware testing without this (BN
    // unconditionally bypassed regardless of activation) dispatched
    // cleanly but produced output indistinguishable from `Activation::None`
    // for both sigmoid and tanh -- strong evidence BN is the stage that
    // converts the raw accumulator into the LUT's fixed `[-16384, 16384]`
    // x-domain, and skipping it left EW/LUT indexing on raw, unconverted
    // data (almost certainly saturating at one table edge regardless of
    // input).
    //
    // A single fp16-mode capture's literal bytes weren't enough here (its
    // BN_ALU_CFG turned out to just be -0.0f's bit pattern, an artifact of
    // fp16 mode, not a real int8-domain operand) -- `lut_bn_mul`/
    // `lut_bn_alu` below instead implement a formula reverse-engineered
    // from 5 independent int8-quantized (`do_quantization=true`) sigmoid
    // captures at different calibration scales (see
    // `rknpu-spelunking/NOTES.md`'s LUT section): `BN_MUL_OPERAND`/
    // `_SHIFT` reconstruct *exactly* for all 5 via `multiplier =
    // input_scale * LUT_BN_SCALE_K`, normalized so `operand =
    // round(multiplier * 2^shift)` lands in `[16384, 32768)`. `BN_ALU_
    // OPERAND = -real_zero_point * bn_mul_operand` matched exactly for 3
    // of the 5 (zero points -2, -128, 127); the other 2 (both zero_point
    // 42) were low by a consistent but unexplained ~1/16 factor -- a real,
    // open discrepancy, not resolved. It doesn't affect this crate's
    // existing zero-point-0 test shapes: `input_zero_point: 0` decodes
    // (via the same `wrapping_sub(0x80)` convention `pad_con1` already
    // uses below) to a real signed zero-point of -128, one of the 3
    // exactly-matching cases.
    match lut_bn_mul_result {
        Some((bn_mul_operand, bn_mul_shift)) => {
            let real_zero_point = shape.input_zero_point.wrapping_sub(0x80) as i8 as i32;
            assert!(
                lut_bn_alu_supports_zero_point(real_zero_point),
                "Activation::Lut only supports input_zero_point values whose decoded signed \
                 zero point is one of -128, -2, 0, or 127 for now; got raw input_zero_point={} \
                 (decoded {}). The BN_ALU formula is known not to match vendor captures for \
                 every zero point yet.",
                shape.input_zero_point,
                real_zero_point
            );
            let bn_alu_operand = lut_bn_alu(real_zero_point, bn_mul_operand);
            cmds.push(
                Register::<DpuBnCfg>::new()
                    .bn_alu_algo(Bits::new(2))
                    .bn_alu_src(Bits::new(0))
                    .bn_relux_en(Bits::new(0))
                    .bn_relu_bypass(Bits::new(1))
                    .bn_mul_prelu(Bits::new(0))
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
        }
        None => {
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
        }
    }

    // EW (elementwise): add_tensor == -1 branch (unconditional -- no
    // add_tensor support in this function). Fully bypassed unless
    // `Activation::Lut` requests the LUT path -- confirmed via
    // `rknpu-spelunking/NOTES.md`'s LUT decode: every non-LUT op in the
    // vendor capture has `EW_CFG=0x383` (all bypassed, matching the
    // `else` branch below byte-for-byte), while the sigmoid/tanh op flips
    // `EW_LUT_BYPASS` and `EW_BYPASS`. `EW_OP_BYPASS` is per-function in
    // the capture (`0` for sigmoid, `1` for tanh), so it is stored on
    // `LutTable` rather than inferred here.
    cmds.push(if let Some(table) = lut {
        Register::<DpuEwCfg>::new()
            .ew_relu_bypass(Bits::new(1))
            .ew_op_cvt_bypass(Bits::new(1))
            .ew_lut_bypass(Bits::new(0))
            .ew_op_bypass(Bits::new(table.ew_op_bypass as u32))
            .ew_bypass(Bits::new(0))
            .build()
    } else {
        Register::<DpuEwCfg>::new()
            .ew_relu_bypass(Bits::new(1))
            .ew_op_cvt_bypass(Bits::new(1))
            .ew_lut_bypass(Bits::new(1))
            .ew_op_bypass(Bits::new(1))
            .ew_bypass(Bits::new(1))
            .build()
    });
    cmds.push(zero::<DpuEwCvtOffsetValue>());
    cmds.push(
        Register::<DpuEwCvtScaleValue>::new()
            .ew_op_cvt_scale(Bits::new(1))
            .build(),
    );
    cmds.push(zero::<DpuEwReluxCmpValue>());

    let (out_cvt_scale, out_cvt_shift) = match lut {
        Some(_) => lut_out_cvt(shape.output_scale),
        None => (out_scale, out_shift - 1),
    };
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
            .surf_add(Bits::new(surfaces_per_row))
            .build(),
    );
    cmds.push(RegCmd::new(DOMAIN_DPU, 0x40c4, 0)); // TRM-mandated reserved write, no REG_DPU_* name

    match lut {
        Some(table) => {
            // Table upload: this crate always builds one self-contained
            // kicked task per call, unlike the vendor compiler's
            // preload-once-reuse-across-ops graph, so the table content
            // is (re-)uploaded inline here, ahead of the LUT_CFG/INFO/
            // START/END/SLOPE registers that reference it -- both are
            // part of the same task in our model, kicked together.
            push_lut_tables_and_config(&mut cmds, table);
        }
        None => {
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
        }
    }

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
            .height(Bits::new(shape.output_height - 1))
            .build(),
    );
    cmds.push(
        Register::<DpuRdmaDataCubeChannel>::new()
            .channel(Bits::new(task_output_channels - 1))
            .build(),
    );

    cmds.push(zero::<DpuRdmaSrcBaseAddr>()); // add_tensor == -1
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

    cmds.push(
        Register::<DpuRdmaErdmaCfg>::new()
            .erdma_disable(Bits::new(1))
            .build(),
    ); // add_tensor == -1
    cmds.push(zero::<DpuRdmaEwBaseAddr>());
    cmds.push(zero::<DpuRdmaEwSurfStride>());

    let mut rdma_feat_mode_builder = Register::<DpuRdmaFeatureModeCfg>::new();
    rdma_feat_mode_builder
        .burst_len(Bits::new(15))
        .mrdma_disable(Bits::new(1));
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
fn push_kick(cmds: &mut Vec<RegCmd>, enable_mask: u32) {
    cmds.push(RegCmd::new_raw(0x0)); // PC_BASE_ADDRESS placeholder (single task)
    cmds.push(RegCmd::new(DOMAIN_PC, REG_PC_REGISTER_AMOUNTS, 0));
    cmds.push(RegCmd::new_raw(0x0041000000000000)); // TRM: required immediately before op_en
    cmds.push(RegCmd::new(0x81, REG_PC_OPERATION_ENABLE, enable_mask));

    if cmds.len() % 2 != 0 {
        cmds.push(RegCmd::new_raw(0x0));
    }
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

pub fn build_conv_regcmd(shape: &ConvShape, bufs: &ConvBuffers) -> Vec<RegCmd> {
    let mut cmds = build_conv_cna_core_dpu_dpu_rdma(shape, bufs, 2); // 2 = outside/memory only, unchanged original behavior
    push_kick(&mut cmds, KICK_CNA | KICK_CORE | KICK_DPU | KICK_DPU_RDMA);
    cmds
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
    let mut bypass_cmds = build_conv_cna_core_dpu_dpu_rdma(
        &bypass_shape_with_activation,
        &ConvBuffers {
            input_addr: bufs.input_addr,
            weights_addr: bufs.weights_addr,
            bias_addr: bufs.bias_addr,
            output_addr: bufs.bypass_output_addr,
        },
        2, // outside/memory, matching the real capture
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
    let mut cmds = build_conv_cna_core_dpu_dpu_rdma(
        conv_shape,
        &ConvBuffers {
            input_addr: bufs.input_addr,
            weights_addr: bufs.weights_addr,
            bias_addr: bufs.bias_addr,
            output_addr: 0,
        },
        1,
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
