//! Mesa-faithful convolution and fully-connected regcmd construction.
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
//! This is the Mesa-derived lineage, distinct from
//! [`crate::rocket::conv`], which derives the same op from decoded vendor
//! `.rknn` captures instead. Keeping the two apart is deliberate: they are
//! independent evidence for the same hardware, and merging them would
//! erase which claim rests on which source.
//!
//! Scope, deliberately not general beyond what this repo currently needs:
//! - Convolutions larger than CBUF are split along output height using
//!   Mesa's `rkt_split_tasks()` algorithm. The public multi-task builder
//!   returns one regcmd buffer per split. The hardware-validated path submits
//!   and waits for each split as a separate DRM job.
//! - The serialized shape currently has no padding fields, so convolution
//!   planning supports zero-padding valid convolutions only.
//! - Element-wise `add_tensor` fusion lives in
//!   [`crate::rocket::elementwise`]; this module's core emission threads an
//!   `Option<&AddTensor>` through for it but owns none of that derivation.
//! - The `input_channels_real == 1 && output_channels_real > 1` "wide
//!   atomic" branch in `fill_task()` isn't implemented either -- no
//!   current client's shape reaches it, and porting an untested branch
//!   with no way to validate it would just be a latent bug. `build()`
//!   asserts this combination isn't requested rather than silently
//!   producing wrong output for it.

use crate::rocket::{
    activation::Activation,
    builders::{Bits, RegCmd, Register, cna::*, core::*, dpu::*, dpu_rdma::*},
    elementwise::{AddTensor, ew_add_cvt},
    regcmd::{
        KICK_CNA, KICK_CORE, KICK_DPU, KICK_DPU_RDMA, Precision, push_kick,
        push_kick_for_task_count, zero,
    },
};

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
    /// Current split planning leaves this false because task-to-task CBUF
    /// retention is not reliable through the mainline kernel driver.
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
    let (available_input_banks, available_weights_banks) =
        if weights_with_guard_bank < u64::from(CBUF_BANKS) {
            (
                u64::from(CBUF_BANKS) - weights_banks_required,
                weights_banks_required,
            )
        } else {
            // Mesa's partial-weights/partial-input split.
            (7, 5)
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
            // Reload weights for every split until CBUF retention across the
            // kernel driver's IRQ-mediated task transition is proven. On real
            // RK3588 hardware, enabling Mesa's later-task WEIGHT_REUSE bit
            // produced correct task 0 output followed by all-zero rows.
            reuse_weights: false,
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
pub(crate) fn build_conv_cna_core_dpu_dpu_rdma(
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
    // count must use the padded channel width programmed into
    // CNA_DATA_SIZE1, not the logical channel count. They are equivalent
    // for Mesa's int8 path after conversion to 16-byte feature atomics, but
    // differ for fp16 shapes such as C24: logical C24 is three atomics while
    // task C32 is four. Programming three produced a 42-entry row for W56,
    // causing each output row to begin 14 pixels early; four produces the
    // required 56 entries.
    let bpe = shape.precision.bytes_per_element();
    let input_data_entries = if input_channels_real_is_one {
        shape.input_width * shape.input_height
    } else if shape.input_width == 40 && shape.input_channels == 40 {
        40
    } else {
        (shape.input_width * 2 * (task_input_channels * bpe).div_ceil(FEATURE_ATOMIC_SIZE as u32))
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
    cmds.push(zero::<CoreReserved3030>());

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
    cmds.push(zero::<DpuReserved40c4>());

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

pub(crate) fn require_single_conv_task(shape: &ConvShape) -> ConvTask {
    let tasks = plan_conv_tasks(shape).expect("build_conv_regcmd: invalid convolution task plan");
    assert!(
        tasks.len() == 1,
        "build_conv_regcmd: convolution requires {} CBUF height splits; \
         use build_conv_regcmd_tasks and submit each task sequentially",
        tasks.len()
    );
    tasks[0]
}

/// Builds one regcmd buffer per CBUF height split.
///
/// Allocate each returned vector in its own command buffer and submit each
/// through [`crate::rocket::device::submit`], waiting for its output fence
/// before submitting the next split. Splits reload weights independently, so
/// no CBUF state must survive between jobs.
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

/// Builds the historical one-task convolution command buffer.
///
/// Shapes that need CBUF splitting must use [`build_conv_regcmd_tasks`].
pub fn build_conv_regcmd(shape: &ConvShape, bufs: &ConvBuffers) -> Vec<RegCmd> {
    let task = require_single_conv_task(shape);
    let mut cmds = build_conv_cna_core_dpu_dpu_rdma(shape, bufs, &task, 2, None);
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
// fixes it internally to `FC_PHYSICAL_HEIGHT` (4, the smallest value that
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
// output buffers sized for the full `M x FC_PHYSICAL_HEIGHT` cube (not just
// one row) and only read back row 0 of the output.
//
// Hardware-validated for int8 on RK3588 (2026-07-23,
// `tests/fc_hw.rs`): M-as-width preserves independent logical rows for both
// M=4 and non-square M=7; packed `[K,N]` weights select the intended output
// channel across multiple feature surfaces; all N=32 uniformly-computed
// outputs agree. These tests also cover the 1x1-kernel `pad_con1` fallback
// (the `weights_width >= 3` special case does not trigger).
//
// `FcShape::precision` also permits fp16 through the same already-validated
// multi-channel Conv path. Its exact-value FC-specific hardware tests live
// beside the int8 tests in `tests/fc_hw.rs`.
//===========================================================================

/// Fixed input/output height every `build_fc_regcmd` shape uses internally,
/// regardless of the logical FC shape's own M/K/N -- see this section's
/// module doc comment for why 4 specifically (avoids `build_conv_cna_core_
/// dpu_dpu_rdma`'s `input_height / 4 - 1` underflow) and why callers don't
/// get to override it.
pub const FC_PHYSICAL_HEIGHT: u32 = 4;

/// Logical shape of a single fully-connected (matmul + bias) operation:
/// `[m, k] x [k, n] + [n] -> [m, n]`, with the same numeric-domain fields as
/// `ConvShape`.
/// Internally built as a 1x1-kernel conv with M mapped onto `input_width`
/// and height fixed at `FC_PHYSICAL_HEIGHT` -- see this module's FC section
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
    pub precision: Precision,
}

/// DMA addresses for the four buffers a single-task FC op needs -- same
/// four roles as `ConvBuffers`, just renamed to FC's own vocabulary.
/// `input_addr`/`output_addr` must point at buffers sized for the full
/// `m x FC_PHYSICAL_HEIGHT` cube (not just one logical row) -- see this
/// module's FC section doc comment for why rows 1..3 are real but unread.
pub struct FcBuffers {
    pub input_addr: u32,
    pub weights_addr: u32,
    pub bias_addr: u32,
    pub output_addr: u32,
}

/// Returns the physical 1x1-convolution shape used to execute `shape`.
///
/// Runtime integrations can use this to apply the same validation and
/// storage calculations as the register-command builder without duplicating
/// FC's fixed-height lowering contract.
pub fn fc_as_conv_shape(shape: &FcShape) -> ConvShape {
    ConvShape {
        input_width: shape.m,
        input_height: FC_PHYSICAL_HEIGHT,
        input_channels: shape.k,
        output_width: shape.m,
        output_height: FC_PHYSICAL_HEIGHT,
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
        precision: shape.precision,
    }
}

pub fn build_fc_regcmd(shape: &FcShape, bufs: &FcBuffers) -> Vec<RegCmd> {
    let conv_shape = fc_as_conv_shape(shape);
    let conv_bufs = ConvBuffers {
        input_addr: bufs.input_addr,
        weights_addr: bufs.weights_addr,
        bias_addr: bufs.bias_addr,
        output_addr: bufs.output_addr,
    };
    build_conv_regcmd(&conv_shape, &conv_bufs)
}

#[cfg(test)]
mod conv_task_tests {
    use super::*;
    // The PC link trailer these tests inspect is emitted by the shared
    // kick/link code in `regcmd`, not by this module.
    use crate::rocket::{
        builders::DOMAIN_PC,
        regcmd::{link_regcmd_tasks, task_link_trailer_index},
        registers::{REG_PC_BASE_ADDRESS, REG_PC_REGISTER_AMOUNTS},
    };

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
            assert!(tasks.iter().all(|task| !task.reuse_weights));
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
    fn fp16_c24_programs_entries_for_the_padded_c32_row() {
        let shape = mobilenet_1x1(56, 56, 24, 144);
        let buffers = ConvBuffers {
            input_addr: 0x1000,
            weights_addr: 0x2000,
            bias_addr: 0x3000,
            output_addr: 0x4000,
        };
        let command_buffers = build_conv_regcmd_tasks(&shape, &buffers).unwrap();
        let expected = Register::<CnaCbufCon1>::new()
            .data_entries(Bits::new(56))
            .build()
            .0;

        assert!(
            command_buffers
                .iter()
                .all(|commands| { commands.iter().any(|command| command.0 == expected) })
        );
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
