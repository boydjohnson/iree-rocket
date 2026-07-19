//! NPU diagnostic: minimal 1x1 conv, built as a faithful field-for-field
//! port of Mesa's real gallium driver (`mesa-rocket-userspace/rkt_regcmd.c`,
//! `fill_first_regcmd()` + `rkt_task.c`, `fill_task()`/`rkt_split_tasks()`),
//! specialized for this file's one fixed operation shape (4x4 spatial,
//! 1 input channel, 1 output channel, 1x1 kernel, stride 1, no padding, no
//! depthwise, no bias/batchnorm/elementwise-add tensors).
//!
//! Why the full rewrite: a live comparison against a real, hardware-
//! confirmed-working regcmd stream (captured via `ROCKET_DEBUG=dump_bos`
//! while running Mesa's own Teflon TFLite delegate through the same kernel
//! driver on this board -- see rknpu-spelunking/NOTES.md, "Real Mesa
//! rocket/Teflon driver built and run end-to-end") showed this file was
//! missing roughly half of the real register program (~56 entries here vs.
//! ~130 in the real capture) -- entire register families (DCOMP_AMOUNT0-15,
//! all DPU_BS_*/DPU_WDMA_*/DPU_EW_OP_VALUE_*/DPU_LUT_*, most of DPU_RDMA_*,
//! CNA_FC_*) were simply never written. Since these are real MMIO registers
//! with persistent hardware state, not writing them left CNA/DPU/DPU_RDMA
//! running on stale/undefined state from whatever job last touched that
//! core -- confirmed via `scripts/rocket_kernel_trace.bt`: the real Teflon
//! client's jobs get a genuine DPU completion IRQ on 210/210 tries, while
//! this file's job never triggers `rocket_job_irq_handler` even once before
//! `drm_sched`'s 500ms timeout gives up. Everything upstream of hardware
//! register programming (ioctl dispatch, DRM scheduling, PM, IOMMU) was
//! already confirmed identical and working in both cases.
//!
//! Beyond raw completeness, going line-by-line through `fill_first_regcmd()`
//! also caught several previously-guessed field VALUES that turned out
//! wrong (not just missing registers) -- all cited inline at the write site
//! below, but the headline ones: CORE_MISC_CFG needs QD_EN(1), which this
//! file never set (it set an invented `proc_precision` field instead, which
//! Mesa never touches on this register); DPU_BS_CFG is never actually
//! bypassed by Mesa the way this file assumed -- it always runs the bias-
//! subtract ALU (ALU_ALGO(2)|ALU_SRC(1)), which in turn means a real
//! (if zero-filled) biases buffer has to exist and be wired into
//! DPU_RDMA_RDMA_BS_BASE_ADDR, something this file never allocated at all;
//! several CNA_CONV_CON1/CONV_CON2/DPU_DATA_FORMAT/DPU_RDMA_FEATURE_MODE_CFG
//! fields this file set (conv_mode, in/proc_precision, cmd_fifo_srst) are
//! never touched by Mesa's real code at all for this register/branch
//! combination -- they were carried over from an earlier cross-check against
//! a *vendor*-compiled regcmd (conv.rknn, a different compiler/toolchain
//! than Mesa entirely), which is a lower-confidence source now that a real,
//! byte-exact Mesa capture exists; and every channel/bank count below needed
//! the hardware's own alignment padding (input channels round up to 16,
//! output channels to 32, weight kernel count to a multiple of 2) rather
//! than the raw logical shape -- e.g. CBUF_CON0.weight_bank is 11, not the
//! 2 this file used to compute, because `rkt_split_tasks()`'s single-task
//! branch sets it to `CBUF_BANKS - input_banks`, not the weights-size-based
//! formula that value used to come from.

use std::{fs::OpenOptions, mem, num::NonZeroUsize, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    api::{
        DRM_COMMAND_BASE, DRM_IOCTL_BASE, DRM_ROCKET_CREATE_BO, DRM_ROCKET_FINI_BO,
        DRM_ROCKET_PREP_BO, DRM_ROCKET_SUBMIT, drm_rocket_create_bo, drm_rocket_fini_bo,
        drm_rocket_job, drm_rocket_prep_bo, drm_rocket_submit, drm_rocket_task,
    },
    builders::{
        Bits, DOMAIN_CORE, DOMAIN_DPU, DOMAIN_PC, RegCmd, Register, RegisterMeta,
        cna::{
            CnaCbufCon0, CnaCbufCon1, CnaConvCon1, CnaConvCon2, CnaConvCon3, CnaCvtCon0,
            CnaCvtCon1, CnaCvtCon2, CnaCvtCon3, CnaCvtCon4, CnaCvtCon5, CnaDataSize0, CnaDataSize1,
            CnaDataSize2, CnaDataSize3, CnaDcompAddr0, CnaDcompAmount0, CnaDcompAmount1,
            CnaDcompAmount2, CnaDcompAmount3, CnaDcompAmount4, CnaDcompAmount5, CnaDcompAmount6,
            CnaDcompAmount7, CnaDcompAmount8, CnaDcompAmount9, CnaDcompAmount10, CnaDcompAmount11,
            CnaDcompAmount12, CnaDcompAmount13, CnaDcompAmount14, CnaDcompAmount15, CnaDcompCtrl,
            CnaDcompRegnum, CnaDmaCon0, CnaDmaCon1, CnaDmaCon2, CnaFcCon0, CnaFcCon1, CnaFcCon2,
            CnaFcDataSize0, CnaFcDataSize1, CnaFeatureDataAddr, CnaPadCon0, CnaPadCon1,
            CnaWeightSize0, CnaWeightSize1, CnaWeightSize2,
        },
        core::{CoreClipTruncate, CoreDataoutSize0, CoreDataoutSize1, CoreMiscCfg},
        dpu::{
            DpuBnAluCfg, DpuBnCfg, DpuBnMulCfg, DpuBnReluxCmpValue, DpuBsAluCfg, DpuBsCfg,
            DpuBsMulCfg, DpuBsOwCfg, DpuBsOwOp, DpuBsReluxCmpValue, DpuDataCubeChannel,
            DpuDataCubeHeight, DpuDataCubeNotchAddr, DpuDataCubeWidth, DpuDataFormat,
            DpuDstBaseAddr, DpuDstSurfStride, DpuEwCfg, DpuEwCvtOffsetValue, DpuEwCvtScaleValue,
            DpuEwOpValue0, DpuEwOpValue1, DpuEwOpValue2, DpuEwOpValue3, DpuEwOpValue4,
            DpuEwOpValue5, DpuEwOpValue6, DpuEwOpValue7, DpuEwReluxCmpValue, DpuFeatureModeCfg,
            DpuLutAccessCfg, DpuLutAccessData, DpuLutCfg, DpuLutInfo, DpuLutLeEnd,
            DpuLutLeSlopeScale, DpuLutLeSlopeShift, DpuLutLeStart, DpuLutLoEnd, DpuLutLoSlopeScale,
            DpuLutLoSlopeShift, DpuLutLoStart, DpuOffsetPend, DpuOutCvtOffset, DpuOutCvtScale,
            DpuOutCvtShift, DpuSPointer, DpuSurfaceAdd, DpuWdmaSize0, DpuWdmaSize1,
        },
        dpu_rdma::{
            DpuRdmaBnBaseAddr, DpuRdmaBrdmaCfg, DpuRdmaBsBaseAddr, DpuRdmaDataCubeChannel,
            DpuRdmaDataCubeHeight, DpuRdmaDataCubeWidth, DpuRdmaErdmaCfg, DpuRdmaEwBaseAddr,
            DpuRdmaEwSurfNotch, DpuRdmaEwSurfStride, DpuRdmaFeatureModeCfg, DpuRdmaNrdmaCfg,
            DpuRdmaPadCfg, DpuRdmaSPointer, DpuRdmaSrcBaseAddr, DpuRdmaSrcDmaCfg, DpuRdmaSurfNotch,
            DpuRdmaWeight,
        },
    },
    debug::dump_cmds,
    registers::{
        PC_OPERATION_ENABLE_OP_EN, PC_OPERATION_ENABLE_RESERVED_0, REG_PC_OPERATION_ENABLE,
        REG_PC_REGISTER_AMOUNTS,
    },
};
use nix::{
    ioctl_readwrite, ioctl_write_ptr,
    sys::mman::{MapFlags, ProtFlags, mmap},
};

// 1 input channel, 1 output channel, 1x1 kernel, 4x4 spatial, stride 1, no
// padding, no dilation -- the smallest real convolution (not just an
// elementwise op) this hardware can do. These are the *logical* operation
// dimensions Mesa calls `operation->*` -- the hardware-visible register
// values are mostly the alignment-padded `task->*` versions computed below,
// not these raw numbers directly.
const IN_WIDTH: u32 = 4;
const IN_HEIGHT: u32 = 4;
const IN_CHANNELS: u32 = 1;
const OUT_CHANNELS: u32 = 1;
const OUT_WIDTH: u32 = IN_WIDTH;
const OUT_HEIGHT: u32 = IN_HEIGHT;
const ZERO_POINT: u32 = 0; // raw non-quantized test data: input/output/weights all use 0

// Helper for registers we write as a bare zero -- still routes through the
// real register type's RegisterMeta (correct domain+offset), just with no
// fields set, matching how Mesa's EMIT(REG_..., 0) calls behave.
fn zero<R: RegisterMeta>() -> RegCmd {
    Register::<R>::new().build()
}

fn main() {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/accel/accel0")
        .expect("Failed to open device");
    let fd = file.as_raw_fd();

    println!("--- NPU Diagnostic: Minimal 1x1 Conv (full Mesa regcmd port) ---");

    let tensor_size = 4096; // 4KB aligned

    unsafe {
        let buf_a = Buffer::new(fd, tensor_size, &file);
        ptr::write_bytes(buf_a.host_ptr, 10, tensor_size);

        let buf_w = Buffer::new(fd, tensor_size, &file);
        ptr::write_bytes(buf_w.host_ptr, 2, tensor_size);

        let buf_c = Buffer::new(fd, tensor_size, &file);
        ptr::write_bytes(buf_c.host_ptr, 0, tensor_size);

        // Biases buffer -- Mesa's fill_first_regcmd() unconditionally wires
        // DPU_RDMA_RDMA_BS_BASE_ADDR at operation->biases's physical address
        // and always runs DPU's BS (bias-subtract) ALU (see DPU_BS_CFG
        // below); there's no "no bias" bypass path. Zero-filled: a zero
        // bias is a numeric no-op for this diagnostic, but the buffer
        // itself and its DMA wiring have to be real.
        let buf_bias = Buffer::new(fd, tensor_size, &file);
        ptr::write_bytes(buf_bias.host_ptr, 0, tensor_size);

        println!(
            "Buffers: A@0x{:x}, W@0x{:x}, Bias@0x{:x}, C@0x{:x}",
            buf_a.dma_address, buf_w.dma_address, buf_bias.dma_address, buf_c.dma_address
        );

        // ==================================================================
        // Task-level derived values -- ported from rkt_task.c's
        // calc_entries_per_slice()/calc_input_banks()/fill_task() and
        // rkt_split_tasks()'s "full weights, full input" single-task
        // branch, specialized to this file's fixed IN_*/OUT_* shape. This
        // op is far too small to ever need the general multi-task
        // splitting logic, but the *formulas* that branch uses for buffer
        // geometry and CBUF bank allocation still apply and matter.
        // ==================================================================

        // rkt_ml.h geometry constants.
        const CBUF_ENTRY_SIZE: u32 = 128; // CBUF_BANK_SIZE(32768) / CBUF_ENTRIES_PER_BANK(256)
        const CBUF_ENTRIES_PER_BANK: u32 = 256;
        const CBUF_BANKS: u32 = 12;
        const FEATURE_ATOMIC_SIZE: u32 = 16;
        const ATOMIC_K_SIZE: u32 = 16;

        // calc_entries_per_slice() (bpe=1, int8).
        let atomics_per_entry = CBUF_ENTRY_SIZE / FEATURE_ATOMIC_SIZE;
        let total_c_atomics = IN_CHANNELS.div_ceil(FEATURE_ATOMIC_SIZE);
        let last_c_atomics = total_c_atomics % atomics_per_entry;
        let int_c_entries = (total_c_atomics / atomics_per_entry) * IN_WIDTH;
        let frac_c_entries = if last_c_atomics == 3 {
            IN_WIDTH
        } else {
            (last_c_atomics * IN_WIDTH).div_ceil(atomics_per_entry)
        };
        let entries_per_slice = int_c_entries + frac_c_entries;

        // calc_input_banks().
        let input_banks = (entries_per_slice * IN_HEIGHT).div_ceil(CBUF_ENTRIES_PER_BANK);

        // rkt_split_tasks(): the single-task branch sets weight_banks to
        // CBUF_BANKS - input_banks directly -- NOT calc_weights_banks()'s
        // own result (that function's output only decides the
        // reuse/no-reuse branch elsewhere, never reaches CBUF_CON0 here).
        let weight_banks = CBUF_BANKS - input_banks;

        // fill_task(): channel/kernel counts round up to the hardware's
        // atomic granularity, independent of the op's logical shape.
        let task_input_channels = IN_CHANNELS
            .max(FEATURE_ATOMIC_SIZE)
            .next_multiple_of(FEATURE_ATOMIC_SIZE);
        let task_output_channels = OUT_CHANNELS.max(32).next_multiple_of(32); // not depthwise: no extra doubling
        let weights_kernels = OUT_CHANNELS.next_multiple_of(2); // not depthwise: align(output_channels, 2)

        let line_stride = |w: u32| w * ATOMIC_K_SIZE; // calc_line_stride()
        // input_channels_real(1)==1 and not(output_channels_real>1 or
        // addition input) -> fill_task()'s else branch:
        let input_line_stride = line_stride(IN_WIDTH) / 4;
        let input_surface_stride = input_line_stride * (IN_HEIGHT / 4 - 1);
        let output_surface_stride = (line_stride(OUT_WIDTH) * OUT_HEIGHT) / FEATURE_ATOMIC_SIZE;
        let surfaces_per_row = OUT_WIDTH * OUT_HEIGHT * 2; // not depthwise: no further *2

        // DPU_OUT_CVT_SCALE/SHIFT: rkt_regcmd.c's non-add_tensor branch.
        // Our buffers aren't real quantized tensors (input/weights/output
        // scale are all 1.0, i.e. no rescaling) but the register values
        // still go through the same float-bits fixed-point conversion Mesa
        // uses -- f32::to_bits() is a bit-exact match for the C `fui()`
        // reinterpret-cast this formula is built on.
        let conv_scale: f32 = (1.0_f32 * 1.0_f32) / 1.0_f32; // input_scale * weights_scale / output_scale
        let scale_bits = conv_scale.to_bits();
        let out_cvt_shift = 127 + 31 - 32 - (scale_bits >> 23) + 16; // truncate_bits=0: no decrement
        let mut out_cvt_scale = ((scale_bits >> 9) & 0x7fff) + 1;
        if out_cvt_scale < (1 << 14) {
            out_cvt_scale |= 1 << 14;
        }
        let out_cvt_offset = ZERO_POINT.wrapping_sub(0x80); // output_zero_point - 0x80

        let mut cmds: Vec<RegCmd> = Vec::new();

        // ==================================================================
        // CNA: feature/weight DMA staging + convolution shape. Order and
        // field values below follow mesa-rocket-userspace/rkt_regcmd.c's
        // fill_first_regcmd() line for line.
        // ==================================================================

        let cbuf_con0 = Register::<CnaCbufCon0>::new()
            .weight_bank(Bits::new(weight_banks))
            .data_bank(Bits::new(input_banks))
            .build();
        // Written here (before anything else) AND again after WEIGHT_SIZE2
        // below -- Mesa's fill_first_regcmd() genuinely emits it twice,
        // confirmed both in source and in the real regcmd capture.
        cmds.push(
            Register::<CnaCbufCon0>::new()
                .weight_bank(Bits::new(weight_banks))
                .data_bank(Bits::new(input_banks))
                .build(),
        );

        cmds.push(zero::<CnaDcompRegnum>());
        cmds.push(zero::<CnaDcompCtrl>());

        // CONV_CON1: Mesa's con1 starts at 0 and only ORs in
        // NONALIGN_DMA/GROUP_LINE_OFF/ARGB_IN when input_channels_real==1
        // (our case) and CONV_MODE(3) when depthwise (not our case) --
        // conv_mode/in_precision/proc_precision/deconv are NOT set by this
        // function at all for this register. Previous revisions of this
        // file set those extra fields too, sourced from cross-checking a
        // *vendor*-toolchain-compiled regcmd (conv.rknn) rather than Mesa's
        // own code -- a different compiler entirely, now superseded by a
        // real byte-exact Mesa capture as the higher-confidence source.
        let conv_con1 = Register::<CnaConvCon1>::new()
            .nonalign_dma(Bits::new(1))
            .group_line_off(Bits::new(1))
            .argb_in(Bits::new(8))
            .build();
        cmds.push(
            Register::<CnaConvCon1>::new()
                .nonalign_dma(Bits::new(1))
                .group_line_off(Bits::new(1))
                .argb_in(Bits::new(8))
                .build(),
        );

        // DPU_S_POINTER / DPU_RDMA_RDMA_S_POINTER ("ping-pong pointer")
        // immediately after the first CONV_CON1 write, then CONV_CON1 is
        // written a second time -- confirmed both by the real conv.rknn
        // decode and by Mesa's source, in this exact position.
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
        cmds.push(conv_con1);

        // CONV_CON2: only FEATURE_GRAINS is set by Mesa here (its own
        // comment: "Magic: Seems to pass the most tests") -- no
        // cmd_fifo_srst field, which a previous revision of this file
        // invented.
        cmds.push(
            Register::<CnaConvCon2>::new()
                .feature_grains(Bits::new(50 + 1 + 1)) // 50 + stride_y(1) + 1
                .build(),
        );

        // CONV_CON3: stride only -- Mesa doesn't set atrous dilation
        // fields here either (they default to 0, which is what our no-
        // dilation case needs anyway, so no functional difference either
        // way -- omitted here to match Mesa's actual EMIT call exactly).
        cmds.push(
            Register::<CnaConvCon3>::new()
                .conv_x_stride(Bits::new(1))
                .conv_y_stride(Bits::new(1))
                .build(),
        );

        cmds.push(
            Register::<CnaDataSize0>::new()
                .datain_width(Bits::new(IN_WIDTH))
                .datain_height(Bits::new(IN_HEIGHT))
                .build(),
        );
        // datain_channel_real is N-1 (task->input_channels_real - 1);
        // datain_channel is the atomic-aligned count (task->input_channels,
        // 16 here) -- NOT the raw logical IN_CHANNELS(1) this file used to
        // write to both subfields.
        cmds.push(
            Register::<CnaDataSize1>::new()
                .datain_channel_real(Bits::new(IN_CHANNELS - 1))
                .datain_channel(Bits::new(task_input_channels))
                .build(),
        );
        cmds.push(
            Register::<CnaDataSize2>::new()
                .dataout_width(Bits::new(OUT_WIDTH))
                .build(),
        );
        cmds.push(
            Register::<CnaDataSize3>::new()
                .dataout_atomics(Bits::new(OUT_WIDTH * OUT_HEIGHT))
                .build(),
        );

        // WEIGHT_SIZE0/1 use the atomic-aligned input channel count and
        // the aligned weights_kernels(2), not the raw logical values this
        // file used before (weights_width*weights_height*1*1=1 byte) --
        // real hardware always reads weight data laid out against the
        // aligned channel/kernel geometry.
        cmds.push(
            Register::<CnaWeightSize0>::new()
                .weight_bytes(Bits::new(1 * 1 * task_input_channels * weights_kernels))
                .build(),
        );
        cmds.push(
            Register::<CnaWeightSize1>::new()
                .weight_bytes_per_kernel(Bits::new(1 * 1 * task_input_channels))
                .build(),
        );
        cmds.push(
            Register::<CnaWeightSize2>::new()
                .weight_width(Bits::new(1))
                .weight_height(Bits::new(1))
                .weight_kernels(Bits::new(weights_kernels))
                .build(),
        );

        cmds.push(cbuf_con0);

        cmds.push(
            Register::<CnaCbufCon1>::new()
                .data_entries(Bits::new(IN_WIDTH * IN_HEIGHT)) // input_channels_real==1 branch
                .build(),
        );

        // CVT_CON0-5: input_channels_real==1, no addition_input branch.
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
                .feature_base_addr(Bits::new(buf_a.dma_address))
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
                .dma_width(Bits::new(IN_WIDTH)) // operation->input_width
                .dma_height(Bits::new(IN_HEIGHT)) // task->input_height
                .build(),
        );
        cmds.push(
            Register::<CnaFcDataSize1>::new()
                .dma_channel(Bits::new(task_input_channels))
                .build(),
        );

        // DCOMP_CTRL/REGNUM written again (Mesa emits both twice -- once
        // in the preamble above, once here). Plain 0, no wt_dec_bypass
        // field -- a previous revision of this file set that field
        // (reasoning: buf_w holds raw uncompressed bytes), but Mesa's real
        // code never touches it for this path at all.
        cmds.push(zero::<CnaDcompCtrl>());
        cmds.push(zero::<CnaDcompRegnum>());
        cmds.push(
            Register::<CnaDcompAddr0>::new()
                .decompress_addr0(Bits::new(buf_w.dma_address))
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
            Register::<CnaCvtCon5>::new()
                .per_channel_cvt_en(Bits::new(65535)) // input_channels_real==1 branch
                .build(),
        );

        // PAD_CON1: weights_width(1) < 3, not addition, not depthwise ->
        // input_zero_point - 0x80.
        cmds.push(
            Register::<CnaPadCon1>::new()
                .pad_value(Bits::new(out_cvt_offset)) // same formula/value as out_cvt_offset (both zero points are 0)
                .build(),
        );

        // ==================================================================
        // CORE: MAC array / accumulation
        // ==================================================================

        // QD_EN(1) unconditionally -- this file previously never set it
        // (and instead set an invented `proc_precision` field Mesa never
        // touches on CORE_MISC_CFG at all).
        cmds.push(Register::<CoreMiscCfg>::new().qd_en(Bits::new(1)).build());
        // DATAOUT_HEIGHT/WIDTH are N-1 (task->output_height/width - 1) --
        // this file previously wrote the raw counts here, inconsistent
        // with its own sibling DPU_DATA_CUBE_WIDTH/HEIGHT below (which
        // already had the -1 right).
        cmds.push(
            Register::<CoreDataoutSize0>::new()
                .dataout_width(Bits::new(OUT_WIDTH - 1))
                .dataout_height(Bits::new(OUT_HEIGHT - 1))
                .build(),
        );
        // dataout_channel uses the atomic-aligned output channel count
        // (task->output_channels - 1 = 31), not the raw OUT_CHANNELS-1(0).
        cmds.push(
            Register::<CoreDataoutSize1>::new()
                .dataout_channel(Bits::new(task_output_channels - 1))
                .build(),
        );
        cmds.push(
            Register::<CoreClipTruncate>::new()
                .clip_truncate(Bits::new(0)) // operation->truncate_bits = 0
                .build(),
        );
        // Undocumented/unnamed CORE register at 0x3030 -- TRM-mandated
        // reserved-bit write right after CLIP_TRUNCATE, no REG_CORE_*
        // constant covers it. See mesa-rocket-userspace/rkt_regcmd.c.
        cmds.push(RegCmd::new(DOMAIN_CORE, 0x3030, 0));

        // ==================================================================
        // DPU: output requantization + writeback
        // ==================================================================

        cmds.push(
            Register::<DpuFeatureModeCfg>::new()
                .burst_len(Bits::new(15))
                .output_mode(Bits::new(2))
                .build(),
        );
        // Plain 0 -- Mesa never sets precision fields on DPU_DATA_FORMAT
        // for this path (a previous revision invented in/out/proc_precision
        // here, sourced from the vendor-toolchain cross-check mentioned in
        // the top-of-file doc comment).
        cmds.push(zero::<DpuDataFormat>());
        cmds.push(zero::<DpuOffsetPend>());
        cmds.push(
            Register::<DpuDstBaseAddr>::new()
                .dst_base_addr(Bits::new(buf_c.dma_address))
                .build(),
        );
        cmds.push(
            Register::<DpuDstSurfStride>::new()
                .dst_surf_stride(Bits::new(output_surface_stride))
                .build(),
        );
        cmds.push(
            Register::<DpuDataCubeWidth>::new()
                .width(Bits::new(OUT_WIDTH - 1))
                .build(),
        );
        cmds.push(
            Register::<DpuDataCubeHeight>::new()
                .height(Bits::new(OUT_HEIGHT - 1))
                .build(),
        );
        cmds.push(zero::<DpuDataCubeNotchAddr>());
        cmds.push(
            Register::<DpuDataCubeChannel>::new()
                .orig_channel(Bits::new(OUT_CHANNELS - 1)) // output_channels_real - 1
                .channel(Bits::new(task_output_channels - 1)) // aligned output_channels - 1
                .build(),
        );

        // BS (bias-subtract): Mesa never bypasses this block outright --
        // it always runs the ALU (ALGO(2)|SRC(1)), with RELU/MUL bypassed.
        // A previous revision of this file set only bs_bypass(1), which
        // isn't what Mesa's real value is at all -- and never allocated
        // the biases buffer this implies DPU_RDMA needs to feed it (see
        // buf_bias above / DPU_RDMA_RDMA_BS_BASE_ADDR below).
        cmds.push(
            Register::<DpuBsCfg>::new()
                .bs_alu_algo(Bits::new(2))
                .bs_alu_src(Bits::new(1))
                .bs_relu_bypass(Bits::new(1))
                .bs_mul_bypass(Bits::new(1))
                .build(),
        );
        cmds.push(zero::<DpuBsAluCfg>());
        cmds.push(zero::<DpuBsMulCfg>());
        cmds.push(zero::<DpuBsReluxCmpValue>());
        // Not depthwise: SIZE_E_0/1/2 = 1 each (depthwise branch uses 3).
        cmds.push(
            Register::<DpuBsOwCfg>::new()
                .size_e_0(Bits::new(1))
                .size_e_1(Bits::new(1))
                .size_e_2(Bits::new(1))
                .build(),
        );
        cmds.push(
            Register::<DpuBsOwOp>::new()
                .ow_op(Bits::new(0x80 - ZERO_POINT)) // 0x80 - weights_zero_point
                .build(),
        );
        cmds.push(
            Register::<DpuWdmaSize0>::new()
                .channel_wdma(Bits::new(task_output_channels - 1))
                .build(),
        );
        cmds.push(
            Register::<DpuWdmaSize1>::new()
                .height_wdma(Bits::new(OUT_HEIGHT - 1))
                .width_wdma(Bits::new(OUT_WIDTH - 1))
                .build(),
        );

        // BN (batchnorm): genuinely fully bypassed by Mesa, unlike BS above
        // -- but all four bypass bits are set (RELU/MUL/ALU/BN), not just
        // bn_bypass(1) alone the way this file had it before.
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

        // EW (elementwise): add_tensor == -1 branch -- fully bypassed, all
        // five bypass bits set (this file previously set only ew_bypass(1)).
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

        // Output requantization -- real formula (float-bits based fixed-
        // point conversion), not the placeholder scale=1/shift=0/offset=0
        // this file used before. See the derivation above.
        cmds.push(
            Register::<DpuOutCvtOffset>::new()
                .out_cvt_offset(Bits::new(out_cvt_offset))
                .build(),
        );
        cmds.push(
            Register::<DpuOutCvtScale>::new()
                .out_cvt_scale(Bits::new(out_cvt_scale))
                .build(),
        );
        cmds.push(
            Register::<DpuOutCvtShift>::new()
                .out_cvt_shift(Bits::new(out_cvt_shift - 1))
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

        // Undocumented/unnamed DPU register at 0x40c4 -- TRM-mandated
        // reserved-bit write right after SURFACE_ADD.
        cmds.push(RegCmd::new(DOMAIN_DPU, 0x40c4, 0));

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

        // ==================================================================
        // DPU_RDMA -- architecturally chained to DPU; PC's completion
        // aggregation needs it configured and participating even though
        // this op doesn't need it to read anything itself (MRDMA_DISABLE=1
        // below). Also now feeds BS's biases buffer (see DpuBsCfg above).
        // ==================================================================

        cmds.push(
            Register::<DpuRdmaDataCubeWidth>::new()
                .width(Bits::new(OUT_WIDTH - 1))
                .build(),
        );
        cmds.push(
            Register::<DpuRdmaDataCubeHeight>::new()
                .height(Bits::new(OUT_HEIGHT - 1))
                .build(),
        );
        // Aligned output channel count, matching CORE_DATAOUT_SIZE_1/
        // DPU_DATA_CUBE_CHANNEL's `channel` subfield above -- not the raw
        // OUT_CHANNELS-1(0) this file used before.
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
                .bs_base_addr(Bits::new(buf_bias.dma_address))
                .build(),
        );
        cmds.push(zero::<DpuRdmaNrdmaCfg>());
        cmds.push(zero::<DpuRdmaBnBaseAddr>());

        // add_tensor == -1 branch throughout below.
        cmds.push(
            Register::<DpuRdmaErdmaCfg>::new()
                .erdma_disable(Bits::new(1))
                .build(),
        );
        cmds.push(zero::<DpuRdmaEwBaseAddr>());
        cmds.push(zero::<DpuRdmaEwSurfStride>());

        cmds.push(
            Register::<DpuRdmaFeatureModeCfg>::new()
                .burst_len(Bits::new(15))
                .mrdma_disable(Bits::new(1))
                .build(),
        );
        cmds.push(zero::<DpuRdmaSrcDmaCfg>());
        cmds.push(zero::<DpuRdmaSurfNotch>()); // add_tensor == -1 branch
        cmds.push(zero::<DpuRdmaPadCfg>());
        cmds.push(
            Register::<DpuRdmaWeight>::new()
                .e_weight(Bits::new(1))
                .n_weight(Bits::new(1))
                .b_weight(Bits::new(1))
                .m_weight(Bits::new(1))
                .build(),
        );
        cmds.push(zero::<DpuRdmaEwSurfNotch>()); // add_tensor == -1 branch

        // ==================================================================
        // Kick sequence -- ported verbatim from Mesa's reference gallium
        // driver (rkt_regcmd.c, fill_first_regcmd's tail, num_tasks == 1
        // branch). Four entries, in this exact order:
        //   1. PC_BASE_ADDRESS placeholder -- bare untagged 0 (single task,
        //      only patched for multi-task chaining, which we don't do).
        //   2. PC_REGISTER_AMOUNTS placeholder -- also 0 here.
        //   3. A raw, untagged magic word ("TRM: before op_en,
        //      64'h0041_xxxx_xxxx_xxxx must be set").
        //   4. The actual kick: domain 0x81 (broadcast override, not a
        //      real addressable block) at REG_PC_OPERATION_ENABLE.
        // ==================================================================

        cmds.push(RegCmd::new_raw(0x0)); // PC_BASE_ADDRESS placeholder (single task)
        cmds.push(RegCmd::new(DOMAIN_PC, REG_PC_REGISTER_AMOUNTS, 0)); // PC_REGISTER_AMOUNTS placeholder
        cmds.push(RegCmd::new_raw(0x0041000000000000)); // TRM: required immediately before op_en
        cmds.push(RegCmd::new(
            0x81,
            REG_PC_OPERATION_ENABLE,
            PC_OPERATION_ENABLE_RESERVED_0(14) | PC_OPERATION_ENABLE_OP_EN(1),
        ));

        // Pad to an even entry count -- PC reads regcmd entries in pairs
        // ((regcmd_count + 1) / 2 - 1 in the kernel), so an odd count
        // would make it read one word past our populated data. Still
        // within the same zeroed, page-granular GEM allocation either way.
        if cmds.len() % 2 != 0 {
            cmds.push(RegCmd::new_raw(0x0));
        }

        // Printed to stderr before touching hardware, so it's visible even
        // if SUBMIT/PREP_BO hangs afterward.
        dump_cmds("rkt-basic", &cmds);

        let cmd_bytes = cmds.len() * mem::size_of::<u64>();
        let cmd_len = cmd_bytes.next_multiple_of(4096);
        let buf_cmd = Buffer::new(fd, cmd_len, &file);
        let cmd_slice = std::slice::from_raw_parts_mut(buf_cmd.host_ptr as *mut u64, cmds.len());
        for (i, c) in cmds.iter().enumerate() {
            cmd_slice[i] = c.0;
        }

        rocket_fini_bo(
            fd,
            &drm_rocket_fini_bo {
                handle: buf_a.handle,
                reserved: 0,
            },
        )
        .ok();
        rocket_fini_bo(
            fd,
            &drm_rocket_fini_bo {
                handle: buf_w.handle,
                reserved: 0,
            },
        )
        .ok();
        rocket_fini_bo(
            fd,
            &drm_rocket_fini_bo {
                handle: buf_bias.handle,
                reserved: 0,
            },
        )
        .ok();
        rocket_fini_bo(
            fd,
            &drm_rocket_fini_bo {
                handle: buf_c.handle,
                reserved: 0,
            },
        )
        .ok();
        rocket_fini_bo(
            fd,
            &drm_rocket_fini_bo {
                handle: buf_cmd.handle,
                reserved: 0,
            },
        )
        .ok();

        // Not doubled -- Mesa's rkt_ml_subgraph_invoke() sets
        // ktask->regcmd_count = task->regcfg_amount, the raw entry count
        // (cmds.len()), no `* 2`. The kernel's own
        // `(regcmd_count + 1) / 2 - 1` halving (rocket_job_hw_submit) is
        // its internal conversion to whatever unit PC_REGISTER_AMOUNTS
        // counts in, not a pre-multiplication contract on this field.
        let task = drm_rocket_task {
            regcmd: buf_cmd.dma_address,
            regcmd_count: cmds.len() as u32,
        };

        let in_handles = vec![buf_cmd.handle, buf_a.handle, buf_w.handle, buf_bias.handle];
        let out_handles = vec![buf_c.handle];

        let job = drm_rocket_job {
            tasks: &task as *const _ as u64,
            in_bo_handles: in_handles.as_ptr() as u64,
            out_bo_handles: out_handles.as_ptr() as u64,
            task_count: 1,
            task_struct_size: mem::size_of::<drm_rocket_task>() as u32,
            in_bo_handle_count: in_handles.len() as u32,
            out_bo_handle_count: out_handles.len() as u32,
        };

        let mut submit = drm_rocket_submit {
            jobs: &job as *const _ as u64,
            job_count: 1,
            job_struct_size: mem::size_of::<drm_rocket_job>() as u32,
            reserved: 0,
        };

        println!("Submitting...");
        rocket_submit(fd, &mut submit).expect("IOCTL Failed");

        let prep = drm_rocket_prep_bo {
            handle: buf_c.handle,
            reserved: 0,
            timeout_ns: 2_000_000_000,
        };

        match rocket_prep_bo(fd, &prep) {
            Ok(_) => {
                let res = *buf_c.host_ptr;
                println!("Done! Result[0]: {}", res);
            }
            Err(e) => {
                println!("TIMEOUT/error: {}", e);
            }
        }
    }
}

ioctl_readwrite!(
    rocket_create_bo,
    DRM_IOCTL_BASE,
    DRM_COMMAND_BASE + DRM_ROCKET_CREATE_BO,
    drm_rocket_create_bo
);
ioctl_readwrite!(
    rocket_submit,
    DRM_IOCTL_BASE,
    DRM_COMMAND_BASE + DRM_ROCKET_SUBMIT,
    drm_rocket_submit
);
ioctl_write_ptr!(
    rocket_prep_bo,
    DRM_IOCTL_BASE,
    DRM_COMMAND_BASE + DRM_ROCKET_PREP_BO,
    drm_rocket_prep_bo
);
ioctl_write_ptr!(
    rocket_fini_bo,
    DRM_IOCTL_BASE,
    DRM_COMMAND_BASE + DRM_ROCKET_FINI_BO,
    drm_rocket_fini_bo
);

struct Buffer {
    handle: u32,
    dma_address: u32,
    size: usize,
    host_ptr: *mut u8,
}

impl Buffer {
    unsafe fn new(fd: i32, size: usize, file: &std::fs::File) -> Self {
        let mut create_params = drm_rocket_create_bo {
            size: size as u32,
            handle: 0,
            dma_address: 0,
            offset: 0,
        };
        rocket_create_bo(fd, &mut create_params).expect("Failed to create BO");
        let map_len = NonZeroUsize::new(size).unwrap();
        let map_addr = mmap(
            None,
            map_len,
            ProtFlags::PROT_READ | ProtFlags::PROT_WRITE,
            MapFlags::MAP_SHARED,
            file,
            create_params.offset as i64,
        )
        .expect("mmap failed");
        Buffer {
            handle: create_params.handle,
            dma_address: create_params.dma_address as u32,
            size,
            host_ptr: map_addr.as_ptr() as *mut u8,
        }
    }
}
