//! NPU diagnostic: minimal 1x1 conv, built from the typed register builders
//! instead of hand-rolled offsets.
//!
//! The previous version of this file defined its own `REG_CNA_CONV_CON0 =
//! 0x1010` and wrote 0 to it. That register doesn't exist -- per
//! `rkt_registers.h`, CNA register numbering starts at CONV_CON1 (0x100c),
//! and 0x1010 is actually CONV_CON2. So the old code silently wrote to the
//! wrong register (zeroing feature_grains/kernel_group) and never touched
//! CONV_CON1 at all, meaning conv_mode/in_precision/proc_precision were
//! left unset. That's the leading suspect for the EBUSY timeout this file
//! already had error handling for.
//!
//! This version routes every register write through `crate::rocket::builders`
//! (`Register<T>` + `RegisterMeta`), which covers every CNA/CORE/DPU/PC/
//! Global field by name and mask, generated from `rkt_registers.h`. Fields
//! left at 0 (ping-pong pointers, reserved bits, DMA burst lengths) are
//! left unset deliberately, matching hardware POR-style defaults.
//!
//! Also worth knowing: `builders.rs`'s `DOMAIN_CNA`/`DOMAIN_PC`/
//! `DOMAIN_CORE`/`DOMAIN_DPU` (which every register in this file's
//! `RegCmd`s is tagged with, via each type's `RegisterMeta::DOMAIN`) were
//! wrong until just before this revision -- they didn't match Mesa's own
//! registers.xml `target` enum, even though other builder files in this
//! same crate (dpu_rdma.rs, ppu.rs, ...) already used the correct,
//! bindgen-generated equivalents. See `builders.rs` and NOTES.md for the
//! full story. This was arguably a bigger deal than any single field
//! value guessed at below -- a wrong domain tag can misdirect (or fail
//! to reach) the intended hardware engine regardless of how correct the
//! register's own value is.
//!
//! Confidence levels. Most of the enum/mode fields below were validated by
//! decoding a real regcmd program out of a compiled conv.rknn (see
//! NOTES.md in rknpu-spelunking for the byte-offset scan + field decode) --
//! the container format is undocumented but the wire encoding of each
//! individual (domain, offset, value) entry matches `rkt_registers.h`
//! exactly, so real per-field values could be read straight out of it and
//! cross-checked against every field we'd guessed at:
//!   - CONFIRMED against the real decode: conv_mode=0, deconv=0,
//!     in/out/proc_precision=2 (not 0 -- corrected), kernel_group=0 (not 1
//!     -- corrected), atrous dilation=0 for "none" (not 1 -- corrected;
//!     dilation and stride turned out to use different encoding
//!     conventions in the same register), DMA/DPU burst lengths=15/max
//!     (not 0 -- corrected, and CNA_DMA_CON0 was missing entirely before),
//!     DPU output_mode=2 (not 0 -- corrected), cvt_bypass=1, bn_bypass=1,
//!     ew_bypass=1, out_cvt_scale=1 (all as originally guessed).
//!   - HIGH but not from the decode: all addresses and dimensions (shape-
//!     dependent, so not directly comparable to the 32x32x8-ish real
//!     example). PC's own registers (BASE_ADDRESS/REGISTER_AMOUNTS/
//!     TASK_CON) are left to the kernel driver via
//!     `drm_rocket_task.regcmd`/`regcmd_count` rather than pushed into the
//!     stream -- confirmed directly from the mainline kernel driver's own
//!     source now (`drivers/accel/rocket/rocket_job.c`,
//!     `rocket_job_hw_submit`), not just inferred from the vendor driver's
//!     `rknpu_job_subcore_commit` disassembly the way it originally was.
//!     That source is also where the `regcmd_count * 2` below comes from
//!     -- see the comment at the `drm_rocket_task` construction.
//!   - STILL UNCONFIRMED, flagged inline: feature_grains and
//!     CBUF_CON1.data_entries (both shape-dependent; the real conv.rknn
//!     example's values don't cleanly scale down to our much smaller test
//!     tensor, so no formula yet). CBUF_CON0's weight_bank/data_bank *are*
//!     now formula-derived (see the comment at that register) rather than
//!     guessed -- weight_bank was wrong (1, should be 2) until corrected
//!     by running that formula, a plausible cause of hangs since it under-
//!     allocates CNA's weight cache. out_cvt_shift (the real example only confirms
//!     scale=1, not shift=0 -- DPU_OUT_CVT_SCALE also has an
//!     FP32TOFP16_EN bit the current builder doesn't expose at all, a gap
//!     worth fixing in builders/dpu.rs separately).

use std::{fs::OpenOptions, mem, num::NonZeroUsize, os::unix::io::AsRawFd, ptr};

use iree_rocket_hal::rocket::{
    api::{
        DRM_COMMAND_BASE, DRM_IOCTL_BASE, DRM_ROCKET_CREATE_BO, DRM_ROCKET_FINI_BO,
        DRM_ROCKET_PREP_BO, DRM_ROCKET_SUBMIT, drm_rocket_create_bo, drm_rocket_fini_bo,
        drm_rocket_job, drm_rocket_prep_bo, drm_rocket_submit, drm_rocket_task,
    },
    builders::{
        Bits, DOMAIN_CORE, DOMAIN_DPU, DOMAIN_PC, RegCmd, Register,
        cna::{
            CnaCbufCon0, CnaCbufCon1, CnaConvCon1, CnaConvCon2, CnaConvCon3, CnaCvtCon0,
            CnaDataSize0, CnaDataSize1, CnaDataSize2, CnaDataSize3, CnaDcompAddr0, CnaDcompCtrl,
            CnaDmaCon0, CnaFeatureDataAddr, CnaOperationEnable, CnaPadCon0, CnaWeightSize0,
            CnaWeightSize1, CnaWeightSize2,
        },
        core::{
            CoreClipTruncate, CoreDataoutSize0, CoreDataoutSize1, CoreMiscCfg, CoreOperationEnable,
        },
        dpu::{
            DpuBnCfg, DpuBsCfg, DpuDataCubeChannel, DpuDataCubeHeight, DpuDataCubeWidth,
            DpuDataFormat, DpuDstBaseAddr, DpuDstSurfStride, DpuEwCfg, DpuFeatureModeCfg,
            DpuOperationEnable, DpuOutCvtOffset, DpuOutCvtScale, DpuOutCvtShift,
        },
        dpu_rdma::{
            DpuRdmaDataCubeChannel, DpuRdmaDataCubeHeight, DpuRdmaDataCubeWidth,
            DpuRdmaFeatureModeCfg, DpuRdmaOperationEnable,
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
// elementwise op) this hardware can do.
const IN_WIDTH: u32 = 4;
const IN_HEIGHT: u32 = 4;
const IN_CHANNELS: u32 = 1;
const OUT_CHANNELS: u32 = 1;
// 1x1 stride-1 no-pad conv preserves spatial dims.
const OUT_WIDTH: u32 = IN_WIDTH;
const OUT_HEIGHT: u32 = IN_HEIGHT;
// int8 weights, 1x1 kernel, 1 in-channel, 1 out-channel.
const WEIGHT_BYTES: u32 = 1;

fn main() {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/accel/accel0")
        .expect("Failed to open device");
    let fd = file.as_raw_fd();

    println!("--- NPU Diagnostic: Minimal 1x1 Conv (typed register builders) ---");

    let tensor_size = 4096; // 4KB aligned

    unsafe {
        let buf_a = Buffer::new(fd, tensor_size, &file);
        ptr::write_bytes(buf_a.host_ptr, 10, tensor_size);

        let buf_w = Buffer::new(fd, tensor_size, &file);
        ptr::write_bytes(buf_w.host_ptr, 2, tensor_size);

        let buf_c = Buffer::new(fd, tensor_size, &file);
        ptr::write_bytes(buf_c.host_ptr, 0, tensor_size);

        println!(
            "Buffers: A@0x{:x}, W@0x{:x}, C@0x{:x}",
            buf_a.dma_address, buf_w.dma_address, buf_c.dma_address
        );

        let mut cmds: Vec<RegCmd> = Vec::new();

        // ==================================================================
        // CNA: feature/weight DMA staging + convolution shape
        // ==================================================================

        // CONV_CON1 (0x100c) -- the register the old code never wrote.
        // conv_mode=0 (plain direct conv) and deconv=0 (forward, not
        // transpose) confirmed against a real compiled regcmd (see
        // NOTES.md's conv.rknn decode). in_precision/proc_precision=2 is
        // also taken directly from that decode -- it's what the real
        // compiler used for this model's float16 tensors; codepoint 2
        // isn't independently confirmed to mean "float16" for OUR raw
        // test buffers, just that 2 is a real, hardware-accepted value
        // (unlike our original guess of 0).
        cmds.push(
            Register::<CnaConvCon1>::new()
                .conv_mode(Bits::new(0))
                .in_precision(Bits::new(2))
                .proc_precision(Bits::new(2))
                .deconv(Bits::new(0))
                .build(),
        );

        // CONV_CON2 (0x1010) -- what the old code mislabeled CONV_CON0 and
        // zeroed. kernel_group=0 confirmed (real decode showed 0, not the
        // "1 group = literal 1" guess this had before). feature_grains is
        // still an open guess -- the real decode used 36 for a 32x32x8
        // input, which doesn't cleanly divide down to anything obvious for
        // our 4x4x1 case, so this stays at the previous literal-reading
        // guess of 1 pending a real derivation.
        cmds.push(
            Register::<CnaConvCon2>::new()
                .cmd_fifo_srst(Bits::new(1)) // reset CNA's command fifo before this task
                .feature_grains(Bits::new(1))
                .kernel_group(Bits::new(0))
                .build(),
        );

        // CONV_CON3 (0x1014) -- stride 1, no dilation. Confirmed against
        // the real decode: stride is stored literally (1 = stride 1,
        // matching what we already had), but dilation is NOT -- the real
        // no-dilation encoding is 0, not 1. Different encoding convention
        // for two fields in the same register; the earlier guess of 1 for
        // dilation was wrong.
        cmds.push(
            Register::<CnaConvCon3>::new()
                .conv_x_stride(Bits::new(1))
                .conv_y_stride(Bits::new(1))
                .atrous_x_dilation(Bits::new(0))
                .atrous_y_dilation(Bits::new(0))
                .build(),
        );

        // DATA_SIZE0-3: input/output geometry.
        cmds.push(
            Register::<CnaDataSize0>::new()
                .datain_width(Bits::new(IN_WIDTH))
                .datain_height(Bits::new(IN_HEIGHT))
                .build(),
        );
        cmds.push(
            Register::<CnaDataSize1>::new()
                .datain_channel(Bits::new(IN_CHANNELS))
                .datain_channel_real(Bits::new(IN_CHANNELS))
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

        // WEIGHT_SIZE0-2: 1x1 kernel, 1 in-channel, OUT_CHANNELS kernels.
        cmds.push(
            Register::<CnaWeightSize0>::new()
                .weight_bytes(Bits::new(WEIGHT_BYTES))
                .build(),
        );
        cmds.push(
            Register::<CnaWeightSize1>::new()
                .weight_bytes_per_kernel(Bits::new(WEIGHT_BYTES))
                .build(),
        );
        cmds.push(
            Register::<CnaWeightSize2>::new()
                .weight_width(Bits::new(1))
                .weight_height(Bits::new(1))
                .weight_kernels(Bits::new(OUT_CHANNELS))
                .build(),
        );

        // CBUF_CON0/1: internal SRAM cache-buffer bank allocation. These
        // were both hardcoded to 1 (smallest legal allocation) as an
        // unconfirmed guess. rkt-job.rs separately ports a real formula
        // for this (ConvTaskConfig::calculate, constants named after
        // rkt_ml.h) rather than guessing -- running that formula for this
        // file's actual 4x4x1 shape gives data_bank=1 (matches) but
        // weight_bank=2 (did NOT match -- this file was under-allocating
        // CNA's weight cache by one bank). Formula, for reference:
        //   w_bytes = weights_w * weights_h * in_c * bpe (* out_c if not
        //             depthwise)
        //   w_entries = ceil(w_bytes / CBUF_ENTRY_SIZE)      # 128
        //   weight_bank = ceil(w_entries / CBUF_ENTRIES_PER_BANK) + 1  # 256
        // For our 1x1x1x1 int8 weight: w_bytes=1, w_entries=1,
        // weight_bank=ceil(1/256)+1=2.
        cmds.push(
            Register::<CnaCbufCon0>::new()
                .weight_bank(Bits::new(2))
                .data_bank(Bits::new(1))
                .build(),
        );
        cmds.push(
            Register::<CnaCbufCon1>::new()
                .data_entries(Bits::new(IN_WIDTH * IN_HEIGHT))
                .build(),
        );

        // CVT_CON0: bypass input requantization entirely -- we're feeding
        // raw values, not a quantized real model, so there's no scale/
        // offset to apply. cvt_bypass=1 confirmed correct against the real
        // decode; cvt_type=1/data_sign=1 are set there too even though
        // CVT is bypassed (so presumably don't-care when bypassed, but
        // matching real usage in case they aren't).
        cmds.push(
            Register::<CnaCvtCon0>::new()
                .cvt_bypass(Bits::new(1))
                .cvt_type(Bits::new(1))
                .data_sign(Bits::new(1))
                .build(),
        );

        // DMA_CON0: burst lengths. The real decode uses max burst (15) for
        // both feature and weight DMA rather than the 0 we'd otherwise
        // default to -- this register was missing entirely from the first
        // draft of this file.
        cmds.push(
            Register::<CnaDmaCon0>::new()
                .data_burst_len(Bits::new(15))
                .weight_burst_len(Bits::new(15))
                .build(),
        );

        // PAD_CON0: no padding.
        cmds.push(
            Register::<CnaPadCon0>::new()
                .pad_top(Bits::new(0))
                .pad_left(Bits::new(0))
                .build(),
        );

        // DCOMP_CTRL: bypass weight decompression -- buf_w holds raw
        // uncompressed weight bytes, not Rockchip's compressed weight
        // format.
        cmds.push(
            Register::<CnaDcompCtrl>::new()
                .wt_dec_bypass(Bits::new(1))
                .build(),
        );

        cmds.push(
            Register::<CnaFeatureDataAddr>::new()
                .feature_base_addr(Bits::new(buf_a.dma_address))
                .build(),
        );
        cmds.push(
            Register::<CnaDcompAddr0>::new()
                .decompress_addr0(Bits::new(buf_w.dma_address))
                .build(),
        );

        cmds.push(
            Register::<CnaOperationEnable>::new()
                .op_en(Bits::new(1))
                .build(),
        );

        // ==================================================================
        // CORE: MAC array / accumulation
        // ==================================================================

        cmds.push(
            Register::<CoreMiscCfg>::new()
                .proc_precision(Bits::new(2)) // must agree with CNA's proc_precision -- confirmed =2, not 0
                .build(),
        );
        cmds.push(
            Register::<CoreDataoutSize0>::new()
                .dataout_width(Bits::new(OUT_WIDTH))
                .dataout_height(Bits::new(OUT_HEIGHT))
                .build(),
        );
        // dataout_channel is N-1, not the direct count -- see the
        // DpuDataCubeWidth/Height/Channel comment below, same finding.
        cmds.push(
            Register::<CoreDataoutSize1>::new()
                .dataout_channel(Bits::new(OUT_CHANNELS - 1))
                .build(),
        );
        // clip_truncate=0: no extra right-shift when narrowing the MAC
        // accumulator back down. Least-confirmed CORE field -- if output is
        // saturated/garbage this is the first thing to revisit.
        cmds.push(
            Register::<CoreClipTruncate>::new()
                .clip_truncate(Bits::new(0))
                .build(),
        );

        // Undocumented/unnamed CORE register at offset 0x3030 -- not in
        // rkt_registers.h at all (no REG_CORE_* constant covers it; it's
        // the very next 4-byte slot after CLIP_TRUNCATE's 12316). Mesa's
        // rkt_regcmd.c writes it to 0 unconditionally right after
        // CLIP_TRUNCATE (`emit_raw(regs, CORE | 0x1, 0x3030, 0);`) with no
        // explanation -- presumably a TRM-mandated reserved-bit write.
        // Previously missing entirely from this file.
        cmds.push(RegCmd::new(DOMAIN_CORE, 0x3030, 0));

        cmds.push(
            Register::<CoreOperationEnable>::new()
                .op_en(Bits::new(1))
                .build(),
        );

        // ==================================================================
        // DPU: output requantization + writeback (bias/batchnorm/elementwise
        // all bypassed -- nothing but a straight conv here)
        // ==================================================================

        // conv_mode=0 confirmed. output_mode=2 and burst_len=15(max) are
        // corrections from the real decode -- both were guessed at 0.
        cmds.push(
            Register::<DpuFeatureModeCfg>::new()
                .conv_mode(Bits::new(0))
                .output_mode(Bits::new(2))
                .burst_len(Bits::new(15))
                .build(),
        );
        // Precision fields =2 across the board (CNA/CORE/DPU), matching
        // CnaConvCon1 above -- confirmed via the real decode, corrected
        // from the original guess of 0.
        cmds.push(
            Register::<DpuDataFormat>::new()
                .in_precision(Bits::new(2))
                .out_precision(Bits::new(2))
                .proc_precision(Bits::new(2))
                .build(),
        );
        // DPU_DATA_CUBE_WIDTH/HEIGHT and CHANNEL's own `channel` subfield
        // are all N-1, not the direct count -- this was wrong in the
        // previous revision (used the direct value) and is the most
        // likely reason that version still hit EBUSY/timeout even after
        // the domain-tag and precision-field fixes. Found by cross-
        // referencing rkt-job.rs/rkt-simple-job.rs (both already used
        // N-1 here) against the conv.rknn decode, which confirms it:
        // DPU_DATA_CUBE_WIDTH read back as 0x1f=31 for a real width of
        // 32. `orig_channel` is NOT N-1 though -- rkt-simple-job.rs uses
        // the direct count for it, a different encoding for two
        // subfields of the same register.
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
        cmds.push(
            Register::<DpuDataCubeChannel>::new()
                .channel(Bits::new(OUT_CHANNELS - 1))
                .orig_channel(Bits::new(OUT_CHANNELS))
                .build(),
        );

        cmds.push(Register::<DpuBsCfg>::new().bs_bypass(Bits::new(1)).build());
        cmds.push(Register::<DpuBnCfg>::new().bn_bypass(Bits::new(1)).build());
        cmds.push(Register::<DpuEwCfg>::new().ew_bypass(Bits::new(1)).build());

        // Output conversion is not bypass-able the way BS/BN/EW are, so
        // this needs an identity-ish scale/shift even with no real
        // quantization: out = (acc * scale) >> shift + offset, scale=1
        // shift=0 offset=0. scale=1 is confirmed against the real decode
        // (so it likely is a plain integer multiplier, not a fixed-point
        // Q-format one as originally worried) -- shift=0 and offset=0 are
        // still unconfirmed guesses, since the real example's shift wasn't
        // isolated from its other fields. The real decode also showed an
        // FP32TOFP16_EN bit in this same register that the builder below
        // doesn't expose at all (see the confidence-level doc comment up
        // top) -- not needed here since we're not asking for fp16 output,
        // but worth knowing it exists.
        cmds.push(
            Register::<DpuOutCvtOffset>::new()
                .out_cvt_offset(Bits::new(0))
                .build(),
        );
        cmds.push(
            Register::<DpuOutCvtScale>::new()
                .out_cvt_scale(Bits::new(1))
                .build(),
        );
        cmds.push(
            Register::<DpuOutCvtShift>::new()
                .out_cvt_shift(Bits::new(0))
                .build(),
        );

        cmds.push(
            Register::<DpuDstBaseAddr>::new()
                .dst_base_addr(Bits::new(buf_c.dma_address))
                .build(),
        );
        cmds.push(
            Register::<DpuDstSurfStride>::new()
                .dst_surf_stride(Bits::new(OUT_WIDTH * OUT_HEIGHT))
                .build(),
        );

        // Undocumented/unnamed DPU register at offset 0x40c4 -- same
        // situation as CORE's 0x3030 above: not in rkt_registers.h, sits
        // right after REG_DPU_SURFACE_ADD (16576) in the address space,
        // and Mesa's rkt_regcmd.c writes it to 0 unconditionally
        // (`emit_raw(regs, DPU | 0x1, 0x40c4, 0);`) right after
        // SURFACE_ADD (which this file doesn't otherwise set -- EW is
        // fully bypassed here so surfaces_per_row shouldn't matter, but
        // the reserved-register write below is unconditional in Mesa
        // regardless of EW state, so it's kept independent of that).
        cmds.push(RegCmd::new(DOMAIN_DPU, 0x40c4, 0));

        cmds.push(
            Register::<DpuOperationEnable>::new()
                .op_en(Bits::new(1))
                .build(),
        );

        // ==================================================================
        // DPU_RDMA -- required even though nothing about this test needs
        // it to actually read anything. Missing entirely (like it was in
        // an earlier draft of this file, and still is in rkt-simple-job.rs)
        // is the leading suspect for the EBUSY hang: the real, confirmed-
        // working conv.rknn regcmd decode (NOTES.md) enables CNA+CORE+DPU+
        // DPU_RDMA together (GLOBAL_OPERATION_ENABLE=0x1d), never just the
        // first three -- DPU_RDMA is architecturally chained to DPU, and
        // skipping it plausibly leaves PC's completion/interrupt
        // aggregation waiting on an engine that was never told to
        // participate. The specific field values are also non-obvious and
        // came from that decode, not guesswork: MRDMA_DISABLE=1 (the main
        // read-DMA path is OFF) in the real, working capture --
        // rkt-job.rs guesses 0 there (enabled), the opposite of what
        // real hardware does. Data-cube dims mirror DPU's own (the real
        // decode's DPU_RDMA dims matched DPU's dims exactly for that
        // layer).
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
        cmds.push(
            Register::<DpuRdmaDataCubeChannel>::new()
                .channel(Bits::new(OUT_CHANNELS - 1))
                .build(),
        );
        cmds.push(
            Register::<DpuRdmaFeatureModeCfg>::new()
                .mrdma_disable(Bits::new(1))
                .burst_len(Bits::new(15))
                .proc_precision(Bits::new(2))
                .in_precision(Bits::new(2))
                .build(),
        );
        cmds.push(
            Register::<DpuRdmaOperationEnable>::new()
                .op_en(Bits::new(1))
                .build(),
        );

        // ==================================================================
        // Kick sequence -- ported verbatim from Mesa's reference gallium
        // driver (rkt_regcmd.c, fill_first_regcmd, the only userspace
        // that's ever actually driven this mainline kernel module
        // successfully), not reconstructed from guesswork. This replaces
        // this file's previous two-write GlobalOperationEnable +
        // PCOperationEnable kick, which doesn't correspond to anything
        // Mesa actually sends -- there is no "GLOBAL" register block with
        // individually addressable per-engine enable bits; DOMAIN_GLOBAL
        // (0x81) is really a broadcast-style domain override used only for
        // this one final write, at PC's own OPERATION_ENABLE offset.
        //
        // Four entries, in this exact order (see mesa-rocket-userspace/
        // rkt_regcmd.c around line 436):
        //   1. PC_BASE_ADDRESS placeholder -- Mesa emits this as a bare
        //      untagged 0 for a single-task job (only patched/tagged for
        //      multi-task chaining, which we don't do).
        //   2. PC_REGISTER_AMOUNTS placeholder -- also just 0 here; only
        //      patched by Mesa's compile_operation() when chaining to a
        //      next task.
        //   3. A raw, untagged magic word ("TRM: before op_en,
        //      64'h0041_xxxx_xxxx_xxxx must be set") -- not decomposable
        //      into a domain/offset/value triple, so appended literally
        //      like Mesa does.
        //   4. The actual kick: domain 0x81 at REG_PC_OPERATION_ENABLE's
        //      offset, value = PC_OPERATION_ENABLE_RESERVED_0(14) |
        //      PC_OPERATION_ENABLE_OP_EN(1) -- computed via the same
        //      bindgen-generated bit-packing functions Mesa's C macros
        //      expand to, so this is byte-for-byte what Mesa's own build
        //      produces, not a hand-transcribed literal.
        // ==================================================================

        cmds.push(RegCmd::new_raw(0x0)); // PC_BASE_ADDRESS placeholder (single task)
        cmds.push(RegCmd::new(DOMAIN_PC, REG_PC_REGISTER_AMOUNTS, 0)); // PC_REGISTER_AMOUNTS placeholder
        cmds.push(RegCmd::new_raw(0x0041000000000000)); // TRM: required immediately before op_en
        cmds.push(RegCmd::new(
            0x81,
            REG_PC_OPERATION_ENABLE,
            PC_OPERATION_ENABLE_RESERVED_0(14) | PC_OPERATION_ENABLE_OP_EN(1),
        ));

        // Printed to stderr before touching hardware, so it's visible even
        // if SUBMIT/PREP_BO hangs afterward. Compare against the conv.rknn
        // regcmd decode (rknpu-spelunking/NOTES.md) register-by-register.
        dump_cmds("rkt-basic", &cmds);

        // Create Command Buffer -- sized from the actual cmds vector
        // (8 bytes per RegCmd) rather than a hardcoded 4096, rounded up to
        // a page since mmap'd GEM allocations are page-granular anyway.
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

        // NOT doubled -- reversed again, this time from Mesa's actual
        // submission code (rkt_ml.c, rkt_ml_subgraph_invoke):
        //
        //   ktask->regcmd_count = task->regcfg_amount;
        //
        // where regcfg_amount is set in compile_operation() to the raw
        // util_dynarray_num_elements() count of uint64_t regcmd entries --
        // i.e. exactly `cmds.len()`, no `* 2`. The kernel's own
        // `(task->regcmd_count + 1) / 2 - 1` halving (rocket_job.c,
        // rocket_job_hw_submit) is the kernel's internal conversion to
        // whatever unit PC_REGISTER_AMOUNTS is actually counted in (likely
        // 16-byte/2-entry DMA burst granules) -- it describes what the
        // kernel does WITH the count it's given, not what value userspace
        // is supposed to pre-multiply by. Mistook that distinction
        // previously and restored the `* 2` based on the kernel formula
        // alone; Mesa's own reference client is the actual ground truth
        // for what regcmd_count means as an ioctl input, and it does not
        // double it.
        let task = drm_rocket_task {
            regcmd: buf_cmd.dma_address,
            regcmd_count: cmds.len() as u32,
        };

        let in_handles = vec![buf_cmd.handle, buf_a.handle, buf_w.handle];
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
