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
//!     stream -- see the comment at the kick sequence below for why,
//!     cross-checked against the vendor kernel driver's own
//!     `rknpu_job_subcore_commit` disassembly.
//!   - STILL UNCONFIRMED, flagged inline: feature_grains and cache-buffer
//!     bank allocation (both shape-dependent; the real example's values
//!     don't cleanly scale down to our much smaller test tensor, so no
//!     formula yet), and out_cvt_shift (the real example only confirms
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
        Bits, RegCmd, Register,
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
        global::GlobalOperationEnable,
        pc::PCOperationEnable,
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

        // CBUF_CON0/1: internal SRAM cache-buffer bank allocation. 1 bank
        // each is the smallest legal allocation and this tensor is tiny;
        // exact bank-count semantics unconfirmed.
        cmds.push(
            Register::<CnaCbufCon0>::new()
                .weight_bank(Bits::new(1))
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
        cmds.push(
            Register::<CoreDataoutSize1>::new()
                .dataout_channel(Bits::new(OUT_CHANNELS))
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
        cmds.push(
            Register::<DpuDataCubeWidth>::new()
                .width(Bits::new(OUT_WIDTH))
                .build(),
        );
        cmds.push(
            Register::<DpuDataCubeHeight>::new()
                .height(Bits::new(OUT_HEIGHT))
                .build(),
        );
        cmds.push(
            Register::<DpuDataCubeChannel>::new()
                .channel(Bits::new(OUT_CHANNELS))
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

        cmds.push(
            Register::<DpuOperationEnable>::new()
                .op_en(Bits::new(1))
                .build(),
        );

        // ==================================================================
        // Kick: Global master-enable, selecting which engines participate.
        //
        // PC_BASE_ADDRESS / PC_REGISTER_AMOUNTS / PC_TASK_CON deliberately
        // are NOT pushed into this stream, unlike an earlier draft of this
        // file. PC has to already know where the regcmd buffer is and how
        // long it is *before* it can read the first entry out of it, so
        // those three can't logically live inside the stream they describe
        // -- that's exactly why DRM_ROCKET_SUBMIT takes `regcmd`/
        // `regcmd_count` as explicit ioctl fields below: the kernel driver
        // programs those PC registers itself before dispatching. This
        // matches what the vendor driver's `rknpu_job_subcore_commit` does
        // kernel-side (see rknpu-spelunking/NOTES.md) -- mainline almost
        // certainly does the analogous thing from `drm_rocket_task`.
        // PC_OPERATION_ENABLE is kept as a harmless no-op parity with the
        // previous version of this file: by the time PC reaches this last
        // entry in the stream it's already reading (driver-triggered), so
        // re-setting the same bit here shouldn't do anything -- but it
        // also doesn't need to be here, so it's the first thing to try
        // removing if something goes wrong.
        // ==================================================================

        cmds.push(
            Register::<GlobalOperationEnable>::new()
                .cna_op_en(Bits::new(1))
                .core_op_en(Bits::new(1))
                .dpu_op_en(Bits::new(1))
                .build(),
        );
        cmds.push(Register::<PCOperationEnable>::new().op_enable(true).build());

        // Create Command Buffer
        let cmd_len = 4096;
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

        let task = drm_rocket_task {
            regcmd: buf_cmd.dma_address,
            regcmd_count: cmds.len() as u32 * 2,
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
