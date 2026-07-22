use crate::rocket::{
    builders::{Bits, Register, RegisterMeta},
    registers::*,
};

// ========================================================================
// S_STATUS (0x6000)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct PpuSStatus;

impl RegisterMeta for PpuSStatus {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_PPU;
    const OFFSET: u32 = REG_PPU_S_STATUS;
}

impl Register<PpuSStatus> {
    /// Description: Reports executer 0's current ping-pong operating state.
    ///
    /// Bit width: 2
    /// Range of values: 0 = idle; 1 = operating; 2 = operating, executer 1 waiting to operate; 3 = reserved.
    /// Known limitations: TRM marks this field RO (hardware-reported status) — the setter exists for register-model completeness but writes have no effect on real silicon.
    /// Related registers: `PpuSStatus::status_1`; `PpuSPointer::executer`/`executer_pp_en`.
    pub fn status_0(&mut self, status_0: Bits<2>) -> &mut Self {
        self.set_field(PPU_S_STATUS_STATUS_0__MASK, unsafe {
            PPU_S_STATUS_STATUS_0(status_0.val())
        })
    }

    /// Description: Reports executer 1's current ping-pong operating state.
    ///
    /// Bit width: 2
    /// Range of values: 0 = idle; 1 = operating; 2 = operating, executer 1 waiting to operate; 3 = reserved.
    /// Known limitations: TRM marks this field RO (hardware-reported status) — the setter exists for register-model completeness but writes have no effect on real silicon.
    /// Related registers: `PpuSStatus::status_0`; `PpuSPointer::executer`/`executer_pp_en`.
    pub fn status_1(&mut self, status_1: Bits<2>) -> &mut Self {
        self.set_field(PPU_S_STATUS_STATUS_1__MASK, unsafe {
            PPU_S_STATUS_STATUS_1(status_1.val())
        })
    }
}

// ========================================================================
// S_POINTER (0x6004)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct PpuSPointer;

impl RegisterMeta for PpuSPointer {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_PPU;
    const OFFSET: u32 = REG_PPU_S_POINTER;
}

impl Register<PpuSPointer> {
    /// Description: Selects which of the two shadow register groups is ready to be used.
    ///
    /// Bit width: 1
    /// Range of values: 0 = register group 0; 1 = register group 1.
    /// Known limitations: Only meaningful when ping-pong is being driven manually; when `pointer_pp_en` is set, hardware toggles this value automatically per `pointer_pp_mode`'s rule.
    /// Related registers: `pointer_pp_en`, `pointer_pp_mode`, `pointer_pp_clear`, `PpuOperationEnable::op_en` (and every register shadowed after it).
    pub fn pointer(&mut self, pointer: Bits<1>) -> &mut Self {
        self.set_field(PPU_S_POINTER_POINTER__MASK, unsafe {
            PPU_S_POINTER_POINTER(pointer.val())
        })
    }

    /// Description: Enables ping-pong toggling of the register group pointer.
    ///
    /// Bit width: 1
    /// Range of values: 0 = disable; 1 = enable.
    /// Known limitations: The toggle rule used once enabled is selected by `pointer_pp_mode`.
    /// Related registers: `pointer`, `pointer_pp_mode`, `pointer_pp_clear`.
    pub fn pointer_pp_en(&mut self, pointer_pp_en: Bits<1>) -> &mut Self {
        self.set_field(PPU_S_POINTER_POINTER_PP_EN__MASK, unsafe {
            PPU_S_POINTER_POINTER_PP_EN(pointer_pp_en.val())
        })
    }

    /// Description: Enables ping-pong toggling of the executer group.
    ///
    /// Bit width: 1
    /// Range of values: 0 = disable; 1 = enable.
    /// Known limitations: None documented.
    /// Related registers: `executer`, `executer_pp_clear`.
    pub fn executer_pp_en(&mut self, executer_pp_en: Bits<1>) -> &mut Self {
        self.set_field(PPU_S_POINTER_EXECUTER_PP_EN__MASK, unsafe {
            PPU_S_POINTER_EXECUTER_PP_EN(executer_pp_en.val())
        })
    }

    /// Description: Selects the rule used to toggle the register-group pointer during ping-pong.
    ///
    /// Bit width: 1
    /// Range of values: 0 = pointer ping-pongs by executer (e.g. if current executer is 0, next pointer toggles to 1); 1 = pointer ping-pongs by pointer (e.g. if current pointer is 0, next pointer toggles to 1).
    /// Known limitations: Only takes effect while `pointer_pp_en` is enabled.
    /// Related registers: `pointer`, `pointer_pp_en`, `executer_pp_en`.
    pub fn pointer_pp_mode(&mut self, pointer_pp_mode: Bits<1>) -> &mut Self {
        self.set_field(PPU_S_POINTER_POINTER_PP_MODE__MASK, unsafe {
            PPU_S_POINTER_POINTER_PP_MODE(pointer_pp_mode.val())
        })
    }

    /// Description: Clears the register group pointer back to 0.
    ///
    /// Bit width: 1
    /// Range of values: Write 1 to clear the pointer to 0; write 0 for no effect.
    /// Known limitations: TRM marks this attribute W1C (write-1-to-clear) — it is a self-clearing pulse, not a persistent state bit.
    /// Related registers: `pointer`, `pointer_pp_mode`.
    pub fn pointer_pp_clear(&mut self, pointer_pp_clear: Bits<1>) -> &mut Self {
        self.set_field(PPU_S_POINTER_POINTER_PP_CLEAR__MASK, unsafe {
            PPU_S_POINTER_POINTER_PP_CLEAR(pointer_pp_clear.val())
        })
    }

    /// Description: Clears the executer group pointer back to 0.
    ///
    /// Bit width: 1
    /// Range of values: Write 1 to clear the pointer to 0; write 0 for no effect.
    /// Known limitations: TRM marks this attribute W1C (write-1-to-clear) — it is a self-clearing pulse, not a persistent state bit.
    /// Related registers: `executer`, `executer_pp_en`.
    pub fn executer_pp_clear(&mut self, executer_pp_clear: Bits<1>) -> &mut Self {
        self.set_field(PPU_S_POINTER_EXECUTER_PP_CLEAR__MASK, unsafe {
            PPU_S_POINTER_EXECUTER_PP_CLEAR(executer_pp_clear.val())
        })
    }

    /// Description: Selects which of the two executer register groups is currently in use.
    ///
    /// Bit width: 1
    /// Range of values: 0 = executer group 0; 1 = executer group 1.
    /// Known limitations: TRM marks this bit RO (hardware-reported) — the setter exists for register-model completeness but writes may have no effect on real silicon.
    /// Related registers: `executer_pp_en`, `executer_pp_clear`, `PpuSStatus::status_0`/`status_1`.
    pub fn executer(&mut self, executer: Bits<1>) -> &mut Self {
        self.set_field(PPU_S_POINTER_EXECUTER__MASK, unsafe {
            PPU_S_POINTER_EXECUTER(executer.val())
        })
    }
}

// ========================================================================
// OPERATION_ENABLE (0x6008)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct PpuOperationEnable;

impl RegisterMeta for PpuOperationEnable {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_PPU;
    const OFFSET: u32 = REG_PPU_OPERATION_ENABLE;
}

impl Register<PpuOperationEnable> {
    /// Description: Triggers the PPU block to begin operating on the currently configured register group.
    ///
    /// Bit width: 1
    /// Range of values: 0 = disable; 1 = enable.
    /// Known limitations: This register and every register after it in the block are shadowed for ping-pong operation; the effective values used are gated by `PpuSPointer`'s pointer/executer selection and ping-pong enables.
    /// Related registers: `PpuSPointer` (pointer/executer ping-pong control); all subsequent PPU configuration registers (shadowed together with this one).
    pub fn op_en(&mut self, op_en: Bits<1>) -> &mut Self {
        self.set_field(PPU_OPERATION_ENABLE_OP_EN__MASK, unsafe {
            PPU_OPERATION_ENABLE_OP_EN(op_en.val())
        })
    }
}

// ========================================================================
// DATA_CUBE_IN_WIDTH (0x600C)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct PpuDataCubeInWidth;

impl RegisterMeta for PpuDataCubeInWidth {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_PPU;
    const OFFSET: u32 = REG_PPU_DATA_CUBE_IN_WIDTH;
}

impl Register<PpuDataCubeInWidth> {
    /// Description: Width of the input feature cube fed into pooling.
    ///
    /// Bit width: 13
    /// Range of values: 0-8191, encoded as (actual width - 1).
    /// Known limitations: Value must be programmed as the real width minus 1 (N-1 encoding).
    /// Related registers: `PpuDataCubeInHeight::cube_in_height`, `PpuDataCubeInChannel::cube_in_channel`, `PpuDataCubeOutWidth::cube_out_width`, `PpuPoolingKernelCfg` (kernel/stride sizing), `PpuPoolingPaddingCfg` (pad_left/pad_right).
    pub fn cube_in_width(&mut self, cube_in_width: Bits<13>) -> &mut Self {
        self.set_field(PPU_DATA_CUBE_IN_WIDTH_CUBE_IN_WIDTH__MASK, unsafe {
            PPU_DATA_CUBE_IN_WIDTH_CUBE_IN_WIDTH(cube_in_width.val())
        })
    }
}

// ========================================================================
// DATA_CUBE_IN_HEIGHT (0x6010)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct PpuDataCubeInHeight;

impl RegisterMeta for PpuDataCubeInHeight {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_PPU;
    const OFFSET: u32 = REG_PPU_DATA_CUBE_IN_HEIGHT;
}

impl Register<PpuDataCubeInHeight> {
    /// Description: Height of the input feature cube fed into pooling.
    ///
    /// Bit width: 13
    /// Range of values: 0-8191, encoded as (actual height - 1).
    /// Known limitations: Value must be programmed as the real height minus 1 (N-1 encoding).
    /// Related registers: `PpuDataCubeInWidth::cube_in_width`, `PpuDataCubeInChannel::cube_in_channel`, `PpuDataCubeOutHeight::cube_out_height`, `PpuPoolingKernelCfg` (kernel/stride sizing), `PpuPoolingPaddingCfg` (pad_top/pad_bottom).
    pub fn cube_in_height(&mut self, cube_in_height: Bits<13>) -> &mut Self {
        self.set_field(PPU_DATA_CUBE_IN_HEIGHT_CUBE_IN_HEIGHT__MASK, unsafe {
            PPU_DATA_CUBE_IN_HEIGHT_CUBE_IN_HEIGHT(cube_in_height.val())
        })
    }
}

// ========================================================================
// DATA_CUBE_IN_CHANNEL (0x6014)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct PpuDataCubeInChannel;

impl RegisterMeta for PpuDataCubeInChannel {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_PPU;
    const OFFSET: u32 = REG_PPU_DATA_CUBE_IN_CHANNEL;
}

impl Register<PpuDataCubeInChannel> {
    /// Description: Channel count of the input feature cube fed into pooling.
    ///
    /// Bit width: 13
    /// Range of values: 0-8191, encoded as (actual channel count - 1).
    /// Known limitations: Value must be programmed as the real channel count minus 1 (N-1 encoding).
    /// Related registers: `PpuDataCubeInWidth::cube_in_width`, `PpuDataCubeInHeight::cube_in_height`, `PpuDataCubeOutChannel::cube_out_channel`.
    pub fn cube_in_channel(&mut self, cube_in_channel: Bits<13>) -> &mut Self {
        self.set_field(PPU_DATA_CUBE_IN_CHANNEL_CUBE_IN_CHANNEL__MASK, unsafe {
            PPU_DATA_CUBE_IN_CHANNEL_CUBE_IN_CHANNEL(cube_in_channel.val())
        })
    }
}

// ========================================================================
// DATA_CUBE_OUT_WIDTH (0x6018)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct PpuDataCubeOutWidth;

impl RegisterMeta for PpuDataCubeOutWidth {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_PPU;
    const OFFSET: u32 = REG_PPU_DATA_CUBE_OUT_WIDTH;
}

impl Register<PpuDataCubeOutWidth> {
    /// Description: Width of the pooling output cube.
    ///
    /// Bit width: 13
    /// Range of values: 0-8191, encoded as (actual output width - 1).
    /// Known limitations: Value must be programmed as the real output width minus 1 (N-1 encoding); must be consistent with `cube_in_width`, kernel width/stride, and padding.
    /// Related registers: `PpuDataCubeInWidth::cube_in_width`, `PpuPoolingKernelCfg::kernel_width`/`kernel_stride_width`, `PpuPoolingPaddingCfg::pad_left`/`pad_right`.
    pub fn cube_out_width(&mut self, cube_out_width: Bits<13>) -> &mut Self {
        self.set_field(PPU_DATA_CUBE_OUT_WIDTH_CUBE_OUT_WIDTH__MASK, unsafe {
            PPU_DATA_CUBE_OUT_WIDTH_CUBE_OUT_WIDTH(cube_out_width.val())
        })
    }
}

// ========================================================================
// DATA_CUBE_OUT_HEIGHT (0x601C)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct PpuDataCubeOutHeight;

impl RegisterMeta for PpuDataCubeOutHeight {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_PPU;
    const OFFSET: u32 = REG_PPU_DATA_CUBE_OUT_HEIGHT;
}

impl Register<PpuDataCubeOutHeight> {
    /// Description: Height of the pooling output cube.
    ///
    /// Bit width: 13
    /// Range of values: 0-8191, encoded as (actual output height - 1).
    /// Known limitations: Value must be programmed as the real output height minus 1 (N-1 encoding); must be consistent with `cube_in_height`, kernel height/stride, and padding.
    /// Related registers: `PpuDataCubeInHeight::cube_in_height`, `PpuPoolingKernelCfg::kernel_height`/`kernel_stride_height`, `PpuPoolingPaddingCfg::pad_top`/`pad_bottom`.
    pub fn cube_out_height(&mut self, cube_out_height: Bits<13>) -> &mut Self {
        self.set_field(PPU_DATA_CUBE_OUT_HEIGHT_CUBE_OUT_HEIGHT__MASK, unsafe {
            PPU_DATA_CUBE_OUT_HEIGHT_CUBE_OUT_HEIGHT(cube_out_height.val())
        })
    }
}

// ========================================================================
// DATA_CUBE_OUT_CHANNEL (0x6020)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct PpuDataCubeOutChannel;

impl RegisterMeta for PpuDataCubeOutChannel {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_PPU;
    const OFFSET: u32 = REG_PPU_DATA_CUBE_OUT_CHANNEL;
}

impl Register<PpuDataCubeOutChannel> {
    /// Description: Channel count of the pooling output cube.
    ///
    /// Bit width: 13
    /// Range of values: 0-8191, encoded as (actual output channel count - 1).
    /// Known limitations: Value must be programmed as the real output channel count minus 1 (N-1 encoding); pooling does not change channel count so this normally matches `cube_in_channel`.
    /// Related registers: `PpuDataCubeInChannel::cube_in_channel`.
    pub fn cube_out_channel(&mut self, cube_out_channel: Bits<13>) -> &mut Self {
        self.set_field(PPU_DATA_CUBE_OUT_CHANNEL_CUBE_OUT_CHANNEL__MASK, unsafe {
            PPU_DATA_CUBE_OUT_CHANNEL_CUBE_OUT_CHANNEL(cube_out_channel.val())
        })
    }
}

// ========================================================================
// OPERATION_MODE_CFG (0x6024)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct PpuOperationModeCfg;

impl RegisterMeta for PpuOperationModeCfg {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_PPU;
    const OFFSET: u32 = REG_PPU_OPERATION_MODE_CFG;
}

impl Register<PpuOperationModeCfg> {
    /// Description: Selects the pooling algorithm applied to each kernel window.
    ///
    /// Bit width: 2
    /// Range of values: 0 = average pooling; 1 = max pooling; 2 = min pooling; 3 = reserved.
    /// Known limitations: Average pooling implements the divide as a fixed-point multiply by the kernel-dimension reciprocal (×2^16) rather than true division, so `PpuRecipKernelWidth`/`PpuRecipKernelHeight` must be programmed to match the kernel size whenever this is set to average.
    /// Related registers: `PpuPoolingKernelCfg` (kernel size), `PpuRecipKernelWidth::recip_kernel_width`, `PpuRecipKernelHeight::recip_kernel_height`.
    pub fn pooling_method(&mut self, pooling_method: Bits<2>) -> &mut Self {
        self.set_field(PPU_OPERATION_MODE_CFG_POOLING_METHOD__MASK, unsafe {
            PPU_OPERATION_MODE_CFG_POOLING_METHOD(pooling_method.val())
        })
    }

    /// Description: Selects whether PPU's input cube comes from the DPU pipeline or from outside memory via PPU_RDMA (standalone "flying" mode).
    ///
    /// Bit width: 1
    /// Range of values: 0 = DPU (pipelined directly after DPU's output); 1 = Outside (fed by PPU_RDMA, PPU running standalone).
    /// Known limitations: When set to Outside/flying, PPU_RDMA's fetch-config registers (src_base_addr, cube_in_width/height/channel, data_format, etc.) must be configured instead of relying on the DPU pipeline handoff.
    /// Related registers: `PpuDataFormat::dpu_flyin`; PPU_RDMA block registers (offset range 0x7000-0x7fff).
    pub fn flying_mode(&mut self, flying_mode: Bits<1>) -> &mut Self {
        self.set_field(PPU_OPERATION_MODE_CFG_FLYING_MODE__MASK, unsafe {
            PPU_OPERATION_MODE_CFG_FLYING_MODE(flying_mode.val())
        })
    }

    /// Description: Use count value for the pooling operation.
    ///
    /// Bit width: 3
    /// Range of values: 0-7.
    /// Known limitations: None documented beyond the field name in the TRM.
    /// Related registers: None.
    pub fn use_cnt(&mut self, use_cnt: Bits<3>) -> &mut Self {
        self.set_field(PPU_OPERATION_MODE_CFG_USE_CNT__MASK, unsafe {
            PPU_OPERATION_MODE_CFG_USE_CNT(use_cnt.val())
        })
    }

    /// Description: Number of pixels from the end of the width to the end of the shape line.
    ///
    /// Bit width: 13
    /// Range of values: 0-8191.
    /// Known limitations: Bookkeeping for end-of-row alignment; interacts with non-align output mode.
    /// Related registers: `PpuDataCubeInWidth::cube_in_width`, `PpuDataCubeOutWidth::cube_out_width`, `PpuMiscCtrl::nonalign`/`surf_len`.
    pub fn notch_addr(&mut self, notch_addr: Bits<13>) -> &mut Self {
        self.set_field(PPU_OPERATION_MODE_CFG_NOTCH_ADDR__MASK, unsafe {
            PPU_OPERATION_MODE_CFG_NOTCH_ADDR(notch_addr.val())
        })
    }

    /// Description: Enables outputting the position (argmax/argmin index) of each pooling kernel window instead of just its value.
    ///
    /// Bit width: 1
    /// Range of values: 0 = disable; 1 = enable.
    /// Known limitations: When enabled, `PpuDataFormat::index_add` must be set to dst_surface_stride times the number of cube surfaces (8 bytes per surface); when disabled, `index_add` must instead equal dst_surface_stride.
    /// Related registers: `PpuDataFormat::index_add`, `PpuDstSurfStride::dst_surf_stride`.
    pub fn index_en(&mut self, index_en: Bits<1>) -> &mut Self {
        self.set_field(PPU_OPERATION_MODE_CFG_INDEX_EN__MASK, unsafe {
            PPU_OPERATION_MODE_CFG_INDEX_EN(index_en.val())
        })
    }
}

// ========================================================================
// POOLING_KERNEL_CFG (0x6034)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct PpuPoolingKernelCfg;

impl RegisterMeta for PpuPoolingKernelCfg {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_PPU;
    const OFFSET: u32 = REG_PPU_POOLING_KERNEL_CFG;
}

impl Register<PpuPoolingKernelCfg> {
    /// Description: Width of the pooling kernel window.
    ///
    /// Bit width: 4
    /// Range of values: 0-15, encoded as (actual kernel width - 1).
    /// Known limitations: For average pooling, `PpuRecipKernelWidth::recip_kernel_width` must be programmed as the reciprocal of this width (×2^16 fixed point) to implement the divide as a multiply.
    /// Related registers: `kernel_stride_width`, `PpuRecipKernelWidth::recip_kernel_width`, `PpuOperationModeCfg::pooling_method`.
    pub fn kernel_width(&mut self, kernel_width: Bits<4>) -> &mut Self {
        self.set_field(PPU_POOLING_KERNEL_CFG_KERNEL_WIDTH__MASK, unsafe {
            PPU_POOLING_KERNEL_CFG_KERNEL_WIDTH(kernel_width.val())
        })
    }

    /// Description: Height of the pooling kernel window.
    ///
    /// Bit width: 4
    /// Range of values: 0-15, encoded as (actual kernel height - 1).
    /// Known limitations: For average pooling, `PpuRecipKernelHeight::recip_kernel_height` must be programmed as the reciprocal of this height (×2^16 fixed point) to implement the divide as a multiply.
    /// Related registers: `kernel_stride_height`, `PpuRecipKernelHeight::recip_kernel_height`, `PpuOperationModeCfg::pooling_method`.
    pub fn kernel_height(&mut self, kernel_height: Bits<4>) -> &mut Self {
        self.set_field(PPU_POOLING_KERNEL_CFG_KERNEL_HEIGHT__MASK, unsafe {
            PPU_POOLING_KERNEL_CFG_KERNEL_HEIGHT(kernel_height.val())
        })
    }

    /// Description: Horizontal stride between successive pooling kernel windows.
    ///
    /// Bit width: 4
    /// Range of values: 0-15, encoded as (actual stride width - 1).
    /// Known limitations: Determines `cube_out_width` in conjunction with `kernel_width`, `cube_in_width`, and padding.
    /// Related registers: `kernel_width`, `PpuDataCubeInWidth::cube_in_width`, `PpuDataCubeOutWidth::cube_out_width`.
    pub fn kernel_stride_width(&mut self, kernel_stride_width: Bits<4>) -> &mut Self {
        self.set_field(PPU_POOLING_KERNEL_CFG_KERNEL_STRIDE_WIDTH__MASK, unsafe {
            PPU_POOLING_KERNEL_CFG_KERNEL_STRIDE_WIDTH(kernel_stride_width.val())
        })
    }

    /// Description: Vertical stride between successive pooling kernel windows.
    ///
    /// Bit width: 4
    /// Range of values: 0-15, encoded as (actual stride height - 1).
    /// Known limitations: Determines `cube_out_height` in conjunction with `kernel_height`, `cube_in_height`, and padding.
    /// Related registers: `kernel_height`, `PpuDataCubeInHeight::cube_in_height`, `PpuDataCubeOutHeight::cube_out_height`.
    pub fn kernel_stride_height(&mut self, kernel_stride_height: Bits<4>) -> &mut Self {
        self.set_field(PPU_POOLING_KERNEL_CFG_KERNEL_STRIDE_HEIGHT__MASK, unsafe {
            PPU_POOLING_KERNEL_CFG_KERNEL_STRIDE_HEIGHT(kernel_stride_height.val())
        })
    }
}

// ========================================================================
// RECIP_KERNEL_WIDTH (0x6038)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct PpuRecipKernelWidth;

impl RegisterMeta for PpuRecipKernelWidth {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_PPU;
    const OFFSET: u32 = REG_PPU_RECIP_KERNEL_WIDTH;
}

impl Register<PpuRecipKernelWidth> {
    /// Description: Precomputed reciprocal of the pooling kernel width, used to implement average pooling's divide as a multiply.
    ///
    /// Bit width: 17
    /// Range of values: 0-131071; fixed-point value equal to the reciprocal of the kernel width multiplied by 2^16.
    /// Known limitations: Value is the kernel-width reciprocal times 2^16 (i.e. `round(65536 / kernel_width)`); only meaningful when `PpuOperationModeCfg::pooling_method` selects average pooling — ignored for max/min pooling.
    /// Related registers: `PpuPoolingKernelCfg::kernel_width`, `PpuOperationModeCfg::pooling_method`, `PpuRecipKernelHeight::recip_kernel_height`.
    pub fn recip_kernel_width(&mut self, recip_kernel_width: Bits<17>) -> &mut Self {
        self.set_field(PPU_RECIP_KERNEL_WIDTH_RECIP_KERNEL_WIDTH__MASK, unsafe {
            PPU_RECIP_KERNEL_WIDTH_RECIP_KERNEL_WIDTH(recip_kernel_width.val())
        })
    }
}

// ========================================================================
// RECIP_KERNEL_HEIGHT (0x603C)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct PpuRecipKernelHeight;

impl RegisterMeta for PpuRecipKernelHeight {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_PPU;
    const OFFSET: u32 = REG_PPU_RECIP_KERNEL_HEIGHT;
}

impl Register<PpuRecipKernelHeight> {
    /// Description: Precomputed reciprocal of the pooling kernel height, used to implement average pooling's divide as a multiply.
    ///
    /// Bit width: 17
    /// Range of values: 0-131071; fixed-point value equal to the reciprocal of the kernel height multiplied by 2^16.
    /// Known limitations: Value is the kernel-height reciprocal times 2^16 (i.e. `round(65536 / kernel_height)`); only meaningful when `PpuOperationModeCfg::pooling_method` selects average pooling — ignored for max/min pooling.
    /// Related registers: `PpuPoolingKernelCfg::kernel_height`, `PpuOperationModeCfg::pooling_method`, `PpuRecipKernelWidth::recip_kernel_width`.
    pub fn recip_kernel_height(&mut self, recip_kernel_height: Bits<17>) -> &mut Self {
        self.set_field(PPU_RECIP_KERNEL_HEIGHT_RECIP_KERNEL_HEIGHT__MASK, unsafe {
            PPU_RECIP_KERNEL_HEIGHT_RECIP_KERNEL_HEIGHT(recip_kernel_height.val())
        })
    }
}

// ========================================================================
// POOLING_PADDING_CFG (0x6040)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct PpuPoolingPaddingCfg;

impl RegisterMeta for PpuPoolingPaddingCfg {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_PPU;
    const OFFSET: u32 = REG_PPU_POOLING_PADDING_CFG;
}

impl Register<PpuPoolingPaddingCfg> {
    /// Description: Number of padding pixels added to the left side of the input cube before pooling.
    ///
    /// Bit width: 3
    /// Range of values: 0-7.
    /// Known limitations: The value written into the padded region is set separately via `PpuPaddingValue1Cfg`/`PpuPaddingValue2Cfg`.
    /// Related registers: `pad_right`, `PpuPaddingValue1Cfg::pad_value_0`, `PpuPaddingValue2Cfg::pad_value_1`, `PpuDataCubeInWidth::cube_in_width`.
    pub fn pad_left(&mut self, pad_left: Bits<3>) -> &mut Self {
        self.set_field(PPU_POOLING_PADDING_CFG_PAD_LEFT__MASK, unsafe {
            PPU_POOLING_PADDING_CFG_PAD_LEFT(pad_left.val())
        })
    }

    /// Description: Number of padding pixels added to the top side of the input cube before pooling.
    ///
    /// Bit width: 3
    /// Range of values: 0-7.
    /// Known limitations: The value written into the padded region is set separately via `PpuPaddingValue1Cfg`/`PpuPaddingValue2Cfg`.
    /// Related registers: `pad_bottom`, `PpuPaddingValue1Cfg::pad_value_0`, `PpuPaddingValue2Cfg::pad_value_1`, `PpuDataCubeInHeight::cube_in_height`.
    pub fn pad_top(&mut self, pad_top: Bits<3>) -> &mut Self {
        self.set_field(PPU_POOLING_PADDING_CFG_PAD_TOP__MASK, unsafe {
            PPU_POOLING_PADDING_CFG_PAD_TOP(pad_top.val())
        })
    }

    /// Description: Number of padding pixels added to the right side of the input cube before pooling.
    ///
    /// Bit width: 3
    /// Range of values: 0-7.
    /// Known limitations: The value written into the padded region is set separately via `PpuPaddingValue1Cfg`/`PpuPaddingValue2Cfg`.
    /// Related registers: `pad_left`, `PpuPaddingValue1Cfg::pad_value_0`, `PpuPaddingValue2Cfg::pad_value_1`, `PpuDataCubeInWidth::cube_in_width`.
    pub fn pad_right(&mut self, pad_right: Bits<3>) -> &mut Self {
        self.set_field(PPU_POOLING_PADDING_CFG_PAD_RIGHT__MASK, unsafe {
            PPU_POOLING_PADDING_CFG_PAD_RIGHT(pad_right.val())
        })
    }

    /// Description: Number of padding pixels added to the bottom side of the input cube before pooling.
    ///
    /// Bit width: 3
    /// Range of values: 0-7.
    /// Known limitations: The value written into the padded region is set separately via `PpuPaddingValue1Cfg`/`PpuPaddingValue2Cfg`.
    /// Related registers: `pad_top`, `PpuPaddingValue1Cfg::pad_value_0`, `PpuPaddingValue2Cfg::pad_value_1`, `PpuDataCubeInHeight::cube_in_height`.
    pub fn pad_bottom(&mut self, pad_bottom: Bits<3>) -> &mut Self {
        self.set_field(PPU_POOLING_PADDING_CFG_PAD_BOTTOM__MASK, unsafe {
            PPU_POOLING_PADDING_CFG_PAD_BOTTOM(pad_bottom.val())
        })
    }
}

// ========================================================================
// PADDING_VALUE_1_CFG (0x6044)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct PpuPaddingValue1Cfg;

impl RegisterMeta for PpuPaddingValue1Cfg {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_PPU;
    const OFFSET: u32 = REG_PPU_PADDING_VALUE_1_CFG;
}

impl Register<PpuPaddingValue1Cfg> {
    /// Description: Low 32 bits (bits [31:0]) of the value written into padded regions of the input cube.
    ///
    /// Bit width: 32
    /// Range of values: Full 32-bit value.
    /// Known limitations: Forms only the low bits of a wider (35-bit) padding value; the remaining bits [34:32] live in `PpuPaddingValue2Cfg::pad_value_1`.
    /// Related registers: `PpuPaddingValue2Cfg::pad_value_1`, `PpuPoolingPaddingCfg` (pad_left/pad_top/pad_right/pad_bottom).
    pub fn pad_value_0(&mut self, pad_value_0: Bits<32>) -> &mut Self {
        self.set_field(PPU_PADDING_VALUE_1_CFG_PAD_VALUE_0__MASK, unsafe {
            PPU_PADDING_VALUE_1_CFG_PAD_VALUE_0(pad_value_0.val())
        })
    }
}

// ========================================================================
// PADDING_VALUE_2_CFG (0x6048)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct PpuPaddingValue2Cfg;

impl RegisterMeta for PpuPaddingValue2Cfg {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_PPU;
    const OFFSET: u32 = REG_PPU_PADDING_VALUE_2_CFG;
}

impl Register<PpuPaddingValue2Cfg> {
    /// Description: High bits (bits [34:32]) of the value written into padded regions of the input cube.
    ///
    /// Bit width: 3
    /// Range of values: 0-7.
    /// Known limitations: Forms only the top bits of a wider (35-bit) padding value split across two registers; the low bits live in `PpuPaddingValue1Cfg::pad_value_0`.
    /// Related registers: `PpuPaddingValue1Cfg::pad_value_0`, `PpuPoolingPaddingCfg` (pad_left/pad_top/pad_right/pad_bottom).
    pub fn pad_value_1(&mut self, pad_value_1: Bits<3>) -> &mut Self {
        self.set_field(PPU_PADDING_VALUE_2_CFG_PAD_VALUE_1__MASK, unsafe {
            PPU_PADDING_VALUE_2_CFG_PAD_VALUE_1(pad_value_1.val())
        })
    }
}

// ========================================================================
// DST_BASE_ADDR (0x6070)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct PpuDstBaseAddr;

impl RegisterMeta for PpuDstBaseAddr {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_PPU;
    const OFFSET: u32 = REG_PPU_DST_BASE_ADDR;
}

impl Register<PpuDstBaseAddr> {
    /// Description: Base address in memory the pooling output cube is written to.
    ///
    /// Bit width: 28
    /// Range of values: Any value representing bits [31:4] of a 32-bit address.
    /// Known limitations: Occupies bits [31:4] of the register; bits [3:0] are reserved/read-only, so the effective address must be 16-byte aligned.
    /// Related registers: `PpuDstSurfStride::dst_surf_stride`, `PpuDataFormat::index_add`.
    pub fn dst_base_addr(&mut self, dst_base_addr: Bits<28>) -> &mut Self {
        self.set_field(PPU_DST_BASE_ADDR_DST_BASE_ADDR__MASK, unsafe {
            PPU_DST_BASE_ADDR_DST_BASE_ADDR(dst_base_addr.val())
        })
    }
}

// ========================================================================
// DST_SURF_STRIDE (0x607C)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct PpuDstSurfStride;

impl RegisterMeta for PpuDstSurfStride {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_PPU;
    const OFFSET: u32 = REG_PPU_DST_SURF_STRIDE;
}

impl Register<PpuDstSurfStride> {
    /// Description: Stride (area) of one output surface — the byte span occupied by one surface of the output shape.
    ///
    /// Bit width: 28
    /// Range of values: Any value representing bits [31:4] of a 32-bit stride.
    /// Known limitations: Occupies bits [31:4] of the register; bits [3:0] are reserved/read-only.
    /// Related registers: `PpuDstBaseAddr::dst_base_addr`, `PpuDataFormat::index_add` (which equals this value, or this value times surface count when `index_en` is set).
    pub fn dst_surf_stride(&mut self, dst_surf_stride: Bits<28>) -> &mut Self {
        self.set_field(PPU_DST_SURF_STRIDE_DST_SURF_STRIDE__MASK, unsafe {
            PPU_DST_SURF_STRIDE_DST_SURF_STRIDE(dst_surf_stride.val())
        })
    }
}

// ========================================================================
// DATA_FORMAT (0x6084)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct PpuDataFormat;

impl RegisterMeta for PpuDataFormat {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_PPU;
    const OFFSET: u32 = REG_PPU_DATA_FORMAT;
}

impl Register<PpuDataFormat> {
    /// Description: Selects the data-processing precision PPU operates in.
    ///
    /// Bit width: 3
    /// Range of values: Not itemized for this specific register in the TRM (only "Process precision" is given); by analogy with the same-width `proc_precision` field on DPU's `dpu_data_format` register, likely selects among int8/int16/fp16/bf16/int32/fp32-style encodings, but the exact PPU enum is unconfirmed.
    /// Known limitations: Exact enum mapping not confirmed for this register — cross-check against a captured `.rknn` regcmd before relying on a specific numeric value.
    /// Related registers: None.
    pub fn proc_precision(&mut self, proc_precision: Bits<3>) -> &mut Self {
        self.set_field(PPU_DATA_FORMAT_PROC_PRECISION__MASK, unsafe {
            PPU_DATA_FORMAT_PROC_PRECISION(proc_precision.val())
        })
    }

    /// Description: Indicates the input data comes from DPU, where DPU itself is fed from outside (DPU running in its own flying mode).
    ///
    /// Bit width: 1
    /// Range of values: 0 = not this case; 1 = set when data is from DPU and DPU's data is from outside.
    /// Known limitations: Distinct from `PpuOperationModeCfg::flying_mode` (which selects PPU's own source, DPU vs PPU_RDMA) — this bit instead describes DPU's own upstream source when PPU is pipelined after DPU.
    /// Related registers: `PpuOperationModeCfg::flying_mode`; DPU_RDMA's `flying_mode` field (DPU's own standalone mode).
    pub fn dpu_flyin(&mut self, dpu_flyin: Bits<1>) -> &mut Self {
        self.set_field(PPU_DATA_FORMAT_DPU_FLYIN__MASK, unsafe {
            PPU_DATA_FORMAT_DPU_FLYIN(dpu_flyin.val())
        })
    }

    /// Description: Address increment used per index entry when outputting pooling-window positions.
    ///
    /// Bit width: 28
    /// Range of values: If `index_en` is enabled, this equals dst_surface_stride multiplied by the number of cube surfaces (8 bytes per surface); otherwise it must equal dst_surface_stride.
    /// Known limitations: Must be recomputed whenever `index_en`, `dst_surf_stride`, or the cube's surface count changes.
    /// Related registers: `PpuOperationModeCfg::index_en`, `PpuDstSurfStride::dst_surf_stride`.
    pub fn index_add(&mut self, index_add: Bits<28>) -> &mut Self {
        self.set_field(PPU_DATA_FORMAT_INDEX_ADD__MASK, unsafe {
            PPU_DATA_FORMAT_INDEX_ADD(index_add.val())
        })
    }
}

// ========================================================================
// MISC_CTRL (0x60DC)
// ========================================================================
#[derive(Debug, Clone, Copy)]
pub struct PpuMiscCtrl;

impl RegisterMeta for PpuMiscCtrl {
    const DOMAIN: u32 = crate::rocket::builders::DOMAIN_PPU;
    const OFFSET: u32 = REG_PPU_MISC_CTRL;
}

impl Register<PpuMiscCtrl> {
    /// Description: Selects the AXI burst length used for PPU's output write DMA.
    ///
    /// Bit width: 4
    /// Range of values: 3 = Burst4; 7 = Burst8; 15 = Burst16 (other values not documented/reserved).
    /// Known limitations: Only the three listed encodings (3, 7, 15) are documented in the TRM.
    /// Related registers: `nonalign`, `surf_len`.
    pub fn burst_len(&mut self, burst_len: Bits<4>) -> &mut Self {
        self.set_field(PPU_MISC_CTRL_BURST_LEN__MASK, unsafe {
            PPU_MISC_CTRL_BURST_LEN(burst_len.val())
        })
    }

    /// Description: Enables non-align output mode, for feature-map sizes that don't fit the normal 8-byte-per-pixel aligned output layout.
    ///
    /// Bit width: 1
    /// Range of values: 0 = disable; 1 = enable.
    /// Known limitations: When enabled, `surf_len` must be configured to describe the surface count length for the non-aligned layout.
    /// Related registers: `surf_len`, `PpuOperationModeCfg::notch_addr`.
    pub fn nonalign(&mut self, nonalign: Bits<1>) -> &mut Self {
        self.set_field(PPU_MISC_CTRL_NONALIGN__MASK, unsafe {
            PPU_MISC_CTRL_NONALIGN(nonalign.val())
        })
    }

    /// Description: Enables outputting to multiple surfaces.
    ///
    /// Bit width: 1
    /// Range of values: 0 = disable; 1 = enable.
    /// Known limitations: None documented beyond the enable/disable meaning.
    /// Related registers: `PpuDstSurfStride::dst_surf_stride`, `surf_len`.
    pub fn mc_surf_out(&mut self, mc_surf_out: Bits<1>) -> &mut Self {
        self.set_field(PPU_MISC_CTRL_MC_SURF_OUT__MASK, unsafe {
            PPU_MISC_CTRL_MC_SURF_OUT(mc_surf_out.val())
        })
    }

    /// Description: Surface count length used by non-align output mode.
    ///
    /// Bit width: 16
    /// Range of values: 0-65535.
    /// Known limitations: Only meaningful when `nonalign` is enabled.
    /// Related registers: `nonalign`, `mc_surf_out`.
    pub fn surf_len(&mut self, surf_len: Bits<16>) -> &mut Self {
        self.set_field(PPU_MISC_CTRL_SURF_LEN__MASK, unsafe {
            PPU_MISC_CTRL_SURF_LEN(surf_len.val())
        })
    }
}
