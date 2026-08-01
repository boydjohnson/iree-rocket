//! Prints the PPU/PPU_RDMA S_POINTER words the pooling builder emits, so the
//! value can be compared against the vendor's 0x0e without hardware.

use iree_rocket_hal::rocket::pooling::{
    PoolingBuffers, PoolingMethod, PoolingPlan, PoolingPrecision, PoolingShape,
};

fn main() {
    let shape = PoolingShape {
        input_width: 8,
        input_height: 8,
        input_channels: 8,
        output_width: 4,
        output_height: 4,
        output_channels: 8,
        precision: PoolingPrecision::Fp16,
        kernel_width: 2,
        kernel_height: 2,
        stride_x: 2,
        stride_y: 2,
        method: PoolingMethod::Max,
        pad_left: 0,
        pad_top: 0,
        pad_right: 0,
        pad_bottom: 0,
        pad_value: 0,
    };

    let plan = PoolingPlan::new(shape);
    let bufs = PoolingBuffers {
        input_addr: 0x1000_0000,
        output_addr: 0x2000_0000,
    };
    let programs = plan.programs_with_buffers(&bufs);

    // The two S_POINTER writes are the first two commands of every tile task.
    for (task, cmds) in programs.iter().enumerate() {
        for (which, cmd) in cmds.iter().take(2).enumerate() {
            let raw = cmd.0;
            println!(
                "  task {task} cmd {which}: raw=0x{raw:016x} domain=0x{:04x} \
                 value=0x{:08x} offset=0x{:04x}",
                (raw >> 48) as u16,
                ((raw >> 16) & 0xffff_ffff) as u32,
                (raw & 0xffff) as u16,
            );
        }
    }
}
