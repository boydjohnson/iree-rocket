//! Prints ConvPlan's actual real-dispatch-path register program for one
//! tile, decoded (domain, offset, value), in the same format
//! `sweep_extract.py` uses for vendor captures -- for direct register-by-
//! register diffing against a real vendor capture of the same shape.
//! Goes through `ConvPlan::programs()` itself (not `conv_2d_tile`/
//! `dump_conv`'s single-tile path), so it reflects whatever real compiled
//! dispatches actually submit, including the experimental feature_grains
//! cap if one is active.
//!
//!   cargo run -p iree-rocket-hal --example dump_conv_plan_regcmd -- \
//!       <width> <height> <stride> <cin> <cout> <kernel> <tile_index> \
//!       [pad_top] [pad_left]
//!
//! `width`/`height` are the `Shape`'s own extent -- the *padded* input
//! extent when the caller supplies explicit `pad_top`/`pad_left` (matching
//! how the real compiler programs a `linalg` op with no implicit-padding
//! attribute, e.g. a physically pre-padded 66x66 input for a 3x3 kernel
//! producing a 64x64 output), or the implicit-SAME-padded extent (input ==
//! output) when `pad_top`/`pad_left` are omitted.

use iree_rocket_hal::rocket::conv::{ConvPlan, Shape};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let value = |index: usize| args[index].parse::<u32>().expect("numeric argument");

    let width = value(0);
    let height = value(1);
    let stride = value(2);
    let cin = value(3);
    let cout = value(4);
    let kernel = value(5) as usize;
    let tile_index = value(6) as usize;

    let mut shape = Shape::with_precision(
        width,
        height,
        stride,
        cin,
        cout,
        iree_rocket_hal::rocket::conv::Precision::Fp16,
    );
    if args.len() > 7 {
        let pad_top = value(7) as usize;
        let pad_left = value(8) as usize;
        shape = shape.with_padding([pad_top, pad_left]);
    }
    let kernels = [kernel, kernel];
    let plan = ConvPlan::new(shape, kernels);
    let programs = plan.programs();
    let program = &programs[tile_index];

    eprintln!(
        "tile {tile_index}: in_rows={} pad_top={} out_first={} out_rows={} data_banks={} weight_banks={}",
        plan.tiles()[tile_index].rows.in_rows,
        plan.tiles()[tile_index].rows.pad_top,
        plan.tiles()[tile_index].rows.out_first,
        plan.tiles()[tile_index].rows.out_rows,
        plan.data_banks(),
        plan.weight_banks(),
    );

    for command in program {
        let domain = (command.0 >> 48) & 0xffff;
        let value = (command.0 >> 16) & 0xffff_ffff;
        let offset = command.0 & 0xffff;
        println!("{domain:#06x}\t{offset:#06x}\t{value:#010x}");
    }
}
