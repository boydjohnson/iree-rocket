//! Prints the ConvPlan CBUF bank split and tile/job count for a given
//! (width, height, stride, cin, cout, kernel) shape, with zero hardware
//! I/O -- pure planning logic. Used to bulk-classify real-model conv
//! shapes for how extreme their CBUF partition is, as a cheap pre-filter
//! before spending hardware time on the poison-predecessor bug.
//!
//!   cargo run -p iree-rocket-hal --example dump_conv_plan -- \
//!       <width> <height> <stride> <cin> <cout> <kernel>

use iree_rocket_hal::rocket::conv::{ConvPlan, Shape, feature_grains};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let value = |index: usize| args[index].parse::<u32>().expect("numeric argument");

    let width = value(0);
    let height = value(1);
    let stride = value(2);
    let cin = value(3);
    let cout = value(4);
    let kernel = value(5) as usize;

    let shape = Shape::with_precision(
        width,
        height,
        stride,
        cin,
        cout,
        iree_rocket_hal::rocket::conv::Precision::Fp16,
    );
    let kernels = [kernel, kernel];
    let plan = ConvPlan::new(shape, kernels);

    println!(
        "{width}x{height} cin={cin} cout={cout} k={kernel}  data_banks={} weight_banks={} tiles={}",
        plan.data_banks(),
        plan.weight_banks(),
        plan.tiles().len(),
    );
    for (i, tile) in plan.tiles().iter().enumerate() {
        let grains = feature_grains(kernels, &tile.rows);
        println!(
            "  tile {i}: in_rows={} pad_top={} out_first={} out_rows={}  feature_grains={grains}",
            tile.rows.in_rows, tile.rows.pad_top, tile.rows.out_first, tile.rows.out_rows,
        );
    }
}
