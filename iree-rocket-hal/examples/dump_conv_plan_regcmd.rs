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
//! `ROCKET_DUMP_DEPTHWISE=1` builds a depthwise shape instead of a dense
//! one. Useful for diffing which registers a depthwise program sets against
//! a dense one: a regcmd program is a register *delta*, so a register one
//! kind of dispatch writes and the other leaves alone is inherited stale
//! when the two share a command buffer.
//!
//! `ROCKET_DUMP_PRECISION=int8acc` switches the shape to
//! `Int8Accumulator` (zero points zeroed, unit multiplier) instead of the
//! default fp16, which is what the accumulator output-parity work needs --
//! the parity rule only applies to accumulator output.
//!
//! `width`/`height` are the `Shape`'s own extent -- the *padded* input
//! extent when the caller supplies explicit `pad_top`/`pad_left` (matching
//! how the real compiler programs a `linalg` op with no implicit-padding
//! attribute, e.g. a physically pre-padded 66x66 input for a 3x3 kernel
//! producing a 64x64 output), or the implicit-SAME-padded extent (input ==
//! output) when `pad_top`/`pad_left` are omitted.

use iree_rocket_hal::rocket::conv::{ConvPlan, Multiplier, Precision, Quantization, Shape};

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

    let precision = match std::env::var("ROCKET_DUMP_PRECISION").as_deref() {
        Ok("int8acc") => Precision::Int8Accumulator(Quantization {
            input_zero_point: 0,
            output_zero_point: 0,
            weight_zero_point: 0,
            input_scale: 1.0,
            weights_scale: 1.0,
            multiplier: Multiplier { scale: 1, shift: 0 },
        }),
        _ => Precision::Fp16,
    };
    let mut shape = Shape::with_precision(width, height, stride, cin, cout, precision);
    if std::env::var("ROCKET_DUMP_DEPTHWISE").is_ok() {
        shape = shape.with_depthwise();
    }
    if args.len() > 7 {
        let pad_top = value(7) as usize;
        let pad_left = value(8) as usize;
        shape = shape.with_padding([pad_top, pad_left]);
    }
    let kernels = [kernel, kernel];
    let plan = ConvPlan::new(shape, kernels);
    eprintln!(
        "shape: padded_out_channels={} output_atom_bytes={} output_scratch_bytes={} tiles={}",
        shape.padded_out_channels(),
        shape.output_atom_bytes(),
        shape.output_scratch_bytes(kernels),
        plan.tiles().len(),
    );
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
