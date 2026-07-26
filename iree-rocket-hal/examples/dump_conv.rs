//! Prints a convolution program as hex regcmd words, for diffing against a
//! vendor capture outside the test harness.
//!
//!   cargo run -p iree-rocket-hal --example dump_conv -- \
//!       <width> <height> <stride> <cin> <cout> <kernel> [i8 <zp> <ozp> <scale> <shift>]

use iree_rocket_hal::rocket::conv::{
    Multiplier, Precision, Quantization, Shape, Tile, conv_2d_tile,
};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let value = |index: usize| args[index].parse::<u32>().expect("numeric argument");
    let signed = |index: usize| args[index].parse::<i32>().expect("numeric argument");

    let kernel = value(5) as usize;
    let precision = if args.len() > 6 && args[6] == "i8" {
        Precision::Int8(Quantization {
            input_zero_point: signed(7),
            output_zero_point: signed(8),
            multiplier: Multiplier {
                scale: value(9),
                shift: value(10),
            },
        })
    } else {
        Precision::Fp16
    };

    let shape = Shape::with_precision(value(0), value(1), value(2), value(3), value(4), precision);
    let kernels = [kernel, kernel];
    for command in conv_2d_tile(shape, kernels, &Tile::whole(shape, kernels)) {
        println!("{:016x}", command.0);
    }
}
