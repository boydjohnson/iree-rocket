//! Prints a convolution program as hex regcmd words, for diffing against a
//! vendor capture outside the test harness.
//!
//!   cargo run -p iree-rocket-hal --example dump_conv -- \
//!       <width> <height> <stride> <cin> <cout> <kernel> \
//!       [i8 <zp> <ozp> <scale> <shift>] \
//!       [--tiles <count>] [--vendor-grains] [--decoded]

use iree_rocket_hal::rocket::conv::{
    Multiplier, Precision, Quantization, Shape, Tile, conv_2d_tile, conv_2d_tile_with_grains,
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
    let flag_value = |flag: &str| {
        args.iter().position(|arg| arg == flag).map(|index| {
            args[index + 1]
                .parse::<u32>()
                .expect("numeric flag argument")
        })
    };
    let tile_count = flag_value("--tiles").unwrap_or(1);
    let vendor_grains = args.iter().any(|arg| arg == "--vendor-grains");
    let decoded = args.iter().any(|arg| arg == "--decoded");

    for (tile_index, tile) in Tile::split(shape, kernels, tile_count).iter().enumerate() {
        let mut commands = if vendor_grains {
            conv_2d_tile_with_grains(shape, kernels, tile, tile.in_rows)
        } else {
            conv_2d_tile(shape, kernels, tile)
        };
        if vendor_grains && tile_count > 1 {
            for command in &mut commands {
                let domain = (command.0 >> 48) & 0xffff;
                let offset = command.0 & 0xffff;
                if domain == 0x0201 && offset == 0x1010 {
                    command.0 =
                        (command.0 & !(0xf000_0000u64 << 16)) | (u64::from(tile_count - 1) << 44);
                }
            }
            commands.drain(..4);
        }
        for command in commands {
            if decoded {
                let domain = (command.0 >> 48) & 0xffff;
                let value = (command.0 >> 16) & 0xffff_ffff;
                let offset = command.0 & 0xffff;
                println!("{tile_index}\t{domain:#06x}\t{offset:#06x}\t{value:#010x}");
            } else {
                println!("{:016x}", command.0);
            }
        }
    }
}
