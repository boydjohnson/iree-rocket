// Standalone debugger for RegCmd formation.
// Run with: cargo run --bin chk-regcmd
//
// Used to reimplement its own local `RegCmd` and build the target/value
// pair from raw bindgen macro calls instead of the typed builders, and
// asserted the domain should be the bare `target_PPU` (0x4000) with no
// extra bit -- which is what every RegisterMeta impl assumed too, before
// decoding a real regcmd program out of conv.rknn showed every domain
// that could be checked (including PPU) is actually tagged as
// `target | 1` (see builders.rs and rknpu-spelunking/NOTES.md). Rewritten
// to go through the same `Register<T>`/`RegCmd` the rest of the crate
// uses, and to check against the corrected expected value.

use iree_rocket_hal::rocket::builders::{Bits, DOMAIN_PPU, Register, ppu::PpuSPointer};

fn main() {
    println!("--- RegCmd Verification Tool ---");

    let cmd = Register::<PpuSPointer>::new()
        .executer(Bits::new(0))
        .executer_pp_en(Bits::new(1))
        .pointer_pp_mode(Bits::new(1))
        .build();

    // Manual check on the value alone (independent of domain/offset):
    // EXECUTER_PP_EN (bit 2) -> 0x4
    // POINTER_PP_MODE (bit 3) -> 0x8
    // Result should be 0xC (1100 binary)
    let p_val = (cmd.0 >> 16) & 0xFFFFFFFF;
    let expected_val = 0xC;
    println!("\n1. Value Calculation Check:");
    println!("   Value (Binary): {:032b}", p_val);
    println!("   Value (Hex):    0x{:08X}", p_val);
    if p_val == expected_val {
        println!("   OK: Calculated 0xC correctly.");
    } else {
        println!(
            "   MISMATCH! Expected 0x{:X}, Got 0x{:X}",
            expected_val, p_val
        );
    }

    println!("\n2. RegCmd Packing Check:");
    println!("   Raw 64-bit Command: 0x{:016X}", cmd.0);

    let p_domain = (cmd.0 >> 48) & 0xFFFF;
    let p_offset = cmd.0 & 0xFFFF;

    // DOMAIN_PPU already includes the confirmed `| 1` -- see builders.rs.
    println!(
        "   [Domain] Expected: 0x{:04X} (target_PPU | 1) | Got: 0x{:04X}",
        DOMAIN_PPU, p_domain
    );
    println!(
        "   [Value ] Expected: 0x{:08X} | Got: 0x{:08X}",
        expected_val, p_val
    );
    // The offset field is the literal absolute REG_PPU_S_POINTER address
    // (0x6004), not something relative to a PPU-block base -- confirmed
    // by the conv.rknn decode, where every domain's offset matched
    // rkt_registers.h's absolute REG_* value directly. The original
    // version of this tool expected 0x004 here, which was wrong for the
    // same reason the domain was wrong: nobody had ground truth yet.
    const EXPECTED_OFFSET: u64 = 0x6004;
    println!(
        "   [Offset] Expected: 0x{:04X} | Got: 0x{:04X}",
        EXPECTED_OFFSET, p_offset
    );

    if p_domain == DOMAIN_PPU as u64 && p_val == expected_val && p_offset == EXPECTED_OFFSET {
        println!("   OK: Packing matches.");
    } else {
        println!("   MISMATCH in packing.");
    }
}
