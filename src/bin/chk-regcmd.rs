// Standalone debugger for RegCmd formation
// Run with: cargo run --bin check_regcmd

// ============================================================================
// 1. Mocking the Header Constants (from rkt_registers.h)
// ============================================================================

// ============================================================================
// 2. The Builder Logic (What we want to verify)
// ============================================================================

use iree_rocket_hal::rocket::registers::{
    PPU_S_POINTER_EXECUTER, PPU_S_POINTER_EXECUTER_PP_EN, PPU_S_POINTER_POINTER_PP_MODE,
    REG_PPU_S_POINTER, target_PPU,
};

struct RegCmd(u64);

impl RegCmd {
    fn new(domain: u32, reg: u32, val: u32) -> Self {
        // 2. Pack: [Domain: 16b] [Value: 32b] [Offset: 16b]
        let packed = (domain as u64) << 48 | ((val as u64) << 16) | (reg as u64);

        RegCmd(packed)
    }
}

// ============================================================================
// 3. Verification Main
// ============================================================================

fn main() {
    println!("--- RegCmd Verification Tool ---");

    let val_to_write = unsafe {
        PPU_S_POINTER_EXECUTER(0)
            | PPU_S_POINTER_EXECUTER_PP_EN(1)
            | PPU_S_POINTER_POINTER_PP_MODE(1)
    };

    // Build the command
    let cmd = RegCmd::new(target_PPU, REG_PPU_S_POINTER, val_to_write);

    // DECODE / VERIFY
    // --------------------------------------------------------
    println!("\n1. Value Calculation Check:");
    println!("   Target Value (Binary): {:032b}", val_to_write);
    println!("   Target Value (Hex):    0x{:08X}", val_to_write);

    // Manual Check:
    // PP_EN (Shift 2, Val 1) -> 100 (binary) -> 0x4
    // PP_MODE (Shift 3, Val 1) -> 1000 (binary) -> 0x8
    // Result should be 0xC (1100 binary)
    let expected_val = 0xC;
    if val_to_write == expected_val {
        println!("   ✅ Value Match: Calculated 0xC correctly.");
    } else {
        println!(
            "   ❌ Value Mismatch! Expected 0x{:X}, Got 0x{:X}",
            expected_val, val_to_write
        );
    }

    println!("\n2. RegCmd Packing Check:");
    println!("   Raw 64-bit Command: 0x{:016X}", cmd.0);

    // Unpack bits
    let p_domain = (cmd.0 >> 48) & 0xFFFF;
    let p_val = (cmd.0 >> 16) & 0xFFFFFFFF;
    let p_offset = cmd.0 & 0xFFFF;

    println!(
        "   [Domain] Expected: 0x{:04X} | Got: 0x{:04X}",
        target_PPU, p_domain
    );
    println!(
        "   [Value ] Expected: 0x{:08X} | Got: 0x{:08X}",
        val_to_write, p_val
    );
    println!(
        "   [Offset] Expected: 0x{:04X} | Got: 0x{:04X}",
        0x004, p_offset
    ); // 0x6004 & 0xFFF = 0x004

    if p_domain == target_PPU as u64 && p_val == (val_to_write as u64) && p_offset == 0x004 {
        println!("   ✅ Packing Successful.");
    } else {
        println!("   ❌ Packing Failed.");
    }
}
