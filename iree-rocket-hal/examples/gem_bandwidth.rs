//! Raw memory bandwidth of a Rocket GEM buffer's CPU mapping, against
//! ordinary heap memory of the same size.
//!
//! Every host-side transform this driver runs has a GEM BO on at least one
//! side -- the NC1HWC2 scratch, and IREE's own buffers, are all
//! `DRM_ROCKET_CREATE_BO` + `mmap`. `layout_bench` measures those transforms
//! on heap memory and lands 3x faster than the same transforms measured
//! inside a real inference by `ROCKET_PROFILE`, cold caches included. That
//! gap is either the mapping's cacheability (a write-combining mapping reads
//! at DRAM latency with no prefetch, which is exactly the shape of the
//! discrepancy) or its page size. This measures it directly rather than
//! reading the kernel's `mmap` handler.
//!
//! Board only:
//!
//! ```sh
//! CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!   cargo build -p iree-rocket-hal --release \
//!   --target aarch64-unknown-linux-gnu --example gem_bandwidth
//! ```

use std::{os::fd::AsRawFd, time::Instant};

use iree_rocket_hal::rocket::device::OwnedBuffer;

const BYTES: usize = 4 << 20;
const REPEATS: usize = 20;

fn report(label: &str, seconds: f64, bytes: usize) {
    println!(
        "{:<28} {:>8.3} ms {:>9.0} MB/s",
        label,
        seconds * 1e3,
        bytes as f64 / seconds / 1e6
    );
}

/// Sums every 8th byte, which is enough to touch every 64-byte line without
/// the loop itself becoming the bottleneck.
fn read_pass(buffer: &[u8]) -> u64 {
    let mut sum = 0u64;
    for chunk in buffer.chunks_exact(8) {
        sum = sum.wrapping_add(u64::from_le_bytes(chunk.try_into().unwrap()));
    }
    sum
}

fn time<F: FnMut()>(mut body: F) -> f64 {
    body();
    let started = Instant::now();
    for _ in 0..REPEATS {
        body();
    }
    started.elapsed().as_secs_f64() / REPEATS as f64
}

fn main() {
    let file = match std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/accel/accel0")
    {
        Ok(file) => file,
        Err(err) => {
            eprintln!("open /dev/accel/accel0: {err} (run this on the board)");
            return;
        }
    };
    let fd = file.as_raw_fd();
    let bo = unsafe { OwnedBuffer::new(fd, BYTES, &file) };
    let gem = unsafe { std::slice::from_raw_parts_mut(bo.host_ptr, BYTES) };
    let mut heap = vec![0u8; BYTES];
    let mut heap_b = vec![0u8; BYTES];
    gem.fill(0x5a);
    heap.fill(0x5a);

    report(
        "heap read",
        time(|| {
            std::hint::black_box(read_pass(&heap));
        }),
        BYTES,
    );
    report(
        "gem  read",
        time(|| {
            std::hint::black_box(read_pass(gem));
        }),
        BYTES,
    );
    report("heap write", time(|| heap.fill(0x11)), BYTES);
    report("gem  write", time(|| gem.fill(0x11)), BYTES);
    report(
        "heap -> heap copy",
        time(|| heap_b.copy_from_slice(&heap)),
        BYTES,
    );
    report(
        "gem  -> heap copy",
        time(|| heap_b.copy_from_slice(gem)),
        BYTES,
    );
    report(
        "heap -> gem  copy",
        time(|| gem.copy_from_slice(&heap)),
        BYTES,
    );
}
