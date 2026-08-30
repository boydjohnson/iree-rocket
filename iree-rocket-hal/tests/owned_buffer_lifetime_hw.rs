//! Planck-only ownership check for the RAII GEM buffer wrapper.
//!
//! Cross-compile and run the ignored test on an RK3588 board. Dropping an
//! `OwnedBuffer` must unmap its VMA and close its DRM GEM handle; trying to
//! close the same handle again is the observable kernel-side check.
//!
//! ```text
//! CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!   cargo test --target aarch64-unknown-linux-gnu \
//!   --test owned_buffer_lifetime_hw --no-run
//! ./owned_buffer_lifetime_hw-<hash> --ignored --nocapture
//! ```

use std::{fs::OpenOptions, os::fd::AsRawFd};

use iree_rocket_hal::rocket::device::{OwnedBuffer, close_bo};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const PAGE_BYTES: usize = 4096;

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn owned_buffer_drop_closes_every_gem_handle() {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let fd = file.as_raw_fd();

    for iteration in 0..256 {
        let handle = {
            let buffer = unsafe { OwnedBuffer::new(fd, PAGE_BYTES, &file) };
            buffer.handle
        };
        assert!(
            unsafe { close_bo(fd, handle) }.is_err(),
            "iteration {iteration}: GEM handle {handle} remained live after OwnedBuffer::drop"
        );
    }
}
