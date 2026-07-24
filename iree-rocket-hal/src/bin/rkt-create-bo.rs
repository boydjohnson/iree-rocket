use iree_rocket_hal::rocket::api::{
    DRM_COMMAND_BASE, DRM_IOCTL_BASE, DRM_ROCKET_CREATE_BO, drm_rocket_create_bo,
};
use nix::sys::mman::{MapFlags, ProtFlags, mmap};
use std::{
    fs::OpenOptions,
    num::NonZeroUsize,
    os::{
        fd::{FromRawFd, RawFd},
        unix::io::AsRawFd,
    },
    ptr,
};

// Base 'd' (0x64), Index 0x40 (DRM_COMMAND_BASE + DRM_ROCKET_CREATE_BO)
nix::ioctl_readwrite!(
    rocket_create_bo,
    DRM_IOCTL_BASE,
    DRM_COMMAND_BASE + DRM_ROCKET_CREATE_BO,
    drm_rocket_create_bo
);

fn main() {
    println!("--- Level 2: Allocation & Mapping ---");

    // Open with Read/Write permissions
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/accel/accel0")
        .expect("Failed to open device");
    let fd = file.as_raw_fd();

    // --- Step A: Allocate 4KB (1 Page) ---
    let mut create_params = drm_rocket_create_bo {
        size: 4096,
        handle: 0,
        dma_address: 0,
        offset: 0,
    };

    unsafe {
        rocket_create_bo(fd, &mut create_params).expect("Failed to create BO");
    }

    println!("Success! BO Created.");
    println!("  Handle:      {}", create_params.handle);
    println!(
        "  DMA Address: 0x{:x} (The NPU sees this address)",
        create_params.dma_address
    );
    println!(
        "  Mmap Offset: 0x{:x} (Use this to map to CPU)",
        create_params.offset
    );

    // --- Step B: Map to CPU (mmap) ---
    // We map the file descriptor at the specific offset the driver gave us.
    println!("Attempting mmap...");
    let map_len = NonZeroUsize::new(4096).unwrap();
    let map_addr = unsafe {
        mmap(
            None,                                         // Let OS choose address
            map_len,                                      // Length
            ProtFlags::PROT_READ | ProtFlags::PROT_WRITE, // Read/Write
            MapFlags::MAP_SHARED,                         // Share with device
            file,                                         // The device FD
            create_params.offset as i64,                  // The offset from CREATE_BO
        )
        .expect("mmap failed")
    };

    println!("Mapped at CPU address: {:p}", map_addr);

    // --- Step C: Write Test ---
    // Let's write a "poison" value to the first 32-bit word
    let ptr = map_addr.as_ptr() as *mut _ as *mut u32;
    unsafe {
        ptr::write_volatile(ptr, 0xCAFEBABE);
        let read_back = ptr::read_volatile(ptr);
        println!("Wrote 0xCAFEBABE, Read back: 0x{:X}", read_back);

        if read_back == 0xCAFEBABE {
            println!("Memory checks out! CPU and NPU are linked.");
        } else {
            eprintln!("Data corruption detected!");
        }
    }

    // Cleanup (Optional, OS does it on exit)
    // In a real driver, you'd also call DRM_IOCTL_GEM_CLOSE on the handle
}
