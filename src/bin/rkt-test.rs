use std::{ffi::c_char, fs::File, mem::MaybeUninit, os::fd::AsRawFd};

use iree_rocket_hal::rocket::api::drm_version;

nix::ioctl_readwrite!(drm_get_version, b'd', 0x00, drm_version);

fn main() {
    println!("Opening /dev/accel/accel0...");
    let file = File::open("/dev/accel/accel0").expect("Failed to open device");
    let fd = file.as_raw_fd();

    // 3. First Pass: Get the Version Numbers and String Lengths
    // We initialize the pointers to null/0 so the kernel knows we just want sizes.
    let mut version = drm_version {
        version_major: 0,
        version_minor: 0,
        version_patchlevel: 0,
        name_len: 0,
        date_len: 0,
        desc_len: 0,
        name: std::ptr::null_mut(),
        date: std::ptr::null_mut(),
        desc: std::ptr::null_mut(),
    };

    println!("Sending DRM_IOCTL_VERSION (Pass 1)...");
    unsafe {
        if let Err(e) = drm_get_version(fd, &mut version) {
            eprintln!("IOCTL Failed: {}", e);
            return;
        }
    }

    println!("--- Driver Info ---");
    println!("Major: {}", version.version_major);
    println!("Minor: {}", version.version_minor);
    println!("Patch: {}", version.version_patchlevel);
    println!("Name Len Needed: {}", version.name_len);

    // 4. Second Pass: Get the Actual Strings
    // Now that we know the lengths, we allocate buffers and call it again.

    // Allocate a buffer for the name (add 1 for null terminator safety)
    let mut name_vec: Vec<u8> = vec![0; version.name_len as usize + 1];
    let mut date_vec: Vec<u8> = vec![0; version.date_len as usize + 1];
    let mut desc_vec: Vec<u8> = vec![0; version.desc_len as usize + 1];

    version.name = name_vec.as_mut_ptr() as *mut c_char;
    version.date = date_vec.as_mut_ptr() as *mut c_char;
    version.desc = desc_vec.as_mut_ptr() as *mut c_char;

    println!("Sending DRM_IOCTL_VERSION (Pass 2)...");
    unsafe {
        drm_get_version(fd, &mut version).expect("Pass 2 failed");
    }

    // Convert raw C strings back to Rust strings for printing
    let name = unsafe { std::ffi::CStr::from_ptr(version.name).to_string_lossy() };
    let date = unsafe { std::ffi::CStr::from_ptr(version.date).to_string_lossy() };
    let desc = unsafe { std::ffi::CStr::from_ptr(version.desc).to_string_lossy() };

    println!("Name: {}", name);
    println!("Date: {}", date);
    println!("Desc: {}", desc);
}
