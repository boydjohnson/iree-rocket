//! Times the two host-side layout transforms every Rocket dispatch pays.
//!
//! `ROCKET_PROFILE` (rocket-hal-driver) measures them inside a real
//! inference, which is the number that matters but a slow way to iterate:
//! each change needs a cross-build, a copy to the board and a model run, and
//! the result is mixed with governor noise. This runs the same two functions
//! over the same shapes MobileNetV2 fp16 actually presents, on plain heap
//! memory, so a loop-order or blocking change can be judged in one command.
//!
//! Run it on the board -- the host's cache hierarchy and prefetchers are not
//! the A76's, and this is entirely a memory-system benchmark. Pass `cold` as
//! the second argument to evict the caches before every timed pass, which is
//! the state the driver actually runs in: the NPU scratch a compaction reads
//! was just invalidated by `PREP_BO`, and the dense buffer a packing reads
//! was written by a CPU dispatch long enough ago to have been evicted. A
//! warm-cache number here flatters the transform by several times and does
//! not predict anything in `ROCKET_PROFILE`.
//!
//! ```sh
//! CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!   cargo build -p iree-rocket-hal --release \
//!   --target aarch64-unknown-linux-gnu --example layout_bench
//! scp target/aarch64-unknown-linux-gnu/release/examples/layout_bench planck:/tmp/
//! ssh planck /tmp/layout_bench
//! ```

use std::time::{Duration, Instant};

use iree_rocket_hal::rocket::tensor_layout::{
    FEATURE_ATOMIC_BYTES, compact_atomic_output, nc1hwc2_storage_size, pack_nhwc_to_nc1hwc2_padded,
};

/// One dispatch's two transforms, as MobileNetV2 fp16 presents them.
///
/// `in_*` describe the packing (dense NHWC in, NC1HWC2 out) and `out_*` the
/// compaction (NC1HWC2 in, dense NHWC out). Taken from a `ROCKET_PROFILE=1`
/// per-op table of `mnv2.fp16.vmfb`, one entry per distinct dispatch shape
/// with the count that shape occurs.
struct Case {
    label: &'static str,
    count: usize,
    in_pixels: usize,
    in_channels: usize,
    out_pixels: usize,
    out_channels: usize,
}

const ELEMENT_BYTES: usize = 2;

const CASES: &[Case] = &[
    Case {
        label: "stem 225x225x3->112x112x48",
        count: 1,
        in_pixels: 225 * 225,
        in_channels: 3,
        out_pixels: 112 * 112,
        out_channels: 48,
    },
    Case {
        label: "112x112x48->24",
        count: 1,
        in_pixels: 112 * 112,
        in_channels: 48,
        out_pixels: 112 * 112,
        out_channels: 24,
    },
    Case {
        label: "112x112x24->144",
        count: 1,
        in_pixels: 112 * 112,
        in_channels: 24,
        out_pixels: 112 * 112,
        out_channels: 144,
    },
    Case {
        label: "56x56x144->32",
        count: 1,
        in_pixels: 56 * 56,
        in_channels: 144,
        out_pixels: 56 * 56,
        out_channels: 32,
    },
    Case {
        label: "56x56x32->192",
        count: 2,
        in_pixels: 56 * 56,
        in_channels: 32,
        out_pixels: 56 * 56,
        out_channels: 192,
    },
    Case {
        label: "56x56x192->32",
        count: 1,
        in_pixels: 56 * 56,
        in_channels: 192,
        out_pixels: 56 * 56,
        out_channels: 32,
    },
    Case {
        label: "28x28x192->48",
        count: 1,
        in_pixels: 28 * 28,
        in_channels: 192,
        out_pixels: 28 * 28,
        out_channels: 48,
    },
    Case {
        label: "28x28x48->288",
        count: 3,
        in_pixels: 28 * 28,
        in_channels: 48,
        out_pixels: 28 * 28,
        out_channels: 288,
    },
    Case {
        label: "28x28x288->48",
        count: 2,
        in_pixels: 28 * 28,
        in_channels: 288,
        out_pixels: 28 * 28,
        out_channels: 48,
    },
    Case {
        label: "14x14x288->88",
        count: 1,
        in_pixels: 14 * 14,
        in_channels: 288,
        out_pixels: 14 * 14,
        out_channels: 88,
    },
    Case {
        label: "14x14x88->528",
        count: 4,
        in_pixels: 14 * 14,
        in_channels: 88,
        out_pixels: 14 * 14,
        out_channels: 528,
    },
    Case {
        label: "14x14x528->88",
        count: 3,
        in_pixels: 14 * 14,
        in_channels: 528,
        out_pixels: 14 * 14,
        out_channels: 88,
    },
    Case {
        label: "14x14x528->136",
        count: 1,
        in_pixels: 14 * 14,
        in_channels: 528,
        out_pixels: 14 * 14,
        out_channels: 136,
    },
    Case {
        label: "14x14x136->816",
        count: 3,
        in_pixels: 14 * 14,
        in_channels: 136,
        out_pixels: 14 * 14,
        out_channels: 816,
    },
    Case {
        label: "14x14x816->136",
        count: 2,
        in_pixels: 14 * 14,
        in_channels: 816,
        out_pixels: 14 * 14,
        out_channels: 136,
    },
    Case {
        label: "7x7x816->224",
        count: 1,
        in_pixels: 7 * 7,
        in_channels: 816,
        out_pixels: 7 * 7,
        out_channels: 224,
    },
    Case {
        label: "7x7x224->1344",
        count: 3,
        in_pixels: 7 * 7,
        in_channels: 224,
        out_pixels: 7 * 7,
        out_channels: 1344,
    },
    Case {
        label: "7x7x1344->224",
        count: 2,
        in_pixels: 7 * 7,
        in_channels: 1344,
        out_pixels: 7 * 7,
        out_channels: 224,
    },
    Case {
        label: "7x7x1344->448",
        count: 1,
        in_pixels: 7 * 7,
        in_channels: 1344,
        out_pixels: 7 * 7,
        out_channels: 448,
    },
    Case {
        label: "7x7x448->1792",
        count: 1,
        in_pixels: 7 * 7,
        in_channels: 448,
        out_pixels: 7 * 7,
        out_channels: 1792,
    },
];

/// What `command_buffer::dispatch` computes for the packed pixel width: the
/// channel count rounded to a whole 16-byte atom, with a floor of one atom.
fn packed_bytes_per_pixel(channels: usize) -> usize {
    channels
        .max(FEATURE_ATOMIC_BYTES / ELEMENT_BYTES)
        .next_multiple_of(FEATURE_ATOMIC_BYTES / ELEMENT_BYTES)
        * ELEMENT_BYTES
}

fn main() {
    // Enough repetitions that the smallest case is still tens of
    // milliseconds; the governor needs a while to notice this is work.
    let repeats: usize = std::env::args()
        .nth(1)
        .and_then(|value| value.parse().ok())
        .unwrap_or(20);
    let cold = std::env::args().any(|arg| arg == "cold");
    // Comfortably past the 3 MiB L3, walked at one access per 64-byte line.
    let mut evictor = vec![0u8; 16 << 20];
    let mut evict = |cold: bool, evictor: &mut Vec<u8>| {
        if !cold {
            return;
        }
        let mut sum = 0u8;
        for index in (0..evictor.len()).step_by(64) {
            sum = sum.wrapping_add(evictor[index]);
            evictor[index] = sum;
        }
        std::hint::black_box(sum);
    };

    println!(
        "{:<28} {:>4} {:>9} {:>9} {:>9} {:>9}",
        "case", "n", "pack ms", "pack MB/s", "cmpt ms", "cmpt MB/s"
    );

    let mut pack_total = 0.0;
    let mut compact_total = 0.0;
    for case in CASES {
        let dense_bpp = case.in_channels * ELEMENT_BYTES;
        let packed_bpp = packed_bytes_per_pixel(case.in_channels);
        let packed_len = nc1hwc2_storage_size(case.in_pixels, packed_bpp).unwrap();
        let dense_in = vec![0x5au8; case.in_pixels * dense_bpp];
        let mut packed = vec![0u8; packed_len];

        let out_bpp = case.out_channels * ELEMENT_BYTES;
        let scratch_len = nc1hwc2_storage_size(case.out_pixels, out_bpp).unwrap();
        let scratch = vec![0xa5u8; scratch_len];
        let mut dense_out = vec![0u8; case.out_pixels * out_bpp];

        // One untimed pass so the first timed one is not paying for the
        // page faults of a freshly allocated vector.
        pack_nhwc_to_nc1hwc2_padded(
            &dense_in,
            case.in_pixels,
            dense_bpp,
            packed_bpp,
            &mut packed,
        )
        .unwrap();
        compact_atomic_output(
            &scratch,
            case.out_pixels,
            case.out_pixels,
            out_bpp,
            FEATURE_ATOMIC_BYTES,
            &mut dense_out,
        );

        let mut pack_elapsed = Duration::ZERO;
        for _ in 0..repeats {
            evict(cold, &mut evictor);
            let started = Instant::now();
            pack_nhwc_to_nc1hwc2_padded(
                &dense_in,
                case.in_pixels,
                dense_bpp,
                packed_bpp,
                &mut packed,
            )
            .unwrap();
            pack_elapsed += started.elapsed();
        }
        let pack_seconds = pack_elapsed.as_secs_f64() / repeats as f64;

        let mut compact_elapsed = Duration::ZERO;
        for _ in 0..repeats {
            evict(cold, &mut evictor);
            let started = Instant::now();
            compact_atomic_output(
                &scratch,
                case.out_pixels,
                case.out_pixels,
                out_bpp,
                FEATURE_ATOMIC_BYTES,
                &mut dense_out,
            );
            compact_elapsed += started.elapsed();
        }
        let compact_seconds = compact_elapsed.as_secs_f64() / repeats as f64;

        // Per inference: the shape's cost times how often the model runs it.
        pack_total += pack_seconds * case.count as f64;
        compact_total += compact_seconds * case.count as f64;
        println!(
            "{:<28} {:>4} {:>9.3} {:>9.0} {:>9.3} {:>9.0}",
            case.label,
            case.count,
            pack_seconds * 1e3,
            packed_len as f64 / pack_seconds / 1e6,
            compact_seconds * 1e3,
            dense_out.len() as f64 / compact_seconds / 1e6,
        );
    }
    println!(
        "\nper inference: pack {:.1} ms, compact {:.1} ms, total {:.1} ms",
        pack_total * 1e3,
        compact_total * 1e3,
        (pack_total + compact_total) * 1e3,
    );
}
