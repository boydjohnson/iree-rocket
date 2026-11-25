//! What the dense coefficient packer costs, both arms, at the board's own
//! clock.
//!
//! `rocket_core::weights::pack_hwcf_to_rocket_weights` is host work on the
//! hot path twice -- the driver packs per dispatch
//! (`rocket-hal-driver/src/command_buffer.rs`, with `weight_cache.rs`
//! caching across inferences) and the compiler packs constants at
//! `rocket-pack-weights` -- and SOFTMAX.md section 11.3 measured it at 214
//! MB/s, 139x off `memcpy` of the same bytes. It now routes 1- and 2-byte
//! symmetric filters to a strided arm;
//! `pack_hwcf_to_rocket_weights_reference` is the old per-coefficient arm,
//! kept and exposed so both can be timed inside one binary.
//!
//! One binary is the point. Within a binary these samples agree to about
//! 0.3%, but the same code rebuilt has moved 190-220 MB/s on the reference
//! arm -- codegen layout, not the hardware -- so an absolute rate is worth
//! a range and only a *ratio measured in one process* is worth a decimal.
//!
//! No device is opened; this is pure host code. It is still an `_hw` test
//! because the number only means anything on the board's A76. Build and run
//! it the usual way (`conv_phase1_validation_hw.rs`'s doc comment has the
//! recipe):
//!
//! ```text
//! CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!   cargo test --target aarch64-unknown-linux-gnu --release \
//!     --test weight_pack_rate_hw --no-run
//!
//! taskset -c 4-7 ./weight_pack_rate_hw-<hash> --ignored --nocapture
//! ```

use std::time::Instant;

use rocket_core::weights::{
    pack_hwcf_to_rocket_weights, pack_hwcf_to_rocket_weights_reference,
    rocket_weight_storage_size,
};

/// One real coefficient tensor: `kh x kw x Cin x Cout` at `element_size`
/// bytes, and where in the stack it comes from.
struct Case {
    label: &'static str,
    kernel: usize,
    input_channels: usize,
    output_channels: usize,
    element_size: usize,
}

/// Shapes that actually dominate the three places this packer runs: the
/// pointwise convolutions MobileNetV2 is mostly made of, two ResNet50
/// blocks, the two attention operands ISSUES.md C15 pays for per head per
/// layer, and one int8 convolution.
const CASES: &[Case] = &[
    Case {
        label: "mnv2   1x1 fp16  960 ->  320",
        kernel: 1,
        input_channels: 960,
        output_channels: 320,
        element_size: 2,
    },
    Case {
        label: "mnv2   1x1 fp16  576 ->   96",
        kernel: 1,
        input_channels: 576,
        output_channels: 96,
        element_size: 2,
    },
    Case {
        label: "mnv2   3x3 fp16    3 ->   32",
        kernel: 3,
        input_channels: 3,
        output_channels: 32,
        element_size: 2,
    },
    Case {
        label: "rn50   1x1 fp16 2048 ->  512",
        kernel: 1,
        input_channels: 2048,
        output_channels: 512,
        element_size: 2,
    },
    Case {
        label: "rn50   3x3 fp16  512 ->  512",
        kernel: 3,
        input_channels: 512,
        output_channels: 512,
        element_size: 2,
    },
    Case {
        label: "rn50   7x7 fp16    3 ->   64",
        kernel: 7,
        input_channels: 3,
        output_channels: 64,
        element_size: 2,
    },
    Case {
        label: "ViT  Q^T 1x1 fp16   64 ->  197",
        kernel: 1,
        input_channels: 64,
        output_channels: 197,
        element_size: 2,
    },
    Case {
        label: "ViT  P^T 1x1 fp16  197 ->  197",
        kernel: 1,
        input_channels: 197,
        output_channels: 197,
        element_size: 2,
    },
    Case {
        label: "int8   3x3        256 ->  256",
        kernel: 3,
        input_channels: 256,
        output_channels: 256,
        element_size: 1,
    },
    Case {
        label: "int8   1x1        512 -> 1024",
        kernel: 1,
        input_channels: 512,
        output_channels: 1024,
        element_size: 1,
    },
];

/// Median of five timed batches, in milliseconds per call, with the batch
/// size chosen so a batch is long enough to time.
fn median_ms(dense_bytes: usize, mut call: impl FnMut()) -> (f64, f64, f64) {
    let iterations = (4_000_000 / dense_bytes.max(1)).clamp(10, 2000);
    for _ in 0..3 {
        call();
    }
    let mut samples = Vec::new();
    for _ in 0..5 {
        let start = Instant::now();
        for _ in 0..iterations {
            call();
        }
        samples.push(start.elapsed().as_secs_f64() * 1e3 / iterations as f64);
    }
    samples.sort_by(f64::total_cmp);
    (
        samples[samples.len() / 2],
        samples[0],
        samples[samples.len() - 1],
    )
}

fn rate_mb_s(dense_bytes: usize, ms: f64) -> f64 {
    dense_bytes as f64 / 1e6 / (ms / 1e3)
}

/// Both arms, every case, plus `memcpy` of the same bytes as the scale.
///
/// Asserts the one thing that is not a measurement: that the two arms agree
/// byte for byte at every shape it times. The rates are printed, not
/// asserted -- a rate assertion on a shared board is a flake, and the
/// equivalence sweep in `rocket-core/src/weights.rs` is where correctness is
/// actually gated.
#[test]
#[ignore = "board measurement: run on the RK3588 with --ignored --nocapture"]
fn the_dense_packer_s_rate_on_both_arms() {
    println!("--- dense coefficient packing, per call, median of 5 ---");
    println!(
        "{:<30} {:>10} {:>12} {:>12} {:>8}",
        "case", "dense KiB", "reference", "strided", "speedup"
    );

    let mut total_reference = 0.0;
    let mut total_strided = 0.0;
    for case in CASES {
        let dense_bytes = case.kernel
            * case.kernel
            * case.input_channels
            * case.output_channels
            * case.element_size;
        let packed_bytes = rocket_weight_storage_size(
            case.kernel,
            case.kernel,
            case.input_channels,
            case.output_channels,
            case.element_size,
        )
        .expect("packed size");
        let dense: Vec<u8> = (0..dense_bytes).map(|i| (i % 251 + 1) as u8).collect();

        let mut reference = vec![0u8; packed_bytes];
        pack_hwcf_to_rocket_weights_reference(
            &dense,
            case.kernel,
            case.kernel,
            case.input_channels,
            case.output_channels,
            case.element_size,
            &mut reference,
        )
        .expect("reference pack");
        let mut strided = vec![0u8; packed_bytes];
        pack_hwcf_to_rocket_weights(
            &dense,
            case.kernel,
            case.kernel,
            case.input_channels,
            case.output_channels,
            case.element_size,
            &mut strided,
        )
        .expect("strided pack");
        assert_eq!(
            strided, reference,
            "the two arms disagree at {}",
            case.label
        );

        let mut scratch = vec![0u8; packed_bytes];
        let (reference_ms, _, _) = median_ms(dense_bytes, || {
            pack_hwcf_to_rocket_weights_reference(
                &dense,
                case.kernel,
                case.kernel,
                case.input_channels,
                case.output_channels,
                case.element_size,
                &mut scratch,
            )
            .expect("reference pack");
        });
        assert_eq!(scratch, reference);
        let (strided_ms, strided_low, strided_high) = median_ms(dense_bytes, || {
            pack_hwcf_to_rocket_weights(
                &dense,
                case.kernel,
                case.kernel,
                case.input_channels,
                case.output_channels,
                case.element_size,
                &mut scratch,
            )
            .expect("strided pack");
        });
        assert_eq!(scratch, strided);

        total_reference += reference_ms;
        total_strided += strided_ms;
        println!(
            "{:<30} {:>10.1} {:>7.4} ms {:>7.4} ms {:>7.2}x   \
             ({:.0} -> {:.0} MB/s, spread {:.1}%)",
            case.label,
            dense_bytes as f64 / 1024.0,
            reference_ms,
            strided_ms,
            reference_ms / strided_ms,
            rate_mb_s(dense_bytes, reference_ms),
            rate_mb_s(dense_bytes, strided_ms),
            100.0 * (strided_high - strided_low) / strided_ms,
        );
    }

    println!();
    println!(
        "sum over {} cases: {:.3} ms -> {:.3} ms, {:.2}x",
        CASES.len(),
        total_reference,
        total_strided,
        total_reference / total_strided
    );

    // The scale: one pass over the same bytes with no permutation at all.
    let bytes = 1 << 20;
    let source = vec![0xA5u8; bytes];
    let mut destination = vec![0u8; bytes];
    let (copy_ms, _, _) = median_ms(bytes, || {
        // Without the barrier the whole copy is dead code and the "rate"
        // comes back as the timer's own resolution.
        destination.copy_from_slice(&source);
        std::hint::black_box(&mut destination);
    });
    println!(
        "memcpy of 1 MiB: {:.4} ms ({:.0} MB/s)",
        copy_ms,
        rate_mb_s(bytes, copy_ms)
    );
    assert_eq!(destination[0], 0xA5);
}
