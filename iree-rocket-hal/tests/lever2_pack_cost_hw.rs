//! ROADMAP lever 2's gate, measured: what it costs to put a probability
//! matrix down the **weight** path at runtime.
//!
//! Lever 2 folds the softmax's epilogue into the two attention matmuls, and
//! every one of its four claims needs `P^T` to reach the hardware as a
//! coefficient tensor (SOFTMAX.md §1.1: the second matmul consumes the
//! probabilities as its weights, which is what keeps the queries on
//! channels and avoids a transpose). ROADMAP says of exactly this: "This is
//! probably the real gate on a fused SDPA, and it is **unmeasured**."
//!
//! So measure it. Coefficient packing is host work -- `WeightPlan::pack`,
//! the same code the driver runs at dispatch and the compiler runs at
//! `rocket-pack-weights` -- so this file opens no device. It is still an
//! `_hw` test because the number only means something at the board's own
//! A76 clock and memory system; run it the usual way
//! (`conv_phase1_validation_hw.rs`'s doc comment has the recipe):
//!
//!   CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!     cargo test --target aarch64-unknown-linux-gnu --release \
//!       --test lever2_pack_cost_hw --no-run
//!
//! The figure to beat is SOFTMAX.md §10.3's. At 1792 rows the whole softmax
//! is 1.80 ms over 19 device tasks for 9.1 ViT-B/16 heads, so a head costs
//! about 0.198 ms, and the sum tree lever 2 would delete is 11 of those 19
//! tasks -- about 0.11 ms a head. **Lever 2 has to pack `P^T` in less than
//! that**, per head, per layer, or it loses to the tree it replaces.

use std::time::Instant;

use rocket_core::{conv::Precision, fc, weights::WeightPlan};

/// ViT-B/16: 197 tokens, 12 heads of width 64, 12 layers.
const KEYS: u32 = 197;
const HEAD_WIDTH: u32 = 64;
const HEADS: u32 = 12;
const LAYERS: u32 = 12;

/// §10.3's cost for one head of softmax at the measured 1792-row batch, and
/// the part of it lever 2 claims back.
const SOFTMAX_MS_PER_HEAD: f64 = 1.80 / 9.1;
const SUM_TREE_MS_PER_HEAD: f64 = 1.80 * (11.0 / 19.0) / 9.1;

/// `S^T = K Q^T`: K on width, head width on `Cin`, queries on `Cout`.
/// The weight operand is `Q^T`, `[d, M]`.
fn qk_shape() -> fc::Shape {
    fc::Shape::new(KEYS, HEAD_WIDTH, KEYS, Precision::Fp16)
}

/// `O^T = V^T P^T`: `V^T` on width (plus the one all-ones column that makes
/// the denominator free), keys on `Cin`, queries on `Cout`. The weight
/// operand is `P^T`, `[N, M]` -- the probabilities themselves.
fn pv_shape() -> fc::Shape {
    fc::Shape::new(HEAD_WIDTH + 1, KEYS, KEYS, Precision::Fp16)
}

fn time_pack(label: &str, shape: fc::Shape) -> f64 {
    let plan = WeightPlan::for_matmul(shape).expect("matmul weight plan");
    let dense_bytes = plan.dense_bytes().expect("dense bytes");
    let packed_bytes = plan.packed_bytes().expect("packed bytes");

    // Content is irrelevant to the permutation's cost; a varying pattern
    // only keeps the optimiser from proving anything about it.
    let dense: Vec<u8> = (0..dense_bytes).map(|i| (i % 251) as u8).collect();
    let mut packed = vec![0u8; packed_bytes];

    for _ in 0..3 {
        plan.pack(&dense, &mut packed).expect("warm-up pack");
    }

    let iterations = 50;
    let mut samples = Vec::new();
    for _ in 0..5 {
        let start = Instant::now();
        for _ in 0..iterations {
            plan.pack(&dense, &mut packed).expect("pack");
        }
        samples.push(start.elapsed().as_secs_f64() * 1e3 / f64::from(iterations));
    }
    samples.sort_by(f64::total_cmp);
    let median = samples[samples.len() / 2];

    println!(
        "{label}: dense {dense_bytes} B -> packed {packed_bytes} B, \
         {:.4} ms (min {:.4}, max {:.4}), {:.0} MB/s",
        median,
        samples[0],
        samples[samples.len() - 1],
        f64::from(dense_bytes as u32) / 1e6 / (median / 1e3),
    );
    median
}

/// The gate. Prints the per-head and per-inference packing bill against the
/// softmax cost lever 2 would claim back, and asserts only what is
/// structural: that the plans exist and the packers accept these shapes.
#[test]
fn lever_2_weight_packing_cost() {
    println!("--- ROADMAP lever 2's gate: runtime coefficient packing ---");
    let qk = time_pack("Q^T  [d=64, M=197] fp16 ", qk_shape());
    let pv = time_pack("P^T  [N=197, M=197] fp16", pv_shape());

    let per_head = qk + pv;
    println!();
    println!(
        "per head: {:.4} ms packing vs {:.4} ms of softmax, of which the \
         sum tree lever 2 deletes is {:.4} ms",
        per_head, SOFTMAX_MS_PER_HEAD, SUM_TREE_MS_PER_HEAD
    );
    println!(
        "  P^T alone is {:.2}x the sum tree it would replace",
        pv / SUM_TREE_MS_PER_HEAD
    );
    println!(
        "  whole-head packing is {:.2}x the whole softmax",
        per_head / SOFTMAX_MS_PER_HEAD
    );
    println!(
        "per inference ({HEADS} heads x {LAYERS} layers): {:.1} ms of packing",
        per_head * f64::from(HEADS * LAYERS)
    );

    assert!(qk > 0.0 && pv > 0.0, "both packs must be measurable");
}

/// Lever 2's other half of the same gate, and the one no amount of packing
/// speed fixes: `P^T` is produced **on the device** by the `exp` task and
/// consumed **on the device** by the PV matmul, but the packer is host
/// code. So the probabilities have to come back across the bus, be
/// permuted, and go out again -- against a current op that never moves them
/// at all.
///
/// This measures the round trip's own floor: one read pass and one write
/// pass over the same bytes, with no permutation and no cache maintenance.
/// Anything lever 2 actually does costs at least this.
#[test]
fn lever_2_round_trip_floor() {
    let plan = WeightPlan::for_matmul(pv_shape()).expect("matmul weight plan");
    let dense_bytes = plan.dense_bytes().expect("dense bytes");
    let source: Vec<u8> = (0..dense_bytes).map(|i| (i % 251) as u8).collect();
    let mut destination = vec![0u8; dense_bytes];

    for _ in 0..3 {
        destination.copy_from_slice(&source);
    }

    let iterations = 200;
    let start = Instant::now();
    for _ in 0..iterations {
        destination.copy_from_slice(&source);
    }
    let copy = start.elapsed().as_secs_f64() * 1e3 / f64::from(iterations);

    println!(
        "P^T straight memcpy, {dense_bytes} B: {:.4} ms ({:.0} MB/s) -- \
         the floor a device-to-host-to-device trip cannot beat",
        copy,
        f64::from(dense_bytes as u32) / 1e6 / (copy / 1e3),
    );
    println!(
        "  against {:.4} ms of sum tree a head",
        SUM_TREE_MS_PER_HEAD
    );
    assert_eq!(destination, source);
}

/// How much of the 205 MB/s is the permutation and how much is the loop?
///
/// `pack_hwcf_to_rocket_weights` drives `WeightLayout::visit_slots`, a
/// six-deep nest that calls a closure once per coefficient and does two
/// multiplies and a bounds-checked two-byte `copy_from_slice` inside it.
/// The *order* it produces is the hardware's and is not in question; the
/// question is whether that order can be produced faster, because the answer
/// decides whether attention's matmuls can be offloaded at all.
///
/// This is the same order written the obvious other way: one contiguous run
/// of destination coefficients per `(output block, input group, output
/// lane)`, gathered from the dense filter with a constant stride, as an
/// iterator pair rather than an indexed closure. Specialised to the 1x1
/// fp16 shape a matmul takes -- which is all lever 2 needs -- and checked
/// byte-for-byte against the real packer before it is timed.
///
/// **This generalised and landed on 2026-11-24** (`rocket_core::weights`'
/// strided arm, COMPILER_ROADMAP.md 6.3), so the comparison below is no
/// longer prototype-against-production: it is now an independent
/// restatement of the layout agreeing byte-for-byte with the packer that
/// ships, which is worth keeping for exactly that reason.
fn pack_matmul_fp16_striped(k: usize, n: usize, dense: &[u8], packed: &mut [u8]) {
    // The fp16 coefficient layout, restated: 16 output kernels per 32-byte
    // weight atom, 32 input channels per coefficient group, input channels
    // padded to 8-channel atoms with the 3-mod-4 atom bump.
    const OUTPUT_BLOCK: usize = 16;
    const INPUT_GROUP: usize = 32;
    const CHANNELS_PER_ATOM: usize = 8;

    let input_atoms = k.div_ceil(CHANNELS_PER_ATOM);
    let padded_atoms = if input_atoms % 4 == 3 {
        input_atoms + 1
    } else {
        input_atoms
    };
    let padded_k = padded_atoms * CHANNELS_PER_ATOM;
    let input_groups = padded_k.div_ceil(INPUT_GROUP);
    let output_blocks = n.div_ceil(OUTPUT_BLOCK);

    packed.fill(0);
    let mut slot = 0usize;
    for output_block in 0..output_blocks {
        for input_group in 0..input_groups {
            let group_base = input_group * INPUT_GROUP;
            let lanes = INPUT_GROUP.min(padded_k - group_base);
            // Lanes the filter reaches; the rest stay the zero fill above.
            let live = lanes.min(k.saturating_sub(group_base));
            for output_lane in 0..OUTPUT_BLOCK {
                let output_channel = output_block * OUTPUT_BLOCK + output_lane;
                if output_channel >= n {
                    continue;
                }
                let destination = &mut packed[slot * 2..(slot + live) * 2];
                let source = &dense[(group_base * n + output_channel) * 2..];
                for (out, value) in destination
                    .as_chunks_mut::<2>()
                    .0
                    .iter_mut()
                    .zip(source.as_chunks::<2>().0.iter().step_by(n))
                {
                    out.copy_from_slice(value);
                }
                slot += lanes;
            }
        }
    }
}

/// The restatement above must be byte-identical to the packer the hardware
/// is validated against, at every shape lever 2 and `fc.rs` reach -- the
/// partial final input group, the partial final output block, and the
/// 3-mod-4 atom bump included. Runs anywhere; no device.
#[test]
fn the_striped_packer_is_byte_identical() {
    // 197 keys gives 25 atoms (1 mod 4); 184 gives 23 (3 mod 4, the bump);
    // 197 queries gives a partial final output block.
    for (k, n) in [
        (64, 197),
        (197, 197),
        (197, 64),
        (184, 48),
        (184, 197),
        (32, 16),
        (8, 1),
        (200, 2364),
    ] {
        let shape = fc::Shape::new(1, k, n, Precision::Fp16);
        let plan = WeightPlan::for_matmul(shape).expect("matmul weight plan");
        let dense_bytes = plan.dense_bytes().expect("dense bytes");
        let packed_bytes = plan.packed_bytes().expect("packed bytes");
        let dense: Vec<u8> = (0..dense_bytes).map(|i| (i % 251) as u8).collect();

        let mut reference = vec![0u8; packed_bytes];
        plan.pack(&dense, &mut reference).expect("reference pack");

        let mut striped = vec![0u8; packed_bytes];
        pack_matmul_fp16_striped(k as usize, n as usize, &dense, &mut striped);

        assert_eq!(
            striped, reference,
            "striped packer disagrees at k={k}, n={n}"
        );
    }
}

/// And the headroom, in milliseconds, against the same two shapes the gate
/// above measured.
#[test]
fn the_striped_packer_s_headroom() {
    println!("--- the same coefficient order, written as strided runs ---");
    for (label, shape) in [
        ("Q^T  [d=64, M=197] fp16 ", qk_shape()),
        ("P^T  [N=197, M=197] fp16", pv_shape()),
    ] {
        let plan = WeightPlan::for_matmul(shape).expect("matmul weight plan");
        let dense_bytes = plan.dense_bytes().expect("dense bytes");
        let packed_bytes = plan.packed_bytes().expect("packed bytes");
        let dense: Vec<u8> = (0..dense_bytes).map(|i| (i % 251) as u8).collect();
        let mut packed = vec![0u8; packed_bytes];
        let k = plan.shape.in_channels as usize;
        let n = plan.shape.out_channels as usize;

        for _ in 0..3 {
            pack_matmul_fp16_striped(k, n, &dense, &mut packed);
        }
        let iterations = 200;
        let mut samples = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            for _ in 0..iterations {
                pack_matmul_fp16_striped(k, n, &dense, &mut packed);
            }
            samples.push(start.elapsed().as_secs_f64() * 1e3 / f64::from(iterations));
        }
        samples.sort_by(f64::total_cmp);
        let median = samples[samples.len() / 2];
        println!(
            "{label}: {:.4} ms, {:.0} MB/s",
            median,
            f64::from(dense_bytes as u32) / 1e6 / (median / 1e3),
        );
        if shape == pv_shape() {
            println!(
                "  P^T would be {:.2}x the {:.4} ms sum tree it replaces",
                median / SUM_TREE_MS_PER_HEAD,
                SUM_TREE_MS_PER_HEAD
            );
        }
    }
}
