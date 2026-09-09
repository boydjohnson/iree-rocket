//! Where decomposition actually becomes necessary.
//!
//! With `ROCKET_ALLOW_UNBACKED_CHANNELS` lifting the capture-backing
//! ceilings, sweeps each axis until the planner refuses for a reason
//! decomposition cannot argue with -- a hardware field or a CBUF capacity
//! -- and prints the first such extent. Anything refused only as
//! `UnvalidatedConfiguration` below that is a measurement away, not a
//! rewrite away.
use rocket_core::{
    conv::{self, ConvPlan, Precision},
    error::PlanErrorCode,
    fc,
    policy::{PlanningPolicy, with_policy},
};

fn conv_at(
    w: u32,
    h: u32,
    cin: u32,
    cout: u32,
    k: usize,
) -> Result<usize, (PlanErrorCode, String)> {
    let shape = conv::Shape::try_with_precision(w, h, 1, cin, cout, Precision::Fp16)
        .map_err(|e| (e.code(), e.to_string()))?;
    ConvPlan::try_new(shape, [k, k])
        .map(|p| p.tiles().len())
        .map_err(|e| (e.code(), e.to_string()))
}

fn matmul_at(m: u32, k: u32, n: u32) -> Result<usize, (PlanErrorCode, String)> {
    let shape =
        fc::Shape::try_new(m, k, n, Precision::Fp16).map_err(|e| (e.code(), e.to_string()))?;
    fc::Plan::try_new(shape)
        .map(|p| p.conv_plan().tiles().len())
        .map_err(|e| (e.code(), e.to_string()))
}

fn sweep(label: &str, values: &[u32], f: impl Fn(u32) -> Result<usize, (PlanErrorCode, String)>) {
    println!("\n== {label}");
    for &v in values {
        match f(v) {
            Ok(tiles) => println!("  {v:>6}: ok, {tiles} tile(s)"),
            Err((code, message)) => {
                println!("  {v:>6}: {code:?} -- {message}");
                if code != PlanErrorCode::UnvalidatedConfiguration {
                    println!("  ^ first refusal decomposition cannot argue with");
                    return;
                }
            }
        }
    }
}

fn main() {
    let policy = PlanningPolicy {
        allow_unbacked_channels: true,
        allow_large_kernel_probing: true,
    };
    with_policy(policy, || {
        sweep(
            "conv Cout at 14x14 Cin 256 k1x1",
            &[3584, 4096, 8192, 16384, 32768, 65536, 131072],
            |cout| conv_at(14, 14, 256, cout, 1),
        );
        sweep(
            "conv Cin at 14x14 Cout 256 k1x1",
            &[3584, 4096, 8192, 16384, 32768, 65536],
            |cin| conv_at(14, 14, cin, 256, 1),
        );
        sweep(
            "matmul M at K 768 N 768",
            &[2047, 2048, 4096, 8192, 16384, 32768, 65536],
            |m| matmul_at(m, 768, 768),
        );
        sweep(
            "matmul N at M 197 K 768",
            &[3584, 4096, 8192, 16384, 32768, 65536],
            |n| matmul_at(197, 768, n),
        );
        sweep(
            "matmul K at M 197 N 768",
            &[3584, 4096, 8192, 16384, 32768],
            |k| matmul_at(197, k, 768),
        );
    });
}
