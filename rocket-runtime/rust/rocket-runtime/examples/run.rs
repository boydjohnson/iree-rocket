//! Runs one function of a rocket-compiler module, the Rust counterpart of
//! `iree-run-module`:
//!
//! ```text
//! run [--cpu-only] [--cpus=4,5,6,7] [--iterations=N] [--output-dir=DIR]
//!     MODULE.vmfb module.FUNCTION SHAPExTYPE=@input.bin [...]
//! ```
//!
//! Each input is `1x3x224x224xf16=@input.bin`: a shape, an element type
//! (f16 bf16 f32 f64 i8 i16 i32 i64) and a file of its raw row-major bytes --
//! the spelling `iree-run-module --input=` takes. Results are printed as
//! shape and type, and with `--output-dir` written as `result_N.bin`, the raw
//! bytes `iree-run-module --output=@file.bin` writes, so the two can be
//! compared with `cmp`.

use std::{path::PathBuf, process::ExitCode, time::Instant};

use rocket_runtime::{
    BufferView, CpuPlacement, Devices, ElementType, Instance, Session, SessionOptions,
};

fn parse_input(session: &Session, spec: &str) -> Result<BufferView, String> {
    let (layout, value) = spec
        .split_once('=')
        .ok_or_else(|| format!("input {spec:?}: expected SHAPExTYPE=@file"))?;
    let path = value
        .strip_prefix('@')
        .ok_or_else(|| format!("input {spec:?}: only @file values are supported"))?;
    let mut parts: Vec<&str> = layout.split('x').collect();
    let type_name = parts.pop().unwrap_or_default();
    let element_type = match type_name {
        "f16" => ElementType::Float16,
        "bf16" => ElementType::BFloat16,
        "f32" => ElementType::Float32,
        "f64" => ElementType::Float64,
        "i8" => ElementType::Int8,
        "i16" => ElementType::Int16,
        "i32" => ElementType::Int32,
        "i64" => ElementType::Int64,
        other => return Err(format!("input {spec:?}: unknown element type {other:?}")),
    };
    let shape = parts
        .iter()
        .map(|d| {
            d.parse::<usize>()
                .map_err(|_| format!("input {spec:?}: bad dimension {d:?}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let bytes = std::fs::read(path).map_err(|e| format!("reading {path}: {e}"))?;
    BufferView::from_bytes(session, &shape, element_type, &bytes).map_err(|e| e.to_string())
}

fn run() -> Result<(), String> {
    let mut options = SessionOptions::new();
    let mut iterations = 1usize;
    let mut output_dir: Option<PathBuf> = None;
    let mut positional = Vec::new();
    for arg in std::env::args().skip(1) {
        if arg == "--cpu-only" {
            options.devices = Devices::CpuOnly;
        } else if let Some(list) = arg.strip_prefix("--cpus=") {
            let cpus = list
                .split(',')
                .map(|c| c.parse().map_err(|_| format!("bad CPU {c:?}")))
                .collect::<Result<_, _>>()?;
            options.cpu_placement = CpuPlacement::Cpus(cpus);
        } else if let Some(n) = arg.strip_prefix("--iterations=") {
            iterations = n.parse().map_err(|_| format!("bad --iterations {n:?}"))?;
        } else if let Some(dir) = arg.strip_prefix("--output-dir=") {
            output_dir = Some(dir.into());
        } else if arg.starts_with("--") {
            return Err(format!("unknown flag {arg}"));
        } else {
            positional.push(arg);
        }
    }
    let [module, function, inputs @ ..] = positional.as_slice() else {
        return Err(
            "usage: run [flags] MODULE.vmfb module.FUNCTION SHAPExTYPE=@input.bin ...".into(),
        );
    };

    let instance = Instance::new().map_err(|e| e.to_string())?;
    let mut session = Session::new(&instance, &options).map_err(|e| e.to_string())?;
    session
        .load_module_file(module)
        .map_err(|e| e.to_string())?;
    let inputs = inputs
        .iter()
        .map(|spec| parse_input(&session, spec))
        .collect::<Result<Vec<_>, _>>()?;
    let input_refs: Vec<&BufferView> = inputs.iter().collect();

    let mut function = session.function(function).map_err(|e| e.to_string())?;
    let mut outputs = Vec::new();
    let mut times = Vec::with_capacity(iterations);
    for _ in 0..iterations.max(1) {
        let start = Instant::now();
        outputs = function.invoke(&input_refs).map_err(|e| e.to_string())?;
        times.push(start.elapsed().as_secs_f64() * 1e3);
    }

    eprintln!("session: {} device(s)", session.device_count());
    if times.len() > 1 {
        let rest = &times[1..];
        eprintln!(
            "first call {:.3} ms, then {:.3} ms mean over {}",
            times[0],
            rest.iter().sum::<f64>() / rest.len() as f64,
            rest.len()
        );
    } else {
        eprintln!("call {:.3} ms", times[0]);
    }
    for (i, output) in outputs.iter().enumerate() {
        println!(
            "result[{i}]: {:?} {:?}",
            output.shape(),
            output.element_type()
        );
        if let Some(dir) = &output_dir {
            let path = dir.join(format!("result_{i}.bin"));
            let bytes = output.to_bytes().map_err(|e| e.to_string())?;
            std::fs::write(&path, bytes).map_err(|e| format!("writing {}: {e}", path.display()))?;
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}
