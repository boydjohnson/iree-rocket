//! Cartesian Conv2D correctness sweep against an independent logical oracle.
//!
//! Every case executes through `ConvPlan::new`, production HWCF weight
//! packing, and the RK3588 device. Failures are accumulated: one bad case
//! never prevents later cases from running, and the test asserts only after
//! printing the complete summary. The one exception is a *sick device* --
//! see `run_hardware_case_matrix`, which halts there because past that point
//! a sweep measures the NPU's health rather than the shapes.
//!
//! Cross-compile and run on the board:
//!
//! ```text
//! CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
//!   cargo test --target aarch64-unknown-linux-gnu --release \
//!     --test conv2d_oracle_hw --no-run
//!
//! ./conv2d_oracle_hw-<hash> --ignored --nocapture
//! ```

#[path = "support/conv2d_oracle.rs"]
mod conv2d_oracle;

use std::{
    any::Any,
    collections::BTreeMap,
    fs::OpenOptions,
    mem,
    os::unix::io::AsRawFd,
    panic::{AssertUnwindSafe, catch_unwind},
    ptr,
};

use conv2d_oracle::{
    Conv2dCase, Conv2dFixture, OraclePattern, OraclePrecision, build_fixture, expected_output,
    f16_to_f32, output_offset, output_storage_bytes,
};
use iree_rocket_hal::rocket::{
    conv::{AccumulatorOutputTile, Buffers, ConvPlan, Shape},
    device::{Buffer, JobDesc, close_bo, fini_bo, prep_bo, submit_jobs, unmap_bo},
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const PAGE_BYTES: usize = 4096;
const PER_CASE_TIMEOUT_NS: u64 = 5_000_000_000;
const OUTPUT_SENTINEL: u8 = 0xa5;

fn page_aligned_size(size: usize) -> usize {
    size.max(1).div_ceil(PAGE_BYTES) * PAGE_BYTES
}

struct OwnedBuffer {
    fd: i32,
    buffer: Buffer,
}

impl OwnedBuffer {
    unsafe fn new(fd: i32, size: usize, file: &std::fs::File) -> Self {
        Self {
            fd,
            buffer: unsafe { Buffer::new(fd, page_aligned_size(size), file) },
        }
    }

    unsafe fn from_bytes(fd: i32, bytes: &[u8], file: &std::fs::File) -> Self {
        let buffer = unsafe { Self::new(fd, bytes.len(), file) };
        unsafe {
            ptr::write_bytes(buffer.buffer.host_ptr, 0, buffer.buffer.size);
            ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.buffer.host_ptr, bytes.len());
        }
        buffer
    }
}

impl Drop for OwnedBuffer {
    fn drop(&mut self) {
        unsafe {
            let _ = unmap_bo(&self.buffer);
            let _ = close_bo(self.fd, self.buffer.handle);
        }
    }
}

struct MismatchReport {
    mismatches: usize,
    max_abs_difference: f32,
    samples: Vec<String>,
    tile_mismatches: Vec<usize>,
}

struct CaseSuccess {
    data_banks: u32,
    weight_banks: u32,
    tiles: usize,
}

struct CaseExecution {
    plan: ConvPlan,
    output: Vec<u8>,
}

fn assemble_staged_accumulator_output(
    shape: Shape,
    kernels: [usize; 2],
    scratch: &[u8],
    tiles: &[AccumulatorOutputTile],
) -> Result<Vec<u8>, String> {
    let output_width = shape.output_width(kernels) as usize;
    let output_height = shape.output_height(kernels) as usize;
    let output_pixels = output_width * output_height;
    let block_bytes = shape.output_atom_bytes() as usize;
    let bytes_per_pixel =
        shape.padded_out_channels() as usize * shape.precision.output_element_bytes() as usize;
    let blocks_per_pixel = bytes_per_pixel.div_ceil(block_bytes);
    let mut output = vec![OUTPUT_SENTINEL; output_storage_bytes(shape, kernels)];

    for (index, tile) in tiles.iter().enumerate() {
        let tile_pixels = tile.output_rows * tile.output_columns;
        let expected_tile_bytes = tile_pixels * blocks_per_pixel * block_bytes;
        if tile.scratch_bytes != expected_tile_bytes {
            return Err(format!(
                "tile {index} declares {} scratch bytes, expected {expected_tile_bytes}",
                tile.scratch_bytes
            ));
        }
        let tile_end = tile
            .scratch_offset
            .checked_add(tile.scratch_bytes)
            .ok_or_else(|| format!("tile {index} scratch range overflow"))?;
        if tile_end > scratch.len() {
            return Err(format!(
                "tile {index} scratch range {}..{tile_end} exceeds {} bytes",
                tile.scratch_offset,
                scratch.len()
            ));
        }

        for surface in 0..blocks_per_pixel {
            for row in 0..tile.output_rows {
                for column in 0..tile.output_columns {
                    let local_pixel = row * tile.output_columns + column;
                    let output_row = tile.output_row + row;
                    let output_column = tile.output_column + column;
                    if output_row >= output_height || output_column >= output_width {
                        return Err(format!(
                            "tile {index} output ({output_row}, {output_column}) exceeds \
                             {output_height}x{output_width}"
                        ));
                    }
                    let source = tile.scratch_offset
                        + surface * tile_pixels * block_bytes
                        + local_pixel * block_bytes;
                    let destination = surface * output_pixels * block_bytes
                        + (output_row * output_width + output_column) * block_bytes;
                    output[destination..destination + block_bytes]
                        .copy_from_slice(&scratch[source..source + block_bytes]);
                }
            }
        }
    }

    Ok(output)
}

fn tile_for_output(plan: &ConvPlan, y: usize, x: usize) -> Option<usize> {
    plan.tiles().iter().position(|tile| {
        (tile.rows.out_first as usize..(tile.rows.out_first + tile.rows.out_rows) as usize)
            .contains(&y)
            && (tile.columns.out_first as usize
                ..(tile.columns.out_first + tile.columns.out_cols) as usize)
                .contains(&x)
    })
}

fn compare_output(fixture: &Conv2dFixture, plan: &ConvPlan, output: &[u8]) -> MismatchReport {
    let case = fixture.case;
    let shape = fixture.shape;
    let out_height = shape.output_height(case.kernel) as usize;
    let out_width = shape.output_width(case.kernel) as usize;
    let tolerance = if case.precision == OraclePrecision::Int8 {
        1.0
    } else {
        0.0
    };
    let mut report = MismatchReport {
        mismatches: 0,
        max_abs_difference: 0.0,
        samples: Vec::new(),
        tile_mismatches: vec![0; plan.tiles().len()],
    };

    for y in 0..out_height {
        for x in 0..out_width {
            for channel in 0..case.cout as usize {
                let offset = output_offset(shape, case.kernel, channel, y, x);
                let got = match case.precision {
                    OraclePrecision::Fp16 => {
                        f16_to_f32(u16::from_le_bytes([output[offset], output[offset + 1]]))
                    }
                    OraclePrecision::Int8 => f32::from(output[offset] as i8),
                    OraclePrecision::Int8Accumulator => {
                        i32::from_le_bytes(output[offset..offset + 4].try_into().unwrap()) as f32
                    }
                };
                let want = expected_output(case, channel, y, x) as f32;
                let difference = (got - want).abs();
                report.max_abs_difference = report.max_abs_difference.max(difference);
                if !got.is_finite() || difference > tolerance {
                    report.mismatches += 1;
                    if let Some(tile) = tile_for_output(plan, y, x) {
                        report.tile_mismatches[tile] += 1;
                    }
                    if report.samples.len() < 12 {
                        report
                            .samples
                            .push(format!("[y={y}, x={x}, c={channel}] want {want} got {got}"));
                    }
                }
            }
        }
    }
    report
}

fn execute_case_output(
    file: &std::fs::File,
    fixture: &Conv2dFixture,
) -> Result<CaseExecution, String> {
    let plan = ConvPlan::new(fixture.shape, fixture.case.kernel);
    execute_case_output_with_plan(file, fixture, plan)
}

fn execute_case_output_with_plan(
    file: &std::fs::File,
    fixture: &Conv2dFixture,
    plan: ConvPlan,
) -> Result<CaseExecution, String> {
    let fd = file.as_raw_fd();
    let shape = fixture.shape;
    let kernels = fixture.case.kernel;
    let output_len = output_storage_bytes(shape, kernels);

    unsafe {
        let input = OwnedBuffer::from_bytes(fd, &fixture.input, file);
        let weights = OwnedBuffer::from_bytes(fd, &fixture.weights, file);
        let bias = OwnedBuffer::from_bytes(fd, &fixture.bias, file);
        let output = OwnedBuffer::new(fd, output_len, file);
        // A zero fill can turn an unwritten output lane into a plausible
        // convolution result. Poison the whole allocation so missing tail
        // pixels/blocks fail loudly and remain distinguishable from a real
        // all-zero accumulator.
        ptr::write_bytes(output.buffer.host_ptr, OUTPUT_SENTINEL, output.buffer.size);

        let buffers = Buffers {
            input: input.buffer.dma_address,
            weights: weights.buffer.dma_address,
            bias: bias.buffer.dma_address,
            output: output.buffer.dma_address,
        };
        let (programs, accumulator_tiles) = if shape.precision.writes_accumulators() {
            let staged = plan.programs_with_staged_accumulator_output(buffers);
            (staged.programs, Some(staged.tiles))
        } else {
            (plan.programs_with_buffers(buffers), None)
        };
        let mut command_buffers = Vec::with_capacity(programs.len());
        for program in &programs {
            let command_bytes = program.len() * mem::size_of::<u64>();
            let buffer = OwnedBuffer::new(fd, command_bytes, file);
            ptr::write_bytes(buffer.buffer.host_ptr, 0, buffer.buffer.size);
            let words =
                std::slice::from_raw_parts_mut(buffer.buffer.host_ptr as *mut u64, program.len());
            for (destination, command) in words.iter_mut().zip(program) {
                *destination = command.0;
            }
            command_buffers.push((buffer, program.len() as u32));
        }

        for handle in [
            input.buffer.handle,
            weights.buffer.handle,
            bias.buffer.handle,
            output.buffer.handle,
        ] {
            fini_bo(fd, handle).map_err(|error| format!("sync data BO: {error}"))?;
        }
        for (buffer, _) in &command_buffers {
            fini_bo(fd, buffer.buffer.handle)
                .map_err(|error| format!("sync regcmd BO: {error}"))?;
        }

        let tasks = command_buffers
            .iter()
            .map(|(buffer, count)| [(buffer.buffer.dma_address, *count)])
            .collect::<Vec<_>>();
        let input_handles = command_buffers
            .iter()
            .map(|(buffer, _)| {
                [
                    buffer.buffer.handle,
                    input.buffer.handle,
                    weights.buffer.handle,
                    bias.buffer.handle,
                ]
            })
            .collect::<Vec<_>>();
        let output_handles = [output.buffer.handle];
        let jobs = tasks
            .iter()
            .zip(&input_handles)
            .map(|(tasks, input_handles)| JobDesc {
                tasks,
                in_handles: input_handles,
                out_handles: &output_handles,
            })
            .collect::<Vec<_>>();

        submit_jobs(fd, &jobs).map_err(|error| format!("submit: {error}"))?;
        prep_bo(fd, output.buffer.handle, PER_CASE_TIMEOUT_NS)
            .map_err(|error| format!("completion wait: {error}"))?;

        let raw_output = std::slice::from_raw_parts(output.buffer.host_ptr, output_len).to_vec();
        let output = match accumulator_tiles {
            Some(tiles) => assemble_staged_accumulator_output(shape, kernels, &raw_output, &tiles)?,
            None => raw_output,
        };
        Ok(CaseExecution { plan, output })
    }
}

fn execute_case(file: &std::fs::File, fixture: &Conv2dFixture) -> Result<CaseSuccess, String> {
    let execution = execute_case_output(file, fixture)?;
    let report = compare_output(fixture, &execution.plan, &execution.output);
    if report.mismatches != 0 {
        return Err(format!(
            "{} mismatches, max|diff|={}, tile_mismatches={:?}\n      {}",
            report.mismatches,
            report.max_abs_difference,
            report.tile_mismatches,
            report.samples.join("\n      "),
        ));
    }
    Ok(CaseSuccess {
        data_banks: execution.plan.data_banks(),
        weight_banks: execution.plan.weight_banks(),
        tiles: execution.plan.tiles().len(),
    })
}

fn cartesian_cases() -> Vec<Conv2dCase> {
    let mut cases = Vec::new();
    let input_channels = [3u32, 4, 5, 64, 128, 192, 256, 320, 384, 448, 512];
    let output_channels = [64u32, 128, 256, 512];

    // The broad Cartesian core: exactly the extent used by the expanded
    // vendor corpus, with both kernel sizes and both precisions.
    for precision in [OraclePrecision::Fp16, OraclePrecision::Int8] {
        for cin in input_channels {
            for cout in output_channels {
                for kernel in [1usize, 3] {
                    cases.push(Conv2dCase {
                        width: 28,
                        height: 28,
                        cin,
                        cout,
                        kernel: [kernel, kernel],
                        stride: 1,
                        padding: [kernel / 2, kernel / 2],
                        precision,
                        pattern: OraclePattern::Counting,
                    });
                }
            }
        }
    }

    // A smaller Cartesian selector grid catches indexing/permutation bugs
    // without doubling every expensive high-channel case.
    for precision in [OraclePrecision::Fp16, OraclePrecision::Int8] {
        for cin in [3u32, 5, 128, 256, 512] {
            for cout in [64u32, 256, 512] {
                cases.push(Conv2dCase {
                    width: 28,
                    height: 28,
                    cin,
                    cout,
                    kernel: [3, 3],
                    stride: 1,
                    padding: [1, 1],
                    precision,
                    pattern: if precision == OraclePrecision::Int8 {
                        OraclePattern::SelectorsAffine { phase: 0 }
                    } else {
                        OraclePattern::Selectors { phase: 0 }
                    },
                });
            }
        }
    }

    // IREE materializes VGG's padding as real zero-border pixels. These are
    // the nine unique physical shapes behind all sixteen convolutions.
    let vgg = [
        (226, 3, 64),
        (226, 64, 64),
        (114, 64, 128),
        (114, 128, 128),
        (58, 128, 256),
        (58, 256, 256),
        (30, 256, 512),
        (30, 512, 512),
        (16, 512, 512),
    ];
    for precision in [OraclePrecision::Fp16, OraclePrecision::Int8] {
        for (extent, cin, cout) in vgg {
            for selector_case in [false, true] {
                let pattern = if selector_case {
                    if precision == OraclePrecision::Int8 {
                        OraclePattern::SelectorsAffine { phase: 1 }
                    } else {
                        OraclePattern::Selectors { phase: 1 }
                    }
                } else {
                    OraclePattern::Counting
                };
                cases.push(Conv2dCase {
                    width: extent,
                    height: extent,
                    cin,
                    cout,
                    kernel: [3, 3],
                    stride: 1,
                    padding: [0, 0],
                    precision,
                    pattern,
                });
            }
        }
    }
    cases
}

/// Focused boundaries around the two output-channel-group cutoffs found by
/// the Cartesian sweep, rather than the sweep's coarse 64/128/256/512 steps.
///
/// K3/P1 at 28x28 with small Cin fails starting at output channel 224:
/// channels 0-223 are correct and 224 onward reads zero at both Cout 256 and
/// 512. Cout 224 itself is untested by the Cartesian grid -- it may be the
/// largest count one CNA task can cover, with 225 the first count that needs
/// a second, currently-unprogrammed output-channel-group task. Cout 256 is
/// repeated here (already covered by the Cartesian grid) so 224/225/256
/// print together as one boundary rather than requiring a second test run
/// to line up against.
///
/// Separately, fp16 K1/P0 at 28x28 with Cin 384 or 448 fails starting at
/// output channel 32: channels 0-31 are correct and 32 onward reads zero at
/// Cout 512. The same question applies at a different cutoff: is 32 the
/// largest single-task count, with 33 the first that needs a second task?
/// No int8 K1 case failed, so this side stays fp16-only.
fn output_channel_group_boundary_cases() -> Vec<Conv2dCase> {
    let mut cases = Vec::new();
    for precision in [OraclePrecision::Fp16, OraclePrecision::Int8] {
        for cin in [3u32, 4, 5] {
            for cout in [224u32, 225, 256] {
                cases.push(Conv2dCase {
                    width: 28,
                    height: 28,
                    cin,
                    cout,
                    kernel: [3, 3],
                    stride: 1,
                    padding: [1, 1],
                    precision,
                    pattern: OraclePattern::Counting,
                });
            }
        }
    }
    for cin in [384u32, 448] {
        for cout in [32u32, 33, 512] {
            cases.push(Conv2dCase {
                width: 28,
                height: 28,
                cin,
                cout,
                kernel: [1, 1],
                stride: 1,
                padding: [0, 0],
                precision: OraclePrecision::Fp16,
                pattern: OraclePattern::Counting,
            });
        }
    }
    cases
}

/// Confirms the mechanism behind `output_channel_group_boundary_cases`,
/// rather than just its location.
///
/// That probe found every one of 224/225/32/33 exact -- the cutoff is not a
/// fixed per-task channel count, and 256/512 failing is not evidence of
/// missing output-channel-group task splitting after all. Host-side
/// `ConvPlan::data_banks`/`weight_banks` explains it instead:
/// `demand_based_cbuf_partition`'s starved-to-`streamed_preference` branch
/// (`conv.rs` around line 1183) returns as soon as coefficient demand
/// exceeds one bank, without ever comparing against the independently
/// hardware-validated `weight_banks_floor` minimum a few lines below it --
/// that comparison is unreachable once the earlier branch has already
/// returned. For Cin 3/4/5 K3 fp16, weight_channels is 8: `floor(8) = 3`
/// banks, but `streamed_preference` is only 1, so the plan starves itself to
/// 1 weight bank the moment coefficient demand needs more than one. Walking
/// `ConvPlan::data_banks`/`weight_banks` across Cout by hand finds the exact
/// bank-count flip (1/11 -> 11/1, etc.) at:
///
/// - fp16 K3, Cin 3/4/5: Cout 227 (still 1/11) -> 228 (11/1)
/// - int8 K3, Cin 3/4/5: Cout 226 (still 1/11) -> 227 (11/1)
/// - fp16 K1, Cin 384: Cout 426 (still 2/10) -> 427 (11/1)
/// - fp16 K1, Cin 448: Cout 365 (still 2/10) -> 366 (11/1)
///
/// If this is the real mechanism, each pair here should flip from correct to
/// all-zero-past-some-channel at exactly its listed boundary, with no gap
/// like the 225-vs-256 one that motivated this probe.
fn bank_partition_flip_boundary_cases() -> Vec<Conv2dCase> {
    let mut cases = Vec::new();
    for (precision, boundary) in [
        (OraclePrecision::Fp16, 227u32),
        (OraclePrecision::Int8, 226),
    ] {
        for cin in [3u32, 4, 5] {
            for cout in [boundary, boundary + 1] {
                cases.push(Conv2dCase {
                    width: 28,
                    height: 28,
                    cin,
                    cout,
                    kernel: [3, 3],
                    stride: 1,
                    padding: [1, 1],
                    precision,
                    pattern: OraclePattern::Counting,
                });
            }
        }
    }
    for (cin, boundary) in [(384u32, 426u32), (448, 365)] {
        for cout in [boundary, boundary + 1] {
            cases.push(Conv2dCase {
                width: 28,
                height: 28,
                cin,
                cout,
                kernel: [1, 1],
                stride: 1,
                padding: [0, 0],
                precision: OraclePrecision::Fp16,
                pattern: OraclePattern::Counting,
            });
        }
    }
    cases
}

/// A real fp16 VGG-19 run through blocks 4-5 (all four Cin/Cout-512
/// convolutions) previously showed severe end-to-end disagreement against
/// CPU -- MAE roughly 30x worse than the same run capped at Cin/Cout <= 256,
/// with only 20.70% of logits within the 0.05 tolerance that the <= 256 run hit
/// 100% on. None of these shapes are affected by the CBUF bank-partition
/// bug fixed above (VGG never asks for small Cin with large Cout, or the
/// specific high-Cin K1 combinations that bug needed), and the Cartesian
/// oracle's `Counting`/`Selectors` patterns already pass at exactly these
/// shapes. The leading hypothesis is that something specific to *dense,
/// fully-diverse* coefficient data -- every tap and channel nonzero and
/// distinct, as real trained weights are -- breaks in a way neither of
/// those patterns exercises (`Counting` is uniform 1s; `Selectors` is
/// three nonzero taps per output with everything else zero).
///
/// This runs `Dense` at the two suspect Cin/Cout-512 shapes (VGG blocks 4
/// and 5) alongside the same pattern at a known-safe Cin/Cout-256 shape
/// (VGG block 3) as a control: if block 3 passes and blocks 4/5 fail, that
/// isolates the effect to Cin/Cout 512 specifically rather than to
/// something wrong with the `Dense` pattern or harness itself.
///
/// Status (2026-08-30, `f6843e4`): all five cases pass on RK3588 hardware
/// after the CBUF partition fixes. Keep this as a regression gate because
/// the earlier Counting/Selectors coverage passed while a real trained-weight
/// model did not. `tools/e2e_conv_regression.py` cross-builds and runs this exact
/// test before checking the compiler -> VMFB -> public-driver path too.
fn dense_coefficient_vgg_block_cases() -> Vec<Conv2dCase> {
    let mut cases = Vec::new();
    for (extent, cin, cout) in [
        // VGG block 3 control -- known-safe, Counting/Selectors already
        // pass here, and it is well outside the CBUF bank-partition bug's
        // range.
        (58u32, 128u32, 256u32),
        (58, 256, 256),
        // VGG block 4 -- the two suspect shapes.
        (30, 256, 512),
        (30, 512, 512),
        // VGG block 5 -- the other suspect shape.
        (16, 512, 512),
    ] {
        cases.push(Conv2dCase {
            width: extent,
            height: extent,
            cin,
            cout,
            kernel: [3, 3],
            stride: 1,
            padding: [0, 0],
            precision: OraclePrecision::Fp16,
            pattern: OraclePattern::Dense { phase: 0 },
        });
    }
    cases
}

/// Hardware-confirmed exact i32-accumulator regression cases through the
/// shared production oracle. Together these cover a partial second output
/// block, a partial third block at stride 2, the K3 Cin=32 ceiling, the K1
/// transition from one tile at Cin=352 to two at Cin=353, the tile-local
/// contract's current Cin=384 ceiling, a large three-tile image, and Cout=512
/// ceilings with dense signed data.
fn int8_accumulator_regression_cases() -> Vec<Conv2dCase> {
    vec![
        Conv2dCase {
            width: 9,
            height: 7,
            cin: 5,
            cout: 63,
            kernel: [1, 1],
            stride: 1,
            padding: [0, 0],
            precision: OraclePrecision::Int8Accumulator,
            pattern: OraclePattern::Dense { phase: 0 },
        },
        Conv2dCase {
            width: 34,
            height: 34,
            cin: 32,
            cout: 64,
            kernel: [3, 3],
            stride: 1,
            padding: [0, 0],
            precision: OraclePrecision::Int8Accumulator,
            pattern: OraclePattern::Dense { phase: 2 },
        },
        Conv2dCase {
            width: 32,
            height: 32,
            cin: 352,
            cout: 64,
            kernel: [1, 1],
            stride: 1,
            padding: [0, 0],
            precision: OraclePrecision::Int8Accumulator,
            pattern: OraclePattern::Dense { phase: 1 },
        },
        Conv2dCase {
            width: 32,
            height: 32,
            cin: 353,
            cout: 64,
            kernel: [1, 1],
            stride: 1,
            padding: [0, 0],
            precision: OraclePrecision::Int8Accumulator,
            pattern: OraclePattern::Dense { phase: 1 },
        },
        Conv2dCase {
            width: 32,
            height: 32,
            cin: 384,
            cout: 64,
            kernel: [1, 1],
            stride: 1,
            padding: [0, 0],
            precision: OraclePrecision::Int8Accumulator,
            pattern: OraclePattern::Dense { phase: 1 },
        },
        Conv2dCase {
            width: 32,
            height: 32,
            cin: 16,
            cout: 512,
            kernel: [1, 1],
            stride: 1,
            padding: [0, 0],
            precision: OraclePrecision::Int8Accumulator,
            pattern: OraclePattern::Dense { phase: 3 },
        },
        Conv2dCase {
            width: 34,
            height: 34,
            cin: 16,
            cout: 512,
            kernel: [3, 3],
            stride: 1,
            padding: [0, 0],
            precision: OraclePrecision::Int8Accumulator,
            pattern: OraclePattern::Dense { phase: 4 },
        },
        Conv2dCase {
            width: 33,
            height: 33,
            cin: 16,
            cout: 65,
            kernel: [3, 3],
            stride: 2,
            padding: [0, 0],
            precision: OraclePrecision::Int8Accumulator,
            pattern: OraclePattern::Dense { phase: 5 },
        },
        Conv2dCase {
            width: 226,
            height: 226,
            cin: 3,
            cout: 64,
            kernel: [3, 3],
            stride: 1,
            padding: [0, 0],
            precision: OraclePrecision::Int8Accumulator,
            pattern: OraclePattern::Selectors { phase: 6 },
        },
    ]
}

/// Sweeps every native 16-channel input atom through and beyond the current
/// K1 matcher range. With tile-local accumulator destinations, RK3588 passes
/// Cin 16..=384 exactly, including the transition to two tiles above Cin=352.
/// Cin 400..=512 still fail with the second 32-channel output block unwritten,
/// isolating a subsequent CBUF/weight working-set boundary from this fix.
fn int8_accumulator_k1_cin_atom_sweep_cases() -> Vec<Conv2dCase> {
    (16..=512)
        .step_by(16)
        .map(|cin| Conv2dCase {
            width: 32,
            height: 32,
            cin,
            cout: 64,
            kernel: [1, 1],
            stride: 1,
            padding: [0, 0],
            precision: OraclePrecision::Int8Accumulator,
            pattern: OraclePattern::Dense { phase: 1 },
        })
        .collect()
}

/// Guards the exact logical-Cin planning transition: Cin 351 and 352 use one
/// tile, while 353, 354, and 367 use two. All five pass on RK3588 when each
/// accumulator tile owns its destination geometry (2026-08-31).
fn int8_accumulator_k1_cin_boundary_cases() -> Vec<Conv2dCase> {
    [351u32, 352, 353, 354, 367]
        .into_iter()
        .map(|cin| Conv2dCase {
            width: 32,
            height: 32,
            cin,
            cout: 64,
            kernel: [1, 1],
            stride: 1,
            padding: [0, 0],
            precision: OraclePrecision::Int8Accumulator,
            pattern: OraclePattern::Dense { phase: 1 },
        })
        .collect()
}

/// Shapes the first expanded oracle run (RK3588, 2026-08-31) proved are not
/// yet safe. Kept separate from the green regression gate so the failures
/// remain reproducible without making every normal board gate fail:
///
/// * odd 9x7 output extents leave the final logical accumulator(s) unwritten;
/// * several small-Cin shapes corrupt most values at exact 32-lane block
///   boundaries.
fn int8_accumulator_known_limitation_cases() -> Vec<Conv2dCase> {
    let mut cases = Vec::new();
    for (cin, cout, kernel) in [
        (3u32, 1u32, 1usize),
        (3, 31, 3),
        (3, 32, 3),
        (3, 33, 3),
        (5, 64, 1),
        (5, 65, 1),
    ] {
        cases.push(Conv2dCase {
            width: 9,
            height: 7,
            cin,
            cout,
            kernel: [kernel, kernel],
            stride: 1,
            padding: [kernel / 2, kernel / 2],
            precision: OraclePrecision::Int8Accumulator,
            pattern: OraclePattern::Dense { phase: 0 },
        });
    }
    cases
}

fn int8_neutral80_one_hot_four_way_cases() -> Vec<Conv2dCase> {
    let mut cases = Vec::with_capacity(4);
    for signed_input in [false, true] {
        for kernel in [1usize, 3] {
            cases.push(Conv2dCase {
                width: 9,
                height: 7,
                cin: 3,
                cout: 32,
                kernel: [kernel, kernel],
                stride: 1,
                // No padding keeps every selected K3 tap in-bounds, making
                // a tap-order failure distinct from padding semantics.
                padding: [0, 0],
                precision: OraclePrecision::Int8,
                pattern: OraclePattern::OneHotNeutral80 {
                    phase: 2,
                    signed_input,
                },
            });
        }
    }
    cases
}

fn int8_selector_cases(pattern: fn(usize) -> OraclePattern) -> Vec<Conv2dCase> {
    let mut cases = Vec::with_capacity(24);
    for cin in [3u32, 5, 128, 256, 512] {
        for cout in [64u32, 256, 512] {
            cases.push(Conv2dCase {
                width: 28,
                height: 28,
                cin,
                cout,
                kernel: [3, 3],
                stride: 1,
                padding: [1, 1],
                precision: OraclePrecision::Int8,
                pattern: pattern(0),
            });
        }
    }
    for (extent, cin, cout) in [
        (226, 3, 64),
        (226, 64, 64),
        (114, 64, 128),
        (114, 128, 128),
        (58, 128, 256),
        (58, 256, 256),
        (30, 256, 512),
        (30, 512, 512),
        (16, 512, 512),
    ] {
        cases.push(Conv2dCase {
            width: extent,
            height: extent,
            cin,
            cout,
            kernel: [3, 3],
            stride: 1,
            padding: [0, 0],
            precision: OraclePrecision::Int8,
            pattern: pattern(1),
        });
    }
    cases
}

fn int8_raw_coefficient_byte_sweep_case(unit_gain: bool) -> Conv2dCase {
    Conv2dCase {
        width: 1,
        height: 1,
        cin: 1,
        cout: 256,
        kernel: [1, 1],
        stride: 1,
        padding: [0, 0],
        precision: OraclePrecision::Int8,
        pattern: if unit_gain {
            OraclePattern::RawByteSweepUnit
        } else {
            OraclePattern::RawByteSweep
        },
    }
}

fn panic_message(payload: Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else {
        "non-string panic payload".to_string()
    }
}

fn assert_planable_and_gap_free(cases: Vec<Conv2dCase>) {
    for case in cases {
        let shape = case.shape();
        let plan = ConvPlan::new(shape, case.kernel);
        let out_width = shape.output_width(case.kernel) as usize;
        let out_height = shape.output_height(case.kernel) as usize;
        let mut coverage = vec![0u8; out_width * out_height];
        for tile in plan.tiles() {
            let tile_row_capacity = shape.max_tile_input_rows_for_width_and_data_banks(
                tile.columns.in_cols,
                plan.data_banks(),
            );
            assert!(
                tile.rows.in_rows <= tile_row_capacity,
                "{}: tile reads {} rows beyond capacity {}",
                case.label(),
                tile.rows.in_rows,
                tile_row_capacity,
            );
            assert!(
                shape.dense_feature_offset_safe(tile.rows.in_first),
                "{}: tile at in_first={} has input byte offset {} mod 16",
                case.label(),
                tile.rows.in_first,
                tile.rows.in_first * shape.input_row_stride() % 16,
            );
            for y in
                tile.rows.out_first as usize..(tile.rows.out_first + tile.rows.out_rows) as usize
            {
                for x in tile.columns.out_first as usize
                    ..(tile.columns.out_first + tile.columns.out_cols) as usize
                {
                    coverage[y * out_width + x] += 1;
                }
            }
        }
        assert!(
            coverage.iter().all(|count| *count == 1),
            "{}: output coverage has gaps or overlaps",
            case.label(),
        );
    }
}

#[test]
fn cartesian_matrix_is_planable_and_gap_free() {
    let cases = cartesian_cases();
    assert_eq!(cases.len(), 242);
    assert_planable_and_gap_free(cases);
}

#[test]
fn output_channel_group_boundary_matrix_is_planable_and_gap_free() {
    let cases = output_channel_group_boundary_cases();
    assert_eq!(cases.len(), 24);
    assert_planable_and_gap_free(cases);
}

#[test]
fn bank_partition_flip_boundary_matrix_is_planable_and_gap_free() {
    let cases = bank_partition_flip_boundary_cases();
    assert_eq!(cases.len(), 16);
    assert_planable_and_gap_free(cases);
}

#[test]
fn dense_coefficient_vgg_block_matrix_is_planable_and_gap_free() {
    let cases = dense_coefficient_vgg_block_cases();
    assert_eq!(cases.len(), 5);
    assert_planable_and_gap_free(cases);
}

#[test]
fn int8_accumulator_matrices_are_planable_and_gap_free() {
    let regression = int8_accumulator_regression_cases();
    assert_eq!(regression.len(), 9);
    assert_eq!(
        ConvPlan::new(regression[3].shape(), regression[3].kernel)
            .tiles()
            .len(),
        2
    );
    assert_eq!(
        ConvPlan::new(regression[8].shape(), regression[8].kernel)
            .tiles()
            .len(),
        3
    );
    assert_planable_and_gap_free(regression);

    let limitations = int8_accumulator_known_limitation_cases();
    assert_eq!(limitations.len(), 6);
    assert_planable_and_gap_free(limitations);
}

#[test]
fn int8_accumulator_k1_cin_atom_sweep_is_planable_and_gap_free() {
    let cases = int8_accumulator_k1_cin_atom_sweep_cases();
    assert_eq!(cases.len(), 32);
    assert_eq!(cases.first().unwrap().cin, 16);
    assert_eq!(cases.last().unwrap().cin, 512);
    assert!(cases.windows(2).all(|pair| pair[1].cin - pair[0].cin == 16));
    assert_planable_and_gap_free(cases);
}

#[test]
fn int8_accumulator_k1_cin_boundary_is_planable_and_gap_free() {
    let cases = int8_accumulator_k1_cin_boundary_cases();
    assert_eq!(
        cases.iter().map(|case| case.cin).collect::<Vec<_>>(),
        [351, 352, 353, 354, 367]
    );
    assert_eq!(
        ConvPlan::new(cases[1].shape(), cases[1].kernel)
            .tiles()
            .len(),
        1
    );
    assert_eq!(
        ConvPlan::new(cases[2].shape(), cases[2].kernel)
            .tiles()
            .len(),
        2
    );
    assert_planable_and_gap_free(cases);
}

#[test]
fn int8_accumulator_cbuf_split_probe_matrix_is_planable() {
    for cin in [384u32, 385, 400, 512] {
        let case = Conv2dCase {
            width: 32,
            height: 32,
            cin,
            cout: 64,
            kernel: [1, 1],
            stride: 1,
            padding: [0, 0],
            precision: OraclePrecision::Int8Accumulator,
            pattern: OraclePattern::Dense { phase: 1 },
        };
        for weight_banks in 1..=5 {
            let data_banks = 12 - weight_banks;
            let plan =
                ConvPlan::with_cbuf_banks(case.shape(), case.kernel, data_banks, weight_banks);
            assert_eq!(plan.data_banks(), data_banks, "Cin={cin}");
            assert_eq!(plan.weight_banks(), weight_banks, "Cin={cin}");
            assert!(
                !plan.tiles().is_empty(),
                "Cin={cin}, {data_banks}/{weight_banks}"
            );
        }
    }
}

/// Index to resume a hardware sweep at, for picking one back up after a
/// reboot has cleared a sick device. Zero, the default, starts at the top.
fn probe_resume_index() -> usize {
    std::env::var("ROCKET_PROBE_RESUME_AT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}

/// A shape well inside the known-good region, used to prove the device is
/// still healthy. Dense signed K1 accumulator at Cin=64 passes on any
/// healthy RK3588, so a failure here is a statement about the *device*
/// rather than about whatever shape is under test.
fn accumulator_canary_passes(file: &std::fs::File) -> bool {
    let case = Conv2dCase {
        width: 32,
        height: 32,
        cin: 64,
        cout: 64,
        kernel: [1, 1],
        stride: 1,
        padding: [0, 0],
        precision: OraclePrecision::Int8Accumulator,
        pattern: OraclePattern::Dense { phase: 1 },
    };
    let result = catch_unwind(AssertUnwindSafe(|| {
        let fixture = build_fixture(case)?;
        execute_case(file, &fixture)
    }));
    matches!(result, Ok(Ok(_)))
}

/// Runs a case matrix, accumulating shape failures but halting on a sick
/// device.
///
/// Shape failures accumulate exactly as they always did: one bad case never
/// prevents later cases from running. The one exception is the RK3588's sick
/// state. After roughly fifteen to twenty failing jobs the NPU starts
/// returning wrong results for *every* shape, including ones that passed
/// seconds earlier, and nothing reports it -- `prep_bo` still succeeds well
/// inside its timeout. Past that point a sweep measures the device rather
/// than the shapes, so every failure re-checks a known-good canary and the
/// run stops the moment the canary goes.
///
/// The canary is the only reliable discriminator, so do not try to read the
/// device's health off the values. A `got -1515870800` means the DPU never
/// wrote those bytes, which is how the dense Cin>384 failures present, but
/// fully written *wrong* values occur both for genuine shape failures (the
/// Cout=31/33 partial-block cases in the known-limitations probe report
/// `max|diff|` of 45 on a perfectly healthy device) and for a sick one.
///
/// Only a reboot clears the sick state -- not idle, not a fresh process, not
/// a reopened fd. Reboot the board, then resume with
/// `ROCKET_PROBE_RESUME_AT`, which this prints for you.
///
/// **The canary is necessary but not sufficient.** Results here are also
/// order dependent, and at small shapes outright flaky, in ways a Cin=64
/// canary does not detect. In `int8_accumulator_known_limitations_probe`,
/// Cin=3 Cout=33 and Cin=5 Cout=64 both *pass* when they run first and fail
/// mid-sweep, and even in isolation Cin=3 Cout=33 failed once in six
/// otherwise identical runs. Contamination is per-process: a fresh process
/// clears it, intervening successful jobs do not, which points at buffer or
/// DMA-address reuse rather than NPU state. So a verdict on one shape needs
/// `ROCKET_PROBE_RESUME_AT` to run it *first*, repeated a few times -- never
/// one row from one sweep.
fn run_hardware_case_matrix(title: &str, cases: Vec<Conv2dCase>) {
    let total_cases = cases.len();
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let resume = probe_resume_index();
    let mut failures = Vec::new();
    let mut sick_after = None;

    println!("\n=== {title} ===");
    if resume != 0 {
        println!("  resuming at index {resume} of {total_cases}");
    }

    if !accumulator_canary_passes(&file) {
        println!(
            "  CANARY FAILED before any measurement (dense K1 Cin=64, which passes on any\n  \
             healthy device). The NPU is sick -- reboot the board and re-run. Nothing this\n  \
             sweep could print below would be trustworthy."
        );
        panic!("{title} did not run: the NPU is sick, reboot the board");
    }

    for (index, case) in cases.into_iter().enumerate() {
        if index < resume {
            continue;
        }
        let label = case.label();
        let result = catch_unwind(AssertUnwindSafe(|| {
            let fixture = build_fixture(case)?;
            execute_case(&file, &fixture)
        }));
        let failure = match result {
            Ok(Ok(success)) => {
                println!(
                    "[{}/{}] ok   {label} banks={}/{} tiles={}",
                    index + 1,
                    total_cases,
                    success.data_banks,
                    success.weight_banks,
                    success.tiles,
                );
                continue;
            }
            Ok(Err(error)) => {
                println!(
                    "[{}/{}] FAIL {label}\n      {error}",
                    index + 1,
                    total_cases
                );
                format!("{label}: {error}")
            }
            Err(payload) => {
                let error = panic_message(payload);
                println!(
                    "[{}/{}] PANIC {label}\n      {error}",
                    index + 1,
                    total_cases
                );
                format!("{label}: panic: {error}")
            }
        };
        failures.push(failure);

        if !accumulator_canary_passes(&file) {
            sick_after = Some(index);
            break;
        }
    }

    let not_run = match sick_after {
        Some(index) => total_cases - index - 1,
        None => 0,
    };
    let attempted = total_cases - resume.min(total_cases) - not_run;

    println!("\n=== {title} summary ===");
    println!(
        "  passed: {} of {attempted} attempted",
        attempted - failures.len()
    );
    println!("  failed: {}", failures.len());
    if resume != 0 {
        println!("  skipped before resume: {resume}");
    }
    if sick_after.is_some() {
        println!("  not run (device went sick): {not_run}");
    }
    for (index, failure) in failures.iter().enumerate() {
        println!("    {}. {failure}", index + 1);
    }

    if let Some(index) = sick_after {
        println!(
            "\n  CANARY FAILED after case {} of {total_cases}. The device is sick from here on,\n  \
             so that case is the last trustworthy row and the remaining {not_run} were not run.\n  \
             Reboot the board, then continue with:\n    \
             ROCKET_PROBE_RESUME_AT={} <this binary> <test name> --ignored --nocapture",
            index + 1,
            index + 1,
        );
        panic!(
            "{title} stopped after case {} of {total_cases}: the NPU went sick. Reboot the \
             board and resume with ROCKET_PROBE_RESUME_AT={}",
            index + 1,
            index + 1,
        );
    }

    assert!(
        failures.is_empty(),
        "{} of {attempted} cases failed in {title}; complete diagnostics are above",
        failures.len(),
    );
}
#[test]
#[ignore = "needs /dev/accel/accel0 -- validates production affine int8 weights and BS constants"]
fn int8_affine_selector_matrix_runs_every_case_before_failing() {
    let cases = int8_selector_cases(|phase| OraclePattern::SelectorsAffine { phase });
    assert_eq!(cases.len(), 24);
    println!("  raw weight: logical coefficient + per-output zero point");
    println!("  physical Cin padding: per-output zero point");
    println!("  BS constant: -per-output zero point; BS multiplier: 0x4000");
    run_hardware_case_matrix("int8 affine selector matrix", cases);
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- locates the two output-channel-group cutoffs exactly"]
fn output_channel_group_boundary_probe_runs_every_case_before_failing() {
    let cases = output_channel_group_boundary_cases();
    assert_eq!(cases.len(), 24);
    run_hardware_case_matrix("output-channel-group boundary probe", cases);
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- confirms failure starts exactly at ConvPlan's bank-partition flip"]
fn bank_partition_flip_boundary_probe_runs_every_case_before_failing() {
    let cases = bank_partition_flip_boundary_cases();
    assert_eq!(cases.len(), 16);
    run_hardware_case_matrix("bank-partition flip boundary probe", cases);
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- guards dense diverse coefficients at VGG's Cin/Cout-512 blocks"]
fn dense_coefficient_vgg_blocks_match_oracle() {
    let cases = dense_coefficient_vgg_block_cases();
    assert_eq!(cases.len(), 5);
    run_hardware_case_matrix("dense-coefficient VGG block probe", cases);
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- validates exact i32 accumulator values, partial blocks, channel ceilings, and stride"]
fn int8_accumulator_regression_matrix_matches_oracle() {
    let cases = int8_accumulator_regression_cases();
    assert_eq!(cases.len(), 9);
    run_hardware_case_matrix("int8 accumulator regression matrix", cases);
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- locates the dense signed K1 Int8Accumulator Cin boundary"]
fn int8_accumulator_k1_cin_atom_sweep() {
    let cases = int8_accumulator_k1_cin_atom_sweep_cases();
    assert_eq!(cases.len(), 32);
    run_hardware_case_matrix("int8 accumulator K1 Cin atom sweep", cases);
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- resolves the exact dense signed K1 Int8Accumulator Cin transition"]
fn int8_accumulator_k1_cin_boundary_probe() {
    let cases = int8_accumulator_k1_cin_boundary_cases();
    assert_eq!(cases.len(), 5);
    run_hardware_case_matrix("int8 accumulator K1 Cin boundary probe", cases);
}

/// Resolves the Cin/CBUF split matrix, guarding every row with a canary.
///
/// The RK3588 NPU can drop into a state where *every* job returns wrong
/// results, including shapes that passed seconds earlier. Observed on
/// `planck` 2026-08-31 during a wide accumulator sweep: from one case
/// onwards nothing was correct again, `int8_accumulator_k1_cin_boundary_probe`
/// went 5/5 pass to 5/5 fail, and only rebooting the board restored it.
/// Nothing reports an error when this happens -- `prep_bo` still succeeds
/// well inside its timeout -- so a long sweep silently stops measuring
/// shapes and starts measuring the sick device.
///
/// The canary is the only reliable discriminator; the values themselves are
/// not. Every failure in this matrix presents as `got -1515870800`, the
/// sentinel the DPU never overwrote, but elsewhere a healthy device produces
/// fully written wrong values for genuine shape failures too, so that alone
/// proves nothing about the device.
///
/// A single bad shape does not cause it: 15 consecutive Cin=385 failures each
/// left the canary clean, and the whole Cin=384 block still passes 5/5
/// immediately after a Cin=385 failure. It takes roughly fifteen to twenty
/// failing jobs in one process -- one full pass over this matrix is enough,
/// and did it twice. So passes chain freely and failures are worth
/// continuing past, but every failure re-checks the canary and the probe
/// stops the moment the device itself is implicated.
/// `ROCKET_PROBE_RESUME_AT` then picks the matrix back up, by index, once
/// the board has been rebooted.
///
/// The canary between cases also makes the rows *deterministic*: without it
/// the failing rows had ragged mismatch counts and a first-bad channel of
/// 32; with it every failing row is a clean 65536/65536 from c=0.
#[test]
#[ignore = "needs /dev/accel/accel0 -- isolates the accumulator Cin>384 CBUF split boundary"]
fn int8_accumulator_k1_cin_cbuf_split_probe() {
    const CIN: [u32; 4] = [384, 385, 400, 512];
    const SPLITS: u32 = 5;

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");

    let resume = probe_resume_index();

    println!("\n=== int8 accumulator K1 Cin/CBUF split probe ===");
    println!("  matrix: Cin 384/385/400/512 x weight banks 1..=5, ascending");
    println!(
        "  every failure re-checks a Cin=64 canary; the probe stops only if the device is sick"
    );
    if resume != 0 {
        println!("  resuming at index {resume}");
    }

    if !accumulator_canary_passes(&file) {
        println!(
            "\n  CANARY FAILED before any measurement (dense K1 Cin=64, which passes on any\n  \
             healthy device). The NPU is sick -- reboot the board and re-run. Nothing this\n  \
             probe could print below would be trustworthy."
        );
        return;
    }
    println!("  canary ok (dense K1 Cin=64): the device is healthy");

    let mut failures = 0usize;
    for (cin_index, cin) in CIN.into_iter().enumerate() {
        let base = cin_index * SPLITS as usize;
        if base + SPLITS as usize <= resume {
            continue;
        }

        let case = Conv2dCase {
            width: 32,
            height: 32,
            cin,
            cout: 64,
            kernel: [1, 1],
            stride: 1,
            padding: [0, 0],
            precision: OraclePrecision::Int8Accumulator,
            pattern: OraclePattern::Dense { phase: 1 },
        };
        let fixture = build_fixture(case).expect("build focused accumulator fixture");
        println!("\n  Cin={cin}, coefficient_bytes={}", fixture.weights.len());

        for (split_index, weight_banks) in (1..=SPLITS).enumerate() {
            let index = base + split_index;
            let data_banks = 12 - weight_banks;
            if index < resume {
                println!("    [{index:2}] skip banks={data_banks}/{weight_banks} (before resume)");
                continue;
            }

            let result = catch_unwind(AssertUnwindSafe(|| {
                let plan = ConvPlan::with_cbuf_banks(
                    fixture.shape,
                    fixture.case.kernel,
                    data_banks,
                    weight_banks,
                );
                let execution = execute_case_output_with_plan(&file, &fixture, plan)?;
                let report = compare_output(&fixture, &execution.plan, &execution.output);
                Ok::<_, String>((execution.plan.tiles().len(), report))
            }));

            let failure = match result {
                Ok(Ok((tiles, report))) if report.mismatches == 0 => {
                    println!(
                        "    [{index:2}] PASS banks={data_banks}/{weight_banks} tiles={tiles}"
                    );
                    continue;
                }
                Ok(Ok((tiles, report))) => format!(
                    "FAIL banks={data_banks}/{weight_banks} tiles={tiles} mismatches={} \
                     tile_mismatches={:?} first={}",
                    report.mismatches,
                    report.tile_mismatches,
                    report.samples.first().map(String::as_str).unwrap_or("none"),
                ),
                Ok(Err(error)) => format!("ERROR banks={data_banks}/{weight_banks}: {error}"),
                Err(payload) => format!(
                    "PANIC banks={data_banks}/{weight_banks}: {}",
                    panic_message(payload)
                ),
            };

            failures += 1;
            println!("    [{index:2}] {failure}");

            if accumulator_canary_passes(&file) {
                println!("         canary still ok: the device is healthy, this row is real");
                continue;
            }

            println!(
                "\n  CANARY FAILED after index {index} (Cin={cin}, banks={data_banks}/{weight_banks}).\n  \
                 The device is now sick, so this row is the last trustworthy one and every\n  \
                 later point would measure the device rather than its own shape.\n  \
                 Reboot the board, then continue with:\n    \
                 ROCKET_PROBE_RESUME_AT={} <this binary> int8_accumulator_k1_cin_cbuf_split_probe \
                 --ignored --nocapture",
                index + 1,
            );
            return;
        }
    }

    println!("\n=== int8 accumulator K1 Cin/CBUF split probe summary ===");
    println!(
        "  covered the whole matrix: {} of {} points failed, device healthy throughout",
        failures,
        CIN.len() * SPLITS as usize,
    );
}

#[test]
#[ignore = "known RK3588 failures -- manually characterizes accumulator tails and output-block boundaries"]
fn int8_accumulator_known_limitations_probe() {
    let cases = int8_accumulator_known_limitation_cases();
    assert_eq!(cases.len(), 6);
    run_hardware_case_matrix("int8 accumulator known-limitations probe", cases);
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- confirms 0x80 neutral int8 coefficient bytes"]
fn int8_neutral80_one_hot_four_way_confirmation_runs_every_case_before_failing() {
    println!("  packed background/padding byte: 0x80");
    println!("  packed live logical +1 byte:    0x00");
    run_hardware_case_matrix(
        "int8 neutral80 one-hot four-way confirmation",
        int8_neutral80_one_hot_four_way_cases(),
    );
}

fn run_raw_coefficient_byte_sweep(unit_gain: bool, title: &str, conversion: &str) -> Vec<i8> {
    let case = int8_raw_coefficient_byte_sweep_case(unit_gain);
    let fixture = build_fixture(case).expect("build 256-byte coefficient sweep");
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let execution = execute_case_output(&file, &fixture).expect("execute coefficient-byte sweep");
    let observed = (0usize..256)
        .map(|channel| {
            let offset = output_offset(fixture.shape, case.kernel, channel, 0, 0);
            execution.output[offset] as i8
        })
        .collect::<Vec<_>>();

    println!("\n=== {title} ===");
    println!("  input: 1, K1/Cin1, output channel c gets raw coefficient byte c");
    println!("  padding: 0x80, output conversion: {conversion}");
    println!(
        "  banks={}/{} tiles={}",
        execution.plan.data_banks(),
        execution.plan.weight_banks(),
        execution.plan.tiles().len(),
    );
    for base in (0usize..256).step_by(16) {
        let values = observed[base..base + 16]
            .iter()
            .map(|value| format!("{value:>4}"))
            .collect::<Vec<_>>()
            .join(" ");
        println!("  raw 0x{base:02x}..0x{:02x}: {values}", base + 15);
    }
    observed
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- maps every dense int8 coefficient byte"]
fn int8_dense_coefficient_raw_byte_sweep() {
    let observed = run_raw_coefficient_byte_sweep(
        false,
        "dense int8 coefficient raw-byte sweep",
        "gain 128, zero point -128",
    );
    println!("  hypothesis: output(raw) = raw interpreted as signed i8");

    let differences = observed
        .iter()
        .enumerate()
        .map(|(raw, &got)| i32::from(got) - i32::from(raw as u8 as i8))
        .collect::<Vec<_>>();
    let exact = differences
        .iter()
        .filter(|&&difference| difference == 0)
        .count();
    let within_one = differences
        .iter()
        .filter(|&&difference| difference.abs() <= 1)
        .count();
    let max_difference = differences
        .iter()
        .map(|difference| difference.abs())
        .max()
        .unwrap_or(0);
    println!("  exact: {exact}/256");
    println!("  within one LSB: {within_one}/256");
    println!("  max |difference|: {max_difference}");
    for (raw, (&got, &difference)) in observed.iter().zip(&differences).enumerate() {
        if difference.abs() > 1 {
            println!(
                "    raw=0x{raw:02x} signed={} expected={} got={got} difference={difference}",
                raw as u8 as i8, raw as u8 as i8,
            );
        }
    }

    assert_eq!(
        within_one, 256,
        "coefficient-byte mapping differed from signed-byte hypothesis by more than one LSB; \
         complete map is printed above",
    );
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- maps dense int8 coefficient bytes at unit gain"]
fn int8_dense_coefficient_raw_byte_unit_gain_sweep() {
    let observed = run_raw_coefficient_byte_sweep(
        true,
        "dense int8 coefficient raw-byte unit-gain sweep",
        "ordinary unit gain, zero point 0",
    );
    let distinct = observed
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    println!("  distinct outputs ({}): {distinct:?}", distinct.len());
    for raw in [0x00usize, 0x01, 0x7f, 0x80, 0x81, 0xff] {
        println!(
            "  landmark raw=0x{raw:02x} signed={:>4} -> output={:>4}",
            raw as u8 as i8, observed[raw],
        );
    }
    assert_eq!(
        observed[0x80], 0,
        "raw 0x80 stopped being neutral at ordinary unit gain; complete map is above",
    );
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- cross-compile for aarch64 and run on the RK3588 board"]
fn cartesian_conv2d_oracle_sweep_runs_every_case_before_failing() {
    let cases = cartesian_cases();
    let total_cases = cases.len();
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let mut failures = Vec::new();
    let mut passed_by_kind = BTreeMap::<(&str, &str), usize>::new();

    println!(
        "\n=== Conv2D oracle Cartesian sweep: {} cases ===",
        cases.len()
    );
    for (index, case) in cases.into_iter().enumerate() {
        let label = case.label();
        let result = catch_unwind(AssertUnwindSafe(|| {
            let fixture = build_fixture(case)?;
            execute_case(&file, &fixture)
        }));
        match result {
            Ok(Ok(success)) => {
                println!(
                    "[{}/{}] ok   {label} banks={}/{} tiles={}",
                    index + 1,
                    total_cases,
                    success.data_banks,
                    success.weight_banks,
                    success.tiles,
                );
                *passed_by_kind
                    .entry((case.precision.name(), case.pattern.name()))
                    .or_default() += 1;
            }
            Ok(Err(error)) => {
                println!(
                    "[{}/{}] FAIL {label}\n      {error}",
                    index + 1,
                    total_cases,
                );
                failures.push(format!("{label}: {error}"));
            }
            Err(payload) => {
                let error = panic_message(payload);
                println!(
                    "[{}/{}] PANIC {label}\n      {error}",
                    index + 1,
                    total_cases,
                );
                failures.push(format!("{label}: panic: {error}"));
            }
        }
    }

    println!("\n=== Conv2D oracle summary ===");
    for ((precision, pattern), passed) in passed_by_kind {
        println!("  {precision:>4} {pattern:<9}: {passed} passed");
    }
    println!("  failures: {}", failures.len());
    for (index, failure) in failures.iter().enumerate() {
        println!("    {}. {failure}", index + 1);
    }

    assert!(
        failures.is_empty(),
        "{} of {} Conv2D oracle cases failed; complete diagnostics are printed above",
        failures.len(),
        total_cases,
    );
}
