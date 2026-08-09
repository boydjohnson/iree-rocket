//! Cartesian Conv2D correctness sweep against an independent logical oracle.
//!
//! Every case executes through `ConvPlan::new`, production HWCF weight
//! packing, and the RK3588 device. Failures are accumulated: one bad case
//! never prevents later cases from running, and the test asserts only after
//! printing the complete summary.
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
    conv::{Buffers, ConvPlan},
    device::{Buffer, JobDesc, close_bo, fini_bo, prep_bo, submit_jobs, unmap_bo},
};

const DEVICE_PATH: &str = "/dev/accel/accel0";
const PAGE_BYTES: usize = 4096;
const PER_CASE_TIMEOUT_NS: u64 = 5_000_000_000;

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
    let fd = file.as_raw_fd();
    let shape = fixture.shape;
    let kernels = fixture.case.kernel;
    let plan = ConvPlan::new(shape, kernels);
    let output_len = output_storage_bytes(shape, kernels);

    unsafe {
        let input = OwnedBuffer::from_bytes(fd, &fixture.input, file);
        let weights = OwnedBuffer::from_bytes(fd, &fixture.weights, file);
        let bias = OwnedBuffer::from_bytes(fd, &fixture.bias, file);
        let output = OwnedBuffer::new(fd, output_len, file);
        ptr::write_bytes(output.buffer.host_ptr, 0, output.buffer.size);

        let programs = plan.programs_with_buffers(Buffers {
            input: input.buffer.dma_address,
            weights: weights.buffer.dma_address,
            bias: bias.buffer.dma_address,
            output: output.buffer.dma_address,
        });
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

        let output = std::slice::from_raw_parts(output.buffer.host_ptr, output_len).to_vec();
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
/// convolutions) shows severe end-to-end disagreement against CPU -- MAE
/// roughly 30x worse than the same run capped at Cin/Cout <= 256, only
/// 20.70% of logits within the 0.05 tolerance that the <= 256 run hit
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

/// Cin=8/Cout=16, 1x1, 30x30, fp16, dense coefficients -- the exact shape a
/// real compiled IREE module (iree-rocket-design-spike's two_conv_probe,
/// two independent Rocket dispatches sharing one command buffer) produced
/// scattered wrong-pixel corruption at (roughly 0.3-4.5% of output
/// elements, clustered by whole pixel across all 16 channels at once, not
/// at any spatial edge). A 50ms sleep between the two hardware submissions
/// made no difference, and a single, standalone dispatch of this exact
/// shape reproduced the identical corruption rate -- ruling out both a
/// scheduling race and the two-dispatches-per-command-buffer change
/// itself. The Cartesian sweep above never covered Cin=8 or Cout=16 (its
/// `input_channels`/`output_channels` grids jump 5 -> 64 and start at 64
/// respectively), so this shape was untested at the ConvPlan/hardware
/// level before now.
///
/// Cin=8 needs NC1HWC2 input padding (`command_buffer.rs`'s
/// `packed_input_bytes_per_pixel` pads to `in_channels.max(16)` channels,
/// a threshold that is not itself precision-aware even though fp16's
/// natural atomic block is 8 channels, not 16 -- untested territory this
/// case exercises for the first time). Caveat: this oracle fixture
/// constructs its packed input directly via `feature_offset` (an
/// independently written formula, not a call through
/// `pack_nhwc_to_nc1hwc2_padded`, the function the real driver's
/// `dispatch()` actually uses for a dense NHWC input) -- if this case
/// passes here but the real compiled-module path still fails, that
/// narrows the bug specifically to `pack_nhwc_to_nc1hwc2_padded`'s own
/// implementation rather than ConvPlan/CBUF/hardware execution at this
/// shape.
fn small_input_channel_dynamic_dispatch_cases() -> Vec<Conv2dCase> {
    vec![Conv2dCase {
        width: 30,
        height: 30,
        cin: 8,
        cout: 16,
        kernel: [1, 1],
        stride: 1,
        padding: [0, 0],
        precision: OraclePrecision::Fp16,
        pattern: OraclePattern::Dense { phase: 0 },
    }]
}

/// ResNet-50's distinct convolution shapes (ImageNet, 224x224x3 input),
/// filtered to what is both new coverage and constructible today.
///
/// Every 1x1/3x3 convolution in the stem and all four stages reduces to 23
/// unique (width, height, Cin, Cout, kernel, stride, pad) tuples once
/// repeated blocks and shared shapes (a stage's expansion conv is the same
/// shape whether it is block 1's or a later block's) are collapsed. Two
/// filters were applied to that list:
///
/// - Four stage-2/3 1x1 shapes at width 28, stride 1 exactly match a
///   `cartesian_cases` main-grid point (Cin/Cout both drawn from its
///   {3,4,5,64,128,192,256,320,384,448,512} x {64,128,256,512} grid, kernel
///   1 or 3, stride 1) and are already exercised there under `Counting`.
///   Dropped rather than duplicated.
/// - Eleven shapes need Cin or Cout of 1024 or 2048: stage 3's expansion/
///   downsample (Cout=1024) and reduce (Cin=1024), and all four of stage
///   4's channel-changing 1x1 convs (up to Cin=2048/Cout=2048).
///   `MAX_INPUT_CHANNELS`/`MAX_OUTPUT_CHANNELS` cap the builder at 512 --
///   the vendor-capture-validated range -- so `Shape::with_out_channels`
///   panics on these before a case could even be constructed. Extending
///   past 512 needs new vendor captures (the large-Cin sweep that raised
///   the ceiling from 80/128 to 512 -- see DESIGN_NOTES.md -- stopped
///   there) and is out of scope here; stage 3 and 4's *spatial* 3x3 convs
///   (Cin=Cout=256 and Cin=Cout=512, both within range) are still included.
/// - The 224x224 K7-stride-2 stem is excluded for a sharper reason than
///   "untested": `ConvPlan::new`'s `assert_large_kernel_plan_case`
///   (`conv.rs`) explicitly refuses any kernel above 3x3 combined with a
///   stride other than 1 -- "automatic planning above 3x3 currently has
///   capture backing only at stride 1" -- so this case cannot be
///   constructed at all today, not merely omitted from prior coverage.
///   Every ResNet family's stem uses this exact K7/stride-2 shape, which
///   makes it worth its own follow-up (a vendor capture at K7/stride-2,
///   the formula fix, and only then a hardware case here) rather than
///   folding into this sweep.
///
/// The remaining eleven are new: three spatial extents (56, 14, 7) no
/// existing case touches at all, and the first hardware coverage anywhere
/// in this file of stride 2 at real ResNet channel counts -- every
/// existing case (`cartesian_cases` and every other `_cases` function) is
/// stride 1; `conv_wide_shape_hw.rs` covers stride 2..4 but only at
/// Cin=3/Cout=8.
///
/// `Dense` was chosen for full-coefficient-diversity coverage rather than
/// `Counting`/`Selectors`, per its doc comment's caveat that per-shape fp16
/// exactness was "verified by hand" -- confirmed here too, by computing
/// `expected_accumulator` over every output pixel/channel of all eleven
/// shapes: worst case is 63, nowhere near the 2048 exact-integer ceiling a
/// bit-exact (tolerance 0.0) fp16 comparison depends on.
fn resnet50_probe_cases() -> Vec<Conv2dCase> {
    let shapes: [(u32, u32, u32, u32, [usize; 2], u32, [usize; 2], &str); 11] = [
        (56, 56, 64, 64, [1, 1], 1, [0, 0], "stage1 reduce (block 1)"),
        (56, 56, 64, 64, [3, 3], 1, [1, 1], "stage1 3x3"),
        (
            56,
            56,
            64,
            256,
            [1, 1],
            1,
            [0, 0],
            "stage1 expand / block-1 downsample",
        ),
        (
            56,
            56,
            256,
            64,
            [1, 1],
            1,
            [0, 0],
            "stage1 reduce (blocks 2-3)",
        ),
        (
            56,
            56,
            256,
            128,
            [1, 1],
            1,
            [0, 0],
            "stage2 reduce (block 1)",
        ),
        (56, 56, 128, 128, [3, 3], 2, [1, 1], "stage2 3x3, stride 2"),
        (
            56,
            56,
            256,
            512,
            [1, 1],
            2,
            [0, 0],
            "stage2 downsample, stride 2",
        ),
        (28, 28, 256, 256, [3, 3], 2, [1, 1], "stage3 3x3, stride 2"),
        (14, 14, 256, 256, [3, 3], 1, [1, 1], "stage3 3x3"),
        (14, 14, 512, 512, [3, 3], 2, [1, 1], "stage4 3x3, stride 2"),
        (7, 7, 512, 512, [3, 3], 1, [1, 1], "stage4 3x3"),
    ];
    shapes
        .into_iter()
        .map(
            |(width, height, cin, cout, kernel, stride, padding, _label)| Conv2dCase {
                width,
                height,
                cin,
                cout,
                kernel,
                stride,
                padding,
                precision: OraclePrecision::Fp16,
                pattern: OraclePattern::Dense { phase: 0 },
            },
        )
        .collect()
}

/// The exact hardware-job shapes and order that isolated a persistent-CBUF
/// state bug found while bringing up ResNet-50: a `128x128` Cin=Cout=64 1x1
/// convolution plans to the extreme CBUF split `data_banks=11,
/// weight_banks=1`, and when that specific job runs immediately before a
/// `66x66` Cin=Cout=128 3x3 convolution (banks 3/9), the second job's upper
/// output-channel half comes back wrong on real hardware (`max|CPU diff| =
/// 19.0008`, only 5,089 of 262,144 values within tolerance) even though the
/// same 3x3 convolution is exact both standalone and when preceded instead
/// by the harmless `130x130` Cin=Cout=64 3x3 predecessor (banks 9/3). See
/// DESIGN_NOTES.md, "Planck result: the single Cout=64 1x1 job is the
/// poison".
///
/// `Counting` keeps both cases bit-exact in fp16 (worst-case accumulator is
/// 1,152, well under the 2,048 exact-integer ceiling) rather than `Dense`,
/// because this isolates a CBUF register/state mechanism, not coefficient
/// content -- the original full-model isolation already showed the
/// corruption does not depend on what the weights are.
///
/// At the time that isolation was done, this was an open defect and the
/// compiler workaround kept this 1x1 shape on CPU rather than fixing the
/// driver/hardware. Run on `planck` after `f731cc8` (weight-demand CBUF
/// partition fix) and `53dc3b6` (`data_entries` rounding fix), this exact
/// sequence now passes -- the corruption no longer reproduces, so the
/// poison this case isolated appears to have been resolved as a side
/// effect of one or both of those fixes rather than by a dedicated CBUF
/// reset. The case stays as a permanent regression against that finding
/// re-appearing.
fn poison_predecessor_cases() -> Vec<Conv2dCase> {
    vec![
        Conv2dCase {
            width: 128,
            height: 128,
            cin: 64,
            cout: 64,
            kernel: [1, 1],
            stride: 1,
            padding: [0, 0],
            precision: OraclePrecision::Fp16,
            pattern: OraclePattern::Counting,
        },
        Conv2dCase {
            width: 66,
            height: 66,
            cin: 128,
            cout: 128,
            kernel: [3, 3],
            stride: 1,
            padding: [0, 0],
            precision: OraclePrecision::Fp16,
            pattern: OraclePattern::Counting,
        },
    ]
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
fn small_input_channel_dynamic_dispatch_matrix_is_planable_and_gap_free() {
    let cases = small_input_channel_dynamic_dispatch_cases();
    assert_eq!(cases.len(), 1);
    assert_planable_and_gap_free(cases);
}

#[test]
fn resnet50_probe_matrix_is_planable_and_gap_free() {
    let cases = resnet50_probe_cases();
    assert_eq!(cases.len(), 11);
    assert_planable_and_gap_free(cases);
}

#[test]
fn poison_predecessor_matrix_is_planable_and_gap_free() {
    let cases = poison_predecessor_cases();
    assert_eq!(cases.len(), 2);
    assert_planable_and_gap_free(cases);
}

/// The poison mechanism is specifically about the 11/1 -> 3/9 bank-split
/// sequence, not just these two shapes. Lock in the exact split so a future
/// `ConvPlan` formula change can't silently stop this test from exercising
/// the bug it exists to catch.
#[test]
fn poison_predecessor_matrix_hits_the_isolated_bank_split() {
    let banks = poison_predecessor_cases()
        .into_iter()
        .map(|case| {
            let plan = ConvPlan::new(case.shape(), case.kernel);
            (plan.data_banks(), plan.weight_banks())
        })
        .collect::<Vec<_>>();
    assert_eq!(
        banks,
        vec![(11, 1), (3, 9)],
        "poison predecessor cases no longer plan to the isolated 11/1 -> 3/9 bank split; \
         the DESIGN_NOTES.md mechanism this test targets may no longer apply to these shapes",
    );
}

fn run_hardware_case_matrix(title: &str, cases: Vec<Conv2dCase>) {
    let total_cases = cases.len();
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(DEVICE_PATH)
        .expect("failed to open RK3588 NPU device");
    let mut failures = Vec::new();

    println!("\n=== {title} ===");
    for (index, case) in cases.into_iter().enumerate() {
        let label = case.label();
        let result = catch_unwind(AssertUnwindSafe(|| {
            let fixture = build_fixture(case)?;
            execute_case(&file, &fixture)
        }));
        match result {
            Ok(Ok(success)) => println!(
                "[{}/{}] ok   {label} banks={}/{} tiles={}",
                index + 1,
                total_cases,
                success.data_banks,
                success.weight_banks,
                success.tiles,
            ),
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

    println!("\n=== {title} summary ===");
    println!("  passed: {}", total_cases - failures.len());
    println!("  failed: {}", failures.len());
    for (index, failure) in failures.iter().enumerate() {
        println!("    {}. {failure}", index + 1);
    }
    assert!(
        failures.is_empty(),
        "{} of {} cases failed in {title}; complete diagnostics are above",
        failures.len(),
        total_cases,
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
#[ignore = "needs /dev/accel/accel0 -- tests whether dense diverse coefficients break at VGG's Cin/Cout-512 blocks"]
fn dense_coefficient_vgg_block_probe_runs_every_case_before_failing() {
    let cases = dense_coefficient_vgg_block_cases();
    assert_eq!(cases.len(), 5);
    run_hardware_case_matrix("dense-coefficient VGG block probe", cases);
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- confirms whether Cin=8/Cout=16 1x1 reproduces the scattered-pixel corruption a real compiled module hit"]
fn small_input_channel_dynamic_dispatch_probe_runs_every_case_before_failing() {
    let cases = small_input_channel_dynamic_dispatch_cases();
    assert_eq!(cases.len(), 1);
    run_hardware_case_matrix("small input-channel dynamic dispatch probe", cases);
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- ResNet-50's shapes not already exercised by the cartesian sweep, at real spatial extents/strides"]
fn resnet50_probe_runs_every_case_before_failing() {
    let cases = resnet50_probe_cases();
    assert_eq!(cases.len(), 11);
    run_hardware_case_matrix("ResNet-50 shape probe", cases);
}

#[test]
#[ignore = "needs /dev/accel/accel0 -- regression for a persistent-CBUF poison sequence (Cout=64 \
            1x1, banks 11/1, immediately before Cout=128 3x3, banks 3/9) that used to corrupt \
            the second job's output; passing as of the weight-demand and data_entries CBUF \
            fixes, see DESIGN_NOTES.md"]
fn poison_predecessor_sequential_regression_runs_every_case_before_failing() {
    let cases = poison_predecessor_cases();
    assert_eq!(cases.len(), 2);
    run_hardware_case_matrix("poison predecessor sequential regression", cases);
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
