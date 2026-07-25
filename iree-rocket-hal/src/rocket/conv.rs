//! Caller-driven tiled convolution dispatched as a *sequence of hardware
//! tasks on one NPU core*, using the core's ping-pong register groups.
//!
//! # Why this is a separate module from `regcmd.rs`
//!
//! `regcmd.rs` already splits an oversized conv along height
//! (`plan_conv_tasks`, a port of Mesa's `rkt_split_tasks()`), but that
//! split is driven purely by *CBUF capacity*: it makes each tile as tall as
//! the 12 CBUF banks allow and no taller. Its hardware-validated dispatch
//! path is also one DRM job per tile, each fenced from the CPU
//! (`conv_hw.rs::run_position_conv_fp16_with_weights`), because a real
//! RK3588 run of the same regcmds submitted as multiple tasks in one
//! `drm_rocket_job` produced correct task-0 rows and left every later row
//! at zero.
//!
//! This module is about the other half of that story: tiling chosen by the
//! *caller* (from the MAC-limit/row-tiling model in the design notes, not
//! just CBUF pressure), dispatched as a task sequence the hardware itself
//! walks, with the per-block register groups ping-ponged so tile N+1's
//! registers are fetched while tile N is still executing.
//!
//! # The ping-pong mechanism (TRM chapter 36)
//!
//! Every engine block (CNA/CORE/DPU/DPU_RDMA/PPU/PPU_RDMA) has **two**
//! shadow register groups. Per `RKNN_cna_operation_enable`
//! (chapter36.txt:960): "This register and after this are all shadowed for
//! ping-pong operation" -- i.e. every register at or past each block's
//! `op_enable` offset is double-buffered, while `S_STATUS` (+0x000) and
//! `S_POINTER` (+0x004) below it are not. `S_POINTER`
//! (chapter36.txt:912-946) is therefore the control surface:
//!
//! - `pointer` -- which group the *next* register writes land in;
//! - `pointer_pp_en` -- let hardware toggle `pointer` automatically;
//! - `pointer_pp_mode` -- toggle by executer (0) or by pointer (1);
//! - `executer_pp_en` -- also alternate the two executers;
//! - `pointer_pp_clear`/`executer_pp_clear` -- W1C, reset both to group 0.
//!
//! On the PC side, `RKNN_pc_task_con` (chapter36.txt:813-841) carries
//! `task_number` (total tasks in the run), `task_count_clear` ("before task
//! started, it is suggested to clear"), and `task_pp_en`: with it clear,
//! "the second group register setting is fetched after first group task
//! operation is finished"; with it set, "the second group register setting
//! is fetched immediately after first group's register fetching is
//! finished". That is the actual latency win -- register fetch for tile N+1
//! overlaps tile N's compute.
//!
//! # Who programs ping-pong: the driver already does
//!
//! **[`PointerMode`] has been hardware-tested and makes no difference.**
//! Keeping it, and this explanation, because the reason is worth knowing.
//!
//! Within this crate, ping-pong looks half-configured:
//! `build_conv_cna_core_dpu_dpu_rdma` emits `DPU_S_POINTER` and
//! `DPU_RDMA_S_POINTER` with `pointer_pp_en`, `executer_pp_en` and
//! `pointer_pp_mode=1` all set (`regcmd.rs`, immediately after
//! `CNA_CONV_CON1` -- a faithful port of Mesa, which sets them
//! unconditionally), while nothing here ever writes `CNA_S_POINTER` or
//! `CORE_S_POINTER`.
//!
//! But the *kernel* fills that gap. Mainline `rocket_job.c`'s
//! `rocket_job_hw_submit()` writes, over AHB, before every single task's
//! kick:
//!
//! ```text
//! rocket_cna_writel(core, S_POINTER, CNA_S_POINTER_POINTER_PP_EN(1) |
//!                    CNA_S_POINTER_EXECUTER_PP_EN(1) |
//!                    CNA_S_POINTER_POINTER_PP_MODE(1) | extra_bit);
//! rocket_core_writel(core, S_POINTER, CORE_S_POINTER_POINTER_PP_EN(1) |
//!                     CORE_S_POINTER_EXECUTER_PP_EN(1) |
//!                     CORE_S_POINTER_POINTER_PP_MODE(1) | extra_bit);
//! ```
//!
//! (`extra_bit = 0x10000000 * core->index`, a reserved-bit core selector
//! not documented in the TRM.) So all four blocks are already armed
//! identically before this module writes anything, which is exactly why
//! [`PointerMode::Off`], [`PointerMode::AutoToggle`] and
//! [`PointerMode::ExplicitPerTask`] produced identical hardware results:
//! nothing here was ever the deciding factor.
//!
//! Note also that the driver's value leaves the `pointer` field itself at
//! 0, re-selecting group 0 on every task -- so whatever advances a group
//! between tasks, it is not a pointer the driver preserves.
//!
//! `PC_TASK_CON` is likewise the driver's, and it hardcodes **one** task
//! per kick:
//!
//! ```text
//! rocket_pc_writel(core, TASK_CON, PC_TASK_CON_RESERVED_0(1) |
//!                   PC_TASK_CON_TASK_COUNT_CLEAR(1) |
//!                   PC_TASK_CON_TASK_NUMBER(1) |
//!                   PC_TASK_CON_TASK_PP_EN(1));
//! ```
//!
//! That is the load-bearing constraint on everything below: the driver's
//! design is one PC task per kernel task, re-kicked from each task's
//! completion IRQ -- not one PC run walking a chain.
//!
//! # Two ways to run the sequence
//!
//! `plan_tiled_conv` + `build_tiled_conv_regcmds` produce one regcmd buffer
//! per tile. Which entity walks that list is the caller's choice, and the
//! distinction matters:
//!
//! - **kernel-walked**: hand all buffers to `device::submit_tasks` as one
//!   job's task array. Mainline `rocket_job.c` writes `PC_BASE_ADDRESS`/
//!   `PC_REGISTER_AMOUNTS`/`PC_OPERATION_ENABLE` itself per task and only
//!   dispatches task N+1 from task N's completion IRQ, so there is an IRQ
//!   round-trip between tiles and the PC never sees `task_number > 1`.
//! - **hardware-walked**: [`link_tiled_conv_regcmds`] patches each tile's
//!   trailing `PC_BASE_ADDRESS`/`PC_REGISTER_AMOUNTS` pair to point at the
//!   *next* tile's regcmd buffer, then the caller submits **only tile 0**
//!   as a single-task job. The PC would follow the embedded chain for the
//!   rest, raising one completion IRQ for the whole run
//!   (`pc_interrupt_mask` "sets the masking that applies to the last task
//!   in the running group", chapter36.txt:93-94 as documented in
//!   `builders/pc.rs`).
//!
//!   **This cannot work through the mainline driver as it stands**, and the
//!   hardware run agrees (only tile 0's rows landed, identically for both
//!   [`RegisterAmount`] encodings). `rocket_job_hw_submit()` writes
//!   `PC_TASK_CON` with `TASK_NUMBER(1)` and `TASK_COUNT_CLEAR(1)` on every
//!   kick, so the PC is told there is exactly one task to run. A tile 0
//!   regcmd that raises `task_number` is fetched *after* that write, i.e.
//!   after the run it would need to describe has already started. Making
//!   this path real needs a driver change (pass the job's task count
//!   through to `PC_TASK_CON`), not a regcmd change; it is kept here
//!   because the emission side is ready for that driver, and because it
//!   cleanly rules the chain out as an explanation for anything else.
//!
//! # Hardware status (`tests/tiled_conv_hw.rs`, real RK3588)
//!
//! - one tile, and row tiles as one DRM job each: **correct**, exact
//!   against a position-dependent oracle;
//! - row tiles as multiple tasks in one job: **tile 0's rows correct, every
//!   later row zero**, identically across all three [`PointerMode`]s and
//!   both [`RegisterAmount`]s.
//!
//! The fence *is* signaled rather than timing out, and the driver only
//! signals `done_fence` once `next_task_idx == task_count`, so every task
//! was submitted -- the later tiles run and produce nothing, rather than
//! never being dispatched. Root cause still open; see that file's
//! diagnostic tests.

use crate::rocket::{
    builders::{
        Bits, DOMAIN_PC, RegCmd, Register, RegisterMeta, cna::CnaSPointer, core::CoreSPointer,
        dpu::DpuSPointer, dpu_rdma::DpuRdmaSPointer, pc::PCTaskCon,
    },
    executable_format::validate_conv_shape,
    regcmd::{
        ATOMIC_K_SIZE, CBUF_BANKS, CBUF_ENTRIES_PER_BANK, ConvBuffers, ConvShape, ConvTask,
        KICK_CNA, KICK_CORE, KICK_DPU, KICK_DPU_RDMA, Precision, build_conv_cna_core_dpu_dpu_rdma,
        compute_task_input_channels, conv_entries_per_slice, conv_weights_banks, link_regcmd_tasks,
        push_kick_for_task_count, task_link_trailer_index,
    },
    registers::{REG_PC_BASE_ADDRESS, REG_PC_REGISTER_AMOUNTS},
};

/// NPU cores on an RK3588. The design notes' worked example row-tiles a
/// 256x256 conv 86/85/85 across these three; this module dispatches a tile
/// sequence to a *single* core (the mainline driver picks which), so the
/// count is only used as the default tile count and by
/// [`TiledConv::parallel_cycles`]'s three-core comparison figure.
pub const NPU_CORE_COUNT: u32 = 3;

/// The MAC array's two axes for a given precision.
///
/// The TRM documents int8 as a literal 32x32 array (1024 MACs/cycle,
/// chapter36.txt:1083) and fp16 as loading 16 kernels per group instead of
/// int8's 32 (chapter36.txt:1043-1045). Since 512/16 = 32, only the
/// kernel-group axis shrinks going int8 -> fp16; the channel-lane axis
/// stays at 32. Multiply by [`NPU_CORE_COUNT`] to recover the design
/// notes' "1024x3 / 512x3 MACs per cycle" headline numbers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MacArray {
    /// Output channels (kernels) resident per array pass.
    pub kernel_groups: u32,
    /// Input channels reduced per array pass.
    pub channel_lanes: u32,
}

impl MacArray {
    /// MACs/cycle for one core.
    pub fn macs_per_cycle(&self) -> u32 {
        self.kernel_groups * self.channel_lanes
    }
}

/// Per-precision MAC array geometry. Only the two precisions
/// [`Precision`] models are covered; int4 (2048 MACs/cycle/core) and
/// tf32 (256) appear in the design notes' limit table but have no
/// `Precision` variant, so they are deliberately absent rather than
/// guessed at.
pub fn mac_array(precision: Precision) -> MacArray {
    match precision {
        Precision::Int8 => MacArray {
            kernel_groups: 32,
            channel_lanes: 32,
        },
        Precision::Fp16 => MacArray {
            kernel_groups: 16,
            channel_lanes: 32,
        },
    }
}

/// Array passes one output pixel costs, following the design notes'
/// per-tap model: each of the `Kh*Kw` kernel taps is one array pass over
/// `(Cin, Cout)`, accumulating into the output pixel across taps via the
/// CNA accumulator (chapter36.txt:95-96), so a shape whose channel counts
/// fit the array in one pass costs exactly `Kh*Kw` cycles per pixel.
///
/// The two `div_ceil` factors extend that model past the notes' worked
/// example (which assumed `Cout <= kernel_groups` and
/// `Cin_padded <= channel_lanes`, both true at Cin=Cout=3) to shapes that
/// need multiple passes per tap. Channel counts are the hardware's
/// *padded* ones ([`compute_task_input_channels`]) -- lanes are consumed by
/// the padding, which is exactly the notes' "array-consumed vs. real MACs"
/// distinction.
///
/// Depthwise is a distinct hardware mode (`conv_mode = Depthwise
/// convolution`, chapter36.txt:1028-1034) with no cross-channel reduction:
/// the lanes carry channel parallelism instead of a reduction, so the
/// kernel-group factor drops out.
pub fn cycles_per_pixel(shape: &ConvShape) -> u64 {
    let array = mac_array(shape.precision);
    let taps = u64::from(shape.weights_width) * u64::from(shape.weights_height);
    let lane_passes = u64::from(compute_task_input_channels(shape).div_ceil(array.channel_lanes));
    if shape.depthwise {
        taps * lane_passes
    } else {
        let kernel_passes = u64::from(shape.output_channels.div_ceil(array.kernel_groups));
        taps * lane_passes * kernel_passes
    }
}

/// MACs a single output pixel actually needs, ignoring array padding --
/// the design notes' "real MACs/px" column.
pub fn real_macs_per_pixel(shape: &ConvShape) -> u64 {
    let taps = u64::from(shape.weights_width) * u64::from(shape.weights_height);
    if shape.depthwise {
        taps * u64::from(shape.input_channels)
    } else {
        taps * u64::from(shape.input_channels) * u64::from(shape.output_channels)
    }
}

/// How the register-group pointer is managed across a tile sequence.
///
/// `S_POINTER` is *not* shadowed (it sits below each block's `op_enable`,
/// see this module's doc comment), so a write to it takes effect for the
/// register writes that follow it in the same task program. That makes all
/// three of these expressible purely in the regcmd stream, with no AHB
/// slave-mode access -- but it also means the payload's own
/// `DPU_S_POINTER`/`DPU_RDMA_S_POINTER` writes have to be reconciled with
/// whatever this asks for, not just prepended to (see
/// [`build_tiled_conv_regcmds`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PointerMode {
    /// Hardware owns the pointer. Tile 0 pulses `pointer_pp_clear`/
    /// `executer_pp_clear` on all four blocks and arms them with
    /// `pointer_pp_en` + `pointer_pp_mode=1` ("pointer ping-pong by
    /// pointer": current pointer 0 -> next toggles to 1); tiles 1..N touch
    /// no `S_POINTER` at all, including the payload's own DPU/DPU_RDMA
    /// writes, which are dropped so nothing re-forces `pointer = 0` after
    /// the hardware has advanced it.
    ///
    /// This is the arrangement the TRM describes, and the default.
    #[default]
    AutoToggle,
    /// Every tile drives `pointer = index & 1` explicitly on all four
    /// blocks with `pointer_pp_en` clear -- the payload's DPU/DPU_RDMA
    /// writes are rewritten in place rather than dropped, so they keep
    /// their position ahead of the DPU's shadowed registers. Same
    /// alternation as `AutoToggle`, driven by the regcmd stream instead of
    /// by hardware; the fallback if hardware advances the pointer
    /// differently than the TRM text implies.
    ExplicitPerTask,
    /// Leave the payload's `S_POINTER` programming exactly as it is: DPU
    /// and DPU_RDMA armed for ping-pong, CNA and CORE untouched at reset.
    /// Byte-for-byte what `build_conv_regcmd_tasks` emits today, so it is
    /// the control case for attributing a hardware difference to this
    /// module's ping-pong changes rather than to its tiling.
    Off,
}

/// Ping-pong configuration for a tile sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PingPong {
    pub pointer_mode: PointerMode,
    /// Also alternate the two hardware executers (`executer_pp_en`).
    /// Ignored under [`PointerMode::Off`], which writes no `S_POINTER` of
    /// its own to carry it (the payload's DPU/DPU_RDMA writes set it
    /// regardless).
    pub executers: bool,
    /// Program `PC_TASK_CON` in tile 0: `task_count_clear`, the real
    /// `task_number`, and `task_pp_en` (fetch tile N+1's registers as soon
    /// as tile N's *fetch* completes, rather than after its execution).
    ///
    /// Only meaningful for a hardware-walked chain: under kernel-walked
    /// submission the driver re-kicks the PC per task, so it never runs a
    /// multi-task group for this to describe.
    pub pc_task_fetch: bool,
}

impl Default for PingPong {
    /// Everything on, `AutoToggle` -- the configuration this module exists
    /// to test.
    fn default() -> Self {
        Self {
            pointer_mode: PointerMode::default(),
            executers: true,
            pc_task_fetch: true,
        }
    }
}

impl PingPong {
    /// Today's register programming: no `S_POINTER`, no `PC_TASK_CON`.
    pub fn off() -> Self {
        Self {
            pointer_mode: PointerMode::Off,
            executers: false,
            pc_task_fetch: false,
        }
    }
}

/// How the output rows get divided into tiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tiling {
    /// A fixed number of output rows per tile; the last tile takes the
    /// remainder.
    OutputRows(u32),
    /// Split the output rows into this many balanced tiles -- 256 rows
    /// into 3 gives the design notes' 86/85/85, not 86/86/84.
    Tiles(u32),
}

impl Default for Tiling {
    fn default() -> Self {
        Tiling::Tiles(NPU_CORE_COUNT)
    }
}

/// A convolution decomposed into a sequence of hardware tasks.
#[derive(Clone, Debug, PartialEq)]
pub struct TiledConv {
    /// The whole (untiled) convolution. Every tile's registers are derived
    /// from this plus its own [`ConvTask`] row window, exactly as
    /// `plan_conv_tasks`' CBUF splits are.
    pub shape: ConvShape,
    /// One entry per tile, in dispatch order.
    pub tiles: Vec<ConvTask>,
    pub ping_pong: PingPong,
}

impl TiledConv {
    pub fn task_count(&self) -> usize {
        self.tiles.len()
    }

    /// Cycles for one tile: its output pixels times [`cycles_per_pixel`].
    pub fn tile_cycles(&self, tile: &ConvTask) -> u64 {
        u64::from(self.shape.output_width)
            * u64::from(tile.output_height)
            * cycles_per_pixel(&self.shape)
    }

    /// Cycles for the whole sequence as this module actually dispatches it:
    /// tiles run back-to-back on one core, so they add up. Ping-pong hides
    /// register-fetch latency *between* tiles, which this compute-only
    /// model never charged for in the first place -- so a ping-pong win
    /// shows up as measured wall clock approaching this number, not as a
    /// smaller number here.
    pub fn sequential_cycles(&self) -> u64 {
        self.tiles.iter().map(|tile| self.tile_cycles(tile)).sum()
    }

    /// The design notes' figure for the same tiling: the slowest tile, i.e.
    /// what this conv would cost if the tiles ran on separate cores
    /// concurrently. Included for comparison against
    /// [`Self::sequential_cycles`] -- nothing in this module dispatches
    /// that way.
    pub fn parallel_cycles(&self) -> u64 {
        self.tiles
            .iter()
            .map(|tile| self.tile_cycles(tile))
            .max()
            .unwrap_or(0)
    }

    /// Fraction of the array's MAC capacity the sequence actually uses:
    /// real MACs over array-consumed MACs. The design notes' 2.34% for a
    /// Cin=3/Cout=3 fp16 conv comes out of this ratio -- a small-channel
    /// early layer leaves nearly the whole 16x32 array idle.
    pub fn mac_utilization(&self) -> f64 {
        let pixels: u64 = self
            .tiles
            .iter()
            .map(|tile| u64::from(self.shape.output_width) * u64::from(tile.output_height))
            .sum();
        let consumed =
            self.sequential_cycles() * u64::from(mac_array(self.shape.precision).macs_per_cycle());
        if consumed == 0 {
            return 0.0;
        }
        (pixels * real_macs_per_pixel(&self.shape)) as f64 / consumed as f64
    }
}

/// Plans a row-tiled convolution.
///
/// Requires the same zero-padding valid-convolution geometry
/// `plan_conv_tasks` does (`output = (input - kernel) / stride + 1`) --
/// there are no padding fields in the shape to express anything else. Each
/// tile's input row window is widened to cover its output rows' full
/// receptive field, which for `weights_height > 1` means neighbouring tiles
/// overlap by `weights_height - stride` input rows (the halo the design
/// notes' row tiling calls for; `feature_grains` buffers it on-chip,
/// chapter36.txt:1049-1051).
///
/// Every tile is validated to fit CBUF and to be a shape the register
/// builder will accept, so a plan that comes back `Ok` is dispatchable
/// tile-by-tile without further checks.
pub fn plan_tiled_conv(
    shape: &ConvShape,
    tiling: Tiling,
    ping_pong: PingPong,
) -> Result<TiledConv, &'static str> {
    validate_conv_shape(shape)?;
    if shape.stride == 0 {
        return Err("convolution stride must be nonzero");
    }
    if shape.weights_width > shape.input_width || shape.weights_height > shape.input_height {
        return Err("convolution kernel exceeds the input extent");
    }
    let expected_output_width = (shape.input_width - shape.weights_width) / shape.stride + 1;
    let expected_output_height = (shape.input_height - shape.weights_height) / shape.stride + 1;
    if shape.output_width != expected_output_width || shape.output_height != expected_output_height
    {
        return Err("output shape does not match zero-padding valid-convolution geometry");
    }

    let output_rows = split_output_rows(shape.output_height, tiling)?;

    // Uniform banks across the sequence, sized for the tallest tile: the
    // CBUF weight region has to land at the same bank offset in every tile
    // for a later `reuse_weights` to be even conceivable, and a
    // shorter tile gaining a bank it doesn't need buys nothing.
    let entries_per_slice = conv_entries_per_slice(shape)?;
    if entries_per_slice == 0 {
        return Err("convolution requires zero CBUF entries per input slice");
    }
    let tallest_input_rows = output_rows
        .iter()
        .map(|&rows| input_rows_for_output_rows(shape, rows))
        .max()
        .expect("split_output_rows returns at least one tile");
    let input_banks = entries_per_slice
        .checked_mul(u64::from(tallest_input_rows))
        .ok_or("input bank count overflows tile planning")?
        .div_ceil(CBUF_ENTRIES_PER_BANK);
    if input_banks == 0 || input_banks >= u64::from(CBUF_BANKS) {
        return Err("tile input rows do not fit in CBUF -- use fewer output rows per tile");
    }
    let weights_banks = u64::from(CBUF_BANKS) - input_banks;
    if weights_banks < conv_weights_banks(shape)? {
        return Err("tile leaves too few CBUF banks for this convolution's weights");
    }
    let input_banks = u32::try_from(input_banks).map_err(|_| "input bank count exceeds u32")?;
    let weights_banks =
        u32::try_from(weights_banks).map_err(|_| "weight bank count exceeds u32")?;

    let input_line_stride = u64::from(shape.input_width) * ATOMIC_K_SIZE;
    let output_line_stride = u64::from(shape.output_width) * ATOMIC_K_SIZE;
    let mut tiles: Vec<ConvTask> = Vec::with_capacity(output_rows.len());
    let mut output_top = 0u32;
    let mut previous_input_bottom: Option<u32> = None;
    for (index, output_height) in output_rows.iter().copied().enumerate() {
        let input_top = output_top * shape.stride;
        let input_height = input_rows_for_output_rows(shape, output_height);
        let input_bottom = input_top + input_height - 1;
        if input_bottom >= shape.input_height {
            return Err("tile input row window runs past the input height");
        }

        let overlap_slices = match previous_input_bottom {
            Some(bottom) if bottom >= input_top => bottom - input_top + 1,
            _ => 0,
        };
        if overlap_slices > 0 {
            // The producing tile has to hold those rows for its successor;
            // mirrors `plan_conv_tasks`' own retain/overlap bookkeeping.
            tiles[index - 1].retain_slices = overlap_slices;
        }

        let input_offset = input_line_stride
            .checked_mul(u64::from(input_top))
            .ok_or("tile input offset overflows")?;
        let output_offset = output_line_stride
            .checked_mul(u64::from(output_top))
            .ok_or("tile output offset overflows")?;

        tiles.push(ConvTask {
            index: u32::try_from(index).map_err(|_| "tile count exceeds u32")?,
            input_top,
            input_height,
            output_top,
            output_height,
            input_offset_bytes: u32::try_from(input_offset)
                .map_err(|_| "tile input offset exceeds u32")?,
            output_offset_bytes: u32::try_from(output_offset)
                .map_err(|_| "tile output offset exceeds u32")?,
            input_banks,
            weights_banks,
            overlap_slices,
            retain_slices: 0,
            // Each tile reloads its own weights. `plan_conv_tasks` reached
            // the same conclusion the hard way: enabling Mesa's later-task
            // WEIGHT_REUSE bit on real RK3588 produced correct task 0
            // output followed by all-zero rows. Ping-pong is about
            // register-fetch overlap, not CBUF retention, so nothing here
            // depends on revisiting that.
            reuse_weights: false,
        });

        previous_input_bottom = Some(input_bottom);
        output_top += output_height;
    }

    if output_top != shape.output_height {
        return Err("tiles do not cover the declared output height exactly");
    }

    // Each tile is dispatched as a conv over its own row window, so the
    // per-tile shape -- not just the whole-conv shape validated above --
    // has to be one the register builder accepts (`input_height >= 4`, the
    // 11-bit height fields, and so on).
    for tile in &tiles {
        validate_conv_shape(&tile_shape(shape, tile))?;
    }

    Ok(TiledConv {
        shape: *shape,
        tiles,
        ping_pong,
    })
}

/// The whole-conv shape narrowed to one tile's row window -- what that
/// tile's registers actually describe.
pub fn tile_shape(shape: &ConvShape, tile: &ConvTask) -> ConvShape {
    ConvShape {
        input_height: tile.input_height,
        output_height: tile.output_height,
        ..*shape
    }
}

/// Input rows a tile needs to produce `output_rows` output rows: the first
/// output row's full kernel footprint plus a stride per additional row.
fn input_rows_for_output_rows(shape: &ConvShape, output_rows: u32) -> u32 {
    shape.weights_height + (output_rows - 1) * shape.stride
}

fn split_output_rows(output_height: u32, tiling: Tiling) -> Result<Vec<u32>, &'static str> {
    match tiling {
        Tiling::OutputRows(rows) => {
            if rows == 0 {
                return Err("Tiling::OutputRows requires a nonzero row count");
            }
            let full = output_height / rows;
            let remainder = output_height % rows;
            let mut split = vec![rows; full as usize];
            if remainder > 0 {
                split.push(remainder);
            }
            if split.is_empty() {
                return Err("Tiling::OutputRows produced no tiles");
            }
            Ok(split)
        }
        Tiling::Tiles(count) => {
            if count == 0 {
                return Err("Tiling::Tiles requires a nonzero tile count");
            }
            if count > output_height {
                return Err("Tiling::Tiles asks for more tiles than there are output rows");
            }
            // Balanced, not div_ceil: 256 rows / 3 tiles is 86/85/85 (the
            // design notes' split), not 86/86/84.
            let base = output_height / count;
            let remainder = output_height % count;
            Ok((0..count)
                .map(|index| base + u32::from(index < remainder))
                .collect())
        }
    }
}

/// Emits the `S_POINTER`/`PC_TASK_CON` preamble for one tile.
///
/// Ordering is load-bearing in two ways. `S_POINTER` must precede the
/// shadowed payload registers it selects a group for, so this goes at the
/// very front of the tile's regcmd program. And `PC_TASK_CON.task_number`
/// must be applied before the PC finishes fetching tile 0's registers,
/// since with `task_pp_en` set it starts fetching tile 1 at that moment --
/// putting it in tile 0's own preamble satisfies that.
fn push_pc_task_con(cmds: &mut Vec<RegCmd>, ping_pong: &PingPong, task_count: usize) {
    if !ping_pong.pc_task_fetch {
        return;
    }
    let task_number = u32::try_from(task_count).expect("task count fits u32");
    debug_assert!(
        task_number < (1 << 12),
        "PC_TASK_CON.task_number is 12 bits"
    );
    cmds.push(
        Register::<PCTaskCon>::new()
            .count_clear(true)
            .task_number(Bits::new(task_number))
            .task_pp_enable(true)
            .build(),
    );
}

/// The `S_POINTER` value every kicked block should carry for this tile, or
/// `None` if this tile should carry no `S_POINTER` write at all.
///
/// One value for all four blocks: the TRM documents the same register
/// layout per block (`RKNN_cna_s_pointer` at +0x1004 and its CORE/DPU/
/// DPU_RDMA twins at +0x3004/+0x4004/+0x5004 have identical bit
/// assignments), and `s_pointer_word_matches_every_block`'s round-trip
/// through each block's own typed builder holds that assumption to account.
fn s_pointer_value(ping_pong: &PingPong, task_index: usize) -> Option<u32> {
    let mut reg = Register::<CnaSPointer>::new();
    match ping_pong.pointer_mode {
        PointerMode::AutoToggle if task_index == 0 => {
            reg.pointer_pp_clear(Bits::new(1))
                .executer_pp_clear(Bits::new(1))
                .pointer(Bits::new(0))
                .pointer_pp_en(Bits::new(1))
                // "Pointer ping-pong by pointer": current pointer 0 -> next
                // toggles to 1. The by-executer alternative couples the
                // toggle to executer alternation, which `executers`
                // controls independently.
                .pointer_pp_mode(Bits::new(1))
                .executer_pp_en(Bits::new(u32::from(ping_pong.executers)));
        }
        // Armed by tile 0; a later write would only fight the toggle.
        PointerMode::AutoToggle => return None,
        PointerMode::ExplicitPerTask => {
            reg.pointer(Bits::new((task_index & 1) as u32))
                .pointer_pp_en(Bits::new(0))
                .executer_pp_en(Bits::new(u32::from(ping_pong.executers)));
        }
        PointerMode::Off => return None,
    }
    Some(reg.into_val())
}

fn is_register<R: RegisterMeta>(command: &RegCmd) -> bool {
    ((command.0 >> 48) & 0xffff) == u64::from(R::DOMAIN)
        && (command.0 & 0xffff) == u64::from(R::OFFSET)
}

/// Reconciles the conv payload's own `DPU_S_POINTER`/`DPU_RDMA_S_POINTER`
/// writes with `value`: rewritten in place when this tile carries a value
/// (keeping their position ahead of the DPU's shadowed registers), dropped
/// when it doesn't.
///
/// Dropping matters as much as rewriting. Under
/// [`PointerMode::AutoToggle`], tiles 1..N must not touch `S_POINTER`, and
/// the payload writes `pointer = 0` unconditionally -- leaving those words
/// in place would re-select group 0 on every tile in exactly the two blocks
/// whose group the hardware is supposed to be advancing.
fn apply_payload_s_pointer(payload: &mut Vec<RegCmd>, value: Option<u32>) {
    match value {
        Some(value) => {
            for command in payload.iter_mut() {
                if is_register::<DpuSPointer>(command) {
                    *command = Register::<DpuSPointer>::from_val(value).build();
                } else if is_register::<DpuRdmaSPointer>(command) {
                    *command = Register::<DpuRdmaSPointer>::from_val(value).build();
                }
            }
        }
        None => payload.retain(|command| {
            !is_register::<DpuSPointer>(command) && !is_register::<DpuRdmaSPointer>(command)
        }),
    }
}

/// Builds one regcmd buffer per tile, in dispatch order.
///
/// Each buffer is a complete conv task program for its row window
/// (`build_conv_regcmd`'s emission, with the tile's own input/output byte
/// offsets and CBUF bank split) wrapped in this module's ping-pong
/// preamble and a multi-task kick tail. The tail's trailing
/// `PC_BASE_ADDRESS`/`PC_REGISTER_AMOUNTS` pair is left zeroed; a
/// hardware-walked chain patches it via [`link_tiled_conv_regcmds`] once
/// the command buffers' DMA addresses are known, and a kernel-walked
/// submission leaves it alone.
pub fn build_tiled_conv_regcmds(
    plan: &TiledConv,
    bufs: &ConvBuffers,
) -> Result<Vec<Vec<RegCmd>>, &'static str> {
    if plan.tiles.is_empty() {
        return Err("tiled convolution plan has no tiles");
    }
    for tile in &plan.tiles {
        bufs.input_addr
            .checked_add(tile.input_offset_bytes)
            .ok_or("tile input DMA address exceeds u32")?;
        bufs.output_addr
            .checked_add(tile.output_offset_bytes)
            .ok_or("tile output DMA address exceeds u32")?;
    }

    let task_count = plan.tiles.len();
    Ok(plan
        .tiles
        .iter()
        .enumerate()
        .map(|(index, tile)| {
            let mut payload = build_conv_cna_core_dpu_dpu_rdma(
                &plan.shape,
                bufs,
                tile,
                // DPU output mode 2 -- what every conv path in this crate
                // uses (direct DPU write-back, no PPU stage).
                2,
                None,
            );

            let mut cmds = Vec::with_capacity(payload.len() + 8);
            if index == 0 {
                // `PC_TASK_CON.task_number` has to be applied before the PC
                // finishes fetching tile 0's registers, since with
                // `task_pp_en` set it starts fetching tile 1 at that moment.
                // First word of the first tile is the one place that holds
                // for certain.
                push_pc_task_con(&mut cmds, &plan.ping_pong, task_count);
            }

            let s_pointer = s_pointer_value(&plan.ping_pong, index);
            if !matches!(plan.ping_pong.pointer_mode, PointerMode::Off) {
                // CNA and CORE get theirs prepended -- nothing in the
                // payload writes them -- and the payload's own DPU/DPU_RDMA
                // writes are rewritten or dropped to agree.
                if let Some(value) = s_pointer {
                    cmds.push(Register::<CnaSPointer>::from_val(value).build());
                    cmds.push(Register::<CoreSPointer>::from_val(value).build());
                }
                apply_payload_s_pointer(&mut payload, s_pointer);
            }
            cmds.extend(payload);
            // The multi-task tail even for `task_count == 1`: it costs one
            // regcmd word over the single-task placeholder and keeps the
            // link trailer at a findable offset, so a one-tile plan stays
            // byte-comparable with a multi-tile one.
            push_kick_for_task_count(
                &mut cmds,
                KICK_CNA | KICK_CORE | KICK_DPU | KICK_DPU_RDMA,
                task_count.max(2),
            );
            cmds
        })
        .collect())
}

/// What value a chained tile's `PC_REGISTER_AMOUNTS` link should carry for
/// its successor.
///
/// `drm_rocket_task.regcmd_count` is documented as "number of commands in
/// the register command buffer" (`vendor/linux-headers/drm/rocket_accel.h`),
/// i.e. 64-bit words, and the kernel converts that to the register's own
/// units before writing it. A regcmd-embedded link bypasses the conversion
/// and must carry the already-converted value, so the two are different
/// numbers.
///
/// [`Self::Driver`] is that conversion, read off `rocket_job_hw_submit()`,
/// and is the only one of these with any authority. The other two are the
/// guesses that predated finding it, kept only so the hardware tests that
/// ran against them stay reproducible.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RegisterAmount {
    /// The kernel's own formula, verbatim from `rocket_job_hw_submit()`:
    ///
    /// ```text
    /// rocket_pc_writel(core, REGISTER_AMOUNTS,
    ///     PC_REGISTER_AMOUNTS_PC_DATA_AMOUNT((task->regcmd_count + 1) / 2 - 1));
    /// ```
    ///
    /// So the register counts *pairs* of 64-bit words, less one -- a
    /// 128-word tile is 63, not 128 and not 64. Underflows for a 0-word
    /// task, which cannot occur here (every tile carries a payload).
    #[default]
    Driver,
    /// `regcmd::link_regcmd_tasks`' pre-existing formula, `(words / 2)`
    /// rounded up to even. Inherited from Mesa-era patching; now known to
    /// be off by one against [`Self::Driver`] even where it rounds to the
    /// same pair count.
    MesaHalvedEven,
    /// The successor's raw 64-bit-word count -- the number handed to the
    /// kernel as `regcmd_count`, on the since-disproved theory that the
    /// driver passed it through unscaled.
    KernelWordCount,
}

/// `PC_REGISTER_AMOUNTS.pc_data_amount` for a regcmd of `words` 64-bit
/// commands, as the kernel computes it:
///
/// ```text
/// PC_REGISTER_AMOUNTS_PC_DATA_AMOUNT((task->regcmd_count + 1) / 2 - 1)
/// ```
///
/// Spelled the same way as `rocket_job_hw_submit()` rather than as
/// `words.div_ceil(2) - 1` so it reads identically to the C it is taken
/// from. Panics (debug) / wraps (release) at `words == 0`, matching the
/// kernel's own lack of a guard; every tile here carries a payload.
#[allow(clippy::manual_div_ceil)]
pub fn driver_register_amount(words: u32) -> u32 {
    (words + 1) / 2 - 1
}

/// Patches each tile's trailing PC link to point at the next tile's regcmd
/// buffer, for a hardware-walked chain.
///
/// `task_dma_addresses[i]` is the DMA address of the buffer holding
/// `tasks[i]`. After this returns, the caller submits **only** `tasks[0]`
/// (as a single-task job) and the PC follows the chain; the last tile keeps
/// its zeroed link, which terminates the run.
pub fn link_tiled_conv_regcmds(
    tasks: &mut [Vec<RegCmd>],
    task_dma_addresses: &[u32],
    amount: RegisterAmount,
) -> Result<(), &'static str> {
    match amount {
        // Same trailer format, so the incumbent formula stays in one place
        // rather than being reimplemented here.
        RegisterAmount::MesaHalvedEven => link_regcmd_tasks(tasks, task_dma_addresses),
        RegisterAmount::Driver | RegisterAmount::KernelWordCount => {
            if tasks.len() != task_dma_addresses.len() {
                return Err("regcmd task and DMA-address counts differ");
            }
            if tasks.len() <= 1 {
                return Ok(());
            }
            let trailers = tasks
                .iter()
                .map(|commands| task_link_trailer_index(commands))
                .collect::<Result<Vec<_>, _>>()?;
            let word_counts = tasks
                .iter()
                .map(|commands| {
                    let words = u32::try_from(commands.len())
                        .map_err(|_| "regcmd word count exceeds u32")?;
                    Ok(match amount {
                        RegisterAmount::Driver => driver_register_amount(words),
                        _ => words,
                    })
                })
                .collect::<Result<Vec<_>, &'static str>>()?;
            for index in 0..tasks.len() - 1 {
                let trailer = trailers[index];
                tasks[index][trailer] = RegCmd::new(
                    DOMAIN_PC,
                    REG_PC_BASE_ADDRESS,
                    task_dma_addresses[index + 1],
                );
                tasks[index][trailer + 1] =
                    RegCmd::new(DOMAIN_PC, REG_PC_REGISTER_AMOUNTS, word_counts[index + 1]);
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rocket::{
        builders::{DOMAIN_PC, RegisterMeta, pc::PCTaskCon},
        regcmd::{Activation, CONV_OUTPUT_ATOMIC_STRIDE},
    };

    /// The design notes' worked example, minus its SAME padding: a
    /// 256x256x3 fp16 conv. Valid geometry only (no padding fields exist),
    /// so a 3x3 kernel gives 254x254 output rather than the notes' 256x256
    /// -- the cycle *model* is what these tests check against the notes,
    /// not the row count.
    fn notes_shape(kernel: u32, depthwise: bool) -> ConvShape {
        ConvShape {
            input_width: 256,
            input_height: 256,
            input_channels: 3,
            output_width: 256 - kernel + 1,
            output_height: 256 - kernel + 1,
            output_channels: 3,
            weights_width: kernel,
            weights_height: kernel,
            stride: 1,
            depthwise,
            input_zero_point: 0,
            output_zero_point: 0,
            weights_zero_point: 0,
            input_scale: 1.0,
            weights_scale: 1.0,
            output_scale: 1.0,
            truncate_bits: 0,
            activation: Activation::None,
            precision: Precision::Fp16,
        }
    }

    /// `conv_hw.rs`'s hardware-validated fp16 C32 -> C16 1x1 geometry, the
    /// shape `tests/tiled_conv_hw.rs` tiles.
    fn fp16_c32_to_c16(width: u32, height: u32) -> ConvShape {
        ConvShape {
            input_width: width,
            input_height: height,
            input_channels: 32,
            output_width: width,
            output_height: height,
            output_channels: 16,
            weights_width: 1,
            weights_height: 1,
            stride: 1,
            depthwise: false,
            input_zero_point: 0,
            output_zero_point: 0,
            weights_zero_point: 0,
            input_scale: 1.0,
            weights_scale: 1.0,
            output_scale: 1.0,
            truncate_bits: 0,
            activation: Activation::None,
            precision: Precision::Fp16,
        }
    }

    #[test]
    fn mac_array_matches_the_design_notes_limit_table() {
        assert_eq!(mac_array(Precision::Int8).macs_per_cycle(), 1024);
        assert_eq!(mac_array(Precision::Fp16).macs_per_cycle(), 512);
        assert_eq!(
            mac_array(Precision::Fp16).macs_per_cycle() * NPU_CORE_COUNT,
            512 * 3
        );
    }

    /// The notes' results table: 1x1 dense costs 1 cycle/px, 3x3 dense and
    /// 3x3 depthwise both cost 9, and real MACs/px are 9 / 81 / 27.
    #[test]
    fn cycle_model_matches_the_design_notes_results_table() {
        let one_by_one = notes_shape(1, false);
        assert_eq!(cycles_per_pixel(&one_by_one), 1);
        assert_eq!(real_macs_per_pixel(&one_by_one), 9);

        let three_by_three = notes_shape(3, false);
        assert_eq!(cycles_per_pixel(&three_by_three), 9);
        assert_eq!(real_macs_per_pixel(&three_by_three), 81);

        let depthwise = notes_shape(3, true);
        assert_eq!(cycles_per_pixel(&depthwise), 9);
        assert_eq!(real_macs_per_pixel(&depthwise), 27);
    }

    /// The notes' central takeaway, as an assertion: at Cin=Cout=3 dense
    /// and depthwise cost identical wall clock despite depthwise doing 3x
    /// fewer real MACs, because both are pixel/tap bound rather than
    /// compute bound.
    #[test]
    fn depthwise_and_dense_cost_the_same_wall_clock_at_small_channel_counts() {
        // 60-row tiles, not the notes' 3-way split: a third of 254 rows is
        // 85 input rows, whose 32 CBUF entries per slice need 11 of the 12
        // banks and leave too few for even this tiny weight tensor. The
        // notes' row split is a MAC-throughput argument, not a CBUF one.
        let dense = plan_tiled_conv(
            &notes_shape(3, false),
            Tiling::OutputRows(60),
            PingPong::default(),
        )
        .unwrap();
        let depthwise = plan_tiled_conv(
            &notes_shape(3, true),
            Tiling::OutputRows(60),
            PingPong::default(),
        )
        .unwrap();

        assert_eq!(dense.sequential_cycles(), depthwise.sequential_cycles());
        assert_eq!(dense.parallel_cycles(), depthwise.parallel_cycles());
        assert!(
            depthwise.mac_utilization() < dense.mac_utilization(),
            "depthwise does 3x fewer real MACs in the same cycles, so it must \
             utilize the array less: dense={}, depthwise={}",
            dense.mac_utilization(),
            depthwise.mac_utilization()
        );
    }

    /// The notes' 3-core row split of 256 rows is 86/85/85.
    #[test]
    fn balanced_tiles_match_the_design_notes_row_split() {
        assert_eq!(
            split_output_rows(256, Tiling::Tiles(3)).unwrap(),
            vec![86, 85, 85]
        );
        assert_eq!(
            split_output_rows(112, Tiling::Tiles(3)).unwrap(),
            vec![38, 37, 37]
        );
        assert_eq!(
            split_output_rows(112, Tiling::OutputRows(40)).unwrap(),
            vec![40, 40, 32]
        );
        assert_eq!(
            split_output_rows(120, Tiling::OutputRows(40)).unwrap(),
            vec![40, 40, 40]
        );
    }

    #[test]
    fn tiles_cover_the_output_exactly_with_monotone_offsets() {
        let shape = fp16_c32_to_c16(112, 112);
        let plan = plan_tiled_conv(&shape, Tiling::Tiles(4), PingPong::default()).unwrap();

        assert_eq!(plan.task_count(), 4);
        assert_eq!(
            plan.tiles.iter().map(|t| t.output_height).sum::<u32>(),
            shape.output_height
        );
        let mut expected_output_top = 0;
        for (index, tile) in plan.tiles.iter().enumerate() {
            assert_eq!(tile.index as usize, index);
            assert_eq!(tile.output_top, expected_output_top);
            assert_eq!(
                tile.output_offset_bytes,
                tile.output_top * shape.output_width * CONV_OUTPUT_ATOMIC_STRIDE
            );
            assert_eq!(
                tile.input_offset_bytes,
                tile.input_top * shape.input_width * CONV_OUTPUT_ATOMIC_STRIDE
            );
            // 1x1 stride-1: no halo, so input and output windows coincide.
            assert_eq!(tile.input_height, tile.output_height);
            assert_eq!(tile.overlap_slices, 0);
            assert_eq!(tile.retain_slices, 0);
            expected_output_top += tile.output_height;
        }
    }

    /// A `weights_height > stride` kernel makes neighbouring tiles share
    /// input rows -- the halo the design notes' row tiling calls for.
    #[test]
    fn tiles_overlap_by_the_kernel_halo() {
        let mut shape = fp16_c32_to_c16(64, 64);
        shape.weights_width = 3;
        shape.weights_height = 3;
        shape.output_width = 62;
        shape.output_height = 62;

        let plan = plan_tiled_conv(&shape, Tiling::Tiles(2), PingPong::default()).unwrap();
        assert_eq!(plan.task_count(), 2);
        assert_eq!(plan.tiles[0].output_height, 31);
        assert_eq!(plan.tiles[0].input_height, 33);
        // Tile 1 starts at input row 31 while tile 0 read through 32.
        assert_eq!(plan.tiles[1].input_top, 31);
        assert_eq!(plan.tiles[1].overlap_slices, 2);
        assert_eq!(plan.tiles[0].retain_slices, 2);
        assert_eq!(
            plan.tiles.iter().map(|t| t.output_height).sum::<u32>(),
            shape.output_height
        );
    }

    #[test]
    fn every_tile_gets_the_same_cbuf_bank_split() {
        let plan = plan_tiled_conv(
            &fp16_c32_to_c16(112, 112),
            Tiling::Tiles(3),
            PingPong::default(),
        )
        .unwrap();
        let first = plan.tiles[0];
        assert!(first.input_banks >= 1);
        assert_eq!(first.input_banks + first.weights_banks, CBUF_BANKS);
        assert!(
            plan.tiles
                .iter()
                .all(|t| t.input_banks == first.input_banks
                    && t.weights_banks == first.weights_banks)
        );
    }

    /// `tests/tiled_conv_hw.rs` hardcodes these two plans (and asserts the
    /// row split again on the board). Pinning them here means a broken
    /// premise shows up in a host `cargo test` instead of only after
    /// cross-compiling and copying a binary to the RK3588.
    #[test]
    fn hardware_test_shapes_plan_as_expected() {
        let anchor = plan_tiled_conv(
            &fp16_c32_to_c16(4, 4),
            Tiling::Tiles(1),
            PingPong::default(),
        )
        .unwrap();
        assert_eq!(
            anchor
                .tiles
                .iter()
                .map(|tile| tile.output_height)
                .collect::<Vec<_>>(),
            vec![4]
        );

        let tiled = plan_tiled_conv(
            &fp16_c32_to_c16(112, 112),
            Tiling::Tiles(3),
            PingPong::default(),
        )
        .unwrap();
        assert_eq!(
            tiled
                .tiles
                .iter()
                .map(|tile| tile.output_height)
                .collect::<Vec<_>>(),
            vec![38, 37, 37]
        );
    }

    #[test]
    fn rejects_tiles_that_do_not_fit_cbuf() {
        // One tile for a 112-row C32 fp16 input needs more than the 11
        // input banks CBUF can spare -- `plan_conv_tasks` splits this same
        // shape three ways for exactly that reason.
        let error = plan_tiled_conv(
            &fp16_c32_to_c16(112, 112),
            Tiling::Tiles(1),
            PingPong::default(),
        )
        .unwrap_err();
        assert!(error.contains("CBUF"), "unexpected error: {error}");
    }

    #[test]
    fn rejects_degenerate_tilings() {
        let shape = fp16_c32_to_c16(112, 112);
        assert!(plan_tiled_conv(&shape, Tiling::Tiles(0), PingPong::default()).is_err());
        assert!(plan_tiled_conv(&shape, Tiling::OutputRows(0), PingPong::default()).is_err());
        assert!(plan_tiled_conv(&shape, Tiling::Tiles(200), PingPong::default()).is_err());
        // Tiles shorter than 4 input rows underflow the register builder's
        // input_surface_stride formula -- caught by the per-tile
        // `validate_conv_shape`, not left to the hardware.
        assert!(plan_tiled_conv(&shape, Tiling::OutputRows(2), PingPong::default()).is_err());
    }

    fn value_of(command: &RegCmd) -> u32 {
        ((command.0 >> 16) & 0xffff_ffff) as u32
    }

    /// Every write to `R` in `commands`, in order -- plural because the conv
    /// payload writes some registers more than once (`CNA_CONV_CON1` twice,
    /// faithfully to Mesa), and a test asserting "this register says X" has
    /// to mean every write of it, not just the first.
    fn writes<R: RegisterMeta>(commands: &[RegCmd]) -> Vec<u32> {
        commands
            .iter()
            .filter(|command| is_register::<R>(command))
            .map(value_of)
            .collect()
    }

    fn find<R: RegisterMeta>(commands: &[RegCmd]) -> Option<u32> {
        writes::<R>(commands).first().copied()
    }

    /// The four blocks' `S_POINTER` registers must really have the
    /// identical bit layout `s_pointer_value` assumes when it builds one
    /// value through CNA's setters and replays it into the others.
    #[test]
    fn s_pointer_word_matches_every_block() {
        let expected = Register::<CnaSPointer>::new()
            .pointer(Bits::new(1))
            .pointer_pp_en(Bits::new(1))
            .executer_pp_en(Bits::new(1))
            .pointer_pp_mode(Bits::new(1))
            .pointer_pp_clear(Bits::new(1))
            .executer_pp_clear(Bits::new(1))
            .into_val();
        assert_eq!(expected, 0x3f);
        assert_eq!(
            Register::<CoreSPointer>::new()
                .pointer(Bits::new(1))
                .pointer_pp_en(Bits::new(1))
                .executer_pp_en(Bits::new(1))
                .pointer_pp_mode(Bits::new(1))
                .pointer_pp_clear(Bits::new(1))
                .executer_pp_clear(Bits::new(1))
                .into_val(),
            expected
        );
        assert_eq!(
            Register::<DpuSPointer>::new()
                .pointer(Bits::new(1))
                .pointer_pp_en(Bits::new(1))
                .executer_pp_en(Bits::new(1))
                .pointer_pp_mode(Bits::new(1))
                .pointer_pp_clear(Bits::new(1))
                .executer_pp_clear(Bits::new(1))
                .into_val(),
            expected
        );
        assert_eq!(
            Register::<DpuRdmaSPointer>::new()
                .pointer(Bits::new(1))
                .pointer_pp_en(Bits::new(1))
                .executer_pp_en(Bits::new(1))
                .pointer_pp_mode(Bits::new(1))
                .pointer_pp_clear(Bits::new(1))
                .executer_pp_clear(Bits::new(1))
                .into_val(),
            expected
        );
    }

    fn buffers() -> ConvBuffers {
        ConvBuffers {
            input_addr: 0x1000,
            weights_addr: 0x2000,
            bias_addr: 0x3000,
            output_addr: 0x4000,
        }
    }

    #[test]
    fn auto_toggle_arms_ping_pong_on_the_first_tile_only() {
        let plan = plan_tiled_conv(
            &fp16_c32_to_c16(112, 112),
            Tiling::Tiles(3),
            PingPong::default(),
        )
        .unwrap();
        let tasks = build_tiled_conv_regcmds(&plan, &buffers()).unwrap();
        assert_eq!(tasks.len(), 3);

        // Tile 0, all four blocks: pointer group 0, both pp_clear pulses,
        // pointer_pp_en, pointer_pp_mode=1, executer_pp_en -- bits
        // 5|4|3|2|1 set, bit 0 (pointer) clear.
        let expected = (1 << 5) | (1 << 4) | (1 << 3) | (1 << 2) | (1 << 1);
        assert_eq!(writes::<CnaSPointer>(&tasks[0]), vec![expected]);
        assert_eq!(writes::<CoreSPointer>(&tasks[0]), vec![expected]);
        assert_eq!(writes::<DpuSPointer>(&tasks[0]), vec![expected]);
        assert_eq!(writes::<DpuRdmaSPointer>(&tasks[0]), vec![expected]);
        assert_eq!(
            tasks[0][0].0,
            Register::<PCTaskCon>::new()
                .count_clear(true)
                .task_number(Bits::new(3))
                .task_pp_enable(true)
                .build()
                .0,
            "PC_TASK_CON must be the very first word so task_number is applied \
             before the PC starts fetching tile 1"
        );

        // Later tiles must not touch S_POINTER at all -- including the
        // payload's own DPU/DPU_RDMA writes, which would otherwise re-force
        // pointer 0 on every tile.
        for task in &tasks[1..] {
            assert_eq!(writes::<CnaSPointer>(task), Vec::<u32>::new());
            assert_eq!(writes::<CoreSPointer>(task), Vec::<u32>::new());
            assert_eq!(writes::<DpuSPointer>(task), Vec::<u32>::new());
            assert_eq!(writes::<DpuRdmaSPointer>(task), Vec::<u32>::new());
            assert_eq!(find::<PCTaskCon>(task), None);
        }
    }

    #[test]
    fn explicit_pointer_mode_alternates_groups_per_tile() {
        let plan = plan_tiled_conv(
            &fp16_c32_to_c16(112, 112),
            Tiling::Tiles(3),
            PingPong {
                pointer_mode: PointerMode::ExplicitPerTask,
                executers: false,
                pc_task_fetch: false,
            },
        )
        .unwrap();
        let tasks = build_tiled_conv_regcmds(&plan, &buffers()).unwrap();

        for (index, task) in tasks.iter().enumerate() {
            // pointer = index & 1, everything else clear (`executers:
            // false` here, so not even executer_pp_en).
            let expected = vec![(index & 1) as u32];
            assert_eq!(writes::<CnaSPointer>(task), expected);
            assert_eq!(writes::<CoreSPointer>(task), expected);
            assert_eq!(writes::<DpuSPointer>(task), expected);
            assert_eq!(writes::<DpuRdmaSPointer>(task), expected);
            assert_eq!(find::<PCTaskCon>(task), None);
        }
    }

    /// The control case must be register-identical to today's programming:
    /// CNA/CORE untouched, DPU/DPU_RDMA carrying exactly the payload's own
    /// armed value (`pointer_pp_en | executer_pp_en | pointer_pp_mode`), no
    /// `PC_TASK_CON`.
    #[test]
    fn ping_pong_off_leaves_the_payload_exactly_as_it_is() {
        let shape = fp16_c32_to_c16(112, 112);
        let bufs = buffers();
        let plan = plan_tiled_conv(&shape, Tiling::Tiles(3), PingPong::off()).unwrap();
        let tasks = build_tiled_conv_regcmds(&plan, &bufs).unwrap();

        let payload_value = (1 << 3) | (1 << 2) | (1 << 1);
        for (tile, task) in plan.tiles.iter().zip(&tasks) {
            assert_eq!(writes::<CnaSPointer>(task), Vec::<u32>::new());
            assert_eq!(writes::<CoreSPointer>(task), Vec::<u32>::new());
            assert_eq!(writes::<DpuSPointer>(task), vec![payload_value]);
            assert_eq!(writes::<DpuRdmaSPointer>(task), vec![payload_value]);
            assert_eq!(find::<PCTaskCon>(task), None);

            // Nothing added, nothing removed: the tile's words are the
            // payload's words plus the kick tail.
            let payload = crate::rocket::regcmd::build_conv_cna_core_dpu_dpu_rdma(
                &shape, &bufs, tile, 2, None,
            );
            assert_eq!(
                task[..payload.len()]
                    .iter()
                    .map(|command| command.0)
                    .collect::<Vec<_>>(),
                payload.iter().map(|command| command.0).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn tiles_program_their_own_row_window_and_byte_offsets() {
        let shape = fp16_c32_to_c16(112, 112);
        let bufs = buffers();
        let plan = plan_tiled_conv(&shape, Tiling::Tiles(3), PingPong::default()).unwrap();
        let tasks = build_tiled_conv_regcmds(&plan, &bufs).unwrap();

        for (tile, commands) in plan.tiles.iter().zip(&tasks) {
            let single = crate::rocket::regcmd::build_conv_cna_core_dpu_dpu_rdma(
                &shape, &bufs, tile, 2, None,
            );
            // Every payload word must survive into the tiled build except
            // the two S_POINTER words, which ping-pong deliberately
            // rewrites or drops (covered by the S_POINTER tests above).
            for expected in single.iter().filter(|command| {
                !is_register::<DpuSPointer>(command) && !is_register::<DpuRdmaSPointer>(command)
            }) {
                assert!(
                    commands.iter().any(|command| command.0 == expected.0),
                    "tile {} dropped payload regcmd {:#018x}",
                    tile.index,
                    expected.0
                );
            }
        }
    }

    /// Independent check of the kernel's amount formula against numbers
    /// worked by hand, so the helper isn't only ever compared to itself:
    /// this module's tiles are 128 words (and tile 0 is 134), which the
    /// register wants as pair-counts-less-one, not word counts.
    #[test]
    fn driver_register_amount_counts_word_pairs_less_one() {
        assert_eq!(driver_register_amount(128), 63);
        assert_eq!(driver_register_amount(134), 66);
        assert_eq!(driver_register_amount(2), 0);
        // The value this supersedes, for contrast: `link_regcmd_tasks`'
        // formula on a 124-word payload yields 62, one short of the 63 a
        // 128-word task really needs.
        assert_eq!((124 / 2u32).next_multiple_of(2), 62);
    }

    /// A hardware-walked chain needs every tile but the last pointing at
    /// its successor's command buffer, under either amount convention.
    #[test]
    fn linking_chains_each_tile_to_its_successor() {
        let addresses = [0x1000_0000u32, 0x1000_1000, 0x1000_2000];
        for amount in [
            RegisterAmount::Driver,
            RegisterAmount::MesaHalvedEven,
            RegisterAmount::KernelWordCount,
        ] {
            let plan = plan_tiled_conv(
                &fp16_c32_to_c16(112, 112),
                Tiling::Tiles(3),
                PingPong::default(),
            )
            .unwrap();
            let mut tasks = build_tiled_conv_regcmds(&plan, &buffers()).unwrap();
            let trailers = tasks
                .iter()
                .map(|task| crate::rocket::regcmd::task_link_trailer_index(task).unwrap())
                .collect::<Vec<_>>();
            let word_counts = tasks.iter().map(|task| task.len()).collect::<Vec<_>>();

            link_tiled_conv_regcmds(&mut tasks, &addresses, amount).unwrap();

            for (index, task) in tasks.iter().enumerate() {
                let trailer = trailers[index];
                let expected_address = addresses.get(index + 1).copied().unwrap_or(0);
                assert_eq!(
                    task[trailer].0,
                    RegCmd::new(DOMAIN_PC, REG_PC_BASE_ADDRESS, expected_address).0,
                    "{amount:?}: tile {index} links to the wrong successor"
                );
                let expected_amount = if index + 1 == tasks.len() {
                    0
                } else {
                    match amount {
                        RegisterAmount::Driver => {
                            driver_register_amount(word_counts[index + 1] as u32)
                        }
                        RegisterAmount::MesaHalvedEven => {
                            (trailers[index + 1] / 2).next_multiple_of(2) as u32
                        }
                        RegisterAmount::KernelWordCount => word_counts[index + 1] as u32,
                    }
                };
                assert_eq!(
                    task[trailer + 1].0,
                    RegCmd::new(DOMAIN_PC, REG_PC_REGISTER_AMOUNTS, expected_amount).0,
                    "{amount:?}: tile {index} carries the wrong successor amount"
                );
            }
        }
    }

    #[test]
    fn rejects_mismatched_link_address_counts() {
        let plan = plan_tiled_conv(
            &fp16_c32_to_c16(112, 112),
            Tiling::Tiles(3),
            PingPong::default(),
        )
        .unwrap();
        let mut tasks = build_tiled_conv_regcmds(&plan, &buffers()).unwrap();
        for amount in [
            RegisterAmount::Driver,
            RegisterAmount::MesaHalvedEven,
            RegisterAmount::KernelWordCount,
        ] {
            assert!(link_tiled_conv_regcmds(&mut tasks, &[0x1000_0000], amount).is_err());
        }
    }
}
