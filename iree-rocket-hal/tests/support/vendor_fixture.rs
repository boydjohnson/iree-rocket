//! Shared vendor-fixture comparison support for `conv_vendor_fixture*.rs`.
//!
//! Vendor RKNN plans repeat the same row/column tile once per output-channel
//! group when `Cout` needs more than one CNA task's worth of kernels.
//! `normalize_row_tiles` collapses those repeats down to one representative
//! program per tile so the spatial fields can be checked against `ConvPlan`,
//! which does not model output-channel tiling yet. That collapse used to be
//! the end of the story: nothing checked that the discarded repeats were
//! actually well-formed. `output_channel_groups` is the other half -- it
//! looks at every program in a plan, not just the row-tile representative,
//! and verifies that the repeats form a consistent output-channel grouping.
//!
//! Two facts distinguish a group repeat from a distinct row tile, and both
//! are enforced here:
//!   - every row tile in a plan repeats the *same* sequence of per-group
//!     kernel counts (`CNA_WEIGHT_SIZE2.WEIGHT_KERNELS`), and those counts
//!     sum to `Cout`;
//!   - each group's destination offset steps from the previous group's by
//!     the *full* output surface (`out_height * out_width`, not the tile's
//!     own row extent) times that group's kernel count times the element
//!     size. Destination planes are channel-group-major over the whole
//!     feature map, not tile-major.
//!
//! Both held on every repeated row tile in the checked-in corpus (357 fp16
//! and 338 int8 group instances) when this was written.
//!
//! This module is pulled into three separate test binaries via `#[path]`,
//! and not every binary uses every function here -- `dead_code` cannot see
//! across that boundary, hence the blanket allow.

#![allow(dead_code)]

use std::collections::BTreeMap;

use iree_rocket_hal::rocket::{
    builders::{RegCmd, RegisterMeta, cna::CnaCbufCon1},
    conv::ConvPlan,
};
use serde::Deserialize;

#[derive(Deserialize)]
pub struct FixtureFile {
    pub schema: u32,
    pub cases: Vec<Case>,
}

#[derive(Deserialize)]
pub struct Case {
    pub model: String,
    pub shape: ShapeFixture,
    pub vendor_programs: Vec<Program>,
}

#[derive(Deserialize)]
pub struct ShapeFixture {
    pub width: u32,
    pub height: u32,
    pub cin: u32,
    pub cout: u32,
    pub kernel_h: u32,
    pub kernel_w: u32,
    pub pad_h: u32,
    pub pad_w: u32,
    pub stride: u32,
}

#[derive(Deserialize)]
pub struct Program {
    pub plan_index: u32,
    pub in_width: u32,
    pub in_rows: u32,
    pub out_width: u32,
    pub out_atomics: u32,
    pub out_rows: u32,
    pub in_offset: u32,
    pub out_offset: u32,
    pub feature_grains: u32,
    pub kernels: u32,
    pub cbuf_data_banks: u32,
    pub cbuf_weight_banks: u32,
    pub cbuf_data_entries: u32,
}

pub fn output_width(shape: &ShapeFixture) -> u32 {
    (shape.width + 2 * shape.pad_w - shape.kernel_w) / shape.stride + 1
}

pub fn output_height(shape: &ShapeFixture) -> u32 {
    (shape.height + 2 * shape.pad_h - shape.kernel_h) / shape.stride + 1
}

pub fn register_value<R: RegisterMeta>(program: &[RegCmd]) -> Option<u32> {
    program
        .iter()
        .find(|command| command.0 as u32 & 0xffff == R::OFFSET)
        .map(|command| ((command.0 >> 16) & 0xffff_ffff) as u32)
}

type RowTileKey = (u32, u32, u32, u32, u32, u32);

fn row_tile_key(program: &Program) -> RowTileKey {
    (
        program.in_width,
        program.in_rows,
        program.in_offset,
        program.out_width,
        program.out_rows,
        program.feature_grains,
    )
}

/// Groups programs that share row/column tile geometry, each group sorted
/// into vendor output-channel order by `out_offset`. A group of length > 1
/// is one row tile repeated once per output-channel group; ordinary
/// single-group plans always come back with every group length 1.
pub fn group_by_row_tile<'a>(programs: &[&'a Program]) -> BTreeMap<RowTileKey, Vec<&'a Program>> {
    let mut grouped: BTreeMap<RowTileKey, Vec<&'a Program>> = BTreeMap::new();
    for &program in programs {
        grouped
            .entry(row_tile_key(program))
            .or_default()
            .push(program);
    }
    for group in grouped.values_mut() {
        group.sort_by_key(|program| program.out_offset);
    }
    grouped
}

/// One representative program per row tile, for the spatial-only comparison
/// against `ConvPlan`. Discards the output-channel-group axis entirely; use
/// `output_channel_groups` alongside this to check what it discards.
pub fn normalize_row_tiles<'a>(programs: &[&'a Program]) -> Vec<&'a Program> {
    group_by_row_tile(programs)
        .into_values()
        .map(|group| group[0])
        .collect()
}

// Every field here is read through the `Debug` derive at diagnostic
// print sites, which `dead_code` does not credit as a read.
#[allow(dead_code)]
#[derive(Debug)]
pub enum GroupError {
    NoRowTiles,
    InconsistentGroupCount {
        counts: BTreeMap<RowTileKey, usize>,
    },
    InconsistentGroupComposition {
        row_tile: RowTileKey,
    },
    GroupsDontCoverProgrammedKernels {
        sum: u32,
        expected: u32,
    },
    BadGroupStride {
        row_tile: RowTileKey,
        index: usize,
        expected: u32,
        actual: u32,
    },
}

/// Verifies that every row tile in `programs` repeats the same sequence of
/// output-channel groups, that the groups' kernel counts sum to
/// `expected_kernels` (`Shape::programmed_kernels`, which pads `Cout` to
/// even for int8 -- the raw `Cout` undercounts by one whenever it is odd),
/// and that each group's destination offset steps by the full output
/// surface rather than by the tile's own row extent. Returns the shared
/// per-group kernel counts (length 1 for an ordinary single-group plan) on
/// success.
pub fn output_channel_groups(
    programs: &[&Program],
    shape: &ShapeFixture,
    expected_kernels: u32,
    elem_bytes: u32,
) -> Result<Vec<u32>, GroupError> {
    let grouped = group_by_row_tile(programs);
    let mut tiles = grouped.iter();
    let Some((&first_key, first_group)) = tiles.next() else {
        return Err(GroupError::NoRowTiles);
    };
    let canonical: Vec<u32> = first_group.iter().map(|program| program.kernels).collect();

    let counts: BTreeMap<RowTileKey, usize> = grouped
        .iter()
        .map(|(&key, group)| (key, group.len()))
        .collect();
    if counts.values().any(|&count| count != canonical.len()) {
        return Err(GroupError::InconsistentGroupCount { counts });
    }

    let out_height = output_height(shape);
    let out_width = output_width(shape);
    for (key, group) in
        std::iter::once((first_key, first_group)).chain(tiles.map(|(&key, group)| (key, group)))
    {
        if group
            .iter()
            .map(|program| program.kernels)
            .collect::<Vec<_>>()
            != canonical
        {
            return Err(GroupError::InconsistentGroupComposition { row_tile: key });
        }
        for index in 1..group.len() {
            let expected = group[index - 1].out_offset
                + out_height * out_width * group[index - 1].kernels * elem_bytes;
            if group[index].out_offset != expected {
                return Err(GroupError::BadGroupStride {
                    row_tile: key,
                    index,
                    expected,
                    actual: group[index].out_offset,
                });
            }
        }
    }

    let sum: u32 = canonical.iter().sum();
    if sum != expected_kernels {
        return Err(GroupError::GroupsDontCoverProgrammedKernels {
            sum,
            expected: expected_kernels,
        });
    }
    Ok(canonical)
}

pub fn complete_row_plan(programs: &[&Program], shape: &ShapeFixture) -> bool {
    let out_width = output_width(shape);
    if programs.is_empty()
        || programs
            .iter()
            .any(|program| program.out_width != out_width || program.in_width != shape.width)
    {
        return false;
    }
    let mut sorted = programs.to_vec();
    sorted.sort_by_key(|program| program.out_offset);
    let mut covered_rows = 0;
    for program in sorted {
        if program.out_offset != covered_rows * out_width * 16 {
            return false;
        }
        covered_rows += program.out_rows;
    }
    covered_rows == output_height(shape)
}

pub fn exact_plan_match(programs: &[&Program], shape: &ShapeFixture, plan: &ConvPlan) -> bool {
    if programs.len() != plan.tiles().len()
        || programs.iter().any(|program| {
            program.cbuf_data_banks != plan.data_banks()
                || program.cbuf_weight_banks != plan.weight_banks()
        })
    {
        return false;
    }
    let out_width = output_width(shape);
    let mut vendor = programs.to_vec();
    vendor.sort_by_key(|program| program.out_offset);
    let generated = plan.programs();
    vendor
        .iter()
        .zip(plan.tiles())
        .zip(generated.iter())
        .all(|((program, tile), commands)| {
            program.in_width == tile.columns.in_cols
                && program.out_width == tile.columns.out_cols
                && program.out_rows == tile.rows.out_rows
                && program.in_rows == tile.rows.in_rows
                && program.out_offset
                    == tile.rows.out_first * out_width * 16 + tile.columns.out_first * 16
                && program.out_atomics == program.out_width * program.out_rows
                && register_value::<CnaCbufCon1>(commands) == Some(program.cbuf_data_entries)
        })
}

pub fn decoded_prefix_matches(
    programs: &[&Program],
    shape: &ShapeFixture,
    plan: &ConvPlan,
) -> bool {
    if programs.is_empty()
        || programs.len() >= plan.tiles().len()
        || programs.iter().any(|program| {
            program.cbuf_data_banks != plan.data_banks()
                || program.cbuf_weight_banks != plan.weight_banks()
        })
    {
        return false;
    }
    let out_width = output_width(shape);
    let mut vendor = programs.to_vec();
    vendor.sort_by_key(|program| program.out_offset);
    let generated = plan.programs();
    vendor
        .iter()
        .zip(plan.tiles())
        .zip(generated.iter())
        .all(|((program, tile), commands)| {
            program.in_width == tile.columns.in_cols
                && program.out_width == tile.columns.out_cols
                && program.out_rows == tile.rows.out_rows
                && program.in_rows == tile.rows.in_rows
                && program.out_offset
                    == tile.rows.out_first * out_width * 16 + tile.columns.out_first * 16
                && register_value::<CnaCbufCon1>(commands) == Some(program.cbuf_data_entries)
        })
}
