//! Dataset loader. Reads every Parquet partition per table into owned Vecs.
//! Keeps the whole dataset in memory because per-seed datasets are small
//! (tens of MBs); swap for streaming iterators if runs grow large.

use super::{BehaviorIntervalRow, OrganismLifetimeRow, TickSummaryRow};
use anyhow::{Context, Result};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::Deserialize;
use std::ffi::OsStr;
use std::fs::{self, File};
use std::path::Path;

pub struct DatasetReader {
    #[allow(dead_code)]
    pub tick_summary: Vec<TickSummaryRow>,
    pub behavior_intervals: Vec<BehaviorIntervalRow>,
    #[allow(dead_code)]
    pub organism_lifetimes: Vec<OrganismLifetimeRow>,
}

impl DatasetReader {
    pub fn load(root: &Path) -> Result<Self> {
        Ok(Self {
            tick_summary: load_table(root, "tick_summary")?,
            behavior_intervals: load_table(root, "behavior_intervals")?,
            organism_lifetimes: load_table(root, "organism_lifetimes")?,
        })
    }
}

fn load_table<T>(root: &Path, table: &str) -> Result<Vec<T>>
where
    T: for<'de> Deserialize<'de>,
{
    let dir = root.join(table);
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut files = Vec::new();
    for entry in fs::read_dir(&dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension() == Some(OsStr::new("parquet")) {
            files.push(path);
        }
    }
    files.sort();

    let mut rows = Vec::new();
    for path in files {
        let file = File::open(&path).with_context(|| format!("opening {}", path.display()))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .with_context(|| format!("reading parquet header {}", path.display()))?;
        let reader = builder
            .build()
            .with_context(|| format!("building reader {}", path.display()))?;
        for batch in reader {
            let batch = batch.with_context(|| format!("decoding batch in {}", path.display()))?;
            let mut batch_rows: Vec<T> = serde_arrow::from_record_batch(&batch)
                .with_context(|| format!("deserializing {}", path.display()))?;
            rows.append(&mut batch_rows);
        }
    }
    Ok(rows)
}
