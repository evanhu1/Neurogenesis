//! Partitioned Parquet writer. Buffers rows per table in memory and flushes
//! one file per table per interval to enable tailing — each closed file is
//! immediately readable by DuckDB/polars/pandas via a glob of the table dir.

use super::schema::{
    ActionCountRow, GenomeSnapshotIndexRow, OrganismLifetimeRow, PopulationSnapshotRow,
    ReproductionEventRow, TickSummaryRow,
};
use anyhow::{Context, Result};
use arrow::array::RecordBatch;
use arrow::datatypes::FieldRef;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use serde::Serialize;
use serde_arrow::schema::{SchemaLike, TracingOptions};
use sim_types::OrganismGenome;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

/// Minimal write-side abstraction. The orchestration loop only sees this
/// trait; the in-process Parquet impl is the sole concrete implementor today,
/// but this boundary keeps orchestration testable and lets us swap in a
/// JSONL or in-memory sink later without changing emission code.
pub trait DatasetWriter {
    fn emit_tick(&mut self, row: TickSummaryRow);
    fn emit_population_snapshot(&mut self, row: PopulationSnapshotRow);
    fn emit_action_count(&mut self, row: ActionCountRow);
    fn emit_organism_lifetime(&mut self, row: OrganismLifetimeRow);
    fn emit_reproduction_event(&mut self, row: ReproductionEventRow);
    /// Serialize `genome` to `genomes/t{tick:06}.bin` relative to the dataset
    /// root and enqueue an index row pointing at it. Returns the relative
    /// file path stored in the index so tests and callers can verify.
    fn emit_genome_snapshot(
        &mut self,
        tick: u64,
        organism_id: u64,
        species_id: u64,
        generation: u64,
        num_offspring: u32,
        genome: &OrganismGenome,
    ) -> Result<String>;

    /// Write every buffered table to its partition directory and clear the
    /// buffers. Called at each reporting interval so tailers can observe
    /// progress.
    fn flush(&mut self) -> Result<()>;

    /// Final flush. Consumes self so the writer can't be reused.
    fn finalize(self: Box<Self>) -> Result<()>;
}

pub struct PartitionedParquetWriter {
    root: PathBuf,
    partition_index: u32,
    next_snapshot_id: u64,
    tick_summary: TableBuffer<TickSummaryRow>,
    population_snapshots: TableBuffer<PopulationSnapshotRow>,
    action_counts: TableBuffer<ActionCountRow>,
    organism_lifetimes: TableBuffer<OrganismLifetimeRow>,
    reproduction_events: TableBuffer<ReproductionEventRow>,
    genome_snapshots: TableBuffer<GenomeSnapshotIndexRow>,
}

const TABLES: &[&str] = &[
    "tick_summary",
    "population_snapshots",
    "action_counts",
    "organism_lifetimes",
    "reproduction_events",
    "genome_snapshots",
];

const GENOMES_DIR: &str = "genomes";

impl PartitionedParquetWriter {
    pub fn new(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        fs::create_dir_all(&root)?;
        for table in TABLES {
            fs::create_dir_all(root.join(table))?;
        }
        fs::create_dir_all(root.join(GENOMES_DIR))?;
        Ok(Self {
            root,
            partition_index: 0,
            next_snapshot_id: 0,
            tick_summary: TableBuffer::new("tick_summary"),
            population_snapshots: TableBuffer::new("population_snapshots"),
            action_counts: TableBuffer::new("action_counts"),
            organism_lifetimes: TableBuffer::new("organism_lifetimes"),
            reproduction_events: TableBuffer::new("reproduction_events"),
            genome_snapshots: TableBuffer::new("genome_snapshots"),
        })
    }
}

impl DatasetWriter for PartitionedParquetWriter {
    fn emit_tick(&mut self, row: TickSummaryRow) {
        self.tick_summary.push(row);
    }
    fn emit_population_snapshot(&mut self, row: PopulationSnapshotRow) {
        self.population_snapshots.push(row);
    }
    fn emit_action_count(&mut self, row: ActionCountRow) {
        self.action_counts.push(row);
    }
    fn emit_organism_lifetime(&mut self, row: OrganismLifetimeRow) {
        self.organism_lifetimes.push(row);
    }
    fn emit_reproduction_event(&mut self, row: ReproductionEventRow) {
        self.reproduction_events.push(row);
    }

    fn emit_genome_snapshot(
        &mut self,
        tick: u64,
        organism_id: u64,
        species_id: u64,
        generation: u64,
        num_offspring: u32,
        genome: &OrganismGenome,
    ) -> Result<String> {
        let file_name = format!("t{:06}.bin", tick);
        let rel_path = format!("{GENOMES_DIR}/{file_name}");
        let abs_path = self.root.join(&rel_path);
        atomic_write(&abs_path, |file| {
            let mut writer = BufWriter::new(file);
            bincode::serialize_into(&mut writer, genome)
                .with_context(|| format!("serializing genome to {}", abs_path.display()))?;
            writer.flush()?;
            Ok(())
        })?;

        let snapshot_id = self.next_snapshot_id;
        self.next_snapshot_id += 1;
        self.genome_snapshots.push(GenomeSnapshotIndexRow {
            snapshot_id,
            tick,
            organism_id,
            species_id,
            generation,
            num_offspring,
            file_path: rel_path.clone(),
        });
        Ok(rel_path)
    }

    fn flush(&mut self) -> Result<()> {
        let part = self.partition_index;
        self.tick_summary.flush(&self.root, part)?;
        self.population_snapshots.flush(&self.root, part)?;
        self.action_counts.flush(&self.root, part)?;
        self.organism_lifetimes.flush(&self.root, part)?;
        self.reproduction_events.flush(&self.root, part)?;
        self.genome_snapshots.flush(&self.root, part)?;
        self.partition_index += 1;
        Ok(())
    }

    fn finalize(mut self: Box<Self>) -> Result<()> {
        self.flush()
    }
}

struct TableBuffer<T> {
    table: &'static str,
    rows: Vec<T>,
    /// Arrow field definitions derived from `T`'s serde shape — computed
    /// eagerly on construction rather than on every flush, since the shape
    /// doesn't change for the lifetime of the writer.
    fields: Vec<FieldRef>,
}

impl<T> TableBuffer<T>
where
    T: Serialize + for<'de> serde::Deserialize<'de> + 'static,
{
    fn new(table: &'static str) -> Self {
        let fields = Vec::<FieldRef>::from_type::<T>(
            TracingOptions::default().allow_null_fields(true),
        )
        .unwrap_or_else(|err| panic!("deriving schema for table {table}: {err}"));
        Self {
            table,
            rows: Vec::new(),
            fields,
        }
    }

    fn push(&mut self, row: T) {
        self.rows.push(row);
    }

    /// Writes the buffer to `{root}/{table}/part_{index:06}.parquet` and
    /// clears it. Skipped entirely if the buffer is empty, so partition
    /// numbering stays contiguous but empty intervals don't produce empty
    /// files.
    fn flush(&mut self, root: &Path, partition_index: u32) -> Result<()> {
        if self.rows.is_empty() {
            return Ok(());
        }
        let filename = format!("part_{:06}.parquet", partition_index);
        let dir = root.join(self.table);
        let final_path = dir.join(&filename);

        let batch: RecordBatch = serde_arrow::to_record_batch(&self.fields, &self.rows)
            .with_context(|| format!("serializing {} rows", self.table))?;
        let schema = batch.schema();

        atomic_write(&final_path, |file| {
            let props = WriterProperties::builder()
                .set_compression(Compression::SNAPPY)
                .build();
            let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
            writer.write(&batch)?;
            writer.close()?;
            Ok(())
        })?;
        self.rows.clear();
        Ok(())
    }
}

/// Write-then-rename: write into `<final>.tmp` and rename on success so a
/// concurrent reader globbing the table directory never sees a partial file.
fn atomic_write<F>(final_path: &Path, write: F) -> Result<()>
where
    F: FnOnce(File) -> Result<()>,
{
    let tmp_path = final_path.with_extension(match final_path.extension() {
        Some(ext) => format!("{}.tmp", ext.to_string_lossy()),
        None => "tmp".to_owned(),
    });
    let file = File::create(&tmp_path)
        .with_context(|| format!("creating {}", tmp_path.display()))?;
    write(file)?;
    fs::rename(&tmp_path, final_path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::reader::DatasetReader;
    use super::super::schema::{
        ActionCountRow, OrganismLifetimeRow, PopulationSnapshotRow, ReproductionEventRow,
        ReproductionOutcome, TickSummaryRow, JOINT_LEN,
    };
    use super::*;
    use std::env;

    fn tmp_dir(tag: &str) -> PathBuf {
        let dir = env::temp_dir().join(format!(
            "sim-evaluation-dataset-test-{}-{}-{}",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn tick_summary_round_trip_across_partitions() {
        let dir = tmp_dir("tick");
        let mut writer = PartitionedParquetWriter::new(&dir).unwrap();

        let batch_one = vec![
            TickSummaryRow {
                tick: 1,
                population: 300,
                max_generation: Some(1),
                births: 2,
                deaths: 1,
                food_count: 800,
                consumptions: 5,
                predations: 0,
                food_spawned: 6,
            },
            TickSummaryRow {
                tick: 2,
                population: 301,
                max_generation: Some(1),
                births: 1,
                deaths: 0,
                food_count: 802,
                consumptions: 3,
                predations: 0,
                food_spawned: 4,
            },
        ];
        let batch_two = vec![TickSummaryRow {
            tick: 3,
            population: 302,
            max_generation: None,
            births: 1,
            deaths: 0,
            food_count: 804,
            consumptions: 4,
            predations: 1,
            food_spawned: 2,
        }];

        for row in &batch_one {
            writer.emit_tick(row.clone());
        }
        writer.flush().unwrap();
        for row in &batch_two {
            writer.emit_tick(row.clone());
        }
        Box::new(writer).finalize().unwrap();

        let dataset = DatasetReader::load(&dir).unwrap();
        let mut expected = batch_one.clone();
        expected.extend(batch_two);
        assert_eq!(dataset.tick_summary, expected);
    }

    #[test]
    fn every_table_round_trips() {
        let dir = tmp_dir("all");
        let mut writer = PartitionedParquetWriter::new(&dir).unwrap();

        writer.emit_tick(TickSummaryRow {
            tick: 1,
            population: 10,
            max_generation: None,
            births: 0,
            deaths: 0,
            food_count: 20,
            consumptions: 0,
            predations: 0,
            food_spawned: 0,
        });
        writer.emit_population_snapshot(PopulationSnapshotRow {
            tick: 1,
            brain_size_mean: Some(24.0),
            brain_size_stddev: Some(2.5),
            brain_size_p10: Some(20.0),
            brain_size_p50: Some(24.0),
            brain_size_p90: Some(28.0),
            lineage_diversity: Some(0.9),
        });
        writer.emit_action_count(ActionCountRow {
            tick: 1,
            action_type: 3,
            count: 7,
            failed_count: 2,
            juvenile_count: 3,
            adult_count: 4,
        });
        writer.emit_organism_lifetime(OrganismLifetimeRow {
            id: 42,
            parent_id: Some(7),
            species_id: 1,
            birth_tick: 0,
            death_tick: Some(100),
            generation: 2,
            age_of_maturity: 15,
            num_offspring: 3,
            total_consumptions: 15,
            total_actions: 100,
            action_histogram: vec![10, 20, 30, 15, 10, 10, 5],
            utilization: 0.42,
            food_ahead_ticks: 8,
            fwd_when_food_ahead: 5,
            joint_juvenile: vec![0; JOINT_LEN],
            joint_adult: vec![0; JOINT_LEN],
        });
        writer.emit_reproduction_event(ReproductionEventRow {
            tick: 50,
            parent_id: 42,
            parent_species_id: 1,
            parent_generation: 2,
            parent_age_turns: 50,
            child_id: Some(99),
            investment_energy: 400.0,
            parent_energy_after: 800.0,
            outcome: ReproductionOutcome::Success.code(),
        });
        Box::new(writer).finalize().unwrap();

        let dataset = DatasetReader::load(&dir).unwrap();
        assert_eq!(dataset.tick_summary.len(), 1);
        assert_eq!(dataset.population_snapshots.len(), 1);
        assert_eq!(dataset.action_counts.len(), 1);
        assert_eq!(dataset.organism_lifetimes.len(), 1);
        assert_eq!(dataset.reproduction_events.len(), 1);
        assert_eq!(dataset.genome_snapshots.len(), 0);
        assert_eq!(dataset.organism_lifetimes[0].action_histogram.len(), 7);
        assert_eq!(
            dataset.organism_lifetimes[0].joint_juvenile.len(),
            JOINT_LEN
        );
    }
}
