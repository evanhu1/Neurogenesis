use anyhow::Result;
use serde::{Deserialize, Serialize};
use sim_types::WorldConfig;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

/// Bump when the on-disk dataset schema changes in a way that older readers
/// can't tolerate. The reader refuses datasets with a different version.
pub const SCHEMA_VERSION: u32 = 7;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub schema_version: u32,
    pub seed: u64,
    pub total_ticks: u64,
    pub report_every: u64,
    pub created_at_utc: String,
    pub world_config: WorldConfig,
}

impl Manifest {
    pub fn write(&self, dir: &Path) -> Result<()> {
        let file = File::create(dir.join("manifest.json"))?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }

    pub fn read(dir: &Path) -> Result<Self> {
        let file = File::open(dir.join("manifest.json"))?;
        let reader = BufReader::new(file);
        // Two-stage parse: check schema_version on the raw JSON first so a
        // version mismatch reports as such instead of failing deep inside
        // typed deserialization (e.g. a "missing field" error from an old
        // WorldConfig shape).
        let value: serde_json::Value = serde_json::from_reader(reader)?;
        let schema_version = value
            .get("schema_version")
            .and_then(serde_json::Value::as_u64);
        if schema_version != Some(u64::from(SCHEMA_VERSION)) {
            anyhow::bail!(
                "dataset schema_version {} does not match reader's {}",
                schema_version
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "<missing>".to_owned()),
                SCHEMA_VERSION
            );
        }
        Ok(serde_json::from_value(value)?)
    }
}
