use anyhow::{Context, Result};
use sim_types::{world_config_from_toml_str, WorldConfig};
use std::path::{Path, PathBuf};

pub mod protocol;

const DEFAULT_WORLD_CONFIG_REL_PATH: &str = "../config/default.toml";

pub fn default_world_config_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(DEFAULT_WORLD_CONFIG_REL_PATH)
}

pub fn load_default_world_config() -> Result<WorldConfig> {
    load_world_config_from_path(&default_world_config_path())
}

fn load_world_config_from_path(path: &Path) -> Result<WorldConfig> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read world config from {}", path.display()))?;
    world_config_from_toml_str(&raw)
        .context("world config TOML failed schema deserialization")
        .with_context(|| format!("failed to parse world config from {}", path.display()))
}
