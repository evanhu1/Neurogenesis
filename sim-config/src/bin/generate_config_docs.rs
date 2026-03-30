use anyhow::Result;
use sim_config::world_config_reference_markdown;
use std::fs;
use std::path::Path;

fn main() -> Result<()> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let output_path = manifest_dir.join("CONFIG_REFERENCE.md");
    fs::write(output_path, world_config_reference_markdown())?;
    Ok(())
}
