//! `sim-evaluation analyze <run>` entrypoint. Re-derives metrics and pillars
//! from a persisted dataset and rewrites the human-readable artifacts
//! (`summary.json`, `timeseries.csv`, `report.html`) without re-running the
//! simulation.
//!
//! The argument can be any of:
//! - a path to a run root (containing `seed_*` subdirs) — re-analyzes every
//!   seed and rebuilds the top-level aggregate report
//! - a path to a single seed dataset (containing `manifest.json`)
//! - a timestamp prefix (e.g. `20260416T002137Z`) resolved under
//!   `artifacts/evaluation/`
//! - the literal `latest`, selecting the newest run with persisted datasets

use super::{
    analyze, average_pillar_scores, average_timeseries, write_aggregate_artifacts,
    write_per_seed_artifacts, AnalysisOptions,
};
use crate::dataset::{DatasetReader, Manifest};
use crate::output::print_evaluation_summary;
use crate::types::{EvaluationSummary, SeedEvaluationSummary, SeedRunSummary};
use anyhow::{anyhow, Result};
use chrono::Utc;
use std::path::{Path, PathBuf};

const EVALUATION_ROOT: &str = "artifacts/evaluation";

pub fn analyze_run(identifier: &str) -> Result<()> {
    let target = resolve_run_path(identifier)?;
    let summary = if target.join("manifest.json").is_file() {
        analyze_single_seed(&target)?
    } else {
        analyze_run_root(&target)?
    };
    print_evaluation_summary(&target, &summary);
    Ok(())
}

fn analyze_single_seed(dataset_dir: &Path) -> Result<EvaluationSummary> {
    let (seed_summary, manifest) = reanalyze_seed(dataset_dir)?;
    write_per_seed_artifacts(
        dataset_dir,
        &seed_summary,
        manifest.report_every,
        "re-analyzed from dataset",
    )?;
    Ok(wrap_single_seed(seed_summary))
}

fn analyze_run_root(run_dir: &Path) -> Result<EvaluationSummary> {
    let seed_dirs = collect_seed_dirs(run_dir)?;
    if seed_dirs.is_empty() {
        anyhow::bail!(
            "no seed datasets found under {} (expected seed_*/manifest.json)",
            run_dir.display()
        );
    }

    let mut seed_summaries = Vec::with_capacity(seed_dirs.len());
    // The first seed's manifest is the source of truth for run-level options
    // (report_every, ticks). `control` comes from the persisted world config.
    let mut first_manifest = None;
    for seed_dir in &seed_dirs {
        let (summary, manifest) = reanalyze_seed(seed_dir)?;
        write_per_seed_artifacts(
            seed_dir,
            &summary,
            manifest.report_every,
            "re-analyzed from dataset",
        )?;
        seed_summaries.push(summary);
        if first_manifest.is_none() {
            first_manifest = Some(manifest);
        }
    }
    seed_summaries.sort_by_key(|summary| summary.seed);

    let first_manifest =
        first_manifest.ok_or_else(|| anyhow!("no seed manifests found under {run_dir:?}"))?;
    let control = first_manifest.world_config.force_random_actions;

    let evaluation_summary = EvaluationSummary {
        title: None,
        seeds: seed_summaries.iter().map(|s| s.seed).collect(),
        ticks: first_manifest.total_ticks,
        control,
        worker_threads: 0,
        total_time_seconds: 0.0,
        pillars: average_pillar_scores(&seed_summaries),
        seed_summaries: seed_summaries
            .iter()
            .map(|summary| SeedRunSummary {
                seed: summary.seed,
                out_dir: PathBuf::from(format!("seed_{}", summary.seed)),
                total_time_seconds: summary.total_time_seconds,
                pillars: summary.pillars.clone(),
                state_hash: summary.state_hash.clone(),
            })
            .collect(),
        timeseries: average_timeseries(&seed_summaries),
    };

    let generated_at_utc = Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string();
    write_aggregate_artifacts(
        run_dir,
        &evaluation_summary,
        first_manifest.report_every,
        &generated_at_utc,
    )?;
    Ok(evaluation_summary)
}

fn wrap_single_seed(summary: SeedEvaluationSummary) -> EvaluationSummary {
    let seed_run_summary = SeedRunSummary {
        seed: summary.seed,
        out_dir: PathBuf::new(),
        total_time_seconds: summary.total_time_seconds,
        pillars: summary.pillars.clone(),
        state_hash: summary.state_hash.clone(),
    };
    EvaluationSummary {
        title: summary.title,
        seeds: vec![summary.seed],
        ticks: summary.ticks,
        control: summary.control,
        worker_threads: 0,
        total_time_seconds: summary.total_time_seconds,
        pillars: summary.pillars,
        seed_summaries: vec![seed_run_summary],
        timeseries: summary.timeseries,
    }
}

fn reanalyze_seed(dataset_dir: &Path) -> Result<(SeedEvaluationSummary, Manifest)> {
    let manifest = Manifest::read(dataset_dir)?;
    let dataset = DatasetReader::load(dataset_dir)?;
    let analysis = analyze(
        &dataset,
        &AnalysisOptions {
            report_every: manifest.report_every,
            total_ticks: manifest.total_ticks,
        },
    );
    let summary = SeedEvaluationSummary {
        title: None,
        seed: manifest.seed,
        ticks: manifest.total_ticks,
        control: manifest.world_config.force_random_actions,
        total_time_seconds: 0.0,
        pillars: analysis.pillars,
        state_hash: String::new(),
        timeseries: analysis.timeseries,
    };
    Ok((summary, manifest))
}

fn collect_seed_dirs(run_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(run_dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_dir()
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("seed_"))
                && path.join("manifest.json").is_file()
        })
        .collect();
    dirs.sort();
    Ok(dirs)
}

fn resolve_run_path(identifier: &str) -> Result<PathBuf> {
    let direct = Path::new(identifier);
    if direct.exists() {
        return Ok(direct.to_path_buf());
    }

    let root = Path::new(EVALUATION_ROOT);
    if !root.is_dir() {
        return Err(anyhow!(
            "could not resolve {identifier:?}: path does not exist and {EVALUATION_ROOT}/ is not present"
        ));
    }

    let mut runs: Vec<PathBuf> = std::fs::read_dir(root)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_dir() && run_has_dataset(path))
        .collect();
    // Run directories are prefixed with an ISO-like UTC timestamp, so lex sort
    // on filename descending = newest first.
    runs.sort_by(|a, b| b.file_name().cmp(&a.file_name()));

    if identifier == "latest" {
        return runs.into_iter().next().ok_or_else(|| {
            anyhow!("no evaluation runs with persisted datasets found under {EVALUATION_ROOT}/")
        });
    }

    runs.into_iter()
        .find(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with(identifier))
        })
        .ok_or_else(|| {
            anyhow!("no evaluation run matching {identifier:?} found under {EVALUATION_ROOT}/")
        })
}

fn run_has_dataset(path: &Path) -> bool {
    if path.join("manifest.json").is_file() {
        return true;
    }
    std::fs::read_dir(path)
        .map(|entries| {
            entries.filter_map(|e| e.ok()).any(|entry| {
                entry
                    .file_name()
                    .to_str()
                    .is_some_and(|name| name.starts_with("seed_"))
                    && entry.path().join("manifest.json").is_file()
            })
        })
        .unwrap_or(false)
}
