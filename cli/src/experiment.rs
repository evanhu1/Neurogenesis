use anyhow::{anyhow, bail, Context, Result};
use evolution::{GenerationSummary, RunResult};
use ring::digest::{Context as ShaContext, SHA256};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use views::{atomic_write, sibling_metrics_path};

const MANIFEST_SCHEMA: u32 = 1;
const SUMMARY_SCHEMA: u32 = 2;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BatchManifest {
    manifest_schema_version: u32,
    experiment: String,
    created_unix_ms: u128,
    completed_unix_ms: Option<u128>,
    command_argv: Vec<String>,
    shared_neat_args: Vec<String>,
    total_workers: usize,
    executable: String,
    executable_sha256: Option<String>,
    source: SourceIdentity,
    schedule: Vec<SeedRun>,
    resolved_contract: Option<Value>,
    validation_error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SourceIdentity {
    git_revision: Option<String>,
    dirty: Option<bool>,
    status_porcelain: Option<Vec<String>>,
    tracked_patch_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SeedRun {
    seed: u64,
    workers: usize,
    status: SeedStatus,
    elapsed_ms: Option<u128>,
    result: String,
    champion_world: String,
    champion_metrics: String,
    stdout_log: String,
    progress_log: String,
    result_sha256: Option<String>,
    champion_world_sha256: Option<String>,
    champion_metrics_sha256: Option<String>,
    error: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum SeedStatus {
    Pending,
    Succeeded,
    Failed,
}

struct BatchOptions {
    experiment: String,
    seeds: Vec<u64>,
    total_workers: usize,
    replace: bool,
    shared_args: Vec<String>,
}

struct SeedOutcome {
    seed: u64,
    elapsed: Duration,
    hashes: Option<(String, String, String)>,
    error: Option<String>,
}

pub(crate) fn run_batch_cli(args: &[&str], out_dir: &str, out: &mut impl Write) -> Result<()> {
    if args.iter().any(|arg| matches!(*arg, "--help" | "-h")) {
        writeln!(
            out,
            "cli batch --experiment SLUG --seeds N,N --total-workers N [--replace] -- [RUN OPTIONS]\n\
             Runs independent evolutionary seeds concurrently under one exact contract.\n\
             The total worker budget is divided deterministically in seed order.\n\
             Canonical run options include --horizon, --lineages-per-world,\n\
             --memberships-per-genome or --cases-per-genome, and --objective.\n\
             Batch owns --seed, --workers, and --out-dir."
        )?;
        return Ok(());
    }
    let options = parse_batch_options(args)?;
    let executable = std::env::current_exe().context("locating the running cli executable")?;
    let resolved_plan = preflight_plan(&executable, &options.shared_args)?;
    let worker_schedule = allocate_workers(options.total_workers, options.seeds.len())?;
    let experiment_dir = PathBuf::from(out_dir).join(&options.experiment);
    prepare_experiment_dir(&experiment_dir, options.replace)?;
    let mut manifest = BatchManifest {
        manifest_schema_version: MANIFEST_SCHEMA,
        experiment: options.experiment.clone(),
        created_unix_ms: unix_ms(),
        completed_unix_ms: None,
        command_argv: std::env::args().collect(),
        shared_neat_args: options.shared_args.clone(),
        total_workers: options.total_workers,
        executable: executable.to_string_lossy().into_owned(),
        executable_sha256: sha256_file(&executable).ok(),
        source: source_identity(),
        schedule: options
            .seeds
            .iter()
            .copied()
            .zip(worker_schedule.iter().copied())
            .map(|(seed, workers)| seed_run(seed, workers))
            .collect(),
        resolved_contract: Some(json!({ "plan": resolved_plan })),
        validation_error: None,
    };
    let manifest_path = experiment_dir.join("manifest.json");
    write_json_atomic(&manifest_path, &manifest)?;

    let mut handles = Vec::with_capacity(options.seeds.len());
    for (seed, workers) in options.seeds.iter().copied().zip(worker_schedule) {
        let executable = executable.clone();
        let experiment_dir = experiment_dir.clone();
        let shared_args = options.shared_args.clone();
        handles.push(std::thread::spawn(move || {
            run_seed_job(&executable, &experiment_dir, seed, workers, &shared_args)
        }));
    }
    let mut outcomes = BTreeMap::new();
    for handle in handles {
        let outcome = handle
            .join()
            .map_err(|_| anyhow!("a NEAT batch worker thread panicked"))?;
        outcomes.insert(outcome.seed, outcome);
    }
    for run in &mut manifest.schedule {
        let outcome = outcomes
            .remove(&run.seed)
            .ok_or_else(|| anyhow!("missing batch outcome for seed {}", run.seed))?;
        run.elapsed_ms = Some(outcome.elapsed.as_millis());
        run.error = outcome.error;
        run.status = if run.error.is_none() {
            SeedStatus::Succeeded
        } else {
            SeedStatus::Failed
        };
        if let Some((result, world, metrics)) = outcome.hashes {
            run.result_sha256 = Some(result);
            run.champion_world_sha256 = Some(world);
            run.champion_metrics_sha256 = Some(metrics);
        }
    }

    // Persist process outcomes before opening any very large result. A schema
    // or contract failure must not leave a misleading all-pending manifest.
    write_json_atomic(&manifest_path, &manifest)?;
    if manifest
        .schedule
        .iter()
        .all(|run| matches!(run.status, SeedStatus::Succeeded))
    {
        match validate_manifest_contracts(&experiment_dir, &manifest) {
            Ok(result_contract) => {
                manifest.resolved_contract = Some(json!({
                    "plan": resolved_plan,
                    "result_contract": result_contract,
                }));
            }
            Err(error) => {
                manifest.validation_error = Some(format!("{error:#}"));
                manifest.completed_unix_ms = Some(unix_ms());
                write_json_atomic(&manifest_path, &manifest)?;
                return Err(error);
            }
        }
    }
    manifest.completed_unix_ms = Some(unix_ms());
    write_json_atomic(&manifest_path, &manifest)?;

    let failures = manifest
        .schedule
        .iter()
        .filter(|run| matches!(run.status, SeedStatus::Failed))
        .map(|run| json!({ "seed": run.seed, "error": run.error }))
        .collect::<Vec<_>>();
    if !failures.is_empty() {
        bail!(
            "batch failed; inspect `{}`: {}",
            manifest_path.display(),
            Value::Array(failures)
        );
    }
    writeln!(
        out,
        "{}",
        json!({
            "experiment_dir": experiment_dir,
            "manifest": manifest_path,
            "seeds": options.seeds,
            "total_workers": options.total_workers,
            "schedule": manifest.schedule,
        })
    )
    .map_err(Into::into)
}

fn parse_batch_options(args: &[&str]) -> Result<BatchOptions> {
    let separator = args
        .iter()
        .position(|arg| *arg == "--")
        .ok_or_else(|| anyhow!("batch requires `--` before shared NEAT run options"))?;
    let batch_args = &args[..separator];
    let shared_args = &args[separator + 1..];
    let mut experiment = None;
    let mut seeds = None;
    let mut total_workers = None;
    let mut replace = false;
    let mut i = 0usize;
    while i < batch_args.len() {
        match batch_args[i] {
            "--experiment" => {
                experiment = Some(required_value(batch_args, i, "--experiment")?.to_string());
                i += 2;
            }
            "--seeds" => {
                seeds = Some(parse_seeds(required_value(batch_args, i, "--seeds")?)?);
                i += 2;
            }
            "--total-workers" => {
                total_workers = Some(required_value(batch_args, i, "--total-workers")?.parse()?);
                i += 2;
            }
            "--replace" => {
                replace = true;
                i += 1;
            }
            other => bail!("unknown batch argument `{other}`"),
        }
    }
    let experiment = experiment.ok_or_else(|| anyhow!("batch needs --experiment SLUG"))?;
    validate_slug(&experiment)?;
    let seeds = seeds.ok_or_else(|| anyhow!("batch needs --seeds N,N"))?;
    let total_workers =
        total_workers.ok_or_else(|| anyhow!("batch needs one machine-wide --total-workers N"))?;
    for forbidden in ["--seed", "--workers", "--out-dir"] {
        if shared_args.contains(&forbidden) {
            bail!("{forbidden} is owned by batch and cannot appear after `--`");
        }
    }
    Ok(BatchOptions {
        experiment,
        seeds,
        total_workers,
        replace,
        shared_args: shared_args.iter().map(|arg| (*arg).to_string()).collect(),
    })
}

fn preflight_plan(executable: &Path, shared_args: &[String]) -> Result<Value> {
    let output = Command::new(executable)
        .arg("plan")
        .args(shared_args)
        // Planning needs a valid worker count but worker allocation is owned
        // by batch and recorded separately in the schedule.
        .args(["--workers", "1"])
        .output()
        .context("launching batch preflight plan")?;
    if !output.status.success() {
        bail!(
            "batch preflight failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    String::from_utf8(output.stdout)
        .context("plan stdout was not UTF-8")?
        .lines()
        .rev()
        .find(|line| !line.trim().is_empty())
        .ok_or_else(|| anyhow!("batch preflight emitted no plan"))?
        .parse()
        .context("parsing batch preflight plan")
}

fn validate_slug(slug: &str) -> Result<()> {
    if slug.is_empty()
        || slug == "."
        || slug == ".."
        || !slug
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        bail!("--experiment must use only letters, digits, `.`, `_`, or `-`");
    }
    Ok(())
}

fn parse_seeds(raw: &str) -> Result<Vec<u64>> {
    let seeds = raw
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| part.trim().parse::<u64>())
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if seeds.is_empty() {
        bail!("--seeds needs at least one seed");
    }
    let unique = seeds.iter().copied().collect::<BTreeSet<_>>();
    if unique.len() != seeds.len() {
        bail!("--seeds cannot contain duplicates");
    }
    Ok(seeds)
}

fn allocate_workers(total: usize, jobs: usize) -> Result<Vec<usize>> {
    if jobs == 0 || total < jobs {
        bail!("--total-workers must be at least the number of evolution seeds ({jobs})");
    }
    let base = total / jobs;
    let remainder = total % jobs;
    Ok((0..jobs)
        .map(|index| base + usize::from(index < remainder))
        .collect())
}

fn prepare_experiment_dir(path: &Path, replace: bool) -> Result<()> {
    if path.exists() {
        let nonempty = fs::read_dir(path)
            .with_context(|| format!("reading `{}`", path.display()))?
            .next()
            .is_some();
        if nonempty && !replace {
            bail!(
                "experiment directory `{}` already contains artifacts; use --replace explicitly",
                path.display()
            );
        }
        if nonempty {
            fs::remove_dir_all(path).with_context(|| format!("replacing `{}`", path.display()))?;
        }
    }
    fs::create_dir_all(path).with_context(|| format!("creating `{}`", path.display()))
}

fn seed_run(seed: u64, workers: usize) -> SeedRun {
    SeedRun {
        seed,
        workers,
        status: SeedStatus::Pending,
        elapsed_ms: None,
        result: format!("seed-{seed}.result.json"),
        champion_world: format!("seed-{seed}.champion.world.bin"),
        champion_metrics: format!("seed-{seed}.champion.world.metrics"),
        stdout_log: format!("seed-{seed}.stdout.jsonl"),
        progress_log: format!("seed-{seed}.progress.jsonl"),
        result_sha256: None,
        champion_world_sha256: None,
        champion_metrics_sha256: None,
        error: None,
    }
}

fn run_seed_job(
    executable: &Path,
    experiment_dir: &Path,
    seed: u64,
    workers: usize,
    shared_args: &[String],
) -> SeedOutcome {
    let started = Instant::now();
    match run_seed_job_inner(executable, experiment_dir, seed, workers, shared_args) {
        Ok(hashes) => SeedOutcome {
            seed,
            elapsed: started.elapsed(),
            hashes: Some(hashes),
            error: None,
        },
        Err(error) => SeedOutcome {
            seed,
            elapsed: started.elapsed(),
            hashes: None,
            error: Some(format!("{error:#}")),
        },
    }
}

fn run_seed_job_inner(
    executable: &Path,
    experiment_dir: &Path,
    seed: u64,
    workers: usize,
    shared_args: &[String],
) -> Result<(String, String, String)> {
    let temporary_dir = experiment_dir.join(format!(".seed-{seed}.partial"));
    fs::create_dir(&temporary_dir)
        .with_context(|| format!("creating `{}`", temporary_dir.display()))?;
    let output = Command::new(executable)
        .arg("run")
        .args(shared_args)
        .args([
            "--seed",
            &seed.to_string(),
            "--workers",
            &workers.to_string(),
            "--out-dir",
            &temporary_dir.to_string_lossy(),
        ])
        .output()
        .with_context(|| format!("launching NEAT seed {seed}"))?;
    let stdout_path = experiment_dir.join(format!("seed-{seed}.stdout.jsonl"));
    let progress_path = experiment_dir.join(format!("seed-{seed}.progress.jsonl"));
    fs::write(&stdout_path, &output.stdout)?;
    fs::write(&progress_path, &output.stderr)?;
    if !output.status.success() {
        bail!(
            "seed {seed} exited with {}; progress is in `{}`",
            output.status,
            progress_path.display()
        );
    }
    let completion = String::from_utf8(output.stdout)
        .context("NEAT child stdout was not UTF-8")?
        .lines()
        .rev()
        .find(|line| !line.trim().is_empty())
        .ok_or_else(|| anyhow!("seed {seed} emitted no completion record"))?
        .parse::<Value>()?;
    let source_result = completion
        .get("wrote")
        .and_then(Value::as_str)
        .map(PathBuf::from)
        .ok_or_else(|| anyhow!("seed {seed} completion record has no `wrote` path"))?;
    let source_world = completion
        .get("champion_world")
        .and_then(Value::as_str)
        .map(PathBuf::from)
        .ok_or_else(|| anyhow!("seed {seed} completion record has no `champion_world` path"))?;
    let source_metrics = PathBuf::from(sibling_metrics_path(&source_world.to_string_lossy()));
    let result_path = experiment_dir.join(format!("seed-{seed}.result.json"));
    let world_path = experiment_dir.join(format!("seed-{seed}.champion.world.bin"));
    let metrics_path = experiment_dir.join(format!("seed-{seed}.champion.world.metrics"));
    fs::rename(&source_result, &result_path)?;
    fs::rename(&source_world, &world_path)?;
    fs::rename(&source_metrics, &metrics_path)?;
    fs::remove_dir(&temporary_dir)?;
    Ok((
        sha256_file(&result_path)?,
        sha256_file(&world_path)?,
        sha256_file(&metrics_path)?,
    ))
}

/// Parse each potentially large result exactly once and release it before the
/// next seed. Only the compact contract value survives this pass.
fn validate_manifest_contracts(experiment_dir: &Path, manifest: &BatchManifest) -> Result<Value> {
    let mut expected_schema = None;
    let mut expected_generations = None;
    let mut expected_contract = None;
    for seed_run in &manifest.schedule {
        let path = checked_result_path(experiment_dir, seed_run)?;
        let result = read_result(&path)?;
        validate_generation_sequence(&result.generations, &path)?;
        let schema = result.result_schema_version;
        let generations = generation_ids(&result);
        let contract = comparable_contract(&result);
        if let Some(expected) = expected_schema {
            if schema != expected {
                bail!("result schema mismatch in `{}`", path.display());
            }
        } else {
            expected_schema = Some(schema);
        }
        if let Some(expected) = expected_generations.as_ref() {
            if &generations != expected {
                bail!("generation sequence mismatch in `{}`", path.display());
            }
        } else {
            expected_generations = Some(generations);
        }
        if let Some(expected) = expected_contract.as_ref() {
            if &contract != expected {
                bail!(
                    "resolved evaluation/evolution contract mismatch in `{}`",
                    path.display()
                );
            }
        } else {
            expected_contract = Some(contract);
        }
        if result.seed != seed_run.seed {
            bail!(
                "result `{}` claims seed {}, manifest expects {}",
                path.display(),
                result.seed,
                seed_run.seed
            );
        }
    }
    expected_contract.ok_or_else(|| anyhow!("batch produced no results"))
}

pub(crate) fn run_summarize_cli(args: &[&str], out: &mut impl Write) -> Result<()> {
    if args.iter().any(|arg| matches!(*arg, "--help" | "-h")) {
        writeln!(
            out,
            "cli summarize EXPERIMENT_DIR_OR_MANIFEST --tail START:END\n\
             START and END are inclusive and must exist in every run.\n\
             Results are schema/contract checked and parsed sequentially; only compact trajectories are retained."
        )?;
        return Ok(());
    }
    let source = args
        .first()
        .ok_or_else(|| anyhow!("summarize needs an experiment directory or manifest path"))?;
    let mut tail = None;
    let mut i = 1usize;
    while i < args.len() {
        match args[i] {
            "--tail" => {
                tail = Some(parse_tail(required_value(args, i, "--tail")?)?);
                i += 2;
            }
            other => bail!("unknown summarize argument `{other}`"),
        }
    }
    let (tail_start, tail_end) =
        tail.ok_or_else(|| anyhow!("summarize requires an explicit --tail START:END"))?;
    let manifest_path = if Path::new(source).is_dir() {
        Path::new(source).join("manifest.json")
    } else {
        PathBuf::from(source)
    };
    let experiment_dir = manifest_path
        .parent()
        .ok_or_else(|| anyhow!("manifest path has no parent directory"))?;
    let manifest: BatchManifest = serde_json::from_reader(BufReader::new(
        File::open(&manifest_path)
            .with_context(|| format!("opening `{}`", manifest_path.display()))?,
    ))
    .with_context(|| format!("parsing `{}`", manifest_path.display()))?;
    if manifest.manifest_schema_version != MANIFEST_SCHEMA {
        bail!(
            "unsupported batch manifest schema {}; expected {}",
            manifest.manifest_schema_version,
            MANIFEST_SCHEMA
        );
    }

    let mut compact_runs = Vec::with_capacity(manifest.schedule.len());
    let mut expected_schema = None;
    let mut expected_contract = None;
    let mut expected_generation_ids = None;
    for seed_run in &manifest.schedule {
        let path = checked_result_path(experiment_dir, seed_run)?;
        let result = read_result(&path)?;
        validate_generation_sequence(&result.generations, &path)?;
        let ids = generation_ids(&result);
        let contract = comparable_contract(&result);
        if let Some(schema) = expected_schema {
            if result.result_schema_version != schema {
                bail!("result schema mismatch in `{}`", path.display());
            }
        } else {
            expected_schema = Some(result.result_schema_version);
        }
        if let Some(expected) = expected_generation_ids.as_ref() {
            if &ids != expected {
                bail!("generation sequence mismatch in `{}`", path.display());
            }
        } else {
            expected_generation_ids = Some(ids.clone());
        }
        if let Some(expected) = expected_contract.as_ref() {
            if &contract != expected {
                bail!(
                    "resolved evaluation/evolution contract mismatch in `{}`",
                    path.display()
                );
            }
        } else {
            expected_contract = Some(contract);
        }
        if result.seed != seed_run.seed {
            bail!("manifest/result seed mismatch for `{}`", path.display());
        }
        if !ids.contains(&tail_start) || !ids.contains(&tail_end) {
            bail!(
                "tail {tail_start}:{tail_end} is outside `{}` generation range",
                path.display()
            );
        }
        compact_runs.push(compact_run(&path, &result, tail_start, tail_end));
        // `result`, including its full population checkpoints, is dropped here
        // before the next large JSON file is opened.
    }
    if compact_runs.is_empty() {
        bail!("manifest contains no completed seed runs");
    }
    let cohort = cohort_trajectory(&compact_runs)?;
    let tail_cohort = cohort
        .iter()
        .filter(|row| row.generation >= tail_start && row.generation <= tail_end)
        .collect::<Vec<_>>();
    writeln!(
        out,
        "{}",
        json!({
            "summary_schema_version": SUMMARY_SCHEMA,
            "manifest": manifest_path,
            "result_schema_version": expected_schema,
            "validated_contract": expected_contract,
            "seeds": compact_runs.iter().map(|run| run.seed).collect::<Vec<_>>(),
            "tail": { "start": tail_start, "end": tail_end, "inclusive": true },
            "per_seed": compact_runs,
            "cohort_trajectory": cohort,
            "cohort_tail": tail_cohort,
        })
    )
    .map_err(Into::into)
}

fn checked_result_path(experiment_dir: &Path, seed_run: &SeedRun) -> Result<PathBuf> {
    if !matches!(seed_run.status, SeedStatus::Succeeded) {
        bail!(
            "seed {} is not complete (status {:?})",
            seed_run.seed,
            seed_run.status
        );
    }
    let path = experiment_dir.join(&seed_run.result);
    if let Some(expected) = seed_run.result_sha256.as_deref() {
        if sha256_file(&path)? != expected {
            bail!("result hash mismatch for `{}`", path.display());
        }
    }
    Ok(path)
}

fn read_result(path: &Path) -> Result<RunResult> {
    serde_json::from_reader(BufReader::new(
        File::open(path).with_context(|| format!("opening `{}`", path.display()))?,
    ))
    .with_context(|| {
        format!(
            "parsing current NEAT result/metric schema from `{}`",
            path.display()
        )
    })
}

fn parse_tail(raw: &str) -> Result<(u32, u32)> {
    let (start, end) = raw
        .split_once(':')
        .ok_or_else(|| anyhow!("--tail wants inclusive START:END"))?;
    let start = start.parse::<u32>()?;
    let end = end.parse::<u32>()?;
    if start > end {
        bail!("--tail start must be <= end");
    }
    Ok((start, end))
}

#[derive(Debug, Serialize)]
struct CompactRun {
    result: String,
    seed: u64,
    champion_generation: u32,
    champion_fitness: f64,
    trajectory: Vec<CompactGeneration>,
    tail: Value,
}

#[derive(Debug, Serialize)]
struct CompactGeneration {
    generation: u32,
    best_fitness: f64,
    mean_fitness: f64,
    median_fitness: f64,
    best_absolute_survival: f64,
    mean_absolute_survival: f64,
    best_alive_ticks: f64,
    mean_alive_ticks: f64,
    champion_relative_advantage: f64,
    champion_total_energy_accumulated: f64,
    mean_total_energy_accumulated: f64,
    median_total_energy_accumulated: f64,
    champion_net_energy_profit: f64,
    mean_net_energy_profit: f64,
    median_net_energy_profit: f64,
    champion_plant_energy: f64,
    mean_plant_energy: f64,
    champion_attack_energy_received: f64,
    champion_attack_energy_lost: f64,
    champion_attack_attempt_energy_cost: f64,
    champion_net_attack_energy: f64,
    mean_attack_energy_received: f64,
    mean_attack_energy_lost: f64,
    mean_attack_attempt_energy_cost: f64,
    mean_net_attack_energy: f64,
    champion_action_effectiveness: Option<f64>,
    mean_action_effectiveness: Option<f64>,
    champion_attack_precision: Option<f64>,
    population_attack_precision: Option<f64>,
    champion_mean_attack_kills: f64,
    champion_distinct_attack_victims: f64,
    mean_distinct_attack_victims: f64,
    champion_action_fractions: [f64; 6],
    mean_action_fractions: [f64; 6],
    champion_plant_rate: Option<f64>,
    champion_prey_rate: Option<f64>,
    mean_plant_rate: Option<f64>,
    mean_prey_rate: Option<f64>,
    champion_plant_capture_fraction: Option<f64>,
    mean_plant_capture_fraction: Option<f64>,
    champion_standing_plant_fraction: f64,
    mean_standing_plant_fraction: f64,
    champion_spatial_coverage: f64,
    mean_spatial_coverage: f64,
    champion_end_survival_fraction: f64,
    mean_end_survival_fraction: f64,
    mean_opponent_score_stddev: Option<f64>,
    max_opponent_score_stddev: Option<f64>,
    species: usize,
    champion_expressed_hidden_nodes: usize,
    champion_expressed_connections: usize,
    mean_expressed_hidden_nodes: f64,
    mean_expressed_connections: f64,
    new_connection_innovations: usize,
    new_node_innovations: usize,
}

impl CompactGeneration {
    fn from_summary(g: &GenerationSummary) -> Self {
        let champion = g
            .population_checkpoint
            .iter()
            .max_by(|left, right| left.fitness.total_cmp(&right.fitness))
            .expect("a generation summary has a nonempty population checkpoint");
        let champion_evaluation = &champion.evaluation;
        let champion_total_energy_accumulated = champion_evaluation.mean_gross_energy_acquired;
        let champion_net_energy_profit = net_energy_profit(champion_evaluation);
        let mut population_net_energy_profit = g
            .population_checkpoint
            .iter()
            .map(|member| net_energy_profit(&member.evaluation))
            .collect::<Vec<_>>();
        population_net_energy_profit.sort_by(f64::total_cmp);
        let mean_net_energy_profit = population_net_energy_profit.iter().sum::<f64>()
            / population_net_energy_profit.len() as f64;
        let median_net_energy_profit = median_sorted(&population_net_energy_profit);
        let population_attack_hits = g
            .population_checkpoint
            .iter()
            .map(|member| member.evaluation.mean_attack_hits)
            .sum::<f64>();
        let population_attack_attempts = g
            .population_checkpoint
            .iter()
            .map(|member| attack_attempts(&member.evaluation))
            .sum::<f64>();
        Self {
            generation: g.generation,
            best_fitness: g.best_fitness,
            mean_fitness: g.mean_fitness,
            median_fitness: g.median_fitness,
            best_absolute_survival: g.best_absolute_survival_fraction,
            mean_absolute_survival: g.mean_absolute_survival_fraction,
            best_alive_ticks: g.best_candidate_alive_ticks,
            mean_alive_ticks: g.mean_candidate_alive_ticks,
            champion_relative_advantage: g.best_relative_survival_advantage,
            champion_total_energy_accumulated,
            mean_total_energy_accumulated: g.mean_gross_energy_acquired,
            median_total_energy_accumulated: g.median_gross_energy_acquired,
            champion_net_energy_profit,
            mean_net_energy_profit,
            median_net_energy_profit,
            champion_plant_energy: g.champion_plant_energy_acquired,
            mean_plant_energy: g.mean_plant_energy_acquired,
            champion_attack_energy_received: g.champion_attack_energy_received,
            champion_attack_energy_lost: g.champion_attack_energy_lost,
            champion_attack_attempt_energy_cost: g.champion_attack_attempt_energy_cost,
            champion_net_attack_energy: g.champion_net_attack_energy_balance,
            mean_attack_energy_received: g.mean_attack_energy_received,
            mean_attack_energy_lost: g.mean_attack_energy_lost,
            mean_attack_attempt_energy_cost: g.mean_attack_attempt_energy_cost,
            mean_net_attack_energy: g.mean_net_attack_energy_balance,
            champion_action_effectiveness: g.best_action_effectiveness,
            mean_action_effectiveness: g.mean_action_effectiveness,
            champion_attack_precision: attack_precision(champion_evaluation),
            population_attack_precision: (population_attack_attempts > 0.0)
                .then(|| population_attack_hits / population_attack_attempts),
            champion_mean_attack_kills: g.best_mean_attack_kills,
            champion_distinct_attack_victims: g.champion_distinct_attack_victims,
            mean_distinct_attack_victims: g.mean_distinct_attack_victims,
            champion_action_fractions: g.champion_action_fractions,
            mean_action_fractions: g.mean_action_fractions,
            champion_plant_rate: g.best_plant_consumption_rate,
            champion_prey_rate: g.best_prey_consumption_rate,
            mean_plant_rate: g.mean_plant_consumption_rate,
            mean_prey_rate: g.mean_prey_consumption_rate,
            champion_plant_capture_fraction: g.champion_plant_capture_fraction,
            mean_plant_capture_fraction: g.mean_plant_capture_fraction,
            champion_standing_plant_fraction: g.champion_standing_plant_fraction,
            mean_standing_plant_fraction: g.mean_standing_plant_fraction,
            champion_spatial_coverage: g.champion_spatial_coverage,
            mean_spatial_coverage: g.mean_spatial_coverage,
            champion_end_survival_fraction: g.best_end_survival_fraction,
            mean_end_survival_fraction: g.mean_end_survival_fraction,
            mean_opponent_score_stddev: g.mean_opponent_score_stddev,
            max_opponent_score_stddev: g.max_opponent_score_stddev,
            species: g.species.len(),
            champion_expressed_hidden_nodes: g.best_expressed_hidden_nodes,
            champion_expressed_connections: g.best_expressed_connections,
            mean_expressed_hidden_nodes: g.mean_expressed_hidden_nodes,
            mean_expressed_connections: g.mean_expressed_connections,
            new_connection_innovations: g.new_connection_innovations,
            new_node_innovations: g.new_node_innovations,
        }
    }
}

fn attack_attempts(evaluation: &evolution::Evaluation) -> f64 {
    evaluation.mean_attack_no_organism_targets
        + evaluation.mean_attack_same_pool_blocked
        + evaluation.mean_attack_insufficient_energy
        + evaluation.mean_attack_eligible_attempts
}

fn attack_precision(evaluation: &evolution::Evaluation) -> Option<f64> {
    let attempts = attack_attempts(evaluation);
    (attempts > 0.0).then(|| evaluation.mean_attack_hits / attempts)
}

fn net_energy_profit(evaluation: &evolution::Evaluation) -> f64 {
    evaluation.mean_plant_energy_acquired + evaluation.mean_attack_energy_received
        - evaluation.mean_attack_energy_lost
        - evaluation.mean_attack_attempt_energy_cost
}

fn median_sorted(values: &[f64]) -> f64 {
    if values.len().is_multiple_of(2) {
        let high = values.len() / 2;
        (values[high - 1] + values[high]) / 2.0
    } else {
        values[values.len() / 2]
    }
}

#[derive(Debug, Serialize)]
struct CohortGeneration {
    generation: u32,
    metrics: BTreeMap<&'static str, Distribution>,
}

#[derive(Debug, Serialize)]
struct Distribution {
    mean: f64,
    min: f64,
    max: f64,
}

fn compact_run(path: &Path, result: &RunResult, tail_start: u32, tail_end: u32) -> CompactRun {
    let trajectory = result
        .generations
        .iter()
        .map(CompactGeneration::from_summary)
        .collect::<Vec<_>>();
    let tail = trajectory
        .iter()
        .filter(|generation| {
            generation.generation >= tail_start && generation.generation <= tail_end
        })
        .collect::<Vec<_>>();
    let tail = tail_summary(&tail);
    CompactRun {
        result: path.to_string_lossy().into_owned(),
        seed: result.seed,
        champion_generation: result.champion_generation,
        champion_fitness: result.champion_fitness,
        trajectory,
        tail,
    }
}

fn cohort_trajectory(runs: &[CompactRun]) -> Result<Vec<CohortGeneration>> {
    let generation_count = runs[0].trajectory.len();
    if runs
        .iter()
        .any(|run| run.trajectory.len() != generation_count)
    {
        bail!("cannot aggregate mismatched trajectory lengths");
    }
    let mut rows = Vec::with_capacity(generation_count);
    for index in 0..generation_count {
        let generations = runs
            .iter()
            .map(|run| &run.trajectory[index])
            .collect::<Vec<_>>();
        let generation = generations[0].generation;
        if generations.iter().any(|row| row.generation != generation) {
            bail!("cannot aggregate mismatched generation coordinates");
        }
        let mut metrics = BTreeMap::new();
        macro_rules! metric {
            ($name:literal, $field:ident) => {
                metrics.insert(
                    $name,
                    distribution(generations.iter().map(|row| row.$field)),
                );
            };
        }
        macro_rules! optional_metric {
            ($name:literal, $field:ident) => {
                if generations.iter().all(|row| row.$field.is_some()) {
                    metrics.insert(
                        $name,
                        distribution(generations.iter().filter_map(|row| row.$field)),
                    );
                }
            };
        }
        metric!("best_fitness", best_fitness);
        metric!("mean_fitness", mean_fitness);
        metric!("median_fitness", median_fitness);
        metric!("best_absolute_survival", best_absolute_survival);
        metric!("mean_absolute_survival", mean_absolute_survival);
        metric!("best_alive_ticks", best_alive_ticks);
        metric!("mean_alive_ticks", mean_alive_ticks);
        metric!("champion_relative_advantage", champion_relative_advantage);
        metric!(
            "champion_total_energy_accumulated",
            champion_total_energy_accumulated
        );
        metric!(
            "mean_total_energy_accumulated",
            mean_total_energy_accumulated
        );
        metric!(
            "median_total_energy_accumulated",
            median_total_energy_accumulated
        );
        metric!("champion_net_energy_profit", champion_net_energy_profit);
        metric!("mean_net_energy_profit", mean_net_energy_profit);
        metric!("median_net_energy_profit", median_net_energy_profit);
        metric!("champion_plant_energy", champion_plant_energy);
        metric!(
            "champion_attack_energy_received",
            champion_attack_energy_received
        );
        metric!("champion_attack_energy_lost", champion_attack_energy_lost);
        metric!(
            "champion_attack_attempt_energy_cost",
            champion_attack_attempt_energy_cost
        );
        metric!("champion_net_attack_energy", champion_net_attack_energy);
        optional_metric!(
            "champion_action_effectiveness",
            champion_action_effectiveness
        );
        optional_metric!("mean_action_effectiveness", mean_action_effectiveness);
        optional_metric!("champion_attack_precision", champion_attack_precision);
        optional_metric!("population_attack_precision", population_attack_precision);
        optional_metric!("champion_plant_rate", champion_plant_rate);
        optional_metric!("champion_prey_rate", champion_prey_rate);
        optional_metric!("mean_plant_rate", mean_plant_rate);
        optional_metric!("mean_prey_rate", mean_prey_rate);
        optional_metric!(
            "champion_plant_capture_fraction",
            champion_plant_capture_fraction
        );
        optional_metric!("mean_plant_capture_fraction", mean_plant_capture_fraction);
        metric!("champion_spatial_coverage", champion_spatial_coverage);
        metric!("mean_spatial_coverage", mean_spatial_coverage);
        metric!(
            "champion_end_survival_fraction",
            champion_end_survival_fraction
        );
        metric!("mean_end_survival_fraction", mean_end_survival_fraction);
        metric!(
            "champion_standing_plant_fraction",
            champion_standing_plant_fraction
        );
        for (name, action_index) in [
            ("champion_action_idle_fraction", 0),
            ("champion_action_turn_left_fraction", 1),
            ("champion_action_turn_right_fraction", 2),
            ("champion_action_forward_fraction", 3),
            ("champion_action_eat_fraction", 4),
            ("champion_action_attack_fraction", 5),
        ] {
            metrics.insert(
                name,
                distribution(
                    generations
                        .iter()
                        .map(|row| row.champion_action_fractions[action_index]),
                ),
            );
        }
        rows.push(CohortGeneration {
            generation,
            metrics,
        });
    }
    Ok(rows)
}

fn distribution(values: impl Iterator<Item = f64>) -> Distribution {
    let values = values.collect::<Vec<_>>();
    Distribution {
        mean: values.iter().sum::<f64>() / values.len() as f64,
        min: values.iter().copied().fold(f64::INFINITY, f64::min),
        max: values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    }
}

fn tail_summary(tail: &[&CompactGeneration]) -> Value {
    let first = tail.first().expect("validated nonempty tail");
    let last = tail.last().expect("validated nonempty tail");
    let x = tail
        .iter()
        .map(|generation| generation.generation as f64)
        .collect::<Vec<_>>();
    let champion_action_effectiveness = tail
        .iter()
        .filter_map(|generation| generation.champion_action_effectiveness)
        .collect::<Vec<_>>();
    let mean_action_effectiveness = tail
        .iter()
        .filter_map(|generation| generation.mean_action_effectiveness)
        .collect::<Vec<_>>();
    json!({
        "start_generation": first.generation,
        "end_generation": last.generation,
        "samples": tail.len(),
        "best_fitness_delta": last.best_fitness - first.best_fitness,
        "mean_fitness_delta": last.mean_fitness - first.mean_fitness,
        "best_absolute_survival_delta": last.best_absolute_survival - first.best_absolute_survival,
        "mean_absolute_survival_delta": last.mean_absolute_survival - first.mean_absolute_survival,
        "best_alive_ticks_delta": last.best_alive_ticks - first.best_alive_ticks,
        "mean_alive_ticks_delta": last.mean_alive_ticks - first.mean_alive_ticks,
        "champion_total_energy_accumulated_delta": last.champion_total_energy_accumulated - first.champion_total_energy_accumulated,
        "mean_total_energy_accumulated_delta": last.mean_total_energy_accumulated - first.mean_total_energy_accumulated,
        "champion_net_energy_profit_delta": last.champion_net_energy_profit - first.champion_net_energy_profit,
        "mean_net_energy_profit_delta": last.mean_net_energy_profit - first.mean_net_energy_profit,
        "champion_action_effectiveness_delta": optional_delta(first.champion_action_effectiveness, last.champion_action_effectiveness),
        "mean_action_effectiveness_delta": optional_delta(first.mean_action_effectiveness, last.mean_action_effectiveness),
        "best_fitness_slope": slope(&x, &tail.iter().map(|g| g.best_fitness).collect::<Vec<_>>()),
        "mean_fitness_slope": slope(&x, &tail.iter().map(|g| g.mean_fitness).collect::<Vec<_>>()),
        "best_absolute_survival_slope": slope(&x, &tail.iter().map(|g| g.best_absolute_survival).collect::<Vec<_>>()),
        "mean_absolute_survival_slope": slope(&x, &tail.iter().map(|g| g.mean_absolute_survival).collect::<Vec<_>>()),
        "champion_net_energy_profit_slope": slope(&x, &tail.iter().map(|g| g.champion_net_energy_profit).collect::<Vec<_>>()),
        "mean_net_energy_profit_slope": slope(&x, &tail.iter().map(|g| g.mean_net_energy_profit).collect::<Vec<_>>()),
        "champion_action_effectiveness_slope": (champion_action_effectiveness.len() == x.len())
            .then(|| slope(&x, &champion_action_effectiveness)).flatten(),
        "mean_action_effectiveness_slope": (mean_action_effectiveness.len() == x.len())
            .then(|| slope(&x, &mean_action_effectiveness)).flatten(),
    })
}

fn optional_delta(first: Option<f64>, last: Option<f64>) -> Option<f64> {
    Some(last? - first?)
}

fn slope(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() != y.len() || x.len() < 2 {
        return None;
    }
    let mean_x = x.iter().sum::<f64>() / x.len() as f64;
    let mean_y = y.iter().sum::<f64>() / y.len() as f64;
    let covariance = x
        .iter()
        .zip(y)
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum::<f64>();
    let variance = x.iter().map(|x| (x - mean_x).powi(2)).sum::<f64>();
    (variance > 0.0).then_some(covariance / variance)
}

fn generation_ids(result: &RunResult) -> Vec<u32> {
    result
        .generations
        .iter()
        .map(|generation| generation.generation)
        .collect()
}

fn validate_generation_sequence(generations: &[GenerationSummary], path: &Path) -> Result<()> {
    if generations.is_empty() {
        bail!("result `{}` has no generations", path.display());
    }
    for (index, generation) in generations.iter().enumerate() {
        if generation.generation as usize != index {
            bail!(
                "result `{}` has a missing or reordered generation at index {index}",
                path.display()
            );
        }
    }
    Ok(())
}

fn comparable_contract(result: &RunResult) -> Value {
    let mut neat_config = serde_json::to_value(&result.neat_config).expect("NeatConfig serializes");
    // Worker count changes throughput, not deterministic evolutionary
    // semantics, and batch deliberately divides this budget across jobs.
    neat_config
        .as_object_mut()
        .expect("NeatConfig serializes as an object")
        .remove("evaluator_workers");
    json!({
        "algorithm": result.algorithm,
        "objective": result.objective,
        "neat_config": neat_config,
        "frozen_outer_loop_contract": result.frozen_outer_loop_contract,
        "world_width": result.world_width,
        "founder_cohort_size": result.founder_cohort_size,
        "food_energy": result.food_energy,
        "replay_anchor_scenarios": result.replay_anchor_scenarios,
        "final_training_scenarios": result.final_training_scenarios,
    })
}

fn source_identity() -> SourceIdentity {
    let revision = git_output(["rev-parse", "HEAD"]);
    let status = git_output(["status", "--porcelain=v1"]);
    let patch = Command::new("git")
        .args(["diff", "--binary", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| sha256_bytes(&output.stdout));
    SourceIdentity {
        git_revision: revision,
        dirty: status.as_ref().map(|status| !status.is_empty()),
        status_porcelain: status.map(|status| status.lines().map(str::to_string).collect()),
        tracked_patch_sha256: patch,
    }
}

fn git_output<const N: usize>(args: [&str; N]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn required_value<'a>(args: &[&'a str], index: usize, flag: &str) -> Result<&'a str> {
    args.get(index + 1)
        .copied()
        .ok_or_else(|| anyhow!("{flag} needs a value"))
}

fn unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<()> {
    let path = path.to_string_lossy();
    atomic_write(&path, |writer| {
        serde_json::to_writer_pretty(writer, value).map_err(Into::into)
    })
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut reader = BufReader::new(
        File::open(path).with_context(|| format!("opening `{}` for hashing", path.display()))?,
    );
    let mut context = ShaContext::new(&SHA256);
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        context.update(&buffer[..count]);
    }
    Ok(context
        .finish()
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut context = ShaContext::new(&SHA256);
    context.update(bytes);
    context
        .finish()
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
