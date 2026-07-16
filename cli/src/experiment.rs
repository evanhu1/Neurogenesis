use anyhow::{anyhow, bail, Context, Result};
use evolution::{GenerationSummary, RunResult};
use ring::digest::{Context as ShaContext, SHA256};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use views::{atomic_write, sibling_metrics_path};

const MANIFEST_SCHEMA: u32 = 1;
const SUMMARY_SCHEMA: u32 = 3;

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
    final_winner_world: String,
    final_winner_metrics: String,
    stdout_log: String,
    progress_log: String,
    result_sha256: Option<String>,
    final_winner_world_sha256: Option<String>,
    final_winner_metrics_sha256: Option<String>,
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
             Canonical run options include --horizon, --opponents-per-genome,\n\
             --cases-per-genome, and --cvar.\n\
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
            run.final_winner_world_sha256 = Some(world);
            run.final_winner_metrics_sha256 = Some(metrics);
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
        result: format!("seed-{seed}.result.json.zst"),
        final_winner_world: format!("seed-{seed}.final-winner.world.bin"),
        final_winner_metrics: format!("seed-{seed}.final-winner.world.metrics"),
        stdout_log: format!("seed-{seed}.stdout.jsonl"),
        progress_log: format!("seed-{seed}.progress.jsonl"),
        result_sha256: None,
        final_winner_world_sha256: None,
        final_winner_metrics_sha256: None,
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
    let mut child = Command::new(executable)
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
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("launching NEAT seed {seed}"))?;
    let stdout_path = experiment_dir.join(format!("seed-{seed}.stdout.jsonl"));
    let progress_path = experiment_dir.join(format!("seed-{seed}.progress.jsonl"));
    let child_stderr = child
        .stderr
        .take()
        .ok_or_else(|| anyhow!("NEAT seed {seed} has no stderr pipe"))?;
    let streamed_progress_path = progress_path.clone();
    let progress_thread = std::thread::spawn(move || -> std::io::Result<()> {
        let mut progress_log = File::create(streamed_progress_path)?;
        for line in BufReader::new(child_stderr).lines() {
            let line = line?;
            writeln!(progress_log, "{line}")?;
            eprintln!("{line}");
        }
        progress_log.flush()
    });
    let output = child
        .wait_with_output()
        .with_context(|| format!("waiting for NEAT seed {seed}"))?;
    progress_thread
        .join()
        .map_err(|_| anyhow!("progress reader for seed {seed} panicked"))?
        .with_context(|| format!("streaming progress for seed {seed}"))?;
    fs::write(&stdout_path, &output.stdout)?;
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
        .get("final_winner_world")
        .and_then(Value::as_str)
        .map(PathBuf::from)
        .ok_or_else(|| anyhow!("seed {seed} completion record has no `final_winner_world` path"))?;
    let source_metrics = PathBuf::from(sibling_metrics_path(&source_world.to_string_lossy()));
    let result_path = experiment_dir.join(format!("seed-{seed}.result.json.zst"));
    let world_path = experiment_dir.join(format!("seed-{seed}.final-winner.world.bin"));
    let metrics_path = experiment_dir.join(format!("seed-{seed}.final-winner.world.metrics"));
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
    let file = File::open(path).with_context(|| format!("opening `{}`", path.display()))?;
    let reader: Box<dyn Read> = if path.extension().is_some_and(|extension| extension == "zst") {
        Box::new(zstd::stream::read::Decoder::new(file)?)
    } else {
        Box::new(BufReader::new(file))
    };
    serde_json::from_reader(reader).with_context(|| {
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
    final_generation: u32,
    final_winner_contextual_score: f64,
    trajectory: Vec<CompactGeneration>,
    tail: Value,
}

#[derive(Debug, Serialize)]
struct CompactGeneration {
    generation: u32,
    winner_contextual_score: f64,
    winner_case_score_stddev: f64,
    mean_case_score_stddev: f64,
    max_case_score_stddev: f64,
    winner_absolute_survival: f64,
    mean_absolute_survival: f64,
    winner_alive_ticks: f64,
    mean_alive_ticks: f64,
    winner_relative_advantage: f64,
    winner_total_energy_accumulated: f64,
    mean_total_energy_accumulated: f64,
    median_total_energy_accumulated: f64,
    winner_net_energy_profit: f64,
    mean_net_energy_profit: f64,
    median_net_energy_profit: f64,
    winner_attack_energy_received: f64,
    winner_attack_energy_lost: f64,
    winner_attack_attempt_energy_cost: f64,
    winner_net_attack_energy: f64,
    mean_attack_energy_received: f64,
    mean_attack_energy_lost: f64,
    mean_attack_attempt_energy_cost: f64,
    mean_net_attack_energy: f64,
    winner_action_effectiveness: Option<f64>,
    mean_action_effectiveness: Option<f64>,
    winner_successful_attack_rate: Option<f64>,
    mean_successful_attack_rate: Option<f64>,
    winner_attack_precision: Option<f64>,
    population_attack_precision: Option<f64>,
    winner_mean_attack_kills: f64,
    winner_distinct_attack_victims: f64,
    mean_distinct_attack_victims: f64,
    winner_attack_target_evaded: f64,
    mean_attack_target_evaded: f64,
    winner_action_fractions: [f64; 5],
    mean_action_fractions: [f64; 5],
    winner_commands_per_tick: f64,
    mean_commands_per_tick: f64,
    winner_multi_command_tick_fraction: f64,
    mean_multi_command_tick_fraction: f64,
    winner_spatial_coverage: f64,
    mean_spatial_coverage: f64,
    winner_end_survival_fraction: f64,
    mean_end_survival_fraction: f64,
    mean_opponent_score_stddev: Option<f64>,
    max_opponent_score_stddev: Option<f64>,
    species: usize,
    winner_expressed_hidden_nodes: usize,
    winner_expressed_connections: usize,
    mean_expressed_hidden_nodes: f64,
    mean_expressed_connections: f64,
    new_connection_innovations: usize,
    new_node_innovations: usize,
}

impl CompactGeneration {
    fn from_summary(g: &GenerationSummary) -> Self {
        Self {
            generation: g.generation,
            winner_contextual_score: g.winner_contextual_score,
            winner_case_score_stddev: g.winner_case_score_stddev,
            mean_case_score_stddev: g.mean_case_score_stddev,
            max_case_score_stddev: g.max_case_score_stddev,
            winner_absolute_survival: g.winner_absolute_survival_fraction,
            mean_absolute_survival: g.mean_absolute_survival_fraction,
            winner_alive_ticks: g.winner_candidate_alive_ticks,
            mean_alive_ticks: g.mean_candidate_alive_ticks,
            winner_relative_advantage: g.winner_relative_survival_advantage,
            winner_total_energy_accumulated: g.winner_gross_energy_acquired,
            mean_total_energy_accumulated: g.mean_gross_energy_acquired,
            median_total_energy_accumulated: g.median_gross_energy_acquired,
            winner_net_energy_profit: g.winner_net_energy_profit,
            mean_net_energy_profit: g.mean_net_energy_profit,
            median_net_energy_profit: g.median_net_energy_profit,
            winner_attack_energy_received: g.winner_attack_energy_received,
            winner_attack_energy_lost: g.winner_attack_energy_lost,
            winner_attack_attempt_energy_cost: g.winner_attack_attempt_energy_cost,
            winner_net_attack_energy: g.winner_net_attack_energy_balance,
            mean_attack_energy_received: g.mean_attack_energy_received,
            mean_attack_energy_lost: g.mean_attack_energy_lost,
            mean_attack_attempt_energy_cost: g.mean_attack_attempt_energy_cost,
            mean_net_attack_energy: g.mean_net_attack_energy_balance,
            winner_action_effectiveness: g.winner_action_effectiveness,
            mean_action_effectiveness: g.mean_action_effectiveness,
            winner_successful_attack_rate: g.winner_successful_attack_rate,
            mean_successful_attack_rate: g.mean_successful_attack_rate,
            winner_attack_precision: g.winner_attack_precision,
            population_attack_precision: g.population_attack_precision,
            winner_mean_attack_kills: g.winner_mean_attack_kills,
            winner_distinct_attack_victims: g.winner_distinct_attack_victims,
            mean_distinct_attack_victims: g.mean_distinct_attack_victims,
            winner_attack_target_evaded: g.winner_attack_target_evaded,
            mean_attack_target_evaded: g.mean_attack_target_evaded,
            winner_action_fractions: g.winner_action_fractions,
            mean_action_fractions: g.mean_action_fractions,
            winner_commands_per_tick: g.winner_commands_per_tick,
            mean_commands_per_tick: g.mean_commands_per_tick,
            winner_multi_command_tick_fraction: g.winner_multi_command_tick_fraction,
            mean_multi_command_tick_fraction: g.mean_multi_command_tick_fraction,
            winner_spatial_coverage: g.winner_spatial_coverage,
            mean_spatial_coverage: g.mean_spatial_coverage,
            winner_end_survival_fraction: g.winner_end_survival_fraction,
            mean_end_survival_fraction: g.mean_end_survival_fraction,
            mean_opponent_score_stddev: g.mean_opponent_score_stddev,
            max_opponent_score_stddev: g.max_opponent_score_stddev,
            species: g.species.len(),
            winner_expressed_hidden_nodes: g.winner_expressed_hidden_nodes,
            winner_expressed_connections: g.winner_expressed_connections,
            mean_expressed_hidden_nodes: g.mean_expressed_hidden_nodes,
            mean_expressed_connections: g.mean_expressed_connections,
            new_connection_innovations: g.new_connection_innovations,
            new_node_innovations: g.new_node_innovations,
        }
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
        final_generation: trajectory
            .last()
            .map_or(0, |generation| generation.generation),
        final_winner_contextual_score: trajectory
            .last()
            .map_or(0.0, |generation| generation.winner_contextual_score),
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
        metric!("winner_contextual_score", winner_contextual_score);
        metric!("winner_case_score_stddev", winner_case_score_stddev);
        metric!("mean_case_score_stddev", mean_case_score_stddev);
        metric!("max_case_score_stddev", max_case_score_stddev);
        metric!("winner_absolute_survival", winner_absolute_survival);
        metric!("mean_absolute_survival", mean_absolute_survival);
        metric!("winner_alive_ticks", winner_alive_ticks);
        metric!("mean_alive_ticks", mean_alive_ticks);
        metric!("winner_relative_advantage", winner_relative_advantage);
        metric!(
            "winner_total_energy_accumulated",
            winner_total_energy_accumulated
        );
        metric!(
            "mean_total_energy_accumulated",
            mean_total_energy_accumulated
        );
        metric!(
            "median_total_energy_accumulated",
            median_total_energy_accumulated
        );
        metric!("winner_net_energy_profit", winner_net_energy_profit);
        metric!("mean_net_energy_profit", mean_net_energy_profit);
        metric!("median_net_energy_profit", median_net_energy_profit);
        metric!(
            "winner_attack_energy_received",
            winner_attack_energy_received
        );
        metric!("winner_attack_energy_lost", winner_attack_energy_lost);
        metric!(
            "winner_attack_attempt_energy_cost",
            winner_attack_attempt_energy_cost
        );
        metric!("winner_net_attack_energy", winner_net_attack_energy);
        optional_metric!("winner_action_effectiveness", winner_action_effectiveness);
        optional_metric!("mean_action_effectiveness", mean_action_effectiveness);
        optional_metric!(
            "winner_successful_attack_rate",
            winner_successful_attack_rate
        );
        optional_metric!("mean_successful_attack_rate", mean_successful_attack_rate);
        optional_metric!("winner_attack_precision", winner_attack_precision);
        optional_metric!("population_attack_precision", population_attack_precision);
        metric!("winner_spatial_coverage", winner_spatial_coverage);
        metric!("mean_spatial_coverage", mean_spatial_coverage);
        metric!("winner_end_survival_fraction", winner_end_survival_fraction);
        metric!("mean_end_survival_fraction", mean_end_survival_fraction);
        metric!("winner_commands_per_tick", winner_commands_per_tick);
        metric!("mean_commands_per_tick", mean_commands_per_tick);
        metric!(
            "winner_multi_command_tick_fraction",
            winner_multi_command_tick_fraction
        );
        metric!(
            "mean_multi_command_tick_fraction",
            mean_multi_command_tick_fraction
        );
        metric!("winner_attack_target_evaded", winner_attack_target_evaded);
        metric!("mean_attack_target_evaded", mean_attack_target_evaded);
        for (name, action_index) in [
            ("winner_action_idle_fraction", 0),
            ("winner_action_turn_left_fraction", 1),
            ("winner_action_turn_right_fraction", 2),
            ("winner_action_forward_fraction", 3),
            ("winner_action_attack_fraction", 4),
        ] {
            metrics.insert(
                name,
                distribution(
                    generations
                        .iter()
                        .map(|row| row.winner_action_fractions[action_index]),
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
    json!({
        "start_generation": first.generation,
        "end_generation": last.generation,
        "samples": tail.len(),
        "score_semantics": "contextual_within_generation_only",
        "longitudinal_validation": "use_crossplay",
        "generation_contexts": tail.iter().map(|generation| json!({
            "generation": generation.generation,
            "winner_contextual_score": generation.winner_contextual_score,
            "winner_case_score_stddev": generation.winner_case_score_stddev,
            "winner_action_effectiveness": generation.winner_action_effectiveness,
            "winner_successful_attack_rate": generation.winner_successful_attack_rate,
            "winner_net_energy_profit": generation.winner_net_energy_profit,
        })).collect::<Vec<_>>(),
    })
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
        "evaluation_scenarios": result.evaluation_scenarios,
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
