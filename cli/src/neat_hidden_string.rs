use crate::{neat, run_output_path};
use anyhow::{anyhow, bail, Result};
use evolution::{
    run_neat_controlled,
    tasks::hidden_string::{
        HiddenStringCondition, HiddenStringEvaluation, HiddenStringRunResult, HiddenStringTask,
        HiddenStringTaskConfig, DEFAULT_TARGET_PANEL_SEED, OBJECTIVE_NAME, TASK_NAME,
    },
    EvaluationPool, FrozenGenomeArtifact, NeatCheckpoint, NeatConfig, RunTerminationStatus,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use signal_hook::consts::{SIGINT, SIGTERM};
use std::cell::RefCell;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::time::Instant;
use types::{OrganismGenome, SeedGenomeConfig};
use views::atomic_write;

const DEFAULT_SEED_CONFIG: &str = config::CANONICAL_SEED_GENOME_CONFIG_PATH;

struct Request {
    seed_config_path: String,
    run_seed: u64,
    neat: NeatConfig,
    task: HiddenStringTask,
    thresholds: Vec<f64>,
}

type HiddenStringCheckpoint = NeatCheckpoint<HiddenStringTaskConfig, HiddenStringEvaluation>;
type HiddenStringFrozenGenome =
    FrozenGenomeArtifact<HiddenStringTaskConfig, HiddenStringEvaluation>;

pub(crate) fn run_cli(args: &[&str], out_dir: &str, out: &mut impl Write) -> Result<()> {
    if args.first() == Some(&"run") {
        return run_cli(&args[1..], out_dir, out);
    }
    if args.first() == Some(&"plan") {
        return plan(&args[1..], out);
    }
    if args.first() == Some(&"analyze") {
        return analyze(&args[1..], out);
    }
    if args.first() == Some(&"calibrate") {
        return calibrate(&args[1..], out);
    }
    if args.first() == Some(&"horizon") {
        return horizon_sweep(&args[1..], out);
    }
    if args.first() == Some(&"resume") {
        return resume(&args[1..], out);
    }
    if args.first() == Some(&"reevaluate") {
        return reevaluate(&args[1..], out);
    }
    let Some(request) = parse(args, out)? else {
        return Ok(());
    };
    execute(request, out_dir, out)
}

fn parse(args: &[&str], out: &mut impl Write) -> Result<Option<Request>> {
    let mut seed_config_path = DEFAULT_SEED_CONFIG.to_owned();
    let mut run_seed = 0_u64;
    let mut neat_config = NeatConfig::default();
    let task = HiddenStringTask::default();
    let mut thresholds = Vec::<f64>::new();
    let mut index = 0;
    while index < args.len() {
        match args[index] {
            "--seed-config" => {
                seed_config_path = value(args, index, "--seed-config")?.to_owned();
                index += 2;
            }
            "--seed" => {
                run_seed = value(args, index, "--seed")?.parse()?;
                index += 2;
            }
            "--population" => {
                neat_config.population_size = value(args, index, "--population")?.parse()?;
                index += 2;
            }
            "--generations" => {
                neat_config.generations = value(args, index, "--generations")?.parse()?;
                index += 2;
            }
            "--population-checkpoint-interval" => {
                neat_config.population_checkpoint_interval =
                    value(args, index, "--population-checkpoint-interval")?.parse()?;
                index += 2;
            }
            "--workers" => {
                neat_config.evaluation_workers = value(args, index, "--workers")?.parse()?;
                index += 2;
            }
            "--threshold" => {
                thresholds.push(value(args, index, "--threshold")?.parse()?);
                index += 2;
            }
            "--param" => {
                let raw = value(args, index, "--param")?;
                let (key, value) = raw
                    .split_once('=')
                    .ok_or_else(|| anyhow!("expected key=value, got `{raw}`"))?;
                neat::apply_neat_param(&mut neat_config, key.trim(), value.trim())?;
                index += 2;
            }
            "--help" | "-h" => {
                help(out)?;
                return Ok(None);
            }
            other => bail!("unknown hidden-string argument `{other}`"),
        }
    }
    neat_config.validate()?;
    task.config.validate()?;
    if thresholds.is_empty() {
        thresholds = vec![0.2, 0.5, 0.8, 0.9];
    }
    if thresholds
        .iter()
        .any(|threshold| !threshold.is_finite() || !(0.0..=1.0).contains(threshold))
    {
        bail!("--threshold must be a finite value in [0, 1]");
    }
    Ok(Some(Request {
        seed_config_path,
        run_seed,
        neat: neat_config,
        task,
        thresholds,
    }))
}

fn help(out: &mut impl Write) -> Result<()> {
    writeln!(
        out,
        "Hidden-string adaptation: cli hidden-string [run|plan] [OPTIONS]\n\
         Analyze: cli hidden-string analyze RESULT...\n\
         Calibrate: cli hidden-string calibrate RESULT --generations G,G,G [--workers N] [--out PATH]\n\
         Horizon: cli hidden-string horizon RESULT [--targets N] [--rollouts N] [--workers N] [--out PATH]\n\
         Resume: cli hidden-string resume CHECKPOINT [--generations N]\n\
         Reevaluate: cli hidden-string reevaluate FROZEN [--attempts N,N] [--panel training|development|sealed|custom] [--condition NAME]\n\
         \n\
         --seed N                    evolutionary run seed\n\
         --population N              genomes per generation\n\
         --generations N             generations to evaluate\n\
         --population-checkpoint-interval N\n\
                                     persist every Nth full population\n\
         --workers N                 parallel genome evaluations\n\
         --threshold F               repeatable normalized-fitness threshold\n\
         --seed-config PATH          seed genome TOML\n\
         --param key=value           NEAT override (valid: {})\n\
         \n\
         REEVALUATE OPTIONS\n\
         --attempts N,N              one or more inference horizons\n\
         --panel NAME                training|development|sealed|custom\n\
         --panel-seed N --targets N  required provenance for a custom panel\n\
         --rollouts N                generate N deterministic rollout seeds\n\
         --rollout-seed N            explicit repeatable rollout seed\n\
         --condition NAME            primary|plasticity-off|shuffled-reward|reset-weights\n\
         --out PATH                  optional JSON report path",
        neat::PARAMS
    )?;
    Ok(())
}

fn plan(args: &[&str], out: &mut impl Write) -> Result<()> {
    let Some(request) = parse(args, out)? else {
        return Ok(());
    };
    let seed_genome = load_seed_config(&request.seed_config_path)?;
    let (training_panel, development_panel, sealed_panel) = request.task.target_panel_summaries();
    let training_targets = request.task.config.training_target_count as u128;
    let attempts = u128::from(request.task.config.attempts);
    let training_rollouts = request.task.config.training_rollout_seeds.len() as u128;
    let outputs_per_attempt = 4_u128;
    let genome_evaluations =
        request.neat.population_size as u128 * request.neat.generations as u128;
    writeln!(
        out,
        "{}",
        json!({
            "mode": "neat_plan",
            "valid": true,
            "task": TASK_NAME,
            "objective": OBJECTIVE_NAME,
            "run_seed": request.run_seed,
            "population": request.neat.population_size,
            "generations": request.neat.generations,
            "neat_config": request.neat,
            "genome_evaluations": genome_evaluations,
            "training_panel": training_panel,
            "development_panel": development_panel,
            "sealed_panel": sealed_panel,
            "development_interval": request.task.config.development_interval,
            "attempts_per_target": attempts,
            "outputs_per_attempt": outputs_per_attempt,
            "training_action_decisions": genome_evaluations * training_targets * training_rollouts * attempts * outputs_per_attempt,
            "training_probe_schedule": request.task.config.training_probe_after_attempts,
            "report_probe_schedule": request.task.config.report_probe_after_attempts,
            "sensors": "all_zero",
            "actions": ["a", "b", "c", "d", "e", "f", "g", "h"],
            "selection_fitness": "mean final-probe greedy longest-correct-prefix length divided by 4",
            "normalized_fitness": "hard greedy exact-string rate",
            "seed_genome": seed_genome,
            "normalized_fitness_thresholds": request.thresholds,
        })
    )?;
    Ok(())
}

fn execute(request: Request, out_dir: &str, out: &mut impl Write) -> Result<()> {
    let seed_genome = load_seed_config(&request.seed_config_path)?;
    execute_with_state(request, seed_genome, None, None, out_dir, out)
}

fn execute_with_state(
    request: Request,
    seed_genome: SeedGenomeConfig,
    resume_from: Option<HiddenStringCheckpoint>,
    existing_run_dir: Option<PathBuf>,
    out_dir: &str,
    out: &mut impl Write,
) -> Result<()> {
    let total_generations = request.neat.generations;
    let session_start_generation = resume_from
        .as_ref()
        .map(|checkpoint| checkpoint.next_generation)
        .unwrap_or(0);
    let started = Instant::now();
    let run_dir = if let Some(run_dir) = existing_run_dir {
        run_dir
    } else {
        let unique = run_output_path(out_dir, "neat-hidden-string-run")?;
        unique.with_extension("")
    };
    std::fs::create_dir_all(&run_dir).map_err(|error| {
        anyhow!(
            "cannot create hidden-string run directory `{}`: {error}",
            run_dir.display()
        )
    })?;
    let store = RefCell::new(ArtifactStore::new(run_dir.clone()));
    store.borrow().persist_manifest("running", None)?;
    let signal = Arc::new(AtomicUsize::new(0));
    signal_hook::flag::register_usize(SIGINT, Arc::clone(&signal), SIGINT as usize)?;
    signal_hook::flag::register_usize(SIGTERM, Arc::clone(&signal), SIGTERM as usize)?;
    eprintln!(
        "{}",
        json!({
            "event": "neat_started",
            "task": TASK_NAME,
            "population": request.neat.population_size,
            "generations": total_generations,
            "training_targets": request.task.config.training_target_count,
            "development_targets": request.task.config.development_target_count,
            "sealed_targets": request.task.config.sealed_target_count,
            "attempts": request.task.config.attempts,
            "training_rollouts": request.task.config.training_rollout_seeds.len(),
            "development_interval": request.task.config.development_interval,
            "evaluation_workers": request.neat.evaluation_workers,
            "run_dir": run_dir,
            "resumed": resume_from.is_some(),
        })
    );
    let result = run_neat_controlled(
        &request.task,
        request.neat,
        seed_genome,
        request.run_seed,
        resume_from,
        request.thresholds,
        false,
        |generation| {
            let training = &generation.winner_evaluation;
            let development = generation.winner_validation.as_ref();
            let elapsed = started.elapsed().as_secs_f64();
            let completed = generation.generation + 1;
            let completed_this_session = completed.saturating_sub(session_start_generation);
            let eta = elapsed / f64::from(completed_this_session.max(1))
                * f64::from(total_generations.saturating_sub(completed));
            eprintln!(
                "{}",
                json!({
                    "event": "neat_generation",
                    "generation": generation.generation,
                    "training_fitness": generation.winner_fitness,
                    "training_final_accuracy": training.primary.final_accuracy,
                    "training_final_exact_string_rate": training.primary.final_exact_string_rate,
                    "training_final_greedy_prefix_score": training.primary.final_greedy_prefix_score,
                    "development_final_accuracy": development.map(|evaluation| evaluation.primary.final_accuracy),
                    "development_final_exact_string_rate": development.map(|evaluation| evaluation.primary.final_exact_string_rate),
                    "development_final_greedy_prefix_score": development.map(|evaluation| evaluation.primary.final_greedy_prefix_score),
                    "learning_rate": training.learning_rate,
                    "species": generation.species.len(),
                    "hidden_nodes": generation.winner_hidden_nodes,
                    "enabled_connections": generation.winner_enabled_connections,
                    "elapsed_seconds": elapsed,
                    "eta_seconds": eta,
                    "population_genome_evaluations": generation.work.population.genome_evaluations,
                    "population_brain_synapse_operations": generation.work.population.brain_synapse_operations,
                    "generation_wall_time_seconds": generation.wall_time_seconds,
                })
            );
        },
        |checkpoint| store.borrow_mut().persist_checkpoint(checkpoint),
        |champion| store.borrow_mut().persist_historical_champion(champion),
        || match signal.load(Ordering::Relaxed) as i32 {
            SIGINT => Some("signal:SIGINT".to_owned()),
            SIGTERM => Some("signal:SIGTERM".to_owned()),
            _ => None,
        },
    )?;

    let result_path = run_dir.join("result.json.zst");
    let result_path_string = result_path.to_string_lossy().into_owned();
    persist_compressed_json(&result_path, &result)?;
    let terminal = terminal_frozen_genome(&result)?;
    store.borrow_mut().persist_terminal_champion(&terminal)?;
    let status = match result.termination.status {
        RunTerminationStatus::Completed => "completed",
        RunTerminationStatus::EarlyStopped => "early_stopped",
    };
    store
        .borrow()
        .persist_manifest(status, Some(&result_path))?;
    writeln!(out, "{}", summarize_result(&result, &result_path_string)?)?;
    Ok(())
}

struct ArtifactStore {
    run_dir: PathBuf,
    latest_checkpoint: Option<PathBuf>,
    historical_champions: Vec<PathBuf>,
    terminal_champion: Option<PathBuf>,
}

impl ArtifactStore {
    fn new(run_dir: PathBuf) -> Self {
        let mut checkpoints = directory_files(&run_dir.join("checkpoints"), ".checkpoint.json.zst");
        let mut historical_champions =
            directory_files(&run_dir.join("champions"), ".frozen.json.zst");
        historical_champions.retain(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("historical-generation-"))
        });
        let terminal = run_dir.join("champions/terminal.frozen.json.zst");
        Self {
            run_dir,
            latest_checkpoint: checkpoints.pop(),
            historical_champions,
            terminal_champion: terminal.exists().then_some(terminal),
        }
    }

    fn persist_checkpoint(&mut self, checkpoint: &HiddenStringCheckpoint) -> Result<()> {
        let path = self.run_dir.join("checkpoints").join(format!(
            "generation-{:06}.checkpoint.json.zst",
            checkpoint.next_generation
        ));
        persist_compressed_json(&path, checkpoint)?;
        let pointer = self.run_dir.join("checkpoints/latest.json");
        persist_plain_json(
            &pointer,
            &json!({
                "checkpoint_schema_version": checkpoint.checkpoint_schema_version,
                "next_generation": checkpoint.next_generation,
                "path": path,
            }),
        )?;
        self.latest_checkpoint = Some(path);
        self.persist_manifest("running", None)
    }

    fn persist_historical_champion(&mut self, champion: &HiddenStringFrozenGenome) -> Result<()> {
        let path = self.run_dir.join("champions").join(format!(
            "historical-generation-{:06}.frozen.json.zst",
            champion.source_generation
        ));
        persist_compressed_json(&path, champion)?;
        self.historical_champions.push(path);
        self.persist_manifest("running", None)
    }

    fn persist_terminal_champion(&mut self, champion: &HiddenStringFrozenGenome) -> Result<()> {
        let path = self.run_dir.join("champions/terminal.frozen.json.zst");
        persist_compressed_json(&path, champion)?;
        self.terminal_champion = Some(path);
        Ok(())
    }

    fn persist_manifest(&self, status: &str, result: Option<&Path>) -> Result<()> {
        persist_plain_json(
            &self.run_dir.join("manifest.json"),
            &json!({
                "run_artifact_schema_version": 1,
                "task": TASK_NAME,
                "status": status,
                "latest_checkpoint": self.latest_checkpoint,
                "historical_champions": self.historical_champions,
                "terminal_champion": self.terminal_champion,
                "result": result,
            }),
        )
    }
}

fn directory_files(directory: &Path, suffix: &str) -> Vec<PathBuf> {
    let mut files = std::fs::read_dir(directory)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.ends_with(suffix))
        })
        .collect::<Vec<_>>();
    files.sort();
    files
}

fn terminal_frozen_genome(result: &HiddenStringRunResult) -> Result<HiddenStringFrozenGenome> {
    let generation = result
        .generations
        .last()
        .ok_or_else(|| anyhow!("cannot freeze terminal winner without a generation"))?;
    let winner = result
        .final_population
        .get(result.final_winner_population_index)
        .ok_or_else(|| anyhow!("cannot freeze missing terminal winner"))?;
    Ok(HiddenStringFrozenGenome {
        frozen_genome_schema_version: 1,
        task: result.task.clone(),
        objective: result.objective.clone(),
        task_config: result.task_config.clone(),
        run_seed: result.seed,
        source_generation: generation.generation,
        source_population_index: winner.population_index,
        role: "terminal_winner".to_owned(),
        fitness: winner.fitness,
        normalized_fitness: Some(winner.evaluation.primary.final_exact_string_rate),
        hidden_nodes: winner.genome.hidden_node_count(),
        enabled_connections: winner.genome.enabled_connection_count(),
        training_evaluation: winner.evaluation.clone(),
        genome: winner.genome.clone(),
    })
}

fn resume(args: &[&str], out: &mut impl Write) -> Result<()> {
    let mut source = None;
    let mut generations = None;
    let mut index = 0;
    while index < args.len() {
        match args[index] {
            "--generations" => {
                generations = Some(value(args, index, "--generations")?.parse()?);
                index += 2;
            }
            value if !value.starts_with('-') && source.is_none() => {
                source = Some(value.to_owned());
                index += 1;
            }
            other => bail!("unknown hidden-string resume argument `{other}`"),
        }
    }
    let source = source.ok_or_else(|| anyhow!("resume needs a checkpoint artifact"))?;
    let mut checkpoint: HiddenStringCheckpoint = serde_json::from_reader(result_reader(&source)?)
        .map_err(|error| {
        anyhow!("cannot parse hidden-string checkpoint `{source}`: {error}")
    })?;
    if checkpoint.checkpoint_schema_version != 1
        || checkpoint.boundary != "before_generation_evaluation"
    {
        bail!("`{source}` is not a supported before-generation checkpoint");
    }
    if checkpoint.task != TASK_NAME || checkpoint.objective != OBJECTIVE_NAME {
        bail!(
            "checkpoint task `{}` / objective `{}` is incompatible with hidden-string `{TASK_NAME}` / `{OBJECTIVE_NAME}`",
            checkpoint.task,
            checkpoint.objective
        );
    }
    let target_generations = generations.unwrap_or(checkpoint.neat_config.generations);
    if target_generations <= checkpoint.next_generation {
        bail!(
            "resume --generations {target_generations} must exceed checkpoint next_generation {}",
            checkpoint.next_generation
        );
    }
    if target_generations > checkpoint.neat_config.generations
        && checkpoint.next_generation == checkpoint.neat_config.generations
        && !checkpoint
            .next_generation
            .is_multiple_of(checkpoint.task_config.development_interval)
    {
        // The old terminal generation received development only because it was
        // terminal. Remove that observational assay when extending the budget
        // so the resumed summaries match an uninterrupted longer run.
        if let Some(last) = checkpoint.generations.last_mut() {
            last.winner_validation = None;
            checkpoint
                .deterministic_work
                .winner_validation
                .genome_evaluations = checkpoint
                .deterministic_work
                .winner_validation
                .genome_evaluations
                .saturating_sub(last.work.winner_validation.genome_evaluations);
            checkpoint
                .deterministic_work
                .winner_validation
                .brain_synapse_operations = checkpoint
                .deterministic_work
                .winner_validation
                .brain_synapse_operations
                .saturating_sub(last.work.winner_validation.brain_synapse_operations);
            last.work.winner_validation = evolution::WorkBreakdown::default();
        }
    }
    let task = HiddenStringTask::new(checkpoint.task_config.clone())?;
    let mut neat = checkpoint.neat_config.clone();
    neat.generations = target_generations;
    let seed_genome = checkpoint.seed_genome_config.clone();
    let run_seed = checkpoint.seed;
    let thresholds = checkpoint.thresholds.clone();
    let run_dir = Path::new(&source)
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| anyhow!("checkpoint `{source}` is not inside RUN/checkpoints/"))?
        .to_path_buf();
    if let Some(latest) =
        directory_files(&run_dir.join("checkpoints"), ".checkpoint.json.zst").pop()
    {
        let requested_name = Path::new(&source).file_name();
        if latest.file_name() != requested_name {
            bail!(
                "checkpoint `{source}` is not the run's latest checkpoint `{}`; refusing to overwrite a later continuation",
                latest.display()
            );
        }
    }
    let request = Request {
        seed_config_path: "<checkpoint>".to_owned(),
        run_seed,
        neat,
        task,
        thresholds,
    };
    execute_with_state(
        request,
        seed_genome,
        Some(checkpoint),
        Some(run_dir.clone()),
        run_dir.to_string_lossy().as_ref(),
        out,
    )
}

fn reevaluate(args: &[&str], out: &mut impl Write) -> Result<()> {
    let mut source = None;
    let mut attempts = None;
    let mut panel = "sealed".to_owned();
    let mut panel_seed = None;
    let mut target_count = None;
    let mut rollout_count = None;
    let mut explicit_rollout_seeds = Vec::new();
    let mut conditions = Vec::new();
    let mut output_path = None;
    let mut index = 0;
    while index < args.len() {
        match args[index] {
            "--attempts" => {
                attempts = Some(parse_u32_list(value(args, index, "--attempts")?)?);
                index += 2;
            }
            "--panel" => {
                panel = value(args, index, "--panel")?.to_owned();
                index += 2;
            }
            "--panel-seed" => {
                panel_seed = Some(value(args, index, "--panel-seed")?.parse()?);
                index += 2;
            }
            "--targets" => {
                target_count = Some(value(args, index, "--targets")?.parse()?);
                index += 2;
            }
            "--rollouts" => {
                rollout_count = Some(value(args, index, "--rollouts")?.parse()?);
                index += 2;
            }
            "--rollout-seed" => {
                explicit_rollout_seeds.push(value(args, index, "--rollout-seed")?.parse()?);
                index += 2;
            }
            "--condition" => {
                for raw in value(args, index, "--condition")?.split(',') {
                    conditions.push(parse_condition(raw.trim())?);
                }
                index += 2;
            }
            "--out" => {
                output_path = Some(value(args, index, "--out")?.to_owned());
                index += 2;
            }
            value if !value.starts_with('-') && source.is_none() => {
                source = Some(value.to_owned());
                index += 1;
            }
            other => bail!("unknown hidden-string reevaluate argument `{other}`"),
        }
    }
    let source = source.ok_or_else(|| anyhow!("reevaluate needs a frozen-genome artifact"))?;
    let frozen: HiddenStringFrozenGenome = serde_json::from_reader(result_reader(&source)?)
        .map_err(|error| anyhow!("cannot parse frozen genome `{source}`: {error}"))?;
    if frozen.frozen_genome_schema_version != 1 || frozen.task != TASK_NAME {
        bail!("`{source}` is not a supported hidden-string frozen genome");
    }
    if rollout_count.is_some() && !explicit_rollout_seeds.is_empty() {
        bail!("use either --rollouts or repeatable --rollout-seed, not both");
    }
    if conditions.is_empty() {
        conditions.push(HiddenStringCondition::Primary);
    }
    conditions.dedup();
    let mut horizons = attempts.unwrap_or_else(|| vec![frozen.task_config.attempts]);
    horizons.sort_unstable();
    horizons.dedup();
    if horizons.is_empty() || horizons.contains(&0) {
        bail!("--attempts must contain positive horizons");
    }
    let custom = panel == "custom";
    if custom && (panel_seed.is_none() || target_count.is_none()) {
        bail!("--panel custom requires explicit --panel-seed and --targets");
    }
    if !custom && (panel_seed.is_some() || target_count.is_some()) {
        bail!("--panel-seed and --targets are only valid with --panel custom");
    }
    if !matches!(
        panel.as_str(),
        "training" | "development" | "sealed" | "custom"
    ) {
        bail!("--panel must be training, development, sealed, or custom");
    }

    let mut measurements = Vec::with_capacity(horizons.len());
    for horizon in &horizons {
        let mut config = frozen.task_config.clone();
        config.attempts = *horizon;
        config.training_probe_after_attempts = vec![*horizon];
        config.report_probe_after_attempts = vec![0, *horizon];
        let resolved_panel = if custom {
            config.target_panel_seed = panel_seed.expect("validated custom seed");
            config.training_target_count = target_count.expect("validated custom count");
            "training"
        } else {
            panel.as_str()
        };
        let selected_rollout_seeds = if !explicit_rollout_seeds.is_empty() {
            explicit_rollout_seeds.clone()
        } else if let Some(count) = rollout_count {
            if count == 0 {
                bail!("--rollouts must be positive");
            }
            reevaluation_rollout_seeds(count)
        } else {
            match resolved_panel {
                "training" => config.training_rollout_seeds.clone(),
                "development" => config.development_rollout_seeds.clone(),
                "sealed" => config.sealed_rollout_seeds.clone(),
                _ => unreachable!(),
            }
        };
        match resolved_panel {
            "training" => config.training_rollout_seeds = selected_rollout_seeds.clone(),
            "development" => config.development_rollout_seeds = selected_rollout_seeds.clone(),
            "sealed" => config.sealed_rollout_seeds = selected_rollout_seeds.clone(),
            _ => unreachable!(),
        }
        let task = HiddenStringTask::new(config.clone())?;
        let evaluations =
            task.evaluate_frozen_conditions(&frozen.genome, resolved_panel, &conditions)?;
        measurements.push(json!({
            "attempts": horizon,
            "resolved_task_config": config,
            "evaluations": evaluations,
        }));
    }
    let value = json!({
        "mode": "hidden_string_frozen_genome_reevaluation",
        "source": source,
        "source_contract": {
            "task": frozen.task,
            "objective": frozen.objective,
            "run_seed": frozen.run_seed,
            "source_generation": frozen.source_generation,
            "source_population_index": frozen.source_population_index,
            "role": frozen.role,
            "original_task_config": frozen.task_config,
        },
        "resolved_panel": {
            "name": panel,
            "custom_panel_seed": panel_seed,
            "custom_target_count": target_count,
            "rollout_count_override": rollout_count,
            "explicit_rollout_seeds": explicit_rollout_seeds,
        },
        "conditions": conditions,
        "horizons": horizons,
        "measurements": measurements,
    });
    persist_json(output_path.as_deref(), &value)?;
    writeln!(out, "{value}")?;
    Ok(())
}

fn parse_condition(raw: &str) -> Result<HiddenStringCondition> {
    match raw {
        "primary" => Ok(HiddenStringCondition::Primary),
        "plasticity-off" | "plasticity_off" => Ok(HiddenStringCondition::PlasticityOff),
        "shuffled-reward" | "shuffled_reward" | "permuted-reward" => {
            Ok(HiddenStringCondition::ShuffledReward)
        }
        "reset-weights" | "reset_weights_each_attempt" => {
            Ok(HiddenStringCondition::ResetWeightsEachAttempt)
        }
        other => bail!("unknown reevaluation condition `{other}`"),
    }
}

fn reevaluation_rollout_seeds(count: usize) -> Vec<u64> {
    (0..count)
        .map(|index| 0x5245_4556_414c_0001_u64.wrapping_add(index as u64))
        .collect()
}

fn analyze(args: &[&str], out: &mut impl Write) -> Result<()> {
    if args.is_empty() {
        bail!("analyze needs at least one result path");
    }
    let analyses = args
        .iter()
        .map(|path| {
            let result: HiddenStringRunResult = serde_json::from_reader(result_reader(path)?)
                .map_err(|error| anyhow!("cannot parse hidden-string result `{path}`: {error}"))?;
            summarize_result(&result, path)
        })
        .collect::<Result<Vec<_>>>()?;
    let output = if analyses.len() == 1 {
        analyses.into_iter().next().expect("one analysis")
    } else {
        json!({ "runs": analyses })
    };
    writeln!(out, "{output}")?;
    Ok(())
}

fn summarize_result(result: &HiddenStringRunResult, path: &str) -> Result<Value> {
    let winner = result
        .final_population
        .get(result.final_winner_population_index)
        .ok_or_else(|| anyhow!("hidden-string result has no final winner"))?;
    let development = result
        .final_winner_validation
        .as_ref()
        .ok_or_else(|| anyhow!("hidden-string result has no development evaluation"))?;
    let sealed = result
        .final_winner_final_evaluation
        .as_ref()
        .ok_or_else(|| anyhow!("hidden-string result has no sealed evaluation"))?;
    Ok(json!({
        "path": path,
        "run_seed": result.seed,
        "task": result.task,
        "objective": result.objective,
        "population": result.neat_config.population_size,
        "generations": result.generations.len(),
        "termination": result.termination,
        "threshold_events": result.threshold_events,
        "deterministic_work": result.deterministic_work,
        "total_work": result.total_work,
        "session_timings": result.session_timings,
        "total_wall_time_seconds": result.total_wall_time_seconds,
        "winner": {
            "population_index": winner.population_index,
            "fitness": winner.fitness,
            "learning_rate": winner.evaluation.learning_rate,
            "hidden_nodes": result.generations.last().map(|generation| generation.winner_hidden_nodes),
            "enabled_connections": result.generations.last().map(|generation| generation.winner_enabled_connections),
        },
        "training": metric_summary(&winner.evaluation),
        "development": metric_summary(development),
        "sealed": metric_summary(sealed),
        "sealed_controls": control_summary(sealed),
        "sealed_probes": sealed.primary.probes,
        "trajectory": result.generations.iter().map(|generation| json!({
            "generation": generation.generation,
            "training_final_accuracy": generation.winner_evaluation.primary.final_accuracy,
            "training_final_exact_string_rate": generation.winner_evaluation.primary.final_exact_string_rate,
            "training_final_greedy_prefix_score": generation.winner_evaluation.primary.final_greedy_prefix_score,
            "development_final_accuracy": generation.winner_validation.as_ref().map(|evaluation| evaluation.primary.final_accuracy),
            "development_final_exact_string_rate": generation.winner_validation.as_ref().map(|evaluation| evaluation.primary.final_exact_string_rate),
            "development_final_greedy_prefix_score": generation.winner_validation.as_ref().map(|evaluation| evaluation.primary.final_greedy_prefix_score),
            "learning_rate": generation.winner_evaluation.learning_rate,
            "species": generation.species.len(),
        })).collect::<Vec<_>>(),
    }))
}

fn metric_summary(evaluation: &HiddenStringEvaluation) -> Value {
    let initial_exact_string_rate = evaluation
        .primary
        .probes
        .iter()
        .find(|probe| probe.after_attempts == 0)
        .map(|probe| probe.exact_string_rate);
    json!({
        "fitness": evaluation.fitness,
        "pre_learning_accuracy": evaluation.primary.pre_learning_accuracy,
        "final_accuracy": evaluation.primary.final_accuracy,
        "adaptation_gain": evaluation.primary.pre_learning_accuracy.map(|pre| evaluation.primary.final_accuracy - pre),
        "final_exact_string_rate": evaluation.primary.final_exact_string_rate,
        "final_greedy_prefix_score": evaluation.primary.final_greedy_prefix_score,
        "exact_string_adaptation_gain": initial_exact_string_rate.map(|pre| evaluation.primary.final_exact_string_rate - pre),
    })
}

fn control_summary(evaluation: &HiddenStringEvaluation) -> Value {
    evaluation
        .controls
        .as_ref()
        .map_or(Value::Null, |controls| {
            json!({
                "shuffled_reward_final_accuracy": controls.shuffled_reward.final_accuracy,
                "shuffled_reward_final_exact_string_rate": controls.shuffled_reward.final_exact_string_rate,
                "shuffled_reward_final_greedy_prefix_score": controls.shuffled_reward.final_greedy_prefix_score,
                "reset_weights_final_accuracy": controls.reset_weights_each_attempt.final_accuracy,
                "reset_weights_final_exact_string_rate": controls.reset_weights_each_attempt.final_exact_string_rate,
                "reset_weights_final_greedy_prefix_score": controls.reset_weights_each_attempt.final_greedy_prefix_score,
            })
        })
}

#[derive(Debug, Deserialize)]
struct GenomeArtifact {
    generations: Vec<GenomeGeneration>,
    final_population: Vec<GenomeMember>,
    final_winner_population_index: usize,
}

#[derive(Debug, Deserialize)]
struct GenomeGeneration {
    generation: u32,
    population_checkpoint: Vec<GenomeMember>,
}

#[derive(Debug, Clone, Deserialize)]
struct GenomeMember {
    population_index: usize,
    genome: OrganismGenome,
}

#[derive(Debug, Clone, Copy, Serialize)]
struct CandidateContract {
    targets: usize,
    rollouts: usize,
}

#[derive(Debug, Serialize)]
struct RankComparison {
    panel_seed: u64,
    generation: u32,
    spearman: f64,
    top_eight_overlap: usize,
    reference_winner_candidate_rank: usize,
    passes: bool,
}

#[derive(Debug, Serialize)]
struct CandidateCalibration {
    contract: CandidateContract,
    passes_all: bool,
    comparisons: Vec<RankComparison>,
}

fn calibrate(args: &[&str], out: &mut impl Write) -> Result<()> {
    let mut source = None;
    let mut generations = None;
    let mut workers = std::thread::available_parallelism()
        .map(|parallelism| parallelism.get())
        .unwrap_or(1);
    let mut output_path = None;
    let mut index = 0;
    while index < args.len() {
        match args[index] {
            "--generations" => {
                generations = Some(parse_u32_list(value(args, index, "--generations")?)?);
                index += 2;
            }
            "--workers" => {
                workers = value(args, index, "--workers")?.parse()?;
                index += 2;
            }
            "--out" => {
                output_path = Some(value(args, index, "--out")?.to_owned());
                index += 2;
            }
            value if !value.starts_with('-') && source.is_none() => {
                source = Some(value.to_owned());
                index += 1;
            }
            other => bail!("unknown hidden-string calibrate argument `{other}`"),
        }
    }
    let source = source.ok_or_else(|| anyhow!("calibrate needs a result artifact"))?;
    let generations = generations.ok_or_else(|| {
        anyhow!("calibrate needs explicit --generations for early/mid/late checkpoints")
    })?;
    let artifact: GenomeArtifact = serde_json::from_reader(result_reader(&source)?)
        .map_err(|error| anyhow!("cannot parse calibration artifact `{source}`: {error}"))?;
    let populations = generations
        .iter()
        .map(|generation| {
            artifact
                .generations
                .iter()
                .find(|entry| entry.generation == *generation)
                .filter(|entry| !entry.population_checkpoint.is_empty())
                .map(|entry| (entry.generation, entry.population_checkpoint.clone()))
                .ok_or_else(|| {
                    anyhow!("artifact has no population checkpoint at generation {generation}")
                })
        })
        .collect::<Result<Vec<_>>>()?;

    let candidates = [
        CandidateContract {
            targets: 128,
            rollouts: 1,
        },
        CandidateContract {
            targets: 256,
            rollouts: 1,
        },
        CandidateContract {
            targets: 256,
            rollouts: 2,
        },
        CandidateContract {
            targets: 512,
            rollouts: 1,
        },
        CandidateContract {
            targets: 512,
            rollouts: 2,
        },
        CandidateContract {
            targets: 768,
            rollouts: 2,
        },
        CandidateContract {
            targets: 1024,
            rollouts: 1,
        },
        CandidateContract {
            targets: 1024,
            rollouts: 2,
        },
    ];
    let panel_seeds = [
        DEFAULT_TARGET_PANEL_SEED,
        DEFAULT_TARGET_PANEL_SEED ^ 0x9e37_79b9_7f4a_7c15,
        DEFAULT_TARGET_PANEL_SEED ^ 0xd6e8_feb8_6659_fd93,
    ];
    let pool = EvaluationPool::new(workers)?;
    let mut comparisons = candidates
        .iter()
        .map(|_| Vec::new())
        .collect::<Vec<Vec<RankComparison>>>();

    for panel_seed in panel_seeds {
        let reference_task = evaluation_task(1024, 2, 32, panel_seed, false)?;
        let candidate_tasks = candidates
            .iter()
            .map(|candidate| {
                evaluation_task(candidate.targets, candidate.rollouts, 32, panel_seed, false)
            })
            .collect::<Result<Vec<_>>>()?;
        for (generation, population) in &populations {
            let genomes = population
                .iter()
                .map(|member| member.genome.clone())
                .collect::<Vec<_>>();
            let reference_scores = pool
                .evaluate(&reference_task, &genomes)?
                .into_iter()
                .map(|(fitness, _)| fitness)
                .collect::<Vec<_>>();
            for (candidate_index, task) in candidate_tasks.iter().enumerate() {
                let candidate_scores = pool
                    .evaluate(task, &genomes)?
                    .into_iter()
                    .map(|(fitness, _)| fitness)
                    .collect::<Vec<_>>();
                comparisons[candidate_index].push(compare_rankings(
                    panel_seed,
                    *generation,
                    population,
                    &reference_scores,
                    &candidate_scores,
                ));
            }
        }
    }

    let calibrated = candidates
        .into_iter()
        .zip(comparisons)
        .map(|(contract, comparisons)| CandidateCalibration {
            contract,
            passes_all: comparisons.iter().all(|comparison| comparison.passes),
            comparisons,
        })
        .collect::<Vec<_>>();
    let selected = calibrated
        .iter()
        .find(|candidate| candidate.passes_all)
        .map(|candidate| candidate.contract);
    let value = json!({
        "mode": "hidden_string_sampling_calibration",
        "source": source,
        "generations": generations,
        "workers": workers,
        "reference": { "targets": 1024, "rollouts": 2 },
        "acceptance": {
            "minimum_spearman": 0.95,
            "minimum_top_eight_overlap": 6,
            "maximum_reference_winner_rank": 3,
            "required_across_every_panel_and_generation": true,
        },
        "selected": selected,
        "candidates": calibrated,
    });
    persist_json(output_path.as_deref(), &value)?;
    writeln!(out, "{value}")?;
    Ok(())
}

fn horizon_sweep(args: &[&str], out: &mut impl Write) -> Result<()> {
    let mut source = None;
    let mut targets = 256_usize;
    let mut rollouts = 2_usize;
    let mut workers = std::thread::available_parallelism()
        .map(|parallelism| parallelism.get())
        .unwrap_or(1);
    let mut output_path = None;
    let mut index = 0;
    while index < args.len() {
        match args[index] {
            "--targets" => {
                targets = value(args, index, "--targets")?.parse()?;
                index += 2;
            }
            "--rollouts" => {
                rollouts = value(args, index, "--rollouts")?.parse()?;
                index += 2;
            }
            "--workers" => {
                workers = value(args, index, "--workers")?.parse()?;
                index += 2;
            }
            "--out" => {
                output_path = Some(value(args, index, "--out")?.to_owned());
                index += 2;
            }
            value if !value.starts_with('-') && source.is_none() => {
                source = Some(value.to_owned());
                index += 1;
            }
            other => bail!("unknown hidden-string horizon argument `{other}`"),
        }
    }
    let source = source.ok_or_else(|| anyhow!("horizon needs a result artifact"))?;
    let artifact: GenomeArtifact = serde_json::from_reader(result_reader(&source)?)
        .map_err(|error| anyhow!("cannot parse horizon artifact `{source}`: {error}"))?;
    let winner = artifact
        .final_population
        .get(artifact.final_winner_population_index)
        .ok_or_else(|| anyhow!("artifact has no final winner"))?;
    let pool = EvaluationPool::new(workers)?;
    let horizons = [8_u32, 16, 32, 64, 128];
    let mut measurements = Vec::with_capacity(horizons.len());
    for attempts in horizons {
        let task = evaluation_task(targets, rollouts, attempts, DEFAULT_TARGET_PANEL_SEED, true)?;
        let (_, evaluation) = pool
            .evaluate(&task, std::slice::from_ref(&winner.genome))?
            .into_iter()
            .next()
            .expect("one winner evaluation");
        measurements.push(json!({
            "attempts": attempts,
            "pre_learning_accuracy": evaluation.primary.pre_learning_accuracy,
            "final_accuracy": evaluation.primary.final_accuracy,
            "final_exact_string_rate": evaluation.primary.final_exact_string_rate,
            "final_greedy_prefix_score": evaluation.primary.final_greedy_prefix_score,
        }));
    }
    let reference_exact = measurements
        .last()
        .and_then(|measurement| measurement["final_exact_string_rate"].as_f64())
        .expect("128-attempt measurement");
    let reference_accuracy = measurements
        .last()
        .and_then(|measurement| measurement["final_accuracy"].as_f64())
        .expect("128-attempt measurement");
    let reference_prefix = measurements
        .last()
        .and_then(|measurement| measurement["final_greedy_prefix_score"].as_f64())
        .expect("128-attempt measurement");
    let select = |field: &str, reference: f64, fraction: f64| {
        measurements.iter().find_map(|measurement| {
            let value = measurement[field].as_f64()?;
            (value >= reference * fraction).then(|| measurement["attempts"].as_u64())
        })
    };
    let exact_monotonic = measurements.windows(2).all(|pair| {
        pair[0]["final_exact_string_rate"].as_f64() <= pair[1]["final_exact_string_rate"].as_f64()
    });
    let value = json!({
        "mode": "hidden_string_horizon_sweep",
        "source": source,
        "winner_population_index": winner.population_index,
        "targets": targets,
        "rollouts": rollouts,
        "workers": workers,
        "reference_attempts": 128,
        "reference_final_accuracy": reference_accuracy,
        "reference_final_exact_string_rate": reference_exact,
        "reference_final_greedy_prefix_score": reference_prefix,
        "exact_string_curve_monotonic": exact_monotonic,
        "smallest_accuracy_at_90_percent": select("final_accuracy", reference_accuracy, 0.90),
        "smallest_accuracy_at_95_percent": select("final_accuracy", reference_accuracy, 0.95),
        "smallest_exact_string_at_90_percent": select("final_exact_string_rate", reference_exact, 0.90),
        "smallest_exact_string_at_95_percent": select("final_exact_string_rate", reference_exact, 0.95),
        "smallest_greedy_prefix_at_90_percent": select("final_greedy_prefix_score", reference_prefix, 0.90),
        "smallest_greedy_prefix_at_95_percent": select("final_greedy_prefix_score", reference_prefix, 0.95),
        "measurements": measurements,
    });
    persist_json(output_path.as_deref(), &value)?;
    writeln!(out, "{value}")?;
    Ok(())
}

fn evaluation_task(
    targets: usize,
    rollouts: usize,
    attempts: u32,
    panel_seed: u64,
    include_initial_probe: bool,
) -> Result<HiddenStringTask> {
    let config = HiddenStringTaskConfig {
        attempts,
        training_probe_after_attempts: if include_initial_probe {
            vec![0, attempts]
        } else {
            vec![attempts]
        },
        report_probe_after_attempts: vec![0, attempts],
        target_panel_seed: panel_seed,
        training_target_count: targets,
        development_target_count: 8,
        sealed_target_count: 8,
        training_rollout_seeds: rollout_seeds(rollouts),
        ..HiddenStringTaskConfig::default()
    };
    HiddenStringTask::new(config)
}

fn rollout_seeds(count: usize) -> Vec<u64> {
    (0..count)
        .map(|index| 0x5452_4149_4e5f_3031_u64.wrapping_add(index as u64))
        .collect()
}

fn compare_rankings(
    panel_seed: u64,
    generation: u32,
    population: &[GenomeMember],
    reference: &[f64],
    candidate: &[f64],
) -> RankComparison {
    let spearman = spearman_correlation(reference, candidate);
    let reference_order = descending_order(population, reference);
    let candidate_order = descending_order(population, candidate);
    let top_eight_overlap = reference_order
        .iter()
        .take(8)
        .filter(|index| candidate_order.iter().take(8).any(|other| other == *index))
        .count();
    let reference_winner = reference_order[0];
    let reference_winner_candidate_rank = candidate_order
        .iter()
        .position(|index| *index == reference_winner)
        .expect("same population in both rankings")
        + 1;
    let passes = spearman >= 0.95 && top_eight_overlap >= 6 && reference_winner_candidate_rank <= 3;
    RankComparison {
        panel_seed,
        generation,
        spearman,
        top_eight_overlap,
        reference_winner_candidate_rank,
        passes,
    }
}

fn descending_order(population: &[GenomeMember], scores: &[f64]) -> Vec<usize> {
    let mut order = (0..scores.len()).collect::<Vec<_>>();
    order.sort_unstable_by(|left, right| {
        scores[*right].total_cmp(&scores[*left]).then_with(|| {
            population[*left]
                .population_index
                .cmp(&population[*right].population_index)
        })
    });
    order
}

fn spearman_correlation(left: &[f64], right: &[f64]) -> f64 {
    let left_ranks = average_ranks(left);
    let right_ranks = average_ranks(right);
    pearson_correlation(&left_ranks, &right_ranks)
}

fn average_ranks(values: &[f64]) -> Vec<f64> {
    let mut order = (0..values.len()).collect::<Vec<_>>();
    order.sort_unstable_by(|left, right| values[*left].total_cmp(&values[*right]));
    let mut ranks = vec![0.0; values.len()];
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && values[order[start]].total_cmp(&values[order[end]]).is_eq() {
            end += 1;
        }
        let rank = (start + 1 + end) as f64 / 2.0;
        for index in &order[start..end] {
            ranks[*index] = rank;
        }
        start = end;
    }
    ranks
}

fn pearson_correlation(left: &[f64], right: &[f64]) -> f64 {
    let count = left.len() as f64;
    let left_mean = left.iter().sum::<f64>() / count;
    let right_mean = right.iter().sum::<f64>() / count;
    let mut covariance = 0.0;
    let mut left_variance = 0.0;
    let mut right_variance = 0.0;
    for (left, right) in left.iter().zip(right) {
        let left_delta = left - left_mean;
        let right_delta = right - right_mean;
        covariance += left_delta * right_delta;
        left_variance += left_delta * left_delta;
        right_variance += right_delta * right_delta;
    }
    let denominator = (left_variance * right_variance).sqrt();
    if denominator == 0.0 {
        0.0
    } else {
        covariance / denominator
    }
}

fn parse_u32_list(raw: &str) -> Result<Vec<u32>> {
    raw.split(',')
        .map(|value| value.trim().parse().map_err(Into::into))
        .collect()
}

fn persist_json(path: Option<&str>, value: &Value) -> Result<()> {
    let Some(path) = path else {
        return Ok(());
    };
    atomic_write(path, |writer| {
        serde_json::to_writer_pretty(writer, value)?;
        Ok(())
    })
}

fn persist_compressed_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let path = path.to_string_lossy();
    atomic_write(path.as_ref(), |writer| {
        let mut encoder = zstd::stream::write::Encoder::new(writer, 3)?;
        serde_json::to_writer(&mut encoder, value)?;
        encoder.finish()?;
        Ok(())
    })
}

fn persist_plain_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let path = path.to_string_lossy();
    atomic_write(path.as_ref(), |writer| {
        serde_json::to_writer_pretty(writer, value)?;
        Ok(())
    })
}

fn load_seed_config(path: &str) -> Result<SeedGenomeConfig> {
    config::load_seed_genome_config_from_path(Path::new(path))
}

fn result_reader(path: &str) -> Result<Box<dyn Read>> {
    let file = File::open(path).map_err(|error| anyhow!("cannot open `{path}`: {error}"))?;
    if path.ends_with(".zst") {
        Ok(Box::new(zstd::stream::read::Decoder::new(file)?))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
}

fn value<'a>(args: &[&'a str], index: usize, flag: &str) -> Result<&'a str> {
    args.get(index + 1)
        .copied()
        .ok_or_else(|| anyhow!("{flag} needs a value"))
}
