use crate::run_output_path;
use anyhow::{anyhow, bail, Result};
use evolution::{
    run_resource_ecology, ActionSelection, AgentEvaluationConfig, AsexualSearchConfig,
    LearningNormalization, LearningRule, ResourceEcologyConfig, ResourceEcologyTask,
    SymbolicEcologyAudit, SymbolicEcologyMetrics, TaskEcology,
};
use serde_json::json;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::Path;
use std::time::Instant;
use task_library::{
    continual_learning::{ContinualLearningConfig, ContinualLearningTask},
    memory::{MemoryConfig, MemoryTask},
    next_token_prediction::{NextTokenPredictionConfig, NextTokenPredictionTask},
    reaction::{ReactionConfig, ReactionTask},
    renewable_resource::{RenewableResourceConfig, RenewableResourceTask},
    SymbolicTask,
};
use types::{OrganismGenome, SeedGenomeConfig};
use views::atomic_write;

const ALGORITHM: &str = "task_ecology_asexual_v1";
const DEFAULT_SEED_CONFIG: &str = config::CANONICAL_SEED_GENOME_CONFIG_PATH;

struct CommonRequest {
    seed_config_path: String,
    seed: u64,
    search: AsexualSearchConfig,
    ecology: ResourceEcologyConfig,
    agent: AgentEvaluationConfig,
    task_args: Vec<String>,
}

pub(crate) fn run_cli(args: &[&str], out_dir: &str, out: &mut impl Write) -> Result<()> {
    let Some((&task_name, tail)) = args.split_first() else {
        return write_help(out);
    };
    if matches!(task_name, "help" | "--help" | "-h") {
        return write_help(out);
    }
    if task_name == "analyze" {
        return analyze(tail, out);
    }
    let evaluate_source = if tail.first() == Some(&"evaluate") {
        Some(
            *tail
                .get(1)
                .ok_or_else(|| anyhow!("ecology {task_name} evaluate needs a frozen genome"))?,
        )
    } else {
        None
    };
    let tail = if evaluate_source.is_some() {
        &tail[2..]
    } else if tail.first() == Some(&"run") {
        &tail[1..]
    } else {
        tail
    };
    let plan = tail.first() == Some(&"plan");
    let tail = if plan { &tail[1..] } else { tail };
    let common = parse_common(task_name, tail)?;
    match task_name {
        "reaction" => {
            let task = ReactionTask {
                config: parse_reaction(&common.task_args)?,
            };
            dispatch(task, common, out_dir, plan, evaluate_source, out)
        }
        "memory" => {
            let task = MemoryTask {
                config: parse_memory(&common.task_args)?,
            };
            dispatch(task, common, out_dir, plan, evaluate_source, out)
        }
        "next-token" => {
            let task = NextTokenPredictionTask {
                config: parse_next_token(&common.task_args)?,
            };
            dispatch(task, common, out_dir, plan, evaluate_source, out)
        }
        "continual" => {
            let task = ContinualLearningTask {
                config: parse_continual(&common.task_args)?,
            };
            dispatch(task, common, out_dir, plan, evaluate_source, out)
        }
        "renewable" => {
            let task = RenewableResourceTask {
                config: parse_renewable(&common.task_args)?,
            };
            dispatch(task, common, out_dir, plan, evaluate_source, out)
        }
        other => bail!(
            "unknown ecology task `{other}`; expected reaction, memory, next-token, continual, or renewable"
        ),
    }
}

fn parse_common(task_name: &str, args: &[&str]) -> Result<CommonRequest> {
    let mut agent = AgentEvaluationConfig::default();
    if task_name == "memory" {
        agent.training_instances = 100;
        agent.development_instances = 100;
        agent.sealed_instances = 100;
        agent.training_rollouts = 2;
        agent.development_rollouts = 1;
        agent.sealed_rollouts = 2;
        agent.learning_rule = LearningRule::ImmediatePolicy;
    } else if task_name == "next-token" {
        agent.training_instances = 1;
        agent.development_instances = 1;
        agent.sealed_instances = 1;
        agent.training_rollouts = 1;
        agent.development_rollouts = 1;
        agent.sealed_rollouts = 1;
        agent.learning_rule = LearningRule::TargetPredictionError;
        agent.action_selection = ActionSelection::Greedy;
    }
    let mut request = CommonRequest {
        seed_config_path: DEFAULT_SEED_CONFIG.to_owned(),
        seed: 0,
        search: AsexualSearchConfig::default(),
        ecology: ResourceEcologyConfig::default(),
        agent,
        task_args: Vec::new(),
    };
    let mut index = 0;
    while index < args.len() {
        let flag = args[index];
        let raw = args
            .get(index + 1)
            .copied()
            .ok_or_else(|| anyhow!("{flag} needs a value"))?;
        match flag {
            "--seed" => request.seed = raw.parse()?,
            "--population" => request.search.population_size = raw.parse()?,
            "--generations" => request.search.generations = raw.parse()?,
            "--workers" => request.search.evaluation_workers = raw.parse()?,
            "--training-instances" => request.agent.training_instances = raw.parse()?,
            "--development-instances" => request.agent.development_instances = raw.parse()?,
            "--sealed-instances" => request.agent.sealed_instances = raw.parse()?,
            "--training-rollouts" => request.agent.training_rollouts = raw.parse()?,
            "--development-rollouts" => request.agent.development_rollouts = raw.parse()?,
            "--sealed-rollouts" => request.agent.sealed_rollouts = raw.parse()?,
            "--seed-config" => request.seed_config_path = raw.to_owned(),
            "--exact-elites" => request.ecology.exact_elite_copies = raw.parse()?,
            "--tournament-size" => request.ecology.tournament_size = raw.parse()?,
            "--exploration-temperature" => request.agent.exploration_temperature = raw.parse()?,
            "--audit-interval" => request.agent.audit_interval = raw.parse()?,
            "--reset-dynamics-at-trial-boundary" => {
                request.agent.reset_dynamics_at_trial_boundary = raw.parse()?
            }
            "--learning-normalization" => {
                request.agent.learning_normalization = match raw {
                    "none" => LearningNormalization::None,
                    "nlms" => LearningNormalization::Nlms,
                    _ => bail!("learning normalization must be none or nlms"),
                }
            }
            "--learning-rule" => {
                request.agent.learning_rule = match raw {
                    "disabled" => LearningRule::Disabled,
                    "immediate_policy" => LearningRule::ImmediatePolicy,
                    "target_prediction_error" => LearningRule::TargetPredictionError,
                    "temporal_prediction_error" => LearningRule::TemporalPredictionError,
                    _ => {
                        bail!("learning rule must be disabled, immediate_policy, target_prediction_error, or temporal_prediction_error")
                    }
                }
            }
            "--action-selection" => {
                request.agent.action_selection = match raw {
                    "greedy" => ActionSelection::Greedy,
                    "sampled" => ActionSelection::Sampled,
                    _ => bail!("action selection must be greedy or sampled"),
                }
            }
            "--param" => {
                let (key, value) = raw
                    .split_once('=')
                    .ok_or_else(|| anyhow!("expected key=value"))?;
                apply_search_param(&mut request.search, key, value)?;
            }
            _ => {
                request.task_args.push(flag.to_owned());
                request.task_args.push(raw.to_owned());
            }
        }
        index += 2;
    }
    request.search.validate()?;
    request.ecology.validate(request.search.population_size)?;
    Ok(request)
}

fn parse_reaction(args: &[String]) -> Result<ReactionConfig> {
    let mut config = ReactionConfig::default();
    parse_pairs(args, |flag, raw| {
        match flag {
            "--symbols" => config.symbols_per_instance = raw.parse()?,
            other => bail!("unknown reaction option `{other}`"),
        }
        Ok(())
    })?;
    Ok(config)
}

fn parse_memory(args: &[String]) -> Result<MemoryConfig> {
    let mut config = MemoryConfig::default();
    parse_pairs(args, |flag, raw| {
        match flag {
            "--length" => config.sequence_length = raw.parse()?,
            "--attempts" => config.attempts = raw.parse()?,
            other => bail!("unknown memory option `{other}`"),
        }
        Ok(())
    })?;
    Ok(config)
}

fn parse_next_token(args: &[String]) -> Result<NextTokenPredictionConfig> {
    let mut config = NextTokenPredictionConfig::default();
    parse_pairs(args, |flag, raw| {
        match flag {
            "--snippet" => config.snippet = raw.to_owned(),
            "--learning-passes" => config.learning_passes = raw.parse()?,
            other => bail!("unknown next-token option `{other}`"),
        }
        Ok(())
    })?;
    Ok(config)
}

fn parse_continual(args: &[String]) -> Result<ContinualLearningConfig> {
    let mut config = ContinualLearningConfig::default();
    parse_pairs(args, |flag, raw| {
        match flag {
            "--lifetime-ticks" => config.lifetime_ticks = raw.parse()?,
            "--minimum-regime-ticks" => config.minimum_regime_ticks = raw.parse()?,
            "--maximum-regime-ticks" => config.maximum_regime_ticks = raw.parse()?,
            other => bail!("unknown continual-learning option `{other}`"),
        }
        Ok(())
    })?;
    Ok(config)
}

fn parse_renewable(args: &[String]) -> Result<RenewableResourceConfig> {
    let mut config = RenewableResourceConfig::default();
    parse_pairs(args, |flag, raw| {
        match flag {
            "--lifetime-ticks" => config.ticks_per_instance = raw.parse()?,
            "--resource-stock" => config.stock = raw.parse()?,
            other => bail!("unknown renewable option `{other}`"),
        }
        Ok(())
    })?;
    Ok(config)
}

fn parse_pairs(args: &[String], mut parse: impl FnMut(&str, &str) -> Result<()>) -> Result<()> {
    if !args.len().is_multiple_of(2) {
        bail!("task options require flag/value pairs");
    }
    for pair in args.chunks_exact(2) {
        parse(&pair[0], &pair[1])?;
    }
    Ok(())
}

fn dispatch<T: SymbolicTask + Clone>(
    task: T,
    request: CommonRequest,
    out_dir: &str,
    plan: bool,
    evaluate_source: Option<&str>,
    out: &mut impl Write,
) -> Result<()> {
    task.validate()?;
    if let Some(source) = evaluate_source {
        return evaluate_frozen(
            TaskEcology::new(task, request.agent),
            source,
            request.seed,
            out,
        );
    }
    if plan {
        return writeln!(out, "{}", json!({
            "mode":"task_ecology_plan", "algorithm":ALGORITHM, "task":task.name(),
            "search":request.search, "ecology":request.ecology, "agent":request.agent,
            "task_config":task.config(),
            "maximum_task_steps": request.search.population_size as u128 * request.search.generations as u128 * request.agent.training_instances as u128 * request.agent.training_rollouts as u128 * (task.max_steps_per_instance() + task.probe_steps_per_instance()) as u128,
        })).map_err(Into::into);
    }
    execute(
        TaskEcology::new(task, request.agent.clone()),
        request,
        out_dir,
        out,
    )
}

fn execute<T: SymbolicTask + Clone>(
    task: TaskEcology<T>,
    request: CommonRequest,
    out_dir: &str,
    out: &mut impl Write,
) -> Result<()> {
    let seed_genome = load_seed_config(&request.seed_config_path)?;
    let total_generations = request.search.generations;
    let population = request.search.population_size;
    let started = Instant::now();
    eprintln!(
        "{}",
        json!({"event":"task_ecology_started","task":task.task.name(),"population":population,"generations":total_generations,"workers":request.search.evaluation_workers})
    );
    let result = run_resource_ecology(
        &task,
        request.search,
        request.ecology,
        seed_genome,
        request.seed,
        |generation| {
            let completed = generation.generation + 1;
            let elapsed = started.elapsed().as_secs_f64();
            eprintln!(
                "{}",
                json!({
                    "event":"task_ecology_generation","task":task.task.name(),"generation":generation.generation,
                    "completed_generations":completed,"total_generations":total_generations,
                    "progress_percent":100.0*f64::from(completed)/f64::from(total_generations),
                    "leading_accuracy":generation.leading_evaluation.accuracy,
                    "leading_trial_success_rate":generation.leading_evaluation.trial_success_rate,
                    "leading_resource_units":generation.leading_evaluation.resource_units,
                    "development_accuracy":generation.leading_audit.as_ref().map(|audit| audit.primary.accuracy),
                    "hidden_nodes":generation.leading_hidden_nodes,"enabled_connections":generation.leading_enabled_connections,
                    "elapsed_seconds":elapsed,"eta_seconds":elapsed/f64::from(completed)*f64::from(total_generations-completed),
                })
            );
        },
    )?;
    let mut path = run_output_path(out_dir, &format!("task-ecology-{}", task.task.name()))?;
    path.set_extension("json.zst");
    let path_string = path.to_string_lossy().into_owned();
    atomic_write(&path_string, |writer| {
        let mut encoder = zstd::stream::write::Encoder::new(writer, 3)?;
        serde_json::to_writer(&mut encoder, &result)?;
        encoder.finish()?;
        Ok(())
    })?;
    writeln!(
        out,
        "{}",
        json!({
            "wrote":path_string,"algorithm":result.algorithm,"task":result.task,"termination":result.termination,
            "selected_generation":result.selected_generation,"development":audit_summary(&result.selected_development_audit),
            "sealed":audit_summary(&result.sealed_audit),"work":result.work,"wall_time_seconds":result.total_wall_time_seconds,
        })
    )?;
    Ok(())
}

fn evaluate_frozen<T: SymbolicTask + Clone>(
    task: TaskEcology<T>,
    source: &str,
    seed: u64,
    out: &mut impl Write,
) -> Result<()> {
    task.validate()?;
    let value: serde_json::Value = serde_json::from_reader(result_reader(source)?)?;
    let genome_value = value.get("genome").unwrap_or(&value).clone();
    let genome: OrganismGenome = serde_json::from_value(genome_value)
        .map_err(|error| anyhow!("cannot decode genome from `{source}`: {error}"))?;
    let audit = task.audit(&genome, "sealed", seed)?;
    writeln!(
        out,
        "{}",
        json!({
            "mode":"task_ecology_frozen_evaluation",
            "source":source,
            "task":task.task.name(),
            "task_config":task.task.config(),
            "agent":task.agent,
            "hidden_nodes":genome.hidden_node_count(),
            "enabled_connections":genome.enabled_connection_count(),
            "sealed":audit_summary(&audit),
        })
    )?;
    Ok(())
}

fn audit_summary(audit: &SymbolicEcologyAudit) -> serde_json::Value {
    json!({"cohort":audit.cohort,"primary":metrics(&audit.primary),"efference_copy_off":metrics(&audit.efference_copy_off),"prediction_error_feedback_off":metrics(&audit.prediction_error_feedback_off)})
}
fn metrics(value: &SymbolicEcologyMetrics) -> serde_json::Value {
    json!({"accuracy":value.accuracy,"learning_accuracy":value.learning_accuracy,"probe_accuracy":value.probe_accuracy,"exact_string_rate":value.trial_success_rate,"successful_trials":value.successful_trials,"completed_trials":value.completed_trials,"mean_probe_target_probability":value.mean_probe_target_probability,"mean_probe_sequence_probability":value.mean_probe_sequence_probability,"resource_units":value.resource_units,"resource_throughput_per_1000_ticks":value.resource_throughput_per_1000_ticks,"mean_reward":value.mean_reward})
}

fn analyze(args: &[&str], out: &mut impl Write) -> Result<()> {
    if args.is_empty() {
        bail!("ecology analyze needs at least one result");
    }
    let values = args.iter().map(|path| -> Result<_> { let value: serde_json::Value = serde_json::from_reader(result_reader(path)?)?; Ok(json!({"path":path,"task":value["task"],"algorithm":value["algorithm"],"termination":value["termination"],"selected_generation":value["selected_generation"],"sealed_audit":value["sealed_audit"],"work":value["work"]})) }).collect::<Result<Vec<_>>>()?;
    writeln!(
        out,
        "{}",
        if values.len() == 1 {
            values.into_iter().next().unwrap()
        } else {
            json!({"runs":values})
        }
    )?;
    Ok(())
}

fn load_seed_config(path: &str) -> Result<SeedGenomeConfig> {
    config::load_seed_genome_config_from_path(Path::new(path))
}
fn result_reader(path: &str) -> Result<Box<dyn Read>> {
    let file = File::open(path)?;
    if path.ends_with(".zst") {
        Ok(Box::new(zstd::stream::read::Decoder::new(file)?))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
}

fn apply_search_param(config: &mut AsexualSearchConfig, key: &str, value: &str) -> Result<()> {
    match key {
        "mutate_weight_probability" => config.mutate_weight_probability = value.parse()?,
        "replace_weight_probability" => config.replace_weight_probability = value.parse()?,
        "weight_perturb_stddev" => config.weight_perturb_stddev = value.parse()?,
        "mutate_bias_probability" => config.mutate_bias_probability = value.parse()?,
        "bias_perturb_stddev" => config.bias_perturb_stddev = value.parse()?,
        "mutate_time_constant_probability" => {
            config.mutate_time_constant_probability = value.parse()?
        }
        "time_constant_perturb_stddev" => config.time_constant_perturb_stddev = value.parse()?,
        "mutate_learning_rate_probability" => {
            config.mutate_learning_rate_probability = value.parse()?
        }
        "learning_rate_perturb_stddev" => config.learning_rate_perturb_stddev = value.parse()?,
        "mutate_plasticity_coefficient_probability" => {
            config.mutate_plasticity_coefficient_probability = value.parse()?
        }
        "plasticity_coefficient_perturb_stddev" => {
            config.plasticity_coefficient_perturb_stddev = value.parse()?
        }
        "add_connection_probability" => config.add_connection_probability = value.parse()?,
        "delete_connection_probability" => config.delete_connection_probability = value.parse()?,
        "add_node_probability" => config.add_node_probability = value.parse()?,
        "delete_node_probability" => config.delete_node_probability = value.parse()?,
        "mutate_only_active_interface" => config.mutate_only_active_interface = value.parse()?,
        "recurrent_node_self_connection" => {
            config.recurrent_node_self_connection = value.parse()?
        }
        other => bail!("unknown search parameter `{other}`"),
    }
    Ok(())
}

fn write_help(out: &mut impl Write) -> Result<()> {
    writeln!(out, "Task ecology:\n  cli ecology reaction [run|plan] [OPTIONS]\n  cli ecology memory [run|plan] [OPTIONS]\n  cli ecology memory evaluate FROZEN [OPTIONS]\n  cli ecology next-token [run|plan] [OPTIONS]\n  cli ecology continual [run|plan] [OPTIONS]\n  cli ecology renewable [run|plan] [OPTIONS]\n  cli ecology analyze RESULT...\n\nShared: --seed N --population N --generations N --workers N --training-instances N --development-instances N --sealed-instances N --training-rollouts N --development-rollouts N --sealed-rollouts N --exact-elites N --tournament-size N --exploration-temperature F --action-selection greedy|sampled --learning-rule disabled|immediate_policy|target_prediction_error|temporal_prediction_error --learning-normalization none|nlms --reset-dynamics-at-trial-boundary true|false --audit-interval N --param key=value\nReaction: --symbols N\nMemory: --length N --attempts N\nNext token: --snippet TEXT --learning-passes N\nContinual: --lifetime-ticks N --minimum-regime-ticks N --maximum-regime-ticks N\nRenewable: --lifetime-ticks N --resource-stock N").map_err(Into::into)
}
