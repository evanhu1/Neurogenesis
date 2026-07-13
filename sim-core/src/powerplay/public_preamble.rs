//! Executable zero-shot compatibility probe for a public preamble.
//!
//! PowerPlay solvers are selected on ordinary task episodes. This evaluator-
//! owned probe asks a narrower question before that premise is imported into a
//! future algorithm: does an accepted solver already make task-specific use of
//! a public, physical FoodRay preamble that it never saw during selection? It
//! compares a meaningful program encoding against all-blank and left/right-
//! permuted controls on disjoint contexts. Failure rejects only zero-shot import
//! of these checkpoints. The probe is diagnostic and is never evidence about a
//! trainable public decoder, branch transfer, or open-endedness.

use super::{
    add_zero_residual_module, configure_task_world, evaluate_program, execute_program_on_sim,
    grammar, knockout_module, materialize_genome, mix64, mutate_module, prepare_episode_state,
    randomize_module, resolve_stage_context, run_powerplay, EcologyProgram, EpisodeEvidence,
    ModuleSpec, PowerPlayConfig, PowerPlayResult, ProgramEvaluation, ResourceMotion,
};
use crate::{
    grid::{hex_neighbor, rotate_by_steps},
    progressive::{exact_fingerprint, ProtectedResidual},
    Simulation,
};
use anyhow::{anyhow, bail, Result};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use sim_config::WorldConfig;
use sim_types::{
    food_visual, ActionType, EnergyLedgerRow, FacingDirection, FoodId, FoodKind, FoodState,
    Occupant, OrganismGenome, OrganismState,
};
use std::cmp::Ordering;
use std::collections::BTreeSet;

const PUBLIC_PREAMBLE_RESULT_SCHEMA_VERSION: u32 = 2;
const PROBE_CONTEXT_COUNT: usize = 16;
const MAX_ENCODED_STAGES: usize = 2;
const PREAMBLE_CONTEXT_DOMAIN: u64 = 0x5052_4541_4d42_4c45;
const MAGIC_BITS: [u8; 8] = [1, 0, 1, 1, 0, 1, 0, 0];
const TERMINATOR_BITS: [u8; 8] = [0, 0, 1, 0, 1, 1, 0, 1];
const PROTOCOL_VERSION: u8 = 1;
const FIXED_PREAMBLE_TICKS: usize = 36;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicPreambleProbeConfig {
    /// Independent PowerPlay construction seeds. Each run constructs its own
    /// program and accepted checkpoints through the production outer loop.
    pub source_run_seeds: Vec<u64>,
    /// Template for the actual source run. `run_seed` is replaced by each
    /// entry above and `max_depth` is fixed at two for this premise probe.
    pub source_powerplay: PowerPlayConfig,
}

impl Default for PublicPreambleProbeConfig {
    fn default() -> Self {
        Self {
            source_run_seeds: vec![7, 42, 123],
            source_powerplay: PowerPlayConfig {
                max_depth: 2,
                ..PowerPlayConfig::default()
            },
        }
    }
}

type RenderedCue = (Option<i8>, Option<(i32, i32)>);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicPreambleProtocol {
    pub name: String,
    pub version: u8,
    pub bit_order: String,
    pub fixed_preamble_ticks: usize,
    pub maximum_encoded_stages: usize,
    pub magic_bits: Vec<u8>,
    pub stage_count_width_bits: usize,
    pub stage_slot_schema: Vec<String>,
    pub padding_rule: String,
    pub terminator_bits: Vec<u8>,
    pub meaningful_rendering: String,
    pub blank_rendering: String,
    pub permuted_rendering: String,
    pub reset_contract: String,
    pub task_contract: String,
}

impl PublicPreambleProtocol {
    fn canonical() -> Self {
        Self {
            name: "fixed_width_resolved_powerplay_program_v1".to_string(),
            version: PROTOCOL_VERSION,
            bit_order: "most-significant bit first within every integer field".to_string(),
            fixed_preamble_ticks: FIXED_PREAMBLE_TICKS,
            maximum_encoded_stages: MAX_ENCODED_STAGES,
            magic_bits: MAGIC_BITS.to_vec(),
            stage_count_width_bits: 2,
            stage_slot_schema: vec![
                "present:1".to_string(),
                "resolved_relative_turns_plus_3:3".to_string(),
                "resolved_distance_minus_1:2".to_string(),
                "motion(static=0,left_drift=1,right_drift=2):2".to_string(),
            ],
            padding_rule: "unused stage slots are eight zero bits".to_string(),
            terminator_bits: TERMINATOR_BITS.to_vec(),
            meaningful_rendering: "semantic 0 = one zero-energy plant at FoodRay -1; semantic 1 = one zero-energy plant at FoodRay +1".to_string(),
            blank_rendering: "no physical cue for any semantic bit; one FoodId is still reserved per tick so task resource identities remain matched".to_string(),
            permuted_rendering: "semantic 0 = FoodRay +1 and semantic 1 = FoodRay -1".to_string(),
            reset_contract: "before every preamble tick and once before the task, restore the initial body and East-facing center pose while preserving the complete live BrainState; clear all food and regrowth state".to_string(),
            task_contract: "run the actual PowerPlay program with its fixed energy escrow for the full ordinary horizon; completion deadlines are relative to the first post-preamble task tick".to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PreambleArmKind {
    Meaningful,
    Blank,
    Permuted,
}

impl PreambleArmKind {
    const ALL: [Self; 3] = [Self::Meaningful, Self::Blank, Self::Permuted];
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PreambleTickTrace {
    pub bit_index: usize,
    pub absolute_turn: u64,
    pub semantic_bit: u8,
    pub rendered_ray_offset: Option<i8>,
    pub cue_q: Option<i32>,
    pub cue_r: Option<i32>,
    pub selected_action: ActionType,
    pub q_after_tick: i32,
    pub r_after_tick: i32,
    pub facing_after_tick: FacingDirection,
    pub organism_energy_before_bits: u32,
    pub organism_energy_after_bits: u32,
    pub food_ray_activation_bits: Vec<u32>,
    pub core_energy_ledger: EnergyLedgerRow,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PublicPreambleArmEvidence {
    pub arm: PreambleArmKind,
    pub preamble_ticks: usize,
    pub task_ticks: u64,
    pub total_ticks: u64,
    pub prefix_plant_consumptions: u64,
    pub prefix_initial_organism_energy_bits: u32,
    pub prefix_final_organism_energy_bits: u32,
    pub maximum_prefix_energy_ledger_residual: f64,
    pub maximum_prefix_energy_ledger_tolerance: f64,
    pub prefix_energy_closed: bool,
    pub prefix_trace: Vec<PreambleTickTrace>,
    pub task_prefix_successes: Vec<bool>,
    pub full_task_success: bool,
    pub task_episode: EpisodeEvidence,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicPreambleContextEvidence {
    pub context_seed: u64,
    pub resolved_program: EcologyProgram,
    pub semantic_bits: Vec<u8>,
    pub arms: Vec<PublicPreambleArmEvidence>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicPreambleArmAggregate {
    pub arm: PreambleArmKind,
    pub prefix_success_counts: Vec<usize>,
    pub full_task_success_count: usize,
    pub all_energy_closed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicPreambleStrictGate {
    pub meaningful_required_minimum: usize,
    pub control_allowed_maximum: usize,
    pub meaningful_success_count: usize,
    pub blank_success_count: usize,
    pub permuted_success_count: usize,
    pub identical_preamble_ticks: bool,
    pub identical_task_ticks: bool,
    pub all_energy_closed: bool,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicPreamblePairEvidence {
    pub solver_depth: u32,
    pub task_depth: u32,
    pub solver_genome: OrganismGenome,
    pub solver_genome_fingerprint: String,
    pub task_program: EcologyProgram,
    pub task_program_fingerprint: String,
    pub context_seeds: Vec<u64>,
    pub contexts_disjoint_from_source_search_and_admission: bool,
    pub contexts: Vec<PublicPreambleContextEvidence>,
    pub aggregates: Vec<PublicPreambleArmAggregate>,
    pub strict_gate: PublicPreambleStrictGate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicPreambleSourceRunEvidence {
    pub source_run_seed: u64,
    pub source_config: PowerPlayConfig,
    pub effective_task_world: WorldConfig,
    pub effective_task_world_fingerprint: String,
    pub source_result_fingerprint: String,
    pub accepted_depth: u32,
    pub accepted_program: EcologyProgram,
    pub accepted_program_fingerprint: String,
    pub champion_genome_fingerprint: String,
    pub source_stopped_reason: Option<String>,
    pub blockers: Vec<String>,
    pub pairs: Vec<PublicPreamblePairEvidence>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ImportedPreamblePremiseVerdict {
    SurvivedStrictGate,
    Falsified,
    NotEvaluable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicPreambleProbeResult {
    pub result_schema_version: u32,
    pub algorithm: String,
    pub claim_scope: String,
    pub evaluator_owned: bool,
    pub evidentiary: bool,
    pub open_endedness_demonstrated: bool,
    pub config: PublicPreambleProbeConfig,
    pub base_world_fingerprint: String,
    pub protocol: PublicPreambleProtocol,
    pub source_runs: Vec<PublicPreambleSourceRunEvidence>,
    pub requested_pair_count: usize,
    pub evaluated_pair_count: usize,
    pub passed_pair_count: usize,
    pub verdict: ImportedPreamblePremiseVerdict,
    pub verdict_reason: String,
    pub branch_transfer_status: String,
    pub limitations: Vec<String>,
}

/// Construct actual PowerPlay checkpoints, then run the evaluator-owned public
/// preamble zero-shot compatibility check on independent contexts.
pub fn run_public_preamble_probe(
    base_world: WorldConfig,
    config: PublicPreambleProbeConfig,
) -> Result<PublicPreambleProbeResult> {
    validate_probe_config(&config)?;
    let base_world_fingerprint = fingerprint(&base_world)?;
    let protocol = PublicPreambleProtocol::canonical();
    let mut globally_reserved_contexts = config
        .source_powerplay
        .search_seeds
        .iter()
        .chain(&config.source_powerplay.episode_seeds)
        .copied()
        .collect::<BTreeSet<_>>();
    let mut source_runs = Vec::with_capacity(config.source_run_seeds.len());

    for &source_run_seed in &config.source_run_seeds {
        let mut source_config = config.source_powerplay.clone();
        source_config.run_seed = source_run_seed;
        let mut effective_task_world = base_world.clone();
        configure_task_world(&mut effective_task_world, &source_config);
        let effective_task_world_fingerprint = fingerprint(&effective_task_world)?;
        let source_result = run_powerplay(base_world.clone(), source_config.clone())?;
        let source_result_fingerprint = fingerprint(&source_result)?;
        let accepted_program_fingerprint = fingerprint(&source_result.program)?;
        let champion_genome_fingerprint = fingerprint(&source_result.champion_genome)?;
        let context_seeds = derive_context_seeds(
            source_run_seed,
            &source_config,
            &mut globally_reserved_contexts,
        );
        let mut blockers = Vec::new();
        let mut pairs = Vec::new();

        for depth in 1..=MAX_ENCODED_STAGES as u32 {
            if source_result.accepted_depth < depth {
                blockers.push(format!(
                    "source PowerPlay run accepted depth {} but the depth-{depth} solver/task pair is required",
                    source_result.accepted_depth
                ));
                continue;
            }
            let accepted_depth = source_result
                .depths
                .iter()
                .find(|entry| entry.depth == depth && entry.accepted)
                .ok_or_else(|| {
                    anyhow!(
                        "source run reports accepted depth {depth} without accepted depth evidence"
                    )
                })?;
            let genome = accepted_depth.candidate_genome.as_ref().ok_or_else(|| {
                anyhow!("accepted source depth {depth} has no exact candidate genome")
            })?;
            let task = EcologyProgram {
                stages: source_result.program.stages[..depth as usize].to_vec(),
            };
            pairs.push(evaluate_pair(
                &effective_task_world,
                &source_config,
                depth,
                genome,
                &task,
                &context_seeds,
            )?);
        }

        source_runs.push(PublicPreambleSourceRunEvidence {
            source_run_seed,
            source_config,
            effective_task_world,
            effective_task_world_fingerprint,
            source_result_fingerprint,
            accepted_depth: source_result.accepted_depth,
            accepted_program: source_result.program,
            accepted_program_fingerprint,
            champion_genome_fingerprint,
            source_stopped_reason: source_result.stopped_reason,
            blockers,
            pairs,
        });
    }

    let requested_pair_count = config.source_run_seeds.len() * MAX_ENCODED_STAGES;
    let evaluated_pair_count = source_runs.iter().map(|run| run.pairs.len()).sum::<usize>();
    let passed_pair_count = source_runs
        .iter()
        .flat_map(|run| &run.pairs)
        .filter(|pair| pair.strict_gate.passed)
        .count();
    let any_failed = source_runs
        .iter()
        .flat_map(|run| &run.pairs)
        .any(|pair| !pair.strict_gate.passed);
    let (verdict, verdict_reason) = if any_failed {
        (
            ImportedPreamblePremiseVerdict::Falsified,
            "at least one exact accepted legacy solver/task pair failed the zero-shot >=14/16 meaningful and <=2/16 blank/permuted matched-arm gate".to_string(),
        )
    } else if evaluated_pair_count == requested_pair_count
        && passed_pair_count == requested_pair_count
    {
        (
            ImportedPreamblePremiseVerdict::SurvivedStrictGate,
            "every requested exact accepted solver/task pair passed the strict matched-arm gate"
                .to_string(),
        )
    } else {
        (
            ImportedPreamblePremiseVerdict::NotEvaluable,
            "no evaluated pair failed, but at least one requested source run did not construct both accepted solver/task depths".to_string(),
        )
    };

    let branch_transfer_status =
        "not_implemented_by_this_zero_shot_compatibility_probe".to_string();

    Ok(PublicPreambleProbeResult {
        result_schema_version: PUBLIC_PREAMBLE_RESULT_SCHEMA_VERSION,
        algorithm: "powerplay_public_preamble_zero_shot_compatibility_v2".to_string(),
        claim_scope: "evaluator-owned zero-shot compatibility of exact legacy checkpoints with one unfamiliar public preamble; not evidence about trainable decoder capacity, branch transfer, selection, novelty, or open-endedness".to_string(),
        evaluator_owned: true,
        evidentiary: false,
        open_endedness_demonstrated: false,
        config,
        base_world_fingerprint,
        protocol,
        source_runs,
        requested_pair_count,
        evaluated_pair_count,
        passed_pair_count,
        verdict,
        verdict_reason,
        branch_transfer_status,
        limitations: vec![
            "the accepted PowerPlay controllers were not selected or trained on this public preamble".to_string(),
            "a failure rejects importing this controller interface; it does not prove that no evolvable public interface can work".to_string(),
            "the task grammar and inspected depths remain bounded".to_string(),
            "the probe deliberately restores body and pose between cue ticks, so only controller state can carry the prefix".to_string(),
        ],
    })
}

fn validate_probe_config(config: &PublicPreambleProbeConfig) -> Result<()> {
    if config.source_run_seeds.is_empty() {
        bail!("public-preamble probe requires at least one source run seed");
    }
    if config
        .source_run_seeds
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .len()
        != config.source_run_seeds.len()
    {
        bail!("public-preamble source run seeds must be unique");
    }
    if config.source_powerplay.max_depth != MAX_ENCODED_STAGES as u32 {
        bail!("public-preamble source PowerPlay depth must be exactly two");
    }
    config.source_powerplay.validate()
}

fn derive_context_seeds(
    source_run_seed: u64,
    source_config: &PowerPlayConfig,
    globally_reserved: &mut BTreeSet<u64>,
) -> Vec<u64> {
    globally_reserved.extend(source_config.search_seeds.iter().copied());
    globally_reserved.extend(source_config.episode_seeds.iter().copied());
    let mut contexts = Vec::with_capacity(PROBE_CONTEXT_COUNT);
    let mut nonce = 0_u64;
    while contexts.len() < PROBE_CONTEXT_COUNT {
        let candidate = mix64(
            source_run_seed ^ PREAMBLE_CONTEXT_DOMAIN ^ nonce.wrapping_mul(0x9e37_79b9_7f4a_7c15),
        );
        nonce = nonce.wrapping_add(1);
        if globally_reserved.insert(candidate) {
            contexts.push(candidate);
        }
    }
    contexts
}

fn evaluate_pair(
    world: &WorldConfig,
    config: &PowerPlayConfig,
    depth: u32,
    genome: &OrganismGenome,
    task: &EcologyProgram,
    context_seeds: &[u64],
) -> Result<PublicPreamblePairEvidence> {
    let solver_genome_fingerprint = fingerprint(genome)?;
    let task_program_fingerprint = fingerprint(task)?;
    let source_contexts = config
        .search_seeds
        .iter()
        .chain(&config.episode_seeds)
        .copied()
        .collect::<BTreeSet<_>>();
    let contexts_disjoint_from_source_search_and_admission = context_seeds
        .iter()
        .all(|seed| !source_contexts.contains(seed));
    if !contexts_disjoint_from_source_search_and_admission {
        bail!("public-preamble context overlaps source search or admission contexts");
    }

    let mut contexts = Vec::with_capacity(context_seeds.len());
    for &context_seed in context_seeds {
        let (resolved_program, semantic_bits) = encode_resolved_program(task, context_seed)?;
        let mut arms = Vec::with_capacity(PreambleArmKind::ALL.len());
        for arm in PreambleArmKind::ALL {
            arms.push(run_arm(
                world,
                config,
                genome,
                task,
                context_seed,
                &semantic_bits,
                arm,
            )?);
        }
        contexts.push(PublicPreambleContextEvidence {
            context_seed,
            resolved_program,
            semantic_bits,
            arms,
        });
    }

    let aggregates = PreambleArmKind::ALL
        .iter()
        .copied()
        .map(|arm| aggregate_arm(&contexts, arm, task.stages.len()))
        .collect::<Result<Vec<_>>>()?;
    let meaningful = aggregate_for(&aggregates, PreambleArmKind::Meaningful)?;
    let blank = aggregate_for(&aggregates, PreambleArmKind::Blank)?;
    let permuted = aggregate_for(&aggregates, PreambleArmKind::Permuted)?;
    let identical_preamble_ticks = contexts.iter().all(|context| {
        context
            .arms
            .iter()
            .all(|arm| arm.preamble_ticks == FIXED_PREAMBLE_TICKS)
    });
    let expected_task_ticks = config
        .ticks_per_stage
        .saturating_mul(task.stages.len() as u64);
    let identical_task_ticks = contexts.iter().all(|context| {
        context
            .arms
            .iter()
            .all(|arm| arm.task_ticks == expected_task_ticks)
    });
    let all_energy_closed = aggregates
        .iter()
        .all(|aggregate| aggregate.all_energy_closed);
    let meaningful_success_count = meaningful.full_task_success_count;
    let blank_success_count = blank.full_task_success_count;
    let permuted_success_count = permuted.full_task_success_count;
    let passed = meaningful_success_count >= 14
        && blank_success_count <= 2
        && permuted_success_count <= 2
        && identical_preamble_ticks
        && identical_task_ticks
        && all_energy_closed;

    Ok(PublicPreamblePairEvidence {
        solver_depth: depth,
        task_depth: depth,
        solver_genome: genome.clone(),
        solver_genome_fingerprint,
        task_program: task.clone(),
        task_program_fingerprint,
        context_seeds: context_seeds.to_vec(),
        contexts_disjoint_from_source_search_and_admission,
        contexts,
        aggregates,
        strict_gate: PublicPreambleStrictGate {
            meaningful_required_minimum: 14,
            control_allowed_maximum: 2,
            meaningful_success_count,
            blank_success_count,
            permuted_success_count,
            identical_preamble_ticks,
            identical_task_ticks,
            all_energy_closed,
            passed,
        },
    })
}

fn run_arm(
    world: &WorldConfig,
    config: &PowerPlayConfig,
    genome: &OrganismGenome,
    task: &EcologyProgram,
    context_seed: u64,
    semantic_bits: &[u8],
    arm: PreambleArmKind,
) -> Result<PublicPreambleArmEvidence> {
    if semantic_bits.len() != FIXED_PREAMBLE_TICKS {
        bail!("public-preamble encoding did not produce the fixed tick count");
    }
    let mut sim =
        Simulation::new_with_champion_pool(world.clone(), context_seed, vec![genome.clone()])
            .map_err(|error| anyhow!("public-preamble world construction failed: {error}"))?;
    prepare_episode_state(&mut sim)?;
    let baseline = sim
        .organisms
        .first()
        .cloned()
        .ok_or_else(|| anyhow!("public-preamble world has no founder"))?;
    let prefix_initial_organism_energy_bits = baseline.energy.to_bits();
    let consumptions_before = sim.metrics().total_plant_consumptions;
    let mut prefix_trace = Vec::with_capacity(semantic_bits.len());
    let mut maximum_prefix_energy_ledger_residual = 0.0_f64;
    let mut maximum_prefix_energy_ledger_tolerance = 0.0_f64;

    for (bit_index, &semantic_bit) in semantic_bits.iter().enumerate() {
        reset_body_and_scene(&mut sim, &baseline)?;
        let energy_before = sim.organisms[0].energy;
        let (rendered_ray_offset, cue_position) = render_cue(&mut sim, semantic_bit, arm)?;
        let delta = sim.tick();
        let organism = sim
            .organisms
            .first()
            .ok_or_else(|| anyhow!("public-preamble founder died during cue prefix"))?;
        let ledger = delta.metrics.energy_ledger_last_turn;
        let maximum_residual = maximum_ledger_residual(ledger);
        maximum_prefix_energy_ledger_residual =
            maximum_prefix_energy_ledger_residual.max(maximum_residual);
        maximum_prefix_energy_ledger_tolerance =
            maximum_prefix_energy_ledger_tolerance.max(ledger.residual_tolerance);
        if organism.energy.to_bits() != energy_before.to_bits() {
            bail!(
                "zero-energy public cue changed organism energy at bit {bit_index}: {} -> {}",
                energy_before,
                organism.energy
            );
        }
        prefix_trace.push(PreambleTickTrace {
            bit_index,
            absolute_turn: delta.turn,
            semantic_bit,
            rendered_ray_offset,
            cue_q: cue_position.map(|position| position.0),
            cue_r: cue_position.map(|position| position.1),
            selected_action: organism.last_action_taken,
            q_after_tick: organism.q,
            r_after_tick: organism.r,
            facing_after_tick: organism.facing,
            organism_energy_before_bits: energy_before.to_bits(),
            organism_energy_after_bits: organism.energy.to_bits(),
            food_ray_activation_bits: organism
                .brain
                .sensory
                .iter()
                .take(3)
                .map(|sensor| sensor.neuron.activation.to_bits())
                .collect(),
            core_energy_ledger: ledger,
        });
    }

    let prefix_plant_consumptions = sim
        .metrics()
        .total_plant_consumptions
        .saturating_sub(consumptions_before);
    if prefix_plant_consumptions != 0 {
        bail!("side-ray public cue was unexpectedly consumed");
    }
    reset_body_and_scene(&mut sim, &baseline)?;
    let prefix_final_organism_energy_bits = sim.organisms[0].energy.to_bits();
    let prefix_energy_closed = prefix_initial_organism_energy_bits
        == prefix_final_organism_energy_bits
        && maximum_prefix_energy_ledger_residual <= maximum_prefix_energy_ledger_tolerance;
    let task_turn_origin = sim.turn();
    let task_episode = execute_program_on_sim(
        &mut sim,
        task,
        config,
        context_seed,
        true,
        true,
        task_turn_origin,
    )?;
    let task_ticks = config
        .ticks_per_stage
        .saturating_mul(task.stages.len() as u64);
    let task_prefix_successes = (1..=task.stages.len())
        .map(|prefix| {
            let deadline = config.ticks_per_stage.saturating_mul(prefix as u64);
            task_episode
                .completion_ticks
                .get(prefix - 1)
                .is_some_and(|tick| *tick <= deadline)
        })
        .collect::<Vec<_>>();
    let full_task_success =
        !task_prefix_successes.is_empty() && task_prefix_successes.iter().all(|success| *success);
    let total_ticks = sim.turn();
    let expected_total_ticks = FIXED_PREAMBLE_TICKS as u64 + task_ticks;
    if total_ticks != expected_total_ticks || task_episode.steps.len() as u64 != task_ticks {
        bail!(
            "matched-arm horizon mismatch: expected {expected_total_ticks} total and {task_ticks} task ticks, got {total_ticks} and {}",
            task_episode.steps.len()
        );
    }

    Ok(PublicPreambleArmEvidence {
        arm,
        preamble_ticks: semantic_bits.len(),
        task_ticks,
        total_ticks,
        prefix_plant_consumptions,
        prefix_initial_organism_energy_bits,
        prefix_final_organism_energy_bits,
        maximum_prefix_energy_ledger_residual,
        maximum_prefix_energy_ledger_tolerance,
        prefix_energy_closed,
        prefix_trace,
        task_prefix_successes,
        full_task_success,
        task_episode,
    })
}

fn reset_body_and_scene(sim: &mut Simulation, baseline: &OrganismState) -> Result<()> {
    if sim.organisms.len() != 1 || sim.organisms[0].id != baseline.id {
        bail!("public-preamble body reset requires the original single founder");
    }
    sim.foods.clear();
    sim.food_tiles.fill(false);
    sim.food_regrowth_due_turn.fill(u64::MAX);
    sim.food_regrowth_schedule.clear();
    sim.occupancy.fill(None);
    let live_brain = sim.organisms[0].brain.clone();
    let live_genome = sim.organisms[0].genome.clone();
    let mut reset = baseline.clone();
    reset.brain = live_brain;
    reset.genome = live_genome;
    sim.organisms[0] = reset;
    let center_idx = sim.cell_index(baseline.q, baseline.r);
    sim.occupancy[center_idx] = Some(Occupant::Organism(baseline.id));
    sim.debug_assert_consistent_state();
    Ok(())
}

fn render_cue(sim: &mut Simulation, semantic_bit: u8, arm: PreambleArmKind) -> Result<RenderedCue> {
    if semantic_bit > 1 {
        bail!("public-preamble semantic bits must be binary");
    }
    // Reserve exactly one ID in every arm on every tick. The blank control has
    // no food, but the later task food therefore receives the same identity.
    let id = FoodId(sim.next_food_id);
    sim.next_food_id = sim.next_food_id.saturating_add(1);
    let ray_offset = match arm {
        PreambleArmKind::Blank => return Ok((None, None)),
        PreambleArmKind::Meaningful => {
            if semantic_bit == 0 {
                -1
            } else {
                1
            }
        }
        PreambleArmKind::Permuted => {
            if semantic_bit == 0 {
                1
            } else {
                -1
            }
        }
    };
    let organism = sim
        .organisms
        .first()
        .ok_or_else(|| anyhow!("public-preamble cue rendering has no founder"))?;
    let direction = rotate_by_steps(organism.facing, ray_offset);
    let position = hex_neighbor(
        (organism.q, organism.r),
        direction,
        sim.config.world_width as i32,
    );
    let cell_idx = sim.cell_index(position.0, position.1);
    if sim.occupancy[cell_idx].is_some() {
        bail!("public-preamble cue target is occupied");
    }
    sim.occupancy[cell_idx] = Some(Occupant::Food(id));
    sim.foods.push(FoodState {
        id,
        q: position.0,
        r: position.1,
        energy: 0.0,
        kind: FoodKind::Plant,
        visual: food_visual(FoodKind::Plant),
    });
    sim.debug_assert_consistent_state();
    Ok((Some(ray_offset), Some(position)))
}

fn encode_resolved_program(
    program: &EcologyProgram,
    context_seed: u64,
) -> Result<(EcologyProgram, Vec<u8>)> {
    if program.stages.is_empty() || program.stages.len() > MAX_ENCODED_STAGES {
        bail!("public-preamble protocol supports task depths one and two");
    }
    let resolved_stages = program
        .stages
        .iter()
        .copied()
        .enumerate()
        .map(|(index, stage)| resolve_stage_context(stage, context_seed, index))
        .collect::<Vec<_>>();
    let resolved_program = EcologyProgram {
        stages: resolved_stages,
    };
    let bits = encode_program_bits(&resolved_program)?;
    Ok((resolved_program, bits))
}

/// Encode a program whose stages have already been resolved.  The protected
/// decoder experiment uses this entry point so public program meaning is not
/// coupled to the host world's nuisance seed.
fn encode_program_bits(program: &EcologyProgram) -> Result<Vec<u8>> {
    if program.stages.is_empty() || program.stages.len() > MAX_ENCODED_STAGES {
        bail!("public-preamble protocol supports task depths one and two");
    }
    let mut bits = Vec::with_capacity(FIXED_PREAMBLE_TICKS);
    bits.extend(MAGIC_BITS);
    push_bits(&mut bits, u64::from(PROTOCOL_VERSION), 2);
    push_bits(&mut bits, program.stages.len() as u64, 2);
    for slot in 0..MAX_ENCODED_STAGES {
        let Some(stage) = program.stages.get(slot).copied() else {
            bits.extend([0; 8]);
            continue;
        };
        bits.push(1);
        let relative_code = i16::from(stage.relative_turns) + 3;
        if !(0..=6).contains(&relative_code) || !(1..=3).contains(&stage.distance) {
            bail!("resolved PowerPlay stage cannot be represented by public protocol");
        }
        push_bits(&mut bits, relative_code as u64, 3);
        push_bits(&mut bits, u64::from(stage.distance - 1), 2);
        let motion_code = match stage.motion {
            ResourceMotion::Static => 0,
            ResourceMotion::FacingCoupledLeftDriftEveryThreeTicks => 1,
            ResourceMotion::FacingCoupledRightDriftEveryThreeTicks => 2,
        };
        push_bits(&mut bits, motion_code, 2);
    }
    bits.extend(TERMINATOR_BITS);
    if bits.len() != FIXED_PREAMBLE_TICKS {
        bail!(
            "public-preamble protocol length drifted: expected {FIXED_PREAMBLE_TICKS}, got {}",
            bits.len()
        );
    }
    Ok(bits)
}

fn push_bits(bits: &mut Vec<u8>, value: u64, width: usize) {
    for shift in (0..width).rev() {
        bits.push(((value >> shift) & 1) as u8);
    }
}

fn aggregate_arm(
    contexts: &[PublicPreambleContextEvidence],
    arm: PreambleArmKind,
    task_depth: usize,
) -> Result<PublicPreambleArmAggregate> {
    let mut prefix_success_counts = vec![0; task_depth];
    let mut full_task_success_count = 0;
    let mut all_energy_closed = true;
    for context in contexts {
        let evidence = context
            .arms
            .iter()
            .find(|candidate| candidate.arm == arm)
            .ok_or_else(|| anyhow!("public-preamble context is missing a matched arm"))?;
        for (index, success) in evidence.task_prefix_successes.iter().copied().enumerate() {
            prefix_success_counts[index] += usize::from(success);
        }
        full_task_success_count += usize::from(evidence.full_task_success);
        all_energy_closed &= evidence.prefix_energy_closed
            && evidence.task_episode.max_engine_energy_ledger_residual
                <= evidence.task_episode.max_engine_energy_ledger_tolerance
            && evidence.task_episode.resource_energy_closure_error.abs()
                <= evidence.task_episode.task_energy_residual_tolerance
            && evidence.task_episode.organism_energy_closure_error.abs()
                <= evidence.task_episode.organism_transfer_residual_tolerance;
    }
    Ok(PublicPreambleArmAggregate {
        arm,
        prefix_success_counts,
        full_task_success_count,
        all_energy_closed,
    })
}

fn aggregate_for(
    aggregates: &[PublicPreambleArmAggregate],
    arm: PreambleArmKind,
) -> Result<&PublicPreambleArmAggregate> {
    aggregates
        .iter()
        .find(|aggregate| aggregate.arm == arm)
        .ok_or_else(|| anyhow!("public-preamble aggregate is missing an arm"))
}

fn maximum_ledger_residual(row: EnergyLedgerRow) -> f64 {
    row.organism_residual
        .abs()
        .max(row.food_residual.abs())
        .max(row.artifact_residual.abs())
        .max(row.food_split_transfer_residual.abs())
        .max(row.artifact_release_transfer_residual.abs())
        .max(row.total_residual.abs())
        .max(row.transfer_residual.abs())
}

fn fingerprint<T: Serialize>(value: &T) -> Result<String> {
    exact_fingerprint(value).map_err(|error| anyhow!(error.to_string()))
}

// -------------------------------------------------------------------------
// Protected public-decoder capacity and descendant-checkpoint reuse falsifier.

const PUBLIC_DECODER_RESULT_SCHEMA_VERSION: u32 = 2;
const DECODER_MODULE_DEPTH: u32 = 100;
const DECODER_SEARCH_DOMAIN: u64 = 0x4445_434f_4445_5253;
const DECODER_ADMISSION_DOMAIN: u64 = 0x4445_434f_4445_5241;
const DECODER_TRANSFER_DOMAIN: u64 = 0x4445_434f_4445_5254;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicDecoderProbeConfig {
    pub source_powerplay: PowerPlayConfig,
    pub decoder_population_size: usize,
    pub decoder_generations: u32,
    pub decoder_module_width: usize,
    pub declaration_panel_size: usize,
    pub meaningful_minimum: usize,
    pub code_swap_minimum: usize,
    pub control_maximum: usize,
    pub ordinary_retention_minimum: usize,
}

impl Default for PublicDecoderProbeConfig {
    fn default() -> Self {
        Self {
            source_powerplay: PowerPlayConfig {
                run_seed: 7,
                max_depth: 2,
                ..PowerPlayConfig::default()
            },
            decoder_population_size: 64,
            decoder_generations: 120,
            decoder_module_width: 12,
            declaration_panel_size: 16,
            meaningful_minimum: 14,
            code_swap_minimum: 14,
            control_maximum: 2,
            ordinary_retention_minimum: 14,
        }
    }
}

impl PublicDecoderProbeConfig {
    fn validate(&self) -> Result<()> {
        self.source_powerplay.validate()?;
        if self.source_powerplay.max_depth != 2 {
            bail!("public decoder source PowerPlay depth must be exactly two");
        }
        if self.decoder_population_size < 2 || self.decoder_generations == 0 {
            bail!("decoder population must be >=2 and generations must be positive");
        }
        if self.decoder_module_width == 0 || self.decoder_module_width > 16 {
            bail!("decoder module width must be in 1..=16");
        }
        if self.declaration_panel_size != PROBE_CONTEXT_COUNT
            || self.meaningful_minimum != 14
            || self.code_swap_minimum != 14
            || self.control_maximum != 2
            || self.ordinary_retention_minimum != 14
        {
            bail!(
                "public decoder audit is fixed at 16 cases, positive >=14, controls <=2, and ordinary retention >=14"
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeclarationCase {
    pub evidence_id: String,
    pub simulation_seed: u64,
    pub program: EcologyProgram,
    pub program_fingerprint: String,
    pub expected_action: ActionType,
    pub code_swap_source_evidence_id: String,
    pub code_swap_program: EcologyProgram,
    pub code_swap_program_fingerprint: String,
    pub code_swap_expected_action: ActionType,
    pub code_swap_changes_expected_action: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeclarationPanel {
    pub name: String,
    pub cases: Vec<DeclarationCase>,
    pub panel_fingerprint: String,
    pub program_fingerprints_unique: bool,
    pub exact_programs_disjoint_from_training: bool,
    pub every_field_value_seen_in_training: bool,
    pub response_counts: Vec<(ActionType, usize)>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DeclarationArmKind {
    Meaningful,
    Blank,
    PolaritySwap,
    ValidCodeSwap,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DeclarationArmEvidence {
    pub arm: DeclarationArmKind,
    pub supplied_program_fingerprint: String,
    pub expected_action: ActionType,
    pub selected_action: ActionType,
    pub success: bool,
    pub total_ticks: u64,
    pub initial_energy_bits: u32,
    pub final_energy_bits: u32,
    pub plant_consumptions: u64,
    pub maximum_energy_ledger_residual: f64,
    pub maximum_energy_ledger_tolerance: f64,
    pub energy_closed: bool,
    pub prefix_trace: Vec<PreambleTickTrace>,
    pub declaration_energy_ledger: EnergyLedgerRow,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DeclarationCaseEvidence {
    pub evidence_id: String,
    pub simulation_seed: u64,
    pub program_fingerprint: String,
    pub expected_action: ActionType,
    pub code_swap_source_evidence_id: String,
    pub code_swap_program_fingerprint: String,
    pub code_swap_expected_action: ActionType,
    pub arms: Vec<DeclarationArmEvidence>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DeclarationPanelEvaluation {
    pub panel_name: String,
    pub panel_fingerprint: String,
    pub controller_fingerprint: String,
    pub meaningful_success_count: usize,
    pub blank_success_count: usize,
    pub polarity_swap_success_count: usize,
    pub valid_code_swap_success_count: usize,
    pub all_code_swaps_changed_expected_action: bool,
    pub all_energy_closed: bool,
    pub cases: Vec<DeclarationCaseEvidence>,
}

impl DeclarationPanelEvaluation {
    fn passes(&self, config: &PublicDecoderProbeConfig) -> bool {
        self.meaningful_success_count >= config.meaningful_minimum
            && self.valid_code_swap_success_count >= config.code_swap_minimum
            && self.blank_success_count <= config.control_maximum
            && self.polarity_swap_success_count <= config.control_maximum
            && self.all_code_swaps_changed_expected_action
            && self.all_energy_closed
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecoderGenerationEvidence {
    pub generation: u32,
    pub best_meaningful_success_count: usize,
    pub best_code_swap_success_count: usize,
    pub best_blank_success_count: usize,
    pub best_polarity_swap_success_count: usize,
    pub best_ordinary_success_count: usize,
    pub best_passed_search_gate: bool,
    pub qualifying_candidate_count: usize,
    pub qualifying_candidate_fingerprints_in_selection_order: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicDecoderModuleEvidence {
    pub hidden_node_ids: Vec<sim_types::GeneNodeId>,
    pub connection_innovations: Vec<sim_types::InnovationId>,
    pub zero_extension_matches_source: bool,
    pub knockout_restores_source_exactly: bool,
    pub protected_projection_verified: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicDecoderSourceEvidence {
    /// Complete source artifact, making the result fingerprint and checkpoint
    /// extraction independently recomputable from this file alone.
    pub source_result: PowerPlayResult,
    pub source_result_fingerprint: String,
    pub source_accepted_depth: u32,
    pub source_program: EcologyProgram,
    pub source_program_fingerprint: String,
    pub depth1_task: EcologyProgram,
    pub depth1_task_fingerprint: String,
    pub depth1_checkpoint_genome: OrganismGenome,
    pub depth1_checkpoint_fingerprint: String,
    pub depth2_task: Option<EcologyProgram>,
    pub depth2_task_fingerprint: Option<String>,
    pub depth2_checkpoint_genome: Option<OrganismGenome>,
    pub depth2_checkpoint_fingerprint: Option<String>,
    pub checkpoint_fingerprints_recompute_exactly: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicDecoderTrainingEvidence {
    pub generations: Vec<DecoderGenerationEvidence>,
    pub search_qualified: bool,
    pub search_exhausted_declared_budget: bool,
    pub declaration_admission_panel_touched: bool,
    pub stored_candidate_role: String,
    pub candidate_genome: Option<OrganismGenome>,
    pub candidate_genome_fingerprint: Option<String>,
    pub search_declaration_evaluation: Option<DeclarationPanelEvaluation>,
    pub search_ordinary_evaluation: Option<ProgramEvaluation>,
    pub admission_declaration_evaluation: Option<DeclarationPanelEvaluation>,
    pub admission_ordinary_evaluation: Option<ProgramEvaluation>,
    pub source_admission_declaration_evaluation: Option<DeclarationPanelEvaluation>,
    pub knockout_admission_declaration_evaluation: Option<DeclarationPanelEvaluation>,
    pub source_search_declaration_evaluation: DeclarationPanelEvaluation,
    pub knockout_search_declaration_evaluation: Option<DeclarationPanelEvaluation>,
    pub source_search_ordinary_evaluation: ProgramEvaluation,
    pub knockout_search_ordinary_evaluation: Option<ProgramEvaluation>,
    pub source_admission_ordinary_evaluation: ProgramEvaluation,
    pub knockout_admission_ordinary_evaluation: Option<ProgramEvaluation>,
    pub candidate_materializes_exactly: bool,
    pub knockout_restores_source_exactly: bool,
    pub knockout_reproduces_source_ordinary_behavior_exactly: bool,
    pub sealed_decoder_gate_passed: bool,
    pub stopped_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicDecoderReuseEvidence {
    pub status: String,
    pub reused_genome: Option<OrganismGenome>,
    pub reused_genome_fingerprint: Option<String>,
    pub exact_decoder_module_fingerprint_before: Option<String>,
    pub exact_decoder_module_fingerprint_after: Option<String>,
    pub module_reused_without_change: bool,
    pub reuse_declaration_evaluation: Option<DeclarationPanelEvaluation>,
    pub source_target_declaration_success_count: usize,
    pub reused_ordinary_evaluation: Option<ProgramEvaluation>,
    pub source_target_ordinary_evaluation: Option<ProgramEvaluation>,
    pub knockout_target_ordinary_evaluation: Option<ProgramEvaluation>,
    pub knockout_restores_target_checkpoint_exactly: bool,
    pub knockout_reproduces_target_ordinary_behavior_exactly: bool,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicDecoderProbeResult {
    pub result_schema_version: u32,
    pub algorithm: String,
    pub claim_scope: String,
    pub evaluator_owned: bool,
    pub evidentiary: bool,
    pub open_endedness_demonstrated: bool,
    pub config: PublicDecoderProbeConfig,
    pub base_world_fingerprint: String,
    pub effective_task_world: WorldConfig,
    pub effective_task_world_fingerprint: String,
    pub protocol: PublicPreambleProtocol,
    pub declaration_response_rule: String,
    pub valid_code_swap_scope: String,
    pub descendant_reuse_retention_scope: String,
    pub git_commit: Option<String>,
    pub executable_sha256: Option<String>,
    pub search_panel: DeclarationPanel,
    pub admission_panel: DeclarationPanel,
    pub reuse_panel: DeclarationPanel,
    pub panels_pairwise_seed_disjoint: bool,
    pub panels_pairwise_program_disjoint: bool,
    pub source: PublicDecoderSourceEvidence,
    pub module: PublicDecoderModuleEvidence,
    pub training: PublicDecoderTrainingEvidence,
    pub descendant_checkpoint_reuse: PublicDecoderReuseEvidence,
    pub verdict: String,
    pub verdict_reason: String,
    pub limitations: Vec<String>,
}

#[derive(Clone)]
struct ProgramCandidate {
    program: EcologyProgram,
    fingerprint: String,
    action: ActionType,
    fields: BTreeSet<String>,
    rank: u64,
}

#[derive(Clone, Copy, Default)]
struct DecoderScore {
    meaningful: usize,
    blank: usize,
    polarity: usize,
    code_swap: usize,
    ordinary: usize,
    energy_closed: bool,
}

impl DecoderScore {
    fn passes(self, config: &PublicDecoderProbeConfig) -> bool {
        self.meaningful >= config.meaningful_minimum
            && self.code_swap >= config.code_swap_minimum
            && self.blank <= config.control_maximum
            && self.polarity <= config.control_maximum
            && self.ordinary >= config.ordinary_retention_minimum
            && self.energy_closed
    }
}

#[derive(Clone)]
struct DecoderCandidate {
    genome: OrganismGenome,
    score: DecoderScore,
}

/// Train a protected public decoder before attempting to reuse that exact
/// residual in its descendant depth-2 PowerPlay checkpoint.
/// This remains an evaluator-owned capacity falsifier, never TCPE evidence.
pub fn run_public_decoder_probe(
    base_world: WorldConfig,
    config: PublicDecoderProbeConfig,
) -> Result<PublicDecoderProbeResult> {
    config.validate()?;
    let base_world_fingerprint = fingerprint(&base_world)?;
    let mut effective_task_world = base_world.clone();
    configure_task_world(&mut effective_task_world, &config.source_powerplay);
    let effective_task_world_fingerprint = fingerprint(&effective_task_world)?;

    let source_result = run_powerplay(base_world, config.source_powerplay.clone())?;
    let source_result_fingerprint = fingerprint(&source_result)?;
    let depth1 = source_result
        .depths
        .iter()
        .find(|entry| entry.depth == 1 && entry.accepted)
        .ok_or_else(|| anyhow!("source PowerPlay did not admit the required depth-1 checkpoint"))?;
    let depth1_checkpoint = depth1
        .candidate_genome
        .as_ref()
        .ok_or_else(|| anyhow!("accepted source depth-1 checkpoint has no exact genome"))?
        .clone();
    let depth1_task = EcologyProgram {
        stages: vec![depth1.generated_stage],
    };
    let depth1_checkpoint_fingerprint = fingerprint(&depth1_checkpoint)?;
    let depth1_task_fingerprint = fingerprint(&depth1_task)?;
    let depth2_entry = source_result
        .depths
        .iter()
        .find(|entry| entry.depth == 2 && entry.accepted);
    let depth2_checkpoint = depth2_entry.and_then(|entry| entry.candidate_genome.clone());
    let depth2_task = depth2_entry.map(|_| EcologyProgram {
        stages: source_result.program.stages[..2].to_vec(),
    });
    let depth2_checkpoint_fingerprint = depth2_checkpoint.as_ref().map(fingerprint).transpose()?;
    let depth2_task_fingerprint = depth2_task.as_ref().map(fingerprint).transpose()?;
    let source_program_fingerprint = fingerprint(&source_result.program)?;

    let mut reserved_seeds = config
        .source_powerplay
        .search_seeds
        .iter()
        .chain(&config.source_powerplay.episode_seeds)
        .copied()
        .collect::<BTreeSet<_>>();
    reserved_seeds.insert(config.source_powerplay.run_seed);
    let mut used_programs = BTreeSet::new();
    let search_panel = build_declaration_panel(
        "search",
        DECODER_SEARCH_DOMAIN ^ config.source_powerplay.run_seed,
        &mut reserved_seeds,
        &mut used_programs,
        None,
    )?;
    let training_fields = panel_field_values(&search_panel);
    let training_programs = panel_program_fingerprints(&search_panel);
    let admission_panel = build_declaration_panel(
        "sealed_admission",
        DECODER_ADMISSION_DOMAIN ^ config.source_powerplay.run_seed,
        &mut reserved_seeds,
        &mut used_programs,
        Some(&training_fields),
    )?;
    let reuse_panel = build_declaration_panel(
        "sealed_descendant_checkpoint_reuse",
        DECODER_TRANSFER_DOMAIN ^ config.source_powerplay.run_seed,
        &mut reserved_seeds,
        &mut used_programs,
        Some(&training_fields),
    )?;
    let panels_pairwise_seed_disjoint =
        panels_seed_disjoint([&search_panel, &admission_panel, &reuse_panel]);
    let panels_pairwise_program_disjoint = [&admission_panel, &reuse_panel].iter().all(|panel| {
        panel
            .cases
            .iter()
            .all(|case| !training_programs.contains(&case.program_fingerprint))
    }) && admission_panel.cases.iter().all(|left| {
        reuse_panel
            .cases
            .iter()
            .all(|right| left.program_fingerprint != right.program_fingerprint)
    });
    if !panels_pairwise_seed_disjoint || !panels_pairwise_program_disjoint {
        bail!("public decoder panels are not pairwise disjoint");
    }

    let protection = ProtectedResidual::seal(&depth1_checkpoint)
        .map_err(|error| anyhow!("cannot seal source depth-1 checkpoint: {error}"))?;
    let protected_seed = protection.seed_extension();
    let (mut zero_extended, module) = add_zero_residual_module(
        &protected_seed,
        DECODER_MODULE_DEPTH,
        &PowerPlayConfig {
            module_width: config.decoder_module_width,
            ..config.source_powerplay.clone()
        },
    )?;
    protection.project(&mut zero_extended);
    protection
        .verify(&zero_extended)
        .map_err(|error| anyhow!("zero decoder extension violates protection: {error}"))?;
    let zero_extension_matches_source =
        knockout_module(&zero_extended, &module) == depth1_checkpoint;

    let mut search_task_config = config.source_powerplay.clone();
    search_task_config
        .episode_seeds
        .clone_from(&config.source_powerplay.search_seeds);
    let (selected, best_observed, generations) = search_decoder(DecoderSearchRequest {
        world: &effective_task_world,
        zero_extended: &zero_extended,
        module: &module,
        protection: &protection,
        ordinary_task: &depth1_task,
        ordinary_config: &search_task_config,
        search_panel: &search_panel,
        config: &config,
    })?;

    let source_search_declaration_evaluation = evaluate_declaration_panel(
        &effective_task_world,
        &depth1_checkpoint,
        &search_panel,
        true,
    )?;
    let source_search_ordinary_evaluation = evaluate_program(
        &effective_task_world,
        &depth1_checkpoint,
        &depth1_task,
        &search_task_config,
        true,
    )?;
    let source_admission_ordinary_evaluation = evaluate_program(
        &effective_task_world,
        &depth1_checkpoint,
        &depth1_task,
        &config.source_powerplay,
        true,
    )?;
    let mut training = PublicDecoderTrainingEvidence {
        generations,
        search_qualified: selected.is_some(),
        search_exhausted_declared_budget: selected.is_none(),
        declaration_admission_panel_touched: selected.is_some(),
        stored_candidate_role: if selected.is_some() {
            "first_search_qualified".to_string()
        } else {
            "best_observed_search_failure".to_string()
        },
        candidate_genome: None,
        candidate_genome_fingerprint: None,
        search_declaration_evaluation: None,
        search_ordinary_evaluation: None,
        admission_declaration_evaluation: None,
        admission_ordinary_evaluation: None,
        source_admission_declaration_evaluation: None,
        knockout_admission_declaration_evaluation: None,
        source_search_declaration_evaluation,
        knockout_search_declaration_evaluation: None,
        source_search_ordinary_evaluation,
        knockout_search_ordinary_evaluation: None,
        source_admission_ordinary_evaluation,
        knockout_admission_ordinary_evaluation: None,
        candidate_materializes_exactly: false,
        knockout_restores_source_exactly: false,
        knockout_reproduces_source_ordinary_behavior_exactly: false,
        sealed_decoder_gate_passed: false,
        stopped_reason: None,
    };

    let mut decoder_candidate = None;
    if let Some(selected) = selected {
        let candidate_fingerprint = fingerprint(&selected.genome)?;
        let search_declaration = evaluate_declaration_panel(
            &effective_task_world,
            &selected.genome,
            &search_panel,
            true,
        )?;
        let search_ordinary = evaluate_program(
            &effective_task_world,
            &selected.genome,
            &depth1_task,
            &search_task_config,
            false,
        )?;
        let admission_declaration = evaluate_declaration_panel(
            &effective_task_world,
            &selected.genome,
            &admission_panel,
            true,
        )?;
        let admission_ordinary = evaluate_program(
            &effective_task_world,
            &selected.genome,
            &depth1_task,
            &config.source_powerplay,
            true,
        )?;
        let knockout = knockout_module(&selected.genome, &module);
        let knockout_ordinary = evaluate_program(
            &effective_task_world,
            &knockout,
            &depth1_task,
            &config.source_powerplay,
            true,
        )?;
        let knockout_search_ordinary = evaluate_program(
            &effective_task_world,
            &knockout,
            &depth1_task,
            &search_task_config,
            true,
        )?;
        let knockout_search_declaration =
            evaluate_declaration_panel(&effective_task_world, &knockout, &search_panel, true)?;
        let source_declaration = evaluate_declaration_panel(
            &effective_task_world,
            &depth1_checkpoint,
            &admission_panel,
            true,
        )?;
        let knockout_declaration =
            evaluate_declaration_panel(&effective_task_world, &knockout, &admission_panel, true)?;
        let candidate_materializes_exactly = materialize_genome(
            &effective_task_world,
            &selected.genome,
            config.source_powerplay.run_seed,
        )? == selected.genome;
        let knockout_restores_source_exactly = knockout == depth1_checkpoint;
        let knockout_reproduces_source_ordinary_behavior_exactly =
            knockout_ordinary == training.source_admission_ordinary_evaluation;
        let admission_ordinary_success = all_deadline_success_count(
            &admission_ordinary,
            config.source_powerplay.ticks_per_stage,
        );
        let sealed_decoder_gate_passed = search_declaration.passes(&config)
            && all_deadline_success_count(
                &search_ordinary,
                config.source_powerplay.ticks_per_stage,
            ) >= config.ordinary_retention_minimum
            && admission_declaration.passes(&config)
            && admission_ordinary_success >= config.ordinary_retention_minimum
            && source_declaration.meaningful_success_count <= config.control_maximum
            && knockout_declaration.meaningful_success_count <= config.control_maximum
            && candidate_materializes_exactly
            && knockout_restores_source_exactly
            && knockout_reproduces_source_ordinary_behavior_exactly;

        training.candidate_genome = Some(selected.genome.clone());
        training.candidate_genome_fingerprint = Some(candidate_fingerprint);
        training.search_declaration_evaluation = Some(search_declaration);
        training.search_ordinary_evaluation = Some(search_ordinary);
        training.admission_declaration_evaluation = Some(admission_declaration);
        training.admission_ordinary_evaluation = Some(admission_ordinary);
        training.source_admission_declaration_evaluation = Some(source_declaration);
        training.knockout_admission_declaration_evaluation = Some(knockout_declaration);
        training.knockout_search_ordinary_evaluation = Some(knockout_search_ordinary);
        training.knockout_search_declaration_evaluation = Some(knockout_search_declaration);
        training.knockout_admission_ordinary_evaluation = Some(knockout_ordinary);
        training.candidate_materializes_exactly = candidate_materializes_exactly;
        training.knockout_restores_source_exactly = knockout_restores_source_exactly;
        training.knockout_reproduces_source_ordinary_behavior_exactly =
            knockout_reproduces_source_ordinary_behavior_exactly;
        training.sealed_decoder_gate_passed = sealed_decoder_gate_passed;
        if sealed_decoder_gate_passed {
            decoder_candidate = Some(selected.genome);
        } else {
            training.stopped_reason = Some(
                "the first search-qualified decoder failed its one-shot sealed admission, source-control, knockout, or ordinary-retention gate"
                    .to_string(),
            );
        }
    } else {
        protection
            .verify(&best_observed.genome)
            .map_err(|error| anyhow!("stored best decoder violates protection: {error}"))?;
        let best_fingerprint = fingerprint(&best_observed.genome)?;
        let best_declaration = evaluate_declaration_panel(
            &effective_task_world,
            &best_observed.genome,
            &search_panel,
            true,
        )?;
        let best_ordinary = evaluate_program(
            &effective_task_world,
            &best_observed.genome,
            &depth1_task,
            &search_task_config,
            true,
        )?;
        let knockout = knockout_module(&best_observed.genome, &module);
        let knockout_search_ordinary = evaluate_program(
            &effective_task_world,
            &knockout,
            &depth1_task,
            &search_task_config,
            true,
        )?;
        let knockout_search_declaration =
            evaluate_declaration_panel(&effective_task_world, &knockout, &search_panel, true)?;
        training.candidate_materializes_exactly = materialize_genome(
            &effective_task_world,
            &best_observed.genome,
            config.source_powerplay.run_seed,
        )? == best_observed.genome;
        training.knockout_restores_source_exactly = knockout == depth1_checkpoint;
        training.knockout_reproduces_source_ordinary_behavior_exactly =
            knockout_search_ordinary == training.source_search_ordinary_evaluation;
        training.knockout_search_ordinary_evaluation = Some(knockout_search_ordinary);
        training.knockout_search_declaration_evaluation = Some(knockout_search_declaration);
        training.candidate_genome = Some(best_observed.genome.clone());
        training.candidate_genome_fingerprint = Some(best_fingerprint);
        training.search_declaration_evaluation = Some(best_declaration);
        training.search_ordinary_evaluation = Some(best_ordinary);
        training.stopped_reason = Some(format!(
            "no protected decoder passed the complete mutable search gate in {} generations",
            config.decoder_generations
        ));
    }

    let descendant_checkpoint_reuse =
        if let (Some(decoder), Some(target_checkpoint), Some(target_task)) = (
            decoder_candidate.as_ref(),
            depth2_checkpoint.as_ref(),
            depth2_task.as_ref(),
        ) {
            evaluate_descendant_checkpoint_reuse(
                &effective_task_world,
                decoder,
                &module,
                target_checkpoint,
                target_task,
                &reuse_panel,
                &config,
            )?
        } else {
            PublicDecoderReuseEvidence {
                status: if !training.search_qualified {
                    "not_attempted_because_no_search_candidate_qualified".to_string()
                } else if !training.sealed_decoder_gate_passed {
                    "not_attempted_because_first_precommitted_decoder_failed_sealed_gate"
                        .to_string()
                } else {
                    "not_attempted_because_source_depth2_checkpoint_missing".to_string()
                },
                reused_genome: None,
                reused_genome_fingerprint: None,
                exact_decoder_module_fingerprint_before: None,
                exact_decoder_module_fingerprint_after: None,
                module_reused_without_change: false,
                reuse_declaration_evaluation: None,
                source_target_declaration_success_count: 0,
                reused_ordinary_evaluation: None,
                source_target_ordinary_evaluation: None,
                knockout_target_ordinary_evaluation: None,
                knockout_restores_target_checkpoint_exactly: false,
                knockout_reproduces_target_ordinary_behavior_exactly: false,
                passed: false,
            }
        };

    let source = PublicDecoderSourceEvidence {
        source_result: source_result.clone(),
        source_result_fingerprint: source_result_fingerprint.clone(),
        source_accepted_depth: source_result.accepted_depth,
        source_program: source_result.program.clone(),
        source_program_fingerprint,
        depth1_task,
        depth1_task_fingerprint,
        depth1_checkpoint_genome: depth1_checkpoint.clone(),
        depth1_checkpoint_fingerprint: depth1_checkpoint_fingerprint.clone(),
        depth2_task,
        depth2_task_fingerprint,
        depth2_checkpoint_genome: depth2_checkpoint.clone(),
        depth2_checkpoint_fingerprint: depth2_checkpoint_fingerprint.clone(),
        checkpoint_fingerprints_recompute_exactly: fingerprint(&source_result)?
            == source_result_fingerprint
            && fingerprint(&depth1_checkpoint)? == depth1_checkpoint_fingerprint
            && match (&depth2_checkpoint, &depth2_checkpoint_fingerprint) {
                (Some(genome), Some(expected)) => fingerprint(genome)? == *expected,
                (None, None) => true,
                _ => false,
            },
    };
    let verdict = if descendant_checkpoint_reuse.passed {
        "decoder_and_descendant_checkpoint_reuse_passed"
    } else if training.sealed_decoder_gate_passed && depth2_checkpoint.is_none() {
        "decoder_passed_reuse_not_evaluable_source_depth2_missing"
    } else if training.sealed_decoder_gate_passed {
        "decoder_passed_descendant_checkpoint_reuse_failed"
    } else if training.search_qualified {
        "first_precommitted_search_qualifier_failed_one_shot_sealed_admission"
    } else {
        "no_search_qualified_decoder_for_optimizer_seed_encoding_and_budget"
    }
    .to_string();
    let verdict_reason = if descendant_checkpoint_reuse.passed {
        "the protected decoder passed one-shot held-out declaration controls, retained the source task, and was reused unchanged in its descendant depth-2 checkpoint while retaining that checkpoint's task".to_string()
    } else if training.sealed_decoder_gate_passed && depth2_checkpoint.is_none() {
        "the protected decoder passed its one-shot sealed gate, but descendant-checkpoint reuse was not evaluable because the source PowerPlay run did not admit depth 2".to_string()
    } else if training.sealed_decoder_gate_passed {
        "the decoder capacity premise survived, but exact module reuse did not pass the sealed descendant-checkpoint declaration and retention gate".to_string()
    } else if training.search_qualified {
        "the first deterministically precommitted search-qualified decoder failed its single sealed audit; other same-generation qualifiers and later generations were not tested".to_string()
    } else {
        format!(
            "no evaluated candidate met the complete mutable search gate across this optimizer, source seed, encoding, and {}-by-{} budget; the sealed decoder declaration-admission panel remained untouched and descendant-checkpoint reuse was not attempted",
            config.decoder_population_size, config.decoder_generations
        )
    };

    Ok(PublicDecoderProbeResult {
        result_schema_version: PUBLIC_DECODER_RESULT_SCHEMA_VERSION,
        algorithm: "protected_public_program_decoder_descendant_checkpoint_reuse_v1".to_string(),
        claim_scope: "evaluator-owned protected decoder-capacity and exact descendant-checkpoint module-reuse falsifier; not cross-branch transfer, TCPE, or open-endedness evidence".to_string(),
        evaluator_owned: true,
        evidentiary: false,
        open_endedness_demonstrated: false,
        config,
        base_world_fingerprint,
        effective_task_world,
        effective_task_world_fingerprint,
        protocol: PublicPreambleProtocol::canonical(),
        declaration_response_rule: "for two valid stages, compute (relative0+3) + 3*(relative1+3) + 2*(distance0-1) + 5*(distance1-1) + 7*motion0 + 11*motion1 modulo four; 0=turn_left, 1=turn_right, 2=forward, 3=eat".to_string(),
        valid_code_swap_scope: "positive declaration equivariance only: re-pair another panel case's valid public program with that supplied program's checksum-derived expected action under the host case's simulation seed and identical empty declaration scene; it is not a wrong-code/same-ecology-task intervention, and no ecology program executes in a declaration arm".to_string(),
        descendant_reuse_retention_scope: "one composite depth-2 PowerPlay task evaluated by the conjunction of both chronological deadlines on the frozen source admission contexts; this is not separate standalone replay of depth-1 and depth-2 archive obligations".to_string(),
        git_commit: None,
        executable_sha256: None,
        search_panel,
        admission_panel,
        reuse_panel,
        panels_pairwise_seed_disjoint,
        panels_pairwise_program_disjoint,
        source,
        module: PublicDecoderModuleEvidence {
            hidden_node_ids: module.hidden_node_ids.clone(),
            connection_innovations: module.connection_innovations.clone(),
            zero_extension_matches_source,
            knockout_restores_source_exactly: training.knockout_restores_source_exactly,
            protected_projection_verified: true,
        },
        training,
        descendant_checkpoint_reuse,
        verdict,
        verdict_reason,
        limitations: vec![
            "the declaration scheduler, zero-energy cues, pose reset, and score are evaluator-owned".to_string(),
            "declaration success has no payoff and executes no ecology task; decoder topology and computation are not energy-priced".to_string(),
            "the public grammar is finite and capped at two stages; passing is decoder capacity, not open-endedness".to_string(),
            "failure is exact for this declared search budget and encoding, not a proof that every possible decoder is impossible".to_string(),
            "descendant-checkpoint reuse is attempted only after the protected decoder passes its sealed gate; it is not foreign or cross-branch transfer".to_string(),
            "the artifact embeds exact configs, panels, source PowerPlay result, checkpoints, candidate genomes, and fingerprints, but does not embed the git commit or executable SHA-256; those must accompany the external artifact hash".to_string(),
        ],
    })
}

fn declaration_action(program: &EcologyProgram) -> Result<ActionType> {
    if program.stages.len() != 2 {
        bail!("decoder declaration programs must contain exactly two stages");
    }
    let left = program.stages[0];
    let right = program.stages[1];
    let score = i32::from(left.relative_turns)
        + 3
        + 3 * (i32::from(right.relative_turns) + 3)
        + 2 * i32::from(left.distance - 1)
        + 5 * i32::from(right.distance - 1)
        + 7 * i32::from(motion_code(left.motion))
        + 11 * i32::from(motion_code(right.motion));
    Ok(match score.rem_euclid(4) {
        0 => ActionType::TurnLeft,
        1 => ActionType::TurnRight,
        2 => ActionType::Forward,
        3 => ActionType::Eat,
        _ => unreachable!(),
    })
}

fn motion_code(motion: ResourceMotion) -> u8 {
    match motion {
        ResourceMotion::Static => 0,
        ResourceMotion::FacingCoupledLeftDriftEveryThreeTicks => 1,
        ResourceMotion::FacingCoupledRightDriftEveryThreeTicks => 2,
    }
}

fn declaration_action_index(action: ActionType) -> Result<usize> {
    match action {
        ActionType::TurnLeft => Ok(0),
        ActionType::TurnRight => Ok(1),
        ActionType::Forward => Ok(2),
        ActionType::Eat => Ok(3),
        ActionType::Idle | ActionType::Attack => {
            bail!("declaration response is not one of the four enabled physical actions")
        }
    }
}

fn program_field_values(program: &EcologyProgram) -> BTreeSet<String> {
    let mut values = BTreeSet::new();
    for (index, stage) in program.stages.iter().enumerate() {
        values.insert(format!(
            "stage{index}:relative_turns:{}",
            stage.relative_turns
        ));
        values.insert(format!("stage{index}:distance:{}", stage.distance));
        values.insert(format!("stage{index}:motion:{}", motion_code(stage.motion)));
    }
    values
}

fn enumerate_program_candidates(domain: u64) -> Result<Vec<ProgramCandidate>> {
    let stages = grammar();
    let mut candidates = Vec::with_capacity(stages.len() * stages.len());
    for (left_index, &left) in stages.iter().enumerate() {
        for (right_index, &right) in stages.iter().enumerate() {
            let program = EcologyProgram {
                stages: vec![left, right],
            };
            let fingerprint = fingerprint(&program)?;
            let action = declaration_action(&program)?;
            let lexical_index = left_index
                .checked_mul(stages.len())
                .and_then(|value| value.checked_add(right_index))
                .ok_or_else(|| anyhow!("program enumeration index overflow"))?;
            candidates.push(ProgramCandidate {
                fields: program_field_values(&program),
                program,
                fingerprint,
                action,
                rank: mix64(domain ^ lexical_index as u64),
            });
        }
    }
    candidates.sort_by(|left, right| {
        left.rank
            .cmp(&right.rank)
            .then_with(|| left.fingerprint.cmp(&right.fingerprint))
    });
    Ok(candidates)
}

fn build_declaration_panel(
    name: &str,
    domain: u64,
    reserved_seeds: &mut BTreeSet<u64>,
    used_programs: &mut BTreeSet<String>,
    allowed_fields: Option<&BTreeSet<String>>,
) -> Result<DeclarationPanel> {
    let candidates = enumerate_program_candidates(domain)?;
    let mut selected = Vec::<ProgramCandidate>::with_capacity(PROBE_CONTEXT_COUNT);
    let mut response_counts = [0_usize; 4];
    let mut covered_fields = BTreeSet::<String>::new();
    while selected.len() < PROBE_CONTEXT_COUNT {
        let mut best: Option<(usize, usize)> = None;
        for (index, candidate) in candidates.iter().enumerate() {
            if used_programs.contains(&candidate.fingerprint)
                || selected
                    .iter()
                    .any(|entry| entry.fingerprint == candidate.fingerprint)
                || allowed_fields.is_some_and(|allowed| !candidate.fields.is_subset(allowed))
            {
                continue;
            }
            let action_index = declaration_action_index(candidate.action)?;
            if response_counts[action_index] >= PROBE_CONTEXT_COUNT / 4 {
                continue;
            }
            let novelty = candidate.fields.difference(&covered_fields).count();
            if best.is_none_or(|(_, incumbent_novelty)| novelty > incumbent_novelty) {
                best = Some((index, novelty));
            }
        }
        let (index, _) =
            best.ok_or_else(|| anyhow!("could not construct balanced declaration panel `{name}`"))?;
        let candidate = candidates[index].clone();
        response_counts[declaration_action_index(candidate.action)?] += 1;
        covered_fields.extend(candidate.fields.iter().cloned());
        selected.push(candidate);
    }

    let groups = (0..4)
        .map(|action_index| {
            selected
                .iter()
                .enumerate()
                .filter_map(|(index, candidate)| {
                    (declaration_action_index(candidate.action).ok() == Some(action_index))
                        .then_some(index)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut occurrences = [0_usize; 4];
    let mut cases = Vec::with_capacity(selected.len());
    for (index, candidate) in selected.iter().enumerate() {
        let action_index = declaration_action_index(candidate.action)?;
        let occurrence = occurrences[action_index];
        occurrences[action_index] += 1;
        let swap_index = groups[(action_index + 1) % 4][occurrence % groups[0].len()];
        let swap = &selected[swap_index];
        let simulation_seed = derive_decoder_seed(domain, index, reserved_seeds);
        cases.push(DeclarationCase {
            evidence_id: format!("{name}:{index:02}"),
            simulation_seed,
            program: candidate.program.clone(),
            program_fingerprint: candidate.fingerprint.clone(),
            expected_action: candidate.action,
            code_swap_source_evidence_id: format!("{name}:{swap_index:02}"),
            code_swap_program: swap.program.clone(),
            code_swap_program_fingerprint: swap.fingerprint.clone(),
            code_swap_expected_action: swap.action,
            code_swap_changes_expected_action: swap.action != candidate.action,
        });
    }
    for candidate in &selected {
        used_programs.insert(candidate.fingerprint.clone());
    }
    let panel_fingerprint = fingerprint(&cases)?;
    let program_fingerprints_unique = cases
        .iter()
        .map(|case| &case.program_fingerprint)
        .collect::<BTreeSet<_>>()
        .len()
        == cases.len();
    let response_counts = [
        ActionType::TurnLeft,
        ActionType::TurnRight,
        ActionType::Forward,
        ActionType::Eat,
    ]
    .into_iter()
    .map(|action| {
        (
            action,
            cases
                .iter()
                .filter(|case| case.expected_action == action)
                .count(),
        )
    })
    .collect();
    Ok(DeclarationPanel {
        name: name.to_string(),
        cases,
        panel_fingerprint,
        program_fingerprints_unique,
        exact_programs_disjoint_from_training: allowed_fields.is_some(),
        every_field_value_seen_in_training: allowed_fields
            .is_none_or(|allowed| covered_fields.is_subset(allowed)),
        response_counts,
    })
}

fn derive_decoder_seed(domain: u64, index: usize, reserved: &mut BTreeSet<u64>) -> u64 {
    let mut nonce = index as u64;
    loop {
        let seed = mix64(domain ^ nonce.wrapping_mul(0x9e37_79b9_7f4a_7c15));
        nonce = nonce.wrapping_add(PROBE_CONTEXT_COUNT as u64);
        if reserved.insert(seed) {
            return seed;
        }
    }
}

fn panel_field_values(panel: &DeclarationPanel) -> BTreeSet<String> {
    panel
        .cases
        .iter()
        .flat_map(|case| program_field_values(&case.program))
        .collect()
}

fn panel_program_fingerprints(panel: &DeclarationPanel) -> BTreeSet<String> {
    panel
        .cases
        .iter()
        .map(|case| case.program_fingerprint.clone())
        .collect()
}

fn panels_seed_disjoint<const N: usize>(panels: [&DeclarationPanel; N]) -> bool {
    let mut seeds = BTreeSet::new();
    panels.iter().all(|panel| {
        panel
            .cases
            .iter()
            .all(|case| seeds.insert(case.simulation_seed))
    })
}

fn evaluate_declaration_panel(
    world: &WorldConfig,
    genome: &OrganismGenome,
    panel: &DeclarationPanel,
    capture_trace: bool,
) -> Result<DeclarationPanelEvaluation> {
    let controller_fingerprint = fingerprint(genome)?;
    let mut cases = Vec::with_capacity(panel.cases.len());
    for case in &panel.cases {
        let mut arms = Vec::with_capacity(4);
        for arm in [
            DeclarationArmKind::Meaningful,
            DeclarationArmKind::Blank,
            DeclarationArmKind::PolaritySwap,
            DeclarationArmKind::ValidCodeSwap,
        ] {
            arms.push(run_declaration_arm(
                world,
                genome,
                case,
                arm,
                capture_trace,
            )?);
        }
        cases.push(DeclarationCaseEvidence {
            evidence_id: case.evidence_id.clone(),
            simulation_seed: case.simulation_seed,
            program_fingerprint: case.program_fingerprint.clone(),
            expected_action: case.expected_action,
            code_swap_source_evidence_id: case.code_swap_source_evidence_id.clone(),
            code_swap_program_fingerprint: case.code_swap_program_fingerprint.clone(),
            code_swap_expected_action: case.code_swap_expected_action,
            arms,
        });
    }
    let successes = |arm: DeclarationArmKind| {
        cases
            .iter()
            .filter(|case| {
                case.arms
                    .iter()
                    .find(|entry| entry.arm == arm)
                    .is_some_and(|entry| entry.success)
            })
            .count()
    };
    let all_energy_closed = cases
        .iter()
        .flat_map(|case| &case.arms)
        .all(|arm| arm.energy_closed);
    Ok(DeclarationPanelEvaluation {
        panel_name: panel.name.clone(),
        panel_fingerprint: panel.panel_fingerprint.clone(),
        controller_fingerprint,
        meaningful_success_count: successes(DeclarationArmKind::Meaningful),
        blank_success_count: successes(DeclarationArmKind::Blank),
        polarity_swap_success_count: successes(DeclarationArmKind::PolaritySwap),
        valid_code_swap_success_count: successes(DeclarationArmKind::ValidCodeSwap),
        all_code_swaps_changed_expected_action: panel
            .cases
            .iter()
            .all(|case| case.code_swap_changes_expected_action),
        all_energy_closed,
        cases,
    })
}

fn run_declaration_arm(
    world: &WorldConfig,
    genome: &OrganismGenome,
    case: &DeclarationCase,
    arm: DeclarationArmKind,
    capture_trace: bool,
) -> Result<DeclarationArmEvidence> {
    let (supplied_program, expected_action, render_arm) = match arm {
        DeclarationArmKind::Meaningful => (
            &case.program,
            case.expected_action,
            PreambleArmKind::Meaningful,
        ),
        DeclarationArmKind::Blank => (&case.program, case.expected_action, PreambleArmKind::Blank),
        DeclarationArmKind::PolaritySwap => (
            &case.program,
            case.expected_action,
            PreambleArmKind::Permuted,
        ),
        DeclarationArmKind::ValidCodeSwap => (
            &case.code_swap_program,
            case.code_swap_expected_action,
            PreambleArmKind::Meaningful,
        ),
    };
    let supplied_program_fingerprint = fingerprint(supplied_program)?;
    let semantic_bits = encode_program_bits(supplied_program)?;
    let mut sim = Simulation::new_with_champion_pool(
        world.clone(),
        case.simulation_seed,
        vec![genome.clone()],
    )
    .map_err(|error| anyhow!("public decoder world construction failed: {error}"))?;
    prepare_episode_state(&mut sim)?;
    let baseline = sim
        .organisms
        .first()
        .cloned()
        .ok_or_else(|| anyhow!("public decoder world has no founder"))?;
    let initial_energy_bits = baseline.energy.to_bits();
    let consumptions_before = sim.metrics().total_plant_consumptions;
    let mut maximum_energy_ledger_residual = 0.0_f64;
    let mut maximum_energy_ledger_tolerance = 0.0_f64;
    let mut prefix_trace = Vec::with_capacity(if capture_trace {
        FIXED_PREAMBLE_TICKS
    } else {
        0
    });
    for (bit_index, semantic_bit) in semantic_bits.iter().copied().enumerate() {
        reset_body_and_scene(&mut sim, &baseline)?;
        let energy_before = sim.organisms[0].energy;
        let (rendered_ray_offset, cue_position) = render_cue(&mut sim, semantic_bit, render_arm)?;
        let delta = sim.tick();
        let organism = sim
            .organisms
            .first()
            .ok_or_else(|| anyhow!("public decoder founder died during preamble"))?;
        let ledger = delta.metrics.energy_ledger_last_turn;
        maximum_energy_ledger_residual =
            maximum_energy_ledger_residual.max(maximum_ledger_residual(ledger));
        maximum_energy_ledger_tolerance =
            maximum_energy_ledger_tolerance.max(ledger.residual_tolerance);
        if organism.energy.to_bits() != energy_before.to_bits() {
            bail!("zero-energy decoder cue changed organism energy");
        }
        if capture_trace {
            prefix_trace.push(PreambleTickTrace {
                bit_index,
                absolute_turn: delta.turn,
                semantic_bit,
                rendered_ray_offset,
                cue_q: cue_position.map(|position| position.0),
                cue_r: cue_position.map(|position| position.1),
                selected_action: organism.last_action_taken,
                q_after_tick: organism.q,
                r_after_tick: organism.r,
                facing_after_tick: organism.facing,
                organism_energy_before_bits: energy_before.to_bits(),
                organism_energy_after_bits: organism.energy.to_bits(),
                food_ray_activation_bits: organism
                    .brain
                    .sensory
                    .iter()
                    .take(3)
                    .map(|sensor| sensor.neuron.activation.to_bits())
                    .collect(),
                core_energy_ledger: ledger,
            });
        }
    }
    reset_body_and_scene(&mut sim, &baseline)?;
    let declaration_delta = sim.tick();
    let declaration_energy_ledger = declaration_delta.metrics.energy_ledger_last_turn;
    maximum_energy_ledger_residual =
        maximum_energy_ledger_residual.max(maximum_ledger_residual(declaration_energy_ledger));
    maximum_energy_ledger_tolerance =
        maximum_energy_ledger_tolerance.max(declaration_energy_ledger.residual_tolerance);
    let organism = sim
        .organisms
        .first()
        .ok_or_else(|| anyhow!("public decoder founder died on declaration tick"))?;
    let selected_action = organism.last_action_taken;
    let final_energy_bits = organism.energy.to_bits();
    let plant_consumptions = sim
        .metrics()
        .total_plant_consumptions
        .saturating_sub(consumptions_before);
    let total_ticks = sim.turn();
    let energy_closed = initial_energy_bits == final_energy_bits
        && plant_consumptions == 0
        && maximum_energy_ledger_residual <= maximum_energy_ledger_tolerance
        && total_ticks == (FIXED_PREAMBLE_TICKS + 1) as u64;
    Ok(DeclarationArmEvidence {
        arm,
        supplied_program_fingerprint,
        expected_action,
        selected_action,
        success: selected_action == expected_action,
        total_ticks,
        initial_energy_bits,
        final_energy_bits,
        plant_consumptions,
        maximum_energy_ledger_residual,
        maximum_energy_ledger_tolerance,
        energy_closed,
        prefix_trace,
        declaration_energy_ledger,
    })
}

fn all_deadline_success_count(evaluation: &ProgramEvaluation, ticks_per_stage: u64) -> usize {
    evaluation
        .episodes
        .iter()
        .filter(|episode| {
            !episode.resolved_stages.is_empty()
                && episode
                    .resolved_stages
                    .iter()
                    .enumerate()
                    .all(|(index, _)| {
                        let deadline = ticks_per_stage.saturating_mul(index as u64 + 1);
                        episode
                            .completion_ticks
                            .get(index)
                            .is_some_and(|tick| *tick <= deadline)
                    })
        })
        .count()
}

struct DecoderSearchRequest<'a> {
    world: &'a WorldConfig,
    zero_extended: &'a OrganismGenome,
    module: &'a ModuleSpec,
    protection: &'a ProtectedResidual,
    ordinary_task: &'a EcologyProgram,
    ordinary_config: &'a PowerPlayConfig,
    search_panel: &'a DeclarationPanel,
    config: &'a PublicDecoderProbeConfig,
}

fn search_decoder(
    request: DecoderSearchRequest<'_>,
) -> Result<(
    Option<DecoderCandidate>,
    DecoderCandidate,
    Vec<DecoderGenerationEvidence>,
)> {
    let DecoderSearchRequest {
        world,
        zero_extended,
        module,
        protection,
        ordinary_task,
        ordinary_config,
        search_panel,
        config,
    } = request;
    let mut rng = ChaCha8Rng::seed_from_u64(
        config.source_powerplay.run_seed ^ DECODER_SEARCH_DOMAIN ^ 0x9e37_79b9_7f4a_7c15,
    );
    let mut genomes = Vec::with_capacity(config.decoder_population_size);
    genomes.push(zero_extended.clone());
    while genomes.len() < config.decoder_population_size {
        let mut genome = zero_extended.clone();
        randomize_module(&mut genome, module, &mut rng);
        protection.project(&mut genome);
        protection
            .verify(&genome)
            .map_err(|error| anyhow!("randomized decoder violates protection: {error}"))?;
        genomes.push(genome);
    }

    let mut history = Vec::new();
    let mut best_observed = None::<DecoderCandidate>;
    let elite_count = (config.decoder_population_size / 8).max(2);
    for generation in 0..config.decoder_generations {
        let mut population = Vec::with_capacity(genomes.len());
        for genome in genomes {
            let declarations = evaluate_declaration_panel(world, &genome, search_panel, false)?;
            let ordinary = evaluate_program(world, &genome, ordinary_task, ordinary_config, false)?;
            population.push(DecoderCandidate {
                genome,
                score: DecoderScore {
                    meaningful: declarations.meaningful_success_count,
                    blank: declarations.blank_success_count,
                    polarity: declarations.polarity_swap_success_count,
                    code_swap: declarations.valid_code_swap_success_count,
                    ordinary: all_deadline_success_count(
                        &ordinary,
                        ordinary_config.ticks_per_stage,
                    ),
                    energy_closed: declarations.all_energy_closed,
                },
            });
        }
        population.sort_by(decoder_candidate_ordering);
        let best = &population[0];
        let qualifying = population
            .iter()
            .filter(|candidate| candidate.score.passes(config))
            .collect::<Vec<_>>();
        let qualifying_candidate_fingerprints_in_selection_order = qualifying
            .iter()
            .map(|candidate| fingerprint(&candidate.genome))
            .collect::<Result<Vec<_>>>()?;
        if best_observed
            .as_ref()
            .is_none_or(|incumbent| decoder_candidate_ordering(best, incumbent) == Ordering::Less)
        {
            best_observed = Some(best.clone());
        }
        history.push(DecoderGenerationEvidence {
            generation,
            best_meaningful_success_count: best.score.meaningful,
            best_code_swap_success_count: best.score.code_swap,
            best_blank_success_count: best.score.blank,
            best_polarity_swap_success_count: best.score.polarity,
            best_ordinary_success_count: best.score.ordinary,
            best_passed_search_gate: !qualifying.is_empty(),
            qualifying_candidate_count: qualifying.len(),
            qualifying_candidate_fingerprints_in_selection_order,
        });
        if let Some(selected) = qualifying.first() {
            return Ok((Some((*selected).clone()), best.clone(), history));
        }

        let elites = population.into_iter().take(elite_count).collect::<Vec<_>>();
        genomes = elites
            .iter()
            .map(|candidate| candidate.genome.clone())
            .collect();
        let progress = generation as f32 / config.decoder_generations.max(1) as f32;
        let sigma = 1.25 * (1.0 - progress) + 0.15;
        while genomes.len() < config.decoder_population_size {
            let parent = &elites[rng.random_range(0..elites.len())];
            let mut child = parent.genome.clone();
            mutate_module(&mut child, module, sigma, &mut rng);
            protection.project(&mut child);
            protection
                .verify(&child)
                .map_err(|error| anyhow!("mutated decoder violates protection: {error}"))?;
            genomes.push(child);
        }
    }
    Ok((
        None,
        best_observed.ok_or_else(|| anyhow!("decoder search produced no candidate"))?,
        history,
    ))
}

fn decoder_candidate_ordering(left: &DecoderCandidate, right: &DecoderCandidate) -> Ordering {
    let left_floor = left
        .score
        .meaningful
        .min(left.score.code_swap)
        .min(left.score.ordinary);
    let right_floor = right
        .score
        .meaningful
        .min(right.score.code_swap)
        .min(right.score.ordinary);
    right_floor
        .cmp(&left_floor)
        .then_with(|| {
            (left.score.blank + left.score.polarity)
                .cmp(&(right.score.blank + right.score.polarity))
        })
        .then_with(|| right.score.meaningful.cmp(&left.score.meaningful))
        .then_with(|| right.score.code_swap.cmp(&left.score.code_swap))
        .then_with(|| right.score.ordinary.cmp(&left.score.ordinary))
}

fn decoder_module_fingerprint(genome: &OrganismGenome, module: &ModuleSpec) -> Result<String> {
    let nodes = genome
        .brain
        .hidden_nodes
        .iter()
        .filter(|node| module.hidden_node_ids.contains(&node.id))
        .copied()
        .collect::<Vec<_>>();
    let edges = genome
        .brain
        .edges
        .iter()
        .filter(|edge| module.connection_innovations.contains(&edge.innovation))
        .copied()
        .collect::<Vec<_>>();
    fingerprint(&(nodes, edges))
}

fn reuse_decoder_module_in_descendant(
    decoder: &OrganismGenome,
    module: &ModuleSpec,
    target: &OrganismGenome,
) -> Result<OrganismGenome> {
    let mut transferred = target.clone();
    for node in decoder
        .brain
        .hidden_nodes
        .iter()
        .filter(|node| module.hidden_node_ids.contains(&node.id))
    {
        if transferred
            .brain
            .hidden_nodes
            .iter()
            .any(|existing| existing.id == node.id)
        {
            bail!("decoder reuse hidden-node identity collides with descendant checkpoint");
        }
        transferred.brain.hidden_nodes.push(*node);
    }
    for edge in decoder
        .brain
        .edges
        .iter()
        .filter(|edge| module.connection_innovations.contains(&edge.innovation))
    {
        if transferred.brain.edges.iter().any(|existing| {
            existing.innovation == edge.innovation
                || (existing.pre_node_id == edge.pre_node_id
                    && existing.post_node_id == edge.post_node_id)
        }) {
            bail!("decoder reuse connection identity collides with descendant checkpoint");
        }
        transferred.brain.edges.push(*edge);
    }
    transferred
        .brain
        .hidden_nodes
        .sort_unstable_by_key(|node| node.id);
    transferred
        .brain
        .edges
        .sort_unstable_by_key(|edge| edge.innovation);
    Ok(transferred)
}

fn evaluate_descendant_checkpoint_reuse(
    world: &WorldConfig,
    decoder: &OrganismGenome,
    module: &ModuleSpec,
    target_checkpoint: &OrganismGenome,
    target_task: &EcologyProgram,
    reuse_panel: &DeclarationPanel,
    config: &PublicDecoderProbeConfig,
) -> Result<PublicDecoderReuseEvidence> {
    let transferred = reuse_decoder_module_in_descendant(decoder, module, target_checkpoint)?;
    let before_fingerprint = decoder_module_fingerprint(decoder, module)?;
    let after_fingerprint = decoder_module_fingerprint(&transferred, module)?;
    let module_reused_without_change = before_fingerprint == after_fingerprint;
    let knockout = knockout_module(&transferred, module);
    let knockout_restores_target_checkpoint_exactly = knockout == *target_checkpoint;
    let reuse_declaration = evaluate_declaration_panel(world, &transferred, reuse_panel, true)?;
    let source_declaration =
        evaluate_declaration_panel(world, target_checkpoint, reuse_panel, false)?;
    let transferred_ordinary = evaluate_program(
        world,
        &transferred,
        target_task,
        &config.source_powerplay,
        true,
    )?;
    let source_ordinary = evaluate_program(
        world,
        target_checkpoint,
        target_task,
        &config.source_powerplay,
        true,
    )?;
    let knockout_ordinary = evaluate_program(
        world,
        &knockout,
        target_task,
        &config.source_powerplay,
        true,
    )?;
    let knockout_reproduces_target_ordinary_behavior_exactly = knockout_ordinary == source_ordinary;
    let passed = reuse_declaration.passes(config)
        && source_declaration.meaningful_success_count <= config.control_maximum
        && all_deadline_success_count(
            &transferred_ordinary,
            config.source_powerplay.ticks_per_stage,
        ) >= config.ordinary_retention_minimum
        && module_reused_without_change
        && knockout_restores_target_checkpoint_exactly
        && knockout_reproduces_target_ordinary_behavior_exactly;
    Ok(PublicDecoderReuseEvidence {
        status: if passed {
            "passed".to_string()
        } else {
            "attempted_and_failed".to_string()
        },
        reused_genome_fingerprint: Some(fingerprint(&transferred)?),
        reused_genome: Some(transferred),
        exact_decoder_module_fingerprint_before: Some(before_fingerprint),
        exact_decoder_module_fingerprint_after: Some(after_fingerprint),
        module_reused_without_change,
        reuse_declaration_evaluation: Some(reuse_declaration),
        source_target_declaration_success_count: source_declaration.meaningful_success_count,
        reused_ordinary_evaluation: Some(transferred_ordinary),
        source_target_ordinary_evaluation: Some(source_ordinary),
        knockout_target_ordinary_evaluation: Some(knockout_ordinary),
        knockout_restores_target_checkpoint_exactly,
        knockout_reproduces_target_ordinary_behavior_exactly,
        passed,
    })
}
