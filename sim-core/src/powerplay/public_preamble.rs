//! Executable falsification probe for the imported public-preamble premise.
//!
//! PowerPlay solvers are selected on ordinary task episodes. This evaluator-
//! owned probe asks a narrower question before that premise is imported into a
//! future algorithm: does an accepted solver make task-specific use of a
//! public, physical FoodRay preamble? It compares a meaningful program encoding
//! against all-blank and left/right-permuted controls on disjoint contexts.
//! The probe is diagnostic only and is never evidence of open-endedness.

use super::{
    configure_task_world, execute_program_on_sim, mix64, prepare_episode_state,
    resolve_stage_context, run_powerplay, EcologyProgram, EpisodeEvidence, PowerPlayConfig,
    ResourceMotion,
};
use crate::{
    grid::{hex_neighbor, rotate_by_steps},
    progressive::exact_fingerprint,
    Simulation,
};
use anyhow::{anyhow, bail, Result};
use serde::{Deserialize, Serialize};
use sim_config::WorldConfig;
use sim_types::{
    food_visual, ActionType, EnergyLedgerRow, FacingDirection, FoodId, FoodKind, FoodState,
    Occupant, OrganismGenome, OrganismState,
};
use std::collections::BTreeSet;

const PUBLIC_PREAMBLE_RESULT_SCHEMA_VERSION: u32 = 1;
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
/// preamble falsification on independent contexts.
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
            "at least one exact accepted solver/task pair failed the >=14/16 meaningful and <=2/16 blank/permuted matched-arm gate".to_string(),
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

    let branch_transfer_status = match verdict {
        ImportedPreamblePremiseVerdict::Falsified => {
            "not_attempted_because_public_semantics_failed"
        }
        ImportedPreamblePremiseVerdict::SurvivedStrictGate
        | ImportedPreamblePremiseVerdict::NotEvaluable => {
            "not_attempted_by_this_imported-premise_probe"
        }
    }
    .to_string();

    Ok(PublicPreambleProbeResult {
        result_schema_version: PUBLIC_PREAMBLE_RESULT_SCHEMA_VERSION,
        algorithm: "powerplay_public_preamble_import_falsification_v1".to_string(),
        claim_scope: "evaluator-owned falsification of a public-preamble capacity premise; not selection, admission, novelty, capability, or open-endedness evidence".to_string(),
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
    let full_task_success = task_prefix_successes.last().copied().unwrap_or(false);
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
    let mut bits = Vec::with_capacity(FIXED_PREAMBLE_TICKS);
    bits.extend(MAGIC_BITS);
    push_bits(&mut bits, u64::from(PROTOCOL_VERSION), 2);
    push_bits(&mut bits, resolved_stages.len() as u64, 2);
    for slot in 0..MAX_ENCODED_STAGES {
        let Some(stage) = resolved_stages.get(slot).copied() else {
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
    Ok((
        EcologyProgram {
            stages: resolved_stages,
        },
        bits,
    ))
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
        .max(row.total_residual.abs())
        .max(row.transfer_residual.abs())
}

fn fingerprint<T: Serialize>(value: &T) -> Result<String> {
    exact_fingerprint(value).map_err(|error| anyhow!(error.to_string()))
}
