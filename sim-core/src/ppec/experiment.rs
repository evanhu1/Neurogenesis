//! Evaluator-owned PPEC mechanism-engagement experiment.
//!
//! This is deliberately a Stage-0 falsifier, not an evolutionary result. It
//! constructs exact two-producer scenes, exercises the ordinary plant-consume
//! funding path, then runs matched public-interaction controls against the
//! persisted cache state. No score from this module enters selection.

use super::{
    evaluate_public_protocol, mix64, permuted_public_challenge, permuted_public_program,
    public_protocol_fingerprint,
};
use crate::{
    genome::generate_seed_genome, grid::hex_neighbor, progressive::exact_fingerprint, Simulation,
};
use anyhow::{anyhow, bail, Result};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use sim_types::{
    food_visual, seed_hidden_gene_node_id, ActionType, ArtifactCacheState, ArtifactId,
    ArtifactInteractionEvent, ArtifactInteractionOutcome, EnergyLedgerRow, FacingDirection, FoodId,
    FoodKind, FoodState, HiddenNodeGene, Occupant, OrganismGenome, OrganismId, WorldConfig,
};

const EXPERIMENT_DOMAIN: u64 = 0x5050_4543_4558_5030;
const RANDOM_RESPONSE_DOMAIN: u64 = 0x5050_4543_5241_4e44;
const WORLD_WIDTH: u32 = 15;
const FOOD_ENERGY: f32 = 20.0;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PpecMechanismExperimentConfig {
    pub run_seeds: Vec<u64>,
    pub contexts_per_seed: u32,
    pub persistence_ticks: u32,
    pub cache_energy_fraction: f32,
    pub protocol_interaction_energy_cost: f32,
}

impl Default for PpecMechanismExperimentConfig {
    fn default() -> Self {
        Self {
            run_seeds: vec![7, 42, 123],
            contexts_per_seed: 8,
            persistence_ticks: 3,
            cache_energy_fraction: 0.35,
            protocol_interaction_energy_cost: 0.25,
        }
    }
}

impl PpecMechanismExperimentConfig {
    fn validate(&self) -> Result<()> {
        if self.run_seeds.is_empty() {
            bail!("PPEC mechanism experiment needs at least one run seed");
        }
        if self.contexts_per_seed == 0 {
            bail!("PPEC mechanism experiment needs at least one context per seed");
        }
        if self.persistence_ticks == 0 {
            bail!("PPEC persistence horizon must be at least one tick");
        }
        if !self.cache_energy_fraction.is_finite()
            || self.cache_energy_fraction <= 0.0
            || self.cache_energy_fraction >= 1.0
        {
            bail!("PPEC cache fraction must be finite and strictly between zero and one");
        }
        if !self.protocol_interaction_energy_cost.is_finite()
            || self.protocol_interaction_energy_cost <= 0.0
        {
            bail!("PPEC mechanism probe interaction cost must be finite and positive");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum PpecControlArm {
    OwnProtocol,
    ForeignProtocol,
    NoPayoff,
    CodePermutation,
    ChallengePermutation,
    ArtifactKnockout,
    #[serde(rename = "constant_response_0")]
    ConstantResponse0,
    #[serde(rename = "constant_response_1")]
    ConstantResponse1,
    #[serde(rename = "constant_response_2")]
    ConstantResponse2,
    #[serde(rename = "constant_response_3")]
    ConstantResponse3,
    RandomResponse,
}

impl PpecControlArm {
    const ALL: [Self; 11] = [
        Self::OwnProtocol,
        Self::ForeignProtocol,
        Self::NoPayoff,
        Self::CodePermutation,
        Self::ChallengePermutation,
        Self::ArtifactKnockout,
        Self::ConstantResponse0,
        Self::ConstantResponse1,
        Self::ConstantResponse2,
        Self::ConstantResponse3,
        Self::RandomResponse,
    ];
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PpecProductionEvidence {
    pub plants_consumed: u64,
    pub cache_count: usize,
    pub cache_energy: f64,
    pub expected_cache_energy: f64,
    pub cache_cells_absent_from_occupancy: bool,
    pub distinct_protocol_fingerprints: usize,
    pub distinct_program_lengths: usize,
    pub challenge_lengths: Vec<usize>,
    pub same_tick_future_requests_rejected: bool,
    pub same_tick_events: Vec<ArtifactInteractionEvent>,
    pub ledger: EnergyLedgerRow,
    pub energy_closed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PpecPersistenceEvidence {
    pub waited_ticks: u32,
    pub caches_unchanged: bool,
    pub queued_request_count_before_save: usize,
    pub queued_organism_id: OrganismId,
    pub queued_artifact_id: ArtifactId,
    pub queued_response_bits: u8,
    pub queued_events_after_advance: Vec<ArtifactInteractionEvent>,
    pub post_advance_energy_ledger: EnergyLedgerRow,
    pub exact_save_load_roundtrip: bool,
    pub post_load_advance_exact: bool,
    pub world_fingerprint_before_save: String,
    pub world_fingerprint_after_load: String,
    pub world_fingerprint_after_direct_advance: String,
    pub world_fingerprint_after_loaded_advance: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PpecArmAssignment {
    pub organism_id: OrganismId,
    pub artifact_id: ArtifactId,
    pub artifact_creator_id: OrganismId,
    pub response_bits: u8,
    pub expected_response_bits: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PpecArmEvidence {
    pub arm: PpecControlArm,
    pub assignments: Vec<PpecArmAssignment>,
    pub events: Vec<ArtifactInteractionEvent>,
    /// Own-arm adversarial sequence: a lower-ID nonadjacent request is rejected
    /// without claiming the slot, then the higher-ID adjacent request releases.
    pub lower_id_nonadjacent_then_valid_succeeds: bool,
    pub accepted_count: usize,
    pub released_energy: f64,
    pub release_loss: f64,
    pub knocked_out_energy: f64,
    pub remaining_cache_count: usize,
    pub remaining_cache_energy: f64,
    pub ledger: EnergyLedgerRow,
    pub energy_closed: bool,
    pub final_world_fingerprint: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PpecAffordabilityEvidence {
    pub interaction_cost: f32,
    pub insufficient_actor_energy: f32,
    pub artifact_id: ArtifactId,
    pub lower_organism_id: OrganismId,
    pub higher_organism_id: OrganismId,
    pub response_bits: u8,
    pub events: Vec<ArtifactInteractionEvent>,
    pub insufficient_nonclaiming_then_valid_succeeds: bool,
    pub ledger: EnergyLedgerRow,
    pub energy_closed: bool,
    pub final_world_fingerprint: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PpecContentionCaseEvidence {
    pub winner_response_correct: bool,
    pub artifact_id: ArtifactId,
    pub lower_organism_id: OrganismId,
    pub higher_organism_id: OrganismId,
    pub cache_energy_before: f32,
    pub cache_challenge_before: Vec<u8>,
    pub cache_challenge_after: Option<Vec<u8>>,
    pub lower_organism_energy_before: f32,
    pub lower_organism_energy_after: f32,
    pub higher_organism_energy_before: f32,
    pub higher_organism_energy_after: f32,
    pub events: Vec<ArtifactInteractionEvent>,
    pub cache_present_after: bool,
    pub exact_gate_passed: bool,
    pub ledger: EnergyLedgerRow,
    pub energy_closed: bool,
    pub final_world_fingerprint: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PpecContentionEvidence {
    pub correct_lower_id_winner: PpecContentionCaseEvidence,
    pub wrong_lower_id_winner: PpecContentionCaseEvidence,
    pub all_passed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PpecContextEvidence {
    pub run_seed: u64,
    pub context_index: u32,
    pub context_seed: u64,
    pub production: PpecProductionEvidence,
    pub persistence: PpecPersistenceEvidence,
    pub produced_caches: Vec<ArtifactCacheState>,
    pub affordability: PpecAffordabilityEvidence,
    pub contention: PpecContentionEvidence,
    pub arms: Vec<PpecArmEvidence>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PpecArmAggregate {
    pub arm: PpecControlArm,
    pub trials: usize,
    pub accepted: usize,
    pub released_energy: f64,
    pub release_loss: f64,
    pub all_energy_closed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PpecSeedPublicSemanticsEvidence {
    pub run_seed: u64,
    pub total_protocol_challenges: usize,
    pub expected_response_counts: [usize; 4],
    pub minimum_required_per_response: usize,
    pub strong_control_max_acceptances: usize,
    pub constant_response_acceptances: [usize; 4],
    pub code_permutation_acceptances: usize,
    pub challenge_permutation_acceptances: usize,
    pub random_response_acceptances: usize,
    pub all_passed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PpecPublicSemanticsEvidence {
    pub total_protocol_challenges: usize,
    pub expected_response_counts: [usize; 4],
    pub minimum_required_per_response: usize,
    pub distinct_response_counts_by_protocol: std::collections::BTreeMap<String, usize>,
    pub minimum_distinct_responses_per_protocol: usize,
    pub strong_control_max_acceptances: usize,
    pub constant_response_acceptances: [usize; 4],
    pub code_permutation_acceptances: usize,
    pub challenge_permutation_acceptances: usize,
    pub random_response_acceptances: usize,
    pub per_run_seed: Vec<PpecSeedPublicSemanticsEvidence>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PpecMechanismGates {
    pub production_split_closed: bool,
    pub persistent_nonblocking_state: bool,
    pub exact_save_load_replay: bool,
    pub variable_public_protocols: bool,
    pub public_output_balance: bool,
    pub constant_controls_match_expected_histogram: bool,
    pub challenge_varies_response_within_each_protocol: bool,
    pub every_seed_public_semantics_passes: bool,
    pub own_use_positive: bool,
    pub foreign_use_positive: bool,
    pub no_payoff_is_causally_separated: bool,
    pub code_permutation_strongly_reduces_acceptance: bool,
    pub challenge_permutation_strongly_reduces_acceptance: bool,
    pub artifact_knockout_removes_payoff: bool,
    pub all_constant_responses_strongly_reduced: bool,
    pub random_response_strongly_reduced: bool,
    pub invalid_lower_id_does_not_claim_slot: bool,
    pub insufficient_energy_does_not_claim_slot: bool,
    pub valid_contention_is_canonical_and_accounted: bool,
    pub all_energy_ledgers_close: bool,
    pub all_passed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PpecMechanismExperimentResult {
    pub claim_scope: String,
    pub evaluator_owned: bool,
    pub open_endedness_demonstrated: bool,
    pub limitations: Vec<String>,
    pub config: PpecMechanismExperimentConfig,
    pub effective_world: WorldConfig,
    pub effective_world_fingerprint: String,
    pub contexts: Vec<PpecContextEvidence>,
    pub aggregates: Vec<PpecArmAggregate>,
    pub public_semantics: PpecPublicSemanticsEvidence,
    pub gates: PpecMechanismGates,
    pub result_fingerprint: String,
}

/// Execute the bounded mechanism-engagement experiment. The returned artifact
/// contains exact public programs, challenges, responses, events, and ledger
/// rows so aggregate gates can be audited without trusting the summary.
pub fn run_ppec_mechanism_experiment(
    base_world: WorldConfig,
    config: PpecMechanismExperimentConfig,
) -> Result<PpecMechanismExperimentResult> {
    config.validate()?;
    let effective_world = mechanism_world(base_world, &config);
    let effective_world_fingerprint = fingerprint(&effective_world)?;
    let mut contexts = Vec::with_capacity(
        config
            .run_seeds
            .len()
            .saturating_mul(config.contexts_per_seed as usize),
    );
    for &run_seed in &config.run_seeds {
        for context_index in 0..config.contexts_per_seed {
            let context_seed = mix64(
                run_seed
                    ^ EXPERIMENT_DOMAIN
                    ^ u64::from(context_index).wrapping_mul(0x9e37_79b9_7f4a_7c15),
            );
            contexts.push(run_context(
                &effective_world,
                &config,
                run_seed,
                context_index,
                context_seed,
            )?);
        }
    }

    let aggregates = PpecControlArm::ALL
        .into_iter()
        .map(|arm| aggregate_arm(&contexts, arm))
        .collect::<Result<Vec<_>>>()?;
    let public_semantics = build_public_semantics_evidence(&contexts, &aggregates)?;
    let gates = build_gates(&contexts, &aggregates, &public_semantics)?;
    let mut result = PpecMechanismExperimentResult {
        claim_scope: "evaluator-owned Stage-0 PPEC mechanism engagement; not open-endedness evidence"
            .to_owned(),
        evaluator_owned: true,
        open_endedness_demonstrated: false,
        limitations: vec![
            "Responses are supplied by the evaluator through the public interaction API; no neural response head has evolved.".to_owned(),
            "The public protocol is a genome-derived Stage-0 acyclic NAND-byte interpretation, not the intended evolvable ProtocolGenome; it has a u16 input-arity bound and byte growth, canonicalization, and dead/unreachable-gate resistance are not established.".to_owned(),
            "Caches can stack and have no expiry, decay, capacity limit, or garbage collection, so Stage 0 is not an unbounded-run ecology.".to_owned(),
            "Every cache clones the full genome-derived opcode vector and evaluation allocates and replays work linear in program bytes; combined with stacking and no collection, enabled long runs can grow memory/work for the wrong reason. Stage 1 needs canonical protocol interning/reference counting plus a bounded lifecycle and energy-priced execution.".to_owned(),
            "Valid same-cache contention is resolved by canonical organism-ID order; Stage 0 does not establish ID-order robustness, so future evolutionary panels must rotate founder/ID assignments symmetrically.".to_owned(),
            "Artifacts are public through the Simulation API and CLI evidence but are not yet exposed through the world wire snapshot, organism receptors, or a neural response head.".to_owned(),
            "The one-shot two-bit response gives blind guessing 1/4 success; at the probe's 7-energy cache and 0.25 cost its expected payoff is positive, so this is not yet a protective adversarial ecology. Stage 1 needs a multi-round proof or cost/reward contract with negative random-policy net payoff and a hard random-payoff gate.".to_owned(),
            "Own/foreign use and causal controls establish mechanism engagement only; they do not establish adaptive novelty, capability growth, or tail open-endedness.".to_owned(),
        ],
        config,
        effective_world,
        effective_world_fingerprint,
        contexts,
        aggregates,
        public_semantics,
        gates,
        result_fingerprint: String::new(),
    };
    result.result_fingerprint = fingerprint(&result)?;
    Ok(result)
}

fn run_context(
    world: &WorldConfig,
    config: &PpecMechanismExperimentConfig,
    run_seed: u64,
    context_index: u32,
    context_seed: u64,
) -> Result<PpecContextEvidence> {
    let genomes = producer_genomes(world, context_seed);
    let mut sim = Simulation::new_with_champion_pool(world.clone(), context_seed, genomes)
        .map_err(|error| anyhow!(error.to_string()))?;
    prepare_production_scene(&mut sim, context_seed)?;
    let production_delta = sim.tick();
    if sim.artifact_caches.len() != 2 {
        bail!(
            "PPEC production scene created {} caches instead of two",
            sim.artifact_caches.len()
        );
    }
    let production_ledger = production_delta.metrics.energy_ledger_last_turn;
    let expected_cache_energy =
        f64::from(FOOD_ENERGY) * f64::from(config.cache_energy_fraction) * 2.0;
    let cache_energy = artifact_energy(&sim);
    let production = PpecProductionEvidence {
        plants_consumed: production_delta.metrics.plant_consumptions_last_turn,
        cache_count: sim.artifact_caches.len(),
        cache_energy,
        expected_cache_energy,
        cache_cells_absent_from_occupancy: sim
            .artifact_caches
            .iter()
            .all(|cache| sim.occupancy[sim.cell_index(cache.q, cache.r)].is_none()),
        distinct_protocol_fingerprints: distinct_count(
            sim.artifact_caches
                .iter()
                .map(|cache| cache.owner_protocol_fingerprint.as_str()),
        ),
        distinct_program_lengths: distinct_count(
            sim.artifact_caches
                .iter()
                .map(|cache| cache.public_program.opcodes.len()),
        ),
        challenge_lengths: sim
            .artifact_caches
            .iter()
            .map(|cache| cache.challenge_bits.len())
            .collect(),
        same_tick_future_requests_rejected: sim.artifact_events_last_turn.len() == 2
            && sim.artifact_events_last_turn.iter().all(|event| {
                event.outcome == ArtifactInteractionOutcome::CreatedThisTick
                    && event.released_energy == 0.0
                    && event.cache_energy_after == event.cache_energy_before
            }),
        same_tick_events: sim.artifact_events_last_turn.clone(),
        ledger: production_ledger,
        energy_closed: ledger_closes(production_ledger),
    };
    let produced_caches = sim.artifact_caches.clone();

    let persisted_state = sim.artifact_caches.clone();
    for _ in 0..config.persistence_ticks {
        let delta = sim.tick();
        if !ledger_closes(delta.metrics.energy_ledger_last_turn) {
            bail!("PPEC persistence tick energy ledger did not close");
        }
    }
    let caches_unchanged = sim.artifact_caches == persisted_state;

    // Persist a genuinely pending, eligible interaction rather than proving
    // replay only for an empty queue. The untouched `sim` remains the common
    // ancestor for all matched intervention arms below.
    let mut queued = sim.clone();
    let queued_caches = queued.artifact_caches.clone();
    position_actors(&mut queued, &queued_caches, false)?;
    let queued_artifact = queued_caches[0].clone();
    let queued_organism_id = queued.organisms[0].id;
    let queued_response_bits = evaluate_public_protocol(
        &queued_artifact.public_program,
        &queued_artifact.challenge_bits,
    )?;
    queued.queue_artifact_interaction(queued_organism_id, queued_artifact.id, queued_response_bits);
    let queued_request_count_before_save = queued.pending_artifact_interactions.len();
    let world_fingerprint_before_save = fingerprint(&queued)?;
    let mut saved = Vec::new();
    queued
        .save(&mut saved)
        .map_err(|error| anyhow!(error.to_string()))?;
    let mut loaded =
        Simulation::load(saved.as_slice()).map_err(|error| anyhow!(error.to_string()))?;
    let world_fingerprint_after_load = fingerprint(&loaded)?;
    let mut resaved = Vec::new();
    loaded
        .save(&mut resaved)
        .map_err(|error| anyhow!(error.to_string()))?;
    let exact_save_load_roundtrip = saved == resaved
        && queued.pending_artifact_interactions == loaded.pending_artifact_interactions
        && queued.artifact_caches == loaded.artifact_caches
        && world_fingerprint_before_save == world_fingerprint_after_load;

    let mut direct = queued;
    let direct_delta = direct.tick();
    let loaded_delta = loaded.tick();
    let world_fingerprint_after_direct_advance = fingerprint(&direct)?;
    let world_fingerprint_after_loaded_advance = fingerprint(&loaded)?;
    let mut direct_after_advance = Vec::new();
    direct
        .save(&mut direct_after_advance)
        .map_err(|error| anyhow!(error.to_string()))?;
    let mut loaded_after_advance = Vec::new();
    loaded
        .save(&mut loaded_after_advance)
        .map_err(|error| anyhow!(error.to_string()))?;
    let queued_events_after_advance = direct.artifact_events_last_turn.clone();
    let post_advance_energy_ledger = direct_delta.metrics.energy_ledger_last_turn;
    let post_load_advance_exact = queued_request_count_before_save == 1
        && direct_delta == loaded_delta
        && direct.artifact_events_last_turn == loaded.artifact_events_last_turn
        && direct.metrics.energy_ledger_last_turn == loaded.metrics.energy_ledger_last_turn
        && direct.pending_artifact_interactions.is_empty()
        && loaded.pending_artifact_interactions.is_empty()
        && direct_after_advance == loaded_after_advance
        && world_fingerprint_after_direct_advance == world_fingerprint_after_loaded_advance
        && queued_events_after_advance.len() == 1
        && queued_events_after_advance[0].organism_id == queued_organism_id
        && queued_events_after_advance[0].artifact_id == queued_artifact.id
        && queued_events_after_advance[0].response_bits == queued_response_bits
        && queued_events_after_advance[0].outcome == ArtifactInteractionOutcome::Released
        && ledger_closes(post_advance_energy_ledger);

    let affordability = run_affordability_probe(sim.clone())?;
    let contention = run_contention_probe(sim.clone())?;
    let own_from_original = run_arm(sim.clone(), PpecControlArm::OwnProtocol, context_seed)?;
    let mut arms = Vec::with_capacity(PpecControlArm::ALL.len());
    arms.push(own_from_original);
    for arm in PpecControlArm::ALL.into_iter().skip(1) {
        arms.push(run_arm(sim.clone(), arm, context_seed)?);
    }

    Ok(PpecContextEvidence {
        run_seed,
        context_index,
        context_seed,
        production,
        persistence: PpecPersistenceEvidence {
            waited_ticks: config.persistence_ticks,
            caches_unchanged,
            queued_request_count_before_save,
            queued_organism_id,
            queued_artifact_id: queued_artifact.id,
            queued_response_bits,
            queued_events_after_advance,
            post_advance_energy_ledger,
            exact_save_load_roundtrip,
            post_load_advance_exact,
            world_fingerprint_before_save,
            world_fingerprint_after_load,
            world_fingerprint_after_direct_advance,
            world_fingerprint_after_loaded_advance,
        },
        produced_caches,
        affordability,
        contention,
        arms,
    })
}

fn mechanism_world(mut world: WorldConfig, config: &PpecMechanismExperimentConfig) -> WorldConfig {
    world.world_width = WORLD_WIDTH;
    world.num_organisms = 2;
    world.food_energy = FOOD_ENERGY;
    world.passive_metabolism_cost_per_unit = 0.0;
    world.body_mass_metabolic_cost_coeff = 0.0;
    world.move_action_energy_cost = 0.0;
    world.action_temperature = 0.01;
    world.intent_parallel_threads = 1;
    world.food_regrowth_interval = 100_000;
    world.food_regrowth_jitter = 0;
    world.food_tile_fraction = 0.0;
    world.terrain_threshold = 1.0;
    world.runtime_plasticity_enabled = false;
    world.leaky_neurons_enabled = false;
    world.predation_enabled = false;
    world.force_random_actions = false;
    world.protocol_cache_enabled = true;
    world.cache_energy_fraction = config.cache_energy_fraction;
    world.cache_release_efficiency = 1.0;
    world.protocol_interaction_energy_cost = config.protocol_interaction_energy_cost;
    world.seed_genome_config.num_neurons = 0;
    world.seed_genome_config.num_synapses = 0;
    world.seed_genome_config.hebb_eta_gain = 0.0;
    world
}

fn producer_genomes(world: &WorldConfig, seed: u64) -> Vec<OrganismGenome> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ EXPERIMENT_DOMAIN);
    let mut first = generate_seed_genome(&world.seed_genome_config, false, &mut rng);
    set_eat_policy(&mut first);
    let mut second = first.clone();
    second.brain.hidden_nodes.push(HiddenNodeGene {
        id: seed_hidden_gene_node_id(10_000),
        bias: 0.375,
        log_time_constant: -0.25,
    });
    set_eat_policy(&mut second);
    vec![first, second]
}

fn set_eat_policy(genome: &mut OrganismGenome) {
    genome.brain.action_biases.fill(-20.0);
    let eat_index = ActionType::ALL
        .iter()
        .position(|action| *action == ActionType::Eat)
        .expect("Eat is a stable active action");
    genome.brain.action_biases[eat_index] = 20.0;
}

fn prepare_production_scene(sim: &mut Simulation, context_seed: u64) -> Result<()> {
    if sim.organisms.len() != 2 || sim.terrain_map.iter().any(|blocked| *blocked) {
        bail!("PPEC experiment requires two founders and an obstacle-free arena");
    }
    sim.foods.clear();
    sim.food_tiles.fill(false);
    sim.food_regrowth_due_turn.fill(u64::MAX);
    sim.food_regrowth_schedule.clear();
    sim.artifact_caches.clear();
    sim.pending_artifact_interactions.clear();
    sim.artifact_events_last_turn.clear();
    sim.occupancy.fill(None);
    let offset = (mix64(context_seed) % 3) as i32;
    let anchors = [(2 + offset, 3), (9 + offset, 10)];
    for (organism, &(q, r)) in sim.organisms.iter_mut().zip(&anchors) {
        organism.q = q;
        organism.r = r;
        organism.facing = FacingDirection::East;
        organism.energy_at_last_sensing = organism.energy;
        organism.last_action_taken = ActionType::Idle;
        let idx = r as usize * sim.config.world_width as usize + q as usize;
        sim.occupancy[idx] = Some(Occupant::Organism(organism.id));
    }
    let targets = sim
        .organisms
        .iter()
        .map(|organism| {
            (
                organism.id,
                hex_neighbor(
                    (organism.q, organism.r),
                    organism.facing,
                    sim.config.world_width as i32,
                ),
            )
        })
        .collect::<Vec<_>>();
    for (_, (q, r)) in targets {
        let idx = sim.cell_index(q, r);
        if sim.occupancy[idx].is_some() {
            bail!("PPEC production target is occupied");
        }
        let id = FoodId(sim.next_food_id);
        sim.next_food_id = sim.next_food_id.saturating_add(1);
        sim.occupancy[idx] = Some(Occupant::Food(id));
        sim.foods.push(FoodState {
            id,
            q,
            r,
            energy: FOOD_ENERGY,
            kind: FoodKind::Plant,
            visual: food_visual(FoodKind::Plant),
        });
    }
    // Deliberately predict the IDs that the pending plant-consumption commit
    // will mint. The tick-start frontier must reject these requests even if a
    // supplied response happens to match the later public challenge.
    let organism_ids = sim
        .organisms
        .iter()
        .map(|organism| organism.id)
        .collect::<Vec<_>>();
    for (index, organism_id) in organism_ids.into_iter().enumerate() {
        sim.queue_artifact_interaction(
            organism_id,
            ArtifactId(sim.next_artifact_id.saturating_add(index as u64)),
            0,
        );
    }
    sim.debug_assert_consistent_state();
    Ok(())
}

fn run_arm(mut sim: Simulation, arm: PpecControlArm, context_seed: u64) -> Result<PpecArmEvidence> {
    let caches = sim.artifact_caches.clone();
    if caches.len() != 2 || sim.organisms.len() != 2 {
        bail!("PPEC matched arms require two organisms and two caches");
    }
    let foreign = arm == PpecControlArm::ForeignProtocol;
    position_actors(&mut sim, &caches, foreign)?;
    if arm == PpecControlArm::NoPayoff {
        sim.config.cache_release_efficiency = 0.0;
    }
    let knocked_out_energy = if arm == PpecControlArm::ArtifactKnockout {
        let energy = artifact_energy(&sim);
        sim.artifact_caches.clear();
        energy
    } else {
        0.0
    };

    let actor_ids = sim
        .organisms
        .iter()
        .map(|organism| organism.id)
        .collect::<Vec<_>>();
    let mut assignments = Vec::with_capacity(caches.len() + 1);
    if arm == PpecControlArm::OwnProtocol {
        let cache = &caches[1];
        let expected_response_bits =
            evaluate_public_protocol(&cache.public_program, &cache.challenge_bits)?;
        // Organism 0 has the lower canonical ID but is positioned at cache 0,
        // far from cache 1. It must not consume cache 1's interaction slot
        // before adjacent organism 1's valid request.
        sim.queue_artifact_interaction(actor_ids[0], cache.id, expected_response_bits);
        assignments.push(PpecArmAssignment {
            organism_id: actor_ids[0],
            artifact_id: cache.id,
            artifact_creator_id: cache.creator_id,
            response_bits: expected_response_bits,
            expected_response_bits,
        });
    }
    for (index, cache) in caches.iter().enumerate() {
        let actor_index = if foreign { 1 - index } else { index };
        let organism_id = actor_ids[actor_index];
        let expected_response_bits =
            evaluate_public_protocol(&cache.public_program, &cache.challenge_bits)?;
        let response_bits = match arm {
            PpecControlArm::OwnProtocol
            | PpecControlArm::ForeignProtocol
            | PpecControlArm::NoPayoff
            | PpecControlArm::ArtifactKnockout => expected_response_bits,
            PpecControlArm::CodePermutation => evaluate_public_protocol(
                &permuted_public_program(&cache.public_program),
                &cache.challenge_bits,
            )?,
            PpecControlArm::ChallengePermutation => evaluate_public_protocol(
                &cache.public_program,
                &permuted_public_challenge(&cache.challenge_bits),
            )?,
            PpecControlArm::ConstantResponse0 => 0,
            PpecControlArm::ConstantResponse1 => 1,
            PpecControlArm::ConstantResponse2 => 2,
            PpecControlArm::ConstantResponse3 => 3,
            PpecControlArm::RandomResponse => {
                (mix64(
                    context_seed
                        ^ RANDOM_RESPONSE_DOMAIN
                        ^ cache.id.0.rotate_left(19)
                        ^ u64::from(index as u32),
                ) & 0b11) as u8
            }
        };
        sim.queue_artifact_interaction(organism_id, cache.id, response_bits);
        assignments.push(PpecArmAssignment {
            organism_id,
            artifact_id: cache.id,
            artifact_creator_id: cache.creator_id,
            response_bits,
            expected_response_bits,
        });
    }
    let delta = sim.tick();
    let events = sim.artifact_events_last_turn.clone();
    let accepted_count = events
        .iter()
        .filter(|event| {
            matches!(
                event.outcome,
                ArtifactInteractionOutcome::Released | ArtifactInteractionOutcome::AcceptedNoPayoff
            )
        })
        .count();
    let released_energy = events
        .iter()
        .map(|event| f64::from(event.released_energy))
        .sum();
    let release_loss = events
        .iter()
        .map(|event| f64::from(event.release_loss))
        .sum();
    let ledger = delta.metrics.energy_ledger_last_turn;
    let lower_id_nonadjacent_then_valid_succeeds = if arm == PpecControlArm::OwnProtocol {
        let probe_events = events
            .iter()
            .filter(|event| event.artifact_id == caches[1].id)
            .collect::<Vec<_>>();
        probe_events.len() == 2
            && probe_events[0].organism_id == actor_ids[0]
            && probe_events[0].outcome == ArtifactInteractionOutcome::NotAdjacent
            && probe_events[1].organism_id == actor_ids[1]
            && probe_events[1].outcome == ArtifactInteractionOutcome::Released
    } else {
        false
    };
    Ok(PpecArmEvidence {
        arm,
        assignments,
        events,
        lower_id_nonadjacent_then_valid_succeeds,
        accepted_count,
        released_energy,
        release_loss,
        knocked_out_energy,
        remaining_cache_count: sim.artifact_caches.len(),
        remaining_cache_energy: artifact_energy(&sim),
        ledger,
        energy_closed: ledger_closes(ledger),
        final_world_fingerprint: fingerprint(&sim)?,
    })
}

fn run_affordability_probe(mut sim: Simulation) -> Result<PpecAffordabilityEvidence> {
    let caches = sim.artifact_caches.clone();
    if caches.len() != 2 || sim.organisms.len() != 2 {
        bail!("PPEC affordability probe requires two organisms and two caches");
    }
    let cache = &caches[1];
    for slot in &mut sim.occupancy {
        if matches!(slot, Some(Occupant::Organism(_))) {
            *slot = None;
        }
    }
    let positions = [
        hex_neighbor(
            (cache.q, cache.r),
            FacingDirection::West,
            sim.config.world_width as i32,
        ),
        hex_neighbor(
            (cache.q, cache.r),
            FacingDirection::East,
            sim.config.world_width as i32,
        ),
    ];
    let interaction_cost = sim.config.protocol_interaction_energy_cost;
    let insufficient_actor_energy = interaction_cost * 0.5;
    let organism_ids = sim
        .organisms
        .iter()
        .map(|organism| organism.id)
        .collect::<Vec<_>>();
    for (index, &(q, r)) in positions.iter().enumerate() {
        let idx = sim.cell_index(q, r);
        if sim.occupancy[idx].is_some() || sim.terrain_map[idx] {
            bail!("PPEC affordability placement collided at ({q}, {r})");
        }
        let organism = &mut sim.organisms[index];
        organism.q = q;
        organism.r = r;
        organism.facing = if index == 0 {
            FacingDirection::East
        } else {
            FacingDirection::West
        };
        if index == 0 {
            organism.energy = insufficient_actor_energy;
        }
        organism.energy_at_last_sensing = organism.energy;
        sim.occupancy[idx] = Some(Occupant::Organism(organism.id));
    }
    let response_bits = evaluate_public_protocol(&cache.public_program, &cache.challenge_bits)?;
    for &organism_id in &organism_ids {
        sim.queue_artifact_interaction(organism_id, cache.id, response_bits);
    }
    sim.debug_assert_consistent_state();
    let delta = sim.tick();
    let events = sim.artifact_events_last_turn.clone();
    let insufficient_nonclaiming_then_valid_succeeds = events.len() == 2
        && events[0].organism_id == organism_ids[0]
        && events[0].outcome == ArtifactInteractionOutcome::InsufficientEnergy
        && events[0].organism_energy_before == insufficient_actor_energy
        && events[0].organism_energy_after == insufficient_actor_energy
        && events[0].cache_energy_before == events[0].cache_energy_after
        && events[1].organism_id == organism_ids[1]
        && events[1].outcome == ArtifactInteractionOutcome::Released;
    let ledger = delta.metrics.energy_ledger_last_turn;
    Ok(PpecAffordabilityEvidence {
        interaction_cost,
        insufficient_actor_energy,
        artifact_id: cache.id,
        lower_organism_id: organism_ids[0],
        higher_organism_id: organism_ids[1],
        response_bits,
        events,
        insufficient_nonclaiming_then_valid_succeeds,
        ledger,
        energy_closed: ledger_closes(ledger),
        final_world_fingerprint: fingerprint(&sim)?,
    })
}

fn run_contention_probe(sim: Simulation) -> Result<PpecContentionEvidence> {
    let correct_lower_id_winner = run_contention_case(sim.clone(), true)?;
    let wrong_lower_id_winner = run_contention_case(sim, false)?;
    let all_passed = correct_lower_id_winner.exact_gate_passed
        && correct_lower_id_winner.energy_closed
        && wrong_lower_id_winner.exact_gate_passed
        && wrong_lower_id_winner.energy_closed;
    Ok(PpecContentionEvidence {
        correct_lower_id_winner,
        wrong_lower_id_winner,
        all_passed,
    })
}

fn run_contention_case(
    mut sim: Simulation,
    winner_response_correct: bool,
) -> Result<PpecContentionCaseEvidence> {
    let caches = sim.artifact_caches.clone();
    if caches.len() != 2 || sim.organisms.len() != 2 {
        bail!("PPEC contention probe requires two organisms and two caches");
    }
    let cache = caches[1].clone();
    let organism_ids = position_both_adjacent_to_cache(&mut sim, &cache)?;
    let lower_organism_energy_before = organism_energy(&sim, organism_ids[0])?;
    let higher_organism_energy_before = organism_energy(&sim, organism_ids[1])?;
    let expected_response = evaluate_public_protocol(&cache.public_program, &cache.challenge_bits)?;
    let lower_response = if winner_response_correct {
        expected_response
    } else {
        expected_response.wrapping_add(1) & 0b11
    };
    sim.queue_artifact_interaction(organism_ids[0], cache.id, lower_response);
    sim.queue_artifact_interaction(organism_ids[1], cache.id, expected_response);
    sim.debug_assert_consistent_state();

    let delta = sim.tick();
    let ledger = delta.metrics.energy_ledger_last_turn;
    let events = sim.artifact_events_last_turn.clone();
    let lower_organism_energy_after = organism_energy(&sim, organism_ids[0])?;
    let higher_organism_energy_after = organism_energy(&sim, organism_ids[1])?;
    let cache_after = sim
        .artifact_caches
        .binary_search_by_key(&cache.id, |candidate| candidate.id)
        .ok()
        .map(|index| &sim.artifact_caches[index]);
    let cache_present_after = cache_after.is_some();
    let cache_challenge_after = cache_after.map(|state| state.challenge_bits.clone());
    let tolerance = ledger.residual_tolerance;
    let one_cost = approximately_equal(
        ledger.protocol_interaction_cost_energy,
        f64::from(sim.config.protocol_interaction_energy_cost),
        tolerance,
    );
    let loser_unchanged = higher_organism_energy_before == higher_organism_energy_after;
    let canonical_sequence = events.len() == 2
        && events[0].organism_id == organism_ids[0]
        && events[0].artifact_id == cache.id
        && events[1].organism_id == organism_ids[1]
        && events[1].artifact_id == cache.id
        && events[1].outcome == ArtifactInteractionOutcome::ContendedThisTick
        && events[1].challenge_bits == cache.challenge_bits
        && events[1].cache_energy_before == cache.energy
        && events[1].response_bits == expected_response
        && events[1].expected_response_bits == Some(expected_response)
        && events[1].organism_energy_before == higher_organism_energy_before
        && events[1].organism_energy_after == higher_organism_energy_before
        && events[1].released_energy == 0.0
        && events[1].release_loss == 0.0;
    let exact_gate_passed = if winner_response_correct {
        canonical_sequence
            && events[0].outcome == ArtifactInteractionOutcome::Released
            && events[0].challenge_bits == cache.challenge_bits
            && events[0].cache_energy_before == cache.energy
            && events[0].cache_energy_after == 0.0
            && events[0].released_energy == cache.energy
            && events[0].release_loss == 0.0
            && events[1].cache_energy_after == 0.0
            && !cache_present_after
            && cache_challenge_after.is_none()
            && loser_unchanged
            && one_cost
            && approximately_equal(
                ledger.artifact_release_debit,
                f64::from(cache.energy),
                tolerance,
            )
            && approximately_equal(
                ledger.artifact_release_credit,
                f64::from(cache.energy),
                tolerance,
            )
            && approximately_equal(ledger.artifact_release_loss, 0.0, tolerance)
    } else {
        canonical_sequence
            && events[0].outcome == ArtifactInteractionOutcome::WrongResponse
            && events[0].challenge_bits == cache.challenge_bits
            && events[0].cache_energy_before == cache.energy
            && events[0].cache_energy_after == cache.energy
            && events[0].released_energy == 0.0
            && events[0].release_loss == 0.0
            && events[1].cache_energy_after == cache.energy
            && cache_present_after
            && cache_after.is_some_and(|state| {
                state.energy == cache.energy
                    && state.attempt_ordinal == cache.attempt_ordinal.saturating_add(1)
                    && state.challenge_bits != cache.challenge_bits
            })
            && loser_unchanged
            && one_cost
            && approximately_equal(ledger.artifact_release_debit, 0.0, tolerance)
            && approximately_equal(ledger.artifact_release_credit, 0.0, tolerance)
            && approximately_equal(ledger.artifact_release_loss, 0.0, tolerance)
    };
    Ok(PpecContentionCaseEvidence {
        winner_response_correct,
        artifact_id: cache.id,
        lower_organism_id: organism_ids[0],
        higher_organism_id: organism_ids[1],
        cache_energy_before: cache.energy,
        cache_challenge_before: cache.challenge_bits,
        cache_challenge_after,
        lower_organism_energy_before,
        lower_organism_energy_after,
        higher_organism_energy_before,
        higher_organism_energy_after,
        events,
        cache_present_after,
        exact_gate_passed,
        ledger,
        energy_closed: ledger_closes(ledger),
        final_world_fingerprint: fingerprint(&sim)?,
    })
}

fn position_both_adjacent_to_cache(
    sim: &mut Simulation,
    cache: &ArtifactCacheState,
) -> Result<[OrganismId; 2]> {
    for slot in &mut sim.occupancy {
        if matches!(slot, Some(Occupant::Organism(_))) {
            *slot = None;
        }
    }
    let positions = [
        (
            hex_neighbor(
                (cache.q, cache.r),
                FacingDirection::West,
                sim.config.world_width as i32,
            ),
            FacingDirection::East,
        ),
        (
            hex_neighbor(
                (cache.q, cache.r),
                FacingDirection::East,
                sim.config.world_width as i32,
            ),
            FacingDirection::West,
        ),
    ];
    let mut organism_ids = [OrganismId(0), OrganismId(0)];
    for (index, &((q, r), facing)) in positions.iter().enumerate() {
        let cell_index = sim.cell_index(q, r);
        if sim.occupancy[cell_index].is_some() || sim.terrain_map[cell_index] {
            bail!("PPEC same-cache placement collided at ({q}, {r})");
        }
        let organism = &mut sim.organisms[index];
        organism.q = q;
        organism.r = r;
        organism.facing = facing;
        organism.energy_at_last_sensing = organism.energy;
        organism_ids[index] = organism.id;
        sim.occupancy[cell_index] = Some(Occupant::Organism(organism.id));
    }
    if organism_ids[0] >= organism_ids[1] {
        bail!("PPEC founders are not in canonical organism-ID order");
    }
    sim.debug_assert_consistent_state();
    Ok(organism_ids)
}

fn organism_energy(sim: &Simulation, id: OrganismId) -> Result<f32> {
    sim.organisms
        .binary_search_by_key(&id, |organism| organism.id)
        .ok()
        .map(|index| sim.organisms[index].energy)
        .ok_or_else(|| anyhow!("PPEC probe organism {} is missing", id.0))
}

fn approximately_equal(actual: f64, expected: f64, tolerance: f64) -> bool {
    actual.is_finite() && expected.is_finite() && (actual - expected).abs() <= tolerance
}

fn position_actors(
    sim: &mut Simulation,
    caches: &[ArtifactCacheState],
    foreign: bool,
) -> Result<()> {
    for slot in &mut sim.occupancy {
        if matches!(slot, Some(Occupant::Organism(_))) {
            *slot = None;
        }
    }
    for organism_index in 0..sim.organisms.len() {
        let cache_index = if foreign {
            1 - organism_index
        } else {
            organism_index
        };
        let cache = &caches[cache_index];
        let (q, r) = hex_neighbor(
            (cache.q, cache.r),
            FacingDirection::West,
            sim.config.world_width as i32,
        );
        let idx = sim.cell_index(q, r);
        if sim.occupancy[idx].is_some() || sim.terrain_map[idx] {
            bail!("PPEC actor placement collided at ({q}, {r})");
        }
        let organism = &mut sim.organisms[organism_index];
        organism.q = q;
        organism.r = r;
        organism.facing = FacingDirection::East;
        organism.energy_at_last_sensing = organism.energy;
        sim.occupancy[idx] = Some(Occupant::Organism(organism.id));
    }
    sim.debug_assert_consistent_state();
    Ok(())
}

fn artifact_energy(sim: &Simulation) -> f64 {
    sim.artifact_caches
        .iter()
        .map(|cache| f64::from(cache.energy))
        .sum()
}

fn ledger_closes(row: EnergyLedgerRow) -> bool {
    [
        row.organism_residual,
        row.food_residual,
        row.artifact_residual,
        row.food_split_transfer_residual,
        row.artifact_release_transfer_residual,
        row.transfer_residual,
        row.total_residual,
    ]
    .into_iter()
    .all(|residual| residual.is_finite() && residual.abs() <= row.residual_tolerance)
}

fn distinct_count<T: Ord>(values: impl Iterator<Item = T>) -> usize {
    values.collect::<std::collections::BTreeSet<_>>().len()
}

fn aggregate_arm(
    contexts: &[PpecContextEvidence],
    arm: PpecControlArm,
) -> Result<PpecArmAggregate> {
    let evidence = contexts
        .iter()
        .map(|context| {
            context
                .arms
                .iter()
                .find(|candidate| candidate.arm == arm)
                .ok_or_else(|| anyhow!("PPEC context is missing a matched control arm"))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(PpecArmAggregate {
        arm,
        trials: evidence.len() * 2,
        accepted: evidence.iter().map(|arm| arm.accepted_count).sum(),
        released_energy: evidence.iter().map(|arm| arm.released_energy).sum(),
        release_loss: evidence.iter().map(|arm| arm.release_loss).sum(),
        all_energy_closed: evidence.iter().all(|arm| arm.energy_closed),
    })
}

fn build_public_semantics_evidence(
    contexts: &[PpecContextEvidence],
    aggregates: &[PpecArmAggregate],
) -> Result<PpecPublicSemanticsEvidence> {
    let mut expected_response_counts = [0_usize; 4];
    let mut responses_by_protocol =
        std::collections::BTreeMap::<String, std::collections::BTreeSet<u8>>::new();
    for context in contexts {
        for cache in &context.produced_caches {
            let response = evaluate_public_protocol(&cache.public_program, &cache.challenge_bits)?;
            expected_response_counts[usize::from(response)] += 1;
            responses_by_protocol
                .entry(public_protocol_fingerprint(&cache.public_program))
                .or_default()
                .insert(response);
        }
    }
    let total_protocol_challenges: usize = expected_response_counts.iter().sum();
    let minimum_required_per_response = (total_protocol_challenges / 10).max(1);
    let distinct_response_counts_by_protocol = responses_by_protocol
        .into_iter()
        .map(|(fingerprint, responses)| (fingerprint, responses.len()))
        .collect();
    let constant_response_acceptances = [
        aggregate_for(aggregates, PpecControlArm::ConstantResponse0)?.accepted,
        aggregate_for(aggregates, PpecControlArm::ConstantResponse1)?.accepted,
        aggregate_for(aggregates, PpecControlArm::ConstantResponse2)?.accepted,
        aggregate_for(aggregates, PpecControlArm::ConstantResponse3)?.accepted,
    ];
    let mut contexts_by_seed = std::collections::BTreeMap::<u64, Vec<&PpecContextEvidence>>::new();
    for context in contexts {
        contexts_by_seed
            .entry(context.run_seed)
            .or_default()
            .push(context);
    }
    let mut per_run_seed = Vec::with_capacity(contexts_by_seed.len());
    for (run_seed, seed_contexts) in contexts_by_seed {
        let mut seed_response_counts = [0_usize; 4];
        for context in &seed_contexts {
            for cache in &context.produced_caches {
                let response =
                    evaluate_public_protocol(&cache.public_program, &cache.challenge_bits)?;
                seed_response_counts[usize::from(response)] += 1;
            }
        }
        let seed_total: usize = seed_response_counts.iter().sum();
        let seed_minimum = (seed_total / 10).max(1);
        let seed_strong_max = seed_total / 2;
        let accepted = |arm| -> Result<usize> {
            seed_contexts
                .iter()
                .map(|context| {
                    arm_for(context, arm)
                        .map(|evidence| evidence.accepted_count)
                        .ok_or_else(|| anyhow!("PPEC seed context is missing a control arm"))
                })
                .sum()
        };
        let seed_constants = [
            accepted(PpecControlArm::ConstantResponse0)?,
            accepted(PpecControlArm::ConstantResponse1)?,
            accepted(PpecControlArm::ConstantResponse2)?,
            accepted(PpecControlArm::ConstantResponse3)?,
        ];
        let seed_code = accepted(PpecControlArm::CodePermutation)?;
        let seed_challenge = accepted(PpecControlArm::ChallengePermutation)?;
        let seed_random = accepted(PpecControlArm::RandomResponse)?;
        let all_passed = seed_response_counts
            .iter()
            .all(|count| *count >= seed_minimum)
            && seed_constants == seed_response_counts
            && seed_constants.iter().all(|count| *count <= seed_strong_max)
            && seed_code <= seed_strong_max
            && seed_challenge <= seed_strong_max
            && seed_random <= seed_strong_max;
        per_run_seed.push(PpecSeedPublicSemanticsEvidence {
            run_seed,
            total_protocol_challenges: seed_total,
            expected_response_counts: seed_response_counts,
            minimum_required_per_response: seed_minimum,
            strong_control_max_acceptances: seed_strong_max,
            constant_response_acceptances: seed_constants,
            code_permutation_acceptances: seed_code,
            challenge_permutation_acceptances: seed_challenge,
            random_response_acceptances: seed_random,
            all_passed,
        });
    }
    Ok(PpecPublicSemanticsEvidence {
        total_protocol_challenges,
        expected_response_counts,
        minimum_required_per_response,
        distinct_response_counts_by_protocol,
        minimum_distinct_responses_per_protocol: 2,
        strong_control_max_acceptances: total_protocol_challenges / 2,
        constant_response_acceptances,
        code_permutation_acceptances: aggregate_for(aggregates, PpecControlArm::CodePermutation)?
            .accepted,
        challenge_permutation_acceptances: aggregate_for(
            aggregates,
            PpecControlArm::ChallengePermutation,
        )?
        .accepted,
        random_response_acceptances: aggregate_for(aggregates, PpecControlArm::RandomResponse)?
            .accepted,
        per_run_seed,
    })
}

fn build_gates(
    contexts: &[PpecContextEvidence],
    aggregates: &[PpecArmAggregate],
    public_semantics: &PpecPublicSemanticsEvidence,
) -> Result<PpecMechanismGates> {
    let own = aggregate_for(aggregates, PpecControlArm::OwnProtocol)?;
    let foreign = aggregate_for(aggregates, PpecControlArm::ForeignProtocol)?;
    let no_payoff = aggregate_for(aggregates, PpecControlArm::NoPayoff)?;
    let knockout = aggregate_for(aggregates, PpecControlArm::ArtifactKnockout)?;
    let production_split_closed = contexts.iter().all(|context| {
        context.production.plants_consumed == 2
            && context.production.cache_count == 2
            && context.production.same_tick_future_requests_rejected
            && (context.production.cache_energy - context.production.expected_cache_energy).abs()
                <= context.production.ledger.residual_tolerance
            && context.production.energy_closed
    });
    let persistent_nonblocking_state = contexts.iter().all(|context| {
        context.production.cache_cells_absent_from_occupancy && context.persistence.caches_unchanged
    });
    let exact_save_load_replay = contexts.iter().all(|context| {
        context.persistence.queued_request_count_before_save == 1
            && context.persistence.queued_events_after_advance.len() == 1
            && context.persistence.queued_events_after_advance[0].outcome
                == ArtifactInteractionOutcome::Released
            && context.persistence.exact_save_load_roundtrip
            && context.persistence.post_load_advance_exact
            && ledger_closes(context.persistence.post_advance_energy_ledger)
    });
    let variable_public_protocols = contexts.iter().all(|context| {
        context.production.distinct_protocol_fingerprints == 2
            && context.production.distinct_program_lengths == 2
            && context.produced_caches.iter().all(|cache| {
                cache.challenge_bits.len() == usize::from(cache.public_program.input_arity)
            })
    });
    let public_output_balance = public_semantics
        .expected_response_counts
        .iter()
        .all(|count| *count >= public_semantics.minimum_required_per_response);
    let constant_controls_match_expected_histogram =
        public_semantics.constant_response_acceptances == public_semantics.expected_response_counts;
    let challenge_varies_response_within_each_protocol = public_semantics
        .distinct_response_counts_by_protocol
        .values()
        .all(|count| *count >= public_semantics.minimum_distinct_responses_per_protocol);
    let every_seed_public_semantics_passes = public_semantics
        .per_run_seed
        .iter()
        .all(|seed| seed.all_passed);
    let own_use_positive = own.accepted == own.trials && own.released_energy > 0.0;
    let foreign_use_positive = foreign.accepted == foreign.trials && foreign.released_energy > 0.0;
    let no_payoff_is_causally_separated = no_payoff.accepted == no_payoff.trials
        && no_payoff.released_energy == 0.0
        && no_payoff.release_loss > 0.0;
    let code_permutation_strongly_reduces_acceptance = public_semantics
        .code_permutation_acceptances
        <= public_semantics.strong_control_max_acceptances;
    let challenge_permutation_strongly_reduces_acceptance = public_semantics
        .challenge_permutation_acceptances
        <= public_semantics.strong_control_max_acceptances;
    let artifact_knockout_removes_payoff = knockout.accepted == 0
        && knockout.released_energy == 0.0
        && contexts.iter().all(|context| {
            arm_for(context, PpecControlArm::ArtifactKnockout)
                .is_some_and(|arm| arm.knocked_out_energy > 0.0)
        });
    let all_constant_responses_strongly_reduced = public_semantics
        .constant_response_acceptances
        .iter()
        .all(|accepted| *accepted <= public_semantics.strong_control_max_acceptances);
    let random_response_strongly_reduced = public_semantics.random_response_acceptances
        <= public_semantics.strong_control_max_acceptances;
    let invalid_lower_id_does_not_claim_slot = contexts.iter().all(|context| {
        arm_for(context, PpecControlArm::OwnProtocol)
            .is_some_and(|arm| arm.lower_id_nonadjacent_then_valid_succeeds)
    });
    let insufficient_energy_does_not_claim_slot = contexts.iter().all(|context| {
        context
            .affordability
            .insufficient_nonclaiming_then_valid_succeeds
            && context.affordability.energy_closed
    });
    let valid_contention_is_canonical_and_accounted =
        contexts.iter().all(|context| context.contention.all_passed);
    let all_energy_ledgers_close = aggregates
        .iter()
        .all(|aggregate| aggregate.all_energy_closed)
        && contexts.iter().all(|context| {
            context.production.energy_closed
                && context.affordability.energy_closed
                && context.contention.correct_lower_id_winner.energy_closed
                && context.contention.wrong_lower_id_winner.energy_closed
                && ledger_closes(context.persistence.post_advance_energy_ledger)
        });
    let mut gates = PpecMechanismGates {
        production_split_closed,
        persistent_nonblocking_state,
        exact_save_load_replay,
        variable_public_protocols,
        public_output_balance,
        constant_controls_match_expected_histogram,
        challenge_varies_response_within_each_protocol,
        every_seed_public_semantics_passes,
        own_use_positive,
        foreign_use_positive,
        no_payoff_is_causally_separated,
        code_permutation_strongly_reduces_acceptance,
        challenge_permutation_strongly_reduces_acceptance,
        artifact_knockout_removes_payoff,
        all_constant_responses_strongly_reduced,
        random_response_strongly_reduced,
        invalid_lower_id_does_not_claim_slot,
        insufficient_energy_does_not_claim_slot,
        valid_contention_is_canonical_and_accounted,
        all_energy_ledgers_close,
        all_passed: false,
    };
    gates.all_passed = gates.production_split_closed
        && gates.persistent_nonblocking_state
        && gates.exact_save_load_replay
        && gates.variable_public_protocols
        && gates.public_output_balance
        && gates.constant_controls_match_expected_histogram
        && gates.challenge_varies_response_within_each_protocol
        && gates.every_seed_public_semantics_passes
        && gates.own_use_positive
        && gates.foreign_use_positive
        && gates.no_payoff_is_causally_separated
        && gates.code_permutation_strongly_reduces_acceptance
        && gates.challenge_permutation_strongly_reduces_acceptance
        && gates.artifact_knockout_removes_payoff
        && gates.all_constant_responses_strongly_reduced
        && gates.random_response_strongly_reduced
        && gates.invalid_lower_id_does_not_claim_slot
        && gates.insufficient_energy_does_not_claim_slot
        && gates.valid_contention_is_canonical_and_accounted
        && gates.all_energy_ledgers_close;
    Ok(gates)
}

fn aggregate_for(
    aggregates: &[PpecArmAggregate],
    arm: PpecControlArm,
) -> Result<&PpecArmAggregate> {
    aggregates
        .iter()
        .find(|aggregate| aggregate.arm == arm)
        .ok_or_else(|| anyhow!("PPEC aggregate is missing a matched control arm"))
}

fn arm_for(context: &PpecContextEvidence, arm: PpecControlArm) -> Option<&PpecArmEvidence> {
    context.arms.iter().find(|candidate| candidate.arm == arm)
}

fn fingerprint<T: Serialize>(value: &T) -> Result<String> {
    exact_fingerprint(value).map_err(|error| anyhow!(error.to_string()))
}
