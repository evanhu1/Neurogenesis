mod commit;
mod intents;
mod lifecycle;
mod moves;
mod snapshot;

#[cfg(feature = "instrumentation")]
use crate::brain::scan_rays;
use crate::brain::{action_index, evaluate_brain, BrainEvalContext, BrainScratch};
use crate::grid::{hex_neighbor, rotate_left, rotate_right};
use crate::plasticity::{apply_runtime_weight_updates, compute_pending_coactivations};
use crate::Simulation;
#[cfg(feature = "profiling")]
use crate::{profiling, profiling::TurnPhase};
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
#[cfg(feature = "instrumentation")]
use sim_types::ActionRecord;
use sim_types::{
    ActionType, EnergyLedgerRow, EntityId, FacingDirection, FoodKind, FoodState, Occupant,
    OrganismFacing, OrganismId, OrganismMove, OrganismState, RemovedEntityPosition, SpeciesId,
    TickDelta,
};
use std::sync::Arc;
#[cfg(feature = "profiling")]
use std::time::Instant;

const RNG_TURN_MIX: u64 = 0x9E37_79B9_7F4A_7C15;
const RNG_ORGANISM_MIX: u64 = 0xBF58_476D_1CE4_E5B9;
const RNG_PREY_MIX: u64 = 0x2545_F491_4F6C_DD1D;
pub(crate) const ATTACK_DAMAGE_FRACTION: f32 = 0.5;
pub(crate) const HEALTH_REGEN_FRACTION: f32 = 0.10;
const INTENT_PARALLEL_MIN_LEN: usize = 64;
/// Conservative f32 roundoff budget for a tick's compartment totals and
/// explicitly enumerated flows. The scale term is computed per row, so tiny
/// task-energy escrows remain tightly audited while large worlds do not fail
/// merely because many persisted f32 values were summed.
const ENERGY_LEDGER_EPSILON_MULTIPLIER: f64 = 32.0;

#[derive(Debug, Clone, Copy)]
struct OrganismIntent {
    idx: usize,
    id: OrganismId,
    from: (i32, i32),
    facing_after_actions: FacingDirection,
    wants_move: bool,
    wants_eat: bool,
    wants_attack: bool,
    move_target: Option<(i32, i32)>,
    interaction_target: Option<(i32, i32)>,
    move_confidence: f32,
    took_action: bool,
    synapse_ops: u64,
}

struct BuiltIntent {
    intent: OrganismIntent,
    #[cfg(feature = "instrumentation")]
    action_record: Option<ActionRecord>,
}

#[derive(Debug, Clone, Copy)]
struct MoveCandidate {
    actor_idx: usize,
    actor_id: OrganismId,
    from: (i32, i32),
    target: (i32, i32),
    confidence: f32,
}

#[derive(Debug, Clone, Copy)]
struct MoveResolution {
    actor_idx: usize,
    actor_id: OrganismId,
    from: (i32, i32),
    to: (i32, i32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AttackOutcome {
    NoOrganismTarget,
    SamePoolBlocked,
    Missed,
    NonlethalHit,
    Killed,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct AttackEvent {
    pub(crate) turn: u64,
    pub(crate) attacker_id: OrganismId,
    pub(crate) attacker_species_id: SpeciesId,
    pub(crate) victim_id: Option<OrganismId>,
    pub(crate) victim_species_id: Option<SpeciesId>,
    pub(crate) outcome: AttackOutcome,
    pub(crate) victim_health_before: f32,
    pub(crate) victim_health_after: f32,
    pub(crate) damage: f32,
    pub(crate) energy_gained: f32,
}

/// Reusable per-tick scratch buffers owned by `Simulation` so the commit /
/// reproduction / move phases avoid O(population) heap allocations every tick.
/// Each user takes a buffer with `std::mem::take`, clears + resizes it, and
/// returns it when done, so contents never leak across ticks.
#[derive(Debug, Default)]
pub(crate) struct TurnScratch {
    removed_food: Vec<bool>,
    dead_organisms: Vec<bool>,
    gestation_started: Vec<bool>,
    move_candidates: Vec<(usize, MoveCandidate)>,
    move_resolutions: Vec<MoveResolution>,
    intents: Vec<OrganismIntent>,
}

#[derive(Default)]
struct CommitResult {
    moves: Vec<OrganismMove>,
    facing_updates: Vec<OrganismFacing>,
    removed_positions: Vec<RemovedEntityPosition>,
    food_spawned: Vec<FoodState>,
    consumptions: u64,
    plant_consumptions: u64,
    predations: u64,
    actions_applied: u64,
    attack_events: Vec<AttackEvent>,
    action_cost_energy: f64,
    food_consumption_debit: f64,
    food_consumption_credit: f64,
    predation_prey_energy_removed: f64,
    predation_energy_credit: f64,
    predation_retention_loss: f64,
    corpse_source_energy_removed: f64,
    corpse_retention_loss: f64,
    removal_adjustment: f64,
}

#[derive(Debug, Default)]
struct LifecycleEnergyFlow {
    passive_metabolism_energy: f64,
    unrecycled_energy_removed: f64,
    removal_adjustment: f64,
}

macro_rules! profile_turn_phase {
    ($phase:expr, $body:block) => {{
        #[cfg(feature = "profiling")]
        let phase_started = Instant::now();
        let value = $body;
        #[cfg(feature = "profiling")]
        profiling::record_turn_phase($phase, phase_started.elapsed());
        value
    }};
}

fn build_sim_parallel_pool(thread_count: u32) -> Arc<ThreadPool> {
    let requested_threads = thread_count.max(1) as usize;
    Arc::new(
        ThreadPoolBuilder::new()
            .num_threads(requested_threads)
            .thread_name(|idx| format!("sim-core-worker-{idx}"))
            .build()
            .expect("failed to build sim-core rayon thread pool"),
    )
}

impl Simulation {
    pub(crate) fn parallel_pool(&self) -> Arc<ThreadPool> {
        let threads = self.config.intent_parallel_threads;
        self.cached_thread_pool
            .get_or_init(|| build_sim_parallel_pool(threads))
            .clone()
    }

    pub fn tick(&mut self) -> TickDelta {
        self.clear_turn_transient_state();

        let organism_energy_before = checked_energy_total(
            "organism energy before tick",
            self.organisms.iter().map(|organism| organism.energy),
        );
        let food_energy_before = checked_energy_total(
            "food energy before tick",
            self.foods.iter().map(|food| food.energy),
        );

        #[cfg(feature = "profiling")]
        let tick_started = Instant::now();

        let (
            starvations,
            age_deaths,
            lifecycle_removed_positions,
            lifecycle_food_spawned,
            lifecycle_energy,
        ) = profile_turn_phase!(TurnPhase::Lifecycle, { self.lifecycle_phase() });

        let world_width = self.config.world_width as i32;

        let intents = profile_turn_phase!(TurnPhase::Intents, { self.build_intents(world_width) });
        let synapse_ops = intents.iter().map(|intent| intent.synapse_ops).sum::<u64>();

        // No in-world reproduction exists: nothing gestates, so the
        // gestation-started scratch is always empty. It is still threaded into
        // the commit phase (and `compact_organism_state`) to preserve the exact
        // organism-compaction path.
        let mut gestation_started = std::mem::take(&mut self.turn_scratch.gestation_started);
        gestation_started.clear();
        gestation_started.resize(self.organisms.len(), false);

        let resolutions =
            profile_turn_phase!(TurnPhase::MoveResolution, { self.resolve_moves(&intents) });

        let mut commit = profile_turn_phase!(TurnPhase::Commit, {
            self.commit_phase(world_width, &intents, &resolutions, &mut gestation_started)
        });
        self.attack_events_last_turn = std::mem::take(&mut commit.attack_events);
        // Old-age corpses spawned during the lifecycle phase ride the same
        // food-spawned channel as commit-phase corpses so the client sees them.
        commit.food_spawned.extend(lifecycle_food_spawned);
        self.turn_scratch.gestation_started = gestation_started;

        // Commit was the last reader of the intents and move resolutions;
        // return the buffers to TurnScratch for reuse next tick.
        self.turn_scratch.intents = intents;
        self.turn_scratch.move_resolutions = resolutions;

        profile_turn_phase!(TurnPhase::Age, {
            self.increment_age_for_survivors();
        });

        // Reproduction is owned entirely by the NEAT outer loop, so the world
        // produces no births; the former Spawn phase now only runs the
        // post-commit plasticity pass.
        profile_turn_phase!(TurnPhase::Spawn, {
            self.apply_post_commit_runtime_weight_updates();
        });
        let spawned: Vec<OrganismState> = Vec::new();

        profile_turn_phase!(TurnPhase::ConsistencyCheck, {
            self.debug_assert_consistent_state();
        });

        let energy_ledger = build_energy_ledger_row(
            self.turn.saturating_add(1),
            organism_energy_before,
            food_energy_before,
            &self.organisms,
            &self.foods,
            &commit,
            &lifecycle_energy,
        );

        let removed_positions = profile_turn_phase!(TurnPhase::MetricsAndDelta, {
            self.turn = self.turn.saturating_add(1);
            self.metrics.turns = self.turn;
            self.metrics.synapse_ops_last_turn = synapse_ops;
            self.metrics.actions_applied_last_turn = commit.actions_applied;
            self.metrics.consumptions_last_turn = commit.consumptions;
            self.metrics.plant_consumptions_last_turn = commit.plant_consumptions;
            self.metrics.predations_last_turn = commit.predations;
            self.metrics.total_consumptions += commit.consumptions;
            self.metrics.total_plant_consumptions += commit.plant_consumptions;
            self.metrics.starvations_last_turn = starvations;
            self.metrics.age_deaths_last_turn = age_deaths;
            self.metrics.energy_ledger_last_turn = energy_ledger;
            self.refresh_population_metrics();

            let mut removed_positions = commit.removed_positions;
            removed_positions.extend(lifecycle_removed_positions);
            removed_positions
        });

        #[cfg(feature = "profiling")]
        profiling::record_tick_total(tick_started.elapsed());

        TickDelta {
            turn: self.turn,
            moves: commit.moves,
            facing_updates: commit.facing_updates,
            removed_positions,
            spawned,
            food_spawned: commit.food_spawned,
            metrics: self.metrics.clone(),
        }
    }

    fn apply_post_commit_runtime_weight_updates(&mut self) {
        if !self.config.runtime_plasticity_enabled || self.config.force_random_actions {
            // Nothing on this path consumes the plasticity pass, so skipping it
            // keeps organisms byte-identical. Sensing momentum uses the
            // separate `energy_at_last_sensing` stash, untouched here.
            return;
        }

        let any_learners = self
            .organisms
            .iter()
            .any(|o| o.genome.plasticity.hebb_eta_gain > 0.0);

        #[cfg(feature = "profiling")]
        let plasticity_started = Instant::now();

        let body_mass_metabolic_cost_coeff = self.config.body_mass_metabolic_cost_coeff;
        // Organisms with `age_turns == 0` are skipped: their brains have never
        // been evaluated (intents ran before they existed), so their eligibility
        // traces carry no pending coactivation yet; they get their first weight
        // update at the end of their first active tick instead. The skip is a
        // pure per-organism predicate, so rayon scheduling cannot affect
        // determinism.
        // Acquired before the disjoint field borrows below; the pool is
        // already cached from the intent phase, so this is an Arc clone.
        let thread_pool = self.parallel_pool();
        let organisms = &mut self.organisms;
        if any_learners {
            thread_pool.install(|| {
                organisms
                    .par_iter_mut()
                    .with_min_len(INTENT_PARALLEL_MIN_LEN)
                    .for_each(|organism| {
                        if organism.age_turns == 0 {
                            return;
                        }
                        apply_runtime_weight_updates(organism, body_mass_metabolic_cost_coeff);
                    });
            });
        } else {
            for organism in organisms.iter_mut() {
                if organism.age_turns == 0 {
                    continue;
                }
                apply_runtime_weight_updates(organism, body_mass_metabolic_cost_coeff);
            }
        }

        #[cfg(feature = "profiling")]
        profiling::record_brain_plasticity_total(plasticity_started.elapsed());
    }
}

fn checked_energy_total(label: &str, values: impl Iterator<Item = f32>) -> f64 {
    let mut total = 0.0_f64;
    for value in values {
        assert!(
            value.is_finite(),
            "energy ledger: nonfinite {label}: {value}"
        );
        total += f64::from(value);
    }
    assert!(total.is_finite(), "energy ledger: nonfinite {label} total");
    total
}

fn build_energy_ledger_row(
    turn: u64,
    organism_energy_before: f64,
    food_energy_before: f64,
    organisms: &[OrganismState],
    foods: &[FoodState],
    commit: &CommitResult,
    lifecycle: &LifecycleEnergyFlow,
) -> EnergyLedgerRow {
    let organism_energy_after = checked_energy_total(
        "organism energy after tick",
        organisms.iter().map(|organism| organism.energy),
    );
    let food_energy_after = checked_energy_total(
        "food energy after tick",
        foods.iter().map(|food| food.energy),
    );
    let plant_spawn_energy = checked_energy_total(
        "plant spawn energy",
        commit
            .food_spawned
            .iter()
            .filter(|food| food.kind == FoodKind::Plant)
            .map(|food| food.energy),
    );
    let corpse_spawn_energy = checked_energy_total(
        "corpse spawn energy",
        commit
            .food_spawned
            .iter()
            .filter(|food| food.kind == FoodKind::Corpse)
            .map(|food| food.energy),
    );
    let removal_adjustment = lifecycle.removal_adjustment + commit.removal_adjustment;

    let organism_expected =
        organism_energy_before - lifecycle.passive_metabolism_energy - commit.action_cost_energy
            + commit.food_consumption_credit
            + commit.predation_energy_credit
            - lifecycle.unrecycled_energy_removed
            - commit.predation_prey_energy_removed
            - commit.corpse_source_energy_removed;
    let food_expected = food_energy_before - commit.food_consumption_debit
        + plant_spawn_energy
        + corpse_spawn_energy;
    let organism_residual = organism_energy_after - organism_expected;
    let food_residual = food_energy_after - food_expected;
    let transfer_residual = commit.food_consumption_credit - commit.food_consumption_debit;
    let total_expected = organism_energy_before + food_energy_before + plant_spawn_energy
        - lifecycle.passive_metabolism_energy
        - commit.action_cost_energy
        - commit.predation_retention_loss
        - commit.corpse_retention_loss
        + removal_adjustment;
    let total_residual = organism_energy_after + food_energy_after - total_expected;
    let flow_scale = (organism_energy_before + food_energy_before).abs()
        + plant_spawn_energy.abs()
        + lifecycle.passive_metabolism_energy.abs()
        + commit.action_cost_energy.abs()
        + commit.food_consumption_debit.abs()
        + commit.food_consumption_credit.abs()
        + commit.predation_prey_energy_removed.abs()
        + commit.predation_energy_credit.abs()
        + commit.predation_retention_loss.abs()
        + commit.corpse_source_energy_removed.abs()
        + corpse_spawn_energy.abs()
        + commit.corpse_retention_loss.abs()
        + lifecycle.unrecycled_energy_removed.abs()
        + removal_adjustment.abs();
    let residual_tolerance =
        ENERGY_LEDGER_EPSILON_MULTIPLIER * f64::from(f32::EPSILON) * flow_scale.max(1.0);

    let row = EnergyLedgerRow {
        turn,
        organism_energy_before,
        organism_energy_after,
        food_energy_before,
        food_energy_after,
        plant_spawn_energy,
        passive_metabolism_energy: lifecycle.passive_metabolism_energy,
        action_cost_energy: commit.action_cost_energy,
        food_consumption_debit: commit.food_consumption_debit,
        food_consumption_credit: commit.food_consumption_credit,
        predation_prey_energy_removed: commit.predation_prey_energy_removed,
        predation_energy_credit: commit.predation_energy_credit,
        predation_retention_loss: commit.predation_retention_loss,
        corpse_source_energy_removed: commit.corpse_source_energy_removed,
        corpse_spawn_energy,
        corpse_retention_loss: commit.corpse_retention_loss,
        unrecycled_energy_removed: lifecycle.unrecycled_energy_removed,
        removal_adjustment,
        organism_residual,
        food_residual,
        total_residual,
        transfer_residual,
        residual_tolerance,
    };
    assert_energy_ledger_closes(&row);
    row
}

fn assert_energy_ledger_closes(row: &EnergyLedgerRow) {
    let finite_values = [
        row.organism_energy_before,
        row.organism_energy_after,
        row.food_energy_before,
        row.food_energy_after,
        row.plant_spawn_energy,
        row.passive_metabolism_energy,
        row.action_cost_energy,
        row.food_consumption_debit,
        row.food_consumption_credit,
        row.predation_prey_energy_removed,
        row.predation_energy_credit,
        row.predation_retention_loss,
        row.corpse_source_energy_removed,
        row.corpse_spawn_energy,
        row.corpse_retention_loss,
        row.unrecycled_energy_removed,
        row.removal_adjustment,
        row.organism_residual,
        row.food_residual,
        row.total_residual,
        row.transfer_residual,
        row.residual_tolerance,
    ];
    assert!(
        finite_values.iter().all(|value| value.is_finite()),
        "energy ledger: nonfinite row at turn {}: {row:?}",
        row.turn
    );
    assert!(
        row.predation_retention_loss >= -row.residual_tolerance
            && row.corpse_retention_loss >= -row.residual_tolerance,
        "energy ledger: negative retention loss at turn {}: {row:?}",
        row.turn
    );
    for (label, residual) in [
        ("organism compartment", row.organism_residual),
        ("food compartment", row.food_residual),
        ("food transfer", row.transfer_residual),
        ("total compartments", row.total_residual),
    ] {
        assert!(
            residual.abs() <= row.residual_tolerance,
            "energy ledger: {label} does not close at turn {} (residual {}, tolerance {}): {row:?}",
            row.turn,
            residual,
            row.residual_tolerance
        );
    }
}

fn organism_health_regeneration(organism: &OrganismState, regen_fraction: f32) -> f32 {
    (organism.max_health.max(1.0) * regen_fraction).max(0.0)
}

fn organism_index_by_id(organisms: &[OrganismState], id: OrganismId) -> Option<usize> {
    organisms
        .binary_search_by_key(&id, |organism| organism.id)
        .ok()
}

fn food_index_by_id(foods: &[FoodState], id: sim_types::FoodId) -> Option<usize> {
    foods.binary_search_by_key(&id, |food| food.id).ok()
}

fn action_rng_seed(sim_seed: u64, tick: u64, organism_id: OrganismId) -> u64 {
    let mixed =
        sim_seed ^ tick.wrapping_mul(RNG_TURN_MIX) ^ organism_id.0.wrapping_mul(RNG_ORGANISM_MIX);
    mix_u64(mixed)
}

pub(crate) fn deterministic_action_sample(
    sim_seed: u64,
    tick: u64,
    organism_id: OrganismId,
) -> f32 {
    let sample = (action_rng_seed(sim_seed, tick, organism_id) >> 40) as u32;
    sample as f32 / ((1_u32 << 24) - 1) as f32
}

/// Predation roll in [0, 1] as a deterministic hash of
/// `(seed, turn, predator id, prey id)` — same scheme as
/// `deterministic_action_sample`, so attack outcomes are independent of
/// commit iteration order and shared RNG state.
fn deterministic_predation_sample(
    sim_seed: u64,
    tick: u64,
    predator_id: OrganismId,
    prey_id: OrganismId,
) -> f32 {
    let mixed = sim_seed
        ^ tick.wrapping_mul(RNG_TURN_MIX)
        ^ predator_id.0.wrapping_mul(RNG_ORGANISM_MIX)
        ^ prey_id.0.wrapping_mul(RNG_PREY_MIX);
    let sample = (mix_u64(mixed) >> 40) as u32;
    sample as f32 / ((1_u32 << 24) - 1) as f32
}

fn uniform_random_action(action_sample: f32, predation_enabled: bool) -> ActionType {
    let active_count = ActionType::active(predation_enabled).count();
    let buckets = active_count + 1;
    let scaled = action_sample.clamp(0.0, 1.0 - f32::EPSILON) * buckets as f32;
    let idx = (scaled.floor() as usize).min(buckets - 1);
    ActionType::active(predation_enabled)
        .nth(idx)
        .unwrap_or(ActionType::Idle)
}

fn mix_u64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    value
}
