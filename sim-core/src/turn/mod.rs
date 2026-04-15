mod commit;
mod intents;
mod lifecycle;
mod moves;
mod reproduction;
mod snapshot;

#[cfg(feature = "instrumentation")]
use crate::brain::scan_rays;
use crate::brain::{action_index, evaluate_brain, BrainEvalContext, BrainScratch, ACTION_COUNT};
use crate::grid::{hex_neighbor, opposite_direction, rotate_left, rotate_right, wrap_position};
use crate::plasticity::{apply_runtime_weight_updates, compute_pending_coactivations};
use crate::spawn::{ReproductionSpawn, SpawnRequest, SpawnRequestKind};
#[cfg(feature = "profiling")]
use crate::{profiling, profiling::TurnPhase};
use crate::{PendingActionKind, PendingActionState, Simulation};
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use reproduction::ReproductionPhaseState;
#[cfg(feature = "instrumentation")]
use sim_types::ActionRecord;
use sim_types::{
    ActionType, EntityId, FacingDirection, FoodKind, FoodState, Occupant, OrganismFacing,
    OrganismId, OrganismMove, OrganismState, RemovedEntityPosition, TickDelta, VisualProperties,
};
use std::collections::HashSet;
use std::sync::Arc;
#[cfg(feature = "profiling")]
use std::time::Instant;

const RNG_TURN_MIX: u64 = 0x9E37_79B9_7F4A_7C15;
const RNG_ORGANISM_MIX: u64 = 0xBF58_476D_1CE4_E5B9;
const PLANT_CONSUMPTION_ENERGY_FRACTION: f32 = 0.20;
const CORPSE_CONSUMPTION_ENERGY_FRACTION: f32 = 0.80;
const SPIKE_DAMAGE_FRACTION: f32 = 0.10;
const ATTACK_DAMAGE_FRACTION: f32 = 0.5;
const HEALTH_REGEN_FRACTION: f32 = 0.10;
const INTENT_PARALLEL_MIN_LEN: usize = 64;

#[derive(Clone, Copy)]
struct SnapshotOrganismState {
    q: i32,
    r: i32,
    facing: FacingDirection,
}

struct TurnSnapshot {
    world_width: i32,
    organism_count: usize,
}

#[derive(Clone, Copy)]
struct OrganismIntent {
    idx: usize,
    id: OrganismId,
    from: (i32, i32),
    facing_after_actions: FacingDirection,
    wants_move: bool,
    wants_eat: bool,
    wants_attack: bool,
    wants_reproduce: bool,
    move_target: Option<(i32, i32)>,
    interaction_target: Option<(i32, i32)>,
    move_confidence: f32,
    action_cost_count: u8,
    synapse_ops: u64,
}

struct BuiltIntent {
    intent: OrganismIntent,
    #[cfg(feature = "instrumentation")]
    action_record: Option<ActionRecord>,
}

#[derive(Clone, Copy)]
struct MoveCandidate {
    actor_idx: usize,
    actor_id: OrganismId,
    from: (i32, i32),
    target: (i32, i32),
    confidence: f32,
}

#[derive(Clone, Copy)]
struct MoveResolution {
    actor_idx: usize,
    actor_id: OrganismId,
    from: (i32, i32),
    to: (i32, i32),
}

#[derive(Default)]
struct CommitResult {
    moves: Vec<OrganismMove>,
    facing_updates: Vec<OrganismFacing>,
    removed_positions: Vec<RemovedEntityPosition>,
    food_spawned: Vec<FoodState>,
    consumptions: u64,
    predations: u64,
    actions_applied: u64,
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
        self.reconcile_pending_actions();
        self.reconcile_reward_ledgers();
        self.clear_turn_transient_state();

        #[cfg(feature = "profiling")]
        let tick_started = Instant::now();

        let (starvations, starved_removed_positions, lifecycle_reproduction_events) =
            profile_turn_phase!(TurnPhase::Lifecycle, { self.lifecycle_phase() });

        let snapshot = TurnSnapshot {
            world_width: self.config.world_width as i32,
            organism_count: self.organisms.len(),
        };

        let intents = profile_turn_phase!(TurnPhase::Intents, { self.build_intents(&snapshot) });
        let synapse_ops = intents.iter().map(|intent| intent.synapse_ops).sum::<u64>();

        let mut reproduction_state = ReproductionPhaseState::new(snapshot.organism_count);
        reproduction_state.extend_reproduction_events(lifecycle_reproduction_events);
        profile_turn_phase!(TurnPhase::Reproduction, {
            reproduction_state.apply_triggers(
                &mut self.organisms,
                &mut self.pending_actions,
                &intents,
                &self.occupancy,
                snapshot.world_width,
                #[cfg(feature = "instrumentation")]
                &mut self.action_records,
            );
        });

        let resolutions = profile_turn_phase!(TurnPhase::MoveResolution, {
            self.resolve_moves(snapshot.world_width, &self.occupancy, &intents)
        });

        let commit = profile_turn_phase!(TurnPhase::Commit, {
            let commit = self.commit_phase(
                snapshot.world_width,
                &intents,
                &resolutions,
                reproduction_state.gestation_started_this_tick_mut(),
            );
            reproduction_state.queue_completions(self, snapshot.world_width);
            commit
        });

        profile_turn_phase!(TurnPhase::Age, {
            self.increment_age_for_survivors();
        });

        let spawned = profile_turn_phase!(TurnPhase::Spawn, {
            let spawn_requests = reproduction_state.spawn_requests_mut();
            let reproduction_spawn_count = spawn_requests.len();
            self.enqueue_periodic_injections(spawn_requests);
            let spawned = self.resolve_spawn_requests(spawn_requests);
            let reproduction_events = reproduction_state
                .finalize_reproduction_events(&spawned[..reproduction_spawn_count]);
            self.apply_post_commit_runtime_weight_updates();
            (spawned, reproduction_events)
        });
        let reproductions = spawned.0.len() as u64;

        profile_turn_phase!(TurnPhase::ConsistencyCheck, {
            self.debug_assert_consistent_state();
        });

        let removed_positions = profile_turn_phase!(TurnPhase::MetricsAndDelta, {
            self.turn = self.turn.saturating_add(1);
            self.metrics.turns = self.turn;
            self.metrics.synapse_ops_last_turn = synapse_ops;
            self.metrics.actions_applied_last_turn = commit.actions_applied + reproductions;
            self.metrics.consumptions_last_turn = commit.consumptions;
            self.metrics.predations_last_turn = commit.predations;
            self.metrics.total_consumptions += commit.consumptions;
            self.metrics.reproductions_last_turn = reproductions;
            self.metrics.starvations_last_turn = starvations;
            self.refresh_population_metrics();

            let mut removed_positions = commit.removed_positions;
            removed_positions.extend(starved_removed_positions);
            removed_positions
        });

        #[cfg(feature = "profiling")]
        profiling::record_tick_total(tick_started.elapsed());

        TickDelta {
            turn: self.turn,
            moves: commit.moves,
            facing_updates: commit.facing_updates,
            removed_positions,
            spawned: spawned.0,
            reproduction_events: spawned.1,
            food_spawned: commit.food_spawned,
            metrics: self.metrics.clone(),
        }
    }

    fn apply_post_commit_runtime_weight_updates(&mut self) {
        if !self.config.runtime_plasticity_enabled {
            return;
        }

        let food_energy = self.config.food_energy;
        for (organism, ledger) in self.organisms.iter().zip(self.reward_ledgers.iter_mut()) {
            ledger.observe(organism, food_energy);
        }

        let any_learners = self
            .organisms
            .iter()
            .any(|o| o.genome.plasticity.hebb_eta_gain > 0.0);

        #[cfg(feature = "profiling")]
        let plasticity_started = Instant::now();

        if any_learners {
            let thread_pool = self.parallel_pool();
            thread_pool.install(|| {
                self.organisms
                    .par_iter_mut()
                    .zip(self.reward_ledgers.par_iter())
                    .with_min_len(INTENT_PARALLEL_MIN_LEN)
                    .for_each(|(organism, reward_ledger)| {
                        apply_runtime_weight_updates(organism, *reward_ledger);
                    });
            });
        } else {
            for (organism, reward_ledger) in
                self.organisms.iter_mut().zip(self.reward_ledgers.iter())
            {
                apply_runtime_weight_updates(organism, *reward_ledger);
            }
        }

        #[cfg(feature = "profiling")]
        profiling::record_brain_plasticity_total(plasticity_started.elapsed());
    }
}

fn organism_health_regeneration(organism: &OrganismState) -> f32 {
    (organism.max_health.max(1.0) * HEALTH_REGEN_FRACTION).max(0.0)
}

fn food_consumption_energy_fraction(kind: FoodKind) -> f32 {
    match kind {
        FoodKind::Plant => PLANT_CONSUMPTION_ENERGY_FRACTION,
        FoodKind::Corpse => CORPSE_CONSUMPTION_ENERGY_FRACTION,
    }
}

fn occupancy_snapshot_cell(
    occupancy: &[Option<Occupant>],
    world_width: i32,
    q: i32,
    r: i32,
) -> Option<Occupant> {
    let (q, r) = wrap_position((q, r), world_width);
    let idx = r as usize * world_width as usize + q as usize;
    occupancy[idx]
}

fn organism_index_by_id(organisms: &[OrganismState], id: OrganismId) -> Option<usize> {
    organisms
        .binary_search_by_key(&id, |organism| organism.id)
        .ok()
}

fn food_index_by_id(foods: &[FoodState], id: sim_types::FoodId) -> Option<usize> {
    foods.binary_search_by_key(&id, |food| food.id).ok()
}

fn reproduction_target(
    world_width: i32,
    parent_q: i32,
    parent_r: i32,
    parent_facing: FacingDirection,
) -> (i32, i32) {
    let opposite_facing = opposite_direction(parent_facing);
    hex_neighbor((parent_q, parent_r), opposite_facing, world_width)
}

fn action_rng_seed(sim_seed: u64, tick: u64, organism_id: OrganismId) -> u64 {
    let mixed =
        sim_seed ^ tick.wrapping_mul(RNG_TURN_MIX) ^ organism_id.0.wrapping_mul(RNG_ORGANISM_MIX);
    mix_u64(mixed)
}

fn deterministic_action_sample(sim_seed: u64, tick: u64, organism_id: OrganismId) -> f32 {
    let sample = (action_rng_seed(sim_seed, tick, organism_id) >> 40) as u32;
    sample as f32 / ((1_u32 << 24) - 1) as f32
}

fn uniform_random_action(action_sample: f32) -> ActionType {
    let scaled = action_sample.clamp(0.0, 1.0 - f32::EPSILON) * ACTION_COUNT as f32;
    let idx = scaled.floor() as usize;
    ActionType::ALL[idx.min(ACTION_COUNT - 1)]
}

fn mix_u64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    value
}
