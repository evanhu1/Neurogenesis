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
use crate::spawn::{ReproductionSpawn, SpawnRequest};
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
    OrganismId, OrganismMove, OrganismState, RemovedEntityPosition, ReproductionEvent, TickDelta,
    VisualProperties,
};
use std::sync::Arc;
#[cfg(feature = "profiling")]
use std::time::Instant;

const RNG_TURN_MIX: u64 = 0x9E37_79B9_7F4A_7C15;
const RNG_ORGANISM_MIX: u64 = 0xBF58_476D_1CE4_E5B9;
const RNG_PREY_MIX: u64 = 0x2545_F491_4F6C_DD1D;
const SPIKE_DAMAGE_FRACTION: f32 = 0.10;
const ATTACK_DAMAGE_FRACTION: f32 = 0.5;
// Eating is lossy: only a fraction of the item's stored energy transfers to
// the eater. Plants are inefficient to digest; corpses much less so. These
// losses are the ecosystem's primary energy sink — without them consumption
// is lossless, predation recycles energy for free, and the population
// equilibrium roughly triples (which is what a 500k-tick evaluation run
// surfaced as a 3-5x wall-clock blowup).
const PLANT_CONSUMPTION_ENERGY_FRACTION: f32 = 0.20;
const CORPSE_CONSUMPTION_ENERGY_FRACTION: f32 = 0.80;
const HEALTH_REGEN_FRACTION: f32 = 0.10;
const INTENT_PARALLEL_MIN_LEN: usize = 64;

#[derive(Debug, Clone, Copy)]
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
    reproduction_events: Vec<ReproductionEvent>,
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
        self.clear_turn_transient_state();

        #[cfg(feature = "profiling")]
        let tick_started = Instant::now();

        let (
            starvations,
            age_deaths,
            lifecycle_removed_positions,
            lifecycle_reproduction_events,
            lifecycle_food_spawned,
        ) = profile_turn_phase!(TurnPhase::Lifecycle, { self.lifecycle_phase() });

        let world_width = self.config.world_width as i32;

        let intents = profile_turn_phase!(TurnPhase::Intents, { self.build_intents(world_width) });
        let synapse_ops = intents.iter().map(|intent| intent.synapse_ops).sum::<u64>();

        let mut reproduction_state = ReproductionPhaseState::new(
            self.organisms.len(),
            std::mem::take(&mut self.turn_scratch.gestation_started),
        );
        reproduction_state.extend_reproduction_events(lifecycle_reproduction_events);
        profile_turn_phase!(TurnPhase::Reproduction, {
            reproduction_state.apply_triggers(
                &mut self.organisms,
                &mut self.pending_actions,
                &intents,
                &self.occupancy,
                world_width,
                #[cfg(feature = "instrumentation")]
                &mut self.action_records,
            );
        });

        let resolutions =
            profile_turn_phase!(TurnPhase::MoveResolution, { self.resolve_moves(&intents) });

        let mut commit = profile_turn_phase!(TurnPhase::Commit, {
            let mut commit = self.commit_phase(
                world_width,
                &intents,
                &resolutions,
                reproduction_state.gestation_started_this_tick_mut(),
            );
            reproduction_state
                .extend_reproduction_events(std::mem::take(&mut commit.reproduction_events));
            reproduction_state.queue_completions(self, world_width);
            commit
        });
        // Old-age corpses spawned during the lifecycle phase ride the same
        // food-spawned channel as commit-phase corpses so the client sees
        // them. Zero-cost on ticks without age deaths.
        commit.food_spawned.extend(lifecycle_food_spawned);

        // Commit was the last reader of the intents and move resolutions;
        // return the buffers to TurnScratch for reuse next tick.
        self.turn_scratch.intents = intents;
        self.turn_scratch.move_resolutions = resolutions;

        profile_turn_phase!(TurnPhase::Age, {
            self.increment_age_for_survivors();
        });

        let (spawned, reproductions) = profile_turn_phase!(TurnPhase::Spawn, {
            let spawn_requests = reproduction_state.spawn_requests_mut();
            let reproduction_spawn_count = spawn_requests.len();
            self.enqueue_periodic_injections(spawn_requests);
            // Outcomes are aligned 1:1 with the request queue (reproductions
            // first, then injections). Reproduction requests target cells that
            // queue_completions verified empty and reserved, and nothing
            // mutates occupancy in between, so they must all succeed; enforce
            // that loudly so child-event attribution can never silently
            // misalign onto injection organisms.
            let spawn_results = self.resolve_spawn_requests(spawn_requests);
            assert!(
                spawn_results[..reproduction_spawn_count]
                    .iter()
                    .all(Option::is_some),
                "reproduction spawn request failed despite its cell being reserved by queue_completions"
            );
            let spawned: Vec<_> = spawn_results.into_iter().flatten().collect();
            let (reproduction_events, gestation_started_scratch) = reproduction_state
                .finalize_reproduction_events(&spawned[..reproduction_spawn_count]);
            self.turn_scratch.gestation_started = gestation_started_scratch;
            self.apply_post_commit_runtime_weight_updates();
            (
                (spawned, reproduction_events),
                reproduction_spawn_count as u64,
            )
        });

        profile_turn_phase!(TurnPhase::ConsistencyCheck, {
            self.debug_assert_consistent_state();
        });

        let removed_positions = profile_turn_phase!(TurnPhase::MetricsAndDelta, {
            self.turn = self.turn.saturating_add(1);
            self.metrics.turns = self.turn;
            self.metrics.synapse_ops_last_turn = synapse_ops;
            // Reproduce trigger ticks already count via intent.took_action;
            // birth completions are not actions (see reproductions_last_turn).
            self.metrics.actions_applied_last_turn = commit.actions_applied;
            self.metrics.consumptions_last_turn = commit.consumptions;
            self.metrics.predations_last_turn = commit.predations;
            self.metrics.total_consumptions += commit.consumptions;
            self.metrics.reproductions_last_turn = reproductions;
            self.metrics.starvations_last_turn = starvations;
            self.metrics.age_deaths_last_turn = age_deaths;
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
            spawned: spawned.0,
            reproduction_events: spawned.1,
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
        // Organisms spawned this tick are skipped: the Age phase has already
        // incremented every survivor, so `age_turns == 0` here identifies
        // exactly the newborns appended by resolve_spawn_requests. Their brains
        // have never been evaluated (intents ran before the spawn), so their
        // eligibility traces carry no pending coactivation yet; they get their
        // first weight update at the end of their first active tick instead.
        //
        // Organisms action-locked by gestation this tick (pending action kind
        // Reproduce) are also skipped entirely: learning pauses during the
        // lock, so plasticity, weight decay, and pruning are all withheld.
        // Their eligibility traces freeze and the first post-unlock update
        // spans the gap — the intended pause/resume semantics. The skip is a
        // pure per-organism predicate, so rayon scheduling cannot affect
        // determinism.
        debug_assert_eq!(self.pending_actions.len(), self.organisms.len());
        // Acquired before the disjoint field borrows below; the pool is
        // already cached from the intent phase, so this is an Arc clone.
        let thread_pool = self.parallel_pool();
        let organisms = &mut self.organisms;
        let pending_actions = self.pending_actions.as_slice();
        if any_learners {
            thread_pool.install(|| {
                organisms
                    .par_iter_mut()
                    .zip(pending_actions.par_iter())
                    .with_min_len(INTENT_PARALLEL_MIN_LEN)
                    .for_each(|(organism, pending_action)| {
                        if organism.age_turns == 0
                            || pending_action.kind == PendingActionKind::Reproduce
                        {
                            return;
                        }
                        apply_runtime_weight_updates(organism, body_mass_metabolic_cost_coeff);
                    });
            });
        } else {
            for (organism, pending_action) in organisms.iter_mut().zip(pending_actions.iter()) {
                if organism.age_turns == 0 || pending_action.kind == PendingActionKind::Reproduce {
                    continue;
                }
                apply_runtime_weight_updates(organism, body_mass_metabolic_cost_coeff);
            }
        }

        #[cfg(feature = "profiling")]
        profiling::record_brain_plasticity_total(plasticity_started.elapsed());
    }
}

fn organism_health_regeneration(organism: &OrganismState) -> f32 {
    (organism.max_health.max(1.0) * HEALTH_REGEN_FRACTION).max(0.0)
}

fn occupancy_snapshot_cell(
    occupancy: &[Option<Occupant>],
    world_width: i32,
    q: i32,
    r: i32,
) -> Option<Occupant> {
    debug_assert_eq!(wrap_position((q, r), world_width), (q, r));
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

fn food_consumption_energy_fraction(kind: FoodKind) -> f32 {
    match kind {
        FoodKind::Plant => PLANT_CONSUMPTION_ENERGY_FRACTION,
        FoodKind::Corpse => CORPSE_CONSUMPTION_ENERGY_FRACTION,
    }
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

fn uniform_random_action(action_sample: f32) -> ActionType {
    // Sample over the same action support as the real policy: the six action
    // neurons in `ActionType::ALL` plus explicit Idle, which
    // `sample_action_from_logits` reaches via its `idle_weight` term.
    const RANDOM_ACTION_BUCKETS: usize = ACTION_COUNT + 1;
    let scaled = action_sample.clamp(0.0, 1.0 - f32::EPSILON) * RANDOM_ACTION_BUCKETS as f32;
    let idx = (scaled.floor() as usize).min(RANDOM_ACTION_BUCKETS - 1);
    ActionType::ALL
        .get(idx)
        .copied()
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
