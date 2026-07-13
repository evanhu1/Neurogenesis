//! Evaluator-owned Stage-0 procedural-ecology mechanics probe.
//!
//! The canonical simulation has no procedural-ecology state or hook. This
//! module wraps ordinary [`Simulation::tick`] calls and accounts every
//! evaluator boundary transfer through an explicit ecology escrow. It is a
//! bounded mechanics falsifier, not an evolutionary algorithm or evidence of
//! open-endedness.

use crate::{
    genome::generate_seed_genome, grid::hex_neighbor, progressive::exact_fingerprint, Simulation,
};
use anyhow::{anyhow, bail, Result};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use sim_types::{
    food_visual, ActionType, EnergyLedgerRow, FacingDirection, FoodId, FoodKind, FoodState,
    Occupant, OrganismGenome, WorldConfig,
};

const EXPERIMENT_DOMAIN: u64 = 0x5052_4f43_4543_4f30;
const WORLD_WIDTH: u32 = 31;
const HOME: (i32, i32) = (8, 15);
const DEFAULT_TRANSLATION: (i32, i32) = (7, 9);
const BASELINE_REPLAY_TICKS: u32 = 64;
const LEDGER_EPSILON_MULTIPLIER: f64 = 64.0;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProceduralEcologyStage0Config {
    pub run_seeds: Vec<u64>,
    pub horizon_ticks: u64,
    pub release_interval_ticks: u64,
    pub plant_energy: f32,
    pub translation: (i32, i32),
}

impl Default for ProceduralEcologyStage0Config {
    fn default() -> Self {
        Self {
            run_seeds: vec![7, 42, 123],
            horizon_ticks: 2_000,
            release_interval_ticks: 100,
            plant_energy: 10.0,
            translation: DEFAULT_TRANSLATION,
        }
    }
}

impl ProceduralEcologyStage0Config {
    fn validate(&self) -> Result<()> {
        if self.run_seeds.is_empty() {
            bail!("procedural ecology Stage 0 needs at least one run seed");
        }
        if self.horizon_ticks == 0 {
            bail!("procedural ecology Stage 0 horizon must be positive");
        }
        if self.release_interval_ticks == 0 {
            bail!("procedural ecology Stage 0 release interval must be positive");
        }
        if !self
            .horizon_ticks
            .is_multiple_of(self.release_interval_ticks)
        {
            bail!("procedural ecology Stage 0 horizon must be divisible by release interval");
        }
        if !self.plant_energy.is_finite() || self.plant_energy <= 0.0 {
            bail!("procedural ecology Stage 0 plant energy must be finite and positive");
        }
        let release_count = self.horizon_ticks / self.release_interval_ticks;
        if release_count >= u64::from(WORLD_WIDTH) {
            bail!(
                "Stage-0 moving front needs fewer than {WORLD_WIDTH} releases to avoid wrapping into the fixed consumer"
            );
        }
        Ok(())
    }

    fn release_schedule(&self) -> Vec<u64> {
        (0..self.horizon_ticks)
            .step_by(self.release_interval_ticks as usize)
            .collect()
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ProceduralEcologyPolicy {
    Stationary,
    MovingFront,
    ConsumptionResponsive,
}

impl ProceduralEcologyPolicy {
    const ALL: [Self; 3] = [
        Self::Stationary,
        Self::MovingFront,
        Self::ConsumptionResponsive,
    ];
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
struct EcologyCellState {
    home: bool,
    carrier: bool,
}

#[derive(Debug, Clone, Copy)]
struct ActiveRelease {
    id: FoodId,
    cell: (i32, i32),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EcologyReleaseEvent {
    pub ordinal: u64,
    pub before_turn: u64,
    pub carrier_before: (i32, i32),
    pub carrier_after: (i32, i32),
    pub release_position: (i32, i32),
    pub released_food_id: FoodId,
    pub release_energy: f64,
    pub raw_consumption_input: bool,
    pub applied_consumption_input: bool,
    pub behavior_inputs_clamped: bool,
    pub reclaimed_food_id: Option<FoodId>,
    pub reclaimed_energy: f64,
    pub escrow_before: f64,
    pub escrow_after_reclaim: f64,
    pub escrow_after_release: f64,
    pub consumed_by_end: bool,
    pub synchronous_translation_equivariance: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct ProceduralEcologyEnergyLedgerRow {
    pub turn: u64,
    pub organism_energy_before: f64,
    pub organism_energy_after: f64,
    pub food_energy_before: f64,
    pub food_energy_after: f64,
    pub artifact_energy_before: f64,
    pub artifact_energy_after: f64,
    pub ecology_energy_before: f64,
    pub ecology_energy_after: f64,
    pub food_reclaim_debit: f64,
    pub ecology_reclaim_credit: f64,
    pub ecology_release_debit: f64,
    pub food_release_credit: f64,
    pub organism_residual: f64,
    pub food_residual: f64,
    pub artifact_residual: f64,
    pub ecology_residual: f64,
    pub reclaim_transfer_residual: f64,
    pub release_transfer_residual: f64,
    pub total_residual: f64,
    pub residual_tolerance: f64,
    pub engine: EnergyLedgerRow,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProceduralEcologyBaselineEvidence {
    pub run_seed: u64,
    pub compared_ticks: u32,
    pub direct_initial_fingerprint: String,
    pub disabled_hook_initial_fingerprint: String,
    pub direct_final_fingerprint: String,
    pub disabled_hook_final_fingerprint: String,
    pub exact: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProceduralEcologyCaseEvidence {
    pub run_seed: u64,
    pub policy: ProceduralEcologyPolicy,
    pub behavior_inputs_clamped: bool,
    pub translated: bool,
    pub home: (i32, i32),
    pub release_events: Vec<EcologyReleaseEvent>,
    pub energy_ledger: Vec<ProceduralEcologyEnergyLedgerRow>,
    pub initial_escrow_energy: f64,
    pub final_escrow_energy: f64,
    pub minimum_escrow_energy: f64,
    pub total_release_energy: f64,
    pub total_reclaimed_energy: f64,
    pub raw_consumption_inputs: u64,
    pub applied_consumption_inputs: u64,
    pub consumed_releases: u64,
    pub all_engine_ledgers_close: bool,
    pub all_combined_ledgers_close: bool,
    /// Coordinate-normalized release positions, physical energy rows, and
    /// consumption outcomes only. Policy labels and input bookkeeping are
    /// deliberately excluded.
    pub normalized_physical_trace_fingerprint: String,
    /// Includes raw/applied policy-input bits for mechanism diagnosis. This is
    /// not a behavioral novelty descriptor.
    pub normalized_diagnostic_trace_fingerprint: String,
    pub initial_world_fingerprint: String,
    pub final_world_fingerprint: String,
    pub case_fingerprint: String,
    pub duplicate_replay_fingerprint: String,
    pub duplicate_replay_exact: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProceduralEcologyStage0Gates {
    pub canonical_disabled_hook_byte_identical: bool,
    pub duplicate_replay_exact: bool,
    pub fixed_release_schedule_count_and_energy: bool,
    pub initial_escrow_matches_full_schedule: bool,
    pub every_release_explicitly_escrow_funded: bool,
    pub boundary_transfer_rows_match_release_events: bool,
    pub no_unaccounted_engine_plant_spawn: bool,
    pub synchronous_translation_equivariance: bool,
    pub translated_traces_match: bool,
    pub stationary_policy_is_stationary: bool,
    pub moving_front_advances: bool,
    pub behavior_input_engaged: bool,
    pub behavior_input_clamp_changes_responsive_trace: bool,
    pub identical_physics_share_physical_fingerprint: bool,
    pub all_organism_ledgers_close: bool,
    pub all_food_ledgers_close: bool,
    pub all_ecology_ledgers_close: bool,
    pub all_transfer_ledgers_close: bool,
    pub all_total_ledgers_close: bool,
    pub all_passed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProceduralEcologyStage0Result {
    pub schema_version: u32,
    pub claim_scope: String,
    pub evaluator_owned: bool,
    pub open_endedness_demonstrated: bool,
    pub stage_1_authorized: bool,
    pub conservation_scope: String,
    pub stage_1_blockers: Vec<String>,
    pub policy_observations: Vec<String>,
    pub limitations: Vec<String>,
    pub config: ProceduralEcologyStage0Config,
    pub effective_world: WorldConfig,
    pub effective_world_fingerprint: String,
    pub release_schedule: Vec<u64>,
    pub baseline: Vec<ProceduralEcologyBaselineEvidence>,
    pub cases: Vec<ProceduralEcologyCaseEvidence>,
    pub gates: ProceduralEcologyStage0Gates,
    pub result_fingerprint: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct NormalizedPhysicalReleaseEvent {
    ordinal: u64,
    before_turn: u64,
    relative_position: (i32, i32),
    consumed_by_end: bool,
    release_energy: f64,
    reclaimed_energy: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct NormalizedPhysicalCaseTrace {
    events: Vec<NormalizedPhysicalReleaseEvent>,
    energy_ledger: Vec<ProceduralEcologyEnergyLedgerRow>,
    final_escrow_energy: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct NormalizedDiagnosticReleaseEvent {
    physical: NormalizedPhysicalReleaseEvent,
    raw_consumption_input: bool,
    applied_consumption_input: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct NormalizedDiagnosticCaseTrace {
    events: Vec<NormalizedDiagnosticReleaseEvent>,
    energy_ledger: Vec<ProceduralEcologyEnergyLedgerRow>,
    final_escrow_energy: f64,
}

#[derive(Debug, Clone, Copy, Default)]
struct BoundaryFlow {
    food_reclaim_debit: f64,
    ecology_reclaim_credit: f64,
    ecology_release_debit: f64,
    food_release_credit: f64,
}

struct EcologyHarness {
    policy: ProceduralEcologyPolicy,
    behavior_inputs_clamped: bool,
    width: i32,
    state: Vec<EcologyCellState>,
    active_release: Option<ActiveRelease>,
    escrow_energy: f64,
    minimum_escrow_energy: f64,
    translation_probe: (i32, i32),
}

/// Run the bounded Stage-0 mechanics experiment. No returned score enters any
/// evolutionary selection loop.
pub fn run_procedural_ecology_stage0(
    base_world: WorldConfig,
    config: ProceduralEcologyStage0Config,
) -> Result<ProceduralEcologyStage0Result> {
    config.validate()?;
    let effective_world = mechanism_world(base_world.clone(), &config);
    let effective_world_fingerprint = fingerprint(&effective_world)?;
    let release_schedule = config.release_schedule();
    let baseline = config
        .run_seeds
        .iter()
        .copied()
        .map(|seed| baseline_evidence(&base_world, seed))
        .collect::<Result<Vec<_>>>()?;

    let mut cases = Vec::new();
    for &run_seed in &config.run_seeds {
        for policy in ProceduralEcologyPolicy::ALL {
            for translated in [false, true] {
                cases.push(run_case_with_replay(
                    &effective_world,
                    &config,
                    run_seed,
                    policy,
                    false,
                    translated,
                )?);
            }
        }
        for translated in [false, true] {
            cases.push(run_case_with_replay(
                &effective_world,
                &config,
                run_seed,
                ProceduralEcologyPolicy::ConsumptionResponsive,
                true,
                translated,
            )?);
        }
    }

    let gates = build_gates(&config, &release_schedule, &baseline, &cases)?;
    let initial_escrow_energy = release_schedule.len() as f64 * f64::from(config.plant_energy);
    let mut result = ProceduralEcologyStage0Result {
        schema_version: 1,
        claim_scope: "evaluator-owned Stage-0 procedural-ecology mechanics; not an evolutionary or open-endedness result".to_owned(),
        evaluator_owned: true,
        open_endedness_demonstrated: false,
        stage_1_authorized: false,
        conservation_scope: "within-episode closure after an evaluator-preloaded fixed ecology escrow; escrow initialization is outside the recorded transfer ledger".to_owned(),
        stage_1_blockers: vec![
            format!("The {initial_escrow_energy}-energy per-case initial escrow is evaluator-preloaded and has no source-compartment debit row; only subsequent within-episode transfers close."),
            "The diagnostic trace includes applied/raw policy-input labels and is not a behavior descriptor. Stage 0 now publishes a separate physical fingerprint, but no organism behavior descriptor has been validated.".to_owned(),
            "A scheduled boundary can reclaim and release in the same wrapper call; the combined row closes the net transfers but does not serialize intermediate food/ecology physical checkpoints.".to_owned(),
            "A carrier landing on an occupied cell aborts the case instead of applying a deterministic defer/fallback rule suitable for evolved organisms.".to_owned(),
            "The disabled-hook comparison invokes an intentionally empty function and is therefore an internal no-op regression check, not independent proof of canonical equivalence.".to_owned(),
            "Consumption input is inferred through private Simulation::foods membership rather than a public TickDelta removal fact.".to_owned(),
        ],
        policy_observations: vec![
            "The local program observes only its translated carrier/home cell state and whether the previously released public FoodId remains in Simulation::foods.".to_owned(),
            "It receives no organism ID, absolute-coordinate input, objective value, success label, or evaluator reward.".to_owned(),
        ],
        limitations: vec![
            "The three policies are hand-authored and never evolved.".to_owned(),
            "The procedural state is evaluator-owned and lasts for one episode; no ecology genome, mutation, archive, transfer, or selection exists.".to_owned(),
            "The consumption-responsive rule is a one-bit two-location mechanism, so its causal engagement is deliberately bounded.".to_owned(),
            "The fixed eater is an evaluator fixture, not evidence that an evolved organism can exploit a procedural ecology.".to_owned(),
            "The exact fixture intentionally removes seed-conditioned action variation, so equal traces across the three run seeds are replay evidence, not a seed-robust behavioral result.".to_owned(),
            "The moving-front probe is restricted to fewer releases than the world width and does not test wraparound or indefinite operation.".to_owned(),
            "All Stage-0 mechanics gates may pass while Stage 1 remains explicitly unauthorized until the serialized blockers are repaired and adversarially re-audited.".to_owned(),
        ],
        config,
        effective_world,
        effective_world_fingerprint,
        release_schedule,
        baseline,
        cases,
        gates,
        result_fingerprint: String::new(),
    };
    result.result_fingerprint = fingerprint(&result)?;
    Ok(result)
}

fn run_case_with_replay(
    world: &WorldConfig,
    config: &ProceduralEcologyStage0Config,
    run_seed: u64,
    policy: ProceduralEcologyPolicy,
    behavior_inputs_clamped: bool,
    translated: bool,
) -> Result<ProceduralEcologyCaseEvidence> {
    let mut primary = run_case(
        world,
        config,
        run_seed,
        policy,
        behavior_inputs_clamped,
        translated,
    )?;
    let replay = run_case(
        world,
        config,
        run_seed,
        policy,
        behavior_inputs_clamped,
        translated,
    )?;
    primary.duplicate_replay_fingerprint = replay.case_fingerprint.clone();
    primary.duplicate_replay_exact = primary.case_fingerprint == replay.case_fingerprint;
    Ok(primary)
}

fn run_case(
    world: &WorldConfig,
    config: &ProceduralEcologyStage0Config,
    run_seed: u64,
    policy: ProceduralEcologyPolicy,
    behavior_inputs_clamped: bool,
    translated: bool,
) -> Result<ProceduralEcologyCaseEvidence> {
    let genome = eater_genome(world, run_seed);
    let mut sim = Simulation::new_with_champion_pool(world.clone(), run_seed, vec![genome])
        .map_err(|error| anyhow!(error.to_string()))?;
    let home = if translated {
        translate_position(HOME, config.translation, WORLD_WIDTH as i32)
    } else {
        HOME
    };
    prepare_scene(&mut sim, home)?;
    let initial_world_fingerprint = fingerprint(&sim)?;
    let release_count = config.horizon_ticks / config.release_interval_ticks;
    let initial_escrow_energy = release_count as f64 * f64::from(config.plant_energy);
    let mut harness = EcologyHarness::new(
        policy,
        behavior_inputs_clamped,
        WORLD_WIDTH as i32,
        home,
        initial_escrow_energy,
        config.translation,
    );
    let mut release_events = Vec::with_capacity(release_count as usize);
    let mut energy_ledger = Vec::with_capacity(config.horizon_ticks as usize);

    while sim.turn() < config.horizon_ticks {
        let organism_before = organism_energy(&sim);
        let food_before = food_energy(&sim);
        let artifact_before = artifact_energy(&sim);
        let ecology_before = harness.escrow_energy;
        let mut boundary = BoundaryFlow::default();
        if sim.turn().is_multiple_of(config.release_interval_ticks) {
            let event = harness.release_boundary(
                &mut sim,
                release_events.len() as u64,
                config.plant_energy,
                &mut boundary,
            )?;
            release_events.push(event);
        }
        let delta = sim.tick();
        if let Some(active) = harness.active_release {
            if !food_exists(&sim, active.id) {
                let event = release_events
                    .last_mut()
                    .ok_or_else(|| anyhow!("active ecology release has no evidence row"))?;
                if event.released_food_id != active.id {
                    bail!("active ecology FoodId does not match latest release evidence");
                }
                event.consumed_by_end = true;
            }
        }
        let row = build_combined_ledger(
            organism_before,
            food_before,
            artifact_before,
            ecology_before,
            harness.escrow_energy,
            boundary,
            delta.metrics.energy_ledger_last_turn,
        );
        energy_ledger.push(row);
    }

    let final_world_fingerprint = fingerprint(&sim)?;
    let normalized_physical_trace = normalized_physical_trace(
        home,
        WORLD_WIDTH as i32,
        &release_events,
        &energy_ledger,
        harness.escrow_energy,
    );
    let normalized_diagnostic_trace = normalized_diagnostic_trace(
        home,
        WORLD_WIDTH as i32,
        &release_events,
        &energy_ledger,
        harness.escrow_energy,
    );
    let normalized_physical_trace_fingerprint = fingerprint(&normalized_physical_trace)?;
    let normalized_diagnostic_trace_fingerprint = fingerprint(&normalized_diagnostic_trace)?;
    let all_engine_ledgers_close = energy_ledger
        .iter()
        .all(|row| engine_ledger_closes(row.engine));
    let all_combined_ledgers_close = energy_ledger.iter().all(combined_ledger_closes);
    let mut evidence = ProceduralEcologyCaseEvidence {
        run_seed,
        policy,
        behavior_inputs_clamped,
        translated,
        home,
        release_events,
        energy_ledger,
        initial_escrow_energy,
        final_escrow_energy: harness.escrow_energy,
        minimum_escrow_energy: harness.minimum_escrow_energy,
        total_release_energy: release_count as f64 * f64::from(config.plant_energy),
        total_reclaimed_energy: 0.0,
        raw_consumption_inputs: 0,
        applied_consumption_inputs: 0,
        consumed_releases: 0,
        all_engine_ledgers_close,
        all_combined_ledgers_close,
        normalized_physical_trace_fingerprint,
        normalized_diagnostic_trace_fingerprint,
        initial_world_fingerprint,
        final_world_fingerprint,
        case_fingerprint: String::new(),
        duplicate_replay_fingerprint: String::new(),
        duplicate_replay_exact: false,
    };
    evidence.total_reclaimed_energy = evidence
        .release_events
        .iter()
        .map(|event| event.reclaimed_energy)
        .sum();
    evidence.raw_consumption_inputs = evidence
        .release_events
        .iter()
        .filter(|event| event.raw_consumption_input)
        .count() as u64;
    evidence.applied_consumption_inputs = evidence
        .release_events
        .iter()
        .filter(|event| event.applied_consumption_input)
        .count() as u64;
    evidence.consumed_releases = evidence
        .release_events
        .iter()
        .filter(|event| event.consumed_by_end)
        .count() as u64;
    evidence.case_fingerprint = fingerprint(&evidence)?;
    Ok(evidence)
}

impl EcologyHarness {
    fn new(
        policy: ProceduralEcologyPolicy,
        behavior_inputs_clamped: bool,
        width: i32,
        home: (i32, i32),
        escrow_energy: f64,
        translation_probe: (i32, i32),
    ) -> Self {
        let mut state = vec![EcologyCellState::default(); width as usize * width as usize];
        let home_idx = cell_index(home, width);
        state[home_idx] = EcologyCellState {
            home: true,
            carrier: true,
        };
        Self {
            policy,
            behavior_inputs_clamped,
            width,
            state,
            active_release: None,
            escrow_energy,
            minimum_escrow_energy: escrow_energy,
            translation_probe,
        }
    }

    fn release_boundary(
        &mut self,
        sim: &mut Simulation,
        ordinal: u64,
        plant_energy: f32,
        flow: &mut BoundaryFlow,
    ) -> Result<EcologyReleaseEvent> {
        let escrow_before = self.escrow_energy;
        let carrier_before = carrier_position(&self.state, self.width)?;
        let mut raw_input = vec![false; self.state.len()];
        let mut reclaimed_food_id = None;
        let mut reclaimed_energy = 0.0;
        if let Some(active) = self.active_release.take() {
            if let Ok(food_idx) = sim.foods.binary_search_by_key(&active.id, |food| food.id) {
                let food = sim.foods.remove(food_idx);
                let idx = sim.cell_index(food.q, food.r);
                if sim.occupancy[idx] != Some(Occupant::Food(food.id)) {
                    bail!("procedural ecology reclaim found inconsistent occupancy");
                }
                sim.occupancy[idx] = None;
                reclaimed_food_id = Some(food.id);
                reclaimed_energy = f64::from(food.energy);
                flow.food_reclaim_debit = reclaimed_energy;
                flow.ecology_reclaim_credit = reclaimed_energy;
                self.escrow_energy += reclaimed_energy;
            } else {
                raw_input[cell_index(active.cell, self.width)] = true;
            }
        }
        let escrow_after_reclaim = self.escrow_energy;
        let applied_input = if self.behavior_inputs_clamped {
            vec![false; raw_input.len()]
        } else {
            raw_input.clone()
        };

        let synchronous_translation_equivariance = if ordinal == 0 {
            true
        } else {
            update_commutes_with_translation(
                self.policy,
                self.width,
                &self.state,
                &applied_input,
                self.translation_probe,
            )?
        };
        if ordinal > 0 {
            self.state = synchronous_update(self.policy, self.width, &self.state, &applied_input)?;
        }
        let carrier_after = carrier_position(&self.state, self.width)?;
        let release_energy = f64::from(plant_energy);
        if self.escrow_energy + f64::EPSILON < release_energy {
            bail!(
                "procedural ecology escrow underflow: have {}, need {release_energy}",
                self.escrow_energy
            );
        }
        if sim.terrain_map[sim.cell_index(carrier_after.0, carrier_after.1)] {
            bail!("procedural ecology attempted to release onto terrain");
        }
        if sim.occupancy[sim.cell_index(carrier_after.0, carrier_after.1)].is_some() {
            bail!("procedural ecology attempted to release onto an occupied cell");
        }
        self.escrow_energy -= release_energy;
        self.minimum_escrow_energy = self.minimum_escrow_energy.min(self.escrow_energy);
        flow.ecology_release_debit = release_energy;
        flow.food_release_credit = release_energy;
        let id = FoodId(sim.next_food_id);
        sim.next_food_id = sim.next_food_id.saturating_add(1);
        let food = FoodState {
            id,
            q: carrier_after.0,
            r: carrier_after.1,
            energy: plant_energy,
            kind: FoodKind::Plant,
            visual: food_visual(FoodKind::Plant),
        };
        let idx = sim.cell_index(food.q, food.r);
        sim.occupancy[idx] = Some(Occupant::Food(id));
        sim.foods.push(food);
        self.active_release = Some(ActiveRelease {
            id,
            cell: carrier_after,
        });
        sim.debug_assert_consistent_state();

        Ok(EcologyReleaseEvent {
            ordinal,
            before_turn: sim.turn(),
            carrier_before,
            carrier_after,
            release_position: carrier_after,
            released_food_id: id,
            release_energy,
            raw_consumption_input: raw_input.iter().any(|value| *value),
            applied_consumption_input: applied_input.iter().any(|value| *value),
            behavior_inputs_clamped: self.behavior_inputs_clamped,
            reclaimed_food_id,
            reclaimed_energy,
            escrow_before,
            escrow_after_reclaim,
            escrow_after_release: self.escrow_energy,
            consumed_by_end: false,
            synchronous_translation_equivariance,
        })
    }
}

fn synchronous_update(
    policy: ProceduralEcologyPolicy,
    width: i32,
    state: &[EcologyCellState],
    consumption_input: &[bool],
) -> Result<Vec<EcologyCellState>> {
    if state.len() != consumption_input.len() || state.len() != width as usize * width as usize {
        bail!("procedural ecology state/input shape mismatch");
    }
    let mut next = state
        .iter()
        .map(|cell| EcologyCellState {
            home: cell.home,
            carrier: false,
        })
        .collect::<Vec<_>>();
    for (idx, cell) in state.iter().copied().enumerate() {
        if !cell.carrier {
            continue;
        }
        let position = position_from_index(idx, width);
        let direction = match policy {
            ProceduralEcologyPolicy::Stationary => None,
            ProceduralEcologyPolicy::MovingFront => Some(FacingDirection::East),
            ProceduralEcologyPolicy::ConsumptionResponsive => {
                if consumption_input[idx] {
                    Some(FacingDirection::East)
                } else if cell.home {
                    None
                } else {
                    Some(FacingDirection::West)
                }
            }
        };
        let destination = direction.map_or(position, |direction| {
            hex_neighbor(position, direction, width)
        });
        let destination_idx = cell_index(destination, width);
        if next[destination_idx].carrier {
            bail!("procedural ecology synchronous update produced a carrier collision");
        }
        next[destination_idx].carrier = true;
    }
    if next.iter().filter(|cell| cell.carrier).count() != 1
        || next.iter().filter(|cell| cell.home).count() != 1
    {
        bail!("procedural ecology must preserve exactly one home and one carrier");
    }
    Ok(next)
}

fn update_commutes_with_translation(
    policy: ProceduralEcologyPolicy,
    width: i32,
    state: &[EcologyCellState],
    input: &[bool],
    delta: (i32, i32),
) -> Result<bool> {
    let updated = synchronous_update(policy, width, state, input)?;
    let translated_state = translate_cells(state, width, delta);
    let translated_input = translate_bools(input, width, delta);
    let translated_updated =
        synchronous_update(policy, width, &translated_state, &translated_input)?;
    Ok(translate_cells(&updated, width, delta) == translated_updated)
}

fn translate_cells(
    values: &[EcologyCellState],
    width: i32,
    delta: (i32, i32),
) -> Vec<EcologyCellState> {
    let mut translated = vec![EcologyCellState::default(); values.len()];
    for (idx, value) in values.iter().copied().enumerate() {
        let position = position_from_index(idx, width);
        let target = translate_position(position, delta, width);
        translated[cell_index(target, width)] = value;
    }
    translated
}

fn translate_bools(values: &[bool], width: i32, delta: (i32, i32)) -> Vec<bool> {
    let mut translated = vec![false; values.len()];
    for (idx, value) in values.iter().copied().enumerate() {
        let position = position_from_index(idx, width);
        let target = translate_position(position, delta, width);
        translated[cell_index(target, width)] = value;
    }
    translated
}

fn build_combined_ledger(
    organism_before: f64,
    food_before: f64,
    artifact_before: f64,
    ecology_before: f64,
    ecology_after: f64,
    boundary: BoundaryFlow,
    engine: EnergyLedgerRow,
) -> ProceduralEcologyEnergyLedgerRow {
    let organism_after = engine.organism_energy_after;
    let food_after = engine.food_energy_after;
    let artifact_after = engine.artifact_energy_after;
    let organism_expected = organism_before
        - engine.passive_metabolism_energy
        - engine.action_cost_energy
        - engine.protocol_interaction_cost_energy
        + engine.food_consumption_credit
        + engine.artifact_release_credit
        + engine.predation_energy_credit
        - engine.unrecycled_energy_removed
        - engine.predation_prey_energy_removed
        - engine.corpse_source_energy_removed;
    let food_expected = food_before - boundary.food_reclaim_debit + boundary.food_release_credit
        - engine.food_consumption_debit
        + engine.plant_spawn_energy
        + engine.corpse_spawn_energy;
    let artifact_expected =
        artifact_before + engine.food_to_artifact_credit - engine.artifact_release_debit;
    let ecology_expected =
        ecology_before + boundary.ecology_reclaim_credit - boundary.ecology_release_debit;
    let organism_residual = organism_after - organism_expected;
    let food_residual = food_after - food_expected;
    let artifact_residual = artifact_after - artifact_expected;
    let ecology_residual = ecology_after - ecology_expected;
    let reclaim_transfer_residual = boundary.ecology_reclaim_credit - boundary.food_reclaim_debit;
    let release_transfer_residual = boundary.food_release_credit - boundary.ecology_release_debit;
    let total_expected = organism_before
        + food_before
        + artifact_before
        + ecology_before
        + engine.plant_spawn_energy
        - engine.passive_metabolism_energy
        - engine.action_cost_energy
        - engine.protocol_interaction_cost_energy
        - engine.predation_retention_loss
        - engine.corpse_retention_loss
        - engine.artifact_release_loss
        + engine.removal_adjustment;
    let total_after = organism_after + food_after + artifact_after + ecology_after;
    let total_residual = total_after - total_expected;
    let flow_scale = organism_before.abs()
        + food_before.abs()
        + artifact_before.abs()
        + ecology_before.abs()
        + boundary.food_reclaim_debit.abs()
        + boundary.ecology_reclaim_credit.abs()
        + boundary.ecology_release_debit.abs()
        + boundary.food_release_credit.abs()
        + engine.passive_metabolism_energy.abs()
        + engine.action_cost_energy.abs()
        + engine.food_consumption_debit.abs()
        + engine.food_consumption_credit.abs()
        + engine.plant_spawn_energy.abs()
        + engine.corpse_spawn_energy.abs();
    let residual_tolerance =
        LEDGER_EPSILON_MULTIPLIER * f64::from(f32::EPSILON) * flow_scale.max(1.0);
    ProceduralEcologyEnergyLedgerRow {
        turn: engine.turn,
        organism_energy_before: organism_before,
        organism_energy_after: organism_after,
        food_energy_before: food_before,
        food_energy_after: food_after,
        artifact_energy_before: artifact_before,
        artifact_energy_after: artifact_after,
        ecology_energy_before: ecology_before,
        ecology_energy_after: ecology_after,
        food_reclaim_debit: boundary.food_reclaim_debit,
        ecology_reclaim_credit: boundary.ecology_reclaim_credit,
        ecology_release_debit: boundary.ecology_release_debit,
        food_release_credit: boundary.food_release_credit,
        organism_residual,
        food_residual,
        artifact_residual,
        ecology_residual,
        reclaim_transfer_residual,
        release_transfer_residual,
        total_residual,
        residual_tolerance,
        engine,
    }
}

fn mechanism_world(mut world: WorldConfig, config: &ProceduralEcologyStage0Config) -> WorldConfig {
    world.world_width = WORLD_WIDTH;
    world.num_organisms = 1;
    world.food_energy = config.plant_energy;
    world.passive_metabolism_cost_per_unit = 0.0;
    world.body_mass_metabolic_cost_coeff = 0.0;
    world.move_action_energy_cost = 0.0;
    world.action_temperature = 0.01;
    world.intent_parallel_threads = 1;
    world.food_regrowth_interval = u32::MAX;
    world.food_regrowth_jitter = 0;
    world.food_tile_fraction = 0.0;
    world.terrain_threshold = 1.0;
    world.runtime_plasticity_enabled = false;
    world.leaky_neurons_enabled = false;
    world.predation_enabled = false;
    world.force_random_actions = false;
    world.protocol_cache_enabled = false;
    world.cache_energy_fraction = 0.0;
    world.protocol_interaction_energy_cost = 0.0;
    world.seed_genome_config.num_neurons = 0;
    world.seed_genome_config.num_synapses = 0;
    world.seed_genome_config.hebb_eta_gain = 0.0;
    world
}

fn eater_genome(world: &WorldConfig, seed: u64) -> OrganismGenome {
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ EXPERIMENT_DOMAIN);
    let mut genome = generate_seed_genome(&world.seed_genome_config, false, &mut rng);
    genome.brain.action_biases.fill(-1.0);
    let eat_index = ActionType::ALL
        .iter()
        .position(|action| *action == ActionType::Eat)
        .expect("Eat is a stable active action");
    genome.brain.action_biases[eat_index] = 1.0;
    genome
}

fn prepare_scene(sim: &mut Simulation, home: (i32, i32)) -> Result<()> {
    if sim.organisms.len() != 1 || sim.terrain_map.iter().any(|blocked| *blocked) {
        bail!("procedural ecology Stage 0 needs one founder and an obstacle-free arena");
    }
    sim.foods.clear();
    sim.food_tiles.fill(false);
    sim.food_regrowth_due_turn.fill(u64::MAX);
    sim.food_regrowth_schedule.clear();
    sim.artifact_caches.clear();
    sim.pending_artifact_interactions.clear();
    sim.artifact_events_last_turn.clear();
    sim.occupancy.fill(None);
    let consumer_position = hex_neighbor(home, FacingDirection::West, WORLD_WIDTH as i32);
    let organism = &mut sim.organisms[0];
    organism.q = consumer_position.0;
    organism.r = consumer_position.1;
    organism.facing = FacingDirection::East;
    organism.energy_at_last_sensing = organism.energy;
    organism.last_action_taken = ActionType::Idle;
    let organism_id = organism.id;
    let consumer_idx = sim.cell_index(consumer_position.0, consumer_position.1);
    sim.occupancy[consumer_idx] = Some(Occupant::Organism(organism_id));
    sim.debug_assert_consistent_state();
    Ok(())
}

fn baseline_evidence(
    world: &WorldConfig,
    run_seed: u64,
) -> Result<ProceduralEcologyBaselineEvidence> {
    let mut direct =
        Simulation::new(world.clone(), run_seed).map_err(|error| anyhow!(error.to_string()))?;
    let mut disabled =
        Simulation::new(world.clone(), run_seed).map_err(|error| anyhow!(error.to_string()))?;
    let direct_initial_fingerprint = fingerprint(&direct)?;
    disabled_ecology_hook(&mut disabled);
    let disabled_hook_initial_fingerprint = fingerprint(&disabled)?;
    for _ in 0..BASELINE_REPLAY_TICKS {
        let _ = direct.tick();
        disabled_ecology_hook(&mut disabled);
        let _ = disabled.tick();
    }
    let direct_final_fingerprint = fingerprint(&direct)?;
    let disabled_hook_final_fingerprint = fingerprint(&disabled)?;
    let exact = direct_initial_fingerprint == disabled_hook_initial_fingerprint
        && direct_final_fingerprint == disabled_hook_final_fingerprint;
    Ok(ProceduralEcologyBaselineEvidence {
        run_seed,
        compared_ticks: BASELINE_REPLAY_TICKS,
        direct_initial_fingerprint,
        disabled_hook_initial_fingerprint,
        direct_final_fingerprint,
        disabled_hook_final_fingerprint,
        exact,
    })
}

#[inline]
fn disabled_ecology_hook(_sim: &mut Simulation) {}

fn build_gates(
    config: &ProceduralEcologyStage0Config,
    release_schedule: &[u64],
    baseline: &[ProceduralEcologyBaselineEvidence],
    cases: &[ProceduralEcologyCaseEvidence],
) -> Result<ProceduralEcologyStage0Gates> {
    let canonical_disabled_hook_byte_identical = baseline.iter().all(|entry| entry.exact);
    let duplicate_replay_exact = cases.iter().all(|case| case.duplicate_replay_exact);
    let expected_release_energy = f64::from(config.plant_energy);
    let fixed_release_schedule_count_and_energy = cases.iter().all(|case| {
        case.release_events.len() == release_schedule.len()
            && case
                .release_events
                .iter()
                .map(|event| event.before_turn)
                .eq(release_schedule.iter().copied())
            && case
                .release_events
                .iter()
                .all(|event| event.release_energy.to_bits() == expected_release_energy.to_bits())
    });
    let expected_total_release_energy = release_schedule.len() as f64 * expected_release_energy;
    let initial_escrow_matches_full_schedule = cases.iter().all(|case| {
        case.initial_escrow_energy.to_bits() == expected_total_release_energy.to_bits()
            && case.total_release_energy.to_bits() == expected_total_release_energy.to_bits()
    });
    let every_release_explicitly_escrow_funded = cases.iter().all(|case| {
        case.minimum_escrow_energy >= 0.0
            && case.release_events.iter().all(|event| {
                event.escrow_after_reclaim + f64::EPSILON >= event.release_energy
                    && event.escrow_after_release
                        == event.escrow_after_reclaim - event.release_energy
            })
    });
    let boundary_transfer_rows_match_release_events = cases.iter().all(|case| {
        let event_release = case
            .release_events
            .iter()
            .map(|event| event.release_energy)
            .sum::<f64>();
        let event_reclaim = case
            .release_events
            .iter()
            .map(|event| event.reclaimed_energy)
            .sum::<f64>();
        let row_release_debit = case
            .energy_ledger
            .iter()
            .map(|row| row.ecology_release_debit)
            .sum::<f64>();
        let row_release_credit = case
            .energy_ledger
            .iter()
            .map(|row| row.food_release_credit)
            .sum::<f64>();
        let row_reclaim_debit = case
            .energy_ledger
            .iter()
            .map(|row| row.food_reclaim_debit)
            .sum::<f64>();
        let row_reclaim_credit = case
            .energy_ledger
            .iter()
            .map(|row| row.ecology_reclaim_credit)
            .sum::<f64>();
        event_release.to_bits() == row_release_debit.to_bits()
            && event_release.to_bits() == row_release_credit.to_bits()
            && event_reclaim.to_bits() == row_reclaim_debit.to_bits()
            && event_reclaim.to_bits() == row_reclaim_credit.to_bits()
            && event_reclaim.to_bits() == case.total_reclaimed_energy.to_bits()
    });
    let no_unaccounted_engine_plant_spawn = cases.iter().all(|case| {
        case.energy_ledger
            .iter()
            .all(|row| row.engine.plant_spawn_energy == 0.0)
    });
    let synchronous_translation_equivariance = cases.iter().all(|case| {
        case.release_events
            .iter()
            .all(|event| event.synchronous_translation_equivariance)
    });
    let translated_traces_match = config.run_seeds.iter().copied().all(|seed| {
        ProceduralEcologyPolicy::ALL.into_iter().all(|policy| {
            translated_pair_matches(cases, seed, policy, false)
                && (policy != ProceduralEcologyPolicy::ConsumptionResponsive
                    || translated_pair_matches(cases, seed, policy, true))
        })
    });
    let stationary_policy_is_stationary = cases
        .iter()
        .filter(|case| {
            case.policy == ProceduralEcologyPolicy::Stationary && !case.behavior_inputs_clamped
        })
        .all(|case| {
            case.release_events
                .iter()
                .all(|event| event.release_position == case.home)
        });
    let moving_front_advances = cases
        .iter()
        .filter(|case| {
            case.policy == ProceduralEcologyPolicy::MovingFront && !case.behavior_inputs_clamped
        })
        .all(|case| {
            let mut expected = case.home;
            case.release_events.iter().all(|event| {
                let matches = event.release_position == expected;
                expected = hex_neighbor(expected, FacingDirection::East, WORLD_WIDTH as i32);
                matches
            })
        });
    let behavior_input_engaged = config.run_seeds.iter().copied().all(|seed| {
        find_case(
            cases,
            seed,
            ProceduralEcologyPolicy::ConsumptionResponsive,
            false,
            false,
        )
        .is_some_and(|case| case.raw_consumption_inputs > 0 && case.applied_consumption_inputs > 0)
    });
    let behavior_input_clamp_changes_responsive_trace =
        config.run_seeds.iter().copied().all(|seed| {
            let Some(treatment) = find_case(
                cases,
                seed,
                ProceduralEcologyPolicy::ConsumptionResponsive,
                false,
                false,
            ) else {
                return false;
            };
            let Some(clamped) = find_case(
                cases,
                seed,
                ProceduralEcologyPolicy::ConsumptionResponsive,
                true,
                false,
            ) else {
                return false;
            };
            clamped.raw_consumption_inputs > 0
                && clamped.applied_consumption_inputs == 0
                && treatment.normalized_physical_trace_fingerprint
                    != clamped.normalized_physical_trace_fingerprint
                && treatment
                    .release_events
                    .iter()
                    .map(|event| event.release_position)
                    .ne(clamped
                        .release_events
                        .iter()
                        .map(|event| event.release_position))
        });
    let identical_physics_share_physical_fingerprint =
        config.run_seeds.iter().copied().all(|seed| {
            let Some(stationary) = find_case(
                cases,
                seed,
                ProceduralEcologyPolicy::Stationary,
                false,
                false,
            ) else {
                return false;
            };
            let Some(clamped) = find_case(
                cases,
                seed,
                ProceduralEcologyPolicy::ConsumptionResponsive,
                true,
                false,
            ) else {
                return false;
            };
            stationary.normalized_physical_trace_fingerprint
                == clamped.normalized_physical_trace_fingerprint
                && stationary.normalized_diagnostic_trace_fingerprint
                    != clamped.normalized_diagnostic_trace_fingerprint
        });
    let all_organism_ledgers_close = cases
        .iter()
        .flat_map(|case| &case.energy_ledger)
        .all(|row| residual_closes(row.organism_residual, row.residual_tolerance));
    let all_food_ledgers_close = cases
        .iter()
        .flat_map(|case| &case.energy_ledger)
        .all(|row| {
            residual_closes(row.food_residual, row.residual_tolerance)
                && residual_closes(row.artifact_residual, row.residual_tolerance)
        });
    let all_ecology_ledgers_close = cases
        .iter()
        .flat_map(|case| &case.energy_ledger)
        .all(|row| residual_closes(row.ecology_residual, row.residual_tolerance));
    let all_transfer_ledgers_close = cases
        .iter()
        .flat_map(|case| &case.energy_ledger)
        .all(|row| {
            residual_closes(row.reclaim_transfer_residual, row.residual_tolerance)
                && residual_closes(row.release_transfer_residual, row.residual_tolerance)
                && row.all_engine_transfers_close()
        });
    let all_total_ledgers_close = cases
        .iter()
        .flat_map(|case| &case.energy_ledger)
        .all(|row| {
            residual_closes(row.total_residual, row.residual_tolerance)
                && engine_ledger_closes(row.engine)
        });
    let mut gates = ProceduralEcologyStage0Gates {
        canonical_disabled_hook_byte_identical,
        duplicate_replay_exact,
        fixed_release_schedule_count_and_energy,
        initial_escrow_matches_full_schedule,
        every_release_explicitly_escrow_funded,
        boundary_transfer_rows_match_release_events,
        no_unaccounted_engine_plant_spawn,
        synchronous_translation_equivariance,
        translated_traces_match,
        stationary_policy_is_stationary,
        moving_front_advances,
        behavior_input_engaged,
        behavior_input_clamp_changes_responsive_trace,
        identical_physics_share_physical_fingerprint,
        all_organism_ledgers_close,
        all_food_ledgers_close,
        all_ecology_ledgers_close,
        all_transfer_ledgers_close,
        all_total_ledgers_close,
        all_passed: false,
    };
    gates.all_passed = gates.canonical_disabled_hook_byte_identical
        && gates.duplicate_replay_exact
        && gates.fixed_release_schedule_count_and_energy
        && gates.initial_escrow_matches_full_schedule
        && gates.every_release_explicitly_escrow_funded
        && gates.boundary_transfer_rows_match_release_events
        && gates.no_unaccounted_engine_plant_spawn
        && gates.synchronous_translation_equivariance
        && gates.translated_traces_match
        && gates.stationary_policy_is_stationary
        && gates.moving_front_advances
        && gates.behavior_input_engaged
        && gates.behavior_input_clamp_changes_responsive_trace
        && gates.identical_physics_share_physical_fingerprint
        && gates.all_organism_ledgers_close
        && gates.all_food_ledgers_close
        && gates.all_ecology_ledgers_close
        && gates.all_transfer_ledgers_close
        && gates.all_total_ledgers_close;
    Ok(gates)
}

impl ProceduralEcologyEnergyLedgerRow {
    fn all_engine_transfers_close(self) -> bool {
        residual_closes(
            self.engine.food_split_transfer_residual,
            self.engine.residual_tolerance,
        ) && residual_closes(
            self.engine.artifact_release_transfer_residual,
            self.engine.residual_tolerance,
        )
    }
}

fn translated_pair_matches(
    cases: &[ProceduralEcologyCaseEvidence],
    seed: u64,
    policy: ProceduralEcologyPolicy,
    clamped: bool,
) -> bool {
    let Some(canonical) = find_case(cases, seed, policy, clamped, false) else {
        return false;
    };
    let Some(translated) = find_case(cases, seed, policy, clamped, true) else {
        return false;
    };
    canonical.normalized_physical_trace_fingerprint
        == translated.normalized_physical_trace_fingerprint
        && canonical.normalized_diagnostic_trace_fingerprint
            == translated.normalized_diagnostic_trace_fingerprint
}

fn find_case(
    cases: &[ProceduralEcologyCaseEvidence],
    seed: u64,
    policy: ProceduralEcologyPolicy,
    clamped: bool,
    translated: bool,
) -> Option<&ProceduralEcologyCaseEvidence> {
    cases.iter().find(|case| {
        case.run_seed == seed
            && case.policy == policy
            && case.behavior_inputs_clamped == clamped
            && case.translated == translated
    })
}

fn normalized_physical_trace(
    home: (i32, i32),
    width: i32,
    events: &[EcologyReleaseEvent],
    energy_ledger: &[ProceduralEcologyEnergyLedgerRow],
    final_escrow_energy: f64,
) -> NormalizedPhysicalCaseTrace {
    NormalizedPhysicalCaseTrace {
        events: events
            .iter()
            .map(|event| NormalizedPhysicalReleaseEvent {
                ordinal: event.ordinal,
                before_turn: event.before_turn,
                relative_position: (
                    (event.release_position.0 - home.0).rem_euclid(width),
                    (event.release_position.1 - home.1).rem_euclid(width),
                ),
                consumed_by_end: event.consumed_by_end,
                release_energy: event.release_energy,
                reclaimed_energy: event.reclaimed_energy,
            })
            .collect(),
        energy_ledger: energy_ledger.to_vec(),
        final_escrow_energy,
    }
}

fn normalized_diagnostic_trace(
    home: (i32, i32),
    width: i32,
    events: &[EcologyReleaseEvent],
    energy_ledger: &[ProceduralEcologyEnergyLedgerRow],
    final_escrow_energy: f64,
) -> NormalizedDiagnosticCaseTrace {
    NormalizedDiagnosticCaseTrace {
        events: events
            .iter()
            .map(|event| NormalizedDiagnosticReleaseEvent {
                physical: NormalizedPhysicalReleaseEvent {
                    ordinal: event.ordinal,
                    before_turn: event.before_turn,
                    relative_position: (
                        (event.release_position.0 - home.0).rem_euclid(width),
                        (event.release_position.1 - home.1).rem_euclid(width),
                    ),
                    consumed_by_end: event.consumed_by_end,
                    release_energy: event.release_energy,
                    reclaimed_energy: event.reclaimed_energy,
                },
                raw_consumption_input: event.raw_consumption_input,
                applied_consumption_input: event.applied_consumption_input,
            })
            .collect(),
        energy_ledger: energy_ledger.to_vec(),
        final_escrow_energy,
    }
}

fn engine_ledger_closes(row: EnergyLedgerRow) -> bool {
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
    .all(|residual| residual_closes(residual, row.residual_tolerance))
}

fn combined_ledger_closes(row: &ProceduralEcologyEnergyLedgerRow) -> bool {
    engine_ledger_closes(row.engine)
        && [
            row.organism_residual,
            row.food_residual,
            row.artifact_residual,
            row.ecology_residual,
            row.reclaim_transfer_residual,
            row.release_transfer_residual,
            row.total_residual,
        ]
        .into_iter()
        .all(|residual| residual_closes(residual, row.residual_tolerance))
}

fn residual_closes(residual: f64, tolerance: f64) -> bool {
    residual.is_finite() && tolerance.is_finite() && residual.abs() <= tolerance
}

fn food_exists(sim: &Simulation, id: FoodId) -> bool {
    sim.foods.binary_search_by_key(&id, |food| food.id).is_ok()
}

fn organism_energy(sim: &Simulation) -> f64 {
    sim.organisms
        .iter()
        .map(|organism| f64::from(organism.energy))
        .sum()
}

fn food_energy(sim: &Simulation) -> f64 {
    sim.foods.iter().map(|food| f64::from(food.energy)).sum()
}

fn artifact_energy(sim: &Simulation) -> f64 {
    sim.artifact_caches
        .iter()
        .map(|artifact| f64::from(artifact.energy))
        .sum()
}

fn carrier_position(state: &[EcologyCellState], width: i32) -> Result<(i32, i32)> {
    let mut carriers = state
        .iter()
        .enumerate()
        .filter(|(_, cell)| cell.carrier)
        .map(|(idx, _)| position_from_index(idx, width));
    let carrier = carriers
        .next()
        .ok_or_else(|| anyhow!("procedural ecology state has no carrier"))?;
    if carriers.next().is_some() {
        bail!("procedural ecology state has multiple carriers");
    }
    Ok(carrier)
}

fn cell_index(position: (i32, i32), width: i32) -> usize {
    position.1.rem_euclid(width) as usize * width as usize + position.0.rem_euclid(width) as usize
}

fn position_from_index(index: usize, width: i32) -> (i32, i32) {
    (
        (index % width as usize) as i32,
        (index / width as usize) as i32,
    )
}

fn translate_position(position: (i32, i32), delta: (i32, i32), width: i32) -> (i32, i32) {
    (
        (position.0 + delta.0).rem_euclid(width),
        (position.1 + delta.1).rem_euclid(width),
    )
}

fn fingerprint<T: Serialize>(value: &T) -> Result<String> {
    exact_fingerprint(value).map_err(|error| anyhow!(error.to_string()))
}
