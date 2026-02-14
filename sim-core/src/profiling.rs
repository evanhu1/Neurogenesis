use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

#[derive(Clone, Copy, Debug)]
pub(crate) enum TurnPhase {
    Lifecycle,
    Snapshot,
    Intents,
    Reproduction,
    MoveResolution,
    Commit,
    Age,
    Spawn,
    PruneSpecies,
    ConsistencyCheck,
    MetricsAndDelta,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum BrainStage {
    ScanAhead,
    SensoryEncoding,
    InterSetup,
    InterAccumulation,
    InterActivation,
    ActionAccumulation,
    ActionActivationResolve,
    PlasticitySetup,
    PlasticitySensoryTuning,
    PlasticityInterTuning,
    PlasticityPrune,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PhaseCounterSnapshot {
    pub total_ns: u64,
    pub calls: u64,
}

#[derive(Clone, Debug, Default)]
pub struct ProfilingSnapshot {
    pub tick_total: PhaseCounterSnapshot,
    pub lifecycle: PhaseCounterSnapshot,
    pub snapshot: PhaseCounterSnapshot,
    pub intents: PhaseCounterSnapshot,
    pub reproduction: PhaseCounterSnapshot,
    pub move_resolution: PhaseCounterSnapshot,
    pub commit: PhaseCounterSnapshot,
    pub age: PhaseCounterSnapshot,
    pub spawn: PhaseCounterSnapshot,
    pub prune_species: PhaseCounterSnapshot,
    pub consistency_check: PhaseCounterSnapshot,
    pub metrics_and_delta: PhaseCounterSnapshot,
    pub brain_eval_total: PhaseCounterSnapshot,
    pub brain_plasticity_total: PhaseCounterSnapshot,
    pub brain_scan_ahead: PhaseCounterSnapshot,
    pub brain_sensory_encoding: PhaseCounterSnapshot,
    pub brain_inter_setup: PhaseCounterSnapshot,
    pub brain_inter_accumulation: PhaseCounterSnapshot,
    pub brain_inter_activation: PhaseCounterSnapshot,
    pub brain_action_accumulation: PhaseCounterSnapshot,
    pub brain_action_activation_resolve: PhaseCounterSnapshot,
    pub brain_plasticity_setup: PhaseCounterSnapshot,
    pub brain_plasticity_sensory_tuning: PhaseCounterSnapshot,
    pub brain_plasticity_inter_tuning: PhaseCounterSnapshot,
    pub brain_plasticity_prune: PhaseCounterSnapshot,
}

#[derive(Debug)]
struct AtomicPhaseCounter {
    total_ns: AtomicU64,
    calls: AtomicU64,
}

impl AtomicPhaseCounter {
    const fn new() -> Self {
        Self {
            total_ns: AtomicU64::new(0),
            calls: AtomicU64::new(0),
        }
    }

    fn record(&self, elapsed: Duration) {
        let elapsed_ns = duration_to_ns(elapsed);
        self.total_ns.fetch_add(elapsed_ns, Ordering::Relaxed);
        self.calls.fetch_add(1, Ordering::Relaxed);
    }

    fn reset(&self) {
        self.total_ns.store(0, Ordering::Relaxed);
        self.calls.store(0, Ordering::Relaxed);
    }

    fn snapshot(&self) -> PhaseCounterSnapshot {
        PhaseCounterSnapshot {
            total_ns: self.total_ns.load(Ordering::Relaxed),
            calls: self.calls.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug)]
struct ProfilingState {
    tick_total: AtomicPhaseCounter,
    lifecycle: AtomicPhaseCounter,
    snapshot: AtomicPhaseCounter,
    intents: AtomicPhaseCounter,
    reproduction: AtomicPhaseCounter,
    move_resolution: AtomicPhaseCounter,
    commit: AtomicPhaseCounter,
    age: AtomicPhaseCounter,
    spawn: AtomicPhaseCounter,
    prune_species: AtomicPhaseCounter,
    consistency_check: AtomicPhaseCounter,
    metrics_and_delta: AtomicPhaseCounter,
    brain_eval_total: AtomicPhaseCounter,
    brain_plasticity_total: AtomicPhaseCounter,
    brain_scan_ahead: AtomicPhaseCounter,
    brain_sensory_encoding: AtomicPhaseCounter,
    brain_inter_setup: AtomicPhaseCounter,
    brain_inter_accumulation: AtomicPhaseCounter,
    brain_inter_activation: AtomicPhaseCounter,
    brain_action_accumulation: AtomicPhaseCounter,
    brain_action_activation_resolve: AtomicPhaseCounter,
    brain_plasticity_setup: AtomicPhaseCounter,
    brain_plasticity_sensory_tuning: AtomicPhaseCounter,
    brain_plasticity_inter_tuning: AtomicPhaseCounter,
    brain_plasticity_prune: AtomicPhaseCounter,
}

impl ProfilingState {
    const fn new() -> Self {
        Self {
            tick_total: AtomicPhaseCounter::new(),
            lifecycle: AtomicPhaseCounter::new(),
            snapshot: AtomicPhaseCounter::new(),
            intents: AtomicPhaseCounter::new(),
            reproduction: AtomicPhaseCounter::new(),
            move_resolution: AtomicPhaseCounter::new(),
            commit: AtomicPhaseCounter::new(),
            age: AtomicPhaseCounter::new(),
            spawn: AtomicPhaseCounter::new(),
            prune_species: AtomicPhaseCounter::new(),
            consistency_check: AtomicPhaseCounter::new(),
            metrics_and_delta: AtomicPhaseCounter::new(),
            brain_eval_total: AtomicPhaseCounter::new(),
            brain_plasticity_total: AtomicPhaseCounter::new(),
            brain_scan_ahead: AtomicPhaseCounter::new(),
            brain_sensory_encoding: AtomicPhaseCounter::new(),
            brain_inter_setup: AtomicPhaseCounter::new(),
            brain_inter_accumulation: AtomicPhaseCounter::new(),
            brain_inter_activation: AtomicPhaseCounter::new(),
            brain_action_accumulation: AtomicPhaseCounter::new(),
            brain_action_activation_resolve: AtomicPhaseCounter::new(),
            brain_plasticity_setup: AtomicPhaseCounter::new(),
            brain_plasticity_sensory_tuning: AtomicPhaseCounter::new(),
            brain_plasticity_inter_tuning: AtomicPhaseCounter::new(),
            brain_plasticity_prune: AtomicPhaseCounter::new(),
        }
    }

    fn reset(&self) {
        self.tick_total.reset();
        self.lifecycle.reset();
        self.snapshot.reset();
        self.intents.reset();
        self.reproduction.reset();
        self.move_resolution.reset();
        self.commit.reset();
        self.age.reset();
        self.spawn.reset();
        self.prune_species.reset();
        self.consistency_check.reset();
        self.metrics_and_delta.reset();
        self.brain_eval_total.reset();
        self.brain_plasticity_total.reset();
        self.brain_scan_ahead.reset();
        self.brain_sensory_encoding.reset();
        self.brain_inter_setup.reset();
        self.brain_inter_accumulation.reset();
        self.brain_inter_activation.reset();
        self.brain_action_accumulation.reset();
        self.brain_action_activation_resolve.reset();
        self.brain_plasticity_setup.reset();
        self.brain_plasticity_sensory_tuning.reset();
        self.brain_plasticity_inter_tuning.reset();
        self.brain_plasticity_prune.reset();
    }

    fn snapshot(&self) -> ProfilingSnapshot {
        ProfilingSnapshot {
            tick_total: self.tick_total.snapshot(),
            lifecycle: self.lifecycle.snapshot(),
            snapshot: self.snapshot.snapshot(),
            intents: self.intents.snapshot(),
            reproduction: self.reproduction.snapshot(),
            move_resolution: self.move_resolution.snapshot(),
            commit: self.commit.snapshot(),
            age: self.age.snapshot(),
            spawn: self.spawn.snapshot(),
            prune_species: self.prune_species.snapshot(),
            consistency_check: self.consistency_check.snapshot(),
            metrics_and_delta: self.metrics_and_delta.snapshot(),
            brain_eval_total: self.brain_eval_total.snapshot(),
            brain_plasticity_total: self.brain_plasticity_total.snapshot(),
            brain_scan_ahead: self.brain_scan_ahead.snapshot(),
            brain_sensory_encoding: self.brain_sensory_encoding.snapshot(),
            brain_inter_setup: self.brain_inter_setup.snapshot(),
            brain_inter_accumulation: self.brain_inter_accumulation.snapshot(),
            brain_inter_activation: self.brain_inter_activation.snapshot(),
            brain_action_accumulation: self.brain_action_accumulation.snapshot(),
            brain_action_activation_resolve: self.brain_action_activation_resolve.snapshot(),
            brain_plasticity_setup: self.brain_plasticity_setup.snapshot(),
            brain_plasticity_sensory_tuning: self.brain_plasticity_sensory_tuning.snapshot(),
            brain_plasticity_inter_tuning: self.brain_plasticity_inter_tuning.snapshot(),
            brain_plasticity_prune: self.brain_plasticity_prune.snapshot(),
        }
    }
}

static PROFILING_STATE: ProfilingState = ProfilingState::new();

fn duration_to_ns(elapsed: Duration) -> u64 {
    elapsed.as_nanos().min(u128::from(u64::MAX)) as u64
}

pub(crate) fn record_tick_total(elapsed: Duration) {
    PROFILING_STATE.tick_total.record(elapsed);
}

pub(crate) fn record_turn_phase(phase: TurnPhase, elapsed: Duration) {
    match phase {
        TurnPhase::Lifecycle => PROFILING_STATE.lifecycle.record(elapsed),
        TurnPhase::Snapshot => PROFILING_STATE.snapshot.record(elapsed),
        TurnPhase::Intents => PROFILING_STATE.intents.record(elapsed),
        TurnPhase::Reproduction => PROFILING_STATE.reproduction.record(elapsed),
        TurnPhase::MoveResolution => PROFILING_STATE.move_resolution.record(elapsed),
        TurnPhase::Commit => PROFILING_STATE.commit.record(elapsed),
        TurnPhase::Age => PROFILING_STATE.age.record(elapsed),
        TurnPhase::Spawn => PROFILING_STATE.spawn.record(elapsed),
        TurnPhase::PruneSpecies => PROFILING_STATE.prune_species.record(elapsed),
        TurnPhase::ConsistencyCheck => PROFILING_STATE.consistency_check.record(elapsed),
        TurnPhase::MetricsAndDelta => PROFILING_STATE.metrics_and_delta.record(elapsed),
    }
}

pub(crate) fn record_brain_eval_total(elapsed: Duration) {
    PROFILING_STATE.brain_eval_total.record(elapsed);
}

pub(crate) fn record_brain_plasticity_total(elapsed: Duration) {
    PROFILING_STATE.brain_plasticity_total.record(elapsed);
}

pub(crate) fn record_brain_stage(stage: BrainStage, elapsed: Duration) {
    match stage {
        BrainStage::ScanAhead => PROFILING_STATE.brain_scan_ahead.record(elapsed),
        BrainStage::SensoryEncoding => PROFILING_STATE.brain_sensory_encoding.record(elapsed),
        BrainStage::InterSetup => PROFILING_STATE.brain_inter_setup.record(elapsed),
        BrainStage::InterAccumulation => PROFILING_STATE.brain_inter_accumulation.record(elapsed),
        BrainStage::InterActivation => PROFILING_STATE.brain_inter_activation.record(elapsed),
        BrainStage::ActionAccumulation => PROFILING_STATE.brain_action_accumulation.record(elapsed),
        BrainStage::ActionActivationResolve => PROFILING_STATE
            .brain_action_activation_resolve
            .record(elapsed),
        BrainStage::PlasticitySetup => PROFILING_STATE.brain_plasticity_setup.record(elapsed),
        BrainStage::PlasticitySensoryTuning => PROFILING_STATE
            .brain_plasticity_sensory_tuning
            .record(elapsed),
        BrainStage::PlasticityInterTuning => PROFILING_STATE
            .brain_plasticity_inter_tuning
            .record(elapsed),
        BrainStage::PlasticityPrune => PROFILING_STATE.brain_plasticity_prune.record(elapsed),
    }
}

pub fn reset() {
    PROFILING_STATE.reset();
}

pub fn snapshot() -> ProfilingSnapshot {
    PROFILING_STATE.snapshot()
}
