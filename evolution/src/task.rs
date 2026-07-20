use anyhow::Result;
use serde::Serialize;
use types::{OrganismGenome, SensoryReceptor, Symbol};

#[derive(Debug, Default, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct TaskWorkReport {
    pub brain_synapse_operations: u64,
}

/// Brain-interface policy used by generic structural mutation.
pub trait GenomeTask: Sync {
    fn sensor_enabled(&self, _sensor: SensoryReceptor) -> bool {
        true
    }

    fn action_enabled(&self, _symbol: Symbol) -> bool {
        true
    }

    fn action_feedback_enabled(&self) -> bool {
        self.temporal_credit_enabled()
    }

    fn temporal_credit_enabled(&self) -> bool {
        false
    }

    fn value_prediction_enabled(&self) -> bool {
        self.temporal_credit_enabled()
    }

    /// Whether evaluation changes expressed weights within an organism's
    /// lifetime. This exposes no task state; it only prevents inert tasks from
    /// consuming search variation in plasticity timescale genes.
    fn lifetime_learning_enabled(&self) -> bool {
        self.temporal_credit_enabled()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ResourceLifetimeContext {
    pub generation: u32,
    pub lifetime_ticks: usize,
    pub individual_id: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResourceLifetimeOutcome<E> {
    pub evaluation: E,
    /// Physical resource units consumed during the evaluation. Each unit is
    /// one reproductive ticket; the outer loop never interprets task metrics.
    pub reproductive_tickets: u64,
    pub work: TaskWorkReport,
}

/// A task ecology emits binary, repeatable solve events rather than a scalar
/// fitness. Each genome receives the same task-defined panel of uninterrupted
/// physical lifetimes. Reproduction occurs only after every lifetime has ended,
/// and offspring begin the next generation with freshly expressed runtime state.
pub trait ResourceEcologyTask: GenomeTask {
    type Config: Clone + Serialize;
    type LifetimeState: Send;
    type LifetimeEvaluation: Clone + Serialize + Send + Sync;
    type AuditEvaluation: Clone + Serialize + Send + Sync;

    fn name(&self) -> &'static str;
    fn objective(&self) -> &'static str;
    fn config(&self) -> Self::Config;
    fn lifetime_ticks(&self) -> usize;
    /// Independent physical lifetimes used to estimate one genome's resource
    /// production. Every genome in a generation receives the same panel.
    fn evaluation_lifetimes(&self) -> usize {
        1
    }
    fn validate(&self) -> Result<()>;

    fn initialize_lifetime(
        &self,
        genome: &OrganismGenome,
        individual_id: u64,
        run_seed: u64,
        generation: u32,
    ) -> Result<Self::LifetimeState>;

    fn evaluate_lifetime(
        &self,
        genome: &OrganismGenome,
        state: &mut Self::LifetimeState,
        context: ResourceLifetimeContext,
    ) -> Result<ResourceLifetimeOutcome<Self::LifetimeEvaluation>>;

    fn audit(
        &self,
        genome: &OrganismGenome,
        cohort: &str,
        audit_seed: u64,
    ) -> Result<Self::AuditEvaluation>;

    /// Stable development score used only to retain the best observed
    /// representative for final sealed evaluation. It never affects tickets.
    fn audit_score(&self, audit: &Self::AuditEvaluation) -> f64;

    fn audit_due(&self, generation: u32, total_generations: u32) -> bool {
        generation + 1 == total_generations
    }
}
