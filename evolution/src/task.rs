use anyhow::Result;
use serde::Serialize;
use types::{OrganismGenome, SensoryReceptor, Symbol};

#[derive(Debug, Default, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct TaskWorkReport {
    pub brain_synapse_operations: u64,
}

/// One independently reusable genome-evaluation task for the NEAT outer loop.
///
/// Implementations own the task contract and all task-specific state. The
/// evolutionary algorithm sees only absolute fitness plus an optional
/// normalized score for reporting. Rust monomorphizes this boundary, so task
/// modularity does not add dynamic dispatch to the evaluation hot path.
pub trait EvaluationTask: Sync {
    type Config: Clone + Serialize;
    type Evaluation: Clone + Serialize + Send + Sync;

    fn name(&self) -> &'static str;
    fn objective(&self) -> &'static str;
    fn config(&self) -> Self::Config;
    fn validate(&self) -> Result<()>;
    fn evaluate(&self, genome: &OrganismGenome) -> Result<Self::Evaluation>;
    /// Finite, nonnegative scalar used by ranking and selection.
    fn fitness(&self, evaluation: &Self::Evaluation) -> f64;

    fn normalized_fitness(&self, _evaluation: &Self::Evaluation) -> Option<f64> {
        None
    }

    /// Deterministic task work represented by one completed evaluation.
    /// Tasks that do not instrument brain work retain the zero-cost default.
    fn work_report(&self, _evaluation: &Self::Evaluation) -> TaskWorkReport {
        TaskWorkReport::default()
    }

    fn sensor_enabled(&self, _sensor: SensoryReceptor) -> bool {
        true
    }

    /// Whether a task exposes this action output to structural mutation.
    fn action_enabled(&self, _symbol: Symbol) -> bool {
        true
    }

    fn prepare_founder_genome(&self, _genome: &mut OrganismGenome) -> Result<()> {
        Ok(())
    }

    /// Optional evaluation that is reported for the training winner but never
    /// participates in ranking, selection, or breeding.
    fn validation_evaluation(&self, _genome: &OrganismGenome) -> Result<Option<Self::Evaluation>> {
        Ok(None)
    }

    /// Whether the generation winner should receive the task's development
    /// evaluation. The final generation should normally remain enabled so the
    /// run artifact has a development result for its selected winner.
    fn validation_due(&self, _generation: u32, _total_generations: u32) -> bool {
        true
    }

    /// Sealed evaluation called once for the final evolutionary winner.
    fn final_evaluation(&self, _genome: &OrganismGenome) -> Result<Option<Self::Evaluation>> {
        Ok(None)
    }
}
