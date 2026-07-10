// Multi-tick action state (currently only gestation for Reproduce). Lives
// outside the brain's runtime plasticity: an organism with a pending Reproduce
// is locked out of acting and of weight updates until gestation completes.

use sim_types::{
    OrganismGenome, OrganismId, OrganismState, ReproductionEvent, ReproductionFailureCause,
};

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq, Default)]
pub(crate) enum PendingActionKind {
    #[default]
    None,
    Reproduce,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq, Default)]
pub(crate) struct PendingActionState {
    pub(crate) kind: PendingActionKind,
    pub(crate) turns_remaining: u8,
    pub(crate) reproduction_energy_bits: u32,
}

impl PendingActionState {
    pub(crate) fn reproduction_energy(self) -> f32 {
        f32::from_bits(self.reproduction_energy_bits)
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub(crate) struct PendingReproductionState {
    /// The parent's genome, cloned at conception so later parent death or
    /// movement cannot alter a gestating child's heredity. Reproduction is
    /// clonal — this is the exact genome the child is born with.
    pub(crate) base_genome: OrganismGenome,
    pub(crate) offspring_generation: u64,
    pub(crate) parent_age_turns_at_conception: u64,
}

impl PendingReproductionState {
    pub(crate) fn event(
        &self,
        parent: &OrganismState,
        investment_energy: f32,
        child_id: Option<OrganismId>,
        failure_cause: Option<ReproductionFailureCause>,
    ) -> ReproductionEvent {
        ReproductionEvent {
            parent_id: parent.id,
            parent_species_id: parent.species_id,
            parent_age_turns: self.parent_age_turns_at_conception,
            parent_generation: parent.generation,
            investment_energy,
            parent_energy_after_event: parent.energy,
            child_id,
            failure_cause,
        }
    }
}
