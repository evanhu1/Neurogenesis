// Multi-tick action state (currently only gestation for Reproduce). Lives
// outside the brain's runtime plasticity: an organism with a pending Reproduce
// is locked out of acting and of weight updates until gestation completes.

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
