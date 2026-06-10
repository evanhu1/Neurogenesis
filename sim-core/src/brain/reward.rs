use sim_types::OrganismState;

// Per-tick tonic reward signals, overwritten each tick by `observe`. Channels
// are unit-normalized so the weighted sum sits in roughly [-2, +2] under
// default weights and feeds directly into tanh without a scaling divisor.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RewardLedger {
    pub(crate) energy_level: f32,
    pub(crate) energy_delta_gain: f32,
    pub(crate) energy_delta_loss: f32,
    pub(crate) health_level: f32,
    pub(crate) health_delta_gain: f32,
    pub(crate) health_delta_loss: f32,
    // Fires on failed Forward/Eat/Attack/Reproduce when
    // `wasted_action_reward_enabled` is on (see
    // `mark_wasted_contingent_actions` in turn/commit.rs).
    pub(crate) contingent_action_wasted: f32,
}

impl RewardLedger {
    pub(crate) fn observe(&mut self, organism: &OrganismState, food_energy: f32) {
        let food_scale = food_energy.max(1e-3);
        let health_scale = organism.max_health.max(1e-3);

        let energy_delta = organism.energy - organism.energy_prev;
        let health_delta = organism.health - organism.health_prev;

        self.energy_level = (organism.energy / (4.0 * food_scale)).clamp(0.0, 1.0) - 0.5;
        self.energy_delta_gain = energy_delta.max(0.0) / food_scale;
        self.energy_delta_loss = (-energy_delta).max(0.0) / food_scale;
        self.health_level = (organism.health / health_scale).clamp(0.0, 1.0) - 0.5;
        self.health_delta_gain = health_delta.max(0.0) / health_scale;
        self.health_delta_loss = (-health_delta).max(0.0) / health_scale;
        self.contingent_action_wasted = if organism.contingent_action_wasted_last_turn {
            1.0
        } else {
            0.0
        };
    }

    // Every live genome carries exactly `REWARD_WEIGHT_COUNT` weights: seed
    // genomes start from `DEFAULT_REWARD_WEIGHTS` and sanitization
    // pads/truncates on every mutation.
    pub(crate) fn weighted_reward_signal(self, weights: &[f32]) -> f32 {
        debug_assert_eq!(weights.len(), REWARD_WEIGHT_COUNT);
        weights[0] * self.energy_level
            + weights[1] * self.energy_delta_gain
            + weights[2] * self.energy_delta_loss
            + weights[3] * self.health_level
            + weights[4] * self.health_delta_gain
            + weights[5] * self.health_delta_loss
            + weights[6] * self.contingent_action_wasted
    }
}

pub(crate) use sim_types::{DEFAULT_REWARD_WEIGHTS, REWARD_WEIGHT_COUNT};

// Bound each weight so evolution can flip signs and scale up to 3x without
// producing arbitrarily large dopamine magnitudes.
pub(crate) const REWARD_WEIGHT_MIN: f32 = -3.0;
pub(crate) const REWARD_WEIGHT_MAX: f32 = 3.0;

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
