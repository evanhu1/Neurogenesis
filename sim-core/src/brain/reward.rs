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

    // Short/malformed slices fall back to `DEFAULT_REWARD_WEIGHTS` element-wise
    // so legacy genomes deserializing with an empty vector keep baseline behavior
    // even if sanitization hasn't run yet.
    pub(crate) fn weighted_reward_signal(self, weights: &[f32]) -> f32 {
        let w = |i: usize| weights.get(i).copied().unwrap_or(DEFAULT_REWARD_WEIGHTS[i]);
        w(0) * self.energy_level
            + w(1) * self.energy_delta_gain
            + w(2) * self.energy_delta_loss
            + w(3) * self.health_level
            + w(4) * self.health_delta_gain
            + w(5) * self.health_delta_loss
            + w(6) * self.contingent_action_wasted
    }
}

pub(crate) const REWARD_WEIGHT_COUNT: usize = 7;

// Ordered to match the ledger fields:
// [energy_level, energy_delta_gain, energy_delta_loss,
//  health_level, health_delta_gain, health_delta_loss,
//  contingent_action_wasted].
// energy_level starts at 0 (absolute energy is a predictor, not a goal on its
// own) while health_level starts at +1 (higher health is always rewarded).
// contingent_action_wasted fires on failed Forward/Eat/Attack/Reproduce —
// negative default since "did a thing, the thing did nothing" is objectively
// wasted effort.
pub(crate) const DEFAULT_REWARD_WEIGHTS: [f32; REWARD_WEIGHT_COUNT] =
    [0.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0];

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
